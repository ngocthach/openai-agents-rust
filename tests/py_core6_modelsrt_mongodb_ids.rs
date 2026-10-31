//! Port of `test_string_session_ids_remain_literal_and_isolated`
//! (`tests/extensions/memory/test_mongodb_session.py`): session ids that look like query
//! operators or JSON stay literal values and never reach another session's history. Runs
//! against the in-process OP_MSG server copied from `py_memory_mongodb.rs`.
#![cfg(feature = "mongodb")]
#![allow(unused_imports)]
#![allow(dead_code)]

use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

use mongodb::Client;
use mongodb::bson::{Bson, Document, doc, oid::ObjectId};
use openai_agents_sdk::memory::{MongoDBSession, Session, SessionSettings};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

// ---------------------------------------------------------------------------
// Mini MongoDB
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Db {
    /// Collection name -> documents, in insertion order.
    collections: std::collections::HashMap<String, Vec<Document>>,
    index_requests: usize,
    commands: Vec<String>,
}

fn num(value: &Bson) -> Option<f64> {
    match value {
        Bson::Int32(n) => Some(f64::from(*n)),
        Bson::Int64(n) => Some(*n as f64),
        Bson::Double(n) => Some(*n),
        _ => None,
    }
}

fn bson_eq(a: &Bson, b: &Bson) -> bool {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

fn compare(a: &Bson, b: &Bson) -> Ordering {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        _ => Ordering::Equal,
    }
}

fn field_matches(actual: Option<&Bson>, expected: &Bson) -> bool {
    if let Bson::Document(ops) = expected
        && ops.keys().next().is_some_and(|k| k.starts_with('$'))
    {
        return ops.iter().all(|(op, arg)| match op.as_str() {
            "$in" => arg.as_array().is_some_and(|options| {
                options.iter().any(|o| match actual {
                    Some(a) => bson_eq(a, o),
                    None => matches!(o, Bson::Null),
                })
            }),
            "$ne" => actual.is_none_or(|a| !bson_eq(a, arg)),
            "$lt" => actual.is_some_and(|a| compare(a, arg) == Ordering::Less),
            "$exists" => actual.is_some() == arg.as_bool().unwrap_or(true),
            other => panic!("mock mongo: unsupported operator {other}"),
        });
    }
    match actual {
        Some(a) => bson_eq(a, expected),
        None => matches!(expected, Bson::Null),
    }
}

fn matches(doc: &Document, filter: &Document) -> bool {
    filter.iter().all(|(key, expected)| {
        if key == "$or" {
            expected
                .as_array()
                .is_some_and(|alts| alts.iter().any(|a| matches(doc, a.as_document().unwrap())))
        } else {
            field_matches(doc.get(key), expected)
        }
    })
}

fn sort_docs(docs: &mut [Document], sort: &Document) {
    docs.sort_by(|a, b| {
        for (key, direction) in sort {
            let (x, y) = (a.get(key), b.get(key));
            let order = match (x, y) {
                (Some(x), Some(y)) => compare(x, y),
                _ => Ordering::Equal,
            };
            let order = if num(direction) == Some(-1.0) {
                order.reverse()
            } else {
                order
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    });
}

fn apply_update(doc: &mut Document, update: &Bson, inserting: bool) {
    match update {
        Bson::Document(ops) => {
            for (op, fields) in ops {
                let fields = fields.as_document().unwrap();
                match op.as_str() {
                    "$set" => doc.extend(fields.clone()),
                    "$setOnInsert" if inserting => doc.extend(fields.clone()),
                    "$setOnInsert" => {}
                    "$inc" => {
                        for (k, v) in fields {
                            let add = num(v).unwrap() as i64;
                            let current = doc.get(k).and_then(num).unwrap_or(0.0) as i64;
                            doc.insert(k.clone(), Bson::Int64(current + add));
                        }
                    }
                    other => panic!("mock mongo: unsupported update operator {other}"),
                }
            }
        }
        // The only pipeline in use: drop the last entry of the `message_data` batch array.
        Bson::Array(_) => {
            let trimmed = match doc.get("message_data") {
                Some(Bson::Array(entries)) => {
                    Bson::Array(entries[..entries.len().saturating_sub(1)].to_vec())
                }
                _ => Bson::Array(vec![]),
            };
            doc.insert("message_data", trimmed);
        }
        other => panic!("mock mongo: unsupported update {other:?}"),
    }
}

#[derive(Clone)]
struct MockMongo {
    db: Arc<Mutex<Db>>,
    uri: String,
}

impl MockMongo {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mock = MockMongo {
            db: Arc::new(Mutex::new(Db::default())),
            uri: format!(
                "mongodb://127.0.0.1:{port}/?directConnection=true&serverSelectionTimeoutMS=2000\
                 &connectTimeoutMS=2000&heartbeatFrequencyMS=500"
            ),
        };
        let shared = mock.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mock = shared.clone();
                tokio::spawn(async move {
                    loop {
                        let mut header = [0u8; 16];
                        if sock.read_exact(&mut header).await.is_err() {
                            return;
                        }
                        let len = i32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                        let request_id = i32::from_le_bytes(header[4..8].try_into().unwrap());
                        let mut body = vec![0u8; len - 16];
                        if sock.read_exact(&mut body).await.is_err() {
                            return;
                        }
                        let command = parse_op_msg(&body);
                        let reply = mock.run(command);
                        let mut payload = Vec::new();
                        payload.extend_from_slice(&0u32.to_le_bytes());
                        payload.push(0);
                        reply.to_writer(&mut payload).unwrap();
                        let mut out = Vec::new();
                        out.extend_from_slice(&((payload.len() + 16) as i32).to_le_bytes());
                        out.extend_from_slice(&1i32.to_le_bytes());
                        out.extend_from_slice(&request_id.to_le_bytes());
                        out.extend_from_slice(&2013i32.to_le_bytes());
                        out.extend_from_slice(&payload);
                        if sock.write_all(&out).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        mock
    }

    async fn client(&self) -> Client {
        Client::with_uri_str(&self.uri).await.unwrap()
    }

    async fn session(&self, id: &str) -> MongoDBSession {
        MongoDBSession::with_client(id, &self.client().await, "testdb")
    }

    fn docs(&self, collection: &str) -> Vec<Document> {
        self.db
            .lock()
            .unwrap()
            .collections
            .get(collection)
            .cloned()
            .unwrap_or_default()
    }

    fn insert_raw(&self, collection: &str, doc: Document) {
        self.db
            .lock()
            .unwrap()
            .collections
            .entry(collection.into())
            .or_default()
            .push(doc);
    }

    fn run(&self, command: Document) -> Document {
        let name = command.keys().next().cloned().unwrap_or_default();
        let mut db = self.db.lock().unwrap();
        db.commands.push(name.clone());
        match name.as_str() {
            "hello" | "isMaster" | "ismaster" => doc! {
                "ismaster": true,
                "isWritablePrimary": true,
                "helloOk": true,
                "maxBsonObjectSize": 16777216,
                "maxMessageSizeBytes": 48000000,
                "maxWriteBatchSize": 100000,
                "minWireVersion": 0,
                "maxWireVersion": 17,
                "ok": 1.0,
            },
            "createIndexes" => {
                db.index_requests += 1;
                doc! {"ok": 1.0, "numIndexesBefore": 1, "numIndexesAfter": 2}
            }
            "find" => {
                let collection = command.get_str("find").unwrap().to_string();
                let filter = command.get_document("filter").cloned().unwrap_or_default();
                let mut found: Vec<Document> = db
                    .collections
                    .get(&collection)
                    .map(|docs| {
                        docs.iter()
                            .filter(|d| matches(d, &filter))
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                if let Ok(sort) = command.get_document("sort") {
                    sort_docs(&mut found, sort);
                }
                let limit = command.get("limit").and_then(num).unwrap_or(0.0) as usize;
                if limit > 0 {
                    found.truncate(limit);
                }
                doc! {
                    "cursor": {
                        "id": 0i64,
                        "ns": format!("testdb.{collection}"),
                        "firstBatch": found.into_iter().map(Bson::Document).collect::<Vec<_>>(),
                    },
                    "ok": 1.0,
                }
            }
            "insert" => {
                let collection = command.get_str("insert").unwrap().to_string();
                let documents = command.get_array("documents").unwrap().clone();
                let list = db.collections.entry(collection).or_default();
                for mut document in documents
                    .into_iter()
                    .map(|d| d.as_document().unwrap().clone())
                {
                    if !document.contains_key("_id") {
                        document.insert("_id", ObjectId::new());
                    }
                    list.push(document);
                }
                doc! {"n": 1, "ok": 1.0}
            }
            "findAndModify" => {
                let collection = command.get_str("findAndModify").unwrap().to_string();
                let query = command.get_document("query").cloned().unwrap_or_default();
                let update = command.get("update").cloned().unwrap();
                let upsert = command.get_bool("upsert").unwrap_or(false);
                let want_new = command.get_bool("new").unwrap_or(false);
                let list = db.collections.entry(collection).or_default();
                let mut candidates: Vec<usize> = (0..list.len())
                    .filter(|i| matches(&list[*i], &query))
                    .collect();
                if let Ok(sort) = command.get_document("sort") {
                    let mut ordered: Vec<Document> =
                        candidates.iter().map(|i| list[*i].clone()).collect();
                    sort_docs(&mut ordered, sort);
                    if let Some(first) = ordered.first() {
                        candidates = vec![list.iter().position(|d| d == first).unwrap()];
                    }
                }
                if let Some(&index) = candidates.first() {
                    let before = list[index].clone();
                    apply_update(&mut list[index], &update, false);
                    let value = if want_new {
                        list[index].clone()
                    } else {
                        before
                    };
                    doc! {
                        "value": value,
                        "lastErrorObject": {"n": 1, "updatedExisting": true},
                        "ok": 1.0,
                    }
                } else if upsert {
                    let mut created = Document::new();
                    for (k, v) in &query {
                        if !k.starts_with('$') && !matches!(v, Bson::Document(_)) {
                            created.insert(k.clone(), v.clone());
                        }
                    }
                    created.insert("_id", ObjectId::new());
                    apply_update(&mut created, &update, true);
                    list.push(created.clone());
                    doc! {
                        "value": if want_new { Bson::Document(created) } else { Bson::Null },
                        "lastErrorObject": {"n": 1, "updatedExisting": false},
                        "ok": 1.0,
                    }
                } else {
                    doc! {"value": Bson::Null, "lastErrorObject": {"n": 0, "updatedExisting": false}, "ok": 1.0}
                }
            }
            "delete" => {
                let collection = command.get_str("delete").unwrap().to_string();
                let deletes = command.get_array("deletes").unwrap().clone();
                let list = db.collections.entry(collection).or_default();
                let mut removed = 0;
                for spec in deletes {
                    let spec = spec.as_document().unwrap();
                    let query = spec.get_document("q").unwrap();
                    let single = spec.get("limit").and_then(num) == Some(1.0);
                    let mut i = 0;
                    while i < list.len() {
                        if matches(&list[i], query) {
                            list.remove(i);
                            removed += 1;
                            if single {
                                break;
                            }
                        } else {
                            i += 1;
                        }
                    }
                }
                doc! {"n": removed, "ok": 1.0}
            }
            _ => doc! {"ok": 1.0},
        }
    }
}

/// Decode an OP_MSG body into the command document, folding in kind-1 document sequences.
fn parse_op_msg(body: &[u8]) -> Document {
    let mut pos = 4; // flag bits
    let mut command = Document::new();
    while pos < body.len() {
        let kind = body[pos];
        pos += 1;
        if kind == 0 {
            let size = i32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
            command = Document::from_reader(&mut &body[pos..pos + size]).unwrap();
            pos += size;
        } else {
            let size = i32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
            let section = &body[pos + 4..pos + size];
            let end = section.iter().position(|b| *b == 0).unwrap();
            let identifier = String::from_utf8_lossy(&section[..end]).into_owned();
            let mut rest = &section[end + 1..];
            let mut documents = Vec::new();
            while !rest.is_empty() {
                let size = i32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
                documents.push(Bson::Document(
                    Document::from_reader(&mut &rest[..size]).unwrap(),
                ));
                rest = &rest[size..];
            }
            command.insert(identifier, documents);
            pos += size;
        }
    }
    command
}

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

#[tokio::test]
async fn string_session_ids_remain_literal_and_isolated() {
    for id in ["", "$ne", "{\"$ne\": null}", "会話-123"] {
        let mock = MockMongo::start().await;
        let session = mock.session(id).await;
        let other = mock.session("other-session").await;
        other.add_items(vec![user("other history")]).await.unwrap();
        session.add_items(vec![user("own history")]).await.unwrap();

        assert_eq!(session.session_id(), id);
        assert_eq!(
            texts(&session.get_items(None).await.unwrap()),
            vec!["own history"],
            "{id:?}"
        );
        assert_eq!(
            item_text(&session.pop_item().await.unwrap().unwrap()),
            "own history",
            "{id:?}"
        );
        assert!(session.get_items(None).await.unwrap().is_empty(), "{id:?}");
        session.add_items(vec![user("own history")]).await.unwrap();
        session.clear_session().await.unwrap();
        assert!(session.get_items(None).await.unwrap().is_empty(), "{id:?}");
        assert_eq!(
            texts(&other.get_items(None).await.unwrap()),
            vec!["other history"],
            "{id:?}"
        );
    }
}
