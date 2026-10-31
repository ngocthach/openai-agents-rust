//! Parity port of `tests/extensions/memory/test_mongodb_session.py`.
//!
//! Python injects fake `pymongo` modules; Rust's `MongoDBSession` takes a real
//! `mongodb::Client`, so these tests run an in-process OP_MSG server that answers the
//! handful of commands the session issues (`hello`/`isMaster`, `ping`, `createIndexes`,
//! `find`, `insert`, `findAndModify`, `delete`) over a small document store with a mini query
//! engine (equality, `$in`, `$ne`, `$lt`, `$exists`, `$or`; `$set`/`$setOnInsert`/`$inc`
//! updates and the one `$slice` pipeline `pop_item` uses). The hello reply omits
//! `logicalSessionTimeoutMinutes`, so the driver sends no implicit sessions.
#![cfg(feature = "mongodb")]

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_and_get_items() {
    let mock = MockMongo::start().await;
    let session = mock.session("s1").await;
    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello", "Hi there!"]
    );
}

#[tokio::test]
async fn an_empty_add_is_a_noop_and_an_empty_session_reads_empty() {
    let mock = MockMongo::start().await;
    let session = mock.session("empty").await;
    session.add_items(vec![]).await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(mock.docs("agent_messages").is_empty());
    assert!(mock.docs("agent_sessions").is_empty());
}

#[tokio::test]
async fn pop_item_returns_the_last_item() {
    let mock = MockMongo::start().await;
    let session = mock.session("pop").await;
    session
        .add_items(vec![user("first"), assistant("second")])
        .await
        .unwrap();

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "second"
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["first"]
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "first"
    );
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn pop_deletes_the_exhausted_batch_document() {
    let mock = MockMongo::start().await;
    let session = mock.session("pop_batch").await;
    session.add_items(vec![user("only")]).await.unwrap();
    assert_eq!(mock.docs("agent_messages").len(), 1);

    session.pop_item().await.unwrap();
    assert!(
        mock.docs("agent_messages").is_empty(),
        "an exhausted batch must not linger"
    );
}

#[tokio::test]
async fn pop_claims_distinct_items_from_one_batch() {
    let mock = MockMongo::start().await;
    let session = Arc::new(mock.session("claims").await);
    session
        .add_items(vec![user("a"), user("b"), user("c"), user("d")])
        .await
        .unwrap();

    let mut popped = Vec::new();
    for _ in 0..4 {
        popped.push(item_text(&session.pop_item().await.unwrap().unwrap()));
    }
    assert_eq!(popped, vec!["d", "c", "b", "a"]);
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn pop_selects_the_newest_batch_first() {
    let mock = MockMongo::start().await;
    let session = mock.session("newest").await;
    session
        .add_items(vec![user("old 1"), user("old 2")])
        .await
        .unwrap();
    session
        .add_items(vec![user("new 1"), user("new 2")])
        .await
        .unwrap();

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "new 2"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "new 1"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "old 2"
    );
}

#[tokio::test]
async fn clear_session_empties_only_this_session() {
    let mock = MockMongo::start().await;
    let one = mock.session("one").await;
    let two = mock.session("two").await;
    one.add_items(vec![user("one")]).await.unwrap();
    two.add_items(vec![user("two")]).await.unwrap();

    one.clear_session().await.unwrap();

    assert!(one.get_items(None).await.unwrap().is_empty());
    assert_eq!(texts(&two.get_items(None).await.unwrap()), vec!["two"]);
    // New history after a clear starts a fresh generation.
    one.add_items(vec![user("again")]).await.unwrap();
    assert_eq!(texts(&one.get_items(None).await.unwrap()), vec!["again"]);
}

#[tokio::test]
async fn a_clear_hides_history_even_when_cleanup_leaves_documents_behind() {
    // Python's partial-cleanup test: the generation bump is the clear boundary, so documents
    // that survive best-effort cleanup stay invisible.
    let mock = MockMongo::start().await;
    let session = mock.session("generation").await;
    session.add_items(vec![user("old")]).await.unwrap();
    let stale = mock.docs("agent_messages");

    session.clear_session().await.unwrap();
    for document in stale {
        mock.insert_raw("agent_messages", document);
    }

    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn multiple_adds_accumulate_in_order() {
    let mock = MockMongo::start().await;
    let session = mock.session("accumulate").await;
    session.add_items(vec![user("1")]).await.unwrap();
    session.add_items(vec![user("2"), user("3")]).await.unwrap();
    session.add_items(vec![user("4")]).await.unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["1", "2", "3", "4"]
    );
    assert_eq!(
        mock.docs("agent_messages").len(),
        3,
        "one document per add_items call"
    );
}

#[tokio::test]
async fn session_metadata_timestamps_are_written() {
    let mock = MockMongo::start().await;
    let session = mock.session("meta").await;
    session.add_items(vec![user("hello")]).await.unwrap();

    let sessions = mock.docs("agent_sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].get_str("session_id").unwrap(), "meta");
    assert!(matches!(
        sessions[0].get("created_at"),
        Some(Bson::DateTime(_))
    ));
    assert!(matches!(
        sessions[0].get("updated_at"),
        Some(Bson::DateTime(_))
    ));
}

#[tokio::test]
async fn limits_apply_explicitly_by_default_and_as_zero() {
    let mock = MockMongo::start().await;
    let session = mock.session("limits").await;
    session
        .add_items((0..4).map(|i| user(&format!("{i}"))).collect())
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["2", "3"]
    );
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
    assert_eq!(session.get_items(Some(100)).await.unwrap().len(), 4);

    let limited = mock
        .session("limits")
        .await
        .with_settings(SessionSettings::with_limit(3));
    assert_eq!(
        texts(&limited.get_items(None).await.unwrap()),
        vec!["1", "2", "3"]
    );
    assert_eq!(texts(&limited.get_items(Some(1)).await.unwrap()), vec!["3"]);
}

#[tokio::test]
async fn sessions_with_similar_ids_stay_isolated() {
    let mock = MockMongo::start().await;
    let a = mock.session("conversation").await;
    let b = mock.session("conversation.*").await;
    a.add_items(vec![user("a")]).await.unwrap();
    b.add_items(vec![user("b")]).await.unwrap();
    assert_eq!(texts(&a.get_items(None).await.unwrap()), vec!["a"]);
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["b"]);
}

#[tokio::test]
async fn unicode_and_json_special_characters_round_trip() {
    let mock = MockMongo::start().await;
    let session = mock.session("text").await;
    let samples = [
        "こんにちは",
        "😊👍",
        "{\"a\": [1, 2]}",
        "line\nbreak \"quoted\" \\ back",
        "O'Reilly",
    ];
    session
        .add_items(samples.iter().map(|s| user(s)).collect())
        .await
        .unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), samples);
}

#[tokio::test]
async fn corrupted_and_malformed_documents_are_skipped() {
    let mock = MockMongo::start().await;
    let session = mock.session("corrupt").await;
    session.add_items(vec![user("valid")]).await.unwrap();
    let generation_doc = mock.docs("agent_sessions");
    assert_eq!(generation_doc.len(), 1);
    for (seq, message_data) in [
        (
            10i64,
            Bson::Array(vec![Bson::String("not valid json {{{".into())]),
        ),
        (11, Bson::Int32(5)),
        (12, Bson::Array(vec![Bson::Int32(1)])),
    ] {
        mock.insert_raw(
            "agent_messages",
            doc! {"session_id": "corrupt", "seq": seq, "generation": 0i64, "message_data": message_data},
        );
    }
    mock.insert_raw(
        "agent_messages",
        doc! {"session_id": "corrupt", "seq": 13i64, "generation": 0i64},
    );

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["valid"]
    );
}

#[tokio::test]
async fn limit_counts_valid_items_past_corrupt_newest_documents() {
    let mock = MockMongo::start().await;
    let session = mock.session("limit_corrupt").await;
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    mock.insert_raw(
        "agent_messages",
        doc! {"session_id": "limit_corrupt", "seq": 50i64, "generation": 0i64, "message_data": ["not valid json {{{"]},
    );

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
    assert_eq!(
        texts(&session.get_items(Some(10)).await.unwrap()),
        vec!["valid 0", "valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn pop_skips_corrupt_documents_and_reports_none_when_only_corrupt_remain() {
    let mock = MockMongo::start().await;
    let session = mock.session("pop_corrupt").await;
    session.add_items(vec![user("valid")]).await.unwrap();
    mock.insert_raw(
        "agent_messages",
        doc! {"session_id": "pop_corrupt", "seq": 50i64, "generation": 0i64, "message_data": ["not valid json {{{"]},
    );

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "valid"
    );
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn legacy_single_item_documents_read_alongside_batches() {
    let mock = MockMongo::start().await;
    let session = mock.session("legacy").await;
    mock.insert_raw(
        "agent_messages",
        doc! {"session_id": "legacy", "seq": 0i64, "message_data": json!({"role": "user", "content": "legacy"}).to_string()},
    );
    session.add_items(vec![assistant("batched")]).await.unwrap();

    let items = session.get_items(None).await.unwrap();
    assert_eq!(texts(&items), vec!["legacy", "batched"]);
}

#[tokio::test]
async fn indexes_are_created_once_per_session_and_again_for_a_new_client() {
    let mock = MockMongo::start().await;
    let session = mock.session("indexes").await;
    session.add_items(vec![user("a")]).await.unwrap();
    session.add_items(vec![user("b")]).await.unwrap();
    session.get_items(None).await.unwrap();
    let after_first = mock.db.lock().unwrap().index_requests;
    assert_eq!(after_first, 2, "one index per collection, created once");

    let other = mock.session("indexes").await;
    other.get_items(None).await.unwrap();
    assert_eq!(mock.db.lock().unwrap().index_requests, after_first * 2);
}

#[tokio::test]
async fn ping_succeeds_against_a_live_server_and_fails_without_one() {
    let mock = MockMongo::start().await;
    assert!(mock.session("ping").await.ping().await);

    let dead = Client::with_uri_str(
        "mongodb://127.0.0.1:1/?directConnection=true&serverSelectionTimeoutMS=300&connectTimeoutMS=300",
    )
    .await
    .unwrap();
    assert!(
        !MongoDBSession::with_client("ping", &dead, "testdb")
            .ping()
            .await
    );
}

#[tokio::test]
async fn custom_collection_names_are_used() {
    let mock = MockMongo::start().await;
    let session = mock
        .session("custom")
        .await
        .with_collections("my_sessions", "my_messages");
    session.add_items(vec![user("hi")]).await.unwrap();
    assert_eq!(mock.docs("my_messages").len(), 1);
    assert_eq!(mock.docs("my_sessions").len(), 1);
    assert!(mock.docs("agent_messages").is_empty());
}

#[tokio::test]
async fn the_runner_stores_turns_isolates_sessions_and_honours_run_config_limits() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mock = MockMongo::start().await;
    let session = Arc::new(mock.session("runner").await);
    let other = Arc::new(mock.session("runner_other").await);
    session
        .add_items((0..10).map(|i| user(&format!("Turn {i}"))).collect())
        .await
        .unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Got it")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "New question",
        RunOptions::new(()).session(session.clone()).run_config(
            RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2)),
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Got it"));

    let seen = texts(&model.last_call().unwrap().input);
    assert_eq!(
        seen.iter().filter(|c| *c != "New question").count(),
        2,
        "{seen:?}"
    );
    let stored = texts(&session.get_items(None).await.unwrap());
    assert!(stored.contains(&"Got it".to_string()));
    assert!(other.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn an_external_client_stays_usable_across_sessions() {
    let mock = MockMongo::start().await;
    let client = mock.client().await;
    let first = MongoDBSession::with_client("shared", &client, "testdb");
    first.add_items(vec![user("one")]).await.unwrap();
    drop(first);
    let second = MongoDBSession::with_client("shared", &client, "testdb");
    assert_eq!(texts(&second.get_items(None).await.unwrap()), vec!["one"]);
}
