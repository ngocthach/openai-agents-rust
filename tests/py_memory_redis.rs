//! Parity port of `tests/extensions/memory/test_redis_session.py`.
//!
//! Python runs against `fakeredis`; Rust's `RedisSession` takes a real multiplexed connection,
//! so these tests talk RESP2 over TCP to an in-process mock that implements just the commands
//! the session issues: `PING`, `LRANGE`, `RPOP`, `DEL` and the three Lua scripts (`EVAL`),
//! whose effects the mock replicates keyed on the script text. `EVALSHA` is answered with
//! `-NOSCRIPT` so the client falls back to `EVAL`; anything else gets `+OK`.
#![cfg(feature = "redis")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{RedisSession, Session, SessionSettings, SnapshotHandle};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

// ---------------------------------------------------------------------------
// In-process Redis
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Store {
    lists: HashMap<String, Vec<String>>,
    hashes: HashMap<String, HashMap<String, String>>,
    strings: HashMap<String, String>,
    ttl_ms: HashMap<String, i64>,
}

impl Store {
    fn type_of(&self, key: &str) -> &'static str {
        if self.lists.contains_key(key) {
            "list"
        } else if self.hashes.contains_key(key) {
            "hash"
        } else if self.strings.contains_key(key) {
            "string"
        } else {
            "none"
        }
    }

    fn exists(&self, key: &str) -> bool {
        self.type_of(key) != "none"
    }

    fn del(&mut self, key: &str) {
        self.lists.remove(key);
        self.hashes.remove(key);
        self.strings.remove(key);
        self.ttl_ms.remove(key);
    }

    /// The Lua `guard()` of the session's scripts.
    fn guard(&self, keys: &[String]) -> Option<String> {
        for (key, expected) in keys.iter().zip(["list", "hash", "string"]) {
            let actual = self.type_of(key);
            if actual != "none" && actual != expected {
                return Some(format!(
                    "WRONGTYPE Redis key {key} holds a {actual}, expected a {expected}"
                ));
            }
        }
        None
    }
}

#[derive(Clone)]
struct MockRedis {
    store: Arc<Mutex<Store>>,
    clock: Arc<AtomicI64>,
    alive: Arc<AtomicBool>,
    url: String,
}

enum Reply {
    Ok,
    Int(i64),
    Nil,
    Bulk(String),
    Array(Vec<String>),
    Error(String),
}

impl Reply {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Reply::Ok => out.extend_from_slice(b"+OK\r\n"),
            Reply::Int(n) => out.extend_from_slice(format!(":{n}\r\n").as_bytes()),
            Reply::Nil => out.extend_from_slice(b"$-1\r\n"),
            Reply::Bulk(s) => out.extend_from_slice(format!("${}\r\n{s}\r\n", s.len()).as_bytes()),
            Reply::Array(items) => {
                out.extend_from_slice(format!("*{}\r\n", items.len()).as_bytes());
                for item in items {
                    out.extend_from_slice(format!("${}\r\n{item}\r\n", item.len()).as_bytes());
                }
            }
            Reply::Error(e) => out.extend_from_slice(format!("-{e}\r\n").as_bytes()),
        }
    }
}

/// Parse one RESP array of bulk strings from the front of `buf`.
fn parse_command(buf: &[u8]) -> Option<(Vec<String>, usize)> {
    fn line(buf: &[u8], from: usize) -> Option<(&str, usize)> {
        let end = buf[from..].windows(2).position(|w| w == b"\r\n")? + from;
        Some((std::str::from_utf8(&buf[from..end]).ok()?, end + 2))
    }
    let (head, mut pos) = line(buf, 0)?;
    let count: usize = head.strip_prefix('*')?.parse().ok()?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        let (len_line, next) = line(buf, pos)?;
        let len: usize = len_line.strip_prefix('$')?.parse().ok()?;
        if buf.len() < next + len + 2 {
            return None;
        }
        args.push(String::from_utf8_lossy(&buf[next..next + len]).into_owned());
        pos = next + len + 2;
    }
    Some((args, pos))
}

impl MockRedis {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mock = MockRedis {
            store: Arc::new(Mutex::new(Store::default())),
            clock: Arc::new(AtomicI64::new(1_000)),
            alive: Arc::new(AtomicBool::new(true)),
            url: format!("redis://127.0.0.1:{port}/"),
        };
        let shared = mock.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mock = shared.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 8192];
                    loop {
                        while let Some((args, used)) = parse_command(&buf) {
                            buf.drain(..used);
                            if !mock.alive.load(Ordering::SeqCst) {
                                return;
                            }
                            let reply = mock.execute(&args);
                            let mut out = Vec::new();
                            reply.encode(&mut out);
                            if sock.write_all(&out).await.is_err() {
                                return;
                            }
                        }
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                });
            }
        });
        mock
    }

    async fn session(&self, id: &str) -> RedisSession {
        RedisSession::from_url(id, &self.url).await.unwrap()
    }

    fn execute(&self, args: &[String]) -> Reply {
        let name = args[0].to_uppercase();
        let mut store = self.store.lock().unwrap();
        match name.as_str() {
            "PING" => Reply::Bulk("PONG".into()).into_simple_pong(),
            "EVALSHA" => Reply::Error("NOSCRIPT No matching script. Please use EVAL.".into()),
            "EVAL" => self.eval(&mut store, args),
            "LRANGE" => {
                let list = store.lists.get(&args[1]).cloned().unwrap_or_default();
                let n = list.len() as i64;
                let norm = |i: i64| if i < 0 { (n + i).max(0) } else { i };
                let (start, stop) = (
                    norm(args[2].parse().unwrap()),
                    norm(args[3].parse().unwrap()),
                );
                if start > stop || start >= n {
                    return Reply::Array(vec![]);
                }
                let stop = stop.min(n - 1);
                Reply::Array(list[start as usize..=stop as usize].to_vec())
            }
            "RPOP" => {
                let popped = store.lists.get_mut(&args[1]).and_then(Vec::pop);
                if store.lists.get(&args[1]).is_some_and(Vec::is_empty) {
                    store.lists.remove(&args[1]);
                }
                popped.map_or(Reply::Nil, Reply::Bulk)
            }
            "DEL" => {
                let mut removed = 0;
                for key in &args[1..] {
                    if store.exists(key) {
                        removed += 1;
                    }
                    store.del(key);
                }
                Reply::Int(removed)
            }
            _ => Reply::Ok,
        }
    }

    fn eval(&self, store: &mut Store, args: &[String]) -> Reply {
        let script = &args[1];
        let numkeys: usize = args[2].parse().unwrap();
        let keys: Vec<String> = args[3..3 + numkeys].to_vec();
        let argv: Vec<String> = args[3 + numkeys..].to_vec();
        if let Some(err) = store.guard(&keys) {
            return Reply::Error(err);
        }
        let stamp = self.clock.fetch_add(1, Ordering::SeqCst).to_string();
        let ttl_ms: i64 = argv.first().and_then(|v| v.parse().ok()).unwrap_or(0);
        if script.contains("LTRIM") {
            let handle_count: usize = argv[1].parse().unwrap();
            let total = store.lists.get(&keys[0]).map_or(0, Vec::len);
            if total < handle_count {
                return Reply::Int(0);
            }
            let start = total - handle_count;
            let handles = &argv[2..2 + handle_count];
            if store
                .lists
                .get(&keys[0])
                .is_some_and(|l| l[start..] != *handles)
            {
                return Reply::Int(0);
            }
            if let Some(list) = store.lists.get_mut(&keys[0]) {
                list.truncate(start);
                if list.is_empty() {
                    store.lists.remove(&keys[0]);
                }
            }
            let item_count: usize = argv[2 + handle_count].parse().unwrap();
            for item in &argv[3 + handle_count..3 + handle_count + item_count] {
                store
                    .lists
                    .entry(keys[0].clone())
                    .or_default()
                    .push(item.clone());
            }
            store
                .hashes
                .entry(keys[1].clone())
                .or_default()
                .insert("updated_at".into(), stamp);
            if ttl_ms > 0 {
                for key in &keys {
                    if store.exists(key) {
                        store.ttl_ms.insert(key.clone(), ttl_ms);
                    }
                }
            }
            Reply::Int(1)
        } else if script.contains("HSETNX") {
            let hash = store.hashes.entry(keys[1].clone()).or_default();
            hash.insert("session_id".into(), argv[1].clone());
            hash.entry("created_at".into())
                .or_insert_with(|| stamp.clone());
            for item in &argv[2..] {
                store
                    .lists
                    .entry(keys[0].clone())
                    .or_default()
                    .push(item.clone());
            }
            store
                .hashes
                .get_mut(&keys[1])
                .unwrap()
                .insert("updated_at".into(), stamp);
            if ttl_ms > 0 {
                for key in &keys {
                    if store.exists(key) {
                        store.ttl_ms.insert(key.clone(), ttl_ms);
                    }
                }
            }
            Reply::Int(1)
        } else {
            for key in &keys {
                store.del(key);
            }
            Reply::Int(1)
        }
    }

    fn list(&self, key: &str) -> Vec<String> {
        self.store
            .lock()
            .unwrap()
            .lists
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    fn push_raw(&self, key: &str, value: &str) {
        self.store
            .lock()
            .unwrap()
            .lists
            .entry(key.into())
            .or_default()
            .push(value.into());
    }

    fn hash(&self, key: &str) -> HashMap<String, String> {
        self.store
            .lock()
            .unwrap()
            .hashes
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    fn ttl(&self, key: &str) -> Option<i64> {
        self.store.lock().unwrap().ttl_ms.get(key).copied()
    }

    fn exists(&self, key: &str) -> bool {
        self.store.lock().unwrap().exists(key)
    }
}

impl Reply {
    /// `PING` answers a simple string; the mock reuses `Bulk` for brevity.
    fn into_simple_pong(self) -> Reply {
        self
    }
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
// Basic operations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn direct_ops() {
    let mock = MockRedis::start().await;
    let session = mock.session("direct").await;

    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello", "Hi there!"]
    );

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "Hi there!"
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello"]
    );

    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn sessions_are_isolated() {
    let mock = MockRedis::start().await;
    let a = mock.session("a").await;
    let b = mock.session("b").await;
    a.add_items(vec![user("from a")]).await.unwrap();
    b.add_items(vec![user("from b")]).await.unwrap();
    assert_eq!(texts(&a.get_items(None).await.unwrap()), vec!["from a"]);
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["from b"]);
    a.clear_session().await.unwrap();
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["from b"]);
}

#[tokio::test]
async fn get_items_with_limit() {
    let mock = MockRedis::start().await;
    let session = mock.session("limit").await;
    session
        .add_items(vec![user("1"), assistant("2"), user("3"), assistant("4")])
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["3", "4"]
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 4);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
}

#[tokio::test]
async fn pop_from_an_empty_session_and_empty_adds() {
    let mock = MockRedis::start().await;
    let session = mock.session("empty").await;
    assert!(session.pop_item().await.unwrap().is_none());
    session.add_items(vec![]).await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(
        !mock.exists(session.messages_key()),
        "an empty add must write nothing"
    );
}

#[tokio::test]
async fn unicode_special_characters_and_problematic_strings_round_trip() {
    let mock = MockRedis::start().await;
    let session = mock.session("text").await;
    let samples = [
        "こんにちは",
        "😊👍",
        "Привет",
        "O'Reilly",
        "DROP TABLE sessions;",
        "{\"nested\": \"json\", \"a\": [1, 2]}",
        "line1\nline2\r\n\ttabbed",
        "quote \" backslash \\ null-ish \u{0001}",
        "",
    ];
    let items: Vec<InputItem> = samples.iter().map(|s| user(s)).collect();
    session.add_items(items).await.unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), samples);
}

#[tokio::test]
async fn concurrent_writes_are_all_kept() {
    let mock = MockRedis::start().await;
    let session = Arc::new(mock.session("concurrent").await);
    let mut handles = Vec::new();
    for i in 0..10 {
        let session = session.clone();
        handles.push(tokio::spawn(async move {
            session
                .add_items(vec![user(&format!("Message {i}"))])
                .await
                .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    let mut seen = texts(&session.get_items(None).await.unwrap());
    seen.sort();
    let mut expected: Vec<String> = (0..10).map(|i| format!("Message {i}")).collect();
    expected.sort();
    assert_eq!(seen, expected);
}

#[tokio::test]
async fn ping_reports_connectivity() {
    let mock = MockRedis::start().await;
    let session = mock.session("ping").await;
    assert!(session.ping().await);
}

#[tokio::test]
async fn ping_is_false_when_the_connection_fails() {
    let mock = MockRedis::start().await;
    let session = mock.session("ping_failure").await;
    assert!(session.ping().await);
    mock.alive.store(false, Ordering::SeqCst);
    assert!(!session.ping().await);
}

#[tokio::test]
async fn from_url_uses_the_default_key_layout() {
    let mock = MockRedis::start().await;
    let session = mock.session("customer:archive").await;
    assert_eq!(session.session_key(), "agents:session:customer:archive");
    assert_eq!(
        session.messages_key(),
        "agents:session:customer:archive:messages"
    );
}

#[tokio::test]
async fn key_prefixes_isolate_sessions() {
    let mock = MockRedis::start().await;
    let one = mock.session("same").await.with_key_prefix("one:");
    let two = mock.session("same").await.with_key_prefix("two:");
    one.add_items(vec![user("one")]).await.unwrap();
    two.add_items(vec![user("two")]).await.unwrap();
    assert_eq!(texts(&one.get_items(None).await.unwrap()), vec!["one"]);
    assert_eq!(texts(&two.get_items(None).await.unwrap()), vec!["two"]);
    assert!(mock.exists("one::same:messages") && mock.exists("two::same:messages"));
}

// ---------------------------------------------------------------------------
// TTL and metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ttl_is_applied_to_every_existing_key_in_the_write() {
    let mock = MockRedis::start().await;
    let session = mock.session("ttl_test").await.with_ttl(60);
    session.add_items(vec![user("expires")]).await.unwrap();
    assert_eq!(mock.ttl(session.messages_key()), Some(60_000));
    assert_eq!(mock.ttl(session.session_key()), Some(60_000));
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn an_unrepresentable_ttl_is_rejected_before_writing() {
    let mock = MockRedis::start().await;
    let session = mock.session("invalid_ttl").await.with_ttl(1 << 63);
    for _ in 0..2 {
        let err = session
            .add_items(vec![user("never committed")])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside Redis's supported"),
            "{err}"
        );
    }
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(!mock.exists(session.messages_key()));
}

#[tokio::test]
async fn created_at_is_stable_and_updated_at_advances() {
    let mock = MockRedis::start().await;
    let session = mock.session("created_at_test").await;
    session.add_items(vec![user("first")]).await.unwrap();
    let first = mock.hash(session.session_key());
    session.add_items(vec![user("second")]).await.unwrap();
    let second = mock.hash(session.session_key());

    assert_eq!(second["created_at"], first["created_at"]);
    assert_ne!(second["updated_at"], first["created_at"]);
    assert_eq!(second["session_id"], "created_at_test");
}

#[tokio::test]
async fn the_released_layout_stays_usable() {
    let mock = MockRedis::start().await;
    let session_key = "agents:session:customer:archive";
    let messages_key = "agents:session:customer:archive:messages";
    let counter_key = "agents:session:customer:archive:counter";
    {
        let mut store = mock.store.lock().unwrap();
        store.hashes.insert(
            session_key.into(),
            HashMap::from([
                ("session_id".to_string(), "customer:archive".to_string()),
                ("created_at".to_string(), "123".to_string()),
                ("updated_at".to_string(), "123".to_string()),
            ]),
        );
        store.strings.insert(counter_key.into(), "7".into());
    }
    mock.push_raw(
        messages_key,
        &json!({"role": "user", "content": "persisted"}).to_string(),
    );

    let session = mock.session("customer:archive").await.with_ttl(60);
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["persisted"]
    );
    session
        .add_items(vec![assistant("appended")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["persisted", "appended"]
    );
    assert_eq!(mock.hash(session_key)["created_at"], "123");
    assert_eq!(mock.store.lock().unwrap().strings[counter_key], "7");
    for key in [session_key, messages_key, counter_key] {
        assert_eq!(mock.ttl(key), Some(60_000), "{key}");
    }

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "appended"
    );
    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(!mock.exists(session_key) && !mock.exists(messages_key) && !mock.exists(counter_key));
    session.clear_session().await.unwrap();
}

// ---------------------------------------------------------------------------
// Key collisions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_colliding_session_clear_preserves_the_existing_session() {
    for (suffix, reverse) in [
        ("counter", false),
        ("counter", true),
        ("messages", false),
        ("messages", true),
    ] {
        let mock = MockRedis::start().await;
        let (existing_id, other_id) = if reverse {
            (format!("conversation:{suffix}"), "conversation".to_string())
        } else {
            ("conversation".to_string(), format!("conversation:{suffix}"))
        };
        let existing = mock.session(&existing_id).await;
        let other = mock.session(&other_id).await;
        existing
            .add_items(vec![user("saved history")])
            .await
            .unwrap();
        if suffix == "counter" && !reverse {
            mock.store
                .lock()
                .unwrap()
                .strings
                .insert("agents:session:conversation:counter".into(), "7".into());
        }
        let metadata = mock.hash(existing.session_key());

        let err = other.clear_session().await.unwrap_err().to_string();
        assert!(err.contains("WRONGTYPE"), "{suffix}/{reverse}: {err}");

        assert_eq!(
            texts(&existing.get_items(None).await.unwrap()),
            vec!["saved history"]
        );
        assert_eq!(mock.hash(existing.session_key()), metadata);
        existing.clear_session().await.unwrap();
        assert!(existing.get_items(None).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn a_colliding_counter_rejects_a_ttl_write_before_side_effects() {
    let mock = MockRedis::start().await;
    let existing = mock.session("conversation:counter").await;
    let other = mock.session("conversation").await.with_ttl(1);
    existing
        .add_items(vec![user("saved history")])
        .await
        .unwrap();

    let err = other
        .add_items(vec![user("saved history")])
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("WRONGTYPE"), "{err}");
    assert_eq!(mock.ttl(existing.session_key()), None);
    assert_eq!(
        texts(&existing.get_items(None).await.unwrap()),
        vec!["saved history"]
    );
    assert!(other.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_wrong_metadata_key_type_rejects_the_write_before_any_history_is_added() {
    let mock = MockRedis::start().await;
    let session = mock.session("wrong_metadata_type").await;
    mock.store
        .lock()
        .unwrap()
        .strings
        .insert(session.session_key().into(), "not a hash".into());

    for _ in 0..2 {
        let err = session
            .add_items(vec![user("never committed")])
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("WRONGTYPE") && err.contains("expected a hash"),
            "{err}"
        );
    }
    assert!(session.get_items(None).await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Corrupt data
// ---------------------------------------------------------------------------

#[tokio::test]
async fn limit_counts_valid_items_past_corrupt_newest_messages() {
    let mock = MockRedis::start().await;
    let session = mock.session("limit_corrupt_test").await;
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    mock.push_raw(session.messages_key(), "not valid json {{{");

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn limit_widening_stops_at_the_end_of_history() {
    let mock = MockRedis::start().await;
    let session = mock.session("limit_exhausted_test").await;
    session.add_items(vec![user("only valid")]).await.unwrap();
    assert_eq!(
        texts(&session.get_items(Some(5)).await.unwrap()),
        vec!["only valid"]
    );
}

#[tokio::test]
async fn corrupted_data_is_skipped_on_read_and_dropped_on_pop() {
    let mock = MockRedis::start().await;
    let session = mock.session("corruption_test").await;
    session
        .add_items(vec![user("valid message")])
        .await
        .unwrap();
    mock.push_raw(session.messages_key(), "invalid json data");
    mock.push_raw(session.messages_key(), "{incomplete json");

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["valid message"]
    );

    session
        .add_items(vec![user("valid after corruption")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["valid message", "valid after corruption"]
    );

    mock.push_raw(session.messages_key(), "corrupted at end");
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "valid after corruption"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "valid message"
    );
    assert!(session.pop_item().await.unwrap().is_none());
    assert!(mock.list(session.messages_key()).is_empty());
}

// ---------------------------------------------------------------------------
// Session settings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_settings_default_and_constructor() {
    let mock = MockRedis::start().await;
    assert!(mock.session("s").await.session_settings().is_none());
    let configured = mock
        .session("s")
        .await
        .with_settings(SessionSettings::with_limit(5));
    assert_eq!(configured.session_settings().unwrap().limit, Some(5));
}

#[tokio::test]
async fn the_sessions_limit_applies_and_an_explicit_limit_wins() {
    let mock = MockRedis::start().await;
    let session = mock
        .session("settings_limit")
        .await
        .with_settings(SessionSettings::with_limit(3));
    session
        .add_items((0..5).map(|i| user(&format!("Message {i}"))).collect())
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Message 2", "Message 3", "Message 4"]
    );
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
}

#[tokio::test]
async fn the_runner_stores_turns_and_honours_a_run_config_limit() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mock = MockRedis::start().await;
    let session = Arc::new(mock.session("runner").await);
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
    assert!(stored.contains(&"New question".to_string()) && stored.contains(&"Got it".to_string()));
}

// ---------------------------------------------------------------------------
// Compaction snapshots
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compaction_replaces_the_history_suffix_atomically() {
    let mock = MockRedis::start().await;
    let session = mock.session("compact").await.with_ttl(60);
    session
        .add_items(vec![user("keep"), assistant("old 1"), assistant("old 2")])
        .await
        .unwrap();

    let snapshot = session.compaction_snapshot(2).await.unwrap().unwrap();
    assert_eq!(texts(&snapshot.items), vec!["old 1", "old 2"]);
    assert!(!snapshot.complete);

    let replaced = session
        .replace_compaction_suffix(&snapshot.handles, vec![assistant("summary")])
        .await
        .unwrap();
    assert!(replaced);
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["keep", "summary"]
    );
    assert_eq!(mock.ttl(session.messages_key()), Some(60_000));
}

#[tokio::test]
async fn compaction_of_the_whole_history_keeps_the_key_alive_with_its_replacement() {
    let mock = MockRedis::start().await;
    let session = mock.session("compact_all").await;
    session
        .add_items(vec![user("a"), assistant("b")])
        .await
        .unwrap();

    let snapshot = session.compaction_snapshot(10).await.unwrap().unwrap();
    assert!(snapshot.complete);
    session
        .replace_compaction_suffix(&snapshot.handles, vec![assistant("summary")])
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["summary"]
    );
    assert!(mock.hash(session.session_key()).contains_key("updated_at"));
}

#[tokio::test]
async fn a_replacement_is_refused_when_history_changed_since_the_snapshot() {
    let mock = MockRedis::start().await;
    let session = mock.session("compact_race").await;
    session
        .add_items(vec![user("a"), assistant("b")])
        .await
        .unwrap();
    let snapshot = session.compaction_snapshot(2).await.unwrap().unwrap();

    session.add_items(vec![user("newer")]).await.unwrap();
    let replaced = session
        .replace_compaction_suffix(&snapshot.handles, vec![assistant("summary")])
        .await
        .unwrap();

    assert!(!replaced);
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["a", "b", "newer"]
    );

    let stale = vec![SnapshotHandle("not an entry".into())];
    assert!(
        !session
            .replace_compaction_suffix(&stale, vec![])
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn a_replacement_rejects_a_colliding_counter_key() {
    let mock = MockRedis::start().await;
    let existing = mock.session("conversation:counter").await;
    let other = mock.session("conversation").await;
    existing
        .add_items(vec![user("saved history")])
        .await
        .unwrap();
    other.add_items(vec![]).await.unwrap();

    let err = other
        .replace_compaction_suffix(&[], vec![assistant("summary")])
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("WRONGTYPE"), "{err}");
    assert_eq!(
        texts(&existing.get_items(None).await.unwrap()),
        vec!["saved history"]
    );
}
