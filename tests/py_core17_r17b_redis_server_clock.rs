//! Redis-backed compaction cases of `tests/extensions/memory/test_redis_session.py` that depend on
//! the server clock and on a caller that goes away after Redis committed:
//!
//! - `test_add_items_uses_server_absolute_expiration_after_serialization`,
//!   `test_add_items_rejects_unrepresentable_ttl_before_writing`: a TTL is valid when the absolute
//!   expiration (the server clock plus the TTL) is representable, and a TTL that is not is
//!   rejected before anything is written, so a retry cannot duplicate history.
//! - `test_compaction_cancel_after_redis_commit_invalidates_cached_history`: the caller of a
//!   compaction goes away after the replacement committed; the next compaction must read the
//!   committed history instead of replaying the candidates cached before the replacement.
//!
//! The mock is an in-process RESP2 server that replicates the session scripts' effects keyed on
//! the script text, as `py_core8_r8g3_redis_compaction.rs` does.
#![cfg(feature = "redis")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, RedisSession,
    Session,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::{item_text, mock_openai};

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

/// Notified once the replacement is applied, and awaited before the reply is sent.
type ReplacementHold = (Arc<Notify>, Arc<Notify>);

#[derive(Clone)]
struct MockRedis {
    store: Arc<Mutex<Store>>,
    clock: Arc<AtomicI64>,
    alive: Arc<AtomicBool>,
    /// Command names in arrival order.
    commands: Arc<Mutex<Vec<String>>>,
    /// Armed by a test: after the next suffix replacement is applied, tell the test and hold the
    /// reply until it is released. A replacement is either of the two scripts that rewrite history.
    hold_replacement: Arc<Mutex<Option<ReplacementHold>>>,
    evals: Arc<AtomicUsize>,
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
            commands: Arc::new(Mutex::new(Vec::new())),
            hold_replacement: Arc::new(Mutex::new(None)),
            evals: Arc::new(AtomicUsize::new(0)),
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
                            let held = if args[0].eq_ignore_ascii_case("EVAL")
                                && (args[1].contains("LTRIM") || args[1].contains("#ARGV > 2"))
                            {
                                mock.hold_replacement.lock().unwrap().take()
                            } else {
                                None
                            };
                            if let Some((committed, release)) = held {
                                committed.notify_one();
                                release.notified().await;
                            }
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
        self.commands.lock().unwrap().push(name.clone());
        if name == "EVAL" {
            self.evals.fetch_add(1, Ordering::SeqCst);
        }
        let mut store = self.store.lock().unwrap();
        match name.as_str() {
            "PING" => Reply::Bulk("PONG".into()).into_simple_pong(),
            "TIME" => Reply::Array(vec![
                self.clock.load(Ordering::SeqCst).to_string(),
                "500000".into(),
            ]),
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
        if script.contains("#ARGV > 2") {
            // The whole-history replacement: clear every key, then append like an add.
            for key in &keys {
                store.del(key);
            }
            if argv.len() > 2 {
                let hash = store.hashes.entry(keys[1].clone()).or_default();
                hash.insert("session_id".into(), argv[1].clone());
                hash.insert("created_at".into(), stamp.clone());
                hash.insert("updated_at".into(), stamp);
                for item in &argv[2..] {
                    store
                        .lists
                        .entry(keys[0].clone())
                        .or_default()
                        .push(item.clone());
                }
                if ttl_ms > 0 {
                    for key in &keys {
                        if store.exists(key) {
                            store.ttl_ms.insert(key.clone(), ttl_ms);
                        }
                    }
                }
            }
            Reply::Int(1)
        } else if script.contains("LTRIM") {
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

    #[allow(dead_code)]
    fn list(&self, key: &str) -> Vec<String> {
        self.store
            .lock()
            .unwrap()
            .lists
            .get(key)
            .cloned()
            .unwrap_or_default()
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

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn compact_reply(output: &[Value]) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })
}

fn force_input_mode() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        compaction_mode: Some(CompactionMode::Input),
        force: true,
        ..Default::default()
    }
}

const TTL_ERROR: &str = "outside Redis's supported expiration range";

#[tokio::test]
async fn a_ttl_is_valid_while_the_absolute_expiration_is_representable() {
    let mock = MockRedis::start().await;
    let server_seconds = 1_750_000_000;
    mock.clock.store(server_seconds, Ordering::SeqCst);
    let server_time_ms = server_seconds * 1000 + 500;
    let ttl = ((i64::MAX - server_time_ms) / 1000) as u64;

    let session = mock.session("absolute_ttl").await.with_ttl(ttl);
    session.add_items(vec![user("once")]).await.unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), ["once"]);
    // The server clock is read after the items are serialized and before the write.
    let commands = mock.commands.lock().unwrap().clone();
    let time = commands.iter().position(|c| c == "TIME").unwrap();
    let eval = commands.iter().position(|c| c == "EVAL").unwrap();
    assert!(time < eval, "{commands:?}");
}

#[tokio::test]
async fn a_ttl_past_the_server_clock_horizon_is_rejected_before_writing() {
    let mock = MockRedis::start().await;
    let server_seconds = 1_750_000_000;
    mock.clock.store(server_seconds, Ordering::SeqCst);
    let server_time_ms = server_seconds * 1000 + 500;
    // One second more than the largest TTL whose absolute expiration still fits.
    let ttl = ((i64::MAX - server_time_ms) / 1000) as u64 + 1;
    assert!(
        i64::try_from(ttl * 1000).is_ok(),
        "the relative TTL itself fits"
    );

    let session = mock.session("past_horizon").await.with_ttl(ttl);
    for _ in 0..2 {
        let error = session
            .add_items(vec![user("never committed")])
            .await
            .unwrap_err();
        assert!(error.to_string().contains(TTL_ERROR), "{error}");
    }
    assert_eq!(mock.evals.load(Ordering::SeqCst), 0, "no script may run");
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_compaction_replacement_checks_the_ttl_horizon_before_writing() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mock = MockRedis::start().await;
    mock.clock.store(1_750_000_000, Ordering::SeqCst);
    let seeding = mock.session("replace_horizon").await;
    seeding.add_items(vec![user("original")]).await.unwrap();
    let evals_before = mock.evals.load(Ordering::SeqCst);

    let ttl = ((i64::MAX - 1_750_000_000_500) / 1000) as u64 + 1;
    let expiring = Arc::new(mock.session("replace_horizon").await.with_ttl(ttl));
    let output = vec![json!({"role": "user", "content": "summary"})];
    let (base, _captured) = mock_openai(vec![compact_reply(&output)]).await;
    let compaction = OpenAIResponsesCompactionSession::new("replace_horizon", expiring)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input);

    let error = compaction
        .run_compaction(force_input_mode())
        .await
        .unwrap_err();
    assert!(error.to_string().contains(TTL_ERROR), "{error}");
    assert_eq!(mock.evals.load(Ordering::SeqCst), evals_before);
    assert_eq!(texts(&seeding.get_items(None).await.unwrap()), ["original"]);
}

#[tokio::test]
async fn the_next_compaction_reads_history_committed_after_the_caller_went_away() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mock = MockRedis::start().await;
    let backend = Arc::new(mock.session("cancel-after-commit").await);
    backend.add_items(vec![user("original")]).await.unwrap();
    let output = vec![json!({"role": "user", "content": "summary"})];
    let (base, captured) = mock_openai(vec![compact_reply(&output), compact_reply(&output)]).await;
    let compaction = Arc::new(
        OpenAIResponsesCompactionSession::new("cancel-after-commit", backend.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_compaction_mode(CompactionMode::Input),
    );

    let committed = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *mock.hold_replacement.lock().unwrap() = Some((committed.clone(), release.clone()));
    let caller = {
        let compaction = compaction.clone();
        tokio::spawn(async move {
            let _ = compaction.run_compaction(force_input_mode()).await;
        })
    };
    tokio::time::timeout(Duration::from_secs(5), committed.notified())
        .await
        .expect("the replacement never committed");
    // Redis applied the replacement; the caller goes away before it hears back.
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    release.notify_one();

    let mut stored = Vec::new();
    for _ in 0..200 {
        stored = texts(&backend.get_items(None).await.unwrap());
        if stored == ["summary"] {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(stored, ["summary"]);

    // The next request must load the committed history, not the candidates cached before the
    // replacement.
    compaction.run_compaction(force_input_mode()).await.unwrap();
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].body["input"], json!(output));
}
