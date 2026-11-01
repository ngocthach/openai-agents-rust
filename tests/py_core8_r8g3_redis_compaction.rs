//! Compaction over `RedisSession`, ports of the compaction tests in
//! `tests/extensions/memory/test_redis_session.py`:
//! `test_compaction_replaces_redis_history_atomically` (4),
//! `test_atomic_replacement_preserves_clear_add_metadata` (2) and
//! `test_redis_compaction_preserves_encryption_wrapper` (2).
//!
//! Python replaces the whole history in one Redis transaction. Rust replaces the snapshot suffix
//! in one Lua script; the mock below is an in-process RESP2 server that replicates the scripts'
//! effects keyed on the script text (as `py_memory_redis.rs` does) and records the number of
//! history entries after every command, so a clear-then-append gap would be visible.
#![cfg(all(feature = "redis", feature = "encrypt"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    CompactionMode, EncryptedSession, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, RedisSession, Session,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

mod py_mcpmem_support;
use py_mcpmem_support::{item_text, mock_openai};

const KEY: &str = "0123456789abcdef0123456789abcdef";

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
    /// Total number of history entries after each command.
    trace: Arc<Mutex<Vec<usize>>>,
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
            trace: Arc::new(Mutex::new(Vec::new())),
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
                            mock.observe();
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

    fn observe(&self) {
        let total = self
            .store
            .lock()
            .unwrap()
            .lists
            .values()
            .map(Vec::len)
            .sum();
        self.trace.lock().unwrap().push(total);
    }

    fn set_hash_field(&self, key: &str, field: &str, value: &str) {
        self.store
            .lock()
            .unwrap()
            .hashes
            .entry(key.into())
            .or_default()
            .insert(field.into(), value.into());
    }

    fn set_string(&self, key: &str, value: &str) {
        self.store
            .lock()
            .unwrap()
            .strings
            .insert(key.into(), value.into());
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

    #[allow(dead_code)]
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

fn summary_output(empty: bool) -> Vec<Value> {
    if empty {
        Vec::new()
    } else {
        vec![json!({"role": "user", "content": "summary"})]
    }
}

fn expected_texts(output: &[Value]) -> Vec<String> {
    output
        .iter()
        .map(|v| v["content"].as_str().unwrap().to_string())
        .collect()
}

fn force_input_mode() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        compaction_mode: Some(CompactionMode::Input),
        force: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn compaction_replaces_redis_history_atomically() {
    openai_agents_sdk::set_tracing_disabled(true);
    for automatic in [false, true] {
        for empty in [false, true] {
            let case = format!("automatic={automatic} empty={empty}");
            let mock = MockRedis::start().await;
            let backend = Arc::new(mock.session("atomic-compaction").await);
            let reader = mock.session("atomic-compaction").await;
            backend.add_items(vec![user("original")]).await.unwrap();

            let output = summary_output(empty);
            let (base, captured) = mock_openai(vec![compact_reply(&output)]).await;
            let compaction = Arc::new(
                OpenAIResponsesCompactionSession::new("atomic-compaction", backend.clone())
                    .unwrap()
                    .with_client(OpenAIClient::new("sk-test").with_base_url(base))
                    .with_compaction_mode(CompactionMode::Input)
                    .with_should_trigger_compaction(|_| true),
            );

            mock.trace.lock().unwrap().clear();
            if automatic {
                let model = Arc::new(FakeModel::new());
                model.add_turn(vec![assistant_message("reply")]);
                let agent = Agent::<()>::new("test").with_model_instance(model);
                Runner::run_with_options(
                    &agent,
                    "next",
                    RunOptions::new(()).session(compaction.clone() as Arc<dyn Session>),
                )
                .await
                .unwrap();
            } else {
                compaction.run_compaction(force_input_mode()).await.unwrap();
            }

            assert_eq!(captured.lock().unwrap().len(), 1, "{case}");
            let trace = mock.trace.lock().unwrap().clone();
            assert!(!trace.is_empty(), "{case}");
            assert_eq!(*trace.last().unwrap(), output.len(), "{case}: {trace:?}");
            if !empty {
                // The replacement never exposes an empty history to a second reader.
                assert!(trace.iter().all(|len| *len > 0), "{case}: {trace:?}");
            }
            assert_eq!(
                texts(&reader.get_items(None).await.unwrap()),
                expected_texts(&output),
                "{case}"
            );
        }
    }
}

#[tokio::test]
async fn replacing_the_whole_history_behaves_like_clear_then_add() {
    for empty in [false, true] {
        let case = format!("empty={empty}");
        let mock = MockRedis::start().await;
        let session = Arc::new(mock.session("atomic-replacement-ttl").await.with_ttl(3600));
        session.add_items(vec![user("before")]).await.unwrap();
        mock.set_hash_field(session.session_key(), "created_at", "1");
        mock.set_string(&format!("{}:counter", session.session_key()), "10");

        let output = summary_output(empty);
        let (base, _) = mock_openai(vec![compact_reply(&output)]).await;
        let compaction =
            OpenAIResponsesCompactionSession::new("atomic-replacement-ttl", session.clone())
                .unwrap()
                .with_client(OpenAIClient::new("sk-test").with_base_url(base));
        compaction.run_compaction(force_input_mode()).await.unwrap();

        assert_eq!(
            texts(&session.get_items(None).await.unwrap()),
            expected_texts(&output),
            "{case}"
        );
        assert!(
            !mock.exists(&format!("{}:counter", session.session_key())),
            "{case}"
        );
        if empty {
            assert!(!mock.exists(session.session_key()), "{case}");
        } else {
            assert_ne!(
                mock.hash(session.session_key())
                    .get("created_at")
                    .map(String::as_str),
                Some("1"),
                "{case}"
            );
            for key in [session.messages_key(), session.session_key()] {
                let ttl = mock.ttl(key).unwrap_or(0);
                assert!(0 < ttl && ttl <= 3_600_000, "{case}: {key} ttl {ttl}");
            }
        }
    }
}

#[tokio::test]
async fn redis_compaction_preserves_the_encryption_wrapper() {
    for encryption_outermost in [false, true] {
        let case = format!("encryption_outermost={encryption_outermost}");
        let mock = MockRedis::start().await;
        let backend = Arc::new(mock.session("encrypted-compaction").await);
        let output = vec![json!({"role": "user", "content": "synthetic summary"})];
        let (base, _) = mock_openai(vec![compact_reply(&output)]).await;
        let client = OpenAIClient::new("sk-test").with_base_url(base);

        let (session, compaction): (Arc<dyn Session>, Arc<OpenAIResponsesCompactionSession>) =
            if encryption_outermost {
                let compaction = Arc::new(
                    OpenAIResponsesCompactionSession::new("encrypted-compaction", backend.clone())
                        .unwrap()
                        .with_client(client),
                );
                let encrypted = EncryptedSession::new(
                    "encrypted-compaction",
                    compaction.clone() as Arc<dyn Session>,
                    KEY,
                    600,
                )
                .unwrap();
                (Arc::new(encrypted), compaction)
            } else {
                let encrypted = Arc::new(
                    EncryptedSession::new(
                        "encrypted-compaction",
                        backend.clone() as Arc<dyn Session>,
                        KEY,
                        600,
                    )
                    .unwrap(),
                );
                let compaction = Arc::new(
                    OpenAIResponsesCompactionSession::new("encrypted-compaction", encrypted)
                        .unwrap()
                        .with_client(client),
                );
                (compaction.clone() as Arc<dyn Session>, compaction)
            };

        session
            .add_items(vec![user("synthetic original")])
            .await
            .unwrap();
        compaction.run_compaction(force_input_mode()).await.unwrap();

        assert_eq!(
            texts(&session.get_items(None).await.unwrap()),
            vec!["synthetic summary".to_string()],
            "{case}"
        );
        let stored = backend.get_items(None).await.unwrap();
        assert_eq!(stored.len(), 1, "{case}");
        assert_eq!(stored[0].as_value()["__enc__"], json!(1), "{case}");
        assert!(
            !serde_json::to_string(&stored)
                .unwrap()
                .contains("synthetic summary"),
            "{case}"
        );
    }
}
