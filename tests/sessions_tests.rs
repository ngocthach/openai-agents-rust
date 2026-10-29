//! Exercise the session backends end to end: encryption with TTL, SQLite branching and usage
//! analytics, Responses compaction against a mock OpenAI endpoint, and the Dapr state API
//! against a mock sidecar. No real network or external services are used.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "sqlite")]
use openai_agents_sdk::memory::AdvancedSqliteSession;
#[cfg(feature = "encrypt")]
use openai_agents_sdk::memory::{CompactionMode, EncryptedSession};
use openai_agents_sdk::memory::{
    DaprConsistency, DaprSession, InMemorySession, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, Session, SessionRunCompletion,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[cfg(feature = "encrypt")]
const TEST_KEY: &str = "0123456789abcdef0123456789abcdef";

/// One request seen by a mock server: method, path with query, and parsed JSON body.
#[derive(Debug, Clone)]
struct Request {
    method: String,
    path: String,
    body: Value,
}

/// What a mock server answers with.
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Reply {
    fn json(value: Value) -> Self {
        Reply {
            status: 200,
            headers: vec![("content-type".into(), "application/json".into())],
            body: value.to_string(),
        }
    }

    fn empty(status: u16) -> Self {
        Reply {
            status,
            headers: Vec::new(),
            body: String::new(),
        }
    }
}

type Handler = Arc<dyn Fn(&Request) -> Reply + Send + Sync>;

/// Start a mock HTTP server driven by `handler`; returns its base URL.
async fn serve(handler: Handler) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                let Some(request) = read_request(&mut sock).await else {
                    return;
                };
                let reply = handler(&request);
                let mut head = format!(
                    "HTTP/1.1 {} OK\r\ncontent-length: {}\r\nconnection: close\r\n",
                    reply.status,
                    reply.body.len()
                );
                for (k, v) in &reply.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                head.push_str(&reply.body);
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

/// Read one HTTP request, honouring `content-length`.
async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        let Some(idx) = text.find("\r\n\r\n") else {
            continue;
        };
        let headers = &text[..idx];
        let len: usize = headers
            .lines()
            .find_map(|l| {
                l.to_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse().unwrap_or(0))
            })
            .unwrap_or(0);
        if buf.len() < idx + 4 + len {
            continue;
        }
        let mut parts = headers.split_whitespace();
        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();
        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null);
        return Some(Request { method, path, body });
    }
}

/// A mock OpenAI endpoint answering the queued replies in order; returns the base URL for
/// [`OpenAIClient::with_base_url`] and the captured requests.
async fn mock_openai(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<Request>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let queue = Arc::new(Mutex::new(replies.into_iter().collect::<Vec<_>>()));
    let cap = captured.clone();
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        let mut queue = queue.lock().unwrap();
        let reply = if queue.is_empty() {
            json!({"error": {"message": "unexpected request"}})
        } else {
            queue.remove(0)
        };
        Reply::json(reply)
    }))
    .await;
    (format!("{base}/v1"), captured)
}

fn compaction_reply(text: &str) -> Value {
    json!({
        "id": "cmp_1",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {"input_tokens": 500, "output_tokens": 30, "total_tokens": 530},
    })
}

/// A long history: one user message plus enough model items to pass the default threshold.
fn long_history() -> Vec<InputItem> {
    let mut items = vec![InputItem::user("What is the weather?")];
    for i in 0..12 {
        items.push(InputItem::assistant(format!("turn {i}")));
    }
    items
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn encrypted_session_round_trips_and_hides_plaintext() {
    let inner = Arc::new(InMemorySession::new("s1"));
    let session = EncryptedSession::new("s1", inner.clone(), TEST_KEY, 600).unwrap();

    session
        .add_items(vec![
            InputItem::user("my social security number"),
            InputItem::assistant("noted"),
        ])
        .await
        .unwrap();

    let stored = inner.get_items(None).await.unwrap();
    assert_eq!(stored.len(), 2);
    for item in &stored {
        let v = item.as_value();
        assert_eq!(v["__enc__"], 1);
        assert_eq!(v["kid"], "hkdf-v1");
        assert_eq!(v["v"], 1);
        assert!(v["payload"].as_str().is_some());
        assert!(!v.to_string().contains("social"));
    }

    let items = session.get_items(None).await.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].text_content().as_deref(),
        Some("my social security number")
    );
    assert_eq!(session.get_items(Some(1)).await.unwrap().len(), 1);

    assert_eq!(
        session
            .pop_item()
            .await
            .unwrap()
            .unwrap()
            .text_content()
            .as_deref(),
        Some("noted")
    );
    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(inner.get_items(None).await.unwrap().is_empty());
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn encrypted_session_skips_expired_items() {
    let inner = Arc::new(InMemorySession::new("s1"));
    let session = EncryptedSession::new("s1", inner.clone(), TEST_KEY, 1).unwrap();
    session
        .add_items(vec![InputItem::user("short lived")])
        .await
        .unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);

    tokio::time::sleep(std::time::Duration::from_millis(2100)).await;

    // The envelope is still stored, but it is no longer returned as history.
    assert_eq!(inner.get_items(None).await.unwrap().len(), 1);
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(session.get_items(Some(5)).await.unwrap().is_empty());
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn encrypted_session_rejects_a_wrong_key_and_a_wrong_session_id() {
    let inner = Arc::new(InMemorySession::new("s1"));
    let writer = EncryptedSession::new("s1", inner.clone(), TEST_KEY, 600).unwrap();
    writer
        .add_items(vec![InputItem::user("classified")])
        .await
        .unwrap();

    let wrong_key =
        EncryptedSession::new("s1", inner.clone(), "fedcba9876543210fedcba9876543210", 600)
            .unwrap();
    assert!(wrong_key.get_items(None).await.unwrap().is_empty());

    // The session id is the HKDF salt, so another session cannot read these items either.
    let wrong_salt = EncryptedSession::new("s2", inner.clone(), TEST_KEY, 600).unwrap();
    assert!(wrong_salt.get_items(None).await.unwrap().is_empty());

    // The original key still reads them.
    assert_eq!(writer.get_items(None).await.unwrap().len(), 1);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn advanced_sqlite_session_tracks_branches_turns_and_usage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("advanced.db");
    let session = AdvancedSqliteSession::open("u1", &path).unwrap();

    session
        .add_items(vec![
            InputItem::user("What is Rust?"),
            InputItem(json!({"type": "function_call", "name": "search", "arguments": "{}"})),
            InputItem::assistant("A systems language"),
        ])
        .await
        .unwrap();
    session
        .store_run_usage(&Usage::single(200, 40))
        .await
        .unwrap();
    session
        .add_items(vec![
            InputItem::user("And Python?"),
            InputItem::assistant("A scripting language"),
        ])
        .await
        .unwrap();
    session
        .store_run_usage(&Usage::single(80, 20))
        .await
        .unwrap();

    assert_eq!(session.get_items(None).await.unwrap().len(), 5);
    let turns = session.get_conversation_turns(None).await.unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[1].content, "And Python?");

    let tools = session.get_tool_usage(None).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_name, "search");

    let usage = session.get_session_usage(None).await.unwrap().unwrap();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.total_tokens, 340);
    let turn_usage = session.get_turn_usage(Some(1), None).await.unwrap();
    assert_eq!(turn_usage.len(), 1);
    assert_eq!(turn_usage[0].input_tokens, 200);

    // Fork before turn 2: the first turn's items are shared, the rest is dropped.
    let branch = session
        .create_branch_from_turn(2, Some("retry"))
        .await
        .unwrap();
    assert_eq!(branch, "retry");
    assert_eq!(session.current_branch(), "retry");
    let forked = session.get_items(None).await.unwrap();
    assert_eq!(forked.len(), 3);
    session
        .add_items(vec![InputItem::user("And Go?")])
        .await
        .unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
    assert_eq!(session.get_conversation_turns(None).await.unwrap().len(), 2);

    session.switch_to_branch("main").await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 5);

    let branches = session.list_branches().await.unwrap();
    assert_eq!(branches.len(), 2);
    let main = branches.iter().find(|b| b.branch_id == "main").unwrap();
    assert!(main.is_current);
    assert_eq!(main.user_turns, 2);

    // Reopening the file keeps structure and usage.
    drop(session);
    let reopened = AdvancedSqliteSession::open("u1", &path).unwrap();
    assert_eq!(reopened.get_items(None).await.unwrap().len(), 5);
    assert_eq!(reopened.list_branches().await.unwrap().len(), 2);
    assert_eq!(
        reopened
            .get_session_usage(Some("main"))
            .await
            .unwrap()
            .unwrap()
            .total_tokens,
        340
    );

    reopened.delete_branch("retry", false).await.unwrap();
    assert_eq!(reopened.list_branches().await.unwrap().len(), 1);
    assert_eq!(reopened.get_items(None).await.unwrap().len(), 5);
}

#[tokio::test]
async fn compaction_session_sends_previous_response_id_and_replaces_history() {
    let (base, captured) = mock_openai(vec![compaction_reply("Summary of the conversation")]).await;
    let inner = Arc::new(InMemorySession::new("s1"));
    inner.add_items(long_history()).await.unwrap();

    let session = OpenAIResponsesCompactionSession::new("s1", inner.clone())
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));

    let usage = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp_42".into()),
            store: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/v1/responses/compact");
    assert_eq!(requests[0].body["model"], "gpt-4.1");
    assert_eq!(requests[0].body["previous_response_id"], "resp_42");
    assert!(requests[0].body.get("input").is_none());

    // History was replaced by the compacted output.
    let items = inner.get_items(None).await.unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].text_content().as_deref(),
        Some("Summary of the conversation")
    );
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 530);
}

#[tokio::test]
async fn compaction_session_declines_below_the_threshold_and_honours_force() {
    let (base, captured) = mock_openai(vec![compaction_reply("Summary")]).await;
    let inner = Arc::new(InMemorySession::new("s1"));
    inner
        .add_items(vec![InputItem::user("hi"), InputItem::assistant("hello")])
        .await
        .unwrap();
    let session = OpenAIResponsesCompactionSession::new("s1", inner.clone())
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));

    let declined = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp_1".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(declined, Usage::default());
    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(inner.get_items(None).await.unwrap().len(), 2);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp_1".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(captured.lock().unwrap().len(), 1);
    assert_eq!(inner.get_items(None).await.unwrap().len(), 1);
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn compaction_over_an_encrypted_store_sends_plaintext_and_rewrites_envelopes() {
    let (base, captured) = mock_openai(vec![compaction_reply("Compacted")]).await;
    let store = Arc::new(InMemorySession::new("s1"));
    let encrypted = Arc::new(EncryptedSession::new("s1", store.clone(), TEST_KEY, 600).unwrap());
    encrypted
        .add_items(vec![
            InputItem::user("remember my secret phrase"),
            InputItem::assistant("ok"),
        ])
        .await
        .unwrap();

    let session = OpenAIResponsesCompactionSession::new("s1", encrypted.clone())
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp_1".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    // The compaction request carries decrypted history, not envelopes.
    let body = captured.lock().unwrap()[0].body.clone();
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert!(body.to_string().contains("remember my secret phrase"));
    assert!(!body.to_string().contains("__enc__"));

    // The replacement went back through the encryption layer.
    let raw = store.get_items(None).await.unwrap();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0].as_value()["__enc__"], 1);
    assert_eq!(
        encrypted.get_items(None).await.unwrap()[0]
            .text_content()
            .as_deref(),
        Some("Compacted")
    );
}

#[tokio::test]
async fn runner_triggers_compaction_after_a_run() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, captured) = mock_openai(vec![compaction_reply("Recap")]).await;

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Hello!")]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model.clone());

    let inner = Arc::new(InMemorySession::new("s1"));
    inner.add_items(long_history()).await.unwrap();
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("s1", inner.clone())
            .with_client(OpenAIClient::new("sk-test").with_base_url(base)),
    );

    let result = Runner::run_with_options(
        &agent,
        "Say hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Hello!"));

    // The runner called `on_run_completed`, which compacted using the model's response id.
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, "/v1/responses/compact");
    assert_eq!(requests[0].body["previous_response_id"], "resp_fake_1");
    assert_eq!(model.calls().len(), 1);

    // History is the compacted output, and the compaction usage is part of the run usage.
    let items = inner.get_items(None).await.unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].text_content().as_deref(), Some("Recap"));
    let usage = result.usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.input_tokens, 10 + 500);
    assert_eq!(usage.output_tokens, 5 + 30);
}

/// State of the mock Dapr sidecar: the store (value + etag per key) and the requests seen.
#[derive(Debug, Default)]
struct DaprState {
    store: HashMap<String, (String, String)>,
    requests: Vec<Request>,
    conflicts_left: u32,
}

/// A mock Dapr sidecar: an in-memory state store with ETags.
///
/// The first `conflicts` conditional writes to the messages key are rejected with 409 so the
/// session's retry path is exercised.
async fn mock_dapr(conflicts: u32) -> (String, Arc<Mutex<DaprState>>) {
    let shared = Arc::new(Mutex::new(DaprState {
        conflicts_left: conflicts,
        ..Default::default()
    }));
    let revision = Arc::new(Mutex::new(0u64));
    let state = shared.clone();
    let base = serve(Arc::new(move |req: &Request| {
        let path = req.path.split('?').next().unwrap_or_default().to_string();
        let Some(rest) = path.strip_prefix("/v1.0/state/statestore") else {
            return Reply::empty(404);
        };
        let key = rest.trim_start_matches('/').to_string();
        let mut state = state.lock().unwrap();
        state.requests.push(req.clone());
        match req.method.as_str() {
            "GET" => match state.store.get(&key) {
                None => Reply::empty(204),
                Some((value, etag)) => Reply {
                    status: 200,
                    headers: vec![
                        ("content-type".into(), "application/json".into()),
                        ("etag".into(), etag.clone()),
                    ],
                    body: value.clone(),
                },
            },
            "POST" => {
                let entry = &req.body[0];
                let entry_key = entry["key"].as_str().unwrap_or_default().to_string();
                let sent_etag = entry["etag"].as_str().map(str::to_string);
                let current = state.store.get(&entry_key).map(|(_, e)| e.clone());
                let seeded_conflict = sent_etag.is_some()
                    && entry_key.ends_with(":messages")
                    && state.conflicts_left > 0;
                if seeded_conflict {
                    state.conflicts_left -= 1;
                }
                if seeded_conflict || (sent_etag.is_some() && sent_etag != current) {
                    return Reply {
                        status: 409,
                        headers: Vec::new(),
                        body: json!({"errorCode": "ERR_STATE_SAVE"}).to_string(),
                    };
                }
                let mut revision = revision.lock().unwrap();
                *revision += 1;
                state.store.insert(
                    entry_key,
                    (entry["value"].to_string(), revision.to_string()),
                );
                Reply::empty(204)
            }
            "DELETE" => {
                state.store.remove(&key);
                Reply::empty(204)
            }
            _ => Reply::empty(405),
        }
    }))
    .await;
    (base, shared)
}

#[tokio::test]
async fn dapr_session_round_trips_against_a_mock_sidecar() {
    let (base, state) = mock_dapr(1).await;
    let session = DaprSession::new("user-123", "statestore", &base)
        .with_consistency(DaprConsistency::Strong)
        .with_ttl(600);

    assert_eq!(session.messages_key(), "user-123:messages");
    assert_eq!(session.metadata_key(), "user-123:metadata");
    assert!(session.get_items(None).await.unwrap().is_empty());

    session
        .add_items(vec![InputItem::user("hello"), InputItem::assistant("hi")])
        .await
        .unwrap();
    // The second append hits the seeded 409 and succeeds on retry.
    session
        .add_items(vec![InputItem::user("again")])
        .await
        .unwrap();

    let items = session.get_items(None).await.unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].text_content().as_deref(), Some("hello"));
    assert_eq!(items[2].text_content().as_deref(), Some("again"));
    assert_eq!(session.get_items(Some(2)).await.unwrap().len(), 2);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());

    {
        let state = state.lock().unwrap();
        assert_eq!(
            state.conflicts_left, 0,
            "the seeded conflict was not served"
        );

        // The stored value is a JSON array of serialized items, as Python writes it.
        let (value, _) = state.store.get("user-123:messages").unwrap();
        let parsed: Value = serde_json::from_str(value).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
        let metadata: Value =
            serde_json::from_str(&state.store.get("user-123:metadata").unwrap().0).unwrap();
        assert_eq!(metadata["session_id"], "user-123");
        assert!(metadata["created_at"].as_str().is_some());

        // Reads carry the consistency level; writes carry it plus the TTL and, once an ETag
        // has been read, first-write concurrency.
        let gets: Vec<&Request> = state
            .requests
            .iter()
            .filter(|r| r.method == "GET")
            .collect();
        assert!(!gets.is_empty());
        assert!(gets.iter().all(|r| r.path.contains("consistency=strong")));

        let posts: Vec<&Request> = state
            .requests
            .iter()
            .filter(|r| r.method == "POST")
            .collect();
        assert!(posts.len() >= 3);
        for post in &posts {
            assert_eq!(post.path, "/v1.0/state/statestore");
            let entry = &post.body[0];
            assert_eq!(entry["metadata"]["ttlInSeconds"], "600");
            assert_eq!(entry["options"]["consistency"], "strong");
        }
        let conditional: Vec<&&Request> = posts
            .iter()
            .filter(|p| p.body[0]["etag"].is_string())
            .collect();
        assert!(conditional.len() >= 2);
        assert!(
            conditional
                .iter()
                .all(|p| p.body[0]["options"]["concurrency"] == "first-write")
        );
    }

    assert_eq!(
        session
            .pop_item()
            .await
            .unwrap()
            .unwrap()
            .text_content()
            .as_deref(),
        Some("again")
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);

    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(state.lock().unwrap().store.is_empty());
    assert!(session.pop_item().await.unwrap().is_none());
}

/// A session that appends an extra item the moment compaction reads the history, standing in
/// for another process writing to the same store.
#[derive(Debug)]
struct RacingSession {
    inner: Arc<InMemorySession>,
    inject_on_read: Mutex<Option<InputItem>>,
}

#[async_trait::async_trait]
impl Session for RacingSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    async fn get_items(&self, limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        let items = self.inner.get_items(limit).await?;
        // The first read is the compaction snapshot; append behind its back afterwards.
        let extra = self.inject_on_read.lock().unwrap().take();
        if let Some(extra) = extra {
            self.inner.add_items(vec![extra]).await?;
        }
        Ok(items)
    }

    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        self.inner.add_items(items).await
    }

    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        self.inner.clear_session().await
    }
}

#[tokio::test]
async fn compaction_retains_history_when_another_writer_appends_mid_request() {
    let (base, captured) = mock_openai(vec![compaction_reply("Summary")]).await;
    let inner = Arc::new(InMemorySession::new("s1"));
    inner.add_items(long_history()).await.unwrap();
    let racing = Arc::new(RacingSession {
        inner: inner.clone(),
        inject_on_read: Mutex::new(Some(InputItem::user("snuck in"))),
    });

    let session = OpenAIResponsesCompactionSession::new("s1", racing)
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));

    let usage = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp_1".into()),
            store: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();

    // The request was made and billed, but the replacement was skipped because the stored
    // history no longer matched the snapshot, so the concurrent append is not lost.
    assert_eq!(captured.lock().unwrap().len(), 1);
    assert_eq!(usage.requests, 1);
    let items = inner.get_items(None).await.unwrap();
    assert_eq!(items.len(), long_history().len() + 1);
    assert_eq!(
        items.last().unwrap().text_content().as_deref(),
        Some("snuck in")
    );
}

#[tokio::test]
async fn compaction_skips_without_a_response_id_and_defers_for_local_tool_outputs() {
    let (base, captured) = mock_openai(vec![compaction_reply("Recap")]).await;
    let inner = Arc::new(InMemorySession::new("s1"));
    inner.add_items(long_history()).await.unwrap();
    let session = OpenAIResponsesCompactionSession::new("s1", inner.clone())
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));

    // No response id: there is nothing to compact against, so no request is sent.
    let usage = session
        .on_run_completed(&SessionRunCompletion {
            response_id: None,
            store: None,
            has_local_tool_outputs: false,
        })
        .await
        .unwrap();
    assert_eq!(usage, Usage::default());
    assert!(captured.lock().unwrap().is_empty());

    // Local tool outputs defer the compaction instead of running it.
    let usage = session
        .on_run_completed(&SessionRunCompletion {
            response_id: Some("resp_1".into()),
            store: Some(true),
            has_local_tool_outputs: true,
        })
        .await
        .unwrap();
    assert_eq!(usage, Usage::default());
    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(inner.get_items(None).await.unwrap().len(), 13);

    // The next settled turn forces the deferred compaction even though the decision hook now
    // declines; the stored history is unchanged, so only the deferral can explain the call.
    let session = session.with_should_trigger_compaction(|_| false);
    let usage = session
        .on_run_completed(&SessionRunCompletion {
            response_id: Some("resp_2".into()),
            store: Some(true),
            has_local_tool_outputs: false,
        })
        .await
        .unwrap();
    assert_eq!(usage.requests, 1);
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].body["previous_response_id"], "resp_2");
    assert_eq!(inner.get_items(None).await.unwrap().len(), 1);
}

/// An envelope and master key produced by the Python SDK's `EncryptedSession` with
/// `cryptography` 48, pinning cross-language compatibility of the stored history.
#[cfg(feature = "encrypt")]
#[tokio::test]
async fn encrypted_session_reads_history_written_by_the_python_sdk() {
    const PYTHON_PAYLOAD: &str = "gAAAAABlU_EAyicwKYzApg0fhqXu4Mf3MVoIR0uJ44-qrbeqamzFwpJDScnj0z_en8pDpJuSDaVsUBtisfKKy0CLq497sBTG0Y4xoA_8WDOETvmk20rM7SUC2AHvNPk5Z7cctoi6GNrROOABA0B_fT-L9HlY_1DTjA==";
    const CREATED_AT: u64 = 1_700_000_000;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let inner = Arc::new(InMemorySession::new("user-123"));
    inner
        .add_items(vec![InputItem(json!({
            "__enc__": 1,
            "v": 1,
            "kid": "hkdf-v1",
            "payload": PYTHON_PAYLOAD,
        }))])
        .await
        .unwrap();

    // The vector was minted in 2023, so the TTL has to still cover it.
    let session =
        EncryptedSession::new("user-123", inner.clone(), TEST_KEY, now - CREATED_AT + 600).unwrap();
    let items = session.get_items(None).await.unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].role(), Some("user"));
    assert_eq!(
        items[0].text_content().as_deref(),
        Some("hello from python (raw)")
    );

    // Appending through the Rust session leaves a history Python can still read in full: both
    // envelopes carry Fernet tokens under the same key.
    session
        .add_items(vec![InputItem::assistant("hello from rust")])
        .await
        .unwrap();
    let raw = inner.get_items(None).await.unwrap();
    assert_eq!(raw.len(), 2);
    for envelope in &raw {
        let v = envelope.as_value();
        assert_eq!(v["__enc__"], 1);
        assert_eq!(v["v"], 1);
        assert_eq!(v["kid"], "hkdf-v1");
        let payload = v["payload"].as_str().unwrap();
        assert!(payload.starts_with("gAAAAA"), "{payload}");
        assert!(session.envelope_timestamp(envelope).is_some());
    }
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);
}
