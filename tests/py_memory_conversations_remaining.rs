//! Remaining parity ports of `tests/memory/test_openai_conversations_session.py`.
//!
//! Python patches the OpenAI SDK; Rust's client is plain HTTP, so these tests run against an
//! async mock Conversations endpoint that can hold, fail or drop individual requests.
//!
//! Python-only here: asyncio cancellation settling (`clear_session` / batch cancellation keep
//! running to completion, repeated `task.cancel(..)` messages), the SDK-level retry layer
//! (`max_retries`, `with_options`), constructor kwargs and `start_openai_conversations_session`.
//! In Rust, dropping a future simply stops it; the tests below pin that behaviour instead.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::memory::{OpenAIConversationsSession, Session};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text};

// ---------------------------------------------------------------------------
// Async mock endpoint
// ---------------------------------------------------------------------------

enum Action {
    Reply(Reply),
    /// Close the connection without answering (a lost response).
    Drop,
}

type Handler = Arc<dyn Fn(Request) -> Pin<Box<dyn Future<Output = Action> + Send>> + Send + Sync>;

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
        let head = &text[..idx];
        let mut headers = Vec::new();
        for line in head.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_lowercase(), v.trim().to_string()));
            }
        }
        let len: usize = headers
            .iter()
            .find(|(k, _)| k == "content-length")
            .map(|(_, v)| v.parse().unwrap_or(0))
            .unwrap_or(0);
        if buf.len() < idx + 4 + len {
            continue;
        }
        let mut parts = head.split_whitespace();
        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();
        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null);
        return Some(Request {
            method,
            path,
            headers,
            body,
        });
    }
}

async fn serve_async(handler: Handler) -> OpenAIClient {
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
                match handler(request).await {
                    Action::Drop => {
                        let _ = sock.shutdown().await;
                    }
                    Action::Reply(reply) => {
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
                    }
                }
            });
        }
    });
    OpenAIClient::new("sk-test").with_base_url(format!("http://{addr}/v1"))
}

fn ok_page() -> Reply {
    Reply::json(json!({"object": "list", "data": [], "has_more": false}))
}

fn texts(batch: &Value) -> Vec<String> {
    batch
        .as_array()
        .map(|a| {
            a.iter()
                .map(|i| i["content"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn many(count: usize) -> Vec<InputItem> {
    (0..count)
        .map(|i| InputItem::user(format!("message {i}")))
        .collect()
}

fn is_item_post(req: &Request) -> bool {
    req.method == "POST" && req.path.ends_with("/items")
}

/// A stateful in-memory Conversations backend. The first item batch that brings the store to
/// exactly 20 items parks until `release` fires.
struct Backend {
    stored: Mutex<Vec<Value>>,
    hold_first_batch: bool,
    first_started: Notify,
    release: Notify,
    requests: Mutex<Vec<Request>>,
}

impl Backend {
    fn new(hold_first_batch: bool) -> Arc<Self> {
        Arc::new(Backend {
            stored: Mutex::new(Vec::new()),
            hold_first_batch,
            first_started: Notify::new(),
            release: Notify::new(),
            requests: Mutex::new(Vec::new()),
        })
    }

    async fn client(self: &Arc<Self>) -> OpenAIClient {
        let this = self.clone();
        serve_async(Arc::new(move |req| {
            let this = this.clone();
            Box::pin(async move { this.handle(req).await })
        }))
        .await
    }

    async fn handle(&self, req: Request) -> Action {
        self.requests.lock().unwrap().push(req.clone());
        if req.method == "POST" && req.path.ends_with("/conversations") {
            return Action::Reply(Reply::json(json!({"id": "conv_new"})));
        }
        if is_item_post(&req) {
            let reached_twenty = {
                let mut stored = self.stored.lock().unwrap();
                for item in req.body["items"].as_array().cloned().unwrap_or_default() {
                    let mut item = item;
                    item["id"] = json!(format!("item_{}", stored.len()));
                    stored.push(item);
                }
                stored.len() == 20
            };
            if self.hold_first_batch && reached_twenty {
                self.first_started.notify_one();
                self.release.notified().await;
            }
            return Action::Reply(ok_page());
        }
        if req.method == "DELETE" && req.path.contains("/items/") {
            self.stored.lock().unwrap().pop();
            return Action::Reply(Reply::json(json!({"deleted": true})));
        }
        if req.method == "DELETE" {
            self.stored.lock().unwrap().clear();
            return Action::Reply(Reply::json(json!({"deleted": true})));
        }
        // GET items.
        let stored = self.stored.lock().unwrap().clone();
        let data: Vec<Value> = if req.path.contains("order=desc") {
            stored.into_iter().rev().collect()
        } else {
            stored
        };
        Action::Reply(Reply::json(
            json!({"object": "list", "data": data, "has_more": false}),
        ))
    }
}

// ---------------------------------------------------------------------------
// test_operations_wait_for_complete_append (add / pop / clear / read)
// ---------------------------------------------------------------------------

async fn operations_wait_for(mutation: &str) {
    let backend = Backend::new(true);
    let client = backend.client().await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client,
    ));

    let append = {
        let session = session.clone();
        tokio::spawn(async move { session.add_items(many(21)).await })
    };
    tokio::time::timeout(Duration::from_secs(5), backend.first_started.notified())
        .await
        .expect("the first batch of 20 must reach the server");

    let follower = {
        let session = session.clone();
        let mutation = mutation.to_string();
        tokio::spawn(async move {
            match mutation.as_str() {
                "add" => {
                    session
                        .add_items(vec![InputItem::user("other")])
                        .await
                        .unwrap();
                    None
                }
                "pop" => session.pop_item().await.unwrap(),
                "read" => Some(InputItem::from_value(json!(item_texts(
                    &session.get_items(None).await.unwrap()
                )))),
                _ => {
                    session.clear_session().await.unwrap();
                    None
                }
            }
        })
    };
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !follower.is_finished(),
        "{mutation}: must wait for the append"
    );
    assert_eq!(backend.stored.lock().unwrap().len(), 20, "{mutation}");

    backend.release.notify_one();
    append.await.unwrap().unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), follower)
        .await
        .unwrap()
        .unwrap();

    let stored = backend.stored.lock().unwrap().clone();
    let stored_texts: Vec<String> = stored
        .iter()
        .map(|i| i["content"].as_str().unwrap_or_default().to_string())
        .collect();
    let expected: Vec<String> = (0..21).map(|i| format!("message {i}")).collect();
    match mutation {
        "add" => {
            let mut with_other = expected.clone();
            with_other.push("other".into());
            assert_eq!(stored_texts, with_other);
        }
        "read" => {
            let read: Vec<String> =
                serde_json::from_value(result.unwrap().as_value().clone()).unwrap();
            assert_eq!(read, expected);
            assert_eq!(stored_texts, expected);
        }
        "pop" => {
            assert_eq!(item_text(&result.unwrap()), "message 20");
            assert_eq!(stored_texts, expected[..20].to_vec());
        }
        _ => {
            assert!(stored_texts.is_empty());
            assert!(session.try_session_id().is_err());
        }
    }
}

fn item_texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_add_waits_for_a_complete_append() {
    operations_wait_for("add").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pop_waits_for_a_complete_append() {
    operations_wait_for("pop").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clear_waits_for_a_complete_append() {
    operations_wait_for("clear").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_read_waits_for_a_complete_append() {
    operations_wait_for("read").await;
}

// ---------------------------------------------------------------------------
// test_request_sizes_and_order
// ---------------------------------------------------------------------------

async fn request_sizes(count: usize, expected: Vec<usize>) {
    let backend = Backend::new(false);
    let client = backend.client().await;
    let session = OpenAIConversationsSession::from_conversation_id("conv_test", client);
    session.add_items(many(count)).await.unwrap();

    let requests = backend.requests.lock().unwrap().clone();
    assert!(
        requests
            .iter()
            .all(|r| r.path == "/v1/conversations/conv_test/items")
    );
    let sizes: Vec<usize> = requests
        .iter()
        .map(|r| r.body["items"].as_array().map(Vec::len).unwrap_or(0))
        .collect();
    assert_eq!(sizes, expected);
    let sent: Vec<String> = requests
        .iter()
        .flat_map(|r| texts(&r.body["items"]))
        .collect();
    let want: Vec<String> = (0..count).map(|i| format!("message {i}")).collect();
    assert_eq!(sent, want);
}

#[tokio::test]
async fn request_sizes_for_zero_items() {
    request_sizes(0, vec![]).await;
}

#[tokio::test]
async fn request_sizes_for_exactly_one_batch() {
    request_sizes(20, vec![20]).await;
}

#[tokio::test]
async fn request_sizes_for_one_over_a_batch() {
    request_sizes(21, vec![20, 1]).await;
}

#[tokio::test]
async fn request_sizes_for_two_full_batches_and_one() {
    request_sizes(41, vec![20, 20, 1]).await;
}

// ---------------------------------------------------------------------------
// test_later_failure_preserves_prefix_and_stops
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_later_failure_keeps_the_prefix_and_a_follow_up_write_succeeds() {
    let batches: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let saved: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (b, s) = (batches.clone(), saved.clone());
    let client = serve_async(Arc::new(move |req| {
        let (b, s) = (b.clone(), s.clone());
        Box::pin(async move {
            let batch = texts(&req.body["items"]);
            let mut batches = b.lock().unwrap();
            batches.push(batch.clone());
            if batches.len() == 2 {
                return Action::Reply(Reply::status(
                    400,
                    "{\"error\": {\"message\": \"synthetic failure\"}}",
                ));
            }
            s.lock().unwrap().extend(batch);
            Action::Reply(ok_page())
        })
    }))
    .await;
    let session = OpenAIConversationsSession::from_conversation_id("conv_test", client);
    let items = many(41);

    let error = session
        .add_items(items.clone())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("synthetic failure"), "{error}");
    session.add_items(items[40..].to_vec()).await.unwrap();

    let all: Vec<String> = (0..41).map(|i| format!("message {i}")).collect();
    assert_eq!(
        *batches.lock().unwrap(),
        vec![all[..20].to_vec(), all[20..40].to_vec(), all[40..].to_vec()]
    );
    let mut expected = all[..20].to_vec();
    expected.push("message 40".into());
    assert_eq!(*saved.lock().unwrap(), expected);
}

// ---------------------------------------------------------------------------
// test_chunk_lost_response_is_not_retried (the Rust client has no retry layer, so the
// parity claim is that a lost response is surfaced and never re-sent)
// ---------------------------------------------------------------------------

async fn lost_response(lost_batch: usize) {
    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let client = serve_async(Arc::new(move |req| {
        let log = log.clone();
        Box::pin(async move {
            let mut seen = log.lock().unwrap();
            seen.push(texts(&req.body["items"]));
            if seen.len() == lost_batch {
                Action::Drop
            } else {
                Action::Reply(ok_page())
            }
        })
    }))
    .await;
    let session = OpenAIConversationsSession::from_conversation_id("conv_test", client);

    assert!(session.add_items(many(41)).await.is_err());
    assert_eq!(
        seen.lock().unwrap().len(),
        lost_batch,
        "no batch is re-sent or continued"
    );
}

#[tokio::test]
async fn a_lost_first_response_is_not_retried() {
    lost_response(1).await;
}

#[tokio::test]
async fn a_lost_second_response_is_not_retried() {
    lost_response(2).await;
}

// ---------------------------------------------------------------------------
// Dropped callers (the Rust counterpart of the asyncio cancellation tests)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_writer_stops_sending_chunks_and_releases_the_lock() {
    let backend = Backend::new(true);
    let client = backend.client().await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client,
    ));

    let write = {
        let session = session.clone();
        tokio::spawn(async move { session.add_items(many(41)).await })
    };
    tokio::time::timeout(Duration::from_secs(5), backend.first_started.notified())
        .await
        .unwrap();
    write.abort();
    let _ = write.await;
    backend.release.notify_one();

    // The lock is free again, and the batches after the dropped one were never sent.
    tokio::time::timeout(
        Duration::from_secs(5),
        session.add_items(vec![InputItem::user("survivor")]),
    )
    .await
    .expect("the lock must be released")
    .unwrap();
    let sizes: Vec<usize> = backend
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|r| is_item_post(r))
        .map(|r| r.body["items"].as_array().map(Vec::len).unwrap_or(0))
        .collect();
    assert_eq!(sizes, vec![20, 1], "{sizes:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_single_request_caller_does_not_wait_for_the_response() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let (s, r) = (started.clone(), release.clone());
    let client = serve_async(Arc::new(move |_req| {
        let (s, r) = (s.clone(), r.clone());
        Box::pin(async move {
            s.notify_one();
            r.notified().await;
            Action::Reply(ok_page())
        })
    }))
    .await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client,
    ));

    let write = {
        let session = session.clone();
        tokio::spawn(async move { session.add_items(many(20)).await })
    };
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    write.abort();
    let outcome = tokio::time::timeout(Duration::from_secs(5), write)
        .await
        .unwrap();
    assert!(outcome.unwrap_err().is_cancelled());
    release.notify_one();
}

// ---------------------------------------------------------------------------
// clear_session vs a concurrent conversation-id lookup
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_concurrent_id_lookup_during_clear_does_not_clobber_the_new_id() {
    let delete_started = Arc::new(Notify::new());
    let allow_delete = Arc::new(Notify::new());
    let creates = Arc::new(Mutex::new(0usize));
    let (ds, ad, c) = (
        delete_started.clone(),
        allow_delete.clone(),
        creates.clone(),
    );
    let client = serve_async(Arc::new(move |req| {
        let (ds, ad, c) = (ds.clone(), ad.clone(), c.clone());
        Box::pin(async move {
            if req.method == "DELETE" {
                ds.notify_one();
                ad.notified().await;
                return Action::Reply(Reply::json(json!({"deleted": true})));
            }
            if req.method == "POST" && req.path.ends_with("/conversations") {
                *c.lock().unwrap() += 1;
                return Action::Reply(Reply::json(json!({"id": "new_id"})));
            }
            Action::Reply(ok_page())
        })
    }))
    .await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "old_id", client,
    ));

    let clear = {
        let session = session.clone();
        tokio::spawn(async move { session.clear_session().await })
    };
    tokio::time::timeout(Duration::from_secs(5), delete_started.notified())
        .await
        .unwrap();
    let lookup = {
        let session = session.clone();
        tokio::spawn(async move { session.conversation_id().await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    allow_delete.notify_one();

    clear.await.unwrap().unwrap();
    let new_id = lookup.await.unwrap().unwrap();
    assert_eq!(new_id, "new_id");
    assert_eq!(session.try_session_id().unwrap(), "new_id");
    assert_eq!(*creates.lock().unwrap(), 1);
}

// ---------------------------------------------------------------------------
// Lazy creation and creation-failure recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_conversation_id_is_created_once_across_lookups() {
    let backend = Backend::new(false);
    let client = backend.client().await;
    let session = OpenAIConversationsSession::with_client(client);

    let a = session.conversation_id().await.unwrap();
    let b = session.conversation_id().await.unwrap();
    let c = session.conversation_id().await.unwrap();
    assert_eq!(
        (a.as_str(), b.as_str(), c.as_str()),
        ("conv_new", "conv_new", "conv_new")
    );
    let creates = backend
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/conversations"))
        .count();
    assert_eq!(creates, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiting_writer_recovers_when_the_first_creation_fails() {
    let first_started = Arc::new(Notify::new());
    let release_first = Arc::new(Notify::new());
    let creations = Arc::new(Mutex::new(0usize));
    let item_writes: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (fs, rf, cr, iw) = (
        first_started.clone(),
        release_first.clone(),
        creations.clone(),
        item_writes.clone(),
    );
    let client = serve_async(Arc::new(move |req| {
        let (fs, rf, cr, iw) = (fs.clone(), rf.clone(), cr.clone(), iw.clone());
        Box::pin(async move {
            if req.method == "POST" && req.path.ends_with("/conversations") {
                let n = {
                    let mut n = cr.lock().unwrap();
                    *n += 1;
                    *n
                };
                if n == 1 {
                    fs.notify_one();
                    rf.notified().await;
                    return Action::Reply(Reply::status(
                        500,
                        "{\"error\": {\"message\": \"Conversation creation failed\"}}",
                    ));
                }
                return Action::Reply(Reply::json(json!({"id": "surviving_conversation"})));
            }
            iw.lock().unwrap().push(req.path.clone());
            Action::Reply(ok_page())
        })
    }))
    .await;
    let session = Arc::new(OpenAIConversationsSession::with_client(client));

    let failed = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .add_items(vec![InputItem::user("Failed writer")])
                .await
        })
    };
    tokio::time::timeout(Duration::from_secs(5), first_started.notified())
        .await
        .unwrap();
    let surviving = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .add_items(vec![InputItem::user("Surviving writer")])
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        *creations.lock().unwrap(),
        1,
        "the second writer waits for the first"
    );
    release_first.notify_one();

    assert!(failed.await.unwrap().is_err());
    surviving.await.unwrap().unwrap();

    assert_eq!(*creations.lock().unwrap(), 2);
    assert_eq!(
        *item_writes.lock().unwrap(),
        vec!["/v1/conversations/surviving_conversation/items".to_string()]
    );
    assert_eq!(session.try_session_id().unwrap(), "surviving_conversation");
}

// ---------------------------------------------------------------------------
// Runner integration
// ---------------------------------------------------------------------------

async fn runner_batches_input(stream: bool, exact_split: bool) {
    use futures::StreamExt;
    openai_agents_sdk::set_tracing_disabled(true);
    let backend = Backend::new(false);
    let client = backend.client().await;
    let session = Arc::new(OpenAIConversationsSession::with_client(client));

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let input = many(41);

    if stream {
        let mut result = Runner::run_streamed_with_options(
            &agent,
            input,
            RunOptions::new(()).session(session.clone()),
        );
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
    } else {
        let result =
            Runner::run_with_options(&agent, input, RunOptions::new(()).session(session.clone()))
                .await
                .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
    }

    let requests = backend.requests.lock().unwrap().clone();
    let creates = requests
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/conversations"))
        .count();
    assert_eq!(creates, 1, "the conversation is created lazily, once");
    let sizes: Vec<usize> = requests
        .iter()
        .filter(|r| is_item_post(r))
        .map(|r| r.body["items"].as_array().map(Vec::len).unwrap_or(0))
        .collect();
    assert!(sizes.iter().all(|n| *n <= 20), "{sizes:?}");
    let written: Vec<String> = requests
        .iter()
        .filter(|r| is_item_post(r))
        .flat_map(|r| texts(&r.body["items"]))
        .take(41)
        .collect();
    let want: Vec<String> = (0..41).map(|i| format!("message {i}")).collect();
    assert_eq!(written, want);
    assert_eq!(
        sizes.iter().sum::<usize>(),
        42,
        "41 inputs plus the reply: {sizes:?}"
    );
    if exact_split {
        // Python persists the input before the model call and the reply afterwards.
        assert_eq!(sizes, vec![20, 20, 1, 1], "{sizes:?}");
    }
}

#[tokio::test]
async fn the_runner_batches_input_with_lazy_creation() {
    runner_batches_input(false, false).await;
}

#[tokio::test]
async fn the_streamed_runner_batches_input_with_lazy_creation() {
    runner_batches_input(true, false).await;
}

/// Cross-area: `src/run_internal/mod.rs` (`save_session`, ~:1536) appends the input and the
/// model output in one call, so 41 inputs + 1 reply is `[20, 20, 2]`. Python saves the input
/// first and the reply afterwards (`[20, 20, 1, 1]`), so a failing model call still leaves the
/// input stored.
#[ignore = "cross-area: run_internal persists input and output in a single append"]
#[tokio::test]
async fn the_runner_persists_input_before_the_reply() {
    runner_batches_input(false, true).await;
}

#[tokio::test]
async fn the_runner_sends_the_stored_conversation_history_to_the_model() {
    openai_agents_sdk::set_tracing_disabled(true);
    let page = json!({
        "object": "list",
        "data": [
            {"id": "m1", "role": "user", "content": "What city is the Golden Gate Bridge in?"},
            {"id": "m2", "role": "assistant", "content": "San Francisco"},
        ],
        "has_more": false,
    });
    let client = serve_async(Arc::new(move |req| {
        let page = page.clone();
        Box::pin(async move {
            if req.method == "GET" {
                Action::Reply(Reply::json(page))
            } else {
                Action::Reply(ok_page())
            }
        })
    }))
    .await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_hist",
        client,
    ));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("California")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    let result = Runner::run_with_options(
        &agent,
        "What state is it in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("California"));

    let input = model.last_call().unwrap().input;
    assert!(input.len() > 1, "{input:?}");
    assert!(
        input
            .iter()
            .any(|i| item_text(i).contains("Golden Gate Bridge")),
        "{input:?}"
    );
}

#[tokio::test]
async fn an_item_without_an_id_cannot_be_popped() {
    // Python indexes `item["id"]` and raises KeyError; the item stays in the conversation.
    // Rust (`OpenAIConversationsSession::pop_item`) instead returns the item and skips the
    // delete, so it is reported as popped while still stored. Fixing it means changing
    // `an_item_without_an_id_is_returned_without_a_delete` in py_memory_conversations.rs,
    // which this fork may not edit; see the report.
    let client = serve_async(Arc::new(|req| {
        Box::pin(async move {
            if req.method == "GET" {
                Action::Reply(Reply::json(json!({
                    "data": [{"role": "assistant", "content": "No ID"}], "has_more": false
                })))
            } else {
                Action::Reply(ok_page())
            }
        })
    }))
    .await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);
    assert!(session.pop_item().await.is_err());
}
