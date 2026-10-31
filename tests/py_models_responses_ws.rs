//! Port of the WebSocket cluster of `tests/models/test_openai_responses.py` and of
//! `tests/models/test_responses_websocket_session.py`.
//!
//! Python swaps `_open_websocket_connection` for a scripted fake. Rust has no such seam, so
//! each test runs a real local WebSocket server whose per-connection behaviour is scripted
//! with [`Step`]s (and [`Conn`] for handshake-level misbehaviour).
#![cfg(feature = "websocket")]
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{ModelRequest, ModelTracing};
use openai_agents_sdk::models::multi_provider::PrefixMode;
use openai_agents_sdk::models::openai_responses_ws::{
    OpenAIResponsesWSModel, OpenAIResponsesWebSocketOptions, ResponsesWebSocketSession,
    WS_CLOSE_INVALIDATED_MESSAGE, WS_PRE_EVENT_DISCONNECT_MESSAGE,
};
use openai_agents_sdk::models::{Model, OpenAIClient, websocket_pre_event_retries_disabled};
use openai_agents_sdk::retry::{ModelRetryAdviceRequest, ReplaySafety};
use openai_agents_sdk::{Agent, AgentsError, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

// ------------------------------------------------------------------------------ harness

/// What the server does on one connection, in order.
#[derive(Clone, Debug)]
enum Step {
    /// Wait for one client text frame and record it.
    Recv,
    /// Send one JSON text frame.
    Send(Value),
    /// Pause for this many milliseconds.
    Sleep(u64),
    /// Send a close frame and end the connection.
    Close,
    /// Drop the TCP connection with no close frame.
    Drop,
}

#[derive(Clone, Debug)]
enum Conn {
    /// Complete the handshake, then run the steps.
    Ws(Vec<Step>),
    /// Pause before answering the handshake.
    SlowWs(u64, Vec<Step>),
    /// Accept the TCP connection and close it before the handshake.
    DropAtAccept,
    /// Answer the handshake request with bytes that are not HTTP.
    Garbage,
    /// Accept the TCP connection and never answer.
    Stall,
}

#[derive(Clone, Debug, Default)]
struct Handshake {
    /// Request target, e.g. `/v1/responses?token=abc`.
    uri: String,
    /// Lowercased header name -> values.
    headers: HashMap<String, Vec<String>>,
}

impl Handshake {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .and_then(|v| v.first())
            .map(String::as_str)
    }

    fn path(&self) -> &str {
        self.uri.split('?').next().unwrap_or_default()
    }

    fn query(&self) -> HashMap<String, String> {
        self.uri
            .split_once('?')
            .map(|(_, q)| {
                q.split('&')
                    .filter_map(|kv| kv.split_once('='))
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

struct Server {
    /// `http://127.0.0.1:port/v1`, for `OpenAIClient::with_base_url`.
    base_url: String,
    port: u16,
    accepted: Arc<AtomicUsize>,
    /// (connection index, frame).
    frames: Arc<Mutex<Vec<(usize, Value)>>>,
    handshakes: Arc<Mutex<Vec<Handshake>>>,
    /// Connection indexes the server saw end.
    ended: Arc<Mutex<Vec<usize>>>,
}

impl Server {
    fn frames(&self) -> Vec<Value> {
        self.frames
            .lock()
            .unwrap()
            .iter()
            .map(|f| f.1.clone())
            .collect()
    }
    fn handshakes(&self) -> Vec<Handshake> {
        self.handshakes.lock().unwrap().clone()
    }
    fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }
    fn ended(&self) -> Vec<usize> {
        self.ended.lock().unwrap().clone()
    }
    fn model(&self) -> OpenAIResponsesWSModel {
        OpenAIResponsesWSModel::new("gpt-4", self.client())
    }
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("sk-test").with_base_url(self.base_url.clone())
    }
}

async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..400 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn spawn_server(conns: Vec<Conn>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = Server {
        base_url: format!("http://127.0.0.1:{port}/v1"),
        port,
        accepted: Arc::default(),
        frames: Arc::default(),
        handshakes: Arc::default(),
        ended: Arc::default(),
    };
    let (accepted, frames, handshakes, ended) = (
        server.accepted.clone(),
        server.frames.clone(),
        server.handshakes.clone(),
        server.ended.clone(),
    );
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let idx = accepted.fetch_add(1, Ordering::SeqCst);
            let script = conns.get(idx).cloned().unwrap_or(Conn::DropAtAccept);
            let (frames, handshakes, ended) = (frames.clone(), handshakes.clone(), ended.clone());
            tokio::spawn(async move {
                let steps = match script {
                    Conn::DropAtAccept => {
                        drop(tcp);
                        ended.lock().unwrap().push(idx);
                        return;
                    }
                    Conn::Garbage => {
                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                        let mut tcp = tcp;
                        let mut buf = [0u8; 1024];
                        let _ = tcp.read(&mut buf).await;
                        let _ = tcp.write_all(b"this is not http\r\n\r\n").await;
                        let _ = tcp.shutdown().await;
                        ended.lock().unwrap().push(idx);
                        return;
                    }
                    Conn::Stall => {
                        use tokio::io::AsyncReadExt;
                        let mut tcp = tcp;
                        let mut buf = [0u8; 1024];
                        while let Ok(n) = tcp.read(&mut buf).await {
                            if n == 0 {
                                break;
                            }
                        }
                        ended.lock().unwrap().push(idx);
                        return;
                    }
                    Conn::SlowWs(ms, steps) => {
                        tokio::time::sleep(Duration::from_millis(ms)).await;
                        steps
                    }
                    Conn::Ws(steps) => steps,
                };
                let hs = handshakes.clone();
                let cb = move |req: &Request, resp: Response| {
                    let mut headers: HashMap<String, Vec<String>> = HashMap::new();
                    for (k, v) in req.headers() {
                        headers
                            .entry(k.as_str().to_ascii_lowercase())
                            .or_default()
                            .push(v.to_str().unwrap_or_default().to_string());
                    }
                    hs.lock().unwrap().push(Handshake {
                        uri: req.uri().to_string(),
                        headers,
                    });
                    Ok(resp)
                };
                let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(tcp, cb).await else {
                    ended.lock().unwrap().push(idx);
                    return;
                };
                for step in steps {
                    match step {
                        Step::Recv => loop {
                            match ws.next().await {
                                Some(Ok(Message::Text(t))) => {
                                    let v: Value = serde_json::from_str(&t).unwrap();
                                    frames.lock().unwrap().push((idx, v));
                                    break;
                                }
                                Some(Ok(_)) => continue,
                                _ => {
                                    ended.lock().unwrap().push(idx);
                                    return;
                                }
                            }
                        },
                        Step::Send(v) => {
                            if ws.send(Message::text(v.to_string())).await.is_err() {
                                ended.lock().unwrap().push(idx);
                                return;
                            }
                        }
                        Step::Sleep(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
                        Step::Close => {
                            let _ = ws.close(None).await;
                            drop(ws);
                            ended.lock().unwrap().push(idx);
                            return;
                        }
                        Step::Drop => {
                            drop(ws);
                            ended.lock().unwrap().push(idx);
                            return;
                        }
                    }
                }
                // Script finished: keep answering control frames until the client leaves.
                while let Some(Ok(_)) = ws.next().await {}
                ended.lock().unwrap().push(idx);
            });
        }
    });
    server
}

fn response_event(event_type: &str, id: &str, status: &str, seq: u64) -> Value {
    json!({"type": event_type, "sequence_number": seq, "response": {
        "id": id, "object": "response", "created_at": 0, "model": "fake",
        "status": status, "output": [], "tool_choice": "none", "tools": [],
        "parallel_tool_calls": false, "top_p": null, "usage": null}})
}

fn completed(id: &str) -> Value {
    response_event("response.completed", id, "completed", 1)
}

fn message_output(text: &str) -> Value {
    json!([{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]}])
}

fn completed_without_usage(output: Value) -> Value {
    let mut event = completed("resp-no-usage");
    event["response"]["output"] = output;
    event["response"]["usage"] = Value::Null;
    event
}

fn created(id: &str) -> Value {
    response_event("response.created", id, "in_progress", 1)
}

fn req(input: &str) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem::user(input)],
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

fn req_with(input: &str, settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        model_settings: settings,
        ..req(input)
    }
}

fn stateful(input: &str, previous: &str) -> ModelRequest {
    ModelRequest {
        previous_response_id: Some(previous.into()),
        ..req(input)
    }
}

fn timeout_settings(seconds: f64) -> ModelSettings {
    ModelSettings {
        timeout: Some(seconds),
        ..Default::default()
    }
}

async fn collect(
    model: &OpenAIResponsesWSModel,
    request: ModelRequest,
) -> (Vec<Value>, Option<AgentsError>) {
    let mut events = Vec::new();
    let mut stream = model.stream_response(request).await.unwrap();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => return (events, Some(error)),
        }
    }
    (events, None)
}

// ------------------------------------------------------------- connection and frames

#[tokio::test]
async fn reuses_connection_and_sends_response_create_frames() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-1")),
        Step::Recv,
        Step::Send(completed("resp-2")),
    ])])
    .await;
    let model = server.model();
    let settings = ModelSettings {
        reasoning: Some(Reasoning {
            mode: Some("pro".into()),
            effort: Some("max".into()),
            context: Some("all_turns".into()),
            ..Default::default()
        }),
        context_management: Some(vec![
            json!({"type": "compaction", "compact_threshold": 200000}),
        ]),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        prompt_cache_retention: Some("24h".into()),
        store: Some(false),
        metadata: Some(HashMap::from([(
            "request".to_string(),
            "example".to_string(),
        )])),
        ..Default::default()
    };
    let first = model.get_response(req_with("hi", settings)).await.unwrap();
    let second = model
        .get_response(stateful("next", "resp-1"))
        .await
        .unwrap();

    assert_eq!(first.response_id.as_deref(), Some("resp-1"));
    assert_eq!(second.response_id.as_deref(), Some("resp-2"));
    assert_eq!(server.accepted(), 1, "one connection serves both requests");
    let frames = server.frames();
    assert_eq!(frames[0]["type"], "response.create");
    assert_eq!(frames[0]["stream"], true);
    assert_eq!(
        frames[0]["reasoning"],
        json!({"context": "all_turns", "effort": "max", "mode": "pro"})
    );
    assert_eq!(
        frames[0]["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
    assert_eq!(
        frames[0]["context_management"],
        json!([{"type": "compaction", "compact_threshold": 200000}])
    );
    assert_eq!(frames[0]["prompt_cache_retention"], "24h");
    assert_eq!(frames[0]["store"], false);
    assert_eq!(frames[0]["metadata"], json!({"request": "example"}));
    assert_eq!(frames[1]["type"], "response.create");
    assert_eq!(frames[1]["stream"], true);
    assert_eq!(frames[1]["previous_response_id"], "resp-1");
}

#[tokio::test]
async fn message_size_limit_is_unbounded_by_default_and_configurable() {
    let big = "x".repeat(1_048_576);
    let reply = |text: &str| {
        let mut event = completed("resp-big");
        event["response"]["metadata"] = json!({"blob": text});
        event
    };
    // Default: no limit (tungstenite's own 64 MiB default is replaced by `None`).
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(reply(&big))])]).await;
    let response = server.model().get_response(req("hi")).await.unwrap();
    assert_eq!(response.response_id.as_deref(), Some("resp-big"));

    // An explicit limit is enforced.
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(reply(&big))])]).await;
    let model = server.model().with_websocket_options(
        OpenAIResponsesWebSocketOptions::default().with_max_size(Some(8 * 1024)),
    );
    assert_eq!(model.websocket_options().max_size, Some(8 * 1024));
    let error = model.get_response(req("hi")).await.unwrap_err();
    assert!(
        error.to_string().contains("Responses websocket"),
        "oversized frame must fail the request: {error}"
    );
}

#[tokio::test]
async fn stream_response_yields_typed_events() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-stream")),
    ])])
    .await;
    let (events, error) = collect(&server.model(), req("hi")).await;
    assert!(error.is_none());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "response.completed");
    assert_eq!(events[0]["response"]["id"], "resp-stream");
}

#[tokio::test]
async fn get_response_rejects_failed_terminal_response_payload_events() {
    for terminal in ["response.incomplete", "response.failed"] {
        let server = spawn_server(vec![Conn::Ws(vec![
            Step::Recv,
            Step::Send(response_event(terminal, "resp-terminal", "completed", 1)),
        ])])
        .await;
        let error = server.model().get_response(req("hi")).await.unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains(terminal), "{terminal}: {error}");
    }
}

#[tokio::test]
async fn stream_response_rejects_failed_terminal_response_payload_events() {
    for terminal in ["response.incomplete", "response.failed"] {
        let server = spawn_server(vec![Conn::Ws(vec![
            Step::Recv,
            Step::Send(response_event(terminal, "resp-terminal", "completed", 1)),
        ])])
        .await;
        let (events, error) = collect(&server.model(), req("hi")).await;
        // The terminal event reaches the consumer first, then the error is raised.
        assert_eq!(events.len(), 1, "{terminal}");
        assert_eq!(events[0]["type"], terminal);
        assert_eq!(events[0]["response"]["id"], "resp-terminal");
        let error = error.expect("error after the terminal event");
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains(terminal), "{terminal}: {error}");
    }
}

fn error_frame() -> Value {
    json!({"type": "response.error", "sequence_number": 1,
        "error": {"code": "invalid_request_error", "message": "bad request", "param": null}})
}

#[tokio::test]
async fn get_response_surfaces_response_error_event() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(error_frame())])]).await;
    let error = server.model().get_response(req("hi")).await.unwrap_err();
    let message = error.to_string();
    assert!(message.contains("response.error"), "{message}");
    assert!(message.contains("invalid_request_error"), "{message}");
    assert!(message.contains("bad request"), "{message}");
}

#[tokio::test]
async fn stream_response_raises_on_response_error_event() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(error_frame())])]).await;
    let (events, error) = collect(&server.model(), req("hi")).await;
    assert!(events.is_empty());
    let message = error.expect("error").to_string();
    assert!(message.contains("response.error"), "{message}");
    assert!(message.contains("invalid_request_error"), "{message}");
    assert!(message.contains("bad request"), "{message}");
}

#[tokio::test]
async fn preserves_pre_event_server_error_frame_message() {
    let frame = json!({"type": "error",
        "error": {"message": "bad auth", "type": "invalid_request_error"}});
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(frame)])]).await;
    let error = server.model().get_response(req("hi")).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.starts_with("Responses websocket error:"),
        "{message}"
    );
    assert!(!message.contains("feature may not be enabled"), "{message}");
    assert!(message.contains("invalid_request_error"), "{message}");
}

#[tokio::test]
async fn preserves_pre_event_user_error() {
    // A bad websocket URL is a configuration mistake: the user error is not wrapped.
    let model = OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test"))
        .with_websocket_base_url("not a url");
    let error = model.get_response(req("hi")).await.unwrap_err();
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("Invalid Responses websocket URL"),
        "{error}"
    );
}

// ------------------------------------------------ persistent connection lifecycle

#[tokio::test]
async fn dropping_a_stream_before_its_terminal_event_drops_the_connection() {
    let server = spawn_server(vec![
        Conn::Ws(vec![
            Step::Recv,
            Step::Send(created("resp-created")),
            Step::Sleep(100),
            Step::Send(completed("resp-complete")),
        ]),
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-next"))]),
    ])
    .await;
    let model = server.model();
    {
        let mut stream = model.stream_response(req("hi")).await.unwrap();
        let event = stream.next().await.unwrap().unwrap();
        assert_eq!(event["type"], "response.created");
        // Dropped here: unread frames are still queued, so the socket must not be reused.
    }
    // The request lock is free again and a fresh connection serves the next request.
    let next = timeout(Duration::from_secs(5), model.get_response(req("next")))
        .await
        .expect("the request lock was released")
        .unwrap();
    assert_eq!(next.response_id.as_deref(), Some("resp-next"));
    assert_eq!(
        server.accepted(),
        2,
        "the half-read connection is not reused"
    );
}

#[tokio::test]
async fn dropping_a_stream_after_its_terminal_event_keeps_the_connection() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-complete-1")),
        Step::Recv,
        Step::Send(completed("resp-complete-2")),
    ])])
    .await;
    let model = server.model();
    {
        let mut stream = model.stream_response(req("hi")).await.unwrap();
        let event = stream.next().await.unwrap().unwrap();
        assert_eq!(event["type"], "response.completed");
    }
    let second = model.get_response(req("next")).await.unwrap();
    assert_eq!(second.response_id.as_deref(), Some("resp-complete-2"));
    assert_eq!(server.accepted(), 1);
}

#[tokio::test]
async fn cancelling_a_request_drops_the_connection() {
    let server = spawn_server(vec![
        Conn::Ws(vec![Step::Recv, Step::Sleep(5_000)]),
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-after-cancel"))]),
    ])
    .await;
    let model = server.model();
    let cancelled = timeout(Duration::from_millis(150), model.get_response(req("hi"))).await;
    assert!(
        cancelled.is_err(),
        "the first request is still waiting for the server"
    );
    let next = model.get_response(req("next")).await.unwrap();
    assert_eq!(next.response_id.as_deref(), Some("resp-after-cancel"));
    assert_eq!(server.accepted(), 2);
}

#[tokio::test]
async fn reconnects_if_cached_connection_is_closed() {
    let server = spawn_server(vec![
        Conn::Ws(vec![
            Step::Recv,
            Step::Send(completed("resp-1")),
            Step::Close,
        ]),
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-2"))]),
    ])
    .await;
    let model = server.model();
    let first = model.get_response(req("hi")).await.unwrap();
    assert_eq!(first.response_id.as_deref(), Some("resp-1"));
    assert_eq!(server.accepted(), 1);
    // Let the server's close frame reach the cached socket.
    wait_until("server to end connection 0", || server.ended().contains(&0)).await;
    let second = model.get_response(req("next")).await.unwrap();
    assert_eq!(second.response_id.as_deref(), Some("resp-2"));
    assert_eq!(server.accepted(), 2);
}

#[tokio::test]
async fn retries_if_handshake_fails_before_request() {
    let server = spawn_server(vec![
        Conn::DropAtAccept,
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-retried"))]),
    ])
    .await;
    let response = server.model().get_response(req("hi")).await.unwrap();
    assert_eq!(response.response_id.as_deref(), Some("resp-retried"));
    assert_eq!(server.accepted(), 2);
    assert_eq!(server.frames().len(), 1);
}

#[tokio::test]
async fn does_not_retry_malformed_handshake() {
    let server = spawn_server(vec![Conn::Garbage]).await;
    let error = server.model().get_response(req("hi")).await.unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("Responses websocket connect failed"),
        "{error}"
    );
    assert_eq!(server.accepted(), 1);
}

#[tokio::test]
async fn exhausts_one_retry_after_repeated_handshake_disconnects() {
    let server = spawn_server(vec![
        Conn::DropAtAccept,
        Conn::DropAtAccept,
        Conn::DropAtAccept,
    ])
    .await;
    let error = server.model().get_response(req("hi")).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("before any response events were received"),
        "{error}"
    );
    assert_eq!(server.accepted(), 2, "one retry, then give up");
    assert!(server.frames().is_empty());
}

#[tokio::test]
async fn websocket_pre_event_retry_can_be_disabled() {
    let server = spawn_server(vec![
        Conn::DropAtAccept,
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-never"))]),
    ])
    .await;
    let model = server.model();
    let error = websocket_pre_event_retries_disabled(true, model.get_response(req("hi")))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("before any response events were received"),
        "{error}"
    );
    assert_eq!(server.accepted(), 1, "no retry while the retry is disabled");
}

#[tokio::test]
async fn does_not_retry_if_the_peer_drops_after_writing_on_a_reused_connection() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-1")),
        Step::Recv,
        Step::Drop,
    ])])
    .await;
    let model = server.model();
    let first = model.get_response(req("hi")).await.unwrap();
    assert_eq!(first.response_id.as_deref(), Some("resp-1"));
    let error = model.get_response(req("next")).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("before any response events were received"),
        "{error}"
    );
    assert_eq!(
        server.accepted(),
        1,
        "no reconnect once the frame was written"
    );
    assert_eq!(server.frames().len(), 2);
}

#[tokio::test]
async fn does_not_retry_after_pre_event_disconnect_once_request_sent() {
    let server = spawn_server(vec![
        Conn::Ws(vec![Step::Recv, Step::Drop]),
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-unexpected"))]),
    ])
    .await;
    let model = server.model();
    let error = model.get_response(req("hi")).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("before any response events were received"),
        "{error}"
    );
    assert_eq!(server.accepted(), 1);
    assert_eq!(server.frames().len(), 1);
    // A later request starts from a clean slate on a new connection.
    let next = model.get_response(req("again")).await.unwrap();
    assert_eq!(next.response_id.as_deref(), Some("resp-unexpected"));
    assert_eq!(server.accepted(), 2);
}

#[tokio::test]
async fn dead_connection_without_events_is_reported_as_a_pre_event_disconnect() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Close])]).await;
    let error = server.model().get_response(req("hi")).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.starts_with(WS_PRE_EVENT_DISCONNECT_MESSAGE),
        "{message}"
    );
    assert!(
        message.ends_with("(the request frame was already sent)"),
        "{message}"
    );
}

// -------------------------------------------------------------- close() semantics

#[tokio::test]
async fn close_closes_the_persistent_connection() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-close")),
    ])])
    .await;
    let model = server.model();
    model.get_response(req("hi")).await.unwrap();
    assert!(
        server.ended().is_empty(),
        "the connection stays open between requests"
    );
    model.close().await;
    wait_until("server to see the close", || server.ended().contains(&0)).await;
}

#[tokio::test]
async fn close_does_not_wait_for_a_held_request_lock() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Sleep(5_000)])]).await;
    let model = server.model();
    let in_flight = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("hi")).await })
    };
    wait_until("the request frame", || !server.frames().is_empty()).await;
    timeout(Duration::from_millis(500), model.close())
        .await
        .expect("close must not wait for the in-flight request");
    let error = timeout(Duration::from_secs(5), in_flight)
        .await
        .expect("the request is aborted by close")
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(WS_CLOSE_INVALIDATED_MESSAGE),
        "{error}"
    );
}

#[tokio::test]
async fn does_not_retry_after_client_initiated_close() {
    let server = spawn_server(vec![
        Conn::Ws(vec![Step::Recv, Step::Sleep(5_000)]),
        Conn::Ws(vec![Step::Recv, Step::Send(completed("resp-unexpected"))]),
    ])
    .await;
    let model = server.model();
    let in_flight = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("hi")).await })
    };
    wait_until("the request frame", || !server.frames().is_empty()).await;
    timeout(Duration::from_secs(1), model.close())
        .await
        .unwrap();
    let error = timeout(Duration::from_secs(5), in_flight)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(WS_CLOSE_INVALIDATED_MESSAGE),
        "{error}"
    );
    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .expect("advice");
    assert_eq!(advice.suggested, Some(false));
    assert!(advice.normalized.expect("normalized").is_abort);
    assert_eq!(
        server.accepted(),
        1,
        "no reconnect after the caller closed the model"
    );
    assert_eq!(server.frames().len(), 1);
}

#[tokio::test]
async fn close_during_failing_handshake_prevents_retry() {
    let server = spawn_server(vec![Conn::Stall, Conn::DropAtAccept]).await;
    let model = server.model();
    let in_flight = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("hi")).await })
    };
    wait_until("the TCP connection", || server.accepted() == 1).await;
    model.close().await;
    let error = timeout(Duration::from_secs(5), in_flight)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(WS_CLOSE_INVALIDATED_MESSAGE),
        "{error}"
    );
    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .expect("advice");
    assert!(advice.normalized.expect("normalized").is_abort);
    assert_eq!(server.accepted(), 1, "the retry never happens");
}

#[tokio::test]
async fn close_during_retry_handshake_discards_the_acquired_connection() {
    let server = spawn_server(vec![
        Conn::DropAtAccept,
        Conn::SlowWs(
            400,
            vec![Step::Recv, Step::Send(completed("resp-after-close"))],
        ),
    ])
    .await;
    let model = server.model();
    let in_flight = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("hi")).await })
    };
    wait_until("the retry handshake", || server.accepted() == 2).await;
    model.close().await;
    let error = timeout(Duration::from_secs(5), in_flight)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(WS_CLOSE_INVALIDATED_MESSAGE),
        "{error}"
    );
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        server.frames().is_empty(),
        "nothing is sent on the discarded connection"
    );
}

#[tokio::test]
async fn close_during_cached_connection_replacement_prevents_the_replacement() {
    let server = spawn_server(vec![
        Conn::Ws(vec![
            Step::Recv,
            Step::Send(completed("resp-1")),
            Step::Close,
        ]),
        Conn::SlowWs(400, vec![Step::Recv, Step::Send(completed("resp-2"))]),
    ])
    .await;
    let model = server.model();
    model.get_response(req("hi")).await.unwrap();
    wait_until("server to end connection 0", || server.ended().contains(&0)).await;
    let in_flight = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("next")).await })
    };
    wait_until("the replacement handshake", || server.accepted() == 2).await;
    model.close().await;
    let error = timeout(Duration::from_secs(5), in_flight)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains(WS_CLOSE_INVALIDATED_MESSAGE),
        "{error}"
    );
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(
        server.frames().len(),
        1,
        "the replacement never carried a request"
    );
}

// ----------------------------------------------------------------------- timeouts

#[tokio::test]
async fn get_response_applies_timeout_to_recv() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Sleep(500),
        Step::Send(completed("resp-timeout")),
    ])])
    .await;
    let error = server
        .model()
        .get_response(req_with("hi", timeout_settings(0.05)))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Responses websocket receive timed out"),
        "{error}"
    );
    wait_until("the timed-out connection to close", || {
        server.ended().contains(&0)
    })
    .await;
}

#[tokio::test]
async fn partial_receive_timeout_is_unsafe_to_replay() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(created("resp-partial")),
        Step::Sleep(500),
        Step::Send(completed("resp-partial")),
    ])])
    .await;
    let model = server.model();
    let error = model
        .get_response(req_with("hi", timeout_settings(0.1)))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Responses websocket receive timed out"),
        "{error}"
    );
    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .expect("advice");
    assert_eq!(advice.suggested, Some(false));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Unsafe));
    assert!(advice.response_started);
    wait_until("the timed-out connection to close", || {
        server.ended().contains(&0)
    })
    .await;
}

#[tokio::test]
async fn get_response_applies_timeout_while_waiting_for_request_lock() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Sleep(400),
        Step::Send(completed("resp-lock-1")),
    ])])
    .await;
    let model = server.model();
    let first = {
        let model = model.clone();
        tokio::spawn(async move { model.get_response(req("first")).await })
    };
    wait_until("the first request frame", || !server.frames().is_empty()).await;
    let error = model
        .get_response(req_with("second", timeout_settings(0.05)))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("request lock wait timed out"),
        "{error}"
    );
    let first = first.await.unwrap().unwrap();
    assert_eq!(first.response_id.as_deref(), Some("resp-lock-1"));
    assert_eq!(
        server.frames().len(),
        1,
        "the lock-wait timeout sent nothing"
    );
}

#[tokio::test]
async fn connect_timeout_is_applied_and_safe_to_replay() {
    let server = spawn_server(vec![Conn::Stall]).await;
    let model = server.model();
    let error = model
        .get_response(req_with("hi", timeout_settings(0.1)))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Responses websocket connect timed out"),
        "{error}"
    );
    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: Some("resp_prev"),
            conversation_id: None,
        })
        .expect("advice");
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
}

// ----------------------------------------------------- request preparation on the wire

#[tokio::test]
async fn frame_preserves_envelope_fields_over_extra_body() {
    let model = OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test"));
    let mut extra_body = serde_json::Map::new();
    extra_body.insert("type".into(), json!("not-response-create"));
    extra_body.insert("stream".into(), json!(false));
    extra_body.insert("custom".into(), json!("value"));
    let frame = model
        .build_frame(&req_with(
            "hi",
            ModelSettings {
                extra_body: Some(extra_body),
                ..Default::default()
            },
        ))
        .unwrap();
    assert_eq!(frame["type"], "response.create");
    assert_eq!(frame["stream"], true);
    assert_eq!(frame["custom"], "value");
}

#[tokio::test]
async fn frame_strips_the_client_timeout() {
    let model = OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test"));
    let frame = model
        .build_frame(&req_with(
            "hi",
            ModelSettings {
                timeout: Some(30.0),
                metadata: Some(HashMap::from([(
                    "request_id".to_string(),
                    "123".to_string(),
                )])),
                ..Default::default()
            },
        ))
        .unwrap();
    assert_eq!(frame["type"], "response.create");
    assert_eq!(frame["metadata"], json!({"request_id": "123"}));
    assert!(frame.get("timeout").is_none());
}

#[tokio::test]
async fn handshake_includes_client_auth_headers() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    server.model().get_response(req("hi")).await.unwrap();
    let hs = server.handshakes();
    assert_eq!(hs[0].path(), "/v1/responses");
    assert_eq!(hs[0].header("authorization"), Some("Bearer sk-test"));
    assert!(
        hs[0]
            .header("user-agent")
            .is_some_and(|ua| ua.starts_with("Agents/"))
    );
}

#[tokio::test]
async fn handshake_default_headers_override_auth_case_insensitively() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    let client = server
        .client()
        .with_header("authorization", "Bearer override-key");
    OpenAIResponsesWSModel::new("gpt-4", client)
        .get_response(req("hi"))
        .await
        .unwrap();
    let hs = server.handshakes();
    assert_eq!(
        hs[0].headers["authorization"],
        vec!["Bearer override-key".to_string()]
    );
}

#[tokio::test]
async fn handshake_can_remove_an_inherited_header() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    let client = server.client().without_header("User-Agent");
    OpenAIResponsesWSModel::new("gpt-4", client)
        .get_response(req("hi"))
        .await
        .unwrap();
    let hs = server.handshakes();
    assert!(hs[0].header("authorization").is_some());
    assert!(hs[0].header("user-agent").is_none());
}

#[tokio::test]
async fn handshake_extra_headers_replace_case_insensitively() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    let settings = ModelSettings {
        extra_headers: Some(HashMap::from([
            (
                "authorization".to_string(),
                "Bearer override-key".to_string(),
            ),
            ("user-agent".to_string(), "Custom UA".to_string()),
        ])),
        ..Default::default()
    };
    server
        .model()
        .get_response(req_with("hi", settings))
        .await
        .unwrap();
    let hs = server.handshakes();
    assert_eq!(
        hs[0].headers["authorization"],
        vec!["Bearer override-key".to_string()]
    );
    assert_eq!(hs[0].headers["user-agent"], vec!["Custom UA".to_string()]);
}

#[tokio::test]
async fn url_maps_a_non_tls_http_base_to_ws() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    // `http://127.0.0.1:port/v1/` (trailing slash) becomes `ws://.../v1/responses`.
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{}/", server.base_url));
    OpenAIResponsesWSModel::new("gpt-4", client)
        .get_response(req("hi"))
        .await
        .unwrap();
    assert_eq!(server.handshakes()[0].path(), "/v1/responses");
}

#[tokio::test]
async fn url_appends_the_path_after_an_existing_query_and_normalizes_http_schemes() {
    for scheme in ["ws", "http"] {
        let server =
            spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
        let model = OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test"))
            .with_websocket_base_url(format!("{scheme}://127.0.0.1:{}/v1?token=abc", server.port));
        let extra_query = HashMap::from([("route".to_string(), "team-a".to_string())]);
        model
            .get_response(req_with(
                "hi",
                ModelSettings {
                    extra_query: Some(extra_query),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        let hs = server.handshakes();
        assert_eq!(hs[0].path(), "/v1/responses", "{scheme}");
        assert_eq!(
            hs[0].query(),
            HashMap::from([
                ("token".to_string(), "abc".to_string()),
                ("route".to_string(), "team-a".to_string()),
            ]),
            "{scheme}"
        );
    }
}

#[tokio::test]
async fn url_merges_client_default_query_under_request_query() {
    let server = spawn_server(vec![Conn::Ws(vec![Step::Recv, Step::Send(completed("r"))])]).await;
    let client = OpenAIClient::new("sk-test")
        .with_default_query("api-version", "2025-01-01-preview")
        .with_websocket_base_url(format!("ws://127.0.0.1:{}/v1?token=abc", server.port));
    let settings = ModelSettings {
        extra_query: Some(HashMap::from([
            ("route".to_string(), "team-a".to_string()),
            ("api-version".to_string(), "2026-01-01-preview".to_string()),
        ])),
        ..Default::default()
    };
    OpenAIResponsesWSModel::new("gpt-4", client)
        .get_response(req_with("hi", settings))
        .await
        .unwrap();
    let hs = server.handshakes();
    assert_eq!(hs[0].path(), "/v1/responses");
    assert_eq!(
        hs[0].query(),
        HashMap::from([
            ("token".to_string(), "abc".to_string()),
            ("api-version".to_string(), "2026-01-01-preview".to_string()),
            ("route".to_string(), "team-a".to_string()),
        ])
    );
}

// ------------------------------------------------------------- usage accounting

#[tokio::test]
async fn request_is_counted_when_the_provider_omits_usage() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed("resp-no-usage")),
    ])])
    .await;
    let response = server.model().get_response(req("hi")).await.unwrap();
    assert_eq!(response.usage.requests, 1);
    assert_eq!(response.usage.total_tokens, 0);
}

#[tokio::test]
async fn streamed_request_is_counted_when_the_provider_omits_usage() {
    let server = spawn_server(vec![Conn::Ws(vec![
        Step::Recv,
        Step::Send(completed_without_usage(message_output("ok"))),
    ])])
    .await;
    let agent = Agent::<()>::new("test").with_model_instance(Arc::new(server.model()));
    let options = RunOptions::new(())
        .run_config(openai_agents_sdk::RunConfig::new().with_tracing_disabled(true));
    let mut result = Runner::run_streamed_with_options(&agent, "hi", options);
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    let usage = result.into_result().expect("completed run").usage();
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 0);
}

// ------------------------------------------------------------------------- session

/// One turn that ends the run: the answer is a message, so no further model call is needed.
fn ws_response(id: &str) -> Vec<Step> {
    let mut event = completed(id);
    event["response"]["output"] = message_output("ok");
    vec![Step::Recv, Step::Send(event)]
}

async fn run_through(session: &ResponsesWebSocketSession, model: &str) -> Result<(), AgentsError> {
    let agent = Agent::<()>::new("test").with_model(model);
    let options = RunOptions::new(()).run_config(session.run_config().with_tracing_disabled(true));
    Runner::run_with_options(&agent, "hello", options)
        .await
        .map(|_| ())
}

#[tokio::test]
async fn session_builds_a_shared_responses_websocket_run_config() {
    let server = spawn_server(vec![Conn::Ws(
        [ws_response("resp-a"), ws_response("resp-b")].concat(),
    )])
    .await;
    let session = ResponsesWebSocketSession::new(server.client());
    run_through(&session, "gpt-4.1").await.unwrap();
    run_through(&session, "gpt-4.1").await.unwrap();
    // Both runs went over one websocket connection created by the shared provider.
    assert_eq!(server.accepted(), 1);
    assert_eq!(server.frames().len(), 2);
    assert_eq!(server.frames()[0]["type"], "response.create");
    session.close().await;
}

#[tokio::test]
async fn session_preserves_openai_prefix_routing() {
    let server = spawn_server(vec![Conn::Ws(ws_response("resp-a"))]).await;
    let session = ResponsesWebSocketSession::new(server.client());
    run_through(&session, "openai/gpt-4.1").await.unwrap();
    assert_eq!(server.frames()[0]["model"], "gpt-4.1");
    session.close().await;
}

#[tokio::test]
async fn session_can_preserve_openai_prefix_model_ids() {
    let server = spawn_server(vec![Conn::Ws(ws_response("resp-a"))]).await;
    let session = ResponsesWebSocketSession::new(server.client())
        .with_openai_prefix_mode(PrefixMode::ModelId);
    run_through(&session, "openai/gpt-4.1").await.unwrap();
    assert_eq!(server.frames()[0]["model"], "openai/gpt-4.1");
    session.close().await;
}

#[tokio::test]
async fn session_can_preserve_unknown_prefix_model_ids() {
    let server = spawn_server(vec![Conn::Ws(ws_response("resp-a"))]).await;
    let session = ResponsesWebSocketSession::new(server.client())
        .with_unknown_prefix_mode(PrefixMode::ModelId);
    run_through(&session, "openrouter/openai/gpt-4.1")
        .await
        .unwrap();
    assert_eq!(server.frames()[0]["model"], "openrouter/openai/gpt-4.1");
    session.close().await;

    // The default rejects an unknown prefix instead of guessing.
    let strict = ResponsesWebSocketSession::new(server.client());
    let error = run_through(&strict, "openrouter/openai/gpt-4.1")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Unknown prefix"), "{error}");
}

#[tokio::test]
async fn session_close_closes_the_provider_connections() {
    let server = spawn_server(vec![Conn::Ws(ws_response("resp-a"))]).await;
    let session = ResponsesWebSocketSession::new(server.client());
    run_through(&session, "gpt-4.1").await.unwrap();
    assert!(
        server.ended().is_empty(),
        "the connection stays warm after the run"
    );
    session.close().await;
    wait_until("the shared connection to close", || {
        server.ended().contains(&0)
    })
    .await;
}
