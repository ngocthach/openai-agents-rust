//! Sandbox cases the earlier audits left as Python-only:
//!
//! - `tests/extensions/sandbox/test_daytona.py::test_tty_waiter_closes_output_after_stream_callback_finishes`:
//!   a terminal whose last output arrives just before it exits still delivers that tail; the
//!   process is only reported finished after the tail was recorded.
//! - `tests/extensions/sandbox/test_vercel_provider_contract.py::test_installed_provider_closes_unadopted_reconnect_client[timeout]`:
//!   a resume whose session stays `pending` past the wait budget creates a replacement and
//!   leaves the original sandbox alone.
//! - `tests/sandbox/test_session_sinks.py::test_sandbox_session_aclose_flushes_best_effort_sink_tasks`:
//!   a sink that defers delivery until it is flushed has seen the `stop` and `shutdown` finish
//!   events once the instrumented session is shut down.

#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt as _, StreamExt as _};
use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::events::{
    EventPhase, EventSink, InstrumentedSession, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    ExecShell, Manifest, PtyExecRequest, PtyWriteRequest, SandboxClient, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------------------------
// Daytona: the terminal tail is delivered before the exit
// ---------------------------------------------------------------------------------------------

fn daytona_platform(request: &Recorded) -> Option<MockResponse> {
    let path = request.path.as_str();
    if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
        return Some(MockResponse::json(
            json!({ "id": "sbx-daytona", "state": "started" }),
        ));
    }
    None
}

fn command_result(stdout: &str, exit_code: i64) -> MockResponse {
    MockResponse::json(json!({
        "output": stdout,
        "stdout": stdout,
        "stderr": "",
        "exitCode": exit_code,
    }))
}

fn control(status: &str, extra: Value) -> Message {
    let mut value = json!({ "type": "control", "status": status });
    if let (Some(object), Some(extra)) = (value.as_object_mut(), extra.as_object()) {
        object.extend(extra.clone());
    }
    Message::Text(value.to_string().into())
}

/// A toolbox whose terminal greets with `$ ` and, on `exit`, writes a late `tail` before it
/// reports the exit and closes.
async fn tail_terminal_server() -> MockServer {
    MockServer::start_raw(move |request, mut stream| async move {
        let upgrade = request
            .header("upgrade")
            .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
        if !upgrade {
            let response = daytona_platform(&request).unwrap_or_else(|| command_result("", 0));
            let _ = support::write_response(&mut stream, response).await;
            return;
        }
        let protocol = request
            .header("sec-websocket-protocol")
            .and_then(|value| value.split(',').next())
            .map(|value| value.trim().to_string());
        let mut socket = support::accept_websocket(stream, &request, protocol.as_deref()).await;
        let _ = socket.send(control("connected", json!({}))).await;
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Binary(bytes)) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    if text.contains("exit") {
                        // The tail is still being produced when the process ends.
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        let _ = socket.send(Message::Binary(b"tail".to_vec().into())).await;
                        let _ = socket
                            .send(control("exited", json!({ "exitCode": 0 })))
                            .await;
                        let _ = socket.close(None).await;
                    } else {
                        let _ = socket.send(Message::Binary(b"$ ".to_vec().into())).await;
                    }
                }
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(_) => {}
            }
        }
    })
    .await
}

/// Python: `test_tty_waiter_closes_output_after_stream_callback_finishes`.
#[tokio::test]
async fn a_terminal_tail_is_delivered_before_the_exit_is_reported() {
    let server = tail_terminal_server().await;
    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("sh")
                .with_shell(ExecShell::None)
                .with_tty(true)
                .with_yield_time_ms(300),
        )
        .await
        .expect("start");
    let process_id = started.process_id.expect("still running");

    let finished = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "exit\n").with_yield_time_ms(3_000))
        .await
        .expect("exit");

    assert_eq!(
        finished.process_id, None,
        "the process is reported finished"
    );
    assert_eq!(finished.exit_code, Some(0));
    assert!(
        finished.output_text().ends_with("tail"),
        "{:?}",
        finished.output_text()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Vercel: a resume whose pending session never starts in time is replaced
// ---------------------------------------------------------------------------------------------

const VERCEL_ROOT: &str = "/vercel/sandbox";

fn vercel_session(id: &str, status: &str) -> Value {
    json!({
        "id": id,
        "memory": 2048,
        "vcpus": 2,
        "region": "iad1",
        "timeout": 270000,
        "status": status,
        "requestedAt": 1,
        "createdAt": 1,
        "updatedAt": 1,
        "cwd": VERCEL_ROOT,
    })
}

/// Python: `test_installed_provider_closes_unadopted_reconnect_client[timeout]`.
#[tokio::test]
async fn a_resume_whose_pending_session_times_out_creates_a_replacement() {
    let server = MockServer::start(|request: &Recorded| {
        match (request.method.as_str(), request.path.as_str()) {
            // The original session never leaves `pending`.
            ("GET", "/v2/sandboxes/sessions/sess-1") => MockResponse::json(
                json!({ "session": vercel_session("sess-1", "pending"), "routes": [] }),
            ),
            ("POST", "/v3/sandboxes") => MockResponse::json(json!({
                "sandbox": {
                    "name": "agents-test",
                    "persistent": false,
                    "createdAt": 1,
                    "updatedAt": 1,
                    "currentSessionId": "sess-2",
                },
                "session": vercel_session("sess-2", "running"),
                "routes": [],
            })),
            ("GET", "/v2/sandboxes/sessions/sess-2") => MockResponse::json(
                json!({ "session": vercel_session("sess-2", "running"), "routes": [] }),
            ),
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await;
    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(5))
            .with_wait_for_running(Duration::from_millis(150)),
    );
    let state = SandboxSessionState::new(
        "vercel",
        "state-1",
        VERCEL_ROOT,
        &Manifest::new().with_root(VERCEL_ROOT),
    )
    .with_provider_id(Some("agents-test:sess-1".to_string()));

    let resumed = client
        .resume(&state, Manifest::new().with_root(VERCEL_ROOT))
        .await
        .expect("a timed-out wait falls back to a replacement");

    assert_eq!(server.count("POST", "/v3/sandboxes"), 1);
    assert!(
        server.count("GET", "/v2/sandboxes/sessions/sess-1") >= 2,
        "the pending session was polled until the budget ran out"
    );
    assert!(!resumed.system_state_preserved_on_start());
    // The original sandbox is neither stopped nor deleted by the fallback.
    assert!(
        server
            .requests()
            .iter()
            .all(|request| request.method != "DELETE" && !request.path.ends_with("/stop")),
        "{:?}",
        server
            .requests()
            .iter()
            .map(|request| format!("{} {}", request.method, request.path))
            .collect::<Vec<_>>()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Sinks: deferred delivery is flushed by shutdown
// ---------------------------------------------------------------------------------------------

/// Holds events back until it is flushed, as Python's best-effort sink tasks run behind the
/// operation that produced them.
#[derive(Debug, Default)]
struct DeferredSink {
    pending: Mutex<Vec<(String, EventPhase)>>,
    delivered: Mutex<Vec<(String, EventPhase)>>,
}

#[async_trait]
impl EventSink for DeferredSink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        self.pending
            .lock()
            .unwrap()
            .push((event.op.clone(), event.phase));
    }

    async fn flush(&self) {
        let pending = std::mem::take(&mut *self.pending.lock().unwrap());
        self.delivered.lock().unwrap().extend(pending);
    }
}

/// Python: `test_sandbox_session_aclose_flushes_best_effort_sink_tasks`.
#[tokio::test]
async fn shutting_down_flushes_deferred_sink_deliveries() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = UnixLocalSandboxClient::new()
        .create(Some(Manifest::new().with_root(root.to_string_lossy())))
        .await
        .unwrap();
    let sink = Arc::new(DeferredSink::default());
    let wrapped = InstrumentedSession::new(inner, sink.clone());

    use openai_agents_sdk::sandbox::session::SandboxSession as _;
    wrapped.start().await.unwrap();
    wrapped.stop().await.unwrap();
    wrapped.shutdown().await.unwrap();

    let delivered = sink.delivered.lock().unwrap().clone();
    assert!(
        delivered.contains(&("stop".to_string(), EventPhase::Finish)),
        "{delivered:?}"
    );
    assert!(
        delivered.contains(&("shutdown".to_string(), EventPhase::Finish)),
        "{delivered:?}"
    );
    assert!(sink.pending.lock().unwrap().is_empty());
}
