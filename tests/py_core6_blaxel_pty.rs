//! Ports of the `TestPtyExec` group of `tests/extensions/sandbox/test_blaxel.py`.
//!
//! Python swaps `aiohttp` for a fake WebSocket; the Rust session opens a real WebSocket to the
//! sandbox's `/terminal/ws`, so the same cases run against a local server that upgrades the
//! connection, plays a script of frames, and records what the client sent.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelSandboxClient, BlaxelSandboxClientOptions, BlaxelTimeouts,
};
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxClient, SandboxSession};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded, Shared};
use tokio_tungstenite::tungstenite::Message;

/// One step of the terminal's script.
#[derive(Clone)]
enum Frame {
    Text(String),
    Binary(Vec<u8>),
    /// Close the WebSocket cleanly.
    Close,
    /// Drop the TCP connection without a close frame, as a network error does.
    Abort,
}

fn output(data: &str) -> Frame {
    Frame::Text(json!({"type": "output", "data": data}).to_string())
}

#[derive(Clone, Default)]
struct Terminal {
    script: Vec<Frame>,
    /// Never answer the upgrade request.
    hang: bool,
    /// Every text frame the client sent, in order.
    received: Vec<String>,
    /// How many times a client opened the terminal.
    connections: usize,
    /// Whether a client closed the WebSocket.
    client_closed: bool,
}

async fn start(terminal: Shared<Terminal>) -> MockServer {
    let url = Shared::new(String::new());
    let slot = url.clone();
    let server = MockServer::start_raw(move |request: Recorded, stream| {
        let terminal = terminal.clone();
        let own_url = slot.get();
        async move {
            if request.path == "/terminal/ws" {
                if terminal.get().hang {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                terminal.with(|t| t.connections += 1);
                let mut ws = support::accept_websocket(stream, &request, None).await;
                let script = terminal.get().script;
                for frame in script {
                    match frame {
                        Frame::Text(text) => {
                            let _ = ws.send(Message::Text(text.into())).await;
                        }
                        Frame::Binary(bytes) => {
                            let _ = ws.send(Message::Binary(bytes.into())).await;
                        }
                        Frame::Close => {
                            let _ = ws.close(None).await;
                        }
                        Frame::Abort => return,
                    }
                }
                while let Some(Ok(message)) = ws.next().await {
                    match message {
                        Message::Text(text) => {
                            terminal.with(|t| t.received.push(text.to_string()));
                        }
                        Message::Close(_) => {
                            terminal.with(|t| t.client_closed = true);
                            break;
                        }
                        _ => {}
                    }
                }
                terminal.with(|t| t.client_closed = true);
                return;
            }
            let mut stream = stream;
            let response = match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v0/sandboxes") => MockResponse::json(
                    json!({ "metadata": { "name": "agents-test", "url": own_url } }),
                ),
                _ => MockResponse::status(404, json!({})),
            };
            let _ = support::write_response(&mut stream, response).await;
        }
    })
    .await;
    url.set(server.base_url.clone());
    server
}

fn options(base_url: &str) -> BlaxelSandboxClientOptions {
    BlaxelSandboxClientOptions::new()
        .with_name("agents-test")
        .with_api_key("bl-token")
        .with_workspace("acme")
        .with_region("us-pdx-1")
        .with_api_base_url(format!("{base_url}/v0"))
        .with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(5),
            ..BlaxelTimeouts::default()
        })
}

async fn session_with(options: BlaxelSandboxClientOptions) -> Arc<dyn SandboxSession> {
    BlaxelSandboxClient::new(options)
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create")
}

async fn session(server: &MockServer) -> Arc<dyn SandboxSession> {
    session_with(options(&server.base_url)).await
}

fn frames(received: &[String]) -> Vec<Value> {
    received
        .iter()
        .map(|frame| serde_json::from_str(frame).expect("a JSON frame"))
        .collect()
}

async fn eventually(mut check: impl FnMut() -> bool) {
    for _ in 0..100 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the condition never held");
}

fn request(command: &str) -> PtyExecRequest {
    PtyExecRequest::new(command).with_yield_time_ms(1500)
}

/// `test_pty_exec_start_decodes_output_messages`: text and binary frames, the capitalized field
/// names older sandbox builds send, and a malformed frame that is skipped.
#[tokio::test]
async fn pty_decodes_output_frames() {
    let cases: Vec<(&str, Vec<Frame>, &str)> = vec![
        ("text", vec![output("hello from pty")], "hello from pty"),
        (
            "binary",
            vec![Frame::Binary(
                json!({"type": "output", "data": "binary-data"})
                    .to_string()
                    .into_bytes(),
            )],
            "binary-data",
        ),
        (
            "capitalized-keys",
            vec![Frame::Text(
                json!({"Type": "output", "Data": "cap-data"}).to_string(),
            )],
            "cap-data",
        ),
        (
            "invalid-json-ignored",
            vec![Frame::Text("not json".into()), output("valid")],
            "valid",
        ),
    ];
    for (name, mut script, expected) in cases {
        script.push(Frame::Close);
        let terminal = Shared::new(Terminal {
            script,
            ..Terminal::default()
        });
        let server = start(terminal).await;
        let session = session(&server).await;

        let update = session
            .pty_exec_start(request("echo hello"))
            .await
            .expect(name);

        assert!(
            update.output_text().contains(expected),
            "{name}: {update:?}"
        );
        assert_eq!(
            update.process_id, None,
            "{name}: the closed terminal ended the session"
        );
        server.stop();
    }
}

/// `test_pty_exec_start_preserves_active_session` and `test_pty_exec_default_yield_time`'s
/// still-running half: a terminal that stays open keeps its session id.
#[tokio::test]
async fn pty_keeps_an_active_session_registered() {
    let terminal = Shared::new(Terminal {
        script: vec![output("still running")],
        ..Terminal::default()
    });
    let server = start(terminal.clone()).await;
    let session = session(&server).await;

    let update = session
        .pty_exec_start(PtyExecRequest::new("echo hello").with_yield_time_ms(300))
        .await
        .expect("start");

    assert!(update.output_text().contains("still running"));
    let id = update.process_id.expect("the terminal is still open");
    // The session accepts a follow-up poll, which proves it is registered.
    let poll = session
        .pty_write_stdin(PtyWriteRequest::new(id, "").with_yield_time_ms(300))
        .await
        .expect("poll");
    assert_eq!(poll.process_id, Some(id));
    session.pty_terminate_all().await.expect("terminate");
    server.stop();
}

/// The command is sent as the first input frame, as the Python session does.
#[tokio::test]
async fn pty_sends_the_command_as_the_first_input_frame() {
    let terminal = Shared::new(Terminal::default());
    let server = start(terminal.clone()).await;
    let session = session(&server).await;

    let update = session
        .pty_exec_start(PtyExecRequest::new("echo hello").with_yield_time_ms(300))
        .await
        .expect("start");
    assert!(update.process_id.is_some());

    eventually(|| !terminal.get().received.is_empty()).await;
    let sent = frames(&terminal.get().received);
    assert_eq!(sent[0]["type"], "input");
    assert!(
        sent[0]["data"].as_str().unwrap().ends_with('\n'),
        "{}",
        sent[0]
    );
    assert!(sent[0]["data"].as_str().unwrap().contains("echo hello"));
    session.pty_terminate_all().await.unwrap();
    server.stop();
}

/// `test_pty_exec_start_timeout` and `test_pty_exec_start_timeout_reports_default_timeout`: a
/// terminal that never answers the upgrade is an `ExecTimeout` carrying the session's exec timeout.
#[tokio::test]
async fn pty_connect_timeout_reports_the_exec_timeout() {
    let terminal = Shared::new(Terminal {
        hang: true,
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session_with(options(&server.base_url).with_timeouts(BlaxelTimeouts {
        exec: Duration::from_millis(50),
        ..BlaxelTimeouts::default()
    }))
    .await;

    let error = session
        .pty_exec_start(PtyExecRequest::new("echo hello"))
        .await
        .expect_err("timeout");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert_eq!(
        error.context.get("timeout_s").map(String::as_str),
        Some("0.05")
    );
    server.stop();
}

/// `test_pty_exec_start_connection_error`
#[tokio::test]
async fn pty_connection_failure_is_a_transport_error() {
    let terminal = Shared::new(Terminal::default());
    let server = start(terminal).await;
    let session =
        session_with(options(&server.base_url).with_force_sandbox_url("http://127.0.0.1:1")).await;

    let error = session
        .pty_exec_start(PtyExecRequest::new("echo hello"))
        .await
        .expect_err("refused");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    server.stop();
}

/// `test_pty_write_stdin_sends_only_nonempty_input`
#[tokio::test]
async fn pty_write_stdin_sends_only_nonempty_input() {
    for (chars, extra_frames) in [("input\n", 1usize), ("", 0)] {
        let terminal = Shared::new(Terminal::default());
        let server = start(terminal.clone()).await;
        let session = session(&server).await;
        let update = session
            .pty_exec_start(PtyExecRequest::new("cat").with_yield_time_ms(300))
            .await
            .expect("start");
        let id = update.process_id.expect("open terminal");
        eventually(|| terminal.get().received.len() == 1).await;

        let update = session
            .pty_write_stdin(PtyWriteRequest::new(id, chars).with_yield_time_ms(300))
            .await
            .expect("write");

        assert_eq!(update.process_id, Some(id), "{chars:?}");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let received = frames(&terminal.get().received);
        assert_eq!(received.len(), 1 + extra_frames, "{chars:?}: {received:?}");
        if extra_frames == 1 {
            assert_eq!(received[1], json!({"type": "input", "data": "input\n"}));
        }
        session.pty_terminate_all().await.unwrap();
        server.stop();
    }
}

/// `test_pty_terminate_all`: every terminal is closed, and the ids stop resolving.
#[tokio::test]
async fn pty_terminate_all_closes_every_terminal() {
    let terminal = Shared::new(Terminal::default());
    let server = start(terminal.clone()).await;
    let session = session(&server).await;
    let update = session
        .pty_exec_start(PtyExecRequest::new("cat").with_yield_time_ms(300))
        .await
        .expect("start");
    let id = update.process_id.expect("open terminal");

    session.pty_terminate_all().await.expect("terminate");

    eventually(|| terminal.get().client_closed).await;
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(id, "x"))
        .await
        .expect_err("the id is gone");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// `test_stop_calls_pty_terminate`: stopping the session terminates its terminals.
#[tokio::test]
async fn pty_stop_terminates_the_terminals() {
    let terminal = Shared::new(Terminal::default());
    let server = start(terminal.clone()).await;
    let session = session(&server).await;
    let update = session
        .pty_exec_start(PtyExecRequest::new("cat").with_yield_time_ms(300))
        .await
        .expect("start");
    let id = update.process_id.expect("open terminal");

    session.stop().await.expect("stop");

    eventually(|| terminal.get().client_closed).await;
    assert_eq!(
        session
            .pty_write_stdin(PtyWriteRequest::new(id, ""))
            .await
            .expect_err("gone")
            .code,
        ErrorCode::PtySessionNotFound
    );
    server.stop();
}

/// `test_pty_ws_reader_error_message`: an `error` frame's text is output and ends the session.
#[tokio::test]
async fn pty_error_frame_is_output_and_ends_the_session() {
    let terminal = Shared::new(Terminal {
        script: vec![Frame::Text(
            json!({"type": "error", "data": "something failed"}).to_string(),
        )],
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session(&server).await;

    let update = session
        .pty_exec_start(request("bad_cmd"))
        .await
        .expect("start");

    assert!(
        update.output_text().contains("something failed"),
        "{update:?}"
    );
    assert_eq!(update.process_id, None);
    server.stop();
}

/// `test_pty_ws_reader_close_message`
#[tokio::test]
async fn pty_close_frame_ends_the_session_after_its_output() {
    let terminal = Shared::new(Terminal {
        script: vec![output("hi"), Frame::Close],
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session(&server).await;

    let update = session
        .pty_exec_start(request("echo test"))
        .await
        .expect("start");

    assert!(update.output_text().contains("hi"));
    assert_eq!(update.process_id, None);
    server.stop();
}

/// `test_pty_ws_reader_error_type_message` and `test_pty_ws_reader_exception_in_iter`: a
/// connection that dies without a close frame ends the stream; the update still comes back.
#[tokio::test]
async fn pty_broken_connection_ends_the_session() {
    let terminal = Shared::new(Terminal {
        script: vec![Frame::Abort],
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session(&server).await;

    let result = session.pty_exec_start(request("echo test")).await;

    match result {
        Ok(update) => assert_eq!(update.process_id, None, "{update:?}"),
        // The command frame can race the drop; that is a transport failure, not a hang.
        Err(error) => assert_eq!(error.code, ErrorCode::ExecTransportError),
    }
    server.stop();
}

/// `test_pty_exec_default_yield_time`: a command that finishes inside the default window returns
/// as soon as the stream ends.
#[tokio::test]
async fn pty_returns_early_when_the_stream_ends_inside_the_default_window() {
    let terminal = Shared::new(Terminal {
        script: vec![output("quick"), Frame::Close],
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session(&server).await;

    let started = std::time::Instant::now();
    let update = session
        .pty_exec_start(PtyExecRequest::new("echo test"))
        .await
        .expect("start");

    assert!(update.output_text().contains("quick"));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    server.stop();
}

/// `test_pty_max_output_tokens`: output over the budget is truncated and reports its size.
#[tokio::test]
async fn pty_truncates_output_to_the_token_budget() {
    let long_output = "x".repeat(10_000);
    let terminal = Shared::new(Terminal {
        script: vec![output(&long_output), Frame::Close],
        ..Terminal::default()
    });
    let server = start(terminal).await;
    let session = session(&server).await;

    let mut pty_request = request("echo test");
    pty_request.max_output_tokens = Some(10);
    let update = session.pty_exec_start(pty_request).await.expect("start");

    assert!(
        update.output.len() < long_output.len(),
        "{}",
        update.output.len()
    );
    assert!(update.original_token_count.is_some());
    server.stop();
}

/// `test_pty_exec_start_cancellation_closes_unregistered_http_session`: abandoning a start that
/// is still connecting registers nothing, so no session id resolves afterwards.
#[tokio::test]
async fn pty_cancelled_start_leaves_no_session() {
    let terminal = Shared::new(Terminal {
        hang: true,
        ..Terminal::default()
    });
    let server = start(terminal.clone()).await;
    let session = session(&server).await;
    let task = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .pty_exec_start(PtyExecRequest::new("echo hello"))
                .await
        })
    };
    // The connect attempt is in flight once the server has seen the upgrade request.
    eventually(|| {
        server
            .requests()
            .iter()
            .any(|request| request.path == "/terminal/ws")
    })
    .await;

    task.abort();
    assert!(task.await.expect_err("aborted").is_cancelled());

    assert_eq!(terminal.get().connections, 0);
    for id in [1u32, 2, 1000] {
        assert_eq!(
            session
                .pty_write_stdin(PtyWriteRequest::new(id, ""))
                .await
                .expect_err("nothing registered")
                .code,
            ErrorCode::PtySessionNotFound
        );
    }
    server.stop();
}

/// `test_pty_exec_with_pruning` and `test_pty_warning_threshold`: a finished terminal never
/// occupies a slot, so a session that has run many short commands still starts another.
#[tokio::test]
async fn pty_finished_terminals_do_not_fill_the_session_table() {
    let terminal = Shared::new(Terminal {
        script: vec![output("pruned-test"), Frame::Close],
        ..Terminal::default()
    });
    let server = start(terminal.clone()).await;
    let session = session(&server).await;

    for _ in 0..70 {
        let update = session
            .pty_exec_start(PtyExecRequest::new("echo test").with_yield_time_ms(1500))
            .await
            .expect("start");
        assert!(update.output_text().contains("pruned-test"));
        assert_eq!(update.process_id, None);
    }

    assert_eq!(terminal.get().connections, 70);
    server.stop();
}

/// `test_pty_prune_below_max`: registering a terminal below the session limit prunes none.
#[test]
fn pty_registering_below_the_limit_prunes_nothing() {
    let manager = openai_agents_sdk::sandbox::pty::RemotePtyManager::new();
    for _ in 0..3 {
        let (_, pruned) = manager.register(true).expect("register");
        assert!(pruned.is_empty());
    }
    assert_eq!(manager.len(), 3);
}

/// `test_collect_output_timeout_path`: a window that ends with no output returns an empty update
/// and leaves the session open.
#[tokio::test]
async fn pty_collect_with_no_output_returns_empty_and_stays_open() {
    let manager = openai_agents_sdk::sandbox::pty::RemotePtyManager::new();
    let (process, _) = manager.register(true).expect("register");

    let update = process.collect(1, None).await;

    assert!(update.output.is_empty());
    assert_eq!(update.process_id, Some(process.id()));
}
