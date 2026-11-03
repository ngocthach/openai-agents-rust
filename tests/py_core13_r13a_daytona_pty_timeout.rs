//! The Daytona PTY start timeout: `tests/extensions/sandbox/test_daytona.py`
//! `test_pty_start_maps_sdk_timeout_failures`, where `pty_exec_start(..., timeout=2.0)` bounds
//! every provider call of the start with `asyncio.wait_for(..., timeout=exec_timeout)`.
//!
//! A step that outlasts the call's timeout is an `ExecTimeout`, the half-created toolbox session
//! is cleaned up, and nothing stays registered. Without a timeout a slow step is waited for.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecShell, PtyExecRequest, PtyWriteRequest, SandboxClient, SandboxSession,
};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded};

const TOOLBOX: &str = "/toolbox/sbx-daytona/toolbox";

/// A server whose toolbox answers each request after the delay `toolbox` names.
async fn delayed_server(
    toolbox: impl Fn(&Recorded) -> (u64, MockResponse) + Send + Sync + 'static,
) -> MockServer {
    let toolbox = Arc::new(toolbox);
    MockServer::start_raw(move |request, mut stream| {
        let toolbox = toolbox.clone();
        async move {
            let path = request.path.as_str();
            let (delay_ms, response) = if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
                (
                    0,
                    MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" })),
                )
            } else {
                toolbox(&request)
            };
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let _ = support::write_response(&mut stream, response).await;
        }
    })
    .await
}

async fn session(server: &MockServer) -> Arc<dyn SandboxSession> {
    DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create")
}

fn request(tty: bool, timeout: Option<Duration>) -> PtyExecRequest {
    PtyExecRequest::new("python3")
        .with_shell(ExecShell::None)
        .with_tty(tty)
        .with_yield_time_ms(200)
        .with_timeout(timeout)
}

fn deletes(server: &MockServer) -> usize {
    server
        .requests()
        .iter()
        .filter(|r| r.method == "DELETE" && r.path.contains("/process/session/"))
        .count()
}

fn cmd_started() -> MockResponse {
    MockResponse::json(json!({ "cmdId": "cmd-1" }))
}

/// A stalled toolbox session create is an `ExecTimeout` once the call's timeout elapses.
#[tokio::test]
async fn a_stalled_session_create_times_out() {
    let server = delayed_server(|request| {
        if request.method == "POST" && request.path == format!("{TOOLBOX}/process/session") {
            return (3_000, MockResponse::json(json!({})));
        }
        (0, MockResponse::json(json!({})))
    })
    .await;
    let session = session(&server).await;
    let began = Instant::now();

    let error = session
        .pty_exec_start(request(false, Some(Duration::from_millis(150))))
        .await
        .expect_err("the session create stalled");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error:?}");
    assert!(began.elapsed() < Duration::from_secs(2));
    assert_eq!(deletes(&server), 1, "the session that may exist is removed");
    let after = session
        .pty_write_stdin(PtyWriteRequest::new(1_000, ""))
        .await
        .expect_err("nothing registered");
    assert_eq!(after.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// A stalled command start is an `ExecTimeout`, and the session it was created in is deleted.
#[tokio::test]
async fn a_stalled_command_start_times_out_and_removes_its_session() {
    let server = delayed_server(|request| {
        if request.method == "POST" && request.path.ends_with("/exec") {
            return (3_000, cmd_started());
        }
        (0, MockResponse::json(json!({})))
    })
    .await;
    let session = session(&server).await;

    let error = session
        .pty_exec_start(request(false, Some(Duration::from_millis(150))))
        .await
        .expect_err("the start stalled");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error:?}");
    assert_eq!(deletes(&server), 1);
    server.stop();
}

/// A terminal whose connection never answers is an `ExecTimeout`, and nothing stays registered.
#[tokio::test]
async fn a_stalled_terminal_connection_times_out() {
    let server = delayed_server(|request| {
        if request.path.ends_with("/process/pty/create-connect") {
            return (
                3_000,
                MockResponse::status(500, json!({ "message": "late" })),
            );
        }
        (0, MockResponse::json(json!({})))
    })
    .await;
    let session = session(&server).await;
    let began = Instant::now();

    let error = session
        .pty_exec_start(request(true, Some(Duration::from_millis(200))))
        .await
        .expect_err("the connection stalled");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error:?}");
    assert!(began.elapsed() < Duration::from_secs(2));
    let after = session
        .pty_write_stdin(PtyWriteRequest::new(1_000, ""))
        .await
        .expect_err("nothing registered");
    assert_eq!(after.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// A timeout that is not reached changes nothing, and a start without one waits for a slow step.
#[tokio::test]
async fn a_start_inside_its_timeout_or_without_one_succeeds() {
    for timeout in [Some(Duration::from_secs(5)), None] {
        let server = delayed_server(|request| {
            if request.method == "POST" && request.path == format!("{TOOLBOX}/process/session") {
                return (300, MockResponse::json(json!({})));
            }
            if request.method == "POST" && request.path.ends_with("/exec") {
                return (0, cmd_started());
            }
            (0, MockResponse::json(json!({ "exitCode": 0 })))
        })
        .await;
        let session = session(&server).await;

        let started = session
            .pty_exec_start(request(false, timeout))
            .await
            .expect("the start finishes");

        assert!(
            started.process_id.is_some() || started.exit_code.is_some(),
            "{started:?}"
        );
        session.pty_terminate_all().await.expect("terminate");
        server.stop();
    }
}

/// A zero timeout is the smallest bound, so any real step outlasts it.
#[tokio::test]
async fn a_zero_timeout_expires_at_once() {
    let server = delayed_server(|request| {
        if request.method == "POST" && request.path == format!("{TOOLBOX}/process/session") {
            return (200, MockResponse::json(json!({})));
        }
        (0, MockResponse::json(json!({})))
    })
    .await;
    let session = session(&server).await;

    let error = session
        .pty_exec_start(request(false, Some(Duration::ZERO)))
        .await
        .expect_err("a zero budget");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error:?}");
    server.stop();
}
