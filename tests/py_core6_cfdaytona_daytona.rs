//! Daytona cases of `tests/extensions/sandbox/test_daytona.py` that `py_sandbox_daytona.rs` does
//! not port: PTY terminals and logged commands, the HTTP status table, persist excludes and
//! failures, exec budgets, resume state and the mount strategy lifecycle. Python patches the
//! vendor SDK; the Rust backend speaks the platform and toolbox HTTP APIs, so the same behaviour is
//! observed through a local mock server that records requests and speaks websocket frames.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::{SinkExt as _, StreamExt as _};
use openai_agents_sdk::extensions::sandbox::ProviderMountStrategy;
use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaCloudBucketMountStrategy, DaytonaSandboxClient, DaytonaSandboxClientOptions,
    DaytonaSandboxTimeouts,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, ExecShell, Manifest, PtyExecRequest, PtyWriteRequest, S3Mount,
    SandboxClient, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

const TOOLBOX: &str = "/toolbox/sbx-daytona/toolbox";

fn command_result(stdout: &str, exit_code: i64) -> MockResponse {
    MockResponse::json(json!({
        "output": stdout,
        "stdout": stdout,
        "stderr": "",
        "exitCode": exit_code,
    }))
}

fn sandbox_started() -> MockResponse {
    MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
}

fn options(server: &MockServer) -> DaytonaSandboxClientOptions {
    DaytonaSandboxClientOptions::new()
        .with_api_key("dtn_test")
        .with_api_url(&server.base_url)
        .with_organization_id("org-1")
        .with_state_poll_interval(Duration::from_millis(1))
}

fn client(server: &MockServer) -> DaytonaSandboxClient {
    DaytonaSandboxClient::with_options(options(server))
}

/// The platform routes every test shares; `toolbox` answers everything else.
fn platform(request: &Recorded) -> Option<MockResponse> {
    let path = request.path.as_str();
    if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
        return Some(sandbox_started());
    }
    None
}

async fn server(toolbox: impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static) -> MockServer {
    support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        platform(request).unwrap_or_else(|| toolbox(request))
    })
    .await
}

/// Like [`server`], but the toolbox handler also chooses how long to wait before answering.
async fn delayed_server(
    toolbox: impl Fn(&Recorded) -> (u64, MockResponse) + Send + Sync + 'static,
) -> MockServer {
    let toolbox = Arc::new(toolbox);
    MockServer::start_raw(move |request, mut stream| {
        let toolbox = toolbox.clone();
        async move {
            let (delay_ms, response) = match platform(&request) {
                Some(response) => (0, response),
                None => toolbox(&request),
            };
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let _ = support::write_response(&mut stream, response).await;
        }
    })
    .await
}

fn commands(server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/exec"))
        .map(|r| r.json()["command"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn pty_request(command: &str, tty: bool, yield_ms: u64) -> PtyExecRequest {
    PtyExecRequest::new(command)
        .with_shell(ExecShell::None)
        .with_tty(tty)
        .with_yield_time_ms(yield_ms)
}

fn resumable_state() -> SandboxSessionState {
    SandboxSessionState::new(
        "daytona",
        "session-1",
        "/home/daytona/workspace",
        &Manifest::new().with_root("/home/daytona/workspace"),
    )
    .with_provider_id(Some("sbx-daytona".to_string()))
}

// ---------------------------------------------------------------------------------------------
// Retryability table
// ---------------------------------------------------------------------------------------------

/// `test_daytona_retryability_status_table`: a failed file read carries the documented
/// classification of its HTTP status (a 404 is a missing file, which is its own error).
#[tokio::test]
async fn retryability_status_table() {
    for (status, expected) in [
        (400u16, Some(false)),
        (401, Some(false)),
        (403, Some(false)),
        (409, Some(false)),
        (429, Some(true)),
        (500, Some(true)),
        (502, Some(true)),
        (503, Some(true)),
        (504, Some(true)),
        (418, None),
    ] {
        let server = server(move |request| {
            if request.path.ends_with("/files/download") {
                return support::MockResponse::error(status, "denied");
            }
            command_result("", 0)
        })
        .await;
        let session = client(&server).create(None).await.expect("create");

        let error = session
            .read_file("a.txt")
            .await
            .expect_err("the toolbox refused");

        assert_eq!(error.retryable, expected, "status {status}: {error:?}");
        assert_eq!(
            error.context.get("http_status").map(String::as_str),
            Some(status.to_string().as_str())
        );
        server.stop();
    }

    let server = server(|request| {
        if request.path.ends_with("/files/download") {
            return MockResponse::error(404, "missing");
        }
        command_result("", 0)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let error = session.read_file("a.txt").await.expect_err("missing");
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Exec budgets
// ---------------------------------------------------------------------------------------------

/// `test_exec_timeout_budget_includes_session_create`: the caller's timeout covers creating the
/// toolbox session, not only running the command in it.
#[tokio::test]
async fn exec_timeout_budget_includes_session_create() {
    let server = delayed_server(|request| {
        if request.method == "POST" && request.path == format!("{TOOLBOX}/process/session") {
            return (600, MockResponse::json(json!({})));
        }
        (0, command_result("done", 0))
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["echo", "done"]).with_timeout(Some(Duration::from_millis(200))))
        .await
        .expect_err("the session create ate the budget");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    server.stop();
}

/// `test_exec_delete_session_cleanup_is_bounded`: a toolbox that never answers the session
/// delete cannot hold the command's result back beyond the cleanup budget.
#[tokio::test]
async fn exec_delete_session_cleanup_is_bounded() {
    let server = delayed_server(|request| {
        if request.method == "DELETE" && request.path.contains("/process/session/") {
            return (10_000, MockResponse::json(json!({})));
        }
        (0, command_result("done", 0))
    })
    .await;
    let timeouts = DaytonaSandboxTimeouts {
        cleanup: Duration::from_millis(150),
        ..DaytonaSandboxTimeouts::default()
    };
    let session = DaytonaSandboxClient::with_options(options(&server).with_timeouts(timeouts))
        .create(None)
        .await
        .expect("create");

    let started = Instant::now();
    let result = session
        .exec(ExecRequest::argv(["echo", "done"]).with_timeout(Some(Duration::from_secs(5))))
        .await
        .expect("exec");

    assert_eq!(result.stdout_text(), "done");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------------------------

/// `test_resume_reconnects_paused_sandbox_and_preserves_state`,
/// `test_resume_reconnects_unpaused_live_sandbox_after_unclean_worker_exit` and
/// `test_resume_recreates_unpaused_sandbox_when_reconnect_fails`: a sandbox that is found keeps
/// its operating-system state, a replacement does not.
#[tokio::test]
async fn resume_marks_system_state_preserved_only_when_the_sandbox_is_found() {
    for found in [true, false] {
        let server =
            support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
                if request.method == "GET" && request.path == "/sandbox/sbx-daytona" && !found {
                    return MockResponse::empty(404);
                }
                if request.path == "/sandbox" {
                    return MockResponse::json(json!({ "id": "sbx-new", "state": "started" }));
                }
                MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
            })
            .await;

        let session = client(&server)
            .resume(
                &resumable_state(),
                Manifest::new().with_root("/home/daytona/workspace"),
            )
            .await
            .expect("resume");

        assert_eq!(session.system_state_preserved_on_start(), found);
        assert_eq!(
            server.find_opt("POST", "/sandbox").is_some(),
            !found,
            "{:?}",
            server.routes()
        );
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Persist
// ---------------------------------------------------------------------------------------------

/// `test_persist_workspace_uses_nested_mount_targets_and_runtime_skip_paths` (the exclude half):
/// every mount path and every registered skip path stay out of the archive.
#[tokio::test]
async fn persist_workspace_excludes_mounts_and_runtime_skip_paths() {
    let server = server(|request| {
        if request.path.ends_with("/files/download") {
            return MockResponse::bytes(b"fake-tar-bytes".to_vec());
        }
        command_result("", 0)
    })
    .await;
    let strategy = DaytonaCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let manifest = Manifest::new()
        .with_root("/home/daytona/workspace")
        .with_entry("repo", Entry::mount(strategy.clone().mount()))
        .with_entry("nested/child", Entry::mount(strategy.mount()));
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");
    session
        .register_persist_workspace_skip_path("runtime.tmp")
        .expect("skip path");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, b"fake-tar-bytes");
    let tar = commands(&server)
        .into_iter()
        .find(|c| c.contains("tar -czf"))
        .expect("a tar command");
    assert!(tar.contains("--exclude ./repo "), "{tar}");
    assert!(tar.contains("--exclude ./nested/child "), "{tar}");
    assert!(tar.contains("--exclude ./runtime.tmp "), "{tar}");
    server.stop();
}

/// `test_persist_workspace_marks_stopped_sandbox_non_retryable`: a stopped sandbox is reported as
/// such and is not retried.
#[tokio::test]
async fn persist_workspace_marks_a_stopped_sandbox_non_retryable() {
    let attempts = support::Counter::new();
    let counted = attempts.clone();
    let server = server(move |request| {
        if request.path.ends_with("/exec") {
            counted.next("exec");
            return MockResponse::status(
                400,
                json!({
                    "message": "bad request: failed to resolve container IP after 3 attempts: \
                                no IP address found. Is the Sandbox started?"
                }),
            );
        }
        command_result("", 0)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session.persist_workspace().await.expect_err("stopped");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert_eq!(error.retryable, Some(false));
    assert_eq!(
        error.context.get("backend").map(String::as_str),
        Some("daytona")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("sandbox_not_running")
    );
    assert_eq!(attempts.get("exec"), 1, "a stopped sandbox is not retried");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Mount strategy lifecycle
// ---------------------------------------------------------------------------------------------

/// `test_daytona_mount_strategy_round_trips_through_manifest`
#[test]
fn mount_strategy_round_trips_through_the_provider_registry() {
    let strategy = DaytonaCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let payload = serde_json::to_value(ProviderMountStrategy::DaytonaCloudBucket(strategy))
        .expect("serialize");
    assert_eq!(payload["type"], "daytona_cloud_bucket");

    let parsed = ProviderMountStrategy::parse(&payload).expect("parse");

    assert_eq!(parsed.strategy_type(), "daytona_cloud_bucket");
}

fn public_bucket() -> (
    DaytonaCloudBucketMountStrategy,
    openai_agents_sdk::sandbox::Mount,
) {
    let strategy = DaytonaCloudBucketMountStrategy::for_mount(S3Mount::new("public-bucket"));
    let mount = strategy.clone().mount();
    (strategy, mount)
}

fn is_preflight(command: &str) -> bool {
    command.contains("/dev/fuse") || command.contains("command -v rclone")
}

/// `test_daytona_session_guard_accepts_correct_type` and `test_activate_calls_preflights_and_delegates`:
/// a Daytona session passes the guard, the preflights run first, and the mount follows.
#[tokio::test]
async fn activate_runs_the_preflights_then_delegates() {
    let server = server(|_| command_result("", 0)).await;
    let session = client(&server).create(None).await.expect("create");
    let (strategy, mount) = public_bucket();
    let before = commands(&server).len();

    let outcome = strategy
        .activate(&mount, &*session, "/home/daytona/workspace/bucket")
        .await;

    if let Err(error) = &outcome {
        assert!(
            !error.message.contains("requires a Daytona sandbox session"),
            "{}",
            error.message
        );
    }
    let ran = commands(&server).split_off(before);
    assert!(
        ran.iter().any(|c| c.contains("/dev/fuse")),
        "the fuse probe runs: {ran:?}"
    );
    assert!(
        ran.iter().any(|c| c.contains("rclone")),
        "the rclone probe or mount runs: {ran:?}"
    );
    let first_probe = ran
        .iter()
        .position(|c| is_preflight(c))
        .expect("a preflight");
    let first_mount = ran
        .iter()
        .position(|c| c.contains("rclone mount") || c.contains("rclone nfsmount"));
    if let Some(first_mount) = first_mount {
        assert!(first_probe < first_mount, "{ran:?}");
    }
    server.stop();
}

/// `test_activate_rejects_credentials_before_preflights` and
/// `test_restore_after_snapshot_rejects_credentials_before_preflights`: a credentialed mount
/// that was never acknowledged is refused before the sandbox is asked anything.
#[tokio::test]
async fn credentials_are_rejected_before_any_preflight() {
    let server = server(|_| command_result("", 0)).await;
    let session = client(&server).create(None).await.expect("create");
    let strategy = DaytonaCloudBucketMountStrategy::for_mount(
        S3Mount::new("bucket").with_credentials("access-key", "secret-key"),
    );
    let mount = strategy.clone().mount();
    let before = commands(&server).len();

    let activate = strategy
        .activate(&mount, &*session, "/home/daytona/workspace/bucket")
        .await
        .expect_err("unacknowledged credentials");
    let restore = strategy
        .restore_after_snapshot(&mount, &*session, "/home/daytona/workspace/bucket")
        .await
        .expect_err("unacknowledged credentials");

    assert_eq!(activate.code, ErrorCode::MountConfigInvalid);
    assert_eq!(restore.code, ErrorCode::MountConfigInvalid);
    assert_eq!(commands(&server).len(), before, "no command was run");
    assert!(!format!("{activate:?}{restore:?}").contains("secret-key"));
    server.stop();
}

/// `test_deactivate_delegates_without_preflights` and `test_teardown_delegates_without_preflights`
#[tokio::test]
async fn deactivate_and_teardown_skip_the_preflights() {
    let server = server(|_| command_result("", 0)).await;
    let session = client(&server).create(None).await.expect("create");
    let (strategy, mount) = public_bucket();
    let path = "/home/daytona/workspace/bucket";
    let before = commands(&server).len();

    let deactivated = strategy.deactivate(&mount, &*session, path).await;
    let torn_down = strategy
        .teardown_for_snapshot(&mount, &*session, path)
        .await;

    let ran = commands(&server).split_off(before);
    assert!(deactivated.is_ok(), "{deactivated:?}");
    assert!(torn_down.is_ok(), "{torn_down:?}");
    assert!(!ran.is_empty(), "the unmount ran");
    assert!(
        ran.iter().all(|c| !is_preflight(c)),
        "no preflight before an unmount: {ran:?}"
    );
    server.stop();
}

/// `test_restore_after_snapshot_reruns_preflights`
#[tokio::test]
async fn restore_after_snapshot_reruns_the_preflights() {
    let server = server(|_| command_result("", 0)).await;
    let session = client(&server).create(None).await.expect("create");
    let (strategy, mount) = public_bucket();
    let before = commands(&server).len();

    let _ = strategy
        .restore_after_snapshot(&mount, &*session, "/home/daytona/workspace/bucket")
        .await;

    let ran = commands(&server).split_off(before);
    assert!(ran.iter().any(|c| c.contains("/dev/fuse")), "{ran:?}");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Logged (non-TTY) commands
// ---------------------------------------------------------------------------------------------

fn logged_server(
    logs: MockResponse,
    status: MockResponse,
) -> impl std::future::Future<Output = MockServer> {
    let logs = Arc::new(Mutex::new(logs));
    let status = Arc::new(Mutex::new(status));
    server(move |request| {
        let path = request.path.as_str();
        if path.ends_with("/exec") {
            return MockResponse::json(json!({ "cmdId": "cmd-1" }));
        }
        if path.ends_with("/command/cmd-1/logs") {
            return clone_response(&logs.lock().unwrap());
        }
        if path.ends_with("/command/cmd-1") && request.method == "GET" {
            return clone_response(&status.lock().unwrap());
        }
        MockResponse::json(json!({}))
    })
}

fn clone_response(response: &MockResponse) -> MockResponse {
    MockResponse {
        status: response.status,
        content_type: response.content_type,
        body: response.body.clone(),
        stream: response.stream.clone(),
        headers: response.headers.clone(),
    }
}

/// `test_session_reader_keeps_entry_live_when_logs_fail_without_exit_code`: a log endpoint that
/// fails while the command has no exit code leaves the session live.
#[tokio::test]
async fn logged_command_stays_live_when_logs_fail_without_an_exit_code() {
    let server = logged_server(
        MockResponse::status(500, json!({ "message": "logs failed" })),
        MockResponse::json(json!({ "id": "cmd-1" })),
    )
    .await;
    let session = client(&server).create(None).await.expect("create");

    let started = session
        .pty_exec_start(pty_request("sleep 60", false, 400))
        .await
        .expect("start");

    assert!(started.process_id.is_some(), "{started:?}");
    assert_eq!(started.exit_code, None);
    session.pty_terminate_all().await.expect("terminate");
    server.stop();
}

/// `test_session_reader_closes_entry_when_logs_fail_after_known_exit`: once the command is known
/// to have exited, the session closes with that exit code even though the logs failed.
#[tokio::test]
async fn logged_command_closes_when_logs_fail_after_a_known_exit() {
    let server = logged_server(
        MockResponse::status(500, json!({ "message": "logs failed" })),
        MockResponse::json(json!({ "id": "cmd-1", "exitCode": 3 })),
    )
    .await;
    let session = client(&server).create(None).await.expect("create");

    let started = session
        .pty_exec_start(pty_request("false", false, 2_000))
        .await
        .expect("start");

    assert_eq!(started.process_id, None);
    assert_eq!(started.exit_code, Some(3));
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// PTY terminals over the toolbox websocket
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct TerminalLog {
    received: Vec<String>,
    closed: bool,
    query: String,
}

fn control(status: &str, extra: Value) -> Message {
    let mut value = json!({ "type": "control", "status": status });
    if let (Some(object), Some(extra)) = (value.as_object_mut(), extra.as_object()) {
        object.extend(extra.clone());
    }
    Message::Text(value.to_string().into())
}

/// A toolbox whose terminal greets with `>>> `, answers `5 + 5` with `10` and exits on `exit`.
async fn terminal_server() -> (MockServer, Arc<Mutex<Vec<Arc<Mutex<TerminalLog>>>>>) {
    let logs: Arc<Mutex<Vec<Arc<Mutex<TerminalLog>>>>> = Arc::new(Mutex::new(Vec::new()));
    let served = logs.clone();
    let server = MockServer::start_raw(move |request, mut stream| {
        let served = served.clone();
        async move {
            let upgrade = request
                .header("upgrade")
                .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
            if !upgrade {
                let response = platform(&request).unwrap_or_else(|| command_result("", 0));
                let _ = support::write_response(&mut stream, response).await;
                return;
            }
            let log = Arc::new(Mutex::new(TerminalLog {
                query: request.query.clone(),
                ..TerminalLog::default()
            }));
            served.lock().unwrap().push(log.clone());
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
                        log.lock().unwrap().received.push(text.clone());
                        if text.contains("python3") {
                            let _ = socket.send(Message::Binary(b">>> ".to_vec().into())).await;
                        } else if text.contains("5 + 5") {
                            let _ = socket.send(Message::Binary(b"10\n".to_vec().into())).await;
                        } else if text.contains("exit") {
                            let _ = socket
                                .send(control("exited", json!({ "exitCode": 0 })))
                                .await;
                            let _ = socket.close(None).await;
                        }
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            log.lock().unwrap().closed = true;
        }
    })
    .await;
    (server, logs)
}

/// `test_pty_start_write_and_exit`
#[tokio::test]
async fn pty_start_write_and_exit() {
    let (server, logs) = terminal_server().await;
    let session = client(&server).create(None).await.expect("create");

    let started = session
        .pty_exec_start(pty_request("python3", true, 400))
        .await
        .expect("start");

    let process_id = started.process_id.expect("still running");
    assert!(
        started.output_text().contains(">>>"),
        "{:?}",
        started.output_text()
    );
    let updated = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "5 + 5\n").with_yield_time_ms(400))
        .await
        .expect("write");
    assert_eq!(updated.process_id, Some(process_id));
    assert!(
        updated.output_text().contains("10"),
        "{:?}",
        updated.output_text()
    );

    let finished = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "exit\n").with_yield_time_ms(2_000))
        .await
        .expect("exit");
    assert_eq!(finished.process_id, None);
    assert_eq!(finished.exit_code, Some(0));
    let received = logs.lock().unwrap()[0].lock().unwrap().received.clone();
    assert!(received[0].contains("python3"), "{received:?}");
    assert_eq!(&received[1..], ["5 + 5\n", "exit\n"]);
    let query = logs.lock().unwrap()[0].lock().unwrap().query.clone();
    assert!(
        query.contains("cols=80") && query.contains("rows=24"),
        "{query}"
    );
    server.stop();
}

/// `test_stop_terminates_live_pty_sessions`
#[tokio::test]
async fn stop_terminates_live_pty_sessions() {
    let (server, _logs) = terminal_server().await;
    let session = client(&server).create(None).await.expect("create");
    let started = session
        .pty_exec_start(pty_request("python3", true, 400))
        .await
        .expect("start");
    assert!(started.process_id.is_some());

    session.stop().await.expect("stop");

    let killed = server
        .requests()
        .into_iter()
        .filter(|r| r.method == "DELETE" && r.path.contains("/process/pty/"))
        .count();
    assert_eq!(killed, 1, "{:?}", server.routes());
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(started.process_id.expect("id"), ""))
        .await
        .expect_err("terminated");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// `test_pty_start_wraps_startup_failures`
#[tokio::test]
async fn pty_start_wraps_startup_failures() {
    let server = server(|request| {
        if request.path.ends_with("/process/pty/create-connect") {
            return MockResponse::status(500, json!({ "message": "missing-shell" }));
        }
        command_result("", 0)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .pty_exec_start(pty_request("python3", true, 300))
        .await
        .expect_err("startup failed");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(
        error.context.get("backend").map(String::as_str),
        Some("daytona")
    );
    server.stop();
}

/// `test_pty_start_marks_documented_sdk_not_found_non_retryable` and
/// `test_pty_start_marks_documented_sdk_rate_limit_retryable`
#[tokio::test]
async fn pty_start_classifies_provider_statuses() {
    for (status, retryable) in [(404u16, Some(false)), (429, Some(true))] {
        let server = server(move |request| {
            if request.path.ends_with("/process/pty/create-connect") {
                return MockResponse::status(status, json!({ "message": "provider says no" }));
            }
            command_result("", 0)
        })
        .await;
        let session = client(&server).create(None).await.expect("create");

        let error = session
            .pty_exec_start(pty_request("python3", true, 300))
            .await
            .expect_err("refused");

        assert_eq!(error.code, ErrorCode::ExecTransportError, "{status}");
        assert_eq!(error.retryable, retryable, "{status}");
        assert_eq!(
            error.context.get("backend").map(String::as_str),
            Some("daytona")
        );
        assert_eq!(
            error.context.get("http_status").map(String::as_str),
            Some(status.to_string().as_str())
        );
        server.stop();
    }
}
