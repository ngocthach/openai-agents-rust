//! E2B cases of `tests/extensions/sandbox/test_e2b.py` that `py_sandbox_e2b.rs` does not port:
//! PTY sessions, exec error classification, resume lifecycle, persist excludes and native
//! snapshots. Python replaces the vendor SDK with fakes; the Rust backend speaks the envd Connect
//! protocol and the platform REST API, so the same behaviour is observed through a local mock
//! server that records every request and streams Connect frames.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use openai_agents_sdk::extensions::sandbox::ProviderMountStrategy;
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BCloudBucketMountStrategy, E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxTimeouts,
    E2BTimeoutAction, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, ExecShell, Manifest, PtyExecRequest, PtyWriteRequest, SandboxClient,
    SandboxSession, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn connect_frame(flags: u8, value: Value) -> Vec<u8> {
    let payload = serde_json::to_vec(&value).expect("serializable");
    let mut frame = vec![flags];
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn start_event(pid: u64) -> Vec<u8> {
    connect_frame(0, json!({ "event": { "start": { "pid": pid } } }))
}

fn data_event(field: &str, text: &str) -> Vec<u8> {
    connect_frame(
        0,
        json!({ "event": { "data": { field: support::b64(text) } } }),
    )
}

fn end_event(exit_code: i64) -> Vec<u8> {
    let mut frames = connect_frame(
        0,
        json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
    );
    frames.extend(connect_frame(2, json!({})));
    frames
}

/// A `process.Process/Start` reply that delivers `parts`, each after its delay in milliseconds.
fn stream(parts: Vec<(Vec<u8>, u64)>) -> MockResponse {
    MockResponse::streamed("application/connect+json", parts)
}

/// A reply for a command that prints `stdout` and exits with `exit_code`.
fn finished(stdout: &str, exit_code: i64) -> MockResponse {
    let mut body = start_event(4242);
    if !stdout.is_empty() {
        body.extend(data_event("stdout", stdout));
    }
    body.extend(end_event(exit_code));
    stream(vec![(body, 0)])
}

fn created() -> Value {
    json!({ "sandboxID": "sb-123", "envdAccessToken": "envd-token" })
}

fn options(server: &MockServer) -> E2BSandboxClientOptions {
    E2BSandboxClientOptions::new()
        .with_api_key("e2b_test")
        .with_api_base_url(&server.base_url)
        .with_sandbox_base_url(&server.base_url)
}

fn client(server: &MockServer) -> E2BSandboxClient {
    E2BSandboxClient::with_options(options(server))
}

/// A server answering the platform API, and `start` with `on_start` for every process.
async fn envd(on_start: impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static) -> MockServer {
    support::start_resolving(
        support::Resolver::E2b,
        move |request: &Recorded| match request.path.as_str() {
            "/v2/sandboxes" => MockResponse::json(created()),
            "/process.Process/Start" => on_start(request),
            _ => MockResponse::json(json!({})),
        },
    )
    .await
}

fn start_body(request: &Recorded) -> Value {
    serde_json::from_slice(&request.body[5..]).expect("a JSON start request")
}

fn pty_request(command: &str, tty: bool, yield_ms: u64) -> PtyExecRequest {
    PtyExecRequest::new(command)
        .with_shell(ExecShell::None)
        .with_tty(tty)
        .with_yield_time_ms(yield_ms)
}

fn tar_with_one_file() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(2);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, "a.txt", &b"hi"[..])
        .expect("append");
    builder.into_inner().expect("tar")
}

// ---------------------------------------------------------------------------------------------
// Mount strategy
// ---------------------------------------------------------------------------------------------

/// `test_e2b_mount_strategy_round_trips_through_manifest`
#[test]
fn e2b_mount_strategy_round_trips_through_the_provider_registry() {
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let payload =
        serde_json::to_value(ProviderMountStrategy::E2bCloudBucket(strategy)).expect("serialize");
    assert_eq!(payload["type"], "e2b_cloud_bucket");

    let parsed = ProviderMountStrategy::parse(&payload).expect("parse");

    assert_eq!(parsed.strategy_type(), "e2b_cloud_bucket");
}

/// `test_e2b_session_guard_rejects_wrong_type`: a mount strategy refuses a foreign session.
#[tokio::test]
async fn e2b_mount_strategy_rejects_a_session_of_another_backend() {
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let mount = strategy.clone().mount();
    let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec).returning_exec(
        ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        },
    )]);

    let error = strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .expect_err("not an E2B session");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("requires a E2B sandbox session"),
        "{}",
        error.message
    );
}

/// `test_e2b_session_guard_accepts_correct_type`: a real E2B session passes the guard, so
/// whatever fails later is not the session-type refusal.
#[tokio::test]
async fn e2b_mount_strategy_accepts_an_e2b_session() {
    let server = envd(|_| finished("", 0)).await;
    let session = client(&server).create(None).await.expect("create");
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let mount = strategy.clone().mount();

    let outcome = strategy
        .activate(&mount, &*session, "/workspace/data")
        .await;

    if let Err(error) = outcome {
        assert!(
            !error.message.contains("requires a E2B sandbox session"),
            "{}",
            error.message
        );
    }
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------------------------

fn resumable_state() -> SandboxSessionState {
    SandboxSessionState::new("e2b", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("sb-persisted".to_string()))
}

/// `test_e2b_resume_reuses_paused_timeout_lifecycle_sandbox` and
/// `test_e2b_resume_reuses_live_kill_timeout_sandbox`: the reconnect carries the configured
/// timeout, reuses the sandbox without creating one, and marks its system state preserved.
#[tokio::test]
async fn e2b_resume_reconnects_with_the_timeout_and_marks_state_preserved() {
    for on_timeout in [E2BTimeoutAction::Pause, E2BTimeoutAction::Kill] {
        let server =
            support::start_resolving(
                support::Resolver::E2b,
                move |request: &Recorded| match request.path.as_str() {
                    "/v2/sandboxes/sb-persisted/connect" => {
                        MockResponse::json(json!({ "sandboxID": "sb-persisted" }))
                    }
                    _ => MockResponse::json(created()),
                },
            )
            .await;
        let client = E2BSandboxClient::with_options(
            options(&server)
                .with_timeout_secs(15)
                .with_on_timeout(on_timeout),
        );

        let session = client
            .resume(&resumable_state(), Manifest::new())
            .await
            .expect("resume");

        assert_eq!(
            server
                .find("POST", "/v2/sandboxes/sb-persisted/connect")
                .json(),
            json!({ "timeout": 15 })
        );
        assert!(server.find_opt("POST", "/v2/sandboxes").is_none());
        assert_eq!(
            session.session_state().provider_id.as_deref(),
            Some("sb-persisted")
        );
        assert!(session.system_state_preserved_on_start());
        server.stop();
    }
}

/// `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`: the replacement is
/// created with the kill lifecycle and the MCP servers, and carries no preserved state.
#[tokio::test]
async fn e2b_resume_recreates_a_dead_sandbox_with_the_kill_lifecycle_and_mcp() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes/sb-persisted/connect" => MockResponse::empty(404),
                "/v2/sandboxes" => MockResponse::json(json!({ "sandboxID": "sb-recreated" })),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let client = E2BSandboxClient::with_options(
        options(&server)
            .with_timeout_secs(15)
            .with_on_timeout(E2BTimeoutAction::Kill)
            .with_mcp_server("exa", [("apiKey", "exa-key")]),
    );

    let session = client
        .resume(&resumable_state(), Manifest::new())
        .await
        .expect("resume");

    let body = server.find("POST", "/v2/sandboxes").json();
    assert_eq!(body["timeout"], 15);
    assert_eq!(body["autoPause"], false);
    assert!(body.get("autoResume").is_none(), "{body}");
    assert_eq!(body["mcp"], json!({ "exa": { "apiKey": "exa-key" } }));
    assert_eq!(
        session.session_state().provider_id.as_deref(),
        Some("sb-recreated")
    );
    assert!(!session.system_state_preserved_on_start());
    server.stop();
}

/// `test_e2b_preserved_start_still_prepares_workspace_root_for_resumed_exec_cwd`: a resumed
/// session still creates the workspace root before the first command runs there.
#[tokio::test]
async fn e2b_preserved_start_still_prepares_the_workspace_root() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes/sb-persisted/connect" => {
                    MockResponse::json(json!({ "sandboxID": "sb-persisted" }))
                }
                // The resumed sandbox lost its workspace root: the `test -d` probe fails.
                "/process.Process/Start" if start_body(request)["process"]["cmd"] == "test" => {
                    finished("", 1)
                }
                "/process.Process/Start" => finished("", 0),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server)
        .resume(&resumable_state(), Manifest::new())
        .await
        .expect("resume");

    session.start().await.expect("start");
    session
        .exec(ExecRequest::argv(["pwd"]))
        .await
        .expect("exec");

    assert!(server.count("POST", "/filesystem.Filesystem/MakeDir") >= 1);
    let bodies: Vec<Value> = server
        .all("POST", "/process.Process/Start")
        .iter()
        .map(start_body)
        .collect();
    assert_eq!(bodies[0]["process"]["cmd"], "test");
    assert_eq!(bodies[1]["process"]["cmd"], "mkdir");
    assert_eq!(bodies[1]["process"]["cwd"], "/");
    assert_eq!(bodies.last().expect("pwd")["process"]["cwd"], "/workspace");
    assert!(session.system_state_preserved_on_start());
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Persist and hydrate
// ---------------------------------------------------------------------------------------------

fn tar_command(server: &MockServer) -> String {
    let body = start_body(
        server
            .all("POST", "/process.Process/Start")
            .last()
            .expect("a command"),
    );
    body["process"]["args"]
        .as_array()
        .and_then(|args| args.last())
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `test_e2b_persist_workspace_excludes_runtime_skip_paths`
#[tokio::test]
async fn e2b_persist_workspace_excludes_runtime_skip_paths() {
    let archive = support::b64(b"fake-tar-bytes");
    let server = envd(move |_| finished(&archive, 0)).await;
    let session = client(&server).create(None).await.expect("create");
    session
        .register_persist_workspace_skip_path("logs/events.jsonl")
        .expect("skip path");

    let persisted = session.persist_workspace().await.expect("persist");

    assert_eq!(persisted, b"fake-tar-bytes");
    let command = tar_command(&server);
    assert!(
        command.contains("--exclude ./logs/events.jsonl"),
        "{command}"
    );
    assert!(command.starts_with("tar -czf - -C /workspace"), "{command}");
    assert!(command.ends_with(". | base64 -w0"), "{command}");
    server.stop();
}

/// `test_e2b_persist_workspace_native_snapshot_falls_back_to_tar_for_plain_skip_paths`: a native
/// snapshot cannot leave a path out, so a registered skip path forces the tar archive.
#[tokio::test]
async fn e2b_native_snapshot_falls_back_to_tar_for_plain_skip_paths() {
    let archive = support::b64(b"fake-tar-bytes");
    let server = envd(move |_| finished(&archive, 0)).await;
    let session = E2BSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .create(None)
    .await
    .expect("create");
    session
        .register_persist_workspace_skip_path("logs/events.jsonl")
        .expect("skip path");

    let persisted = session.persist_workspace().await.expect("persist");

    assert_eq!(persisted, b"fake-tar-bytes");
    assert_eq!(server.count("POST", "/sandboxes/sb-123/snapshots"), 0);
    assert_eq!(server.count("POST", "/process.Process/Start"), 1);
    server.stop();
}

/// `test_e2b_persist_workspace_native_snapshot_times_out_and_remounts_mounts` (the timeout
/// half): a snapshot that outlasts the archive budget is a persist error.
#[tokio::test]
async fn e2b_native_snapshot_times_out() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.path.ends_with("/snapshots") {
            return MockResponse::streamed(
                "application/json",
                vec![(br#"{"snapshotID":"snap-1"}"#.to_vec(), 3_000)],
            );
        }
        MockResponse::json(created())
    })
    .await;
    let timeouts = E2BSandboxTimeouts {
        snapshot_tar: Duration::from_millis(100),
        ..E2BSandboxTimeouts::default()
    };
    let session = E2BSandboxClient::with_options(
        options(&server)
            .with_workspace_persistence(WorkspacePersistence::Snapshot)
            .with_timeouts(timeouts),
    )
    .create(None)
    .await
    .expect("create");

    let started = Instant::now();
    let error = session.persist_workspace().await.expect_err("too slow");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(
        error.message.contains("native_snapshot_failed"),
        "{}",
        error.message
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    server.stop();
}

/// `test_e2b_persist_workspace_uses_nested_mount_targets_and_resolved_excludes` (the exclude
/// half): every mount path is left out of the tar archive.
#[tokio::test]
async fn e2b_persist_workspace_excludes_every_mount_path() {
    let archive = support::b64(b"fake-tar-bytes");
    let server = envd(move |_| finished(&archive, 0)).await;
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let manifest = Manifest::new()
        .with_entry("repo", Entry::mount(strategy.clone().mount()))
        .with_entry("repo/sub", Entry::mount(strategy.mount()));
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    session.persist_workspace().await.expect("persist");

    let command = tar_command(&server);
    assert!(command.contains("--exclude ./repo "), "{command}");
    assert!(command.contains("--exclude ./repo/sub "), "{command}");
    server.stop();
}

/// `test_e2b_hydrate_workspace_raises_on_nonzero_extract_exit` (cleanup half): the scratch
/// archive is removed whatever the extraction outcome.
#[tokio::test]
async fn e2b_hydrate_removes_the_scratch_archive_after_a_failed_extract() {
    let server = envd(|_| finished("", 2)).await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .hydrate_workspace(&tar_with_one_file())
        .await
        .expect_err("extract failed");

    assert_eq!(error.code, ErrorCode::SnapshotRestoreError);
    assert_eq!(server.count("POST", "/filesystem.Filesystem/Remove"), 1);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Exec error classification
// ---------------------------------------------------------------------------------------------

async fn failing_exec(
    status: u16,
    body: Value,
) -> Result<(), openai_agents_sdk::sandbox::SandboxError> {
    let server = envd(move |_| MockResponse::status(status, body.clone())).await;
    let session = client(&server).create(None).await.expect("create");
    let outcome = session
        .exec(ExecRequest::argv(["python3", "build.py"]))
        .await;
    server.stop();
    outcome.map(|_| ())
}

/// `test_e2b_exec_marks_rate_limit_retryable`
#[tokio::test]
async fn e2b_exec_marks_rate_limit_retryable() {
    let error = failing_exec(429, json!({ "message": "rate limit exceeded" }))
        .await
        .expect_err("rate limited");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(error.retryable, Some(true));
    assert!(
        error.message.contains("rate limit exceeded"),
        "{}",
        error.message
    );
}

/// `test_e2b_exec_marks_deterministic_provider_errors_non_retryable`
#[tokio::test]
async fn e2b_exec_marks_deterministic_provider_errors_non_retryable() {
    for status in [400u16, 401, 403] {
        let error = failing_exec(status, json!({ "message": "git authentication failed" }))
            .await
            .expect_err("deterministic");

        assert_eq!(error.code, ErrorCode::ExecTransportError, "{status}");
        assert_eq!(error.retryable, Some(false), "{status}");
        assert!(
            error.message.contains("git authentication failed"),
            "{}",
            error.message
        );
    }
}

/// `test_e2b_exec_maps_missing_sandbox_not_found_to_transport_error`
#[tokio::test]
async fn e2b_exec_maps_a_missing_sandbox_to_a_non_retryable_transport_error() {
    let error = failing_exec(
        404,
        json!({ "code": "not_found", "message": "The sandbox was not found: request failed" }),
    )
    .await
    .expect_err("not found");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(error.retryable, Some(false));
    assert_eq!(
        error.context.get("http_status").map(String::as_str),
        Some("404")
    );
    assert!(
        error
            .message
            .contains("The sandbox was not found: request failed"),
        "{}",
        error.message
    );
}

/// `test_e2b_exec_maps_httpcore_read_timeout_to_timeout_error`: a stream that stalls past the
/// command timeout is a timeout, not a transport error.
#[tokio::test]
async fn e2b_exec_maps_a_stalled_stream_to_a_timeout_error() {
    let server = envd(|_| stream(vec![(start_event(1), 0), (Vec::new(), 5_000)])).await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .exec(
            ExecRequest::argv(["python3", "build.py"])
                .with_timeout(Some(Duration::from_millis(150))),
        )
        .await
        .expect_err("stalled");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// PTY sessions
// ---------------------------------------------------------------------------------------------

async fn pty_session(server: &MockServer) -> Arc<dyn SandboxSession> {
    client(server).create(None).await.expect("create")
}

/// `test_e2b_pty_start_and_write_stdin`: a TTY session opens a pty, accepts input through
/// `SendInput`, and returns the output the process produced in response.
#[tokio::test]
async fn e2b_pty_start_and_write_stdin() {
    let mut first = start_event(77);
    first.extend(data_event("pty", ">>> "));
    let server = envd(move |_| {
        stream(vec![
            (first.clone(), 0),
            (data_event("pty", "10\n"), 700),
            (Vec::new(), 4_000),
        ])
    })
    .await;
    let session = pty_session(&server).await;

    let started = session
        .pty_exec_start(pty_request("python3", true, 300))
        .await
        .expect("start");

    let process_id = started.process_id.expect("still running");
    assert!(
        started.output_text().contains(">>>"),
        "{:?}",
        started.output_text()
    );
    let request = start_body(&server.find("POST", "/process.Process/Start"));
    assert_eq!(request["process"]["cmd"], "python3");
    assert_eq!(request["stdin"], true);
    assert!(request["pty"]["size"].is_object(), "{request}");

    let updated = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "5 + 5\n").with_yield_time_ms(1500))
        .await
        .expect("write");

    assert_eq!(updated.process_id, Some(process_id));
    assert!(
        updated.output_text().contains("10"),
        "{:?}",
        updated.output_text()
    );
    let input = server.find("POST", "/process.Process/SendInput").json();
    assert_eq!(input["process"]["pid"], 77);
    assert_eq!(input["input"]["pty"], support::b64("5 + 5\n"));
    session.pty_terminate_all().await.expect("terminate");
    let signal = server.find("POST", "/process.Process/SendSignal").json();
    assert_eq!(signal["process"]["pid"], 77);
    server.stop();
}

/// `test_e2b_pty_start_non_tty_uses_commands_run_in_background`: a non-TTY command opens no
/// stdin, returns what it printed, and reports the exit code once it finished.
#[tokio::test]
async fn e2b_pty_start_non_tty_returns_output_without_stdin() {
    let server = envd(|_| finished("started\n", 0)).await;
    let session = pty_session(&server).await;

    let started = session
        .pty_exec_start(pty_request("python3", false, 300))
        .await
        .expect("start");

    assert_eq!(started.process_id, None);
    assert_eq!(started.exit_code, Some(0));
    assert_eq!(started.output_text(), "started\n");
    let request = start_body(&server.find("POST", "/process.Process/Start"));
    assert_eq!(request["stdin"], false);
    assert_eq!(request["process"]["cwd"], "/workspace");
    server.stop();
}

/// `test_e2b_pty_start_non_tty_wakes_when_exit_follows_last_output`,
/// `test_e2b_pty_start_tty_wakes_when_session_exits_after_output`,
/// `test_e2b_pty_start_non_tty_wakes_on_quiet_exit` and
/// `test_e2b_pty_start_non_tty_wakes_on_nonzero_wait_exit`: a process that ends does not make the
/// caller wait out the whole yield window, and its exit code is reported.
#[tokio::test]
async fn e2b_pty_start_wakes_as_soon_as_the_process_exits() {
    struct Case {
        tty: bool,
        field: &'static str,
        output: &'static str,
        exit_code: i64,
    }
    for case in [
        Case {
            tty: false,
            field: "stdout",
            output: "started\n",
            exit_code: 0,
        },
        Case {
            tty: true,
            field: "pty",
            output: "bye\n",
            exit_code: 0,
        },
        Case {
            tty: false,
            field: "stdout",
            output: "",
            exit_code: 0,
        },
        Case {
            tty: false,
            field: "stdout",
            output: "",
            exit_code: 2,
        },
    ] {
        let mut body = start_event(9);
        if !case.output.is_empty() {
            body.extend(data_event(case.field, case.output));
        }
        let end = end_event(case.exit_code);
        let server = envd(move |_| stream(vec![(body.clone(), 0), (end.clone(), 50)])).await;
        let session = pty_session(&server).await;

        let started_at = Instant::now();
        let started = session
            .pty_exec_start(pty_request("true", case.tty, 10_000))
            .await
            .expect("start");

        assert!(started_at.elapsed() < Duration::from_secs(3));
        assert_eq!(started.process_id, None);
        assert_eq!(started.exit_code, Some(case.exit_code as i32));
        assert_eq!(started.output_text(), case.output);
        assert_eq!(server.count("POST", "/process.Process/SendSignal"), 0);
        server.stop();
    }
}

/// `test_e2b_pty_start_non_tty_running_command_cleans_up_waiter`: a command still running when
/// the window closes keeps its session, and terminating it signals the process.
#[tokio::test]
async fn e2b_pty_running_command_keeps_a_session_until_terminated() {
    let mut body = start_event(31);
    body.extend(data_event("stdout", "tick\n"));
    let server = envd(move |_| stream(vec![(body.clone(), 0), (Vec::new(), 5_000)])).await;
    let session = pty_session(&server).await;

    let started = session
        .pty_exec_start(pty_request("sleep 60", false, 300))
        .await
        .expect("start");

    assert!(started.process_id.is_some());
    assert_eq!(started.exit_code, None);
    session.pty_terminate_all().await.expect("terminate");
    assert_eq!(
        server.find("POST", "/process.Process/SendSignal").json()["process"]["pid"],
        31
    );
    server.stop();
}

/// `test_e2b_pty_write_stdin` family: an unknown session and a non-TTY session refuse input.
#[tokio::test]
async fn e2b_pty_write_stdin_refuses_unknown_and_non_tty_sessions() {
    let mut body = start_event(5);
    body.extend(data_event("stdout", "x"));
    let server = envd(move |_| stream(vec![(body.clone(), 0), (Vec::new(), 5_000)])).await;
    let session = pty_session(&server).await;
    let started = session
        .pty_exec_start(pty_request("cat", false, 300))
        .await
        .expect("start");

    let unknown = session
        .pty_write_stdin(PtyWriteRequest::new(999_999, ""))
        .await
        .expect_err("unknown");
    let no_stdin = session
        .pty_write_stdin(PtyWriteRequest::new(
            started.process_id.expect("id"),
            "hello",
        ))
        .await
        .expect_err("no stdin");

    assert_eq!(unknown.code, ErrorCode::PtySessionNotFound);
    assert_eq!(no_stdin.code, ErrorCode::PtyStdinUnavailable);
    session.pty_terminate_all().await.expect("terminate");
    server.stop();
}

/// `test_e2b_pty_start_wraps_startup_failures`
#[tokio::test]
async fn e2b_pty_start_wraps_startup_failures() {
    let server = envd(|_| MockResponse::status(500, json!({ "message": "missing-shell" }))).await;
    let session = pty_session(&server).await;

    let error = session
        .pty_exec_start(pty_request("python3", true, 300))
        .await
        .expect_err("startup failed");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("missing-shell"), "{}", error.message);
    server.stop();
}

/// `test_e2b_pty_start_maps_missing_sandbox_not_found_to_transport_error`
#[tokio::test]
async fn e2b_pty_start_maps_a_missing_sandbox_to_a_non_retryable_transport_error() {
    let server = envd(|_| {
        MockResponse::status(
            404,
            json!({ "code": "not_found", "message": "The sandbox was not found: request failed" }),
        )
    })
    .await;
    let session = pty_session(&server).await;

    let error = session
        .pty_exec_start(pty_request("python3", true, 300))
        .await
        .expect_err("not found");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(error.retryable, Some(false));
    assert!(
        error
            .message
            .contains("The sandbox was not found: request failed"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_e2b_stop_terminates_live_pty_sessions`
#[tokio::test]
async fn e2b_stop_terminates_live_pty_sessions() {
    let mut body = start_event(88);
    body.extend(data_event("pty", "$ "));
    let server = envd(move |_| stream(vec![(body.clone(), 0), (Vec::new(), 5_000)])).await;
    let session = pty_session(&server).await;
    let started = session
        .pty_exec_start(pty_request("bash", true, 300))
        .await
        .expect("start");
    assert!(started.process_id.is_some());

    session.stop().await.expect("stop");

    assert_eq!(
        server.find("POST", "/process.Process/SendSignal").json()["process"]["pid"],
        88
    );
    assert_eq!(server.count("POST", "/sandboxes/sb-123/pause"), 1);
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(started.process_id.expect("id"), ""))
        .await
        .expect_err("terminated");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// `test_e2b_pty_start_cleans_up_partially_created_session_on_failure`: a session whose
/// startup failed leaves nothing registered, so a later write cannot reach it.
#[tokio::test]
async fn e2b_pty_start_failure_registers_no_session() {
    let server = envd(|_| MockResponse::status(500, json!({ "message": "boom" }))).await;
    let session = pty_session(&server).await;
    let _ = session
        .pty_exec_start(pty_request("python3", true, 300))
        .await
        .expect_err("startup failed");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(1_000, ""))
        .await
        .expect_err("nothing registered");

    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    server.stop();
}
