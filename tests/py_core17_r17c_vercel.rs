//! Ports of cases of `tests/extensions/sandbox/test_vercel.py` the earlier rounds filed as
//! Python-only: workspace commands wait while an S3 mount is being attached
//! (`test_vercel_s3_nested_activation_serializes_workspace_commands`), a session without S3
//! mounts never serializes them (`test_vercel_without_s3_mounts_does_not_serialize_workspace_commands`),
//! a command whose output stalls past its timeout is a timeout
//! (`test_vercel_mount_command_timeout_includes_output_collection`), and an instrumented mounted
//! session is still shut down when `aclose` fails (`test_vercel_s3_aclose_bypasses_failing_instrumentation`).
#![cfg(feature = "sandbox")]
#![allow(dead_code)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{
    CallbackSink, ErrorCode, EventPhase, ExecRequest, InstrumentedSession, Manifest, SandboxClient,
    SandboxError, SandboxSession, SandboxSessionLifecycle,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

/// One scripted answer to a command.
#[derive(Clone)]
struct Answer {
    stdout: String,
    stderr: String,
    exit_code: i64,
}

fn ok(stdout: &str) -> Answer {
    Answer {
        stdout: stdout.to_string(),
        stderr: String::new(),
        exit_code: 0,
    }
}

fn failed(stderr: &str, exit_code: i64) -> Answer {
    Answer {
        stdout: String::new(),
        stderr: stderr.to_string(),
        exit_code,
    }
}

#[derive(Default)]
struct Script {
    /// Queued answers per command, used before the defaults.
    queued: Mutex<HashMap<String, VecDeque<Answer>>>,
    /// How many `fs/write` calls fail with a 500 before one succeeds.
    write_failures: Mutex<usize>,
    /// Whether `fs/read` reports the file missing.
    read_missing: Mutex<bool>,
    /// The archive `fs/read` serves.
    archive: Mutex<Vec<u8>>,
    /// Commands whose output arrives only after this many milliseconds.
    stalled: Mutex<HashMap<String, u64>>,
}

impl Script {
    fn queue(&self, command: &str, answers: Vec<Answer>) {
        self.queued
            .lock()
            .expect("poisoned")
            .entry(command.to_string())
            .or_default()
            .extend(answers);
    }

    fn next(&self, command: &str) -> Option<Answer> {
        self.queued
            .lock()
            .expect("poisoned")
            .get_mut(command)
            .and_then(VecDeque::pop_front)
    }
}

fn session_body(status: &str) -> Value {
    json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": "sess-1",
        },
        "session": {
            "id": "sess-1",
            "memory": 2048,
            "vcpus": 2,
            "region": "iad1",
            "timeout": 270000,
            "status": status,
            "requestedAt": 1,
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": "/vercel/sandbox",
        },
        "routes": [],
    })
}

fn command_response(answer: &Answer) -> MockResponse {
    MockResponse::ndjson(&[
        json!({ "stream": "stdout", "data": answer.stdout }),
        json!({ "stream": "stderr", "data": answer.stderr }),
        json!({
            "command": {
                "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                "sessionId": "sess-1", "exitCode": answer.exit_code, "startedAt": 1,
            },
        }),
    ])
}

fn argv_of(request: &Recorded) -> Vec<String> {
    let body = request.json();
    let mut argv = vec![body["command"].as_str().unwrap_or_default().to_string()];
    argv.extend(
        body["args"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|arg| arg.as_str().map(str::to_string)),
    );
    argv
}

/// The default answer for a command the mount flow runs.
fn default_answer(argv: &[String]) -> Answer {
    let fingerprint_helper =
        openai_agents_sdk::sandbox::runtime_helpers::WORKSPACE_FINGERPRINT_HELPER
            .install_path
            .clone();
    match argv.first().map(String::as_str).unwrap_or_default() {
        command if command == fingerprint_helper && argv.len() > 2 => {
            ok(r#"{"fingerprint":"same","version":"workspace_tar_sha256_v1"}"#)
        }
        "/usr/bin/rpm" => ok("1.21.0"),
        "/usr/bin/realpath" => ok(argv.last().map(String::as_str).unwrap_or_default()),
        "/usr/bin/find" => ok(""),
        "/usr/bin/id" => ok(if argv.contains(&"-u".to_string()) {
            "1000"
        } else {
            "1001"
        }),
        "/usr/bin/findmnt" => ok("mountpoint-s3"),
        // Anything else, including the in-sandbox path resolver, answers with something.
        _ => ok("ok"),
    }
}

fn stalled_response(answer: &Answer, delay_ms: u64) -> MockResponse {
    let line = |value: Value| {
        let mut bytes = serde_json::to_vec(&value).expect("serializable");
        bytes.push(b'\n');
        bytes
    };
    MockResponse::streamed(
        "application/x-ndjson",
        vec![
            (
                line(json!({ "stream": "stdout", "data": answer.stdout })),
                0,
            ),
            (
                line(json!({
                    "command": {
                        "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                        "sessionId": "sess-1", "exitCode": answer.exit_code, "startedAt": 1,
                    },
                })),
                delay_ms,
            ),
        ],
    )
}

fn handler(script: Arc<Script>) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    move |request: &Recorded| match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => MockResponse::json(session_body("running")),
        ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
            let body = session_body("running");
            MockResponse::json(json!({ "session": body["session"], "routes": [] }))
        }
        ("POST", path) if path.ends_with("/fs/mkdir") => MockResponse::json(json!({})),
        ("POST", path) if path.ends_with("/fs/write") => {
            let mut failures = script.write_failures.lock().expect("poisoned");
            if *failures > 0 {
                *failures -= 1;
                return MockResponse::error(500, "write failed");
            }
            MockResponse::json(json!({}))
        }
        ("POST", path) if path.ends_with("/fs/read") => {
            if *script.read_missing.lock().expect("poisoned") {
                return MockResponse::error(404, "missing");
            }
            MockResponse::bytes(script.archive.lock().expect("poisoned").clone())
        }
        ("POST", path) if path.ends_with("/snapshot") => {
            MockResponse::json(json!({ "snapshot": { "id": "snap-1" } }))
        }
        ("POST", path) if path.ends_with("/stop") => {
            let body = session_body("stopped");
            MockResponse::json(json!({ "session": body["session"] }))
        }
        ("POST", path) if path.ends_with("/cmd") => {
            let argv = argv_of(request);
            let name = argv.first().map(String::as_str).unwrap_or_default();
            let answer = script.next(name).unwrap_or_else(|| default_answer(&argv));
            match script.stalled.lock().expect("poisoned").get(name) {
                Some(delay_ms) => stalled_response(&answer, *delay_ms),
                None => command_response(&answer),
            }
        }
        _ => MockResponse::error(404, "unexpected"),
    }
}

fn s3_strategy() -> VercelCloudBucketMountStrategy {
    VercelCloudBucketMountStrategy::new("test-bucket").with_region("us-west-2")
}

fn credentialed_strategy() -> VercelCloudBucketMountStrategy {
    s3_strategy()
        .with_credentials("test-access-key", "test-secret-key")
        .with_session_token("test-session-token")
}

const ROOT: &str = "/vercel/sandbox";

fn mount_manifest() -> Manifest {
    Manifest::new().with_root(ROOT).with_entry(
        "parent",
        Entry::dir().with_child("remote", Entry::mount(s3_strategy().mount())),
    )
}

fn client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    )
}

fn commands_named(server: &MockServer, command: &str) -> Vec<Recorded> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd") && argv_of(request)[0] == command)
        .collect()
}

async fn eventually(mut check: impl FnMut() -> bool) {
    for _ in 0..500 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the condition never held");
}

/// Python: `test_vercel_s3_nested_activation_serializes_workspace_commands`. While the mount
/// command is running, a command from another task waits for the mount to finish instead of
/// running against a half-mounted workspace.
#[tokio::test]
async fn workspace_commands_wait_while_an_s3_mount_is_attached() {
    let script = Arc::new(Script::default());
    script
        .stalled
        .lock()
        .expect("poisoned")
        .insert("/usr/bin/mount-s3".to_string(), 700);
    let server = MockServer::start(handler(script)).await;
    let session = client(&server)
        .create(Some(mount_manifest()))
        .await
        .expect("create");

    let starting = session.clone();
    let start_task = tokio::spawn(async move { starting.start().await });
    eventually(|| !commands_named(&server, "/usr/bin/mount-s3").is_empty()).await;
    let running = session.clone();
    let exec_task =
        tokio::spawn(async move { running.exec(ExecRequest::argv(["workspace-probe"])).await });
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(!exec_task.is_finished(), "the command ran during the mount");
    assert!(
        commands_named(&server, "workspace-probe").is_empty(),
        "the command reached the sandbox during the mount"
    );

    start_task.await.expect("joined").expect("start");
    let result = exec_task.await.expect("joined").expect("exec");
    assert!(result.ok());
    assert_eq!(commands_named(&server, "workspace-probe").len(), 1);
    assert_eq!(commands_named(&server, "/usr/bin/mount-s3").len(), 1);
    session.shutdown().await.expect("shutdown");
    server.stop();
}

/// Python: `test_vercel_without_s3_mounts_does_not_serialize_workspace_commands`.
#[tokio::test]
async fn a_session_without_mounts_runs_commands_concurrently() {
    let script = Arc::new(Script::default());
    script
        .stalled
        .lock()
        .expect("poisoned")
        .insert("slow".to_string(), 1500);
    let server = MockServer::start(handler(script)).await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root(ROOT)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let slow = session.clone();
    let slow_task = tokio::spawn(async move { slow.exec(ExecRequest::argv(["slow"])).await });
    eventually(|| !commands_named(&server, "slow").is_empty()).await;

    let fast = tokio::time::timeout(
        Duration::from_millis(1000),
        session.exec(ExecRequest::argv(["true"])),
    )
    .await
    .expect("the fast command is not queued behind the slow one")
    .expect("exec");
    assert!(fast.ok());
    assert!(!slow_task.is_finished());
    slow_task.await.expect("joined").expect("slow exec");
    server.stop();
}

/// Python: `test_vercel_mount_command_timeout_includes_output_collection`. The command's timeout
/// covers collecting its output, so a command whose output never completes is a timeout.
#[tokio::test]
async fn a_command_timeout_covers_collecting_its_output() {
    let script = Arc::new(Script::default());
    script
        .stalled
        .lock()
        .expect("poisoned")
        .insert("stuck".to_string(), 5000);
    let server = MockServer::start(handler(script)).await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root(ROOT)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let started = std::time::Instant::now();
    let error = session
        .exec(ExecRequest::argv(["stuck"]).with_timeout(Some(Duration::from_millis(300))))
        .await
        .expect_err("the output never completes");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error}");
    assert!(started.elapsed() < Duration::from_secs(3));
    server.stop();
}

/// Python: `test_vercel_s3_aclose_bypasses_failing_instrumentation`. A sink that fails the stop
/// event fails `aclose`, but the mounted sandbox is still shut down: a failed close never leaves
/// credentialed mounts running.
#[tokio::test]
async fn aclose_shuts_a_mounted_session_down_when_instrumentation_fails() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let inner = client(&server)
        .create(Some(mount_manifest()))
        .await
        .expect("create");
    let sink = CallbackSink::raising(|event| {
        if event.op == "stop" && event.phase == EventPhase::Start {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                "stop sink failed",
            ));
        }
        Ok(())
    });
    let session: Arc<dyn SandboxSession> =
        Arc::new(InstrumentedSession::new(inner, Arc::new(sink)));
    session.start().await.expect("start");
    assert_eq!(commands_named(&server, "/usr/bin/mount-s3").len(), 1);

    let error = session.aclose().await.expect_err("the sink fails the stop");

    assert!(
        error.message.contains("sandbox event sink failed"),
        "{error}"
    );
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        1
    );
    assert_eq!(commands_named(&server, "/usr/bin/umount").len(), 1);
    server.stop();
}
