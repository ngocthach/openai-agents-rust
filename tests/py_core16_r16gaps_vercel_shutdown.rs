//! Ports of `test_vercel_s3_shutdown_preserves_first_stop_failure` and
//! `test_vercel_s3_aclose_retries_failed_transition_stop`: shutting down a mounted session
//! unmounts first, a failed unmount stops the sandbox, a failed stop is retried by the final
//! stop, and the first failure is the one reported.
#![cfg(feature = "sandbox")]
#![allow(dead_code)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient};
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
            let answer = script
                .next(argv.first().map(String::as_str).unwrap_or_default())
                .unwrap_or_else(|| default_answer(&argv));
            command_response(&answer)
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

fn mount_manifest(strategy: VercelCloudBucketMountStrategy) -> Manifest {
    Manifest::new()
        .with_root(ROOT)
        .with_entry("remote", Entry::mount(strategy.mount()))
}

fn acknowledged(strategy: VercelCloudBucketMountStrategy) -> Manifest {
    mount_manifest(strategy)
        .with_in_container_mount_credential_exposure_acknowledged(["/vercel/sandbox/remote"])
}

fn client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(options(server))
}

fn options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

fn commands_named(server: &MockServer, command: &str) -> Vec<Recorded> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd") && argv_of(request)[0] == command)
        .collect()
}

/// The command of every `cmd` request, in order, restricted to the mount-lifecycle ones.
fn lifecycle(server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter_map(|request| {
            if request.path.ends_with("/cmd") {
                let command = argv_of(&request)[0].clone();
                if ["/usr/bin/mount-s3", "/usr/bin/findmnt", "/usr/bin/umount"]
                    .contains(&command.as_str())
                    || command == "sh"
                {
                    return Some(command.rsplit('/').next().unwrap_or_default().to_string());
                }
            } else if request.path.ends_with("/snapshot") {
                return Some("snapshot".to_string());
            } else if request.path.ends_with("/stop") {
                return Some("stop".to_string());
            }
            None
        })
        .collect()
}

const STOP: &str = "/v2/sandboxes/sessions/sess-1/stop";

/// A handler whose first `failures` stop requests fail with a distinct 500 body each.
fn handler_with_failing_stops(
    script: Arc<Script>,
    failures: usize,
) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    let inner = handler(script);
    let stops = AtomicUsize::new(0);
    move |request: &Recorded| {
        if request.method == "POST" && request.path.ends_with("/stop") {
            let n = stops.fetch_add(1, Ordering::SeqCst);
            if n < failures {
                let ordinal = if n == 0 { "first" } else { "second" };
                return MockResponse::error(500, &format!("{ordinal} stop failed"));
            }
        }
        inner(request)
    }
}

fn busy_unmount(script: &Script) {
    script.queue("/usr/bin/umount", vec![failed("busy", 32)]);
}

/// A clean shutdown detaches the mount before it stops the sandbox.
#[tokio::test]
async fn shutdown_unmounts_before_it_stops() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    session.shutdown().await.expect("shutdown");

    let order = lifecycle(&server);
    let umount = order.iter().position(|c| c == "umount").expect("unmounted");
    let stop = order.iter().position(|c| c == "stop").expect("stopped");
    assert!(umount < stop, "{order:?}");
    assert_eq!(server.count("POST", STOP), 1);
    server.stop();
}

/// `test_vercel_s3_shutdown_preserves_first_stop_failure`: the unmount fails, the transition's
/// stop fails, the final stop fails too; both stops were sent and the first failure is reported.
#[tokio::test]
async fn shutdown_preserves_the_first_stop_failure() {
    let script = Arc::new(Script::default());
    busy_unmount(&script);
    let server = MockServer::start(handler_with_failing_stops(script, 2)).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = session.shutdown().await.expect_err("the stops fail");

    assert!(error.message.contains("first stop failed"), "{error}");
    assert!(!error.message.contains("second stop failed"), "{error}");
    assert_eq!(server.count("POST", STOP), 2);
    server.stop();
}

/// `test_vercel_s3_aclose_retries_failed_transition_stop`: the transition's stop fails, the
/// close stops again and succeeds, and the session is then unusable.
#[tokio::test]
async fn close_retries_a_failed_transition_stop() {
    let script = Arc::new(Script::default());
    busy_unmount(&script);
    let server = MockServer::start(handler_with_failing_stops(script, 1)).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = session.shutdown().await.expect_err("the unmount failed");
    assert!(!error.message.contains("second stop failed"), "{error}");
    assert_eq!(server.count("POST", STOP), 2);
    let exec = session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect_err("a stopped mounted session does not run commands");
    assert_eq!(exec.code, ErrorCode::WorkspaceStartError, "{exec}");
    server.stop();
}
