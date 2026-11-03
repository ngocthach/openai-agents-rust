//! Ports of the S3-mounted close paths of `tests/extensions/sandbox/test_vercel.py`: a session
//! that failed to snapshot or whose pre-stop hook failed is still shut down, and a closed mounted
//! session cannot be used again.
#![cfg(feature = "sandbox")]
#![allow(dead_code)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
    WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
use openai_agents_sdk::sandbox::session_lifecycle::SandboxSessionLifecycle;
use openai_agents_sdk::sandbox::snapshot::{
    RemoteSnapshotClient, SnapshotSpec, register_remote_snapshot_client,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxError, SandboxResult,
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

fn create_count(server: &MockServer) -> usize {
    server.count("POST", "/v3/sandboxes")
}

const STOP: &str = "/v2/sandboxes/sessions/sess-1/stop";

fn tar_options(server: &MockServer) -> VercelSandboxClientOptions {
    options(server).with_workspace_persistence(WorkspacePersistence::Tar)
}

#[derive(Debug)]
struct FailingUpload;

#[async_trait]
impl RemoteSnapshotClient for FailingUpload {
    async fn upload(&self, _snapshot_id: &str, _data: &[u8]) -> SandboxResult<()> {
        Err(SandboxError::new(
            ErrorCode::SnapshotPersistError,
            "snapshot persist failed",
        ))
    }
    async fn download(&self, _snapshot_id: &str) -> SandboxResult<Vec<u8>> {
        Err(SandboxError::new(
            ErrorCode::SnapshotRestoreError,
            "nothing stored",
        ))
    }
    async fn exists(&self, _snapshot_id: &str) -> SandboxResult<bool> {
        Ok(false)
    }
}

fn failing_snapshot_client(server: &MockServer) -> VercelSandboxClient {
    let key = format!("tests.r15c.failing_upload.{}", uuid::Uuid::new_v4());
    register_remote_snapshot_client(key.clone(), Arc::new(FailingUpload));
    VercelSandboxClient::with_options(tar_options(server))
        .with_snapshot(SnapshotSpec::Remote { client_key: key })
}

/// `test_vercel_s3_aclose_shuts_down_after_snapshot_persist_failure`: the snapshot failure is the
/// error `aclose` reports, and the sandbox is still stopped once.
#[tokio::test]
async fn aclose_shuts_down_after_a_snapshot_persist_failure() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = failing_snapshot_client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = session.aclose().await.expect_err("the snapshot fails");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(error.message.contains("snapshot persist failed"));
    assert_eq!(server.count("POST", STOP), 1);
    server.stop();
}

/// `test_vercel_s3_stop_preserves_session_after_snapshot_persist_failure`: a failed snapshot does
/// not stop the sandbox, and the mount is attached again; a later shutdown stops it once.
#[tokio::test]
async fn a_failed_snapshot_leaves_the_session_running_with_its_mount() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = failing_snapshot_client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = persist_session_snapshot(session.as_ref())
        .await
        .expect_err("the snapshot fails");
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);

    assert_eq!(server.count("POST", STOP), 0);
    assert_eq!(
        lifecycle(&server)
            .into_iter()
            .filter(|entry| entry != "sh")
            .collect::<Vec<_>>(),
        vec!["mount-s3", "findmnt", "umount", "mount-s3"],
        "detached for the archive and attached again"
    );
    session.shutdown().await.expect("shutdown");
    assert_eq!(server.count("POST", STOP), 1);
    server.stop();
}

/// `test_vercel_s3_aclose_shuts_down_after_pre_stop_hook_failure`
#[tokio::test]
async fn aclose_shuts_down_after_a_pre_stop_hook_failure() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    session.register_pre_stop_hook(Arc::new(|| {
        Box::pin(async {
            Err(SandboxError::new(
                ErrorCode::ExecNonZero,
                "pre-stop hook failed",
            ))
        })
    }));

    let error = session.aclose().await.expect_err("the hook fails");

    assert!(error.message.contains("pre-stop hook failed"), "{error}");
    assert_eq!(server.count("POST", STOP), 1);
    server.stop();
}

/// `test_vercel_s3_closed_session_does_not_recreate_sandbox`: after the close a mounted session
/// refuses to execute or start, and nothing is created or stopped again.
#[tokio::test]
async fn a_closed_mounted_session_cannot_be_used_again() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    let created = create_count(&server);

    session.shutdown().await.expect("first shutdown");
    session.shutdown().await.expect("second shutdown");
    session.aclose().await.expect("aclose");

    let exec = session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect_err("a closed mounted session does not run commands");
    assert_eq!(exec.code, ErrorCode::WorkspaceStartError, "{exec}");
    assert!(session.start().await.is_err());
    assert_eq!(server.count("POST", STOP), 1);
    assert_eq!(create_count(&server), created);
    server.stop();
}

/// `test_vercel_s3_mount_transition_serializes_workspace_commands`: while a snapshot has the mount
/// detached, the model-facing check refuses to touch the workspace (Rust refuses where Python
/// queues the command), and it is allowed again once the mount is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_workspace_is_refused_while_the_mount_is_detached() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Notify;

    let script = Arc::new(Script::default());
    let respond = Arc::new(handler(script));
    let unmount_started = Arc::new(Notify::new());
    let release_unmount = Arc::new(Notify::new());
    let gate_open = Arc::new(AtomicBool::new(false));
    let server = {
        let (respond, started, release, gate) = (
            respond.clone(),
            unmount_started.clone(),
            release_unmount.clone(),
            gate_open.clone(),
        );
        MockServer::start_raw(move |recorded: Recorded, mut stream| {
            let (respond, started, release, gate) = (
                respond.clone(),
                started.clone(),
                release.clone(),
                gate.clone(),
            );
            async move {
                let gated = recorded.path.ends_with("/cmd")
                    && argv_of(&recorded)[0] == "/usr/bin/umount"
                    && !gate.swap(true, Ordering::SeqCst);
                if gated {
                    started.notify_one();
                    release.notified().await;
                }
                let response = respond(&recorded);
                let _ = support::write_response(&mut stream, response).await;
            }
        })
        .await
    };
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = VercelSandboxClient::with_options(tar_options(&server))
        .with_snapshot(SnapshotSpec::Local {
            base_path: snapshots.path().to_path_buf(),
        })
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    assert!(session.ensure_mounts_settled().is_ok());

    let persisting = {
        let session = session.clone();
        tokio::spawn(async move { persist_session_snapshot(session.as_ref()).await })
    };
    tokio::time::timeout(Duration::from_secs(5), unmount_started.notified())
        .await
        .expect("the unmount started");

    let refused = session
        .ensure_mounts_settled()
        .expect_err("the mount is detached");
    assert_eq!(refused.code, ErrorCode::MountFailed);

    release_unmount.notify_one();
    persisting.await.expect("task").expect("persist");
    assert!(session.ensure_mounts_settled().is_ok());
    server.stop();
}
