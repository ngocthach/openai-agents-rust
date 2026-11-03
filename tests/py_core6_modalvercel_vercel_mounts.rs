//! Ports of the S3 mount lifecycle group of `tests/extensions/sandbox/test_vercel.py`.
//!
//! Python patches the `vercel.sandbox` SDK; the Rust client speaks the Vercel sandbox REST API, so
//! the same behaviour is asserted against a local mock that records each request. The mock
//! answers every `cmd` request from a script: a queue of results per command, with a fixed default
//! for the commands the mount flow runs.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
    WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
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

fn create_count(server: &MockServer) -> usize {
    server.count("POST", "/v3/sandboxes")
}

// ---------------------------------------------------------------------------------------------
// Creation-time validation
// ---------------------------------------------------------------------------------------------

/// `test_vercel_create_requires_explicit_s3_credential_exposure`: a mount that carries its own
/// credentials is refused until the manifest acknowledges its path, and no sandbox is created.
#[tokio::test]
async fn create_requires_an_explicit_credential_exposure_acknowledgement() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;

    let outcome = client(&server)
        .create(Some(mount_manifest(credentialed_strategy())))
        .await;
    let created = match outcome {
        Ok(session) => session.start().await.map(|_| ()),
        Err(error) => Err(error),
    };

    let error = created.expect_err("the exposure is not acknowledged");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        !error.message.contains("test-secret-key"),
        "{}",
        error.message
    );
    assert!(
        commands_named(&server, "/usr/bin/mount-s3").is_empty(),
        "no credential reached the sandbox"
    );
    server.stop();
}

/// `test_vercel_create_accepts_manifest_mount_scoped_acknowledgement`
#[tokio::test]
async fn create_accepts_the_mount_scoped_acknowledgement() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;

    let session = client(&server)
        .create(Some(acknowledged(credentialed_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    assert_eq!(create_count(&server), 1);
    let mount = &commands_named(&server, "/usr/bin/mount-s3")[0];
    assert_eq!(mount.json()["env"]["AWS_ACCESS_KEY_ID"], "test-access-key");
    assert_eq!(
        mount.json()["env"]["AWS_SECRET_ACCESS_KEY"],
        "test-secret-key"
    );
    assert_eq!(
        mount.json()["env"]["AWS_SESSION_TOKEN"],
        "test-session-token"
    );
    assert_eq!(mount.json()["env"]["AWS_REGION"], "us-west-2");
    assert_eq!(mount.json()["sudo"], true);
    server.stop();
}

/// `test_vercel_s3_mount_validates_credentials_and_lifecycle`: half-configured credentials and a
/// mount that asks to be durable are refused when the manifest is validated.
#[tokio::test]
async fn mount_validation_refuses_partial_credentials_and_durable_mounts() {
    let partial = VercelCloudBucketMountStrategy {
        bucket: "test-bucket".to_string(),
        access_key_id: Some("test-access-key".to_string()),
        ..VercelCloudBucketMountStrategy::default()
    };
    let error = partial
        .validate(&partial.clone().mount())
        .expect_err("a key without a secret");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("needs both"), "{}", error.message);

    let durable = s3_strategy();
    let error = durable
        .validate(&durable.clone().mount().with_ephemeral(false))
        .expect_err("a durable mount");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("must be ephemeral"),
        "{}",
        error.message
    );
}

/// `test_vercel_rejects_root_and_overlapping_s3_mounts`: a mount at the workspace root, outside
/// it, nested in another mount, or over a manifest entry is refused before any sandbox exists.
#[tokio::test]
async fn create_rejects_root_outside_and_overlapping_mounts() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let client = client(&server);

    let at_root = Manifest::new().with_root("/custom-workspace").with_entry(
        "remote",
        Entry::mount(
            VercelCloudBucketMountStrategy::new("root-bucket")
                .mount()
                .with_mount_path("/custom-workspace"),
        ),
    );
    let outside = Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(
            VercelCloudBucketMountStrategy::new("outside-bucket")
                .mount()
                .with_mount_path("/tmp/remote"),
        ),
    );
    let nested = Manifest::new()
        .with_root(ROOT)
        .with_entry(
            "remote",
            Entry::mount(VercelCloudBucketMountStrategy::new("outer").mount()),
        )
        .with_entry(
            "remote/nested",
            Entry::mount(VercelCloudBucketMountStrategy::new("inner").mount()),
        );
    let over_entry = Manifest::new()
        .with_root(ROOT)
        .with_entry(
            "remote",
            Entry::mount(
                VercelCloudBucketMountStrategy::new("physical-overlap")
                    .mount()
                    .with_mount_path("actual"),
            ),
        )
        .with_entry("actual/config.json", Entry::file(b"{}".to_vec()));

    for (label, manifest, needle) in [
        ("workspace root", at_root, "workspace root"),
        (
            "outside the workspace",
            outside,
            "within the workspace root",
        ),
        ("nested mounts", nested, "overlap"),
        ("over an entry", over_entry, "overlap"),
    ] {
        let outcome = client.create(Some(manifest)).await;
        let error = match outcome {
            Ok(session) => session.start().await.expect_err(label),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{label}");
        assert!(error.message.contains(needle), "{label}: {}", error.message);
    }
    assert!(
        commands_named(&server, "/usr/bin/mount-s3").is_empty(),
        "no mount command ran"
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------------------------

/// `test_vercel_s3_mount_is_create_time_only_and_credentials_are_not_serialized`: the credentials
/// reach the mount command as environment, and nothing the session persists carries them.
#[tokio::test]
async fn mount_credentials_are_not_serialized() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server)
        .create(Some(acknowledged(credentialed_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let state = serde_json::to_string(&session.session_state()).expect("serialize");

    for secret in ["test-access-key", "test-secret-key", "test-session-token"] {
        assert!(!state.contains(secret), "{secret} leaked into the state");
    }
    // The topology cannot change once the sandbox exists.
    let strategy = credentialed_strategy();
    let mount = strategy.clone().mount();
    let late = strategy
        .activate(&mount, session.as_ref(), "/vercel/sandbox/remote")
        .await
        .expect_err("a late mount");
    assert_eq!(late.code, ErrorCode::MountConfigInvalid);
    assert!(late.message.contains("fixed"), "{}", late.message);
    server.stop();
}

/// `test_vercel_s3_manifest_sanitization_preserves_typed_environment`: the manifest environment
/// reaches the create request next to a credentialed mount, and the credentials stay out of it.
#[tokio::test]
async fn manifest_environment_survives_next_to_a_credentialed_mount() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let manifest = acknowledged(credentialed_strategy())
        .with_env("DIRECT", "direct-value")
        .with_env("ENTRY", "entry-value");

    client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let create = server.find("POST", "/v3/sandboxes");
    assert_eq!(create.json()["env"]["DIRECT"], "direct-value");
    assert_eq!(create.json()["env"]["ENTRY"], "entry-value");
    assert!(
        !create.text().contains("test-secret-key"),
        "{}",
        create.text()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Start ordering
// ---------------------------------------------------------------------------------------------

/// `test_vercel_s3_entry_failure_happens_before_mount_activation`: a manifest entry that fails to
/// write stops the start before the mount command, and the session can start again. The entry is
/// declared after the mount, and still lands first: every S3 mount is attached last.
#[tokio::test]
async fn an_entry_failure_happens_before_mount_activation() {
    let script = Arc::new(Script::default());
    // More failures than the write retry policy makes attempts.
    *script.write_failures.lock().expect("poisoned") = 10;
    let server = MockServer::start(handler(script.clone())).await;
    let manifest = Manifest::new()
        .with_root(ROOT)
        .with_entry("remote", Entry::mount(s3_strategy().mount()))
        .with_entry("later.txt", Entry::file(b"later".to_vec()));
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session.start().await.expect_err("the write fails");

    assert!(!error.message.is_empty());
    assert!(commands_named(&server, "/usr/bin/mount-s3").is_empty());
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        0
    );

    *script.write_failures.lock().expect("poisoned") = 0;
    session.start().await.expect("a second start");
    assert_eq!(commands_named(&server, "/usr/bin/mount-s3").len(), 1);
    server.stop();
}

/// `test_vercel_s3_snapshot_entries_materialize_before_mount_activation`, for the half this
/// backend shares: a mount point that is not empty fails the start without a mount command.
#[tokio::test]
async fn a_populated_mountpoint_fails_the_start_without_a_mount_command() {
    let script = Arc::new(Script::default());
    script.queue(
        "/usr/bin/find",
        vec![ok("/vercel/sandbox/actual/remote/config.json")],
    );
    let server = MockServer::start(handler(script)).await;
    let manifest = Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(s3_strategy().mount().with_mount_path("actual/remote")),
    );
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session
        .start()
        .await
        .expect_err("the mount point is populated");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("empty mount directory"),
        "{}",
        error.message
    );
    assert!(commands_named(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

/// `test_vercel_s3_mount_rejects_symlink_components`: a mount path that resolves through a
/// symlink is refused before any mount command.
#[tokio::test]
async fn a_mount_path_through_a_symlink_is_refused() {
    let script = Arc::new(Script::default());
    // `realpath` reports a different location: a component of the path is a symlink.
    script.queue(
        "/usr/bin/realpath",
        vec![ok("/vercel/sandbox/durable/remote")],
    );
    let server = MockServer::start(handler(script)).await;
    let manifest = Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(s3_strategy().mount().with_mount_path("link/remote")),
    );
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session.start().await.expect_err("a symlinked path");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("must not resolve through symlinks"),
        "{}",
        error.message
    );
    assert!(commands_named(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Persistence around a mount
// ---------------------------------------------------------------------------------------------

fn tar_options(server: &MockServer) -> VercelSandboxClientOptions {
    options(server).with_workspace_persistence(WorkspacePersistence::Tar)
}

fn snapshot_client(server: &MockServer, directory: &std::path::Path) -> VercelSandboxClient {
    VercelSandboxClient::with_options(tar_options(server)).with_snapshot(SnapshotSpec::Local {
        base_path: directory.to_path_buf(),
    })
}

/// `test_vercel_s3_mount_detaches_for_tar_persistence_and_unmounts_on_shutdown`: a tar snapshot
/// detaches the mount, archives, and attaches it again.
#[tokio::test]
async fn tar_persistence_detaches_the_mount_and_attaches_it_again() {
    let script = Arc::new(Script::default());
    *script.archive.lock().expect("poisoned") = b"archive".to_vec();
    let server = MockServer::start(handler(script)).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = snapshot_client(&server, snapshots.path())
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    persist_session_snapshot(session.as_ref())
        .await
        .expect("persist");

    let order: Vec<String> = lifecycle(&server)
        .into_iter()
        .filter(|entry| entry != "sh")
        .collect();
    assert_eq!(
        order,
        vec!["mount-s3", "findmnt", "umount", "mount-s3"],
        "attach, detach for the archive, attach again"
    );
    let archived = server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd"))
        .map(|request| argv_of(&request).join(" "))
        .find(|command| command.contains("tar -czf"))
        .expect("an archive command");
    assert!(archived.contains("--no-wildcards"), "{archived}");
    server.stop();
}

/// `test_vercel_s3_missing_tracked_mount_stops_session`: if the mount this session attached is no
/// longer there, the sandbox is stopped rather than archived.
#[tokio::test]
async fn a_missing_tracked_mount_stops_the_session() {
    let script = Arc::new(Script::default());
    let server = MockServer::start(handler(script.clone())).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = snapshot_client(&server, snapshots.path())
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    script.queue("/usr/bin/findmnt", vec![failed("", 1)]);

    let error = persist_session_snapshot(session.as_ref())
        .await
        .expect_err("the mount is gone");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        1
    );
    server.stop();
}

/// `test_vercel_s3_mount_disappearing_during_unmount_stops_session`: an unmount that fails because
/// the mount vanished meanwhile also stops the sandbox.
#[tokio::test]
async fn a_mount_disappearing_during_unmount_stops_the_session() {
    let script = Arc::new(Script::default());
    let server = MockServer::start(handler(script.clone())).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = snapshot_client(&server, snapshots.path())
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    script.queue("/usr/bin/findmnt", vec![ok("mountpoint-s3"), failed("", 1)]);
    script.queue("/usr/bin/umount", vec![failed("missing", 32)]);

    let error = persist_session_snapshot(session.as_ref())
        .await
        .expect_err("the unmount fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        1
    );
    server.stop();
}

/// `test_vercel_s3_unexpected_persist_error_restores_mount`: an archive that cannot be read back
/// still re-attaches the mount, and the sandbox keeps running.
#[tokio::test]
async fn an_unexpected_persist_error_restores_the_mount() {
    let script = Arc::new(Script::default());
    *script.read_missing.lock().expect("poisoned") = true;
    let server = MockServer::start(handler(script)).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = snapshot_client(&server, snapshots.path())
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = persist_session_snapshot(session.as_ref())
        .await
        .expect_err("the archive is missing");

    assert!(!error.message.is_empty());
    assert_eq!(
        commands_named(&server, "/usr/bin/mount-s3").len(),
        2,
        "re-attached"
    );
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        0
    );
    assert!(
        session
            .exec(ExecRequest::argv(["true"]))
            .await
            .expect("still usable")
            .ok()
    );
    server.stop();
}

/// `test_vercel_snapshot_persistence_tears_down_ephemeral_mounts`: a native snapshot is taken with
/// the mount detached.
#[tokio::test]
async fn native_snapshot_is_taken_with_the_mount_detached() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let session = VercelSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .with_snapshot(SnapshotSpec::Local {
        base_path: snapshots.path().to_path_buf(),
    })
    .create(Some(mount_manifest(s3_strategy())))
    .await
    .expect("create");
    session.start().await.expect("start");

    persist_session_snapshot(session.as_ref())
        .await
        .expect("persist");

    let order: Vec<String> = lifecycle(&server)
        .into_iter()
        .filter(|entry| entry != "sh")
        .collect();
    assert_eq!(
        order,
        vec!["mount-s3", "findmnt", "umount", "snapshot", "mount-s3"]
    );
    server.stop();
}

/// `test_vercel_s3_hydrate_rejects_mount_overlaps_before_detach`: an archive member under the
/// mount path is refused before anything is uploaded or detached.
#[tokio::test]
async fn hydrate_rejects_members_under_a_mount_path() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server)
        .create(Some(mount_manifest(s3_strategy())))
        .await
        .expect("create");
    session.start().await.expect("start");
    let writes = server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write");

    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(19);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    builder
        .append_data(
            &mut header,
            "remote/hidden.txt",
            &b"must-not-be-written"[..],
        )
        .expect("append");
    let archive = builder.into_inner().expect("finish");

    let error = session
        .hydrate_workspace(&archive)
        .await
        .expect_err("a member under the mount");

    assert!(
        error.message.contains("overlaps protected path"),
        "{}",
        error.message
    );
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write"),
        writes,
        "nothing was uploaded"
    );
    assert!(commands_named(&server, "/usr/bin/findmnt").is_empty());
    server.stop();
}

/// `test_vercel_tar_persistence_treats_mount_exclusions_as_literal_paths`: an entry whose name
/// holds a glob character is excluded literally.
#[tokio::test]
async fn tar_exclusions_are_literal_paths() {
    let script = Arc::new(Script::default());
    *script.archive.lock().expect("poisoned") = b"archive".to_vec();
    let server = MockServer::start(handler(script)).await;
    let manifest = Manifest::new()
        .with_root(ROOT)
        .with_entry("cache[1]", Entry::dir().with_ephemeral(true));
    let session = VercelSandboxClient::with_options(tar_options(&server))
        .create(Some(manifest))
        .await
        .expect("create");

    session.persist_workspace().await.expect("persist");

    let tar = server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd"))
        .map(|request| argv_of(&request))
        .find(|argv| argv.iter().any(|part| part.contains("tar -czf")))
        .expect("a tar command");
    let script = tar.last().expect("script").clone();
    let no_wildcards = script.find("--no-wildcards").expect("--no-wildcards");
    let exclude = script
        .find("--exclude './cache[1]'")
        .expect("a literal exclude");
    assert!(no_wildcards < exclude, "{script}");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Protected mount configuration: provider text never reaches the caller
// ---------------------------------------------------------------------------------------------

/// `test_vercel_protected_session_public_operations_redact_provider_failures`: with a credentialed
/// mount in play, an exec, read, write or port failure reports no provider text.
#[tokio::test]
async fn public_operations_redact_provider_failures() {
    for operation in ["exec", "read", "write", "port"] {
        let secret = "public-provider-secret";
        let script = Arc::new(Script::default());
        let inner = handler(script);
        let server = MockServer::start(move |request: &Recorded| {
            let failing = match operation {
                "exec" => request.path.ends_with("/cmd") && argv_of(request)[0] == "true",
                "read" => request.path.ends_with("/fs/read"),
                "write" => request.path.ends_with("/fs/write"),
                _ => false,
            };
            if failing {
                return MockResponse::error(500, secret);
            }
            inner(request)
        })
        .await;
        let session = VercelSandboxClient::with_options(options(&server).with_exposed_port(3000))
            .create(Some(acknowledged(credentialed_strategy())))
            .await
            .expect("create");
        session.start().await.expect("start");

        let error = match operation {
            "exec" => session.exec(ExecRequest::argv(["true"])).await.map(|_| ()),
            "read" => session.read_file("file.txt").await.map(|_| ()),
            "write" => session.write_file("file.txt", b"content").await,
            _ => session.resolve_exposed_port(3000).await.map(|_| ()),
        }
        .expect_err(operation);

        assert!(
            error.message.contains("protected mount configuration"),
            "{operation}: {}",
            error.message
        );
        assert!(
            !error.message.contains(secret),
            "{operation}: {}",
            error.message
        );
        assert!(error.context.is_empty(), "{operation}: {:?}", error.context);
        server.stop();
    }
}

/// `test_vercel_direct_persist_redacts_protected_mount_provider_error` and
/// `test_vercel_client_delete_propagates_redacted_shutdown_failure`: a failed persist or delete
/// reports no provider text either.
#[tokio::test]
async fn persist_and_delete_redact_provider_failures() {
    let secret = "direct-vercel-secret";
    let script = Arc::new(Script::default());
    let inner = handler(script);
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/stop") || request.path.ends_with("/fs/read") {
            return MockResponse::error(500, secret);
        }
        inner(request)
    })
    .await;
    let client = VercelSandboxClient::with_options(tar_options(&server));
    let session = client
        .create(Some(acknowledged(credentialed_strategy())))
        .await
        .expect("create");

    let persisted = session
        .persist_workspace()
        .await
        .expect_err("the read fails");
    let deleted = client.delete(session).await.expect_err("the stop fails");

    for error in [persisted, deleted] {
        assert!(
            error.message.contains("protected mount configuration"),
            "{}",
            error.message
        );
        assert!(!error.message.contains(secret), "{}", error.message);
    }
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Validation before a sandbox exists
// ---------------------------------------------------------------------------------------------

/// `test_vercel_create_revalidates_mutated_s3_mount`: a mount that asks to be durable is refused
/// before the sandbox is created.
#[tokio::test]
async fn create_refuses_a_durable_mount_before_allocating() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let manifest = Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(s3_strategy().mount().with_ephemeral(false)),
    );

    let error = client(&server)
        .create(Some(manifest))
        .await
        .map(|_| ())
        .expect_err("a durable mount");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("must be ephemeral"),
        "{}",
        error.message
    );
    assert_eq!(create_count(&server), 0);
    server.stop();
}

/// `test_vercel_credential_opt_in_does_not_allow_signed_endpoint_urls`: an endpoint URL that
/// carries a signature is refused, whether or not the mount is acknowledged.
#[tokio::test]
async fn a_signed_endpoint_url_is_refused() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let strategy =
        credentialed_strategy().with_endpoint_url("https://example.test?signature=endpoint-secret");

    let error = client(&server)
        .create(Some(acknowledged(strategy)))
        .await
        .map(|_| ())
        .expect_err("a signed endpoint");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("does not support exposing"),
        "{}",
        error.message
    );
    assert!(
        !error.message.contains("endpoint-secret"),
        "{}",
        error.message
    );
    assert_eq!(create_count(&server), 0);
    server.stop();
}

/// `test_vercel_credential_opt_in_redacts_provider_validation_errors`: a validation failure on a
/// credentialed mount never repeats the secret.
#[tokio::test]
async fn validation_errors_never_repeat_the_secret() {
    let sentinel = "vercel-validation-secret";
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let partial = VercelCloudBucketMountStrategy {
        bucket: "test-bucket".to_string(),
        secret_access_key: Some(sentinel.to_string()),
        ..VercelCloudBucketMountStrategy::default()
    };
    let at_root = Manifest::new().with_root("/custom-workspace").with_entry(
        "remote",
        Entry::mount(
            credentialed_strategy_with(sentinel)
                .mount()
                .with_mount_path("/custom-workspace"),
        ),
    );

    for (label, manifest) in [
        ("partial credentials", acknowledged(partial)),
        ("workspace root", at_root),
    ] {
        let error = client(&server)
            .create(Some(manifest))
            .await
            .map(|_| ())
            .expect_err(label);

        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{label}");
        assert!(
            !error.message.contains(sentinel),
            "{label}: {}",
            error.message
        );
        assert!(
            error
                .context
                .values()
                .all(|value| !value.contains(sentinel)),
            "{label}"
        );
    }
    assert_eq!(create_count(&server), 0);
    server.stop();
}

fn credentialed_strategy_with(secret: &str) -> VercelCloudBucketMountStrategy {
    s3_strategy().with_credentials("test-access-key", secret)
}

/// `test_vercel_apply_manifest_uses_central_mount_validation`: an in-container mount with its own
/// credentials is refused by the shared credential boundary, before any command runs in the
/// sandbox beyond preparing the workspace root.
#[tokio::test]
async fn the_central_credential_boundary_guards_other_mount_strategies() {
    use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let manifest = Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(
            ConfiguredMountStrategy::mountpoint(
                S3Mount::new("example-bucket")
                    .with_credentials("example-access-key", "example-secret-key"),
            )
            .mount(),
        ),
    );
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session
        .start()
        .await
        .expect_err("an unacknowledged credential");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        !error.message.contains("example-secret-key"),
        "{}",
        error.message
    );
    assert!(
        server
            .requests()
            .iter()
            .filter(|request| request.path.ends_with("/cmd"))
            .all(|request| argv_of(request)[0] == "mkdir"),
        "only the workspace root was prepared"
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume, restore and timeouts around a mount
// ---------------------------------------------------------------------------------------------

/// `test_vercel_s3_mount_starts_after_restorable_tar_snapshot`: a resumed session whose tar
/// snapshot is restorable restores it and then attaches the mount exactly once.
#[tokio::test]
async fn a_resumed_session_restores_the_tar_snapshot_then_mounts_once() {
    let script = Arc::new(Script::default());
    *script.archive.lock().expect("poisoned") = b"archive".to_vec();
    let server = MockServer::start(handler(script)).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let client = snapshot_client(&server, snapshots.path());
    let manifest = mount_manifest(s3_strategy());
    let first = client.create(Some(manifest.clone())).await.expect("create");
    // An empty gzipped tar is a restorable snapshot of an empty workspace.
    let empty = {
        let raw = tar::Builder::new(Vec::new()).into_inner().expect("finish");
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &raw).expect("compress");
        encoder.finish().expect("finish")
    };
    first.snapshot().persist(&empty).await.expect("persist");
    let state = first.session_state();

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");

    assert_eq!(commands_named(&server, "/usr/bin/mount-s3").len(), 1);
    assert!(
        server
            .find_opt("POST", "/v2/sandboxes/sessions/sess-1/fs/write")
            .is_some(),
        "the snapshot was restored first"
    );
    server.stop();
}

/// `test_vercel_s3_hydrate_rejects_mount_overlaps_before_detach` (the native-snapshot half): a
/// mounted session refuses to restore a native snapshot, which would replace the sandbox and
/// drop the mount.
#[tokio::test]
async fn a_mounted_session_refuses_a_native_snapshot_restore() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = VercelSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .create(Some(mount_manifest(s3_strategy())))
    .await
    .expect("create");
    session.start().await.expect("start");
    let reference = session
        .persist_workspace()
        .await
        .expect("a native reference");

    let error = session
        .hydrate_workspace(&reference)
        .await
        .expect_err("a native restore under a mount");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("native snapshot"),
        "{}",
        error.message
    );
    assert_eq!(create_count(&server), 1, "no replacement sandbox");
    server.stop();
}

/// `test_vercel_exec_timeout_includes_output_collection_and_releases_mount_lock`: the timeout
/// covers collecting the output, not only starting the command, and the session is usable after.
#[tokio::test]
async fn the_exec_timeout_covers_output_collection() {
    let inner = handler(Arc::new(Script::default()));
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/cmd") && argv_of(request)[0] == "slow" {
            // The headers arrive at once; the body stalls.
            return MockResponse::streamed(
                "application/x-ndjson",
                vec![(b"{\"stream\":\"stdout\",\"data\":\"x\"}\n".to_vec(), 5_000)],
            );
        }
        inner(request)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["slow"]).with_timeout(Some(Duration::from_millis(300))))
        .await
        .expect_err("the output never finishes");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert!(
        session
            .exec(ExecRequest::argv(["true"]))
            .await
            .expect("usable afterwards")
            .ok()
    );
    server.stop();
}

/// `test_vercel_s3_snapshot_entries_materialize_before_mount_activation`: an entry that reaches the
/// mount path through an alias is written before the mount, so the populated mount point is
/// caught, and no mount command runs.
#[tokio::test]
async fn entries_materialize_before_the_mount_and_a_populated_mountpoint_fails() {
    let script = Arc::new(Script::default());
    script.queue(
        "/usr/bin/find",
        vec![ok("/vercel/sandbox/actual/remote/config.json")],
    );
    let server = MockServer::start(handler(script)).await;
    let manifest = Manifest::new()
        .with_root(ROOT)
        .with_entry(
            "remote",
            Entry::mount(s3_strategy().mount().with_mount_path("actual/remote")),
        )
        .with_entry(
            "alias/remote/config.json",
            Entry::file(b"{}".to_vec()).with_ephemeral(true),
        );
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session
        .start()
        .await
        .expect_err("the mount point is populated");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("empty mount directory"),
        "{}",
        error.message
    );
    assert!(
        server
            .find_opt("POST", "/v2/sandboxes/sessions/sess-1/fs/write")
            .is_some(),
        "the entry was written first"
    );
    assert!(commands_named(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

fn empty_gz_tar() -> Vec<u8> {
    let raw = tar::Builder::new(Vec::new()).into_inner().expect("finish");
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &raw).expect("compress");
    encoder.finish().expect("finish")
}

/// The same ordering on a resume that restores a snapshot: the ephemeral entries come back before
/// the mount is attached.
#[tokio::test]
async fn a_resumed_start_attaches_the_mount_after_the_restored_entries() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let snapshots = tempfile::tempdir().expect("snapshots");
    let client = snapshot_client(&server, snapshots.path());
    let manifest = mount_manifest(s3_strategy()).with_entry(
        "later.txt",
        Entry::file(b"later".to_vec()).with_ephemeral(true),
    );
    let first = client.create(Some(manifest.clone())).await.expect("create");
    first
        .snapshot()
        .persist(&empty_gz_tar())
        .await
        .expect("persist");
    let state = first.session_state();

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");

    let order: Vec<String> = server
        .requests()
        .iter()
        .filter_map(|request| {
            if request.path.ends_with("/fs/write") {
                Some("write".to_string())
            } else if request.path.ends_with("/cmd") && argv_of(request)[0] == "/usr/bin/mount-s3" {
                Some("mount".to_string())
            } else {
                None
            }
        })
        .collect();
    let mount = order
        .iter()
        .position(|step| step == "mount")
        .expect("mounted");
    assert!(
        order[..mount]
            .iter()
            .filter(|step| *step == "write")
            .count()
            >= 2,
        "the archive and the entry are written before the mount: {order:?}"
    );
    server.stop();
}

/// `test_vercel` resume fast path (the Vercel side of
/// `test_modal_resume_marks_reconnected_sandbox_preserved_before_snapshot_reuse`): a reattached
/// sandbox whose workspace still matches the stored fingerprint skips the snapshot restore, and one
/// that changed restores it.
#[tokio::test]
async fn resume_skips_the_restore_when_the_workspace_fingerprint_matches() {
    for (stored, restored) in [("same", false), ("stale", true)] {
        let server = MockServer::start(handler(Arc::new(Script::default()))).await;
        let snapshots = tempfile::tempdir().expect("snapshots");
        let client = snapshot_client(&server, snapshots.path());
        let manifest = Manifest::new().with_root(ROOT);
        let first = client.create(Some(manifest.clone())).await.expect("create");
        first
            .snapshot()
            .persist(&empty_gz_tar())
            .await
            .expect("persist");
        let state = first.session_state().with_snapshot_fingerprint(Some(
            openai_agents_sdk::sandbox::SnapshotFingerprint {
                fingerprint: stored.to_string(),
                version: "workspace_tar_sha256_v1".to_string(),
            },
        ));

        let resumed = client.resume(&state, manifest).await.expect("resume");
        resumed.start().await.expect("start");

        let uploaded = server
            .find_opt("POST", "/v2/sandboxes/sessions/sess-1/fs/write")
            .is_some();
        assert_eq!(uploaded, restored, "stored fingerprint {stored}");
        assert_eq!(create_count(&server), 1, "{stored}: the sandbox is reused");
        server.stop();
    }
}

/// The fingerprint a session records travels in its state.
#[tokio::test]
async fn the_session_state_carries_the_recorded_fingerprint() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server).create(None).await.expect("create");

    session.record_snapshot_fingerprint(Some(openai_agents_sdk::sandbox::SnapshotFingerprint {
        fingerprint: "abc".to_string(),
        version: "workspace_tar_sha256_v2".to_string(),
    }));

    let state = session.session_state();
    assert_eq!(state.snapshot_fingerprint.as_deref(), Some("abc"));
    server.stop();
}
