//! E2B cases of `tests/extensions/sandbox/test_e2b.py` that the earlier ports left as
//! Python-only: the remount error that wins over an archive failure, the workspace-root `start`
//! and `mkdir` rules, and the resume clean-up that leaves nested mounts alone.
//!
//! The sandbox is a local mock server speaking the platform API and the envd Connect protocol.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BSandboxClient, E2BSandboxClientOptions, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::session::{
    clear_workspace_root_on_resume, persist_session_snapshot,
};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxError, SandboxResult, SandboxSession,
};
use serde_json::{Value, json};
use support::{MockResponse, Recorded};

type Events = Arc<Mutex<Vec<String>>>;

/// A mount that records its detach and attach calls and can be told to fail either.
#[derive(Debug)]
struct RecordingMount {
    name: &'static str,
    events: Events,
    fail_detach: bool,
    fail_attach: bool,
}

impl RecordingMount {
    fn mount(name: &'static str, events: &Events, fail_detach: bool, fail_attach: bool) -> Mount {
        Mount::new(Arc::new(RecordingMount {
            name,
            events: events.clone(),
            fail_detach,
            fail_attach,
        }))
    }

    fn record(&self, what: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("{what}:{}", self.name));
    }
}

#[async_trait]
impl MountStrategy for RecordingMount {
    fn strategy_type(&self) -> &str {
        "recording"
    }

    /// Like every provider mount: the session-level snapshot flow detaches it.
    fn supports_native_snapshot_detach(&self) -> bool {
        false
    }

    async fn activate(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        if self.fail_attach {
            self.record("mount_fail");
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!("boom while remounting {}", self.name),
            ));
        }
        self.record("mount");
        Ok(())
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        if self.fail_detach {
            self.record("unmount_fail");
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!("boom while unmounting {}", self.name),
            ));
        }
        self.record("unmount");
        Ok(())
    }
}

fn frame(flags: u8, value: Value) -> Vec<u8> {
    let payload = serde_json::to_vec(&value).expect("serializable");
    let mut frame = vec![flags];
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// A finished envd command printing `stdout` and exiting with `exit_code`.
fn finished(stdout: &str, exit_code: i64) -> MockResponse {
    let mut body = frame(0, json!({ "event": { "start": { "pid": 4242 } } }));
    body.extend(frame(
        0,
        json!({ "event": { "data": { "stdout": support::b64(stdout) } } }),
    ));
    body.extend(frame(
        0,
        json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
    ));
    body.extend(frame(2, json!({})));
    MockResponse::streamed("application/connect+json", vec![(body, 0)])
}

fn manifest(events: &Events, second_detach_fails: bool, second_attach_fails: bool) -> Manifest {
    Manifest::new()
        .with_entry(
            "repo/mount1",
            Entry::mount(RecordingMount::mount("mount1", events, false, false)),
        )
        .with_entry(
            "repo/mount2",
            Entry::mount(RecordingMount::mount(
                "mount2",
                events,
                second_detach_fails,
                second_attach_fails,
            )),
        )
}

fn recorded(events: &Events) -> Vec<String> {
    events.lock().unwrap().clone()
}

fn command_line(request: &Recorded) -> String {
    support::command_argv(request).join(" ")
}

/// A sandbox whose tar command fails with exit code 2, and whose events log tar and snapshot.
async fn failing_archive_sandbox(
    events: &Events,
    persistence: WorkspacePersistence,
    snapshots: &tempfile::TempDir,
) -> (support::MockServer, E2BSandboxClient) {
    let log = events.clone();
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => {
                    MockResponse::json(json!({ "sandboxID": "sb-123", "envdAccessToken": "envd" }))
                }
                path if path.ends_with("/snapshots") => {
                    log.lock().unwrap().push("snapshot".to_string());
                    MockResponse::empty(500)
                }
                "/process.Process/Start" if command_line(request).contains("tar -czf") => {
                    log.lock().unwrap().push("tar".to_string());
                    failed("tar failed", 2)
                }
                "/process.Process/Start" => finished("", 0),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_workspace_persistence(persistence),
    )
    .with_snapshot(SnapshotSpec::Local {
        base_path: snapshots.path().to_path_buf(),
    });
    (server, client)
}

fn failed(stderr: &str, exit_code: i64) -> MockResponse {
    let mut body = frame(0, json!({ "event": { "start": { "pid": 4242 } } }));
    body.extend(frame(
        0,
        json!({ "event": { "data": { "stderr": support::b64(stderr) } } }),
    ));
    body.extend(frame(
        0,
        json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
    ));
    body.extend(frame(2, json!({})));
    MockResponse::streamed("application/connect+json", vec![(body, 0)])
}

/// `test_e2b_persist_workspace_keeps_remounting_and_raises_remount_error_first`: the archive
/// fails, one remount fails too, and the remaining mount is still attached. The remount error is
/// what the caller sees, and the archive failure rides along in the context.
#[tokio::test]
async fn a_failed_remount_keeps_attaching_and_wins_over_the_archive_error() {
    let events: Events = Arc::default();
    let snapshots = tempfile::tempdir().expect("tempdir");
    let (server, client) =
        failing_archive_sandbox(&events, WorkspacePersistence::Tar, &snapshots).await;
    let session = client
        .create(Some(manifest(&events, false, true)))
        .await
        .expect("create");

    let error = persist_session_snapshot(&*session)
        .await
        .expect_err("the archive and a remount fail");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        recorded(&events),
        [
            "unmount:mount1",
            "unmount:mount2",
            "tar",
            "mount_fail:mount2",
            "mount:mount1"
        ]
    );
    assert_eq!(
        error.context.get("cause").map(String::as_str),
        Some("boom while remounting mount2")
    );
    let before = error
        .context
        .get("snapshot_error_before_remount_corruption")
        .expect("the archive failure is kept");
    assert!(before.contains("tar"), "{before}");
    server.stop();
}

/// The same failure over a native snapshot: the platform refuses the snapshot, the second mount
/// cannot be re-attached, and the first one still is.
#[tokio::test]
async fn a_failed_native_snapshot_then_a_failed_remount_still_attaches_the_rest() {
    let events: Events = Arc::default();
    let snapshots = tempfile::tempdir().expect("tempdir");
    let (server, client) =
        failing_archive_sandbox(&events, WorkspacePersistence::Snapshot, &snapshots).await;
    let session = client
        .create(Some(manifest(&events, false, true)))
        .await
        .expect("create");

    let error = persist_session_snapshot(&*session)
        .await
        .expect_err("the snapshot and a remount fail");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        recorded(&events),
        [
            "unmount:mount1",
            "unmount:mount2",
            "snapshot",
            "mount_fail:mount2",
            "mount:mount1"
        ]
    );
    assert_eq!(
        error.context.get("cause").map(String::as_str),
        Some("boom while remounting mount2")
    );
    assert!(
        error
            .context
            .contains_key("snapshot_error_before_remount_corruption"),
        "{error:?}"
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Workspace root
// ---------------------------------------------------------------------------------------------

/// A sandbox whose workspace root exists (`root_exists`) and that logs every command.
async fn root_sandbox(
    root_exists: bool,
    make_dir_status: u16,
    commands: &Events,
) -> (support::MockServer, E2BSandboxClient) {
    let log = commands.clone();
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => {
                    MockResponse::json(json!({ "sandboxID": "sb-123", "envdAccessToken": "envd" }))
                }
                "/filesystem.Filesystem/MakeDir" => {
                    log.lock()
                        .unwrap()
                        .push(format!("make_dir {}", request.json()["path"]));
                    if make_dir_status == 200 {
                        MockResponse::json(json!({}))
                    } else {
                        MockResponse::status(
                            make_dir_status,
                            json!({ "code": "unavailable", "message": "files API unavailable" }),
                        )
                    }
                }
                "/process.Process/Start" => {
                    let line = command_line(request);
                    log.lock().unwrap().push(line.clone());
                    if line == "test -d /workspace" && !root_exists {
                        finished("", 1)
                    } else {
                        finished("", 0)
                    }
                }
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url),
    );
    (server, client)
}

/// `test_e2b_start_skips_files_api_when_workspace_root_exists`: a root that already exists is
/// never created through the filesystem API, so an unavailable API cannot fail `start`.
#[tokio::test]
async fn start_skips_the_files_api_when_the_workspace_root_exists() {
    let commands: Events = Arc::default();
    let (server, client) = root_sandbox(true, 503, &commands).await;
    let session = client.create(None).await.expect("create");

    session.start().await.expect("start");

    assert_eq!(server.count("POST", "/filesystem.Filesystem/MakeDir"), 0);
    let log = recorded(&commands);
    assert_eq!(log[0], "test -d /workspace");
    assert_eq!(log[1], "mkdir -p -- /workspace");
    server.stop();
}

/// A missing root is created through the filesystem API, once: the manifest applier's own root
/// `mkdir` right after `start` does not repeat the call.
#[tokio::test]
async fn start_creates_a_missing_root_through_the_files_api_only_once() {
    let commands: Events = Arc::default();
    let (server, client) = root_sandbox(false, 200, &commands).await;
    let session = client.create(None).await.expect("create");

    session.start().await.expect("start");

    assert_eq!(server.count("POST", "/filesystem.Filesystem/MakeDir"), 1);
    assert_eq!(
        recorded(&commands)[..3],
        [
            "test -d /workspace",
            "make_dir \"/workspace\"",
            "mkdir -p -- /workspace"
        ]
    );
    server.stop();
}

/// `test_e2b_mkdir_recreates_workspace_root_when_readiness_is_stale`: a public `mkdir` of the
/// root goes straight to the filesystem API, with no command in between that needs the root as
/// its working directory.
#[tokio::test]
async fn mkdir_recreates_the_workspace_root_without_running_a_command() {
    let commands: Events = Arc::default();
    let (server, client) = root_sandbox(true, 200, &commands).await;
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");
    let before = recorded(&commands);

    session.mkdir("/workspace", true).await.expect("mkdir");

    let after = recorded(&commands);
    assert_eq!(after[..before.len()], before[..]);
    assert_eq!(after[before.len()..], ["make_dir \"/workspace\""]);
    server.stop();
}

/// `test_e2b_mkdir_probes_the_parent_with_a_posix_path`: without `parents`, the parent is probed
/// with `test -d` and a forward-slash path, and a missing parent fails the call.
#[tokio::test]
async fn mkdir_without_parents_probes_the_parent_with_a_posix_path() {
    let commands: Events = Arc::default();
    let (server, client) = root_sandbox(true, 200, &commands).await;
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");

    session.mkdir("sub/dir", false).await.expect("mkdir");

    let log = recorded(&commands);
    assert!(
        log.contains(&"test -d /workspace/sub".to_string()),
        "{log:?}"
    );
    assert!(log.iter().all(|line| !line.contains('\\')), "{log:?}");
    assert_eq!(log.last().unwrap(), "make_dir \"/workspace/sub/dir\"");
    server.stop();
}

/// The probe is what keeps a plain `mkdir` from creating missing parents.
#[tokio::test]
async fn mkdir_without_parents_fails_when_the_parent_is_missing() {
    let commands: Events = Arc::default();
    let log = commands.clone();
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => {
                    MockResponse::json(json!({ "sandboxID": "sb-123", "envdAccessToken": "envd" }))
                }
                "/filesystem.Filesystem/MakeDir" => {
                    log.lock().unwrap().push("make_dir".to_string());
                    MockResponse::json(json!({}))
                }
                "/process.Process/Start" => {
                    let line = command_line(request);
                    if line == "test -d /workspace/missing" {
                        finished("", 1)
                    } else {
                        finished("", 0)
                    }
                }
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url),
    );
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");
    let made_before = recorded(&commands).len();

    let error = session
        .mkdir("missing/leaf", false)
        .await
        .expect_err("the parent is missing");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert_eq!(recorded(&commands).len(), made_before);
    session
        .mkdir("missing/leaf", true)
        .await
        .expect("with parents the files API creates them");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume clean-up
// ---------------------------------------------------------------------------------------------

fn dir_entry(path: &str) -> Value {
    json!({ "name": path.rsplit('/').next().unwrap(), "path": path, "type": "FILE_TYPE_DIRECTORY" })
}

fn file_entry(path: &str) -> Value {
    json!({ "name": path.rsplit('/').next().unwrap(), "path": path, "type": "FILE_TYPE_FILE" })
}

/// `test_e2b_clear_workspace_root_on_resume_preserves_nested_mounts`: only the entries around a
/// nested mount go; the mount directory and its ancestors stay.
#[tokio::test]
async fn clearing_the_root_on_resume_leaves_the_nested_mount_alone() {
    let events: Events = Arc::default();
    let listed: Events = Arc::default();
    let removed: Events = Arc::default();
    let (list_log, remove_log) = (listed.clone(), removed.clone());
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => {
                    MockResponse::json(json!({ "sandboxID": "sb-123", "envdAccessToken": "envd" }))
                }
                "/filesystem.Filesystem/ListDir" => {
                    let path = request.json()["path"].as_str().unwrap().to_string();
                    list_log.lock().unwrap().push(path.clone());
                    match path.as_str() {
                        "/workspace" => MockResponse::json(json!({ "entries": [
                        dir_entry("/workspace/a"), file_entry("/workspace/root.txt")
                    ] })),
                        "/workspace/a" => MockResponse::json(json!({ "entries": [
                        dir_entry("/workspace/a/b"), file_entry("/workspace/a/local.txt")
                    ] })),
                        other => panic!("unexpected ls path: {other}"),
                    }
                }
                "/filesystem.Filesystem/Remove" => {
                    remove_log
                        .lock()
                        .unwrap()
                        .push(request.json()["path"].as_str().unwrap().to_string());
                    MockResponse::json(json!({}))
                }
                "/process.Process/Start" => finished("", 0),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url),
    );
    let manifest = Manifest::new().with_entry(
        "a/b",
        Entry::mount(RecordingMount::mount("nested", &events, false, false)),
    );
    let session = client.create(Some(manifest)).await.expect("create");

    clear_workspace_root_on_resume(&*session)
        .await
        .expect("clear");

    assert_eq!(recorded(&listed), ["/workspace", "/workspace/a"]);
    let mut gone = recorded(&removed);
    gone.sort();
    assert_eq!(gone, ["/workspace/a/local.txt", "/workspace/root.txt"]);
    assert!(recorded(&events).is_empty(), "no mount was touched");
    server.stop();
}
