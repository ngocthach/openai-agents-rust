//! Parity ports of the E2B mount transitions around a workspace persist in
//! `tests/extensions/sandbox/test_e2b.py`: `persist_workspace_remounts_mounts_after_snapshot`
//! and `persist_workspace_remounts_prior_mounts_after_unmount_failure`.
//!
//! The persist goes through `persist_session_snapshot`, the flow that detaches the mounts of a
//! provider whose strategies do not detach natively.
//!
//! The sandbox is a local mock server speaking the platform API and the envd Connect protocol;
//! the mounts are recording strategies, so the order of their detach and attach calls relative to
//! the archive or snapshot request is what is asserted.
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
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
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

/// A sandbox whose snapshot request and tar command are written into `events` as they arrive.
async fn sandbox(
    events: &Events,
    persistence: WorkspacePersistence,
    snapshot_status: u16,
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
                    if snapshot_status == 200 {
                        MockResponse::json(json!({ "snapshotID": "snap-1" }))
                    } else {
                        MockResponse::empty(snapshot_status)
                    }
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

/// `test_e2b_persist_workspace_remounts_mounts_after_snapshot`: a native snapshot is taken with
/// the mounts detached, and they are attached again in reverse order afterwards.
#[tokio::test]
async fn a_native_snapshot_is_taken_with_the_mounts_detached_and_they_come_back() {
    let events: Events = Arc::default();
    let snapshots = tempfile::tempdir().expect("tempdir");
    let (server, client) = sandbox(&events, WorkspacePersistence::Snapshot, 200, &snapshots).await;
    let session = client
        .create(Some(manifest(&events, false, false)))
        .await
        .expect("create");

    persist_session_snapshot(&*session).await.expect("persist");

    assert_eq!(
        recorded(&events),
        [
            "unmount:mount1",
            "unmount:mount2",
            "snapshot",
            "mount:mount2",
            "mount:mount1"
        ]
    );
    server.stop();
}

/// `test_e2b_persist_workspace_remounts_prior_mounts_after_unmount_failure`: a mount that cannot
/// be detached stops the sequence before any snapshot, and the mounts already detached are put
/// back.
#[tokio::test]
async fn a_failed_detach_attaches_the_earlier_mounts_again_and_takes_no_snapshot() {
    let events: Events = Arc::default();
    let snapshots = tempfile::tempdir().expect("tempdir");
    let (server, client) = sandbox(&events, WorkspacePersistence::Snapshot, 200, &snapshots).await;
    let session = client
        .create(Some(manifest(&events, true, false)))
        .await
        .expect("create");

    let error = persist_session_snapshot(&*session)
        .await
        .expect_err("detach fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        recorded(&events),
        ["unmount:mount1", "unmount_fail:mount2", "mount:mount1"]
    );
    assert_eq!(server.count("POST", "/sandboxes/sb-123/snapshots"), 0);
    server.stop();
}
