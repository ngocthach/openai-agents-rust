//! Port of `tests/sandbox/test_mount_lifecycle.py`: the detach, operate, re-attach sequence a
//! snapshot runs around its archive. The three cancellation rows are Python-only (they shield
//! asyncio tasks from `cancel()`; a dropped Rust future simply stops).
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::session::{
    ExecRequest, SandboxSession, with_ephemeral_mounts_removed,
};
use openai_agents_sdk::sandbox::types::ExecResult;

type Events = Arc<Mutex<Vec<String>>>;

#[derive(Debug)]
struct FakeStrategy {
    events: Events,
    name: &'static str,
    fail_teardown: bool,
    fail_restore: bool,
}

#[async_trait]
impl MountStrategy for FakeStrategy {
    fn strategy_type(&self) -> &str {
        "fake"
    }
    async fn activate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }
    async fn deactivate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }
    async fn teardown_for_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        _: &str,
    ) -> SandboxResult<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("teardown:{}", self.name));
        if self.fail_teardown {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!("teardown failed: {}", self.name),
            ));
        }
        Ok(())
    }
    async fn restore_after_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        _: &str,
    ) -> SandboxResult<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("restore:{}", self.name));
        if self.fail_restore {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!("restore failed: {}", self.name),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
    shutdowns: Mutex<usize>,
}

#[async_trait]
impl SandboxSession for FakeSession {
    fn id(&self) -> &str {
        "fake"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        *self.shutdowns.lock().unwrap() += 1;
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, _: ExecRequest) -> SandboxResult<ExecResult> {
        Ok(ExecResult::default())
    }
    async fn read_file(&self, _: &str) -> SandboxResult<Vec<u8>> {
        unreachable!()
    }
    async fn write_file(&self, _: &str, _: &[u8]) -> SandboxResult<()> {
        unreachable!()
    }
    async fn hydrate_workspace(&self, _: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn mount(events: &Events, name: &'static str, fail_teardown: bool, fail_restore: bool) -> Mount {
    Mount::new(Arc::new(FakeStrategy {
        events: events.clone(),
        name,
        fail_teardown,
        fail_restore,
    }))
}

fn session() -> FakeSession {
    FakeSession {
        manifest: Manifest::new().with_root("/workspace"),
        shutdowns: Mutex::new(0),
    }
}

fn events_of(events: &Events) -> Vec<String> {
    events.lock().unwrap().clone()
}

/// Python: `test_with_ephemeral_mounts_removed_restores_in_reverse_order`.
#[tokio::test]
async fn mounts_are_restored_in_reverse_order() {
    let events: Events = Arc::default();
    let left = mount(&events, "left", false, false);
    let right = mount(&events, "right", false, false);
    let session = session();
    let log = events.clone();

    let result = with_ephemeral_mounts_removed(
        &session,
        vec![
            (&left, "/workspace/left".into()),
            (&right, "/workspace/right".into()),
        ],
        || async move {
            log.lock().unwrap().push("operation".to_string());
            Ok("persisted")
        },
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        Some("snapshot_error_before_remount_corruption"),
        true,
    )
    .await
    .unwrap();

    assert_eq!(result, "persisted");
    assert_eq!(
        events_of(&events),
        [
            "teardown:left",
            "teardown:right",
            "operation",
            "restore:right",
            "restore:left"
        ]
    );
}

/// Python: `test_with_ephemeral_mounts_removed_reports_restore_error_after_operation_error`.
#[tokio::test]
async fn a_restore_error_wins_and_keeps_the_operation_error() {
    let events: Events = Arc::default();
    let broken = mount(&events, "mount", false, true);
    let session = session();
    let log = events.clone();

    let error = with_ephemeral_mounts_removed::<(), _, _>(
        &session,
        vec![(&broken, "/workspace/mount".into())],
        || async move {
            log.lock().unwrap().push("operation".to_string());
            Err(SandboxError::new(
                ErrorCode::WorkspaceArchiveReadError,
                "persist failed",
            ))
        },
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        Some("snapshot_error_before_remount_corruption"),
        true,
    )
    .await
    .unwrap_err();

    assert_eq!(
        events_of(&events),
        ["teardown:mount", "operation", "restore:mount"]
    );
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        error
            .context
            .get("snapshot_error_before_remount_corruption")
            .map(String::as_str),
        Some("persist failed")
    );
    assert!(error.context["cause"].contains("restore failed"));
    assert_eq!(*session.shutdowns.lock().unwrap(), 1);
}

/// Python: `test_with_ephemeral_mounts_removed_restores_after_unexpected_operation_error`. The
/// operation's own error comes back unchanged and the session is not shut down.
#[tokio::test]
async fn an_operation_error_restores_and_is_returned_unchanged() {
    let events: Events = Arc::default();
    let healthy = mount(&events, "mount", false, false);
    let session = session();
    let log = events.clone();

    let error = with_ephemeral_mounts_removed::<(), _, _>(
        &session,
        vec![(&healthy, "/workspace/mount".into())],
        || async move {
            log.lock().unwrap().push("operation".to_string());
            Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "unexpected persistence failure",
            ))
        },
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        None,
        true,
    )
    .await
    .unwrap_err();

    assert_eq!(error.message, "unexpected persistence failure");
    assert_eq!(
        events_of(&events),
        ["teardown:mount", "operation", "restore:mount"]
    );
    assert_eq!(*session.shutdowns.lock().unwrap(), 0);
}

/// Python: `test_with_ephemeral_mounts_removed_restores_before_ambiguous_shutdown`.
#[tokio::test]
async fn a_failed_detach_restores_the_detached_mounts_then_shuts_down() {
    let events: Events = Arc::default();
    let left = mount(&events, "left", false, false);
    let right = mount(&events, "right", true, false);
    let session = session();

    let error = with_ephemeral_mounts_removed::<(), _, _>(
        &session,
        vec![
            (&left, "/workspace/left".into()),
            (&right, "/workspace/right".into()),
        ],
        || async { panic!("operation must not run after a teardown failure") },
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        None,
        true,
    )
    .await
    .unwrap_err();

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        events_of(&events),
        ["teardown:left", "teardown:right", "restore:left"]
    );
    assert_eq!(*session.shutdowns.lock().unwrap(), 1);
}

/// `restore_on_success=false` leaves the mounts detached, for a caller about to tear down.
#[tokio::test]
async fn restore_on_success_can_be_declined() {
    let events: Events = Arc::default();
    let only = mount(&events, "only", false, false);
    let session = session();

    with_ephemeral_mounts_removed(
        &session,
        vec![(&only, "/workspace/only".into())],
        || async { Ok(()) },
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        None,
        false,
    )
    .await
    .unwrap();

    assert_eq!(events_of(&events), ["teardown:only"]);
}
