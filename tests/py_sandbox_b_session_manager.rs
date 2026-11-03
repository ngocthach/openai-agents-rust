//! Ports of `tests/sandbox/test_session_manager.py`: how a new session's snapshot spec is chosen.
#![cfg(feature = "sandbox")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use openai_agents_sdk::sandbox::runtime::resolve_snapshot_spec_with;
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError};

/// Python: `test_session_manager_uses_custom_snapshot_spec_without_resolving_default`.
#[tokio::test]
async fn session_manager_uses_custom_snapshot_spec_without_resolving_default() {
    let called = AtomicBool::new(false);
    let custom = SnapshotSpec::Local {
        base_path: PathBuf::from("/tmp/custom-sandbox-snapshots"),
    };

    let resolved = resolve_snapshot_spec_with(Some(&custom), || async {
        called.store(true, Ordering::SeqCst);
        Err(SandboxError::new(
            ErrorCode::SnapshotPersistError,
            "default snapshot resolution should not run",
        ))
    })
    .await;

    assert_eq!(resolved.unwrap(), custom);
    assert!(!called.load(Ordering::SeqCst));
}

/// Python: `test_session_manager_falls_back_to_noop_when_default_snapshot_resolution_fails`.
#[tokio::test]
async fn session_manager_falls_back_to_noop_when_default_snapshot_resolution_fails() {
    let resolved = resolve_snapshot_spec_with(None, || async {
        Err(SandboxError::new(
            ErrorCode::SnapshotPersistError,
            "read-only home",
        ))
    })
    .await;

    assert_eq!(resolved.unwrap(), SnapshotSpec::Noop);
}

/// Only an I/O failure falls back to no snapshots; any other error reaches the caller.
#[tokio::test]
async fn session_manager_propagates_non_io_default_snapshot_errors() {
    let resolved = resolve_snapshot_spec_with(None, || async {
        Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "bad default snapshot configuration",
        ))
    })
    .await;

    assert_eq!(resolved.unwrap_err().code, ErrorCode::SandboxConfigInvalid);
}
