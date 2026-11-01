//! Port of `test_local_snapshot_persist_is_atomic_on_copy_failure` from
//! `tests/sandbox/test_snapshot.py`. Python feeds `persist` a stream that fails part-way; a Rust
//! snapshot receives the whole archive as bytes, so the failure it can meet is the final write
//! to the destination. A destination that cannot be replaced must leave the previous snapshot in
//! place and no temporary file behind, and report a persist error.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::ErrorCode;
use openai_agents_sdk::sandbox::snapshot::{LocalSnapshot, Snapshot};

#[tokio::test]
async fn a_failed_persist_keeps_the_previous_snapshot_and_leaves_no_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot = LocalSnapshot::new("atomic", dir.path());

    // A first persist succeeds, so there is a previous snapshot to protect.
    snapshot.persist(b"previous-snapshot").await.unwrap();
    assert_eq!(snapshot.restore().await.unwrap(), b"previous-snapshot");

    // Replace the snapshot file by a non-empty directory: the final rename cannot succeed.
    let path = dir.path().join("atomic.tar");
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("keep"), b"previous-snapshot").unwrap();

    let error = snapshot.persist(b"new-snapshot").await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert_eq!(
        std::fs::read(path.join("keep")).unwrap(),
        b"previous-snapshot"
    );

    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["atomic.tar".to_string()],
        "no temporary file may remain"
    );
}
