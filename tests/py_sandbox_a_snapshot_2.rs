//! Ports `tests/sandbox/test_snapshot.py` from the Python SDK.
//!
//! Python drives `BaseSandboxSession.stop()` / `start()` through a hand-rolled subclass. Rust
//! keeps that orchestration in the free functions `persist_session_snapshot` and
//! `restore_session_snapshot`, so the tests below drive those through a fake
//! [`SandboxSession`] that runs commands on the host the way Python's `_exec_internal` does.
//! The fingerprint helper script therefore runs for real, against a real temporary workspace.
#![cfg(feature = "sandbox")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{
    ExecRequest, persist_session_snapshot, restore_session_snapshot,
};
use openai_agents_sdk::sandbox::snapshot::{
    LocalSnapshot, NoopSnapshot, RemoteSnapshot, RemoteSnapshotClient, Snapshot, SnapshotSpec,
    register_remote_snapshot_client, resolve_snapshot,
};
use openai_agents_sdk::sandbox::state::{
    SNAPSHOT_FINGERPRINT_VERSION, SNAPSHOT_FINGERPRINT_VERSION_V1, SnapshotFingerprint,
    manifest_digest,
};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxResult, SandboxSession,
    read_cached_snapshot_fingerprint,
};
use sha2::{Digest, Sha256};

// ----------------------------------------------------------------------------------------------
// Test doubles
// ----------------------------------------------------------------------------------------------

/// Python's `TestNoopSnapshot`: a custom snapshot type that stores nothing and cannot restore.
#[derive(Debug)]
struct CustomNoopSnapshot {
    id: String,
}

#[async_trait]
impl Snapshot for CustomNoopSnapshot {
    fn snapshot_type(&self) -> &str {
        "test-noop"
    }
    fn id(&self) -> &str {
        &self.id
    }
    async fn persist(&self, _data: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
    async fn restore(&self) -> SandboxResult<Vec<u8>> {
        Err(openai_agents_sdk::sandbox::SandboxError::new(
            ErrorCode::SnapshotNotRestorable,
            "<test-noop>",
        ))
    }
    async fn restorable(&self) -> SandboxResult<bool> {
        Ok(false)
    }
}

/// Python's `TestRestorableSnapshot`: always restorable, always returns `payload`.
#[derive(Debug)]
struct RestorableSnapshot {
    id: String,
    payload: Vec<u8>,
}

#[async_trait]
impl Snapshot for RestorableSnapshot {
    fn snapshot_type(&self) -> &str {
        "test-restorable"
    }
    fn id(&self) -> &str {
        &self.id
    }
    async fn persist(&self, _data: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
    async fn restore(&self) -> SandboxResult<Vec<u8>> {
        Ok(self.payload.clone())
    }
    async fn restorable(&self) -> SandboxResult<bool> {
        Ok(true)
    }
}

/// Python's `_PersistTrackingSession` + `_ResumeTrackingSession` in one: a session over a real
/// host directory that counts what the snapshot orchestration asks of it.
#[derive(Debug)]
struct TrackingSession {
    id: String,
    manifest: Manifest,
    snapshot: Arc<dyn Snapshot>,
    persist_payload: Vec<u8>,
    /// What `resume_snapshot_fingerprint` reports; `record_snapshot_fingerprint` overwrites it.
    fingerprint: Mutex<Option<SnapshotFingerprint>>,
    persist_workspace_calls: AtomicUsize,
    /// `rm` commands issued: a resume clears the workspace one entry at a time.
    rm_calls: AtomicUsize,
    account_commands: AtomicUsize,
    hydrate_payloads: Mutex<Vec<Vec<u8>>>,
}

impl TrackingSession {
    fn new(snapshot: Arc<dyn Snapshot>, workspace_root: &Path) -> Self {
        // The manifest root must be the resolved path: macOS reports temp dirs under `/var`
        // while the real directory is `/private/var`.
        let root = std::fs::canonicalize(workspace_root).expect("canonical workspace root");
        TrackingSession {
            id: uuid::Uuid::new_v4().to_string(),
            manifest: Manifest::new().with_root(root.to_string_lossy().to_string()),
            snapshot,
            persist_payload: b"tracked".to_vec(),
            fingerprint: Mutex::new(None),
            persist_workspace_calls: AtomicUsize::new(0),
            rm_calls: AtomicUsize::new(0),
            account_commands: AtomicUsize::new(0),
            hydrate_payloads: Mutex::new(Vec::new()),
        }
    }

    fn set_manifest(&mut self, manifest: Manifest) {
        self.manifest = manifest;
    }

    fn edit_manifest(&mut self, edit: impl FnOnce(Manifest) -> Manifest) {
        let current = std::mem::take(&mut self.manifest);
        self.manifest = edit(current);
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            Path::new(self.manifest.root()).join(path)
        }
    }

    fn rm_calls(&self) -> usize {
        self.rm_calls.load(Ordering::SeqCst)
    }

    /// Account-provisioning commands the restore asked for. Python's fake overrides
    /// `provision_manifest_accounts` and counts the call instead of creating real users, so this
    /// fake does the same with the commands rather than running `useradd` on the host.
    fn account_commands(&self) -> usize {
        self.account_commands.load(Ordering::SeqCst)
    }

    fn hydrate_payloads(&self) -> Vec<Vec<u8>> {
        self.hydrate_payloads.lock().expect("payloads").clone()
    }
}

#[async_trait]
impl SandboxSession for TrackingSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "tracking"
    }

    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let argv = request.resolved_argv();
        if argv.first().map(String::as_str) == Some("rm") {
            self.rm_calls.fetch_add(1, Ordering::SeqCst);
        }
        if matches!(
            argv.first().map(String::as_str),
            Some("useradd" | "groupadd" | "usermod")
        ) {
            self.account_commands.fetch_add(1, Ordering::SeqCst);
            return Ok(ExecResult {
                stdout: Vec::new(),
                stderr: Vec::new(),
                exit_code: 0,
            });
        }
        let output = tokio::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .await
            .map_err(|e| {
                openai_agents_sdk::sandbox::SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("spawn failed: {e}"),
                )
            })?;
        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let full = self.resolve(path);
        std::fs::read(&full).map_err(|e| {
            openai_agents_sdk::sandbox::SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("{}: {e}", full.display()),
            )
        })
    }

    /// Re-applying the ephemeral entries after a restore writes real files, so the fake writes
    /// them into the real workspace directory.
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let full = self.resolve(path);
        if let Some(parent) = full.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&full, contents).map_err(|e| {
            openai_agents_sdk::sandbox::SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                format!("{}: {e}", full.display()),
            )
        })
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        self.persist_workspace_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.persist_payload.clone())
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.hydrate_payloads
            .lock()
            .expect("payloads")
            .push(archive.to_vec());
        Ok(())
    }

    fn validate_restorable_archive(&self, _archive: &[u8]) -> SandboxResult<()> {
        // The restorable test snapshots return opaque bytes, not a tar.
        Ok(())
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        self.snapshot.clone()
    }

    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        self.fingerprint.lock().expect("fingerprint").clone()
    }

    fn record_snapshot_fingerprint(&self, record: Option<SnapshotFingerprint>) {
        *self.fingerprint.lock().expect("fingerprint") = record;
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A remote client that records every call, standing in for Python's `_FakeRemoteSnapshotClient`.
#[derive(Debug, Default)]
struct FakeRemoteClient {
    uploads: Mutex<Vec<(String, Vec<u8>)>>,
    downloads: Mutex<Vec<String>>,
    exists_calls: Mutex<Vec<String>>,
    stored: Mutex<std::collections::BTreeMap<String, Vec<u8>>>,
}

#[async_trait]
impl RemoteSnapshotClient for FakeRemoteClient {
    async fn upload(&self, snapshot_id: &str, data: &[u8]) -> SandboxResult<()> {
        self.uploads
            .lock()
            .expect("uploads")
            .push((snapshot_id.to_string(), data.to_vec()));
        self.stored
            .lock()
            .expect("stored")
            .insert(snapshot_id.to_string(), data.to_vec());
        Ok(())
    }

    async fn download(&self, snapshot_id: &str) -> SandboxResult<Vec<u8>> {
        self.downloads
            .lock()
            .expect("downloads")
            .push(snapshot_id.to_string());
        self.stored
            .lock()
            .expect("stored")
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| {
                openai_agents_sdk::sandbox::SandboxError::new(
                    ErrorCode::SnapshotNotRestorable,
                    "no archive",
                )
            })
    }

    async fn exists(&self, snapshot_id: &str) -> SandboxResult<bool> {
        self.exists_calls
            .lock()
            .expect("exists")
            .push(snapshot_id.to_string());
        Ok(self
            .stored
            .lock()
            .expect("stored")
            .contains_key(snapshot_id))
    }
}

/// Register a fresh client under a key unique to the calling test: the registry is process-wide
/// and tests run in parallel, so a shared key would let tests see each other's uploads.
fn register_unique_client(label: &str) -> (String, Arc<FakeRemoteClient>) {
    let key = format!(
        "tests.remote_snapshot_client.{label}.{}",
        uuid::Uuid::new_v4()
    );
    let client = Arc::new(FakeRemoteClient::default());
    register_remote_snapshot_client(key.clone(), client.clone());
    (key, client)
}

fn restorable(id: &str, payload: &[u8]) -> Arc<dyn Snapshot> {
    Arc::new(RestorableSnapshot {
        id: id.to_string(),
        payload: payload.to_vec(),
    })
}

// ----------------------------------------------------------------------------------------------
// Snapshot serialization
// ----------------------------------------------------------------------------------------------

/// `SnapshotBase.model_dump(exclude_unset=True)` keeps the `type` discriminator. In Rust the
/// discriminator is part of the enum's serialized form, so it can never be dropped.
#[test]
fn snapshot_spec_always_serializes_its_type_discriminator() {
    let local = SnapshotSpec::Local {
        base_path: PathBuf::from("/tmp/snapshots"),
    };
    assert_eq!(
        serde_json::to_value(&local).expect("serialize"),
        serde_json::json!({"type": "local", "base_path": "/tmp/snapshots"})
    );
    assert_eq!(
        serde_json::to_value(SnapshotSpec::Noop).expect("serialize"),
        serde_json::json!({"type": "noop"})
    );
}

// ----------------------------------------------------------------------------------------------
// LocalSnapshot
// ----------------------------------------------------------------------------------------------

/// A directory where the archive should be is not a restorable snapshot; only a file is.
#[tokio::test]
async fn local_snapshot_restorable_requires_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = LocalSnapshot::new("local-snapshot", dir.path());
    let snapshot_path = dir.path().join("local-snapshot.tar");

    assert!(!snapshot.restorable().await.expect("restorable"));

    std::fs::create_dir(&snapshot_path).expect("mkdir");
    assert!(!snapshot.restorable().await.expect("restorable"));

    std::fs::remove_dir(&snapshot_path).expect("rmdir");
    std::fs::write(&snapshot_path, b"workspace").expect("write");
    assert!(snapshot.restorable().await.expect("restorable"));
}

/// Ids that are not a single path segment are refused for both persist and restore, and nothing
/// is written anywhere.
#[tokio::test]
async fn local_snapshot_rejects_non_basename_ids() {
    for snapshot_id in [
        "../escape",
        "..\\escape",
        "nested/escape",
        "../",
        "..//",
        "..\\",
        "nested/",
        "nested//",
        "nested\\",
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = LocalSnapshot::new(snapshot_id, dir.path().join("snapshots"));

        let persist = snapshot.persist(b"payload").await.expect_err(snapshot_id);
        assert!(
            persist.message.contains("single path segment"),
            "{snapshot_id}: {}",
            persist.message
        );
        let restore = snapshot.restore().await.expect_err(snapshot_id);
        assert!(
            restore.message.contains("single path segment"),
            "{snapshot_id}: {}",
            restore.message
        );
        assert_eq!(count_tar_files(dir.path()), 0, "{snapshot_id}");
    }
}

fn count_tar_files(root: &Path) -> usize {
    let mut count = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("tar") {
                count += 1;
            }
        }
    }
    count
}

/// A failed persist leaves the previous snapshot intact and no temporary file behind.
///
/// Adapted: Python fails midway through copying a stream. Rust persists from an in-memory
/// buffer, so the failure is induced by making the snapshot directory read-only instead.
#[cfg(unix)]
#[tokio::test]
async fn local_snapshot_persist_is_atomic_on_failure() {
    use std::os::unix::fs::PermissionsExt;

    // A read-only directory does not stop root, which makes the induced failure unreliable.
    if unsafe_is_root() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = LocalSnapshot::new("atomic", dir.path());
    let path = dir.path().join("atomic.tar");
    std::fs::write(&path, b"previous-snapshot").expect("write previous");
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).expect("chmod");

    let error = snapshot
        .persist(b"new-snapshot")
        .await
        .expect_err("persist into a read-only directory must fail");

    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).expect("chmod");
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert_eq!(std::fs::read(&path).expect("read"), b"previous-snapshot");
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read_dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(names, vec!["atomic.tar".to_string()]);
}

#[cfg(unix)]
fn unsafe_is_root() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "0")
        .unwrap_or(false)
}

// ----------------------------------------------------------------------------------------------
// Persisting on stop
// ----------------------------------------------------------------------------------------------

/// A no-op snapshot never archives the workspace.
#[tokio::test]
async fn noop_snapshot_stop_skips_workspace_persist() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = TrackingSession::new(Arc::new(NoopSnapshot::new("noop")), dir.path());

    persist_session_snapshot(&session).await.expect("persist");

    assert_eq!(session.persist_workspace_calls.load(Ordering::SeqCst), 0);
}

/// Any other snapshot type archives the workspace exactly once.
#[tokio::test]
async fn non_noop_snapshot_stop_persists_workspace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = TrackingSession::new(
        Arc::new(CustomNoopSnapshot {
            id: "custom-snapshot".into(),
        }),
        dir.path(),
    );

    persist_session_snapshot(&session).await.expect("persist");

    assert_eq!(session.persist_workspace_calls.load(Ordering::SeqCst), 1);
}

/// Persisting records the fingerprint of the tree that went into the archive, in memory and in
/// the cache file inside the sandbox.
#[tokio::test]
async fn non_noop_snapshot_stop_records_snapshot_fingerprint() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("tracked.txt"), b"tracked").expect("write");
    let session = TrackingSession::new(
        Arc::new(CustomNoopSnapshot {
            id: "custom-snapshot".into(),
        }),
        dir.path(),
    );

    persist_session_snapshot(&session).await.expect("persist");

    let recorded = session
        .resume_snapshot_fingerprint()
        .expect("a fingerprint is recorded");
    assert_eq!(recorded.version, SNAPSHOT_FINGERPRINT_VERSION);
    assert_eq!(recorded.version, "workspace_tar_sha256_v2");
    let cached = read_cached_snapshot_fingerprint(&session)
        .await
        .expect("cache file");
    assert_eq!(cached.fingerprint, recorded.fingerprint);
    assert_eq!(cached.version, recorded.version);
}

// ----------------------------------------------------------------------------------------------
// Resuming from a snapshot
// ----------------------------------------------------------------------------------------------

/// A workspace that still hashes to the recorded fingerprint is left alone: nothing is cleared
/// and the archive is not unpacked over it.
#[tokio::test]
async fn start_skips_snapshot_restore_when_live_workspace_fingerprint_matches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = TrackingSession::new(
        restorable("resume-snapshot", b"restored-workspace"),
        dir.path(),
    );
    std::fs::write(dir.path().join("tracked.txt"), b"tracked").expect("write");

    persist_session_snapshot(&session).await.expect("persist");
    let restored = restore_session_snapshot(&session).await.expect("restore");

    assert!(restored);
    assert_eq!(session.rm_calls(), 0);
    assert!(session.hydrate_payloads().is_empty());
}

/// A workspace that drifted since the snapshot is cleared and rebuilt from the archive.
#[tokio::test]
async fn start_restores_snapshot_when_live_workspace_fingerprint_mismatches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = TrackingSession::new(
        restorable("resume-snapshot", b"restored-workspace"),
        dir.path(),
    );
    let tracked = dir.path().join("tracked.txt");
    std::fs::write(&tracked, b"tracked").expect("write");

    persist_session_snapshot(&session).await.expect("persist");
    std::fs::write(&tracked, b"drifted").expect("drift");
    let restored = restore_session_snapshot(&session).await.expect("restore");

    assert!(restored);
    assert!(
        session.rm_calls() > 0,
        "the stale workspace must be cleared"
    );
    assert_eq!(
        session.hydrate_payloads(),
        vec![b"restored-workspace".to_vec()]
    );
}

/// A manifest that changed between the snapshot and the resume invalidates the fingerprint even
/// when no file changed, because the digest of the manifest topology is part of it.
#[tokio::test]
async fn start_restores_snapshot_when_resume_manifest_changes() {
    for mutation in ["ephemeral_entry", "user"] {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut session = TrackingSession::new(
            restorable("resume-snapshot", b"restored-workspace"),
            dir.path(),
        );
        std::fs::write(dir.path().join("tracked.txt"), b"tracked").expect("write");

        persist_session_snapshot(&session).await.expect("persist");
        match mutation {
            "ephemeral_entry" => session.edit_manifest(|manifest| {
                manifest.with_entry(
                    "ephemeral.txt",
                    Entry::text_file("temp").with_ephemeral(true),
                )
            }),
            _ => session.edit_manifest(|manifest| manifest.with_user("sandbox-user")),
        }
        restore_session_snapshot(&session).await.expect("restore");

        assert!(
            session.rm_calls() > 0,
            "{mutation}: workspace must be cleared"
        );
        assert_eq!(
            session.hydrate_payloads(),
            vec![b"restored-workspace".to_vec()],
            "{mutation}"
        );
        // The archive restores files, not users, so a restored workspace on a fresh system gets
        // its manifest's accounts provisioned; a manifest with none provisions nothing.
        assert_eq!(
            session.account_commands() > 0,
            mutation == "user",
            "{mutation}: accounts are provisioned on restore only when the manifest has some"
        );
    }
}

/// A record written by the released v1 algorithm is compared with *that* algorithm, so a
/// preserved workspace is not restored over live edits just because the algorithm moved on.
/// When the workspace did drift, the archive is restored; afterwards a new persist records v2.
#[tokio::test]
async fn start_compares_legacy_fingerprint_before_restoring_workspace() {
    for workspace_drifted in [false, true] {
        let dir = tempfile::tempdir().expect("tempdir");
        let workspace = std::fs::canonicalize(dir.path()).expect("canonical");
        std::fs::create_dir_all(workspace.join("data")).expect("mkdir data");
        std::fs::create_dir_all(workspace.join("app/data")).expect("mkdir app/data");
        let durable = workspace.join("app/data/users.csv");
        std::fs::write(&durable, b"id,name").expect("write durable");
        let ordinary = workspace.join("app/main.py");
        std::fs::write(&ordinary, b"print(1)").expect("write ordinary");
        std::fs::write(workspace.join("data/scratch.txt"), b"scratch").expect("write scratch");

        // Capture the released archive and fingerprint independently of the current helper.
        let legacy_archive = std::process::Command::new("tar")
            .args(["--exclude=data", "--exclude=./data", "-C"])
            .arg(&workspace)
            .args(["-cf", "-", "."])
            .output()
            .expect("tar")
            .stdout;

        let mut session = TrackingSession::new(restorable("legacy", &legacy_archive), &workspace);
        let manifest = Manifest::new()
            .with_root(workspace.to_string_lossy().to_string())
            .with_entry("data", Entry::dir().with_ephemeral(true));
        session.set_manifest(manifest.clone());
        let archive_hash = hex(&Sha256::digest(&legacy_archive));
        let legacy_hash = hex(&Sha256::digest(
            format!("{archive_hash}\n{}\n", manifest_digest(&manifest)).as_bytes(),
        ));
        session.record_snapshot_fingerprint(Some(SnapshotFingerprint {
            fingerprint: legacy_hash,
            version: SNAPSHOT_FINGERPRINT_VERSION_V1.to_string(),
        }));
        if workspace_drifted {
            std::fs::write(&ordinary, b"print(2)").expect("drift");
        }

        restore_session_snapshot(&session).await.expect("restore");

        assert_eq!(
            session.rm_calls() > 0,
            workspace_drifted,
            "drifted={workspace_drifted}"
        );
        if workspace_drifted {
            assert_eq!(session.hydrate_payloads(), vec![legacy_archive.clone()]);
        } else {
            assert!(session.hydrate_payloads().is_empty());
            assert_eq!(std::fs::read(&durable).expect("read"), b"id,name");
            // The next persist records the current algorithm.
            persist_session_snapshot(&session).await.expect("persist");
            assert_eq!(
                session
                    .resume_snapshot_fingerprint()
                    .expect("fingerprint")
                    .version,
                "workspace_tar_sha256_v2"
            );
        }
    }
}

// ----------------------------------------------------------------------------------------------
// What a resume materializes when no archive is restorable (Unix-local backend)
// ----------------------------------------------------------------------------------------------

fn start_mode_manifest() -> Manifest {
    Manifest::new()
        .with_entry("durable.txt", Entry::text_file("durable"))
        .with_entry(
            "ephemeral.txt",
            Entry::text_file("ephemeral").with_ephemeral(true),
        )
}

/// A fresh session materializes the whole manifest.
#[tokio::test]
async fn start_applies_full_manifest_for_fresh_non_restorable_backend() {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(start_mode_manifest()))
        .await
        .expect("create");
    session.start().await.expect("start");

    assert_eq!(
        session.read_file("durable.txt").await.expect("durable"),
        b"durable"
    );
    assert_eq!(
        session.read_file("ephemeral.txt").await.expect("ephemeral"),
        b"ephemeral"
    );
    client.delete(session).await.expect("delete");
}

/// A resumed session whose workspace survived and has no archive to restore re-applies only the
/// ephemeral entries: the durable files it still holds are not rewritten.
#[tokio::test]
async fn start_reapplies_only_ephemeral_manifest_for_preserved_non_restorable_backend() {
    let manifest = start_mode_manifest();
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");
    let state = session.session_state().with_workspace_root_ready(true);
    // Remove both files and leave a marker, so the workspace is populated but neither entry is.
    session
        .write_file("marker.txt", b"preserved")
        .await
        .expect("marker");
    session.rm("durable.txt", false).await.expect("rm durable");
    session
        .rm("ephemeral.txt", false)
        .await
        .expect("rm ephemeral");

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");

    assert!(
        resumed.read_file("durable.txt").await.is_err(),
        "a preserved workspace must not have its durable entries re-applied"
    );
    assert_eq!(
        resumed.read_file("ephemeral.txt").await.expect("ephemeral"),
        b"ephemeral"
    );
    assert_eq!(
        resumed.read_file("marker.txt").await.expect("marker"),
        b"preserved"
    );
    client.delete(resumed).await.expect("delete");
}

/// A resumed session whose workspace is gone cannot trust anything and materializes the whole
/// manifest instead of handing the model an empty workspace.
#[tokio::test]
async fn start_applies_full_manifest_when_preserved_non_restorable_workspace_unproven() {
    let manifest = start_mode_manifest();
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");
    let state = session.session_state().with_workspace_root_ready(true);
    let root = PathBuf::from(session.workspace_root());
    std::fs::remove_dir_all(&root).expect("evict the workspace");

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");

    assert_eq!(
        resumed.read_file("durable.txt").await.expect("durable"),
        b"durable"
    );
    assert_eq!(
        resumed.read_file("ephemeral.txt").await.expect("ephemeral"),
        b"ephemeral"
    );
    client.delete(resumed).await.expect("delete");
}

/// DEFERRED: Python re-applies only the ephemeral entries when the backend reports its workspace
/// preserved *even if* `workspace_root_ready` is false, because a probe of the root succeeds.
/// The Unix-local backend (`src/sandbox/sandboxes/unix_local.rs`, `start`) instead requires
/// `materialized` (seeded from `workspace_root_ready`), so a state recording an unfinished
/// materialization re-applies the whole manifest. That is the safer reading of the same state
/// and is asserted as current behavior.
#[tokio::test]
async fn start_with_unready_workspace_state_applies_the_full_manifest() {
    let manifest = start_mode_manifest();
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");
    let state = session.session_state().with_workspace_root_ready(false);
    session
        .write_file("marker.txt", b"preserved")
        .await
        .expect("marker");
    session.rm("durable.txt", false).await.expect("rm durable");

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");

    assert_eq!(
        resumed.read_file("durable.txt").await.expect("durable"),
        b"durable"
    );
    client.delete(resumed).await.expect("delete");
}

// ----------------------------------------------------------------------------------------------
// Remote snapshots
// ----------------------------------------------------------------------------------------------

/// Persist, restorable and restore all go through the registered client.
#[tokio::test]
async fn remote_snapshot_persist_restore_and_restorable_use_the_registered_client() {
    let (key, client) = register_unique_client("persist-restore");
    let snapshot = RemoteSnapshot::new("snap-123", key);

    assert!(!snapshot.restorable().await.expect("restorable"));

    snapshot.persist(b"workspace-tar").await.expect("persist");

    assert_eq!(
        *client.uploads.lock().expect("uploads"),
        vec![("snap-123".to_string(), b"workspace-tar".to_vec())]
    );
    assert!(snapshot.restorable().await.expect("restorable"));
    assert_eq!(
        *client.exists_calls.lock().expect("exists"),
        vec!["snap-123".to_string(), "snap-123".to_string()]
    );

    let restored = snapshot.restore().await.expect("restore");

    assert_eq!(
        *client.downloads.lock().expect("downloads"),
        vec!["snap-123".to_string()]
    );
    assert_eq!(restored, b"workspace-tar");
}

/// A remote spec builds a remote snapshot carrying the session's snapshot id and the client key.
#[test]
fn remote_snapshot_spec_builds_remote_snapshot() {
    let spec = SnapshotSpec::Remote {
        client_key: "tests.remote_snapshot_client".into(),
    };
    let snapshot = resolve_snapshot(Some(&spec), "snap-123");

    assert_eq!(snapshot.snapshot_type(), "remote");
    assert_eq!(snapshot.id(), "snap-123");
    // The key is observable through the spec the snapshot was built from.
    assert_eq!(
        spec,
        SnapshotSpec::Remote {
            client_key: "tests.remote_snapshot_client".into()
        }
    );
}

/// A session whose snapshot is remote uploads the archive through the registered client when it
/// is persisted. Python reaches the client through `set_dependencies`; Rust resolves it by key.
#[tokio::test]
async fn session_persist_uploads_through_the_registered_remote_client() {
    let (key, client) = register_unique_client("session-persist");
    let dir = tempfile::tempdir().expect("tempdir");
    let session = TrackingSession::new(Arc::new(RemoteSnapshot::new("snap-123", key)), dir.path());

    persist_session_snapshot(&session).await.expect("persist");

    assert_eq!(
        *client.uploads.lock().expect("uploads"),
        vec![("snap-123".to_string(), b"tracked".to_vec())]
    );
}

/// With no client registered under the key, persisting says exactly what is missing, and a
/// resume sees nothing restorable rather than failing.
#[tokio::test]
async fn remote_snapshot_without_a_registered_client_is_reported() {
    let snapshot = RemoteSnapshot::new(
        "snap-404",
        format!("never-registered-{}", uuid::Uuid::new_v4()),
    );

    assert!(!snapshot.restorable().await.expect("restorable"));
    let error = snapshot.persist(b"x").await.expect_err("persist must fail");
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(
        error.message.contains("never-registered"),
        "{}",
        error.message
    );
}
