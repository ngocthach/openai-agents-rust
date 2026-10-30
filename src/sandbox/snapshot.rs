//! Workspace snapshots. Mirrors `agents.sandbox.snapshot` and
//! `agents.sandbox.snapshot_defaults`.
//!
//! A snapshot stores the gzipped tar archive produced by
//! [`SandboxSession::persist_workspace`](crate::sandbox::SandboxSession::persist_workspace), so a
//! later session can restore the same workspace with
//! [`hydrate_workspace`](crate::sandbox::SandboxSession::hydrate_workspace).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// How long a snapshot in the SDK-managed default directory is kept.
pub const DEFAULT_LOCAL_SNAPSHOT_TTL: Duration = Duration::from_secs(60 * 60 * 24 * 30);

/// Subdirectory of the platform state directory used for default snapshots.
const DEFAULT_LOCAL_SNAPSHOT_SUBDIR: &str = "openai-agents-rust/sandbox/snapshots";

/// Somewhere a workspace archive can be stored and read back.
#[async_trait]
pub trait Snapshot: Send + Sync + fmt::Debug {
    /// The snapshot type, matching the Python `type` discriminator.
    fn snapshot_type(&self) -> &str;

    /// The snapshot identifier.
    fn id(&self) -> &str;

    /// Store a workspace archive.
    async fn persist(&self, data: &[u8]) -> SandboxResult<()>;

    /// Read the stored workspace archive back.
    async fn restore(&self) -> SandboxResult<Vec<u8>>;

    /// Whether a stored archive is available.
    async fn restorable(&self) -> SandboxResult<bool>;
}

/// A snapshot stored as a file on the machine running the SDK.
#[derive(Debug, Clone)]
pub struct LocalSnapshot {
    id: String,
    base_path: PathBuf,
}

impl LocalSnapshot {
    /// A snapshot stored at `<base_path>/<id>.tar`.
    pub fn new(id: impl Into<String>, base_path: impl Into<PathBuf>) -> Self {
        LocalSnapshot {
            id: id.into(),
            base_path: base_path.into(),
        }
    }

    /// The directory holding the snapshot file.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// The snapshot file path, rejecting ids that are not a single path segment.
    pub fn path(&self) -> SandboxResult<PathBuf> {
        let is_segment = !self.id.is_empty()
            && self.id != "."
            && self.id != ".."
            && !self.id.contains('/')
            && !self.id.contains('\\');
        if !is_segment {
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                "LocalSnapshot id must be a single path segment",
            ));
        }
        Ok(self.base_path.join(format!("{}.tar", self.id)))
    }
}

#[async_trait]
impl Snapshot for LocalSnapshot {
    fn snapshot_type(&self) -> &str {
        "local"
    }

    fn id(&self) -> &str {
        &self.id
    }

    async fn persist(&self, data: &[u8]) -> SandboxResult<()> {
        let path = self.path()?;
        let temp_path = path.with_file_name(format!(
            ".{}.{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy(),
            uuid::Uuid::new_v4().simple()
        ));
        let persist = async {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&temp_path, data).await?;
            // Replace atomically so a crash mid-write never leaves a truncated snapshot.
            tokio::fs::rename(&temp_path, &path).await
        };
        match persist.await {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = tokio::fs::remove_file(&temp_path).await;
                Err(SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    format!("Failed to persist snapshot `{}`: {e}", self.id),
                ))
            }
        }
    }

    async fn restore(&self) -> SandboxResult<Vec<u8>> {
        let path = self.path()?;
        tokio::fs::read(&path).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to restore snapshot `{}`: {e}", self.id),
            )
        })
    }

    async fn restorable(&self) -> SandboxResult<bool> {
        let Ok(path) = self.path() else {
            return Ok(false);
        };
        Ok(tokio::fs::metadata(&path)
            .await
            .map(|m| m.is_file())
            .unwrap_or(false))
    }
}

/// A snapshot that stores nothing, used when a session needs no workspace persistence.
#[derive(Debug, Clone)]
pub struct NoopSnapshot {
    id: String,
}

impl NoopSnapshot {
    /// A no-op snapshot with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        NoopSnapshot { id: id.into() }
    }
}

#[async_trait]
impl Snapshot for NoopSnapshot {
    fn snapshot_type(&self) -> &str {
        "noop"
    }

    fn id(&self) -> &str {
        &self.id
    }

    async fn persist(&self, _data: &[u8]) -> SandboxResult<()> {
        Ok(())
    }

    async fn restore(&self) -> SandboxResult<Vec<u8>> {
        Err(SandboxError::new(
            ErrorCode::SnapshotNotRestorable,
            format!("snapshot `{}` does not store workspace archives", self.id),
        ))
    }

    async fn restorable(&self) -> SandboxResult<bool> {
        Ok(false)
    }
}

/// Where a session's snapshots are stored. This is the serializable half of a [`Snapshot`], so
/// it can travel inside [`SandboxSessionState`](crate::sandbox::SandboxSessionState).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SnapshotSpec {
    /// Store nothing.
    #[default]
    Noop,
    /// Store snapshot files under `base_path` on the machine running the SDK.
    Local {
        /// The directory holding the snapshot files.
        base_path: PathBuf,
    },
    /// Store snapshot files through a [`RemoteSnapshotClient`] registered under `client_key`.
    Remote {
        /// The key the client was registered under.
        client_key: String,
    },
}

impl SnapshotSpec {
    /// Build the snapshot for a session id.
    pub fn build(&self, snapshot_id: &str) -> Arc<dyn Snapshot> {
        match self {
            SnapshotSpec::Noop => Arc::new(NoopSnapshot::new(snapshot_id)),
            SnapshotSpec::Local { base_path } => {
                Arc::new(LocalSnapshot::new(snapshot_id, base_path.clone()))
            }
            SnapshotSpec::Remote { client_key } => {
                Arc::new(RemoteSnapshot::new(snapshot_id, client_key.clone()))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Remote snapshots
// ---------------------------------------------------------------------------------------------

/// Somewhere a workspace archive can be stored that is not this machine. Mirrors the client a
/// Python `RemoteSnapshot` resolves out of its session dependencies.
///
/// Implement this for object storage, a provider's own snapshot API, or anything else that
/// survives the process. It is what makes a paused run resumable from a different machine.
#[async_trait]
pub trait RemoteSnapshotClient: Send + Sync + fmt::Debug {
    /// Store `data` under `snapshot_id`.
    async fn upload(&self, snapshot_id: &str, data: &[u8]) -> SandboxResult<()>;

    /// Fetch what was stored under `snapshot_id`.
    async fn download(&self, snapshot_id: &str) -> SandboxResult<Vec<u8>>;

    /// Whether anything is stored under `snapshot_id`.
    async fn exists(&self, snapshot_id: &str) -> SandboxResult<bool>;
}

/// The process-wide registry of remote snapshot clients.
///
/// [`SandboxSessionState`](crate::sandbox::SandboxSessionState) has to survive being written to
/// disk and read back in another process, so it can only carry a *name* for the storage, never
/// the client object. The application registers the client under that name at start-up and a
/// resumed state finds it again. Nothing here is a credential: the client holds those.
static REMOTE_SNAPSHOT_CLIENTS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::BTreeMap<String, Arc<dyn RemoteSnapshotClient>>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::BTreeMap::new()));

/// Register `client` under `client_key`, replacing any client already registered under it.
///
/// Call this before creating or resuming a session whose snapshot spec names `client_key`.
pub fn register_remote_snapshot_client(
    client_key: impl Into<String>,
    client: Arc<dyn RemoteSnapshotClient>,
) {
    if let Ok(mut clients) = REMOTE_SNAPSHOT_CLIENTS.write() {
        clients.insert(client_key.into(), client);
    }
}

/// The client registered under `client_key`, when there is one.
pub fn remote_snapshot_client(client_key: &str) -> Option<Arc<dyn RemoteSnapshotClient>> {
    REMOTE_SNAPSHOT_CLIENTS
        .read()
        .ok()
        .and_then(|clients| clients.get(client_key).cloned())
}

/// A workspace snapshot stored through a [`RemoteSnapshotClient`]. Mirrors Python's
/// `RemoteSnapshot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteSnapshot {
    /// The snapshot id.
    pub id: String,
    /// The key the storage client is registered under.
    pub client_key: String,
}

impl RemoteSnapshot {
    /// A snapshot stored through the client registered under `client_key`.
    pub fn new(id: impl Into<String>, client_key: impl Into<String>) -> Self {
        RemoteSnapshot {
            id: id.into(),
            client_key: client_key.into(),
        }
    }

    /// The registered client, or an error naming what is missing.
    fn client(&self, code: ErrorCode) -> SandboxResult<Arc<dyn RemoteSnapshotClient>> {
        remote_snapshot_client(&self.client_key).ok_or_else(|| {
            SandboxError::new(
                code,
                format!(
                    "no remote snapshot client is registered under `{}`; register one before \
                     creating or resuming this session",
                    self.client_key
                ),
            )
            .with_context("client_key", self.client_key.clone())
        })
    }
}

#[async_trait]
impl Snapshot for RemoteSnapshot {
    fn id(&self) -> &str {
        &self.id
    }

    fn snapshot_type(&self) -> &str {
        "remote"
    }

    async fn persist(&self, data: &[u8]) -> SandboxResult<()> {
        self.client(ErrorCode::SnapshotPersistError)?
            .upload(&self.id, data)
            .await
    }

    async fn restore(&self) -> SandboxResult<Vec<u8>> {
        self.client(ErrorCode::SnapshotRestoreError)?
            .download(&self.id)
            .await
    }

    async fn restorable(&self) -> SandboxResult<bool> {
        // A missing client means nothing is restorable, not that the run should fail: a resume
        // that cannot find its archive falls back to materializing the manifest.
        match remote_snapshot_client(&self.client_key) {
            Some(client) => client.exists(&self.id).await,
            None => Ok(false),
        }
    }
}

/// Build the snapshot for a session, defaulting to [`SnapshotSpec::Noop`].
pub fn resolve_snapshot(spec: Option<&SnapshotSpec>, snapshot_id: &str) -> Arc<dyn Snapshot> {
    spec.cloned().unwrap_or_default().build(snapshot_id)
}

/// The platform directory the SDK stores default snapshots in.
pub fn default_local_snapshot_base_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let base = if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library").join("Application Support"))
    } else if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("APPDATA"))
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join("AppData").join("Local")))
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| home.map(|h| h.join(".local").join("state")))
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join(DEFAULT_LOCAL_SNAPSHOT_SUBDIR)
}

/// Create the SDK-managed default snapshot directory and return a spec pointing at it.
pub async fn resolve_default_local_snapshot_spec() -> SandboxResult<SnapshotSpec> {
    let base_path = default_local_snapshot_base_dir();
    tokio::fs::create_dir_all(&base_path).await.map_err(|e| {
        SandboxError::new(
            ErrorCode::SnapshotPersistError,
            format!(
                "Failed to create the snapshot directory `{}`: {e}",
                base_path.display()
            ),
        )
    })?;
    Ok(SnapshotSpec::Local { base_path })
}

/// Delete stale snapshot files from the SDK-managed default directory.
///
/// This is deliberately limited to `*.tar` files in that one directory: snapshots are not
/// deleted during normal teardown, because a paused run may still need them.
pub async fn cleanup_stale_default_local_snapshots(base_path: &Path, max_age: Duration) {
    let Ok(mut entries) = tokio::fs::read_dir(base_path).await else {
        return;
    };
    let cutoff = SystemTime::now()
        .checked_sub(max_age)
        .unwrap_or(SystemTime::UNIX_EPOCH);
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("tar") {
            continue;
        }
        let Ok(metadata) = entry.metadata().await else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        match metadata.modified() {
            Ok(modified) if modified < cutoff => {
                let _ = tokio::fs::remove_file(&path).await;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct MemoryClient {
        stored: std::sync::Mutex<std::collections::BTreeMap<String, Vec<u8>>>,
    }

    #[async_trait]
    impl RemoteSnapshotClient for MemoryClient {
        async fn upload(&self, snapshot_id: &str, data: &[u8]) -> SandboxResult<()> {
            self.stored
                .lock()
                .unwrap()
                .insert(snapshot_id.to_string(), data.to_vec());
            Ok(())
        }

        async fn download(&self, snapshot_id: &str) -> SandboxResult<Vec<u8>> {
            self.stored
                .lock()
                .unwrap()
                .get(snapshot_id)
                .cloned()
                .ok_or_else(|| SandboxError::new(ErrorCode::SnapshotNotRestorable, "no archive"))
        }

        async fn exists(&self, snapshot_id: &str) -> SandboxResult<bool> {
            Ok(self.stored.lock().unwrap().contains_key(snapshot_id))
        }
    }

    #[tokio::test]
    async fn remote_snapshots_round_trip_through_their_client() {
        register_remote_snapshot_client("test-store", Arc::new(MemoryClient::default()));
        let spec = SnapshotSpec::Remote {
            client_key: "test-store".to_string(),
        };
        let snapshot = spec.build("session-1");
        assert_eq!(snapshot.snapshot_type(), "remote");
        assert!(!snapshot.restorable().await.unwrap());
        snapshot.persist(b"archive").await.unwrap();
        assert!(snapshot.restorable().await.unwrap());
        assert_eq!(snapshot.restore().await.unwrap(), b"archive");
    }

    #[tokio::test]
    async fn an_unregistered_client_is_reported_not_guessed() {
        let snapshot = SnapshotSpec::Remote {
            client_key: "never-registered".to_string(),
        }
        .build("session-2");
        // Nothing can be restored, which a resume handles by materializing the manifest.
        assert!(!snapshot.restorable().await.unwrap());
        // Persisting, which would silently lose the workspace, says exactly what is missing.
        let error = snapshot.persist(b"archive").await.unwrap_err();
        assert_eq!(error.code, ErrorCode::SnapshotPersistError);
        assert!(
            error.message.contains("never-registered"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_remote_spec_survives_serialization() {
        let spec = SnapshotSpec::Remote {
            client_key: "store".to_string(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            serde_json::from_str::<SnapshotSpec>(&json).unwrap(),
            spec,
            "{json}"
        );
    }

    #[tokio::test]
    async fn local_snapshots_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = SnapshotSpec::Local {
            base_path: dir.path().to_path_buf(),
        }
        .build("session-1");
        assert_eq!(snapshot.snapshot_type(), "local");
        assert!(!snapshot.restorable().await.unwrap());
        snapshot.persist(b"archive-bytes").await.unwrap();
        assert!(snapshot.restorable().await.unwrap());
        assert_eq!(snapshot.restore().await.unwrap(), b"archive-bytes");
        assert!(dir.path().join("session-1.tar").is_file());
    }

    #[tokio::test]
    async fn rejects_snapshot_ids_with_separators() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = LocalSnapshot::new("../escape", dir.path());
        assert!(snapshot.path().is_err());
        assert!(snapshot.persist(b"x").await.is_err());
        assert!(!snapshot.restorable().await.unwrap());
    }

    #[tokio::test]
    async fn noop_snapshots_store_nothing() {
        let snapshot = resolve_snapshot(None, "s");
        assert_eq!(snapshot.snapshot_type(), "noop");
        snapshot.persist(b"ignored").await.unwrap();
        assert!(!snapshot.restorable().await.unwrap());
        let err = snapshot.restore().await.unwrap_err();
        assert_eq!(err.code, ErrorCode::SnapshotNotRestorable);
    }

    #[tokio::test]
    async fn cleans_up_stale_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("fresh.tar"), b"x").unwrap();
        std::fs::write(dir.path().join("keep.txt"), b"x").unwrap();
        // A zero max age makes every file stale.
        cleanup_stale_default_local_snapshots(dir.path(), Duration::ZERO).await;
        assert!(!dir.path().join("fresh.tar").exists());
        assert!(dir.path().join("keep.txt").exists());
    }

    #[test]
    fn serializes_snapshot_specs() {
        let spec = SnapshotSpec::Local {
            base_path: PathBuf::from("/tmp/snaps"),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, r#"{"type":"local","base_path":"/tmp/snaps"}"#);
        assert_eq!(serde_json::from_str::<SnapshotSpec>(&json).unwrap(), spec);
        assert_eq!(
            serde_json::to_string(&SnapshotSpec::Noop).unwrap(),
            r#"{"type":"noop"}"#
        );
    }

    #[test]
    fn default_base_dir_is_absolute() {
        assert!(default_local_snapshot_base_dir().is_absolute());
    }
}
