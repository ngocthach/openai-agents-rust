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
}

impl SnapshotSpec {
    /// Build the snapshot for a session id.
    pub fn build(&self, snapshot_id: &str) -> Arc<dyn Snapshot> {
        match self {
            SnapshotSpec::Noop => Arc::new(NoopSnapshot::new(snapshot_id)),
            SnapshotSpec::Local { base_path } => {
                Arc::new(LocalSnapshot::new(snapshot_id, base_path.clone()))
            }
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
