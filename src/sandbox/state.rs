//! Persistable sandbox session state. Mirrors
//! `agents.sandbox.session.sandbox_session_state`.
//!
//! # Design
//!
//! A paused run has two halves of state: the agent loop state, which
//! [`RunState`](crate::RunState) already serializes, and the sandbox workspace. This module adds
//! the second half.
//!
//! [`SandboxSessionState`] is a small JSON document the caller persists next to the `RunState`.
//! It is deliberately **credential-free and authority-free**:
//!
//! - It never carries manifest contents, host paths to copy from, mount credentials, or
//!   environment values. It carries only the backend identity, the workspace root, the snapshot
//!   location and a digest of the manifest topology.
//! - On resume the caller supplies the same trusted [`Manifest`] again, built by application
//!   code. The digest is checked against it, so a tampered state cannot make the SDK read new
//!   host paths or mount new storage.
//! - A workspace root is treated as owned by the SDK (and therefore deletable) only when it
//!   still looks like an SDK-created temporary directory, so an edited state cannot turn
//!   teardown into `rm -rf` on an arbitrary directory.
//!
//! This mirrors Python's `sanitize_manifest_mount_authority` /
//! `rebind_manifest_mount_authority` pair without serializing the manifest itself.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::sandbox::entries::EntryKind;
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::snapshot::SnapshotSpec;

/// Prefix of the temporary workspace directories the SDK creates and may delete.
pub const OWNED_WORKSPACE_PREFIX: &str = "openai-agents-sandbox-";

/// Version of the session-state payload format.
pub const SESSION_STATE_SCHEMA_VERSION: &str = "openai-agents-sandbox-state-v1";

/// The credential-free description of a live sandbox session, for the caller to persist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxSessionState {
    /// Payload format version.
    #[serde(rename = "$schemaVersion")]
    pub schema_version: String,
    /// The backend that owns the session (`unix_local`, `docker`, ...).
    pub backend: String,
    /// A stable identifier for the session.
    pub session_id: String,
    /// The absolute workspace root inside the sandbox.
    pub workspace_root: String,
    /// Whether the SDK created the workspace root and may remove it.
    #[serde(default)]
    pub workspace_root_owned: bool,
    /// Backend-specific resource identity, such as a Docker container id.
    #[serde(default)]
    pub provider_id: Option<String>,
    /// Where workspace archives are stored.
    #[serde(default)]
    pub snapshot: SnapshotSpec,
    /// The snapshot identifier for this session.
    pub snapshot_id: String,
    /// Digest of the manifest topology, checked when the session resumes.
    pub manifest_digest: String,
    /// Ports exposed by the session.
    #[serde(default)]
    pub exposed_ports: Vec<u16>,
}

impl SandboxSessionState {
    /// Build state for a session.
    pub fn new(
        backend: impl Into<String>,
        session_id: impl Into<String>,
        workspace_root: impl Into<String>,
        manifest: &Manifest,
    ) -> Self {
        let session_id = session_id.into();
        SandboxSessionState {
            schema_version: SESSION_STATE_SCHEMA_VERSION.to_string(),
            backend: backend.into(),
            snapshot_id: session_id.clone(),
            session_id,
            workspace_root: workspace_root.into(),
            workspace_root_owned: false,
            provider_id: None,
            snapshot: SnapshotSpec::Noop,
            manifest_digest: manifest_digest(manifest),
            exposed_ports: Vec::new(),
        }
    }

    /// Mark the workspace root as created by the SDK.
    pub fn with_workspace_root_owned(mut self, owned: bool) -> Self {
        self.workspace_root_owned = owned;
        self
    }

    /// Attach the backend resource identity.
    pub fn with_provider_id(mut self, provider_id: Option<String>) -> Self {
        self.provider_id = provider_id;
        self
    }

    /// Attach the snapshot location.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = snapshot;
        self
    }

    /// Serialize to a JSON string the caller can persist.
    pub fn to_json_string(&self) -> SandboxResult<String> {
        serde_json::to_string(self).map_err(|e| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                format!("Failed to serialize sandbox session state: {e}"),
            )
        })
    }

    /// Parse persisted session state.
    pub fn from_json_str(payload: &str) -> SandboxResult<Self> {
        let state: SandboxSessionState = serde_json::from_str(payload).map_err(|e| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                format!("Failed to parse sandbox session state: {e}"),
            )
        })?;
        if state.schema_version != SESSION_STATE_SCHEMA_VERSION {
            return Err(SandboxError::new(
                ErrorCode::SessionStateInvalid,
                format!(
                    "Unsupported sandbox session state version `{}`",
                    state.schema_version
                ),
            ));
        }
        Ok(state)
    }

    /// Check that `manifest` is the same topology this session was created from.
    pub fn verify_manifest(&self, manifest: &Manifest) -> SandboxResult<()> {
        let digest = manifest_digest(manifest);
        if digest != self.manifest_digest {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                "the manifest supplied on resume does not match the one the sandbox session was \
                 created from",
            ));
        }
        Ok(())
    }

    /// Whether teardown may remove the workspace root.
    ///
    /// The flag alone is not trusted: a persisted state is caller-held data, so the path must
    /// still look like an SDK-created temporary workspace. Otherwise a hand-edited state could
    /// point teardown at a home directory.
    pub fn may_remove_workspace_root(&self) -> bool {
        self.workspace_root_owned && is_sdk_owned_workspace_root(Path::new(&self.workspace_root))
    }
}

/// Whether `path` is a temporary workspace directory the SDK created.
pub fn is_sdk_owned_workspace_root(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if !name.starts_with(OWNED_WORKSPACE_PREFIX) {
        return false;
    }
    let Some(parent) = path.parent() else {
        return false;
    };
    let temp = std::env::temp_dir();
    // Compare both the raw and the resolved temp dir: macOS resolves `/var` to `/private/var`.
    let resolved_temp = std::fs::canonicalize(&temp).unwrap_or_else(|_| temp.clone());
    let resolved_parent = std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    resolved_parent == resolved_temp || parent == temp
}

/// A stable digest of the manifest topology.
///
/// The digest covers what resume has to agree on — the entry tree and its types, mount placement
/// and strategy, accounts and environment variable names — and deliberately excludes file
/// contents, host source paths and environment values, so it can be written to disk without
/// leaking anything.
///
/// Entry paths are hashed relative to the workspace root, and the root itself is not hashed: a
/// backend that owns its workspace (such as Unix-local) replaces the root with a real directory
/// when it creates the session, and resume has to recognise the caller's original manifest. The
/// root travels in [`SandboxSessionState::workspace_root`] instead.
pub fn manifest_digest(manifest: &Manifest) -> String {
    let root = manifest.root().trim_end_matches('/').to_string();
    let mut hasher = Sha256::new();
    hasher.update(SESSION_STATE_SCHEMA_VERSION.as_bytes());
    hasher.update([0]);
    for (dest, entry) in manifest.iter_entries() {
        let relative = dest
            .strip_prefix(&root)
            .map(|r| r.trim_start_matches('/'))
            .unwrap_or(dest.as_str());
        hasher.update(relative.as_bytes());
        hasher.update([0x1f]);
        hasher.update(entry.type_name().as_bytes());
        hasher.update([0x1f]);
        hasher.update([u8::from(entry.is_dir()), u8::from(entry.ephemeral)]);
        hasher.update(entry.permissions.octal_string().as_bytes());
        if let EntryKind::GitRepo {
            host,
            repo,
            git_ref,
            ..
        } = &entry.kind
        {
            hasher.update(host.as_bytes());
            hasher.update([0x1f]);
            hasher.update(repo.as_bytes());
            hasher.update([0x1f]);
            hasher.update(git_ref.as_bytes());
        }
        if let Some(mount) = entry.as_mount() {
            hasher.update(mount.strategy.strategy_type().as_bytes());
            hasher.update([0x1f]);
            hasher.update(mount.mount_path.as_deref().unwrap_or("").as_bytes());
            hasher.update([u8::from(mount.read_only)]);
        }
        hasher.update([0]);
    }
    for user in manifest.users() {
        hasher.update(user.name.as_bytes());
        hasher.update([0]);
    }
    for group in manifest.groups() {
        hasher.update(group.name.as_bytes());
        hasher.update([0]);
    }
    for key in manifest.environment().keys() {
        hasher.update(key.as_bytes());
        hasher.update([0]);
    }
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::entries::Entry;
    use crate::sandbox::mounts::{BindMountStrategy, MountPatternConfig};

    fn state_for(manifest: &Manifest) -> SandboxSessionState {
        SandboxSessionState::new("unix_local", "sid", manifest.root(), manifest)
    }

    #[test]
    fn round_trips_through_json() {
        let manifest = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));
        let state = state_for(&manifest)
            .with_provider_id(Some("container-1".into()))
            .with_snapshot(SnapshotSpec::Local {
                base_path: "/tmp/snaps".into(),
            });
        let json = state.to_json_string().unwrap();
        assert_eq!(SandboxSessionState::from_json_str(&json).unwrap(), state);
        assert!(json.contains("$schemaVersion"));
    }

    #[test]
    fn rejects_unknown_versions() {
        let err = SandboxSessionState::from_json_str(r#"{"$schemaVersion":"nope"}"#).unwrap_err();
        assert_eq!(err.code, ErrorCode::SessionStateInvalid);
    }

    #[test]
    fn digest_covers_topology_but_not_contents() {
        let base = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));
        let same_topology = Manifest::new().with_entry("a.txt", Entry::text_file("goodbye"));
        assert_eq!(manifest_digest(&base), manifest_digest(&same_topology));

        let extra_entry = base.clone().with_entry("b.txt", Entry::text_file("x"));
        assert_ne!(manifest_digest(&base), manifest_digest(&extra_entry));

        // The root is backend-assigned, so it must not change the digest.
        let other_root = base.clone().with_root("/other");
        assert_eq!(manifest_digest(&base), manifest_digest(&other_root));

        let mounted = base
            .clone()
            .with_entry("m", Entry::mount(BindMountStrategy::mount("/host/a")));
        let mounted_elsewhere = base
            .clone()
            .with_entry("m", Entry::mount(BindMountStrategy::mount("/host/b")));
        // Host sources are not hashed, but the mount topology is.
        assert_eq!(
            manifest_digest(&mounted),
            manifest_digest(&mounted_elsewhere)
        );
        assert_ne!(manifest_digest(&base), manifest_digest(&mounted));

        let env_key = base.clone().with_env("TOKEN", "secret-1");
        let env_value = base.clone().with_env("TOKEN", "secret-2");
        assert_eq!(manifest_digest(&env_key), manifest_digest(&env_value));
        assert_ne!(manifest_digest(&base), manifest_digest(&env_key));
    }

    #[test]
    fn verifies_the_manifest_on_resume() {
        let manifest = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));
        let state = state_for(&manifest);
        state.verify_manifest(&manifest).unwrap();
        let tampered = manifest
            .clone()
            .with_entry("evil", Entry::local_dir("/etc"));
        let err = state.verify_manifest(&tampered).unwrap_err();
        assert_eq!(err.code, ErrorCode::SessionStateMismatch);
    }

    #[test]
    fn only_removes_sdk_created_workspace_roots() {
        let manifest = Manifest::new();
        let owned_root = std::env::temp_dir().join(format!("{OWNED_WORKSPACE_PREFIX}abc123"));
        let state = SandboxSessionState::new(
            "unix_local",
            "sid",
            owned_root.to_string_lossy().to_string(),
            &manifest,
        )
        .with_workspace_root_owned(true);
        assert!(state.may_remove_workspace_root());

        // A hand-edited state must not be able to point teardown at a real directory.
        let tampered = SandboxSessionState {
            workspace_root: "/Users/someone".to_string(),
            workspace_root_owned: true,
            ..state.clone()
        };
        assert!(!tampered.may_remove_workspace_root());

        let unowned = SandboxSessionState {
            workspace_root_owned: false,
            ..state
        };
        assert!(!unowned.may_remove_workspace_root());
    }

    #[test]
    fn state_json_never_contains_mount_credentials() {
        let config = MountPatternConfig::new().with_config_text("secret_key=SUPERSECRET");
        assert!(config.has_config_text());
        let manifest = Manifest::new()
            .with_entry(
                "m",
                Entry::mount(BindMountStrategy::mount("/host/secret-dir")),
            )
            .with_env("AWS_SECRET_ACCESS_KEY", "SUPERSECRET");
        let json = state_for(&manifest).to_json_string().unwrap();
        assert!(!json.contains("SUPERSECRET"), "{json}");
        assert!(!json.contains("secret-dir"), "{json}");
    }
}
