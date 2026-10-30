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
use crate::sandbox::mounts::MountCredentialAuthority;
use crate::sandbox::snapshot::SnapshotSpec;

/// The serde default for fields that older states did not carry.
fn default_true() -> bool {
    true
}

/// Prefix of the temporary workspace directories the SDK creates and may delete.
pub const OWNED_WORKSPACE_PREFIX: &str = "openai-agents-sandbox-";

/// Version of the session-state payload format.
pub const SESSION_STATE_SCHEMA_VERSION: &str = "openai-agents-sandbox-state-v1";

/// The current workspace fingerprint algorithm. Mirrors `SNAPSHOT_FINGERPRINT_VERSION`.
pub const SNAPSHOT_FINGERPRINT_VERSION: &str = "workspace_tar_sha256_v2";

/// The previous algorithm, recomputed as-is when comparing an older record.
pub const SNAPSHOT_FINGERPRINT_VERSION_V1: &str = "workspace_tar_sha256_v1";

/// A workspace fingerprint and the algorithm that produced it.
///
/// Mirrors the `{"fingerprint": ..., "version": ...}` record Python's workspace fingerprint
/// helper prints.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnapshotFingerprint {
    /// The hash of the workspace archive combined with the manifest digest.
    pub fingerprint: String,
    /// The algorithm identifier, such as `workspace_tar_sha256_v2`.
    pub version: String,
}

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
    /// The memory rollout this session's runs belong to.
    ///
    /// Carried across a suspend so the two halves of a paused run are consolidated into one
    /// memory artifact instead of two.
    #[serde(default)]
    pub rollout_id: Option<String>,
    /// Whether the workspace was fully materialized before the session was suspended.
    ///
    /// A run that was interrupted while the manifest was still being applied leaves a workspace
    /// that looks populated but is not, so a resume re-applies the manifest instead of trusting
    /// what is there.
    ///
    /// A state written before this field existed defaults to `true`, which keeps the behaviour
    /// those states were written under: defaulting to `false` would make a resume re-apply the
    /// manifest over the edits the first half of the run had already made.
    #[serde(default = "default_true")]
    pub workspace_root_ready: bool,
    /// Whether the host path grants have to be re-supplied by the caller on resume.
    ///
    /// Grants widen what the SDK may touch outside the workspace, so they are never carried in
    /// persisted state. Mirrors Python's `path_grants_require_rebind`.
    #[serde(default)]
    pub path_grants_require_rebind: bool,
    /// Whether mount credential authority was stripped when this state was written.
    ///
    /// Mirrors Python's `mount_authority_redacted`: it records that the live manifest held
    /// credential-bearing mounts, without recording the credentials.
    #[serde(default)]
    pub mount_authority_redacted: bool,
    /// Whether the caller has re-supplied that authority on the trusted manifest.
    ///
    /// Mirrors Python's `mount_authority_rebound`.
    #[serde(default)]
    pub mount_authority_rebound: bool,
    /// The grant paths whose `host_path` was dropped when this state was written.
    ///
    /// Mirrors Python's `REDACTED_HOST_PATH_GRANT_PATHS_KEY`. The paths themselves are inside
    /// the sandbox, so recording them leaks nothing about the host; the host side is exactly
    /// what [`rebind_path_grants`](Self::rebind_path_grants) makes the caller supply again.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redacted_host_path_grant_paths: Vec<String>,
    /// The fingerprint of the workspace the snapshot was taken from.
    ///
    /// Mirrors Python's `snapshot_fingerprint`: a resume whose live workspace still hashes to
    /// this value can skip restoring the archive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_fingerprint: Option<String>,
    /// The algorithm that produced [`snapshot_fingerprint`](Self::snapshot_fingerprint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_fingerprint_version: Option<String>,
    /// Backend-specific identity the backend needs to reattach, such as Docker's labels.
    ///
    /// Mirrors the extra fields Python's `SandboxSessionState` subclasses add. It never holds a
    /// credential: everything a mount needs comes from the trusted manifest at resume time.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub provider_metadata: serde_json::Map<String, serde_json::Value>,
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
            rollout_id: None,
            workspace_root_ready: false,
            // The live manifest is the only place a grant or a credential may live, so a
            // resumed state always needs the caller to supply them again.
            path_grants_require_rebind: !manifest.host_path_grants().is_empty()
                || !manifest.extra_path_grants().is_empty(),
            mount_authority_redacted: manifest.mount_targets().iter().any(|(mount, _)| {
                mount.strategy.credential_authority() != MountCredentialAuthority::None
            }),
            mount_authority_rebound: false,
            redacted_host_path_grant_paths: manifest
                .extra_path_grants()
                .iter()
                .filter(|grant| grant.host_path.is_some())
                .map(|grant| grant.path.clone())
                .collect(),
            snapshot_fingerprint: None,
            snapshot_fingerprint_version: None,
            provider_metadata: serde_json::Map::new(),
        }
    }

    /// Record the workspace fingerprint the snapshot was taken from.
    pub fn with_snapshot_fingerprint(mut self, record: Option<SnapshotFingerprint>) -> Self {
        match record {
            None => {
                self.snapshot_fingerprint = None;
                self.snapshot_fingerprint_version = None;
            }
            Some(record) => {
                self.snapshot_fingerprint = Some(record.fingerprint);
                self.snapshot_fingerprint_version = Some(record.version);
            }
        }
        self
    }

    /// Attach a backend-specific value under `key`.
    pub fn with_provider_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_metadata.insert(key.into(), value);
        self
    }

    /// Read a backend-specific value.
    pub fn provider_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.provider_metadata.get(key)
    }

    /// Rebind the persisted grants and mount authority to the current trusted manifest.
    ///
    /// Mirrors Python's `rebind_persisted_path_grants` + `rebind_persisted_mount_authority`. A
    /// persisted state never carries a host path or a mount credential, so every grant the live
    /// manifest had must be supplied again by application code or the resume is refused:
    /// otherwise a hand-edited state could widen what the sandbox reaches.
    pub fn rebind_path_grants(&self, trusted: &Manifest) -> SandboxResult<Self> {
        if !self.path_grants_require_rebind {
            let mut rebound = self.clone();
            if rebound.mount_authority_redacted {
                rebound.mount_authority_redacted = false;
                rebound.mount_authority_rebound = true;
            }
            return Ok(rebound);
        }
        let available: std::collections::BTreeSet<&str> = trusted
            .extra_path_grants()
            .iter()
            .filter(|grant| grant.host_path.is_some())
            .map(|grant| grant.path.as_str())
            .collect();
        let missing: Vec<&str> = self
            .redacted_host_path_grant_paths
            .iter()
            .map(String::as_str)
            .filter(|path| !available.contains(path))
            .collect();
        if !missing.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SessionStateInvalid,
                format!(
                    "sandbox session state requires current trusted host_path values for these \
                     path grants: {}",
                    missing.join(", ")
                ),
            ));
        }
        let mut rebound = self.clone();
        rebound.path_grants_require_rebind = false;
        rebound.redacted_host_path_grant_paths.clear();
        if rebound.mount_authority_redacted {
            // `verify_manifest` already proved the trusted manifest has the same mount topology,
            // so its strategies are the authority the state dropped.
            rebound.mount_authority_redacted = false;
            rebound.mount_authority_rebound = true;
        }
        Ok(rebound)
    }

    /// Refuse a resume that still needs the caller to supply grants or mount authority.
    ///
    /// Mirrors Python's `assert_path_grants_rebound`.
    pub fn assert_path_grants_rebound(&self) -> SandboxResult<()> {
        if self.mount_authority_redacted && !self.mount_authority_rebound {
            return Err(SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "sandbox session state with cloud mount credentials cannot be resumed on its \
                 own; resume through SandboxRunner with the current trusted manifest",
            ));
        }
        if self.path_grants_require_rebind {
            return Err(SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "sandbox session state path grants must be rebound from a current trusted \
                 manifest before resume; resume through SandboxRunner with \
                 SandboxRunConfig::manifest",
            ));
        }
        Ok(())
    }

    /// The state as it is safe to persist.
    ///
    /// Mirrors Python's `_sanitize_persisted_provider_identity`: once the live manifest held
    /// mount credentials, the backend resource those credentials attached storage to must not be
    /// reattachable from the state alone, so the provider identity is dropped and the session id
    /// is replaced by one derived from it. The derived id is stable, so two writes of the same
    /// state still name the same snapshot.
    pub fn sanitized_for_persistence(&self) -> Self {
        if !self.mount_authority_redacted || self.mount_authority_rebound {
            return self.clone();
        }
        let mut sanitized = self.clone();
        sanitized.provider_id = None;
        sanitized.session_id = uuid::Uuid::new_v5(
            &uuid::Uuid::NAMESPACE_URL,
            format!("openai-agents:mount-authority-redacted:{}", self.session_id).as_bytes(),
        )
        .to_string();
        sanitized.workspace_root_ready = false;
        sanitized
    }

    /// Record that the workspace was fully materialized.
    pub fn with_workspace_root_ready(mut self, ready: bool) -> Self {
        self.workspace_root_ready = ready;
        self
    }

    /// Record that the caller has re-supplied the mount authority on the trusted manifest.
    pub fn with_mount_authority_rebound(mut self, rebound: bool) -> Self {
        self.mount_authority_rebound = rebound;
        self
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

    /// Attach the memory rollout the session's runs belong to.
    pub fn with_rollout_id(mut self, rollout_id: Option<String>) -> Self {
        self.rollout_id = rollout_id;
        self
    }

    /// Serialize to a JSON string the caller can persist.
    ///
    /// The provider identity is sanitized first, so a state whose live manifest held mount
    /// credentials never persists a reattachable backend resource.
    pub fn to_json_string(&self) -> SandboxResult<String> {
        serde_json::to_string(&self.sanitized_for_persistence()).map_err(|e| {
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
            .with_rollout_id(Some("rollout-7".into()))
            .with_snapshot(SnapshotSpec::Local {
                base_path: "/tmp/snaps".into(),
            });
        let json = state.to_json_string().unwrap();
        assert_eq!(SandboxSessionState::from_json_str(&json).unwrap(), state);
        assert!(json.contains("$schemaVersion"));
        assert!(json.contains("rollout-7"));
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
