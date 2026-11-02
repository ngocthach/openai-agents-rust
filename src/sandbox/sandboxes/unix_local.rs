//! A sandbox session that uses a host directory as the workspace. Mirrors
//! `agents.sandbox.sandboxes.unix_local`.
//!
//! <div class="warning">
//!
//! Commands run on the host with the permissions of the calling process. This backend adds no
//! OS-level confinement: use it for trusted local development or inside externally provided
//! isolation. For untrusted commands, use [`DockerSandboxClient`] or a hosted backend.
//!
//! </div>
//!
//! [`DockerSandboxClient`]: crate::sandbox::sandboxes::docker::DockerSandboxClient

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use super::pinned_ops;
use crate::sandbox::archive::{
    TarValidationOptions, extract_tar_to, rebase_archive_symlink_targets, validate_tar_bytes,
};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, capture_workspace_tar, restore_session_snapshot,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{
    SandboxSessionState, SnapshotFingerprint, is_sdk_owned_workspace_root,
};
use crate::sandbox::types::{ExecResult, SandboxArchiveLimits};
use crate::sandbox::workspace_paths::SandboxPathGrant;

/// Prefix of the temporary workspace directories created for Unix-local sessions.
const WORKSPACE_PREFIX: &str = "openai-agents-sandbox-";

/// What a caller is told when a create failed after it had claimed the destination.
const INCOMPLETE_CREATE_MESSAGE: &str = "File creation failed after claiming the destination. The file may contain incomplete contents. Inspect the destination before using update_file or removing it to retry; retrying Add File without inspection is unsafe.";

/// Options for a host-shell sandbox, mirroring Python's `UnixLocalSandboxClientOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnixLocalSandboxClientOptions {
    /// Host ports the sandbox exposes, as [`UnixLocalSandboxClient::with_exposed_ports`].
    #[serde(default)]
    pub exposed_ports: Vec<u16>,
}

impl UnixLocalSandboxClientOptions {
    /// The client these options describe.
    pub fn client(&self) -> UnixLocalSandboxClient {
        UnixLocalSandboxClient::new().with_exposed_ports(self.exposed_ports.clone())
    }
}

/// Creates [`UnixLocalSandboxSession`]s backed by a temporary host directory.
///
/// <div class="warning">
///
/// Commands run on the host without OS-level confinement.
///
/// </div>
#[derive(Debug, Clone)]
pub struct UnixLocalSandboxClient {
    inherit_host_environment: bool,
    host_environment_allowlist: Option<Vec<String>>,
    snapshot: Option<SnapshotSpec>,
    exposed_ports: Vec<u16>,
}

impl Default for UnixLocalSandboxClient {
    fn default() -> Self {
        UnixLocalSandboxClient::new()
    }
}

impl UnixLocalSandboxClient {
    /// A client whose sessions inherit the host environment.
    pub fn new() -> Self {
        UnixLocalSandboxClient {
            inherit_host_environment: true,
            host_environment_allowlist: None,
            snapshot: None,
            exposed_ports: Vec::new(),
        }
    }

    /// Whether sandbox commands inherit the host process environment (default: true).
    pub fn with_inherit_host_environment(mut self, inherit: bool) -> Self {
        self.inherit_host_environment = inherit;
        self
    }

    /// Replace the host variables a non-inheriting session passes through.
    ///
    /// Mirrors Python's `UnixLocalSandboxClient(host_environment_allowlist=...)`, including its
    /// refusal to combine one with environment inheritance: an allowlist that narrows nothing is
    /// a configuration mistake worth reporting rather than silently ignoring. A Rust builder
    /// cannot fail, so the refusal lands on [`create`](SandboxClient::create) and
    /// [`resume`](SandboxClient::resume) instead of on this method.
    pub fn with_host_environment_allowlist<I, S>(mut self, allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.host_environment_allowlist = Some(allowlist.into_iter().map(Into::into).collect());
        self
    }

    /// Build a session, storing its snapshots at the client's own spec or, failing that, at
    /// `default_snapshot`.
    async fn create_session(
        &self,
        manifest: Option<Manifest>,
        default_snapshot: Option<SnapshotSpec>,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        let manifest = manifest.unwrap_or_default();
        self.validate_configuration(&manifest)?;
        // A runner-created local session always gets a dedicated temp root unless the caller
        // chose a custom workspace root explicitly.
        let (manifest, owned_root) = if manifest.root() == DEFAULT_MANIFEST_ROOT {
            let dir = create_temp_workspace().await?;
            (manifest.with_root(path_to_posix(&dir)), true)
        } else {
            (manifest, false)
        };
        let session = self
            .apply_environment_policy(UnixLocalSandboxSession::new(manifest, owned_root)?)
            .with_exposed_ports(self.exposed_ports.clone())
            .with_snapshot(default_snapshot.or_else(|| self.snapshot.clone()));
        Ok(Arc::new(session))
    }

    /// Refuse a client whose allowlist can never apply, and a grant this backend cannot honour.
    ///
    /// Both checks run before the workspace directory is created, so a misconfigured client
    /// leaves nothing behind on the host.
    fn validate_configuration(&self, manifest: &Manifest) -> SandboxResult<()> {
        if self.host_environment_allowlist.is_some() && self.inherit_host_environment {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "host_environment_allowlist requires inherit_host_environment=False",
            ));
        }
        // The workspace is a host directory already: there is nothing to bind-mount a separate
        // host path onto, so honouring `host_path` would silently hand back the sandbox path.
        for grant in manifest.extra_path_grants() {
            if grant.host_path.is_some() {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!(
                        "UnixLocalSandboxClient does not support sandbox path grant host_path \
                         for '{}'; omit host_path when both paths are the same or use \
                         DockerSandboxClient",
                        grant.path
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Apply this client's environment policy to a session it just built.
    fn apply_environment_policy(
        &self,
        session: UnixLocalSandboxSession,
    ) -> UnixLocalSandboxSession {
        let session = session.with_inherit_host_environment(self.inherit_host_environment);
        match &self.host_environment_allowlist {
            None => session,
            Some(allowlist) => session.with_host_environment_allowlist(allowlist.clone()),
        }
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Declare the sandbox-side ports host code may reach through
    /// [`SandboxSession::resolve_exposed_port`].
    ///
    /// Mirrors Python's `UnixLocalSandboxClientOptions.exposed_ports`. Commands run on the host
    /// here, so there is no port mapping to perform: declaring a port only says which ones the
    /// session will hand back a loopback endpoint for. Duplicates are dropped, keeping the first
    /// occurrence, so the session's port list is the set that was asked for.
    pub fn with_exposed_ports<I>(mut self, ports: I) -> Self
    where
        I: IntoIterator<Item = u16>,
    {
        self.exposed_ports = dedupe_ports(ports);
        self
    }
}

/// Drop repeated ports, keeping the order they were declared in.
fn dedupe_ports<I: IntoIterator<Item = u16>>(ports: I) -> Vec<u16> {
    let mut seen = Vec::new();
    for port in ports {
        if !seen.contains(&port) {
            seen.push(port);
        }
    }
    seen
}

#[async_trait]
impl SandboxClient for UnixLocalSandboxClient {
    fn backend_id(&self) -> &str {
        "unix_local"
    }

    fn configured_snapshot(&self) -> Option<SnapshotSpec> {
        self.snapshot.clone()
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.create_session(manifest, None).await
    }

    /// Create a session that stores its snapshots at `snapshot`.
    ///
    /// Mirrors Python's `UnixLocalSandboxClient.create(snapshot=...)`: the argument is the snapshot
    /// the session uses. The runner has already preferred a snapshot this client was configured
    /// with over the SDK default (see [`configured_snapshot`](SandboxClient::configured_snapshot)),
    /// so what arrives here is the one to honor.
    async fn create_with_snapshot(
        &self,
        manifest: Option<Manifest>,
        snapshot: SnapshotSpec,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.create_session(manifest, Some(snapshot)).await
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        if state.backend != "unix_local" {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "sandbox session state was created by the `{}` backend",
                    state.backend
                ),
            ));
        }
        self.validate_configuration(&manifest)?;
        state.verify_manifest(&manifest)?;
        // The trusted manifest supplies the host paths the persisted state deliberately dropped.
        let state = state.rebind_path_grants(&manifest)?;
        // The state carries no manifest contents of its own: the workspace root comes from it,
        // but everything that could read host paths comes from the caller's trusted manifest.
        let manifest = manifest.with_root(state.workspace_root.clone());
        let session = self
            .apply_environment_policy(UnixLocalSandboxSession::new(
                manifest,
                state.may_remove_workspace_root(),
            )?)
            .with_exposed_ports(state.exposed_ports.clone())
            .with_snapshot(Some(state.snapshot.clone()))
            .with_identity(state.session_id.clone(), state.snapshot_id.clone())
            .resuming()
            .with_workspace_root_ready(state.workspace_root_ready)
            .with_snapshot_fingerprint(
                state.snapshot_fingerprint.clone(),
                state.snapshot_fingerprint_version.clone(),
            );
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        // Python's runtime stops a session and then shuts it down, each as its own traced
        // operation; a client that is asked to release a session performs both, so the spans
        // appear whichever caller does the release.
        session.stop().await?;
        session.shutdown().await
    }
}

async fn create_temp_workspace() -> SandboxResult<PathBuf> {
    let base = std::env::temp_dir();
    for _ in 0..16 {
        let candidate = base.join(format!(
            "{WORKSPACE_PREFIX}{}",
            uuid::Uuid::new_v4().simple()
        ));
        match tokio::fs::create_dir(&candidate).await {
            Ok(()) => {
                // macOS resolves `/var` to `/private/var`; canonicalize once so every later
                // prefix check compares real paths.
                return tokio::fs::canonicalize(&candidate).await.map_err(|e| {
                    SandboxError::new(
                        ErrorCode::WorkspaceStartError,
                        format!("Failed to resolve workspace `{}`: {e}", candidate.display()),
                    )
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("Failed to create a sandbox workspace directory: {e}"),
                ));
            }
        }
    }
    Err(SandboxError::new(
        ErrorCode::WorkspaceStartError,
        "Failed to create a unique sandbox workspace directory",
    ))
}

fn path_to_posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// A sandbox session whose workspace is a directory on the host filesystem.
///
/// <div class="warning">
///
/// Commands run on the host without OS-level confinement.
///
/// </div>
#[derive(Debug)]
pub struct UnixLocalSandboxSession {
    id: String,
    manifest: Manifest,
    /// Path grants a capability changed after the session started.
    live_grants: crate::sandbox::session::LiveGrants,
    root: PathBuf,
    owned_root: bool,
    inherit_host_environment: bool,
    host_environment_allowlist: Option<Vec<String>>,
    /// The real directory the workspace root named when the session was built, or when it first
    /// came into existence. Replacing the root with a symlink afterwards must not re-authorize
    /// whatever it now points at, so every later check compares against this.
    pinned_root: std::sync::OnceLock<PathBuf>,
    /// The real directories this manifest's extra path grants named when the session was built,
    /// with each grant's write policy. Re-pointing a grant alias afterwards widens nothing.
    pinned_grants: std::sync::RwLock<Vec<PinnedGrant>>,
    /// The manifest's start-up environment lookups, performed once when the session starts.
    resolved_environment: std::sync::OnceLock<std::collections::BTreeMap<String, String>>,
    /// Whether the manifest finished materializing, so a resume can trust the workspace.
    materialized: AtomicBool,
    /// Keeps concurrent work out of the workspace while a snapshot detaches its mounts.
    mount_transition: crate::sandbox::mounts::MountTransition,
    running: AtomicBool,
    /// Runs between a path being validated and the file being opened. Production code never
    /// sets it; a test uses it to swap a directory for a symlink inside that window.
    after_validate: Option<AfterValidateHook>,
    after_pin: Option<AfterValidateHook>,
    /// Whether the mounts were already detached by `stop`, so `shutdown` does not detach them a
    /// second time. Cleared whenever the session starts.
    mounts_detached: AtomicBool,
    ptys: PtyManager,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
    /// The workspace fingerprint of the snapshot this session resumes, when it has one.
    resume_fingerprint: std::sync::Mutex<Option<SnapshotFingerprint>>,
    /// The archive resource limits the runner resolved for this session.
    archive_limits: std::sync::Mutex<Option<SandboxArchiveLimits>>,
    /// The sandbox-side ports this session hands out loopback endpoints for.
    exposed_ports: Vec<u16>,
    /// Workspace-relative paths a workspace archive leaves out.
    persist_skip: std::sync::Mutex<std::collections::BTreeSet<String>>,
}

impl UnixLocalSandboxSession {
    /// The real directory and write policy of every extra path grant in force.
    fn pinned_grants(&self) -> Vec<(PathBuf, bool)> {
        self.pinned_grants
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(_, real, read_only)| (real.clone(), *read_only))
            .collect()
    }

    /// Create a session over an existing or to-be-created host directory.
    ///
    /// `owned_root` marks the directory for removal when the session shuts down.
    pub fn new(manifest: Manifest, owned_root: bool) -> SandboxResult<Self> {
        let root = PathBuf::from(manifest.root());
        if !root.is_absolute() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "Unix-local workspace root `{}` must be absolute",
                    manifest.root()
                ),
            ));
        }
        let id = uuid::Uuid::new_v4().to_string();
        let pinned_root = std::sync::OnceLock::new();
        // A root that does not exist yet is pinned by `start`, which creates it.
        if let Ok(canonical) = root.canonicalize() {
            let _ = pinned_root.set(canonical);
        }
        let pinned_grants = pin_grants(manifest.extra_path_grants(), &[]);
        Ok(UnixLocalSandboxSession {
            snapshot_id: id.clone(),
            id,
            manifest,
            root,
            owned_root,
            inherit_host_environment: true,
            host_environment_allowlist: None,
            pinned_root,
            pinned_grants: std::sync::RwLock::new(pinned_grants),
            live_grants: Default::default(),
            resolved_environment: std::sync::OnceLock::new(),
            materialized: AtomicBool::new(false),
            mount_transition: crate::sandbox::mounts::MountTransition::new(),
            running: AtomicBool::new(false),
            after_validate: None,
            after_pin: None,
            mounts_detached: AtomicBool::new(false),
            ptys: PtyManager::new(),
            snapshot: None,
            resuming: false,
            resume_fingerprint: std::sync::Mutex::new(None),
            archive_limits: std::sync::Mutex::new(None),
            exposed_ports: Vec::new(),
            persist_skip: std::sync::Mutex::new(std::collections::BTreeSet::new()),
        })
    }

    /// Carry the workspace fingerprint a resumed snapshot was taken from.
    pub fn with_snapshot_fingerprint(
        self,
        fingerprint: Option<String>,
        version: Option<String>,
    ) -> Self {
        if let (Some(fingerprint), Some(version)) = (fingerprint, version) {
            *self.resume_fingerprint.lock().expect("fingerprint lock") =
                Some(SnapshotFingerprint {
                    fingerprint,
                    version,
                });
        }
        self
    }

    /// Remove the workspace root as each user the manifest declares, then as the session owner.
    ///
    /// Mirrors what Python's `DockerRemovalService` achieves, without the privileged worker: the
    /// trusted manifest already names every account the sandbox ran commands as, so the files
    /// those accounts own can be unlinked by them directly.
    async fn remove_workspace_as_manifest_users(&self) -> SandboxResult<()> {
        let root = self.root.to_string_lossy().to_string();
        let users: Vec<crate::sandbox::types::User> = self.manifest.users().to_vec();
        for user in users {
            let _ = self
                .exec(
                    ExecRequest::argv(["rm", "-rf", "--", root.as_str()])
                        .with_user(Some(user.clone())),
                )
                .await;
        }
        match tokio::fs::remove_dir_all(&self.root).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                format!(
                    "Failed to remove workspace `{}` as the session owner or as any manifest \
                     user: {e}",
                    self.root.display()
                ),
            )),
        }
    }

    /// Store workspace snapshots at `snapshot`.
    pub fn with_snapshot(mut self, snapshot: Option<SnapshotSpec>) -> Self {
        self.snapshot = snapshot;
        self
    }

    /// Declare the sandbox-side ports this session resolves to a loopback endpoint.
    ///
    /// Duplicates are dropped, keeping the order they were declared in.
    pub fn with_exposed_ports<I>(mut self, ports: I) -> Self
    where
        I: IntoIterator<Item = u16>,
    {
        self.exposed_ports = dedupe_ports(ports);
        self
    }

    /// Reuse a persisted session and snapshot identity.
    pub fn with_identity(mut self, session_id: String, snapshot_id: String) -> Self {
        self.id = session_id;
        self.snapshot_id = snapshot_id;
        self
    }

    /// Mark the session as resuming, so `start` restores the snapshot instead of materializing
    /// the full manifest.
    pub fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// Whether the workspace this session resumes was fully materialized before the pause.
    ///
    /// A run interrupted mid-materialization leaves a workspace that has files in it but is not
    /// the one the manifest describes, so resuming it has to re-apply the manifest rather than
    /// trust what survived.
    pub fn with_workspace_root_ready(self, ready: bool) -> Self {
        self.materialized.store(ready, Ordering::SeqCst);
        self
    }

    /// Whether sandbox commands inherit the host process environment.
    pub fn with_inherit_host_environment(mut self, inherit: bool) -> Self {
        self.inherit_host_environment = inherit;
        self
    }

    /// Replace the host variables a non-inheriting session passes through.
    ///
    /// Mirrors Python's `host_environment_allowlist`; `None` uses
    /// [`HOST_ENVIRONMENT_ALLOWLIST`]. It has no effect while the session inherits the host
    /// environment, which already passes everything.
    pub fn with_host_environment_allowlist<I, S>(mut self, allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.host_environment_allowlist = Some(allowlist.into_iter().map(Into::into).collect());
        self
    }

    /// The host directory backing the workspace.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a model-provided path to a host path inside the workspace root, rejecting
    /// absolute paths outside the root, `..` segments and symlinks that escape the workspace.
    pub async fn host_path(&self, path: &str) -> SandboxResult<PathBuf> {
        self.host_path_for(path, false).await
    }

    /// The workspace roots an archived symlink target may be recorded against.
    ///
    /// Mirrors Python's `roots = (root, root.resolve(strict=False))`: a link created through the
    /// canonical path records `/private/var/...` on macOS where the workspace root is `/var/...`,
    /// and both spellings name the same directory.
    async fn rebase_roots(&self) -> Vec<String> {
        let mut roots = vec![self.root.to_string_lossy().into_owned()];
        if let Ok(canonical) = tokio::fs::canonicalize(&self.root).await {
            let canonical = canonical.to_string_lossy().into_owned();
            if !roots.contains(&canonical) {
                roots.push(canonical);
            }
        }
        roots
    }

    /// Refuse a recursive removal whose *resolved* tree contains a read-only grant.
    ///
    /// The path policy checks the spelling, which cannot see that `link/tree` in the workspace is
    /// really `external/tree`, so a link standing in for an ancestor of a protected grant would
    /// pass it. The check runs again on the real location, with each grant resolved as it was
    /// when the session was built, and the grant's own effective access decides (a writable
    /// grant nested in a read-only one stays removable).
    async fn refuse_removal_over_read_only_grants(
        &self,
        host: &Path,
        path: &str,
    ) -> SandboxResult<()> {
        let real = resolve_existing_prefix(host).await;
        let pinned = self.pinned_grants();
        for (grant_root, read_only) in &pinned {
            // Inside the workspace the root always wins over a grant that also covers it.
            let in_workspace = self
                .pinned_root
                .get()
                .is_some_and(|root| grant_root.starts_with(root));
            if !*read_only || in_workspace || !grant_root.starts_with(&real) {
                continue;
            }
            let effective_read_only = pinned
                .iter()
                .filter(|(root, _)| grant_root.starts_with(root))
                .fold(None::<(&PathBuf, bool)>, |best, (root, ro)| match best {
                    Some((current, _))
                        if root.components().count() <= current.components().count() =>
                    {
                        best
                    }
                    _ => Some((root, *ro)),
                })
                .is_some_and(|(_, ro)| ro);
            if effective_read_only {
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceArchiveWriteError,
                    format!("failed to write archive for path: {path}"),
                )
                .with_context("path", path.to_string())
                .with_context("reason", "read_only_extra_path_grant")
                .with_context("grant_path", grant_root.to_string_lossy().into_owned()));
            }
        }
        Ok(())
    }

    /// Read a regular workspace file, all of it or the first `limit` bytes.
    async fn read_workspace_file(&self, path: &str, limit: Option<u64>) -> SandboxResult<Vec<u8>> {
        let host = self.host_path_for(path, false).await?;
        // A FIFO with no peer blocks forever on open, and a device node would hand the model
        // whatever the kernel produces. Python's file ops `fstat` the descriptor and refuse
        // anything that is not a regular file; checking the entry first does the same without
        // ever opening it.
        reject_special_file(&host, path, false).await?;
        self.run_after_validate(&host);
        let read_error = |e: std::io::Error| {
            let code = if e.kind() == std::io::ErrorKind::NotFound {
                ErrorCode::WorkspaceReadNotFound
            } else {
                ErrorCode::WorkspaceArchiveReadError
            };
            SandboxError::new(code, format!("Failed to read `{path}`: {e}"))
        };
        use tokio::io::AsyncReadExt as _;
        // Non-blocking, so a FIFO swapped in after the entry check is opened (and then refused
        // as not a regular file) instead of blocking forever waiting for a writer.
        let mut file = tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK)
            .open(&host)
            .await
            .map_err(read_error)?;
        let handle = file.metadata().await.map_err(read_error)?;
        if !handle.is_file() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceArchiveReadError,
                format!("`{path}` is not a regular file"),
            ));
        }
        self.verify_opened(&handle, &host, path, false).await?;
        let mut contents = Vec::new();
        match limit {
            Some(limit) => file.take(limit).read_to_end(&mut contents).await,
            None => file.read_to_end(&mut contents).await,
        }
        .map_err(read_error)?;
        Ok(contents)
    }

    /// Resolve a model-provided path for reading or writing.
    ///
    /// Writes through a read-only mount are rejected.
    pub async fn host_path_for(&self, path: &str, for_write: bool) -> SandboxResult<PathBuf> {
        let resolved = if for_write {
            self.normalize_path_for_write(path)?
        } else {
            self.normalize_path(path)?
        };
        let candidate = PathBuf::from(&resolved);
        let real = resolve_existing_prefix(&candidate).await;
        self.authorize_real(candidate, real, path, for_write).await
    }

    /// Decide whether the already-resolved host path `real` may be reached, and return the path
    /// to use. `candidate` is the lexical spelling, `path` the caller's own.
    async fn authorize_real(
        &self,
        candidate: PathBuf,
        real: PathBuf,
        path: &str,
        for_write: bool,
    ) -> SandboxResult<PathBuf> {
        // The root is canonicalized too: a caller-supplied root can contain symlinks (on macOS
        // `/var/folders/...` resolves to `/private/var/folders/...`), and comparing a resolved
        // candidate against an unresolved root would reject every path inside the workspace.
        // The pinned value is what counts: replacing the root directory with a symlink after the
        // session was built must not turn the link's target into the workspace.
        let real_root = match self.pinned_root.get() {
            None => resolve_existing_prefix(&self.root).await,
            Some(pinned) => {
                match tokio::fs::canonicalize(&self.root).await {
                    Ok(current) if current != *pinned => {
                        return Err(SandboxError::new(
                            ErrorCode::InvalidManifestPath,
                            format!(
                                "sandbox workspace root `{}` no longer names the directory the \
                                 session was created over",
                                self.root.display()
                            ),
                        ));
                    }
                    // A root that is simply gone authorizes nothing, and the pinned value makes
                    // every path under it fail the prefix check below.
                    _ => pinned.clone(),
                }
            }
        };
        if real.starts_with(&real_root) {
            return Ok(candidate);
        }
        // An extra path grant deliberately points outside the workspace. It is matched against
        // the directory it named when the session was built, so re-pointing the alias it was
        // declared through cannot widen what the sandbox reaches. `normalize_path_for_write`
        // has already refused a write through a read-only grant.
        // The most specific grant decides, and the first declared wins a tie, as in Python: a
        // writable grant nested in a read-only one has to stay writable.
        let pinned = self.pinned_grants();
        let mut effective: Option<(&PathBuf, bool)> = None;
        for (grant_root, read_only) in &pinned {
            if real.starts_with(grant_root)
                && effective.is_none_or(|(best, _)| {
                    grant_root.components().count() > best.components().count()
                })
            {
                effective = Some((grant_root, *read_only));
            }
        }
        if let Some((_, read_only)) = effective {
            {
                if read_only && for_write {
                    return Err(SandboxError::new(
                        ErrorCode::WorkspaceArchiveWriteError,
                        format!("failed to write archive for path: {path}"),
                    )
                    .with_context("path", path.to_string())
                    .with_context("reason", "read_only_extra_path_grant"));
                }
                return Ok(real);
            }
        }
        // A bind mount deliberately points outside the workspace, so its source is an explicit
        // grant. Writes through a read-only grant are still refused.
        // Only the mounts are consulted here: the manifest's extra path grants were pinned when
        // the session was built, and re-resolving them now would follow a re-pointed alias.
        let mount_grants: Vec<_> = self
            .manifest
            .mount_targets()
            .into_iter()
            .filter_map(|(mount, _)| mount.strategy.host_path_grant(mount))
            .collect();
        for grant in mount_grants {
            let real_grant = resolve_existing_prefix(&grant.path).await;
            if real.starts_with(&real_grant) {
                if grant.read_only && for_write {
                    return Err(SandboxError::new(
                        ErrorCode::WorkspaceWriteTypeError,
                        format!("sandbox path `{path}` is inside a read-only mount"),
                    ));
                }
                return Ok(real);
            }
        }
        Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("sandbox path `{path}` escapes the workspace root"),
        ))
    }

    /// Run `hook` with the validated host path just before a file is opened.
    ///
    /// This exists so a test can swap a directory for a symlink between validation and use,
    /// the window the open-then-verify check below closes. It has no production use.
    #[doc(hidden)]
    pub fn with_after_validate_hook(mut self, hook: Arc<dyn Fn(&Path) + Send + Sync>) -> Self {
        self.after_validate = Some(AfterValidateHook(hook));
        self
    }

    /// Run `hook` once an operation has pinned the directory that holds its entry and before it
    /// acts on the entry. A test swaps the parent or the leaf here to show the operation is bound
    /// to the descriptor it already holds. It has no production use.
    #[doc(hidden)]
    pub fn with_after_pin_hook(mut self, hook: Arc<dyn Fn(&Path) + Send + Sync>) -> Self {
        self.after_pin = Some(AfterValidateHook(hook));
        self
    }

    fn run_after_validate(&self, host: &Path) {
        if let Some(hook) = &self.after_validate {
            (hook.0)(host);
        }
    }

    /// Open the directory that holds `host` and keep it open, so the operation that follows acts
    /// on the entry relative to that descriptor.
    ///
    /// The parent is resolved *now*, after validation and after any window in which a workspace
    /// process could have swapped it, and the resolution must still be authorized. It is then
    /// opened component by component without following a link, so a component replaced between
    /// the resolution and the open is refused, and one replaced afterwards cannot matter. The
    /// leaf is never resolved: the operation acts on the entry itself.
    async fn pin_parent(
        &self,
        host: &Path,
        path: &str,
        for_write: bool,
    ) -> SandboxResult<(std::os::fd::OwnedFd, std::ffi::OsString)> {
        let changed = || {
            SandboxError::new(
                ErrorCode::InvalidManifestPath,
                format!("`{path}` changed while it was being validated"),
            )
            .with_context("path", path.to_string())
        };
        let (Some(parent), Some(name)) = (host.parent(), host.file_name()) else {
            return Err(changed());
        };
        let real_parent = tokio::fs::canonicalize(parent)
            .await
            .map_err(|_| changed())?;
        self.authorize_real(host.to_path_buf(), real_parent.join(name), path, for_write)
            .await
            .map_err(|_| changed())?;
        let pinned = real_parent.clone();
        let fd = tokio::task::spawn_blocking(move || pinned_ops::open_dir_pinned(&pinned))
            .await
            .map_err(|_| changed())?
            .map_err(|_| changed())?;
        if let Some(hook) = &self.after_pin {
            (hook.0)(host);
        }
        Ok((fd, name.to_os_string()))
    }

    /// Confirm that an opened handle is still the workspace file `host` names.
    ///
    /// Validation canonicalizes the path, but a workspace process can replace a directory with a
    /// symlink before the open that follows, and the open then follows it out of the workspace.
    /// This resolves the path again *after* the open and requires that (a) the resolution is
    /// still inside the workspace or a grant, and (b) it names the same device and inode as the
    /// handle. A handle that landed outside therefore fails (b) or (a), and for a write this runs
    /// before anything is truncated or written.
    ///
    /// Residual window, which a descriptor-relative walk (`openat` with `O_NOFOLLOW`, as Python's
    /// `_FileOps`) would close but this crate cannot do under `#![forbid(unsafe_code)]`: the
    /// check proves the handle was inside at the instant of the check, so a swap that is undone
    /// before the check (A to B to A) is not seen, and a hardlink to an outside inode planted
    /// inside the workspace is indistinguishable from a legitimate file. Neither lets the swap
    /// attacker choose the outside target without already controlling a hardlink or a timed
    /// double swap.
    async fn verify_opened(
        &self,
        handle: &std::fs::Metadata,
        host: &Path,
        path: &str,
        for_write: bool,
    ) -> SandboxResult<()> {
        let changed = || {
            SandboxError::new(
                ErrorCode::InvalidManifestPath,
                format!("sandbox path `{path}` changed after it was validated"),
            )
        };
        let real = tokio::fs::canonicalize(host).await.map_err(|_| changed())?;
        let current = tokio::fs::metadata(&real).await.map_err(|_| changed())?;
        if file_identity(&current) != file_identity(handle) {
            return Err(changed());
        }
        self.authorize_real(host.to_path_buf(), real, path, for_write)
            .await
            .map_err(|_| changed())?;
        Ok(())
    }

    /// Unlink a file this call created, once its handle failed verification.
    ///
    /// Only the inode the handle names is removed: if the path now resolves to anything else, a
    /// concurrent writer owns it and it is left alone.
    async fn discard_created(&self, handle: &std::fs::Metadata, host: &Path) {
        if let Ok(real) = tokio::fs::canonicalize(host).await
            && let Ok(current) = tokio::fs::metadata(&real).await
            && file_identity(&current) == file_identity(handle)
        {
            let _ = tokio::fs::remove_file(&real).await;
        }
    }

    /// The attributes a `sandbox.<op>` span carries. Mirrors Python's `_trace_span_data`.
    fn span_data(&self, op: &str) -> serde_json::Value {
        serde_json::json!({
            "sandbox.backend": "unix_local",
            "sandbox.operation": op,
            "sandbox.session.id": self.id,
            "session_id": self.id,
        })
    }

    /// The shutdown itself: the `sandbox.shutdown` span wraps this and nothing else.
    async fn shutdown_untraced(&self) -> SandboxResult<()> {
        self.ptys.terminate_all();
        self.running.store(false, Ordering::SeqCst);
        // Detach every mount before touching the workspace: an in-container mount materializes a
        // real filesystem inside the root, and removing the root over a live mount would delete
        // the mounted storage, not just the workspace.
        if !self.mounts_detached.load(Ordering::SeqCst) {
            self.manifest.unmount_all(self).await?;
            // Only after it worked: a failed detach must be retried by the next shutdown.
            self.mounts_detached.store(true, Ordering::SeqCst);
        }
        if self.owned_root && is_sdk_owned_workspace_root(&self.root) {
            match tokio::fs::remove_dir_all(&self.root).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    // A `run_as` user's commands create files the session owner cannot unlink —
                    // a `0700` directory made by `sudo -u agent` is enough. This is the case
                    // Python solves with its privileged removal worker; here the manifest names
                    // the users, so the workspace can simply be unlinked as each of them.
                    self.remove_workspace_as_manifest_users().await?;
                }
                Err(e) => {
                    return Err(SandboxError::new(
                        ErrorCode::WorkspaceStopError,
                        format!("Failed to remove workspace `{}`: {e}", self.root.display()),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The `PATH` a command started with `request_env` will see.
    fn child_path(
        &self,
        request_env: &std::collections::BTreeMap<String, String>,
    ) -> Option<String> {
        request_env
            .get("PATH")
            .cloned()
            .or_else(|| {
                self.exec_env()
                    .into_iter()
                    .rev()
                    .find(|(key, _)| key == "PATH")
                    .map(|(_, value)| value)
            })
            .or_else(|| {
                self.inherit_host_environment
                    .then(|| std::env::var("PATH").ok())
                    .flatten()
            })
    }

    /// Wrap `argv` in `sandbox-exec` when running on macOS, and return the directory to start it
    /// in. Elsewhere the command is returned as it was given.
    ///
    /// Mirrors the preparation in Python's `_exec_internal`: absolute arguments under the
    /// workspace are rewritten relative to it, a `sh -c` command is anchored at the workspace
    /// with an explicit `cd`, and the result is wrapped by
    /// [`confinement::confined_exec_command`].
    fn confine_command(
        &self,
        argv: Vec<String>,
        cwd: PathBuf,
        child_path: Option<String>,
    ) -> SandboxResult<(Vec<String>, PathBuf)> {
        if !cfg!(target_os = "macos") {
            return Ok((argv, cwd));
        }
        let workspace_root = confinement::resolve_lenient(&self.root);
        let argv = workspace_relative_command_parts(argv, &workspace_root);
        let (process_cwd, argv) = shell_workspace_process_context(argv, &workspace_root, &cwd);
        let host_home = confinement::resolve_lenient(Path::new(
            &std::env::var("HOME").unwrap_or_else(|_| "/".to_string()),
        ));
        let host_path = std::env::var("PATH").ok();
        let which = |name: &str, path: Option<&str>| confinement::which_in(name, path);
        let wrapped = self.confined_exec_command(
            &argv,
            &workspace_root,
            child_path.as_deref(),
            &confinement::HostFacts {
                is_darwin: true,
                platform: "darwin",
                host_path: host_path.as_deref(),
                host_home: &host_home,
                which: &which,
            },
        )?;
        Ok((wrapped, process_cwd))
    }

    /// The `sandbox-exec` invocation this session would run `command_parts` under.
    ///
    /// Mirrors Python's `_confined_exec_command`. `child_path` is the `PATH` the command will
    /// see; `host` carries the facts about the machine, so a configuration can be checked without
    /// being on it. Off macOS the command comes back unchanged.
    pub fn confined_exec_command(
        &self,
        command_parts: &[String],
        workspace_root: &Path,
        child_path: Option<&str>,
        host: &confinement::HostFacts<'_>,
    ) -> SandboxResult<Vec<String>> {
        // A host entry may widen a `bin` grant to its virtual environment root only when the
        // client lets the host `PATH` through to the child at all.
        let host_path_inherited = self
            .host_environment_allowlist
            .as_ref()
            .is_none_or(|allowlist| allowlist.iter().any(|name| name == "PATH"));
        let grants = self.extra_path_grants();
        confinement::confined_exec_command(&confinement::ConfinementInputs {
            is_darwin: host.is_darwin,
            platform: host.platform,
            command_parts,
            workspace_root,
            child_path,
            host_path: host.host_path,
            host_path_inherited,
            host_home: host.host_home,
            extra_path_grants: &grants,
            which: host.which,
        })
    }

    /// Follow symlink components that dangle, so a create lands where the link points.
    ///
    /// `canonicalize` cannot resolve a link whose target does not exist yet, so the existing
    /// prefix stops short of it and a plain `create_dir_all` then fails with "File exists" on
    /// the link itself. Python resolves such safe internal aliases. A resolved target is
    /// authorized again, so a dangling link aimed outside the workspace is refused rather than
    /// followed.
    async fn resolve_for_create(
        &self,
        host: PathBuf,
        path: &str,
        for_write: bool,
    ) -> SandboxResult<PathBuf> {
        let resolved = resolve_dangling_components(&host).await;
        if resolved == host {
            return Ok(host);
        }
        let real = resolve_existing_prefix(&resolved).await;
        self.authorize_real(resolved.clone(), real, path, for_write)
            .await?;
        Ok(resolved)
    }

    /// Normalize `path`, also accepting an absolute path spelled through the resolved workspace
    /// root.
    ///
    /// Mirrors Python's `_resolved_host_path_and_grant`, which compares against
    /// `root.resolve()` as well as `root`: a root reached through a symlink (or `/var` on macOS)
    /// names the same directory as its canonical spelling, and a caller that resolved a path
    /// itself must not be refused for it.
    fn normalize_spelled(&self, path: &str, for_write: bool) -> SandboxResult<String> {
        let policy = self.workspace_path_policy()?;
        match policy.normalize_sandbox_path(path, for_write) {
            Err(error) if error.code == ErrorCode::InvalidManifestPath => {
                if let Some(pinned) = self.pinned_root.get()
                    && let Ok(rest) = Path::new(path).strip_prefix(pinned)
                {
                    let root = self.root.to_string_lossy();
                    let rebased = if rest.as_os_str().is_empty() {
                        root.to_string()
                    } else {
                        format!("{}/{}", root.trim_end_matches('/'), rest.display())
                    };
                    return policy.normalize_sandbox_path(&rebased, for_write);
                }
                Err(error)
            }
            other => other,
        }
    }

    /// Whether the workspace root still holds anything, used to decide how much a resumed
    /// session has to re-materialize.
    async fn workspace_has_content(&self) -> bool {
        match tokio::fs::read_dir(&self.root).await {
            Err(_) => false,
            Ok(mut entries) => matches!(entries.next_entry().await, Ok(Some(_))),
        }
    }

    /// The environment a sandbox command starts with.
    ///
    /// Mirrors Python's `_exec_environment`: the host variables the session is allowed to pass
    /// through, then the manifest's own environment, then `HOME` pinned to the workspace root so
    /// a tool writing to `~` writes inside the sandbox.
    fn exec_env(&self) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = Vec::new();
        if !self.inherit_host_environment {
            for key in self.host_environment_allowlist() {
                if let Ok(value) = std::env::var(key) {
                    env.push((key.to_string(), value));
                }
            }
        }
        match self.resolved_environment.get() {
            // Once the session has started this already includes the manifest's literals.
            Some(resolved) => {
                env.extend(resolved.iter().map(|(k, v)| (k.clone(), v.clone())));
            }
            None => env.extend(
                self.manifest()
                    .environment()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            ),
        }
        env.push(("HOME".to_string(), self.root.to_string_lossy().into_owned()));
        env
    }

    /// The host variables this session passes through when it does not inherit the environment.
    fn host_environment_allowlist(&self) -> Vec<&str> {
        match &self.host_environment_allowlist {
            Some(allowlist) => allowlist.iter().map(String::as_str).collect(),
            None => HOST_ENVIRONMENT_ALLOWLIST.to_vec(),
        }
    }
}

/// macOS filesystem confinement for Unix-local commands.
///
/// Mirrors the `sandbox-exec` wrapping in Python's `unix_local.py` (`_confined_exec_command`,
/// `_darwin_exec_profile`, `_darwin_additional_read_paths`, `_darwin_extra_path_grant_roots`).
/// The profile denies the host's user data and system directories, then re-allows the workspace
/// root, the extra path grants and the directories the command's own interpreter lives in. It
/// restricts filesystem access only; it provides no network isolation.
///
/// Everything here takes the facts it depends on (the platform, `PATH`, `which`, the home
/// directory) as parameters, so the profile a given configuration produces can be checked on any
/// host. The session passes the real values.
pub mod confinement {
    use std::path::{Path, PathBuf};

    use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
    use crate::sandbox::workspace_paths::SandboxPathGrant;

    /// Locations a confined command may not read or write unless a later rule re-allows them.
    const DENIED_PATHS: [&str; 12] = [
        "/Users",
        "/Volumes",
        "/Applications",
        "/Library",
        "/opt",
        "/etc",
        "/private/etc",
        "/tmp",
        "/private/tmp",
        "/private",
        "/var",
        "/usr",
    ];

    /// Looks an executable up the way `shutil.which(name, path=...)` does.
    pub type Which<'a> = &'a dyn Fn(&str, Option<&str>) -> Option<PathBuf>;

    /// Everything [`confined_exec_command`] needs to decide how to wrap a command.
    pub struct ConfinementInputs<'a> {
        /// Whether the host is macOS. Anywhere else the command is returned unchanged.
        pub is_darwin: bool,
        /// The platform name reported in the error when confinement is unavailable.
        pub platform: &'a str,
        /// The command and its arguments.
        pub command_parts: &'a [String],
        /// The resolved workspace root.
        pub workspace_root: &'a Path,
        /// The `PATH` the child will see.
        pub child_path: Option<&'a str>,
        /// The host process's own `PATH`.
        pub host_path: Option<&'a str>,
        /// Whether the client lets the host `PATH` through to the child. Only then may a host
        /// entry widen a `bin` grant to its virtual environment root.
        pub host_path_inherited: bool,
        /// The resolved home directory of the host user.
        pub host_home: &'a Path,
        /// The manifest's extra path grants.
        pub extra_path_grants: &'a [SandboxPathGrant],
        /// Executable lookup.
        pub which: Which<'a>,
    }

    /// The facts about the host that decide how a command is wrapped. The session passes the
    /// real ones; a caller checking a configuration passes its own.
    pub struct HostFacts<'a> {
        /// Whether the host is macOS.
        pub is_darwin: bool,
        /// The platform name reported when confinement is unavailable.
        pub platform: &'a str,
        /// The host process's own `PATH`.
        pub host_path: Option<&'a str>,
        /// The resolved home directory of the host user.
        pub host_home: &'a Path,
        /// Executable lookup.
        pub which: Which<'a>,
    }

    impl std::fmt::Debug for ConfinementInputs<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // `which` is a closure and has nothing to print.
            f.debug_struct("ConfinementInputs")
                .field("is_darwin", &self.is_darwin)
                .field("platform", &self.platform)
                .field("command_parts", &self.command_parts)
                .field("workspace_root", &self.workspace_root)
                .field("child_path", &self.child_path)
                .field("host_path_inherited", &self.host_path_inherited)
                .finish_non_exhaustive()
        }
    }

    impl std::fmt::Debug for HostFacts<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HostFacts")
                .field("is_darwin", &self.is_darwin)
                .field("platform", &self.platform)
                .field("host_home", &self.host_home)
                .finish_non_exhaustive()
        }
    }

    /// Wrap a command in `sandbox-exec` on macOS, or return it unchanged elsewhere.
    ///
    /// Fails with [`ErrorCode::ExecTransportError`] and the context reason
    /// `unix_local_confinement_unavailable` when `sandbox-exec` cannot be found: running the
    /// command unconfined would silently drop the restriction the caller relies on.
    pub fn confined_exec_command(inputs: &ConfinementInputs<'_>) -> SandboxResult<Vec<String>> {
        if !inputs.is_darwin {
            return Ok(inputs.command_parts.to_vec());
        }
        let Some(sandbox_exec) = (inputs.which)("sandbox-exec", None) else {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!(
                    "Failed to run `{}`: sandbox-exec is not available",
                    inputs.command_parts.join(" ")
                ),
            )
            .with_context("reason", "unix_local_confinement_unavailable")
            .with_context("platform", inputs.platform)
            .with_context(
                "workspace_root",
                inputs.workspace_root.to_string_lossy().into_owned(),
            ));
        };
        let profile = exec_profile(
            inputs.workspace_root,
            &additional_read_paths(inputs),
            &extra_path_grant_roots(inputs.extra_path_grants)?,
        );
        let mut command = vec![
            sandbox_exec.to_string_lossy().into_owned(),
            "-p".to_string(),
            profile,
        ];
        command.extend(inputs.command_parts.iter().cloned());
        Ok(command)
    }

    /// The `sandbox-exec` profile for a workspace.
    ///
    /// Rules are ordered, and the last matching one wins: the denials come first and every allow
    /// after them, so a read-write grant listed before a read-only grant nested inside it is
    /// overridden by the deny that follows.
    pub fn exec_profile(
        workspace_root: &Path,
        extra_read_paths: &[PathBuf],
        extra_path_grants: &[(PathBuf, bool)],
    ) -> String {
        let denied_paths: Vec<PathBuf> = DENIED_PATHS.iter().map(PathBuf::from).collect();
        let mut allow_rules = vec![
            format!(
                "(allow file-read-data file-read-metadata (subpath {}))",
                literal(workspace_root)
            ),
            format!("(allow file-write* (subpath {}))", literal(workspace_root)),
        ];
        allow_rules.extend(extra_read_paths.iter().map(|path| {
            format!(
                "(allow file-read-data file-read-metadata (subpath {}))",
                literal(path)
            )
        }));
        allow_rules.extend(extra_path_grants.iter().map(|(path, _)| {
            format!(
                "(allow file-read-data file-read-metadata (subpath {}))",
                literal(path)
            )
        }));
        allow_rules.extend(
            extra_path_grants
                .iter()
                .filter(|(_, read_only)| !read_only)
                .map(|(path, _)| format!("(allow file-write* (subpath {}))", literal(path))),
        );
        allow_rules.extend(
            extra_path_grants
                .iter()
                .filter(|(_, read_only)| *read_only)
                .map(|(path, _)| format!("(deny file-write* (subpath {}))", literal(path))),
        );
        for fixed in [
            r#"(allow file-read-data file-read-metadata (subpath "/usr/bin"))"#,
            r#"(allow file-read-data file-read-metadata (subpath "/usr/lib"))"#,
            r#"(allow file-read-data file-read-metadata (subpath "/bin"))"#,
            r#"(allow file-read-data file-read-metadata (subpath "/System"))"#,
            r#"(allow file-read-data file-read-metadata (literal "/private/var/select/sh"))"#,
            r#"(allow file-write* (literal "/dev/null"))"#,
        ] {
            allow_rules.push(fixed.to_string());
        }
        let deny_rules = denied_paths
            .iter()
            .map(|path| {
                format!(
                    "(deny file-read-data (subpath {0}))\n(deny file-write* (subpath {0}))",
                    literal(path)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut lines = vec![
            "(version 1)".to_string(),
            "(allow default)".to_string(),
            // Launch Services can start applications outside this process's sandbox.
            "(deny lsopen)".to_string(),
            deny_rules,
        ];
        lines.extend(allow_rules);
        lines.join("\n")
    }

    /// The extra path grants as `(root, read_only)` pairs, each spelled as declared and as
    /// resolved, so a profile matches whichever spelling the kernel reports.
    ///
    /// A grant that is, or resolves to, the filesystem root is refused: it would hand the
    /// command the whole machine.
    pub fn extra_path_grant_roots(
        grants: &[SandboxPathGrant],
    ) -> SandboxResult<Vec<(PathBuf, bool)>> {
        let mut roots: Vec<(PathBuf, bool)> = Vec::new();
        let mut append = |path: PathBuf, read_only: bool| -> SandboxResult<()> {
            raise_if_filesystem_root(&path)?;
            if !roots.iter().any(|(p, ro)| *p == path && *ro == read_only) {
                roots.push((path, read_only));
            }
            Ok(())
        };
        for grant in grants {
            let grant_path = PathBuf::from(&grant.path);
            let resolved = resolve_lenient(&grant_path);
            raise_if_filesystem_root(&resolved)?;
            append(grant_path.clone(), grant.read_only)?;
            if resolved != grant_path {
                append(resolved, grant.read_only)?;
            }
        }
        Ok(roots)
    }

    fn raise_if_filesystem_root(path: &Path) -> SandboxResult<()> {
        if path == Path::new("/") {
            return Err(SandboxError::new(
                ErrorCode::InvalidManifestPath,
                "sandbox path grant path must not resolve to filesystem root",
            ));
        }
        Ok(())
    }

    /// The directories outside the workspace a command needs to read to run at all: every `PATH`
    /// entry, the directory the executable lives in, and the roots those belong to (a Homebrew
    /// prefix, a dotted directory under the home directory).
    pub fn additional_read_paths(inputs: &ConfinementInputs<'_>) -> Vec<PathBuf> {
        let mut allowed: Vec<PathBuf> = Vec::new();
        let mut append = |path: Option<&Path>, allow_virtual_environment_root: bool| {
            let Some(path) = path else { return };
            let candidate = expanduser(path, inputs.host_home);
            if !candidate.is_absolute() {
                return;
            }
            for root in
                allowable_read_roots(&candidate, inputs.host_home, allow_virtual_environment_root)
            {
                if !allowed.contains(&root) {
                    allowed.push(root);
                }
            }
        };

        let child_entries: Vec<&str> = inputs
            .child_path
            .map(|path| path.split(':').collect())
            .unwrap_or_default();
        for entry in &child_entries {
            if !entry.is_empty() {
                append(Some(Path::new(entry)), false);
            }
        }
        let executable = inputs
            .command_parts
            .first()
            .and_then(|name| (inputs.which)(name, inputs.child_path));
        append(executable.as_deref(), false);

        // The client has to let `PATH` through before a retained host entry may widen a `bin`
        // grant to its virtual environment root. A value the manifest merely happens to share
        // with the host cannot grant that authority.
        if inputs.host_path_inherited {
            for entry in inputs.host_path.unwrap_or("").split(':') {
                if !entry.is_empty() && child_entries.contains(&entry) {
                    append(Some(Path::new(entry)), true);
                }
            }
        }
        allowed
    }

    /// The directories a read of `path` should be allowed to reach.
    pub fn allowable_read_roots(
        path: &Path,
        host_home: &Path,
        allow_virtual_environment_root: bool,
    ) -> Vec<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        let mut add = |candidate: PathBuf| {
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        };
        let normalized = expanduser(path, host_home);
        let resolved = resolve_lenient(&normalized);
        let dir_or_parent = |p: &Path| -> PathBuf {
            if p.is_dir() {
                p.to_path_buf()
            } else {
                p.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| p.to_path_buf())
            }
        };
        add(dir_or_parent(&normalized));
        add(dir_or_parent(&resolved));

        if allow_virtual_environment_root {
            for candidate in [&normalized, &resolved] {
                if candidate.file_name().is_none_or(|name| name != "bin") {
                    continue;
                }
                if let Some(root) = candidate.parent()
                    && root.join("pyvenv.cfg").is_file()
                {
                    add(root.to_path_buf());
                }
            }
        }

        let resolved_text = resolved.to_string_lossy();
        for prefix in ["/opt/homebrew", "/usr/local", "/Library/Frameworks"] {
            if resolved_text == prefix || resolved_text.starts_with(&format!("{prefix}/")) {
                add(PathBuf::from(prefix));
            }
        }

        if let Ok(relative) = resolved.strip_prefix(host_home) {
            let mut parts = relative.components();
            if let Some(first) = parts.next() {
                let first_text = first.as_os_str().to_string_lossy().into_owned();
                if first_text.starts_with('.') {
                    add(host_home.join(&first_text));
                } else if first_text == "Library"
                    && parts
                        .next()
                        .is_some_and(|second| second.as_os_str() == "Python")
                {
                    add(host_home.join("Library").join("Python"));
                }
            }
        }

        candidates.sort_by(|a, b| {
            (a.components().count(), a.to_string_lossy().into_owned())
                .cmp(&(b.components().count(), b.to_string_lossy().into_owned()))
        });
        candidates
    }

    /// `~` and `~/...` expanded against `home`.
    fn expanduser(path: &Path, home: &Path) -> PathBuf {
        match path.strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => path.to_path_buf(),
        }
    }

    /// `Path.resolve(strict=False)`: the deepest existing ancestor is canonicalized and the
    /// missing tail re-attached.
    pub fn resolve_lenient(path: &Path) -> PathBuf {
        let mut existing = path.to_path_buf();
        let mut tail: Vec<std::ffi::OsString> = Vec::new();
        loop {
            match existing.canonicalize() {
                Ok(real) => {
                    let mut out = real;
                    for part in tail.iter().rev() {
                        out.push(part);
                    }
                    return out;
                }
                Err(_) => match existing.file_name() {
                    Some(name) => {
                        tail.push(name.to_os_string());
                        if !existing.pop() {
                            return path.to_path_buf();
                        }
                    }
                    None => return path.to_path_buf(),
                },
            }
        }
    }

    /// `shutil.which`: the first executable `name` resolves to along `search_path` (the process's
    /// own `PATH` when `None`).
    pub fn which_in(name: &str, search_path: Option<&str>) -> Option<PathBuf> {
        use std::os::unix::fs::PermissionsExt as _;
        let is_executable = |path: &Path| {
            path.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        };
        if name.contains('/') {
            let path = PathBuf::from(name);
            return is_executable(&path).then_some(path);
        }
        let owned;
        let search = match search_path {
            Some(path) => path,
            None => {
                owned = std::env::var("PATH").unwrap_or_default();
                &owned
            }
        };
        search
            .split(':')
            .filter(|dir| !dir.is_empty())
            .map(|dir| Path::new(dir).join(name))
            .find(|candidate| is_executable(candidate))
    }

    /// A profile string literal: backslashes and quotes escaped.
    fn literal(path: &Path) -> String {
        let escaped = path
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!("\"{escaped}\"")
    }
}

/// Host environment variables a non-inheriting Unix-local session still passes through.
///
/// Mirrors Python's `_HOST_ENVIRONMENT_ALLOWLIST`. Clearing the environment entirely breaks
/// things the sandbox has no way to rediscover: without `SSL_CERT_FILE` and friends a command
/// cannot verify TLS at all, and without the locale variables tools misread non-ASCII output.
/// `HOME` is not here because the session sets it to the workspace root, as Python does, which
/// keeps the agent out of the real home directory.
pub const HOST_ENVIRONMENT_ALLOWLIST: &[&str] = &[
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_COLLATE",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TZ",
    "TERM",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "UV_PYTHON",
    "NO_COLOR",
    "FORCE_COLOR",
    "CI",
];

/// Rewrite a captured archive so every member it keeps can be restored into a fresh root.
///
/// The host's `tar` records what it finds: a hardlinked file becomes a link member pointing at
/// the first copy it saw, and a FIFO, socket or device node becomes a member of that type. The
/// hydrate policy refuses both, so an ordinary workspace would be unrestorable. Python never has
/// the problem because `unix_local._archive_workspace` walks the tree itself and adds each file
/// as a regular member, skipping anything that is not a file, directory or symlink. This brings
/// the captured archive to the same shape: the second name of a hardlinked pair is archived as
/// an independent regular file, and special files are dropped.
fn normalize_restorable_archive(raw: &[u8], root: &Path) -> SandboxResult<Vec<u8>> {
    use std::io::Read as _;

    let gzipped = raw.starts_with(&[0x1f, 0x8b]);
    // The payload of every regular member, so a hardlink is materialized from the archive rather
    // than from a workspace that may have changed since the capture started.
    let mut payloads: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    {
        let mut archive = tar::Archive::new(archive_reader(raw));
        for entry in archive.entries().map_err(unreadable_capture)? {
            let mut entry = entry.map_err(unreadable_capture)?;
            if entry.header().entry_type() != tar::EntryType::Regular {
                continue;
            }
            let name = archive_member_key(&entry.path_bytes());
            let mut contents = Vec::new();
            entry
                .read_to_end(&mut contents)
                .map_err(unreadable_capture)?;
            payloads.insert(name, contents);
        }
    }

    let mut out: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut out);
        let mut archive = tar::Archive::new(archive_reader(raw));
        for entry in archive.entries().map_err(unreadable_capture)? {
            let mut entry = entry.map_err(unreadable_capture)?;
            let mut header = entry.header().clone();
            let path = PathBuf::from(String::from_utf8_lossy(&entry.path_bytes()).into_owned());
            match header.entry_type() {
                tar::EntryType::Directory => builder
                    .append_data(&mut header, &path, std::io::empty())
                    .map_err(unreadable_capture)?,
                tar::EntryType::Symlink => {
                    let target = entry
                        .link_name_bytes()
                        .map(|bytes| PathBuf::from(String::from_utf8_lossy(&bytes).into_owned()))
                        .unwrap_or_default();
                    builder
                        .append_link(&mut header, &path, &target)
                        .map_err(unreadable_capture)?;
                }
                tar::EntryType::Regular => {
                    let mut contents = Vec::new();
                    entry
                        .read_to_end(&mut contents)
                        .map_err(unreadable_capture)?;
                    builder
                        .append_data(&mut header, &path, &contents[..])
                        .map_err(unreadable_capture)?;
                }
                tar::EntryType::Link => {
                    let key = entry
                        .link_name_bytes()
                        .map(|bytes| archive_member_key(&bytes))
                        .unwrap_or_default();
                    let contents = match payloads.get(&key) {
                        Some(contents) => contents.clone(),
                        // The first copy was excluded from the archive, so the bytes only exist
                        // in the workspace.
                        None => std::fs::read(root.join(&key)).unwrap_or_default(),
                    };
                    header.set_entry_type(tar::EntryType::Regular);
                    header.set_size(contents.len() as u64);
                    builder
                        .append_data(&mut header, &path, &contents[..])
                        .map_err(unreadable_capture)?;
                }
                // A FIFO, socket or device node holds nothing a restore could recreate.
                _ => continue,
            }
        }
        builder.finish().map_err(unreadable_capture)?;
    }
    if !gzipped {
        return Ok(out);
    }
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &out).map_err(unreadable_capture)?;
    encoder.finish().map_err(unreadable_capture)
}

fn rm_failure(path: &str, error: &std::io::Error) -> SandboxError {
    SandboxError::new(
        ErrorCode::WorkspaceArchiveWriteError,
        format!("Failed to remove `{path}`: {error}"),
    )
    .with_context("path", path.to_string())
}

/// Rewrite absolute arguments under the workspace root to be relative to it.
///
/// Mirrors Python's `_workspace_relative_command_parts`. The program itself is left alone.
fn workspace_relative_command_parts(command: Vec<String>, workspace_root: &Path) -> Vec<String> {
    let mut parts = command.into_iter();
    let Some(program) = parts.next() else {
        return Vec::new();
    };
    let mut rewritten = vec![program];
    for part in parts {
        let path = Path::new(&part);
        if !path.is_absolute() {
            rewritten.push(part);
            continue;
        }
        match path.strip_prefix(workspace_root) {
            Ok(relative) if relative.as_os_str().is_empty() => rewritten.push(".".to_string()),
            Ok(relative) => rewritten.push(relative.to_string_lossy().into_owned()),
            Err(_) => rewritten.push(part),
        }
    }
    rewritten
}

/// Anchor a `sh -c` command at the workspace and start the process from `/`.
///
/// Mirrors Python's `_shell_workspace_process_context`: the shell changes into the directory
/// itself, so the process never needs a working directory the profile might not allow.
fn shell_workspace_process_context(
    command: Vec<String>,
    workspace_root: &Path,
    cwd: &Path,
) -> (PathBuf, Vec<String>) {
    if command.len() < 3 || command[0] != "sh" || command[1] != "-c" {
        return (cwd.to_path_buf(), command);
    }
    let target = if cwd == workspace_root || cwd.as_os_str().is_empty() {
        workspace_root.to_path_buf()
    } else {
        confinement::resolve_lenient(cwd)
    };
    let mut rewritten = command;
    // A brace group, not a bare prefix: in `cd X && a & wait; b` the `&` would background the
    // whole `cd X && a`, leaving `b` in the process's own directory, so the directory would not
    // apply to the entire command list. The group also keeps a multi-line script together.
    rewritten[2] = format!(
        "cd {} && {{\n{}\n}}",
        crate::sandbox::session::shell_quote(&target.to_string_lossy()),
        rewritten[2]
    );
    (PathBuf::from("/"), rewritten)
}

/// Replace each dangling symlink component of `path` with what it points at.
///
/// Bounded at 40 hops, the same limit the kernel applies, so a link loop terminates.
async fn resolve_dangling_components(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    for _ in 0..40 {
        let components: Vec<_> = current.components().collect();
        let mut prefix = PathBuf::new();
        let mut replacement = None;
        for (index, component) in components.iter().enumerate() {
            prefix.push(component.as_os_str());
            let Ok(metadata) = tokio::fs::symlink_metadata(&prefix).await else {
                break;
            };
            if metadata.file_type().is_symlink() && tokio::fs::metadata(&prefix).await.is_err() {
                if let Ok(target) = tokio::fs::read_link(&prefix).await {
                    let base = prefix.parent().unwrap_or(Path::new("/"));
                    let mut next = if target.is_absolute() {
                        target
                    } else {
                        base.join(target)
                    };
                    for rest in &components[index + 1..] {
                        next.push(rest.as_os_str());
                    }
                    replacement = Some(next);
                }
                break;
            }
        }
        match replacement {
            Some(next) => current = next,
            None => return current,
        }
    }
    current
}

/// A callback invoked with the validated host path just before the file is opened.
#[derive(Clone)]
struct AfterValidateHook(Arc<dyn Fn(&Path) + Send + Sync>);

impl std::fmt::Debug for AfterValidateHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AfterValidateHook")
    }
}

/// The identity of the file behind an open handle or a path: device and inode.
fn file_identity(metadata: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt as _;
    (metadata.dev(), metadata.ino())
}

/// The synchronous twin of [`resolve_existing_prefix`], used while a session is being built.
/// One extra path grant resolved against the filesystem: the grant's spelling, the real
/// directory it named when it was pinned, and its write policy.
type PinnedGrant = (String, PathBuf, bool);

/// Pin every grant to the real directory it names now. A grant whose spelling is already in
/// `previous` keeps that pin, so re-pointing an alias between two capability updates widens
/// nothing; a read-only change on an unchanged alias still takes effect.
fn pin_grants(grants: &[SandboxPathGrant], previous: &[PinnedGrant]) -> Vec<PinnedGrant> {
    grants
        .iter()
        .map(|grant| {
            let real = previous
                .iter()
                .find(|(path, _, _)| *path == grant.path)
                // A grant may name a file that does not exist yet — a move destination, say — so
                // the deepest existing ancestor is what gets pinned.
                .map_or_else(
                    || pin_existing_prefix(Path::new(&grant.path)),
                    |(_, real, _)| real.clone(),
                );
            (grant.path.clone(), real, grant.read_only)
        })
        .collect()
}

fn pin_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(real) => {
                let mut out = real;
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return out;
            }
            Err(_) => match existing.file_name() {
                Some(name) => {
                    tail.push(name.to_os_string());
                    if !existing.pop() {
                        return path.to_path_buf();
                    }
                }
                None => return path.to_path_buf(),
            },
        }
    }
}

/// `O_NONBLOCK` for the platforms this backend runs on, so no `libc` dependency is needed to ask
/// for a non-blocking open.
#[cfg(target_os = "linux")]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(all(unix, not(target_os = "linux")))]
const O_NONBLOCK: i32 = 0x0004;

/// Refuse a workspace entry that exists but is not a regular file.
///
/// Mirrors the `fstat` check Python's `_unix_local_file_ops` performs on every descriptor it
/// opens: a FIFO, socket or device node is never a file the sandbox API may read or write. The
/// entry itself is examined, so a symlink is followed only as far as the kernel already would —
/// a dangling link is left to the open to report as missing.
async fn reject_special_file(host: &Path, path: &str, for_write: bool) -> SandboxResult<()> {
    let Ok(metadata) = tokio::fs::metadata(host).await else {
        // Missing, or unreadable for a reason the real operation reports better.
        return Ok(());
    };
    if metadata.is_file() || metadata.is_dir() {
        return Ok(());
    }
    let code = if for_write {
        ErrorCode::WorkspaceArchiveWriteError
    } else {
        ErrorCode::WorkspaceArchiveReadError
    };
    Err(
        SandboxError::new(code, format!("`{path}` is not a regular file"))
            .with_context("path", path.to_string())
            .with_context("reason", "not_a_regular_file"),
    )
}

/// Read a captured archive, transparently decompressing the gzip the default capture produces.
fn archive_reader(raw: &[u8]) -> Box<dyn std::io::Read + '_> {
    if raw.starts_with(&[0x1f, 0x8b]) {
        Box::new(flate2::read::GzDecoder::new(raw))
    } else {
        Box::new(raw)
    }
}

/// The archive-relative name a member and a hardlink target are matched on.
fn archive_member_key(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw)
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_string()
}

fn unreadable_capture(error: std::io::Error) -> SandboxError {
    SandboxError::new(
        ErrorCode::SnapshotPersistError,
        format!("Failed to normalize the captured workspace archive: {error}"),
    )
}

/// Canonicalize the deepest existing ancestor of `path` and re-attach the missing tail.
///
/// This follows symlinks for the parts that exist, so a link planted inside the workspace that
/// points outside it is detected before the path is used.
async fn resolve_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match tokio::fs::canonicalize(&existing).await {
            Ok(real) => {
                let mut out = real;
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return out;
            }
            Err(_) => match existing.file_name() {
                Some(name) => {
                    tail.push(name.to_os_string());
                    if !existing.pop() {
                        return path.to_path_buf();
                    }
                }
                None => return path.to_path_buf(),
            },
        }
    }
}

#[async_trait]
impl SandboxSession for UnixLocalSandboxSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn live_grants(&self) -> Option<&crate::sandbox::session::LiveGrants> {
        Some(&self.live_grants)
    }

    fn update_extra_path_grants(&self, grants: &[SandboxPathGrant]) {
        self.live_grants.replace(grants.to_vec());
        // Unchanged aliases stay pinned to the directory they named first, so an occupant
        // re-pointing one between updates cannot widen what the sandbox reaches.
        let mut pinned = self
            .pinned_grants
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *pinned = pin_grants(grants, &pinned);
    }

    fn backend_id(&self) -> &str {
        "unix_local"
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.exposed_ports.clone()
    }

    fn persist_skip_registry(
        &self,
    ) -> Option<&std::sync::Mutex<std::collections::BTreeSet<String>>> {
        Some(&self.persist_skip)
    }

    /// A declared port is already reachable on loopback: the command ran on this host.
    ///
    /// Mirrors Python's `UnixLocalSandboxSession.resolve_exposed_port`, which hands back
    /// `127.0.0.1` without TLS. An undeclared port is still refused, so the declaration stays
    /// the contract between the application and the sandbox.
    async fn resolve_exposed_port(
        &self,
        port: u16,
    ) -> SandboxResult<crate::sandbox::types::ExposedPortEndpoint> {
        if self.exposed_ports.contains(&port) {
            return Ok(crate::sandbox::types::ExposedPortEndpoint::new(
                "127.0.0.1",
                port,
                false,
            ));
        }
        Err(crate::sandbox::types::exposed_port_unavailable(
            self.backend_id(),
            port,
            &self.exposed_ports,
            crate::sandbox::types::ExposedPortUnavailable::NotConfigured,
        ))
    }

    async fn start(&self) -> SandboxResult<()> {
        tokio::fs::create_dir_all(&self.root).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!("Failed to create workspace `{}`: {e}", self.root.display()),
            )
        })?;
        // The root exists now, so a session created over a path that did not yet exist gets its
        // pin here, before any path check can run.
        if self.pinned_root.get().is_none()
            && let Ok(canonical) = tokio::fs::canonicalize(&self.root).await
        {
            let _ = self.pinned_root.set(canonical);
        }
        // Provisioning accounts would run on the host machine, so Python refuses a manifest that
        // names users or groups rather than quietly skipping them.
        if !self.manifest.users().is_empty() || !self.manifest.groups().is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "UnixLocalSandboxSession does not support manifest users or groups because \
                 provisioning would run on the host machine",
            ));
        }
        self.running.store(true, Ordering::SeqCst);
        self.mounts_detached.store(false, Ordering::SeqCst);
        // Start-up environment lookups run once, before any command can observe the environment.
        if !self.manifest().resolved_environment().is_empty() {
            let resolved = self.manifest().resolve_environment().await?;
            let _ = self.resolved_environment.set(resolved);
        }
        let result = if self.resuming {
            match restore_session_snapshot(self).await {
                Ok(true) => Ok(()),
                // Without a stored archive, the workspace the paused run left behind is all
                // there is. Re-apply only the ephemeral entries when it survived; when it is
                // gone (an evicted temp directory, a replacement container, a snapshot-less
                // backend) materialize the whole manifest instead of handing the model an
                // empty workspace.
                Ok(false) => {
                    let preserved = self.materialized.load(Ordering::SeqCst)
                        && self.workspace_has_content().await;
                    let manifest = self.manifest.clone();
                    if preserved {
                        manifest.apply_ephemeral(self).await.map(|_| ())
                    } else {
                        manifest.apply(self).await.map(|_| ())
                    }
                }
                Err(error) => Err(error),
            }
        } else {
            self.manifest.clone().apply(self).await.map(|_| ())
        };
        result.inspect_err(|_| {
            self.running.store(false, Ordering::SeqCst);
        })?;
        // Only now is the workspace what the manifest says it is; a resume before this point
        // would find a half-materialized tree that looks populated.
        self.materialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        crate::sandbox::runtime::sandbox_span_result(
            "sandbox.stop",
            self.span_data("stop"),
            async {
                self.ptys.terminate_all();
                self.manifest.unmount_all(self).await?;
                self.mounts_detached.store(true, Ordering::SeqCst);
                self.running.store(false, Ordering::SeqCst);
                Ok(())
            },
        )
        .await
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        crate::sandbox::runtime::sandbox_span_result(
            "sandbox.shutdown",
            self.span_data("shutdown"),
            self.shutdown_untraced(),
        )
        .await
    }

    async fn running(&self) -> SandboxResult<bool> {
        Ok(self.running.load(Ordering::SeqCst))
    }

    fn resolves_environment(&self) -> bool {
        true
    }

    fn mount_transition(&self) -> Option<&crate::sandbox::mounts::MountTransition> {
        Some(&self.mount_transition)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        self.normalize_spelled(path, false)
    }

    fn normalize_path_for_write(&self, path: &str) -> SandboxResult<String> {
        self.normalize_spelled(path, true)
    }

    fn supports_pty(&self) -> bool {
        true
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let cwd = match &request.cwd {
            None => self.root.clone(),
            Some(cwd) => self.host_path_for(cwd, false).await?,
        };
        let (argv, cwd) = self.confine_command(
            request.resolved_argv(),
            cwd,
            self.child_path(&Default::default()),
        )?;
        let spawn = PtySpawn {
            argv,
            cwd: Some(cwd),
            env: self.exec_env(),
            clear_env: !self.inherit_host_environment,
            tty: request.tty,
            remote_kill_command: None,
        };
        let (update, _cleanup) = self
            .ptys
            .start(spawn, request.yield_time_ms, request.max_output_tokens)
            .await?;
        Ok(update)
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let (update, _cleanup) = self.ptys.write_stdin(&request).await?;
        Ok(update)
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        self.ptys.terminate_all();
        Ok(())
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let argv = request.resolved_argv();
        let Some((program, args)) = argv.split_first() else {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "exec requires at least one command word",
            ));
        };
        let cwd = match &request.cwd {
            None => self.root.clone(),
            Some(cwd) => self.host_path_for(cwd, false).await?,
        };
        let (confined, process_cwd) =
            self.confine_command(argv.clone(), cwd, self.child_path(&request.env))?;
        let (program, args) = confined.split_first().unwrap_or((program, args));
        let mut command = tokio::process::Command::new(program);
        command
            .args(args)
            .current_dir(&process_cwd)
            .stdin(Stdio::null())
            .kill_on_drop(true);
        if !self.inherit_host_environment {
            command.env_clear();
        }
        // `exec_env` already folds in the manifest's literals and its start-up lookups, in that
        // order, so re-applying the literals here would let one override a looked-up value of
        // the same name. Only the request's own variables are applied on top.
        for (key, value) in self.exec_env() {
            command.env(key, value);
        }
        for (key, value) in &request.env {
            command.env(key, value);
        }
        let run = command.output();
        let output = match request.timeout {
            None => run.await,
            Some(timeout) => match tokio::time::timeout(timeout, run).await {
                Ok(output) => output,
                Err(_) => {
                    return Err(SandboxError::new(
                        ErrorCode::ExecTimeout,
                        format!("Command `{}` timed out after {:?}", argv.join(" "), timeout),
                    ));
                }
            },
        };
        let output = output.map_err(|e| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to run `{}`: {e}", argv.join(" ")),
            )
        })?;
        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.read_workspace_file(path, None).await
    }

    /// A native prefix read: only `max_bytes` leave the file, so a log the workload keeps
    /// appending to is never read whole just to look at its start. No process is started.
    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        self.read_workspace_file(path, Some(max_bytes as u64)).await
    }

    /// Move a workspace entry with `rename(2)`.
    ///
    /// The default implementation shells out to `mv`, which puts the source *inside* an existing
    /// directory destination and exits 0. A caller that then removes the source — the editor's
    /// `move_to` does — would delete the file. Python uses `os.rename` for exactly this reason,
    /// and refuses a source that is not a regular file or a symlink so a tree can never be moved
    /// by a path operation.
    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let from = self.host_path_for(source, true).await?;
        let to = self.host_path_for(destination, true).await?;
        self.run_after_validate(&from);
        let (from_dir, from_name) = self.pin_parent(&from, source, true).await?;
        let (to_dir, to_name) = self.pin_parent(&to, destination, true).await?;
        let (source_label, destination_label) = (source.to_string(), destination.to_string());
        tokio::task::spawn_blocking(move || {
            let moving = |message: String| SandboxError::new(ErrorCode::ExecNonZero, message);
            let stat = pinned_ops::stat_at(&from_dir, &from_name, false)
                .map_err(|e| moving(format!("Failed to move `{source_label}`: {e}")))?;
            let kind = pinned_ops::kind_of(&stat);
            if !(kind == rustix::fs::FileType::RegularFile || kind == rustix::fs::FileType::Symlink)
            {
                return Err(moving(format!(
                    "`{source_label}` must be a regular file or symlink to be moved"
                )));
            }
            pinned_ops::rename_at(&from_dir, &from_name, &to_dir, &to_name).map_err(|e| {
                moving(format!(
                    "Failed to move `{source_label}` to `{destination_label}`: {e}"
                ))
            })
        })
        .await
        .map_err(|e| SandboxError::new(ErrorCode::ExecNonZero, format!("move task failed: {e}")))?
    }

    /// Claim the leaf exclusively, so a name that appeared after the caller looked is refused.
    async fn write_new_file(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        if user.is_some() {
            // A run-as user goes through `exec`: the contents are staged and then hard-linked
            // into place, which fails for any name that exists by then.
            if self.read_file_as(path, user).await.is_ok() {
                return Err(SandboxError::new(
                    ErrorCode::ApplyPatchInvalidPath,
                    format!("`{path}` already exists"),
                ));
            }
            let target = self.normalize_path_for_write(path)?;
            let parent = target.rsplit_once('/').map_or("/", |(parent, _)| parent);
            self.mkdir_as(parent, true, user).await?;
            let host = self.host_path_for(path, true).await?;
            self.run_after_validate(&host);
            return self.write_file_as_with(path, contents, user, true).await;
        }
        use tokio::io::AsyncWriteExt as _;
        let host = self.host_path_for(path, true).await?;
        if let Some(parent) = host.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("Failed to create `{}`: {e}", parent.display()),
                )
            })?;
        }
        self.run_after_validate(&host);
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&host)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    SandboxError::new(
                        ErrorCode::ApplyPatchInvalidPath,
                        format!("`{path}` already exists"),
                    )
                } else {
                    SandboxError::new(
                        ErrorCode::WorkspaceWriteTypeError,
                        format!("Failed to write `{path}`: {e}"),
                    )
                }
            })?;
        let failed = |e: std::io::Error| {
            SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to write `{path}`: {e}"),
            )
        };
        let handle = file.metadata().await.map_err(failed)?;
        if let Err(error) = self.verify_opened(&handle, &host, path, true).await {
            // This call created the file, so a creation that landed outside is undone.
            self.discard_created(&handle, &host).await;
            return Err(error);
        }
        // From here the name is claimed. A failure leaves whatever was written and the file is
        // never deleted: another process may already own it, and retrying without looking is
        // unsafe, which the error says.
        let incomplete = |_: std::io::Error| {
            SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                INCOMPLETE_CREATE_MESSAGE,
            )
            .with_context("path", host.to_string_lossy())
            .with_retryable(false)
        };
        file.write_all(contents).await.map_err(incomplete)?;
        // `tokio::fs::File` hands its writes to a blocking thread, and dropping it does not wait
        // for them. Without a flush a caller that reads the file straight back can see it empty.
        file.flush().await.map_err(incomplete)
    }

    /// Compare device and inode of the two leaves directly instead of shelling out to `test`.
    async fn same_file(
        &self,
        left: &str,
        right: &str,
        follow_symlinks: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<bool> {
        if user.is_some() {
            // Fall back to the exec-based check, which runs as the requested user.
            let l = self.normalize_path(left)?;
            let r = self.normalize_path(right)?;
            let mut test = "[ \"$1\" -ef \"$2\" ]".to_string();
            if !follow_symlinks {
                test = format!("[ ! -L \"$1\" ] && [ ! -L \"$2\" ] && {test}");
            }
            let result = self
                .exec(
                    ExecRequest::argv(["sh", "-c", test.as_str(), "sh", l.as_str(), r.as_str()])
                        .with_shell(crate::sandbox::session::ExecShell::None)
                        .with_user(user.cloned()),
                )
                .await?;
            return match result.exit_code {
                0 => Ok(true),
                1 => Ok(false),
                _ => Err(SandboxError::exec_nonzero(
                    &[
                        "sh".to_string(),
                        "-c".to_string(),
                        "<same_file_check>".to_string(),
                    ],
                    &result,
                )),
            };
        }
        let left_host = self.host_path_for(left, false).await?;
        let right_host = self.host_path_for(right, false).await?;
        self.run_after_validate(&left_host);
        let (left_dir, left_name) = self.pin_parent(&left_host, left, false).await?;
        let (right_dir, right_name) = self.pin_parent(&right_host, right, false).await?;
        tokio::task::spawn_blocking(move || {
            let stat =
                |dir, name: &std::ffi::OsStr| pinned_ops::stat_at(dir, name, follow_symlinks);
            let (Ok(l), Ok(r)) = (stat(&left_dir, &left_name), stat(&right_dir, &right_name))
            else {
                return false;
            };
            if !follow_symlinks
                && (pinned_ops::kind_of(&l) == rustix::fs::FileType::Symlink
                    || pinned_ops::kind_of(&r) == rustix::fs::FileType::Symlink)
            {
                return false;
            }
            (l.st_dev, l.st_ino) == (r.st_dev, r.st_ino)
        })
        .await
        .map_err(|e| {
            SandboxError::new(
                ErrorCode::ExecNonZero,
                format!("same_file task failed: {e}"),
            )
        })
    }

    /// List a directory natively, so a name is never re-parsed out of `ls -la` text.
    ///
    /// Mirrors Python's native listing, which reports exact names: a name containing a newline
    /// or runs of spaces survives intact. Symlinks inside the workspace are resolved first, so a
    /// listing through a safe alias reports the entries' real locations; the session's own root
    /// spelling is kept so callers that match entries against the workspace root still do.
    async fn ls(&self, path: &str) -> SandboxResult<Vec<crate::sandbox::types::FileEntry>> {
        let host = self.host_path_for(path, false).await?;
        // A directory swapped for a link after validation must not be listed through it.
        self.run_after_validate(&host);
        let host = self.host_path_for(path, false).await?;
        let base = match tokio::fs::canonicalize(&host).await {
            Ok(real) => match self
                .pinned_root
                .get()
                .and_then(|root| real.strip_prefix(root).ok())
            {
                Some(rest) => self.root.join(rest),
                None => real,
            },
            Err(_) => host.clone(),
        };
        let inaccessible = |e: &dyn std::fmt::Display| {
            SandboxError::new(
                ErrorCode::ExecNonZero,
                format!("ls: cannot access `{path}`: {e}"),
            )
            .with_context("path", path.to_string())
        };
        // The directory is resolved again now, must still be authorized, and is then opened
        // without following a link: a directory swapped for a link after validation is refused,
        // and one swapped later cannot redirect the listing.
        let real = tokio::fs::canonicalize(&host)
            .await
            .map_err(|e| inaccessible(&e))?;
        self.authorize_real(host.clone(), real.clone(), path, false)
            .await
            .map_err(|_| {
                SandboxError::new(
                    ErrorCode::InvalidManifestPath,
                    format!("`{path}` changed while it was being validated"),
                )
                .with_context("path", path.to_string())
            })?;
        let found = tokio::task::spawn_blocking(move || {
            let dir = pinned_ops::open_dir_pinned(&real)?;
            pinned_ops::list_dir(&dir)
        })
        .await
        .map_err(|e| inaccessible(&e))?
        .map_err(|e| inaccessible(&e))?;
        let mut listing = Vec::new();
        for (os_name, stat) in found {
            let name = os_name.to_string_lossy().into_owned();
            let kind = pinned_ops::kind_of(&stat);
            listing.push(crate::sandbox::types::FileEntry {
                path: base.join(&name).to_string_lossy().into_owned(),
                name,
                size: Some(stat.st_size as u64),
                permissions: Some(crate::sandbox::types::Permissions::from_mode(
                    stat.st_mode as u32,
                )),
                is_dir: kind == rustix::fs::FileType::Directory,
                is_symlink: kind == rustix::fs::FileType::Symlink,
            });
        }
        listing.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(listing)
    }

    /// List a directory for `user`: the user's own `ls` must succeed under `sudo -u`, and the
    /// entries are then read natively, so a name holding a newline or runs of spaces is reported
    /// exactly, as Python's user-listing worker reports it.
    async fn ls_as(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<Vec<crate::sandbox::types::FileEntry>> {
        let Some(user) = user else {
            return self.ls(path).await;
        };
        let host = self.host_path_for(path, false).await?;
        let command = vec![
            "ls".to_string(),
            "-la".to_string(),
            "--".to_string(),
            host.to_string_lossy().into_owned(),
        ];
        let result = self
            .exec(ExecRequest::argv(command.clone()).with_user(Some(user.clone())))
            .await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        self.ls(path).await
    }

    /// Create a directory for `user`: their permission is probed with the access-check script
    /// under `sudo -u`, and the directory is then created natively.
    ///
    /// Mirrors Python's `UnixLocalSandboxSession.mkdir(user=...)`, which checks with
    /// `_check_mkdir_with_exec` and then uses the local filesystem, so nothing is created by
    /// (or owned by) the impersonated account.
    async fn mkdir_as(
        &self,
        path: &str,
        parents: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.mkdir(path, parents).await;
        };
        let host = self.host_path_for(path, true).await?;
        self.check_mkdir_access(&host.to_string_lossy(), parents, Some(user))
            .await?;
        self.mkdir(path, parents).await
    }

    /// Remove a path for `user`: their permission is probed under `sudo -u`, then the removal
    /// itself is native. Mirrors Python's `UnixLocalSandboxSession.rm(user=...)`.
    async fn rm_as(
        &self,
        path: &str,
        recursive: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.rm(path, recursive).await;
        };
        if recursive {
            self.workspace_path_policy()?
                .validate_recursive_remove(path)?;
        }
        self.host_path_for(path, true).await?;
        let normalized = self.normalize_path_for_write(path)?;
        self.check_rm_access(&normalized, recursive, Some(user))
            .await?;
        self.rm(path, recursive).await
    }

    /// Remove a file or directory natively.
    ///
    /// Mirrors Python's `rm`: the path is normalized and authorized first, and a leaf symlink
    /// that stays inside the workspace is followed, so removing a safe internal alias
    /// removes what it names. A link that dangles, or that points outside the workspace, is
    /// unlinked itself and its target is never touched. A missing path is not an error (`rm
    /// -f`), and a directory needs `recursive`.
    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let policy = self.workspace_path_policy()?;
        if recursive {
            // A recursive removal of an ancestor is a write to every grant under it.
            policy.validate_recursive_remove(path)?;
        }
        // `host_path_for` authorizes the path, but for a path under a grant it hands back the
        // *resolved* location. Removal has to act on the entry the caller named, never on what a
        // link in that position points at, so the normalized spelling is used instead.
        self.host_path_for(path, true).await?;
        let host = PathBuf::from(self.normalize_path_for_write(path)?);
        self.run_after_validate(&host);
        if recursive {
            self.refuse_removal_over_read_only_grants(&host, path)
                .await?;
        }
        let mut target = host.clone();
        if let Ok(metadata) = tokio::fs::symlink_metadata(&host).await
            && metadata.file_type().is_symlink()
            && let Ok(real) = tokio::fs::canonicalize(&host).await
            && self
                .pinned_root
                .get()
                .is_some_and(|root| real.starts_with(root))
        {
            // Only an alias that stays inside the workspace is followed. A link into a grant is
            // unlinked as a link: resuming a snapshot clears the workspace through this method,
            // and a stale link must never reach through to the directory it names.
            target = real;
        }
        // The entry's parent is pinned now; the entry is inspected and removed relative to it.
        let (dir, name) = self.pin_parent(&target, path, true).await?;
        let shown = target.to_string_lossy().into_owned();
        let label = path.to_string();
        tokio::task::spawn_blocking(move || {
            match pinned_ops::stat_at(&dir, &name, false) {
                Ok(_) => {}
                // A recursive removal of nothing succeeds, so clearing a workspace is
                // idempotent. A plain `rm` of a path that is not there is the caller's mistake,
                // and Python reports it as a non-zero `rm`.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && recursive => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(SandboxError::exec_nonzero(
                        &["rm".to_string(), "--".to_string(), shown.clone()],
                        &ExecResult {
                            stdout: Vec::new(),
                            stderr: format!("rm: cannot remove `{shown}`: {e}").into_bytes(),
                            exit_code: 1,
                        },
                    ));
                }
                Err(e) => return Err(rm_failure(&label, &e)),
            }
            // `rmdir` for a directory without `recursive`: an empty one goes, a populated one is
            // refused.
            pinned_ops::remove_at(&dir, &name, recursive).map_err(|e| rm_failure(&label, &e))
        })
        .await
        .map_err(|e| {
            SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                format!("rm task failed: {e}"),
            )
        })?
    }

    /// Create a directory natively.
    ///
    /// Mirrors Python's `_FileOps.mkdir`: an existing directory is success, but a symlink in the
    /// leaf position is not, because accepting it would let a swapped-in link stand in for the
    /// directory the caller asked for.
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        let host = self.host_path_for(path, true).await?;
        // A directory or parent swapped for a link after validation must not be created through.
        self.run_after_validate(&host);
        let host = self.host_path_for(path, true).await?;
        let host = self.resolve_for_create(host, path, true).await?;
        let failed = |e: &dyn std::fmt::Display| {
            SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                format!("Failed to create directory `{path}`: {e}"),
            )
            .with_context("path", path.to_string())
        };
        // A link that already resolves to an authorized directory (the workspace root spelled
        // through an alias, say) is that directory; `create_dir_all` has always accepted it.
        if parents
            && tokio::fs::symlink_metadata(&host)
                .await
                .is_ok_and(|metadata| metadata.file_type().is_symlink())
            && let Ok(real) = tokio::fs::canonicalize(&host).await
            && tokio::fs::metadata(&real).await.is_ok_and(|m| m.is_dir())
            && self
                .authorize_real(host.clone(), real, path, true)
                .await
                .is_ok()
        {
            return Ok(());
        }
        // The deepest ancestor that exists is resolved now, must still be authorized and is
        // pinned; the missing components are created below it one at a time, each opened
        // without following a link before the next is made.
        let mut base = host.parent().map(Path::to_path_buf).unwrap_or_default();
        let mut rest: Vec<std::ffi::OsString> = Vec::new();
        if let Some(leaf) = host.file_name() {
            rest.push(leaf.to_os_string());
        }
        let real_base = loop {
            match tokio::fs::canonicalize(&base).await {
                Ok(real) => break real,
                Err(_) => match (base.file_name(), base.parent()) {
                    (Some(name), Some(parent)) => {
                        rest.insert(0, name.to_os_string());
                        base = parent.to_path_buf();
                    }
                    _ => return Err(failed(&"no existing ancestor")),
                },
            }
        };
        let mut intended = real_base.clone();
        intended.extend(&rest);
        self.authorize_real(host.clone(), intended, path, true)
            .await
            .map_err(|_| {
                SandboxError::new(
                    ErrorCode::InvalidManifestPath,
                    format!("`{path}` changed while it was being validated"),
                )
                .with_context("path", path.to_string())
            })?;
        if !parents && rest.len() > 1 {
            return Err(failed(&"No such file or directory"));
        }
        tokio::task::spawn_blocking(move || {
            let dir = pinned_ops::open_dir_pinned(&real_base)?;
            pinned_ops::mkdir_below(&dir, &rest, parents)
        })
        .await
        .map_err(|e| failed(&e))?
        .map_err(|e| failed(&e))
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let host = self.host_path_for(path, true).await?;
        let host = self.resolve_for_create(host, path, true).await?;
        reject_special_file(&host, path, true).await?;
        if let Some(parent) = host.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("Failed to create `{}`: {e}", parent.display()),
                )
            })?;
        }
        self.run_after_validate(&host);
        let failed = |e: std::io::Error| {
            SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to write `{path}`: {e}"),
            )
        };
        use tokio::io::AsyncWriteExt as _;
        let existed = tokio::fs::symlink_metadata(&host).await.is_ok();
        // Opened without truncation: nothing of the existing file is lost until the handle has
        // been shown to be the workspace file.
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(O_NONBLOCK)
            .open(&host)
            .await
            .map_err(failed)?;
        let handle = file.metadata().await.map_err(failed)?;
        if !handle.is_file() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                format!("`{path}` is not a regular file"),
            ));
        }
        if let Err(error) = self.verify_opened(&handle, &host, path, true).await {
            if !existed {
                self.discard_created(&handle, &host).await;
            }
            return Err(error);
        }
        file.set_len(0).await.map_err(failed)?;
        file.write_all(contents).await.map_err(failed)?;
        file.flush().await.map_err(failed)
    }

    fn session_state(&self) -> SandboxSessionState {
        SandboxSessionState::new(
            "unix_local",
            &self.id,
            self.workspace_root(),
            &self.manifest,
        )
        .with_workspace_root_owned(self.owned_root)
        .with_workspace_root_ready(self.materialized.load(Ordering::SeqCst))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
        .with_snapshot_fingerprint(self.resume_snapshot_fingerprint())
        .with_exposed_ports(self.exposed_ports.clone())
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        // A workspace that was never fully materialized is not the one the fingerprint
        // describes, so the fast path must not consider it. Mirrors Python's
        // `_can_reuse_preserved_workspace_on_resume`.
        if !self.materialized.load(Ordering::SeqCst) {
            return None;
        }
        self.resume_fingerprint
            .lock()
            .expect("fingerprint lock")
            .clone()
    }

    fn record_snapshot_fingerprint(&self, record: Option<SnapshotFingerprint>) {
        *self.resume_fingerprint.lock().expect("fingerprint lock") = record;
    }

    fn archive_limits(&self) -> Option<SandboxArchiveLimits> {
        *self.archive_limits.lock().expect("archive limits lock")
    }

    fn set_archive_limits(&self, limits: Option<SandboxArchiveLimits>) {
        *self.archive_limits.lock().expect("archive limits lock") = limits;
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        validate_tar_bytes(archive, &TarValidationOptions::hydrate())
    }

    async fn persist_workspace_once(&self) -> SandboxResult<Vec<u8>> {
        let raw = capture_workspace_tar(self).await?;
        // A workspace link recorded as `<root>/lib/thing` names a path that no longer exists once
        // the snapshot is restored under a different root, and the hydrate policy refuses an
        // absolute target anyway. Python rewrites those targets while it captures the archive
        // (`unix_local._archive_workspace`); so does this.
        let roots = self.rebase_roots().await;
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || {
            let normalized = normalize_restorable_archive(&raw, &root)?;
            rebase_archive_symlink_targets(&normalized, &roots)
        })
        .await
        .map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!("Failed to archive workspace `{}`: {e}", self.root.display()),
            )
        })?
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        let root = self.root.clone();
        let raw = archive.to_vec();
        // The archive is unpacked by the SDK, not by whichever `tar` the host happens to ship:
        // every member is validated first, and traversal, hardlink and device members are
        // refused before a byte is written.
        tokio::task::spawn_blocking(move || {
            extract_tar_to(&root, &raw, &TarValidationOptions::hydrate())
        })
        .await
        .map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to restore workspace `{}`: {e}", self.root.display()),
            )
        })??;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::entries::Entry;

    /// Build a workspace holding `links` (link path → target) beside `lib/real.so`.
    async fn session_with_links(
        client: &UnixLocalSandboxClient,
        links: &[(&str, &str)],
    ) -> std::sync::Arc<dyn SandboxSession> {
        let session = client.create(None).await.unwrap();
        session.start().await.unwrap();
        let root = PathBuf::from(session.workspace_root());
        tokio::fs::create_dir_all(root.join("bin")).await.unwrap();
        tokio::fs::create_dir_all(root.join("lib")).await.unwrap();
        tokio::fs::write(root.join("lib/real.so"), b"ok")
            .await
            .unwrap();
        let canonical_root = tokio::fs::canonicalize(&root).await.unwrap();
        for (link, target) in links {
            let target = target.replace("<root>", &canonical_root.to_string_lossy());
            std::os::unix::fs::symlink(&target, root.join(link)).unwrap();
        }
        session
    }

    /// The link target the archive records for `name`.
    fn archived_link_target(archive: &[u8], name: &str) -> String {
        crate::sandbox::archive::read_tar_members(archive)
            .unwrap()
            .into_iter()
            .find(|member| member.name.trim_start_matches("./") == name)
            .unwrap_or_else(|| panic!("`{name}` is archived"))
            .link_target
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn persist_rebases_a_link_into_the_workspace_so_a_restore_follows_the_new_root() {
        let client = UnixLocalSandboxClient::new();
        let source = session_with_links(
            &client,
            &[
                ("bin/inside", "<root>/lib/real.so"),
                ("bin/relative", "../lib/real.so"),
            ],
        )
        .await;
        let archive = source.persist_workspace().await.unwrap();

        // Captured relative to the link's own directory, as Python's `_archive_workspace` does.
        assert_eq!(
            archived_link_target(&archive, "bin/inside"),
            "../lib/real.so"
        );
        assert_eq!(
            archived_link_target(&archive, "bin/relative"),
            "../lib/real.so"
        );

        // Restored under a different root, the link still resolves to the archived file.
        let target = client.create(None).await.unwrap();
        target.start().await.unwrap();
        target.hydrate_workspace(&archive).await.unwrap();
        let root = PathBuf::from(target.workspace_root());
        assert_eq!(
            tokio::fs::read_link(root.join("bin/inside")).await.unwrap(),
            PathBuf::from("../lib/real.so")
        );
        assert_eq!(
            tokio::fs::read(root.join("bin/inside")).await.unwrap(),
            b"ok"
        );

        client.delete(source).await.unwrap();
        client.delete(target).await.unwrap();
    }

    #[tokio::test]
    async fn a_link_out_of_the_workspace_stays_absolute_and_is_refused_on_restore() {
        let client = UnixLocalSandboxClient::new();
        let source = session_with_links(&client, &[("bin/python", "/usr/bin/env")]).await;
        let archive = source.persist_workspace().await.unwrap();

        // Only targets under the workspace are rewritten; a virtualenv-style link out keeps its
        // absolute form, which the hydrate policy then refuses, exactly as Python's does.
        assert_eq!(archived_link_target(&archive, "bin/python"), "/usr/bin/env");
        let target = client.create(None).await.unwrap();
        target.start().await.unwrap();
        let error = target.hydrate_workspace(&archive).await.unwrap_err();
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("absolute symlink target not allowed: /usr/bin/env")
        );
        assert!(
            !PathBuf::from(target.workspace_root())
                .join("lib/real.so")
                .exists()
        );

        client.delete(source).await.unwrap();
        client.delete(target).await.unwrap();
    }

    #[tokio::test]
    async fn starts_materializes_and_execs() {
        let client = UnixLocalSandboxClient::new();
        let manifest = Manifest::new()
            .with_entry(
                "repo",
                Entry::dir().with_child("task.md", Entry::text_file("fix it")),
            )
            .with_env("SANDBOX_MARKER", "yes");
        let session = client.create(Some(manifest)).await.unwrap();
        session.start().await.unwrap();
        assert!(session.running().await.unwrap());
        assert_ne!(session.workspace_root(), DEFAULT_MANIFEST_ROOT);

        let content = session.read_file("repo/task.md").await.unwrap();
        assert_eq!(String::from_utf8(content).unwrap(), "fix it");

        let result = session
            .exec(ExecRequest::shell("cat repo/task.md; echo $SANDBOX_MARKER"))
            .await
            .unwrap();
        assert!(result.ok());
        assert!(result.stdout_text().contains("fix it"));
        assert!(result.stdout_text().contains("yes"));

        let listing = session.ls("repo").await.unwrap();
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].name, "task.md");

        let root = PathBuf::from(session.workspace_root());
        client.delete(session).await.unwrap();
        assert!(!root.exists());
    }

    #[tokio::test]
    async fn rejects_path_traversal_and_symlink_escapes() {
        let client = UnixLocalSandboxClient::new();
        let session = client.create(None).await.unwrap();
        session.start().await.unwrap();
        for bad in ["../outside.txt", "/etc/passwd", "a/../../b"] {
            assert!(session.read_file(bad).await.is_err(), "{bad}");
            assert!(session.write_file(bad, b"x").await.is_err(), "{bad}");
        }

        // A symlink planted inside the workspace must not become a read/write channel.
        session
            .exec(ExecRequest::shell("ln -s /etc escaped"))
            .await
            .unwrap();
        let err = session.read_file("escaped/passwd").await.unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidManifestPath);
        assert!(session.write_file("escaped/x", b"x").await.is_err());

        client.delete(session).await.unwrap();
    }

    #[tokio::test]
    async fn exec_honors_timeouts() {
        let client = UnixLocalSandboxClient::new();
        let session = client.create(None).await.unwrap();
        session.start().await.unwrap();
        let err = session
            .exec(
                ExecRequest::shell("sleep 5")
                    .with_timeout(Some(std::time::Duration::from_millis(50))),
            )
            .await
            .unwrap_err();
        assert!(err.is_timeout());
        client.delete(session).await.unwrap();
    }

    #[tokio::test]
    async fn persists_and_hydrates_the_workspace() {
        let client = UnixLocalSandboxClient::new();
        let source = client
            .create(Some(
                Manifest::new().with_entry("a.txt", Entry::text_file("one")),
            ))
            .await
            .unwrap();
        source.start().await.unwrap();
        let archive = source.persist_workspace().await.unwrap();
        assert!(!archive.is_empty());

        let target = client.create(None).await.unwrap();
        target.start().await.unwrap();
        target.hydrate_workspace(&archive).await.unwrap();
        let content = target.read_file("a.txt").await.unwrap();
        assert_eq!(String::from_utf8(content).unwrap(), "one");

        client.delete(source).await.unwrap();
        client.delete(target).await.unwrap();
    }
}
