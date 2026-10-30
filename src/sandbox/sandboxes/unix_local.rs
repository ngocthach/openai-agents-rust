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

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, is_sdk_owned_workspace_root};
use crate::sandbox::types::ExecResult;

/// Prefix of the temporary workspace directories created for Unix-local sessions.
const WORKSPACE_PREFIX: &str = "openai-agents-sandbox-";

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
    snapshot: Option<SnapshotSpec>,
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
            snapshot: None,
        }
    }

    /// Whether sandbox commands inherit the host process environment (default: true).
    pub fn with_inherit_host_environment(mut self, inherit: bool) -> Self {
        self.inherit_host_environment = inherit;
        self
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }
}

#[async_trait]
impl SandboxClient for UnixLocalSandboxClient {
    fn backend_id(&self) -> &str {
        "unix_local"
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        let manifest = manifest.unwrap_or_default();
        // A runner-created local session always gets a dedicated temp root unless the caller
        // chose a custom workspace root explicitly.
        let (manifest, owned_root) = if manifest.root() == DEFAULT_MANIFEST_ROOT {
            let dir = create_temp_workspace().await?;
            (manifest.with_root(path_to_posix(&dir)), true)
        } else {
            (manifest, false)
        };
        let session = UnixLocalSandboxSession::new(manifest, owned_root)?
            .with_inherit_host_environment(self.inherit_host_environment)
            .with_snapshot(self.snapshot.clone());
        Ok(Arc::new(session))
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
        state.verify_manifest(&manifest)?;
        // The state carries no manifest contents of its own: the workspace root comes from it,
        // but everything that could read host paths comes from the caller's trusted manifest.
        let manifest = manifest.with_root(state.workspace_root.clone());
        let session = UnixLocalSandboxSession::new(manifest, state.may_remove_workspace_root())?
            .with_inherit_host_environment(self.inherit_host_environment)
            .with_snapshot(Some(state.snapshot.clone()))
            .with_identity(state.session_id.clone(), state.snapshot_id.clone())
            .resuming()
            .with_workspace_root_ready(state.workspace_root_ready);
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
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
    root: PathBuf,
    owned_root: bool,
    inherit_host_environment: bool,
    host_environment_allowlist: Option<Vec<String>>,
    /// The manifest's start-up environment lookups, performed once when the session starts.
    resolved_environment: std::sync::OnceLock<std::collections::BTreeMap<String, String>>,
    /// Whether the manifest finished materializing, so a resume can trust the workspace.
    materialized: AtomicBool,
    /// Keeps concurrent work out of the workspace while a snapshot detaches its mounts.
    mount_transition: crate::sandbox::mounts::MountTransition,
    running: AtomicBool,
    ptys: PtyManager,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
}

impl UnixLocalSandboxSession {
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
        Ok(UnixLocalSandboxSession {
            snapshot_id: id.clone(),
            id,
            manifest,
            root,
            owned_root,
            inherit_host_environment: true,
            host_environment_allowlist: None,
            resolved_environment: std::sync::OnceLock::new(),
            materialized: AtomicBool::new(false),
            mount_transition: crate::sandbox::mounts::MountTransition::new(),
            running: AtomicBool::new(false),
            ptys: PtyManager::new(),
            snapshot: None,
            resuming: false,
        })
    }

    /// Store workspace snapshots at `snapshot`.
    pub fn with_snapshot(mut self, snapshot: Option<SnapshotSpec>) -> Self {
        self.snapshot = snapshot;
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

    /// Resolve a model-provided path for reading or writing.
    ///
    /// Writes through a read-only mount are rejected.
    pub async fn host_path_for(&self, path: &str, for_write: bool) -> SandboxResult<PathBuf> {
        let resolved = self.normalize_path(path)?;
        let candidate = PathBuf::from(&resolved);
        let real = resolve_existing_prefix(&candidate).await;
        // The root is canonicalized too: a caller-supplied root can contain symlinks (on macOS
        // `/var/folders/...` resolves to `/private/var/folders/...`), and comparing a resolved
        // candidate against an unresolved root would reject every path inside the workspace.
        let real_root = resolve_existing_prefix(&self.root).await;
        if real.starts_with(&real_root) {
            return Ok(candidate);
        }
        // A bind mount deliberately points outside the workspace, so its source is an explicit
        // grant. Writes through a read-only grant are still refused.
        for grant in self.manifest.host_path_grants() {
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

    fn backend_id(&self) -> &str {
        "unix_local"
    }

    async fn start(&self) -> SandboxResult<()> {
        tokio::fs::create_dir_all(&self.root).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!("Failed to create workspace `{}`: {e}", self.root.display()),
            )
        })?;
        self.running.store(true, Ordering::SeqCst);
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
        self.ptys.terminate_all();
        self.manifest.unmount_all(self).await?;
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.ptys.terminate_all();
        self.running.store(false, Ordering::SeqCst);
        // Detach every mount before touching the workspace: an in-container mount materializes a
        // real filesystem inside the root, and removing the root over a live mount would delete
        // the mounted storage, not just the workspace.
        self.manifest.unmount_all(self).await?;
        if self.owned_root && is_sdk_owned_workspace_root(&self.root) {
            match tokio::fs::remove_dir_all(&self.root).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
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

    async fn running(&self) -> SandboxResult<bool> {
        Ok(self.running.load(Ordering::SeqCst))
    }

    fn resolves_environment(&self) -> bool {
        true
    }

    fn mount_transition(&self) -> Option<&crate::sandbox::mounts::MountTransition> {
        Some(&self.mount_transition)
    }

    fn supports_pty(&self) -> bool {
        true
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let cwd = match &request.cwd {
            None => self.root.clone(),
            Some(cwd) => self.host_path_for(cwd, false).await?,
        };
        let spawn = PtySpawn {
            argv: request.resolved_argv(),
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
        let mut command = tokio::process::Command::new(program);
        command
            .args(args)
            .current_dir(&cwd)
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
        let host = self.host_path_for(path, false).await?;
        tokio::fs::read(&host).await.map_err(|e| {
            let code = if e.kind() == std::io::ErrorKind::NotFound {
                ErrorCode::WorkspaceReadNotFound
            } else {
                ErrorCode::WorkspaceWriteTypeError
            };
            SandboxError::new(code, format!("Failed to read `{path}`: {e}"))
        })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let host = self.host_path_for(path, true).await?;
        if let Some(parent) = host.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("Failed to create `{}`: {e}", parent.display()),
                )
            })?;
        }
        tokio::fs::write(&host, contents).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to write `{path}`: {e}"),
            )
        })
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
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        tokio::fs::create_dir_all(&self.root).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to create workspace `{}`: {e}", self.root.display()),
            )
        })?;
        extract_tar_gz(&self.root, archive).await
    }
}

/// Extract a gzipped tar archive into `root` with the `tar` CLI.
pub(crate) async fn extract_tar_gz(root: &Path, archive: &[u8]) -> SandboxResult<()> {
    use tokio::io::AsyncWriteExt;

    let mut child = tokio::process::Command::new("tar")
        .arg("-xzf")
        .arg("-")
        .arg("-C")
        .arg(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to run `tar`: {e}"),
            )
        })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(archive).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to stream the archive into `tar`: {e}"),
            )
        })?;
        stdin.shutdown().await.ok();
    }
    let output = child.wait_with_output().await.map_err(|e| {
        SandboxError::new(
            ErrorCode::SnapshotRestoreError,
            format!("Failed to wait for `tar`: {e}"),
        )
    })?;
    // The rebase runs whether or not `tar` succeeded: a partial restore still holds links that
    // have to follow the workspace to its new root.
    rebase_absolute_symlinks(root).await;
    if !output.status.success() {
        return Err(SandboxError::new(
            ErrorCode::SnapshotRestoreError,
            format!(
                "`tar` failed to restore the workspace: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    Ok(())
}

/// Rewrite absolute symlink targets that point inside the workspace as link-relative ones.
///
/// Mirrors Python's `_rebase_symlink_target`. A snapshot can be restored under a different root
/// than the one it was taken at — a new temporary directory, a different container — and an
/// absolute link recorded as `<old root>/lib/thing` then points at a path that no longer exists.
/// Rewriting it relative to the link's own directory makes it follow the workspace wherever it
/// lands.
///
/// Targets outside the workspace are left exactly as they are, as Python leaves them: a
/// workspace legitimately contains links out, such as a virtualenv's
/// `bin/python -> /usr/bin/python3`, and deleting or rewriting those would break the very tools
/// the agent restored the workspace to use. Containment is enforced where a path is *used*
/// rather than here — `host_path_for` canonicalizes both the candidate and the root before every
/// file operation, and `tar` itself refuses to extract through a symlink.
async fn rebase_absolute_symlinks(root: &Path) {
    let Ok(canonical_root) = tokio::fs::canonicalize(root).await else {
        return;
    };
    let mut pending = vec![canonical_root.clone()];
    while let Some(directory) = pending.pop() {
        let Ok(mut read_dir) = tokio::fs::read_dir(&directory).await else {
            continue;
        };
        while let Ok(Some(entry)) = read_dir.next_entry().await {
            let path = entry.path();
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if !file_type.is_symlink() {
                continue;
            }
            let Ok(target) = tokio::fs::read_link(&path).await else {
                continue;
            };
            let Some(rebased) = rebase_symlink_target(&canonical_root, &path, &target) else {
                continue;
            };
            // Replacing a link is two steps; if the second fails the link is gone, so the
            // original is put back rather than left missing.
            if tokio::fs::remove_file(&path).await.is_ok()
                && let Err(error) = tokio::fs::symlink(&rebased, &path).await
            {
                log::warn!(
                    "failed to rebase the symlink `{}` onto the restored workspace: {error}",
                    path.display()
                );
                let _ = tokio::fs::symlink(&target, &path).await;
            }
        }
    }
}

/// The link-relative form of `target`, when it is absolute and points inside `root`.
///
/// Returns `None` when there is nothing to do: a relative target already follows the workspace,
/// and an absolute one pointing outside it is the caller's own reference to the host.
fn rebase_symlink_target(root: &Path, link: &Path, target: &Path) -> Option<PathBuf> {
    if !target.is_absolute() {
        return None;
    }
    let rest = target.strip_prefix(root).ok()?;
    // Climb out of the link's own directory, then descend the part after the root. The
    // components after the root are kept verbatim: `..` following a symlink component is
    // resolved by the kernel against that link's target, so normalizing here would change
    // where the link points.
    let depth = link
        .parent()
        .and_then(|parent| parent.strip_prefix(root).ok())
        .map(|relative| relative.components().count())
        .unwrap_or(0);
    let mut rebased = PathBuf::new();
    for _ in 0..depth {
        rebased.push("..");
    }
    rebased.push(rest);
    if rebased.as_os_str().is_empty() {
        rebased.push(".");
    }
    Some(rebased)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::entries::Entry;

    #[tokio::test]
    async fn a_restored_workspace_keeps_its_links_and_follows_its_new_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("workspace");
        tokio::fs::create_dir_all(root.join("bin")).await.unwrap();
        tokio::fs::create_dir_all(root.join("lib")).await.unwrap();
        tokio::fs::write(root.join("lib/real.so"), b"ok")
            .await
            .unwrap();
        let canonical_root = tokio::fs::canonicalize(&root).await.unwrap();

        // An absolute link into the workspace: this is the one that breaks when a snapshot is
        // restored under a different root, so it is rewritten to follow the workspace.
        std::os::unix::fs::symlink(canonical_root.join("lib/real.so"), root.join("bin/inside"))
            .unwrap();
        // A link out of the workspace, as a virtualenv interpreter is. The agent restored the
        // workspace to use it, so it stays exactly as it was.
        std::os::unix::fs::symlink("/usr/bin/env", root.join("bin/python")).unwrap();
        // A relative link already follows the workspace.
        std::os::unix::fs::symlink("../lib/real.so", root.join("bin/relative")).unwrap();

        rebase_absolute_symlinks(&root).await;

        assert_eq!(
            tokio::fs::read_link(root.join("bin/inside")).await.unwrap(),
            PathBuf::from("../lib/real.so")
        );
        assert_eq!(
            tokio::fs::read_link(root.join("bin/python")).await.unwrap(),
            PathBuf::from("/usr/bin/env")
        );
        assert_eq!(
            tokio::fs::read_link(root.join("bin/relative"))
                .await
                .unwrap(),
            PathBuf::from("../lib/real.so")
        );
        // The rewritten link still resolves to the same file.
        assert_eq!(
            tokio::fs::read(root.join("bin/inside")).await.unwrap(),
            b"ok"
        );
    }

    #[test]
    fn only_absolute_in_workspace_targets_are_rebased() {
        let root = Path::new("/w");
        assert_eq!(
            rebase_symlink_target(root, Path::new("/w/bin/x"), Path::new("/w/lib/y")),
            Some(PathBuf::from("../lib/y"))
        );
        // At the root there is nothing to climb out of.
        assert_eq!(
            rebase_symlink_target(root, Path::new("/w/x"), Path::new("/w/lib/y")),
            Some(PathBuf::from("lib/y"))
        );
        // Outside the workspace, and already relative: left alone.
        assert_eq!(
            rebase_symlink_target(root, Path::new("/w/bin/x"), Path::new("/usr/bin/env")),
            None
        );
        assert_eq!(
            rebase_symlink_target(root, Path::new("/w/bin/x"), Path::new("../lib/y")),
            None
        );
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
