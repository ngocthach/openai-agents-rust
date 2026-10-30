//! The sandbox session and client interfaces. Mirrors
//! `agents.sandbox.session.base_sandbox_session` and `agents.sandbox.session.sandbox_client`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mounts::{Mount, MountTransition};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::runtime_helpers::{RuntimeHelperScript, WORKSPACE_FINGERPRINT_HELPER};
use crate::sandbox::snapshot::{Snapshot, resolve_snapshot};
use crate::sandbox::state::{
    SNAPSHOT_FINGERPRINT_VERSION, SNAPSHOT_FINGERPRINT_VERSION_V1, SandboxSessionState,
    SnapshotFingerprint,
};
use crate::sandbox::types::{
    ExecResult, ExposedPortEndpoint, ExposedPortUnavailable, FileEntry, User,
    exposed_port_unavailable, parse_ls_la,
};
use crate::sandbox::workspace_paths::{WorkspacePathPolicy, is_provider_scratch_path};

/// How a command is handed to the sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecShell {
    /// Run the joined command through `sh -lc` (the Python default).
    Login,
    /// Run the joined command through `sh -c`.
    Plain,
    /// Run the command words directly, without a shell.
    None,
    /// Run the joined command through a custom shell prefix, for example `["bash", "-lc"]`.
    Custom(Vec<String>),
}

impl ExecShell {
    /// The shell prefix, or `None` when the command runs directly.
    pub fn prefix(&self) -> Option<Vec<String>> {
        match self {
            ExecShell::Login => Some(vec!["sh".into(), "-lc".into()]),
            ExecShell::Plain => Some(vec!["sh".into(), "-c".into()]),
            ExecShell::None => None,
            ExecShell::Custom(prefix) => Some(prefix.clone()),
        }
    }
}

/// One command to run inside a sandbox session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecRequest {
    /// The command and its arguments.
    pub command: Vec<String>,
    /// How the command reaches the sandbox.
    pub shell: ExecShell,
    /// Optional wall-clock timeout.
    pub timeout: Option<Duration>,
    /// Workspace-relative or absolute working directory.
    pub cwd: Option<String>,
    /// Sandbox user to run as (`sudo -u <name> --`).
    pub user: Option<User>,
    /// Environment variables for this command, applied on top of the manifest environment.
    ///
    /// A backend passes them the way its transport does: as `docker exec -e`, as the process
    /// environment of a local command, or as the provider's own per-command environment. A
    /// backend that does not read them yet ignores them, so a command that depends on one should
    /// check that its backend honors them.
    pub env: BTreeMap<String, String>,
}

impl ExecRequest {
    /// A shell command string, run through `sh -lc`.
    pub fn shell(command: impl Into<String>) -> Self {
        ExecRequest {
            command: vec![command.into()],
            shell: ExecShell::Login,
            timeout: None,
            cwd: None,
            user: None,
            env: BTreeMap::new(),
        }
    }

    /// A command run directly, without a shell.
    pub fn argv<I, S>(command: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ExecRequest {
            command: command.into_iter().map(Into::into).collect(),
            shell: ExecShell::None,
            timeout: None,
            cwd: None,
            user: None,
            env: BTreeMap::new(),
        }
    }

    /// Set the shell handling.
    pub fn with_shell(mut self, shell: ExecShell) -> Self {
        self.shell = shell;
        self
    }

    /// Set the wall-clock timeout.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the working directory.
    pub fn with_cwd(mut self, cwd: Option<String>) -> Self {
        self.cwd = cwd;
        self
    }

    /// Set the sandbox user.
    pub fn with_user(mut self, user: Option<User>) -> Self {
        self.user = user;
        self
    }

    /// Set one environment variable for this command.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Set several environment variables for this command.
    pub fn with_envs<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env
            .extend(env.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// The environment this command runs with: the manifest's, then this request's.
    ///
    /// Every backend resolves it the same way, so a per-command variable always wins over the
    /// manifest value of the same name.
    pub fn resolved_env(
        &self,
        manifest_env: &BTreeMap<String, String>,
    ) -> BTreeMap<String, String> {
        let mut resolved = manifest_env.clone();
        resolved.extend(self.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        resolved
    }

    /// The final argument vector, after applying the shell prefix and the `sudo` wrapper.
    pub fn resolved_argv(&self) -> Vec<String> {
        let mut argv = match self.shell.prefix() {
            None => self.command.clone(),
            Some(prefix) => {
                let joined = if self.command.len() == 1 {
                    self.command[0].clone()
                } else {
                    self.command
                        .iter()
                        .map(|c| shell_quote(c))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let mut argv = prefix;
                argv.push(joined);
                argv
            }
        };
        if let Some(user) = &self.user {
            let mut sudo = vec![
                "sudo".to_string(),
                "-u".to_string(),
                user.name.clone(),
                "--".to_string(),
            ];
            sudo.append(&mut argv);
            argv = sudo;
        }
        argv
    }
}

/// Quote a single argument for POSIX shells.
pub fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'_' | b'-' | b'.' | b'/' | b':' | b'=' | b'@' | b'+' | b','
                )
        })
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// A live sandbox workspace: a place to run commands and read and write files.
///
/// Implementations are shared behind an `Arc` and must be safe to use concurrently. All paths
/// accepted by these methods are workspace-relative or absolute inside
/// [`manifest().root`](Manifest::root); anything else is rejected with
/// [`ErrorCode::InvalidManifestPath`].
#[async_trait]
pub trait SandboxSession: Send + Sync + fmt::Debug {
    /// A stable identifier for the session.
    fn id(&self) -> &str;

    /// The manifest materialized into this workspace.
    fn manifest(&self) -> &Manifest;

    /// The absolute workspace root inside the sandbox.
    fn workspace_root(&self) -> &str {
        self.manifest().root()
    }

    /// The backend that created the session (`unix_local`, `docker`, ...).
    fn backend_id(&self) -> &str;

    /// Start the sandbox and materialize the manifest.
    async fn start(&self) -> SandboxResult<()>;

    /// Stop the sandbox, keeping any backend resources that `shutdown` releases.
    async fn stop(&self) -> SandboxResult<()>;

    /// Release the backend resources behind the session.
    async fn shutdown(&self) -> SandboxResult<()>;

    /// Whether the sandbox is currently running.
    async fn running(&self) -> SandboxResult<bool>;

    /// Run a command inside the sandbox.
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult>;

    /// Read a file from the workspace.
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>>;

    /// Write a file to the workspace, creating parent directories.
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()>;

    /// Read a file as `user`.
    ///
    /// A sandbox agent that runs its commands as an unprivileged user should read files as that
    /// user too: reading as the session owner can succeed where the agent's own shell commands
    /// would fail, which hides a permission problem until the model trips over it. The default
    /// runs `cat` through [`exec`](Self::exec), which already applies the `sudo` wrapper, so
    /// every backend gets this without implementing it.
    async fn read_file_as(&self, path: &str, user: Option<&User>) -> SandboxResult<Vec<u8>> {
        let Some(user) = user else {
            return self.read_file(path).await;
        };
        let path = self.normalize_path(path)?;
        let result = self
            .exec(ExecRequest::argv(["cat", "--", path.as_str()]).with_user(Some(user.clone())))
            .await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!(
                    "`{}` could not read `{path}` as `{}`: {}",
                    self.backend_id(),
                    user.name,
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(result.stdout)
    }

    /// Write a file as `user`, so the file ends up owned by them.
    ///
    /// The contents travel base64-encoded, so arbitrary binary content and shell metacharacters
    /// are safe. They are also written in chunks: a single argument is capped at 128 KiB on
    /// Linux, so one big `printf` would fail with `E2BIG` on any file over about 96 KB. The
    /// chunks land in a sibling temporary file that is moved into place, so a failed write never
    /// leaves a half-written file where the old one was.
    async fn write_file_as(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        /// Bytes of base64 per command, comfortably inside the single-argument limit.
        const CHUNK_BYTES: usize = 48 * 1024;

        let Some(user) = user else {
            return self.write_file(path, contents).await;
        };
        let path = self.normalize_path_for_write(path)?;
        let encoded = {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(contents)
        };
        let staging = format!("{path}.openai-agents-write");
        let quoted_path = shell_quote(&path);
        let quoted_staging = shell_quote(&staging);

        let run = async |command: String| -> SandboxResult<ExecResult> {
            self.exec(
                ExecRequest::argv(["sh", "-lc", command.as_str()])
                    .with_shell(ExecShell::None)
                    .with_user(Some(user.clone())),
            )
            .await
        };
        let failed = |action: &str, result: &ExecResult| {
            SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!(
                    "`{}` could not {action} `{path}` as `{}`: {}",
                    self.backend_id(),
                    user.name,
                    result.stderr_text().trim()
                ),
            )
        };

        // `write_file` creates parent directories, so the run-as form has to as well.
        let prepared = run(format!(
            "mkdir -p -- \"$(dirname {quoted_path})\" && : > {quoted_staging}"
        ))
        .await?;
        if !prepared.ok() {
            // Say *why* the path cannot be written as this user, rather than only that it
            // could not be: a missing parent and an unwritable one look identical otherwise.
            self.check_write_access(&path, Some(user)).await?;
            return Err(failed("prepare", &prepared));
        }

        let chunks: Vec<&str> = if encoded.is_empty() {
            Vec::new()
        } else {
            encoded
                .as_bytes()
                .chunks(CHUNK_BYTES)
                .map(|chunk| {
                    // base64 is ASCII, so every chunk is a valid string.
                    std::str::from_utf8(chunk).unwrap_or_default()
                })
                .collect()
        };
        for chunk in chunks {
            let appended = run(format!(
                "printf %s {} | base64 -d >> {quoted_staging}",
                shell_quote(chunk)
            ))
            .await?;
            if !appended.ok() {
                let _ = run(format!("rm -f -- {quoted_staging}")).await;
                return Err(failed("write", &appended));
            }
        }

        let moved = run(format!("mv -- {quoted_staging} {quoted_path}")).await?;
        if !moved.ok() {
            let _ = run(format!("rm -f -- {quoted_staging}")).await;
            return Err(failed("write", &moved));
        }
        Ok(())
    }

    /// Create a directory as `user`, so it ends up owned by them.
    async fn mkdir_as(&self, path: &str, parents: bool, user: Option<&User>) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.mkdir(path, parents).await;
        };
        let path = self.normalize_path_for_write(path)?;
        let mut argv = vec!["mkdir".to_string()];
        if parents {
            argv.push("-p".to_string());
        }
        argv.extend(["--".to_string(), path.clone()]);
        let result = self
            .exec(ExecRequest::argv(argv.clone()).with_user(Some(user.clone())))
            .await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&argv, &result));
        }
        Ok(())
    }

    /// Remove a path as `user`.
    async fn rm_as(&self, path: &str, recursive: bool, user: Option<&User>) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.rm(path, recursive).await;
        };
        let policy = self.workspace_path_policy()?;
        if recursive {
            policy.validate_recursive_remove(path)?;
        }
        let path = policy.normalize_sandbox_path(path, true)?;
        let mut argv = vec!["rm".to_string()];
        if recursive {
            argv.push("-r".to_string());
        }
        argv.extend(["-f".to_string(), "--".to_string(), path.clone()]);
        let result = self
            .exec(ExecRequest::argv(argv.clone()).with_user(Some(user.clone())))
            .await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&argv, &result));
        }
        Ok(())
    }

    /// Move a path as `user`.
    async fn mv_as(
        &self,
        source: &str,
        destination: &str,
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.mv(source, destination).await;
        };
        let source = self.normalize_path_for_write(source)?;
        let destination = self.normalize_path_for_write(destination)?;
        let argv = vec!["mv".to_string(), "--".to_string(), source, destination];
        let result = self
            .exec(ExecRequest::argv(argv.clone()).with_user(Some(user.clone())))
            .await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&argv, &result));
        }
        Ok(())
    }

    /// Whether the session can run commands in a PTY (`exec_command` / `write_stdin`).
    fn supports_pty(&self) -> bool {
        false
    }

    /// Whether this backend performs the manifest's start-up environment lookups.
    ///
    /// A backend that returns `false` would silently drop a variable declared with
    /// [`Manifest::with_env_entry`](crate::sandbox::Manifest::with_env_entry), leaving the
    /// sandbox without a secret the caller believes it has. Materialization refuses instead, so
    /// the gap is visible at start-up rather than as a confusing failure inside the sandbox.
    fn resolves_environment(&self) -> bool {
        false
    }

    /// The session's mount-transition tracker, when the backend has one.
    ///
    /// A backend that can snapshot a workspace with mounts attached returns one so concurrent
    /// work can be refused while the mounts are detached. Backends without it never detach
    /// mounts behind a running command.
    fn mount_transition(&self) -> Option<&MountTransition> {
        None
    }

    /// Fail when this session's mounts are mid-snapshot.
    ///
    /// Model-facing tools call this before touching the workspace: during a snapshot a mounted
    /// bucket is an empty directory, and a command that reads it reports the emptiness as fact.
    fn ensure_mounts_settled(&self) -> SandboxResult<()> {
        match self.mount_transition() {
            Some(transition) => transition.ensure_settled(),
            None => Ok(()),
        }
    }

    /// Whether the container runtime attaches mounts before the session starts, so that
    /// [`MountStrategy::docker_create_args`](crate::sandbox::mounts::MountStrategy::docker_create_args)
    /// replaces a live activation command.
    fn attaches_mounts_at_create(&self) -> bool {
        false
    }

    /// Check a mount against the manifest's credential-exposure policy before activating it.
    fn authorize_mount_activation(&self, mount: &Mount, mount_path: &str) -> SandboxResult<()> {
        self.manifest()
            .authorize_mount_activation(mount, mount_path)
    }

    /// Start a command in a PTY and collect its first output window.
    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let _ = request;
        Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "PTY execution is not supported by this sandbox session",
        ))
    }

    /// Write to (or poll) a running PTY session.
    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let _ = request;
        Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "PTY execution is not supported by this sandbox session",
        ))
    }

    /// Terminate every live PTY session.
    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        Ok(())
    }

    /// This session as [`Any`](std::any::Any), for code that needs the concrete backend type.
    ///
    /// A provider's mount strategy only ever receives a `&dyn SandboxSession`, yet it may need
    /// the session's own transport to attach storage. Returning `Some(self)` lets it recover the
    /// concrete session; the default `None` keeps every other backend working unchanged. A type
    /// that wraps another session must forward this to the session it holds, or the strategies
    /// that rely on it stop finding their transport.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Normalize a model-provided path to an absolute sandbox path inside the workspace root.
    ///
    /// A manifest can grant access to absolute paths outside the workspace. The grant is trusted
    /// application configuration, so a path under one is allowed through; a path under no grant
    /// is still refused.
    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        self.workspace_path_policy()?
            .normalize_sandbox_path(path, false)
    }

    /// Like [`normalize_path`](Self::normalize_path), refusing a read-only grant.
    ///
    /// Every write a backend performs goes through this, so a read-only
    /// [`SandboxPathGrant`](crate::sandbox::workspace_paths::SandboxPathGrant) is enforced by the
    /// SDK and not only by whatever the backend's own filesystem does.
    fn normalize_path_for_write(&self, path: &str) -> SandboxResult<String> {
        self.workspace_path_policy()?
            .normalize_sandbox_path(path, true)
    }

    /// The path policy for this session's workspace root and grants.
    fn workspace_path_policy(&self) -> SandboxResult<WorkspacePathPolicy> {
        WorkspacePathPolicy::with_grants(
            self.workspace_root(),
            self.manifest().extra_path_grants().to_vec(),
        )
    }

    /// Resolve a path the SDK itself placed, which may sit outside the workspace root.
    ///
    /// An SDK-owned scratch path (see
    /// [`provider_scratch_path`](crate::sandbox::workspace_paths::provider_scratch_path)) is
    /// returned unchanged, because a backend that stages a snapshot archive in `/tmp` must be
    /// able to read it back; anything else goes through [`normalize_path`](Self::normalize_path).
    /// Model-provided paths never reach this method.
    fn resolve_internal_path(&self, path: &str) -> SandboxResult<String> {
        if is_provider_scratch_path(path) {
            return Ok(path.to_string());
        }
        self.normalize_path(path)
    }

    /// List a directory.
    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        let resolved = self.normalize_path(path)?;
        let command = vec![
            "ls".to_string(),
            "-la".to_string(),
            "--".to_string(),
            resolved.clone(),
        ];
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(parse_ls_la(&result.stdout_text(), &resolved))
    }

    /// Create a directory.
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        let resolved = self.normalize_path_for_write(path)?;
        let mut command = vec!["mkdir".to_string()];
        if parents {
            command.push("-p".to_string());
        }
        command.push("--".to_string());
        command.push(resolved.clone());
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            // `mkdir` reports "cannot create directory" for a missing parent, an unwritable
            // parent and a path that is already a file alike. The probe separates them, so the
            // caller gets the reason rather than the command's exit code.
            self.check_mkdir_access(&resolved, parents, None).await?;
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    /// Remove a file or directory.
    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let policy = self.workspace_path_policy()?;
        let resolved = if recursive {
            // A recursive removal of an ancestor is a write to every grant under it.
            policy.validate_recursive_remove(path)?;
            policy.normalize_sandbox_path(path, true)?
        } else {
            policy.normalize_sandbox_path(path, true)?
        };
        let mut command = vec!["rm".to_string()];
        command.push(if recursive { "-rf" } else { "-f" }.to_string());
        command.push("--".to_string());
        command.push(resolved.clone());
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            self.check_rm_access(&resolved, recursive, None).await?;
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    /// Move or rename a workspace entry.
    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        // A move writes at both ends: it unlinks the source and creates the destination.
        let from = self.normalize_path_for_write(source)?;
        let to = self.normalize_path_for_write(destination)?;
        let command = vec![
            "mv".to_string(),
            "-f".to_string(),
            "--".to_string(),
            from,
            to,
        ];
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    /// Capture the workspace as a gzipped tar archive.
    ///
    /// The default implementation shells out to `tar`, which every supported backend provides.
    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        // Mirrors the `retry_async` Python puts on `DockerSandboxSession.persist_workspace`: a
        // transport that drops mid-archive is the one failure worth repeating, and losing a
        // suspend's workspace to it would lose the paused run's work.
        crate::sandbox::retry::retry_async(
            crate::sandbox::retry::RetryPolicy::new(),
            || self.persist_workspace_once(),
            |error, _attempt| crate::sandbox::retry::is_transient_sandbox_error(error),
        )
        .await
    }

    /// One attempt at capturing the workspace archive.
    async fn persist_workspace_once(&self) -> SandboxResult<Vec<u8>> {
        let root = self.workspace_root().to_string();
        let mut command = format!("tar -czf - -C {}", shell_quote(&root));
        for exclude in workspace_archive_excludes(self.manifest(), &root) {
            command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
        }
        command.push_str(" .");
        let result = self.exec(ExecRequest::shell(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "Failed to archive workspace `{root}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(result.stdout)
    }

    /// Restore a gzipped tar archive produced by [`persist_workspace`](Self::persist_workspace).
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()>;

    /// A short description of the workspace layout, for logs and prompts.
    fn describe(&self) -> String {
        self.manifest().describe()
    }

    /// The credential-free state a caller can persist to resume this session later.
    fn session_state(&self) -> SandboxSessionState {
        SandboxSessionState::new(
            self.backend_id(),
            self.id(),
            self.workspace_root(),
            self.manifest(),
        )
    }

    /// Where this session stores workspace archives.
    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(None, self.id())
    }

    /// Install a helper script inside the sandbox and return its path there.
    ///
    /// Mirrors Python's `_ensure_runtime_helper_installed`. A helper is installed at a path
    /// derived from its own hash, so an already-present copy is reused and a new SDK version
    /// installs its script beside the old one instead of over it.
    async fn ensure_runtime_helper_installed(
        &self,
        helper: &RuntimeHelperScript,
    ) -> SandboxResult<String> {
        let present = self
            .exec(ExecRequest::argv(helper.present_command()))
            .await?;
        if present.ok() {
            return Ok(helper.install_path.clone());
        }
        let command = helper.install_command();
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(
                &[
                    "install_runtime_helper".to_string(),
                    helper.name.to_string(),
                ],
                &result,
            ));
        }
        Ok(helper.install_path.clone())
    }

    /// The workspace fingerprint the snapshot this session resumes was taken from.
    ///
    /// A session that returns `Some` lets [`restore_session_snapshot`] skip the archive when the
    /// live workspace still hashes to the same value. Mirrors Python's
    /// `SandboxSessionState.snapshot_fingerprint`.
    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        None
    }

    /// Record the fingerprint of the workspace that was just archived.
    ///
    /// The default drops it, which only costs a resumed session the fast path.
    fn record_snapshot_fingerprint(&self, record: Option<SnapshotFingerprint>) {
        let _ = record;
    }

    /// The ports this session asked the backend to expose on the host.
    ///
    /// Mirrors `SandboxSessionState.exposed_ports`. A backend that exposes nothing returns an
    /// empty slice, which is what makes the default
    /// [`resolve_exposed_port`](Self::resolve_exposed_port) refuse every port.
    fn exposed_ports(&self) -> Vec<u16> {
        Vec::new()
    }

    /// Resolve a sandbox-side port to an endpoint a host process can reach.
    ///
    /// Mirrors Python's `BaseSandboxSession.resolve_exposed_port`. A sandbox port is almost
    /// never reachable at the number the workload bound it to — the provider maps it, usually
    /// onto a TLS name with a signed query string — so the host side has to ask the backend.
    /// The default refuses: a backend with no port mapping has nothing to hand out, and a port
    /// the client never declared can never start working, which is why `not_configured` is
    /// explicitly non-retryable.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        let exposed = self.exposed_ports();
        let reason = if exposed.contains(&port) {
            ExposedPortUnavailable::BackendUnavailable
        } else {
            ExposedPortUnavailable::NotConfigured
        };
        Err(exposed_port_unavailable(
            self.backend_id(),
            port,
            &exposed,
            reason,
        ))
    }

    /// Read at most `max_bytes` bytes from the start of a workspace file.
    ///
    /// Mirrors Python's `read_bounded`. A result of exactly `max_bytes` does **not** mean the
    /// file ended there. This matters for anything that reads a file the workload can write: an
    /// in-workspace log the agent appends to has no size the SDK controls, and reading it whole
    /// to look at its first kilobyte is how a sandbox turns a disk into memory pressure on the
    /// host.
    ///
    /// The default reads through [`read_file`](Self::read_file) and truncates, which bounds what
    /// the *caller* holds but not what the backend fetched. A backend whose transport can bound
    /// the fetch should override this, and report a transport that refuses the request with
    /// `context["reason"] = "bounded_read_wire_limit"` so the caller can tell the two apart.
    async fn read_bounded(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if max_bytes == 0 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "read_bounded max_bytes must be positive",
            ));
        }
        let mut contents = self.read_file(path).await?;
        contents.truncate(max_bytes);
        Ok(contents)
    }

    /// The runtime objects this session's snapshot and manifest entries may need.
    ///
    /// Mirrors Python's `BaseSandboxSession.dependencies`. The default container is empty, which
    /// is all a backend that needs nothing by name requires.
    fn dependencies(&self) -> Option<&Arc<crate::sandbox::dependencies::Dependencies>> {
        None
    }

    /// Whether [`persist_session_snapshot`] should fingerprint the workspace before archiving it.
    ///
    /// Mirrors Python's `_should_compute_snapshot_fingerprint_on_persist`.
    fn computes_snapshot_fingerprint(&self) -> bool {
        true
    }

    /// Whether a path exists inside the sandbox, following symlinks without leaving the probe.
    ///
    /// Mirrors Python's `_READ_PATH_PROBE_SCRIPT`. A failed archive read is ambiguous — a
    /// missing file and an unreadable one both fail — so the probe is what tells a
    /// `WorkspaceReadNotFound` apart from a real read error.
    async fn probe_read_path(&self, path: &str) -> SandboxResult<bool> {
        let result = self
            .exec(
                ExecRequest::argv(["sh", "-c", READ_PATH_PROBE_SCRIPT, "sh", path])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(ACCESS_PROBE_TIMEOUT)),
            )
            .await?;
        Ok(result.exit_code == 0)
    }

    /// Check that a file write at `path` can succeed, and say why not.
    ///
    /// Mirrors Python's `_check_write_with_exec`. Unlike Python this runs after a failed write
    /// rather than before every write: it turns the backend's error into the same typed one
    /// without costing a round trip on the path that works.
    async fn check_write_access(&self, path: &str, user: Option<&User>) -> SandboxResult<()> {
        self.run_access_check(
            WRITE_ACCESS_CHECK_SCRIPT,
            "write_access_check",
            &[path],
            user,
        )
        .await
    }

    /// Check that `mkdir` at `path` can succeed, and say why not.
    ///
    /// Mirrors Python's `_check_mkdir_with_exec`.
    async fn check_mkdir_access(
        &self,
        path: &str,
        parents: bool,
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let parents_flag = if parents { "1" } else { "0" };
        self.run_access_check(
            MKDIR_ACCESS_CHECK_SCRIPT,
            "mkdir_access_check",
            &[path, parents_flag],
            user,
        )
        .await
    }

    /// Check that `rm` at `path` can succeed, and say why not.
    ///
    /// Mirrors Python's `_check_rm_with_exec`.
    async fn check_rm_access(
        &self,
        path: &str,
        recursive: bool,
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let recursive_flag = if recursive { "1" } else { "0" };
        self.run_access_check(
            RM_ACCESS_CHECK_SCRIPT,
            "rm_access_check",
            &[path, recursive_flag],
            user,
        )
        .await
    }

    /// Run one access-check script and turn a non-zero exit into a typed write error.
    async fn run_access_check(
        &self,
        script: &str,
        label: &str,
        args: &[&str],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let mut argv = vec![
            "sh".to_string(),
            "-c".to_string(),
            script.to_string(),
            "sh".to_string(),
        ];
        argv.extend(args.iter().map(|arg| arg.to_string()));
        let result = self
            .exec(
                ExecRequest::argv(argv)
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(ACCESS_PROBE_TIMEOUT))
                    .with_user(user.cloned()),
            )
            .await?;
        if result.ok() {
            return Ok(());
        }
        Err(SandboxError::new(
            ErrorCode::WorkspaceArchiveWriteError,
            format!(
                "`{}` refused the sandbox path `{}`: {}",
                label,
                args.first().copied().unwrap_or(""),
                result.stderr_text().trim()
            ),
        )
        .with_context("check", label.to_string())
        .with_context("path", args.first().copied().unwrap_or("").to_string()))
    }
}

/// How long an access-probe command may run. Mirrors Python's `_READ_PATH_PROBE_TIMEOUT_S`.
pub const ACCESS_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Report whether a path exists, resolving symlinks without following one out of its depth.
pub const READ_PATH_PROBE_SCRIPT: &str = include_str!("helpers/read-path-probe.sh");

/// Report whether a file write at `$1` can succeed.
pub const WRITE_ACCESS_CHECK_SCRIPT: &str = include_str!("helpers/write-access-check.sh");

/// Report whether `mkdir` at `$1` (with `$2` = `1` for `-p`) can succeed.
pub const MKDIR_ACCESS_CHECK_SCRIPT: &str = include_str!("helpers/mkdir-access-check.sh");

/// Report whether `rm` at `$1` (with `$2` = `1` for `-r`) can succeed.
pub const RM_ACCESS_CHECK_SCRIPT: &str = include_str!("helpers/rm-access-check.sh");

/// Where a session caches its workspace fingerprint inside the sandbox.
///
/// Mirrors Python's `snapshot_fingerprint_cache_path`. It is outside the workspace root, so it
/// never lands in an archive and never shows up in a directory listing the model reads.
pub fn snapshot_fingerprint_cache_path(session_id: &str) -> String {
    format!("/tmp/openai-agents/session-state/{session_id}/fingerprint.json")
}

/// The workspace-relative paths the fingerprint and the archive both leave out.
fn workspace_fingerprint_skip_relpaths(manifest: &Manifest, root: &str) -> Vec<String> {
    let mut relpaths: Vec<String> = workspace_archive_excludes(manifest, root)
        .into_iter()
        .map(|exclude| exclude.trim_start_matches("./").to_string())
        .filter(|relative| !relative.is_empty())
        .collect();
    relpaths.extend(workspace_resume_mount_skip_relpaths(manifest, root));
    relpaths.sort();
    relpaths.dedup();
    relpaths
}

/// The workspace-relative paths of the ephemeral mounts a resume must not clear.
///
/// Mirrors Python's `workspace_resume_mount_skip_relpaths`: a mount materializes live remote
/// storage inside the workspace, so clearing it would delete the bucket, not the workspace.
pub fn workspace_resume_mount_skip_relpaths(manifest: &Manifest, root: &str) -> Vec<String> {
    let trimmed = root.trim_end_matches('/');
    manifest
        .ephemeral_mount_targets()
        .into_iter()
        .filter_map(|(_, path)| {
            path.strip_prefix(trimmed)
                .map(|relative| relative.trim_start_matches('/').to_string())
                .filter(|relative| !relative.is_empty())
        })
        .collect()
}

/// Compute the workspace fingerprint inside the sandbox and cache it there.
///
/// Mirrors Python's `compute_and_cache_snapshot_fingerprint`. The hashing happens in the sandbox
/// because the workspace is there: streaming the tree to the host to hash it would cost the whole
/// archive, which is exactly what the fast path exists to avoid.
pub async fn compute_and_cache_snapshot_fingerprint(
    session: &dyn SandboxSession,
    version: Option<&str>,
) -> SandboxResult<SnapshotFingerprint> {
    let helper_path = session
        .ensure_runtime_helper_installed(&WORKSPACE_FINGERPRINT_HELPER)
        .await?;
    let root = session.workspace_root().to_string();
    let manifest = session.manifest();
    let mut command = vec![
        helper_path,
        root.clone(),
        version.unwrap_or(SNAPSHOT_FINGERPRINT_VERSION).to_string(),
        snapshot_fingerprint_cache_path(session.id()),
        crate::sandbox::state::manifest_digest(manifest),
    ];
    command.extend(workspace_fingerprint_skip_relpaths(manifest, &root));
    let result = session.exec(ExecRequest::argv(command.clone())).await?;
    if !result.ok() {
        return Err(SandboxError::exec_nonzero(&command, &result));
    }
    parse_snapshot_fingerprint_record(&result.stdout_text())
}

/// Read the fingerprint record the helper cached inside the sandbox.
pub async fn read_cached_snapshot_fingerprint(
    session: &dyn SandboxSession,
) -> SandboxResult<SnapshotFingerprint> {
    let path = snapshot_fingerprint_cache_path(session.id());
    let command = vec!["cat".to_string(), "--".to_string(), path];
    let result = session.exec(ExecRequest::argv(command.clone())).await?;
    if !result.ok() {
        return Err(SandboxError::exec_nonzero(&command, &result));
    }
    parse_snapshot_fingerprint_record(&result.stdout_text())
}

/// Parse the `{"fingerprint": ..., "version": ...}` record the helper prints.
pub fn parse_snapshot_fingerprint_record(payload: &str) -> SandboxResult<SnapshotFingerprint> {
    let invalid = |reason: &str| {
        SandboxError::new(
            ErrorCode::SessionStateInvalid,
            format!("snapshot fingerprint payload {reason}"),
        )
    };
    let value: serde_json::Value =
        serde_json::from_str(payload.trim()).map_err(|_| invalid("is not JSON"))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("is not an object"))?;
    let field = |name: &str| {
        object
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    Ok(SnapshotFingerprint {
        fingerprint: field("fingerprint").ok_or_else(|| invalid("is missing `fingerprint`"))?,
        version: field("version").ok_or_else(|| invalid("is missing `version`"))?,
    })
}

/// Whether the live workspace still matches the snapshot this session resumes.
///
/// Mirrors Python's `live_workspace_matches_snapshot_on_resume`. An old record is recomputed with
/// *its* algorithm, because switching algorithms before the comparison would make a preserved
/// workspace look stale and restore an archive over live edits.
pub async fn live_workspace_matches_snapshot_on_resume(session: &dyn SandboxSession) -> bool {
    let Some(stored) = session.resume_snapshot_fingerprint() else {
        return false;
    };
    if stored.fingerprint.is_empty() || stored.version.is_empty() {
        return false;
    }
    let version =
        (stored.version == SNAPSHOT_FINGERPRINT_VERSION_V1).then_some(stored.version.as_str());
    match compute_and_cache_snapshot_fingerprint(session, version).await {
        Err(_) => false,
        Ok(current) => current == stored,
    }
}

/// Remove everything a resume is about to replace, leaving live mounts alone.
///
/// Mirrors Python's `clear_workspace_root_on_resume`. A mount's directory is skipped because the
/// files under it are remote storage; its *ancestors* are walked rather than removed, so a stale
/// sibling next to a mount still goes.
pub async fn clear_workspace_root_on_resume(session: &dyn SandboxSession) -> SandboxResult<()> {
    let root = session.workspace_root().to_string();
    let skip = workspace_resume_mount_skip_relpaths(session.manifest(), &root);
    if skip
        .iter()
        .any(|relative| relative == "." || relative.is_empty())
    {
        return Ok(());
    }
    clear_workspace_dir_on_resume_pruned(session, &root, &root, &skip).await
}

async fn clear_workspace_dir_on_resume_pruned(
    session: &dyn SandboxSession,
    root: &str,
    current_dir: &str,
    skip: &[String],
) -> SandboxResult<()> {
    let Ok(entries) = session.ls(current_dir).await else {
        // A root that is missing or unlistable is as good as empty: materialization recreates it.
        return Ok(());
    };
    let trimmed_root = root.trim_end_matches('/');
    for entry in entries {
        let Some(relative) = entry
            .path
            .strip_prefix(trimmed_root)
            .map(|relative| relative.trim_start_matches('/').to_string())
            .filter(|relative| !relative.is_empty())
        else {
            let _ = session.rm(&entry.path, true).await;
            continue;
        };
        if skip.iter().any(|skipped| skipped == &relative) {
            continue;
        }
        let is_ancestor_of_mount = skip
            .iter()
            .any(|skipped| skipped.starts_with(&format!("{relative}/")));
        if is_ancestor_of_mount {
            if entry.is_dir {
                Box::pin(clear_workspace_dir_on_resume_pruned(
                    session,
                    root,
                    &entry.path,
                    skip,
                ))
                .await?;
            } else {
                let _ = session.rm(&entry.path, true).await;
            }
            continue;
        }
        let _ = session.rm(&entry.path, true).await;
    }
    Ok(())
}

/// Resolve `path` against the manifest's extra path grants.
///
/// Returns the normalized path when it falls under a grant. A read-only grant still resolves:
/// the grant's `read_only` flag is enforced where the write happens, not here, so an attempt to
/// write reports a permission problem instead of looking like a path that does not exist.
/// The `tar --exclude` patterns every workspace archive applies.
///
/// Two kinds of path must never reach an archive:
///
/// - The manifest's ephemeral entries. Mounts and the like are runtime-attached state, never
///   durable workspace content; archiving them would copy whole mounted buckets into the
///   snapshot.
/// - The staging directories a mount writes its credentials and cache to
///   ([`RCLONE_CONFIG_DIR`](crate::sandbox::mounts::RCLONE_CONFIG_DIR) and
///   [`MOUNT_SCRATCH_DIRS`](crate::sandbox::mount_patterns::MOUNT_SCRATCH_DIRS)). They live
///   inside the workspace because the session's file API cannot write outside it, so the archive
///   has to leave them out explicitly. One of them is a blob cache that can reach tens of
///   gigabytes.
///
/// A backend that overrides [`persist_workspace`](SandboxSession::persist_workspace) — every
/// hosted one does, because their command output is text — calls this for the same exclusions.
pub fn workspace_archive_excludes(manifest: &Manifest, root: &str) -> Vec<String> {
    let trimmed = root.trim_end_matches('/');
    let mut excludes: Vec<String> = manifest
        .ephemeral_paths()
        .into_iter()
        .filter_map(|path| {
            path.strip_prefix(trimmed)
                .map(|relative| relative.trim_start_matches('/').to_string())
                .filter(|relative| !relative.is_empty())
                .map(|relative| format!("./{relative}"))
        })
        .collect();
    excludes.push(format!("./{}", crate::sandbox::mounts::RCLONE_CONFIG_DIR));
    excludes.extend(
        crate::sandbox::mount_patterns::MOUNT_SCRATCH_DIRS
            .iter()
            .map(|directory| format!("./{directory}")),
    );
    excludes
}

/// Capture a session's workspace into its snapshot.
///
/// Mounts whose strategy cannot be snapshotted in place are detached first and re-attached
/// afterwards, mirroring Python's `teardown_for_snapshot` / `restore_after_snapshot` pair.
/// Sessions without a configured snapshot are a no-op.
pub async fn persist_session_snapshot(session: &dyn SandboxSession) -> SandboxResult<()> {
    let snapshot = session.snapshot();
    if snapshot.snapshot_type() == "noop" {
        return Ok(());
    }
    // Fingerprint before the archive, so the record describes the tree that goes into it. A
    // failure here only costs the resume its fast path, so it is not fatal.
    let fingerprint = if session.computes_snapshot_fingerprint() {
        compute_and_cache_snapshot_fingerprint(session, None)
            .await
            .ok()
    } else {
        None
    };
    session.record_snapshot_fingerprint(fingerprint);
    let manifest = session.manifest().clone();
    let detached: Vec<(&Mount, String)> = manifest
        .mount_targets()
        .into_iter()
        .filter(|(mount, _)| !mount.strategy.supports_native_snapshot_detach())
        .collect();
    // While these are detached the workspace is not what the manifest describes, so other tasks
    // are kept out of it until they are back. The guard is released even if the archive fails.
    let _transition = match session.mount_transition() {
        Some(transition) if !detached.is_empty() => Some(transition.begin()?),
        _ => None,
    };
    for (mount, path) in &detached {
        mount
            .strategy
            .teardown_for_snapshot(mount, session, path)
            .await?;
    }
    let archive = session.persist_workspace().await;
    for (mount, path) in &detached {
        mount
            .strategy
            .restore_after_snapshot(mount, session, path)
            .await?;
    }
    let dependencies = session.dependencies().map(Arc::as_ref);
    match snapshot.persist_with(&archive?, dependencies).await {
        Ok(()) => Ok(()),
        Err(error) => {
            // A cached fingerprint that no archive matches would let the next resume skip a
            // restore it needs.
            session.record_snapshot_fingerprint(None);
            Err(error)
        }
    }
}

/// Restore a session's workspace from its snapshot and re-apply the ephemeral entries.
///
/// Returns `false` when no archive is stored, leaving the workspace untouched.
pub async fn restore_session_snapshot(session: &dyn SandboxSession) -> SandboxResult<bool> {
    let snapshot = session.snapshot();
    if !snapshot.restorable().await? {
        return Ok(false);
    }
    let manifest = session.manifest().clone();
    // The fast path: a workspace that survived the pause and still hashes to the fingerprint the
    // archive was taken from is already the archive, so unpacking it would only overwrite the
    // files with identical copies — and a full restore of a large workspace is the single most
    // expensive step of a resume.
    if live_workspace_matches_snapshot_on_resume(session).await {
        manifest.apply_ephemeral(session).await?;
        return Ok(true);
    }
    // Everything the archive does not contain has to go, or a file the paused run deleted comes
    // back. The live mounts are the exception: under them is remote storage, not workspace.
    clear_workspace_root_on_resume(session).await?;
    let archive = snapshot
        .restore_with(session.dependencies().map(Arc::as_ref))
        .await?;
    session.hydrate_workspace(&archive).await?;
    manifest.apply_ephemeral(session).await?;
    Ok(true)
}

/// Creates and releases [`SandboxSession`]s for one backend.
#[async_trait]
pub trait SandboxClient: Send + Sync + fmt::Debug {
    /// The backend identifier (`unix_local`, `docker`, ...).
    fn backend_id(&self) -> &str;

    /// Create a fresh session that will materialize `manifest` when it starts.
    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>>;

    /// Release the backend resources behind a session created by this client.
    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()>;

    /// Reattach to (or recreate) the session described by `state`.
    ///
    /// `manifest` is the trusted manifest the application builds; `state` carries no manifest
    /// contents of its own, so the client checks the two agree before touching anything. A
    /// backend whose resource is gone may create a replacement and hydrate it from the snapshot
    /// recorded in `state`.
    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        let _ = (state, manifest);
        Err(SandboxError::new(
            ErrorCode::SessionStateInvalid,
            format!(
                "the `{}` sandbox backend does not support resuming a persisted session",
                self.backend_id()
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_shell_and_sudo_wrappers() {
        let req = ExecRequest::shell("echo hi");
        assert_eq!(req.resolved_argv(), vec!["sh", "-lc", "echo hi"]);

        let req = ExecRequest::argv(["ls", "-la", "/a b"]);
        assert_eq!(req.resolved_argv(), vec!["ls", "-la", "/a b"]);

        let req = ExecRequest::argv(["ls", "/a b"]).with_shell(ExecShell::Plain);
        assert_eq!(req.resolved_argv(), vec!["sh", "-c", "ls '/a b'"]);

        let req = ExecRequest::shell("id").with_user(Some(User::new("agent")));
        assert_eq!(
            req.resolved_argv(),
            vec!["sudo", "-u", "agent", "--", "sh", "-lc", "id"]
        );
    }

    #[test]
    fn merges_request_env_over_the_manifest_env() {
        let manifest_env = BTreeMap::from([
            ("FOO".to_string(), "manifest".to_string()),
            ("BAR".to_string(), "keep".to_string()),
        ]);
        let request = ExecRequest::shell("env").with_env("FOO", "request");
        assert_eq!(
            request.resolved_env(&manifest_env),
            BTreeMap::from([
                ("BAR".to_string(), "keep".to_string()),
                ("FOO".to_string(), "request".to_string()),
            ])
        );
        // A request without its own environment leaves the manifest's untouched.
        assert_eq!(
            ExecRequest::shell("env").resolved_env(&manifest_env),
            manifest_env
        );
    }

    #[test]
    fn quotes_arguments() {
        assert_eq!(shell_quote("plain-arg_1.txt"), "plain-arg_1.txt");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote(""), "''");
    }
}
