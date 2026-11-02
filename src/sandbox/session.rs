//! The sandbox session and client interfaces. Mirrors
//! `agents.sandbox.session.base_sandbox_session` and `agents.sandbox.session.sandbox_client`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::archive::{self, CompressionScheme};
use crate::sandbox::errors::SandboxOp;
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{Manifest, SandboxConcurrencyLimits};
use crate::sandbox::mounts::{Mount, MountTransition};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::runtime_helpers::{RuntimeHelperScript, WORKSPACE_FINGERPRINT_HELPER};
use crate::sandbox::snapshot::{Snapshot, resolve_snapshot};
use crate::sandbox::state::{
    SNAPSHOT_FINGERPRINT_VERSION, SNAPSHOT_FINGERPRINT_VERSION_V1, SandboxSessionState,
    SnapshotFingerprint,
};
use crate::sandbox::types::{
    ExecResult, ExposedPortEndpoint, ExposedPortUnavailable, FileEntry, SandboxArchiveLimits, User,
    exposed_port_unavailable, parse_ls_la,
};
use crate::sandbox::workspace_paths::{
    SandboxPathGrant, WorkspacePathPolicy, is_provider_scratch_path,
};

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

/// The extra path grants a capability applied to a live session after it started.
///
/// Empty until a capability changes them; once set, the stored list (even an empty one) replaces
/// the manifest's grants for every path check the session makes.
#[derive(Debug, Default)]
pub struct LiveGrants(std::sync::RwLock<Option<Vec<SandboxPathGrant>>>);

impl LiveGrants {
    /// The grants in force: the last applied list, else the manifest's.
    pub fn current(&self, manifest: &Manifest) -> Vec<SandboxPathGrant> {
        match &*self.0.read().unwrap_or_else(|e| e.into_inner()) {
            Some(grants) => grants.clone(),
            None => manifest.extra_path_grants().to_vec(),
        }
    }

    /// Replace the grants in force.
    pub fn replace(&self, grants: Vec<SandboxPathGrant>) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = Some(grants);
    }
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

    /// The cell holding the path grants a capability changed after the session started, for a
    /// backend that supports it. `None` means the manifest's grants are final.
    fn live_grants(&self) -> Option<&LiveGrants> {
        None
    }

    /// The extra path grants in force: the ones a capability last applied to this live session,
    /// else the manifest's.
    fn extra_path_grants(&self) -> Vec<SandboxPathGrant> {
        match self.live_grants() {
            Some(live) => live.current(self.manifest()),
            None => self.manifest().extra_path_grants().to_vec(),
        }
    }

    /// Replace the extra path grants of a running session, so a capability's change to what the
    /// workspace may reach takes effect on an injected session.
    ///
    /// Mirrors Python replacing `session.state.manifest` with the processed manifest. A backend
    /// with no [`live_grants`](Self::live_grants) cell ignores the call.
    fn update_extra_path_grants(&self, grants: &[SandboxPathGrant]) {
        if let Some(live) = self.live_grants() {
            live.replace(grants.to_vec());
        }
    }

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
            // A failed `cat` does not say why: the probe separates a missing file from one the
            // user may not read, which the model reacts to very differently.
            return Err(self.read_error_from_exec(&path, &result, Some(user)).await);
        }
        Ok(result.stdout)
    }

    /// Read a file as `user`, where the errors in `expected` are an ordinary answer.
    ///
    /// Mirrors Python's `_read_with_expected_span_errors`. The result is exactly that of
    /// [`read_file_as`](Self::read_file_as); a session that records trace spans leaves the span
    /// of an expected failure unmarked, so asking "is this file there yet?" does not paint the
    /// trace with an error. The default ignores `expected`.
    async fn read_file_expecting(
        &self,
        path: &str,
        user: Option<&User>,
        expected: &[ErrorCode],
    ) -> SandboxResult<Vec<u8>> {
        let _ = expected;
        self.read_file_as(path, user).await
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
        self.write_file_as_with(path, contents, user, false).await
    }

    /// [`write_file_as`](Self::write_file_as) that can refuse to replace what is already there.
    ///
    /// With `exclusive`, the staged file is hard-linked into place instead of moved over the
    /// destination, so a name that exists by then (a regular file, a directory, or a symlink,
    /// dangling ones included) makes the write fail with `already exists` and is never touched.
    async fn write_file_as_with(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
        exclusive: bool,
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

        if exclusive {
            let linked = run(format!(
                "ln -- {quoted_staging} {quoted_path} && rm -f -- {quoted_staging}"
            ))
            .await?;
            if !linked.ok() {
                let occupied = run(format!("[ -e {quoted_path} ] || [ -L {quoted_path} ]")).await?;
                let _ = run(format!("rm -f -- {quoted_staging}")).await;
                if occupied.ok() {
                    return Err(SandboxError::new(
                        ErrorCode::ApplyPatchInvalidPath,
                        format!("`{path}` already exists"),
                    ));
                }
                return Err(failed("write", &linked));
            }
            return Ok(());
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
        if self.has_read_only_grant() || self.resolves_paths_remotely() {
            self.validate_remote_path_access(path, true, recursive)
                .await?;
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

    /// Move a path as `user`. See [`mv`](Self::mv) for what can be moved.
    async fn mv_as(
        &self,
        source: &str,
        destination: &str,
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.mv(source, destination).await;
        };
        mv_via_exec(self, source, destination, Some(user)).await
    }

    /// Whether two paths name the same file on the sandbox filesystem.
    ///
    /// Mirrors Python's `BaseSandboxSession.same_file`. The filesystem is asked, through
    /// `test -ef`, which compares device and inode, because two spellings can be one file: a
    /// filesystem that folds case stores `notes.txt` and `Notes.txt` as one entry, and APFS also
    /// folds Unicode normalization. No string comparison answers that, and neither does the host
    /// driving the session, which need not be the kind of system the sandbox runs on.
    ///
    /// With `follow_symlinks` false the answer is false whenever either leaf is a symlink, which
    /// is the mode used before removing an `apply_patch` source.
    async fn same_file(
        &self,
        left: &str,
        right: &str,
        follow_symlinks: bool,
        user: Option<&User>,
    ) -> SandboxResult<bool> {
        let left = self.normalize_path(left)?;
        let right = self.normalize_path(right)?;
        let mut test = "[ \"$1\" -ef \"$2\" ]".to_string();
        if !follow_symlinks {
            test = format!("[ ! -L \"$1\" ] && [ ! -L \"$2\" ] && {test}");
        }
        let argv = vec![
            "sh".to_string(),
            "-c".to_string(),
            test,
            "sh".to_string(),
            left.clone(),
            right.clone(),
        ];
        let result = self
            .exec(
                ExecRequest::argv(argv)
                    .with_shell(ExecShell::None)
                    .with_user(user.cloned()),
            )
            .await?;
        match result.exit_code {
            0 => Ok(true),
            // `[` answers "different file" with 1 and reports its own failures with 2; a missing
            // shell exits 127. Only 1 is an answer. A caller about to delete a file on the
            // strength of this must not read "could not tell" as "different".
            1 => Ok(false),
            _ => {
                let shown = vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "<same_file_check>".to_string(),
                    left,
                    right,
                ];
                Err(SandboxError::exec_nonzero(&shown, &result))
            }
        }
    }

    /// Create a file, creating its parent directories.
    ///
    /// Mirrors Python's `_write_new_file` hook. A backend with a native no-clobber primitive
    /// overrides this and fails with an error whose message contains `already exists`, which the
    /// editor reports to the model as an occupied name. The default has no such primitive, so it
    /// is a plain mkdir plus write and keeps the provider's write semantics: creating over an
    /// existing name overwrites it, exactly as Python's base class does. A read probe would not
    /// close the gap, only make it racy, so the default does not pretend to.
    async fn write_new_file(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let target = self.normalize_path_for_write(path)?;
        if let Some((parent, _)) = target.rsplit_once('/')
            && !parent.is_empty()
        {
            self.mkdir_as(parent, true, user).await?;
        }
        self.write_file_as(&target, contents, user).await
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

    /// Whether the operating-system state (accounts, packages) survived the pause this session is
    /// resuming from.
    ///
    /// Mirrors Python's `should_provision_manifest_accounts_on_resume` being false: a backend that
    /// resumes into the same container or VM it paused already has the manifest's accounts, so
    /// restoring a snapshot must not run `useradd` for them again. The default `false` is the
    /// safe one for a fresh sandbox, which has none of them.
    fn system_state_preserved_on_start(&self) -> bool {
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
    /// [`SandboxPathGrant`] is enforced by the
    /// SDK and not only by whatever the backend's own filesystem does.
    fn normalize_path_for_write(&self, path: &str) -> SandboxResult<String> {
        self.workspace_path_policy()?
            .normalize_sandbox_path(path, true)
    }

    /// The path policy for this session's workspace root and grants.
    fn workspace_path_policy(&self) -> SandboxResult<WorkspacePathPolicy> {
        WorkspacePathPolicy::with_grants(self.workspace_root(), self.extra_path_grants())
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
        let resolved = if self.resolves_paths_remotely() {
            self.validate_remote_path_access(path, false, false).await?
        } else {
            self.normalize_path(path)?
        };
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

    /// List a directory as `user`.
    ///
    /// Mirrors Python's `ls(path, user=...)`: the listing runs as that user, so a directory the
    /// user may not read fails as it would in their own shell. The default runs `ls -la` through
    /// [`exec`](Self::exec), which applies the `sudo` wrapper.
    async fn ls_as(&self, path: &str, user: Option<&User>) -> SandboxResult<Vec<FileEntry>> {
        let Some(user) = user else {
            return self.ls(path).await;
        };
        let resolved = if self.resolves_paths_remotely() {
            self.validate_remote_path_access(path, false, false).await?
        } else {
            self.normalize_path(path)?
        };
        let command = vec![
            "ls".to_string(),
            "-la".to_string(),
            "--".to_string(),
            resolved.clone(),
        ];
        let result = self
            .exec(ExecRequest::argv(command.clone()).with_user(Some(user.clone())))
            .await?;
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
        // With a read-only grant in play, an alias the lexical check cannot see could point a
        // writable-looking path at protected data, so the sandbox resolves the real tree.
        if self.has_read_only_grant() || self.resolves_paths_remotely() {
            self.validate_remote_path_access(path, true, recursive)
                .await?;
        }
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
    ///
    /// Mirrors Python's `BaseSandboxSession.mv`: only a regular file or a symlink can be moved,
    /// and an existing destination entry is replaced. `mv -T` makes a destination directory an
    /// error instead of a place to put the source, which matters because `move_to` comes from
    /// the model: a plain `mv` would report success for a move that never happened. A pre-check
    /// for the directory would not do, since one created between the check and the move would
    /// still receive the source.
    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        mv_via_exec(self, source, destination, None).await
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
        capture_workspace_tar(self).await
    }

    /// Restore a gzipped tar archive produced by [`persist_workspace`](Self::persist_workspace).
    ///
    /// Every implementation validates the archive with
    /// [`validate_tar_bytes`](crate::sandbox::archive::validate_tar_bytes) under
    /// [`TarValidationOptions::hydrate`](crate::sandbox::archive::TarValidationOptions::hydrate) before a member can reach a filesystem, local or remote.
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()>;

    /// Reject an archive this session could not safely hydrate, before a restore clears the
    /// workspace.
    ///
    /// Mirrors the validation Python runs in `_restore_workspace_snapshot` ahead of
    /// `_clear_workspace_root_on_resume`: a workspace must never be emptied for an archive that
    /// turns out to be unusable. The default accepts everything, because a backend whose
    /// "archive" is a provider-native snapshot reference has no tar to inspect; a backend that
    /// unpacks a tar overrides it with [`validate_tar_bytes`](crate::sandbox::archive::validate_tar_bytes).
    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        let _ = archive;
        Ok(())
    }

    /// The archive resource limits this session extracts under.
    ///
    /// Mirrors Python's `BaseSandboxSession._archive_limits`. `None` means no SDK ceiling, which
    /// is also what [`SandboxRunConfig::archive_limits`] defaults to.
    ///
    /// [`SandboxRunConfig::archive_limits`]: crate::sandbox::SandboxRunConfig::archive_limits
    fn archive_limits(&self) -> Option<SandboxArchiveLimits> {
        None
    }

    /// Record the archive limits the runner resolved for this session.
    ///
    /// Mirrors Python's `_set_archive_limits`. The default drops them, which costs a backend
    /// nothing but the per-session default: a caller can still pass limits to
    /// [`extract_with_limits`](Self::extract_with_limits) explicitly.
    fn set_archive_limits(&self, limits: Option<SandboxArchiveLimits>) {
        let _ = limits;
    }

    /// Record the concurrency limits the runner resolved for this session.
    ///
    /// Mirrors Python's `_set_concurrency_limits`. Limits travel on the manifest a session
    /// materializes, so the default drops them; a backend that applies work outside its manifest
    /// can keep them.
    fn set_concurrency_limits(&self, limits: SandboxConcurrencyLimits) {
        let _ = limits;
    }

    /// Write an archive into the workspace and extract it next to itself.
    ///
    /// Mirrors Python's `BaseSandboxSession.extract`. `compression_scheme` is inferred from the
    /// path's extension when it is `None`, and the session's [`archive_limits`](Self::archive_limits)
    /// apply. Every member is validated in process first: a member that is absolute, traverses
    /// out of the archive, is a link, or is a device node is refused before anything is written,
    /// on every backend.
    async fn extract(
        &self,
        path: &str,
        data: &[u8],
        compression_scheme: Option<CompressionScheme>,
    ) -> SandboxResult<()> {
        self.extract_with_limits(path, data, compression_scheme, self.archive_limits())
            .await
    }

    /// [`extract`](Self::extract) with per-call resource limits.
    async fn extract_with_limits(
        &self,
        path: &str,
        data: &[u8],
        compression_scheme: Option<CompressionScheme>,
        limits: Option<SandboxArchiveLimits>,
    ) -> SandboxResult<()> {
        if let Some(limits) = limits.as_ref() {
            limits.validate()?;
        }
        let scheme = compression_scheme
            .or_else(|| CompressionScheme::infer_from_path(path))
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::InvalidCompressionScheme,
                    format!("`{path}` has no supported compression scheme: expected tar or zip"),
                )
                .with_context("path", path)
                .with_retryable(false)
            })?;
        let archive_path = self.normalize_path_for_write(path)?;
        let destination_root = parent_sandbox_path(&archive_path);

        archive::check_input_bytes(data.len() as u64, limits.as_ref())?;
        // The archive itself lands in the workspace first, as Python's `extract_archive` does:
        // the caller asked for it at `path`, and the members are restored beside it.
        self.write_file(&archive_path, data).await?;

        let entries = match scheme {
            CompressionScheme::Tar => archive::validated_tar_entries(
                data,
                &archive::TarValidationOptions::extract(limits),
            )?,
            CompressionScheme::Zip => archive::validated_zip_entries(data, limits.as_ref())?,
        };

        // Children of each listed directory, as `name -> is_symlink`. A member must not be
        // written through a link that is already in the destination, and the listing is what
        // says so before the write happens.
        let mut listings: BTreeMap<String, BTreeMap<String, bool>> = BTreeMap::new();
        for entry in entries {
            let relative = sandbox_rel_parts(&entry.rel_path);
            let mut current = destination_root.clone();
            let mut traversed = String::new();
            for part in &relative {
                if !listings.contains_key(&current) {
                    let children = self
                        .ls(&current)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .map(|child| (child.name, child.is_symlink))
                        .collect();
                    listings.insert(current.clone(), children);
                }
                let Some(is_symlink) = listings[&current].get(part).copied() else {
                    break;
                };
                traversed = if traversed.is_empty() {
                    part.clone()
                } else {
                    format!("{traversed}/{part}")
                };
                if is_symlink {
                    return Err(archive::unsafe_tar_member(
                        &entry.rel_path.to_string_lossy(),
                        format!("symlink in parent path: {traversed}"),
                    ));
                }
                current = join_sandbox_path(&current, part);
            }

            let dest = join_sandbox_path(&destination_root, &relative.join("/"));
            if entry.is_dir {
                self.mkdir(&dest, true).await?;
            } else {
                self.mkdir(&parent_sandbox_path(&dest), true).await?;
                self.write_file(&dest, &entry.contents).await?;
            }
            // What was just created is known not to be a link, so a later member descending
            // through it does not list the directory again.
            if let (Some(parent), Some(name)) = (
                listings.get_mut(&parent_sandbox_path(&dest)),
                relative.last(),
            ) {
                parent.insert(name.clone(), false);
            }
        }
        Ok(())
    }

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

    /// Validate a path against the sandbox's own filesystem before an operation touches it.
    ///
    /// Mirrors Python's `_validate_remote_path_access`. The lexical check cannot see a symlink
    /// that leads out of an allowed root, or one that makes a read-only grant reachable from a
    /// writable path, so the installed resolver walks the real tree inside the sandbox. A
    /// recursive removal passes `recursive_remove` so the resolver also refuses a tree that
    /// contains a read-only grant. The returned path is the normalized request, not the
    /// realpath: a leaf symlink keeps its own semantics, so removing a link removes the link.
    async fn validate_remote_path_access(
        &self,
        path: &str,
        for_write: bool,
        recursive_remove: bool,
    ) -> SandboxResult<String> {
        use crate::sandbox::runtime_helpers::RESOLVE_WORKSPACE_PATH_HELPER as HELPER;

        let policy = self.workspace_path_policy()?;
        let lexical = policy.normalize_sandbox_path(path, for_write)?;
        let mode = if recursive_remove {
            "2"
        } else if for_write {
            "1"
        } else {
            "0"
        };
        let helper_path = self.ensure_runtime_helper_installed(&HELPER).await?;
        let mut argv = vec![
            helper_path,
            policy.sandbox_root().to_string(),
            lexical.clone(),
            mode.to_string(),
        ];
        for (grant_path, read_only) in policy.extra_path_grant_rules()? {
            argv.push(grant_path);
            argv.push(if read_only { "1" } else { "0" }.to_string());
        }
        let result = self
            .exec(ExecRequest::argv(argv.clone()).with_shell(ExecShell::None))
            .await?;
        let stderr = result.stderr_text();
        match result.exit_code {
            0 if !result.stdout_text().trim().is_empty() => Ok(lexical),
            0 => Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "the path resolution helper returned nothing",
            )
            .with_context("reason", "empty_stdout")),
            111 => Err(SandboxError::new(
                ErrorCode::InvalidManifestPath,
                format!("manifest path must not escape root: {path}"),
            )
            .with_context("rel", path.to_string())
            .with_context(
                "reason",
                if path.starts_with('/') {
                    "absolute"
                } else {
                    "escape_root"
                },
            )
            .with_context("resolved_path", stderr.trim().to_string())),
            113 => Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                stderr.trim().to_string(),
            )),
            114 => {
                let field = |label: &str| {
                    stderr
                        .lines()
                        .find_map(|line| line.strip_prefix(label))
                        .map(|value| value.trim().to_string())
                };
                let mut error = SandboxError::new(
                    ErrorCode::WorkspaceArchiveWriteError,
                    format!("failed to write archive for path: {lexical}"),
                )
                .with_context("path", lexical.clone())
                .with_context("reason", "read_only_extra_path_grant");
                if let Some(grant) = field("read-only extra path grant: ") {
                    error = error.with_context("grant_path", grant);
                }
                if let Some(resolved) = field("resolved path: ") {
                    error = error.with_context("resolved_path", resolved);
                }
                Err(error)
            }
            _ => Err(SandboxError::exec_nonzero(&argv, &result)),
        }
    }

    /// Whether `rm`, `ls` and `mv` resolve every path inside the sandbox first.
    ///
    /// Mirrors Python's base class, which runs `_validate_remote_path_access` for those
    /// operations so a workspace symlink cannot lead a lexically safe path out of the root. The
    /// default is `false` because the resolver costs extra `exec` round trips and local
    /// backends check through the real filesystem already; a provider whose files live behind an
    /// API turns it on.
    fn resolves_paths_remotely(&self) -> bool {
        false
    }

    /// A provider's veto on applying `manifest` to this live (injected) session.
    ///
    /// `session_running` is the session's running state when the caller already asked for it;
    /// `None` leaves the provider to ask. Port of Python's `_validate_manifest_application`,
    /// which `SandboxRuntimeSessionManager` runs before it processes a live manifest. The
    /// default accepts everything.
    async fn validate_manifest_application(
        &self,
        _manifest: &Manifest,
        _session_running: Option<bool>,
    ) -> SandboxResult<()> {
        Ok(())
    }

    /// Whether any extra path grant is read-only, which is what makes a recursive removal need
    /// the in-sandbox resolver.
    fn has_read_only_grant(&self) -> bool {
        self.workspace_path_policy()
            .and_then(|policy| policy.extra_path_grant_rules())
            .map(|rules| rules.iter().any(|(_, read_only)| *read_only))
            .unwrap_or(false)
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
    /// Failures are payload-free. A missing file is `WorkspaceReadNotFound`; every other failure
    /// becomes a `WorkspaceArchiveReadError` whose context holds only the path and a reason
    /// (`bounded_read_failed`, or `bounded_read_wire_limit` for a transport that refused an
    /// over-long encoded response), and whose retryability is the provider's own. What the
    /// provider said, including any response text, is dropped at this boundary.
    ///
    /// Backends override [`read_bounded_unchecked`](Self::read_bounded_unchecked), not this.
    async fn read_bounded(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if max_bytes == 0 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "read_bounded max_bytes must be positive",
            ));
        }
        match self.read_bounded_unchecked(path, max_bytes).await {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("Workspace path `{path}` was not found"),
            )
            .with_context("path", path.to_string())),
            Err(error) => {
                let reason = if error.context.get("reason").map(String::as_str)
                    == Some("bounded_read_wire_limit")
                {
                    "bounded_read_wire_limit"
                } else {
                    "bounded_read_failed"
                };
                let mut sanitized = SandboxError::new(
                    ErrorCode::WorkspaceArchiveReadError,
                    format!("Failed to read workspace path `{path}`"),
                )
                .with_context("path", path.to_string())
                .with_context("reason", reason);
                sanitized.retryable = error.retryable;
                Err(sanitized)
            }
        }
    }

    /// The backend hook behind [`read_bounded`](Self::read_bounded).
    ///
    /// The default reads through [`read_file`](Self::read_file) and truncates, which bounds what
    /// the *caller* holds but not what the backend fetched. A backend whose transport can bound
    /// the fetch should override this, and report a transport that refuses the request with
    /// `context["reason"] = "bounded_read_wire_limit"` so the caller can tell the two apart.
    /// `max_bytes` is already known to be positive.
    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
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

    /// Whether `aclose` still shuts the session down after the snapshot capture or `stop`
    /// failed. Backends that hold mount credentials in the sandbox return `true`, so a failed
    /// close does not leave them running.
    fn shutdown_after_failed_close(&self) -> bool {
        false
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

    /// Where this session keeps the paths registered with
    /// [`register_persist_workspace_skip_path`](Self::register_persist_workspace_skip_path).
    ///
    /// A backend that wants registrations to take effect returns a set it owns; the default
    /// `None` accepts a registration and forgets it, which is what a backend whose archive is
    /// not built from this workspace (a provider-native snapshot) has to do anyway.
    fn persist_skip_registry(
        &self,
    ) -> Option<&std::sync::Mutex<std::collections::BTreeSet<String>>> {
        None
    }

    /// The workspace-relative paths a workspace archive leaves out, beyond the manifest's own
    /// ephemeral entries.
    fn persist_workspace_skip_paths(&self) -> Vec<String> {
        self.persist_skip_registry()
            .map(|registry| {
                registry
                    .lock()
                    .expect("persist skip registry poisoned")
                    .iter()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Keep a runtime-written path (an event outbox, say) out of workspace archives.
    ///
    /// Mirrors Python's `register_persist_workspace_skip_path`. The path is workspace-relative,
    /// and a path that overlaps a mount is refused: skipping a mount's ancestor would drop the
    /// mount's own persistence rules, and skipping something inside one is a no-op that hides a
    /// misunderstanding. Returns the normalized relative path.
    fn register_persist_workspace_skip_path(&self, path: &str) -> SandboxResult<String> {
        let relative = self.workspace_path_policy()?.relative_path(path)?;
        let root = self.workspace_root().trim_end_matches('/').to_string();
        for (_, mount_path) in self.manifest().mount_targets() {
            let mount_relative = mount_path
                .strip_prefix(&root)
                .unwrap_or(&mount_path)
                .trim_start_matches('/')
                .to_string();
            let overlaps = relative == mount_relative
                || mount_relative.starts_with(&format!("{relative}/"))
                || relative.starts_with(&format!("{mount_relative}/"));
            if overlaps {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "persist workspace skip path must not overlap mount path",
                )
                .with_context("path", relative)
                .with_context("mount_path", mount_path));
            }
        }
        if let Some(registry) = self.persist_skip_registry() {
            registry
                .lock()
                .expect("persist skip registry poisoned")
                .insert(relative.clone());
        }
        Ok(relative)
    }

    /// Check that `path` is readable by `user`, and say why not.
    ///
    /// Mirrors Python's `_check_read_with_exec`. Returns the normalized path when it is
    /// readable. A failed check is ambiguous (a missing file and an unreadable one both fail), so
    /// the existence probe decides between [`ErrorCode::WorkspaceReadNotFound`] and
    /// [`ErrorCode::WorkspaceArchiveReadError`]. A check that exits with anything but 1 is the
    /// shell itself failing, which no probe can explain, so it is reported without one.
    async fn check_read_access(&self, path: &str, user: Option<&User>) -> SandboxResult<String> {
        let resolved = self.normalize_path(path)?;
        let argv = vec![
            "sh".to_string(),
            "-c".to_string(),
            "[ -r \"$1\" ]".to_string(),
            "sh".to_string(),
            resolved.clone(),
        ];
        let result = self
            .exec(
                ExecRequest::argv(argv)
                    .with_shell(ExecShell::None)
                    .with_user(user.cloned()),
            )
            .await?;
        if result.ok() {
            return Ok(resolved);
        }
        Err(self.read_error_from_exec(&resolved, &result, user).await)
    }

    /// Classify a failed read of `path`. Mirrors Python's `_raise_read_error_from_exec`.
    ///
    /// The error context records the sizes of what the command printed but never the partial
    /// stdout itself: it can be file content the caller was not allowed to see in a log.
    async fn read_error_from_exec(
        &self,
        path: &str,
        result: &ExecResult,
        user: Option<&User>,
    ) -> SandboxError {
        let stderr = safe_decode(&result.stderr, 4096);
        let archive_error = |message: String| {
            SandboxError::new(ErrorCode::WorkspaceArchiveReadError, message)
                .with_op(SandboxOp::Read)
                .with_context("path", path.to_string())
                .with_context("stdout_bytes", result.stdout.len().to_string())
                .with_context("stderr", stderr.clone())
        };
        if result.exit_code != 1 {
            return archive_error(format!("failed to read archive: {path}"));
        }
        let probe = self
            .exec(
                ExecRequest::argv(["sh", "-c", READ_PATH_PROBE_SCRIPT, "sh", path])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(ACCESS_PROBE_TIMEOUT))
                    .with_user(user.cloned()),
            )
            .await;
        let probe = match probe {
            Ok(probe) => probe,
            Err(error) => {
                return archive_error(format!("failed to read archive: {path}")).caused_by(&error);
            }
        };
        if probe.exit_code == 1 {
            return SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("workspace path not found: {path}"),
            )
            .with_op(SandboxOp::Read)
            .with_context("path", path.to_string())
            .with_context("stdout_bytes", result.stdout.len().to_string())
            .with_context("stderr", stderr)
            .with_context("existence_probe_exit_code", probe.exit_code.to_string());
        }
        archive_error(format!("failed to read archive: {path}"))
            .with_context("existence_probe_exit_code", probe.exit_code.to_string())
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

/// Decode `bytes` for a log field, cutting it at `max_chars` and marking the cut with `…`.
///
/// Mirrors Python's `_safe_decode`. Invalid UTF-8 is replaced rather than failing, because the
/// bytes are diagnostics from a process the SDK does not control.
pub fn safe_decode(bytes: &[u8], max_chars: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.chars().count() <= max_chars {
        return text.into_owned();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// The directory an absolute sandbox path lives in.
fn parent_sandbox_path(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rsplit_once('/') {
        Some(("", _)) => "/".to_string(),
        Some((parent, _)) => parent.to_string(),
        None => ".".to_string(),
    }
}

/// Append `part` to an absolute sandbox path.
fn join_sandbox_path(base: &str, part: &str) -> String {
    if part.is_empty() {
        return base.to_string();
    }
    format!("{}/{}", base.trim_end_matches('/'), part)
}

/// The components of a validated archive member path.
fn sandbox_rel_parts(rel_path: &std::path::Path) -> Vec<String> {
    rel_path
        .iter()
        .map(|part| part.to_string_lossy().into_owned())
        .collect()
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
    let mut skipped = workspace_fingerprint_skip_relpaths(manifest, &root);
    // The runtime-registered skip paths are left out of the fingerprint as they are left out of
    // the archive, or a registered outbox would make every workspace look modified.
    skipped.extend(session.persist_workspace_skip_paths());
    skipped.sort();
    skipped.dedup();
    command.extend(skipped);
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
    // A mount at the workspace root makes the whole workspace the mount. The relative-path list
    // drops the empty relpath, so the root mount is detected from the resolved targets.
    let trimmed_root = root.trim_end_matches('/');
    let root_is_mounted = session
        .manifest()
        .ephemeral_mount_targets()
        .iter()
        .any(|(_, path)| path.trim_end_matches('/') == trimmed_root);
    if root_is_mounted || skip.iter().any(|relative| relative == ".") {
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

/// `tar --exclude=...` arguments that skip `skip_relpaths` in an archive rooted at `.`.
///
/// Paths are sorted by their POSIX form and anchored with `./`, because a bare pattern also
/// matches unrelated nested paths of the same name. Empty and `.` paths are dropped, and an
/// absolute path is treated as workspace-relative. Mirrors `shell_tar_exclude_args`.
pub fn shell_tar_exclude_args<I, P>(skip_relpaths: I) -> Vec<String>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut posix: Vec<String> = skip_relpaths
        .into_iter()
        .map(|path| path.as_ref().to_string_lossy().replace('\\', "/"))
        .collect();
    posix.sort();
    posix
        .iter()
        .map(|path| path.trim_start_matches('/'))
        .filter(|path| !path.is_empty() && *path != ".")
        .map(|path| format!("--exclude={}", shell_quote(&format!("./{path}"))))
        .collect()
}

/// Capture a session's workspace as a gzipped tar through the backend's own `tar`.
///
/// This is the body of the default [`SandboxSession::persist_workspace_once`], extracted so a
/// backend that post-processes its archive — the ones whose workspace can hold a symlink
/// recorded against the host root, see
/// [`rebase_archive_symlink_targets`](crate::sandbox::archive::rebase_archive_symlink_targets) —
/// can capture it exactly the same way.
pub async fn capture_workspace_tar<S>(session: &S) -> SandboxResult<Vec<u8>>
where
    S: SandboxSession + ?Sized,
{
    let root = session.workspace_root().to_string();
    let mut command = format!("tar -czf - -C {}", shell_quote(&root));
    for exclude in workspace_archive_excludes(session.manifest(), &root) {
        command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
    }
    for skipped in session.persist_workspace_skip_paths() {
        command.push_str(&format!(
            " --exclude {}",
            shell_quote(&format!("./{skipped}"))
        ));
    }
    command.push_str(" .");
    // A plain `sh -c`, never a login shell: the workspace is where the model writes files, and a
    // login shell reads startup files (`.profile`, ...) from the user's home, which can be that
    // workspace. The SDK's own archive command must not run what the model put there.
    let result = session
        .exec(ExecRequest::argv(["sh", "-c", command.as_str()]).with_shell(ExecShell::None))
        .await?;
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

/// Run `operation` with the given mounts detached, then re-attach them.
///
/// Mirrors Python's `with_ephemeral_mounts_removed`. The rules it exists to keep:
///
/// - Mounts are detached in order and re-attached in reverse, so nested mounts unwind the way
///   they were stacked.
/// - A failed detach stops the sequence and skips `operation`, but the mounts that did detach are
///   put back before the error is returned. A detach that failed part-way leaves the session in
///   an unknown state, so it is shut down once the others are restored.
/// - A failed `operation` still re-attaches. If re-attaching fails too, the restore error wins,
///   because a workspace with a missing mount is the worse outcome; the operation's own message
///   is kept on it under `operation_error_context_key`, and the session is shut down.
/// - `restore_on_success` false leaves the mounts detached after a successful `operation`, for a
///   caller that is about to tear the session down anyway.
///
/// `error_code` is the error class for transition failures (an archive read or write, or a start
/// failure). Cancellation needs no special handling here: dropping the future stops it, and the
/// caller's own guard decides what to restore.
pub async fn with_ephemeral_mounts_removed<T, F, Fut>(
    session: &dyn SandboxSession,
    mounts: Vec<(&Mount, String)>,
    operation: F,
    error_path: &str,
    error_code: ErrorCode,
    operation_error_context_key: Option<&str>,
    restore_on_success: bool,
) -> SandboxResult<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = SandboxResult<T>>,
{
    let transition_error = |cause: &SandboxError| {
        SandboxError::new(
            error_code,
            format!("failed to detach or restore mounts for {error_path}"),
        )
        .with_context("path", error_path.to_string())
        .caused_by(cause)
        .with_context("cause", cause.message.clone())
    };

    let mut detached: Vec<(&Mount, &str)> = Vec::new();
    let mut detach_error: Option<SandboxError> = None;
    for (mount, path) in &mounts {
        match mount
            .strategy
            .teardown_for_snapshot(mount, session, path)
            .await
        {
            Ok(()) => detached.push((mount, path.as_str())),
            Err(error) => {
                detach_error = Some(transition_error(&error));
                break;
            }
        }
    }

    let mut operation_result: Option<SandboxResult<T>> = None;
    if detach_error.is_none() {
        operation_result = Some(operation().await);
    }

    let operation_ok = matches!(operation_result, Some(Ok(_)));
    let should_restore = (operation_ok && restore_on_success)
        || detach_error.is_some()
        || matches!(operation_result, Some(Err(_)));
    let mut restore_error: Option<SandboxError> = None;
    if should_restore {
        let mut additional: Vec<String> = Vec::new();
        for (mount, path) in detached.iter().rev() {
            if let Err(error) = mount
                .strategy
                .restore_after_snapshot(mount, session, path)
                .await
            {
                let current = transition_error(&error);
                match &restore_error {
                    None => restore_error = Some(current),
                    Some(_) => additional.push(current.message),
                }
            }
        }
        if let Some(error) = restore_error.as_mut()
            && !additional.is_empty()
        {
            error
                .context
                .insert("additional_remount_errors".into(), additional.join("; "));
        }
        // A restore that failed leaves the workspace in an unknown state.
        if let Some(error) = restore_error.as_mut()
            && session.shutdown().await.is_err()
        {
            error
                .context
                .insert("terminal_cleanup_failed".into(), "true".into());
        }
    }
    if detach_error.is_some()
        && restore_error.is_none()
        && let Err(_) = session.shutdown().await
        && let Some(error) = detach_error.as_mut()
    {
        error
            .context
            .insert("terminal_cleanup_failed".into(), "true".into());
    }

    if let Some(mut error) = restore_error {
        if let (Some(Err(operation_error)), Some(key)) =
            (&operation_result, operation_error_context_key)
        {
            error
                .context
                .insert(key.to_string(), operation_error.message.clone());
        }
        return Err(error);
    }
    if let Some(error) = detach_error {
        return Err(error);
    }
    operation_result.expect("the operation ran when no detach failed")
}

/// Capture a session's workspace into its snapshot.
///
/// Mounts whose strategy cannot be snapshotted in place are detached first and re-attached
/// afterwards, mirroring Python's `teardown_for_snapshot` / `restore_after_snapshot` pair.
/// Sessions without a configured snapshot are a no-op.
pub async fn persist_session_snapshot(session: &dyn SandboxSession) -> SandboxResult<()> {
    // Before any snapshot work, as Python's `stop()` does: refusing an unsafe mount after some
    // mounts were already detached would leave the workspace half-way between two states.
    session.manifest().validate_mount_credential_boundaries()?;
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
    let root = session.workspace_root().to_string();
    let archive = with_ephemeral_mounts_removed(
        session,
        detached,
        || session.persist_workspace(),
        &root,
        ErrorCode::WorkspaceArchiveReadError,
        Some("snapshot_error_before_remount_corruption"),
        true,
    )
    .await;
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
    let archive = snapshot
        .restore_with(session.dependencies().map(Arc::as_ref))
        .await?;
    // An older snapshot may hold members this SDK refuses. A backend that unpacks a tar rejects
    // them before the live files are discarded: `validate_restorable_archive` is a no-op for the
    // backends whose archive is a provider-native snapshot reference rather than a tar.
    session.validate_restorable_archive(&archive)?;
    // Everything the archive does not contain has to go, or a file the paused run deleted comes
    // back. The live mounts are the exception: under them is remote storage, not workspace.
    clear_workspace_root_on_resume(session).await?;
    session.hydrate_workspace(&archive).await?;
    // A fresh sandbox has none of the manifest's accounts, and the archive restores files, not
    // users, so they are provisioned before the ephemeral entries that may be owned by them.
    if !session.system_state_preserved_on_start() {
        manifest.provision_accounts(session).await?;
    }
    manifest.apply_ephemeral(session).await?;
    Ok(true)
}

/// Move `source` to `destination` through `exec`, the fallback every backend gets.
async fn mv_via_exec<S: SandboxSession + ?Sized>(
    session: &S,
    source: &str,
    destination: &str,
    user: Option<&User>,
) -> SandboxResult<()> {
    // A move writes at both ends: it unlinks the source and creates the destination.
    let (source, destination) = if session.resolves_paths_remotely() {
        (
            session
                .validate_remote_path_access(source, true, false)
                .await?,
            session
                .validate_remote_path_access(destination, true, false)
                .await?,
        )
    } else {
        (
            session.normalize_path_for_write(source)?,
            session.normalize_path_for_write(destination)?,
        )
    };
    let script = "if [ ! -L \"$1\" ] && [ ! -f \"$1\" ]; then \
         printf \"%s\\n\" \"Move source must be a regular file or symlink\" >&2; exit 1; fi; \
         exec mv -fT -- \"$1\" \"$2\"";
    let argv = vec![
        "sh".to_string(),
        "-c".to_string(),
        script.to_string(),
        "sh".to_string(),
        source.clone(),
        destination.clone(),
    ];
    let result = session
        .exec(
            ExecRequest::argv(argv)
                .with_shell(ExecShell::None)
                .with_user(user.cloned()),
        )
        .await?;
    if !result.ok() {
        // Report a placeholder for the script, the way Python does, so the message names the
        // operation rather than a screenful of shell.
        let shown = vec![
            "sh".to_string(),
            "-c".to_string(),
            "<mv>".to_string(),
            source,
            destination,
        ];
        return Err(SandboxError::exec_nonzero(&shown, &result));
    }
    Ok(())
}

/// Creates and releases [`SandboxSession`]s for one backend.
#[async_trait]
pub trait SandboxClient: Send + Sync + fmt::Debug {
    /// The backend identifier (`unix_local`, `docker`, ...).
    fn backend_id(&self) -> &str;

    /// Create a fresh session that will materialize `manifest` when it starts.
    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>>;

    /// The snapshot this client was built with, when its owner chose one.
    ///
    /// Python has no client-level snapshot: `create(snapshot=...)` is the only source, and the
    /// runner fills it with the SDK default when the run config names none. A Rust client can be
    /// configured with one up front, so the runner asks here before falling back to that default;
    /// an explicit run-config snapshot still wins over both, as the `create` argument does.
    fn configured_snapshot(&self) -> Option<crate::sandbox::snapshot::SnapshotSpec> {
        None
    }

    /// Create a session whose workspace snapshots are stored at `snapshot`.
    ///
    /// Mirrors Python's `client.create(snapshot=...)`. The runner resolves the SDK default local
    /// snapshot and passes it here. A backend that stores snapshots itself overrides this; the
    /// default ignores the spec and calls [`create`](Self::create).
    async fn create_with_snapshot(
        &self,
        manifest: Option<Manifest>,
        snapshot: crate::sandbox::snapshot::SnapshotSpec,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        let _ = snapshot;
        self.create(manifest).await
    }

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
