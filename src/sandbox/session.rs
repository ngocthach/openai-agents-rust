//! The sandbox session and client interfaces. Mirrors
//! `agents.sandbox.session.base_sandbox_session` and `agents.sandbox.session.sandbox_client`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mounts::Mount;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::snapshot::{Snapshot, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::{ExecResult, FileEntry, User, parse_ls_la};
use crate::sandbox::workspace_paths::{is_provider_scratch_path, resolve_workspace_path};

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

    /// Whether the session can run commands in a PTY (`exec_command` / `write_stdin`).
    fn supports_pty(&self) -> bool {
        false
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
    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        resolve_workspace_path(self.workspace_root(), path)
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
        let resolved = self.normalize_path(path)?;
        let mut command = vec!["mkdir".to_string()];
        if parents {
            command.push("-p".to_string());
        }
        command.push("--".to_string());
        command.push(resolved);
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    /// Remove a file or directory.
    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        let mut command = vec!["rm".to_string()];
        command.push(if recursive { "-rf" } else { "-f" }.to_string());
        command.push("--".to_string());
        command.push(resolved);
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    /// Move or rename a workspace entry.
    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let from = self.normalize_path(source)?;
        let to = self.normalize_path(destination)?;
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
}

/// The `tar --exclude` patterns every workspace archive applies.
///
/// Two kinds of path must never reach an archive:
///
/// - The manifest's ephemeral entries. Mounts and the like are runtime-attached state, never
///   durable workspace content; archiving them would copy whole mounted buckets into the
///   snapshot.
/// - The staging directory a mount writes its credentials to
///   ([`RCLONE_CONFIG_DIR`](crate::sandbox::mounts::RCLONE_CONFIG_DIR)). It lives inside the
///   workspace because the session's file API cannot write outside it, so the archive has to
///   leave it out explicitly.
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
    let manifest = session.manifest().clone();
    let detached: Vec<(&Mount, String)> = manifest
        .mount_targets()
        .into_iter()
        .filter(|(mount, _)| !mount.strategy.supports_native_snapshot_detach())
        .collect();
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
    snapshot.persist(&archive?).await
}

/// Restore a session's workspace from its snapshot and re-apply the ephemeral entries.
///
/// Returns `false` when no archive is stored, leaving the workspace untouched.
pub async fn restore_session_snapshot(session: &dyn SandboxSession) -> SandboxResult<bool> {
    let snapshot = session.snapshot();
    if !snapshot.restorable().await? {
        return Ok(false);
    }
    let archive = snapshot.restore().await?;
    session.hydrate_workspace(&archive).await?;
    let manifest = session.manifest().clone();
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
