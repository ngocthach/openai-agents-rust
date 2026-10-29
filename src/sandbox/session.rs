//! The sandbox session and client interfaces. Mirrors
//! `agents.sandbox.session.base_sandbox_session` and `agents.sandbox.session.sandbox_client`.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::types::{ExecResult, FileEntry, User, parse_ls_la};
use crate::sandbox::workspace_paths::resolve_workspace_path;

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

    /// Whether the session can allocate a PTY. The Rust port never does.
    fn supports_pty(&self) -> bool {
        false
    }

    /// Normalize a model-provided path to an absolute sandbox path inside the workspace root.
    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        resolve_workspace_path(self.workspace_root(), path)
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
        let command = format!("tar -czf - -C {} .", shell_quote(&root));
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
    fn quotes_arguments() {
        assert_eq!(shell_quote("plain-arg_1.txt"), "plain-arg_1.txt");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote(""), "''");
    }
}
