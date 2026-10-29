//! A sandbox session backed by a Docker container, driven through the `docker` CLI. Mirrors
//! `agents.sandbox.sandboxes.docker`.
//!
//! The Python SDK talks to the Docker daemon through `docker-py`. This port shells out to the
//! `docker` executable instead, so it needs no extra dependency: `docker create`, `docker start`,
//! `docker exec` and `docker rm` cover the whole session lifecycle.

use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot, shell_quote,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::ExecResult;

/// The default image used when none is configured, matching the Python default.
pub const DEFAULT_PYTHON_SANDBOX_IMAGE: &str = "python:3.14-slim";

/// Settings used when creating a Docker-backed session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerSandboxClientOptions {
    /// The image the container runs.
    pub image: String,
    /// Disable container networking (`--network none`).
    pub network_disabled: bool,
    /// Extra arguments inserted before the image in `docker create`.
    pub extra_create_args: Vec<String>,
}

impl Default for DockerSandboxClientOptions {
    fn default() -> Self {
        DockerSandboxClientOptions {
            image: DEFAULT_PYTHON_SANDBOX_IMAGE.to_string(),
            network_disabled: false,
            extra_create_args: Vec::new(),
        }
    }
}

impl DockerSandboxClientOptions {
    /// Options for the given image.
    pub fn new(image: impl Into<String>) -> Self {
        DockerSandboxClientOptions {
            image: image.into(),
            ..Default::default()
        }
    }

    /// Disable container networking.
    pub fn with_network_disabled(mut self, disabled: bool) -> Self {
        self.network_disabled = disabled;
        self
    }

    /// Add an extra `docker create` argument.
    pub fn with_create_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_create_args.push(arg.into());
        self
    }
}

/// Creates [`DockerSandboxSession`]s with the `docker` CLI.
#[derive(Debug, Clone, Default)]
pub struct DockerSandboxClient {
    options: DockerSandboxClientOptions,
    snapshot: Option<SnapshotSpec>,
}

impl DockerSandboxClient {
    /// A client using the default image.
    pub fn new() -> Self {
        DockerSandboxClient::default()
    }

    /// A client using explicit options.
    pub fn with_options(options: DockerSandboxClientOptions) -> Self {
        DockerSandboxClient {
            options,
            snapshot: None,
        }
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Whether a usable Docker daemon is reachable (`docker info`).
    pub async fn is_available() -> bool {
        run_docker(
            &[
                "info".to_string(),
                "--format".to_string(),
                "{{.ID}}".to_string(),
            ],
            None,
        )
        .await
        .map(|o| o.exit_code == 0)
        .unwrap_or(false)
    }

    /// Create a container for `manifest` and return the concrete session.
    async fn create_session(&self, manifest: Manifest) -> SandboxResult<DockerSandboxSession> {
        let mut argv = vec![
            "create".to_string(),
            "--label".to_string(),
            "openai-agents-sandbox=1".to_string(),
            "-w".to_string(),
            manifest.root().to_string(),
        ];
        if self.options.network_disabled {
            argv.push("--network".to_string());
            argv.push("none".to_string());
        }
        // Bind and volume mounts are attached by the container runtime, before the session
        // starts, so they are resolved from the manifest here rather than during materialization.
        for (mount, mount_path) in manifest.mount_targets() {
            mount.validate()?;
            if let Some(args) = mount.strategy.docker_create_args(mount, &mount_path) {
                argv.extend(args);
            }
        }
        argv.extend(self.options.extra_create_args.iter().cloned());
        argv.push(self.options.image.clone());
        // Keep PID 1 alive so `docker exec` has something to attach to.
        argv.extend([
            "sh".to_string(),
            "-c".to_string(),
            "while :; do sleep 3600; done".to_string(),
        ]);
        let output = run_docker(&argv, None).await?;
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "`docker create` failed for image `{}`: {}",
                    self.options.image,
                    output.stderr_text().trim()
                ),
            ));
        }
        let container_id = output.stdout_text().trim().to_string();
        if container_id.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                "`docker create` returned no container id",
            ));
        }
        Ok(DockerSandboxSession::new(container_id, manifest).with_snapshot(self.snapshot.clone()))
    }

    /// Whether `image` is already present locally (`docker image inspect`).
    ///
    /// Tests should check this before creating a container: `docker create` pulls a missing
    /// image, which needs the network.
    pub async fn image_exists(image: &str) -> bool {
        run_docker(
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
            None,
        )
        .await
        .map(|o| o.exit_code == 0)
        .unwrap_or(false)
    }
}

#[async_trait]
impl SandboxClient for DockerSandboxClient {
    fn backend_id(&self) -> &str {
        "docker"
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        Ok(Arc::new(
            self.create_session(manifest.unwrap_or_default()).await?,
        ))
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        if state.backend != "docker" {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "sandbox session state was created by the `{}` backend",
                    state.backend
                ),
            ));
        }
        state.verify_manifest(&manifest)?;
        let manifest = manifest.with_root(state.workspace_root.clone());
        let container_id = state.provider_id.clone().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "docker sandbox session state is missing its container id",
            )
        })?;
        let exists = run_docker(
            &[
                "container".to_string(),
                "inspect".to_string(),
                container_id.clone(),
            ],
            None,
        )
        .await
        .map(|o| o.exit_code == 0)
        .unwrap_or(false);
        if exists {
            return Ok(Arc::new(
                DockerSandboxSession::new(container_id, manifest)
                    .with_snapshot(Some(state.snapshot.clone()))
                    .with_snapshot_id(state.snapshot_id.clone())
                    .resuming(),
            ));
        }
        // The original container is gone: create a replacement and hydrate it from the snapshot.
        let replacement = self.create_session(manifest).await?;
        Ok(Arc::new(
            replacement
                .with_snapshot(Some(state.snapshot.clone()))
                .with_snapshot_id(state.snapshot_id.clone())
                .resuming(),
        ))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// A sandbox session whose workspace lives inside a Docker container.
#[derive(Debug)]
pub struct DockerSandboxSession {
    container_id: String,
    manifest: Manifest,
    started: AtomicBool,
    removed: Mutex<bool>,
    ptys: PtyManager,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
}

impl DockerSandboxSession {
    /// Attach a session to an existing container.
    pub fn new(container_id: impl Into<String>, manifest: Manifest) -> Self {
        let container_id = container_id.into();
        DockerSandboxSession {
            snapshot_id: container_id.clone(),
            container_id,
            manifest,
            started: AtomicBool::new(false),
            removed: Mutex::new(false),
            ptys: PtyManager::new(),
            snapshot: None,
            resuming: false,
        }
    }

    /// Store workspace snapshots at `snapshot`.
    pub fn with_snapshot(mut self, snapshot: Option<SnapshotSpec>) -> Self {
        self.snapshot = snapshot;
        self
    }

    /// Reuse a persisted snapshot identity.
    pub fn with_snapshot_id(mut self, snapshot_id: String) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Restore the snapshot on start instead of materializing the full manifest.
    pub fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// The container backing the workspace.
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    fn exec_argv(&self, request: &ExecRequest) -> SandboxResult<Vec<String>> {
        let mut argv = vec!["exec".to_string()];
        if let Some(user) = &request.user {
            argv.push("-u".to_string());
            argv.push(user.name.clone());
        }
        let workdir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        argv.push("-w".to_string());
        argv.push(workdir);
        for (key, value) in request.resolved_env(self.manifest.environment()) {
            argv.push("-e".to_string());
            argv.push(format!("{key}={value}"));
        }
        argv.push(self.container_id.clone());
        argv.extend(request.resolved_argv());
        Ok(argv)
    }

    /// Whether the container workspace still holds anything.
    async fn workspace_has_content(&self) -> bool {
        let root = shell_quote(self.manifest.root());
        match self
            .exec(ExecRequest::shell(format!(
                "[ -n \"$(ls -A {root} 2>/dev/null)\" ]"
            )))
            .await
        {
            Ok(result) => result.ok(),
            Err(_) => false,
        }
    }

    /// Kill every PTY process, including the container-side process each one proxies.
    async fn terminate_pty_sessions(&self) {
        self.run_pty_cleanup(self.ptys.terminate_all()).await;
    }

    /// Run the container-side cleanup of PTY processes that finished or were pruned.
    async fn run_pty_cleanup(&self, commands: Vec<String>) {
        for command in commands {
            let _ = self.exec(ExecRequest::shell(command)).await;
        }
    }
}

#[async_trait]
impl SandboxSession for DockerSandboxSession {
    fn id(&self) -> &str {
        &self.container_id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "docker"
    }

    async fn start(&self) -> SandboxResult<()> {
        let output = run_docker(&["start".to_string(), self.container_id.clone()], None).await?;
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "`docker start` failed for `{}`: {}",
                    self.container_id,
                    output.stderr_text().trim()
                ),
            ));
        }
        self.started.store(true, Ordering::SeqCst);
        let result = if self.resuming {
            match restore_session_snapshot(self).await {
                Ok(true) => Ok(()),
                // Without a stored archive, the workspace the paused run left behind is all
                // there is. Re-apply only the ephemeral entries when it survived; when it is
                // gone (an evicted temp directory, a replacement container, a snapshot-less
                // backend) materialize the whole manifest instead of handing the model an
                // empty workspace.
                Ok(false) => {
                    let preserved = self.workspace_has_content().await;
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
            self.started.store(false, Ordering::SeqCst);
        })?;
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        self.terminate_pty_sessions().await;
        self.manifest.unmount_all(self).await?;
        let output = run_docker(
            &[
                "stop".to_string(),
                "-t".to_string(),
                "1".to_string(),
                self.container_id.clone(),
            ],
            None,
        )
        .await?;
        self.started.store(false, Ordering::SeqCst);
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                format!(
                    "`docker stop` failed for `{}`: {}",
                    self.container_id,
                    output.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.terminate_pty_sessions().await;
        // In-container mounts have to be released while the container is still alive. Bind and
        // volume mounts are detached by the runtime when the container goes away.
        if let Err(error) = self.manifest.unmount_all(self).await {
            log::warn!("Failed to detach a container mount before removal: {error}");
        }
        let mut removed = self.removed.lock().await;
        if *removed {
            return Ok(());
        }
        let output = run_docker(
            &[
                "rm".to_string(),
                "-f".to_string(),
                self.container_id.clone(),
            ],
            None,
        )
        .await?;
        self.started.store(false, Ordering::SeqCst);
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                format!(
                    "`docker rm` failed for `{}`: {}",
                    self.container_id,
                    output.stderr_text().trim()
                ),
            ));
        }
        *removed = true;
        Ok(())
    }

    fn supports_pty(&self) -> bool {
        true
    }

    fn attaches_mounts_at_create(&self) -> bool {
        true
    }

    fn session_state(&self) -> SandboxSessionState {
        SandboxSessionState::new(
            "docker",
            &self.container_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.container_id.clone()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let workdir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        // The host-side `docker exec` process is only a proxy: killing it leaves the process
        // inside the container running, so the command records its pid and the session kills it
        // in the container during teardown.
        let pid_path = format!(
            "/tmp/openai-agents-sandbox-pty-{}.pid",
            uuid::Uuid::new_v4().simple()
        );
        let inner = format!(
            "echo $$ > {pid}; exec {argv}",
            pid = shell_quote(&pid_path),
            argv = request
                .resolved_argv()
                .iter()
                .map(|a| shell_quote(a))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let mut argv = vec!["docker".to_string(), "exec".to_string(), "-i".to_string()];
        if request.tty {
            argv.push("-t".to_string());
        }
        argv.push("-w".to_string());
        argv.push(workdir);
        for (key, value) in self.manifest.environment() {
            argv.push("-e".to_string());
            argv.push(format!("{key}={value}"));
        }
        argv.push(self.container_id.clone());
        argv.extend(["sh".to_string(), "-c".to_string(), inner]);

        let spawn = PtySpawn {
            argv,
            cwd: None,
            env: Vec::new(),
            clear_env: false,
            tty: request.tty,
            remote_kill_command: Some(format!(
                "if [ -f {pid} ]; then kill -TERM \"$(cat {pid})\" 2>/dev/null || true; \
                 rm -f {pid}; fi",
                pid = shell_quote(&pid_path)
            )),
        };
        let (update, cleanup) = self
            .ptys
            .start(spawn, request.yield_time_ms, request.max_output_tokens)
            .await?;
        self.run_pty_cleanup(cleanup).await;
        Ok(update)
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let (update, cleanup) = self.ptys.write_stdin(&request).await?;
        self.run_pty_cleanup(cleanup).await;
        Ok(update)
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        self.terminate_pty_sessions().await;
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        let output = run_docker(
            &[
                "inspect".to_string(),
                "-f".to_string(),
                "{{.State.Running}}".to_string(),
                self.container_id.clone(),
            ],
            None,
        )
        .await?;
        Ok(output.exit_code == 0 && output.stdout_text().trim() == "true")
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let argv = self.exec_argv(&request)?;
        match request.timeout {
            None => run_docker(&argv, None).await,
            Some(timeout) => match tokio::time::timeout(timeout, run_docker(&argv, None)).await {
                Ok(result) => result,
                Err(_) => Err(SandboxError::new(
                    ErrorCode::ExecTimeout,
                    format!(
                        "Command `{}` timed out after {timeout:?}",
                        request.command.join(" ")
                    ),
                )),
            },
        }
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        let result = self
            .exec(ExecRequest::argv(["cat", "--", &resolved]))
            .await?;
        if !result.ok() {
            let code = if result.stderr_text().contains("No such file") {
                ErrorCode::WorkspaceReadNotFound
            } else {
                ErrorCode::WorkspaceWriteTypeError
            };
            return Err(SandboxError::new(
                code,
                format!("Failed to read `{path}`: {}", result.stderr_text().trim()),
            ));
        }
        Ok(result.stdout)
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        // The destination is passed as a positional argument, never interpolated into the
        // script, so a path containing shell metacharacters cannot inject a command.
        let script = "mkdir -p \"$(dirname \"$1\")\" && cat > \"$1\"";
        let argv = vec![
            "exec".to_string(),
            "-i".to_string(),
            "-w".to_string(),
            self.manifest.root().to_string(),
            self.container_id.clone(),
            "sh".to_string(),
            "-c".to_string(),
            script.to_string(),
            "sh".to_string(),
            resolved,
        ];
        let output = run_docker(&argv, Some(contents)).await?;
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to write `{path}`: {}", output.stderr_text().trim()),
            ));
        }
        Ok(())
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        let script = format!(
            "mkdir -p {root} && tar -xzf - -C {root}",
            root = shell_quote(&root)
        );
        let argv = vec![
            "exec".to_string(),
            "-i".to_string(),
            self.container_id.clone(),
            "sh".to_string(),
            "-c".to_string(),
            script,
        ];
        let output = run_docker(&argv, Some(archive)).await?;
        if output.exit_code != 0 {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!(
                    "Failed to restore the container workspace: {}",
                    output.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }
}

/// Run the `docker` CLI, optionally piping `stdin` into it.
async fn run_docker(argv: &[String], stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
    let mut command = tokio::process::Command::new("docker");
    command
        .args(argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|e| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to run `docker {}`: {e}", argv.join(" ")),
        )
    })?;
    if let (Some(data), Some(mut pipe)) = (stdin, child.stdin.take()) {
        pipe.write_all(data).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to stream stdin into `docker`: {e}"),
            )
        })?;
        pipe.shutdown().await.ok();
    }
    let output = child.wait_with_output().await.map_err(|e| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to wait for `docker {}`: {e}", argv.join(" ")),
        )
    })?;
    Ok(ExecResult {
        stdout: output.stdout,
        stderr: output.stderr,
        exit_code: output.status.code().unwrap_or(-1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_exec_arguments() {
        let manifest = Manifest::new().with_env("FOO", "bar");
        let session = DockerSandboxSession::new("abc123", manifest);
        let argv = session
            .exec_argv(&ExecRequest::shell("echo hi").with_cwd(Some("repo".into())))
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "exec",
                "-w",
                "/workspace/repo",
                "-e",
                "FOO=bar",
                "abc123",
                "sh",
                "-lc",
                "echo hi"
            ]
        );
    }

    #[test]
    fn rejects_paths_outside_the_workspace() {
        let session = DockerSandboxSession::new("abc123", Manifest::new());
        assert!(session.normalize_path("../etc").is_err());
        assert!(session.normalize_path("/etc/passwd").is_err());
        assert_eq!(session.normalize_path("a/b").unwrap(), "/workspace/a/b");
    }
}
