//! A sandbox session backed by a Docker container, driven through the `docker` CLI. Mirrors
//! `agents.sandbox.sandboxes.docker`.
//!
//! The Python SDK talks to the Docker daemon through `docker-py`. This port shells out to the
//! `docker` executable instead, so it needs no extra dependency: `docker create`, `docker start`,
//! `docker exec` and `docker rm` cover the whole session lifecycle.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mounts::{ContainerPrivileges, docker_volume_name};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, capture_workspace_tar, restore_session_snapshot,
    shell_quote,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, SnapshotFingerprint};
use crate::sandbox::types::ExecResult;
use crate::sandbox::workspace_paths::normalize_segments;

/// The `provider_metadata` key recording whether the container has no network stack.
pub const DOCKER_NETWORK_DISABLED_KEY: &str = "docker_network_disabled";
/// The `provider_metadata` key recording the container labels.
pub const DOCKER_LABELS_KEY: &str = "docker_labels";
/// The `provider_metadata` key recording the SDK session identity that names the volumes.
pub const DOCKER_SESSION_ID_KEY: &str = "docker_session_id";

/// The default image used when none is configured, matching the Python default.
pub const DEFAULT_PYTHON_SANDBOX_IMAGE: &str = "python:3.14-slim";

/// Settings used when creating a Docker-backed session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerSandboxClientOptions {
    /// The image the container runs.
    pub image: String,
    /// Disable container networking (`--network none`).
    ///
    /// Python spells this `network_mode="none"`; the two are the same setting.
    pub network_disabled: bool,
    /// Container ports published on `127.0.0.1`.
    pub exposed_ports: Vec<u16>,
    /// Labels applied to the container at create time.
    pub labels: BTreeMap<String, String>,
    /// Extra arguments inserted before the image in `docker create`.
    pub extra_create_args: Vec<String>,
}

impl Default for DockerSandboxClientOptions {
    fn default() -> Self {
        DockerSandboxClientOptions {
            image: DEFAULT_PYTHON_SANDBOX_IMAGE.to_string(),
            network_disabled: false,
            exposed_ports: Vec::new(),
            labels: BTreeMap::new(),
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

    /// Publish `port` on `127.0.0.1` with a daemon-assigned host port.
    pub fn with_exposed_port(mut self, port: u16) -> Self {
        self.exposed_ports.push(port);
        self
    }

    /// Publish every port in `ports`.
    pub fn with_exposed_ports(mut self, ports: impl IntoIterator<Item = u16>) -> Self {
        self.exposed_ports.extend(ports);
        self
    }

    /// Label the container.
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Label the container with every pair in `labels`.
    pub fn with_labels(mut self, labels: impl IntoIterator<Item = (String, String)>) -> Self {
        self.labels.extend(labels);
        self
    }

    /// Add an extra `docker create` argument.
    pub fn with_create_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_create_args.push(arg.into());
        self
    }

    /// Reject a configuration Docker cannot honour.
    ///
    /// Mirrors Python's `_validate_docker_network_configuration`: a container with no network
    /// stack cannot publish a port, so asking for both is a configuration mistake rather than a
    /// silently ignored setting.
    pub fn validate(&self) -> SandboxResult<()> {
        if self.network_disabled && !self.exposed_ports.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "exposed_ports cannot be used when network_disabled is set",
            ));
        }
        Ok(())
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
    async fn create_session(
        &self,
        manifest: Manifest,
        session_id: Option<String>,
    ) -> SandboxResult<DockerSandboxSession> {
        self.options.validate()?;
        validate_docker_path_grants(&manifest)?;
        // The volume names are derived from the session id, so it has to exist before the
        // container does. A replacement container keeps the paused session's id, so the volumes
        // the first container created are the ones this one attaches — and the ones its
        // `shutdown` removes.
        let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        let mut argv = vec![
            "create".to_string(),
            "--label".to_string(),
            "openai-agents-sandbox=1".to_string(),
            "-w".to_string(),
            manifest.root().to_string(),
        ];
        for (key, value) in &self.options.labels {
            argv.push("--label".to_string());
            argv.push(format!("{key}={value}"));
        }
        if self.options.network_disabled {
            argv.push("--network".to_string());
            argv.push("none".to_string());
        }
        for port in &self.options.exposed_ports {
            // Python publishes on 127.0.0.1 with a daemon-assigned host port, so a sandbox port
            // is never reachable from another machine.
            argv.push("-p".to_string());
            argv.push(format!("127.0.0.1::{port}/tcp"));
        }
        // Bind and volume mounts are attached by the container runtime, before the session
        // starts, so they are resolved from the manifest here rather than during materialization.
        let mut privileges = ContainerPrivileges::None;
        for (mount, mount_path) in manifest.mount_targets() {
            mount.validate()?;
            privileges = privileges.max(mount.strategy.container_privileges());
            if let Some(args) =
                mount
                    .strategy
                    .docker_create_args_for_session(mount, &mount_path, Some(&session_id))
            {
                argv.extend(args);
            }
        }
        // A capability or a device cannot be added to a running container, so an in-container
        // FUSE or `mount(2)` helper has to be provided for here or it can never work.
        argv.extend(container_privilege_args(privileges));
        // A manifest grant with a host path is a bind mount: the sandbox cannot reach a host
        // directory the runtime did not attach.
        for grant in manifest.extra_path_grants() {
            if grant.host_path.is_none() {
                continue;
            }
            let mut spec = format!("{}:{}", grant.host_source(), grant.path);
            if grant.read_only {
                spec.push_str(":ro");
            }
            argv.push("-v".to_string());
            argv.push(spec);
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
        Ok(DockerSandboxSession::new(container_id, manifest)
            .with_session_id(session_id)
            .with_client_options(&self.options)
            .with_snapshot(self.snapshot.clone()))
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
            self.create_session(manifest.unwrap_or_default(), None)
                .await?,
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
        // The trusted manifest supplies the host paths and the mount credentials the persisted
        // state deliberately dropped, so the rebind has to happen before anything reattaches.
        let state = state.rebind_path_grants(&manifest)?;
        let manifest = manifest.with_root(state.workspace_root.clone());
        validate_docker_path_grants(&manifest)?;
        let network_disabled = state
            .provider_metadata(DOCKER_NETWORK_DISABLED_KEY)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(self.options.network_disabled);
        let labels: BTreeMap<String, String> = state
            .provider_metadata(DOCKER_LABELS_KEY)
            .and_then(serde_json::Value::as_object)
            .map(|map| {
                map.iter()
                    .filter_map(|(key, value)| {
                        value.as_str().map(|value| (key.clone(), value.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_else(|| self.options.labels.clone());
        let session_id = state
            .provider_metadata(DOCKER_SESSION_ID_KEY)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.session_id.clone());
        // A manifest whose mounts carry credentials gets a fresh container: reattaching one the
        // state alone identifies would let a hand-edited state point the credentials at it.
        // That is also why the container id is only *required* on the reattach branch — a
        // sanitized state deliberately carries none, and must still resume into a replacement.
        let requires_fresh_resource = state.mount_authority_rebound;
        let container_id = if requires_fresh_resource {
            state.provider_id.clone().unwrap_or_default()
        } else {
            state.provider_id.clone().ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::SessionStateInvalid,
                    "docker sandbox session state is missing its container id",
                )
            })?
        };
        let attrs = if requires_fresh_resource {
            None
        } else {
            inspect_container(&container_id).await
        };
        if let Some(attrs) = attrs {
            // The container has been reachable to anything that could run `docker` since the
            // pause, so its configuration is re-checked against the trusted manifest rather than
            // trusted because the id matched.
            assert_container_path_grants_match(&attrs, &manifest)?;
            assert_container_network_matches(&attrs, network_disabled)?;
            assert_container_labels_match(&attrs, &labels)?;
            return Ok(Arc::new(
                DockerSandboxSession::new(container_id, manifest)
                    .with_session_id(session_id)
                    .with_resume_identity(network_disabled, labels, state.exposed_ports.clone())
                    .with_snapshot(Some(state.snapshot.clone()))
                    .with_snapshot_id(state.snapshot_id.clone())
                    .resuming()
                    .with_workspace_root_ready(state.workspace_root_ready)
                    .with_snapshot_fingerprint(
                        state.snapshot_fingerprint.clone(),
                        state.snapshot_fingerprint_version.clone(),
                    ),
            ));
        }
        // The original container is gone: create a replacement and hydrate it from the snapshot.
        // Keep the paused session's identity unless the state's mount authority was rebound,
        // in which case a fresh resource — and fresh volumes — is the whole point.
        let replacement_session_id = (!requires_fresh_resource).then_some(session_id);
        let replacement = self
            .create_session(manifest, replacement_session_id)
            .await?;
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
    /// Whether the manifest finished materializing, so a resume can trust the workspace.
    materialized: AtomicBool,
    /// Keeps concurrent work out of the workspace while a snapshot detaches its mounts.
    mount_transition: crate::sandbox::mounts::MountTransition,
    /// Host file holding the manifest's start-up environment lookups, written `0600`.
    ///
    /// Docker passes environment variables to `docker exec` as `-e KEY=VALUE`, which the host's
    /// process table shows to every local user. A value fetched from a secret store must not go
    /// that way, so it travels through `--env-file` instead.
    resolved_env_file: std::sync::OnceLock<std::path::PathBuf>,
    removed: Mutex<bool>,
    ptys: PtyManager,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
    /// The SDK session identity, which names this session's Docker volumes.
    session_id: String,
    /// Whether the container was created with `--network none`.
    network_disabled: bool,
    /// The labels the container was created with, checked again on resume.
    labels: BTreeMap<String, String>,
    /// The container ports published on `127.0.0.1`.
    exposed_ports: Vec<u16>,
    /// The workspace fingerprint of the snapshot this session resumes, when it has one.
    resume_fingerprint: std::sync::Mutex<Option<SnapshotFingerprint>>,
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
            materialized: AtomicBool::new(false),
            mount_transition: crate::sandbox::mounts::MountTransition::new(),
            resolved_env_file: std::sync::OnceLock::new(),
            removed: Mutex::new(false),
            ptys: PtyManager::new(),
            snapshot: None,
            resuming: false,
            session_id: uuid::Uuid::new_v4().simple().to_string(),
            network_disabled: false,
            labels: BTreeMap::new(),
            exposed_ports: Vec::new(),
            resume_fingerprint: std::sync::Mutex::new(None),
        }
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

    /// Use an explicit SDK session identity, which names this session's Docker volumes.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// Record the persisted create-time identity a resumed container is checked against.
    pub fn with_resume_identity(
        mut self,
        network_disabled: bool,
        labels: BTreeMap<String, String>,
        exposed_ports: Vec<u16>,
    ) -> Self {
        self.network_disabled = network_disabled;
        self.labels = labels;
        self.exposed_ports = exposed_ports;
        self
    }

    /// Record the create-time options so a resume can check the container still matches them.
    pub fn with_client_options(mut self, options: &DockerSandboxClientOptions) -> Self {
        self.network_disabled = options.network_disabled;
        self.labels = options.labels.clone();
        self.exposed_ports = options.exposed_ports.clone();
        self
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

    /// Whether the workspace this session resumes was fully materialized before the pause.
    ///
    /// A run interrupted mid-materialization leaves a workspace that has files in it but is not
    /// the one the manifest describes, so resuming it has to re-apply the manifest.
    pub fn with_workspace_root_ready(self, ready: bool) -> Self {
        self.materialized.store(ready, Ordering::SeqCst);
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
        argv.extend(self.resolved_environment_args());
        for (key, value) in request.resolved_env(self.manifest.environment()) {
            argv.push("-e".to_string());
            argv.push(format!("{key}={value}"));
        }
        argv.push(self.container_id.clone());
        argv.extend(request.resolved_argv());
        Ok(argv)
    }

    /// Perform the manifest's start-up environment lookups into an owner-only host file.
    ///
    /// Docker reads `--env-file` itself, so the values never appear in an argument vector.
    async fn write_resolved_environment_file(&self) -> SandboxResult<()> {
        if self.manifest.resolved_environment().is_empty() {
            return Ok(());
        }
        // `stop()` then `start()` reuses the session. Writing a second file would leave the
        // first one behind, because `OnceLock::set` would quietly refuse the new path and
        // teardown only knows about the one it holds.
        if self.resolved_env_file.get().is_some() {
            return Ok(());
        }
        let resolved = self.manifest.resolved_environment().resolve().await?;
        let mut body = String::new();
        for (key, value) in &resolved {
            // `--env-file` is line-oriented, so a value with a newline cannot be expressed and
            // would silently become a different variable.
            if value.contains(['\n', '\r']) {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!("the environment value for `{key}` must not contain a line break"),
                ));
            }
            body.push_str(&format!("{key}={value}\n"));
        }
        // The name is unpredictable and the file is created `0600` in one step. Creating it
        // world-readable and tightening it afterwards leaves a window in a shared directory, and
        // a predictable name invites a planted symlink for the write to follow.
        let path = std::env::temp_dir().join(format!(
            "openai-agents-sandbox-env-{}",
            uuid::Uuid::new_v4()
        ));
        let mut options = tokio::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let staged = async {
            use tokio::io::AsyncWriteExt as _;
            let mut file = options.open(&path).await?;
            file.write_all(body.as_bytes()).await?;
            file.flush().await
        };
        staged.await.map_err(|error| {
            SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!("Failed to stage the sandbox environment file: {error}"),
            )
        })?;
        let _ = self.resolved_env_file.set(path);
        Ok(())
    }

    /// The `docker exec` argument vector that starts one interactive process.
    ///
    /// The host-side `docker exec` is only a proxy: killing it leaves the process inside the
    /// container running, so the command records its pid for teardown to kill.
    fn pty_argv(&self, request: &PtyExecRequest, workdir: &str, pid_path: &str) -> Vec<String> {
        let inner = format!(
            "echo $$ > {pid}; exec {argv}",
            pid = shell_quote(pid_path),
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
        argv.push(workdir.to_string());
        // An interactive process gets the same environment a command does, including the
        // start-up lookups; without this a secret would be present for `shell` and missing for
        // `exec_command`, which is the silent gap `resolves_environment` exists to prevent.
        argv.extend(self.resolved_environment_args());
        for (key, value) in self.manifest.environment() {
            argv.push("-e".to_string());
            argv.push(format!("{key}={value}"));
        }
        argv.push(self.container_id.clone());
        argv.extend(["sh".to_string(), "-c".to_string(), inner]);
        argv
    }

    /// The `docker exec` arguments that supply the manifest's start-up environment lookups.
    ///
    /// They go through `--env-file`, which `docker` reads itself, rather than `-e KEY=VALUE`,
    /// which the host's process table shows to every local user.
    fn resolved_environment_args(&self) -> Vec<String> {
        match self.resolved_env_file.get() {
            Some(path) => vec![
                "--env-file".to_string(),
                path.to_string_lossy().into_owned(),
            ],
            None => Vec::new(),
        }
    }

    /// Remove the staged environment file, if one was written.
    async fn remove_resolved_environment_file(&self) {
        if let Some(path) = self.resolved_env_file.get() {
            let _ = tokio::fs::remove_file(path).await;
        }
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
        self.write_resolved_environment_file().await?;
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
            self.started.store(false, Ordering::SeqCst);
        })?;
        // Only now is the workspace what the manifest says it is; a resume before this point
        // would find a half-materialized tree that looks populated.
        self.materialized.store(true, Ordering::SeqCst);
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
        // The staged secrets must not outlive the container that used them.
        self.remove_resolved_environment_file().await;
        // In-container mounts have to be released while the container is still alive. Bind and
        // volume mounts are detached by the runtime when the container goes away.
        if let Err(error) = self.manifest.unmount_all(self).await {
            log::warn!(target: crate::logger::LOG_TARGET, "Failed to detach a container mount before removal: {error}");
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
        // A named volume outlives its container, so the storage a mount attached would leak.
        for volume in docker_volume_names_for_manifest(&self.manifest, Some(&self.session_id)) {
            let output = run_docker(
                &["volume".to_string(), "rm".to_string(), volume.clone()],
                None,
            )
            .await;
            if let Ok(output) = output
                && output.exit_code != 0
                && !output.stderr_text().contains("no such volume")
            {
                log::warn!(target: crate::logger::LOG_TARGET,
                    "Failed to remove the sandbox Docker volume `{volume}`: {}",
                    output.stderr_text().trim()
                );
            }
        }
        Ok(())
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

    fn attaches_mounts_at_create(&self) -> bool {
        true
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.exposed_ports.clone()
    }

    fn session_state(&self) -> SandboxSessionState {
        let mut state = SandboxSessionState::new(
            "docker",
            &self.container_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.container_id.clone()))
        .with_workspace_root_ready(self.materialized.load(Ordering::SeqCst))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
        .with_provider_metadata(
            DOCKER_NETWORK_DISABLED_KEY,
            serde_json::Value::Bool(self.network_disabled),
        )
        .with_provider_metadata(
            DOCKER_LABELS_KEY,
            serde_json::Value::Object(
                self.labels
                    .iter()
                    .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
                    .collect(),
            ),
        )
        .with_provider_metadata(
            DOCKER_SESSION_ID_KEY,
            serde_json::Value::String(self.session_id.clone()),
        );
        state.exposed_ports = self.exposed_ports.clone();
        state.with_snapshot_fingerprint(self.resume_snapshot_fingerprint())
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
        let argv = self.pty_argv(&request, &workdir, &pid_path);

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
        let resolved = self.normalize_path_for_write(path)?;
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

    async fn persist_workspace_once(&self) -> SandboxResult<Vec<u8>> {
        let raw = capture_workspace_tar(self).await?;
        // A container workspace carries links recorded against its own root, and the hydrate
        // policy refuses an absolute target. Python rewrites them as it captures the archive
        // (`strip_tar_member_prefix(relativize_symlinks_under=root)`); so does this.
        let roots = vec![self.manifest.root().to_string()];
        crate::sandbox::archive::rebase_archive_symlink_targets(&raw, &roots)
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        // A resume clears the workspace before it hydrates, so an archive that cannot be
        // restored must be refused while the live files are still there.
        crate::sandbox::archive::validate_hydrate_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)?;
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
    command.args(argv);
    run_piped(command, &format!("docker {}", argv.join(" ")), stdin).await
}

/// Spawn `command` with piped streams, feeding `stdin` while draining stdout and stderr.
///
/// The three streams are pumped concurrently because they deadlock otherwise: a child that
/// writes more than a pipe buffer before it finishes reading stdin blocks on its own stdout,
/// stops draining stdin, and leaves the caller blocked mid-write forever. Writing stdin to
/// completion first is only safe when the child is known to stay quiet until EOF, which is not
/// a property the caller can guarantee for `docker exec`.
///
/// This is the job Python's `_LENGTH_FRAMED_STDIN_SCRIPT` and `_DockerExecSocket` do over a raw
/// hijacked socket, where `docker-py` cannot half-close a TLS stdin. Through the CLI the half
/// close is the `docker` client's own (`moby/client/hijack.go`), so the framing is unnecessary
/// — but the concurrent pumping is not.
async fn run_piped(
    mut command: tokio::process::Command,
    label: &str,
    stdin: Option<&[u8]>,
) -> SandboxResult<ExecResult> {
    command
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
            format!("Failed to run `{label}`: {e}"),
        )
    })?;

    let stdin_pipe = child.stdin.take();
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    let feed = async move {
        if let (Some(data), Some(mut pipe)) = (stdin, stdin_pipe) {
            let written = pipe.write_all(data).await;
            // Closing the write side is the child's EOF; the handle drops with this block.
            pipe.shutdown().await.ok();
            // A child that exits without reading all of its input leaves a broken pipe. That
            // is the child's business, not a transport failure, and its exit code reports it.
            if let Err(e) = written
                && e.kind() != std::io::ErrorKind::BrokenPipe
            {
                return Err(e);
            }
        }
        Ok(())
    };
    let drain_stdout = async {
        match stdout_pipe.as_mut() {
            Some(pipe) => pipe.read_to_end(&mut stdout).await.map(|_| ()),
            None => Ok(()),
        }
    };
    let drain_stderr = async {
        match stderr_pipe.as_mut() {
            Some(pipe) => pipe.read_to_end(&mut stderr).await.map(|_| ()),
            None => Ok(()),
        }
    };

    tokio::try_join!(feed, drain_stdout, drain_stderr).map_err(|e| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to stream `{label}`: {e}"),
        )
    })?;

    let status = child.wait().await.map_err(|e| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to wait for `{label}`: {e}"),
        )
    })?;
    Ok(ExecResult {
        stdout,
        stderr,
        exit_code: status.code().unwrap_or(-1),
    })
}

/// The `docker create` arguments that grant `privileges`.
///
/// Mirrors Python's `_manifest_requires_fuse` / `_manifest_requires_sys_admin` branches.
/// `apparmor:unconfined` goes with `CAP_SYS_ADMIN` because the default AppArmor profile denies
/// `mount(2)` even to a capable process.
fn container_privilege_args(privileges: ContainerPrivileges) -> Vec<String> {
    let mut argv: Vec<String> = Vec::new();
    match privileges {
        ContainerPrivileges::None => return argv,
        ContainerPrivileges::FuseDevice => {
            argv.extend(["--device".to_string(), "/dev/fuse".to_string()]);
        }
        ContainerPrivileges::SysAdmin => {}
    }
    argv.extend([
        "--cap-add".to_string(),
        "SYS_ADMIN".to_string(),
        "--security-opt".to_string(),
        "apparmor:unconfined".to_string(),
    ]);
    argv
}

/// The `docker container inspect` attributes of `container_id`, or `None` when it is gone.
async fn inspect_container(container_id: &str) -> Option<serde_json::Value> {
    let output = run_docker(
        &[
            "container".to_string(),
            "inspect".to_string(),
            container_id.to_string(),
        ],
        None,
    )
    .await
    .ok()?;
    if output.exit_code != 0 {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    parsed.as_array()?.first().cloned()
}

/// Compare Linux mount paths lexically, collapsing leading-slash aliases.
///
/// Mirrors Python's `_docker_mount_path_identity`: POSIX `normpath` keeps exactly two leading
/// slashes, but Linux treats `//workspace` and `/workspace` as the same path.
fn docker_mount_path_identity(path: &str) -> String {
    let collapsed = {
        let trimmed = path.trim_start_matches('/');
        if trimmed.len() == path.len() {
            path.to_string()
        } else {
            format!("/{trimmed}")
        }
    };
    normalize_segments(&collapsed)
}

/// Whether `path` is `root` or lies under it.
fn is_at_or_under(path: &str, root: &str) -> bool {
    path == root || path.starts_with(&format!("{}/", root.trim_end_matches('/')))
}

/// The mount paths this manifest attaches as named Docker volumes.
///
/// Mirrors Python's `_docker_volume_mounts_for_manifest`, which selects the mounts whose strategy
/// produces a volume driver configuration.
fn docker_volume_mount_paths(manifest: &Manifest) -> Vec<String> {
    manifest
        .mount_targets()
        .into_iter()
        .filter(|(mount, path)| {
            mount
                .strategy
                .docker_create_args(mount, path)
                .is_some_and(|args| {
                    args.first().map(String::as_str) == Some("--mount")
                        && args
                            .get(1)
                            .is_some_and(|spec| spec.starts_with("type=volume"))
                })
        })
        .map(|(_, path)| path)
        .collect()
}

/// The Docker volume names `session_id` uses for `manifest`.
///
/// Mirrors Python's `_docker_volume_names_for_manifest`.
pub fn docker_volume_names_for_manifest(
    manifest: &Manifest,
    session_id: Option<&str>,
) -> Vec<String> {
    docker_volume_mount_paths(manifest)
        .iter()
        .map(|path| docker_volume_name(session_id, path))
        .collect()
}

/// Reject a manifest whose host path grants Docker cannot attach safely.
///
/// A verbatim port of Python's `_validate_docker_path_grants`.
pub fn validate_docker_path_grants(manifest: &Manifest) -> SandboxResult<()> {
    let mut privileges = ContainerPrivileges::None;
    for (mount, _) in manifest.mount_targets() {
        privileges = privileges.max(mount.strategy.container_privileges());
    }
    let has_read_only_host_grant = manifest
        .extra_path_grants()
        .iter()
        .any(|grant| grant.host_path.is_some() && grant.read_only);
    if has_read_only_host_grant && privileges != ContainerPrivileges::None {
        // `CAP_SYS_ADMIN` lets a sandbox process remount a read-only host bind writable, so the
        // grant would not be read-only in practice.
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            "Docker read-only host_path grants cannot be combined with in-container storage \
             mounts that require SYS_ADMIN; remove the host grant or use a storage strategy that \
             does not require container mount privileges",
        ));
    }

    let root = docker_mount_path_identity(manifest.root());
    let volume_targets: std::collections::BTreeSet<String> = docker_volume_mount_paths(manifest)
        .iter()
        .map(|path| docker_mount_path_identity(path))
        .collect();
    let mut seen_targets: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut explicit_targets: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    for grant in manifest.extra_path_grants() {
        let target = docker_mount_path_identity(&grant.path);
        if seen_targets.contains(&target)
            && (grant.host_path.is_some() || explicit_targets.contains(&target))
        {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("duplicate Docker sandbox path grant target: {}", grant.path),
            ));
        }
        seen_targets.insert(target.clone());
        if grant.host_path.is_none() {
            continue;
        }
        explicit_targets.insert(target.clone());
        if is_at_or_under(&target, &root) || is_at_or_under(&root, &target) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "Docker sandbox path grant host_path target must be outside the workspace \
                     root: {}",
                    grant.path
                ),
            ));
        }
        if volume_targets.contains(&target) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "Docker sandbox path grant target conflicts with a manifest mount: {}",
                    grant.path
                ),
            ));
        }
    }
    Ok(())
}

fn container_inspect_mismatch(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::SessionStateMismatch, message)
}

/// Reject reattaching to a container whose network stack is not the persisted one.
///
/// A verbatim port of Python's `_assert_existing_container_network_configuration_matches`.
pub fn assert_container_network_matches(
    attrs: &serde_json::Value,
    network_disabled: bool,
) -> SandboxResult<()> {
    if !network_disabled {
        return Ok(());
    }
    let actual_mode = attrs
        .get("HostConfig")
        .and_then(|config| config.get("NetworkMode"))
        .and_then(serde_json::Value::as_str);
    let networks = attrs
        .get("NetworkSettings")
        .and_then(|settings| settings.get("Networks"))
        .and_then(serde_json::Value::as_object);
    let attached_beyond_none = match networks {
        None => true,
        Some(map) => map.keys().any(|name| name != "none"),
    };
    if actual_mode != Some("none") || attached_beyond_none {
        return Err(container_inspect_mismatch(
            "Existing Docker sandbox network configuration does not match the persisted \
             network_mode='none'; create a fresh sandbox session",
        ));
    }
    Ok(())
}

/// Reject reattaching to a container that no longer carries the persisted labels.
///
/// A verbatim port of Python's `_assert_existing_container_labels_match`.
pub fn assert_container_labels_match(
    attrs: &serde_json::Value,
    labels: &BTreeMap<String, String>,
) -> SandboxResult<()> {
    if labels.is_empty() {
        return Ok(());
    }
    let actual = attrs
        .get("Config")
        .and_then(|config| config.get("Labels"))
        .and_then(serde_json::Value::as_object);
    for (key, value) in labels {
        let found = actual
            .and_then(|map| map.get(key))
            .and_then(serde_json::Value::as_str);
        if found != Some(value.as_str()) {
            return Err(container_inspect_mismatch(
                "Existing Docker sandbox labels do not match the persisted labels; create a \
                 fresh sandbox session",
            ));
        }
    }
    Ok(())
}

/// Reject reattaching to a container whose bind mounts differ from the trusted manifest.
///
/// A verbatim port of Python's `_assert_existing_container_path_grants_match`. This is the check
/// that stops a persisted state from pointing the SDK at a container somebody else prepared with
/// extra host directories bound into it.
pub fn assert_container_path_grants_match(
    attrs: &serde_json::Value,
    manifest: &Manifest,
) -> SandboxResult<()> {
    let mounts = attrs
        .get("Mounts")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let expected: std::collections::BTreeMap<
        &str,
        &crate::sandbox::workspace_paths::SandboxPathGrant,
    > = manifest
        .extra_path_grants()
        .iter()
        .filter(|grant| grant.host_path.is_some())
        .map(|grant| (grant.path.as_str(), grant))
        .collect();

    let mut actual: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
        std::collections::BTreeMap::new();
    for mount in mounts {
        if mount.get("Type").and_then(serde_json::Value::as_str) != Some("bind") {
            continue;
        }
        let Some(destination) = mount.get("Destination").and_then(serde_json::Value::as_str) else {
            return Err(container_inspect_mismatch(
                "Existing Docker sandbox has a bind mount without a valid destination; create a \
                 fresh sandbox session",
            ));
        };
        actual
            .entry(destination.to_string())
            .or_default()
            .push(mount);
    }

    let unexpected: Vec<&str> = actual
        .keys()
        .map(String::as_str)
        .filter(|destination| !expected.contains_key(destination))
        .collect();
    if !unexpected.is_empty() {
        return Err(container_inspect_mismatch(format!(
            "Existing Docker sandbox has bind mounts that are not present in the current trusted \
             manifest: {}; create a fresh sandbox session",
            unexpected.join(", ")
        )));
    }

    for (path, grant) in &expected {
        let target_mounts = actual.get(*path).map(Vec::as_slice).unwrap_or(&[]);
        let mismatch = || {
            container_inspect_mismatch(format!(
                "Existing Docker sandbox path grant mount does not match the current trusted \
                 manifest for '{path}'; create a fresh sandbox session"
            ))
        };
        if target_mounts.len() != 1 {
            return Err(mismatch());
        }
        let mount = target_mounts[0];
        let source = mount
            .get("Source")
            .and_then(serde_json::Value::as_str)
            .map(normalize_segments);
        let read_only = mount.get("RW").and_then(serde_json::Value::as_bool) == Some(false);
        if source.as_deref() != Some(normalize_segments(grant.host_source()).as_str())
            || read_only != grant.read_only
        {
            return Err(mismatch());
        }
    }
    Ok(())
}

#[cfg(test)]
mod privilege_tests {
    use super::*;

    #[tokio::test]
    async fn a_looked_up_secret_reaches_both_exec_paths_without_the_process_table() {
        use crate::sandbox::environment::EnvEntry;

        let session = DockerSandboxSession::new(
            "container-under-test".to_string(),
            Manifest::new()
                .with_root("/workspace")
                .with_env("PLAIN", "visible")
                .with_env_entry("TOKEN", EnvEntry::literal("SUPERSECRET")),
        );
        session.write_resolved_environment_file().await.unwrap();

        let path = session
            .resolved_env_file
            .get()
            .expect("a looked-up environment is staged in a file")
            .clone();
        // The file is owner-only, and its name is not derived from the container id.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = tokio::fs::metadata(&path)
                .await
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "{mode:o}");
        }
        assert!(
            !path.to_string_lossy().contains("container-under-test"),
            "{}",
            path.display()
        );
        let body = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(body.contains("TOKEN=SUPERSECRET"), "{body}");

        // Both the command path and the interactive path read it through `--env-file`, and
        // neither puts the value in an argument the host's process table would show.
        let command_argv = session
            .exec_argv(&ExecRequest::argv(["echo", "hi"]))
            .unwrap();
        let pty_argv = session.pty_argv(&PtyExecRequest::new("sh"), "/workspace", "/tmp/pid");
        for argv in [command_argv, pty_argv] {
            let joined = argv.join(" ");
            assert!(joined.contains("--env-file"), "{joined}");
            assert!(!joined.contains("SUPERSECRET"), "{joined}");
            // A plain manifest value is not a secret and still travels as `-e`.
            assert!(joined.contains("PLAIN=visible"), "{joined}");
        }

        session.remove_resolved_environment_file().await;
        assert!(tokio::fs::metadata(&path).await.is_err());
    }

    #[test]
    fn grants_only_what_a_mount_needs() {
        assert!(container_privilege_args(ContainerPrivileges::None).is_empty());
        assert_eq!(
            container_privilege_args(ContainerPrivileges::SysAdmin),
            vec![
                "--cap-add",
                "SYS_ADMIN",
                "--security-opt",
                "apparmor:unconfined"
            ]
        );
        assert_eq!(
            container_privilege_args(ContainerPrivileges::FuseDevice),
            vec![
                "--device",
                "/dev/fuse",
                "--cap-add",
                "SYS_ADMIN",
                "--security-opt",
                "apparmor:unconfined"
            ]
        );
    }

    #[test]
    fn the_strongest_requirement_wins() {
        // `max` over the enum is what `create_session` folds the manifest's mounts with.
        assert_eq!(
            ContainerPrivileges::SysAdmin.max(ContainerPrivileges::FuseDevice),
            ContainerPrivileges::FuseDevice
        );
        assert_eq!(
            ContainerPrivileges::None.max(ContainerPrivileges::SysAdmin),
            ContainerPrivileges::SysAdmin
        );
    }
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

    /// Payload larger than any pipe buffer, so a sequential write-then-read would wedge.
    const BIG: usize = 4 * 1024 * 1024;

    fn sh(script: &str) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("sh");
        command.arg("-c").arg(script);
        command
    }

    #[tokio::test]
    async fn a_large_stdin_payload_survives_a_child_that_floods_stdout_first() {
        // The child writes megabytes before it reads a byte of stdin. Feeding stdin to
        // completion first would block: the child is stuck on its own full stdout pipe.
        let payload = vec![b'x'; BIG];
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            run_piped(
                sh("dd if=/dev/zero bs=1024 count=4096 2>/dev/null; wc -c"),
                "flood then read",
                Some(&payload),
            ),
        )
        .await
        .expect("run_piped deadlocked on a large stdin payload")
        .expect("run_piped");

        assert_eq!(result.exit_code, 0);
        // 4 MiB of zeroes, then `wc -c` reporting every byte of stdin it drained.
        assert!(result.stdout.len() > BIG, "stdout was truncated");
        assert!(result.stdout[..BIG].iter().all(|b| *b == 0));
        let tail = String::from_utf8_lossy(&result.stdout[BIG..]);
        assert_eq!(
            tail.split_whitespace().next(),
            Some(BIG.to_string().as_str()),
            "stdin was truncated: {tail:?}"
        );
    }

    #[tokio::test]
    async fn stdout_and_stderr_stay_separate_and_the_exit_code_survives() {
        let result = run_piped(
            sh("echo out; echo err >&2; exit 7"),
            "streams",
            Some(b"ignored"),
        )
        .await
        .expect("run_piped");
        assert_eq!(result.exit_code, 7);
        assert_eq!(result.stdout_text(), "out\n");
        assert_eq!(result.stderr_text(), "err\n");
    }

    #[tokio::test]
    async fn stdin_reaches_eof_so_a_draining_child_can_finish() {
        // `cat` only exits when its stdin is closed, so this hangs unless the write side is
        // dropped. That half-close is what Python's length framing exists to work around on a
        // TLS socket docker-py cannot shut down.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_piped(sh("cat"), "cat", Some(b"hello")),
        )
        .await
        .expect("stdin never reached EOF")
        .expect("run_piped");
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout_text(), "hello");
    }

    /// A one-member archive: a symlink when `link_target` is set, a regular file otherwise.
    fn probe_archive(name: &str, link_target: Option<&str>) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(0);
        match link_target {
            Some(target) => {
                header.set_entry_type(tar::EntryType::Symlink);
                builder.append_link(&mut header, name, target).unwrap();
            }
            None => {
                header.set_size(2);
                builder.append_data(&mut header, name, &b"hi"[..]).unwrap();
            }
        }
        builder.into_inner().unwrap()
    }

    #[test]
    fn a_resume_refuses_an_archive_that_cannot_replace_the_workspace() {
        let session = DockerSandboxSession::new("abc123", Manifest::new());

        // A resume clears the container workspace before it hydrates, so the archive is judged
        // while the live files are still there. Mirrors Python's
        // `_restore_snapshot_into_workspace_on_resume`.
        let error = session
            .validate_restorable_archive(&probe_archive("bin/python", Some("/usr/bin/python3")))
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("absolute symlink target not allowed: /usr/bin/python3")
        );

        session
            .validate_restorable_archive(&probe_archive("notes.txt", None))
            .expect("a restorable archive is accepted");
    }

    #[tokio::test]
    async fn a_child_that_ignores_its_stdin_is_not_a_transport_error() {
        // A broken pipe here is the child's choice, reported by its exit code, not a failure
        // of the transport.
        let payload = vec![b'x'; BIG];
        let result = run_piped(sh("exit 3"), "early exit", Some(&payload))
            .await
            .expect("an unread stdin must not fail the call");
        assert_eq!(result.exit_code, 3);
    }
}
