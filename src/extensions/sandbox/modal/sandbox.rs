//! A sandbox session backed by a Modal sandbox. Mirrors
//! `agents.extensions.sandbox.modal.sandbox`.
//!
//! # Why a helper process
//!
//! Modal's client talks to its control plane over gRPC, with a protocol that is not published as
//! a REST surface, so there is nothing for `reqwest` to call: `modal.Sandbox.create`,
//! `Sandbox.exec` and `Sandbox.snapshot_filesystem` exist only in Modal's own SDKs, and the
//! `modal` CLI exposes no sandbox lifecycle commands. This port therefore drives the official
//! Python SDK through a small bundled helper
//! ([`HELPER_SOURCE`]): each operation spawns
//! the helper, writes one JSON request to its stdin and reads one JSON response from its stdout.
//!
//! Every invocation is stateless and addresses the sandbox by its Modal id, so concurrent calls
//! never share process state. The cost is one interpreter start-up (and one `import modal`) per
//! operation, which is why [`ModalSandboxClientOptions::with_helper_command`] exists: a long-lived
//! bridge, or a fake helper in tests, can replace it without touching the session code.
//!
//! The helper also serves PTY sessions: [`pty_exec_start`](SandboxSession::pty_exec_start) runs
//! it under the core [`PtyManager`], which gives the model a real terminal over Modal's own
//! `pty=True` exec.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Map, Value, json};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::Mutex;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::sandboxes::docker::DEFAULT_PYTHON_SANDBOX_IMAGE;
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::ExecResult;

use super::mounts::ModalCloudBucketMountStrategy;

/// The bundled helper that drives the official `modal` Python SDK.
pub const HELPER_SOURCE: &str = include_str!("helper.py");

/// The default exec timeout, matching the Python port.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// The default `snapshot_filesystem` timeout.
pub const DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT: Duration = Duration::from_secs(60);
/// The Modal image builder version pinned by the Python port.
pub const DEFAULT_IMAGE_BUILDER_VERSION: &str = "2025.06";
/// The environment variable that replaces the whole helper command.
pub const HELPER_COMMAND_ENV: &str = "OPENAI_AGENTS_MODAL_HELPER";
/// The environment variable that replaces the helper's Python interpreter.
pub const HELPER_PYTHON_ENV: &str = "OPENAI_AGENTS_MODAL_PYTHON";

/// The magic prefix of a snapshot payload that references a Modal filesystem snapshot.
const SNAPSHOT_FILESYSTEM_MAGIC: &[u8] = b"MODAL_SANDBOX_FS_SNAPSHOT_V1\n";

/// How a Modal workspace is captured into a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModalWorkspacePersistence {
    /// Stream a gzipped tar of the workspace out of the sandbox.
    #[default]
    Tar,
    /// Call Modal's `snapshot_filesystem` and store a reference to the resulting image.
    SnapshotFilesystem,
}

/// Settings used when creating a Modal-backed session.
#[derive(Debug, Clone, PartialEq)]
pub struct ModalSandboxClientOptions {
    /// The Modal app the sandbox belongs to.
    pub app_name: String,
    /// The registry tag the sandbox image is built from.
    pub image_tag: Option<String>,
    /// An existing Modal image id, which takes precedence over the tag.
    pub image_id: Option<String>,
    /// How the workspace is captured into a snapshot.
    pub workspace_persistence: ModalWorkspacePersistence,
    /// Ports Modal exposes over an encrypted tunnel.
    pub exposed_ports: Vec<u16>,
    /// The GPU reservation, such as `A100` or `H100:8`.
    pub gpu: Option<String>,
    /// The sandbox lifetime in seconds.
    pub timeout_secs: u32,
    /// The idle timeout in seconds.
    pub idle_timeout_secs: Option<u32>,
    /// The CPU reservation.
    pub cpu: Option<f64>,
    /// The memory reservation in MiB.
    pub memory_mb: Option<u32>,
    /// Override the image command with `sleep infinity` so the sandbox stays up for `exec`.
    pub use_sleep_cmd: bool,
    /// The Modal image builder version.
    pub image_builder_version: Option<String>,
    /// How long `Sandbox.create` may take.
    pub sandbox_create_timeout: Option<Duration>,
    /// The default exec timeout.
    pub exec_timeout: Duration,
    /// How long `snapshot_filesystem` may take.
    pub snapshot_filesystem_timeout: Duration,
    /// The command that runs the helper, or the bundled helper when unset.
    pub helper_command: Option<Vec<String>>,
    /// The interpreter the bundled helper runs under.
    pub python: Option<String>,
}

impl ModalSandboxClientOptions {
    /// Options for a Modal app.
    pub fn new(app_name: impl Into<String>) -> Self {
        ModalSandboxClientOptions {
            app_name: app_name.into(),
            image_tag: None,
            image_id: None,
            workspace_persistence: ModalWorkspacePersistence::Tar,
            exposed_ports: Vec::new(),
            gpu: None,
            timeout_secs: 300,
            idle_timeout_secs: None,
            cpu: None,
            memory_mb: None,
            use_sleep_cmd: true,
            image_builder_version: Some(DEFAULT_IMAGE_BUILDER_VERSION.to_string()),
            sandbox_create_timeout: None,
            exec_timeout: DEFAULT_TIMEOUT,
            snapshot_filesystem_timeout: DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT,
            helper_command: None,
            python: None,
        }
    }

    /// Build the sandbox image from a registry tag.
    pub fn with_image_tag(mut self, image_tag: impl Into<String>) -> Self {
        self.image_tag = Some(image_tag.into());
        self
    }

    /// Boot the sandbox from an existing Modal image.
    pub fn with_image_id(mut self, image_id: impl Into<String>) -> Self {
        self.image_id = Some(image_id.into());
        self
    }

    /// Choose how the workspace is captured into a snapshot.
    pub fn with_workspace_persistence(mut self, persistence: ModalWorkspacePersistence) -> Self {
        self.workspace_persistence = persistence;
        self
    }

    /// Expose ports over Modal's encrypted tunnels.
    pub fn with_exposed_ports<I: IntoIterator<Item = u16>>(mut self, ports: I) -> Self {
        self.exposed_ports = ports.into_iter().collect();
        self
    }

    /// Reserve a GPU.
    pub fn with_gpu(mut self, gpu: impl Into<String>) -> Self {
        self.gpu = Some(gpu.into());
        self
    }

    /// Set the sandbox lifetime.
    pub fn with_timeout_secs(mut self, timeout_secs: u32) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Set the idle timeout.
    pub fn with_idle_timeout_secs(mut self, idle_timeout_secs: u32) -> Self {
        self.idle_timeout_secs = Some(idle_timeout_secs);
        self
    }

    /// Reserve CPU.
    pub fn with_cpu(mut self, cpu: f64) -> Self {
        self.cpu = Some(cpu);
        self
    }

    /// Reserve memory, in MiB.
    pub fn with_memory_mb(mut self, memory_mb: u32) -> Self {
        self.memory_mb = Some(memory_mb);
        self
    }

    /// Keep or drop the `sleep infinity` command override.
    pub fn with_use_sleep_cmd(mut self, use_sleep_cmd: bool) -> Self {
        self.use_sleep_cmd = use_sleep_cmd;
        self
    }

    /// Pin the Modal image builder version.
    pub fn with_image_builder_version(mut self, version: impl Into<String>) -> Self {
        self.image_builder_version = Some(version.into());
        self
    }

    /// Bound `Sandbox.create`.
    pub fn with_sandbox_create_timeout(mut self, timeout: Duration) -> Self {
        self.sandbox_create_timeout = Some(timeout);
        self
    }

    /// Set the default exec timeout.
    pub fn with_exec_timeout(mut self, timeout: Duration) -> Self {
        self.exec_timeout = timeout;
        self
    }

    /// Bound `snapshot_filesystem`.
    pub fn with_snapshot_filesystem_timeout(mut self, timeout: Duration) -> Self {
        self.snapshot_filesystem_timeout = timeout;
        self
    }

    /// Run a different helper command, such as a long-lived bridge or a test double.
    pub fn with_helper_command<I, S>(mut self, command: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.helper_command = Some(command.into_iter().map(Into::into).collect());
        self
    }

    /// Run the bundled helper under a specific interpreter.
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python = Some(python.into());
        self
    }

    fn resolve_python(&self) -> String {
        self.python
            .clone()
            .or_else(|| std::env::var(HELPER_PYTHON_ENV).ok())
            .filter(|python| !python.is_empty())
            .unwrap_or_else(|| "python3".to_string())
    }

    /// The command that runs the helper.
    ///
    /// The options win over [`HELPER_COMMAND_ENV`], which wins over the bundled helper. The
    /// environment value is split on whitespace, so a path containing spaces belongs in the
    /// options instead.
    fn resolve_helper_command(&self) -> SandboxResult<Vec<String>> {
        if let Some(command) = &self.helper_command {
            if command.is_empty() {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "ModalSandboxClientOptions::with_helper_command needs at least a program",
                ));
            }
            return Ok(command.clone());
        }
        if let Ok(command) = std::env::var(HELPER_COMMAND_ENV)
            && !command.trim().is_empty()
        {
            return Ok(command.split_whitespace().map(str::to_string).collect());
        }
        // The helper source travels as a `-c` argument rather than a temporary file: a
        // predictable path in a shared `/tmp` would be an executable another user could plant,
        // and this process runs it with the Modal tokens in its environment.
        Ok(vec![
            self.resolve_python(),
            "-c".to_string(),
            HELPER_SOURCE.to_string(),
        ])
    }
}

/// The helper command and sandbox identity behind one Modal sandbox.
#[derive(Debug, Clone)]
pub struct ModalHelper {
    command: Vec<String>,
}

impl ModalHelper {
    /// A helper that runs `command`.
    pub fn new(command: Vec<String>) -> Self {
        ModalHelper { command }
    }

    /// The program and arguments the helper is spawned with.
    pub fn command(&self) -> &[String] {
        &self.command
    }

    /// Run one helper operation and parse its JSON response.
    pub async fn call(&self, request: Value) -> SandboxResult<Value> {
        let op = request
            .get("op")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let payload = serde_json::to_vec(&request).map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("failed to encode the `{op}` request for the Modal helper: {error}"),
            )
        })?;
        let mut child = tokio::process::Command::new(&self.command[0])
            .args(&self.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!(
                        "failed to run the Modal helper `{}`: {error}",
                        self.command.join(" ")
                    ),
                )
            })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&payload).await.ok();
            stdin.write_all(b"\n").await.ok();
            stdin.shutdown().await.ok();
        }
        let output = child.wait_with_output().await.map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("failed to wait for the Modal helper `{op}` operation: {error}"),
            )
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next_back().unwrap_or("").trim();
        if line.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!(
                    "the Modal helper `{op}` operation produced no response: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            ));
        }
        let response: Value = serde_json::from_str(line).map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("the Modal helper `{op}` operation returned a malformed response: {error}"),
            )
        })?;
        if let Some(detail) = response.get("error").and_then(Value::as_str) {
            let code = match response.get("code").and_then(Value::as_str) {
                Some("not_found") => ErrorCode::WorkspaceReadNotFound,
                Some("write_failed") => ErrorCode::WorkspaceWriteTypeError,
                Some("snapshot_failed") => ErrorCode::SnapshotPersistError,
                _ => ErrorCode::ExecTransportError,
            };
            return Err(SandboxError::new(
                code,
                format!("Modal helper `{op}` failed: {detail}"),
            ));
        }
        Ok(response)
    }
}

/// Creates [`ModalSandboxSession`]s through the bundled Modal helper.
#[derive(Debug, Clone)]
pub struct ModalSandboxClient {
    options: ModalSandboxClientOptions,
    snapshot: Option<SnapshotSpec>,
}

impl ModalSandboxClient {
    /// A client for a Modal app.
    pub fn new(options: ModalSandboxClientOptions) -> Self {
        ModalSandboxClient {
            options,
            snapshot: None,
        }
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// The `create` request the helper is given for `manifest`.
    fn create_request(&self, manifest: &Manifest, image_id: Option<&str>) -> Value {
        let mut request = Map::new();
        request.insert("op".to_string(), Value::String("create".to_string()));
        request.insert(
            "app_name".to_string(),
            Value::String(self.options.app_name.clone()),
        );
        match image_id.or(self.options.image_id.as_deref()) {
            Some(image_id) => {
                request.insert("image_id".to_string(), Value::String(image_id.to_string()));
            }
            None => {
                request.insert(
                    "image_tag".to_string(),
                    Value::String(
                        self.options
                            .image_tag
                            .clone()
                            .unwrap_or_else(|| DEFAULT_PYTHON_SANDBOX_IMAGE.to_string()),
                    ),
                );
            }
        }
        request.insert(
            "workdir".to_string(),
            Value::String(manifest.root().to_string()),
        );
        request.insert(
            "env".to_string(),
            Value::Object(
                manifest
                    .environment()
                    .iter()
                    .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                    .collect(),
            ),
        );
        request.insert(
            "encrypted_ports".to_string(),
            Value::Array(
                self.options
                    .exposed_ports
                    .iter()
                    .map(|port| Value::from(*port))
                    .collect(),
            ),
        );
        request.insert(
            "volumes".to_string(),
            Value::Array(ModalCloudBucketMountStrategy::volume_specs(manifest)),
        );
        request.insert(
            "timeout".to_string(),
            Value::from(self.options.timeout_secs),
        );
        request.insert(
            "use_sleep_cmd".to_string(),
            Value::Bool(self.options.use_sleep_cmd),
        );
        if let Some(gpu) = &self.options.gpu {
            request.insert("gpu".to_string(), Value::String(gpu.clone()));
        }
        if let Some(cpu) = self.options.cpu {
            request.insert("cpu".to_string(), Value::from(cpu));
        }
        if let Some(memory) = self.options.memory_mb {
            request.insert("memory".to_string(), Value::from(memory));
        }
        if let Some(idle_timeout) = self.options.idle_timeout_secs {
            request.insert("idle_timeout".to_string(), Value::from(idle_timeout));
        }
        if let Some(version) = &self.options.image_builder_version {
            request.insert(
                "image_builder_version".to_string(),
                Value::String(version.clone()),
            );
        }
        if let Some(timeout) = self.options.sandbox_create_timeout {
            request.insert(
                "create_timeout".to_string(),
                Value::from(timeout.as_secs_f64()),
            );
        }
        Value::Object(request)
    }

    async fn create_session(
        &self,
        manifest: Manifest,
        image_id: Option<&str>,
    ) -> SandboxResult<ModalSandboxSession> {
        let helper = ModalHelper::new(self.options.resolve_helper_command()?);
        let response = helper
            .call(self.create_request(&manifest, image_id))
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("failed to create the Modal sandbox: {error}"),
                )
            })?;
        let sandbox_id = response
            .get("sandbox_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    "the Modal helper returned no sandbox id",
                )
            })?
            .to_string();
        Ok(
            ModalSandboxSession::new(helper, sandbox_id, manifest, self.options.clone())
                .with_image_id(
                    response
                        .get("image_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                )
                .with_snapshot(self.snapshot.clone()),
        )
    }
}

#[async_trait]
impl SandboxClient for ModalSandboxClient {
    fn backend_id(&self) -> &str {
        "modal"
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
        if state.backend != "modal" {
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
        let helper = ModalHelper::new(self.options.resolve_helper_command()?);
        if let Some(sandbox_id) = state.provider_id.clone().filter(|id| !id.is_empty()) {
            let running = helper
                .call(json!({"op": "poll", "sandbox_id": sandbox_id}))
                .await
                .ok()
                .and_then(|response| response.get("running").and_then(Value::as_bool))
                .unwrap_or(false);
            if running {
                return Ok(Arc::new(
                    ModalSandboxSession::new(helper, sandbox_id, manifest, self.options.clone())
                        .with_snapshot(Some(state.snapshot.clone()))
                        .with_snapshot_id(state.snapshot_id.clone())
                        .resuming(),
                ));
            }
        }
        // The recorded sandbox is gone: a replacement starts from the recorded snapshot.
        Ok(Arc::new(
            self.create_session(manifest, None)
                .await?
                .with_snapshot(Some(state.snapshot.clone()))
                .with_snapshot_id(state.snapshot_id.clone())
                .resuming(),
        ))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// A sandbox session whose workspace lives inside a Modal sandbox.
#[derive(Debug)]
pub struct ModalSandboxSession {
    helper: ModalHelper,
    sandbox_id: std::sync::Mutex<String>,
    session_id: String,
    manifest: Manifest,
    options: ModalSandboxClientOptions,
    image_id: std::sync::Mutex<Option<String>>,
    started: AtomicBool,
    terminated: Mutex<bool>,
    ptys: PtyManager,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
}

impl ModalSandboxSession {
    /// Attach a session to a sandbox the helper already created.
    pub fn new(
        helper: ModalHelper,
        sandbox_id: String,
        manifest: Manifest,
        options: ModalSandboxClientOptions,
    ) -> Self {
        // The session id is its own value, as in the Python port: restoring a filesystem
        // snapshot replaces the Modal sandbox behind the same session.
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        ModalSandboxSession {
            helper,
            snapshot_id: session_id.clone(),
            session_id,
            sandbox_id: std::sync::Mutex::new(sandbox_id),
            manifest,
            options,
            image_id: std::sync::Mutex::new(None),
            started: AtomicBool::new(false),
            terminated: Mutex::new(false),
            ptys: PtyManager::new(),
            snapshot: None,
            resuming: false,
        }
    }

    /// Record the Modal image the sandbox booted from.
    pub fn with_image_id(self, image_id: Option<String>) -> Self {
        if let Ok(mut slot) = self.image_id.lock() {
            *slot = image_id;
        }
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

    /// The helper command behind this session.
    pub fn helper(&self) -> &ModalHelper {
        &self.helper
    }

    /// Modal's current id for this sandbox.
    ///
    /// Restoring a filesystem snapshot replaces the sandbox, so this can change over a session's
    /// lifetime while [`id`](SandboxSession::id) stays stable.
    pub fn sandbox_id(&self) -> String {
        self.sandbox_id
            .lock()
            .map(|id| id.clone())
            .unwrap_or_default()
    }

    /// The Modal image the sandbox booted from, once one is known.
    pub fn image_id(&self) -> Option<String> {
        self.image_id.lock().ok().and_then(|id| id.clone())
    }

    /// Snapshot the sandbox filesystem and return the resulting Modal image id.
    pub async fn snapshot_filesystem(&self) -> SandboxResult<String> {
        let response = self
            .helper
            .call(json!({
                "op": "snapshot_filesystem",
                "sandbox_id": self.sandbox_id(),
                "timeout": self.options.snapshot_filesystem_timeout.as_secs(),
            }))
            .await?;
        let image_id = response
            .get("image_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    "`snapshot_filesystem` returned no image id",
                )
            })?
            .to_string();
        if let Ok(mut slot) = self.image_id.lock() {
            *slot = Some(image_id.clone());
        }
        Ok(image_id)
    }

    async fn helper_exec(
        &self,
        argv: &[String],
        workdir: &str,
        timeout: Option<Duration>,
    ) -> SandboxResult<ExecResult> {
        self.helper_exec_with_env(argv, workdir, timeout, &BTreeMap::new())
            .await
    }

    async fn helper_exec_with_env(
        &self,
        argv: &[String],
        workdir: &str,
        timeout: Option<Duration>,
        env: &BTreeMap<String, String>,
    ) -> SandboxResult<ExecResult> {
        let mut request = Map::new();
        request.insert("op".to_string(), Value::String("exec".to_string()));
        request.insert("sandbox_id".to_string(), Value::String(self.sandbox_id()));
        request.insert(
            "argv".to_string(),
            Value::Array(argv.iter().map(|a| Value::String(a.clone())).collect()),
        );
        request.insert("workdir".to_string(), Value::String(workdir.to_string()));
        if !env.is_empty() {
            // The sandbox was created with the manifest environment, so this carries only what a
            // command adds on top of it.
            request.insert(
                "env".to_string(),
                Value::Object(
                    env.iter()
                        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                        .collect(),
                ),
            );
        }
        if let Some(timeout) = timeout {
            // Modal's exec timeout is whole seconds and never shorter than its own floor.
            request.insert(
                "timeout".to_string(),
                Value::from(timeout.as_secs().max(DEFAULT_TIMEOUT.as_secs())),
            );
        }
        let response = self.helper.call(Value::Object(request)).await?;
        Ok(ExecResult {
            stdout: decode_base64_field(&response, "stdout")?,
            stderr: decode_base64_field(&response, "stderr")?,
            exit_code: response
                .get("exit_code")
                .and_then(Value::as_i64)
                .unwrap_or_default() as i32,
        })
    }

    /// Replace the sandbox with one booted from `image_id`, as a snapshot restore does.
    async fn restore_from_image(&self, image_id: &str) -> SandboxResult<()> {
        let replacement = ModalSandboxClient {
            options: self.options.clone(),
            snapshot: self.snapshot.clone(),
        }
        .create_session(self.manifest.clone(), Some(image_id))
        .await
        .map_err(|error| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("failed to restore the Modal filesystem snapshot: {error}"),
            )
        })?;
        let new_sandbox_id = replacement.sandbox_id();
        let previous = match self.sandbox_id.lock() {
            Ok(mut sandbox_id) => std::mem::replace(&mut *sandbox_id, new_sandbox_id),
            Err(_) => {
                return Err(SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    "the Modal sandbox id is poisoned after an earlier panic",
                ));
            }
        };
        if let Ok(mut slot) = self.image_id.lock() {
            *slot = Some(image_id.to_string());
        }
        // The sandbox the snapshot replaced keeps billing until it is terminated.
        let _ = self
            .helper
            .call(json!({"op": "terminate", "sandbox_id": previous}))
            .await;
        Ok(())
    }

    /// Run the sandbox-side cleanup of PTY processes that finished or were pruned.
    async fn run_pty_cleanup(&self, commands: Vec<String>) {
        for command in commands {
            let _ = self.exec(ExecRequest::shell(command)).await;
        }
    }

    /// Kill every PTY process, including the sandbox-side process each one proxies.
    async fn terminate_pty_sessions(&self) {
        for command in self.ptys.terminate_all() {
            let _ = self.exec(ExecRequest::shell(command)).await;
        }
    }
}

#[async_trait]
impl SandboxSession for ModalSandboxSession {
    fn id(&self) -> &str {
        &self.session_id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "modal"
    }

    async fn start(&self) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        let result = self
            .helper_exec(
                &[
                    "mkdir".to_string(),
                    "-p".to_string(),
                    "--".to_string(),
                    root.clone(),
                ],
                "/",
                Some(self.options.exec_timeout),
            )
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("failed to start session: {error}"),
                )
            })?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "failed to create the workspace root `{root}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        self.started.store(true, Ordering::SeqCst);
        let applied = if self.resuming {
            match restore_session_snapshot(self).await {
                Ok(false) => self.manifest.apply_ephemeral(self).await.map(|_| ()),
                Ok(true) => Ok(()),
                Err(error) => Err(error),
            }
        } else {
            self.manifest.apply(self).await.map(|_| ())
        };
        applied.inspect_err(|_| self.started.store(false, Ordering::SeqCst))?;
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        self.terminate_pty_sessions().await;
        self.started.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.terminate_pty_sessions().await;
        let mut terminated = self.terminated.lock().await;
        self.started.store(false, Ordering::SeqCst);
        if *terminated {
            return Ok(());
        }
        let response = self
            .helper
            .call(json!({"op": "terminate", "sandbox_id": self.sandbox_id()}))
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStopError,
                    format!("failed to terminate the Modal sandbox: {error}"),
                )
            })?;
        if response.get("ok").and_then(Value::as_bool) == Some(false) {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                format!(
                    "failed to terminate the Modal sandbox: {}",
                    response
                        .get("detail")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                ),
            ));
        }
        *terminated = true;
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        if !self.started.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let response = self
            .helper
            .call(json!({"op": "poll", "sandbox_id": self.sandbox_id()}))
            .await;
        Ok(response
            .ok()
            .and_then(|response| response.get("running").and_then(Value::as_bool))
            .unwrap_or(false))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let workdir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        let argv = request.resolved_argv();
        // A command without a timeout runs unbounded, as in the Python port: the timeout path
        // below retires the whole sandbox, which must not happen to an ordinary command merely
        // because it outlived a default.
        let env = request.env.clone();
        let Some(timeout) = request.timeout else {
            return self.helper_exec_with_env(&argv, &workdir, None, &env).await;
        };
        match tokio::time::timeout(
            timeout,
            self.helper_exec_with_env(&argv, &workdir, Some(timeout), &env),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                // Python terminates the sandbox on an exec timeout, because a Modal sandbox has
                // no way to cancel a command that is already running.
                let _ = self
                    .helper
                    .call(json!({"op": "terminate", "sandbox_id": self.sandbox_id()}))
                    .await;
                self.started.store(false, Ordering::SeqCst);
                Err(SandboxError::new(
                    ErrorCode::ExecTimeout,
                    format!(
                        "Command `{}` timed out after {timeout:?}",
                        request.command.join(" ")
                    ),
                ))
            }
        }
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        let response = self
            .helper
            .call(json!({
                "op": "read_file",
                "sandbox_id": self.sandbox_id(),
                "path": resolved,
            }))
            .await?;
        decode_base64_field(&response, "data")
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        self.helper
            .call(json!({
                "op": "write_file",
                "sandbox_id": self.sandbox_id(),
                "path": resolved,
                "data": BASE64.encode(contents),
            }))
            .await
            .map(|_| ())
    }

    fn supports_pty(&self) -> bool {
        true
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let workdir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        // The helper process is only a proxy: killing it leaves the command running inside the
        // Modal sandbox, so the command records its pid and the session kills it there during
        // teardown, exactly as the Docker backend does.
        let pid_path = crate::sandbox::workspace_paths::provider_scratch_path(&format!(
            "modal-pty-{}.pid",
            uuid::Uuid::new_v4().simple()
        ));
        let inner = format!(
            "echo $$ > {pid}; exec {argv}",
            pid = crate::sandbox::session::shell_quote(&pid_path),
            argv = request
                .resolved_argv()
                .iter()
                .map(|part| crate::sandbox::session::shell_quote(part))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let payload = json!({
            "op": "pty",
            "sandbox_id": self.sandbox_id(),
            "argv": ["sh", "-c", inner],
            "tty": request.tty,
            "workdir": workdir,
        });
        // The helper reads its request from stdin, but a PTY's stdin belongs to the model's
        // keystrokes, so the request travels on the command line instead.
        let mut argv = self.helper.command.clone();
        argv.push(serde_json::to_string(&payload).map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("failed to encode the Modal PTY request: {error}"),
            )
        })?);
        let spawn = PtySpawn {
            argv,
            cwd: None,
            env: Vec::new(),
            clear_env: false,
            tty: request.tty,
            remote_kill_command: Some(format!(
                "if [ -f {pid} ]; then kill -TERM \"$(cat {pid})\" 2>/dev/null || true; \
                 rm -f {pid}; fi",
                pid = crate::sandbox::session::shell_quote(&pid_path)
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

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        if self.options.workspace_persistence == ModalWorkspacePersistence::SnapshotFilesystem {
            let image_id = self.snapshot_filesystem().await?;
            return Ok(encode_snapshot_filesystem_ref(&image_id));
        }
        // The tar flow is the trait's own `tar -czf -` over `exec`.
        default_persist_workspace(self).await
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        if let Some(image_id) = decode_snapshot_filesystem_ref(archive) {
            return self.restore_from_image(&image_id).await;
        }
        let root = self.manifest.root().to_string();
        // The archive is staged through the helper's own chunked stdin writer: a base64 command
        // argument would exceed the kernel's per-argument limit for any real workspace.
        let staged = crate::sandbox::workspace_paths::provider_scratch_path(&format!(
            "modal-hydrate-{}.tar.gz",
            uuid::Uuid::new_v4().simple()
        ));
        self.helper
            .call(json!({
                "op": "write_file",
                "sandbox_id": self.sandbox_id(),
                "path": staged,
                "data": BASE64.encode(archive),
            }))
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    format!("Failed to upload the workspace archive: {error}"),
                )
            })?;
        let script = format!(
            "mkdir -p {root} && tar -xzf {staged} -C {root}",
            root = crate::sandbox::session::shell_quote(&root),
            staged = crate::sandbox::session::shell_quote(&staged),
        );
        // No timeout: extracting a large workspace must not be cut short, and the Python port
        // does not bound it either.
        let result = self
            .helper_exec(&["sh".to_string(), "-c".to_string(), script], "/", None)
            .await;
        let _ = self
            .helper_exec(
                &[
                    "rm".to_string(),
                    "-f".to_string(),
                    "--".to_string(),
                    staged.clone(),
                ],
                "/",
                None,
            )
            .await;
        let result = result?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!(
                    "Failed to restore the Modal workspace: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    fn session_state(&self) -> SandboxSessionState {
        let mut state = SandboxSessionState::new(
            "modal",
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.sandbox_id()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default());
        state.snapshot_id = self.snapshot_id.clone();
        state.exposed_ports = self.options.exposed_ports.clone();
        state
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }
}

/// The trait's default workspace archive, reachable from the overridden method.
async fn default_persist_workspace(session: &ModalSandboxSession) -> SandboxResult<Vec<u8>> {
    let root = session.workspace_root().to_string();
    let mut command = format!(
        "tar -czf - -C {}",
        crate::sandbox::session::shell_quote(&root)
    );
    for path in session.manifest().ephemeral_paths() {
        if let Some(relative) = path
            .strip_prefix(root.trim_end_matches('/'))
            .map(|relative| relative.trim_start_matches('/'))
            .filter(|relative| !relative.is_empty())
        {
            command.push_str(&format!(
                " --exclude {}",
                crate::sandbox::session::shell_quote(&format!("./{relative}"))
            ));
        }
    }
    command.push_str(" .");
    let result = session.exec(ExecRequest::shell(command)).await?;
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

/// The envelope that lets a Modal filesystem snapshot travel through a [`Snapshot`].
pub fn encode_snapshot_filesystem_ref(image_id: &str) -> Vec<u8> {
    let body = json!({
        "snapshot_id": image_id,
        "workspace_persistence": "snapshot_filesystem",
    });
    let mut payload = SNAPSHOT_FILESYSTEM_MAGIC.to_vec();
    payload.extend_from_slice(serde_json::to_string(&body).unwrap_or_default().as_bytes());
    payload
}

/// The Modal image id inside a snapshot envelope, when the payload is one.
pub fn decode_snapshot_filesystem_ref(payload: &[u8]) -> Option<String> {
    let body = payload.strip_prefix(SNAPSHOT_FILESYSTEM_MAGIC)?;
    let parsed: Value = serde_json::from_slice(body).ok()?;
    parsed
        .get("snapshot_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

fn decode_base64_field(response: &Value, field: &str) -> SandboxResult<Vec<u8>> {
    let encoded = response.get(field).and_then(Value::as_str).unwrap_or("");
    BASE64.decode(encoded).map_err(|error| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("the Modal helper returned an undecodable `{field}` field: {error}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_create_request() {
        let client = ModalSandboxClient::new(
            ModalSandboxClientOptions::new("agents")
                .with_image_tag("python:3.12-slim")
                .with_gpu("A100")
                .with_cpu(2.0)
                .with_memory_mb(4096)
                .with_exposed_ports([8080])
                .with_idle_timeout_secs(60)
                .with_sandbox_create_timeout(Duration::from_secs(45)),
        );
        let request = client.create_request(&Manifest::new().with_env("FOO", "bar"), None);
        assert_eq!(request["op"], "create");
        assert_eq!(request["app_name"], "agents");
        assert_eq!(request["image_tag"], "python:3.12-slim");
        assert_eq!(request["workdir"], "/workspace");
        assert_eq!(request["env"], json!({"FOO": "bar"}));
        assert_eq!(request["encrypted_ports"], json!([8080]));
        assert_eq!(request["gpu"], "A100");
        assert_eq!(request["cpu"], 2.0);
        assert_eq!(request["memory"], 4096);
        assert_eq!(request["timeout"], 300);
        assert_eq!(request["idle_timeout"], 60);
        assert_eq!(request["use_sleep_cmd"], true);
        assert_eq!(
            request["image_builder_version"],
            DEFAULT_IMAGE_BUILDER_VERSION
        );
        assert_eq!(request["create_timeout"], 45.0);
        assert_eq!(request["volumes"], json!([]));
    }

    #[test]
    fn prefers_an_image_id_over_a_tag() {
        let client = ModalSandboxClient::new(
            ModalSandboxClientOptions::new("agents").with_image_tag("python:3.12-slim"),
        );
        let request = client.create_request(&Manifest::new(), Some("im-123"));
        assert_eq!(request["image_id"], "im-123");
        assert!(request.get("image_tag").is_none());
    }

    #[test]
    fn defaults_to_the_shared_python_image() {
        let client = ModalSandboxClient::new(ModalSandboxClientOptions::new("agents"));
        let request = client.create_request(&Manifest::new(), None);
        assert_eq!(request["image_tag"], DEFAULT_PYTHON_SANDBOX_IMAGE);
    }

    #[test]
    fn round_trips_a_snapshot_reference() {
        let payload = encode_snapshot_filesystem_ref("im-abc");
        assert!(payload.starts_with(SNAPSHOT_FILESYSTEM_MAGIC));
        assert_eq!(
            decode_snapshot_filesystem_ref(&payload).as_deref(),
            Some("im-abc")
        );
        assert!(decode_snapshot_filesystem_ref(b"plain tar bytes").is_none());
    }

    #[test]
    fn resolves_the_helper_command() {
        let options = ModalSandboxClientOptions::new("agents")
            .with_helper_command(["python3", "/tmp/fake_modal_helper.py"]);
        assert_eq!(
            options.resolve_helper_command().unwrap(),
            vec!["python3", "/tmp/fake_modal_helper.py"]
        );
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            ModalSandboxClientOptions::new("agents")
                .with_helper_command(empty)
                .resolve_helper_command()
                .unwrap_err()
                .code,
            ErrorCode::SandboxConfigInvalid
        );
    }

    #[test]
    fn runs_the_bundled_helper_from_source() {
        // The helper source is passed to the interpreter directly, so nothing is written to a
        // shared temporary directory where another user could replace it.
        let command = ModalSandboxClientOptions::new("agents")
            .with_python("python3")
            .resolve_helper_command()
            .unwrap();
        assert_eq!(command[0], "python3");
        assert_eq!(command[1], "-c");
        assert_eq!(command[2], HELPER_SOURCE);
        assert!(HELPER_SOURCE.contains("def _op_exec"));
    }
}
