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

use super::super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, assert_exposed_port_configured,
    exposed_port_unavailable,
};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyManager, PtySpawn, PtyWriteRequest};
use crate::sandbox::retry::{RetryPolicy, is_transient_sandbox_error, retry_async};
use crate::sandbox::sandboxes::docker::DEFAULT_PYTHON_SANDBOX_IMAGE;
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot,
    with_ephemeral_mounts_removed,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, SnapshotFingerprint};
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
/// The magic prefix of a snapshot payload that references a Modal directory snapshot.
const SNAPSHOT_DIRECTORY_MAGIC: &[u8] = b"MODAL_SANDBOX_DIR_SNAPSHOT_V1\n";

/// How a Modal workspace is captured into a snapshot.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModalWorkspacePersistence {
    /// Stream a gzipped tar of the workspace out of the sandbox.
    #[default]
    Tar,
    /// Call Modal's `snapshot_filesystem` and store a reference to the resulting image.
    SnapshotFilesystem,
    /// Call Modal's `snapshot_directory` on the workspace root and reattach it with
    /// `mount_image` on restore.
    ///
    /// Unlike [`SnapshotFilesystem`](Self::SnapshotFilesystem), restoring keeps the running
    /// sandbox and remounts only the workspace, so a restore does not replace the sandbox id.
    SnapshotDirectory,
}

/// Which Modal image a sandbox boots from.
///
/// Mirrors `agents.extensions.sandbox.modal.sandbox.ModalImageSelector`, minus its `image` kind:
/// Python can hold a live `modal.Image` because it runs in the same interpreter as the Modal
/// SDK, while this port drives Modal through a stateless subprocess that addresses everything by
/// id over one JSON request. Python itself only persists `image.object_id` from a live image, so
/// [`Id`](Self::Id) carries everything that survives a process boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalImageSelector {
    /// An existing Modal image object id, such as the id of an earlier snapshot.
    Id(String),
    /// A registry tag, which is the name form: `debian:bookworm`, `ghcr.io/org/img:tag`.
    Tag(String),
}

impl ModalImageSelector {
    /// Select an image by its Modal object id. Mirrors `ModalImageSelector.from_id`.
    pub fn from_id(image_id: impl Into<String>) -> Self {
        ModalImageSelector::Id(image_id.into())
    }

    /// Select an image by registry tag. Mirrors `ModalImageSelector.from_tag`.
    pub fn from_tag(image_tag: impl Into<String>) -> Self {
        ModalImageSelector::Tag(image_tag.into())
    }
}

/// Which Modal sandbox a session runs in.
///
/// Mirrors `agents.extensions.sandbox.modal.sandbox.ModalSandboxSelector`, minus its `sandbox`
/// kind, for the same reason [`ModalImageSelector`] drops `image`: Python stores only
/// `sandbox.object_id` from a live `modal.Sandbox`, and an id is what crosses the helper
/// boundary. Python has no name form for a sandbox, so neither does this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalSandboxSelector {
    /// An existing Modal sandbox id to adopt instead of creating a sandbox.
    Id(String),
}

impl ModalSandboxSelector {
    /// Adopt the sandbox with this Modal id. Mirrors `ModalSandboxSelector.from_id`.
    pub fn from_id(sandbox_id: impl Into<String>) -> Self {
        ModalSandboxSelector::Id(sandbox_id.into())
    }
}

/// Settings used when creating a Modal-backed session.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq)]
pub struct ModalSandboxClientOptions {
    /// The Modal app the sandbox belongs to.
    pub app_name: String,
    /// The registry tag the sandbox image is built from.
    pub image_tag: Option<String>,
    /// An existing Modal image id, which takes precedence over the tag.
    pub image_id: Option<String>,
    /// An existing Modal sandbox id to adopt instead of creating one.
    pub sandbox_id: Option<String>,
    /// How the workspace is captured into a snapshot.
    #[serde(default)]
    pub workspace_persistence: ModalWorkspacePersistence,
    /// Ports Modal exposes over an encrypted tunnel.
    #[serde(default)]
    pub exposed_ports: Vec<u16>,
    /// The GPU reservation, such as `A100` or `H100:8`.
    pub gpu: Option<String>,
    /// The sandbox lifetime in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u32,
    /// The idle timeout in seconds.
    pub idle_timeout_secs: Option<u32>,
    /// The CPU reservation, which Modal reads as the request of a `(request, limit)` pair.
    pub cpu: Option<f64>,
    /// The CPU limit, which turns [`cpu`](Self::cpu) into Modal's `(request, limit)` pair.
    pub cpu_limit: Option<f64>,
    /// The memory reservation in MiB, which Modal reads as the request of a `(request, limit)`
    /// pair.
    pub memory_mb: Option<u32>,
    /// The memory limit in MiB, which turns [`memory_mb`](Self::memory_mb) into Modal's
    /// `(request, limit)` pair.
    pub memory_limit_mb: Option<u32>,
    /// Override the image command with `sleep infinity` so the sandbox stays up for `exec`.
    #[serde(default = "default_use_sleep_cmd")]
    pub use_sleep_cmd: bool,
    /// The Modal image builder version.
    #[serde(default = "default_image_builder_version")]
    pub image_builder_version: Option<String>,
    /// How long `Sandbox.create` may take.
    pub sandbox_create_timeout: Option<Duration>,
    /// The default exec timeout.
    #[serde(default = "default_exec_timeout")]
    pub exec_timeout: Duration,
    /// How long `snapshot_filesystem` may take.
    #[serde(default = "default_snapshot_timeout")]
    pub snapshot_filesystem_timeout: Duration,
    /// How long restoring a filesystem snapshot may take.
    ///
    /// Restoring boots a replacement sandbox from the snapshot image, which is a different call
    /// from the capture, so Modal bounds it separately.
    #[serde(default = "default_snapshot_timeout")]
    pub snapshot_filesystem_restore_timeout: Duration,
    /// The command that runs the helper, or the bundled helper when unset.
    pub helper_command: Option<Vec<String>>,
    /// The interpreter the bundled helper runs under.
    pub python: Option<String>,
}

fn default_timeout_secs() -> u32 {
    300
}

fn default_use_sleep_cmd() -> bool {
    true
}

fn default_image_builder_version() -> Option<String> {
    Some(DEFAULT_IMAGE_BUILDER_VERSION.to_string())
}

fn default_exec_timeout() -> Duration {
    DEFAULT_TIMEOUT
}

fn default_snapshot_timeout() -> Duration {
    DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT
}

// The helper command may carry a token in its environment arguments, so `Debug` names only the
// program and how many arguments follow.
impl std::fmt::Debug for ModalSandboxClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let helper = self
            .helper_command
            .as_ref()
            .map(|command| (command.first().cloned(), command.len().saturating_sub(1)));
        f.debug_struct("ModalSandboxClientOptions")
            .field("app_name", &self.app_name)
            .field("image_tag", &self.image_tag)
            .field("image_id", &self.image_id)
            .field("sandbox_id", &self.sandbox_id)
            .field("workspace_persistence", &self.workspace_persistence)
            .field("exposed_ports", &self.exposed_ports)
            .field("gpu", &self.gpu)
            .field("timeout_secs", &self.timeout_secs)
            .field("idle_timeout_secs", &self.idle_timeout_secs)
            .field("cpu", &self.cpu)
            .field("cpu_limit", &self.cpu_limit)
            .field("memory_mb", &self.memory_mb)
            .field("memory_limit_mb", &self.memory_limit_mb)
            .field("use_sleep_cmd", &self.use_sleep_cmd)
            .field("image_builder_version", &self.image_builder_version)
            .field("sandbox_create_timeout", &self.sandbox_create_timeout)
            .field("exec_timeout", &self.exec_timeout)
            .field(
                "snapshot_filesystem_timeout",
                &self.snapshot_filesystem_timeout,
            )
            .field(
                "snapshot_filesystem_restore_timeout",
                &self.snapshot_filesystem_restore_timeout,
            )
            .field("helper_command", &helper)
            .field("python", &self.python)
            .finish()
    }
}

impl ModalSandboxClientOptions {
    /// Options for a Modal app.
    pub fn new(app_name: impl Into<String>) -> Self {
        ModalSandboxClientOptions {
            app_name: app_name.into(),
            image_tag: None,
            image_id: None,
            sandbox_id: None,
            workspace_persistence: ModalWorkspacePersistence::Tar,
            exposed_ports: Vec::new(),
            gpu: None,
            timeout_secs: 300,
            idle_timeout_secs: None,
            cpu: None,
            cpu_limit: None,
            memory_mb: None,
            memory_limit_mb: None,
            use_sleep_cmd: true,
            image_builder_version: Some(DEFAULT_IMAGE_BUILDER_VERSION.to_string()),
            sandbox_create_timeout: None,
            exec_timeout: DEFAULT_TIMEOUT,
            snapshot_filesystem_timeout: DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT,
            snapshot_filesystem_restore_timeout: DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT,
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

    /// Choose the image with one selector instead of juggling the tag and the id.
    ///
    /// Mirrors the `image=ModalImageSelector(...)` argument of Python's `ModalSandboxClient`.
    /// Setting a selector clears the other form, so the last call wins.
    pub fn with_image(mut self, image: ModalImageSelector) -> Self {
        match image {
            ModalImageSelector::Id(image_id) => {
                self.image_id = Some(image_id);
                self.image_tag = None;
            }
            ModalImageSelector::Tag(image_tag) => {
                self.image_tag = Some(image_tag);
                self.image_id = None;
            }
        }
        self
    }

    /// Adopt an existing Modal sandbox instead of creating one.
    ///
    /// Mirrors the `sandbox=ModalSandboxSelector(...)` argument of Python's
    /// `ModalSandboxClient`: `create` attaches to the named sandbox, which must still be
    /// running, rather than calling `Sandbox.create`.
    pub fn with_sandbox(mut self, sandbox: ModalSandboxSelector) -> Self {
        let ModalSandboxSelector::Id(sandbox_id) = sandbox;
        self.sandbox_id = Some(sandbox_id);
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

    /// Reserve CPU, leaving the limit to Modal.
    pub fn with_cpu(mut self, cpu: f64) -> Self {
        self.cpu = Some(cpu);
        self.cpu_limit = None;
        self
    }

    /// Reserve CPU with an explicit limit, Modal's `(request, limit)` pair.
    ///
    /// Modal rejects a limit below the request, and rejects either end being zero; the error
    /// comes back from the control plane rather than being anticipated here, so the message the
    /// caller sees is Modal's own.
    pub fn with_cpu_range(mut self, request: f64, limit: f64) -> Self {
        self.cpu = Some(request);
        self.cpu_limit = Some(limit);
        self
    }

    /// Reserve memory, in MiB, leaving the limit to Modal.
    pub fn with_memory_mb(mut self, memory_mb: u32) -> Self {
        self.memory_mb = Some(memory_mb);
        self.memory_limit_mb = None;
        self
    }

    /// Reserve memory with an explicit limit, in MiB: Modal's `(request, limit)` pair.
    ///
    /// Modal rejects a limit below the request, and reports it from the control plane.
    pub fn with_memory_mb_range(mut self, request: u32, limit: u32) -> Self {
        self.memory_mb = Some(request);
        self.memory_limit_mb = Some(limit);
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

    /// Bound a filesystem snapshot restore, which boots a replacement sandbox.
    pub fn with_snapshot_filesystem_restore_timeout(mut self, timeout: Duration) -> Self {
        self.snapshot_filesystem_restore_timeout = timeout;
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
        let mut command = tokio::process::Command::new(&self.command[0]);
        command
            .args(&self.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // On unix a dropped call asks the helper to stop with SIGTERM (see `TerminateOnDrop`), so a
        // bounded read can close its provider descriptor on the way out. SIGKILL would skip it.
        #[cfg(not(unix))]
        command.kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| {
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
        let terminate = TerminateOnDrop::new(child.id());
        let output = child.wait_with_output().await;
        terminate.disarm();
        let output = output.map_err(|error| {
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
                Some("bounded_read_failed") => {
                    // Nothing the provider said is repeated: only the reason and whether a
                    // retry may help.
                    let mut error = SandboxError::new(
                        ErrorCode::WorkspaceArchiveReadError,
                        "Failed to read the workspace path",
                    )
                    .with_context("reason", "bounded_read_failed");
                    if let Some(retryable) = response.get("retryable").and_then(Value::as_bool) {
                        error = error.with_retryable(retryable);
                    }
                    return Err(error);
                }
                _ => ErrorCode::ExecTransportError,
            };
            let mut error =
                SandboxError::new(code, format!("Modal helper `{op}` failed: {detail}"));
            // The helper classifies the Modal exception type; a caller that retries on a
            // transient failure needs to know which ones are permanent.
            if let Some(retryable) = response.get("retryable").and_then(Value::as_bool) {
                error = error.with_retryable(retryable);
            }
            return Err(error);
        }
        Ok(response)
    }
}

/// Sends SIGTERM to a helper process whose call was dropped before it answered.
///
/// A dropped `call` is a cancelled operation. The helper turns SIGTERM into a cancellation of its
/// in-flight work, so a bounded read's `finally` still closes the provider descriptor.
struct TerminateOnDrop {
    pid: Option<u32>,
}

impl TerminateOnDrop {
    fn new(pid: Option<u32>) -> Self {
        TerminateOnDrop { pid }
    }

    /// The call completed: the helper has answered and nothing needs stopping.
    fn disarm(mut self) {
        self.pid = None;
    }
}

impl Drop for TerminateOnDrop {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.pid.take() {
            // The system `kill` sends the signal; a pid that already exited is a no-op for it.
            let _ = std::process::Command::new("/bin/kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
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
        // Modal takes either a scalar reservation or a `(request, limit)` pair. The pair travels
        // as a two-element array and the helper turns it back into a tuple, which is the only
        // shape Modal's own resource conversion recognizes.
        if let Some(cpu) = self.options.cpu {
            request.insert(
                "cpu".to_string(),
                match self.options.cpu_limit {
                    Some(limit) => Value::Array(vec![Value::from(cpu), Value::from(limit)]),
                    None => Value::from(cpu),
                },
            );
        }
        if let Some(memory) = self.options.memory_mb {
            request.insert(
                "memory".to_string(),
                match self.options.memory_limit_mb {
                    Some(limit) => Value::Array(vec![Value::from(memory), Value::from(limit)]),
                    None => Value::from(memory),
                },
            );
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

    /// Refuse a manifest that `snapshot_directory` cannot capture.
    ///
    /// Mirrors Python's `_validate_manifest_for_workspace_persistence`: the directory snapshot
    /// replaces the workspace root wholesale, so a Modal cloud bucket mount at or under the root
    /// would be captured — and, on restore, shadowed — instead of staying a live mount.
    fn validate_manifest_for_workspace_persistence(
        &self,
        manifest: &Manifest,
    ) -> SandboxResult<()> {
        if self.options.workspace_persistence != ModalWorkspacePersistence::SnapshotDirectory {
            return Ok(());
        }
        let root = manifest.root().trim_end_matches('/').to_string();
        for mount_path in ModalCloudBucketMountStrategy::mount_paths(manifest) {
            let normalized = mount_path.trim_end_matches('/');
            if normalized == root || normalized.starts_with(&format!("{root}/")) {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "snapshot_directory is not supported when a Modal cloud bucket mount lives \
                     at or under the workspace root",
                )
                .with_context("workspace_root", root)
                .with_context("mount_path", mount_path)
                .with_context("workspace_persistence", "snapshot_directory")
                .with_retryable(false));
            }
        }
        Ok(())
    }

    /// Attach to a sandbox the caller already owns, instead of creating one.
    async fn adopt_session(
        &self,
        manifest: Manifest,
        sandbox_id: &str,
    ) -> SandboxResult<ModalSandboxSession> {
        let helper = ModalHelper::new(self.options.resolve_helper_command()?);
        let running = helper
            .call(json!({"op": "poll", "sandbox_id": sandbox_id}))
            .await?
            .get("running")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !running {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!("the Modal sandbox `{sandbox_id}` is not running"),
            )
            .with_retryable(false));
        }
        Ok(ModalSandboxSession::new(
            helper,
            sandbox_id.to_string(),
            manifest,
            self.options.clone(),
        )
        .with_image_id(self.options.image_id.clone())
        .with_snapshot(self.snapshot.clone()))
    }

    async fn create_session(
        &self,
        manifest: Manifest,
        image_id: Option<&str>,
    ) -> SandboxResult<ModalSandboxSession> {
        // A mount whose path climbs out of the workspace would otherwise be dropped silently.
        manifest.try_mount_targets()?;
        self.validate_manifest_for_workspace_persistence(&manifest)?;
        if image_id.is_none()
            && let Some(sandbox_id) = self
                .options
                .sandbox_id
                .as_deref()
                .filter(|id| !id.is_empty())
        {
            return self.adopt_session(manifest, sandbox_id).await;
        }
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
        // A state whose mount authority was re-supplied by the caller must not reattach the
        // sandbox that holds the old authority: a fresh one is created from the current manifest.
        let reattachable = !state.mount_authority_rebound;
        if let Some(sandbox_id) = state
            .provider_id
            .clone()
            .filter(|id| reattachable && !id.is_empty())
        {
            let running = helper
                .call(json!({"op": "poll", "sandbox_id": sandbox_id}))
                .await
                .ok()
                .and_then(|response| response.get("running").and_then(Value::as_bool))
                .unwrap_or(false);
            if running {
                let record = state
                    .snapshot_fingerprint
                    .clone()
                    .zip(state.snapshot_fingerprint_version.clone())
                    .map(|(fingerprint, version)| SnapshotFingerprint {
                        fingerprint,
                        version,
                    });
                return Ok(Arc::new(
                    ModalSandboxSession::new(helper, sandbox_id, manifest, self.options.clone())
                        .with_snapshot(Some(state.snapshot.clone()))
                        .with_snapshot_id(state.snapshot_id.clone())
                        .with_state_preserved(true)
                        .with_snapshot_fingerprint(record)
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
    /// Whether this session reattached a sandbox that kept its workspace and system state.
    state_preserved: bool,
    /// The workspace fingerprint the stored snapshot was taken from.
    snapshot_fingerprint: std::sync::Mutex<Option<SnapshotFingerprint>>,
    persist_skips: std::sync::Mutex<std::collections::BTreeSet<String>>,
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
            state_preserved: false,
            snapshot_fingerprint: std::sync::Mutex::new(None),
            persist_skips: std::sync::Mutex::new(std::collections::BTreeSet::new()),
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

    /// Mark the sandbox as one that survived the pause with its workspace intact.
    pub fn with_state_preserved(mut self, preserved: bool) -> Self {
        self.state_preserved = preserved;
        self
    }

    /// Carry the workspace fingerprint a resumed snapshot was taken from.
    pub fn with_snapshot_fingerprint(self, record: Option<SnapshotFingerprint>) -> Self {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
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
    ///
    /// The image captures the whole filesystem, so the paths a workspace archive excludes are set
    /// aside first and put back afterwards, exactly as for a directory snapshot.
    pub async fn snapshot_filesystem(&self) -> SandboxResult<String> {
        self.with_excluded_paths_stashed("filesystem", || async {
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
        })
        .await
    }

    /// Snapshot the workspace root into a Modal image with `Sandbox.snapshot_directory`.
    ///
    /// The image captures the whole directory, so the paths a workspace archive excludes —
    /// ephemeral entries and the staged rclone configuration, which holds a mount credential —
    /// are moved aside first and put back afterwards, the way Python's
    /// `_persist_workspace_via_snapshot_directory` does. Without that, a credential would be
    /// baked into an image that outlives the sandbox.
    pub async fn snapshot_directory(&self) -> SandboxResult<String> {
        let root = self.manifest.root().to_string();
        self.with_excluded_paths_stashed("directory", || async {
            // The image captures the whole root, so durable mounts inside it are detached for
            // the snapshot and attached again afterwards.
            let mounts = self.durable_mounts_in_root()?;
            with_ephemeral_mounts_removed(
                self,
                mounts,
                || self.snapshot_directory_image(&root),
                &root,
                ErrorCode::WorkspaceArchiveReadError,
                Some("snapshot_error_before_remount_corruption"),
                true,
            )
            .await
        })
        .await
    }

    /// The durable (non-ephemeral) mounts at or under the workspace root that Modal does not
    /// hold itself. Modal bucket mounts live outside the image, so they are never detached.
    fn durable_mounts_in_root(
        &self,
    ) -> SandboxResult<Vec<(&crate::sandbox::mounts::Mount, String)>> {
        let root = self.manifest.root().trim_end_matches('/').to_string();
        Ok(self
            .manifest
            .try_mount_targets()?
            .into_iter()
            .filter(|(mount, path)| {
                !mount.ephemeral
                    && mount.strategy.strategy_type() != "modal_cloud_bucket"
                    && (path.trim_end_matches('/') == root || path.starts_with(&format!("{root}/")))
            })
            .collect())
    }

    /// Run `operation` with every path a workspace archive excludes set aside, then put them back.
    ///
    /// The stash is restored whether or not `operation` succeeded, so a failure does not leave the
    /// live workspace missing its ephemeral entries. A stash that fails part way is restored too,
    /// because the removal may already have taken some of the paths.
    async fn with_excluded_paths_stashed<T, F, Fut>(
        &self,
        kind: &str,
        operation: F,
    ) -> SandboxResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = SandboxResult<T>>,
    {
        let root = self.manifest.root().to_string();
        let mut excluded =
            crate::sandbox::session::workspace_archive_excludes(&self.manifest, &root)
                .into_iter()
                .map(|relative| relative.trim_start_matches("./").to_string())
                .filter(|relative| !relative.is_empty())
                .collect::<Vec<_>>();
        for skipped in self.persist_workspace_skip_paths() {
            if !excluded.contains(&skipped) {
                excluded.push(skipped);
            }
        }
        if kind == "filesystem" {
            return self
                .with_excluded_paths_held_in_memory(&root, &excluded, operation)
                .await;
        }
        let backup = (!excluded.is_empty()).then(|| {
            crate::sandbox::workspace_paths::provider_scratch_path(&format!(
                "modal-snapshot-{kind}-ephemeral-{}.tar",
                self.session_id
            ))
        });
        if let Some(backup) = &backup
            && let Err(error) = self.stash_excluded_paths(&root, &excluded, backup).await
        {
            // The removal may have taken some paths before it failed.
            let _ = self.restore_excluded_paths(&root, backup).await;
            return Err(error);
        }
        let result = operation().await;
        if let Some(backup) = &backup {
            let restored = self.restore_excluded_paths(&root, backup).await;
            let value = result?;
            restored?;
            return Ok(value);
        }
        result
    }

    /// The filesystem-snapshot variant of [`with_excluded_paths_stashed`]: the set-aside archive
    /// travels through this process, never through the sandbox's disk.
    ///
    /// A filesystem image captures the whole sandbox, `/tmp` included, so a stash file written
    /// there would carry the paths it was meant to keep out (an rclone configuration holds a mount
    /// credential) into an image that outlives the sandbox. The archive is therefore read from the
    /// command's output, and written back only after the image exists.
    async fn with_excluded_paths_held_in_memory<T, F, Fut>(
        &self,
        root: &str,
        excluded: &[String],
        operation: F,
    ) -> SandboxResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = SandboxResult<T>>,
    {
        let quoted = excluded
            .iter()
            .map(|relative| crate::sandbox::session::shell_quote(relative))
            .collect::<Vec<_>>()
            .join(" ");
        // A path that does not exist is skipped, and nothing at all is an empty archive.
        let archive_script = format!(
            "cd -- {root} && {{ for rel in {quoted}; do if [ -e \"$rel\" ]; then \
             printf '%s\\n' \"$rel\"; fi; done; }} | (tar cf - -T - 2>/dev/null || true)",
            root = crate::sandbox::session::shell_quote(root),
        );
        let archived = self
            .helper_exec(
                &["sh".to_string(), "-lc".to_string(), archive_script],
                "/",
                None,
            )
            .await?;
        let backup = archived.stdout;
        let mut absolute = vec!["rm".to_string(), "-rf".to_string(), "--".to_string()];
        absolute.extend(
            excluded
                .iter()
                .map(|relative| format!("{}/{relative}", root.trim_end_matches('/'))),
        );
        let removed = if backup.is_empty() {
            Ok(())
        } else {
            self.helper_exec(&absolute, "/", None)
                .await
                .and_then(|removed| {
                    if removed.ok() {
                        Ok(())
                    } else {
                        Err(SandboxError::new(
                            ErrorCode::SnapshotPersistError,
                            format!(
                                "failed to set aside the paths a Modal snapshot must not capture: {}",
                                removed.stderr_text().trim()
                            ),
                        )
                        .with_context("workspace_root", root))
                    }
                })
        };
        if let Err(error) = removed {
            let _ = self.restore_from_memory(root, &backup).await;
            return Err(error);
        }
        let result = operation().await;
        let restored = self.restore_from_memory(root, &backup).await;
        let value = result?;
        restored?;
        Ok(value)
    }

    /// Put an in-memory set-aside archive back into `root`. An empty archive has nothing to restore.
    async fn restore_from_memory(&self, root: &str, backup: &[u8]) -> SandboxResult<()> {
        if backup.is_empty() {
            return Ok(());
        }
        let staged = crate::sandbox::workspace_paths::provider_scratch_path(&format!(
            "modal-snapshot-filesystem-restore-{}.tar",
            self.session_id
        ));
        self.helper
            .call(json!({
                "op": "write_file",
                "sandbox_id": self.sandbox_id(),
                "path": staged,
                "data": BASE64.encode(backup),
            }))
            .await?;
        self.restore_excluded_paths(root, &staged).await
    }

    /// Archive `excluded` out of `root` into `backup`, then delete them from the workspace.
    async fn stash_excluded_paths(
        &self,
        root: &str,
        excluded: &[String],
        backup: &str,
    ) -> SandboxResult<()> {
        let quoted = excluded
            .iter()
            .map(|relative| crate::sandbox::session::shell_quote(relative))
            .collect::<Vec<_>>()
            .join(" ");
        let backup_quoted = crate::sandbox::session::shell_quote(backup);
        let root_quoted = crate::sandbox::session::shell_quote(root);
        // A path that does not exist is skipped rather than failing `tar`, and the archive is
        // written even when nothing matched so the restore is unconditional.
        let script = format!(
            "mkdir -p -- \"$(dirname -- {backup_quoted})\" && cd -- {root_quoted} && \
             {{ for rel in {quoted}; do if [ -e \"$rel\" ]; then printf '%s\\n' \"$rel\"; fi; \
             done; }} | tar cf {backup_quoted} -T - 2>/dev/null; \
             test -f {backup_quoted} && rm -rf -- {quoted}"
        );
        let result = self
            .helper_exec(&["sh".to_string(), "-lc".to_string(), script], "/", None)
            .await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "failed to set aside the paths a Modal snapshot must not capture: {}",
                    result.stderr_text().trim()
                ),
            )
            .with_context("workspace_root", root));
        }
        Ok(())
    }

    /// Put the stashed paths back and delete the stash.
    async fn restore_excluded_paths(&self, root: &str, backup: &str) -> SandboxResult<()> {
        let script = format!(
            "tar xf {backup} -C {root} && rm -f -- {backup}",
            backup = crate::sandbox::session::shell_quote(backup),
            root = crate::sandbox::session::shell_quote(root),
        );
        let result = self
            .helper_exec(&["sh".to_string(), "-lc".to_string(), script], "/", None)
            .await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "failed to restore the paths set aside for a Modal snapshot: {}",
                    result.stderr_text().trim()
                ),
            )
            .with_context("workspace_root", root));
        }
        Ok(())
    }

    async fn snapshot_directory_image(&self, root: &str) -> SandboxResult<String> {
        let response = self
            .helper
            .call(json!({
                "op": "snapshot_directory",
                "sandbox_id": self.sandbox_id(),
                "path": root,
                "timeout": self.options.snapshot_filesystem_timeout.as_secs(),
            }))
            .await?;
        response
            .get("image_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    "`snapshot_directory` returned no image id",
                )
            })
    }

    /// Reattach a directory snapshot over the workspace root with `Sandbox.mount_image`.
    ///
    /// Unlike a filesystem snapshot, this keeps the running sandbox, so the sandbox id and every
    /// open PTY survive the restore.
    async fn mount_snapshot_directory(&self, image_id: &str) -> SandboxResult<()> {
        self.helper
            .call(json!({
                "op": "mount_image",
                "sandbox_id": self.sandbox_id(),
                "path": self.manifest.root(),
                "image_id": image_id,
                "timeout": self.options.snapshot_filesystem_restore_timeout.as_secs(),
            }))
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    format!("failed to restore the Modal directory snapshot: {error}"),
                )
                .with_context("snapshot_id", image_id)
                .caused_by(&error)
            })?;
        // The image replaced the root's contents, so the durable mounts inside it are attached
        // again.
        for (mount, path) in self.durable_mounts_in_root()? {
            mount
                .strategy
                .restore_after_snapshot(mount, self, &path)
                .await
                .map_err(|error| {
                    SandboxError::new(
                        ErrorCode::SnapshotRestoreError,
                        "failed to attach a mount again after restoring the Modal directory \
                         snapshot",
                    )
                    .with_context("snapshot_id", image_id)
                    .with_context("mount_path", path.clone())
                    .caused_by(&error)
                })?;
        }
        Ok(())
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
        // A restore is bounded by its own timeout rather than by the one that bounds a fresh
        // `Sandbox.create`: the replacement boots from a snapshot image, which Modal pulls before
        // the sandbox is reachable.
        let mut options = self.options.clone();
        options.sandbox_create_timeout = Some(self.options.snapshot_filesystem_restore_timeout);
        let replacement = ModalSandboxClient {
            options,
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

    /// Check a workspace tar before any byte of it leaves the host.
    ///
    /// Beyond the generic hydrate policy, no member may overlap a path the manifest owns as
    /// ephemeral: the archive is extracted as validated, so there is no later point at which a
    /// member could be dropped.
    fn validate_tar_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        let mut options = crate::sandbox::archive::TarValidationOptions::hydrate();
        options.reject_rel_paths = self.manifest.ephemeral_persistence_paths();
        crate::sandbox::archive::validate_tar_bytes(archive, &options)
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

    fn persist_skip_registry(
        &self,
    ) -> Option<&std::sync::Mutex<std::collections::BTreeSet<String>>> {
        Some(&self.persist_skips)
    }

    fn system_state_preserved_on_start(&self) -> bool {
        self.state_preserved
    }

    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        // A replacement sandbox starts empty, so the restore must not be skipped for it.
        if !self.state_preserved {
            return None;
        }
        self.snapshot_fingerprint.lock().expect("poisoned").clone()
    }

    fn record_snapshot_fingerprint(&self, record: Option<SnapshotFingerprint>) {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
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
        let result: SandboxResult<()> = async {
            self.terminate_pty_sessions().await;
            let mut terminated = self.terminated.lock().await;
            self.started.store(false, Ordering::SeqCst);
            if *terminated {
                return Ok(());
            }
            // Python's `_shutdown_backend` swallows every provider error and forgets the sandbox
            // either way: teardown of a finished run must not fail it.
            let response = self
                .helper
                .call(json!({"op": "terminate", "sandbox_id": self.sandbox_id()}))
                .await;
            match response {
                Err(_) => log::debug!(target: crate::logger::LOG_TARGET, "failed to terminate the Modal sandbox on shutdown"),
                Ok(response) if response.get("ok").and_then(Value::as_bool) == Some(false) => {
                    log::debug!(target: crate::logger::LOG_TARGET, "the Modal helper could not terminate the sandbox on shutdown");
                }
                Ok(_) => {}
            }
            *terminated = true;
            Ok(())
        }
        .await;
        result.map_err(|error| {
            crate::sandbox::sandboxes::docker::redact_for_protected_mounts(&self.manifest, error)
        })
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
        // The in-sandbox resolver sees a symlink the lexical check cannot.
        let resolved = self.validate_remote_path_access(path, false, false).await?;
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

    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if max_bytes == 0 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "read_bounded max_bytes must be positive",
            ));
        }
        let resolved = self.validate_remote_path_access(path, false, false).await?;
        let response = self
            .helper
            .call(json!({
                "op": "read_bounded",
                "sandbox_id": self.sandbox_id(),
                "path": resolved,
                "max_bytes": max_bytes,
            }))
            .await?;
        decode_base64_field(&response, "data")
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.validate_remote_path_access(path, true, false).await?;
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

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.options.exposed_ports.clone()
    }

    /// Resolve an exposed port to a reachable endpoint, through Modal's encrypted tunnels.
    ///
    /// Mirrors `ModalSandboxSession._resolve_exposed_port`.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        assert_exposed_port_configured("modal", port, &self.options.exposed_ports)?;
        let unavailable = || {
            exposed_port_unavailable(
                "modal",
                port,
                &self.options.exposed_ports,
                ExposedPortUnavailable::BackendUnavailable,
            )
        };
        let response = self
            .helper
            .call(json!({
                "op": "tunnels",
                "sandbox_id": self.sandbox_id(),
                "timeout": 10,
            }))
            .await
            .map_err(|error| unavailable().caused_by(&error))?;
        let tunnel = response
            .get("tunnels")
            .and_then(Value::as_object)
            .and_then(|tunnels| tunnels.get(&port.to_string()))
            .ok_or_else(unavailable)?;
        let host = tunnel
            .get("host")
            .and_then(Value::as_str)
            .filter(|host| !host.is_empty())
            .ok_or_else(unavailable)?;
        let host_port = tunnel
            .get("port")
            .and_then(Value::as_u64)
            .and_then(|port| u16::try_from(port).ok())
            .ok_or_else(unavailable)?;
        // A Modal tunnel terminates TLS, which is why its port is 443 in practice.
        Ok(ExposedPortEndpoint::new(host, host_port, host_port == 443))
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
        let result: SandboxResult<Vec<u8>> = async {
            // A mount that cannot be detached in place would be missing from, or baked into, a native
            // image, so such a manifest persists as a tar archive instead, as Python's
            // `_native_snapshot_requires_tar_fallback` does.
            let native_snapshot_allowed = self
                .manifest
                .try_mount_targets()?
                .iter()
                .all(|(mount, _)| mount.strategy.supports_native_snapshot_detach());
            let persistence = if native_snapshot_allowed {
                self.options.workspace_persistence
            } else {
                ModalWorkspacePersistence::Tar
            };
            match persistence {
                ModalWorkspacePersistence::SnapshotFilesystem => {
                    let image_id = self.snapshot_filesystem().await?;
                    return Ok(encode_snapshot_filesystem_ref(&image_id));
                }
                ModalWorkspacePersistence::SnapshotDirectory => {
                    let image_id = self.snapshot_directory().await?;
                    return Ok(encode_snapshot_directory_ref(&image_id));
                }
                ModalWorkspacePersistence::Tar => {}
            }
            // The tar flow is the trait's own `tar -czf -` over `exec`, retried on a transient
            // transport failure exactly as Python's `@retry_async` wrapper does.
            retry_async(
                RetryPolicy::new(),
                || default_persist_workspace(self),
                |error, _attempt| is_transient_sandbox_error(error),
            )
            .await
        }
        .await;
        result.map_err(|error| {
            crate::sandbox::sandboxes::docker::redact_for_protected_mounts(&self.manifest, error)
        })
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        if decode_snapshot_filesystem_ref(archive).is_some()
            || decode_snapshot_directory_ref(archive).is_some()
        {
            return Ok(());
        }
        self.validate_tar_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        let result: SandboxResult<()> = async {
            if let Some(image_id) = decode_snapshot_filesystem_ref(archive) {
                return self.restore_from_image(&image_id).await;
            }
            if let Some(image_id) = decode_snapshot_directory_ref(archive) {
                return self.mount_snapshot_directory(&image_id).await;
            }
            self.validate_tar_archive(archive)?;
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
        .await;
        result.map_err(|error| {
            crate::sandbox::sandboxes::docker::redact_for_protected_mounts(&self.manifest, error)
        })
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
        state.with_snapshot_fingerprint(self.snapshot_fingerprint.lock().expect("poisoned").clone())
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
    for exclude in crate::sandbox::session::workspace_archive_excludes(session.manifest(), &root) {
        command.push_str(&format!(
            " --exclude {}",
            crate::sandbox::session::shell_quote(&exclude)
        ));
    }
    // Paths registered at runtime (generated mount config, sink output) are not durable state.
    for skipped in session.persist_workspace_skip_paths() {
        command.push_str(&format!(
            " --exclude {}",
            crate::sandbox::session::shell_quote(&format!("./{skipped}"))
        ));
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
    decode_snapshot_ref(payload, SNAPSHOT_FILESYSTEM_MAGIC)
}

/// The snapshot envelope a directory snapshot is stored as.
///
/// Byte-identical to Python's `_encode_snapshot_directory_ref`, so a reference written by either
/// port restores through the other.
pub fn encode_snapshot_directory_ref(image_id: &str) -> Vec<u8> {
    let body = json!({
        "snapshot_id": image_id,
        "workspace_persistence": "snapshot_directory",
    });
    let mut payload = SNAPSHOT_DIRECTORY_MAGIC.to_vec();
    payload.extend_from_slice(serde_json::to_string(&body).unwrap_or_default().as_bytes());
    payload
}

/// The Modal image id inside a directory-snapshot envelope, when the payload is one.
pub fn decode_snapshot_directory_ref(payload: &[u8]) -> Option<String> {
    decode_snapshot_ref(payload, SNAPSHOT_DIRECTORY_MAGIC)
}

fn decode_snapshot_ref(payload: &[u8], magic: &[u8]) -> Option<String> {
    let body = payload.strip_prefix(magic)?;
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
    fn serializes_a_cpu_and_memory_range_as_a_pair() {
        let client = ModalSandboxClient::new(
            ModalSandboxClientOptions::new("agents")
                .with_cpu_range(2.0, 4.0)
                .with_memory_mb_range(4096, 8192),
        );
        let request = client.create_request(&Manifest::new(), None);
        assert_eq!(request["cpu"], json!([2.0, 4.0]));
        assert_eq!(request["memory"], json!([4096, 8192]));
    }

    #[test]
    fn a_scalar_reservation_drops_an_earlier_limit() {
        // The two builders describe the same reservation, so setting one must not leave the
        // other's limit behind and turn a scalar back into a pair.
        let client = ModalSandboxClient::new(
            ModalSandboxClientOptions::new("agents")
                .with_cpu_range(2.0, 4.0)
                .with_memory_mb_range(4096, 8192)
                .with_cpu(1.0)
                .with_memory_mb(512),
        );
        let request = client.create_request(&Manifest::new(), None);
        assert_eq!(request["cpu"], 1.0);
        assert_eq!(request["memory"], 512);
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
