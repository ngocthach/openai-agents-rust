//! The Daytona sandbox backend (<https://daytona.io>). Port of
//! `agents.extensions.sandbox.daytona`.
//!
//! # Design
//!
//! Python wraps the `daytona` SDK. This port calls the two Daytona HTTP surfaces directly with
//! `reqwest`:
//!
//! - The **platform API** (`https://app.daytona.io/api` by default, authenticated with
//!   `Authorization: Bearer <DAYTONA_API_KEY>`) creates, starts, stops and deletes sandboxes.
//! - The **toolbox API** runs inside the sandbox and is reached through the per-sandbox
//!   `toolboxProxyUrl` the platform API returns, under `<proxy>/<sandbox id>/...`. It provides
//!   `process/execute`, the `process/session` family and the `files` endpoints.
//!
//! Like Python, [`exec`](DaytonaSandboxSession::exec) runs each command in a throwaway toolbox
//! session, because only the session API reports standard output and standard error separately;
//! the command itself is wrapped as `cd <root> && env -- K=V ... <command>`.
//! [`pty_exec_start`](DaytonaSandboxSession::pty_exec_start) keeps a session open, starts the
//! command asynchronously and streams its log endpoint, which is how Daytona supports an
//! interactive process without a websocket.
//!
//! ```no_run
//! use openai_agents_sdk::extensions::sandbox::daytona::{
//!     DaytonaSandboxClient, DaytonaSandboxClientOptions,
//! };
//! use openai_agents_sdk::sandbox::SandboxClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads `DAYTONA_API_KEY` from the environment, as the Python SDK does.
//! let client = DaytonaSandboxClient::with_options(DaytonaSandboxClientOptions::new());
//! let session = client.create(None).await?;
//! session.start().await?;
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Divergences from Python
//!
//! - Daytona's interactive endpoint (`process/pty/{id}/connect`) is a websocket, which this
//!   crate's `sandbox` feature does not pull in. The PTY methods use the session log and input
//!   endpoints instead: output streams and `write_stdin` works, but there is no terminal
//!   resize and no true TTY allocation.
//! - Daytona has no per-workspace snapshot API, so
//!   [`persist_workspace`](DaytonaSandboxSession::persist_workspace) always archives with `tar`,
//!   matching Python.
//! - Exposed-port preview URLs and sandbox metrics are not ported.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD;
use futures::StreamExt as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mount_entries::{BuiltinMountPattern, ConfiguredMount};
use crate::sandbox::mounts::{
    InContainerMountStrategy, Mount, MountCredentialAuthority, MountPatternConfig, MountStrategy,
    RCLONE_OPTION_READ_ONLY, RCLONE_OPTION_REMOTE_NAME, RCLONE_OPTION_REMOTE_PATH, RcloneMountMode,
    RcloneMountPattern,
};
use crate::sandbox::pty::{
    PtyExecRequest, PtyExecUpdate, PtyWriteRequest, RemotePtyManager, RemotePtyProcess,
    resolve_pty_write_yield_time_ms,
};
use crate::sandbox::retry::{RetryPolicy, is_transient_sandbox_error, retry_async};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot, shell_quote,
    workspace_archive_excludes,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, SnapshotFingerprint};
use crate::sandbox::types::{ExecResult, FileEntry, Permissions};
use crate::sandbox::workspace_paths::provider_scratch_path;

use super::e2b::adopt_recorded_root;
use super::e2b::rclone::{ensure_fuse_support, ensure_rclone};
use super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, assert_exposed_port_configured, endpoint_from_url,
    exposed_port_unavailable,
};
use crate::sandbox::http::{
    as_archive_read, bounded_read_zero, expect_success, parse_json, read_body_prefix,
    read_status_error, transport,
};

/// The backend identifier reported by this client and its sessions.
pub const BACKEND_ID: &str = "daytona";
/// Environment variable holding the Daytona API key, as in the Python SDK.
pub const DAYTONA_API_KEY_ENV: &str = "DAYTONA_API_KEY";
/// Environment variable overriding the Daytona API URL, as in the Python SDK.
pub const DAYTONA_API_URL_ENV: &str = "DAYTONA_API_URL";
/// Environment variable naming the Daytona organization, as in the Python SDK.
pub const DAYTONA_ORGANIZATION_ID_ENV: &str = "DAYTONA_ORGANIZATION_ID";
/// Environment variable naming the Daytona region, as in the Python SDK.
pub const DAYTONA_TARGET_ENV: &str = "DAYTONA_TARGET";
/// The default Daytona API URL.
pub const DEFAULT_DAYTONA_API_URL: &str = "https://app.daytona.io/api";
/// The workspace root Daytona sessions default to, matching Python.
pub const DEFAULT_DAYTONA_WORKSPACE_ROOT: &str = "/home/daytona/workspace";
/// Value of the `X-Daytona-Source` header this client sends.
pub const DAYTONA_SOURCE: &str = "openai-agents-sdk-rust";

/// Subprotocol that asks the daemon to report the PTY exit code in a control frame.
const PTY_EXIT_CONTROL_SUBPROTOCOL: &str = "X-Daytona-Pty-Exit-Control";
/// Prefix of the subprotocol token carrying the PTY environment.
///
/// The environment travels as a subprotocol rather than a query parameter or header so secrets
/// and large values stay out of URLs and access logs, exactly as the vendor SDK does it.
const PTY_ENVS_SUBPROTOCOL_PREFIX: &str = "X-Daytona-Pty-Envs~";
/// Terminal columns a new PTY session starts with, matching the vendor default.
const PTY_TERMINAL_COLS: u16 = 80;
/// Terminal rows a new PTY session starts with, matching the vendor default.
const PTY_TERMINAL_ROWS: u16 = 24;
/// How long a PTY websocket is given to acknowledge the connection.
const PTY_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How often the client pings an idle PTY websocket, matching the vendor SDK.
const PTY_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(20);

/// Resource limits for a Daytona sandbox. Mirrors Python's `DaytonaSandboxResources`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct DaytonaSandboxResources {
    /// CPU cores.
    pub cpu: Option<u32>,
    /// Memory in GB.
    pub memory: Option<u32>,
    /// Disk in GB.
    pub disk: Option<u32>,
}

impl DaytonaSandboxResources {
    /// Resources with nothing pinned.
    pub fn new() -> Self {
        DaytonaSandboxResources::default()
    }

    /// Request CPU cores.
    pub fn with_cpu(mut self, cpu: u32) -> Self {
        self.cpu = Some(cpu);
        self
    }

    /// Request memory, in GB.
    pub fn with_memory(mut self, memory: u32) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Request disk, in GB.
    pub fn with_disk(mut self, disk: u32) -> Self {
        self.disk = Some(disk);
        self
    }

    fn is_empty(&self) -> bool {
        self.cpu.is_none() && self.memory.is_none() && self.disk.is_none()
    }
}

/// Whether a failed Daytona file read is worth repeating, by HTTP status.
///
/// Mirrors Python's `_DAYTONA_HTTP_STATUS_RETRYABLE`: the statuses it names are classified, and
/// any other status is left unclassified (`None`) rather than guessed at.
fn daytona_status_retryable(status: u16) -> Option<bool> {
    match status {
        400 | 401 | 403 | 404 | 409 => Some(false),
        429 | 500 | 502 | 503 | 504 => Some(true),
        _ => None,
    }
}

/// Timeouts applied to Daytona operations. Mirrors Python's `DaytonaSandboxTimeouts`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DaytonaSandboxTimeouts {
    /// Sent to the toolbox when a command has no timeout of its own.
    pub exec_unbounded: Duration,
    /// Budget for liveness checks.
    pub keepalive: Duration,
    /// Budget for best-effort cleanup, such as deleting a toolbox session.
    pub cleanup: Duration,
    /// Budget for small filesystem operations.
    pub fast_op: Duration,
    /// Budget for file uploads.
    pub file_upload: Duration,
    /// Budget for file downloads.
    pub file_download: Duration,
    /// Budget for workspace archive operations.
    pub workspace_tar: Duration,
}

impl Default for DaytonaSandboxTimeouts {
    fn default() -> Self {
        DaytonaSandboxTimeouts {
            exec_unbounded: Duration::from_secs(24 * 60 * 60),
            keepalive: Duration::from_secs(10),
            cleanup: Duration::from_secs(30),
            fast_op: Duration::from_secs(30),
            file_upload: Duration::from_secs(1800),
            file_download: Duration::from_secs(1800),
            workspace_tar: Duration::from_secs(300),
        }
    }
}

/// Settings used when creating a Daytona-backed session.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct DaytonaSandboxClientOptions {
    /// API key for the platform API. Falls back to `DAYTONA_API_KEY`.
    pub api_key: Option<String>,
    /// Platform API base URL. Falls back to `DAYTONA_API_URL`, then
    /// [`DEFAULT_DAYTONA_API_URL`].
    pub api_url: Option<String>,
    /// Organization the sandbox belongs to. Falls back to `DAYTONA_ORGANIZATION_ID`.
    pub organization_id: Option<String>,
    /// Region the sandbox is created in. Falls back to `DAYTONA_TARGET`.
    pub target: Option<String>,
    /// Daytona snapshot (image template) the sandbox starts from.
    pub sandbox_snapshot_name: Option<String>,
    /// Container image the sandbox starts from, when no snapshot is named.
    pub image: Option<String>,
    /// Resource limits.
    pub resources: DaytonaSandboxResources,
    /// Environment variables set when the sandbox is created.
    pub env_vars: BTreeMap<String, String>,
    /// Labels attached to the sandbox.
    pub labels: BTreeMap<String, String>,
    /// Stop instead of deleting the sandbox when the session shuts down.
    pub pause_on_exit: bool,
    /// How long to wait for a created sandbox to report `started`.
    pub create_timeout: Duration,
    /// How long to wait for a restarted sandbox to report `started`.
    pub start_timeout: Duration,
    /// How often the client polls the sandbox state while waiting.
    pub state_poll_interval: Duration,
    /// Name of the sandbox; defaults to the session id.
    pub name: Option<String>,
    /// Auto-stop interval in minutes; `0` disables it.
    pub auto_stop_interval: u32,
    /// Operation timeouts.
    pub timeouts: DaytonaSandboxTimeouts,
    /// Ports the sandbox exposes.
    pub exposed_ports: Vec<u16>,
    /// How long a signed preview URL stays valid, in seconds.
    pub exposed_port_url_ttl_secs: u32,
}

impl Default for DaytonaSandboxClientOptions {
    fn default() -> Self {
        DaytonaSandboxClientOptions {
            api_key: None,
            api_url: None,
            organization_id: None,
            target: None,
            sandbox_snapshot_name: None,
            image: None,
            resources: DaytonaSandboxResources::default(),
            env_vars: BTreeMap::new(),
            labels: BTreeMap::new(),
            pause_on_exit: false,
            create_timeout: Duration::from_secs(60),
            start_timeout: Duration::from_secs(60),
            state_poll_interval: Duration::from_millis(500),
            name: None,
            auto_stop_interval: 0,
            timeouts: DaytonaSandboxTimeouts::default(),
            exposed_ports: Vec::new(),
            exposed_port_url_ttl_secs: 3600,
        }
    }
}

// The API key must never reach a log line through the `SandboxClient: Debug` bound.
impl std::fmt::Debug for DaytonaSandboxClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaytonaSandboxClientOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("api_url", &self.api_url)
            .field("organization_id", &self.organization_id)
            .field("target", &self.target)
            .field("sandbox_snapshot_name", &self.sandbox_snapshot_name)
            .field("image", &self.image)
            .field("resources", &self.resources)
            .field("env_vars", &self.env_vars.keys().collect::<Vec<_>>())
            .field("labels", &self.labels)
            .field("pause_on_exit", &self.pause_on_exit)
            .field("create_timeout", &self.create_timeout)
            .field("start_timeout", &self.start_timeout)
            .field("name", &self.name)
            .field("auto_stop_interval", &self.auto_stop_interval)
            .field("timeouts", &self.timeouts)
            .field("exposed_ports", &self.exposed_ports)
            .finish()
    }
}

impl DaytonaSandboxClientOptions {
    /// Options that read the API key and URL from the environment.
    pub fn new() -> Self {
        DaytonaSandboxClientOptions::default()
    }

    /// Use an explicit API key instead of `DAYTONA_API_KEY`.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use an explicit API URL instead of `DAYTONA_API_URL`.
    ///
    /// Tests use this to drive the client against a local HTTP server.
    pub fn with_api_url(mut self, api_url: impl Into<String>) -> Self {
        self.api_url = Some(api_url.into());
        self
    }

    /// Scope the sandbox to an organization.
    pub fn with_organization_id(mut self, organization_id: impl Into<String>) -> Self {
        self.organization_id = Some(organization_id.into());
        self
    }

    /// Create the sandbox in a region.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Start the sandbox from a Daytona snapshot.
    pub fn with_sandbox_snapshot_name(mut self, name: impl Into<String>) -> Self {
        self.sandbox_snapshot_name = Some(name.into());
        self
    }

    /// Start the sandbox from a container image.
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Pin the sandbox resources.
    pub fn with_resources(mut self, resources: DaytonaSandboxResources) -> Self {
        self.resources = resources;
        self
    }

    /// Set an environment variable at sandbox creation.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Attach a label to the sandbox.
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Stop instead of deleting the sandbox on shutdown.
    pub fn with_pause_on_exit(mut self, pause_on_exit: bool) -> Self {
        self.pause_on_exit = pause_on_exit;
        self
    }

    /// Name the sandbox explicitly.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the auto-stop interval in minutes.
    pub fn with_auto_stop_interval(mut self, minutes: u32) -> Self {
        self.auto_stop_interval = minutes;
        self
    }

    /// Replace the operation timeouts.
    pub fn with_timeouts(mut self, timeouts: DaytonaSandboxTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// How long to wait for a created sandbox to become ready.
    pub fn with_create_timeout(mut self, timeout: Duration) -> Self {
        self.create_timeout = timeout;
        self
    }

    /// How often the client polls the sandbox state while waiting.
    pub fn with_state_poll_interval(mut self, interval: Duration) -> Self {
        self.state_poll_interval = interval;
        self
    }

    /// Expose a port.
    pub fn with_exposed_port(mut self, port: u16) -> Self {
        self.exposed_ports.push(port);
        self
    }

    /// Set how long a signed preview URL stays valid.
    pub fn with_exposed_port_url_ttl_secs(mut self, ttl_secs: u32) -> Self {
        self.exposed_port_url_ttl_secs = ttl_secs;
        self
    }

    /// The resolved platform API base URL.
    fn api_base(&self) -> String {
        self.api_url
            .clone()
            .or_else(|| {
                std::env::var(DAYTONA_API_URL_ENV)
                    .ok()
                    .filter(|url| !url.is_empty())
            })
            .unwrap_or_else(|| DEFAULT_DAYTONA_API_URL.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    /// The API key, from the options or the environment.
    fn resolved_api_key(&self) -> SandboxResult<String> {
        // An explicitly empty key is a misconfiguration, not a request to read the environment.
        if let Some(api_key) = &self.api_key {
            return if api_key.is_empty() {
                Err(missing_api_key())
            } else {
                Ok(api_key.clone())
            };
        }
        std::env::var(DAYTONA_API_KEY_ENV)
            .ok()
            .filter(|key| !key.is_empty())
            .ok_or_else(missing_api_key)
    }

    fn resolved_organization_id(&self) -> Option<String> {
        self.organization_id.clone().or_else(|| {
            std::env::var(DAYTONA_ORGANIZATION_ID_ENV)
                .ok()
                .filter(|id| !id.is_empty())
        })
    }

    fn resolved_target(&self) -> Option<String> {
        self.target.clone().or_else(|| {
            std::env::var(DAYTONA_TARGET_ENV)
                .ok()
                .filter(|target| !target.is_empty())
        })
    }
}

/// The error raised when no usable Daytona API key is configured.
fn missing_api_key() -> SandboxError {
    SandboxError::new(
        ErrorCode::SandboxConfigInvalid,
        format!(
            "the Daytona sandbox backend needs an API key: set `{DAYTONA_API_KEY_ENV}` or \
             `DaytonaSandboxClientOptions::with_api_key`"
        ),
    )
}

/// A sandbox as the Daytona platform API reports it.
#[derive(Debug, Clone, Deserialize)]
struct DaytonaSandbox {
    id: String,
    #[serde(default)]
    state: Option<String>,
    #[serde(rename = "toolboxProxyUrl", default)]
    toolbox_proxy_url: Option<String>,
}

/// Creates [`DaytonaSandboxSession`]s against the Daytona HTTP API.
#[derive(Debug, Clone)]
pub struct DaytonaSandboxClient {
    options: DaytonaSandboxClientOptions,
    http: reqwest::Client,
    snapshot: Option<SnapshotSpec>,
}

impl Default for DaytonaSandboxClient {
    fn default() -> Self {
        DaytonaSandboxClient::with_options(DaytonaSandboxClientOptions::default())
    }
}

impl DaytonaSandboxClient {
    /// A client reading its credentials from the environment.
    pub fn new() -> Self {
        DaytonaSandboxClient::default()
    }

    /// A client using explicit options.
    pub fn with_options(options: DaytonaSandboxClientOptions) -> Self {
        DaytonaSandboxClient {
            options,
            http: reqwest::Client::new(),
            snapshot: None,
        }
    }

    /// Reuse an existing HTTP client.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// The options this client was built with.
    pub fn options(&self) -> &DaytonaSandboxClientOptions {
        &self.options
    }

    /// Attach the platform authentication and organization scope to a request.
    fn authorize(
        &self,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        let mut builder = builder
            .bearer_auth(api_key)
            .header("X-Daytona-Source", DAYTONA_SOURCE);
        if let Some(organization_id) = self.options.resolved_organization_id() {
            builder = builder.header("X-Daytona-Organization-ID", organization_id);
        }
        builder
    }

    async fn create_sandbox(
        &self,
        api_key: &str,
        name: &str,
        envs: &BTreeMap<String, String>,
    ) -> SandboxResult<DaytonaSandbox> {
        let mut body = json!({
            "name": name,
            "autoStopInterval": self.options.auto_stop_interval,
        });
        let map = body.as_object_mut().expect("object");
        if let Some(snapshot) = &self.options.sandbox_snapshot_name {
            map.insert("snapshot".into(), json!(snapshot));
        } else if let Some(image) = &self.options.image {
            // Daytona builds an image-based sandbox from a one-line Dockerfile, the way the
            // vendor SDK's `CreateSandboxFromImageParams` does.
            map.insert(
                "buildInfo".into(),
                json!({ "dockerfileContent": format!("FROM {image}") }),
            );
        }
        if !envs.is_empty() {
            map.insert("env".into(), json!(envs));
        }
        if !self.options.labels.is_empty() {
            map.insert("labels".into(), json!(self.options.labels));
        }
        if let Some(target) = self.options.resolved_target() {
            map.insert("target".into(), json!(target));
        }
        if !self.options.resources.is_empty() {
            if let Some(cpu) = self.options.resources.cpu {
                map.insert("cpu".into(), json!(cpu));
            }
            if let Some(memory) = self.options.resources.memory {
                map.insert("memory".into(), json!(memory));
            }
            if let Some(disk) = self.options.resources.disk {
                map.insert("disk".into(), json!(disk));
            }
        }
        let response = self
            .authorize(
                self.http
                    .post(format!("{}/sandbox", self.options.api_base())),
                api_key,
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| transport("create a Daytona sandbox", e))?;
        parse_json(response, "create a Daytona sandbox").await
    }
}

#[async_trait]
impl SandboxClient for DaytonaSandboxClient {
    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        // Daytona images place the agent's home under `/home/daytona`, so a caller who does not
        // supply a manifest gets that root rather than the SDK-wide `/workspace` default.
        let manifest =
            manifest.unwrap_or_else(|| Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT));
        let api_key = self.options.resolved_api_key()?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let name = self
            .options
            .name
            .clone()
            .unwrap_or_else(|| session_id.clone());
        let mut envs = self.options.env_vars.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        let sandbox = self.create_sandbox(&api_key, &name, &envs).await?;
        let session = DaytonaSandboxSession::new(
            session_id,
            self.options.clone(),
            api_key,
            self.http.clone(),
            sandbox,
            manifest,
            self.snapshot.clone(),
        );
        session
            .wait_until_started(self.options.create_timeout)
            .await?;
        Ok(Arc::new(session))
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        if state.backend != BACKEND_ID {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "sandbox session state was created by the `{}` backend",
                    state.backend
                ),
            ));
        }
        let manifest = adopt_recorded_root(manifest, state);
        state.verify_manifest(&manifest)?;
        let sandbox_id = state.provider_id.clone().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "Daytona sandbox session state is missing its sandbox id",
            )
        })?;
        let api_key = self.options.resolved_api_key()?;
        let mut envs = self.options.env_vars.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }

        let existing = self
            .authorize(
                self.http
                    .get(format!("{}/sandbox/{sandbox_id}", self.options.api_base())),
                &api_key,
            )
            .send()
            .await
            .ok()
            .filter(|response| response.status().is_success());
        let sandbox = match existing {
            Some(response) => parse_json::<DaytonaSandbox>(response, "load a Daytona sandbox")
                .await
                .ok(),
            None => None,
        };
        let reattached = sandbox.is_some();
        let sandbox = match sandbox {
            Some(sandbox) => sandbox,
            // The original sandbox is gone: create a replacement and hydrate it from the snapshot.
            None => {
                self.create_sandbox(&api_key, &state.session_id, &envs)
                    .await?
            }
        };
        let session = DaytonaSandboxSession::new(
            state.session_id.clone(),
            self.options.clone(),
            api_key,
            self.http.clone(),
            sandbox,
            manifest,
            Some(state.snapshot.clone()),
        )
        .with_snapshot_id(state.snapshot_id.clone())
        .with_state_preserved(reattached)
        .with_snapshot_fingerprint(if reattached {
            state
                .snapshot_fingerprint
                .clone()
                .zip(state.snapshot_fingerprint_version.clone())
                .map(|(fingerprint, version)| SnapshotFingerprint {
                    fingerprint,
                    version,
                })
        } else {
            None
        })
        .resuming();
        // A run that ended with `stop` left the sandbox stopped; it has to be started again
        // before the toolbox answers.
        session.start_sandbox_if_stopped().await?;
        session
            .wait_until_started(self.options.start_timeout)
            .await?;
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// A sandbox session whose workspace lives in a Daytona sandbox.
pub struct DaytonaSandboxSession {
    session_id: String,
    manifest: Manifest,
    options: DaytonaSandboxClientOptions,
    api_key: String,
    api_base: String,
    http: reqwest::Client,
    sandbox_id: String,
    toolbox_base: RwLock<String>,
    started: AtomicBool,
    stopped: AtomicBool,
    released: tokio::sync::Mutex<bool>,
    ptys: RemotePtyManager,
    /// The provider half of each live PTY session.
    pty_handles: Mutex<HashMap<u32, Arc<DaytonaPtyHandle>>>,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
    /// Workspace-relative paths a persist leaves out of the archive.
    persist_skips: Mutex<std::collections::BTreeSet<String>>,
    /// The sandbox survived the pause this session resumes from.
    state_preserved: bool,
    /// The workspace fingerprint the stored snapshot was taken from.
    snapshot_fingerprint: Mutex<Option<SnapshotFingerprint>>,
}

// The API key must never appear in a log line.
impl std::fmt::Debug for DaytonaSandboxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaytonaSandboxSession")
            .field("session_id", &self.session_id)
            .field("sandbox_id", &self.sandbox_id)
            .field("workspace_root", &self.manifest.root())
            .field("resuming", &self.resuming)
            .finish_non_exhaustive()
    }
}

impl DaytonaSandboxSession {
    fn new(
        session_id: String,
        options: DaytonaSandboxClientOptions,
        api_key: String,
        http: reqwest::Client,
        sandbox: DaytonaSandbox,
        manifest: Manifest,
        snapshot: Option<SnapshotSpec>,
    ) -> Self {
        let api_base = options.api_base();
        let toolbox_base = toolbox_base_url(&api_base, &sandbox);
        DaytonaSandboxSession {
            manifest,
            options,
            api_key,
            api_base,
            http,
            sandbox_id: sandbox.id,
            toolbox_base: RwLock::new(toolbox_base),
            snapshot_id: session_id.clone(),
            session_id,
            started: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            released: tokio::sync::Mutex::new(false),
            ptys: RemotePtyManager::new(),
            pty_handles: Mutex::new(HashMap::new()),
            snapshot,
            resuming: false,
            persist_skips: Mutex::new(std::collections::BTreeSet::new()),
            state_preserved: false,
            snapshot_fingerprint: Mutex::new(None),
        }
    }

    /// Carry the workspace fingerprint a resumed snapshot was taken from.
    fn with_snapshot_fingerprint(self, record: Option<SnapshotFingerprint>) -> Self {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
        self
    }

    /// Record that the sandbox survived the pause this session resumes from.
    fn with_state_preserved(mut self, preserved: bool) -> Self {
        self.state_preserved = preserved;
        self
    }

    /// Reuse a persisted snapshot identity.
    fn with_snapshot_id(mut self, snapshot_id: String) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Restore the snapshot on start instead of materializing the full manifest.
    fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// The Daytona sandbox backing this session.
    pub fn sandbox_id(&self) -> &str {
        &self.sandbox_id
    }

    /// Attach the platform authentication and organization scope to a request.
    fn authorize(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder
            .bearer_auth(&self.api_key)
            .header("X-Daytona-Source", DAYTONA_SOURCE);
        if let Some(organization_id) = self.options.resolved_organization_id() {
            builder = builder.header("X-Daytona-Organization-ID", organization_id);
        }
        builder
    }

    /// The base URL of this sandbox's toolbox API.
    fn toolbox(&self, path: &str) -> String {
        format!("{}{path}", self.toolbox_base.read().expect("poisoned"))
    }

    /// Read the sandbox state, refreshing the toolbox base URL.
    async fn fetch_sandbox(&self) -> SandboxResult<DaytonaSandbox> {
        let response = self
            .authorize(
                self.http
                    .get(format!("{}/sandbox/{}", self.api_base, self.sandbox_id)),
            )
            .send()
            .await
            .map_err(|e| transport("load the Daytona sandbox", e))?;
        let sandbox: DaytonaSandbox = parse_json(response, "load the Daytona sandbox").await?;
        if sandbox.toolbox_proxy_url.is_some() {
            *self.toolbox_base.write().expect("poisoned") =
                toolbox_base_url(&self.api_base, &sandbox);
        }
        Ok(sandbox)
    }

    /// Start the sandbox when a previous run left it stopped or archived.
    async fn start_sandbox_if_stopped(&self) -> SandboxResult<()> {
        let sandbox = self.fetch_sandbox().await?;
        if !matches!(
            sandbox.state.as_deref(),
            Some("stopped") | Some("archived") | Some("stopping")
        ) {
            return Ok(());
        }
        let response = self
            .authorize(self.http.post(format!(
                "{}/sandbox/{}/start",
                self.api_base, self.sandbox_id
            )))
            .send()
            .await
            .map_err(|e| transport("start the Daytona sandbox", e))?;
        expect_success(
            response,
            "start the Daytona sandbox",
            ErrorCode::WorkspaceStartError,
        )
        .await
    }

    /// Poll until the sandbox reports `started`, or the budget elapses.
    async fn wait_until_started(&self, budget: Duration) -> SandboxResult<()> {
        let deadline = Instant::now() + budget;
        loop {
            let sandbox = self.fetch_sandbox().await?;
            match sandbox.state.as_deref() {
                Some("started") => return Ok(()),
                Some("error") | Some("build_failed") => {
                    return Err(SandboxError::new(
                        ErrorCode::WorkspaceStartError,
                        format!(
                            "Daytona sandbox `{}` entered state `{}`",
                            self.sandbox_id,
                            sandbox.state.unwrap_or_default()
                        ),
                    ));
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!(
                        "Daytona sandbox `{}` did not become ready within {budget:?}",
                        self.sandbox_id
                    ),
                ));
            }
            tokio::time::sleep(self.options.state_poll_interval).await;
        }
    }

    /// The environment one command runs with: the session's, then the request's own.
    ///
    /// The session environment is the client options' plus the manifest's, which is what Python
    /// passes per command; `ExecRequest::env` then overrides individual names.
    fn request_envs(&self, request: &ExecRequest) -> BTreeMap<String, String> {
        let mut envs = self.command_envs();
        envs.extend(request.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        envs
    }

    /// The environment every command runs with.
    fn command_envs(&self) -> BTreeMap<String, String> {
        let mut envs = self.options.env_vars.clone();
        for (key, value) in self.manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        envs
    }

    /// Wrap `argv` as a shell command that enters `cwd` and exports `envs`.
    ///
    /// Mirrors Python's `DaytonaSandboxSession._exec_internal`: the toolbox API takes one command
    /// string, with no separate working directory or environment, so both are expressed in the
    /// command itself.
    fn session_command(
        &self,
        argv: &[String],
        cwd: &str,
        envs: &BTreeMap<String, String>,
    ) -> String {
        let joined = argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "cd {} && {}{joined}",
            shell_quote(cwd),
            env_command_prefix(envs)
        )
    }

    /// Resolve the working directory of a request, defaulting to the workspace root.
    fn resolve_cwd(&self, cwd: &Option<String>) -> SandboxResult<String> {
        match cwd {
            None => Ok(self.manifest.root().to_string()),
            Some(cwd) => self.normalize_path(cwd),
        }
    }

    /// Create a toolbox session, which is what gives separate stdout and stderr.
    async fn create_toolbox_session(&self, session_id: &str) -> SandboxResult<()> {
        let response = self
            .authorize(self.http.post(self.toolbox("/process/session")))
            .json(&json!({ "sessionId": session_id }))
            .send()
            .await
            .map_err(|e| transport("create a Daytona toolbox session", e))?;
        expect_success(
            response,
            "create a Daytona toolbox session",
            ErrorCode::ExecTransportError,
        )
        .await
    }

    /// Delete a toolbox session, best effort.
    async fn delete_toolbox_session(&self, session_id: &str) {
        let request = self
            .authorize(
                self.http
                    .delete(self.toolbox(&format!("/process/session/{session_id}"))),
            )
            .send();
        let _ = tokio::time::timeout(self.options.timeouts.cleanup, request).await;
    }

    /// Run one command in a throwaway toolbox session.
    async fn exec_in_session(
        &self,
        argv: Vec<String>,
        cwd: String,
        envs: &BTreeMap<String, String>,
        timeout: Option<Duration>,
    ) -> SandboxResult<ExecResult> {
        let what = "run a command in the Daytona sandbox";
        let toolbox_session = format!("sandbox-{}", uuid::Uuid::new_v4().simple());
        let command = self.session_command(&argv, &cwd, envs);
        let budget = timeout.unwrap_or(self.options.timeouts.exec_unbounded);
        let run = async {
            self.create_toolbox_session(&toolbox_session).await?;
            let response = self
                .authorize(
                    self.http
                        .post(self.toolbox(&format!("/process/session/{toolbox_session}/exec"))),
                )
                .json(&json!({ "command": command, "runAsync": false }))
                .send()
                .await
                .map_err(|e| transport(what, e))?;
            let result: SessionExecuteResponse = parse_json(response, what).await?;
            Ok(result.into_exec_result())
        };
        let result = match tokio::time::timeout(budget, run).await {
            Ok(result) => result,
            Err(_) => Err(SandboxError::new(
                ErrorCode::ExecTimeout,
                format!("Command `{}` timed out after {budget:?}", argv.join(" ")),
            )),
        };
        self.delete_toolbox_session(&toolbox_session).await;
        result
    }

    /// Download a file by its absolute sandbox path.
    ///
    /// Workspace reads go through [`read_file`](SandboxSession::read_file), which validates the
    /// path first; this is for the paths the session itself owns, such as a temporary archive
    /// under `/tmp`.
    async fn download_at(&self, path: &str, limit: Option<usize>) -> SandboxResult<Vec<u8>> {
        let what = "read a file from the Daytona sandbox";
        let response = self
            .authorize(self.http.get(self.toolbox("/files/download")))
            .query(&[("path", path)])
            .timeout(self.options.timeouts.file_download)
            .send()
            .await
            .map_err(|e| as_archive_read(transport(what, e)))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("`{path}` does not exist in the sandbox"),
            ));
        }
        if !response.status().is_success() {
            let status = response.status();
            return Err(read_status_error(
                what,
                status,
                daytona_status_retryable(status.as_u16()),
            ));
        }
        read_body_prefix(response, limit, what).await
    }

    /// Upload a file by its absolute sandbox path.
    async fn upload_at(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let what = "write a file to the Daytona sandbox";
        let name = path.rsplit('/').next().unwrap_or("file").to_string();
        let (content_type, body) = multipart_file("file", &name, contents);
        let response = self
            .authorize(self.http.post(self.toolbox("/files/upload-v2")))
            .query(&[("path", path)])
            .header("Content-Type", content_type)
            .timeout(self.options.timeouts.file_upload)
            .body(body)
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    /// Remove a path by its absolute sandbox path.
    async fn remove_at(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let what = "remove a path in the Daytona sandbox";
        let response = self
            .authorize(self.http.delete(self.toolbox("/files")))
            .query(&[
                ("path", path),
                ("recursive", if recursive { "true" } else { "false" }),
            ])
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    /// Stop the sandbox without deleting it.
    async fn stop_sandbox(&self) -> SandboxResult<()> {
        let response = self
            .authorize(self.http.post(format!(
                "{}/sandbox/{}/stop",
                self.api_base, self.sandbox_id
            )))
            .send()
            .await
            .map_err(|e| transport("stop the Daytona sandbox", e))?;
        expect_success(
            response,
            "stop the Daytona sandbox",
            ErrorCode::WorkspaceStopError,
        )
        .await
    }

    /// Delete the sandbox and release its resources.
    async fn delete_sandbox(&self) -> SandboxResult<()> {
        let response = self
            .authorize(
                self.http
                    .delete(format!("{}/sandbox/{}", self.api_base, self.sandbox_id)),
            )
            .send()
            .await
            .map_err(|e| transport("delete the Daytona sandbox", e))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(
            response,
            "delete the Daytona sandbox",
            ErrorCode::WorkspaceStopError,
        )
        .await
    }

    /// Release the sandbox exactly once, stopping it when the options ask for that.
    async fn terminate(&self) -> SandboxResult<()> {
        let mut terminated = self.released.lock().await;
        if *terminated {
            return Ok(());
        }
        self.pty_terminate_all().await.ok();
        let result = if self.options.pause_on_exit {
            // A `stop` already stopped it; stopping again could fail and the fallback would
            // delete a sandbox the caller asked to keep.
            if self.stopped.load(Ordering::SeqCst) {
                Ok(())
            } else {
                match self.stop_sandbox().await {
                    Ok(()) => Ok(()),
                    // A stop that cannot be taken must not leak the sandbox.
                    Err(error) => {
                        log::warn!(target: crate::logger::LOG_TARGET,
                            "failed to stop the Daytona sandbox, deleting it instead: {error}"
                        );
                        self.delete_sandbox().await
                    }
                }
            }
        } else {
            self.delete_sandbox().await
        };
        self.started.store(false, Ordering::SeqCst);
        *terminated = true;
        // Python's `_shutdown_backend` swallows every provider error: teardown of a finished run
        // must not fail it, and the sandbox auto-stops on its own timeout anyway.
        if let Err(error) = result {
            log::warn!(target: crate::logger::LOG_TARGET, "failed to release the Daytona sandbox on shutdown: {error}");
        }
        Ok(())
    }
}

/// A session command result as the toolbox API reports it.
#[derive(Debug, Clone, Deserialize)]
struct SessionExecuteResponse {
    #[serde(rename = "exitCode", default)]
    exit_code: Option<i32>,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
}

impl SessionExecuteResponse {
    /// Split the response into stdout and stderr, as the Python SDK does.
    fn into_exec_result(self) -> ExecResult {
        let exit_code = self.exit_code.unwrap_or(0);
        let (stdout, stderr) = match (self.stdout, self.stderr) {
            (None, None) => {
                // Older toolbox builds report one combined stream; attribute it by exit code.
                let output = self.output.unwrap_or_default();
                if exit_code == 0 {
                    (output, String::new())
                } else {
                    (String::new(), output)
                }
            }
            (stdout, stderr) => (stdout.unwrap_or_default(), stderr.unwrap_or_default()),
        };
        ExecResult {
            stdout: stdout.into_bytes(),
            stderr: stderr.into_bytes(),
            exit_code,
        }
    }
}

/// The toolbox base URL of `sandbox`, falling back to the platform API's toolbox proxy route.
fn toolbox_base_url(api_base: &str, sandbox: &DaytonaSandbox) -> String {
    match &sandbox.toolbox_proxy_url {
        Some(url) if !url.is_empty() => {
            format!("{}/{}", url.trim_end_matches('/'), sandbox.id)
        }
        _ => format!("{api_base}/toolbox/{}/toolbox", sandbox.id),
    }
}

#[async_trait]
impl SandboxSession for DaytonaSandboxSession {
    fn id(&self) -> &str {
        &self.session_id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    async fn start(&self) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        // The toolbox uses the workspace root as the working directory of every command, so it
        // has to exist before the manifest is materialized.
        let prepared = self
            .exec_in_session(
                vec!["mkdir".into(), "-p".into(), "--".into(), root.clone()],
                "/".to_string(),
                &self.command_envs(),
                Some(self.options.timeouts.fast_op),
            )
            .await?;
        if !prepared.ok() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "Failed to create the Daytona workspace root `{root}`: {}",
                    prepared.stderr_text().trim()
                ),
            ));
        }
        self.started.store(true, Ordering::SeqCst);
        let result = if self.resuming {
            match restore_session_snapshot(self).await {
                Ok(false) => self
                    .manifest
                    .clone()
                    .apply_ephemeral(self)
                    .await
                    .map(|_| ()),
                Ok(true) => Ok(()),
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
        // `stop` keeps what `shutdown` releases: a stopped Daytona sandbox keeps its filesystem
        // and can be started again by id.
        if *self.released.lock().await {
            return Ok(());
        }
        self.pty_terminate_all().await.ok();
        let result = self.stop_sandbox().await;
        self.started.store(false, Ordering::SeqCst);
        if result.is_ok() {
            self.stopped.store(true, Ordering::SeqCst);
        }
        result
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.terminate().await
    }

    async fn running(&self) -> SandboxResult<bool> {
        let Ok(sandbox) =
            tokio::time::timeout(self.options.timeouts.keepalive, self.fetch_sandbox()).await
        else {
            return Ok(false);
        };
        Ok(sandbox
            .map(|s| s.state.as_deref() == Some("started"))
            .unwrap_or(false))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let cwd = self.resolve_cwd(&request.cwd)?;
        // Daytona has no user parameter, so the `sudo -u` wrapper stays in the command.
        let envs = self.request_envs(&request);
        self.exec_in_session(request.resolved_argv(), cwd, &envs, request.timeout)
            .await
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        // The sandbox's own filesystem decides: a workspace symlink can lead out of the root.
        let resolved = self.validate_remote_path_access(path, false, false).await?;
        self.download_at(&resolved, None).await
    }

    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if max_bytes == 0 {
            return Err(bounded_read_zero());
        }
        let resolved = self.validate_remote_path_access(path, false, false).await?;
        self.download_at(&resolved, Some(max_bytes)).await
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.validate_remote_path_access(path, true, false).await?;
        self.upload_at(&resolved, contents).await
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.options.exposed_ports.clone()
    }

    /// Resolve an exposed port to a signed preview endpoint.
    ///
    /// Mirrors `DaytonaSandboxSession._resolve_exposed_port`, whose `create_signed_preview_url`
    /// is `GET /sandbox/{id}/ports/{port}/signed-preview-url?expiresInSeconds=…`. The signature
    /// is part of the returned URL, so the endpoint's query carries it.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        assert_exposed_port_configured(BACKEND_ID, port, &self.options.exposed_ports)?;
        let unavailable = || {
            exposed_port_unavailable(
                BACKEND_ID,
                port,
                &self.options.exposed_ports,
                ExposedPortUnavailable::BackendUnavailable,
            )
        };
        let response = self
            .authorize(
                self.http
                    .get(format!(
                        "{}/sandbox/{}/ports/{port}/signed-preview-url",
                        self.api_base, self.sandbox_id
                    ))
                    .query(&[(
                        "expiresInSeconds",
                        self.options.exposed_port_url_ttl_secs.to_string(),
                    )]),
            )
            .send()
            .await
            .map_err(|error| {
                unavailable().caused_by(&transport("resolve a Daytona preview URL", error))
            })?;
        let body: Value = parse_json(response, "resolve a Daytona preview URL")
            .await
            .map_err(|error| unavailable().caused_by(&error))?;
        let url = body
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
            .ok_or_else(unavailable)?;
        let mut endpoint = endpoint_from_url(url).ok_or_else(unavailable)?;
        if let Some((_, query)) = url.split_once('?') {
            endpoint = endpoint.with_query(query);
        }
        Ok(endpoint)
    }

    fn supports_pty(&self) -> bool {
        true
    }

    /// The sandbox's own filesystem decides every path, so `rm`, `ls` and `mv` resolve it first.
    fn resolves_paths_remotely(&self) -> bool {
        true
    }

    fn persist_skip_registry(&self) -> Option<&Mutex<std::collections::BTreeSet<String>>> {
        Some(&self.persist_skips)
    }

    fn system_state_preserved_on_start(&self) -> bool {
        self.state_preserved
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        let resolved = self.validate_remote_path_access(path, false, false).await?;
        let what = "list a directory in the Daytona sandbox";
        let response = self
            .authorize(self.http.get(self.toolbox("/files")))
            .query(&[("path", resolved.as_str())])
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        let entries: Vec<Value> = parse_json(response, what).await?;
        Ok(entries
            .iter()
            .map(|entry| file_info_to_file_entry(entry, &resolved))
            .collect())
    }

    async fn mkdir_as(
        &self,
        path: &str,
        parents: bool,
        user: Option<&crate::sandbox::User>,
    ) -> SandboxResult<()> {
        let Some(user) = user else {
            return self.mkdir(path, parents).await;
        };
        // The toolbox files API runs as the sandbox's own user, so the requested user's
        // permission is checked first and the folder is then created through the API.
        let resolved = self.validate_remote_path_access(path, true, false).await?;
        self.check_mkdir_access(&resolved, parents, Some(user))
            .await?;
        self.mkdir(path, parents).await
    }

    async fn mkdir(&self, path: &str, _parents: bool) -> SandboxResult<()> {
        // The toolbox always creates the missing parents.
        let resolved = self.validate_remote_path_access(path, true, false).await?;
        let what = "create a directory in the Daytona sandbox";
        let response = self
            .authorize(self.http.post(self.toolbox("/files/folder")))
            .query(&[("path", resolved.as_str()), ("mode", "755")])
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        // The session contract is `mkdir -p`, and the manifest applier creates the workspace
        // root again after `start` already made it, so an existing directory is not a failure.
        if response.status() == reqwest::StatusCode::CONFLICT {
            return Ok(());
        }
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        if recursive {
            self.workspace_path_policy()?
                .validate_recursive_remove(path)?;
        }
        let resolved = self
            .validate_remote_path_access(path, true, recursive)
            .await?;
        self.remove_at(&resolved, recursive).await
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let source = self
            .validate_remote_path_access(source, true, false)
            .await?;
        let destination = self
            .validate_remote_path_access(destination, true, false)
            .await?;
        let what = "move a path in the Daytona sandbox";
        let response = self
            .authorize(self.http.post(self.toolbox("/files/move")))
            .query(&[
                ("source", source.as_str()),
                ("destination", destination.as_str()),
            ])
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let cwd = self.resolve_cwd(&request.cwd)?;
        let argv = request.resolved_argv();
        // Python only opens the PTY websocket when the model asked for a writable terminal; a
        // read-only interactive process goes through the session log stream instead.
        if request.tty {
            self.start_terminal(&argv, &cwd, &request).await
        } else {
            self.start_logged_command(&argv, &cwd, &request).await
        }
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let process = self.ptys.get(request.session_id)?;
        if !request.chars.is_empty() {
            let handle = self.pty_handle(request.session_id).ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::PtySessionNotFound,
                    format!("PTY session `{}` is not running", request.session_id),
                )
            })?;
            match handle.as_ref() {
                DaytonaPtyHandle::Terminal { input, .. } => {
                    input.send(request.chars.as_bytes().to_vec()).map_err(|_| {
                        SandboxError::new(
                            ErrorCode::PtySessionNotFound,
                            format!(
                                "PTY session `{}` is no longer connected",
                                request.session_id
                            ),
                        )
                    })?;
                    // The daemon echoes through the PTY; give it a moment, as Python does.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                DaytonaPtyHandle::Logged {
                    toolbox_session,
                    command_id,
                } => {
                    // Python raises for a process started without a terminal; only the websocket
                    // path has a real stdin.
                    if !process.tty() {
                        return Err(SandboxError::new(
                            ErrorCode::PtyStdinUnavailable,
                            format!(
                                "PTY session `{}` was not started with a TTY, so it has no stdin",
                                request.session_id
                            ),
                        ));
                    }
                    let what = "write to an interactive process in the Daytona sandbox";
                    let response = self
                        .authorize(self.http.post(self.toolbox(&format!(
                            "/process/session/{toolbox_session}/command/{command_id}/input"
                        ))))
                        .json(&json!({ "data": request.chars }))
                        .send()
                        .await
                        .map_err(|e| transport(what, e))?;
                    expect_success(response, what, ErrorCode::ExecTransportError).await?;
                }
            }
        }
        let yield_time_ms =
            resolve_pty_write_yield_time_ms(request.yield_time_ms, request.chars.is_empty());
        let update = process
            .collect(yield_time_ms, request.max_output_tokens)
            .await;
        self.forget_finished_pty(&update, process.id()).await;
        Ok(update)
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        for process in self.ptys.take_all() {
            process.close(None);
            self.release_pty(process.id()).await;
        }
        Ok(())
    }

    fn session_state(&self) -> SandboxSessionState {
        SandboxSessionState::new(
            BACKEND_ID,
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.sandbox_id.clone()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
        .with_snapshot_fingerprint(self.snapshot_fingerprint.lock().expect("poisoned").clone())
    }

    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        // Only a sandbox that survived the pause can still hold the workspace the fingerprint
        // describes; a replacement starts empty, so its restore must not be skipped.
        if !self.state_preserved {
            return None;
        }
        self.snapshot_fingerprint.lock().expect("poisoned").clone()
    }

    fn record_snapshot_fingerprint(&self, record: Option<SnapshotFingerprint>) {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        let root = self.manifest.root().to_string();
        let archive_path = self.resolve_internal_path(&provider_scratch_path(&format!(
            "{}.tar.gz",
            self.session_id
        )))?;
        let mut command = format!(
            "tar -czf {archive} -C {root}",
            archive = shell_quote(&archive_path),
            root = shell_quote(&root),
        );
        for exclude in workspace_archive_excludes(&self.manifest, &root) {
            command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
        }
        for skipped in self.persist_workspace_skip_paths() {
            command.push_str(&format!(
                " --exclude {}",
                shell_quote(&format!("./{skipped}"))
            ));
        }
        command.push_str(" .");
        // The toolbox reports command output as text, so the archive travels over the file API.
        // Python retries just this command on a transient transport failure
        // (`daytona/sandbox.py` L1033), not the download that follows it.
        let envs = self.command_envs();
        let result = retry_async(
            RetryPolicy::new(),
            || {
                self.exec_in_session(
                    vec!["sh".into(), "-lc".into(), command.clone()],
                    root.clone(),
                    &envs,
                    Some(self.options.timeouts.workspace_tar),
                )
            },
            |error, _attempt| is_transient_sandbox_error(error),
        )
        .await
        .map_err(|error| {
            // A stopped sandbox has no container address the toolbox can reach, and starting it
            // again is not something a retry does.
            if error.message.contains("Is the Sandbox started")
                || error.message.contains("failed to resolve container IP")
            {
                return SandboxError::new(ErrorCode::SnapshotPersistError, error.message)
                    .with_retryable(false)
                    .with_context("backend", "daytona")
                    .with_context("reason", "sandbox_not_running");
            }
            error
        })?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "Failed to archive workspace `{root}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        let archive = self.download_at(&archive_path, None).await;
        let _ = self.remove_at(&archive_path, false).await;
        archive.map_err(|error| {
            SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "Failed to download the workspace archive: {}",
                    error.message
                ),
            )
        })
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)?;
        let root = self.manifest.root().to_string();
        let archive_path = self.resolve_internal_path(&provider_scratch_path(&format!(
            "{}.tar.gz",
            self.session_id
        )))?;
        self.upload_at(&archive_path, archive).await?;
        let script = format!(
            "mkdir -p {root} && tar -xzf {archive} -C {root}",
            root = shell_quote(&root),
            archive = shell_quote(&archive_path),
        );
        let result = self
            .exec_in_session(
                vec!["sh".into(), "-lc".into(), script],
                "/".to_string(),
                &self.command_envs(),
                Some(self.options.timeouts.workspace_tar),
            )
            .await;
        let _ = self.remove_at(&archive_path, false).await;
        let result = result?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!(
                    "Failed to restore the Daytona workspace: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }
}

impl DaytonaSandboxSession {
    /// The provider handle behind one PTY session.
    fn pty_handle(&self, session_id: u32) -> Option<Arc<DaytonaPtyHandle>> {
        self.pty_handles
            .lock()
            .expect("poisoned")
            .get(&session_id)
            .cloned()
    }

    /// Drop the bookkeeping of a session whose command has exited.
    async fn forget_finished_pty(&self, update: &PtyExecUpdate, session_id: u32) {
        if update.process_id.is_none() {
            self.ptys.remove(session_id);
            self.release_pty(session_id).await;
        }
    }

    /// Release the provider-side resources behind one PTY session.
    async fn release_pty(&self, session_id: u32) {
        let handle = self
            .pty_handles
            .lock()
            .expect("poisoned")
            .remove(&session_id);
        match handle.as_deref() {
            Some(DaytonaPtyHandle::Terminal { pty_session, .. }) => {
                self.kill_pty_session(pty_session).await
            }
            // Deleting the toolbox session terminates the commands running in it, which is the
            // only command teardown the toolbox API exposes.
            Some(DaytonaPtyHandle::Logged {
                toolbox_session, ..
            }) => self.delete_toolbox_session(toolbox_session).await,
            None => {}
        }
    }

    /// Release the sessions the registry pruned to make room.
    async fn release_pruned(&self, pruned: &[Arc<RemotePtyProcess>]) {
        for process in pruned {
            process.close(None);
            self.release_pty(process.id()).await;
        }
    }

    /// Start an interactive command as an asynchronous toolbox session command.
    ///
    /// The command's output is read back from the session log endpoint, which is how a
    /// non-writable interactive process works in Python too.
    /// Bound one step of starting a PTY process by the request's timeout.
    ///
    /// Python wraps each provider call of `pty_exec_start` in `asyncio.wait_for(...,
    /// timeout=exec_timeout)`: without a timeout the step is unbounded, a zero timeout is the
    /// smallest bound, and an elapsed one is an `ExecTimeout`.
    async fn within_start_timeout<T>(
        &self,
        request: &PtyExecRequest,
        argv: &[String],
        step: impl std::future::Future<Output = SandboxResult<T>>,
    ) -> SandboxResult<T> {
        let Some(timeout) = request.timeout else {
            return step.await;
        };
        let budget = if timeout.is_zero() {
            Duration::from_millis(1)
        } else {
            timeout
        };
        match tokio::time::timeout(budget, step).await {
            Ok(result) => result,
            Err(_) => Err(SandboxError::new(
                ErrorCode::ExecTimeout,
                format!("Command `{}` timed out after {budget:?}", argv.join(" ")),
            )),
        }
    }

    async fn start_logged_command(
        &self,
        argv: &[String],
        cwd: &str,
        request: &PtyExecRequest,
    ) -> SandboxResult<PtyExecUpdate> {
        let command = self.session_command(argv, cwd, &self.command_envs());
        let toolbox_session = format!("pty-{}", uuid::Uuid::new_v4().simple());
        let what = "start an interactive process in the Daytona sandbox";
        #[derive(Deserialize)]
        struct Started {
            #[serde(rename = "cmdId")]
            cmd_id: String,
        }
        // Each step of the start is bounded by the call's timeout, as Python bounds each
        // awaited provider call of `pty_exec_start`.
        if let Err(error) = self
            .within_start_timeout(request, argv, self.create_toolbox_session(&toolbox_session))
            .await
        {
            // A step that timed out may still have created the session.
            self.delete_toolbox_session(&toolbox_session).await;
            return Err(error);
        }
        let started: Result<Started, SandboxError> = self
            .within_start_timeout(request, argv, async {
                let response = self
                    .authorize(
                        self.http.post(
                            self.toolbox(&format!("/process/session/{toolbox_session}/exec")),
                        ),
                    )
                    .json(&json!({ "command": command, "runAsync": true }))
                    .send()
                    .await
                    .map_err(|e| transport(what, e))?;
                parse_json(response, what).await
            })
            .await;
        let started = match started {
            Ok(started) => started,
            Err(error) => {
                self.delete_toolbox_session(&toolbox_session).await;
                return Err(error);
            }
        };
        let (process, pruned) = self.ptys.register(false)?;
        self.release_pruned(&pruned).await;
        self.pty_handles.lock().expect("poisoned").insert(
            process.id(),
            Arc::new(DaytonaPtyHandle::Logged {
                toolbox_session: toolbox_session.clone(),
                command_id: started.cmd_id.clone(),
            }),
        );
        self.spawn_log_poller(process.clone(), toolbox_session, started.cmd_id);
        let update = process
            .collect(request.yield_time_ms, request.max_output_tokens)
            .await;
        self.forget_finished_pty(&update, process.id()).await;
        Ok(update)
    }

    /// Poll a session command's log into `process` until the command exits.
    ///
    /// The log endpoint always returns the whole log, so the task remembers how much of it it
    /// has already delivered.
    fn spawn_log_poller(
        &self,
        process: Arc<RemotePtyProcess>,
        toolbox_session: String,
        command_id: String,
    ) {
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let organization_id = self.options.resolved_organization_id();
        let toolbox_base = self.toolbox_base.read().expect("poisoned").clone();
        let interval = self.options.state_poll_interval;
        tokio::spawn(async move {
            let authorize = |builder: reqwest::RequestBuilder| {
                let mut builder = builder
                    .bearer_auth(&api_key)
                    .header("X-Daytona-Source", DAYTONA_SOURCE);
                if let Some(organization_id) = &organization_id {
                    builder = builder.header("X-Daytona-Organization-ID", organization_id);
                }
                builder
            };
            let logs_url = format!(
                "{toolbox_base}/process/session/{toolbox_session}/command/{command_id}/logs"
            );
            let status_url =
                format!("{toolbox_base}/process/session/{toolbox_session}/command/{command_id}");
            let mut delivered = 0usize;
            while !process.is_done() {
                if let Ok(response) = authorize(http.get(&logs_url)).send().await
                    && response.status().is_success()
                    && let Ok(body) = response.bytes().await
                    && body.len() > delivered
                {
                    process.push_output(&body[delivered..]);
                    delivered = body.len();
                }
                let exit_code = match authorize(http.get(&status_url)).send().await {
                    Ok(response) if response.status().is_success() => response
                        .json::<Value>()
                        .await
                        .ok()
                        .and_then(|value| value.get("exitCode").and_then(Value::as_i64))
                        .map(|code| code as i32),
                    _ => None,
                };
                if let Some(exit_code) = exit_code {
                    // One last read, so output written just before the exit is not dropped.
                    if let Ok(response) = authorize(http.get(&logs_url)).send().await
                        && response.status().is_success()
                        && let Ok(body) = response.bytes().await
                        && body.len() > delivered
                    {
                        process.push_output(&body[delivered..]);
                    }
                    process.close(Some(exit_code));
                    return;
                }
                tokio::time::sleep(interval).await;
            }
        });
    }

    /// Start a writable terminal over the toolbox PTY websocket.
    ///
    /// Mirrors the vendor SDK's `create_pty_session`: one `create-connect` upgrade carries the
    /// session id and terminal size as query parameters and the environment as a WebSocket
    /// subprotocol token, so neither secrets nor large values land in a URL or an access log.
    async fn start_terminal(
        &self,
        argv: &[String],
        cwd: &str,
        request: &PtyExecRequest,
    ) -> SandboxResult<PtyExecUpdate> {
        let requested_session = format!("pty-{}", uuid::Uuid::new_v4().simple());
        let (process, pruned) = self.ptys.register(true)?;
        self.release_pruned(&pruned).await;
        let connected = match self
            .within_start_timeout(
                request,
                argv,
                self.connect_terminal(&process, &requested_session, cwd),
            )
            .await
        {
            Ok(connected) => connected,
            Err(error) => {
                self.ptys.remove(process.id());
                self.kill_pty_session(&requested_session).await;
                return Err(error);
            }
        };
        // A fallback connection runs under a different id than the one first requested.
        let pty_session = connected.pty_session.clone();
        self.pty_handles.lock().expect("poisoned").insert(
            process.id(),
            Arc::new(DaytonaPtyHandle::Terminal {
                pty_session: pty_session.clone(),
                input: connected.input.clone(),
            }),
        );
        // The terminal runs a shell; the command reaches it as a typed line, as in Python.
        let joined = argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let line = if connected.envs_delivered {
            format!("{joined}\n")
        } else {
            // The handshake could not carry the environment, so it is exported on the typed
            // line the same way the non-terminal path exports it.
            format!("{}{joined}\n", env_command_prefix(&self.command_envs()))
        };
        if connected.input.send(line.into_bytes()).is_err() {
            self.ptys.remove(process.id());
            self.release_pty(process.id()).await;
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("the Daytona PTY session `{pty_session}` closed before it could be used"),
            ));
        }
        let update = process
            .collect(request.yield_time_ms, request.max_output_tokens)
            .await;
        self.forget_finished_pty(&update, process.id()).await;
        Ok(update)
    }

    /// Open the PTY websocket and pump it into `process`.
    ///
    /// The vendor SDK sends the sandbox environment as a WebSocket subprotocol token, but
    /// RFC 6455 lets a server accept an upgrade without selecting a subprotocol, and
    /// `tungstenite` then fails the connection where `aiohttp` (which the vendor SDK uses) does
    /// not. When that happens the connection is retried without subprotocols and the caller
    /// passes the environment on the typed command line instead.
    async fn connect_terminal(
        &self,
        process: &Arc<RemotePtyProcess>,
        pty_session: &str,
        cwd: &str,
    ) -> SandboxResult<ConnectedTerminal> {
        let envs = self.command_envs();
        let mut protocols = vec![PTY_EXIT_CONTROL_SUBPROTOCOL.to_string()];
        if !envs.is_empty() {
            let encoded =
                BASE64_URL_SAFE_NO_PAD.encode(serde_json::to_vec(&envs).expect("serializable"));
            protocols.push(format!("{PTY_ENVS_SUBPROTOCOL_PREFIX}{encoded}"));
        }
        let mut pty_session = pty_session.to_string();
        let mut envs_delivered = !envs.is_empty();
        let stream = match self
            .open_pty_socket(&pty_session, cwd, Some(&protocols))
            .await
        {
            Ok(stream) => stream,
            Err(TerminalConnectError::NoSubProtocol) => {
                // The upgrade already returned 101, so the daemon has created this PTY session
                // and started its shell. Reusing the id would be rejected and would orphan that
                // shell, so it is killed and the retry gets a fresh one.
                self.kill_pty_session(&pty_session).await;
                pty_session = format!("pty-{}", uuid::Uuid::new_v4().simple());
                envs_delivered = false;
                self.open_pty_socket(&pty_session, cwd, None)
                    .await
                    .map_err(TerminalConnectError::into_sandbox_error)?
            }
            Err(error) => return Err(error.into_sandbox_error()),
        };
        let (sink, source) = stream.split();

        let handshake = Arc::new(TerminalHandshake::default());
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
        spawn_terminal_reader(process.clone(), handshake.clone(), source);
        spawn_terminal_writer(sink, input_rx);

        // The daemon acknowledges with a `connected` control frame, or reports `exited` when the
        // shell is already gone.
        let deadline = Instant::now() + PTY_CONNECT_TIMEOUT;
        while !handshake.settled.load(Ordering::SeqCst) {
            if process.is_done() {
                break;
            }
            if Instant::now() >= deadline {
                return Err(SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("the Daytona PTY session `{pty_session}` did not connect in time"),
                ));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // The daemon reports a refused session as a control frame rather than a close.
        if let Some(message) = handshake.error.lock().expect("poisoned").clone() {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("the PTY session `{pty_session}` failed to start: {message}"),
            ));
        }
        Ok(ConnectedTerminal {
            pty_session,
            input: input_tx,
            envs_delivered,
        })
    }

    /// Open one PTY websocket, with or without the vendor's subprotocol tokens.
    async fn open_pty_socket(
        &self,
        pty_session: &str,
        cwd: &str,
        protocols: Option<&[String]>,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        TerminalConnectError,
    > {
        let mut url = reqwest::Url::parse(&self.toolbox("/process/pty/create-connect"))
            .map_err(|e| TerminalConnectError::Other(e.to_string()))?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("id", pty_session);
            query.append_pair("cols", &PTY_TERMINAL_COLS.to_string());
            query.append_pair("rows", &PTY_TERMINAL_ROWS.to_string());
            if !cwd.is_empty() {
                query.append_pair("cwd", cwd);
            }
        }
        // The toolbox base is an HTTP URL; the websocket lives on the same host and path.
        let ws_url = url.as_str().replacen("http", "ws", 1);
        let mut request = ws_url
            .as_str()
            .into_client_request()
            .map_err(|e| TerminalConnectError::Other(e.to_string()))?;
        self.apply_ws_headers(request.headers_mut(), protocols)
            .map_err(|error| TerminalConnectError::Other(error.message))?;

        crate::util::ensure_crypto_provider();
        match tokio_tungstenite::connect_async(request).await {
            Ok((stream, _response)) => Ok(stream),
            Err(error) if is_no_subprotocol_error(&error) => {
                Err(TerminalConnectError::NoSubProtocol)
            }
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                Err(TerminalConnectError::Http(response.status().as_u16()))
            }
            Err(error) => Err(TerminalConnectError::Other(error.to_string())),
        }
    }

    /// Attach the REST credentials, and optionally the subprotocols, to a websocket handshake.
    fn apply_ws_headers(
        &self,
        headers: &mut tokio_tungstenite::tungstenite::http::HeaderMap,
        protocols: Option<&[String]>,
    ) -> SandboxResult<()> {
        let bearer = HeaderValue::from_str(&format!("Bearer {}", self.api_key)).map_err(|_| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "the Daytona API key contains characters that cannot be sent in a header",
            )
        })?;
        headers.insert(HeaderName::from_static("authorization"), bearer);
        headers.insert(
            HeaderName::from_static("x-daytona-source"),
            HeaderValue::from_static(DAYTONA_SOURCE),
        );
        if let Some(organization_id) = self.options.resolved_organization_id()
            && let Ok(value) = HeaderValue::from_str(&organization_id)
        {
            headers.insert(HeaderName::from_static("x-daytona-organization-id"), value);
        }
        if let Some(protocols) = protocols {
            let protocol = HeaderValue::from_str(&protocols.join(", ")).map_err(|_| {
                SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "a sandbox environment variable cannot be encoded for the PTY handshake",
                )
            })?;
            headers.insert(HeaderName::from_static("sec-websocket-protocol"), protocol);
        }
        Ok(())
    }

    /// Resize the terminal behind one PTY session.
    ///
    /// Terminal sessions only; a logged command has no terminal to resize.
    pub async fn resize_terminal(
        &self,
        session_id: u32,
        cols: u16,
        rows: u16,
    ) -> SandboxResult<()> {
        let handle = self.pty_handle(session_id).ok_or_else(|| {
            SandboxError::new(
                ErrorCode::PtySessionNotFound,
                format!("PTY session `{session_id}` is not running"),
            )
        })?;
        let DaytonaPtyHandle::Terminal { pty_session, .. } = handle.as_ref() else {
            return Err(SandboxError::new(
                ErrorCode::PtyStdinUnavailable,
                format!("PTY session `{session_id}` is not a writable terminal"),
            ));
        };
        let what = "resize the Daytona PTY session";
        let response = self
            .authorize(
                self.http
                    .post(self.toolbox(&format!("/process/pty/{pty_session}/resize"))),
            )
            .json(&json!({ "cols": cols, "rows": rows }))
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        expect_success(response, what, ErrorCode::ExecTransportError).await
    }

    /// Kill a PTY session, best effort.
    async fn kill_pty_session(&self, pty_session: &str) {
        let request = self
            .authorize(
                self.http
                    .delete(self.toolbox(&format!("/process/pty/{pty_session}"))),
            )
            .send();
        let _ = tokio::time::timeout(self.options.timeouts.cleanup, request).await;
    }
}

// -------------------------------------------------------------------------------------------
// Cloud bucket mounts
// -------------------------------------------------------------------------------------------

/// Refuse a mount on any backend other than Daytona.
///
/// Mirrors Python's `_assert_daytona_session`: the strategy's preflight assumes this
/// provider's image and its privilege model, so running it elsewhere would be guesswork.
fn assert_daytona_session(session: &dyn SandboxSession) -> SandboxResult<()> {
    let matches = session
        .as_any()
        .is_some_and(|any| any.is::<DaytonaSandboxSession>());
    if !matches {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "a Daytona cloud bucket mount requires a Daytona sandbox session, not the \
                 `{}` backend",
                session.backend_id()
            ),
        ));
    }
    Ok(())
}

/// Mounts rclone-backed cloud storage inside a Daytona sandbox. Mirrors
/// `agents.extensions.sandbox.daytona.mounts.DaytonaCloudBucketMountStrategy`.
///
/// Activation verifies FUSE support, installs `fuse3` and `rclone` when the image lacks them, and
/// then delegates to [`InContainerMountStrategy`]. The rclone configuration carries the bucket
/// credentials, so the mount reports [`MountCredentialAuthority::MountScoped`] and the manifest
/// has to acknowledge the exact mount path before it activates.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct DaytonaCloudBucketMountStrategy {
    /// The rclone pattern that attaches the bucket.
    pub pattern: RcloneMountPattern,
    /// The secret-bearing rclone configuration and the remote selection options.
    ///
    /// Ignored when [`spec`](Self::spec) is set, which derives the configuration per session.
    #[serde(skip)]
    pub config: MountPatternConfig,
    /// The storage this strategy attaches, when it was built from a configured mount.
    ///
    /// A configured mount knows how to render its own rclone section, so the configuration is
    /// built on each activation against the session it is attached to instead of being pinned
    /// when the strategy is declared.
    #[serde(skip)]
    pub spec: Option<Arc<dyn ConfiguredMount>>,
    /// Whether a missing `rclone` (or `fuse3`) may be installed into the sandbox.
    pub install_tools: bool,
}

impl DaytonaCloudBucketMountStrategy {
    /// A strategy mounting `remote_name:remote_path` from `config_text`.
    pub fn new(
        remote_name: impl Into<String>,
        remote_path: impl Into<String>,
        config_text: impl Into<String>,
    ) -> Self {
        DaytonaCloudBucketMountStrategy {
            pattern: RcloneMountPattern::new(),
            spec: None,
            install_tools: true,
            config: MountPatternConfig::new()
                .with_config_text(config_text)
                .with_option(RCLONE_OPTION_REMOTE_NAME, remote_name)
                .with_option(RCLONE_OPTION_REMOTE_PATH, remote_path),
        }
    }

    /// A strategy attaching one of the built-in remote storage mounts.
    ///
    /// The mount renders its own rclone remote on every activation, so the credentials never
    /// have to be rendered by the caller and the remote name is derived from the live session.
    pub fn for_mount(spec: impl ConfiguredMount + 'static) -> Self {
        DaytonaCloudBucketMountStrategy {
            pattern: RcloneMountPattern::new(),
            spec: Some(Arc::new(spec)),
            install_tools: true,
            config: MountPatternConfig::new(),
        }
    }

    /// Replace the rclone pattern.
    pub fn with_pattern(mut self, pattern: RcloneMountPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Whether a missing `rclone` (or `fuse3`) may be installed into the sandbox.
    pub fn with_install_tools(mut self, install_tools: bool) -> Self {
        self.install_tools = install_tools;
        self
    }

    /// A [`Mount`] using this strategy.
    pub fn mount(self) -> Mount {
        Mount::new(Arc::new(self))
    }

    fn delegate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        include_credentials: bool,
    ) -> SandboxResult<InContainerMountStrategy> {
        let pattern = self.fuse_pattern();
        let config = match &self.spec {
            Some(spec) => spec.build_in_container_config(
                session.id(),
                &BuiltinMountPattern::Rclone(pattern.clone()),
                mount.read_only,
                include_credentials,
            )?,
            None => {
                let mut config = self.config.clone();
                config.options.insert(
                    RCLONE_OPTION_READ_ONLY.to_string(),
                    mount.read_only.to_string(),
                );
                config
            }
        };
        Ok(InContainerMountStrategy::new(Arc::new(pattern), config))
    }

    /// The pattern the delegate mounts with.
    ///
    /// Daytona passes the configured pattern through untouched, as its Python counterpart does
    /// (`daytona/mounts.py` `_delegate`). Unlike E2B, it never calls `rclone_pattern_for_session`,
    /// so no `--allow-other`, `--uid` or `--gid` is added: a FUSE mount that has to be readable
    /// by a non-root command user carries those flags in the pattern's `extra_args`.
    fn fuse_pattern(&self) -> RcloneMountPattern {
        self.pattern.clone()
    }

    async fn preflight(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        if self.pattern.mode == RcloneMountMode::Fuse {
            ensure_fuse_support(session, self.install_tools).await?;
        }
        ensure_rclone(session, self.install_tools).await
    }
}

#[async_trait]
impl MountStrategy for DaytonaCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "daytona_cloud_bucket"
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        if let Some(spec) = &self.spec {
            return spec.credential_authority("rclone", self.pattern.config_file_path.is_some());
        }
        if self.config.has_credentials() {
            MountCredentialAuthority::MountScoped
        } else {
            MountCredentialAuthority::None
        }
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        let Some(spec) = &self.spec else {
            return Ok(());
        };
        spec.validate()?;
        if !spec.supported_in_container_patterns().contains(&"rclone") {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "`{}` cannot be attached with the `rclone` mount pattern",
                    spec.mount_type()
                ),
            ));
        }
        Ok(())
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        false
    }

    async fn activate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        assert_daytona_session(session)?;
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        self.delegate(mount, session, true)?
            .activate(mount, session, mount_path)
            .await
    }

    async fn deactivate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        assert_daytona_session(session)?;
        self.delegate(mount, session, false)?
            .deactivate(mount, session, mount_path)
            .await
    }

    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        assert_daytona_session(session)?;
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        self.delegate(mount, session, true)?
            .activate(mount, session, mount_path)
            .await
    }
}

// -------------------------------------------------------------------------------------------
// Interactive process bookkeeping
// -------------------------------------------------------------------------------------------

/// What the session needs to drive one interactive process.
///
/// The core [`RemotePtyManager`] owns the session id, the output window and the pruning; this is
/// the provider-side half it knows nothing about.
#[derive(Debug)]
enum DaytonaPtyHandle {
    /// A writable terminal on the toolbox PTY websocket.
    Terminal {
        /// The PTY session id the toolbox knows it by.
        pty_session: String,
        /// Sends bytes to the websocket writer task.
        input: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    },
    /// An asynchronous session command, read back from the log endpoint.
    Logged {
        /// The toolbox session holding the command.
        toolbox_session: String,
        /// The command inside that session.
        command_id: String,
    },
}

/// The handshake state of a terminal, observed while the connection is being established.
#[derive(Debug, Default)]
struct TerminalHandshake {
    /// Whether the daemon has answered, either by connecting or by refusing.
    settled: AtomicBool,
    /// The failure the daemon reported in a control frame, if any.
    error: Mutex<Option<String>>,
}

/// A terminal whose websocket is open.
struct ConnectedTerminal {
    /// The PTY session id the connection actually used.
    pty_session: String,
    /// Sends bytes to the websocket writer task.
    input: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    /// Whether the handshake carried the sandbox environment.
    envs_delivered: bool,
}

/// Why a PTY websocket could not be opened.
enum TerminalConnectError {
    /// The server answered the upgrade with an HTTP error.
    Http(u16),
    /// The server accepted the upgrade without selecting a subprotocol.
    NoSubProtocol,
    /// Anything else.
    Other(String),
}

impl TerminalConnectError {
    fn into_sandbox_error(self) -> SandboxError {
        if let TerminalConnectError::Http(status) = self {
            let mut error = SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to open the Daytona PTY websocket: HTTP {status}"),
            )
            .with_context("backend", "daytona")
            .with_context("http_status", status.to_string());
            error.retryable = daytona_status_retryable(status);
            return error;
        }
        let detail = match self {
            TerminalConnectError::NoSubProtocol => {
                "the server selected no websocket subprotocol".to_string()
            }
            TerminalConnectError::Other(detail) => detail,
            TerminalConnectError::Http(_) => unreachable!("handled above"),
        };
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to open the Daytona PTY websocket: {detail}"),
        )
    }
}

/// Read the PTY websocket into `process` until it closes.
fn spawn_terminal_reader<S>(
    process: Arc<RemotePtyProcess>,
    handshake: Arc<TerminalHandshake>,
    source: S,
) where
    S: futures::Stream<
            Item = Result<
                tokio_tungstenite::tungstenite::Message,
                tokio_tungstenite::tungstenite::Error,
            >,
        > + Send
        + Unpin
        + 'static,
{
    use tokio_tungstenite::tungstenite::Message;
    tokio::spawn(async move {
        let mut source = source;
        let mut exit_code = None;
        // The daemon encodes the exit code in the close frame, so the close reason matters.
        let mut close_code = None;
        let mut close_reason = None;
        while let Some(message) = source.next().await {
            match message {
                Ok(Message::Text(text)) => {
                    match parse_terminal_control(text.as_str()) {
                        Some(control) => {
                            handshake.settled.store(true, Ordering::SeqCst);
                            if let Some(message) = control.error {
                                *handshake.error.lock().expect("poisoned") = Some(message);
                            }
                            if let Some(code) = control.exit_code {
                                exit_code = Some(code);
                            }
                        }
                        // Anything that is not a control frame is terminal output.
                        None => process.push_output(text.as_bytes()),
                    }
                }
                Ok(Message::Binary(data)) => process.push_output(&data),
                Ok(Message::Close(frame)) => {
                    if let Some(frame) = frame {
                        close_code = Some(u16::from(frame.code));
                        let reason = frame.reason.to_string();
                        if !reason.is_empty() {
                            close_reason = Some(reason);
                        }
                    }
                    break;
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        if exit_code.is_none() {
            exit_code = close_frame_exit_code(close_code, close_reason.as_deref());
        }
        // A connection that drops without ever answering must not leave the connect wait hanging.
        handshake.settled.store(true, Ordering::SeqCst);
        process.close(exit_code);
    });
}

/// One `{"type":"control"}` frame from the PTY daemon.
#[derive(Debug, Default, PartialEq, Eq)]
struct TerminalControl {
    /// The exit code, when the frame reports the shell exiting.
    exit_code: Option<i32>,
    /// The failure message, when the frame reports an error.
    error: Option<String>,
}

/// Parse a PTY control frame; anything else is terminal output.
fn parse_terminal_control(text: &str) -> Option<TerminalControl> {
    let value = serde_json::from_str::<Value>(text).ok()?;
    if value.get("type").and_then(Value::as_str) != Some("control") {
        return None;
    }
    let mut control = TerminalControl::default();
    match value.get("status").and_then(Value::as_str) {
        Some("exited") => {
            control.exit_code = value
                .get("exitCode")
                .and_then(Value::as_i64)
                .map(|code| code as i32);
        }
        Some("error") => {
            control.error = Some(
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown connection error")
                    .to_string(),
            );
        }
        _ => {}
    }
    Some(control)
}

/// The exit code a websocket close frame carries.
///
/// The daemon encodes `{"exitCode": n}` in the close reason so the client needs no extra round
/// trip; a clean close with no structured reason means the shell exited successfully.
fn close_frame_exit_code(close_code: Option<u16>, close_reason: Option<&str>) -> Option<i32> {
    if let Some(reason) = close_reason
        && let Ok(value) = serde_json::from_str::<Value>(reason)
        && let Some(code) = value.get("exitCode").and_then(Value::as_i64)
    {
        return Some(code as i32);
    }
    (close_code == Some(1000)).then_some(0)
}

/// Forward queued input to the PTY websocket, keeping it alive with periodic pings.
fn spawn_terminal_writer<S>(sink: S, mut input: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>)
where
    S: futures::Sink<tokio_tungstenite::tungstenite::Message> + Send + Unpin + 'static,
{
    use futures::SinkExt as _;
    use tokio_tungstenite::tungstenite::Message;
    tokio::spawn(async move {
        let mut sink = sink;
        let mut keepalive = tokio::time::interval(PTY_KEEPALIVE_INTERVAL);
        keepalive.tick().await;
        loop {
            tokio::select! {
                chunk = input.recv() => match chunk {
                    Some(chunk) => {
                        if sink.send(Message::binary(chunk)).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                },
                // A terminal can stay silent for a long time, and proxies drop idle
                // connections; a ping keeps it open without typing into the shell.
                _ = keepalive.tick() => {
                    if sink.send(Message::Ping(Default::default())).await.is_err() {
                        break;
                    }
                }
            }
        }
        let _ = sink.close().await;
    });
}

// -------------------------------------------------------------------------------------------
// Shared helpers
// -------------------------------------------------------------------------------------------

/// Whether a websocket failure is `tungstenite`'s "the server selected no subprotocol" check.
fn is_no_subprotocol_error(error: &tokio_tungstenite::tungstenite::Error) -> bool {
    use tokio_tungstenite::tungstenite::error::{ProtocolError, SubProtocolError};
    matches!(
        error,
        tokio_tungstenite::tungstenite::Error::Protocol(
            ProtocolError::SecWebSocketSubProtocolError(SubProtocolError::NoSubProtocol)
        )
    )
}

/// The `env -- K=V ...` prefix that exports `envs` for one command.
fn env_command_prefix(envs: &BTreeMap<String, String>) -> String {
    if envs.is_empty() {
        return String::new();
    }
    let assignments = envs
        .iter()
        .map(|(key, value)| shell_quote(&format!("{key}={value}")))
        .collect::<Vec<_>>()
        .join(" ");
    format!("env -- {assignments} ")
}

/// Build a one-field `multipart/form-data` body and its `Content-Type` header.
///
/// `reqwest`'s `multipart` feature is not enabled by the `sandbox` feature, and the toolbox
/// upload endpoint only ever needs a single part.
fn multipart_file(field: &str, filename: &str, contents: &[u8]) -> (String, Vec<u8>) {
    let boundary = format!("openai-agents-{}", uuid::Uuid::new_v4().simple());
    // The boundary is a fresh UUID, so no filename can terminate the part early.
    let escaped = filename.replace('"', "");
    let mut body = Vec::with_capacity(contents.len() + 256);
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"{field}\"; filename=\"{escaped}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(contents);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Convert a toolbox `FileInfo` into a [`FileEntry`].
fn file_info_to_file_entry(value: &Value, base: &str) -> FileEntry {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let is_dir = value
        .get("isDir")
        .and_then(Value::as_bool)
        .unwrap_or_default();
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}/{name}", base.trim_end_matches('/')));
    let permissions = value
        .get("permissions")
        .and_then(Value::as_str)
        .and_then(|permissions| {
            u32::from_str_radix(permissions, 8)
                .ok()
                .map(|mode| super::e2b::permissions_from_mode(mode, is_dir))
                .or_else(|| {
                    Permissions::from_ls_field(&format!(
                        "{}{permissions}",
                        if is_dir { "d" } else { "-" }
                    ))
                })
        });
    FileEntry {
        name,
        path,
        size: value.get("size").and_then(Value::as_u64),
        permissions,
        is_dir,
        // The toolbox file-info payload reports no link metadata.
        is_symlink: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_the_api_key() {
        let options = DaytonaSandboxClientOptions::new().with_api_key("dtn_secret");
        let debug = format!("{options:?}");
        assert!(!debug.contains("dtn_secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn resolves_the_toolbox_base_url() {
        let sandbox = DaytonaSandbox {
            id: "sb-1".into(),
            state: Some("started".into()),
            toolbox_proxy_url: Some("https://proxy.daytona.io/toolbox/".into()),
        };
        assert_eq!(
            toolbox_base_url("https://app.daytona.io/api", &sandbox),
            "https://proxy.daytona.io/toolbox/sb-1"
        );
        let without_proxy = DaytonaSandbox {
            id: "sb-1".into(),
            state: None,
            toolbox_proxy_url: None,
        };
        assert_eq!(
            toolbox_base_url("https://app.daytona.io/api", &without_proxy),
            "https://app.daytona.io/api/toolbox/sb-1/toolbox"
        );
    }

    #[test]
    fn wraps_commands_with_the_working_directory_and_environment() {
        let manifest = Manifest::new()
            .with_root("/home/daytona/workspace")
            .with_env("FOO", "bar baz");
        let session = DaytonaSandboxSession::new(
            "session-1".into(),
            DaytonaSandboxClientOptions::new().with_api_key("dtn_test"),
            "dtn_test".into(),
            reqwest::Client::new(),
            DaytonaSandbox {
                id: "sb-1".into(),
                state: Some("started".into()),
                toolbox_proxy_url: None,
            },
            manifest,
            None,
        );
        let command = session.session_command(
            &["echo".to_string(), "hi there".to_string()],
            "/home/daytona/workspace",
            &session.command_envs(),
        );
        assert_eq!(
            command,
            "cd /home/daytona/workspace && env -- 'FOO=bar baz' echo 'hi there'"
        );
        // A per-command variable wins over the manifest value of the same name.
        let overridden = session.session_command(
            &["echo".to_string()],
            "/home/daytona/workspace",
            &ExecRequest::argv(["echo"])
                .with_env("FOO", "override")
                .resolved_env(session.manifest().environment()),
        );
        assert_eq!(
            overridden,
            "cd /home/daytona/workspace && env -- FOO=override echo"
        );
    }

    #[test]
    fn splits_session_output_by_exit_code() {
        let combined = SessionExecuteResponse {
            exit_code: Some(2),
            output: Some("boom".into()),
            stdout: None,
            stderr: None,
        }
        .into_exec_result();
        assert_eq!(combined.stderr_text(), "boom");
        assert!(combined.stdout.is_empty());

        let split = SessionExecuteResponse {
            exit_code: None,
            output: None,
            stdout: Some("out".into()),
            stderr: Some("err".into()),
        }
        .into_exec_result();
        assert_eq!(split.exit_code, 0);
        assert_eq!(split.stdout_text(), "out");
        assert_eq!(split.stderr_text(), "err");
    }

    #[test]
    fn builds_a_single_part_multipart_body() {
        let (content_type, body) = multipart_file("file", "task.md", b"hello");
        let boundary = content_type
            .split("boundary=")
            .nth(1)
            .expect("boundary")
            .to_string();
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.contains("name=\"file\"; filename=\"task.md\""));
        assert!(text.ends_with(&format!("\r\n--{boundary}--\r\n")));
        assert!(text.contains("hello"));
    }

    #[test]
    fn maps_file_info_to_file_entries() {
        let entry = file_info_to_file_entry(
            &json!({ "name": "task.md", "isDir": false, "size": 12, "permissions": "644" }),
            "/home/daytona/workspace",
        );
        assert_eq!(entry.path, "/home/daytona/workspace/task.md");
        assert_eq!(entry.size, Some(12));
        assert_eq!(entry.permissions.unwrap().octal_string(), "0644");
    }
}
