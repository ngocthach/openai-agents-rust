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
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mounts::{
    InContainerMountStrategy, Mount, MountCredentialAuthority, MountPatternConfig, MountStrategy,
};
use crate::sandbox::pty::{
    PTY_PROCESS_ID_MAX_EXCLUSIVE, PTY_PROCESS_ID_MIN, PTY_PROCESSES_MAX, PtyExecRequest,
    PtyExecUpdate, PtyWriteRequest, clamp_pty_yield_time_ms, resolve_pty_write_yield_time_ms,
};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot, shell_quote,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::{ExecResult, FileEntry, Permissions};
use crate::sandbox::util::truncate_text_by_tokens;

use super::e2b::rclone::{
    self, RcloneMountMode, RcloneMountPattern, ensure_fuse_support, ensure_rclone,
};
use super::e2b::{
    adopt_recorded_root, expect_success, http_error, parse_json, transport, workspace_excludes,
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

/// Resource limits for a Daytona sandbox. Mirrors Python's `DaytonaSandboxResources`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

/// Timeouts applied to Daytona operations. Mirrors Python's `DaytonaSandboxTimeouts`.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Clone, PartialEq, Eq)]
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
            .header("X-Daytona-Source", "openai-agents-sdk-rust");
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
    ptys: PtyRegistry,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
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
            ptys: PtyRegistry::default(),
            snapshot,
            resuming: false,
        }
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
            .header("X-Daytona-Source", "openai-agents-sdk-rust");
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

    /// The environment every command runs with.
    fn command_envs(&self) -> BTreeMap<String, String> {
        let mut envs = self.options.env_vars.clone();
        for (key, value) in self.manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        envs
    }

    /// Wrap `argv` as a shell command that enters `cwd` and exports the manifest environment.
    ///
    /// Mirrors Python's `DaytonaSandboxSession._exec_internal`: the toolbox API takes one command
    /// string, with no separate working directory or environment.
    fn session_command(&self, argv: &[String], cwd: &str) -> String {
        let joined = argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let envs = self.command_envs();
        let env_prefix = if envs.is_empty() {
            String::new()
        } else {
            let assignments = envs
                .iter()
                .map(|(key, value)| shell_quote(&format!("{key}={value}")))
                .collect::<Vec<_>>()
                .join(" ");
            format!("env -- {assignments} ")
        };
        format!("cd {} && {env_prefix}{joined}", shell_quote(cwd))
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
        timeout: Option<Duration>,
    ) -> SandboxResult<ExecResult> {
        let what = "run a command in the Daytona sandbox";
        let toolbox_session = format!("sandbox-{}", uuid::Uuid::new_v4().simple());
        let command = self.session_command(&argv, &cwd);
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

    /// Fetch the combined log of one session command.
    async fn command_logs(
        &self,
        toolbox_session: &str,
        command_id: &str,
    ) -> SandboxResult<Vec<u8>> {
        let what = "read Daytona command logs";
        let response = self
            .authorize(self.http.get(self.toolbox(&format!(
                "/process/session/{toolbox_session}/command/{command_id}/logs"
            ))))
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(http_error(
                ErrorCode::ExecTransportError,
                what,
                status,
                &body,
            ));
        }
        Ok(response
            .bytes()
            .await
            .map_err(|e| transport(what, e))?
            .to_vec())
    }

    /// Fetch the exit code of one session command, once it has one.
    async fn command_exit_code(
        &self,
        toolbox_session: &str,
        command_id: &str,
    ) -> SandboxResult<Option<i32>> {
        let what = "read a Daytona command status";
        let response = self
            .authorize(self.http.get(self.toolbox(&format!(
                "/process/session/{toolbox_session}/command/{command_id}"
            ))))
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        #[derive(Deserialize)]
        struct Command {
            #[serde(rename = "exitCode", default)]
            exit_code: Option<i32>,
        }
        let command: Command = parse_json(response, what).await?;
        Ok(command.exit_code)
    }

    /// Download a file by its absolute sandbox path.
    ///
    /// Workspace reads go through [`read_file`](SandboxSession::read_file), which validates the
    /// path first; this is for the paths the session itself owns, such as a temporary archive
    /// under `/tmp`.
    async fn download_at(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let what = "read a file from the Daytona sandbox";
        let response = self
            .authorize(self.http.get(self.toolbox("/files/download")))
            .query(&[("path", path)])
            .timeout(self.options.timeouts.file_download)
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("`{path}` does not exist in the sandbox"),
            ));
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(http_error(
                ErrorCode::WorkspaceWriteTypeError,
                what,
                status,
                &body,
            ));
        }
        Ok(response
            .bytes()
            .await
            .map_err(|e| transport(what, e))?
            .to_vec())
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
                        log::warn!(
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
        result
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

    async fn start(&self) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        // The toolbox uses the workspace root as the working directory of every command, so it
        // has to exist before the manifest is materialized.
        let prepared = self
            .exec_in_session(
                vec!["mkdir".into(), "-p".into(), "--".into(), root.clone()],
                "/".to_string(),
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
        self.exec_in_session(request.resolved_argv(), cwd, request.timeout)
            .await
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        self.download_at(&resolved).await
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        self.upload_at(&resolved, contents).await
    }

    fn supports_pty(&self) -> bool {
        true
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        let resolved = self.normalize_path(path)?;
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

    async fn mkdir(&self, path: &str, _parents: bool) -> SandboxResult<()> {
        // The toolbox always creates the missing parents.
        let resolved = self.normalize_path(path)?;
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
        let resolved = self.normalize_path(path)?;
        self.remove_at(&resolved, recursive).await
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let source = self.normalize_path(source)?;
        let destination = self.normalize_path(destination)?;
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
        let command = self.session_command(&argv, &cwd);
        let toolbox_session = format!("pty-{}", uuid::Uuid::new_v4().simple());
        self.create_toolbox_session(&toolbox_session).await?;
        let what = "start an interactive process in the Daytona sandbox";
        let response = self
            .authorize(
                self.http
                    .post(self.toolbox(&format!("/process/session/{toolbox_session}/exec"))),
            )
            .json(&json!({ "command": command, "runAsync": true }))
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        #[derive(Deserialize)]
        struct Started {
            #[serde(rename = "cmdId")]
            cmd_id: String,
        }
        let started: Started = match parse_json(response, what).await {
            Ok(started) => started,
            Err(error) => {
                self.delete_toolbox_session(&toolbox_session).await;
                return Err(error);
            }
        };
        let process = self
            .ptys
            .insert(toolbox_session.clone(), started.cmd_id.clone())?;
        Ok(self
            .poll_pty(&process, request.yield_time_ms, request.max_output_tokens)
            .await)
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let process = self.ptys.get(request.session_id)?;
        if !request.chars.is_empty() {
            let what = "write to an interactive process in the Daytona sandbox";
            let response = self
                .authorize(self.http.post(self.toolbox(&format!(
                    "/process/session/{}/command/{}/input",
                    process.toolbox_session, process.command_id
                ))))
                .json(&json!({ "data": request.chars }))
                .send()
                .await
                .map_err(|e| transport(what, e))?;
            expect_success(response, what, ErrorCode::ExecTransportError).await?;
        }
        let yield_time_ms =
            resolve_pty_write_yield_time_ms(request.yield_time_ms, request.chars.is_empty());
        Ok(self
            .poll_pty(&process, yield_time_ms, request.max_output_tokens)
            .await)
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        // Deleting the toolbox session terminates the commands running in it, which is the only
        // teardown the toolbox API exposes.
        for process in self.ptys.terminate_all() {
            self.delete_toolbox_session(&process.toolbox_session).await;
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
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        let root = self.manifest.root().to_string();
        let archive_path = format!("/tmp/openai-agents-{}.tar.gz", self.session_id);
        let mut command = format!(
            "tar -czf {archive} -C {root}",
            archive = shell_quote(&archive_path),
            root = shell_quote(&root),
        );
        for exclude in workspace_excludes(&self.manifest, &root) {
            command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
        }
        command.push_str(" .");
        // The toolbox reports command output as text, so the archive travels over the file API.
        let result = self
            .exec_in_session(
                vec!["sh".into(), "-lc".into(), command],
                root.clone(),
                Some(self.options.timeouts.workspace_tar),
            )
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
        let archive = self.download_at(&archive_path).await;
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

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        let archive_path = format!("/tmp/openai-agents-{}.tar.gz", self.session_id);
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
    /// Collect interactive output until the yield window elapses or the command finishes.
    ///
    /// The toolbox log endpoint always returns the whole log, so the process entry remembers how
    /// much of it the model has already seen.
    async fn poll_pty(
        &self,
        process: &Arc<PtyProcess>,
        yield_time_ms: u64,
        max_output_tokens: Option<usize>,
    ) -> PtyExecUpdate {
        let deadline =
            Instant::now() + Duration::from_millis(clamp_pty_yield_time_ms(yield_time_ms));
        let mut output = Vec::new();
        let mut finished = false;
        loop {
            if let Ok(log) = self
                .command_logs(&process.toolbox_session, &process.command_id)
                .await
            {
                output.extend(process.advance(&log));
            }
            if let Ok(Some(exit_code)) = self
                .command_exit_code(&process.toolbox_session, &process.command_id)
                .await
            {
                *process.exit_code.lock().expect("poisoned") = Some(exit_code);
                finished = true;
                // One last read, so output written just before the exit is not dropped.
                if let Ok(log) = self
                    .command_logs(&process.toolbox_session, &process.command_id)
                    .await
                {
                    output.extend(process.advance(&log));
                }
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let notified = process.notify.notified();
            let wait = (deadline - now).min(self.options.state_poll_interval);
            let _ = tokio::time::timeout(wait, notified).await;
        }
        process.touch();
        if finished {
            self.ptys.remove(process.id);
            self.delete_toolbox_session(&process.toolbox_session).await;
        }
        let text = String::from_utf8_lossy(&output).into_owned();
        let (text, original_token_count) = truncate_text_by_tokens(&text, max_output_tokens);
        PtyExecUpdate {
            process_id: (!finished).then_some(process.id),
            output: text.into_bytes(),
            exit_code: *process.exit_code.lock().expect("poisoned"),
            original_token_count,
        }
    }
}

// -------------------------------------------------------------------------------------------
// Cloud bucket mounts
// -------------------------------------------------------------------------------------------

/// Mounts rclone-backed cloud storage inside a Daytona sandbox. Mirrors
/// `agents.extensions.sandbox.daytona.mounts.DaytonaCloudBucketMountStrategy`.
///
/// Activation verifies FUSE support, installs `fuse3` and `rclone` when the image lacks them, and
/// then delegates to [`InContainerMountStrategy`]. The rclone configuration carries the bucket
/// credentials, so the mount reports [`MountCredentialAuthority::MountScoped`] and the manifest
/// has to acknowledge the exact mount path before it activates.
#[derive(Debug, Clone)]
pub struct DaytonaCloudBucketMountStrategy {
    /// The rclone pattern that attaches the bucket.
    pub pattern: RcloneMountPattern,
    /// The secret-bearing rclone configuration and the remote selection options.
    pub config: MountPatternConfig,
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
            config: MountPatternConfig::new()
                .with_config_text(config_text)
                .with_option(rclone::OPTION_REMOTE_NAME, remote_name)
                .with_option(rclone::OPTION_REMOTE_PATH, remote_path),
        }
    }

    /// Replace the rclone pattern.
    pub fn with_pattern(mut self, pattern: RcloneMountPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// A [`Mount`] using this strategy.
    pub fn mount(self) -> Mount {
        Mount::new(Arc::new(self))
    }

    fn delegate(&self, mount: &Mount) -> InContainerMountStrategy {
        let mut config = self.config.clone();
        config.options.insert(
            rclone::OPTION_READ_ONLY.to_string(),
            mount.read_only.to_string(),
        );
        InContainerMountStrategy::new(Arc::new(self.pattern.clone()), config)
    }

    async fn preflight(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        if self.pattern.mode == RcloneMountMode::Fuse {
            ensure_fuse_support(session, self.pattern.install_if_missing).await?;
        }
        ensure_rclone(session, self.pattern.install_if_missing).await
    }
}

#[async_trait]
impl MountStrategy for DaytonaCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "daytona_cloud_bucket"
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        if self.config.has_config_text() {
            MountCredentialAuthority::MountScoped
        } else {
            MountCredentialAuthority::None
        }
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
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        self.delegate(mount)
            .activate(mount, session, mount_path)
            .await
    }

    async fn deactivate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.delegate(mount)
            .deactivate(mount, session, mount_path)
            .await
    }

    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        self.delegate(mount)
            .activate(mount, session, mount_path)
            .await
    }
}

// -------------------------------------------------------------------------------------------
// Interactive process bookkeeping
// -------------------------------------------------------------------------------------------

/// One asynchronous toolbox command the model drives interactively.
struct PtyProcess {
    id: u32,
    toolbox_session: String,
    command_id: String,
    delivered: Mutex<usize>,
    exit_code: Mutex<Option<i32>>,
    notify: Notify,
    last_used: Mutex<Instant>,
}

impl PtyProcess {
    fn touch(&self) {
        *self.last_used.lock().expect("poisoned") = Instant::now();
    }

    /// The part of `log` the model has not seen yet.
    fn advance(&self, log: &[u8]) -> Vec<u8> {
        let mut delivered = self.delivered.lock().expect("poisoned");
        if log.len() <= *delivered {
            return Vec::new();
        }
        let fresh = log[*delivered..].to_vec();
        *delivered = log.len();
        fresh
    }
}

/// The live interactive commands of one session.
#[derive(Default)]
struct PtyRegistry {
    processes: Mutex<HashMap<u32, Arc<PtyProcess>>>,
}

impl PtyRegistry {
    fn insert(
        &self,
        toolbox_session: String,
        command_id: String,
    ) -> SandboxResult<Arc<PtyProcess>> {
        let mut processes = self.processes.lock().expect("poisoned");
        if processes.len() >= PTY_PROCESSES_MAX {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("at most {PTY_PROCESSES_MAX} interactive processes may run per session"),
            ));
        }
        let mut id = None;
        for _ in 0..1024 {
            let span = PTY_PROCESS_ID_MAX_EXCLUSIVE - PTY_PROCESS_ID_MIN;
            let candidate =
                PTY_PROCESS_ID_MIN + (uuid::Uuid::new_v4().as_u128() % span as u128) as u32;
            if !processes.contains_key(&candidate) {
                id = Some(candidate);
                break;
            }
        }
        let id = id.ok_or_else(|| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                "could not allocate a free PTY session id",
            )
        })?;
        let process = Arc::new(PtyProcess {
            id,
            toolbox_session,
            command_id,
            delivered: Mutex::new(0),
            exit_code: Mutex::new(None),
            notify: Notify::new(),
            last_used: Mutex::new(Instant::now()),
        });
        processes.insert(id, process.clone());
        Ok(process)
    }

    fn get(&self, id: u32) -> SandboxResult<Arc<PtyProcess>> {
        self.processes
            .lock()
            .expect("poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::PtySessionNotFound,
                    format!("PTY session `{id}` is not running"),
                )
            })
    }

    fn remove(&self, id: u32) {
        self.processes.lock().expect("poisoned").remove(&id);
    }

    fn terminate_all(&self) -> Vec<Arc<PtyProcess>> {
        let mut processes = self.processes.lock().expect("poisoned");
        let live = processes.values().cloned().collect();
        for process in processes.values() {
            process.notify.notify_waiters();
        }
        processes.clear();
        live
    }
}

// -------------------------------------------------------------------------------------------
// Shared helpers
// -------------------------------------------------------------------------------------------

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
        );
        assert_eq!(
            command,
            "cd /home/daytona/workspace && env -- 'FOO=bar baz' echo 'hi there'"
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
