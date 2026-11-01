//! The E2B sandbox backend (<https://e2b.dev>). Port of `agents.extensions.sandbox.e2b`.
//!
//! # Design
//!
//! Python wraps the `e2b` / `e2b-code-interpreter` SDKs. This port talks to the two E2B HTTP
//! surfaces directly with `reqwest`, so it needs no extra dependency:
//!
//! - The **platform API** (`https://api.<domain>`, authenticated with `X-API-Key`) creates,
//!   inspects, pauses, resumes, snapshots and kills sandboxes.
//! - **envd**, the agent running inside each sandbox, is reached through the shared sandbox host
//!   (`https://sandbox.<domain>`). Every request carries `E2b-Sandbox-Id`, `E2b-Sandbox-Port`
//!   and the per-sandbox `X-Access-Token` returned by create, connect and resume.
//!
//! envd exposes `GET`/`POST /files` for file transfer and a Connect RPC surface for everything
//! else. Unary calls (`filesystem.Filesystem/*`, `process.Process/SendInput`, `SendSignal`) are
//! plain JSON `POST`s; `process.Process/Start` is a server-streaming call, so its body and
//! response use Connect's enveloped framing (one flag byte, a big-endian length, then JSON).
//! [`exec`](E2BSandboxSession::exec) buffers that stream, while
//! [`pty_exec_start`](E2BSandboxSession::pty_exec_start) keeps it open and hands the model the
//! output collected inside each yield window, which is how E2B supports a real PTY.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openai_agents_sdk::extensions::sandbox::e2b::{E2BSandboxClient, E2BSandboxClientOptions};
//! use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads `E2B_API_KEY` from the environment, as the Python SDK does.
//! let client = E2BSandboxClient::with_options(E2BSandboxClientOptions::new());
//! let session = client.create(Some(Manifest::new())).await?;
//! session.start().await?;
//! let result = session
//!     .exec(openai_agents_sdk::sandbox::ExecRequest::shell("echo hi"))
//!     .await?;
//! println!("{}", result.stdout_text());
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Divergences from Python
//!
//! - `sandbox_type` selects the default template name only; both E2B SDK classes speak the same
//!   HTTP API, so there is no separate code-interpreter transport.
//! - [`SandboxSessionState`] stays credential-free: the sandbox access token is never persisted,
//!   and [`resume`](E2BSandboxClient::resume) fetches a fresh one from
//!   `POST /v2/sandboxes/{id}/connect`. Provider options come from the client on resume, not from
//!   the persisted state.
//! - MCP server configuration, sandbox metrics and exposed-port endpoint resolution are not
//!   ported.

pub mod rclone;

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::StreamExt as _;
use serde::Deserialize;
use serde_json::{Value, json};

use super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, assert_exposed_port_configured,
    endpoint_from_authority, exposed_port_unavailable,
};
use super::{is_secret_option_name, persistable_identifier, redact_string_map, strip_url_secrets};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::http::{
    as_archive_read, bounded_read_zero, expect_success, http_error, parse_json, read_body_prefix,
    read_status_error, status_is_retryable, transport,
};
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

use self::rclone::{ensure_fuse_support, ensure_rclone, rclone_pattern_for_session};

/// The backend identifier reported by this client and its sessions.
pub const BACKEND_ID: &str = "e2b";
/// Environment variable holding the E2B API key, as in the Python SDK.
pub const E2B_API_KEY_ENV: &str = "E2B_API_KEY";
/// Environment variable overriding the E2B domain, as in the Python SDK.
pub const E2B_DOMAIN_ENV: &str = "E2B_DOMAIN";
/// The default E2B domain.
pub const DEFAULT_E2B_DOMAIN: &str = "e2b.app";
/// The port envd listens on inside the sandbox.
pub const DEFAULT_ENVD_PORT: u16 = 49983;
/// Default template of the plain `e2b` sandbox type.
pub const DEFAULT_E2B_TEMPLATE: &str = "base";
/// Default template of the `e2b_code_interpreter` sandbox type.
pub const DEFAULT_CODE_INTERPRETER_TEMPLATE: &str = "code-interpreter-v1";

/// Marks a workspace archive that only references a native E2B sandbox snapshot.
const SNAPSHOT_REF_MAGIC: &[u8] = b"E2B_SANDBOX_SNAPSHOT_V1\n";

/// Which E2B sandbox interface a client creates.
///
/// Both interfaces speak the same HTTP API; the variant selects the default template.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum E2BSandboxType {
    /// The plain `e2b` sandbox.
    #[default]
    E2B,
    /// The `e2b-code-interpreter` sandbox.
    CodeInterpreter,
}

impl E2BSandboxType {
    /// The wire name of the sandbox type.
    pub fn as_str(self) -> &'static str {
        match self {
            E2BSandboxType::E2B => "e2b",
            E2BSandboxType::CodeInterpreter => "e2b_code_interpreter",
        }
    }

    /// The template used when the options name none.
    pub fn default_template(self) -> &'static str {
        match self {
            E2BSandboxType::E2B => DEFAULT_E2B_TEMPLATE,
            E2BSandboxType::CodeInterpreter => DEFAULT_CODE_INTERPRETER_TEMPLATE,
        }
    }
}

/// What E2B does when a sandbox reaches its time-to-live.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum E2BTimeoutAction {
    /// Pause the sandbox, keeping its filesystem and memory.
    #[default]
    Pause,
    /// Kill the sandbox.
    Kill,
}

/// How a session persists its workspace into a [`Snapshot`].
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspacePersistence {
    /// Archive the workspace with `tar`, so the bytes are portable across backends.
    #[default]
    Tar,
    /// Take a native E2B sandbox snapshot and store only a reference to it.
    Snapshot,
}

/// Timeouts applied to E2B operations. Mirrors Python's `E2BSandboxTimeouts`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct E2BSandboxTimeouts {
    /// Sent to envd when a command has no timeout of its own; E2B would otherwise default to 60s.
    pub exec_unbounded: Duration,
    /// Budget for liveness checks.
    pub keepalive: Duration,
    /// Budget for best-effort cleanup, such as removing a temporary archive.
    pub cleanup: Duration,
    /// Budget for small filesystem operations.
    pub fast_op: Duration,
    /// Budget for file uploads.
    pub file_upload: Duration,
    /// Budget for workspace archive operations.
    pub snapshot_tar: Duration,
}

impl Default for E2BSandboxTimeouts {
    fn default() -> Self {
        E2BSandboxTimeouts {
            exec_unbounded: Duration::from_secs(24 * 60 * 60),
            keepalive: Duration::from_secs(5),
            cleanup: Duration::from_secs(30),
            fast_op: Duration::from_secs(10),
            file_upload: Duration::from_secs(30),
            snapshot_tar: Duration::from_secs(60),
        }
    }
}

/// Settings used when creating an E2B-backed session.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct E2BSandboxClientOptions {
    /// API key for the platform API. Falls back to `E2B_API_KEY`.
    pub api_key: Option<String>,
    /// E2B domain. Falls back to `E2B_DOMAIN`, then [`DEFAULT_E2B_DOMAIN`].
    pub domain: Option<String>,
    /// Override of the platform API base URL, for tests and self-hosted clusters.
    pub api_base_url: Option<String>,
    /// Override of the shared sandbox host base URL, for tests and self-hosted clusters.
    pub sandbox_base_url: Option<String>,
    /// The port envd listens on inside the sandbox.
    pub envd_port: u16,
    /// Which E2B sandbox interface to create.
    pub sandbox_type: E2BSandboxType,
    /// Template the sandbox is created from.
    pub template: Option<String>,
    /// Sandbox time-to-live in seconds.
    pub timeout_secs: Option<u32>,
    /// Metadata attached to the sandbox.
    pub metadata: BTreeMap<String, String>,
    /// Environment variables set when the sandbox is created.
    pub envs: BTreeMap<String, String>,
    /// Whether the sandbox may reach the internet.
    pub allow_internet_access: Option<bool>,
    /// Ports the sandbox exposes; a non-empty list enables public traffic.
    pub exposed_ports: Vec<u16>,
    /// MCP servers E2B starts inside the sandbox, keyed by server name.
    ///
    /// Travels verbatim as the `NewSandbox.mcp` object, which the API types as a free-form
    /// mapping. Mirrors Python's `mcp: dict[str, dict[str, str]]`.
    pub mcp: BTreeMap<String, BTreeMap<String, String>>,
    /// Operation timeouts.
    pub timeouts: E2BSandboxTimeouts,
    /// Pause instead of killing the sandbox when the session shuts down.
    pub pause_on_exit: bool,
    /// What E2B does when the sandbox's time-to-live elapses.
    pub on_timeout: E2BTimeoutAction,
    /// Whether incoming traffic resumes a paused sandbox.
    pub auto_resume: bool,
    /// How the session persists its workspace.
    pub workspace_persistence: WorkspacePersistence,
}

impl Default for E2BSandboxClientOptions {
    fn default() -> Self {
        E2BSandboxClientOptions {
            api_key: None,
            domain: None,
            api_base_url: None,
            sandbox_base_url: None,
            envd_port: DEFAULT_ENVD_PORT,
            sandbox_type: E2BSandboxType::default(),
            template: None,
            timeout_secs: None,
            metadata: BTreeMap::new(),
            envs: BTreeMap::new(),
            allow_internet_access: None,
            exposed_ports: Vec::new(),
            mcp: BTreeMap::new(),
            timeouts: E2BSandboxTimeouts::default(),
            pause_on_exit: false,
            on_timeout: E2BTimeoutAction::default(),
            auto_resume: true,
            workspace_persistence: WorkspacePersistence::default(),
        }
    }
}

// The API key must never reach a log line through the `SandboxClient: Debug` bound.
impl std::fmt::Debug for E2BSandboxClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E2BSandboxClientOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("domain", &self.domain)
            .field("api_base_url", &self.api_base_url)
            .field("sandbox_base_url", &self.sandbox_base_url)
            .field("envd_port", &self.envd_port)
            .field("sandbox_type", &self.sandbox_type)
            .field("template", &self.template)
            .field("timeout_secs", &self.timeout_secs)
            .field("metadata", &self.metadata)
            .field("envs", &self.envs.keys().collect::<Vec<_>>())
            .field("allow_internet_access", &self.allow_internet_access)
            .field("exposed_ports", &self.exposed_ports)
            .field("mcp", &self.mcp.keys().collect::<Vec<_>>())
            .field("timeouts", &self.timeouts)
            .field("pause_on_exit", &self.pause_on_exit)
            .field("on_timeout", &self.on_timeout)
            .field("auto_resume", &self.auto_resume)
            .field("workspace_persistence", &self.workspace_persistence)
            .finish()
    }
}

impl E2BSandboxClientOptions {
    /// Options that read the API key and domain from the environment.
    pub fn new() -> Self {
        E2BSandboxClientOptions::default()
    }

    /// Use an explicit API key instead of `E2B_API_KEY`.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use an explicit domain instead of `E2B_DOMAIN`.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Point both the platform API and the sandbox host at one base URL.
    ///
    /// Tests use this to drive the client against a local HTTP server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.api_base_url = Some(base_url.clone());
        self.sandbox_base_url = Some(base_url);
        self
    }

    /// Override only the platform API base URL.
    pub fn with_api_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.api_base_url = Some(base_url.into());
        self
    }

    /// Override only the shared sandbox host base URL.
    pub fn with_sandbox_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.sandbox_base_url = Some(base_url.into());
        self
    }

    /// Select the sandbox interface, and therefore the default template.
    pub fn with_sandbox_type(mut self, sandbox_type: E2BSandboxType) -> Self {
        self.sandbox_type = sandbox_type;
        self
    }

    /// Create the sandbox from an explicit template.
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Set the sandbox time-to-live in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u32) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Attach a metadata key.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set an environment variable at sandbox creation.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.envs.insert(key.into(), value.into());
        self
    }

    /// Allow or deny internet access from inside the sandbox.
    pub fn with_allow_internet_access(mut self, allow: bool) -> Self {
        self.allow_internet_access = Some(allow);
        self
    }

    /// Expose a port, which also enables public traffic to the sandbox.
    pub fn with_exposed_port(mut self, port: u16) -> Self {
        self.exposed_ports.push(port);
        self
    }

    /// Configure one MCP server the sandbox starts, keyed by name.
    ///
    /// Mirrors the `mcp=` argument of Python's `E2BSandboxClientOptions`; the mapping reaches
    /// the API as `NewSandbox.mcp` unchanged.
    pub fn with_mcp_server<I, K, V>(mut self, name: impl Into<String>, config: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.mcp.insert(
            name.into(),
            config
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        );
        self
    }

    /// Replace the operation timeouts.
    pub fn with_timeouts(mut self, timeouts: E2BSandboxTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Pause instead of killing the sandbox on shutdown.
    pub fn with_pause_on_exit(mut self, pause_on_exit: bool) -> Self {
        self.pause_on_exit = pause_on_exit;
        self
    }

    /// Set what E2B does when the sandbox's time-to-live elapses.
    pub fn with_on_timeout(mut self, on_timeout: E2BTimeoutAction) -> Self {
        self.on_timeout = on_timeout;
        self
    }

    /// Whether traffic resumes a paused sandbox.
    pub fn with_auto_resume(mut self, auto_resume: bool) -> Self {
        self.auto_resume = auto_resume;
        self
    }

    /// Select how the session persists its workspace.
    pub fn with_workspace_persistence(mut self, persistence: WorkspacePersistence) -> Self {
        self.workspace_persistence = persistence;
        self
    }

    /// The resolved platform API base URL.
    fn api_base(&self) -> String {
        match &self.api_base_url {
            Some(url) => url.trim_end_matches('/').to_string(),
            None => format!("https://api.{}", self.resolved_domain()),
        }
    }

    /// The resolved shared sandbox host base URL.
    fn sandbox_base(&self) -> String {
        match &self.sandbox_base_url {
            Some(url) => url.trim_end_matches('/').to_string(),
            None => format!("https://sandbox.{}", self.resolved_domain()),
        }
    }

    fn resolved_domain(&self) -> String {
        self.domain
            .clone()
            .or_else(|| std::env::var(E2B_DOMAIN_ENV).ok().filter(|v| !v.is_empty()))
            .unwrap_or_else(|| DEFAULT_E2B_DOMAIN.to_string())
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
        std::env::var(E2B_API_KEY_ENV)
            .ok()
            .filter(|key| !key.is_empty())
            .ok_or_else(missing_api_key)
    }

    /// The template the sandbox is created from.
    fn resolved_template(&self) -> String {
        self.template
            .clone()
            .unwrap_or_else(|| self.sandbox_type.default_template().to_string())
    }
}

/// Adopt the workspace root recorded in `state` when the manifest never chose one.
///
/// A provider may rewrite the SDK-wide default root at create time, so the digest recorded in the
/// state covers the rewritten root. Verifying a manifest that still carries the default against
/// that digest would always fail. Any other root is left alone, so a manifest that *did* choose a
/// root must still match what the session was created from.
pub(crate) fn adopt_recorded_root(manifest: Manifest, state: &SandboxSessionState) -> Manifest {
    if manifest.root() == crate::sandbox::manifest::DEFAULT_MANIFEST_ROOT {
        manifest.with_root(state.workspace_root.clone())
    } else {
        manifest
    }
}

/// The error raised when no usable E2B API key is configured.
fn missing_api_key() -> SandboxError {
    SandboxError::new(
        ErrorCode::SandboxConfigInvalid,
        format!(
            "the E2B sandbox backend needs an API key: set `{E2B_API_KEY_ENV}` or \
             `E2BSandboxClientOptions::with_api_key`"
        ),
    )
}

/// The E2B options a resumed session is rebuilt from.
///
/// Mirrors the `mcp`, `template`, `metadata`, `base_envs` and `timeouts` fields of Python's
/// `E2BSandboxSessionState`, with one deliberate
/// difference: the persisted state stays credential-free, so only the non-secret shape of the
/// MCP config is recorded. Server names are kept; an option whose name looks secret (`apiKey`,
/// `token`, `secret`, `password`, `authorization`, ...) is dropped, and a URL loses its userinfo,
/// query and fragment. The resuming client supplies the secrets from its own options: its
/// config for a server wins key by key over the recorded shape. A payload without the field
/// reads as no MCP servers.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(default)]
pub struct E2BPersistedOptions {
    /// The non-secret shape of the MCP servers E2B starts inside the sandbox, by server name.
    pub mcp: BTreeMap<String, BTreeMap<String, String>>,
    /// The template the sandbox was created from, so a replacement uses the same one. Absent
    /// when it looked like a credential.
    pub template: Option<String>,
    /// Sandbox metadata, redacted the same way as the MCP options.
    pub metadata: BTreeMap<String, String>,
    /// Environment variables set at creation (Python's `base_envs`), redacted: entries whose
    /// name or value looks secret are not recorded.
    pub envs: BTreeMap<String, String>,
    /// The operation timeouts the session ran with. Absent in a payload that predates them.
    pub timeouts: Option<E2BSandboxTimeouts>,
}

impl std::fmt::Debug for E2BPersistedOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E2BPersistedOptions")
            .field("mcp", &self.mcp.keys().collect::<Vec<_>>())
            .field("template", &self.template)
            .field("metadata", &self.metadata.keys().collect::<Vec<_>>())
            .field("envs", &self.envs.keys().collect::<Vec<_>>())
            .field("timeouts", &self.timeouts)
            .finish()
    }
}

impl E2BPersistedOptions {
    /// The persisted, credential-free subset of `options`.
    pub fn from_options(options: &E2BSandboxClientOptions) -> Self {
        E2BPersistedOptions {
            mcp: options
                .mcp
                .iter()
                .map(|(server, config)| {
                    let shape = config
                        .iter()
                        .filter(|(name, _)| !is_secret_option_name(name))
                        .map(|(name, value)| (name.clone(), strip_url_secrets(value)))
                        .collect();
                    (server.clone(), shape)
                })
                .collect(),
            template: persistable_identifier(&options.resolved_template()),
            metadata: redact_string_map(&options.metadata),
            envs: redact_string_map(&options.envs),
            timeouts: Some(options.timeouts.clone()),
        }
    }

    /// Layer the recorded options under the client's own.
    ///
    /// The MCP config, metadata and environment merge key by key with the client's value
    /// winning (the client holds the secrets the record left out), and a server or key only the
    /// client knows is kept. The template and timeouts are the original session's: a
    /// replacement sandbox is created from them.
    pub fn apply_to(self, options: &mut E2BSandboxClientOptions) {
        fn layer<V>(recorded: BTreeMap<String, V>, own: &mut BTreeMap<String, V>) {
            let mut merged = recorded;
            merged.extend(std::mem::take(own));
            *own = merged;
        }
        let mut merged = self.mcp;
        for (server, config) in std::mem::take(&mut options.mcp) {
            merged.entry(server).or_default().extend(config);
        }
        options.mcp = merged;
        layer(self.metadata, &mut options.metadata);
        layer(self.envs, &mut options.envs);
        if self.template.is_some() {
            options.template = self.template;
        }
        if let Some(timeouts) = self.timeouts {
            options.timeouts = timeouts;
        }
    }

    /// The options persisted in `state`, when it carries any.
    pub fn from_state(state: &SandboxSessionState) -> Option<Self> {
        state.provider_options()
    }
}

/// Creates [`E2BSandboxSession`]s against the E2B HTTP API.
#[derive(Debug, Clone)]
pub struct E2BSandboxClient {
    options: E2BSandboxClientOptions,
    http: reqwest::Client,
    snapshot: Option<SnapshotSpec>,
}

impl Default for E2BSandboxClient {
    fn default() -> Self {
        E2BSandboxClient::with_options(E2BSandboxClientOptions::default())
    }
}

/// The HTTP client a client builds for itself.
///
/// The E2B SDK gives every request a 60 second timeout, so a response that stalls surfaces as a
/// retryable `TimeoutException` instead of holding the caller forever. The same bound applies
/// here as an idle limit between reads, which leaves a large download that keeps flowing alone.
fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(60))
        .read_timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_default()
}

impl E2BSandboxClient {
    /// This client with the options `state` recorded layered over its own.
    ///
    /// A state written before options were recorded leaves the client's options in place.
    fn with_persisted_options(&self, state: &SandboxSessionState) -> Self {
        let mut client = self.clone();
        if let Some(persisted) = E2BPersistedOptions::from_state(state) {
            persisted.apply_to(&mut client.options);
        }
        client
    }

    /// A client reading its credentials from the environment.
    pub fn new() -> Self {
        E2BSandboxClient::default()
    }

    /// A client using explicit options.
    pub fn with_options(options: E2BSandboxClientOptions) -> Self {
        E2BSandboxClient {
            options,
            http: default_http_client(),
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
    pub fn options(&self) -> &E2BSandboxClientOptions {
        &self.options
    }

    /// Create a sandbox from `template` and return its id and access token.
    async fn create_sandbox(
        &self,
        api_key: &str,
        template: &str,
        envs: &BTreeMap<String, String>,
    ) -> SandboxResult<CreatedSandbox> {
        let mut body = json!({
            "templateID": template,
            "autoPause": self.options.on_timeout == E2BTimeoutAction::Pause,
        });
        let map = body.as_object_mut().expect("object");
        if let Some(timeout) = self.options.timeout_secs {
            map.insert("timeout".into(), json!(timeout));
        }
        if !self.options.metadata.is_empty() {
            map.insert("metadata".into(), json!(self.options.metadata));
        }
        if !envs.is_empty() {
            map.insert("envVars".into(), json!(envs));
        }
        if let Some(allow) = self.options.allow_internet_access {
            map.insert("allow_internet_access".into(), json!(allow));
        }
        if !self.options.exposed_ports.is_empty() {
            map.insert("network".into(), json!({ "allowPublicTraffic": true }));
        }
        if !self.options.mcp.is_empty() {
            map.insert("mcp".into(), json!(self.options.mcp));
        }
        if self.options.on_timeout == E2BTimeoutAction::Pause {
            map.insert(
                "autoResume".into(),
                json!({ "enabled": self.options.auto_resume }),
            );
        }
        let response = self
            .http
            .post(format!("{}/v2/sandboxes", self.options.api_base()))
            .header("X-API-Key", api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| transport("create an E2B sandbox", e))?;
        let created: CreatedSandbox = parse_json(response, "create an E2B sandbox").await?;
        Ok(created)
    }
}

/// A sandbox as the platform API reports it on create, connect and resume.
#[derive(Debug, Clone, Deserialize)]
struct CreatedSandbox {
    #[serde(rename = "sandboxID")]
    sandbox_id: String,
    #[serde(rename = "envdAccessToken", default)]
    envd_access_token: Option<String>,
}

#[async_trait]
impl SandboxClient for E2BSandboxClient {
    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        let manifest = manifest.unwrap_or_default();
        let api_key = self.options.resolved_api_key()?;
        // Manifest environment variables win over the client's, as in Python.
        let mut envs = self.options.envs.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        let created = self
            .create_sandbox(&api_key, &self.options.resolved_template(), &envs)
            .await?;
        Ok(Arc::new(E2BSandboxSession::new(
            self.options.clone(),
            api_key,
            self.http.clone(),
            created,
            manifest,
            self.snapshot.clone(),
        )))
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        // A replacement sandbox is built from what the state recorded, not from the options
        // of the client that happens to resume it.
        let this = &self.with_persisted_options(state);
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
                "E2B sandbox session state is missing its sandbox id",
            )
        })?;
        let api_key = this.options.resolved_api_key()?;
        let mut envs = this.options.envs.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }

        // Reattaching returns a fresh access token; the persisted state never carries one.
        let connected = self
            .http
            .post(format!(
                "{}/v2/sandboxes/{sandbox_id}/connect",
                this.options.api_base()
            ))
            .header("X-API-Key", &api_key)
            // Reconnecting restarts the sandbox's lifetime clock, so the configured timeout
            // goes with it.
            .json(&match this.options.timeout_secs {
                Some(timeout) => json!({ "timeout": timeout }),
                None => json!({}),
            })
            .send()
            .await
            .ok()
            .filter(|response| response.status().is_success());
        let created = match connected {
            Some(response) => Some(parse_json::<CreatedSandbox>(response, "connect").await?),
            None => None,
        };
        let reattached = created.is_some();
        let (created, resuming) = match created {
            Some(created) => (created, true),
            // The original sandbox is gone: create a replacement and hydrate it from the snapshot.
            None => (
                this.create_sandbox(&api_key, &this.options.resolved_template(), &envs)
                    .await?,
                true,
            ),
        };
        let session = E2BSandboxSession::new(
            this.options.clone(),
            api_key,
            this.http.clone(),
            created,
            manifest,
            Some(state.snapshot.clone()),
        )
        .with_session_id(state.session_id.clone())
        .with_snapshot_id(state.snapshot_id.clone());
        let session = session
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
            });
        Ok(Arc::new(if resuming {
            session.resuming()
        } else {
            session
        }))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// The mutable provider identity of a session: E2B replaces both when a snapshot is restored.
#[derive(Debug, Clone)]
struct RemoteHandle {
    sandbox_id: String,
    access_token: Option<String>,
}

/// A sandbox session whose workspace lives in an E2B sandbox.
pub struct E2BSandboxSession {
    session_id: String,
    manifest: Manifest,
    options: E2BSandboxClientOptions,
    api_key: String,
    api_base: String,
    sandbox_base: String,
    http: reqwest::Client,
    remote: RwLock<RemoteHandle>,
    started: AtomicBool,
    /// `start` already created the workspace root, so the manifest applier's root `mkdir` that
    /// follows it must not go through the filesystem API a second time.
    skip_next_root_mkdir: AtomicBool,
    paused: AtomicBool,
    released: tokio::sync::Mutex<bool>,
    ptys: RemotePtyManager,
    /// Sandbox-side pid of each live PTY session, for `SendInput` and `SendSignal`.
    pty_pids: Arc<Mutex<HashMap<u32, u32>>>,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
    /// The sandbox survived the pause this session resumes from, so its accounts and packages
    /// are still there.
    state_preserved: bool,
    /// Workspace-relative paths a persist leaves out of the archive.
    persist_skips: Mutex<std::collections::BTreeSet<String>>,
    /// The workspace fingerprint the stored snapshot was taken from.
    snapshot_fingerprint: Mutex<Option<SnapshotFingerprint>>,
}

// Neither the API key nor the sandbox access token may appear in a log line.
impl std::fmt::Debug for E2BSandboxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E2BSandboxSession")
            .field("session_id", &self.session_id)
            .field("sandbox_id", &self.sandbox_id())
            .field("workspace_root", &self.manifest.root())
            .field("resuming", &self.resuming)
            .finish_non_exhaustive()
    }
}

impl E2BSandboxSession {
    fn new(
        options: E2BSandboxClientOptions,
        api_key: String,
        http: reqwest::Client,
        created: CreatedSandbox,
        manifest: Manifest,
        snapshot: Option<SnapshotSpec>,
    ) -> Self {
        let session_id = uuid::Uuid::new_v4().to_string();
        E2BSandboxSession {
            api_base: options.api_base(),
            sandbox_base: options.sandbox_base(),
            options,
            api_key,
            http,
            remote: RwLock::new(RemoteHandle {
                sandbox_id: created.sandbox_id,
                access_token: created.envd_access_token,
            }),
            manifest,
            snapshot_id: session_id.clone(),
            session_id,
            started: AtomicBool::new(false),
            skip_next_root_mkdir: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            released: tokio::sync::Mutex::new(false),
            ptys: RemotePtyManager::new(),
            pty_pids: Arc::new(Mutex::new(HashMap::new())),
            snapshot,
            resuming: false,
            state_preserved: false,
            persist_skips: Mutex::new(std::collections::BTreeSet::new()),
            snapshot_fingerprint: Mutex::new(None),
        }
    }

    /// Reuse a persisted session identity.
    fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = session_id;
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

    /// The E2B sandbox currently backing this session.
    pub fn sandbox_id(&self) -> String {
        self.remote.read().expect("poisoned").sandbox_id.clone()
    }

    /// Attach the platform authentication to a request.
    fn platform(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header("X-API-Key", &self.api_key)
    }

    /// Attach the envd routing headers and the sandbox access token to a request.
    fn envd(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let handle = self.remote.read().expect("poisoned");
        let mut builder = builder
            .header("E2b-Sandbox-Id", &handle.sandbox_id)
            .header("E2b-Sandbox-Port", self.options.envd_port.to_string());
        if let Some(token) = &handle.access_token {
            builder = builder.header("X-Access-Token", token);
        }
        builder
    }

    /// Call a unary Connect RPC on envd.
    async fn connect_unary(&self, method: &str, body: Value, what: &str) -> SandboxResult<Value> {
        self.connect_unary_tolerating(method, body, what, &[]).await
    }

    /// Call a unary Connect RPC, accepting the listed Connect error codes as success.
    ///
    /// envd answers `MakeDir` on an existing directory with `already_exists` and `Remove` on a
    /// missing path with `not_found`. The vendor SDK swallows both, and the sandbox session
    /// contract expects `mkdir -p` and `rm -f` semantics, so they are not failures here either.
    async fn connect_unary_tolerating(
        &self,
        method: &str,
        body: Value,
        what: &str,
        tolerated: &[&str],
    ) -> SandboxResult<Value> {
        let response = self
            .envd(self.http.post(format!("{}/{method}", self.sandbox_base)))
            .header("Connect-Protocol-Version", "1")
            .json(&body)
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        let status = response.status();
        let text = response.text().await.map_err(|e| transport(what, e))?;
        if status.is_success() {
            return serde_json::from_str(&text).or(Ok(Value::Null));
        }
        let code = serde_json::from_str::<Value>(&text).ok().and_then(|value| {
            value
                .get("code")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        if let Some(code) = &code
            && tolerated.contains(&code.as_str())
        {
            return Ok(Value::Null);
        }
        Err(http_error(
            ErrorCode::ExecTransportError,
            what,
            status,
            &text,
        ))
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
        let mut envs = self.options.envs.clone();
        for (key, value) in self.manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        envs
    }

    /// The timeout envd is told to apply, honouring the "no timeout" sentinel.
    fn envd_timeout(&self, timeout: Option<Duration>) -> Duration {
        match timeout {
            None => self.options.timeouts.exec_unbounded,
            Some(timeout) if timeout.is_zero() => Duration::from_secs(1),
            Some(timeout) => timeout,
        }
    }

    /// Build a `process.StartRequest` for `argv`.
    fn start_request(
        &self,
        argv: &[String],
        cwd: Option<String>,
        env: &BTreeMap<String, String>,
        pty: Option<(u16, u16)>,
        stdin: bool,
    ) -> Value {
        let (cmd, args) = argv
            .split_first()
            .map(|(cmd, args)| (cmd.clone(), args.to_vec()))
            .unwrap_or_else(|| ("true".to_string(), Vec::new()));
        let mut process = json!({ "cmd": cmd, "args": args, "envs": env });
        if let Some(cwd) = cwd {
            process
                .as_object_mut()
                .expect("object")
                .insert("cwd".into(), json!(cwd));
        }
        let mut request = json!({ "process": process, "stdin": stdin });
        if let Some((rows, cols)) = pty {
            request.as_object_mut().expect("object").insert(
                "pty".into(),
                json!({ "size": { "rows": rows, "cols": cols } }),
            );
        }
        request
    }

    /// Resolve the working directory of a request, defaulting to the workspace root.
    fn resolve_cwd(&self, cwd: &Option<String>) -> SandboxResult<String> {
        match cwd {
            None => Ok(self.manifest.root().to_string()),
            Some(cwd) => self.normalize_path(cwd),
        }
    }

    /// Open `process.Process/Start` and return the framed response stream.
    async fn open_process_stream(
        &self,
        request: Value,
        user: Option<&str>,
        timeout: Duration,
        what: &str,
    ) -> SandboxResult<reqwest::Response> {
        let mut builder = self
            .envd(
                self.http
                    .post(format!("{}/process.Process/Start", self.sandbox_base)),
            )
            .header("Connect-Protocol-Version", "1")
            .header("Content-Type", "application/connect+json")
            .header("Connect-Timeout-Ms", timeout.as_millis().to_string())
            .body(encode_connect_frame(&request));
        if let Some(user) = user {
            // envd takes the system user through basic auth, with an empty password.
            builder = builder.basic_auth(user, None::<&str>);
        }
        let response = builder.send().await.map_err(|e| transport(what, e))?;
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
        Ok(response)
    }

    /// Run a command without the workspace-root default, used while the root is still missing.
    async fn exec_at(
        &self,
        argv: Vec<String>,
        cwd: Option<String>,
        env: &BTreeMap<String, String>,
        timeout: Option<Duration>,
        user: Option<String>,
    ) -> SandboxResult<ExecResult> {
        let what = "run a command in the E2B sandbox";
        let envd_timeout = self.envd_timeout(timeout);
        let request = self.start_request(&argv, cwd, env, None, false);
        let response = self
            .open_process_stream(request, user.as_deref(), envd_timeout, what)
            .await?;
        let collect = async {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut exit_code = None;
            let mut stream = response.bytes_stream();
            let mut decoder = ConnectFrameDecoder::default();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| transport(what, e))?;
                decoder.push(&chunk);
                for frame in decoder.drain() {
                    match frame {
                        ConnectFrame::Message(value) => {
                            apply_process_event(&value, &mut stdout, &mut stderr, &mut exit_code);
                        }
                        ConnectFrame::EndStream(value) => {
                            if let Some(error) = connect_stream_error(&value) {
                                return Err(SandboxError::new(
                                    ErrorCode::ExecTransportError,
                                    format!("E2B rejected the command: {error}"),
                                ));
                            }
                        }
                    }
                }
            }
            // envd always reports an end event; without one the stream was cut short and the
            // exit code is unknown, so reporting success would be a lie.
            let exit_code = exit_code.ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!(
                        "the E2B command stream for `{}` ended before the process finished",
                        argv.join(" ")
                    ),
                )
            })?;
            Ok(ExecResult {
                stdout,
                stderr,
                exit_code,
            })
        };
        match timeout {
            None => collect.await,
            Some(limit) => match tokio::time::timeout(limit, collect).await {
                Ok(result) => result,
                Err(_) => Err(SandboxError::new(
                    ErrorCode::ExecTimeout,
                    format!("Command `{}` timed out after {limit:?}", argv.join(" ")),
                )),
            },
        }
    }

    /// Create a directory with the filesystem API, which works before the root exists.
    async fn make_dir(&self, path: &str) -> SandboxResult<()> {
        self.connect_unary_tolerating(
            "filesystem.Filesystem/MakeDir",
            json!({ "path": path }),
            "create a directory in the E2B sandbox",
            &["already_exists"],
        )
        .await
        .map(|_| ())
    }

    /// The sandbox-side pid of one PTY session.
    fn pty_pid(&self, session_id: u32) -> Option<u32> {
        self.pty_pids
            .lock()
            .expect("poisoned")
            .get(&session_id)
            .copied()
    }

    /// Drop the bookkeeping of a session whose process has exited.
    fn forget_finished_pty(&self, update: &PtyExecUpdate, session_id: u32) {
        if update.process_id.is_none() {
            self.ptys.remove(session_id);
            self.pty_pids.lock().expect("poisoned").remove(&session_id);
        }
    }

    /// Kill the sandbox-side processes behind the given PTY sessions, best effort.
    async fn release_pty_processes(&self, processes: &[Arc<RemotePtyProcess>]) {
        let pids: Vec<(u32, Option<u32>)> = processes
            .iter()
            .map(|process| (process.id(), self.pty_pid(process.id())))
            .collect();
        for (session_id, pid) in pids {
            // Releasing a session the model can no longer reach must also stop its stream.
            if let Some(process) = processes.iter().find(|p| p.id() == session_id) {
                process.close(None);
            }
            self.pty_pids.lock().expect("poisoned").remove(&session_id);
            if let Some(pid) = pid {
                self.signal_pty_processes(&[pid]).await;
            }
        }
    }

    /// Kill the listed sandbox-side PTY processes, best effort.
    async fn signal_pty_processes(&self, pids: &[u32]) {
        for pid in pids {
            let _ = self
                .connect_unary(
                    "process.Process/SendSignal",
                    json!({ "process": { "pid": pid }, "signal": "SIGNAL_SIGKILL" }),
                    "terminate a PTY process in the E2B sandbox",
                )
                .await;
        }
    }

    /// Pause the sandbox, keeping its filesystem and memory.
    async fn pause(&self) -> SandboxResult<()> {
        let response = self
            .platform(self.http.post(format!(
                "{}/sandboxes/{}/pause",
                self.api_base,
                self.sandbox_id()
            )))
            .send()
            .await
            .map_err(|e| transport("pause the E2B sandbox", e))?;
        expect_success(
            response,
            "pause the E2B sandbox",
            ErrorCode::WorkspaceStopError,
        )
        .await
    }

    /// Kill the sandbox and release its resources.
    async fn kill(&self) -> SandboxResult<()> {
        let response = self
            .platform(self.http.delete(format!(
                "{}/sandboxes/{}",
                self.api_base,
                self.sandbox_id()
            )))
            .send()
            .await
            .map_err(|e| transport("kill the E2B sandbox", e))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(
            response,
            "kill the E2B sandbox",
            ErrorCode::WorkspaceStopError,
        )
        .await
    }

    /// Release the sandbox exactly once, pausing it when the options ask for that.
    async fn terminate(&self) -> SandboxResult<()> {
        let mut terminated = self.released.lock().await;
        if *terminated {
            return Ok(());
        }
        self.pty_terminate_all().await.ok();
        // As in Python, a failure to release the sandbox is logged rather than raised: shutdown
        // runs in cleanup paths where a second error would hide the first.
        if self.options.pause_on_exit {
            // A `stop` already paused it; pausing a paused sandbox is a conflict, and falling
            // back to a kill would destroy a sandbox the caller asked to keep.
            if !self.paused.load(Ordering::SeqCst)
                && let Err(error) = self.pause().await
            {
                // A pause that cannot be taken must not leak the sandbox.
                crate::logger::log_tool_action_warning(
                    "Failed to pause E2B sandbox on shutdown; falling back to kill.",
                    &error,
                );
                if let Err(error) = self.kill().await {
                    crate::logger::log_tool_action_warning(
                        "Failed to kill E2B sandbox after pause fallback failure.",
                        &error,
                    );
                }
            }
        } else if let Err(error) = self.kill().await {
            crate::logger::log_tool_action_warning(
                "Failed to kill E2B sandbox on shutdown.",
                &error,
            );
        }
        self.started.store(false, Ordering::SeqCst);
        *terminated = true;
        Ok(())
    }

    /// Replace the backing sandbox with one created from a native snapshot template.
    async fn restore_native_snapshot(&self, snapshot_id: &str) -> SandboxResult<()> {
        // Reuse the client's create path so the snapshot sandbox keeps the configured timeout,
        // metadata, network and lifecycle options instead of a hand-written subset.
        let client = E2BSandboxClient {
            options: self.options.clone(),
            http: self.http.clone(),
            snapshot: None,
        };
        let created = client
            .create_sandbox(&self.api_key, snapshot_id, &self.command_envs())
            .await?;
        // Killing the sandbox the snapshot replaces keeps the account from leaking sandboxes.
        let previous = {
            let mut handle = self.remote.write().expect("poisoned");
            let previous = handle.sandbox_id.clone();
            handle.sandbox_id = created.sandbox_id;
            handle.access_token = created.envd_access_token;
            previous
        };
        let _ = self
            .platform(
                self.http
                    .delete(format!("{}/sandboxes/{previous}", self.api_base)),
            )
            .send()
            .await;
        Ok(())
    }

    /// Take a native E2B snapshot of the whole sandbox.
    async fn create_native_snapshot(&self) -> SandboxResult<String> {
        let response = self
            .platform(self.http.post(format!(
                "{}/sandboxes/{}/snapshots",
                self.api_base,
                self.sandbox_id()
            )))
            .json(&json!({ "memory": false }))
            .send()
            .await
            .map_err(|e| transport("snapshot the E2B sandbox", e))?;
        #[derive(Deserialize)]
        struct SnapshotInfo {
            #[serde(rename = "snapshotID")]
            snapshot_id: String,
        }
        let info: SnapshotInfo = parse_json(response, "snapshot the E2B sandbox").await?;
        Ok(info.snapshot_id)
    }

    /// Download a file by its absolute sandbox path.
    ///
    /// Workspace reads go through [`read_file`](SandboxSession::read_file), which validates the
    /// path first; this is for the paths the session itself owns, such as a temporary archive
    /// under `/tmp`.
    async fn download_at(&self, path: &str, limit: Option<usize>) -> SandboxResult<Vec<u8>> {
        let what = "read a file from the E2B sandbox";
        let response = self
            .envd(self.http.get(format!("{}/files", self.sandbox_base)))
            .query(&[("path", path)])
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
                Some(status_is_retryable(status)),
            ));
        }
        read_body_prefix(response, limit, what).await
    }

    /// Upload a file by its absolute sandbox path.
    async fn upload_at(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let what = "write a file to the E2B sandbox";
        let response = self
            .envd(self.http.post(format!("{}/files", self.sandbox_base)))
            .query(&[("path", path)])
            .header("Content-Type", "application/octet-stream")
            .timeout(self.options.timeouts.file_upload)
            .body(contents.to_vec())
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    /// Remove a path by its absolute sandbox path.
    async fn remove_at(&self, path: &str) -> SandboxResult<()> {
        self.connect_unary_tolerating(
            "filesystem.Filesystem/Remove",
            json!({ "path": path }),
            "remove a path in the E2B sandbox",
            &["not_found"],
        )
        .await
        .map(|_| ())
    }

    /// Archive the workspace, excluding the manifest's ephemeral paths.
    async fn persist_workspace_tar(&self) -> SandboxResult<Vec<u8>> {
        let root = self.manifest.root().to_string();
        let mut command = format!("tar -czf - -C {}", shell_quote(&root));
        for exclude in workspace_archive_excludes(&self.manifest, &root) {
            command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
        }
        for skipped in self.persist_workspace_skip_paths() {
            command.push_str(&format!(
                " --exclude {}",
                shell_quote(&format!("./{skipped}"))
            ));
        }
        // envd reports command output as text, so the archive travels base64-encoded.
        command.push_str(" . | base64 -w0");
        let result = self
            .exec_at(
                vec!["sh".into(), "-lc".into(), command],
                Some(root.clone()),
                &self.command_envs(),
                Some(self.options.timeouts.snapshot_tar),
                None,
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
        BASE64.decode(result.stdout_text().trim()).map_err(|e| {
            SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!("E2B returned a workspace archive that is not valid base64: {e}"),
            )
        })
    }
}

#[async_trait]
impl SandboxSession for E2BSandboxSession {
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
        // The filesystem API can create roots the command user cannot, so it goes first when the
        // root is missing; an existing root needs no filesystem API call at all. The command
        // below then proves the root is usable as a working directory.
        let root_exists = self
            .exec_at(
                vec!["test".into(), "-d".into(), root.clone()],
                None,
                &self.command_envs(),
                Some(self.options.timeouts.fast_op),
                None,
            )
            .await
            .is_ok_and(|probe| probe.ok());
        if !root_exists {
            self.make_dir(&root).await.map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!(
                        "Failed to create the E2B workspace root `{root}`: {}",
                        error.message
                    ),
                )
            })?;
        }
        let prepared = self
            .exec_at(
                vec!["mkdir".into(), "-p".into(), "--".into(), root.clone()],
                Some("/".to_string()),
                &self.command_envs(),
                Some(self.options.timeouts.fast_op),
                None,
            )
            .await?;
        if !prepared.ok() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "Failed to prepare the E2B workspace root `{root}`: {}",
                    prepared.stderr_text().trim()
                ),
            ));
        }
        self.started.store(true, Ordering::SeqCst);
        self.skip_next_root_mkdir.store(true, Ordering::SeqCst);
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
            self.skip_next_root_mkdir.store(false, Ordering::SeqCst);
        })?;
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        // `stop` keeps what `shutdown` releases, so it pauses the sandbox: a paused E2B sandbox
        // keeps its filesystem and can be resumed by id.
        if *self.released.lock().await {
            return Ok(());
        }
        self.pty_terminate_all().await.ok();
        let result = self.pause().await;
        self.started.store(false, Ordering::SeqCst);
        if result.is_ok() {
            self.paused.store(true, Ordering::SeqCst);
        }
        result
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.terminate().await
    }

    async fn running(&self) -> SandboxResult<bool> {
        // A session whose workspace root `start` has not prepared is not running yet, whatever
        // the platform says about the sandbox behind it (Python's `workspace_root_ready` gate).
        // The runner relies on this to start a session a caller built but never started, and the
        // workspace event sink holds its buffer until the session is usable.
        if !self.started.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let response = tokio::time::timeout(
            self.options.timeouts.keepalive,
            self.platform(self.http.get(format!(
                "{}/sandboxes/{}",
                self.api_base,
                self.sandbox_id()
            )))
            .send(),
        )
        .await;
        let Ok(Ok(response)) = response else {
            return Ok(false);
        };
        if !response.status().is_success() {
            return Ok(false);
        }
        #[derive(Deserialize)]
        struct Detail {
            #[serde(default)]
            state: Option<String>,
        }
        let detail: Detail = response.json().await.unwrap_or(Detail { state: None });
        Ok(detail.state.as_deref() == Some("running"))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        // Until `start` has prepared the workspace root it may not exist, so a command with no
        // explicit directory runs wherever envd puts it instead of failing on a missing cwd.
        let cwd = match (&request.cwd, self.started.load(Ordering::SeqCst)) {
            (None, false) => None,
            _ => Some(self.resolve_cwd(&request.cwd)?),
        };
        let user = request.user.as_ref().map(|user| user.name.clone());
        // envd takes the user out of band, so the `sudo` wrapper must not be built here.
        let env = self.request_envs(&request);
        let argv = ExecRequest {
            user: None,
            ..request.clone()
        }
        .resolved_argv();
        self.exec_at(argv, cwd, &env, request.timeout, user).await
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

    /// Resolve an exposed port to the host E2B routes it on.
    ///
    /// Mirrors `E2BSandboxSession._resolve_exposed_port`, which asks the vendor SDK for
    /// `get_host(port)`. That call is pure string building — `{port}-{sandbox id}.{domain}` —
    /// and issues no request, so this port builds the same host rather than adding a round trip.
    /// A host without an explicit port is reached over TLS on 443, as `_e2b_endpoint_from_host`
    /// decides.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        assert_exposed_port_configured(BACKEND_ID, port, &self.options.exposed_ports)?;
        let host = format!(
            "{port}-{sandbox_id}.{domain}",
            sandbox_id = self.sandbox_id(),
            domain = self.options.resolved_domain(),
        );
        endpoint_from_authority(&host).ok_or_else(|| {
            exposed_port_unavailable(
                BACKEND_ID,
                port,
                &self.options.exposed_ports,
                ExposedPortUnavailable::BackendUnavailable,
            )
        })
    }

    fn supports_pty(&self) -> bool {
        true
    }

    /// The sandbox's own filesystem decides every path, so `rm`, `ls` and `mv` resolve it first.
    fn resolves_paths_remotely(&self) -> bool {
        true
    }

    fn system_state_preserved_on_start(&self) -> bool {
        self.state_preserved
    }

    fn persist_skip_registry(&self) -> Option<&Mutex<std::collections::BTreeSet<String>>> {
        Some(&self.persist_skips)
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        let resolved = self.validate_remote_path_access(path, false, false).await?;
        let value = self
            .connect_unary(
                "filesystem.Filesystem/ListDir",
                json!({ "path": resolved, "depth": 1 }),
                "list a directory in the E2B sandbox",
            )
            .await?;
        Ok(value
            .get("entries")
            .and_then(Value::as_array)
            .map(|entries| entries.iter().map(entry_info_to_file_entry).collect())
            .unwrap_or_default())
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        let normalized = self.normalize_path_for_write(path)?;
        if normalized.trim_end_matches('/') == self.manifest.root().trim_end_matches('/') {
            if self.skip_next_root_mkdir.swap(false, Ordering::SeqCst) {
                return Ok(());
            }
            // The root has to stay creatable when it disappeared remotely, and remote path
            // validation runs its helper with the workspace as the working directory.
            return self.make_dir(&normalized).await;
        }
        let resolved = self.validate_remote_path_access(path, true, false).await?;
        // envd's MakeDir always creates the missing parents, so the contract of a plain `mkdir`
        // is kept by probing the parent first.
        if !parents {
            let parent = match resolved.rfind('/') {
                Some(0) => "/".to_string(),
                Some(index) => resolved[..index].to_string(),
                None => ".".to_string(),
            };
            let argv = vec!["test".to_string(), "-d".to_string(), parent];
            let probe = self.exec(ExecRequest::argv(argv.clone())).await?;
            if !probe.ok() {
                return Err(SandboxError::exec_nonzero(&argv, &probe));
            }
        }
        self.make_dir(&resolved).await
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        if recursive {
            self.workspace_path_policy()?
                .validate_recursive_remove(path)?;
        }
        let resolved = self
            .validate_remote_path_access(path, true, recursive)
            .await?;
        if recursive {
            return self.remove_at(&resolved).await;
        }
        // envd's `Remove` always deletes recursively, so a non-recursive request runs `rm -f`
        // instead: a model removing "a file" must not be able to wipe a directory tree.
        let command = vec![
            "rm".to_string(),
            "-f".to_string(),
            "--".to_string(),
            resolved,
        ];
        let result = self
            .exec_at(
                command.clone(),
                Some("/".to_string()),
                &self.command_envs(),
                Some(self.options.timeouts.fast_op),
                None,
            )
            .await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let source = self
            .validate_remote_path_access(source, true, false)
            .await?;
        let destination = self
            .validate_remote_path_access(destination, true, false)
            .await?;
        self.connect_unary(
            "filesystem.Filesystem/Move",
            json!({ "source": source, "destination": destination }),
            "move a path in the E2B sandbox",
        )
        .await
        .map(|_| ())
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let cwd = self.resolve_cwd(&request.cwd)?;
        let user = request.user.as_ref().map(|user| user.name.clone());
        let env = self.command_envs();
        let argv = PtyExecRequest {
            user: None,
            ..request.clone()
        }
        .resolved_argv();
        let start = self.start_request(&argv, Some(cwd), &env, Some((24, 80)), request.tty);
        // The per-call timeout is the process lifetime envd enforces, and it also bounds how
        // long the start request may stall, as Python passes it to `pty.create`.
        let what = "start a PTY process in the E2B sandbox";
        let envd_timeout = self.envd_timeout(request.timeout);
        let opened = self.open_process_stream(start, user.as_deref(), envd_timeout, what);
        let response = match request.timeout {
            None => opened.await?,
            Some(_) => tokio::time::timeout(envd_timeout, opened)
                .await
                .map_err(|_| {
                    SandboxError::new(
                        ErrorCode::ExecTimeout,
                        format!(
                            "Command `{}` timed out after {envd_timeout:?}",
                            argv.join(" ")
                        ),
                    )
                })??,
        };
        let (process, pruned) = self.ptys.register(request.tty)?;
        // A pruned session keeps running inside the sandbox until it is signalled.
        self.release_pty_processes(&pruned).await;
        pump_pty_stream(
            process.clone(),
            self.pty_pids.clone(),
            response.bytes_stream(),
        );
        let update = process
            .collect(request.yield_time_ms, request.max_output_tokens)
            .await;
        self.forget_finished_pty(&update, process.id());
        Ok(update)
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let process = self.ptys.get(request.session_id)?;
        if !request.chars.is_empty() {
            if !process.tty() {
                return Err(SandboxError::new(
                    ErrorCode::PtyStdinUnavailable,
                    format!(
                        "PTY session `{}` was not started with a TTY, so it has no stdin",
                        request.session_id
                    ),
                ));
            }
            let pid = self.pty_pid(request.session_id).ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::PtySessionNotFound,
                    format!(
                        "PTY session `{}` has not reported its process id yet",
                        request.session_id
                    ),
                )
            })?;
            self.connect_unary(
                "process.Process/SendInput",
                json!({
                    "process": { "pid": pid },
                    "input": { "pty": BASE64.encode(request.chars.as_bytes()) },
                }),
                "write to a PTY process in the E2B sandbox",
            )
            .await?;
        }
        let yield_time_ms =
            resolve_pty_write_yield_time_ms(request.yield_time_ms, request.chars.is_empty());
        let update = process
            .collect(yield_time_ms, request.max_output_tokens)
            .await;
        self.forget_finished_pty(&update, process.id());
        Ok(update)
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        let processes = self.ptys.take_all();
        self.release_pty_processes(&processes).await;
        Ok(())
    }

    fn session_state(&self) -> SandboxSessionState {
        SandboxSessionState::new(
            BACKEND_ID,
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.sandbox_id()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
        .with_snapshot_fingerprint(self.snapshot_fingerprint.lock().expect("poisoned").clone())
        .with_provider_options(&E2BPersistedOptions::from_options(&self.options))
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
        // A native snapshot captures the whole sandbox and cannot leave a path out, so a
        // registered skip path falls back to the tar archive.
        if self.options.workspace_persistence == WorkspacePersistence::Snapshot
            && self.persist_workspace_skip_paths().is_empty()
        {
            let timeout = self.options.timeouts.snapshot_tar;
            let snapshot_id = tokio::time::timeout(timeout, self.create_native_snapshot())
                .await
                .map_err(|_| {
                    SandboxError::new(
                        ErrorCode::SnapshotPersistError,
                        format!(
                            "native_snapshot_failed: the E2B snapshot took longer than {timeout:?}"
                        ),
                    )
                    .with_retryable(false)
                })??;
            return Ok(encode_snapshot_ref(&snapshot_id));
        }
        // Python wraps the sandbox-side `tar` command in `@retry_async` on a transient transport
        // failure (`e2b/sandbox.py` L1340).
        retry_async(
            RetryPolicy::new(),
            || self.persist_workspace_tar(),
            |error, _attempt| is_transient_sandbox_error(error),
        )
        .await
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        if decode_snapshot_ref(archive).is_some() {
            return Ok(());
        }
        crate::sandbox::archive::validate_hydrate_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        if let Some(snapshot_id) = decode_snapshot_ref(archive) {
            return self.restore_native_snapshot(&snapshot_id).await;
        }
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
            .exec_at(
                vec!["sh".into(), "-lc".into(), script],
                Some("/".to_string()),
                &self.command_envs(),
                Some(self.options.timeouts.snapshot_tar),
                None,
            )
            .await;
        // The archive is scratch space; removing it must not mask the extraction outcome.
        let _ = tokio::time::timeout(self.options.timeouts.cleanup, self.remove_at(&archive_path))
            .await;
        let result = result?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!(
                    "Failed to restore the E2B workspace: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------
// Cloud bucket mounts
// -------------------------------------------------------------------------------------------

/// Refuse a mount on any backend other than E2B.
///
/// Mirrors Python's `_assert_e2b_session`: the strategy's preflight assumes this
/// provider's image and its privilege model, so running it elsewhere would be guesswork.
fn assert_e2b_session(session: &dyn SandboxSession) -> SandboxResult<()> {
    let matches = session
        .as_any()
        .is_some_and(|any| any.is::<E2BSandboxSession>());
    if !matches {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "a E2B cloud bucket mount requires a E2B sandbox session, not the \
                 `{}` backend",
                session.backend_id()
            ),
        ));
    }
    Ok(())
}

/// Mounts rclone-backed cloud storage inside an E2B sandbox. Mirrors
/// `agents.extensions.sandbox.e2b.mounts.E2BCloudBucketMountStrategy`.
///
/// Activation checks that the sandbox supports FUSE, makes `/dev/fuse` usable by the command
/// user, installs `rclone` when it is missing, and then delegates to
/// [`InContainerMountStrategy`]. The rclone configuration carries the bucket credentials, so the
/// mount reports [`MountCredentialAuthority::MountScoped`] and the manifest has to acknowledge
/// the exact mount path before it activates.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct E2BCloudBucketMountStrategy {
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

impl E2BCloudBucketMountStrategy {
    /// A strategy mounting `remote_name:remote_path` from `config_text`.
    pub fn new(
        remote_name: impl Into<String>,
        remote_path: impl Into<String>,
        config_text: impl Into<String>,
    ) -> Self {
        E2BCloudBucketMountStrategy {
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
        E2BCloudBucketMountStrategy {
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

    /// The delegate used for one activation, with the mount's read-only flag applied.
    fn delegate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        pattern: RcloneMountPattern,
        include_credentials: bool,
    ) -> SandboxResult<InContainerMountStrategy> {
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

    /// Run the preflight the provider's image needs before an rclone mount.
    async fn preflight(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        if self.pattern.mode == RcloneMountMode::Fuse {
            ensure_fuse_support(session, self.install_tools).await?;
        }
        ensure_rclone(session, self.install_tools).await
    }
}

#[async_trait]
impl MountStrategy for E2BCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "e2b_cloud_bucket"
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
        assert_e2b_session(session)?;
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        let pattern = rclone_pattern_for_session(session, &self.pattern).await?;
        self.delegate(mount, session, pattern, true)?
            .activate(mount, session, mount_path)
            .await
    }

    async fn deactivate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        assert_e2b_session(session)?;
        // Teardown only needs the remote the mount attached, so it does not pay for the `id`
        // round trip, as Python's plain `_delegate()` does not either.
        self.delegate(mount, session, self.pattern.clone(), false)?
            .deactivate(mount, session, mount_path)
            .await
    }

    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        assert_e2b_session(session)?;
        session.authorize_mount_activation(mount, mount_path)?;
        self.preflight(session).await?;
        let pattern = rclone_pattern_for_session(session, &self.pattern).await?;
        self.delegate(mount, session, pattern, true)?
            .activate(mount, session, mount_path)
            .await
    }
}

// -------------------------------------------------------------------------------------------
// Remote PTY bookkeeping
// -------------------------------------------------------------------------------------------

/// Decode a `process.Process/Start` stream into a [`RemotePtyProcess`].
///
/// The core manager owns the session id, the output window and the pruning; this task owns the
/// transport, so it records the sandbox pid the session needs for `SendInput` and `SendSignal`
/// and closes the process when the stream ends.
fn pump_pty_stream<S>(
    process: Arc<RemotePtyProcess>,
    pids: Arc<Mutex<HashMap<u32, u32>>>,
    stream: S,
) where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        let mut stream = stream;
        let mut decoder = ConnectFrameDecoder::default();
        let mut exit_code = None;
        while let Some(chunk) = stream.next().await {
            let Ok(chunk) = chunk else { break };
            decoder.push(&chunk);
            for frame in decoder.drain() {
                let ConnectFrame::Message(value) = frame else {
                    continue;
                };
                let event = value.get("event").unwrap_or(&Value::Null);
                if let Some(pid) = event
                    .get("start")
                    .and_then(|start| start.get("pid"))
                    .and_then(Value::as_u64)
                {
                    pids.lock()
                        .expect("poisoned")
                        .insert(process.id(), pid as u32);
                }
                if let Some(data) = event.get("data") {
                    // A PTY reports everything on `pty`; a plain process splits the streams.
                    for field in ["pty", "stdout", "stderr"] {
                        if let Some(encoded) = data.get(field).and_then(Value::as_str)
                            && let Ok(bytes) = BASE64.decode(encoded)
                        {
                            process.push_output(&bytes);
                        }
                    }
                }
                if let Some(end) = event.get("end") {
                    // proto3 JSON omits zero values, so a missing `exitCode` means success.
                    exit_code =
                        Some(end.get("exitCode").and_then(Value::as_i64).unwrap_or(0) as i32);
                }
            }
        }
        process.close(exit_code);
    });
}

// -------------------------------------------------------------------------------------------
// Connect protocol framing
// -------------------------------------------------------------------------------------------

/// One frame of a Connect streaming body.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConnectFrame {
    /// A protocol message.
    Message(Value),
    /// The trailing frame, which carries any RPC-level error.
    EndStream(Value),
}

/// Encode `message` as a single Connect envelope: flags, length, JSON payload.
fn encode_connect_frame(message: &Value) -> Vec<u8> {
    let payload = serde_json::to_vec(message).expect("serializable");
    let mut frame = Vec::with_capacity(payload.len() + 5);
    frame.push(0);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// Reassembles Connect envelopes from a chunked HTTP body.
#[derive(Debug, Default)]
struct ConnectFrameDecoder {
    buffer: Vec<u8>,
}

impl ConnectFrameDecoder {
    fn push(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
    }

    /// Take every complete frame currently buffered.
    fn drain(&mut self) -> Vec<ConnectFrame> {
        let mut frames = Vec::new();
        let mut offset = 0usize;
        while self.buffer.len() >= offset + 5 {
            let flags = self.buffer[offset];
            let length = u32::from_be_bytes([
                self.buffer[offset + 1],
                self.buffer[offset + 2],
                self.buffer[offset + 3],
                self.buffer[offset + 4],
            ]) as usize;
            if self.buffer.len() < offset + 5 + length {
                break;
            }
            let payload = &self.buffer[offset + 5..offset + 5 + length];
            let value = serde_json::from_slice::<Value>(payload).unwrap_or(Value::Null);
            frames.push(if flags & 0x02 != 0 {
                ConnectFrame::EndStream(value)
            } else {
                ConnectFrame::Message(value)
            });
            offset += 5 + length;
        }
        self.buffer.drain(..offset);
        frames
    }
}

/// The error message carried by an `EndStreamResponse`, when the RPC failed.
fn connect_stream_error(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown error");
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    Some(format!("{code}: {message}"))
}

/// Fold a `process.StartResponse` message into the collected command output.
fn apply_process_event(
    value: &Value,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    exit_code: &mut Option<i32>,
) {
    let Some(event) = value.get("event") else {
        return;
    };
    if let Some(data) = event.get("data") {
        for (field, sink) in [("stdout", &mut *stdout), ("stderr", &mut *stderr)] {
            if let Some(encoded) = data.get(field).and_then(Value::as_str)
                && let Ok(bytes) = BASE64.decode(encoded)
            {
                sink.extend_from_slice(&bytes);
            }
        }
        // A process started with a PTY reports everything on the `pty` field.
        if let Some(encoded) = data.get("pty").and_then(Value::as_str)
            && let Ok(bytes) = BASE64.decode(encoded)
        {
            stdout.extend_from_slice(&bytes);
        }
    }
    if let Some(end) = event.get("end") {
        // proto3 JSON omits zero values, so a missing `exitCode` means success.
        *exit_code = Some(end.get("exitCode").and_then(Value::as_i64).unwrap_or(0) as i32);
    }
}

// -------------------------------------------------------------------------------------------
// Shared helpers
// -------------------------------------------------------------------------------------------

/// Convert an envd `filesystem.EntryInfo` into a [`FileEntry`].
fn entry_info_to_file_entry(value: &Value) -> FileEntry {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let is_dir = value.get("type").and_then(Value::as_str) == Some("FILE_TYPE_DIRECTORY");
    let size = value.get("size").and_then(|size| {
        size.as_u64()
            .or_else(|| size.as_str().and_then(|s| s.parse().ok()))
    });
    let permissions = value
        .get("mode")
        .and_then(Value::as_u64)
        .map(|mode| permissions_from_mode(mode as u32, is_dir));
    FileEntry {
        name,
        path,
        size,
        permissions,
        is_dir,
        is_symlink: value.get("type").and_then(Value::as_str) == Some("FILE_TYPE_SYMLINK"),
    }
}

/// Build [`Permissions`] from a numeric file mode.
pub(crate) fn permissions_from_mode(mode: u32, directory: bool) -> Permissions {
    Permissions::new((mode >> 6) & 0o7, (mode >> 3) & 0o7, mode & 0o7).as_directory(directory)
}

/// Encode a reference to a provider-native snapshot as a workspace archive.
pub(crate) fn encode_snapshot_ref(snapshot_id: &str) -> Vec<u8> {
    let mut encoded = SNAPSHOT_REF_MAGIC.to_vec();
    encoded.extend_from_slice(
        serde_json::to_string(&json!({ "snapshot_id": snapshot_id }))
            .expect("serializable")
            .as_bytes(),
    );
    encoded
}

/// Read back a snapshot reference written by [`encode_snapshot_ref`].
pub(crate) fn decode_snapshot_ref(raw: &[u8]) -> Option<String> {
    let body = raw.strip_prefix(SNAPSHOT_REF_MAGIC)?;
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("snapshot_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_the_api_key() {
        let options = E2BSandboxClientOptions::new().with_api_key("e2b_secret");
        let debug = format!("{options:?}");
        assert!(!debug.contains("e2b_secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn resolves_base_urls_from_the_domain() {
        let options = E2BSandboxClientOptions::new().with_domain("example.dev");
        assert_eq!(options.api_base(), "https://api.example.dev");
        assert_eq!(options.sandbox_base(), "https://sandbox.example.dev");
        let overridden = E2BSandboxClientOptions::new().with_base_url("http://127.0.0.1:9/");
        assert_eq!(overridden.api_base(), "http://127.0.0.1:9");
        assert_eq!(overridden.sandbox_base(), "http://127.0.0.1:9");
    }

    #[test]
    fn selects_the_default_template_per_sandbox_type() {
        assert_eq!(
            E2BSandboxClientOptions::new().resolved_template(),
            DEFAULT_E2B_TEMPLATE
        );
        assert_eq!(
            E2BSandboxClientOptions::new()
                .with_sandbox_type(E2BSandboxType::CodeInterpreter)
                .resolved_template(),
            DEFAULT_CODE_INTERPRETER_TEMPLATE
        );
        assert_eq!(
            E2BSandboxClientOptions::new()
                .with_template("custom")
                .resolved_template(),
            "custom"
        );
    }

    #[test]
    fn round_trips_connect_frames() {
        let frame = encode_connect_frame(&json!({ "a": 1 }));
        assert_eq!(frame[0], 0);
        let mut decoder = ConnectFrameDecoder::default();
        // Split the frame to prove the decoder reassembles partial chunks.
        decoder.push(&frame[..3]);
        assert!(decoder.drain().is_empty());
        decoder.push(&frame[3..]);
        assert_eq!(
            decoder.drain(),
            vec![ConnectFrame::Message(json!({ "a": 1 }))]
        );
    }

    #[test]
    fn decodes_end_stream_frames() {
        let payload = br#"{"error":{"code":"internal","message":"boom"}}"#;
        let mut frame = vec![2u8];
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        let mut decoder = ConnectFrameDecoder::default();
        decoder.push(&frame);
        let frames = decoder.drain();
        let ConnectFrame::EndStream(value) = &frames[0] else {
            panic!("expected an end-stream frame: {frames:?}");
        };
        assert_eq!(
            connect_stream_error(value).as_deref(),
            Some("internal: boom")
        );
    }

    #[test]
    fn folds_process_events_into_output() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exit_code = None;
        apply_process_event(
            &json!({ "event": { "start": { "pid": 7 } } }),
            &mut stdout,
            &mut stderr,
            &mut exit_code,
        );
        apply_process_event(
            &json!({ "event": { "data": { "stdout": BASE64.encode("hi\n") } } }),
            &mut stdout,
            &mut stderr,
            &mut exit_code,
        );
        apply_process_event(
            &json!({ "event": { "data": { "stderr": BASE64.encode("warn\n") } } }),
            &mut stdout,
            &mut stderr,
            &mut exit_code,
        );
        apply_process_event(
            &json!({ "event": { "end": { "exited": true, "exitCode": 3 } } }),
            &mut stdout,
            &mut stderr,
            &mut exit_code,
        );
        assert_eq!(stdout, b"hi\n");
        assert_eq!(stderr, b"warn\n");
        assert_eq!(exit_code, Some(3));
        // proto3 JSON omits zero values, so a success end event carries no `exitCode`.
        apply_process_event(
            &json!({ "event": { "end": { "exited": true } } }),
            &mut stdout,
            &mut stderr,
            &mut exit_code,
        );
        assert_eq!(exit_code, Some(0));
    }

    #[test]
    fn round_trips_snapshot_references() {
        let encoded = encode_snapshot_ref("team/snap:v1");
        assert_eq!(
            decode_snapshot_ref(&encoded).as_deref(),
            Some("team/snap:v1")
        );
        assert_eq!(decode_snapshot_ref(b"not a snapshot"), None);
    }

    #[test]
    fn maps_entry_info_to_file_entries() {
        let entry = entry_info_to_file_entry(&json!({
            "name": "task.md",
            "path": "/workspace/task.md",
            "type": "FILE_TYPE_FILE",
            "size": "12",
            "mode": 420,
        }));
        assert_eq!(entry.name, "task.md");
        assert_eq!(entry.size, Some(12));
        assert!(!entry.is_dir);
        assert_eq!(entry.permissions.unwrap().octal_string(), "0644");
    }
}
