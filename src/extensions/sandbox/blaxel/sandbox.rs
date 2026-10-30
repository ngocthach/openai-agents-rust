//! A sandbox session backed by a Blaxel sandbox. Mirrors
//! `agents.extensions.sandbox.blaxel.sandbox`.
//!
//! The Python SDK drives `blaxel.core.sandbox.SandboxInstance`; this port calls the same two
//! Blaxel HTTP surfaces directly.
//!
//! The control plane (`https://api.blaxel.ai/v0`, headers `X-Blaxel-Authorization` and
//! `X-Blaxel-Workspace`) owns the sandbox lifecycle:
//!
//! | Operation | Request |
//! | --- | --- |
//! | create | `POST /sandboxes?createIfNotExist=true` |
//! | reconnect | `GET /sandboxes/{name}` |
//! | delete | `DELETE /sandboxes/{name}` |
//! | snapshot | `POST /sandboxes/{name}/snapshots` |
//! | restore | `POST /sandboxes/{name}/snapshots/{id}/restore` |
//!
//! The sandbox itself answers on `metadata.url`, falling back to
//! `https://run.blaxel.ai/{workspace}/sandboxes/{name}`, with the same
//! `X-Blaxel-Authorization` header:
//!
//! | Operation | Request |
//! | --- | --- |
//! | exec | `POST /process` with `{command, workingDir, waitForCompletion, timeout, env}` |
//! | read | `GET /filesystem/{path}` with `Accept: application/octet-stream` |
//! | write | `PUT /filesystem/{path}` as `multipart/form-data` |
//! | mkdir | `PUT /filesystem/{path}` with `{"isDirectory": true}` |
//! | list | `GET /filesystem/{path}` |
//! | remove | `DELETE /filesystem/{path}?recursive=true` |
//! | mount drive | `POST /drives/mount`, `DELETE /drives/mount/{path}` |
//!
//! PTY sessions run over the sandbox's terminal WebSocket; see [`super::pty`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::retry::{RetryPolicy, is_transient_sandbox_error, retry_async};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot, shell_quote,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::{ExecResult, FileEntry, Permissions};
use crate::sandbox::workspace_paths::provider_scratch_path;

use super::super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, endpoint_from_url, exposed_port_unavailable,
};
use super::pty::{BlaxelPtySessions, terminal_url};

/// The workspace root Blaxel sandboxes use by default.
pub const DEFAULT_BLAXEL_WORKSPACE_ROOT: &str = "/workspace";
/// The control-plane base URL used when none is configured.
pub const DEFAULT_API_BASE_URL: &str = "https://api.blaxel.ai/v0";
/// The sandbox base URL used when the control plane reports no `metadata.url`.
pub const DEFAULT_RUN_BASE_URL: &str = "https://run.blaxel.ai";
/// The region used when neither the options nor `BL_REGION` name one.
pub const DEFAULT_REGION: &str = "us-pdx-1";
/// The environment variable holding the Blaxel API key.
pub const API_KEY_ENV: &str = "BL_API_KEY";
/// The environment variable holding the Blaxel workspace.
pub const WORKSPACE_ENV: &str = "BL_WORKSPACE";
/// The environment variable holding the default Blaxel region.
pub const REGION_ENV: &str = "BL_REGION";

/// Timeouts for the Blaxel sandbox operations, matching Python's `BlaxelTimeouts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlaxelTimeouts {
    /// The default exec timeout.
    pub exec: Duration,
    /// The timeout for best-effort cleanup commands.
    pub cleanup: Duration,
    /// The timeout for uploading a file.
    pub file_upload: Duration,
    /// The timeout for downloading a file.
    pub file_download: Duration,
    /// The timeout for the `tar` commands behind snapshots.
    pub workspace_tar: Duration,
    /// The timeout for short control-plane calls.
    pub fast_op: Duration,
}

impl Default for BlaxelTimeouts {
    fn default() -> Self {
        BlaxelTimeouts {
            exec: Duration::from_secs(300),
            cleanup: Duration::from_secs(30),
            file_upload: Duration::from_secs(1800),
            file_download: Duration::from_secs(1800),
            workspace_tar: Duration::from_secs(300),
            fast_op: Duration::from_secs(30),
        }
    }
}

/// Settings used when creating a Blaxel-backed session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlaxelSandboxClientOptions {
    /// The sandbox image, or the Blaxel default when unset.
    pub image: Option<String>,
    /// The memory allocation in MiB.
    pub memory: Option<u32>,
    /// The region, defaulting to `BL_REGION` and then [`DEFAULT_REGION`].
    pub region: Option<String>,
    /// Ports declared at creation, each as the Blaxel port object.
    pub ports: Vec<Value>,
    /// Environment variables the sandbox is created with.
    pub env_vars: BTreeMap<String, String>,
    /// Labels attached to the sandbox.
    pub labels: BTreeMap<String, String>,
    /// The sandbox time-to-live.
    pub ttl: Option<String>,
    /// An explicit sandbox name, or a generated `agents-<id>` name.
    pub name: Option<String>,
    /// Keep the sandbox alive on shutdown so a later `resume` reconnects to it.
    pub pause_on_exit: bool,
    /// The operation timeouts.
    pub timeouts: BlaxelTimeouts,
    /// The API key, defaulting to `BL_API_KEY`.
    pub api_key: Option<String>,
    /// The workspace, defaulting to `BL_WORKSPACE`.
    pub workspace: Option<String>,
    /// The control-plane base URL, defaulting to [`DEFAULT_API_BASE_URL`].
    pub api_base_url: Option<String>,
    /// The sandbox base URL prefix, defaulting to [`DEFAULT_RUN_BASE_URL`].
    pub run_base_url: Option<String>,
    /// Address the sandbox at this URL instead of the one the control plane reports.
    pub force_sandbox_url: Option<String>,
    /// Whether a preview created for an exposed port is public.
    pub exposed_port_public: bool,
    /// How long a private preview's access token stays valid, in seconds.
    pub exposed_port_url_ttl_secs: u32,
}

impl Default for BlaxelSandboxClientOptions {
    fn default() -> Self {
        BlaxelSandboxClientOptions {
            image: None,
            memory: None,
            region: None,
            ports: Vec::new(),
            env_vars: BTreeMap::new(),
            labels: BTreeMap::new(),
            ttl: None,
            name: None,
            pause_on_exit: false,
            timeouts: BlaxelTimeouts::default(),
            api_key: None,
            workspace: None,
            api_base_url: None,
            run_base_url: None,
            force_sandbox_url: None,
            // Python's `BlaxelSandboxClientOptions` defaults, not Rust's zero values.
            exposed_port_public: true,
            exposed_port_url_ttl_secs: 3600,
        }
    }
}

impl BlaxelSandboxClientOptions {
    /// Options using the Blaxel defaults.
    pub fn new() -> Self {
        BlaxelSandboxClientOptions::default()
    }

    /// Run the sandbox from `image`.
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Allocate `memory` MiB.
    pub fn with_memory(mut self, memory: u32) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Create the sandbox in `region`.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Declare a port at creation.
    pub fn with_port(mut self, port: Value) -> Self {
        self.ports.push(port);
        self
    }

    /// Create the sandbox with an environment variable.
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Label the sandbox.
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Set the sandbox time-to-live.
    pub fn with_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.ttl = Some(ttl.into());
        self
    }

    /// Name the sandbox instead of generating a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Keep the sandbox alive on shutdown.
    pub fn with_pause_on_exit(mut self, pause_on_exit: bool) -> Self {
        self.pause_on_exit = pause_on_exit;
        self
    }

    /// Replace the operation timeouts.
    pub fn with_timeouts(mut self, timeouts: BlaxelTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Authenticate with an explicit API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use an explicit workspace.
    pub fn with_workspace(mut self, workspace: impl Into<String>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    /// Talk to a control plane at `url` instead of [`DEFAULT_API_BASE_URL`].
    pub fn with_api_base_url(mut self, url: impl Into<String>) -> Self {
        self.api_base_url = Some(url.into());
        self
    }

    /// Build fallback sandbox URLs from `url` instead of [`DEFAULT_RUN_BASE_URL`].
    pub fn with_run_base_url(mut self, url: impl Into<String>) -> Self {
        self.run_base_url = Some(url.into());
        self
    }

    /// Address the sandbox at `url`, ignoring the control plane's `metadata.url`.
    pub fn with_force_sandbox_url(mut self, url: impl Into<String>) -> Self {
        self.force_sandbox_url = Some(url.into());
        self
    }

    /// The API key, falling back to [`API_KEY_ENV`].
    pub fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .filter(|key| !key.is_empty())
    }

    /// The workspace, falling back to [`WORKSPACE_ENV`].
    pub fn resolve_workspace(&self) -> SandboxResult<String> {
        self.workspace
            .clone()
            .or_else(|| std::env::var(WORKSPACE_ENV).ok())
            .filter(|workspace| !workspace.is_empty())
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!(
                        "the Blaxel workspace is missing: set `{WORKSPACE_ENV}` or \
                         `BlaxelSandboxClientOptions::with_workspace`"
                    ),
                )
            })
    }

    /// The region, falling back to [`REGION_ENV`] and then [`DEFAULT_REGION`].
    pub fn resolve_region(&self) -> String {
        self.region
            .clone()
            .or_else(|| std::env::var(REGION_ENV).ok())
            .filter(|region| !region.is_empty())
            .unwrap_or_else(|| DEFAULT_REGION.to_string())
    }

    fn api_base_url(&self) -> String {
        self.api_base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_API_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    fn run_base_url(&self) -> String {
        self.run_base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_RUN_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string()
    }
}

/// The sandbox endpoint and credentials behind one Blaxel sandbox.
///
/// A mount strategy recovers it through [`SandboxSession::as_any`], because
/// [`MountStrategy`](crate::sandbox::mounts::MountStrategy) only receives a
/// `&dyn SandboxSession`.
#[derive(Debug)]
pub struct BlaxelConnection {
    http: reqwest::Client,
    sandbox_name: String,
    sandbox_url: String,
    api_base_url: String,
    workspace: String,
    api_key: Option<String>,
    timeouts: BlaxelTimeouts,
    exposed_port_public: bool,
    exposed_port_url_ttl_secs: u32,
}

impl BlaxelConnection {
    /// The sandbox this connection addresses.
    pub fn sandbox_name(&self) -> &str {
        &self.sandbox_name
    }

    /// The sandbox base URL, without a trailing slash.
    pub fn sandbox_url(&self) -> &str {
        &self.sandbox_url
    }

    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self
            .http
            .request(method, url)
            .header("X-Blaxel-Workspace", &self.workspace);
        if let Some(key) = &self.api_key {
            builder = builder.header("X-Blaxel-Authorization", format!("Bearer {key}"));
        }
        builder
    }

    fn sandbox_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.sandbox_url, path);
        self.request(method, &url)
    }

    fn control_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.api_base_url, path);
        self.request(method, &url)
    }

    /// Create, or reuse, a preview for `port` and return a reachable endpoint.
    ///
    /// Mirrors `BlaxelSandboxSession._resolve_exposed_port`: `previews.create_if_not_exists`
    /// (`POST /sandboxes/{name}/previews`, falling back to the existing preview on a conflict)
    /// and, for a private preview, `preview.tokens.create`
    /// (`POST /sandboxes/{name}/previews/{preview}/tokens`), whose token travels in the URL's
    /// `bl_preview_token` query.
    ///
    /// Blaxel mints a preview for any port on demand, so, as in Python, the port does not have
    /// to be declared at creation.
    pub async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        let unavailable = |detail: &str| {
            exposed_port_unavailable(
                "blaxel",
                port,
                &[],
                ExposedPortUnavailable::BackendUnavailable,
            )
            .with_context("detail", detail)
        };
        let preview_name = format!("port-{port}");
        // `create_if_not_exists` reads first and creates only on a 404, then falls back to a
        // second read if the create races another caller to a 409.
        let preview: Value = match self.read_preview(port, &preview_name).await? {
            Some(preview) => preview,
            None => {
                let created = self
                    .control_request(
                        reqwest::Method::POST,
                        &format!("/sandboxes/{}/previews", self.sandbox_name),
                    )
                    .json(&json!({
                        "metadata": { "name": preview_name },
                        "spec": { "port": port, "public": self.exposed_port_public },
                    }))
                    .timeout(self.timeouts.fast_op)
                    .send()
                    .await
                    .map_err(|_| unavailable("preview_creation_failed"))?;
                if created.status().is_success() {
                    created
                        .json()
                        .await
                        .map_err(|_| unavailable("preview_creation_failed"))?
                } else if created.status() == reqwest::StatusCode::CONFLICT {
                    self.read_preview(port, &preview_name)
                        .await?
                        .ok_or_else(|| unavailable("preview_creation_failed"))?
                } else {
                    return Err(unavailable("preview_creation_failed"));
                }
            }
        };
        let url = preview_url(&preview).ok_or_else(|| unavailable("invalid_preview_url"))?;
        let mut endpoint =
            endpoint_from_url(&url).ok_or_else(|| unavailable("url_parse_failed"))?;
        if !self.exposed_port_public {
            let expires_at = (chrono::Utc::now()
                + chrono::Duration::seconds(i64::from(self.exposed_port_url_ttl_secs)))
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
            let response = self
                .control_request(
                    reqwest::Method::POST,
                    &format!(
                        "/sandboxes/{}/previews/{preview_name}/tokens",
                        self.sandbox_name
                    ),
                )
                .json(&json!({ "spec": { "expiresAt": expires_at } }))
                .timeout(self.timeouts.fast_op)
                .send()
                .await
                .map_err(|_| unavailable("preview_token_creation_failed"))?;
            if !response.status().is_success() {
                return Err(unavailable("preview_token_creation_failed"));
            }
            let token: Value = response
                .json()
                .await
                .map_err(|_| unavailable("preview_token_creation_failed"))?;
            if let Some(value) = token
                .get("spec")
                .and_then(|spec| spec.get("token"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                endpoint = endpoint.with_query(format!("bl_preview_token={value}"));
            }
        }
        Ok(endpoint)
    }

    /// Read the preview for `port` by name, or `None` when the control plane says it is absent.
    async fn read_preview(&self, port: u16, preview_name: &str) -> SandboxResult<Option<Value>> {
        let failed = || {
            exposed_port_unavailable(
                "blaxel",
                port,
                &[],
                ExposedPortUnavailable::BackendUnavailable,
            )
            .with_context("detail", "preview_lookup_failed")
            .with_context("preview", preview_name)
        };
        let response = self
            .control_request(
                reqwest::Method::GET,
                &format!("/sandboxes/{}/previews/{preview_name}", self.sandbox_name),
            )
            .timeout(self.timeouts.fast_op)
            .send()
            .await
            .map_err(|_| failed())?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(failed());
        }
        response.json().await.map(Some).map_err(|_| failed())
    }

    /// Attach a Blaxel Drive at `mount_path` (`POST /drives/mount`).
    pub async fn mount_drive(&self, body: Value, mount_path: &str) -> SandboxResult<()> {
        let response = self
            .sandbox_request(reqwest::Method::POST, "/drives/mount")
            .json(&body)
            .timeout(self.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!("blaxel drive mount at `{mount_path}` failed: {error}"),
                )
            })?;
        if !response.status().is_success() {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "blaxel drive mount at `{mount_path}` failed: HTTP {}",
                    response.status().as_u16()
                ),
            ));
        }
        Ok(())
    }

    /// Detach the drive at `mount_path` (`DELETE /drives/mount/{mountPath}`).
    pub async fn unmount_drive(&self, mount_path: &str) -> SandboxResult<()> {
        let path = format!("/drives/mount/{}", encode_sandbox_path(mount_path));
        let response = self
            .sandbox_request(reqwest::Method::DELETE, &path)
            .timeout(self.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!("blaxel drive unmount at `{mount_path}` failed: {error}"),
                )
            })?;
        if !response.status().is_success() {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "blaxel drive unmount at `{mount_path}` failed: HTTP {}",
                    response.status().as_u16()
                ),
            ));
        }
        Ok(())
    }
}

/// The URL of a preview, which the SDK reads from `spec.url` and then `status.url`.
///
/// Mirrors `_extract_preview_url`, including its fallbacks to a top-level `url`/`endpoint` and
/// to a nested `preview` object.
fn preview_url(preview: &Value) -> Option<String> {
    for nested in ["spec", "status"] {
        if let Some(url) = preview
            .get(nested)
            .and_then(|object| object.get("url"))
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
        {
            return Some(url.to_string());
        }
    }
    for field in ["url", "endpoint"] {
        if let Some(url) = preview
            .get(field)
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
        {
            return Some(url.to_string());
        }
    }
    preview.get("preview").and_then(preview_url)
}

/// Percent-encode a sandbox path for a Blaxel filesystem URL.
///
/// Blaxel resolves a path without a leading slash against the filesystem service's working
/// directory, so an absolute path must encode its leading slash as `%2F`; the separators inside
/// stay literal.
pub fn encode_sandbox_path(path: &str) -> String {
    let mut encoded = String::with_capacity(path.len() + 2);
    for (index, byte) in path.as_bytes().iter().enumerate() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(*byte as char)
            }
            b'/' if index > 0 => encoded.push('/'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

/// Creates [`BlaxelSandboxSession`]s through the Blaxel control plane.
#[derive(Debug, Clone)]
pub struct BlaxelSandboxClient {
    options: BlaxelSandboxClientOptions,
    snapshot: Option<SnapshotSpec>,
    http: reqwest::Client,
}

impl BlaxelSandboxClient {
    /// A client using the given options.
    pub fn new(options: BlaxelSandboxClientOptions) -> Self {
        BlaxelSandboxClient {
            options,
            snapshot: None,
            http: reqwest::Client::new(),
        }
    }

    /// Store workspace snapshots so a paused run can resume the same workspace.
    pub fn with_snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Share an existing HTTP client.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The `POST /sandboxes` body for a sandbox name.
    fn create_body(&self, name: &str) -> Value {
        let mut metadata = Map::new();
        metadata.insert("name".to_string(), Value::String(name.to_string()));
        if !self.options.labels.is_empty() {
            metadata.insert(
                "labels".to_string(),
                Value::Object(
                    self.options
                        .labels
                        .iter()
                        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                        .collect(),
                ),
            );
        }

        let mut runtime = Map::new();
        if let Some(image) = &self.options.image {
            runtime.insert("image".to_string(), Value::String(image.clone()));
        }
        if let Some(memory) = self.options.memory {
            runtime.insert("memory".to_string(), Value::from(memory));
        }
        if !self.options.ports.is_empty() {
            runtime.insert(
                "ports".to_string(),
                Value::Array(self.options.ports.clone()),
            );
        }
        if !self.options.env_vars.is_empty() {
            runtime.insert(
                "envs".to_string(),
                Value::Array(
                    self.options
                        .env_vars
                        .iter()
                        .map(|(name, value)| json!({"name": name, "value": value}))
                        .collect(),
                ),
            );
        }
        if let Some(ttl) = &self.options.ttl {
            runtime.insert("ttl".to_string(), Value::String(ttl.clone()));
        }

        let mut spec = Map::new();
        spec.insert("runtime".to_string(), Value::Object(runtime));
        spec.insert(
            "region".to_string(),
            Value::String(self.options.resolve_region()),
        );

        json!({"metadata": Value::Object(metadata), "spec": Value::Object(spec)})
    }

    fn request(
        &self,
        method: reqwest::Method,
        url: &str,
        workspace: &str,
    ) -> reqwest::RequestBuilder {
        let mut builder = self
            .http
            .request(method, url)
            .header("X-Blaxel-Workspace", workspace);
        if let Some(key) = self.options.resolve_api_key() {
            builder = builder.header("X-Blaxel-Authorization", format!("Bearer {key}"));
        }
        builder
    }

    /// Create the sandbox, or return the existing one (`createIfNotExist=true`).
    async fn create_sandbox(&self, name: &str, workspace: &str) -> SandboxResult<Value> {
        let url = format!("{}/sandboxes", self.options.api_base_url());
        let response = self
            .request(reqwest::Method::POST, &url, workspace)
            .query(&[("createIfNotExist", "true")])
            .json(&self.create_body(name))
            .timeout(self.options.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("`POST /sandboxes` failed for `{name}`: {error}"),
                )
            })?;
        parse_control_plane_body(response, "POST /sandboxes", ErrorCode::WorkspaceStartError).await
    }

    /// Look up an existing sandbox (`GET /sandboxes/{name}`).
    async fn get_sandbox(&self, name: &str, workspace: &str) -> SandboxResult<Value> {
        let url = format!("{}/sandboxes/{name}", self.options.api_base_url());
        let response = self
            .request(reqwest::Method::GET, &url, workspace)
            .timeout(self.options.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("`GET /sandboxes/{name}` failed: {error}"),
                )
            })?;
        parse_control_plane_body(
            response,
            &format!("GET /sandboxes/{name}"),
            ErrorCode::WorkspaceStartError,
        )
        .await
    }

    fn connection(
        &self,
        name: String,
        workspace: String,
        sandbox: &Value,
    ) -> Arc<BlaxelConnection> {
        let sandbox_url = self
            .options
            .force_sandbox_url
            .clone()
            .or_else(|| {
                sandbox
                    .pointer("/metadata/url")
                    .and_then(Value::as_str)
                    .filter(|url| !url.is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| {
                format!(
                    "{}/{workspace}/sandboxes/{name}",
                    self.options.run_base_url()
                )
            });
        Arc::new(BlaxelConnection {
            http: self.http.clone(),
            sandbox_name: name,
            sandbox_url: sandbox_url.trim_end_matches('/').to_string(),
            api_base_url: self.options.api_base_url(),
            workspace,
            api_key: self.options.resolve_api_key(),
            timeouts: self.options.timeouts.clone(),
            exposed_port_public: self.options.exposed_port_public,
            exposed_port_url_ttl_secs: self.options.exposed_port_url_ttl_secs,
        })
    }

    /// Create a session with its concrete type, for the Blaxel-only operations.
    ///
    /// [`create`](SandboxClient::create) returns a `dyn SandboxSession`, which cannot reach
    /// [`BlaxelSandboxSession::create_sandbox_snapshot`] and its siblings.
    pub async fn create_blaxel_session(
        &self,
        manifest: Option<Manifest>,
    ) -> SandboxResult<BlaxelSandboxSession> {
        let workspace = self.options.resolve_workspace()?;
        let manifest =
            manifest.unwrap_or_else(|| Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT));
        let name = self.options.name.clone().unwrap_or_else(|| {
            format!(
                "agents-{}",
                &uuid::Uuid::new_v4().simple().to_string()[..12]
            )
        });
        let sandbox = self.create_sandbox(&name, &workspace).await?;
        let connection = self.connection(name, workspace, &sandbox);
        Ok(self.session(connection, manifest))
    }

    fn session(
        &self,
        connection: Arc<BlaxelConnection>,
        manifest: Manifest,
    ) -> BlaxelSandboxSession {
        BlaxelSandboxSession::new(connection, manifest)
            .with_snapshot(self.snapshot.clone())
            .with_base_env_vars(self.options.env_vars.clone())
            .with_pause_on_exit(self.options.pause_on_exit)
    }
}

#[async_trait]
impl SandboxClient for BlaxelSandboxClient {
    fn backend_id(&self) -> &str {
        "blaxel"
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        Ok(Arc::new(self.create_blaxel_session(manifest).await?))
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        if state.backend != "blaxel" {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "sandbox session state was created by the `{}` backend",
                    state.backend
                ),
            ));
        }
        state.verify_manifest(&manifest)?;
        let workspace = self.options.resolve_workspace()?;
        let name = state.provider_id.clone().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "blaxel sandbox session state is missing its sandbox name",
            )
        })?;
        // A paused sandbox keeps its filesystem and resumes on the next connection, so a
        // successful lookup means the workspace is already materialized.
        let reconnected = if self.options.pause_on_exit {
            self.get_sandbox(&name, &workspace).await.ok()
        } else {
            None
        };
        let (sandbox, preserved) = match reconnected {
            Some(sandbox) => (sandbox, true),
            None => (self.create_sandbox(&name, &workspace).await?, false),
        };
        let connection = self.connection(name, workspace, &sandbox);
        let mut session = self
            .session(connection, manifest)
            .with_snapshot(Some(state.snapshot.clone()))
            .with_snapshot_id(state.snapshot_id.clone());
        if preserved {
            session = session.preserving_workspace();
        } else {
            session = session.resuming();
        }
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// A sandbox session whose workspace lives inside a Blaxel sandbox.
#[derive(Debug)]
pub struct BlaxelSandboxSession {
    connection: Arc<BlaxelConnection>,
    session_id: String,
    manifest: Manifest,
    base_env_vars: BTreeMap<String, String>,
    started: AtomicBool,
    deleted: tokio::sync::Mutex<bool>,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    pause_on_exit: bool,
    resuming: bool,
    preserved_workspace: bool,
    ptys: BlaxelPtySessions,
}

impl BlaxelSandboxSession {
    /// Attach a session to a sandbox the control plane already created.
    pub fn new(connection: Arc<BlaxelConnection>, manifest: Manifest) -> Self {
        // The session id is its own value, as in the Python port: one Blaxel sandbox can back
        // several sessions over time, and a mount must reach the connection of its own.
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let snapshot_id = session_id.clone();
        BlaxelSandboxSession {
            connection,
            session_id,
            manifest,
            base_env_vars: BTreeMap::new(),
            started: AtomicBool::new(false),
            deleted: tokio::sync::Mutex::new(false),
            snapshot: None,
            snapshot_id,
            pause_on_exit: false,
            resuming: false,
            preserved_workspace: false,
            ptys: BlaxelPtySessions::new(),
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

    /// Prepend the sandbox's creation environment to every command.
    pub fn with_base_env_vars(mut self, base_env_vars: BTreeMap<String, String>) -> Self {
        self.base_env_vars = base_env_vars;
        self
    }

    /// Keep the sandbox alive on shutdown.
    pub fn with_pause_on_exit(mut self, pause_on_exit: bool) -> Self {
        self.pause_on_exit = pause_on_exit;
        self
    }

    /// Restore the snapshot on start instead of materializing the full manifest.
    pub fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// Treat the sandbox filesystem as already materialized, as a paused sandbox is.
    pub fn preserving_workspace(mut self) -> Self {
        self.preserved_workspace = true;
        self
    }

    /// The sandbox endpoint behind this session.
    pub fn connection(&self) -> &Arc<BlaxelConnection> {
        &self.connection
    }

    /// The control plane's name for this sandbox.
    pub fn sandbox_name(&self) -> &str {
        &self.connection.sandbox_name
    }

    /// The environment a command runs with: the sandbox's own, the manifest's, then the
    /// request's.
    fn resolved_env(&self, request: &ExecRequest) -> BTreeMap<String, String> {
        let mut env = self.base_env_vars.clone();
        env.extend(request.resolved_env(self.manifest.environment()));
        env
    }

    /// The `POST /process` body for one command.
    fn process_body(&self, request: &ExecRequest) -> SandboxResult<Value> {
        let command = request
            .resolved_argv()
            .iter()
            .map(|part| shell_quote(part))
            .collect::<Vec<_>>()
            .join(" ");
        let working_dir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        let timeout = request.timeout.unwrap_or(self.connection.timeouts.exec);
        let mut body = Map::new();
        body.insert("command".to_string(), Value::String(command));
        body.insert("workingDir".to_string(), Value::String(working_dir));
        body.insert("waitForCompletion".to_string(), Value::Bool(true));
        body.insert(
            "timeout".to_string(),
            Value::from(timeout.as_millis().max(1) as u64),
        );
        let env = self.resolved_env(request);
        if !env.is_empty() {
            body.insert(
                "env".to_string(),
                Value::Object(
                    env.into_iter()
                        .map(|(key, value)| (key, Value::String(value)))
                        .collect(),
                ),
            );
        }
        Ok(Value::Object(body))
    }

    /// Turn a `ProcessResponse` into an [`ExecResult`].
    ///
    /// Older sandbox builds report only a combined `logs` stream; it is attributed to stdout on
    /// success and to stderr on failure, as the Python port does.
    fn parse_process_response(payload: &Value) -> ExecResult {
        let exit_code = payload
            .get("exitCode")
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32;
        let stdout = payload.get("stdout").and_then(Value::as_str);
        let stderr = payload.get("stderr").and_then(Value::as_str);
        if stdout.is_some() || stderr.is_some() {
            return ExecResult {
                stdout: stdout.unwrap_or_default().as_bytes().to_vec(),
                stderr: stderr.unwrap_or_default().as_bytes().to_vec(),
                exit_code,
            };
        }
        let logs = payload
            .get("logs")
            .or_else(|| payload.get("output"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .as_bytes()
            .to_vec();
        if exit_code == 0 {
            ExecResult {
                stdout: logs,
                stderr: Vec::new(),
                exit_code,
            }
        } else {
            ExecResult {
                stdout: Vec::new(),
                stderr: logs,
                exit_code,
            }
        }
    }

    async fn filesystem_put(&self, resolved: &str, body: Value) -> SandboxResult<()> {
        let path = format!("/filesystem/{}", encode_sandbox_path(resolved));
        let response = self
            .connection
            .sandbox_request(reqwest::Method::PUT, &path)
            .json(&body)
            .timeout(self.connection.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("blaxel filesystem write to `{resolved}` failed: {error}"),
                )
            })?;
        if !response.status().is_success() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!(
                    "blaxel filesystem write to `{resolved}` failed: HTTP {}",
                    response.status().as_u16()
                ),
            ));
        }
        Ok(())
    }

    /// Capture a snapshot of the sandbox (`POST /sandboxes/{name}/snapshots`).
    ///
    /// Blaxel names the snapshot itself; the returned name is what
    /// [`restore_sandbox_snapshot`](Self::restore_sandbox_snapshot) takes.
    pub async fn create_sandbox_snapshot(&self) -> SandboxResult<String> {
        let path = format!("/sandboxes/{}/snapshots", self.connection.sandbox_name);
        let response = self
            .connection
            .control_request(reqwest::Method::POST, &path)
            .timeout(self.connection.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    format!("`POST {path}` failed: {error}"),
                )
            })?;
        let payload = parse_control_plane_body(
            response,
            &format!("POST {path}"),
            ErrorCode::SnapshotPersistError,
        )
        .await?;
        payload
            .pointer("/metadata/name")
            .or_else(|| payload.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    "the Blaxel snapshot response carried no snapshot name",
                )
            })
    }

    /// Roll the sandbox back onto one of its snapshots.
    pub async fn restore_sandbox_snapshot(&self, snapshot_name: &str) -> SandboxResult<()> {
        let path = format!(
            "/sandboxes/{}/snapshots/{snapshot_name}/restore",
            self.connection.sandbox_name
        );
        let response = self
            .connection
            .control_request(reqwest::Method::POST, &path)
            .timeout(self.connection.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    format!("`POST {path}` failed: {error}"),
                )
            })?;
        if !response.status().is_success() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("`POST {path}` failed: HTTP {}", response.status().as_u16()),
            ));
        }
        Ok(())
    }

    /// Read a file the SDK itself placed, which may be an SDK-owned scratch path.
    ///
    /// Snapshots stage their archive outside the workspace the model may touch, so the path goes
    /// through [`resolve_internal_path`](SandboxSession::resolve_internal_path) rather than the
    /// workspace guard that model-provided paths meet.
    async fn read_internal(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.resolve_internal_path(path)?;
        let resolved = resolved.as_str();
        let url_path = format!("/filesystem/{}", encode_sandbox_path(resolved));
        let response = self
            .connection
            .sandbox_request(reqwest::Method::GET, &url_path)
            .header("Accept", "application/octet-stream")
            .timeout(self.connection.timeouts.file_download)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("Failed to read `{resolved}`: {error}"),
                )
            })?;
        let status = response.status();
        if status.as_u16() == 404 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("`{resolved}` does not exist in the sandbox workspace"),
            ));
        }
        let body = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!(
                    "Failed to read `{resolved}`: HTTP {}{}",
                    status.as_u16(),
                    blaxel_error_detail(&body)
                        .map(|detail| format!(": {detail}"))
                        .unwrap_or_default()
                ),
            ));
        }
        Ok(body.to_vec())
    }

    /// Write a file the SDK itself placed, as `multipart/form-data`.
    async fn write_internal(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.resolve_internal_path(path)?;
        let resolved = resolved.as_str();
        let url_path = format!("/filesystem/{}", encode_sandbox_path(resolved));
        let boundary = format!("openai-agents-{}", uuid::Uuid::new_v4().simple());
        let body = multipart_file_body(&boundary, resolved, contents);
        let response = self
            .connection
            .sandbox_request(reqwest::Method::PUT, &url_path)
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .timeout(self.connection.timeouts.file_upload)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("Failed to write `{resolved}`: {error}"),
                )
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!(
                    "Failed to write `{resolved}`: HTTP {}{}",
                    status.as_u16(),
                    blaxel_error_detail(&body)
                        .map(|detail| format!(": {detail}"))
                        .unwrap_or_default()
                ),
            ));
        }
        Ok(())
    }

    /// Remove a path the SDK itself placed.
    async fn remove_internal(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let resolved = self.resolve_internal_path(path)?;
        let resolved = resolved.as_str();
        let url_path = format!("/filesystem/{}", encode_sandbox_path(resolved));
        let mut builder = self
            .connection
            .sandbox_request(reqwest::Method::DELETE, &url_path)
            .timeout(self.connection.timeouts.fast_op);
        if recursive {
            builder = builder.query(&[("recursive", "true")]);
        }
        let response = builder.send().await.map_err(|error| {
            SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to remove `{resolved}`: {error}"),
            )
        })?;
        let status = response.status();
        if !status.is_success() && status.as_u16() != 404 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                format!("Failed to remove `{resolved}`: HTTP {}", status.as_u16()),
            ));
        }
        Ok(())
    }

    fn tar_path(&self, kind: &str) -> String {
        provider_scratch_path(&format!(
            "blaxel-{kind}-{}.tar",
            self.connection.sandbox_name
        ))
    }

    fn tar_exclude_args(&self) -> String {
        let root = self.manifest.root().trim_end_matches('/').to_string();
        let mut excludes: Vec<String> =
            crate::sandbox::session::workspace_archive_excludes(&self.manifest, &root)
                .into_iter()
                .map(|e| e.trim_start_matches("./").to_string())
                .collect();
        excludes.sort();
        excludes.dedup();
        excludes
            .into_iter()
            .map(|relative| format!("--exclude {}", shell_quote(&format!("./{relative}"))))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[async_trait]
impl SandboxSession for BlaxelSandboxSession {
    fn id(&self) -> &str {
        &self.session_id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "blaxel"
    }

    async fn start(&self) -> SandboxResult<()> {
        self.started.store(true, Ordering::SeqCst);
        if self.preserved_workspace {
            // A reconnected paused sandbox still holds the files the last run left behind, so
            // re-materializing the manifest would overwrite them.
            return Ok(());
        }
        // Blaxel base images run as root and ship no workspace directory, and the manifest
        // materializer runs its first `mkdir` with the workspace root as the working directory,
        // so the root is created from `/` first.
        let root = self.manifest.root().to_string();
        self.mkdir(&root, true).await.map_err(|error| {
            SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!("failed to create the workspace root `{root}`: {error}"),
            )
        })?;
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
        self.ptys.terminate_all().await;
        self.started.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.ptys.terminate_all().await;
        let mut deleted = self.deleted.lock().await;
        self.started.store(false, Ordering::SeqCst);
        if *deleted || self.pause_on_exit {
            // Blaxel keeps a paused sandbox and resumes it on the next connection.
            return Ok(());
        }
        let path = format!("/sandboxes/{}", self.connection.sandbox_name);
        let response = self
            .connection
            .control_request(reqwest::Method::DELETE, &path)
            .timeout(self.connection.timeouts.cleanup)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStopError,
                    format!("`DELETE {path}` failed: {error}"),
                )
            })?;
        let status = response.status();
        if (status.is_client_error() && status.as_u16() != 404) || status.is_server_error() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                format!("`DELETE {path}` failed: HTTP {}", status.as_u16()),
            ));
        }
        *deleted = true;
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        let path = format!("/filesystem/{}", encode_sandbox_path("/"));
        let response = self
            .connection
            .sandbox_request(reqwest::Method::GET, &path)
            .timeout(Duration::from_secs(10))
            .send()
            .await;
        Ok(response.map(|r| r.status().is_success()).unwrap_or(false))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let body = self.process_body(&request)?;
        let timeout = request.timeout.unwrap_or(self.connection.timeouts.exec);
        let response = self
            .connection
            .sandbox_request(reqwest::Method::POST, "/process")
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    SandboxError::new(
                        ErrorCode::ExecTimeout,
                        format!(
                            "Command `{}` timed out after {timeout:?}",
                            request.command.join(" ")
                        ),
                    )
                } else {
                    SandboxError::new(
                        ErrorCode::ExecTransportError,
                        format!("Blaxel exec failed: {error}"),
                    )
                }
            })?;
        let status = response.status();
        let payload = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            let code = if matches!(status.as_u16(), 408 | 504) {
                ErrorCode::ExecTimeout
            } else {
                ErrorCode::ExecTransportError
            };
            return Err(SandboxError::new(
                code,
                format!(
                    "Blaxel exec failed: HTTP {}{}",
                    status.as_u16(),
                    blaxel_error_detail(&payload)
                        .map(|detail| format!(": {detail}"))
                        .unwrap_or_default()
                ),
            ));
        }
        let payload: Value = serde_json::from_slice(&payload).map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Blaxel exec returned a malformed body: {error}"),
            )
        })?;
        Ok(Self::parse_process_response(&payload))
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        self.read_internal(&resolved).await
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path_for_write(path)?;
        self.write_internal(&resolved, contents).await
    }

    /// Resolve an exposed port to a reachable preview endpoint.
    ///
    /// Blaxel mints a preview for any port on demand, so, as in Python's
    /// `_assert_exposed_port_configured` override, the port does not have to be declared at
    /// creation; the control-plane calls live on [`BlaxelConnection`].
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        self.connection.resolve_exposed_port(port).await
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        if resolved == "/" {
            return Ok(());
        }
        // The Blaxel filesystem service creates intermediate directories itself, so `parents`
        // only decides whether a missing parent is an error worth reporting.
        let _ = parents;
        self.filesystem_put(
            &resolved,
            json!({"isDirectory": true, "permissions": "0755"}),
        )
        .await
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        self.remove_internal(&resolved, recursive).await
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        let resolved = self.normalize_path(path)?;
        let url_path = format!("/filesystem/{}", encode_sandbox_path(&resolved));
        let response = self
            .connection
            .sandbox_request(reqwest::Method::GET, &url_path)
            .timeout(self.connection.timeouts.fast_op)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("Failed to list `{resolved}`: {error}"),
                )
            })?;
        let status = response.status();
        let body = response.bytes().await.unwrap_or_default();
        if status.as_u16() == 404 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("`{resolved}` does not exist in the sandbox workspace"),
            ));
        }
        if !status.is_success() {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to list `{resolved}`: HTTP {}", status.as_u16()),
            ));
        }
        let payload: Value = serde_json::from_slice(&body).map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to list `{resolved}`: malformed body: {error}"),
            )
        })?;
        Ok(parse_directory(&payload, &resolved))
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        let root = self.manifest.root().to_string();
        let tar_path = self.tar_path("persist");
        let excludes = self.tar_exclude_args();
        let command = format!(
            "tar {excludes} -C {root} -cf {tar} .",
            root = shell_quote(&root),
            tar = shell_quote(&tar_path),
        );
        // Python wraps the whole `persist_workspace` in `@retry_async` for a timeout or a
        // transient status (`blaxel/sandbox.py` L664); the archive command is the part that can
        // fail that way.
        let result = retry_async(
            RetryPolicy::new(),
            || {
                self.exec(
                    ExecRequest::shell(command.clone())
                        .with_timeout(Some(self.connection.timeouts.workspace_tar)),
                )
            },
            // Python retries a timeout here as well as a transient status.
            |error, _attempt| {
                is_transient_sandbox_error(error) || error.code == ErrorCode::ExecTimeout
            },
        )
        .await?;
        if !result.ok() {
            let _ = self.remove_internal(&tar_path, false).await;
            return Err(SandboxError::new(
                ErrorCode::SnapshotPersistError,
                format!(
                    "Failed to archive workspace `{root}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        let archive = self.read_internal(&tar_path).await;
        let _ = self.remove_internal(&tar_path, false).await;
        archive
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)?;
        let root = self.manifest.root().to_string();
        let tar_path = self.tar_path("hydrate");
        self.mkdir(&root, true).await?;
        self.write_internal(&tar_path, archive)
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    format!("Failed to upload the workspace archive: {error}"),
                )
            })?;
        let command = format!(
            "tar -C {root} -xf {tar}",
            root = shell_quote(&root),
            tar = shell_quote(&tar_path),
        );
        let result = self
            .exec(
                ExecRequest::shell(command)
                    .with_timeout(Some(self.connection.timeouts.workspace_tar)),
            )
            .await;
        let _ = self.remove_internal(&tar_path, false).await;
        let result = result?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!(
                    "Failed to restore the sandbox workspace: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn supports_pty(&self) -> bool {
        // The terminal is addressed with the API key as a query parameter, so a session without
        // one cannot open it.
        self.connection.api_key.is_some()
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        let Some(token) = self.connection.api_key.clone() else {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "a Blaxel PTY session needs an API key: set `BL_API_KEY` or \
                 `BlaxelSandboxClientOptions::with_api_key`",
            ));
        };
        let working_dir = match &request.cwd {
            None => self.manifest.root().to_string(),
            Some(cwd) => self.normalize_path(cwd)?,
        };
        let session_id = format!("pty-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
        let url = terminal_url(
            &self.connection.sandbox_url,
            &token,
            &session_id,
            &working_dir,
            80,
            24,
        );
        self.ptys
            .start(&url, &request, self.connection.timeouts.exec)
            .await
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        self.ptys.write_stdin(&request).await
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        self.ptys.terminate_all().await;
        Ok(())
    }

    fn session_state(&self) -> SandboxSessionState {
        let mut state = SandboxSessionState::new(
            "blaxel",
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.connection.sandbox_name.clone()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default());
        state.snapshot_id = self.snapshot_id.clone();
        state
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }
}

/// Build the `multipart/form-data` body the Blaxel filesystem service accepts for a binary write.
fn multipart_file_body(boundary: &str, path: &str, contents: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(contents.len() + 512);
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    };
    field("permissions", "0644");
    field("path", path);
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"binary-file.bin\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(contents);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// Turn a Blaxel `Directory` payload into workspace entries.
fn parse_directory(payload: &Value, base: &str) -> Vec<FileEntry> {
    let base = base.trim_end_matches('/');
    let mut entries = Vec::new();
    for directory in payload
        .get("subdirectories")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let name = directory
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let path = directory
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{base}/{name}"));
        entries.push(FileEntry {
            name,
            path,
            size: None,
            permissions: None,
            is_dir: true,
            is_symlink: false,
        });
    }
    for file in payload
        .get("files")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let name = file
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let path = file
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{base}/{name}"));
        entries.push(FileEntry {
            name,
            path,
            size: file.get("size").and_then(Value::as_u64),
            permissions: file
                .get("permissions")
                .and_then(Value::as_str)
                .and_then(Permissions::from_ls_field),
            is_dir: false,
            // The directory listing endpoint reports no link metadata.
            is_symlink: false,
        });
    }
    entries
}

/// Read a control-plane response, mapping an error status onto `code`.
async fn parse_control_plane_body(
    response: reqwest::Response,
    operation: &str,
    code: ErrorCode,
) -> SandboxResult<Value> {
    let status = response.status();
    let body = response.bytes().await.unwrap_or_default();
    if !status.is_success() {
        return Err(SandboxError::new(
            code,
            format!(
                "`{operation}` failed: HTTP {}{}",
                status.as_u16(),
                blaxel_error_detail(&body)
                    .map(|detail| format!(": {detail}"))
                    .unwrap_or_default()
            ),
        ));
    }
    if body.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(&body).map_err(|error| {
        SandboxError::new(
            code,
            format!("`{operation}` returned a malformed body: {error}"),
        )
    })
}

/// Render Blaxel's structured error payload (`{"error": {"code", "message"}}`).
fn blaxel_error_detail(body: &[u8]) -> Option<String> {
    let payload: Value = serde_json::from_slice(body).ok()?;
    let error = payload.get("error")?;
    if let Some(message) = error.as_str() {
        return Some(message.to_string());
    }
    let code = error.get("code").and_then(Value::as_str);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.get("error").and_then(Value::as_str));
    match (code, message) {
        (Some(code), Some(message)) => Some(format!("{code}: {message}")),
        (Some(code), None) => Some(code.to_string()),
        (None, Some(message)) => Some(message.to_string()),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(manifest: Manifest) -> BlaxelSandboxSession {
        let connection = Arc::new(BlaxelConnection {
            http: reqwest::Client::new(),
            sandbox_name: "agents-abc".to_string(),
            sandbox_url: "https://sbx.example".to_string(),
            api_base_url: DEFAULT_API_BASE_URL.to_string(),
            workspace: "acme".to_string(),
            api_key: Some("key".to_string()),
            timeouts: BlaxelTimeouts::default(),
            exposed_port_public: true,
            exposed_port_url_ttl_secs: 3600,
        });
        BlaxelSandboxSession::new(connection, manifest)
            .with_base_env_vars(BTreeMap::from([("BASE".to_string(), "1".to_string())]))
    }

    #[test]
    fn builds_the_process_body() {
        let session = session(
            Manifest::new()
                .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
                .with_env("FOO", "bar"),
        );
        let body = session
            .process_body(
                &ExecRequest::argv(["ls", "-la", "/workspace/a b"])
                    .with_cwd(Some("repo".into()))
                    .with_timeout(Some(Duration::from_secs(5))),
            )
            .unwrap();
        assert_eq!(body["command"], "ls -la '/workspace/a b'");
        assert_eq!(body["workingDir"], "/workspace/repo");
        assert_eq!(body["waitForCompletion"], true);
        assert_eq!(body["timeout"], 5000);
        assert_eq!(body["env"], json!({"BASE": "1", "FOO": "bar"}));
    }

    #[test]
    fn encodes_absolute_filesystem_paths() {
        assert_eq!(
            encode_sandbox_path("/workspace/a b.txt"),
            "%2Fworkspace/a%20b.txt"
        );
        assert_eq!(encode_sandbox_path("/"), "%2F");
        assert_eq!(encode_sandbox_path("rel/path"), "rel/path");
    }

    #[test]
    fn parses_split_and_combined_process_output() {
        let split = BlaxelSandboxSession::parse_process_response(
            &json!({"exitCode": 2, "stdout": "out", "stderr": "err"}),
        );
        assert_eq!(split.exit_code, 2);
        assert_eq!(split.stdout_text(), "out");
        assert_eq!(split.stderr_text(), "err");

        let failed =
            BlaxelSandboxSession::parse_process_response(&json!({"exitCode": 1, "logs": "boom"}));
        assert_eq!(failed.stderr_text(), "boom");
        assert!(failed.stdout.is_empty());

        let ok =
            BlaxelSandboxSession::parse_process_response(&json!({"exitCode": 0, "logs": "hi"}));
        assert_eq!(ok.stdout_text(), "hi");
    }

    #[test]
    fn parses_directory_listings() {
        let entries = parse_directory(
            &json!({
                "name": "workspace",
                "path": "/workspace",
                "subdirectories": [{"name": "repo", "path": "/workspace/repo"}],
                "files": [{
                    "name": "task.md",
                    "path": "/workspace/task.md",
                    "size": 12,
                    "permissions": "-rw-r--r--",
                }],
            }),
            "/workspace",
        );
        assert_eq!(entries.len(), 2);
        assert!(entries[0].is_dir);
        assert_eq!(entries[0].path, "/workspace/repo");
        assert_eq!(entries[1].size, Some(12));
        assert!(entries[1].permissions.is_some());
    }

    #[test]
    fn builds_a_multipart_write_body() {
        let body = multipart_file_body("BOUND", "/workspace/a.bin", b"\x00\x01");
        let rendered = String::from_utf8_lossy(&body);
        assert!(
            rendered.contains("--BOUND\r\nContent-Disposition: form-data; name=\"permissions\"")
        );
        assert!(rendered.contains("name=\"path\"\r\n\r\n/workspace/a.bin"));
        assert!(rendered.contains("name=\"file\"; filename=\"binary-file.bin\""));
        assert!(rendered.ends_with("--BOUND--\r\n"));
    }

    #[test]
    fn builds_the_create_body() {
        let client = BlaxelSandboxClient::new(
            BlaxelSandboxClientOptions::new()
                .with_image("blaxel/base:latest")
                .with_memory(2048)
                .with_region("eu-lon-1")
                .with_env_var("TOKEN", "abc")
                .with_label("team", "agents")
                .with_ttl("1h"),
        );
        let body = client.create_body("agents-1");
        assert_eq!(body["metadata"]["name"], "agents-1");
        assert_eq!(body["metadata"]["labels"]["team"], "agents");
        assert_eq!(body["spec"]["runtime"]["image"], "blaxel/base:latest");
        assert_eq!(body["spec"]["runtime"]["memory"], 2048);
        assert_eq!(body["spec"]["runtime"]["ttl"], "1h");
        assert_eq!(
            body["spec"]["runtime"]["envs"],
            json!([{"name": "TOKEN", "value": "abc"}])
        );
        assert_eq!(body["spec"]["region"], "eu-lon-1");
    }

    #[test]
    fn renders_structured_error_details() {
        assert_eq!(
            blaxel_error_detail(b"{\"error\":{\"code\":\"FORBIDDEN\",\"message\":\"nope\"}}"),
            Some("FORBIDDEN: nope".to_string())
        );
        assert_eq!(
            blaxel_error_detail(b"{\"error\":\"plain\"}"),
            Some("plain".to_string())
        );
        assert_eq!(blaxel_error_detail(b"not json"), None);
    }

    #[test]
    fn excludes_ephemeral_paths_from_the_archive() {
        use crate::sandbox::entries::Entry;

        let session = session(
            Manifest::new()
                .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
                .with_entry("cache", Entry::dir().with_ephemeral(true)),
        );
        assert_eq!(
            session.tar_exclude_args(),
            "--exclude ./.sandbox-blobfuse-cache --exclude ./.sandbox-blobfuse-config \
             --exclude ./.sandbox-mountpoint-env --exclude ./.sandbox-rclone-config \
             --exclude ./cache"
        );
    }
}
