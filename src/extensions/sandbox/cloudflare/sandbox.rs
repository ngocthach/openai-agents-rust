//! A sandbox session backed by a Cloudflare Worker. Mirrors
//! `agents.extensions.sandbox.cloudflare.sandbox`.
//!
//! The Python SDK talks to a deployed Cloudflare Worker over HTTP, so this port speaks the same
//! protocol with `reqwest` instead of `aiohttp`:
//!
//! | Operation | Request |
//! | --- | --- |
//! | create | `POST {worker}/v1/sandbox` → `{"id": "..."}` |
//! | exec | `POST {worker}/v1/sandbox/{id}/exec` with `{argv, timeout_ms}` → SSE |
//! | read | `GET {worker}/v1/sandbox/{id}/file/{path}` |
//! | write | `PUT {worker}/v1/sandbox/{id}/file/{path}` |
//! | running | `GET {worker}/v1/sandbox/{id}/running` → `{"running": bool}` |
//! | persist | `POST {worker}/v1/sandbox/{id}/persist?excludes=...` → tar bytes |
//! | hydrate | `POST {worker}/v1/sandbox/{id}/hydrate` with tar bytes |
//! | mount | `POST {worker}/v1/sandbox/{id}/mount` with `{bucket, mountPath, options}` |
//! | unmount | `POST {worker}/v1/sandbox/{id}/unmount` with `{mountPath}` |
//! | shutdown | `DELETE {worker}/v1/sandbox/{id}` |
//!
//! The worker's PTY endpoint is a WebSocket, which this port does not speak:
//! [`supports_pty`](SandboxSession::supports_pty) stays `false`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::ExecResult;

use super::sse::{SseDecoder, SseLineDecoder};

/// The exec timeout the worker is given when a request carries none.
pub const DEFAULT_EXEC_TIMEOUT: Duration = Duration::from_secs(30);
/// The timeout applied to every non-exec worker request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// The environment variable holding the worker's bearer token.
pub const API_KEY_ENV: &str = "CLOUDFLARE_SANDBOX_API_KEY";

const MAX_ERROR_BODY_CHARS: usize = 2000;

/// Settings used when creating a Cloudflare-backed session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudflareSandboxClientOptions {
    /// The base URL of the deployed sandbox worker.
    pub worker_url: String,
    /// The bearer token, defaulting to `CLOUDFLARE_SANDBOX_API_KEY`.
    pub api_key: Option<String>,
    /// Ports the worker deployment exposes, recorded in the session state.
    pub exposed_ports: Vec<u16>,
    /// The default exec timeout.
    pub exec_timeout: Duration,
    /// The timeout applied to every non-exec request.
    pub request_timeout: Duration,
}

impl CloudflareSandboxClientOptions {
    /// Options for a worker deployment.
    pub fn new(worker_url: impl Into<String>) -> Self {
        CloudflareSandboxClientOptions {
            worker_url: worker_url.into(),
            api_key: None,
            exposed_ports: Vec::new(),
            exec_timeout: DEFAULT_EXEC_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    /// Authenticate with an explicit bearer token.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Record the ports the worker exposes.
    pub fn with_exposed_ports<I: IntoIterator<Item = u16>>(mut self, ports: I) -> Self {
        self.exposed_ports = ports.into_iter().collect();
        self
    }

    /// Set the default exec timeout.
    pub fn with_exec_timeout(mut self, timeout: Duration) -> Self {
        self.exec_timeout = timeout;
        self
    }

    /// Set the timeout applied to every non-exec request.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// The bearer token, falling back to [`API_KEY_ENV`].
    pub fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .filter(|key| !key.is_empty())
    }
}

/// The worker endpoint and credentials behind one sandbox.
///
/// Mount strategies reach it through [`connection_for_session`], because
/// [`MountStrategy`](crate::sandbox::mounts::MountStrategy) only receives a
/// `&dyn SandboxSession`.
#[derive(Debug)]
pub struct CloudflareConnection {
    http: reqwest::Client,
    worker_url: String,
    sandbox_id: String,
    api_key: Option<String>,
    exec_timeout: Duration,
    request_timeout: Duration,
}

impl CloudflareConnection {
    /// The sandbox this connection addresses.
    pub fn sandbox_id(&self) -> &str {
        &self.sandbox_id
    }

    /// The worker base URL, without a trailing slash.
    pub fn worker_url(&self) -> &str {
        &self.worker_url
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/v1/sandbox/{}/{}",
            self.worker_url,
            self.sandbox_id,
            path.trim_start_matches('/')
        )
    }

    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self.http.request(method, url);
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        builder
    }

    /// Attach an R2, S3 or GCS bucket at `mount_path`.
    pub async fn mount_bucket(
        &self,
        bucket: &str,
        mount_path: &str,
        options: Value,
    ) -> SandboxResult<()> {
        let payload = json!({
            "bucket": bucket,
            "mountPath": mount_path,
            "options": options,
        });
        self.post_mount_request("mount", payload, mount_path).await
    }

    /// Detach the bucket mounted at `mount_path`.
    pub async fn unmount_bucket(&self, mount_path: &str) -> SandboxResult<()> {
        let payload = json!({ "mountPath": mount_path });
        self.post_mount_request("unmount", payload, mount_path)
            .await
    }

    async fn post_mount_request(
        &self,
        path: &str,
        payload: Value,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let url = self.url(path);
        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&payload)
            // The mount payload carries bucket credentials, so the failure message below never
            // echoes the request body back into an error a model can read.
            .timeout(self.request_timeout)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!("cloudflare bucket {path} at `{mount_path}` failed: {error}"),
                )
            })?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!("cloudflare bucket {path} at `{mount_path}` failed: HTTP {status}"),
            ));
        }
        Ok(())
    }
}

type ConnectionRegistry = Mutex<Vec<(String, Weak<CloudflareConnection>)>>;

fn registry() -> &'static ConnectionRegistry {
    static REGISTRY: OnceLock<ConnectionRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// The live connection behind a session id, for mount strategies that need the worker endpoint.
///
/// [`MountStrategy`](crate::sandbox::mounts::MountStrategy) receives only a `&dyn SandboxSession`,
/// which cannot hand back a provider endpoint, so each session registers its connection here
/// under its own id while it is alive.
pub fn connection_for_session(session_id: &str) -> Option<Arc<CloudflareConnection>> {
    let guard = registry().lock().ok()?;
    guard
        .iter()
        .find(|(id, _)| id == session_id)
        .and_then(|(_, weak)| weak.upgrade())
}

fn register_connection(session_id: &str, connection: &Arc<CloudflareConnection>) {
    if let Ok(mut guard) = registry().lock() {
        guard.retain(|(_, weak)| weak.strong_count() > 0);
        guard.push((session_id.to_string(), Arc::downgrade(connection)));
    }
}

/// Creates [`CloudflareSandboxSession`]s against a deployed worker.
#[derive(Debug, Clone)]
pub struct CloudflareSandboxClient {
    options: CloudflareSandboxClientOptions,
    snapshot: Option<SnapshotSpec>,
    http: reqwest::Client,
}

impl CloudflareSandboxClient {
    /// A client for a worker deployment.
    pub fn new(options: CloudflareSandboxClientOptions) -> Self {
        CloudflareSandboxClient {
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

    fn worker_url(&self) -> SandboxResult<String> {
        let worker_url = self.options.worker_url.trim_end_matches('/');
        if worker_url.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "CloudflareSandboxClientOptions.worker_url must not be empty",
            ));
        }
        Ok(worker_url.to_string())
    }

    fn check_manifest(manifest: &Manifest) -> SandboxResult<()> {
        if manifest.root() != DEFAULT_MANIFEST_ROOT {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "Cloudflare sandboxes only support a `{DEFAULT_MANIFEST_ROOT}` manifest root \
                     because persistence and hydration are fixed to it, got `{}`",
                    manifest.root()
                ),
            ));
        }
        Ok(())
    }

    fn connection(&self, worker_url: String, sandbox_id: String) -> Arc<CloudflareConnection> {
        Arc::new(CloudflareConnection {
            http: self.http.clone(),
            worker_url,
            sandbox_id,
            api_key: self.options.resolve_api_key(),
            exec_timeout: self.options.exec_timeout,
            request_timeout: self.options.request_timeout,
        })
    }

    /// Ask the worker for a server-generated sandbox id (`POST /v1/sandbox`).
    async fn request_sandbox_id(&self, worker_url: &str) -> SandboxResult<String> {
        let url = format!("{worker_url}/v1/sandbox");
        let mut builder = self.http.post(&url).timeout(self.options.request_timeout);
        if let Some(key) = self.options.resolve_api_key() {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        let response = builder.send().await.map_err(|error| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("POST /sandbox request failed: {error}"),
            )
        })?;
        let status = response.status();
        let body = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                http_error_message(
                    "POST /sandbox",
                    status.as_u16(),
                    format_response_body(&body),
                ),
            ));
        }
        let payload: Value = serde_json::from_slice(&body).map_err(|error| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("POST /sandbox returned a malformed body: {error}"),
            )
        })?;
        match payload.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => Ok(id.to_string()),
            _ => Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "POST /sandbox returned invalid id",
            )),
        }
    }
}

#[async_trait]
impl SandboxClient for CloudflareSandboxClient {
    fn backend_id(&self) -> &str {
        "cloudflare"
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        let worker_url = self.worker_url()?;
        let manifest = manifest.unwrap_or_default();
        Self::check_manifest(&manifest)?;
        let sandbox_id = self.request_sandbox_id(&worker_url).await?;
        let connection = self.connection(worker_url, sandbox_id);
        Ok(Arc::new(
            CloudflareSandboxSession::new(connection, manifest)
                .with_snapshot(self.snapshot.clone())
                .with_exposed_ports(self.options.exposed_ports.clone()),
        ))
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        if state.backend != "cloudflare" {
            return Err(SandboxError::new(
                ErrorCode::SessionStateMismatch,
                format!(
                    "sandbox session state was created by the `{}` backend",
                    state.backend
                ),
            ));
        }
        state.verify_manifest(&manifest)?;
        Self::check_manifest(&manifest)?;
        let worker_url = self.worker_url()?;
        let sandbox_id = state.provider_id.clone().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "cloudflare sandbox session state is missing its sandbox id",
            )
        })?;
        let connection = self.connection(worker_url, sandbox_id);
        let session = CloudflareSandboxSession::new(connection, manifest)
            .with_snapshot(Some(state.snapshot.clone()))
            .with_snapshot_id(state.snapshot_id.clone())
            .with_exposed_ports(state.exposed_ports.clone())
            .resuming();
        if !session.running().await? {
            // The worker no longer serves this id: the replacement sandbox the next
            // `POST /v1/sandbox` creates starts from the snapshot instead.
            let sandbox_id = self
                .request_sandbox_id(session.connection.worker_url())
                .await?;
            let connection =
                self.connection(session.connection.worker_url().to_string(), sandbox_id);
            return Ok(Arc::new(
                CloudflareSandboxSession::new(connection, session.manifest.clone())
                    .with_snapshot(Some(state.snapshot.clone()))
                    .with_snapshot_id(state.snapshot_id.clone())
                    .with_exposed_ports(state.exposed_ports.clone())
                    .resuming(),
            ));
        }
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// A sandbox session whose workspace lives inside a Cloudflare Worker sandbox.
#[derive(Debug)]
pub struct CloudflareSandboxSession {
    connection: Arc<CloudflareConnection>,
    session_id: String,
    manifest: Manifest,
    started: AtomicBool,
    deleted: tokio::sync::Mutex<bool>,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    exposed_ports: Vec<u16>,
    resuming: bool,
}

impl CloudflareSandboxSession {
    /// Attach a session to a sandbox the worker already created.
    pub fn new(connection: Arc<CloudflareConnection>, manifest: Manifest) -> Self {
        // The session id is its own value, as in the Python port: the worker's sandbox id is
        // recorded separately, and a mount must reach the connection of its own session.
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let snapshot_id = session_id.clone();
        register_connection(&session_id, &connection);
        CloudflareSandboxSession {
            connection,
            session_id,
            manifest,
            started: AtomicBool::new(false),
            deleted: tokio::sync::Mutex::new(false),
            snapshot: None,
            snapshot_id,
            exposed_ports: Vec::new(),
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

    /// Record the ports the worker exposes.
    pub fn with_exposed_ports(mut self, exposed_ports: Vec<u16>) -> Self {
        self.exposed_ports = exposed_ports;
        self
    }

    /// Restore the snapshot on start instead of materializing the full manifest.
    pub fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// The worker endpoint behind this session.
    pub fn connection(&self) -> &Arc<CloudflareConnection> {
        &self.connection
    }

    /// The worker's id for this sandbox.
    pub fn sandbox_id(&self) -> &str {
        &self.connection.sandbox_id
    }

    /// The argument vector sent to `POST /exec`.
    ///
    /// The worker's exec body carries only `argv`, so the manifest environment becomes an `env`
    /// prefix and a working directory becomes a `sh -c` wrapper, exactly as a shell would.
    fn exec_argv(&self, request: &ExecRequest) -> SandboxResult<Vec<String>> {
        let mut argv = request.resolved_argv();
        if let Some(cwd) = &request.cwd {
            let workdir = self.normalize_path(cwd)?;
            let mut wrapped = vec![
                "sh".to_string(),
                "-c".to_string(),
                "cd -- \"$1\" && shift && exec \"$@\"".to_string(),
                "sh".to_string(),
                workdir,
            ];
            wrapped.append(&mut argv);
            argv = wrapped;
        }
        let environment = self.manifest.environment();
        if !environment.is_empty() {
            let mut prefixed = vec!["env".to_string()];
            prefixed.extend(
                environment
                    .iter()
                    .map(|(key, value)| format!("{key}={value}")),
            );
            prefixed.append(&mut argv);
            argv = prefixed;
        }
        Ok(argv)
    }

    async fn exec_streamed(&self, argv: &[String], timeout: Duration) -> SandboxResult<ExecResult> {
        let url = self.connection.url("exec");
        let payload = json!({ "argv": argv, "timeout_ms": timeout.as_millis() as u64 });
        let response = self
            .connection
            .request(reqwest::Method::POST, &url)
            .json(&payload)
            .timeout(timeout + Duration::from_secs(5))
            .send()
            .await
            .map_err(|error| exec_error(argv, &error))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                http_error_message("POST /exec", status.as_u16(), format_response_body(&body)),
            ));
        }

        let mut stdout: Vec<u8> = Vec::new();
        let mut stderr: Vec<u8> = Vec::new();
        let mut lines = SseLineDecoder::new();
        let mut events = SseDecoder::new();
        let mut body = response.bytes_stream();
        let mut saw_bytes = false;
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|error| exec_error(argv, &error))?;
            saw_bytes |= !chunk.is_empty();
            for line in lines.decode(&chunk) {
                if let Some(event) = events.decode(&line)
                    && let Some(result) =
                        apply_exec_event(&event.event, &event.data, &mut stdout, &mut stderr, argv)?
                {
                    return Ok(result);
                }
            }
        }
        for line in lines.flush() {
            if let Some(event) = events.decode(&line)
                && let Some(result) =
                    apply_exec_event(&event.event, &event.data, &mut stdout, &mut stderr, argv)?
            {
                return Ok(result);
            }
        }
        let _ = saw_bytes;
        Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "SSE stream ended without exit event",
        ))
    }

    /// Decode a worker body that may arrive either raw or SSE-encoded.
    ///
    /// The worker streams larger payloads as `metadata` / `chunk` / `complete` events, with
    /// binary chunks base64-encoded.
    fn decode_streamed_payload(body: &[u8]) -> SandboxResult<Vec<u8>> {
        if !body.starts_with(b"data: {") {
            return Ok(body.to_vec());
        }
        let mut lines = SseLineDecoder::new();
        let mut events = SseDecoder::new();
        let mut chunks: Vec<u8> = Vec::new();
        let mut is_binary = false;
        let mut saw_metadata = false;
        let mut saw_chunk = false;
        let mut saw_complete = false;

        let mut all_lines = lines.decode(body);
        all_lines.extend(lines.flush());
        for line in all_lines {
            let Some(event) = events.decode(&line) else {
                continue;
            };
            if event.event != "message" || event.data.is_empty() {
                continue;
            }
            let Ok(message) = serde_json::from_str::<Value>(&event.data) else {
                return Ok(body.to_vec());
            };
            match message.get("type").and_then(Value::as_str) {
                Some("metadata") => {
                    is_binary = message
                        .get("isBinary")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    saw_metadata = true;
                }
                Some("chunk") => {
                    if !saw_metadata {
                        return Ok(body.to_vec());
                    }
                    let data = message.get("data").and_then(Value::as_str).unwrap_or("");
                    if is_binary {
                        let decoded = BASE64.decode(data).map_err(|error| {
                            SandboxError::new(
                                ErrorCode::ExecTransportError,
                                format!("worker sent an undecodable chunk: {error}"),
                            )
                        })?;
                        chunks.extend_from_slice(&decoded);
                    } else {
                        chunks.extend_from_slice(data.as_bytes());
                    }
                    saw_chunk = true;
                }
                Some("complete") => {
                    if !saw_metadata {
                        return Ok(body.to_vec());
                    }
                    saw_complete = true;
                }
                _ => {}
            }
        }
        if !saw_metadata || (!saw_chunk && !saw_complete) {
            return Ok(body.to_vec());
        }
        if !saw_complete {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "SSE payload ended without complete event",
            ));
        }
        Ok(chunks)
    }

    fn file_url(&self, resolved: &str) -> String {
        self.connection.url(&format!(
            "file/{}",
            quote_path(resolved.trim_start_matches('/'))
        ))
    }

    async fn archive_request(
        &self,
        path: &str,
        query: &[(String, String)],
        body: Option<Vec<u8>>,
        code: ErrorCode,
    ) -> SandboxResult<Vec<u8>> {
        let url = self.connection.url(path);
        let mut builder = self
            .connection
            .request(reqwest::Method::POST, &url)
            .timeout(self.connection.request_timeout);
        if !query.is_empty() {
            builder = builder.query(query);
        }
        if let Some(body) = body {
            builder = builder
                .header("Content-Type", "application/octet-stream")
                .body(body);
        }
        let response = builder.send().await.map_err(|error| {
            SandboxError::new(code, format!("cloudflare `{path}` request failed: {error}"))
        })?;
        let status = response.status();
        let payload = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            return Err(SandboxError::new(
                code,
                http_error_message(
                    &format!("POST /{path}"),
                    status.as_u16(),
                    format_response_body(&payload),
                ),
            ));
        }
        Self::decode_streamed_payload(&payload)
    }
}

#[async_trait]
impl SandboxSession for CloudflareSandboxSession {
    fn id(&self) -> &str {
        &self.session_id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "cloudflare"
    }

    async fn start(&self) -> SandboxResult<()> {
        // The worker's base image ships no workspace directory, and the manifest materializer
        // runs its first `mkdir` with the workspace root as the working directory, so the root
        // is created from `/` first.
        let root = self.manifest.root().to_string();
        // The worker's exec body carries no working directory, so creating the root needs no
        // `cd` wrapper: the absolute path is enough.
        let result = self
            .exec(ExecRequest::argv(["mkdir", "-p", "--", &root]))
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
        // A Cloudflare sandbox has no pause: the worker keeps it until `shutdown` deletes it.
        self.started.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        let mut deleted = self.deleted.lock().await;
        if *deleted {
            return Ok(());
        }
        let url = format!(
            "{}/v1/sandbox/{}",
            self.connection.worker_url, self.connection.sandbox_id
        );
        let response = self
            .connection
            .request(reqwest::Method::DELETE, &url)
            .timeout(self.connection.request_timeout)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceStopError,
                    format!("`DELETE /sandbox` failed: {error}"),
                )
            })?;
        self.started.store(false, Ordering::SeqCst);
        let status = response.status();
        if status.is_client_error() && status.as_u16() != 404 || status.is_server_error() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                http_error_message("DELETE /sandbox", status.as_u16(), None),
            ));
        }
        *deleted = true;
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        let url = self.connection.url("running");
        let response = self
            .connection
            .request(reqwest::Method::GET, &url)
            .timeout(self.connection.request_timeout)
            .send()
            .await;
        let Ok(response) = response else {
            return Ok(false);
        };
        if !response.status().is_success() {
            return Ok(false);
        }
        let Ok(payload) = response.json::<Value>().await else {
            return Ok(false);
        };
        Ok(payload
            .get("running")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let argv = self.exec_argv(&request)?;
        let timeout = request.timeout.unwrap_or(self.connection.exec_timeout);
        match tokio::time::timeout(
            timeout + Duration::from_secs(5),
            self.exec_streamed(&argv, timeout),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(SandboxError::new(
                ErrorCode::ExecTimeout,
                format!(
                    "Command `{}` timed out after {timeout:?}",
                    request.command.join(" ")
                ),
            )),
        }
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        let url = self.file_url(&resolved);
        let response = self
            .connection
            .request(reqwest::Method::GET, &url)
            .timeout(self.connection.request_timeout)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("Failed to read `{path}`: {error}"),
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
                http_error_message(
                    &format!("GET /file{resolved}"),
                    status.as_u16(),
                    format_response_body(&body),
                ),
            ));
        }
        Self::decode_streamed_payload(&body)
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        let url = self.file_url(&resolved);
        let response = self
            .connection
            .request(reqwest::Method::PUT, &url)
            .header("Content-Type", "application/octet-stream")
            .body(contents.to_vec())
            .timeout(self.connection.request_timeout)
            .send()
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("Failed to write `{path}`: {error}"),
                )
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                http_error_message(
                    &format!("PUT /file{resolved}"),
                    status.as_u16(),
                    format_response_body(&body),
                ),
            ));
        }
        Ok(())
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        let root = self.manifest.root().trim_end_matches('/').to_string();
        let mut excludes: Vec<String> = self
            .manifest
            .ephemeral_paths()
            .into_iter()
            .filter_map(|path| {
                path.strip_prefix(&root)
                    .map(|relative| relative.trim_start_matches('/').to_string())
                    .filter(|relative| !relative.is_empty())
            })
            .collect();
        excludes.sort();
        excludes.dedup();
        let query = if excludes.is_empty() {
            Vec::new()
        } else {
            vec![("excludes".to_string(), excludes.join(","))]
        };
        self.archive_request("persist", &query, None, ErrorCode::SnapshotPersistError)
            .await
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.archive_request(
            "hydrate",
            &[],
            Some(archive.to_vec()),
            ErrorCode::SnapshotRestoreError,
        )
        .await
        .map(|_| ())
    }

    fn session_state(&self) -> SandboxSessionState {
        let mut state = SandboxSessionState::new(
            "cloudflare",
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.connection.sandbox_id.clone()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default());
        state.snapshot_id = self.snapshot_id.clone();
        state.exposed_ports = self.exposed_ports.clone();
        state
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }
}

fn apply_exec_event(
    event: &str,
    data: &str,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    argv: &[String],
) -> SandboxResult<Option<ExecResult>> {
    match event {
        "stdout" | "stderr" => {
            let decoded = BASE64.decode(data).map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("worker sent an undecodable `{event}` event: {error}"),
                )
            })?;
            if event == "stdout" {
                stdout.extend_from_slice(&decoded);
            } else {
                stderr.extend_from_slice(&decoded);
            }
            Ok(None)
        }
        "exit" => {
            let payload: Value = serde_json::from_str(data).map_err(|error| {
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("worker sent a malformed `exit` event: {error}"),
                )
            })?;
            let exit_code = payload
                .get("exit_code")
                .and_then(Value::as_i64)
                .unwrap_or_default() as i32;
            Ok(Some(ExecResult {
                stdout: std::mem::take(stdout),
                stderr: std::mem::take(stderr),
                exit_code,
            }))
        }
        "error" => {
            let payload: Value = serde_json::from_str(data).unwrap_or(Value::Null);
            let detail = payload
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!(
                    "Command `{}` failed in the worker: {detail}",
                    argv.join(" ")
                ),
            ))
        }
        _ => Ok(None),
    }
}

fn exec_error(argv: &[String], error: &reqwest::Error) -> SandboxError {
    if error.is_timeout() {
        return SandboxError::new(
            ErrorCode::ExecTimeout,
            format!("Command `{}` timed out: {error}", argv.join(" ")),
        );
    }
    SandboxError::new(
        ErrorCode::ExecTransportError,
        format!("Cloudflare exec transport failed: {error}"),
    )
}

fn http_error_message(operation: &str, status: u16, detail: Option<String>) -> String {
    match detail {
        Some(detail) => format!("{operation} failed: HTTP {status}: {detail}"),
        None => format!("{operation} failed: HTTP {status}"),
    }
}

/// Render a worker error body, preferring its `code` and `error` fields.
fn format_response_body(body: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(body);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let Ok(payload) = serde_json::from_str::<Value>(trimmed) else {
        return Some(truncate_error_body(trimmed));
    };
    let error = payload.get("error").and_then(Value::as_str);
    let code = payload.get("code").and_then(Value::as_str);
    Some(truncate_error_body(&match (code, error) {
        (Some(code), Some(error)) => format!("{code}: {error}"),
        (None, Some(error)) => error.to_string(),
        _ => trimmed.to_string(),
    }))
}

fn truncate_error_body(value: &str) -> String {
    if value.chars().count() <= MAX_ERROR_BODY_CHARS {
        return value.to_string();
    }
    let head: String = value.chars().take(MAX_ERROR_BODY_CHARS).collect();
    format!("{head}... [truncated]")
}

/// Percent-encode a sandbox path for a worker URL, keeping `/` literal.
///
/// Mirrors Python's `quote(path, safe="/")`.
pub fn quote_path(path: &str) -> String {
    let mut encoded = String::with_capacity(path.len());
    for byte in path.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                encoded.push(*byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(manifest: Manifest) -> CloudflareSandboxSession {
        let connection = Arc::new(CloudflareConnection {
            http: reqwest::Client::new(),
            worker_url: "https://worker.example".to_string(),
            sandbox_id: "sbx-1".to_string(),
            api_key: None,
            exec_timeout: DEFAULT_EXEC_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        });
        CloudflareSandboxSession::new(connection, manifest)
    }

    #[test]
    fn builds_exec_arguments_with_env_and_cwd() {
        let session = session(Manifest::new().with_env("FOO", "bar"));
        let argv = session
            .exec_argv(&ExecRequest::shell("echo hi").with_cwd(Some("repo".into())))
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "env",
                "FOO=bar",
                "sh",
                "-c",
                "cd -- \"$1\" && shift && exec \"$@\"",
                "sh",
                "/workspace/repo",
                "sh",
                "-lc",
                "echo hi",
            ]
        );
    }

    #[test]
    fn builds_file_urls_with_percent_encoded_paths() {
        let session = session(Manifest::new());
        assert_eq!(
            session.file_url("/workspace/a b/c#d.txt"),
            "https://worker.example/v1/sandbox/sbx-1/file/workspace/a%20b/c%23d.txt"
        );
        assert_eq!(quote_path("plain/path-1.txt"), "plain/path-1.txt");
        assert_eq!(quote_path("é"), "%C3%A9");
    }

    #[test]
    fn decodes_an_sse_encoded_file_body() {
        let body = concat!(
            "data: {\"type\":\"metadata\",\"isBinary\":true}\n\n",
            "data: {\"type\":\"chunk\",\"data\":\"aGVsbG8=\"}\n\n",
            "data: {\"type\":\"complete\"}\n\n",
        );
        let decoded = CloudflareSandboxSession::decode_streamed_payload(body.as_bytes()).unwrap();
        assert_eq!(decoded, b"hello");

        // A raw body is returned untouched.
        let raw = CloudflareSandboxSession::decode_streamed_payload(b"plain bytes").unwrap();
        assert_eq!(raw, b"plain bytes");
    }

    #[test]
    fn rejects_a_non_default_manifest_root() {
        let error = CloudflareSandboxClient::check_manifest(&Manifest::new().with_root("/srv"))
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
        assert!(CloudflareSandboxClient::check_manifest(&Manifest::new()).is_ok());
    }

    #[test]
    fn formats_worker_error_bodies() {
        assert_eq!(
            format_response_body(b"{\"error\":\"boom\",\"code\":\"E1\"}"),
            Some("E1: boom".to_string())
        );
        assert_eq!(
            format_response_body(b"{\"error\":\"boom\"}"),
            Some("boom".to_string())
        );
        assert_eq!(format_response_body(b"   "), None);
        assert_eq!(
            format_response_body(b"not json"),
            Some("not json".to_string())
        );
    }
}
