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
//! PTY sessions run over the worker's WebSocket endpoint; see [`super::pty`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::http::read_status_error;
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::retry::{
    RetryPolicy, is_transient_http_status, is_transient_sandbox_error, retry_async,
};

use super::super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, assert_exposed_port_configured,
    exposed_port_unavailable,
};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, live_workspace_matches_snapshot_on_resume,
    restore_session_snapshot,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, SnapshotFingerprint};
use crate::sandbox::types::ExecResult;

use super::pty::{CloudflarePtySessions, pty_url};
use super::sse::{SseDecoder, SseLineDecoder};

/// The exec timeout the worker is given when a request carries none.
pub const DEFAULT_EXEC_TIMEOUT: Duration = Duration::from_secs(30);
/// The timeout applied to every non-exec worker request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// The environment variable holding the worker's bearer token.
pub const API_KEY_ENV: &str = "CLOUDFLARE_SANDBOX_API_KEY";

const MAX_ERROR_BODY_CHARS: usize = 2000;

/// Settings used when creating a Cloudflare-backed session.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
pub struct CloudflareSandboxClientOptions {
    /// The base URL of the deployed sandbox worker.
    pub worker_url: String,
    /// The bearer token, defaulting to `CLOUDFLARE_SANDBOX_API_KEY`.
    pub api_key: Option<String>,
    /// Ports the worker deployment exposes, recorded in the session state.
    #[serde(default)]
    pub exposed_ports: Vec<u16>,
    /// The default exec timeout.
    #[serde(default = "default_exec_timeout")]
    pub exec_timeout: Duration,
    /// The timeout applied to every non-exec request.
    #[serde(default = "default_request_timeout")]
    pub request_timeout: Duration,
}

fn default_exec_timeout() -> Duration {
    DEFAULT_EXEC_TIMEOUT
}

fn default_request_timeout() -> Duration {
    DEFAULT_REQUEST_TIMEOUT
}

// The API key is a credential, so `Debug` never shows it.
impl std::fmt::Debug for CloudflareSandboxClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflareSandboxClientOptions")
            .field("worker_url", &self.worker_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("exposed_ports", &self.exposed_ports)
            .field("exec_timeout", &self.exec_timeout)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
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
/// A mount strategy recovers it through [`SandboxSession::as_any`], because
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
        // The worker holds the bucket credentials for as long as the sandbox lives, so a
        // sandbox attached with them is never reattached from a persisted state: a new session
        // is created from the current trusted configuration instead.
        if state.mount_authority_redacted
            || state.mount_authority_rebound
            || manifest
                .mount_targets()
                .iter()
                .any(|(mount, _)| mount.strategy.carries_credentials(mount))
        {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "Cloudflare sandbox sessions with protected bucket configuration cannot be \
                 resumed; create a new session from current trusted configuration",
            )
            .with_context("backend", "cloudflare"));
        }
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
        let record = state
            .snapshot_fingerprint
            .clone()
            .zip(state.snapshot_fingerprint_version.clone())
            .map(|(fingerprint, version)| SnapshotFingerprint {
                fingerprint,
                version,
            });
        Ok(Arc::new(
            session
                .with_state_preserved(true)
                .with_snapshot_fingerprint(record),
        ))
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
    /// The worker still served this sandbox when the session resumed, so its accounts and
    /// packages survived the pause.
    state_preserved: bool,
    /// A mount transition ended in an unknown state and the sandbox was deleted: the session
    /// refuses further work.
    mount_transition_terminal: AtomicBool,
    /// The workspace fingerprint the stored snapshot was taken from.
    snapshot_fingerprint: std::sync::Mutex<Option<SnapshotFingerprint>>,
    /// Workspace-relative paths a persist leaves out of the archive.
    persist_skip_paths: std::sync::Mutex<std::collections::BTreeSet<String>>,
    ptys: CloudflarePtySessions,
}

impl CloudflareSandboxSession {
    /// Whether the manifest configures a mount whose credentials the worker holds, so provider
    /// text that may echo them must stay out of logs.
    fn has_protected_mount_authority(&self) -> bool {
        self.manifest
            .mount_targets()
            .iter()
            .any(|(mount, _)| mount.strategy.carries_credentials(mount))
    }

    /// Attach a session to a sandbox the worker already created.
    pub fn new(connection: Arc<CloudflareConnection>, manifest: Manifest) -> Self {
        // The session id is its own value, as in the Python port: the worker's sandbox id is
        // recorded separately, and a mount must reach the connection of its own session.
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let snapshot_id = session_id.clone();
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
            state_preserved: false,
            mount_transition_terminal: AtomicBool::new(false),
            snapshot_fingerprint: std::sync::Mutex::new(None),
            persist_skip_paths: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            ptys: CloudflarePtySessions::new(),
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

    /// Carry the workspace fingerprint a resumed snapshot was taken from.
    pub fn with_snapshot_fingerprint(self, record: Option<SnapshotFingerprint>) -> Self {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
        self
    }

    /// Record that the sandbox survived the pause this session resumes from.
    pub fn with_state_preserved(mut self, preserved: bool) -> Self {
        self.state_preserved = preserved;
        self
    }

    /// Refuse work once a mount transition left the sandbox in an unknown state.
    fn ensure_usable(&self) -> SandboxResult<()> {
        if self.mount_transition_terminal.load(Ordering::SeqCst) {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                "sandbox session is unavailable after an ambiguous mount transition",
            )
            .with_context("backend", "cloudflare"));
        }
        Ok(())
    }

    /// Tag a failed `DELETE` with the backend and, when the worker answered, the
    /// status.
    fn delete_failure(&self, error: SandboxError, status: Option<u16>) -> SandboxError {
        let mut error = error.with_context("backend", "cloudflare");
        if let Some(status) = status {
            error = error.with_context("http_status", status.to_string());
        }
        error
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
        let environment = request.resolved_env(self.manifest.environment());
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
            return Err(with_status_retryability(
                SandboxError::new(
                    ErrorCode::ExecTransportError,
                    http_error_message("POST /exec", status.as_u16(), format_response_body(&body)),
                ),
                status.as_u16(),
            ));
        }

        let mut stdout: Vec<u8> = Vec::new();
        let mut raw_stream: Vec<u8> = Vec::new();
        let mut stderr: Vec<u8> = Vec::new();
        let mut lines = SseLineDecoder::new();
        let mut events = SseDecoder::new();
        let mut body = response.bytes_stream();
        let mut saw_bytes = false;
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|error| exec_error(argv, &error))?;
            saw_bytes |= !chunk.is_empty();
            raw_stream.extend_from_slice(&chunk);
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
        // A 200 reply whose body is not an event stream is the worker's error payload; its text
        // is more useful than "ended without exit event".
        let detail = (!raw_stream.is_empty() && !looks_like_sse_stream(&raw_stream))
            .then(|| format_response_body(&raw_stream))
            .flatten();
        let message = match detail {
            Some(detail) => format!("POST /exec returned non-SSE error body: {detail}"),
            None => "SSE stream ended without exit event".to_string(),
        };
        Err(with_status_retryability(
            SandboxError::new(ErrorCode::ExecTransportError, message),
            status.as_u16(),
        ))
    }

    /// Restore the stored snapshot, detaching live bucket mounts first.
    ///
    /// A sandbox the worker kept alive across the pause still has its buckets attached, and
    /// unpacking an archive over an attached bucket would write through it, so each mount is
    /// detached first and `restore_session_snapshot` attaches them again afterwards. A workspace
    /// that already matches the snapshot is not unpacked, so its mounts are left alone.
    async fn restore_snapshot_on_resume(&self) -> SandboxResult<bool> {
        let detached: Vec<(&crate::sandbox::mounts::Mount, String)> = if self.state_preserved {
            self.manifest
                .mount_targets()
                .into_iter()
                .filter(|(mount, _)| !mount.strategy.supports_native_snapshot_detach())
                .collect()
        } else {
            Vec::new()
        };
        if detached.is_empty()
            || !self.snapshot().restorable().await?
            || live_workspace_matches_snapshot_on_resume(self).await
        {
            return restore_session_snapshot(self).await;
        }
        // Detached in order. A failure while re-attaching them afterwards is terminal, so a
        // failing re-attach is never retried here: `start` deletes the sandbox instead.
        let mut torn_down: Vec<&(&crate::sandbox::mounts::Mount, String)> = Vec::new();
        for entry in &detached {
            let (mount, path) = entry;
            if let Err(cause) = mount
                .strategy
                .teardown_for_snapshot(mount, self, path)
                .await
            {
                self.reattach(&torn_down).await;
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!("failed to detach mounts for {}", self.manifest.root()),
                )
                .with_context("path", self.manifest.root().to_string())
                .caused_by(&cause)
                .with_context("cause", cause.message.clone()));
            }
            torn_down.push(entry);
        }
        match restore_session_snapshot(self).await {
            // The restore re-attaches the buckets itself when it succeeds, and a failing
            // re-attach is the terminal case `start` handles.
            Ok(restored) => Ok(restored),
            Err(error)
                if matches!(
                    error.code,
                    ErrorCode::MountFailed | ErrorCode::MountConfigInvalid
                ) =>
            {
                Err(error)
            }
            // Any other failure (the archive, the hydrate) leaves the buckets detached, so they
            // are put back before the error is returned, as Python's transition helper does.
            Err(error) => {
                self.reattach(&torn_down).await;
                Err(error)
            }
        }
    }

    /// Attach the given buckets again, in reverse order, best effort.
    async fn reattach(&self, mounts: &[&(&crate::sandbox::mounts::Mount, String)]) {
        for (mount, path) in mounts.iter().rev().map(|entry| (&entry.0, &entry.1)) {
            if let Err(error) = mount
                .strategy
                .restore_after_snapshot(mount, self, path)
                .await
            {
                log::warn!(target: crate::logger::LOG_TARGET, "failed to re-attach a bucket after a failed resume: {error}");
            }
        }
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

    /// Run one workspace archive request, retrying a transient transport or 5xx failure.
    ///
    /// Python decorates `_persist_workspace_via_http` and `_hydrate_workspace_via_http` with
    /// `@retry_async` (`cloudflare/sandbox.py` L1445, L1488); both reach the worker through this
    /// one request.
    async fn archive_request(
        &self,
        path: &str,
        query: &[(String, String)],
        body: Option<Vec<u8>>,
        code: ErrorCode,
    ) -> SandboxResult<Vec<u8>> {
        retry_async(
            RetryPolicy::new(),
            || self.archive_request_once(path, query, body.clone(), code),
            |error, _attempt| is_transient_sandbox_error(error),
        )
        .await
    }

    async fn archive_request_once(
        &self,
        path: &str,
        query: &[(String, String)],
        body: Option<Vec<u8>>,
        code: ErrorCode,
    ) -> SandboxResult<Vec<u8>> {
        self.ensure_usable()?;
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
                // A connection that never produced a response is worth another attempt.
                .with_retryable(true)
        })?;
        let status = response.status();
        let payload = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            return Err(with_status_retryability(
                SandboxError::new(
                    code,
                    http_error_message(
                        &format!("POST /{path}"),
                        status.as_u16(),
                        format_response_body(&payload),
                    ),
                ),
                status.as_u16(),
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
            match self.restore_snapshot_on_resume().await {
                Ok(false) => self.manifest.apply_ephemeral(self).await.map(|_| ()),
                Ok(true) => Ok(()),
                Err(error) => Err(error),
            }
        } else {
            self.manifest.apply(self).await.map(|_| ())
        };
        if let Err(error) = applied {
            self.started.store(false, Ordering::SeqCst);
            // A mount that failed while a resumed workspace was being re-attached leaves the
            // sandbox half-mounted, so it is deleted rather than reused and the session is
            // unusable from then on. Mirrors Python's `_reapply_ephemeral_manifest_on_resume`.
            // A sandbox that was already deleted stays unusable too.
            if self.resuming {
                let deleted = *self.deleted.lock().await;
                let bare_mount_failure = !deleted
                    && matches!(
                        error.code,
                        ErrorCode::MountFailed | ErrorCode::MountConfigInvalid
                    );
                if deleted || bare_mount_failure {
                    self.mount_transition_terminal.store(true, Ordering::SeqCst);
                }
                if bare_mount_failure && let Err(stop) = self.shutdown().await {
                    return Err(stop.with_context("reason", "terminal_delete_failed"));
                }
            }
            return Err(error);
        }
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        // A Cloudflare sandbox has no pause: the worker keeps it until `shutdown` deletes it.
        self.ptys.terminate_all().await;
        self.started.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.ptys.terminate_all().await;
        let mut deleted = self.deleted.lock().await;
        if *deleted {
            return Ok(());
        }
        let url = format!(
            "{}/v1/sandbox/{}",
            self.connection.worker_url, self.connection.sandbox_id
        );
        // Python swallows a failed delete (a debug log with no provider text) so teardown never
        // fails a finished run; only a terminal mount transition surfaces it, because there the
        // sandbox has to be gone.
        let terminal = self.mount_transition_terminal.load(Ordering::SeqCst);
        let response = self
            .connection
            .request(reqwest::Method::DELETE, &url)
            .timeout(self.connection.request_timeout)
            .send()
            .await;
        self.started.store(false, Ordering::SeqCst);
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                if terminal {
                    return Err(self.delete_failure(
                        SandboxError::new(
                            ErrorCode::WorkspaceStopError,
                            format!("`DELETE /sandbox` failed: {error}"),
                        ),
                        None,
                    ));
                }
                if self.has_protected_mount_authority() {
                    log::debug!(target: crate::logger::LOG_TARGET, "Failed to delete Cloudflare sandbox on shutdown");
                } else {
                    crate::logger::log_tool_action_debug(
                        "Failed to delete Cloudflare sandbox on shutdown",
                        &error,
                    );
                }
                return Ok(());
            }
        };
        let status = response.status();
        if status.is_client_error() && status.as_u16() != 404 || status.is_server_error() {
            if terminal {
                // The reply body is never read: after an ambiguous mount transition it may echo
                // the mount request, which carries bucket credentials.
                return Err(self.delete_failure(
                    SandboxError::new(
                        ErrorCode::WorkspaceStopError,
                        http_error_message("DELETE /sandbox", status.as_u16(), None),
                    ),
                    Some(status.as_u16()),
                ));
            }
            // The reply body may echo a mount request, so it is read only when neither the
            // tool-data policy nor protected mount authority forbids logging it.
            if self.has_protected_mount_authority() || crate::debug::dont_log_tool_data() {
                log::debug!(target: crate::logger::LOG_TARGET, "Failed to delete Cloudflare sandbox on shutdown");
            } else {
                let body = response.bytes().await.unwrap_or_default();
                log::debug!(
                    target: crate::logger::LOG_TARGET,
                    "Failed to delete Cloudflare sandbox on shutdown: {}",
                    http_error_message(
                        "DELETE /sandbox",
                        status.as_u16(),
                        format_response_body(&body)
                    )
                );
            }
            return Ok(());
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
        self.ensure_usable()?;
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

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        // The worker's own filesystem decides: a workspace symlink can lead out of the root, or
        // onto a read-only grant.
        let resolved = self.validate_remote_path_access(path, true, false).await?;
        let mut command = vec!["mkdir".to_string()];
        if parents {
            command.push("-p".to_string());
        }
        command.extend(["--".to_string(), resolved.clone()]);
        let result = self.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            self.check_mkdir_access(&resolved, parents, None).await?;
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.ensure_usable()?;
        // The worker's own filesystem decides: a workspace symlink can lead out of the root.
        let resolved = self.validate_remote_path_access(path, false, false).await?;
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
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                http_error_message(
                    &format!("GET /file{resolved}"),
                    status.as_u16(),
                    format_response_body(&body),
                ),
            ));
        }
        // A body that fails part-way is an error, never an empty file.
        let body = response.bytes().await.map_err(|error| {
            SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("Failed to read `{path}`: {error}"),
            )
        })?;
        Self::decode_streamed_payload(&body)
    }

    /// Read at most `max_bytes`, bounding what the worker is allowed to send as well.
    ///
    /// Mirrors `CloudflareSandboxSession._read_bounded`. The worker answers a file either as raw
    /// bytes or as an SSE stream whose base64 chunks are several times the size of the file, so
    /// bounding only the decoded result would still let the sandbox decide how much this process
    /// buffers. The wire budget is `8 * max_bytes + 65536`, and a body that exceeds it is
    /// refused with `reason = "bounded_read_wire_limit"` — the convention the core
    /// [`read_bounded`](SandboxSession::read_bounded) doc names — rather than decoded.
    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        self.ensure_usable()?;
        if max_bytes == 0 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "read_bounded max_bytes must be positive",
            ));
        }
        let resolved = self.validate_remote_path_access(path, false, false).await?;
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
                .with_retryable(true)
            })?;
        let status = response.status();
        if status.as_u16() == 404 {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("`{resolved}` does not exist in the sandbox workspace"),
            ));
        }
        if !status.is_success() {
            // A 403 is the worker refusing the path, which will never start working. The body
            // is not read: a bounded read's failure carries the status alone.
            let retryable = status.as_u16() != 403 && is_transient_http_status(status.as_u16());
            return Err(
                read_status_error(&format!("read `{resolved}`"), status, Some(retryable))
                    .with_context("path", resolved),
            );
        }
        // One byte over the budget is enough to prove it was exceeded, so the rest of an
        // oversized body is never pulled off the socket.
        let wire_limit = max_bytes.saturating_mul(8).saturating_add(65536);
        let mut body: Vec<u8> = Vec::new();
        let mut stream = response.bytes_stream();
        let mut over_budget = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                SandboxError::new(
                    ErrorCode::WorkspaceArchiveReadError,
                    format!("Failed to read `{path}`: {error}"),
                )
                .with_retryable(true)
            })?;
            body.extend_from_slice(&chunk);
            // The budget only applies once the prefix shows this is an SSE body; a raw body is
            // bounded by `max_bytes` itself.
            let budget = if body.len() >= 7 && !body.starts_with(b"data: {") {
                max_bytes
            } else {
                wire_limit + 1
            };
            if body.len() >= budget {
                over_budget = body.len() > wire_limit;
                body.truncate(budget);
                break;
            }
        }
        if body.starts_with(b"data: {") && over_budget {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceArchiveReadError,
                format!("`{resolved}` exceeds the bounded-read wire limit of {wire_limit} bytes"),
            )
            .with_context("reason", "bounded_read_wire_limit")
            .with_context("path", resolved)
            .with_retryable(false));
        }
        let mut decoded = Self::decode_streamed_payload(&body)?;
        decoded.truncate(max_bytes);
        Ok(decoded)
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.ensure_usable()?;
        let resolved = self.validate_remote_path_access(path, true, false).await?;
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
        let mut excludes: Vec<String> =
            crate::sandbox::session::workspace_archive_excludes(&self.manifest, &root)
                .into_iter()
                .map(|e| e.trim_start_matches("./").to_string())
                .collect();
        // Paths other components asked to keep out of the archive, such as a mount's scratch
        // directory.
        excludes.extend(
            self.persist_workspace_skip_paths()
                .into_iter()
                .map(|path| path.trim_start_matches("./").to_string()),
        );
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

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        crate::sandbox::archive::validate_hydrate_archive(archive)?;
        self.archive_request(
            "hydrate",
            &[],
            Some(archive.to_vec()),
            ErrorCode::SnapshotRestoreError,
        )
        .await
        .map(|_| ())
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.exposed_ports.clone()
    }

    /// Always refuses: the Cloudflare sandbox worker has no port-resolution endpoint.
    ///
    /// Mirrors `CloudflareSandboxSession._resolve_exposed_port`, which raises for every port
    /// rather than guessing a URL. Declaring a port is still meaningful — the worker opens it
    /// inside the sandbox — but nothing outside the Worker can address it. The override exists
    /// only to carry Python's explanatory detail; the refusal itself matches the trait default.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        assert_exposed_port_configured("cloudflare", port, &self.exposed_ports)?;
        Err(exposed_port_unavailable(
            "cloudflare",
            port,
            &self.exposed_ports,
            ExposedPortUnavailable::BackendUnavailable,
        )
        .with_context(
            "detail",
            "the Cloudflare sandbox worker exposes no port-resolution endpoint",
        ))
    }

    /// The sandbox's own filesystem decides every path, so `rm`, `ls` and `mv` resolve it first.
    fn resolves_paths_remotely(&self) -> bool {
        true
    }

    fn system_state_preserved_on_start(&self) -> bool {
        self.state_preserved
    }

    fn persist_skip_registry(
        &self,
    ) -> Option<&std::sync::Mutex<std::collections::BTreeSet<String>>> {
        Some(&self.persist_skip_paths)
    }

    fn supports_pty(&self) -> bool {
        true
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        // The worker's terminal takes a command line and no working directory, so an explicit
        // `cwd` becomes a `cd` in front of it.
        let cwd = match &request.cwd {
            None => None,
            Some(cwd) => Some(self.normalize_path(cwd)?),
        };
        self.ptys
            .start(
                &pty_url(
                    &self.connection.worker_url,
                    &self.connection.sandbox_id,
                    80,
                    24,
                ),
                self.connection.api_key.as_deref(),
                &request,
                cwd.as_deref(),
            )
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
            "cloudflare",
            &self.session_id,
            self.manifest.root(),
            &self.manifest,
        )
        .with_provider_id(Some(self.connection.sandbox_id.clone()))
        .with_snapshot(self.snapshot.clone().unwrap_or_default());
        state.snapshot_id = self.snapshot_id.clone();
        state.exposed_ports = self.exposed_ports.clone();
        state.with_snapshot_fingerprint(self.snapshot_fingerprint.lock().expect("poisoned").clone())
    }

    fn resume_snapshot_fingerprint(&self) -> Option<SnapshotFingerprint> {
        // Only a sandbox the worker kept alive can still hold the workspace the fingerprint
        // describes; a replacement starts empty, so the restore must not be skipped for it.
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

/// Whether the worker documents a status as retryable.
///
/// Mirrors Python's `_cloudflare_retryability_for_status`: 400 and 500 fail fast, 503 is the
/// worker's "try again" signal, and any other status is not classified.
fn retryability_for_status(status: u16) -> Option<bool> {
    match status {
        400 | 500 => Some(false),
        503 => Some(true),
        _ => None,
    }
}

/// Attach the worker's documented retry classification of `status` to `error`.
fn with_status_retryability(mut error: SandboxError, status: u16) -> SandboxError {
    error.retryable = retryability_for_status(status);
    error
}

/// Mirrors Python's `_looks_like_sse_stream`.
fn looks_like_sse_stream(body: &[u8]) -> bool {
    let text = String::from_utf8_lossy(body);
    let text = text.trim_start();
    ["event:", "data:", "id:", "retry:", ":"]
        .iter()
        .any(|prefix| text.starts_with(prefix))
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
