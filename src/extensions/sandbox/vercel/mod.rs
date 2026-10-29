//! The Vercel Sandbox backend (<https://vercel.com/docs/vercel-sandbox>). Port of
//! `agents.extensions.sandbox.vercel`.
//!
//! # Design
//!
//! Python wraps the `vercel` SDK through its own `ProviderSandbox` adapter. This port calls the
//! Sandbox REST API directly with `reqwest`. One base URL (`https://vercel.com/api` by default)
//! serves everything, every request carries `Authorization: Bearer <token>` and, when configured,
//! a `teamId` query parameter:
//!
//! - `POST /v3/sandboxes` creates a sandbox and its first runtime session.
//! - `/v2/sandboxes/sessions/{id}/cmd` runs commands; with `wait` and `logs` the response is an
//!   NDJSON stream of the command record, its output lines and the finished record.
//! - `/v2/sandboxes/sessions/{id}/fs/read`, `fs/write` and `fs/mkdir` move files. `fs/write`
//!   takes a **gzipped tar archive** whose entry names are relative to the `x-cwd` header, so
//!   this module contains a small `ustar` writer and a gzip framer (stored deflate blocks plus a
//!   CRC-32), avoiding a compression dependency the crate does not have.
//! - `/v2/sandboxes/sessions/{id}/snapshot` takes a native snapshot, and a new sandbox can be
//!   created from `source: { type: "snapshot", snapshotId }`.
//!
//! ```no_run
//! use openai_agents_sdk::extensions::sandbox::vercel::{
//!     VercelSandboxClient, VercelSandboxClientOptions,
//! };
//! use openai_agents_sdk::sandbox::SandboxClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads `VERCEL_OIDC_TOKEN` or `VERCEL_TOKEN` from the environment, as the Python SDK does.
//! let client = VercelSandboxClient::with_options(VercelSandboxClientOptions::new());
//! let session = client.create(None).await?;
//! session.start().await?;
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Divergences from Python
//!
//! - Vercel sandboxes have no PTY, so [`supports_pty`] stays `false`, exactly as in Python. The
//!   REST API does expose `cmd` with `attachStdin` and a stdin endpoint; that streaming mode is
//!   not ported, because the Python backend does not expose it either.
//! - Python refuses any `user` argument. This port maps `root` to the API's `sudo` flag, so the
//!   provider's own `mount-s3` mount strategy can run, and still rejects every other user.
//! - OIDC token refresh, drives, forks and exposed-route resolution are not ported.
//!
//! [`supports_pty`]: crate::sandbox::SandboxSession::supports_pty

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::mounts::{Mount, MountCredentialAuthority, MountPatternConfig, MountStrategy};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, restore_session_snapshot, shell_quote,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::{ExecResult, User};

use super::e2b::{
    adopt_recorded_root, decode_snapshot_ref, encode_snapshot_ref, expect_success, http_error,
    parse_json, transport, workspace_excludes,
};

pub mod tar_gz;

use self::tar_gz::TarGzBuilder;

/// The backend identifier reported by this client and its sessions.
pub const BACKEND_ID: &str = "vercel";
/// Environment variable holding a Vercel OIDC token, preferred over the personal token.
pub const VERCEL_OIDC_TOKEN_ENV: &str = "VERCEL_OIDC_TOKEN";
/// Environment variable holding a Vercel personal access token.
pub const VERCEL_TOKEN_ENV: &str = "VERCEL_TOKEN";
/// Environment variable naming the Vercel project.
pub const VERCEL_PROJECT_ID_ENV: &str = "VERCEL_PROJECT_ID";
/// Environment variable naming the Vercel team.
pub const VERCEL_TEAM_ID_ENV: &str = "VERCEL_TEAM_ID";
/// The default Sandbox API base URL.
pub const DEFAULT_VERCEL_API_BASE_URL: &str = "https://vercel.com/api";
/// The workspace root Vercel sessions default to, matching Python.
pub const DEFAULT_VERCEL_WORKSPACE_ROOT: &str = "/vercel/sandbox";
/// The default sandbox execution time limit, in milliseconds, matching Python.
pub const DEFAULT_VERCEL_SANDBOX_TIMEOUT_MS: u64 = 270_000;
/// How long a freshly created sandbox is given to report `running`.
pub const DEFAULT_WAIT_FOR_RUNNING: Duration = Duration::from_secs(45);

/// How a session persists its workspace into a [`Snapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspacePersistence {
    /// Archive the workspace with `tar`, so the bytes are portable across backends.
    #[default]
    Tar,
    /// Take a native Vercel snapshot and store only a reference to it.
    Snapshot,
}

/// The network policy applied to a sandbox. Mirrors `agents.extensions.sandbox.vercel`'s
/// `NetworkPolicy`, kept as the released Agents JSON shape rather than the provider's types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// Allow all egress.
    AllowAll,
    /// Deny all egress.
    DenyAll,
    /// Allow the listed destinations only.
    Custom {
        /// Domains egress may reach.
        allow: Vec<String>,
        /// Subnets egress may reach.
        allow_subnets: Vec<String>,
        /// Subnets egress may never reach.
        deny_subnets: Vec<String>,
    },
}

impl NetworkPolicy {
    /// The JSON body the API expects.
    fn to_api(&self) -> Value {
        match self {
            NetworkPolicy::AllowAll => json!({ "allow": "all" }),
            NetworkPolicy::DenyAll => json!({ "allow": [] }),
            NetworkPolicy::Custom {
                allow,
                allow_subnets,
                deny_subnets,
            } => {
                let mut policy = json!({ "allow": allow });
                if !allow_subnets.is_empty() || !deny_subnets.is_empty() {
                    policy.as_object_mut().expect("object").insert(
                        "subnets".into(),
                        json!({ "allow": allow_subnets, "deny": deny_subnets }),
                    );
                }
                policy
            }
        }
    }
}

/// Resource limits for a Vercel sandbox.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VercelSandboxResources {
    /// Virtual CPUs.
    pub vcpus: Option<u32>,
}

impl VercelSandboxResources {
    /// Resources with nothing pinned.
    pub fn new() -> Self {
        VercelSandboxResources::default()
    }

    /// Request virtual CPUs.
    pub fn with_vcpus(mut self, vcpus: u32) -> Self {
        self.vcpus = Some(vcpus);
        self
    }
}

/// Settings used when creating a Vercel-backed session.
#[derive(Clone, PartialEq, Eq)]
pub struct VercelSandboxClientOptions {
    /// Bearer token. Falls back to `VERCEL_OIDC_TOKEN`, then `VERCEL_TOKEN`.
    pub token: Option<String>,
    /// Sandbox API base URL. Defaults to [`DEFAULT_VERCEL_API_BASE_URL`].
    pub base_url: Option<String>,
    /// Project the sandbox belongs to. Falls back to `VERCEL_PROJECT_ID`.
    pub project_id: Option<String>,
    /// Team the sandbox belongs to. Falls back to `VERCEL_TEAM_ID`.
    pub team_id: Option<String>,
    /// Execution time limit in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Runtime image alias, such as `node24`; `None` uses the API default.
    pub runtime: Option<String>,
    /// Resource limits.
    pub resources: VercelSandboxResources,
    /// Environment variables set when the sandbox is created.
    pub env: BTreeMap<String, String>,
    /// Ports the sandbox exposes.
    pub exposed_ports: Vec<u16>,
    /// How the session persists its workspace.
    pub workspace_persistence: WorkspacePersistence,
    /// Snapshot expiration in milliseconds.
    pub snapshot_expiration_ms: Option<u64>,
    /// Network policy applied to the sandbox.
    pub network_policy: Option<NetworkPolicy>,
    /// How long a created sandbox is given to report `running`.
    pub wait_for_running: Duration,
    /// How often the client polls the session status while waiting.
    pub status_poll_interval: Duration,
}

impl Default for VercelSandboxClientOptions {
    fn default() -> Self {
        VercelSandboxClientOptions {
            token: None,
            base_url: None,
            project_id: None,
            team_id: None,
            timeout_ms: Some(DEFAULT_VERCEL_SANDBOX_TIMEOUT_MS),
            runtime: None,
            resources: VercelSandboxResources::default(),
            env: BTreeMap::new(),
            exposed_ports: Vec::new(),
            workspace_persistence: WorkspacePersistence::default(),
            snapshot_expiration_ms: None,
            network_policy: None,
            wait_for_running: DEFAULT_WAIT_FOR_RUNNING,
            status_poll_interval: Duration::from_millis(500),
        }
    }
}

// The bearer token must never reach a log line through the `SandboxClient: Debug` bound.
impl std::fmt::Debug for VercelSandboxClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VercelSandboxClientOptions")
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("project_id", &self.project_id)
            .field("team_id", &self.team_id)
            .field("timeout_ms", &self.timeout_ms)
            .field("runtime", &self.runtime)
            .field("resources", &self.resources)
            .field("env", &self.env.keys().collect::<Vec<_>>())
            .field("exposed_ports", &self.exposed_ports)
            .field("workspace_persistence", &self.workspace_persistence)
            .field("snapshot_expiration_ms", &self.snapshot_expiration_ms)
            .field("network_policy", &self.network_policy)
            .finish()
    }
}

impl VercelSandboxClientOptions {
    /// Options that read the token and scopes from the environment.
    pub fn new() -> Self {
        VercelSandboxClientOptions::default()
    }

    /// Use an explicit bearer token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Override the Sandbox API base URL.
    ///
    /// Tests use this to drive the client against a local HTTP server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Scope the sandbox to a project.
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Scope the sandbox to a team.
    pub fn with_team_id(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }

    /// Set the execution time limit in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Select the runtime image alias.
    pub fn with_runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = Some(runtime.into());
        self
    }

    /// Pin the sandbox resources.
    pub fn with_resources(mut self, resources: VercelSandboxResources) -> Self {
        self.resources = resources;
        self
    }

    /// Set an environment variable at sandbox creation.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Expose a port.
    pub fn with_exposed_port(mut self, port: u16) -> Self {
        self.exposed_ports.push(port);
        self
    }

    /// Select how the session persists its workspace.
    pub fn with_workspace_persistence(mut self, persistence: WorkspacePersistence) -> Self {
        self.workspace_persistence = persistence;
        self
    }

    /// Expire native snapshots after this many milliseconds.
    pub fn with_snapshot_expiration_ms(mut self, expiration_ms: u64) -> Self {
        self.snapshot_expiration_ms = Some(expiration_ms);
        self
    }

    /// Apply a network policy.
    pub fn with_network_policy(mut self, policy: NetworkPolicy) -> Self {
        self.network_policy = Some(policy);
        self
    }

    /// How long a created sandbox is given to report `running`.
    pub fn with_wait_for_running(mut self, wait: Duration) -> Self {
        self.wait_for_running = wait;
        self
    }

    /// How often the client polls the session status while waiting.
    pub fn with_status_poll_interval(mut self, interval: Duration) -> Self {
        self.status_poll_interval = interval;
        self
    }

    /// The resolved API base URL.
    fn api_base(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_VERCEL_API_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    /// The bearer token, from the options or the environment.
    fn resolved_token(&self) -> SandboxResult<String> {
        // An explicitly empty token is a misconfiguration, not a request to read the environment.
        if let Some(token) = &self.token {
            return if token.is_empty() {
                Err(missing_token())
            } else {
                Ok(token.clone())
            };
        }
        env_var(VERCEL_OIDC_TOKEN_ENV)
            .or_else(|| env_var(VERCEL_TOKEN_ENV))
            .ok_or_else(missing_token)
    }

    fn resolved_project_id(&self) -> Option<String> {
        self.project_id
            .clone()
            .or_else(|| env_var(VERCEL_PROJECT_ID_ENV))
    }

    fn resolved_team_id(&self) -> Option<String> {
        self.team_id.clone().or_else(|| env_var(VERCEL_TEAM_ID_ENV))
    }
}

/// The error raised when no usable Vercel token is configured.
fn missing_token() -> SandboxError {
    SandboxError::new(
        ErrorCode::SandboxConfigInvalid,
        format!(
            "the Vercel sandbox backend needs a token: set `{VERCEL_OIDC_TOKEN_ENV}` or \
             `{VERCEL_TOKEN_ENV}`, or use `VercelSandboxClientOptions::with_token`"
        ),
    )
}

/// Read a non-empty environment variable.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// The sandbox and session a create or resume call returns.
#[derive(Debug, Clone, Deserialize)]
struct SandboxAndSession {
    sandbox: SandboxRecord,
    session: SessionRecord,
}

/// The sandbox half of a create response.
#[derive(Debug, Clone, Deserialize)]
struct SandboxRecord {
    name: String,
}

/// The runtime session half of a create response.
#[derive(Debug, Clone, Deserialize)]
struct SessionRecord {
    id: String,
    #[serde(default)]
    status: Option<String>,
}

/// Creates [`VercelSandboxSession`]s against the Vercel Sandbox REST API.
#[derive(Debug, Clone)]
pub struct VercelSandboxClient {
    options: VercelSandboxClientOptions,
    http: reqwest::Client,
    snapshot: Option<SnapshotSpec>,
}

impl Default for VercelSandboxClient {
    fn default() -> Self {
        VercelSandboxClient::with_options(VercelSandboxClientOptions::default())
    }
}

impl VercelSandboxClient {
    /// A client reading its credentials from the environment.
    pub fn new() -> Self {
        VercelSandboxClient::default()
    }

    /// A client using explicit options.
    pub fn with_options(options: VercelSandboxClientOptions) -> Self {
        VercelSandboxClient {
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
    pub fn options(&self) -> &VercelSandboxClientOptions {
        &self.options
    }

    async fn create_sandbox(
        &self,
        token: &str,
        envs: &BTreeMap<String, String>,
        source: Option<Value>,
    ) -> SandboxResult<SandboxAndSession> {
        let name = format!("agents-{}", uuid::Uuid::new_v4().simple());
        let mut body = json!({ "name": name, "persistent": false });
        let map = body.as_object_mut().expect("object");
        if let Some(project_id) = self.options.resolved_project_id() {
            map.insert("projectId".into(), json!(project_id));
        }
        if !self.options.exposed_ports.is_empty() {
            map.insert("ports".into(), json!(self.options.exposed_ports));
        }
        if let Some(source) = source {
            map.insert("source".into(), source);
        }
        if let Some(timeout_ms) = self.options.timeout_ms {
            map.insert("timeout".into(), json!(timeout_ms));
        }
        if let Some(vcpus) = self.options.resources.vcpus {
            map.insert("resources".into(), json!({ "vcpus": vcpus }));
        }
        if let Some(runtime) = &self.options.runtime {
            map.insert("runtime".into(), json!(runtime));
        }
        if let Some(policy) = &self.options.network_policy {
            map.insert("networkPolicy".into(), policy.to_api());
        }
        if !envs.is_empty() {
            map.insert("env".into(), json!(envs));
        }
        if let Some(expiration) = self.options.snapshot_expiration_ms {
            map.insert("snapshotExpiration".into(), json!(expiration));
        }
        // `/v3/sandboxes` is the current endpoint; `/v2` is kept for an explicit runtime alias.
        let path = if self.options.runtime.is_some() {
            "/v2/sandboxes"
        } else {
            "/v3/sandboxes"
        };
        let response = self
            .request(
                self.http.post(format!("{}{path}", self.options.api_base())),
                token,
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| transport("create a Vercel sandbox", e))?;
        parse_json(response, "create a Vercel sandbox").await
    }

    /// Attach the bearer token and the team scope to a request.
    fn request(&self, builder: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
        let mut builder = builder.bearer_auth(token);
        if let Some(team_id) = self.options.resolved_team_id() {
            builder = builder.query(&[("teamId", team_id)]);
        }
        builder
    }
}

#[async_trait]
impl SandboxClient for VercelSandboxClient {
    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        let manifest = resolve_manifest_root(manifest.unwrap_or_default());
        let token = self.options.resolved_token()?;
        let mut envs = self.options.env.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        let created = self.create_sandbox(&token, &envs, None).await?;
        let session = VercelSandboxSession::new(
            self.options.clone(),
            token,
            self.http.clone(),
            created,
            manifest,
            self.snapshot.clone(),
        );
        session.wait_until_running().await?;
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
        // `create` rewrites the SDK-wide default root to the Vercel workspace, so the recorded
        // digest covers the rewritten root.
        let manifest = adopt_recorded_root(manifest, state);
        state.verify_manifest(&manifest)?;
        let provider_id = state.provider_id.clone().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "Vercel sandbox session state is missing its sandbox identity",
            )
        })?;
        let (sandbox_name, session_id) = split_provider_id(&provider_id)?;
        let token = self.options.resolved_token()?;
        let mut envs = self.options.env.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }

        let existing = self
            .request(
                self.http.get(format!(
                    "{}/v2/sandboxes/sessions/{session_id}",
                    self.options.api_base()
                )),
                &token,
            )
            .send()
            .await
            .ok()
            .filter(|response| response.status().is_success());
        #[derive(Deserialize)]
        struct SessionOnly {
            session: SessionRecord,
        }
        let reattached = match existing {
            Some(response) => parse_json::<SessionOnly>(response, "load a Vercel session")
                .await
                .ok()
                .filter(|loaded| loaded.session.status.as_deref() == Some("running"))
                .map(|loaded| SandboxAndSession {
                    sandbox: SandboxRecord {
                        name: sandbox_name.clone(),
                    },
                    session: loaded.session,
                }),
            None => None,
        };
        let created = match reattached {
            Some(created) => created,
            // The original session is gone: create a replacement and hydrate it from the snapshot.
            None => self.create_sandbox(&token, &envs, None).await?,
        };
        let session = VercelSandboxSession::new(
            self.options.clone(),
            token,
            self.http.clone(),
            created,
            manifest,
            Some(state.snapshot.clone()),
        )
        .with_session_key(state.session_id.clone())
        .with_snapshot_id(state.snapshot_id.clone())
        .resuming();
        session.wait_until_running().await?;
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// Apply the Vercel workspace root when the manifest still carries the SDK-wide default.
fn resolve_manifest_root(manifest: Manifest) -> Manifest {
    if manifest.root() == DEFAULT_MANIFEST_ROOT {
        manifest.with_root(DEFAULT_VERCEL_WORKSPACE_ROOT)
    } else {
        manifest
    }
}

/// Split the `<sandbox name>:<session id>` pair stored in the persisted state.
fn split_provider_id(provider_id: &str) -> SandboxResult<(String, String)> {
    provider_id
        .split_once(':')
        .map(|(name, session)| (name.to_string(), session.to_string()))
        .filter(|(name, session)| !name.is_empty() && !session.is_empty())
        .ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SessionStateInvalid,
                "Vercel sandbox session state carries a malformed sandbox identity",
            )
        })
}

/// A sandbox session whose workspace lives in a Vercel sandbox.
pub struct VercelSandboxSession {
    session_key: String,
    manifest: Manifest,
    options: VercelSandboxClientOptions,
    token: String,
    api_base: String,
    http: reqwest::Client,
    remote: std::sync::RwLock<RemoteHandle>,
    started: AtomicBool,
    terminated: tokio::sync::Mutex<bool>,
    snapshot: Option<SnapshotSpec>,
    snapshot_id: String,
    resuming: bool,
}

/// The mutable provider identity: a snapshot restore replaces both halves.
#[derive(Debug, Clone)]
struct RemoteHandle {
    sandbox_name: String,
    session_id: String,
}

// The bearer token must never appear in a log line.
impl std::fmt::Debug for VercelSandboxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let handle = self.remote.read().expect("poisoned");
        f.debug_struct("VercelSandboxSession")
            .field("session_key", &self.session_key)
            .field("sandbox_name", &handle.sandbox_name)
            .field("provider_session_id", &handle.session_id)
            .field("workspace_root", &self.manifest.root())
            .field("resuming", &self.resuming)
            .finish_non_exhaustive()
    }
}

impl VercelSandboxSession {
    fn new(
        options: VercelSandboxClientOptions,
        token: String,
        http: reqwest::Client,
        created: SandboxAndSession,
        manifest: Manifest,
        snapshot: Option<SnapshotSpec>,
    ) -> Self {
        let session_key = uuid::Uuid::new_v4().to_string();
        VercelSandboxSession {
            api_base: options.api_base(),
            options,
            token,
            http,
            remote: std::sync::RwLock::new(RemoteHandle {
                sandbox_name: created.sandbox.name,
                session_id: created.session.id,
            }),
            manifest,
            snapshot_id: session_key.clone(),
            session_key,
            started: AtomicBool::new(false),
            terminated: tokio::sync::Mutex::new(false),
            snapshot,
            resuming: false,
        }
    }

    fn with_session_key(mut self, session_key: String) -> Self {
        self.session_key = session_key;
        self
    }

    fn with_snapshot_id(mut self, snapshot_id: String) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    fn resuming(mut self) -> Self {
        self.resuming = true;
        self
    }

    /// The Vercel runtime session currently backing this sandbox session.
    pub fn provider_session_id(&self) -> String {
        self.remote.read().expect("poisoned").session_id.clone()
    }

    /// The Vercel sandbox name.
    pub fn sandbox_name(&self) -> String {
        self.remote.read().expect("poisoned").sandbox_name.clone()
    }

    /// Attach the bearer token and team scope to a request.
    fn request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder.bearer_auth(&self.token);
        if let Some(team_id) = self.options.resolved_team_id() {
            builder = builder.query(&[("teamId", team_id)]);
        }
        builder
    }

    /// A URL under the current runtime session.
    fn session_url(&self, path: &str) -> String {
        format!(
            "{}/v2/sandboxes/sessions/{}{path}",
            self.api_base,
            self.provider_session_id()
        )
    }

    /// The environment every command runs with.
    fn command_envs(&self) -> BTreeMap<String, String> {
        let mut envs = self.options.env.clone();
        for (key, value) in self.manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        envs
    }

    /// Resolve the working directory of a request, defaulting to the workspace root.
    fn resolve_cwd(&self, cwd: &Option<String>) -> SandboxResult<String> {
        match cwd {
            None => Ok(self.manifest.root().to_string()),
            Some(cwd) => self.normalize_path(cwd),
        }
    }

    /// Map the requested sandbox user onto the API's `sudo` flag.
    fn resolve_sudo(user: Option<&User>) -> SandboxResult<bool> {
        match user {
            None => Ok(false),
            Some(user) if user.name == "root" => Ok(true),
            Some(user) => Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "the Vercel sandbox backend can only run commands as the session user or \
                     `root`, not `{}`",
                    user.name
                ),
            )),
        }
    }

    /// Read the current session status.
    async fn session_status(&self) -> SandboxResult<Option<String>> {
        let response = self
            .request(self.http.get(self.session_url("")))
            .send()
            .await
            .map_err(|e| transport("load the Vercel session", e))?;
        #[derive(Deserialize)]
        struct SessionOnly {
            session: SessionRecord,
        }
        let loaded: SessionOnly = parse_json(response, "load the Vercel session").await?;
        Ok(loaded.session.status)
    }

    /// Poll until the session reports `running`, or the budget elapses.
    async fn wait_until_running(&self) -> SandboxResult<()> {
        if self.remote.read().expect("poisoned").session_id.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                "Vercel returned a sandbox without a runtime session",
            ));
        }
        let deadline = Instant::now() + self.options.wait_for_running;
        loop {
            match self.session_status().await?.as_deref() {
                Some("running") => return Ok(()),
                // Only `pending` can still reach `running`.
                Some("pending") | None => {}
                Some(status) => {
                    return Err(SandboxError::new(
                        ErrorCode::WorkspaceStartError,
                        format!(
                            "Vercel session `{}` entered the terminal state `{status}`",
                            self.provider_session_id()
                        ),
                    ));
                }
            }
            if Instant::now() >= deadline {
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceStartError,
                    format!(
                        "Vercel session `{}` did not start within {:?}",
                        self.provider_session_id(),
                        self.options.wait_for_running
                    ),
                ));
            }
            tokio::time::sleep(self.options.status_poll_interval).await;
        }
    }

    /// Run one command and collect its output from the NDJSON stream.
    async fn run_command(
        &self,
        argv: Vec<String>,
        cwd: String,
        timeout: Option<Duration>,
        sudo: bool,
    ) -> SandboxResult<ExecResult> {
        let what = "run a command in the Vercel sandbox";
        let (command, args) = argv
            .split_first()
            .map(|(command, args)| (command.clone(), args.to_vec()))
            .unwrap_or_else(|| ("true".to_string(), Vec::new()));
        let mut body = json!({
            "command": command,
            "args": args,
            "cwd": cwd,
            "wait": true,
            "logs": true,
        });
        let map = body.as_object_mut().expect("object");
        let envs = self.command_envs();
        if !envs.is_empty() {
            map.insert("env".into(), json!(envs));
        }
        if sudo {
            map.insert("sudo".into(), json!(true));
        }
        if let Some(timeout) = timeout {
            map.insert("timeout".into(), json!(timeout.as_millis() as u64));
        }
        let send = async {
            let response = self
                .request(self.http.post(self.session_url("/cmd")))
                .json(&body)
                .send()
                .await
                .map_err(|e| transport(what, e))?;
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(http_error(
                    ErrorCode::ExecTransportError,
                    what,
                    status,
                    &text,
                ));
            }
            let text = response.text().await.map_err(|e| transport(what, e))?;
            parse_command_stream(&text)
        };
        match timeout {
            None => send.await,
            Some(limit) => match tokio::time::timeout(limit, send).await {
                Ok(result) => result,
                Err(_) => Err(SandboxError::new(
                    ErrorCode::ExecTimeout,
                    format!("Command `{}` timed out after {limit:?}", argv.join(" ")),
                )),
            },
        }
    }

    /// Download a file by its absolute sandbox path.
    ///
    /// Workspace reads go through [`read_file`](SandboxSession::read_file), which validates the
    /// path first; this is for the paths the session itself owns, such as a temporary archive
    /// under `/tmp`.
    async fn download_at(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let what = "read a file from the Vercel sandbox";
        let response = self
            .request(self.http.post(self.session_url("/fs/read")))
            .json(&json!({ "path": path, "cwd": self.manifest.root() }))
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

    /// Upload a gzipped tar archive, extracting it under `extract_dir`.
    async fn write_archive(&self, extract_dir: &str, archive: Vec<u8>) -> SandboxResult<()> {
        let what = "write files to the Vercel sandbox";
        let response = self
            .request(self.http.post(self.session_url("/fs/write")))
            .header("Content-Type", "application/gzip")
            .header("x-cwd", extract_dir)
            .body(archive)
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
    }

    /// Take a native Vercel snapshot of the session.
    async fn create_native_snapshot(&self) -> SandboxResult<String> {
        let what = "snapshot the Vercel sandbox";
        let body = match self.options.snapshot_expiration_ms {
            Some(expiration) => json!({ "expiration": expiration }),
            None => json!({}),
        };
        let response = self
            .request(self.http.post(self.session_url("/snapshot")))
            .json(&body)
            .send()
            .await
            .map_err(|e| transport(what, e))?;
        #[derive(Deserialize)]
        struct CreateSnapshot {
            snapshot: SnapshotRecord,
        }
        #[derive(Deserialize)]
        struct SnapshotRecord {
            id: String,
        }
        let created: CreateSnapshot = parse_json(response, what).await?;
        Ok(created.snapshot.id)
    }

    /// Replace the backing sandbox with one created from a native snapshot.
    async fn restore_native_snapshot(&self, snapshot_id: &str) -> SandboxResult<()> {
        let what = "restore a Vercel sandbox snapshot";
        let mut envs = self.options.env.clone();
        for (key, value) in self.manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }
        let client = VercelSandboxClient {
            options: self.options.clone(),
            http: self.http.clone(),
            snapshot: None,
        };
        let created = client
            .create_sandbox(
                &self.token,
                &envs,
                Some(json!({ "type": "snapshot", "snapshotId": snapshot_id })),
            )
            .await?;
        let previous = {
            let mut handle = self.remote.write().expect("poisoned");
            let previous = handle.session_id.clone();
            handle.sandbox_name = created.sandbox.name;
            handle.session_id = created.session.id;
            previous
        };
        // Stop the session the snapshot replaces so the project does not leak sandboxes.
        let _ = self
            .request(self.http.post(format!(
                "{}/v2/sandboxes/sessions/{previous}/stop",
                self.api_base
            )))
            .send()
            .await;
        self.wait_until_running().await.map_err(|error| {
            SandboxError::new(
                ErrorCode::SnapshotRestoreError,
                format!("Failed to {what}: {}", error.message),
            )
        })
    }

    /// Stop the runtime session exactly once.
    async fn terminate(&self) -> SandboxResult<()> {
        let mut terminated = self.terminated.lock().await;
        if *terminated {
            return Ok(());
        }
        let response = self
            .request(self.http.post(self.session_url("/stop")))
            .send()
            .await
            .map_err(|e| transport("stop the Vercel sandbox", e))?;
        self.started.store(false, Ordering::SeqCst);
        *terminated = true;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        expect_success(
            response,
            "stop the Vercel sandbox",
            ErrorCode::WorkspaceStopError,
        )
        .await
    }
}

#[async_trait]
impl SandboxSession for VercelSandboxSession {
    fn id(&self) -> &str {
        &self.session_key
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    async fn start(&self) -> SandboxResult<()> {
        let root = self.manifest.root().to_string();
        let prepared = self
            .run_command(
                vec!["mkdir".into(), "-p".into(), "--".into(), root.clone()],
                "/".to_string(),
                None,
                false,
            )
            .await?;
        if !prepared.ok() {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                format!(
                    "Failed to create the Vercel workspace root `{root}`: {}",
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
        self.terminate().await
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.terminate().await
    }

    async fn running(&self) -> SandboxResult<bool> {
        Ok(self
            .session_status()
            .await
            .map(|status| status.as_deref() == Some("running"))
            .unwrap_or(false))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let cwd = self.resolve_cwd(&request.cwd)?;
        let sudo = Self::resolve_sudo(request.user.as_ref())?;
        // The API takes the privilege escalation out of band, so no `sudo` wrapper is built here.
        let argv = ExecRequest {
            user: None,
            ..request.clone()
        }
        .resolved_argv();
        self.run_command(argv, cwd, request.timeout, sudo).await
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        self.download_at(&resolved).await
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path(path)?;
        // `fs/write` only accepts a gzipped tar whose entry names are relative to `x-cwd`.
        let relative = resolved.trim_start_matches('/');
        let mut archive = TarGzBuilder::new();
        archive.add_file(relative, contents, 0o644);
        self.write_archive("/", archive.finish()).await
    }

    fn session_state(&self) -> SandboxSessionState {
        let handle = self.remote.read().expect("poisoned");
        SandboxSessionState::new(
            BACKEND_ID,
            &self.session_key,
            self.manifest.root(),
            &self.manifest,
        )
        // A Vercel sandbox is addressed by both its name and its runtime session id.
        .with_provider_id(Some(format!(
            "{}:{}",
            handle.sandbox_name, handle.session_id
        )))
        .with_snapshot(self.snapshot.clone().unwrap_or_default())
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        if self.options.workspace_persistence == WorkspacePersistence::Snapshot {
            let snapshot_id = self.create_native_snapshot().await?;
            return Ok(encode_snapshot_ref(&snapshot_id));
        }
        let root = self.manifest.root().to_string();
        let archive_path = format!("/tmp/openai-agents-{}.tar.gz", self.session_key);
        let mut command = format!(
            "tar -czf {archive} -C {root}",
            archive = shell_quote(&archive_path),
            root = shell_quote(&root),
        );
        for exclude in workspace_excludes(&self.manifest, &root) {
            command.push_str(&format!(" --exclude {}", shell_quote(&exclude)));
        }
        command.push_str(" .");
        // Command output arrives as text, so the archive travels over the filesystem API.
        let result = self
            .run_command(
                vec!["sh".into(), "-lc".into(), command],
                root.clone(),
                None,
                false,
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
        let _ = self
            .run_command(
                vec!["rm".into(), "-f".into(), "--".into(), archive_path.clone()],
                "/".to_string(),
                None,
                false,
            )
            .await;
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
        if let Some(snapshot_id) = decode_snapshot_ref(archive) {
            return self.restore_native_snapshot(&snapshot_id).await;
        }
        let root = self.manifest.root().to_string();
        self.mkdir(&root, true).await?;
        // `fs/write` extracts a gzipped tar directly, which is exactly the archive format
        // `persist_workspace` produces.
        self.write_archive(&root, archive.to_vec())
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::SnapshotRestoreError,
                    format!("Failed to restore the Vercel workspace: {}", error.message),
                )
            })
    }
}

/// Parse the NDJSON body of a `cmd` call with `wait` and `logs` enabled.
fn parse_command_stream(body: &str) -> SandboxResult<ExecResult> {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut exit_code = None;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(command) = value.get("command") {
            if let Some(code) = command.get("exitCode").and_then(Value::as_i64) {
                exit_code = Some(code as i32);
            }
            continue;
        }
        match value.get("stream").and_then(Value::as_str) {
            Some("stdout") => {
                if let Some(data) = value.get("data").and_then(Value::as_str) {
                    stdout.extend_from_slice(data.as_bytes());
                }
            }
            Some("stderr") => {
                if let Some(data) = value.get("data").and_then(Value::as_str) {
                    stderr.extend_from_slice(data.as_bytes());
                }
            }
            Some("error") => {
                let message = value
                    .get("data")
                    .and_then(|data| data.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(SandboxError::new(
                    ErrorCode::ExecTransportError,
                    format!("Vercel reported a command stream error: {message}"),
                ));
            }
            _ => {}
        }
    }
    let exit_code = exit_code.ok_or_else(|| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            "the Vercel command stream ended before the command finished",
        )
    })?;
    Ok(ExecResult {
        stdout,
        stderr,
        exit_code,
    })
}

// -------------------------------------------------------------------------------------------
// Cloud bucket mounts
// -------------------------------------------------------------------------------------------

/// Mounts an S3 bucket inside a Vercel sandbox with `mount-s3`. Mirrors
/// `agents.extensions.sandbox.vercel.mounts.VercelCloudBucketMountStrategy`.
///
/// Vercel's released runtime images ship `mount-s3` (Mountpoint for Amazon S3), so the mount is
/// an in-sandbox command rather than part of the sandbox specification. When the mount carries
/// credentials they reach the sandbox as process environment variables, so the mount reports
/// [`MountCredentialAuthority::MountScoped`] and the manifest has to acknowledge the exact mount
/// path before it activates.
#[derive(Debug, Clone)]
pub struct VercelCloudBucketMountStrategy {
    /// The bucket to mount.
    pub bucket: String,
    /// Optional key prefix inside the bucket.
    pub prefix: Option<String>,
    /// Optional AWS region.
    pub region: Option<String>,
    /// Optional S3-compatible endpoint.
    pub endpoint_url: Option<String>,
    /// The credentials handed to `mount-s3`, as environment assignments.
    pub credentials: MountPatternConfig,
}

impl VercelCloudBucketMountStrategy {
    /// A credential-free (anonymous) mount of `bucket`.
    pub fn new(bucket: impl Into<String>) -> Self {
        VercelCloudBucketMountStrategy {
            bucket: bucket.into(),
            prefix: None,
            region: None,
            endpoint_url: None,
            credentials: MountPatternConfig::new(),
        }
    }

    /// Mount only the keys under `prefix`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Pin the bucket's region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Use an S3-compatible endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Authenticate with static credentials, passed as `KEY=value` lines.
    pub fn with_credentials(mut self, credentials: impl Into<String>) -> Self {
        self.credentials = MountPatternConfig::new().with_config_text(credentials);
        self
    }

    /// A [`Mount`] using this strategy.
    pub fn mount(self) -> Mount {
        Mount::new(Arc::new(self))
    }

    /// The `mount-s3` argument vector for `mount_path`.
    fn mount_argv(&self, mount_path: &str, read_only: bool) -> Vec<String> {
        let mut argv = vec![
            "mount-s3".to_string(),
            self.bucket.clone(),
            mount_path.to_string(),
            "--allow-other".to_string(),
        ];
        if read_only {
            argv.push("--read-only".to_string());
        }
        if self.credentials.has_config_text() {
            argv.push("--allow-delete".to_string());
        } else {
            argv.push("--no-sign-request".to_string());
        }
        if let Some(region) = &self.region {
            argv.extend(["--region".to_string(), region.clone()]);
        }
        if let Some(endpoint_url) = &self.endpoint_url {
            argv.extend(["--endpoint-url".to_string(), endpoint_url.clone()]);
        }
        if let Some(prefix) = &self.prefix {
            let prefix = if prefix.ends_with('/') {
                prefix.clone()
            } else {
                format!("{prefix}/")
            };
            argv.extend(["--prefix".to_string(), prefix]);
        }
        argv
    }
}

#[async_trait]
impl MountStrategy for VercelCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "vercel_cloud_bucket"
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        if self.bucket.trim().is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a Vercel cloud bucket mount requires a bucket name",
            ));
        }
        Ok(())
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        if self.credentials.has_config_text() {
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
        session.mkdir(mount_path, true).await?;
        let argv = self.mount_argv(mount_path, mount.read_only);
        let quoted_argv = argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        // `ExecRequest` carries no environment, and anything in the command line is readable
        // from the sandbox's process table and is recorded by the provider's command log, so the
        // credentials are written to a `600` file, sourced, and removed before `mount-s3` runs.
        let mut staged_credentials = None;
        let command = match self.credentials.config_text() {
            None => quoted_argv,
            Some(credentials) => {
                let credentials_path = format!(
                    "{}/.openai-agents-mount-s3-{}.env",
                    session.workspace_root().trim_end_matches('/'),
                    uuid::Uuid::new_v4().simple()
                );
                session
                    .write_file(&credentials_path, credentials.as_bytes())
                    .await
                    .map_err(|error| {
                        SandboxError::new(
                            ErrorCode::MountFailed,
                            format!("failed to stage the bucket credentials: {}", error.message),
                        )
                    })?;
                staged_credentials = Some(credentials_path.clone());
                let quoted_path = shell_quote(&credentials_path);
                // The trap fires on every exit path, so a failure between staging the file and
                // starting `mount-s3` cannot leave the credential behind in the workspace.
                [
                    format!("trap 'rm -f -- {quoted_path}' EXIT INT TERM HUP"),
                    format!("chmod 600 -- {quoted_path}"),
                    "set -a".to_string(),
                    format!(". {quoted_path}"),
                    "set +a".to_string(),
                    format!("rm -f -- {quoted_path}"),
                    quoted_argv,
                ]
                .join("; ")
            }
        };
        let result = session
            .exec(
                ExecRequest::argv(["sh", "-lc", command.as_str()])
                    .with_shell(crate::sandbox::session::ExecShell::None)
                    .with_timeout(Some(Duration::from_secs(120))),
            )
            .await?;
        if !result.ok() {
            // The shell trap normally removes the credential; make sure of it when the command
            // never ran at all.
            if let Some(path) = &staged_credentials {
                let _ = session.rm(path, false).await;
            }
            // Never echo the command back: it names the bucket the credentials belong to.
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "`mount-s3` failed for `{mount_path}` with exit code {}: {}",
                    result.exit_code,
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let quoted = shell_quote(mount_path);
        let script = format!(
            "fusermount3 -u -- {quoted} 2>/dev/null || fusermount -u -- {quoted} 2>/dev/null || \
             umount -- {quoted} 2>/dev/null || true"
        );
        session
            .exec(
                ExecRequest::argv(["sh", "-lc", script.as_str()])
                    .with_shell(crate::sandbox::session::ExecShell::None)
                    .with_timeout(Some(Duration::from_secs(60))),
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_the_token() {
        let options = VercelSandboxClientOptions::new().with_token("vercel_secret");
        let debug = format!("{options:?}");
        assert!(!debug.contains("vercel_secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn applies_the_vercel_workspace_root_only_to_the_default() {
        let defaulted = resolve_manifest_root(Manifest::new());
        assert_eq!(defaulted.root(), DEFAULT_VERCEL_WORKSPACE_ROOT);
        let explicit = resolve_manifest_root(Manifest::new().with_root("/srv/app"));
        assert_eq!(explicit.root(), "/srv/app");
    }

    #[test]
    fn parses_the_command_stream() {
        let body = concat!(
            r#"{"command":{"id":"cmd-1","name":"sh","args":["-lc","echo hi"],"cwd":"/vercel/sandbox","sessionId":"s","exitCode":null,"startedAt":1}}"#,
            "\n",
            r#"{"stream":"stdout","data":"hi\n"}"#,
            "\n",
            r#"{"stream":"stderr","data":"warn\n"}"#,
            "\n",
            r#"{"command":{"id":"cmd-1","name":"sh","args":[],"cwd":"/vercel/sandbox","sessionId":"s","exitCode":3,"startedAt":1}}"#,
            "\n",
        );
        let result = parse_command_stream(body).unwrap();
        assert_eq!(result.stdout_text(), "hi\n");
        assert_eq!(result.stderr_text(), "warn\n");
        assert_eq!(result.exit_code, 3);
    }

    #[test]
    fn surfaces_command_stream_errors() {
        let body = r#"{"stream":"error","data":{"code":"oom","message":"out of memory"}}"#;
        let error = parse_command_stream(body).unwrap_err();
        assert_eq!(error.code, ErrorCode::ExecTransportError);
        assert!(error.message.contains("out of memory"));
    }

    #[test]
    fn requires_a_finished_command_record() {
        let error = parse_command_stream(r#"{"stream":"stdout","data":"hi"}"#).unwrap_err();
        assert_eq!(error.code, ErrorCode::ExecTransportError);
    }

    #[test]
    fn maps_only_root_onto_sudo() {
        assert!(!VercelSandboxSession::resolve_sudo(None).unwrap());
        assert!(VercelSandboxSession::resolve_sudo(Some(&User::new("root"))).unwrap());
        assert_eq!(
            VercelSandboxSession::resolve_sudo(Some(&User::new("agent")))
                .unwrap_err()
                .code,
            ErrorCode::SandboxConfigInvalid
        );
    }

    #[test]
    fn splits_the_provider_identity() {
        assert_eq!(
            split_provider_id("agents-abc:sess-1").unwrap(),
            ("agents-abc".to_string(), "sess-1".to_string())
        );
        assert!(split_provider_id("agents-abc").is_err());
        assert!(split_provider_id(":sess-1").is_err());
    }

    #[test]
    fn builds_mount_s3_arguments() {
        let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
            .with_prefix("data")
            .with_region("us-east-1");
        assert_eq!(
            strategy.mount_argv("/vercel/sandbox/data", true),
            vec![
                "mount-s3",
                "my-bucket",
                "/vercel/sandbox/data",
                "--allow-other",
                "--read-only",
                "--no-sign-request",
                "--region",
                "us-east-1",
                "--prefix",
                "data/",
            ]
        );
    }

    #[test]
    fn serializes_network_policies() {
        assert_eq!(NetworkPolicy::AllowAll.to_api(), json!({ "allow": "all" }));
        assert_eq!(NetworkPolicy::DenyAll.to_api(), json!({ "allow": [] }));
        let custom = NetworkPolicy::Custom {
            allow: vec!["example.com".into()],
            allow_subnets: vec!["10.0.0.0/8".into()],
            deny_subnets: Vec::new(),
        };
        assert_eq!(
            custom.to_api(),
            json!({
                "allow": ["example.com"],
                "subnets": { "allow": ["10.0.0.0/8"], "deny": [] },
            })
        );
    }
}
