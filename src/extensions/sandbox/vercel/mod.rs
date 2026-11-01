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
//! - Python installs a shell helper to resolve a mount path through its symlinks. This port has
//!   no runtime-helper installer, so [`mounts`] runs `realpath -m` instead, which has the same
//!   semantics for a path whose leaf may not exist yet.
//!
//! [`supports_pty`]: crate::sandbox::SandboxSession::supports_pty

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
use crate::sandbox::retry::{RetryPolicy, is_transient_sandbox_error, retry_async};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, clear_workspace_root_on_resume,
    live_workspace_matches_snapshot_on_resume, shell_quote, workspace_archive_excludes,
};
use crate::sandbox::snapshot::{Snapshot, SnapshotSpec, resolve_snapshot};
use crate::sandbox::state::{SandboxSessionState, SnapshotFingerprint};
use crate::sandbox::types::{ExecResult, User};
use crate::sandbox::workspace_paths::provider_scratch_path;

use super::e2b::adopt_recorded_root;
use super::ports::{
    ExposedPortEndpoint, ExposedPortUnavailable, assert_exposed_port_configured, endpoint_from_url,
    exposed_port_unavailable,
};
use crate::sandbox::http::{
    as_archive_read, bounded_read_zero, expect_success, http_error, parse_json, read_body_prefix,
    read_status_error, status_is_retryable, transport,
};

pub mod mounts;
pub mod tar_gz;

pub use self::mounts::VercelCloudBucketMountStrategy;
use self::tar_gz::TarGzBuilder;

/// The backend identifier reported by this client and its sessions.
pub const BACKEND_ID: &str = "vercel";
/// The strategy type [`VercelCloudBucketMountStrategy`] reports.
const VERCEL_S3_MOUNT_STRATEGY: &str = "vercel_cloud_bucket";
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
/// The runtime alias a sandbox is created from when none is configured, matching Python.
pub const DEFAULT_VERCEL_RUNTIME: &str = "node24";

/// The concrete image each runtime alias names. Mirrors Python's `_RUNTIME_IMAGES`.
///
/// The released runtime images are Amazon Linux, which is what ships the S3 mount tools the
/// [`mounts`] strategy drives, so the alias is resolved here rather than left to the API.
const RUNTIME_IMAGES: [(&str, &str); 4] = [
    ("node22", "vercel/sandbox/node:al-22"),
    ("node24", "vercel/sandbox/node:al-24"),
    ("node26", "vercel/sandbox/node:al-26"),
    ("python3.13", "vercel/sandbox/python:al-3.13.1"),
];

/// Marks a workspace archive that only references a native Vercel sandbox snapshot.
///
/// Vercel has its own marker so a reference written here round-trips with the Python SDK, and so
/// a reference from another backend is never mistaken for one of these.
const SNAPSHOT_REF_MAGIC: &[u8] = b"UC_VERCEL_SNAPSHOT_V1\n";

/// The image a runtime alias names, defaulting to [`DEFAULT_VERCEL_RUNTIME`].
fn resolve_runtime_image(runtime: Option<&str>) -> SandboxResult<&'static str> {
    let runtime = runtime.unwrap_or(DEFAULT_VERCEL_RUNTIME);
    RUNTIME_IMAGES
        .iter()
        .find(|(alias, _)| *alias == runtime)
        .map(|(_, image)| *image)
        .ok_or_else(|| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("unsupported Vercel runtime: `{runtime}`"),
            )
        })
}

/// Encode a reference to a native Vercel snapshot as a workspace archive.
fn encode_snapshot_ref(snapshot_id: &str) -> Vec<u8> {
    let mut encoded = SNAPSHOT_REF_MAGIC.to_vec();
    encoded.extend_from_slice(
        serde_json::to_string(&json!({ "snapshot_id": snapshot_id }))
            .expect("serializable")
            .as_bytes(),
    );
    encoded
}

/// Read back a snapshot reference written by [`encode_snapshot_ref`].
fn decode_snapshot_ref(raw: &[u8]) -> Option<String> {
    let body = raw.strip_prefix(SNAPSHOT_REF_MAGIC)?;
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("snapshot_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// How a session persists its workspace into a [`Snapshot`].
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspacePersistence {
    /// Archive the workspace with `tar`, so the bytes are portable across backends.
    #[default]
    Tar,
    /// Take a native Vercel snapshot and store only a reference to it.
    Snapshot,
}

/// One rewrite applied to a request that matched a domain. Mirrors
/// `agents.extensions.sandbox.vercel._network_policy.NetworkTransformer`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct NetworkTransformer {
    /// Headers injected into, or replaced on, the outgoing request.
    pub headers: BTreeMap<String, String>,
}

impl NetworkTransformer {
    /// A transformer that rewrites nothing.
    pub fn new() -> Self {
        NetworkTransformer::default()
    }

    /// Inject or replace one header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }
}

/// One rule attached to an allowed domain. Mirrors
/// `agents.extensions.sandbox.vercel._network_policy.NetworkPolicyRule`.
///
/// A domain that carries rules is intercepted rather than proxied through an encrypted
/// connection, which is what lets the platform rewrite the request.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct NetworkPolicyRule {
    /// The rewrites applied to a matching request, in order.
    pub transform: Vec<NetworkTransformer>,
}

impl NetworkPolicyRule {
    /// A rule that allows the domain without rewriting anything.
    pub fn new() -> Self {
        NetworkPolicyRule::default()
    }

    /// Append a rewrite.
    pub fn with_transform(mut self, transform: NetworkTransformer) -> Self {
        self.transform.push(transform);
        self
    }

    /// The JSON body the API expects.
    fn to_api(&self) -> Value {
        json!({
            "transform": self
                .transform
                .iter()
                .map(|transform| json!({ "headers": transform.headers }))
                .collect::<Vec<_>>(),
        })
    }
}

/// The network policy applied to a sandbox. Mirrors `agents.extensions.sandbox.vercel`'s
/// `NetworkPolicy`, kept as the released Agents JSON shape rather than the provider's types.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
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
    /// Allow the listed domains, each with its own rules.
    ///
    /// This is Python's per-domain `NetworkPolicyCustom.allow` mapping: a domain with no rules
    /// is allowed as-is, and a domain with rules has those rewrites applied to its requests.
    CustomRules {
        /// Domains egress may reach, and the rules attached to each.
        allow: BTreeMap<String, Vec<NetworkPolicyRule>>,
        /// Subnets egress may reach.
        allow_subnets: Vec<String>,
        /// Subnets egress may never reach.
        deny_subnets: Vec<String>,
    },
}

impl NetworkPolicy {
    /// The JSON body the API expects.
    ///
    /// `POST /v3/sandboxes` types `networkPolicy` as a `oneOf` of two objects: the mode form
    /// (`{mode, allowedDomains, allowedCIDRs, deniedCIDRs, injectionRules}`) and the rule form
    /// (`{allow, subnets}`, where `allow` is an array or an object). `allow-all` and `deny-all`
    /// are the mode form, which is what the provider SDK sends for its `allow_all()`/`deny_all()`
    /// constructors; a custom policy is the rule form.
    fn to_api(&self) -> Value {
        let (allow, allow_subnets, deny_subnets) = match self {
            NetworkPolicy::AllowAll => return json!({ "mode": "allow-all" }),
            NetworkPolicy::DenyAll => return json!({ "mode": "deny-all" }),
            NetworkPolicy::Custom {
                allow,
                allow_subnets,
                deny_subnets,
            } => (json!(allow), allow_subnets, deny_subnets),
            NetworkPolicy::CustomRules {
                allow,
                allow_subnets,
                deny_subnets,
            } => (
                Value::Object(
                    allow
                        .iter()
                        .map(|(domain, rules)| {
                            (
                                domain.clone(),
                                Value::Array(rules.iter().map(NetworkPolicyRule::to_api).collect()),
                            )
                        })
                        .collect(),
                ),
                allow_subnets,
                deny_subnets,
            ),
        };
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

impl NetworkPolicyRule {
    /// Read a rule from the `{"transform": [{"headers": {...}}]}` shape `to_api` writes.
    fn from_state_json(rule: &Value) -> NetworkPolicyRule {
        let transform = rule
            .get("transform")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| NetworkTransformer {
                headers: row
                    .get("headers")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .filter_map(|(name, value)| Some((name.clone(), value.as_str()?.to_string())))
                    .collect(),
            })
            .collect();
        NetworkPolicyRule { transform }
    }
}

impl NetworkPolicy {
    /// The policy in the JSON shape Python's `VercelSandboxSessionState.network_policy` has:
    /// `"allow-all"`, `"deny-all"`, or `{"allow": [...] | {domain: [rule]}, "subnets":
    /// {"allow": [...] | null, "deny": [...] | null} | null}`.
    pub fn to_state_json(&self) -> Value {
        let (allow, allow_subnets, deny_subnets) = match self {
            NetworkPolicy::AllowAll => return json!("allow-all"),
            NetworkPolicy::DenyAll => return json!("deny-all"),
            NetworkPolicy::Custom {
                allow,
                allow_subnets,
                deny_subnets,
            } => (json!(allow), allow_subnets, deny_subnets),
            NetworkPolicy::CustomRules {
                allow,
                allow_subnets,
                deny_subnets,
            } => (
                Value::Object(
                    allow
                        .iter()
                        .map(|(domain, rules)| {
                            (
                                domain.clone(),
                                Value::Array(rules.iter().map(NetworkPolicyRule::to_api).collect()),
                            )
                        })
                        .collect(),
                ),
                allow_subnets,
                deny_subnets,
            ),
        };
        let list = |subnets: &Vec<String>| {
            if subnets.is_empty() {
                Value::Null
            } else {
                json!(subnets)
            }
        };
        let subnets = if allow_subnets.is_empty() && deny_subnets.is_empty() {
            Value::Null
        } else {
            json!({ "allow": list(allow_subnets), "deny": list(deny_subnets) })
        };
        json!({ "allow": allow, "subnets": subnets })
    }

    /// Read a policy written by [`to_state_json`](Self::to_state_json) (or by Python's state
    /// payload). `None` for a value of another shape.
    pub fn from_state_json(value: &Value) -> Option<NetworkPolicy> {
        match value {
            Value::String(mode) if mode == "allow-all" => return Some(NetworkPolicy::AllowAll),
            Value::String(mode) if mode == "deny-all" => return Some(NetworkPolicy::DenyAll),
            _ => {}
        }
        let object = value.as_object()?;
        let strings = |value: Option<&Value>| -> Option<Vec<String>> {
            match value {
                None | Some(Value::Null) => Some(Vec::new()),
                Some(Value::Array(rows)) => rows
                    .iter()
                    .map(|row| row.as_str().map(str::to_string))
                    .collect(),
                Some(_) => None,
            }
        };
        let (allow_subnets, deny_subnets) = match object.get("subnets") {
            None | Some(Value::Null) => (Vec::new(), Vec::new()),
            Some(Value::Object(subnets)) => (
                strings(subnets.get("allow"))?,
                strings(subnets.get("deny"))?,
            ),
            Some(_) => return None,
        };
        match object.get("allow")? {
            Value::Array(_) => Some(NetworkPolicy::Custom {
                allow: strings(object.get("allow"))?,
                allow_subnets,
                deny_subnets,
            }),
            Value::Object(domains) => {
                let mut allow = BTreeMap::new();
                for (domain, rules) in domains {
                    let rules = rules
                        .as_array()?
                        .iter()
                        .map(NetworkPolicyRule::from_state_json)
                        .collect();
                    allow.insert(domain.clone(), rules);
                }
                Some(NetworkPolicy::CustomRules {
                    allow,
                    allow_subnets,
                    deny_subnets,
                })
            }
            _ => None,
        }
    }
}

/// Serde adapter that stores an optional [`NetworkPolicy`] in Python's state-payload shape.
mod network_policy_state {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;

    use super::NetworkPolicy;

    pub(super) fn serialize<S: Serializer>(
        policy: &Option<NetworkPolicy>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        policy
            .as_ref()
            .map(NetworkPolicy::to_state_json)
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<NetworkPolicy>, D::Error> {
        match Option::<Value>::deserialize(deserializer)? {
            None | Some(Value::Null) => Ok(None),
            Some(value) => NetworkPolicy::from_state_json(&value)
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("unrecognised Vercel network policy")),
        }
    }
}

/// Resource limits for a Vercel sandbox.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
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

/// The Vercel options a resumed session is rebuilt from.
///
/// Mirrors the `runtime`, `resources`, `timeout_ms`, `snapshot_expiration_ms` and
/// `network_policy` fields of Python's `VercelSandboxSessionState`. It travels in the session
/// state's provider options, so a replacement sandbox created on resume keeps the runtime,
/// limits and network policy of the original instead of the resuming client's. Every field has
/// a serde default, so a payload that predates a field still reads.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(default)]
pub struct VercelPersistedOptions {
    /// The runtime image alias.
    pub runtime: Option<String>,
    /// Resource limits.
    pub resources: VercelSandboxResources,
    /// Execution time limit in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Snapshot expiration in milliseconds.
    pub snapshot_expiration_ms: Option<u64>,
    /// Network policy applied to the sandbox, in the shape Python's state payload uses.
    #[serde(with = "network_policy_state")]
    pub network_policy: Option<NetworkPolicy>,
}

impl VercelPersistedOptions {
    /// The persisted subset of `options`.
    pub fn from_options(options: &VercelSandboxClientOptions) -> Self {
        VercelPersistedOptions {
            runtime: options.runtime.clone(),
            resources: options.resources.clone(),
            timeout_ms: options.timeout_ms,
            snapshot_expiration_ms: options.snapshot_expiration_ms,
            network_policy: options.network_policy.clone(),
        }
    }

    /// Replace the corresponding fields of `options`.
    pub fn apply_to(self, options: &mut VercelSandboxClientOptions) {
        options.runtime = self.runtime;
        options.resources = self.resources;
        options.timeout_ms = self.timeout_ms;
        options.snapshot_expiration_ms = self.snapshot_expiration_ms;
        options.network_policy = self.network_policy;
    }

    /// The options persisted in `state`, when it carries any.
    pub fn from_state(state: &SandboxSessionState) -> Option<Self> {
        state.provider_options()
    }
}

/// Settings used when creating a Vercel-backed session.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
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
    /// Runtime image alias, such as `node24`; `None` uses [`DEFAULT_VERCEL_RUNTIME`].
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
            .or_else(|| oidc_claim(VERCEL_OIDC_PROJECT_CLAIM))
    }

    fn resolved_team_id(&self) -> Option<String> {
        self.team_id
            .clone()
            .or_else(|| env_var(VERCEL_TEAM_ID_ENV))
            .or_else(|| oidc_claim(VERCEL_OIDC_TEAM_CLAIM))
    }
}

/// The `project_id` claim of a Vercel OIDC token.
const VERCEL_OIDC_PROJECT_CLAIM: &str = "project_id";
/// The claim naming the owning team of a Vercel OIDC token.
const VERCEL_OIDC_TEAM_CLAIM: &str = "owner_id";

/// Read one claim out of the OIDC token in the environment.
///
/// Mirrors `vercel.oidc.credentials.get_credentials`, which falls back to the token's own
/// `project_id` and `owner_id` claims when the environment names neither: a deployment that
/// carries only `VERCEL_OIDC_TOKEN` still knows which project and team it belongs to. The
/// payload is read, never verified — the API is what validates the token.
fn oidc_claim(claim: &str) -> Option<String> {
    let token = env_var(VERCEL_OIDC_TOKEN_ENV)?;
    let payload = token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    claims
        .get(claim)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
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
    /// The public route the API assigned to each exposed port.
    #[serde(default)]
    routes: Vec<RouteRecord>,
}

/// One public route of a sandbox, as `POST /v3/sandboxes` reports it.
#[derive(Debug, Clone, Deserialize)]
struct RouteRecord {
    port: u16,
    url: String,
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
    /// This client with the options `state` recorded layered over its own.
    ///
    /// A state written before options were recorded leaves the client's options in place.
    fn with_persisted_options(&self, state: &SandboxSessionState) -> Self {
        let mut client = self.clone();
        if let Some(persisted) = VercelPersistedOptions::from_state(state) {
            persisted.apply_to(&mut client.options);
        }
        client
    }

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
        // Python resolves the runtime alias to a concrete image, and `/v3/sandboxes` has no
        // `runtime` field at all: an unresolved alias would silently fall back to the universal
        // image, which does not ship the S3 mount tools.
        map.insert(
            "image".into(),
            json!(resolve_runtime_image(self.options.runtime.as_deref())?),
        );
        if let Some(policy) = &self.options.network_policy {
            map.insert("networkPolicy".into(), policy.to_api());
        }
        if !envs.is_empty() {
            map.insert("env".into(), json!(envs));
        }
        if let Some(expiration) = self.options.snapshot_expiration_ms {
            map.insert("snapshotExpiration".into(), json!(expiration));
        }
        // Python wraps `AsyncSandbox.create` in `@retry_async` for a transient create failure
        // (`vercel/sandbox.py` L150). `parse_json` classifies the status, so a 400 or a 402
        // fails once instead of three times.
        let what = "create a Vercel sandbox";
        retry_async(
            RetryPolicy::new(),
            || async {
                let response = self
                    .request(
                        self.http
                            .post(format!("{}/v3/sandboxes", self.options.api_base())),
                        token,
                    )
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| transport(what, e))?;
                parse_json(response, what).await
            },
            |error: &SandboxError, _attempt| is_transient_sandbox_error(error),
        )
        .await
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
        // Refuse an unusable mount layout before a sandbox exists to be billed for.
        validate_s3_mount_topology(&manifest)?;
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
        let token = this.options.resolved_token()?;
        let mut envs = this.options.env.clone();
        for (key, value) in manifest.environment() {
            envs.insert(key.clone(), value.clone());
        }

        let existing = self
            .request(
                this.http.get(format!(
                    "{}/v2/sandboxes/sessions/{session_id}",
                    this.options.api_base()
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
            #[serde(default)]
            routes: Vec<RouteRecord>,
        }
        let reattached = match existing {
            Some(response) => parse_json::<SessionOnly>(response, "load a Vercel session")
                .await
                .ok()
                // A `pending` sandbox is still starting and keeps its workspace, so it is waited
                // for below; every other state either runs or cannot reach `running` again.
                .filter(|loaded| {
                    matches!(
                        loaded.session.status.as_deref(),
                        Some("running" | "pending")
                    )
                })
                .map(|loaded| SandboxAndSession {
                    sandbox: SandboxRecord {
                        name: sandbox_name.clone(),
                    },
                    session: loaded.session,
                    routes: loaded.routes,
                }),
            None => None,
        };
        let build = |created: SandboxAndSession, manifest: Manifest| {
            VercelSandboxSession::new(
                this.options.clone(),
                token.clone(),
                this.http.clone(),
                created,
                manifest,
                Some(state.snapshot.clone()),
            )
            .with_session_key(state.session_id.clone())
            .with_snapshot_id(state.snapshot_id.clone())
            .resuming()
        };
        if let Some(created) = reattached {
            let record = state
                .snapshot_fingerprint
                .clone()
                .zip(state.snapshot_fingerprint_version.clone())
                .map(|(fingerprint, version)| SnapshotFingerprint {
                    fingerprint,
                    version,
                });
            let session = build(created, manifest.clone())
                .with_state_preserved(true)
                .with_snapshot_fingerprint(record);
            // A sandbox that never reaches `running` is replaced, as a missing one is.
            if session.wait_until_running().await.is_ok() {
                return Ok(Arc::new(session));
            }
        }
        // The original session is gone: create a replacement and hydrate it from the snapshot.
        let created = this.create_sandbox(&token, &envs, None).await?;
        let session = build(created, manifest);
        session.wait_until_running().await?;
        Ok(Arc::new(session))
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        session.shutdown().await
    }
}

/// Check where a manifest puts its Vercel S3 mounts.
///
/// Mirrors Python's `_vercel_s3_mount_map`: a mount may not replace the workspace root, sit outside
/// it, overlap another mount, or overlap an entry the manifest materializes (a directory entry
/// that merely contains the mount is fine).
fn validate_s3_mount_topology(manifest: &Manifest) -> SandboxResult<()> {
    let root = mount_path_key(manifest.root());
    let invalid = |message: &str, mount_path: &str| {
        SandboxError::new(ErrorCode::MountConfigInvalid, message.to_string())
            .with_context("backend", BACKEND_ID)
            .with_context("mount_path", mount_path.to_string())
    };
    let targets: Vec<(String, bool)> = manifest
        .mount_targets()
        .into_iter()
        .map(|(mount, path)| {
            (
                mount_path_key(&path),
                mount.strategy.strategy_type() == VERCEL_S3_MOUNT_STRATEGY,
            )
        })
        .collect();
    let contains = |outer: &str, inner: &str| {
        inner == outer || inner.starts_with(&format!("{}/", outer.trim_end_matches('/')))
    };
    // Misconfigured mounts fail here, before a sandbox exists to be billed for.
    for (mount, _) in manifest.mount_targets() {
        if mount.strategy.strategy_type() == VERCEL_S3_MOUNT_STRATEGY {
            mount.strategy.validate(mount)?;
        }
    }
    for (index, (path, is_vercel)) in targets.iter().enumerate() {
        if !is_vercel {
            continue;
        }
        if *path == root {
            return Err(invalid(
                "Vercel does not support mounting an S3 bucket at the workspace root",
                path,
            ));
        }
        if !contains(&root, path) {
            return Err(invalid(
                "Vercel S3 mount paths must stay within the workspace root",
                path,
            ));
        }
        for (other_index, (other, _)) in targets.iter().enumerate() {
            if other_index != index && (contains(path, other) || contains(other, path)) {
                return Err(invalid(
                    "Vercel S3 mount paths must not overlap other mounts",
                    path,
                ));
            }
        }
        for (entry_path, entry) in manifest.iter_entries() {
            if entry.as_mount().is_some() {
                continue;
            }
            let entry_path = mount_path_key(&entry_path);
            let is_dir = matches!(entry.kind, crate::sandbox::entries::EntryKind::Dir { .. });
            let ancestor_dir = is_dir && entry_path != *path && contains(&entry_path, path);
            if !ancestor_dir && (contains(path, &entry_path) || contains(&entry_path, path)) {
                return Err(invalid(
                    "Vercel S3 mount paths must not overlap manifest entries",
                    path,
                ));
            }
        }
    }
    Ok(())
}

/// Apply the Vercel workspace root when the manifest still carries the SDK-wide default.
fn resolve_manifest_root(manifest: Manifest) -> Manifest {
    if manifest.root() == DEFAULT_MANIFEST_ROOT {
        manifest.with_root(DEFAULT_VERCEL_WORKSPACE_ROOT)
    } else {
        manifest
    }
}

/// The key a mount path is tracked under, so one trailing slash cannot hide a tracked mount.
fn mount_path_key(mount_path: &str) -> String {
    let trimmed = mount_path.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Whether the Vercel API documents a status as worth retrying.
///
/// Mirrors Python's `_vercel_provider_retryability` table: request timeouts (408), too-early
/// (425), rate limits (429) and server errors retry; every other client error does not.
fn vercel_status_is_retryable(status: u16) -> bool {
    matches!(status, 408 | 425 | 429) || (500..=599).contains(&status)
}

/// Split the `<sandbox name>:<session id>` pair stored in the persisted state.
///
/// A bare id without a name predates sandbox names; like Python, it is both the session id and
/// the name.
fn split_provider_id(provider_id: &str) -> SandboxResult<(String, String)> {
    if !provider_id.is_empty() && !provider_id.contains(':') {
        return Ok((provider_id.to_string(), provider_id.to_string()));
    }
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
    /// Whether this session reattached a sandbox that kept its workspace and system state.
    state_preserved: bool,
    /// The workspace fingerprint the stored snapshot was taken from.
    snapshot_fingerprint: std::sync::Mutex<Option<SnapshotFingerprint>>,
    /// Whether the session is inside [`start`](SandboxSession::start), the only point at which
    /// its S3 mount topology may be applied.
    s3_mounts_startable: AtomicBool,
    /// The mount paths currently carrying a Mountpoint filesystem.
    active_s3_mounts: std::sync::Mutex<BTreeSet<String>>,
    /// The mount paths detached so a native snapshot could be taken.
    detached_s3_mounts: std::sync::Mutex<BTreeSet<String>>,
    /// Whether the fixed S3 mount topology has already been applied once.
    s3_mounts_started: AtomicBool,
    /// Workspace paths registered at runtime that a workspace archive leaves out.
    persist_skips: std::sync::Mutex<BTreeSet<String>>,
}

/// The mutable provider identity: a snapshot restore replaces both halves.
#[derive(Debug, Clone)]
struct RemoteHandle {
    sandbox_name: String,
    session_id: String,
    /// The public URL of each exposed port, keyed by the sandbox-side port.
    routes: BTreeMap<u16, String>,
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
                routes: created
                    .routes
                    .into_iter()
                    .map(|route| (route.port, route.url))
                    .collect(),
            }),
            manifest,
            snapshot_id: session_key.clone(),
            session_key,
            started: AtomicBool::new(false),
            terminated: tokio::sync::Mutex::new(false),
            snapshot,
            resuming: false,
            state_preserved: false,
            snapshot_fingerprint: std::sync::Mutex::new(None),
            s3_mounts_startable: AtomicBool::new(false),
            active_s3_mounts: std::sync::Mutex::new(BTreeSet::new()),
            detached_s3_mounts: std::sync::Mutex::new(BTreeSet::new()),
            s3_mounts_started: AtomicBool::new(false),
            persist_skips: std::sync::Mutex::new(BTreeSet::new()),
        }
    }

    /// Whether `manifest` holds exactly the mounts this session was created with.
    ///
    /// Mirrors Python's `_runtime_s3_mount_topology_matches`.
    fn s3_topology_matches(&self, manifest: &Manifest) -> bool {
        let signature = |manifest: &Manifest| -> BTreeSet<(String, String)> {
            manifest
                .mount_targets()
                .into_iter()
                .map(|(mount, path)| (mount_path_key(&path), format!("{mount:?}")))
                .collect()
        };
        signature(&self.manifest) == signature(manifest)
    }

    /// Whether an S3 mount may be attached right now.
    ///
    /// Mirrors Python's `_runtime_s3_mount_activation_allowed`: Vercel fixes its mount topology
    /// when the sandbox is created, so a manifest applied to a running session must not be able
    /// to hand a fresh credential to a sandbox that has already run model-controlled code.
    pub(crate) fn s3_mount_activation_allowed(&self) -> bool {
        self.s3_mounts_startable.load(Ordering::SeqCst)
    }

    /// Fail unless `mount_path` is one of the mount targets this session was created with.
    ///
    /// Mirrors Python's `_runtime_trusted_s3_mount`, which only knows the mounts captured at
    /// creation: the manifest the session holds is that same trusted topology.
    pub(crate) fn assert_trusted_s3_mount(&self, mount_path: &str) -> SandboxResult<()> {
        let key = mount_path_key(mount_path);
        let trusted = self
            .manifest
            .mount_targets()
            .into_iter()
            .any(|(_, path)| mount_path_key(&path) == key);
        if trusted {
            return Ok(());
        }
        Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "the Vercel S3 mount configuration for `{mount_path}` is unavailable outside \
                 sandbox creation"
            ),
        ))
    }

    /// Whether the manifest carries any Vercel S3 mount. Mirrors `_vercel_s3_mounts`.
    fn has_s3_mounts(&self) -> bool {
        self.manifest
            .mount_targets()
            .iter()
            .any(|(mount, _)| mount.strategy.strategy_type() == VERCEL_S3_MOUNT_STRATEGY)
    }

    /// Whether this session attached a mount at `mount_path`.
    pub(crate) fn s3_mount_is_active(&self, mount_path: &str) -> bool {
        self.active_s3_mounts
            .lock()
            .expect("poisoned")
            .contains(&mount_path_key(mount_path))
    }

    /// Whether this session detached a mount at `mount_path` for a snapshot.
    pub(crate) fn s3_mount_is_detached(&self, mount_path: &str) -> bool {
        self.detached_s3_mounts
            .lock()
            .expect("poisoned")
            .contains(&mount_path_key(mount_path))
    }

    /// Record that a mount is attached at `mount_path`.
    pub(crate) fn record_s3_mount_active(&self, mount_path: &str) {
        let key = mount_path_key(mount_path);
        self.detached_s3_mounts
            .lock()
            .expect("poisoned")
            .remove(&key);
        self.active_s3_mounts.lock().expect("poisoned").insert(key);
    }

    /// Record that a mount is gone from `mount_path`.
    pub(crate) fn record_s3_mount_inactive(&self, mount_path: &str) {
        let key = mount_path_key(mount_path);
        self.active_s3_mounts.lock().expect("poisoned").remove(&key);
        self.detached_s3_mounts
            .lock()
            .expect("poisoned")
            .remove(&key);
    }

    /// Record that a mount was detached at `mount_path` so a snapshot could be taken.
    pub(crate) fn record_s3_mount_detached(&self, mount_path: &str) {
        let key = mount_path_key(mount_path);
        self.active_s3_mounts.lock().expect("poisoned").remove(&key);
        self.detached_s3_mounts
            .lock()
            .expect("poisoned")
            .insert(key);
    }

    /// Stop the sandbox after a mount transition failed, and return the original error.
    ///
    /// Mirrors Python's `_runtime_fail_s3_mount_transition`: a privileged mount command that
    /// failed part way leaves the sandbox in a state no later step can reason about, so the
    /// sandbox does not survive it.
    pub(crate) async fn fail_s3_mount_transition(&self, error: SandboxError) -> SandboxError {
        let _ = self.terminate().await;
        error
    }

    fn with_session_key(mut self, session_key: String) -> Self {
        self.session_key = session_key;
        self
    }

    fn with_snapshot_id(mut self, snapshot_id: String) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    /// Mark the sandbox as one that survived the pause with its workspace intact.
    fn with_state_preserved(mut self, preserved: bool) -> Self {
        self.state_preserved = preserved;
        self
    }

    /// Carry the workspace fingerprint a resumed snapshot was taken from.
    fn with_snapshot_fingerprint(self, record: Option<SnapshotFingerprint>) -> Self {
        *self.snapshot_fingerprint.lock().expect("poisoned") = record;
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
        envs: &BTreeMap<String, String>,
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
                let mut error = http_error(ErrorCode::ExecTransportError, what, status, &text);
                error.retryable = Some(vercel_status_is_retryable(status.as_u16()));
                return Err(error);
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
    async fn download_at(&self, path: &str, limit: Option<usize>) -> SandboxResult<Vec<u8>> {
        let what = "read a file from the Vercel sandbox";
        let response = self
            .request(self.http.post(self.session_url("/fs/read")))
            .json(&json!({ "path": path, "cwd": self.manifest.root() }))
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

    /// Upload a gzipped tar archive, extracting it under `extract_dir`.
    ///
    /// Retried on a transient transport failure, as Python's `_write_files_with_retry` is.
    async fn write_archive(&self, extract_dir: &str, archive: Vec<u8>) -> SandboxResult<()> {
        let what = "write files to the Vercel sandbox";
        retry_async(
            RetryPolicy::new(),
            || async {
                let response = self
                    .request(self.http.post(self.session_url("/fs/write")))
                    .header("Content-Type", "application/gzip")
                    .header("x-cwd", extract_dir)
                    .body(archive.clone())
                    .send()
                    .await
                    .map_err(|e| transport(what, e))?;
                expect_success(response, what, ErrorCode::WorkspaceWriteTypeError).await
            },
            |error, _attempt| is_transient_sandbox_error(error),
        )
        .await
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
            // The replacement sandbox gets its own public routes.
            handle.routes = created
                .routes
                .into_iter()
                .map(|route| (route.port, route.url))
                .collect();
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

    /// Apply the manifest with every S3 mount attached last.
    ///
    /// Mirrors Python, which activates the mounts only after all other entries are written: an
    /// entry written after the mount is attached (or one that reaches the mount path through an
    /// alias) would land in the bucket, and a failed entry write must not leave a credential
    /// already handed to the sandbox. With `ephemeral_only`, only the ephemeral entries are
    /// re-applied, as a resume does.
    async fn apply_deferring_mounts(&self, ephemeral_only: bool) -> SandboxResult<()> {
        if !self.has_s3_mounts() {
            let manifest = self.manifest.clone();
            return if ephemeral_only {
                manifest.apply_ephemeral(self).await.map(|_| ())
            } else {
                manifest.apply(self).await.map(|_| ())
            };
        }
        let is_s3 = |mount: &crate::sandbox::mounts::Mount| {
            mount.strategy.strategy_type() == VERCEL_S3_MOUNT_STRATEGY
        };
        let rest = self.manifest.split_mounts(&is_s3, false);
        let mounts = self.manifest.split_mounts(&is_s3, true);
        if ephemeral_only {
            rest.apply_ephemeral(self).await?;
            mounts.apply_ephemeral(self).await?;
        } else {
            rest.apply(self).await?;
            // The accounts were provisioned with the first half.
            mounts.apply_without_account_provisioning(self).await?;
        }
        Ok(())
    }

    /// Restore the stored snapshot into the workspace, then re-apply the ephemeral entries with
    /// the S3 mounts last.
    ///
    /// The same steps as the shared `restore_session_snapshot`, whose final step applies the
    /// ephemeral entries in manifest order.
    async fn restore_deferring_mounts(&self) -> SandboxResult<bool> {
        let snapshot = self.snapshot();
        if !snapshot.restorable().await? {
            return Ok(false);
        }
        if live_workspace_matches_snapshot_on_resume(self).await {
            self.apply_deferring_mounts(true).await?;
            return Ok(true);
        }
        let archive = snapshot
            .restore_with(self.dependencies().map(Arc::as_ref))
            .await?;
        self.validate_restorable_archive(&archive)?;
        clear_workspace_root_on_resume(self).await?;
        self.hydrate_workspace(&archive).await?;
        if !self.system_state_preserved_on_start() {
            self.manifest.provision_accounts(self).await?;
        }
        self.apply_deferring_mounts(true).await?;
        Ok(true)
    }

    /// Replace provider text in `result`'s error when a credentialed mount is in play.
    ///
    /// A provider error can echo the environment a mount command was handed, so a session that
    /// carries a mount credential reports only the error code. Mirrors Python's
    /// `_replace_protected_mount_error`.
    fn redacted<T>(&self, result: SandboxResult<T>) -> SandboxResult<T> {
        result.map_err(|error| {
            crate::sandbox::sandboxes::docker::redact_for_protected_mounts(&self.manifest, error)
        })
    }

    /// Check a workspace tar before any byte of it is uploaded.
    ///
    /// Beyond the generic hydrate policy, no member may overlap a path the manifest owns as
    /// ephemeral (an S3 mount's path included): the archive is extracted as validated.
    fn validate_tar_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        let mut options = crate::sandbox::archive::TarValidationOptions::hydrate();
        options.reject_rel_paths = self.manifest.ephemeral_persistence_paths();
        crate::sandbox::archive::validate_tar_bytes(archive, &options)
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
        // A session that is already gone is as good as stopped.
        let stopped = if response.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(())
        } else {
            expect_success(
                response,
                "stop the Vercel sandbox",
                ErrorCode::WorkspaceStopError,
            )
            .await
        };
        // Only a confirmed stop ends the session: a failed one leaves cleanup retryable, so a
        // second `shutdown` tries again instead of reporting a sandbox that still runs as closed.
        if stopped.is_ok() {
            self.started.store(false, Ordering::SeqCst);
            *terminated = true;
        }
        stopped
    }
}

#[async_trait]
impl SandboxSession for VercelSandboxSession {
    /// Mirrors Python's `_validate_manifest_application`: Vercel fixes its S3 mount topology when
    /// the sandbox is created, so a live manifest may not change it, and non-mount entries can
    /// only be added to a running session.
    async fn validate_manifest_application(
        &self,
        manifest: &Manifest,
        session_running: Option<bool>,
    ) -> SandboxResult<()> {
        let refuse = |message: &str| {
            SandboxError::new(ErrorCode::MountConfigInvalid, message.to_string())
                .with_context("backend", BACKEND_ID)
        };
        if self.s3_mount_activation_allowed() {
            return Ok(());
        }
        if !self.s3_topology_matches(manifest) {
            return Err(refuse(
                "Vercel S3 mount topology is fixed when the sandbox is created; dynamic \
                 manifest application is not supported",
            ));
        }
        let has_fixed_mounts = self
            .manifest
            .mount_targets()
            .into_iter()
            .any(|(mount, _)| mount.strategy.strategy_type() == VERCEL_S3_MOUNT_STRATEGY);
        if has_fixed_mounts {
            let running = match session_running {
                Some(running) => running,
                None => self.running().await?,
            };
            if !running {
                return Err(refuse(
                    "Vercel sessions with fixed S3 mounts must be running before non-mount \
                     manifest entries can be applied; create a new session instead",
                ));
            }
        }
        Ok(())
    }

    fn id(&self) -> &str {
        &self.session_key
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        BACKEND_ID
    }

    fn persist_skip_registry(&self) -> Option<&std::sync::Mutex<BTreeSet<String>>> {
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

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    async fn start(&self) -> SandboxResult<()> {
        // Vercel fixes the S3 mount topology when the sandbox is created, so a second `start`
        // on a mounted session would re-present credentials to a sandbox that has already run
        // model-controlled code. Mirrors Python's `_s3_mounts_started` guard.
        let has_s3_mounts = self.has_s3_mounts();
        if has_s3_mounts && self.s3_mounts_started.load(Ordering::SeqCst) {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "Vercel S3 mount topology is fixed when the sandbox is created; starting the \
                 same mounted session again is not supported",
            )
            .with_context("backend", BACKEND_ID)
            .with_retryable(false));
        }
        let root = self.manifest.root().to_string();
        let prepared = self
            .run_command(
                vec!["mkdir".into(), "-p".into(), "--".into(), root.clone()],
                "/".to_string(),
                &self.command_envs(),
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
        // The S3 mount topology is applied here and nowhere else, so the window in which a mount
        // may attach is exactly the manifest application below.
        self.s3_mounts_startable.store(true, Ordering::SeqCst);
        let result = if self.resuming {
            match self.restore_deferring_mounts().await {
                Ok(false) => self.apply_deferring_mounts(true).await,
                Ok(true) => Ok(()),
                Err(error) => Err(error),
            }
        } else {
            self.apply_deferring_mounts(false).await
        };
        self.s3_mounts_startable.store(false, Ordering::SeqCst);
        // Python closes the window on success, and on failure only once a mount actually
        // attached: a start that failed before any credential reached the sandbox is safe to
        // retry, and must not leave the session permanently unstartable.
        if has_s3_mounts
            && (result.is_ok() || !self.active_s3_mounts.lock().expect("poisoned").is_empty())
        {
            self.s3_mounts_started.store(true, Ordering::SeqCst);
        }
        result.inspect_err(|_| {
            self.started.store(false, Ordering::SeqCst);
        })?;
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        self.redacted(self.terminate().await)
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.redacted(self.terminate().await)
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
        let envs = self.request_envs(&request);
        let argv = ExecRequest {
            user: None,
            ..request.clone()
        }
        .resolved_argv();
        self.redacted(
            self.run_command(argv, cwd, &envs, request.timeout, sudo)
                .await,
        )
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        // The in-sandbox resolver sees a symlink the lexical check cannot.
        let resolved = self.redacted(self.validate_remote_path_access(path, false, false).await)?;
        self.redacted(self.download_at(&resolved, None).await)
    }

    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if max_bytes == 0 {
            return Err(bounded_read_zero());
        }
        // A bounded read stays on the lexical check: the in-sandbox resolver costs extra
        // commands per read, and the bounded-read contract tests drive it without one.
        let resolved = self.normalize_path(path)?;
        self.redacted(self.download_at(&resolved, Some(max_bytes)).await)
    }

    fn exposed_ports(&self) -> Vec<u16> {
        self.options.exposed_ports.clone()
    }

    /// Resolve an exposed port to its public Vercel route.
    ///
    /// Mirrors `VercelSandboxSession._resolve_exposed_port`, whose `sandbox.domain(port)` reads
    /// the route the API assigned when the sandbox was created. The REST create and session
    /// responses carry those routes, so this needs no extra round trip.
    async fn resolve_exposed_port(&self, port: u16) -> SandboxResult<ExposedPortEndpoint> {
        assert_exposed_port_configured(BACKEND_ID, port, &self.options.exposed_ports)?;
        let url = self
            .remote
            .read()
            .expect("poisoned")
            .routes
            .get(&port)
            .cloned();
        self.redacted(url.as_deref().and_then(endpoint_from_url).ok_or_else(|| {
            exposed_port_unavailable(
                BACKEND_ID,
                port,
                &self.options.exposed_ports,
                ExposedPortUnavailable::BackendUnavailable,
            )
        }))
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.redacted(self.validate_remote_path_access(path, true, false).await)?;
        // `fs/write` only accepts a gzipped tar whose entry names are relative to `x-cwd`.
        let relative = resolved.trim_start_matches('/');
        let mut archive = TarGzBuilder::new();
        archive.add_file(relative, contents, 0o644);
        self.redacted(self.write_archive("/", archive.finish()).await)
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
        .with_snapshot_fingerprint(self.snapshot_fingerprint.lock().expect("poisoned").clone())
        .with_provider_options(&VercelPersistedOptions::from_options(&self.options))
    }

    fn snapshot(&self) -> Arc<dyn Snapshot> {
        resolve_snapshot(self.snapshot.as_ref(), &self.snapshot_id)
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        let result: SandboxResult<Vec<u8>> = async {
            if self.options.workspace_persistence == WorkspacePersistence::Snapshot {
                let snapshot_id = self.create_native_snapshot().await?;
                return Ok(encode_snapshot_ref(&snapshot_id));
            }
            let root = self.manifest.root().to_string();
            let archive_path = self.resolve_internal_path(&provider_scratch_path(&format!(
                "{}.tar.gz",
                self.session_key
            )))?;
            // `--no-wildcards` makes every `--exclude` a literal path: a mount or entry named
            // `cache[1]` must exclude itself, not match a sibling named `cache1`.
            let mut command = format!(
                "tar -czf {archive} -C {root} --no-wildcards",
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
            // Command output arrives as text, so the archive travels over the filesystem API.
            let result = self
                .run_command(
                    vec!["sh".into(), "-lc".into(), command],
                    root.clone(),
                    &self.command_envs(),
                    None,
                    false,
                )
                .await?;
            if !result.ok() {
                // The scratch tarball may be partly written; it must not outlive the failure.
                let _ = self
                    .run_command(
                        vec!["rm".into(), "-f".into(), "--".into(), archive_path.clone()],
                        "/".to_string(),
                        &self.command_envs(),
                        None,
                        false,
                    )
                    .await;
                return Err(SandboxError::new(
                    ErrorCode::SnapshotPersistError,
                    format!(
                        "Failed to archive workspace `{root}`: {}",
                        result.stderr_text().trim()
                    ),
                ));
            }
            let archive = self.download_at(&archive_path, None).await;
            let _ = self
                .run_command(
                    vec!["rm".into(), "-f".into(), "--".into(), archive_path.clone()],
                    "/".to_string(),
                    &self.command_envs(),
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
        .await;
        self.redacted(result)
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        if decode_snapshot_ref(archive).is_some() {
            return Ok(());
        }
        self.validate_tar_archive(archive)
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        if let Some(snapshot_id) = decode_snapshot_ref(archive) {
            // A native snapshot restore replaces the sandbox, and a replacement has none of the
            // mounts this session attached at creation.
            if self.has_s3_mounts() {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "Vercel S3 mounts cannot be restored from a native snapshot; the mount \
                     topology is fixed when the sandbox is created",
                )
                .with_context("backend", BACKEND_ID)
                .with_retryable(false));
            }
            return self.restore_native_snapshot(&snapshot_id).await;
        }
        self.validate_tar_archive(archive)?;
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
        assert_eq!(
            split_provider_id("legacy-id").unwrap(),
            ("legacy-id".to_string(), "legacy-id".to_string())
        );
        assert!(split_provider_id("").is_err());
        assert!(split_provider_id(":sess-1").is_err());
    }

    #[test]
    fn serializes_network_policies() {
        assert_eq!(
            NetworkPolicy::AllowAll.to_api(),
            json!({ "mode": "allow-all" })
        );
        assert_eq!(
            NetworkPolicy::DenyAll.to_api(),
            json!({ "mode": "deny-all" })
        );
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

    #[test]
    fn serializes_per_domain_network_policy_rules() {
        let policy = NetworkPolicy::CustomRules {
            allow: BTreeMap::from([
                ("open.example.com".to_string(), Vec::new()),
                (
                    "api.example.com".to_string(),
                    vec![NetworkPolicyRule::new().with_transform(
                        NetworkTransformer::new().with_header("x-api-key", "injected"),
                    )],
                ),
            ]),
            allow_subnets: Vec::new(),
            deny_subnets: vec!["10.0.0.0/8".into()],
        };
        assert_eq!(
            policy.to_api(),
            json!({
                "allow": {
                    "api.example.com": [
                        { "transform": [{ "headers": { "x-api-key": "injected" } }] },
                    ],
                    "open.example.com": [],
                },
                "subnets": { "allow": [], "deny": ["10.0.0.0/8"] },
            })
        );
    }

    #[test]
    fn resolves_runtime_aliases_to_amazon_linux_images() {
        assert_eq!(
            resolve_runtime_image(None).unwrap(),
            "vercel/sandbox/node:al-24"
        );
        assert_eq!(
            resolve_runtime_image(Some("python3.13")).unwrap(),
            "vercel/sandbox/python:al-3.13.1"
        );
        let error = resolve_runtime_image(Some("node18")).unwrap_err();
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
        assert!(error.message.contains("node18"), "{}", error.message);
    }

    #[test]
    fn round_trips_a_vercel_snapshot_reference() {
        let encoded = encode_snapshot_ref("snap-1");
        assert!(encoded.starts_with(b"UC_VERCEL_SNAPSHOT_V1\n"));
        assert_eq!(decode_snapshot_ref(&encoded).as_deref(), Some("snap-1"));
        // A reference written by another backend is not a Vercel snapshot.
        assert_eq!(
            decode_snapshot_ref(b"E2B_SANDBOX_SNAPSHOT_V1\n{\"snapshot_id\":\"snap-1\"}"),
            None
        );
        assert_eq!(decode_snapshot_ref(b"not a snapshot"), None);
    }

    #[test]
    fn tracks_mount_paths_regardless_of_a_trailing_slash() {
        assert_eq!(
            mount_path_key("/vercel/sandbox/data/"),
            "/vercel/sandbox/data"
        );
        assert_eq!(mount_path_key("/"), "/");
    }
}
