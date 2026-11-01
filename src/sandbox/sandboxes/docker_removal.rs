//! Opt-in, host-side removal for private, rootful Linux Docker containers. Mirrors
//! `agents.sandbox.sandboxes.docker_removal`.
//!
//! Run this service in a trusted process on the Docker daemon host, **as root**: the library
//! never escalates, `sudo`s or grants a privilege itself, and it is only compiled with the
//! explicit `docker-removal` feature. It uses the `docker-removal-worker` executable (a host
//! binary, never one supplied by the container), does not listen on a network socket, and
//! should only ever be handed to trusted application code.
//!
//! Only Docker sessions explicitly configured with this service use its removal authority.
//! Other backends, and Docker clients without the service, keep their `docker exec rm` behaviour.
//!
//! The service requires a trusted image, Docker 26+ with its builtin seccomp profile, and the
//! `runc` runtime. For new client sessions it checks container eligibility, then creates a
//! missing workspace and its parents with the trusted image's default user before binding, as
//! normal session start-up does. Path-only grant roots must exist at binding time; the client
//! does not create unrelated grant directories. Canonical workspace and grant roots and removal
//! targets must not contain backslashes. Read-only host bind mounts are supported outside the
//! private workspace; writable shared mounts, additional capabilities, user namespaces and
//! missing grant roots are excluded. The application must exclusively own container lifecycle
//! and Docker API access; other host administrators are trusted. A service/worker transport
//! failure leaves the container paused: stop all service workers before resuming it by hand.
//!
//! Use one service for the client's lifetime and close it after deleting its sessions. Binding
//! is live authority and is never serialized. Reattaching a running container requires the same
//! live binding; a new service must create a new container.
//!
//! ```no_run
//! # async fn demo() -> openai_agents_sdk::sandbox::SandboxResult<()> {
//! use openai_agents_sdk::sandbox::sandboxes::docker::DockerSandboxClient;
//! use openai_agents_sdk::sandbox::sandboxes::docker_removal::{
//!     DockerRemovalOptions, DockerRemovalService,
//! };
//!
//! // The process must already run as root on the Docker host; this fails closed otherwise.
//! let service = DockerRemovalService::new(DockerRemovalOptions::new(4, 100_000, 10))?;
//! let client = DockerSandboxClient::new().with_removal_service(service.clone());
//! # drop(client);
//! service.close().await?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, Semaphore};

use super::docker::{
    DockerRunner, assert_container_path_grants_match, validate_docker_path_grants,
};
use super::docker_removal_worker::RemoveRequest;
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::workspace_paths::{
    SandboxPathGrant, WorkspacePathPolicy, sandbox_path_grant_host_path,
};

/// Why a worker request failed: Python's `OSError` (the worker answered `ok: false`) or
/// `RuntimeError` (the transport broke, so the outcome is uncertain).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerFailure {
    /// The worker answered with a structured failure.
    Os {
        /// The errno, when the failure had one.
        errno: Option<i32>,
        /// The Python exception class name the worker reported.
        reason: String,
    },
    /// The request or response was lost.
    Transport(String),
}

/// One live worker process bound to one container.
#[async_trait]
pub trait RemovalWorker: Send {
    /// Send one request and wait for its response.
    ///
    /// The shapes are Python's: `bind`, `inspect` and `remove` (see the module docs of
    /// `docker_removal_worker`).
    async fn request(&mut self, request: Value) -> Result<Value, WorkerFailure>;

    /// Whether a transport failure left the outcome of a request unknown.
    fn uncertain(&self) -> bool;

    /// Stop the worker and reap it.
    async fn close(&mut self) -> io::Result<()>;
}

/// Starts workers.
#[async_trait]
pub trait WorkerLauncher: Send + Sync {
    /// Start a worker that serves the container whose init process is `pid`.
    async fn launch(&self, pid: u32) -> Result<Box<dyn RemovalWorker>, WorkerFailure>;
}

/// Reads the host's `/proc`.
pub trait RemovalHost: Send + Sync {
    /// The text of `/proc/<pid>/<name>`.
    fn read_proc(&self, pid: u32, name: &str) -> io::Result<String>;
}

/// The real `/proc`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcHost;

impl RemovalHost for ProcHost {
    fn read_proc(&self, pid: u32, name: &str) -> io::Result<String> {
        std::fs::read_to_string(format!("/proc/{pid}/{name}"))
    }
}

/// The limits and the worker executable of a [`DockerRemovalService`].
#[derive(Debug, Clone)]
pub struct DockerRemovalOptions {
    /// How many removals may pause containers at once; calls beyond it are rejected before
    /// anything is paused.
    pub max_concurrent_removals: usize,
    /// How many directory entries one removal may visit, counting repeated visits to ancestors.
    pub max_entry_visits: u64,
    /// The CPU-time limit of each removal child, in seconds.
    pub max_cpu_seconds: u64,
    /// The `docker-removal-worker` executable. Defaults to the one next to the running
    /// executable.
    pub worker_program: Option<PathBuf>,
}

impl DockerRemovalOptions {
    /// Options with the given resource limits.
    pub fn new(
        max_concurrent_removals: usize,
        max_entry_visits: u64,
        max_cpu_seconds: u64,
    ) -> Self {
        DockerRemovalOptions {
            max_concurrent_removals,
            max_entry_visits,
            max_cpu_seconds,
            worker_program: None,
        }
    }

    /// Use this `docker-removal-worker` executable.
    pub fn with_worker_program(mut self, program: impl Into<PathBuf>) -> Self {
        self.worker_program = Some(program.into());
        self
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.max_concurrent_removals == 0
            || self.max_entry_visits == 0
            || self.max_cpu_seconds == 0
        {
            return Err(value_error(
                "Docker removal resource limits must be positive",
            ));
        }
        Ok(())
    }
}

fn value_error(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::SandboxConfigInvalid, message)
}

fn runtime_error(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::WorkspaceStartError, message)
}

fn is_relative_to(path: &str, base: &str) -> bool {
    path == base || path.starts_with(&format!("{}/", base.trim_end_matches('/')))
}

type Configuration = (String, Vec<(String, bool, Option<String>)>);

fn configuration(manifest: &Manifest) -> SandboxResult<Configuration> {
    let mut grants = Vec::new();
    for grant in manifest.extra_path_grants() {
        let host = match &grant.host_path {
            Some(_) => Some(sandbox_path_grant_host_path(grant)?),
            None => None,
        };
        grants.push((grant.path.clone(), grant.read_only, host));
    }
    Ok((manifest.root().to_string(), grants))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Incarnation {
    pid: u32,
    started_at: String,
}

struct ContainerState {
    attrs: Value,
    incarnation: Incarnation,
}

struct Paused {
    incarnation: Incarnation,
    already_paused: bool,
    attrs: Value,
}

struct Binding {
    worker: Arc<Mutex<Box<dyn RemovalWorker>>>,
    incarnation: Incarnation,
    configuration: Configuration,
    policy: WorkspacePathPolicy,
}

#[derive(Default)]
struct Registry {
    closed: bool,
    bindings: HashMap<String, Arc<Binding>>,
}

struct Inner {
    runner: Arc<dyn DockerRunner>,
    host: Arc<dyn RemovalHost>,
    launcher: Arc<dyn WorkerLauncher>,
    registry: std::sync::Mutex<Registry>,
    slots: Arc<Semaphore>,
    max_entry_visits: u64,
    max_cpu_seconds: u64,
}

/// Owns fixed grant bindings and paused-workload removal on a local Docker host.
///
/// Configure the client with [`DockerSandboxClient::with_removal_service`]. The service must not
/// be shared with untrusted callers or exposed to containers. Only newly created, private
/// containers can acquire a binding. Replacing a bound canonical root invalidates it; changing
/// its original symlink alias does not. Workspace and grant roots must resolve to distinct
/// canonical paths. Recursive removal rejects the workspace root and effective external grant
/// roots, including their ancestors: remove their children to clear them instead.
///
/// Each removal pauses all workload processes until it finishes, which can affect concurrent
/// command deadlines. Stopped or restarted containers need a fresh session; a binding is never
/// reconstructed from saved paths. Read-only host bind mounts outside the private workspace are
/// supported; writable shared mounts and custom security profiles are not. Other host processes
/// that can modify mount sources must be trusted: pausing the container does not pause them.
///
/// Choose the limits for the host and share the service across the managed clients. Each
/// removal child has a CPU-time limit (`RLIMIT_CPU`); entry visits include repeated visits to
/// ancestors. Reaching a limit may leave a partially removed tree. The limits do not impose a
/// wall-clock deadline on stalled kernel I/O. An uncertain worker outcome consumes a concurrency
/// slot until every worker has been stopped and the application replaces the service.
///
/// A removal runs to completion on its own task: dropping the caller's future never leaves a
/// container paused.
///
/// [`DockerSandboxClient::with_removal_service`]: super::docker::DockerSandboxClient::with_removal_service
#[derive(Clone)]
pub struct DockerRemovalService {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for DockerRemovalService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DockerRemovalService")
            .field("max_entry_visits", &self.inner.max_entry_visits)
            .field("max_cpu_seconds", &self.inner.max_cpu_seconds)
            .finish_non_exhaustive()
    }
}

/// Run `future` on its own task so the caller dropping it cannot interrupt it half way.
async fn detached<T: Send + 'static>(
    future: impl std::future::Future<Output = SandboxResult<T>> + Send + 'static,
) -> SandboxResult<T> {
    match tokio::spawn(future).await {
        Ok(result) => result,
        Err(error) => Err(SandboxError::new(
            ErrorCode::WorkspaceStopError,
            format!("The Docker removal task failed: {error}"),
        )),
    }
}

impl DockerRemovalService {
    /// Create a service that reaches the daemon through the `docker` executable.
    ///
    /// Fails closed unless the process runs as root on Linux, because the worker enters
    /// containers' mount namespaces. The caller supplies that privilege; the library never
    /// obtains it.
    pub fn new(options: DockerRemovalOptions) -> SandboxResult<Self> {
        options.validate()?;
        #[cfg(target_os = "linux")]
        let rooted = rustix::process::geteuid().is_root();
        #[cfg(not(target_os = "linux"))]
        let rooted = false;
        if !rooted {
            return Err(runtime_error(
                "DockerRemovalService requires root on the Linux Docker host",
            ));
        }
        let program = match &options.worker_program {
            Some(program) => program.clone(),
            None => default_worker_program()?,
        };
        Ok(Self::from_parts(
            options,
            Arc::new(super::docker::DockerCliRunner),
            Arc::new(ProcHost),
            Arc::new(ProcessLauncher::new(program)),
        ))
    }

    /// Assemble a service from explicit parts, without the root check. This is the seam for
    /// embedding a custom daemon connection or worker transport, and for tests; it grants no
    /// privilege by itself, since the worker the launcher starts still needs root.
    pub fn from_parts(
        options: DockerRemovalOptions,
        runner: Arc<dyn DockerRunner>,
        host: Arc<dyn RemovalHost>,
        launcher: Arc<dyn WorkerLauncher>,
    ) -> Self {
        DockerRemovalService {
            inner: Arc::new(Inner {
                runner,
                host,
                launcher,
                registry: std::sync::Mutex::new(Registry::default()),
                slots: Arc::new(Semaphore::new(options.max_concurrent_removals)),
                max_entry_visits: options.max_entry_visits,
                max_cpu_seconds: options.max_cpu_seconds,
            }),
        }
    }

    /// Like [`from_parts`](Self::from_parts) but validates the limits first, as
    /// [`new`](Self::new) does.
    pub fn try_from_parts(
        options: DockerRemovalOptions,
        runner: Arc<dyn DockerRunner>,
        host: Arc<dyn RemovalHost>,
        launcher: Arc<dyn WorkerLauncher>,
    ) -> SandboxResult<Self> {
        options.validate()?;
        Ok(Self::from_parts(options, runner, host, launcher))
    }

    /// The Docker connection the service uses. The client must use the same one.
    pub fn runner(&self) -> Arc<dyn DockerRunner> {
        self.inner.runner.clone()
    }

    /// Whether `container_id` currently has a live binding.
    pub fn is_bound(&self, container_id: &str) -> bool {
        self.inner
            .registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .bindings
            .contains_key(container_id)
    }

    /// Verify that `container_id` is a running private container on this host and return its
    /// init pid and start time.
    pub async fn verify_container(&self, container_id: &str) -> SandboxResult<(u32, String)> {
        let state = self.inner.state(container_id).await?;
        Ok((state.incarnation.pid, state.incarnation.started_at))
    }

    /// Bind before a newly created session is returned to its trusted application.
    pub async fn bind_new(&self, container_id: &str, manifest: &Manifest) -> SandboxResult<()> {
        let inner = self.inner.clone();
        let (id, manifest) = (container_id.to_string(), manifest.clone());
        detached(async move { inner.bind_new(&id, &manifest, false).await }).await
    }

    /// Bind a freshly started container, first creating its workspace with the image's default
    /// user. Used by the Docker client for new sessions.
    pub async fn bind_new_with_workspace_bootstrap(
        &self,
        container_id: &str,
        manifest: &Manifest,
    ) -> SandboxResult<()> {
        let inner = self.inner.clone();
        let (id, manifest) = (container_id.to_string(), manifest.clone());
        detached(async move { inner.bind_new(&id, &manifest, true).await }).await
    }

    /// Check that `manifest` is still the configuration the live binding was created for.
    pub async fn assert_bound(&self, container_id: &str, manifest: &Manifest) -> SandboxResult<()> {
        let inner = self.inner.clone();
        let (id, manifest) = (container_id.to_string(), manifest.clone());
        detached(async move {
            let _guard = inner.bound(&id, &manifest).await?;
            Ok(())
        })
        .await
    }

    /// Authorize and remove `path` while the workload is paused; never uses `docker exec`.
    ///
    /// `user` is the Docker user to remove as; without one the container's configured user (or
    /// `0`) is used.
    pub async fn remove(
        &self,
        container_id: &str,
        manifest: &Manifest,
        path: &str,
        user: Option<&str>,
    ) -> SandboxResult<()> {
        let inner = self.inner.clone();
        let (id, manifest, path) = (container_id.to_string(), manifest.clone(), path.to_string());
        let user = user.map(str::to_string);
        detached(async move { inner.remove(&id, &manifest, &path, user.as_deref()).await }).await
    }

    /// Release authority after its container has been deleted.
    pub async fn release(&self, container_id: &str) -> SandboxResult<()> {
        let inner = self.inner.clone();
        let id = container_id.to_string();
        detached(async move { inner.release(&id).await }).await
    }

    /// Release every worker. An uncertain operation does not automatically thaw its container.
    ///
    /// Every worker is attempted; the first failure is reported.
    pub async fn close(&self) -> SandboxResult<()> {
        let inner = self.inner.clone();
        detached(async move {
            let bindings: Vec<Arc<Binding>> = {
                let mut registry = inner
                    .registry
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                registry.closed = true;
                registry
                    .bindings
                    .drain()
                    .map(|(_, binding)| binding)
                    .collect()
            };
            let mut first: Option<SandboxError> = None;
            for binding in bindings {
                if let Err(error) = binding.worker.lock().await.close().await {
                    first.get_or_insert_with(|| close_error(&error));
                }
            }
            first.map_or(Ok(()), Err)
        })
        .await
    }
}

fn close_error(error: &io::Error) -> SandboxError {
    SandboxError::new(
        ErrorCode::WorkspaceStopError,
        format!("Failed to stop a Docker removal worker: {error}"),
    )
}

fn failure_to_start_error(failure: WorkerFailure) -> SandboxError {
    match failure {
        WorkerFailure::Os { reason, .. } => {
            runtime_error(format!("Docker removal worker request failed: {reason}"))
        }
        WorkerFailure::Transport(message) => runtime_error(message),
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
        Some(Value::Number(number)) => number.as_f64() != Some(0.0),
        Some(Value::Bool(true)) => true,
    }
}

fn is_unset(value: Option<&Value>, allowed: &[&str]) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => allowed.contains(&text.as_str()),
        Some(_) => false,
    }
}

fn is_paused(attrs: &Value) -> bool {
    attrs["State"]["Paused"].as_bool().unwrap_or(false)
}

fn is_container_id(identity: &str) -> bool {
    identity.len() == 64
        && identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl Inner {
    async fn docker(&self, argv: &[&str]) -> SandboxResult<crate::sandbox::types::ExecResult> {
        let argv: Vec<String> = argv.iter().map(|part| part.to_string()).collect();
        self.runner.run(&argv, None).await
    }

    async fn docker_ok(&self, argv: &[&str]) -> SandboxResult<crate::sandbox::types::ExecResult> {
        let output = self.docker(argv).await?;
        if output.exit_code != 0 {
            return Err(runtime_error(format!(
                "`docker {}` failed: {}",
                argv.join(" "),
                output.stderr_text().trim()
            )));
        }
        Ok(output)
    }

    async fn inspect(&self, container_id: &str) -> SandboxResult<Value> {
        let output = self
            .docker_ok(&["container", "inspect", container_id])
            .await?;
        let parsed: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
            runtime_error(format!("`docker inspect` returned bad JSON: {error}"))
        })?;
        parsed
            .as_array()
            .and_then(|items| items.first().cloned())
            .ok_or_else(|| runtime_error("`docker inspect` returned no container"))
    }

    /// Check that the container is a running private container on this host.
    async fn state(&self, container_id: &str) -> SandboxResult<ContainerState> {
        let attrs = self.inspect(container_id).await?;
        let info_output = self.docker_ok(&["info", "--format", "{{json .}}"]).await?;
        let daemon: Value = serde_json::from_slice(&info_output.stdout)
            .map_err(|error| runtime_error(format!("`docker info` returned bad JSON: {error}")))?;
        let version_output = self
            .docker_ok(&["version", "--format", "{{.Server.Version}}"])
            .await?;
        let version = String::from_utf8_lossy(&version_output.stdout)
            .trim()
            .to_string();
        let major: i64 = version
            .split('.')
            .next()
            .and_then(|major| major.parse().ok())
            .ok_or_else(|| runtime_error("`docker version` returned no server version"))?;
        let security: Vec<&str> = daemon["SecurityOptions"]
            .as_array()
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        // Kernel-submitted namespace operations must not outlive the paused tasks. Docker's
        // builtin profile blocks io_uring; custom profiles are not equivalent.
        if major < 26
            || !security.contains(&"name=seccomp,profile=builtin")
            || security.contains(&"name=selinux")
            || daemon["DefaultRuntime"].as_str() != Some("runc")
        {
            return Err(value_error(
                "Docker removal requires Docker 26+ with builtin seccomp and runc",
            ));
        }
        let host = &attrs["HostConfig"];
        let state = &attrs["State"];
        let bad_mount = attrs["Mounts"].as_array().is_some_and(|mounts| {
            mounts.iter().any(|mount| {
                mount["Type"].as_str() != Some("bind")
                    || mount["RW"].as_bool() != Some(false)
                    || mount["Propagation"].as_str() != Some("rprivate")
            })
        });
        if bad_mount
            || truthy(host.get("Privileged"))
            || truthy(host.get("CapAdd"))
            || truthy(host.get("CapDrop"))
            || truthy(host.get("GroupAdd"))
            || truthy(host.get("SecurityOpt"))
            || !is_unset(host.get("Runtime"), &["", "runc"])
            || !is_unset(host.get("PidMode"), &[""])
            || !is_unset(host.get("UsernsMode"), &["", "host"])
            || !state["Running"].as_bool().unwrap_or(false)
            || state["Restarting"].as_bool().unwrap_or(false)
        {
            return Err(value_error(
                "Docker removal service requires a running private container",
            ));
        }
        let pid = state["Pid"]
            .as_u64()
            .and_then(|pid| u32::try_from(pid).ok())
            .ok_or_else(|| {
                value_error("Docker removal service requires a running private container")
            })?;
        let identity = attrs["Id"].as_str().unwrap_or_default();
        if !is_container_id(identity) {
            return Err(value_error(
                "Docker removal service requires a complete container ID",
            ));
        }
        let cgroups = self
            .host
            .read_proc(pid, "cgroup")
            .map_err(|error| value_error(format!("Cannot read the container's cgroup: {error}")))?;
        let docker_suffix = format!("/docker/{identity}");
        let scope_suffix = format!("/docker-{identity}.scope");
        if !cgroups
            .lines()
            .any(|line| line.ends_with(&docker_suffix) || line.ends_with(&scope_suffix))
        {
            return Err(value_error(
                "Docker container is not on this service's host",
            ));
        }
        for name in ["uid_map", "gid_map"] {
            let mapping = self.host.read_proc(pid, name).map_err(|error| {
                value_error(format!("Cannot read the container's {name}: {error}"))
            })?;
            if mapping.split_whitespace().collect::<Vec<_>>() != ["0", "0", "4294967295"] {
                return Err(value_error(
                    "Docker removal service does not support remapped users",
                ));
            }
        }
        let started_at = state["StartedAt"].as_str().unwrap_or_default().to_string();
        Ok(ContainerState {
            attrs,
            incarnation: Incarnation { pid, started_at },
        })
    }

    async fn pause_begin(&self, container_id: &str) -> SandboxResult<Paused> {
        let before = self.state(container_id).await?;
        let already_paused = is_paused(&before.attrs);
        if !already_paused {
            self.docker_ok(&["pause", container_id]).await?;
        }
        // The Docker pause request completes before exec or filesystem work begins.
        let after = self.state(container_id).await?;
        if after.incarnation != before.incarnation || !is_paused(&after.attrs) {
            return Err(runtime_error(
                "Docker container changed while pausing; left paused",
            ));
        }
        Ok(Paused {
            incarnation: before.incarnation,
            already_paused,
            attrs: after.attrs,
        })
    }

    /// Thaw the container unless the caller owned the pause or a worker outcome is unknown.
    async fn pause_end(
        &self,
        container_id: &str,
        paused: &Paused,
        worker_uncertain: bool,
    ) -> SandboxResult<()> {
        if paused.already_paused || worker_uncertain {
            return Ok(());
        }
        let current = self.state(container_id).await?;
        if current.incarnation == paused.incarnation && is_paused(&current.attrs) {
            self.docker_ok(&["unpause", container_id]).await?;
        }
        Ok(())
    }

    async fn bind_new(
        self: &Arc<Self>,
        container_id: &str,
        manifest: &Manifest,
        bootstrap_workspace: bool,
    ) -> SandboxResult<()> {
        {
            let registry = self
                .registry
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if registry.closed {
                return Err(value_error("Docker removal service is closed"));
            }
            if registry.bindings.contains_key(container_id) {
                return Err(value_error("Docker removal authority is already bound"));
            }
        }
        if manifest
            .extra_path_grants()
            .iter()
            .any(|grant| grant.host_path.is_some() && !grant.read_only)
        {
            return Err(value_error(
                "Docker removal service does not support writable shared host paths",
            ));
        }
        validate_docker_path_grants(manifest)?;
        let attrs = self.inspect(container_id).await?;
        assert_container_path_grants_match(&attrs, manifest)?;
        if bootstrap_workspace {
            // Reject unsupported mounts and security settings before an image symlink could
            // redirect mkdir into a shared host directory.
            self.state(container_id).await?;
            // Only the client requests bootstrap, for fresh containers before application
            // workloads run. Preserve the image's default user.
            let output = self
                .docker(&[
                    "exec",
                    "-w",
                    "/",
                    container_id,
                    "mkdir",
                    "-p",
                    "--",
                    manifest.root(),
                ])
                .await?;
            if output.exit_code != 0 {
                return Err(runtime_error(
                    "Unable to create Docker workspace before removal binding",
                ));
            }
        }
        let paused = self.pause_begin(container_id).await?;
        let (outcome, uncertain) = self.bind_in_pause(container_id, manifest, &paused).await;
        let cleanup = self.pause_end(container_id, &paused, uncertain).await;
        match outcome {
            Ok(()) => cleanup,
            Err(error) => Err(error),
        }
    }

    /// Bind the roots while the container is paused. Returns the outcome and whether a worker
    /// transport failure left the result unknown (which keeps the container paused).
    async fn bind_in_pause(
        self: &Arc<Self>,
        container_id: &str,
        manifest: &Manifest,
        paused: &Paused,
    ) -> (SandboxResult<()>, bool) {
        let mut worker = match self.launcher.launch(paused.incarnation.pid).await {
            Ok(worker) => worker,
            Err(failure) => return (Err(failure_to_start_error(failure)), false),
        };
        let result = self.bind_roots(&mut worker, manifest).await;
        let uncertain = worker.uncertain();
        match result {
            Ok(policy) => {
                let registered = {
                    let mut registry = self
                        .registry
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    if registry.closed || registry.bindings.contains_key(container_id) {
                        false
                    } else {
                        match configuration(manifest) {
                            Ok(configuration) => {
                                registry.bindings.insert(
                                    container_id.to_string(),
                                    Arc::new(Binding {
                                        worker: Arc::new(Mutex::new(std::mem::replace(
                                            &mut worker,
                                            Box::new(ClosedWorker),
                                        ))),
                                        incarnation: paused.incarnation.clone(),
                                        configuration,
                                        policy,
                                    }),
                                );
                                true
                            }
                            Err(_) => false,
                        }
                    }
                };
                if registered {
                    (Ok(()), uncertain)
                } else {
                    let _ = worker.close().await;
                    (
                        Err(value_error("Docker removal authority cannot be registered")),
                        uncertain,
                    )
                }
            }
            Err(error) => {
                let _ = worker.close().await;
                (Err(error), uncertain)
            }
        }
    }

    async fn bind_roots(
        &self,
        worker: &mut Box<dyn RemovalWorker>,
        manifest: &Manifest,
    ) -> SandboxResult<WorkspacePathPolicy> {
        let mut paths = vec![manifest.root().to_string()];
        paths.extend(
            manifest
                .extra_path_grants()
                .iter()
                .map(|grant| grant.path.clone()),
        );
        let response = worker
            .request(json!({"operation": "bind", "paths": paths}))
            .await
            .map_err(failure_to_start_error)?;
        let bound: Vec<String> = response["paths"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|path| path.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let grants = manifest.extra_path_grants();
        if bound.len() != grants.len() + 1 {
            return Err(runtime_error(
                "Docker removal worker returned unexpected bindings",
            ));
        }
        // Linux canonical names must not be reinterpreted as caller path syntax.
        if bound.iter().any(|path| path.contains('\\')) {
            return Err(value_error(
                "Docker removal does not support canonical paths containing backslashes",
            ));
        }
        let root = &bound[0];
        for (grant, mounted) in grants.iter().zip(&bound[1..]) {
            if grant.host_path.is_some()
                && (is_relative_to(mounted, root) || is_relative_to(root, mounted))
            {
                return Err(value_error(
                    "Docker removal requires host mounts outside the workspace",
                ));
            }
        }
        let mut unique = bound.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != bound.len() {
            return Err(value_error(
                "Docker removal requires distinct canonical workspace and grant roots",
            ));
        }
        let canonical_grants: Vec<SandboxPathGrant> = grants
            .iter()
            .zip(&bound[1..])
            .map(|(grant, path)| SandboxPathGrant {
                path: path.clone(),
                ..grant.clone()
            })
            .collect();
        WorkspacePathPolicy::with_grants(root, canonical_grants)
    }

    /// The live binding for `container_id`, locked for the caller's whole operation.
    async fn bound(
        &self,
        container_id: &str,
        manifest: &Manifest,
    ) -> SandboxResult<(
        Arc<Binding>,
        tokio::sync::OwnedMutexGuard<Box<dyn RemovalWorker>>,
    )> {
        let original =
            || value_error("Docker removal requires the original live authority binding");
        let binding = self
            .registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .bindings
            .get(container_id)
            .cloned()
            .ok_or_else(original)?;
        let guard = binding.worker.clone().lock_owned().await;
        let current = self
            .registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .bindings
            .get(container_id)
            .cloned();
        if !current.is_some_and(|current| Arc::ptr_eq(&current, &binding))
            || binding.configuration != configuration(manifest)?
        {
            return Err(original());
        }
        if guard.uncertain() || self.state(container_id).await?.incarnation != binding.incarnation {
            return Err(value_error("Docker removal authority is no longer usable"));
        }
        Ok((binding, guard))
    }

    async fn remove(
        self: &Arc<Self>,
        container_id: &str,
        manifest: &Manifest,
        path: &str,
        user: Option<&str>,
    ) -> SandboxResult<()> {
        let (binding, mut worker) = self.bound(container_id, manifest).await?;
        let original_policy = WorkspacePathPolicy::with_grants(
            manifest.root(),
            manifest.extra_path_grants().to_vec(),
        )?;
        let original = original_policy.normalize_sandbox_path(path, false)?;
        let Ok(permit) = self.slots.clone().try_acquire_owned() else {
            return Err(removal_error(&original, "docker_removal_capacity"));
        };
        let paused = match self.pause_begin(container_id).await {
            Ok(paused) => paused,
            Err(error) => {
                permit_release(permit, worker.uncertain());
                return Err(error);
            }
        };
        // A relative request keeps the workspace's original spelling, so the worker can check the
        // alias still resolves to the pinned root and preserve the user's traversal permissions.
        let workspace_root = (!path.starts_with('/')).then(|| manifest.root());
        let outcome = self
            .remove_in_pause(
                &binding,
                &mut worker,
                &paused,
                &original,
                workspace_root,
                user,
            )
            .await;
        let uncertain = worker.uncertain();
        let cleanup = self.pause_end(container_id, &paused, uncertain).await;
        permit_release(permit, uncertain);
        match outcome {
            Ok(()) => cleanup,
            Err(error) => Err(error),
        }
    }

    async fn remove_in_pause(
        &self,
        binding: &Binding,
        worker: &mut tokio::sync::OwnedMutexGuard<Box<dyn RemovalWorker>>,
        paused: &Paused,
        original: &str,
        workspace_root: Option<&str>,
        user: Option<&str>,
    ) -> SandboxResult<()> {
        let inspection = worker
            .request(json!({
                "operation": "inspect",
                "path": original,
                "workspace_root": workspace_root,
            }))
            .await
            .map_err(|failure| removal_failure(original, failure))?;
        let target = inspection["path"].as_str().unwrap_or_default();
        if !target.is_empty() {
            // Policy coercion would treat literal Linux backslashes as separators.
            if target.contains('\\') {
                return Err(removal_error(original, "docker_removal_canonical_path"));
            }
            let is_directory = inspection["is_directory"].as_bool().unwrap_or(false);
            let root = binding.policy.normalize_sandbox_path(".", false)?;
            let mut live_roots = vec![root.clone()];
            for (grant, _) in binding.policy.extra_path_grant_rules()? {
                if !is_relative_to(&grant, &root) {
                    live_roots.push(grant);
                }
            }
            if live_roots.iter().any(|bound_root| {
                target == bound_root || (is_directory && is_relative_to(bound_root, target))
            }) {
                return Err(removal_error(original, "docker_removal_bound_root"));
            }
            if is_directory {
                binding.policy.validate_recursive_remove(target)?;
            } else {
                binding.policy.normalize_sandbox_path(target, true)?;
            }
        }
        let configured_user = paused.attrs["Config"]["User"]
            .as_str()
            .filter(|user| !user.is_empty());
        let docker_user = user
            .filter(|user| !user.is_empty())
            .or(configured_user)
            .unwrap_or("0");
        worker
            .request(json!({
                "operation": "remove",
                "user": docker_user,
                "max_entry_visits": self.max_entry_visits,
                "max_cpu_seconds": self.max_cpu_seconds,
            }))
            .await
            .map_err(|failure| removal_failure(original, failure))?;
        Ok(())
    }

    async fn release(&self, container_id: &str) -> SandboxResult<()> {
        let binding = self
            .registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .bindings
            .remove(container_id);
        if let Some(binding) = binding {
            binding
                .worker
                .lock()
                .await
                .close()
                .await
                .map_err(|error| close_error(&error))?;
        }
        Ok(())
    }
}

fn permit_release(permit: tokio::sync::OwnedSemaphorePermit, uncertain: bool) {
    if uncertain {
        // An uncertain worker outcome keeps its concurrency slot until the service is replaced.
        permit.forget();
    }
}

fn removal_error(path: &str, reason: &str) -> SandboxError {
    SandboxError::new(
        ErrorCode::WorkspaceArchiveWriteError,
        format!("Docker removal of `{path}` was refused: {reason}"),
    )
    .with_context("path", path)
    .with_context("reason", reason)
}

fn removal_failure(path: &str, failure: WorkerFailure) -> SandboxError {
    match failure {
        WorkerFailure::Os { errno, reason } => {
            let mut error = SandboxError::new(
                ErrorCode::WorkspaceArchiveWriteError,
                format!("Docker removal of `{path}` failed: {reason}"),
            )
            .with_context("path", path)
            .with_context("reason", "docker_removal_failed")
            .with_context("worker_reason", reason);
            if let Some(errno) = errno {
                error = error.with_context("errno", errno.to_string());
            }
            error
        }
        WorkerFailure::Transport(message) => SandboxError::new(
            ErrorCode::WorkspaceArchiveWriteError,
            format!("Docker removal of `{path}` failed: {message}"),
        )
        .with_context("path", path)
        .with_context("reason", "docker_removal_failed"),
    }
}

/// A worker whose binding has been moved into the registry.
struct ClosedWorker;

#[async_trait]
impl RemovalWorker for ClosedWorker {
    async fn request(&mut self, _request: Value) -> Result<Value, WorkerFailure> {
        Err(WorkerFailure::Transport(
            "Docker removal worker is closed".to_string(),
        ))
    }

    fn uncertain(&self) -> bool {
        false
    }

    async fn close(&mut self) -> io::Result<()> {
        Ok(())
    }
}

const WORKER_PROGRAM_NAME: &str = "docker-removal-worker";

/// The `docker-removal-worker` executable next to the running one (or, for examples and tests,
/// in its parent directory).
fn default_worker_program() -> SandboxResult<PathBuf> {
    let executable = std::env::current_exe()
        .map_err(|error| runtime_error(format!("Cannot locate the running executable: {error}")))?;
    let mut directory = executable.parent();
    while let Some(candidate) = directory {
        let program = candidate.join(WORKER_PROGRAM_NAME);
        if program.is_file() {
            return Ok(program);
        }
        directory = candidate.parent().filter(|parent| {
            executable
                .ancestors()
                .take(3)
                .any(|ancestor| ancestor == *parent)
        });
    }
    Err(runtime_error(format!(
        "The `{WORKER_PROGRAM_NAME}` executable was not found next to the running one; build it \
         with the `docker-removal` feature and pass its path with \
         `DockerRemovalOptions::with_worker_program`"
    )))
}

/// Starts the `docker-removal-worker` executable.
#[derive(Debug, Clone)]
pub struct ProcessLauncher {
    program: PathBuf,
}

impl ProcessLauncher {
    /// Launch workers from `program`.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        ProcessLauncher {
            program: program.into(),
        }
    }
}

#[async_trait]
impl WorkerLauncher for ProcessLauncher {
    async fn launch(&self, pid: u32) -> Result<Box<dyn RemovalWorker>, WorkerFailure> {
        let mut command = host_command(&self.program, "worker", pid);
        command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        let mut child = command.spawn().map_err(|error| WorkerFailure::Os {
            errno: error.raw_os_error(),
            reason: "OSError".to_string(),
        })?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(WorkerFailure::Transport(
                "Docker removal worker transport failed".to_string(),
            ));
        };
        Ok(Box::new(ProcessWorker {
            program: self.program.clone(),
            pid,
            child,
            stdin: Some(stdin),
            stdout: tokio::io::BufReader::new(stdout),
            uncertain: false,
            requested_path: String::new(),
        }))
    }
}

/// A host command that inherits nothing: no environment, its own process group, no stderr.
fn host_command(program: &std::path::Path, mode: &str, pid: u32) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(program);
    command
        .arg(mode)
        .arg(pid.to_string())
        .env_clear()
        .env("LC_ALL", "C.UTF-8")
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .kill_on_drop(true);
    command
}

/// A running `docker-removal-worker worker <pid>` process.
///
/// `bind` and `inspect` travel to the worker as JSON lines. `remove` starts a sibling
/// `docker-removal-worker remove <pid>` child for the path the last `inspect` selected, because
/// the worker, once inside the container, can no longer start a host executable.
struct ProcessWorker {
    program: PathBuf,
    pid: u32,
    child: tokio::process::Child,
    stdin: Option<tokio::process::ChildStdin>,
    stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
    uncertain: bool,
    requested_path: String,
}

impl ProcessWorker {
    fn lost(&mut self) -> WorkerFailure {
        self.uncertain = true;
        WorkerFailure::Transport("Docker removal worker transport failed".to_string())
    }

    async fn exchange(&mut self, request: &Value) -> Result<Value, WorkerFailure> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(self.lost());
        };
        let line = format!("{request}\n");
        if stdin.write_all(line.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
            return Err(self.lost());
        }
        let mut response = String::new();
        match self.stdout.read_line(&mut response).await {
            Ok(read) if read > 0 => {}
            _ => return Err(self.lost()),
        }
        let Ok(response) = serde_json::from_str::<Value>(&response) else {
            return Err(self.lost());
        };
        outcome_to_result(response)
    }

    async fn remove(&mut self, request: &Value) -> Result<Value, WorkerFailure> {
        let path = std::mem::take(&mut self.requested_path);
        let removal = RemoveRequest {
            path,
            user: request["user"].as_str().unwrap_or("0").to_string(),
            max_entry_visits: request["max_entry_visits"].as_u64().unwrap_or(0),
            max_cpu_seconds: request["max_cpu_seconds"].as_u64().unwrap_or(0),
        };
        let os_failure = |reason: &str, errno: Option<i32>| WorkerFailure::Os {
            errno,
            reason: reason.to_string(),
        };
        let mut command = host_command(&self.program, "remove", self.pid);
        command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| os_failure("OSError", error.raw_os_error()))?;
        if let Some(mut stdin) = child.stdin.take() {
            let payload = removal.to_json().to_string();
            // A child that exits without reading is reported by its exit status below.
            let _ = stdin.write_all(payload.as_bytes()).await;
        }
        let mut result = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout.read_to_end(&mut result).await;
        }
        let status = child.wait().await;
        if !status.is_ok_and(|status| status.success()) || result.is_empty() {
            return Err(os_failure("RuntimeError", None));
        }
        let outcome: Value =
            serde_json::from_slice(&result).map_err(|_| os_failure("RuntimeError", None))?;
        outcome_to_result(outcome)
    }
}

fn outcome_to_result(response: Value) -> Result<Value, WorkerFailure> {
    if response["ok"].as_bool() == Some(true) {
        return Ok(response);
    }
    Err(WorkerFailure::Os {
        errno: response["errno"]
            .as_i64()
            .and_then(|errno| i32::try_from(errno).ok()),
        reason: response["reason"].as_str().unwrap_or("OSError").to_string(),
    })
}

#[async_trait]
impl RemovalWorker for ProcessWorker {
    async fn request(&mut self, request: Value) -> Result<Value, WorkerFailure> {
        match request["operation"].as_str() {
            Some("remove") => {
                if self.requested_path.is_empty() {
                    return Err(WorkerFailure::Os {
                        errno: None,
                        reason: "ValueError".to_string(),
                    });
                }
                self.remove(&request).await
            }
            Some("inspect") => {
                self.requested_path.clear();
                let response = self.exchange(&request).await?;
                self.requested_path = request["path"].as_str().unwrap_or_default().to_string();
                Ok(response)
            }
            _ => self.exchange(&request).await,
        }
    }

    fn uncertain(&self) -> bool {
        self.uncertain
    }

    async fn close(&mut self) -> io::Result<()> {
        let mut first: Option<io::Error> = None;
        if let Some(mut stdin) = self.stdin.take()
            && let Err(error) = stdin.shutdown().await
        {
            first.get_or_insert(error);
        }
        if let Err(error) = self.child.wait().await {
            first.get_or_insert(error);
        }
        first.map_or(Ok(()), Err)
    }
}

#[async_trait]
impl super::docker::RemovalAuthority for DockerRemovalService {
    fn runner(&self) -> Arc<dyn DockerRunner> {
        DockerRemovalService::runner(self)
    }

    async fn bind_new(&self, container_id: &str, manifest: &Manifest) -> SandboxResult<()> {
        self.bind_new_with_workspace_bootstrap(container_id, manifest)
            .await
    }

    async fn assert_bound(&self, container_id: &str, manifest: &Manifest) -> SandboxResult<()> {
        DockerRemovalService::assert_bound(self, container_id, manifest).await
    }

    async fn remove(
        &self,
        container_id: &str,
        manifest: &Manifest,
        path: &str,
        user: Option<&str>,
    ) -> SandboxResult<()> {
        DockerRemovalService::remove(self, container_id, manifest, path, user).await
    }

    async fn release(&self, container_id: &str) -> SandboxResult<()> {
        DockerRemovalService::release(self, container_id).await
    }
}
