//! Running a [`SandboxAgent`] through the ordinary [`Runner`]. Mirrors
//! `agents.sandbox.runtime` and `agents.sandbox.runtime_session_manager`.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::Agent;
use crate::error::{AgentsError, Result};
use crate::items::RunInput;
use crate::result::{RunResult, RunResultStreaming};
use crate::run::{RunOptions, Runner};
use crate::run_config::ModelInputData;
use crate::sandbox::agent::{PreparedSandboxGraph, SandboxAgent};
use crate::sandbox::capabilities::CapabilityBinding;
use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::errors::ErrorCode;
use crate::sandbox::manifest::{Manifest, SandboxConcurrencyLimits};
use crate::sandbox::memory::SandboxMemoryGenerationManager;
use crate::sandbox::memory::rollouts::{
    RolloutTerminalMetadata, TerminalState, build_rollout_payload,
};
use crate::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, persist_session_snapshot,
};
use crate::sandbox::snapshot::{SnapshotSpec, resolve_default_local_snapshot_spec};
use crate::sandbox::state::SandboxSessionState;
use crate::sandbox::types::SandboxArchiveLimits;
use crate::sandbox::workspace_paths::SandboxWorkspaceScope;

/// Where a sandbox run executes, mirroring Python's `RunConfig(sandbox=...)` options.
///
/// Either [`client`](Self::client) or [`session`](Self::session) must be set: a client creates
/// (and tears down) a session for the run, while a session is reused as-is and never torn down
/// by the runner. With [`session_state`](Self::session_state) the client reattaches to the
/// session a previous run left suspended.
#[derive(Debug, Clone, Default)]
pub struct SandboxRunConfig {
    /// Sandbox client used to create or resume a session for the run.
    pub client: Option<Arc<dyn SandboxClient>>,
    /// A live sandbox session to reuse instead of creating one.
    pub session: Option<Arc<dyn SandboxSession>>,
    /// State persisted by a previous run, used to resume its workspace.
    ///
    /// The state carries no manifest contents: the trusted manifest still comes from the agent
    /// or from [`manifest`](Self::manifest), and the client rejects a mismatch.
    pub session_state: Option<SandboxSessionState>,
    /// Manifest override, taking precedence over [`SandboxAgent::default_manifest`].
    pub manifest: Option<Manifest>,
    /// Identifier of the memory rollout this run belongs to.
    ///
    /// Runs that share a rollout id are consolidated into one memory artifact. A fresh id is
    /// generated when this is unset.
    pub rollout_id: Option<String>,
    /// Model-facing working directory, relative to the sandbox workspace root.
    ///
    /// Relative paths used by the capability tools resolve from this directory. It changes
    /// relative-path resolution only: it does not confine the run to `cwd`, nor change
    /// [`Manifest::root`](Manifest::root).
    pub cwd: Option<String>,
    /// Whether the handoff graph shares one session or gets one per agent.
    pub session_mode: SandboxSessionMode,
    /// Materialization concurrency limits, overriding the ones on the manifest.
    ///
    /// Mirrors Python's `SandboxRunConfig.concurrency_limits`.
    pub concurrency_limits: Option<SandboxConcurrencyLimits>,
    /// Resource limits for archives the session extracts.
    ///
    /// Mirrors Python's `SandboxRunConfig.archive_limits`. `None` keeps the default behavior
    /// with no SDK archive resource limits; [`SandboxArchiveLimits::new`] enables the SDK
    /// defaults.
    pub archive_limits: Option<SandboxArchiveLimits>,
    /// Where workspace snapshots are stored.
    ///
    /// Mirrors Python's `SandboxRunConfig.snapshot`. `None` uses the SDK default local snapshot
    /// directory, as Python does.
    pub snapshot: Option<SnapshotSpec>,
}

/// How a [`SandboxAgent`] handoff graph is bound to sandbox sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SandboxSessionMode {
    /// Every agent in the graph shares one session, so a handoff target sees the files the
    /// previous agent wrote. This is the default.
    #[default]
    SharedGraph,
    /// Every agent in the graph gets its own session.
    ///
    /// Mirrors Python's per-agent sessions. Each agent materializes its own manifest and runs
    /// as its own user, which isolates a sub-agent from the files the others hold — at the cost
    /// of a handoff no longer carrying the workspace with it. It needs a
    /// [`client`](SandboxRunConfig::client): a single injected session cannot be split.
    PerAgent,
}

impl SandboxRunConfig {
    /// A configuration that creates a fresh session with `client`.
    pub fn with_client(client: Arc<dyn SandboxClient>) -> Self {
        SandboxRunConfig {
            client: Some(client),
            ..Default::default()
        }
    }

    /// A configuration that reuses a live session owned by the caller.
    pub fn with_session(session: Arc<dyn SandboxSession>) -> Self {
        SandboxRunConfig {
            session: Some(session),
            ..Default::default()
        }
    }

    /// Resume the session a previous run suspended.
    pub fn session_state(mut self, state: SandboxSessionState) -> Self {
        self.session_state = Some(state);
        self
    }

    /// Override the manifest used for the session.
    pub fn manifest(mut self, manifest: Manifest) -> Self {
        self.manifest = Some(manifest);
        self
    }

    /// Set the model-facing working directory.
    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Set the memory rollout this run belongs to.
    pub fn rollout_id(mut self, rollout_id: impl Into<String>) -> Self {
        self.rollout_id = Some(rollout_id.into());
        self
    }

    /// Give every agent in the handoff graph its own sandbox session.
    pub fn session_mode(mut self, mode: SandboxSessionMode) -> Self {
        self.session_mode = mode;
        self
    }

    /// Override the manifest's materialization concurrency limits.
    pub fn concurrency_limits(mut self, limits: SandboxConcurrencyLimits) -> Self {
        self.concurrency_limits = Some(limits);
        self
    }

    /// Store workspace snapshots at `snapshot` instead of the SDK default local directory.
    pub fn snapshot(mut self, snapshot: SnapshotSpec) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Apply resource limits to every archive the session extracts.
    pub fn archive_limits(mut self, limits: SandboxArchiveLimits) -> Self {
        self.archive_limits = Some(limits);
        self
    }
}

/// Run `fut` inside a custom span named `name`, when the current trace exports spans.
///
/// Mirrors the `custom_span(...) if _supports_trace_spans() else nullcontext()` pattern in
/// Python's sandbox runtime: without an active, enabled trace the work runs untraced.
pub async fn sandbox_span<F: std::future::Future>(
    name: &'static str,
    data: serde_json::Value,
    fut: F,
) -> F::Output {
    let exports = crate::tracing::current_trace()
        .and_then(|trace| trace.export())
        .is_some();
    // The preparation and teardown futures are large; boxing keeps them off the caller's stack.
    let fut = Box::pin(fut);
    if exports {
        crate::tracing::custom_span(name, data).run(fut).await
    } else {
        fut.await
    }
}

/// Run a fallible sandbox operation inside a custom span named `name`, recording a failure on the
/// span the way Python's `_apply_trace_finish_data` does.
///
/// On `Err` the span data gains `error.type`, `error_type`, `error_code` and (when known)
/// `error_retryable`, and the span carries an error whose message is the error type and whose data
/// names the operation and error code. A success leaves the span untouched.
pub async fn sandbox_span_result<T, F>(
    name: &'static str,
    data: serde_json::Value,
    fut: F,
) -> crate::sandbox::SandboxResult<T>
where
    F: std::future::Future<Output = crate::sandbox::SandboxResult<T>>,
{
    sandbox_span_result_expecting(name, data, &[], fut).await
}

/// Like [`sandbox_span_result`], but an error whose code is in `expected` leaves the span clean.
///
/// Mirrors `_read_with_expected_span_errors`: a probe whose "not found" is an ordinary answer
/// (is this skill already loaded?) must not paint the trace with an error. The error still
/// reaches the caller, and the session's audit events still record the failure.
pub async fn sandbox_span_result_expecting<T, F>(
    name: &'static str,
    data: serde_json::Value,
    expected: &[crate::sandbox::ErrorCode],
    fut: F,
) -> crate::sandbox::SandboxResult<T>
where
    F: std::future::Future<Output = crate::sandbox::SandboxResult<T>>,
{
    let exports = crate::tracing::current_trace()
        .and_then(|trace| trace.export())
        .is_some();
    let fut = Box::pin(fut);
    if !exports {
        return fut.await;
    }
    let span = crate::tracing::custom_span(name, data);
    span.start();
    let out = crate::tracing::scope::with_span(span.clone(), fut).await;
    if let Err(error) = &out
        && !expected.contains(&error.code)
    {
        record_span_error(&span, name, error);
    }
    span.finish();
    out
}

fn record_span_error(
    span: &crate::tracing::Span,
    name: &str,
    error: &crate::sandbox::SandboxError,
) {
    let operation = name.strip_prefix("sandbox.").unwrap_or(name).to_string();
    let code = error.code.as_str();
    span.update_data(|data| {
        if let crate::tracing::SpanData::Custom { data, .. } = data
            && let Some(object) = data.as_object_mut()
        {
            object.insert("error.type".into(), "SandboxError".into());
            object.insert("error_type".into(), "SandboxError".into());
            object.insert("error_code".into(), code.into());
            if let Some(retryable) = error.retryable {
                object.insert("error_retryable".into(), retryable.into());
            }
        }
    });
    let mut details = serde_json::json!({"operation": operation, "error_code": code});
    if let Some(retryable) = error.retryable {
        details["error_retryable"] = retryable.into();
    }
    span.set_error("SandboxError", Some(details));
}

/// Run `fut` with tracing masked when `disabled`, so the `sandbox.*` spans inside it are not
/// recorded. Mirrors the disabled `TraceCtxManager` Python enters for cleanup when the run has
/// `tracing_disabled` set.
async fn masked<F: std::future::Future>(disabled: bool, fut: F) -> F::Output {
    if disabled {
        crate::tracing::trace("Sandbox cleanup")
            .disabled(true)
            .build()
            .run(fut)
            .await
    } else {
        fut.await
    }
}

/// Owns the teardown of a sandbox session for the duration of a run.
///
/// A session created from a [`SandboxRunConfig::client`] is released by
/// [`close`](Self::close) or [`suspend`](Self::suspend) (which also releases it, after capturing
/// the state a later run resumes from); a session supplied
/// through [`SandboxRunConfig::session`] is left untouched, because the caller owns it.
#[derive(Debug, Clone)]
pub struct SandboxSessionGuard {
    session: Arc<dyn SandboxSession>,
    client: Option<Arc<dyn SandboxClient>>,
    /// Whether the run that owns the guard has tracing disabled, so teardown records no spans.
    tracing_disabled: bool,
}

impl SandboxSessionGuard {
    /// The live session.
    pub fn session(&self) -> &Arc<dyn SandboxSession> {
        &self.session
    }

    /// Whether closing the guard tears the session down.
    pub fn owns_session(&self) -> bool {
        self.client.is_some()
    }

    /// Tear down the session when the guard owns it.
    ///
    /// PTY sessions are terminated first, so no interactive process outlives the workspace. A
    /// session with a snapshot configured is captured before it is released, as Python's
    /// `stop()` does, so what the run produced outlives the workspace. The session is released
    /// even when the capture fails, and the failure is reported afterwards.
    pub async fn close(self) -> Result<()> {
        self.release(true).await.map(|_| ())
    }

    /// Release the session and return the state needed to resume it later.
    ///
    /// Mirrors Python, where a paused run is cleaned up exactly like a finished one: interactive
    /// processes are terminated, the workspace is captured into the session's snapshot, and the
    /// session is stopped, shut down and deleted. Nothing is left on the host or at the backend;
    /// the returned state names the snapshot a later run restores the workspace from. A session
    /// the caller supplied is only captured, never released, because the caller owns it.
    pub async fn suspend(self) -> Result<SandboxSessionState> {
        let session = self.session.clone();
        let persisted = persist_session_snapshot(session.as_ref()).await;
        // The state is read after the capture, so it names the snapshot just taken, and before
        // the release, because deleting the session invalidates what it reports.
        let state = session.session_state();
        // Capturing happened above; the release must not capture a second time.
        let released = self.release(false).await;
        persisted?;
        released?;
        Ok(state)
    }

    async fn release(self, persist: bool) -> Result<()> {
        let owned = self.client.is_some();
        let tracing_disabled = self.tracing_disabled;
        let teardown = async move {
            let _ = self.session.pty_terminate_all().await;
            match self.client {
                None => Ok(()),
                Some(client) => {
                    // A session with a snapshot configured is captured before it is released,
                    // and released even when the capture fails; the failure is reported after.
                    let persisted = if persist {
                        persist_session_snapshot(self.session.as_ref()).await
                    } else {
                        Ok(())
                    };
                    client.delete(self.session).await?;
                    Ok(persisted?)
                }
            }
        };
        if !owned {
            return teardown.await;
        }
        masked(
            tracing_disabled,
            sandbox_span(
                "sandbox.cleanup",
                serde_json::json!({}),
                sandbox_span(
                    "sandbox.cleanup_sessions",
                    serde_json::json!({"session_count": 1}),
                    teardown,
                ),
            ),
        )
        .await
    }
}

/// The snapshot spec a new session is created with.
///
/// Mirrors Python's `SandboxRuntimeSessionManager._resolve_snapshot_spec`. An explicit spec is
/// returned as given and the default directory is never resolved, so no directory is created for
/// it. Without one, `default` supplies the SDK default directory. Only an I/O failure creating
/// that directory falls back to no snapshots, which is the case Python catches as `OSError`; every
/// other error propagates.
pub async fn resolve_snapshot_spec_with<F, Fut>(
    explicit: Option<&SnapshotSpec>,
    default: F,
) -> crate::sandbox::SandboxResult<SnapshotSpec>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = crate::sandbox::SandboxResult<SnapshotSpec>>,
{
    if let Some(spec) = explicit {
        return Ok(spec.clone());
    }
    match default().await {
        Ok(spec) => Ok(spec),
        Err(error) if error.code == ErrorCode::SnapshotPersistError => Ok(SnapshotSpec::Noop),
        Err(error) => Err(error),
    }
}

/// A [`SandboxAgent`] graph prepared against a live session, ready for [`Runner`].
pub struct PreparedSandboxRun<C> {
    /// The prepared agent graph.
    pub graph: PreparedSandboxGraph<C>,
    /// The session teardown guard for the starting agent.
    pub guard: SandboxSessionGuard,
    /// Teardown guards for the other agents' sessions, in
    /// [`SandboxSessionMode::PerAgent`] mode.
    pub agent_guards: Vec<(String, SandboxSessionGuard)>,
    /// The memory generation manager for this session, when the agent enables generation.
    pub memory_manager: Option<Arc<SandboxMemoryGenerationManager>>,
    /// The memory rollout every run against this session belongs to.
    ///
    /// Taken from [`SandboxRunConfig::rollout_id`], then from the resumed
    /// [`SandboxSessionState::rollout_id`], and generated otherwise, so the two halves of a
    /// paused run form one rollout.
    pub rollout_id: String,
}

impl<C> fmt::Debug for PreparedSandboxRun<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSandboxRun")
            .field("graph", &self.graph)
            .field("session", &self.guard.session().id())
            .field("memory_manager", &self.memory_manager.is_some())
            .field("rollout_id", &self.rollout_id)
            .finish()
    }
}

impl<C: Send + Sync + 'static> PreparedSandboxRun<C> {
    /// The memory generation manager bound to this session, when there is one.
    pub fn memory_manager(&self) -> Option<&Arc<SandboxMemoryGenerationManager>> {
        self.memory_manager.as_ref()
    }

    /// The live sandbox session the graph is bound to.
    pub fn session(&self) -> &Arc<dyn SandboxSession> {
        self.guard.session()
    }

    /// The prepared agent the run starts from.
    pub fn agent(&self) -> &Arc<crate::agent::Agent<C>> {
        self.graph.starting_agent()
    }
}

/// A finished sandbox run plus the state needed to resume its workspace.
pub struct SandboxRunOutcome<C> {
    /// The result of the agent run.
    pub result: RunResult<C>,
    /// Session state to persist, present when the run left the sandbox suspended.
    ///
    /// This is set when the run ended with approval interruptions: the workspace and the backend
    /// resource are kept, so a later run can continue with
    /// [`SandboxRunConfig::session_state`].
    pub sandbox_state: Option<SandboxSessionState>,
    /// Session state for the other agents' sessions, in
    /// [`SandboxSessionMode::PerAgent`] mode.
    ///
    /// Mirrors Python's `sessions_by_agent`. Rust rejects duplicate agent names in a graph, so
    /// the agent name is the resume key and Python's key-disambiguation machinery is not needed.
    pub sandbox_states_by_agent: std::collections::BTreeMap<String, SandboxSessionState>,
    /// The memory generation manager of the session, when the run did not flush it.
    ///
    /// The runner flushes only when it tears down a session it owns, so this is set for a
    /// suspended run and for a session the caller supplied.
    pub memory_manager: Option<Arc<SandboxMemoryGenerationManager>>,
}

impl<C: Send + Sync + 'static> SandboxRunOutcome<C> {
    /// The memory generation manager of the session, when the run did not flush it.
    pub fn memory_manager(&self) -> Option<&Arc<SandboxMemoryGenerationManager>> {
        self.memory_manager.as_ref()
    }
}

impl<C> fmt::Debug for SandboxRunOutcome<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxRunOutcome")
            .field("result", &self.result)
            .field("sandbox_state", &self.sandbox_state)
            .field("sandbox_states_by_agent", &self.sandbox_states_by_agent)
            .field("memory_manager", &self.memory_manager.is_some())
            .finish()
    }
}

/// Runs [`SandboxAgent`]s: creates, resumes or reuses a sandbox session, binds the capability
/// tools of the whole handoff graph to it, hands the prepared agent to [`Runner`], and either
/// tears the session down or suspends it afterwards.
#[derive(Debug, Clone, Copy, Default)]
pub struct SandboxRunner;

impl SandboxRunner {
    /// Create, resume or reuse the sandbox session and prepare the agent graph against it.
    ///
    /// The caller owns the returned [`SandboxSessionGuard`] and must `close()` or `suspend()` it.
    pub async fn prepare<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        config: &SandboxRunConfig,
    ) -> Result<PreparedSandboxRun<C>> {
        sandbox_span(
            "sandbox.prepare_agent",
            serde_json::json!({"agent_name": agent.name()}),
            Self::prepare_untraced(agent, config),
        )
        .await
    }

    async fn prepare_untraced<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        config: &SandboxRunConfig,
    ) -> Result<PreparedSandboxRun<C>> {
        let scope = SandboxWorkspaceScope::from_cwd(config.cwd.as_deref())?;
        if let Some(limits) = &config.concurrency_limits {
            limits.validate()?;
        }
        if let Some(limits) = &config.archive_limits {
            limits.validate()?;
        }
        let (session, client) = match (&config.session, &config.client) {
            (Some(session), _) => {
                if config.session_mode == SandboxSessionMode::PerAgent {
                    return Err(AgentsError::user(
                        "SandboxSessionMode::PerAgent needs SandboxRunConfig::client: one \
                         injected session cannot be split across the handoff graph",
                    ));
                }
                // An injected session is already live, so the capabilities process the manifest
                // the session was *started with*, not the agent's default: the session owns the
                // workspace root, the accounts and the mounts, and the capabilities only add
                // entries to them. Mirrors Python's `_process_live_session_manifest`.
                let mut processed = agent.resolved_manifest(Some(session.manifest().clone()));
                // The limits govern the entries the capabilities add to the live workspace.
                if let Some(limits) = config.concurrency_limits {
                    processed = processed.with_concurrency_limits(limits)?;
                }
                // An unsafe manifest is refused before the session is so much as probed or
                // started, so nothing of it ever runs.
                validate_mount_credential_boundaries(&processed)?;
                // A caller may hand over a session it created but never started. Starting it is
                // the run's job; a session that is already running is left alone, so a caller
                // that is driving the same workspace across several runs never has its
                // workspace re-applied underneath it. Mirrors Python's `ensure_session`.
                if !session.running().await? {
                    session.start().await?;
                }
                apply_live_session_manifest(session.as_ref(), &processed).await?;
                session.set_archive_limits(config.archive_limits);
                session.set_concurrency_limits(config.concurrency_limits.unwrap_or_default());
                (session.clone(), None)
            }
            (None, Some(client)) => {
                let manifest = Self::effective_manifest(agent, config)?;
                let span_data = serde_json::json!({
                    "agent_name": agent.name(),
                    "backend_id": client.backend_id(),
                });
                let session = match &config.session_state {
                    None => {
                        let snapshot = Self::snapshot_spec(config, client.as_ref()).await?;
                        sandbox_span_result(
                            "sandbox.create_session",
                            span_data,
                            client.create_with_snapshot(Some(manifest), snapshot),
                        )
                        .await?
                    }
                    Some(state) => {
                        sandbox_span_result(
                            "sandbox.resume_session",
                            span_data,
                            client.resume(state, manifest),
                        )
                        .await?
                    }
                };
                session.set_archive_limits(config.archive_limits);
                session.set_concurrency_limits(config.concurrency_limits.unwrap_or_default());
                if let Err(error) =
                    sandbox_span_result("sandbox.start", serde_json::json!({}), session.start())
                        .await
                {
                    // Release the backend resources the failed start may already hold.
                    let _ = client.delete(session.clone()).await;
                    return Err(error.into());
                }
                (session, Some(client.clone()))
            }
            (None, None) => {
                return Err(AgentsError::user(
                    "SandboxAgent execution requires SandboxRunConfig::client or \
                     SandboxRunConfig::session",
                ));
            }
        };
        let guard = SandboxSessionGuard {
            session: session.clone(),
            client,
            tracing_disabled: false,
        };
        if let Err(error) = validate_workspace_scope(session.as_ref(), &scope, agent).await {
            let _ = guard.close().await;
            return Err(error);
        }
        let binding = CapabilityBinding::new(session)
            .with_run_as(agent.run_as.clone())
            .with_scope(scope.clone());
        // One manager per session, so reusing an agent across sessions or concurrent runs never
        // shares rollout state or latches generation off.
        let memory_manager = match agent.memory_capability().map(|m| m.build_manager()) {
            None => None,
            Some(Ok(manager)) => manager,
            Some(Err(error)) => {
                let _ = guard.close().await;
                return Err(error);
            }
        };
        let rollout_id = config
            .rollout_id
            .clone()
            .or_else(|| {
                config
                    .session_state
                    .as_ref()
                    .and_then(|state| state.rollout_id.clone())
            })
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        // In per-agent mode every other agent in the graph gets a session of its own, created
        // from the same client. The starting agent keeps the session above, so a graph of one
        // agent behaves identically in both modes.
        let mut agent_guards: Vec<(String, SandboxSessionGuard)> = Vec::new();
        let mut bindings: HashMap<String, CapabilityBinding> = HashMap::new();
        if config.session_mode == SandboxSessionMode::PerAgent {
            let names = match agent.graph_agent_names() {
                Ok(names) => names,
                Err(error) => {
                    let _ = guard.close().await;
                    return Err(error);
                }
            };
            let client = config
                .client
                .clone()
                .expect("per-agent mode requires a client");
            for name in names {
                if name == agent.name() {
                    continue;
                }
                match Self::create_agent_session(&client, agent, &name, config).await {
                    Ok(agent_session) => {
                        bindings.insert(
                            name.clone(),
                            CapabilityBinding::new(agent_session.session().clone())
                                .with_scope(scope.clone()),
                        );
                        agent_guards.push((name, agent_session));
                    }
                    Err(error) => {
                        for (_, extra) in agent_guards {
                            let _ = extra.close().await;
                        }
                        let _ = guard.close().await;
                        return Err(error);
                    }
                }
            }
        }

        match agent.prepare_graph_with_bindings(&binding, &bindings).await {
            Ok(graph) => Ok(PreparedSandboxRun {
                graph,
                guard,
                agent_guards,
                memory_manager,
                rollout_id,
            }),
            Err(error) => {
                for (_, extra) in agent_guards {
                    let _ = extra.close().await;
                }
                let _ = guard.close().await;
                Err(error)
            }
        }
    }

    /// The snapshot spec a fresh session is created with: the run config's, else one the client
    /// was configured with, else the SDK default local snapshot directory. Mirrors
    /// `_resolve_snapshot_spec`, which has no client-level step because a Python client takes its
    /// snapshot only as a `create` argument.
    async fn snapshot_spec(
        config: &SandboxRunConfig,
        client: &dyn SandboxClient,
    ) -> Result<SnapshotSpec> {
        let explicit = config
            .snapshot
            .clone()
            .or_else(|| client.configured_snapshot());
        Ok(
            resolve_snapshot_spec_with(explicit.as_ref(), resolve_default_local_snapshot_spec)
                .await?,
        )
    }

    /// The manifest a session is created with, with the run config's concurrency limits applied.
    fn effective_manifest<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        config: &SandboxRunConfig,
    ) -> Result<Manifest> {
        let mut manifest = agent.resolved_manifest(config.manifest.clone());
        // In shared-graph mode every agent runs in *this* session, so every agent's `run_as`
        // account has to exist in it. Python never needs this because each agent brings its own
        // session; here a handoff target with its own `run_as` would otherwise run
        // `sudo -u <account that was never created>`.
        if config.session_mode == SandboxSessionMode::SharedGraph {
            for name in agent.graph_agent_names().unwrap_or_default() {
                let Some(node) = agent.graph_agent(&name) else {
                    continue;
                };
                if let Some(run_as) = &node.run_as
                    && !manifest.users().iter().any(|user| user.name == run_as.name)
                {
                    manifest = manifest.with_user(run_as.clone());
                }
            }
        }
        Ok(match config.concurrency_limits {
            None => manifest,
            Some(limits) => manifest.with_concurrency_limits(limits)?,
        })
    }

    /// Create and start the session one named agent of the graph runs in.
    async fn create_agent_session<C: Send + Sync + 'static>(
        client: &Arc<dyn SandboxClient>,
        root: &SandboxAgent<C>,
        name: &str,
        config: &SandboxRunConfig,
    ) -> Result<SandboxSessionGuard> {
        let node = root
            .graph_agent(name)
            .ok_or_else(|| AgentsError::other(format!("sandbox agent `{name}` left the graph")))?;
        // Each agent brings its own manifest and its own `run_as` user; only the explicit
        // `SandboxRunConfig::manifest` override is shared, as it is in Python.
        let manifest = Self::effective_manifest(node.as_ref(), config)?;
        let snapshot = Self::snapshot_spec(config, client.as_ref()).await?;
        let session = client
            .create_with_snapshot(Some(manifest), snapshot)
            .await?;
        session.set_archive_limits(config.archive_limits);
        session.set_concurrency_limits(config.concurrency_limits.unwrap_or_default());
        if let Err(error) = session.start().await {
            let _ = client.delete(session.clone()).await;
            return Err(error.into());
        }
        Ok(SandboxSessionGuard {
            session,
            client: Some(client.clone()),
            tracing_disabled: false,
        })
    }

    /// Run a sandbox agent with a default app context.
    pub async fn run<C: Default + Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
    ) -> Result<RunResult<C>> {
        Self::run_with_options(agent, input, config, RunOptions::default()).await
    }

    /// Run a sandbox agent with explicit [`RunOptions`].
    ///
    /// A run that completes releases the session. A run that pauses on approval interruptions
    /// keeps it: use [`run_to_outcome`](Self::run_to_outcome) to receive the
    /// [`SandboxSessionState`] needed to resume it.
    pub async fn run_with_options<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
        options: RunOptions<C>,
    ) -> Result<RunResult<C>> {
        Self::run_to_outcome(agent, input, config, options)
            .await
            .map(|outcome| outcome.result)
    }

    /// Run a sandbox agent and report whether the sandbox was suspended.
    ///
    /// When the run pauses on approval interruptions the session is suspended rather than torn
    /// down: PTY processes are terminated, the workspace is captured into the session snapshot,
    /// and [`SandboxRunOutcome::sandbox_state`] carries the state to persist alongside
    /// [`RunResult::to_state`](crate::RunResult::to_state).
    pub async fn run_to_outcome<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
        options: RunOptions<C>,
    ) -> Result<SandboxRunOutcome<C>> {
        let tracing_disabled = options.run_config.tracing_disabled;
        let rollout_override = Self::grouping_rollout_id(config, &options);
        let model_provider = options.run_config.model_provider.clone();
        let input = input.into();
        let resuming = matches!(input, RunInput::State(_));
        let (options, starting_agent, deferred) = begin_preparation(
            options,
            agent,
            config,
            tracing_disabled,
            rollout_override,
            resuming,
        )
        .await?;
        let result = Runner::run_with_options(starting_agent, input, options).await;
        // A run that never prepared (a guardrail rejected the input, or preparation failed) has
        // no session to release and no result to carry.
        let rest = deferred.rest.lock().expect("poisoned").take();
        let guard = deferred.guard.lock().expect("poisoned").take();
        let (Some(prepared), Some(guard)) = (rest, guard) else {
            return match result {
                Err(error) => Err(error),
                Ok(_) => Err(AgentsError::user(
                    "the sandbox was never prepared for this run",
                )),
            };
        };

        // Record the run as a memory rollout segment, whether it finished or paused.
        let memory = match &result {
            Ok(result) => active_memory_manager(&prepared, agent.name(), &result.last_agent.name),
            Err(_) => prepared.memory_manager.clone(),
        };
        if let (Some(manager), Ok(result)) = (&memory, &result)
            && let Err(error) = manager
                .enqueue_result(guard.session().as_ref(), &prepared.rollout_id, result)
                .await
        {
            // The error can carry model or tool data, so both logging policies apply.
            crate::logger::log_model_and_tool_action_warning(
                "Failed to enqueue sandbox memory after run",
                &error,
            );
        }

        let suspend = matches!(&result, Ok(result) if !result.interruptions.is_empty());
        // Memory generation runs only on a final close: a suspended session will be resumed, and
        // a caller-owned session is flushed by the caller.
        let mut flushed = false;
        if !suspend
            && guard.owns_session()
            && let Some(manager) = &memory
        {
            flushed = true;
            if let Err(error) = manager.flush(guard.session().clone(), model_provider).await {
                log::warn!(target: crate::logger::LOG_TARGET, "Failed to generate sandbox memory: {error}");
            }
        }

        let rollout_id = prepared.rollout_id.clone();
        let sandbox_states_by_agent =
            release_agent_guards(prepared.agent_guards, suspend, &rollout_id).await;
        let protected = session_protects_mounts(guard.session());
        let sandbox_state = if suspend {
            match guard.suspend().await {
                // Carry the rollout id across the suspend so the resumed half appends to the
                // same rollout instead of starting a new one.
                Ok(state) => Some(state.with_rollout_id(Some(rollout_id))),
                Err(error) => {
                    log_sandbox_failure(protected, "Failed to suspend the sandbox session", &error);
                    None
                }
            }
        } else {
            if let Err(error) = guard.close().await {
                log_sandbox_failure(protected, "Failed to tear down the sandbox session", &error);
            }
            None
        };
        Ok(SandboxRunOutcome {
            result: result?,
            sandbox_state,
            sandbox_states_by_agent,
            // The manager is handed back only when this run did not flush it.
            memory_manager: if flushed { None } else { memory },
        })
    }

    /// The rollout id a run belongs to: an explicit one, the one a resumed state carries, else
    /// the run grouping (conversation, SDK session, trace group, or a generated per-run value).
    /// Mirrors Python's `_sandbox_memory_rollout_id`.
    fn grouping_rollout_id<C>(
        config: &SandboxRunConfig,
        options: &RunOptions<C>,
    ) -> Option<String> {
        if config.rollout_id.is_some()
            || config
                .session_state
                .as_ref()
                .is_some_and(|state| state.rollout_id.is_some())
        {
            return None;
        }
        Some(crate::resolve_run_grouping_id(
            options.conversation_id.as_deref(),
            options.session.as_ref(),
            options.run_config.group_id.as_deref(),
        ))
    }

    /// Run a sandbox agent in streaming mode with a default app context.
    pub async fn run_streamed<C: Default + Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
    ) -> Result<SandboxStreamedRun<C>> {
        Self::run_streamed_with_options(agent, input, config, RunOptions::default()).await
    }

    /// Run a sandbox agent in streaming mode.
    ///
    /// The returned guard must be closed (or suspended) once the stream is drained; tearing the
    /// session down earlier would pull the workspace out from under the running tools.
    pub async fn run_streamed_with_options<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
        options: RunOptions<C>,
    ) -> Result<SandboxStreamedRun<C>> {
        let tracing_disabled = options.run_config.tracing_disabled;
        let rollout_override = Self::grouping_rollout_id(config, &options);
        let model_provider = options.run_config.model_provider.clone();
        let input = input.into();
        let resuming = matches!(input, RunInput::State(_));
        let (options, starting_agent, mut deferred) = begin_preparation(
            options,
            agent,
            config,
            tracing_disabled,
            rollout_override,
            resuming,
        )
        .await?;
        let mut streaming = Runner::run_streamed_with_options(starting_agent, input, options);

        let sandbox_state = Arc::new(Mutex::new(None));
        let starting_name = agent.name().to_string();
        let guard = deferred.guard.clone();
        let hook_state = sandbox_state.clone();
        let hook_guard = guard.clone();
        let prepared_slot = deferred.rest.clone();
        streaming.set_on_complete(Arc::new(move |completion| {
            let hook_state = hook_state.clone();
            let hook_guard = hook_guard.clone();
            let prepared_slot = prepared_slot.clone();
            let model_provider = model_provider.clone();
            let starting_name = starting_name.clone();
            Box::pin(async move {
                // A run stopped before preparation (a rejected input) holds no session.
                let Some(prepared) = prepared_slot.lock().expect("poisoned").take() else {
                    return;
                };
                let memory_manager =
                    active_memory_manager(&prepared, &starting_name, &completion.last_agent_name);
                let PreparedRest {
                    agent_guards,
                    rollout_id,
                    ..
                } = prepared;
                // The caller may have taken the guard over; then the lifecycle is theirs.
                let Some(guard) = hook_guard.lock().expect("poisoned").take() else {
                    return;
                };
                // Record the streamed run as a rollout segment before releasing the session. A
                // failed run has nothing to extract, so it is not recorded: an empty segment
                // would trigger a phase-1 model call on nothing.
                if completion.succeeded
                    && let Some(manager) = &memory_manager
                {
                    let payload = build_rollout_payload(
                        &completion.input,
                        &completion.new_items,
                        completion.final_output.as_ref(),
                        completion.interruptions,
                        &RolloutTerminalMetadata::new(
                            if completion.is_suspended() {
                                TerminalState::Interrupted
                            } else if completion.succeeded {
                                TerminalState::Completed
                            } else {
                                TerminalState::Failed
                            },
                            completion.final_output.is_some(),
                        ),
                    );
                    if let Err(error) = manager
                        .enqueue_rollout_payload(guard.session().as_ref(), &rollout_id, &payload)
                        .await
                    {
                        crate::logger::log_model_and_tool_action_warning(
                            "Failed to enqueue sandbox memory after streamed run",
                            &error,
                        );
                    }
                }
                release_agent_guards(agent_guards, completion.is_suspended(), &rollout_id).await;
                let state = finish_sandbox_run(
                    guard,
                    completion.is_suspended(),
                    memory_manager,
                    model_provider,
                    rollout_id,
                )
                .await;
                *hook_state.lock().expect("poisoned") = state;
            })
        }));

        // The stream runs on its own task. Wait until it has prepared the sandbox (or ended
        // without doing so), so the session is held by the time the caller has the handle. A
        // run whose input guardrail trips drops the hook unused, which also releases this wait.
        let _ = std::mem::replace(&mut deferred.ready, tokio::sync::oneshot::channel().1).await;

        Ok(SandboxStreamedRun {
            streaming,
            guard,
            sandbox_state,
        })
    }
}

/// A streamed sandbox run and the sandbox state it leaves behind.
///
/// The sandbox session is released (or suspended) automatically when the stream finishes, through
/// [`RunResultStreaming::with_on_complete`]. Drain [`streaming`](Self::streaming) to the end — or
/// call [`RunResultStreaming::finish`] — and then read
/// [`sandbox_state`](Self::sandbox_state).
pub struct SandboxStreamedRun<C> {
    /// The streamed run. Consume its events as usual.
    pub streaming: RunResultStreaming<C>,
    guard: Arc<Mutex<Option<SandboxSessionGuard>>>,
    sandbox_state: Arc<Mutex<Option<SandboxSessionState>>>,
}

impl<C> fmt::Debug for SandboxStreamedRun<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxStreamedRun")
            .field("streaming", &self.streaming)
            .field("released", &self.guard.lock().expect("poisoned").is_none())
            .finish()
    }
}

impl<C: Send + Sync + 'static> SandboxStreamedRun<C> {
    /// The live sandbox session this stream is running against, until it is released.
    ///
    /// This is the half of Python's `SandboxRuntime.apply_result_metadata` that attaches
    /// `_sandbox_session` to a `RunResultStreaming`. The other halves are
    /// [`sandbox_state`](Self::sandbox_state) (Python's `_sandbox_resume_state`) and the
    /// completion hook installed by `run_streamed_with_options` (Python's `_sandbox_cleanup`),
    /// which the stream fires for itself rather than leaving to the caller.
    pub fn session(&self) -> Option<Arc<dyn SandboxSession>> {
        self.guard
            .lock()
            .expect("poisoned")
            .as_ref()
            .map(|guard| guard.session().clone())
    }

    /// The session state to persist, once the stream has finished and the run was suspended.
    pub fn sandbox_state(&self) -> Option<SandboxSessionState> {
        self.sandbox_state.lock().expect("poisoned").clone()
    }

    /// Take over the session lifecycle, disarming the automatic teardown.
    ///
    /// Returns `None` once the stream has already finished and released the session.
    pub fn take_guard(&self) -> Option<SandboxSessionGuard> {
        self.guard.lock().expect("poisoned").take()
    }

    /// Hand the session lifecycle back, re-arming the automatic teardown.
    pub fn restore_guard(&self, guard: SandboxSessionGuard) {
        *self.guard.lock().expect("poisoned") = Some(guard);
    }

    /// Drain the stream and release the sandbox session.
    pub async fn finish(&mut self) -> Result<()> {
        self.streaming.finish().await
    }
}

/// Release a sandbox session after a run: record memory, then suspend or tear down.
/// Suspend or close the per-agent sessions of a finished run.
async fn release_agent_guards(
    guards: Vec<(String, SandboxSessionGuard)>,
    suspend: bool,
    rollout_id: &str,
) -> std::collections::BTreeMap<String, SandboxSessionState> {
    let mut states = std::collections::BTreeMap::new();
    for (name, guard) in guards {
        let protected = session_protects_mounts(guard.session());
        if suspend {
            match guard.suspend().await {
                Ok(state) => {
                    states.insert(name, state.with_rollout_id(Some(rollout_id.to_string())));
                }
                Err(error) => {
                    log_sandbox_failure(
                        protected,
                        &format!("Failed to suspend the sandbox session of `{name}`"),
                        &error,
                    );
                }
            }
            continue;
        }
        if let Err(error) = guard.close().await {
            log_sandbox_failure(
                protected,
                &format!("Failed to tear down the sandbox session of `{name}`"),
                &error,
            );
        }
    }
    states
}

async fn finish_sandbox_run(
    guard: SandboxSessionGuard,
    suspend: bool,
    memory_manager: Option<Arc<SandboxMemoryGenerationManager>>,
    model_provider: Arc<dyn crate::models::ModelProvider>,
    rollout_id: String,
) -> Option<SandboxSessionState> {
    if !suspend
        && guard.owns_session()
        && let Some(manager) = &memory_manager
        && let Err(error) = manager.flush(guard.session().clone(), model_provider).await
    {
        log::warn!(target: crate::logger::LOG_TARGET, "Failed to generate sandbox memory: {error}");
    }
    let protected = session_protects_mounts(guard.session());
    if suspend {
        match guard.suspend().await {
            Ok(state) => return Some(state.with_rollout_id(Some(rollout_id))),
            Err(error) => {
                log_sandbox_failure(protected, "Failed to suspend the sandbox session", &error);
                return None;
            }
        }
    }
    if let Err(error) = guard.close().await {
        log_sandbox_failure(protected, "Failed to tear down the sandbox session", &error);
    }
    None
}

/// Whether the session's manifest holds a mount whose credentials a provider failure could echo.
fn session_protects_mounts(session: &Arc<dyn crate::sandbox::session::SandboxSession>) -> bool {
    session
        .manifest()
        .mount_targets()
        .iter()
        .any(|(mount, _)| mount.strategy.carries_credentials(mount))
}

/// Log a suspend or teardown failure that never fails the run. A session with a credentialed
/// mount logs the sentence alone, whatever the logging policy says, because a provider's error can
/// repeat the mount's configuration; any other failure follows the tool-data policy.
fn log_sandbox_failure(protected: bool, message: &str, error: &dyn std::fmt::Display) {
    if protected {
        log::warn!(target: crate::logger::LOG_TARGET, "{message}");
    } else {
        crate::logger::log_tool_action_warning(message, error);
    }
}

/// Reconcile a live session's manifest with the one the agent's capabilities want.
///
/// Mirrors Python's `_process_live_session_manifest`. An injected session was started with
/// whatever manifest its creator chose; a capability such as Memory or Skills then asks for
/// entries that may not be there. Adding a plain file or directory to a running workspace is
/// safe, so those are materialized. Anything that would change what the session can *reach* is
/// not, and is refused with the reason, because a running session has already handed its
/// workspace, its mounts and its accounts to the model.
///
/// The session keeps its original manifest object: nothing the update may add carries a path
/// grant or a mount, so no consumer of the stored manifest sees a different answer afterwards.
pub async fn apply_live_session_manifest(
    session: &dyn SandboxSession,
    processed: &Manifest,
) -> Result<()> {
    let current = session.manifest();
    if entry_signature_map(current) == entry_signature_map(processed)
        && manifest_topology_matches(current, processed)
    {
        session.validate_manifest_application(current, None).await?;
        return Ok(());
    }
    validate_live_session_host_path_grants(current, processed)?;
    let running = session.running().await?;
    session
        .validate_manifest_application(processed, Some(running))
        .await?;
    if !running {
        // A session that has not started yet materializes the manifest it was built with, so
        // there is nothing to patch into a live workspace.
        return Ok(());
    }
    validate_running_live_session_manifest_update(current, processed)?;
    let additions = diff_live_session_entries(current.entries(), processed.entries(), "")?;
    for (rel_path, entry) in additions {
        let dest =
            crate::sandbox::workspace_paths::resolve_workspace_path(processed.root(), &rel_path)
                .map_err(crate::error::AgentsError::from)?;
        if entry.is_dir() {
            session.mkdir(&dest, true).await?;
        } else if let Some((parent, _)) = dest.rsplit_once('/')
            && !parent.is_empty()
        {
            session.mkdir(parent, true).await?;
        }
        // The processed manifest owns the limits, as `Manifest::apply` does for a fresh session.
        let entry =
            entry.with_copy_concurrency(processed.concurrency_limits().local_dir_file_width());
        entry.apply(session, &dest).await?;
    }
    Ok(())
}

/// A structural signature of every entry, used where Python compares models for equality.
fn entry_signature_map(manifest: &Manifest) -> std::collections::BTreeMap<String, String> {
    manifest
        .entries()
        .iter()
        .map(|(name, entry)| (name.clone(), format!("{entry:?}")))
        .collect()
}

fn manifest_topology_matches(current: &Manifest, processed: &Manifest) -> bool {
    host_path_grant_topology(current) == host_path_grant_topology(processed)
        && current.root() == processed.root()
        && current.environment() == processed.environment()
        && current.users() == processed.users()
        && current.groups() == processed.groups()
}

/// Refuse a manifest holding a mount whose credential authority was never acknowledged.
///
/// Mirrors Python's `validate_manifest_mount_credential_boundaries`.
fn validate_mount_credential_boundaries(manifest: &Manifest) -> Result<()> {
    for (mount, path) in manifest.mount_targets() {
        manifest.authorize_mount_activation(mount, &path)?;
    }
    Ok(())
}

/// The grants that describe host-backed targets, as the tuple Python compares.
///
/// A target is host-backed when any grant for it names a host path. Every grant for such a target
/// is kept, path-only ones included: a path-only grant added next to a host-backed one for the
/// same target changes what the target means.
fn host_path_grant_topology(manifest: &Manifest) -> Vec<(String, bool, Option<String>)> {
    let host_backed: std::collections::BTreeSet<&str> = manifest
        .extra_path_grants()
        .iter()
        .filter(|grant| grant.host_path.is_some())
        .map(|grant| grant.path.as_str())
        .collect();
    manifest
        .extra_path_grants()
        .iter()
        .filter(|grant| host_backed.contains(grant.path.as_str()))
        .map(|grant| (grant.path.clone(), grant.read_only, grant.host_path.clone()))
        .collect()
}

fn validate_live_session_host_path_grants(current: &Manifest, processed: &Manifest) -> Result<()> {
    if host_path_grant_topology(current) != host_path_grant_topology(processed) {
        return Err(AgentsError::user(
            "injected sandbox sessions do not support capability changes to host-backed \
             `manifest.extra_path_grants`; use a fresh session or a session_state resume flow",
        ));
    }
    Ok(())
}

fn validate_running_live_session_manifest_update(
    current: &Manifest,
    processed: &Manifest,
) -> Result<()> {
    if current.root() != processed.root() {
        return Err(AgentsError::user(
            "running injected sandbox sessions do not support capability changes to \
             `manifest.root`; use a fresh session or a session_state resume flow",
        ));
    }
    if current.environment() != processed.environment() {
        return Err(AgentsError::user(
            "running injected sandbox sessions do not support capability changes to \
             `manifest.environment`; use a fresh session or a session_state resume flow",
        ));
    }
    if current.users() != processed.users() || current.groups() != processed.groups() {
        // `useradd` on a session whose processes are already running changes the accounts under
        // them, so Python refuses it too. The common way to land here is a `run_as` agent whose
        // account the injected session's manifest never declared, which is worth naming.
        return Err(AgentsError::user(
            "running injected sandbox sessions do not support capability changes to \
             `manifest.users` or `manifest.groups`; declare every account the agents run as \
             (`SandboxAgent::with_run_as`) on the manifest the session was created with, or use \
             a fresh session or a session_state resume flow",
        ));
    }
    Ok(())
}

/// The entries a running session has to materialize to catch up with `processed`.
///
/// Mirrors Python's `_diff_live_session_entries`: an entry only present in the processed manifest
/// is added, an entry whose contents changed is re-applied, and anything removed, retyped or
/// touching a mount is refused.
fn diff_live_session_entries(
    current: &std::collections::BTreeMap<String, Entry>,
    processed: &std::collections::BTreeMap<String, Entry>,
    parent_rel: &str,
) -> Result<Vec<(String, Entry)>> {
    let join = |name: &str| {
        if parent_rel.is_empty() {
            name.to_string()
        } else {
            format!("{parent_rel}/{name}")
        }
    };
    let removed: Vec<String> = current
        .keys()
        .filter(|name| !processed.contains_key(*name))
        .map(|name| join(name))
        .collect();
    if !removed.is_empty() {
        return Err(AgentsError::user(format!(
            "running injected sandbox sessions do not support removing manifest entries: {}",
            removed.join(", ")
        )));
    }

    let mut to_apply: Vec<(String, Entry)> = Vec::new();
    for (name, processed_entry) in processed {
        let rel_path = join(name);
        let Some(current_entry) = current.get(name) else {
            if entry_contains_mount(processed_entry) {
                return Err(AgentsError::user(format!(
                    "running injected sandbox sessions do not support capability-added mount \
                     entries at {rel_path}; use a fresh session or a session_state resume flow"
                )));
            }
            to_apply.push((rel_path, processed_entry.clone()));
            continue;
        };
        if format!("{current_entry:?}") == format!("{processed_entry:?}") {
            continue;
        }
        if current_entry.type_name() != processed_entry.type_name()
            || current_entry.is_dir() != processed_entry.is_dir()
        {
            return Err(AgentsError::user(format!(
                "running injected sandbox sessions do not support replacing manifest entry types \
                 at {rel_path}; use a fresh session or a session_state resume flow"
            )));
        }
        if current_entry.as_mount().is_some() {
            return Err(AgentsError::user(format!(
                "running injected sandbox sessions do not support capability changes to mount \
                 entries at {rel_path}; use a fresh session or a session_state resume flow"
            )));
        }
        if let (
            EntryKind::Dir {
                children: current_children,
            },
            EntryKind::Dir {
                children: processed_children,
            },
        ) = (&current_entry.kind, &processed_entry.kind)
        {
            let nested =
                diff_live_session_entries(current_children, processed_children, &rel_path)?;
            to_apply.extend(nested);
            continue;
        }
        to_apply.push((rel_path, processed_entry.clone()));
    }
    Ok(to_apply)
}

fn entry_contains_mount(entry: &Entry) -> bool {
    if entry.as_mount().is_some() {
        return true;
    }
    match &entry.kind {
        EntryKind::Dir { children } => children.values().any(entry_contains_mount),
        _ => false,
    }
}

/// Where a deferred preparation leaves what it built.
struct DeferredPreparation<C> {
    /// Everything but the session guard, which has its own slot so a streamed run's caller can
    /// take it over.
    rest: Arc<Mutex<Option<PreparedRest<C>>>>,
    guard: Arc<Mutex<Option<SandboxSessionGuard>>>,
    /// Resolves once preparation has finished, failed, or the run ended without reaching it.
    ready: tokio::sync::oneshot::Receiver<()>,
}

/// Sends the readiness signal when dropped.
struct ReadySignal(Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>);

impl Drop for ReadySignal {
    fn drop(&mut self) {
        if let Some(sender) = self.0.lock().expect("poisoned").take() {
            let _ = sender.send(());
        }
    }
}

/// A prepared run without its session guard.
struct PreparedRest<C> {
    graph: PreparedSandboxGraph<C>,
    agent_guards: Vec<(String, SandboxSessionGuard)>,
    memory_manager: Option<Arc<SandboxMemoryGenerationManager>>,
    rollout_id: String,
}

/// Prepare the sandbox into `slots`, returning the agent the run should start from.
async fn prepare_into<C: Send + Sync + 'static>(
    agent: &SandboxAgent<C>,
    config: &SandboxRunConfig,
    tracing_disabled: bool,
    rollout_override: Option<String>,
    graph_cell: &std::sync::OnceLock<PreparedSandboxGraph<C>>,
    rest_slot: &Mutex<Option<PreparedRest<C>>>,
    guard_slot: &Mutex<Option<SandboxSessionGuard>>,
) -> Result<Arc<Agent<C>>> {
    let mut prepared = masked(tracing_disabled, SandboxRunner::prepare(agent, config)).await?;
    prepared.guard.tracing_disabled = tracing_disabled;
    for (_, extra) in prepared.agent_guards.iter_mut() {
        extra.tracing_disabled = tracing_disabled;
    }
    if let Some(id) = rollout_override {
        prepared.rollout_id = id;
    }
    let starting = prepared.graph.starting_agent().clone();
    let _ = graph_cell.set(prepared.graph.clone());
    let PreparedSandboxRun {
        graph,
        guard,
        agent_guards,
        memory_manager,
        rollout_id,
    } = prepared;
    *guard_slot.lock().expect("poisoned") = Some(guard);
    *rest_slot.lock().expect("poisoned") = Some(PreparedRest {
        graph,
        agent_guards,
        memory_manager,
        rollout_id,
    });
    Ok(starting)
}

/// Prepare the sandbox after the first turn's blocking input guardrails have passed.
///
/// Python creates the session after those guardrails, so rejected input never costs a
/// session. The run starts from the agent's plain form; the hook swaps in the prepared graph's
/// starting agent, and the prepared run waits in the returned slots for teardown.
///
/// A resumed run is prepared up front instead: its saved state names the agent it stopped at,
/// and that agent can only be found in the prepared handoff graph. Its guardrails already ran
/// before the pause.
async fn begin_preparation<C: Send + Sync + 'static>(
    options: RunOptions<C>,
    agent: &SandboxAgent<C>,
    config: &SandboxRunConfig,
    tracing_disabled: bool,
    rollout_override: Option<String>,
    resuming: bool,
) -> Result<(RunOptions<C>, Arc<Agent<C>>, DeferredPreparation<C>)> {
    let slot: Arc<Mutex<Option<PreparedRest<C>>>> = Arc::new(Mutex::new(None));
    let guard_slot: Arc<Mutex<Option<SandboxSessionGuard>>> = Arc::new(Mutex::new(None));
    let graph = Arc::new(std::sync::OnceLock::new());
    let options = install_per_turn_preparation(options, graph.clone());
    let (ready_tx, ready) = tokio::sync::oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let deferred = DeferredPreparation {
        rest: slot.clone(),
        guard: guard_slot.clone(),
        ready,
    };
    if resuming {
        let _signal = ReadySignal(ready_tx);
        // Boxed so the rarely used eager path does not widen this function's own future.
        let starting = Box::pin(prepare_into(
            agent,
            config,
            tracing_disabled,
            rollout_override,
            &graph,
            &slot,
            &guard_slot,
        ))
        .await?;
        return Ok((options, starting, deferred));
    }
    let (hook_agent, hook_config) = (agent.clone(), config.clone());
    let options = options.prepare_agent(move |_| {
        let ready_tx = ready_tx.clone();
        let (slot, guard_slot, agent, config, graph, rollout) = (
            slot.clone(),
            guard_slot.clone(),
            hook_agent.clone(),
            hook_config.clone(),
            graph.clone(),
            rollout_override.clone(),
        );
        async move {
            // However preparation ends, whoever waits on it is released.
            let _signal = ReadySignal(ready_tx);
            // Preparation runs on its own task: it polls from the middle of the run's poll
            // chain, where stack headroom is smallest, and a task starts on a fresh stack. The
            // trace scope travels with it so its spans still nest under the run.
            let scope = crate::tracing::scope::ScopeSnapshot::capture();
            let task = tokio::spawn(scope.enter(Box::pin(async move {
                prepare_into(
                    &agent,
                    &config,
                    tracing_disabled,
                    rollout,
                    &graph,
                    &slot,
                    &guard_slot,
                )
                .await
            })));
            match task.await {
                Ok(result) => result,
                Err(error) => Err(AgentsError::user(format!(
                    "sandbox preparation did not finish: {error}"
                ))),
            }
        }
    });
    Ok((options, Arc::new(agent.agent.clone()), deferred))
}

/// The memory manager that follows the agent active when the run ended.
///
/// The session's manager follows the agent that is active when the run ends, as Python's
/// `_set_active_memory_capability` does on every preparation: a handoff target without
/// generation stops recording, and one that adds it starts.
fn active_memory_manager<C: Send + Sync + 'static>(
    prepared: &PreparedRest<C>,
    starting_name: &str,
    last_agent_name: &str,
) -> Option<Arc<SandboxMemoryGenerationManager>> {
    if last_agent_name == starting_name {
        return prepared.memory_manager.clone();
    }
    match prepared
        .graph
        .agents_by_name()
        .get(last_agent_name)
        .map(|active| active.memory_capability().map(|m| m.build_manager()))
    {
        Some(Some(Ok(manager))) => manager,
        Some(Some(Err(error))) => {
            log::warn!(target: crate::logger::LOG_TARGET, "Failed to build the active agent's sandbox memory manager: {error}");
            None
        }
        Some(None) => None,
        None => prepared.memory_manager.clone(),
    }
}

/// Install the capability `process_context` chain as a per-turn model input filter.
///
/// Python re-prepares a sandbox agent on every turn, which re-applies `process_context` to the
/// model input. `RunConfig::call_model_input_filter` is the equivalent hook here, and it also
/// covers handoff targets, because the filter looks the capability set up by agent name. Any
/// filter the caller already configured still runs first.
fn install_per_turn_preparation<C: Send + Sync + 'static>(
    mut options: RunOptions<C>,
    graph: Arc<std::sync::OnceLock<PreparedSandboxGraph<C>>>,
) -> RunOptions<C> {
    let existing = options.run_config.call_model_input_filter.take();
    options.run_config.call_model_input_filter = Some(Arc::new(move |data| {
        let graph = graph.clone();
        let existing = existing.clone();
        Box::pin(async move {
            let agent_name = data.agent.name.clone();
            let model_data = match existing {
                None => data.model_data,
                Some(filter) => {
                    filter(crate::run_config::CallModelData {
                        model_data: data.model_data,
                        agent: data.agent,
                        context: data.context,
                    })
                    .await?
                }
            };
            // The graph exists once the sandbox has been prepared, which happens before the
            // first model call.
            let input = match graph.get() {
                Some(graph) => graph.process_context(&agent_name, model_data.input),
                None => model_data.input,
            };
            Ok(ModelInputData {
                input,
                instructions: model_data.instructions,
            })
        })
    }));
    options
}

/// Confirm the configured working directory exists and is traversable for the sandbox user.
async fn validate_workspace_scope<C: Send + Sync + 'static>(
    session: &dyn SandboxSession,
    scope: &SandboxWorkspaceScope,
    agent: &SandboxAgent<C>,
) -> Result<()> {
    let Some(cwd) = scope.cwd() else {
        return Ok(());
    };
    let resolved = session.normalize_path(cwd)?;
    for flag in ["-d", "-x"] {
        let result = session
            .exec(ExecRequest::argv(["test", flag, &resolved]).with_user(agent.run_as.clone()))
            .await?;
        if !result.ok() {
            return Err(AgentsError::user(format!(
                "Sandbox working directory `{cwd}` does not exist or is not accessible for the \
                 configured sandbox user"
            )));
        }
    }
    Ok(())
}
