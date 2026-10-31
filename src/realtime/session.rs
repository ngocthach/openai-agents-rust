//! The realtime session: history, tool calls, handoffs and guardrails. Mirrors
//! `agents.realtime.session`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::Stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::guardrail::{OutputGuardrail, OutputGuardrailResult};
use crate::items::{FunctionCall, ToolApprovalItem};
use crate::realtime::agent::{RealtimeAgent, RealtimeHooks};
use crate::realtime::config::{
    RealtimeAudioStorage, RealtimeFunctionToolSpec, RealtimeModelTracing, RealtimeRunConfig,
    RealtimeSessionModelSettings, RealtimeToolErrorArgs, RealtimeToolErrorKind, RealtimeUserInput,
};
use crate::realtime::debug::{log_tool_action_error, log_tool_action_warning};
use crate::realtime::events::{RealtimeEventInfo, RealtimeSessionEvent};
use crate::realtime::handoffs::RealtimeHandoff;
use crate::realtime::items::{
    RealtimeItem, RealtimeToolCallItem, apply_assistant_transcript, apply_input_transcription,
    merge_item, strip_audio,
};
use crate::realtime::model::{RealtimeModel, RealtimeModelConfig};
use crate::realtime::model_events::{RealtimeModelEvent, RealtimeModelToolCallEvent};
use crate::realtime::model_inputs::{RealtimeInterrupt, RealtimeModelSendEvent};
use crate::run_context::RunContext;
use crate::tool::{
    DEFAULT_APPROVAL_REJECTION_MESSAGE, FunctionTool, ToolOutput, ToolTimeoutBehavior,
};
use crate::tool_context::ToolContext;
use crate::tool_guardrails::{
    ToolGuardrailBehavior, ToolInputGuardrailData, ToolOutputGuardrailData,
};

/// What identifies one realtime tool invocation.
///
/// The realtime API correlates a call and its output by `call_id` alone, so a model that reuses a
/// call id for a different tool, role or arguments would silently collide with the first call. The
/// Python SDK fingerprints the same fields through `tool_invocation_identity`; comparing them
/// directly gives the same answers without hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InvocationIdentity {
    name: String,
    role: InvocationRole,
    arguments: String,
}

/// Whether a call id was dispatched to a function tool or to a handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationRole {
    Function,
    Handoff,
    /// Not resolved yet, or resolved to neither.
    Unknown,
}

impl InvocationIdentity {
    /// Whether two dispatches of one call id are the same invocation.
    ///
    /// A role that could not be resolved matches anything, so a call that arrives before its tool
    /// is enabled is not mistaken for a different invocation.
    fn matches(&self, other: &InvocationIdentity) -> bool {
        self.name == other.name
            && self.arguments == other.arguments
            && (self.role == other.role
                || self.role == InvocationRole::Unknown
                || other.role == InvocationRole::Unknown)
    }
}

/// What [`SessionInner::begin_invocation`] decided about a tool call.
enum InvocationAdmission {
    /// Dispatch it.
    Proceed,
    /// Drop it: it is a duplicate of a call already handled or awaiting approval.
    Ignore,
    /// Resend the output cached from an earlier attempt instead of running the tool again.
    ResendCachedOutput,
}

/// A tool output the session produced but could not deliver.
///
/// The call already ran, so a retry must resend this output instead of running it again.
struct PendingToolOutput<C> {
    output: ToolOutput,
    agent: Arc<RealtimeAgent<C>>,
    /// The tool to report in `tool_end`; `None` for a handoff or an unknown tool.
    tool: Option<Arc<FunctionTool<C>>>,
    start_response: bool,
    /// Settings that must reach the model before the output starts a new response.
    session_update: Option<Box<RealtimeSessionModelSettings>>,
    /// Emit an `error` event with this message once the output lands.
    error_message: Option<String>,
}

/// What a function tool's input guardrails decided.
enum ToolInputVerdict {
    /// Run the tool.
    Allow,
    /// Skip the tool and send this message to the model instead.
    Reject(String),
    /// The guardrail tripped its wire; the session ends with this error.
    Raise(Box<AgentsError>),
}

/// A tool call waiting for a human decision.
struct PendingToolCall<C> {
    call: RealtimeModelToolCallEvent,
    agent: Arc<RealtimeAgent<C>>,
    tool: Arc<FunctionTool<C>>,
    approval_item: ToolApprovalItem,
    identity: InvocationIdentity,
}

/// A tool call admitted by the duplicate-call checks, and what is left to do for it.
enum AdmittedCall<C> {
    /// Route the call to a tool, a handoff or a not-found answer.
    Dispatch {
        agent: Arc<RealtimeAgent<C>>,
        tools: Vec<Arc<FunctionTool<C>>>,
        handoffs: Vec<RealtimeHandoff<C>>,
    },
    /// The call already ran; resend the output the transport never accepted.
    ResendCachedOutput,
}

/// Mutable session state. Guards are never held across an `await`.
struct SessionState<C> {
    agent: Arc<RealtimeAgent<C>>,
    tools: Vec<Arc<FunctionTool<C>>>,
    handoffs: Vec<RealtimeHandoff<C>>,
    history: Vec<RealtimeItem>,
    pending_tool_calls: HashMap<String, PendingToolCall<C>>,
    /// Call ids currently dispatching, with what they were dispatched as.
    active_tool_invocations: HashMap<String, InvocationIdentity>,
    /// What every call id this session admitted was dispatched as, kept for the session's life so
    /// a later reuse of the id is recognized even after the call was answered.
    tool_invocation_routes: HashMap<String, InvocationIdentity>,
    /// Call ids whose tool already ran, whether or not the output was delivered.
    executed_tool_calls: HashMap<String, InvocationIdentity>,
    /// Call ids whose output reached the model.
    committed_tool_calls: HashSet<String>,
    /// Outputs that were produced but could not be sent, keyed by call id.
    pending_tool_outputs: HashMap<String, PendingToolOutput<C>>,
    item_transcripts: HashMap<String, String>,
    item_guardrail_runs: HashMap<String, usize>,
    interrupted_response_ids: HashSet<String>,
    active_response_id: Option<String>,
    active_response_agent: Option<Arc<RealtimeAgent<C>>>,
    /// Generation of the response currently producing output, if any.
    active_output_response_generation: Option<u64>,
    /// Guardrail runs still in flight, per response.
    guardrail_runs_by_response: HashMap<String, usize>,
    /// Responses whose transport audio may be released once their guardrails finish.
    responses_awaiting_guardrail_cleanup: HashSet<String>,
}

struct SessionInner<C> {
    model: Arc<dyn RealtimeModel>,
    context: RunContext<C>,
    run_config: RealtimeRunConfig<C>,
    base_settings: RealtimeSessionModelSettings,
    debounce_text_length: usize,
    state: Mutex<SessionState<C>>,
    sender: Mutex<Option<mpsc::UnboundedSender<RealtimeSessionEvent<C>>>>,
    closed: AtomicBool,
    /// Whether the transport finished closing; a failed close can be retried.
    model_closed: AtomicBool,
    /// Serializes `close` so concurrent callers do not close the transport twice.
    close_lock: tokio::sync::Mutex<()>,
    pump: Mutex<Option<JoinHandle<()>>>,
    guardrail_tasks: Mutex<Vec<JoinHandle<()>>>,
    /// Tool calls running on their own task, with `async_tool_calls` on.
    tool_tasks: Mutex<Vec<JoinHandle<()>>>,
    /// The failure that ends the session, kept until the caller takes it.
    stored_error: Mutex<Option<AgentsError>>,
    /// Monotonic counter of output responses, readable from a transport send-boundary check.
    latest_generation: Arc<AtomicU64>,
}

impl<C> SessionInner<C> {
    /// Start a new output-response generation and return it.
    fn next_generation(&self) -> u64 {
        self.latest_generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn latest_generation(&self) -> u64 {
        self.latest_generation.load(Ordering::SeqCst)
    }
}

/// A live connection to a realtime model.
///
/// The session owns the local history, executes tools, runs output guardrails and performs
/// handoffs. Clones share the same connection, history and event stream, so one clone can read
/// events while another sends messages.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openai_agents_sdk::realtime::{RealtimeAgent, RealtimeRunner};
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
/// let session = RealtimeRunner::new(agent).run(()).await?;
/// session.send_message("Hello").await?;
/// while let Some(event) = session.recv().await {
///     println!("{}", event.event_type());
/// }
/// session.close().await?;
/// # Ok(())
/// # }
/// ```
pub struct RealtimeSession<C> {
    inner: Arc<SessionInner<C>>,
    events: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<RealtimeSessionEvent<C>>>>,
}

impl<C> Clone for RealtimeSession<C> {
    fn clone(&self) -> Self {
        RealtimeSession {
            inner: self.inner.clone(),
            events: self.events.clone(),
        }
    }
}

impl<C> fmt::Debug for RealtimeSession<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeSession")
            .field("agent", &self.current_agent_name())
            .field("closed", &self.inner.closed.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> RealtimeSession<C> {
    /// Connect to `model` and start a session for `agent`.
    ///
    /// Prefer [`crate::realtime::RealtimeRunner`], which builds the model for you.
    pub async fn connect(
        model: Arc<dyn RealtimeModel>,
        agent: Arc<RealtimeAgent<C>>,
        context: C,
        model_config: Option<RealtimeModelConfig>,
        run_config: Option<RealtimeRunConfig<C>>,
    ) -> Result<Self> {
        Self::connect_with_context(
            model,
            agent,
            RunContext::new(context),
            model_config,
            run_config,
        )
        .await
    }

    /// Connect using an existing [`RunContext`], so usage and approvals are shared with a wider
    /// application.
    pub async fn connect_with_context(
        model: Arc<dyn RealtimeModel>,
        agent: Arc<RealtimeAgent<C>>,
        context: RunContext<C>,
        model_config: Option<RealtimeModelConfig>,
        run_config: Option<RealtimeRunConfig<C>>,
    ) -> Result<Self> {
        let mut model_config = model_config.unwrap_or_default();
        let run_config = run_config.unwrap_or_default();

        let mut base_settings = run_config.model_settings.clone().unwrap_or_default();
        if let Some(initial) = &model_config.initial_model_settings {
            base_settings.merge(initial);
        }

        let inner = Arc::new(SessionInner {
            model: model.clone(),
            context,
            debounce_text_length: run_config.guardrails_settings.debounce_text_length.max(1),
            run_config,
            base_settings,
            state: Mutex::new(SessionState {
                agent: agent.clone(),
                tools: Vec::new(),
                handoffs: Vec::new(),
                history: Vec::new(),
                pending_tool_calls: HashMap::new(),
                active_tool_invocations: HashMap::new(),
                tool_invocation_routes: HashMap::new(),
                executed_tool_calls: HashMap::new(),
                committed_tool_calls: HashSet::new(),
                pending_tool_outputs: HashMap::new(),
                item_transcripts: HashMap::new(),
                item_guardrail_runs: HashMap::new(),
                interrupted_response_ids: HashSet::new(),
                active_response_id: None,
                active_response_agent: None,
                active_output_response_generation: None,
                guardrail_runs_by_response: HashMap::new(),
                responses_awaiting_guardrail_cleanup: HashSet::new(),
            }),
            sender: Mutex::new(None),
            closed: AtomicBool::new(false),
            model_closed: AtomicBool::new(false),
            close_lock: tokio::sync::Mutex::new(()),
            pump: Mutex::new(None),
            guardrail_tasks: Mutex::new(Vec::new()),
            tool_tasks: Mutex::new(Vec::new()),
            stored_error: Mutex::new(None),
            latest_generation: Arc::new(AtomicU64::new(0)),
        });

        let starting = model_config.initial_model_settings.clone();
        let resolved = inner.build_settings(&agent, starting.as_ref()).await?;
        {
            let mut state = inner.state.lock().expect("realtime session poisoned");
            state.tools = resolved.tools.clone();
            state.handoffs = resolved.handoffs.clone();
        }
        model_config.initial_model_settings = Some(resolved.settings);

        let model_events = model.connect(model_config).await?;

        let (tx, rx) = mpsc::unbounded_channel();
        *inner.sender.lock().expect("realtime session poisoned") = Some(tx);

        let pump_inner = inner.clone();
        let pump = tokio::spawn(async move { pump_inner.pump(model_events).await });
        *inner.pump.lock().expect("realtime session poisoned") = Some(pump);

        let session = RealtimeSession {
            inner,
            events: Arc::new(tokio::sync::Mutex::new(rx)),
        };
        let history = session.history();
        session.inner.emit(RealtimeSessionEvent::HistoryUpdated {
            history,
            info: session.inner.info(),
        });
        Ok(session)
    }

    /// Send a text or structured message and ask the model to respond.
    pub async fn send_message(&self, message: impl Into<RealtimeUserInput>) -> Result<()> {
        self.inner
            .model
            .send_event(RealtimeModelSendEvent::UserInput {
                user_input: message.into(),
            })
            .await
    }

    /// Append raw audio to the model's input buffer.
    ///
    /// With `commit` set, the buffer is committed, which is what sessions without server-side
    /// turn detection need in order to end the user's turn.
    pub async fn send_audio(&self, audio: impl Into<Vec<u8>>, commit: bool) -> Result<()> {
        self.inner
            .model
            .send_event(RealtimeModelSendEvent::Audio {
                audio: audio.into(),
                commit,
            })
            .await
    }

    /// Interrupt generation and audio playback without cancelling running tools.
    pub async fn interrupt(&self) -> Result<()> {
        self.inner
            .model
            .send_event(RealtimeModelSendEvent::interrupt())
            .await
    }

    /// Make `agent` the active agent and apply its settings to the model.
    pub async fn update_agent(&self, agent: Arc<RealtimeAgent<C>>) -> Result<()> {
        let resolved = self.inner.build_settings(&agent, None).await?;
        {
            let mut state = self.inner.state.lock().expect("realtime session poisoned");
            state.agent = agent;
            state.tools = resolved.tools.clone();
            state.handoffs = resolved.handoffs.clone();
        }
        self.inner
            .model
            .send_event(RealtimeModelSendEvent::session_update(resolved.settings))
            .await
    }

    /// Approve a tool call that raised
    /// [`RealtimeSessionEvent::ToolApprovalRequired`]
    /// and run it.
    ///
    /// With `always`, later calls to the same tool by the same agent are approved automatically.
    /// The call runs before this future resolves, so `tool_start` and `tool_end` are already
    /// queued when it returns; this is true whatever
    /// [`RealtimeRunConfig::async_tool_calls`](crate::realtime::RealtimeRunConfig) says, because
    /// that setting governs the transport event loop and this `Result` is the caller's only
    /// report of the failure.
    pub async fn approve_tool_call(&self, call_id: &str, always: bool) -> Result<()> {
        let Some(pending) = self.inner.take_pending_and_reserve(call_id) else {
            return Ok(());
        };
        self.inner
            .context
            .approve_tool(&pending.approval_item, always);
        let result = self
            .inner
            .execute_tool(pending.agent, pending.tool, pending.call)
            .await;
        self.inner.release_invocation(call_id);
        result
    }

    /// Reject a pending tool call and tell the model it was not approved.
    ///
    /// With `always`, later calls to the same tool by the same agent are rejected automatically.
    pub async fn reject_tool_call(
        &self,
        call_id: &str,
        always: bool,
        rejection_message: Option<String>,
    ) -> Result<()> {
        let Some(pending) = self.inner.take_pending_and_reserve(call_id) else {
            return Ok(());
        };
        self.inner
            .context
            .reject_tool(&pending.approval_item, always, rejection_message);
        let result = self
            .inner
            .send_tool_rejection(&pending.agent, &pending.tool, &pending.call)
            .await;
        self.inner.release_invocation(call_id);
        result
    }

    /// Wait for the next session event, or `None` once the session is closed and drained.
    ///
    /// A transport failure or a failed tool task ends the session: the matching
    /// [`RealtimeSessionEvent::Error`] is delivered, then this closes the session and returns
    /// `None`. The failure itself is available from [`RealtimeSession::take_error`]. This is
    /// where the Python SDK raises its stored exception out of `__aiter__`.
    pub async fn recv(&self) -> Option<RealtimeSessionEvent<C>> {
        let event = {
            let mut events = self.events.lock().await;
            events.recv().await
        };
        if event.is_none() && self.has_stored_error() {
            let _ = self.close().await;
        }
        event
    }

    fn has_stored_error(&self) -> bool {
        self.inner
            .stored_error
            .lock()
            .expect("realtime session poisoned")
            .is_some()
    }

    /// Take the failure that ended the session, if there was one.
    ///
    /// Set by a transport exception or by a tool call that failed on its own task; the Rust
    /// equivalent of the exception Python's `__aiter__` raises.
    pub fn take_error(&self) -> Option<AgentsError> {
        self.inner
            .stored_error
            .lock()
            .expect("realtime session poisoned")
            .take()
    }

    /// The session events as an async stream.
    pub fn event_stream(&self) -> impl Stream<Item = RealtimeSessionEvent<C>> + Send + 'static {
        futures::stream::unfold(self.clone(), |session| async move {
            session.recv().await.map(|event| (event, session))
        })
    }

    /// Close the session and its transport.
    pub async fn close(&self) -> Result<()> {
        let _closing = self.inner.close_lock.lock().await;
        if self.inner.closed.swap(true, Ordering::SeqCst) {
            // A repeat call only has work left when the transport failed to close before.
            if self.inner.model_closed.load(Ordering::SeqCst) {
                return Ok(());
            }
            let result = self.inner.model.close().await;
            if result.is_ok() {
                self.inner.model_closed.store(true, Ordering::SeqCst);
            }
            return result;
        }
        // A task that closes the session from inside must not abort itself mid-cleanup.
        let current = tokio::task::try_id();
        let abort = |task: &JoinHandle<()>| {
            if Some(task.id()) != current {
                task.abort();
            }
        };
        if let Some(pump) = self
            .inner
            .pump
            .lock()
            .expect("realtime session poisoned")
            .take()
        {
            abort(&pump);
        }
        for task in self
            .inner
            .guardrail_tasks
            .lock()
            .expect("realtime session poisoned")
            .drain(..)
        {
            abort(&task);
        }
        for task in self
            .inner
            .tool_tasks
            .lock()
            .expect("realtime session poisoned")
            .drain(..)
        {
            abort(&task);
        }
        *self.inner.sender.lock().expect("realtime session poisoned") = None;
        {
            let mut state = self.inner.state.lock().expect("realtime session poisoned");
            state.pending_tool_calls.clear();
            state.active_tool_invocations.clear();
            state.tool_invocation_routes.clear();
            state.executed_tool_calls.clear();
            state.committed_tool_calls.clear();
            state.pending_tool_outputs.clear();
            state.item_transcripts.clear();
            state.item_guardrail_runs.clear();
            state.interrupted_response_ids.clear();
            state.guardrail_runs_by_response.clear();
            state.responses_awaiting_guardrail_cleanup.clear();
            state.active_output_response_generation = None;
            state.active_response_id = None;
            state.active_response_agent = None;
        }
        let result = self.inner.model.close().await;
        if result.is_ok() {
            self.inner.model_closed.store(true, Ordering::SeqCst);
        }
        result
    }
}

impl<C> RealtimeSession<C> {
    /// A snapshot of the session history.
    pub fn history(&self) -> Vec<RealtimeItem> {
        self.inner
            .state
            .lock()
            .expect("realtime session poisoned")
            .history
            .clone()
    }

    /// The active agent.
    pub fn current_agent(&self) -> Arc<RealtimeAgent<C>> {
        self.inner
            .state
            .lock()
            .expect("realtime session poisoned")
            .agent
            .clone()
    }

    /// The name of the active agent.
    pub fn current_agent_name(&self) -> String {
        self.current_agent().name.clone()
    }

    /// The run context shared with tools, handoffs and guardrails.
    pub fn context(&self) -> &RunContext<C> {
        &self.inner.context
    }

    /// The transport backing this session.
    pub fn model(&self) -> &Arc<dyn RealtimeModel> {
        &self.inner.model
    }

    /// Whether [`RealtimeSession::close`] has been called.
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }
}

/// Settings plus the tools and handoffs they were built from.
pub(crate) struct ResolvedSettings<C> {
    pub(crate) settings: RealtimeSessionModelSettings,
    pub(crate) tools: Vec<Arc<FunctionTool<C>>>,
    pub(crate) handoffs: Vec<RealtimeHandoff<C>>,
}

/// Resolve `agent`'s instructions, tools and handoffs into realtime session settings.
///
/// `base` is the run config's settings merged with the connect-time settings; `starting` is
/// applied last so connect-time overrides win over the agent, as in Python. Port of
/// `_build_model_settings_from_agent`.
pub(crate) async fn build_model_settings_from_agent<C: Send + Sync + 'static>(
    agent: &Arc<RealtimeAgent<C>>,
    context: &RunContext<C>,
    base: &RealtimeSessionModelSettings,
    starting: Option<&RealtimeSessionModelSettings>,
    tracing_disabled: bool,
) -> Result<ResolvedSettings<C>> {
    let mut settings = base.clone();
    if let Some(agent_settings) = &agent.model_settings {
        settings.merge(agent_settings);
    }

    settings.instructions = Some(agent.get_system_prompt(context).await?.unwrap_or_default());
    if let Some(prompt) = &agent.prompt {
        settings.prompt = Some(prompt.clone());
    }

    // Includes the agent's MCP tools, as Python's `agent.get_all_tools(ctx)` does.
    let tools = agent.get_all_tools(context).await?;
    let mut handoffs = Vec::new();
    for handoff in &agent.handoffs {
        if handoff.is_enabled.evaluate(context, &agent.name).await {
            handoffs.push(handoff.clone());
        }
    }

    let mut specs: Vec<RealtimeFunctionToolSpec> = tools
        .iter()
        .map(|tool| {
            let mut spec = RealtimeFunctionToolSpec::new(
                tool.name.clone(),
                tool.description.clone(),
                tool.params_json_schema.clone(),
            )
            .with_defer_loading(tool.defer_loading);
            if let Some(namespace) = tool.namespace() {
                spec = spec.with_namespace(namespace);
            }
            spec
        })
        .collect();
    specs.extend(handoffs.iter().map(RealtimeHandoff::to_spec));
    validate_tool_names(&specs)?;
    settings.tools = Some(specs);

    if let Some(starting) = starting {
        settings.merge(starting);
    }
    if tracing_disabled {
        settings.tracing = Some(RealtimeModelTracing::Disabled);
    }

    Ok(ResolvedSettings {
        settings,
        tools,
        handoffs,
    })
}

impl<C: Send + Sync + 'static> SessionInner<C> {
    /// The hooks to notify: the run's, then the active agent's.
    fn hooks_for(&self, agent: &Arc<RealtimeAgent<C>>) -> Vec<Arc<dyn RealtimeHooks<C>>> {
        let mut hooks = Vec::new();
        if let Some(run_hooks) = &self.run_config.hooks {
            hooks.push(run_hooks.clone());
        }
        if let Some(agent_hooks) = &agent.hooks {
            hooks.push(agent_hooks.clone());
        }
        hooks
    }

    fn info(&self) -> RealtimeEventInfo<C> {
        RealtimeEventInfo {
            context: self.context.clone(),
        }
    }

    fn emit(&self, event: RealtimeSessionEvent<C>) -> bool {
        if self.closed.load(Ordering::SeqCst) {
            return false;
        }
        let sender = {
            self.sender
                .lock()
                .expect("realtime session poisoned")
                .clone()
        };
        match sender {
            Some(tx) => tx.send(event).is_ok(),
            None => false,
        }
    }

    fn emit_error(&self, message: impl Into<String>) {
        self.emit(RealtimeSessionEvent::Error {
            error: json!({ "message": message.into() }),
            info: self.info(),
        });
    }

    /// Report a failure and end the event stream, as Python's `_stored_exception` does.
    ///
    /// The error event is queued first, then the sender is dropped, so a caller still receives
    /// every event already queued before `recv` returns `None`.
    fn end_with_error(&self, error: AgentsError) {
        let message = error.to_string();
        self.end_with_error_message(error, message);
    }

    /// A tool task failed: keep the error for the caller, but publish only Python's fixed
    /// message so tool arguments and results never leak through the event stream.
    fn end_with_tool_task_error(&self, error: AgentsError) {
        crate::realtime::debug::log_tool_action_error("Realtime tool call task failed", &error);
        self.end_with_error_message(error, "Tool call task failed".to_string());
    }

    fn end_with_error_message(&self, error: AgentsError, message: String) {
        {
            let mut stored = self.stored_error.lock().expect("realtime session poisoned");
            if stored.is_some() {
                return;
            }
            *stored = Some(error);
        }
        self.emit_error(message);
        *self.sender.lock().expect("realtime session poisoned") = None;
    }

    fn emit_history_updated(&self) {
        let history = {
            self.state
                .lock()
                .expect("realtime session poisoned")
                .history
                .clone()
        };
        self.emit(RealtimeSessionEvent::HistoryUpdated {
            history,
            info: self.info(),
        });
    }

    /// Take a call awaiting approval and reserve its id in one step.
    ///
    /// Reserving under the same lock keeps a redelivered `function_call` from racing the decision:
    /// without it the redelivery would either run the tool a second time or draw a spurious
    /// "already executed" error.
    fn take_pending_and_reserve(&self, call_id: &str) -> Option<PendingToolCall<C>> {
        let mut state = self.state.lock().expect("realtime session poisoned");
        let pending = state.pending_tool_calls.remove(call_id)?;
        state
            .active_tool_invocations
            .insert(call_id.to_string(), pending.identity.clone());
        Some(pending)
    }

    /// Release a reserved call id, whatever the outcome was.
    fn release_invocation(&self, call_id: &str) {
        self.state
            .lock()
            .expect("realtime session poisoned")
            .active_tool_invocations
            .remove(call_id);
    }

    /// Build the settings sent to the model for `agent`.
    async fn build_settings(
        &self,
        agent: &Arc<RealtimeAgent<C>>,
        starting: Option<&RealtimeSessionModelSettings>,
    ) -> Result<ResolvedSettings<C>> {
        build_model_settings_from_agent(
            agent,
            &self.context,
            &self.base_settings,
            starting,
            self.run_config.tracing_disabled,
        )
        .await
    }

    /// Read transport events until the connection ends.
    async fn pump(self: Arc<Self>, mut events: mpsc::UnboundedReceiver<RealtimeModelEvent>) {
        while let Some(event) = events.recv().await {
            if self.closed.load(Ordering::SeqCst) {
                break;
            }
            Box::pin(Self::handle_model_event(&self, event)).await;
        }
        *self.sender.lock().expect("realtime session poisoned") = None;
    }

    async fn handle_model_event(self: &Arc<Self>, event: RealtimeModelEvent) {
        if !self.emit(RealtimeSessionEvent::RawModelEvent {
            data: event.clone(),
            info: self.info(),
        }) {
            return;
        }

        match event {
            RealtimeModelEvent::Error { error } => {
                self.emit(RealtimeSessionEvent::Error {
                    error,
                    info: self.info(),
                });
            }
            RealtimeModelEvent::Exception { message, context } => {
                let message = match context {
                    Some(context) => format!("{context}: {message}"),
                    None => message,
                };
                self.end_with_error(AgentsError::other(message));
            }
            RealtimeModelEvent::FunctionCall(call) => {
                if let Err(error) = self.handle_tool_call(call).await {
                    self.emit_error(error.to_string());
                }
            }
            RealtimeModelEvent::Audio(audio) => {
                let interrupted = {
                    self.state
                        .lock()
                        .expect("realtime session poisoned")
                        .interrupted_response_ids
                        .contains(&audio.response_id)
                };
                if !interrupted {
                    let item_id = audio.item_id.clone();
                    let content_index = audio.content_index;
                    self.emit(RealtimeSessionEvent::Audio {
                        audio,
                        item_id,
                        content_index,
                        info: self.info(),
                    });
                }
            }
            RealtimeModelEvent::AudioInterrupted {
                item_id,
                content_index,
            } => {
                self.emit(RealtimeSessionEvent::AudioInterrupted {
                    item_id,
                    content_index,
                    info: self.info(),
                });
            }
            RealtimeModelEvent::AudioDone {
                item_id,
                content_index,
            } => {
                self.emit(RealtimeSessionEvent::AudioEnd {
                    item_id,
                    content_index,
                    info: self.info(),
                });
            }
            RealtimeModelEvent::InputAudioTranscriptionCompleted {
                item_id,
                transcript,
            } => {
                let added = {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    apply_input_transcription(&mut state.history, &item_id, &transcript)
                };
                if added {
                    let item = {
                        self.state
                            .lock()
                            .expect("realtime session poisoned")
                            .history
                            .last()
                            .cloned()
                    };
                    if let Some(item) = item {
                        self.emit(RealtimeSessionEvent::HistoryAdded {
                            item: Box::new(item),
                            info: self.info(),
                        });
                    }
                } else {
                    self.emit_history_updated();
                }
            }
            RealtimeModelEvent::InputAudioTimeoutTriggered { .. } => {
                self.emit(RealtimeSessionEvent::InputAudioTimeoutTriggered { info: self.info() });
            }
            RealtimeModelEvent::TranscriptDelta {
                item_id,
                delta,
                response_id,
            } => {
                self.reconcile_output_response(&response_id);
                let transcript = {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    state
                        .item_transcripts
                        .entry(item_id.clone())
                        .or_default()
                        .push_str(&delta);
                    state.item_transcripts[&item_id].clone()
                };
                {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    apply_assistant_transcript(&mut state.history, &item_id, &transcript);
                }
                self.spawn_output_guardrails(&item_id, &response_id, true);
            }
            RealtimeModelEvent::OutputTextDelta {
                item_id,
                delta,
                response_id,
            } => {
                self.reconcile_output_response(&response_id);
                {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    state
                        .item_transcripts
                        .entry(item_id.clone())
                        .or_default()
                        .push_str(&delta);
                }
                self.spawn_output_guardrails(&item_id, &response_id, false);
            }
            RealtimeModelEvent::ItemUpdated { item } => {
                let mut item = *item;
                if self.run_config.history_audio_storage == RealtimeAudioStorage::DiscardAudio {
                    strip_audio(&mut item);
                }
                let (is_new, stored) = {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    let is_new = !state
                        .history
                        .iter()
                        .any(|existing| existing.item_id() == item.item_id());
                    let item_id = item.item_id().to_string();
                    merge_item(&mut state.history, item);
                    let stored = state
                        .history
                        .iter()
                        .find(|existing| existing.item_id() == item_id)
                        .cloned();
                    (is_new, stored)
                };
                match (is_new, stored) {
                    (true, Some(item)) => {
                        self.emit(RealtimeSessionEvent::HistoryAdded {
                            item: Box::new(item),
                            info: self.info(),
                        });
                    }
                    _ => self.emit_history_updated(),
                }
            }
            RealtimeModelEvent::ItemDeleted { item_id } => {
                {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    state.history.retain(|item| item.item_id() != item_id);
                }
                self.emit_history_updated();
            }
            RealtimeModelEvent::TurnStarted { response_id } => {
                let agent = {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    // A `turn_started` for the response whose deltas already opened a generation
                    // is a late start, not a new turn.
                    let late_start = response_id.is_some()
                        && response_id == state.active_response_id
                        && state.active_output_response_generation.is_some();
                    if !late_start {
                        state.active_output_response_generation = Some(self.next_generation());
                        state.active_response_id = response_id;
                        state.active_response_agent = Some(state.agent.clone());
                    }
                    state.agent.clone()
                };
                for hooks in self.hooks_for(&agent) {
                    hooks.on_agent_start(&self.context, &agent).await;
                }
                self.emit(RealtimeSessionEvent::AgentStart {
                    agent,
                    info: self.info(),
                });
            }
            RealtimeModelEvent::Usage { usage, .. } => {
                self.context.add_usage(&usage);
            }
            RealtimeModelEvent::TurnEnded { response_id } => {
                let (agent, ended_response_id) = {
                    let mut state = self.state.lock().expect("realtime session poisoned");
                    let ended_response_id = response_id
                        .clone()
                        .or_else(|| state.active_response_id.clone());
                    let is_active =
                        response_id.is_none() || response_id == state.active_response_id;
                    if !is_active {
                        (None, ended_response_id)
                    } else {
                        state.item_transcripts.clear();
                        state.item_guardrail_runs.clear();
                        // A handoff inside the response already moved `agent`, so the turn is
                        // attributed to the agent captured when it started.
                        let agent = state
                            .active_response_agent
                            .take()
                            .unwrap_or_else(|| state.agent.clone());
                        state.active_response_id = None;
                        state.active_output_response_generation = None;
                        (Some(agent), ended_response_id)
                    }
                };
                if let Some(response_id) = ended_response_id {
                    self.finish_response_guardrail_lifecycle(&response_id).await;
                }
                if let Some(agent) = agent {
                    for hooks in self.hooks_for(&agent) {
                        hooks.on_agent_end(&self.context, &agent).await;
                    }
                    self.emit(RealtimeSessionEvent::AgentEnd {
                        agent,
                        info: self.info(),
                    });
                }
            }
            RealtimeModelEvent::EndOfStream => {
                // The raw event is already queued; dropping the sender lets a caller drain the
                // queue and then see the stream end, as Python's end-of-stream sentinel does.
                *self.sender.lock().expect("realtime session poisoned") = None;
            }
            RealtimeModelEvent::ConnectionStatus { .. }
            | RealtimeModelEvent::Other { .. }
            | RealtimeModelEvent::RawServerEvent { .. } => {}
        }
    }

    /// Reserve a model tool call against the duplicate-call checks.
    ///
    /// The reservation is synchronous so that two calls arriving back to back are admitted in
    /// arrival order even when the work itself runs on its own task. `None` means the call needs
    /// no further work.
    fn admit_tool_call(
        &self,
        call: &RealtimeModelToolCallEvent,
    ) -> Result<Option<AdmittedCall<C>>> {
        if call.call_id.is_empty() {
            return Err(AgentsError::model_behavior(
                "Realtime tool calls require a non-empty string call ID.",
            ));
        }
        let (agent, tools, handoffs) = {
            let state = self.state.lock().expect("realtime session poisoned");
            (
                state.agent.clone(),
                state.tools.clone(),
                state.handoffs.clone(),
            )
        };

        let role = if tools.iter().any(|tool| tool.name == call.name) {
            InvocationRole::Function
        } else if handoffs.iter().any(|h| h.tool_name == call.name) {
            InvocationRole::Handoff
        } else {
            InvocationRole::Unknown
        };
        let identity = InvocationIdentity {
            name: call.name.clone(),
            role,
            arguments: call.arguments.clone(),
        };
        match self.begin_invocation(&call.call_id, &identity)? {
            InvocationAdmission::Proceed => Ok(Some(AdmittedCall::Dispatch {
                agent,
                tools,
                handoffs,
            })),
            InvocationAdmission::Ignore => Ok(None),
            InvocationAdmission::ResendCachedOutput => Ok(Some(AdmittedCall::ResendCachedOutput)),
        }
    }

    /// Run a call that [`SessionInner::admit_tool_call`] admitted.
    async fn run_admitted_tool_call(
        &self,
        call: RealtimeModelToolCallEvent,
        admitted: AdmittedCall<C>,
    ) -> Result<()> {
        let AdmittedCall::Dispatch {
            agent,
            tools,
            handoffs,
        } = admitted
        else {
            return self.resend_pending_output(&call).await;
        };

        let result = self
            .dispatch_tool_call(call.clone(), agent, tools, handoffs)
            .await;
        self.state
            .lock()
            .expect("realtime session poisoned")
            .active_tool_invocations
            .remove(&call.call_id);
        result
    }

    /// Route a model tool call to a function tool, a handoff, or an error.
    ///
    /// With `RealtimeRunConfig::async_tool_calls` the work runs on its own task, so the session's
    /// event loop keeps delivering audio and transcript events while the tool runs; the
    /// duplicate-call reservation still happens inline, in arrival order.
    async fn handle_tool_call(self: &Arc<Self>, call: RealtimeModelToolCallEvent) -> Result<()> {
        let Some(admitted) = self.admit_tool_call(&call)? else {
            return Ok(());
        };

        if !self.run_config.async_tool_calls {
            if let Err(error) = self.run_admitted_tool_call(call, admitted).await {
                self.end_with_tool_task_error(error);
            }
            return Ok(());
        }

        let inner = self.clone();
        let task = tokio::spawn(async move {
            if let Err(error) = inner.run_admitted_tool_call(call, admitted).await {
                inner.end_with_tool_task_error(error);
            }
        });
        let mut tasks = self.tool_tasks.lock().expect("realtime session poisoned");
        tasks.retain(|task| !task.is_finished());
        tasks.push(task);
        Ok(())
    }

    /// Decide whether a tool call may run, mirroring the realtime duplicate-call checks.
    fn begin_invocation(
        &self,
        call_id: &str,
        identity: &InvocationIdentity,
    ) -> Result<InvocationAdmission> {
        let mut state = self.state.lock().expect("realtime session poisoned");

        // A call id names one invocation; a different name, role or arguments under the same id
        // would silently collide with the first call's output. The route survives the call, so an
        // id answered without running a tool (rejected, guardrail-blocked or not found) is still
        // recognized when it is reused.
        if let Some(recorded) = state.tool_invocation_routes.get(call_id)
            && !recorded.matches(identity)
        {
            return Err(AgentsError::model_behavior(
                "Model reused a Realtime tool call ID for a different invocation. \
                 Use a unique call ID for each tool invocation.",
            ));
        }

        if state.pending_tool_outputs.contains_key(call_id) {
            // The call already ran but its output never landed; resend it rather than rerun it.
            return Ok(InvocationAdmission::ResendCachedOutput);
        }
        if state.active_tool_invocations.contains_key(call_id)
            || state.pending_tool_calls.contains_key(call_id)
            || state.committed_tool_calls.contains(call_id)
        {
            // A redelivered call, an approval still awaiting a decision, or one already answered.
            return Ok(InvocationAdmission::Ignore);
        }
        if state.executed_tool_calls.contains_key(call_id) {
            return Err(AgentsError::model_behavior(
                "A Realtime tool call already executed, but its output was not committed. \
                 Start a new call instead of retrying the invocation.",
            ));
        }

        state
            .active_tool_invocations
            .insert(call_id.to_string(), identity.clone());
        state
            .tool_invocation_routes
            .insert(call_id.to_string(), identity.clone());
        Ok(InvocationAdmission::Proceed)
    }

    /// Resend an output that was produced but never delivered.
    async fn resend_pending_output(&self, call: &RealtimeModelToolCallEvent) -> Result<()> {
        let pending = {
            self.state
                .lock()
                .expect("realtime session poisoned")
                .pending_tool_outputs
                .remove(&call.call_id)
        };
        let Some(pending) = pending else {
            return Ok(());
        };
        self.commit_tool_output(call, pending).await
    }

    async fn dispatch_tool_call(
        &self,
        call: RealtimeModelToolCallEvent,
        agent: Arc<RealtimeAgent<C>>,
        tools: Vec<Arc<FunctionTool<C>>>,
        handoffs: Vec<RealtimeHandoff<C>>,
    ) -> Result<()> {
        // Enablement can change after the tools were resolved, so a call is routed only to a
        // tool or handoff that is still enabled when it is dispatched.
        let mut enabled_tool = None;
        for tool in tools.iter().filter(|tool| tool.name == call.name) {
            if tool.is_enabled.evaluate(&self.context, &agent.name).await {
                enabled_tool = Some(tool.clone());
                break;
            }
        }
        let mut enabled_handoff = None;
        if enabled_tool.is_none() {
            for handoff in handoffs.iter().filter(|h| h.tool_name == call.name) {
                if handoff
                    .is_enabled
                    .evaluate(&self.context, &agent.name)
                    .await
                {
                    enabled_handoff = Some(handoff.clone());
                    break;
                }
            }
        }

        if let Some(tool) = enabled_tool {
            let approval_item = approval_item(&agent.name, &tool.name, &call);
            let decision_for =
                |ctx: &RunContext<C>| ctx.is_tool_approved(&agent.name, &tool.name, &call.call_id);
            // A standing decision wins over the approval rule, so the rule is not consulted
            // once the reviewer has decided for this tool.
            let mut decision = decision_for(&self.context);
            if decision.is_none() {
                let needs_approval = tool
                    .evaluate_needs_approval(&self.context, &call.arguments, &call.call_id)
                    .await;
                // A decision may have landed while the rule was running.
                decision = decision_for(&self.context);
                if decision.is_none() && !needs_approval {
                    decision = Some(true);
                }
            }
            match decision {
                Some(false) => return self.send_tool_rejection(&agent, &tool, &call).await,
                Some(true) => return self.execute_tool(agent, tool, call).await,
                None => {}
            }

            if self
                .run_config
                .tool_execution
                .pre_approval_tool_input_guardrails
                && !tool.tool_input_guardrails.is_empty()
            {
                // Catch a blocked call before bothering a human with it. The same guardrails
                // run again before the tool executes.
                let function_call = function_call_item(&call);
                let tool_context =
                    ToolContext::from_call(self.context.clone(), &agent.name, &function_call);
                let verdict = self
                    .run_tool_input_guardrails(&tool, &tool_context, &agent.name)
                    .await?;
                if let ToolInputVerdict::Raise(error) = verdict {
                    return Err(*error);
                }
                // A standing decision recorded while the guardrails ran takes precedence.
                match decision_for(&self.context) {
                    Some(false) => return self.send_tool_rejection(&agent, &tool, &call).await,
                    Some(true) => return self.execute_tool(agent, tool, call).await,
                    None => {}
                }
                if let ToolInputVerdict::Reject(message) = verdict {
                    return self
                        .complete_tool_call(&agent, &tool, &call, ToolOutput::Text(message))
                        .await;
                }
            }
            {
                let mut state = self.state.lock().expect("realtime session poisoned");
                state.pending_tool_calls.insert(
                    call.call_id.clone(),
                    PendingToolCall {
                        call: call.clone(),
                        agent: agent.clone(),
                        tool: tool.clone(),
                        approval_item,
                        identity: InvocationIdentity {
                            name: call.name.clone(),
                            role: InvocationRole::Function,
                            arguments: call.arguments.clone(),
                        },
                    },
                );
            }
            self.emit(RealtimeSessionEvent::ToolApprovalRequired {
                agent,
                tool,
                call_id: call.call_id.clone(),
                arguments: call.arguments.clone(),
                info: self.info(),
            });
            return Ok(());
        }

        if let Some(handoff) = enabled_handoff {
            return self.execute_handoff(agent, handoff, call).await;
        }

        let default_message = format!("Tool {} not found", call.name);
        let message = self.format_tool_error(
            RealtimeToolErrorKind::ToolNotFound,
            &call.name,
            &call.call_id,
            &default_message,
        );
        self.commit_tool_output(
            &call,
            PendingToolOutput {
                output: ToolOutput::Text(message.clone()),
                agent,
                tool: None,
                // A tool the session cannot run should not prompt a new response.
                start_response: false,
                session_update: None,
                error_message: Some(message),
            },
        )
        .await
    }

    /// Run a function tool's input guardrails and report what they decided.
    async fn run_tool_input_guardrails(
        &self,
        tool: &Arc<FunctionTool<C>>,
        tool_context: &ToolContext<C>,
        agent_name: &str,
    ) -> Result<ToolInputVerdict> {
        for guardrail in &tool.tool_input_guardrails {
            let output = guardrail
                .run(ToolInputGuardrailData {
                    context: tool_context.clone(),
                    agent_name: agent_name.to_string(),
                })
                .await?;
            match output.behavior.clone() {
                ToolGuardrailBehavior::Allow => {}
                ToolGuardrailBehavior::RejectContent { message } => {
                    return Ok(ToolInputVerdict::Reject(message));
                }
                ToolGuardrailBehavior::RaiseException => {
                    return Ok(ToolInputVerdict::Raise(Box::new(
                        AgentsError::ToolInputGuardrailTripwireTriggered {
                            guardrail_name: guardrail.name.clone(),
                            output: Box::new(output),
                            run_data: None,
                        },
                    )));
                }
            }
        }
        Ok(ToolInputVerdict::Allow)
    }

    /// Run a function tool and send its output back to the model.
    async fn execute_tool(
        &self,
        agent: Arc<RealtimeAgent<C>>,
        tool: Arc<FunctionTool<C>>,
        call: RealtimeModelToolCallEvent,
    ) -> Result<()> {
        let function_call = function_call_item(&call);
        let tool_context =
            ToolContext::from_call(self.context.clone(), &agent.name, &function_call);

        match self
            .run_tool_input_guardrails(&tool, &tool_context, &agent.name)
            .await?
        {
            ToolInputVerdict::Allow => {}
            ToolInputVerdict::Reject(message) => {
                return self
                    .complete_tool_call(&agent, &tool, &call, ToolOutput::Text(message))
                    .await;
            }
            ToolInputVerdict::Raise(error) => return Err(*error),
        }

        for hooks in self.hooks_for(&agent) {
            hooks.on_tool_start(&self.context, &agent, &tool).await;
        }
        self.emit(RealtimeSessionEvent::ToolStart {
            agent: agent.clone(),
            tool: tool.clone(),
            arguments: call.arguments.clone(),
            info: self.info(),
        });
        self.mark_executed(&call, InvocationRole::Function);

        let invocation = tool.invoke(tool_context.clone(), call.arguments.clone());
        let result = match tool.timeout {
            Some(duration) => match tokio::time::timeout(duration, invocation).await {
                Ok(result) => result,
                Err(_) => {
                    let error = AgentsError::ToolTimeoutError {
                        tool_name: tool.name.clone(),
                        timeout_seconds: duration.as_secs_f64(),
                        run_data: None,
                    };
                    match tool.timeout_behavior {
                        ToolTimeoutBehavior::RaiseException => return Err(error),
                        ToolTimeoutBehavior::ErrorAsResult => {
                            let message = match &tool.timeout_error_function {
                                Some(f) => f(&self.context, &error),
                                None => format!(
                                    "Tool '{}' timed out after {} seconds.",
                                    tool.name,
                                    duration.as_secs_f64()
                                ),
                            };
                            Ok(ToolOutput::Text(message))
                        }
                    }
                }
            },
            None => invocation.await,
        };

        let mut output = match result {
            Ok(output) => output,
            Err(error) => match &tool.failure_error_function {
                Some(f) => {
                    log_tool_action_error(&format!("Realtime tool {} failed", tool.name), &error);
                    ToolOutput::Text(f(&self.context, &error))
                }
                None => return Err(error),
            },
        };

        for guardrail in &tool.tool_output_guardrails {
            let checked = guardrail
                .run(ToolOutputGuardrailData {
                    context: tool_context.clone(),
                    agent_name: agent.name.clone(),
                    output: output.clone(),
                })
                .await?;
            match checked.behavior.clone() {
                ToolGuardrailBehavior::Allow => {}
                ToolGuardrailBehavior::RejectContent { message } => {
                    output = ToolOutput::Text(message);
                    break;
                }
                ToolGuardrailBehavior::RaiseException => {
                    return Err(AgentsError::ToolOutputGuardrailTripwireTriggered {
                        guardrail_name: guardrail.name.clone(),
                        output: Box::new(checked),
                        run_data: None,
                    });
                }
            }
        }

        self.complete_tool_call(&agent, &tool, &call, output).await
    }

    /// Send a tool output, record it in the history and emit `tool_end`.
    async fn complete_tool_call(
        &self,
        agent: &Arc<RealtimeAgent<C>>,
        tool: &Arc<FunctionTool<C>>,
        call: &RealtimeModelToolCallEvent,
        output: ToolOutput,
    ) -> Result<()> {
        self.commit_tool_output(
            call,
            PendingToolOutput {
                output,
                agent: agent.clone(),
                tool: Some(tool.clone()),
                start_response: true,
                session_update: None,
                error_message: None,
            },
        )
        .await
    }

    /// Deliver a tool output, caching it for a retry when the send fails.
    ///
    /// The call has already run by this point, so a failed send must not lose the output: the
    /// session keeps it under the call id and resends it if the model retries that call.
    async fn commit_tool_output(
        &self,
        call: &RealtimeModelToolCallEvent,
        pending: PendingToolOutput<C>,
    ) -> Result<()> {
        let serialized = pending.output.to_model_string();
        if let Err(error) = self.send_pending_output(call, &pending, &serialized).await {
            log_tool_action_warning(
                "Realtime tool output send failed; the cached output will be retried",
                &error,
            );
            self.state
                .lock()
                .expect("realtime session poisoned")
                .pending_tool_outputs
                .insert(call.call_id.clone(), pending);
            self.emit_error("Tool output send failed; cached output will be retried");
            return Ok(());
        }
        {
            let mut state = self.state.lock().expect("realtime session poisoned");
            state.committed_tool_calls.insert(call.call_id.clone());
            merge_item(
                &mut state.history,
                tool_call_history_item(call, &serialized),
            );
        }
        if let Some(tool) = &pending.tool {
            for hooks in self.hooks_for(&pending.agent) {
                hooks
                    .on_tool_end(&self.context, &pending.agent, tool, &pending.output)
                    .await;
            }
            self.emit(RealtimeSessionEvent::ToolEnd {
                agent: pending.agent.clone(),
                tool: tool.clone(),
                arguments: call.arguments.clone(),
                output: pending.output,
                info: self.info(),
            });
        }
        if let Some(message) = pending.error_message {
            self.emit_error(message);
        }
        Ok(())
    }

    /// Send a pending output, preceded by the session update it depends on.
    async fn send_pending_output(
        &self,
        call: &RealtimeModelToolCallEvent,
        pending: &PendingToolOutput<C>,
        serialized: &str,
    ) -> Result<()> {
        if let Some(settings) = &pending.session_update {
            // The new agent's settings must land before the output starts a new response.
            self.model
                .send_event(RealtimeModelSendEvent::SessionUpdate {
                    session_settings: settings.clone(),
                })
                .await?;
        }
        self.send_tool_output(call, serialized, pending.start_response)
            .await
    }

    /// Send one `function_call_output` to the model.
    async fn send_tool_output(
        &self,
        call: &RealtimeModelToolCallEvent,
        output: &str,
        start_response: bool,
    ) -> Result<()> {
        self.model
            .send_event(RealtimeModelSendEvent::ToolOutput {
                tool_call: call.clone(),
                output: output.to_string(),
                start_response,
            })
            .await
    }

    /// Record that a call's tool ran, so a retry cannot run it twice.
    fn mark_executed(&self, call: &RealtimeModelToolCallEvent, role: InvocationRole) {
        self.state
            .lock()
            .expect("realtime session poisoned")
            .executed_tool_calls
            .insert(
                call.call_id.clone(),
                InvocationIdentity {
                    name: call.name.clone(),
                    role,
                    arguments: call.arguments.clone(),
                },
            );
    }

    /// Tell the model a call was not approved.
    async fn send_tool_rejection(
        &self,
        agent: &Arc<RealtimeAgent<C>>,
        tool: &Arc<FunctionTool<C>>,
        call: &RealtimeModelToolCallEvent,
    ) -> Result<()> {
        // An explicit per-call message from the reviewer wins over any formatter.
        let message = match self
            .context
            .rejection_message(&agent.name, &tool.name, &call.call_id)
        {
            Some(message) => message,
            None => self.format_tool_error(
                RealtimeToolErrorKind::ApprovalRejected,
                &tool.name,
                &call.call_id,
                DEFAULT_APPROVAL_REJECTION_MESSAGE,
            ),
        };
        self.complete_tool_call(agent, tool, call, ToolOutput::Text(message))
            .await
    }

    /// Let the run config reword a message the SDK sends in place of a tool result.
    fn format_tool_error(
        &self,
        kind: RealtimeToolErrorKind,
        tool_name: &str,
        call_id: &str,
        default_message: &str,
    ) -> String {
        let Some(formatter) = &self.run_config.tool_error_formatter else {
            return default_message.to_string();
        };
        let args = RealtimeToolErrorArgs {
            kind,
            tool_type: "function".to_string(),
            tool_name: tool_name.to_string(),
            call_id: call_id.to_string(),
            default_message: default_message.to_string(),
            run_context: &self.context,
            error: None,
        };
        formatter(&args).unwrap_or_else(|| default_message.to_string())
    }

    /// Switch to the handoff target, push its settings and answer the call.
    async fn execute_handoff(
        &self,
        from_agent: Arc<RealtimeAgent<C>>,
        handoff: RealtimeHandoff<C>,
        call: RealtimeModelToolCallEvent,
    ) -> Result<()> {
        self.mark_executed(&call, InvocationRole::Handoff);
        let to_agent = handoff
            .invoke(self.context.clone(), call.arguments.clone())
            .await?;
        let resolved = self.build_settings(&to_agent, None).await?;
        {
            let mut state = self.state.lock().expect("realtime session poisoned");
            state.agent = to_agent.clone();
            state.tools = resolved.tools.clone();
            state.handoffs = resolved.handoffs.clone();
        }

        for hooks in self.hooks_for(&to_agent) {
            hooks
                .on_handoff(&self.context, &from_agent, &to_agent)
                .await;
        }
        self.emit(RealtimeSessionEvent::Handoff {
            from_agent,
            to_agent: to_agent.clone(),
            info: self.info(),
        });

        let output = RealtimeHandoff::<C>::transfer_message(&to_agent.name);
        self.commit_tool_output(
            &call,
            PendingToolOutput {
                output: ToolOutput::Text(output),
                agent: to_agent,
                tool: None,
                start_response: true,
                session_update: Some(Box::new(resolved.settings)),
                error_message: None,
            },
        )
        .await
    }

    /// Open an output-response generation for deltas that arrive before `turn_started`.
    ///
    /// Port of `_reconcile_output_response`.
    fn reconcile_output_response(&self, response_id: &str) {
        let mut state = self.state.lock().expect("realtime session poisoned");
        if state.active_output_response_generation.is_none() {
            state.active_output_response_generation = Some(self.next_generation());
            state.active_response_id = Some(response_id.to_string());
            state.active_response_agent = Some(state.agent.clone());
        } else if state.active_response_id.is_none() {
            state.active_response_id = Some(response_id.to_string());
        }
    }

    /// Run output guardrails once the accumulated transcript crosses the debounce threshold.
    ///
    /// The run happens on its own task so a slow guardrail cannot stall delivery of the audio and
    /// transcript events queued behind it, which is what the Python SDK's background tasks buy.
    fn spawn_output_guardrails(
        self: &Arc<Self>,
        item_id: &str,
        response_id: &str,
        is_audio_output: bool,
    ) {
        let threshold = self.debounce_text_length;
        let (text, agent, generation) = {
            let mut state = self.state.lock().expect("realtime session poisoned");
            let Some(text) = state.item_transcripts.get(item_id).cloned() else {
                return;
            };
            let runs = state
                .item_guardrail_runs
                .get(item_id)
                .copied()
                .unwrap_or_default();
            if text.chars().count() < (runs + 1) * threshold {
                return;
            }
            let crossed = text.chars().count() / threshold;
            state
                .item_guardrail_runs
                .insert(item_id.to_string(), crossed.max(runs + 1));
            let agent = state
                .active_response_agent
                .clone()
                .unwrap_or_else(|| state.agent.clone());
            let generation = state.active_output_response_generation;
            *state
                .guardrail_runs_by_response
                .entry(response_id.to_string())
                .or_default() += 1;
            (text, agent, generation)
        };

        let inner = self.clone();
        let response_id = response_id.to_string();
        let task = tokio::spawn(async move {
            inner
                .run_output_guardrails(&text, &response_id, &agent, generation, is_audio_output)
                .await;
            inner.finish_guardrail_run(&response_id).await;
        });
        let mut tasks = self
            .guardrail_tasks
            .lock()
            .expect("realtime session poisoned");
        tasks.retain(|task| !task.is_finished());
        tasks.push(task);
    }

    /// Run the agent's and the run config's output guardrails on `text`.
    ///
    /// `generation` scopes the interruption: a guardrail that trips on a response the model has
    /// already moved past must not cancel the newer one, so it only stops that response's audio
    /// playback. Port of `_run_output_guardrails`.
    async fn run_output_guardrails(
        &self,
        text: &str,
        response_id: &str,
        agent: &Arc<RealtimeAgent<C>>,
        generation: Option<u64>,
        is_audio_output: bool,
    ) {
        let mut guardrails: Vec<OutputGuardrail<C>> = agent.output_guardrails.clone();
        // A guardrail listed on both the agent and the run config runs once. Python dedupes by
        // object identity; guardrails here are cloneable handles with no exposed identity, so the
        // name is the key.
        for guardrail in &self.run_config.output_guardrails {
            if !guardrails.iter().any(|known| known.name == guardrail.name) {
                guardrails.push(guardrail.clone());
            }
        }
        if guardrails.is_empty() {
            return;
        }
        {
            let state = self.state.lock().expect("realtime session poisoned");
            if state.interrupted_response_ids.contains(response_id) {
                return;
            }
        }

        // Output guardrails are defined against `Agent`; realtime agents carry no model, so the
        // guardrail sees a stand-in agent with the same name, as the Python SDK does.
        let stand_in = Arc::new(Agent::<C>::new(agent.name.clone()));
        let mut triggered: Vec<OutputGuardrailResult> = Vec::new();
        for guardrail in &guardrails {
            match guardrail
                .run(
                    self.context.clone(),
                    stand_in.clone(),
                    Value::String(text.to_string()),
                )
                .await
            {
                Ok(result) if result.output.tripwire_triggered => triggered.push(result),
                Ok(_) => {}
                Err(error) => {
                    // The guardrail name and error can carry model or tool data, so both
                    // logging policies apply and the name appears only when they allow it.
                    let message = if crate::logger::may_log_model_data(true) {
                        format!(
                            "Output guardrail raised an exception; skipping it (guardrail {})",
                            guardrail.name
                        )
                    } else {
                        "Output guardrail raised an exception; skipping it".to_string()
                    };
                    crate::logger::log_model_and_tool_action_warning(&message, &error);
                }
            }
        }
        if triggered.is_empty() {
            return;
        }

        {
            let mut state = self.state.lock().expect("realtime session poisoned");
            if !state
                .interrupted_response_ids
                .insert(response_id.to_string())
            {
                return;
            }
        }
        let latest_generation = self.latest_generation();

        let names: Vec<String> = triggered
            .iter()
            .map(|result| result.guardrail_name.clone())
            .collect();
        if !self.emit(RealtimeSessionEvent::GuardrailTripped {
            guardrail_results: triggered,
            message: text.to_string(),
            info: self.info(),
        }) {
            return;
        }

        let active_generation = {
            self.state
                .lock()
                .expect("realtime session poisoned")
                .active_output_response_generation
        };
        let superseded = generation.is_some_and(|generation| generation != latest_generation);
        let interrupt = match generation {
            // Without a known generation there is nothing newer to protect.
            None => Some(RealtimeInterrupt::forced()),
            Some(_) if superseded => {
                // The model moved on: silence this response's audio but leave the new one running.
                is_audio_output.then(|| RealtimeInterrupt::playback_only(response_id))
            }
            Some(generation) if Some(generation) == active_generation => Some(
                RealtimeInterrupt::forced_for_response(response_id, is_audio_output),
            ),
            Some(_) => is_audio_output.then(|| RealtimeInterrupt::playback_only(response_id)),
        };
        if let Some(interrupt) = interrupt
            && let Err(error) = self
                .model
                .send_event(RealtimeModelSendEvent::Interrupt(interrupt))
                .await
        {
            self.emit_error(error.to_string());
            return;
        }
        if superseded {
            // Telling a newer response about an older response's guardrail would confuse it.
            return;
        }

        let feedback = RealtimeModelSendEvent::UserInput {
            user_input: RealtimeUserInput::Text(format!(
                "guardrail triggered: {}",
                names.join(", ")
            )),
        };
        let result = match generation {
            None => self.model.send_event(feedback).await.map(|()| true),
            Some(generation) => {
                // Recheck at the transport's write boundary: a response may start in between.
                let latest = Arc::clone(&self.latest_generation);
                self.model
                    .send_event_if(
                        feedback,
                        Box::new(move || generation == latest.load(Ordering::SeqCst)),
                    )
                    .await
            }
        };
        if let Err(error) = result {
            self.emit_error(error.to_string());
        }
    }

    /// Note that one guardrail run finished, releasing transport audio when the last one does.
    async fn finish_guardrail_run(&self, response_id: &str) {
        let should_retire = {
            let mut state = self.state.lock().expect("realtime session poisoned");
            let remaining = match state.guardrail_runs_by_response.get_mut(response_id) {
                Some(count) => {
                    *count = count.saturating_sub(1);
                    *count
                }
                None => 0,
            };
            if remaining == 0 {
                state.guardrail_runs_by_response.remove(response_id);
                state
                    .responses_awaiting_guardrail_cleanup
                    .remove(response_id)
            } else {
                false
            }
        };
        if should_retire {
            self.retire_response_audio(response_id).await;
        }
    }

    /// A response ended: release its transport audio, or wait for its guardrails to finish first.
    async fn finish_response_guardrail_lifecycle(&self, response_id: &str) {
        let pending = {
            let mut state = self.state.lock().expect("realtime session poisoned");
            if state.guardrail_runs_by_response.contains_key(response_id) {
                state
                    .responses_awaiting_guardrail_cleanup
                    .insert(response_id.to_string());
                true
            } else {
                false
            }
        };
        if !pending {
            self.retire_response_audio(response_id).await;
        }
    }

    /// Let the transport drop the audio bookkeeping it held for a finished response.
    async fn retire_response_audio(&self, response_id: &str) {
        if let Err(error) = self.model.retire_response_audio(response_id).await {
            self.emit_error(format!("Response audio cleanup failed: {error}"));
        }
        self.state
            .lock()
            .expect("realtime session poisoned")
            .interrupted_response_ids
            .remove(response_id);
    }
}

fn approval_item(
    agent_name: &str,
    tool_name: &str,
    call: &RealtimeModelToolCallEvent,
) -> ToolApprovalItem {
    ToolApprovalItem {
        agent: agent_name.to_string(),
        raw_item: function_call_item(call),
        tool_name: tool_name.to_string(),
        tool_origin: None,
    }
}

fn function_call_item(call: &RealtimeModelToolCallEvent) -> FunctionCall {
    FunctionCall {
        id: call.id.clone(),
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        status: None,
        extra: serde_json::Map::new(),
    }
}

fn tool_call_history_item(call: &RealtimeModelToolCallEvent, output: &str) -> RealtimeItem {
    RealtimeItem::FunctionCall(RealtimeToolCallItem {
        item_id: call.id.clone().unwrap_or_else(|| call.call_id.clone()),
        previous_item_id: call.previous_item_id.clone(),
        call_id: Some(call.call_id.clone()),
        status: "completed".to_string(),
        arguments: call.arguments.clone(),
        name: call.name.clone(),
        output: Some(output.to_string()),
        extra: serde_json::Map::new(),
    })
}

/// Reject duplicate model-visible tool names, as `_tool_validation` does in Python.
fn validate_tool_names(specs: &[RealtimeFunctionToolSpec]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut duplicates: Vec<&str> = Vec::new();
    for spec in specs {
        if !seen.insert(spec.name.as_str()) && !duplicates.contains(&spec.name.as_str()) {
            duplicates.push(spec.name.as_str());
        }
    }
    if duplicates.is_empty() {
        return Ok(());
    }
    Err(AgentsError::user(format!(
        "Duplicate realtime tool names found: {}. Realtime function tool and handoff names must \
         be unique. Rename one of them before starting the session.",
        duplicates.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_tool_names_are_rejected() {
        let specs = vec![
            RealtimeFunctionToolSpec::new("a", "", json!({})),
            RealtimeFunctionToolSpec::new("a", "", json!({})),
        ];
        let error = validate_tool_names(&specs).unwrap_err();
        assert!(error.to_string().contains("Duplicate realtime tool names"));
        assert!(validate_tool_names(&specs[..1]).is_ok());
    }

    #[test]
    fn history_item_prefers_item_id_over_call_id() {
        let mut call = RealtimeModelToolCallEvent::new("t", "call_1", "{}");
        let item = tool_call_history_item(&call, "out");
        assert_eq!(item.item_id(), "call_1");
        call.id = Some("fc_1".to_string());
        let item = tool_call_history_item(&call, "out");
        assert_eq!(item.item_id(), "fc_1");
        assert_eq!(
            item.as_function_call().unwrap().output.as_deref(),
            Some("out")
        );
    }
}
