//! The agent loop. Port of `agents.run_internal`.

pub(crate) mod history;
mod items;
mod local_tools;
mod model_call;
mod oai_conversation;
pub(crate) mod prompt_cache_key;
mod tool_caller;
pub(crate) mod tool_invocation;
mod tools;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentModel, ToolUseBehavior};
use crate::error::{AgentsError, Result, RunErrorDetails};
use crate::guardrail::{InputGuardrail, InputGuardrailResult, OutputGuardrailResult};
use crate::handoffs::{Handoff, HandoffInputData};
use crate::items::{
    FunctionCall, InputItem, ItemHelpers, MessageOutput, ModelResponse, OutputItem, RunInput,
    RunItem, RunItemKind, ToolApprovalItem,
};
use crate::lifecycle::RunHooks;
use crate::memory::Session;
use crate::model_settings::ModelSettings;
use crate::models::interface::{Model, ModelRequest, ModelTracing, ToolSpec};
use crate::result::{RunResult, StreamMsg};
use crate::run::RunOptions;
use crate::run_config::{
    CallModelData, ModelInputData, ReasoningItemIdPolicy, RunConfig, ToolNotFoundBehavior,
};
use crate::run_context::RunContext;
use crate::run_error_handlers::{
    RunErrorData, RunErrorHandler, RunErrorHandlerInput, RunErrorHandlers,
};
use crate::run_state::{
    CURRENT_SCHEMA_VERSION, ProcessedResponseSnapshot, RunState, ToolInvocationRecord,
    ToolRunSnapshot,
};
use crate::stream_events::{RunItemStreamEventName, StreamEvent};
use crate::tool::{FunctionTool, HostedMCPTool, MCPToolApprovalRequest, Tool, ToolOutput};
use crate::tool_guardrails::{
    ToolGuardrailBehavior, ToolInputGuardrailResult, ToolOutputGuardrailResult,
};
use crate::tracing::{self as tr, Span, SpanData, scope};

pub use prompt_cache_key::resolve_run_grouping_id;
pub(crate) use tools::ToolEnv;

/// Sends stream events from the run task to the streamed result.
pub(crate) struct EventSink<C> {
    pub tx: mpsc::UnboundedSender<StreamMsg<C>>,
    pub cancel_after_turn: Arc<AtomicBool>,
}

impl<C> EventSink<C> {
    pub(crate) fn send(&self, ev: StreamEvent<C>) {
        let _ = self.tx.send(StreamMsg::Event(ev));
    }
}

/// What happens after a turn.
enum NextStep<C> {
    FinalOutput(Value),
    Handoff(Arc<Agent<C>>),
    RunAgain,
    Interruption(Vec<ToolApprovalItem>),
}

/// The classified tool activity of a model response.
struct ProcessedResponse<C> {
    functions: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
    handoffs: Vec<(FunctionCall, Handoff<C>)>,
    mcp_approvals: Vec<(Value, Option<HostedMCPTool<C>>)>,
    local_calls: Vec<local_tools::LocalCall<C>>,
    tools_used: Vec<String>,
    last_message: Option<MessageOutput>,
    has_tool_activity: bool,
    /// Position of each tool call id in the model response, used to order the outputs.
    call_positions: HashMap<String, usize>,
}

impl<C> ProcessedResponse<C> {
    /// Capture the classification in the serializable shape `RunState` persists, so an
    /// interrupted snapshot carries what the Python runner needs to continue the turn.
    /// Mirrors `RunState._serialize_processed_response`.
    fn snapshot(&self) -> ProcessedResponseSnapshot {
        let function_run = |call: &FunctionCall, tool: &FunctionTool<C>| ToolRunSnapshot {
            tool_name: tool.name.clone(),
            namespace: tool.namespace().map(str::to_string),
            tool_call: call.to_value(),
            description: Some(tool.description.clone()),
            params_json_schema: Some(tool.params_json_schema.clone()),
        };
        let local_run = |raw: &Value, name: String| ToolRunSnapshot {
            tool_name: name,
            namespace: None,
            tool_call: raw.clone(),
            description: None,
            params_json_schema: None,
        };
        let mut snapshot = ProcessedResponseSnapshot {
            tools_used: self.tools_used.clone(),
            functions: self
                .functions
                .iter()
                .map(|(call, tool)| function_run(call, tool))
                .collect(),
            handoffs: self
                .handoffs
                .iter()
                .map(|(call, h)| ToolRunSnapshot {
                    tool_name: h.tool_name.clone(),
                    namespace: None,
                    tool_call: call.to_value(),
                    description: None,
                    params_json_schema: None,
                })
                .collect(),
            mcp_approval_requests: self
                .mcp_approvals
                .iter()
                .map(|(raw, tool)| {
                    json!({
                        "request_item": {"raw_item": raw},
                        "mcp_tool": match tool.as_ref().and_then(HostedMCPTool::server_label) {
                            Some(label) => json!({"name": label}),
                            None => json!({}),
                        },
                    })
                })
                .collect(),
            ..Default::default()
        };
        for call in &self.local_calls {
            let name = call.tool_name();
            match call {
                local_tools::LocalCall::Computer(raw, _) => {
                    snapshot.computer_actions.push(local_run(raw, name))
                }
                local_tools::LocalCall::Shell(raw, _) => {
                    snapshot.shell_actions.push(local_run(raw, name))
                }
                local_tools::LocalCall::LocalShell(raw, _) => {
                    snapshot.local_shell_actions.push(local_run(raw, name))
                }
                local_tools::LocalCall::ApplyPatch(raw, _) => {
                    snapshot.apply_patch_actions.push(local_run(raw, name))
                }
                local_tools::LocalCall::Custom(raw, _) => {
                    snapshot.custom_tool_actions.push(local_run(raw, name))
                }
            }
        }
        snapshot
    }
}

struct RunLoop<C> {
    starting_agent: Arc<Agent<C>>,
    current_agent: Arc<Agent<C>>,
    ctx: RunContext<C>,
    config: Arc<RunConfig<C>>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    error_handlers: RunErrorHandlers<C>,
    max_turns: Option<usize>,
    session: Option<Arc<dyn Session>>,
    sink: Option<EventSink<C>>,

    original_input: Vec<InputItem>,
    /// Model input prefix: session history merged with the run input (or the run input).
    model_prefix: Vec<InputItem>,
    /// The run input items to persist to the session.
    session_input: Vec<InputItem>,
    generated_items: Vec<RunItem>,
    /// Items forwarded to the model (differs from `generated_items` after handoff filters).
    model_items: Vec<RunItem>,
    model_responses: Vec<ModelResponse>,
    current_turn: usize,
    input_guardrail_results: Vec<InputGuardrailResult>,
    output_guardrail_results: Vec<OutputGuardrailResult>,
    tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
    tools_used: HashMap<String, Vec<String>>,
    previous_response_id: Option<String>,
    conversation_id: Option<String>,
    auto_previous_response_id: bool,
    server_sent_items: usize,
    server_input_sent: bool,
    session_saved_items: usize,
    input_saved_to_session: bool,
    pending_input: Vec<InputItem>,
    resume_pending_tools: bool,
    is_resumed: bool,
    should_run_agent_start_hooks: bool,
    agent_span: Option<Span>,
    trace_id: Option<String>,
    /// Canonical identity of every tool call the run has seen, restored from the snapshot.
    tool_invocations: BTreeMap<String, ToolInvocationRecord>,
    /// Call ids whose restored approval decision no ledger entry binds to an invocation, so
    /// the decision does not authorize the call and the user is asked again. Mirrors
    /// `RunContextWrapper._restored_unbound_approval_call_ids`; empty unless this run resumed
    /// a snapshot that is held to the binding rule.
    unbound_approvals: BTreeSet<String>,
    /// Identifies this run to models that keep per-run state.
    model_owner: crate::models::run_context::ModelRunOwner,
    /// Every model this run resolved, so each is told when the run ends.
    models_used: Vec<Arc<dyn Model>>,
    /// Trace settings the snapshot was traced with, so a resume keeps the same trace.
    trace_state: Option<tr::TraceState>,
    /// Index in `generated_items` where the current turn's items start.
    turn_start: usize,
    /// Classification of the model response the current turn is executing, kept so an
    /// interrupted snapshot can carry Python's `last_processed_response`.
    last_processed_response: Option<ProcessedResponseSnapshot>,
    /// Index in the tool guardrail result lists where the current turn's results start.
    turn_tool_input_guardrail_start: usize,
    turn_tool_output_guardrail_start: usize,
    /// Metadata of the nested `Agent::as_tool` invocation that started this run, if any.
    agent_tool_invocation: Option<crate::result::AgentToolInvocation>,
    /// Whether a handoff filter or nested history made model history differ from the public
    /// history, so `to_input_list(Normalized)` should replay the model view.
    model_history_diverged: bool,
    /// What a server-managed conversation already holds.
    server_conversation: oai_conversation::ServerConversationTracker,
    /// The one prompt cache key this run generates, reused across turns.
    prompt_cache_key: prompt_cache_key::PromptCacheKeyResolver,
}

fn find_agent<C>(start: &Arc<Agent<C>>, name: &str) -> Option<Arc<Agent<C>>> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(a) = queue.pop_front() {
        if a.name == name {
            return Some(a);
        }
        if !seen.insert(Arc::as_ptr(&a) as usize) {
            continue;
        }
        for h in &a.handoffs {
            if let Some(t) = h.agent.as_ref() {
                queue.push_back(t.clone());
            }
        }
    }
    None
}

/// Names shared by two distinct agents in the graph reachable from `start`. A run state
/// identifies its agent by name, so such a graph cannot be resumed unambiguously.
/// Mirrors Python's `_has_ambiguous_hook_identities` check.
fn ambiguous_agent_names<C>(start: &Arc<Agent<C>>) -> Vec<String> {
    let mut seen_ptrs = HashSet::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(a) = queue.pop_front() {
        if !seen_ptrs.insert(Arc::as_ptr(&a) as usize) {
            continue;
        }
        *by_name.entry(a.name.clone()).or_default() += 1;
        for h in &a.handoffs {
            if let Some(t) = h.agent.as_ref() {
                queue.push_back(t.clone());
            }
        }
    }
    let mut names: Vec<String> = by_name
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name)
        .collect();
    names.sort();
    names
}

fn strip_reasoning_id(item: InputItem, policy: ReasoningItemIdPolicy) -> InputItem {
    if policy == ReasoningItemIdPolicy::Omit && item.item_type() == Some("reasoning") {
        let mut v = item.0;
        if let Value::Object(m) = &mut v {
            m.remove("id");
        }
        return InputItem(v);
    }
    item
}

/// Drop the provider-assigned item id, so a store that owns its history can still match a
/// replayed item against what it holds. Port of `digest_input_item(ignore_ids_for_matching)`.
pub(crate) fn strip_item_id(mut item: InputItem) -> InputItem {
    if let Value::Object(map) = &mut item.0 {
        map.remove("id");
    }
    item
}

fn items_to_input(items: &[RunItem]) -> Vec<InputItem> {
    items
        .iter()
        .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
        .map(RunItem::to_input_item)
        .collect()
}

type ToolsAndHandoffs<C> = (Vec<Tool<C>>, Vec<Handoff<C>>);

/// Resolve collisions between bare function tool names and handoff names. The winner is the
/// last handoff with the name, else the last tool.
fn resolve_tool_name_collisions<C>(
    tools: Vec<Tool<C>>,
    handoffs: Vec<Handoff<C>>,
    policy: crate::run_config::ToolNameCollisionPolicy,
) -> Result<ToolsAndHandoffs<C>> {
    // (is_handoff, index) owners per bare name.
    let mut owners: HashMap<String, Vec<(bool, usize)>> = HashMap::new();
    for (i, t) in tools.iter().enumerate() {
        if let Some(f) = t.as_function()
            && f.namespace().is_none()
        {
            owners.entry(f.name.clone()).or_default().push((false, i));
        }
    }
    for (i, h) in handoffs.iter().enumerate() {
        owners
            .entry(h.tool_name.clone())
            .or_default()
            .push((true, i));
    }
    let mut drop_tools = HashSet::new();
    let mut drop_handoffs = HashSet::new();
    for (name, entries) in owners.iter().filter(|(_, e)| e.len() > 1) {
        let kinds: Vec<&str> = entries
            .iter()
            .map(|(h, _)| if *h { "handoff" } else { "function tool" })
            .collect();
        let message = format!(
            "Tool name collision: '{name}' is used by {}. Assign unique tool names (for example \
             with `tool_name_override` on handoffs).",
            kinds.join(", ")
        );
        if policy == crate::run_config::ToolNameCollisionPolicy::Error {
            return Err(AgentsError::user(message));
        }
        log::warn!(target: crate::logger::LOG_TARGET, "{message}");
        let winner = entries
            .iter()
            .rev()
            .find(|(h, _)| *h)
            .or(entries.last())
            .copied()
            .expect("non-empty");
        for e in entries.iter().filter(|e| **e != winner) {
            if e.0 {
                drop_handoffs.insert(e.1);
            } else {
                drop_tools.insert(e.1);
            }
        }
    }
    if drop_tools.is_empty() && drop_handoffs.is_empty() {
        return Ok((tools, handoffs));
    }
    Ok((
        tools
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop_tools.contains(i))
            .map(|(_, t)| t)
            .collect(),
        handoffs
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop_handoffs.contains(i))
            .map(|(_, h)| h)
            .collect(),
    ))
}

/// The function tool key for a call: `namespace.name` for namespaced calls. Deferred top-level
/// tools come back with `namespace == name` and resolve to the bare name.
fn function_lookup_name(fc: &FunctionCall) -> String {
    match fc.extra.get("namespace").and_then(Value::as_str) {
        Some(ns) if !ns.is_empty() && ns != fc.name => format!("{ns}.{}", fc.name),
        _ => fc.name.clone(),
    }
}

/// Match a non-function output item to a local runtime tool of the agent.
fn match_local_call<C>(
    tools: &[Tool<C>],
    v: &Value,
    ty: &str,
    agent_name: &str,
) -> Result<Option<local_tools::LocalCall<C>>> {
    use local_tools::LocalCall;
    let missing = |what: &str| {
        Err(AgentsError::model_behavior(format!(
            "Model produced a {ty} item but the agent has no {what}."
        )))
    };
    match ty {
        "computer_call" => match tools.iter().find_map(|t| match t {
            Tool::Computer(c) => Some(c.clone()),
            _ => None,
        }) {
            Some(c) => {
                tool_caller::ensure_tool_caller_allowed(v, None, c.name(), agent_name)?;
                Ok(Some(LocalCall::Computer(v.clone(), c)))
            }
            None => missing("ComputerTool"),
        },
        // Hosted shells are executed by OpenAI: only local shell tools run here.
        "shell_call" => match tools.iter().find_map(|t| match t {
            Tool::Shell(s) if s.is_local() => Some(s.clone()),
            _ => None,
        }) {
            Some(s) => {
                tool_caller::ensure_tool_caller_allowed(
                    v,
                    s.allowed_callers.as_deref(),
                    &s.name,
                    agent_name,
                )?;
                Ok(Some(LocalCall::Shell(v.clone(), s)))
            }
            None => Ok(None),
        },
        "local_shell_call" => match tools.iter().find_map(|t| match t {
            Tool::LocalShell(s) => Some(s.clone()),
            _ => None,
        }) {
            Some(s) => {
                tool_caller::ensure_tool_caller_allowed(v, None, s.name(), agent_name)?;
                Ok(Some(LocalCall::LocalShell(v.clone(), s)))
            }
            None => missing("LocalShellTool"),
        },
        "apply_patch_call" => match tools.iter().find_map(|t| match t {
            Tool::ApplyPatch(a) => Some(a.clone()),
            _ => None,
        }) {
            Some(a) => {
                tool_caller::ensure_tool_caller_allowed(
                    v,
                    a.allowed_callers.as_deref(),
                    &a.name,
                    agent_name,
                )?;
                Ok(Some(LocalCall::ApplyPatch(v.clone(), a)))
            }
            None => missing("ApplyPatchTool"),
        },
        "custom_tool_call" => {
            let name = v.get("name").and_then(Value::as_str).unwrap_or_default();
            match tools.iter().find_map(|t| match t {
                Tool::Custom(c) if c.name == name => Some(c.clone()),
                _ => None,
            }) {
                Some(c) => {
                    tool_caller::ensure_tool_caller_allowed(
                        v,
                        c.allowed_callers.as_deref(),
                        &c.name,
                        agent_name,
                    )?;
                    Ok(Some(LocalCall::Custom(v.clone(), c)))
                }
                None => Err(AgentsError::model_behavior(format!(
                    "Custom tool {name} not found"
                ))),
            }
        }
        _ => Ok(None),
    }
}

/// Rebuild a blocked tool output from allowlisted fields only.
/// Port of `blocked_function_output_payload`.
fn blocked_tool_output_payload(raw_item: &Value, message: &str) -> Value {
    let mut sanitized = json!({
        "type": raw_item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function_call_output"),
        "output": message,
    });
    for key in ["call_id", "id", "status"] {
        if let Some(v) = raw_item.get(key).and_then(Value::as_str) {
            sanitized[key] = Value::String(v.to_string());
        }
    }
    // Only a direct caller survives sanitization; a program caller would reattach the
    // blocked item to a transcript this run must not replay.
    if raw_item
        .get("caller")
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("direct")
    {
        sanitized["caller"] = json!({"type": "direct"});
    }
    sanitized
}

fn call_id_of(item: &RunItem) -> Option<String> {
    match &item.kind {
        RunItemKind::ToolCallOutput { raw_item, .. }
        | RunItemKind::HandoffOutput {
            raw_item: InputItem(raw_item),
            ..
        } => raw_item
            .get("call_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        RunItemKind::MCPApprovalResponse { raw_item } => raw_item
            .get("approval_request_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

fn clone_simple_error(e: &AgentsError) -> AgentsError {
    match e {
        AgentsError::MaxTurnsExceeded { message, .. } => AgentsError::MaxTurnsExceeded {
            message: message.clone(),
            run_data: None,
        },
        AgentsError::ModelRefusalError { refusal, .. } => AgentsError::ModelRefusalError {
            refusal: refusal.clone(),
            run_data: None,
        },
        other => AgentsError::model_behavior(other.to_string()),
    }
}

/// Entry point used by `Runner`.
pub(crate) async fn run_loop<C: Send + Sync + 'static>(
    starting_agent: Arc<Agent<C>>,
    input: RunInput,
    options: RunOptions<C>,
    sink: Option<EventSink<C>>,
) -> Result<RunResult<C>> {
    let RunOptions {
        context,
        max_turns,
        hooks,
        run_config,
        error_handlers,
        previous_response_id,
        auto_previous_response_id,
        conversation_id,
        session,
        shares_parent_context,
        agent_tool_invocation,
    } = options;

    let mut lp = RunLoop {
        starting_agent: starting_agent.clone(),
        current_agent: starting_agent.clone(),
        ctx: context,
        config: Arc::new(run_config),
        hooks,
        error_handlers,
        max_turns,
        session,
        sink,
        original_input: Vec::new(),
        model_prefix: Vec::new(),
        session_input: Vec::new(),
        generated_items: Vec::new(),
        model_items: Vec::new(),
        agent_tool_invocation,
        model_history_diverged: false,
        server_conversation: Default::default(),
        prompt_cache_key: Default::default(),
        model_responses: Vec::new(),
        current_turn: 0,
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        tools_used: HashMap::new(),
        previous_response_id,
        conversation_id,
        auto_previous_response_id,
        server_sent_items: 0,
        server_input_sent: false,
        session_saved_items: 0,
        input_saved_to_session: false,
        pending_input: Vec::new(),
        resume_pending_tools: false,
        is_resumed: false,
        should_run_agent_start_hooks: true,
        agent_span: None,
        trace_id: None,
        tool_invocations: BTreeMap::new(),
        unbound_approvals: BTreeSet::new(),
        model_owner: crate::models::run_context::new_model_run_owner(),
        models_used: Vec::new(),
        trace_state: None,
        turn_start: 0,
        last_processed_response: None,
        turn_tool_input_guardrail_start: 0,
        turn_tool_output_guardrail_start: 0,
    };

    let server_managed = lp.conversation_id.is_some()
        || lp.previous_response_id.is_some()
        || lp.auto_previous_response_id;
    if lp.session.is_some() && server_managed {
        return Err(AgentsError::user(
            "Session persistence cannot be combined with conversation_id, previous_response_id, or auto_previous_response_id.",
        ));
    }
    // Rejected output cannot be removed from a server-managed conversation, so a blocked
    // terminal output would stay in the provider's history. Port of
    // `validate_output_guardrails_with_server_managed_conversation`, including its exemption:
    // Chat Completions owns its released warn-and-ignore behaviour for this combination.
    if server_managed
        && (!lp.starting_agent.output_guardrails.is_empty()
            || !lp.config.output_guardrails.is_empty())
        && !lp.resolve_model(&lp.starting_agent)?.is_chat_completions()
    {
        return Err(AgentsError::user(
            "Output guardrails cannot be combined with conversation_id, previous_response_id, \
             or auto_previous_response_id because rejected output cannot be removed from \
             server-managed conversation history.",
        ));
    }

    match input {
        RunInput::State(state) => lp.restore(*state, !shares_parent_context)?,
        other => {
            let items = other.to_items();
            lp.original_input = items.clone();
            lp.session_input = items.clone();
            lp.model_prefix = items;
        }
    }

    // Tracing: reuse the current trace, or start one. A run with tracing turned off always
    // gets its own disabled trace so it cannot export spans into an enabled outer trace.
    let settings = resolve_trace_settings(lp.trace_state.as_ref(), &lp.config);
    let created = tr::create_trace_for_run_with_state(
        settings.workflow_name,
        settings.trace_id.or_else(|| lp.trace_id.clone()),
        settings.group_id,
        settings.metadata,
        lp.config.tracing_api_key.clone(),
        lp.config.tracing_disabled,
        lp.trace_state.as_ref(),
        lp.is_resumed,
    );
    let trace = created
        .clone()
        .or_else(scope::current_trace)
        .expect("trace");
    lp.trace_id = Some(trace.trace_id().to_string());
    lp.trace_state = tr::TraceState::from_trace(Some(&trace));
    // A disabled run must not see the caller's span either, or its children would attach to it.
    let mask = lp.config.tracing_disabled.then(tr::masked_run_span);

    let body = async {
        let task_span = if lp.config.include_task_and_turn_spans && !lp.config.tracing_disabled {
            let s = tr::create_span(
                SpanData::Task {
                    name: lp.config.workflow_name.clone(),
                    usage: None,
                },
                None,
                false,
            );
            s.start();
            Some(s)
        } else {
            None
        };
        let res = match &task_span {
            Some(s) => scope::with_span(s.clone(), Box::pin(lp.run())).await,
            None => Box::pin(lp.run()).await,
        };
        if let Some(s) = task_span {
            let usage = lp.ctx.usage();
            s.update_data(|d| {
                if let SpanData::Task { usage: u, .. } = d {
                    *u = Some(json!({
                        "requests": usage.requests,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }));
                }
            });
            if let Err(e) = &res {
                s.set_error(
                    "Run failed",
                    Some(json!({
                        "error": tr::model_error_text(e, lp.config.trace_include_sensitive_data)
                    })),
                );
            }
            s.finish();
        }
        if let Some(s) = lp.agent_span.take() {
            s.finish();
        }
        lp.cleanup_models().await;
        res
    };
    let body = async {
        match mask {
            Some(span) => scope::with_span(span, Box::pin(body)).await,
            None => body.await,
        }
    };
    match created {
        Some(t) => t.run(body).await,
        None => body.await,
    }
}

/// Whether a run item came out of the model response rather than out of executing it.
/// These are the items Python collects into `ProcessedResponse.new_items`.
fn is_model_derived_item(kind: &RunItemKind) -> bool {
    matches!(
        kind,
        RunItemKind::MessageOutput { .. }
            | RunItemKind::Reasoning { .. }
            | RunItemKind::ToolCall { .. }
            | RunItemKind::HandoffCall { .. }
            | RunItemKind::MCPListTools { .. }
            | RunItemKind::MCPApprovalRequest { .. }
            | RunItemKind::Compaction { .. }
    )
}

/// Reject a snapshot whose items no longer match the identities its ledger recorded.
/// Port of the `ModelBehaviorError` branches of `_tool_invocation_status` /
/// `_rebind_tool_invocation`.
fn validate_restored_tool_invocations(state: &RunState) -> Result<()> {
    if state.tool_invocations.is_empty() {
        return Ok(());
    }
    let reused = || {
        AgentsError::model_behavior(
            "Model reused a tool call ID for a different invocation. \
             Use a unique call ID for each tool invocation.",
        )
    };
    let check = |raw: &Value, role: Option<&'static str>| -> Result<()> {
        let Some(id) = tool_invocation::tool_invocation_identity(raw, role) else {
            return Ok(());
        };
        let Some(record) = state.tool_invocations.get(&id.call_id) else {
            return Ok(());
        };
        if record.invocation_type != id.invocation_type
            || record.approval_scope != id.approval_scope
            || record.fingerprint != id.fingerprint
        {
            return Err(reused());
        }
        Ok(())
    };
    for item in &state.generated_items {
        if let Some((raw, role)) = invocation_raw_item(item) {
            check(&raw, role)?;
        }
    }
    // The interruption list is not checked separately: an approval item carries a
    // `FunctionCall`-shaped copy of its call (local tools keep their real payload in
    // `extra`), while the ledger was written from the tool call item in `generated_items`,
    // which is the same call in its wire shape.
    Ok(())
}

/// Build the canonical tool-invocation ledger of a run from its generated items.
/// Mirrors the state `RunContextWrapper._tool_invocations` holds at snapshot time: one record
/// per tool call, marked executed and completed once its output item is in history.
fn tool_invocation_ledger(
    items: &[RunItem],
    unbound_approvals: &BTreeSet<String>,
) -> BTreeMap<String, ToolInvocationRecord> {
    let mut ledger: BTreeMap<String, ToolInvocationRecord> = BTreeMap::new();
    for item in items {
        if let Some((raw, role)) = invocation_raw_item(item)
            && let Some(id) = tool_invocation::tool_invocation_identity(&raw, role)
        {
            // A call whose restored decision is unbound is never registered, so re-snapshotting
            // an un-renewed approval cannot turn it into a bound one. Port of the
            // `is_restored_unbound` early return in `_tool_invocation_status`.
            if unbound_approvals.contains(&id.call_id) {
                continue;
            }
            ledger
                .entry(id.call_id)
                .or_insert_with(|| ToolInvocationRecord {
                    invocation_type: id.invocation_type,
                    approval_scope: id.approval_scope,
                    fingerprint: id.fingerprint,
                    executed: false,
                    completed: false,
                });
        }
    }
    for item in items {
        let raw = match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => raw_item.clone(),
            RunItemKind::HandoffOutput { raw_item, .. } => raw_item.0.clone(),
            RunItemKind::MCPApprovalResponse { raw_item } => raw_item.clone(),
            _ => continue,
        };
        if let Some((_, call_id)) = tool_invocation::tool_output_identity(&raw)
            && let Some(record) = ledger.get_mut(&call_id)
        {
            record.executed = true;
            record.completed = true;
        }
    }
    ledger
}

/// The raw item and invocation role of a run item that represents a tool invocation.
fn invocation_raw_item(item: &RunItem) -> Option<(Value, Option<&'static str>)> {
    match &item.kind {
        RunItemKind::ToolCall { raw_item } => Some((raw_item.clone(), None)),
        RunItemKind::HandoffCall { raw_item } => {
            Some((raw_item.to_value(), Some(tool_invocation::HANDOFF_ROLE)))
        }
        RunItemKind::MCPApprovalRequest { raw_item } => Some((raw_item.clone(), None)),
        _ => None,
    }
}

/// The tracing settings a run starts with, after merging an explicit [`RunConfig`] with the
/// trace a resumed snapshot was already exporting to.
struct ResolvedTraceSettings {
    workflow_name: String,
    trace_id: Option<String>,
    group_id: Option<String>,
    metadata: Option<serde_json::Map<String, Value>>,
}

/// Resolve tracing settings, preferring explicit `RunConfig` overrides over the snapshot's
/// trace. Port of `agent_runner_helpers.resolve_trace_settings`: the workflow name comes from
/// the snapshot only while the config still carries the default one, and the harness id is
/// added to the metadata afterwards.
fn resolve_trace_settings<C>(
    trace_state: Option<&tr::TraceState>,
    config: &RunConfig<C>,
) -> ResolvedTraceSettings {
    let default_workflow_name = RunConfig::<C>::default().workflow_name;
    let mut settings = ResolvedTraceSettings {
        workflow_name: config.workflow_name.clone(),
        trace_id: config.trace_id.clone(),
        group_id: config.group_id.clone(),
        metadata: config.trace_metadata.clone(),
    };
    if let Some(state) = trace_state {
        if settings.workflow_name == default_workflow_name
            && let Some(name) = state.workflow_name.as_ref().filter(|n| !n.is_empty())
        {
            settings.workflow_name = name.clone();
        }
        if settings.trace_id.is_none() {
            settings.trace_id = state.trace_id.clone();
        }
        if settings.group_id.is_none() {
            settings.group_id = state.group_id.clone();
        }
        if settings.metadata.is_none() {
            settings.metadata = state.metadata.clone();
        }
    }
    if let Some(h) = crate::models::resolve_openai_harness_id(config.model_provider.as_ref())
        && !settings
            .metadata
            .as_ref()
            .is_some_and(|m| m.contains_key("agent_harness_id"))
    {
        settings
            .metadata
            .get_or_insert_with(Default::default)
            .insert("agent_harness_id".into(), Value::String(h));
    }
    settings
}

impl<C: Send + Sync + 'static> RunLoop<C> {
    fn restore(&mut self, state: RunState, restore_usage: bool) -> Result<()> {
        if state.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(AgentsError::user(format!(
                "Unsupported run state schema version {}",
                state.schema_version
            )));
        }
        if state.final_output.is_some() {
            return Err(AgentsError::user(
                "This run state already finished; start a new run instead.",
            ));
        }
        // Port of `reject_unrecoverable_resume`: the snapshot owns a terminal result no
        // resume can reproduce, so it is refused before any session, model or tool work.
        if state.terminal_unrecoverable {
            return Err(AgentsError::user(
                "This RunState ended at an unrecoverable boundary and cannot be resumed. \
                 Start a new run instead.",
            ));
        }
        let ambiguous = ambiguous_agent_names(&self.starting_agent);
        if !ambiguous.is_empty() {
            return Err(AgentsError::user(format!(
                "Cannot resume a run whose agent graph has same-named agents ({}). \
                 Use unique agent names before starting a resumable run.",
                ambiguous.join(", ")
            )));
        }
        let agent =
            find_agent(&self.starting_agent, &state.current_agent_name).ok_or_else(|| {
                AgentsError::user(format!(
                    "Could not find agent '{}' reachable from the starting agent",
                    state.current_agent_name
                ))
            })?;
        self.current_agent = agent;
        self.ctx.import_approvals(&state.approvals);
        self.ctx
            .import_nested_states(&state.nested_agent_tool_states);
        if restore_usage {
            self.ctx.set_usage(state.usage.clone());
        }
        self.original_input = state.original_input.clone();
        self.session_input = state.original_input.clone();
        self.generated_items = state
            .generated_items
            .iter()
            .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
            .cloned()
            .collect();
        self.model_items = self.generated_items.clone();
        self.model_responses = state.model_responses.clone();
        self.current_turn = state.current_turn;
        if let Some(m) = state.max_turns
            && self.max_turns.is_some()
        {
            self.max_turns = Some(m);
        }
        self.input_guardrail_results = state.input_guardrail_results.clone();
        self.output_guardrail_results = state.output_guardrail_results.clone();
        self.tool_input_guardrail_results = state.tool_input_guardrail_results.clone();
        self.tool_output_guardrail_results = state.tool_output_guardrail_results.clone();
        if self.previous_response_id.is_none() {
            self.previous_response_id = state.previous_response_id.clone();
        }
        if self.conversation_id.is_none() {
            self.conversation_id = state.conversation_id.clone();
        }
        self.auto_previous_response_id |= state.auto_previous_response_id;
        self.server_sent_items = state.server_sent_items.min(self.model_items.len());
        self.server_input_sent = state.server_input_sent;
        self.server_conversation.hydrate_from_state(
            &state.server_conversation,
            &state.original_input,
            &state.generated_items,
            &state.model_responses,
            state.server_input_sent,
        );
        self.server_conversation
            .set_sent_initial_input(state.server_input_sent);
        self.prompt_cache_key = prompt_cache_key::PromptCacheKeyResolver::from_generated_key(
            state.generated_prompt_cache_key.clone(),
        );
        for (agent_name, tools) in &state.tool_use_tracker {
            self.tools_used
                .entry(agent_name.clone())
                .or_default()
                .extend(tools.iter().cloned());
        }
        self.session_saved_items = state.session_saved_items.min(self.generated_items.len());
        self.input_saved_to_session = state.input_saved_to_session;
        self.pending_input = state.pending_input.clone();
        // A restored call must still be the call its ledger entry was written for. Port of
        // `RunContextWrapper._tool_invocation_status`'s identity check: a snapshot whose tool
        // arguments or routing were edited is rejected instead of silently re-approved.
        validate_restored_tool_invocations(&state)?;
        self.tool_invocations = state.tool_invocations.clone();
        // A decision the snapshot could not bind to a concrete invocation is not an
        // authorization; the resumed run raises the approval again.
        self.unbound_approvals = state.restored_unbound_approval_call_ids.clone();
        self.resume_pending_tools = state.pending_tool_processing;
        // Carried so a snapshot taken before the resumed turn re-executes still names the
        // tools the parked response asked for; `process_response` replaces it.
        self.last_processed_response = state.last_processed_response.clone();
        self.trace_id = state.trace_id.clone();
        self.trace_state = state.trace_state.clone();
        self.turn_start = state.turn_start_index.min(self.generated_items.len());
        self.is_resumed = true;
        self.should_run_agent_start_hooks = false;
        Ok(())
    }

    /// Tell every model this run resolved that the run has ended, so a model holding
    /// per-run state can release it. Port of `cleanup_models_after_run`: a model that fails
    /// to clean up is logged, never fatal.
    async fn cleanup_models(&mut self) {
        for model in std::mem::take(&mut self.models_used) {
            model.cleanup_on_run_end(self.model_owner).await;
        }
    }

    fn emit(&self, ev: StreamEvent<C>) {
        if let Some(s) = &self.sink {
            s.send(ev);
        }
    }

    fn emit_item(&self, name: RunItemStreamEventName, item: &RunItem) {
        self.emit(StreamEvent::RunItem {
            name,
            item: item.clone(),
        });
    }

    fn push_item(&mut self, item: RunItem, event: Option<RunItemStreamEventName>) {
        if let Some(name) = event {
            self.emit_item(name, &item);
        }
        self.model_items.push(item.clone());
        self.generated_items.push(item);
    }

    fn run_details(&self) -> RunErrorDetails {
        RunErrorDetails {
            input: self.original_input.clone(),
            new_items: self.generated_items.clone(),
            raw_responses: self.model_responses.clone(),
            last_agent: self.current_agent.name.clone(),
            usage: self.ctx.usage(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
        }
    }

    fn server_managed(&self) -> bool {
        self.conversation_id.is_some()
            || self.previous_response_id.is_some()
            || self.auto_previous_response_id
    }

    async fn run(&mut self) -> Result<RunResult<C>> {
        let outcome = self.run_inner().await;
        self.dispose_computers().await;
        match outcome {
            Ok(r) => Ok(r),
            Err(e) => {
                let details = self.run_details();
                Err(e.with_run_data(details))
            }
        }
    }

    /// Tear down the computers this run created. Port of `dispose_resolved_computers`.
    async fn dispose_computers(&self) {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut queue = VecDeque::from([self.starting_agent.clone(), self.current_agent.clone()]);
        while let Some(agent) = queue.pop_front() {
            if !seen.insert(Arc::as_ptr(&agent) as usize) {
                continue;
            }
            for tool in agent.tools.iter() {
                if let Tool::Computer(c) = tool {
                    c.dispose().await;
                }
            }
            for h in &agent.handoffs {
                if let Some(t) = h.agent.as_ref() {
                    queue.push_back(t.clone());
                }
            }
        }
    }

    fn start_agent_span(&mut self, tools: &[Tool<C>], handoffs: &[Handoff<C>]) {
        if self.agent_span.is_some() {
            return;
        }
        let agent = &self.current_agent;
        let span = tr::create_span(
            SpanData::Agent {
                name: agent.name.clone(),
                handoffs: Some(handoffs.iter().map(|h| h.agent_name.clone()).collect()),
                tools: Some(tools.iter().map(Tool::name).collect()),
                output_type: Some(
                    agent
                        .output_type
                        .as_ref()
                        .map(|o| o.name())
                        .unwrap_or_else(|| "str".to_string()),
                ),
            },
            None,
            false,
        );
        span.start();
        self.agent_span = Some(span);
    }

    async fn run_inner(&mut self) -> Result<RunResult<C>> {
        self.emit(StreamEvent::AgentUpdated {
            new_agent: self.current_agent.clone(),
        });
        loop {
            if let Some(s) = &self.sink
                && s.cancel_after_turn.load(Ordering::SeqCst)
            {
                self.save_session().await?;
                return Ok(self.build_result(None, Vec::new()));
            }

            let step = if self.resume_pending_tools {
                self.resume_pending_tools = false;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                let handoffs = self.current_agent.get_enabled_handoffs(&self.ctx).await;
                let (tools, handoffs) = resolve_tool_name_collisions(
                    tools,
                    handoffs,
                    self.config.tool_name_collision_policy,
                )?;
                self.start_agent_span(&tools, &handoffs);
                let span = self.agent_span.clone().expect("agent span");
                scope::with_span(span, Box::pin(self.resume_tools(tools, handoffs))).await?
            } else {
                if let Some(max) = self.max_turns
                    && self.current_turn >= max
                {
                    return self.handle_max_turns(max).await;
                }
                self.current_turn += 1;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                let handoffs = self.current_agent.get_enabled_handoffs(&self.ctx).await;
                let (tools, handoffs) = resolve_tool_name_collisions(
                    tools,
                    handoffs,
                    self.config.tool_name_collision_policy,
                )?;
                self.start_agent_span(&tools, &handoffs);
                let agent_span = self.agent_span.clone().expect("agent span");
                if self.config.include_task_and_turn_spans && !self.config.tracing_disabled {
                    let turn_span = tr::create_span(
                        SpanData::Turn {
                            turn: self.current_turn as u64,
                            agent_name: self.current_agent.name.clone(),
                            usage: None,
                        },
                        Some(&agent_span),
                        false,
                    );
                    turn_span.start();
                    let before = self.ctx.usage();
                    let res = scope::with_span(
                        turn_span.clone(),
                        Box::pin(self.run_turn(tools, handoffs)),
                    )
                    .await;
                    let after = self.ctx.usage();
                    turn_span.update_data(|d| {
                        if let SpanData::Turn { usage, .. } = d {
                            *usage = Some(json!({
                                "requests": after.requests - before.requests,
                                "input_tokens": after.input_tokens - before.input_tokens,
                                "output_tokens": after.output_tokens - before.output_tokens,
                            }));
                        }
                    });
                    if let Err(e) = &res {
                        turn_span.set_error(
                            "Turn failed",
                            Some(json!({
                                "error": tr::model_error_text(
                                    e,
                                    self.config.trace_include_sensitive_data,
                                )
                            })),
                        );
                    }
                    turn_span.finish();
                    res?
                } else {
                    scope::with_span(agent_span, Box::pin(self.run_turn(tools, handoffs))).await?
                }
            };

            match step {
                NextStep::FinalOutput(output) => {
                    return Box::pin(self.finish_with_output(output)).await;
                }
                NextStep::Interruption(items) => {
                    self.save_session().await?;
                    self.notify_session_completed().await?;
                    return Ok(self.build_result(None, items));
                }
                NextStep::Handoff(agent) => {
                    if let Some(s) = self.agent_span.take() {
                        s.finish();
                    }
                    self.current_agent = agent;
                    self.should_run_agent_start_hooks = true;
                    self.emit(StreamEvent::AgentUpdated {
                        new_agent: self.current_agent.clone(),
                    });
                    self.save_session().await?;
                }
                NextStep::RunAgain => {
                    self.save_session().await?;
                }
            }
        }
    }

    async fn handle_max_turns(&mut self, max: usize) -> Result<RunResult<C>> {
        let err = AgentsError::MaxTurnsExceeded {
            message: format!("Max turns ({max}) exceeded"),
            run_data: None,
        };
        let handler = self.error_handlers.max_turns.clone();
        self.apply_error_handler(handler, err).await
    }

    async fn apply_error_handler(
        &mut self,
        handler: Option<RunErrorHandler<C>>,
        err: AgentsError,
    ) -> Result<RunResult<C>> {
        let Some(h) = handler else {
            return Err(err);
        };
        let input = RunErrorHandlerInput {
            error: Arc::new(clone_simple_error(&err)),
            context: self.ctx.clone(),
            run_data: RunErrorData {
                input: self.original_input.clone(),
                new_items: self.generated_items.clone(),
                history: {
                    let mut h = self.original_input.clone();
                    h.extend(items_to_input(&self.generated_items));
                    h
                },
                raw_responses: self.model_responses.clone(),
                last_agent: self.current_agent.clone(),
            },
        };
        match h(input).await? {
            None => Err(err),
            Some(res) => {
                if let Some(schema) = &self.current_agent.output_type
                    && !schema.is_plain_text()
                {
                    schema.validate_json(&res.final_output.to_string())?;
                }
                if res.include_in_history {
                    let text = match &res.final_output {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let item = RunItem::new(
                        self.current_agent.name.clone(),
                        RunItemKind::MessageOutput {
                            raw_item: MessageOutput::text(text),
                        },
                    );
                    self.push_item(item, Some(RunItemStreamEventName::MessageOutputCreated));
                }
                self.save_session().await?;
                let mut r = self.build_result(Some(res.final_output), Vec::new());
                r.state.final_output = r.final_output.clone();
                Ok(r)
            }
        }
    }

    async fn finish_with_output(&mut self, output: Value) -> Result<RunResult<C>> {
        let agent = self.current_agent.clone();
        let mut guardrails = agent.output_guardrails.clone();
        guardrails.extend(self.config.output_guardrails.iter().cloned());
        if !guardrails.is_empty() {
            let futs = guardrails.iter().map(|g| {
                let ctx = self.ctx.clone();
                let agent = agent.clone();
                let output = output.clone();
                async move {
                    let span = tr::guardrail_span(&g.name, false);
                    span.start();
                    let r = g.run(ctx, agent, output).await;
                    if let Ok(res) = &r {
                        let tripped = res.output.tripwire_triggered;
                        span.update_data(|d| {
                            if let SpanData::Guardrail { triggered, .. } = d {
                                *triggered = tripped;
                            }
                        });
                    }
                    span.finish();
                    r
                }
            });
            let results = futures::future::join_all(futs).await;
            let batch_start = self.output_guardrail_results.len();
            for r in results {
                let r = r?;
                self.output_guardrail_results.push(r.clone());
                if r.output.tripwire_triggered {
                    return Err(self.block_output(&agent, r, batch_start).await?);
                }
            }
        }
        if let Some(h) = &self.hooks {
            h.on_agent_end(&self.ctx, &agent, &output).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_end(&self.ctx, &agent, &output).await;
        }
        self.save_session().await?;
        self.notify_session_completed().await?;
        let mut r = self.build_result(Some(output), Vec::new());
        r.state.final_output = r.final_output.clone();
        Ok(r)
    }

    /// Handle an output guardrail tripwire. The blocked final turn is not persisted, except for
    /// tool calls whose terminal output became the final output: those are kept with their
    /// outputs (and the guardrail results) replaced by a data-free placeholder.
    async fn block_output(
        &mut self,
        agent: &Arc<Agent<C>>,
        tripped: OutputGuardrailResult,
        batch_start: usize,
    ) -> Result<AgentsError> {
        let turn_start = self.turn_start.min(self.generated_items.len());
        let terminal_tool_output = self.generated_items[turn_start..]
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }));
        if !terminal_tool_output {
            // Persist earlier turns only.
            let current = self.generated_items.split_off(turn_start);
            self.save_session().await?;
            self.generated_items.extend(current);
            self.session_saved_items = turn_start;
            return Ok(AgentsError::OutputGuardrailTripwireTriggered {
                guardrail_result: Box::new(tripped),
                run_data: None,
            });
        }
        let default = crate::run_config::OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT.to_string();
        let message = match &self.config.output_guardrail_blocked_message {
            None => default.clone(),
            Some(crate::run_config::OutputGuardrailBlockedMessage::Text(t)) if !t.is_empty() => {
                t.clone()
            }
            Some(crate::run_config::OutputGuardrailBlockedMessage::Formatter(f)) => {
                f(&crate::run_config::OutputGuardrailBlockedMessageArgs {
                    default_message: default.clone(),
                    guardrail_name: tripped.guardrail_name.clone(),
                    agent,
                    run_context: &self.ctx,
                })
                .filter(|m| !m.is_empty())
                .unwrap_or(default.clone())
            }
            Some(_) => default.clone(),
        };
        let sanitize = |r: &mut OutputGuardrailResult| {
            r.agent_output = Value::String(message.clone());
            r.output.output_info = Value::Null;
        };
        for r in &mut self.output_guardrail_results[batch_start..] {
            sanitize(r);
        }
        let mut tripped = tripped;
        sanitize(&mut tripped);
        // Tool guardrails saw the blocked payload too; their `output_info` and any rejection
        // message must not keep it. Port of `_data_free_tool_output_guardrail_results`.
        let tool_output_start = self
            .turn_tool_output_guardrail_start
            .min(self.tool_output_guardrail_results.len());
        for r in &mut self.tool_output_guardrail_results[tool_output_start..] {
            r.output.output_info = Value::String(message.clone());
            if let ToolGuardrailBehavior::RejectContent { message: m } = &mut r.output.behavior {
                *m = message.clone();
            }
        }
        // Rebuild the blocked tool items from allowlisted fields instead of overwriting the
        // payload in place, so no part of the rejected output survives in the raw item.
        // Port of `blocked_function_call_payload` / `blocked_function_output_payload`.
        let redact = |items: &mut [RunItem], from: usize| {
            for item in items.iter_mut().skip(from) {
                if let RunItemKind::ToolCallOutput {
                    raw_item, output, ..
                } = &mut item.kind
                {
                    *raw_item = blocked_tool_output_payload(raw_item, &message);
                    *output = Value::String(message.clone());
                }
            }
        };
        redact(&mut self.generated_items, turn_start);
        let model_start = self
            .model_items
            .len()
            .saturating_sub(self.generated_items.len() - turn_start);
        redact(&mut self.model_items, model_start);
        // Drop the model's own text of the blocked turn, keep the tool calls and outputs.
        let current = self.generated_items.split_off(turn_start);
        let retained: Vec<RunItem> = current
            .iter()
            .filter(|i| !matches!(i.kind, RunItemKind::MessageOutput { .. }))
            .cloned()
            .collect();
        self.generated_items.extend(retained);
        self.save_session().await?;
        self.generated_items.truncate(turn_start);
        self.generated_items.extend(current);
        self.session_saved_items = self.generated_items.len();
        Ok(AgentsError::OutputGuardrailTripwireTriggered {
            guardrail_result: Box::new(tripped),
            run_data: None,
        })
    }

    fn build_result(
        &self,
        final_output: Option<Value>,
        interruptions: Vec<ToolApprovalItem>,
    ) -> RunResult<C> {
        let state = RunState {
            schema_version: CURRENT_SCHEMA_VERSION.to_string(),
            current_turn: self.current_turn,
            max_turns: self.max_turns,
            current_agent_name: self.current_agent.name.clone(),
            original_input: self.original_input.clone(),
            generated_items: self.generated_items.clone(),
            model_responses: self.model_responses.clone(),
            usage: self.ctx.usage(),
            approvals: self.ctx.export_approvals(),
            nested_agent_tool_states: self.ctx.nested_states(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
            pending_tool_processing: !interruptions.is_empty(),
            interruptions: interruptions.clone(),
            pending_input: Vec::new(),
            previous_response_id: self.previous_response_id.clone(),
            conversation_id: self.conversation_id.clone(),
            auto_previous_response_id: self.auto_previous_response_id,
            server_sent_items: self.server_sent_items,
            server_input_sent: self.server_input_sent,
            session_saved_items: self.session_saved_items,
            input_saved_to_session: self.input_saved_to_session,
            final_output: None,
            trace_id: self.trace_id.clone(),
            trace_state: self.trace_state.clone(),
            context: None,
            turn_start_index: self.turn_start,
            tool_use_tracker: self
                .tools_used
                .iter()
                .map(|(agent, tools)| {
                    let mut names: Vec<String> = tools.clone();
                    names.sort();
                    names.dedup();
                    (agent.clone(), names)
                })
                .collect(),
            generated_prompt_cache_key: self.prompt_cache_key.generated_key(),
            server_conversation: self.server_conversation.state(),
            // Rust returns an error instead of a snapshot when a terminal save fails, so a
            // snapshot this loop builds is always resumable.
            terminal_unrecoverable: false,
            tool_invocations: tool_invocation_ledger(
                &self.generated_items,
                &self.unbound_approvals,
            ),
            // An approval the user has not renewed stays unbound across the snapshot, so the
            // decision cannot quietly become authoritative again on the next resume.
            restored_unbound_approval_call_ids: self.unbound_approvals.clone(),
            // The application owns its context; a snapshot this loop builds never claims to
            // carry one, so nothing needs restoring.
            context_requires_restoration: false,
            // Python keeps the processed response only while a turn is parked on an
            // approval, and clears it once the run finishes.
            last_processed_response: (!interruptions.is_empty())
                .then(|| self.last_processed_response.clone())
                .flatten()
                .map(|mut p| {
                    // Python's `ProcessedResponse.new_items` holds only the items derived
                    // from the model response itself. The tool outputs and the approval
                    // items the turn appended afterwards are listed separately and indexed
                    // through the ownership range, so `new_items` is the turn's leading run
                    // of model-derived items.
                    p.new_items = self.generated_items
                        [self.turn_start.min(self.generated_items.len())..]
                        .iter()
                        .take_while(|i| is_model_derived_item(&i.kind))
                        .cloned()
                        .collect();
                    p.interruptions = interruptions.clone();
                    p
                }),
        };
        RunResult {
            input: self.original_input.clone(),
            new_items: self.generated_items.clone(),
            raw_responses: self.model_responses.clone(),
            final_output,
            last_agent: self.current_agent.clone(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
            context_wrapper: self.ctx.clone(),
            interruptions,
            agent_tool_invocation: self.agent_tool_invocation.clone(),
            model_input_items: self.model_items.clone(),
            replay_from_model_input_items: self.model_history_diverged,
            state,
        }
    }

    async fn save_session(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let mut items = Vec::new();
        if !self.input_saved_to_session {
            items.extend(self.session_input.clone());
        }
        let new: Vec<InputItem> = items_to_input(&self.generated_items[self.session_saved_items..]);
        items.extend(new);
        let items = items::deduplicate_input_items_preferring_latest(items);
        if !items.is_empty() {
            // A cancelled run (a dropped `select!` branch, an elapsed timeout) must not abort
            // a half-written session write. Port of `await_session_mutation`.
            crate::memory::await_session_mutation(async move { session.add_items(items).await })
                .await?;
        }
        self.input_saved_to_session = true;
        self.session_saved_items = self.generated_items.len();
        Ok(())
    }

    /// Notify the session that the run finished (used for automatic compaction).
    async fn notify_session_completed(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let has_local_tool_outputs = self.generated_items
            [self.turn_start.min(self.generated_items.len())..]
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }));
        // Everything the model was sent plus everything it produced this run. A compacting
        // session matches the stored tail against these digests, so it only ever replaces
        // history the model actually saw. Both spellings of a reasoning item are recorded,
        // because the stored copy may keep an id the replayed input omitted. Only sessions that
        // compact ask for this, so an ordinary session pays nothing for it.
        let mut model_exchange = Vec::new();
        if session.wants_model_exchange() {
            let policy = self.config.reasoning_item_id_policy;
            // A store that owns its history assigns its own item ids, so matching on them
            // would never find the replayed item. Port of `_ignore_ids_for_matching`.
            let ignore_ids = session.manages_history_remotely();
            for item in self
                .model_prefix
                .iter()
                .cloned()
                .chain(items_to_input(&self.model_items))
            {
                let item = if ignore_ids {
                    strip_item_id(item)
                } else {
                    item
                };
                let stripped = crate::memory::compaction_item_digest(&strip_reasoning_id(
                    item.clone(),
                    policy,
                ));
                let digest = crate::memory::compaction_item_digest(&item);
                if stripped != digest {
                    model_exchange.push(stripped);
                }
                model_exchange.push(digest);
            }
        }
        let info = crate::memory::SessionRunCompletion {
            response_id: self
                .model_responses
                .last()
                .and_then(|r| r.response_id.clone()),
            store: self.current_agent.model_settings.store,
            has_local_tool_outputs,
            model_exchange,
        };
        // Compaction replaces a stored suffix, so a dropped caller must not leave the store
        // half-rewritten. Port of `await_session_mutation`.
        let usage =
            crate::memory::await_session_mutation(
                async move { session.on_run_completed(&info).await },
            )
            .await?;
        if usage.requests > 0 || usage.total_tokens > 0 {
            self.ctx.add_usage(&usage);
        }
        Ok(())
    }

    async fn load_session_history(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let defaults = session.session_settings().unwrap_or_default();
        let settings = defaults.resolve(self.config.session_settings.as_ref());
        let history = session.get_items(settings.limit).await?;
        self.model_prefix = match &self.config.session_input_callback {
            Some(cb) => cb(history, self.session_input.clone()).await?,
            None => {
                let mut h = history;
                h.extend(self.session_input.clone());
                h
            }
        };
        Ok(())
    }

    fn resolve_model(&self, agent: &Agent<C>) -> Result<Arc<dyn Model>> {
        let choice = self.config.model.as_ref().or(agent.model.as_ref());
        match choice {
            Some(AgentModel::Instance(m)) => Ok(m.clone()),
            Some(AgentModel::Name(n)) => self.config.model_provider.get_model(Some(n)),
            None => self.config.model_provider.get_model(None),
        }
    }

    fn resolve_settings(&mut self, agent: &Agent<C>, model: &Arc<dyn Model>) -> ModelSettings {
        let base = if agent.model_settings_explicit {
            agent.model_settings.clone()
        } else {
            model
                .default_model_settings()
                .map(|d| d.resolve(Some(&agent.model_settings)))
                .unwrap_or_else(|| agent.model_settings.clone())
        };
        let mut s = base.resolve(self.config.model_settings.as_ref());
        if agent.reset_tool_choice
            && self
                .tools_used
                .get(&agent.name)
                .map(|v| !v.is_empty())
                .unwrap_or(false)
        {
            s.tool_choice = None;
        }
        // One generated key per run, grouped by conversation, session, trace group or run.
        // Port of `PromptCacheKeyResolver` + `model_settings_with_prompt_cache_key`.
        let key = self.prompt_cache_key.resolve(
            &s,
            model,
            self.conversation_id.as_deref(),
            self.session.as_ref(),
            self.config.group_id.as_deref(),
        );
        prompt_cache_key::apply_prompt_cache_key(&mut s, key.as_deref());
        s
    }

    fn build_model_input(&mut self) -> Vec<InputItem> {
        let policy = self.config.reasoning_item_id_policy;
        // Orphan pruning applies only to runner-generated history: a caller who passes an
        // unanswered call in the run input gets it back verbatim, like Python.
        if self.server_managed() {
            // Deltas are chosen by server identity and content fingerprint, not by an index
            // cursor, so a retry or a resumed state cannot resend or skip an item.
            let prefix = self.model_prefix.clone();
            let mut input = self.server_conversation.take_initial_input(&prefix);
            // Decide on the raw item: stripping the reasoning id first would hide the
            // server identity the tracker matches on.
            let generated: Vec<InputItem> = items_to_input(&self.model_items)
                .into_iter()
                .filter(|i| self.server_conversation.needs_send(i))
                .map(|i| strip_reasoning_id(i, policy))
                .collect();
            input.extend(items::drop_orphan_function_calls(generated));
            input
        } else {
            let mut input = self.model_prefix.clone();
            let generated: Vec<InputItem> = items_to_input(&self.model_items)
                .into_iter()
                .map(|i| strip_reasoning_id(i, policy))
                .collect();
            input.extend(items::drop_orphan_function_calls(generated));
            input
        }
    }

    async fn run_input_guardrails(
        &self,
        guardrails: Vec<InputGuardrail<C>>,
        input: Vec<InputItem>,
    ) -> Result<Vec<InputGuardrailResult>> {
        let agent = self.current_agent.clone();
        let futs = guardrails.into_iter().map(|g| {
            let ctx = self.ctx.clone();
            let agent = agent.clone();
            let input = input.clone();
            async move {
                let span = tr::guardrail_span(&g.name, false);
                span.start();
                let r = g.run(ctx, agent, input).await;
                if let Ok(res) = &r {
                    let tripped = res.output.tripwire_triggered;
                    span.update_data(|d| {
                        if let SpanData::Guardrail { triggered, .. } = d {
                            *triggered = tripped;
                        }
                    });
                }
                span.finish();
                r
            }
        });
        futures::future::join_all(futs).await.into_iter().collect()
    }

    fn record_input_guardrails(&mut self, results: Vec<InputGuardrailResult>) -> Result<()> {
        let mut tripped = None;
        for r in results {
            if r.output.tripwire_triggered && tripped.is_none() {
                tripped = Some(r.clone());
            }
            self.input_guardrail_results.push(r);
        }
        match tripped {
            Some(r) => Err(AgentsError::InputGuardrailTripwireTriggered {
                guardrail_result: Box::new(r),
                run_data: None,
            }),
            None => Ok(()),
        }
    }

    async fn run_turn(
        &mut self,
        tools: Vec<Tool<C>>,
        handoffs: Vec<Handoff<C>>,
    ) -> Result<NextStep<C>> {
        let agent = self.current_agent.clone();
        if self.should_run_agent_start_hooks {
            self.should_run_agent_start_hooks = false;
            if let Some(h) = &self.hooks {
                h.on_agent_start(&self.ctx, &agent).await;
            }
            if let Some(h) = &agent.hooks {
                h.on_start(&self.ctx, &agent).await;
            }
        }

        self.turn_start = self.generated_items.len();
        self.turn_tool_input_guardrail_start = self.tool_input_guardrail_results.len();
        self.turn_tool_output_guardrail_start = self.tool_output_guardrail_results.len();
        let first_turn = self.current_turn == 1 && !self.is_resumed;
        if first_turn {
            self.load_session_history().await?;
        }

        // Input guardrails apply to the first turn and to staged input on resume.
        let mut guardrail_input: Option<Vec<InputItem>> = None;
        if first_turn {
            guardrail_input = Some(self.original_input.clone());
        }
        if !self.pending_input.is_empty() {
            let staged = std::mem::take(&mut self.pending_input);
            guardrail_input = Some(staged.clone());
            for item in staged {
                let ri = RunItem::new(agent.name.clone(), RunItemKind::Input { raw_item: item });
                self.push_item(ri, None);
            }
        }
        let (blocking, parallel): (Vec<_>, Vec<_>) = if guardrail_input.is_some() {
            let mut all = self.config.input_guardrails.clone();
            all.extend(agent.input_guardrails.iter().cloned());
            all.into_iter().partition(|g| !g.run_in_parallel)
        } else {
            (Vec::new(), Vec::new())
        };
        if !blocking.is_empty() {
            let results = self
                .run_input_guardrails(blocking, guardrail_input.clone().unwrap_or_default())
                .await?;
            self.record_input_guardrails(results)?;
        }

        let system_prompt = agent.get_system_prompt(&self.ctx).await?;
        let prompt = agent.get_prompt(&self.ctx).await?;
        let model = self.resolve_model(&agent)?;
        if !self.models_used.iter().any(|m| Arc::ptr_eq(m, &model)) {
            self.models_used.push(model.clone());
        }
        let settings = self.resolve_settings(&agent, &model);

        let mut input = self.build_model_input();
        let mut instructions = system_prompt;
        if let Some(f) = &self.config.call_model_input_filter {
            let r = f(CallModelData {
                model_data: ModelInputData {
                    input,
                    instructions,
                },
                agent: agent.clone(),
                context: self.ctx.clone(),
            })
            .await?;
            input = r.input;
            instructions = r.instructions;
        }
        input = items::deduplicate_input_items_preferring_latest(input);
        // Hooks and tools see exactly what the model is about to see.
        self.ctx.set_turn_input(input.clone());
        let sent_input = input.clone();

        let mut tool_specs: Vec<ToolSpec> = Vec::with_capacity(tools.len());
        for t in &tools {
            tool_specs.push(match t {
                Tool::Function(f) => ToolSpec::Function(f.to_spec()),
                Tool::Computer(c) => ToolSpec::Hosted {
                    param: c.to_responses_param().await?,
                    includes: Vec::new(),
                },
                other => {
                    let (param, includes) = other.to_responses_param();
                    ToolSpec::Hosted { param, includes }
                }
            });
        }
        let tracing_mode = if self.config.tracing_disabled {
            ModelTracing::Disabled
        } else if self.config.trace_include_sensitive_data {
            ModelTracing::Enabled
        } else {
            ModelTracing::EnabledWithoutData
        };
        let previous_response_id =
            if self.auto_previous_response_id || self.previous_response_id.is_some() {
                self.previous_response_id.clone()
            } else {
                None
            };
        let request = ModelRequest {
            system_instructions: instructions.clone(),
            input: input.clone(),
            model_settings: settings,
            tools: tool_specs,
            output_schema: agent.output_type.clone(),
            handoffs: handoffs.iter().map(Handoff::to_spec).collect(),
            tracing: tracing_mode,
            previous_response_id,
            conversation_id: self.conversation_id.clone(),
            prompt,
        };

        if let Some(h) = &self.hooks {
            h.on_llm_start(&self.ctx, &agent, instructions.as_deref(), &input)
                .await;
        }
        if let Some(h) = &agent.hooks {
            h.on_llm_start(&self.ctx, &agent, instructions.as_deref(), &input)
                .await;
        }

        // Every model call of this run runs under the run's owner id, so a model that keeps
        // per-run state can tell this run's calls from another's. Port of
        // `model_run_context` / `model_run_context_stream`.
        let owner = self.model_owner;
        let response = if parallel.is_empty() {
            Box::pin(crate::models::run_context::model_run_context(
                owner,
                model_call::call_model(&model, request, self.sink.as_ref()),
            ))
            .await?
        } else {
            let g_input = guardrail_input.clone().unwrap_or_default();
            let (guard_results, model_result) = {
                let model_fut = Box::pin(crate::models::run_context::model_run_context(
                    owner,
                    model_call::call_model(&model, request, self.sink.as_ref()),
                ));
                let guard_fut = self.run_input_guardrails(parallel, g_input);
                tokio::pin!(model_fut);
                tokio::pin!(guard_fut);
                tokio::select! {
                    g = &mut guard_fut => {
                        let g = g?;
                        if g.iter().any(|r| r.output.tripwire_triggered) {
                            (g, None)
                        } else {
                            let m = (&mut model_fut).await;
                            (g, Some(m))
                        }
                    }
                    m = &mut model_fut => {
                        let g = (&mut guard_fut).await?;
                        (g, Some(m))
                    }
                }
            };
            self.record_input_guardrails(guard_results)?;
            model_result.expect("model result")?
        };

        self.ctx.add_usage(&response.usage);
        if let Some(h) = &self.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        self.model_responses.push(response.clone());

        let processed =
            self.process_response(&agent, &tools, &handoffs, &response, true, &HashSet::new())?;
        if self.server_managed() {
            self.server_input_sent = true;
            if (self.auto_previous_response_id || self.previous_response_id.is_some())
                && let Some(id) = &response.response_id
            {
                self.previous_response_id = Some(id.clone());
            }
            self.server_conversation.mark_input_as_sent(&sent_input);
            self.server_conversation.track_server_items(&response);
            self.server_sent_items = self.model_items.len();
        }
        Box::pin(self.execute_side_effects(&agent, processed)).await
    }

    /// Re-process the last model response after approvals were resolved.
    async fn resume_tools(
        &mut self,
        tools: Vec<Tool<C>>,
        handoffs: Vec<Handoff<C>>,
    ) -> Result<NextStep<C>> {
        let agent = self.current_agent.clone();
        let Some(response) = self.model_responses.last().cloned() else {
            return Ok(NextStep::RunAgain);
        };
        let done: HashSet<String> = self.generated_items.iter().filter_map(call_id_of).collect();
        let processed =
            self.process_response(&agent, &tools, &handoffs, &response, false, &done)?;
        Box::pin(self.execute_side_effects(&agent, processed)).await
    }

    /// Classify the output of a model response. When `append` is true the items are added to
    /// the run (and streamed); `skip` lists call ids that already have outputs.
    fn process_response(
        &mut self,
        agent: &Arc<Agent<C>>,
        tools: &[Tool<C>],
        handoffs: &[Handoff<C>],
        response: &ModelResponse,
        append: bool,
        skip: &HashSet<String>,
    ) -> Result<ProcessedResponse<C>> {
        let mut processed = ProcessedResponse {
            functions: Vec::new(),
            handoffs: Vec::new(),
            mcp_approvals: Vec::new(),
            local_calls: Vec::new(),
            tools_used: Vec::new(),
            last_message: None,
            has_tool_activity: false,
            call_positions: HashMap::new(),
        };
        let function_map: HashMap<String, &Arc<FunctionTool<C>>> = tools
            .iter()
            .filter_map(|t| t.as_function().map(|f| (f.qualified_name(), f)))
            .collect();
        let handoff_map: HashMap<&str, &Handoff<C>> =
            handoffs.iter().map(|h| (h.tool_name.as_str(), h)).collect();
        let hosted_mcp: Vec<&HostedMCPTool<C>> = tools
            .iter()
            .filter_map(|t| match t {
                Tool::HostedMCP(m) => Some(m),
                _ => None,
            })
            .collect();
        let name = agent.name.clone();
        let programmatic_tool_present = tools
            .iter()
            .any(|t| matches!(t, Tool::ProgrammaticToolCalling(_)));
        let server_managed = self.server_managed();
        // Pre-step items only: this response's own program items are collected as the loop
        // walks its output, so including them here would reject their own children.
        let pre_step = self.turn_start.min(self.generated_items.len());
        let history_raw: Vec<Value> = self.generated_items[..pre_step]
            .iter()
            .map(|i| i.to_input_item().0)
            .collect();
        let (mut program_call_ids, mut completed_program_call_ids) =
            tool_caller::collect_program_parent_state(&history_raw, server_managed);
        let response_raw: Vec<Value> = response.output.iter().map(OutputItem::to_value).collect();
        let (_, response_completed_program_call_ids) =
            tool_caller::collect_program_parent_state(&response_raw, false);
        // One response must never schedule the same tool call twice. Port of
        // `_dedupe_processed_response_invocations` / `_dedupe_tool_call_items`.
        let mut seen_invocations: HashSet<(String, String)> = HashSet::new();

        for out in &response.output {
            let raw_out = out.to_value();
            let out_type = out.item_type();
            // One call id must name one invocation for the whole run. Port of
            // `RunContextWrapper._tool_invocation_status`'s binding step.
            let role = (out_type == "function_call"
                && handoff_map.contains_key(
                    raw_out
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ))
            .then_some(tool_invocation::HANDOFF_ROLE);
            if let Some(id) = tool_invocation::tool_invocation_identity(&raw_out, role) {
                match self.tool_invocations.get(&id.call_id) {
                    Some(record)
                        if record.invocation_type != id.invocation_type
                            || record.approval_scope != id.approval_scope
                            || record.fingerprint != id.fingerprint =>
                    {
                        return Err(AgentsError::model_behavior(
                            "Model reused a tool call ID for a different invocation. \
                             Use a unique call ID for each tool invocation.",
                        ));
                    }
                    Some(_) => {}
                    None if self.unbound_approvals.contains(&id.call_id) => {}
                    None => {
                        self.tool_invocations.insert(
                            id.call_id.clone(),
                            ToolInvocationRecord {
                                invocation_type: id.invocation_type,
                                approval_scope: id.approval_scope,
                                fingerprint: id.fingerprint,
                                executed: false,
                                completed: false,
                            },
                        );
                    }
                }
            }
            let is_program_owned = raw_out
                .get("caller")
                .and_then(|c| c.get("type"))
                .and_then(Value::as_str)
                == Some("program");
            if (out_type == "program" || out_type == "program_output" || is_program_owned)
                && !programmatic_tool_present
            {
                processed
                    .tools_used
                    .push("programmatic_tool_calling".into());
                self.tools_used
                    .entry(agent.name.clone())
                    .or_default()
                    .extend(processed.tools_used.iter().cloned());
                return Err(AgentsError::model_behavior(format!(
                    "Model produced {out_type} item without a programmatic_tool_calling tool."
                )));
            }
            if is_program_owned {
                let mut completed = completed_program_call_ids.clone();
                completed.extend(response_completed_program_call_ids.iter().cloned());
                tool_caller::ensure_programmatic_tool_call_parent(
                    &raw_out,
                    programmatic_tool_present,
                    &program_call_ids,
                    &completed,
                    &agent.name,
                )?;
            }
            match out {
                OutputItem::Message(m) => {
                    processed.last_message = Some(m.clone());
                    if append {
                        let item = RunItem::new(
                            name.clone(),
                            RunItemKind::MessageOutput {
                                raw_item: m.clone(),
                            },
                        );
                        self.push_item(item, Some(RunItemStreamEventName::MessageOutputCreated));
                    }
                }
                OutputItem::Reasoning(r) => {
                    if append {
                        let item = RunItem::new(
                            name.clone(),
                            RunItemKind::Reasoning {
                                raw_item: r.clone(),
                            },
                        );
                        self.push_item(item, Some(RunItemStreamEventName::ReasoningItemCreated));
                    }
                }
                OutputItem::FunctionCall(fc) => {
                    processed.has_tool_activity = true;
                    if let Some(h) = handoff_map.get(fc.name.as_str()) {
                        if !seen_invocations
                            .insert(("handoff_call".to_string(), fc.call_id.clone()))
                        {
                            continue;
                        }
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::HandoffCall {
                                    raw_item: fc.clone(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::HandoffRequested));
                        }
                        if !skip.contains(&fc.call_id) {
                            processed.handoffs.push((fc.clone(), (*h).clone()));
                        }
                    } else if let Some(t) = function_map.get(&function_lookup_name(fc)) {
                        if !seen_invocations
                            .insert(("function_call".to_string(), fc.call_id.clone()))
                        {
                            continue;
                        }
                        tool_caller::ensure_tool_caller_allowed(
                            &raw_out,
                            t.allowed_callers.as_deref(),
                            &t.name,
                            &agent.name,
                        )?;
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCall {
                                    raw_item: fc.to_value(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed.tools_used.push(fc.name.clone());
                        let next = processed.call_positions.len();
                        processed.call_positions.insert(fc.call_id.clone(), next);
                        if !skip.contains(&fc.call_id) {
                            processed.functions.push((fc.clone(), (*t).clone()));
                        }
                    } else {
                        match self.config.tool_not_found_behavior {
                            ToolNotFoundBehavior::RaiseError => {
                                return Err(AgentsError::model_behavior(format!(
                                    "Tool {} not found in agent {}",
                                    fc.name, agent.name
                                )));
                            }
                            ToolNotFoundBehavior::ReturnErrorToModel => {
                                if append && !skip.contains(&fc.call_id) {
                                    let call_item = RunItem::new(
                                        name.clone(),
                                        RunItemKind::ToolCall {
                                            raw_item: fc.to_value(),
                                        },
                                    );
                                    self.push_item(
                                        call_item,
                                        Some(RunItemStreamEventName::ToolCalled),
                                    );
                                    let msg = tools::tool_not_found_message(
                                        &self.config,
                                        &self.ctx,
                                        &fc.name,
                                        &fc.call_id,
                                    );
                                    let out = ToolOutput::Text(msg);
                                    let raw = ItemHelpers::tool_call_output_item(fc, &out);
                                    let item = RunItem::new(
                                        name.clone(),
                                        RunItemKind::ToolCallOutput {
                                            raw_item: raw,
                                            output: out.to_json_value(),
                                            custom_data: None,
                                        },
                                    );
                                    self.push_item(item, Some(RunItemStreamEventName::ToolOutput));
                                }
                            }
                        }
                    }
                }
                OutputItem::Other(v) => {
                    let ty = v.get("type").and_then(Value::as_str).unwrap_or_default();
                    if ty == "program" {
                        let Some(call_id) = v
                            .get("call_id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        else {
                            return Err(AgentsError::model_behavior(
                                "Model produced program item without a valid call_id.",
                            ));
                        };
                        program_call_ids.insert(call_id.to_string());
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCall {
                                    raw_item: v.clone(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed
                            .tools_used
                            .push("programmatic_tool_calling".into());
                        continue;
                    }
                    if ty == "program_output" {
                        let call_id = v
                            .get("call_id")
                            .and_then(Value::as_str)
                            .filter(|s| program_call_ids.contains(*s))
                            .ok_or_else(|| {
                                AgentsError::model_behavior(
                                    "Model produced program_output item that does not match a \
                                     parent program item.",
                                )
                            })?
                            .to_string();
                        if completed_program_call_ids.contains(&call_id) {
                            return Err(AgentsError::model_behavior(
                                "Model produced program_output item whose parent program is \
                                 already completed.",
                            ));
                        }
                        let status = v.get("status").and_then(Value::as_str).unwrap_or_default();
                        if status != "completed" && status != "incomplete" {
                            return Err(AgentsError::model_behavior(
                                "Model produced program_output item without a valid status.",
                            ));
                        }
                        let Some(result) = v.get("result").and_then(Value::as_str) else {
                            return Err(AgentsError::model_behavior(
                                "Model produced program_output item without a string result.",
                            ));
                        };
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCallOutput {
                                    raw_item: v.clone(),
                                    output: Value::String(result.to_string()),
                                    custom_data: None,
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolOutput));
                        }
                        if status == "completed" {
                            completed_program_call_ids.insert(call_id);
                        }
                        processed
                            .tools_used
                            .push("programmatic_tool_calling".into());
                        continue;
                    }
                    if ty == "compaction" {
                        if append {
                            let mut raw = v.clone();
                            if let Value::Object(m) = &mut raw {
                                m.remove("created_by");
                            }
                            // Python emits no stream event for compaction items.
                            self.push_item(
                                RunItem::new(
                                    name.clone(),
                                    RunItemKind::Compaction { raw_item: raw },
                                ),
                                None,
                            );
                        }
                        continue;
                    }
                    if ty == "tool_search_call" || ty == "tool_search_output" {
                        tool_caller::ensure_tool_caller_allowed(
                            v,
                            None,
                            "tool_search",
                            &agent.name,
                        )?;
                        if ty == "tool_search_call"
                            && v.get("execution").and_then(Value::as_str) == Some("client")
                        {
                            return Err(AgentsError::model_behavior(
                                "Client-executed tool_search calls are not supported by the \
                                 standard agent runner. Handle the tool_search_call yourself \
                                 and return a matching tool_search_output item with the same \
                                 call_id.",
                            ));
                        }
                    }
                    if let Some(local) = match_local_call(tools, v, ty, &agent.name)? {
                        if !seen_invocations.insert((ty.to_string(), local_tools::call_id_of(v))) {
                            continue;
                        }
                        processed.has_tool_activity = true;
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCall {
                                    raw_item: v.clone(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed.tools_used.push(local.tool_name());
                        let next = processed.call_positions.len();
                        processed
                            .call_positions
                            .insert(local_tools::call_id_of(v), next);
                        if !skip.contains(&local_tools::call_id_of(v)) {
                            processed.local_calls.push(local);
                        }
                        continue;
                    }
                    if ty.ends_with("_call_output") {
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCallOutput {
                                    raw_item: v.clone(),
                                    output: v.get("output").cloned().unwrap_or(Value::Null),
                                    custom_data: None,
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolOutput));
                        }
                        continue;
                    }
                    match ty {
                        "mcp_list_tools" => {
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::MCPListTools {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(item, Some(RunItemStreamEventName::MCPListTools));
                            }
                        }
                        "mcp_approval_request" => {
                            processed.has_tool_activity = true;
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::MCPApprovalRequest {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(
                                    item,
                                    Some(RunItemStreamEventName::MCPApprovalRequested),
                                );
                            }
                            let id = v.get("id").and_then(Value::as_str).unwrap_or_default();
                            if !skip.contains(id) {
                                let label = v.get("server_label").and_then(Value::as_str);
                                let tool = hosted_mcp
                                    .iter()
                                    .find(|t| t.server_label() == label)
                                    .map(|t| (*t).clone());
                                processed.mcp_approvals.push((v.clone(), tool));
                            }
                        }
                        "tool_search_output" => {
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::ToolCallOutput {
                                        raw_item: v.clone(),
                                        output: Value::Null,
                                        custom_data: None,
                                    },
                                );
                                self.push_item(
                                    item,
                                    Some(RunItemStreamEventName::ToolSearchOutputCreated),
                                );
                            }
                        }
                        _ => {
                            if append {
                                let event = if ty == "tool_search_call" {
                                    RunItemStreamEventName::ToolSearchCalled
                                } else {
                                    RunItemStreamEventName::ToolCalled
                                };
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::ToolCall {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(item, Some(event));
                            }
                            // `tool_search_call` is deliberately excluded from tool-use
                            // tracking so it never resets `tool_choice`.
                            if ty.ends_with("_call") && ty != "tool_search_call" {
                                processed.tools_used.push(ty.to_string());
                            }
                        }
                    }
                }
            }
        }
        if !processed.tools_used.is_empty() {
            self.tools_used
                .entry(agent.name.clone())
                .or_default()
                .extend(processed.tools_used.iter().cloned());
        }
        Ok(processed)
    }

    async fn execute_side_effects(
        &mut self,
        agent: &Arc<Agent<C>>,
        mut processed: ProcessedResponse<C>,
    ) -> Result<NextStep<C>> {
        self.last_processed_response = Some(processed.snapshot());
        let mut interruptions: Vec<ToolApprovalItem> = Vec::new();
        let ran_functions = !processed.functions.is_empty();
        let outputs_start = self.generated_items.len();
        let model_outputs_start = self.model_items.len();
        let call_positions = std::mem::take(&mut processed.call_positions);

        // Function tools.
        let results = {
            let env = ToolEnv {
                ctx: &self.ctx,
                agent,
                hooks: self.hooks.as_ref(),
                config: &self.config,
                unbound_approvals: &self.unbound_approvals,
            };
            Box::pin(tools::execute_function_tools(&env, processed.functions)).await?
        };
        let mut function_results = Vec::new();
        for r in results {
            self.tool_input_guardrail_results
                .extend(r.input_guardrail_results);
            self.tool_output_guardrail_results
                .extend(r.output_guardrail_results);
            let custom_data = r.custom_data.clone();
            match r.outcome {
                tools::ToolOutcome::Output { output, item } => {
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolCallOutput {
                            raw_item: item,
                            output: output.to_json_value(),
                            custom_data: custom_data.clone(),
                        },
                    );
                    self.push_item(ri.clone(), Some(RunItemStreamEventName::ToolOutput));
                    function_results.push(crate::tool::FunctionToolResult {
                        tool_name: r.tool_name,
                        qualified_tool_name: r.qualified_tool_name,
                        custom_data: r.custom_data,
                        call_id: r.call.call_id.clone(),
                        output,
                        run_item: Some(ri),
                        interruptions: Vec::new(),
                    });
                }
                tools::ToolOutcome::Interruptions(items) => {
                    for it in &items {
                        let ri = RunItem::new(
                            it.agent.clone(),
                            RunItemKind::ToolApproval {
                                raw_item: it.raw_item.clone(),
                                tool_name: it.tool_name.clone(),
                            },
                        );
                        self.generated_items.push(ri);
                    }
                    interruptions.extend(items);
                }
            }
        }

        // Hosted MCP approval requests.
        let mut answered_mcp = false;
        for (request, tool) in processed.mcp_approvals {
            let request_id = request
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let tool_name = request
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let label = request
                .get("server_label")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let identity = format!("{label}:{tool_name}");
            let decision = match tool.as_ref().and_then(|t| t.on_approval_request().cloned()) {
                Some(cb) => {
                    let r = cb(MCPToolApprovalRequest {
                        ctx: self.ctx.clone(),
                        data: request.clone(),
                    })
                    .await;
                    Some((r.approve, r.reason))
                }
                None => tools::bound_approval_status(
                    &self.ctx,
                    &self.unbound_approvals,
                    &agent.name,
                    &identity,
                    &request_id,
                )
                .map(|a| {
                    (
                        a,
                        self.ctx
                            .rejection_message(&agent.name, &identity, &request_id),
                    )
                }),
            };
            match decision {
                Some((approve, reason)) => {
                    let mut resp = json!({
                        "type": "mcp_approval_response",
                        "approval_request_id": request_id,
                        "approve": approve,
                    });
                    if let Some(r) = reason.filter(|_| !approve) {
                        resp["reason"] = Value::String(r);
                    }
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::MCPApprovalResponse { raw_item: resp },
                    );
                    self.push_item(ri, Some(RunItemStreamEventName::MCPApprovalResponse));
                    answered_mcp = true;
                }
                None => {
                    let mut extra = serde_json::Map::new();
                    extra.insert("type".into(), Value::String("mcp_approval_request".into()));
                    extra.insert("server_label".into(), Value::String(label));
                    let item = ToolApprovalItem {
                        agent: agent.name.clone(),
                        raw_item: FunctionCall {
                            id: None,
                            call_id: request_id,
                            name: tool_name,
                            arguments: request
                                .get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}")
                                .to_string(),
                            status: None,
                            extra,
                        },
                        tool_name: identity,
                    };
                    self.generated_items.push(RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolApproval {
                            raw_item: item.raw_item.clone(),
                            tool_name: item.tool_name.clone(),
                        },
                    ));
                    interruptions.push(item);
                }
            }
        }

        // Local runtime tools (computer, shell, apply patch, custom).
        let ran_local = !processed.local_calls.is_empty();
        for call in processed.local_calls {
            let outcome = {
                let env = ToolEnv {
                    ctx: &self.ctx,
                    agent,
                    hooks: self.hooks.as_ref(),
                    config: &self.config,
                    unbound_approvals: &self.unbound_approvals,
                };
                Box::pin(local_tools::execute_local_call(&env, call)).await?
            };
            match outcome {
                local_tools::LocalOutcome::Output {
                    raw_item,
                    output,
                    custom_data,
                } => {
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolCallOutput {
                            raw_item,
                            output,
                            custom_data,
                        },
                    );
                    self.push_item(ri, Some(RunItemStreamEventName::ToolOutput));
                }
                local_tools::LocalOutcome::Interruption(item) => {
                    self.generated_items.push(RunItem::new(
                        item.agent.clone(),
                        RunItemKind::ToolApproval {
                            raw_item: item.raw_item.clone(),
                            tool_name: item.tool_name.clone(),
                        },
                    ));
                    interruptions.push(item);
                }
            }
        }

        // Tool outputs are appended in phase order (function, hosted MCP, local); replay them
        // in the order the model asked for them. Port of `order_current_turn_tool_outputs`.
        items::order_current_turn_tool_outputs(
            &mut self.generated_items,
            outputs_start,
            &call_positions,
        );
        items::order_current_turn_tool_outputs(
            &mut self.model_items,
            model_outputs_start,
            &call_positions,
        );

        if !interruptions.is_empty() {
            return Ok(NextStep::Interruption(interruptions));
        }

        // Handoffs.
        if !processed.handoffs.is_empty() {
            return Box::pin(self.execute_handoffs(agent, processed.handoffs)).await;
        }

        // Tool use behavior.
        if ran_functions && !function_results.is_empty() {
            let decision = match &agent.tool_use_behavior {
                ToolUseBehavior::RunLlmAgain => None,
                ToolUseBehavior::StopOnFirstTool => {
                    Some(function_results[0].output.to_json_value())
                }
                ToolUseBehavior::StopAtTools(s) => function_results
                    .iter()
                    .find(|r| {
                        s.stop_at_tool_names.contains(&r.tool_name)
                            || s.stop_at_tool_names.contains(&r.qualified_tool_name)
                    })
                    .map(|r| r.output.to_json_value()),
                ToolUseBehavior::Custom(f) => {
                    let r = f(self.ctx.clone(), function_results.clone()).await?;
                    if r.is_final_output {
                        Some(r.final_output.unwrap_or(Value::Null))
                    } else {
                        None
                    }
                }
            };
            if let Some(out) = decision {
                return Ok(NextStep::FinalOutput(out));
            }
        }
        if ran_functions || answered_mcp || ran_local {
            return Ok(NextStep::RunAgain);
        }

        // Final output from the last message.
        let Some(message) = processed.last_message else {
            return Ok(NextStep::RunAgain);
        };
        match &agent.output_type {
            Some(schema) if !schema.is_plain_text() => {
                if let Some(refusal) = message.refusal()
                    && message.text_content().is_empty()
                {
                    let err = AgentsError::ModelRefusalError {
                        refusal,
                        run_data: None,
                    };
                    let handler = self.error_handlers.model_refusal.clone();
                    return self.handled_final(handler, err).await;
                }
                let text = message.text_content();
                if text.is_empty() {
                    return Ok(NextStep::RunAgain);
                }
                match schema.validate_json(&text) {
                    Ok(v) => Ok(NextStep::FinalOutput(v)),
                    Err(e) => {
                        let handler = self.error_handlers.invalid_final_output.clone();
                        self.handled_final(handler, e).await
                    }
                }
            }
            _ => {
                let text = message.text_content();
                if !text.is_empty() {
                    Ok(NextStep::FinalOutput(Value::String(text)))
                } else if let Some(r) = message.refusal() {
                    Ok(NextStep::FinalOutput(Value::String(r)))
                } else {
                    Ok(NextStep::RunAgain)
                }
            }
        }
    }

    /// Apply an error handler during turn resolution, mapping a recovery to a final output.
    async fn handled_final(
        &mut self,
        handler: Option<RunErrorHandler<C>>,
        err: AgentsError,
    ) -> Result<NextStep<C>> {
        let Some(h) = handler else {
            return Err(err);
        };
        let input = RunErrorHandlerInput {
            error: Arc::new(clone_simple_error(&err)),
            context: self.ctx.clone(),
            run_data: RunErrorData {
                input: self.original_input.clone(),
                new_items: self.generated_items.clone(),
                history: {
                    let mut h = self.original_input.clone();
                    h.extend(items_to_input(&self.generated_items));
                    h
                },
                raw_responses: self.model_responses.clone(),
                last_agent: self.current_agent.clone(),
            },
        };
        match h(input).await? {
            None => Err(err),
            Some(res) => {
                if let Some(schema) = &self.current_agent.output_type
                    && !schema.is_plain_text()
                {
                    schema.validate_json(&res.final_output.to_string())?;
                }
                if !res.include_in_history {
                    // Drop the rejected message from history.
                    if let Some(pos) = self
                        .generated_items
                        .iter()
                        .rposition(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    {
                        self.generated_items.remove(pos);
                    }
                    if let Some(pos) = self
                        .model_items
                        .iter()
                        .rposition(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    {
                        self.model_items.remove(pos);
                    }
                }
                Ok(NextStep::FinalOutput(res.final_output))
            }
        }
    }

    async fn execute_handoffs(
        &mut self,
        agent: &Arc<Agent<C>>,
        handoffs: Vec<(FunctionCall, Handoff<C>)>,
    ) -> Result<NextStep<C>> {
        let mut iter = handoffs.into_iter();
        let (call, handoff) = iter.next().expect("at least one handoff");
        for (extra_call, _) in iter {
            let raw = InputItem::function_call_output(
                extra_call.call_id.clone(),
                "Multiple handoffs detected, ignoring this one.",
            );
            let ri = RunItem::new(
                agent.name.clone(),
                RunItemKind::ToolCallOutput {
                    raw_item: raw.0,
                    output: Value::String("Multiple handoffs detected, ignoring this one.".into()),
                    custom_data: None,
                },
            );
            self.push_item(ri, Some(RunItemStreamEventName::ToolOutput));
        }

        let span = tr::handoff_span(&agent.name, None);
        span.start();
        let new_agent = match scope::with_span(
            span.clone(),
            handoff.invoke(self.ctx.clone(), call.arguments.clone()),
        )
        .await
        {
            Ok(a) => a,
            Err(e) => {
                // A failed invocation is not a transfer: the recovery formatter turns it into
                // a tool output for the calling agent and the run continues. Without one the
                // error ends the run. Port of `execute_handoffs`' failure branch.
                let Some(format_error) = handoff.failure_error_function.clone() else {
                    span.set_error(
                        "Error invoking handoff",
                        Some(json!({
                            "error": tr::model_error_text(
                                &e,
                                self.config.trace_include_sensitive_data,
                            )
                        })),
                    );
                    span.finish();
                    return Err(e);
                };
                span.set_error("Handoff invocation failed", Some(json!({})));
                span.finish();
                let text = format_error(&self.ctx, &e);
                let item = RunItem::new(
                    agent.name.clone(),
                    RunItemKind::ToolCallOutput {
                        raw_item: InputItem::function_call_output(
                            call.call_id.clone(),
                            text.clone(),
                        )
                        .0,
                        output: Value::String(text),
                        custom_data: None,
                    },
                );
                self.push_item(item, Some(RunItemStreamEventName::ToolOutput));
                return Ok(NextStep::RunAgain);
            }
        };
        span.update_data(|d| {
            if let SpanData::Handoff { to_agent, .. } = d {
                *to_agent = Some(new_agent.name.clone());
            }
        });
        span.finish();

        let pre_handoff_count = self.turn_start.min(self.generated_items.len());
        let output_item = RunItem::new(
            agent.name.clone(),
            RunItemKind::HandoffOutput {
                raw_item: InputItem::function_call_output(
                    call.call_id.clone(),
                    Handoff::<C>::get_transfer_message(&new_agent.name),
                ),
                source_agent: agent.name.clone(),
                target_agent: new_agent.name.clone(),
            },
        );
        self.push_item(output_item, Some(RunItemStreamEventName::HandoffOccured));

        if let Some(h) = &self.hooks {
            h.on_handoff(&self.ctx, agent, &new_agent).await;
        }
        if let Some(h) = &new_agent.hooks {
            h.on_handoff(&self.ctx, &new_agent, agent).await;
        }

        let filter = handoff
            .input_filter
            .clone()
            .or_else(|| self.config.handoff_input_filter.clone());
        if let Some(filter) = filter {
            if self.server_managed() {
                return Err(AgentsError::user(
                    "Handoff input filters are not supported with server-managed conversations.",
                ));
            }
            let data = HandoffInputData {
                input_history: self.model_prefix.clone(),
                pre_handoff_items: self.generated_items[..pre_handoff_count].to_vec(),
                new_items: self.generated_items[pre_handoff_count..].to_vec(),
                input_items: None,
            };
            let filtered = filter(data);
            self.original_input = filtered.input_history.clone();
            self.model_prefix = filtered.input_history;
            let mut generated = filtered.pre_handoff_items.clone();
            generated.extend(filtered.new_items.clone());
            let mut model_items = filtered.pre_handoff_items;
            model_items.extend(filtered.input_items.unwrap_or(filtered.new_items));
            self.session_saved_items = self.session_saved_items.min(generated.len());
            self.generated_items = generated;
            self.model_items = model_items;
            self.model_history_diverged = true;
        } else if handoff
            .nest_handoff_history
            .unwrap_or(self.config.nest_handoff_history)
        {
            if self.server_managed() {
                // The server owns the transcript, so a nested rewrite cannot be applied; the
                // next agent gets delta-only input. Port of
                // `_resolve_server_managed_handoff_behavior`'s warning.
                log::warn!(
                    target: crate::logger::LOG_TARGET,
                    "Server-managed conversations do not support nest_handoff_history for \
                     handoff {} -> {}. Disabling nested handoff history and continuing with \
                     delta-only input.",
                    agent.name,
                    new_agent.name,
                );
            } else {
                // Model history can be shorter than the public history after an earlier
                // filter, so anchor the pre-handoff split at its tail.
                let model_pre_count = self
                    .model_items
                    .len()
                    .saturating_sub(self.generated_items.len() - pre_handoff_count);
                let normalized =
                    history::normalize_handoff_items(&self.model_items, model_pre_count);
                self.model_prefix = match &self.config.handoff_history_mapper {
                    Some(m) => {
                        let mut transcript = self.model_prefix.clone();
                        transcript.extend(normalized.into_iter().map(|(item, _)| item));
                        m(transcript)
                    }
                    None => history::build_ordered_history(self.model_prefix.clone(), normalized),
                };
                self.model_items.clear();
                self.server_sent_items = 0;
            }
        }
        Ok(NextStep::Handoff(new_agent))
    }
}
