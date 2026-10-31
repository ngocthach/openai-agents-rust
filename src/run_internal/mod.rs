//! The agent loop. Port of `agents.run_internal`.

pub(crate) mod agent_identity;
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
    /// Consumer acknowledgement and the immediate-cancel signal.
    pub control: Arc<crate::result::StreamControl>,
}

impl<C> EventSink<C> {
    pub(crate) fn send(&self, ev: StreamEvent<C>) {
        self.control.record_sent();
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
    /// Function calls of this response whose outputs already exist (a resumed turn), with
    /// their tool.
    completed_functions: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
    handoffs: Vec<(FunctionCall, Handoff<C>)>,
    mcp_approvals: Vec<(Value, Option<HostedMCPTool<C>>)>,
    local_calls: Vec<local_tools::LocalCall<C>>,
    tools_used: Vec<String>,
    last_message: Option<MessageOutput>,
    has_tool_activity: bool,
    /// Position of each tool call id in the model response, used to order the outputs.
    call_positions: HashMap<String, usize>,
    /// Pending approvals of a resumed turn whose tool the agent no longer has: they stay
    /// pending. Port of the unmatched-approval pass of `_rebuild_function_runs_from_approvals`.
    unmatched_pending: Vec<ToolApprovalItem>,
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
                local_tools::LocalCall::Custom(raw, tool) => {
                    // Python writes the tool description of every action group; a custom
                    // tool is the only local tool with one.
                    let mut run = local_run(raw, name);
                    run.description = Some(tool.description.clone());
                    snapshot.custom_tool_actions.push(run)
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
    tools_used: ToolUseTracker<C>,
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
    /// Call ids still awaiting a decision in the snapshot this run resumed. Within the resumed
    /// turn an undecided one stays pending without consulting its approval rule again.
    /// Mirrors `_function_requires_approval`'s `approval_items_by_call_id` short-circuit.
    resumed_pending_approvals: BTreeSet<String>,
    /// The approval items behind `resumed_pending_approvals`.
    resumed_interruptions: Vec<ToolApprovalItem>,
    /// The run config's reasoning-id policy, or the one a resumed snapshot recorded when the
    /// config keeps the default. Mirrors `resolved_reasoning_item_id_policy`.
    reasoning_item_id_policy: ReasoningItemIdPolicy,
    /// The opaque sandbox payload a Python snapshot carried, written back unchanged.
    restored_sandbox: Option<Value>,
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
    /// Indexes into `generated_items` of items a nested handoff forwarded verbatim into the
    /// rewritten input.
    nested_history_owned_items: Vec<usize>,
    /// Where model history starts in `generated_items` after a nested handoff.
    nested_model_start: Option<usize>,
    /// The input of the latest model call as the model saw it (after `call_model_input_filter`)
    /// and the items it answered with. Compaction only replaces history this exchange covers.
    /// Port of `_session_compaction_model_exchange`.
    compaction_exchange: Option<(Vec<InputItem>, Vec<InputItem>)>,
    /// The record of a session write an interruption withheld (see
    /// `RunState::pending_session_write`). While it stands, the unsaved tail of
    /// `generated_items` is a held batch that settles at the next gate-legal exit.
    pending_write: Option<Value>,
    /// Set while a turn is executing the tools of a processed model response, so a failure
    /// there checkpoints as a turn that resumes by re-processing that response.
    turn_in_flight: bool,
    /// Set when the selected tool failure came after its own output was committed; such a turn
    /// keeps `turn_in_flight`, so a retry re-processes the stored response instead of calling the
    /// model again.
    failed_after_output: bool,
    /// Where a failed run leaves its checkpoint, when the caller asked for one.
    checkpoint: Option<crate::run::RunCheckpoint>,
    /// Prepares the starting agent once the first turn's blocking input guardrails passed.
    prepare_agent: Option<crate::run::PrepareAgentFn<C>>,
    /// The first turn's blocking input guardrails already ran (ahead of `prepare_agent`).
    first_blocking_guardrails_done: bool,
    /// The run resumed a parked turn (an approval or failure checkpoint): calls that already
    /// have outputs count towards the tool-use decision, as they did when they ran.
    resumed_from_failure: bool,
    /// Items already in the history when a resume began. A handoff input filter sees them as
    /// the pre-handoff items of the resumed turn, so the response that parked on an approval
    /// is part of the context the filter reasons about, as in Python.
    resumed_pre_count: Option<usize>,
    /// Source of [`RunItem`] lineage keys.
    next_occurrence: u64,
    /// The response id of a held batch that has already been reported to the session.
    settle_notified: Option<String>,
    /// The `store` setting the latest model call ran under, after the run config merge.
    last_store: Option<bool>,
    /// What a server-managed conversation already holds.
    server_conversation: oai_conversation::ServerConversationTracker,
    /// The one prompt cache key this run generates, reused across turns.
    prompt_cache_key: prompt_cache_key::PromptCacheKeyResolver,
    /// The prompt resolved for the coming turn before its handoffs were evaluated, so an
    /// `is_enabled` check sees what a dynamic prompt put in the context, like Python.
    prefetched_prompt: Option<Option<crate::prompts::Prompt>>,
}

/// Drop calls from a settling batch whose outputs did not survive, together with the reasoning
/// items tied to them. `pending` names calls whose approval is still open: they have no output
/// yet because they have not run, so they settle now and pair up at a later exit.
fn settle_safe_items(items: Vec<InputItem>, pending: &HashSet<String>) -> Vec<InputItem> {
    const PLACEHOLDER: &str = "__settle_placeholder";
    let mut working = items;
    // A pending call is exempt from the orphan prune; the prune only understands outputs, so
    // the exemption rides in as a placeholder output that is discarded again below.
    let placeholders: Vec<InputItem> = working
        .iter()
        .filter_map(|item| {
            let ty = item.item_type()?;
            let call_id = item.0.get("call_id").and_then(Value::as_str)?;
            if !pending.contains(call_id) {
                return None;
            }
            let output_type = items::output_type_of_call(ty)?;
            Some(InputItem(
                json!({"type": output_type, "call_id": call_id, PLACEHOLDER: true}),
            ))
        })
        .collect();
    working.extend(placeholders);
    items::drop_orphan_function_calls(working)
        .into_iter()
        .filter(|item| item.0.get(PLACEHOLDER).is_none())
        .collect()
}

/// The tools each agent has used, keyed by agent object. Port of `AgentToolUseTracker`.
///
/// Runtime checks go by agent identity so two agents sharing a name stay separate; the
/// serialized snapshot is keyed by the stable identity key (`name`, `name#2`, ...).
struct ToolUseTracker<C> {
    agents: Vec<(Arc<Agent<C>>, Vec<String>)>,
}

impl<C> Default for ToolUseTracker<C> {
    fn default() -> Self {
        ToolUseTracker { agents: Vec::new() }
    }
}

impl<C> ToolUseTracker<C> {
    fn add(&mut self, agent: &Arc<Agent<C>>, tools: &[String]) {
        match self.agents.iter_mut().find(|(a, _)| Arc::ptr_eq(a, agent)) {
            Some((_, used)) => used.extend(tools.iter().cloned()),
            None => self.agents.push((agent.clone(), tools.to_vec())),
        }
    }

    fn has_used(&self, agent: &Agent<C>) -> bool {
        self.agents
            .iter()
            .any(|(a, used)| std::ptr::eq(a.as_ref(), agent) && !used.is_empty())
    }

    fn snapshot(&self, start: &Arc<Agent<C>>) -> BTreeMap<String, Vec<String>> {
        let keys = agent_identity::identity_keys_by_ptr(start);
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (agent, tools) in &self.agents {
            let key = keys
                .get(&(Arc::as_ptr(agent) as usize))
                .cloned()
                .unwrap_or_else(|| agent.name.clone());
            out.entry(key).or_default().extend(tools.iter().cloned());
        }
        for tools in out.values_mut() {
            tools.sort();
            tools.dedup();
        }
        out
    }

    /// Seed from a snapshot; an entry whose agent is no longer in the graph is skipped.
    fn hydrate(&mut self, start: &Arc<Agent<C>>, snapshot: &BTreeMap<String, Vec<String>>) {
        for (key, tools) in snapshot {
            if let Some(agent) = agent_identity::resolve_identity(start, key) {
                self.add(&agent, tools);
            }
        }
    }
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

/// Remove server replay metadata from an assistant message read back from a store that owns
/// its history. Port of `_sanitize_openai_conversation_history_item_for_model_input`.
fn sanitize_remote_history_item(mut item: InputItem) -> InputItem {
    let is_assistant_message = item.0.get("type").and_then(Value::as_str) == Some("message")
        && item.0.get("role").and_then(Value::as_str) == Some("assistant");
    if is_assistant_message && let Value::Object(map) = &mut item.0 {
        map.remove("id");
        map.remove("provider_data");
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

/// The diagnostic for a routed-name collision. Port of `_format_tool_name_collision_message`:
/// two owners that both derive the name from an agent name get advice about that agent pair.
fn tool_name_collision_message<C>(
    name: &str,
    entries: &[(bool, usize)],
    tools: &[Tool<C>],
    handoffs: &[Handoff<C>],
) -> String {
    // Agent name -> the parameter that would override its derived name, in entry order.
    let mut derived: Vec<(String, &str)> = Vec::new();
    for &(is_handoff, index) in entries {
        let owner = if is_handoff {
            let h = &handoffs[index];
            (h.tool_name == crate::handoffs::default_tool_name(&h.agent_name))
                .then(|| (h.agent_name.clone(), "tool_name_override"))
        } else {
            tools[index].as_function().and_then(|f| {
                let origin = f.tool_origin.as_ref()?;
                let agent = origin.agent_name.as_ref()?;
                (f.is_agent_tool && f.name == crate::util::transform_string_function_style(agent))
                    .then(|| (agent.clone(), "tool_name"))
            })
        };
        if let Some((agent, parameter)) = owner
            && !derived.iter().any(|(a, _)| *a == agent)
        {
            derived.push((agent, parameter));
        }
    }
    if entries.len() == 2 && derived.len() == 2 {
        let ((prior, prior_param), (agent, param)) = (&derived[0], &derived[1]);
        let (kind, label, instruction) = match (*prior_param, *param) {
            ("tool_name", "tool_name") => ("agent tool", "tool name", "`tool_name=`"),
            ("tool_name_override", "tool_name_override") => {
                ("handoff", "handoff tool name", "`tool_name_override=`")
            }
            _ => (
                "agent routing",
                "tool name",
                "`tool_name=` or `tool_name_override=`",
            ),
        };
        return format!(
            "Ambiguous {kind} configuration: agents '{prior}' and '{agent}' both derive the \
             {label} `{name}`. Pass an explicit {instruction} to one of them."
        );
    }
    let has_tool = entries.iter().any(|(h, _)| !*h);
    let has_handoff = entries.iter().any(|(h, _)| *h);
    match (has_tool, has_handoff) {
        (true, false) => format!(
            "Ambiguous function tool configuration: the tool name `{name}` is used by multiple \
             tools. Assign a unique routed name to every colliding function tool with \
             `name_override=`, `tool_name=`, or a namespace."
        ),
        (false, true) => format!(
            "Ambiguous handoff configuration: the handoff tool name `{name}` is used by multiple \
             handoffs. Pass a unique `tool_name_override=` to each handoff."
        ),
        _ => format!(
            "Ambiguous tool routing configuration: the tool name `{name}` is used by both a \
             function tool and a handoff. Assign a unique routed name to every colliding \
             function tool and handoff with `name_override=`, `tool_name=`, \
             `tool_name_override=`, or a namespace."
        ),
    }
}

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
        // Deferred top-level tools are addressed by their synthetic `name.name` shape, never
        // by the bare name, so they cannot collide with a visible tool.
        if let Some(f) = t.as_function()
            && f.namespace().is_none()
            && !f.defer_loading
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
        let message = tool_name_collision_message(name, entries, &tools, &handoffs);
        if policy == crate::run_config::ToolNameCollisionPolicy::Error {
            return Err(AgentsError::user(message));
        }
        if crate::debug::dont_log_tool_data() {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "Tool name collision detected. Assign unique routed tool names or enable tool \
                 data logging for details."
            );
        } else {
            log::warn!(target: crate::logger::LOG_TARGET, "{message}");
        }
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
/// Resolve function tools by the name a call addresses them with, last definition winning.
/// A deferred top-level tool answers to its synthetic `name.name` shape, and to its bare name
/// only while no visible tool claims that name. A qualified name owned by both a namespaced
/// tool and another tool is ambiguous on the wire and refused. Port of
/// `build_function_tool_lookup_map` / `validate_function_tool_lookup_configuration`.
fn function_tool_lookup_map<C>(
    tools: &[Tool<C>],
) -> Result<HashMap<String, &Arc<FunctionTool<C>>>> {
    let functions: Vec<&Arc<FunctionTool<C>>> =
        tools.iter().filter_map(Tool::as_function).collect();
    let mut owners: HashMap<String, &Arc<FunctionTool<C>>> = HashMap::new();
    for f in &functions {
        let qualified = f.qualified_name();
        if let Some(prior) = owners.insert(qualified.clone(), f)
            && (prior.namespace().is_some() || f.namespace().is_some())
        {
            return Err(AgentsError::user(format!(
                "Ambiguous function tool configuration: the qualified name `{qualified}` is \
                 used by multiple tools. Rename the namespace-wrapped function or dotted \
                 top-level tool to avoid ambiguous dispatch."
            )));
        }
    }
    let deferred_top_level = |f: &FunctionTool<C>| f.defer_loading && f.namespace().is_none();
    let mut map: HashMap<String, &Arc<FunctionTool<C>>> = HashMap::new();
    for f in &functions {
        if deferred_top_level(f) {
            map.insert(deferred_top_level_key(&f.name), f);
        }
    }
    for f in &functions {
        if !deferred_top_level(f) {
            map.insert(f.qualified_name(), f);
        }
    }
    for f in &functions {
        if deferred_top_level(f) {
            map.entry(f.name.clone()).or_insert(f);
        }
    }
    Ok(map)
}

/// Lookup key of the synthetic `name.name` wire shape of a deferred top-level tool; distinct
/// from any qualified name, so a dotted top-level tool cannot shadow it.
fn deferred_top_level_key(name: &str) -> String {
    format!("\u{0}deferred_top_level\u{0}{name}")
}

/// Whether the agent still has the tool a model call routes to.
fn call_has_tool<C>(
    tools: &[Tool<C>],
    handoffs: &HashMap<&str, &Handoff<C>>,
    functions: &HashMap<String, &Arc<FunctionTool<C>>>,
    raw: &Value,
) -> bool {
    let ty = raw
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function_call");
    let name = raw.get("name").and_then(Value::as_str).unwrap_or_default();
    match ty {
        "function_call" => {
            handoffs.contains_key(name)
                || serde_json::from_value::<FunctionCall>(raw.clone())
                    .is_ok_and(|fc| functions.contains_key(&function_lookup_name(&fc)))
                || routes_to_apply_patch(tools, raw)
        }
        "custom_tool_call" => {
            tools
                .iter()
                .any(|t| matches!(t, Tool::Custom(c) if c.name == name))
                || routes_to_apply_patch(tools, raw)
        }
        "shell_call" | "apply_patch_call" | "computer_call" | "local_shell_call" => {
            native_tool_name(tools, ty).is_some()
        }
        _ => true,
    }
}

fn function_lookup_name(fc: &FunctionCall) -> String {
    match fc.extra.get("namespace").and_then(Value::as_str) {
        Some(ns) if ns == fc.name => deferred_top_level_key(&fc.name),
        Some(ns) if !ns.is_empty() => format!("{ns}.{}", fc.name),
        _ => fc.name.clone(),
    }
}

/// Display metadata of a hosted MCP tool from the latest `mcp_list_tools` listing that names
/// it: `(description, title)`, the title preferring `title` over `annotations.title`. Port of
/// `collect_mcp_list_tools_metadata` / `extract_mcp_tool_metadata`.
fn hosted_mcp_tool_metadata<'a>(
    items: impl Iterator<Item = &'a Value>,
    server_label: &str,
    tool_name: &str,
) -> Option<(Option<String>, Option<String>)> {
    let non_empty = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mut found = None;
    for raw in items {
        if raw.get("type").and_then(Value::as_str) != Some("mcp_list_tools")
            || raw.get("server_label").and_then(Value::as_str) != Some(server_label)
        {
            continue;
        }
        let Some(tools) = raw.get("tools").and_then(Value::as_array) else {
            continue;
        };
        for tool in tools {
            if tool.get("name").and_then(Value::as_str) == Some(tool_name) {
                let title = non_empty(tool.get("title"))
                    .or_else(|| non_empty(tool.get("annotations").and_then(|a| a.get("title"))));
                found = Some((non_empty(tool.get("description")), title));
            }
        }
    }
    found
}

/// Whether a tool name addresses the apply_patch tool. Port of `is_apply_patch_name`.
fn is_apply_patch_name<C>(name: &str, tool: Option<&crate::tool::ApplyPatchTool<C>>) -> bool {
    let candidate = name.trim().to_lowercase();
    !candidate.is_empty()
        && (candidate.starts_with("apply_patch")
            || tool.is_some_and(|t| candidate == t.name.trim().to_lowercase()))
}

/// Normalize a custom or function call to `apply_patch` into the `apply_patch_call` it stands
/// for. Port of `normalize_apply_patch_fallback_call`: a custom call's input carries the
/// operation (or the whole payload), a function call's arguments are the operation.
fn apply_patch_fallback_call(v: &Value) -> Result<Value> {
    let invalid = || AgentsError::model_behavior("Invalid apply_patch input JSON.");
    let function = v.get("type").and_then(Value::as_str) == Some("function_call");
    let text = v
        .get(if function { "arguments" } else { "input" })
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Value::Object(parsed) = serde_json::from_str::<Value>(text).map_err(|_| invalid())? else {
        return Err(invalid());
    };
    let mut pseudo = serde_json::Map::new();
    pseudo.insert("type".into(), json!("apply_patch_call"));
    pseudo.insert(
        "call_id".into(),
        v.get("call_id").cloned().unwrap_or(Value::Null),
    );
    if !function && (parsed.contains_key("operation") || parsed.contains_key("operations")) {
        pseudo.extend(parsed);
    } else {
        pseudo.insert("operation".into(), Value::Object(parsed));
    }
    Ok(ItemHelpers::copy_tool_call_caller(v, Value::Object(pseudo)))
}

/// Whether a custom or function call falls back to the agent's apply_patch tool: no tool of
/// its own type has its name, but the name is an apply_patch name.
fn routes_to_apply_patch<C>(tools: &[Tool<C>], raw: &Value) -> bool {
    let name = raw.get("name").and_then(Value::as_str).unwrap_or_default();
    let Some(apply_patch) = tools.iter().find_map(|t| match t {
        Tool::ApplyPatch(a) => Some(a.clone()),
        _ => None,
    }) else {
        return false;
    };
    let owned = match raw.get("type").and_then(Value::as_str) {
        Some("custom_tool_call") => tools
            .iter()
            .any(|t| matches!(t, Tool::Custom(c) if c.name == name)),
        Some("function_call") => tools.iter().any(|t| {
            t.as_function().is_some_and(|f| {
                f.qualified_name() == name || (raw.get("namespace").is_none() && f.name == name)
            })
        }),
        _ => return false,
    };
    !owned && is_apply_patch_name(name, Some(&apply_patch))
}

/// The raw item a call is identified by: an apply_patch fallback is identified as the
/// `apply_patch_call` it runs as, as Python registers the pseudo call.
fn invocation_identity_raw<C>(tools: &[Tool<C>], raw: &Value) -> Value {
    if routes_to_apply_patch(tools, raw)
        && let Ok(pseudo) = apply_patch_fallback_call(raw)
    {
        return pseudo;
    }
    raw.clone()
}

/// Drop the server-assigned `created_by` from a shell or apply_patch call (and the chunks of a
/// shell output) before the item is recorded or executed, as `process_model_response` does.
fn sanitize_local_item(ty: &str, v: &Value) -> Option<Value> {
    if !matches!(ty, "shell_call" | "apply_patch_call" | "shell_call_output") {
        return None;
    }
    let mut raw = v.clone();
    if let Value::Object(m) = &mut raw {
        m.remove("created_by");
        if let Some(Value::Array(chunks)) = m.get_mut("output") {
            for chunk in chunks {
                if let Value::Object(chunk) = chunk {
                    chunk.remove("created_by");
                }
            }
        }
    }
    Some(raw)
}

/// Match a non-function output item to a local runtime tool of the agent.
fn match_local_call<C>(
    tools: &[Tool<C>],
    v: &Value,
    ty: &str,
    agent_name: &str,
) -> Result<Option<local_tools::LocalCall<C>>> {
    use local_tools::LocalCall;
    let missing = |message: &str| Err(AgentsError::model_behavior(message));
    let apply_patch_tool = || {
        tools.iter().find_map(|t| match t {
            Tool::ApplyPatch(a) => Some(a.clone()),
            _ => None,
        })
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
            None => missing("Model produced computer action without a computer tool."),
        },
        // Hosted shells are executed by OpenAI: only local shell tools run here.
        "shell_call" => match tools.iter().find_map(|t| match t {
            Tool::Shell(s) => Some(s.clone()),
            _ => None,
        }) {
            None => missing("Model produced shell call without a shell tool."),
            Some(s) if !s.is_local() => Ok(None),
            Some(s) => {
                tool_caller::ensure_tool_caller_allowed(
                    v,
                    s.allowed_callers.as_deref(),
                    &s.name,
                    agent_name,
                )?;
                Ok(Some(LocalCall::Shell(v.clone(), s)))
            }
        },
        "local_shell_call" => match tools.iter().find_map(|t| match t {
            Tool::LocalShell(s) => Some(s.clone()),
            _ => None,
        }) {
            Some(s) => {
                tool_caller::ensure_tool_caller_allowed(v, None, s.name(), agent_name)?;
                Ok(Some(LocalCall::LocalShell(v.clone(), s)))
            }
            None => missing("Model produced local shell call without a local shell tool."),
        },
        "apply_patch_call" => match apply_patch_tool() {
            Some(a) => {
                tool_caller::ensure_tool_caller_allowed(
                    v,
                    a.allowed_callers.as_deref(),
                    &a.name,
                    agent_name,
                )?;
                Ok(Some(LocalCall::ApplyPatch(v.clone(), a)))
            }
            None => missing("Model produced apply_patch call without an apply_patch tool."),
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
                // Legacy hosted rollouts send apply_patch as a custom tool call. Port of
                // `is_apply_patch_name` / `normalize_apply_patch_fallback_call`.
                None if is_apply_patch_name(name, apply_patch_tool().as_deref()) => {
                    let Some(a) = apply_patch_tool() else {
                        return missing(
                            "Model produced apply_patch call without an apply_patch tool.",
                        );
                    };
                    let pseudo = apply_patch_fallback_call(v)?;
                    tool_caller::ensure_tool_caller_allowed(
                        &pseudo,
                        a.allowed_callers.as_deref(),
                        &a.name,
                        agent_name,
                    )?;
                    Ok(Some(LocalCall::ApplyPatch(pseudo, a)))
                }
                None => Err(AgentsError::model_behavior(format!(
                    "Tool {name} not found in agent {agent_name}"
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
        checkpoint,
        prepare_agent,
    } = options;
    // The provider `RunConfig::default` creates belongs to this run, which closes it at the
    // end; a caller's provider stays open for reuse, and a nested agent-as-tool run leaves the
    // parent's provider alone. Port of `_close_runner_owned_model_provider`.
    let owns_model_provider = run_config.model_provider.is_run_owned()
        && !shares_parent_context
        && agent_tool_invocation.is_none();

    // A nested agent-tool run has its own invocation lifecycle: its call ids are independent
    // of the outer run's.
    let context = if shares_parent_context {
        context.with_fresh_invocations()
    } else {
        context
    };
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
        nested_history_owned_items: Vec::new(),
        nested_model_start: None,
        compaction_exchange: None,
        pending_write: None,
        turn_in_flight: false,
        failed_after_output: false,
        checkpoint,
        prepare_agent,
        first_blocking_guardrails_done: false,
        resumed_from_failure: false,
        resumed_pre_count: None,
        next_occurrence: 0,
        settle_notified: None,
        last_store: None,
        server_conversation: Default::default(),
        prompt_cache_key: Default::default(),
        prefetched_prompt: None,
        model_responses: Vec::new(),
        current_turn: 0,
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        tools_used: ToolUseTracker::default(),
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
        resumed_pending_approvals: BTreeSet::new(),
        resumed_interruptions: Vec::new(),
        reasoning_item_id_policy: ReasoningItemIdPolicy::Preserve,
        restored_sandbox: None,
        model_owner: crate::models::run_context::new_model_run_owner(),
        models_used: Vec::new(),
        trace_state: None,
        turn_start: 0,
        last_processed_response: None,
        turn_tool_input_guardrail_start: 0,
        turn_tool_output_guardrail_start: 0,
    };
    lp.reasoning_item_id_policy = lp.config.reasoning_item_id_policy;

    // Python validates these when the `RunConfig` is built; Rust's config is a plain struct,
    // so the run rejects them before anything else happens.
    if lp.config.tool_execution.max_function_tool_concurrency == Some(0) {
        return Err(AgentsError::user(
            "tool_execution.max_function_tool_concurrency must be at least 1",
        ));
    }
    if matches!(
        &lp.config.output_guardrail_blocked_message,
        Some(crate::run_config::OutputGuardrailBlockedMessage::Text(t)) if t.is_empty()
    ) {
        return Err(AgentsError::user(
            "output_guardrail_blocked_message must be non-empty",
        ));
    }

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
        RunInput::State(state) => {
            // A resumed state may carry its own server-managed conversation. Port of
            // `validate_session_conversation_settings` on the resumed state.
            if lp.session.is_some()
                && (state.conversation_id.is_some()
                    || state.previous_response_id.is_some()
                    || state.auto_previous_response_id)
            {
                return Err(AgentsError::user(
                    "Session persistence cannot be combined with conversation_id, previous_response_id, or auto_previous_response_id.",
                ));
            }
            lp.restore(*state, !shares_parent_context)?
        }
        other => {
            let items = other.to_items();
            lp.original_input = items.clone();
            lp.session_input = items.clone();
            lp.model_prefix = items;
        }
    }
    // Decisions recorded on the context bind the calls they were made for, as Python's
    // `RunContextWrapper._tool_invocations` does for a reused or converted context.
    if !shares_parent_context {
        for (call_id, record) in lp.ctx.export_tool_invocations() {
            lp.tool_invocations.entry(call_id).or_insert(record);
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
        if owns_model_provider {
            lp.config.model_provider.close().await;
        }
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
        // A native call's scope names the tool it resolved to, which the item alone does not
        // carry; the live check re-validates it against the agent's tools on resume.
        let named_native = NATIVE_INVOCATION_TYPES.contains(&id.invocation_type.as_str());
        if record.invocation_type != id.invocation_type
            || (!named_native
                && (record.approval_scope != id.approval_scope
                    || record.fingerprint != id.fingerprint))
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
/// Invocation types whose approval scope includes the name of the tool they resolved to.
const NATIVE_INVOCATION_TYPES: [&str; 4] = [
    "apply_patch_call",
    "computer_call",
    "local_shell_call",
    "shell_call",
];

/// The name of the native tool a call of `invocation_type` resolves to on this agent, which
/// Python passes as `tool_name` when it scopes the invocation.
fn native_tool_name<C>(tools: &[Tool<C>], invocation_type: &str) -> Option<String> {
    tools.iter().find_map(|t| match (invocation_type, t) {
        ("shell_call", Tool::Shell(s)) => Some(s.name.clone()),
        ("apply_patch_call", Tool::ApplyPatch(p)) => Some(p.name.clone()),
        ("computer_call", Tool::Computer(c)) => Some(c.name().to_string()),
        ("local_shell_call", Tool::LocalShell(l)) => Some(l.name().to_string()),
        _ => None,
    })
}

/// The identity of every tool call in `items` whose output is already in `items`, keyed by
/// call id. A call the run's ledger bound keeps the ledger's identity (which carries the tool
/// the call originally resolved to); otherwise it is recomputed against the current tools.
/// Port of `historical_entries` / `get_historical_identity` in
/// `_dedupe_processed_response_invocations`.
fn completed_history_identities<C>(
    items: &[RunItem],
    ledger: &BTreeMap<String, ToolInvocationRecord>,
    tools: &[Tool<C>],
) -> Result<HashMap<String, (String, String, String)>> {
    let outputs: HashSet<String> = items
        .iter()
        .filter_map(|item| {
            let raw = match &item.kind {
                RunItemKind::ToolCallOutput { raw_item, .. } => raw_item.clone(),
                RunItemKind::HandoffOutput { raw_item, .. } => raw_item.0.clone(),
                RunItemKind::MCPApprovalResponse { raw_item } => raw_item.clone(),
                _ => return None,
            };
            tool_invocation::tool_output_identity(&raw).map(|(_, call_id)| call_id)
        })
        .collect();
    let mut history: HashMap<String, (String, String, String)> = HashMap::new();
    for item in items {
        let Some((raw, role)) = invocation_raw_item(item) else {
            continue;
        };
        let ty = raw.get("type").and_then(Value::as_str).unwrap_or_default();
        let Some(id) = tool_invocation::tool_invocation_identity_named(
            &raw,
            native_tool_name(tools, ty).as_deref(),
            role,
        ) else {
            continue;
        };
        if !outputs.contains(&id.call_id) {
            continue;
        }
        let identity = match ledger.get(&id.call_id) {
            Some(r) if r.invocation_type == id.invocation_type => (
                r.invocation_type.clone(),
                r.approval_scope.clone(),
                r.fingerprint.clone(),
            ),
            _ => (id.invocation_type, id.approval_scope, id.fingerprint),
        };
        match history.get(&id.call_id) {
            Some(previous) if *previous != identity => {
                return Err(AgentsError::model_behavior(
                    "Run history reused a tool call ID for different completed invocations. \
                     Use a unique call ID for each tool invocation.",
                ));
            }
            Some(_) => {}
            None => {
                history.insert(id.call_id, identity);
            }
        }
    }
    // Calls the ledger records as completed (a restored snapshot may hold them only in its
    // original input) count as history too.
    for (call_id, record) in ledger {
        if record.completed {
            history.entry(call_id.clone()).or_insert_with(|| {
                (
                    record.invocation_type.clone(),
                    record.approval_scope.clone(),
                    record.fingerprint.clone(),
                )
            });
        }
    }
    Ok(history)
}

/// `live` is the run's own ledger: a call it already bound keeps that identity, which for a
/// native call includes the resolved tool name the item alone cannot reproduce.
fn tool_invocation_ledger(
    items: &[RunItem],
    unbound_approvals: &BTreeSet<String>,
    live: &BTreeMap<String, ToolInvocationRecord>,
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
            let bound = live
                .get(&id.call_id)
                .filter(|r| r.invocation_type == id.invocation_type);
            ledger
                .entry(id.call_id)
                .or_insert_with(|| ToolInvocationRecord {
                    invocation_type: id.invocation_type,
                    approval_scope: bound.map_or(id.approval_scope, |r| r.approval_scope.clone()),
                    fingerprint: bound.map_or(id.fingerprint, |r| r.fingerprint.clone()),
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
        // A finished snapshot resumes like Python's: the run continues after its final output
        // (with the turn budget it already spent), so no `final_output` guard here.
        // Port of `reject_unrecoverable_resume`: the snapshot owns a terminal result no
        // resume can reproduce, so it is refused before any session, model or tool work.
        if state.terminal_unrecoverable {
            return Err(AgentsError::user(
                "This RunState ended at an unrecoverable boundary and cannot be resumed. \
                 Start a new run instead.",
            ));
        }
        // A withheld batch belongs to the session it was parked against; settling it into
        // another conversation would split the call from its history.
        if let (Some(pending), Some(session)) = (&state.pending_session_write, &self.session)
            && pending
                .get("held")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && let Some(expected) = pending.get("session_id").and_then(Value::as_str)
            && session.session_id() != expected
        {
            return Err(AgentsError::user(
                "Resume the pending Session write with the original Session and session ID.",
            ));
        }
        self.pending_write = state.pending_session_write.clone();
        // A graph with same-named agents resumes through the identity key the snapshot recorded
        // for its current agent; a snapshot without one cannot say which duplicate it meant.
        let ambiguous = ambiguous_agent_names(&self.starting_agent);
        if state.current_agent_identity.is_none() && ambiguous.contains(&state.current_agent_name) {
            return Err(AgentsError::user(format!(
                "Cannot resume a run whose agent graph has same-named agents ({}). \
                 Use unique agent names before starting a resumable run.",
                ambiguous.join(", ")
            )));
        }
        let identity = state
            .current_agent_identity
            .as_deref()
            .unwrap_or(&state.current_agent_name);
        let agent =
            agent_identity::resolve_identity(&self.starting_agent, identity).ok_or_else(|| {
                // The message stays value-free: a snapshot's agent name is payload data.
                // Python: "Run state agent not found in agent map".
                AgentsError::user("Run state agent not found in agent map")
            })?;
        self.current_agent = agent;
        // Output guardrails must withhold the parked response's items until they pass, so they
        // need to know exactly which items those are. Port of
        // `_validate_resumed_session_output_guardrail_safety`.
        if state.current_response_boundary_unproven
            && !state.interruptions.is_empty()
            && (!self.current_agent.output_guardrails.is_empty()
                || !self.config.output_guardrails.is_empty())
        {
            return Err(AgentsError::user(
                "Cannot resume a serialized approval checkpoint with output guardrails because \
                 the current response boundary cannot be proven. Start a new run from safe input.",
            ));
        }
        self.ctx.import_approvals(&state.approvals);
        self.ctx
            .import_nested_states(&state.nested_agent_tool_states);
        if restore_usage {
            self.ctx.set_usage(state.usage.clone());
        }
        self.original_input = state.original_input.clone();
        self.session_input = state.original_input.clone();
        // The resumed model input starts from the snapshot's input, without calls that never
        // got an output. Port of `normalize_resumed_input`.
        self.model_prefix = items::drop_orphan_function_calls(state.original_input.clone());
        self.generated_items = state
            .generated_items
            .iter()
            .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
            .cloned()
            .collect();
        // After a nested handoff the earlier items are summarized in the input.
        let model_start = state.model_input_start.min(self.generated_items.len());
        self.model_items = self.generated_items[model_start..].to_vec();
        if model_start > 0 {
            self.nested_model_start = Some(model_start);
            self.model_history_diverged = true;
        }
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
        self.tools_used
            .hydrate(&self.starting_agent, &state.tool_use_tracker);
        self.session_saved_items = state.session_saved_items.min(self.generated_items.len());
        self.input_saved_to_session = state.input_saved_to_session;
        self.pending_input = state.pending_input.clone();
        // A restored call must still be the call its ledger entry was written for. Port of
        // `RunContextWrapper._tool_invocation_status`'s identity check: a snapshot whose tool
        // arguments or routing were edited is rejected instead of silently re-approved.
        validate_restored_tool_invocations(&state)?;
        self.tool_invocations = state.tool_invocations.clone();
        // A call that started running but never committed its output must not run again.
        for (call_id, record) in &state.tool_invocations {
            if record.executed && !record.completed {
                self.ctx.mark_invocation_executed(call_id);
            }
        }
        // A decision the snapshot could not bind to a concrete invocation is not an
        // authorization; the resumed run raises the approval again.
        self.unbound_approvals = state.restored_unbound_approval_call_ids.clone();
        // Only this run's own calls: a nested agent tool's pending call may share a call id
        // with the outer call that runs it.
        self.resumed_pending_approvals = state
            .interruptions
            .iter()
            .filter(|i| i.agent == state.current_agent_name)
            .map(|i| i.call_id().to_string())
            .collect();
        self.resumed_interruptions = state
            .interruptions
            .iter()
            .filter(|i| i.agent == state.current_agent_name)
            .cloned()
            .collect();
        self.restored_sandbox = state.sandbox.clone();
        // A resumed nested run keeps its structured input unless the caller's context brings
        // its own.
        if self.ctx.tool_input().is_none()
            && let Some(tool_input) = &state.tool_input
        {
            self.ctx = self.ctx.with_tool_input(tool_input.clone());
        }
        self.nested_history_owned_items = state
            .nested_history_owned_items
            .iter()
            .copied()
            .filter(|index| *index < self.generated_items.len())
            .collect();
        if self.reasoning_item_id_policy == ReasoningItemIdPolicy::Preserve
            && let Some(policy) = state.reasoning_item_id_policy
        {
            self.reasoning_item_id_policy = policy;
        }
        self.resume_pending_tools = state.pending_tool_processing;
        // Carried so a snapshot taken before the resumed turn re-executes still names the
        // tools the parked response asked for; `process_response` replaces it.
        self.last_processed_response = state.last_processed_response.clone();
        // Tool output guardrail results from before the parked response stay out of the
        // resumed turn's redaction window. Mirrors `tool_output_guardrail_result_start`.
        if let Some(start) = state
            .last_processed_response
            .as_ref()
            .and_then(|p| p.tool_output_guardrail_result_start)
        {
            self.turn_tool_output_guardrail_start = start;
        }
        self.trace_id = state.trace_id.clone();
        self.trace_state = state.trace_state.clone();
        self.turn_start = state.turn_start_index.min(self.generated_items.len());
        self.resumed_pre_count = state
            .pending_tool_processing
            .then_some(self.generated_items.len());
        // Approval and failure checkpoints alike: the calls that already completed in the parked
        // turn feed its tool-use decision. Mirrors `resolve_interrupted_turn`.
        self.resumed_from_failure = state.pending_tool_processing;
        // Output guardrails can still reject the parked turn, so its items must not already be
        // in the session. Port of `_validate_resumed_session_output_guardrail_safety`.
        if state.pending_tool_processing && self.output_guardrails_gate_applies() {
            let persisted = self.session_saved_items.saturating_sub(self.turn_start);
            if persisted > 0 {
                if self.session.is_some() {
                    return Err(AgentsError::user(
                        "Cannot resume an approval checkpoint with output guardrails after \
                         current-turn items were persisted. Start a new run from safe input.",
                    ));
                }
                // A detached run cannot contribute its old persisted prefix to this run.
                self.session_saved_items = self.turn_start;
            }
        }
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
        let output = match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item),
            RunItemKind::HandoffOutput { raw_item, .. } => Some(&raw_item.0),
            _ => None,
        };
        if let Some((_, call_id)) = output.and_then(tool_invocation::tool_output_identity) {
            self.ctx.mark_invocation_completed(&call_id);
            self.ctx.mark_bound_invocation_completed(&call_id);
        }
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

    /// The state a failed run leaves behind. A failure inside a turn that already processed its
    /// model response resumes by re-processing that response: calls whose outputs were
    /// committed are skipped, the rest run.
    fn failure_checkpoint(&self) -> RunState {
        let mut state = *self.build_result(None, Vec::new()).state;
        state.pending_tool_processing = self.turn_in_flight;
        state
    }

    fn server_managed(&self) -> bool {
        self.conversation_id.is_some()
            || self.previous_response_id.is_some()
            || self.auto_previous_response_id
    }

    async fn run(&mut self) -> Result<RunResult<C>> {
        // An immediate cancel of a streamed run stops it at its next await point; the cleanup
        // below still runs, as Python's `finally` blocks do for a cancelled run task.
        let cancel = self.sink.as_ref().map(|s| s.control.clone());
        let outcome = match cancel {
            Some(control) => {
                tokio::select! {
                    biased;
                    _ = control.cancelled() => Err(AgentsError::other("The streamed run was cancelled.")),
                    outcome = self.run_inner() => outcome,
                }
            }
            None => self.run_inner().await,
        };
        self.dispose_computers().await;
        match outcome {
            Ok(r) => Ok(r),
            Err(e) => {
                // A tripped input guardrail still records the user's turn input in the
                // session, never the model output. Port of
                // `persist_session_items_for_guardrail_trip`.
                if matches!(e, AgentsError::InputGuardrailTripwireTriggered { .. })
                    && let Err(save_error) = self.persist_input_for_guardrail_trip().await
                {
                    return Err(save_error);
                }
                if let Some(slot) = &self.checkpoint {
                    slot.set(self.failure_checkpoint());
                }
                // A data-redacted error must not carry the run's payload either.
                if e.is_data_redacted() {
                    return Err(e);
                }
                let details = self.run_details();
                Err(e.with_run_data(details))
            }
        }
    }

    async fn persist_input_for_guardrail_trip(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        if self.server_managed() || self.input_saved_to_session || self.session_input.is_empty() {
            return Ok(());
        }
        let items = self.session_input.clone();
        crate::memory::await_session_mutation(async move { session.add_items(items).await })
            .await?;
        self.input_saved_to_session = true;
        Ok(())
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
        if let Some(prepare) = self.prepare_agent.take() {
            // Python runs the blocking input guardrails before it prepares (creates the
            // sandbox session for) the starting agent, so rejected input never costs one.
            if !self.is_resumed {
                let blocking: Vec<_> = self
                    .config
                    .input_guardrails
                    .iter()
                    .chain(self.current_agent.input_guardrails.iter())
                    .filter(|g| !g.run_in_parallel)
                    .cloned()
                    .collect();
                if !blocking.is_empty() {
                    let input = self.original_input.clone();
                    let results = self.run_input_guardrails(blocking, input).await?;
                    self.record_input_guardrails(results)?;
                }
                self.first_blocking_guardrails_done = true;
            }
            let prepared = prepare(self.starting_agent.clone()).await?;
            self.starting_agent = prepared.clone();
            self.current_agent = prepared;
        }
        loop {
            // A resume that still owes the approved tools their turn runs them first: an
            // after-turn cancel stops the run after that turn, not before it.
            // Between turns a streamed run lets its consumer handle the finished turn's events
            // first, so an after-turn cancel issued while handling them stops the run here.
            // Port of `_wait_for_streamed_turn_events_and_stop_if_cancelled`.
            if !self.resume_pending_tools
                && self.current_turn > 0
                && let Some(s) = &self.sink
            {
                s.control.wait_for_consumption().await;
            }
            if !self.resume_pending_tools
                && let Some(s) = &self.sink
                && s.cancel_after_turn.load(Ordering::SeqCst)
            {
                self.save_session().await?;
                return Ok(self.build_result(None, Vec::new()));
            }

            let step = if self.resume_pending_tools {
                self.resume_pending_tools = false;
                // An edited parked call is refused before any tool inventory callback runs.
                self.validate_parked_response()?;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                let handoffs = self.current_agent.get_enabled_handoffs(&self.ctx).await;
                let (tools, handoffs) = resolve_tool_name_collisions(
                    tools,
                    handoffs,
                    self.config.tool_name_collision_policy,
                )?;
                self.start_agent_span(&tools, &handoffs);
                let span = self.agent_span.clone().expect("agent span");
                let input_results = self.tool_input_guardrail_results.len();
                let output_results = self.tool_output_guardrail_results.len();
                let step =
                    scope::with_span(span, Box::pin(self.resume_tools(tools, handoffs))).await?;
                self.resumed_pending_approvals.clear();
                self.resumed_interruptions.clear();
                // Python reports the tool guardrail results of a resumed turn only when that
                // turn ends the run, hands off or interrupts again; a resumed turn that runs
                // the model again leaves them out (`resolve_interrupted_turn` + run loop).
                if matches!(step, NextStep::RunAgain) {
                    self.tool_input_guardrail_results.truncate(input_results);
                    self.tool_output_guardrail_results.truncate(output_results);
                }
                step
            } else {
                if let Some(max) = self.max_turns
                    && self.current_turn >= max
                {
                    return self.handle_max_turns(max).await;
                }
                self.current_turn += 1;
                // Python's turn order: agent-start hooks, tools, prompt, then handoffs, so an
                // `is_enabled` check sees what the hooks and a dynamic prompt set up.
                self.run_agent_start_hooks().await;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                self.prefetched_prompt = Some(self.current_agent.get_prompt(&self.ctx).await?);
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

            self.resumed_pre_count = None;
            self.turn_in_flight = false;
            match step {
                NextStep::FinalOutput(output) => {
                    return Box::pin(self.finish_with_output(output)).await;
                }
                NextStep::Interruption(items) => {
                    if self.defers_interrupted_writes() {
                        // The parked response may still become guarded terminal output, so it
                        // is withheld: only the input and earlier turns are written now, and
                        // the batch settles with the resolved turn at a later exit.
                        let pending: HashSet<String> =
                            items.iter().map(|i| i.raw_item.call_id.clone()).collect();
                        self.save_session_through(self.turn_start, &pending).await?;
                        self.register_held_write();
                    } else if self.session.is_none() && self.pending_write.is_some() {
                        // A detached resume cannot write, but the standing held record
                        // carries the session identity: the newly parked call folds into it
                        // so the reattach does not settle its output orphaned.
                        self.register_held_write();
                    } else {
                        let pending: HashSet<String> =
                            items.iter().map(|i| i.raw_item.call_id.clone()).collect();
                        self.save_session_through(self.generated_items.len(), &pending)
                            .await?;
                        self.notify_session_completed().await?;
                    }
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
                // Port of `finalize_max_turns_handler_output`: validate, then gate the output
                // with the output guardrails before the synthesized message is recorded.
                // Everything accepted so far (the input included) is persisted on its own, so
                // a rejected or failed final write never takes it down too.
                self.save_session().await?;
                let (output, item) = self.synthesize_handler_output(&res)?;
                let agent = self.current_agent.clone();
                if let Some(h) = &self.hooks {
                    h.on_agent_end(&self.ctx, &agent, &output).await;
                }
                if let Some(h) = &agent.hooks {
                    h.on_end(&self.ctx, &agent, &output).await;
                }
                match self.run_final_output_guardrails(&agent, &output).await {
                    Ok(None) => {}
                    Ok(Some((tripped, _))) => {
                        return Err(AgentsError::OutputGuardrailTripwireTriggered {
                            guardrail_result: Box::new(tripped),
                            run_data: None,
                        });
                    }
                    Err(e) => {
                        // A crashing guardrail does not withdraw the handler output.
                        if let Some(item) = item {
                            self.push_item(item, None);
                            if let Err(save) = self.save_session().await {
                                return Err(e.redact_follow_up(save));
                            }
                        }
                        return Err(e);
                    }
                }
                if let Some(item) = item {
                    // Publish the output only once the session holds it.
                    self.push_item(item.clone(), None);
                    if let Err(e) = self.save_session().await {
                        self.generated_items.pop();
                        self.model_items.pop();
                        return Err(e);
                    }
                    self.emit_item(RunItemStreamEventName::MessageOutputCreated, &item);
                }
                self.notify_session_completed().await?;
                let mut r = self.build_result(Some(output), Vec::new());
                r.state.final_output = r.final_output.clone();
                Ok(r)
            }
        }
    }

    /// Validate an error handler's final output against the agent's output type and build the
    /// message item that records it. Port of `validate_handler_final_output`,
    /// `format_final_output_text` and `create_message_output_item`.
    fn synthesize_handler_output(
        &self,
        res: &crate::run_error_handlers::RunErrorHandlerResult,
    ) -> Result<(Value, Option<RunItem>)> {
        let invalid =
            || AgentsError::user("Invalid run error handler final_output for structured output.");
        let (output, text) = match &self.current_agent.output_type {
            Some(schema) if !schema.is_plain_text() => {
                let payload = res.final_output.to_string();
                match schema.validate_json(&payload) {
                    Ok(v) => (v, payload),
                    Err(_) => {
                        // A wrapped (non-object) output type takes the bare value.
                        let wrapped = json!({ "response": res.final_output }).to_string();
                        let v = schema.validate_json(&wrapped).map_err(|_| invalid())?;
                        (v, wrapped)
                    }
                }
            }
            _ => {
                let text = match &res.final_output {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (res.final_output.clone(), text)
            }
        };
        let item = res.include_in_history.then(|| {
            RunItem::new(
                self.current_agent.name.clone(),
                RunItemKind::MessageOutput {
                    raw_item: MessageOutput::text(text),
                },
            )
        });
        Ok((output, item))
    }

    /// Run the agent's and the run's output guardrails over `output`, recording each result.
    /// Returns the first tripped result with the index its batch started at.
    async fn run_final_output_guardrails(
        &mut self,
        agent: &Arc<Agent<C>>,
        output: &Value,
    ) -> Result<Option<(OutputGuardrailResult, usize)>> {
        let mut guardrails = agent.output_guardrails.clone();
        guardrails.extend(self.config.output_guardrails.iter().cloned());
        if guardrails.is_empty() {
            return Ok(None);
        }
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
                return Ok(Some((r, batch_start)));
            }
        }
        Ok(None)
    }

    async fn finish_with_output(&mut self, output: Value) -> Result<RunResult<C>> {
        let agent = self.current_agent.clone();
        match self.run_final_output_guardrails(&agent, &output).await {
            Ok(None) => {}
            Ok(Some((tripped, batch_start))) => {
                return Err(self.block_output(&agent, tripped, batch_start).await?);
            }
            Err(e) => {
                // A crashing guardrail does not undo the turn that already ran: its calls and
                // their outputs are persisted together before the error propagates, so history
                // never keeps a call without its output. A failed final-turn save is the more
                // actionable error and wins, unless the guardrail error is data-redacted.
                if let Err(save) = self.save_session().await {
                    return Err(e.redact_follow_up(save));
                }
                return Err(e);
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
        // Only a plain function call/output batch can be kept with a placeholder. A response
        // that carries reasoning or any other tool variant cannot be replayed partially, so the
        // whole rejected response goes: from history, the model transcript, the responses and
        // the turn's tool output guardrail results. Port of
        // `_retained_items_for_blocked_output` returning `[]`.
        let unsupported = self.generated_items[turn_start..]
            .iter()
            .any(|i| match &i.kind {
                RunItemKind::Reasoning { .. } => true,
                RunItemKind::ToolCall { raw_item } => {
                    raw_item.get("type").and_then(Value::as_str) != Some("function_call")
                        || raw_item
                            .get("caller")
                            .and_then(|c| c.get("type"))
                            .and_then(Value::as_str)
                            .is_some_and(|t| t != "direct")
                }
                RunItemKind::ToolCallOutput { raw_item, .. } => {
                    raw_item.get("type").and_then(Value::as_str) != Some("function_call_output")
                }
                _ => false,
            });
        if unsupported {
            let dropped = self.generated_items.len() - turn_start;
            self.generated_items.truncate(turn_start);
            let model_keep = self.model_items.len().saturating_sub(dropped);
            self.model_items.truncate(model_keep);
            self.model_responses.pop();
            let tool_output_start = self
                .turn_tool_output_guardrail_start
                .min(self.tool_output_guardrail_results.len());
            self.tool_output_guardrail_results
                .truncate(tool_output_start);
            self.save_session().await?;
            self.session_saved_items = self.generated_items.len();
            for r in &mut self.output_guardrail_results[batch_start..] {
                r.agent_output = Value::String(message.clone());
                r.output.output_info = Value::Null;
            }
            let mut tripped = tripped;
            tripped.agent_output = Value::String(message.clone());
            tripped.output.output_info = Value::Null;
            return Ok(AgentsError::OutputGuardrailTripwireTriggered {
                guardrail_result: Box::new(tripped),
                run_data: None,
            });
        }
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
        // The tripwire of a blocked terminal output is data-redacted, so a failed write of the
        // placeholder history surfaces without its payload. Port of `_safe_redacted_persistence_error`.
        self.save_session()
            .await
            .map_err(|_| AgentsError::data_redacted())?;
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
            current_agent_identity: {
                let key = agent_identity::identity_key(&self.starting_agent, &self.current_agent);
                (key != self.current_agent.name).then_some(key)
            },
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
            model_input_start: self.nested_model_start.unwrap_or(0),
            tool_use_tracker: self.tools_used.snapshot(&self.starting_agent),
            generated_prompt_cache_key: self.prompt_cache_key.generated_key(),
            // Only a non-default policy is recorded: `Preserve` is indistinguishable from an
            // unset config policy, which Python writes as null.
            sandbox: self.restored_sandbox.clone(),
            tool_input: self.ctx.tool_input().cloned(),
            nested_history_owned_items: self.nested_history_owned_items.clone(),
            reasoning_item_id_policy: (self.reasoning_item_id_policy
                == ReasoningItemIdPolicy::Omit)
                .then_some(ReasoningItemIdPolicy::Omit),
            server_conversation: self.server_conversation.state(),
            // Rust returns an error instead of a snapshot when a terminal save fails, so a
            // snapshot this loop builds is always resumable.
            terminal_unrecoverable: false,
            pending_session_write: self.pending_write.clone(),
            tool_invocations: {
                let mut ledger = tool_invocation_ledger(
                    &self.generated_items,
                    &self.unbound_approvals,
                    &self.tool_invocations,
                );
                for (call_id, record) in ledger.iter_mut() {
                    if !record.completed && self.ctx.invocation_executed_uncommitted(call_id) {
                        record.executed = true;
                    }
                }
                ledger
            },
            // An approval the user has not renewed stays unbound across the snapshot, so the
            // decision cannot quietly become authoritative again on the next resume.
            restored_unbound_approval_call_ids: self.unbound_approvals.clone(),
            // A snapshot this loop builds always knows which items its parked response owns.
            current_response_boundary_unproven: false,
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
            nested_history_owned_items: self.nested_history_owned_items.clone(),
            state: Box::new(state),
        }
    }

    async fn save_session(&mut self) -> Result<()> {
        self.save_session_through(self.generated_items.len(), &HashSet::new())
            .await
    }

    /// After a failed session write, count the unsaved items as written when the session tail
    /// already ends with exactly them. Port of the digest check of `resume_pending_session_write`.
    async fn reconcile_failed_session_write(&mut self) {
        let Some(session) = self.session.clone() else {
            return;
        };
        let start = self.session_saved_items.min(self.generated_items.len());
        let batch = items_to_input(&self.generated_items[start..]);
        if batch.is_empty() {
            return;
        }
        if let Ok(tail) = session.get_items(Some(batch.len())).await
            && tail == batch
        {
            self.session_saved_items = self.generated_items.len();
        }
    }

    /// Whether output guardrails could still reject a turn this agent finishes with a tool.
    fn output_guardrails_gate_applies(&self) -> bool {
        (!self.current_agent.output_guardrails.is_empty()
            || !self.config.output_guardrails.is_empty())
            && !matches!(
                self.current_agent.tool_use_behavior,
                ToolUseBehavior::RunLlmAgain
            )
    }

    /// Whether an interruption must withhold its session write: only approval state that could
    /// still become guarded terminal tool output is held back. Port of
    /// `_should_defer_interrupted_session_items`.
    fn defers_interrupted_writes(&self) -> bool {
        self.session.is_some() && self.output_guardrails_gate_applies()
    }

    /// Record the withheld batch on the checkpoint. Registering touches only the checkpoint,
    /// never the session. A detached run (no session) folds into a standing record instead,
    /// which carries the session identity. Port of `defer_interrupted_session_write` /
    /// `extend_held_session_write`.
    fn register_held_write(&mut self) {
        let session_id = match &self.session {
            Some(session) => Some(session.session_id().to_string()),
            None => self
                .pending_write
                .as_ref()
                .and_then(|p| p.get("session_id"))
                .and_then(Value::as_str)
                .map(str::to_string),
        };
        let Some(session_id) = session_id else {
            return;
        };
        let start = self.session_saved_items.min(self.generated_items.len());
        let boundary = self.turn_start.clamp(start, self.generated_items.len());
        let items: Vec<Value> = items_to_input(&self.generated_items[start..])
            .into_iter()
            .map(|i| i.0)
            .collect();
        if items.is_empty() {
            self.pending_write = None;
            return;
        }
        let prior = items_to_input(&self.generated_items[start..boundary]).len();
        // Storage belongs to the response, not to the later run resolving its approvals: an
        // extension of the same response keeps the response's id and setting, while a freshly
        // parked response advances the frontier to its own.
        let same_response = self
            .pending_write
            .as_ref()
            .and_then(|p| p.get("current_response"))
            .and_then(|c| c.get("turn"))
            .and_then(Value::as_u64)
            == Some(self.current_turn as u64);
        let (response_id, store) = match (&self.pending_write, same_response) {
            (Some(existing), true) => (
                existing.get("response_id").cloned().unwrap_or(Value::Null),
                existing.get("store").cloned().unwrap_or(Value::Null),
            ),
            _ => (
                json!(
                    self.model_responses
                        .last()
                        .and_then(|r| r.response_id.clone())
                ),
                json!(self.last_store),
            ),
        };
        self.pending_write = Some(json!({
            "session_id": session_id,
            "items": items,
            "before": Value::Null,
            "persisted_count": 0,
            "held": true,
            "response_id": response_id,
            "store": store,
            "reasoning_item_id_policy": Value::Null,
            "current_response": {"turn": self.current_turn, "start": prior},
        }));
    }

    /// Persist the run input and `generated_items[..limit]` that are not yet saved.
    ///
    /// A batch that was held across an interruption settles through the same append, so a call
    /// and its output land together; a call whose output a handoff filter dropped is dropped
    /// with it, except calls whose approval is still `pending`. Port of
    /// `_held_items_safe_to_settle`.
    async fn save_session_through(
        &mut self,
        limit: usize,
        pending: &HashSet<String>,
    ) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let limit = limit.min(self.generated_items.len());
        let start = self.session_saved_items.min(limit);
        let mut items = Vec::new();
        if !self.input_saved_to_session {
            items.extend(self.session_input.clone());
        }
        let held = self.pending_write.is_some();
        let mut new: Vec<InputItem> = items_to_input(&self.generated_items[start..limit]);
        // Port of `save_result_to_session`'s reasoning-id policy: a local store keeps the ids
        // the run omitted from model input out of history too; a store that owns its history
        // keeps them, because they are its own identities.
        if !session.manages_history_remotely() {
            let policy = self.reasoning_item_id_policy;
            new = new
                .into_iter()
                .map(|item| strip_reasoning_id(item, policy))
                .collect();
        }
        if held {
            new = settle_safe_items(new, pending);
        }
        items.extend(new);
        let items = items::deduplicate_input_items_preferring_latest(items);
        // A held batch was never offered to the session, so a tail that already ends with
        // exactly this batch means an earlier settle wrote it and lost the acknowledgement;
        // appending again would duplicate it. Port of the digest reconciliation in
        // `resume_pending_session_write`.
        let already_written = held
            && !items.is_empty()
            && match session.get_items(Some(items.len())).await {
                Ok(tail) => tail.len() == items.len() && tail == items,
                Err(_) => false,
            };
        if !items.is_empty() && !already_written {
            // A cancelled run (a dropped `select!` branch, an elapsed timeout) must not abort
            // a half-written session write. Port of `await_session_mutation`.
            crate::memory::await_session_mutation(async move { session.add_items(items).await })
                .await?;
        }
        self.input_saved_to_session = true;
        self.session_saved_items = self.session_saved_items.max(limit);
        if limit >= self.generated_items.len() {
            // Everything is written: the withheld batch has settled. The session now hears
            // about the response it belongs to, under the storage mode that response ran with,
            // before any further model request.
            if let Some(record) = self.pending_write.take() {
                let response_id = record
                    .get("response_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let store = record.get("store").and_then(Value::as_bool);
                if response_id.is_some() {
                    self.settle_notified = response_id.clone();
                    self.notify_session(Some((response_id, store))).await?;
                }
            }
        }
        Ok(())
    }

    /// Notify the session that the run finished (used for automatic compaction).
    async fn notify_session_completed(&mut self) -> Result<()> {
        // The response a held batch already settled for was notified when it settled.
        if let Some(settled) = &self.settle_notified
            && self
                .model_responses
                .last()
                .and_then(|r| r.response_id.as_ref())
                == Some(settled)
        {
            return Ok(());
        }
        self.notify_session(None).await
    }

    /// Tell the session a response's items are written. `held` carries the response id and
    /// storage mode a withheld batch was parked under, which the settle must keep.
    async fn notify_session(&mut self, held: Option<(Option<String>, Option<bool>)>) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let has_local_tool_outputs = held.is_some()
            || self.generated_items[self.turn_start.min(self.generated_items.len())..]
                .iter()
                .any(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }));
        // Everything the model was sent plus everything it produced this run. A compacting
        // session matches the stored tail against these digests, so it only ever replaces
        // history the model actually saw. Both spellings of a reasoning item are recorded,
        // because the stored copy may keep an id the replayed input omitted. Only sessions that
        // compact ask for this, so an ordinary session pays nothing for it.
        let mut model_exchange = Vec::new();
        if session.wants_model_exchange()
            && let Some((exchange_input, exchange_output)) = &self.compaction_exchange
        {
            let policy = self.reasoning_item_id_policy;
            // A store that owns its history assigns its own item ids, so matching on them
            // would never find the replayed item. Port of `_ignore_ids_for_matching`.
            let ignore_ids = session.manages_history_remotely();
            for item in exchange_input
                .iter()
                .cloned()
                .chain(exchange_output.iter().cloned())
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
        let (response_id, store) = match held {
            Some((response_id, store)) => (response_id, store),
            None => (
                self.model_responses
                    .last()
                    .and_then(|r| r.response_id.clone()),
                self.last_store.or(self.current_agent.model_settings.store),
            ),
        };
        let info = crate::memory::SessionRunCompletion {
            response_id,
            store,
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
        // Session-only metadata never reaches the model, and history written before the caller
        // opted into omitting reasoning ids follows the policy on read too. Port of the
        // `strip_internal_input_item_metadata` / `apply_reasoning_item_id_policy` pass in
        // `prepare_input_with_session`.
        let policy = self.reasoning_item_id_policy;
        let remote = session.manages_history_remotely();
        let history: Vec<InputItem> = history
            .into_iter()
            .map(|mut item| {
                if let Value::Object(m) = &mut item.0 {
                    m.remove("_agents_tool_description");
                    m.remove("_agents_tool_title");
                    m.remove("created_by");
                }
                if remote {
                    item
                } else {
                    strip_reasoning_id(item, policy)
                }
            })
            .collect();
        // Port of `prepare_input_with_session`: the model input is history plus the new input
        // (or what the callback made of them), and only the items that belong to the new turn
        // are persisted. Rust callbacks receive owned copies, so the callback output is matched
        // back to history and new input by content and frequency (Python also tries object
        // identity first).
        let new_input = self.session_input.clone();
        let history_len = history.len();
        let (prepared, history_indexes, output_pruning, appended) =
            match &self.config.session_input_callback {
                None => {
                    let indexes: HashSet<usize> = (0..history_len).collect();
                    let mut prepared = history;
                    prepared.extend(new_input.clone());
                    // With a limit set, outputs whose call the limit cut off go too, so the
                    // model is never sent a half of a pair.
                    let output_pruning = settings.limit.map(|_| indexes.clone());
                    (prepared, indexes, output_pruning, new_input)
                }
                Some(cb) => {
                    let combined = cb(history.clone(), new_input.clone()).await?;
                    let key = |item: &InputItem, ignore_remote_ids: bool| {
                        let item = if ignore_remote_ids {
                            sanitize_remote_history_item(item.clone())
                        } else {
                            item.clone()
                        };
                        serde_json::to_string(&item.0).unwrap_or_default()
                    };
                    let mut history_counts: HashMap<String, usize> = HashMap::new();
                    for item in &history {
                        *history_counts.entry(key(item, remote)).or_default() += 1;
                    }
                    let mut new_counts: HashMap<String, usize> = HashMap::new();
                    for item in &new_input {
                        *new_counts.entry(key(item, false)).or_default() += 1;
                    }
                    let history_keys: HashSet<String> = history_counts.keys().cloned().collect();
                    let mut indexes = HashSet::new();
                    let mut appended = Vec::new();
                    for (index, item) in combined.iter().enumerate() {
                        // New input first, then history by frequency; a further copy of a
                        // history item is still history (Python matches the repeated object by
                        // identity), anything else the callback added is new.
                        if let Some(n) = new_counts.get_mut(&key(item, false))
                            && *n > 0
                        {
                            *n -= 1;
                            appended.push(item.clone());
                            continue;
                        }
                        let history_key = key(item, remote);
                        if let Some(n) = history_counts.get_mut(&history_key)
                            && *n > 0
                        {
                            *n -= 1;
                            indexes.insert(index);
                            continue;
                        }
                        if history_keys.contains(&history_key) {
                            indexes.insert(index);
                            continue;
                        }
                        appended.push(item.clone());
                    }
                    (combined, indexes, None, appended)
                }
            };
        // A store that owns its history keeps server metadata on assistant messages the model
        // must not be sent back. Port of
        // `_sanitize_openai_conversation_history_items_for_model_input`.
        let prepared: Vec<InputItem> = if remote {
            prepared
                .into_iter()
                .enumerate()
                .map(|(index, item)| {
                    if history_indexes.contains(&index) {
                        sanitize_remote_history_item(item)
                    } else {
                        item
                    }
                })
                .collect()
        } else {
            prepared
        };
        // Stored history is pruned of calls that never got an output; the run's own input
        // stays verbatim.
        let pruned = items::drop_orphan_function_calls_in(
            prepared,
            Some(&history_indexes),
            output_pruning.as_ref(),
        );
        self.model_prefix = items::deduplicate_input_items_preferring_latest(pruned);
        self.session_input = appended;
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
            // Defaults follow the selected model, not whatever the provider reports: a model
            // instance carries no defaults, a plain name its own, and only a routed name or no
            // selection asks the resolved model. Fields the caller set on the agent's settings
            // stay on top. Port of `_model_settings_for_resolved_name`.
            let selection = self.config.model.as_ref().or(agent.model.as_ref());
            let defaults = match selection {
                Some(AgentModel::Instance(_)) => ModelSettings::default(),
                Some(AgentModel::Name(name)) if !name.contains('/') => {
                    crate::models::default_models::get_default_model_settings(Some(name))
                }
                _ => model
                    .default_model_settings()
                    .unwrap_or_else(|| match selection {
                        Some(AgentModel::Name(name)) => {
                            crate::models::default_models::get_default_model_settings(Some(name))
                        }
                        _ => agent.implicit_model_settings.clone(),
                    }),
            };
            defaults.resolve(Some(&agent.model_settings))
        };
        let mut s = base.resolve(self.config.model_settings.as_ref());
        if agent.reset_tool_choice && self.tools_used.has_used(agent) {
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
        let policy = self.reasoning_item_id_policy;
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

    /// Run the agent-start hooks once per agent activation. Port of the
    /// `should_run_agent_start_hooks` block of `run_single_turn`.
    async fn run_agent_start_hooks(&mut self) {
        if !self.should_run_agent_start_hooks {
            return;
        }
        self.should_run_agent_start_hooks = false;
        let agent = self.current_agent.clone();
        // The agent-start hooks see the run input, as `context_wrapper.turn_input` does in
        // Python; the model input replaces it once it has been built.
        self.ctx.set_turn_input(self.original_input.clone());
        if let Some(h) = &self.hooks {
            h.on_agent_start(&self.ctx, &agent).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_start(&self.ctx, &agent).await;
        }
    }

    async fn run_turn(
        &mut self,
        tools: Vec<Tool<C>>,
        handoffs: Vec<Handoff<C>>,
    ) -> Result<NextStep<C>> {
        let agent = self.current_agent.clone();
        self.run_agent_start_hooks().await;

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
        let already_ran = first_turn && std::mem::take(&mut self.first_blocking_guardrails_done);
        if !blocking.is_empty() && !already_ran {
            let results = self
                .run_input_guardrails(blocking, guardrail_input.clone().unwrap_or_default())
                .await?;
            self.record_input_guardrails(results)?;
        }

        let system_prompt = agent.get_system_prompt(&self.ctx).await?;
        let prompt = match self.prefetched_prompt.take() {
            Some(prompt) => prompt,
            None => agent.get_prompt(&self.ctx).await?,
        };
        let model = self.resolve_model(&agent)?;
        if !self.models_used.iter().any(|m| Arc::ptr_eq(m, &model)) {
            self.models_used.push(model.clone());
        }
        let settings = self.resolve_settings(&agent, &model);
        self.last_store = settings.store;

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
        // A failed request must not reuse evidence from an earlier exchange.
        self.compaction_exchange = None;

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
        // Programmatic tool calling has local side effects a replay could repeat.
        let replay_unsafe_request = tools
            .iter()
            .any(|t| matches!(t, Tool::ProgrammaticToolCalling(_)));
        let response = if parallel.is_empty() {
            Box::pin(crate::models::run_context::model_run_context(
                owner,
                model_call::call_model(&model, request, self.sink.as_ref(), replay_unsafe_request),
            ))
            .await?
        } else {
            let g_input = guardrail_input.clone().unwrap_or_default();
            let (guard_results, model_result) = {
                let model_fut = Box::pin(crate::models::run_context::model_run_context(
                    owner,
                    model_call::call_model(
                        &model,
                        request,
                        self.sink.as_ref(),
                        replay_unsafe_request,
                    ),
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
        // A response that reuses a call id for another invocation is rejected before the
        // response counts as delivered: `on_llm_end` does not run for it.
        self.validate_response_invocations(&tools, &handoffs, &response)?;
        if let Some(h) = &self.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        self.model_responses.push(response.clone());
        self.compaction_exchange = Some((
            sent_input.clone(),
            response
                .output
                .iter()
                .map(|o| InputItem(o.to_value()))
                .collect(),
        ));

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
        // A snapshot without its model response still names the calls it parked: the turn is
        // rebuilt from the pending approvals. Port of `_rebuild_function_runs_from_approvals`.
        let response = match self.model_responses.last().cloned() {
            Some(response) => response,
            None if !self.resumed_interruptions.is_empty() => ModelResponse {
                output: self
                    .resumed_interruptions
                    .iter()
                    .map(|item| {
                        let mut raw = item.raw_item.to_value();
                        if let Value::Object(map) = &mut raw {
                            map.entry("type")
                                .or_insert_with(|| Value::String("function_call".into()));
                        }
                        OutputItem::from_value(raw)
                    })
                    .collect(),
                usage: Default::default(),
                response_id: None,
                request_id: None,
                raw_usage: None,
            },
            None => return Ok(NextStep::RunAgain),
        };
        let done: HashSet<String> = self.generated_items.iter().filter_map(call_id_of).collect();
        let processed =
            self.process_response(&agent, &tools, &handoffs, &response, false, &done)?;
        // A call that started running in an earlier attempt and never committed its output
        // is not retried. Port of the `binding_status[2] and not binding_status[1]` branch of
        // `_dedupe_processed_response_invocations`.
        let pending_ids = processed
            .functions
            .iter()
            .map(|(call, _)| call.call_id.clone())
            .chain(
                processed
                    .handoffs
                    .iter()
                    .map(|(call, _)| call.call_id.clone()),
            )
            .chain(
                processed
                    .local_calls
                    .iter()
                    .map(|local| local_tools::call_id_of(local.raw())),
            );
        for call_id in pending_ids {
            if self.ctx.invocation_executed_uncommitted(&call_id) {
                return Err(AgentsError::model_behavior(
                    "A tool call already executed, but its output was not committed. \
                     Start a new run instead of retrying the invocation.",
                ));
            }
        }
        Box::pin(self.execute_side_effects(&agent, processed)).await
    }

    /// Check the parked response of a resumed turn against the run's ledger, without the agent's
    /// tools: a function, custom or hosted MCP call whose bound arguments changed is refused.
    /// Native calls are left to the live check, whose scope needs the tool they resolve to.
    fn validate_parked_response(&self) -> Result<()> {
        let Some(response) = self.model_responses.last() else {
            return Ok(());
        };
        for out in &response.output {
            let raw = out.to_value();
            let Some(call_id) = tool_invocation::invocation_call_id(&raw) else {
                continue;
            };
            let Some(record) = self.tool_invocations.get(&call_id) else {
                continue;
            };
            if self.unbound_approvals.contains(&call_id)
                || !matches!(
                    record.invocation_type.as_str(),
                    "function_call" | "custom_tool_call" | "mcp_approval_request"
                )
            {
                continue;
            }
            let matches = |role: Option<&str>| {
                tool_invocation::tool_invocation_identity(&raw, role).is_some_and(|id| {
                    id.invocation_type == record.invocation_type
                        && id.approval_scope == record.approval_scope
                        && id.fingerprint == record.fingerprint
                })
            };
            let same_type = tool_invocation::tool_invocation_identity(&raw, None)
                .is_some_and(|id| id.invocation_type == record.invocation_type);
            if same_type && !matches(None) && !matches(Some(tool_invocation::HANDOFF_ROLE)) {
                return Err(AgentsError::model_behavior(
                    "Model reused a tool call ID for a different invocation. \
                     Use a unique call ID for each tool invocation.",
                ));
            }
        }
        Ok(())
    }

    /// Reject a model response whose tool calls reuse a call id for a different invocation:
    /// against completed history, against a sibling in the same response, or against the run's
    /// ledger. Read-only; registration happens in `process_response`. Port of the validation
    /// half of `_dedupe_processed_response_invocations`.
    fn validate_response_invocations(
        &self,
        tools: &[Tool<C>],
        handoffs: &[Handoff<C>],
        response: &ModelResponse,
    ) -> Result<()> {
        let history =
            completed_history_identities(&self.generated_items, &self.tool_invocations, tools)?;
        let mut current: HashMap<String, (String, String, String)> = HashMap::new();
        // Call ids of calls that are not complete invocations (a missing payload field).
        let mut uncanonical: HashSet<String> = HashSet::new();
        let reused_in_response = || {
            AgentsError::model_behavior(
                "Model reused a tool call ID for a different invocation in one response. \
                 Use a unique call ID for each tool invocation.",
            )
        };
        for out in &response.output {
            let raw = out.to_value();
            if tool_invocation::lacks_call_id(&raw) {
                return Err(AgentsError::model_behavior(
                    "Tool invocations require a non-empty string call ID before execution.",
                ));
            }
            let ty = out.item_type();
            let role = (ty == "function_call"
                && handoffs.iter().any(|h| {
                    raw.get("name").and_then(Value::as_str) == Some(h.tool_name.as_str())
                }))
            .then_some(tool_invocation::HANDOFF_ROLE);
            let raw = invocation_identity_raw(tools, &raw);
            let ty = raw
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let Some(id) = tool_invocation::tool_invocation_identity_named(
                &raw,
                native_tool_name(tools, &ty).as_deref(),
                role,
            ) else {
                if let Some(call_id) = tool_invocation::invocation_call_id(&raw)
                    && ty != "mcp_approval_request"
                {
                    if self.tool_invocations.contains_key(&call_id)
                        && !self.unbound_approvals.contains(&call_id)
                    {
                        return Err(AgentsError::model_behavior(
                            "Model reused a tool call ID for a different invocation. \
                             Use a unique call ID for each tool invocation.",
                        ));
                    }
                    if current.contains_key(&call_id) || !uncanonical.insert(call_id) {
                        return Err(reused_in_response());
                    }
                }
                continue;
            };
            if uncanonical.contains(&id.call_id) {
                return Err(reused_in_response());
            }
            let identity = (
                id.invocation_type.clone(),
                id.approval_scope.clone(),
                id.fingerprint.clone(),
            );
            if !history.contains_key(&id.call_id)
                && self.ctx.invocation_executed_uncommitted(&id.call_id)
            {
                return Err(AgentsError::model_behavior(
                    "A tool call already executed, but its output was not committed. \
                     Start a new run instead of retrying the invocation.",
                ));
            }
            if let Some(previous) = history.get(&id.call_id) {
                if *previous != identity {
                    return Err(AgentsError::model_behavior(
                        "Model reused a completed tool call ID for a different invocation. \
                         Use a unique call ID for each tool invocation.",
                    ));
                }
                continue;
            }
            match current.get(&id.call_id) {
                Some(previous) if *previous != identity => {
                    return Err(reused_in_response());
                }
                Some(_) => continue,
                None => {
                    current.insert(id.call_id.clone(), identity);
                }
            }
            if let Some(record) = self.tool_invocations.get(&id.call_id)
                && !self.unbound_approvals.contains(&id.call_id)
                && (record.invocation_type != id.invocation_type
                    || record.approval_scope != id.approval_scope
                    || record.fingerprint != id.fingerprint)
            {
                return Err(AgentsError::model_behavior(
                    "Model reused a tool call ID for a different invocation. \
                     Use a unique call ID for each tool invocation.",
                ));
            }
        }
        Ok(())
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
            completed_functions: Vec::new(),
            handoffs: Vec::new(),
            mcp_approvals: Vec::new(),
            local_calls: Vec::new(),
            tools_used: Vec::new(),
            last_message: None,
            has_tool_activity: false,
            call_positions: HashMap::new(),
            unmatched_pending: Vec::new(),
        };
        let function_map = function_tool_lookup_map(tools)?;
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
        let history = if append {
            completed_history_identities(&self.generated_items, &self.tool_invocations, tools)?
        } else {
            HashMap::new()
        };
        // Calls that replay a completed invocation are dropped, and so is the reasoning that
        // leads up to them. Port of the `dropped_reasoning_indexes` pass.
        let replayed: HashSet<usize> = response
            .output
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                tool_invocation::tool_invocation_identity(
                    &invocation_identity_raw(tools, &o.to_value()),
                    None,
                )
                .is_some_and(|id| history.contains_key(&id.call_id))
            })
            .map(|(i, _)| i)
            .collect();
        let dropped_reasoning: HashSet<usize> = (0..response.output.len())
            .filter(|&i| matches!(response.output[i], OutputItem::Reasoning(_)))
            .filter(|&i| {
                (i + 1..response.output.len())
                    .find(|&j| !matches!(response.output[j], OutputItem::Reasoning(_)))
                    .is_some_and(|j| replayed.contains(&j))
            })
            .collect();

        for (out_index, out) in response.output.iter().enumerate() {
            if dropped_reasoning.contains(&out_index) {
                continue;
            }
            let raw_out = out.to_value();
            let out_type = out.item_type();
            if !append
                && let Some(call_id) = tool_invocation::invocation_call_id(&raw_out)
                && self.resumed_pending_approvals.contains(&call_id)
                && !skip.contains(&call_id)
                && !call_has_tool(tools, &handoff_map, &function_map, &raw_out)
                && let Some(pending) = self
                    .resumed_interruptions
                    .iter()
                    .find(|i| i.call_id() == call_id)
                // Only an undecided call stays pending; a decided one is resolved below.
                && (self.unbound_approvals.contains(&call_id)
                    || self
                        .ctx
                        .approval_status_for_item(pending, &call_id)
                        .approved
                        .is_none())
            {
                processed.has_tool_activity = true;
                processed.unmatched_pending.push(pending.clone());
                continue;
            }
            // One call id must name one invocation for the whole run. Port of
            // `RunContextWrapper._tool_invocation_status`'s binding step.
            let role = (out_type == "function_call"
                && raw_out
                    .get("namespace")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                && handoff_map.contains_key(
                    raw_out
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ))
            .then_some(tool_invocation::HANDOFF_ROLE);
            if tool_invocation::lacks_call_id(&raw_out) {
                return Err(AgentsError::model_behavior(
                    "Tool invocations require a non-empty string call ID before execution.",
                ));
            }
            let identity_raw = invocation_identity_raw(tools, &raw_out);
            let identity_type = identity_raw
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let native_name = native_tool_name(tools, &identity_type);
            if let Some(id) = tool_invocation::tool_invocation_identity_named(
                &identity_raw,
                native_name.as_deref(),
                role,
            ) {
                // An exact replay of a call whose output is already committed is dropped: it
                // is neither recorded nor executed again.
                if append && history.contains_key(&id.call_id) {
                    // The model still asked for a tool, so the turn runs again rather than
                    // ending on an empty response.
                    processed.has_tool_activity = true;
                    continue;
                }
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
                self.tools_used.add(agent, &processed.tools_used);
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
                    // Namespaced calls never resolve to a handoff. Port of
                    // `is_handoff_tool_call`.
                    let bare = fc
                        .extra
                        .get("namespace")
                        .and_then(Value::as_str)
                        .is_none_or(str::is_empty);
                    if let Some(h) = handoff_map.get(fc.name.as_str()).filter(|_| bare) {
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
                            )
                            .with_tool_details(Some(t.description.clone()), t.mcp_title.clone())
                            .with_tool_origin(Some(t.tool_origin()));
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed.tools_used.push(fc.name.clone());
                        let next = processed.call_positions.len();
                        processed.call_positions.insert(fc.call_id.clone(), next);
                        if skip.contains(&fc.call_id) {
                            processed
                                .completed_functions
                                .push((fc.clone(), (*t).clone()));
                        }
                        if !skip.contains(&fc.call_id) {
                            processed.functions.push((fc.clone(), (*t).clone()));
                        }
                    } else if routes_to_apply_patch(tools, &raw_out) {
                        // A function call to `apply_patch` runs on the apply_patch tool.
                        let pseudo = apply_patch_fallback_call(&raw_out)?;
                        let Some(local) =
                            match_local_call(tools, &pseudo, "apply_patch_call", &agent.name)?
                        else {
                            continue;
                        };
                        if !seen_invocations
                            .insert(("apply_patch_call".to_string(), fc.call_id.clone()))
                        {
                            continue;
                        }
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCall {
                                    raw_item: local.raw().clone(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed.tools_used.push(local.tool_name());
                        let next = processed.call_positions.len();
                        processed.call_positions.insert(fc.call_id.clone(), next);
                        if !skip.contains(&fc.call_id) {
                            processed.local_calls.push(local);
                        }
                    } else {
                        match self.config.tool_not_found_behavior {
                            ToolNotFoundBehavior::RaiseError => {
                                let qualified = match fc
                                    .extra
                                    .get("namespace")
                                    .and_then(Value::as_str)
                                    .filter(|ns| !ns.is_empty())
                                {
                                    Some(ns) => format!("{ns}.{}", fc.name),
                                    None => fc.name.clone(),
                                };
                                return Err(AgentsError::model_behavior(format!(
                                    "Tool {qualified} not found in agent {}",
                                    agent.name
                                )));
                            }
                            ToolNotFoundBehavior::ReturnErrorToModel => {
                                processed.has_tool_activity = true;
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
                    let sanitized = sanitize_local_item(ty, v);
                    let v = sanitized.as_ref().unwrap_or(v);
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
                        // Server-side tool search is tool activity: a turn holding only it runs
                        // again (it is still kept out of tool-use tracking below).
                        processed.has_tool_activity = true;
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
                    if !append
                        && ty == "custom_tool_call"
                        && !routes_to_apply_patch(tools, v)
                        && let Some(tool_name) = v.get("name").and_then(Value::as_str)
                        && !tools
                            .iter()
                            .any(|t| matches!(t, Tool::Custom(c) if c.name == tool_name))
                    {
                        // A parked custom call whose tool is gone: an undecided call stays
                        // pending and a rejected one still answers the model, but an approved
                        // one has nothing to run. Port of the resume branch of
                        // `resolve_interrupted_turn`.
                        let call_id = local_tools::call_id_of(v);
                        if skip.contains(&call_id) {
                            continue;
                        }
                        if tools::bound_approval_status(
                            &self.ctx,
                            &self.unbound_approvals,
                            &agent.name,
                            tool_name,
                            &call_id,
                        ) == Some(true)
                        {
                            return Err(AgentsError::model_behavior(format!(
                                "Tool {tool_name} not found in agent {}",
                                agent.name
                            )));
                        }
                        let stand_in = Arc::new(
                            crate::tool::CustomTool::new(tool_name, "", |_c, _i| async {
                                Err(AgentsError::model_behavior("Custom tool is not available."))
                            })
                            .with_needs_approval(true),
                        );
                        processed.has_tool_activity = true;
                        let next = processed.call_positions.len();
                        processed.call_positions.insert(call_id, next);
                        processed
                            .local_calls
                            .push(local_tools::LocalCall::Custom(v.clone(), stand_in));
                        continue;
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
                                    raw_item: local.raw().clone(),
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
                        // A hosted shell output is tool activity: a turn holding only it runs
                        // again. Port of `process_model_response`'s `shell_call_output` branch.
                        if ty == "shell_call_output" {
                            processed.has_tool_activity = true;
                            let name = tools
                                .iter()
                                .find_map(|t| match t {
                                    Tool::Shell(s) => Some(s.name.clone()),
                                    _ => None,
                                })
                                .unwrap_or_else(|| "shell".to_string());
                            processed.tools_used.push(name);
                        }
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
                    // A hosted MCP item obeys the `allowed_callers` of its server's tool config,
                    // so a program cannot drive a server that only allows direct calls. Port of
                    // the hosted MCP branches of `process_model_response`.
                    if matches!(ty, "mcp_list_tools" | "mcp_approval_request" | "mcp_call") {
                        let label = v.get("server_label").and_then(Value::as_str);
                        let allowed: Option<Vec<String>> = hosted_mcp
                            .iter()
                            .find(|t| t.server_label() == label)
                            .and_then(|t| t.tool_config.get("allowed_callers"))
                            .and_then(Value::as_array)
                            .map(|callers| {
                                callers
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect()
                            });
                        tool_caller::ensure_tool_caller_allowed(
                            v,
                            allowed.as_deref(),
                            "hosted_mcp",
                            &agent.name,
                        )?;
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
                                let mut item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::ToolCall {
                                        raw_item: v.clone(),
                                    },
                                );
                                if ty == "mcp_call"
                                    && let Some(label) =
                                        v.get("server_label").and_then(Value::as_str)
                                {
                                    item = item.with_tool_origin(Some(
                                        crate::tool::ToolOrigin::mcp(label),
                                    ));
                                    let tool = v.get("name").and_then(Value::as_str);
                                    let listed = self
                                        .generated_items
                                        .iter()
                                        .filter_map(|i| match &i.kind {
                                            RunItemKind::MCPListTools { raw_item } => {
                                                Some(raw_item)
                                            }
                                            _ => None,
                                        })
                                        .chain(response_raw.iter());
                                    if let Some(tool) = tool
                                        && let Some((description, title)) =
                                            hosted_mcp_tool_metadata(listed, label, tool)
                                    {
                                        item = item.with_tool_details(description, title);
                                    }
                                }
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
            self.tools_used.add(agent, &processed.tools_used);
        }
        Ok(processed)
    }

    /// Execute the tools of a processed response. A failure anywhere in the turn keeps only its
    /// accepted part in history, persisted before the error propagates. Port of the
    /// `_publish_completed_tools` guard of `execute_tools_and_side_effects`.
    async fn execute_side_effects(
        &mut self,
        agent: &Arc<Agent<C>>,
        processed: ProcessedResponse<C>,
    ) -> Result<NextStep<C>> {
        let outputs_start = self.generated_items.len();
        let model_outputs_start = self.model_items.len();
        self.failed_after_output = false;
        let result = Box::pin(self.execute_side_effects_inner(agent, processed)).await;
        if result.is_err() {
            self.publish_completed_tools(outputs_start, model_outputs_start)
                .await;
        }
        result
    }

    async fn publish_completed_tools(&mut self, outputs_start: usize, model_outputs_start: usize) {
        // A failure after the selected call's own output was committed keeps the turn whole: the
        // retry re-processes the stored response and skips the completed calls.
        if self.failed_after_output {
            return;
        }
        let outputs_start = outputs_start.min(self.generated_items.len());
        let turn_start = self.turn_start.min(outputs_start);
        let turn = &self.generated_items[turn_start..outputs_start];
        // Only a fresh turn that called tools; a resumed approval turn recovers on its own.
        let called_tools = turn.iter().any(|i| {
            matches!(
                i.kind,
                RunItemKind::ToolCall { .. } | RunItemKind::HandoffCall { .. }
            )
        });
        let resumed = turn
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolApproval { .. }));
        if !called_tools || resumed || self.server_managed() {
            return;
        }
        // Output that could still become guarded terminal output stays private.
        let private = self.output_guardrails_gate_applies();
        let retain = |items: &mut Vec<RunItem>, start: usize, outputs_at: usize| {
            let outputs_at = outputs_at.min(items.len());
            let outputs: Vec<RunItem> = items
                .split_off(outputs_at)
                .into_iter()
                .filter(|i| {
                    matches!(
                        i.kind,
                        RunItemKind::ToolCallOutput { .. }
                            | RunItemKind::MCPApprovalResponse { .. }
                    )
                })
                .collect();
            let model = items.split_off(start.min(outputs_at));
            if private {
                // Tool outputs stay private, but hosted approval decisions are not guarded output
                // and are kept with their order.
                let is_approval = |i: &RunItem| {
                    matches!(i.kind, RunItemKind::MCPApprovalResponse { .. })
                        || i.to_input_item().0.get("type").and_then(Value::as_str)
                            == Some("mcp_approval_request")
                };
                items.extend(
                    model
                        .iter()
                        .chain(outputs.iter())
                        .filter(|i| is_approval(i))
                        .cloned(),
                );
            } else {
                let retained = items::completed_tool_step_items(items, &model, &outputs);
                items.extend(retained);
            }
        };
        retain(&mut self.generated_items, turn_start, outputs_start);
        let model_turn_start = model_outputs_start.saturating_sub(outputs_start - turn_start);
        retain(&mut self.model_items, model_turn_start, model_outputs_start);
        // A sibling or batch failure settles the turn: a retry continues with the next model
        // call. A failure after this call's output was committed keeps the turn in flight.
        if !self.failed_after_output {
            self.turn_in_flight = false;
        }
        // The input (and earlier turns) is written on its own first, as Python persists it
        // before the turn, so a failed write of the retained items cannot take it down too; a
        // failed write never replaces the original error.
        if self
            .save_session_through(turn_start, &HashSet::new())
            .await
            .is_ok()
            && self.generated_items.len() > turn_start
            && self.save_session().await.is_err()
        {
            // The write may have landed before it failed; reconcile against the session tail
            // so a retry does not append the batch twice.
            self.reconcile_failed_session_write().await;
        }
    }

    async fn execute_side_effects_inner(
        &mut self,
        agent: &Arc<Agent<C>>,
        mut processed: ProcessedResponse<C>,
    ) -> Result<NextStep<C>> {
        self.last_processed_response = Some(processed.snapshot());
        self.turn_in_flight = true;
        let mut interruptions: Vec<ToolApprovalItem> =
            std::mem::take(&mut processed.unmatched_pending);
        let mut ran_functions = !processed.functions.is_empty();
        let outputs_start = self.generated_items.len();
        let model_outputs_start = self.model_items.len();
        let call_positions = std::mem::take(&mut processed.call_positions);

        // Function tools. A batch that fails still reports the outputs that were committed, so
        // they land in history (and in the failure checkpoint) before the error propagates.
        let (results, tool_failure) = {
            let env = ToolEnv {
                ctx: &self.ctx,
                agent,
                hooks: self.hooks.as_ref(),
                config: &self.config,
                unbound_approvals: &self.unbound_approvals,
                pending_approvals: &self.resumed_pending_approvals,
            };
            match Box::pin(tools::execute_function_tools(&env, processed.functions)).await {
                Ok(results) => (results, None),
                Err(failure) => {
                    self.failed_after_output = failure.failure_committed;
                    (failure.completed, Some(failure.error))
                }
            }
        };
        let mut function_results = Vec::new();
        for r in results {
            self.tool_input_guardrail_results
                .extend(r.input_guardrail_results);
            self.tool_output_guardrail_results
                .extend(r.output_guardrail_results);
            let custom_data = r.custom_data.clone();
            let origin = r.tool_origin.clone();
            match r.outcome {
                tools::ToolOutcome::Output { output, item } => {
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolCallOutput {
                            raw_item: item,
                            output: output.to_json_value(),
                            custom_data: custom_data.clone(),
                        },
                    )
                    .with_tool_origin(Some(origin));
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
                        )
                        .with_tool_origin(it.tool_origin.clone());
                        self.generated_items.push(ri);
                    }
                    interruptions.extend(items);
                }
            }
        }
        if let Some(error) = tool_failure {
            // Replay the committed outputs in the order the model asked for them, then fail.
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
            return Err(error);
        }
        if self.resumed_from_failure {
            // The failed turn already ran these calls; their outputs still count towards the
            // tool-use decision, as they did when they ran.
            for (call, tool) in std::mem::take(&mut processed.completed_functions) {
                let Some(item) = self.generated_items.iter().rev().find(|i| {
                    matches!(
                        &i.kind,
                        RunItemKind::ToolCallOutput { raw_item, .. }
                            if raw_item
                                .get("call_id")
                                .and_then(Value::as_str)
                                == Some(call.call_id.as_str())
                    )
                }) else {
                    continue;
                };
                let RunItemKind::ToolCallOutput {
                    output,
                    custom_data,
                    ..
                } = &item.kind
                else {
                    continue;
                };
                function_results.push(crate::tool::FunctionToolResult {
                    tool_name: tool.name.clone(),
                    qualified_tool_name: tool.qualified_name(),
                    custom_data: custom_data.clone(),
                    call_id: call.call_id.clone(),
                    output: match output {
                        Value::String(text) => ToolOutput::Text(text.clone()),
                        other => ToolOutput::Json(other.clone()),
                    },
                    run_item: Some(item.clone()),
                    interruptions: Vec::new(),
                });
            }
            function_results.sort_by_key(|r| {
                call_positions
                    .get(&r.call_id)
                    .copied()
                    .unwrap_or(usize::MAX)
            });
            ran_functions |= !function_results.is_empty();
            self.resumed_from_failure = false;
        }

        // Hosted MCP approval requests.
        let mut answered_mcp = false;
        // Exact same-id siblings share one decision: the callback runs once and one response
        // is emitted. Changed siblings were already refused by the invocation validation.
        let mut answered_requests: HashSet<String> = HashSet::new();
        for (request, tool) in processed.mcp_approvals {
            let request_id = request
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !answered_requests.insert(request_id.clone()) {
                continue;
            }
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
            // A request missing the fields that bind it to one invocation (its arguments or
            // server label) cannot carry any decision: it always goes to a person.
            let bindable = tool_invocation::tool_invocation_identity(&request, None).is_some();
            let callback = tool
                .as_ref()
                .and_then(|t| t.on_approval_request().cloned())
                .filter(|_| bindable);
            let decision = match callback {
                None if !bindable => None,
                Some(cb) => {
                    let r = cb(MCPToolApprovalRequest {
                        ctx: self.ctx.clone(),
                        data: request.clone(),
                    })
                    .await;
                    Some((r.approve, r.reason))
                }
                // A restored decision without a ledger binding authorizes nothing.
                None if self.unbound_approvals.contains(&request_id) => None,
                None => {
                    let status =
                        self.ctx
                            .hosted_mcp_decision(&agent.name, &label, &tool_name, &request_id);
                    status.approved.map(|a| (a, status.rejection_message))
                }
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
                    // A program-owned request is answered for the same program.
                    let resp = ItemHelpers::copy_tool_call_caller(&request, resp);
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
                        tool_origin: None,
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
                    pending_approvals: &self.resumed_pending_approvals,
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
                local_tools::LocalOutcome::OutputThenError {
                    raw_item,
                    output,
                    error,
                } => {
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolCallOutput {
                            raw_item,
                            output,
                            custom_data: None,
                        },
                    );
                    self.push_item(ri, Some(RunItemStreamEventName::ToolOutput));
                    return Err(error);
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

        // Final output from the last message, as `execute_tools_and_side_effects` does: a turn
        // that only produced hosted tool activity runs again; an empty turn ends the run.
        let message = processed.last_message;
        if message.is_none() && (!processed.tools_used.is_empty() || processed.has_tool_activity) {
            return Ok(NextStep::RunAgain);
        }
        let refusal = message.as_ref().and_then(|m| m.refusal());
        if let Some(refusal) = refusal {
            let err = AgentsError::ModelRefusalError {
                refusal,
                run_data: None,
            };
            let handler = self.error_handlers.model_refusal.clone();
            return self.handled_final(handler, err).await;
        }
        let text = message
            .as_ref()
            .map(|m| m.text_content())
            .unwrap_or_default();
        match &agent.output_type {
            Some(schema) if !schema.is_plain_text() => {
                if text.is_empty() {
                    let handler = self.error_handlers.invalid_final_output.clone();
                    if handler.is_none() {
                        return Ok(NextStep::RunAgain);
                    }
                    let err = AgentsError::model_behavior(
                        "Model returned no final output for the structured output type.",
                    );
                    return self.handled_final(handler, err).await;
                }
                match schema.validate_json(&text) {
                    Ok(v) => Ok(NextStep::FinalOutput(v)),
                    Err(e) => {
                        let handler = self.error_handlers.invalid_final_output.clone();
                        self.handled_final(handler, e).await
                    }
                }
            }
            _ => Ok(NextStep::FinalOutput(Value::String(text))),
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
        // A handler failure, or an invalid handler output, must not expose the details of a
        // data-redacted error.
        match h(input).await.map_err(|e| err.redact_follow_up(e))? {
            None => Err(err),
            Some(res) => {
                // The model's own message stays; a kept handler output is appended as a new
                // message, as `turn_resolution` does for refusals and invalid outputs.
                let (output, item) = self
                    .synthesize_handler_output(&res)
                    .map_err(|e| err.redact_follow_up(e))?;
                if let Some(item) = item {
                    self.push_item(item, Some(RunItemStreamEventName::MessageOutputCreated));
                }
                Ok(NextStep::FinalOutput(output))
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
        self.ctx.mark_invocation_executed(&call.call_id);
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

        let pre_handoff_count = self
            .resumed_pre_count
            .unwrap_or(self.turn_start)
            .min(self.generated_items.len());
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
                    "Server-managed conversations do not support handoff input filters. Remove \
                     Handoff.input_filter or RunConfig.handoff_input_filter, or disable \
                     conversation_id, previous_response_id, and auto_previous_response_id.",
                ));
            }
            // Tag every item so the loop can tell which ones the filter kept: an item it
            // dropped from history that was already saved cannot be unsaved, but the unsaved
            // tail must follow what the filter decided to keep.
            for item in &mut self.generated_items {
                if item.occurrence.0.is_none() {
                    self.next_occurrence += 1;
                    item.occurrence.0 = Some(self.next_occurrence);
                }
            }
            let saved_keys: HashSet<u64> = self.generated_items
                [..self.session_saved_items.min(self.generated_items.len())]
                .iter()
                .filter_map(|i| i.occurrence.0)
                .collect();
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
            // The saved prefix is whatever the filter kept of what was already written; the
            // first item it did not carry over (a copy it made, or a survivor of the held
            // batch) starts the unsaved tail.
            self.session_saved_items = generated
                .iter()
                .position(|i| i.occurrence.0.is_none_or(|k| !saved_keys.contains(&k)))
                .unwrap_or(generated.len());
            self.generated_items = generated;
            self.model_items = model_items;
            self.model_history_diverged = true;
            self.nested_model_start = None;
            self.nested_history_owned_items.clear();
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
                if self.config.handoff_history_mapper.is_none() {
                    // Items the default history forwards verbatim now live in the run input;
                    // the full-history view must not replay them a second time.
                    let offset = self
                        .generated_items
                        .len()
                        .saturating_sub(self.model_items.len());
                    self.nested_history_owned_items.extend(
                        history::forwarded_item_indexes(&self.model_items, model_pre_count)
                            .into_iter()
                            .map(|index| offset + index),
                    );
                }
                self.model_prefix = match &self.config.handoff_history_mapper {
                    Some(m) => {
                        let mut transcript = self.model_prefix.clone();
                        transcript.extend(normalized.into_iter().map(|(item, _)| item));
                        m(transcript)
                    }
                    None => history::build_ordered_history(self.model_prefix.clone(), normalized),
                };
                // The nested history is the next agent's input, as Python's
                // `original_input = nested.input_history`.
                self.original_input = self.model_prefix.clone();
                self.model_items.clear();
                self.nested_model_start = Some(self.generated_items.len());
                self.model_history_diverged = true;
                self.server_sent_items = 0;
            }
        }
        Ok(NextStep::Handoff(new_agent))
    }
}
