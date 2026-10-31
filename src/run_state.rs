//! Serializable snapshot of a run, used to resume after approvals. Mirrors `agents.run_state`.
//!
//! Serialized state contains approval decisions, pending tool calls and arguments. Only
//! deserialize snapshots from trusted storage: `from_json` does not authenticate the snapshot.
//! The app context `C` is not serialized automatically; pass it again in `RunOptions` when
//! resuming, or store your own serialized copy in [`RunState::context`].

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AgentsError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputItem, ModelResponse, RunItem, ToolApprovalItem};
use crate::run_context::ApprovalsState;
use crate::tool_guardrails::{ToolInputGuardrailResult, ToolOutputGuardrailResult};
use crate::tracing::TraceState;
use crate::usage::Usage;

mod nested_history;
mod sandbox_payload;
mod wire;

pub use wire::{SUPPORTED_SCHEMA_VERSIONS, ToolUseTrackerSnapshot};

/// Current serialization schema version. Matches the Python SDK's `agents.run_state`
/// schema, so snapshots are interoperable between the two SDKs.
pub const CURRENT_SCHEMA_VERSION: &str = "1.21";

/// The flat schema this crate wrote before it adopted the Python envelope.
/// [`RunState::from_json`] still reads it.
pub const LEGACY_RUST_SCHEMA_VERSION: &str = "1.0";

/// The first schema version that binds every approval decision to the tool-invocation ledger.
///
/// From this version on, a per-call approval whose call id has no ledger entry no longer
/// authorizes anything: the call is put back to the user for a fresh decision. Mirrors the
/// `(schema_major, schema_minor) >= (1, 15)` gate in Python's `_deserialize_run_state`, which
/// is also where Python decides whether to rebuild `RunContextWrapper._tool_invocations` at
/// all. Snapshots below it, and the flat Rust `"1.0"` snapshots this crate used to write,
/// predate the ledger and keep the old tolerant behaviour.
pub const STRICT_APPROVAL_BINDING_MIN_SCHEMA_VERSION: (u32, u32) = (1, 15);

/// A resumable snapshot of a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    #[serde(rename = "$schemaVersion")]
    pub schema_version: String,
    /// Number of model turns completed.
    pub current_turn: usize,
    /// Turn limit of the run.
    pub max_turns: Option<usize>,
    /// Name of the agent that was active when the run stopped.
    pub current_agent_name: String,
    /// Identity key of that agent (`name#2`, ...) when another agent in the graph shares its
    /// name; `None` when the name alone identifies it.
    #[serde(default)]
    pub current_agent_identity: Option<String>,
    /// The original input of the run.
    pub original_input: Vec<InputItem>,
    /// Items generated so far.
    pub generated_items: Vec<RunItem>,
    /// Raw model responses so far.
    pub model_responses: Vec<ModelResponse>,
    /// Usage accumulated so far.
    pub usage: Usage,
    /// Approval decisions.
    pub approvals: ApprovalsState,
    /// Paused nested `Agent::as_tool` runs keyed by tool call id.
    #[serde(default)]
    pub nested_agent_tool_states: BTreeMap<String, Value>,
    #[serde(default)]
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    #[serde(default)]
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    #[serde(default)]
    pub tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    #[serde(default)]
    pub tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
    /// The last model response still has tool calls waiting for approval.
    #[serde(default)]
    pub pending_tool_processing: bool,
    /// Pending approvals.
    #[serde(default)]
    pub interruptions: Vec<ToolApprovalItem>,
    /// Input staged with [`RunState::add_input`] for the next model call.
    #[serde(default)]
    pub pending_input: Vec<InputItem>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub auto_previous_response_id: bool,
    /// Generated items already sent to a server-managed conversation.
    #[serde(default)]
    pub server_sent_items: usize,
    /// Whether the original input was sent to a server-managed conversation.
    #[serde(default)]
    pub server_input_sent: bool,
    /// Generated items already persisted to the session.
    #[serde(default)]
    pub session_saved_items: usize,
    /// Whether the input was persisted to the session.
    #[serde(default)]
    pub input_saved_to_session: bool,
    /// The final output, when the run finished.
    #[serde(default)]
    pub final_output: Option<Value>,
    /// Trace id of the run.
    #[serde(default)]
    pub trace_id: Option<String>,
    /// Full trace metadata of the run (workflow name, group id, metadata). A resumed run
    /// restores the settings the snapshot was traced with unless `RunConfig` overrides them.
    /// Mirrors `RunState._trace_state`.
    #[serde(default)]
    pub trace_state: Option<TraceState>,
    /// Optional application context, serialized by the application.
    #[serde(default)]
    pub context: Option<Value>,
    /// Index of the first item generated by the last model turn.
    #[serde(default)]
    pub turn_start_index: usize,
    /// Index into `generated_items` where the model's own history starts: after a nested
    /// handoff the earlier items live in `original_input` as a summary and are not replayed.
    /// `0` replays every item. Mirrors the `generated_items` / `session_items` split of
    /// Python's `RunState`.
    #[serde(default)]
    pub model_input_start: usize,
    /// Tools each agent used so far, so `reset_tool_choice` survives a resume.
    /// Mirrors `serialize_tool_use_tracker` / `hydrate_tool_use_tracker`.
    #[serde(default)]
    pub tool_use_tracker: ToolUseTrackerSnapshot,
    /// The snapshot parked on approvals, but its current-response boundary cannot be proven from
    /// the serialized ownership: the generated items the parked response owns are not an
    /// exactly recorded terminal suffix of the history, or a legacy writer recorded no
    /// ownership at all. Resuming such a state under output guardrails is refused, because the
    /// guardrail could not tell which items belong to the response it must withhold. Mirrors
    /// the `proven` flag of Python's `_current_response_boundary`. Derived on read.
    #[serde(skip)]
    pub current_response_boundary_unproven: bool,
    /// Structured input of the `Agent::as_tool` invocation the run belongs to. Mirrors
    /// `RunContextWrapper.tool_input`, which Python writes at `context.tool_input`: a resumed
    /// run whose context carries none inherits it.
    #[serde(default)]
    pub tool_input: Option<Value>,
    /// Indexes into [`generated_items`](Self::generated_items) of items a nested handoff
    /// already carried into `original_input` verbatim, so the full-history view does not replay
    /// them twice. Mirrors `RunState._nested_history_owned_session_item_refs`; the wire form
    /// records each as `{index, digest, input_index}`.
    #[serde(default)]
    pub nested_history_owned_items: Vec<usize>,
    /// The sandbox resume payload of a Python snapshot, kept as opaque JSON so re-saving a
    /// Python-written state loses nothing. Mount authority is removed on every read and write.
    /// Rust keeps its own sandbox state in `SandboxSessionState`, so nothing here is written by
    /// the Rust runner.
    #[serde(default)]
    pub sandbox: Option<Value>,
    /// Prompt cache key generated by the SDK for this run.
    #[serde(default)]
    pub generated_prompt_cache_key: Option<String>,
    /// How reasoning item ids were handled when the run built model input. Mirrors
    /// `RunState._reasoning_item_id_policy`: a resume whose `RunConfig` keeps the default
    /// policy continues with the one recorded here. Travels only in the envelope.
    #[serde(skip)]
    pub reasoning_item_id_policy: Option<crate::run_config::ReasoningItemIdPolicy>,
    /// Server-managed conversation bookkeeping, so a resume neither resends nor skips items.
    #[serde(default)]
    pub server_conversation: ServerConversationState,
    /// The run ended at a boundary a resume cannot reproduce (a terminal output whose
    /// persistence failed, or a handoff an input guardrail rejected). Mirrors
    /// `RunState._terminal_unrecoverable`: resuming such a snapshot is refused. Rust never
    /// produces one — a failed terminal save returns an error rather than a snapshot — but a
    /// Python-written snapshot can carry it.
    #[serde(default)]
    pub terminal_unrecoverable: bool,
    /// The session write an interruption withheld while an output guardrail could still reject
    /// the turn. Mirrors `RunState._pending_session_write`: a record naming the owning session,
    /// the withheld items and the response boundary. The items it names are also the unsaved
    /// tail of `generated_items`, so a resume settles them together with the resolved turn.
    #[serde(default)]
    pub pending_session_write: Option<Value>,
    /// Canonical identity of every tool call the run has seen, keyed by call id. Mirrors
    /// `RunContextWrapper._tool_invocations`: it binds an approval decision to the exact call
    /// it was given for, so a replay with mutated arguments is rejected rather than reused.
    #[serde(default)]
    pub tool_invocations: BTreeMap<String, ToolInvocationRecord>,
    /// What the run loop made of the model response that produced the pending interruptions.
    /// Mirrors `RunState._last_processed_response`; `None` unless the run stopped at an
    /// interruption, which is also when Python writes it.
    #[serde(default)]
    pub last_processed_response: Option<ProcessedResponseSnapshot>,
    /// Call ids whose restored approval decision no longer authorizes the call, because the
    /// snapshot's [`tool_invocations`](Self::tool_invocations) ledger has no entry binding the
    /// decision to a concrete invocation. Resuming asks for those approvals again.
    ///
    /// Mirrors `RunContextWrapper._restored_unbound_approval_call_ids`. It is derived from the
    /// snapshot on read rather than stored in it, so it never travels on the wire;
    /// [`approve`](Self::approve) and [`reject`](Self::reject) clear an id as soon as a fresh
    /// decision replaces the unbound one, the way Python's `_apply_approval_decision` does.
    #[serde(skip)]
    pub restored_unbound_approval_call_ids: BTreeSet<String>,
    /// The snapshot's `context_meta` says its context cannot be rebuilt from the payload
    /// alone: it was written from a type the JSON does not describe, or omitted outright.
    /// Mirrors `_context_meta_requires_deserializer`. Read-only; derived on deserialization.
    #[serde(skip)]
    pub context_requires_restoration: bool,
}

/// How [`RunState::from_json_with`] treats a snapshot whose context cannot be restored from
/// the payload alone. Mirrors the `context_override` / `strict_context` arguments of Python's
/// `RunState.from_string`.
///
/// Rust never serializes the application context automatically — the application stores its own
/// JSON on [`RunState::context`] — so there is no `context_deserializer` equivalent: rebuilding
/// a value from the stored JSON is the application's own deserialization step. What does carry
/// over is the *decision*: a snapshot whose context is missing or was written from a type the
/// JSON does not describe must not resume as though its context were intact.
#[derive(Debug, Clone, Default)]
pub struct ContextRestoreOptions {
    /// Context to install instead of the serialized one. Supplying it satisfies
    /// [`strict_context`](Self::strict_context).
    pub context_override: Option<Value>,
    /// Fail instead of warning when the snapshot's context needs explicit restoration and no
    /// override was given.
    pub strict_context: bool,
}

impl ContextRestoreOptions {
    /// Require a restorable context, or an override.
    pub fn strict() -> Self {
        Self {
            context_override: None,
            strict_context: true,
        }
    }

    /// Use `context` instead of the serialized one.
    pub fn with_override(mut self, context: Value) -> Self {
        self.context_override = Some(context);
        self
    }
}

/// One entry of the tool-invocation ledger. Mirrors `_ToolInvocationRecord`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInvocationRecord {
    /// The invocation item type (`function_call`, `shell_call`, ...).
    #[serde(rename = "type")]
    pub invocation_type: String,
    /// SHA-256 of what the call is authorized as (its tool routing).
    pub approval_scope: String,
    /// SHA-256 of the call's full semantic payload, arguments included.
    pub fingerprint: String,
    /// The tool started running.
    pub executed: bool,
    /// The tool's output was committed to history.
    pub completed: bool,
}

/// The classified tool activity of the interrupted turn, in the shape Python's
/// `_serialize_processed_response` writes. The Python runner needs it to continue an
/// interrupted turn; Rust rebuilds execution from `model_responses.last()` and keeps this so
/// a snapshot written here stays resumable by the Python SDK.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessedResponseSnapshot {
    /// Items the turn generated before it paused, excluding the approval items themselves.
    pub new_items: Vec<RunItem>,
    /// Names of the tools the response asked for.
    pub tools_used: Vec<String>,
    /// Function-tool calls scheduled by the response.
    pub functions: Vec<ToolRunSnapshot>,
    /// Handoff calls scheduled by the response.
    pub handoffs: Vec<ToolRunSnapshot>,
    /// Computer-tool calls.
    pub computer_actions: Vec<ToolRunSnapshot>,
    /// Custom-tool calls.
    pub custom_tool_actions: Vec<ToolRunSnapshot>,
    /// Local shell calls.
    pub local_shell_actions: Vec<ToolRunSnapshot>,
    /// Shell-tool calls.
    pub shell_actions: Vec<ToolRunSnapshot>,
    /// Apply-patch calls.
    pub apply_patch_actions: Vec<ToolRunSnapshot>,
    /// Hosted-MCP approval requests, as `{request_item, mcp_tool}` payloads.
    pub mcp_approval_requests: Vec<Value>,
    /// The approvals the turn is waiting on.
    pub interruptions: Vec<ToolApprovalItem>,
    /// Hosted-MCP server labels bound to each call id.
    pub mcp_tool_bindings: BTreeMap<String, Vec<String>>,
    /// Index into `RunState::tool_output_guardrail_results` where the results of this
    /// response's still-pending tool calls start. Mirrors
    /// `ProcessedResponse.tool_output_guardrail_result_start` (schema 1.20); `None` when the
    /// snapshot does not distinguish them.
    #[serde(default)]
    pub tool_output_guardrail_result_start: Option<usize>,
}

/// One tool call a [`ProcessedResponseSnapshot`] scheduled, with the metadata the Python
/// deserializer matches against the resumed agent's tools.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolRunSnapshot {
    /// The tool's own name (not namespaced).
    pub tool_name: String,
    /// Namespace of the tool, when it has one.
    pub namespace: Option<String>,
    /// The raw tool-call item.
    pub tool_call: Value,
    /// The tool description, when the group carries one.
    pub description: Option<String>,
    /// The tool parameter schema, written for function tools only.
    pub params_json_schema: Option<Value>,
}

/// What a server-managed conversation has already seen. Mirrors the state
/// `OpenAIServerConversationTracker` hydrates from a `RunState`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConversationState {
    /// Ids of items the server already holds.
    #[serde(default)]
    pub server_item_ids: BTreeSet<String>,
    /// Call ids of tool calls the server already holds.
    #[serde(default)]
    pub server_tool_call_ids: BTreeSet<String>,
    /// Fingerprints of server-produced output items.
    #[serde(default)]
    pub server_output_fingerprints: BTreeSet<String>,
    /// Fingerprints of input items already sent to the server.
    #[serde(default)]
    pub sent_item_fingerprints: BTreeSet<String>,
}

impl RunState {
    /// Pending approvals.
    pub fn get_interruptions(&self) -> Vec<ToolApprovalItem> {
        self.interruptions.clone()
    }

    /// Approve a pending tool call. `always_approve` also approves future calls to the same
    /// tool by the same agent.
    pub fn approve(&mut self, item: &ToolApprovalItem, always_approve: bool) {
        self.approvals.approve(item, always_approve);
        self.bind_decision(item);
        self.restored_unbound_approval_call_ids
            .remove(item.call_id());
    }

    /// Reject a pending tool call, optionally with a custom model-visible message.
    pub fn reject(
        &mut self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) {
        self.approvals
            .reject(item, always_reject, rejection_message);
        self.bind_decision(item);
        self.restored_unbound_approval_call_ids
            .remove(item.call_id());
    }

    /// Bind a decision to the exact function call it was made for, as Python's
    /// `_apply_approval_decision` registers the invocation: the resumed run then refuses the
    /// call id if the model reuses it for different arguments.
    fn bind_decision(&mut self, item: &ToolApprovalItem) {
        if let Some((call_id, record)) = crate::run_context::approval_binding(item) {
            self.tool_invocations.entry(call_id).or_insert(record);
        }
    }

    /// Approve a pending tool call, failing the way Python's `RunState.approve` raises.
    ///
    /// A decision is refused, and nothing is recorded, for an item without a canonical
    /// invocation identity: a recognized tool call without a call id, a hosted MCP request
    /// without its request id (or, for a sticky decision, its server label and tool name), and
    /// an item whose type or payload cannot be fingerprinted. The call then stays pending.
    ///
    /// [`approve`](Self::approve) keeps its infallible signature and applies the same rule
    /// without reporting it.
    pub fn try_approve(&mut self, item: &ToolApprovalItem, always_approve: bool) -> Result<()> {
        check_function_decision(item, always_approve)?;
        self.approvals.try_approve(item, always_approve)?;
        self.restored_unbound_approval_call_ids
            .remove(item.call_id());
        Ok(())
    }

    /// Reject a pending tool call, failing the way Python's `RunState.reject` raises. See
    /// [`try_approve`](Self::try_approve).
    pub fn try_reject(
        &mut self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) -> Result<()> {
        check_function_decision(item, always_reject)?;
        self.approvals
            .try_reject(item, always_reject, rejection_message)?;
        self.restored_unbound_approval_call_ids
            .remove(item.call_id());
        Ok(())
    }

    /// Stage new user input for the next model call of the resumed run.
    pub fn add_input(&mut self, input: impl Into<StagedInput>) -> Result<()> {
        if self.final_output.is_some() {
            return Err(AgentsError::user(
                "Cannot add input to a finished run; start a new run instead.",
            ));
        }
        if let Some(max) = self.max_turns
            && self.current_turn >= max
        {
            return Err(AgentsError::user(
                "Cannot add input: no model turns remain.",
            ));
        }
        match input.into() {
            StagedInput::Text(t) => self.pending_input.push(InputItem::user(t)),
            StagedInput::Items(items) => self.pending_input.extend(items),
        }
        Ok(())
    }

    /// Discard staged input.
    pub fn clear_pending_input(&mut self) {
        self.pending_input.clear();
    }

    /// Serialize to JSON, in the Python SDK's `agents.run_state` envelope.
    pub fn to_json(&self) -> Result<Value> {
        wire::to_wire(self, false)
    }

    /// Serialize to JSON, keeping the explicit tracing export API key in the trace payload.
    ///
    /// Mirrors `to_json(include_tracing_api_key=True)`. [`to_json`](Self::to_json) only keeps
    /// the key's hash, so a resumed run cannot export with a different key unnoticed; opt in
    /// only when the snapshot is stored somewhere the key may live.
    pub fn to_json_with_tracing_api_key(&self) -> Result<Value> {
        wire::to_wire(self, true)
    }

    /// Serialize, refusing a context that no SDK could restore.
    ///
    /// Mirrors `to_json(strict_context=True)`. Python's deserializer requires the serialized
    /// context to be a JSON object ("Serialized run state context must be a mapping"), so a
    /// snapshot carrying a string, number or array context is one the Python SDK — and this
    /// one's own strict read — will refuse. Catching it at write time reports the problem
    /// while the run that produced it is still at hand, instead of at the resume.
    pub fn to_json_strict(&self) -> Result<Value> {
        if let Some(context) = &self.context
            && !context.is_object()
        {
            return Err(AgentsError::user(
                "RunState serialization requires context to be a JSON object when                  strict_context is set. Serialize the application context to an object before                  storing it on the run state.",
            ));
        }
        self.to_json()
    }

    /// Serialize to a JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.to_json()?)?)
    }

    /// Deserialize from JSON (only from trusted storage).
    ///
    /// Reads the Python envelope as well as the flat `"1.0"` snapshots this crate used to
    /// write.
    pub fn from_json(value: Value) -> Result<Self> {
        let version = wire::validate_schema_version(&value)?;
        let version = version.as_str();
        let is_legacy_rust = wire::is_legacy_rust_envelope(&value);
        if !is_legacy_rust {
            wire::validate_envelope(&value, version)?;
        }
        let context_meta = value
            .get("context")
            .and_then(|c| c.get("context_meta"))
            .cloned();
        let (strict_major, strict_minor) = STRICT_APPROVAL_BINDING_MIN_SCHEMA_VERSION;
        // A flat Rust `"1.0"` snapshot predates the ledger entirely: it never recorded one, so
        // every decision in it would look unbound. Only the Python-compatible envelope is held
        // to the strict rule, and only from the version that introduced the binding.
        let strict_binding =
            !is_legacy_rust && wire::schema_version_at_least(version, strict_major, strict_minor);
        let mut state = if is_legacy_rust {
            wire::from_legacy_rust(value.clone())?
        } else {
            wire::from_wire(&value, version)?
        };
        if strict_binding {
            wire::validate_completed_tool_invocations(&state, &value)?;
            state.restored_unbound_approval_call_ids = state.unbound_approval_call_ids();
        } else if !is_legacy_rust {
            wire::reconstruct_legacy_pending_bindings(&mut state);
        }
        if state.context_requires_restoration {
            // Python logs the same warning here and leaves the decision to the caller;
            // `from_json_with` turns it into an error under `strict_context`.
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "{}",
                wire::context_meta_warning(context_meta.as_ref()),
            );
        }
        // The in-memory state is always current; the version only describes the payload read.
        state.schema_version = CURRENT_SCHEMA_VERSION.to_string();
        Ok(state)
    }

    /// Call ids that no ledger entry binds to an invocation: those carrying a per-call
    /// approval decision, and every tool call restored from history. Port of
    /// `RunContextWrapper._mark_restored_unbound_approval_call_ids` plus the
    /// history-occurrence pass of `_reconcile_tool_invocations`, so not even a sticky decision
    /// can authorize a historical call id whose binding is missing.
    fn unbound_approval_call_ids(&self) -> BTreeSet<String> {
        self.approvals
            .0
            .values()
            .flat_map(|record| {
                record
                    .approved_call_ids
                    .iter()
                    .chain(record.rejected_call_ids.iter())
                    .cloned()
            })
            .chain(wire::history_tool_call_ids(self))
            .filter(|call_id| !self.tool_invocations.contains_key(call_id))
            .collect()
    }

    /// Deserialize from a JSON string (only from trusted storage).
    pub fn from_json_str(s: &str) -> Result<Self> {
        Self::from_json(wire::parse_json(s)?)
    }

    /// Deserialize, deciding what to do about a context the snapshot cannot restore.
    ///
    /// Mirrors the `context_override` / `strict_context` arguments of Python's
    /// `RunState.from_string`: a snapshot whose `context_meta` reports a custom or omitted
    /// context is an error under [`ContextRestoreOptions::strict_context`] and a warning
    /// otherwise, and an override silences both by supplying the context itself.
    ///
    /// Neither option authenticates the snapshot. [`from_json`](Self::from_json) is the
    /// non-strict form, and warns on the same condition.
    pub fn from_json_with(value: Value, options: ContextRestoreOptions) -> Result<Self> {
        let warning =
            wire::context_meta_warning(value.get("context").and_then(|c| c.get("context_meta")));
        let mut state = Self::from_json(value)?;
        if let Some(context) = options.context_override {
            state.context = Some(context);
            state.context_requires_restoration = false;
        } else if state.context_requires_restoration && options.strict_context {
            return Err(AgentsError::user(warning));
        }
        Ok(state)
    }

    /// Deserialize from a JSON string with [`ContextRestoreOptions`].
    pub fn from_json_str_with(s: &str, options: ContextRestoreOptions) -> Result<Self> {
        Self::from_json_with(wire::parse_json(s)?, options)
    }

    /// Attach a serialized application context.
    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }
}

/// The identity checks of `RunContextWrapper._apply_approval_decision` for an item that is not a
/// hosted MCP request (the hosted branch is checked by the approvals themselves): a decision is
/// recorded only for a call that has an id and a canonical invocation identity, because a
/// decision is bound to that identity.
fn check_function_decision(item: &ToolApprovalItem, always: bool) -> Result<()> {
    use crate::run_internal::tool_invocation::tool_invocation_identity;

    let raw = item.raw_item.to_value();
    let raw_type = raw.get("type").and_then(Value::as_str).unwrap_or_default();
    let wraps_mcp = raw_type == "hosted_tool_call"
        && raw
            .get("provider_data")
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str)
            == Some("mcp_approval_request");
    if raw_type == "mcp_approval_request" {
        return Ok(());
    }
    if wraps_mcp {
        // A wrapped request is bindable only when the request itself names its server and
        // tool; the wrapper's own fields never stand in for them.
        let field = |name: &str| {
            raw.get("provider_data")
                .and_then(|p| p.get(name))
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty())
        };
        return if field("name") && field("server_label") {
            Ok(())
        } else {
            Err(AgentsError::model_behavior(
                "Approval decisions require a canonical invocation identity.",
            ))
        };
    }
    let call_id = item.call_id();
    let recognized = matches!(
        raw_type,
        "function_call"
            | "apply_patch_call"
            | "computer_call"
            | "custom_tool_call"
            | "local_shell_call"
            | "shell_call"
    );
    if recognized && call_id.is_empty() {
        return Err(AgentsError::model_behavior(
            "Approval decisions require a non-empty call ID for recognized tool invocations.",
        ));
    }
    if call_id.is_empty() && !always {
        return Err(AgentsError::model_behavior(
            "Per-call approval decisions require a non-empty call ID.",
        ));
    }
    if !call_id.is_empty() && tool_invocation_identity(&raw, None).is_none() {
        return Err(AgentsError::model_behavior(
            "Approval decisions require a canonical invocation identity.",
        ));
    }
    Ok(())
}

/// Input that can be staged on a run state.
#[derive(Debug, Clone)]
pub enum StagedInput {
    Text(String),
    Items(Vec<InputItem>),
}

impl From<&str> for StagedInput {
    fn from(value: &str) -> Self {
        StagedInput::Text(value.to_string())
    }
}

impl From<String> for StagedInput {
    fn from(value: String) -> Self {
        StagedInput::Text(value)
    }
}

impl From<Vec<InputItem>> for StagedInput {
    fn from(value: Vec<InputItem>) -> Self {
        StagedInput::Items(value)
    }
}
