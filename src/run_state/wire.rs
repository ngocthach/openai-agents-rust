//! The `RunState` wire format: Python's `agents.run_state` schema 1.21 envelope.
//!
//! Rust keeps a flat in-memory [`RunState`](super::RunState); this module maps it to and from
//! the keyed envelope the Python SDK reads and writes, so a snapshot taken in one SDK can be
//! resumed in the other. Fields that only exist on the Rust side travel under the `rust` key,
//! which Python ignores.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::{
    CURRENT_SCHEMA_VERSION, ProcessedResponseSnapshot, RunState, ToolInvocationRecord,
    ToolRunSnapshot,
};
use crate::error::{AgentsError, Result};
use crate::guardrail::{GuardrailFunctionOutput, InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputItem, ModelResponse, RunItem, ToolApprovalItem};
use crate::run_context::{ApprovalRecord, ApprovalsState};
use crate::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrailResult, ToolOutputGuardrailResult,
};
use crate::usage::Usage;

/// Schema versions this crate can read, oldest first. Mirrors Python's
/// `SUPPORTED_SCHEMA_VERSIONS`.
pub const SUPPORTED_SCHEMA_VERSIONS: [&str; 22] = [
    "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12",
    "1.13", "1.14", "1.15", "1.16", "1.17", "1.18", "1.19", "1.20", "1.21",
];

/// A one-line historical summary of every supported schema version, oldest first. Mirrors
/// Python's `SCHEMA_VERSION_SUMMARIES`: every schema bump adds one entry here, and the keys are
/// exactly [`SUPPORTED_SCHEMA_VERSIONS`].
pub const SCHEMA_VERSION_SUMMARIES: [(&str, &str); 22] = [
    (
        "1.0",
        "Initial RunState snapshot format for HITL pause/resume flows.",
    ),
    (
        "1.1",
        "Same payload as 1.0, but introduces explicit backward-read support policy.",
    ),
    (
        "1.2",
        "Persists reasoning_item_id_policy for resumed and streamed follow-up turns.",
    ),
    (
        "1.3",
        "Updates resumed trace semantics to reattach traces without duplicate starts.",
    ),
    (
        "1.4",
        "Stores request_id alongside each serialized model response.",
    ),
    (
        "1.5",
        "Renumbered unreleased baseline for tool-search snapshots and richer tool metadata.",
    ),
    (
        "1.6",
        "Persists explicit approval rejection messages across resume flows.",
    ),
    (
        "1.7",
        "Persists duplicate-name agent identities across agent-owned state and sandbox resume state.",
    ),
    (
        "1.8",
        "Persists SDK-generated prompt cache keys across resume flows.",
    ),
    (
        "1.9",
        "Persists pending custom tool calls and tool origin metadata across resume flows.",
    ),
    (
        "1.10",
        "Allows serialized RunState snapshots to disable max_turns with null.",
    ),
    (
        "1.11",
        "Persists SDK-only custom data on tool output items across resume flows.",
    ),
    (
        "1.12",
        "Persists input cache-write token usage across resume flows.",
    ),
    (
        "1.13",
        "Persists programmatic tool calling and nested handoff history ownership across resume flows.",
    ),
    (
        "1.14",
        "Scopes hosted MCP approvals and restored requests by server label.",
    ),
    (
        "1.15",
        "Persists canonical tool invocation identity plus sanitized mount authority and trusted rebind metadata, durable pending input, and resumable next-model-call state.",
    ),
    (
        "1.16",
        "Persists Docker network-isolation state and lets an exact call approval decision override a sticky decision for the same tool.",
    ),
    (
        "1.17",
        "Persists Docker container labels and current-response generated-item ownership across resume flows, including pending resumed Session writes and terminal-unrecoverable runs.",
    ),
    (
        "1.18",
        "Binds restored local MCP calls to their configured server and original tool name, preserves independent apply_patch approval scopes, and binds function-tool approval decisions to their owning agent, and retains compaction metadata for pending Session writes, including acknowledgement and model-exchange evidence for compaction retry, and records named Vercel sandboxes alongside exact execution IDs.",
    ),
    (
        "1.19",
        "Persists the interrupted turn's withheld Session write as a held pending write, with the conversion policy its items were registered under and the current-response boundary, so an approval resume can settle it under the output-guardrail gate.",
    ),
    (
        "1.20",
        "Persists the pending tool output guardrail result boundary so a resumed run withholds current-response diagnostics until its terminal guardrails complete.",
    ),
    (
        "1.21",
        "Makes duplicate-agent identity independent of mutable hook instance state.",
    ),
];

/// The first schema that may carry Programmatic Tool Calling items.
const PROGRAMMATIC_TOOL_CALLING_MIN_SCHEMA_VERSION: (u32, u32) = (1, 13);
/// The first schema that may carry agent-owned `function_tool_approvals`.
const AGENT_OWNED_APPROVALS_MIN_SCHEMA_VERSION: (u32, u32) = (1, 18);
/// The first schema that may carry `tool_output_guardrail_result_start`.
const TOOL_OUTPUT_GUARDRAIL_BOUNDARY_MIN_SCHEMA_VERSION: (u32, u32) = (1, 20);

/// Tool-call item types a ledger entry may describe. Mirrors `_TOOL_INVOCATION_TYPES`.
const TOOL_INVOCATION_TYPES: [&str; 7] = [
    "apply_patch_call",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
    "mcp_approval_request",
    "shell_call",
];

/// Parse a snapshot string. Mirrors `RunState.from_string`: the parser's own message is
/// replaced, so a malformed payload never echoes its content into the error.
pub(super) fn parse_json(s: &str) -> Result<Value> {
    serde_json::from_str(s).map_err(|_| AgentsError::user("Failed to parse run state JSON"))
}

/// Python's truthiness, for the `if not schema_version` check.
fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
    }
}

/// Check the payload shape and its schema version. Port of
/// `_validate_run_state_schema_version`; the messages are Python's, and none of them echo the
/// offending value.
pub(super) fn validate_schema_version(value: &Value) -> Result<String> {
    let Value::Object(map) = value else {
        return Err(AgentsError::user("Run state JSON must be an object"));
    };
    let version = match map.get("$schemaVersion") {
        None => return Err(AgentsError::user("Run state is missing schema version")),
        Some(v) if is_falsy(v) => {
            return Err(AgentsError::user("Run state is missing schema version"));
        }
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(AgentsError::user(
                "Run state schema version has an invalid type",
            ));
        }
    };
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&version.as_str()) {
        // Python joins `sorted(SUPPORTED_SCHEMA_VERSIONS)`, a string sort.
        let mut supported = SUPPORTED_SCHEMA_VERSIONS.to_vec();
        supported.sort_unstable();
        return Err(AgentsError::user(format!(
            "Run state schema version is not supported. Supported versions are: {}. New \
             snapshots are written as version {CURRENT_SCHEMA_VERSION}.",
            supported.join(", ")
        )));
    }
    Ok(version)
}

/// Raw items from every durable location of the envelope. Port of `_run_state_raw_items`.
fn run_state_raw_items(value: &Value) -> Vec<&Value> {
    let mut raw = Vec::new();
    for key in ["original_input", "pending_input"] {
        if let Some(Value::Array(items)) = value.get(key) {
            raw.extend(items.iter());
        }
    }
    for key in ["model_responses", "last_model_response"] {
        let responses: Vec<&Value> = match value.get(key) {
            Some(Value::Array(items)) => items.iter().collect(),
            Some(v @ Value::Object(_)) => vec![v],
            _ => Vec::new(),
        };
        for response in responses {
            if let Some(Value::Array(output)) = response.get("output") {
                raw.extend(output.iter());
            }
        }
    }
    fn item_raws(items: Option<&Value>) -> Vec<&Value> {
        match items {
            Some(Value::Array(items)) => items.iter().filter_map(|i| i.get("raw_item")).collect(),
            _ => Vec::new(),
        }
    }
    for key in ["generated_items", "session_items"] {
        raw.extend(item_raws(value.get(key)));
    }
    if let Some(processed @ Value::Object(_)) = value.get("last_processed_response") {
        for key in ["new_items", "interruptions"] {
            raw.extend(item_raws(processed.get(key)));
        }
        for key in [
            "functions",
            "computer_actions",
            "custom_tool_actions",
            "local_shell_actions",
            "shell_actions",
            "apply_patch_actions",
            "handoffs",
        ] {
            if let Some(Value::Array(actions)) = processed.get(key) {
                raw.extend(actions.iter().filter_map(|a| a.get("tool_call")));
            }
        }
    }
    raw.extend(item_raws(
        value
            .get("current_step")
            .and_then(|s| s.get("data"))
            .and_then(|d| d.get("interruptions")),
    ));
    raw
}

/// Port of `_raw_item_uses_programmatic_tool_calling`.
fn raw_item_uses_programmatic_tool_calling(raw: &Value) -> bool {
    let Value::Object(map) = raw else {
        return false;
    };
    if matches!(
        map.get("type").and_then(Value::as_str),
        Some("program" | "program_output")
    ) {
        return true;
    }
    map.get("caller").is_some_and(|caller| {
        caller.get("type").and_then(Value::as_str) == Some("program")
            && caller.get("caller_id").is_some_and(Value::is_string)
    })
}

/// Envelope checks that depend on the schema version, run before anything is rebuilt. Ports
/// the version gates of `_build_run_state_from_json` and `_deserialize_run_state`.
pub(super) fn validate_envelope(value: &Value, version: &str) -> Result<()> {
    let at_least = |(major, minor): (u32, u32)| schema_version_at_least(version, major, minor);
    if !at_least(PROGRAMMATIC_TOOL_CALLING_MIN_SCHEMA_VERSION)
        && run_state_raw_items(value)
            .into_iter()
            .any(raw_item_uses_programmatic_tool_calling)
    {
        let (major, minor) = PROGRAMMATIC_TOOL_CALLING_MIN_SCHEMA_VERSION;
        return Err(AgentsError::user(format!(
            "Run state contains Programmatic Tool Calling data but uses schema version \
             {version}. Programmatic Tool Calling requires schema version {major}.{minor} or \
             later."
        )));
    }
    let context = value.get("context");
    if let Some(approvals) = context.and_then(|c| c.get("function_tool_approvals")) {
        if !is_falsy(approvals) && !at_least(AGENT_OWNED_APPROVALS_MIN_SCHEMA_VERSION) {
            return Err(AgentsError::user(
                "Agent-owned function approvals require RunState schema 1.18.",
            ));
        }
        if !approvals.is_null() && !approvals.is_array() {
            return Err(AgentsError::user(
                "RunState function_tool_approvals must be a list.",
            ));
        }
        for entry in approvals.as_array().into_iter().flatten() {
            if !entry.get("tool_key").is_some_and(Value::is_string)
                || !entry.get("decision").is_some_and(Value::is_object)
            {
                return Err(AgentsError::user(
                    "Invalid agent-owned function approval record.",
                ));
            }
        }
    }
    if let Some(processed @ Value::Object(_)) = value.get("last_processed_response")
        && let Some(start) = processed
            .get("tool_output_guardrail_result_start")
            .filter(|v| !v.is_null())
    {
        if !at_least(TOOL_OUTPUT_GUARDRAIL_BOUNDARY_MIN_SCHEMA_VERSION) {
            return Err(AgentsError::user(
                "Run state tool output guardrail result boundary requires schema 1.20.",
            ));
        }
        // `type(start) is int`: a bool, float or string is not a count.
        let serialized = value
            .get("tool_output_guardrail_results")
            .and_then(Value::as_array)
            .map(|rows| rows.as_slice())
            .unwrap_or_default();
        let kept = serialized
            .iter()
            .filter(|row| tool_guardrail_row_matches(row, "tool_output"))
            .count();
        let valid = start
            .as_u64()
            .is_some_and(|start| start as usize <= kept && kept == serialized.len());
        if !valid {
            // Dropping a serialized entry would shift every later result's boundary.
            return Err(AgentsError::user(
                "Invalid tool output guardrail result boundary",
            ));
        }
    }
    validate_pending_session_write(value, version)?;
    validate_terminal_marker(value, version)?;
    Ok(())
}

/// The keys a pending Session write may carry beyond the four base keys, gated by the schema
/// version that introduced each. Released readers validate the object by exact key set.
fn pending_write_keys(version: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    let compaction = if schema_version_at_least(version, 1, 18) {
        vec![
            "response_id",
            "store",
            "has_local_tool_outputs",
            "append_acknowledged",
            "compaction_model_exchange",
        ]
    } else {
        Vec::new()
    };
    let held = if schema_version_at_least(version, 1, 19) {
        vec!["held", "reasoning_item_id_policy", "current_response"]
    } else {
        Vec::new()
    };
    (compaction, held)
}

fn is_policy_literal(value: &Value) -> bool {
    matches!(value, Value::Null) || matches!(value.as_str(), Some("preserve" | "omit"))
}

/// Port of the pending Session write validation of `_deserialize_run_state`.
fn validate_pending_session_write(value: &Value, version: &str) -> Result<()> {
    let Some(pending) = value.get("pending_session_write").filter(|v| !v.is_null()) else {
        return Ok(());
    };
    let invalid = || AgentsError::user("Run state pending Session write is invalid");
    let step_type = value
        .get("current_step")
        .and_then(|s| s.get("type"))
        .and_then(Value::as_str);
    let current_turn = value
        .get("current_turn")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let Value::Object(map) = pending else {
        return Err(invalid());
    };
    if !schema_version_at_least(version, 1, 17)
        || !matches!(
            step_type,
            Some("next_step_run_again" | "next_step_interruption")
        )
    {
        return Err(invalid());
    }
    let (compaction, held_keys) = pending_write_keys(version);
    let base = ["session_id", "items", "before", "persisted_count"];
    if !base.iter().all(|k| map.contains_key(*k))
        || map.keys().any(|k| {
            !base.contains(&k.as_str())
                && !compaction.contains(&k.as_str())
                && !held_keys.contains(&k.as_str())
        })
    {
        return Err(invalid());
    }
    let held = map.get("held");
    if held.is_some_and(|h| !h.is_boolean()) {
        return Err(invalid());
    }
    let is_held = held == Some(&Value::Bool(true));
    let before = map.get("before").filter(|b| !b.is_null());
    if is_held && (before.is_some() || !map.contains_key("current_response")) {
        return Err(invalid());
    }
    if map
        .get("reasoning_item_id_policy")
        .is_some_and(|p| !is_policy_literal(p))
    {
        return Err(invalid());
    }
    let items = map.get("items").and_then(Value::as_array);
    if let Some(boundary) = map.get("current_response") {
        let ok = boundary.as_object().is_some_and(|b| {
            b.len() == 2
                && b.get("turn")
                    .and_then(Value::as_i64)
                    .is_some_and(|t| (0..=current_turn).contains(&t))
                && b.get("start").and_then(Value::as_i64).is_some_and(|start| {
                    items.is_some_and(|i| start >= 0 && start as usize <= i.len())
                })
        });
        if !ok {
            return Err(invalid());
        }
    }
    if !is_held
        && (map.contains_key("reasoning_item_id_policy") || map.contains_key("current_response"))
    {
        return Err(invalid());
    }
    if !map.get("session_id").is_some_and(Value::is_string)
        || !items.is_some_and(|i| !i.is_empty() && i.iter().all(Value::is_object))
    {
        return Err(invalid());
    }
    if before.is_some_and(|b| {
        !b.as_array()
            .is_some_and(|rows| rows.iter().all(Value::is_string))
    }) {
        return Err(invalid());
    }
    if !map.get("persisted_count").is_some_and(|n| n.is_u64()) {
        return Err(invalid());
    }
    if map
        .get("response_id")
        .is_some_and(|v| !v.is_null() && !v.is_string())
        || map
            .get("store")
            .is_some_and(|v| !v.is_null() && !v.is_boolean())
        || map
            .get("append_acknowledged")
            .is_some_and(|v| !v.is_boolean())
        || (map.get("append_acknowledged") == Some(&Value::Bool(true)) && before.is_none())
        || map
            .get("has_local_tool_outputs")
            .is_some_and(|v| !v.is_boolean())
    {
        return Err(invalid());
    }
    if let Some(exchange) = map.get("compaction_model_exchange") {
        let ok = exchange.as_object().is_some_and(|e| {
            e.len() == 2
                && e.get("item_digests")
                    .and_then(Value::as_array)
                    .is_some_and(|d| d.iter().all(Value::is_string))
                && e.get("reasoning_item_id_policy")
                    .is_some_and(is_policy_literal)
        });
        if !ok {
            return Err(invalid());
        }
    }
    Ok(())
}

/// The terminal marker is only honored on the schema boundary that introduced it.
fn validate_terminal_marker(value: &Value, version: &str) -> Result<()> {
    match value.get("terminal_unrecoverable") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Bool(true)) if schema_version_at_least(version, 1, 17) => Ok(()),
        Some(_) => Err(AgentsError::user("Run state terminal marker is invalid")),
    }
}

/// Whether a serialized tool guardrail result belongs to the `expected` list. Port of the type
/// check in `_parse_tool_guardrail_entry`: an entry naming another guardrail kind is dropped.
fn tool_guardrail_row_matches(row: &Value, expected: &str) -> bool {
    match row
        .get("guardrail")
        .and_then(|g| g.get("type"))
        .filter(|t| !is_falsy(t))
    {
        Some(t) => t.as_str() == Some(expected),
        None => true,
    }
}

/// Rebuild the ledger entries of the pending approvals of a snapshot written before the
/// ledger existed. Port of `RunContextWrapper._restore_pending_approval_binding`: a legacy
/// decision then authorizes only the exact invocation it was given for, so a snapshot whose
/// stored call was edited is refused on resume instead of running the edited payload.
pub(super) fn reconstruct_legacy_pending_bindings(state: &mut RunState) {
    use crate::run_internal::tool_invocation::tool_invocation_identity;

    for item in &state.interruptions {
        let mut raw = item.raw_item.to_value();
        // Hosted MCP requests keep their identity in `id`, which the typed call stores as
        // `call_id`.
        if raw.get("type").and_then(Value::as_str) == Some("mcp_approval_request")
            && let Value::Object(map) = &mut raw
        {
            map.entry("id")
                .or_insert_with(|| Value::String(item.raw_item.call_id.clone()));
        }
        // An empty `arguments` is how the typed call stores a missing one.
        let unbindable = raw.get("type").and_then(Value::as_str) == Some("mcp_approval_request")
            && item.raw_item.arguments.is_empty();
        match tool_invocation_identity(&raw, None).filter(|_| !unbindable) {
            Some(id) => {
                state
                    .tool_invocations
                    .entry(id.call_id)
                    .or_insert(ToolInvocationRecord {
                        invocation_type: id.invocation_type,
                        approval_scope: id.approval_scope,
                        fingerprint: id.fingerprint,
                        executed: false,
                        completed: false,
                    });
            }
            // A pending call that cannot be bound to one invocation carries no restored
            // decision: it goes back to the user. Port of the `identity is None` branch of
            // `_restore_pending_approval_binding`.
            None => {
                state
                    .restored_unbound_approval_call_ids
                    .insert(item.raw_item.call_id.clone());
            }
        }
    }
    // Completed calls a legacy snapshot holds in its history are bound to what they ran as,
    // so a replay under the same call id is recognized. Native calls are left to the live
    // check, which knows the tool they resolve to.
    let raws = restored_call_raw_items(state);
    let completed: BTreeSet<String> = raws
        .iter()
        .filter_map(|(raw, _)| crate::run_internal::tool_invocation::tool_output_identity(raw))
        .map(|(_, call_id)| call_id)
        .collect();
    for (raw, role) in &raws {
        let Some(id) = tool_invocation_identity(raw, *role) else {
            continue;
        };
        if !matches!(
            id.invocation_type.as_str(),
            "function_call" | "custom_tool_call" | "mcp_approval_request"
        ) || !completed.contains(&id.call_id)
        {
            continue;
        }
        // A pending call bound to other arguments under the same id loses to the completed
        // call: the id was already used for a different invocation, which the resumed run
        // refuses instead of running the changed call.
        let record = ToolInvocationRecord {
            invocation_type: id.invocation_type,
            approval_scope: id.approval_scope,
            fingerprint: id.fingerprint,
            executed: true,
            completed: true,
        };
        match state.tool_invocations.get(&id.call_id) {
            Some(existing) if !existing.completed && existing.fingerprint != record.fingerprint => {
                state.tool_invocations.insert(id.call_id, record);
            }
            Some(_) => {}
            None => {
                state.tool_invocations.insert(id.call_id, record);
            }
        }
    }
}

/// Raw tool-call items a restored state holds, with the role their identity is scoped under.
/// The durable locations Python's `_reconcile_tool_invocations` walks: generated items, the
/// parked response's items, the model responses and the original input.
fn restored_call_raw_items(state: &RunState) -> Vec<(Value, Option<&'static str>)> {
    let mut raws: Vec<(Value, Option<&'static str>)> = Vec::new();
    let mut push_items = |items: &[RunItem]| {
        for item in items {
            let role = matches!(item.kind, crate::items::RunItemKind::HandoffCall { .. })
                .then_some(crate::run_internal::tool_invocation::HANDOFF_ROLE);
            raws.push((item.to_input_item().0, role));
        }
    };
    push_items(&state.generated_items);
    if let Some(processed) = &state.last_processed_response {
        push_items(&processed.new_items);
    }
    raws.extend(state.original_input.iter().map(|i| (i.0.clone(), None)));
    raws
}

/// [`restored_call_raw_items`] plus the raw model-response outputs. The runner fingerprints
/// the call it executed, which may be the sanitized form of the provider's raw item, so the
/// responses only show which call ids occurred; they are not compared against the ledger.
fn restored_call_raw_items_with_responses(state: &RunState) -> Vec<(Value, Option<&'static str>)> {
    let mut raws = restored_call_raw_items(state);
    for response in &state.model_responses {
        raws.extend(response.output.iter().map(|o| (o.to_value(), None)));
    }
    raws
}

/// The call id of a raw tool-call item (`id` for a hosted MCP approval request), or `None`
/// when the item is not a tool call.
fn raw_call_id(raw: &Value) -> Option<&str> {
    let raw_type = raw.get("type").and_then(Value::as_str).unwrap_or_default();
    if !TOOL_INVOCATION_TYPES.contains(&raw_type) && raw_type != "hosted_tool_call" {
        return None;
    }
    let id_key = if raw_type == "mcp_approval_request" {
        "id"
    } else {
        "call_id"
    };
    raw.get(id_key)
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
}

/// Call ids of every tool call a restored state holds in its history.
pub(super) fn history_tool_call_ids(state: &RunState) -> Vec<String> {
    // Hosted MCP requests are left out: the runner records their ledger entries when the
    // approval is raised, not with the request item, so a missing entry says nothing about them.
    restored_call_raw_items_with_responses(state)
        .iter()
        .filter(|(raw, _)| {
            !matches!(
                raw.get("type").and_then(Value::as_str),
                Some("mcp_approval_request" | "hosted_tool_call")
            )
        })
        .filter_map(|(raw, _)| raw_call_id(raw).map(str::to_string))
        .collect()
}

/// A completed ledger entry claims the call ran and its output was committed. Port of the
/// completed-record check at the end of `_reconcile_tool_invocations`: every restored
/// occurrence of the call id must carry the recorded invocation, and a matching output must
/// have been restored too, otherwise the snapshot is refused.
pub(super) fn validate_completed_tool_invocations(state: &RunState, wire: &Value) -> Result<()> {
    use crate::run_internal::tool_invocation::{tool_invocation_identity, tool_output_identity};

    let completed: Vec<(&String, &ToolInvocationRecord)> = state
        .tool_invocations
        .iter()
        .filter(|(_, record)| record.completed)
        .collect();
    if completed.is_empty() {
        return Ok(());
    }
    let mut raws = restored_call_raw_items(state);
    // Python also reads the session history, so a call or output present there but removed
    // from the generated items does not make a completed entry unprovable.
    if let Some(items) = wire.get("session_items").and_then(Value::as_array) {
        for item in items {
            if let Some(raw) = item.get("raw_item") {
                let role = (item.get("type").and_then(Value::as_str) == Some("handoff_call_item"))
                    .then_some(crate::run_internal::tool_invocation::HANDOFF_ROLE);
                raws.push((raw.clone(), role));
            }
        }
    }
    let refused = || {
        AgentsError::user(
            "RunState completed tool invocation does not match a restored tool call and output.",
        )
    };
    for (call_id, record) in completed {
        let mut seen_call = false;
        let mut seen_output = false;
        for (raw, role) in &raws {
            if let Some((output_type, output_call_id)) = tool_output_identity(raw)
                && output_call_id == *call_id
                && output_type == record.invocation_type
            {
                seen_output = true;
                continue;
            }
            if raw_call_id(raw) != Some(call_id.as_str()) {
                continue;
            }
            seen_call = true;
            // A native call's fingerprint covers the name of the tool it resolved to, which
            // the restored item does not carry; its payload is re-checked against the agent's
            // tools when the run resumes.
            let native = matches!(
                record.invocation_type.as_str(),
                "apply_patch_call" | "computer_call" | "local_shell_call" | "shell_call"
            );
            let matches = |role: Option<&str>| {
                tool_invocation_identity(raw, role).is_some_and(|id| {
                    id.invocation_type == record.invocation_type
                        && (native || id.fingerprint == record.fingerprint)
                })
            };
            // A raw response or input item may be a handoff call; Python records both
            // readings (`allow_handoff_alternative`).
            let handoff = crate::run_internal::tool_invocation::HANDOFF_ROLE;
            if !(matches(*role) || (role.is_none() && matches(Some(handoff)))) {
                return Err(refused());
            }
        }
        if !seen_call || !seen_output {
            return Err(refused());
        }
    }
    Ok(())
}

/// Split a `major.minor` schema version into its numbers.
///
/// The comparison has to be numeric: `"1.9"` and `"1.15"` order the other way round as strings,
/// which would put every 1.9-and-later snapshot on the wrong side of the binding cut-off.
fn parse_schema_version(version: &str) -> Option<(u32, u32)> {
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Whether `version` is at least `major.minor`, compared numerically.
pub(super) fn schema_version_at_least(version: &str, major: u32, minor: u32) -> bool {
    parse_schema_version(version).is_some_and(|v| v >= (major, minor))
}

fn agent_ref(name: &str) -> Value {
    json!({ "name": name })
}

/// An agent reference with its duplicate-name identity key, when it needs one.
fn agent_ref_with_identity(name: &str, identity: Option<&str>) -> Value {
    match identity {
        Some(identity) if identity != name => json!({ "name": name, "identity": identity }),
        _ => agent_ref(name),
    }
}

/// Give every approval item owned by the agent called `name` that agent's identity key, so a
/// restored snapshot can tell same-named agents apart.
fn stamp_approval_owner_identity(value: &mut Value, name: &str, identity: &str) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("tool_approval_item")
                && let Some(Value::Object(agent)) = map.get_mut("agent")
                && !agent.contains_key("identity")
            {
                let owner = agent.get("name").and_then(Value::as_str);
                // A parked approval names its owner by identity key; the name goes back to
                // being the agent's own.
                if owner == Some(name) || owner == Some(identity) {
                    agent.insert("name".into(), Value::String(name.to_string()));
                    agent.insert("identity".into(), Value::String(identity.to_string()));
                }
            }
            for child in map.values_mut() {
                stamp_approval_owner_identity(child, name, identity);
            }
        }
        Value::Array(items) => {
            for child in items {
                stamp_approval_owner_identity(child, name, identity);
            }
        }
        _ => {}
    }
}

/// Call ids of parked approvals whose recorded owner is another agent sharing the current
/// agent's name. From schema 1.7 duplicate-named owners are persisted by identity, and an owner
/// that resolves to a sibling (or carries no identity, which resolves to the first agent of that
/// name) does not own the call; earlier schemas named owners by agent name only. Port of
/// `_approval_matches_agent`.
pub(super) fn foreign_owner_call_ids(state: &RunState, value: &Value) -> BTreeSet<String> {
    let current_identity = state
        .current_agent_identity
        .as_deref()
        .unwrap_or(&state.current_agent_name);
    state
        .interruptions
        .iter()
        .zip(
            value
                .get("current_step")
                .and_then(|step| step.get("data"))
                .and_then(|data| data.get("interruptions"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter(|(_, wire)| agent_name_of(wire.get("agent")) == state.current_agent_name)
        .filter(|(item, wire)| {
            let owner = agent_identity_of(wire.get("agent")).unwrap_or_else(|| item.agent.clone());
            owner != current_identity
        })
        .map(|(item, _)| item.call_id().to_string())
        .collect()
}

/// The agent name behind each identity key a snapshot mentions: the current agent, the owners
/// of sticky decisions and the owners of parked approvals.
fn owner_names_from_wire(value: &Value) -> std::collections::BTreeMap<String, String> {
    let mut names = std::collections::BTreeMap::new();
    let mut note = |agent: Option<&Value>| {
        if let Some(identity) = agent_identity_of(agent) {
            names.insert(identity, agent_name_of(agent));
        }
    };
    note(value.get("current_agent"));
    let context = value.get("context");
    for entry in context
        .and_then(|c| c.get("function_tool_approvals"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        note(entry.get("agent"));
    }
    for entry in value
        .get("current_step")
        .and_then(|step| step.get("data"))
        .and_then(|data| data.get("interruptions"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        note(entry.get("agent"));
    }
    names
}

/// Call ids of parked hosted MCP requests whose serialized `raw_item` has no `arguments`.
/// A live request always carries them, so such an item cannot be bound to one invocation: it
/// stays pending and is asked again whatever decision or ledger entry the snapshot holds. Port
/// of the `arguments is None` unbindable branch of `_approval_binding`.
pub(super) fn hosted_pending_without_arguments(
    state: &RunState,
    value: &Value,
) -> BTreeSet<String> {
    state
        .interruptions
        .iter()
        .zip(
            value
                .get("current_step")
                .and_then(|step| step.get("data"))
                .and_then(|data| data.get("interruptions"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter(|(_, wire)| {
            wire.get("raw_item")
                .filter(|raw| {
                    raw.get("type").and_then(Value::as_str) == Some("mcp_approval_request")
                })
                .is_some_and(|raw| raw.get("arguments").is_none())
        })
        .map(|(item, _)| item.call_id().to_string())
        .collect()
}

fn agent_identity_of(value: Option<&Value>) -> Option<String> {
    value
        .and_then(|v| v.get("identity"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn agent_name_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(m)) => m
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// Usage in Python's shape: the top-level token details are one-element lists.
fn usage_to_wire(usage: &Usage) -> Value {
    let mut v = serde_json::to_value(usage).unwrap_or_else(|_| json!({}));
    if let Value::Object(m) = &mut v {
        let input = m
            .remove("input_tokens_details")
            .unwrap_or_else(|| json!({}));
        let output = m
            .remove("output_tokens_details")
            .unwrap_or_else(|| json!({}));
        m.insert("input_tokens_details".into(), Value::Array(vec![input]));
        m.insert("output_tokens_details".into(), Value::Array(vec![output]));
    }
    v
}

fn usage_from_wire(value: &Value) -> Usage {
    let mut v = value.clone();
    if let Value::Object(m) = &mut v {
        for key in ["input_tokens_details", "output_tokens_details"] {
            if let Some(Value::Array(items)) = m.get(key).cloned() {
                // Python sums the per-request detail entries into one aggregate.
                let mut merged = Map::new();
                for item in items {
                    if let Value::Object(entry) = item {
                        for (k, val) in entry {
                            let add = val.as_u64().unwrap_or(0);
                            let total = merged
                                .get(&k)
                                .and_then(Value::as_u64)
                                .unwrap_or(0)
                                .saturating_add(add);
                            merged.insert(k, Value::from(total));
                        }
                    }
                }
                m.insert(key.into(), Value::Object(merged));
            }
        }
    }
    serde_json::from_value(v).unwrap_or_default()
}

fn approval_decision_to_wire(record: &ApprovalRecord) -> Value {
    let approved = if record.approved_always {
        Value::Bool(true)
    } else {
        Value::Array(
            record
                .approved_call_ids
                .iter()
                .map(|id| Value::String(id.clone()))
                .collect(),
        )
    };
    let rejected = if record.rejected_always {
        Value::Bool(true)
    } else {
        Value::Array(
            record
                .rejected_call_ids
                .iter()
                .map(|id| Value::String(id.clone()))
                .collect(),
        )
    };
    let mut decision = json!({ "approved": approved, "rejected": rejected });
    if !record.rejection_messages.is_empty() {
        decision["rejection_messages"] = serde_json::to_value(&record.rejection_messages)
            .unwrap_or_else(|_| Value::Object(Map::new()));
    }
    if let Some(msg) = &record.sticky_rejection_message {
        decision["sticky_rejection_message"] = Value::String(msg.clone());
    }
    decision
}

fn approval_decision_from_wire(decision: &Value) -> ApprovalRecord {
    let mut record = ApprovalRecord::default();
    match decision.get("approved") {
        Some(Value::Bool(b)) => record.approved_always = *b,
        Some(Value::Array(ids)) => {
            record.approved_call_ids = ids
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
        }
        _ => {}
    }
    match decision.get("rejected") {
        Some(Value::Bool(b)) => record.rejected_always = *b,
        Some(Value::Array(ids)) => {
            record.rejected_call_ids = ids
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
        }
        _ => {}
    }
    if let Some(Value::Object(messages)) = decision.get("rejection_messages") {
        for (call_id, message) in messages {
            if let Some(m) = message.as_str() {
                record
                    .rejection_messages
                    .insert(call_id.clone(), m.to_string());
            }
        }
    }
    record.sticky_rejection_message = decision
        .get("sticky_rejection_message")
        .and_then(Value::as_str)
        .map(str::to_string);
    record
}

/// `agent \u{1} tool` is the in-memory approval key; on the wire the two halves are separate.
fn split_approval_key(key: &str) -> (String, String) {
    match key.split_once('\u{1}') {
        Some((agent, tool)) => (agent.to_string(), tool.to_string()),
        None => (String::new(), key.to_string()),
    }
}

/// Hosted MCP decisions with their typed identities, in Python's `hosted_mcp_approvals` shape.
fn hosted_approvals_to_wire(approvals: &ApprovalsState) -> Vec<Value> {
    use crate::run_context::HostedApprovalIdentity as H;
    approvals
        .0
        .iter()
        .filter_map(|(key, record)| {
            let identity = match crate::run_context::parse_hosted_key(key)? {
                H::ServerTool {
                    server_label,
                    tool_name,
                } => json!({"type": "server_tool", "server_label": server_label, "tool_name": tool_name}),
                H::Request { request_id } => json!({"type": "request", "request_id": request_id}),
                H::Query {
                    tool_name,
                    request_id,
                } => json!({"type": "query", "tool_name": tool_name, "request_id": request_id}),
            };
            Some(json!({"identity": identity, "decision": approval_decision_to_wire(record)}))
        })
        .collect()
}

/// Whether the serialized current response of a parked turn can be proven to be the exact
/// terminal run of generated items. Port of `_restore_current_response_item_identities` plus
/// the `proven` rule of `_current_response_boundary`: `true` when there is nothing to prove.
///
/// The response is proven when its ownership range matches the serialized items one for one,
/// its interruptions are the approval items after it, and every item of it that is public
/// history sits in the session history as the terminal suffix, through a valid
/// `generated_session_item_indexes` map. Anything else is unproven, except on turn one, where
/// no earlier items exist to confuse it with.
fn current_response_boundary_proven(value: &Value, version: &str) -> bool {
    let processed = value
        .get("last_processed_response")
        .filter(|p| p.is_object())
        .and_then(|p| p.get("new_items"))
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty());
    let interruptions = value
        .get("current_step")
        .and_then(|s| s.get("data"))
        .and_then(|d| d.get("interruptions"))
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty());
    let (Some(processed), Some(interruptions)) = (processed, interruptions) else {
        return true;
    };
    let generated = value
        .get("generated_items")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if relinks(value, version, generated, processed, interruptions) {
        return true;
    }
    // Serialization rebuilds identities; turn one has no accepted prefix to mistake it for.
    value.get("current_turn").and_then(Value::as_u64) == Some(1)
        && generated.len() >= processed.len()
        && generated
            .iter()
            .zip(processed)
            .all(|(g, p)| g.get("type") == p.get("type"))
}

fn relinks(
    value: &Value,
    version: &str,
    generated: &[Value],
    processed: &[Value],
    interruptions: &[Value],
) -> bool {
    if !schema_version_at_least(version, 1, 17) {
        return false;
    }
    let Some(ownership) = value.get("current_response_generated_item_ownership") else {
        return false;
    };
    let index = |v: Option<&Value>| v.and_then(Value::as_u64).map(|v| v as usize);
    let (Some(start), Some(end)) = (index(ownership.get("start")), index(ownership.get("end")))
    else {
        return false;
    };
    let processed_end = start + processed.len();
    if end != generated.len() || processed_end > end {
        return false;
    }
    let Some(owned_interruptions) = ownership.get("interruptions").and_then(Value::as_array) else {
        return false;
    };
    let indexes: Vec<usize> = owned_interruptions
        .iter()
        .filter_map(|v| index(Some(v)))
        .collect();
    if indexes.len() != owned_interruptions.len()
        || indexes.len() != interruptions.len()
        || indexes.iter().any(|i| *i < processed_end || *i >= end)
    {
        return false;
    }
    let distinct: std::collections::BTreeSet<_> = indexes.iter().collect();
    if distinct.len() != indexes.len() {
        return false;
    }
    let source_indexes = (start..processed_end).chain(indexes.iter().copied());
    let expected = processed.iter().chain(interruptions);
    if source_indexes
        .zip(expected)
        .any(|(i, expected)| &generated[i] != expected)
    {
        return false;
    }
    // Every current-response item that is public history must be the terminal suffix of the
    // session history, reached through a valid index map. A tool output committed to execution
    // history but not yet public may be absent from it.
    let session = value
        .get("session_items")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mapping = value
        .get("generated_session_item_indexes")
        .and_then(Value::as_array)
        .filter(|m| m.len() == generated.len());
    let mapped = |generated_index: usize| -> Option<usize> {
        let mapping = mapping?;
        // The map is honoured only if it is valid as a whole.
        let mut seen = std::collections::BTreeSet::new();
        for entry in mapping {
            if entry.is_null() {
                continue;
            }
            let i = index(Some(entry))?;
            if i >= session.len() || !seen.insert(i) {
                return None;
            }
        }
        let i = index(Some(&mapping[generated_index]))?;
        (generated[generated_index] == session[i]).then_some(i)
    };
    let processed_count = processed.len();
    let mut public = Vec::new();
    for (offset, generated_index) in (start..end).enumerate() {
        match mapped(generated_index) {
            Some(session_index) => public.push(session_index),
            None => {
                let is_output = generated[generated_index]
                    .get("type")
                    .and_then(Value::as_str)
                    == Some("tool_call_output_item");
                if offset < processed_count || !is_output {
                    return false;
                }
            }
        }
    }
    let Some(session_start) = session.len().checked_sub(public.len()) else {
        return false;
    };
    public
        .iter()
        .enumerate()
        .all(|(k, session_index)| *session_index == session_start + k)
}

/// Read Python's `hosted_mcp_approvals`. Port of `_rebuild_hosted_mcp_approvals`: an entry
/// whose identity is incomplete or of an unknown type is skipped.
fn hosted_approvals_from_wire(context: &Value, state: &mut ApprovalsState) {
    let Some(Value::Array(entries)) = context.get("hosted_mcp_approvals") else {
        return;
    };
    let field = |identity: &Value, name: &str| {
        identity
            .get(name)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    for entry in entries {
        let (Some(identity), Some(decision)) = (entry.get("identity"), entry.get("decision"))
        else {
            continue;
        };
        if !identity.is_object() || !decision.is_object() {
            continue;
        }
        let key = match identity.get("type").and_then(Value::as_str) {
            Some("server_tool") => match (
                field(identity, "server_label"),
                field(identity, "tool_name"),
            ) {
                (Some(label), Some(tool)) => {
                    crate::run_context::hosted_server_tool_key(&label, &tool)
                }
                _ => continue,
            },
            Some("request") => match field(identity, "request_id") {
                Some(id) => crate::run_context::hosted_request_key(&id),
                None => continue,
            },
            Some("query") => match (field(identity, "tool_name"), field(identity, "request_id")) {
                (Some(tool), Some(id)) => crate::run_context::hosted_query_key(&tool, &id),
                _ => continue,
            },
            _ => continue,
        };
        state.0.insert(key, approval_decision_from_wire(decision));
    }
}

fn approvals_to_wire(state: &RunState) -> Value {
    Value::Array(
        state
            .approvals
            .0
            .iter()
            .filter(|(key, _)| crate::run_context::parse_hosted_key(key).is_none())
            .filter_map(|(key, record)| {
                // A capability-bound decision is written like any agent-owned one, plus the
                // Rust-only `capability` field: Python reads entries by key and binds the
                // decision to its owner, Rust restores the binding.
                let (key, capability) = match crate::run_context::split_bound_approval_key(key) {
                    Some((unbound, fingerprint)) => (unbound, Some(fingerprint)),
                    None => (key.as_str(), None),
                };
                let (owner, tool_key) = split_approval_key(key);
                // An owner that only graph order tells from a sibling carries no authority
                // into a snapshot.
                if state.ambiguous_agent_owners.contains(&owner) {
                    return None;
                }
                let name = state
                    .agent_owner_names
                    .get(&owner)
                    .map_or(owner.as_str(), String::as_str);
                let mut entry = json!({
                    "agent": agent_ref_with_identity(name, Some(&owner)),
                    "tool_key": tool_key,
                    "decision": approval_decision_to_wire(record),
                });
                if let Some(capability) = capability {
                    entry["capability"] = Value::String(capability.to_string());
                }
                Some(entry)
            })
            .collect(),
    )
}

/// The capability registry [`RunState::approval_capabilities`] wrote under the `rust` key.
fn approval_capabilities_from_wire(
    rust: &Value,
) -> std::collections::BTreeMap<String, crate::run_context::ApprovalCapability> {
    rust.get("approval_capabilities")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn approvals_from_wire(context: &Value, version: &str) -> ApprovalsState {
    let mut state = ApprovalsState::default();
    if let Some(Value::Array(entries)) = context.get("function_tool_approvals") {
        for entry in entries {
            // The owner key is the identity of a duplicate-named agent, else its name.
            let agent = agent_identity_of(entry.get("agent"))
                .unwrap_or_else(|| agent_name_of(entry.get("agent")));
            let tool_key = entry
                .get("tool_key")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let decision = entry.get("decision").cloned().unwrap_or(Value::Null);
            let mut key = format!("{agent}\u{1}{tool_key}");
            // A Rust writer's capability-bound decision; see `approvals_to_wire`.
            if let Some(capability) = entry
                .get("capability")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
            {
                key = crate::run_context::join_bound_approval_key(&key, capability);
            }
            state.0.insert(key, approval_decision_from_wire(&decision));
        }
    }
    // Python's agent-agnostic bucket. Those decisions apply whichever agent is running, so
    // they are stored under an empty agent and consulted as a fallback.
    if let Some(Value::Object(entries)) = context.get("approvals") {
        for (tool_key, decision) in entries {
            state
                .0
                .entry(format!("\u{1}{tool_key}"))
                .or_insert_with(|| approval_decision_from_wire(decision));
        }
    }
    // Typed hosted MCP records exist from schema 1.14; an older snapshot's are ignored, as in
    // Python, so a legacy writer's decisions never authorize a server they were not scoped to.
    if schema_version_at_least(version, 1, 14) {
        hosted_approvals_from_wire(context, &mut state);
    }
    state
}

fn run_item_to_wire(item: &RunItem) -> Value {
    // An approval item appears both here and in `current_step.interruptions`. Python compares
    // the two payloads when it relinks a restored response to its generated items, so they
    // have to serialize identically.
    if let crate::items::RunItemKind::ToolApproval {
        raw_item,
        tool_name,
    } = &item.kind
    {
        let mut wire = interruption_to_wire(&ToolApprovalItem {
            agent: item.agent.clone(),
            raw_item: raw_item.clone(),
            tool_name: tool_name.clone(),
            tool_origin: item.tool_origin.clone(),
        });
        if let Value::Object(map) = &mut wire {
            if let Some(key) = &item.tool_lookup_key {
                map.insert("tool_lookup_key".into(), key.clone());
            }
            if item.allow_bare_name_alias {
                map.insert("allow_bare_name_alias".into(), Value::Bool(true));
            }
        }
        return wire;
    }
    let mut value = serde_json::to_value(item).unwrap_or_else(|_| json!({}));
    if let Value::Object(m) = &mut value {
        let agent = m
            .get("agent")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        m.insert("agent".into(), agent_ref(&agent));
        for key in ["source_agent", "target_agent"] {
            if let Some(Value::String(name)) = m.get(key).cloned() {
                m.insert(key.into(), agent_ref(&name));
            }
        }
        // The typed raw items serialize without the `type` discriminator the OpenAI item
        // models require; without it Python drops the item while reading the snapshot.
        if let Some(raw_type) = raw_item_type(&item.kind)
            && let Some(Value::Object(raw)) = m.get_mut("raw_item")
        {
            raw.entry("type".to_string())
                .or_insert_with(|| Value::String(raw_type.to_string()));
        }
    }
    value
}

/// The `raw_item.type` an item carries on the wire, for the kinds whose Rust raw item is a
/// typed struct that does not serialize one.
fn raw_item_type(kind: &crate::items::RunItemKind) -> Option<&'static str> {
    use crate::items::RunItemKind as K;
    match kind {
        K::MessageOutput { .. } => Some("message"),
        K::Reasoning { .. } => Some("reasoning"),
        K::HandoffCall { .. } => Some("function_call"),
        _ => None,
    }
}

/// Build a `FunctionCall` from any approval raw item, including hosted MCP approval
/// requests, whose identity lives in `id` rather than `call_id`.
fn function_call_from_raw(raw: &Value) -> Option<crate::items::FunctionCall> {
    let obj = raw.as_object()?;
    let mut extra = Map::new();
    for (key, value) in obj {
        if !matches!(
            key.as_str(),
            "call_id" | "id" | "name" | "arguments" | "status"
        ) {
            extra.insert(key.clone(), value.clone());
        }
    }
    Some(crate::items::FunctionCall {
        // A call has an item id beside its call id; a hosted request's id is its call id.
        id: obj
            .get("call_id")
            .and_then(|_| obj.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        call_id: obj
            .get("call_id")
            .or_else(|| obj.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        name: obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        // A hosted MCP request without arguments stays without them: it cannot be bound to
        // one invocation, which `reconstruct_legacy_pending_bindings` must see.
        arguments: obj
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or(
                if obj.get("type").and_then(Value::as_str) == Some("mcp_approval_request") {
                    ""
                } else {
                    "{}"
                },
            )
            .to_string(),
        status: obj
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        extra,
    })
}

/// A `tool_call_output_item` holding a `local_shell_call_output` whose call id or output is not
/// a usable string. Python validates the shell output strictly and drops such an item.
fn is_invalid_local_shell_output_item(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("tool_call_output_item")
        && item
            .get("raw_item")
            .filter(|raw| {
                raw.get("type").and_then(Value::as_str) == Some("local_shell_call_output")
            })
            .is_some_and(|raw| {
                raw.get("call_id")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                    || !raw.get("output").is_some_and(Value::is_string)
            })
}

/// Read one serialized run item. `Ok(None)` is an item Python's `_deserialize_items` skips with a
/// warning instead of failing the restore: no `type`, no agent to attribute it to, a handoff
/// output that names no source or target, a tool output whose raw item has no `type`, or a
/// payload its typed model rejects. The warnings follow the model/tool data policy and never
/// echo payload values unless both policies allow it.
fn run_item_from_wire(value: &Value) -> Result<Option<RunItem>> {
    let mut value = value.clone();
    let item_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if item_type.is_empty() {
        log::warn!(target: crate::logger::LOG_TARGET, "Item missing type field, skipping");
        return Ok(None);
    }
    let item_type = item_type.to_string();
    if let Value::Object(m) = &mut value {
        if item_type == "tool_call_output_item"
            && m.get("raw_item").is_some_and(|raw| {
                raw.is_object() && !raw.get("type").is_some_and(Value::is_string)
            })
        {
            log_skipped_item("Failed to deserialize item", &item_type);
            return Ok(None);
        }
        // Python builds the message through its typed model, which needs an id, a role and
        // content (a bare string is accepted from an assistant, as a legacy shape).
        if item_type == "message_output_item"
            && let Some(raw) = m.get("raw_item").and_then(Value::as_object)
        {
            let text_shape = raw.get("role").and_then(Value::as_str) == Some("assistant")
                && raw.get("content").is_some_and(Value::is_string);
            if !text_shape
                && !["id", "role", "content"]
                    .iter()
                    .all(|key| raw.get(*key).is_some_and(|v| !v.is_null()))
            {
                log_skipped_item("Failed to deserialize item", &item_type);
                return Ok(None);
            }
        }
        if item_type == "handoff_output_item"
            && ["source_agent", "target_agent"]
                .iter()
                .any(|key| agent_name_of(m.get(*key)).is_empty())
        {
            crate::logger::log_model_and_tool_data_warning(
                "Skipping handoff output item: could not resolve agents",
                || {
                    format!(
                        "Skipping handoff_output_item: could not resolve agents (source={}, target={}).",
                        agent_name_of(m.get("source_agent")),
                        agent_name_of(m.get("target_agent")),
                    )
                },
            );
            return Ok(None);
        }
        // Python splits tool search into its own item types; Rust merges them into the
        // generic tool-call items with the right stream event names.
        match m.get("type").and_then(Value::as_str) {
            Some("tool_search_call_item") => {
                m.insert("type".into(), Value::String("tool_call_item".into()));
            }
            Some("tool_search_output_item") => {
                m.insert("type".into(), Value::String("tool_call_output_item".into()));
                m.entry("output").or_insert(Value::Null);
            }
            _ => {}
        }
        // A handoff output written without `agent` belongs to its source agent (then its
        // target), as `_deserialize_items` resolves it.
        let agent = ["agent", "source_agent", "target_agent"]
            .iter()
            .map(|key| agent_name_of(m.get(*key)))
            .find(|name| !name.is_empty());
        let Some(agent) = agent else {
            crate::logger::log_model_and_tool_data_warning(
                "Item missing agent field, skipping",
                || format!("Item missing agent field, skipping: {item_type}"),
            );
            return Ok(None);
        };
        m.insert("agent".into(), Value::String(agent));
        for key in ["source_agent", "target_agent"] {
            if let Some(entry) = m.get(key).cloned()
                && entry.is_object()
            {
                m.insert(key.into(), Value::String(agent_name_of(Some(&entry))));
            }
        }
        // `handoff_call_item` and `tool_approval_item` hold a typed `FunctionCall`.
        if matches!(
            m.get("type").and_then(Value::as_str),
            Some("handoff_call_item") | Some("tool_approval_item")
        ) && let Some(raw) = m.get("raw_item").cloned()
        {
            let mut raw = strip_function_call_type(raw);
            // Python falls back to the bare dict for a payload that is not a function call;
            // the typed call takes empty defaults so the item is kept rather than dropped.
            if let Value::Object(call) = &mut raw {
                for (key, default) in [("call_id", ""), ("name", ""), ("arguments", "{}")] {
                    call.entry(key)
                        .or_insert_with(|| Value::String(default.into()));
                }
            }
            m.insert("raw_item".into(), raw);
        }
        // Python validates the shell output item strictly (it has a typed model there): both
        // the call id and the output must be strings, and the call id may not be empty. The
        // refusal never echoes the offending value.
        if m.get("type").and_then(Value::as_str) == Some("tool_call_output_item")
            && let Some(raw) = m.get("raw_item")
            && raw.get("type").and_then(Value::as_str) == Some("local_shell_call_output")
        {
            let call_id_ok = raw
                .get("call_id")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.is_empty());
            let output_ok = raw.get("output").is_some_and(Value::is_string);
            if !call_id_ok || !output_ok {
                return Err(AgentsError::user(
                    "Run state has an invalid local_shell_call_output item",
                ));
            }
        }
        // Fields Python's reader derives when a payload leaves them out.
        match m.get("type").and_then(Value::as_str) {
            Some("tool_call_output_item") if !m.contains_key("output") => {
                let output = m
                    .get("raw_item")
                    .and_then(|r| r.get("output"))
                    .cloned()
                    .unwrap_or(Value::Null);
                m.insert("output".into(), output);
            }
            Some("tool_approval_item") if !m.get("tool_name").is_some_and(Value::is_string) => {
                let name = m
                    .get("raw_item")
                    .and_then(|r| r.get("name"))
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new()));
                m.insert("tool_name".into(), name);
            }
            Some("handoff_output_item") => {
                for key in ["source_agent", "target_agent"] {
                    m.entry(key).or_insert_with(|| Value::String(String::new()));
                }
            }
            _ => {}
        }
        // Python-only bookkeeping that has no Rust field.
        for key in ["input_id", "tool_namespace"] {
            m.remove(key);
        }
        // Read tolerantly like Python: a lookup key must be a mapping and the alias flag is
        // honoured only as a literal `true`.
        if let Some(key) = m
            .remove("tool_lookup_key")
            .as_ref()
            .and_then(read_lookup_key)
        {
            m.insert("tool_lookup_key".into(), key);
        }
        if m.get("allow_bare_name_alias") != Some(&Value::Bool(true)) {
            m.remove("allow_bare_name_alias");
        }
        // Origin metadata is read tolerantly: a malformed or unknown origin is dropped, so a
        // snapshot written by a newer SDK still loads.
        let origin = m
            .remove("tool_origin")
            .and_then(|v| crate::tool::ToolOrigin::from_json_dict(&v));
        if let Some(origin) = origin {
            m.insert("tool_origin".into(), origin.to_json_dict());
        }
        for key in ["description", "title"] {
            if !m.get(key).is_some_and(Value::is_string) {
                m.remove(key);
            }
        }
    }
    Ok(match serde_json::from_value(value) {
        Ok(item) => Some(item),
        Err(_) => {
            log_skipped_item("Failed to deserialize item", &item_type);
            None
        }
    })
}

/// Warn about a skipped item. The item type is payload data, so it appears only when the model
/// and tool data policies both allow it.
fn log_skipped_item(redacted: &str, item_type: &str) {
    crate::logger::log_model_and_tool_data_warning(redacted, || {
        format!("{redacted}: item_type={item_type}")
    });
}

fn interruption_to_wire(item: &ToolApprovalItem) -> Value {
    let mut value = json!({
        "type": "tool_approval_item",
        "raw_item": item.raw_item.to_value(),
        "agent": agent_ref(&item.agent),
        "tool_name": item.tool_name,
    });
    if let Some(origin) = &item.tool_origin
        && let Value::Object(m) = &mut value
    {
        m.insert("tool_origin".into(), origin.to_json_dict());
    }
    value
}

/// `FunctionCall::to_value` writes `"type": "function_call"`; reading it back would otherwise
/// land in the catch-all `extra` map and make the round trip asymmetric.
fn strip_function_call_type(mut raw: Value) -> Value {
    if raw.get("type").and_then(Value::as_str) == Some("function_call")
        && let Value::Object(m) = &mut raw
    {
        m.remove("type");
    }
    raw
}

fn interruption_from_wire(value: &Value) -> Result<ToolApprovalItem> {
    let raw = value
        .get("raw_item")
        .ok_or_else(|| AgentsError::user("Run state interruption is missing raw_item"))?;
    let mut raw_item = function_call_from_raw(&strip_function_call_type(raw.clone()))
        .ok_or_else(|| AgentsError::user("Run state interruption has an invalid raw_item"))?;
    // A hosted MCP request legitimately has no arguments and names its call by `id`. For a
    // function call, an approval without `arguments` or `call_id` is malformed: keep those empty
    // so the resume can hold the turn (`reconcile_resumed_approvals`) instead of inventing them.
    if raw.get("type").and_then(Value::as_str) != Some("mcp_approval_request") {
        if raw.get("arguments").is_none() {
            raw_item.arguments.clear();
        }
        if raw.get("call_id").and_then(Value::as_str).is_none() {
            raw_item.call_id.clear();
        }
    }
    let tool_name = value
        .get("tool_name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| raw_item.name.clone());
    Ok(ToolApprovalItem {
        agent: agent_identity_of(value.get("agent"))
            .unwrap_or_else(|| agent_name_of(value.get("agent"))),
        raw_item,
        tool_name,
        tool_origin: value
            .get("tool_origin")
            .and_then(crate::tool::ToolOrigin::from_json_dict),
    })
}

/// The `{key, wrapper}` pairs of Python's `_serialize_tool_action_groups`.
const ACTION_GROUPS: [(&str, &str); 6] = [
    ("functions", "tool"),
    ("computer_actions", "computer"),
    ("custom_tool_actions", "custom_tool"),
    ("local_shell_actions", "local_shell"),
    ("shell_actions", "shell"),
    ("apply_patch_actions", "apply_patch"),
];

/// Mirrors `deserialize_function_tool_lookup_key`: a persisted key is a mapping with a string
/// `kind` and a non-empty string `name`; `namespaced` also needs a non-empty `namespace`.
/// Anything else reads as no key.
fn read_lookup_key(data: &Value) -> Option<Value> {
    let name = data
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())?;
    match data.get("kind")?.as_str()? {
        kind @ ("bare" | "deferred_top_level") => Some(json!({"kind": kind, "name": name})),
        "namespaced" => {
            let namespace = data
                .get("namespace")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())?;
            Some(json!({"kind": "namespaced", "namespace": namespace, "name": name}))
        }
        _ => None,
    }
}

/// The lookup key Python matches a restored tool call against.
/// Mirrors `serialize_function_tool_lookup_key(get_function_tool_lookup_key(...))`.
fn lookup_key_to_wire(name: &str, namespace: Option<&str>) -> Value {
    match namespace.filter(|n| !n.is_empty()) {
        Some(ns) => json!({"kind": "namespaced", "namespace": ns, "name": name}),
        None => json!({"kind": "bare", "name": name}),
    }
}

fn tool_run_to_wire(run: &ToolRunSnapshot, include_params_schema: bool) -> Value {
    let mut tool = Map::new();
    tool.insert("name".into(), Value::String(run.tool_name.clone()));
    if let Some(ns) = run.namespace.as_ref().filter(|n| !n.is_empty()) {
        tool.insert("namespace".into(), Value::String(ns.clone()));
    }
    // A deferred top-level tool is called under a synthetic namespace equal to its own name;
    // Python keys it as `deferred_top_level` rather than as a namespaced tool.
    let deferred_top_level = run.namespace.as_deref().is_none_or(str::is_empty)
        && run.tool_call.get("namespace").and_then(Value::as_str) == Some(run.tool_name.as_str());
    tool.insert(
        "lookupKey".into(),
        if deferred_top_level {
            json!({"kind": "deferred_top_level", "name": run.tool_name})
        } else {
            lookup_key_to_wire(&run.tool_name, run.namespace.as_deref())
        },
    );
    if let Some(d) = &run.description {
        tool.insert("description".into(), Value::String(d.clone()));
    }
    if include_params_schema && let Some(s) = &run.params_json_schema {
        tool.insert("paramsJsonSchema".into(), s.clone());
    }
    json!({"tool_call": run.tool_call, "__tool": Value::Object(tool)})
}

fn tool_run_group_to_wire(runs: &[ToolRunSnapshot], wrapper: &str) -> Value {
    let include_params_schema = wrapper == "tool";
    Value::Array(
        runs.iter()
            .map(|run| {
                let mut entry = tool_run_to_wire(run, include_params_schema);
                if let Value::Object(m) = &mut entry {
                    let tool = m.remove("__tool").unwrap_or(Value::Null);
                    m.insert(wrapper.to_string(), tool);
                }
                entry
            })
            .collect(),
    )
}

fn tool_run_group_from_wire(value: Option<&Value>, wrapper: &str) -> Vec<ToolRunSnapshot> {
    let Some(Value::Array(entries)) = value else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|entry| {
            let tool = entry.get(wrapper);
            let tool_call = entry.get("tool_call").cloned().unwrap_or(Value::Null);
            let name = tool
                .and_then(|t| t.get("name"))
                .and_then(Value::as_str)
                .or_else(|| tool_call.get("name").and_then(Value::as_str))
                .unwrap_or_default()
                .to_string();
            ToolRunSnapshot {
                tool_name: name,
                namespace: tool
                    .and_then(|t| t.get("namespace"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                tool_call,
                description: tool
                    .and_then(|t| t.get("description"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                params_json_schema: tool.and_then(|t| t.get("paramsJsonSchema")).cloned(),
            }
        })
        .collect()
}

fn handoffs_to_wire(runs: &[ToolRunSnapshot]) -> Value {
    Value::Array(
        runs.iter()
            .map(|run| {
                json!({
                    "tool_call": run.tool_call,
                    "handoff": {"tool_name": run.tool_name},
                })
            })
            .collect(),
    )
}

fn handoffs_from_wire(value: Option<&Value>) -> Vec<ToolRunSnapshot> {
    let Some(Value::Array(entries)) = value else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|entry| ToolRunSnapshot {
            tool_name: entry
                .get("handoff")
                .and_then(|h| h.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            namespace: None,
            tool_call: entry.get("tool_call").cloned().unwrap_or(Value::Null),
            description: None,
            params_json_schema: None,
        })
        .collect()
}

/// Lookup key and alias flag of each approval item in the history, by call id, so the
/// interruptions written beside it carry the same payload (Python compares them).
type ApprovalMetadata = BTreeMap<String, (Option<Value>, bool)>;

fn approval_metadata(state: &RunState) -> ApprovalMetadata {
    state
        .generated_items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::items::RunItemKind::ToolApproval { raw_item, .. }
                if item.tool_lookup_key.is_some() || item.allow_bare_name_alias =>
            {
                Some((
                    raw_item.call_id.clone(),
                    (item.tool_lookup_key.clone(), item.allow_bare_name_alias),
                ))
            }
            _ => None,
        })
        .collect()
}

fn interruption_with_metadata(item: &ToolApprovalItem, metadata: &ApprovalMetadata) -> Value {
    let mut wire = interruption_to_wire(item);
    if let (Some((key, alias)), Value::Object(map)) = (metadata.get(item.call_id()), &mut wire) {
        if let Some(key) = key {
            map.insert("tool_lookup_key".into(), key.clone());
        }
        if *alias {
            map.insert("allow_bare_name_alias".into(), Value::Bool(true));
        }
    }
    wire
}

/// Serialize the interrupted turn's processed response. Mirrors
/// `RunState._serialize_processed_response`.
fn processed_response_to_wire(
    processed: &ProcessedResponseSnapshot,
    metadata: &ApprovalMetadata,
) -> Value {
    let mut out = Map::new();
    out.insert(
        "new_items".into(),
        Value::Array(processed.new_items.iter().map(run_item_to_wire).collect()),
    );
    out.insert("tools_used".into(), json!(processed.tools_used));
    for (key, wrapper) in ACTION_GROUPS {
        let runs = match key {
            "functions" => &processed.functions,
            "computer_actions" => &processed.computer_actions,
            "custom_tool_actions" => &processed.custom_tool_actions,
            "local_shell_actions" => &processed.local_shell_actions,
            "shell_actions" => &processed.shell_actions,
            _ => &processed.apply_patch_actions,
        };
        out.insert(key.into(), tool_run_group_to_wire(runs, wrapper));
    }
    out.insert("handoffs".into(), handoffs_to_wire(&processed.handoffs));
    out.insert(
        "mcp_approval_requests".into(),
        Value::Array(processed.mcp_approval_requests.clone()),
    );
    out.insert(
        "interruptions".into(),
        Value::Array(
            processed
                .interruptions
                .iter()
                .map(|item| interruption_with_metadata(item, metadata))
                .collect(),
        ),
    );
    out.insert(
        "mcp_tool_bindings".into(),
        json!(processed.mcp_tool_bindings),
    );
    if let Some(start) = processed.tool_output_guardrail_result_start {
        out.insert("tool_output_guardrail_result_start".into(), json!(start));
    }
    Value::Object(out)
}

fn processed_response_from_wire(value: &Value) -> Result<ProcessedResponseSnapshot> {
    Ok(ProcessedResponseSnapshot {
        new_items: value
            .get("new_items")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(run_item_from_wire)
                    .collect::<Result<Vec<_>>>()
                    .map(|items| items.into_iter().flatten().collect::<Vec<_>>())
            })
            .transpose()?
            .unwrap_or_default(),
        tools_used: value
            .get("tools_used")
            .and_then(Value::as_array)
            .map(|v| {
                v.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        functions: tool_run_group_from_wire(value.get("functions"), "tool"),
        handoffs: handoffs_from_wire(value.get("handoffs")),
        computer_actions: tool_run_group_from_wire(value.get("computer_actions"), "computer"),
        custom_tool_actions: tool_run_group_from_wire(
            value.get("custom_tool_actions"),
            "custom_tool",
        ),
        local_shell_actions: tool_run_group_from_wire(
            value.get("local_shell_actions"),
            "local_shell",
        ),
        shell_actions: tool_run_group_from_wire(value.get("shell_actions"), "shell"),
        apply_patch_actions: tool_run_group_from_wire(
            value.get("apply_patch_actions"),
            "apply_patch",
        ),
        mcp_approval_requests: value
            .get("mcp_approval_requests")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        interruptions: value
            .get("interruptions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(interruption_from_wire)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        // Python keeps only list-shaped bindings and ignores the rest.
        mcp_tool_bindings: value
            .get("mcp_tool_bindings")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let parts = v.as_array()?;
                        let server = parts.first()?.as_str()?.to_string();
                        let tool = parts.get(1)?.as_str()?.to_string();
                        let index = parts.get(2).and_then(Value::as_u64).map(|i| i as usize);
                        Some((k.clone(), (server, tool, index)))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        // `validate_envelope` already refused anything but a count in range.
        tool_output_guardrail_result_start: value
            .get("tool_output_guardrail_result_start")
            .and_then(Value::as_u64)
            .map(|v| v as usize),
    })
}

/// The `current_response_generated_item_ownership` range Python uses to relink the restored
/// processed response to the restored generated items. Mirrors
/// `RunState._current_response_generated_item_ownership`: `None` when the approval items are
/// not the terminal suffix of the turn, which is the case Python also refuses to record.
fn ownership_to_wire(state: &RunState, processed: &ProcessedResponseSnapshot) -> Option<Value> {
    if processed.new_items.is_empty() || processed.interruptions.is_empty() {
        return None;
    }
    let start = state.turn_start_index;
    let end = state.generated_items.len();
    let processed_end = start.checked_add(processed.new_items.len())?;
    if processed_end > end {
        return None;
    }
    use crate::items::RunItemKind as K;
    let interruptions: Vec<Value> = (processed_end..end)
        .filter(|i| matches!(state.generated_items[*i].kind, K::ToolApproval { .. }))
        .map(Value::from)
        .collect();
    if interruptions.len() != processed.interruptions.len() {
        return None;
    }
    // Everything the turn generated after `new_items` is either one of the approvals or a
    // tool output, which is the only other kind Python accepts in that range.
    let tail_is_valid = (processed_end..end).all(|i| {
        matches!(
            state.generated_items[i].kind,
            K::ToolApproval { .. }
                | K::ToolCallOutput { .. }
                | K::HandoffOutput { .. }
                | K::MCPApprovalResponse { .. }
        )
    });
    if !tail_is_valid {
        return None;
    }
    Some(json!({"start": start, "end": end, "interruptions": interruptions}))
}

fn model_response_to_wire(response: &ModelResponse) -> Value {
    json!({
        "usage": usage_to_wire(&response.usage),
        "output": response.output.iter().map(crate::items::OutputItem::to_value).collect::<Vec<_>>(),
        "response_id": response.response_id,
        "request_id": response.request_id,
    })
}

fn model_response_from_wire(value: &Value) -> Result<ModelResponse> {
    let output = value
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentsError::user("Run state model response is missing output"))?;
    Ok(ModelResponse {
        output: output
            .iter()
            .map(|v| serde_json::from_value(v.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                AgentsError::user(format!("Run state has an unreadable model response: {e}"))
            })?,
        usage: value.get("usage").map(usage_from_wire).unwrap_or_default(),
        response_id: value
            .get("response_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        request_id: value
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        raw_usage: None,
    })
}

fn guardrail_output_to_wire(output: &GuardrailFunctionOutput) -> Value {
    json!({
        "tripwireTriggered": output.tripwire_triggered,
        "outputInfo": output.output_info,
    })
}

fn guardrail_output_from_wire(value: Option<&Value>) -> GuardrailFunctionOutput {
    let value = value.cloned().unwrap_or(Value::Null);
    GuardrailFunctionOutput {
        output_info: value.get("outputInfo").cloned().unwrap_or(Value::Null),
        tripwire_triggered: value
            .get("tripwireTriggered")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

fn tool_guardrail_output_to_wire(output: &ToolGuardrailFunctionOutput) -> Value {
    json!({
        "outputInfo": output.output_info,
        "behavior": serde_json::to_value(&output.behavior).unwrap_or(Value::Null),
    })
}

fn tool_guardrail_output_from_wire(value: Option<&Value>) -> ToolGuardrailFunctionOutput {
    let value = value.cloned().unwrap_or(Value::Null);
    ToolGuardrailFunctionOutput {
        output_info: value.get("outputInfo").cloned().unwrap_or(Value::Null),
        behavior: value
            .get("behavior")
            .cloned()
            .and_then(|b| serde_json::from_value(b).ok())
            .unwrap_or(crate::tool_guardrails::ToolGuardrailBehavior::Allow),
    }
}

fn guardrail_name(value: &Value) -> String {
    value
        .get("guardrail")
        .and_then(|g| g.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Read the tool-invocation ledger. Port of `_rebuild_tool_invocations`: a malformed entry
/// is an error, not a silent drop, because the ledger is what binds an approval decision to
/// the call it was given for.
fn tool_invocations_from_wire(
    context: &Value,
    version: &str,
) -> Result<BTreeMap<String, ToolInvocationRecord>> {
    // Snapshots before the binding schema carry no ledger Python trusts; it starts empty.
    let (major, minor) = super::STRICT_APPROVAL_BINDING_MIN_SCHEMA_VERSION;
    if !schema_version_at_least(version, major, minor) {
        return Ok(BTreeMap::new());
    }
    let Some(raw) = context.get("tool_invocations") else {
        return Ok(BTreeMap::new());
    };
    let map = raw
        .as_object()
        .ok_or_else(|| AgentsError::user("RunState tool_invocations must be a mapping."))?;
    let is_digest = |v: Option<&Value>| {
        v.and_then(Value::as_str).is_some_and(|d| {
            d.len() == 64
                && d.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
    };
    let mut out = BTreeMap::new();
    for (call_id, entry) in map {
        if call_id.is_empty() {
            return Err(AgentsError::user(
                "RunState tool_invocations contains an invalid call ID.",
            ));
        }
        let entry = entry
            .as_object()
            .ok_or_else(|| AgentsError::user("RunState tool invocation must be a mapping."))?;
        let invocation_type = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let executed = entry.get("executed").and_then(Value::as_bool);
        let completed = entry.get("completed").and_then(Value::as_bool);
        match (executed, completed) {
            (Some(executed), Some(completed))
                if TOOL_INVOCATION_TYPES.contains(&invocation_type)
                    && is_digest(entry.get("approval_scope"))
                    && is_digest(entry.get("fingerprint"))
                    // A completed call must also be an executed one.
                    && (executed || !completed) =>
            {
                out.insert(
                    call_id.clone(),
                    ToolInvocationRecord {
                        invocation_type: invocation_type.to_string(),
                        approval_scope: entry["approval_scope"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        fingerprint: entry["fingerprint"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        executed,
                        completed,
                    },
                );
            }
            _ => {
                return Err(AgentsError::user(
                    "RunState tool invocation contains invalid lifecycle data.",
                ));
            }
        }
    }
    Ok(out)
}

/// The `trace` payload: the full `TraceState` when the run carried one, otherwise the bare
/// `{object, id}` reference built from `trace_id`. Mirrors `_serialize_trace_data`.
fn trace_payload(state: &RunState, include_tracing_api_key: bool) -> Value {
    if let Some(payload) = state
        .trace_state
        .as_ref()
        .and_then(|t| t.to_json(include_tracing_api_key))
    {
        return payload;
    }
    state
        .trace_id
        .as_ref()
        .map(|id| json!({"object": "trace", "id": id}))
        .unwrap_or(Value::Null)
}

/// Serialize a run state to Python's schema 1.21 envelope.
pub(super) fn to_wire(state: &RunState, include_tracing_api_key: bool) -> Result<Value> {
    // Python refuses to write a snapshot whose same-named agents differ only by hook instance
    // state, since the saved owner could not be found again.
    if state.hook_identity_ambiguous {
        return Err(AgentsError::user(
            "RunState cannot distinguish same-name agents without hook instance state. Use unique agent names before starting a persistable run.",
        ));
    }
    let generated_items: Vec<Value> = state.generated_items.iter().map(run_item_to_wire).collect();
    let model_responses: Vec<Value> = state
        .model_responses
        .iter()
        .map(model_response_to_wire)
        .collect();

    let context = json!({
        "usage": usage_to_wire(&state.usage),
        "approvals": Value::Object(Map::new()),
        "function_tool_approvals": approvals_to_wire(state),
        "tool_invocations": state.tool_invocations,
        "context": state.context.clone().unwrap_or(Value::Null),
        "context_meta": match &state.context {
            Some(Value::Object(_)) => json!({
                "original_type": "mapping",
                "serialized_via": "mapping",
                "requires_deserializer": false,
                "omitted": false,
            }),
            // A context that is not a JSON object is one Python's deserializer refuses
            // ("Serialized run state context must be a mapping"), so the metadata has to say
            // so rather than claim there was no context at all. `RunState::to_json_strict`
            // rejects the snapshot outright instead of writing this.
            Some(_) => json!({
                "original_type": "custom",
                "serialized_via": "custom",
                "requires_deserializer": true,
                "omitted": false,
            }),
            None => json!({
                "original_type": "none",
                "serialized_via": "none",
                "requires_deserializer": false,
                "omitted": false,
            }),
        },
    });
    let mut context = context;
    let hosted = hosted_approvals_to_wire(&state.approvals);
    if !hosted.is_empty() {
        context["hosted_mcp_approvals"] = Value::Array(hosted);
    }

    let metadata = approval_metadata(state);
    let current_step = if state.pending_tool_processing || !state.interruptions.is_empty() {
        json!({
            "type": "next_step_interruption",
            "data": {
                "interruptions": state
                    .interruptions
                    .iter()
                    .map(|item| interruption_with_metadata(item, &metadata))
                    .collect::<Vec<_>>(),
            },
        })
    } else if state.pending_session_write.is_some() {
        // A standing pending write is only valid on a resumable step.
        json!({"type": "next_step_run_again"})
    } else {
        Value::Null
    };

    // Python serializes the processed response only while a turn is parked on an
    // interruption, and clears it when the run finishes.
    let last_processed = state
        .last_processed_response
        .as_ref()
        .filter(|_| !state.interruptions.is_empty());
    let ownership = last_processed
        .filter(|_| !state.current_response_boundary_unproven)
        .and_then(|p| ownership_to_wire(state, p));

    let mut context = context;
    if let (Some(tool_input), Value::Object(map)) = (&state.tool_input, &mut context) {
        map.insert("tool_input".into(), tool_input.clone());
    }
    let mut envelope = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": state.current_turn,
        "current_agent": agent_ref_with_identity(
            &state.current_agent_name,
            state.current_agent_identity.as_deref(),
        ),
        "original_input": serialize_input(&state.original_input),
        "pending_input": serialize_input(&state.pending_input),
        "model_responses": model_responses,
        "context": context,
        "tool_use_tracker": state.tool_use_tracker,
        "max_turns": state.max_turns,
        "no_active_agent_run": true,
        "input_guardrail_results": state
            .input_guardrail_results
            .iter()
            .map(|r| json!({
                "guardrail": {"type": "input", "name": r.guardrail_name},
                "output": guardrail_output_to_wire(&r.output),
            }))
            .collect::<Vec<_>>(),
        "output_guardrail_results": state
            .output_guardrail_results
            .iter()
            .map(|r| json!({
                "guardrail": {"type": "output", "name": r.guardrail_name},
                "output": guardrail_output_to_wire(&r.output),
                "agentOutput": r.agent_output,
                "agent": agent_ref(&r.agent_name),
            }))
            .collect::<Vec<_>>(),
        "tool_input_guardrail_results": state
            .tool_input_guardrail_results
            .iter()
            .map(|r| json!({
                "guardrail": {"type": "tool_input", "name": r.guardrail_name},
                "output": tool_guardrail_output_to_wire(&r.output),
                "tool_name": r.tool_name,
            }))
            .collect::<Vec<_>>(),
        "tool_output_guardrail_results": state
            .tool_output_guardrail_results
            .iter()
            .map(|r| json!({
                "guardrail": {"type": "tool_output", "name": r.guardrail_name},
                "output": tool_guardrail_output_to_wire(&r.output),
                "tool_name": r.tool_name,
            }))
            .collect::<Vec<_>>(),
        "conversation_id": state.conversation_id,
        "previous_response_id": state.previous_response_id,
        "auto_previous_response_id": state.auto_previous_response_id,
        "generated_prompt_cache_key": state.generated_prompt_cache_key,
        "reasoning_item_id_policy": match state.reasoning_item_id_policy {
            Some(crate::run_config::ReasoningItemIdPolicy::Omit) => json!("omit"),
            Some(crate::run_config::ReasoningItemIdPolicy::Preserve) => json!("preserve"),
            None => Value::Null,
        },
        "nested_history_owned_session_item_refs": super::nested_history::ownership_to_wire(state),
        // A state whose parked response could not be proven keeps that on the wire: its
        // generated items are written as unaliased to the session history, as Python does for
        // items it cannot match, so a second round trip does not make them look proven.
        "generated_session_item_indexes": (0..generated_items.len())
            .map(|i| {
                if state.current_response_boundary_unproven {
                    Value::Null
                } else {
                    Value::from(i)
                }
            })
            .collect::<Vec<_>>(),
        "generated_items": generated_items.clone(),
        "session_items": generated_items,
        "current_step": current_step,
        "last_model_response": model_responses.last().cloned().unwrap_or(Value::Null),
        "last_processed_response": last_processed
            .map(|p| processed_response_to_wire(p, &metadata))
            .unwrap_or(Value::Null),
        "current_turn_persisted_item_count": state
            .session_saved_items
            .saturating_sub(state.turn_start_index),
        "trace": trace_payload(state, include_tracing_api_key),
        // Rust-only run state. Python ignores unknown top-level keys.
        "rust": {
            "server_sent_items": state.server_sent_items,
            "server_input_sent": state.server_input_sent,
            "session_saved_items": state.session_saved_items,
            "input_saved_to_session": state.input_saved_to_session,
            "nested_agent_tool_states": state.nested_agent_tool_states,
            "pending_tool_processing": state.pending_tool_processing,
            "turn_start_index": state.turn_start_index,
            "model_input_start": state.model_input_start,
            "final_output": state.final_output,
            "server_conversation": state.server_conversation,
            "withheld_output_call_ids": state.withheld_output_call_ids,
        },
    });
    // The capabilities pending function-tool approvals were raised for.
    if let Some(Value::Object(rust)) = envelope.get_mut("rust") {
        rust.insert(
            "approval_capabilities".into(),
            serde_json::to_value(&state.approval_capabilities)?,
        );
    }
    // Approvals parked by a duplicate-named current agent record which duplicate owns them.
    if let Some(identity) = state.current_agent_identity.as_deref() {
        stamp_approval_owner_identity(&mut envelope, &state.current_agent_name, identity);
    }
    if let Some(sandbox) = &state.sandbox
        && let Value::Object(map) = &mut envelope
    {
        map.insert(
            "sandbox".into(),
            super::sandbox_payload::sanitize_sandbox_payload(sandbox.clone())?,
        );
    }
    if state.terminal_unrecoverable
        && let Value::Object(map) = &mut envelope
    {
        map.insert("terminal_unrecoverable".into(), Value::Bool(true));
    }
    // Additive key: Rust-only knowledge that the parked approval may end the run. Python ignores
    // unknown envelope keys, and a resume re-derives the decision from the agent when it is absent.
    if state.tool_result_may_end_run
        && let Value::Object(map) = &mut envelope
    {
        map.insert("tool_result_may_end_run".into(), Value::Bool(true));
    }
    if let Some(pending) = &state.pending_session_write
        && let Value::Object(map) = &mut envelope
    {
        map.insert("pending_session_write".into(), pending.clone());
    }
    if let (Some(ownership), Value::Object(map)) = (ownership, &mut envelope) {
        map.insert(
            "current_response_generated_item_ownership".into(),
            ownership,
        );
    }
    Ok(envelope)
}

/// Whether `context_meta` says the snapshot's context cannot be restored from the payload
/// alone. Port of `_context_meta_requires_deserializer`.
pub(super) fn context_meta_requires_restoration(context_meta: Option<&Value>) -> bool {
    let Some(Value::Object(meta)) = context_meta else {
        return false;
    };
    if meta.get("omitted") == Some(&Value::Bool(true)) {
        return true;
    }
    meta.get("requires_deserializer") == Some(&Value::Bool(true))
}

/// The message describing what a snapshot needs before its context can be restored.
/// Port of `_context_meta_warning_message`.
pub(super) fn context_meta_warning(context_meta: Option<&Value>) -> &'static str {
    let Some(Value::Object(meta)) = context_meta else {
        return "RunState context was serialized from a custom type; provide a context override \
                to restore it.";
    };
    if meta.get("omitted") == Some(&Value::Bool(true)) {
        return "RunState context was omitted during serialization; provide a context override \
                to supply it.";
    }
    "RunState context requires explicit restoration; provide a context override to restore it."
}

/// Input items in the shape the Responses API expects. Port of `RunState._serialize_input`:
/// an assistant message gets `output_text` content parts and a `completed` status.
fn serialize_input(items: &[InputItem]) -> Value {
    Value::Array(
        items
            .iter()
            .map(|item| {
                let mut value = item.0.clone();
                if value.get("role").and_then(Value::as_str) == Some("assistant")
                    && let Value::Object(map) = &mut value
                {
                    if let Some(Value::String(text)) = map.get("content").cloned() {
                        map.insert(
                            "content".into(),
                            json!([{"type": "output_text", "text": text}]),
                        );
                    }
                    map.entry("status")
                        .or_insert_with(|| Value::String("completed".into()));
                }
                value
            })
            .collect(),
    )
}

/// Rebuild a run state from Python's envelope.
pub(super) fn from_wire(value: &Value, version: &str) -> Result<RunState> {
    // `pending_input` may be absent but never anything other than a list (a null counts).
    if value.get("pending_input").is_some_and(|v| !v.is_array()) {
        return Err(AgentsError::user("Run state pending_input must be a list"));
    }
    let context = value.get("context").cloned().unwrap_or(Value::Null);
    let rust = value.get("rust").cloned().unwrap_or(Value::Null);

    // A snapshot that carries a ledger, or predates it, loses a shell output its strict model
    // rejects (one without a call id) instead of failing: the completed-invocation check then
    // decides whether the loss is tolerable. Any other snapshot refuses the item.
    let (strict_major, strict_minor) = super::STRICT_APPROVAL_BINDING_MIN_SCHEMA_VERSION;
    let drops_invalid_shell_output = !schema_version_at_least(version, strict_major, strict_minor)
        || context
            .get("tool_invocations")
            .and_then(Value::as_object)
            .is_some_and(|ledger| !ledger.is_empty());
    // An item Python's reader skips keeps a placeholder here, so every index the snapshot
    // records (turn start, ownership, saved counts) still lines up while the rest is read; the
    // placeholders are removed, and those indexes remapped, once the state is built.
    let mut keep_generated: Vec<bool> = Vec::new();
    let mut generated_items: Vec<RunItem> = Vec::new();
    for item in value
        .get("generated_items")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let read = if drops_invalid_shell_output && is_invalid_local_shell_output_item(item) {
            None
        } else {
            run_item_from_wire(item)?
        };
        keep_generated.push(read.is_some());
        generated_items.push(read.unwrap_or_else(|| {
            RunItem::new(
                "",
                crate::items::RunItemKind::Compaction {
                    raw_item: Value::Null,
                },
            )
        }));
    }

    let interruptions: Vec<ToolApprovalItem> = value
        .get("current_step")
        .and_then(|step| step.get("data"))
        .and_then(|data| data.get("interruptions"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(interruption_from_wire)
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();

    let input_guardrail_results = value
        .get("input_guardrail_results")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|r| InputGuardrailResult {
                    guardrail_name: guardrail_name(r),
                    output: guardrail_output_from_wire(r.get("output")),
                })
                .collect()
        })
        .unwrap_or_default();

    let output_guardrail_results = value
        .get("output_guardrail_results")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|r| OutputGuardrailResult {
                    guardrail_name: guardrail_name(r),
                    agent_name: agent_name_of(r.get("agent")),
                    agent_output: r.get("agentOutput").cloned().unwrap_or(Value::Null),
                    output: guardrail_output_from_wire(r.get("output")),
                })
                .collect()
        })
        .unwrap_or_default();

    let tool_results = |key: &str| -> Vec<(String, String, ToolGuardrailFunctionOutput)> {
        value
            .get(key)
            .and_then(Value::as_array)
            .map(|rows| {
                let expected = key.trim_end_matches("_guardrail_results");
                rows.iter()
                    .filter(|r| tool_guardrail_row_matches(r, expected))
                    .map(|r| {
                        (
                            guardrail_name(r),
                            r.get("tool_name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            tool_guardrail_output_from_wire(r.get("output")),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let string_field = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    let rust_usize = |key: &str, default: usize| {
        rust.get(key)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let rust_bool = |key: &str| rust.get(key).and_then(Value::as_bool).unwrap_or(false);

    // Python does not write the `rust` key; its own signal for where the current turn's items
    // start is the ownership range, and failing that the tail the parked response and its
    // approvals occupy.
    let default_turn_start = value
        .get("current_response_generated_item_ownership")
        .and_then(|o| o.get("start"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            let turn_items = value
                .get("last_processed_response")
                .and_then(|p| p.get("new_items"))
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
                + interruptions.len();
            generated_items.len().saturating_sub(turn_items)
        });

    let mut state = RunState {
        schema_version: value
            .get("$schemaVersion")
            .and_then(Value::as_str)
            .unwrap_or(CURRENT_SCHEMA_VERSION)
            .to_string(),
        current_turn: value
            .get("current_turn")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        max_turns: value
            .get("max_turns")
            .and_then(Value::as_u64)
            .map(|v| v as usize),
        current_agent_name: agent_name_of(value.get("current_agent")),
        current_agent_identity: agent_identity_of(value.get("current_agent")),
        original_input: input_items(value.get("original_input")),
        generated_items,
        model_responses: value
            .get("model_responses")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(model_response_from_wire)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        usage: context
            .get("usage")
            .map(usage_from_wire)
            .unwrap_or_default(),
        approvals: approvals_from_wire(&context, version),
        approval_capabilities: approval_capabilities_from_wire(&rust),
        nested_agent_tool_states: rust
            .get("nested_agent_tool_states")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default(),
        input_guardrail_results,
        output_guardrail_results,
        tool_input_guardrail_results: tool_results("tool_input_guardrail_results")
            .into_iter()
            .map(
                |(guardrail_name, tool_name, output)| ToolInputGuardrailResult {
                    guardrail_name,
                    tool_name,
                    output,
                },
            )
            .collect(),
        tool_output_guardrail_results: tool_results("tool_output_guardrail_results")
            .into_iter()
            .map(
                |(guardrail_name, tool_name, output)| ToolOutputGuardrailResult {
                    guardrail_name,
                    tool_name,
                    output,
                },
            )
            .collect(),
        pending_tool_processing: rust_bool("pending_tool_processing") || !interruptions.is_empty(),
        interruptions,
        pending_input: input_items(value.get("pending_input")),
        previous_response_id: string_field("previous_response_id"),
        conversation_id: string_field("conversation_id"),
        auto_previous_response_id: value
            .get("auto_previous_response_id")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        server_sent_items: rust_usize("server_sent_items", 0),
        server_input_sent: rust_bool("server_input_sent"),
        session_saved_items: rust_usize("session_saved_items", 0),
        withheld_output_call_ids: rust
            .get("withheld_output_call_ids")
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        input_saved_to_session: rust_bool("input_saved_to_session"),
        final_output: rust.get("final_output").cloned().filter(|v| !v.is_null()),
        trace_id: value
            .get("trace")
            .and_then(|t| t.get("id").or_else(|| t.get("trace_id")))
            .and_then(Value::as_str)
            .map(str::to_string),
        // A bare `{object, id}` trace carries nothing `trace_id` does not already hold, so it
        // stays out of `trace_state` and the round trip is stable.
        // Python writes the key only when it is `true`, and only accepts it as `true`.
        terminal_unrecoverable: value.get("terminal_unrecoverable") == Some(&Value::Bool(true)),
        pending_session_write: value
            .get("pending_session_write")
            .filter(|v| v.is_object())
            .cloned(),
        tool_invocations: tool_invocations_from_wire(&context, version)?,
        trace_state: crate::tracing::TraceState::from_json(value.get("trace")).filter(|t| {
            t.workflow_name.is_some()
                || t.group_id.is_some()
                || t.metadata.is_some()
                || t.tracing_api_key.is_some()
                || t.tracing_api_key_hash.is_some()
                || !t.extra.is_empty()
        }),
        context: context.get("context").cloned().filter(|v| !v.is_null()),
        context_requires_restoration: context_meta_requires_restoration(
            context.get("context_meta"),
        ),
        // Set by `RunState::from_json` from the payload version, which this literal does not see.
        legacy_hook_identity_snapshot: false,
        legacy_name_owner_snapshot: false,
        nested_history_lineage_unknown: false,
        hook_identity_ambiguous: false,
        agent_owner_names: owner_names_from_wire(value),
        ambiguous_agent_owners: Default::default(),
        restored_function_approval_keys: Default::default(),
        session_write_in_progress: Default::default(),
        tool_result_may_end_run: value
            .get("tool_result_may_end_run")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        turn_start_index: rust_usize("turn_start_index", default_turn_start),
        model_input_start: rust_usize("model_input_start", 0),
        tool_use_tracker: tool_use_tracker_from_wire(value.get("tool_use_tracker")),
        sandbox: match value.get("sandbox") {
            None => None,
            Some(sandbox) => Some(super::sandbox_payload::sanitize_sandbox_payload(
                sandbox.clone(),
            )?),
        },
        generated_prompt_cache_key: value
            .get("generated_prompt_cache_key")
            .and_then(Value::as_str)
            .map(str::to_string),
        reasoning_item_id_policy: match value
            .get("reasoning_item_id_policy")
            .and_then(Value::as_str)
        {
            Some("omit") => Some(crate::run_config::ReasoningItemIdPolicy::Omit),
            Some("preserve") => Some(crate::run_config::ReasoningItemIdPolicy::Preserve),
            _ => None,
        },
        server_conversation: rust
            .get("server_conversation")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default(),
        last_processed_response: match value.get("last_processed_response") {
            Some(Value::Object(_)) => Some(processed_response_from_wire(
                &value["last_processed_response"],
            )?),
            _ => None,
        },
        // Derived by `RunState::from_json` once the schema version is known; the wire format
        // never carries it.
        restored_unbound_approval_call_ids: Default::default(),
        foreign_owner_call_ids: Default::default(),
        nested_history_owned_items: Vec::new(),
        tool_input: context.get("tool_input").cloned().filter(|v| !v.is_null()),
        current_response_boundary_unproven: !current_response_boundary_proven(value, version),
    };
    super::nested_history::check_generated_session_indexes(value);
    state.nested_history_owned_items =
        super::nested_history::ownership_from_wire(value, &state, &keep_generated)?;
    state.remove_generated_items(&keep_generated);
    Ok(state)
}

fn input_items(value: Option<&Value>) -> Vec<InputItem> {
    match value {
        Some(Value::Array(items)) => items.iter().cloned().map(InputItem).collect(),
        // Python allows a bare string input.
        Some(Value::String(s)) => vec![InputItem::user(s.clone())],
        _ => Vec::new(),
    }
}

/// `true` when the payload is a snapshot written by this crate before the Python envelope.
pub(super) fn is_legacy_rust_envelope(value: &Value) -> bool {
    value.get("current_agent_name").is_some() && value.get("current_agent").is_none()
}

/// Read a pre-envelope Rust snapshot (`$schemaVersion` `"1.0"`, flat fields).
pub(super) fn from_legacy_rust(value: Value) -> Result<RunState> {
    serde_json::from_value(value).map_err(|e| {
        AgentsError::user(format!(
            "Run state is not a readable 1.0 Rust snapshot: {e}"
        ))
    })
}

/// Read the tool-use tracker, dropping entries that are not strings instead of the whole
/// snapshot. Mirrors `set_tool_use_tracker_snapshot`'s filtering.
fn tool_use_tracker_from_wire(value: Option<&Value>) -> ToolUseTrackerSnapshot {
    let Some(Value::Object(map)) = value else {
        return ToolUseTrackerSnapshot::new();
    };
    map.iter()
        .filter_map(|(agent, tools)| {
            let tools = tools.as_array()?;
            Some((
                agent.clone(),
                tools
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect(),
            ))
        })
        .collect()
}

/// A tool-use snapshot keyed by agent name. Mirrors `serialize_tool_use_tracker`.
pub type ToolUseTrackerSnapshot = BTreeMap<String, Vec<String>>;

/// Rename the Python field name `async_` of a function call to its provider name `async`.
///
/// Python's pydantic model stores the provider's `async` field as `async_` and writes that name
/// into snapshots. The name is only a spelling of the same field, so a snapshot holding it is
/// restored with the name the model API expects. Only function-call items are rewritten.
pub(super) fn rename_python_async_alias(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("function_call")
                && map.contains_key("async_")
                && !map.contains_key("async")
                && let Some(field) = map.remove("async_")
            {
                map.insert("async".into(), field);
            }
            for child in map.values_mut() {
                rename_python_async_alias(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                rename_python_async_alias(item);
            }
        }
        _ => {}
    }
}
