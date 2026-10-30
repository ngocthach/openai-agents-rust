//! The `RunState` wire format: Python's `agents.run_state` schema 1.21 envelope.
//!
//! Rust keeps a flat in-memory [`RunState`](super::RunState); this module maps it to and from
//! the keyed envelope the Python SDK reads and writes, so a snapshot taken in one SDK can be
//! resumed in the other. Fields that only exist on the Rust side travel under the `rust` key,
//! which Python ignores.

use std::collections::BTreeMap;

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

/// Schema versions this crate can read. Mirrors Python's `SUPPORTED_SCHEMA_VERSIONS`.
pub(super) const SUPPORTED_SCHEMA_VERSIONS: [&str; 22] = [
    "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12",
    "1.13", "1.14", "1.15", "1.16", "1.17", "1.18", "1.19", "1.20", "1.21",
];

fn agent_ref(name: &str) -> Value {
    json!({ "name": name })
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

fn approvals_to_wire(approvals: &ApprovalsState) -> Value {
    Value::Array(
        approvals
            .0
            .iter()
            .map(|(key, record)| {
                let (agent, tool_key) = split_approval_key(key);
                json!({
                    "agent": agent_ref(&agent),
                    "tool_key": tool_key,
                    "decision": approval_decision_to_wire(record),
                })
            })
            .collect(),
    )
}

fn approvals_from_wire(context: &Value) -> ApprovalsState {
    let mut state = ApprovalsState::default();
    if let Some(Value::Array(entries)) = context.get("function_tool_approvals") {
        for entry in entries {
            let agent = agent_name_of(entry.get("agent"));
            let tool_key = entry
                .get("tool_key")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let decision = entry.get("decision").cloned().unwrap_or(Value::Null);
            state.0.insert(
                format!("{agent}\u{1}{tool_key}"),
                approval_decision_from_wire(&decision),
            );
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
        return interruption_to_wire(&ToolApprovalItem {
            agent: item.agent.clone(),
            raw_item: raw_item.clone(),
            tool_name: tool_name.clone(),
        });
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
        id: None,
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
        arguments: obj
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}")
            .to_string(),
        status: obj
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        extra,
    })
}

fn run_item_from_wire(value: &Value) -> Result<RunItem> {
    let mut value = value.clone();
    if let Value::Object(m) = &mut value {
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
        let agent = agent_name_of(m.get("agent"));
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
            m.insert("raw_item".into(), strip_function_call_type(raw));
        }
        // Python-only bookkeeping that has no Rust field.
        for key in [
            "input_id",
            "tool_lookup_key",
            "allow_bare_name_alias",
            "tool_origin",
            "tool_namespace",
            "description",
            "title",
        ] {
            m.remove(key);
        }
    }
    serde_json::from_value(value.clone()).map_err(|e| {
        AgentsError::user(format!(
            "Run state has an unreadable item of type {}: {e}",
            value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("<missing>")
        ))
    })
}

fn interruption_to_wire(item: &ToolApprovalItem) -> Value {
    json!({
        "type": "tool_approval_item",
        "raw_item": item.raw_item.to_value(),
        "agent": agent_ref(&item.agent),
        "tool_name": item.tool_name,
    })
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
    let raw_item = function_call_from_raw(&strip_function_call_type(raw.clone()))
        .ok_or_else(|| AgentsError::user("Run state interruption has an invalid raw_item"))?;
    let tool_name = value
        .get("tool_name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| raw_item.name.clone());
    Ok(ToolApprovalItem {
        agent: agent_name_of(value.get("agent")),
        raw_item,
        tool_name,
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
    tool.insert(
        "lookupKey".into(),
        lookup_key_to_wire(&run.tool_name, run.namespace.as_deref()),
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

/// Serialize the interrupted turn's processed response. Mirrors
/// `RunState._serialize_processed_response`.
fn processed_response_to_wire(processed: &ProcessedResponseSnapshot) -> Value {
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
                .map(interruption_to_wire)
                .collect(),
        ),
    );
    out.insert(
        "mcp_tool_bindings".into(),
        json!(processed.mcp_tool_bindings),
    );
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
        mcp_tool_bindings: value
            .get("mcp_tool_bindings")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, v)| {
                        let labels = v
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_default();
                        (k.clone(), labels)
                    })
                    .collect()
            })
            .unwrap_or_default(),
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
fn tool_invocations_from_wire(context: &Value) -> Result<BTreeMap<String, ToolInvocationRecord>> {
    let Some(raw) = context.get("tool_invocations") else {
        return Ok(BTreeMap::new());
    };
    if raw.is_null() {
        return Ok(BTreeMap::new());
    }
    let map = raw
        .as_object()
        .ok_or_else(|| AgentsError::user("Run state tool_invocations must be a mapping."))?;
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
                "Run state tool_invocations contains an invalid call ID.",
            ));
        }
        let entry = entry
            .as_object()
            .ok_or_else(|| AgentsError::user("Run state tool invocation must be a mapping."))?;
        let invocation_type = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let executed = entry.get("executed").and_then(Value::as_bool);
        let completed = entry.get("completed").and_then(Value::as_bool);
        match (executed, completed) {
            (Some(executed), Some(completed))
                if !invocation_type.is_empty()
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
                    "Run state tool invocation contains invalid lifecycle data.",
                ));
            }
        }
    }
    Ok(out)
}

/// The `trace` payload: the full `TraceState` when the run carried one, otherwise the bare
/// `{object, id}` reference built from `trace_id`. Mirrors `_serialize_trace_data`.
fn trace_payload(state: &RunState) -> Value {
    if let Some(payload) = state.trace_state.as_ref().and_then(|t| t.to_json(false)) {
        return payload;
    }
    state
        .trace_id
        .as_ref()
        .map(|id| json!({"object": "trace", "id": id}))
        .unwrap_or(Value::Null)
}

/// Serialize a run state to Python's schema 1.21 envelope.
pub(super) fn to_wire(state: &RunState) -> Result<Value> {
    let generated_items: Vec<Value> = state.generated_items.iter().map(run_item_to_wire).collect();
    let model_responses: Vec<Value> = state
        .model_responses
        .iter()
        .map(model_response_to_wire)
        .collect();

    let context = json!({
        "usage": usage_to_wire(&state.usage),
        "approvals": Value::Object(Map::new()),
        "function_tool_approvals": approvals_to_wire(&state.approvals),
        "tool_invocations": state.tool_invocations,
        "context": state.context.clone().unwrap_or(Value::Null),
        "context_meta": match &state.context {
            Some(Value::Object(_)) => json!({
                "original_type": "mapping",
                "serialized_via": "mapping",
                "requires_deserializer": false,
                "omitted": false,
            }),
            _ => json!({
                "original_type": "none",
                "serialized_via": "none",
                "requires_deserializer": false,
                "omitted": false,
            }),
        },
    });

    let current_step = if state.pending_tool_processing || !state.interruptions.is_empty() {
        json!({
            "type": "next_step_interruption",
            "data": {
                "interruptions": state
                    .interruptions
                    .iter()
                    .map(interruption_to_wire)
                    .collect::<Vec<_>>(),
            },
        })
    } else {
        Value::Null
    };

    // Python serializes the processed response only while a turn is parked on an
    // interruption, and clears it when the run finishes.
    let last_processed = state
        .last_processed_response
        .as_ref()
        .filter(|_| !state.interruptions.is_empty());
    let ownership = last_processed.and_then(|p| ownership_to_wire(state, p));

    let mut envelope = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": state.current_turn,
        "current_agent": agent_ref(&state.current_agent_name),
        "original_input": state.original_input,
        "pending_input": state.pending_input,
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
        "reasoning_item_id_policy": Value::Null,
        "nested_history_owned_session_item_refs": Value::Array(Vec::new()),
        "generated_session_item_indexes": (0..generated_items.len())
            .map(Value::from)
            .collect::<Vec<_>>(),
        "generated_items": generated_items.clone(),
        "session_items": generated_items,
        "current_step": current_step,
        "last_model_response": model_responses.last().cloned().unwrap_or(Value::Null),
        "last_processed_response": last_processed
            .map(processed_response_to_wire)
            .unwrap_or(Value::Null),
        "current_turn_persisted_item_count": state
            .session_saved_items
            .saturating_sub(state.turn_start_index),
        "trace": trace_payload(state),
        // Rust-only run state. Python ignores unknown top-level keys.
        "rust": {
            "server_sent_items": state.server_sent_items,
            "server_input_sent": state.server_input_sent,
            "session_saved_items": state.session_saved_items,
            "input_saved_to_session": state.input_saved_to_session,
            "nested_agent_tool_states": state.nested_agent_tool_states,
            "pending_tool_processing": state.pending_tool_processing,
            "turn_start_index": state.turn_start_index,
            "final_output": state.final_output,
            "server_conversation": state.server_conversation,
        },
    });
    if state.terminal_unrecoverable
        && let Value::Object(map) = &mut envelope
    {
        map.insert("terminal_unrecoverable".into(), Value::Bool(true));
    }
    if let (Some(ownership), Value::Object(map)) = (ownership, &mut envelope) {
        map.insert(
            "current_response_generated_item_ownership".into(),
            ownership,
        );
    }
    Ok(envelope)
}

/// Rebuild a run state from Python's envelope.
pub(super) fn from_wire(value: &Value) -> Result<RunState> {
    let context = value.get("context").cloned().unwrap_or(Value::Null);
    let rust = value.get("rust").cloned().unwrap_or(Value::Null);

    let generated_items = value
        .get("generated_items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(run_item_from_wire)
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();

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
                rows.iter()
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

    Ok(RunState {
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
        approvals: approvals_from_wire(&context),
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
        tool_invocations: tool_invocations_from_wire(&context)?,
        trace_state: crate::tracing::TraceState::from_json(value.get("trace")).filter(|t| {
            t.workflow_name.is_some()
                || t.group_id.is_some()
                || t.metadata.is_some()
                || t.tracing_api_key.is_some()
                || t.tracing_api_key_hash.is_some()
                || !t.extra.is_empty()
        }),
        context: context.get("context").cloned().filter(|v| !v.is_null()),
        turn_start_index: rust_usize("turn_start_index", default_turn_start),
        tool_use_tracker: value
            .get("tool_use_tracker")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default(),
        generated_prompt_cache_key: value
            .get("generated_prompt_cache_key")
            .and_then(Value::as_str)
            .map(str::to_string),
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
    })
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

/// A tool-use snapshot keyed by agent name. Mirrors `serialize_tool_use_tracker`.
pub type ToolUseTrackerSnapshot = BTreeMap<String, Vec<String>>;
