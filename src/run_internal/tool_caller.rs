//! Runtime enforcement of `allowed_callers` and of programmatic tool-call parents.
//! Port of `agents.run_internal.tool_caller`.

use std::collections::HashSet;

use serde_json::Value;

use crate::error::{AgentsError, Result};
use crate::tracing::scope;

/// Who invoked a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCaller {
    /// The model called the tool directly.
    Direct,
    /// A programmatic tool-calling program called the tool.
    Programmatic,
}

impl ToolCaller {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ToolCaller::Direct => "direct",
            ToolCaller::Programmatic => "programmatic",
        }
    }
}

fn attach_span_error(message: &str, data: Value) {
    if let Some(span) = scope::current_span() {
        span.set_error(message, Some(data));
    }
}

fn caller_of(tool_call: &Value) -> Option<&Value> {
    tool_call.get("caller").filter(|v| !v.is_null())
}

fn call_id_of(tool_call: &Value) -> Option<&str> {
    tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(Value::as_str)
}

/// Reject tool calls whose direct or programmatic caller is not allowed.
/// Port of `ensure_tool_caller_allowed`.
pub(crate) fn ensure_tool_caller_allowed(
    tool_call: &Value,
    allowed_callers: Option<&[String]>,
    tool_name: &str,
    agent_name: &str,
) -> Result<()> {
    let caller_value = caller_of(tool_call);
    let caller_type = caller_value
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str);
    let caller = match (caller_value, caller_type) {
        (None, _) => ToolCaller::Direct,
        (Some(_), Some("direct")) => ToolCaller::Direct,
        (Some(_), Some("program")) => ToolCaller::Programmatic,
        (Some(_), other) => {
            let message =
                format!("Model invoked tool {tool_name} with unsupported caller type {other:?}.");
            attach_span_error(
                &message,
                serde_json::json!({
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": call_id_of(tool_call),
                    "tool_caller_type": other,
                }),
            );
            return Err(AgentsError::model_behavior(message));
        }
    };

    let default_allowed = [String::from("direct")];
    let effective: &[String] = allowed_callers.unwrap_or(&default_allowed);
    if effective.iter().any(|c| c == caller.as_str()) {
        return Ok(());
    }

    let rendered = serde_json::to_string(effective).unwrap_or_else(|_| "[]".into());
    let message = format!(
        "Model invoked tool {tool_name} with caller {}, but the tool allows only {rendered}.",
        caller.as_str()
    );
    attach_span_error(
        &message,
        serde_json::json!({
            "agent_name": agent_name,
            "tool_name": tool_name,
            "tool_call_id": call_id_of(tool_call),
            "tool_caller": caller.as_str(),
        }),
    );
    Err(AgentsError::model_behavior(message))
}

/// Reject program-owned calls without an active parent program.
/// Port of `ensure_programmatic_tool_call_parent`.
pub(crate) fn ensure_programmatic_tool_call_parent(
    tool_call: &Value,
    programmatic_tool_present: bool,
    program_call_ids: &HashSet<String>,
    completed_program_call_ids: &HashSet<String>,
    agent_name: &str,
) -> Result<()> {
    let Some(caller) = caller_of(tool_call) else {
        return Ok(());
    };
    if caller.get("type").and_then(Value::as_str) != Some("program") {
        return Ok(());
    }

    let output_type = tool_call
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let caller_id = caller.get("caller_id").and_then(Value::as_str);

    if !programmatic_tool_present {
        let message =
            format!("Model produced {output_type} item without a programmatic_tool_calling tool.");
        attach_span_error(
            "Programmatic tool not found",
            serde_json::json!({"agent_name": agent_name, "output_type": output_type}),
        );
        return Err(AgentsError::model_behavior(message));
    }

    let Some(caller_id) = caller_id.filter(|id| program_call_ids.contains(*id)) else {
        let message = format!(
            "Model produced {output_type} item with a program caller that does not match a \
             parent program item."
        );
        attach_span_error(
            "Program parent not found",
            serde_json::json!({
                "agent_name": agent_name,
                "output_type": output_type,
                "caller_id": caller_id,
            }),
        );
        return Err(AgentsError::model_behavior(message));
    };

    if completed_program_call_ids.contains(caller_id) {
        let message = format!(
            "Model produced {output_type} item with a program caller whose parent program is \
             already completed."
        );
        attach_span_error(
            "Program parent already completed",
            serde_json::json!({
                "agent_name": agent_name,
                "output_type": output_type,
                "caller_id": caller_id,
            }),
        );
        return Err(AgentsError::model_behavior(message));
    }
    Ok(())
}

/// Collect the program parents retained in history and the ones already completed.
/// Port of `_collect_program_parent_state`.
pub(crate) fn collect_program_parent_state(
    raw_items: &[Value],
    server_manages_conversation: bool,
) -> (HashSet<String>, HashSet<String>) {
    let mut program_call_ids = HashSet::new();
    let mut completed_program_call_ids = HashSet::new();
    for raw in raw_items {
        let item_type = raw.get("type").and_then(Value::as_str);
        let call_id = raw
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        match (item_type, call_id) {
            (Some("program"), Some(id)) => {
                program_call_ids.insert(id.to_string());
            }
            (Some("program_output"), Some(id)) => {
                if server_manages_conversation {
                    program_call_ids.insert(id.to_string());
                }
                if raw.get("status").and_then(Value::as_str) == Some("completed") {
                    completed_program_call_ids.insert(id.to_string());
                }
            }
            _ => {}
        }
        if server_manages_conversation
            && let Some(caller) = caller_of(raw)
            && caller.get("type").and_then(Value::as_str) == Some("program")
            && let Some(caller_id) = caller
                .get("caller_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        {
            program_call_ids.insert(caller_id.to_string());
        }
    }
    (program_call_ids, completed_program_call_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn direct_calls_are_allowed_by_default() {
        let call = json!({"type": "function_call", "call_id": "c1"});
        assert!(ensure_tool_caller_allowed(&call, None, "t", "a").is_ok());
    }

    #[test]
    fn program_calls_need_an_explicit_allowance() {
        let call = json!({
            "type": "function_call",
            "call_id": "c1",
            "caller": {"type": "program", "caller_id": "p1"},
        });
        assert!(ensure_tool_caller_allowed(&call, None, "t", "a").is_err());
        let allowed = vec!["programmatic".to_string()];
        assert!(ensure_tool_caller_allowed(&call, Some(&allowed), "t", "a").is_ok());
    }

    #[test]
    fn unsupported_caller_type_is_rejected() {
        let call = json!({"type": "function_call", "caller": {"type": "weird"}});
        assert!(ensure_tool_caller_allowed(&call, None, "t", "a").is_err());
    }

    #[test]
    fn program_parent_must_exist_and_be_open() {
        let call = json!({
            "type": "function_call",
            "caller": {"type": "program", "caller_id": "p1"},
        });
        let empty = HashSet::new();
        assert!(
            ensure_programmatic_tool_call_parent(&call, false, &empty, &empty, "a").is_err(),
            "no programmatic tool"
        );
        assert!(
            ensure_programmatic_tool_call_parent(&call, true, &empty, &empty, "a").is_err(),
            "no parent"
        );
        let parents: HashSet<String> = ["p1".to_string()].into_iter().collect();
        assert!(ensure_programmatic_tool_call_parent(&call, true, &parents, &empty, "a").is_ok());
        assert!(
            ensure_programmatic_tool_call_parent(&call, true, &parents, &parents, "a").is_err(),
            "parent already completed"
        );
    }
}
