//! Rollout payloads: the memory-filtered record of one run segment. Mirrors
//! `agents.sandbox.memory.rollouts`.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::items::{InputItem, RunItem};
use crate::result::RunResult;

/// Item types that never reach a memory rollout.
const EXCLUDED_MEMORY_ITEM_TYPES: &[&str] = &["compaction", "image_generation_call", "reasoning"];

/// Item types kept in a memory rollout, alongside assistant and user messages.
const INCLUDED_MEMORY_ITEM_TYPES: &[&str] = &[
    "apply_patch_call",
    "apply_patch_call_output",
    "computer_call",
    "computer_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "function_call",
    "function_call_output",
    "local_shell_call",
    "local_shell_call_output",
    "mcp_approval_request",
    "mcp_approval_response",
    "mcp_call",
    "program",
    "program_output",
    "shell_call",
    "shell_call_output",
    "tool_search_call",
    "tool_search_output",
    "web_search_call",
];

/// How a run segment ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalState {
    /// The run produced a final output.
    Completed,
    /// The run paused for approvals.
    Interrupted,
    /// The run was cancelled.
    Cancelled,
    /// The run failed with an error.
    Failed,
    /// The run exceeded its turn limit.
    MaxTurnsExceeded,
    /// A guardrail tripwire stopped the run.
    GuardrailTripped,
}

impl TerminalState {
    /// The wire name of the terminal state.
    pub fn as_str(self) -> &'static str {
        match self {
            TerminalState::Completed => "completed",
            TerminalState::Interrupted => "interrupted",
            TerminalState::Cancelled => "cancelled",
            TerminalState::Failed => "failed",
            TerminalState::MaxTurnsExceeded => "max_turns_exceeded",
            TerminalState::GuardrailTripped => "guardrail_tripped",
        }
    }
}

/// Why and how a run segment ended, recorded alongside the rollout items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutTerminalMetadata {
    /// How the segment ended.
    pub terminal_state: TerminalState,
    /// The error type, when the segment failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exception_type: Option<String>,
    /// The error message, when the segment failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exception_message: Option<String>,
    /// Whether the segment produced a final output.
    pub has_final_output: bool,
}

impl RolloutTerminalMetadata {
    /// Metadata for a terminal state with no error.
    pub fn new(terminal_state: TerminalState, has_final_output: bool) -> Self {
        RolloutTerminalMetadata {
            terminal_state,
            exception_type: None,
            exception_message: None,
            has_final_output,
        }
    }

    /// Metadata for a failed segment.
    pub fn failed(error: &crate::error::AgentsError) -> Self {
        RolloutTerminalMetadata {
            terminal_state: if error.is_max_turns_exceeded() {
                TerminalState::MaxTurnsExceeded
            } else if error.is_guardrail_tripwire() {
                TerminalState::GuardrailTripped
            } else {
                TerminalState::Failed
            },
            exception_type: Some(error_type_name(error).to_string()),
            exception_message: Some(error.to_string()),
            has_final_output: false,
        }
    }
}

fn error_type_name(error: &crate::error::AgentsError) -> &'static str {
    use crate::error::AgentsError;
    match error {
        AgentsError::MaxTurnsExceeded { .. } => "MaxTurnsExceeded",
        AgentsError::ModelBehaviorError { .. } => "ModelBehaviorError",
        AgentsError::ModelRefusalError { .. } => "ModelRefusalError",
        AgentsError::ModelTimeoutError { .. } => "ModelTimeoutError",
        AgentsError::UserError { .. } => "UserError",
        AgentsError::ToolTimeoutError { .. } => "ToolTimeoutError",
        AgentsError::InputGuardrailTripwireTriggered { .. } => "InputGuardrailTripwireTriggered",
        AgentsError::OutputGuardrailTripwireTriggered { .. } => "OutputGuardrailTripwireTriggered",
        AgentsError::ToolInputGuardrailTripwireTriggered { .. } => {
            "ToolInputGuardrailTripwireTriggered"
        }
        AgentsError::ToolOutputGuardrailTripwireTriggered { .. } => {
            "ToolOutputGuardrailTripwireTriggered"
        }
        AgentsError::ApiError { .. } => "ApiError",
        _ => "Error",
    }
}

/// Whether an item belongs in a memory rollout.
pub fn should_include_memory_item(item: &Value) -> bool {
    match item.get("role").and_then(Value::as_str) {
        Some("developer") | Some("system") => return false,
        Some(_) => return true,
        None => {}
    }
    let Some(item_type) = item.get("type").and_then(Value::as_str) else {
        return false;
    };
    if EXCLUDED_MEMORY_ITEM_TYPES.contains(&item_type) {
        return false;
    }
    INCLUDED_MEMORY_ITEM_TYPES.contains(&item_type)
}

/// Serialize one JSONL record.
pub fn dump_rollout_json(payload: &Value) -> String {
    format!("{}\n", serde_json::to_string(payload).unwrap_or_default())
}

/// Build the memory rollout payload of one run segment.
pub fn build_rollout_payload(
    input: &[InputItem],
    new_items: &[RunItem],
    final_output: Option<&Value>,
    interruption_count: usize,
    terminal_metadata: &RolloutTerminalMetadata,
) -> Value {
    let filtered_input: Vec<Value> = input
        .iter()
        .map(|i| i.0.clone())
        .filter(should_include_memory_item)
        .collect();
    let generated_items: Vec<Value> = new_items
        .iter()
        .map(|item| item.to_input_item().0)
        .filter(should_include_memory_item)
        .collect();
    json!({
        "updated_at": chrono::Utc::now().to_rfc3339(),
        "input": filtered_input,
        "generated_items": generated_items,
        "final_output": final_output.cloned().unwrap_or(Value::Null),
        "interruptions": interruption_count,
        "terminal_metadata": terminal_metadata,
    })
}

/// Build the memory rollout payload from a finished [`RunResult`].
pub fn build_rollout_payload_from_result<C>(result: &RunResult<C>) -> Value {
    let terminal_metadata = if result.interruptions.is_empty() {
        RolloutTerminalMetadata::new(TerminalState::Completed, result.final_output.is_some())
    } else {
        RolloutTerminalMetadata::new(TerminalState::Interrupted, false)
    };
    build_rollout_payload(
        &result.input,
        &result.new_items,
        result.final_output.as_ref(),
        result.interruptions.len(),
        &terminal_metadata,
    )
}

/// Build the phase-1 user prompt from a rollout file's JSONL contents.
///
/// Returns `None` when the file holds no records.
pub fn render_phase_one_prompt(rollout_contents: &str, token_limit: usize) -> Option<String> {
    use crate::sandbox::memory::prompts::render_rollout_extraction_user_prompt;
    use crate::sandbox::util::truncate_text_by_tokens;

    let payloads: Vec<Value> = rollout_contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect();
    let last = payloads.last()?;
    let terminal_metadata = if payloads.len() == 1 {
        last.get("terminal_metadata").cloned().unwrap_or(json!({}))
    } else {
        json!({
            "segment_count": payloads.len(),
            "final_terminal_metadata": last.get("terminal_metadata").cloned().unwrap_or(json!({})),
            "terminal_states": payloads
                .iter()
                .map(|item| item
                    .get("terminal_metadata")
                    .and_then(|m| m.get("terminal_state"))
                    .cloned()
                    .unwrap_or(json!("unknown")))
                .collect::<Vec<_>>(),
        })
    };
    let terminal_metadata_json =
        serde_json::to_string_pretty(&terminal_metadata).unwrap_or_default();
    let (truncated, _) = truncate_text_by_tokens(rollout_contents, Some(token_limit));
    let truncated = if truncated == rollout_contents {
        truncated
    } else {
        format!(
            "\n\n[rollout content omitted: this phase-one memory prompt contains a truncated view \
             of the saved rollout. original_chars={}; rendered_chars={}. Do not assume the \
             rendered rollout below is complete.]\n\n{truncated}",
            rollout_contents.len(),
            truncated.len(),
        )
    };
    Some(render_rollout_extraction_user_prompt(
        &terminal_metadata_json,
        &truncated,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_memory_items() {
        assert!(should_include_memory_item(
            &json!({"role": "user", "content": "hi"})
        ));
        assert!(!should_include_memory_item(
            &json!({"role": "system", "content": "x"})
        ));
        assert!(!should_include_memory_item(
            &json!({"role": "developer", "content": "x"})
        ));
        assert!(should_include_memory_item(&json!({"type": "shell_call"})));
        assert!(!should_include_memory_item(&json!({"type": "reasoning"})));
        assert!(!should_include_memory_item(&json!({"type": "compaction"})));
        assert!(!should_include_memory_item(
            &json!({"type": "unknown_type"})
        ));
    }

    #[test]
    fn builds_a_rollout_payload() {
        let input = vec![
            InputItem::from_value(json!({"role": "user", "content": "do it"})),
            InputItem::from_value(json!({"role": "system", "content": "secret"})),
        ];
        let metadata = RolloutTerminalMetadata::new(TerminalState::Completed, true);
        let payload = build_rollout_payload(&input, &[], Some(&json!("done")), 0, &metadata);
        assert_eq!(payload["input"].as_array().unwrap().len(), 1);
        assert_eq!(payload["final_output"], json!("done"));
        assert_eq!(payload["terminal_metadata"]["terminal_state"], "completed");
        assert_eq!(payload["terminal_metadata"]["has_final_output"], true);
        assert!(payload["updated_at"].is_string());
        assert!(dump_rollout_json(&payload).ends_with('\n'));
    }

    #[test]
    fn renders_the_phase_one_prompt() {
        assert!(render_phase_one_prompt("", 1000).is_none());

        let single = render_phase_one_prompt(
            "{\"terminal_metadata\":{\"terminal_state\":\"completed\"}}\n",
            100_000,
        )
        .unwrap();
        assert!(single.contains("\"terminal_state\": \"completed\""));
        assert!(!single.contains("segment_count"));

        let multi = render_phase_one_prompt(
            "{\"terminal_metadata\":{\"terminal_state\":\"interrupted\"}}\n\
             {\"terminal_metadata\":{\"terminal_state\":\"completed\"}}\n",
            100_000,
        )
        .unwrap();
        assert!(multi.contains("\"segment_count\": 2"));
        assert!(multi.contains("interrupted"));

        let long = "{\"terminal_metadata\":{}}\n".repeat(500);
        let truncated = render_phase_one_prompt(&long, 10).unwrap();
        assert!(truncated.contains("rollout content omitted"));
    }

    #[test]
    fn maps_errors_to_terminal_states() {
        let error = crate::error::AgentsError::MaxTurnsExceeded {
            message: "too many".into(),
            run_data: None,
        };
        let metadata = RolloutTerminalMetadata::failed(&error);
        assert_eq!(metadata.terminal_state, TerminalState::MaxTurnsExceeded);
        assert_eq!(metadata.exception_type.as_deref(), Some("MaxTurnsExceeded"));
    }
}
