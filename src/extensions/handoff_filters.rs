//! Common handoff input filters. Mirrors `agents.extensions.handoff_filters`.

use crate::handoffs::HandoffInputData;
pub use crate::handoffs::{default_handoff_history_mapper, nest_handoff_history};
use crate::items::{InputItem, RunItem, RunItemKind};

const TOOL_TYPES: &[&str] = &[
    "function_call",
    "function_call_output",
    "computer_call",
    "computer_call_output",
    "file_search_call",
    "tool_search_call",
    "tool_search_output",
    "web_search_call",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
    "reasoning",
    "code_interpreter_call",
    "image_generation_call",
    "local_shell_call",
    "local_shell_call_output",
    "shell_call",
    "shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "hosted_tool_call",
    "program",
    "program_output",
];

fn is_tool_item(item: &RunItem) -> bool {
    !matches!(
        item.kind,
        RunItemKind::MessageOutput { .. }
            | RunItemKind::Input { .. }
            | RunItemKind::Compaction { .. }
    )
}

fn remove_tools_from_items(items: Vec<RunItem>) -> Vec<RunItem> {
    items.into_iter().filter(|i| !is_tool_item(i)).collect()
}

fn remove_tool_types_from_input(items: Vec<InputItem>) -> Vec<InputItem> {
    items
        .into_iter()
        .filter(|i| {
            !i.item_type()
                .map(|t| TOOL_TYPES.contains(&t))
                .unwrap_or(false)
        })
        .collect()
}

/// Remove all tool items (function calls and outputs, hosted tool calls, reasoning, handoff
/// items, MCP items) from the history the next agent sees.
pub fn remove_all_tools(data: HandoffInputData) -> HandoffInputData {
    HandoffInputData {
        input_history: remove_tool_types_from_input(data.input_history),
        pre_handoff_items: remove_tools_from_items(data.pre_handoff_items),
        new_items: remove_tools_from_items(data.new_items),
        input_items: data.input_items.map(remove_tools_from_items),
    }
}
