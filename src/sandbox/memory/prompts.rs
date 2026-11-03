//! Memory prompt templates. Mirrors `agents.sandbox.memory.prompts`.
//!
//! The four templates are copied verbatim from the Python SDK and loaded with `include_str!`.

use crate::sandbox::memory::storage::PhaseTwoInputSelection;

/// The phase-2 consolidation prompt template.
pub const MEMORY_CONSOLIDATION_PROMPT_TEMPLATE: &str =
    include_str!("prompts/memory_consolidation_prompt.md");
/// The prompt fragment injected when an agent has memory to read.
pub const MEMORY_READ_PROMPT_TEMPLATE: &str = include_str!("prompts/memory_read_prompt.md");
/// The phase-1 extraction system prompt template.
pub const ROLLOUT_EXTRACTION_PROMPT_TEMPLATE: &str =
    include_str!("prompts/rollout_extraction_prompt.md");
/// The phase-1 extraction user message template.
pub const ROLLOUT_EXTRACTION_USER_MESSAGE_TEMPLATE: &str =
    include_str!("prompts/rollout_extraction_user_message.md");

const EXTRA_PROMPT_PLACEHOLDER: &str = "{{ extra_prompt_section }}";
const PHASE_TWO_INPUT_SELECTION_PLACEHOLDER: &str = "{{ phase_two_input_selection }}";

const EXTRA_PROMPT_SECTION_TEMPLATE: &str =
    "============================================================
DEVELOPER-SPECIFIC EXTRA GUIDANCE
============================================================

The developer provided additional guidance for memory writing. Pay extra attention to
capturing these details when they would be useful for future runs, in addition to the
standard user preferences, failure recovery, and task summary signals. Keep following the
schema, safety, and evidence rules above.

{extra_prompt}
";

/// Instructions used when the agent may only read memory.
pub const MEMORY_READ_ONLY_INSTRUCTIONS: &str = "Never update memories. You can only read them.";

/// Instructions used when the agent may update memory during the run.
pub const MEMORY_LIVE_UPDATE_INSTRUCTIONS: &str = r"When to update memory (automatic, same turn; required):

- Treat memory as guidance, not truth: if memory conflicts with current workspace
  state, tool outputs, environment, or user feedback, current evidence wins.
- Memory is writable. You are authorized to edit {memory_dir}/MEMORY.md when stale
  guidance is detected.
- If any memory fact conflicts with current evidence, you MUST update memory in the
  same turn. Do not wait for a separate user prompt.
- If you detect stale memory, updating {memory_dir}/MEMORY.md is part of task
  completion, not optional cleanup.
- Required behavior after detecting stale memory:
  1. Verify the correct replacement using local evidence.
  2. Continue the task using current evidence; do not rely on stale memory.
  3. Edit {memory_dir}/MEMORY.md later in the same turn, before your final response.
  4. Finalize the task after the memory update is written.";

/// Render the instruction fragment an agent receives when memory is available.
pub fn render_memory_read_prompt(
    memory_dir: &str,
    memory_summary: &str,
    live_update: bool,
) -> String {
    let update_instructions = if live_update {
        MEMORY_LIVE_UPDATE_INSTRUCTIONS.replace("{memory_dir}", memory_dir)
    } else {
        MEMORY_READ_ONLY_INSTRUCTIONS.to_string()
    };
    MEMORY_READ_PROMPT_TEMPLATE
        .replace("{memory_dir}", memory_dir)
        .replace("{memory_update_instructions}", &update_instructions)
        .replace("{memory_summary}", memory_summary)
}

/// Render the phase-2 consolidation prompt.
pub fn render_memory_consolidation_prompt(
    memory_root: &str,
    selection: &PhaseTwoInputSelection,
    extra_prompt: Option<&str>,
) -> String {
    MEMORY_CONSOLIDATION_PROMPT_TEMPLATE
        .replace("{{ memory_root }}", memory_root)
        .replace(
            PHASE_TWO_INPUT_SELECTION_PLACEHOLDER,
            &render_phase_two_input_selection(selection),
        )
        .replace(
            EXTRA_PROMPT_PLACEHOLDER,
            &render_extra_prompt_section(extra_prompt),
        )
}

/// Render the phase-1 extraction system prompt.
pub fn render_rollout_extraction_prompt(extra_prompt: Option<&str>) -> String {
    ROLLOUT_EXTRACTION_PROMPT_TEMPLATE.replace(
        EXTRA_PROMPT_PLACEHOLDER,
        &render_extra_prompt_section(extra_prompt),
    )
}

/// Render the phase-1 extraction user message.
pub fn render_rollout_extraction_user_prompt(
    terminal_metadata_json: &str,
    rollout_contents: &str,
) -> String {
    ROLLOUT_EXTRACTION_USER_MESSAGE_TEMPLATE
        .replace("{terminal_metadata_json}", terminal_metadata_json)
        .replace("{rollout_contents}", rollout_contents)
}

fn render_extra_prompt_section(extra_prompt: Option<&str>) -> String {
    match extra_prompt.map(str::trim).filter(|p| !p.is_empty()) {
        None => String::new(),
        Some(extra_prompt) => format!(
            "\n{}",
            EXTRA_PROMPT_SECTION_TEMPLATE.replace("{extra_prompt}", extra_prompt)
        ),
    }
}

fn render_phase_two_input_selection(selection: &PhaseTwoInputSelection) -> String {
    let retained = selection.retained_rollout_ids.len();
    let added = selection.selected.len().saturating_sub(retained);
    let selected_lines = if selection.selected.is_empty() {
        "- none".to_string()
    } else {
        selection
            .selected
            .iter()
            .map(|item| {
                let status = if selection.retained_rollout_ids.contains(&item.rollout_id) {
                    "retained"
                } else {
                    "added"
                };
                format!(
                    "- [{status}] rollout_id={}, rollout_summary_file={}, updated_at={}",
                    item.rollout_id,
                    item.rollout_summary_file,
                    if item.updated_at.is_empty() {
                        "unknown"
                    } else {
                        &item.updated_at
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let removed_lines = if selection.removed.is_empty() {
        "- none".to_string()
    } else {
        selection
            .removed
            .iter()
            .map(|item| {
                format!(
                    "- rollout_id={}, rollout_summary_file={}, updated_at={}",
                    item.rollout_id,
                    item.rollout_summary_file,
                    if item.updated_at.is_empty() {
                        "unknown"
                    } else {
                        &item.updated_at
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "- selected inputs this run: {}\n\
         - newly added since the last successful Phase 2 run: {added}\n\
         - retained from the last successful Phase 2 run: {retained}\n\
         - removed from the last successful Phase 2 run: {}\n\n\
         Current selected Phase 1 inputs:\n{selected_lines}\n\n\
         Removed from the last successful Phase 2 selection:\n{removed_lines}\n",
        selection.selected.len(),
        selection.removed.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::memory::storage::PhaseTwoSelectionItem;

    #[test]
    fn renders_the_read_prompt() {
        let read_only = render_memory_read_prompt("memories", "a summary", false);
        assert!(read_only.contains("a summary"));
        assert!(read_only.contains(MEMORY_READ_ONLY_INSTRUCTIONS));
        assert!(!read_only.contains("{memory_dir}"));

        let live = render_memory_read_prompt("repo/memories", "a summary", true);
        assert!(live.contains("repo/memories/MEMORY.md"));
        assert!(!live.contains("{memory_update_instructions}"));
    }

    #[test]
    fn renders_the_extraction_prompt_with_extra_guidance() {
        let plain = render_rollout_extraction_prompt(None);
        assert!(!plain.contains(EXTRA_PROMPT_PLACEHOLDER));
        assert!(!plain.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));

        let extra = render_rollout_extraction_prompt(Some("  remember the build command  "));
        assert!(extra.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
        assert!(extra.contains("remember the build command"));

        let user =
            render_rollout_extraction_user_prompt("{\"terminal_state\":\"completed\"}", "{}");
        assert!(user.contains("completed"));
        assert!(!user.contains("{rollout_contents}"));
    }

    #[test]
    fn renders_the_consolidation_selection() {
        let selection = PhaseTwoInputSelection {
            selected: vec![
                PhaseTwoSelectionItem {
                    rollout_id: "r1".into(),
                    updated_at: "2026-01-01T00:00:00Z".into(),
                    rollout_path: "sessions/r1.jsonl".into(),
                    rollout_summary_file: "r1.md".into(),
                    terminal_state: "completed".into(),
                },
                PhaseTwoSelectionItem {
                    rollout_id: "r2".into(),
                    updated_at: String::new(),
                    rollout_path: "sessions/r2.jsonl".into(),
                    rollout_summary_file: "r2.md".into(),
                    terminal_state: "interrupted".into(),
                },
            ],
            retained_rollout_ids: ["r1".to_string()].into_iter().collect(),
            removed: Vec::new(),
        };
        let prompt = render_memory_consolidation_prompt("memories", &selection, None);
        assert!(prompt.contains("- selected inputs this run: 2"));
        assert!(prompt.contains("- newly added since the last successful Phase 2 run: 1"));
        assert!(prompt.contains("- retained from the last successful Phase 2 run: 1"));
        assert!(prompt.contains("[retained] rollout_id=r1"));
        assert!(prompt.contains("[added] rollout_id=r2"));
        assert!(prompt.contains("updated_at=unknown"));
        assert!(prompt.contains("Removed from the last successful Phase 2 selection:\n- none"));
        assert!(!prompt.contains(PHASE_TWO_INPUT_SELECTION_PLACEHOLDER));
    }
}
