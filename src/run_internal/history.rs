//! Nested handoff history: summarize the transcript for the next agent, forwarding the
//! items the summary does not already cover. Port of `agents.handoffs.history`.

use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

use crate::items::{InputItem, RunItem, RunItemKind};

const PREAMBLE: &str =
    "For context, here is the conversation so far between the user and the previous agent:";
/// Default opening marker of the nested conversation history block.
pub const DEFAULT_HISTORY_START: &str = "<CONVERSATION HISTORY>";
/// Default closing marker of the nested conversation history block.
pub const DEFAULT_HISTORY_END: &str = "</CONVERSATION HISTORY>";

/// Item types the summary already covers, so they are not forwarded verbatim.
const SUMMARY_ONLY_INPUT_TYPES: [&str; 3] = ["function_call", "function_call_output", "reasoning"];

fn wrappers() -> &'static Mutex<(String, String)> {
    static WRAPPERS: OnceLock<Mutex<(String, String)>> = OnceLock::new();
    WRAPPERS.get_or_init(|| {
        Mutex::new((
            DEFAULT_HISTORY_START.to_string(),
            DEFAULT_HISTORY_END.to_string(),
        ))
    })
}

/// The markers that delimit the nested conversation history block.
pub fn get_conversation_history_wrappers() -> (String, String) {
    wrappers()
        .lock()
        .expect("history wrappers poisoned")
        .clone()
}

/// Override the markers that delimit the nested conversation history block.
/// Passing `None` keeps the current value for that side.
pub fn set_conversation_history_wrappers(start: Option<&str>, end: Option<&str>) {
    let mut w = wrappers().lock().expect("history wrappers poisoned");
    if let Some(s) = start.map(str::trim).filter(|s| !s.is_empty()) {
        w.0 = s.to_string();
    }
    if let Some(e) = end.map(str::trim).filter(|e| !e.is_empty()) {
        w.1 = e.to_string();
    }
}

/// Restore the default conversation history markers.
pub fn reset_conversation_history_wrappers() {
    let mut w = wrappers().lock().expect("history wrappers poisoned");
    w.0 = DEFAULT_HISTORY_START.to_string();
    w.1 = DEFAULT_HISTORY_END.to_string();
}

/// Whether an item belongs to an indivisible hosted-program transcript.
fn is_programmatic_transcript_item(item: &Value) -> bool {
    if matches!(
        item.get("type").and_then(Value::as_str),
        Some("program") | Some("program_output")
    ) {
        return true;
    }
    item.get("caller")
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("program")
}

fn is_summary_only_type(item: &Value) -> bool {
    item.get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| SUMMARY_ONLY_INPUT_TYPES.contains(&t))
}

/// `false` when an item generated before the handoff turn is already in the summary.
fn should_forward_pre_item(item: &Value) -> bool {
    if is_programmatic_transcript_item(item) {
        return false;
    }
    if item.get("role").and_then(Value::as_str) == Some("assistant") {
        return false;
    }
    !is_summary_only_type(item)
}

/// `false` for tool or side-effect items of the handoff turn that the summary covers.
fn should_forward_new_item(item: &Value) -> bool {
    if is_programmatic_transcript_item(item) {
        return false;
    }
    if item
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|r| !r.is_empty())
    {
        return true;
    }
    !is_summary_only_type(item)
}

fn stringify(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn format_item(item: &Value) -> String {
    let mut item = item.clone();
    if let Value::Object(m) = &mut item {
        m.remove("provider_data");
    }
    if let Some(role) = item.get("role").and_then(Value::as_str) {
        let content = item.get("content").cloned().unwrap_or(Value::Null);
        let single_line = matches!(&content, Value::Null)
            || matches!(&content, Value::String(s) if !s.contains('\n') && !s.contains('\r'));
        if single_line {
            let prefix = match item.get("name").and_then(Value::as_str) {
                Some(n) if !n.is_empty() => format!("{role} ({n})"),
                _ => role.to_string(),
            };
            return format!("{prefix}: {}", stringify(&content));
        }
        return item.to_string();
    }
    item.to_string()
}

/// Lines of an earlier summary message, if `item` is one.
fn nested_lines(item: &Value) -> Option<Vec<String>> {
    if item.get("role").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    let content = item.get("content")?.as_str()?;
    let (start_marker, end_marker) = get_conversation_history_wrappers();
    let start = content.find(&start_marker)?;
    let end = content.find(&end_marker)?;
    if end <= start {
        return None;
    }
    let body = &content[start + start_marker.len()..end];
    Some(
        body.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && *l != "(no previous turns recorded)")
            .map(|l| match l.split_once(". ") {
                Some((n, rest)) if n.chars().all(|c| c.is_ascii_digit()) => rest.to_string(),
                _ => l.to_string(),
            })
            .collect(),
    )
}

/// Build the summary message for `transcript`, flattening earlier summaries.
pub(crate) fn build_summary_message(transcript: &[InputItem]) -> InputItem {
    let mut lines: Vec<String> = Vec::new();
    for item in transcript {
        match nested_lines(&item.0) {
            Some(nested) => lines.extend(nested),
            None => lines.push(format_item(&item.0)),
        }
    }
    let (start_marker, end_marker) = get_conversation_history_wrappers();
    let mut content = vec![PREAMBLE.to_string(), start_marker];
    if lines.is_empty() {
        content.push("(no previous turns recorded)".to_string());
    } else {
        content.extend(
            lines
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{}. {l}", i + 1)),
        );
    }
    content.push(end_marker);
    InputItem(json!({"role": "assistant", "content": content.join("\n")}))
}

/// Return a single assistant message summarizing `transcript`.
/// Port of `default_handoff_history_mapper`.
pub fn default_handoff_history_mapper(transcript: Vec<InputItem>) -> Vec<InputItem> {
    vec![build_summary_message(&transcript)]
}

/// Build the model input for the next agent: the summary of everything the summary covers,
/// with the remaining items forwarded verbatim in order.
/// Port of `_build_ordered_default_history`.
pub(crate) fn build_ordered_history(
    flattened_history: Vec<InputItem>,
    normalized_items: Vec<(InputItem, bool)>,
) -> Vec<InputItem> {
    let mut history_items: Vec<InputItem> = Vec::new();
    let mut pending_summary = flattened_history;
    for (plain_input, forward_verbatim) in normalized_items {
        if !forward_verbatim {
            pending_summary.push(plain_input);
            continue;
        }
        if !pending_summary.is_empty() || history_items.is_empty() {
            history_items.extend(default_handoff_history_mapper(std::mem::take(
                &mut pending_summary,
            )));
        }
        history_items.push(plain_input);
    }
    if !pending_summary.is_empty() || history_items.is_empty() {
        history_items.extend(default_handoff_history_mapper(pending_summary));
    }
    history_items
}

/// Classify the run items of a handoff into `(plain input, forward verbatim)` pairs.
/// `pre_count` is how many of `items` were generated before the handoff turn.
pub(crate) fn normalize_handoff_items(
    items: &[RunItem],
    pre_count: usize,
) -> Vec<(InputItem, bool)> {
    items
        .iter()
        .enumerate()
        .filter(|(_, item)| !matches!(item.kind, RunItemKind::ToolApproval { .. }))
        .map(|(index, item)| {
            let plain = item.to_input_item();
            let forward = if index < pre_count {
                should_forward_pre_item(&plain.0)
            } else {
                should_forward_new_item(&plain.0)
            };
            (plain, forward)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_and_flattens() {
        let first = build_summary_message(&[InputItem::user("hi"), InputItem::assistant("hello")]);
        let text = first.text_content().unwrap();
        assert!(text.contains("1. user: hi"));
        assert!(text.contains("2. assistant: hello"));
        let second = build_summary_message(&[first, InputItem::user("more")]);
        let text = second.text_content().unwrap();
        assert!(text.contains("3. user: more"));
        assert_eq!(text.matches(DEFAULT_HISTORY_START).count(), 1);
    }

    #[test]
    fn forwards_user_turns_verbatim_and_summarizes_tool_traffic() {
        let items = vec![
            RunItem::new(
                "A",
                RunItemKind::ToolCall {
                    raw_item: json!({"type": "function_call", "call_id": "c1", "name": "t"}),
                },
            ),
            RunItem::new(
                "A",
                RunItemKind::Input {
                    raw_item: InputItem::user("second question"),
                },
            ),
        ];
        let normalized = normalize_handoff_items(&items, 1);
        assert!(!normalized[0].1, "function_call stays in the summary");
        assert!(normalized[1].1, "user turn is forwarded verbatim");
        let history = build_ordered_history(vec![InputItem::user("first")], normalized);
        assert_eq!(history.len(), 2);
        let summary = history[0].text_content().unwrap();
        assert!(summary.contains("1. user: first"));
        assert!(summary.contains("function_call"));
        assert_eq!(
            history[1].text_content().as_deref(),
            Some("second question")
        );
    }

    #[test]
    fn program_items_are_never_forwarded_verbatim() {
        let items = vec![RunItem::new(
            "A",
            RunItemKind::ToolCall {
                raw_item: json!({"type": "program", "call_id": "p1"}),
            },
        )];
        assert!(!normalize_handoff_items(&items, 0)[0].1);
    }
}
