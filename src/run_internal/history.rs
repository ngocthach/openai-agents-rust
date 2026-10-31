//! Nested handoff history: summarize the transcript for the next agent, forwarding the
//! items the summary does not already cover. Port of `agents.handoffs.history`.

use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

use crate::items::{InputItem, RunItem, RunItemKind};

const PREAMBLE: &str =
    "For context, here is the conversation so far between the user and the previous agent:";
/// Preamble of summaries written before the preamble named the previous agent.
const LEGACY_PREAMBLE: &str = "For context, here is the conversation so far:";
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
        return python_style_json(&item);
    }
    python_style_json(&item)
}

/// Render a value the way Python's `json.dumps(value, ensure_ascii=False)` does, with `", "` and
/// `": "` separators, because the summary text the model reads is the Python rendering.
fn python_style_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), python_style_json(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(python_style_json).collect();
            format!("[{}]", parts.join(", "))
        }
        other => other.to_string(),
    }
}

/// Flatten earlier summary messages back into the transcript items they summarize, keeping
/// every other item as is. Port of `_flatten_nested_history_messages`.
pub(crate) fn flatten_nested_history_messages(items: &[InputItem]) -> Vec<InputItem> {
    let mut flattened = Vec::new();
    for item in items {
        match extract_nested_history_transcript(&item.0) {
            Some(transcript) => flattened.extend(transcript.into_iter().map(InputItem)),
            None => flattened.push(item.clone()),
        }
    }
    flattened
}

/// The items recorded in an earlier summary message, or `None` when `item` is not one.
/// Port of `_extract_nested_history_transcript`.
fn extract_nested_history_transcript(item: &Value) -> Option<Vec<Value>> {
    if item.get("role").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    let content = item.get("content")?.as_str()?;
    let (preamble, wrapped) = content.split_once('\n')?;
    if preamble != PREAMBLE && preamble != LEGACY_PREAMBLE {
        return None;
    }
    let (start_marker, end_marker) = get_conversation_history_wrappers();
    let start_wrapper = format!("{start_marker}\n");
    let end_wrapper = format!("\n{end_marker}");
    if !wrapped.starts_with(&start_wrapper) || !wrapped.ends_with(&end_wrapper) {
        return None;
    }
    // Python slices `wrapped[len(start):-len(end)]`, which is empty when the two overlap.
    let body_end = wrapped.len() - end_wrapper.len();
    let body = if body_end >= start_wrapper.len() {
        &wrapped[start_wrapper.len()..body_end]
    } else {
        ""
    };
    Some(
        split_summary_records(body)
            .iter()
            .filter_map(|record| parse_summary_line(record))
            .collect(),
    )
}

/// Split the body of a summary into records. A numbered record absorbs the following
/// unnumbered lines, which are message content that spans lines. Port of
/// `_split_summary_records`.
fn split_summary_records(body: &str) -> Vec<String> {
    let mut records: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_is_numbered = false;
    for raw_line in body.split('\n') {
        if raw_line.trim().is_empty() {
            continue;
        }
        let starts_numbered = starts_numbered_summary_record(raw_line);
        if current.is_empty() {
            current = vec![raw_line.trim().to_string()];
            current_is_numbered = starts_numbered;
            continue;
        }
        if starts_numbered || !current_is_numbered {
            records.push(current.join("\n"));
            current = vec![raw_line.trim().to_string()];
            current_is_numbered = starts_numbered;
            continue;
        }
        current.push(raw_line.trim_end().to_string());
    }
    if !current.is_empty() {
        records.push(current.join("\n"));
    }
    records
}

fn starts_numbered_summary_record(line: &str) -> bool {
    let stripped = line.trim_start();
    match stripped.find('.') {
        Some(dot) => {
            let number = &stripped[..dot];
            !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}

/// Parse one summary record back into an input item. Port of `_parse_summary_line`.
fn parse_summary_line(line: &str) -> Option<Value> {
    let stripped = line.trim();
    if stripped.is_empty() {
        return None;
    }
    let stripped = strip_summary_line_number(stripped);
    if let Some(item) = parse_summary_json_item(stripped) {
        return Some(item);
    }
    let Some((role_part, remainder)) = stripped.split_once(':') else {
        // Summaries written before the separator was always emitted record an empty turn as a
        // bare role. Recover those, restricted to a lone role with an optional "(name)".
        if !is_bare_role_record(stripped) {
            return None;
        }
        let (role, name) = split_role_and_name(stripped);
        let mut record = json!({"role": role});
        if let Some(name) = name {
            record["name"] = json!(name);
        }
        record["content"] = json!("");
        return Some(record);
    };
    let role_text = role_part.trim();
    if role_text.is_empty() {
        return None;
    }
    let (role, name) = split_role_and_name(role_text);
    let content = remainder.trim();
    if !content.is_empty()
        && let Some(item) = parse_legacy_typed_item(&role, content)
    {
        return Some(item);
    }
    let mut record = json!({"role": role});
    if let Some(name) = name {
        record["name"] = json!(name);
    }
    // Keep an explicit content even when empty, so the turn stays a replayable message.
    record["content"] = json!(content);
    Some(record)
}

fn strip_summary_line_number(stripped: &str) -> &str {
    if starts_numbered_summary_record(stripped)
        && let Some(dot) = stripped.find('.')
    {
        return stripped[dot + 1..].trim_start();
    }
    stripped
}

fn parse_summary_json_item(value: &str) -> Option<Value> {
    let Ok(Value::Object(mut item)) = serde_json::from_str::<Value>(value) else {
        return None;
    };
    item.remove("provider_data");
    Some(Value::Object(item))
}

/// `function_call: {...}`-style records written by the Python SDK. Port of
/// `_parse_legacy_typed_item`.
fn parse_legacy_typed_item(item_type: &str, content: &str) -> Option<Value> {
    if matches!(item_type, "assistant" | "user" | "system" | "developer") {
        return None;
    }
    let Ok(Value::Object(mut item)) = serde_json::from_str::<Value>(content) else {
        return None;
    };
    item.remove("provider_data");
    item.insert("type".to_string(), json!(item_type));
    Some(Value::Object(item))
}

fn is_bare_role_record(stripped: &str) -> bool {
    let candidate = match stripped.rfind('(') {
        Some(open) if stripped.ends_with(')') => stripped[..open].trim(),
        _ => stripped,
    };
    matches!(candidate, "user" | "assistant" | "system" | "developer")
}

/// Split `role (name)` into its role and name. Port of `_split_role_and_name`.
fn split_role_and_name(role_text: &str) -> (String, Option<String>) {
    if role_text.ends_with(')')
        && let Some(open) = role_text.rfind('(')
    {
        let name = role_text[open + 1..role_text.len() - 1].trim();
        if !name.is_empty() {
            let role = role_text[..open].trim();
            let role = if role.is_empty() { "developer" } else { role };
            return (role.to_string(), Some(name.to_string()));
        }
    }
    let role = if role_text.is_empty() {
        "developer"
    } else {
        role_text
    };
    (role.to_string(), None)
}

/// Build the summary message for `transcript`. Earlier summaries are not flattened here; the
/// caller does that. Port of `_build_summary_message`.
pub(crate) fn build_summary_message(transcript: &[InputItem]) -> InputItem {
    let lines: Vec<String> = transcript.iter().map(|i| format_item(&i.0)).collect();
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
        let flattened = flatten_nested_history_messages(&[first, InputItem::user("more")]);
        let second = build_summary_message(&flattened);
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
