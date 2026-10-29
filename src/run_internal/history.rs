//! Nested handoff history: collapse the transcript into one assistant summary message.
//! Simplified port of `agents.handoffs.history`.

use serde_json::{Value, json};

use crate::items::InputItem;

const PREAMBLE: &str =
    "For context, here is the conversation so far between the user and the previous agent:";
const START: &str = "<CONVERSATION HISTORY>";
const END: &str = "</CONVERSATION HISTORY>";

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
    let start = content.find(START)?;
    let end = content.find(END)?;
    if end <= start {
        return None;
    }
    let body = &content[start + START.len()..end];
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
    let mut content = vec![PREAMBLE.to_string(), START.to_string()];
    if lines.is_empty() {
        content.push("(no previous turns recorded)".to_string());
    } else {
        content.extend(lines.iter().enumerate().map(|(i, l)| format!("{}. {l}", i + 1)));
    }
    content.push(END.to_string());
    InputItem(json!({"role": "assistant", "content": content.join("\n")}))
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
        assert_eq!(text.matches(START).count(), 1);
    }
}
