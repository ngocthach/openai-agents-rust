//! A `call_model_input_filter` that trims large tool outputs from older turns.
//!
//! Port of `agents.extensions.tool_output_trimmer`. The last `recent_turns` user messages (and
//! everything after them) are never modified; older `function_call_output` and
//! `tool_search_output` items larger than `max_output_chars` are replaced by a compact preview.
//!
//! ```no_run
//! use openai_agents_sdk::RunConfig;
//! use openai_agents_sdk::extensions::tool_output_trimmer::ToolOutputTrimmer;
//!
//! let trimmer = ToolOutputTrimmer::new()
//!     .with_recent_turns(2)
//!     .with_max_output_chars(500)
//!     .with_trimmable_tools(["search", "execute_code"]);
//! let config = RunConfig::<()>::new().with_call_model_input_filter(move |data| {
//!     let trimmer = trimmer.clone();
//!     async move { Ok(trimmer.apply(data.model_data)) }
//! });
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Map, Value};

use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::run_config::ModelInputData;

const PROSE_SCHEMA_KEYWORDS: &[&str] = &["description", "title", "$comment", "examples"];
const SCHEMA_VALUE_KEYWORDS: &[&str] = &[
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
];
const SCHEMA_LIST_KEYWORDS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems"];
const SCHEMA_MAP_KEYWORDS: &[&str] = &[
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
    // `dependencies` values are schemas or property-name lists; lists are kept as is.
    "dependencies",
];

/// Trims bulky tool outputs from older turns before each model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputTrimmer {
    /// Recent user messages whose items are never trimmed (>= 1).
    pub recent_turns: usize,
    /// Outputs above this many characters are trimmed (>= 1).
    pub max_output_chars: usize,
    /// Characters kept as a preview.
    pub preview_chars: usize,
    /// Only trim outputs of these tools (bare or `namespace.name`); `None` trims all.
    pub trimmable_tools: Option<HashSet<String>>,
}

impl Default for ToolOutputTrimmer {
    fn default() -> Self {
        ToolOutputTrimmer {
            recent_turns: 2,
            max_output_chars: 500,
            preview_chars: 200,
            trimmable_tools: None,
        }
    }
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

fn prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Serialize like Python's `json.dumps(..., ensure_ascii=False, sort_keys=True)`.
fn py_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => {
            let sorted: BTreeMap<&String, &Value> = m.iter().collect();
            let body = sorted
                .iter()
                .map(|(k, v)| format!("{}: {}", Value::String((*k).clone()), py_json(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{body}}}")
        }
        other => other.to_string(),
    }
}

impl ToolOutputTrimmer {
    /// A trimmer with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `recent_turns`.
    pub fn with_recent_turns(mut self, n: usize) -> Self {
        self.recent_turns = n;
        self
    }

    /// Set `max_output_chars`.
    pub fn with_max_output_chars(mut self, n: usize) -> Self {
        self.max_output_chars = n;
        self
    }

    /// Set `preview_chars`.
    pub fn with_preview_chars(mut self, n: usize) -> Self {
        self.preview_chars = n;
        self
    }

    /// Only trim outputs of these tools.
    pub fn with_trimmable_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.trimmable_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Check the settings.
    pub fn validate(&self) -> Result<()> {
        if self.recent_turns < 1 {
            return Err(AgentsError::user("recent_turns must be >= 1"));
        }
        if self.max_output_chars < 1 {
            return Err(AgentsError::user("max_output_chars must be >= 1"));
        }
        Ok(())
    }

    /// Apply the trimmer to model input. Items are copied, never mutated in place.
    pub fn apply(&self, data: ModelInputData) -> ModelInputData {
        let items = &data.input;
        let boundary = self.find_recent_boundary(items);
        if boundary == 0 {
            return data;
        }
        let names = build_call_id_to_names(items);
        let mut trimmed_count = 0;
        let mut saved = 0;
        let input = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if i >= boundary {
                    return item.clone();
                }
                let Value::Object(m) = &item.0 else {
                    return item.clone();
                };
                let ty = m.get("type").and_then(Value::as_str).unwrap_or_default();
                let call_id = m
                    .get("call_id")
                    .or_else(|| m.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let tool_names: Vec<String> = names.get(call_id).cloned().unwrap_or_else(|| {
                    if ty == "tool_search_output" {
                        vec!["tool_search".into()]
                    } else {
                        Vec::new()
                    }
                });
                if let Some(allowed) = &self.trimmable_tools
                    && !tool_names.iter().any(|n| allowed.contains(n))
                {
                    return item.clone();
                }
                let result = match ty {
                    "function_call_output" => self.trim_function_call_output(m, &tool_names),
                    "tool_search_output" => self.trim_tool_search_output(m),
                    _ => None,
                };
                match result {
                    Some((new_item, s)) => {
                        trimmed_count += 1;
                        saved += s;
                        InputItem(Value::Object(new_item))
                    }
                    None => item.clone(),
                }
            })
            .collect();
        if trimmed_count > 0 {
            log::debug!(
                "ToolOutputTrimmer: trimmed {trimmed_count} tool output(s), saved ~{saved} chars"
            );
        }
        ModelInputData {
            input,
            instructions: data.instructions,
        }
    }

    fn find_recent_boundary(&self, items: &[InputItem]) -> usize {
        let mut count = 0;
        for i in (0..items.len()).rev() {
            if items[i].0.get("role").and_then(Value::as_str) == Some("user") {
                count += 1;
                if count >= self.recent_turns {
                    return i;
                }
            }
        }
        0
    }

    fn trim_function_call_output(
        &self,
        item: &Map<String, Value>,
        tool_names: &[String],
    ) -> Option<(Map<String, Value>, usize)> {
        let output = item
            .get("output")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        if let Value::Array(parts) = &output {
            return self.trim_structured_output(item, parts, tool_names);
        }
        let output_str = match &output {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let output_len = chars(&output_str);
        if output_len <= self.max_output_chars {
            return None;
        }
        let display = tool_names
            .first()
            .filter(|n| !n.is_empty())
            .map(String::as_str)
            .unwrap_or("unknown_tool");
        let summary = format!(
            "[Trimmed: {display} output — {output_len} chars → {} char preview]\n{}...",
            self.preview_chars,
            prefix(&output_str, self.preview_chars)
        );
        if chars(&summary) >= output_len {
            return None;
        }
        let mut new_item = item.clone();
        let saved = output_len - chars(&summary);
        new_item.insert("output".into(), Value::String(summary));
        Some((new_item, saved))
    }

    fn trim_structured_output(
        &self,
        item: &Map<String, Value>,
        parts: &[Value],
        tool_names: &[String],
    ) -> Option<(Map<String, Value>, usize)> {
        let (output_len, text, dropped) = structured_output_details(parts)?;
        if output_len <= self.max_output_chars {
            return None;
        }
        let display = tool_names
            .first()
            .filter(|n| !n.is_empty())
            .map(String::as_str)
            .unwrap_or("unknown_tool");
        let dropped_note = if dropped.is_empty() {
            String::new()
        } else {
            format!(
                "; dropped {}",
                dropped
                    .iter()
                    .map(|(t, c)| format!("{c} {t}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let minimal = "[Trimmed]";
        let max = self.max_output_chars;
        let summary = if max < minimal.len() {
            minimal[..max].to_string()
        } else {
            let budget = max - minimal.len() - 1;
            let text_len = chars(&text);
            let preview_len = text_len.min(self.preview_chars).min(budget);
            let mut body = if preview_len > 0 {
                format!("\n{}", prefix(&text, preview_len))
            } else {
                String::new()
            };
            if preview_len < text_len && minimal.len() + chars(&body) + 3 <= max {
                body.push_str("...");
            }
            let preview_note = if text.is_empty() {
                String::new()
            } else {
                format!("; preview {preview_len}")
            };
            let mut headers = vec![format!(
                "[Trimmed: {display}; payload {output_len}{preview_note}{dropped_note}]"
            )];
            if !dropped.is_empty() {
                let types = dropped.keys().cloned().collect::<Vec<_>>().join(", ");
                let total: usize = dropped.values().sum();
                headers.extend([
                    format!("[Trimmed: {display}{dropped_note}]"),
                    format!("[Trimmed{dropped_note}]"),
                    format!("[Trimmed: {types}]"),
                    format!("[Trimmed: dropped {total} opaque]"),
                ]);
            }
            headers.extend([
                format!("[Trimmed: payload {output_len}]"),
                format!("[Trimmed: {display}]"),
                minimal.to_string(),
            ]);
            headers
                .into_iter()
                .find(|h| chars(h) + chars(&body) <= max)
                .map(|h| h + &body)
                .unwrap_or_else(|| minimal.to_string())
        };
        let mut new_item = item.clone();
        let saved = output_len.saturating_sub(chars(&summary));
        new_item.insert("output".into(), Value::String(summary));
        Some((new_item, saved))
    }

    fn trim_tool_search_output(
        &self,
        item: &Map<String, Value>,
    ) -> Option<(Map<String, Value>, usize)> {
        if let Some(results) = item.get("results").filter(|r| r.is_array()) {
            let serialized = py_json(results);
            let len = chars(&serialized);
            if len <= self.max_output_chars {
                return None;
            }
            let summary = format!(
                "[Trimmed: tool_search output — {len} chars → {} char preview]\n{}...",
                self.preview_chars,
                prefix(&serialized, self.preview_chars)
            );
            if chars(&summary) >= len {
                return None;
            }
            let saved = len - chars(&summary);
            let mut new_item = item.clone();
            new_item.insert(
                "results".into(),
                Value::Array(vec![serde_json::json!({"text": summary})]),
            );
            return Some((new_item, saved));
        }
        let tools = item.get("tools")?.as_array()?;
        let original = py_json(&Value::Array(tools.clone()));
        if chars(&original) <= self.max_output_chars {
            return None;
        }
        let trimmed: Vec<Value> = tools.iter().map(|t| self.trim_search_tool(t)).collect();
        let trimmed_s = py_json(&Value::Array(trimmed.clone()));
        if chars(&trimmed_s) >= chars(&original) {
            return None;
        }
        let mut new_item = item.clone();
        new_item.insert("tools".into(), Value::Array(trimmed));
        Some((new_item, chars(&original) - chars(&trimmed_s)))
    }

    fn trim_search_tool(&self, tool: &Value) -> Value {
        let Value::Object(m) = tool else {
            return tool.clone();
        };
        let mut t = m.clone();
        if let Some(Value::String(d)) = m.get("description") {
            let mut short = prefix(d, self.preview_chars).to_string();
            if chars(d) > self.preview_chars {
                short.push_str("...");
            }
            t.insert("description".into(), Value::String(short));
        }
        match m.get("type").and_then(Value::as_str) {
            Some("function") => {
                if let Some(Value::Object(p)) = m.get("parameters") {
                    t.insert("parameters".into(), Value::Object(trim_json_schema(p)));
                }
            }
            Some("namespace") => {
                if let Some(Value::Array(nested)) = m.get("tools") {
                    t.insert(
                        "tools".into(),
                        Value::Array(nested.iter().map(|n| self.trim_search_tool(n)).collect()),
                    );
                }
            }
            _ => {}
        }
        Value::Object(t)
    }
}

fn trim_schema_value(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(trim_json_schema(m)),
        other => other.clone(),
    }
}

fn trim_json_schema(schema: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in schema {
        let k = key.as_str();
        if PROSE_SCHEMA_KEYWORDS.contains(&k) {
            continue;
        }
        let new = if SCHEMA_VALUE_KEYWORDS.contains(&k) {
            match value {
                Value::Object(_) => trim_schema_value(value),
                Value::Array(a) if k == "items" => {
                    Value::Array(a.iter().map(trim_schema_value).collect())
                }
                other => other.clone(),
            }
        } else if SCHEMA_LIST_KEYWORDS.contains(&k) && value.is_array() {
            Value::Array(
                value
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(trim_schema_value)
                    .collect(),
            )
        } else if SCHEMA_MAP_KEYWORDS.contains(&k) && value.is_object() {
            Value::Object(
                value
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(n, s)| (n.clone(), trim_schema_value(s)))
                    .collect(),
            )
        } else {
            value.clone()
        };
        out.insert(key.clone(), new);
    }
    out
}

fn build_call_id_to_names(items: &[InputItem]) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();
    for item in items {
        let v = &item.0;
        match v.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                let Some(call_id) = v.get("call_id").and_then(Value::as_str) else {
                    continue;
                };
                let bare = v
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty());
                let ns = v
                    .get("namespace")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty());
                let qualified = match (bare, ns) {
                    // Deferred top-level tools come back with `namespace == name`.
                    (Some(b), Some(n)) if n != b => Some(format!("{n}.{b}")),
                    (Some(b), _) => Some(b.to_string()),
                    _ => None,
                };
                let mut names = Vec::new();
                if let Some(q) = &qualified {
                    names.push(q.clone());
                }
                if let Some(b) = bare
                    && Some(b) != qualified.as_deref()
                {
                    names.push(b.to_string());
                }
                if !names.is_empty() {
                    map.insert(call_id.to_string(), names);
                }
            }
            Some("tool_search_call") => {
                if let Some(id) = v
                    .get("call_id")
                    .or_else(|| v.get("id"))
                    .and_then(Value::as_str)
                {
                    map.insert(id.to_string(), vec!["tool_search".to_string()]);
                }
            }
            _ => {}
        }
    }
    map
}

/// Payload size, readable text and dropped part counts of canonical structured output parts.
fn structured_output_details(parts: &[Value]) -> Option<(usize, String, BTreeMap<String, usize>)> {
    if parts.is_empty() {
        return None;
    }
    let mut len = 0;
    let mut texts = Vec::new();
    let mut dropped: BTreeMap<String, usize> = BTreeMap::new();
    for part in parts {
        let m = part.as_object()?;
        let ty = m.get("type")?.as_str()?;
        let allowed: &[&str] = match ty {
            "input_text" => &["type", "text"],
            "input_image" => &["type", "image_url", "file_id", "detail"],
            "input_file" => &["type", "file_data", "file_url", "file_id", "filename"],
            _ => return None,
        };
        if !m.keys().all(|k| allowed.contains(&k.as_str())) {
            return None;
        }
        if m.iter().any(|(k, v)| k != "type" && !v.is_string()) {
            return None;
        }
        match ty {
            "input_text" => texts.push(m.get("text")?.as_str()?.to_string()),
            "input_image" => {
                if !m.contains_key("image_url") && !m.contains_key("file_id") {
                    return None;
                }
                if let Some(d) = m.get("detail").and_then(Value::as_str)
                    && !["low", "high", "auto"].contains(&d)
                {
                    return None;
                }
                *dropped.entry(ty.to_string()).or_default() += 1;
            }
            _ => {
                if !["file_data", "file_url", "file_id"]
                    .iter()
                    .any(|f| m.contains_key(*f))
                {
                    return None;
                }
                *dropped.entry(ty.to_string()).or_default() += 1;
            }
        }
        len += m
            .iter()
            .filter(|(k, _)| *k != "type")
            .filter_map(|(_, v)| v.as_str())
            .map(chars)
            .sum::<usize>();
    }
    Some((len, texts.join("\n"), dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn data(items: Vec<Value>) -> ModelInputData {
        ModelInputData {
            input: items.into_iter().map(InputItem).collect(),
            instructions: None,
        }
    }

    #[test]
    fn trims_old_large_outputs_only() {
        let big = "x".repeat(1000);
        let items = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": big}),
            json!({"role": "user", "content": "q2"}),
            json!({"type": "function_call", "call_id": "c2", "name": "search", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c2", "output": big}),
            json!({"role": "user", "content": "q3"}),
        ];
        let out = ToolOutputTrimmer::new()
            .with_preview_chars(10)
            .apply(data(items));
        let first = out.input[2].0["output"].as_str().unwrap();
        assert!(
            first.starts_with(
                "[Trimmed: search output — 1000 chars → 10 char preview]\nxxxxxxxxxx..."
            )
        );
        assert_eq!(out.input[5].0["output"].as_str().unwrap().len(), 1000);

        let skip = ToolOutputTrimmer::new()
            .with_trimmable_tools(["other"])
            .apply(data(vec![
                json!({"role": "user", "content": "q"}),
                json!({"type": "function_call", "call_id": "c", "name": "search", "arguments": "{}"}),
                json!({"type": "function_call_output", "call_id": "c", "output": "y".repeat(900)}),
                json!({"role": "user", "content": "a"}),
                json!({"role": "user", "content": "b"}),
            ]));
        assert_eq!(skip.input[2].0["output"].as_str().unwrap().len(), 900);
    }

    #[test]
    fn trims_structured_outputs_and_schemas() {
        let parts = json!([{"type": "input_text", "text": "t".repeat(600)},
                           {"type": "input_image", "image_url": "data:x"}]);
        let items = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": parts}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "user", "content": "q3"}),
        ];
        let out = ToolOutputTrimmer::new()
            .with_max_output_chars(100)
            .apply(data(items));
        let s = out.input[1].0["output"].as_str().unwrap();
        // The 90-char preview only fits the minimal header within 100 chars.
        assert!(s.starts_with("[Trimmed]\nttt"));
        assert_eq!(s.chars().count(), 100);
        let roomy = ToolOutputTrimmer::new()
            .with_max_output_chars(300)
            .with_preview_chars(20)
            .apply(data(vec![
                json!({"role": "user", "content": "q1"}),
                json!({"type": "function_call_output", "call_id": "c1", "output": json!([
                    {"type": "input_text", "text": "t".repeat(600)},
                    {"type": "input_image", "image_url": "data:x"}])}),
                json!({"role": "user", "content": "q2"}),
                json!({"role": "user", "content": "q3"}),
            ]));
        assert!(roomy.input[1].0["output"].as_str().unwrap().starts_with(
            "[Trimmed: unknown_tool; payload 606; preview 20; dropped 1 input_image]\n"
        ));

        let schema = json!({"type": "object", "description": "long", "properties": {
            "a": {"type": "string", "title": "A", "description": "d"}}});
        let trimmed = trim_json_schema(schema.as_object().unwrap());
        assert_eq!(
            Value::Object(trimmed),
            json!({"type": "object", "properties": {"a": {"type": "string"}}})
        );
    }
}
