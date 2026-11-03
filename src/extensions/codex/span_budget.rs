//! Fitting Codex span payloads into a character budget. Port of the `_enforce_span_data_budget`
//! family of `agents.extensions.experimental.codex.codex_tool`.
//!
//! `span_data_max_chars` is a budget for the **whole** span payload measured as compact
//! `ensure_ascii` JSON, not a per-value limit. When the payload does not fit, the long keys listed
//! in [`SPAN_TRIM_KEYS`] share what is left after the other keys, proportionally to their length.

use std::cmp::Reverse;

use serde_json::{Map, Value, json};

use crate::extensions::codex::py_json;

/// Keys trimmed to fit the span budget, highest priority first. Port of `SPAN_TRIM_KEYS`.
pub(crate) const SPAN_TRIM_KEYS: [&str; 8] = [
    "arguments",
    "command",
    "output",
    "result",
    "error",
    "text",
    "changes",
    "items",
];

/// Truncate a span string to `max_chars` characters with Python's suffix.
/// Port of `_truncate_span_string`.
pub(crate) fn truncate_span_string(value: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return value.to_string();
    };
    if max_chars == 0 {
        return String::new();
    }
    let total = value.chars().count();
    if total <= max_chars {
        return value.to_string();
    }
    let suffix = format!("... [truncated, {total} chars]");
    let suffix_len = suffix.chars().count();
    if max_chars <= suffix_len {
        return value.chars().take(max_chars).collect();
    }
    let mut truncated: String = value.chars().take(max_chars - suffix_len).collect();
    truncated.push_str(&suffix);
    truncated
}

/// Render a span value as the text the budget allocator trims. Port of `_stringify_span_value`.
fn stringify_span_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => py_json::dumps_compact(other),
    }
}

/// Truncate one span value before it is merged into the payload. Port of `_truncate_span_value`.
pub(crate) fn truncate_span_value(value: &Value, max_chars: Option<usize>) -> Value {
    let Some(max_chars) = max_chars else {
        return value.clone();
    };
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        Value::String(text) => Value::String(truncate_span_string(text, Some(max_chars))),
        other => {
            let encoded = py_json::dumps_compact(other);
            if encoded.len() <= max_chars {
                return other.clone();
            }
            json!({
                "preview": truncate_span_string(&encoded, Some(max_chars)),
                "truncated": true,
                "original_length": encoded.len(),
            })
        }
    }
}

/// Port of `_drop_empty_string_fields`: drop keys whose value is the empty string.
fn drop_empty_string_fields(data: &Map<String, Value>) -> Map<String, Value> {
    data.iter()
        .filter(|(_, value)| !matches!(value, Value::String(text) if text.is_empty()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn object_size(data: &Map<String, Value>) -> usize {
    py_json::char_size(&Value::Object(data.clone()))
}

/// Trim `data` so its compact JSON fits `max_chars`. Port of `_enforce_span_data_budget`.
pub(crate) fn enforce_span_data_budget(
    data: &Map<String, Value>,
    max_chars: Option<usize>,
) -> Map<String, Value> {
    let Some(max_chars) = max_chars else {
        return drop_empty_string_fields(data);
    };
    if max_chars == 0 {
        return Map::new();
    }

    let mut trimmed = drop_empty_string_fields(data);
    if object_size(&trimmed) <= max_chars {
        return trimmed;
    }

    let mut kept_keys: Vec<&'static str> = SPAN_TRIM_KEYS
        .iter()
        .copied()
        .filter(|key| trimmed.contains_key(*key))
        .collect();
    if kept_keys.is_empty() {
        return trimmed;
    }

    let mut base = trimmed.clone();
    for key in &kept_keys {
        base.insert((*key).to_string(), Value::String(String::new()));
    }
    let mut base_size = object_size(&base);

    // Drop the lowest-priority keys entirely when even an empty payload does not fit.
    while base_size > max_chars && !kept_keys.is_empty() {
        let drop_key = kept_keys.pop().expect("non-empty");
        base.remove(drop_key);
        trimmed.remove(drop_key);
        base_size = object_size(&base);
    }
    if base_size > max_chars {
        return drop_empty_string_fields(&base);
    }

    // The trimmable values, in priority order.
    let mut values: Vec<(&'static str, String)> = Vec::new();
    for key in &kept_keys {
        match trimmed.get(*key) {
            None | Some(Value::Null) => continue,
            Some(Value::String(text)) if text.is_empty() => continue,
            Some(value) => values.push((key, stringify_span_value(value))),
        }
    }
    values.retain(|(key, text)| {
        if text.is_empty() {
            trimmed.insert((*key).to_string(), Value::String(String::new()));
            false
        } else {
            true
        }
    });
    kept_keys
        .retain(|key| values.iter().any(|(name, _)| name == key) || trimmed.contains_key(*key));
    if kept_keys.is_empty() {
        return drop_empty_string_fields(&base);
    }

    base_size = object_size(&base);
    if base_size >= max_chars {
        return drop_empty_string_fields(&base);
    }
    let available = max_chars - base_size;

    let lengths: Vec<usize> = values
        .iter()
        .map(|(_, text)| text.chars().count())
        .collect();
    let mut budgets: Vec<usize> = vec![0; values.len()];
    let mut remaining = if available >= values.len() {
        budgets.iter_mut().for_each(|budget| *budget = 1);
        available - values.len()
    } else {
        // `values` is already in `SPAN_TRIM_KEYS` order, which is Python's `ordered_keys`.
        budgets.iter_mut().take(available).for_each(|b| *b = 1);
        0
    };

    // Keep arguments intact when they already fit within the budget.
    if let Some(index) = values.iter().position(|(key, _)| *key == "arguments")
        && remaining > 0
    {
        let needed = lengths[index].saturating_sub(budgets[index]);
        if needed > 0 {
            let grant = needed.min(remaining);
            budgets[index] += grant;
            remaining -= grant;
        }
    }

    if remaining > 0 {
        let weights: Vec<usize> = lengths
            .iter()
            .zip(&budgets)
            .map(|(length, budget)| length.saturating_sub(*budget))
            .collect();
        let weight_total: usize = weights.iter().sum();
        if weight_total > 0 {
            for (index, weight) in weights.iter().enumerate() {
                if *weight == 0 {
                    continue;
                }
                let share = remaining as f64 * (*weight as f64 / weight_total as f64);
                budgets[index] += share as usize;
            }
        }
        for (index, budget) in budgets.iter_mut().enumerate() {
            *budget = (*budget).min(lengths[index]);
        }
        let allocated: usize = budgets.iter().sum();
        let mut leftover = available.saturating_sub(allocated);
        if leftover > 0 {
            // `sorted(..., reverse=True)` is stable, so ties keep `SPAN_TRIM_KEYS` order.
            let mut ordered: Vec<usize> = (0..values.len()).collect();
            ordered.sort_by_key(|index| Reverse(weights[*index]));
            let mut idx = 0usize;
            while leftover > 0 {
                let expandable: Vec<usize> = ordered
                    .iter()
                    .copied()
                    .filter(|index| budgets[*index] < lengths[*index])
                    .collect();
                if expandable.is_empty() {
                    break;
                }
                budgets[expandable[idx % expandable.len()]] += 1;
                leftover -= 1;
                idx += 1;
            }
        }
    }

    for key in &kept_keys {
        let replacement = match values.iter().position(|(name, _)| name == key) {
            Some(index) => truncate_span_string(&values[index].1, Some(budgets[index])),
            None => String::new(),
        };
        trimmed.insert((*key).to_string(), Value::String(replacement));
    }

    // Give back one character at a time until the payload fits.
    while object_size(&trimmed) > max_chars && !kept_keys.is_empty() {
        // Python's `max` returns the first maximum; `max_by_key` returns the last, so scan
        // the keys in reverse.
        let key = *kept_keys
            .iter()
            .rev()
            .max_by_key(|key| current_len(&trimmed, key))
            .expect("non-empty");
        let current = current_len(&trimmed, key);
        if current > 0 {
            let source = values
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, text)| text.as_str())
                .unwrap_or("");
            trimmed.insert(
                key.to_string(),
                Value::String(truncate_span_string(source, Some(current - 1))),
            );
        } else {
            kept_keys.retain(|candidate| *candidate != key);
        }
    }

    if object_size(&trimmed) <= max_chars {
        drop_empty_string_fields(&trimmed)
    } else {
        drop_empty_string_fields(&base)
    }
}

/// The character length of the value currently stored under `key`.
fn current_len(trimmed: &Map<String, Value>, key: &str) -> usize {
    match trimmed.get(key) {
        Some(Value::String(text)) => text.chars().count(),
        Some(other) => other.to_string().chars().count(),
        None => 0,
    }
}

/// Merge `updates` over `current` and re-fit the result. Port of `_merge_span_data`.
pub(crate) fn merge_span_data(
    current: &Map<String, Value>,
    updates: &Map<String, Value>,
    max_chars: Option<usize>,
) -> Map<String, Value> {
    let mut merged = current.clone();
    for (key, value) in updates {
        merged.insert(key.clone(), value.clone());
    }
    enforce_span_data_budget(&merged, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(value: Value) -> Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn truncates_span_strings() {
        assert_eq!(truncate_span_string("hello", None), "hello");
        assert_eq!(truncate_span_string("hello", Some(0)), "");
        assert_eq!(truncate_span_string("hello", Some(5)), "hello");
        assert_eq!(truncate_span_string("hello", Some(3)), "hel");
        let value = "x".repeat(100);
        let truncated = truncate_span_string(&value, Some(40));
        assert_eq!(truncated.chars().count(), 40);
        assert!(truncated.ends_with("... [truncated, 100 chars]"));
    }

    #[test]
    fn truncates_non_string_values_into_a_preview() {
        let value = json!({"items": vec!["x".repeat(50); 4]});
        assert_eq!(truncate_span_value(&value, None), value);
        assert_eq!(truncate_span_value(&json!(7), Some(1)), json!(7));
        assert_eq!(
            truncate_span_value(&value, Some(40)),
            json!({
                "preview": "{\"items\":[\"xxx... [truncated, 223 chars]",
                "truncated": true,
                "original_length": 223,
            })
        );
    }

    /// Every expectation below is the output of the Python allocator for the same input.
    #[test]
    fn keeps_payloads_that_already_fit() {
        let data = map(json!({"status": "completed", "exit_code": 0, "command": "pytest"}));
        assert_eq!(enforce_span_data_budget(&data, Some(8192)), data);
    }

    #[test]
    fn drops_empty_strings_and_keeps_nulls() {
        let data = map(json!({"status": "", "exit_code": null, "command": "ls"}));
        assert_eq!(
            enforce_span_data_budget(&data, None),
            map(json!({"exit_code": null, "command": "ls"}))
        );
    }

    #[test]
    fn a_zero_budget_empties_the_payload() {
        let data = map(json!({"status": "completed"}));
        assert!(enforce_span_data_budget(&data, Some(0)).is_empty());
    }

    #[test]
    fn shares_the_budget_between_competing_keys() {
        let data = map(json!({
            "status": "completed",
            "exit_code": 0,
            "command": "c".repeat(300),
            "output": "o".repeat(900),
        }));
        let trimmed = enforce_span_data_budget(&data, Some(200));
        assert_eq!(
            trimmed,
            map(json!({
                "status": "completed",
                "exit_code": 0,
                "command": format!("{}... [truncated, 300 chars]", "c".repeat(9)),
                "output": format!("{}... [truncated, 900 chars]", "o".repeat(78)),
            }))
        );
        assert_eq!(py_json::char_size(&Value::Object(trimmed)), 200);
    }

    #[test]
    fn keeps_arguments_intact_when_they_fit() {
        let data = map(json!({
            "status": "in_progress",
            "arguments": "a".repeat(60),
            "command": "c".repeat(400),
            "output": "o".repeat(400),
        }));
        let trimmed = enforce_span_data_budget(&data, Some(220));
        assert_eq!(
            trimmed,
            map(json!({
                "status": "in_progress",
                "arguments": "a".repeat(60),
                "command": format!("{}... [truncated, 400 chars]", "c".repeat(22)),
                "output": format!("{}... [truncated, 400 chars]", "o".repeat(22)),
            }))
        );
        assert_eq!(py_json::char_size(&Value::Object(trimmed)), 220);
    }

    #[test]
    fn counts_non_ascii_characters_as_escaped() {
        let data = map(json!({
            "status": "completed",
            "exit_code": 0,
            "output": "é".repeat(120),
        }));
        let trimmed = enforce_span_data_budget(&data, Some(160));
        assert_eq!(
            trimmed,
            map(json!({
                "status": "completed",
                "exit_code": 0,
                "output": format!("{}... [truncated, 120 chars]", "é".repeat(14)),
            }))
        );
        assert_eq!(py_json::char_size(&Value::Object(trimmed)), 158);
    }

    #[test]
    fn drops_the_lowest_priority_keys_when_the_base_does_not_fit() {
        let data = map(json!({
            "status": "s".repeat(180),
            "command": "c".repeat(50),
            "output": "o".repeat(50),
            "items": "i".repeat(50),
        }));
        assert_eq!(
            enforce_span_data_budget(&data, Some(200)),
            map(json!({"status": "s".repeat(180)}))
        );
    }

    #[test]
    fn gives_one_character_each_when_the_budget_is_below_the_key_count() {
        let data = map(json!({
            "status": "completed",
            "command": "c".repeat(40),
            "output": "o".repeat(40),
            "text": "t".repeat(40),
        }));
        let trimmed = enforce_span_data_budget(&data, Some(56));
        assert_eq!(
            trimmed,
            map(json!({"status": "completed", "command": "ccccc", "output": "oooo"}))
        );
        assert_eq!(py_json::char_size(&Value::Object(trimmed)), 56);
    }

    #[test]
    fn stringifies_nested_values_before_trimming() {
        let data = map(json!({
            "status": "completed",
            "changes": vec![json!({"path": "a.py", "kind": "edit"}); 6],
        }));
        let trimmed = enforce_span_data_budget(&data, Some(120));
        assert_eq!(
            trimmed,
            map(json!({
                "status": "completed",
                "changes":
                    "[{\"path\":\"a.py\",\"kind\":\"edit\"},{\"path\":\"a.py\",\
                     ... [truncated, 181 chars]",
            }))
        );
        assert_eq!(py_json::char_size(&Value::Object(trimmed)), 119);
    }

    #[test]
    fn merging_re_fits_the_payload() {
        let current = map(json!({"status": "in_progress", "command": "pytest"}));
        let updates = map(json!({"status": "completed", "output": "o".repeat(400)}));
        let merged = merge_span_data(&current, &updates, Some(120));
        assert_eq!(
            merged,
            map(json!({
                "status": "completed",
                // The whole payload must fit, so even a short key is trimmed.
                "command": "p",
                "output": format!("{}... [truncated, 400 chars]", "o".repeat(46)),
            }))
        );
        assert_eq!(py_json::char_size(&Value::Object(merged)), 120);
    }
}
