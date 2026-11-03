//! The generated-item sidecars of the `RunState` envelope: nested-history ownership and the
//! generated/session item index map.
//!
//! Nested handoff history carries some generated items into the next agent's input verbatim, so
//! the full-history view must not replay them twice. Python records each such item as
//! `{index, digest, input_index}`: where the item lives in the session history, a SHA-256 digest
//! of its input form, and where its copy sits in `original_input`. Both the digest format and the
//! validation rules are Python's, so a sidecar written by either SDK is checked by the other.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::RunState;
use crate::error::{AgentsError, Result};

/// `json.dumps(value, sort_keys=True)` with Python's default separators and `ensure_ascii`.
fn python_dumps(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_string(key, out);
                out.push_str(": ");
                python_dumps(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_dumps(item, out);
            }
            out.push(']');
        }
        Value::String(s) => python_string(s, out),
        other => out.push_str(&other.to_string()),
    }
}

fn python_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e && (c as u32) != 0x7f => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Port of `digest_input_item`: a fixed-size digest of an input item for durable occurrence
/// tracking. Assistant messages are normalized first, so an item hashes the same whether its
/// content is a bare string or output-text parts, and a `completed` status is not part of it.
pub(crate) fn digest_input_item(item: &Value) -> String {
    let mut payload = item.clone();
    if let Value::Object(map) = &mut payload {
        for key in [
            "_agents_tool_description",
            "_agents_tool_title",
            "created_by",
        ] {
            map.remove(key);
        }
        if map.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(Value::String(text)) = map.get("content").cloned() {
                map.insert(
                    "content".into(),
                    json!([{"type": "output_text", "text": text}]),
                );
            }
            if map
                .get("status")
                .is_none_or(|s| s.is_null() || s.as_str() == Some("completed"))
            {
                map.remove("status");
            }
        }
    }
    let mut fingerprint = String::new();
    python_dumps(&payload, &mut fingerprint);
    Sha256::digest(fingerprint.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The input form of a generated item, as `run_item_to_input_item` builds it.
fn input_form(item: &crate::items::RunItem) -> Option<Value> {
    if matches!(item.kind, crate::items::RunItemKind::ToolApproval { .. }) {
        return None;
    }
    let mut value = item.to_input_item().0;
    if let Value::Object(map) = &mut value
        && map.get("status").is_some_and(Value::is_null)
    {
        map.remove("status");
    }
    Some(value)
}

/// `nested_history_owned_session_item_refs` for the wire: one `{index, digest, input_index}`
/// per owned generated item whose copy is found in the original input.
pub(super) fn ownership_to_wire(state: &RunState) -> Value {
    let input_digests: Vec<String> = state
        .original_input
        .iter()
        .map(|item| digest_input_item(&item.0))
        .collect();
    let mut used = vec![false; input_digests.len()];
    let mut refs = Vec::new();
    for &index in &state.nested_history_owned_items {
        let Some(item) = state.generated_items.get(index) else {
            continue;
        };
        let Some(input) = input_form(item) else {
            continue;
        };
        let digest = digest_input_item(&input);
        let Some(input_index) =
            (0..input_digests.len()).find(|i| !used[*i] && input_digests[*i] == digest)
        else {
            continue;
        };
        used[input_index] = true;
        refs.push(json!({"index": index, "digest": digest, "input_index": input_index}));
    }
    Value::Array(refs)
}

fn refused(message: &str) -> AgentsError {
    AgentsError::user(message)
}

/// A non-negative JSON integer (not a bool, float or string).
fn index_value(value: Option<&Value>) -> Option<usize> {
    value.and_then(Value::as_u64).map(|v| v as usize)
}

/// Validate `generated_session_item_indexes` and report whether it maps cleanly. Port of the
/// mapping check in `_deserialize_run_state`: a malformed map is ignored with a warning rather
/// than partially binding generated and session items.
///
/// Rust keeps one item list (the session items are the generated items), so the map never
/// changes what is restored; the check keeps the diagnostics, and the aliasing rule for
/// mismatched payloads, identical to Python's.
pub(super) fn check_generated_session_indexes(wire: &Value) {
    let Some(mapping) = wire.get("generated_session_item_indexes") else {
        return;
    };
    let (Some(generated), Some(session)) = (
        wire.get("generated_items").and_then(Value::as_array),
        wire.get("session_items").and_then(Value::as_array),
    ) else {
        warn_invalid_indexes();
        return;
    };
    let Some(mapping) = mapping.as_array().filter(|m| m.len() == generated.len()) else {
        warn_invalid_indexes();
        return;
    };
    let mut used = std::collections::BTreeSet::new();
    for entry in mapping {
        if entry.is_null() {
            continue;
        }
        match index_value(Some(entry)) {
            Some(i) if i < session.len() && used.insert(i) => {}
            _ => {
                warn_invalid_indexes();
                return;
            }
        }
    }
    for (generated_index, entry) in mapping.iter().enumerate() {
        if let Some(session_index) = index_value(Some(entry))
            && generated[generated_index] != session[session_index]
        {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "Ignoring mismatched generated/session occurrence in serialized RunState"
            );
        }
    }
}

fn warn_invalid_indexes() {
    log::warn!(
        target: crate::logger::LOG_TARGET,
        "Ignoring invalid generated_session_item_indexes in serialized RunState"
    );
}

/// Read and validate `nested_history_owned_session_item_refs`, returning the owned indexes into
/// the restored generated items. Port of the ownership pass of `_deserialize_run_state`: every
/// malformed or mismatched reference is an error, never a silent skip, because a wrong
/// reference would drop a real item from the replayed history.
pub(super) fn ownership_from_wire(
    wire: &Value,
    state: &RunState,
    keep: &[bool],
) -> Result<Vec<usize>> {
    let refs = match wire.get("nested_history_owned_session_item_refs") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(refs)) => refs,
        Some(_) => {
            return Err(refused(
                "Run state nested_history_owned_session_item_refs must be a list of objects",
            ));
        }
    };
    let session_len = wire
        .get("session_items")
        .and_then(Value::as_array)
        .map(Vec::len);
    // Which restored generated item stands for each session index.
    let generated_for_session = |session_index: usize| -> Option<usize> {
        match wire
            .get("generated_session_item_indexes")
            .and_then(Value::as_array)
        {
            Some(mapping) if mapping.len() == state.generated_items.len() => mapping
                .iter()
                .position(|v| index_value(Some(v)) == Some(session_index)),
            _ => (session_index < state.generated_items.len()).then_some(session_index),
        }
    };
    let mut owned = Vec::new();
    for entry in refs {
        let malformed = || {
            refused(
                "Run state nested_history_owned_session_item_refs entries must contain a \
                 non-negative integer index and input_index, and 64-character digest",
            )
        };
        let Value::Object(entry) = entry else {
            return Err(malformed());
        };
        let (Some(session_index), Some(input_index), Some(digest)) = (
            index_value(entry.get("index")),
            index_value(entry.get("input_index")),
            entry
                .get("digest")
                .and_then(Value::as_str)
                .filter(|d| d.chars().count() == 64),
        ) else {
            return Err(malformed());
        };
        if session_len.is_some_and(|len| session_index >= len) {
            return Err(refused(
                "Run state nested history ownership references a missing session item",
            ));
        }
        // A session item the reader skips (no type or no agent) owns nothing.
        let session_item_readable = wire
            .get("session_items")
            .and_then(Value::as_array)
            .and_then(|items| items.get(session_index))
            .is_none_or(|item| {
                item.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.is_empty())
                    && ["agent", "source_agent", "target_agent"].iter().any(|key| {
                        match item.get(*key) {
                            Some(Value::String(name)) => !name.is_empty(),
                            Some(Value::Object(agent)) => agent
                                .get("name")
                                .and_then(Value::as_str)
                                .is_some_and(|name| !name.is_empty()),
                            _ => false,
                        }
                    })
            });
        let Some(generated_index) = generated_for_session(session_index)
            .filter(|_| session_item_readable)
            .filter(|i| keep.get(*i).copied().unwrap_or(true))
        else {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "Ignoring nested history ownership for skipped session item at index {session_index}"
            );
            continue;
        };
        let Some(input_item) = state.original_input.get(input_index) else {
            return Err(refused(
                "Run state nested history ownership references a missing input item",
            ));
        };
        let item = &state.generated_items[generated_index];
        // The restored item re-serializes through typed structs, so its digest can differ from
        // the one Python computed from the live item; the serialized payload is what Python
        // hashed, so either one vouches for the reference.
        let serialized = wire
            .get("session_items")
            .and_then(Value::as_array)
            .and_then(|items| items.get(session_index))
            .and_then(|item| item.get("raw_item"))
            .map(|raw| {
                let mut raw = raw.clone();
                if let Value::Object(map) = &mut raw
                    && map.get("status").is_some_and(Value::is_null)
                {
                    map.remove("status");
                }
                digest_input_item(&raw)
            });
        let session_matches = input_form(item)
            .is_some_and(|input| digest_input_item(&input) == digest)
            || serialized.as_deref() == Some(digest);
        if !session_matches {
            return Err(refused(
                "Run state nested history ownership session digest does not match",
            ));
        }
        if digest_input_item(&input_item.0) != digest {
            return Err(refused(
                "Run state nested history ownership input digest does not match",
            ));
        }
        owned.push(generated_index);
    }
    Ok(owned)
}
