//! Canonical tool-invocation identities. Port of `agents._tool_invocation`.
//!
//! Every tool call a model emits gets a stable `(invocation_type, call_id, approval_scope,
//! fingerprint)` identity. The approval scope says *what the call is authorized as* (the tool
//! it routes to); the fingerprint adds the semantic payload, so a replay that changes the
//! arguments under a call id already seen is rejected instead of silently re-approved.
//!
//! The digests are byte-compatible with the Python SDK: both hash the same canonical JSON
//! (sorted keys, no spaces, non-ASCII left as-is), so a ledger written by either SDK is
//! readable by the other.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

/// The item types that count as a tool invocation. Mirrors `_TOOL_INVOCATION_TYPES`.
const TOOL_INVOCATION_TYPES: [&str; 7] = [
    "apply_patch_call",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
    "mcp_approval_request",
    "shell_call",
];

/// Output item type -> the invocation type it completes. Mirrors `_TOOL_OUTPUT_TYPES`.
const TOOL_OUTPUT_TYPES: [(&str, &str); 7] = [
    ("apply_patch_call_output", "apply_patch_call"),
    ("computer_call_output", "computer_call"),
    ("custom_tool_call_output", "custom_tool_call"),
    ("function_call_output", "function_call"),
    ("local_shell_call_output", "local_shell_call"),
    ("mcp_approval_response", "mcp_approval_request"),
    ("shell_call_output", "shell_call"),
];

/// Fields that make up the semantic payload, in Python's order (the digest sorts keys, so the
/// order only matters for readability). Mirrors `_SEMANTIC_FIELDS`.
const SEMANTIC_FIELDS: [&str; 13] = [
    "type",
    "name",
    "namespace",
    "server_label",
    "arguments",
    "input",
    "action",
    "actions",
    "pending_safety_checks",
    "operation",
    "operations",
    "environment",
    "caller",
];

/// The `invocation_role` a handoff call is scoped under. A handoff and a same-named function
/// tool must not share an approval scope.
pub(crate) const HANDOFF_ROLE: &str = "handoff";

/// The identity of one tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolInvocationIdentity {
    pub invocation_type: String,
    pub call_id: String,
    pub approval_scope: String,
    pub fingerprint: String,
}

/// Write a value as Python's `json.dumps(sort_keys=True, separators=(",", ":"),
/// ensure_ascii=False)` would.
///
/// Object keys are sorted explicitly rather than relying on `serde_json::Map` being a
/// `BTreeMap`: a dependency that turns on `serde_json/preserve_order` (the `mongodb` feature
/// does, through `bson`) would otherwise silently change every digest and break agreement
/// with the Python SDK. Sorting `&str` compares UTF-8 bytes, which is the same order as
/// Python's code-point sort.
fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&serde_json::to_string(scalar).unwrap_or_default()),
    }
}

/// Canonical JSON digest. Port of `_fingerprint`.
fn digest(payload: &Value) -> String {
    let mut encoded = String::new();
    write_canonical(payload, &mut encoded);
    let out = Sha256::digest(encoded.as_bytes());
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// Port of `_normalize_value`: recurse into objects and arrays, optionally dropping null
/// members of an object. Scalars pass through.
fn normalize_value(value: &Value, exclude_none: bool) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(_, v)| !(exclude_none && v.is_null()))
                .map(|(k, v)| (k.clone(), normalize_value(v, exclude_none)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| normalize_value(v, exclude_none))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Port of `_normalize_arguments`: a JSON string is parsed first, so re-encoding the same
/// arguments with different whitespace or key order keeps the same fingerprint.
fn normalize_arguments(value: &Value) -> Value {
    match value.as_str() {
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(parsed) => normalize_value(&parsed, false),
            Err(_) => value.clone(),
        },
        None => normalize_value(value, false),
    }
}

/// Port of `_unwrap_hosted_mcp_approval`: a hosted tool call that wraps an MCP approval
/// request is flattened into the request itself.
fn unwrap_hosted_mcp_approval(raw: &Value) -> Option<Map<String, Value>> {
    let map = raw.as_object()?;
    let provider = map.get("provider_data").and_then(Value::as_object);
    match (map.get("type").and_then(Value::as_str), provider) {
        (Some("hosted_tool_call"), Some(provider))
            if provider.get("type").and_then(Value::as_str) == Some("mcp_approval_request") =>
        {
            let mut merged = map.clone();
            for (k, v) in provider {
                merged.insert(k.clone(), v.clone());
            }
            Some(merged)
        }
        _ => Some(map.clone()),
    }
}

/// The lookup key Python matches a function tool on. Port of
/// `get_function_tool_lookup_key` + `serialize_function_tool_lookup_key`, normalized to the
/// list form `_normalize_value` produces for a tuple.
fn lookup_key_value(name: &str, namespace: Option<&str>) -> Option<Value> {
    if name.is_empty() {
        return None;
    }
    Some(match namespace.filter(|n| !n.is_empty()) {
        Some(ns) => json!(["namespaced", ns, name]),
        None => json!(["bare", name]),
    })
}

/// The authorization scope of an invocation. Port of `tool_invocation_approval_scope`.
fn approval_scope(
    raw: &Map<String, Value>,
    invocation_role: Option<&str>,
) -> Option<(String, String)> {
    let invocation_type = raw.get("type").and_then(Value::as_str)?;
    if !TOOL_INVOCATION_TYPES.contains(&invocation_type) {
        return None;
    }
    let mut payload = Map::new();
    payload.insert("type".into(), Value::String(invocation_type.to_string()));
    if let Some(role) = invocation_role {
        payload.insert("invocation_role".into(), Value::String(role.to_string()));
    }
    match invocation_type {
        "function_call" => {
            let name = raw.get("name").and_then(Value::as_str).unwrap_or_default();
            let namespace = raw.get("namespace").and_then(Value::as_str);
            payload.insert("tool_lookup_key".into(), lookup_key_value(name, namespace)?);
        }
        "mcp_approval_request" => {
            let name = raw
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())?;
            let label = raw
                .get("server_label")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())?;
            payload.insert("name".into(), Value::String(name.to_string()));
            payload.insert("server_label".into(), Value::String(label.to_string()));
        }
        _ => {
            if let Some(name) = raw
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
            {
                payload.insert("name".into(), Value::String(name.to_string()));
            }
        }
    }
    Some((invocation_type.to_string(), digest(&Value::Object(payload))))
}

/// The full identity of a tool invocation, or `None` when the item is not one (or is missing
/// the fields its type needs). Port of `tool_invocation_identity_and_scope`.
pub(crate) fn tool_invocation_identity(
    raw_item: &Value,
    invocation_role: Option<&str>,
) -> Option<ToolInvocationIdentity> {
    let raw = unwrap_hosted_mcp_approval(raw_item)?;
    let invocation_type = raw.get("type").and_then(Value::as_str)?.to_string();
    if !TOOL_INVOCATION_TYPES.contains(&invocation_type.as_str()) {
        return None;
    }
    let id_key = if invocation_type == "mcp_approval_request" {
        "id"
    } else {
        "call_id"
    };
    let call_id = raw
        .get(id_key)
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())?
        .to_string();
    let (scope_type, scope) = approval_scope(&raw, invocation_role)?;
    if scope_type != invocation_type {
        return None;
    }

    // Each type needs its own payload field, or the item is not a complete invocation.
    let present = |key: &str| raw.contains_key(key);
    let complete = match invocation_type.as_str() {
        "function_call" | "mcp_approval_request" => present("arguments"),
        "custom_tool_call" => {
            raw.get("name")
                .and_then(Value::as_str)
                .is_some_and(|n| !n.is_empty())
                && present("input")
        }
        "computer_call" | "local_shell_call" | "shell_call" => present("action"),
        "apply_patch_call" => present("operation") || present("operations"),
        _ => false,
    };
    if !complete {
        return None;
    }

    let mut semantic = Map::new();
    semantic.insert("approval_scope".into(), Value::String(scope.clone()));
    for field in SEMANTIC_FIELDS {
        // A function call's routing already lives in the approval scope.
        if invocation_type == "function_call" && matches!(field, "name" | "namespace") {
            continue;
        }
        let Some(value) = raw.get(field) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        semantic.insert(
            field.to_string(),
            if field == "arguments" {
                normalize_arguments(value)
            } else {
                normalize_value(value, true)
            },
        );
    }

    Some(ToolInvocationIdentity {
        invocation_type,
        call_id,
        approval_scope: scope,
        fingerprint: digest(&Value::Object(semantic)),
    })
}

/// The invocation type and call id an output item completes. Port of `tool_output_identity`.
pub(crate) fn tool_output_identity(raw_item: &Value) -> Option<(String, String)> {
    let map = raw_item.as_object()?;
    let output_type = map.get("type").and_then(Value::as_str)?;
    let (_, invocation_type) = TOOL_OUTPUT_TYPES.iter().find(|(o, _)| *o == output_type)?;
    let id_key = if output_type == "mcp_approval_response" {
        "approval_request_id"
    } else {
        "call_id"
    };
    let call_id = map
        .get(id_key)
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())?;
    Some((invocation_type.to_string(), call_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal digests in `tests/fixtures/python_run_state/interrupted_approval.json`,
    /// written by the Python SDK. If this ever drifts the ledger stops being interoperable.
    #[test]
    fn reproduces_the_python_digests_of_the_committed_fixture() {
        let raw = json!({
            "type": "function_call",
            "id": "fc_delete_1",
            "call_id": "call_delete_1",
            "name": "delete_file",
            "arguments": "{\"path\":\"/tmp/report.txt\"}",
        });
        let identity = tool_invocation_identity(&raw, None).unwrap();
        assert_eq!(identity.invocation_type, "function_call");
        assert_eq!(identity.call_id, "call_delete_1");
        assert_eq!(
            identity.approval_scope,
            "1f7953a6d493b5ba202ce0159596ce6f6324ae8f0af83dacc42ea570f1623f18"
        );
        assert_eq!(
            identity.fingerprint,
            "140b81832c339ddb86b483a877e4daacdfbbc673c9ab06cbfa04df1e4e2e9e52"
        );
    }

    #[test]
    fn reproduces_the_python_digests_of_the_handoff_fixture() {
        let raw = json!({
            "type": "function_call",
            "id": "fc_handoff_1",
            "call_id": "call_handoff_1",
            "name": "transfer_to_spanish_agent",
            "arguments": "{}",
        });
        // A handoff call is scoped as a handoff, not as a plain function tool.
        let identity = tool_invocation_identity(&raw, Some(HANDOFF_ROLE)).unwrap();
        assert_eq!(
            identity.approval_scope,
            "606c0d3b1473e5d615eda3cc5aff5685701ed3158b659ea6dceb6e93ac021c26"
        );
        assert_eq!(
            identity.fingerprint,
            "ffe0cadf06ea57198b7ecef01854f9945e1bc271c063cd553cfd7ef15e0fc40b"
        );
    }

    #[test]
    fn the_scope_ignores_the_arguments_but_the_fingerprint_does_not() {
        let call = |args: &str| {
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": args,
            })
        };
        let a = tool_invocation_identity(&call(r#"{"city":"Oslo"}"#), None).unwrap();
        // Different whitespace and key order, same arguments: same fingerprint.
        let b = tool_invocation_identity(&call("{ \"city\" : \"Oslo\" }"), None).unwrap();
        let c = tool_invocation_identity(&call(r#"{"city":"Lisbon"}"#), None).unwrap();
        assert_eq!(a.fingerprint, b.fingerprint);
        assert_eq!(a.approval_scope, c.approval_scope);
        assert_ne!(a.fingerprint, c.fingerprint);
    }

    #[test]
    fn a_namespace_changes_only_the_scope_and_the_scope_changes_the_fingerprint() {
        let bare = tool_invocation_identity(
            &json!({"type":"function_call","call_id":"c","name":"t","arguments":"{}"}),
            None,
        )
        .unwrap();
        let namespaced = tool_invocation_identity(
            &json!({"type":"function_call","call_id":"c","name":"t","namespace":"ns","arguments":"{}"}),
            None,
        )
        .unwrap();
        assert_ne!(bare.approval_scope, namespaced.approval_scope);
        assert_ne!(bare.fingerprint, namespaced.fingerprint);
    }

    #[test]
    fn the_digest_does_not_depend_on_map_insertion_order() {
        // Guards against a dependency enabling `serde_json/preserve_order`.
        let a: Value = serde_json::from_str(r#"{"b":1,"a":{"d":2,"c":3}}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":{"c":3,"d":2},"b":1}"#).unwrap();
        let mut sa = String::new();
        let mut sb = String::new();
        write_canonical(&a, &mut sa);
        write_canonical(&b, &mut sb);
        assert_eq!(sa, r#"{"a":{"c":3,"d":2},"b":1}"#);
        assert_eq!(sa, sb);
        assert_eq!(digest(&a), digest(&b));
    }

    #[test]
    fn a_non_invocation_item_has_no_identity() {
        assert!(tool_invocation_identity(&json!({"type": "message"}), None).is_none());
        // A function call without arguments is not a complete invocation.
        assert!(
            tool_invocation_identity(
                &json!({"type":"function_call","call_id":"c","name":"t"}),
                None
            )
            .is_none()
        );
    }

    #[test]
    fn output_items_name_the_call_they_complete() {
        assert_eq!(
            tool_output_identity(
                &json!({"type":"function_call_output","call_id":"c","output":"x"})
            ),
            Some(("function_call".into(), "c".into()))
        );
        assert_eq!(
            tool_output_identity(
                &json!({"type":"mcp_approval_response","approval_request_id":"r","approve":true})
            ),
            Some(("mcp_approval_request".into(), "r".into()))
        );
        assert_eq!(tool_output_identity(&json!({"type":"message"})), None);
    }
}
