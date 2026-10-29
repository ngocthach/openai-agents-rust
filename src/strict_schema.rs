//! Conversion of JSON schemas to the strict subset accepted by OpenAI structured outputs.
//! Port of `agents.strict_schema`.

use serde_json::{Map, Value};

use crate::error::{AgentsError, Result};

const MAX_SCHEMA_NODES: usize = 100_000;
const MAX_SCHEMA_DEPTH: usize = 100;

const ADDITIONAL_PROPERTIES_ERROR: &str = "additionalProperties should not be set for object types. \
If you really need this, update the function or output tool to not use a strict schema.";

const REF_NON_CONSTRAINING_SIBLINGS: &[&str] = &[
    "description",
    "title",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
];

const ROOT_REF_NON_CONSTRAINING_SIBLINGS: &[&str] = &[
    "description",
    "title",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
    "$defs",
    "definitions",
    "$schema",
    "$id",
];

/// String formats accepted by OpenAI structured outputs.
const SUPPORTED_STRING_FORMATS: &[&str] = &[
    "date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid",
];

/// Remove `format` values that OpenAI strict mode rejects. schemars emits formats such as
/// `uint32`, `int64` and `double` for Rust numeric types; Pydantic never does.
fn strip_unsupported_format(obj: &mut Map<String, Value>) {
    let Some(Value::String(fmt)) = obj.get("format") else {
        return;
    };
    let is_string = match obj.get("type") {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        _ => false,
    };
    if !(is_string && SUPPORTED_STRING_FORMATS.contains(&fmt.as_str())) {
        obj.remove("format");
    }
}

/// Convert `schema` to the strict JSON schema format expected by the OpenAI API.
///
/// Objects get `additionalProperties: false` and every property becomes required, `oneOf` turns
/// into `anyOf`, single-entry `allOf` is merged, `$ref`s with siblings are inlined, and
/// `default: null` is dropped.
pub fn ensure_strict_json_schema(schema: &Value) -> Result<Value> {
    validate_depth(schema, 1)?;
    let Value::Object(obj) = schema else {
        return Err(AgentsError::user("Expected JSON schema to be an object"));
    };
    if obj.is_empty() {
        return Ok(serde_json::json!({
            "additionalProperties": false,
            "type": "object",
            "properties": {},
            "required": []
        }));
    }
    let root = schema.clone();
    let mut budget = MAX_SCHEMA_NODES;
    let strict = ensure_strict(schema.clone(), &root, &mut budget, 1, true)?;
    ensure_strict_root(strict)
}

fn validate_depth(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(AgentsError::user(
            "JSON schema is too deeply nested to process safely. Simplify or flatten the schema.",
        ));
    }
    match value {
        Value::Object(m) => m.values().try_for_each(|v| validate_depth(v, depth + 1)),
        Value::Array(a) => a.iter().try_for_each(|v| validate_depth(v, depth + 1)),
        _ => Ok(()),
    }
}

fn ensure_strict_root(mut schema: Value) -> Result<Value> {
    let Value::Object(obj) = &mut schema else {
        return Ok(schema);
    };
    if obj.get("anyOf").map(Value::is_array).unwrap_or(false) {
        return Err(AgentsError::user(
            "The root of a strict JSON schema must not use `anyOf`.",
        ));
    }
    if let Some(Value::Array(types)) = obj.get("type").cloned() {
        let has_object = types.iter().any(|t| t == "object");
        if has_object {
            if types.len() == 1 {
                obj.insert("type".into(), Value::String("object".into()));
            } else {
                return Err(AgentsError::user(format!(
                    "The root of a strict JSON schema must be a non-nullable object, but its type is {types:?}."
                )));
            }
        }
    }
    Ok(schema)
}

fn spend(budget: &mut usize) -> Result<()> {
    if *budget == 0 {
        return Err(AgentsError::user(
            "JSON schema is too large to convert to strict mode.",
        ));
    }
    *budget -= 1;
    Ok(())
}

fn ensure_strict(
    schema: Value,
    root: &Value,
    budget: &mut usize,
    depth: usize,
    is_root: bool,
) -> Result<Value> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(AgentsError::user(
            "JSON schema is too deeply nested to process safely.",
        ));
    }
    let Value::Object(mut obj) = schema else {
        return Err(AgentsError::user(format!(
            "Expected a JSON schema object, got {schema}"
        )));
    };
    let explicitly_closed = obj.get("additionalProperties") == Some(&Value::Bool(false));
    spend(budget)?;
    strip_unsupported_format(&mut obj);
    if is_root {
        obj.remove("$schema");
    }
    let next = depth + 1;

    if obj.contains_key("$ref") {
        let allowed: &[&str] = if is_root {
            ROOT_REF_NON_CONSTRAINING_SIBLINGS
        } else {
            REF_NON_CONSTRAINING_SIBLINGS
        };
        let incompatible: Vec<&String> = obj
            .keys()
            .filter(|k| k.as_str() != "$ref" && !allowed.contains(&k.as_str()))
            .collect();
        if !incompatible.is_empty() {
            return Err(AgentsError::user(format!(
                "JSON schema contains a `$ref` with incompatible sibling keyword(s) ({}) that cannot be merged without changing its accepted values.",
                incompatible
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    for defs_key in ["$defs", "definitions"] {
        if let Some(Value::Object(defs)) = obj.get(defs_key).cloned() {
            let mut new_defs = Map::new();
            for (name, def) in defs {
                new_defs.insert(name, ensure_strict(def, root, budget, next, false)?);
            }
            obj.insert(defs_key.into(), Value::Object(new_defs));
        }
    }

    let mut typ = obj.get("type").cloned();
    let has_props_obj = obj.get("properties").map(Value::is_object).unwrap_or(false);
    if typ.is_none() && has_props_obj {
        obj.insert("type".into(), Value::String("object".into()));
        typ = Some(Value::String("object".into()));
    } else if typ.is_none()
        && obj
            .get("additionalProperties")
            .map(|v| v != &Value::Bool(false))
            .unwrap_or(false)
    {
        return Err(AgentsError::user(ADDITIONAL_PROPERTIES_ERROR));
    }
    let is_object = match &typ {
        Some(Value::String(s)) => s == "object",
        Some(Value::Array(a)) => a.iter().any(|t| t == "object"),
        _ => false,
    };
    if is_object {
        match obj.get("additionalProperties") {
            None => {
                obj.insert("additionalProperties".into(), Value::Bool(false));
            }
            Some(Value::Bool(false)) => {}
            Some(_) => return Err(AgentsError::user(ADDITIONAL_PROPERTIES_ERROR)),
        }
    }

    if let Some(Value::Object(props)) = obj.get("properties").cloned() {
        let keys: Vec<Value> = props.keys().map(|k| Value::String(k.clone())).collect();
        obj.insert("required".into(), Value::Array(keys));
        let mut new_props = Map::new();
        for (k, v) in props {
            new_props.insert(k, ensure_strict(v, root, budget, next, false)?);
        }
        obj.insert("properties".into(), Value::Object(new_props));
    }

    if let Some(items) = obj.get("items").cloned()
        && items.is_object()
    {
        obj.insert("items".into(), ensure_strict(items, root, budget, next, false)?);
    }

    if let Some(Value::Array(any_of)) = obj.get("anyOf").cloned() {
        let mut out = Vec::with_capacity(any_of.len());
        for v in any_of {
            out.push(ensure_strict(v, root, budget, next, false)?);
        }
        obj.insert("anyOf".into(), Value::Array(out));
    }

    if let Some(Value::Array(one_of)) = obj.get("oneOf").cloned() {
        let mut existing = match obj.get("anyOf") {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        for v in one_of {
            existing.push(ensure_strict(v, root, budget, next, false)?);
        }
        obj.insert("anyOf".into(), Value::Array(existing));
        obj.remove("oneOf");
    }

    if let Some(Value::Array(all_of)) = obj.get("allOf").cloned() {
        if all_of.len() == 1 {
            let entry = all_of[0].clone();
            let strict_entry = ensure_strict(entry, root, budget, next, false)?;
            obj.remove("allOf");
            if !explicitly_closed {
                obj.remove("additionalProperties");
            }
            let merged = merge_single_all_of(&strict_entry, &obj)?;
            return ensure_strict(Value::Object(merged), root, budget, next, false);
        } else {
            let mut out = Vec::with_capacity(all_of.len());
            for v in all_of {
                out.push(ensure_strict(v, root, budget, next, false)?);
            }
            obj.insert("allOf".into(), Value::Array(out));
        }
    }

    if obj.get("default") == Some(&Value::Null) {
        obj.remove("default");
    }

    if let Some(Value::String(r)) = obj.get("$ref").cloned()
        && obj.len() > 1
    {
        let resolved = resolve_ref(root, &r)?;
        let Value::Object(resolved_obj) = resolved else {
            return Err(AgentsError::user(format!(
                "Expected `$ref: {r}` to resolve to an object"
            )));
        };
        obj.remove("$ref");
        let mut merged = resolved_obj;
        for (k, v) in obj {
            merged.insert(k, v);
        }
        return ensure_strict(Value::Object(merged), root, budget, next, false);
    }

    Ok(Value::Object(obj))
}

fn merge_single_all_of(entry: &Value, parent: &Map<String, Value>) -> Result<Map<String, Value>> {
    let Value::Object(entry_obj) = entry else {
        return Err(AgentsError::user("allOf entry must be an object"));
    };
    let mut merged = entry_obj.clone();
    for (k, v) in parent {
        match (k.as_str(), merged.get(k)) {
            ("properties", Some(Value::Object(existing))) => {
                let mut props = existing.clone();
                if let Value::Object(pv) = v {
                    props.extend(pv.clone());
                }
                merged.insert(k.clone(), Value::Object(props));
            }
            ("required", Some(Value::Array(existing))) => {
                let mut req = existing.clone();
                if let Value::Array(rv) = v {
                    for r in rv {
                        if !req.contains(r) {
                            req.push(r.clone());
                        }
                    }
                }
                merged.insert(k.clone(), Value::Array(req));
            }
            _ => {
                merged.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(merged)
}

/// Resolve a local `#/...` JSON pointer against the schema root.
pub fn resolve_ref(root: &Value, reference: &str) -> Result<Value> {
    let Some(path) = reference.strip_prefix("#/") else {
        return Err(AgentsError::user(format!(
            "Unexpected $ref format {reference:?}; does not start with #/"
        )));
    };
    let mut current = root;
    for raw in path.split('/') {
        let key = raw.replace("~1", "/").replace("~0", "~");
        current = match current {
            Value::Object(m) => m.get(&key).ok_or_else(|| {
                AgentsError::user(format!("Encountered missing key {key} while resolving {reference}"))
            })?,
            Value::Array(a) => {
                let idx: usize = key.parse().map_err(|_| {
                    AgentsError::user(format!("Invalid array index {key} while resolving {reference}"))
                })?;
                a.get(idx).ok_or_else(|| {
                    AgentsError::user(format!("Index {idx} out of range while resolving {reference}"))
                })?
            }
            _ => {
                return Err(AgentsError::user(format!(
                    "Cannot resolve {reference}: encountered a non-container value"
                )));
            }
        };
    }
    Ok(current.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closes_objects_and_requires_all_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer", "default": null}
            }
        });
        let strict = ensure_strict_json_schema(&schema).unwrap();
        assert_eq!(strict["additionalProperties"], false);
        assert_eq!(strict["required"], serde_json::json!(["a", "b"]));
        assert!(strict["properties"]["b"].get("default").is_none());
    }

    #[test]
    fn inlines_refs_with_siblings_and_converts_one_of() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "loc": {"$ref": "#/$defs/Loc", "description": "where"},
                "u": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
            },
            "$defs": {"Loc": {"type": "object", "properties": {"lat": {"type": "number"}}}}
        });
        let strict = ensure_strict_json_schema(&schema).unwrap();
        let loc = &strict["properties"]["loc"];
        assert_eq!(loc["type"], "object");
        assert_eq!(loc["description"], "where");
        assert_eq!(loc["additionalProperties"], false);
        assert!(strict["properties"]["u"].get("oneOf").is_none());
        assert!(strict["properties"]["u"]["anyOf"].is_array());
    }

    #[test]
    fn strips_numeric_formats_but_keeps_string_formats() {
        let schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "n": {"type": "integer", "format": "uint32", "minimum": 0},
                "d": {"type": "string", "format": "date-time"},
                "x": {"type": "string", "format": "binary"}
            }
        });
        let strict = ensure_strict_json_schema(&schema).unwrap();
        assert!(strict.get("$schema").is_none());
        assert!(strict["properties"]["n"].get("format").is_none());
        assert_eq!(strict["properties"]["n"]["minimum"], 0);
        assert_eq!(strict["properties"]["d"]["format"], "date-time");
        assert!(strict["properties"]["x"].get("format").is_none());
    }

    #[test]
    fn rejects_open_objects() {
        let schema = serde_json::json!({"type": "object", "additionalProperties": true});
        assert!(ensure_strict_json_schema(&schema).is_err());
    }
}
