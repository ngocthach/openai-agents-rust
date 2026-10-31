//! Conversion of JSON schemas to the strict subset accepted by OpenAI structured outputs.
//! Port of `agents.strict_schema`.

use serde_json::{Map, Value};

use crate::error::{AgentsError, Result};

const MAX_SCHEMA_NODES: usize = 100_000;
const MAX_SCHEMA_DEPTH: usize = 100;

const ADDITIONAL_PROPERTIES_ERROR: &str = "additionalProperties should not be set for object types. \
If you really need this, update the function or output tool to not use a strict schema.";

const NESTED_RESOURCE_REF_ERROR: &str = "JSON schema contains a reference owned by or crossing a nested `$id` resource that cannot \
be resolved against the document root.";

const REF_NON_CONSTRAINING_SIBLINGS: &[&str] = &[
    "$anchor",
    "$comment",
    "$defs",
    "$schema",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];

const ROOT_REF_NON_CONSTRAINING_SIBLINGS: &[&str] = &[
    "$anchor",
    "$comment",
    "$defs",
    "$id",
    "$schema",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];

/// String formats accepted by OpenAI structured outputs.
const SUPPORTED_STRING_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
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
    let strict = ensure_strict(schema.clone(), &root, &mut budget, 1, true, false)?;
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

/// Whether a schema node declares its own `$id` resource, as `_declares_schema_resource` does.
fn declares_schema_resource(obj: &Map<String, Value>) -> bool {
    matches!(obj.get("$id"), Some(Value::String(_)))
}

/// JSON-level equality: booleans never equal numbers, and integers equal equal floats.
fn json_values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(_), _) | (_, Value::Bool(_)) => left == right,
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_values_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| json_values_equal(v, other)))
        }
        _ => left == right,
    }
}

/// Follow a chain of `$ref`s that only carry non-constraining siblings, as
/// `_resolve_non_constraining_ref_chain` does.
fn resolve_non_constraining_ref_chain(
    root: &Value,
    schema: &Map<String, Value>,
    budget: &mut usize,
) -> Result<Value> {
    let mut resolved = schema.clone();
    let mut seen_refs: Vec<String> = Vec::new();
    let mut carried: Map<String, Value> = Map::new();
    loop {
        let Some(Value::String(reference)) = resolved.get("$ref").cloned() else {
            break;
        };
        if resolved
            .keys()
            .any(|k| k != "$ref" && !REF_NON_CONSTRAINING_SIBLINGS.contains(&k.as_str()))
        {
            break;
        }
        if seen_refs.contains(&reference) {
            return Err(AgentsError::user(
                "JSON schema contains a circular `$ref` chain.",
            ));
        }
        seen_refs.push(reference.clone());
        for (key, value) in &resolved {
            if key != "$ref" && !carried.contains_key(key) {
                carried.insert(key.clone(), value.clone());
            }
        }
        spend(budget)?;
        let target = resolve_ref(root, &reference)?;
        let Value::Object(target) = target else {
            return Err(AgentsError::user(format!(
                "Expected `$ref: {reference}` to resolve to an object"
            )));
        };
        resolved = target;
    }
    for (key, value) in carried {
        resolved.insert(key, value);
    }
    Ok(Value::Object(resolved))
}

fn ensure_strict(
    schema: Value,
    root: &Value,
    budget: &mut usize,
    depth: usize,
    is_root: bool,
    inside_nested_resource: bool,
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
    let inside_nested_resource =
        inside_nested_resource || (!is_root && declares_schema_resource(&obj));

    if obj.contains_key("$ref") {
        if inside_nested_resource {
            return Err(AgentsError::user(NESTED_RESOURCE_REF_ERROR));
        }
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
                new_defs.insert(
                    name,
                    ensure_strict(def, root, budget, next, false, inside_nested_resource)?,
                );
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
            new_props.insert(
                k,
                ensure_strict(v, root, budget, next, false, inside_nested_resource)?,
            );
        }
        obj.insert("properties".into(), Value::Object(new_props));
    }

    if let Some(items) = obj.get("items").cloned()
        && items.is_object()
    {
        obj.insert(
            "items".into(),
            ensure_strict(items, root, budget, next, false, inside_nested_resource)?,
        );
    }

    if let Some(Value::Array(any_of)) = obj.get("anyOf").cloned() {
        let mut out = Vec::with_capacity(any_of.len());
        for v in any_of {
            out.push(ensure_strict(
                v,
                root,
                budget,
                next,
                false,
                inside_nested_resource,
            )?);
        }
        obj.insert("anyOf".into(), Value::Array(out));
    }

    if let Some(Value::Array(one_of)) = obj.get("oneOf").cloned() {
        let mut existing = match obj.get("anyOf") {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        for v in one_of {
            existing.push(ensure_strict(
                v,
                root,
                budget,
                next,
                false,
                inside_nested_resource,
            )?);
        }
        obj.insert("anyOf".into(), Value::Array(existing));
        obj.remove("oneOf");
    }

    if let Some(Value::Array(all_of)) = obj.get("allOf").cloned() {
        if all_of.len() == 1 {
            let entry = all_of[0].clone();
            let pure_ref_entry = match &entry {
                Value::Object(e) => {
                    e.contains_key("$ref")
                        && !e.keys().any(|k| {
                            k != "$ref" && !REF_NON_CONSTRAINING_SIBLINGS.contains(&k.as_str())
                        })
                }
                _ => false,
            };
            let strict_entry = if pure_ref_entry {
                if inside_nested_resource {
                    return Err(AgentsError::user(NESTED_RESOURCE_REF_ERROR));
                }
                spend(budget)?;
                let Value::Object(e) = &entry else {
                    unreachable!("checked above");
                };
                resolve_non_constraining_ref_chain(root, e, budget)?
            } else {
                ensure_strict(entry, root, budget, next, false, inside_nested_resource)?
            };
            obj.remove("allOf");
            if !explicitly_closed {
                obj.remove("additionalProperties");
            }
            let merged = merge_single_all_of(&strict_entry, &obj)?;
            return ensure_strict(
                Value::Object(merged),
                root,
                budget,
                next,
                false,
                inside_nested_resource,
            );
        } else {
            let mut out = Vec::with_capacity(all_of.len());
            for v in all_of {
                out.push(ensure_strict(
                    v,
                    root,
                    budget,
                    next,
                    false,
                    inside_nested_resource,
                )?);
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
        return ensure_strict(
            Value::Object(merged),
            root,
            budget,
            next,
            false,
            inside_nested_resource,
        );
    }

    Ok(Value::Object(obj))
}

fn merge_single_all_of(entry: &Value, parent: &Map<String, Value>) -> Result<Map<String, Value>> {
    let Value::Object(entry_obj) = entry else {
        return Err(AgentsError::user("allOf entry must be an object"));
    };
    let mut merged = entry_obj.clone();
    let mut incompatible: Vec<String> = Vec::new();

    let parent_properties_empty = match parent.get("properties") {
        None => true,
        Some(Value::Object(p)) => p.is_empty(),
        Some(_) => false,
    };
    let entry_has_properties =
        matches!(entry_obj.get("properties"), Some(Value::Object(p)) if !p.is_empty());
    if parent.get("additionalProperties") == Some(&Value::Bool(false))
        && parent_properties_empty
        && entry_has_properties
    {
        // `additionalProperties` only sees properties declared in its own schema, not in `allOf`.
        incompatible.push("properties".into());
    }

    for (key, parent_value) in parent {
        match merged.get(key) {
            None => {
                merged.insert(key.clone(), parent_value.clone());
            }
            Some(existing) => {
                if REF_NON_CONSTRAINING_SIBLINGS.contains(&key.as_str()) {
                    merged.insert(key.clone(), parent_value.clone());
                } else if !(json_values_equal(parent_value, existing)
                    || (key == "properties" && parent_value == &Value::Object(Map::new()))
                    || (key == "required" && parent_value == &Value::Array(Vec::new())))
                {
                    incompatible.push(key.clone());
                }
            }
        }
    }

    if !incompatible.is_empty() {
        incompatible.sort();
        incompatible.dedup();
        return Err(AgentsError::user(format!(
            "JSON schema contains a singleton `allOf` entry with incompatible parent keyword(s) \
({}) that cannot be merged without changing its accepted values.",
            incompatible.join(", ")
        )));
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
                AgentsError::user(format!(
                    "Encountered missing key {key} while resolving {reference}"
                ))
            })?,
            Value::Array(a) => {
                let idx: usize = key.parse().map_err(|_| {
                    AgentsError::user(format!(
                        "Invalid array index {key} while resolving {reference}"
                    ))
                })?;
                a.get(idx).ok_or_else(|| {
                    AgentsError::user(format!(
                        "Index {idx} out of range while resolving {reference}"
                    ))
                })?
            }
            _ => {
                return Err(AgentsError::user(format!(
                    "Cannot resolve {reference}: encountered a non-container value"
                )));
            }
        };
        if let Value::Object(obj) = current
            && declares_schema_resource(obj)
        {
            return Err(AgentsError::user(NESTED_RESOURCE_REF_ERROR));
        }
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
