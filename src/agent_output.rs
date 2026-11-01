//! Structured output schemas for agents. Mirrors `agents.agent_output`.

use std::fmt;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::error::{AgentsError, Result};
use crate::strict_schema::ensure_strict_json_schema;

const WRAPPER_DICT_KEY: &str = "response";

/// An object that captures the JSON schema of the output, and validates model output against it.
pub trait AgentOutputSchemaBase: Send + Sync + fmt::Debug {
    /// Whether the output type is plain text (no structured output).
    fn is_plain_text(&self) -> bool;
    /// The name of the output type.
    fn name(&self) -> String;
    /// The JSON schema of the output type.
    fn json_schema(&self) -> Result<Value>;
    /// Whether the schema is in strict mode.
    fn is_strict_json_schema(&self) -> bool;
    /// Validate a JSON string against the output type and return the parsed value.
    fn validate_json(&self, json_str: &str) -> Result<Value>;
}

type Validator = Arc<dyn Fn(&str) -> Result<Value> + Send + Sync>;

/// Output schema derived from a Rust type implementing `schemars::JsonSchema` and
/// `serde::Deserialize`, or from a raw JSON schema.
#[derive(Clone)]
pub struct AgentOutputSchema {
    type_name: String,
    schema: Value,
    strict: bool,
    is_wrapped: bool,
    validator: Validator,
}

impl fmt::Debug for AgentOutputSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentOutputSchema")
            .field("type_name", &self.type_name)
            .field("strict", &self.strict)
            .field("is_wrapped", &self.is_wrapped)
            .finish()
    }
}

/// Drop the module path from every segment of a type name, so
/// `alloc::vec::Vec<alloc::string::String>` reads `Vec<String>`.
fn short_type_name(full: &str) -> String {
    let mut out = String::new();
    let mut segment = String::new();
    let mut chars = full.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            segment.clear();
        } else if c.is_alphanumeric() || c == '_' {
            segment.push(c);
        } else {
            out.push_str(&segment);
            segment.clear();
            out.push(c);
        }
    }
    out.push_str(&segment);
    if out.is_empty() {
        "Output".to_string()
    } else {
        out
    }
}

impl AgentOutputSchema {
    /// Build a strict output schema for `T`.
    pub fn new<T: JsonSchema + DeserializeOwned + 'static>() -> Result<Self> {
        Self::new_with_strict::<T>(true)
    }

    /// Build an output schema for `T`, choosing whether to convert it to strict mode.
    pub fn new_with_strict<T: JsonSchema + DeserializeOwned + 'static>(
        strict: bool,
    ) -> Result<Self> {
        let raw = schemars::schema_for!(T).to_value();
        let type_name = short_type_name(std::any::type_name::<T>());
        let target = format!("output type {type_name}");
        let validator: Validator = Arc::new(move |s: &str| {
            let parsed: T = serde_json::from_str(s)
                .map_err(|e| AgentsError::invalid_json(s, &target, &e, false))?;
            // Re-serialize so the caller gets a JSON value; T: Deserialize only, so use the
            // original string parsed as a generic value.
            drop(parsed);
            serde_json::from_str::<Value>(s).map_err(|e| AgentsError::model_behavior(e.to_string()))
        });
        Self::build(type_name, raw, strict, validator)
    }

    /// Build an output schema from a raw JSON schema. Validation only checks that the output is
    /// valid JSON.
    pub fn from_json_schema(name: impl Into<String>, schema: Value, strict: bool) -> Result<Self> {
        let validator: Validator = Arc::new(|s: &str| {
            serde_json::from_str::<Value>(s)
                .map_err(|e| AgentsError::invalid_json(s, "output type", &e, false))
        });
        Self::build(name.into(), schema, strict, validator)
    }

    fn build(type_name: String, raw: Value, strict: bool, validator: Validator) -> Result<Self> {
        let mut schema = raw;
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
        }
        let is_object = schema.get("type").and_then(Value::as_str) == Some("object")
            || schema.get("properties").is_some();
        let is_wrapped = !is_object;
        if is_wrapped {
            let mut inner = schema.clone();
            let mut defs = Map::new();
            if let Value::Object(m) = &mut inner {
                if let Some(Value::Object(d)) = m.remove("$defs") {
                    defs = d;
                }
                m.remove("title");
            }
            let mut wrapper = json!({
                "type": "object",
                "properties": { WRAPPER_DICT_KEY: inner },
                "required": [WRAPPER_DICT_KEY],
                "title": format!("{type_name}Output"),
            });
            if !defs.is_empty() {
                wrapper["$defs"] = Value::Object(defs);
            }
            schema = wrapper;
        }
        if strict {
            schema = ensure_strict_json_schema(&schema).map_err(|e| {
                AgentsError::user(format!(
                    "Strict JSON schema is enabled, but the output type is not valid. Either make the output type strict, or build the schema with strict=false. Cause: {e}"
                ))
            })?;
        }
        Ok(AgentOutputSchema {
            type_name,
            schema,
            strict,
            is_wrapped,
            validator,
        })
    }

    /// Whether the type was wrapped in a `{"response": ...}` object.
    pub fn is_wrapped(&self) -> bool {
        self.is_wrapped
    }
}

impl AgentOutputSchemaBase for AgentOutputSchema {
    fn is_plain_text(&self) -> bool {
        false
    }

    fn name(&self) -> String {
        self.type_name.clone()
    }

    fn json_schema(&self) -> Result<Value> {
        Ok(self.schema.clone())
    }

    fn is_strict_json_schema(&self) -> bool {
        self.strict
    }

    fn validate_json(&self, json_str: &str) -> Result<Value> {
        if self.is_wrapped {
            let parsed: Value = serde_json::from_str(json_str)
                .map_err(|e| AgentsError::invalid_json(json_str, &self.type_name, &e, false))?;
            // The payload only reaches the message when the model data policy allows it.
            let payload = crate::logger::model_data_payload(json_str, false);
            let Value::Object(mut m) = parsed else {
                return Err(AgentsError::model_behavior(format!(
                    "Expected a dict, got {payload}"
                )));
            };
            let Some(inner) = m.remove(WRAPPER_DICT_KEY) else {
                return Err(AgentsError::model_behavior(format!(
                    "Could not find key {WRAPPER_DICT_KEY} in JSON: {payload}"
                )));
            };
            (self.validator)(&inner.to_string())
        } else {
            (self.validator)(json_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct CalendarEvent {
        name: String,
        date: String,
        participants: Vec<String>,
    }

    #[test]
    fn object_schema_is_strict_and_validates() {
        let schema = AgentOutputSchema::new::<CalendarEvent>().unwrap();
        assert!(!schema.is_wrapped());
        let js = schema.json_schema().unwrap();
        assert_eq!(js["additionalProperties"], false);
        assert_eq!(js["required"].as_array().unwrap().len(), 3);
        let v = schema
            .validate_json(r#"{"name":"a","date":"b","participants":["c"]}"#)
            .unwrap();
        assert_eq!(v["name"], "a");
        assert!(schema.validate_json(r#"{"name":"a"}"#).is_err());
    }

    #[test]
    fn non_object_types_are_wrapped() {
        let schema = AgentOutputSchema::new::<Vec<String>>().unwrap();
        assert!(schema.is_wrapped());
        let js = schema.json_schema().unwrap();
        assert_eq!(js["properties"]["response"]["type"], "array");
        let v = schema.validate_json(r#"{"response":["x","y"]}"#).unwrap();
        assert_eq!(v, json!(["x", "y"]));
    }
}
