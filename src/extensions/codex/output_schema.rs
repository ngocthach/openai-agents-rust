//! Structured output schemas for Codex turns. Mirrors
//! `agents.extensions.experimental.codex.output_schema_file` and the schema helpers of
//! `codex_tool.py`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::error::{AgentsError, Result};
use crate::strict_schema::ensure_strict_json_schema;

/// A JSON primitive usable in an [`OutputSchemaField`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonPrimitive {
    /// `"string"`.
    String,
    /// `"number"`.
    Number,
    /// `"integer"`.
    Integer,
    /// `"boolean"`.
    Boolean,
}

impl JsonPrimitive {
    /// The JSON schema `type` value.
    pub fn as_str(&self) -> &'static str {
        match self {
            JsonPrimitive::String => "string",
            JsonPrimitive::Number => "number",
            JsonPrimitive::Integer => "integer",
            JsonPrimitive::Boolean => "boolean",
        }
    }
}

/// The schema of one property of an [`OutputSchemaDescriptor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputSchemaField {
    /// A JSON primitive, optionally constrained to an enum of strings.
    Primitive {
        /// The primitive type.
        primitive: JsonPrimitive,
        /// Field description.
        description: Option<String>,
        /// Allowed values.
        enum_values: Option<Vec<String>>,
    },
    /// An array of primitives or nested arrays.
    Array {
        /// Field description.
        description: Option<String>,
        /// Schema of the array items.
        items: Box<OutputSchemaField>,
    },
}

impl OutputSchemaField {
    /// A string field.
    pub fn string() -> Self {
        Self::primitive(JsonPrimitive::String)
    }

    /// A number field.
    pub fn number() -> Self {
        Self::primitive(JsonPrimitive::Number)
    }

    /// An integer field.
    pub fn integer() -> Self {
        Self::primitive(JsonPrimitive::Integer)
    }

    /// A boolean field.
    pub fn boolean() -> Self {
        Self::primitive(JsonPrimitive::Boolean)
    }

    /// A field of the given primitive type.
    pub fn primitive(primitive: JsonPrimitive) -> Self {
        OutputSchemaField::Primitive {
            primitive,
            description: None,
            enum_values: None,
        }
    }

    /// An array field.
    pub fn array(items: OutputSchemaField) -> Self {
        OutputSchemaField::Array {
            description: None,
            items: Box::new(items),
        }
    }

    /// Attach a description.
    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        match &mut self {
            OutputSchemaField::Primitive { description, .. }
            | OutputSchemaField::Array { description, .. } => *description = Some(value.into()),
        }
        self
    }

    /// Restrict a primitive field to a set of values.
    pub fn with_enum(mut self, values: Vec<String>) -> Self {
        if let OutputSchemaField::Primitive { enum_values, .. } = &mut self {
            *enum_values = Some(values);
        }
        self
    }

    fn to_json_schema(&self) -> Value {
        match self {
            OutputSchemaField::Primitive {
                primitive,
                description,
                enum_values,
            } => {
                let mut schema = json!({"type": primitive.as_str()});
                let map = schema.as_object_mut().expect("object");
                if let Some(d) = description.as_ref().filter(|d| !d.is_empty()) {
                    map.insert("description".into(), Value::String(d.clone()));
                }
                if let Some(values) = enum_values {
                    map.insert("enum".into(), json!(values));
                }
                schema
            }
            OutputSchemaField::Array { description, items } => {
                let mut schema = json!({"type": "array", "items": items.to_json_schema()});
                if let Some(d) = description.as_ref().filter(|d| !d.is_empty()) {
                    schema
                        .as_object_mut()
                        .expect("object")
                        .insert("description".into(), Value::String(d.clone()));
                }
                schema
            }
        }
    }
}

/// One property of an [`OutputSchemaDescriptor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSchemaProperty {
    /// Property name.
    pub name: String,
    /// Property description, merged into the field schema.
    pub description: Option<String>,
    /// Property schema.
    pub schema: OutputSchemaField,
}

impl OutputSchemaProperty {
    /// A property with the given name and schema.
    pub fn new(name: impl Into<String>, schema: OutputSchemaField) -> Self {
        OutputSchemaProperty {
            name: name.into(),
            description: None,
            schema,
        }
    }

    /// Attach a description.
    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }
}

/// A compact description of the structured output Codex should produce.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputSchemaDescriptor {
    /// Schema title.
    pub title: Option<String>,
    /// Schema description.
    pub description: Option<String>,
    /// The output properties.
    pub properties: Vec<OutputSchemaProperty>,
    /// Properties the model must always produce.
    pub required: Vec<String>,
}

impl OutputSchemaDescriptor {
    /// An empty descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the schema title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the schema description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a property.
    pub fn with_property(mut self, property: OutputSchemaProperty) -> Self {
        self.properties.push(property);
        self
    }

    /// Mark properties as required.
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = required;
        self
    }

    /// Build the strict JSON object schema Codex expects.
    pub fn to_json_schema(&self) -> Result<Value> {
        if self.properties.is_empty() {
            return Err(AgentsError::user(
                "Codex output schema descriptor must include properties.",
            ));
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut properties = Map::new();
        for property in &self.properties {
            let name = property.name.trim();
            if name.is_empty() {
                return Err(AgentsError::user(
                    "Codex output schema properties must include non-empty names.",
                ));
            }
            if !seen.insert(name) {
                return Err(AgentsError::user(format!(
                    "Duplicate property name \"{name}\" in output_schema."
                )));
            }
            let mut schema = property.schema.to_json_schema();
            if let Some(description) = property.description.as_ref().filter(|d| !d.is_empty()) {
                schema
                    .as_object_mut()
                    .expect("object")
                    .insert("description".into(), Value::String(description.clone()));
            }
            properties.insert(name.to_string(), schema);
        }
        for name in &self.required {
            if !seen.contains(name.as_str()) {
                return Err(AgentsError::user(format!(
                    "Required property \"{name}\" must also be defined in \"properties\"."
                )));
            }
        }
        let mut schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": properties,
            "required": self.required,
        });
        let map = schema.as_object_mut().expect("object");
        if let Some(title) = self.title.as_ref().filter(|t| !t.is_empty()) {
            map.insert("title".into(), Value::String(title.clone()));
        }
        if let Some(description) = self.description.as_ref().filter(|d| !d.is_empty()) {
            map.insert("description".into(), Value::String(description.clone()));
        }
        Ok(schema)
    }
}

/// The structured output requested from Codex.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputSchema {
    /// A descriptor converted to a strict JSON schema.
    Descriptor(OutputSchemaDescriptor),
    /// A JSON object schema, converted to strict mode as-is.
    JsonSchema(Value),
}

impl OutputSchema {
    /// Build the JSON schema written to the Codex output schema file.
    pub fn to_json_schema(&self) -> Result<Value> {
        match self {
            OutputSchema::Descriptor(descriptor) => descriptor.to_json_schema(),
            OutputSchema::JsonSchema(schema) => {
                if !schema.is_object() {
                    return Err(AgentsError::user(
                        "Codex output schema must be a plain JSON object.",
                    ));
                }
                match schema.get("type").and_then(Value::as_str) {
                    None | Some("object") => {}
                    Some(_) => {
                        return Err(AgentsError::user(
                            "Codex output schema must be a JSON object schema with type \"object\".",
                        ));
                    }
                }
                ensure_strict_json_schema(schema)
            }
        }
    }
}

impl From<OutputSchemaDescriptor> for OutputSchema {
    fn from(value: OutputSchemaDescriptor) -> Self {
        OutputSchema::Descriptor(value)
    }
}

impl From<Value> for OutputSchema {
    fn from(value: Value) -> Self {
        OutputSchema::JsonSchema(value)
    }
}

/// A JSON schema materialized on disk for the Codex CLI. The file and its directory are removed
/// when the value is dropped.
#[derive(Debug)]
pub struct OutputSchemaFile {
    dir: PathBuf,
    path: PathBuf,
}

impl OutputSchemaFile {
    /// Write `schema` into a fresh temp directory.
    pub fn create(schema: &Value) -> Result<Self> {
        if !schema.is_object() {
            return Err(AgentsError::user(
                "output_schema must be a plain JSON object",
            ));
        }
        let dir =
            std::env::temp_dir().join(format!("codex-output-schema-{}", crate::util::gen_hex32()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("schema.json");
        let file = OutputSchemaFile { dir, path };
        std::fs::write(&file.path, serde_json::to_vec(schema)?)?;
        Ok(file)
    }

    /// Path of the schema file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for OutputSchemaFile {
    fn drop(&mut self) {
        // Best effort: the turn is over either way.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_schema_from_descriptor() {
        let schema = OutputSchemaDescriptor::new()
            .with_title("Summary")
            .with_description("Codex summary")
            .with_property(
                OutputSchemaProperty::new("headline", OutputSchemaField::string())
                    .with_description("One line"),
            )
            .with_property(OutputSchemaProperty::new(
                "tags",
                OutputSchemaField::array(
                    OutputSchemaField::string().with_enum(vec!["a".into(), "b".into()]),
                ),
            ))
            .with_required(vec!["headline".into()])
            .to_json_schema()
            .unwrap();
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "headline": {"type": "string", "description": "One line"},
                    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
                },
                "required": ["headline"],
                "title": "Summary",
                "description": "Codex summary",
            })
        );
    }

    #[test]
    fn rejects_invalid_descriptors() {
        assert!(OutputSchemaDescriptor::new().to_json_schema().is_err());
        let duplicate = OutputSchemaDescriptor::new()
            .with_property(OutputSchemaProperty::new("a", OutputSchemaField::string()))
            .with_property(OutputSchemaProperty::new("a", OutputSchemaField::string()));
        assert!(duplicate.to_json_schema().is_err());
        let missing = OutputSchemaDescriptor::new()
            .with_property(OutputSchemaProperty::new("a", OutputSchemaField::string()))
            .with_required(vec!["b".into()]);
        assert!(missing.to_json_schema().is_err());
    }

    #[test]
    fn rejects_non_object_json_schema() {
        let schema = OutputSchema::JsonSchema(json!({"type": "array"}));
        assert!(schema.to_json_schema().is_err());
    }

    #[test]
    fn makes_json_schema_strict() {
        let schema = OutputSchema::JsonSchema(json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }))
        .to_json_schema()
        .unwrap();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], json!(["answer"]));
    }

    #[test]
    fn writes_and_removes_the_schema_file() {
        let schema = json!({"type": "object", "properties": {}});
        let file = OutputSchemaFile::create(&schema).unwrap();
        let path = file.path().to_path_buf();
        let written: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("schema file")).unwrap();
        assert_eq!(written, schema);
        drop(file);
        assert!(!path.exists());
    }
}
