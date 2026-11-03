//! Safe model feedback for invalid tool arguments. Mirrors
//! `agents.sandbox.capabilities.tools._input_errors`.
//!
//! A serde error echoes the offending value and a generic tool failure surfaces as a run error.
//! For the sandbox tools the model gets one short line naming only the declared fields that
//! failed validation, so it can correct the call without the input value ever being repeated
//! into the transcript, the logs or a trace.

use std::future::Future;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::Result;
use crate::tool::{FunctionTool, IntoToolOutput, ToolOutput};
use crate::tool_context::ToolContext;

/// What a declared argument must look like.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FieldKind {
    /// A required string, optionally non-empty.
    Str { non_empty: bool },
    /// A string or null (or absent).
    OptStr,
    /// A boolean, absent allowed.
    Bool,
    /// A required integer at least `min`.
    Int { min: i64 },
    /// An optional integer at least `min`: absent and null are fine.
    OptInt { min: i64 },
    /// An optional integer at least `min` that may only be absent, never null.
    DefaultedInt { min: i64 },
    /// A string that may be absent, never null.
    DefaultedStr,
}

/// A declared field, in the order the tool schema lists them.
pub(crate) type FieldSpec = (&'static str, FieldKind);

fn valid(kind: FieldKind, value: Option<&Value>) -> bool {
    let integer = |value: &Value, min: i64| {
        value
            .as_i64()
            .or_else(|| {
                value
                    .as_f64()
                    .filter(|f| f.fract() == 0.0)
                    .map(|f| f as i64)
            })
            .is_some_and(|n| n >= min)
    };
    match (kind, value) {
        (FieldKind::Str { non_empty }, Some(Value::String(text))) => !non_empty || !text.is_empty(),
        (FieldKind::Str { .. }, _) => false,
        (FieldKind::OptStr, None | Some(Value::Null | Value::String(_))) => true,
        (FieldKind::OptStr, _) => false,
        (FieldKind::DefaultedStr, None | Some(Value::String(_))) => true,
        (FieldKind::DefaultedStr, _) => false,
        (FieldKind::Bool, None | Some(Value::Bool(_))) => true,
        (FieldKind::Bool, _) => false,
        (FieldKind::Int { min }, Some(value)) => integer(value, min),
        (FieldKind::Int { .. }, None) => false,
        (FieldKind::OptInt { .. }, None | Some(Value::Null)) => true,
        (FieldKind::OptInt { min }, Some(value)) => integer(value, min),
        (FieldKind::DefaultedInt { .. }, None) => true,
        (FieldKind::DefaultedInt { min }, Some(value)) => integer(value, min),
    }
}

/// The feedback line for `raw`, or `None` when every declared field validates.
pub(crate) fn invalid_arguments(raw: &str, fields: &[FieldSpec]) -> Option<String> {
    let trimmed = raw.trim();
    let parsed: Option<Value> = if trimmed.is_empty() {
        Some(Value::Object(Default::default()))
    } else {
        serde_json::from_str(trimmed).ok()
    };
    let Some(Value::Object(object)) = parsed else {
        return Some(feedback(&[]));
    };
    let bad: Vec<&str> = fields
        .iter()
        .filter(|(name, kind)| !valid(*kind, object.get(*name)))
        .map(|(name, _)| *name)
        .collect();
    if bad.is_empty() {
        None
    } else {
        Some(feedback(&bad))
    }
}

fn feedback(fields: &[&str]) -> String {
    let target = if fields.is_empty() {
        String::new()
    } else {
        format!(" for {}", fields.join(", "))
    };
    format!("Invalid tool arguments{target}. Check the tool schema and try again.")
}

/// A function tool whose invalid arguments come back to the model as safe feedback instead of a
/// run error.
pub(crate) fn tool_with_safe_input<C, Args, F, Fut, T>(
    name: &'static str,
    description: &'static str,
    fields: &'static [FieldSpec],
    f: F,
) -> FunctionTool<C>
where
    C: Send + Sync + 'static,
    Args: JsonSchema + DeserializeOwned + Send + 'static,
    F: Fn(ToolContext<C>, Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
    T: IntoToolOutput + Send + 'static,
{
    let mut schema = schemars::schema_for!(Args).to_value();
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
    }
    let f = Arc::new(f);
    FunctionTool::<C>::new_raw(name, description, schema, move |ctx, raw: String| {
        let f = f.clone();
        async move {
            if let Some(message) = invalid_arguments(&raw, fields) {
                return Ok(ToolOutput::Text(message));
            }
            let Ok(args) =
                serde_json::from_str::<Args>(if raw.trim().is_empty() { "{}" } else { &raw })
            else {
                return Ok(ToolOutput::Text(feedback(&[])));
            };
            Ok(f(ctx, args).await?.into_tool_output())
        }
    })
}
