//! Schema and converter cases of Python's `tests/test_programmatic_tool_calling.py`: output
//! schemas declared on function tools, their validation, and the Responses wire shape of
//! programmatic tools.
//!
//! Python infers output schemas from return annotations (`Annotated`, `Awaitable`, PEP 695
//! aliases, TypedDict/dataclass returns). Rust names the output type explicitly with
//! `with_output_type::<T>()`, so the annotation-inference cases are Python-only and are not
//! ported here.

use std::sync::Arc;

use openai_agents_sdk::testing::FakeModel;
use openai_agents_sdk::tool::{FunctionTool, ProgrammaticToolCallingTool, Tool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{AgentsError, RunContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct InventoryOutput {
    sku: String,
    available_units: i64,
}

fn lookup_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Ok::<_, AgentsError>(json!({"sku": "A-1", "available_units": 42}))
        },
    )
    .with_allowed_callers(vec!["programmatic".into()])
}

fn invoke_context(tool_name: &str, arguments: &str) -> ToolContext<()> {
    ToolContext::direct(RunContext::new(()), "agent", tool_name, arguments)
}

#[test]
fn responses_converter_serializes_programmatic_tool_configuration() {
    let lookup = lookup_tool().with_output_type::<InventoryOutput>().unwrap();

    let program_param = Tool::<()>::ProgrammaticToolCalling(ProgrammaticToolCallingTool)
        .to_responses_param()
        .0;
    assert_eq!(program_param, json!({"type": "programmatic_tool_calling"}));

    let function_param = lookup.to_responses_param();
    assert_eq!(function_param["allowed_callers"], json!(["programmatic"]));
    assert_eq!(
        function_param["output_schema"],
        json!({
            "additionalProperties": false,
            "properties": {
                "sku": {"type": "string"},
                "available_units": {"type": "integer"},
            },
            "required": ["sku", "available_units"],
            "title": "InventoryOutput",
            "type": "object",
        })
    );
}

#[test]
fn function_tool_accepts_typed_output_type() {
    let tool = lookup_tool().with_output_type::<InventoryOutput>().unwrap();
    let schema = tool.output_json_schema.clone().expect("output schema");
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], json!(false));
}

#[test]
fn function_tool_rejects_non_object_or_non_strict_raw_output_schema() {
    for schema in [
        json!({"type": "string"}),
        json!({"type": "object", "additionalProperties": true}),
    ] {
        let err = lookup_tool().with_output_json_schema(schema).unwrap_err();
        assert!(
            err.to_string().contains("output_json_schema must define"),
            "{err}"
        );
    }
}

#[test]
fn function_tool_accepts_single_element_object_type_array_output_schema() {
    let tool = lookup_tool()
        .with_output_json_schema(json!({
            "type": ["object"],
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": false,
        }))
        .unwrap();
    let schema = tool.output_json_schema.expect("output schema");
    assert_eq!(schema["additionalProperties"], json!(false));
}

#[tokio::test]
async fn function_tool_validates_declared_output_type() {
    let tool = FunctionTool::new(
        "invalid_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Ok::<_, AgentsError>(json!({"sku": "A-1", "available_units": "many"}))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["programmatic".into()]);

    let err = tool
        .invoke(invoke_context("invalid_tool", "{}"), "{}".into())
        .await
        .unwrap_err();

    assert!(
        err.to_string()
            .contains("does not match its declared output type"),
        "{err}"
    );
}

#[tokio::test]
async fn schema_backed_programmatic_output_is_returned_as_json() {
    let tool = lookup_tool().with_output_type::<InventoryOutput>().unwrap();

    let output = tool
        .invoke(invoke_context("lookup_inventory", "{}"), "{}".into())
        .await
        .unwrap();

    let parsed: InventoryOutput = serde_json::from_str(&output.to_model_string()).unwrap();
    assert_eq!(
        parsed,
        InventoryOutput {
            sku: "A-1".into(),
            available_units: 42,
        }
    );
    let _ = Arc::new(FakeModel::new());
}
