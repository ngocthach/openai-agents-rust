//! Failure-formatting cases of Python's `tests/test_programmatic_tool_calling.py`.
//!
//! A schema-backed programmatic tool that fails raises its error to the caller (it does not get
//! the default failure text). A direct call to the same tool gets the default text, and the
//! runner records that text as the function output.

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{FunctionTool, default_tool_error_function};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct InventoryOutput {
    sku: String,
    available_units: i64,
}

const DEFAULT_ERROR: &str = "An error occurred while running the tool. Please try again.";

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[tokio::test]
async fn schema_backed_programmatic_tool_bypasses_default_failure_formatter() {
    let failing = FunctionTool::new(
        "failing_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Err::<Value, _>(AgentsError::user("inventory unavailable"))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["programmatic".into()]);

    let err = failing
        .invoke(
            ToolContext::direct(RunContext::new(()), "agent", "failing_tool", "{}"),
            "{}".into(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("inventory unavailable"), "{err}");

    // The same holds when the tool declares a raw output schema.
    let declared = FunctionTool::new(
        "failing_declared_schema_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Err::<Value, _>(AgentsError::user("declared schema unavailable"))
        },
    )
    .with_output_json_schema(json!({
        "type": "object",
        "properties": {"error": {"type": "string"}},
        "required": ["error"],
        "additionalProperties": false,
    }))
    .unwrap()
    .with_allowed_callers(vec!["programmatic".into()]);

    let err = declared
        .invoke(
            ToolContext::direct(
                RunContext::new(()),
                "agent",
                "failing_declared_schema_tool",
                "{}",
            ),
            "{}".into(),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("declared schema unavailable"),
        "{err}"
    );
}

/// Python checks the argument-error formatting through `on_invoke_tool`. Rust's `invoke` returns
/// the argument error itself and the runner applies the formatter, so the observable check is
/// the runner path: the model's malformed arguments reach the model as the default text.
#[tokio::test]
async fn schema_backed_direct_tool_preserves_argument_error_formatter() {
    #[derive(Deserialize, JsonSchema)]
    struct Args {
        #[allow(dead_code)]
        sku: String,
    }
    let model = setup();
    let tool = FunctionTool::new(
        "failing_tool",
        "",
        |_ctx: ToolContext<()>, args: Args| async move {
            Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["direct".into(), "programmatic".into()]);
    let agent = Agent::<()>::new("inventory")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "function_call",
        "id": "function_item",
        "call_id": "call_bad_args",
        "name": "failing_tool",
        "arguments": "{}",
    }))]);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, "run").await.unwrap();

    let output = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .expect("function output");
    assert_eq!(output, DEFAULT_ERROR);
    assert!(!output.contains("sku"));
}

#[tokio::test]
async fn runner_preserves_direct_error_for_schema_backed_tool() {
    let model = setup();
    let lookup = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Err::<Value, _>(AgentsError::user("inventory unavailable for A-1"))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["direct".into(), "programmatic".into()]);
    let agent = Agent::<()>::new("inventory")
        .with_tool(lookup)
        .with_model_instance(model.clone());
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "function_call",
        "id": "function_item",
        "call_id": "call_lookup",
        "name": "lookup_inventory",
        "arguments": "{\"sku\":\"A-1\"}",
    }))]);
    model.add_turn(vec![assistant_message("inventory lookup failed")]);

    let result = Runner::run(&agent, "Check inventory").await.unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("inventory lookup failed")
    );
    let output = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput {
                output, raw_item, ..
            } => Some((output.clone(), raw_item.clone())),
            _ => None,
        })
        .expect("function output");
    assert_eq!(output.0.as_str(), Some(DEFAULT_ERROR));
    assert_eq!(output.1["output"], DEFAULT_ERROR);
}

#[test]
fn default_error_function_text_is_stable() {
    let ctx = RunContext::new(());
    assert_eq!(
        default_tool_error_function(&ctx, &AgentsError::user("x")),
        DEFAULT_ERROR
    );
    let _ = Arc::new(());
}
