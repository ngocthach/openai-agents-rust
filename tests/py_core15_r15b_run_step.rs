//! Ported from the Python SDK's `tests/test_run_step_execution.py`: a failing function tool in the
//! same turn as a local shell call must not drop the shell output (Python raises
//! `asyncio.CancelledError` inside the tool; Rust has no cancellation outcome, so the failure
//! path that yields the same default message is used).

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, ShellTool};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};
use std::sync::Arc;

#[tokio::test]
async fn mixed_tool_calls_preserve_shell_output_when_function_tool_fails() {
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![
            function_call_full("failing_tool", "{}", "fn-1", None, None),
            OutputItem::from_value(
                json!({"type": "shell_call", "id": "sh-1", "call_id": "shell-1",
                "status": "completed",
                "action": {"commands": ["echo hi"], "timeout_ms": 1000}}),
            ),
        ]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let failing = FunctionTool::new("failing_tool", "", |_c, _a: Value| async move {
        Err::<String, _>(openai_agents_sdk::AgentsError::user("boom"))
    });
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(failing)
        .with_tool(ShellTool::new(|_req| async move { Ok("shell ok") }));
    let result = Runner::run(&agent, "hello").await.unwrap();
    assert_eq!(model.calls().len(), 2);
    let turn = &result.new_items[..result.new_items.len() - 1];
    assert_eq!(turn.len(), 4, "{turn:#?}");
    match &turn[2].kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => {
            assert_eq!(raw_item["type"], "function_call_output");
            assert_eq!(
                raw_item["output"],
                "An error occurred while running the tool. Please try again."
            );
        }
        other => panic!("expected function output, got {other:?}"),
    }
    match &turn[3].kind {
        RunItemKind::ToolCallOutput {
            raw_item, output, ..
        } => {
            assert_eq!(raw_item["type"], "shell_call_output");
            assert_eq!(output, &json!("shell ok"));
        }
        other => panic!("expected shell output, got {other:?}"),
    }
}
