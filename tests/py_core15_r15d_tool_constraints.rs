//! Ports of the constraint-enforcement cases of `tests/test_function_schema.py`
//! (`test_annotated_constraints_reject_arguments_before_tool_execution`): a value that breaks
//! a constraint the published schema declares never reaches the tool body.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::RunContext;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Args {
    #[schemars(range(min = 1, max = 10))]
    quantity: i64,
    #[schemars(length(min = 2, max = 5))]
    code: String,
}

fn tool(seen: Arc<Mutex<Vec<i64>>>) -> FunctionTool<()> {
    FunctionTool::new("update_inventory", "d", move |_ctx, args: Args| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(args.quantity);
            Ok(args.quantity.to_string())
        }
    })
}

async fn invoke(tool: &FunctionTool<()>, arguments: &str) -> Result<String, String> {
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "update_inventory", arguments);
    tool.invoke(ctx, arguments.to_string())
        .await
        .map(|out| out.to_model_string())
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn constraint_violations_are_rejected_before_the_body_runs() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let tool = tool(seen.clone());
    assert_eq!(
        invoke(&tool, r#"{"quantity":5,"code":"abc"}"#).await,
        Ok("5".to_string())
    );
    for invalid in [
        r#"{"quantity":0,"code":"abc"}"#,
        r#"{"quantity":11,"code":"abc"}"#,
        r#"{"quantity":5,"code":"a"}"#,
        r#"{"quantity":5,"code":"abcdef"}"#,
    ] {
        let result = invoke(&tool, invalid).await;
        assert!(result.is_err(), "{invalid} was accepted: {result:?}");
    }
    assert_eq!(*seen.lock().unwrap(), vec![5]);
}
