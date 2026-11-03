//! Port of the context rows of `test_class_based_functions` in `tests/test_function_schema.py`:
//! a method (`self`) and an associated function (`classmethod` / `staticmethod`) whose first
//! parameter is the tool context both become tools that receive the context, publish a schema
//! without it, and apply the default of an omitted argument.

#![cfg(feature = "macros")]

use openai_agents_sdk::function_tool;
use openai_agents_sdk::items::FunctionCall;
use openai_agents_sdk::tool::{FunctionTool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Result, RunContext};
use serde_json::json;

#[derive(Clone, Default)]
struct Adder;

impl Adder {
    #[function_tool]
    fn foo_ctx(&self, ctx: ToolContext<String>, a: i64, b: Option<i64>) -> Result<String> {
        Ok(format!("{}:{}", ctx.context(), a + b.unwrap_or(5)))
    }

    #[function_tool(associated)]
    fn baz_ctx(ctx: ToolContext<String>, a: i64, b: Option<i64>) -> Result<String> {
        Ok(format!("{}:{}", ctx.context(), a + b.unwrap_or(5)))
    }
}

async fn invoke(tool: &FunctionTool<String>, arguments: &str) -> ToolOutput {
    let mut call = FunctionCall::new(tool.name.clone(), arguments);
    call.call_id = "call_1".into();
    let ctx = ToolContext::from_call(RunContext::new("test".to_string()), "agent", &call);
    tool.invoke(ctx, arguments.to_string()).await.unwrap()
}

fn assert_context_is_not_a_parameter(tool: &FunctionTool<String>) {
    let properties = tool.params_json_schema["properties"]
        .as_object()
        .expect("properties");
    let mut names: Vec<&str> = properties.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["a", "b"]);
}

#[tokio::test]
async fn a_method_taking_the_tool_context_receives_it() {
    let tool: FunctionTool<String> = Adder.foo_ctx_tool();
    assert_eq!(tool.name, "foo_ctx");
    assert_context_is_not_a_parameter(&tool);
    assert_eq!(
        invoke(&tool, r#"{"a": 1, "b": 2}"#).await,
        ToolOutput::Text("test:3".into())
    );
    assert_eq!(
        invoke(&tool, &json!({"a": 1}).to_string()).await,
        ToolOutput::Text("test:6".into())
    );
}

#[tokio::test]
async fn an_associated_function_taking_the_tool_context_receives_it() {
    let tool: FunctionTool<String> = Adder::baz_ctx_tool();
    assert_eq!(tool.name, "baz_ctx");
    assert_context_is_not_a_parameter(&tool);
    assert_eq!(
        invoke(&tool, r#"{"a": 1, "b": 2}"#).await,
        ToolOutput::Text("test:3".into())
    );
    assert_eq!(
        invoke(&tool, &json!({"a": 1}).to_string()).await,
        ToolOutput::Text("test:6".into())
    );
}
