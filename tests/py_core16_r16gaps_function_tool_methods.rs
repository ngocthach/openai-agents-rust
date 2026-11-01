//! Port of `test_instance_method`, `test_classmethod` and the `Bar` fixtures of
//! `tests/test_doc_parsing.py`: `#[function_tool]` also works inside an `impl` block. An
//! associated function (Python's classmethod) gets a `<fn>_tool()` constructor; a method taking
//! `&self` gets `<fn>_tool(&self)`, which binds a clone of the receiver.
#![cfg(feature = "macros")]

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use openai_agents_sdk::function_tool;
use openai_agents_sdk::items::FunctionCall;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool::ToolOutput;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Result, RunContext};
use serde_json::json;

#[derive(Clone, Default)]
struct Bar {
    calls: Arc<AtomicI64>,
}

impl Bar {
    /// This is func_bar.
    ///
    /// Args:
    ///     a: The first argument.
    ///     b: The second argument.
    ///
    /// Returns:
    ///     A result
    #[function_tool]
    fn func_bar(&self, a: i64, b: f64) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(format!("{a} {b}"))
    }

    /// This is func_baz.
    ///
    /// Args:
    ///     a: The first argument.
    ///     b: The second argument.
    ///
    /// Returns:
    ///     A result
    #[function_tool(associated)]
    async fn func_baz(a: i64, b: f64) -> String {
        format!("{a}/{b}")
    }
}

async fn invoke(tool: &FunctionTool<()>, arguments: &str) -> ToolOutput {
    let mut call = FunctionCall::new(tool.name.clone(), arguments);
    call.call_id = "call_1".into();
    let ctx = ToolContext::from_call(RunContext::new(()), "agent", &call);
    tool.invoke(ctx, arguments.to_string()).await.unwrap()
}

fn assert_documented(tool: &FunctionTool<()>, name: &str, description: &str) {
    assert_eq!(tool.name, name);
    assert_eq!(tool.description, description);
    let properties = &tool.params_json_schema["properties"];
    assert_eq!(properties["a"]["description"], json!("The first argument."));
    assert_eq!(
        properties["b"]["description"],
        json!("The second argument.")
    );
    assert!(properties.get("self").is_none());
}

/// Python: `test_instance_method`.
#[tokio::test]
async fn an_instance_method_becomes_a_tool_bound_to_its_receiver() {
    let bar = Bar::default();
    let tool: FunctionTool<()> = bar.func_bar_tool();
    assert_documented(&tool, "func_bar", "This is func_bar.");

    let output = invoke(&tool, r#"{"a": 1, "b": 2.0}"#).await;
    assert_eq!(output, ToolOutput::Text("1 2".into()));
    assert_eq!(bar.calls.load(Ordering::SeqCst), 1);
}

/// Python: `test_classmethod`.
#[tokio::test]
async fn an_associated_function_becomes_a_tool() {
    let tool: FunctionTool<()> = Bar::func_baz_tool();
    assert_documented(&tool, "func_baz", "This is func_baz.");
    let output = invoke(&tool, r#"{"a": 3, "b": 4.5}"#).await;
    assert_eq!(output, ToolOutput::Text("3/4.5".into()));
}
