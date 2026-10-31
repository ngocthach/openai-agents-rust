//! `test_tool_use_behavior.py` cases not carried by the first port: the custom-behavior return
//! contract in streamed runs, and `stop_at_tool_names` for namespaced tools. Python reaches
//! `run_loop.check_for_final_output_from_tools`; here the same turns run through `Runner`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call, function_call_full};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn echo_tool(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: Empty| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

/// `test_custom_tool_use_behavior_return_contract` (valid results; a wrong return type is a
/// compile error in Rust), non-streamed and streamed.
#[tokio::test]
async fn custom_tool_use_behavior_return_contract() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call("ping", "{}")]);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let agent = Agent::<()>::new("test")
            .with_tool(echo_tool("ping", "pong"))
            .with_tool_use_behavior(ToolUseBehavior::custom(move |_ctx, results| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(results.len(), 1);
                    assert_eq!(results[0].output.to_model_string(), "pong");
                    Ok(ToolsToFinalOutputResult {
                        is_final_output: true,
                        final_output: Some(json!("done")),
                    })
                }
            }))
            .with_model_instance(model.clone());
        let output = if streamed {
            let mut result = Runner::run_streamed(&agent, "test");
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            result.final_output_text()
        } else {
            Runner::run(&agent, "test")
                .await
                .unwrap()
                .final_output_text()
        };
        assert_eq!(output.as_deref(), Some("done"), "streamed={streamed}");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "streamed={streamed}");
        assert_eq!(model.calls().len(), 1, "streamed={streamed}");
    }
}

/// `test_stop_at_tool_names_supports_public_and_qualified_names_for_namespaced_tools`.
#[tokio::test]
async fn stop_at_tool_names_supports_public_and_qualified_names_for_namespaced_tools() {
    for stop_name in ["lookup_account", "billing.lookup_account"] {
        let model = setup();
        model.add_turn(vec![function_call_full(
            "lookup_account",
            "{}",
            "call-1",
            None,
            Some("billing".into()),
        )]);
        model.add_turn(vec![assistant_message("kept going")]);
        let tools = tool_namespace(
            "billing",
            "Billing tools",
            vec![echo_tool("lookup_account", "billing-output")],
        )
        .unwrap();
        let agent = Agent::<()>::new("test")
            .with_tools(tools)
            .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                stop_at_tool_names: vec![stop_name.into()],
            }))
            .with_model_instance(model.clone());
        let result = Runner::run(&agent, "x").await.unwrap();
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("billing-output"),
            "{stop_name}"
        );
        assert_eq!(model.calls().len(), 1, "{stop_name}");
    }
}
