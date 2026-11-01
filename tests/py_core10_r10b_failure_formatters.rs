//! Failure formatters (`failure_error_function`, `timeout_error_function`) that await or fail.
//!
//! Ports of the cases that were blocked while the Rust formatter was a synchronous closure
//! returning text:
//! - `tests/test_handoff_failure_error_function.py`: `async_formatter=True` of
//!   `test_custom_handoff_can_handle_or_reraise_application_error`, its re-raise half, and
//!   `failure="formatter"` of `test_failures_outside_recovery_boundary_propagate`.
//! - `tests/test_function_tool.py::test_async_failure_error_function_is_awaited`, plus the
//!   re-raise contract (a formatter that raises replaces the original failure) for function
//!   tools, agent tools, MCP tools and the timeout formatter, in direct and streamed runs.
//!
//! As in Python, a function tool, agent tool or MCP tool formatter that fails surfaces as
//! `Error running tool <name>: <formatter error>`; a handoff formatter's error is raised as is.
//!
//! Python-only and not ported: `failure="cancel"` (asyncio `CancelledError`) and
//! `failure="filter"` (the Rust handoff input filter returns a plain value and cannot fail).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::{
    FunctionTool, ToolOutput, ToolTimeoutBehavior, async_tool_error_fn, try_tool_error_fn,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn failing_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new(name, "always fails", |_ctx, _a: NoArgs| async {
        Err::<String, _>(AgentsError::user("kapow"))
    })
}

/// Runs `agent` directly or streamed and returns the outcome.
async fn run(agent: &Arc<Agent<()>>, streamed: bool) -> Result<RunResult<()>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), "go", RunOptions::new(())).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), "go", RunOptions::new(()));
    let mut error = None;
    {
        let mut pinned = std::pin::pin!(stream.stream_events());
        while let Some(event) = pinned.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

fn function_outputs(model: &FakeModel, call_id_prefix: &str) -> Vec<Value> {
    let calls = model.calls();
    let last = calls.last().expect("a model call");
    last.input
        .iter()
        .filter(|i| {
            i.0["type"] == "function_call_output"
                && i.0["call_id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with(call_id_prefix))
        })
        .map(|i| i.0["output"].clone())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Handoffs
// ---------------------------------------------------------------------------------------------

fn failing_handoff(
    target: &Arc<Agent<()>>,
    handler: openai_agents_sdk::tool::ToolErrorFn<()>,
) -> openai_agents_sdk::handoffs::Handoff<()> {
    handoff(target.clone())
        .with_on_handoff(|_ctx| async { Err(AgentsError::user("invocation failed")) })
        .with_failure_error_handler(handler)
}

/// Python: `test_custom_handoff_can_handle_or_reraise_application_error[async_formatter=True]`.
#[tokio::test]
async fn an_async_handoff_formatter_handles_the_failure() {
    for streamed in [false, true] {
        let model = setup();
        let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let recorder = seen.clone();
        let transfer = failing_handoff(
            &target,
            async_tool_error_fn(move |_ctx, error| {
                let text = error.to_string();
                let recorder = recorder.clone();
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    recorder.lock().unwrap().push(text);
                    Ok("Transfer unavailable; answer here.".to_string())
                })
            }),
        );
        let caller = Arc::new(
            Agent::<()>::new("caller")
                .with_handoff(transfer)
                .with_model_instance(model.clone()),
        );
        model.add_turn(vec![function_call("transfer_to_target", "{}")]);
        model.add_turn(vec![assistant_message("caller answer")]);

        let result = run(&caller, streamed).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("caller answer"));
        assert_eq!(result.last_agent.name, "caller");
        assert_eq!(*seen.lock().unwrap(), ["invocation failed"]);
        assert_eq!(
            function_outputs(&model, ""),
            [json!("Transfer unavailable; answer here.")],
            "streamed={streamed}"
        );
        assert!(
            result
                .new_items
                .iter()
                .all(|i| i.item_type() != "handoff_output_item")
        );
    }
}

/// Python: the re-raise half of `test_custom_handoff_can_handle_or_reraise_application_error`
/// and `test_failures_outside_recovery_boundary_propagate[formatter]`: a formatter that fails
/// ends the run with its own error, and is called exactly once.
#[tokio::test]
async fn a_failing_handoff_formatter_ends_the_run_with_its_error() {
    for streamed in [false, true] {
        for asynchronous in [false, true] {
            let model = setup();
            let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = calls.clone();
            let handler = if asynchronous {
                async_tool_error_fn(move |_ctx, _error| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async {
                        tokio::task::yield_now().await;
                        Err(AgentsError::other("formatter failed"))
                    })
                })
            } else {
                try_tool_error_fn(move |_ctx, error| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    // Re-raising: the formatter hands the caught error back as its own failure.
                    Err(AgentsError::other(format!("formatter failed: {error}")))
                })
            };
            let caller = Arc::new(
                Agent::<()>::new("caller")
                    .with_handoff(failing_handoff(&target, handler))
                    .with_model_instance(model.clone()),
            );
            model.add_turn(vec![function_call("transfer_to_target", "{}")]);
            model.add_turn(vec![assistant_message("unreachable")]);

            let error = run(&caller, streamed).await.unwrap_err();
            assert!(
                error.to_string().contains("formatter failed"),
                "streamed={streamed} async={asynchronous}: {error}"
            );
            assert!(matches!(error, AgentsError::Other(_)), "{error:?}");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                model.calls().len(),
                1,
                "the run stopped before the next turn"
            );
        }
    }
}

/// An `on_handoff` that succeeds never reaches the formatter.
#[tokio::test]
async fn the_handoff_formatter_is_not_called_when_the_handoff_succeeds() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let transfer = handoff(target.clone())
        .with_on_handoff(|_ctx| async { Ok(()) })
        .with_failure_error_handler(try_tool_error_fn(move |_ctx, _error| {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(AgentsError::other("must not run"))
        }));
    let caller = Arc::new(
        Agent::<()>::new("caller")
            .with_handoff(transfer)
            .with_model_instance(model.clone()),
    );
    model.add_turn(vec![function_call("transfer_to_target", "{}")]);
    model.add_turn(vec![assistant_message("target answer")]);
    let result = run(&caller, false).await.unwrap();
    assert_eq!(result.last_agent.name, "target");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------------------------
// Function tools
// ---------------------------------------------------------------------------------------------

/// Python: `test_async_failure_error_function_is_awaited`.
#[tokio::test]
async fn an_async_failure_formatter_is_awaited_by_invoke() {
    let tool =
        failing_tool("boom").with_failure_error_handler(async_tool_error_fn(|_ctx, error| {
            let text = format!("handled:{error}");
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(text)
            })
        }));
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "boom", "{}");
    match tool.invoke(ctx, "{}".into()).await.unwrap() {
        ToolOutput::Text(text) => assert_eq!(text, "handled:kapow"),
        other => panic!("unexpected {other:?}"),
    }
}

/// A formatter error replaces the tool's own error when the tool is invoked directly.
#[tokio::test]
async fn a_failing_formatter_error_propagates_from_invoke() {
    let tool = failing_tool("boom").with_failure_error_handler(try_tool_error_fn(|_ctx, _e| {
        Err(AgentsError::other("formatter failed"))
    }));
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "boom", "{}");
    let error = tool.invoke(ctx, "{}".into()).await.unwrap_err();
    assert_eq!(error.to_string(), "formatter failed");
}

#[tokio::test]
async fn an_async_failure_formatter_reaches_the_model_in_runs() {
    for streamed in [false, true] {
        let model = setup();
        let tool =
            failing_tool("boom").with_failure_error_handler(async_tool_error_fn(|_ctx, error| {
                let text = format!("handled:{error}");
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    Ok(text)
                })
            }));
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_tool(tool)
                .with_model_instance(model.clone()),
        );
        model.add_turn(vec![function_call("boom", "{}")]);
        model.add_turn(vec![assistant_message("done")]);
        let result = run(&agent, streamed).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(function_outputs(&model, ""), [json!("handled:kapow")]);
        let item = result
            .new_items
            .iter()
            .find_map(|i| match &i.kind {
                RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("a tool output item");
        assert_eq!(item, json!("handled:kapow"));
    }
}

/// A formatter that raises ends the run with the formatter's error (the original failure is
/// replaced), in direct and streamed runs, for plain and async formatters.
#[tokio::test]
async fn a_failing_failure_formatter_ends_the_run() {
    for streamed in [false, true] {
        for asynchronous in [false, true] {
            let model = setup();
            let handler = if asynchronous {
                async_tool_error_fn(|_ctx, _e| {
                    Box::pin(async {
                        tokio::task::yield_now().await;
                        Err(AgentsError::other("formatter failed"))
                    })
                })
            } else {
                try_tool_error_fn(|_ctx, _e| Err(AgentsError::other("formatter failed")))
            };
            let agent = Arc::new(
                Agent::<()>::new("agent")
                    .with_tool(failing_tool("boom").with_failure_error_handler(handler))
                    .with_model_instance(model.clone()),
            );
            model.add_turn(vec![function_call("boom", "{}")]);
            model.add_turn(vec![assistant_message("unreachable")]);
            let error = run(&agent, streamed).await.unwrap_err();
            assert_eq!(
                error.to_string(),
                "Error running tool boom: formatter failed",
                "streamed={streamed} async={asynchronous}"
            );
            assert_eq!(model.calls().len(), 1);
        }
    }
}

fn slow_tool() -> FunctionTool<()> {
    FunctionTool::new("slow", "slow", |_ctx, _a: NoArgs| async {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok("done".to_string())
    })
    .with_timeout(Duration::from_millis(10))
    .with_timeout_behavior(ToolTimeoutBehavior::ErrorAsResult)
}

/// An async `timeout_error_function` produces the timeout message.
#[tokio::test]
async fn an_async_timeout_formatter_is_awaited() {
    for streamed in [false, true] {
        let model = setup();
        let tool = slow_tool().with_timeout_error_handler(async_tool_error_fn(|_ctx, error| {
            let text = match error {
                AgentsError::ToolTimeoutError {
                    tool_name,
                    timeout_seconds,
                    ..
                } => format!("async-timeout:{tool_name}:{timeout_seconds}"),
                other => panic!("unexpected {other:?}"),
            };
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(text)
            })
        }));
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_tool(tool)
                .with_model_instance(model.clone()),
        );
        model.add_turn(vec![function_call("slow", "{}")]);
        model.add_turn(vec![assistant_message("done")]);
        run(&agent, streamed).await.unwrap();
        assert_eq!(
            function_outputs(&model, ""),
            [json!("async-timeout:slow:0.01")]
        );
    }
}

/// A timeout formatter that raises ends the run with its error.
#[tokio::test]
async fn a_failing_timeout_formatter_ends_the_run() {
    for streamed in [false, true] {
        let model = setup();
        let tool = slow_tool().with_timeout_error_handler(try_tool_error_fn(|_ctx, _e| {
            Err(AgentsError::other("timeout formatter failed"))
        }));
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_tool(tool)
                .with_model_instance(model.clone()),
        );
        model.add_turn(vec![function_call("slow", "{}")]);
        model.add_turn(vec![assistant_message("unreachable")]);
        let error = run(&agent, streamed).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Error running tool slow: timeout formatter failed"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Agent tools
// ---------------------------------------------------------------------------------------------

fn agent_tool_orchestrator(
    model: &Arc<FakeModel>,
    options: AgentAsToolOptions<()>,
) -> Arc<Agent<()>> {
    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    Arc::new(
        Agent::<()>::new("orchestrator")
            .with_tool(inner.as_tool_with(options.tool_name("nested").tool_description("nested")))
            .with_model_instance(model.clone()),
    )
}

#[tokio::test]
async fn an_async_agent_tool_formatter_handles_the_nested_failure() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call("nested", r#"{"input":"hello"}"#)]);
        model.add_error("nested boom");
        model.add_turn(vec![assistant_message("outer done")]);
        let options = AgentAsToolOptions::new().failure_error_handler(async_tool_error_fn(
            |_ctx: &RunContext<()>, error| {
                let text = format!("handled: {error}");
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    Ok(text)
                })
            },
        ));
        let agent = agent_tool_orchestrator(&model, options);
        let result = run(&agent, streamed).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("outer done"));
        let output = function_outputs(&model, "");
        assert_eq!(output.len(), 1);
        assert!(
            output[0].as_str().unwrap().starts_with("handled:"),
            "{output:?}"
        );
    }
}

#[tokio::test]
async fn a_failing_agent_tool_formatter_fails_the_outer_run() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call("nested", r#"{"input":"hello"}"#)]);
        model.add_error("nested boom");
        let options = AgentAsToolOptions::new().failure_error_handler(try_tool_error_fn(
            |_ctx: &RunContext<()>, _e| Err(AgentsError::other("agent formatter failed")),
        ));
        let agent = agent_tool_orchestrator(&model, options);
        let error = run(&agent, streamed).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Error running tool nested: agent formatter failed"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// MCP tools
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "mcp")]
mod mcp {
    use super::*;
    use async_trait::async_trait;
    use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
    use openai_agents_sdk::{MCPConfig, Result};
    use serde_json::{Map, Value};

    /// A server whose only tool always fails.
    #[derive(Debug, Default)]
    struct FailingServer {
        config: MCPServerConfig,
    }

    #[async_trait]
    impl MCPServer for FailingServer {
        fn name(&self) -> &str {
            "failing"
        }
        fn config(&self) -> &MCPServerConfig {
            &self.config
        }
        async fn connect(&self) -> Result<()> {
            Ok(())
        }
        async fn cleanup(&self) -> Result<()> {
            Ok(())
        }
        async fn list_tools(&self) -> Result<Vec<MCPTool>> {
            Ok(vec![MCPTool {
                meta: None,
                name: "explode".into(),
                title: None,
                description: Some("Always fails".into()),
                input_schema: json!({"type": "object", "properties": {}}),
                annotations: None,
            }])
        }
        async fn call_tool(
            &self,
            _tool_name: &str,
            _arguments: Option<Map<String, Value>>,
        ) -> Result<Value> {
            Err(AgentsError::other("upstream exploded"))
        }
        async fn list_prompts(&self) -> Result<Vec<Value>> {
            Ok(Vec::new())
        }
        async fn get_prompt(
            &self,
            _name: &str,
            _arguments: Option<std::collections::BTreeMap<String, String>>,
        ) -> Result<Value> {
            Err(AgentsError::user("no prompts"))
        }
    }

    fn mcp_agent(config: MCPConfig<()>, model: &Arc<FakeModel>) -> Arc<Agent<()>> {
        Arc::new(
            Agent::<()>::new("Ops")
                .with_mcp_server(Arc::new(FailingServer::default()))
                .with_mcp_config(config)
                .with_model_instance(model.clone()),
        )
    }

    #[tokio::test]
    async fn an_async_mcp_formatter_handles_the_failure() {
        for streamed in [false, true] {
            let model = setup();
            model.add_turn(vec![function_call("explode", "{}")]);
            model.add_turn(vec![assistant_message("recovered")]);
            let config = MCPConfig::<()>::default().with_failure_error_handler(
                async_tool_error_fn(|_ctx, error| {
                    let text = format!("mcp handled:{error}");
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        Ok(text)
                    })
                }),
            );
            let result = run(&mcp_agent(config, &model), streamed).await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
            let output = function_outputs(&model, "");
            assert_eq!(output.len(), 1);
            assert!(
                output[0].as_str().unwrap().starts_with("mcp handled:"),
                "{output:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_failing_mcp_formatter_ends_the_run() {
        for streamed in [false, true] {
            let model = setup();
            model.add_turn(vec![function_call("explode", "{}")]);
            let config = MCPConfig::<()>::default().with_failure_error_handler(try_tool_error_fn(
                |_ctx, _e| Err(AgentsError::other("mcp formatter failed")),
            ));
            let error = run(&mcp_agent(config, &model), streamed).await.unwrap_err();
            assert_eq!(
                error.to_string(),
                "Error running tool explode: mcp formatter failed"
            );
        }
    }
}
