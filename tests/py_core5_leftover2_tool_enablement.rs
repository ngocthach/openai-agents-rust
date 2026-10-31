//! Port of `test_agent_tool_enablement_snapshot.py`.
//!
//! Python's tests mutate `agent.tools` from inside an `is_enabled` callback, or replace it while
//! `get_all_tools` is suspended, to prove the runner works from a snapshot. A Rust `Agent` is
//! shared behind an `Arc` and an `is_enabled` callback only receives the agent name, so neither
//! mutation can happen; the snapshot guarantee is enforced by the type system. The tests below
//! keep the observable contracts: the enabled tool is returned as the original tool, the agent's
//! own tool list is untouched, and every run (and every turn) re-evaluates enablement before the
//! model sees the tools.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn tool(name: &'static str) -> FunctionTool<()> {
    FunctionTool::new(name, "a tool", move |_ctx, _args: NoArgs| async move {
        Ok(name.to_string())
    })
}

fn names(tools: &[Tool<()>]) -> Vec<String> {
    tools.iter().map(|t| t.name()).collect()
}

/// Python: `test_get_all_tools_keeps_enablement_with_original_tools_after_callback_reorders`
/// (the in-callback reorder is Python-only; the identity and untouched-list contract remains).
#[tokio::test]
async fn enablement_returns_the_original_enabled_tool_and_leaves_the_agent_untouched() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let allowed = tool("allowed").with_is_enabled_fn(move |_ctx, _agent| {
        counter.fetch_add(1, Ordering::SeqCst);
        async { true }
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_tool(allowed)
            .with_tool(tool("disabled").with_is_enabled(false)),
    );
    let resolved = agent
        .get_all_tools(&openai_agents_sdk::RunContext::new(()))
        .await
        .unwrap();
    assert_eq!(names(&resolved), ["allowed"]);
    assert_eq!(names(&agent.tools), ["allowed", "disabled"]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

fn two_turn_model() -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(
            (0..2)
                .map(|_| ScriptedStep::output(vec![assistant_message("done")]))
                .collect(),
        )
        .unwrap(),
    )
}

/// Python: `test_runner_exposes_evaluated_tools_when_enablement_changes_configuration`
/// (blocking and streamed). Enablement flips between runs; each run exposes exactly the tools
/// evaluated as enabled for it.
#[tokio::test]
async fn runner_exposes_the_tools_evaluated_for_each_run() {
    for streamed in [false, true] {
        let original_enabled = Arc::new(AtomicBool::new(true));
        let flag = original_enabled.clone();
        let original = tool("original").with_is_enabled_fn(move |_ctx, _agent| {
            let enabled = flag.load(Ordering::SeqCst);
            async move { enabled }
        });
        let model = two_turn_model();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(original)
            .with_tool(tool("disabled").with_is_enabled(false))
            .with_tool(tool("replacement"));

        for _ in 0..2 {
            if streamed {
                let mut stream =
                    Runner::run_streamed_with_options(agent.clone(), "hello", RunOptions::new(()));
                {
                    let events = stream.stream_events();
                    futures::pin_mut!(events);
                    while let Some(event) = events.next().await {
                        event.unwrap();
                    }
                }
                assert_eq!(stream.final_output_text().as_deref(), Some("done"));
            } else {
                let result = Runner::run(&agent, "hello").await.unwrap();
                assert_eq!(result.final_output_text().as_deref(), Some("done"));
            }
            original_enabled.store(false, Ordering::SeqCst);
        }

        let tool_names = |call: usize| -> Vec<String> {
            model.calls()[call]
                .request
                .tools
                .iter()
                .map(|spec| match spec {
                    openai_agents_sdk::models::interface::ToolSpec::Function(f) => f.name.clone(),
                    other => format!("{other:?}"),
                })
                .collect()
        };
        assert_eq!(tool_names(0), ["original", "replacement"]);
        assert_eq!(tool_names(1), ["replacement"]);
    }
}
