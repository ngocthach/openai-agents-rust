//! `test_tool_batch_failure_history.py::test_input_guardrail_rejection_does_not_publish_partial_tools`.
//!
//! Python runs the turn's tools while a parallel input guardrail is still pending and withholds
//! their results when the verdict is a rejection. Rust settles the guardrail before it runs any
//! tool, so a rejected turn never produces a tool result at all; the observable contract is the
//! same: no partial tool history reaches the session, the result or the run state.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, RunOptions, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

#[tokio::test]
async fn input_guardrail_rejection_does_not_publish_partial_tools() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streaming in [false, true] {
        let runs = Arc::new(AtomicUsize::new(0));
        let completed = runs.clone();
        let failed = runs.clone();
        let completed_tool =
            FunctionTool::new("completed_tool", "completes", move |_c, _a: NoArgs| {
                let completed = completed.clone();
                async move {
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok("speculative output".to_string())
                }
            });
        let failed_tool = FunctionTool::new("failed_tool", "fails", move |_c, _a: NoArgs| {
            let failed = failed.clone();
            async move {
                failed.fetch_add(1, Ordering::SeqCst);
                Err::<String, _>(AgentsError::user("synthetic failure"))
            }
        })
        .without_failure_error_function();

        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            function_call_with_id("completed_tool", "{}", "done"),
            function_call_with_id("failed_tool", "{}", "failed"),
        ]);
        let agent = Agent::<()>::new("support")
            .with_tool(completed_tool)
            .with_tool(failed_tool)
            .with_input_guardrail(InputGuardrail::new("reject", |_c, _a, _i| async move {
                Ok(GuardrailFunctionOutput {
                    output_info: Value::String("blocked".into()),
                    tripwire_triggered: true,
                })
            }))
            .with_model_instance(model.clone());
        let session = Arc::new(InMemorySession::new("test"));

        let error = if streaming {
            let mut result = Runner::run_streamed_with_options(
                &agent,
                "go",
                RunOptions::new(()).session(session.clone()),
            );
            let mut seen = None;
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    if let Err(error) = event {
                        seen = Some(error);
                    }
                }
            }
            seen
        } else {
            Runner::run_with_options(&agent, "go", RunOptions::new(()).session(session.clone()))
                .await
                .err()
        };
        let error = error.unwrap_or_else(|| panic!("streaming={streaming}: the run must fail"));
        assert!(
            matches!(
                error,
                AgentsError::InputGuardrailTripwireTriggered { .. } | AgentsError::UserError { .. }
            ),
            "streaming={streaming}: {error:?}"
        );
        let history = session.get_items(None).await.unwrap();
        let shapes: Vec<String> = history
            .iter()
            .map(|item| {
                item.item_type()
                    .or_else(|| item.role())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect();
        assert_eq!(shapes, ["user"], "streaming={streaming}: {history:?}");
        assert!(
            !history
                .iter()
                .any(|item| item.as_value().to_string().contains("speculative output")),
            "streaming={streaming}"
        );
    }
}
