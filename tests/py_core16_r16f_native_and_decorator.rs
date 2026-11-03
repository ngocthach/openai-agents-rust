//! Ports from `tests/test_native_tool_failure_history.py` and
//! `tests/test_function_tool_decorator.py` that earlier rounds left Python-only.
//!
//! A Rust `custom_data_extractor` returns `Option<Value>` and cannot raise; the equivalent
//! failure at the same boundary is an extractor returning a non-object value.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::{CustomTool, FunctionTool, ToolTimeoutBehavior};
use openai_agents_sdk::{Agent, RunOptions, RunState, Runner};
use py_core6_resume_support::{round_trip, scripted};
use serde_json::{Value, json};

fn custom_call() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "custom_tool_call", "name": "native", "call_id": "native",
        "input": "synthetic",
    }))
}

fn outputs(state: &RunState) -> Vec<String> {
    state
        .generated_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => {
                Some(output.as_str().unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn approved_native_output_survives_custom_data_failure() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streaming in [false, true] {
        let effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = effects.clone();
        let native = CustomTool::<()>::new("native", "Synthetic native tool", move |_c, _i| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push("custom".into());
                Ok("completed".to_string())
            }
        })
        .with_needs_approval(true)
        .with_custom_data_extractor(|_ctx| async { Some(json!("synthetic extractor failure")) });
        let model = scripted(vec![vec![custom_call()], vec![assistant_message("done")]]);
        let agent = Agent::<()>::new("approved-native")
            .with_model_instance(model)
            .with_tool(native);

        let first = Runner::run(&agent, "go").await.unwrap();
        let mut state = first.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);

        let slot = RunCheckpoint::new();
        let options = RunOptions::new(()).checkpoint(&slot);
        let error = if streaming {
            let mut run = Runner::run_streamed_with_options(&agent, state.clone(), options);
            let mut failure = None;
            {
                let mut events = std::pin::pin!(run.stream_events());
                while let Some(event) = events.next().await {
                    if let Err(e) = event {
                        failure = Some(e);
                    }
                }
            }
            failure.expect("the extractor failure surfaces")
        } else {
            Runner::run_with_options(&agent, state.clone(), options)
                .await
                .expect_err("the extractor failure surfaces")
        };
        assert!(
            error
                .to_string()
                .contains("custom_data_extractor must return a JSON object"),
            "streaming={streaming}: {error}"
        );
        assert_eq!(*effects.lock().unwrap(), vec!["custom"]);
        let checkpoint = slot.take().expect("checkpoint");
        assert_eq!(
            outputs(&checkpoint),
            vec!["completed"],
            "streaming={streaming}"
        );

        // Recovery keeps the invocation checkpoint, not a live metadata-finalization flag.
        let serialized = checkpoint.to_json().unwrap();
        assert!(!serialized.to_string().contains("_custom_data_pending"));
        let restored = round_trip(&checkpoint);
        let resumed = Runner::run(&agent, restored).await.unwrap();
        assert_eq!(resumed.final_output, Some(Value::String("done".into())));
        assert_eq!(*effects.lock().unwrap(), vec!["custom"]);
    }
}

#[test]
fn decorator_timeout_configuration_is_fully_applied() {
    let tool = FunctionTool::<()>::new_no_args("timeout_configured_tool", "", |_c| async {
        Ok("ok".to_string())
    })
    .with_timeout(Duration::from_millis(1250))
    .with_timeout_behavior(ToolTimeoutBehavior::RaiseException)
    .with_timeout_error_function(|_ctx, error| format!("error_{error}"));
    assert_eq!(tool.timeout, Some(Duration::from_millis(1250)));
    assert_eq!(tool.timeout_behavior, ToolTimeoutBehavior::RaiseException);
    assert!(tool.timeout_error_function.is_some());
}
