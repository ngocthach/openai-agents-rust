//! Port of the `filter` case of `test_failures_outside_recovery_boundary_propagate` from the
//! Python SDK's `tests/test_handoff_failure_error_function.py`: an input filter that fails ends
//! the run with its own error, and the handoff's failure formatter is never consulted for it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::handoffs::try_handoff_input_filter;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::{Agent, AgentsError, Runner, handoff};

#[tokio::test]
async fn a_failing_input_filter_propagates_without_the_failure_formatter() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        let formatter_calls = Arc::new(AtomicUsize::new(0));
        let calls = formatter_calls.clone();
        let next_agent =
            Arc::new(Agent::<()>::new("next").with_model_instance(Arc::new(FakeModel::new())));
        let transfer = handoff(next_agent)
            .with_on_handoff(|_ctx| async { Ok(()) })
            .with_input_filter_handler(try_handoff_input_filter(|_data| {
                Err(AgentsError::other("filter failed"))
            }))
            .with_failure_error_function(move |_ctx, _error| {
                calls.fetch_add(1, Ordering::SeqCst);
                "handled".to_string()
            });
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            &transfer.tool_name,
            "{}",
            "call",
        )]);
        let caller = Arc::new(
            Agent::<()>::new("caller")
                .with_model_instance(model)
                .with_handoff(transfer),
        );

        let error = if streamed {
            let mut stream =
                Runner::run_streamed_with_options(caller.clone(), "go", RunOptions::new(()));
            let mut error = None;
            {
                let mut events = std::pin::pin!(stream.stream_events());
                while let Some(event) = events.next().await {
                    if let Err(e) = event {
                        error = Some(e);
                    }
                }
            }
            error.expect("the filter failure surfaces")
        } else {
            Runner::run_with_options(caller.clone(), "go", RunOptions::new(()))
                .await
                .unwrap_err()
        };

        assert!(
            error.to_string().contains("filter failed"),
            "{streamed}: {error:?}"
        );
        assert_eq!(formatter_calls.load(Ordering::SeqCst), 0, "{streamed}");
    }
}
