//! Port of `test_stream_events_respects_asyncio_timeout_cancellation` from
//! `tests/test_cancel_streaming.py`: a consumer that gives up waiting for the next stream event
//! (a timeout around the read) gets control back promptly instead of waiting for the slow model.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, Runner};

#[tokio::test]
async fn a_timeout_around_the_next_event_returns_promptly() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::respond_async(|_call| async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(ScriptedStep::output(vec![assistant_message(
                "Final response",
            )]))
        })])
        .unwrap(),
    );
    let agent = Arc::new(Agent::<()>::new("TimeoutTester").with_model_instance(model));
    let mut result = Runner::run_streamed(agent, "Please tell me 5 jokes.");

    let started = Instant::now();
    {
        let mut events = std::pin::pin!(result.stream_events());
        // The run announces its agent at once; everything after it waits for the model.
        let first = tokio::time::timeout(Duration::from_millis(100), events.next()).await;
        assert!(first.is_ok(), "the agent-updated event arrives at once");
        let waited = tokio::time::timeout(Duration::from_millis(100), events.next()).await;
        assert!(waited.is_err(), "the slow model has produced no event yet");
    }

    assert!(
        started.elapsed() < Duration::from_millis(300),
        "{:?}",
        started.elapsed()
    );
    result.cancel(openai_agents_sdk::result::CancelMode::Immediate);
}
