//! Ported from the Python SDK's `tests/test_stream_input_guardrail_timing.py`: input guardrails
//! that finish before, during or after a streamed run, and the tripwire and max-turns interplay.
//! The two span-ordering tests live in `py_core3_guardrails_tracing.rs`, which runs in its own
//! process because it captures the global trace processor.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::run_error_handlers::RunErrorHandlers;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, InputGuardrail, RunOptions, Runner, StreamEvent};
use serde_json::json;
use tokio::sync::Notify;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// Mirrors `make_input_guardrail`: completes after `delay`, tripping when `trip` is set.
fn make_input_guardrail(delay: Duration, trip: bool) -> InputGuardrail<()> {
    let name = if trip {
        "tripping_input_guardrail"
    } else {
        "delayed_input_guardrail"
    };
    InputGuardrail::new(name, move |_c, _a, _i| async move {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Ok(GuardrailFunctionOutput::new(
            json!({ "delay": delay.as_secs_f64() }),
            trip,
        ))
    })
}

/// The kind of a stream event, as the Python tests compare `event.type`.
fn event_kind(event: &StreamEvent<()>) -> &'static str {
    match event {
        StreamEvent::RawResponse(_) => "raw_response_event",
        StreamEvent::RunItem { .. } => "run_item_stream_event",
        StreamEvent::AgentUpdated { .. } => "agent_updated_stream_event",
    }
}

/// Drains a streamed run, returning the event kinds in order and the run's error, if any.
async fn collect_events(
    result: &mut openai_agents_sdk::RunResultStreaming<()>,
) -> (Vec<&'static str>, Option<AgentsError>) {
    let mut kinds = Vec::new();
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        match event {
            Ok(event) => kinds.push(event_kind(&event)),
            Err(error) => return (kinds, Some(error)),
        }
    }
    (kinds, None)
}

fn single_text_model(text: &str) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message(text)]))
        .unwrap();
    model
}

#[tokio::test]
async fn input_guardrail_results_follow_completion_order() {
    disable_tracing();
    let fast = InputGuardrail::new("fast_guardrail", |_c, _a, _i| async move {
        tokio::task::yield_now().await;
        Ok(GuardrailFunctionOutput::new(json!({ "delay": 0.0 }), false))
    });
    let slow = InputGuardrail::new("slow_guardrail", |_c, _a, _i| async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(GuardrailFunctionOutput::new(
            json!({ "delay": 0.005 }),
            false,
        ))
    });
    let agent = Agent::<()>::new("TimingAgentOrder")
        .with_model_instance(single_text_model("Final response"))
        .with_input_guardrail(slow)
        .with_input_guardrail(fast);

    let mut result = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    let (_, error) = collect_events(&mut result).await;
    assert!(error.is_none(), "{error:?}");

    let finished = result.into_result().expect("completed result");
    let delays: Vec<f64> = finished
        .input_guardrail_results
        .iter()
        .map(|r| r.output.output_info["delay"].as_f64().unwrap())
        .collect();
    assert_eq!(delays, vec![0.0, 0.005]);
}

async fn timing_is_consistent(guardrail_delay: Duration) {
    disable_tracing();
    let agent = Agent::<()>::new("TimingAgent")
        .with_model_instance(single_text_model("Final response"))
        .with_input_guardrail(make_input_guardrail(guardrail_delay, false));

    let mut result = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    let (kinds, error) = collect_events(&mut result).await;
    assert!(error.is_none(), "{error:?}");

    let finished = result.into_result().expect("completed result");
    assert_eq!(finished.input_guardrail_results.len(), 1);
    assert_eq!(
        finished.input_guardrail_results[0].guardrail_name,
        "delayed_input_guardrail"
    );
    assert!(
        !finished.input_guardrail_results[0]
            .output
            .tripwire_triggered
    );
    assert_eq!(
        finished.final_output_text().as_deref(),
        Some("Final response")
    );

    assert!(kinds.len() >= 3, "Unexpectedly few events: {kinds:?}");
    assert_eq!(kinds[0], "agent_updated_stream_event");
    assert!(kinds.contains(&"raw_response_event"));
}

#[tokio::test]
async fn run_streamed_input_guardrail_timing_is_consistent_without_delay() {
    timing_is_consistent(Duration::ZERO).await;
}

#[tokio::test]
async fn run_streamed_input_guardrail_timing_is_consistent_with_delay() {
    timing_is_consistent(Duration::from_millis(20)).await;
}

async fn event_sequence(guardrail_delay: Duration) -> Vec<&'static str> {
    let agent = Agent::<()>::new("TimingAgent")
        .with_model_instance(single_text_model("Final response"))
        .with_input_guardrail(make_input_guardrail(guardrail_delay, false));
    let mut result = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    let (kinds, error) = collect_events(&mut result).await;
    assert!(error.is_none(), "{error:?}");
    kinds
}

#[tokio::test]
async fn run_streamed_input_guardrail_sequences_match_between_fast_and_slow() {
    disable_tracing();
    let fast = event_sequence(Duration::ZERO).await;
    let slow = event_sequence(Duration::from_millis(20)).await;
    assert_eq!(
        fast, slow,
        "Event sequences differ between guardrail timings:\nfast={fast:?}\nslow={slow:?}"
    );
}

async fn tripwire_raises(guardrail_delay: Duration) {
    disable_tracing();
    let agent = Agent::<()>::new("TimingAgentTrip")
        .with_model_instance(single_text_model("Final response"))
        .with_input_guardrail(make_input_guardrail(guardrail_delay, true));

    let mut result = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    let (_, error) = collect_events(&mut result).await;
    let error = error.expect("tripwire");

    let AgentsError::InputGuardrailTripwireTriggered {
        guardrail_result, ..
    } = &error
    else {
        panic!("expected a tripwire, got {error:?}");
    };
    assert!(guardrail_result.output.tripwire_triggered);
    let run_data = error.run_data().expect("run data");
    assert_eq!(run_data.input_guardrail_results.len(), 1);
    assert_eq!(
        run_data.input_guardrail_results[0].guardrail_name,
        "tripping_input_guardrail"
    );
}

#[tokio::test]
async fn run_streamed_input_guardrail_tripwire_raises_without_delay() {
    tripwire_raises(Duration::ZERO).await;
}

#[tokio::test]
async fn run_streamed_input_guardrail_tripwire_raises_with_delay() {
    tripwire_raises(Duration::from_millis(20)).await;
}

#[tokio::test]
async fn max_turns_does_not_clobber_input_guardrail_tripwire() {
    disable_tracing();
    // The guardrail waits for the max-turns handler, which runs only once the run loop has
    // established that the turn limit is exceeded. Waiting on a signal rather than a delay keeps
    // the test a real regression test: the trip can only resolve after the limit is reached.
    let max_turns_reached = Arc::new(Notify::new());

    let guardrail = {
        let reached = max_turns_reached.clone();
        InputGuardrail::new("trip", move |_c, _a, _i| {
            let reached = reached.clone();
            async move {
                reached.notified().await;
                Ok(GuardrailFunctionOutput::new(
                    json!({ "reason": "blocked" }),
                    true,
                ))
            }
        })
    };

    let echo = FunctionTool::new(
        "some_function",
        "",
        |_c, _a: serde_json::Value| async move { Ok("result".to_string()) },
    );
    let model = Arc::new(ScriptedModel::new());
    for i in 1..10 {
        model
            .enqueue(ScriptedStep::output(vec![
                assistant_message(i.to_string()),
                function_call("some_function", json!({"a": "b"}).to_string()),
            ]))
            .unwrap();
    }
    let agent = Agent::<()>::new("MaxTurnsGuardrailAgent")
        .with_model_instance(model)
        .with_tool(echo)
        .with_input_guardrail(guardrail);

    // Declining (returning `None`) keeps the default max-turns behaviour; the handler only
    // signals that the limit has been reached.
    let handlers = RunErrorHandlers::<()>::default().on_max_turns({
        let reached = max_turns_reached.clone();
        move |_input| {
            let reached = reached.clone();
            async move {
                reached.notify_one();
                Ok(None)
            }
        }
    });
    let options = RunOptions::new(())
        .max_turns(Some(1))
        .error_handlers(handlers);

    let mut result = Runner::run_streamed_with_options(&agent, "user_message", options);
    let (_, error) = collect_events(&mut result).await;
    let error = error.expect("run must fail");

    assert!(
        matches!(error, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "Expected InputGuardrailTripwireTriggered, got {error:?}"
    );
}
