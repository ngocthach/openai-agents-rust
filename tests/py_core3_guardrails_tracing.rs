//! Ported from the Python SDK's `tests/test_stream_input_guardrail_timing.py`: the agent span and
//! the trace finish after a slow input guardrail or a slow model. This binary installs a global
//! trace processor, so it holds both scenarios in one test to avoid racing itself.

use std::sync::Arc;
use std::time::Duration;

use chrono::DateTime;
use futures::StreamExt;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::models::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use std::sync::Mutex;

use openai_agents_sdk::tracing::{Span, Trace, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, InputGuardrail, Result, RunOptions, Runner};
use serde_json::{Value, json};

/// Delays every model request by `delay`, the Python `SlowCompleteScriptedModel`'s effect.
#[derive(Debug)]
struct SlowModel {
    inner: Arc<ScriptedModel>,
    delay: Duration,
}

#[async_trait::async_trait]
impl Model for SlowModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        tokio::time::sleep(self.delay).await;
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        tokio::time::sleep(self.delay).await;
        self.inner.stream_response(request).await
    }

    fn model_name(&self) -> Option<String> {
        self.inner.model_name()
    }
}

fn guardrail(delay: Duration) -> InputGuardrail<()> {
    InputGuardrail::new("delayed_input_guardrail", move |_c, _a, _i| async move {
        tokio::time::sleep(delay).await;
        Ok(GuardrailFunctionOutput::new(
            json!({ "delay": delay.as_secs_f64() }),
            false,
        ))
    })
}

/// Records trace and span lifecycle events in the order the SDK reports them.
#[derive(Debug, Default)]
struct RecordingProcessor {
    events: Mutex<Vec<(&'static str, Option<Value>)>>,
}

impl RecordingProcessor {
    fn events(&self) -> Vec<(&'static str, Option<Value>)> {
        self.events.lock().unwrap().clone()
    }
}

impl TracingProcessor for RecordingProcessor {
    fn on_trace_start(&self, _trace: &Trace) {
        self.events.lock().unwrap().push(("trace_start", None));
    }

    fn on_trace_end(&self, _trace: &Trace) {
        self.events.lock().unwrap().push(("trace_end", None));
    }

    fn on_span_start(&self, _span: &Span) {}

    fn on_span_end(&self, span: &Span) {
        self.events
            .lock()
            .unwrap()
            .push(("span_end", span.export()));
    }
}

fn span_of_type<'a>(
    events: &'a [(&'static str, Option<Value>)],
    span_type: &str,
) -> Option<&'a Value> {
    events.iter().find_map(|(kind, export)| match export {
        Some(export) if *kind == "span_end" && export["span_data"]["type"] == span_type => {
            Some(export)
        }
        _ => None,
    })
}

fn ended_at(span: &Value) -> DateTime<chrono::FixedOffset> {
    DateTime::parse_from_rfc3339(span["ended_at"].as_str().expect("span ended")).unwrap()
}

/// Runs one streamed agent and checks the span ordering the Python tests assert.
async fn run_and_check(agent: &Agent<()>) {
    let processor = Arc::new(RecordingProcessor::default());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let mut result = Runner::run_streamed_with_options(agent, "Hello", RunOptions::new(()));
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event.unwrap();
    }

    let events = processor.events();
    let agent_span = span_of_type(&events, "agent").expect("agent span");
    let guardrail_span = span_of_type(&events, "guardrail").expect("guardrail span");
    let generation_span = span_of_type(&events, "generation").expect("generation span");

    // The agent span must finish last.
    assert!(ended_at(agent_span) >= ended_at(guardrail_span));
    assert!(ended_at(agent_span) >= ended_at(generation_span));

    // The trace ends after every span does.
    assert_eq!(events.last().map(|(kind, _)| *kind), Some("trace_end"));
}

#[tokio::test]
async fn agent_span_and_trace_finish_after_slow_input_guardrail_and_slow_model() {
    openai_agents_sdk::tracing::set_tracing_disabled(false);

    // Scenario 1: the guardrail completes last.
    let model = Arc::new(ScriptedModel::new().with_emit_traces(true));
    model
        .enqueue(ScriptedStep::output(vec![assistant_message(
            "Final response",
        )]))
        .unwrap();
    let agent = Agent::<()>::new("TimingAgentTrace")
        .with_model_instance(model)
        .with_input_guardrail(guardrail(Duration::from_millis(20)));
    run_and_check(&agent).await;

    // Scenario 2: the model completes last.
    let inner = Arc::new(ScriptedModel::new().with_emit_traces(true));
    inner
        .enqueue(ScriptedStep::output(vec![assistant_message(
            "Final response",
        )]))
        .unwrap();
    let slow = Arc::new(SlowModel {
        inner,
        delay: Duration::from_millis(20),
    });
    let agent = Agent::<()>::new("TimingAgentTrace")
        .with_model_instance(slow)
        .with_input_guardrail(guardrail(Duration::ZERO));
    run_and_check(&agent).await;

    set_trace_processors(Vec::new());
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}
