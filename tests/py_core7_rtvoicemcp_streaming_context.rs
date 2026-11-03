//! Port of `tests/fastapi/test_streaming_context.py`.
//!
//! A web handler starts the run in one task and the response body consumes the stream in
//! another. Tracing state must follow the run rather than the task that started it: the stream
//! completes without a panic, the events arrive in order, and the trace is started and ended
//! exactly once.
#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tracing::spans::{Span, Trace};
use openai_agents_sdk::tracing::{TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};

#[derive(Debug, Default)]
struct Recorder {
    traces: Mutex<Vec<(String, String)>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, trace: &Trace) {
        self.traces
            .lock()
            .unwrap()
            .push(("trace_start".to_string(), trace.name().to_string()));
    }

    fn on_trace_end(&self, trace: &Trace) {
        self.traces
            .lock()
            .unwrap()
            .push(("trace_end".to_string(), trace.name().to_string()));
    }

    fn on_span_start(&self, _span: &Span) {}

    fn on_span_end(&self, _span: &Span) {}
}

/// Replace the OpenAI exporter with an in-process recorder, so no test reaches the network.
fn recorder() -> Arc<Recorder> {
    static RECORDER: OnceLock<Arc<Recorder>> = OnceLock::new();
    RECORDER
        .get_or_init(|| {
            let recorder = Arc::new(Recorder::default());
            set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
            recorder
        })
        .clone()
}

/// `test_streaming_context`: the run starts in the test task and is consumed from a spawned one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_started_in_one_task_is_consumed_in_another() {
    let recorder = recorder();
    let workflow = "streaming-context-cross-task";
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(vec![ScriptedStep::output(vec![assistant_message("done")])])
        .unwrap();
    let agent = Agent::<()>::new("streaming_app").with_model_instance(model);

    let mut options = RunOptions::default();
    options.run_config = RunConfig::default().with_workflow_name(workflow);
    let streamed = Runner::run_streamed_with_options(agent, "hi", options);

    let consumer = tokio::spawn(async move {
        let mut streamed = streamed;
        let mut kinds = Vec::new();
        {
            let mut events = std::pin::pin!(streamed.stream_events());
            while let Some(event) = events.next().await {
                kinds.push(event.expect("the stream does not fail").event_type());
            }
        }
        (kinds, streamed.final_output_text())
    });
    let (kinds, final_output) = consumer.await.expect("the consumer task does not panic");

    assert_eq!(final_output.as_deref(), Some("done"));
    assert_eq!(kinds.first(), Some(&"agent_updated_stream_event"));
    assert_eq!(kinds.last(), Some(&"run_item_stream_event"));
    assert_eq!(kinds.len(), 11, "{kinds:?}");
    assert!(
        kinds[1..kinds.len() - 1]
            .iter()
            .all(|kind| *kind == "raw_response_event"),
        "{kinds:?}"
    );

    openai_agents_sdk::tracing::flush_traces();
    let events: Vec<String> = recorder
        .traces
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, name)| name == workflow)
        .map(|(event, _)| event.clone())
        .collect();
    assert_eq!(events, ["trace_start", "trace_end"]);
}
