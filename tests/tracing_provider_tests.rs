//! Installing a custom `TraceProvider`, the run-trace masking rule and the task/turn span
//! helpers. One test: the trace provider is process-global.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::tracing::{
    DefaultTraceProvider, Span, SpanData, Trace, TraceProvider, TracingProcessor,
    create_trace_for_run, current_span, current_trace, masked_run_span, reset_trace_provider,
    set_trace_provider, task_span, trace, turn_span,
};

#[derive(Debug, Default)]
struct Recorder {
    inner: DefaultTraceProvider,
    spans: Mutex<Vec<String>>,
    traces: Mutex<Vec<String>>,
}

impl TraceProvider for Recorder {
    fn is_disabled(&self) -> bool {
        self.inner.is_disabled()
    }
    fn set_disabled(&self, disabled: bool) {
        self.inner.set_disabled(disabled);
    }
    fn set_processors(&self, processors: Vec<Arc<dyn TracingProcessor>>) {
        self.inner.set_processors(processors);
    }
    fn add_processor(&self, processor: Arc<dyn TracingProcessor>) {
        self.inner.add_processor(processor);
    }
    fn on_trace_start(&self, trace: &Trace) {
        self.traces.lock().unwrap().push(trace.name().to_string());
    }
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        let kind = match span.data() {
            SpanData::Task { name, .. } => format!("task:{name}"),
            SpanData::Turn {
                turn, agent_name, ..
            } => format!("turn:{turn}:{agent_name}"),
            other => other.span_type().to_string(),
        };
        self.spans.lock().unwrap().push(kind);
    }
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

#[tokio::test]
async fn custom_provider_masking_and_span_helpers() {
    let recorder = Arc::new(Recorder::default());
    set_trace_provider(recorder.clone());

    trace("workflow")
        .run(async {
            let task = task_span("Agent workflow");
            task.clone()
                .run(async {
                    turn_span(1, "Assistant").run(async {}).await;
                })
                .await;
            assert!(task.started_at().is_some() && task.ended_at().is_some());

            // A nested run that joins the active trace gets no trace of its own.
            assert!(
                create_trace_for_run("nested", None, None, None, None, false).is_none(),
                "an active trace must be reused"
            );

            // A run with tracing disabled always gets a (disabled) trace, which masks the
            // active one, and a disabled span that masks the active span.
            let masking = create_trace_for_run("masked", None, None, None, None, true)
                .expect("a disabled run always gets its own trace");
            assert!(masking.is_disabled());
            let outer_trace_id = current_trace().unwrap().trace_id().to_string();
            masking
                .run(async {
                    masked_run_span()
                        .run(async {
                            assert_ne!(current_trace().unwrap().trace_id(), outer_trace_id);
                            assert!(current_span().unwrap().is_disabled());
                            // Spans created inside the mask are not exported.
                            openai_agents_sdk::custom_span("hidden", serde_json::Value::Null)
                                .run(async {})
                                .await;
                        })
                        .await;
                })
                .await;
        })
        .await;

    assert_eq!(recorder.traces.lock().unwrap().clone(), vec!["workflow"]);
    assert_eq!(
        recorder.spans.lock().unwrap().clone(),
        vec!["turn:1:Assistant", "task:Agent workflow"]
    );

    reset_trace_provider();
}
