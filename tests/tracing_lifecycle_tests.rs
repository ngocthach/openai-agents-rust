//! Processor isolation, the shutdown guard, span-creator options and trace reattachment.
//! The trace provider and its processors are process-global, so these share one binary.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::tracing::{
    SpanData, TraceState, TracingProcessor, custom_span, hash_tracing_api_key, reattach_trace,
    set_trace_processors, shutdown_guard, span, trace, trace_id_was_started,
};
use serde_json::{Map, Value, json};

/// A processor that records what it was told about, and optionally panics first.
#[derive(Debug, Default)]
struct Recorder {
    panics: bool,
    events: Mutex<Vec<String>>,
    shutdowns: AtomicUsize,
}

impl Recorder {
    fn new(panics: bool) -> Arc<Self> {
        Arc::new(Recorder {
            panics,
            ..Default::default()
        })
    }
    fn record(&self, what: &str) {
        if self.panics {
            panic!("processor exploded during {what}");
        }
        self.events.lock().unwrap().push(what.to_string());
    }
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, _t: &openai_agents_sdk::tracing::Trace) {
        self.record("trace_start");
    }
    fn on_trace_end(&self, _t: &openai_agents_sdk::tracing::Trace) {
        self.record("trace_end");
    }
    fn on_span_start(&self, _s: &openai_agents_sdk::tracing::Span) {
        self.record("span_start");
    }
    fn on_span_end(&self, _s: &openai_agents_sdk::tracing::Span) {
        self.record("span_end");
    }
    fn shutdown(&self) {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
    }
}

/// The processor list is process-global, so the tests here run one at a time.
async fn tracing_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn silence_panics() {
    // `catch_unwind` still runs the default hook, which would print a backtrace per panic.
    std::panic::set_hook(Box::new(|_| {}));
}

#[tokio::test(flavor = "current_thread")]
async fn a_panicking_processor_does_not_stop_the_others() {
    let _serial = tracing_lock().await;
    silence_panics();
    let exploding = Recorder::new(true);
    let healthy = Recorder::new(false);
    set_trace_processors(vec![exploding.clone(), healthy.clone()]);

    trace("isolation")
        .run(async {
            custom_span("work", json!({})).run(async {}).await;
        })
        .await;

    assert_eq!(
        healthy.events(),
        ["trace_start", "span_start", "span_end", "trace_end"]
    );
    assert!(exploding.events().is_empty());
    let _ = std::panic::take_hook();
}

#[tokio::test(flavor = "current_thread")]
async fn the_shutdown_guard_shuts_the_pipeline_down_when_dropped() {
    let _serial = tracing_lock().await;
    let recorder = Recorder::new(false);
    set_trace_processors(vec![recorder.clone()]);
    {
        let _tracing = shutdown_guard();
        assert_eq!(recorder.shutdowns.load(Ordering::SeqCst), 0);
    }
    assert_eq!(recorder.shutdowns.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn span_options_cover_the_python_creator_arguments() {
    let _serial = tracing_lock().await;
    let recorder = Recorder::new(false);
    set_trace_processors(vec![recorder.clone()]);
    let parent = trace("builder").build();
    parent.start();

    let mut metadata = Map::new();
    metadata.insert("ticket".into(), json!("AGT-1"));
    let agent = span(SpanData::Agent {
        name: "Assistant".into(),
        handoffs: None,
        tools: None,
        output_type: None,
    })
    .span_id("span_0123456789abcdef01234567")
    .parent_trace(parent.clone())
    .metadata(metadata.clone())
    .build();

    assert_eq!(agent.span_id(), "span_0123456789abcdef01234567");
    assert_eq!(agent.trace_id(), parent.trace_id());
    assert_eq!(agent.parent_id(), None);
    assert_eq!(agent.data_metadata(), Some(metadata));
    // Python keeps `metadata` off the wire; so does this.
    agent.start();
    agent.finish();
    let exported = agent.export().expect("exported");
    assert!(exported["span_data"].get("metadata").is_none());

    // A child nests under an explicit parent span rather than the ambient one.
    let child = span(SpanData::Custom {
        name: "child".into(),
        data: Value::Null,
    })
    .parent_span(agent.clone())
    .build();
    assert_eq!(child.parent_id(), Some(agent.span_id()));

    // `disabled` keeps the span out of the export entirely.
    let off = span(SpanData::Custom {
        name: "off".into(),
        data: Value::Null,
    })
    .parent_trace(parent.clone())
    .disabled(true)
    .build();
    assert!(off.export().is_none());
    parent.finish();
}

#[tokio::test(flavor = "current_thread")]
async fn a_reattached_trace_keeps_the_id_without_re_emitting_the_trace() {
    let _serial = tracing_lock().await;
    let recorder = Recorder::new(false);
    set_trace_processors(vec![recorder.clone()]);

    let original = trace("resumable")
        .trace_id("trace_000000000000000000000000000001")
        .group_id("group_1")
        .tracing_api_key("sk-test")
        .build();
    original.start();
    let state = TraceState::from_trace(Some(&original)).expect("state");
    original.finish();
    assert_eq!(recorder.events(), ["trace_start", "trace_end"]);

    // The snapshot keeps a fingerprint even when the secret is stripped.
    let persisted = state.to_json(false).expect("payload");
    assert!(persisted.get("tracing_api_key").is_none());
    assert_eq!(
        persisted["tracing_api_key_hash"],
        json!(hash_tracing_api_key(Some("sk-test")).unwrap())
    );

    let restored = TraceState::from_json(Some(&persisted)).expect("restored");
    assert!(restored.matches_effective_settings(
        "resumable",
        Some("trace_000000000000000000000000000001"),
        Some("group_1"),
        None,
        Some("sk-test"),
    ));
    assert!(trace_id_was_started(Some(
        "trace_000000000000000000000000000001"
    )));

    let resumed = reattach_trace(&restored, Some("sk-test".into())).expect("reattached");
    let span_in_resumed = resumed
        .clone()
        .run(async {
            custom_span("resumed work", json!({})).run(async {}).await;
            span(SpanData::Custom {
                name: "nested".into(),
                data: Value::Null,
            })
            .build()
        })
        .await;

    // Spans still flow, but the trace itself was not announced a second time.
    assert_eq!(
        recorder.events(),
        ["trace_start", "trace_end", "span_start", "span_end"]
    );
    assert_eq!(
        span_in_resumed.trace_id(),
        "trace_000000000000000000000000000001"
    );
}
