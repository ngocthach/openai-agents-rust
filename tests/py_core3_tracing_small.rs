//! Port of `tests/test_disabled_trace_provider_shutdown.py`: a disabled provider still shuts down
//! the processors it has registered.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::tracing::provider::{DefaultTraceProvider, TraceProvider};
use openai_agents_sdk::tracing::{Span, Trace, TracingProcessor};

#[derive(Debug, Default)]
struct ShutdownCounter {
    shutdowns: AtomicUsize,
}

impl TracingProcessor for ShutdownCounter {
    fn on_trace_start(&self, _: &Trace) {}
    fn on_trace_end(&self, _: &Trace) {}
    fn on_span_start(&self, _: &Span) {}
    fn on_span_end(&self, _: &Span) {}
    fn shutdown(&self) {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn disabled_trace_provider_still_shuts_down_registered_processors() {
    let _serial = support::lock();
    let provider = DefaultTraceProvider::new();
    let processor = Arc::new(ShutdownCounter::default());
    provider.set_processors(vec![processor.clone()]);
    provider.set_disabled(true);

    provider.shutdown();

    assert_eq!(processor.shutdowns.load(Ordering::SeqCst), 1);
}
