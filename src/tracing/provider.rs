//! Global trace provider. Mirrors `agents.tracing.provider` and `agents.tracing.setup`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use super::processors::{BackendSpanExporter, BatchTraceProcessor, TracingProcessor};
use super::spans::{Span, Trace};

/// Holds the registered processors and the global disabled flag.
pub struct TraceProvider {
    processors: RwLock<Option<Vec<Arc<dyn TracingProcessor>>>>,
    disabled: AtomicBool,
    export_api_key: RwLock<Option<String>>,
}

impl fmt::Debug for TraceProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceProvider")
            .field("disabled", &self.disabled.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

static PROVIDER: OnceLock<TraceProvider> = OnceLock::new();
static DEFAULT_PROCESSOR: OnceLock<Arc<BatchTraceProcessor>> = OnceLock::new();

fn env_disabled() -> bool {
    matches!(
        std::env::var("OPENAI_AGENTS_DISABLE_TRACING")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true"
    )
}

/// The global trace provider.
pub fn provider() -> &'static TraceProvider {
    PROVIDER.get_or_init(|| TraceProvider {
        processors: RwLock::new(None),
        disabled: AtomicBool::new(env_disabled()),
        export_api_key: RwLock::new(None),
    })
}

/// The default batch processor exporting to OpenAI.
pub fn default_processor() -> Arc<BatchTraceProcessor> {
    DEFAULT_PROCESSOR
        .get_or_init(|| Arc::new(BatchTraceProcessor::new(Arc::new(BackendSpanExporter::new()))))
        .clone()
}

pub(crate) fn tracing_export_api_key() -> Option<String> {
    provider()
        .export_api_key
        .read()
        .expect("provider poisoned")
        .clone()
}

impl TraceProvider {
    fn processors(&self) -> Vec<Arc<dyn TracingProcessor>> {
        let guard = self.processors.read().expect("provider poisoned");
        match guard.as_ref() {
            Some(list) => list.clone(),
            None => vec![default_processor() as Arc<dyn TracingProcessor>],
        }
    }

    /// Whether tracing is globally disabled.
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }

    /// Globally enable or disable tracing.
    pub fn set_disabled(&self, disabled: bool) {
        self.disabled.store(disabled, Ordering::Relaxed);
    }

    /// Replace all processors.
    pub fn set_processors(&self, processors: Vec<Arc<dyn TracingProcessor>>) {
        *self.processors.write().expect("provider poisoned") = Some(processors);
    }

    /// Add a processor (keeps the default OpenAI exporter).
    pub fn add_processor(&self, processor: Arc<dyn TracingProcessor>) {
        let mut guard = self.processors.write().expect("provider poisoned");
        let list = guard.get_or_insert_with(|| vec![default_processor() as Arc<dyn TracingProcessor>]);
        list.push(processor);
    }

    /// Set the API key used by the default exporter.
    pub fn set_export_api_key(&self, key: impl Into<String>) {
        *self.export_api_key.write().expect("provider poisoned") = Some(key.into());
    }

    pub(crate) fn on_trace_start(&self, trace: &Trace) {
        for p in self.processors() {
            p.on_trace_start(trace);
        }
    }
    pub(crate) fn on_trace_end(&self, trace: &Trace) {
        for p in self.processors() {
            p.on_trace_end(trace);
        }
    }
    pub(crate) fn on_span_start(&self, span: &Span) {
        for p in self.processors() {
            p.on_span_start(span);
        }
    }
    pub(crate) fn on_span_end(&self, span: &Span) {
        for p in self.processors() {
            p.on_span_end(span);
        }
    }

    /// Flush all processors (blocking).
    pub fn force_flush(&self) {
        for p in self.processors() {
            p.force_flush();
        }
    }

    /// Shut down all processors (blocking).
    pub fn shutdown(&self) {
        for p in self.processors() {
            p.shutdown();
        }
    }
}
