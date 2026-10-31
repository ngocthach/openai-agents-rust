//! Global trace provider. Mirrors `agents.tracing.provider` and `agents.tracing.setup`.
//!
//! [`provider`] returns the installed provider; [`set_trace_provider`] replaces it, so a custom
//! tracing backend can take over trace and span lifecycle events entirely.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use super::processors::{BackendSpanExporter, BatchTraceProcessor, TracingProcessor};
use super::spans::{Span, Trace};

/// Receives trace and span lifecycle events and owns the global tracing switches.
pub trait TraceProvider: Send + Sync + fmt::Debug {
    /// Whether tracing is globally disabled.
    fn is_disabled(&self) -> bool;
    /// Globally enable or disable tracing.
    fn set_disabled(&self, disabled: bool);
    /// Replace all processors.
    fn set_processors(&self, processors: Vec<Arc<dyn TracingProcessor>>);
    /// Add a processor, keeping the default OpenAI exporter.
    fn add_processor(&self, processor: Arc<dyn TracingProcessor>);
    /// A trace started.
    fn on_trace_start(&self, trace: &Trace);
    /// A trace finished.
    fn on_trace_end(&self, trace: &Trace);
    /// A span started.
    fn on_span_start(&self, span: &Span);
    /// A span finished.
    fn on_span_end(&self, span: &Span);
    /// Flush all processors (blocking). The default does nothing, as in Python.
    fn force_flush(&self) {}
    /// Shut down all processors (blocking). The default does nothing, as in Python.
    fn shutdown(&self) {}
    /// Shut down all processors, giving up once `timeout` has elapsed. Mirrors
    /// `TraceProvider.shutdown(timeout=...)`; the default ignores the timeout.
    fn shutdown_with_timeout(&self, _timeout: Duration) {
        self.shutdown();
    }
}

/// How long the process-exit shutdown may wait for queued traces, matching Python's
/// `_DEFAULT_SHUTDOWN_TIMEOUT`.
pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// The built-in provider: holds the registered processors and the global disabled flag.
pub struct DefaultTraceProvider {
    processors: RwLock<Option<Vec<Arc<dyn TracingProcessor>>>>,
    /// Set by [`TraceProvider::set_disabled`]; takes precedence over the environment.
    manual_disabled: RwLock<Option<bool>>,
    /// `OPENAI_AGENTS_DISABLE_TRACING`, read once on first use (`u8`: 0 unknown, 1 off, 2 on).
    env_disabled: AtomicU8,
}

impl fmt::Debug for DefaultTraceProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefaultTraceProvider")
            .field("disabled", &self.is_disabled())
            .finish_non_exhaustive()
    }
}

impl Default for DefaultTraceProvider {
    fn default() -> Self {
        Self::new()
    }
}

static INSTALLED: RwLock<Option<Arc<dyn TraceProvider>>> = RwLock::new(None);
static DEFAULT_PROVIDER: OnceLock<Arc<DefaultTraceProvider>> = OnceLock::new();
static DEFAULT_PROCESSOR: OnceLock<Arc<BatchTraceProcessor>> = OnceLock::new();
static EXPORT_API_KEY: RwLock<Option<String>> = RwLock::new(None);

fn env_disabled() -> bool {
    matches!(
        std::env::var("OPENAI_AGENTS_DISABLE_TRACING")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true"
    )
}

/// The installed trace provider.
pub fn provider() -> Arc<dyn TraceProvider> {
    if let Some(p) = INSTALLED.read().expect("provider poisoned").as_ref() {
        return p.clone();
    }
    default_provider() as Arc<dyn TraceProvider>
}

/// The built-in provider instance, whether or not it is currently installed.
pub fn default_provider() -> Arc<DefaultTraceProvider> {
    DEFAULT_PROVIDER
        .get_or_init(|| Arc::new(DefaultTraceProvider::new()))
        .clone()
}

/// Install a custom trace provider for the whole process.
pub fn set_trace_provider(provider: Arc<dyn TraceProvider>) {
    *INSTALLED.write().expect("provider poisoned") = Some(provider);
}

/// Go back to the built-in trace provider.
pub fn reset_trace_provider() {
    *INSTALLED.write().expect("provider poisoned") = None;
}

/// The default batch processor exporting to OpenAI.
pub fn default_processor() -> Arc<BatchTraceProcessor> {
    DEFAULT_PROCESSOR
        .get_or_init(|| {
            Arc::new(BatchTraceProcessor::new(Arc::new(
                BackendSpanExporter::new(),
            )))
        })
        .clone()
}

/// Set the API key used by the default exporter.
pub fn set_export_api_key(key: impl Into<String>) {
    *EXPORT_API_KEY.write().expect("provider poisoned") = Some(key.into());
}

pub(crate) fn tracing_export_api_key() -> Option<String> {
    // An empty key means "unset", so `set_export_api_key("")` clears the global key and the
    // exporter falls back to the environment, as `BackendSpanExporter::set_api_key("")` does.
    EXPORT_API_KEY
        .read()
        .expect("provider poisoned")
        .clone()
        .filter(|k| !k.is_empty())
}

impl DefaultTraceProvider {
    /// A provider with no processors registered, reading the disabled flag from the
    /// `OPENAI_AGENTS_DISABLE_TRACING` environment variable.
    pub fn new() -> Self {
        DefaultTraceProvider {
            processors: RwLock::new(None),
            manual_disabled: RwLock::new(None),
            env_disabled: AtomicU8::new(0),
        }
    }

    /// `OPENAI_AGENTS_DISABLE_TRACING`, read once and then cached, as in Python.
    fn env_disabled_cached(&self) -> bool {
        match self.env_disabled.load(Ordering::Relaxed) {
            0 => {
                let value = env_disabled();
                self.env_disabled
                    .store(if value { 2 } else { 1 }, Ordering::Relaxed);
                value
            }
            v => v == 2,
        }
    }

    fn processors(&self) -> Vec<Arc<dyn TracingProcessor>> {
        let guard = self.processors.read().expect("provider poisoned");
        match guard.as_ref() {
            Some(list) => list.clone(),
            None => vec![default_processor() as Arc<dyn TracingProcessor>],
        }
    }

    /// Set the API key used by the default exporter.
    pub fn set_export_api_key(&self, key: impl Into<String>) {
        set_export_api_key(key);
    }
}

impl TraceProvider for DefaultTraceProvider {
    fn is_disabled(&self) -> bool {
        match *self.manual_disabled.read().expect("provider poisoned") {
            Some(manual) => manual,
            None => self.env_disabled_cached(),
        }
    }

    fn set_disabled(&self, disabled: bool) {
        *self.manual_disabled.write().expect("provider poisoned") = Some(disabled);
    }

    fn set_processors(&self, processors: Vec<Arc<dyn TracingProcessor>>) {
        *self.processors.write().expect("provider poisoned") = Some(processors);
    }

    fn add_processor(&self, processor: Arc<dyn TracingProcessor>) {
        let mut guard = self.processors.write().expect("provider poisoned");
        let list =
            guard.get_or_insert_with(|| vec![default_processor() as Arc<dyn TracingProcessor>]);
        list.push(processor);
    }

    fn on_trace_start(&self, trace: &Trace) {
        for p in self.processors() {
            isolate("on_trace_start", &p, || p.on_trace_start(trace));
        }
    }
    fn on_trace_end(&self, trace: &Trace) {
        for p in self.processors() {
            isolate("on_trace_end", &p, || p.on_trace_end(trace));
        }
    }
    fn on_span_start(&self, span: &Span) {
        for p in self.processors() {
            isolate("on_span_start", &p, || p.on_span_start(span));
        }
    }
    fn on_span_end(&self, span: &Span) {
        for p in self.processors() {
            isolate("on_span_end", &p, || p.on_span_end(span));
        }
    }

    fn force_flush(&self) {
        // Python's `force_flush` refreshes the disabled flag, which pins the environment value
        // on first use; flushing must not change whether a later trace is recorded.
        let _ = self.env_disabled_cached();
        for p in self.processors() {
            isolate("force_flush", &p, || p.force_flush());
        }
    }

    fn shutdown(&self) {
        for p in self.processors() {
            isolate("shutdown", &p, || p.shutdown());
        }
    }

    fn shutdown_with_timeout(&self, timeout: Duration) {
        for p in self.processors() {
            isolate("shutdown", &p, || p.shutdown_with_timeout(timeout));
        }
    }
}

/// Run one processor callback, keeping a panic inside it from stopping the other processors.
///
/// Port of the per-processor `try/except` in `SynchronousMultiTracingProcessor`. A Rust
/// processor signals failure by panicking rather than by raising, so the isolation is
/// `catch_unwind`; the diagnostic names the processor the way `_processor_diagnostic_extra`
/// does. Under `panic = "abort"` the process still aborts — nothing can catch that.
fn isolate(action: &str, processor: &Arc<dyn TracingProcessor>, f: impl FnOnce()) {
    if std::panic::catch_unwind(AssertUnwindSafe(f)).is_err() {
        log::error!(
            target: crate::logger::LOG_TARGET,
            "Error in trace processor during {action} (trace_processor: {processor:?})"
        );
    }
}
