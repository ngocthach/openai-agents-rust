//! Ports of the processor and provider tests in `tests/test_trace_processor.py`.
//!
//! Each test that installs a processor takes the shared tracing lock, because processors and the
//! trace provider are process-global. Exports go to in-memory exporters or to a local mock ingest
//! server, never to the real endpoint.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use openai_agents_sdk::tracing::processors::{
    BackendSpanExporter, BatchTraceProcessor, ExportItem, TracingExporter, TracingProcessor,
};
use openai_agents_sdk::tracing::provider::{
    DefaultTraceProvider, TraceProvider, provider, reset_trace_provider, set_trace_provider,
};
use openai_agents_sdk::tracing::{
    ConsoleSpanExporter, Span, SpanData, Trace, flush_traces, set_tracing_disabled,
    shutdown_tracing,
};
use serde_json::{Value, json};
use support::{
    Ingest, NoApiKey, Pipeline, Reply, agent_data, capture_logs, clear_logs, lock, lock_async,
    logged_text, new_span, new_trace,
};

/// Records every exported item, in order, and the size of each batch.
#[derive(Debug, Default)]
struct Collect {
    batches: Mutex<Vec<Vec<ExportItem>>>,
}

impl Collect {
    fn batch_sizes(&self) -> Vec<usize> {
        self.batches.lock().unwrap().iter().map(Vec::len).collect()
    }

    fn total(&self) -> usize {
        self.batch_sizes().iter().sum()
    }

    fn payloads(&self) -> Vec<Value> {
        self.batches
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .map(|i| i.payload.clone())
            .collect()
    }
}

impl TracingExporter for Collect {
    fn export(&self, items: Vec<ExportItem>) {
        self.batches.lock().unwrap().push(items);
    }
}

/// Signals when an export starts, then blocks until the test releases one export per message.
#[derive(Debug)]
struct Gated {
    started: Mutex<mpsc::Sender<()>>,
    release: Mutex<mpsc::Receiver<()>>,
    exported: Mutex<Vec<Value>>,
}

struct GateHandles {
    started: mpsc::Receiver<()>,
    release: mpsc::Sender<()>,
}

impl Gated {
    fn new() -> (Arc<Self>, GateHandles) {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let gated = Arc::new(Gated {
            started: Mutex::new(started_tx),
            release: Mutex::new(release_rx),
            exported: Mutex::new(Vec::new()),
        });
        (
            gated,
            GateHandles {
                started: started_rx,
                release: release_tx,
            },
        )
    }

    fn exported_names(&self) -> Vec<String> {
        self.exported
            .lock()
            .unwrap()
            .iter()
            .map(item_name)
            .collect()
    }
}

impl TracingExporter for Gated {
    fn export(&self, items: Vec<ExportItem>) {
        let _ = self.started.lock().unwrap().send(());
        let _ = self
            .release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5));
        self.exported
            .lock()
            .unwrap()
            .extend(items.into_iter().map(|i| i.payload));
    }
}

/// The name a trace or span payload carries: the workflow name, or the span data name.
fn item_name(payload: &Value) -> String {
    payload["workflow_name"]
        .as_str()
        .or_else(|| payload["span_data"]["name"].as_str())
        .unwrap_or_default()
        .to_string()
}

fn named_trace(name: &str) -> Trace {
    new_trace(name)
}

fn named_span(trace: &Trace, name: &str) -> Span {
    new_span(trace, agent_data(name))
}

fn wait_until(mut condition: impl FnMut() -> bool, limit: Duration) -> bool {
    let started = Instant::now();
    while started.elapsed() < limit {
        if condition() {
            return true;
        }
        thread::sleep(Duration::from_millis(5));
    }
    condition()
}

// ---------------------------------------------------------------------------------------------
// console exporter
// ---------------------------------------------------------------------------------------------

#[test]
fn console_span_exporter_respects_data_policy() {
    // Python parametrizes this over the two redaction flags; both must redact together.
    for redacted in [true, false] {
        let _serial = lock();
        openai_agents_sdk::debug::set_dont_log_model_data(redacted);
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        let t = named_trace("SECRET_CONSOLE_TRACE");
        let s = named_span(&t, "SECRET_CONSOLE_SPAN");
        let item = ExportItem {
            payload: s.export().unwrap(),
            api_key: None,
            object: None,
        };
        let mut out = Vec::new();
        ConsoleSpanExporter
            .write_items(&[item], &mut out)
            .expect("write");
        let output = String::from_utf8(out).unwrap();
        assert_eq!(
            !output.contains("SECRET_CONSOLE_SPAN"),
            redacted,
            "{output}"
        );
        assert!(output.contains("Export span"), "{output}");
        openai_agents_sdk::debug::reset_dont_log_model_data();
        openai_agents_sdk::debug::reset_dont_log_tool_data();
    }
}

// ---------------------------------------------------------------------------------------------
// batch processor queueing
// ---------------------------------------------------------------------------------------------

#[test]
fn batch_trace_processor_on_trace_start() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor =
        BatchTraceProcessor::with_options(collect.clone(), 8192, 128, Duration::from_secs(60), 0.7);
    let t = named_trace("test_trace");
    processor.on_trace_start(&t);
    processor.force_flush();
    assert_eq!(
        collect.total(),
        1,
        "the trace should be queued and exported"
    );
    assert_eq!(item_name(&collect.payloads()[0]), "test_trace");
    processor.shutdown();
}

#[test]
fn batch_trace_processor_on_span_end() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor =
        BatchTraceProcessor::with_options(collect.clone(), 8192, 128, Duration::from_secs(60), 0.7);
    let t = named_trace("test_trace");
    let s = named_span(&t, "test_agent");
    processor.on_span_end(&s);
    processor.force_flush();
    assert_eq!(collect.total(), 1, "the span should be queued and exported");
    assert_eq!(collect.payloads()[0]["span_data"]["name"], "test_agent");
    processor.shutdown();
}

#[test]
fn batch_trace_processor_queue_full() {
    // With a queue of two, a third and fourth item wait in line while an export is in flight. The
    // span that follows finds the queue full and is dropped, as Python's bounded queue drops it.
    let _serial = lock();
    let (gated, gate) = Gated::new();
    let processor =
        BatchTraceProcessor::with_options(gated.clone(), 2, 10, Duration::from_secs(3600), 1.0);
    let t = named_trace("t");
    processor.on_trace_start(&named_trace("a"));
    processor.on_trace_start(&named_trace("b"));
    gate.started
        .recv_timeout(Duration::from_secs(2))
        .expect("export of a and b started");
    processor.on_trace_start(&named_trace("c"));
    processor.on_trace_start(&named_trace("d"));
    processor.on_trace_start(&named_trace("dropped"));
    processor.on_span_end(&named_span(&t, "dropped_span"));
    gate.release.send(()).unwrap();
    gate.release.send(()).unwrap();
    processor.force_flush();
    assert_eq!(gated.exported_names(), ["a", "b", "c", "d"]);
    processor.shutdown();
}

#[test]
fn batch_processor_doesnt_enqueue_on_trace_end_or_span_start() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor =
        BatchTraceProcessor::with_options(collect.clone(), 8192, 128, Duration::from_secs(60), 0.7);
    let t = named_trace("t");
    let s = named_span(&t, "a");
    processor.on_trace_start(&t);
    processor.on_span_start(&s);
    processor.on_span_end(&s);
    processor.on_trace_end(&t);
    processor.force_flush();
    assert_eq!(
        collect.total(),
        2,
        "the trace and one span, nothing for span start or trace end"
    );
    processor.shutdown();
}

#[test]
fn batch_trace_processor_force_flush() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor =
        BatchTraceProcessor::with_options(collect.clone(), 100, 2, Duration::from_secs(3600), 0.9);
    let t = named_trace("t");
    processor.on_trace_start(&t);
    processor.on_span_end(&named_span(&t, "a"));
    processor.on_span_end(&named_span(&t, "b"));
    processor.force_flush();
    assert_eq!(collect.total(), 3);
    assert_eq!(
        collect.batch_sizes(),
        [2, 1],
        "batches respect max_batch_size"
    );
    processor.shutdown();
}

#[test]
fn batch_trace_processor_force_flush_waits_for_in_flight_background_export() {
    let _serial = lock();
    let (gated, gate) = Gated::new();
    let processor = Arc::new(BatchTraceProcessor::with_options(
        gated.clone(),
        8192,
        128,
        Duration::from_millis(10),
        0.7,
    ));
    processor.on_trace_start(&named_trace("background"));
    gate.started
        .recv_timeout(Duration::from_secs(2))
        .expect("the background export started");

    let flushed = Arc::new(AtomicBool::new(false));
    let flusher = {
        let (processor, flushed) = (processor.clone(), flushed.clone());
        thread::spawn(move || {
            processor.force_flush();
            flushed.store(true, Ordering::SeqCst);
        })
    };
    thread::sleep(Duration::from_millis(150));
    assert!(
        !flushed.load(Ordering::SeqCst),
        "force_flush() should wait for the in-flight export"
    );

    gate.release.send(()).unwrap();
    flusher.join().expect("flush thread");
    assert!(flushed.load(Ordering::SeqCst));
    assert_eq!(gated.exported_names(), ["background"]);
    processor.shutdown();
}

#[test]
fn batch_trace_processor_shutdown_flushes() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor =
        BatchTraceProcessor::with_options(collect.clone(), 8192, 128, Duration::from_secs(60), 0.7);
    let t = named_trace("t");
    processor.on_trace_start(&t);
    processor.on_span_end(&named_span(&t, "a"));
    processor.shutdown();
    assert_eq!(
        collect.total(),
        2,
        "everything queued is exported on shutdown"
    );
}

#[test]
fn batch_trace_processor_shutdown_timeout_returns_when_exporter_blocks() {
    let _serial = lock();
    capture_logs();
    clear_logs();
    let (gated, gate) = Gated::new();
    let processor =
        BatchTraceProcessor::with_options(gated.clone(), 1, 128, Duration::from_secs(60), 1.0);
    processor.on_span_end(&named_span(&named_trace("t"), "blocked"));
    gate.started
        .recv_timeout(Duration::from_secs(2))
        .expect("export started");

    let start = Instant::now();
    processor.shutdown_with_timeout(Duration::from_millis(50));
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(500), "{elapsed:?}");
    assert!(
        logged_text().contains("shutdown timeout reached"),
        "{}",
        logged_text()
    );

    // Release the blocked export so the worker can finish and exit.
    let _ = gate.release.send(());
    processor.shutdown_with_timeout(Duration::from_secs(2));
    drop(processor);
    drop(gate.release);
    assert!(
        wait_until(|| Arc::strong_count(&gated) == 1, Duration::from_secs(3)),
        "the worker should release the exporter once it has exited"
    );
}

#[test]
fn batch_trace_processor_shutdown_uses_custom_exporter_public_method() {
    // A custom exporter that only overrides `export` still receives the queued items on a timed
    // shutdown: the processor never bypasses a public export override.
    #[derive(Debug, Default)]
    struct Plain {
        exported: Mutex<Vec<Value>>,
    }
    impl TracingExporter for Plain {
        fn export(&self, items: Vec<ExportItem>) {
            self.exported
                .lock()
                .unwrap()
                .extend(items.into_iter().map(|i| i.payload));
        }
    }
    let _serial = lock();
    let exporter = Arc::new(Plain::default());
    let processor = BatchTraceProcessor::new(exporter.clone());
    let span = named_span(&named_trace("t"), "custom");
    processor.on_span_end(&span);
    processor.shutdown_with_timeout(Duration::from_secs(1));
    assert_eq!(
        *exporter.exported.lock().unwrap(),
        vec![span.export().unwrap()]
    );
}

// ---------------------------------------------------------------------------------------------
// exporter filters during cleanup
// ---------------------------------------------------------------------------------------------

/// Wraps the OpenAI exporter with an application filter: only traces are forwarded. With
/// `reject`, the whole batch is refused by raising, as the Python test does.
#[derive(Debug)]
struct FilteringExporter {
    inner: BackendSpanExporter,
    reject: bool,
    batch_sizes: Mutex<Vec<usize>>,
}

impl TracingExporter for FilteringExporter {
    fn export(&self, items: Vec<ExportItem>) {
        self.export_with_deadline(items, None);
    }

    fn export_with_deadline(&self, items: Vec<ExportItem>, deadline: Option<Instant>) {
        self.batch_sizes.lock().unwrap().push(items.len());
        if self.reject {
            panic!("batch rejected by application policy");
        }
        let allowed: Vec<ExportItem> = items.into_iter().filter(|i| i.trace().is_some()).collect();
        self.inner.export_with_deadline(allowed, deadline);
    }

    fn request_shutdown(&self) {
        self.inner.request_shutdown();
    }
}

#[derive(Clone, Copy, Debug)]
enum Cleanup {
    Flush,
    Shutdown,
    TimedShutdown,
    Automatic,
}

fn run_cleanup_filter_case(cleanup: Cleanup, reject: bool) {
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let exporter = Arc::new(FilteringExporter {
        inner: BackendSpanExporter::new()
            .with_endpoint(ingest.endpoint.clone())
            .with_api_key("test_key"),
        reject,
        batch_sizes: Mutex::new(Vec::new()),
    });
    // Keep the batch queued until the chosen cleanup entry point drains it.
    let processor = Arc::new(BatchTraceProcessor::with_options(
        exporter.clone(),
        8192,
        128,
        Duration::from_secs(3600),
        0.7,
    ));
    let allowed = named_trace("allowed");
    let sensitive = new_span(
        &allowed,
        SpanData::Function {
            name: "test_tool".into(),
            input: Some("SYNTHETIC_PRIVATE_INPUT".into()),
            output: Some("SYNTHETIC_PRIVATE_OUTPUT".into()),
            mcp_data: None,
        },
    );
    processor.on_trace_start(&allowed);
    processor.on_span_end(&sensitive);

    let _pipeline = Pipeline::install(vec![processor.clone()]);
    match cleanup {
        Cleanup::Flush => processor.force_flush(),
        Cleanup::Shutdown => processor.shutdown(),
        Cleanup::TimedShutdown => processor.shutdown_with_timeout(Duration::from_secs(1)),
        Cleanup::Automatic => shutdown_tracing(),
    }

    let received = ingest.items();
    if reject {
        assert!(received.is_empty(), "{received:?}");
    } else {
        assert_eq!(received, vec![allowed.export().unwrap()]);
    }
    assert!(!json!(received).to_string().contains("SYNTHETIC_PRIVATE"));
    assert_eq!(*exporter.batch_sizes.lock().unwrap(), vec![2]);
    processor.shutdown();
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_filter_flush() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Flush, false);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_filter_shutdown() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Shutdown, false);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_filter_timed_shutdown() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::TimedShutdown, false);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_filter_automatic() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Automatic, false);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_raise_flush() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Flush, true);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_raise_shutdown() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Shutdown, true);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_raise_timed_shutdown() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::TimedShutdown, true);
}

#[test]
fn batch_trace_processor_cleanup_preserves_exporter_filter_raise_automatic() {
    let _serial = lock();
    run_cleanup_filter_case(Cleanup::Automatic, true);
}

// ---------------------------------------------------------------------------------------------
// final drain with retries
// ---------------------------------------------------------------------------------------------

/// A transparent wrapper that forwards every hook, as an inheriting subclass does in Python.
#[derive(Debug)]
struct Inherited(BackendSpanExporter);

impl TracingExporter for Inherited {
    fn export(&self, items: Vec<ExportItem>) {
        self.0.export(items);
    }
    fn export_with_deadline(&self, items: Vec<ExportItem>, deadline: Option<Instant>) {
        self.0.export_with_deadline(items, deadline);
    }
    fn request_shutdown(&self) {
        self.0.request_shutdown();
    }
}

fn timed_shutdown_retries_final_drain(exporter: Arc<dyn TracingExporter>, ingest: &Ingest) {
    let processor = BatchTraceProcessor::new(exporter);
    processor.on_span_end(&named_span(&named_trace("t"), "queued"));
    processor.shutdown_with_timeout(Duration::from_secs(1));
    assert_eq!(
        ingest.count(),
        2,
        "one transient 503, then the successful retry"
    );
    assert_eq!(
        ingest.data()[1].len(),
        1,
        "the retry resends the same batch"
    );
}

#[test]
fn batch_trace_processor_timed_shutdown_retries_final_drain_default() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(503), Reply::new(200)]);
    let exporter = BackendSpanExporter::new()
        .with_endpoint(ingest.endpoint.clone())
        .with_api_key("test_key")
        .with_max_retries(2)
        .with_base_delay(Duration::from_millis(1));
    timed_shutdown_retries_final_drain(Arc::new(exporter), &ingest);
}

#[test]
fn batch_trace_processor_timed_shutdown_retries_final_drain_inherited_export() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(503), Reply::new(200)]);
    let exporter = Inherited(
        BackendSpanExporter::new()
            .with_endpoint(ingest.endpoint.clone())
            .with_api_key("test_key")
            .with_max_retries(2)
            .with_base_delay(Duration::from_millis(1)),
    );
    timed_shutdown_retries_final_drain(Arc::new(exporter), &ingest);
}

#[test]
fn batch_trace_processor_shutdown_interrupts_exporter_retry_backoff() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(504)]);
    let exporter = BackendSpanExporter::new()
        .with_endpoint(ingest.endpoint.clone())
        .with_api_key("test_key")
        .with_max_retries(100)
        .with_base_delay(Duration::from_secs(10))
        .with_max_delay(Duration::from_secs(10));
    let processor =
        BatchTraceProcessor::with_options(Arc::new(exporter), 1, 1, Duration::from_secs(60), 1.0);
    processor.on_span_end(&named_span(&named_trace("t"), "retrying"));
    assert!(wait_until(|| ingest.count() == 1, Duration::from_secs(2)));

    let start = Instant::now();
    processor.shutdown_with_timeout(Duration::from_secs(1));
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(500), "{elapsed:?}");
    assert_eq!(
        ingest.count(),
        1,
        "the 10 second backoff was cut short, no retry was sent"
    );
}

#[test]
fn batch_trace_processor_shutdown_without_timeout_preserves_export_retries() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(504)]);
    let exporter = BackendSpanExporter::new()
        .with_endpoint(ingest.endpoint.clone())
        .with_api_key("test_key")
        .with_max_retries(3)
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_millis(200));
    let processor = BatchTraceProcessor::new(Arc::new(exporter));
    processor.on_span_end(&named_span(&named_trace("t"), "retried"));
    processor.shutdown();
    assert_eq!(
        ingest.count(),
        3,
        "without a timeout every retry is still attempted"
    );
}

// ---------------------------------------------------------------------------------------------
// exporter failures
// ---------------------------------------------------------------------------------------------

/// Panics on its first export, then records everything it is given.
#[derive(Debug, Default)]
struct Flaky {
    calls: AtomicUsize,
    exported: Mutex<Vec<String>>,
}

impl TracingExporter for Flaky {
    fn export(&self, items: Vec<ExportItem>) {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            panic!("simulated exporter failure");
        }
        self.exported
            .lock()
            .unwrap()
            .extend(items.iter().map(|i| item_name(&i.payload)));
    }
}

#[test]
fn batch_trace_processor_survives_exporter_exception() {
    let _serial = lock();
    let exporter = Arc::new(Flaky::default());
    let processor = BatchTraceProcessor::with_options(
        exporter.clone(),
        8192,
        1,
        Duration::from_millis(50),
        0.7,
    );
    let t = named_trace("t");
    processor.on_span_end(&named_span(&t, "first"));
    assert!(wait_until(
        || exporter.calls.load(Ordering::SeqCst) == 1,
        Duration::from_secs(2)
    ));

    // The worker must still be alive to export these.
    processor.on_span_end(&named_span(&t, "second"));
    processor.on_span_end(&named_span(&t, "third"));
    assert!(wait_until(
        || exporter.exported.lock().unwrap().len() == 2,
        Duration::from_secs(2)
    ));
    assert_eq!(*exporter.exported.lock().unwrap(), ["second", "third"]);
    assert_eq!(exporter.calls.load(Ordering::SeqCst), 3);
    processor.shutdown_with_timeout(Duration::from_secs(2));
}

// ---------------------------------------------------------------------------------------------
// flush and provider behaviour
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct FlushCounter {
    flushes: AtomicUsize,
}

impl TracingProcessor for FlushCounter {
    fn on_trace_start(&self, _: &Trace) {}
    fn on_trace_end(&self, _: &Trace) {}
    fn on_span_start(&self, _: &Span) {}
    fn on_span_end(&self, _: &Span) {}
    fn force_flush(&self) {
        self.flushes.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn flush_traces_delegates_to_default_trace_provider() {
    let _serial = lock();
    let counter = Arc::new(FlushCounter::default());
    let _pipeline = Pipeline::install(vec![counter.clone()]);
    flush_traces();
    assert_eq!(counter.flushes.load(Ordering::SeqCst), 1);
}

#[test]
fn flush_traces_is_available_from_the_crate_root() {
    // Python exposes `flush_traces` at the package top level; Rust exposes it at the crate root.
    let _serial = lock();
    let _pipeline = Pipeline::install(Vec::new());
    let flush: fn() = openai_agents_sdk::flush_traces;
    flush();
}

#[test]
fn default_trace_provider_force_flush_still_flushes_when_disabled() {
    let _serial = lock();
    let counter = Arc::new(FlushCounter::default());
    let provider = DefaultTraceProvider::new();
    provider.set_processors(vec![counter.clone()]);
    provider.set_disabled(true);
    provider.force_flush();
    assert_eq!(counter.flushes.load(Ordering::SeqCst), 1);
}

/// A provider that only implements the required methods. `force_flush` and `shutdown` must
/// default to no-ops, as Python's `TraceProvider` does.
#[derive(Debug)]
struct MinimalProvider;

impl TraceProvider for MinimalProvider {
    fn is_disabled(&self) -> bool {
        false
    }
    fn set_disabled(&self, _: bool) {}
    fn set_processors(&self, _: Vec<Arc<dyn TracingProcessor>>) {}
    fn add_processor(&self, _: Arc<dyn TracingProcessor>) {}
    fn on_trace_start(&self, _: &Trace) {}
    fn on_trace_end(&self, _: &Trace) {}
    fn on_span_start(&self, _: &Span) {}
    fn on_span_end(&self, _: &Span) {}
}

#[test]
fn trace_provider_force_flush_and_shutdown_default_to_noops() {
    let _serial = lock();
    let minimal = MinimalProvider;
    minimal.force_flush();
    minimal.shutdown();
}

#[test]
fn get_trace_provider_force_flush_flushes_default_processor() {
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor = Arc::new(BatchTraceProcessor::with_options(
        collect.clone(),
        8192,
        128,
        Duration::from_secs(60),
        0.7,
    ));
    let provider_under_test = Arc::new(DefaultTraceProvider::new());
    provider_under_test.set_processors(vec![processor.clone()]);
    set_trace_provider(provider_under_test);
    set_tracing_disabled(false);

    let t = named_trace("t");
    processor.on_trace_start(&t);
    processor.on_span_end(&named_span(&t, "a"));
    provider().force_flush();
    reset_trace_provider();

    assert_eq!(collect.total(), 2);
    processor.shutdown();
}

#[test]
fn process_exit_shutdown_is_bounded_when_the_exporter_stalls() {
    // Python's exit hook shuts the provider down with a 5 second timeout and exits even while the
    // exporter sits in a 504 backoff. The Rust equivalent is the shutdown guard.
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(504)]);
    let exporter = BackendSpanExporter::new()
        .with_endpoint(ingest.endpoint.clone())
        .with_api_key("test_key")
        .with_max_retries(100)
        .with_base_delay(Duration::from_secs(10))
        .with_max_delay(Duration::from_secs(10));
    let processor = Arc::new(BatchTraceProcessor::with_options(
        Arc::new(exporter),
        1,
        1,
        Duration::from_secs(60),
        1.0,
    ));
    let _pipeline = Pipeline::install(vec![processor.clone()]);
    processor.on_span_end(&named_span(&named_trace("t"), "stalled"));
    assert!(wait_until(|| ingest.count() == 1, Duration::from_secs(2)));

    let start = Instant::now();
    drop(openai_agents_sdk::tracing::shutdown_guard());
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(7), "{elapsed:?}");
    assert_eq!(ingest.count(), 1);
}

#[test]
fn flush_is_a_no_op_for_an_unstarted_processor() {
    // Python's processor flushes synchronously when no thread was started; Rust has no worker to
    // wait for, so flush and shutdown return at once.
    let _serial = lock();
    let collect = Arc::new(Collect::default());
    let processor = BatchTraceProcessor::new(collect.clone());
    let start = Instant::now();
    processor.force_flush();
    processor.shutdown();
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(collect.total(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn async_flush_helper_exports_without_blocking_the_runtime() {
    let _serial = lock_async().await;
    let collect = Arc::new(Collect::default());
    let processor = Arc::new(BatchTraceProcessor::with_options(
        collect.clone(),
        8192,
        128,
        Duration::from_secs(60),
        0.7,
    ));
    let _pipeline = Pipeline::install(vec![processor.clone()]);
    let t = named_trace("async");
    processor.on_trace_start(&t);
    openai_agents_sdk::tracing::flush_traces_async().await;
    assert_eq!(collect.total(), 1);
    processor.shutdown();
}
