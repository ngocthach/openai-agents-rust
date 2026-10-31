//! Ports of `tests/tracing/test_processor_api_key.py`, `test_set_api_key_fix.py`,
//! `test_logger.py`, `test_tracing_env_disable.py`, `test_spans_impl.py`, `test_traces_impl.py`,
//! `test_span_ordering.py` and `test_trace_context.py`.
//!
//! Tracing state (the installed provider, the export key and `OPENAI_AGENTS_DISABLE_TRACING`) is
//! process-global, so every test that touches it takes [`tracing_lock`] first.

#![allow(clippy::await_holding_lock)] // the lock serializes process-global tracing state for a whole test

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use openai_agents_sdk::tracing::processors::OPENAI_TRACING_INGEST_ENDPOINT;
use openai_agents_sdk::tracing::provider::{default_processor, set_export_api_key};
use openai_agents_sdk::tracing::{
    BackendSpanExporter, DefaultTraceProvider, ExportItem, Span, SpanData, TraceProvider,
    TraceState, TracingExporter, TracingProcessor,
};

// ---------------------------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------------------------

fn tracing_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Set `OPENAI_API_KEY` (or remove it with `None`) and restore the old value on drop.
struct EnvGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var(name).ok();
        apply_env(name, value);
        EnvGuard { name, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        apply_env(self.name, self.previous.as_deref());
    }
}

fn apply_env(name: &str, value: Option<&str>) {
    // Safety: every caller holds `tracing_lock`, so no other test thread reads the environment
    // while it changes.
    unsafe {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}

/// One request the mock ingest endpoint received.
#[derive(Debug, Clone)]
struct Received {
    authorization: Option<String>,
    body: serde_json::Value,
}

/// An ingest endpoint that answers `200` and records every request.
fn mock_ingest(expected_requests: usize) -> (String, Arc<Mutex<Vec<Received>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = received.clone();
    std::thread::spawn(move || {
        for _ in 0..expected_requests {
            let Ok((mut sock, _)) = listener.accept() else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            loop {
                let Ok(n) = sock.read(&mut tmp) else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let len: usize = text[..idx]
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break;
                    }
                }
            }
            let text = String::from_utf8_lossy(&buf).to_string();
            let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
            let authorization = head.lines().find_map(|l| {
                let lower = l.to_lowercase();
                lower
                    .strip_prefix("authorization:")
                    .map(|_| l[l.find(':').unwrap() + 1..].trim().to_string())
            });
            sink.lock().unwrap().push(Received {
                authorization,
                body: serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
            });
            let resp = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}";
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });
    (format!("http://{addr}/traces/ingest"), received)
}

fn item(id: &str, api_key: Option<&str>) -> ExportItem {
    ExportItem {
        object: None,
        payload: serde_json::json!({"object": "trace.span", "id": id}),
        api_key: api_key.map(Into::into),
    }
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_logger.py
// ---------------------------------------------------------------------------------------------

/// Python asserts the tracing logger is named `openai.agents.tracing`. The Rust SDK logs every
/// message, tracing included, on the single `openai.agents` target, which is the name Python's
/// parent logger carries and the one a `log` backend filters on.
#[test]
fn tracing_messages_use_the_sdk_log_target() {
    assert_eq!(openai_agents_sdk::logger::LOG_TARGET, "openai.agents");
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_processor_api_key.py + test_set_api_key_fix.py
// ---------------------------------------------------------------------------------------------

#[test]
fn exporter_api_key_is_none_without_an_environment_key() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", None);
    let exporter = BackendSpanExporter::new();
    assert_eq!(exporter.api_key(), None);

    exporter.set_api_key("test_api_key");
    assert_eq!(exporter.api_key().as_deref(), Some("test_api_key"));
}

#[test]
fn exporter_api_key_is_read_from_the_environment_after_construction() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", None);
    let exporter = BackendSpanExporter::new();

    let _set_later = EnvGuard::set("OPENAI_API_KEY", Some("foo_bar_123"));
    assert_eq!(exporter.api_key().as_deref(), Some("foo_bar_123"));
}

#[test]
fn set_api_key_preserves_the_environment_fallback() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", Some("env-key"));
    let exporter = BackendSpanExporter::new();
    assert_eq!(exporter.api_key().as_deref(), Some("env-key"));

    exporter.set_api_key("explicit-key");
    assert_eq!(exporter.api_key().as_deref(), Some("explicit-key"));

    exporter.set_api_key("");
    assert_eq!(exporter.api_key().as_deref(), Some("env-key"));
}

#[test]
fn the_exporter_authenticates_each_item_with_its_own_key() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", None);
    let (endpoint, received) = mock_ingest(3);
    let exporter = BackendSpanExporter::new().with_endpoint(endpoint);
    exporter.set_api_key("global-key");

    exporter.export(vec![
        item("a", Some("key-a")),
        item("b", None),
        item("c", Some("key-b")),
    ]);

    let calls = received.lock().unwrap().clone();
    assert_eq!(calls.len(), 3);
    let by_first_id: HashMap<String, String> = calls
        .iter()
        .map(|c| {
            let ids: Vec<String> = c.body["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d["id"].as_str().unwrap().to_string())
                .collect();
            (ids.join(","), c.authorization.clone().unwrap_or_default())
        })
        .collect();
    assert_eq!(
        by_first_id.get("a").map(String::as_str),
        Some("Bearer key-a")
    );
    assert_eq!(
        by_first_id.get("c").map(String::as_str),
        Some("Bearer key-b")
    );
    assert_eq!(
        by_first_id.get("b").map(String::as_str),
        Some("Bearer global-key")
    );
}

#[test]
fn an_environment_key_set_after_a_keyless_export_is_used() {
    let _lock = tracing_lock();
    let env = EnvGuard::set("OPENAI_API_KEY", None);
    let (endpoint, received) = mock_ingest(1);
    let exporter = BackendSpanExporter::new().with_endpoint(endpoint);

    // No key anywhere: nothing is sent, and the missing key is not remembered.
    exporter.export(vec![item("item", None)]);
    assert!(received.lock().unwrap().is_empty());

    drop(env);
    let _env = EnvGuard::set("OPENAI_API_KEY", Some("sk-set-later"));
    exporter.export(vec![item("item", None)]);

    let calls = received.lock().unwrap().clone();
    assert_eq!(
        calls
            .iter()
            .map(|c| c.authorization.clone())
            .collect::<Vec<_>>(),
        vec![Some("Bearer sk-set-later".to_string())]
    );
    assert_eq!(exporter.api_key().as_deref(), Some("sk-set-later"));
}

#[test]
fn an_environment_key_is_kept_once_resolved() {
    let _lock = tracing_lock();
    let first = EnvGuard::set("OPENAI_API_KEY", Some("sk-first"));
    let (endpoint, received) = mock_ingest(2);
    let exporter = BackendSpanExporter::new().with_endpoint(endpoint);

    exporter.export(vec![item("item", None)]);
    drop(first);
    let _rotated = EnvGuard::set("OPENAI_API_KEY", Some("sk-rotated"));
    exporter.export(vec![item("item", None)]);

    let calls = received.lock().unwrap().clone();
    assert_eq!(
        calls
            .iter()
            .map(|c| c.authorization.clone())
            .collect::<Vec<_>>(),
        vec![
            Some("Bearer sk-first".to_string()),
            Some("Bearer sk-first".to_string())
        ]
    );
}

#[test]
fn an_explicitly_assigned_key_wins_over_the_environment() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", Some("sk-env"));
    let (endpoint, received) = mock_ingest(1);
    let exporter = BackendSpanExporter::new().with_endpoint(endpoint);
    exporter.set_api_key("sk-assigned");

    exporter.export(vec![item("item", None)]);

    let calls = received.lock().unwrap().clone();
    assert_eq!(
        calls
            .iter()
            .map(|c| c.authorization.clone())
            .collect::<Vec<_>>(),
        vec![Some("Bearer sk-assigned".to_string())]
    );
}

/// `set_api_key` on one handle is visible through every clone, the way the Python exporter is a
/// single object shared by the provider.
#[test]
fn set_api_key_is_shared_by_clones() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_API_KEY", None);
    let exporter = BackendSpanExporter::new();
    let clone = exporter.clone();
    clone.set_api_key("sk-clone");
    assert_eq!(exporter.api_key().as_deref(), Some("sk-clone"));
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_tracing_env_disable.py
// ---------------------------------------------------------------------------------------------

/// Run `f` with `provider` installed globally, then restore the built-in provider.
fn with_provider<R>(provider: Arc<dyn TraceProvider>, f: impl FnOnce() -> R) -> R {
    openai_agents_sdk::tracing::set_trace_provider(provider);
    let out = f();
    openai_agents_sdk::tracing::reset_trace_provider();
    out
}

fn new_trace(name: &str) -> openai_agents_sdk::tracing::Trace {
    openai_agents_sdk::tracing::trace(name).build()
}

#[test]
fn the_disable_environment_variable_is_read_on_first_use() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("1"));
    let provider = Arc::new(DefaultTraceProvider::new());
    with_provider(provider, || {
        assert!(new_trace("demo").is_disabled());
    });
}

#[test]
fn the_disable_environment_variable_is_cached_after_first_use() {
    let _lock = tracing_lock();
    let enabled = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("0"));
    let provider = Arc::new(DefaultTraceProvider::new());
    with_provider(provider.clone(), || {
        assert!(!new_trace("first").is_disabled());
    });
    drop(enabled);
    let _disabled = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("1"));
    with_provider(provider, || {
        assert!(!new_trace("second").is_disabled());
    });
}

#[test]
fn force_flush_initializes_the_environment_disable_cache() {
    let _lock = tracing_lock();
    let disabled = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("1"));
    let provider = Arc::new(DefaultTraceProvider::new());
    // Flushing pins the environment value, so a later change cannot re-enable tracing.
    provider.set_processors(vec![]);
    provider.force_flush();
    drop(disabled);
    let _enabled = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("0"));
    with_provider(provider, || {
        assert!(new_trace("still-disabled").is_disabled());
    });
}

#[test]
fn a_manual_override_wins_after_the_environment_is_cached() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("0"));
    let provider = Arc::new(DefaultTraceProvider::new());
    with_provider(provider.clone(), || {
        new_trace("warmup");
        provider.set_disabled(true);
        assert!(new_trace("disabled").is_disabled());
        provider.set_disabled(false);
        assert!(!new_trace("enabled").is_disabled());
    });
}

#[test]
fn a_manual_enable_overrides_the_environment_disable() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("1"));
    let provider = Arc::new(DefaultTraceProvider::new());
    with_provider(provider.clone(), || {
        assert!(new_trace("env_disabled").is_disabled());
        provider.set_disabled(false);
        assert!(!new_trace("reenabled").is_disabled());
    });
}

#[tokio::test]
async fn a_span_without_an_active_trace_records_nothing() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let provider = Arc::new(DefaultTraceProvider::new());
    let span = with_provider(provider, || {
        openai_agents_sdk::tracing::agent_span("missing-trace", None, None, None)
    });
    assert!(span.is_disabled());
    assert!(span.export().is_none());
}

#[tokio::test]
async fn an_explicit_no_op_span_id_records_nothing() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let provider = Arc::new(DefaultTraceProvider::new());
    openai_agents_sdk::tracing::set_trace_provider(provider);
    let trace = openai_agents_sdk::tracing::trace("active")
        .trace_id("trace_123")
        .build();
    let span = trace
        .clone()
        .run(async {
            openai_agents_sdk::tracing::span(SpanData::Agent {
                name: "invalid".into(),
                handoffs: None,
                tools: None,
                output_type: None,
            })
            .span_id("no-op")
            .build()
        })
        .await;
    openai_agents_sdk::tracing::reset_trace_provider();
    assert!(span.is_disabled());
}

#[tokio::test]
async fn a_no_op_parent_span_does_not_become_a_parent_id() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let provider = Arc::new(DefaultTraceProvider::new());
    openai_agents_sdk::tracing::set_trace_provider(provider);
    let trace = openai_agents_sdk::tracing::trace("active")
        .trace_id("trace_123")
        .build();
    let child = trace
        .clone()
        .run(async {
            let invalid_parent = openai_agents_sdk::tracing::span(SpanData::Agent {
                name: "invalid-parent".into(),
                handoffs: None,
                tools: None,
                output_type: None,
            })
            .span_id("no-op")
            .build();
            openai_agents_sdk::tracing::span(SpanData::Agent {
                name: "child".into(),
                handoffs: None,
                tools: None,
                output_type: None,
            })
            .parent_span(invalid_parent)
            .build()
        })
        .await;
    openai_agents_sdk::tracing::reset_trace_provider();
    assert!(child.is_disabled());
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_traces_impl.py + test_spans_impl.py
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct RecordingProcessor {
    started: Mutex<Vec<String>>,
    ended: Mutex<Vec<String>>,
    spans_started: Mutex<Vec<String>>,
    spans_ended: Mutex<Vec<String>>,
}

impl TracingProcessor for RecordingProcessor {
    fn on_trace_start(&self, trace: &openai_agents_sdk::tracing::Trace) {
        self.started.lock().unwrap().push(trace.trace_id().into());
    }
    fn on_trace_end(&self, trace: &openai_agents_sdk::tracing::Trace) {
        self.ended.lock().unwrap().push(trace.trace_id().into());
    }
    fn on_span_start(&self, span: &Span) {
        self.spans_started
            .lock()
            .unwrap()
            .push(span.span_id().into());
    }
    fn on_span_end(&self, span: &Span) {
        self.spans_ended.lock().unwrap().push(span.span_id().into());
    }
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

/// Install `processor` as the only processor of a fresh provider for the duration of `f`.
async fn with_recording<F, Fut, R>(f: F) -> (R, Arc<RecordingProcessor>)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let processor = Arc::new(RecordingProcessor::default());
    let provider = Arc::new(DefaultTraceProvider::new());
    provider.set_disabled(false);
    provider.set_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    openai_agents_sdk::tracing::set_trace_provider(provider);
    let out = f().await;
    openai_agents_sdk::tracing::reset_trace_provider();
    (out, processor)
}

#[tokio::test]
async fn a_trace_lifecycle_sets_the_current_trace_and_exports_its_payload() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let (exported, processor) = with_recording(|| async {
        let trace = openai_agents_sdk::tracing::trace("test-trace")
            .trace_id("trace-123")
            .group_id("group-1")
            .metadata(serde_json::json!({"k": "v"}).as_object().unwrap().clone())
            .build();
        assert!(openai_agents_sdk::tracing::current_trace().is_none());
        let inner = trace
            .clone()
            .run(async {
                let current = openai_agents_sdk::tracing::current_trace().unwrap();
                assert_eq!(current.trace_id(), "trace-123");
                current.export()
            })
            .await;
        assert!(openai_agents_sdk::tracing::current_trace().is_none());
        inner
    })
    .await;

    assert_eq!(*processor.started.lock().unwrap(), vec!["trace-123"]);
    assert_eq!(*processor.ended.lock().unwrap(), vec!["trace-123"]);
    assert_eq!(
        exported.unwrap(),
        serde_json::json!({
            "object": "trace",
            "id": "trace-123",
            "workflow_name": "test-trace",
            "group_id": "group-1",
            "metadata": {"k": "v"},
        })
    );
}

#[tokio::test]
async fn starting_or_finishing_a_trace_twice_notifies_the_processor_once() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let (_, processor) = with_recording(|| async {
        let trace = openai_agents_sdk::tracing::trace("double-start")
            .trace_id("trace-double")
            .build();
        trace.start();
        trace.start();
        trace.finish();
        trace.finish();

        // Finishing a trace that was never started is a no-op, not an error.
        openai_agents_sdk::tracing::trace("finish-no-start")
            .trace_id("trace-never")
            .build()
            .finish();
    })
    .await;

    assert_eq!(*processor.started.lock().unwrap(), vec!["trace-double"]);
    assert_eq!(*processor.ended.lock().unwrap(), vec!["trace-double"]);
}

#[tokio::test]
async fn a_reattached_trace_restores_the_scope_without_re_emitting_events() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let (restored_id, processor) = with_recording(|| async {
        let original = openai_agents_sdk::tracing::trace("test-trace")
            .trace_id("trace-123")
            .group_id("group-1")
            .metadata(serde_json::json!({"k": "v"}).as_object().unwrap().clone())
            .build();
        original.clone().run(async {}).await;

        let state = TraceState::from_trace(Some(&original)).unwrap();
        let restored = openai_agents_sdk::tracing::reattach_trace(&state, None).unwrap();
        assert!(restored.is_reattached());
        let id = restored
            .clone()
            .run(async {
                openai_agents_sdk::tracing::current_trace()
                    .unwrap()
                    .trace_id()
                    .to_string()
            })
            .await;
        assert!(openai_agents_sdk::tracing::current_trace().is_none());
        id
    })
    .await;

    assert_eq!(restored_id, "trace-123");
    assert_eq!(*processor.started.lock().unwrap(), vec!["trace-123"]);
    assert_eq!(*processor.ended.lock().unwrap(), vec!["trace-123"]);
}

#[tokio::test]
async fn a_disabled_trace_never_reaches_the_processors() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let (_, processor) = with_recording(|| async {
        let trace = openai_agents_sdk::tracing::trace("off")
            .disabled(true)
            .build();
        trace.clone().run(async {}).await;
        assert!(trace.export().is_none());
    })
    .await;
    assert!(processor.started.lock().unwrap().is_empty());
    assert!(processor.ended.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_span_notifies_the_processor_on_start_and_end() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let (span_id, processor) = with_recording(|| async {
        openai_agents_sdk::tracing::trace("w")
            .build()
            .run(async {
                let span =
                    openai_agents_sdk::tracing::agent_span("generator-exit", None, None, None);
                let id = span.span_id().to_string();
                span.run(async {}).await;
                id
            })
            .await
    })
    .await;
    assert_eq!(
        *processor.spans_started.lock().unwrap(),
        vec![span_id.clone()]
    );
    assert_eq!(*processor.spans_ended.lock().unwrap(), vec![span_id]);
}

#[tokio::test]
async fn leaving_a_span_restores_the_enclosing_span() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    with_recording(|| async {
        openai_agents_sdk::tracing::trace("w")
            .build()
            .run(async {
                assert!(openai_agents_sdk::tracing::current_span().is_none());
                let outer = openai_agents_sdk::tracing::agent_span("outer", None, None, None);
                let outer_id = outer.span_id().to_string();
                outer
                    .run(async {
                        let inner = openai_agents_sdk::tracing::custom_span(
                            "inner",
                            serde_json::Value::Null,
                        );
                        inner.run(async {}).await;
                        assert_eq!(
                            openai_agents_sdk::tracing::current_span()
                                .unwrap()
                                .span_id(),
                            outer_id
                        );
                    })
                    .await;
                assert!(openai_agents_sdk::tracing::current_span().is_none());
            })
            .await;
    })
    .await;
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_span_ordering.py
// ---------------------------------------------------------------------------------------------

/// The ordering half of Python's `SpanProcessorForTests`: spans are collected on end, but ordered
/// by `(started_at, start order)` keyed on `(trace_id, span_id)`.
///
/// `started_at` is frozen to one value for every span, which is what the Python fixture does to
/// reproduce a coarse clock; ordering then depends entirely on the recorded start order.
#[derive(Debug, Default)]
struct OrderingProcessor {
    start_order: Mutex<Vec<(String, String)>>,
    finished: Mutex<Vec<(String, String)>>,
}

impl OrderingProcessor {
    fn ordered(&self) -> Vec<(String, String)> {
        let order = self.start_order.lock().unwrap().clone();
        let mut finished = self.finished.lock().unwrap().clone();
        // Every span shares one frozen `started_at`, so the recorded start order is the only key.
        finished.sort_by_key(|key| order.iter().position(|k| k == key).unwrap_or(usize::MAX));
        finished
    }
}

impl TracingProcessor for OrderingProcessor {
    fn on_trace_start(&self, _t: &openai_agents_sdk::tracing::Trace) {}
    fn on_trace_end(&self, _t: &openai_agents_sdk::tracing::Trace) {}
    fn on_span_start(&self, span: &Span) {
        let key = (span.trace_id().to_string(), span.span_id().to_string());
        let mut order = self.start_order.lock().unwrap();
        if !order.contains(&key) {
            order.push(key);
        }
    }
    fn on_span_end(&self, span: &Span) {
        self.finished
            .lock()
            .unwrap()
            .push((span.trace_id().to_string(), span.span_id().to_string()));
    }
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

fn assert_every_parent_comes_first(
    ordered: &[(String, String)],
    parents: &HashMap<String, String>,
) {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for key in ordered {
        if let Some(parent) = parents.get(&key.1) {
            assert!(
                seen.contains(&(key.0.clone(), parent.clone())),
                "a span was ordered before its parent"
            );
        }
        seen.insert(key.clone());
    }
}

async fn run_with_ordering<F, Fut>(f: F) -> Arc<OrderingProcessor>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let processor = Arc::new(OrderingProcessor::default());
    let provider = Arc::new(DefaultTraceProvider::new());
    provider.set_disabled(false);
    provider.set_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    openai_agents_sdk::tracing::set_trace_provider(provider);
    f().await;
    openai_agents_sdk::tracing::reset_trace_provider();
    processor
}

#[tokio::test]
async fn ordered_spans_put_a_parent_before_its_child_on_a_tied_timestamp() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let parents: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    let recorded = parents.clone();
    let processor = run_with_ordering(|| async move {
        openai_agents_sdk::tracing::trace("w")
            .build()
            .run(async {
                let parent = openai_agents_sdk::tracing::agent_span("parent", None, None, None);
                let parent_id = parent.span_id().to_string();
                parent
                    .run(async {
                        let child = openai_agents_sdk::tracing::custom_span(
                            "child",
                            serde_json::Value::Null,
                        );
                        recorded
                            .lock()
                            .unwrap()
                            .insert(child.span_id().to_string(), parent_id.clone());
                        child.run(async {}).await;
                    })
                    .await;
            })
            .await;
    })
    .await;

    let ordered = processor.ordered();
    assert_eq!(ordered.len(), 2);
    // Spans end innermost-first, so ordering on the frozen timestamp alone would put the child
    // first; the recorded start order is what keeps the parent ahead of it.
    assert_every_parent_comes_first(&ordered, &parents.lock().unwrap());
}

#[tokio::test]
async fn two_traces_reusing_one_span_id_keep_their_own_order() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    const SHARED_SPAN_ID: &str = "span_00000000000000000000000000";

    let processor = run_with_ordering(|| async {
        openai_agents_sdk::tracing::trace("first")
            .build()
            .run(async {
                openai_agents_sdk::tracing::span(SpanData::Agent {
                    name: "first-parent".into(),
                    handoffs: None,
                    tools: None,
                    output_type: None,
                })
                .span_id(SHARED_SPAN_ID)
                .build()
                .run(async {
                    openai_agents_sdk::tracing::custom_span("first-child", serde_json::Value::Null)
                        .run(async {})
                        .await;
                })
                .await;
            })
            .await;

        openai_agents_sdk::tracing::trace("second")
            .build()
            .run(async {
                openai_agents_sdk::tracing::agent_span("second-parent", None, None, None)
                    .run(async {
                        openai_agents_sdk::tracing::span(SpanData::Custom {
                            name: "second-child".into(),
                            data: serde_json::Value::Null,
                        })
                        .span_id(SHARED_SPAN_ID)
                        .build()
                        .run(async {})
                        .await;
                    })
                    .await;
            })
            .await;
    })
    .await;

    let ordered = processor.ordered();
    assert_eq!(ordered.len(), 4);
    // A span id is unique within a trace only: keying the start order on the id alone would give
    // the second trace's child the first trace's position and push it ahead of its own parent.
    let first_trace = &ordered[0].0;
    assert_eq!(&ordered[1].0, first_trace);
    assert_eq!(ordered[0].1, SHARED_SPAN_ID);
    assert_ne!(&ordered[2].0, first_trace);
    assert_eq!(ordered[3].1, SHARED_SPAN_ID);
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_trace_context.py
// ---------------------------------------------------------------------------------------------

fn new_trace_id() -> String {
    openai_agents_sdk::tracing::gen_trace_id()
}

/// Start and finish a trace so its id counts as started in this process, and snapshot it.
fn mark_trace_as_started(
    workflow_name: &str,
    group_id: Option<&str>,
    metadata: serde_json::Value,
    tracing_api_key: Option<&str>,
) -> TraceState {
    let mut builder = openai_agents_sdk::tracing::trace(workflow_name)
        .trace_id(new_trace_id())
        .metadata(metadata.as_object().unwrap().clone());
    if let Some(g) = group_id {
        builder = builder.group_id(g);
    }
    if let Some(k) = tracing_api_key {
        builder = builder.tracing_api_key(k);
    }
    let original = builder.build();
    original.start();
    original.finish();
    TraceState::from_trace(Some(&original)).unwrap()
}

fn metadata_map(value: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    value.as_object().unwrap().clone()
}

#[test]
fn create_trace_for_run_reattaches_a_matching_started_trace() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), Some("trace-key"));

    let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
        "workflow",
        state.trace_id.clone(),
        state.group_id.clone(),
        Some(metadata_map(&meta)),
        Some("trace-key".into()),
        false,
        Some(&state),
        true,
    )
    .unwrap();

    assert!(created.is_reattached());
    assert_eq!(Some(created.trace_id().to_string()), state.trace_id);
}

#[test]
fn create_trace_for_run_does_not_reattach_when_the_trace_id_differs() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), None);

    let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
        "workflow",
        Some(new_trace_id()),
        state.group_id.clone(),
        Some(metadata_map(&meta)),
        None,
        false,
        Some(&state),
        true,
    )
    .unwrap();

    assert!(!created.is_reattached());
}

#[test]
fn create_trace_for_run_reattaches_a_stripped_key_with_a_matching_resume_key() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), Some("trace-key"));
    let stripped = TraceState::from_json(state.to_json(false).as_ref()).unwrap();
    assert_eq!(stripped.tracing_api_key, None);
    assert_eq!(stripped.tracing_api_key_hash, state.tracing_api_key_hash);

    let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
        "workflow",
        stripped.trace_id.clone(),
        stripped.group_id.clone(),
        Some(metadata_map(&meta)),
        Some("trace-key".into()),
        false,
        Some(&stripped),
        true,
    )
    .unwrap();

    assert!(created.is_reattached());
    assert_eq!(created.tracing_api_key(), Some("trace-key"));
}

#[test]
fn create_trace_for_run_does_not_reattach_a_stripped_key_on_a_mismatch() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), Some("trace-key"));
    let stripped = TraceState::from_json(state.to_json(false).as_ref()).unwrap();

    let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
        "workflow",
        stripped.trace_id.clone(),
        stripped.group_id.clone(),
        Some(metadata_map(&meta)),
        Some("other-trace-key".into()),
        false,
        Some(&stripped),
        true,
    )
    .unwrap();

    assert!(!created.is_reattached());
}

#[test]
fn create_trace_for_run_does_not_reattach_when_the_settings_differ() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), Some("trace-key"));
    let other = serde_json::json!({"key": "override"});

    let cases: Vec<(&str, Option<String>, serde_json::Value, Option<String>)> = vec![
        (
            "workflow-override",
            state.group_id.clone(),
            meta.clone(),
            Some("trace-key".into()),
        ),
        (
            "workflow",
            Some("group-override".into()),
            meta.clone(),
            Some("trace-key".into()),
        ),
        (
            "workflow",
            state.group_id.clone(),
            other,
            Some("trace-key".into()),
        ),
        (
            "workflow",
            state.group_id.clone(),
            meta.clone(),
            Some("other-trace-key".into()),
        ),
    ];

    for (workflow_name, group_id, metadata, tracing_api_key) in cases {
        let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
            workflow_name,
            state.trace_id.clone(),
            group_id,
            Some(metadata_map(&metadata)),
            tracing_api_key,
            false,
            Some(&state),
            true,
        )
        .unwrap();
        assert!(
            !created.is_reattached(),
            "{workflow_name} should not reattach"
        );
    }
}

#[test]
fn create_trace_for_run_respects_the_disabled_flag_for_a_resume() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), None);

    let created = openai_agents_sdk::tracing::create_trace_for_run_with_state(
        "workflow",
        state.trace_id.clone(),
        state.group_id.clone(),
        Some(metadata_map(&meta)),
        None,
        true,
        Some(&state),
        true,
    )
    .unwrap();

    assert!(created.is_disabled());
    assert!(!created.is_reattached());
}

#[tokio::test]
async fn create_trace_for_run_joins_an_existing_current_trace() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "value"});
    let state = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), None);

    let outer = openai_agents_sdk::tracing::trace("outer")
        .trace_id(new_trace_id())
        .build();
    let created = outer
        .run(async {
            openai_agents_sdk::tracing::create_trace_for_run_with_state(
                "workflow",
                state.trace_id.clone(),
                state.group_id.clone(),
                Some(metadata_map(&meta)),
                None,
                false,
                Some(&state),
                true,
            )
        })
        .await;

    assert!(created.is_none());
}

/// Python monkeypatches `_MAX_STARTED_TRACE_IDS` down to 2; the Rust bound is the compiled-in
/// 4096, so the eviction is proven by overflowing it.
#[test]
fn the_started_trace_id_cache_is_bounded() {
    let _lock = tracing_lock();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    openai_agents_sdk::tracing::reset_trace_provider();
    let meta = serde_json::json!({"key": "first"});
    let first = mark_trace_as_started("workflow", Some("group-1"), meta.clone(), None);
    assert!(openai_agents_sdk::tracing::trace_id_was_started(
        first.trace_id.as_deref()
    ));

    for _ in 0..4096 {
        openai_agents_sdk::tracing::trace("filler")
            .trace_id(new_trace_id())
            .build()
            .start();
    }

    assert!(!openai_agents_sdk::tracing::trace_id_was_started(
        first.trace_id.as_deref()
    ));
}

/// `trace_id_was_started` ignores the reserved no-op id and an empty one.
#[test]
fn the_started_trace_id_cache_ignores_reserved_ids() {
    assert!(!openai_agents_sdk::tracing::trace_id_was_started(Some(
        "no-op"
    )));
    assert!(!openai_agents_sdk::tracing::trace_id_was_started(Some("")));
    assert!(!openai_agents_sdk::tracing::trace_id_was_started(None));
}

// ---------------------------------------------------------------------------------------------
// tests/tracing/test_setup.py
// ---------------------------------------------------------------------------------------------

#[test]
fn the_default_provider_is_used_until_one_is_installed() {
    let _lock = tracing_lock();
    openai_agents_sdk::tracing::reset_trace_provider();
    let built_in = openai_agents_sdk::tracing::provider();
    let custom = Arc::new(DefaultTraceProvider::new());
    openai_agents_sdk::tracing::set_trace_provider(custom.clone());
    assert!(!Arc::ptr_eq(
        &openai_agents_sdk::tracing::provider(),
        &built_in
    ));
    openai_agents_sdk::tracing::reset_trace_provider();
    assert!(Arc::ptr_eq(
        &openai_agents_sdk::tracing::provider(),
        &built_in
    ));
}

#[test]
fn the_default_processor_is_a_single_shared_instance() {
    let a = default_processor();
    let b = default_processor();
    assert!(Arc::ptr_eq(&a, &b));
}

#[test]
fn the_shutdown_guard_is_the_rust_equivalent_of_the_atexit_hook() {
    let _lock = tracing_lock();
    // Rust has no process-exit hook, so the guard runs the flush when it is dropped.
    let guard = openai_agents_sdk::tracing::shutdown_guard();
    drop(guard);
}

/// The global export key feeds the default exporter, as `set_tracing_export_api_key` does.
#[test]
fn the_global_export_key_is_used_when_no_explicit_key_is_set() {
    /// The key is process-global: clear it when the test ends, even on failure.
    struct ClearExportKey;
    impl Drop for ClearExportKey {
        fn drop(&mut self) {
            set_export_api_key("");
        }
    }

    let _lock = tracing_lock();
    let _clear = ClearExportKey;
    let _env = EnvGuard::set("OPENAI_API_KEY", None);
    let (endpoint, received) = mock_ingest(1);
    set_export_api_key("sk-global-export");
    let exporter = BackendSpanExporter::new().with_endpoint(endpoint);
    exporter.export(vec![item("item", None)]);
    let calls = received.lock().unwrap().clone();
    assert_eq!(
        calls.first().and_then(|c| c.authorization.clone()),
        Some("Bearer sk-global-export".to_string())
    );
}

/// The ingest endpoint constant matches the one the Python exporter posts to.
#[test]
fn the_default_ingest_endpoint_matches_python() {
    assert_eq!(
        OPENAI_TRACING_INGEST_ENDPOINT,
        "https://api.openai.com/v1/traces/ingest"
    );
}
