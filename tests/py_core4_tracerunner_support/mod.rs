//! Shared harness for the core3 tracing ports. Each test binary pulls it in with
//! `#[path = "py_core3_tracing_support/mod.rs"] mod support;`.
//!
//! Tracing state (the installed provider, processors, the export key and the disable switch) is
//! process-global, so every test that touches it takes [`lock`] first. One lock serves both sync
//! and async tests, so they never interleave.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

use openai_agents_sdk::tracing::{
    Span, SpanData, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled, span,
    trace,
};
use serde_json::{Value, json};

/// Take the process-wide tracing lock from synchronous code.
/// The one lock both sync and async tests take; two locks would not exclude each other.
fn shared_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Take the process-wide tracing lock from synchronous code.
pub fn lock() -> tokio::sync::MutexGuard<'static, ()> {
    shared_lock().blocking_lock()
}

/// Take the process-wide tracing lock from an async test.
pub async fn lock_async() -> tokio::sync::MutexGuard<'static, ()> {
    shared_lock().lock().await
}

/// Remove `OPENAI_API_KEY` for the test and restore it afterwards, so the default exporter can
/// never post to the real endpoint with a developer's key.
pub struct NoApiKey {
    previous: Option<String>,
}

impl NoApiKey {
    pub fn new() -> Self {
        let previous = std::env::var("OPENAI_API_KEY").ok();
        // Safety: every caller holds the tracing lock, so no other test reads the environment.
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        NoApiKey { previous }
    }
}

impl Drop for NoApiKey {
    fn drop(&mut self) {
        if let Some(value) = &self.previous {
            // Safety: see `NoApiKey::new`.
            unsafe { std::env::set_var("OPENAI_API_KEY", value) };
        }
    }
}

/// Turn tool data logging on or off for one test. Callers reset it with
/// [`reset_tool_data_logging`] when done.
pub fn set_tool_data_logging(log: bool) {
    openai_agents_sdk::debug::set_dont_log_tool_data(!log);
}

/// Restore the environment-driven tool data policy.
pub fn reset_tool_data_logging() {
    openai_agents_sdk::debug::reset_dont_log_tool_data();
}

/// Install the processors for one test and restore the default pipeline on drop.
pub struct Pipeline;

impl Pipeline {
    pub fn install(processors: Vec<Arc<dyn TracingProcessor>>) -> Self {
        set_tracing_disabled(false);
        set_trace_processors(processors);
        Pipeline
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        set_trace_processors(Vec::new());
        openai_agents_sdk::tracing::provider::reset_trace_provider();
    }
}

// ---------------------------------------------------------------------------------------------
// log capture
// ---------------------------------------------------------------------------------------------

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

/// Route `log` output into the in-memory buffer read by [`logged_text`]. Idempotent.
pub fn capture_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// Forget everything logged so far.
pub fn clear_logs() {
    RECORDS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// All captured messages, newline separated, as Python's `caplog.text`.
pub fn logged_text() -> String {
    RECORDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(_, m)| m.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------------------------
// traces and spans
// ---------------------------------------------------------------------------------------------

/// A minimal, unstarted trace named `name`.
pub fn new_trace(name: &str) -> Trace {
    trace(name).build()
}

/// A span with `data` under `parent`.
pub fn new_span(parent: &Trace, data: SpanData) -> Span {
    span(data).parent_trace(parent.clone()).build()
}

/// The minimal agent span payload used across the processor tests.
pub fn agent_data(name: &str) -> SpanData {
    SpanData::Agent {
        name: name.to_string(),
        handoffs: None,
        tools: None,
        output_type: None,
    }
}

// ---------------------------------------------------------------------------------------------
// mock ingest
// ---------------------------------------------------------------------------------------------

/// One scripted HTTP reply: a status and extra header lines (each ending in `\r\n`).
#[derive(Clone)]
pub struct Reply {
    pub status: u16,
    pub headers: &'static str,
    pub body: &'static str,
}

impl Reply {
    pub fn new(status: u16) -> Self {
        Reply {
            status,
            headers: "",
            body: "{}",
        }
    }

    pub fn with_headers(status: u16, headers: &'static str) -> Self {
        Reply {
            status,
            headers,
            body: "{}",
        }
    }

    pub fn with_body(status: u16, body: &'static str) -> Self {
        Reply {
            status,
            headers: "",
            body,
        }
    }
}

/// A local HTTP server standing in for the traces ingest endpoint.
///
/// Replies are served in order; the last one repeats for any further request. Every request body
/// is recorded as parsed JSON.
pub struct Ingest {
    pub endpoint: String,
    pub bodies: Arc<Mutex<Vec<Value>>>,
    pub requests: Arc<AtomicUsize>,
}

impl Ingest {
    pub fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(AtomicUsize::new(0));
        let (b, r) = (bodies.clone(), requests.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut sock) = stream else { return };
                let body = read_request_body(&mut sock);
                let index = r.fetch_add(1, Ordering::SeqCst);
                if let Some(parsed) = body {
                    b.lock().unwrap_or_else(|e| e.into_inner()).push(parsed);
                }
                let reply = replies
                    .get(index)
                    .or_else(|| replies.last())
                    .cloned()
                    .unwrap_or(Reply::new(200));
                let reason = if reply.status < 300 { "OK" } else { "Error" };
                let resp = format!(
                    "HTTP/1.1 {} {reason}\r\ncontent-type: application/json\r\n{}content-length: {}\r\nconnection: close\r\n\r\n{}",
                    reply.status,
                    reply.headers,
                    reply.body.len(),
                    reply.body
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        Ingest {
            endpoint: format!("http://{addr}/v1/traces/ingest"),
            bodies,
            requests,
        }
    }

    /// Requests received so far.
    pub fn count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    /// The `data` array of every request, in arrival order.
    pub fn data(&self) -> Vec<Vec<Value>> {
        self.bodies
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|b| b["data"].as_array().cloned().unwrap_or_default())
            .collect()
    }

    /// Every exported item, flattened in arrival order.
    pub fn items(&self) -> Vec<Value> {
        self.data().into_iter().flatten().collect()
    }
}

fn read_request_body(sock: &mut std::net::TcpStream) -> Option<Value> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).ok()?;
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
                        .map(|v| v.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            if buf.len() >= idx + 4 + len {
                let body = &buf[idx + 4..idx + 4 + len];
                return serde_json::from_slice(body).ok();
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// span recorder and normalizer (port of tests/testing_processor.py)
// ---------------------------------------------------------------------------------------------

use std::collections::HashMap;

use serde_json::Map;

/// Stores finished spans and started traces in memory, as Python's `SpanProcessorForTests` does.
///
/// Spans are ordered by start time, then by the order they started, so a parent that starts first
/// sorts before a child even when both carry the same timestamp.
#[derive(Default)]
pub struct SpanRecorder {
    traces: Mutex<Vec<Trace>>,
    spans: Mutex<Vec<Span>>,
    start_order: Mutex<HashMap<(String, String), usize>>,
    events: Mutex<Vec<&'static str>>,
}

impl std::fmt::Debug for SpanRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanRecorder").finish_non_exhaustive()
    }
}

impl SpanRecorder {
    pub fn new() -> Arc<Self> {
        Arc::new(SpanRecorder::default())
    }

    pub fn clear(&self) {
        self.traces
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.spans.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.start_order
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// The lifecycle events seen so far, in order (`trace_start`, `span_start`, ...).
    pub fn events(&self) -> Vec<&'static str> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Exported traces, in the order they started. Empty traces are skipped.
    pub fn traces(&self) -> Vec<Trace> {
        self.traces
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|t| t.export().is_some())
            .cloned()
            .collect()
    }

    /// Exported spans in start order. Empty (disabled) spans are skipped.
    pub fn ordered_spans(&self) -> Vec<Span> {
        let order = self
            .start_order
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut spans: Vec<Span> = self
            .spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|s| s.export().is_some())
            .cloned()
            .collect();
        spans.sort_by_key(|s| {
            (
                s.started_at().unwrap_or_default(),
                order
                    .get(&(s.trace_id().to_string(), s.span_id().to_string()))
                    .copied()
                    .unwrap_or(0),
            )
        });
        spans
    }

    /// The span tree, as Python's `fetch_normalized_spans` builds it.
    pub fn normalized(&self, keep_span_id: bool, keep_trace_id: bool) -> Vec<Value> {
        normalized_spans(self, keep_span_id, keep_trace_id)
    }
}

impl TracingProcessor for SpanRecorder {
    fn on_trace_start(&self, trace: &Trace) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push("trace_start");
        self.traces
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(trace.clone());
    }
    fn on_trace_end(&self, _trace: &Trace) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push("trace_end");
    }
    fn on_span_start(&self, span: &Span) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push("span_start");
        let mut order = self.start_order.lock().unwrap_or_else(|e| e.into_inner());
        let next = order.len();
        order
            .entry((span.trace_id().to_string(), span.span_id().to_string()))
            .or_insert(next);
    }
    fn on_span_end(&self, span: &Span) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push("span_end");
        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(span.clone());
    }
}

/// Build the normalized span tree from a recorder. Port of `fetch_normalized_spans`.
pub fn normalized_spans(
    recorder: &SpanRecorder,
    keep_span_id: bool,
    keep_trace_id: bool,
) -> Vec<Value> {
    struct Node {
        fields: Map<String, Value>,
        children: Vec<usize>,
    }
    let mut arena: Vec<Node> = Vec::new();
    // (trace id, span id or None for the trace itself) -> arena index. A task or turn span maps
    // to its parent's node, so its children attach to the parent.
    let mut nodes: HashMap<(String, Option<String>), usize> = HashMap::new();
    let mut roots = Vec::new();
    for trace in recorder.traces() {
        let mut fields = trace
            .export()
            .expect("exported trace")
            .as_object()
            .cloned()
            .unwrap_or_default();
        fields.remove("object");
        if !keep_trace_id {
            fields.remove("id");
        }
        fields.retain(|_, v| !v.is_null());
        arena.push(Node {
            fields,
            children: Vec::new(),
        });
        let index = arena.len() - 1;
        nodes.insert((trace.trace_id().to_string(), None), index);
        roots.push(index);
    }
    assert!(
        !roots.is_empty(),
        "Use assert_no_traces() to check for empty traces"
    );

    for span in recorder.ordered_spans() {
        let exported = span.export().expect("exported span");
        let mut map = exported.as_object().cloned().unwrap_or_default();
        map.remove("object");
        if !keep_span_id {
            map.remove("id");
        }
        map.remove("started_at");
        map.remove("ended_at");
        let parent_id = map
            .remove("parent_id")
            .and_then(|v| v.as_str().map(str::to_string));
        let mut span_data = map
            .remove("span_data")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        let span_type = span_data
            .remove("type")
            .and_then(|v| v.as_str().map(str::to_string))
            .expect("span type");
        let mut fields = Map::new();
        fields.insert("type".into(), Value::String(span_type.clone()));
        fields.extend(map.into_iter().filter(|(_, v)| !v.is_null()));
        span_data.retain(|_, v| !v.is_null());
        if !span_data.is_empty() {
            fields.insert("data".into(), Value::Object(span_data.clone()));
        }
        let trace_id = fields
            .remove("trace_id")
            .and_then(|v| v.as_str().map(str::to_string))
            .expect("trace id");
        let sdk_span_type = if span_type == "custom" {
            span_data
                .get("data")
                .and_then(|d| d.get("sdk_span_type"))
                .and_then(Value::as_str)
                .map(str::to_string)
        } else {
            None
        };
        let span_id = span.span_id().to_string();
        if matches!(span_type.as_str(), "task" | "turn")
            || matches!(sdk_span_type.as_deref(), Some("task" | "turn"))
        {
            let parent = *nodes
                .get(&(trace_id.clone(), parent_id.clone()))
                .expect("parent node");
            if let Some(error) = fields.get("error").cloned()
                && !arena[parent].fields.contains_key("error")
            {
                arena[parent].fields.insert("error".into(), error);
            }
            nodes.insert((trace_id, Some(span_id)), parent);
            continue;
        }
        arena.push(Node {
            fields,
            children: Vec::new(),
        });
        let index = arena.len() - 1;
        nodes.insert((trace_id.clone(), Some(span_id)), index);
        let parent = *nodes.get(&(trace_id, parent_id)).expect("parent node");
        arena[parent].children.push(index);
    }

    fn build(arena: &[Node], index: usize) -> Value {
        let mut map = arena[index].fields.clone();
        if !arena[index].children.is_empty() {
            let children: Vec<Value> = arena[index]
                .children
                .iter()
                .map(|c| build(arena, *c))
                .collect();
            map.insert("children".into(), Value::Array(children));
        }
        Value::Object(map)
    }
    roots.iter().map(|r| build(&arena, *r)).collect()
}

// ---------------------------------------------------------------------------------------------
// scripted HTTP model endpoint (Responses and Chat Completions)
// ---------------------------------------------------------------------------------------------

/// One scripted reply of the model endpoint.
#[derive(Clone)]
pub enum ModelReply {
    /// 200 with a JSON body.
    Json(Value),
    /// Waits `Duration` before answering `{}`, so a client timeout fires first.
    Stall(std::time::Duration),
    /// 200 SSE: each event, then `[DONE]`.
    Sse(Vec<Value>),
    /// An HTTP error with a JSON body.
    Status(u16, Value),
}

/// A local HTTP server standing in for the model API. Replies are served in order, the last one
/// repeating. Every request body is recorded as parsed JSON.
pub struct ModelEndpoint {
    pub base_url: String,
    pub bodies: Arc<Mutex<Vec<Value>>>,
}

fn sse_text(events: &[Value]) -> String {
    let mut body = String::new();
    for e in events {
        body.push_str(&format!(
            "event: {}\ndata: {}\n\n",
            e["type"].as_str().unwrap_or("message"),
            e
        ));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// Read one HTTP request (head and Content-Length body) from `sock`.
fn read_http_request(sock: &mut impl Read) -> Option<Value> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        let Some(idx) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&buf[..idx]).to_string();
        let len: usize = head
            .lines()
            .find_map(|l| {
                l.to_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse().unwrap_or(0))
            })
            .unwrap_or(0);
        if buf.len() >= idx + 4 + len {
            return Some(
                serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null),
            );
        }
    }
}

/// Start a [`ModelEndpoint`] that answers in order.
pub fn start_model_endpoint(replies: Vec<ModelReply>) -> ModelEndpoint {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let b = bodies.clone();
    std::thread::spawn(move || {
        let mut index = 0usize;
        for stream in listener.incoming() {
            let Ok(mut sock) = stream else { return };
            let Some(body) = read_http_request(&mut sock) else {
                continue;
            };
            b.lock().unwrap_or_else(|e| e.into_inner()).push(body);
            let reply = replies
                .get(index)
                .or_else(|| replies.last())
                .cloned()
                .unwrap_or(ModelReply::Json(json!({})));
            index += 1;
            let resp = match reply {
                ModelReply::Json(v) => {
                    let text = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Sse(events) => {
                    let text = sse_text(&events);
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Stall(wait) => {
                    std::thread::sleep(wait);
                    let text = "{}";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Status(code, v) => {
                    let text = v.to_string();
                    format!(
                        "HTTP/1.1 {code} Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
            };
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    ModelEndpoint {
        base_url: format!("http://{addr}/v1"),
        bodies,
    }
}

/// A Responses API response object, as the provider returns it. `usage` is left out when `None`.
pub fn responses_object(id: &str, status: &str, text: Option<&str>, usage: Option<Value>) -> Value {
    let output = match text {
        Some(text) => json!([{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }]),
        None => json!([]),
    };
    json!({
        "id": id, "object": "response", "created_at": 0, "model": "test-model",
        "status": status, "output": output, "tool_choice": "none", "tools": [],
        "parallel_tool_calls": false, "top_p": null, "usage": usage
    })
}

/// The usage object of the Python `DummyUsage` default: 1 in, 1 out, 2 total, nothing cached.
pub fn dummy_usage() -> Value {
    json!({
        "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0}
    })
}

/// A TLS [`ModelEndpoint`] for `api.openai.com`. The client resolves that host to this listener
/// and accepts its throwaway certificate, so the SDK sees the official endpoint while nothing
/// leaves the machine.
pub struct TlsModelEndpoint {
    pub addr: std::net::SocketAddr,
    pub bodies: Arc<Mutex<Vec<Value>>>,
    /// The throwaway server certificate (DER), for clients that trust it as a root.
    pub cert_der: Vec<u8>,
    _certs: tempfile::TempDir,
}

/// A self-signed certificate for `api.openai.com`, generated into a fresh temporary directory.
/// Nothing is committed: the key only lives for the duration of one test.
fn throwaway_certificate() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let key = dir.path().join("key.pem");
    let cert = dir.path().join("cert.pem");
    let status = std::process::Command::new("openssl")
        .args([
            "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
        ])
        .arg("/CN=api.openai.com")
        .args(["-addext", "subjectAltName=DNS:api.openai.com", "-keyout"])
        .arg(&key)
        .arg("-out")
        .arg(&cert)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("openssl must be installed to generate the test certificate");
    assert!(status.success(), "openssl req failed");
    (dir, cert, key)
}

/// Start a [`TlsModelEndpoint`] that answers in order, like [`start_model_endpoint`].
pub fn start_tls_model_endpoint(replies: Vec<ModelReply>) -> TlsModelEndpoint {
    let (certs_dir, config, cert_der) = tls_server_material();

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let b = bodies.clone();
    std::thread::spawn(move || {
        let mut index = 0usize;
        for stream in listener.incoming() {
            let Ok(sock) = stream else { return };
            let Ok(conn) = rustls::ServerConnection::new(config.clone()) else {
                return;
            };
            let mut tls = rustls::StreamOwned::new(conn, sock);
            let Some(body) = read_http_request(&mut tls) else {
                continue;
            };
            b.lock().unwrap_or_else(|e| e.into_inner()).push(body);
            let reply = replies
                .get(index)
                .or_else(|| replies.last())
                .cloned()
                .unwrap_or(ModelReply::Json(json!({})));
            index += 1;
            let resp = match reply {
                ModelReply::Json(v) => {
                    let text = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Sse(events) => {
                    let text = sse_text(&events);
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Stall(wait) => {
                    std::thread::sleep(wait);
                    let text = "{}";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
                ModelReply::Status(code, v) => {
                    let text = v.to_string();
                    format!(
                        "HTTP/1.1 {code} Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
                        text.len()
                    )
                }
            };
            let _ = tls.write_all(resp.as_bytes());
            tls.conn.send_close_notify();
            let _ = tls.flush();
        }
    });
    TlsModelEndpoint {
        addr,
        bodies,
        cert_der,
        _certs: certs_dir,
    }
}

/// An OpenAI client whose base URL is the official `https://api.openai.com/v1`, with requests
/// routed to `endpoint` through a DNS override and a trust override for its throwaway certificate.
pub fn official_client(endpoint: &TlsModelEndpoint) -> openai_agents_sdk::models::OpenAIClient {
    openai_agents_sdk::models::OpenAIClient::new("sk-test")
        .with_base_url("https://api.openai.com/v1")
        .with_http_client(official_http_client(endpoint))
}

/// A plain HTTP client that sends `api.openai.com` to `endpoint`, trusting its throwaway
/// certificate. For SDK components that take a `reqwest::Client` rather than a model client.
pub fn official_http_client(endpoint: &TlsModelEndpoint) -> reqwest::Client {
    reqwest::Client::builder()
        .resolve("api.openai.com", endpoint.addr)
        .danger_accept_invalid_certs(true)
        .build()
        .expect("reqwest client")
}

/// Whether the `openssl` CLI is available. The TLS tests generate a throwaway certificate with it
/// and skip, with a message, when it is missing.
pub fn openssl_available() -> bool {
    std::process::Command::new("openssl")
        .arg("version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The message a TLS test prints when it skips for want of `openssl`.
pub const OPENSSL_SKIP_MESSAGE: &str =
    "skipped: openssl is not on PATH; this test needs it to generate a throwaway TLS certificate";

/// A server TLS configuration for `api.openai.com` with a throwaway certificate, the temporary
/// directory that holds it, and the certificate's DER bytes.
fn tls_server_material() -> (tempfile::TempDir, Arc<rustls::ServerConfig>, Vec<u8>) {
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};

    let (certs_dir, cert_path, key_path) = throwaway_certificate();
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&cert_path)
        .expect("cert file")
        .collect::<std::result::Result<_, _>>()
        .expect("cert pem");
    let cert_der = certs[0].as_ref().to_vec();
    let key = PrivateKeyDer::from_pem_file(&key_path).expect("key pem");
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("protocol versions")
    .with_no_client_auth()
    .with_single_cert(certs, key)
    .expect("server certificate");
    (certs_dir, Arc::new(config), cert_der)
}

/// A client TLS configuration that trusts only `endpoint`'s throwaway certificate.
pub fn trusting_client_config(endpoint: &TlsModelEndpoint) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(rustls::pki_types::CertificateDer::from(
            endpoint.cert_der.clone(),
        ))
        .expect("trust the throwaway certificate");
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth();
    Arc::new(config)
}

/// A secure WebSocket server with a throwaway certificate. Each connection records the first text
/// frame it receives into `bodies`, then sends `events` as text frames and waits for the client.
#[cfg(feature = "websocket")]
pub fn start_tls_ws_endpoint(events: Vec<Value>) -> TlsModelEndpoint {
    use tokio_tungstenite::tungstenite::{Message, accept};

    let (certs_dir, config, cert_der) = tls_server_material();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorded = bodies.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(sock) = stream else { return };
            let Ok(conn) = rustls::ServerConnection::new(config.clone()) else {
                return;
            };
            let events = events.clone();
            let recorded = recorded.clone();
            std::thread::spawn(move || {
                let tls = rustls::StreamOwned::new(conn, sock);
                let Ok(mut ws) = accept(tls) else { return };
                loop {
                    match ws.read() {
                        Ok(Message::Text(text)) => {
                            let frame = serde_json::from_str(&text).unwrap_or(Value::Null);
                            recorded
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push(frame);
                            break;
                        }
                        Ok(_) => continue,
                        Err(_) => return,
                    }
                }
                for event in &events {
                    if ws.send(Message::text(event.to_string())).is_err() {
                        return;
                    }
                }
                while ws.read().is_ok() {}
            });
        }
    });
    TlsModelEndpoint {
        addr,
        bodies,
        cert_der,
        _certs: certs_dir,
    }
}
