//! Trace processors and exporters. Mirrors `agents.tracing.processors`.
//!
//! The default [`BatchTraceProcessor`] runs on a dedicated OS thread with its own small tokio
//! runtime, so exports survive the caller's runtime being dropped (for example after
//! `Runner::run_sync`). Rust has no automatic flush at process exit: call
//! [`crate::tracing::flush_traces`] before exiting to make sure buffered traces are sent.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::spans::{Span, Trace};

/// The OpenAI traces ingest endpoint.
pub const OPENAI_TRACING_INGEST_ENDPOINT: &str = "https://api.openai.com/v1/traces/ingest";
const MAX_FIELD_BYTES: usize = 100_000;
const TRUNCATION_SUFFIX: &str = "... [truncated]";

/// Receives trace and span lifecycle events.
pub trait TracingProcessor: Send + Sync + fmt::Debug {
    fn on_trace_start(&self, trace: &Trace);
    fn on_trace_end(&self, trace: &Trace);
    fn on_span_start(&self, span: &Span);
    fn on_span_end(&self, span: &Span);
    /// Export everything buffered and stop.
    fn shutdown(&self) {}
    /// Export everything buffered and wait for completion.
    fn force_flush(&self) {}
}

/// A trace or span ready for export.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportItem {
    /// The exported payload (`object: trace` or `object: trace.span`).
    pub payload: Value,
    /// Per-trace API key override.
    pub api_key: Option<String>,
}

/// Sends batches of exported items somewhere. Called on the processor's background thread.
pub trait TracingExporter: Send + Sync + fmt::Debug {
    fn export(&self, items: Vec<ExportItem>);
}

/// Prints exported items to stdout.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConsoleSpanExporter;

impl TracingExporter for ConsoleSpanExporter {
    fn export(&self, items: Vec<ExportItem>) {
        for item in items {
            match item.payload.get("object").and_then(Value::as_str) {
                Some("trace") => println!(
                    "[Exporter] Export trace_id={}, name={}",
                    item.payload["id"], item.payload["workflow_name"]
                ),
                _ => println!("[Exporter] Export span: {}", item.payload),
            }
        }
    }
}

/// Exports traces and spans to the OpenAI traces backend.
#[derive(Clone)]
pub struct BackendSpanExporter {
    endpoint: String,
    api_key: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    max_retries: u32,
    runtime: Arc<Mutex<Option<tokio::runtime::Runtime>>>,
    http: reqwest::Client,
}

impl fmt::Debug for BackendSpanExporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendSpanExporter")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

impl Default for BackendSpanExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendSpanExporter {
    /// Exporter to the OpenAI traces endpoint using the default API key resolution.
    pub fn new() -> Self {
        BackendSpanExporter {
            endpoint: OPENAI_TRACING_INGEST_ENDPOINT.to_string(),
            api_key: None,
            organization: std::env::var("OPENAI_ORG_ID").ok(),
            project: std::env::var("OPENAI_PROJECT_ID").ok(),
            max_retries: 3,
            runtime: Arc::new(Mutex::new(None)),
            http: crate::util::http_client(),
        }
    }

    /// Use a fixed API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Use a custom endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(super::provider::tracing_export_api_key)
            .or_else(crate::models::default_openai_key)
            .or_else(|| {
                std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())
            })
    }

    fn is_openai_endpoint(&self) -> bool {
        self.endpoint.trim_end_matches('/') == OPENAI_TRACING_INGEST_ENDPOINT
    }

    fn send(&self, api_key: &str, data: Vec<Value>) {
        let body = json!({"data": data});
        let mut guard = self.runtime.lock().expect("exporter runtime poisoned");
        if guard.is_none() {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => *guard = Some(rt),
                Err(e) => {
                    log::error!("[non-fatal] Tracing: failed to build export runtime: {e}");
                    return;
                }
            }
        }
        let rt = guard.as_ref().expect("runtime");
        let mut delay = Duration::from_secs(1);
        for attempt in 0..=self.max_retries {
            let mut req = self
                .http
                .post(&self.endpoint)
                .bearer_auth(api_key)
                .header("OpenAI-Beta", "traces=v1")
                .timeout(Duration::from_secs(30))
                .json(&body);
            if let Some(o) = &self.organization {
                req = req.header("OpenAI-Organization", o);
            }
            if let Some(p) = &self.project {
                req = req.header("OpenAI-Project", p);
            }
            let result = rt.block_on(async move {
                let resp = req.send().await?;
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Ok::<_, reqwest::Error>((status, text))
            });
            match result {
                Ok((status, _)) if status.as_u16() < 300 => {
                    log::debug!("Exported {} items", data_len(&body));
                    return;
                }
                Ok((status, text)) => {
                    let code = status.as_u16();
                    let retryable = code >= 500 || matches!(code, 408 | 409 | 429);
                    if !retryable {
                        log::error!("[non-fatal] Tracing client error {code}: {text}");
                        return;
                    }
                    log::warn!("[non-fatal] Tracing: server error {code}, retrying.");
                }
                Err(e) => log::warn!("[non-fatal] Tracing: request failed: {e}"),
            }
            if attempt < self.max_retries {
                thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_secs(30));
            }
        }
        log::error!("[non-fatal] Tracing: max retries reached, giving up on this batch.");
    }
}

fn data_len(body: &Value) -> usize {
    body["data"].as_array().map(Vec::len).unwrap_or(0)
}

fn truncate_value(value: &Value) -> Value {
    let size = value.to_string().len();
    if size <= MAX_FIELD_BYTES {
        return value.clone();
    }
    match value {
        Value::String(s) => {
            let mut cut = MAX_FIELD_BYTES.saturating_sub(TRUNCATION_SUFFIX.len() + 2);
            while cut > 0 && !s.is_char_boundary(cut) {
                cut -= 1;
            }
            Value::String(format!("{}{}", &s[..cut.min(s.len())], TRUNCATION_SUFFIX))
        }
        other => {
            let s = other.to_string();
            let mut cut = MAX_FIELD_BYTES.saturating_sub(TRUNCATION_SUFFIX.len() + 2);
            while cut > 0 && !s.is_char_boundary(cut) {
                cut -= 1;
            }
            Value::String(format!("{}{}", &s[..cut], TRUNCATION_SUFFIX))
        }
    }
}

/// Enforce OpenAI traces ingest limits: truncate large fields and keep usage only on
/// generation spans.
pub(crate) fn sanitize_for_openai(mut payload: Value) -> Value {
    let Some(span_data) = payload.get_mut("span_data").and_then(Value::as_object_mut) else {
        return payload;
    };
    for field in ["input", "output"] {
        if let Some(v) = span_data.get(field) {
            let t = truncate_value(v);
            span_data.insert(field.into(), t);
        }
    }
    let is_generation = span_data.get("type").and_then(Value::as_str) == Some("generation");
    if !is_generation {
        span_data.remove("usage");
    } else if let Some(usage) = span_data.get("usage").cloned() {
        let valid = usage
            .get("input_tokens")
            .map(Value::is_number)
            .unwrap_or(false)
            && usage
                .get("output_tokens")
                .map(Value::is_number)
                .unwrap_or(false);
        if !valid {
            span_data.remove("usage");
        }
    }
    payload
}

impl TracingExporter for BackendSpanExporter {
    fn export(&self, items: Vec<ExportItem>) {
        let mut grouped: BTreeMap<Option<String>, Vec<Value>> = BTreeMap::new();
        for item in items {
            let payload = if self.is_openai_endpoint() {
                sanitize_for_openai(item.payload)
            } else {
                item.payload
            };
            grouped.entry(item.api_key).or_default().push(payload);
        }
        for (key, data) in grouped {
            let Some(api_key) = key.or_else(|| self.resolve_api_key()) else {
                log::warn!("OPENAI_API_KEY is not set, skipping trace export");
                continue;
            };
            for chunk in data.chunks(100) {
                self.send(&api_key, chunk.to_vec());
            }
        }
    }
}

enum Message {
    Item(ExportItem),
    Kick,
    Flush(mpsc::Sender<()>),
    Shutdown(mpsc::Sender<()>),
}

/// Buffers traces and spans and exports them in batches from a background thread.
pub struct BatchTraceProcessor {
    sender: Mutex<Option<mpsc::Sender<Message>>>,
    exporter: Arc<dyn TracingExporter>,
    max_queue_size: usize,
    max_batch_size: usize,
    schedule_delay: Duration,
    export_trigger_size: usize,
}

impl fmt::Debug for BatchTraceProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTraceProcessor")
            .field("exporter", &self.exporter)
            .field("max_queue_size", &self.max_queue_size)
            .finish_non_exhaustive()
    }
}

impl BatchTraceProcessor {
    /// A batch processor with default limits (queue 8192, batch 128, every 5 seconds).
    pub fn new(exporter: Arc<dyn TracingExporter>) -> Self {
        Self::with_options(exporter, 8192, 128, Duration::from_secs(5), 0.7)
    }

    /// A batch processor with custom limits.
    pub fn with_options(
        exporter: Arc<dyn TracingExporter>,
        max_queue_size: usize,
        max_batch_size: usize,
        schedule_delay: Duration,
        export_trigger_ratio: f64,
    ) -> Self {
        BatchTraceProcessor {
            sender: Mutex::new(None),
            exporter,
            max_queue_size,
            max_batch_size: max_batch_size.max(1),
            schedule_delay,
            export_trigger_size: ((max_queue_size as f64 * export_trigger_ratio) as usize).max(1),
        }
    }

    fn send(&self, msg: Message) -> bool {
        let mut guard = self.sender.lock().expect("processor poisoned");
        if guard.is_none() {
            let (tx, rx) = mpsc::channel::<Message>();
            let exporter = self.exporter.clone();
            let max_queue = self.max_queue_size;
            let max_batch = self.max_batch_size;
            let delay = self.schedule_delay;
            let trigger = self.export_trigger_size;
            let spawned = thread::Builder::new()
                .name("agents-trace-exporter".into())
                .spawn(move || worker(rx, exporter, max_queue, max_batch, delay, trigger));
            if spawned.is_err() {
                log::error!("[non-fatal] Tracing: failed to spawn exporter thread");
                return false;
            }
            *guard = Some(tx);
        }
        guard
            .as_ref()
            .map(|tx| tx.send(msg).is_ok())
            .unwrap_or(false)
    }

    fn wait(&self, make: impl FnOnce(mpsc::Sender<()>) -> Message) {
        if self.sender.lock().expect("processor poisoned").is_none() {
            return;
        }
        let (ack_tx, ack_rx) = mpsc::channel();
        if self.send(make(ack_tx)) {
            let _ = ack_rx.recv_timeout(Duration::from_secs(60));
        }
    }
}

fn worker(
    rx: mpsc::Receiver<Message>,
    exporter: Arc<dyn TracingExporter>,
    max_queue: usize,
    max_batch: usize,
    delay: Duration,
    trigger: usize,
) {
    let mut queue: Vec<ExportItem> = Vec::new();
    let export_all = |queue: &mut Vec<ExportItem>| {
        while !queue.is_empty() {
            let n = queue.len().min(max_batch);
            let batch: Vec<ExportItem> = queue.drain(..n).collect();
            exporter.export(batch);
        }
    };
    let mut next_export = Instant::now() + delay;
    loop {
        let timeout = next_export.saturating_duration_since(Instant::now());
        match rx.recv_timeout(timeout) {
            Ok(Message::Item(item)) => {
                if queue.len() < max_queue {
                    queue.push(item);
                } else {
                    log::warn!("Queue is full, dropping trace item.");
                }
                if queue.len() >= trigger {
                    export_all(&mut queue);
                    next_export = Instant::now() + delay;
                }
            }
            Ok(Message::Kick) => {
                export_all(&mut queue);
                next_export = Instant::now() + delay;
            }
            Ok(Message::Flush(ack)) => {
                export_all(&mut queue);
                let _ = ack.send(());
            }
            Ok(Message::Shutdown(ack)) => {
                export_all(&mut queue);
                let _ = ack.send(());
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                export_all(&mut queue);
                next_export = Instant::now() + delay;
            }
            Err(RecvTimeoutError::Disconnected) => {
                export_all(&mut queue);
                return;
            }
        }
    }
}

impl TracingProcessor for BatchTraceProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        if let Some(payload) = trace.export() {
            self.send(Message::Item(ExportItem {
                payload,
                api_key: trace.tracing_api_key().map(str::to_string),
            }));
        }
    }

    fn on_trace_end(&self, _trace: &Trace) {
        self.send(Message::Kick);
    }

    fn on_span_start(&self, _span: &Span) {}

    fn on_span_end(&self, span: &Span) {
        if let Some(payload) = span.export() {
            self.send(Message::Item(ExportItem {
                payload,
                api_key: span.tracing_api_key().map(str::to_string),
            }));
        }
    }

    fn shutdown(&self) {
        self.wait(Message::Shutdown);
        *self.sender.lock().expect("processor poisoned") = None;
    }

    fn force_flush(&self) {
        self.wait(Message::Flush);
    }
}

/// A processor that records exported payloads in memory (useful for tests).
#[derive(Debug, Default)]
pub struct InMemoryTraceProcessor {
    items: Mutex<Vec<Value>>,
}

impl InMemoryTraceProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exported traces and spans, in order of arrival.
    pub fn items(&self) -> Vec<Value> {
        self.items.lock().expect("poisoned").clone()
    }

    /// Clear recorded items.
    pub fn clear(&self) {
        self.items.lock().expect("poisoned").clear();
    }
}

impl TracingProcessor for InMemoryTraceProcessor {
    fn on_trace_start(&self, trace: &Trace) {
        if let Some(p) = trace.export() {
            self.items.lock().expect("poisoned").push(p);
        }
    }
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let Some(p) = span.export() {
            self.items.lock().expect("poisoned").push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_usage_and_truncates() {
        let p = json!({"span_data": {"type": "function", "usage": {"a": 1}, "input": "x".repeat(200_000)}});
        let s = sanitize_for_openai(p);
        assert!(s["span_data"].get("usage").is_none());
        let input = s["span_data"]["input"].as_str().unwrap();
        assert!(input.len() <= MAX_FIELD_BYTES && input.ends_with(TRUNCATION_SUFFIX));
        let g = json!({"span_data": {"type": "generation", "usage": {"input_tokens": 1, "output_tokens": 2}}});
        assert!(sanitize_for_openai(g)["span_data"]["usage"].is_object());
    }

    #[derive(Debug, Default)]
    struct Collect(Mutex<Vec<ExportItem>>);
    impl TracingExporter for Collect {
        fn export(&self, items: Vec<ExportItem>) {
            self.0.lock().unwrap().extend(items);
        }
    }

    #[test]
    fn batch_processor_flushes_on_demand() {
        let exporter = Arc::new(Collect::default());
        let p = BatchTraceProcessor::with_options(
            exporter.clone(),
            100,
            10,
            Duration::from_secs(3600),
            0.9,
        );
        let t = Trace::new("w".into(), None, None, None, None, false);
        p.on_trace_start(&t);
        p.force_flush();
        assert_eq!(exporter.0.lock().unwrap().len(), 1);
        p.shutdown();
    }
}
