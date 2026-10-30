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

/// The live trace or span an [`ExportItem`] was built from.
#[derive(Debug, Clone)]
pub enum ExportedObject {
    /// A trace, from `on_trace_start`.
    Trace(Trace),
    /// A finished span, from `on_span_end`.
    Span(Span),
}

/// A trace or span ready for export.
///
/// `payload` is the wire format; `object` is the live object it came from, so a custom
/// exporter can read typed span data (`SpanData`, the recorded error, the parent id) instead of
/// re-parsing JSON. It is `None` for items an application built by hand.
#[derive(Debug, Clone, Default)]
pub struct ExportItem {
    /// The exported payload (`object: trace` or `object: trace.span`).
    pub payload: Value,
    /// Per-trace API key override.
    pub api_key: Option<String>,
    /// The live object behind `payload`. Mirrors Python's `TracingExporter.export(items)`
    /// receiving `Trace | Span` rather than dictionaries.
    pub object: Option<ExportedObject>,
}

/// Two items are the same when they export the same payload under the same key; the live
/// object is a view on that payload, not extra state.
impl PartialEq for ExportItem {
    fn eq(&self, other: &Self) -> bool {
        self.payload == other.payload && self.api_key == other.api_key
    }
}

impl ExportItem {
    /// An item carrying only a payload (no live object).
    pub fn new(payload: Value, api_key: Option<String>) -> Self {
        ExportItem {
            payload,
            api_key,
            object: None,
        }
    }

    /// The span this item came from, when it is a span.
    pub fn span(&self) -> Option<&Span> {
        match &self.object {
            Some(ExportedObject::Span(s)) => Some(s),
            _ => None,
        }
    }

    /// The trace this item came from, when it is a trace.
    pub fn trace(&self) -> Option<&Trace> {
        match &self.object {
            Some(ExportedObject::Trace(t)) => Some(t),
            _ => None,
        }
    }

    /// The typed span data, when this item is a span.
    pub fn span_data(&self) -> Option<super::span_data::SpanData> {
        self.span().map(Span::data)
    }
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
    base_delay: Duration,
    max_delay: Duration,
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
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
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

    /// Set the OpenAI organization header.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set the OpenAI project header.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// How many times a failed batch is retried (default 3).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// The first backoff delay, doubled on each retry (default 1s).
    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    /// The backoff ceiling, which also bounds a server-sent `Retry-After` (default 30s).
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
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
                    log::error!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing: failed to build export runtime: {e}");
                    return;
                }
            }
        }
        let rt = guard.as_ref().expect("runtime");
        let mut delay = self.base_delay;
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
                let headers: BTreeMap<String, String> = resp
                    .headers()
                    .iter()
                    .filter_map(|(k, v)| {
                        v.to_str()
                            .ok()
                            .map(|v| (k.as_str().to_lowercase(), v.into()))
                    })
                    .collect();
                let text = resp.text().await.unwrap_or_default();
                Ok::<_, reqwest::Error>((status, headers, text))
            });
            let mut retry_after = None;
            match result {
                Ok((status, _, _)) if status.as_u16() < 300 => {
                    log::debug!(target: crate::logger::LOG_TARGET, "Exported {} items", data_len(&body));
                    return;
                }
                Ok((status, headers, text)) => {
                    let code = status.as_u16();
                    retry_after = crate::retry::parse_retry_after(&headers);
                    // `x-should-retry: false` overrides the status classification, as in the
                    // OpenAI client.
                    let server_allows_retry = headers
                        .get("x-should-retry")
                        .map(|v| !v.trim().eq_ignore_ascii_case("false"))
                        .unwrap_or(true);
                    let retryable =
                        server_allows_retry && (code >= 500 || matches!(code, 408 | 409 | 429));
                    if !retryable {
                        log::error!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing client error {code}: {text}");
                        return;
                    }
                    log::warn!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing: server error {code}, retrying.");
                }
                Err(e) => {
                    log::warn!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing: request failed: {e}")
                }
            }
            if attempt < self.max_retries {
                // Honour the server's wait, bounded by max_delay so a large Retry-After
                // cannot stall the export thread.
                let wait = match retry_after {
                    Some(secs) if secs > 0.0 => {
                        delay.max(Duration::from_secs_f64(secs)).min(self.max_delay)
                    }
                    _ => delay,
                };
                thread::sleep(wait);
                delay = (delay * 2).min(self.max_delay);
            }
        }
        log::error!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing: max retries reached, giving up on this batch.");
    }
}

fn data_len(body: &Value) -> usize {
    body["data"].as_array().map(Vec::len).unwrap_or(0)
}

/// Serialized JSON size of `value`.
fn json_size(value: &Value) -> usize {
    value.to_string().len()
}

fn truncate_string_for_limit(s: &str, max_bytes: usize) -> Value {
    // The two quotes of the serialized string count toward the budget.
    let budget = max_bytes.saturating_sub(TRUNCATION_SUFFIX.len() + 2);
    let mut cut = budget.min(s.len());
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    Value::String(format!("{}{}", &s[..cut], TRUNCATION_SUFFIX))
}

/// Shrink `value` until its serialized form fits in `max_bytes`.
///
/// Mappings and lists are truncated recursively, largest child first, so a span whose input is a
/// list of messages keeps its structure instead of becoming one truncated string. Child sizes are
/// cached and the running total is updated incrementally, so a long input is not re-serialized
/// once per removal.
fn truncate_value_for_limit(value: &Value, max_bytes: usize) -> Value {
    if json_size(value) <= max_bytes {
        return value.clone();
    }
    match value {
        Value::String(s) => truncate_string_for_limit(s, max_bytes),
        Value::Object(map) => {
            let mut truncated = map.clone();
            // `{` + `}` plus, per entry, the quoted key, `:` and a separating `,`.
            let overhead = |keys: &serde_json::Map<String, Value>| {
                2 + keys
                    .keys()
                    .map(|k| json_size(&Value::String(k.clone())) + 1)
                    .sum::<usize>()
                    + keys.len().saturating_sub(1)
            };
            let mut sizes: Vec<(String, usize)> = truncated
                .iter()
                .map(|(k, v)| (k.clone(), json_size(v)))
                .collect();
            let mut current = overhead(&truncated) + sizes.iter().map(|(_, s)| *s).sum::<usize>();
            while !sizes.is_empty() && current > max_bytes {
                let pos = sizes
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, (_, size))| *size)
                    .map(|(i, _)| i)
                    .expect("non-empty");
                let (key, child_size) = sizes[pos].clone();
                let child = truncated[&key].clone();
                let budget = max_bytes.saturating_sub(current - child_size);
                let new_child = truncate_value_for_limit(&child, budget);
                let new_size = json_size(&new_child);
                if new_child == child || new_size >= child_size {
                    truncated.remove(&key);
                    sizes.remove(pos);
                    current = overhead(&truncated) + sizes.iter().map(|(_, s)| *s).sum::<usize>();
                } else {
                    truncated.insert(key, new_child);
                    sizes[pos].1 = new_size;
                    current = current - child_size + new_size;
                }
            }
            Value::Object(truncated)
        }
        Value::Array(items) => {
            let mut truncated: Vec<Option<Value>> = items.iter().cloned().map(Some).collect();
            let mut sizes: Vec<(usize, usize)> = items
                .iter()
                .enumerate()
                .map(|(i, v)| (i, json_size(v)))
                .collect();
            // `[` + `]` plus a separating `,` between entries.
            let overhead = |count: usize| 2 + count.saturating_sub(1);
            let mut current = overhead(sizes.len()) + sizes.iter().map(|(_, s)| *s).sum::<usize>();
            while !sizes.is_empty() && current > max_bytes {
                let pos = sizes
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, (_, size))| *size)
                    .map(|(i, _)| i)
                    .expect("non-empty");
                let (index, child_size) = sizes[pos];
                let child = truncated[index].clone().expect("live entry");
                let budget = max_bytes.saturating_sub(current - child_size);
                let new_child = truncate_value_for_limit(&child, budget);
                let new_size = json_size(&new_child);
                if new_child == child || new_size >= child_size {
                    truncated[index] = None;
                    sizes.remove(pos);
                    current = overhead(sizes.len()) + sizes.iter().map(|(_, s)| *s).sum::<usize>();
                } else {
                    truncated[index] = Some(new_child);
                    sizes[pos].1 = new_size;
                    current = current - child_size + new_size;
                }
            }
            Value::Array(truncated.into_iter().flatten().collect())
        }
        other => truncate_string_for_limit(&other.to_string(), max_bytes),
    }
}

fn truncate_value(value: &Value) -> Value {
    truncate_value_for_limit(value, MAX_FIELD_BYTES)
}

/// Drop reasoning from generation span input/output before export.
///
/// The OpenAI traces dashboard does not render reasoning, and replayed thinking blocks are
/// provider-signed data that does not belong in a trace. Mirrors `_omit_generation_reasoning`;
/// the untouched span is still delivered to custom processors.
fn omit_generation_reasoning(value: &Value) -> Value {
    fn omit_message_reasoning(item: &Value) -> Value {
        let Some(map) = item.as_object() else {
            return item.clone();
        };
        if map.get("role").and_then(Value::as_str) != Some("assistant") {
            return item.clone();
        }
        let mut map = map.clone();
        for key in [
            "reasoning",
            "reasoning_content",
            "thinking_blocks",
            "thinking",
        ] {
            map.remove(key);
        }
        if let Some(Value::Array(parts)) = map.get("content") {
            // Legacy thinking replay uses inline Anthropic content blocks.
            let kept: Vec<Value> = parts
                .iter()
                .filter(|p| {
                    !matches!(
                        p.get("type").and_then(Value::as_str),
                        Some("thinking" | "redacted_thinking")
                    )
                })
                .cloned()
                .collect();
            map.insert("content".into(), Value::Array(kept));
        }
        Value::Object(map)
    }

    fn omit_items(items: &[Value]) -> Vec<Value> {
        items
            .iter()
            .filter(|i| i.get("type").and_then(Value::as_str) != Some("reasoning"))
            .map(omit_message_reasoning)
            .collect()
    }

    let Some(items) = value.as_array() else {
        return value.clone();
    };
    Value::Array(
        omit_items(items)
            .into_iter()
            .map(|item| {
                let is_response = item.get("object").and_then(Value::as_str) == Some("response");
                match (is_response, item.get("output").and_then(Value::as_array)) {
                    (true, Some(output)) => {
                        let mut item = item.clone();
                        item["output"] = Value::Array(omit_items(output));
                        item
                    }
                    _ => item,
                }
            })
            .collect(),
    )
}

/// Enforce OpenAI traces ingest limits: truncate large fields and keep usage only on
/// generation spans.
pub(crate) fn sanitize_for_openai(mut payload: Value) -> Value {
    let Some(span_data) = payload.get_mut("span_data").and_then(Value::as_object_mut) else {
        return payload;
    };
    let is_generation = span_data.get("type").and_then(Value::as_str) == Some("generation");
    for field in ["input", "output"] {
        if let Some(v) = span_data.get(field) {
            let v = if is_generation {
                omit_generation_reasoning(v)
            } else {
                v.clone()
            };
            let t = truncate_value(&v);
            span_data.insert(field.into(), t);
        }
    }
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
                log::warn!(target: crate::logger::LOG_TARGET, "OPENAI_API_KEY is not set, skipping trace export");
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
                log::error!(target: crate::logger::LOG_TARGET, "[non-fatal] Tracing: failed to spawn exporter thread");
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
                    log::warn!(target: crate::logger::LOG_TARGET, "Queue is full, dropping trace item.");
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
                object: Some(ExportedObject::Trace(trace.clone())),
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
                object: Some(ExportedObject::Span(span.clone())),
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

    #[test]
    fn truncates_structures_recursively() {
        let big = "x".repeat(MAX_FIELD_BYTES);
        let payload = json!({"span_data": {"type": "generation", "input": [
            {"role": "user", "content": big.clone()},
            {"role": "user", "content": "short"},
        ]}});
        let input = sanitize_for_openai(payload)["span_data"]["input"].clone();
        assert!(input.is_array(), "the list structure is kept: {input}");
        assert!(input.to_string().len() <= MAX_FIELD_BYTES);
        let first = &input[0];
        assert_eq!(first["role"], "user");
        assert!(
            first["content"]
                .as_str()
                .unwrap()
                .ends_with(TRUNCATION_SUFFIX),
            "{first}"
        );
    }

    #[test]
    fn omits_reasoning_from_generation_spans() {
        let payload = json!({"span_data": {"type": "generation", "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret"}]},
            {"role": "assistant", "content": "hi", "reasoning_content": "secret",
             "thinking_blocks": [{"signature": "s"}]},
            {"object": "response", "output": [
                {"type": "reasoning", "summary": []},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "kept"},
                ]},
            ]},
        ], "usage": {"input_tokens": 1, "output_tokens": 1}}});
        let out = sanitize_for_openai(payload)["span_data"]["output"].clone();
        assert_eq!(out.as_array().unwrap().len(), 2, "{out}");
        assert_eq!(out[0]["content"], "hi");
        assert!(out[0].get("reasoning_content").is_none());
        assert!(out[0].get("thinking_blocks").is_none());
        let nested = &out[1]["output"];
        assert_eq!(nested.as_array().unwrap().len(), 1, "{nested}");
        assert_eq!(
            nested[0]["content"],
            json!([{"type": "text", "text": "kept"}])
        );
        // Non-generation spans are untouched.
        let other = json!({"span_data": {"type": "function",
            "output": [{"type": "reasoning", "summary": []}]}});
        assert_eq!(
            sanitize_for_openai(other)["span_data"]["output"],
            json!([{"type": "reasoning", "summary": []}])
        );
    }

    #[test]
    fn truncation_keeps_nested_structure_within_the_limit() {
        let big = "y".repeat(MAX_FIELD_BYTES * 2);
        let value = json!({"keep": "small", "big": {"inner": big}});
        let out = truncate_value(&value);
        assert!(
            out.to_string().len() <= MAX_FIELD_BYTES,
            "{}",
            out.to_string().len()
        );
        assert_eq!(out["keep"], "small");
        assert!(
            out["big"]["inner"]
                .as_str()
                .unwrap()
                .ends_with(TRUNCATION_SUFFIX),
            "{out}"
        );
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
