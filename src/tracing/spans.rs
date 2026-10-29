//! Traces and spans. Mirrors `agents.tracing.traces` and `agents.tracing.spans`.

use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use super::provider::provider;
use super::scope;
use super::span_data::SpanData;
use crate::util::time_iso;

/// An error recorded on a span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanError {
    pub message: String,
    pub data: Option<Value>,
}

struct TraceInner {
    trace_id: String,
    name: String,
    group_id: Option<String>,
    metadata: Option<Map<String, Value>>,
    tracing_api_key: Option<String>,
    disabled: bool,
    started: AtomicBool,
    finished: AtomicBool,
}

/// A trace: the root of a workflow, grouping spans.
#[derive(Clone)]
pub struct Trace(Arc<TraceInner>);

impl fmt::Debug for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Trace")
            .field("trace_id", &self.0.trace_id)
            .field("name", &self.0.name)
            .field("disabled", &self.0.disabled)
            .finish()
    }
}

impl Trace {
    pub(crate) fn new(
        name: String,
        trace_id: Option<String>,
        group_id: Option<String>,
        metadata: Option<Map<String, Value>>,
        tracing_api_key: Option<String>,
        disabled: bool,
    ) -> Self {
        Trace(Arc::new(TraceInner {
            trace_id: trace_id.unwrap_or_else(gen_trace_id),
            name,
            group_id,
            metadata,
            tracing_api_key,
            disabled,
            started: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        }))
    }

    /// The trace id (`trace_<32 hex>`).
    pub fn trace_id(&self) -> &str {
        &self.0.trace_id
    }

    /// The workflow name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// The group id.
    pub fn group_id(&self) -> Option<&str> {
        self.0.group_id.as_deref()
    }

    /// Trace metadata.
    pub fn metadata(&self) -> Option<&Map<String, Value>> {
        self.0.metadata.as_ref()
    }

    /// Whether the trace is disabled (not recorded).
    pub fn is_disabled(&self) -> bool {
        self.0.disabled
    }

    /// API key used to export this trace, if overridden.
    pub fn tracing_api_key(&self) -> Option<&str> {
        self.0.tracing_api_key.as_deref()
    }

    /// Start the trace (notifies processors once).
    pub fn start(&self) {
        if self.0.disabled || self.0.started.swap(true, Ordering::SeqCst) {
            return;
        }
        provider().on_trace_start(self);
    }

    /// Finish the trace (notifies processors once).
    pub fn finish(&self) {
        if self.0.disabled
            || !self.0.started.load(Ordering::SeqCst)
            || self.0.finished.swap(true, Ordering::SeqCst)
        {
            return;
        }
        provider().on_trace_end(self);
    }

    /// Start the trace, run `fut` with it as the current trace, then finish it.
    pub async fn run<F: Future>(self, fut: F) -> F::Output {
        self.start();
        let out = scope::with_trace(self.clone(), fut).await;
        self.finish();
        out
    }

    /// Export to the OpenAI traces wire format (`None` if disabled).
    pub fn export(&self) -> Option<Value> {
        if self.0.disabled {
            return None;
        }
        Some(json!({
            "object": "trace",
            "id": self.0.trace_id,
            "workflow_name": self.0.name,
            "group_id": self.0.group_id,
            "metadata": self.0.metadata,
        }))
    }
}

struct SpanState {
    started_at: Option<String>,
    ended_at: Option<String>,
    data: SpanData,
    error: Option<SpanError>,
}

struct SpanInner {
    trace_id: String,
    span_id: String,
    parent_id: Option<String>,
    disabled: bool,
    tracing_api_key: Option<String>,
    trace_metadata: Option<Map<String, Value>>,
    state: Mutex<SpanState>,
}

/// A span: an operation with a start and end time inside a trace.
#[derive(Clone)]
pub struct Span(Arc<SpanInner>);

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let st = self.0.state.lock().expect("span poisoned");
        f.debug_struct("Span")
            .field("span_id", &self.0.span_id)
            .field("trace_id", &self.0.trace_id)
            .field("parent_id", &self.0.parent_id)
            .field("type", &st.data.span_type())
            .finish()
    }
}

impl Span {
    pub(crate) fn new(
        trace: Option<&Trace>,
        parent: Option<&Span>,
        data: SpanData,
        disabled: bool,
    ) -> Self {
        let (trace_id, api_key, meta) = match (parent, trace) {
            (Some(p), _) => (
                p.0.trace_id.clone(),
                p.0.tracing_api_key.clone(),
                p.0.trace_metadata.clone(),
            ),
            (None, Some(t)) => (
                t.0.trace_id.clone(),
                t.0.tracing_api_key.clone(),
                t.0.metadata.clone(),
            ),
            (None, None) => ("no-op".to_string(), None, None),
        };
        let disabled = disabled
            || (parent.is_none() && trace.is_none())
            || parent.map(|p| p.0.disabled).unwrap_or(false)
            || trace.map(|t| t.is_disabled()).unwrap_or(false);
        Span(Arc::new(SpanInner {
            trace_id,
            span_id: gen_span_id(),
            parent_id: parent.map(|p| p.0.span_id.clone()),
            disabled,
            tracing_api_key: api_key,
            trace_metadata: meta,
            state: Mutex::new(SpanState {
                started_at: None,
                ended_at: None,
                data,
                error: None,
            }),
        }))
    }

    /// The span id.
    pub fn span_id(&self) -> &str {
        &self.0.span_id
    }

    /// The trace id.
    pub fn trace_id(&self) -> &str {
        &self.0.trace_id
    }

    /// The parent span id.
    pub fn parent_id(&self) -> Option<&str> {
        self.0.parent_id.as_deref()
    }

    /// Whether this span is recorded.
    pub fn is_disabled(&self) -> bool {
        self.0.disabled
    }

    /// API key used to export this span, if overridden.
    pub fn tracing_api_key(&self) -> Option<&str> {
        self.0.tracing_api_key.as_deref()
    }

    /// A copy of the span data.
    pub fn data(&self) -> SpanData {
        self.0.state.lock().expect("span poisoned").data.clone()
    }

    /// The recorded error, if any.
    pub fn error(&self) -> Option<SpanError> {
        self.0.state.lock().expect("span poisoned").error.clone()
    }

    /// Mutate the span data.
    pub fn update_data(&self, f: impl FnOnce(&mut SpanData)) {
        f(&mut self.0.state.lock().expect("span poisoned").data);
    }

    /// Record an error.
    pub fn set_error(&self, message: impl Into<String>, data: Option<Value>) {
        self.0.state.lock().expect("span poisoned").error = Some(SpanError {
            message: message.into(),
            data,
        });
    }

    /// Start the span.
    pub fn start(&self) {
        {
            let mut st = self.0.state.lock().expect("span poisoned");
            if st.started_at.is_some() {
                return;
            }
            st.started_at = Some(time_iso());
        }
        if !self.0.disabled {
            provider().on_span_start(self);
        }
    }

    /// Finish the span.
    pub fn finish(&self) {
        {
            let mut st = self.0.state.lock().expect("span poisoned");
            if st.ended_at.is_some() {
                return;
            }
            st.ended_at = Some(time_iso());
        }
        if !self.0.disabled {
            provider().on_span_end(self);
        }
    }

    /// Start the span, run `fut` with it as the current span, then finish it.
    pub async fn run<F: Future>(self, fut: F) -> F::Output {
        self.start();
        let out = scope::with_span(self.clone(), fut).await;
        self.finish();
        out
    }

    /// Export to the OpenAI traces wire format (`None` if disabled).
    pub fn export(&self) -> Option<Value> {
        if self.0.disabled {
            return None;
        }
        let st = self.0.state.lock().expect("span poisoned");
        let mut payload = json!({
            "object": "trace.span",
            "id": self.0.span_id,
            "trace_id": self.0.trace_id,
            "parent_id": self.0.parent_id,
            "started_at": st.started_at,
            "ended_at": st.ended_at,
            "span_data": st.data.export(),
            "error": st.error.as_ref().map(|e| json!({"message": e.message, "data": e.data})),
        });
        if let Some(meta) = &self.0.trace_metadata
            && let Some(v) = meta.get("agent_harness_id")
        {
            payload["metadata"] = json!({"agent_harness_id": v});
        }
        Some(payload)
    }
}

/// Generate a trace id (`trace_<32 hex>`).
pub fn gen_trace_id() -> String {
    format!("trace_{}", crate::util::gen_hex32())
}

/// Generate a span id (`span_<24 hex>`).
pub fn gen_span_id() -> String {
    format!("span_{}", crate::util::gen_hex24())
}

/// Generate a group id (`group_<24 hex>`).
pub fn gen_group_id() -> String {
    format!("group_{}", crate::util::gen_hex24())
}
