//! Built-in tracing compatible with the OpenAI Traces dashboard. Mirrors `agents.tracing`.
//!
//! The runner wraps every run in a trace with agent, generation/response, function, guardrail
//! and handoff spans. Traces are exported in the background; call [`flush_traces`] (or
//! [`flush_traces_async`]) before your process exits so the last traces are delivered.
//!
//! ```no_run
//! # async fn demo() {
//! use openai_agents_sdk::tracing::trace;
//! trace("Joke workflow").run(async {
//!     // several Runner::run calls here share one trace
//! }).await;
//! openai_agents_sdk::tracing::flush_traces();
//! # }
//! ```

pub mod error_tracing;
pub mod processors;
pub mod provider;
pub mod scope;
pub mod span_data;
pub mod spans;
pub mod trace_state;

use std::future::Future;
use std::sync::Arc;

use serde_json::{Map, Value};

pub use error_tracing::{
    REDACTED_TOOL_ERROR_MESSAGE, REDACTED_TRACE_ERROR_MESSAGE, attach_error_to_current_span,
    get_trace_error, get_trace_tool_error, model_error_text, record_model_error_on_span,
    record_tool_error_on_span,
};
pub use processors::{
    BackendSpanExporter, BatchTraceProcessor, ConsoleSpanExporter, ExportItem, ExportedObject,
    InMemoryTraceProcessor, TracingExporter, TracingProcessor,
};
pub use provider::{
    DefaultTraceProvider, TraceProvider, provider, reset_trace_provider, set_trace_provider,
};
pub use scope::{current_span, current_trace};
pub use span_data::SpanData;
pub use spans::{Span, SpanError, Trace, gen_group_id, gen_span_id, gen_trace_id};
pub use trace_state::{TraceState, hash_tracing_api_key, reattach_trace, trace_id_was_started};

/// Builder for a trace.
#[derive(Debug, Clone, Default)]
pub struct TraceBuilder {
    name: String,
    trace_id: Option<String>,
    group_id: Option<String>,
    metadata: Option<Map<String, Value>>,
    disabled: bool,
    tracing_api_key: Option<String>,
}

impl TraceBuilder {
    /// Set the trace id (must look like `trace_<32 alphanumeric>`).
    pub fn trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }
    /// Link traces from the same conversation.
    pub fn group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }
    /// Attach metadata.
    pub fn metadata(mut self, metadata: Map<String, Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
    /// Disable recording for this trace.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
    /// Export this trace with a specific API key.
    pub fn tracing_api_key(mut self, key: impl Into<String>) -> Self {
        self.tracing_api_key = Some(key.into());
        self
    }
    /// Build the trace without starting it.
    pub fn build(self) -> Trace {
        if scope::current_trace().is_some() {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "Trace already exists. Creating a new trace, but this is probably a mistake."
            );
        }
        let disabled = self.disabled || provider().is_disabled();
        if disabled {
            log::debug!(target: crate::logger::LOG_TARGET, "Tracing is disabled. Not creating trace");
        } else {
            log::debug!(target: crate::logger::LOG_TARGET, "Creating trace");
        }
        Trace::new(
            self.name,
            self.trace_id,
            self.group_id,
            self.metadata,
            self.tracing_api_key,
            disabled,
        )
    }
    /// Start the trace, run `fut` inside it, and finish it.
    pub async fn run<F: Future>(self, fut: F) -> F::Output {
        self.build().run(fut).await
    }
}

/// Create a trace for a workflow. Use `.run(future)` to make it current while the future runs,
/// so that every `Runner::run` inside becomes part of the same trace.
pub fn trace(workflow_name: impl Into<String>) -> TraceBuilder {
    TraceBuilder {
        name: workflow_name.into(),
        ..Default::default()
    }
}

/// The name of the disabled span a masked run enters.
pub const MASKED_RUN_SPAN_NAME: &str = "Agent run";

/// The trace a run should use, or `None` when the run joins the already active trace.
///
/// With `disabled`, a disabled trace is always returned so that entering it hides an outer
/// active trace: a run with tracing turned off must not export spans into its caller's trace.
/// Pair it with [`masked_run_span`], which hides the caller's *span* the same way.
pub fn create_trace_for_run(
    workflow_name: impl Into<String>,
    trace_id: Option<String>,
    group_id: Option<String>,
    metadata: Option<Map<String, Value>>,
    tracing_api_key: Option<String>,
    disabled: bool,
) -> Option<Trace> {
    if !disabled && scope::current_trace().is_some() {
        return None;
    }
    Some(Trace::new(
        workflow_name.into(),
        trace_id,
        group_id,
        metadata,
        tracing_api_key,
        disabled || provider().is_disabled(),
    ))
}

/// [`create_trace_for_run`], but reusing the trace a snapshotted run was already exporting to.
///
/// When `reattach_resumed_trace` is set and `trace_state` describes the very trace this run is
/// about to create — same id, workflow name, group id, metadata and export key — and that id
/// was already started in this process, the returned trace is a
/// [`reattach_trace`] handle: spans keep flowing into the original
/// trace and no second `trace.start` event is exported. Port of `create_trace_for_run`'s
/// resume branch in `agents.tracing.context`.
#[allow(clippy::too_many_arguments)]
pub fn create_trace_for_run_with_state(
    workflow_name: impl Into<String>,
    trace_id: Option<String>,
    group_id: Option<String>,
    metadata: Option<Map<String, Value>>,
    tracing_api_key: Option<String>,
    disabled: bool,
    trace_state: Option<&TraceState>,
    reattach_resumed_trace: bool,
) -> Option<Trace> {
    let workflow_name = workflow_name.into();
    if !disabled
        && reattach_resumed_trace
        && scope::current_trace().is_none()
        && let Some(state) = trace_state
        && trace_id_was_started(state.trace_id.as_deref())
        && state.matches_effective_settings(
            &workflow_name,
            trace_id.as_deref(),
            group_id.as_deref(),
            metadata.as_ref(),
            tracing_api_key.as_deref(),
        )
    {
        return reattach_trace(state, tracing_api_key);
    }
    create_trace_for_run(
        workflow_name,
        trace_id,
        group_id,
        metadata,
        tracing_api_key,
        disabled,
    )
}

/// A disabled span entered by a run whose tracing is turned off, so code that reads
/// [`current_span`] does not see the caller's span.
pub fn masked_run_span() -> Span {
    create_span(
        SpanData::Custom {
            name: MASKED_RUN_SPAN_NAME.to_string(),
            data: Value::Null,
        },
        None,
        true,
    )
}

/// Create a span under the current span (or the current trace).
pub fn create_span(data: SpanData, parent: Option<&Span>, disabled: bool) -> Span {
    let mut builder = span(data).disabled(disabled);
    if let Some(p) = parent {
        builder = builder.parent_span(p.clone());
    }
    builder.build()
}

/// The explicit parent of a new span. Mirrors Python's `parent: Trace | Span | None`.
#[derive(Debug, Clone)]
pub enum SpanParent {
    /// Nest the new span under an existing span.
    Span(Span),
    /// Attach the new span directly to a trace.
    Trace(Trace),
}

/// Builder covering the arguments every Python span creator accepts (`span_id`, `parent`,
/// `disabled`) plus the `metadata` slot of the agent/task/turn span data.
///
/// The named helpers ([`agent_span`], [`function_span`], ...) keep Python's positional
/// arguments; this builder is how their keyword arguments are expressed in Rust.
///
/// ```
/// use openai_agents_sdk::tracing::{SpanData, span};
/// let s = span(SpanData::Agent {
///     name: "Assistant".into(),
///     handoffs: None,
///     tools: None,
///     output_type: None,
/// })
/// .span_id("span_000000000000000000000001")
/// .disabled(true)
/// .build();
/// assert_eq!(s.span_id(), "span_000000000000000000000001");
/// ```
#[derive(Debug, Clone)]
pub struct SpanBuilder {
    data: SpanData,
    span_id: Option<String>,
    parent: Option<SpanParent>,
    disabled: bool,
    metadata: Option<Map<String, Value>>,
}

impl SpanBuilder {
    /// Use an explicit span id instead of generating one.
    pub fn span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }
    /// Nest the span under an explicit parent span.
    pub fn parent_span(mut self, parent: Span) -> Self {
        self.parent = Some(SpanParent::Span(parent));
        self
    }
    /// Attach the span directly to an explicit trace.
    pub fn parent_trace(mut self, trace: Trace) -> Self {
        self.parent = Some(SpanParent::Trace(trace));
        self
    }
    /// Set the parent (span or trace).
    pub fn parent(mut self, parent: SpanParent) -> Self {
        self.parent = Some(parent);
        self
    }
    /// Create the span without recording it.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
    /// Fill the span-data `metadata` slot (not exported, as in Python).
    pub fn metadata(mut self, metadata: Map<String, Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
    /// Build the span without starting it.
    pub fn build(self) -> Span {
        let current_span;
        let current_trace;
        let (parent_span, trace) = match &self.parent {
            Some(SpanParent::Span(s)) => (Some(s), None),
            Some(SpanParent::Trace(t)) => (None, Some(t)),
            None => {
                current_span = scope::current_span();
                current_trace = scope::current_trace();
                (current_span.as_ref(), current_trace.as_ref())
            }
        };
        let forced_off = self.disabled || provider().is_disabled();
        log_span_decision(
            &self.data,
            self.span_id.as_deref(),
            parent_span,
            trace,
            forced_off,
            self.parent.is_none(),
        );
        let span = Span::with_id(trace, parent_span, self.data, forced_off, self.span_id);
        if self.metadata.is_some() {
            span.set_data_metadata(self.metadata);
        }
        span
    }
}

/// Emit the debug line `create_span` logs for this span.
///
/// The span data is kept out of the message whenever the logging policy forbids model or tool
/// data, exactly as `_debug.DONT_LOG_MODEL_DATA` / `DONT_LOG_TOOL_DATA` do in Python, so a
/// workflow that disables tracing never leaks a span name into the log.
fn log_span_decision(
    data: &SpanData,
    span_id: Option<&str>,
    parent_span: Option<&Span>,
    trace: Option<&Trace>,
    forced_off: bool,
    implicit_parent: bool,
) {
    const TARGET: &str = crate::logger::LOG_TARGET;
    let may_log = crate::logger::may_log_model_data(true);
    if forced_off {
        if may_log {
            log::debug!(target: TARGET, "Tracing is disabled. Not creating span {data:?}");
        } else {
            log::debug!(target: TARGET, "Tracing is disabled. Not creating span");
        }
        return;
    }
    if span_id == Some("no-op") {
        log::debug!(target: TARGET, "Span id is no-op, returning NoOpSpan");
        return;
    }
    if implicit_parent && trace.is_none() {
        log::debug!(
            target: TARGET,
            "No active trace. Make sure to start a trace with `trace()` first Returning NoOpSpan."
        );
        return;
    }
    let noop_parent = parent_span
        .is_some_and(|p| p.is_disabled() || p.span_id() == "no-op" || p.trace_id() == "no-op")
        || trace.is_some_and(|t| t.is_disabled() || t.trace_id() == "no-op");
    if noop_parent {
        if may_log {
            log::debug!(target: TARGET, "Parent {parent_span:?} or {trace:?} is no-op, returning NoOpSpan");
        } else {
            log::debug!(target: TARGET, "Current trace parent is no-op, returning NoOpSpan");
        }
        return;
    }
    if may_log {
        log::debug!(
            target: TARGET,
            "Creating span {data:?} with id {}",
            span_id.unwrap_or("<generated>")
        );
    } else {
        log::debug!(target: TARGET, "Creating span");
    }
}

/// Start building a span from its data. See [`SpanBuilder`].
pub fn span(data: SpanData) -> SpanBuilder {
    SpanBuilder {
        data,
        span_id: None,
        parent: None,
        disabled: false,
        metadata: None,
    }
}

/// A custom span with arbitrary data.
pub fn custom_span(name: impl Into<String>, data: Value) -> Span {
    create_span(
        SpanData::Custom {
            name: name.into(),
            data,
        },
        None,
        false,
    )
}

/// An agent span.
pub fn agent_span(
    name: &str,
    handoffs: Option<Vec<String>>,
    tools: Option<Vec<String>>,
    output_type: Option<String>,
) -> Span {
    create_span(
        SpanData::Agent {
            name: name.to_string(),
            handoffs,
            tools,
            output_type,
        },
        None,
        false,
    )
}

/// A task span: one `Runner` invocation.
pub fn task_span(name: &str) -> Span {
    create_span(
        SpanData::Task {
            name: name.to_string(),
            usage: None,
        },
        None,
        false,
    )
}

/// A turn span: one model turn of an agent.
pub fn turn_span(turn: u64, agent_name: &str) -> Span {
    create_span(
        SpanData::Turn {
            turn,
            agent_name: agent_name.to_string(),
            usage: None,
        },
        None,
        false,
    )
}

/// A function tool span.
pub fn function_span(name: &str, input: Option<String>) -> Span {
    create_span(
        SpanData::Function {
            name: name.to_string(),
            input,
            output: None,
            mcp_data: None,
        },
        None,
        false,
    )
}

/// A generation span (Chat Completions).
pub fn generation_span(
    input: Option<Value>,
    model: Option<String>,
    model_config: Option<Value>,
    disabled: bool,
) -> Span {
    create_span(
        SpanData::Generation {
            input,
            output: None,
            model,
            model_config,
            usage: None,
        },
        None,
        disabled,
    )
}

/// A response span (Responses API).
pub fn response_span(disabled: bool) -> Span {
    create_span(
        SpanData::Response {
            response_id: None,
            usage: None,
            response: None,
            input: None,
        },
        None,
        disabled,
    )
}

/// A handoff span.
pub fn handoff_span(from_agent: &str, to_agent: Option<&str>) -> Span {
    create_span(
        SpanData::Handoff {
            from_agent: Some(from_agent.to_string()),
            to_agent: to_agent.map(str::to_string),
        },
        None,
        false,
    )
}

/// A guardrail span.
pub fn guardrail_span(name: &str, triggered: bool) -> Span {
    create_span(
        SpanData::Guardrail {
            name: name.to_string(),
            triggered,
        },
        None,
        false,
    )
}

/// An MCP list-tools span.
pub fn mcp_tools_span(server: &str, result: Option<Vec<String>>) -> Span {
    create_span(
        SpanData::MCPListTools {
            server: Some(server.to_string()),
            result,
        },
        None,
        false,
    )
}

/// A speech-to-text span.
pub fn transcription_span(
    model: Option<String>,
    model_config: Option<Value>,
    disabled: bool,
) -> Span {
    create_span(
        SpanData::Transcription {
            input: None,
            input_format: Some("pcm".into()),
            output: None,
            model,
            model_config,
        },
        None,
        disabled,
    )
}

/// A text-to-speech span.
pub fn speech_span(
    model: Option<String>,
    input: Option<String>,
    model_config: Option<Value>,
    disabled: bool,
) -> Span {
    create_span(
        SpanData::Speech {
            input,
            output: None,
            output_format: Some("pcm".into()),
            model,
            model_config,
            first_content_at: None,
        },
        None,
        disabled,
    )
}

/// A span grouping the speech of one voice turn.
pub fn speech_group_span(input: Option<String>, disabled: bool) -> Span {
    create_span(SpanData::SpeechGroup { input }, None, disabled)
}

/// Globally enable or disable tracing.
pub fn set_tracing_disabled(disabled: bool) {
    provider().set_disabled(disabled);
}

/// Set the API key used to export traces to OpenAI.
pub fn set_tracing_export_api_key(key: impl Into<String>) {
    provider::set_export_api_key(key);
}

/// Replace the trace processors (the default OpenAI exporter is removed).
pub fn set_trace_processors(processors: Vec<Arc<dyn TracingProcessor>>) {
    provider().set_processors(processors);
}

/// Add a trace processor alongside the default OpenAI exporter.
pub fn add_trace_processor(processor: Arc<dyn TracingProcessor>) {
    provider().add_processor(processor);
}

/// Block until buffered traces and spans are exported.
pub fn flush_traces() {
    provider().force_flush();
}

/// Stop the tracing pipeline, flushing everything buffered.
pub fn shutdown_tracing() {
    provider().shutdown();
}

/// Export buffered traces and spans without blocking the async runtime.
pub async fn flush_traces_async() {
    let _ = tokio::task::spawn_blocking(flush_traces).await;
}

/// Shuts the tracing pipeline down when dropped. The Rust stand-in for Python's `atexit` flush.
///
/// Rust has no process-exit hook and a `static` is never dropped, so the last buffered traces
/// are only guaranteed to be exported if *something* with a scope calls [`shutdown_tracing`].
/// Hold this guard for the lifetime of `main` and that happens on every normal return, on `?`
/// and while a panic unwinds:
///
/// ```no_run
/// # async fn demo() {
/// let _tracing = openai_agents_sdk::tracing::shutdown_guard();
/// // ... run agents ...
/// # }
/// ```
///
/// A `std::process::exit` or an abort still skips it, as it skips every Rust destructor.
#[derive(Debug)]
#[must_use = "the guard flushes traces when it is dropped; bind it to a variable"]
pub struct TracingShutdownGuard(());

impl Drop for TracingShutdownGuard {
    fn drop(&mut self) {
        shutdown_tracing();
    }
}

/// Create a [`TracingShutdownGuard`].
pub fn shutdown_guard() -> TracingShutdownGuard {
    TracingShutdownGuard(())
}
