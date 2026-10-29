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

pub mod processors;
pub mod provider;
pub mod scope;
pub mod span_data;
pub mod spans;

use std::future::Future;
use std::sync::Arc;

use serde_json::{Map, Value};

pub use processors::{
    BackendSpanExporter, BatchTraceProcessor, ConsoleSpanExporter, ExportItem,
    InMemoryTraceProcessor, TracingExporter, TracingProcessor,
};
pub use provider::{TraceProvider, provider};
pub use scope::{current_span, current_trace};
pub use span_data::SpanData;
pub use spans::{Span, SpanError, Trace, gen_group_id, gen_span_id, gen_trace_id};

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
        let disabled = self.disabled || provider().is_disabled();
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

/// Create a span under the current span (or the current trace).
pub fn create_span(data: SpanData, parent: Option<&Span>, disabled: bool) -> Span {
    let current;
    let parent = match parent {
        Some(p) => Some(p),
        None => {
            current = scope::current_span();
            current.as_ref()
        }
    };
    let trace = scope::current_trace();
    Span::new(
        trace.as_ref(),
        parent,
        data,
        disabled || provider().is_disabled(),
    )
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

/// Globally enable or disable tracing.
pub fn set_tracing_disabled(disabled: bool) {
    provider().set_disabled(disabled);
}

/// Set the API key used to export traces to OpenAI.
pub fn set_tracing_export_api_key(key: impl Into<String>) {
    provider().set_export_api_key(key);
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

/// Export buffered traces and spans without blocking the async runtime.
pub async fn flush_traces_async() {
    let _ = tokio::task::spawn_blocking(flush_traces).await;
}
