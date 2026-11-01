//! Unofficial Rust port of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).
//!
//! The crate keeps the Python SDK's small set of primitives:
//!
//! - [`Agent`]: an LLM configured with instructions, tools, handoffs and guardrails.
//! - [`Runner`]: the agent loop (`run`, `run_sync`, `run_streamed`).
//! - [`FunctionTool`] and hosted tools, [`Handoff`]s, input/output/tool guardrails.
//! - [`memory::Session`] for conversation memory, [`tracing`] for OpenAI traces, and
//!   [`mcp`] (feature `mcp`) for Model Context Protocol servers.
//!
//! ```no_run
//! use openai_agents_sdk::{Agent, Runner};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let agent = Agent::<()>::new("Assistant").with_instructions("You are a helpful assistant");
//!     let result = Runner::run(&agent, "Write a haiku about recursion in programming.").await?;
//!     println!("{}", result.final_output_text().unwrap_or_default());
//!     Ok(())
//! }
//! ```

#![warn(missing_debug_implementations)]
#![forbid(unsafe_code)]

pub mod agent;
pub mod agent_output;
pub mod apply_diff;
pub mod computer;
pub mod debug;
pub mod editor;
pub mod error;
pub mod extensions;
pub mod guardrail;
pub mod handoffs;
pub mod items;
pub mod lifecycle;
pub mod logger;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod model_settings;
pub mod models;
pub mod prompts;
#[cfg(feature = "realtime")]
pub mod realtime;
pub mod repl;
pub mod result;
pub mod retry;
pub mod run;
pub mod run_config;
pub mod run_context;
pub mod run_error_handlers;
mod run_internal;
pub mod run_state;
#[cfg(feature = "sandbox")]
pub mod sandbox;
pub mod stream_events;
pub mod strict_schema;
pub mod testing;
pub mod tool;
pub mod tool_context;
pub mod tool_guardrails;
pub mod tracing;
pub mod usage;
pub(crate) mod util;
#[cfg(feature = "voice")]
pub mod voice;

pub use agent::{
    Agent, AgentAsToolOptions, AgentModel, Instructions, MCPConfig, StopAtTools, ToolUseBehavior,
    ToolsToFinalOutputResult,
};
pub use agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
pub use error::{AgentsError, Result, RunErrorDetails};
pub use guardrail::{
    GuardrailFunctionOutput, InputGuardrail, InputGuardrailResult, OutputGuardrail,
    OutputGuardrailResult,
};
pub use handoffs::{
    Handoff, HandoffHistoryMapper, HandoffInputData, HandoffInputFilter,
    async_handoff_input_filter, handoff, handoff_input_filter, nest_handoff_history,
    try_handoff_input_filter,
};
pub use items::{
    FunctionCall, FunctionCallOutput, InputItem, ItemHelpers, MessageOutput, ModelResponse,
    OutputItem, ReasoningOutput, RunInput, RunItem, RunItemKind, ToolApprovalItem,
};
pub use lifecycle::{AgentHooks, RunHooks};
pub use logger::enable_verbose_stdout_logging;
pub use memory::{InMemorySession, Session, SessionSettings};
pub use model_settings::{MCPToolChoice, ModelSettings, Reasoning, ToolChoice};
pub use models::{
    Model, ModelProvider, ModelRequest, ModelTracing, MultiProvider, OpenAIApi,
    OpenAIChatCompletionsModel, OpenAIClient, OpenAIProvider, OpenAIResponsesModel,
    set_default_openai_api, set_default_openai_client, set_default_openai_harness,
    set_default_openai_key,
};
pub use prompts::{Prompt, PromptVariable};
pub use result::{
    AgentToolInvocation, CancelMode, RunResult, RunResultStreaming, StreamCompletion,
    StreamCompletionHook, ToInputListMode,
};
pub use run::{
    AgentRunner, DEFAULT_MAX_TURNS, DefaultAgentRunner, IntoAgentArc, RunOptions, Runner,
    get_default_agent_runner, set_default_agent_runner,
};
pub use run_config::{
    CallModelData, ModelInputData, RunConfig, ToolErrorFormatter, ToolErrorFormatterArgs,
    ToolErrorKind, ToolExecutionConfig, ToolNotFoundBehavior, async_tool_error_formatter_fn,
    tool_error_formatter_fn, try_tool_error_formatter_fn,
};
pub use run_context::{ApprovalStatus, RunContext};
pub use run_error_handlers::{RunErrorHandlerInput, RunErrorHandlerResult, RunErrorHandlers};
pub use run_internal::resolve_run_grouping_id;
#[doc(hidden)]
pub use run_internal::tool_invocation::tool_invocation_identity_for_tests;
pub use run_state::RunState;
pub use stream_events::{RunItemStreamEventName, StreamEvent};
pub use tool::{
    CodeInterpreterTool, FileSearchTool, FunctionTool, FunctionToolResult, HostedMCPTool,
    ImageGenerationTool, Tool, ToolOrigin, ToolOriginType, ToolOutput, ToolOutputPart,
    WebSearchTool, function_tool,
};
pub use tool_context::ToolContext;
pub use tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData,
    ToolInputGuardrailResult, ToolOutputGuardrail, ToolOutputGuardrailData,
    ToolOutputGuardrailResult,
};
pub use tracing::{
    add_trace_processor, custom_span, flush_traces, flush_traces_async, set_trace_processors,
    set_tracing_disabled, set_tracing_export_api_key, trace,
};
pub use usage::{RequestUsage, Usage};
pub use util::{transform_string_function_style, transform_string_function_style_with};

/// Compiles the README examples as doctests.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

/// Turn a function into a [`FunctionTool`] (feature `macros`). See the macro crate docs.
#[cfg(feature = "macros")]
pub use openai_agents_sdk_macros::function_tool;

/// The server-conversation delivery tracker, exposed so its contract can be tested directly.
#[doc(hidden)]
pub use run_internal::oai_conversation::ServerConversationTracker;

#[doc(hidden)]
pub mod __private {
    pub use schemars;
    pub use serde;
    pub use serde_json;

    /// Run a synchronous tool body off the async worker, like Python's `asyncio.to_thread`, so
    /// a slow blocking function does not stall other tasks. Outside a tokio runtime the body
    /// runs inline. A panic in the body propagates to the caller as it would inline.
    pub async fn run_blocking<T, F>(f: F) -> T
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => match handle.spawn_blocking(f).await {
                Ok(value) => value,
                Err(error) => match error.try_into_panic() {
                    Ok(payload) => std::panic::resume_unwind(payload),
                    Err(error) => panic!("a blocking tool body was cancelled: {error}"),
                },
            },
            Err(_) => f(),
        }
    }
}
