//! Experimental Codex tool: run the Codex CLI as an agent tool.
//!
//! Port of `agents.extensions.experimental.codex`. The [`Codex`] client drives
//! `codex exec --experimental-json` as a subprocess, parses its JSONL stream into typed
//! [`ThreadEvent`]s and [`ThreadItem`]s, and [`codex_tool`] wraps a thread in a
//! [`FunctionTool`](crate::tool::FunctionTool) that returns the final Codex response.
//!
//! The `codex` CLI must be installed; it is resolved from `CODEX_PATH`, then `PATH`, unless
//! [`CodexOptions::codex_path_override`] is set.
//!
//! ```no_run
//! use openai_agents_sdk::extensions::codex::{Codex, CodexOptions, ThreadOptions, SandboxMode};
//!
//! # async fn demo() -> Result<(), openai_agents_sdk::AgentsError> {
//! let codex = Codex::new(CodexOptions::new().with_api_key("sk-..."));
//! let thread = codex.start_thread(Some(
//!     ThreadOptions::new().with_sandbox_mode(SandboxMode::ReadOnly),
//! ));
//! let turn = thread.run("Summarize the test failures", None).await?;
//! println!("{}", turn.final_response);
//! # Ok(())
//! # }
//! ```
//!
//! ```no_run
//! use openai_agents_sdk::Agent;
//! use openai_agents_sdk::extensions::codex::{CodexToolOptions, codex_tool};
//!
//! # fn demo() -> Result<(), openai_agents_sdk::AgentsError> {
//! let agent = Agent::<()>::new("Dev").with_tool(codex_tool(
//!     CodexToolOptions::new().with_persist_session(true),
//! )?);
//! # let _ = agent;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod events;
pub mod exec;
pub mod items;
pub mod options;
pub mod output_schema;
pub mod thread;
pub mod thread_options;
pub mod tool;
pub mod turn_options;

pub use client::Codex;
pub use events::{ThreadError, ThreadEvent, Usage};
pub use exec::{CodexExec, CodexExecArgs, CodexLineStream, build_command_args, find_codex_path};
pub use items::{FileUpdateChange, McpToolCallError, McpToolCallResult, ThreadItem, TodoItem};
pub use options::CodexOptions;
pub use output_schema::{
    JsonPrimitive, OutputSchema, OutputSchemaDescriptor, OutputSchemaField, OutputSchemaFile,
    OutputSchemaProperty,
};
pub use thread::{Input, RunResult, RunStreamedResult, StreamedTurn, Thread, Turn, UserInput};
pub use thread_options::{
    ApprovalMode, ModelReasoningEffort, SandboxMode, ThreadOptions, WebSearchMode,
};
pub use tool::{
    CODEX_TOOL_NAME_PREFIX, CodexToolOptions, CodexToolResult, CodexToolStreamEvent,
    DEFAULT_CODEX_TOOL_DESCRIPTION, DEFAULT_CODEX_TOOL_NAME, DEFAULT_SPAN_DATA_MAX_CHARS,
    ThreadIdGetFn, ThreadIdSetFn, ThreadIdStore, codex_tool, codex_tool_params_schema,
};
pub use turn_options::TurnOptions;
