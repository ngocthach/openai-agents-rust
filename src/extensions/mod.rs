//! Optional helpers: handoff filters, recommended handoff prompts and agent graph
//! visualization. Mirrors `agents.extensions`.

pub mod handoff_filters;
pub mod handoff_prompt;
pub mod tool_output_trimmer;
pub mod visualization;

pub mod codex;
#[cfg(feature = "websocket")]
pub mod hosted_multi_agent;
