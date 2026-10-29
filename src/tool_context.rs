//! Context passed to function tools. Mirrors `agents.tool_context.ToolContext`.

use std::fmt;
use std::ops::Deref;

use crate::items::FunctionCall;
use crate::run_context::RunContext;

/// The run context plus metadata about the tool call being executed.
pub struct ToolContext<C> {
    run_context: RunContext<C>,
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// Id of the tool call.
    pub tool_call_id: String,
    /// Raw JSON arguments.
    pub tool_arguments: String,
    /// The full function call item, when invoked by a model tool call.
    pub tool_call: Option<FunctionCall>,
    /// Name of the agent invoking the tool.
    pub agent_name: String,
}

impl<C> Clone for ToolContext<C> {
    fn clone(&self) -> Self {
        ToolContext {
            run_context: self.run_context.clone(),
            tool_name: self.tool_name.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tool_arguments: self.tool_arguments.clone(),
            tool_call: self.tool_call.clone(),
            agent_name: self.agent_name.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolContext<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("tool_name", &self.tool_name)
            .field("tool_call_id", &self.tool_call_id)
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

impl<C> Deref for ToolContext<C> {
    type Target = RunContext<C>;
    fn deref(&self) -> &Self::Target {
        &self.run_context
    }
}

impl<C> ToolContext<C> {
    /// Build a tool context from a run context and a function call.
    pub fn from_call(run_context: RunContext<C>, agent_name: &str, call: &FunctionCall) -> Self {
        ToolContext {
            run_context,
            tool_name: call.name.clone(),
            tool_call_id: call.call_id.clone(),
            tool_arguments: call.arguments.clone(),
            tool_call: Some(call.clone()),
            agent_name: agent_name.to_string(),
        }
    }

    /// Build a tool context for a direct (non model-driven) invocation.
    pub fn direct(
        run_context: RunContext<C>,
        agent_name: &str,
        tool_name: &str,
        arguments: &str,
    ) -> Self {
        ToolContext {
            run_context,
            tool_name: tool_name.to_string(),
            tool_call_id: format!("call_{}", crate::util::gen_hex24()),
            tool_arguments: arguments.to_string(),
            tool_call: None,
            agent_name: agent_name.to_string(),
        }
    }

    /// The underlying run context.
    pub fn run_context(&self) -> &RunContext<C> {
        &self.run_context
    }
}
