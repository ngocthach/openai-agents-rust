//! Lifecycle hooks. Mirrors `agents.lifecycle`.

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::Agent;
use crate::items::{InputItem, ModelResponse};
use crate::run_context::RunContext;
use crate::tool::{Tool, ToolOutput};
use crate::tool_context::ToolContext;

/// Callbacks observing an entire `Runner::run` invocation, including handoffs.
///
/// All methods have no-op defaults.
#[async_trait]
pub trait RunHooks<C: Send + Sync + 'static>: Send + Sync {
    /// Called before the model is invoked.
    async fn on_llm_start(
        &self,
        _context: &RunContext<C>,
        _agent: &Agent<C>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
    }

    /// Called after the model returns.
    async fn on_llm_end(
        &self,
        _context: &RunContext<C>,
        _agent: &Agent<C>,
        _response: &ModelResponse,
    ) {
    }

    /// Called when an agent starts (each time the current agent changes).
    async fn on_agent_start(&self, _context: &RunContext<C>, _agent: &Agent<C>) {}

    /// Called when an agent produces a final output.
    async fn on_agent_end(&self, _context: &RunContext<C>, _agent: &Agent<C>, _output: &Value) {}

    /// Called when a handoff occurs.
    async fn on_handoff(
        &self,
        _context: &RunContext<C>,
        _from_agent: &Agent<C>,
        _to_agent: &Agent<C>,
    ) {
    }

    /// Called before a function tool is invoked.
    async fn on_tool_start(&self, _context: &ToolContext<C>, _agent: &Agent<C>, _tool: &Tool<C>) {}

    /// Called after a function tool is invoked.
    async fn on_tool_end(
        &self,
        _context: &ToolContext<C>,
        _agent: &Agent<C>,
        _tool: &Tool<C>,
        _result: &ToolOutput,
    ) {
    }
}

/// Callbacks scoped to a specific agent instance.
#[async_trait]
pub trait AgentHooks<C: Send + Sync + 'static>: Send + Sync {
    /// Called before the agent is invoked.
    async fn on_start(&self, _context: &RunContext<C>, _agent: &Agent<C>) {}

    /// Called when the agent produces a final output.
    async fn on_end(&self, _context: &RunContext<C>, _agent: &Agent<C>, _output: &Value) {}

    /// Called when this agent is being handed off to.
    async fn on_handoff(&self, _context: &RunContext<C>, _agent: &Agent<C>, _source: &Agent<C>) {}

    /// Called before a tool is invoked.
    async fn on_tool_start(&self, _context: &ToolContext<C>, _agent: &Agent<C>, _tool: &Tool<C>) {}

    /// Called after a tool is invoked.
    async fn on_tool_end(
        &self,
        _context: &ToolContext<C>,
        _agent: &Agent<C>,
        _tool: &Tool<C>,
        _result: &ToolOutput,
    ) {
    }

    /// Called before the model is invoked.
    async fn on_llm_start(
        &self,
        _context: &RunContext<C>,
        _agent: &Agent<C>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
    }

    /// Called after the model returns.
    async fn on_llm_end(
        &self,
        _context: &RunContext<C>,
        _agent: &Agent<C>,
        _response: &ModelResponse,
    ) {
    }
}

/// A `RunHooks` implementation that does nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopHooks;

#[async_trait]
impl<C: Send + Sync + 'static> RunHooks<C> for NoopHooks {}
