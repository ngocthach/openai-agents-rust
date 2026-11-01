//! Lifecycle hooks. Mirrors `agents.lifecycle`.

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::Agent;
use crate::error::Result;
use crate::items::{InputItem, ModelResponse};
use crate::run_context::RunContext;
use crate::tool::{Tool, ToolOutput};
use crate::tool_context::ToolContext;

/// Callbacks observing an entire `Runner::run` invocation, including handoffs.
///
/// All methods have no-op defaults.
///
/// Each callback has a fallible twin named `try_<callback>` that returns `Result<()>`; the runner
/// calls the twin, and an `Err` fails the run at the point where Python's raising hook would. The
/// twin's default calls the infallible callback and returns `Ok(())`, so an implementation written
/// against the infallible methods keeps working; override the `try_` method to fail the run.
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

    /// Fallible twin of [`RunHooks::on_llm_start`].
    async fn try_on_llm_start(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        system_prompt: Option<&str>,
        input_items: &[InputItem],
    ) -> Result<()> {
        self.on_llm_start(context, agent, system_prompt, input_items)
            .await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_llm_end`].
    async fn try_on_llm_end(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        response: &ModelResponse,
    ) -> Result<()> {
        self.on_llm_end(context, agent, response).await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_agent_start`].
    async fn try_on_agent_start(&self, context: &RunContext<C>, agent: &Agent<C>) -> Result<()> {
        self.on_agent_start(context, agent).await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_agent_end`].
    async fn try_on_agent_end(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        output: &Value,
    ) -> Result<()> {
        self.on_agent_end(context, agent, output).await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_handoff`].
    async fn try_on_handoff(
        &self,
        context: &RunContext<C>,
        from_agent: &Agent<C>,
        to_agent: &Agent<C>,
    ) -> Result<()> {
        self.on_handoff(context, from_agent, to_agent).await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_tool_start`].
    async fn try_on_tool_start(
        &self,
        context: &ToolContext<C>,
        agent: &Agent<C>,
        tool: &Tool<C>,
    ) -> Result<()> {
        self.on_tool_start(context, agent, tool).await;
        Ok(())
    }

    /// Fallible twin of [`RunHooks::on_tool_end`].
    async fn try_on_tool_end(
        &self,
        context: &ToolContext<C>,
        agent: &Agent<C>,
        tool: &Tool<C>,
        result: &ToolOutput,
    ) -> Result<()> {
        self.on_tool_end(context, agent, tool, result).await;
        Ok(())
    }
}

/// Callbacks scoped to a specific agent instance. Every callback has a fallible `try_` twin, as on
/// [`RunHooks`].
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

    /// Fallible twin of [`AgentHooks::on_start`].
    async fn try_on_start(&self, context: &RunContext<C>, agent: &Agent<C>) -> Result<()> {
        self.on_start(context, agent).await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_end`].
    async fn try_on_end(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        output: &Value,
    ) -> Result<()> {
        self.on_end(context, agent, output).await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_handoff`].
    async fn try_on_handoff(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        source: &Agent<C>,
    ) -> Result<()> {
        self.on_handoff(context, agent, source).await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_tool_start`].
    async fn try_on_tool_start(
        &self,
        context: &ToolContext<C>,
        agent: &Agent<C>,
        tool: &Tool<C>,
    ) -> Result<()> {
        self.on_tool_start(context, agent, tool).await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_tool_end`].
    async fn try_on_tool_end(
        &self,
        context: &ToolContext<C>,
        agent: &Agent<C>,
        tool: &Tool<C>,
        result: &ToolOutput,
    ) -> Result<()> {
        self.on_tool_end(context, agent, tool, result).await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_llm_start`].
    async fn try_on_llm_start(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        system_prompt: Option<&str>,
        input_items: &[InputItem],
    ) -> Result<()> {
        self.on_llm_start(context, agent, system_prompt, input_items)
            .await;
        Ok(())
    }

    /// Fallible twin of [`AgentHooks::on_llm_end`].
    async fn try_on_llm_end(
        &self,
        context: &RunContext<C>,
        agent: &Agent<C>,
        response: &ModelResponse,
    ) -> Result<()> {
        self.on_llm_end(context, agent, response).await;
        Ok(())
    }
}

/// A `RunHooks` implementation that does nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopHooks;

#[async_trait]
impl<C: Send + Sync + 'static> RunHooks<C> for NoopHooks {}
