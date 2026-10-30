//! Realtime agents. Mirrors `agents.realtime.agent`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

#[cfg(feature = "mcp")]
use crate::agent::MCPConfig;
use crate::error::{AgentsError, Result};
use crate::guardrail::OutputGuardrail;
use crate::prompts::Prompt;
use crate::realtime::config::RealtimeSessionModelSettings;
use crate::realtime::handoffs::RealtimeHandoff;
use crate::run_context::RunContext;
use crate::tool::{FunctionTool, Tool, ToolOutput};
use crate::util::BoxFuture;

type DynamicInstructionsFn<C> = Arc<
    dyn Fn(RunContext<C>, Arc<RealtimeAgent<C>>) -> BoxFuture<'static, Result<String>>
        + Send
        + Sync,
>;

/// Static instructions, or a function producing them when the session starts or hands off.
pub enum RealtimeInstructions<C> {
    /// A fixed system prompt.
    Static(String),
    /// A function called with the run context and the agent.
    Dynamic(DynamicInstructionsFn<C>),
}

impl<C> Clone for RealtimeInstructions<C> {
    fn clone(&self) -> Self {
        match self {
            RealtimeInstructions::Static(s) => RealtimeInstructions::Static(s.clone()),
            RealtimeInstructions::Dynamic(f) => RealtimeInstructions::Dynamic(f.clone()),
        }
    }
}

impl<C> fmt::Debug for RealtimeInstructions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealtimeInstructions::Static(s) => f.debug_tuple("Static").field(s).finish(),
            RealtimeInstructions::Dynamic(_) => write!(f, "Dynamic(..)"),
        }
    }
}

/// An agent used inside a [`crate::realtime::RealtimeSession`].
///
/// Compared with [`crate::Agent`], a realtime agent has no model, output type or tool-use
/// behavior: every agent in a session is served by the same realtime model, and realtime agents
/// do not produce structured output.
///
/// `C` is the type of the app context passed to tools, handoffs and guardrails.
pub struct RealtimeAgent<C> {
    /// The agent name.
    pub name: String,
    /// Description used when the agent is a handoff target.
    pub handoff_description: Option<String>,
    /// System prompt, static or dynamic.
    pub instructions: Option<RealtimeInstructions<C>>,
    /// Responses API prompt template.
    pub prompt: Option<Prompt>,
    /// Tools exposed to the model.
    pub tools: Vec<Tool<C>>,
    /// MCP servers whose tools are exposed to the model alongside `tools`.
    #[cfg(feature = "mcp")]
    pub mcp_servers: Vec<Arc<dyn crate::mcp::MCPServer>>,
    /// How MCP tools are prepared.
    #[cfg(feature = "mcp")]
    pub mcp_config: MCPConfig<C>,
    /// Agents this one can hand off to.
    pub handoffs: Vec<RealtimeHandoff<C>>,
    /// Guardrails run on the assistant transcript.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Session settings applied when this agent becomes active.
    pub model_settings: Option<RealtimeSessionModelSettings>,
    /// Lifecycle callbacks for this agent, in addition to the session's own.
    pub hooks: Option<Arc<dyn RealtimeHooks<C>>>,
}

impl<C> Clone for RealtimeAgent<C> {
    fn clone(&self) -> Self {
        RealtimeAgent {
            name: self.name.clone(),
            handoff_description: self.handoff_description.clone(),
            instructions: self.instructions.clone(),
            prompt: self.prompt.clone(),
            tools: self.tools.clone(),
            #[cfg(feature = "mcp")]
            mcp_servers: self.mcp_servers.clone(),
            #[cfg(feature = "mcp")]
            mcp_config: self.mcp_config.clone(),
            handoffs: self.handoffs.clone(),
            output_guardrails: self.output_guardrails.clone(),
            model_settings: self.model_settings.clone(),
            hooks: self.hooks.clone(),
        }
    }
}

impl<C> fmt::Debug for RealtimeAgent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeAgent")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("tools", &self.tools)
            .field("handoffs", &self.handoffs)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> RealtimeAgent<C> {
    /// A realtime agent with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        RealtimeAgent {
            name: name.into(),
            handoff_description: None,
            instructions: None,
            prompt: None,
            tools: Vec::new(),
            #[cfg(feature = "mcp")]
            mcp_servers: Vec::new(),
            #[cfg(feature = "mcp")]
            mcp_config: MCPConfig::default(),
            handoffs: Vec::new(),
            output_guardrails: Vec::new(),
            model_settings: None,
            hooks: None,
        }
    }

    /// Set static instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(RealtimeInstructions::Static(instructions.into()));
        self
    }

    /// Set instructions produced at run time.
    pub fn with_instructions_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<RealtimeAgent<C>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.instructions = Some(RealtimeInstructions::Dynamic(Arc::new(
            move |ctx, agent| Box::pin(f(ctx, agent)),
        )));
        self
    }

    /// Set the description used when this agent is a handoff target.
    pub fn with_handoff_description(mut self, description: impl Into<String>) -> Self {
        self.handoff_description = Some(description.into());
        self
    }

    /// Set the Responses API prompt template.
    pub fn with_prompt(mut self, prompt: Prompt) -> Self {
        self.prompt = Some(prompt);
        self
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool: impl Into<Tool<C>>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add several tools.
    pub fn with_tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Tool<C>>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Add an MCP server whose tools the model may call.
    #[cfg(feature = "mcp")]
    pub fn with_mcp_server(mut self, server: Arc<dyn crate::mcp::MCPServer>) -> Self {
        self.mcp_servers.push(server);
        self
    }

    /// Set how MCP tools are prepared (strict schemas, server-prefixed names).
    #[cfg(feature = "mcp")]
    pub fn with_mcp_config(mut self, config: MCPConfig<C>) -> Self {
        self.mcp_config = config;
        self
    }

    /// Add a handoff.
    pub fn with_handoff(mut self, handoff: impl Into<RealtimeHandoff<C>>) -> Self {
        self.handoffs.push(handoff.into());
        self
    }

    /// Add several handoffs.
    pub fn with_handoffs<I, H>(mut self, handoffs: I) -> Self
    where
        I: IntoIterator<Item = H>,
        H: Into<RealtimeHandoff<C>>,
    {
        self.handoffs.extend(handoffs.into_iter().map(Into::into));
        self
    }

    /// Add an output guardrail run on the assistant transcript.
    pub fn with_output_guardrail(mut self, guardrail: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Set the session settings applied when this agent becomes active.
    pub fn with_model_settings(mut self, settings: RealtimeSessionModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    /// Set the lifecycle callbacks for this agent.
    pub fn with_hooks(mut self, hooks: Arc<dyn RealtimeHooks<C>>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Wrap the agent in an `Arc`, as sessions and handoffs expect.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl<C> RealtimeAgent<C> {
    /// Resolve the system prompt for this agent.
    pub async fn get_system_prompt(
        self: &Arc<Self>,
        context: &RunContext<C>,
    ) -> Result<Option<String>> {
        match &self.instructions {
            None => Ok(None),
            Some(RealtimeInstructions::Static(s)) => Ok(Some(s.clone())),
            Some(RealtimeInstructions::Dynamic(f)) => {
                Ok(Some(f(context.clone(), self.clone()).await?))
            }
        }
    }

    /// The function tools declared on this agent, skipping hosted tools, which realtime models
    /// reject.
    ///
    /// This does not include MCP tools; use [`RealtimeAgent::get_all_tools`] for those, which
    /// also rejects a hosted tool instead of skipping it.
    pub fn function_tools(&self) -> Vec<Arc<FunctionTool<C>>> {
        self.tools
            .iter()
            .filter_map(|tool| tool.as_function().cloned())
            .collect()
    }

    /// Fail when the agent declares a tool a realtime session cannot run.
    ///
    /// Port of the `Tool {name} is unsupported. Must be a function tool.` check in
    /// `_tools_to_session_tools`.
    fn reject_hosted_tools(&self) -> Result<()> {
        for tool in &self.tools {
            if tool.as_function().is_none() {
                return Err(AgentsError::user(format!(
                    "Tool {} is unsupported. Must be a function tool.",
                    tool.name()
                )));
            }
        }
        Ok(())
    }
}

impl<C: Send + Sync + 'static> RealtimeAgent<C> {
    /// Every enabled tool the model may call, including the tools of this agent's MCP servers.
    ///
    /// MCP tools come first, as in [`crate::Agent::get_all_tools`]. A hosted tool is an error:
    /// realtime models accept function tools only.
    pub async fn get_all_tools(
        &self,
        context: &RunContext<C>,
    ) -> Result<Vec<Arc<FunctionTool<C>>>> {
        self.reject_hosted_tools()?;
        let mut tools: Vec<Arc<FunctionTool<C>>> = Vec::new();

        #[cfg(feature = "mcp")]
        if !self.mcp_servers.is_empty() {
            let mcp_tools = crate::mcp::util::get_all_function_tools(
                &self.mcp_servers,
                &self.mcp_config,
                context,
                &self.name,
            )
            .await?;
            tools.extend(
                mcp_tools
                    .iter()
                    .filter_map(|tool| tool.as_function().cloned()),
            );
        }

        for tool in self.function_tools() {
            if tool.is_enabled.evaluate(context, &self.name).await {
                tools.push(tool);
            }
        }
        Ok(tools)
    }
}

/// Callbacks for the lifecycle of a realtime session.
///
/// The same moments are also published as [`crate::realtime::RealtimeSessionEvent`]s; hooks are
/// for code that wants to react inline, next to the session, rather than drive the event stream.
/// Mirrors `RealtimeAgentHooks` / `RealtimeRunHooks` in the Python SDK.
#[async_trait::async_trait]
pub trait RealtimeHooks<C>: Send + Sync + fmt::Debug {
    /// An agent started a turn.
    async fn on_agent_start(&self, _context: &RunContext<C>, _agent: &Arc<RealtimeAgent<C>>) {}

    /// An agent finished a turn.
    async fn on_agent_end(&self, _context: &RunContext<C>, _agent: &Arc<RealtimeAgent<C>>) {}

    /// An agent handed off to another agent.
    async fn on_handoff(
        &self,
        _context: &RunContext<C>,
        _from_agent: &Arc<RealtimeAgent<C>>,
        _to_agent: &Arc<RealtimeAgent<C>>,
    ) {
    }

    /// A tool call is about to run.
    async fn on_tool_start(
        &self,
        _context: &RunContext<C>,
        _agent: &Arc<RealtimeAgent<C>>,
        _tool: &Arc<FunctionTool<C>>,
    ) {
    }

    /// A tool call finished.
    async fn on_tool_end(
        &self,
        _context: &RunContext<C>,
        _agent: &Arc<RealtimeAgent<C>>,
        _tool: &Arc<FunctionTool<C>>,
        _output: &ToolOutput,
    ) {
    }
}
