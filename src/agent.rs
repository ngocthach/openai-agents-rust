//! Agents. Mirrors `agents.agent`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use crate::error::Result;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoffs::Handoff;
use crate::lifecycle::AgentHooks;
use crate::model_settings::ModelSettings;
use crate::models::interface::Model;
use crate::prompts::{AgentPrompt, Prompt};
use crate::run_context::RunContext;
use crate::tool::{FunctionToolResult, Tool};
use crate::util::BoxFuture;

type DynamicInstructionsFn<C> =
    Arc<dyn Fn(RunContext<C>, Arc<Agent<C>>) -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// Static instructions or a function producing them at run time.
pub enum Instructions<C> {
    Static(String),
    Dynamic(DynamicInstructionsFn<C>),
}

impl<C> Clone for Instructions<C> {
    fn clone(&self) -> Self {
        match self {
            Instructions::Static(s) => Instructions::Static(s.clone()),
            Instructions::Dynamic(f) => Instructions::Dynamic(f.clone()),
        }
    }
}

impl<C> fmt::Debug for Instructions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instructions::Static(s) => f.debug_tuple("Static").field(s).finish(),
            Instructions::Dynamic(_) => write!(f, "Dynamic(..)"),
        }
    }
}

/// Which model an agent uses.
#[derive(Clone, Debug)]
pub enum AgentModel {
    /// A model name resolved by the run's model provider.
    Name(String),
    /// A concrete model instance.
    Instance(Arc<dyn Model>),
}

impl From<&str> for AgentModel {
    fn from(value: &str) -> Self {
        AgentModel::Name(value.to_string())
    }
}

impl From<String> for AgentModel {
    fn from(value: String) -> Self {
        AgentModel::Name(value)
    }
}

impl<M: Model + 'static> From<Arc<M>> for AgentModel {
    fn from(value: Arc<M>) -> Self {
        AgentModel::Instance(value)
    }
}

/// Result of a custom tool-use behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolsToFinalOutputResult {
    /// Whether this is the final output.
    pub is_final_output: bool,
    /// The final output, when `is_final_output` is true.
    pub final_output: Option<Value>,
}

impl ToolsToFinalOutputResult {
    /// Not final; run the LLM again.
    pub fn not_final() -> Self {
        ToolsToFinalOutputResult {
            is_final_output: false,
            final_output: None,
        }
    }
    /// Final output.
    pub fn final_output(output: impl Into<Value>) -> Self {
        ToolsToFinalOutputResult {
            is_final_output: true,
            final_output: Some(output.into()),
        }
    }
}

/// Stop the run when any of these tools is called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopAtTools {
    pub stop_at_tool_names: Vec<String>,
}

type ToolsToFinalOutputFn<C> = Arc<
    dyn Fn(RunContext<C>, Vec<FunctionToolResult>) -> BoxFuture<'static, Result<ToolsToFinalOutputResult>>
        + Send
        + Sync,
>;

/// How tool outputs are handled.
pub enum ToolUseBehavior<C> {
    /// Run the tools, then the LLM again with the results (default).
    RunLlmAgain,
    /// The output of the first tool call is the final output.
    StopOnFirstTool,
    /// Stop if any of the named tools is called; its output is the final output.
    StopAtTools(StopAtTools),
    /// A custom decision function.
    Custom(ToolsToFinalOutputFn<C>),
}

impl<C> Clone for ToolUseBehavior<C> {
    fn clone(&self) -> Self {
        match self {
            ToolUseBehavior::RunLlmAgain => ToolUseBehavior::RunLlmAgain,
            ToolUseBehavior::StopOnFirstTool => ToolUseBehavior::StopOnFirstTool,
            ToolUseBehavior::StopAtTools(s) => ToolUseBehavior::StopAtTools(s.clone()),
            ToolUseBehavior::Custom(f) => ToolUseBehavior::Custom(f.clone()),
        }
    }
}

impl<C> fmt::Debug for ToolUseBehavior<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolUseBehavior::RunLlmAgain => write!(f, "RunLlmAgain"),
            ToolUseBehavior::StopOnFirstTool => write!(f, "StopOnFirstTool"),
            ToolUseBehavior::StopAtTools(s) => f.debug_tuple("StopAtTools").field(s).finish(),
            ToolUseBehavior::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

impl<C> ToolUseBehavior<C> {
    /// Build a custom tool use behavior.
    pub fn custom<F, Fut>(f: F) -> Self
    where
        F: Fn(RunContext<C>, Vec<FunctionToolResult>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolsToFinalOutputResult>> + Send + 'static,
    {
        ToolUseBehavior::Custom(Arc::new(move |c, r| Box::pin(f(c, r))))
    }
}

/// Agent-level MCP configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MCPConfig {
    /// Try to convert MCP tool schemas to strict JSON schema.
    pub convert_schemas_to_strict: bool,
    /// Prefix local MCP tool names with their server name.
    pub include_server_in_tool_names: bool,
}

/// An agent: an LLM configured with instructions, tools, guardrails, handoffs and more.
///
/// `C` is the type of the app context passed to tools, hooks and guardrails.
pub struct Agent<C> {
    /// The agent name.
    pub name: String,
    /// Description used when the agent is a handoff target.
    pub handoff_description: Option<String>,
    /// System prompt, static or dynamic.
    pub instructions: Option<Instructions<C>>,
    /// Responses API prompt template.
    pub prompt: Option<AgentPrompt<C>>,
    /// Handoff targets.
    pub handoffs: Vec<Handoff<C>>,
    /// Model to use (defaults to the run config model or the provider default).
    pub model: Option<AgentModel>,
    /// Model settings.
    pub model_settings: ModelSettings,
    /// Whether `model_settings` were set explicitly (otherwise model defaults apply).
    pub(crate) model_settings_explicit: bool,
    /// Tools.
    pub tools: Vec<Tool<C>>,
    /// MCP servers providing tools.
    #[cfg(feature = "mcp")]
    pub mcp_servers: Vec<Arc<dyn crate::mcp::MCPServer>>,
    /// MCP tool preparation config.
    pub mcp_config: MCPConfig,
    /// Guardrails on the first input of a run.
    pub input_guardrails: Vec<InputGuardrail<C>>,
    /// Guardrails on the final output.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Structured output type.
    pub output_type: Option<Arc<dyn AgentOutputSchemaBase>>,
    /// Agent-scoped lifecycle hooks.
    pub hooks: Option<Arc<dyn AgentHooks<C>>>,
    /// How tool outputs are handled.
    pub tool_use_behavior: ToolUseBehavior<C>,
    /// Reset `tool_choice` to `auto` after a tool call to avoid loops.
    pub reset_tool_choice: bool,
}

impl<C> Clone for Agent<C> {
    fn clone(&self) -> Self {
        Agent {
            name: self.name.clone(),
            handoff_description: self.handoff_description.clone(),
            instructions: self.instructions.clone(),
            prompt: self.prompt.clone(),
            handoffs: self.handoffs.clone(),
            model: self.model.clone(),
            model_settings: self.model_settings.clone(),
            model_settings_explicit: self.model_settings_explicit,
            tools: self.tools.clone(),
            #[cfg(feature = "mcp")]
            mcp_servers: self.mcp_servers.clone(),
            mcp_config: self.mcp_config.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            output_type: self.output_type.clone(),
            hooks: self.hooks.clone(),
            tool_use_behavior: self.tool_use_behavior.clone(),
            reset_tool_choice: self.reset_tool_choice,
        }
    }
}

impl<C> fmt::Debug for Agent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("handoff_description", &self.handoff_description)
            .field("instructions", &self.instructions)
            .field("model", &self.model)
            .field("tools", &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>())
            .field(
                "handoffs",
                &self.handoffs.iter().map(|h| h.agent_name.clone()).collect::<Vec<_>>(),
            )
            .field("output_type", &self.output_type.as_ref().map(|o| o.name()))
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> Agent<C> {
    /// Create an agent with the given name and default settings.
    pub fn new(name: impl Into<String>) -> Self {
        Agent {
            name: name.into(),
            handoff_description: None,
            instructions: None,
            prompt: None,
            handoffs: Vec::new(),
            model: None,
            model_settings: ModelSettings::default(),
            model_settings_explicit: false,
            tools: Vec::new(),
            #[cfg(feature = "mcp")]
            mcp_servers: Vec::new(),
            mcp_config: MCPConfig::default(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_type: None,
            hooks: None,
            tool_use_behavior: ToolUseBehavior::RunLlmAgain,
            reset_tool_choice: true,
        }
    }

    /// Set static instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(Instructions::Static(instructions.into()));
        self
    }

    /// Set dynamic instructions computed from the context at run time.
    pub fn with_dynamic_instructions<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<Agent<C>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.instructions = Some(Instructions::Dynamic(Arc::new(move |c, a| Box::pin(f(c, a)))));
        self
    }

    /// Set the handoff description.
    pub fn with_handoff_description(mut self, description: impl Into<String>) -> Self {
        self.handoff_description = Some(description.into());
        self
    }

    /// Set a Responses API prompt template.
    pub fn with_prompt(mut self, prompt: impl Into<AgentPrompt<C>>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set a dynamic prompt.
    pub fn with_dynamic_prompt<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<Agent<C>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<Prompt>>> + Send + 'static,
    {
        self.prompt = Some(AgentPrompt::dynamic(f));
        self
    }

    /// Set the model by name or instance.
    pub fn with_model(mut self, model: impl Into<AgentModel>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set a model instance.
    pub fn with_model_instance(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(AgentModel::Instance(model));
        self
    }

    /// Set model settings (explicit settings override model defaults).
    pub fn with_model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self.model_settings_explicit = true;
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

    /// Add a handoff target.
    pub fn with_handoff(mut self, handoff: impl Into<Handoff<C>>) -> Self {
        self.handoffs.push(handoff.into());
        self
    }

    /// Add several handoff targets.
    pub fn with_handoffs<I, H>(mut self, handoffs: I) -> Self
    where
        I: IntoIterator<Item = H>,
        H: Into<Handoff<C>>,
    {
        self.handoffs.extend(handoffs.into_iter().map(Into::into));
        self
    }

    /// Add an MCP server.
    #[cfg(feature = "mcp")]
    pub fn with_mcp_server(mut self, server: Arc<dyn crate::mcp::MCPServer>) -> Self {
        self.mcp_servers.push(server);
        self
    }

    /// Set MCP configuration.
    pub fn with_mcp_config(mut self, config: MCPConfig) -> Self {
        self.mcp_config = config;
        self
    }

    /// Add an input guardrail.
    pub fn with_input_guardrail(mut self, guardrail: InputGuardrail<C>) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail.
    pub fn with_output_guardrail(mut self, guardrail: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Use structured output of type `T` (strict JSON schema).
    pub fn with_output_type<T: JsonSchema + DeserializeOwned + 'static>(self) -> Self {
        let schema = AgentOutputSchema::new::<T>()
            .unwrap_or_else(|e| panic!("invalid output type for agent: {e}"));
        self.with_output_schema(Arc::new(schema))
    }

    /// Use a custom output schema.
    pub fn with_output_schema(mut self, schema: Arc<dyn AgentOutputSchemaBase>) -> Self {
        self.output_type = Some(schema);
        self
    }

    /// Set agent hooks.
    pub fn with_hooks(mut self, hooks: Arc<dyn AgentHooks<C>>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set the tool use behavior.
    pub fn with_tool_use_behavior(mut self, behavior: ToolUseBehavior<C>) -> Self {
        self.tool_use_behavior = behavior;
        self
    }

    /// Set whether tool choice is reset after a tool call.
    pub fn with_reset_tool_choice(mut self, reset: bool) -> Self {
        self.reset_tool_choice = reset;
        self
    }

    /// Clone the agent and modify the copy (Python `agent.clone(...)`).
    pub fn clone_with(&self, f: impl FnOnce(&mut Agent<C>)) -> Agent<C> {
        let mut a = self.clone();
        f(&mut a);
        a
    }

    /// Resolve the system prompt for a run.
    pub async fn get_system_prompt(self: &Arc<Self>, ctx: &RunContext<C>) -> Result<Option<String>> {
        match &self.instructions {
            None => Ok(None),
            Some(Instructions::Static(s)) => Ok(Some(s.clone())),
            Some(Instructions::Dynamic(f)) => f(ctx.clone(), self.clone()).await.map(Some),
        }
    }

    /// Resolve the prompt template for a run.
    pub async fn get_prompt(self: &Arc<Self>, ctx: &RunContext<C>) -> Result<Option<Prompt>> {
        match &self.prompt {
            None => Ok(None),
            Some(p) => p.resolve(ctx, self.clone()).await,
        }
    }

    /// Enabled handoffs for this run.
    pub async fn get_enabled_handoffs(&self, ctx: &RunContext<C>) -> Vec<Handoff<C>> {
        let mut out = Vec::new();
        for h in &self.handoffs {
            if h.is_enabled.evaluate(ctx, &self.name).await {
                out.push(h.clone());
            }
        }
        out
    }

    /// All enabled tools, including MCP tools.
    pub async fn get_all_tools(&self, ctx: &RunContext<C>) -> Result<Vec<Tool<C>>> {
        let mut out = Vec::new();
        #[cfg(feature = "mcp")]
        {
            let mcp_tools = crate::mcp::util::get_all_function_tools(
                &self.mcp_servers,
                &self.mcp_config,
                ctx,
                &self.name,
            )
            .await?;
            out.extend(mcp_tools);
        }
        for t in &self.tools {
            match t {
                Tool::Function(f) => {
                    if f.is_enabled.evaluate(ctx, &self.name).await {
                        out.push(t.clone());
                    }
                }
                other => out.push(other.clone()),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::FunctionTool;

    /// A context type that is neither `Clone` nor `Debug`.
    struct NotCloneCtx {
        _m: std::sync::Mutex<u32>,
    }

    #[tokio::test]
    async fn agents_with_non_clone_context_are_cloneable() {
        let tool = FunctionTool::<NotCloneCtx>::new_no_args("t", "d", |_c| async { Ok("x") });
        let agent = Agent::<NotCloneCtx>::new("a")
            .with_instructions("hi")
            .with_tool(tool);
        let copy = agent.clone_with(|a| a.name = "b".into());
        assert_eq!(copy.name, "b");
        assert_eq!(copy.tools.len(), 1);
        let ctx = RunContext::new(NotCloneCtx {
            _m: std::sync::Mutex::new(0),
        });
        let tools = agent.get_all_tools(&ctx).await.unwrap();
        assert_eq!(tools.len(), 1);
        let _ = format!("{agent:?}");
    }

    #[tokio::test]
    async fn dynamic_instructions() {
        let agent = Arc::new(Agent::<String>::new("a").with_dynamic_instructions(|ctx, agent| async move {
            Ok(format!("{} helps {}", agent.name, ctx.context()))
        }));
        let p = agent
            .get_system_prompt(&RunContext::new("Bob".to_string()))
            .await
            .unwrap();
        assert_eq!(p.as_deref(), Some("a helps Bob"));
    }
}
