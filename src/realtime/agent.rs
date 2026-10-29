//! Realtime agents. Mirrors `agents.realtime.agent`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use crate::error::Result;
use crate::guardrail::OutputGuardrail;
use crate::prompts::Prompt;
use crate::realtime::config::RealtimeSessionModelSettings;
use crate::realtime::handoffs::RealtimeHandoff;
use crate::run_context::RunContext;
use crate::tool::{FunctionTool, Tool};
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
    /// Agents this one can hand off to.
    pub handoffs: Vec<RealtimeHandoff<C>>,
    /// Guardrails run on the assistant transcript.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Session settings applied when this agent becomes active.
    pub model_settings: Option<RealtimeSessionModelSettings>,
}

impl<C> Clone for RealtimeAgent<C> {
    fn clone(&self) -> Self {
        RealtimeAgent {
            name: self.name.clone(),
            handoff_description: self.handoff_description.clone(),
            instructions: self.instructions.clone(),
            prompt: self.prompt.clone(),
            tools: self.tools.clone(),
            handoffs: self.handoffs.clone(),
            output_guardrails: self.output_guardrails.clone(),
            model_settings: self.model_settings.clone(),
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
            handoffs: Vec::new(),
            output_guardrails: Vec::new(),
            model_settings: None,
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

    /// The function tools of this agent, skipping hosted tools, which realtime models reject.
    pub fn function_tools(&self) -> Vec<Arc<FunctionTool<C>>> {
        self.tools
            .iter()
            .filter_map(|tool| tool.as_function().cloned())
            .collect()
    }
}
