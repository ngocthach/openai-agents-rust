//! Starting realtime sessions. Mirrors `agents.realtime.runner`.

use std::fmt;
use std::sync::Arc;

use crate::error::Result;
use crate::realtime::agent::RealtimeAgent;
use crate::realtime::config::RealtimeRunConfig;
use crate::realtime::model::{RealtimeModel, RealtimeModelConfig};
use crate::realtime::openai_model::OpenAIRealtimeWebSocketModel;
use crate::realtime::session::RealtimeSession;
use crate::run_context::RunContext;

/// The realtime counterpart of [`crate::Runner`].
///
/// It keeps a persistent connection to the model; the [`RealtimeSession`] it returns owns the
/// local history, executes tools, runs guardrails and performs handoffs.
///
/// ```no_run
/// # use openai_agents_sdk::realtime::{RealtimeAgent, RealtimeRunner};
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = RealtimeAgent::<()>::new("Assistant")
///     .with_instructions("You are a helpful voice assistant")
///     .into_arc();
/// let session = RealtimeRunner::new(agent).run(()).await?;
/// session.send_message("Hello").await?;
/// # Ok(())
/// # }
/// ```
pub struct RealtimeRunner<C> {
    starting_agent: Arc<RealtimeAgent<C>>,
    model: Arc<dyn RealtimeModel>,
    config: Option<RealtimeRunConfig<C>>,
    model_config: Option<RealtimeModelConfig>,
}

impl<C> fmt::Debug for RealtimeRunner<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeRunner")
            .field("starting_agent", &self.starting_agent.name)
            .field("model", &self.model)
            .field("config", &self.config)
            .field("model_config", &self.model_config)
            .finish()
    }
}

impl<C: Send + Sync + 'static> RealtimeRunner<C> {
    /// A runner that starts with `agent` over the OpenAI realtime WebSocket transport.
    pub fn new(agent: Arc<RealtimeAgent<C>>) -> Self {
        RealtimeRunner {
            starting_agent: agent,
            model: Arc::new(OpenAIRealtimeWebSocketModel::new()),
            config: None,
            model_config: None,
        }
    }

    /// Use a custom transport, for example a fake model in tests.
    pub fn with_model(mut self, model: Arc<dyn RealtimeModel>) -> Self {
        self.model = model;
        self
    }

    /// Set the run configuration.
    pub fn with_config(mut self, config: RealtimeRunConfig<C>) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the options used when connecting to the model.
    pub fn with_model_config(mut self, model_config: RealtimeModelConfig) -> Self {
        self.model_config = Some(model_config);
        self
    }

    /// Connect to the model and return a live session.
    ///
    /// Unlike the Python SDK, which returns an un-entered session you later enter, this connects
    /// before returning: the session is already streaming events when `run` resolves.
    pub async fn run(&self, context: C) -> Result<RealtimeSession<C>> {
        self.run_with_context(RunContext::new(context)).await
    }

    /// Connect using an existing [`RunContext`], so usage and approvals are shared with a wider
    /// application.
    pub async fn run_with_context(&self, context: RunContext<C>) -> Result<RealtimeSession<C>> {
        RealtimeSession::connect_with_context(
            self.model.clone(),
            self.starting_agent.clone(),
            context,
            self.model_config.clone(),
            self.config.clone(),
        )
        .await
    }

    /// The agent the next session starts with.
    pub fn starting_agent(&self) -> &Arc<RealtimeAgent<C>> {
        &self.starting_agent
    }

    /// The transport the next session uses.
    pub fn model(&self) -> &Arc<dyn RealtimeModel> {
        &self.model
    }
}
