//! The runner: the agent loop. Mirrors `agents.run`.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::RunInput;
use crate::lifecycle::RunHooks;
use crate::memory::Session;
use crate::result::{RunResult, RunResultStreaming, StreamMsg};
use crate::run_config::RunConfig;
use crate::run_context::RunContext;
use crate::run_error_handlers::RunErrorHandlers;
use crate::run_internal::{EventSink, run_loop};
use crate::tracing::scope::ScopeSnapshot;

/// Default maximum number of model turns per run.
pub const DEFAULT_MAX_TURNS: usize = 10;

/// Options for a run.
pub struct RunOptions<C> {
    /// The run context wrapping the app context.
    pub context: RunContext<C>,
    /// Maximum model turns (`None` disables the limit).
    pub max_turns: Option<usize>,
    /// Run-wide lifecycle hooks.
    pub hooks: Option<Arc<dyn RunHooks<C>>>,
    /// Run configuration.
    pub run_config: RunConfig<C>,
    /// Error handlers.
    pub error_handlers: RunErrorHandlers<C>,
    /// Continue a Responses API chain.
    pub previous_response_id: Option<String>,
    /// Chain responses automatically from the first turn.
    pub auto_previous_response_id: bool,
    /// Continue an OpenAI conversation.
    pub conversation_id: Option<String>,
    /// Session memory.
    pub session: Option<Arc<dyn Session>>,
    /// Set for nested `Agent::as_tool` runs that share the parent's run context.
    pub(crate) shares_parent_context: bool,
}

impl<C> fmt::Debug for RunOptions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunOptions")
            .field("max_turns", &self.max_turns)
            .field("run_config", &self.run_config)
            .field("previous_response_id", &self.previous_response_id)
            .field("conversation_id", &self.conversation_id)
            .field("session", &self.session)
            .finish_non_exhaustive()
    }
}

impl<C: Default + Send + Sync + 'static> Default for RunOptions<C> {
    fn default() -> Self {
        RunOptions::new(C::default())
    }
}

impl<C: Send + Sync + 'static> RunOptions<C> {
    /// Options with the given app context.
    pub fn new(context: C) -> Self {
        Self::with_run_context(RunContext::new(context))
    }

    /// Options reusing an existing run context (keeps usage and approvals).
    pub fn with_run_context(context: RunContext<C>) -> Self {
        RunOptions {
            context,
            max_turns: Some(DEFAULT_MAX_TURNS),
            hooks: None,
            run_config: RunConfig::default(),
            error_handlers: RunErrorHandlers::default(),
            previous_response_id: None,
            auto_previous_response_id: false,
            conversation_id: None,
            session: None,
            shares_parent_context: false,
        }
    }

    pub fn max_turns(mut self, max_turns: Option<usize>) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn hooks(mut self, hooks: Arc<dyn RunHooks<C>>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    pub fn run_config(mut self, run_config: RunConfig<C>) -> Self {
        self.run_config = run_config;
        self
    }

    pub fn error_handlers(mut self, handlers: RunErrorHandlers<C>) -> Self {
        self.error_handlers = handlers;
        self
    }

    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    pub fn auto_previous_response_id(mut self, auto: bool) -> Self {
        self.auto_previous_response_id = auto;
        self
    }

    pub fn conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    pub fn session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }
}

/// Anything that can be used as the starting agent of a run.
pub trait IntoAgentArc<C> {
    fn into_agent_arc(self) -> Arc<Agent<C>>;
}

impl<C> IntoAgentArc<C> for Arc<Agent<C>> {
    fn into_agent_arc(self) -> Arc<Agent<C>> {
        self
    }
}

impl<C> IntoAgentArc<C> for &Arc<Agent<C>> {
    fn into_agent_arc(self) -> Arc<Agent<C>> {
        self.clone()
    }
}

impl<C> IntoAgentArc<C> for Agent<C> {
    fn into_agent_arc(self) -> Arc<Agent<C>> {
        Arc::new(self)
    }
}

impl<C> IntoAgentArc<C> for &Agent<C> {
    fn into_agent_arc(self) -> Arc<Agent<C>> {
        Arc::new(self.clone())
    }
}

/// Runs agents.
///
/// The loop calls the model, runs tools and handoffs, and stops when the current agent produces a
/// final output, a tool needs approval, or `max_turns` is exceeded.
#[derive(Debug, Clone, Copy, Default)]
pub struct Runner;

impl Runner {
    /// Run `agent` with a default app context.
    pub async fn run<C: Default + Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
    ) -> Result<RunResult<C>> {
        Self::run_with_options(agent, input, RunOptions::default()).await
    }

    /// Run `agent` with explicit options.
    pub async fn run_with_options<C: Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
        options: RunOptions<C>,
    ) -> Result<RunResult<C>> {
        Box::pin(run_loop(
            agent.into_agent_arc(),
            input.into(),
            options,
            None,
        ))
        .await
    }

    /// Run synchronously on a new tokio runtime. Must not be called from within a runtime.
    pub fn run_sync<C: Default + Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
    ) -> Result<RunResult<C>> {
        Self::run_sync_with_options(agent, input, RunOptions::default())
    }

    /// Run synchronously with options. Must not be called from within a runtime.
    pub fn run_sync_with_options<C: Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
        options: RunOptions<C>,
    ) -> Result<RunResult<C>> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(AgentsError::user(
                "Runner::run_sync cannot be called from within an async runtime; use Runner::run",
            ));
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let agent = agent.into_agent_arc();
        let input = input.into();
        let result = rt.block_on(Self::run_with_options(agent, input, options));
        crate::tracing::flush_traces();
        result
    }

    /// Run in streaming mode with a default app context. Must be called within a tokio runtime.
    pub fn run_streamed<C: Default + Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
    ) -> RunResultStreaming<C> {
        Self::run_streamed_with_options(agent, input, RunOptions::default())
    }

    /// Run in streaming mode with options. Must be called within a tokio runtime.
    pub fn run_streamed_with_options<C: Send + Sync + 'static>(
        agent: impl IntoAgentArc<C>,
        input: impl Into<RunInput>,
        options: RunOptions<C>,
    ) -> RunResultStreaming<C> {
        let agent = agent.into_agent_arc();
        let input = input.into();
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let sink = EventSink {
            tx: tx.clone(),
            cancel_after_turn: cancel.clone(),
        };
        let scope = ScopeSnapshot::capture();
        let start_agent = agent.clone();
        let handle = tokio::spawn(async move {
            let result = scope
                .enter(Box::pin(run_loop(agent, input, options, Some(sink))))
                .await;
            let _ = tx.send(StreamMsg::Done(Box::new(result)));
        });
        RunResultStreaming::new(rx, start_agent, cancel, handle)
    }
}
