//! Handlers that turn selected run errors into a controlled final output. Mirrors
//! `agents.run_error_handlers`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::{InputItem, ModelResponse, RunItem};
use crate::run_context::RunContext;
use crate::util::BoxFuture;

/// Snapshot of run data passed to error handlers.
pub struct RunErrorData<C> {
    pub input: Vec<InputItem>,
    pub new_items: Vec<RunItem>,
    pub history: Vec<InputItem>,
    pub raw_responses: Vec<ModelResponse>,
    pub last_agent: Arc<Agent<C>>,
}

impl<C> fmt::Debug for RunErrorData<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunErrorData")
            .field("new_items", &self.new_items.len())
            .field("last_agent", &self.last_agent.name)
            .finish_non_exhaustive()
    }
}

/// Input to an error handler.
pub struct RunErrorHandlerInput<C> {
    /// The error being handled (`MaxTurnsExceeded`, `ModelRefusalError` or `ModelBehaviorError`).
    pub error: Arc<AgentsError>,
    pub context: RunContext<C>,
    pub run_data: RunErrorData<C>,
}

impl<C> fmt::Debug for RunErrorHandlerInput<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunErrorHandlerInput")
            .field("error", &self.error)
            .field("run_data", &self.run_data)
            .finish_non_exhaustive()
    }
}

/// Result returned by an error handler.
#[derive(Debug, Clone, PartialEq)]
pub struct RunErrorHandlerResult {
    /// The final output to return.
    pub final_output: Value,
    /// Append the output to history and session (default `true`).
    pub include_in_history: bool,
}

impl RunErrorHandlerResult {
    pub fn new(final_output: impl Into<Value>) -> Self {
        RunErrorHandlerResult {
            final_output: final_output.into(),
            include_in_history: true,
        }
    }

    pub fn without_history(mut self) -> Self {
        self.include_in_history = false;
        self
    }
}

/// A handler; returning `Ok(None)` declines recovery.
pub type RunErrorHandler<C> = Arc<
    dyn Fn(RunErrorHandlerInput<C>) -> BoxFuture<'static, Result<Option<RunErrorHandlerResult>>>
        + Send
        + Sync,
>;

/// Error handlers keyed by error kind.
pub struct RunErrorHandlers<C> {
    pub max_turns: Option<RunErrorHandler<C>>,
    pub model_refusal: Option<RunErrorHandler<C>>,
    pub invalid_final_output: Option<RunErrorHandler<C>>,
}

impl<C> Default for RunErrorHandlers<C> {
    fn default() -> Self {
        RunErrorHandlers {
            max_turns: None,
            model_refusal: None,
            invalid_final_output: None,
        }
    }
}

impl<C> Clone for RunErrorHandlers<C> {
    fn clone(&self) -> Self {
        RunErrorHandlers {
            max_turns: self.max_turns.clone(),
            model_refusal: self.model_refusal.clone(),
            invalid_final_output: self.invalid_final_output.clone(),
        }
    }
}

impl<C> fmt::Debug for RunErrorHandlers<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunErrorHandlers")
            .field("max_turns", &self.max_turns.is_some())
            .field("model_refusal", &self.model_refusal.is_some())
            .field("invalid_final_output", &self.invalid_final_output.is_some())
            .finish()
    }
}

fn wrap<C, F, Fut>(f: F) -> RunErrorHandler<C>
where
    F: Fn(RunErrorHandlerInput<C>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<RunErrorHandlerResult>>> + Send + 'static,
{
    Arc::new(move |i| Box::pin(f(i)))
}

impl<C> RunErrorHandlers<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle `MaxTurnsExceeded`.
    pub fn on_max_turns<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<RunErrorHandlerResult>>> + Send + 'static,
    {
        self.max_turns = Some(wrap(f));
        self
    }

    /// Handle `ModelRefusalError`.
    pub fn on_model_refusal<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<RunErrorHandlerResult>>> + Send + 'static,
    {
        self.model_refusal = Some(wrap(f));
        self
    }

    /// Handle invalid structured final output (`ModelBehaviorError`).
    pub fn on_invalid_final_output<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunErrorHandlerInput<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<RunErrorHandlerResult>>> + Send + 'static,
    {
        self.invalid_final_output = Some(wrap(f));
        self
    }
}
