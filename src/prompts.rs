//! Responses API prompt templates. Mirrors `agents.prompts`.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Agent;
use crate::error::Result;
use crate::run_context::RunContext;
use crate::util::BoxFuture;

/// A prompt template variable value.
pub type PromptVariable = Value;

/// Reference to a prompt template created in the OpenAI platform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variables: Option<HashMap<String, PromptVariable>>,
}

impl Prompt {
    pub fn new(id: impl Into<String>) -> Self {
        Prompt {
            id: id.into(),
            version: None,
            variables: None,
        }
    }
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
    pub fn with_variable(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.variables
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }
}

type DynamicPromptFn<C> =
    Arc<dyn Fn(RunContext<C>, Arc<Agent<C>>) -> BoxFuture<'static, Result<Option<Prompt>>> + Send + Sync>;

/// A static prompt or a function generating it at run time.
pub enum AgentPrompt<C> {
    Static(Prompt),
    Dynamic(DynamicPromptFn<C>),
}

impl<C> Clone for AgentPrompt<C> {
    fn clone(&self) -> Self {
        match self {
            AgentPrompt::Static(p) => AgentPrompt::Static(p.clone()),
            AgentPrompt::Dynamic(f) => AgentPrompt::Dynamic(f.clone()),
        }
    }
}

impl<C> fmt::Debug for AgentPrompt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentPrompt::Static(p) => f.debug_tuple("Static").field(p).finish(),
            AgentPrompt::Dynamic(_) => write!(f, "Dynamic(..)"),
        }
    }
}

impl<C: Send + Sync + 'static> AgentPrompt<C> {
    /// Build a dynamic prompt.
    pub fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<Agent<C>>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<Prompt>>> + Send + 'static,
    {
        AgentPrompt::Dynamic(Arc::new(move |c, a| Box::pin(f(c, a))))
    }

    /// Resolve the prompt for a run.
    pub async fn resolve(&self, ctx: &RunContext<C>, agent: Arc<Agent<C>>) -> Result<Option<Prompt>> {
        match self {
            AgentPrompt::Static(p) => Ok(Some(p.clone())),
            AgentPrompt::Dynamic(f) => f(ctx.clone(), agent).await,
        }
    }
}

impl<C> From<Prompt> for AgentPrompt<C> {
    fn from(value: Prompt) -> Self {
        AgentPrompt::Static(value)
    }
}
