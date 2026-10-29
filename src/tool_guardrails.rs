//! Guardrails around function tool execution. Mirrors `agents.tool_guardrails`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::tool::ToolOutput;
use crate::tool_context::ToolContext;
use crate::util::BoxFuture;

/// What to do after a tool guardrail runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolGuardrailBehavior {
    /// Continue normally.
    Allow,
    /// Skip the tool (input) or replace its output (output) with `message`.
    RejectContent { message: String },
    /// Halt the run with a tripwire error.
    RaiseException,
}

/// Output of a tool guardrail function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolGuardrailFunctionOutput {
    /// Optional information about the check.
    #[serde(default)]
    pub output_info: Value,
    pub behavior: ToolGuardrailBehavior,
}

impl ToolGuardrailFunctionOutput {
    /// Allow the call.
    pub fn allow() -> Self {
        ToolGuardrailFunctionOutput {
            output_info: Value::Null,
            behavior: ToolGuardrailBehavior::Allow,
        }
    }

    /// Reject the call / output and send `message` to the model instead.
    pub fn reject_content(message: impl Into<String>) -> Self {
        ToolGuardrailFunctionOutput {
            output_info: Value::Null,
            behavior: ToolGuardrailBehavior::RejectContent {
                message: message.into(),
            },
        }
    }

    /// Halt the run.
    pub fn raise_exception() -> Self {
        ToolGuardrailFunctionOutput {
            output_info: Value::Null,
            behavior: ToolGuardrailBehavior::RaiseException,
        }
    }

    /// Attach output info.
    pub fn with_output_info(mut self, info: Value) -> Self {
        self.output_info = info;
        self
    }
}

/// Data passed to a tool input guardrail.
pub struct ToolInputGuardrailData<C> {
    pub context: ToolContext<C>,
    pub agent_name: String,
}

impl<C> Clone for ToolInputGuardrailData<C> {
    fn clone(&self) -> Self {
        ToolInputGuardrailData {
            context: self.context.clone(),
            agent_name: self.agent_name.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolInputGuardrailData<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolInputGuardrailData")
            .field("context", &self.context)
            .field("agent_name", &self.agent_name)
            .finish()
    }
}

/// Data passed to a tool output guardrail.
pub struct ToolOutputGuardrailData<C> {
    pub context: ToolContext<C>,
    pub agent_name: String,
    pub output: ToolOutput,
}

impl<C> Clone for ToolOutputGuardrailData<C> {
    fn clone(&self) -> Self {
        ToolOutputGuardrailData {
            context: self.context.clone(),
            agent_name: self.agent_name.clone(),
            output: self.output.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolOutputGuardrailData<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolOutputGuardrailData")
            .field("context", &self.context)
            .field("agent_name", &self.agent_name)
            .field("output", &self.output)
            .finish()
    }
}

/// Result of a tool input guardrail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInputGuardrailResult {
    pub guardrail_name: String,
    pub tool_name: String,
    pub output: ToolGuardrailFunctionOutput,
}

/// Result of a tool output guardrail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutputGuardrailResult {
    pub guardrail_name: String,
    pub tool_name: String,
    pub output: ToolGuardrailFunctionOutput,
}

type InputFn<C> = Arc<
    dyn Fn(ToolInputGuardrailData<C>) -> BoxFuture<'static, Result<ToolGuardrailFunctionOutput>>
        + Send
        + Sync,
>;

/// A guardrail that runs before a function tool executes.
pub struct ToolInputGuardrail<C> {
    pub name: String,
    func: InputFn<C>,
}

impl<C> Clone for ToolInputGuardrail<C> {
    fn clone(&self) -> Self {
        ToolInputGuardrail {
            name: self.name.clone(),
            func: self.func.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolInputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolInputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl<C: Send + Sync + 'static> ToolInputGuardrail<C> {
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ToolInputGuardrailData<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolGuardrailFunctionOutput>> + Send + 'static,
    {
        ToolInputGuardrail {
            name: name.into(),
            func: Arc::new(move |d| Box::pin(f(d))),
        }
    }

    pub async fn run(&self, data: ToolInputGuardrailData<C>) -> Result<ToolGuardrailFunctionOutput> {
        (self.func)(data).await
    }
}

type OutputFn<C> = Arc<
    dyn Fn(ToolOutputGuardrailData<C>) -> BoxFuture<'static, Result<ToolGuardrailFunctionOutput>>
        + Send
        + Sync,
>;

/// A guardrail that runs after a function tool executes.
pub struct ToolOutputGuardrail<C> {
    pub name: String,
    func: OutputFn<C>,
}

impl<C> Clone for ToolOutputGuardrail<C> {
    fn clone(&self) -> Self {
        ToolOutputGuardrail {
            name: self.name.clone(),
            func: self.func.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolOutputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolOutputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl<C: Send + Sync + 'static> ToolOutputGuardrail<C> {
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ToolOutputGuardrailData<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolGuardrailFunctionOutput>> + Send + 'static,
    {
        ToolOutputGuardrail {
            name: name.into(),
            func: Arc::new(move |d| Box::pin(f(d))),
        }
    }

    pub async fn run(&self, data: ToolOutputGuardrailData<C>) -> Result<ToolGuardrailFunctionOutput> {
        (self.func)(data).await
    }
}
