//! Input and output guardrails. Mirrors `agents.guardrail`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Agent;
use crate::error::Result;
use crate::items::InputItem;
use crate::run_context::RunContext;
use crate::util::BoxFuture;

/// The output of a guardrail function.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GuardrailFunctionOutput {
    /// Optional information about the check (stored in results and traces).
    #[serde(default)]
    pub output_info: Value,
    /// Whether the tripwire was triggered. If true the run is halted.
    pub tripwire_triggered: bool,
}

impl GuardrailFunctionOutput {
    /// A passing result.
    pub fn pass() -> Self {
        GuardrailFunctionOutput {
            output_info: Value::Null,
            tripwire_triggered: false,
        }
    }

    /// A result that halts the run.
    pub fn tripwire() -> Self {
        GuardrailFunctionOutput {
            output_info: Value::Null,
            tripwire_triggered: true,
        }
    }

    /// Build from a flag.
    pub fn new(output_info: Value, tripwire_triggered: bool) -> Self {
        GuardrailFunctionOutput {
            output_info,
            tripwire_triggered,
        }
    }
}

/// Result of running an input guardrail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputGuardrailResult {
    pub guardrail_name: String,
    pub output: GuardrailFunctionOutput,
}

/// Result of running an output guardrail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputGuardrailResult {
    pub guardrail_name: String,
    /// Name of the agent whose output was checked.
    pub agent_name: String,
    /// The checked output.
    pub agent_output: Value,
    pub output: GuardrailFunctionOutput,
}

type InputGuardrailFn<C> = Arc<
    dyn Fn(
            RunContext<C>,
            Arc<Agent<C>>,
            Vec<InputItem>,
        ) -> BoxFuture<'static, Result<GuardrailFunctionOutput>>
        + Send
        + Sync,
>;

/// A guardrail that checks the initial input of a run.
pub struct InputGuardrail<C> {
    pub name: String,
    /// Run concurrently with the agent (default) or before it starts.
    pub run_in_parallel: bool,
    func: InputGuardrailFn<C>,
}

impl<C> Clone for InputGuardrail<C> {
    fn clone(&self) -> Self {
        InputGuardrail {
            name: self.name.clone(),
            run_in_parallel: self.run_in_parallel,
            func: self.func.clone(),
        }
    }
}

impl<C> fmt::Debug for InputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputGuardrail")
            .field("name", &self.name)
            .field("run_in_parallel", &self.run_in_parallel)
            .finish()
    }
}

impl<C: Send + Sync + 'static> InputGuardrail<C> {
    /// Create a guardrail from an async closure `(context, agent, input) -> output`.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<Agent<C>>, Vec<InputItem>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<GuardrailFunctionOutput>> + Send + 'static,
    {
        InputGuardrail {
            name: name.into(),
            run_in_parallel: true,
            func: Arc::new(move |c, a, i| Box::pin(f(c, a, i))),
        }
    }

    /// Run before the agent starts instead of in parallel.
    pub fn blocking(mut self) -> Self {
        self.run_in_parallel = false;
        self
    }

    /// Run the guardrail.
    pub async fn run(
        &self,
        ctx: RunContext<C>,
        agent: Arc<Agent<C>>,
        input: Vec<InputItem>,
    ) -> Result<InputGuardrailResult> {
        let output = (self.func)(ctx, agent, input).await?;
        Ok(InputGuardrailResult {
            guardrail_name: self.name.clone(),
            output,
        })
    }
}

type OutputGuardrailFn<C> = Arc<
    dyn Fn(
            RunContext<C>,
            Arc<Agent<C>>,
            Value,
        ) -> BoxFuture<'static, Result<GuardrailFunctionOutput>>
        + Send
        + Sync,
>;

/// A guardrail that checks the final output of the last agent.
pub struct OutputGuardrail<C> {
    pub name: String,
    func: OutputGuardrailFn<C>,
}

impl<C> Clone for OutputGuardrail<C> {
    fn clone(&self) -> Self {
        OutputGuardrail {
            name: self.name.clone(),
            func: self.func.clone(),
        }
    }
}

impl<C> fmt::Debug for OutputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputGuardrail")
            .field("name", &self.name)
            .finish()
    }
}

impl<C: Send + Sync + 'static> OutputGuardrail<C> {
    /// Create a guardrail from an async closure `(context, agent, final_output) -> output`.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(RunContext<C>, Arc<Agent<C>>, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<GuardrailFunctionOutput>> + Send + 'static,
    {
        OutputGuardrail {
            name: name.into(),
            func: Arc::new(move |c, a, o| Box::pin(f(c, a, o))),
        }
    }

    /// Run the guardrail.
    pub async fn run(
        &self,
        ctx: RunContext<C>,
        agent: Arc<Agent<C>>,
        agent_output: Value,
    ) -> Result<OutputGuardrailResult> {
        let output = (self.func)(ctx, agent.clone(), agent_output.clone()).await?;
        Ok(OutputGuardrailResult {
            guardrail_name: self.name.clone(),
            agent_name: agent.name.clone(),
            agent_output,
            output,
        })
    }
}
