//! Error types raised by the SDK. Mirrors `agents.exceptions` in the Python SDK.

use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputItem, ModelResponse, RunItem};
use crate::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrailResult, ToolOutputGuardrailResult,
};
use crate::usage::Usage;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AgentsError>;

/// Snapshot of the run captured when the runner raises an error.
#[derive(Debug, Clone, Default)]
pub struct RunErrorDetails {
    /// The original input of the run.
    pub input: Vec<InputItem>,
    /// Items generated before the failure.
    pub new_items: Vec<RunItem>,
    /// Raw model responses received before the failure.
    pub raw_responses: Vec<ModelResponse>,
    /// Name of the agent that was active when the run failed.
    pub last_agent: String,
    /// Usage accumulated before the failure.
    pub usage: Usage,
    /// Input guardrail results completed before the failure.
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Output guardrail results completed before the failure.
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    /// Tool input guardrail results completed before the failure.
    pub tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    /// Tool output guardrail results completed before the failure.
    pub tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
}

/// All errors raised by the SDK.
#[derive(Debug, thiserror::Error)]
pub enum AgentsError {
    /// The run exceeded `max_turns`.
    #[error("{message}")]
    MaxTurnsExceeded {
        message: String,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// The model produced malformed or unexpected output (bad JSON, unknown tool, ...).
    #[error("{message}")]
    ModelBehaviorError {
        message: String,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// The model refused to answer while a structured output type was requested.
    #[error("Model refused to produce the requested output: {refusal}")]
    ModelRefusalError {
        refusal: String,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// A model call attempt exceeded `ModelSettings::timeout`.
    #[error("Model call timed out after {timeout_seconds} seconds")]
    ModelTimeoutError {
        timeout_seconds: f64,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// The SDK was used incorrectly (invalid configuration, misuse of the API, ...).
    #[error("{message}")]
    UserError {
        message: String,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// A function tool exceeded its timeout and was configured to raise.
    #[error("Tool '{tool_name}' timed out after {timeout_seconds} seconds")]
    ToolTimeoutError {
        tool_name: String,
        timeout_seconds: f64,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// An input guardrail tripwire was triggered.
    #[error("Guardrail {} triggered tripwire", guardrail_result.guardrail_name)]
    InputGuardrailTripwireTriggered {
        guardrail_result: Box<InputGuardrailResult>,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// An output guardrail tripwire was triggered.
    #[error("Guardrail {} triggered tripwire", guardrail_result.guardrail_name)]
    OutputGuardrailTripwireTriggered {
        guardrail_result: Box<OutputGuardrailResult>,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// A tool input guardrail tripwire was triggered.
    #[error("Tool input guardrail {guardrail_name} triggered tripwire")]
    ToolInputGuardrailTripwireTriggered {
        guardrail_name: String,
        output: Box<ToolGuardrailFunctionOutput>,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// A tool output guardrail tripwire was triggered.
    #[error("Tool output guardrail {guardrail_name} triggered tripwire")]
    ToolOutputGuardrailTripwireTriggered {
        guardrail_name: String,
        output: Box<ToolGuardrailFunctionOutput>,
        run_data: Option<Box<RunErrorDetails>>,
    },

    /// The OpenAI (or compatible) API returned a non-success status.
    #[error("API error (status {status}): {message}")]
    ApiError {
        status: u16,
        message: String,
        body: Option<Box<serde_json::Value>>,
        request_id: Option<String>,
        /// Selected response headers (lowercase names), e.g. `retry-after`, `retry-after-ms`
        /// and `x-should-retry`.
        headers: Option<std::collections::BTreeMap<String, String>>,
    },

    /// Transport-level HTTP failure.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON (de)serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// An I/O failure (sessions, MCP processes, ...).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A failure in a user-provided tool, hook, guardrail or callback.
    #[error("{0}")]
    Other(String),
}

impl AgentsError {
    /// Create a [`AgentsError::UserError`].
    pub fn user(message: impl Into<String>) -> Self {
        AgentsError::UserError {
            message: message.into(),
            run_data: None,
        }
    }

    /// Create a [`AgentsError::ModelBehaviorError`].
    pub fn model_behavior(message: impl Into<String>) -> Self {
        AgentsError::ModelBehaviorError {
            message: message.into(),
            run_data: None,
        }
    }

    /// Create a [`AgentsError::Other`].
    pub fn other(message: impl Into<String>) -> Self {
        AgentsError::Other(message.into())
    }

    /// Attach run details to an error raised inside the runner.
    pub(crate) fn with_run_data(mut self, details: RunErrorDetails) -> Self {
        let boxed = Some(Box::new(details));
        match &mut self {
            AgentsError::MaxTurnsExceeded { run_data, .. }
            | AgentsError::ModelBehaviorError { run_data, .. }
            | AgentsError::ModelRefusalError { run_data, .. }
            | AgentsError::ModelTimeoutError { run_data, .. }
            | AgentsError::UserError { run_data, .. }
            | AgentsError::ToolTimeoutError { run_data, .. }
            | AgentsError::InputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::OutputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::ToolInputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::ToolOutputGuardrailTripwireTriggered { run_data, .. }
                if run_data.is_none() =>
            {
                *run_data = boxed;
            }
            _ => {}
        }
        self
    }

    /// The run details attached to this error, if the runner raised it.
    pub fn run_data(&self) -> Option<&RunErrorDetails> {
        match self {
            AgentsError::MaxTurnsExceeded { run_data, .. }
            | AgentsError::ModelBehaviorError { run_data, .. }
            | AgentsError::ModelRefusalError { run_data, .. }
            | AgentsError::ModelTimeoutError { run_data, .. }
            | AgentsError::UserError { run_data, .. }
            | AgentsError::ToolTimeoutError { run_data, .. }
            | AgentsError::InputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::OutputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::ToolInputGuardrailTripwireTriggered { run_data, .. }
            | AgentsError::ToolOutputGuardrailTripwireTriggered { run_data, .. } => {
                run_data.as_deref()
            }
            _ => None,
        }
    }

    /// True for `MaxTurnsExceeded`.
    pub fn is_max_turns_exceeded(&self) -> bool {
        matches!(self, AgentsError::MaxTurnsExceeded { .. })
    }

    /// True for either input or output guardrail tripwire errors.
    pub fn is_guardrail_tripwire(&self) -> bool {
        matches!(
            self,
            AgentsError::InputGuardrailTripwireTriggered { .. }
                | AgentsError::OutputGuardrailTripwireTriggered { .. }
        )
    }
}

impl From<String> for AgentsError {
    fn from(value: String) -> Self {
        AgentsError::Other(value)
    }
}

impl From<&str> for AgentsError {
    fn from(value: &str) -> Self {
        AgentsError::Other(value.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for AgentsError {
    fn from(value: Box<dyn std::error::Error + Send + Sync>) -> Self {
        AgentsError::Other(value.to_string())
    }
}
