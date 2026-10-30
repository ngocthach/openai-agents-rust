//! Model and model provider interfaces. Mirrors `agents.models.interface`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::Value;

use crate::agent_output::AgentOutputSchemaBase;
use crate::error::Result;
use crate::items::{InputItem, ModelResponse};
use crate::model_settings::ModelSettings;
use crate::prompts::Prompt;

/// Tracing mode passed to models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelTracing {
    /// Tracing is disabled entirely.
    Disabled,
    /// Tracing is enabled, and all data is included.
    #[default]
    Enabled,
    /// Tracing is enabled, but inputs/outputs are not included.
    EnabledWithoutData,
}

impl ModelTracing {
    pub fn is_disabled(self) -> bool {
        self == ModelTracing::Disabled
    }
    pub fn include_data(self) -> bool {
        self == ModelTracing::Enabled
    }
}

/// A function-style tool (function tool or handoff) as seen by the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FunctionToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
    pub defer_loading: bool,
    /// Namespace attached with `tool_namespace` (Responses API only).
    pub namespace: Option<String>,
    /// Description of the namespace.
    pub namespace_description: Option<String>,
    /// Programmatic callers allowed to invoke the tool (Responses API only).
    pub allowed_callers: Option<Vec<String>>,
    /// Strict JSON object schema of the tool's output (Responses API only).
    pub output_json_schema: Option<Value>,
}

impl FunctionToolSpec {
    /// Responses API `function` tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = serde_json::json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        });
        if self.defer_loading {
            v["defer_loading"] = Value::Bool(true);
        }
        if let Some(c) = &self.allowed_callers {
            v["allowed_callers"] = serde_json::json!(c);
        }
        if let Some(schema) = &self.output_json_schema {
            v["output_schema"] = schema.clone();
        }
        v
    }

    /// Chat Completions `function` tool param.
    pub fn to_chat_param(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            }
        })
    }
}

/// A tool as seen by the model, independent of the run context type.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolSpec {
    /// A function tool executed by the SDK.
    Function(FunctionToolSpec),
    /// A hosted tool param (Responses API only) plus the `include` values it needs.
    Hosted { param: Value, includes: Vec<String> },
}

/// Everything a model needs to produce a response.
#[derive(Debug, Clone, Default)]
pub struct ModelRequest {
    /// System instructions (Responses `instructions`, or a system message for Chat Completions).
    pub system_instructions: Option<String>,
    /// Input items in the Responses API format.
    pub input: Vec<InputItem>,
    /// Resolved model settings.
    pub model_settings: ModelSettings,
    /// Tools available to the model.
    pub tools: Vec<ToolSpec>,
    /// Structured output schema, if the agent has an output type.
    pub output_schema: Option<Arc<dyn AgentOutputSchemaBase>>,
    /// Handoffs, exposed as function tools.
    pub handoffs: Vec<FunctionToolSpec>,
    /// Tracing mode.
    pub tracing: ModelTracing,
    /// Responses API `previous_response_id`.
    pub previous_response_id: Option<String>,
    /// Responses API `conversation`.
    pub conversation_id: Option<String>,
    /// Responses API prompt template.
    pub prompt: Option<Prompt>,
}

/// A stream of raw Responses API stream events (`{"type": "response.output_text.delta", ...}`).
///
/// The stream must end with a `response.completed` event carrying the full response.
pub type ResponseEventStream = BoxStream<'static, Result<Value>>;

/// The base interface for calling an LLM.
#[async_trait]
pub trait Model: Send + Sync + fmt::Debug {
    /// Get a full response from the model.
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse>;

    /// Stream a response from the model as Responses API events.
    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream>;

    /// The model name, if known (used for default settings and tracing).
    fn model_name(&self) -> Option<String> {
        None
    }

    /// Default settings for this model, or `None` to keep the agent's settings.
    fn default_model_settings(&self) -> Option<ModelSettings> {
        self.model_name()
            .map(|n| crate::models::default_models::get_default_model_settings(Some(&n)))
    }

    /// Provider-specific retry guidance for a failed call, used by retry policies.
    fn get_retry_advice(
        &self,
        _request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        None
    }

    /// Whether the SDK may add a generated `prompt_cache_key` for this model.
    ///
    /// Only the official OpenAI endpoints understand the field, so a compatible third-party
    /// endpoint keeps the default `false`.
    /// Mirrors `_supports_default_prompt_cache_key`.
    fn supports_default_prompt_cache_key(&self) -> bool {
        false
    }

    /// Whether this model talks to the Chat Completions API.
    ///
    /// Chat Completions owns its released warn-and-ignore behaviour for an output guardrail
    /// that trips on a server-managed conversation, so the runner does not reject that
    /// combination for it. Mirrors the `isinstance(..., OpenAIChatCompletionsModel)` test in
    /// `validate_output_guardrails_with_server_managed_conversation`.
    fn is_chat_completions(&self) -> bool {
        false
    }

    /// Release run-scoped resources after the runner finished using this model for `owner`.
    ///
    /// Mirrors `Model._cleanup_on_run_end`. `owner` is the id installed by
    /// [`model_run_context`](crate::models::run_context::model_run_context) for the run that is
    /// ending; a model holding per-run state must release it only when the state belongs to
    /// that run. The default does nothing.
    async fn cleanup_on_run_end(&self, _owner: crate::models::run_context::ModelRunOwner) {}

    /// Release resources held by the model.
    async fn close(&self) {}
}

/// Looks up models by name.
#[async_trait]
pub trait ModelProvider: Send + Sync + fmt::Debug {
    /// Get a model by name (`None` selects the provider default).
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>>;

    /// The OpenAI agent harness id configured on this provider, if any.
    fn agent_harness_id(&self) -> Option<String> {
        None
    }

    /// Release resources held by the provider and the models it handed out.
    ///
    /// Mirrors `ModelProvider.aclose()`; the default does nothing.
    async fn close(&self) {}
}
