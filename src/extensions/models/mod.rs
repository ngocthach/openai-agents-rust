//! Non-OpenAI model providers. Mirrors `agents.extensions.models` (`LitellmModel`,
//! `LitellmProvider`, `AnyLLMModel`, `AnyLLMProvider`).
//!
//! The Python versions wrap the LiteLLM and any-llm Python libraries. In Rust the same
//! `provider/model` naming is resolved against a registry of providers that expose
//! OpenAI-compatible APIs (Anthropic, Gemini, Mistral, Groq, DeepSeek, OpenRouter, xAI, Ollama,
//! ...), and requests go through the crate's Chat Completions (or Responses) models. A LiteLLM
//! proxy works by passing its URL as `base_url`.
//!
//! Providers without an OpenAI-compatible route are reached by configuring the same models
//! differently: a non-default chat route ([`ProviderEndpoint::chat_path`]), a request-body
//! rewrite, default query parameters, and a signing hook
//! ([`crate::models::RequestAuth`]). AWS Bedrock and SageMaker are signed with SigV4, Vertex AI
//! uses a Google OAuth bearer, Azure OpenAI an `api-key` header plus deployments and an
//! `api-version`, and watsonx.ai an IBM IAM token; all of that lives in [`auth`] behind the
//! `provider-auth` cargo feature.
//!
//! Two of any-llm's provider hooks have no counterpart here, by construction:
//! `_normalize_google_tool_result_roles` rewrites `role: "function"` entries in a *Gemini-native*
//! `contents` payload, which this crate never builds — Gemini and Vertex AI are called through
//! their OpenAI-compatible routes, where a tool result is a `role: "tool"` message.
//! `_clone_provider_without_retries` clones the any-llm provider's HTTP client with
//! `max_retries=0`; the client here performs no retries of its own, so
//! [`crate::models::should_disable_provider_managed_retries`] has nothing to switch off and the
//! SDK's retry policy is always the only one running.
//!
//! ```no_run
//! use std::sync::Arc;
//! use openai_agents_sdk::Agent;
//! use openai_agents_sdk::extensions::models::LitellmModel;
//!
//! # fn demo() -> openai_agents_sdk::Result<()> {
//! // Uses ANTHROPIC_API_KEY.
//! let model = LitellmModel::new("anthropic/claude-sonnet-4-5")?;
//! let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(model));
//! # Ok(()) }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;

pub mod anthropic;
#[cfg(feature = "provider-auth")]
pub mod auth;

use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::model_settings::ModelSettings;
use crate::models::interface::{Model, ModelProvider, ModelRequest, ResponseEventStream};
use crate::models::openai_chatcompletions::{
    OpenAIChatCompletionsModel, ShouldReplayReasoningContent,
};
use crate::models::openai_client::OpenAIClient;
use crate::models::openai_responses::OpenAIResponsesModel;
use serde_json::Value;

/// Providers that require every tool result to immediately follow its tool call.
const STRICT_TOOL_ORDERING_MODELS: &[&str] = &["anthropic", "claude", "gemini"];

/// Whether `model` names a provider that rejects tool results without a preceding tool call.
pub fn requires_strict_tool_ordering(model: &str) -> bool {
    let model = model.to_lowercase();
    STRICT_TOOL_ORDERING_MODELS
        .iter()
        .any(|m| model.contains(m))
}

/// Reorder chat messages so every `tool_calls` assistant message is immediately followed by its
/// tool result. Port of `LitellmModel._fix_tool_message_ordering`.
///
/// An assistant message carrying several tool calls is split into one message per call; only the
/// first split keeps the shared assistant text and reasoning, so signed thinking blocks (which
/// Anthropic rejects when duplicated) appear once.
pub fn fix_tool_message_ordering(messages: Vec<Value>) -> Vec<Value> {
    use std::collections::{HashMap, HashSet};

    // tool call id -> (original index, the single-call assistant message)
    let mut tool_call_messages: HashMap<String, (usize, Value)> = HashMap::new();
    // tool call id -> (original index, the tool result message)
    let mut tool_results: HashMap<String, (usize, Value)> = HashMap::new();
    for (i, message) in messages.iter().enumerate() {
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else {
                    continue;
                };
                for (split, call) in calls.iter().enumerate() {
                    let Some(id) = call.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    let mut single = message.clone();
                    single["tool_calls"] = Value::Array(vec![call.clone()]);
                    if split > 0
                        && let Some(obj) = single.as_object_mut()
                    {
                        for shared in ["content", "thinking_blocks", "reasoning_content"] {
                            obj.remove(shared);
                        }
                    }
                    tool_call_messages.insert(id.to_string(), (i, single));
                }
            }
            Some("tool") => {
                if let Some(id) = message.get("tool_call_id").and_then(Value::as_str) {
                    tool_results.insert(id.to_string(), (i, message.clone()));
                }
            }
            _ => {}
        }
    }

    let paired: HashSet<usize> = tool_call_messages
        .keys()
        .filter_map(|id| tool_results.get(id).map(|(idx, _)| *idx))
        .collect();

    let mut fixed: Vec<Value> = Vec::with_capacity(messages.len());
    for (i, message) in messages.iter().enumerate() {
        match message.get("role").and_then(Value::as_str) {
            Some("assistant")
                if message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|c| !c.is_empty()) =>
            {
                for call in message["tool_calls"].as_array().expect("checked") {
                    let Some(id) = call.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some((_, call_msg)) = tool_call_messages.get(id) else {
                        continue;
                    };
                    fixed.push(call_msg.clone());
                    if let Some((_, result)) = tool_results.get(id) {
                        fixed.push(result.clone());
                    }
                }
            }
            Some("tool") => {
                // Paired results were emitted next to their call above.
                if !paired.contains(&i) {
                    fixed.push(message.clone());
                }
            }
            _ => fixed.push(message.clone()),
        }
    }
    fixed
}

/// Wraps a model whose endpoint has no streaming route: `stream_response` runs the ordinary
/// request and replays the finished response as Responses stream events.
///
/// SageMaker's `/invocations` answers with one JSON body (and, when asked to stream, with AWS's
/// binary event-stream framing, which is not SSE), so a run that streams would otherwise fail
/// mid-parse. any-llm simply refuses to stream there; replaying keeps the run working, at the
/// cost of the text arriving in one piece.
#[derive(Debug)]
struct NonStreamingModel {
    inner: Arc<dyn Model>,
}

#[async_trait]
impl Model for NonStreamingModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let response = self.inner.get_response(request).await?;
        Ok(futures::stream::iter(replay_as_stream_events(&response).into_iter().map(Ok)).boxed())
    }

    fn model_name(&self) -> Option<String> {
        self.inner.model_name()
    }

    fn default_model_settings(&self) -> Option<ModelSettings> {
        self.inner.default_model_settings()
    }

    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        self.inner.get_retry_advice(request)
    }

    async fn close(&self) {
        self.inner.close().await
    }
}

/// Turn a finished [`ModelResponse`] into the Responses stream events a consumer expects.
fn replay_as_stream_events(response: &ModelResponse) -> Vec<Value> {
    let output: Vec<Value> = response.output.iter().map(|o| o.to_value()).collect();
    let usage = &response.usage;
    let payload = serde_json::json!({
        "id": response.response_id.clone()
            .unwrap_or_else(|| crate::models::FAKE_RESPONSES_ID.to_string()),
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": {"cached_tokens": usage.input_tokens_details.cached_tokens},
            "output_tokens_details": {
                "reasoning_tokens": usage.output_tokens_details.reasoning_tokens
            },
        },
    });
    let mut events = vec![serde_json::json!({
        "type": "response.created",
        "response": {"id": payload["id"], "object": "response", "status": "in_progress",
                     "output": []},
    })];
    for (index, item) in output.iter().enumerate() {
        events.push(serde_json::json!({
            "type": "response.output_item.added", "output_index": index, "item": item,
        }));
        if item.get("type").and_then(Value::as_str) == Some("message") {
            for (content_index, part) in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    events.push(serde_json::json!({
                        "type": "response.output_text.delta",
                        "item_id": item.get("id").cloned().unwrap_or(Value::Null),
                        "output_index": index, "content_index": content_index,
                        "delta": text, "logprobs": [],
                    }));
                }
            }
        }
        events.push(serde_json::json!({
            "type": "response.output_item.done", "output_index": index, "item": item,
        }));
    }
    events.push(serde_json::json!({"type": "response.completed", "response": payload}));
    events
}

/// Provider prefixes routed to Anthropic's native Messages API instead of an
/// OpenAI-compatible route. Mirrors any-llm's `anthropic` provider, which speaks the native
/// protocol; the plain `anthropic/` prefix stays on the OpenAI-compatible route.
const ANTHROPIC_NATIVE_PREFIXES: &[&str] = &["anthropic_native", "anthropic_messages"];

/// Build the native Anthropic model for `provider/model`, if the prefix asks for it.
fn native_anthropic(
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<Option<anthropic::AnthropicMessagesModel>> {
    let (provider, provider_model) = split_model_name(model)?;
    if !ANTHROPIC_NATIVE_PREFIXES.contains(&provider.to_ascii_lowercase().as_str()) {
        return Ok(None);
    }
    let key = api_key
        .map(str::to_string)
        .or_else(|| env_first(&["ANTHROPIC_API_KEY"]))
        .unwrap_or_default();
    let native = anthropic::AnthropicMessagesModel::new(provider_model.clone(), key.clone());
    Ok(Some(match base_url {
        None => native,
        Some(url) => anthropic::AnthropicMessagesModel::with_client(
            provider_model,
            native.client().clone().with_base_url(url),
        ),
    }))
}

/// Rewrite the tool calls of Gemini messages into LiteLLM's `provider_specific_fields` shape.
/// Port of `_convert_gemini_extra_content_to_provider_specific_fields`.
///
/// Only tool calls after the last user message are touched: those are the ones being replayed
/// in the current turn, and Gemini validates their thought signatures. A call without one gets
/// the documented skip marker so the validator does not reject the whole request.
pub fn convert_gemini_extra_content_to_provider_specific_fields(
    mut messages: Vec<Value>,
) -> Vec<Value> {
    let last_user = messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"));
    for (i, message) in messages.iter_mut().enumerate() {
        if last_user.is_some_and(|u| i <= u) {
            continue;
        }
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            let Some(call) = call.as_object_mut() else {
                continue;
            };
            let signature = call
                .remove("extra_content")
                .as_ref()
                .and_then(|e| e.pointer("/google/thought_signature"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| "skip_thought_signature_validator".to_string());
            call.insert(
                "provider_specific_fields".into(),
                serde_json::json!({"thought_signature": signature}),
            );
        }
    }
    messages
}

/// Endpoints with no streaming route; their models replay the finished response instead.
const NON_STREAMING_PROVIDERS: &[&str] = &["sagemaker"];

fn wrap_non_streaming(
    endpoint: Option<&'static ProviderEndpoint>,
    model: Arc<dyn Model>,
) -> Arc<dyn Model> {
    match endpoint.filter(|e| NON_STREAMING_PROVIDERS.contains(&e.name)) {
        Some(_) => Arc::new(NonStreamingModel { inner: model }),
        None => model,
    }
}

/// The request-body rewrite a provider needs, if any.
///
/// watsonx.ai is the only endpoint in the table whose body is not plain OpenAI: it names the
/// model `model_id` and requires the owning project or space on every call.
fn provider_body_transform(
    endpoint: Option<&'static ProviderEndpoint>,
    model: &str,
) -> Option<impl Fn(Value) -> Value + Send + Sync + 'static> {
    if endpoint?.name != "watsonx" {
        return None;
    }
    let model = model.to_string();
    let project_id = env_first(&["WATSONX_PROJECT_ID"]);
    let space_id = env_first(&["WATSONX_SPACE_ID"]);
    Some(move |mut body: Value| {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("model");
            obj.insert("model_id".into(), Value::String(model.clone()));
            if let Some(p) = &project_id {
                obj.insert("project_id".into(), Value::String(p.clone()));
            } else if let Some(sp) = &space_id {
                obj.insert("space_id".into(), Value::String(sp.clone()));
            }
        }
        body
    })
}

/// How a provider authenticates requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderAuthKind {
    /// `Authorization: Bearer <api key>`.
    #[default]
    Bearer,
    /// A provider-specific header carrying the key verbatim (Azure OpenAI's `api-key`).
    ApiKeyHeader(&'static str),
    /// AWS Signature Version 4 for the named service.
    AwsSigV4 { service: &'static str },
    /// A Google OAuth bearer token from a service account or Application Default Credentials.
    GoogleOAuth,
    /// An IBM Cloud IAM bearer token exchanged from an API key.
    IbmIam,
}

/// A provider endpoint reachable by the `provider/model` adapters.
///
/// Most entries are OpenAI-compatible endpoints that only need a base URL and an API key.
/// The rest carry what makes them different: a non-default route, a signing scheme, or a
/// base URL that has to be assembled from the account's region, project and deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderEndpoint {
    /// The provider prefix (`anthropic` in `anthropic/claude-...`).
    pub name: &'static str,
    /// Default base URL of the OpenAI-compatible API. May contain `{region}`, `{project}`,
    /// `{endpoint}` and `{model}` placeholders filled from the environment.
    pub base_url: &'static str,
    /// Environment variable overriding the base URL.
    pub base_url_env: Option<&'static str>,
    /// Environment variables holding the API key, in priority order.
    pub api_key_env: &'static [&'static str],
    /// Whether the provider implements the Responses API.
    pub supports_responses: bool,
    /// Route to post chat completions to, when it is not `chat/completions`.
    pub chat_path: Option<&'static str>,
    /// How requests are authenticated.
    pub auth: ProviderAuthKind,
    /// Query parameters appended to every request, as `key=value` pairs whose value may be a
    /// `${VAR}` / `${VAR:default}` reference.
    pub default_query: &'static [&'static str],
    /// Headers sent on every request to this endpoint.
    pub default_headers: &'static [(&'static str, &'static str)],
}

/// A plain OpenAI-compatible endpoint: bearer auth, the standard route, no extra query.
const fn openai_compatible(
    name: &'static str,
    base_url: &'static str,
    base_url_env: Option<&'static str>,
    api_key_env: &'static [&'static str],
    supports_responses: bool,
) -> ProviderEndpoint {
    ProviderEndpoint {
        name,
        base_url,
        base_url_env,
        api_key_env,
        supports_responses,
        chat_path: None,
        auth: ProviderAuthKind::Bearer,
        default_query: &[],
        default_headers: &[],
    }
}

const PROVIDERS: &[ProviderEndpoint] = &[
    openai_compatible(
        "openai",
        "https://api.openai.com/v1",
        Some("OPENAI_BASE_URL"),
        &["OPENAI_API_KEY"],
        true,
    ),
    openai_compatible(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("ANTHROPIC_BASE_URL"),
        &["ANTHROPIC_API_KEY"],
        false,
    ),
    openai_compatible(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        None,
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        false,
    ),
    openai_compatible(
        "mistral",
        "https://api.mistral.ai/v1",
        None,
        &["MISTRAL_API_KEY"],
        false,
    ),
    openai_compatible(
        "groq",
        "https://api.groq.com/openai/v1",
        None,
        &["GROQ_API_KEY"],
        true,
    ),
    openai_compatible(
        "deepseek",
        "https://api.deepseek.com/v1",
        None,
        &["DEEPSEEK_API_KEY"],
        false,
    ),
    openai_compatible(
        "openrouter",
        "https://openrouter.ai/api/v1",
        None,
        &["OPENROUTER_API_KEY"],
        false,
    ),
    openai_compatible(
        "together_ai",
        "https://api.together.xyz/v1",
        None,
        &["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"],
        false,
    ),
    openai_compatible(
        "fireworks_ai",
        "https://api.fireworks.ai/inference/v1",
        None,
        &["FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"],
        true,
    ),
    openai_compatible("xai", "https://api.x.ai/v1", None, &["XAI_API_KEY"], true),
    openai_compatible(
        "perplexity",
        "https://api.perplexity.ai",
        None,
        &["PERPLEXITY_API_KEY", "PERPLEXITYAI_API_KEY"],
        false,
    ),
    openai_compatible(
        "cerebras",
        "https://api.cerebras.ai/v1",
        None,
        &["CEREBRAS_API_KEY"],
        false,
    ),
    openai_compatible(
        "moonshot",
        "https://api.moonshot.ai/v1",
        None,
        &["MOONSHOT_API_KEY"],
        false,
    ),
    openai_compatible(
        "ollama",
        "http://localhost:11434/v1",
        Some("OLLAMA_API_BASE"),
        &["OLLAMA_API_KEY"],
        true,
    ),
    openai_compatible(
        "lm_studio",
        "http://localhost:1234/v1",
        Some("LM_STUDIO_API_BASE"),
        &["LM_STUDIO_API_KEY"],
        true,
    ),
    openai_compatible(
        "litellm_proxy",
        "http://localhost:4000",
        Some("LITELLM_PROXY_API_BASE"),
        &["LITELLM_PROXY_API_KEY"],
        true,
    ),
    // --- Endpoints any-llm reaches that have an OpenAI-compatible route. ---
    openai_compatible(
        "huggingface",
        "https://router.huggingface.co/v1",
        Some("HF_BASE_URL"),
        &["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        false,
    ),
    openai_compatible(
        "databricks",
        "${DATABRICKS_HOST}/serving-endpoints",
        Some("DATABRICKS_BASE_URL"),
        &["DATABRICKS_TOKEN", "DATABRICKS_API_KEY"],
        false,
    ),
    openai_compatible(
        "nebius",
        "https://api.studio.nebius.com/v1",
        None,
        &["NEBIUS_API_KEY"],
        false,
    ),
    openai_compatible(
        "sambanova",
        "https://api.sambanova.ai/v1",
        None,
        &["SAMBANOVA_API_KEY"],
        false,
    ),
    openai_compatible(
        "cohere",
        "https://api.cohere.ai/compatibility/v1",
        None,
        &["COHERE_API_KEY", "CO_API_KEY"],
        false,
    ),
    openai_compatible(
        "llama",
        "https://api.llama.com/compat/v1",
        None,
        &["LLAMA_API_KEY"],
        false,
    ),
    openai_compatible(
        "deepinfra",
        "https://api.deepinfra.com/v1/openai",
        None,
        &["DEEPINFRA_API_KEY"],
        false,
    ),
    openai_compatible(
        "hyperbolic",
        "https://api.hyperbolic.xyz/v1",
        None,
        &["HYPERBOLIC_API_KEY"],
        false,
    ),
    openai_compatible(
        "novita",
        "https://api.novita.ai/v3/openai",
        None,
        &["NOVITA_API_KEY"],
        false,
    ),
    openai_compatible(
        "baseten",
        "https://inference.baseten.co/v1",
        None,
        &["BASETEN_API_KEY"],
        false,
    ),
    openai_compatible(
        "portkey",
        "https://api.portkey.ai/v1",
        Some("PORTKEY_BASE_URL"),
        &["PORTKEY_API_KEY"],
        false,
    ),
    openai_compatible(
        "vercel",
        "https://ai-gateway.vercel.sh/v1",
        None,
        &["AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"],
        false,
    ),
    // --- Endpoints whose auth or shape is not plain OpenAI. ---
    ProviderEndpoint {
        name: "bedrock",
        // Bedrock exposes an OpenAI-compatible route next to the Converse API, and it accepts
        // *both* authentication modes: "The `bedrock-runtime` endpoint supports AWS SigV4
        // authentication and Amazon Bedrock API key authentication", with the published curl
        // examples showing `Authorization: Bearer $OPENAI_API_KEY` and
        // `--aws-sigv4 "aws:amz:<region>:bedrock"` against the same URL. See
        // https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html
        // and https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html.
        //
        // Precedence: an Amazon Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK` wins, and the
        // request is sent as a plain bearer; otherwise it is signed with SigV4 from the usual
        // AWS credentials, under service name `bedrock`. That is the order boto3 itself uses
        // when the variable is set, and the order any-llm's Bedrock provider inherits from it
        // (`ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"`, otherwise the default credential
        // chain). API keys come in both short-term (SigV4-derived, ≤12 h, recommended) and
        // long-term (IAM service-specific credential) forms; both travel as the same header,
        // so this adapter does not distinguish them.
        base_url: "https://bedrock-runtime.${AWS_REGION:us-east-1}.amazonaws.com/openai/v1",
        base_url_env: Some("AWS_BEDROCK_BASE_URL"),
        api_key_env: &["AWS_BEARER_TOKEN_BEDROCK"],
        supports_responses: false,
        chat_path: None,
        auth: ProviderAuthKind::AwsSigV4 { service: "bedrock" },
        default_query: &[],
        default_headers: &[],
    },
    ProviderEndpoint {
        name: "sagemaker",
        // A SageMaker endpoint hosting an OpenAI-compatible container is invoked at
        // `/endpoints/{name}/invocations`; the model name selects the endpoint.
        base_url: "https://runtime.sagemaker.${AWS_REGION:us-east-1}.amazonaws.com/endpoints/{model}",
        base_url_env: Some("AWS_SAGEMAKER_BASE_URL"),
        api_key_env: &[],
        supports_responses: false,
        chat_path: Some("invocations"),
        auth: ProviderAuthKind::AwsSigV4 {
            service: "sagemaker",
        },
        default_query: &[],
        default_headers: &[],
    },
    ProviderEndpoint {
        name: "vertex_ai",
        base_url: "https://${VERTEXAI_LOCATION:us-central1}-aiplatform.googleapis.com/v1/projects/\
                   ${VERTEXAI_PROJECT}/locations/${VERTEXAI_LOCATION:us-central1}/endpoints/\
                   openapi",
        base_url_env: Some("VERTEXAI_BASE_URL"),
        api_key_env: &[],
        supports_responses: false,
        chat_path: None,
        auth: ProviderAuthKind::GoogleOAuth,
        default_query: &[],
        default_headers: &[],
    },
    ProviderEndpoint {
        name: "azure",
        // The deployment is named by the model, as in LiteLLM's `azure/<deployment>`.
        base_url: "${AZURE_OPENAI_ENDPOINT}/openai/deployments/{model}",
        base_url_env: Some("AZURE_OPENAI_BASE_URL"),
        api_key_env: &["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
        supports_responses: false,
        chat_path: None,
        auth: ProviderAuthKind::ApiKeyHeader("api-key"),
        default_query: &["api-version=${AZURE_OPENAI_API_VERSION:2024-10-21}"],
        default_headers: &[],
    },
    ProviderEndpoint {
        name: "azure_ai",
        base_url: "${AZURE_AI_ENDPOINT}/models",
        base_url_env: Some("AZURE_AI_BASE_URL"),
        api_key_env: &["AZURE_AI_API_KEY", "AZURE_API_KEY"],
        supports_responses: false,
        chat_path: None,
        auth: ProviderAuthKind::ApiKeyHeader("api-key"),
        default_query: &["api-version=${AZURE_AI_API_VERSION:2024-05-01-preview}"],
        default_headers: &[],
    },
    ProviderEndpoint {
        name: "watsonx",
        // watsonx.ai has no OpenAI-compatible route; its native chat API is OpenAI-shaped
        // apart from the `model_id`/`project_id` envelope, which a body transform supplies.
        base_url: "https://${WATSONX_REGION:us-south}.ml.cloud.ibm.com",
        base_url_env: Some("WATSONX_URL"),
        api_key_env: &["WATSONX_APIKEY", "WATSONX_API_KEY", "IBM_CLOUD_API_KEY"],
        supports_responses: false,
        chat_path: Some("ml/v1/text/chat"),
        auth: ProviderAuthKind::IbmIam,
        default_query: &["version=${WATSONX_VERSION:2024-10-08}"],
        default_headers: &[],
    },
    ProviderEndpoint {
        // Anthropic's native Messages API, served by `anthropic::AnthropicMessagesModel`.
        // The entry exists so model listing and credential resolution work for the prefix.
        name: "anthropic_native",
        base_url: "https://api.anthropic.com/v1",
        base_url_env: Some("ANTHROPIC_BASE_URL"),
        api_key_env: &["ANTHROPIC_API_KEY"],
        supports_responses: false,
        chat_path: Some("messages"),
        auth: ProviderAuthKind::ApiKeyHeader("x-api-key"),
        default_query: &[],
        default_headers: &[("anthropic-version", anthropic::ANTHROPIC_VERSION)],
    },
];

/// List the models a provider exposes, by prefix (`"groq"`, `"openrouter"`, ...).
///
/// Resolves the provider's base URL and credentials exactly as a model would, then calls the
/// OpenAI-compatible `GET /models`. Providers whose endpoint does not implement that route
/// return the endpoint's error.
pub async fn list_provider_models(
    provider: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<Vec<Value>> {
    // The route does not depend on a model, but resolution does; `{model}` templates are
    // filled with an empty name, which only affects per-deployment endpoints.
    let resolved = resolve(&format!("{provider}/_"), base_url, api_key)?;
    resolved.client.list_models().await
}

/// The providers known to [`LitellmModel`] and [`AnyLLMModel`].
pub fn known_providers() -> &'static [ProviderEndpoint] {
    PROVIDERS
}

/// Look up a provider by prefix (case-insensitive; `together`/`fireworks` aliases accepted).
pub fn provider_endpoint(name: &str) -> Option<&'static ProviderEndpoint> {
    let lower = name.to_ascii_lowercase();
    let canonical = match lower.as_str() {
        "together" | "togetherai" => "together_ai",
        "fireworks" => "fireworks_ai",
        "google" | "gemini_openai" => "gemini",
        "lmstudio" => "lm_studio",
        "grok" | "x_ai" => "xai",
        "vertexai" | "vertex" | "vertex_ai_beta" => "vertex_ai",
        "aws" | "amazon" | "bedrock_converse" => "bedrock",
        "aws_sagemaker" | "sagemaker_chat" => "sagemaker",
        "azure_openai" => "azure",
        "watsonx_text" | "ibm" => "watsonx",
        "anthropic_messages" => "anthropic_native",
        "huggingface_hub" | "hf" => "huggingface",
        other => other,
    };
    PROVIDERS.iter().find(|p| p.name == canonical)
}

/// Split `provider/model` (a bare name means `openai`).
pub fn split_model_name(model: &str) -> Result<(String, String)> {
    if model.is_empty() {
        return Err(AgentsError::user("Model name must not be empty."));
    }
    match model.split_once('/') {
        None => Ok(("openai".into(), model.to_string())),
        Some((p, m)) if !p.is_empty() && !m.is_empty() => Ok((p.to_string(), m.to_string())),
        _ => Err(AgentsError::user(
            "Expected model names in the form 'provider/model', for example \
             'openrouter/openai/gpt-5.4-mini'.",
        )),
    }
}

fn env_first(vars: &[&str]) -> Option<String> {
    vars.iter()
        .filter_map(|v| std::env::var(v).ok())
        .find(|v| !v.is_empty())
}

/// Which wire API an adapter uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterApi {
    /// The Responses API (only for providers that implement it).
    Responses,
    /// The Chat Completions API.
    ChatCompletions,
}

struct Resolved {
    client: OpenAIClient,
    /// A second handle on the same client, for callers that need it after the model took it.
    client_handle: OpenAIClient,
    model: String,
    endpoint: Option<&'static ProviderEndpoint>,
    chat_path: Option<&'static str>,
}

/// Expand `${VAR}` / `${VAR:default}` and `{model}` in a provider template.
fn expand_template(template: &str, model: &str) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after.find('}').ok_or_else(|| {
            AgentsError::user(format!("Unterminated placeholder in '{template}'."))
        })?;
        let (name, default) = match after[..end].split_once(':') {
            Some((n, d)) => (n, Some(d)),
            None => (&after[..end], None),
        };
        let value = env_first(&[name])
            .or_else(|| default.map(str::to_string))
            .ok_or_else(|| {
                AgentsError::user(format!(
                    "Environment variable {name} must be set to use this provider."
                ))
            })?;
        out.push_str(value.trim_end_matches('/'));
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out.replace("{model}", model))
}

/// Install the provider's authentication scheme on the client.
#[cfg_attr(not(feature = "provider-auth"), allow(unused_variables))]
fn apply_auth(
    client: OpenAIClient,
    endpoint: &ProviderEndpoint,
    key: &str,
) -> Result<OpenAIClient> {
    match endpoint.auth {
        ProviderAuthKind::Bearer => Ok(client),
        ProviderAuthKind::ApiKeyHeader(header) => {
            // An Entra ID token is a bearer; a deployment key goes in the provider's header.
            if endpoint.name.starts_with("azure")
                && let Some(token) = env_first(&["AZURE_OPENAI_AD_TOKEN", "AZURE_AD_TOKEN"])
            {
                return Ok(client.with_api_key(token));
            }
            Ok(client.with_bearer_auth(false).with_header(header, key))
        }
        #[cfg(feature = "provider-auth")]
        ProviderAuthKind::AwsSigV4 { service } => {
            // Bedrock accepts either mode on the same route, so a configured Amazon Bedrock
            // API key is preferred and sent as a plain bearer, matching boto3's behaviour when
            // `AWS_BEARER_TOKEN_BEDROCK` is set. SageMaker has no bearer mode and reaches this
            // branch with an empty key, so it always signs.
            if !key.is_empty() {
                return Ok(client);
            }
            let credentials = auth::AwsCredentials::from_env().ok_or_else(|| {
                AgentsError::user(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set to call this \
                     provider (or set AWS_BEARER_TOKEN_BEDROCK).",
                )
            })?;
            let region = env_first(&["AWS_REGION", "AWS_DEFAULT_REGION"])
                .unwrap_or_else(|| "us-east-1".to_string());
            Ok(client
                .with_bearer_auth(false)
                .with_request_auth(Arc::new(auth::AwsSigV4Auth::new(
                    credentials,
                    region,
                    service,
                ))))
        }
        #[cfg(feature = "provider-auth")]
        ProviderAuthKind::GoogleOAuth => {
            if !key.is_empty() {
                return Ok(client);
            }
            Ok(client
                .with_bearer_auth(false)
                .with_request_auth(Arc::new(auth::GoogleOAuthAuth::from_env()?)))
        }
        #[cfg(feature = "provider-auth")]
        ProviderAuthKind::IbmIam => {
            if key.is_empty() {
                return Err(AgentsError::user(
                    "WATSONX_APIKEY must be set to call watsonx.ai.",
                ));
            }
            Ok(client
                .with_bearer_auth(false)
                .with_request_auth(Arc::new(auth::IbmIamAuth::new(key))))
        }
        #[cfg(not(feature = "provider-auth"))]
        ProviderAuthKind::AwsSigV4 { .. }
        | ProviderAuthKind::GoogleOAuth
        | ProviderAuthKind::IbmIam => Err(AgentsError::user(format!(
            "Provider '{}' needs request signing; enable the `provider-auth` cargo feature.",
            endpoint.name
        ))),
    }
}

fn resolve(model: &str, base_url: Option<&str>, api_key: Option<&str>) -> Result<Resolved> {
    let (provider, provider_model) = split_model_name(model)?;
    let endpoint = provider_endpoint(&provider);
    // An explicit base URL (e.g. a LiteLLM proxy) receives the full model name, like LiteLLM.
    let (url, wire_model) = match (base_url, endpoint) {
        (Some(u), _) => (u.to_string(), model.to_string()),
        (None, Some(e)) => {
            let template = e
                .base_url_env
                .and_then(|v| env_first(&[v]))
                .unwrap_or_else(|| e.base_url.to_string());
            (expand_template(&template, &provider_model)?, provider_model)
        }
        (None, None) => {
            return Err(AgentsError::user(format!(
                "Unknown model provider '{provider}'. Pass base_url for an OpenAI-compatible \
                 endpoint, or use one of: {}.",
                PROVIDERS
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    };
    let key = api_key
        .map(str::to_string)
        .or_else(|| endpoint.and_then(|e| env_first(e.api_key_env)))
        .unwrap_or_default();
    let mut client = if endpoint.is_some_and(|e| e.name == "openai") && base_url.is_none() {
        let c = OpenAIClient::from_env();
        if key.is_empty() {
            c
        } else {
            c.with_api_key(key.clone())
        }
    } else {
        OpenAIClient::new(key.clone()).with_base_url(url)
    };
    let mut chat_path = None;
    if let Some(e) = endpoint {
        for pair in e.default_query {
            let (name, template) = pair.split_once('=').ok_or_else(|| {
                AgentsError::user(format!("Malformed provider query parameter '{pair}'."))
            })?;
            client = client.with_default_query(name, expand_template(template, &wire_model)?);
        }
        if base_url.is_none() {
            for (name, value) in e.default_headers {
                client = client.with_header(*name, *value);
            }
            client = apply_auth(client, e, &key)?;
            chat_path = e.chat_path;
        }
    }
    Ok(Resolved {
        client_handle: client.clone(),
        client,
        model: wire_model,
        endpoint,
        chat_path,
    })
}

/// Map OpenAI model ids carried by routing prefixes to their default settings.
fn default_settings_for(model: &str) -> ModelSettings {
    let name = model
        .strip_prefix("openrouter/openai/")
        .or_else(|| model.strip_prefix("openai/"))
        .unwrap_or(model);
    crate::models::default_models::get_default_model_settings(Some(name))
}

macro_rules! delegate_model {
    ($ty:ident) => {
        #[async_trait]
        impl Model for $ty {
            async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
                self.inner.get_response(request).await
            }
            async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
                self.inner.stream_response(request).await
            }
            fn model_name(&self) -> Option<String> {
                Some(self.model.clone())
            }
            fn default_model_settings(&self) -> Option<ModelSettings> {
                Some(default_settings_for(&self.model))
            }
            fn get_retry_advice(
                &self,
                request: &crate::retry::ModelRetryAdviceRequest<'_>,
            ) -> Option<crate::retry::ModelRetryAdvice> {
                self.inner.get_retry_advice(request)
            }
            async fn close(&self) {
                self.inner.close().await
            }
        }
    };
}

/// A model addressed as `provider/model` and called through its OpenAI-compatible Chat
/// Completions API. Port of `LitellmModel`.
#[derive(Clone)]
pub struct LitellmModel {
    model: String,
    client: OpenAIClient,
    inner: Arc<dyn Model>,
}

impl std::fmt::Debug for LitellmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LitellmModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl LitellmModel {
    /// A model resolved from the provider registry and the provider's API key env var.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::with_options(model, None, None, None)
    }

    /// A model with an explicit base URL (e.g. a LiteLLM proxy), API key and reasoning replay
    /// rule.
    pub fn with_options(
        model: impl Into<String>,
        base_url: Option<&str>,
        api_key: Option<&str>,
        should_replay_reasoning_content: Option<ShouldReplayReasoningContent>,
    ) -> Result<Self> {
        let model = model.into();
        if let Some(native) = native_anthropic(&model, base_url, api_key)? {
            let client = native.client().clone();
            return Ok(LitellmModel {
                model,
                client,
                inner: Arc::new(native),
            });
        }
        let explicit_proxy = base_url.is_some();
        let r = resolve(&model, base_url, api_key)?;
        // Python passes preserve_tool_output_all_content=True because LiteLLM then translates
        // the content parts into each provider's native shape. This adapter posts straight to
        // the provider's OpenAI-compatible endpoint with nothing in between, and those
        // endpoints are not guaranteed to accept array content on a `role: tool` message, so
        // the conservative text-only reduction is kept.
        let mut chat = OpenAIChatCompletionsModel::new(r.model.clone(), r.client);
        if let Some(f) = should_replay_reasoning_content {
            chat = chat.with_should_replay_reasoning_content(move |c| f(c));
        }
        let lower = model.to_lowercase();
        let strict_ordering = requires_strict_tool_ordering(&model);
        // LiteLLM's proxy wants Google's thought signature under `provider_specific_fields`;
        // Gemini's own OpenAI-compatible endpoint wants the `extra_content` shape it emitted.
        let gemini_proxy = lower.contains("gemini")
            && r.endpoint.is_some_and(|e| e.name == "litellm_proxy")
            || (lower.contains("gemini") && explicit_proxy);
        if strict_ordering || gemini_proxy {
            chat = chat.with_message_transform(move |messages| {
                let messages = if strict_ordering {
                    fix_tool_message_ordering(messages)
                } else {
                    messages
                };
                if gemini_proxy {
                    convert_gemini_extra_content_to_provider_specific_fields(messages)
                } else {
                    messages
                }
            });
        }
        // Claude's interleaved thinking needs the signed blocks replayed.
        chat = chat.with_preserve_thinking_blocks(true);
        if let Some(path) = r.chat_path {
            chat = chat.with_chat_path(path);
        }
        if let Some(transform) = provider_body_transform(r.endpoint, &r.model) {
            chat = chat.with_body_transform(transform);
        }
        let inner = wrap_non_streaming(r.endpoint, Arc::new(chat));
        Ok(LitellmModel {
            model,
            client: r.client_handle,
            inner,
        })
    }

    /// The models this provider endpoint exposes (`GET /models`).
    pub async fn list_models(&self) -> Result<Vec<Value>> {
        self.client.list_models().await
    }
}

delegate_model!(LitellmModel);

/// Resolves `provider/model` names to [`LitellmModel`]s. Port of `LitellmProvider`.
#[derive(Debug, Clone, Default)]
pub struct LitellmProvider;

impl LitellmProvider {
    /// Create the provider.
    pub fn new() -> Self {
        LitellmProvider
    }
}

impl ModelProvider for LitellmProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name
            .map(str::to_string)
            .unwrap_or_else(crate::models::default_models::get_default_model);
        Ok(Arc::new(LitellmModel::new(name)?))
    }
}

/// A `provider/model` adapter that uses the Responses API where the provider supports it and
/// Chat Completions otherwise. Port of `AnyLLMModel`.
#[derive(Clone)]
pub struct AnyLLMModel {
    model: String,
    api: AdapterApi,
    client: OpenAIClient,
    inner: Arc<dyn Model>,
}

impl std::fmt::Debug for AnyLLMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyLLMModel")
            .field("model", &self.model)
            .field("api", &self.api)
            .finish_non_exhaustive()
    }
}

impl AnyLLMModel {
    /// Create the adapter. `api: None` picks Responses when the provider supports it.
    pub fn new(
        model: impl Into<String>,
        base_url: Option<&str>,
        api_key: Option<&str>,
        api: Option<AdapterApi>,
    ) -> Result<Self> {
        let model = model.into();
        if let Some(native) = native_anthropic(&model, base_url, api_key)? {
            if api == Some(AdapterApi::Responses) {
                return Err(AgentsError::user(format!(
                    "Provider for '{model}' does not support the Responses API; use \
                     chat_completions."
                )));
            }
            let client = native.client().clone();
            return Ok(AnyLLMModel {
                model,
                api: AdapterApi::ChatCompletions,
                client,
                inner: Arc::new(native),
            });
        }
        let r = resolve(&model, base_url, api_key)?;
        let supports_responses = r.endpoint.is_some_and(|e| e.supports_responses);
        let api = match api {
            Some(AdapterApi::Responses) if !supports_responses && base_url.is_none() => {
                return Err(AgentsError::user(format!(
                    "Provider for '{model}' does not support the Responses API; use \
                     chat_completions."
                )));
            }
            Some(a) => a,
            None if supports_responses => AdapterApi::Responses,
            None => AdapterApi::ChatCompletions,
        };
        let inner: Arc<dyn Model> = match api {
            AdapterApi::Responses => Arc::new(OpenAIResponsesModel::new(r.model, r.client)),
            AdapterApi::ChatCompletions => {
                let mut chat = OpenAIChatCompletionsModel::new(r.model.clone(), r.client)
                    .with_preserve_thinking_blocks(true);
                if requires_strict_tool_ordering(&model) {
                    chat = chat.with_message_transform(fix_tool_message_ordering);
                }
                if let Some(path) = r.chat_path {
                    chat = chat.with_chat_path(path);
                }
                if let Some(transform) = provider_body_transform(r.endpoint, &r.model) {
                    chat = chat.with_body_transform(transform);
                }
                wrap_non_streaming(r.endpoint, Arc::new(chat))
            }
        };
        Ok(AnyLLMModel {
            model,
            api,
            client: r.client_handle,
            inner,
        })
    }

    /// The wire API in use.
    pub fn api(&self) -> AdapterApi {
        self.api
    }

    /// The models this provider endpoint exposes (`GET /models`).
    ///
    /// any-llm offers `list_models` per provider; every endpoint in the registry answers the
    /// same OpenAI-compatible route.
    pub async fn list_models(&self) -> Result<Vec<Value>> {
        self.client.list_models().await
    }
}

delegate_model!(AnyLLMModel);

/// Resolves `provider/model` names to [`AnyLLMModel`]s. Port of `AnyLLMProvider`.
#[derive(Debug, Clone, Default)]
pub struct AnyLLMProvider {
    /// API key for every model (otherwise the provider's env var).
    pub api_key: Option<String>,
    /// Base URL for every model.
    pub base_url: Option<String>,
    /// Forced wire API.
    pub api: Option<AdapterApi>,
}

impl ModelProvider for AnyLLMProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.map(str::to_string).unwrap_or_else(|| {
            format!(
                "openai/{}",
                crate::models::default_models::get_default_model()
            )
        });
        Ok(Arc::new(AnyLLMModel::new(
            name,
            self.base_url.as_deref(),
            self.api_key.as_deref(),
            self.api,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_tool_calls_with_their_results() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "thinking",
            "thinking_blocks": [{"signature": "s"}],
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}}
            ]}),
            serde_json::json!({"role": "tool", "tool_call_id": "b", "content": "B"}),
            serde_json::json!({"role": "tool", "tool_call_id": "a", "content": "A"}),
            serde_json::json!({"role": "tool", "tool_call_id": "orphan", "content": "O"}),
        ];
        let fixed = fix_tool_message_ordering(messages);
        let shape: Vec<String> = fixed
            .iter()
            .map(|m| match m["role"].as_str().unwrap() {
                "tool" => format!("tool:{}", m["tool_call_id"].as_str().unwrap()),
                "assistant" => format!("call:{}", m["tool_calls"][0]["id"].as_str().unwrap()),
                other => other.to_string(),
            })
            .collect();
        assert_eq!(
            shape,
            [
                "user",
                "call:a",
                "tool:a",
                "call:b",
                "tool:b",
                "tool:orphan"
            ]
        );
        // Only the first split keeps the shared assistant text and thinking blocks.
        assert_eq!(fixed[1]["content"], "thinking");
        assert!(fixed[3].get("content").is_none());
        assert!(fixed[3].get("thinking_blocks").is_none());
    }

    #[test]
    fn strict_tool_ordering_matches_anthropic_and_gemini() {
        assert!(requires_strict_tool_ordering("anthropic/claude-sonnet-4-5"));
        assert!(requires_strict_tool_ordering("vertex/GEMINI-2.5-pro"));
        assert!(!requires_strict_tool_ordering("groq/llama-3.3-70b"));
    }

    #[test]
    fn resolves_providers() {
        assert_eq!(
            split_model_name("openrouter/openai/gpt-5").unwrap(),
            ("openrouter".into(), "openai/gpt-5".into())
        );
        assert_eq!(split_model_name("gpt-5").unwrap().0, "openai");
        assert!(split_model_name("/x").is_err());
        assert_eq!(provider_endpoint("Together").unwrap().name, "together_ai");
        let r = resolve("groq/llama", None, Some("k")).unwrap();
        assert_eq!(r.model, "llama");
        assert_eq!(r.client.base_url(), "https://api.groq.com/openai/v1");
        let proxy = resolve("anthropic/claude", Some("http://proxy:4000"), Some("k")).unwrap();
        assert_eq!(proxy.model, "anthropic/claude");
        assert!(resolve("nope/x", None, None).is_err());
        let m = AnyLLMModel::new("deepseek/deepseek-chat", None, Some("k"), None).unwrap();
        assert_eq!(m.api(), AdapterApi::ChatCompletions);
        let m = AnyLLMModel::new("openai/gpt-5", None, Some("k"), None).unwrap();
        assert_eq!(m.api(), AdapterApi::Responses);
        assert!(
            AnyLLMModel::new("mistral/x", None, Some("k"), Some(AdapterApi::Responses)).is_err()
        );
    }
}
