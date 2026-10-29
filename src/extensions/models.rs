//! Non-OpenAI model providers. Mirrors `agents.extensions.models` (`LitellmModel`,
//! `LitellmProvider`, `AnyLLMModel`, `AnyLLMProvider`).
//!
//! The Python versions wrap the LiteLLM and any-llm Python libraries. In Rust the same
//! `provider/model` naming is resolved against a registry of providers that expose
//! OpenAI-compatible APIs (Anthropic, Gemini, Mistral, Groq, DeepSeek, OpenRouter, xAI, Ollama,
//! ...), and requests go through the crate's Chat Completions (or Responses) models. A LiteLLM
//! proxy works by passing its URL as `base_url`.
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

use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::model_settings::ModelSettings;
use crate::models::interface::{Model, ModelProvider, ModelRequest, ResponseEventStream};
use crate::models::openai_chatcompletions::{
    OpenAIChatCompletionsModel, ShouldReplayReasoningContent,
};
use crate::models::openai_client::OpenAIClient;
use crate::models::openai_responses::OpenAIResponsesModel;

/// An OpenAI-compatible provider endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderEndpoint {
    /// The provider prefix (`anthropic` in `anthropic/claude-...`).
    pub name: &'static str,
    /// Default base URL of the OpenAI-compatible API.
    pub base_url: &'static str,
    /// Environment variable overriding the base URL.
    pub base_url_env: Option<&'static str>,
    /// Environment variables holding the API key, in priority order.
    pub api_key_env: &'static [&'static str],
    /// Whether the provider implements the Responses API.
    pub supports_responses: bool,
}

const PROVIDERS: &[ProviderEndpoint] = &[
    ProviderEndpoint {
        name: "openai",
        base_url: "https://api.openai.com/v1",
        base_url_env: Some("OPENAI_BASE_URL"),
        api_key_env: &["OPENAI_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "anthropic",
        base_url: "https://api.anthropic.com/v1",
        base_url_env: Some("ANTHROPIC_BASE_URL"),
        api_key_env: &["ANTHROPIC_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "gemini",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        base_url_env: None,
        api_key_env: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "mistral",
        base_url: "https://api.mistral.ai/v1",
        base_url_env: None,
        api_key_env: &["MISTRAL_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
        base_url_env: None,
        api_key_env: &["GROQ_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
        base_url_env: None,
        api_key_env: &["DEEPSEEK_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        base_url_env: None,
        api_key_env: &["OPENROUTER_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "together_ai",
        base_url: "https://api.together.xyz/v1",
        base_url_env: None,
        api_key_env: &["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "fireworks_ai",
        base_url: "https://api.fireworks.ai/inference/v1",
        base_url_env: None,
        api_key_env: &["FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "xai",
        base_url: "https://api.x.ai/v1",
        base_url_env: None,
        api_key_env: &["XAI_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "perplexity",
        base_url: "https://api.perplexity.ai",
        base_url_env: None,
        api_key_env: &["PERPLEXITY_API_KEY", "PERPLEXITYAI_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "cerebras",
        base_url: "https://api.cerebras.ai/v1",
        base_url_env: None,
        api_key_env: &["CEREBRAS_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "moonshot",
        base_url: "https://api.moonshot.ai/v1",
        base_url_env: None,
        api_key_env: &["MOONSHOT_API_KEY"],
        supports_responses: false,
    },
    ProviderEndpoint {
        name: "ollama",
        base_url: "http://localhost:11434/v1",
        base_url_env: Some("OLLAMA_API_BASE"),
        api_key_env: &["OLLAMA_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "lm_studio",
        base_url: "http://localhost:1234/v1",
        base_url_env: Some("LM_STUDIO_API_BASE"),
        api_key_env: &["LM_STUDIO_API_KEY"],
        supports_responses: true,
    },
    ProviderEndpoint {
        name: "litellm_proxy",
        base_url: "http://localhost:4000",
        base_url_env: Some("LITELLM_PROXY_API_BASE"),
        api_key_env: &["LITELLM_PROXY_API_KEY"],
        supports_responses: true,
    },
];

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
    model: String,
    endpoint: Option<&'static ProviderEndpoint>,
}

fn resolve(model: &str, base_url: Option<&str>, api_key: Option<&str>) -> Result<Resolved> {
    let (provider, provider_model) = split_model_name(model)?;
    let endpoint = provider_endpoint(&provider);
    // An explicit base URL (e.g. a LiteLLM proxy) receives the full model name, like LiteLLM.
    let (url, wire_model) = match (base_url, endpoint) {
        (Some(u), _) => (u.to_string(), model.to_string()),
        (None, Some(e)) => (
            e.base_url_env
                .and_then(|v| env_first(&[v]))
                .unwrap_or_else(|| e.base_url.to_string()),
            provider_model,
        ),
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
    let client = if endpoint.is_some_and(|e| e.name == "openai") && base_url.is_none() {
        let c = OpenAIClient::from_env();
        if key.is_empty() {
            c
        } else {
            c.with_api_key(key)
        }
    } else {
        OpenAIClient::new(key).with_base_url(url)
    };
    Ok(Resolved {
        client,
        model: wire_model,
        endpoint,
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
        let r = resolve(&model, base_url, api_key)?;
        let mut chat = OpenAIChatCompletionsModel::new(r.model, r.client);
        if let Some(f) = should_replay_reasoning_content {
            chat = chat.with_should_replay_reasoning_content(move |c| f(c));
        }
        Ok(LitellmModel {
            model,
            inner: Arc::new(chat),
        })
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
                Arc::new(OpenAIChatCompletionsModel::new(r.model, r.client))
            }
        };
        Ok(AnyLLMModel { model, api, inner })
    }

    /// The wire API in use.
    pub fn api(&self) -> AdapterApi {
        self.api
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
