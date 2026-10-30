//! Model interfaces and the OpenAI implementations. Mirrors `agents.models`.

pub mod default_models;
pub mod interface;
pub mod multi_provider;
pub mod openai_chatcompletions;
pub mod openai_client;
pub mod openai_provider;
pub mod openai_responses;
#[cfg(feature = "websocket")]
pub mod openai_responses_ws;
pub mod response_terminal;
pub mod trace;

use std::sync::RwLock;

pub use interface::{
    FunctionToolSpec, Model, ModelProvider, ModelRequest, ModelTracing, ResponseEventStream,
    ToolSpec,
};
pub use multi_provider::{MultiProvider, MultiProviderMap};
pub use openai_chatcompletions::OpenAIChatCompletionsModel;
pub use openai_client::{OpenAIClient, is_official_openai_base_url};
pub use openai_provider::OpenAIProvider;
pub use openai_responses::OpenAIResponsesModel;
#[cfg(feature = "websocket")]
pub use openai_responses_ws::OpenAIResponsesWSModel;
pub use response_terminal::{format_response_error_event, format_response_terminal_failure};
pub use trace::{model_config_for_trace, sanitize_url_for_trace};

/// Placeholder id used for items synthesized from Chat Completions responses.
pub const FAKE_RESPONSES_ID: &str = "__fake_id__";

/// Which OpenAI API shape the default provider uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAIApi {
    /// The Responses API (default).
    #[default]
    Responses,
    /// The Chat Completions API.
    ChatCompletions,
}

static DEFAULT_KEY: RwLock<Option<String>> = RwLock::new(None);
static DEFAULT_CLIENT: RwLock<Option<OpenAIClient>> = RwLock::new(None);
static DEFAULT_API: RwLock<OpenAIApi> = RwLock::new(OpenAIApi::Responses);
static DEFAULT_USE_FOR_TRACING: RwLock<bool> = RwLock::new(true);

static DEFAULT_HARNESS: RwLock<Option<String>> = RwLock::new(None);

/// Set the default OpenAI agent harness id recorded as `agent_harness_id` in trace metadata
/// (falls back to the `OPENAI_AGENT_HARNESS_ID` environment variable).
pub fn set_default_openai_harness(harness_id: Option<String>) {
    *DEFAULT_HARNESS.write().expect("poisoned") = harness_id;
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Resolve the agent harness id: the provider's, then the global default, then the env var.
pub(crate) fn resolve_openai_harness_id(provider: &dyn interface::ModelProvider) -> Option<String> {
    non_empty(provider.agent_harness_id())
        .or_else(|| non_empty(DEFAULT_HARNESS.read().expect("poisoned").clone()))
        .or_else(|| non_empty(std::env::var("OPENAI_AGENT_HARNESS_ID").ok()))
}

/// Set the default OpenAI API key used for model requests (and tracing, unless overridden).
pub fn set_default_openai_key(key: impl Into<String>) {
    *DEFAULT_KEY.write().expect("poisoned") = Some(key.into());
}

/// Set the default OpenAI client used by the default provider. When `use_for_tracing` is true
/// its API key is also used to export traces.
pub fn set_default_openai_client(client: OpenAIClient, use_for_tracing: bool) {
    *DEFAULT_CLIENT.write().expect("poisoned") = Some(client);
    *DEFAULT_USE_FOR_TRACING.write().expect("poisoned") = use_for_tracing;
}

/// Choose the API shape (`Responses` or `ChatCompletions`) for the default provider.
pub fn set_default_openai_api(api: OpenAIApi) {
    *DEFAULT_API.write().expect("poisoned") = api;
}

pub(crate) fn default_openai_key() -> Option<String> {
    if let Some(k) = DEFAULT_KEY.read().expect("poisoned").clone() {
        return Some(k);
    }
    if *DEFAULT_USE_FOR_TRACING.read().expect("poisoned")
        && let Some(c) = DEFAULT_CLIENT.read().expect("poisoned").as_ref()
        && let Some(k) = c.explicit_api_key()
    {
        return Some(k);
    }
    None
}

pub(crate) fn default_openai_client() -> Option<OpenAIClient> {
    DEFAULT_CLIENT.read().expect("poisoned").clone()
}

pub(crate) fn default_openai_api() -> OpenAIApi {
    *DEFAULT_API.read().expect("poisoned")
}
