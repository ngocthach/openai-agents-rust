//! Provider resolving model names to OpenAI models. Mirrors `agents.models.openai_provider`.

use std::sync::Arc;

use super::default_models::get_default_model;
use super::interface::{Model, ModelProvider};
use super::openai_chatcompletions::OpenAIChatCompletionsModel;
use super::openai_client::OpenAIClient;
use super::openai_responses::OpenAIResponsesModel;
use super::{OpenAIApi, default_openai_api, default_openai_client};
use crate::error::Result;

#[cfg(feature = "websocket")]
type WebsocketModelCache =
    Arc<std::sync::Mutex<std::collections::HashMap<(String, bool), Arc<dyn Model>>>>;

/// Resolves model names to OpenAI Responses (default) or Chat Completions models.
#[derive(Debug, Clone, Default)]
pub struct OpenAIProvider {
    client: Option<OpenAIClient>,
    use_responses: Option<bool>,
    agent_harness_id: Option<String>,
    strict_feature_validation: bool,
    #[cfg(feature = "websocket")]
    websocket_base_url: Option<String>,
    #[cfg(feature = "websocket")]
    use_responses_websocket: bool,
    #[cfg(feature = "websocket")]
    websocket_models: WebsocketModelCache,
}

impl OpenAIProvider {
    /// A provider using the default client and API shape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a specific client.
    pub fn with_client(mut self, client: OpenAIClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Choose Responses (`true`) or Chat Completions (`false`).
    pub fn with_use_responses(mut self, use_responses: bool) -> Self {
        self.use_responses = Some(use_responses);
        self
    }

    /// Use the Responses API over a persistent WebSocket. Models are cached per name so their
    /// connection is reused across runs; call [`Model::close`] on them (or drop the provider)
    /// when done.
    #[cfg(feature = "websocket")]
    pub fn with_responses_websocket(mut self, enabled: bool) -> Self {
        self.use_responses_websocket = enabled;
        self
    }

    /// Override the Responses WebSocket base URL. Without it, the `OPENAI_WEBSOCKET_BASE_URL`
    /// environment variable is used, and otherwise the client base URL with a `ws(s)` scheme.
    #[cfg(feature = "websocket")]
    pub fn with_websocket_base_url(mut self, url: impl Into<String>) -> Self {
        self.websocket_base_url = Some(url.into());
        self
    }

    /// Make Chat Completions models reject Responses-only features (`previous_response_id`,
    /// `conversation_id`, `prompt`) instead of warning and ignoring them.
    pub fn with_strict_feature_validation(mut self, strict: bool) -> Self {
        self.strict_feature_validation = strict;
        self
    }

    /// Set the agent harness id recorded in trace metadata for runs using this provider.
    pub fn with_agent_harness_id(mut self, harness_id: impl Into<String>) -> Self {
        self.agent_harness_id = Some(harness_id.into());
        self
    }

    /// Close the cached WebSocket models (and their connections).
    #[cfg(feature = "websocket")]
    pub async fn close_websocket_models(&self) {
        let models: Vec<Arc<dyn Model>> = self
            .websocket_models
            .lock()
            .expect("poisoned")
            .drain()
            .map(|(_, m)| m)
            .collect();
        for m in models {
            m.close().await;
        }
    }

    fn client(&self) -> OpenAIClient {
        self.client
            .clone()
            .or_else(default_openai_client)
            .unwrap_or_else(OpenAIClient::from_env)
    }

    #[cfg(feature = "websocket")]
    fn websocket_base_url(&self) -> Option<String> {
        self.websocket_base_url.clone().or_else(|| {
            std::env::var("OPENAI_WEBSOCKET_BASE_URL")
                .ok()
                .filter(|v| !v.trim().is_empty())
        })
    }

    fn use_responses(&self) -> bool {
        self.use_responses
            .unwrap_or_else(|| default_openai_api() == OpenAIApi::Responses)
    }
}

#[async_trait::async_trait]
impl ModelProvider for OpenAIProvider {
    #[cfg(feature = "websocket")]
    async fn close(&self) {
        self.close_websocket_models().await;
    }

    fn agent_harness_id(&self) -> Option<String> {
        self.agent_harness_id.clone()
    }

    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let explicit = model_name.is_some();
        let name = model_name
            .map(str::to_string)
            .unwrap_or_else(get_default_model);
        let client = self.client();
        #[cfg(feature = "websocket")]
        if self.use_responses_websocket && self.use_responses() {
            let mut cache = self.websocket_models.lock().expect("poisoned");
            let model = cache
                .entry((name.clone(), explicit))
                .or_insert_with(|| {
                    let mut model =
                        super::openai_responses_ws::OpenAIResponsesWSModel::new(name, client)
                            .with_explicit(explicit);
                    if let Some(url) = self.websocket_base_url() {
                        model = model.with_websocket_base_url(url);
                    }
                    Arc::new(model)
                })
                .clone();
            return Ok(model);
        }
        if self.use_responses() {
            Ok(Arc::new(
                OpenAIResponsesModel::new(name, client).with_explicit(explicit),
            ))
        } else {
            Ok(Arc::new(
                OpenAIChatCompletionsModel::new(name, client)
                    .with_strict_feature_validation(self.strict_feature_validation),
            ))
        }
    }
}
