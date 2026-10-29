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

    fn client(&self) -> OpenAIClient {
        self.client
            .clone()
            .or_else(default_openai_client)
            .unwrap_or_else(OpenAIClient::from_env)
    }

    fn use_responses(&self) -> bool {
        self.use_responses
            .unwrap_or_else(|| default_openai_api() == OpenAIApi::Responses)
    }
}

impl ModelProvider for OpenAIProvider {
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
                    Arc::new(
                        super::openai_responses_ws::OpenAIResponsesWSModel::new(name, client)
                            .with_explicit(explicit),
                    )
                })
                .clone();
            return Ok(model);
        }
        if self.use_responses() {
            Ok(Arc::new(
                OpenAIResponsesModel::new(name, client).with_explicit(explicit),
            ))
        } else {
            Ok(Arc::new(OpenAIChatCompletionsModel::new(name, client)))
        }
    }
}
