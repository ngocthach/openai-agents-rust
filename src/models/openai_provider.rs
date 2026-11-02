//! Provider resolving model names to OpenAI models. Mirrors `agents.models.openai_provider`.

use std::sync::Arc;

use super::default_models::get_default_model;
use super::interface::{Model, ModelProvider};
use super::openai_chatcompletions::OpenAIChatCompletionsModel;
use super::openai_client::OpenAIClient;
use super::openai_responses::OpenAIResponsesModel;
use super::{OpenAIApi, default_openai_api, default_openai_client};
use crate::error::Result;

/// The error the OpenAI client raises when it is built without an API key.
const MISSING_API_KEY_MESSAGE: &str = "The api_key client option must be set either by passing \
     api_key to the client or by setting the OPENAI_API_KEY environment variable";

#[cfg(feature = "websocket")]
type WebsocketModelCache =
    Arc<std::sync::Mutex<std::collections::HashMap<(String, bool), Arc<dyn Model>>>>;

/// Resolves model names to OpenAI Responses (default) or Chat Completions models.
///
/// [`get_model`](ModelProvider::get_model) fails with a user error when the provider builds its
/// own client for the official OpenAI API and no key is set (explicitly, with
/// [`set_default_openai_key`](super::set_default_openai_key) or `OPENAI_API_KEY`), as Python's
/// client does. A run resolves the model where Python does: after the blocking input guardrails,
/// and racing the parallel ones. A client for another base URL may stay keyless.
#[derive(Debug, Clone, Default)]
pub struct OpenAIProvider {
    client: Option<OpenAIClient>,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    use_responses: Option<bool>,
    agent_harness_id: Option<String>,
    strict_feature_validation: bool,
    buffer_streamed_tool_calls: bool,
    #[cfg(feature = "websocket")]
    websocket_base_url: Option<String>,
    #[cfg(feature = "websocket")]
    use_responses_websocket: Option<bool>,
    #[cfg(feature = "websocket")]
    websocket_options: super::openai_responses_ws::OpenAIResponsesWebSocketOptions,
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

    /// Use this API key for a client the provider builds itself, instead of the default
    /// client. An explicit value, even an empty one, wins over the global key and the
    /// environment. Not combinable with [`with_client`](Self::with_client).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use this base URL for a client the provider builds itself, instead of the default
    /// client and `OPENAI_BASE_URL`. Not combinable with [`with_client`](Self::with_client).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Use this organization for a client the provider builds itself.
    /// Not combinable with [`with_client`](Self::with_client).
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Use this project for a client the provider builds itself.
    /// Not combinable with [`with_client`](Self::with_client).
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
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
        self.use_responses_websocket = Some(enabled);
        self
    }

    /// Low-level WebSocket connection options for the models this provider hands out.
    #[cfg(feature = "websocket")]
    pub fn with_websocket_options(
        mut self,
        options: super::openai_responses_ws::OpenAIResponsesWebSocketOptions,
    ) -> Self {
        self.websocket_options = options;
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

    /// Make Chat Completions models emit each streamed tool call once, complete, instead of
    /// forwarding the provider's argument fragments. Mirrors `buffer_streamed_tool_calls`.
    pub fn with_buffer_streamed_tool_calls(mut self, buffer: bool) -> Self {
        self.buffer_streamed_tool_calls = buffer;
        self
    }

    /// Set the agent harness id recorded in trace metadata for runs using this provider.
    pub fn with_agent_harness_id(mut self, harness_id: impl Into<String>) -> Self {
        self.agent_harness_id = Some(harness_id.into());
        self
    }

    /// Whether Chat Completions models from this provider buffer streamed tool calls.
    pub fn buffer_streamed_tool_calls(&self) -> bool {
        self.buffer_streamed_tool_calls
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

    /// Whether a client option was set on the provider itself.
    fn has_explicit_client_options(&self) -> bool {
        #[cfg(feature = "websocket")]
        let websocket = self.websocket_base_url.is_some();
        #[cfg(not(feature = "websocket"))]
        let websocket = false;
        self.api_key.is_some()
            || self.base_url.is_some()
            || self.organization.is_some()
            || self.project.is_some()
            || websocket
    }

    /// Reject client options next to an explicit client, which would silently ignore them.
    fn validate_client_options(&self) -> Result<()> {
        if self.client.is_some() && self.has_explicit_client_options() {
            return Err(crate::error::AgentsError::user(
                "Don't provide api_key, base_url, websocket_base_url, organization, or project \
                 if you provide openai_client",
            ));
        }
        Ok(())
    }

    /// The client this provider's models use: the explicit client, else the default client
    /// unless a client option was set on the provider, else a client built from the provider's
    /// options with the global key and the environment filling the gaps. Mirrors `_get_client`.
    pub fn openai_client(&self) -> OpenAIClient {
        if let Some(client) = &self.client {
            return client.clone();
        }
        if !self.has_explicit_client_options()
            && let Some(client) = default_openai_client()
        {
            return client;
        }
        let mut client = OpenAIClient::from_env();
        if let Some(key) = &self.api_key {
            client = client.with_api_key(key.clone());
        }
        if let Some(url) = &self.base_url {
            client = client.with_base_url(url.clone());
        }
        if let Some(organization) = &self.organization {
            client = client.with_organization(organization.clone());
        }
        if let Some(project) = &self.project {
            client = client.with_project(project.clone());
        }
        #[cfg(feature = "websocket")]
        if let Some(url) = self.websocket_base_url() {
            client = client.with_websocket_base_url(url);
        }
        client
    }

    fn client(&self) -> OpenAIClient {
        self.openai_client()
    }

    /// Python's `_get_client` builds an `AsyncOpenAI`, which refuses to exist without a key.
    ///
    /// Only a client the provider builds itself is checked: an explicit client
    /// ([`with_client`](Self::with_client)) or the process default client
    /// ([`set_default_openai_client`](super::set_default_openai_client)) is used as given, as in
    /// Python. A built client aimed at another base URL (a local or self-hosted
    /// OpenAI-compatible server) may stay keyless, which Python does not allow.
    fn require_api_key(&self, client: &OpenAIClient) -> Result<()> {
        let built = self.client.is_none()
            && (self.has_explicit_client_options() || default_openai_client().is_none());
        if built && client.is_openai_api() && client.api_key().is_none() {
            return Err(crate::error::AgentsError::user(MISSING_API_KEY_MESSAGE));
        }
        Ok(())
    }

    /// The low-level WebSocket options handed to this provider's models.
    #[cfg(feature = "websocket")]
    pub fn websocket_options(
        &self,
    ) -> &super::openai_responses_ws::OpenAIResponsesWebSocketOptions {
        &self.websocket_options
    }

    /// The Responses WebSocket base URL this provider hands to its models: the one set with
    /// [`with_websocket_base_url`](Self::with_websocket_base_url), otherwise
    /// `OPENAI_WEBSOCKET_BASE_URL`.
    #[cfg(feature = "websocket")]
    pub fn websocket_base_url(&self) -> Option<String> {
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

    #[cfg(feature = "websocket")]
    fn use_responses_websocket(&self) -> bool {
        self.use_responses_websocket
            .unwrap_or_else(super::get_use_responses_websocket_by_default)
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
        self.validate_client_options()?;
        let explicit = model_name.is_some();
        let name = model_name
            .map(str::to_string)
            .unwrap_or_else(get_default_model);
        let client = self.client();
        self.require_api_key(&client)?;
        #[cfg(feature = "websocket")]
        if self.use_responses_websocket() && self.use_responses() {
            let mut cache = self.websocket_models.lock().expect("poisoned");
            let model = cache
                .entry((name.clone(), explicit))
                .or_insert_with(|| {
                    let mut model =
                        super::openai_responses_ws::OpenAIResponsesWSModel::new(name, client)
                            .with_explicit(explicit)
                            .with_websocket_options(self.websocket_options.clone());
                    if let Some(url) = self.websocket_base_url() {
                        model = model.with_websocket_base_url(url);
                    }
                    Arc::new(model)
                })
                .clone();
            return Ok(model);
        }
        #[cfg(not(feature = "websocket"))]
        if super::get_use_responses_websocket_by_default() && self.use_responses() {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "The Responses WebSocket transport is selected but the `websocket` cargo feature \
                 is off; falling back to HTTP."
            );
        }
        if self.use_responses() {
            Ok(Arc::new(
                OpenAIResponsesModel::new(name, client).with_explicit(explicit),
            ))
        } else {
            Ok(Arc::new(
                OpenAIChatCompletionsModel::new(name, client)
                    .with_strict_feature_validation(self.strict_feature_validation)
                    .with_buffer_streamed_tool_calls(self.buffer_streamed_tool_calls),
            ))
        }
    }
}
