//! The OpenAI voice model provider. Mirrors `agents.voice.models.openai_model_provider`.

use std::sync::Arc;

use crate::models::OpenAIClient;
use crate::voice::model::{STTModel, TTSModel, VoiceModelProvider};
use crate::voice::models::openai_stt::VoiceWebSocketOptions;
use crate::voice::models::{OpenAISTTModel, OpenAITTSModel};

/// The speech-to-text model used when none is given.
pub const DEFAULT_STT_MODEL: &str = "gpt-transcribe";

/// The text-to-speech model used when none is given.
pub const DEFAULT_TTS_MODEL: &str = "gpt-4o-mini-tts";

/// The error Python's constructor raises for an explicit client combined with client options.
const CONFLICTING_CLIENT_OPTIONS: &str =
    "Don't provide api_key, base_url, organization, or project if you provide openai_client";

/// The options that build the provider's own client, as Python's constructor stores them.
#[derive(Debug, Clone, Default)]
struct ClientOptions {
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
}

impl ClientOptions {
    fn any(&self) -> bool {
        self.api_key.is_some()
            || self.base_url.is_some()
            || self.organization.is_some()
            || self.project.is_some()
    }

    /// Python's `_get_client`: with no explicit option the process default client wins;
    /// otherwise a client is built from the environment and the explicit options.
    fn resolve(&self) -> OpenAIClient {
        if !self.any()
            && let Some(client) = crate::models::get_default_openai_client()
        {
            return client;
        }
        let mut client = OpenAIClient::from_env();
        // A missing key falls back to the process default key when the client is used.
        if let Some(key) = &self.api_key {
            client = client.with_api_key(key.clone());
        }
        if let Some(base_url) = &self.base_url {
            client = client.with_base_url(base_url.clone());
        }
        if let Some(organization) = &self.organization {
            client = client.with_organization(organization.clone());
        }
        if let Some(project) = &self.project {
            client = client.with_project(project.clone());
        }
        client
    }
}

/// A voice model provider backed by the OpenAI API.
///
/// Without an explicit client the provider uses the default client set with
/// `set_default_openai_client`, or builds one from the environment, the explicit options and the
/// default key. Python resolves this lazily on first use; Rust resolves it when the provider is
/// built, so set the defaults first.
///
/// An explicit client ([`with_client`](Self::with_client)) cannot be combined with
/// [`with_api_key`](Self::with_api_key), [`with_base_url`](Self::with_base_url),
/// [`with_organization`](Self::with_organization) or [`with_project`](Self::with_project), in
/// either order. Python's constructor rejects the combination; the Rust builders cannot fail, so
/// the conflict is remembered: [`validate`](Self::validate) reports it, and the first operation
/// of a model the provider hands out (`transcribe`, `create_session`, `run`) fails with it. The
/// explicit client is left unchanged.
#[derive(Debug, Clone)]
pub struct OpenAIVoiceModelProvider {
    client: OpenAIClient,
    explicit_client: bool,
    options: ClientOptions,
    http: Option<reqwest::Client>,
    agent_harness_id: Option<String>,
    websocket: VoiceWebSocketOptions,
}

impl Default for OpenAIVoiceModelProvider {
    fn default() -> Self {
        let options = ClientOptions::default();
        OpenAIVoiceModelProvider {
            client: options.resolve(),
            explicit_client: false,
            options,
            http: None,
            agent_harness_id: None,
            websocket: VoiceWebSocketOptions::default(),
        }
    }
}

impl OpenAIVoiceModelProvider {
    /// A provider using the default client, or one configured from the environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use this OpenAI client for both models.
    pub fn with_client(mut self, client: OpenAIClient) -> Self {
        self.client = client;
        self.explicit_client = true;
        self
    }

    /// Use an explicit API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.options.api_key = Some(api_key.into());
        if !self.explicit_client {
            self.client = self.options.resolve();
        }
        self
    }

    /// Use an explicit base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.options.base_url = Some(base_url.into());
        if !self.explicit_client {
            self.client = self.options.resolve();
        }
        self
    }

    /// Use an explicit organization.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.options.organization = Some(organization.into());
        if !self.explicit_client {
            self.client = self.options.resolve();
        }
        self
    }

    /// Use an explicit project.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.options.project = Some(project.into());
        if !self.explicit_client {
            self.client = self.options.resolve();
        }
        self
    }

    /// Share one `reqwest::Client` across the audio requests of both models.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = Some(http);
        self
    }

    /// Set the keepalive and frame-size settings of the streamed transcription websocket.
    pub fn with_websocket_options(mut self, options: VoiceWebSocketOptions) -> Self {
        self.websocket = options;
        self
    }

    /// Set the OpenAI agent harness id. Python's `agent_registration`.
    pub fn with_agent_harness_id(mut self, harness_id: impl Into<String>) -> Self {
        self.agent_harness_id = Some(harness_id.into());
        self
    }

    /// The OpenAI client used by the models this provider creates.
    pub fn client(&self) -> &OpenAIClient {
        &self.client
    }

    /// Fail when an explicit client was combined with an API key, base URL, organization or
    /// project, as Python's constructor does. Models from a provider that fails here fail their
    /// first operation with the same error.
    pub fn validate(&self) -> crate::error::Result<()> {
        match self.conflict() {
            Some(message) => Err(crate::error::AgentsError::user(message)),
            None => Ok(()),
        }
    }

    fn conflict(&self) -> Option<&'static str> {
        (self.explicit_client && self.options.any()).then_some(CONFLICTING_CLIENT_OPTIONS)
    }

    /// The resolved agent harness id: this provider's, the global default, then
    /// `OPENAI_AGENT_HARNESS_ID`.
    ///
    /// Like Python's `OpenAIVoiceModelProvider.agent_registration`, nothing in the voice
    /// pipeline reads it; the harness id reaches trace metadata through the *run's* model
    /// provider (see [`set_default_openai_harness`](crate::models::set_default_openai_harness)).
    pub fn agent_harness_id(&self) -> Option<String> {
        // The default and environment tiers live behind `resolve_openai_harness_id`, which
        // reads the explicit id off a `ModelProvider`; hand it one carrying this id.
        let mut probe = crate::models::OpenAIProvider::default();
        if let Some(id) = &self.agent_harness_id {
            probe = probe.with_agent_harness_id(id.clone());
        }
        crate::models::resolve_openai_harness_id(&probe)
    }
}

impl VoiceModelProvider for OpenAIVoiceModelProvider {
    fn get_stt_model(&self, model_name: Option<&str>) -> Arc<dyn STTModel> {
        let model =
            OpenAISTTModel::new(model_name.unwrap_or(DEFAULT_STT_MODEL), self.client.clone())
                .with_websocket_options(self.websocket)
                .with_config_error(self.conflict());
        Arc::new(match &self.http {
            Some(http) => model.with_http_client(http.clone()),
            None => model,
        })
    }

    fn get_tts_model(&self, model_name: Option<&str>) -> Arc<dyn TTSModel> {
        let model =
            OpenAITTSModel::new(model_name.unwrap_or(DEFAULT_TTS_MODEL), self.client.clone())
                .with_config_error(self.conflict());
        Arc::new(match &self.http {
            Some(http) => model.with_http_client(http.clone()),
            None => model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_the_default_model_names() {
        let provider = OpenAIVoiceModelProvider::new().with_api_key("sk-test");
        assert_eq!(provider.get_stt_model(None).model_name(), DEFAULT_STT_MODEL);
        assert_eq!(provider.get_tts_model(None).model_name(), DEFAULT_TTS_MODEL);
        assert_eq!(
            provider.get_stt_model(Some("whisper-1")).model_name(),
            "whisper-1"
        );
    }

    #[test]
    fn resolves_the_configured_agent_harness_id() {
        let provider = OpenAIVoiceModelProvider::new().with_agent_harness_id("  harness-1  ");
        assert_eq!(provider.agent_harness_id().as_deref(), Some("harness-1"));
    }
}
