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

/// A voice model provider backed by the OpenAI API.
#[derive(Debug, Clone, Default)]
pub struct OpenAIVoiceModelProvider {
    client: OpenAIClient,
    http: Option<reqwest::Client>,
    agent_harness_id: Option<String>,
    websocket: VoiceWebSocketOptions,
}

impl OpenAIVoiceModelProvider {
    /// A provider using a client configured from the environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use this OpenAI client for both models.
    pub fn with_client(mut self, client: OpenAIClient) -> Self {
        self.client = client;
        self
    }

    /// Use an explicit API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = self.client.with_api_key(api_key);
        self
    }

    /// Use an explicit base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client = self.client.with_base_url(base_url);
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
                .with_websocket_options(self.websocket);
        Arc::new(match &self.http {
            Some(http) => model.with_http_client(http.clone()),
            None => model,
        })
    }

    fn get_tts_model(&self, model_name: Option<&str>) -> Arc<dyn TTSModel> {
        let model =
            OpenAITTSModel::new(model_name.unwrap_or(DEFAULT_TTS_MODEL), self.client.clone());
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
