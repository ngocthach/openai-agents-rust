//! The OpenAI text-to-speech model. Mirrors `agents.voice.models.openai_tts`.

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::error::Result;
use crate::models::OpenAIClient;
use crate::voice::model::{AudioByteStream, TTSModel, TTSModelSettings, TTSVoice};
use crate::voice::models::{check_response, request_headers, request_url};

/// The voice used when [`TTSModelSettings::voice`] is not set.
pub const DEFAULT_VOICE: &str = "ash";

/// A text-to-speech model served by the OpenAI `/audio/speech` endpoint.
#[derive(Debug, Clone)]
pub struct OpenAITTSModel {
    model: String,
    client: OpenAIClient,
    http: reqwest::Client,
}

impl OpenAITTSModel {
    /// A text-to-speech model using `client`.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAITTSModel {
            model: model.into(),
            client,
            http: crate::util::http_client(),
        }
    }

    /// Use a custom `reqwest::Client` for the audio request.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The request body sent to `/audio/speech`.
    pub(crate) fn request_body(&self, text: &str, settings: &TTSModelSettings) -> Value {
        let voice = settings.voice.as_ref().map_or_else(
            || Value::String(DEFAULT_VOICE.to_string()),
            TTSVoice::to_json,
        );
        let mut body = json!({
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": "pcm",
            "instructions": settings.instructions,
        });
        if let Some(speed) = settings.speed {
            body["speed"] = json!(speed);
        }
        body
    }
}

#[async_trait]
impl TTSModel for OpenAITTSModel {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn run(&self, text: &str, settings: &TTSModelSettings) -> Result<AudioByteStream> {
        let response = self
            .http
            .post(request_url(&self.client, "/audio/speech"))
            .headers(request_headers(&self.client)?)
            .json(&self.request_body(text, settings))
            .send()
            .await?;
        let response = check_response(response).await?;
        Ok(response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Into::into))
            .boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sends_pcm_with_the_default_voice_and_instructions() {
        let model = OpenAITTSModel::new("test-model", OpenAIClient::new("sk-test"));
        let settings = TTSModelSettings::default();
        let body = model.request_body("hello world", &settings);
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["voice"], "ash");
        assert_eq!(body["input"], "hello world");
        assert_eq!(body["response_format"], "pcm");
        assert_eq!(body["instructions"], settings.instructions);
        assert!(body.get("speed").is_none());
    }

    #[test]
    fn sends_a_custom_voice_and_speed() {
        let model = OpenAITTSModel::new("test-model", OpenAIClient::new("sk-test"));
        let settings = TTSModelSettings::default()
            .with_voice(TTSVoice::custom("voice_1234"))
            .with_speed(1.5);
        let body = model.request_body("hi", &settings);
        assert_eq!(body["voice"], json!({"id": "voice_1234"}));
        assert_eq!(body["speed"], 1.5);
    }
}
