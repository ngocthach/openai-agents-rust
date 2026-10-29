//! OpenAI speech-to-text and text-to-speech models. Mirrors `agents.voice.models`.

pub mod openai_model_provider;
pub mod openai_stt;
pub mod openai_tts;

pub use openai_model_provider::{DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, OpenAIVoiceModelProvider};
pub use openai_stt::{OpenAISTTModel, OpenAISTTTranscriptionSession};
pub use openai_tts::{DEFAULT_VOICE, OpenAITTSModel};

use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::error::{AgentsError, Result};
use crate::models::OpenAIClient;

/// The absolute URL of `path` on the client's base URL.
pub(crate) fn request_url(client: &OpenAIClient, path: &str) -> String {
    format!(
        "{}/{}",
        client.base_url().trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

/// The client's authentication and default headers.
pub(crate) fn request_headers(client: &OpenAIClient) -> Result<HeaderMap> {
    client.headers(None)
}

/// Turn a non-success response into an [`AgentsError::ApiError`].
pub(crate) async fn check_response(response: reqwest::Response) -> Result<reqwest::Response> {
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let text = response.text().await.unwrap_or_default();
    let body: Option<Value> = serde_json::from_str(&text).ok();
    let message = body
        .as_ref()
        .and_then(|body| body.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            if text.is_empty() {
                status.to_string()
            } else {
                text.clone()
            }
        });
    Err(AgentsError::ApiError {
        status: status.as_u16(),
        message,
        body: body.map(Box::new),
        request_id,
        headers: None,
    })
}
