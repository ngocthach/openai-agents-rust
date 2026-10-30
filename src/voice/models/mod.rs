//! OpenAI speech-to-text and text-to-speech models. Mirrors `agents.voice.models`.

pub mod openai_model_provider;
pub mod openai_stt;
pub mod openai_tts;

pub use openai_model_provider::{DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, OpenAIVoiceModelProvider};
pub use openai_stt::{OpenAISTTModel, OpenAISTTTranscriptionSession, VoiceWebSocketOptions};
pub use openai_tts::{DEFAULT_VOICE, OpenAITTSModel};

use reqwest::header::HeaderMap;

use crate::error::Result;
use crate::models::OpenAIClient;

/// The client's authentication and default headers.
pub(crate) fn request_headers(client: &OpenAIClient) -> Result<HeaderMap> {
    client.headers(None)
}
