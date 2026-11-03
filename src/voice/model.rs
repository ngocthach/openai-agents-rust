//! Speech-to-text and text-to-speech interfaces. Mirrors `agents.voice.model`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use serde_json::{Value, json};

use crate::error::Result;
use crate::voice::input::{AudioData, AudioFormat, AudioInput, StreamedAudioInput};
use crate::voice::utils::{DEFAULT_MIN_SENTENCE_LENGTH, TextSplitter, get_sentence_based_splitter};

/// Default instructions sent to the TTS model.
pub const DEFAULT_TTS_INSTRUCTIONS: &str =
    "You will receive partial sentences. Do not complete the sentence just read out the text.";

/// Default number of audio chunks buffered before an audio event is emitted.
pub const DEFAULT_TTS_BUFFER_SIZE: usize = 120;

/// A stream of PCM audio chunks produced by a [`TTSModel`].
pub type AudioByteStream = BoxStream<'static, Result<Bytes>>;

/// Transforms audio before it is emitted, e.g. to reshape it for a playback library.
pub type AudioTransform = Arc<dyn Fn(AudioData) -> AudioData + Send + Sync>;

/// A voice for the OpenAI TTS models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TTSVoice {
    /// The `alloy` built-in voice.
    Alloy,
    /// The `ash` built-in voice.
    Ash,
    /// The `ballad` built-in voice.
    Ballad,
    /// The `coral` built-in voice.
    Coral,
    /// The `echo` built-in voice.
    Echo,
    /// The `fable` built-in voice.
    Fable,
    /// The `onyx` built-in voice.
    Onyx,
    /// The `nova` built-in voice.
    Nova,
    /// The `sage` built-in voice.
    Sage,
    /// The `shimmer` built-in voice.
    Shimmer,
    /// The `verse` built-in voice.
    Verse,
    /// The `marin` built-in voice.
    Marin,
    /// The `cedar` built-in voice.
    Cedar,
    /// A custom voice id (Python's `TTSCustomVoice`).
    Custom(String),
}

impl TTSVoice {
    /// A custom voice referenced by id.
    pub fn custom(id: impl Into<String>) -> Self {
        TTSVoice::Custom(id.into())
    }

    /// The wire name of a built-in voice, or `None` for a custom voice.
    pub fn builtin_name(&self) -> Option<&'static str> {
        Some(match self {
            TTSVoice::Alloy => "alloy",
            TTSVoice::Ash => "ash",
            TTSVoice::Ballad => "ballad",
            TTSVoice::Coral => "coral",
            TTSVoice::Echo => "echo",
            TTSVoice::Fable => "fable",
            TTSVoice::Onyx => "onyx",
            TTSVoice::Nova => "nova",
            TTSVoice::Sage => "sage",
            TTSVoice::Shimmer => "shimmer",
            TTSVoice::Verse => "verse",
            TTSVoice::Marin => "marin",
            TTSVoice::Cedar => "cedar",
            TTSVoice::Custom(_) => return None,
        })
    }

    /// The `voice` request field: a string for built-in voices, `{"id": ...}` for custom ones.
    pub fn to_json(&self) -> Value {
        match self {
            TTSVoice::Custom(id) => json!({ "id": id }),
            other => Value::String(other.builtin_name().expect("builtin voice").to_string()),
        }
    }
}

/// Settings for a [`TTSModel`].
#[derive(Clone)]
pub struct TTSModelSettings {
    /// The voice to use. Defaults to the model's own default voice.
    pub voice: Option<TTSVoice>,
    /// The number of audio chunks buffered before an audio event is emitted.
    pub buffer_size: usize,
    /// The sample format of the emitted audio.
    pub dtype: AudioFormat,
    /// An optional transform applied to each emitted audio buffer.
    pub transform_data: Option<AudioTransform>,
    /// Instructions controlling the tone of the audio output.
    pub instructions: String,
    /// Splits the generated text into chunks that are spoken as they arrive.
    pub text_splitter: TextSplitter,
    /// Reading speed, between 0.25 and 4.0.
    pub speed: Option<f32>,
}

impl fmt::Debug for TTSModelSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TTSModelSettings")
            .field("voice", &self.voice)
            .field("buffer_size", &self.buffer_size)
            .field("dtype", &self.dtype)
            .field("transform_data", &self.transform_data.is_some())
            .field("instructions", &self.instructions)
            .field("speed", &self.speed)
            .finish_non_exhaustive()
    }
}

impl Default for TTSModelSettings {
    fn default() -> Self {
        TTSModelSettings {
            voice: None,
            buffer_size: DEFAULT_TTS_BUFFER_SIZE,
            dtype: AudioFormat::Int16,
            transform_data: None,
            instructions: DEFAULT_TTS_INSTRUCTIONS.to_string(),
            text_splitter: get_sentence_based_splitter(DEFAULT_MIN_SENTENCE_LENGTH),
            speed: None,
        }
    }
}

impl TTSModelSettings {
    /// Default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the voice.
    pub fn with_voice(mut self, voice: TTSVoice) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set how many audio chunks are buffered before an audio event is emitted.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set the sample format of the emitted audio.
    pub fn with_dtype(mut self, dtype: AudioFormat) -> Self {
        self.dtype = dtype;
        self
    }

    /// Transform every emitted audio buffer.
    pub fn with_transform_data(mut self, transform: AudioTransform) -> Self {
        self.transform_data = Some(transform);
        self
    }

    /// Set the instructions sent to the model.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Set the text splitter.
    pub fn with_text_splitter(mut self, splitter: TextSplitter) -> Self {
        self.text_splitter = splitter;
        self
    }

    /// Set the reading speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
}

/// A text-to-speech model that converts text into audio output.
#[async_trait]
pub trait TTSModel: Send + Sync + fmt::Debug {
    /// The name of the TTS model.
    fn model_name(&self) -> &str;

    /// Convert `text` into a stream of PCM audio chunks.
    async fn run(&self, text: &str, settings: &TTSModelSettings) -> Result<AudioByteStream>;
}

/// Settings for an [`STTModel`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct STTModelSettings {
    /// Instructions for the model to follow.
    pub prompt: Option<String>,
    /// The language of the audio input.
    pub language: Option<String>,
    /// The sampling temperature.
    pub temperature: Option<f32>,
    /// Turn detection settings used with streamed audio input.
    pub turn_detection: Option<Value>,
    /// Possible languages of streamed audio input. Takes precedence over
    /// [`STTModelSettings::language`].
    pub languages: Option<Vec<String>>,
    /// Words or phrases guiding the transcription of streamed audio input.
    pub keywords: Option<Vec<String>>,
}

impl STTModelSettings {
    /// Default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the turn detection settings.
    pub fn with_turn_detection(mut self, turn_detection: Value) -> Self {
        self.turn_detection = Some(turn_detection);
        self
    }

    /// Set the candidate languages.
    pub fn with_languages(mut self, languages: Vec<String>) -> Self {
        self.languages = Some(languages);
        self
    }

    /// Set the transcription keywords.
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = Some(keywords);
        self
    }
}

/// A streamed transcription of audio input.
///
/// Python exposes `transcribe_turns()` as an async iterator; here each turn is pulled with
/// [`StreamedTranscriptionSession::next_turn`] until it returns `None`.
#[async_trait]
pub trait StreamedTranscriptionSession: Send + Sync + fmt::Debug {
    /// The next transcribed turn, or `None` once the session is over.
    async fn next_turn(&mut self) -> Result<Option<String>>;

    /// Close the session.
    async fn close(&mut self) -> Result<()>;
}

/// A speech-to-text model that converts audio input into text.
#[async_trait]
pub trait STTModel: Send + Sync + fmt::Debug {
    /// The name of the STT model.
    fn model_name(&self) -> &str;

    /// Transcribe a static audio buffer.
    async fn transcribe(
        &self,
        input: &AudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<String>;

    /// Create a session that transcribes streamed audio turn by turn.
    async fn create_session(
        &self,
        input: &StreamedAudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<Box<dyn StreamedTranscriptionSession>>;
}

/// Creates speech-to-text and text-to-speech models by name.
pub trait VoiceModelProvider: Send + Sync + fmt::Debug {
    /// Get a speech-to-text model, or the provider's default when `model_name` is `None`.
    fn get_stt_model(&self, model_name: Option<&str>) -> Arc<dyn STTModel>;

    /// Get a text-to-speech model, or the provider's default when `model_name` is `None`.
    fn get_tts_model(&self, model_name: Option<&str>) -> Arc<dyn TTSModel>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_builtin_and_custom_voices() {
        assert_eq!(TTSVoice::Marin.to_json(), json!("marin"));
        assert_eq!(
            TTSVoice::custom("voice_1234").to_json(),
            json!({"id": "voice_1234"})
        );
        assert!(TTSVoice::custom("voice_1234").builtin_name().is_none());
    }

    #[test]
    fn default_tts_settings_match_python() {
        let settings = TTSModelSettings::default();
        assert_eq!(settings.buffer_size, 120);
        assert_eq!(settings.instructions, DEFAULT_TTS_INSTRUCTIONS);
        assert_eq!(settings.dtype, AudioFormat::Int16);
        assert!(settings.voice.is_none());
    }
}
