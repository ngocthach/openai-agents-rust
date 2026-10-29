//! Configuration of realtime sessions. Mirrors `agents.realtime.config`.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::guardrail::OutputGuardrail;
use crate::prompts::Prompt;

/// The realtime model used when no model name is configured.
///
/// Matches `DEFAULT_REALTIME_MODEL` in the Python SDK.
pub const DEFAULT_REALTIME_MODEL: &str = "gpt-realtime-2.1";

/// Default voice used when the session does not set one.
pub const DEFAULT_REALTIME_VOICE: &str = "ash";

/// Default transcription model for input audio.
pub const DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL: &str = "gpt-4o-mini-transcribe";

/// Default number of characters accumulated before output guardrails run.
pub const DEFAULT_DEBOUNCE_TEXT_LENGTH: usize = 100;

/// A function tool, or handoff, as the realtime transport sees it.
///
/// The session converts [`crate::tool::FunctionTool`]s and
/// [`crate::realtime::RealtimeHandoff`]s into these before handing settings to a
/// [`crate::realtime::RealtimeModel`], which keeps the model layer free of the context type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeFunctionToolSpec {
    /// Tool name shown to the model.
    pub name: String,
    /// Tool description shown to the model.
    pub description: String,
    /// JSON schema of the tool parameters.
    pub parameters: Value,
}

impl RealtimeFunctionToolSpec {
    /// Build a tool spec.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        RealtimeFunctionToolSpec {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Serialize to the realtime `session.tools` wire format.
    pub fn to_session_param(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        })
    }
}

/// Configuration for transcribing the user's audio.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeInputAudioTranscriptionConfig {
    /// The transcription model, e.g. `gpt-4o-mini-transcribe`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Language code of the expected input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Prompt guiding the transcription.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

impl RealtimeInputAudioTranscriptionConfig {
    /// A transcription config using `model`.
    pub fn new(model: impl Into<String>) -> Self {
        RealtimeInputAudioTranscriptionConfig {
            model: Some(model.into()),
            ..Default::default()
        }
    }
}

/// Noise reduction applied to input audio.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeNoiseReductionConfig {
    /// `near_field` or `far_field`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Server-side voice activity detection settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeTurnDetectionConfig {
    /// `semantic_vad` or `server_vad`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Whether the server creates a response when a turn is detected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,
    /// How eagerly turn boundaries are detected (`auto`, `low`, `medium`, `high`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eagerness: Option<String>,
    /// Whether the server interrupts the assistant when the user speaks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
    /// Padding in milliseconds before detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,
    /// Silence in milliseconds that ends a turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,
    /// Activity detection threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Idle duration in milliseconds after which the server responds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_timeout_ms: Option<u32>,
}

impl RealtimeTurnDetectionConfig {
    /// Semantic VAD that interrupts the assistant response (the SDK default).
    pub fn semantic_vad() -> Self {
        RealtimeTurnDetectionConfig {
            kind: Some("semantic_vad".to_string()),
            interrupt_response: Some(true),
            ..Default::default()
        }
    }
}

/// Tracing metadata sent to the realtime API.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeModelTracingConfig {
    /// Workflow name recorded on the trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_name: Option<String>,
    /// Group id linking multiple traces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    /// Additional metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// How the realtime API traces the session.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeModelTracing {
    /// Let the API choose (the default when nothing is configured).
    Auto,
    /// Do not trace the session.
    Disabled,
    /// Trace with explicit metadata.
    Config(RealtimeModelTracingConfig),
}

impl RealtimeModelTracing {
    /// The `session.tracing` value for this setting, or `None` when nothing should be sent.
    pub fn to_session_value(&self) -> Option<Value> {
        match self {
            RealtimeModelTracing::Auto => Some(Value::String("auto".to_string())),
            RealtimeModelTracing::Disabled => None,
            RealtimeModelTracing::Config(cfg) => serde_json::to_value(cfg).ok(),
        }
    }
}

/// Settings applied to a realtime model session.
///
/// Every field is optional: unset fields keep the transport default, which for the OpenAI model
/// mirrors `DEFAULT_MODEL_SETTINGS` in the Python SDK. Fields typed `Option<Option<T>>`
/// distinguish "unset" (`None`) from "explicitly disabled" (`Some(None)`), which the realtime API
/// encodes as a JSON `null`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeSessionModelSettings {
    /// Name of the realtime model.
    pub model_name: Option<String>,
    /// System instructions.
    pub instructions: Option<String>,
    /// Responses API prompt template.
    pub prompt: Option<Prompt>,
    /// Output modalities, e.g. `["audio"]` or `["text"]`.
    pub output_modalities: Option<Vec<String>>,
    /// Voice used for audio output.
    pub voice: Option<String>,
    /// Playback speed of the model's audio.
    pub speed: Option<f64>,
    /// Maximum output tokens per response (a number, or the string `"inf"`).
    pub max_output_tokens: Option<Value>,
    /// Input audio format (`pcm16`, `g711_ulaw` or `g711_alaw`).
    pub input_audio_format: Option<String>,
    /// Output audio format (`pcm16`, `g711_ulaw` or `g711_alaw`).
    pub output_audio_format: Option<String>,
    /// Transcription of the user's audio.
    pub input_audio_transcription: Option<RealtimeInputAudioTranscriptionConfig>,
    /// Noise reduction for input audio; `Some(None)` disables it explicitly.
    pub input_audio_noise_reduction: Option<Option<RealtimeNoiseReductionConfig>>,
    /// Turn detection; `Some(None)` disables it explicitly.
    pub turn_detection: Option<Option<RealtimeTurnDetectionConfig>>,
    /// How the model chooses tools (`auto`, `required`, `none` or a tool name).
    pub tool_choice: Option<String>,
    /// Whether the model may call tools in parallel.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning configuration, e.g. `{"effort": "low"}`.
    pub reasoning: Option<Value>,
    /// Tools and handoffs exposed to the model.
    pub tools: Option<Vec<RealtimeFunctionToolSpec>>,
    /// Tracing configuration.
    pub tracing: Option<RealtimeModelTracing>,
}

macro_rules! merge_fields {
    ($target:expr, $source:expr, $($field:ident),+ $(,)?) => {
        $(
            if $source.$field.is_some() {
                $target.$field = $source.$field.clone();
            }
        )+
    };
}

impl RealtimeSessionModelSettings {
    /// Empty settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every field that `other` sets, mirroring Python's `{**base, **override}`.
    pub fn merge(&mut self, other: &RealtimeSessionModelSettings) {
        merge_fields!(
            self,
            other,
            model_name,
            instructions,
            prompt,
            output_modalities,
            voice,
            speed,
            max_output_tokens,
            input_audio_format,
            output_audio_format,
            input_audio_transcription,
            input_audio_noise_reduction,
            turn_detection,
            tool_choice,
            parallel_tool_calls,
            reasoning,
            tools,
            tracing,
        );
    }

    /// Return `self` merged with `other`, where `other` wins.
    pub fn merged_with(&self, other: &RealtimeSessionModelSettings) -> Self {
        let mut merged = self.clone();
        merged.merge(other);
        merged
    }

    /// Set the model name.
    pub fn with_model_name(mut self, model: impl Into<String>) -> Self {
        self.model_name = Some(model.into());
        self
    }

    /// Set the system instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set the output modalities.
    pub fn with_output_modalities<I, S>(mut self, modalities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.output_modalities = Some(modalities.into_iter().map(Into::into).collect());
        self
    }

    /// Set the output voice.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set the turn detection config.
    pub fn with_turn_detection(mut self, turn_detection: RealtimeTurnDetectionConfig) -> Self {
        self.turn_detection = Some(Some(turn_detection));
        self
    }

    /// Disable turn detection explicitly.
    pub fn without_turn_detection(mut self) -> Self {
        self.turn_detection = Some(None);
        self
    }

    /// Set the tracing configuration.
    pub fn with_tracing(mut self, tracing: RealtimeModelTracing) -> Self {
        self.tracing = Some(tracing);
        self
    }
}

/// Settings for output guardrail execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeGuardrailsSettings {
    /// Characters of transcript accumulated before guardrails run again.
    ///
    /// Guardrails run each time the accumulated text crosses a multiple of this value.
    pub debounce_text_length: usize,
}

impl Default for RealtimeGuardrailsSettings {
    fn default() -> Self {
        RealtimeGuardrailsSettings {
            debounce_text_length: DEFAULT_DEBOUNCE_TEXT_LENGTH,
        }
    }
}

/// SDK-side execution settings for local realtime tool calls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RealtimeToolExecutionConfig {
    /// Run tool input guardrails before emitting a pending approval event.
    ///
    /// They run again before the tool executes once the call is approved.
    pub pre_approval_tool_input_guardrails: bool,
}

/// Configuration for a whole realtime run.
pub struct RealtimeRunConfig<C> {
    /// Base settings for the model session.
    pub model_settings: Option<RealtimeSessionModelSettings>,
    /// Guardrails run on the assistant transcript, in addition to the agent's own.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Guardrail execution settings.
    pub guardrails_settings: RealtimeGuardrailsSettings,
    /// Disable tracing for this run.
    pub tracing_disabled: bool,
    /// Local tool execution settings.
    pub tool_execution: RealtimeToolExecutionConfig,
}

impl<C> Default for RealtimeRunConfig<C> {
    fn default() -> Self {
        RealtimeRunConfig {
            model_settings: None,
            output_guardrails: Vec::new(),
            guardrails_settings: RealtimeGuardrailsSettings::default(),
            tracing_disabled: false,
            tool_execution: RealtimeToolExecutionConfig::default(),
        }
    }
}

impl<C> Clone for RealtimeRunConfig<C> {
    fn clone(&self) -> Self {
        RealtimeRunConfig {
            model_settings: self.model_settings.clone(),
            output_guardrails: self.output_guardrails.clone(),
            guardrails_settings: self.guardrails_settings.clone(),
            tracing_disabled: self.tracing_disabled,
            tool_execution: self.tool_execution.clone(),
        }
    }
}

impl<C> fmt::Debug for RealtimeRunConfig<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeRunConfig")
            .field("model_settings", &self.model_settings)
            .field("output_guardrails", &self.output_guardrails)
            .field("guardrails_settings", &self.guardrails_settings)
            .field("tracing_disabled", &self.tracing_disabled)
            .field("tool_execution", &self.tool_execution)
            .finish()
    }
}

/// A content part of a structured user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeUserInputContent {
    /// A piece of text.
    InputText {
        /// The text.
        text: String,
    },
    /// An image, by data or remote URL.
    InputImage {
        /// The image URL.
        image_url: String,
        /// Detail hint (`auto`, `low` or `high`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

/// Input sent to the model with [`crate::realtime::RealtimeSession::send_message`].
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeUserInput {
    /// Plain text.
    Text(String),
    /// A structured user message with text and image parts.
    Message(Vec<RealtimeUserInputContent>),
}

impl RealtimeUserInput {
    /// Serialize to a realtime `conversation.item.create` item.
    pub fn to_conversation_item(&self) -> Value {
        let content: Vec<Value> = match self {
            RealtimeUserInput::Text(text) => {
                vec![serde_json::json!({"type": "input_text", "text": text})]
            }
            RealtimeUserInput::Message(parts) => parts
                .iter()
                .filter_map(|part| serde_json::to_value(part).ok())
                .collect(),
        };
        serde_json::json!({"type": "message", "role": "user", "content": content})
    }
}

impl From<&str> for RealtimeUserInput {
    fn from(value: &str) -> Self {
        RealtimeUserInput::Text(value.to_string())
    }
}

impl From<String> for RealtimeUserInput {
    fn from(value: String) -> Self {
        RealtimeUserInput::Text(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_overrides_only_set_fields() {
        let mut base = RealtimeSessionModelSettings::new()
            .with_voice("ash")
            .with_instructions("base");
        let override_settings = RealtimeSessionModelSettings::new().with_instructions("override");
        base.merge(&override_settings);
        assert_eq!(base.voice.as_deref(), Some("ash"));
        assert_eq!(base.instructions.as_deref(), Some("override"));
    }

    #[test]
    fn turn_detection_distinguishes_unset_from_disabled() {
        let unset = RealtimeSessionModelSettings::new();
        assert!(unset.turn_detection.is_none());
        let disabled = RealtimeSessionModelSettings::new().without_turn_detection();
        assert_eq!(disabled.turn_detection, Some(None));
    }

    #[test]
    fn user_input_serializes_text() {
        let item = RealtimeUserInput::from("hello").to_conversation_item();
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["text"], "hello");
        assert_eq!(item["content"][0]["type"], "input_text");
    }
}
