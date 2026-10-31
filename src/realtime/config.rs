//! Configuration of realtime sessions. Mirrors `agents.realtime.config`.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::guardrail::OutputGuardrail;
use crate::prompts::Prompt;
use crate::run_config::{ToolErrorFormatterArgs, ToolErrorKind};

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

/// The settings the OpenAI realtime transport falls back to for anything unset.
///
/// Mirrors `DEFAULT_MODEL_SETTINGS` in the Python SDK. These are what the transport sends when a
/// session configures nothing; a session that attaches to an existing call leaves the audio
/// formats alone, since the call already negotiated them.
pub fn default_model_settings() -> RealtimeSessionModelSettings {
    RealtimeSessionModelSettings {
        voice: Some(RealtimeVoice::Name(DEFAULT_REALTIME_VOICE.to_string())),
        modalities: Some(vec!["audio".to_string()]),
        input_audio_format: Some(RealtimeAudioFormat::Name("pcm16".to_string())),
        output_audio_format: Some(RealtimeAudioFormat::Name("pcm16".to_string())),
        input_audio_transcription: Some(RealtimeInputAudioTranscriptionConfig::new(
            DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL,
        )),
        turn_detection: Some(Some(RealtimeTurnDetectionConfig::semantic_vad())),
        ..Default::default()
    }
}

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
    /// The `tool_namespace()` namespace this tool was declared in, if any.
    ///
    /// The Realtime API has no namespaces, so a transport rejects a spec that carries one, as
    /// Python's `ensure_function_tool_supports_responses_only_features` does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Whether the tool asked for deferred loading through hosted tool search, which only the
    /// Responses API supports.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl RealtimeFunctionToolSpec {
    /// Build a tool spec.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        RealtimeFunctionToolSpec {
            name: name.into(),
            description: description.into(),
            parameters,
            namespace: None,
            defer_loading: false,
        }
    }

    /// Record the namespace the tool was declared in.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Record that the tool asked for deferred loading.
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }

    /// The Responses-only features this tool uses, which a Realtime transport cannot honor.
    pub fn responses_only_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();
        if self.namespace.is_some() {
            features.push("tool_namespace()");
        }
        if self.defer_loading {
            features.push("defer_loading=True");
        }
        features
    }

    /// Fail when the tool uses a feature only the Responses API supports.
    pub fn ensure_realtime_supported(&self) -> crate::error::Result<()> {
        let features = self.responses_only_features();
        if features.is_empty() {
            return Ok(());
        }
        let name = match &self.namespace {
            Some(namespace) => format!("{namespace}.{}", self.name),
            None => self.name.clone(),
        };
        Err(crate::error::AgentsError::user(format!(
            "The following function-tool features are only supported with OpenAI Responses models: {}. Tool `{name}` cannot be used with Realtime models.",
            features.join(", ")
        )))
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

/// An audio format for a realtime stream.
///
/// The GA wire format is an object such as `{"type": "audio/pcm", "rate": 24000}`; the legacy
/// names (`pcm16`, `g711_ulaw`, `g711_alaw`) and the GA names are both accepted and normalized to
/// that object. Port of `audio_formats.to_realtime_audio_format`.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeAudioFormat {
    /// A format name, legacy or GA.
    Name(String),
    /// An explicit GA format object, passed through after validation.
    Object(Value),
}

impl RealtimeAudioFormat {
    /// The `audio.*.format` value to send, or `None` when the format is not recognized.
    pub fn to_session_value(&self) -> Option<Value> {
        match self {
            RealtimeAudioFormat::Name(name) => match name.as_str() {
                "pcm16" | "audio/pcm" | "pcm" => {
                    Some(serde_json::json!({"type": "audio/pcm", "rate": 24000}))
                }
                "g711_ulaw" | "audio/pcmu" | "pcmu" => {
                    Some(serde_json::json!({"type": "audio/pcmu"}))
                }
                "g711_alaw" | "audio/pcma" | "pcma" => {
                    Some(serde_json::json!({"type": "audio/pcma"}))
                }
                other => {
                    crate::realtime::debug::log_model_data_debug(
                        "Unknown realtime audio format",
                        &other,
                    );
                    None
                }
            },
            RealtimeAudioFormat::Object(value) => {
                let kind = value.get("type").and_then(Value::as_str);
                match kind {
                    Some("audio/pcm") => {
                        // The API supports one PCM rate; anything else is coerced, as in Python.
                        // Any other rate, a non-number included, is reported and coerced.
                        if let Some(rate) = value.get("rate")
                            && rate.as_f64() != Some(24000.0)
                        {
                            crate::realtime::debug::log_model_data_debug(
                                "Unknown PCM rate in a realtime audio format",
                                &rate,
                            );
                        }
                        Some(serde_json::json!({"type": "audio/pcm", "rate": 24000}))
                    }
                    Some("audio/pcmu") => Some(serde_json::json!({"type": "audio/pcmu"})),
                    Some("audio/pcma") => Some(serde_json::json!({"type": "audio/pcma"})),
                    _ => {
                        crate::realtime::debug::log_model_data_debug(
                            "Unknown realtime audio format mapping",
                            &value,
                        );
                        None
                    }
                }
            }
        }
    }

    /// The format name the audio-length math uses.
    pub fn name(&self) -> Option<&str> {
        match self {
            RealtimeAudioFormat::Name(name) => Some(name.as_str()),
            RealtimeAudioFormat::Object(value) => value.get("type").and_then(Value::as_str),
        }
    }
}

impl From<&str> for RealtimeAudioFormat {
    fn from(value: &str) -> Self {
        RealtimeAudioFormat::Name(value.to_string())
    }
}

impl From<String> for RealtimeAudioFormat {
    fn from(value: String) -> Self {
        RealtimeAudioFormat::Name(value)
    }
}

impl From<Value> for RealtimeAudioFormat {
    fn from(value: Value) -> Self {
        RealtimeAudioFormat::Object(value)
    }
}

/// The voice the model speaks with.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeVoice {
    /// A built-in voice name, such as `ash` or `marin`.
    Name(String),
    /// A custom voice, identified by id.
    Custom {
        /// The custom voice id.
        id: String,
    },
}

impl RealtimeVoice {
    /// The `audio.output.voice` value to send.
    pub fn to_session_value(&self) -> Value {
        match self {
            RealtimeVoice::Name(name) => Value::String(name.clone()),
            RealtimeVoice::Custom { id } => serde_json::json!({ "id": id }),
        }
    }
}

impl From<&str> for RealtimeVoice {
    fn from(value: &str) -> Self {
        RealtimeVoice::Name(value.to_string())
    }
}

impl From<String> for RealtimeVoice {
    fn from(value: String) -> Self {
        RealtimeVoice::Name(value)
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
    /// Literal terms that may appear in the audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,
    /// Expected input languages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    /// Latency/accuracy trade-off for streaming transcription (`minimal` .. `xhigh`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay: Option<String>,
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
    /// Backend-specific VAD model identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
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

/// Reasoning configuration for realtime model responses.
///
/// Port of Python's `RealtimeReasoningConfig`. Unknown keys live in
/// [`extra`](RealtimeReasoningConfig::extra) and are sent verbatim, so a field the API adds needs
/// no SDK release.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeReasoningConfig {
    /// Reasoning effort: `minimal`, `low`, `medium`, `high` or `xhigh`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Any further keys, serialized alongside `effort`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl RealtimeReasoningConfig {
    /// A reasoning config with only the effort set.
    pub fn new(effort: impl Into<String>) -> Self {
        RealtimeReasoningConfig {
            effort: Some(effort.into()),
            ..Default::default()
        }
    }

    /// Add a key that this struct does not model.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// The `session.reasoning` value to send.
    pub fn to_session_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

impl From<&str> for RealtimeReasoningConfig {
    fn from(effort: &str) -> Self {
        RealtimeReasoningConfig::new(effort)
    }
}

impl From<String> for RealtimeReasoningConfig {
    fn from(effort: String) -> Self {
        RealtimeReasoningConfig::new(effort)
    }
}

/// Input half of the nested [`RealtimeAudioConfig`] tree.
///
/// These take precedence over the flat `input_audio_*` and `turn_detection` settings, as in
/// Python's `_get_session_config`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeAudioInputConfig {
    /// Format of the audio the user sends.
    pub format: Option<RealtimeAudioFormat>,
    /// Noise reduction; `Some(None)` disables it explicitly.
    pub noise_reduction: Option<Option<RealtimeNoiseReductionConfig>>,
    /// Transcription of the user's audio.
    pub transcription: Option<RealtimeInputAudioTranscriptionConfig>,
    /// Turn detection; `Some(None)` disables it explicitly.
    pub turn_detection: Option<Option<RealtimeTurnDetectionConfig>>,
}

/// Output half of the nested [`RealtimeAudioConfig`] tree.
///
/// These take precedence over the flat `output_audio_format`, `voice` and `speed` settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeAudioOutputConfig {
    /// Format of the audio the model produces.
    pub format: Option<RealtimeAudioFormat>,
    /// Voice the model speaks with.
    pub voice: Option<RealtimeVoice>,
    /// Playback speed.
    pub speed: Option<f64>,
}

/// The GA nested audio configuration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeAudioConfig {
    /// Settings for the audio the user sends.
    pub input: Option<RealtimeAudioInputConfig>,
    /// Settings for the audio the model produces.
    pub output: Option<RealtimeAudioOutputConfig>,
}

impl RealtimeAudioConfig {
    /// An empty audio configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input half.
    pub fn with_input(mut self, input: RealtimeAudioInputConfig) -> Self {
        self.input = Some(input);
        self
    }

    /// Set the output half.
    pub fn with_output(mut self, output: RealtimeAudioOutputConfig) -> Self {
        self.output = Some(output);
        self
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
    /// Legacy name for [`RealtimeSessionModelSettings::output_modalities`], used when that is
    /// unset.
    pub modalities: Option<Vec<String>>,
    /// The GA nested audio configuration, which wins over the flat audio keys below.
    pub audio: Option<RealtimeAudioConfig>,
    /// Voice used for audio output.
    pub voice: Option<RealtimeVoice>,
    /// Playback speed of the model's audio.
    pub speed: Option<f64>,
    /// Maximum output tokens per response (a number, or the string `"inf"`).
    pub max_output_tokens: Option<Value>,
    /// Input audio format (`pcm16`, `g711_ulaw` or `g711_alaw`, or a GA format object).
    pub input_audio_format: Option<RealtimeAudioFormat>,
    /// Output audio format (`pcm16`, `g711_ulaw` or `g711_alaw`, or a GA format object).
    pub output_audio_format: Option<RealtimeAudioFormat>,
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
    /// Reasoning configuration, e.g. `RealtimeReasoningConfig::new("low")`.
    pub reasoning: Option<RealtimeReasoningConfig>,
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
            modalities,
            audio,
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
    pub fn with_voice(mut self, voice: impl Into<RealtimeVoice>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set the nested audio configuration, which wins over the flat audio keys.
    pub fn with_audio(mut self, audio: RealtimeAudioConfig) -> Self {
        self.audio = Some(audio);
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

    /// Set the reasoning configuration.
    pub fn with_reasoning(mut self, reasoning: impl Into<RealtimeReasoningConfig>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Set the tracing configuration.
    pub fn with_tracing(mut self, tracing: RealtimeModelTracing) -> Self {
        self.tracing = Some(tracing);
        self
    }
}

/// Whether the session keeps base64 audio in its local history.
///
/// The realtime API includes audio bytes in some conversation items, and a long call can
/// accumulate a lot of them. Transcripts are always kept.
///
/// The Python SDK has only a `TODO (rm) Add history audio storage config` here, so this knob is
/// this port's implementation of that intent. [`RealtimeAudioStorage::Keep`] is the default
/// because it is what Python does today.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RealtimeAudioStorage {
    /// Keep audio exactly as the transport reports it (the Python behavior).
    #[default]
    Keep,
    /// Drop the audio bytes from history items, keeping transcripts and every other field.
    DiscardAudio,
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

/// Why the SDK is answering a tool call itself rather than running the tool.
///
/// The realtime session reuses the core [`ToolErrorKind`], as Python's realtime config reuses
/// `agents.run_config.ToolErrorFormatter`. Only `ApprovalRejected` and `ToolNotFound` reach a
/// realtime formatter; `ToolException` is produced by the core runner.
pub type RealtimeToolErrorKind = ToolErrorKind;

/// What a [`RealtimeToolErrorFormatter`] is given.
///
/// An alias of the core [`ToolErrorFormatterArgs`], so one formatter can serve both a
/// [`crate::RunConfig`] and a [`RealtimeRunConfig`].
pub type RealtimeToolErrorArgs<'a, C> = ToolErrorFormatterArgs<'a, C>;

/// Formats the model-visible message for a tool call the SDK answers itself.
///
/// Returning `None` keeps the default message.
pub type RealtimeToolErrorFormatter<C> =
    std::sync::Arc<dyn for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Option<String> + Send + Sync>;

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
    /// Whether a function tool call delivered by the transport runs on its own task.
    ///
    /// `true` (the default, as in Python) keeps the session's event loop free while a tool runs,
    /// so audio, transcript and interrupt events are still delivered. `false` awaits each call
    /// inside the event loop, which serializes tool calls with every other transport event.
    pub async_tool_calls: bool,
    /// Local tool execution settings.
    pub tool_execution: RealtimeToolExecutionConfig,
    /// Whether base64 audio is kept in the session history.
    pub history_audio_storage: RealtimeAudioStorage,
    /// Formats the message sent to the model when a tool call is rejected or not found.
    pub tool_error_formatter: Option<RealtimeToolErrorFormatter<C>>,
    /// Lifecycle callbacks for the whole run, in addition to each agent's own.
    pub hooks: Option<std::sync::Arc<dyn crate::realtime::agent::RealtimeHooks<C>>>,
}

impl<C> Default for RealtimeRunConfig<C> {
    fn default() -> Self {
        RealtimeRunConfig {
            model_settings: None,
            output_guardrails: Vec::new(),
            guardrails_settings: RealtimeGuardrailsSettings::default(),
            tracing_disabled: false,
            async_tool_calls: true,
            tool_execution: RealtimeToolExecutionConfig::default(),
            history_audio_storage: RealtimeAudioStorage::default(),
            tool_error_formatter: None,
            hooks: None,
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
            async_tool_calls: self.async_tool_calls,
            tool_execution: self.tool_execution.clone(),
            history_audio_storage: self.history_audio_storage,
            tool_error_formatter: self.tool_error_formatter.clone(),
            hooks: self.hooks.clone(),
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
            .field("async_tool_calls", &self.async_tool_calls)
            .field("tool_execution", &self.tool_execution)
            .field("history_audio_storage", &self.history_audio_storage)
            .field(
                "tool_error_formatter",
                &self.tool_error_formatter.as_ref().map(|_| "..."),
            )
            .field("hooks", &self.hooks)
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
        assert_eq!(base.voice, Some(RealtimeVoice::from("ash")));
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
    fn audio_formats_normalize_to_ga_objects() {
        assert_eq!(
            RealtimeAudioFormat::from("pcm16").to_session_value(),
            Some(serde_json::json!({"type": "audio/pcm", "rate": 24000}))
        );
        assert_eq!(
            RealtimeAudioFormat::from("g711_ulaw").to_session_value(),
            Some(serde_json::json!({"type": "audio/pcmu"}))
        );
        assert_eq!(
            RealtimeAudioFormat::from("g711_alaw").to_session_value(),
            Some(serde_json::json!({"type": "audio/pcma"}))
        );
        assert_eq!(
            RealtimeAudioFormat::from("nonsense").to_session_value(),
            None
        );
    }

    #[test]
    fn a_format_object_is_validated_and_its_rate_coerced() {
        let coerced =
            RealtimeAudioFormat::from(serde_json::json!({"type": "audio/pcm", "rate": 8000}))
                .to_session_value();
        // The API supports one PCM rate, so an unexpected one is coerced rather than forwarded.
        assert_eq!(
            coerced,
            Some(serde_json::json!({"type": "audio/pcm", "rate": 24000}))
        );
        assert_eq!(
            RealtimeAudioFormat::from(serde_json::json!({"type": "audio/pcmu"})).to_session_value(),
            Some(serde_json::json!({"type": "audio/pcmu"}))
        );
        assert_eq!(
            RealtimeAudioFormat::from(serde_json::json!({"type": "audio/unknown"}))
                .to_session_value(),
            None
        );
        assert_eq!(
            RealtimeAudioFormat::from(serde_json::json!({"type": "audio/pcmu"})).name(),
            Some("audio/pcmu")
        );
    }

    #[test]
    fn a_custom_voice_is_sent_as_an_object() {
        assert_eq!(
            RealtimeVoice::from("ash").to_session_value(),
            serde_json::json!("ash")
        );
        assert_eq!(
            RealtimeVoice::Custom {
                id: "voice_123".to_string()
            }
            .to_session_value(),
            serde_json::json!({"id": "voice_123"})
        );
    }

    #[test]
    fn user_input_serializes_text() {
        let item = RealtimeUserInput::from("hello").to_conversation_item();
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["text"], "hello");
        assert_eq!(item["content"][0]["type"], "input_text");
    }
}
