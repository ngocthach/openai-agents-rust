//! Configuration of a [`VoicePipeline`](crate::voice::VoicePipeline). Mirrors
//! `agents.voice.pipeline_config`.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::tracing::gen_group_id;
use crate::voice::model::{STTModelSettings, TTSModelSettings, VoiceModelProvider};
use crate::voice::models::OpenAIVoiceModelProvider;

/// Configuration for a [`VoicePipeline`](crate::voice::VoicePipeline).
#[derive(Debug, Clone)]
pub struct VoicePipelineConfig {
    /// The voice model provider used for models given by name. Defaults to OpenAI.
    pub model_provider: Arc<dyn VoiceModelProvider>,
    /// Disable tracing of the pipeline, including when a caller trace is active.
    pub tracing_disabled: bool,
    /// Include sensitive data (transcripts, spoken text) in traces. Defaults to `true`.
    ///
    /// This covers the voice pipeline only, not what happens inside the workflow.
    pub trace_include_sensitive_data: bool,
    /// Include audio data in traces. Defaults to `true`.
    pub trace_include_sensitive_audio_data: bool,
    /// The workflow name used for tracing. Defaults to `Voice Agent`.
    pub workflow_name: String,
    /// A grouping identifier linking traces from the same conversation.
    pub group_id: String,
    /// Additional metadata attached to the trace.
    pub trace_metadata: Option<Map<String, Value>>,
    /// Settings for the STT model.
    pub stt_settings: STTModelSettings,
    /// Settings for the TTS model.
    pub tts_settings: TTSModelSettings,
}

impl Default for VoicePipelineConfig {
    fn default() -> Self {
        VoicePipelineConfig {
            model_provider: Arc::new(OpenAIVoiceModelProvider::new()),
            tracing_disabled: false,
            trace_include_sensitive_data: true,
            trace_include_sensitive_audio_data: true,
            workflow_name: "Voice Agent".to_string(),
            group_id: gen_group_id(),
            trace_metadata: None,
            stt_settings: STTModelSettings::default(),
            tts_settings: TTSModelSettings::default(),
        }
    }
}

impl VoicePipelineConfig {
    /// Default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the voice model provider.
    pub fn with_model_provider(mut self, provider: Arc<dyn VoiceModelProvider>) -> Self {
        self.model_provider = provider;
        self
    }

    /// Disable tracing of the pipeline.
    pub fn with_tracing_disabled(mut self, disabled: bool) -> Self {
        self.tracing_disabled = disabled;
        self
    }

    /// Include transcripts and spoken text in traces.
    pub fn with_trace_include_sensitive_data(mut self, include: bool) -> Self {
        self.trace_include_sensitive_data = include;
        self
    }

    /// Include audio data in traces.
    pub fn with_trace_include_sensitive_audio_data(mut self, include: bool) -> Self {
        self.trace_include_sensitive_audio_data = include;
        self
    }

    /// Set the workflow name used for tracing.
    pub fn with_workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = name.into();
        self
    }

    /// Set the trace group id.
    pub fn with_group_id(mut self, group_id: impl Into<String>) -> Self {
        self.group_id = group_id.into();
        self
    }

    /// Attach metadata to the trace.
    pub fn with_trace_metadata(mut self, metadata: Map<String, Value>) -> Self {
        self.trace_metadata = Some(metadata);
        self
    }

    /// Set the STT settings.
    pub fn with_stt_settings(mut self, settings: STTModelSettings) -> Self {
        self.stt_settings = settings;
        self
    }

    /// Set the TTS settings.
    pub fn with_tts_settings(mut self, settings: TTSModelSettings) -> Self {
        self.tts_settings = settings;
        self
    }
}
