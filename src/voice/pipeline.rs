//! The voice pipeline: transcribe, run a workflow, speak. Mirrors `agents.voice.pipeline`.

use std::fmt;
use std::sync::Arc;

use futures::StreamExt;

use crate::error::Result;
use crate::tracing::scope::ScopeSnapshot;
use crate::tracing::spans::Trace;
use crate::tracing::trace;
use crate::voice::input::{AudioInput, StreamedAudioInput};
use crate::voice::model::{STTModel, TTSModel};
use crate::voice::pipeline_config::VoicePipelineConfig;
use crate::voice::result::{AudioResultWriter, StreamedAudioResult};
use crate::voice::workflow::VoiceWorkflowBase;

/// The audio a [`VoicePipeline`] can run on.
#[derive(Debug)]
pub enum VoiceInput {
    /// A single static buffer.
    Static(AudioInput),
    /// A stream of audio that the caller appends to.
    Streamed(StreamedAudioInput),
}

impl From<AudioInput> for VoiceInput {
    fn from(input: AudioInput) -> Self {
        VoiceInput::Static(input)
    }
}

impl From<StreamedAudioInput> for VoiceInput {
    fn from(input: StreamedAudioInput) -> Self {
        VoiceInput::Streamed(input)
    }
}

/// An opinionated voice agent pipeline. It works in three steps:
///
/// 1. Transcribe audio input into text.
/// 2. Run the provided workflow, which produces a sequence of text responses.
/// 3. Convert the text responses into streaming audio output.
pub struct VoicePipeline {
    workflow: Arc<dyn VoiceWorkflowBase>,
    stt_model: Option<Arc<dyn STTModel>>,
    tts_model: Option<Arc<dyn TTSModel>>,
    stt_model_name: Option<String>,
    tts_model_name: Option<String>,
    config: Arc<VoicePipelineConfig>,
}

impl fmt::Debug for VoicePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoicePipeline")
            .field("workflow", &self.workflow)
            .field(
                "stt_model",
                &self.stt_model.as_ref().map(|m| m.model_name()),
            )
            .field(
                "tts_model",
                &self.tts_model.as_ref().map(|m| m.model_name()),
            )
            .field("stt_model_name", &self.stt_model_name)
            .field("tts_model_name", &self.tts_model_name)
            .finish_non_exhaustive()
    }
}

impl VoicePipeline {
    /// A pipeline running `workflow` with the configured provider's default models.
    pub fn new(workflow: Arc<dyn VoiceWorkflowBase>) -> Self {
        VoicePipeline {
            workflow,
            stt_model: None,
            tts_model: None,
            stt_model_name: None,
            tts_model_name: None,
            config: Arc::new(VoicePipelineConfig::default()),
        }
    }

    /// Use this speech-to-text model.
    pub fn with_stt_model(mut self, model: Arc<dyn STTModel>) -> Self {
        self.stt_model = Some(model);
        self
    }

    /// Resolve the speech-to-text model by name through the configured provider.
    pub fn with_stt_model_name(mut self, name: impl Into<String>) -> Self {
        self.stt_model_name = Some(name.into());
        self
    }

    /// Use this text-to-speech model.
    pub fn with_tts_model(mut self, model: Arc<dyn TTSModel>) -> Self {
        self.tts_model = Some(model);
        self
    }

    /// Resolve the text-to-speech model by name through the configured provider.
    pub fn with_tts_model_name(mut self, name: impl Into<String>) -> Self {
        self.tts_model_name = Some(name.into());
        self
    }

    /// Set the pipeline configuration.
    pub fn with_config(mut self, config: VoicePipelineConfig) -> Self {
        self.config = Arc::new(config);
        self
    }

    /// The pipeline configuration.
    pub fn config(&self) -> &VoicePipelineConfig {
        &self.config
    }

    fn stt(&self) -> Arc<dyn STTModel> {
        match &self.stt_model {
            Some(model) => model.clone(),
            None => self
                .config
                .model_provider
                .get_stt_model(self.stt_model_name.as_deref()),
        }
    }

    fn tts(&self) -> Arc<dyn TTSModel> {
        match &self.tts_model {
            Some(model) => model.clone(),
            None => self
                .config
                .model_provider
                .get_tts_model(self.tts_model_name.as_deref()),
        }
    }

    /// Run the pipeline on static or streamed audio.
    ///
    /// Transcription, the workflow and speech synthesis run in the background; stream the audio
    /// and lifecycle events from the returned [`StreamedAudioResult`].
    pub async fn run(&self, audio_input: impl Into<VoiceInput>) -> Result<StreamedAudioResult> {
        let (mut result, writer) = StreamedAudioResult::new(
            self.tts(),
            Arc::new(self.config.tts_settings.clone()),
            self.config.clone(),
        );
        let stt = self.stt();
        let config = self.config.clone();
        let workflow = self.workflow.clone();
        let input = audio_input.into();

        // Keep the caller's trace, or open the pipeline's own, for the whole async lifecycle.
        let scope = ScopeSnapshot::capture();
        let created = self.trace_to_create(&scope);
        let producer = tokio::spawn(async move {
            let body = run_turns(input, stt, workflow, writer, config);
            match created {
                Some(trace) => trace.run(body).await,
                None => scope.enter(body).await,
            }
        });
        result.set_producer(producer);
        Ok(result)
    }

    /// The trace to open for this run, or `None` to keep the caller's.
    fn trace_to_create(&self, scope: &ScopeSnapshot) -> Option<Trace> {
        if scope.trace.is_some() && !self.config.tracing_disabled {
            return None;
        }
        let mut builder = trace(self.config.workflow_name.clone())
            .group_id(self.config.group_id.clone())
            .disabled(self.config.tracing_disabled);
        if let Some(metadata) = &self.config.trace_metadata {
            builder = builder.metadata(metadata.clone());
        }
        Some(builder.build())
    }
}

/// Closes the session when the producer task ends, whichever way it ends.
struct SessionGuard(Arc<AudioResultWriter>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.close_segments();
    }
}

/// Drive transcription, the workflow and speech synthesis until the session ends.
async fn run_turns(
    input: VoiceInput,
    stt: Arc<dyn STTModel>,
    workflow: Arc<dyn VoiceWorkflowBase>,
    writer: Arc<AudioResultWriter>,
    config: Arc<VoicePipelineConfig>,
) {
    // However this task ends, including a panic or a cancellation, the consumer must still see
    // a terminal event instead of waiting on a session that can no longer produce audio.
    let _guard = SessionGuard(writer.clone());
    let outcome = match input {
        VoiceInput::Static(audio) => {
            run_single_turn(&audio, &stt, workflow.as_ref(), &writer, &config).await
        }
        VoiceInput::Streamed(audio) => {
            run_multi_turn(&audio, &stt, workflow.as_ref(), &writer, &config).await
        }
    };
    match outcome {
        Ok(()) => writer.finish().await,
        Err(error) => {
            log::error!("Error processing voice turns: {error}");
            // A synthesis failure already queued its own error, in audio order.
            if !writer.has_failed() {
                writer.add_error(error);
            }
        }
    }
}

async fn run_single_turn(
    audio: &AudioInput,
    stt: &Arc<dyn STTModel>,
    workflow: &dyn VoiceWorkflowBase,
    writer: &Arc<AudioResultWriter>,
    config: &VoicePipelineConfig,
) -> Result<()> {
    let transcription = stt
        .transcribe(
            audio,
            &config.stt_settings,
            config.trace_include_sensitive_data,
            config.trace_include_sensitive_audio_data,
        )
        .await?;
    run_workflow_turn(workflow, transcription, writer).await?;
    writer.turn_done().await
}

async fn run_multi_turn(
    audio: &StreamedAudioInput,
    stt: &Arc<dyn STTModel>,
    workflow: &dyn VoiceWorkflowBase,
    writer: &Arc<AudioResultWriter>,
    config: &VoicePipelineConfig,
) -> Result<()> {
    let mut emitted_intro = false;
    {
        let mut intro = workflow.on_start();
        while let Some(text) = intro.next().await {
            match text {
                Ok(text) => {
                    writer.add_text(&text);
                    emitted_intro = true;
                }
                Err(error) => {
                    log::warn!("Voice workflow on_start failed: {error}");
                    break;
                }
            }
        }
    }
    if emitted_intro {
        // Finalize the intro turn as part of startup: leaving it open would hold a greeting
        // without sentence-final punctuation until the session ends.
        writer.turn_done().await?;
    }

    let mut session = stt
        .create_session(
            audio,
            &config.stt_settings,
            config.trace_include_sensitive_data,
            config.trace_include_sensitive_audio_data,
        )
        .await?;

    let mut turns = async {
        while let Some(transcription) = session.next_turn().await? {
            run_workflow_turn(workflow, transcription, writer).await?;
            writer.turn_done().await?;
        }
        Ok(())
    }
    .await;

    // Report a close failure only when the turns themselves succeeded.
    if let Err(error) = session.close().await {
        log::error!("Error closing voice transcription session: {error}");
        if turns.is_ok() {
            turns = Err(error);
        }
    }
    turns
}

async fn run_workflow_turn(
    workflow: &dyn VoiceWorkflowBase,
    transcription: String,
    writer: &Arc<AudioResultWriter>,
) -> Result<()> {
    let mut text = workflow.run(transcription);
    while let Some(chunk) = text.next().await {
        writer.add_text(&chunk?);
    }
    Ok(())
}
