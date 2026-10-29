//! Voice pipelines (feature `voice`). Mirrors `agents.voice`.
//!
//! A [`VoicePipeline`] works in three steps: transcribe audio into text with an [`STTModel`],
//! run a [`VoiceWorkflowBase`] on the transcription, and speak its text with a [`TTSModel`].
//!
//! Audio is 16-bit PCM (`Vec<i16>`) instead of NumPy arrays; use
//! [`f32_to_i16`] / [`i16_to_f32`] for float buffers.
//!
//! ```no_run
//! use std::sync::Arc;
//! use openai_agents_sdk::Agent;
//! use openai_agents_sdk::voice::{
//!     AudioInput, SingleAgentVoiceWorkflow, VoicePipeline, VoiceStreamEvent,
//! };
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = Agent::<()>::new("Assistant").with_instructions("Answer in one sentence.");
//! let pipeline = VoicePipeline::new(Arc::new(SingleAgentVoiceWorkflow::new(agent)));
//! let mut result = pipeline.run(AudioInput::new(vec![0i16; 24_000])).await?;
//!
//! let mut stream = std::pin::pin!(result.stream());
//! use futures::StreamExt;
//! while let Some(event) = stream.next().await {
//!     if let VoiceStreamEvent::Audio(audio) = event {
//!         // play `audio`
//!         let _ = audio.to_int16();
//!     }
//! }
//! # Ok(())
//! # }
//! ```

pub mod events;
pub mod input;
pub mod model;
pub mod models;
pub mod pipeline;
pub mod pipeline_config;
pub mod result;
pub mod testing;
pub mod utils;
pub mod workflow;

pub use events::{VoiceLifecycleEvent, VoiceStreamEvent};
pub use input::{
    AudioData, AudioFormat, AudioInput, DEFAULT_SAMPLE_RATE, StreamedAudioInput, f32_to_i16,
    i16_to_f32,
};
pub use model::{
    AudioByteStream, AudioTransform, DEFAULT_TTS_BUFFER_SIZE, DEFAULT_TTS_INSTRUCTIONS, STTModel,
    STTModelSettings, StreamedTranscriptionSession, TTSModel, TTSModelSettings, TTSVoice,
    VoiceModelProvider,
};
pub use models::{
    DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, OpenAISTTModel, OpenAISTTTranscriptionSession,
    OpenAITTSModel, OpenAIVoiceModelProvider,
};
pub use pipeline::{VoiceInput, VoicePipeline};
pub use pipeline_config::VoicePipelineConfig;
pub use result::StreamedAudioResult;
pub use utils::{DEFAULT_MIN_SENTENCE_LENGTH, TextSplitter, get_sentence_based_splitter};
pub use workflow::{
    SingleAgentVoiceWorkflow, SingleAgentWorkflowCallbacks, TextStream, VoiceWorkflowBase,
    VoiceWorkflowHelper,
};
