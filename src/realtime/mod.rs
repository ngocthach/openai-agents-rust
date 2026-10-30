//! Realtime agents (feature `realtime`). Mirrors `agents.realtime`.
//!
//! A [`RealtimeRunner`] keeps a persistent connection to a realtime model and returns a
//! [`RealtimeSession`] that streams [`RealtimeSessionEvent`]s while you send text and audio. The
//! session owns the local history, executes function tools, runs output guardrails on the
//! assistant transcript and switches agents on handoffs.
//!
//! ```no_run
//! use openai_agents_sdk::realtime::{RealtimeAgent, RealtimeRunner, RealtimeSessionEvent};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let agent = RealtimeAgent::<()>::new("Assistant")
//!         .with_instructions("You are a helpful voice assistant")
//!         .into_arc();
//!
//!     let session = RealtimeRunner::new(agent).run(()).await?;
//!     session.send_message("Hello!").await?;
//!
//!     while let Some(event) = session.recv().await {
//!         if let RealtimeSessionEvent::AudioEnd { .. } = event {
//!             break;
//!         }
//!     }
//!     session.close().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Logging
//!
//! Transcripts, audio and tool arguments are kept out of logs by default. See
//! [`debug`] to opt in.
//!
//! # Interruptions and playback
//!
//! Realtime models generate audio far faster than it is played, so an interruption must truncate
//! the item where the user actually stopped hearing it. By default the transport assumes audio is
//! played immediately at real-time speed, which is right for local playback. For delayed or remote
//! playback, such as a phone call, pass a [`RealtimePlaybackTracker`] through
//! [`RealtimeModelConfig::playback_tracker`] and report progress as audio is played.
//!
//! # Telephony
//!
//! [`OpenAIRealtimeSIPModel`] attaches to a call the Realtime Calls API already created. Build the
//! accept payload with [`OpenAIRealtimeSIPModel::build_initial_session_payload`], POST it to
//! `/v1/realtime/calls/{call_id}/accept`, then attach with the same `call_id`.
//!
//! # Differences from the Python SDK
//!
//! - Transports deliver events through a channel instead of listener objects, so a slow consumer
//!   cannot block the transport's read loop.
//! - [`RealtimeSessionModelSettings`] carries [`RealtimeFunctionToolSpec`]s rather than
//!   `Tool` values, which keeps [`RealtimeModel`] object safe and free of the context type.
//! - [`RealtimeRunner::run`] connects before returning; there is no separate "enter" step.
//! - Interruption modes live in one [`RealtimeInterrupt`] value rather than four loose flags.

pub mod agent;
pub mod config;
pub mod debug;
pub mod events;
pub mod handoffs;
pub mod items;
pub mod model;
pub mod model_events;
pub mod model_inputs;
pub mod openai_model;
pub mod runner;
pub mod session;
pub mod testing;

pub use agent::{RealtimeAgent, RealtimeHooks, RealtimeInstructions};
pub use debug::{
    dont_log_model_data, dont_log_tool_data, reset_dont_log_model_data, reset_dont_log_tool_data,
    set_dont_log_model_data, set_dont_log_tool_data,
};

pub use config::{
    DEFAULT_DEBOUNCE_TEXT_LENGTH, DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL, DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE, RealtimeAudioConfig, RealtimeAudioFormat, RealtimeAudioInputConfig,
    RealtimeAudioOutputConfig, RealtimeAudioStorage, RealtimeFunctionToolSpec,
    RealtimeGuardrailsSettings, RealtimeInputAudioTranscriptionConfig, RealtimeModelTracing,
    RealtimeModelTracingConfig, RealtimeNoiseReductionConfig, RealtimeReasoningConfig,
    RealtimeRunConfig, RealtimeSessionModelSettings, RealtimeToolErrorArgs,
    RealtimeToolErrorFormatter, RealtimeToolErrorKind, RealtimeToolExecutionConfig,
    RealtimeTurnDetectionConfig, RealtimeUserInput, RealtimeUserInputContent, RealtimeVoice,
    default_model_settings,
};
pub use events::{RealtimeEventInfo, RealtimeSessionEvent};
pub use handoffs::{RealtimeHandoff, realtime_handoff};
pub use items::{
    RealtimeContent, RealtimeItem, RealtimeMessageItem, RealtimeMessageRole, RealtimeResponse,
    RealtimeToolCallItem,
};
pub use model::{
    RealtimeApiKey, RealtimeApiKeyProvider, RealtimeModel, RealtimeModelConfig,
    RealtimeModelEvents, RealtimePlaybackState, RealtimePlaybackTracker, RealtimeSendCondition,
    RealtimeTransportConfig, calculate_audio_length_ms,
};
pub use model_events::{
    RealtimeConnectionStatus, RealtimeModelAudioEvent, RealtimeModelCachedTokensDetails,
    RealtimeModelEvent, RealtimeModelInputTokensDetails, RealtimeModelOutputTokensDetails,
    RealtimeModelToolCallEvent,
};
pub use model_inputs::{RealtimeInterrupt, RealtimeModelSendEvent};
pub use openai_model::{OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, get_api_key};
pub use runner::RealtimeRunner;
pub use session::RealtimeSession;
pub use testing::{
    FakeRealtimeModel, RealtimeConnectCall, RealtimeScriptError, RealtimeSendMatcher, RealtimeStep,
    ScriptedRealtimeModel, fake_realtime_model,
};
