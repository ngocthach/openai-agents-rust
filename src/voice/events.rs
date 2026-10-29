//! Events streamed by a [`VoicePipeline`](crate::voice::VoicePipeline). Mirrors
//! `agents.voice.events`.

use std::sync::Arc;

use crate::error::AgentsError;
use crate::voice::input::AudioData;

/// A point in the lifecycle of a voice session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceLifecycleEvent {
    /// A new turn started: the pipeline began generating speech for it.
    TurnStarted,
    /// The current turn produced all of its audio.
    TurnEnded,
    /// The session is over; no further events follow.
    SessionEnded,
}

impl VoiceLifecycleEvent {
    /// The Python SDK name of the event (`turn_started`, `turn_ended`, `session_ended`).
    pub fn as_str(self) -> &'static str {
        match self {
            VoiceLifecycleEvent::TurnStarted => "turn_started",
            VoiceLifecycleEvent::TurnEnded => "turn_ended",
            VoiceLifecycleEvent::SessionEnded => "session_ended",
        }
    }
}

/// An event from a [`VoicePipeline`](crate::voice::VoicePipeline), streamed via
/// [`StreamedAudioResult::stream`](crate::voice::StreamedAudioResult::stream).
#[derive(Debug, Clone)]
pub enum VoiceStreamEvent {
    /// Synthesized audio, in the format configured by
    /// [`TTSModelSettings::dtype`](crate::voice::TTSModelSettings).
    Audio(AudioData),
    /// A lifecycle transition.
    Lifecycle(VoiceLifecycleEvent),
    /// The session failed. This is always the last event.
    Error(Arc<AgentsError>),
}

impl VoiceStreamEvent {
    /// The Python SDK event type (`voice_stream_event_audio`, `voice_stream_event_lifecycle`,
    /// `voice_stream_event_error`).
    pub fn event_type(&self) -> &'static str {
        match self {
            VoiceStreamEvent::Audio(_) => "voice_stream_event_audio",
            VoiceStreamEvent::Lifecycle(_) => "voice_stream_event_lifecycle",
            VoiceStreamEvent::Error(_) => "voice_stream_event_error",
        }
    }

    /// The audio payload, if this is an audio event.
    pub fn audio(&self) -> Option<&AudioData> {
        match self {
            VoiceStreamEvent::Audio(data) => Some(data),
            _ => None,
        }
    }

    /// The lifecycle event, if this is a lifecycle event.
    pub fn lifecycle(&self) -> Option<VoiceLifecycleEvent> {
        match self {
            VoiceStreamEvent::Lifecycle(event) => Some(*event),
            _ => None,
        }
    }

    /// The error, if this is an error event.
    pub fn error(&self) -> Option<&Arc<AgentsError>> {
        match self {
            VoiceStreamEvent::Error(error) => Some(error),
            _ => None,
        }
    }
}
