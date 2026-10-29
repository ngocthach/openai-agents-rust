//! Events emitted by a realtime transport. Mirrors `agents.realtime.model_events`.

use serde_json::Value;

use crate::realtime::items::RealtimeItem;
use crate::usage::Usage;

/// State of the transport connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeConnectionStatus {
    /// The transport is connecting.
    Connecting,
    /// The transport is connected.
    Connected,
    /// The transport is disconnected.
    Disconnected,
}

/// The model attempted a function call.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RealtimeModelToolCallEvent {
    /// Name of the tool.
    pub name: String,
    /// Call id used to correlate the output.
    pub call_id: String,
    /// JSON-encoded arguments.
    pub arguments: String,
    /// Item id of the call, when the transport reports one.
    pub id: Option<String>,
    /// Id of the item this call follows, when known.
    pub previous_item_id: Option<String>,
}

impl RealtimeModelToolCallEvent {
    /// A tool call event with the given name, call id and arguments.
    pub fn new(
        name: impl Into<String>,
        call_id: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        RealtimeModelToolCallEvent {
            name: name.into(),
            call_id: call_id.into(),
            arguments: arguments.into(),
            id: None,
            previous_item_id: None,
        }
    }
}

/// Raw audio produced by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeModelAudioEvent {
    /// Decoded audio bytes.
    pub data: Vec<u8>,
    /// The response this audio belongs to.
    pub response_id: String,
    /// The item containing the audio.
    pub item_id: String,
    /// Index of the audio content inside the item.
    pub content_index: usize,
}

/// An event emitted by a [`crate::realtime::RealtimeModel`].
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeModelEvent {
    /// A transport or server error.
    Error {
        /// The error payload reported by the transport.
        error: Value,
    },
    /// The model called a function tool.
    FunctionCall(RealtimeModelToolCallEvent),
    /// New audio from the model.
    Audio(Box<RealtimeModelAudioEvent>),
    /// Audio playback was interrupted.
    AudioInterrupted {
        /// The item containing the audio.
        item_id: String,
        /// Index of the audio content inside the item.
        content_index: usize,
    },
    /// The model finished generating audio for an item.
    AudioDone {
        /// The item containing the audio.
        item_id: String,
        /// Index of the audio content inside the item.
        content_index: usize,
    },
    /// The user's audio was transcribed.
    InputAudioTranscriptionCompleted {
        /// The transcribed item.
        item_id: String,
        /// The transcript.
        transcript: String,
    },
    /// The server detected a period of user inactivity.
    InputAudioTimeoutTriggered {
        /// The item the timeout applies to.
        item_id: String,
        /// Start of the audio window, in milliseconds.
        audio_start_ms: u64,
        /// End of the audio window, in milliseconds.
        audio_end_ms: u64,
    },
    /// A partial transcript of the model's audio.
    TranscriptDelta {
        /// The item being transcribed.
        item_id: String,
        /// The new text.
        delta: String,
        /// The response this transcript belongs to.
        response_id: String,
    },
    /// A partial text output from the model.
    OutputTextDelta {
        /// The item being written.
        item_id: String,
        /// The new text.
        delta: String,
        /// The response this text belongs to.
        response_id: String,
    },
    /// An item was added to, or updated in, the conversation.
    ItemUpdated {
        /// The new or updated item.
        item: Box<RealtimeItem>,
    },
    /// An item was removed from the conversation.
    ItemDeleted {
        /// The removed item id.
        item_id: String,
    },
    /// The connection status changed.
    ConnectionStatus {
        /// The new status.
        status: RealtimeConnectionStatus,
    },
    /// The event stream ended and no further events will arrive.
    EndOfStream,
    /// The model started generating a response.
    TurnStarted {
        /// The response id, when the transport reports one.
        response_id: Option<String>,
    },
    /// Token usage for a completed response.
    Usage {
        /// The usage totals.
        usage: Box<Usage>,
    },
    /// The model finished generating a response.
    TurnEnded {
        /// The response id, when the transport reports one.
        response_id: Option<String>,
    },
    /// A vendor-specific event the SDK does not model.
    Other {
        /// The raw payload.
        data: Value,
    },
    /// The transport raised an exception.
    Exception {
        /// Description of the failure.
        message: String,
        /// Where the failure happened.
        context: Option<String>,
    },
    /// A raw server event, forwarded before interpretation.
    RawServerEvent {
        /// The raw payload.
        data: Value,
    },
}

impl RealtimeModelEvent {
    /// Build an audio event.
    pub fn audio(event: RealtimeModelAudioEvent) -> Self {
        RealtimeModelEvent::Audio(Box::new(event))
    }

    /// Build an item-updated event.
    pub fn item_updated(item: impl Into<RealtimeItem>) -> Self {
        RealtimeModelEvent::ItemUpdated {
            item: Box::new(item.into()),
        }
    }

    /// The Python event `type` string, useful for logging and matching.
    pub fn event_type(&self) -> &'static str {
        match self {
            RealtimeModelEvent::Error { .. } => "error",
            RealtimeModelEvent::FunctionCall(_) => "function_call",
            RealtimeModelEvent::Audio(_) => "audio",
            RealtimeModelEvent::AudioInterrupted { .. } => "audio_interrupted",
            RealtimeModelEvent::AudioDone { .. } => "audio_done",
            RealtimeModelEvent::InputAudioTranscriptionCompleted { .. } => {
                "input_audio_transcription_completed"
            }
            RealtimeModelEvent::InputAudioTimeoutTriggered { .. } => {
                "input_audio_timeout_triggered"
            }
            RealtimeModelEvent::TranscriptDelta { .. } => "transcript_delta",
            RealtimeModelEvent::OutputTextDelta { .. } => "output_text_delta",
            RealtimeModelEvent::ItemUpdated { .. } => "item_updated",
            RealtimeModelEvent::ItemDeleted { .. } => "item_deleted",
            RealtimeModelEvent::ConnectionStatus { .. } => "connection_status",
            RealtimeModelEvent::EndOfStream => "end_of_stream",
            RealtimeModelEvent::TurnStarted { .. } => "turn_started",
            RealtimeModelEvent::Usage { .. } => "usage",
            RealtimeModelEvent::TurnEnded { .. } => "turn_ended",
            RealtimeModelEvent::Other { .. } => "other",
            RealtimeModelEvent::Exception { .. } => "exception",
            RealtimeModelEvent::RawServerEvent { .. } => "raw_server_event",
        }
    }
}
