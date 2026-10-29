//! Events sent to a realtime transport. Mirrors `agents.realtime.model_inputs`.

use serde_json::Value;

use crate::realtime::config::{RealtimeSessionModelSettings, RealtimeUserInput};
use crate::realtime::model_events::RealtimeModelToolCallEvent;

/// An event sent to a [`crate::realtime::RealtimeModel`].
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeModelSendEvent {
    /// Send a raw client event, forwarded verbatim.
    RawMessage {
        /// The full client event, including its `type`.
        message: Value,
    },
    /// Send user input and ask the model to respond.
    UserInput {
        /// The user input.
        user_input: RealtimeUserInput,
    },
    /// Append audio to the input buffer.
    Audio {
        /// Raw audio bytes in the configured input format.
        audio: Vec<u8>,
        /// Whether to commit the input buffer afterwards.
        commit: bool,
    },
    /// Send the output of a function tool call.
    ToolOutput {
        /// The call this output answers.
        tool_call: RealtimeModelToolCallEvent,
        /// The model-visible output.
        output: String,
        /// Whether the model should start a new response.
        start_response: bool,
    },
    /// Interrupt generation and audio playback.
    Interrupt {
        /// Cancel the active response even when the server interrupts automatically.
        force_response_cancel: bool,
    },
    /// Apply new session settings.
    SessionUpdate {
        /// The settings to apply.
        session_settings: Box<RealtimeSessionModelSettings>,
    },
}

impl RealtimeModelSendEvent {
    /// Build a session update event.
    pub fn session_update(settings: RealtimeSessionModelSettings) -> Self {
        RealtimeModelSendEvent::SessionUpdate {
            session_settings: Box::new(settings),
        }
    }

    /// The Python event name, useful for logging and matching.
    pub fn event_type(&self) -> &'static str {
        match self {
            RealtimeModelSendEvent::RawMessage { .. } => "raw_message",
            RealtimeModelSendEvent::UserInput { .. } => "user_input",
            RealtimeModelSendEvent::Audio { .. } => "audio",
            RealtimeModelSendEvent::ToolOutput { .. } => "tool_output",
            RealtimeModelSendEvent::Interrupt { .. } => "interrupt",
            RealtimeModelSendEvent::SessionUpdate { .. } => "session_update",
        }
    }

    /// The tool output payload, if this is a tool output event.
    pub fn as_tool_output(&self) -> Option<(&RealtimeModelToolCallEvent, &str, bool)> {
        match self {
            RealtimeModelSendEvent::ToolOutput {
                tool_call,
                output,
                start_response,
            } => Some((tool_call, output.as_str(), *start_response)),
            _ => None,
        }
    }

    /// The session settings, if this is a session update event.
    pub fn as_session_update(&self) -> Option<&RealtimeSessionModelSettings> {
        match self {
            RealtimeModelSendEvent::SessionUpdate { session_settings } => Some(session_settings),
            _ => None,
        }
    }
}
