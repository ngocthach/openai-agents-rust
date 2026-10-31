//! Events sent to a realtime transport. Mirrors `agents.realtime.model_inputs`.

use serde_json::Value;

use crate::error::{AgentsError, Result};
use crate::realtime::config::{RealtimeSessionModelSettings, RealtimeUserInput};
use crate::realtime::model_events::RealtimeModelToolCallEvent;

/// How an interruption should stop generation and playback.
///
/// A plain [`RealtimeInterrupt::forced`] stops everything. The response-scoped forms let the
/// session interrupt one response without disturbing a newer one, which matters when an output
/// guardrail trips on a response the model has already moved past.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RealtimeInterrupt {
    /// Cancel the active response even when the server cancels it automatically.
    pub force_response_cancel: bool,
    /// Scope cancellation and playback interruption to this response.
    pub response_id: Option<String>,
    /// Cancel `response_id` without interrupting audio playback. Requires `response_id`.
    pub cancel_response_only: bool,
    /// Interrupt playback for `response_id` without cancelling any response.
    pub playback_only: bool,
}

impl RealtimeInterrupt {
    /// Stop generation and playback, cancelling the active response.
    pub fn forced() -> Self {
        RealtimeInterrupt {
            force_response_cancel: true,
            ..Default::default()
        }
    }

    /// Stop generation and playback for one response.
    ///
    /// With `audio_output` false the audio is left alone and only the response is cancelled,
    /// which is what a guardrail tripping on text output needs.
    pub fn forced_for_response(response_id: impl Into<String>, audio_output: bool) -> Self {
        RealtimeInterrupt {
            force_response_cancel: true,
            response_id: Some(response_id.into()),
            cancel_response_only: !audio_output,
            playback_only: false,
        }
    }

    /// Stop playback of one response without cancelling anything.
    pub fn playback_only(response_id: impl Into<String>) -> Self {
        RealtimeInterrupt {
            force_response_cancel: false,
            response_id: Some(response_id.into()),
            cancel_response_only: false,
            playback_only: true,
        }
    }

    /// Cancel one response without touching audio playback.
    pub fn cancel_response_only(response_id: impl Into<String>) -> Self {
        RealtimeInterrupt {
            force_response_cancel: false,
            response_id: Some(response_id.into()),
            cancel_response_only: true,
            playback_only: false,
        }
    }

    /// Reject contradictory combinations before they reach the transport.
    pub fn validate(&self) -> Result<()> {
        if self.playback_only && (self.cancel_response_only || self.force_response_cancel) {
            return Err(AgentsError::user(
                "playback_only cannot be combined with explicit cancellation modes",
            ));
        }
        if self.cancel_response_only && self.response_id.is_none() {
            return Err(AgentsError::user(
                "cancel_response_only requires response_id",
            ));
        }
        Ok(())
    }

    /// Whether this interruption is scoped to a single response.
    pub fn is_response_scoped(&self) -> bool {
        self.response_id.is_some()
    }
}

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
    Interrupt(RealtimeInterrupt),
    /// Apply new session settings.
    SessionUpdate {
        /// The settings to apply.
        session_settings: Box<RealtimeSessionModelSettings>,
    },
}

impl RealtimeModelSendEvent {
    /// Stop generation and playback, cancelling the active response.
    pub fn interrupt() -> Self {
        RealtimeModelSendEvent::Interrupt(RealtimeInterrupt::forced())
    }

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
            RealtimeModelSendEvent::Interrupt(_) => "interrupt",
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

    /// The interruption, if this is an interrupt event.
    pub fn as_interrupt(&self) -> Option<&RealtimeInterrupt> {
        match self {
            RealtimeModelSendEvent::Interrupt(interrupt) => Some(interrupt),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupt_builders_set_consistent_modes() {
        let forced = RealtimeInterrupt::forced();
        assert!(forced.force_response_cancel);
        assert!(!forced.is_response_scoped());
        forced.validate().unwrap();

        let audio = RealtimeInterrupt::forced_for_response("resp_1", true);
        assert!(!audio.cancel_response_only);
        assert!(audio.is_response_scoped());
        audio.validate().unwrap();

        let text = RealtimeInterrupt::forced_for_response("resp_1", false);
        assert!(text.cancel_response_only);
        text.validate().unwrap();

        RealtimeInterrupt::playback_only("resp_1")
            .validate()
            .unwrap();
        RealtimeInterrupt::cancel_response_only("resp_1")
            .validate()
            .unwrap();
    }

    #[test]
    fn contradictory_interrupts_are_rejected() {
        let both = RealtimeInterrupt {
            playback_only: true,
            force_response_cancel: true,
            ..Default::default()
        };
        assert!(both.validate().is_err());

        let unscoped = RealtimeInterrupt {
            cancel_response_only: true,
            ..Default::default()
        };
        assert!(unscoped.validate().is_err());
    }
}
