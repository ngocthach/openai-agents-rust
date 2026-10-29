//! Events emitted by a realtime session. Mirrors `agents.realtime.events`.

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use crate::guardrail::OutputGuardrailResult;
use crate::realtime::agent::RealtimeAgent;
use crate::realtime::items::RealtimeItem;
use crate::realtime::model_events::{RealtimeModelAudioEvent, RealtimeModelEvent};
use crate::run_context::RunContext;
use crate::tool::{FunctionTool, ToolOutput};

/// Information attached to every session event.
pub struct RealtimeEventInfo<C> {
    /// The run context of the session.
    pub context: RunContext<C>,
}

impl<C> Clone for RealtimeEventInfo<C> {
    fn clone(&self) -> Self {
        RealtimeEventInfo {
            context: self.context.clone(),
        }
    }
}

impl<C> fmt::Debug for RealtimeEventInfo<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeEventInfo")
            .field("context", &self.context)
            .finish()
    }
}

/// An event emitted by a [`crate::realtime::RealtimeSession`].
pub enum RealtimeSessionEvent<C> {
    /// A new agent started a turn.
    AgentStart {
        /// The agent that started.
        agent: Arc<RealtimeAgent<C>>,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// An agent finished a turn.
    AgentEnd {
        /// The agent that finished.
        agent: Arc<RealtimeAgent<C>>,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// An agent handed off to another agent.
    Handoff {
        /// The agent that handed off.
        from_agent: Arc<RealtimeAgent<C>>,
        /// The agent that took over.
        to_agent: Arc<RealtimeAgent<C>>,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// A tool call is starting.
    ToolStart {
        /// The agent calling the tool.
        agent: Arc<RealtimeAgent<C>>,
        /// The tool being called.
        tool: Arc<FunctionTool<C>>,
        /// JSON-encoded arguments.
        arguments: String,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// A tool call finished.
    ToolEnd {
        /// The agent that called the tool.
        agent: Arc<RealtimeAgent<C>>,
        /// The tool that was called.
        tool: Arc<FunctionTool<C>>,
        /// JSON-encoded arguments.
        arguments: String,
        /// The output of the call.
        output: ToolOutput,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// A tool call needs human approval before it runs.
    ToolApprovalRequired {
        /// The agent requesting approval.
        agent: Arc<RealtimeAgent<C>>,
        /// The tool awaiting approval.
        tool: Arc<FunctionTool<C>>,
        /// The call id to pass to `approve_tool_call` or `reject_tool_call`.
        call_id: String,
        /// JSON-encoded arguments.
        arguments: String,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// A raw event from the transport.
    RawModelEvent {
        /// The transport event.
        data: RealtimeModelEvent,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// New audio to play.
    Audio {
        /// The audio event from the transport.
        audio: Box<RealtimeModelAudioEvent>,
        /// The item containing the audio.
        item_id: String,
        /// Index of the audio content inside the item.
        content_index: usize,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// The agent stopped generating audio for an item.
    AudioEnd {
        /// The item containing the audio.
        item_id: String,
        /// Index of the audio content inside the item.
        content_index: usize,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// Audio playback was interrupted; stop playing buffered audio.
    AudioInterrupted {
        /// The item containing the audio.
        item_id: String,
        /// Index of the audio content inside the item.
        content_index: usize,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// An error occurred.
    Error {
        /// The error payload.
        error: Value,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// The history changed; carries the full history.
    HistoryUpdated {
        /// The full session history.
        history: Vec<RealtimeItem>,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// A new item was appended to the history.
    HistoryAdded {
        /// The new item.
        item: Box<RealtimeItem>,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// An output guardrail tripped and the response was interrupted.
    GuardrailTripped {
        /// Results of every guardrail that triggered.
        guardrail_results: Vec<OutputGuardrailResult>,
        /// The transcript that triggered the guardrails.
        message: String,
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
    /// The model detected a period of silence from the user.
    InputAudioTimeoutTriggered {
        /// Common event info.
        info: RealtimeEventInfo<C>,
    },
}

impl<C> RealtimeSessionEvent<C> {
    /// The Python event `type` string.
    pub fn event_type(&self) -> &'static str {
        match self {
            RealtimeSessionEvent::AgentStart { .. } => "agent_start",
            RealtimeSessionEvent::AgentEnd { .. } => "agent_end",
            RealtimeSessionEvent::Handoff { .. } => "handoff",
            RealtimeSessionEvent::ToolStart { .. } => "tool_start",
            RealtimeSessionEvent::ToolEnd { .. } => "tool_end",
            RealtimeSessionEvent::ToolApprovalRequired { .. } => "tool_approval_required",
            RealtimeSessionEvent::RawModelEvent { .. } => "raw_model_event",
            RealtimeSessionEvent::Audio { .. } => "audio",
            RealtimeSessionEvent::AudioEnd { .. } => "audio_end",
            RealtimeSessionEvent::AudioInterrupted { .. } => "audio_interrupted",
            RealtimeSessionEvent::Error { .. } => "error",
            RealtimeSessionEvent::HistoryUpdated { .. } => "history_updated",
            RealtimeSessionEvent::HistoryAdded { .. } => "history_added",
            RealtimeSessionEvent::GuardrailTripped { .. } => "guardrail_tripped",
            RealtimeSessionEvent::InputAudioTimeoutTriggered { .. } => {
                "input_audio_timeout_triggered"
            }
        }
    }

    /// Common info carried by every event.
    pub fn info(&self) -> &RealtimeEventInfo<C> {
        match self {
            RealtimeSessionEvent::AgentStart { info, .. }
            | RealtimeSessionEvent::AgentEnd { info, .. }
            | RealtimeSessionEvent::Handoff { info, .. }
            | RealtimeSessionEvent::ToolStart { info, .. }
            | RealtimeSessionEvent::ToolEnd { info, .. }
            | RealtimeSessionEvent::ToolApprovalRequired { info, .. }
            | RealtimeSessionEvent::RawModelEvent { info, .. }
            | RealtimeSessionEvent::Audio { info, .. }
            | RealtimeSessionEvent::AudioEnd { info, .. }
            | RealtimeSessionEvent::AudioInterrupted { info, .. }
            | RealtimeSessionEvent::Error { info, .. }
            | RealtimeSessionEvent::HistoryUpdated { info, .. }
            | RealtimeSessionEvent::HistoryAdded { info, .. }
            | RealtimeSessionEvent::GuardrailTripped { info, .. }
            | RealtimeSessionEvent::InputAudioTimeoutTriggered { info } => info,
        }
    }
}

impl<C> Clone for RealtimeSessionEvent<C> {
    fn clone(&self) -> Self {
        match self {
            RealtimeSessionEvent::AgentStart { agent, info } => RealtimeSessionEvent::AgentStart {
                agent: agent.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::AgentEnd { agent, info } => RealtimeSessionEvent::AgentEnd {
                agent: agent.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::Handoff {
                from_agent,
                to_agent,
                info,
            } => RealtimeSessionEvent::Handoff {
                from_agent: from_agent.clone(),
                to_agent: to_agent.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::ToolStart {
                agent,
                tool,
                arguments,
                info,
            } => RealtimeSessionEvent::ToolStart {
                agent: agent.clone(),
                tool: tool.clone(),
                arguments: arguments.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::ToolEnd {
                agent,
                tool,
                arguments,
                output,
                info,
            } => RealtimeSessionEvent::ToolEnd {
                agent: agent.clone(),
                tool: tool.clone(),
                arguments: arguments.clone(),
                output: output.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::ToolApprovalRequired {
                agent,
                tool,
                call_id,
                arguments,
                info,
            } => RealtimeSessionEvent::ToolApprovalRequired {
                agent: agent.clone(),
                tool: tool.clone(),
                call_id: call_id.clone(),
                arguments: arguments.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::RawModelEvent { data, info } => {
                RealtimeSessionEvent::RawModelEvent {
                    data: data.clone(),
                    info: info.clone(),
                }
            }
            RealtimeSessionEvent::Audio {
                audio,
                item_id,
                content_index,
                info,
            } => RealtimeSessionEvent::Audio {
                audio: audio.clone(),
                item_id: item_id.clone(),
                content_index: *content_index,
                info: info.clone(),
            },
            RealtimeSessionEvent::AudioEnd {
                item_id,
                content_index,
                info,
            } => RealtimeSessionEvent::AudioEnd {
                item_id: item_id.clone(),
                content_index: *content_index,
                info: info.clone(),
            },
            RealtimeSessionEvent::AudioInterrupted {
                item_id,
                content_index,
                info,
            } => RealtimeSessionEvent::AudioInterrupted {
                item_id: item_id.clone(),
                content_index: *content_index,
                info: info.clone(),
            },
            RealtimeSessionEvent::Error { error, info } => RealtimeSessionEvent::Error {
                error: error.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::HistoryUpdated { history, info } => {
                RealtimeSessionEvent::HistoryUpdated {
                    history: history.clone(),
                    info: info.clone(),
                }
            }
            RealtimeSessionEvent::HistoryAdded { item, info } => {
                RealtimeSessionEvent::HistoryAdded {
                    item: item.clone(),
                    info: info.clone(),
                }
            }
            RealtimeSessionEvent::GuardrailTripped {
                guardrail_results,
                message,
                info,
            } => RealtimeSessionEvent::GuardrailTripped {
                guardrail_results: guardrail_results.clone(),
                message: message.clone(),
                info: info.clone(),
            },
            RealtimeSessionEvent::InputAudioTimeoutTriggered { info } => {
                RealtimeSessionEvent::InputAudioTimeoutTriggered { info: info.clone() }
            }
        }
    }
}

impl<C> fmt::Debug for RealtimeSessionEvent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("RealtimeSessionEvent");
        debug.field("type", &self.event_type());
        match self {
            RealtimeSessionEvent::AgentStart { agent, .. }
            | RealtimeSessionEvent::AgentEnd { agent, .. } => {
                debug.field("agent", &agent.name);
            }
            RealtimeSessionEvent::Handoff {
                from_agent,
                to_agent,
                ..
            } => {
                debug.field("from_agent", &from_agent.name);
                debug.field("to_agent", &to_agent.name);
            }
            RealtimeSessionEvent::ToolStart {
                tool, arguments, ..
            } => {
                debug.field("tool", &tool.name);
                debug.field("arguments", arguments);
            }
            RealtimeSessionEvent::ToolEnd {
                tool,
                arguments,
                output,
                ..
            } => {
                debug.field("tool", &tool.name);
                debug.field("arguments", arguments);
                debug.field("output", output);
            }
            RealtimeSessionEvent::ToolApprovalRequired { tool, call_id, .. } => {
                debug.field("tool", &tool.name);
                debug.field("call_id", call_id);
            }
            RealtimeSessionEvent::RawModelEvent { data, .. } => {
                debug.field("data", data);
            }
            RealtimeSessionEvent::Audio {
                item_id,
                content_index,
                ..
            }
            | RealtimeSessionEvent::AudioEnd {
                item_id,
                content_index,
                ..
            }
            | RealtimeSessionEvent::AudioInterrupted {
                item_id,
                content_index,
                ..
            } => {
                debug.field("item_id", item_id);
                debug.field("content_index", content_index);
            }
            RealtimeSessionEvent::Error { error, .. } => {
                debug.field("error", error);
            }
            RealtimeSessionEvent::HistoryUpdated { history, .. } => {
                debug.field("history_len", &history.len());
            }
            RealtimeSessionEvent::HistoryAdded { item, .. } => {
                debug.field("item", item);
            }
            RealtimeSessionEvent::GuardrailTripped {
                guardrail_results,
                message,
                ..
            } => {
                debug.field("guardrail_results", guardrail_results);
                debug.field("message", message);
            }
            RealtimeSessionEvent::InputAudioTimeoutTriggered { .. } => {}
        }
        debug.finish_non_exhaustive()
    }
}
