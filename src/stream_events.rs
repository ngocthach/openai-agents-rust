//! Events emitted by streamed runs. Mirrors `agents.stream_events`.

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::items::RunItem;

/// Semantic name of a run item event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunItemStreamEventName {
    MessageOutputCreated,
    HandoffRequested,
    /// Spelled `handoff_occured` on the wire for backward compatibility.
    HandoffOccured,
    ToolCalled,
    ToolOutput,
    ReasoningItemCreated,
    MCPApprovalRequested,
    MCPApprovalResponse,
    MCPListTools,
    ToolSearchCalled,
    ToolSearchOutputCreated,
}

impl RunItemStreamEventName {
    /// The Python SDK name of the event.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MessageOutputCreated => "message_output_created",
            Self::HandoffRequested => "handoff_requested",
            Self::HandoffOccured => "handoff_occured",
            Self::ToolCalled => "tool_called",
            Self::ToolOutput => "tool_output",
            Self::ReasoningItemCreated => "reasoning_item_created",
            Self::MCPApprovalRequested => "mcp_approval_requested",
            Self::MCPApprovalResponse => "mcp_approval_response",
            Self::MCPListTools => "mcp_list_tools",
            Self::ToolSearchCalled => "tool_search_called",
            Self::ToolSearchOutputCreated => "tool_search_output_created",
        }
    }
}

/// An event in a streamed run.
#[allow(clippy::large_enum_variant)]
pub enum StreamEvent<C> {
    /// A raw Responses API stream event (`response.output_text.delta`, ...).
    RawResponse(Value),
    /// A run item was fully generated.
    RunItem {
        name: RunItemStreamEventName,
        item: RunItem,
    },
    /// The current agent changed.
    AgentUpdated { new_agent: Arc<Agent<C>> },
}

impl<C> Clone for StreamEvent<C> {
    fn clone(&self) -> Self {
        match self {
            StreamEvent::RawResponse(v) => StreamEvent::RawResponse(v.clone()),
            StreamEvent::RunItem { name, item } => StreamEvent::RunItem {
                name: *name,
                item: item.clone(),
            },
            StreamEvent::AgentUpdated { new_agent } => StreamEvent::AgentUpdated {
                new_agent: new_agent.clone(),
            },
        }
    }
}

impl<C> fmt::Debug for StreamEvent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamEvent::RawResponse(v) => f.debug_tuple("RawResponse").field(v).finish(),
            StreamEvent::RunItem { name, item } => f
                .debug_struct("RunItem")
                .field("name", name)
                .field("item", item)
                .finish(),
            StreamEvent::AgentUpdated { new_agent } => f
                .debug_struct("AgentUpdated")
                .field("new_agent", &new_agent.name)
                .finish(),
        }
    }
}

impl<C> StreamEvent<C> {
    /// The Python SDK event type (`raw_response_event`, `run_item_stream_event`,
    /// `agent_updated_stream_event`).
    pub fn event_type(&self) -> &'static str {
        match self {
            StreamEvent::RawResponse(_) => "raw_response_event",
            StreamEvent::RunItem { .. } => "run_item_stream_event",
            StreamEvent::AgentUpdated { .. } => "agent_updated_stream_event",
        }
    }

    /// The text delta, if this is a `response.output_text.delta` raw event.
    pub fn text_delta(&self) -> Option<&str> {
        match self {
            StreamEvent::RawResponse(v)
                if v.get("type").and_then(Value::as_str) == Some("response.output_text.delta") =>
            {
                v.get("delta").and_then(Value::as_str)
            }
            _ => None,
        }
    }
}
