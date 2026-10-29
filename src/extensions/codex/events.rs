//! Events emitted by the Codex CLI JSONL stream. Mirrors
//! `agents.extensions.experimental.codex.events`.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::extensions::codex::items::ThreadItem;

/// Token usage reported when a turn completes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens consumed by the turn.
    pub input_tokens: u64,
    /// Input tokens served from the prompt cache.
    pub cached_input_tokens: u64,
    /// Output tokens produced by the turn.
    pub output_tokens: u64,
}

impl Usage {
    /// Convert to the SDK-wide usage type, counting the turn as one request.
    pub fn to_agent_usage(self) -> crate::usage::Usage {
        crate::usage::Usage {
            requests: 1,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            total_tokens: self.input_tokens + self.output_tokens,
            input_tokens_details: crate::usage::InputTokensDetails {
                cached_tokens: self.cached_input_tokens,
                cache_write_tokens: 0,
            },
            output_tokens_details: crate::usage::OutputTokensDetails::default(),
            request_usage_entries: Vec::new(),
        }
    }
}

/// An error reported by a failed turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadError {
    /// Error message.
    #[serde(default)]
    pub message: String,
}

/// An event of the Codex CLI JSONL stream.
///
/// Unrecognized event types are preserved as [`ThreadEvent::Unknown`].
#[derive(Debug, Clone, PartialEq)]
pub enum ThreadEvent {
    /// The thread started; carries the id used to resume it later.
    ThreadStarted {
        /// Codex thread id.
        thread_id: String,
    },
    /// A turn started.
    TurnStarted,
    /// A turn completed successfully.
    TurnCompleted {
        /// Token usage of the turn, when reported.
        usage: Option<Usage>,
    },
    /// A turn failed.
    TurnFailed {
        /// The failure.
        error: ThreadError,
    },
    /// An item started.
    ItemStarted {
        /// The item.
        item: ThreadItem,
    },
    /// An item was updated.
    ItemUpdated {
        /// The item.
        item: ThreadItem,
    },
    /// An item completed.
    ItemCompleted {
        /// The item.
        item: ThreadItem,
    },
    /// A stream-level error.
    Error {
        /// Error message.
        message: String,
    },
    /// An event type this SDK does not know about.
    Unknown {
        /// The raw `type` field, or `unknown` when absent.
        event_type: String,
        /// The raw payload.
        payload: Value,
    },
}

fn event_item(raw: &Value) -> Result<ThreadItem> {
    match raw.get("item") {
        Some(item) if !item.is_null() => ThreadItem::from_value(item.clone()),
        _ => ThreadItem::from_value(json!({"type": "unknown"})),
    }
}

impl ThreadEvent {
    /// Parse an event payload, falling back to [`ThreadEvent::Unknown`] for unknown types.
    pub fn from_value(raw: Value) -> Result<Self> {
        if !raw.is_object() {
            return Err(AgentsError::other("Codex event payload must be an object."));
        }
        let event_type = raw.get("type").and_then(Value::as_str).unwrap_or_default();
        let event = match event_type {
            "thread.started" => ThreadEvent::ThreadStarted {
                thread_id: raw
                    .get("thread_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        AgentsError::other("thread.started is missing the \"thread_id\" field.")
                    })?
                    .to_string(),
            },
            "turn.started" => ThreadEvent::TurnStarted,
            "turn.completed" => ThreadEvent::TurnCompleted {
                usage: match raw.get("usage") {
                    Some(v) if !v.is_null() => Some(serde_json::from_value(v.clone())?),
                    _ => None,
                },
            },
            "turn.failed" => ThreadEvent::TurnFailed {
                error: match raw.get("error") {
                    Some(v) if !v.is_null() => serde_json::from_value(v.clone())?,
                    _ => ThreadError::default(),
                },
            },
            "item.started" => ThreadEvent::ItemStarted {
                item: event_item(&raw)?,
            },
            "item.updated" => ThreadEvent::ItemUpdated {
                item: event_item(&raw)?,
            },
            "item.completed" => ThreadEvent::ItemCompleted {
                item: event_item(&raw)?,
            },
            "error" => ThreadEvent::Error {
                message: raw
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            other => ThreadEvent::Unknown {
                event_type: if other.is_empty() {
                    "unknown".to_string()
                } else {
                    other.to_string()
                },
                payload: raw.clone(),
            },
        };
        Ok(event)
    }

    /// Parse one JSONL line of the Codex stream.
    pub fn from_json_line(line: &str) -> Result<Self> {
        let raw: Value = serde_json::from_str(line)
            .map_err(|_| AgentsError::other(format!("Failed to parse event: {line}")))?;
        ThreadEvent::from_value(raw)
            .map_err(|_| AgentsError::other(format!("Failed to parse event: {line}")))
    }

    /// The raw JSON form of the event.
    pub fn to_value(&self) -> Value {
        match self {
            ThreadEvent::ThreadStarted { thread_id } => {
                json!({"type": "thread.started", "thread_id": thread_id})
            }
            ThreadEvent::TurnStarted => json!({"type": "turn.started"}),
            ThreadEvent::TurnCompleted { usage } => {
                json!({"type": "turn.completed", "usage": usage})
            }
            ThreadEvent::TurnFailed { error } => json!({"type": "turn.failed", "error": error}),
            ThreadEvent::ItemStarted { item } => json!({"type": "item.started", "item": item}),
            ThreadEvent::ItemUpdated { item } => json!({"type": "item.updated", "item": item}),
            ThreadEvent::ItemCompleted { item } => json!({"type": "item.completed", "item": item}),
            ThreadEvent::Error { message } => json!({"type": "error", "message": message}),
            ThreadEvent::Unknown { payload, .. } => payload.clone(),
        }
    }

    /// The event `type` as reported by the Codex CLI.
    pub fn event_type(&self) -> &str {
        match self {
            ThreadEvent::ThreadStarted { .. } => "thread.started",
            ThreadEvent::TurnStarted => "turn.started",
            ThreadEvent::TurnCompleted { .. } => "turn.completed",
            ThreadEvent::TurnFailed { .. } => "turn.failed",
            ThreadEvent::ItemStarted { .. } => "item.started",
            ThreadEvent::ItemUpdated { .. } => "item.updated",
            ThreadEvent::ItemCompleted { .. } => "item.completed",
            ThreadEvent::Error { .. } => "error",
            ThreadEvent::Unknown { event_type, .. } => event_type.as_str(),
        }
    }
}

impl Serialize for ThreadEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ThreadEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = Value::deserialize(deserializer)?;
        ThreadEvent::from_value(raw).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thread_lifecycle_events() {
        let started = ThreadEvent::from_json_line(r#"{"type":"thread.started","thread_id":"t1"}"#)
            .expect("parses");
        assert_eq!(
            started,
            ThreadEvent::ThreadStarted {
                thread_id: "t1".into()
            }
        );
        let completed = ThreadEvent::from_json_line(
            r#"{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":1,"output_tokens":5}}"#,
        )
        .expect("parses");
        assert_eq!(
            completed,
            ThreadEvent::TurnCompleted {
                usage: Some(Usage {
                    input_tokens: 10,
                    cached_input_tokens: 1,
                    output_tokens: 5,
                })
            }
        );
        let failed =
            ThreadEvent::from_json_line(r#"{"type":"turn.failed","error":{"message":"boom"}}"#)
                .expect("parses");
        assert_eq!(
            failed,
            ThreadEvent::TurnFailed {
                error: ThreadError {
                    message: "boom".into()
                }
            }
        );
    }

    #[test]
    fn item_events_without_item_become_unknown_items() {
        let event = ThreadEvent::from_json_line(r#"{"type":"item.started"}"#).expect("parses");
        match event {
            ThreadEvent::ItemStarted { item } => assert_eq!(item.item_type(), "unknown"),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn unknown_events_keep_payload() {
        let event = ThreadEvent::from_json_line(r#"{"type":"turn.pondered","extra":1}"#).unwrap();
        assert_eq!(event.event_type(), "turn.pondered");
        assert_eq!(event.to_value()["extra"], 1);
    }

    #[test]
    fn invalid_lines_report_the_raw_line() {
        let err = ThreadEvent::from_json_line("{bad").unwrap_err();
        assert_eq!(err.to_string(), "Failed to parse event: {bad");
        let err = ThreadEvent::from_json_line(r#"{"type":"thread.started"}"#).unwrap_err();
        assert!(err.to_string().starts_with("Failed to parse event:"));
    }

    #[test]
    fn converts_usage_to_agent_usage() {
        let usage = Usage {
            input_tokens: 10,
            cached_input_tokens: 1,
            output_tokens: 5,
        }
        .to_agent_usage();
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.input_tokens_details.cached_tokens, 1);
    }
}
