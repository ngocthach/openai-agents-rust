//! Items emitted inside `item.*` events of the Codex CLI JSONL stream. Mirrors
//! `agents.extensions.experimental.codex.items`.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::error::{AgentsError, Result};

/// A single file change inside a [`ThreadItem::FileChange`] item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileUpdateChange {
    /// Path of the changed file.
    pub path: String,
    /// `add`, `delete` or `update`.
    pub kind: String,
}

/// Successful result of an MCP tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolCallResult {
    /// MCP content blocks returned by the tool.
    #[serde(default)]
    pub content: Vec<Value>,
    /// Structured content returned by the tool, if any.
    #[serde(default)]
    pub structured_content: Value,
}

/// Failure of an MCP tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolCallError {
    /// Error message.
    pub message: String,
}

/// One entry of a [`ThreadItem::TodoList`] item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// Task text.
    pub text: String,
    /// Whether the task is done.
    pub completed: bool,
}

/// An item produced by Codex during a turn.
///
/// Unrecognized item types are preserved as [`ThreadItem::Unknown`] so a newer Codex CLI never
/// breaks the stream.
#[derive(Debug, Clone, PartialEq)]
pub enum ThreadItem {
    /// A message from the Codex agent.
    AgentMessage {
        /// Item id.
        id: String,
        /// Message text.
        text: String,
    },
    /// Reasoning summary text.
    Reasoning {
        /// Item id.
        id: String,
        /// Reasoning text.
        text: String,
    },
    /// A shell command run by Codex.
    CommandExecution {
        /// Item id.
        id: String,
        /// The command line.
        command: String,
        /// `in_progress`, `completed` or `failed`.
        status: String,
        /// Combined stdout and stderr collected so far.
        aggregated_output: String,
        /// Process exit code once finished.
        exit_code: Option<i64>,
    },
    /// A patch applied to the workspace.
    FileChange {
        /// Item id.
        id: String,
        /// The individual file changes.
        changes: Vec<FileUpdateChange>,
        /// `completed` or `failed`.
        status: String,
    },
    /// A call to an MCP tool.
    McpToolCall {
        /// Item id.
        id: String,
        /// MCP server name.
        server: String,
        /// Tool name.
        tool: String,
        /// Tool arguments.
        arguments: Value,
        /// `in_progress`, `completed` or `failed`.
        status: String,
        /// Result once the call succeeded.
        result: Option<McpToolCallResult>,
        /// Error once the call failed.
        error: Option<McpToolCallError>,
    },
    /// A web search performed by Codex.
    WebSearch {
        /// Item id.
        id: String,
        /// The search query.
        query: String,
    },
    /// A todo list maintained by Codex.
    TodoList {
        /// Item id.
        id: String,
        /// The todo entries.
        items: Vec<TodoItem>,
    },
    /// An error item.
    Error {
        /// Item id.
        id: String,
        /// Error message.
        message: String,
    },
    /// An item type this SDK does not know about.
    Unknown {
        /// The raw `type` field, or `unknown` when absent.
        item_type: String,
        /// The raw payload.
        payload: Value,
        /// The raw `id` field, when present.
        id: Option<String>,
    },
}

fn require_str(raw: &Map<String, Value>, key: &str) -> Result<String> {
    raw.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| AgentsError::other(format!("Codex item is missing the \"{key}\" field.")))
}

fn optional_str(raw: &Map<String, Value>, key: &str) -> String {
    raw.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

impl ThreadItem {
    /// Parse an item payload, falling back to [`ThreadItem::Unknown`] for unknown types.
    pub fn from_value(raw: Value) -> Result<Self> {
        let Value::Object(map) = &raw else {
            return Err(AgentsError::other("Codex item payload must be an object."));
        };
        let item_type = map.get("type").and_then(Value::as_str).unwrap_or_default();
        let item = match item_type {
            "agent_message" => ThreadItem::AgentMessage {
                id: require_str(map, "id")?,
                text: optional_str(map, "text"),
            },
            "reasoning" => ThreadItem::Reasoning {
                id: require_str(map, "id")?,
                text: optional_str(map, "text"),
            },
            "command_execution" => ThreadItem::CommandExecution {
                id: require_str(map, "id")?,
                command: require_str(map, "command")?,
                status: require_str(map, "status")?,
                aggregated_output: optional_str(map, "aggregated_output"),
                exit_code: map.get("exit_code").and_then(Value::as_i64),
            },
            "file_change" => {
                let changes = match map.get("changes") {
                    Some(Value::Array(items)) => items
                        .iter()
                        .map(|c| {
                            serde_json::from_value::<FileUpdateChange>(c.clone())
                                .map_err(AgentsError::from)
                        })
                        .collect::<Result<Vec<_>>>()?,
                    _ => Vec::new(),
                };
                ThreadItem::FileChange {
                    id: require_str(map, "id")?,
                    changes,
                    status: require_str(map, "status")?,
                }
            }
            "mcp_tool_call" => ThreadItem::McpToolCall {
                id: require_str(map, "id")?,
                server: require_str(map, "server")?,
                tool: require_str(map, "tool")?,
                arguments: map.get("arguments").cloned().unwrap_or(Value::Null),
                status: require_str(map, "status")?,
                result: match map.get("result") {
                    Some(v) if !v.is_null() => Some(serde_json::from_value(v.clone())?),
                    _ => None,
                },
                error: match map.get("error") {
                    Some(v) if !v.is_null() => Some(serde_json::from_value(v.clone())?),
                    _ => None,
                },
            },
            "web_search" => ThreadItem::WebSearch {
                id: require_str(map, "id")?,
                query: optional_str(map, "query"),
            },
            "todo_list" => {
                let items = match map.get("items") {
                    Some(Value::Array(entries)) => entries
                        .iter()
                        .map(|e| TodoItem {
                            text: e
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            completed: e.get("completed").and_then(Value::as_bool).unwrap_or(false),
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                ThreadItem::TodoList {
                    id: require_str(map, "id")?,
                    items,
                }
            }
            "error" => ThreadItem::Error {
                id: optional_str(map, "id"),
                message: optional_str(map, "message"),
            },
            other => ThreadItem::Unknown {
                item_type: if other.is_empty() {
                    "unknown".to_string()
                } else {
                    other.to_string()
                },
                id: map.get("id").and_then(Value::as_str).map(str::to_string),
                payload: raw.clone(),
            },
        };
        Ok(item)
    }

    /// The raw JSON form of the item.
    pub fn to_value(&self) -> Value {
        match self {
            ThreadItem::AgentMessage { id, text } => {
                json!({"type": "agent_message", "id": id, "text": text})
            }
            ThreadItem::Reasoning { id, text } => {
                json!({"type": "reasoning", "id": id, "text": text})
            }
            ThreadItem::CommandExecution {
                id,
                command,
                status,
                aggregated_output,
                exit_code,
            } => json!({
                "type": "command_execution",
                "id": id,
                "command": command,
                "status": status,
                "aggregated_output": aggregated_output,
                "exit_code": exit_code,
            }),
            ThreadItem::FileChange {
                id,
                changes,
                status,
            } => json!({"type": "file_change", "id": id, "changes": changes, "status": status}),
            ThreadItem::McpToolCall {
                id,
                server,
                tool,
                arguments,
                status,
                result,
                error,
            } => json!({
                "type": "mcp_tool_call",
                "id": id,
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "status": status,
                "result": result,
                "error": error,
            }),
            ThreadItem::WebSearch { id, query } => {
                json!({"type": "web_search", "id": id, "query": query})
            }
            ThreadItem::TodoList { id, items } => {
                json!({"type": "todo_list", "id": id, "items": items})
            }
            ThreadItem::Error { id, message } => {
                json!({"type": "error", "id": id, "message": message})
            }
            ThreadItem::Unknown { payload, .. } => payload.clone(),
        }
    }

    /// The item id, when the item has one.
    pub fn id(&self) -> Option<&str> {
        match self {
            ThreadItem::AgentMessage { id, .. }
            | ThreadItem::Reasoning { id, .. }
            | ThreadItem::CommandExecution { id, .. }
            | ThreadItem::FileChange { id, .. }
            | ThreadItem::McpToolCall { id, .. }
            | ThreadItem::WebSearch { id, .. }
            | ThreadItem::TodoList { id, .. }
            | ThreadItem::Error { id, .. } => Some(id.as_str()),
            ThreadItem::Unknown { id, .. } => id.as_deref(),
        }
    }

    /// The item `type` as reported by the Codex CLI.
    pub fn item_type(&self) -> &str {
        match self {
            ThreadItem::AgentMessage { .. } => "agent_message",
            ThreadItem::Reasoning { .. } => "reasoning",
            ThreadItem::CommandExecution { .. } => "command_execution",
            ThreadItem::FileChange { .. } => "file_change",
            ThreadItem::McpToolCall { .. } => "mcp_tool_call",
            ThreadItem::WebSearch { .. } => "web_search",
            ThreadItem::TodoList { .. } => "todo_list",
            ThreadItem::Error { .. } => "error",
            ThreadItem::Unknown { item_type, .. } => item_type.as_str(),
        }
    }

    /// The text of an agent message item.
    pub fn as_agent_message(&self) -> Option<&str> {
        match self {
            ThreadItem::AgentMessage { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }
}

impl Serialize for ThreadItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ThreadItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = Value::deserialize(deserializer)?;
        ThreadItem::from_value(raw).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_items() {
        let item = ThreadItem::from_value(json!({
            "type": "command_execution",
            "id": "cmd-1",
            "command": "pytest",
            "status": "completed",
            "aggregated_output": "All good",
            "exit_code": 0,
        }))
        .unwrap();
        assert_eq!(item.id(), Some("cmd-1"));
        match item {
            ThreadItem::CommandExecution {
                command,
                status,
                aggregated_output,
                exit_code,
                ..
            } => {
                assert_eq!(command, "pytest");
                assert_eq!(status, "completed");
                assert_eq!(aggregated_output, "All good");
                assert_eq!(exit_code, Some(0));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn defaults_optional_item_fields() {
        let item = ThreadItem::from_value(json!({"type": "agent_message", "id": "a1"})).unwrap();
        assert_eq!(item.as_agent_message(), Some(""));
        let item = ThreadItem::from_value(json!({"type": "error", "message": "boom"})).unwrap();
        assert_eq!(item.id(), Some(""));
    }

    #[test]
    fn unknown_items_keep_payload() {
        let item = ThreadItem::from_value(json!({"type": "brand_new", "id": "x"})).unwrap();
        assert_eq!(item.item_type(), "brand_new");
        assert_eq!(item.id(), Some("x"));
        assert_eq!(item.to_value()["type"], "brand_new");
        let item = ThreadItem::from_value(json!({})).unwrap();
        assert_eq!(item.item_type(), "unknown");
        assert_eq!(item.id(), None);
    }

    #[test]
    fn missing_required_field_is_an_error() {
        assert!(ThreadItem::from_value(json!({"type": "agent_message"})).is_err());
    }

    #[test]
    fn round_trips_through_serde() {
        let raw = json!({
            "type": "todo_list",
            "id": "todo-1",
            "items": [{"text": "write tests", "completed": true}],
        });
        let item = ThreadItem::from_value(raw.clone()).unwrap();
        assert_eq!(serde_json::to_value(&item).unwrap(), raw);
        let back: ThreadItem = serde_json::from_value(raw).unwrap();
        assert_eq!(back, item);
    }
}
