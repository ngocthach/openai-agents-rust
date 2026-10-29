//! Input items, model output items, run items and helpers. Mirrors `agents.items`.
//!
//! Items follow the OpenAI Responses API wire format. Items that the SDK interprets (messages,
//! function calls, reasoning) are typed; everything else is carried as raw JSON so new item types
//! keep flowing through unchanged.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::{AgentsError, Result};
use crate::tool::{ToolOutput, ToolOutputPart};
use crate::usage::Usage;

/// An input item in the Responses API format (a message, a function call output, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputItem(pub Value);

impl InputItem {
    /// Wrap a raw Responses input item.
    pub fn from_value(value: Value) -> Self {
        InputItem(value)
    }

    /// A message with the given role and text content.
    pub fn message(role: &str, content: impl Into<String>) -> Self {
        InputItem(json!({"role": role, "content": content.into()}))
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::message("user", content)
    }

    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::message("system", content)
    }

    /// A developer message.
    pub fn developer(content: impl Into<String>) -> Self {
        Self::message("developer", content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::message("assistant", content)
    }

    /// A user message with structured content parts (`input_text`, `input_image`, `input_file`).
    pub fn user_parts(parts: Vec<Value>) -> Self {
        InputItem(json!({"role": "user", "content": parts}))
    }

    /// A `function_call_output` item.
    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        InputItem(json!({
            "type": "function_call_output",
            "call_id": call_id.into(),
            "output": output.into(),
        }))
    }

    /// The raw JSON value.
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// Consume into the raw JSON value.
    pub fn into_value(self) -> Value {
        self.0
    }

    /// The `type` field, if present.
    pub fn item_type(&self) -> Option<&str> {
        self.0.get("type").and_then(Value::as_str)
    }

    /// The `role` field, if present.
    pub fn role(&self) -> Option<&str> {
        self.0.get("role").and_then(Value::as_str)
    }

    /// True if this item is a plain message (has a role but is not a tool item).
    pub fn is_message(&self) -> bool {
        self.role().is_some() && matches!(self.item_type(), None | Some("message"))
    }

    /// The textual content of a message item, concatenating text parts.
    pub fn text_content(&self) -> Option<String> {
        match self.0.get("content") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(parts)) => {
                let mut out = String::new();
                for p in parts {
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        out.push_str(t);
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

impl From<Value> for InputItem {
    fn from(value: Value) -> Self {
        InputItem(value)
    }
}

impl From<&str> for InputItem {
    fn from(value: &str) -> Self {
        InputItem::user(value)
    }
}

impl From<String> for InputItem {
    fn from(value: String) -> Self {
        InputItem::user(value)
    }
}

/// Input accepted by the runner: a user string, a list of items or a resumable state.
#[derive(Debug, Clone)]
pub enum RunInput {
    /// Treated as a single user message.
    Text(String),
    /// A list of input items in the Responses API format.
    Items(Vec<InputItem>),
    /// A paused run to resume (see [`crate::RunState`]).
    State(Box<crate::run_state::RunState>),
}

impl RunInput {
    /// Convert to a list of input items (states return their original input).
    pub fn to_items(&self) -> Vec<InputItem> {
        match self {
            RunInput::Text(t) => vec![InputItem::user(t.clone())],
            RunInput::Items(items) => items.clone(),
            RunInput::State(state) => state.original_input.clone(),
        }
    }
}

impl From<&str> for RunInput {
    fn from(value: &str) -> Self {
        RunInput::Text(value.to_string())
    }
}

impl From<String> for RunInput {
    fn from(value: String) -> Self {
        RunInput::Text(value)
    }
}

impl From<Vec<InputItem>> for RunInput {
    fn from(value: Vec<InputItem>) -> Self {
        RunInput::Items(value)
    }
}

impl From<Vec<Value>> for RunInput {
    fn from(value: Vec<Value>) -> Self {
        RunInput::Items(value.into_iter().map(InputItem).collect())
    }
}

impl From<crate::run_state::RunState> for RunInput {
    fn from(value: crate::run_state::RunState) -> Self {
        RunInput::State(Box::new(value))
    }
}

/// A function tool call emitted by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Item id (`fc_...`), if the provider assigned one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Call id used to correlate the output.
    pub call_id: String,
    /// Tool name.
    pub name: String,
    /// JSON-encoded arguments.
    #[serde(default)]
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Any other provider fields (e.g. `namespace`, `caller`).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl FunctionCall {
    /// Build a function call with a generated call id.
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        FunctionCall {
            id: None,
            call_id: format!("call_{}", crate::util::gen_hex24()),
            name: name.into(),
            arguments: arguments.into(),
            status: None,
            extra: Map::new(),
        }
    }

    /// Serialize to a Responses `function_call` item.
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(m) = &mut v {
            m.insert("type".into(), Value::String("function_call".into()));
        }
        v
    }
}

/// A `function_call_output` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallOutput {
    pub call_id: String,
    /// Either a string or a list of content parts.
    pub output: Value,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl FunctionCallOutput {
    /// Serialize to a Responses `function_call_output` item.
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(m) = &mut v {
            m.insert("type".into(), Value::String("function_call_output".into()));
        }
        v
    }
}

/// A content part of an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Value", into = "Value")]
pub enum MessageContent {
    /// Output text with optional annotations and logprobs.
    OutputText {
        text: String,
        annotations: Vec<Value>,
        logprobs: Option<Value>,
    },
    /// A refusal.
    Refusal { refusal: String },
    /// Any other content part.
    Other(Value),
}

impl TryFrom<Value> for MessageContent {
    type Error = String;
    fn try_from(value: Value) -> std::result::Result<Self, Self::Error> {
        match value.get("type").and_then(Value::as_str) {
            Some("output_text") => Ok(MessageContent::OutputText {
                text: value
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                annotations: value
                    .get("annotations")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                logprobs: value.get("logprobs").cloned().filter(|v| !v.is_null()),
            }),
            Some("refusal") => Ok(MessageContent::Refusal {
                refusal: value
                    .get("refusal")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            }),
            _ => Ok(MessageContent::Other(value)),
        }
    }
}

impl From<MessageContent> for Value {
    fn from(value: MessageContent) -> Self {
        match value {
            MessageContent::OutputText {
                text,
                annotations,
                logprobs,
            } => {
                let mut m =
                    json!({"type": "output_text", "text": text, "annotations": annotations});
                if let Some(lp) = logprobs {
                    m["logprobs"] = lp;
                }
                m
            }
            MessageContent::Refusal { refusal } => json!({"type": "refusal", "refusal": refusal}),
            MessageContent::Other(v) => v,
        }
    }
}

/// An assistant message produced by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageOutput {
    #[serde(default)]
    pub id: String,
    #[serde(default = "assistant_role")]
    pub role: String,
    #[serde(default)]
    pub content: Vec<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn assistant_role() -> String {
    "assistant".to_string()
}

impl MessageOutput {
    /// An assistant message with a single text part.
    pub fn text(text: impl Into<String>) -> Self {
        MessageOutput {
            id: format!("msg_{}", crate::util::gen_hex24()),
            role: "assistant".into(),
            content: vec![MessageContent::OutputText {
                text: text.into(),
                annotations: Vec::new(),
                logprobs: None,
            }],
            status: Some("completed".into()),
            extra: Map::new(),
        }
    }

    /// Serialize to a Responses `message` output item.
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(m) = &mut v {
            m.insert("type".into(), Value::String("message".into()));
        }
        v
    }

    /// Concatenated text of all `output_text` parts.
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let MessageContent::OutputText { text, .. } = part {
                out.push_str(text);
            }
        }
        out
    }

    /// Concatenated refusal text, if any.
    pub fn refusal(&self) -> Option<String> {
        let mut out = String::new();
        for part in &self.content {
            if let MessageContent::Refusal { refusal } = part {
                out.push_str(refusal);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

/// A reasoning item produced by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningOutput {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub summary: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ReasoningOutput {
    /// Serialize to a Responses `reasoning` output item.
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(m) = &mut v {
            m.insert("type".into(), Value::String("reasoning".into()));
        }
        v
    }
}

/// An output item from the model in the Responses API format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Value", into = "Value")]
pub enum OutputItem {
    /// An assistant message.
    Message(MessageOutput),
    /// A function tool call.
    FunctionCall(FunctionCall),
    /// A reasoning item.
    Reasoning(ReasoningOutput),
    /// Any other item (hosted tool calls, MCP items, ...), kept verbatim.
    Other(Value),
}

impl OutputItem {
    /// Parse a raw Responses output item.
    pub fn from_value(value: Value) -> Self {
        let parse = |value: &Value| -> Option<OutputItem> {
            let mut without_type = value.clone();
            if let Value::Object(m) = &mut without_type {
                m.remove("type");
            }
            match value.get("type").and_then(Value::as_str) {
                Some("message")
                    if value.get("role").and_then(Value::as_str) == Some("assistant") =>
                {
                    serde_json::from_value(without_type)
                        .ok()
                        .map(OutputItem::Message)
                }
                Some("function_call") => serde_json::from_value(without_type)
                    .ok()
                    .map(OutputItem::FunctionCall),
                Some("reasoning") => serde_json::from_value(without_type)
                    .ok()
                    .map(OutputItem::Reasoning),
                _ => None,
            }
        };
        parse(&value).unwrap_or(OutputItem::Other(value))
    }

    /// Serialize to the Responses wire format.
    pub fn to_value(&self) -> Value {
        match self {
            OutputItem::Message(m) => m.to_value(),
            OutputItem::FunctionCall(f) => f.to_value(),
            OutputItem::Reasoning(r) => r.to_value(),
            OutputItem::Other(v) => v.clone(),
        }
    }

    /// The `type` of the item.
    pub fn item_type(&self) -> String {
        match self {
            OutputItem::Message(_) => "message".into(),
            OutputItem::FunctionCall(_) => "function_call".into(),
            OutputItem::Reasoning(_) => "reasoning".into(),
            OutputItem::Other(v) => v
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        }
    }

    /// The item id, if any.
    pub fn id(&self) -> Option<&str> {
        match self {
            OutputItem::Message(m) => Some(m.id.as_str()),
            OutputItem::FunctionCall(f) => f.id.as_deref(),
            OutputItem::Reasoning(r) => Some(r.id.as_str()),
            OutputItem::Other(v) => v.get("id").and_then(Value::as_str),
        }
    }
}

impl TryFrom<Value> for OutputItem {
    type Error = String;
    fn try_from(value: Value) -> std::result::Result<Self, Self::Error> {
        Ok(OutputItem::from_value(value))
    }
}

impl From<OutputItem> for Value {
    fn from(value: OutputItem) -> Self {
        value.to_value()
    }
}

/// A tool call that paused the run waiting for approval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolApprovalItem {
    /// Name of the agent that requested the tool call.
    pub agent: String,
    /// The raw tool call item.
    pub raw_item: FunctionCall,
    /// Tool name (may differ from `raw_item.name` for MCP tools).
    pub tool_name: String,
}

impl ToolApprovalItem {
    /// The tool name.
    pub fn name(&self) -> &str {
        &self.tool_name
    }
    /// The call id.
    pub fn call_id(&self) -> &str {
        &self.raw_item.call_id
    }
    /// The JSON arguments.
    pub fn arguments(&self) -> &str {
        &self.raw_item.arguments
    }
}

/// The kind of a [`RunItem`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunItemKind {
    /// Input admitted from `RunState::add_input` before a resumed model call.
    #[serde(rename = "input_item")]
    Input { raw_item: InputItem },
    /// An assistant message.
    #[serde(rename = "message_output_item")]
    MessageOutput { raw_item: MessageOutput },
    /// A reasoning item.
    #[serde(rename = "reasoning_item")]
    Reasoning { raw_item: ReasoningOutput },
    /// A tool call (function or hosted).
    #[serde(rename = "tool_call_item")]
    ToolCall { raw_item: Value },
    /// A tool call output. `raw_item` is the wire item; `output` is the tool's return value.
    #[serde(rename = "tool_call_output_item")]
    ToolCallOutput { raw_item: Value, output: Value },
    /// A tool call that paused for approval.
    #[serde(rename = "tool_approval_item")]
    ToolApproval {
        raw_item: FunctionCall,
        tool_name: String,
    },
    /// A handoff request from the model.
    #[serde(rename = "handoff_call_item")]
    HandoffCall { raw_item: FunctionCall },
    /// A completed handoff.
    #[serde(rename = "handoff_output_item")]
    HandoffOutput {
        raw_item: InputItem,
        source_agent: String,
        target_agent: String,
    },
    /// A hosted MCP tool listing.
    #[serde(rename = "mcp_list_tools_item")]
    MCPListTools { raw_item: Value },
    /// A hosted MCP approval request.
    #[serde(rename = "mcp_approval_request_item")]
    MCPApprovalRequest { raw_item: Value },
    /// A hosted MCP approval response.
    #[serde(rename = "mcp_approval_response_item")]
    MCPApprovalResponse { raw_item: Value },
}

/// An item generated during a run, tagged with the agent that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunItem {
    /// Name of the agent associated with the item.
    pub agent: String,
    /// The item.
    #[serde(flatten)]
    pub kind: RunItemKind,
}

impl RunItem {
    pub fn new(agent: impl Into<String>, kind: RunItemKind) -> Self {
        RunItem {
            agent: agent.into(),
            kind,
        }
    }

    /// The item type string (`message_output_item`, `tool_call_item`, ...).
    pub fn item_type(&self) -> &'static str {
        match &self.kind {
            RunItemKind::Input { .. } => "input_item",
            RunItemKind::MessageOutput { .. } => "message_output_item",
            RunItemKind::Reasoning { .. } => "reasoning_item",
            RunItemKind::ToolCall { .. } => "tool_call_item",
            RunItemKind::ToolCallOutput { .. } => "tool_call_output_item",
            RunItemKind::ToolApproval { .. } => "tool_approval_item",
            RunItemKind::HandoffCall { .. } => "handoff_call_item",
            RunItemKind::HandoffOutput { .. } => "handoff_output_item",
            RunItemKind::MCPListTools { .. } => "mcp_list_tools_item",
            RunItemKind::MCPApprovalRequest { .. } => "mcp_approval_request_item",
            RunItemKind::MCPApprovalResponse { .. } => "mcp_approval_response_item",
        }
    }

    /// Convert to an input item for the next model call.
    pub fn to_input_item(&self) -> InputItem {
        match &self.kind {
            RunItemKind::Input { raw_item } => raw_item.clone(),
            RunItemKind::MessageOutput { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::Reasoning { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::ToolCall { raw_item } => InputItem(strip_created_by(raw_item.clone())),
            RunItemKind::ToolCallOutput { raw_item, .. } => {
                InputItem(strip_created_by(raw_item.clone()))
            }
            RunItemKind::ToolApproval { raw_item, .. } => InputItem(raw_item.to_value()),
            RunItemKind::HandoffCall { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::HandoffOutput { raw_item, .. } => raw_item.clone(),
            RunItemKind::MCPListTools { raw_item }
            | RunItemKind::MCPApprovalRequest { raw_item }
            | RunItemKind::MCPApprovalResponse { raw_item } => InputItem(raw_item.clone()),
        }
    }

    /// The message, if this is a message output item.
    pub fn as_message(&self) -> Option<&MessageOutput> {
        match &self.kind {
            RunItemKind::MessageOutput { raw_item } => Some(raw_item),
            _ => None,
        }
    }

    /// The tool approval item, if this is one.
    pub fn as_tool_approval(&self) -> Option<ToolApprovalItem> {
        match &self.kind {
            RunItemKind::ToolApproval {
                raw_item,
                tool_name,
            } => Some(ToolApprovalItem {
                agent: self.agent.clone(),
                raw_item: raw_item.clone(),
                tool_name: tool_name.clone(),
            }),
            _ => None,
        }
    }

    /// Tool output, if this is a tool call output item.
    pub fn tool_output(&self) -> Option<&Value> {
        match &self.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output),
            _ => None,
        }
    }
}

fn strip_created_by(mut value: Value) -> Value {
    if let Value::Object(m) = &mut value {
        m.remove("created_by");
    }
    value
}

/// A full response from a model call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Output items in the Responses API format.
    pub output: Vec<OutputItem>,
    /// Token usage of this call.
    pub usage: Usage,
    /// Response id (`resp_...`), used for `previous_response_id` chaining.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request id (`x-request-id`), when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Provider usage payload before normalization (opt-in via `preserve_raw_usage`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Value>,
}

impl ModelResponse {
    /// Convert the output items to input items for the next turn.
    pub fn to_input_items(&self) -> Vec<InputItem> {
        self.output
            .iter()
            .map(|o| InputItem(strip_created_by(o.to_value())))
            .collect()
    }
}

/// Helpers for working with items. Mirrors `agents.items.ItemHelpers`.
#[derive(Debug)]
pub struct ItemHelpers;

impl ItemHelpers {
    /// Extract the last content of a message (text or refusal). Empty for non-messages.
    pub fn extract_last_content(item: &OutputItem) -> Result<String> {
        let OutputItem::Message(msg) = item else {
            return Ok(String::new());
        };
        match msg.content.last() {
            None => Ok(String::new()),
            Some(MessageContent::OutputText { text, .. }) => Ok(text.clone()),
            Some(MessageContent::Refusal { refusal }) => Ok(refusal.clone()),
            Some(MessageContent::Other(v)) => Err(AgentsError::model_behavior(format!(
                "Unexpected content type: {}",
                v.get("type").and_then(Value::as_str).unwrap_or("unknown")
            ))),
        }
    }

    /// The last text part of a message, if any.
    pub fn extract_last_text(item: &OutputItem) -> Option<String> {
        let OutputItem::Message(msg) = item else {
            return None;
        };
        match msg.content.last() {
            Some(MessageContent::OutputText { text, .. }) => Some(text.clone()),
            _ => None,
        }
    }

    /// All text of a message, or `None` if empty / not a message.
    pub fn extract_text(item: &OutputItem) -> Option<String> {
        let OutputItem::Message(msg) = item else {
            return None;
        };
        let t = msg.text_content();
        if t.is_empty() { None } else { Some(t) }
    }

    /// All refusal text of a message, or `None`.
    pub fn extract_refusal(item: &OutputItem) -> Option<String> {
        let OutputItem::Message(msg) = item else {
            return None;
        };
        msg.refusal()
    }

    /// Convert runner input into a list of input items.
    pub fn input_to_new_input_list(input: &RunInput) -> Vec<InputItem> {
        input.to_items()
    }

    /// Concatenate the text of all message output items.
    pub fn text_message_outputs(items: &[RunItem]) -> String {
        let mut out = String::new();
        for item in items {
            if let RunItemKind::MessageOutput { raw_item } = &item.kind {
                out.push_str(&raw_item.text_content());
            }
        }
        out
    }

    /// The text of a single message output item.
    pub fn text_message_output(item: &RunItem) -> String {
        match &item.kind {
            RunItemKind::MessageOutput { raw_item } => raw_item.text_content(),
            _ => String::new(),
        }
    }

    /// Build a `function_call_output` wire item for a tool call and its output.
    pub fn tool_call_output_item(call: &FunctionCall, output: &ToolOutput) -> Value {
        let converted = match output {
            ToolOutput::Text(s) => Value::String(s.clone()),
            ToolOutput::Json(v) => match v {
                Value::String(s) => Value::String(s.clone()),
                other => Value::String(other.to_string()),
            },
            ToolOutput::Parts(parts) => {
                Value::Array(parts.iter().map(ToolOutputPart::to_input_content).collect())
            }
        };
        let mut item = json!({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": converted,
        });
        if let Some(caller) = call.extra.get("caller") {
            item["caller"] = caller.clone();
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_and_unknown_output_items() {
        let msg = OutputItem::from_value(json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]
        }));
        assert!(matches!(&msg, OutputItem::Message(m) if m.text_content() == "hi"));
        assert_eq!(msg.to_value()["type"], "message");

        let fc = OutputItem::from_value(json!({
            "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"
        }));
        assert!(matches!(fc, OutputItem::FunctionCall(_)));

        let other = OutputItem::from_value(json!({"type": "web_search_call", "id": "ws_1"}));
        assert!(matches!(other, OutputItem::Other(_)));
        assert_eq!(other.item_type(), "web_search_call");
    }

    #[test]
    fn run_item_round_trips_through_serde() {
        let item = RunItem::new(
            "a",
            RunItemKind::MessageOutput {
                raw_item: MessageOutput::text("x"),
            },
        );
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "message_output_item");
        let back: RunItem = serde_json::from_value(v).unwrap();
        assert_eq!(back, item);
    }

    #[test]
    fn tool_output_items() {
        let call = FunctionCall::new("f", "{}");
        let v = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Text("ok".into()));
        assert_eq!(v["type"], "function_call_output");
        assert_eq!(v["output"], "ok");
        let parts = ToolOutput::Parts(vec![ToolOutputPart::image_url("http://x/y.png")]);
        let v = ItemHelpers::tool_call_output_item(&call, &parts);
        assert_eq!(v["output"][0]["type"], "input_image");
    }
}
