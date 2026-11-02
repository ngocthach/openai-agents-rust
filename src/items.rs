//! Input items, model output items, run items and helpers. Mirrors `agents.items`.
//!
//! Items follow the OpenAI Responses API wire format. Items that the SDK interprets (messages,
//! function calls, reasoning) are typed; everything else is carried as raw JSON so new item types
//! keep flowing through unchanged.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::fmt;

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
            // A call that stands in for another tool item (`apply_patch_call`,
            // `custom_tool_call`, `mcp_approval_request`, ...) keeps that item's type.
            m.entry("type")
                .or_insert_with(|| Value::String("function_call".into()));
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
    /// Where the approved tool call came from. Mirrors `ToolApprovalItem.tool_origin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_origin: Option<crate::tool::ToolOrigin>,
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
    /// The namespace of the called tool, when the call carries one. Mirrors
    /// `ToolApprovalItem.tool_namespace`.
    pub fn tool_namespace(&self) -> Option<&str> {
        self.raw_item
            .extra
            .get("namespace")
            .and_then(Value::as_str)
            .filter(|ns| !ns.is_empty())
    }
    /// `namespace.name`, or the bare name when there is no namespace or the namespace is the
    /// synthetic one a deferred top-level tool gets (equal to the tool name). Mirrors
    /// `ToolApprovalItem.qualified_name`.
    pub fn qualified_name(&self) -> String {
        match self.tool_namespace() {
            Some(ns) if ns != self.tool_name => format!("{ns}.{}", self.tool_name),
            _ => self.tool_name.clone(),
        }
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
    /// `custom_data` is SDK-only data attached by a `custom_data_extractor`: it is never part
    /// of `raw_item` and is not replayed to the model.
    #[serde(rename = "tool_call_output_item")]
    ToolCallOutput {
        raw_item: Value,
        output: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_data: Option<Value>,
    },
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
    /// A compaction item produced by `responses.compact`.
    #[serde(rename = "compaction_item")]
    Compaction { raw_item: Value },
}

/// An item generated during a run, tagged with the agent that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunItem {
    /// Name of the agent associated with the item.
    pub agent: String,
    /// The item.
    #[serde(flatten)]
    pub kind: RunItemKind,
    /// Description of the tool, when known at item creation time (tool calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Short display label of the tool, when known at item creation time (tool calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Where the item's tool came from (tool calls, outputs and approvals). Mirrors
    /// `ToolCallItem.tool_origin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_origin: Option<crate::tool::ToolOrigin>,
    /// The function-tool lookup key the item resolves under, in the snapshot's wire form
    /// (`{"kind": "bare" | "namespaced" | "deferred_top_level", ...}`). Mirrors
    /// `ToolApprovalItem.tool_lookup_key`; kept so a restored approval still names the tool it
    /// was raised for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_lookup_key: Option<Value>,
    /// Whether an approval for a namespaced tool also covers the bare tool name. Mirrors
    /// `ToolApprovalItem._allow_bare_name_alias`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_bare_name_alias: bool,
    /// Identity of one admitted occurrence of staged input ([`RunItemKind::Input`] only). Two
    /// admitted inputs with the same content are distinct occurrences, so a server-managed
    /// conversation delivers each exactly once. Mirrors `InputItem.input_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_id: Option<String>,
    /// Run-local lineage key that survives copies, so the loop can tell which items a handoff
    /// input filter kept. Never serialized and never part of equality. Mirrors
    /// `_NESTED_HISTORY_RUN_ITEM_OCCURRENCE_KEY`.
    #[serde(skip)]
    pub(crate) occurrence: OccurrenceKey,
}

/// An opaque copy-lineage tag on a [`RunItem`]: invisible to equality, serialization and
/// `Debug`, so it cannot change how items compare or what a snapshot contains.
#[derive(Clone, Copy, Default)]
pub(crate) struct OccurrenceKey(pub(crate) Option<u64>);

impl PartialEq for OccurrenceKey {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl fmt::Debug for OccurrenceKey {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl RunItem {
    pub fn new(agent: impl Into<String>, kind: RunItemKind) -> Self {
        RunItem {
            agent: agent.into(),
            kind,
            description: None,
            title: None,
            tool_origin: None,
            tool_lookup_key: None,
            allow_bare_name_alias: false,
            input_id: None,
            occurrence: OccurrenceKey::default(),
        }
    }

    /// Attach the tool's description and display title.
    pub fn with_tool_details(mut self, description: Option<String>, title: Option<String>) -> Self {
        self.description = description;
        self.title = title;
        self
    }

    /// Attach where the item's tool came from.
    pub fn with_tool_origin(mut self, origin: Option<crate::tool::ToolOrigin>) -> Self {
        self.tool_origin = origin;
        self
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
            RunItemKind::Compaction { .. } => "compaction_item",
        }
    }

    /// Convert to an input item for the next model call.
    pub fn to_input_item(&self) -> InputItem {
        match &self.kind {
            RunItemKind::Input { raw_item } => raw_item.clone(),
            RunItemKind::MessageOutput { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::Reasoning { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::ToolCall { raw_item } => InputItem(strip_created_by(raw_item.clone())),
            RunItemKind::ToolCallOutput { raw_item, .. } => InputItem(strip_created_by(
                sanitize_shell_call_output(raw_item.clone()),
            )),
            RunItemKind::ToolApproval { raw_item, .. } => InputItem(raw_item.to_value()),
            RunItemKind::HandoffCall { raw_item } => InputItem(raw_item.to_value()),
            RunItemKind::HandoffOutput { raw_item, .. } => raw_item.clone(),
            RunItemKind::MCPListTools { raw_item }
            | RunItemKind::MCPApprovalRequest { raw_item }
            | RunItemKind::MCPApprovalResponse { raw_item }
            | RunItemKind::Compaction { raw_item } => InputItem(raw_item.clone()),
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
                tool_origin: self.tool_origin.clone(),
            }),
            _ => None,
        }
    }

    /// The tool name of a tool call item (`ToolCallItem.tool_name`); `None` for a call payload
    /// that carries no `name` (a file search call) and for every other item kind.
    pub fn tool_name(&self) -> Option<String> {
        match &self.kind {
            RunItemKind::ToolCall { raw_item } => raw_item
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            _ => None,
        }
    }

    /// The call identifier of a tool call or tool output item (`ToolCallItem.call_id`,
    /// `ToolCallOutputItem.call_id`): the payload's non-empty `call_id`, else its `id`.
    pub fn call_id(&self) -> Option<String> {
        let raw = match &self.kind {
            RunItemKind::ToolCall { raw_item } | RunItemKind::ToolCallOutput { raw_item, .. } => {
                raw_item
            }
            _ => return None,
        };
        ["call_id", "id"].iter().find_map(|key| {
            raw.get(*key)
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        })
    }

    /// SDK-only custom data attached to a tool output item.
    pub fn custom_data(&self) -> Option<&Value> {
        match &self.kind {
            RunItemKind::ToolCallOutput { custom_data, .. } => custom_data.as_ref(),
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

/// The `input_*` parts of a JSON tool output that already has the shape of a structured output
/// (`{"type": "text" | "image" | "file", ...}`, or a non-empty list of them, all valid). Port of
/// `ItemHelpers._convert_tool_output_as_structured` for untyped values; unknown extra fields are
/// ignored.
fn structured_tool_output(value: &Value) -> Option<Vec<Value>> {
    fn part(value: &Value) -> Option<Value> {
        let map = value.as_object()?;
        let text = |key: &str| match map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(Value::String(s.clone()))),
            Some(_) => Err(()),
        };
        let copy = |keys: &[&str], mut out: serde_json::Map<String, Value>| {
            for key in keys {
                if let Some(v) = text(key).ok()? {
                    out.insert((*key).to_string(), v);
                }
            }
            Some(out)
        };
        match map.get("type")?.as_str()? {
            "text" => {
                let text = text("text").ok()??;
                Some(json!({"type": "input_text", "text": text}))
            }
            "image" => {
                if let Some(detail) = map.get("detail").filter(|d| !d.is_null())
                    && !matches!(detail.as_str(), Some("low" | "high" | "auto"))
                {
                    return None;
                }
                let out = copy(
                    &["image_url", "file_id", "detail"],
                    serde_json::Map::from_iter([("type".to_string(), json!("input_image"))]),
                )?;
                (out.contains_key("image_url") || out.contains_key("file_id"))
                    .then_some(Value::Object(out))
            }
            "file" => {
                let out = copy(
                    &["file_data", "file_url", "file_id", "filename"],
                    serde_json::Map::from_iter([("type".to_string(), json!("input_file"))]),
                )?;
                (out.contains_key("file_data")
                    || out.contains_key("file_url")
                    || out.contains_key("file_id"))
                .then_some(Value::Object(out))
            }
            _ => None,
        }
    }
    match value {
        Value::Array(items) if !items.is_empty() => items.iter().map(part).collect(),
        Value::Object(_) => part(value).map(|p| vec![p]),
        _ => None,
    }
}

/// Drop server-assigned `created_by` metadata, which the Responses input schema rejects.
///
/// `shell_call_output.output` is a list of content chunks that each carry their own
/// `created_by`, so both levels are stripped.
fn strip_created_by(mut value: Value) -> Value {
    if let Value::Object(m) = &mut value {
        m.remove("created_by");
        if m.get("type").and_then(Value::as_str) == Some("shell_call_output")
            && let Some(Value::Array(chunks)) = m.get_mut("output")
        {
            for chunk in chunks {
                if let Value::Object(chunk) = chunk {
                    chunk.remove("created_by");
                }
            }
        }
    }
    value
}

/// Hosted shell outputs carry SDK book-keeping fields the Responses API does not accept.
fn sanitize_shell_call_output(mut value: Value) -> Value {
    if let Value::Object(m) = &mut value
        && m.get("type").and_then(Value::as_str) == Some("shell_call_output")
    {
        m.remove("status");
        m.remove("shell_output");
        m.remove("provider_data");
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
                // A dict (or non-empty list of dicts) with an explicit, valid `type` is a
                // structured output, like the typed parts; anything else is stringified.
                other => match structured_tool_output(other) {
                    Some(parts) => Value::Array(parts),
                    None => Value::String(other.to_string()),
                },
            },
            ToolOutput::Parts(parts) => {
                Value::Array(parts.iter().map(ToolOutputPart::to_input_content).collect())
            }
        };
        let item = json!({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": converted,
        });
        Self::copy_tool_call_caller(&call.to_value(), item)
    }

    /// Copy a program caller relationship from a tool call onto its output item.
    /// Port of `ItemHelpers.copy_tool_call_caller`.
    pub fn copy_tool_call_caller(tool_call: &Value, mut output_item: Value) -> Value {
        if let Some(caller) = tool_call.get("caller")
            && !caller.is_null()
        {
            output_item["caller"] = caller.clone();
        }
        output_item
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
