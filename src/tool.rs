//! Tools: function tools and hosted OpenAI tools. Mirrors `agents.tool`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::items::{RunItem, ToolApprovalItem};
use crate::run_context::RunContext;
use crate::strict_schema::ensure_strict_json_schema;
use crate::tool_context::ToolContext;
use crate::tool_guardrails::{ToolInputGuardrail, ToolOutputGuardrail};
use crate::util::BoxFuture;

mod local;
pub use local::*;

/// The runtime source of a function-tool-backed run item. Mirrors `ToolOriginType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOriginType {
    /// An ordinary function tool.
    Function,
    /// A tool served by an MCP server.
    Mcp,
    /// An agent exposed through `Agent::as_tool`.
    AgentAsTool,
}

impl ToolOriginType {
    /// The wire spelling of the type.
    pub fn as_str(self) -> &'static str {
        match self {
            ToolOriginType::Function => "function",
            ToolOriginType::Mcp => "mcp",
            ToolOriginType::AgentAsTool => "agent_as_tool",
        }
    }
}

/// Serializable metadata describing where a function-tool-backed item came from.
/// Mirrors `ToolOrigin`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolOrigin {
    /// What kind of tool produced the item.
    #[serde(rename = "type")]
    pub r#type: ToolOriginType,
    /// Log-safe name of the MCP server, for [`ToolOriginType::Mcp`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_server_name: Option<String>,
    /// Name of the wrapped agent, for [`ToolOriginType::AgentAsTool`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    /// Name the wrapped agent's tool is exposed under, for [`ToolOriginType::AgentAsTool`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_tool_name: Option<String>,
}

impl ToolOrigin {
    /// An origin of this type with no further detail.
    pub fn new(r#type: ToolOriginType) -> Self {
        ToolOrigin {
            r#type,
            mcp_server_name: None,
            agent_name: None,
            agent_tool_name: None,
        }
    }

    /// The origin of a tool served by the MCP server named `server_name`.
    pub fn mcp(server_name: impl Into<String>) -> Self {
        ToolOrigin {
            mcp_server_name: Some(server_name.into()),
            ..ToolOrigin::new(ToolOriginType::Mcp)
        }
    }

    /// The origin of an agent exposed as the tool `tool_name`.
    pub fn agent_as_tool(agent_name: impl Into<String>, tool_name: impl Into<String>) -> Self {
        ToolOrigin {
            agent_name: Some(agent_name.into()),
            agent_tool_name: Some(tool_name.into()),
            ..ToolOrigin::new(ToolOriginType::AgentAsTool)
        }
    }

    /// Convert to JSON-compatible data. Mirrors `ToolOrigin.to_json_dict`.
    pub fn to_json_dict(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Read from JSON-compatible data; anything malformed is `None`, so a snapshot written by a
    /// newer SDK still loads. Mirrors `ToolOrigin.from_json_dict`.
    pub fn from_json_dict(value: &Value) -> Option<ToolOrigin> {
        let object = value.as_object()?;
        let r#type = match object.get("type")?.as_str()? {
            "function" => ToolOriginType::Function,
            "mcp" => ToolOriginType::Mcp,
            "agent_as_tool" => ToolOriginType::AgentAsTool,
            _ => return None,
        };
        let optional = |key: &str| object.get(key).and_then(Value::as_str).map(str::to_string);
        Some(ToolOrigin {
            r#type,
            mcp_server_name: optional("mcp_server_name"),
            agent_name: optional("agent_name"),
            agent_tool_name: optional("agent_tool_name"),
        })
    }
}

/// A structured part of a tool output (text, image or file).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutputPart {
    /// Text content.
    Text { text: String },
    /// An image, by URL (or data URL) or file id.
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// `low`, `high` or `auto`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// A file, by base64 data, URL or file id.
    File {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl ToolOutputPart {
    pub fn text(text: impl Into<String>) -> Self {
        ToolOutputPart::Text { text: text.into() }
    }
    pub fn image_url(url: impl Into<String>) -> Self {
        ToolOutputPart::Image {
            image_url: Some(url.into()),
            file_id: None,
            detail: None,
        }
    }
    pub fn file_id(id: impl Into<String>) -> Self {
        ToolOutputPart::File {
            file_data: None,
            file_url: None,
            file_id: Some(id.into()),
            filename: None,
        }
    }

    /// Convert to a Responses `input_*` content part.
    pub fn to_input_content(&self) -> Value {
        match self {
            ToolOutputPart::Text { text } => json!({"type": "input_text", "text": text}),
            ToolOutputPart::Image {
                image_url,
                file_id,
                detail,
            } => {
                let mut v = json!({"type": "input_image"});
                if let Some(u) = image_url {
                    v["image_url"] = Value::String(u.clone());
                }
                if let Some(f) = file_id {
                    v["file_id"] = Value::String(f.clone());
                }
                if let Some(d) = detail {
                    v["detail"] = Value::String(d.clone());
                }
                v
            }
            ToolOutputPart::File {
                file_data,
                file_url,
                file_id,
                filename,
            } => {
                let mut v = json!({"type": "input_file"});
                if let Some(d) = file_data {
                    v["file_data"] = Value::String(d.clone());
                }
                if let Some(u) = file_url {
                    v["file_url"] = Value::String(u.clone());
                }
                if let Some(f) = file_id {
                    v["file_id"] = Value::String(f.clone());
                }
                if let Some(n) = filename {
                    v["filename"] = Value::String(n.clone());
                }
                v
            }
        }
    }
}

/// The output of a function tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    /// Plain text sent to the model as-is.
    Text(String),
    /// Structured parts (text, images, files).
    Parts(Vec<ToolOutputPart>),
    /// A JSON value, sent to the model as its JSON string.
    Json(Value),
}

impl ToolOutput {
    /// Text output.
    pub fn text(text: impl Into<String>) -> Self {
        ToolOutput::Text(text.into())
    }

    /// JSON output from any serializable value.
    pub fn json<T: Serialize>(value: &T) -> Result<Self> {
        Ok(ToolOutput::Json(serde_json::to_value(value)?))
    }

    /// The model-visible string form.
    pub fn to_model_string(&self) -> String {
        match self {
            ToolOutput::Text(s) => s.clone(),
            ToolOutput::Json(Value::String(s)) => s.clone(),
            ToolOutput::Json(v) => v.to_string(),
            ToolOutput::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ToolOutputPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// A JSON representation of the output for run items and state.
    pub fn to_json_value(&self) -> Value {
        match self {
            ToolOutput::Text(s) => Value::String(s.clone()),
            ToolOutput::Json(v) => v.clone(),
            ToolOutput::Parts(parts) => serde_json::to_value(parts).unwrap_or(Value::Null),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(value: String) -> Self {
        ToolOutput::Text(value)
    }
}

impl From<&str> for ToolOutput {
    fn from(value: &str) -> Self {
        ToolOutput::Text(value.to_string())
    }
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        ToolOutput::Json(value)
    }
}

impl From<Vec<ToolOutputPart>> for ToolOutput {
    fn from(value: Vec<ToolOutputPart>) -> Self {
        ToolOutput::Parts(value)
    }
}

/// Values that can be returned from a function tool closure.
pub trait IntoToolOutput {
    fn into_tool_output(self) -> ToolOutput;
}

impl IntoToolOutput for ToolOutput {
    fn into_tool_output(self) -> ToolOutput {
        self
    }
}
impl IntoToolOutput for String {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::Text(self)
    }
}
impl IntoToolOutput for &str {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::Text(self.to_string())
    }
}
impl IntoToolOutput for Value {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::Json(self)
    }
}
impl IntoToolOutput for Vec<ToolOutputPart> {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::Parts(self)
    }
}
impl IntoToolOutput for () {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::Text(String::new())
    }
}
macro_rules! into_tool_output_display {
    ($($t:ty),*) => {
        $( impl IntoToolOutput for $t {
            fn into_tool_output(self) -> ToolOutput { ToolOutput::Text(self.to_string()) }
        } )*
    };
}
into_tool_output_display!(bool, i32, i64, u32, u64, f32, f64, usize);

/// Result of invoking a function tool during a turn.
#[derive(Debug, Clone)]
pub struct FunctionToolResult {
    /// Tool name.
    pub tool_name: String,
    /// Namespaced tool name (`namespace.name`), equal to `tool_name` when there is none.
    pub qualified_tool_name: String,
    /// SDK-only data attached by the tool or its `custom_data_extractor`.
    pub custom_data: Option<Value>,
    /// Call id.
    pub call_id: String,
    /// The output.
    pub output: ToolOutput,
    /// The run item produced (output item or approval item).
    pub run_item: Option<RunItem>,
    /// Approval interruptions raised by the call.
    pub interruptions: Vec<ToolApprovalItem>,
}

/// Behaviour when a function tool times out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolTimeoutBehavior {
    /// Send a model-visible timeout message (default).
    #[default]
    ErrorAsResult,
    /// Raise `AgentsError::ToolTimeoutError` and fail the run.
    RaiseException,
}

/// Dynamic enablement check for tools and handoffs.
pub type IsEnabledFn<C> =
    Arc<dyn Fn(RunContext<C>, String) -> BoxFuture<'static, bool> + Send + Sync>;

/// A boolean or a dynamic check receiving the run context and the agent name.
pub enum IsEnabled<C> {
    Static(bool),
    Dynamic(IsEnabledFn<C>),
}

impl<C> Clone for IsEnabled<C> {
    fn clone(&self) -> Self {
        match self {
            IsEnabled::Static(b) => IsEnabled::Static(*b),
            IsEnabled::Dynamic(f) => IsEnabled::Dynamic(f.clone()),
        }
    }
}

impl<C> fmt::Debug for IsEnabled<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsEnabled::Static(b) => write!(f, "IsEnabled::Static({b})"),
            IsEnabled::Dynamic(_) => write!(f, "IsEnabled::Dynamic(..)"),
        }
    }
}

impl<C> IsEnabled<C> {
    /// Evaluate the check.
    pub async fn evaluate(&self, ctx: &RunContext<C>, agent_name: &str) -> bool {
        match self {
            IsEnabled::Static(b) => *b,
            IsEnabled::Dynamic(f) => f(ctx.clone(), agent_name.to_string()).await,
        }
    }
}

impl<C> From<bool> for IsEnabled<C> {
    fn from(value: bool) -> Self {
        IsEnabled::Static(value)
    }
}

/// Per-call approval decision: `(context, parsed arguments, call id) -> needs approval`.
pub type NeedsApprovalFn<C> =
    Arc<dyn Fn(RunContext<C>, Value, String) -> BoxFuture<'static, bool> + Send + Sync>;

/// Whether a tool needs approval.
pub enum NeedsApproval<C> {
    Static(bool),
    Dynamic(NeedsApprovalFn<C>),
}

impl<C> Clone for NeedsApproval<C> {
    fn clone(&self) -> Self {
        match self {
            NeedsApproval::Static(b) => NeedsApproval::Static(*b),
            NeedsApproval::Dynamic(f) => NeedsApproval::Dynamic(f.clone()),
        }
    }
}

impl<C> fmt::Debug for NeedsApproval<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeedsApproval::Static(b) => write!(f, "NeedsApproval::Static({b})"),
            NeedsApproval::Dynamic(_) => write!(f, "NeedsApproval::Dynamic(..)"),
        }
    }
}

impl<C> NeedsApproval<C> {
    /// Evaluate the rule. A dynamic rule fails closed (requires approval) when the arguments
    /// are not a JSON object.
    pub async fn evaluate(&self, ctx: &RunContext<C>, arguments: &str, call_id: &str) -> bool {
        match self {
            NeedsApproval::Static(b) => *b,
            NeedsApproval::Dynamic(f) => {
                let parsed: Option<Value> = serde_json::from_str(arguments).ok();
                match parsed {
                    Some(v @ Value::Object(_)) => f(ctx.clone(), v, call_id.to_string()).await,
                    _ => true,
                }
            }
        }
    }
}

impl<C> From<bool> for NeedsApproval<C> {
    fn from(value: bool) -> Self {
        NeedsApproval::Static(value)
    }
}

/// Function converting a tool failure into a model-visible message.
///
/// Port of Python's `ToolErrorFunction`: the handler may await, and it may return an error,
/// which aborts the run in place of the original failure (Python lets a formatter re-raise).
/// Build one with [`tool_error_fn`] (plain closure), [`try_tool_error_fn`] (plain closure that
/// may fail) or [`async_tool_error_fn`] (async).
pub type ToolErrorFn<C> = Arc<
    dyn for<'a> Fn(&'a RunContext<C>, &'a AgentsError) -> BoxFuture<'a, Result<String>>
        + Send
        + Sync,
>;

/// Wrap a synchronous closure that returns the message as a [`ToolErrorFn`].
pub fn tool_error_fn<C, F>(f: F) -> ToolErrorFn<C>
where
    F: Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync + 'static,
{
    async_tool_error_fn(move |ctx, error| Box::pin(std::future::ready(Ok(f(ctx, error)))))
}

/// Wrap a synchronous closure that may fail as a [`ToolErrorFn`]. An `Err` aborts the run with
/// that error, as a Python formatter that re-raises does.
pub fn try_tool_error_fn<C, F>(f: F) -> ToolErrorFn<C>
where
    F: Fn(&RunContext<C>, &AgentsError) -> Result<String> + Send + Sync + 'static,
{
    async_tool_error_fn(move |ctx, error| Box::pin(std::future::ready(f(ctx, error))))
}

/// Wrap an async closure as a [`ToolErrorFn`]. The closure returns a boxed future, usually
/// `Box::pin(async move { ... })`; an `Err` aborts the run with that error.
pub fn async_tool_error_fn<C, F>(f: F) -> ToolErrorFn<C>
where
    F: for<'a> Fn(&'a RunContext<C>, &'a AgentsError) -> BoxFuture<'a, Result<String>>
        + Send
        + Sync
        + 'static,
{
    Arc::new(f)
}

/// What a function-tool custom-data extractor receives.
/// Mirrors `agents.tool.FunctionToolCustomDataContext`.
pub struct FunctionToolCustomDataContext<'a, C> {
    /// The tool invocation context.
    pub tool_context: &'a ToolContext<C>,
    /// The tool that was invoked.
    pub tool: &'a FunctionTool<C>,
    /// The model-visible tool output.
    pub output: &'a ToolOutput,
    /// The raw output item that will be replayed to the model.
    pub raw_item: &'a Value,
    /// The `_meta` of the MCP call result for an MCP tool, `None` otherwise. Mirrors Python's
    /// `FunctionToolCustomDataContext.result_meta`.
    ///
    /// Adding this field is breaking for code that builds the struct with a literal; the SDK
    /// itself is the only constructor.
    pub result_meta: Option<Value>,
}

impl<C> fmt::Debug for FunctionToolCustomDataContext<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionToolCustomDataContext")
            .field("tool", &self.tool.name)
            .finish_non_exhaustive()
    }
}

/// Produces SDK-only data to attach to a tool output item.
/// Mirrors `agents.util._custom_data.CustomDataExtractor`.
pub type CustomDataExtractor<C> = Arc<
    dyn for<'a> Fn(&FunctionToolCustomDataContext<'a, C>) -> BoxFuture<'static, Option<Value>>
        + Send
        + Sync,
>;

/// Return a JSON-object copy of custom tool-output data, or `None` when it is empty.
/// Port of `normalize_custom_data`.
pub fn normalize_custom_data(value: Option<Value>) -> Result<Option<Value>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) if map.is_empty() => Ok(None),
        Some(Value::Object(map)) => Ok(Some(Value::Object(map))),
        Some(_) => Err(AgentsError::user(
            "custom_data_extractor must return a JSON object or None.",
        )),
    }
}

/// Merge custom-data objects; later values win. Port of `merge_custom_data`.
pub fn merge_custom_data(values: &[Option<Value>]) -> Result<Option<Value>> {
    let mut merged = serde_json::Map::new();
    for value in values {
        if let Some(Value::Object(map)) = normalize_custom_data(value.clone())? {
            merged.extend(map);
        }
    }
    if merged.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Value::Object(merged)))
    }
}

/// The error for tool arguments that do not parse or validate. Port of
/// `_parse_function_tool_json_input` and the argument-validation branch of `function_tool`.
///
/// The payload is tool data and a serde error echoes the offending value, so neither reaches
/// the message unless the tool data logging policy allows it.
pub(crate) fn invalid_tool_input_error(
    tool_name: &str,
    payload: &str,
    error: &serde_json::Error,
) -> AgentsError {
    let base = format!("Invalid JSON input for tool {tool_name}");
    if crate::debug::dont_log_tool_data() {
        log::debug!(target: crate::logger::LOG_TARGET, "{base}");
        return AgentsError::model_behavior(base);
    }
    // A decode error reports the raw payload; a validation error reports what failed.
    let detail = if serde_json::from_str::<Value>(payload).is_err() {
        payload.to_string()
    } else {
        error.to_string()
    };
    let message = format!("{base}: {detail}");
    log::debug!(target: crate::logger::LOG_TARGET, "{message}");
    AgentsError::model_behavior(message)
}

/// The error for a tool input that parses as JSON but is not an object. The payload is not
/// echoed, so it needs no tool-data gate.
pub(crate) fn invalid_tool_input_not_object(tool_name: &str) -> AgentsError {
    log::debug!(
        target: crate::logger::LOG_TARGET,
        "Invalid JSON input for tool {tool_name}: expected a JSON object"
    );
    AgentsError::model_behavior(format!(
        "Invalid JSON input for tool {tool_name}: expected a JSON object"
    ))
}

/// One shared [`default_tool_error_function`] handle per context type, so a configuration
/// that defaults to it (such as `MCPConfig::default`) can later be recognized by identity.
pub(crate) fn shared_default_tool_error_function<C: 'static>() -> ToolErrorFn<C> {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(TypeId::of::<C>())
        .or_insert_with(|| {
            let f: ToolErrorFn<C> = tool_error_fn(default_tool_error_function::<C>);
            Box::new(f)
        })
        .downcast_ref::<ToolErrorFn<C>>()
        .expect("cached by its own type id")
        .clone()
}

/// Whether `f` is the shared [`default_tool_error_function`] handle.
#[cfg(feature = "mcp")]
pub(crate) fn is_shared_default_tool_error_function<C: 'static>(f: &ToolErrorFn<C>) -> bool {
    Arc::ptr_eq(f, &shared_default_tool_error_function::<C>())
}

fn type_id_of_val<T: 'static>(_: &T) -> std::any::TypeId {
    std::any::TypeId::of::<T>()
}

/// The default message sent to the model when a tool fails. It does not expose the error.
pub fn default_tool_error_function<C>(_ctx: &RunContext<C>, _error: &AgentsError) -> String {
    "An error occurred while running the tool. Please try again.".to_string()
}

/// Default model-visible message when a tool call is rejected by a reviewer.
pub const DEFAULT_APPROVAL_REJECTION_MESSAGE: &str = "Tool execution was not approved.";

type InvokeFn<C> =
    Arc<dyn Fn(ToolContext<C>, String) -> BoxFuture<'static, Result<ToolOutput>> + Send + Sync>;

/// Checks a tool's return value against the output type declared with
/// [`FunctionTool::with_output_type`], returning the validation error message.
type OutputValidatorFn = Arc<dyn Fn(&Value) -> std::result::Result<(), String> + Send + Sync>;

/// Whether parsed arguments are exactly the invocation a typed tool would run: they parse into
/// its argument type and name every declared parameter and nothing else, so no default is
/// filled in and no key is dropped. Mirrors the `unchanged` check of `prepare_arguments`.
type ApprovalArgsCheckFn = Arc<dyn Fn(&Value) -> bool + Send + Sync>;

/// A tool that wraps a Rust async function.
pub struct FunctionTool<C> {
    /// Tool name shown to the model.
    pub name: String,
    /// Tool description shown to the model.
    pub description: String,
    /// JSON schema of the parameters.
    pub params_json_schema: Value,
    /// The parameters schema as given, before strict conversion. Turning strictness off restores
    /// it, as Python keeps the original schema for `strict_json_schema=False`.
    pub(crate) raw_params_json_schema: Value,
    /// Whether the schema is strict.
    pub strict_json_schema: bool,
    /// Enablement check.
    pub is_enabled: IsEnabled<C>,
    /// Approval rule.
    pub needs_approval: NeedsApproval<C>,
    /// Timeout for each invocation.
    pub timeout: Option<Duration>,
    /// What to do on timeout.
    pub timeout_behavior: ToolTimeoutBehavior,
    /// Guardrails run before the tool executes.
    pub tool_input_guardrails: Vec<ToolInputGuardrail<C>>,
    /// Guardrails run after the tool executes.
    pub tool_output_guardrails: Vec<ToolOutputGuardrail<C>>,
    /// Hide the tool until loaded through hosted tool search (Responses API only).
    pub defer_loading: bool,
    /// Error formatter; `None` re-raises tool errors.
    pub failure_error_function: Option<ToolErrorFn<C>>,
    /// Whether `failure_error_function` is still the SDK default. When it is, the run's
    /// `tool_error_formatter` gets a `tool_exception` chance to rewrite the message.
    pub(crate) uses_default_failure_error_function: bool,
    /// Whether the failure policy is [`default_tool_error_function`], set implicitly or
    /// explicitly. That policy keeps the error detail out of traces.
    pub(crate) default_failure_error_function: bool,
    /// Timeout message formatter.
    pub timeout_error_function: Option<ToolErrorFn<C>>,
    /// Programmatic callers allowed to invoke this tool (Responses API only).
    pub allowed_callers: Option<Vec<String>>,
    /// Attaches SDK-only data to the tool output item.
    pub custom_data_extractor: Option<CustomDataExtractor<C>>,
    /// Strict JSON object schema of the tool's output, sent as `output_schema` on the
    /// Responses API. Mirrors `FunctionTool.output_json_schema`.
    pub output_json_schema: Option<Value>,
    /// Runtime check of the tool's return value against a declared output type. Set by
    /// [`FunctionTool::with_output_type`]; mirrors `FunctionTool._output_type_adapter`.
    output_validator: Option<OutputValidatorFn>,
    /// See [`ApprovalArgsCheckFn`]; set by [`FunctionTool::new`].
    approval_args_check: Option<ApprovalArgsCheckFn>,
    /// Whether the argument type runs application code while parsing. See
    /// [`FunctionTool::with_args_have_validators`].
    args_have_validators: bool,
    /// Human-readable title of the MCP tool this one was derived from, when it declared one.
    /// Mirrors `FunctionTool._mcp_title`.
    pub mcp_title: Option<String>,
    /// Whether this tool came from `codex_tool`. Mirrors `FunctionTool._is_codex_tool`; the
    /// agent uses it to fail fast on two Codex tools sharing a name.
    pub(crate) is_codex_tool: bool,
    invoke: InvokeFn<C>,
    pub(crate) is_agent_tool: bool,
    pub(crate) mcp_server_name: Option<String>,
    /// The annotations the MCP server declared for this tool, captured when it was converted.
    pub(crate) mcp_annotations: Option<Value>,
    /// Set on the stand-in the runner builds for a resumed call whose tool is gone. Its
    /// recorded decision was taken on the original tool, so the call is matched against the
    /// capability that interruption registered rather than the stand-in's own.
    pub(crate) resumes_registered_capability: bool,
    /// Which MCP server and raw tool this function tool calls: `(server log name, raw tool
    /// name, index in the agent's server list)`. Persisted per call so an approved call can only
    /// resume on the recipient it was approved for. Mirrors `FunctionTool._mcp_tool_binding`.
    pub(crate) mcp_tool_binding: Option<(String, String, Option<usize>)>,
    /// Explicit origin metadata for items this tool produces; see [`FunctionTool::tool_origin`].
    pub(crate) tool_origin: Option<ToolOrigin>,
    pub(crate) namespace: Option<String>,
    pub(crate) namespace_description: Option<String>,
}

impl<C> Clone for FunctionTool<C> {
    fn clone(&self) -> Self {
        FunctionTool {
            name: self.name.clone(),
            description: self.description.clone(),
            params_json_schema: self.params_json_schema.clone(),
            raw_params_json_schema: self.raw_params_json_schema.clone(),
            strict_json_schema: self.strict_json_schema,
            is_enabled: self.is_enabled.clone(),
            needs_approval: self.needs_approval.clone(),
            timeout: self.timeout,
            timeout_behavior: self.timeout_behavior,
            tool_input_guardrails: self.tool_input_guardrails.clone(),
            tool_output_guardrails: self.tool_output_guardrails.clone(),
            defer_loading: self.defer_loading,
            failure_error_function: self.failure_error_function.clone(),
            uses_default_failure_error_function: self.uses_default_failure_error_function,
            default_failure_error_function: self.default_failure_error_function,
            timeout_error_function: self.timeout_error_function.clone(),
            allowed_callers: self.allowed_callers.clone(),
            custom_data_extractor: self.custom_data_extractor.clone(),
            output_json_schema: self.output_json_schema.clone(),
            output_validator: self.output_validator.clone(),
            approval_args_check: self.approval_args_check.clone(),
            args_have_validators: self.args_have_validators,
            mcp_title: self.mcp_title.clone(),
            is_codex_tool: self.is_codex_tool,
            invoke: self.invoke.clone(),
            is_agent_tool: self.is_agent_tool,
            mcp_server_name: self.mcp_server_name.clone(),
            mcp_annotations: self.mcp_annotations.clone(),
            resumes_registered_capability: self.resumes_registered_capability,
            mcp_tool_binding: self.mcp_tool_binding.clone(),
            tool_origin: self.tool_origin.clone(),
            namespace: self.namespace.clone(),
            namespace_description: self.namespace_description.clone(),
        }
    }
}

impl<C> fmt::Debug for FunctionTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("needs_approval", &self.needs_approval)
            .finish_non_exhaustive()
    }
}

/// Whether `value` has exactly the shape `schema` declares, so that parsing it would neither
/// fail nor fill in, drop or convert anything. Conservative: a keyword this does not understand,
/// a declared property that is absent and an undeclared one that is present all answer `false`,
/// and the call then needs a manual approval. Numbers follow Python's float rule: a `number`
/// must be written as a float, because an integer literal would be converted.
fn schema_shape_accepts(root: &Value, schema: &Value, value: &Value, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    let Value::Object(s) = schema else {
        return matches!(schema, Value::Bool(true));
    };
    if let Some(reference) = s.get("$ref").and_then(Value::as_str) {
        let Some(target) = reference.strip_prefix('#').and_then(|p| root.pointer(p)) else {
            return false;
        };
        if !schema_shape_accepts(root, target, value, depth + 1) {
            return false;
        }
    }
    if let Some(all) = s.get("allOf").and_then(Value::as_array)
        && !all
            .iter()
            .all(|sub| schema_shape_accepts(root, sub, value, depth + 1))
    {
        return false;
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(any) = s.get(key).and_then(Value::as_array)
            && !any
                .iter()
                .any(|sub| schema_shape_accepts(root, sub, value, depth + 1))
        {
            return false;
        }
    }
    if let Some(constant) = s.get("const")
        && constant != value
    {
        return false;
    }
    if let Some(options) = s.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        return false;
    }
    if s.contains_key("prefixItems") {
        return false;
    }
    let type_accepts = |name: &str| match name {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_f64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "array" => value.as_array().is_some_and(|items| match s.get("items") {
            Some(item_schema) => items
                .iter()
                .all(|item| schema_shape_accepts(root, item_schema, item, depth + 1)),
            None => true,
        }),
        "object" => value
            .as_object()
            .is_some_and(|map| object_shape_accepts(root, s, map, depth)),
        _ => false,
    };
    match s.get("type") {
        None => true,
        Some(Value::String(name)) => type_accepts(name),
        Some(Value::Array(names)) => names
            .iter()
            .any(|name| name.as_str().is_some_and(type_accepts)),
        Some(_) => false,
    }
}

fn object_shape_accepts(
    root: &Value,
    schema: &serde_json::Map<String, Value>,
    map: &serde_json::Map<String, Value>,
    depth: usize,
) -> bool {
    let properties = schema.get("properties").and_then(Value::as_object);
    let additional = schema.get("additionalProperties");
    if let Some(properties) = properties
        && !properties.keys().all(|key| map.contains_key(key))
    {
        return false;
    }
    map.iter().all(|(key, item)| {
        match properties.and_then(|props| props.get(key)) {
            Some(declared) => schema_shape_accepts(root, declared, item, depth + 1),
            None => match additional {
                Some(extra @ Value::Object(_)) => {
                    schema_shape_accepts(root, extra, item, depth + 1)
                }
                Some(Value::Bool(false)) => false,
                // Without declared properties any key is data; with them an undeclared key
                // would be dropped by parsing.
                _ => properties.is_none(),
            },
        }
    })
}

fn empty_object_schema() -> Value {
    json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
}

impl<C: Send + Sync + 'static> FunctionTool<C> {
    /// Create a tool from an async closure taking typed arguments.
    ///
    /// `Args` must implement `schemars::JsonSchema` and `serde::Deserialize`; its schema is used
    /// as the tool parameters (converted to strict mode).
    ///
    /// # Panics
    ///
    /// Panics when the argument type's schema cannot be made strict, as Python's
    /// `function_tool` raises at definition time. Use [`try_new`](Self::try_new) to get the
    /// error instead.
    pub fn new<Args, F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        Args: JsonSchema + DeserializeOwned + Send + 'static,
        F: Fn(ToolContext<C>, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        Self::try_new(name, description, f).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Fallible form of [`new`](Self::new). Returns `UserError` when the argument type's schema
    /// cannot be made strict, instead of falling back to a non-strict tool.
    pub fn try_new<Args, F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Result<Self>
    where
        Args: JsonSchema + DeserializeOwned + Send + 'static,
        F: Fn(ToolContext<C>, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        Self::try_new_with_strictness(name, description, true, f)
    }

    /// Typed constructor with an explicit strictness flag, the analog of Python's
    /// `function_tool(strict_mode=...)`. With `strict` unset the schema is kept as generated, so an
    /// argument type such as a map is accepted; with it set the schema must be strict-convertible.
    pub fn try_new_with_strictness<Args, F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        strict: bool,
        f: F,
    ) -> Result<Self>
    where
        Args: JsonSchema + DeserializeOwned + Send + 'static,
        F: Fn(ToolContext<C>, Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        let mut schema = schemars::schema_for!(Args).to_value();
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
            m.remove("title");
        }
        let is_object = schema.get("type").and_then(Value::as_str) == Some("object")
            || schema.get("properties").is_some();
        if !is_object {
            schema = empty_object_schema();
        }
        let f = Arc::new(f);
        let constraint_schema = Arc::new(schema.clone());
        let invoke: InvokeFn<C> = Arc::new(move |ctx, args_json| {
            let f = f.clone();
            let constraint_schema = constraint_schema.clone();
            Box::pin(async move {
                let invalid_json = |payload: &str, e: serde_json::Error, tool_name: &str| {
                    invalid_tool_input_error(tool_name, payload, &e)
                };
                let args: Args = if args_json.trim().is_empty() {
                    serde_json::from_str("{}").map_err(|e| invalid_json("{}", e, &ctx.tool_name))?
                } else {
                    let value: Value = serde_json::from_str(&args_json)
                        .map_err(|e| invalid_json(&args_json, e, &ctx.tool_name))?;
                    if !value.is_object() {
                        return Err(invalid_tool_input_not_object(&ctx.tool_name));
                    }
                    // Python validates arguments with pydantic, which enforces the schema's value
                    // bounds; serde would accept any value of the right type.
                    crate::strict_schema::validate_value_constraints(&constraint_schema, &value)
                        .map_err(|message| {
                            invalid_json(
                                &args_json,
                                <serde_json::Error as serde::de::Error>::custom(message),
                                &ctx.tool_name,
                            )
                        })?;
                    serde_json::from_value(value)
                        .map_err(|e| invalid_json(&args_json, e, &ctx.tool_name))?
                };
                let out = f(ctx, args).await?;
                Ok(out.into_tool_output())
            })
        });
        let declared: Option<std::collections::BTreeSet<String>> = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|props| props.keys().cloned().collect());
        // The inspection never runs the argument type's `Deserialize`: that may be custom code
        // (a `deserialize_with` function, an application default), which belongs after the
        // approval. The arguments are judged against the declared schema instead.
        let inspection_schema = schema.clone();
        let approval_args_check: ApprovalArgsCheckFn = Arc::new(move |args: &Value| {
            let Value::Object(map) = args else {
                return false;
            };
            if let Some(declared) = &declared
                && (map.len() != declared.len() || !map.keys().all(|k| declared.contains(k)))
            {
                return false;
            }
            schema_shape_accepts(&inspection_schema, &inspection_schema, args, 0)
                && crate::strict_schema::validate_value_constraints(&inspection_schema, args)
                    .is_ok()
        });
        let mut tool = if strict {
            Self::build_strict(name.into(), description.into(), schema, invoke)?
        } else {
            Self::finish(
                name.into(),
                description.into(),
                schema.clone(),
                schema,
                false,
                invoke,
            )
        };
        tool.approval_args_check = Some(approval_args_check);
        Ok(tool)
    }

    /// Create a tool without parameters.
    pub fn new_no_args<F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        F: Fn(ToolContext<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        let f = Arc::new(f);
        let invoke: InvokeFn<C> = Arc::new(move |ctx, _args| {
            let f = f.clone();
            Box::pin(async move { Ok(f(ctx).await?.into_tool_output()) })
        });
        Self::build(
            name.into(),
            description.into(),
            empty_object_schema(),
            invoke,
        )
    }

    /// Create a tool from a raw JSON schema and a closure receiving the raw JSON arguments.
    pub fn new_raw<F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        f: F,
    ) -> Self
    where
        F: Fn(ToolContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        let f = Arc::new(f);
        let invoke: InvokeFn<C> = Arc::new(move |ctx, args| {
            let f = f.clone();
            Box::pin(async move { Ok(f(ctx, args).await?.into_tool_output()) })
        });
        Self::build(name.into(), description.into(), params_json_schema, invoke)
    }

    /// Fallible form of [`new_raw`](Self::new_raw) with an explicit strictness flag.
    ///
    /// With `strict` set, a schema that cannot be made strict is an error instead of a silent
    /// downgrade to a non-strict tool; with it unset the schema is kept exactly as given. Mirrors
    /// `FunctionTool(params_json_schema=..., strict_json_schema=...)`, whose constructor raises
    /// `UserError` in the strict case.
    pub fn try_new_raw<F, Fut, T>(
        name: impl Into<String>,
        description: impl Into<String>,
        params_json_schema: Value,
        strict: bool,
        f: F,
    ) -> Result<Self>
    where
        F: Fn(ToolContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: IntoToolOutput + Send + 'static,
    {
        let raw = params_json_schema.clone();
        let schema = if strict {
            ensure_strict_json_schema(&params_json_schema)?
        } else {
            params_json_schema
        };
        let f = Arc::new(f);
        let invoke: InvokeFn<C> = Arc::new(move |ctx, args| {
            let f = f.clone();
            Box::pin(async move { Ok(f(ctx, args).await?.into_tool_output()) })
        });
        Ok(Self::finish(
            name.into(),
            description.into(),
            schema,
            raw,
            strict,
            invoke,
        ))
    }

    /// Build a strict tool from a schema, failing when it cannot be made strict.
    fn build_strict(
        name: String,
        description: String,
        schema: Value,
        invoke: InvokeFn<C>,
    ) -> Result<Self> {
        let strict = ensure_strict_json_schema(&schema)?;
        Ok(Self::finish(
            name,
            description,
            strict,
            schema,
            true,
            invoke,
        ))
    }

    fn build(name: String, description: String, schema: Value, invoke: InvokeFn<C>) -> Self {
        let raw = schema.clone();
        let (schema, strict) = match ensure_strict_json_schema(&schema) {
            Ok(s) => (s, true),
            Err(_) => (schema, false),
        };
        Self::finish(name, description, schema, raw, strict, invoke)
    }

    fn finish(
        name: String,
        description: String,
        schema: Value,
        raw: Value,
        strict: bool,
        invoke: InvokeFn<C>,
    ) -> Self {
        FunctionTool {
            name,
            description,
            params_json_schema: schema,
            raw_params_json_schema: raw,
            strict_json_schema: strict,
            is_enabled: IsEnabled::Static(true),
            needs_approval: NeedsApproval::Static(false),
            timeout: None,
            timeout_behavior: ToolTimeoutBehavior::ErrorAsResult,
            tool_input_guardrails: Vec::new(),
            tool_output_guardrails: Vec::new(),
            defer_loading: false,
            failure_error_function: Some(tool_error_fn(default_tool_error_function::<C>)),
            uses_default_failure_error_function: true,
            default_failure_error_function: true,
            timeout_error_function: None,
            invoke,
            is_agent_tool: false,
            mcp_server_name: None,
            mcp_annotations: None,
            resumes_registered_capability: false,
            mcp_tool_binding: None,
            tool_origin: None,
            allowed_callers: None,
            custom_data_extractor: None,
            output_json_schema: None,
            output_validator: None,
            approval_args_check: None,
            args_have_validators: false,
            mcp_title: None,
            is_codex_tool: false,
            namespace: None,
            namespace_description: None,
        }
    }

    /// Set whether the parameters schema is strict. Turning strictness off restores the schema
    /// as given; turning it on converts it again, and leaves the tool as it is when the schema
    /// cannot be made strict.
    pub fn with_strict_json_schema(mut self, strict: bool) -> Self {
        if strict {
            if let Ok(schema) = ensure_strict_json_schema(&self.raw_params_json_schema) {
                self.params_json_schema = schema;
                self.strict_json_schema = true;
            }
        } else {
            self.params_json_schema = self.raw_params_json_schema.clone();
            self.strict_json_schema = false;
        }
        self
    }

    /// Replace the parameters schema. The new schema is the one a later
    /// [`with_strict_json_schema`](Self::with_strict_json_schema) call starts from.
    pub fn with_params_json_schema(mut self, schema: Value) -> Self {
        self.raw_params_json_schema = schema.clone();
        self.params_json_schema = schema;
        self
    }

    /// Set a static or dynamic enablement check.
    pub fn with_is_enabled(mut self, is_enabled: impl Into<IsEnabled<C>>) -> Self {
        self.is_enabled = is_enabled.into();
        self
    }

    /// Set a dynamic enablement check.
    pub fn with_is_enabled_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.is_enabled = IsEnabled::Dynamic(Arc::new(move |ctx, agent| Box::pin(f(ctx, agent))));
        self
    }

    /// Require approval (always, or never).
    pub fn with_needs_approval(mut self, needs_approval: impl Into<NeedsApproval<C>>) -> Self {
        self.needs_approval = needs_approval.into();
        self
    }

    /// Require approval based on a per-call decision.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, Value, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.needs_approval =
            NeedsApproval::Dynamic(Arc::new(move |ctx, args, id| Box::pin(f(ctx, args, id))));
        self
    }

    /// Evaluate the approval rule for one call.
    ///
    /// A dynamic rule sees only the invocation that can execute: when the arguments of a typed
    /// tool would not run unchanged (they fail to parse into its argument type, omit a parameter
    /// a default would fill, or carry an undeclared key), the rule is not consulted and the call
    /// requires approval. Mirrors `evaluate_function_tool_approval`.
    pub async fn evaluate_needs_approval(
        &self,
        ctx: &RunContext<C>,
        arguments: &str,
        call_id: &str,
    ) -> bool {
        if self.args_have_validators && matches!(self.needs_approval, NeedsApproval::Dynamic(_)) {
            return true;
        }
        if let (NeedsApproval::Dynamic(_), Some(check)) =
            (&self.needs_approval, &self.approval_args_check)
        {
            let parsed: Option<Value> = serde_json::from_str(arguments).ok();
            if !parsed.as_ref().is_some_and(|args| check(args)) {
                return true;
            }
        }
        self.needs_approval.evaluate(ctx, arguments, call_id).await
    }

    /// Declare that the argument type runs application code while it is parsed (a
    /// `deserialize_with` function, a custom `Deserialize`, a validating constructor).
    ///
    /// Rust cannot see such code, so the tool states it. A dynamic approval rule is then never
    /// consulted: every call pauses for a manual decision before any parsing runs, and the
    /// application's validation happens only after approval, then follows the tool's failure
    /// policy. This is the counterpart of Python skipping the rule for argument models with
    /// custom validators, validator plugins, `model_post_init` or default factories. Without the
    /// marker a dynamic rule is consulted for arguments that match the declared schema, and
    /// `Deserialize` never runs before approval. A static `needs_approval` is unaffected.
    pub fn with_args_have_validators(mut self, have_validators: bool) -> Self {
        self.args_have_validators = have_validators;
        self
    }

    /// Set the per-call timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the per-call timeout, rejecting a zero duration. Mirrors `FunctionTool`, whose
    /// `timeout_seconds` must be greater than 0.
    pub fn try_with_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(AgentsError::user("timeout_seconds must be greater than 0."));
        }
        self.timeout = Some(timeout);
        Ok(self)
    }

    /// Set the timeout behaviour.
    pub fn with_timeout_behavior(mut self, behavior: ToolTimeoutBehavior) -> Self {
        self.timeout_behavior = behavior;
        self
    }

    /// Customize the timeout message (used with `ErrorAsResult`).
    pub fn with_timeout_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.timeout_error_function = Some(tool_error_fn(f));
        self
    }

    /// Customize the timeout message with a handler that may await or fail
    /// (see [`ToolErrorFn`]).
    pub fn with_timeout_error_handler(mut self, f: ToolErrorFn<C>) -> Self {
        self.timeout_error_function = Some(f);
        self
    }

    /// Customize the model-visible message when the tool fails.
    pub fn with_failure_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync + 'static,
    {
        // An explicit `default_tool_error_function` keeps the default policy's trace redaction
        // (Python compares the handler's identity with `default_tool_error_function`).
        self.default_failure_error_function =
            std::any::TypeId::of::<F>() == type_id_of_val(&default_tool_error_function::<C>);
        self.failure_error_function = Some(tool_error_fn(f));
        self.uses_default_failure_error_function = false;
        self
    }

    /// Customize the model-visible message with a handler that may await or fail
    /// (see [`ToolErrorFn`]). An `Err` from the handler aborts the run.
    pub fn with_failure_error_handler(mut self, f: ToolErrorFn<C>) -> Self {
        self.default_failure_error_function = false;
        self.failure_error_function = Some(f);
        self.uses_default_failure_error_function = false;
        self
    }

    /// Re-raise tool errors instead of sending a message to the model.
    pub fn without_failure_error_function(mut self) -> Self {
        self.failure_error_function = None;
        self.uses_default_failure_error_function = false;
        self.default_failure_error_function = false;
        self
    }

    /// Add an input guardrail.
    pub fn with_input_guardrail(mut self, guardrail: ToolInputGuardrail<C>) -> Self {
        self.tool_input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail.
    pub fn with_output_guardrail(mut self, guardrail: ToolOutputGuardrail<C>) -> Self {
        self.tool_output_guardrails.push(guardrail);
        self
    }

    /// Hide the tool until loaded via hosted tool search.
    pub fn with_defer_loading(mut self, defer: bool) -> Self {
        self.defer_loading = defer;
        self
    }

    /// Declare the tool's output as a strict JSON object schema, so a programmatic caller
    /// knows its shape. Mirrors `function_tool(output_json_schema=...)`.
    ///
    /// Returns an error when the schema is not a JSON object schema or cannot be made strict.
    pub fn with_output_json_schema(self, schema: Value) -> Result<Self> {
        if self.output_validator.is_some() {
            return Err(AgentsError::user(
                "output_type and output_json_schema cannot both be provided.",
            ));
        }
        self.set_output_json_schema(schema)
    }

    fn set_output_json_schema(mut self, mut schema: Value) -> Result<Self> {
        // A JSON Schema type array holding only "object" means the same as "object".
        if let Some(types) = schema.get("type").and_then(Value::as_array)
            && types.len() == 1
            && types[0].as_str() == Some("object")
        {
            schema["type"] = Value::String("object".into());
        }
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(AgentsError::user(
                "Function tool output_json_schema must define a JSON object schema.",
            ));
        }
        let strict = crate::strict_schema::ensure_strict_json_schema(&schema).map_err(|e| {
            AgentsError::user(format!(
                "Function tool output_json_schema must define a strict JSON object schema: {e}"
            ))
        })?;
        self.output_json_schema = Some(strict);
        Ok(self)
    }

    /// Declare the tool's output type.
    ///
    /// The strict JSON object schema is derived from `T`'s [`schemars::JsonSchema`]
    /// implementation and reaches the wire as `output_schema`, exactly like
    /// [`with_output_json_schema`](Self::with_output_json_schema). In addition the tool's
    /// return value is validated against `T` before it reaches hooks, guardrails or the
    /// model. Rust equivalent of `function_tool(output_type=...)`: Rust has no return-type
    /// reflection, so the type is named rather than inferred from the closure.
    ///
    /// Returns an error when `T`'s schema is not a JSON object schema, as Python does for an
    /// `output_type` that is not a model, TypedDict or dataclass.
    pub fn with_output_type<T>(self) -> Result<Self>
    where
        T: JsonSchema + DeserializeOwned + 'static,
    {
        // The tool's result is serialized, so the schema uses serialization names (for example a
        // `#[serde(rename(serialize = ...))]` field), as Python's output schema does.
        let mut schema = schemars::generate::SchemaSettings::default()
            .for_serialize()
            .into_generator()
            .into_root_schema_for::<T>()
            .to_value();
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
        }
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(AgentsError::user(
                "Function tool output_type must define a strict JSON object schema. Use a \
                 struct deriving `schemars::JsonSchema`, or provide with_output_json_schema \
                 directly.",
            ));
        }
        if self.output_json_schema.is_some() {
            return Err(AgentsError::user(
                "output_type and output_json_schema cannot both be provided.",
            ));
        }
        let mut tool = self.set_output_json_schema(schema)?;
        tool.output_validator = Some(Arc::new(|value: &Value| {
            serde_json::from_value::<T>(value.clone())
                .map(|_| ())
                .map_err(|e| e.to_string())
        }));
        Ok(tool)
    }

    /// Mark this tool as produced by `codex_tool`, so an agent holding two same-named Codex
    /// tools fails fast. Mirrors setting `FunctionTool._is_codex_tool`.
    /// Where items produced by this tool come from. Mirrors `get_function_tool_origin`: an
    /// explicit origin wins, a tool served by an MCP server reports that server, and everything
    /// else is a plain function tool.
    pub fn tool_origin(&self) -> ToolOrigin {
        if let Some(origin) = &self.tool_origin {
            return origin.clone();
        }
        match &self.mcp_server_name {
            Some(server) => ToolOrigin::mcp(server.clone()),
            None => ToolOrigin::new(ToolOriginType::Function),
        }
    }

    pub(crate) fn mark_codex_tool(&mut self) {
        self.is_codex_tool = true;
    }

    /// Set the human-readable title of the MCP tool this one was derived from.
    /// Mirrors `function_tool(mcp_title=...)`.
    pub fn with_mcp_title(mut self, title: Option<String>) -> Self {
        self.mcp_title = title;
        self
    }

    /// Attach SDK-only data to this tool's output items.
    pub fn with_custom_data_extractor<F, Fut>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&FunctionToolCustomDataContext<'a, C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Value>> + Send + 'static,
    {
        self.custom_data_extractor = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Restrict which programmatic callers may invoke the tool (Responses API only).
    pub fn with_allowed_callers(mut self, callers: Vec<String>) -> Self {
        self.allowed_callers = Some(callers);
        self
    }
}

/// The error for a tool output that does not match its declared type. The detail echoes the
/// offending value, so it is dropped unless the logging policy allows tool data.
fn output_mismatch(tool_name: &str, detail: &str) -> AgentsError {
    let base = format!(
        "Function tool {tool_name} returned an output that does not match its declared output type"
    );
    AgentsError::user(if !crate::debug::dont_log_tool_data() {
        format!("{base}: {detail}")
    } else {
        base
    })
}

impl<C> FunctionTool<C> {
    /// Invoke the tool with raw JSON arguments (bypasses guardrails, approvals and timeouts).
    pub async fn invoke(&self, ctx: ToolContext<C>, arguments: String) -> Result<ToolOutput> {
        let name = ctx.tool_name.clone();
        let run_context = ctx.run_context().clone();
        match self.invoke_body(ctx, arguments).await {
            Ok(output) => {
                self.validate_output(&name, &output)?;
                Ok(output)
            }
            // A custom failure handler supplies the output in place of the error, as Python's
            // `on_invoke_tool` does. The default policy is applied by the runner instead.
            Err(error) if !self.uses_default_failure_error_function => {
                match &self.failure_error_function {
                    Some(handler) => {
                        let text = handler(&run_context, &error).await?;
                        self.validate_failure_text(&name, &text)?;
                        Ok(ToolOutput::Text(text))
                    }
                    None => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Run the tool body without checking the declared output type. The runner validates
    /// separately so a type mismatch fails the run instead of being handed to the tool's
    /// `failure_error_function`, which is how Python's `AgentsException` re-raise behaves.
    pub(crate) async fn invoke_body(
        &self,
        ctx: ToolContext<C>,
        arguments: String,
    ) -> Result<ToolOutput> {
        if !crate::debug::dont_log_tool_data() {
            log::debug!(target: crate::logger::LOG_TARGET, "Tool call args: {arguments}");
        }
        (self.invoke)(ctx, arguments).await
    }

    /// Check the output against the type declared with [`Self::with_output_type`].
    /// Port of `_validate_function_tool_output`: the validation error echoes the offending
    /// value, so it is dropped unless the logging policy allows tool data.
    pub(crate) fn validate_output(&self, tool_name: &str, output: &ToolOutput) -> Result<()> {
        self.check_output_value(tool_name, &output.to_json_value())
    }

    /// Check a custom failure handler's text against the declared output type. The handler's
    /// text is the tool's structured result, so it must be a JSON document that matches the type.
    pub(crate) fn validate_failure_text(&self, tool_name: &str, text: &str) -> Result<()> {
        if self.output_validator.is_none() {
            return Ok(());
        }
        match serde_json::from_str::<Value>(text) {
            Ok(value) => self.check_output_value(tool_name, &value),
            Err(error) => Err(output_mismatch(tool_name, &error.to_string())),
        }
    }

    fn check_output_value(&self, tool_name: &str, value: &Value) -> Result<()> {
        let Some(validate) = &self.output_validator else {
            return Ok(());
        };
        validate(value).map_err(|detail| output_mismatch(tool_name, &detail))
    }

    /// Convert to a Responses API `function` tool param.
    pub fn to_responses_param(&self) -> Value {
        self.to_spec().to_responses_param()
    }

    /// The namespace attached with [`tool_namespace`], if any.
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// `namespace.name` for namespaced tools, otherwise the name.
    pub fn qualified_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{ns}.{}", self.name),
            None => self.name.clone(),
        }
    }

    /// Convert to a Chat Completions `function` tool param.
    pub fn to_chat_param(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
                "strict": self.strict_json_schema,
            }
        })
    }

    /// The MCP server that provides this tool, if any.
    pub fn mcp_server_name(&self) -> Option<&str> {
        self.mcp_server_name.as_deref()
    }

    /// A copy of the annotations (`readOnlyHint`, `destructiveHint`, ...) the MCP server
    /// declared for this tool when it was converted. `None` for tools that are not MCP tools
    /// and for MCP tools without annotations. Mirrors `FunctionTool.mcp_annotations`.
    pub fn mcp_annotations(&self) -> Option<Value> {
        self.mcp_annotations.clone()
    }

    /// The tool spec exposed to models.
    pub fn to_spec(&self) -> crate::models::interface::FunctionToolSpec {
        crate::models::interface::FunctionToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.params_json_schema.clone(),
            strict: self.strict_json_schema,
            defer_loading: self.defer_loading,
            namespace: self.namespace.clone(),
            namespace_description: self.namespace_description.clone(),
            allowed_callers: self.allowed_callers.clone(),
            output_json_schema: self.output_json_schema.clone(),
            needs_approval: !matches!(self.needs_approval, NeedsApproval::Static(false)),
        }
    }
}

/// Group function tools under a namespace for Responses API tool search. The model sees one
/// `namespace` tool and calls its members as `namespace.name`.
pub fn tool_namespace<C>(
    name: &str,
    description: &str,
    tools: Vec<FunctionTool<C>>,
) -> Result<Vec<FunctionTool<C>>> {
    let name = name.trim();
    let description = description.trim();
    if name.is_empty() {
        return Err(AgentsError::user(
            "tool_namespace() requires a non-empty namespace name.",
        ));
    }
    if description.is_empty() {
        return Err(AgentsError::user(
            "tool_namespace() requires a non-empty description.",
        ));
    }
    tools
        .into_iter()
        .map(|mut t| {
            if t.name == name {
                return Err(AgentsError::user(format!(
                    "Responses tool-search reserves the synthetic namespace `{name}.{name}` for \
                     deferred top-level function tools. Rename the namespace or tool name to \
                     avoid ambiguous dispatch."
                )));
            }
            t.namespace = Some(name.to_string());
            t.namespace_description = Some(description.to_string());
            Ok(t)
        })
        .collect()
}

/// Hosted web search tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebSearchTool {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_location: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<Value>,
    /// `low`, `medium` (default) or `high`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_content_types: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_settings: Option<Value>,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to a Responses tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({"type": "web_search"});
        let m = v.as_object_mut().expect("object");
        if let Some(l) = &self.user_location {
            m.insert("user_location".into(), l.clone());
        }
        if let Some(f) = &self.filters {
            m.insert("filters".into(), f.clone());
        }
        m.insert(
            "search_context_size".into(),
            Value::String(
                self.search_context_size
                    .clone()
                    .unwrap_or_else(|| "medium".into()),
            ),
        );
        if let Some(e) = self.external_web_access {
            m.insert("external_web_access".into(), Value::Bool(e));
        }
        if let Some(t) = &self.search_content_types {
            m.insert("search_content_types".into(), json!(t));
        }
        if let Some(i) = &self.image_settings {
            m.insert("image_settings".into(), i.clone());
        }
        v
    }

    /// Whether image results were requested (adds `web_search_call.results` to `include`).
    pub fn wants_image_results(&self) -> bool {
        self.search_content_types
            .as_ref()
            .map(|t| t.iter().any(|s| s == "image"))
            .unwrap_or(false)
    }
}

/// Hosted file search tool over OpenAI vector stores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FileSearchTool {
    pub vector_store_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
    #[serde(default)]
    pub include_search_results: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<Value>,
}

impl FileSearchTool {
    pub fn new(vector_store_ids: Vec<String>) -> Self {
        FileSearchTool {
            vector_store_ids,
            ..Default::default()
        }
    }

    /// Convert to a Responses tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({"type": "file_search", "vector_store_ids": self.vector_store_ids});
        let m = v.as_object_mut().expect("object");
        if let Some(n) = self.max_num_results {
            m.insert("max_num_results".into(), json!(n));
        }
        if let Some(r) = &self.ranking_options {
            m.insert("ranking_options".into(), r.clone());
        }
        if let Some(f) = &self.filters {
            m.insert("filters".into(), f.clone());
        }
        v
    }
}

/// Hosted code interpreter tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeInterpreterTool {
    /// Raw tool config (`{"type": "code_interpreter", "container": {...}}`).
    pub tool_config: Value,
}

impl CodeInterpreterTool {
    /// Code interpreter with an auto container.
    pub fn new() -> Self {
        CodeInterpreterTool {
            tool_config: json!({"type": "code_interpreter", "container": {"type": "auto"}}),
        }
    }
    pub fn to_responses_param(&self) -> Value {
        self.tool_config.clone()
    }
}

impl Default for CodeInterpreterTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Hosted image generation tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationTool {
    /// Raw tool config (`{"type": "image_generation", ...}`).
    pub tool_config: Value,
}

impl ImageGenerationTool {
    pub fn new() -> Self {
        ImageGenerationTool {
            tool_config: json!({"type": "image_generation"}),
        }
    }
    pub fn to_responses_param(&self) -> Value {
        self.tool_config.clone()
    }
}

impl Default for ImageGenerationTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Approval request from a hosted MCP tool.
pub struct MCPToolApprovalRequest<C> {
    pub ctx: RunContext<C>,
    /// The raw `mcp_approval_request` item.
    pub data: Value,
}

impl<C> Clone for MCPToolApprovalRequest<C> {
    fn clone(&self) -> Self {
        MCPToolApprovalRequest {
            ctx: self.ctx.clone(),
            data: self.data.clone(),
        }
    }
}

impl<C> fmt::Debug for MCPToolApprovalRequest<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPToolApprovalRequest")
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

/// Decision for a hosted MCP approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MCPToolApprovalResult {
    pub approve: bool,
    pub reason: Option<String>,
}

type MCPApprovalFn<C> = Arc<
    dyn Fn(MCPToolApprovalRequest<C>) -> BoxFuture<'static, MCPToolApprovalResult> + Send + Sync,
>;

/// Hosted MCP server tool, executed by the Responses API.
pub struct HostedMCPTool<C> {
    /// Raw tool config (`{"type": "mcp", "server_label": ..., "server_url": ..., ...}`).
    pub tool_config: Value,
    on_approval_request: Option<MCPApprovalFn<C>>,
}

impl<C> Clone for HostedMCPTool<C> {
    fn clone(&self) -> Self {
        HostedMCPTool {
            tool_config: self.tool_config.clone(),
            on_approval_request: self.on_approval_request.clone(),
        }
    }
}

impl<C> fmt::Debug for HostedMCPTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostedMCPTool")
            .field("tool_config", &self.tool_config)
            .finish_non_exhaustive()
    }
}

impl<C> HostedMCPTool<C> {
    /// Create from a raw config. `type` is set to `mcp` if missing.
    pub fn new(mut tool_config: Value) -> Self {
        if let Value::Object(m) = &mut tool_config {
            m.entry("type").or_insert(Value::String("mcp".into()));
        }
        HostedMCPTool {
            tool_config,
            on_approval_request: None,
        }
    }

    /// Create for a public server URL.
    pub fn with_server(label: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(json!({
            "type": "mcp",
            "server_label": label.into(),
            "server_url": url.into(),
        }))
    }

    /// Set `require_approval` (`always`, `never`, or a per-tool object).
    pub fn with_require_approval(mut self, require_approval: Value) -> Self {
        self.tool_config["require_approval"] = require_approval;
        self
    }

    /// Decide approvals in code.
    pub fn with_on_approval_request<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(MCPToolApprovalRequest<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = MCPToolApprovalResult> + Send + 'static,
    {
        self.on_approval_request = Some(Arc::new(move |req| Box::pin(f(req))));
        self
    }

    /// The approval callback, if any.
    pub fn on_approval_request(&self) -> Option<&MCPApprovalFn<C>> {
        self.on_approval_request.as_ref()
    }

    /// The server label.
    pub fn server_label(&self) -> Option<&str> {
        self.tool_config.get("server_label").and_then(Value::as_str)
    }

    pub fn to_responses_param(&self) -> Value {
        self.tool_config.clone()
    }
}

/// A tool available to an agent.
pub enum Tool<C> {
    /// A function tool executed by the SDK.
    Function(Arc<FunctionTool<C>>),
    /// Hosted web search.
    WebSearch(WebSearchTool),
    /// Hosted file search.
    FileSearch(FileSearchTool),
    /// Hosted code interpreter.
    CodeInterpreter(CodeInterpreterTool),
    /// Hosted image generation.
    ImageGeneration(ImageGenerationTool),
    /// Hosted MCP server.
    HostedMCP(HostedMCPTool<C>),
    /// A computer controlled by the model.
    Computer(Arc<ComputerTool<C>>),
    /// The shell tool (local executor or hosted container).
    Shell(Arc<ShellTool<C>>),
    /// The legacy local shell tool.
    LocalShell(Arc<LocalShellTool<C>>),
    /// File edits through V4A diffs.
    ApplyPatch(Arc<ApplyPatchTool<C>>),
    /// A custom (raw string input) tool.
    Custom(Arc<CustomTool<C>>),
    /// Hosted tool search.
    ToolSearch(ToolSearchTool),
    /// Hosted programmatic tool calling.
    ProgrammaticToolCalling(ProgrammaticToolCallingTool),
    /// Any other Responses tool param, passed through verbatim.
    Raw(Value),
}

impl<C> Clone for Tool<C> {
    fn clone(&self) -> Self {
        match self {
            Tool::Function(t) => Tool::Function(t.clone()),
            Tool::WebSearch(t) => Tool::WebSearch(t.clone()),
            Tool::FileSearch(t) => Tool::FileSearch(t.clone()),
            Tool::CodeInterpreter(t) => Tool::CodeInterpreter(t.clone()),
            Tool::ImageGeneration(t) => Tool::ImageGeneration(t.clone()),
            Tool::HostedMCP(t) => Tool::HostedMCP(t.clone()),
            Tool::Computer(t) => Tool::Computer(t.clone()),
            Tool::Shell(t) => Tool::Shell(t.clone()),
            Tool::LocalShell(t) => Tool::LocalShell(t.clone()),
            Tool::ApplyPatch(t) => Tool::ApplyPatch(t.clone()),
            Tool::Custom(t) => Tool::Custom(t.clone()),
            Tool::ToolSearch(t) => Tool::ToolSearch(t.clone()),
            Tool::ProgrammaticToolCalling(t) => Tool::ProgrammaticToolCalling(*t),
            Tool::Raw(v) => Tool::Raw(v.clone()),
        }
    }
}

impl<C> fmt::Debug for Tool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tool::Function(t) => f.debug_tuple("Function").field(t).finish(),
            Tool::WebSearch(t) => f.debug_tuple("WebSearch").field(t).finish(),
            Tool::FileSearch(t) => f.debug_tuple("FileSearch").field(t).finish(),
            Tool::CodeInterpreter(t) => f.debug_tuple("CodeInterpreter").field(t).finish(),
            Tool::ImageGeneration(t) => f.debug_tuple("ImageGeneration").field(t).finish(),
            Tool::HostedMCP(t) => f.debug_tuple("HostedMCP").field(t).finish(),
            Tool::Computer(t) => f.debug_tuple("Computer").field(t).finish(),
            Tool::Shell(t) => f.debug_tuple("Shell").field(t).finish(),
            Tool::LocalShell(t) => f.debug_tuple("LocalShell").field(t).finish(),
            Tool::ApplyPatch(t) => f.debug_tuple("ApplyPatch").field(t).finish(),
            Tool::Custom(t) => f.debug_tuple("Custom").field(t).finish(),
            Tool::ToolSearch(t) => f.debug_tuple("ToolSearch").field(t).finish(),
            Tool::ProgrammaticToolCalling(t) => {
                f.debug_tuple("ProgrammaticToolCalling").field(t).finish()
            }
            Tool::Raw(t) => f.debug_tuple("Raw").field(t).finish(),
        }
    }
}

impl<C> Tool<C> {
    /// The tool name (hosted tools use their type).
    pub fn name(&self) -> String {
        match self {
            Tool::Function(t) => t.name.clone(),
            Tool::WebSearch(_) => "web_search".into(),
            Tool::FileSearch(_) => "file_search".into(),
            Tool::CodeInterpreter(_) => "code_interpreter".into(),
            Tool::ImageGeneration(_) => "image_generation".into(),
            Tool::HostedMCP(t) => t.server_label().unwrap_or("mcp").to_string(),
            Tool::Computer(t) => t.name().to_string(),
            Tool::Shell(t) => t.name.clone(),
            Tool::LocalShell(t) => t.name().to_string(),
            Tool::ApplyPatch(t) => t.name.clone(),
            Tool::Custom(t) => t.name.clone(),
            Tool::ToolSearch(_) => "tool_search".into(),
            Tool::ProgrammaticToolCalling(_) => "programmatic_tool_calling".into(),
            Tool::Raw(v) => v
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("raw")
                .to_string(),
        }
    }

    /// The name tracing displays: the GA `computer` alias for the computer tool (whose runtime
    /// name keeps the preview-era value for hooks and persisted state), the tool name otherwise.
    pub fn trace_name(&self) -> String {
        match self {
            Tool::Computer(_) => "computer".into(),
            other => other.name(),
        }
    }

    /// The function tool, if this is one.
    pub fn as_function(&self) -> Option<&Arc<FunctionTool<C>>> {
        match self {
            Tool::Function(t) => Some(t),
            _ => None,
        }
    }

    /// Convert to a Responses API tool param, plus any `include` values it needs.
    ///
    /// For a computer tool this resolves the computer synchronously only when it is already
    /// created; the runner uses [`ComputerTool::to_responses_param`] instead.
    pub fn to_responses_param(&self) -> (Value, Vec<String>) {
        match self {
            Tool::Function(t) => (t.to_responses_param(), Vec::new()),
            Tool::WebSearch(t) => {
                let includes = if t.wants_image_results() {
                    vec!["web_search_call.results".to_string()]
                } else {
                    Vec::new()
                };
                (t.to_responses_param(), includes)
            }
            Tool::FileSearch(t) => {
                let includes = if t.include_search_results {
                    vec!["file_search_call.results".to_string()]
                } else {
                    Vec::new()
                };
                (t.to_responses_param(), includes)
            }
            Tool::CodeInterpreter(t) => (
                t.to_responses_param(),
                vec!["code_interpreter_call.outputs".to_string()],
            ),
            Tool::ImageGeneration(t) => (t.to_responses_param(), Vec::new()),
            Tool::HostedMCP(t) => (t.to_responses_param(), Vec::new()),
            Tool::Computer(_) => (json!({"type": "computer"}), Vec::new()),
            Tool::Shell(t) => (t.to_responses_param(), Vec::new()),
            Tool::LocalShell(t) => (t.to_responses_param(), Vec::new()),
            Tool::ApplyPatch(t) => (t.to_responses_param(), Vec::new()),
            Tool::Custom(t) => (t.to_responses_param(), Vec::new()),
            Tool::ToolSearch(t) => (t.to_responses_param(), Vec::new()),
            Tool::ProgrammaticToolCalling(t) => (t.to_responses_param(), Vec::new()),
            Tool::Raw(v) => (v.clone(), Vec::new()),
        }
    }
}

impl<C> From<FunctionTool<C>> for Tool<C> {
    fn from(value: FunctionTool<C>) -> Self {
        Tool::Function(Arc::new(value))
    }
}

impl<C> From<Arc<FunctionTool<C>>> for Tool<C> {
    fn from(value: Arc<FunctionTool<C>>) -> Self {
        Tool::Function(value)
    }
}

impl<C> From<WebSearchTool> for Tool<C> {
    fn from(value: WebSearchTool) -> Self {
        Tool::WebSearch(value)
    }
}

impl<C> From<FileSearchTool> for Tool<C> {
    fn from(value: FileSearchTool) -> Self {
        Tool::FileSearch(value)
    }
}

impl<C> From<CodeInterpreterTool> for Tool<C> {
    fn from(value: CodeInterpreterTool) -> Self {
        Tool::CodeInterpreter(value)
    }
}

impl<C> From<ImageGenerationTool> for Tool<C> {
    fn from(value: ImageGenerationTool) -> Self {
        Tool::ImageGeneration(value)
    }
}

impl<C> From<HostedMCPTool<C>> for Tool<C> {
    fn from(value: HostedMCPTool<C>) -> Self {
        Tool::HostedMCP(value)
    }
}

macro_rules! tool_from_arc {
    ($($ty:ident => $variant:ident),* $(,)?) => {$(
        impl<C> From<$ty<C>> for Tool<C> {
            fn from(value: $ty<C>) -> Self {
                Tool::$variant(Arc::new(value))
            }
        }
        impl<C> From<Arc<$ty<C>>> for Tool<C> {
            fn from(value: Arc<$ty<C>>) -> Self {
                Tool::$variant(value)
            }
        }
    )*};
}

tool_from_arc!(
    ComputerTool => Computer,
    ShellTool => Shell,
    LocalShellTool => LocalShell,
    ApplyPatchTool => ApplyPatch,
    CustomTool => Custom,
);

impl<C> From<ToolSearchTool> for Tool<C> {
    fn from(value: ToolSearchTool) -> Self {
        Tool::ToolSearch(value)
    }
}

impl<C> From<ProgrammaticToolCallingTool> for Tool<C> {
    fn from(value: ProgrammaticToolCallingTool) -> Self {
        Tool::ProgrammaticToolCalling(value)
    }
}

/// Build a function tool from a typed async closure. Shorthand for [`FunctionTool::new`].
pub fn function_tool<C, Args, F, Fut, T>(
    name: impl Into<String>,
    description: impl Into<String>,
    f: F,
) -> FunctionTool<C>
where
    C: Send + Sync + 'static,
    Args: JsonSchema + DeserializeOwned + Send + 'static,
    F: Fn(ToolContext<C>, Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
    T: IntoToolOutput + Send + 'static,
{
    FunctionTool::new(name, description, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    struct WeatherArgs {
        /// The city to look up.
        city: String,
    }

    #[tokio::test]
    async fn builds_strict_schema_and_invokes() {
        let tool: FunctionTool<()> = FunctionTool::new(
            "get_weather",
            "Get weather",
            |_ctx, args: WeatherArgs| async move { Ok(format!("sunny in {}", args.city)) },
        );
        assert!(tool.strict_json_schema);
        assert_eq!(
            tool.params_json_schema["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(
            tool.params_json_schema["properties"]["city"]["description"],
            "The city to look up."
        );
        let ctx = ToolContext::direct(RunContext::new(()), "agent", "get_weather", "{}");
        let out = tool
            .invoke(ctx, r#"{"city":"Tokyo"}"#.into())
            .await
            .unwrap();
        assert_eq!(out.to_model_string(), "sunny in Tokyo");
        let param = tool.to_responses_param();
        assert_eq!(param["type"], "function");
        assert_eq!(param["strict"], true);
    }

    #[tokio::test]
    async fn no_args_tool() {
        let tool: FunctionTool<()> =
            FunctionTool::new_no_args("ping", "Ping", |_ctx| async { Ok("pong") });
        assert_eq!(tool.params_json_schema["type"], "object");
        let ctx = ToolContext::direct(RunContext::new(()), "agent", "ping", "");
        assert_eq!(
            tool.invoke(ctx, "".into()).await.unwrap().to_model_string(),
            "pong"
        );
    }
}
