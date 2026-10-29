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
pub type IsEnabledFn<C> = Arc<dyn Fn(RunContext<C>, String) -> BoxFuture<'static, bool> + Send + Sync>;

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
pub type ToolErrorFn<C> = Arc<dyn Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync>;

/// The default message sent to the model when a tool fails.
pub fn default_tool_error_function<C>(_ctx: &RunContext<C>, error: &AgentsError) -> String {
    format!("An error occurred while running the tool. Please try again. Error: {error}")
}

type InvokeFn<C> =
    Arc<dyn Fn(ToolContext<C>, String) -> BoxFuture<'static, Result<ToolOutput>> + Send + Sync>;

/// A tool that wraps a Rust async function.
pub struct FunctionTool<C> {
    /// Tool name shown to the model.
    pub name: String,
    /// Tool description shown to the model.
    pub description: String,
    /// JSON schema of the parameters.
    pub params_json_schema: Value,
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
    /// Timeout message formatter.
    pub timeout_error_function: Option<ToolErrorFn<C>>,
    invoke: InvokeFn<C>,
    pub(crate) is_agent_tool: bool,
    pub(crate) mcp_server_name: Option<String>,
}

impl<C> Clone for FunctionTool<C> {
    fn clone(&self) -> Self {
        FunctionTool {
            name: self.name.clone(),
            description: self.description.clone(),
            params_json_schema: self.params_json_schema.clone(),
            strict_json_schema: self.strict_json_schema,
            is_enabled: self.is_enabled.clone(),
            needs_approval: self.needs_approval.clone(),
            timeout: self.timeout,
            timeout_behavior: self.timeout_behavior,
            tool_input_guardrails: self.tool_input_guardrails.clone(),
            tool_output_guardrails: self.tool_output_guardrails.clone(),
            defer_loading: self.defer_loading,
            failure_error_function: self.failure_error_function.clone(),
            timeout_error_function: self.timeout_error_function.clone(),
            invoke: self.invoke.clone(),
            is_agent_tool: self.is_agent_tool,
            mcp_server_name: self.mcp_server_name.clone(),
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

fn empty_object_schema() -> Value {
    json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
}

impl<C: Send + Sync + 'static> FunctionTool<C> {
    /// Create a tool from an async closure taking typed arguments.
    ///
    /// `Args` must implement `schemars::JsonSchema` and `serde::Deserialize`; its schema is used
    /// as the tool parameters (converted to strict mode).
    pub fn new<Args, F, Fut, T>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
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
        let invoke: InvokeFn<C> = Arc::new(move |ctx, args_json| {
            let f = f.clone();
            Box::pin(async move {
                let args: Args = if args_json.trim().is_empty() {
                    serde_json::from_str("{}").map_err(|e| {
                        AgentsError::model_behavior(format!(
                            "Invalid JSON input for tool {}: {e}",
                            ctx.tool_name
                        ))
                    })?
                } else {
                    serde_json::from_str(&args_json).map_err(|e| {
                        AgentsError::model_behavior(format!(
                            "Invalid JSON input for tool {}: {e}",
                            ctx.tool_name
                        ))
                    })?
                };
                let out = f(ctx, args).await?;
                Ok(out.into_tool_output())
            })
        });
        Self::build(name.into(), description.into(), schema, invoke)
    }

    /// Create a tool without parameters.
    pub fn new_no_args<F, Fut, T>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
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
        Self::build(name.into(), description.into(), empty_object_schema(), invoke)
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

    fn build(name: String, description: String, schema: Value, invoke: InvokeFn<C>) -> Self {
        let (schema, strict) = match ensure_strict_json_schema(&schema) {
            Ok(s) => (s, true),
            Err(_) => (schema, false),
        };
        FunctionTool {
            name,
            description,
            params_json_schema: schema,
            strict_json_schema: strict,
            is_enabled: IsEnabled::Static(true),
            needs_approval: NeedsApproval::Static(false),
            timeout: None,
            timeout_behavior: ToolTimeoutBehavior::ErrorAsResult,
            tool_input_guardrails: Vec::new(),
            tool_output_guardrails: Vec::new(),
            defer_loading: false,
            failure_error_function: Some(Arc::new(default_tool_error_function::<C>)),
            timeout_error_function: None,
            invoke,
            is_agent_tool: false,
            mcp_server_name: None,
        }
    }

    /// Use a non-strict schema (keeps the schema as provided).
    pub fn with_strict_json_schema(mut self, strict: bool) -> Self {
        self.strict_json_schema = strict;
        self
    }

    /// Replace the parameters schema.
    pub fn with_params_json_schema(mut self, schema: Value) -> Self {
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

    /// Set the per-call timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
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
        self.timeout_error_function = Some(Arc::new(f));
        self
    }

    /// Customize the model-visible message when the tool fails.
    pub fn with_failure_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.failure_error_function = Some(Arc::new(f));
        self
    }

    /// Re-raise tool errors instead of sending a message to the model.
    pub fn without_failure_error_function(mut self) -> Self {
        self.failure_error_function = None;
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

}

impl<C> FunctionTool<C> {
    /// Invoke the tool with raw JSON arguments (bypasses guardrails, approvals and timeouts).
    pub async fn invoke(&self, ctx: ToolContext<C>, arguments: String) -> Result<ToolOutput> {
        (self.invoke)(ctx, arguments).await
    }

    /// Convert to a Responses API `function` tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params_json_schema,
            "strict": self.strict_json_schema,
        });
        if self.defer_loading {
            v["defer_loading"] = Value::Bool(true);
        }
        v
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

    /// The tool spec exposed to models.
    pub fn to_spec(&self) -> crate::models::interface::FunctionToolSpec {
        crate::models::interface::FunctionToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.params_json_schema.clone(),
            strict: self.strict_json_schema,
            defer_loading: self.defer_loading,
        }
    }
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

type MCPApprovalFn<C> =
    Arc<dyn Fn(MCPToolApprovalRequest<C>) -> BoxFuture<'static, MCPToolApprovalResult> + Send + Sync>;

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
    /// Any other Responses tool param, passed through verbatim (e.g. hosted shell).
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
            Tool::Raw(v) => v
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("raw")
                .to_string(),
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
        assert_eq!(tool.params_json_schema["properties"]["city"]["type"], "string");
        assert_eq!(
            tool.params_json_schema["properties"]["city"]["description"],
            "The city to look up."
        );
        let ctx = ToolContext::direct(RunContext::new(()), "agent", "get_weather", "{}");
        let out = tool.invoke(ctx, r#"{"city":"Tokyo"}"#.into()).await.unwrap();
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
        assert_eq!(tool.invoke(ctx, "".into()).await.unwrap().to_model_string(), "pong");
    }
}
