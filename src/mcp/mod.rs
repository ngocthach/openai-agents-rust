//! Model Context Protocol servers (feature `mcp`). Mirrors `agents.mcp`.
//!
//! Local MCP servers run their tools through the SDK: the runner lists the server's tools on
//! each run and calls them as function tools. Supported transports: stdio
//! ([`MCPServerStdio`]), Streamable HTTP ([`MCPServerStreamableHttp`]) and the deprecated
//! HTTP+SSE transport ([`MCPServerSse`]). For OpenAI-hosted MCP, use
//! [`crate::tool::HostedMCPTool`] instead.

mod handler;
mod http;
pub mod logging;
mod sse;
mod stdio;
pub mod util;

pub use logging::{
    MCPToolMetadata, collect_mcp_list_tools_metadata, extract_mcp_tool_metadata,
    extract_mcp_tool_metadata_value, get_mcp_server_log_message, get_mcp_server_log_name,
    redact_credentials, resolve_mcp_tool_description, resolve_mcp_tool_description_for_model,
    resolve_mcp_tool_description_for_model_value, resolve_mcp_tool_description_value,
    resolve_mcp_tool_title, resolve_mcp_tool_title_value,
};

use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use rmcp::service::RunningService;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::error::{AgentsError, Result};

/// A tool advertised by an MCP server.
#[derive(Debug, Clone, PartialEq)]
pub struct MCPTool {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    /// JSON schema of the arguments.
    pub input_schema: Value,
    /// Advisory annotations (`readOnlyHint`, `destructiveHint`, ...).
    pub annotations: Option<Value>,
    /// The tool's static `_meta`, sent as request metadata on every call (Python's
    /// `MCPUtil._extract_static_meta`). Each call gets its own copy, so a server that mutates the
    /// metadata it receives cannot change what the next call sends.
    pub meta: Option<Map<String, Value>>,
}

impl MCPTool {
    fn from_value(v: &Value) -> Option<Self> {
        Some(MCPTool {
            name: v.get("name")?.as_str()?.to_string(),
            title: v.get("title").and_then(Value::as_str).map(str::to_string),
            description: v
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            input_schema: v
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            annotations: v.get("annotations").cloned(),
            meta: v.get("_meta").and_then(Value::as_object).cloned(),
        })
    }
}

/// Context passed to a dynamic tool filter. Mirrors `agents.mcp.util.ToolFilterContext`.
///
/// `run_context` and `agent` are type-erased because [`MCPServer`] is not generic over the run
/// context type: making it generic would change every `Arc<dyn MCPServer>` in the crate.
/// Python types both fields `Any` for the same reason, and
/// [`ToolFilterContext::run_context`] / [`ToolFilterContext::agent`] recover the concrete type.
#[derive(Clone)]
pub struct ToolFilterContext {
    pub agent_name: String,
    pub server_name: String,
    /// The active [`crate::run_context::RunContext<C>`], when the caller supplied one.
    pub run_context_any: Option<Arc<dyn Any + Send + Sync>>,
    /// The agent whose tools are being listed, when the caller supplied it.
    pub agent_any: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for ToolFilterContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolFilterContext")
            .field("agent_name", &self.agent_name)
            .field("server_name", &self.server_name)
            .field("run_context", &self.run_context_any.is_some())
            .field("agent", &self.agent_any.is_some())
            .finish()
    }
}

impl ToolFilterContext {
    /// A context with no run context and no agent attached.
    pub fn new(agent_name: impl Into<String>, server_name: impl Into<String>) -> Self {
        ToolFilterContext {
            agent_name: agent_name.into(),
            server_name: server_name.into(),
            run_context_any: None,
            agent_any: None,
        }
    }

    /// The run context, when it was supplied and `C` is the run's context type.
    pub fn run_context<C: Send + Sync + 'static>(
        &self,
    ) -> Option<&crate::run_context::RunContext<C>> {
        self.run_context_any
            .as_ref()?
            .downcast_ref::<crate::run_context::RunContext<C>>()
    }

    /// The agent, when it was supplied and `C` is the run's context type.
    pub fn agent<C: Send + Sync + 'static>(&self) -> Option<&Arc<crate::agent::Agent<C>>> {
        self.agent_any
            .as_ref()?
            .downcast_ref::<Arc<crate::agent::Agent<C>>>()
    }
}

type DynamicFilter =
    Arc<dyn Fn(&ToolFilterContext, &MCPTool) -> BoxFuture<'static, bool> + Send + Sync>;

/// Which tools of a server are exposed.
#[derive(Clone, Default)]
pub enum ToolFilter {
    /// Expose all tools.
    #[default]
    All,
    /// Allow-list first, then block-list.
    Static {
        allowed_tool_names: Option<Vec<String>>,
        blocked_tool_names: Option<Vec<String>>,
    },
    /// Decide per tool.
    Dynamic(DynamicFilter),
}

impl fmt::Debug for ToolFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolFilter::All => write!(f, "All"),
            ToolFilter::Static {
                allowed_tool_names,
                blocked_tool_names,
            } => f
                .debug_struct("Static")
                .field("allowed_tool_names", allowed_tool_names)
                .field("blocked_tool_names", blocked_tool_names)
                .finish(),
            ToolFilter::Dynamic(_) => write!(f, "Dynamic(..)"),
        }
    }
}

/// Build a static allow/block filter.
pub fn create_static_tool_filter(
    allowed_tool_names: Option<Vec<String>>,
    blocked_tool_names: Option<Vec<String>>,
) -> ToolFilter {
    ToolFilter::Static {
        allowed_tool_names,
        blocked_tool_names,
    }
}

impl ToolFilter {
    /// A dynamic filter from a synchronous predicate.
    pub fn dynamic<F>(f: F) -> Self
    where
        F: Fn(&ToolFilterContext, &MCPTool) -> bool + Send + Sync + 'static,
    {
        ToolFilter::Dynamic(Arc::new(move |ctx, tool| {
            let allowed = f(ctx, tool);
            Box::pin(async move { allowed })
        }))
    }

    /// A dynamic filter that may await, as Python's `ToolFilterCallable` may be a coroutine.
    pub fn dynamic_async<F, Fut>(f: F) -> Self
    where
        F: Fn(&ToolFilterContext, &MCPTool) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        ToolFilter::Dynamic(Arc::new(move |ctx, tool| Box::pin(f(ctx, tool))))
    }

    async fn allows(&self, ctx: &ToolFilterContext, tool: &MCPTool) -> bool {
        match self {
            ToolFilter::All => true,
            ToolFilter::Static {
                allowed_tool_names,
                blocked_tool_names,
            } => {
                if let Some(a) = allowed_tool_names
                    && !a.contains(&tool.name)
                {
                    return false;
                }
                if let Some(b) = blocked_tool_names
                    && b.contains(&tool.name)
                {
                    return false;
                }
                true
            }
            ToolFilter::Dynamic(f) => f(ctx, tool).await,
        }
    }
}

type ApprovalFn = Arc<dyn Fn(&str, &MCPTool) -> BoxFuture<'static, bool> + Send + Sync>;

/// Approval policy for local MCP tools.
#[derive(Clone, Default)]
pub enum RequireApproval {
    /// Never require approval (default).
    #[default]
    Never,
    /// Always require approval.
    Always,
    /// Per-tool policy: `true` requires approval. Unlisted tools do not.
    PerTool(HashMap<String, bool>),
    /// Decide per tool from `(agent_name, tool)`.
    Dynamic(ApprovalFn),
}

impl fmt::Debug for RequireApproval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequireApproval::Never => write!(f, "Never"),
            RequireApproval::Always => write!(f, "Always"),
            RequireApproval::PerTool(m) => f.debug_tuple("PerTool").field(m).finish(),
            RequireApproval::Dynamic(_) => write!(f, "Dynamic(..)"),
        }
    }
}

impl RequireApproval {
    /// A per-call policy from a synchronous predicate.
    pub fn dynamic<F>(f: F) -> Self
    where
        F: Fn(&str, &MCPTool) -> bool + Send + Sync + 'static,
    {
        RequireApproval::Dynamic(Arc::new(move |agent_name, tool| {
            let needed = f(agent_name, tool);
            Box::pin(async move { needed })
        }))
    }

    /// A per-call policy that may await, as Python's approval callable may be a coroutine.
    pub fn dynamic_async<F, Fut>(f: F) -> Self
    where
        F: Fn(&str, &MCPTool) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        RequireApproval::Dynamic(Arc::new(move |agent_name, tool| {
            Box::pin(f(agent_name, tool))
        }))
    }

    /// Whether `tool` requires approval for `agent_name`.
    pub async fn requires(&self, agent_name: &str, tool: &MCPTool) -> bool {
        match self {
            RequireApproval::Never => false,
            RequireApproval::Always => true,
            RequireApproval::PerTool(m) => m.get(&tool.name).copied().unwrap_or(false),
            RequireApproval::Dynamic(f) => f(agent_name, tool).await,
        }
    }
}

/// What a [`MCPToolMetaResolver`] is told about the call it is producing `_meta` for.
#[derive(Debug, Clone)]
pub struct MCPToolMetaContext {
    /// The server's name, as configured.
    pub server_name: String,
    /// The tool's name on the server (never the exposed function-tool name).
    pub tool_name: String,
    /// The arguments the model produced, when the call has any.
    pub arguments: Option<Map<String, Value>>,
}

/// Produces protocol-level request metadata (`_meta`) for a tool call.
///
/// The SDK calls it before `call_tool`; returning `None` sends no `_meta`.
pub type MCPToolMetaResolver =
    Arc<dyn Fn(&MCPToolMetaContext) -> Option<Map<String, Value>> + Send + Sync>;

/// A notification delivered by an MCP server, as raw JSON.
///
/// Mirrors Python's `message_handler`, which receives whatever the session delivers.
pub type MCPMessageHandler = Arc<dyn Fn(&Value) + Send + Sync>;

/// Per-server tool policy: the settings that depend on the run context type.
///
/// Python keeps `failure_error_function`, `tool_input_guardrails`, `tool_output_guardrails` and
/// `custom_data_extractor` directly on the server, typed `Any`. [`MCPServer`] is not generic
/// over `C`, so this port keeps them in one typed struct that
/// [`MCPServerConfig::tool_policy`] stores type-erased; the tool builder recovers it with a
/// downcast for the concrete `C` of the run.
pub struct MCPToolPolicy<C> {
    /// Turns a failure of any tool on this server into a model-visible message.
    ///
    /// `None` re-raises. Leave [`MCPToolPolicy::failure_error_function_set`] `false` to inherit
    /// the agent's `MCPConfig::failure_error_function`, which is Python's `_UNSET` default.
    pub failure_error_function: Option<crate::tool::ToolErrorFn<C>>,
    /// Whether `failure_error_function` was set explicitly (Python's `_UNSET` sentinel).
    pub failure_error_function_set: bool,
    /// Guardrails applied to every tool on this server.
    pub tool_input_guardrails: Vec<crate::tool_guardrails::ToolInputGuardrail<C>>,
    /// Guardrails applied to every tool result from this server.
    pub tool_output_guardrails: Vec<crate::tool_guardrails::ToolOutputGuardrail<C>>,
    /// Produces SDK-only custom data for every tool on this server.
    pub custom_data_extractor: Option<crate::tool::CustomDataExtractor<C>>,
}

impl<C> Default for MCPToolPolicy<C> {
    fn default() -> Self {
        MCPToolPolicy {
            failure_error_function: None,
            failure_error_function_set: false,
            tool_input_guardrails: Vec::new(),
            tool_output_guardrails: Vec::new(),
            custom_data_extractor: None,
        }
    }
}

impl<C> Clone for MCPToolPolicy<C> {
    fn clone(&self) -> Self {
        MCPToolPolicy {
            failure_error_function: self.failure_error_function.clone(),
            failure_error_function_set: self.failure_error_function_set,
            tool_input_guardrails: self.tool_input_guardrails.clone(),
            tool_output_guardrails: self.tool_output_guardrails.clone(),
            custom_data_extractor: self.custom_data_extractor.clone(),
        }
    }
}

impl<C> fmt::Debug for MCPToolPolicy<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPToolPolicy")
            .field(
                "failure_error_function",
                &self.failure_error_function.is_some(),
            )
            .field(
                "failure_error_function_set",
                &self.failure_error_function_set,
            )
            .field("tool_input_guardrails", &self.tool_input_guardrails.len())
            .field("tool_output_guardrails", &self.tool_output_guardrails.len())
            .field(
                "custom_data_extractor",
                &self.custom_data_extractor.is_some(),
            )
            .finish()
    }
}

impl<C: Send + Sync + 'static> MCPToolPolicy<C> {
    /// An empty policy.
    pub fn new() -> Self {
        MCPToolPolicy::default()
    }

    /// Set the failure formatter (Python's explicit `failure_error_function=`).
    pub fn with_failure_error_function(mut self, f: Option<crate::tool::ToolErrorFn<C>>) -> Self {
        self.failure_error_function = f;
        self.failure_error_function_set = true;
        self
    }

    /// Apply `guardrails` to every tool on the server.
    pub fn with_tool_input_guardrails(
        mut self,
        guardrails: Vec<crate::tool_guardrails::ToolInputGuardrail<C>>,
    ) -> Self {
        self.tool_input_guardrails = guardrails;
        self
    }

    /// Apply `guardrails` to every tool result from the server.
    pub fn with_tool_output_guardrails(
        mut self,
        guardrails: Vec<crate::tool_guardrails::ToolOutputGuardrail<C>>,
    ) -> Self {
        self.tool_output_guardrails = guardrails;
        self
    }

    /// Attach SDK-only custom data to every tool result from the server.
    pub fn with_custom_data_extractor(
        mut self,
        extractor: Option<crate::tool::CustomDataExtractor<C>>,
    ) -> Self {
        self.custom_data_extractor = extractor;
        self
    }
}

/// Settings shared by local MCP servers.
#[derive(Clone)]
pub struct MCPServerConfig {
    /// Cache `list_tools` results (call `invalidate_tools_cache` to refresh).
    pub cache_tools_list: bool,
    /// Which tools are exposed.
    pub tool_filter: ToolFilter,
    /// Approval policy.
    pub require_approval: RequireApproval,
    /// Timeout of each MCP request.
    pub client_session_timeout: Option<Duration>,
    /// Retries for `list_tools` and `call_tool`. A negative value retries forever, as Python's
    /// `max_retry_attempts = -1` does.
    pub max_retry_attempts: i32,
    /// Base backoff between retries.
    pub retry_backoff_seconds_base: f64,
    /// Upper bound on the backoff between retries (`None` leaves it uncapped).
    pub retry_backoff_seconds_max: Option<f64>,
    /// Maximum successful pages fetched by one automatic `list_tools`/`list_prompts` call.
    ///
    /// `None`, the default, applies no page limit. A list that reaches the limit on its final
    /// page succeeds; otherwise the call fails rather than returning a partial list.
    pub max_list_pages: Option<usize>,
    /// Prefer `structuredContent` over content blocks.
    pub use_structured_content: bool,
    /// Optional source of protocol-level request metadata (`_meta`) for tool calls.
    pub tool_meta_resolver: Option<MCPToolMetaResolver>,
    /// Called for every notification the server sends. Mirrors `message_handler`.
    pub message_handler: Option<MCPMessageHandler>,
    /// A type-erased [`MCPToolPolicy<C>`]; set it with
    /// [`MCPServerConfig::with_tool_policy`].
    pub tool_policy: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for MCPServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPServerConfig")
            .field("cache_tools_list", &self.cache_tools_list)
            .field("tool_filter", &self.tool_filter)
            .field("require_approval", &self.require_approval)
            .field("client_session_timeout", &self.client_session_timeout)
            .field("max_retry_attempts", &self.max_retry_attempts)
            .field(
                "retry_backoff_seconds_base",
                &self.retry_backoff_seconds_base,
            )
            .field("retry_backoff_seconds_max", &self.retry_backoff_seconds_max)
            .field("max_list_pages", &self.max_list_pages)
            .field("use_structured_content", &self.use_structured_content)
            .field(
                "tool_meta_resolver",
                &self.tool_meta_resolver.as_ref().map(|_| "Fn(..)"),
            )
            .field("message_handler", &self.message_handler.is_some())
            .field("tool_policy", &self.tool_policy.is_some())
            .finish()
    }
}

impl MCPServerConfig {
    /// Store the per-server tool policy for runs whose context type is `C`.
    pub fn with_tool_policy<C: Send + Sync + 'static>(mut self, policy: MCPToolPolicy<C>) -> Self {
        self.tool_policy = Some(Arc::new(policy));
        self
    }

    /// The per-server tool policy, when one was set for context type `C`.
    pub fn tool_policy<C: Send + Sync + 'static>(&self) -> Option<&MCPToolPolicy<C>> {
        self.tool_policy
            .as_ref()?
            .downcast_ref::<MCPToolPolicy<C>>()
    }
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        MCPServerConfig {
            cache_tools_list: false,
            tool_filter: ToolFilter::All,
            require_approval: RequireApproval::Never,
            client_session_timeout: Some(Duration::from_secs(5)),
            max_retry_attempts: 0,
            retry_backoff_seconds_base: 1.0,
            retry_backoff_seconds_max: None,
            max_list_pages: None,
            use_structured_content: false,
            tool_meta_resolver: None,
            message_handler: None,
            tool_policy: None,
        }
    }
}

/// A Model Context Protocol server.
#[async_trait]
pub trait MCPServer: Send + Sync + fmt::Debug {
    /// Readable server name.
    fn name(&self) -> &str;
    /// Server settings.
    fn config(&self) -> &MCPServerConfig;
    /// Connect (spawn the process / open the session).
    async fn connect(&self) -> Result<()>;
    /// Close the connection.
    async fn cleanup(&self) -> Result<()>;
    /// All tools advertised by the server (unfiltered).
    async fn list_tools(&self) -> Result<Vec<MCPTool>>;
    /// Call a tool; returns the `CallToolResult` as JSON.
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
    ) -> Result<Value>;
    /// Call a tool, attaching protocol-level request metadata (`_meta`).
    ///
    /// The default implementation delegates to [`MCPServer::call_tool`] when `meta` is `None`
    /// and reports that the server cannot carry `_meta` otherwise.
    async fn call_tool_with_meta(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
        meta: Option<Map<String, Value>>,
    ) -> Result<Value> {
        if meta.is_some() {
            return Err(AgentsError::user(format!(
                "MCP server '{}' does not support request metadata (_meta).",
                get_mcp_server_log_name(self.name())
            )));
        }
        self.call_tool(tool_name, arguments).await
    }
    /// The most recently fetched tool list, when one is cached.
    ///
    /// Returns `None` when tools were never fetched or caching is disabled.
    async fn cached_tools(&self) -> Option<Vec<MCPTool>> {
        None
    }
    /// The server's `initialize` result, available once the server is connected.
    async fn server_initialize_result(&self) -> Option<Value> {
        None
    }
    /// The transport-level session id, for transports that have one.
    async fn session_id(&self) -> Option<String> {
        None
    }
    /// List the resources the server exposes.
    ///
    /// `cursor` is an opaque pagination cursor returned as `nextCursor` by a previous call;
    /// `None` fetches the first page. Servers that do not support resources return an error.
    async fn list_resources(&self, _cursor: Option<&str>) -> Result<Value> {
        Err(AgentsError::user(format!(
            "MCP server '{}' does not support list_resources.",
            get_mcp_server_log_name(self.name())
        )))
    }
    /// List the resource templates the server exposes. See [`MCPServer::list_resources`].
    async fn list_resource_templates(&self, _cursor: Option<&str>) -> Result<Value> {
        Err(AgentsError::user(format!(
            "MCP server '{}' does not support list_resource_templates.",
            get_mcp_server_log_name(self.name())
        )))
    }
    /// Read the contents of one resource by URI.
    async fn read_resource(&self, _uri: &str) -> Result<Value> {
        Err(AgentsError::user(format!(
            "MCP server '{}' does not support read_resource.",
            get_mcp_server_log_name(self.name())
        )))
    }
    /// List prompt templates.
    async fn list_prompts(&self) -> Result<Vec<Value>>;
    /// Fetch a prompt.
    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<BTreeMap<String, String>>,
    ) -> Result<Value>;
    /// Drop cached tools.
    async fn invalidate_tools_cache(&self) {}
}

#[derive(Debug, Clone)]
enum TransportSpec {
    Stdio(MCPServerStdioParams),
    StreamableHttp(MCPServerStreamableHttpParams),
    Sse(MCPServerSseParams),
}

/// Produces the bearer token sent as `Authorization: Bearer <token>` on every HTTP MCP request.
///
/// Python passes an `httpx.Auth` object; the Rust equivalent is a callback consulted when the
/// transport is built, so a short-lived credential can be refreshed by reconnecting. A value
/// that already starts with `Bearer ` is accepted and not doubled up.
pub type MCPAuthProvider = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// The bare token of a credential that may or may not already carry the `Bearer ` prefix.
pub(crate) fn bearer_token(value: &str) -> &str {
    value.strip_prefix("Bearer ").unwrap_or(value)
}

/// Parameters for an HTTP+SSE MCP server.
#[derive(Clone)]
pub struct MCPServerSseParams {
    /// The SSE endpoint URL.
    pub url: String,
    /// Extra headers sent on the SSE request and on every POST.
    pub headers: HashMap<String, String>,
    /// Connection timeout (default 5 seconds).
    pub timeout: Duration,
    /// How long to wait for a new SSE event (default 5 minutes).
    pub sse_read_timeout: Duration,
    /// Supplies the `Authorization` header. Mirrors the `auth` parameter.
    pub auth: Option<MCPAuthProvider>,
    /// A caller-built HTTP client, in place of the one the transport would build.
    /// Mirrors `httpx_client_factory`.
    pub http_client: Option<reqwest::Client>,
}

impl fmt::Debug for MCPServerSseParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPServerSseParams")
            .field("url", &get_mcp_server_log_name(&self.url))
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .field("timeout", &self.timeout)
            .field("sse_read_timeout", &self.sse_read_timeout)
            .field("auth", &self.auth.is_some())
            .field("http_client", &self.http_client.is_some())
            .finish()
    }
}

impl Default for MCPServerSseParams {
    fn default() -> Self {
        MCPServerSseParams {
            url: String::new(),
            headers: HashMap::new(),
            timeout: Duration::from_secs(5),
            sse_read_timeout: Duration::from_secs(300),
            auth: None,
            http_client: None,
        }
    }
}

/// What to do with bytes a stdio server writes that are not valid UTF-8.
///
/// Mirrors Python's `encoding_error_handler`. The MCP spec requires UTF-8 on stdio, so the
/// `encoding` parameter itself has no Rust counterpart (see [`MCPServerStdioParams::encoding`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EncodingErrorHandler {
    /// Fail the connection when the server writes invalid UTF-8 (Python's default).
    #[default]
    Strict,
    /// Drop the invalid bytes.
    Ignore,
    /// Replace each invalid sequence with U+FFFD.
    Replace,
}

/// Parameters for a stdio MCP server.
#[derive(Debug, Clone)]
pub struct MCPServerStdioParams {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<String>,
    /// The text encoding of the server's stream. Only `utf-8` is accepted: the MCP stdio
    /// transport mandates UTF-8 and Rust strings are UTF-8, so any other value is a
    /// configuration error rather than a silent transcoding. Mirrors Python's `encoding`,
    /// which also defaults to `utf-8`.
    pub encoding: String,
    /// How invalid UTF-8 from the server is handled.
    pub encoding_error_handler: EncodingErrorHandler,
}

impl Default for MCPServerStdioParams {
    fn default() -> Self {
        MCPServerStdioParams {
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
            encoding: "utf-8".to_string(),
            encoding_error_handler: EncodingErrorHandler::Strict,
        }
    }
}

/// Parameters for a Streamable HTTP MCP server.
#[derive(Clone)]
pub struct MCPServerStreamableHttpParams {
    pub url: String,
    /// Extra headers. An `Authorization: Bearer <token>` header is used as the bearer token.
    pub headers: HashMap<String, String>,
    /// Timeout of one HTTP request (default 5 seconds).
    pub timeout: Duration,
    /// How long to wait for a new SSE event on a response stream (default 5 minutes).
    pub sse_read_timeout: Duration,
    /// Send `DELETE <url>` to end the server-side session on close (default `true`).
    pub terminate_on_close: bool,
    /// Treat a failed `notifications/initialized` POST as success, for servers that reject it.
    pub ignore_initialized_notification_failure: bool,
    /// Supplies the `Authorization` header. Mirrors the `auth` parameter.
    pub auth: Option<MCPAuthProvider>,
    /// A caller-built HTTP client. Mirrors `httpx_client_factory`.
    pub http_client: Option<reqwest::Client>,
}

impl fmt::Debug for MCPServerStreamableHttpParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPServerStreamableHttpParams")
            .field("url", &get_mcp_server_log_name(&self.url))
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .field("timeout", &self.timeout)
            .field("sse_read_timeout", &self.sse_read_timeout)
            .field("terminate_on_close", &self.terminate_on_close)
            .field(
                "ignore_initialized_notification_failure",
                &self.ignore_initialized_notification_failure,
            )
            .field("auth", &self.auth.is_some())
            .field("http_client", &self.http_client.is_some())
            .finish()
    }
}

impl Default for MCPServerStreamableHttpParams {
    fn default() -> Self {
        MCPServerStreamableHttpParams {
            url: String::new(),
            headers: HashMap::new(),
            timeout: Duration::from_secs(5),
            sse_read_timeout: Duration::from_secs(300),
            terminate_on_close: true,
            ignore_initialized_notification_failure: false,
            auth: None,
            http_client: None,
        }
    }
}

/// One live connection to a server.
///
/// `child` is only set for a stdio server whose decoder is not
/// [`EncodingErrorHandler::Strict`]: that path spawns the process here so the output can be
/// sanitized before the JSON-RPC codec sees it, which also means this port owns the child's
/// lifetime instead of `rmcp`.
struct Connection {
    service: RunningService<RoleClient, handler::MCPClientHandler>,
    child: Option<tokio::process::Child>,
}

impl Connection {
    async fn close(self) {
        let _ = self.service.cancel().await;
        if let Some(mut child) = self.child {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

struct Core {
    name: String,
    transport: TransportSpec,
    config: MCPServerConfig,
    connection: Mutex<Option<Connection>>,
    tools_cache: Mutex<Option<Vec<MCPTool>>>,
    session_id: Arc<std::sync::Mutex<Option<String>>>,
}

/// Turn an `rmcp` failure into an `AgentsError`, removing any credential it embeds.
///
/// A transport error carries the request URL, which is exactly where an MCP credential lives.
/// This is the Rust counterpart of Python's `_credential_safe_exception_group`.
fn svc_err(e: impl fmt::Display) -> AgentsError {
    AgentsError::other(redact_credentials(&format!("MCP error: {e}")))
}

/// Whether a failed request should be retried on a freshly opened session.
///
/// Port of `_should_retry_in_isolated_session`: a transport that was cancelled, closed, timed
/// out or failed to send is worth a second session; a server-side JSON-RPC error is not.
fn should_retry_in_isolated_session(e: &rmcp::ServiceError) -> bool {
    matches!(
        e,
        rmcp::ServiceError::TransportSend(_)
            | rmcp::ServiceError::TransportClosed
            | rmcp::ServiceError::Cancelled { .. }
            | rmcp::ServiceError::Timeout { .. }
    )
}

/// A failed `call_tool` attempt, and whether the failure came from the isolated retry.
enum CallFailure {
    /// The shared session failed and no isolated retry was attempted.
    Shared(AgentsError),
    /// The isolated retry was attempted and also failed.
    Isolated(AgentsError),
}

impl CallFailure {
    fn into_error(self) -> AgentsError {
        match self {
            CallFailure::Shared(e) | CallFailure::Isolated(e) => e,
        }
    }
}

impl Core {
    /// Open a new connection to the server, without touching the shared one.
    async fn build_connection(&self) -> Result<Connection> {
        let client = handler::MCPClientHandler::new(self.config.message_handler.clone());
        let connection = match &self.transport {
            TransportSpec::Stdio(p) => {
                if !stdio::is_utf8_encoding(&p.encoding) {
                    return Err(AgentsError::user(format!(
                        "MCP stdio server '{}' requested encoding '{}'. The MCP stdio transport \
                         is UTF-8 only.",
                        self.log_name(),
                        p.encoding
                    )));
                }
                if p.encoding_error_handler == EncodingErrorHandler::Strict {
                    let p = p.clone();
                    let cmd = tokio::process::Command::new(&p.command).configure(move |c| {
                        c.args(&p.args);
                        c.envs(&p.env);
                        if let Some(cwd) = &p.cwd {
                            c.current_dir(cwd);
                        }
                    });
                    let transport = TokioChildProcess::new(cmd)?;
                    Connection {
                        service: client.serve(transport).await.map_err(svc_err)?,
                        child: None,
                    }
                } else {
                    let mut command = tokio::process::Command::new(&p.command);
                    command
                        .args(&p.args)
                        .envs(&p.env)
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .kill_on_drop(true);
                    if let Some(cwd) = &p.cwd {
                        command.current_dir(cwd);
                    }
                    let mut child = command.spawn()?;
                    let stdin = child.stdin.take().ok_or_else(|| {
                        AgentsError::other("MCP stdio server has no stdin pipe".to_string())
                    })?;
                    let stdout = child.stdout.take().ok_or_else(|| {
                        AgentsError::other("MCP stdio server has no stdout pipe".to_string())
                    })?;
                    let reader = stdio::Utf8Sanitizer::new(stdout, p.encoding_error_handler);
                    Connection {
                        service: client.serve((reader, stdin)).await.map_err(svc_err)?,
                        child: Some(child),
                    }
                }
            }
            TransportSpec::StreamableHttp(p) => {
                let mut config =
                    rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(
                        p.url.clone(),
                    );
                let mut custom = HashMap::new();
                for (k, v) in &p.headers {
                    if k.eq_ignore_ascii_case("authorization")
                        && let Some(token) = v.strip_prefix("Bearer ")
                    {
                        config = config.auth_header(token.to_string());
                        continue;
                    }
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| AgentsError::user(format!("Invalid header {k}: {e}")))?;
                    let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                        AgentsError::user(format!("Invalid header value for {k}: {e}"))
                    })?;
                    custom.insert(name, value);
                }
                config = config.custom_headers(custom);
                if let Some(auth) = &p.auth
                    && let Some(value) = auth()
                {
                    config = config.auth_header(bearer_token(&value).to_string());
                }
                let http = match &p.http_client {
                    Some(c) => c.clone(),
                    None => {
                        crate::util::ensure_crypto_provider();
                        reqwest::Client::builder()
                            .connect_timeout(p.timeout)
                            .read_timeout(p.sse_read_timeout)
                            .build()?
                    }
                };
                let client_wrapper = http::ObservedHttpClient::new(
                    http,
                    self.session_id.clone(),
                    p.terminate_on_close,
                    p.ignore_initialized_notification_failure,
                );
                let transport = StreamableHttpClientTransport::with_client(client_wrapper, config);
                Connection {
                    service: client.serve(transport).await.map_err(svc_err)?,
                    child: None,
                }
            }
            TransportSpec::Sse(p) => {
                let transport = sse::connect(p).await?;
                Connection {
                    service: client.serve(transport).await.map_err(svc_err)?,
                    child: None,
                }
            }
        };
        Ok(connection)
    }

    async fn connect(&self) -> Result<()> {
        let mut guard = self.connection.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        *guard = Some(self.build_connection().await?);
        Ok(())
    }

    async fn cleanup(&self) -> Result<()> {
        if let Some(c) = self.connection.lock().await.take() {
            c.close().await;
        }
        if let Ok(mut slot) = self.session_id.lock() {
            *slot = None;
        }
        Ok(())
    }

    /// The server name with URL credentials, query and fragment removed.
    fn log_name(&self) -> String {
        get_mcp_server_log_name(&self.name)
    }

    fn timeout_error(&self) -> AgentsError {
        AgentsError::other(format!("MCP request to '{}' timed out", self.log_name()))
    }

    async fn with_timeout<T>(
        &self,
        fut: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        match self.config.client_session_timeout {
            Some(d) if !d.is_zero() => tokio::time::timeout(d, fut)
                .await
                .unwrap_or_else(|_| Err(self.timeout_error())),
            _ => fut.await,
        }
    }

    async fn session(&self) -> Result<tokio::sync::MutexGuard<'_, Option<Connection>>> {
        let guard = self.connection.lock().await;
        if guard.is_none() {
            return Err(AgentsError::user(
                "Server not initialized. Make sure you call `connect()` first.".to_string(),
            ));
        }
        Ok(guard)
    }

    /// Whether the retry budget is unlimited (Python's `max_retry_attempts == -1`).
    fn unlimited_retries(&self) -> bool {
        self.config.max_retry_attempts < 0
    }

    /// Wait out the backoff for the `n`-th retry already taken.
    async fn backoff(&self, taken: u32) {
        let mut delay = self.config.retry_backoff_seconds_base * 2f64.powi(taken as i32);
        if let Some(max) = self.config.retry_backoff_seconds_max {
            delay = delay.min(max);
        }
        tokio::time::sleep(Duration::from_secs_f64(delay.max(0.0))).await;
    }

    fn log_retry(&self, e: &AgentsError) {
        crate::logger::log_tool_action_warning(
            &get_mcp_server_log_message("Retrying failed request to MCP server", &self.name),
            e,
        );
    }

    async fn retry<T, F, Fut>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt: i32 = 0;
        loop {
            match self.with_timeout(f()).await {
                Ok(v) => return Ok(v),
                Err(e) if self.unlimited_retries() || attempt < self.config.max_retry_attempts => {
                    self.log_retry(&e);
                    self.backoff(attempt.max(0) as u32).await;
                    attempt = attempt.saturating_add(1);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Decide whether a paginated list continues, bounded by `max_list_pages`.
    ///
    /// Returns the next cursor, or `None` when the listing is complete. A server that repeats a
    /// cursor, or one that still has pages left when the page budget runs out, fails the call
    /// rather than returning a partial list.
    ///
    /// `operation` is `tools` or `prompts`; the messages are the ones Python raises, which
    /// deliberately name neither the cursor nor the page payload.
    fn check_pagination(
        &self,
        next: Option<String>,
        previous: Option<String>,
        pages: usize,
        operation: &str,
    ) -> Result<Option<String>> {
        let Some(next) = next.filter(|c| !c.is_empty()) else {
            return Ok(None);
        };
        if previous.as_deref() == Some(next.as_str()) {
            return Err(AgentsError::user(format!(
                "MCP server '{}' returned a repeated cursor while listing {operation}.",
                self.log_name()
            )));
        }
        if self.config.max_list_pages.is_some_and(|max| pages >= max) {
            let subject = if operation == "tools" {
                "MCP tool listing"
            } else {
                "MCP prompt listing"
            };
            return Err(AgentsError::user(format!(
                "{subject} exceeded max_list_pages. Increase max_list_pages to allow more \
                 pages, or check the server pagination."
            )));
        }
        Ok(Some(next))
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        // Python checks for a session before it consults the cache, so a disconnected server
        // never answers from a stale list.
        drop(self.session().await?);
        if self.config.cache_tools_list
            && let Some(t) = self.tools_cache.lock().await.clone()
        {
            return Ok(t);
        }
        // Each page request retries on its own, as Python's `_list_tools_page` does, so a
        // transient failure on page N never refetches (or recounts) the pages already read.
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;
        loop {
            let page_cursor = cursor.clone();
            let page = self
                .retry(|| async {
                    let guard = self.session().await?;
                    let s = &guard.as_ref().expect("session checked").service;
                    s.list_tools(page_cursor.clone().map(|c| {
                        rmcp::model::PaginatedRequestParams::default().with_cursor(Some(c))
                    }))
                    .await
                    .map_err(svc_err)
                })
                .await?;
            pages += 1;
            out.extend(
                page.tools
                    .iter()
                    .filter_map(|t| serde_json::to_value(t).ok())
                    .filter_map(|v| MCPTool::from_value(&v)),
            );
            let next = page.next_cursor.map(|c| c.to_string());
            match self.check_pagination(next, cursor, pages, "tools")? {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        let tools = out;
        if self.config.cache_tools_list {
            *self.tools_cache.lock().await = Some(tools.clone());
        }
        Ok(tools)
    }

    /// Reject a call that omits a parameter the cached tool schema requires, before any request
    /// is sent. Port of `_validate_required_parameters`: it only applies while the tool list is
    /// cached and the tool is in it.
    async fn validate_required_parameters(
        &self,
        tool_name: &str,
        arguments: Option<&Map<String, Value>>,
    ) -> Result<()> {
        let cache = self.tools_cache.lock().await;
        let Some(tool) = cache
            .as_ref()
            .and_then(|tools| tools.iter().find(|t| t.name == tool_name))
        else {
            return Ok(());
        };
        let Some(required) = tool
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .filter(|r| !r.is_empty())
        else {
            return Ok(());
        };
        let mut missing: Vec<&str> = required
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| arguments.is_none_or(|a| !a.contains_key(*name)))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        Err(AgentsError::user(format!(
            "Failed to call tool '{tool_name}' on MCP server '{}': missing required parameters: {}",
            self.log_name(),
            missing.join(", ")
        )))
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
        meta: Option<Map<String, Value>>,
    ) -> Result<Value> {
        self.validate_required_parameters(tool_name, arguments.as_ref())
            .await?;
        let mut req = json!({"name": tool_name});
        if let Some(a) = arguments {
            req["arguments"] = Value::Object(a);
        }
        if let Some(m) = meta {
            req["_meta"] = Value::Object(m);
        }
        let params: rmcp::model::CallToolRequestParams = serde_json::from_value(req)?;

        // Port of `call_tool`'s retry loop: the budget counts retries, not backoffs, and a
        // failed isolated retry charges the budget on its own.
        let unlimited = self.unlimited_retries();
        let mut retries_used: i32 = 0;
        let mut backoffs_taken: u32 = 0;
        let mut first_attempt = true;
        loop {
            if !first_attempt && !unlimited {
                retries_used += 1;
            }
            let allow_isolated_retry = unlimited || retries_used < self.config.max_retry_attempts;
            match self.call_tool_once(&params, allow_isolated_retry).await {
                Ok(value) => return Ok(value),
                Err(failure) => {
                    if matches!(failure, CallFailure::Isolated(_)) {
                        retries_used += 1;
                    }
                    if !unlimited && retries_used >= self.config.max_retry_attempts {
                        return Err(failure.into_error());
                    }
                    let error = failure.into_error();
                    self.log_retry(&error);
                    self.backoff(backoffs_taken).await;
                    backoffs_taken = backoffs_taken.saturating_add(1);
                }
            }
            first_attempt = false;
        }
    }

    /// One `call_tool` attempt on the shared session, with an optional retry on a fresh one.
    ///
    /// Port of `_call_tool_with_isolated_retry`: a transport failure can mean the shared
    /// session itself is unusable, so the call is replayed once against a connection opened
    /// only for it, which is then closed. The shared session is left untouched either way.
    async fn call_tool_once(
        &self,
        params: &rmcp::model::CallToolRequestParams,
        allow_isolated_retry: bool,
    ) -> std::result::Result<Value, CallFailure> {
        /// What the shared session produced, before the isolated retry is decided.
        enum SharedOutcome {
            Done(rmcp::model::CallToolResult),
            NeedsIsolation,
            Failed(AgentsError),
        }

        let attempt = async {
            let guard = match self.session().await {
                Ok(guard) => guard,
                Err(e) => return SharedOutcome::Failed(e),
            };
            let s = &guard.as_ref().expect("session checked").service;
            match s.call_tool(params.clone()).await {
                Ok(result) => SharedOutcome::Done(result),
                Err(e) if allow_isolated_retry && should_retry_in_isolated_session(&e) => {
                    SharedOutcome::NeedsIsolation
                }
                Err(e) => SharedOutcome::Failed(svc_err(e)),
            }
        };
        let outcome = match self.config.client_session_timeout {
            Some(d) if !d.is_zero() => match tokio::time::timeout(d, attempt).await {
                Ok(outcome) => outcome,
                // A client-side timeout is one of Python's isolation triggers too.
                Err(_) if allow_isolated_retry => SharedOutcome::NeedsIsolation,
                Err(_) => SharedOutcome::Failed(self.timeout_error()),
            },
            _ => attempt.await,
        };
        match outcome {
            SharedOutcome::Done(result) => {
                return serde_json::to_value(result)
                    .map_err(|e| CallFailure::Shared(AgentsError::from(e)));
            }
            SharedOutcome::Failed(e) => return Err(CallFailure::Shared(e)),
            SharedOutcome::NeedsIsolation => {}
        }
        let connection = self
            .build_connection()
            .await
            .map_err(CallFailure::Isolated)?;
        let outcome = self
            .with_timeout(async {
                connection
                    .service
                    .call_tool(params.clone())
                    .await
                    .map_err(svc_err)
            })
            .await;
        connection.close().await;
        match outcome {
            Ok(result) => serde_json::to_value(result)
                .map_err(|e| CallFailure::Isolated(AgentsError::from(e))),
            Err(e) => Err(CallFailure::Isolated(e)),
        }
    }

    async fn list_prompts(&self) -> Result<Vec<Value>> {
        let mut out: Vec<Value> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;
        loop {
            let guard = self.session().await?;
            let s = &guard.as_ref().expect("session checked").service;
            let params = cursor
                .clone()
                .map(|c| rmcp::model::PaginatedRequestParams::default().with_cursor(Some(c)));
            let page = self
                .with_timeout(async { s.list_prompts(params).await.map_err(svc_err) })
                .await?;
            pages += 1;
            out.extend(
                page.prompts
                    .iter()
                    .filter_map(|p| serde_json::to_value(p).ok()),
            );
            let next = page.next_cursor.map(|c| c.to_string());
            drop(guard);
            match self.check_pagination(next, cursor, pages, "prompts")? {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    async fn list_resources(&self, cursor: Option<&str>) -> Result<Value> {
        let guard = self.session().await?;
        let s = &guard.as_ref().expect("session checked").service;
        let params = cursor.map(|c| {
            rmcp::model::PaginatedRequestParams::default().with_cursor(Some(c.to_string()))
        });
        let result = self
            .with_timeout(async { s.list_resources(params).await.map_err(svc_err) })
            .await?;
        Ok(serde_json::to_value(result)?)
    }

    async fn list_resource_templates(&self, cursor: Option<&str>) -> Result<Value> {
        let guard = self.session().await?;
        let s = &guard.as_ref().expect("session checked").service;
        let params = cursor.map(|c| {
            rmcp::model::PaginatedRequestParams::default().with_cursor(Some(c.to_string()))
        });
        let result = self
            .with_timeout(async { s.list_resource_templates(params).await.map_err(svc_err) })
            .await?;
        Ok(serde_json::to_value(result)?)
    }

    async fn read_resource(&self, uri: &str) -> Result<Value> {
        let params: rmcp::model::ReadResourceRequestParams =
            serde_json::from_value(json!({"uri": uri}))?;
        let guard = self.session().await?;
        let s = &guard.as_ref().expect("session checked").service;
        let result = self
            .with_timeout(async { s.read_resource(params).await.map_err(svc_err) })
            .await?;
        Ok(serde_json::to_value(result)?)
    }

    async fn server_initialize_result(&self) -> Option<Value> {
        let guard = self.connection.lock().await;
        let info = guard.as_ref()?.service.peer_info()?;
        serde_json::to_value(&*info).ok()
    }

    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<BTreeMap<String, String>>,
    ) -> Result<Value> {
        let mut req = json!({"name": name});
        if let Some(a) = arguments {
            req["arguments"] = json!(a);
        }
        let params: rmcp::model::GetPromptRequestParams = serde_json::from_value(req)?;
        let guard = self.session().await?;
        let s = &guard.as_ref().expect("session checked").service;
        let r = self
            .with_timeout(async { s.get_prompt(params).await.map_err(svc_err) })
            .await?;
        Ok(serde_json::to_value(r)?)
    }
}

macro_rules! mcp_server_type {
    ($ty:ident, $params:ident, $variant:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $ty {
            core: Core,
        }

        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($ty))
                    .field("name", &self.core.name)
                    .field("config", &self.core.config)
                    .finish_non_exhaustive()
            }
        }

        impl $ty {
            /// Create the server with default settings.
            pub fn new(name: impl Into<String>, params: $params) -> Self {
                $ty {
                    core: Core {
                        name: name.into(),
                        transport: TransportSpec::$variant(params),
                        config: MCPServerConfig::default(),
                        connection: Mutex::new(None),
                        tools_cache: Mutex::new(None),
                        session_id: Arc::new(std::sync::Mutex::new(None)),
                    },
                }
            }

            /// Replace all settings.
            pub fn with_config(mut self, config: MCPServerConfig) -> Self {
                self.core.config = config;
                self
            }

            /// Cache the tool list.
            pub fn with_cache_tools_list(mut self, cache: bool) -> Self {
                self.core.config.cache_tools_list = cache;
                self
            }

            /// Expose a subset of tools.
            pub fn with_tool_filter(mut self, filter: ToolFilter) -> Self {
                self.core.config.tool_filter = filter;
                self
            }

            /// Approval policy.
            pub fn with_require_approval(mut self, policy: RequireApproval) -> Self {
                self.core.config.require_approval = policy;
                self
            }

            /// Timeout of each request (`None` disables it).
            pub fn with_client_session_timeout(mut self, timeout: Option<Duration>) -> Self {
                self.core.config.client_session_timeout = timeout;
                self
            }

            /// Retry `list_tools` and `call_tool`. A negative value retries forever.
            pub fn with_max_retry_attempts(mut self, attempts: i32) -> Self {
                self.core.config.max_retry_attempts = attempts;
                self
            }

            /// Receive every notification the server sends. Mirrors `message_handler`.
            pub fn with_message_handler<F>(mut self, handler: F) -> Self
            where
                F: Fn(&Value) + Send + Sync + 'static,
            {
                self.core.config.message_handler = Some(Arc::new(handler));
                self
            }

            /// Apply the per-server tool policy for runs whose context type is `C`.
            pub fn with_tool_policy<C: Send + Sync + 'static>(
                mut self,
                policy: MCPToolPolicy<C>,
            ) -> Self {
                self.core.config.tool_policy = Some(Arc::new(policy));
                self
            }

            /// Cap the exponential retry backoff (`None` leaves it uncapped).
            pub fn with_retry_backoff_seconds_max(mut self, seconds: Option<f64>) -> Self {
                self.core.config.retry_backoff_seconds_max = seconds;
                self
            }

            /// Bound the pages fetched by one automatic `list_tools`/`list_prompts` call.
            pub fn with_max_list_pages(mut self, pages: Option<usize>) -> Self {
                self.core.config.max_list_pages = pages;
                self
            }

            /// Prefer structured content in tool results.
            pub fn with_use_structured_content(mut self, use_structured: bool) -> Self {
                self.core.config.use_structured_content = use_structured;
                self
            }

            /// Produce protocol-level request metadata (`_meta`) for every tool call.
            pub fn with_tool_meta_resolver<F>(mut self, resolver: F) -> Self
            where
                F: Fn(&MCPToolMetaContext) -> Option<Map<String, Value>> + Send + Sync + 'static,
            {
                self.core.config.tool_meta_resolver = Some(Arc::new(resolver));
                self
            }
        }

        #[async_trait]
        impl MCPServer for $ty {
            fn name(&self) -> &str {
                &self.core.name
            }
            fn config(&self) -> &MCPServerConfig {
                &self.core.config
            }
            async fn connect(&self) -> Result<()> {
                self.core.connect().await
            }
            async fn cleanup(&self) -> Result<()> {
                self.core.cleanup().await
            }
            async fn list_tools(&self) -> Result<Vec<MCPTool>> {
                self.core.list_tools().await
            }
            async fn call_tool(
                &self,
                tool_name: &str,
                arguments: Option<Map<String, Value>>,
            ) -> Result<Value> {
                self.core.call_tool(tool_name, arguments, None).await
            }
            async fn call_tool_with_meta(
                &self,
                tool_name: &str,
                arguments: Option<Map<String, Value>>,
                meta: Option<Map<String, Value>>,
            ) -> Result<Value> {
                self.core.call_tool(tool_name, arguments, meta).await
            }
            async fn cached_tools(&self) -> Option<Vec<MCPTool>> {
                self.core.tools_cache.lock().await.clone()
            }
            async fn server_initialize_result(&self) -> Option<Value> {
                self.core.server_initialize_result().await
            }
            async fn session_id(&self) -> Option<String> {
                self.core.session_id.lock().ok()?.clone()
            }
            async fn list_resources(&self, cursor: Option<&str>) -> Result<Value> {
                self.core.list_resources(cursor).await
            }
            async fn list_resource_templates(&self, cursor: Option<&str>) -> Result<Value> {
                self.core.list_resource_templates(cursor).await
            }
            async fn read_resource(&self, uri: &str) -> Result<Value> {
                self.core.read_resource(uri).await
            }
            async fn list_prompts(&self) -> Result<Vec<Value>> {
                self.core.list_prompts().await
            }
            async fn get_prompt(
                &self,
                name: &str,
                arguments: Option<BTreeMap<String, String>>,
            ) -> Result<Value> {
                self.core.get_prompt(name, arguments).await
            }
            async fn invalidate_tools_cache(&self) {
                *self.core.tools_cache.lock().await = None;
            }
        }
    };
}

mcp_server_type!(
    MCPServerStdio,
    MCPServerStdioParams,
    Stdio,
    "An MCP server launched as a subprocess and spoken to over stdin/stdout."
);
mcp_server_type!(
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
    StreamableHttp,
    "An MCP server reached over the Streamable HTTP transport."
);
mcp_server_type!(
    MCPServerSse,
    MCPServerSseParams,
    Sse,
    "An MCP server reached over the deprecated HTTP+SSE transport."
);

/// Connects several servers and exposes the ones that connected.
///
/// Python runs every server on its own worker task because anyio requires a transport's
/// connect and cleanup to happen in the same task. tokio and `rmcp` have no such requirement,
/// so connect and cleanup run directly here; parallel connect is expressed with `join_all`
/// rather than a per-server command queue.
///
/// The lifecycle rules match `agents.mcp.manager.MCPServerManager`: a server that is already
/// connected is not connected again, a retry cleans the previous attempt up first and skips a
/// server whose cleanup failed, and a strict connect failure cleans up everything it opened
/// before returning.
#[derive(Debug)]
pub struct MCPServerManager {
    servers: Vec<Arc<dyn MCPServer>>,
    active: Vec<Arc<dyn MCPServer>>,
    failed: Vec<(Arc<dyn MCPServer>, String)>,
    cleanup_errors: Vec<(Arc<dyn MCPServer>, String)>,
    /// Identities (by `Arc` pointer) of the servers whose `connect` succeeded and whose
    /// `cleanup` has not run since. Mirrors Python's `_connected_servers`.
    connected: HashSet<usize>,
    strict: bool,
    drop_failed_servers: bool,
    connect_in_parallel: bool,
    connect_timeout: Option<Duration>,
    cleanup_timeout: Option<Duration>,
}

/// The identity of a server, as the manager tracks it: the `Arc`'s address.
fn server_id(server: &Arc<dyn MCPServer>) -> usize {
    Arc::as_ptr(server) as *const () as usize
}

impl MCPServerManager {
    /// Manage `servers` (duplicates by pointer are kept once).
    pub fn new(servers: Vec<Arc<dyn MCPServer>>) -> Self {
        let mut seen = HashSet::new();
        let servers: Vec<Arc<dyn MCPServer>> = servers
            .into_iter()
            .filter(|s| seen.insert(server_id(s)))
            .collect();
        MCPServerManager {
            // Python exposes every managed server until a connection attempt has been made.
            active: servers.clone(),
            servers,
            failed: Vec::new(),
            cleanup_errors: Vec::new(),
            connected: HashSet::new(),
            strict: false,
            drop_failed_servers: true,
            connect_in_parallel: false,
            connect_timeout: Some(Duration::from_secs(10)),
            cleanup_timeout: Some(Duration::from_secs(10)),
        }
    }

    /// Fail on the first connection error.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Change the strict flag after construction, as Python's `manager.strict = ...` does.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    /// Whether the manager fails on the first connection error.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Whether [`MCPServerManager::active_servers`] hides servers that failed to connect
    /// (the default). With `false`, every managed server stays active.
    ///
    /// Mirrors Python's `drop_failed_servers`.
    pub fn with_drop_failed_servers(mut self, drop_failed: bool) -> Self {
        self.drop_failed_servers = drop_failed;
        self
    }

    /// Whether failed servers are hidden from [`MCPServerManager::active_servers`].
    pub fn drops_failed_servers(&self) -> bool {
        self.drop_failed_servers
    }

    /// Connect every server concurrently instead of one after another.
    ///
    /// The active and failed lists keep the manager's server order either way, so a run is
    /// reproducible; in strict mode the first failure in that order is the one returned.
    pub fn with_connect_in_parallel(mut self, parallel: bool) -> Self {
        self.connect_in_parallel = parallel;
        self
    }

    /// Connection timeout per server.
    pub fn with_connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Cleanup timeout per server (`None` waits indefinitely).
    pub fn with_cleanup_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.cleanup_timeout = timeout;
        self
    }

    /// The connection timeout in force, as Python's `connect_timeout_seconds` property.
    pub fn connect_timeout(&self) -> Option<Duration> {
        self.connect_timeout
    }

    /// The cleanup timeout in force, as Python's `cleanup_timeout_seconds` property.
    pub fn cleanup_timeout(&self) -> Option<Duration> {
        self.cleanup_timeout
    }

    /// Change the connection timeout after construction.
    pub fn set_connect_timeout(&mut self, timeout: Option<Duration>) {
        self.connect_timeout = timeout;
    }

    /// Change the cleanup timeout after construction.
    pub fn set_cleanup_timeout(&mut self, timeout: Option<Duration>) {
        self.cleanup_timeout = timeout;
    }

    async fn connect_one(timeout: Option<Duration>, s: &Arc<dyn MCPServer>) -> Result<()> {
        match timeout {
            Some(d) => tokio::time::timeout(d, s.connect()).await.map_err(|_| {
                AgentsError::other(format!(
                    "Timed out connecting to MCP server '{}'",
                    get_mcp_server_log_name(s.name())
                ))
            })?,
            None => s.connect().await,
        }
    }

    async fn cleanup_one(timeout: Option<Duration>, s: &Arc<dyn MCPServer>) -> Result<()> {
        match timeout {
            Some(d) => tokio::time::timeout(d, s.cleanup())
                .await
                .unwrap_or_else(|_| {
                    Err(AgentsError::other(format!(
                        "Timed out cleaning up MCP server '{}'",
                        get_mcp_server_log_name(s.name())
                    )))
                }),
            None => s.cleanup().await,
        }
    }

    /// Forget a server's recorded failure, as Python does when a retry connects.
    fn clear_failure(&mut self, id: usize) {
        self.failed.retain(|(s, _)| server_id(s) != id);
        self.cleanup_errors.retain(|(s, _)| server_id(s) != id);
    }

    fn record_connect_failure(&mut self, s: &Arc<dyn MCPServer>, e: &AgentsError) {
        crate::logger::log_tool_action_error(
            &get_mcp_server_log_message("Failed to connect MCP server", s.name()),
            e,
        );
        let id = server_id(s);
        self.failed
            .retain(|(existing, _)| server_id(existing) != id);
        self.failed.push((s.clone(), e.to_string()));
    }

    fn record_cleanup_failure(&mut self, s: &Arc<dyn MCPServer>, e: &AgentsError) {
        crate::logger::log_tool_action_error(
            &get_mcp_server_log_message("Failed to cleanup MCP server", s.name()),
            e,
        );
        let id = server_id(s);
        self.cleanup_errors
            .retain(|(existing, _)| server_id(existing) != id);
        self.cleanup_errors.push((s.clone(), e.to_string()));
    }

    /// Recompute the active list from the connected set.
    fn refresh_active(&mut self) {
        self.active = if self.drop_failed_servers {
            self.servers
                .iter()
                .filter(|s| self.connected.contains(&server_id(s)))
                .cloned()
                .collect()
        } else {
            self.servers.clone()
        };
    }

    /// The servers that still need a connection attempt (Python's `_servers_to_connect`).
    fn servers_to_connect(&self, servers: &[Arc<dyn MCPServer>]) -> Vec<Arc<dyn MCPServer>> {
        if self.connected.is_empty() {
            return servers.to_vec();
        }
        servers
            .iter()
            .filter(|s| !self.connected.contains(&server_id(s)))
            .cloned()
            .collect()
    }

    /// Close `servers` in reverse order, returning the ones that closed cleanly.
    ///
    /// A failure is recorded and the server is left out of the returned list, so a retry never
    /// opens a second connection over a transport that was never closed.
    async fn cleanup_servers(
        &mut self,
        servers: Vec<Arc<dyn MCPServer>>,
    ) -> Vec<Arc<dyn MCPServer>> {
        let timeout = self.cleanup_timeout;
        let mut cleaned: HashSet<usize> = HashSet::new();
        for s in servers.iter().rev() {
            match Self::cleanup_one(timeout, s).await {
                Ok(()) => {
                    self.connected.remove(&server_id(s));
                    cleaned.insert(server_id(s));
                }
                Err(e) => {
                    self.connected.remove(&server_id(s));
                    self.record_cleanup_failure(s, &e);
                }
            }
        }
        servers
            .into_iter()
            .filter(|s| cleaned.contains(&server_id(s)))
            .collect()
    }

    async fn attempt_connect(&mut self, s: &Arc<dyn MCPServer>) -> Result<()> {
        match Self::connect_one(self.connect_timeout, s).await {
            Ok(()) => {
                self.connected.insert(server_id(s));
                self.clear_failure(server_id(s));
                Ok(())
            }
            Err(e) => {
                self.record_connect_failure(s, &e);
                Err(e)
            }
        }
    }

    /// Connect `servers`, returning the first failure in input order.
    async fn connect_servers(&mut self, servers: &[Arc<dyn MCPServer>]) -> Option<AgentsError> {
        let mut first_error: Option<AgentsError> = None;
        if self.connect_in_parallel {
            let timeout = self.connect_timeout;
            let results = futures::future::join_all(
                servers
                    .iter()
                    .map(|s| async move { (s.clone(), Self::connect_one(timeout, s).await) }),
            )
            .await;
            for (s, outcome) in results {
                match outcome {
                    Ok(()) => {
                        self.connected.insert(server_id(&s));
                        self.clear_failure(server_id(&s));
                    }
                    Err(e) => {
                        self.record_connect_failure(&s, &e);
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                }
            }
        } else {
            for s in servers {
                if let Err(e) = self.attempt_connect(s).await {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                    // Strict mode stops at the first failure, as Python's `raise_on_error` does.
                    if self.strict {
                        break;
                    }
                }
            }
        }
        first_error
    }

    /// Connect all servers.
    ///
    /// Servers that are already connected are left alone. In strict mode the first failure is
    /// returned after every server this call opened has been closed again.
    pub async fn connect_all(&mut self) -> Result<()> {
        let previously_connected: HashSet<usize> = self.connected.clone();
        let previous_active = self.active.clone();
        self.failed.clear();
        self.cleanup_errors.clear();

        let to_connect = self.servers_to_connect(&self.servers.clone());
        let error = self.connect_servers(&to_connect).await;

        match error {
            Some(e) if self.strict => {
                // Everything this call opened — and the server that failed half-way through
                // its own connect — has to be closed before the error reaches the caller.
                let opened: Vec<Arc<dyn MCPServer>> = to_connect
                    .iter()
                    .filter(|s| {
                        self.connected.contains(&server_id(s))
                            && !previously_connected.contains(&server_id(s))
                    })
                    .cloned()
                    .collect();
                let failed: Vec<Arc<dyn MCPServer>> =
                    self.failed.iter().map(|(s, _)| s.clone()).collect();
                let mut to_clean = opened;
                for s in failed {
                    if !to_clean.iter().any(|existing| Arc::ptr_eq(existing, &s)) {
                        to_clean.push(s);
                    }
                }
                self.cleanup_servers(to_clean).await;
                if self.drop_failed_servers {
                    self.active = self
                        .servers
                        .iter()
                        .filter(|s| previously_connected.contains(&server_id(s)))
                        .cloned()
                        .collect();
                } else {
                    self.active = previous_active;
                }
                Err(e)
            }
            _ => {
                self.refresh_active();
                Ok(())
            }
        }
    }

    /// Retry failed servers (or reconnect all).
    ///
    /// With `failed_only`, every previously failed server is closed before it is retried, and a
    /// server whose cleanup fails is not retried at all — its cleanup error stays in
    /// [`MCPServerManager::errors`].
    pub async fn reconnect(&mut self, failed_only: bool) -> Result<()> {
        let to_retry = if failed_only {
            let failed: Vec<Arc<dyn MCPServer>> =
                self.failed.iter().map(|(s, _)| s.clone()).collect();
            self.cleanup_servers(failed).await
        } else {
            self.cleanup_all().await;
            self.failed.clear();
            self.cleanup_errors.clear();
            self.servers.clone()
        };
        let to_retry = self.servers_to_connect(&to_retry);
        let error = self.connect_servers(&to_retry).await;
        self.refresh_active();
        match error {
            Some(e) if self.strict => Err(e),
            _ => Ok(()),
        }
    }

    /// Successfully connected servers.
    pub fn active_servers(&self) -> Vec<Arc<dyn MCPServer>> {
        self.active.clone()
    }

    /// Every server this manager was built with, in order.
    pub fn all_servers(&self) -> Vec<Arc<dyn MCPServer>> {
        self.servers.clone()
    }

    /// Servers that failed to connect, with their errors.
    pub fn failed_servers(&self) -> &[(Arc<dyn MCPServer>, String)] {
        &self.failed
    }

    /// The recorded connect and cleanup failures, by server name.
    ///
    /// One entry per server, as Python's `errors` dict: a later cleanup failure replaces the
    /// connect failure recorded for the same server.
    pub fn errors(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for (server, error) in self.failed.iter().chain(&self.cleanup_errors) {
            let name = server.name().to_string();
            match out.iter_mut().find(|(existing, _)| *existing == name) {
                Some(slot) => slot.1 = error.clone(),
                None => out.push((name, error.clone())),
            }
        }
        out
    }

    /// Close all servers, in reverse order, bounded by the cleanup timeout.
    ///
    /// Cleanup failures are recorded in [`MCPServerManager::errors`] so a later reconnect does
    /// not open a second connection over a transport that was never closed.
    pub async fn cleanup_all(&mut self) {
        self.cleanup_errors.clear();
        let servers = self.servers.clone();
        self.cleanup_servers(servers).await;
        self.refresh_active();
    }
}
