//! Model Context Protocol servers (feature `mcp`). Mirrors `agents.mcp`.
//!
//! Local MCP servers run their tools through the SDK: the runner lists the server's tools on
//! each run and calls them as function tools. Supported transports: stdio
//! ([`MCPServerStdio`]), Streamable HTTP ([`MCPServerStreamableHttp`]) and the deprecated
//! HTTP+SSE transport ([`MCPServerSse`]). For OpenAI-hosted MCP, use
//! [`crate::tool::HostedMCPTool`] instead.

pub mod logging;
mod sse;
pub mod util;

pub use logging::{
    MCPToolMetadata, extract_mcp_tool_metadata, get_mcp_server_log_message,
    get_mcp_server_log_name, resolve_mcp_tool_description, resolve_mcp_tool_description_for_model,
    resolve_mcp_tool_title,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
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
        })
    }
}

/// Context passed to a dynamic tool filter.
#[derive(Debug, Clone)]
pub struct ToolFilterContext {
    pub agent_name: String,
    pub server_name: String,
}

type DynamicFilter = Arc<dyn Fn(&ToolFilterContext, &MCPTool) -> bool + Send + Sync>;

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
    fn allows(&self, ctx: &ToolFilterContext, tool: &MCPTool) -> bool {
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
            ToolFilter::Dynamic(f) => f(ctx, tool),
        }
    }
}

type ApprovalFn = Arc<dyn Fn(&str, &MCPTool) -> bool + Send + Sync>;

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
    /// Whether `tool` requires approval for `agent_name`.
    pub fn requires(&self, agent_name: &str, tool: &MCPTool) -> bool {
        match self {
            RequireApproval::Never => false,
            RequireApproval::Always => true,
            RequireApproval::PerTool(m) => m.get(&tool.name).copied().unwrap_or(false),
            RequireApproval::Dynamic(f) => f(agent_name, tool),
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
    /// Retries for `list_tools` and `call_tool`.
    pub max_retry_attempts: u32,
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
            .finish()
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

/// Parameters for an HTTP+SSE MCP server.
#[derive(Debug, Clone)]
pub struct MCPServerSseParams {
    /// The SSE endpoint URL.
    pub url: String,
    /// Extra headers sent on the SSE request and on every POST.
    pub headers: HashMap<String, String>,
    /// Connection timeout (default 5 seconds).
    pub timeout: Duration,
    /// How long to wait for a new SSE event (default 5 minutes).
    pub sse_read_timeout: Duration,
}

impl Default for MCPServerSseParams {
    fn default() -> Self {
        MCPServerSseParams {
            url: String::new(),
            headers: HashMap::new(),
            timeout: Duration::from_secs(5),
            sse_read_timeout: Duration::from_secs(300),
        }
    }
}

/// Parameters for a stdio MCP server.
#[derive(Debug, Clone, Default)]
pub struct MCPServerStdioParams {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<String>,
}

/// Parameters for a Streamable HTTP MCP server.
#[derive(Debug, Clone, Default)]
pub struct MCPServerStreamableHttpParams {
    pub url: String,
    /// Extra headers. An `Authorization: Bearer <token>` header is used as the bearer token.
    pub headers: HashMap<String, String>,
}

struct Core {
    name: String,
    transport: TransportSpec,
    config: MCPServerConfig,
    service: Mutex<Option<RunningService<RoleClient, ()>>>,
    tools_cache: Mutex<Option<Vec<MCPTool>>>,
}

fn svc_err(e: impl fmt::Display) -> AgentsError {
    AgentsError::other(format!("MCP error: {e}"))
}

impl Core {
    async fn connect(&self) -> Result<()> {
        let mut guard = self.service.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        let service = match &self.transport {
            TransportSpec::Stdio(p) => {
                let p = p.clone();
                let cmd = tokio::process::Command::new(&p.command).configure(move |c| {
                    c.args(&p.args);
                    c.envs(&p.env);
                    if let Some(cwd) = &p.cwd {
                        c.current_dir(cwd);
                    }
                });
                let transport = TokioChildProcess::new(cmd)?;
                ().serve(transport).await.map_err(svc_err)?
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
                let transport = StreamableHttpClientTransport::from_config(config);
                ().serve(transport).await.map_err(svc_err)?
            }
            TransportSpec::Sse(p) => {
                let transport =
                    sse::connect(&p.url, &p.headers, p.timeout, p.sse_read_timeout).await?;
                ().serve(transport).await.map_err(svc_err)?
            }
        };
        *guard = Some(service);
        Ok(())
    }

    async fn cleanup(&self) -> Result<()> {
        if let Some(s) = self.service.lock().await.take() {
            let _ = s.cancel().await;
        }
        Ok(())
    }

    /// The server name with URL credentials, query and fragment removed.
    fn log_name(&self) -> String {
        get_mcp_server_log_name(&self.name)
    }

    async fn with_timeout<T>(
        &self,
        fut: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        match self.config.client_session_timeout {
            Some(d) if !d.is_zero() => tokio::time::timeout(d, fut).await.map_err(|_| {
                AgentsError::other(format!("MCP request to '{}' timed out", self.log_name()))
            })?,
            _ => fut.await,
        }
    }

    async fn session(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, Option<RunningService<RoleClient, ()>>>> {
        let guard = self.service.lock().await;
        if guard.is_none() {
            return Err(AgentsError::user(format!(
                "Server '{}' not initialized. Call connect() first.",
                self.log_name()
            )));
        }
        Ok(guard)
    }

    async fn retry<T, F, Fut>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        loop {
            match self.with_timeout(f()).await {
                Ok(v) => return Ok(v),
                Err(e) if attempt < self.config.max_retry_attempts => {
                    attempt += 1;
                    let mut delay =
                        self.config.retry_backoff_seconds_base * 2f64.powi(attempt as i32 - 1);
                    if let Some(max) = self.config.retry_backoff_seconds_max {
                        delay = delay.min(max);
                    }
                    crate::logger::log_tool_action_warning(
                        &get_mcp_server_log_message(
                            "Retrying failed request to MCP server",
                            &self.name,
                        ),
                        &e,
                    );
                    tokio::time::sleep(Duration::from_secs_f64(delay.max(0.0))).await;
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
                "MCP server '{}' returned a repeated cursor while paginating {operation}.",
                self.log_name()
            )));
        }
        if let Some(max) = self.config.max_list_pages
            && pages >= max
        {
            return Err(AgentsError::user(format!(
                "MCP server '{}' exceeded max_list_pages ({max}) while paginating {operation}.",
                self.log_name()
            )));
        }
        Ok(Some(next))
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        if self.config.cache_tools_list
            && let Some(t) = self.tools_cache.lock().await.clone()
        {
            return Ok(t);
        }
        let tools = self
            .retry(|| async {
                let mut out = Vec::new();
                let mut cursor: Option<String> = None;
                let mut pages = 0usize;
                loop {
                    let page = {
                        let guard = self.session().await?;
                        let s = guard.as_ref().expect("session checked");
                        s.list_tools(cursor.clone().map(|c| {
                            rmcp::model::PaginatedRequestParams::default().with_cursor(Some(c))
                        }))
                        .await
                        .map_err(svc_err)?
                    };
                    pages += 1;
                    out.extend(
                        page.tools
                            .iter()
                            .filter_map(|t| serde_json::to_value(t).ok())
                            .filter_map(|v| MCPTool::from_value(&v)),
                    );
                    let next = page.next_cursor.map(|c| c.to_string());
                    match self.check_pagination(next, cursor, pages, "list tools")? {
                        Some(next) => cursor = Some(next),
                        None => break,
                    }
                }
                Ok(out)
            })
            .await?;
        if self.config.cache_tools_list {
            *self.tools_cache.lock().await = Some(tools.clone());
        }
        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
        meta: Option<Map<String, Value>>,
    ) -> Result<Value> {
        self.retry(|| {
            let arguments = arguments.clone();
            let meta = meta.clone();
            async move {
                let mut req = json!({"name": tool_name});
                if let Some(a) = arguments {
                    req["arguments"] = Value::Object(a);
                }
                if let Some(m) = meta {
                    req["_meta"] = Value::Object(m);
                }
                let params: rmcp::model::CallToolRequestParams = serde_json::from_value(req)?;
                let guard = self.session().await?;
                let s = guard.as_ref().expect("session checked");
                let result = s.call_tool(params).await.map_err(svc_err)?;
                Ok(serde_json::to_value(result)?)
            }
        })
        .await
    }

    async fn list_prompts(&self) -> Result<Vec<Value>> {
        let mut out: Vec<Value> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;
        loop {
            let guard = self.session().await?;
            let s = guard.as_ref().expect("session checked");
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
            match self.check_pagination(next, cursor, pages, "list prompts")? {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    async fn list_resources(&self, cursor: Option<&str>) -> Result<Value> {
        let guard = self.session().await?;
        let s = guard.as_ref().expect("session checked");
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
        let s = guard.as_ref().expect("session checked");
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
        let s = guard.as_ref().expect("session checked");
        let result = self
            .with_timeout(async { s.read_resource(params).await.map_err(svc_err) })
            .await?;
        Ok(serde_json::to_value(result)?)
    }

    async fn server_initialize_result(&self) -> Option<Value> {
        let guard = self.service.lock().await;
        let info = guard.as_ref()?.peer_info()?;
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
        let s = guard.as_ref().expect("session checked");
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
                        service: Mutex::new(None),
                        tools_cache: Mutex::new(None),
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

            /// Retry `list_tools` and `call_tool`.
            pub fn with_max_retry_attempts(mut self, attempts: u32) -> Self {
                self.core.config.max_retry_attempts = attempts;
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
#[derive(Debug)]
pub struct MCPServerManager {
    servers: Vec<Arc<dyn MCPServer>>,
    active: Vec<Arc<dyn MCPServer>>,
    failed: Vec<(Arc<dyn MCPServer>, String)>,
    cleanup_errors: Vec<(Arc<dyn MCPServer>, String)>,
    strict: bool,
    connect_in_parallel: bool,
    connect_timeout: Option<Duration>,
    cleanup_timeout: Option<Duration>,
}

impl MCPServerManager {
    /// Manage `servers` (duplicates by pointer are kept once).
    pub fn new(servers: Vec<Arc<dyn MCPServer>>) -> Self {
        let mut seen = HashSet::new();
        let servers = servers
            .into_iter()
            .filter(|s| seen.insert(Arc::as_ptr(s) as *const () as usize))
            .collect();
        MCPServerManager {
            servers,
            active: Vec::new(),
            failed: Vec::new(),
            cleanup_errors: Vec::new(),
            strict: false,
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

    fn record_connect_results(
        &mut self,
        results: Vec<(Arc<dyn MCPServer>, Result<()>)>,
    ) -> Result<()> {
        let mut first_error = None;
        for (s, outcome) in results {
            match outcome {
                Ok(()) => self.active.push(s),
                Err(e) => {
                    crate::logger::log_tool_action_warning(
                        &get_mcp_server_log_message("Failed to connect MCP server", s.name()),
                        &e,
                    );
                    self.failed.push((s, e.to_string()));
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        match first_error {
            Some(e) if self.strict => Err(e),
            _ => Ok(()),
        }
    }

    /// Connect all servers.
    pub async fn connect_all(&mut self) -> Result<()> {
        self.active.clear();
        self.failed.clear();
        let servers = self.servers.clone();
        let timeout = self.connect_timeout;
        let results = if self.connect_in_parallel {
            futures::future::join_all(
                servers
                    .iter()
                    .map(|s| async move { (s.clone(), Self::connect_one(timeout, s).await) }),
            )
            .await
        } else {
            let mut results = Vec::with_capacity(servers.len());
            for s in &servers {
                let outcome = Self::connect_one(timeout, s).await;
                let failed_fast = self.strict && outcome.is_err();
                results.push((s.clone(), outcome));
                if failed_fast {
                    break;
                }
            }
            results
        };
        self.record_connect_results(results)
    }

    /// Retry failed servers (or reconnect all).
    pub async fn reconnect(&mut self, failed_only: bool) -> Result<()> {
        if !failed_only {
            self.cleanup_all().await;
            return self.connect_all().await;
        }
        let failed = std::mem::take(&mut self.failed);
        for (s, _) in failed {
            match Self::connect_one(self.connect_timeout, &s).await {
                Ok(()) => self.active.push(s),
                Err(e) => self.failed.push((s, e.to_string())),
            }
        }
        Ok(())
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
    pub fn errors(&self) -> Vec<(String, String)> {
        self.failed
            .iter()
            .chain(&self.cleanup_errors)
            .map(|(s, e)| (s.name().to_string(), e.clone()))
            .collect()
    }

    /// Close all servers, bounded by the cleanup timeout.
    ///
    /// Cleanup failures are recorded in [`MCPServerManager::errors`] so a later reconnect does
    /// not open a second connection over a transport that was never closed.
    pub async fn cleanup_all(&mut self) {
        self.cleanup_errors.clear();
        for s in self.servers.clone() {
            let outcome = match self.cleanup_timeout {
                Some(d) => tokio::time::timeout(d, s.cleanup())
                    .await
                    .unwrap_or_else(|_| {
                        Err(AgentsError::other(format!(
                            "Timed out cleaning up MCP server '{}'",
                            get_mcp_server_log_name(s.name())
                        )))
                    }),
                None => s.cleanup().await,
            };
            if let Err(e) = outcome {
                crate::logger::log_tool_action_warning(
                    &get_mcp_server_log_message("Failed to cleanup MCP server", s.name()),
                    &e,
                );
                self.cleanup_errors.push((s, e.to_string()));
            }
        }
        self.active.clear();
    }
}
