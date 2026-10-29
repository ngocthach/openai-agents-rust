//! Model Context Protocol servers (feature `mcp`). Mirrors `agents.mcp`.
//!
//! Local MCP servers run their tools through the SDK: the runner lists the server's tools on
//! each run and calls them as function tools. Two transports are supported, matching the
//! transports recommended by the MCP project: stdio ([`MCPServerStdio`]) and Streamable HTTP
//! ([`MCPServerStreamableHttp`]). The deprecated HTTP+SSE transport is not supported. For
//! OpenAI-hosted MCP, use [`crate::tool::HostedMCPTool`] instead.

pub mod util;

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

/// Settings shared by local MCP servers.
#[derive(Debug, Clone)]
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
    /// Prefer `structuredContent` over content blocks.
    pub use_structured_content: bool,
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
            use_structured_content: false,
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

    async fn with_timeout<T>(
        &self,
        fut: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        match self.config.client_session_timeout {
            Some(d) if !d.is_zero() => tokio::time::timeout(d, fut).await.map_err(|_| {
                AgentsError::other(format!("MCP request to {} timed out", self.name))
            })?,
            _ => fut.await,
        }
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
                    let delay =
                        self.config.retry_backoff_seconds_base * 2f64.powi(attempt as i32 - 1);
                    log::warn!("MCP request to {} failed ({e}); retrying", self.name);
                    tokio::time::sleep(Duration::from_secs_f64(delay.max(0.0))).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        if self.config.cache_tools_list
            && let Some(t) = self.tools_cache.lock().await.clone()
        {
            return Ok(t);
        }
        let tools = self
            .retry(|| async {
                let guard = self.service.lock().await;
                let s = guard.as_ref().ok_or_else(|| {
                    AgentsError::user(format!(
                        "Server {} not initialized. Call connect() first.",
                        self.name
                    ))
                })?;
                let tools = s.list_all_tools().await.map_err(svc_err)?;
                Ok(tools
                    .iter()
                    .filter_map(|t| serde_json::to_value(t).ok())
                    .filter_map(|v| MCPTool::from_value(&v))
                    .collect::<Vec<_>>())
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
    ) -> Result<Value> {
        self.retry(|| {
            let arguments = arguments.clone();
            async move {
                let mut req = json!({"name": tool_name});
                if let Some(a) = arguments {
                    req["arguments"] = Value::Object(a);
                }
                let params: rmcp::model::CallToolRequestParams = serde_json::from_value(req)?;
                let guard = self.service.lock().await;
                let s = guard.as_ref().ok_or_else(|| {
                    AgentsError::user(format!(
                        "Server {} not initialized. Call connect() first.",
                        self.name
                    ))
                })?;
                let result = s.call_tool(params).await.map_err(svc_err)?;
                Ok(serde_json::to_value(result)?)
            }
        })
        .await
    }

    async fn list_prompts(&self) -> Result<Vec<Value>> {
        let guard = self.service.lock().await;
        let s = guard
            .as_ref()
            .ok_or_else(|| AgentsError::user(format!("Server {} not initialized", self.name)))?;
        let prompts = self
            .with_timeout(async { s.list_all_prompts().await.map_err(svc_err) })
            .await?;
        Ok(prompts
            .iter()
            .filter_map(|p| serde_json::to_value(p).ok())
            .collect())
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
        let guard = self.service.lock().await;
        let s = guard
            .as_ref()
            .ok_or_else(|| AgentsError::user(format!("Server {} not initialized", self.name)))?;
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

            /// Prefer structured content in tool results.
            pub fn with_use_structured_content(mut self, use_structured: bool) -> Self {
                self.core.config.use_structured_content = use_structured;
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
                self.core.call_tool(tool_name, arguments).await
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

/// Connects several servers and exposes the ones that connected.
#[derive(Debug)]
pub struct MCPServerManager {
    servers: Vec<Arc<dyn MCPServer>>,
    active: Vec<Arc<dyn MCPServer>>,
    failed: Vec<(Arc<dyn MCPServer>, String)>,
    strict: bool,
    connect_timeout: Option<Duration>,
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
            strict: false,
            connect_timeout: Some(Duration::from_secs(10)),
        }
    }

    /// Fail on the first connection error.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Connection timeout per server.
    pub fn with_connect_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.connect_timeout = timeout;
        self
    }

    async fn connect_one(&self, s: &Arc<dyn MCPServer>) -> Result<()> {
        match self.connect_timeout {
            Some(d) => tokio::time::timeout(d, s.connect())
                .await
                .map_err(|_| AgentsError::other(format!("Timed out connecting to {}", s.name())))?,
            None => s.connect().await,
        }
    }

    /// Connect all servers.
    pub async fn connect_all(&mut self) -> Result<()> {
        self.active.clear();
        self.failed.clear();
        for s in self.servers.clone() {
            match self.connect_one(&s).await {
                Ok(()) => self.active.push(s),
                Err(e) => {
                    if self.strict {
                        return Err(e);
                    }
                    log::warn!("Failed to connect MCP server {}: {e}", s.name());
                    self.failed.push((s, e.to_string()));
                }
            }
        }
        Ok(())
    }

    /// Retry failed servers (or reconnect all).
    pub async fn reconnect(&mut self, failed_only: bool) -> Result<()> {
        if !failed_only {
            self.cleanup_all().await;
            return self.connect_all().await;
        }
        let failed = std::mem::take(&mut self.failed);
        for (s, _) in failed {
            match self.connect_one(&s).await {
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

    /// Servers that failed to connect, with their errors.
    pub fn failed_servers(&self) -> &[(Arc<dyn MCPServer>, String)] {
        &self.failed
    }

    /// Close all servers.
    pub async fn cleanup_all(&mut self) {
        for s in &self.servers {
            let _ = s.cleanup().await;
        }
        self.active.clear();
    }
}
