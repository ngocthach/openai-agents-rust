//! Helpers shared by the `py_mcpmem_*` parity tests ported from the Python SDK suite
//! (`tests/mcp`, `tests/memory`, `tests/extensions/memory`).
//!
//! The Python suite keeps these in `tests/mcp/helpers.py`, `tests/utils/simple_session.py` and
//! a handful of per-file fakes; the Rust equivalents live here so no parity test needs a real
//! process, server or network.
#![allow(dead_code, unused_imports)] // each feature-gated test binary uses a different subset

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::memory::{PopValidator, Session, SessionSettings};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// Port of `tests/utils/simple_session.py`: `SimpleListSession`, `CountingSession` and
/// `IdStrippingSession` collapsed into one configurable store.
#[derive(Debug)]
pub struct ListSession {
    session_id: String,
    settings: Option<SessionSettings>,
    items: Mutex<Vec<InputItem>>,
    pop_calls: Mutex<usize>,
    strip_ids: bool,
}

impl ListSession {
    pub fn new(session_id: impl Into<String>) -> Self {
        ListSession {
            session_id: session_id.into(),
            settings: None,
            items: Mutex::new(Vec::new()),
            pop_calls: Mutex::new(0),
            strip_ids: false,
        }
    }

    pub fn with_history(session_id: impl Into<String>, history: Vec<InputItem>) -> Self {
        let s = ListSession::new(session_id);
        *s.items.lock().unwrap() = history;
        s
    }

    /// Mimic a hosted store that reassigns ids: drop `id` on write.
    pub fn stripping_ids(mut self) -> Self {
        self.strip_ids = true;
        self
    }

    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// The raw stored items.
    pub fn items(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }

    /// How many times `pop_item` ran (Python's `CountingSession.pop_calls`).
    pub fn pop_calls(&self) -> usize {
        *self.pop_calls.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = openai_agents_sdk::memory::resolve_session_limit(limit, self.settings.as_ref());
        let items = self.items.lock().unwrap();
        Ok(match limit {
            None => items.clone(),
            Some(0) => Vec::new(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let mut stored = self.items.lock().unwrap();
        for item in items {
            if self.strip_ids {
                let mut value = item.as_value().clone();
                if let Value::Object(map) = &mut value {
                    map.remove("id");
                }
                stored.push(InputItem::from_value(value));
            } else {
                stored.push(item);
            }
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        *self.pop_calls.lock().unwrap() += 1;
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

/// A session whose every operation fails, for error-propagation tests.
#[derive(Debug)]
pub struct FailingSession {
    pub message: String,
}

impl FailingSession {
    pub fn new(message: impl Into<String>) -> Self {
        FailingSession {
            message: message.into(),
        }
    }

    fn err<T>(&self) -> Result<T> {
        Err(openai_agents_sdk::error::AgentsError::other(
            self.message.clone(),
        ))
    }
}

#[async_trait::async_trait]
impl Session for FailingSession {
    fn session_id(&self) -> &str {
        "failing"
    }

    async fn get_items(&self, _limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.err()
    }

    async fn add_items(&self, _items: Vec<InputItem>) -> Result<()> {
        self.err()
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.err()
    }

    async fn pop_item_validated(&self, _validate: PopValidator) -> Result<Option<InputItem>> {
        self.err()
    }

    async fn clear_session(&self) -> Result<()> {
        self.err()
    }
}

/// `InputItem` for a user message, as Python's `tests/test_responses.get_text_input_item`.
pub fn user_item(text: &str) -> InputItem {
    InputItem::user(text)
}

/// `InputItem` for an assistant message, as Python's `get_text_message` once persisted.
pub fn assistant_item(text: &str) -> InputItem {
    InputItem::from_value(json!({
        "id": "__fake_id__",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }))
}

/// The `content` text of a message item, whatever shape it is stored in.
pub fn item_text(item: &InputItem) -> String {
    let value = item.as_value();
    match value.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Mock HTTP server (shape copied from `tests/sessions_tests.rs`)
// ---------------------------------------------------------------------------

/// One request seen by a mock server.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// What a mock server answers with.
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Reply {
    pub fn json(value: Value) -> Self {
        Reply {
            status: 200,
            headers: vec![("content-type".into(), "application/json".into())],
            body: value.to_string(),
        }
    }

    pub fn status(status: u16, body: impl Into<String>) -> Self {
        Reply {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn empty(status: u16) -> Self {
        Reply {
            status,
            headers: Vec::new(),
            body: String::new(),
        }
    }
}

pub type Handler = Arc<dyn Fn(&Request) -> Reply + Send + Sync>;

/// Start a mock HTTP server driven by `handler`; returns its base URL.
pub async fn serve(handler: Handler) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                let Some(request) = read_request(&mut sock).await else {
                    return;
                };
                let reply = handler(&request);
                let mut head = format!(
                    "HTTP/1.1 {} OK\r\ncontent-length: {}\r\nconnection: close\r\n",
                    reply.status,
                    reply.body.len()
                );
                for (k, v) in &reply.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                head.push_str(&reply.body);
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

/// Read one HTTP request, honouring `content-length`.
async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        let Some(idx) = text.find("\r\n\r\n") else {
            continue;
        };
        let head = &text[..idx];
        let mut headers = Vec::new();
        for line in head.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_lowercase(), v.trim().to_string()));
            }
        }
        let len: usize = headers
            .iter()
            .find(|(k, _)| k == "content-length")
            .map(|(_, v)| v.parse().unwrap_or(0))
            .unwrap_or(0);
        if buf.len() < idx + 4 + len {
            continue;
        }
        let mut parts = head.split_whitespace();
        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();
        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null);
        return Some(Request {
            method,
            path,
            headers,
            body,
        });
    }
}

/// A mock OpenAI endpoint answering the queued replies in order; returns the base URL for
/// `OpenAIClient::with_base_url` and the captured requests.
pub async fn mock_openai(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<Request>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let queue = Arc::new(Mutex::new(replies));
    let cap = captured.clone();
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        let mut queue = queue.lock().unwrap();
        if queue.is_empty() {
            Reply::json(json!({"error": {"message": "unexpected request"}}))
        } else {
            Reply::json(queue.remove(0))
        }
    }))
    .await;
    (format!("{base}/v1"), captured)
}

/// A mock OpenAI endpoint whose replies are produced per request.
pub async fn mock_openai_with<F>(f: F) -> (String, Arc<Mutex<Vec<Request>>>)
where
    F: Fn(&Request) -> Value + Send + Sync + 'static,
{
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        Reply::json(f(req))
    }))
    .await;
    (format!("{base}/v1"), captured)
}

// ---------------------------------------------------------------------------
// MCP
// ---------------------------------------------------------------------------

#[cfg(feature = "mcp")]
#[allow(unused_imports)]
pub use mcp_support::*;

#[cfg(feature = "mcp")]
mod mcp_support {
    use super::*;
    use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};

    /// What a `FakeMcpServer` call_tool returns.
    #[derive(Clone)]
    pub enum FakeCallResult {
        /// Echo the tool name as text content, like Python's `FakeMCPServer`.
        Echo,
        /// A ready-made `CallToolResult` JSON value.
        Value(Value),
        /// Fail the call with this message.
        Error(String),
    }

    /// Port of `tests/mcp/helpers.py::FakeMCPServer`: an in-process MCP server with scripted
    /// tools, recording every call it receives.
    pub struct FakeMcpServer {
        name: String,
        config: MCPServerConfig,
        tools: Mutex<Vec<MCPTool>>,
        tool_calls: Mutex<Vec<String>>,
        tool_results: Mutex<Vec<String>>,
        tool_metas: Mutex<Vec<Option<Map<String, Value>>>>,
        prompts: Mutex<Vec<Value>>,
        prompt_result: Mutex<Option<Value>>,
        resources: Mutex<Option<Value>>,
        resource_templates: Mutex<Option<Value>>,
        resource_contents: Mutex<Option<Value>>,
        result: Mutex<FakeCallResult>,
        list_calls: Mutex<usize>,
        connected: Mutex<bool>,
        connect_calls: Mutex<usize>,
        cleanup_calls: Mutex<usize>,
        connect_error: Mutex<Option<String>>,
        cleanup_error: Mutex<Option<String>>,
        list_error: Mutex<Option<String>>,
        session_id: Mutex<Option<String>>,
    }

    impl std::fmt::Debug for FakeMcpServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("FakeMcpServer")
                .field("name", &self.name)
                .finish_non_exhaustive()
        }
    }

    impl FakeMcpServer {
        pub fn new(name: impl Into<String>) -> Self {
            FakeMcpServer {
                name: name.into(),
                config: MCPServerConfig::default(),
                tools: Mutex::new(Vec::new()),
                tool_calls: Mutex::new(Vec::new()),
                tool_results: Mutex::new(Vec::new()),
                tool_metas: Mutex::new(Vec::new()),
                prompts: Mutex::new(Vec::new()),
                prompt_result: Mutex::new(None),
                resources: Mutex::new(None),
                resource_templates: Mutex::new(None),
                resource_contents: Mutex::new(None),
                result: Mutex::new(FakeCallResult::Echo),
                list_calls: Mutex::new(0),
                connected: Mutex::new(false),
                connect_calls: Mutex::new(0),
                cleanup_calls: Mutex::new(0),
                connect_error: Mutex::new(None),
                cleanup_error: Mutex::new(None),
                list_error: Mutex::new(None),
                session_id: Mutex::new(None),
            }
        }

        pub fn with_config(mut self, config: MCPServerConfig) -> Self {
            self.config = config;
            self
        }

        /// Mutate the server's config before it is shared.
        pub fn configure(mut self, f: impl FnOnce(&mut MCPServerConfig)) -> Self {
            f(&mut self.config);
            self
        }

        pub fn add_tool(self, name: &str, input_schema: Value) -> Self {
            self.tools.lock().unwrap().push(MCPTool {
                meta: None,
                name: name.to_string(),
                title: None,
                description: None,
                input_schema,
                annotations: None,
            });
            self
        }

        pub fn add_full_tool(self, tool: MCPTool) -> Self {
            self.tools.lock().unwrap().push(tool);
            self
        }

        pub fn with_prompts(self, prompts: Vec<Value>) -> Self {
            *self.prompts.lock().unwrap() = prompts;
            self
        }

        pub fn with_prompt_result(self, result: Value) -> Self {
            *self.prompt_result.lock().unwrap() = Some(result);
            self
        }

        pub fn with_resources(self, resources: Value) -> Self {
            *self.resources.lock().unwrap() = Some(resources);
            self
        }

        pub fn with_resource_templates(self, templates: Value) -> Self {
            *self.resource_templates.lock().unwrap() = Some(templates);
            self
        }

        pub fn with_resource_contents(self, contents: Value) -> Self {
            *self.resource_contents.lock().unwrap() = Some(contents);
            self
        }

        pub fn with_result(self, result: FakeCallResult) -> Self {
            *self.result.lock().unwrap() = result;
            self
        }

        pub fn with_session_id(self, id: &str) -> Self {
            *self.session_id.lock().unwrap() = Some(id.to_string());
            self
        }

        pub fn failing_connect(self, message: &str) -> Self {
            *self.connect_error.lock().unwrap() = Some(message.to_string());
            self
        }

        pub fn failing_cleanup(self, message: &str) -> Self {
            *self.cleanup_error.lock().unwrap() = Some(message.to_string());
            self
        }

        pub fn failing_list(self, message: &str) -> Self {
            *self.list_error.lock().unwrap() = Some(message.to_string());
            self
        }

        /// Replace the scripted result at runtime.
        pub fn set_result(&self, result: FakeCallResult) {
            *self.result.lock().unwrap() = result;
        }

        pub fn set_tools(&self, tools: Vec<MCPTool>) {
            *self.tools.lock().unwrap() = tools;
        }

        pub fn tool_calls(&self) -> Vec<String> {
            self.tool_calls.lock().unwrap().clone()
        }

        pub fn tool_results(&self) -> Vec<String> {
            self.tool_results.lock().unwrap().clone()
        }

        pub fn tool_metas(&self) -> Vec<Option<Map<String, Value>>> {
            self.tool_metas.lock().unwrap().clone()
        }

        pub fn list_calls(&self) -> usize {
            *self.list_calls.lock().unwrap()
        }

        pub fn connect_calls(&self) -> usize {
            *self.connect_calls.lock().unwrap()
        }

        pub fn cleanup_calls(&self) -> usize {
            *self.cleanup_calls.lock().unwrap()
        }

        pub fn is_connected(&self) -> bool {
            *self.connected.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl MCPServer for FakeMcpServer {
        fn name(&self) -> &str {
            &self.name
        }

        fn config(&self) -> &MCPServerConfig {
            &self.config
        }

        async fn connect(&self) -> Result<()> {
            *self.connect_calls.lock().unwrap() += 1;
            if let Some(message) = self.connect_error.lock().unwrap().clone() {
                return Err(openai_agents_sdk::error::AgentsError::other(message));
            }
            *self.connected.lock().unwrap() = true;
            Ok(())
        }

        async fn cleanup(&self) -> Result<()> {
            *self.cleanup_calls.lock().unwrap() += 1;
            if let Some(message) = self.cleanup_error.lock().unwrap().clone() {
                return Err(openai_agents_sdk::error::AgentsError::other(message));
            }
            *self.connected.lock().unwrap() = false;
            Ok(())
        }

        async fn list_tools(&self) -> Result<Vec<MCPTool>> {
            *self.list_calls.lock().unwrap() += 1;
            if let Some(message) = self.list_error.lock().unwrap().clone() {
                return Err(openai_agents_sdk::error::AgentsError::other(message));
            }
            Ok(self.tools.lock().unwrap().clone())
        }

        async fn call_tool(
            &self,
            tool_name: &str,
            arguments: Option<Map<String, Value>>,
        ) -> Result<Value> {
            self.call_tool_with_meta(tool_name, arguments, None).await
        }

        async fn call_tool_with_meta(
            &self,
            tool_name: &str,
            _arguments: Option<Map<String, Value>>,
            meta: Option<Map<String, Value>>,
        ) -> Result<Value> {
            self.tool_calls.lock().unwrap().push(tool_name.to_string());
            self.tool_metas.lock().unwrap().push(meta);
            let result = self.result.lock().unwrap().clone();
            match result {
                FakeCallResult::Echo => {
                    self.tool_results
                        .lock()
                        .unwrap()
                        .push(format!("result_{tool_name}"));
                    Ok(json!({
                        "content": [{"type": "text", "text": format!("result_{tool_name}")}],
                        "isError": false,
                    }))
                }
                FakeCallResult::Value(value) => {
                    self.tool_results.lock().unwrap().push(value.to_string());
                    Ok(value)
                }
                FakeCallResult::Error(message) => {
                    Err(openai_agents_sdk::error::AgentsError::other(message))
                }
            }
        }

        async fn list_prompts(&self) -> Result<Vec<Value>> {
            Ok(self.prompts.lock().unwrap().clone())
        }

        async fn get_prompt(
            &self,
            name: &str,
            _arguments: Option<BTreeMap<String, String>>,
        ) -> Result<Value> {
            match self.prompt_result.lock().unwrap().clone() {
                Some(result) => Ok(result),
                None => Err(openai_agents_sdk::error::AgentsError::user(format!(
                    "Prompt '{name}' not found"
                ))),
            }
        }

        async fn list_resources(&self, _cursor: Option<&str>) -> Result<Value> {
            match self.resources.lock().unwrap().clone() {
                Some(value) => Ok(value),
                None => Err(openai_agents_sdk::error::AgentsError::user(format!(
                    "MCP server '{}' does not support list_resources.",
                    self.name
                ))),
            }
        }

        async fn list_resource_templates(&self, _cursor: Option<&str>) -> Result<Value> {
            match self.resource_templates.lock().unwrap().clone() {
                Some(value) => Ok(value),
                None => Err(openai_agents_sdk::error::AgentsError::user(format!(
                    "MCP server '{}' does not support list_resource_templates.",
                    self.name
                ))),
            }
        }

        async fn read_resource(&self, _uri: &str) -> Result<Value> {
            match self.resource_contents.lock().unwrap().clone() {
                Some(value) => Ok(value),
                None => Err(openai_agents_sdk::error::AgentsError::user(format!(
                    "MCP server '{}' does not support read_resource.",
                    self.name
                ))),
            }
        }

        async fn session_id(&self) -> Option<String> {
            self.session_id.lock().unwrap().clone()
        }
    }

    /// An `MCPTool` with an empty object schema, as Python's `FakeMCPServer.add_tool`.
    pub fn fake_tool(name: &str) -> MCPTool {
        MCPTool {
            meta: None,
            name: name.to_string(),
            title: None,
            description: None,
            input_schema: json!({}),
            annotations: None,
        }
    }
}
