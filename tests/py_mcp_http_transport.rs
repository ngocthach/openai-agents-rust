//! Parity port of the behavioural core of `tests/mcp/test_client_session_retries.py`,
//! `test_mcp_resources.py`, `test_prompt_server.py`, `test_streamable_http_session_id.py`,
//! `test_mcp_auth_params.py` and `test_streamable_http_client_factory.py`.
//!
//! Python mocks `ClientSession`; Rust's transports own their session, so every test here talks to
//! a local Streamable HTTP MCP server that scripts the responses.
#![allow(clippy::field_reassign_with_default)] // mirrors Python fixtures that tweak one field of a default config
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openai_agents_sdk::mcp::{MCPServer, MCPServerStreamableHttp, MCPServerStreamableHttpParams};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

mod py_mcpmem_support;

// ---------------------------------------------------------------------------
// A scriptable Streamable HTTP MCP server
// ---------------------------------------------------------------------------

/// How the mock answers one JSON-RPC request.
#[derive(Clone)]
pub enum McpReply {
    /// A JSON-RPC `result` payload.
    Result(Value),
    /// A bare HTTP status, which `rmcp` surfaces as a transport failure.
    Status(u16),
}

type Responder = Arc<dyn Fn(&str, &Value) -> McpReply + Send + Sync>;

/// What a mock recorded: every `(method, params)` pair it was asked for.
#[derive(Default, Debug)]
pub struct McpLog {
    pub requests: Mutex<Vec<(String, Value)>>,
    pub authorizations: Mutex<Vec<String>>,
    pub deletes: AtomicUsize,
}

impl McpLog {
    pub fn methods(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|(m, _)| m.clone())
            .collect()
    }

    pub fn calls_of(&self, method: &str) -> Vec<Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, _)| m == method)
            .map(|(_, p)| p.clone())
            .collect()
    }
}

/// Spawn a mock MCP server; returns its URL and the request log.
pub async fn spawn_mcp(responder: Responder) -> (String, Arc<McpLog>) {
    let log = Arc::new(McpLog::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_log = log.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let responder = responder.clone();
            let log = server_log.clone();
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let header_end = loop {
                        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break p + 4;
                        }
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    if let Some(auth) = head.lines().find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("authorization")
                            .then(|| v.trim().to_string())
                    }) {
                        log.authorizations.lock().unwrap().push(auth);
                    }
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse().ok())?
                        })
                        .unwrap_or(0);
                    while buf.len() < header_end + len {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    let body = buf[header_end..header_end + len].to_vec();
                    buf.drain(..header_end + len);
                    let status_only =
                        |code: &str| format!("HTTP/1.1 {code}\r\ncontent-length: 0\r\n\r\n");

                    if head.starts_with("GET ") {
                        if sock
                            .write_all(status_only("405 Method Not Allowed").as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    if head.starts_with("DELETE ") {
                        log.deletes.fetch_add(1, Ordering::SeqCst);
                        if sock
                            .write_all(status_only("200 OK").as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    let Ok(req) = serde_json::from_slice::<Value>(&body) else {
                        let _ = sock
                            .write_all(status_only("400 Bad Request").as_bytes())
                            .await;
                        return;
                    };
                    let method = req["method"].as_str().unwrap_or("").to_string();
                    let params = req.get("params").cloned().unwrap_or(Value::Null);
                    let Some(id) = req.get("id").cloned() else {
                        // A notification: accept it without a body.
                        log.requests.lock().unwrap().push((method, params));
                        if sock
                            .write_all(status_only("202 Accepted").as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    };
                    log.requests
                        .lock()
                        .unwrap()
                        .push((method.clone(), params.clone()));
                    let reply = responder(&method, &params);
                    let raw = match reply {
                        McpReply::Status(code) => status_only(&format!("{code} Error")),
                        McpReply::Result(result) => {
                            let payload =
                                json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
                            let session = if method == "initialize" {
                                "mcp-session-id: sess-pagination\r\n"
                            } else {
                                ""
                            };
                            format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{session}\
                                 content-length: {}\r\n\r\n{payload}",
                                payload.len()
                            )
                        }
                    };
                    if sock.write_all(raw.as_bytes()).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    (format!("http://{addr}/mcp"), log)
}

/// The `initialize` result every mock returns.
pub fn initialize_result(params: &Value) -> Value {
    json!({
        "protocolVersion": params["protocolVersion"],
        "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
        "serverInfo": {"name": "page-mock", "version": "1"},
    })
}

fn params(url: &str) -> MCPServerStreamableHttpParams {
    MCPServerStreamableHttpParams {
        url: url.to_string(),
        ..Default::default()
    }
}

fn server(url: &str) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new("http-mock", params(url))
}

fn tool_result(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}]})
}

/// A server whose first `fail` `tools/call` requests answer with `status`.
fn failing_calls(fail: usize, status: u16) -> Responder {
    let seen = Arc::new(AtomicUsize::new(0));
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/call" => {
            if seen.fetch_add(1, Ordering::SeqCst) < fail {
                McpReply::Status(status)
            } else {
                McpReply::Result(tool_result("ok"))
            }
        }
        _ => McpReply::Result(json!({})),
    })
}

fn text_of(result: &Value) -> &str {
    result["content"][0]["text"].as_str().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Retries (test_client_session_retries.py)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn call_tool_retries_until_success() {
    let (url, log) = spawn_mcp(failing_calls(2, 503)).await;
    let server = server(&url).with_max_retry_attempts(2);
    server.connect().await.unwrap();
    // Backoff base defaults to 1s in Rust; keep the test fast.
    let result = server.call_tool("tool", None).await.unwrap();
    assert_eq!(text_of(&result), "ok");
    assert_eq!(log.calls_of("tools/call").len(), 3);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn list_tools_unlimited_retries() {
    let seen = Arc::new(AtomicUsize::new(0));
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" => {
            if seen.fetch_add(1, Ordering::SeqCst) < 3 {
                McpReply::Status(503)
            } else {
                McpReply::Result(json!({"tools": [{"name": "tool", "inputSchema": {}}]}))
            }
        }
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.max_retry_attempts = -1;
    config.retry_backoff_seconds_base = 0.0;
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();
    let tools = server.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "tool");
    assert_eq!(log.calls_of("tools/list").len(), 4);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn streamable_http_retries_5xx_on_isolated_session() {
    let (url, log) = spawn_mcp(failing_calls(1, 504)).await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.max_retry_attempts = 1;
    config.retry_backoff_seconds_base = 0.0;
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();

    let result = server.call_tool("tool", None).await.unwrap();
    assert_eq!(text_of(&result), "ok");
    // The retry ran on a freshly opened session.
    assert_eq!(log.calls_of("initialize").len(), 2);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn streamable_http_does_not_retry_4xx_on_isolated_session() {
    let (url, log) = spawn_mcp(failing_calls(5, 401)).await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.retry_backoff_seconds_base = 0.0;
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();

    let error = server.call_tool("tool", None).await.unwrap_err();
    assert!(error.to_string().contains("401"), "{error}");
    assert_eq!(log.calls_of("initialize").len(), 1);
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn streamable_http_does_not_isolated_retry_without_retry_budget() {
    let (url, log) = spawn_mcp(failing_calls(5, 503)).await;
    let server = server(&url).with_max_retry_attempts(0);
    server.connect().await.unwrap();

    assert!(server.call_tool("tool", None).await.is_err());
    assert_eq!(log.calls_of("initialize").len(), 1);
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn streamable_http_counts_isolated_retry_against_retry_budget() {
    let (url, log) = spawn_mcp(failing_calls(100, 503)).await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.max_retry_attempts = 2;
    config.retry_backoff_seconds_base = 0.0;
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();

    assert!(server.call_tool("tool", None).await.is_err());
    // Python: two shared attempts and one isolated retry; the retry spends budget.
    assert_eq!(log.calls_of("tools/call").len(), 3);
    assert_eq!(log.calls_of("initialize").len(), 2);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn retry_backoff_seconds_max_caps_the_backoff() {
    let (url, _log) = spawn_mcp(failing_calls(3, 503)).await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.max_retry_attempts = 3;
    config.retry_backoff_seconds_base = 0.3;
    config.retry_backoff_seconds_max = Some(0.01);
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();

    let started = Instant::now();
    let result = server.call_tool("tool", None).await.unwrap();
    assert_eq!(text_of(&result), "ok");
    // Uncapped, the three backoffs would take 0.3 + 0.6 + 1.2 seconds.
    assert!(
        started.elapsed() < Duration::from_millis(900),
        "{:?}",
        started.elapsed()
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn generic_backoff_remains_uncapped_by_default() {
    let (url, _log) = spawn_mcp(failing_calls(2, 503)).await;
    let mut config = openai_agents_sdk::mcp::MCPServerConfig::default();
    config.max_retry_attempts = 2;
    config.retry_backoff_seconds_base = 0.1;
    let server = server(&url).with_config(config);
    server.connect().await.unwrap();

    let started = Instant::now();
    server.call_tool("tool", None).await.unwrap();
    // The first failure is replayed on an isolated session, which fails too; one 0.1s backoff
    // precedes the shared retry that succeeds.
    assert!(
        started.elapsed() >= Duration::from_millis(95),
        "{:?}",
        started.elapsed()
    );
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// Required-parameter validation
// ---------------------------------------------------------------------------

fn schema_server(schema: Value) -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" => {
            McpReply::Result(json!({"tools": [{"name": "tool", "inputSchema": schema}]}))
        }
        "tools/call" => McpReply::Result(tool_result("called")),
        _ => McpReply::Result(json!({})),
    })
}

#[tokio::test]
async fn call_tool_validates_required_parameters_before_remote_call() {
    let (url, log) = spawn_mcp(schema_server(json!({
        "type": "object",
        "properties": {"param_a": {"type": "string"}},
        "required": ["param_a"],
    })))
    .await;
    let server = server(&url).with_cache_tools_list(true);
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();

    let error = server
        .call_tool("tool", Some(serde_json::Map::new()))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing required parameters: param_a"),
        "{error}"
    );
    assert!(log.calls_of("tools/call").is_empty());

    // The same call with the parameter still reaches the server.
    let mut args = serde_json::Map::new();
    args.insert("param_a".into(), json!("value"));
    server.call_tool("tool", Some(args)).await.unwrap();
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn call_tool_validates_required_parameters_when_arguments_is_none() {
    let (url, log) = spawn_mcp(schema_server(json!({"required": ["param_a"]}))).await;
    let server = server(&url).with_cache_tools_list(true);
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();

    let error = server.call_tool("tool", None).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing required parameters: param_a"),
        "{error}"
    );
    assert!(log.calls_of("tools/call").is_empty());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn call_tool_skips_validation_without_a_cached_tool_or_required_list() {
    // Tool missing from the cache.
    let (url, log) = spawn_mcp(schema_server(json!({"required": ["param_a"]}))).await;
    let server = server(&url).with_cache_tools_list(true);
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();
    server.call_tool("other_tool", None).await.unwrap();
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();

    // No `required` list.
    let (url, log) = spawn_mcp(schema_server(json!({"type": "object"}))).await;
    let server = self::server(&url).with_cache_tools_list(true);
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();
    server.call_tool("tool", None).await.unwrap();
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();

    // Cache not populated (caching disabled): validation does not apply.
    let (url, log) = spawn_mcp(schema_server(json!({"required": ["param_a"]}))).await;
    let server = self::server(&url);
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();
    server.call_tool("tool", None).await.unwrap();
    assert_eq!(log.calls_of("tools/call").len(), 1);
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// Not connected (test_mcp_resources.py, test_server_errors.py)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn requests_before_connect_raise_a_user_error() {
    let server = server("http://127.0.0.1:1/mcp");
    let message = "Server not initialized. Make sure you call `connect()` first.";
    for error in [
        server.list_tools().await.unwrap_err(),
        server.call_tool("foo", None).await.unwrap_err(),
        server.list_prompts().await.unwrap_err(),
        server.get_prompt("p", None).await.unwrap_err(),
        server.list_resources(None).await.unwrap_err(),
        server.list_resource_templates(None).await.unwrap_err(),
        server.read_resource("file:///etc/hosts").await.unwrap_err(),
    ] {
        assert!(error.to_string().contains(message), "{error}");
        assert!(matches!(
            error,
            openai_agents_sdk::error::AgentsError::UserError { .. }
        ));
    }
}

// ---------------------------------------------------------------------------
// Resources and prompts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resources_round_trip_and_forward_the_cursor() {
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "resources/list" => McpReply::Result(if p.get("cursor").is_some() {
            json!({"resources": []})
        } else {
            json!({"resources": [{"uri": "file:///readme.md", "name": "readme.md", "mimeType": "text/markdown"}]})
        }),
        "resources/templates/list" => McpReply::Result(if p.get("cursor").is_some() {
            json!({"resourceTemplates": []})
        } else {
            json!({"resourceTemplates": [{"uriTemplate": "file:///{path}", "name": "file"}]})
        }),
        "resources/read" => McpReply::Result(json!({"contents": [{
            "uri": p["uri"], "text": "# Hello", "mimeType": "text/markdown"}]})),
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = server(&url);
    server.connect().await.unwrap();

    let first = server.list_resources(None).await.unwrap();
    assert_eq!(first["resources"][0]["uri"], "file:///readme.md");
    let page2 = server.list_resources(Some("tok_abc")).await.unwrap();
    assert_eq!(page2["resources"], json!([]));
    let calls = log.calls_of("resources/list");
    assert!(calls[0].get("cursor").is_none_or(Value::is_null));
    assert_eq!(calls[1]["cursor"], "tok_abc");

    let templates = server.list_resource_templates(None).await.unwrap();
    assert_eq!(
        templates["resourceTemplates"][0]["uriTemplate"],
        "file:///{path}"
    );
    server
        .list_resource_templates(Some("tok_xyz"))
        .await
        .unwrap();
    assert_eq!(
        log.calls_of("resources/templates/list")[1]["cursor"],
        "tok_xyz"
    );

    let read = server.read_resource("file:///readme.md").await.unwrap();
    assert_eq!(read["contents"][0]["text"], "# Hello");
    assert_eq!(
        log.calls_of("resources/read")[0]["uri"],
        "file:///readme.md"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn default_trait_methods_report_unsupported_resources() {
    // `test_base_methods_raise_not_implemented`: Python raises NotImplementedError; the Rust
    // trait default is a UserError naming the method.
    let minimal = py_mcpmem_support::FakeMcpServer::new("minimal");
    for (error, method) in [
        (
            minimal.list_resources(None).await.unwrap_err(),
            "list_resources",
        ),
        (
            minimal.list_resource_templates(None).await.unwrap_err(),
            "list_resource_templates",
        ),
        (
            minimal.read_resource("file:///test.txt").await.unwrap_err(),
            "read_resource",
        ),
    ] {
        assert!(error.to_string().contains(method), "{error}");
    }
}

#[tokio::test]
async fn prompts_are_listed_and_fetched_with_arguments() {
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "prompts/list" => McpReply::Result(json!({"prompts": [
            {"name": "generate_code_review_instructions", "description": "Generate agent instructions for code review tasks"},
            {"name": "generate_testing_instructions", "description": "Generate testing instructions"},
        ]})),
        "prompts/get" => {
            let language = p["arguments"]["language"].as_str().unwrap_or("{language}");
            McpReply::Result(json!({
                "description": format!("Generated prompt for {}", p["name"]),
                "messages": [{"role": "user", "content": {"type": "text",
                    "text": format!("You are a senior {language} code review specialist.")}}],
            }))
        }
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = server(&url);
    server.connect().await.unwrap();

    let prompts = server.list_prompts().await.unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(
        prompts[0]["description"]
            .as_str()
            .unwrap()
            .contains("code review")
    );

    let mut args = std::collections::BTreeMap::new();
    args.insert("language".to_string(), "python".to_string());
    let prompt = server
        .get_prompt("generate_code_review_instructions", Some(args))
        .await
        .unwrap();
    assert_eq!(
        prompt["messages"][0]["content"]["text"],
        "You are a senior python code review specialist."
    );
    assert_eq!(
        log.calls_of("prompts/get")[0]["arguments"]["language"],
        "python"
    );
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// Session id (test_streamable_http_session_id.py)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_id_lifecycle() {
    let (url, _log) = spawn_mcp(failing_calls(0, 503)).await;
    let server = server(&url);
    assert_eq!(server.session_id().await, None);

    server.connect().await.unwrap();
    assert_eq!(
        server.session_id().await.as_deref(),
        Some("sess-pagination")
    );

    server.cleanup().await.unwrap();
    assert_eq!(server.session_id().await, None);
}

// ---------------------------------------------------------------------------
// Auth (test_mcp_auth_params.py)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streamable_http_forwards_auth_and_headers() {
    let (url, log) = spawn_mcp(failing_calls(0, 503)).await;

    let mut p = params(&url);
    p.auth = Some(Arc::new(|| Some("token-from-auth".to_string())));
    let server = MCPServerStreamableHttp::new("auth", p);
    server.connect().await.unwrap();
    server.call_tool("tool", None).await.unwrap();
    server.cleanup().await.unwrap();
    assert!(
        log.authorizations
            .lock()
            .unwrap()
            .iter()
            .all(|a| a == "Bearer token-from-auth"),
        "{:?}",
        log.authorizations
    );
    assert!(!log.authorizations.lock().unwrap().is_empty());

    // No auth: no Authorization header at all.
    let (url, log) = spawn_mcp(failing_calls(0, 503)).await;
    let server = self::server(&url);
    server.connect().await.unwrap();
    server.cleanup().await.unwrap();
    assert!(log.authorizations.lock().unwrap().is_empty());

    // A `Bearer ` prefix in the provider's value is not doubled.
    let (url, log) = spawn_mcp(failing_calls(0, 503)).await;
    let mut p = params(&url);
    p.auth = Some(Arc::new(|| Some("Bearer already".to_string())));
    let server = MCPServerStreamableHttp::new("auth", p);
    server.connect().await.unwrap();
    server.cleanup().await.unwrap();
    assert_eq!(log.authorizations.lock().unwrap()[0], "Bearer already");
}

// ---------------------------------------------------------------------------
// Session termination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminate_on_close_controls_the_delete_request() {
    for terminate in [true, false] {
        let (url, log) = spawn_mcp(failing_calls(0, 503)).await;
        let mut p = params(&url);
        p.terminate_on_close = terminate;
        let server = MCPServerStreamableHttp::new("t", p);
        server.connect().await.unwrap();
        server.cleanup().await.unwrap();
        assert_eq!(
            log.deletes.load(Ordering::SeqCst),
            usize::from(terminate),
            "terminate_on_close={terminate}"
        );
    }
}
