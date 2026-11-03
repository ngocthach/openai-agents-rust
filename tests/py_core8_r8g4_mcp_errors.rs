//! Parity ports of the user-visible error text of `tests/mcp/test_server_errors.py`: every
//! failed request names its operation and the credential-free server name, and reduces a
//! transport failure to its HTTP status, a lost connection or a timeout.
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::mcp::{MCPServer, MCPServerStreamableHttp, MCPServerStreamableHttpParams};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
    /// A JSON-RPC error object.
    Error(i64, String),
    /// Close the connection without answering.
    Drop,
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
                        // A notification: accepted without a body unless the script says otherwise.
                        log.requests
                            .lock()
                            .unwrap()
                            .push((method.clone(), params.clone()));
                        let raw = match responder(&method, &params) {
                            McpReply::Status(code) => status_only(&format!("{code} Error")),
                            McpReply::Drop => return,
                            _ => status_only("202 Accepted"),
                        };
                        if sock.write_all(raw.as_bytes()).await.is_err() {
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
                        McpReply::Drop => return,
                        McpReply::Error(code, message) => {
                            let payload = json!({"jsonrpc": "2.0", "id": id,
                                "error": {"code": code, "message": message}})
                            .to_string();
                            format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                                 content-length: {}\r\n\r\n{payload}",
                                payload.len()
                            )
                        }
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

const SECRETS: [&str; 3] = ["s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"];

/// The mock URL with credentials, a query secret and a fragment secret spliced in.
fn credentialed(url: &str) -> String {
    let rest = url.strip_prefix("http://").unwrap();
    format!("http://user:s3cr3t_pw@{rest}?api_key=SECRET_QS_KEY#SECRET_FRAGMENT")
}

fn server_for(url: &str) -> MCPServerStreamableHttp {
    let full = credentialed(url);
    MCPServerStreamableHttp::new(
        format!("streamable_http: {full}"),
        MCPServerStreamableHttpParams {
            url: full,
            ..Default::default()
        },
    )
    .with_max_retry_attempts(0)
}

fn assert_no_secrets(error: &AgentsError) {
    let text = format!("{error} {error:?}");
    for secret in SECRETS {
        assert!(!text.contains(secret), "{secret} leaked: {text}");
    }
}

fn erroring(status: u16) -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list"
        | "tools/call"
        | "prompts/list"
        | "prompts/get"
        | "resources/list"
        | "resources/templates/list"
        | "resources/read" => McpReply::Status(status),
        _ => McpReply::Result(json!({})),
    })
}

use openai_agents_sdk::mcp::get_mcp_server_log_name;

fn safe_name(url: &str) -> String {
    get_mcp_server_log_name(&format!("streamable_http: {}", credentialed(url)))
}

fn assert_message(error: &AgentsError, expected: &str) {
    assert_no_secrets(error);
    assert_eq!(error.to_string(), expected);
}

#[tokio::test]
async fn request_errors_name_the_operation_and_hide_credentials() {
    // test_prompt_and_resource_request_errors_hide_url_credentials and
    // test_list_tools/call_tool_http_error_hides_url_credentials_from_exception_graph.
    let (url, _log) = spawn_mcp(erroring(503)).await;
    let server = server_for(&url);
    server.connect().await.unwrap();
    let name = safe_name(&url);
    assert!(!name.contains("s3cr3t_pw"));

    let cases = vec![
        (
            server.list_tools().await.unwrap_err(),
            format!("Failed to list tools from MCP server '{name}': HTTP error 503"),
        ),
        (
            server.call_tool("tool", None).await.unwrap_err(),
            format!("Failed to call tool 'tool' on MCP server '{name}': HTTP error 503"),
        ),
        (
            server.list_prompts().await.unwrap_err(),
            format!("Failed to list prompts on MCP server '{name}': HTTP error 503"),
        ),
        (
            server.get_prompt("safe_prompt", None).await.unwrap_err(),
            format!("Failed to get prompt on MCP server '{name}': HTTP error 503"),
        ),
        (
            server.list_resources(None).await.unwrap_err(),
            format!("Failed to list resources on MCP server '{name}': HTTP error 503"),
        ),
        (
            server.list_resource_templates(None).await.unwrap_err(),
            format!("Failed to list resource templates on MCP server '{name}': HTTP error 503"),
        ),
        (
            server.read_resource("file:///safe.txt").await.unwrap_err(),
            format!("Failed to read resource on MCP server '{name}': HTTP error 503"),
        ),
    ];
    for (error, expected) in &cases {
        assert_message(error, expected);
    }
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn connect_errors_use_the_safe_http_diagnostics() {
    // test_connect_http_error_hides_url_credentials_from_exception_graph and
    // test_call_tool_connect_error_hides_url_credentials_from_exception_graph.
    let (url, _log) = spawn_mcp(Arc::new(|_m: &str, _p: &Value| McpReply::Status(503))).await;
    let error = server_for(&url).connect().await.unwrap_err();
    assert_message(
        &error,
        &format!(
            "Failed to connect to MCP server '{}': HTTP error 503 (Service Unavailable)",
            safe_name(&url)
        ),
    );

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let closed = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let error = server_for(&closed).connect().await.unwrap_err();
    assert_message(
        &error,
        &format!(
            "Failed to connect to MCP server '{}': Could not reach the server.",
            safe_name(&closed)
        ),
    );
}

fn cursor_then_failure(method: &'static str, key: &'static str) -> Responder {
    Arc::new(move |m: &str, p: &Value| {
        if m == "initialize" {
            return McpReply::Result(initialize_result(p));
        }
        if m != method {
            return McpReply::Result(json!({}));
        }
        if p.get("cursor").is_some() {
            McpReply::Status(500)
        } else {
            McpReply::Result(json!({key: [], "nextCursor": "SECRET_OPAQUE_CURSOR"}))
        }
    })
}

#[tokio::test]
async fn paginated_list_failure_does_not_name_the_cursor() {
    // test_paginated_list_failure_does_not_retain_opaque_cursor: the continuation page fails;
    // the error names the server and the failure, never the cursor, and nothing is cached.
    for (method, key, operation) in [
        ("tools/list", "tools", "list tools from"),
        ("prompts/list", "prompts", "list prompts on"),
    ] {
        let (url, _log) = spawn_mcp(cursor_then_failure(method, key)).await;
        let server = server_for(&url).with_cache_tools_list(true);
        server.connect().await.unwrap();
        let error = if key == "tools" {
            server.list_tools().await.unwrap_err()
        } else {
            server.list_prompts().await.unwrap_err()
        };
        assert_message(
            &error,
            &format!(
                "Failed to {operation} MCP server '{}': HTTP error 500",
                safe_name(&url)
            ),
        );
        assert!(!format!("{error:?}").contains("SECRET_OPAQUE_CURSOR"));
        assert!(server.cached_tools().await.is_none());
        server.cleanup().await.unwrap();
    }
}

// ---------------------------------------------------------------------------
// tests/mcp/test_mcp_v2_http.py: the transport-independent HTTP failure behaviour
// ---------------------------------------------------------------------------

/// A server whose first `tools/call` answers 503 and whose later calls succeed; also reports
/// every status it returned for `tools/call`.
fn first_tool_call_fails(statuses: Arc<Mutex<Vec<u16>>>) -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/call" => {
            let mut seen = statuses.lock().unwrap();
            if seen.is_empty() {
                seen.push(503);
                McpReply::Status(503)
            } else {
                seen.push(200);
                McpReply::Result(json!({"content": [{"type": "text", "text": "ok"}]}))
            }
        }
        "tools/list" => McpReply::Result(json!({"tools": [
            {"name": "test", "inputSchema": {"type": "object", "properties": {}}}]})),
        _ => McpReply::Result(json!({})),
    })
}

fn plain_server(url: &str) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "v2-http",
        MCPServerStreamableHttpParams {
            url: url.to_string(),
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn a_5xx_fails_only_that_request() {
    // test_v2_streamable_http_5xx_fails_only_that_request
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let (url, _log) = spawn_mcp(first_tool_call_fails(statuses.clone())).await;
    let server = plain_server(&url).with_max_retry_attempts(0);
    server.connect().await.unwrap();

    let error = server.call_tool("test", None).await.unwrap_err();
    assert!(error.to_string().contains("HTTP error 503"), "{error}");
    let result = server.call_tool("test", None).await.unwrap();
    assert_eq!(result["content"][0]["text"], "ok");
    assert_eq!(server.list_tools().await.unwrap().len(), 1);
    assert_eq!(*statuses.lock().unwrap(), vec![503, 200]);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_5xx_is_retried_with_a_retry_budget() {
    // test_v2_streamable_http_retries_5xx_on_shared_session
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let (url, _log) = spawn_mcp(first_tool_call_fails(statuses.clone())).await;
    let server = plain_server(&url)
        .with_max_retry_attempts(1)
        .with_retry_backoff_seconds_max(Some(0.0));
    server.connect().await.unwrap();

    let result = server.call_tool("test", None).await.unwrap();
    assert_eq!(result["content"][0]["text"], "ok");
    assert_eq!(*statuses.lock().unwrap(), vec![503, 200]);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_handshake_5xx_fails_connect_and_leaves_no_session() {
    // test_v2_streamable_http_handshake_5xx_fails_connect_without_legacy_fallback
    let (url, log) = spawn_mcp(Arc::new(|_m: &str, _p: &Value| McpReply::Status(503))).await;
    let server = plain_server(&url);
    let error = server.connect().await.unwrap_err();
    assert!(error.to_string().contains("HTTP error 503"), "{error}");
    assert_eq!(log.methods(), vec!["initialize".to_string()]);
    assert!(server.list_tools().await.is_err());
}

// ---------------------------------------------------------------------------
// tests/mcp/test_prompt_server.py over a real MCP server
// ---------------------------------------------------------------------------

use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, Runner};

/// Python's `FakeMCPPromptServer.get_prompt`: `{name}` placeholders are filled from the
/// arguments, and a template that cannot be filled is returned unchanged.
fn prompt_server() -> Responder {
    const PROMPTS: [(&str, &str, &str); 4] = [
        (
            "generate_code_review_instructions",
            "Generate agent instructions for code review tasks",
            "You are a senior {language} code review specialist. Focus on {focus}.",
        ),
        (
            "generate_testing_instructions",
            "Generate agent instructions for testing tasks",
            "You are a test engineer.",
        ),
        (
            "simple_prompt",
            "A simple prompt",
            "You are a helpful assistant.",
        ),
        (
            "incomplete_prompt",
            "Prompt with missing arguments",
            "You are a {role} working on {task}.",
        ),
    ];
    Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" => McpReply::Result(json!({"tools": []})),
        "prompts/list" => McpReply::Result(json!({"prompts": PROMPTS
            .iter()
            .map(|(name, description, _)| json!({"name": name, "description": description}))
            .collect::<Vec<_>>()})),
        "prompts/get" => {
            let name = p["name"].as_str().unwrap_or("");
            let Some((_, _, template)) = PROMPTS.iter().find(|(n, _, _)| *n == name) else {
                return McpReply::Error(-32602, format!("Prompt '{name}' not found"));
            };
            let mut text = template.to_string();
            let mut complete = true;
            if let Some(arguments) = p["arguments"].as_object() {
                for (key, value) in arguments {
                    text = text.replace(&format!("{{{key}}}"), value.as_str().unwrap_or(""));
                }
                complete = !text.contains('{');
            }
            if !complete {
                text = template.to_string();
            }
            McpReply::Result(json!({
                "description": format!("Generated prompt for {name}"),
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            }))
        }
        _ => McpReply::Result(json!({})),
    })
}

fn prompt_text(result: &Value) -> String {
    result["messages"][0]["content"]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

fn arguments(pairs: &[(&str, &str)]) -> Option<std::collections::BTreeMap<String, String>> {
    Some(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    )
}

#[tokio::test]
async fn prompts_list_and_render_through_a_server() {
    // test_list_prompts, test_multiple_prompts, test_prompt_with_complex_arguments (four
    // placeholders are filled by the same substitution as the two above), test_get_prompt_without_arguments,
    // test_get_prompt_with_arguments, test_prompt_with_missing_arguments,
    // test_prompt_server_cleanup.
    let (url, log) = spawn_mcp(prompt_server()).await;
    let server = plain_server(&url);
    server.connect().await.unwrap();

    let prompts = server.list_prompts().await.unwrap();
    let names: Vec<&str> = prompts
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"generate_code_review_instructions"));
    assert!(names.contains(&"generate_testing_instructions"));
    let review = prompts
        .iter()
        .find(|p| p["name"] == "generate_code_review_instructions")
        .unwrap();
    assert!(
        review["description"]
            .as_str()
            .unwrap()
            .contains("code review")
    );

    let simple = server.get_prompt("simple_prompt", None).await.unwrap();
    assert_eq!(prompt_text(&simple), "You are a helpful assistant.");

    let rendered = server
        .get_prompt(
            "generate_code_review_instructions",
            arguments(&[
                ("focus", "security vulnerabilities"),
                ("language", "python"),
            ]),
        )
        .await
        .unwrap();
    assert_eq!(
        prompt_text(&rendered),
        "You are a senior python code review specialist. Focus on security vulnerabilities."
    );
    let wire = log.calls_of("prompts/get");
    assert_eq!(wire[1]["arguments"]["language"], "python");

    let testing = server
        .get_prompt("generate_testing_instructions", None)
        .await
        .unwrap();
    assert_eq!(prompt_text(&testing), "You are a test engineer.");

    let partial = server
        .get_prompt("incomplete_prompt", arguments(&[("role", "developer")]))
        .await
        .unwrap();
    assert_eq!(prompt_text(&partial), "You are a {role} working on {task}.");

    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn an_unknown_prompt_is_an_error_naming_it() {
    // test_get_prompt_not_found
    let (url, _log) = spawn_mcp(prompt_server()).await;
    let server = plain_server(&url);
    server.connect().await.unwrap();
    let error = server.get_prompt("nonexistent", None).await.unwrap_err();
    assert!(
        error.to_string().contains("Prompt 'nonexistent' not found"),
        "{error}"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn an_agent_runs_with_instructions_taken_from_a_prompt() {
    // test_agent_with_prompt_instructions and test_agent_with_prompt_instructions_streaming.
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, _log) = spawn_mcp(prompt_server()).await;
    let server = Arc::new(plain_server(&url)) as Arc<dyn MCPServer>;
    server.connect().await.unwrap();
    let prompt = server
        .get_prompt(
            "generate_code_review_instructions",
            arguments(&[("language", "Python"), ("focus", "security")]),
        )
        .await
        .unwrap();
    let instructions = prompt_text(&prompt);
    assert_eq!(
        instructions,
        "You are a senior Python code review specialist. Focus on security."
    );

    for streaming in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_turns(vec![vec![assistant_message("Security analysis complete.")]]);
        let agent: Agent<()> = Agent::new("prompt_agent")
            .with_instructions(instructions.clone())
            .with_mcp_server(server.clone())
            .with_model_instance(model.clone());
        let output = if streaming {
            let mut result = Runner::run_streamed(&agent, "Review code");
            {
                use futures::StreamExt;
                let mut events = Box::pin(result.stream_events());
                while events.next().await.is_some() {}
            }
            result.final_output_text()
        } else {
            Runner::run(&agent, "Review code")
                .await
                .unwrap()
                .final_output_text()
        };
        assert_eq!(output.as_deref(), Some("Security analysis complete."));
    }
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// tests/mcp/test_list_page_limits.py::test_nonpositive_page_limit_rejected_before_connect
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_zero_page_limit_is_rejected_before_anything_connects() {
    use openai_agents_sdk::mcp::{
        MCPServerSse, MCPServerSseParams, MCPServerStdio, MCPServerStdioParams,
    };
    // Python raises while building the server (negative limits cannot be spelled with a
    // `usize`); the Rust builders cannot fail, so connecting raises the same message first.
    let (url, log) = spawn_mcp(prompt_server()).await;
    let stdio = MCPServerStdio::new(
        "stdio",
        MCPServerStdioParams {
            command: "this-command-must-never-be-spawned".into(),
            ..Default::default()
        },
    )
    .with_max_list_pages(Some(0));
    let sse = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: url.clone(),
            headers: Default::default(),
            timeout: std::time::Duration::from_secs(1),
            sse_read_timeout: std::time::Duration::from_secs(1),
            auth: None,
            http_client: None,
        },
    )
    .with_max_list_pages(Some(0));
    let http = plain_server(&url).with_max_list_pages(Some(0));

    for server in [
        &stdio as &dyn MCPServer,
        &sse as &dyn MCPServer,
        &http as &dyn MCPServer,
    ] {
        let error = server.connect().await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("max_list_pages must be a positive integer or None"),
            "{error}"
        );
    }
    assert!(log.methods().is_empty(), "nothing may reach the server");
}

// ---------------------------------------------------------------------------
// tests/mcp/test_streamable_http_client_factory.py: the tolerant `initialized` notification
// ---------------------------------------------------------------------------

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());
static LOG_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        // Only the SDK's own records (third-party crates log their own diagnostics).
        if record.target() != "openai.agents" {
            return;
        }
        RECORDS
            .lock()
            .unwrap()
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

fn install_capture() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// A server that rejects the `initialized` notification the given way and serves tools.
fn rejects_initialized(reply: fn() -> McpReply) -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "notifications/initialized" => reply(),
        "tools/list" => McpReply::Result(json!({"tools": [
            {"name": "test", "inputSchema": {"type": "object", "properties": {}}}]})),
        _ => McpReply::Result(json!({})),
    })
}

fn tolerant(url: String, ignore: bool) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "tolerant",
        MCPServerStreamableHttpParams {
            url,
            ignore_initialized_notification_failure: ignore,
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn a_failed_initialized_notification_is_tolerated_only_when_asked() {
    // test_initialized_notification_failure_returns_synthetic_success and
    // test_initialized_notification_transport_exception_returns_synthetic_success, plus the
    // default (strict) behaviour they are the opt-out of.
    for reply in [
        (|| McpReply::Status(503)) as fn() -> McpReply,
        (|| McpReply::Drop) as fn() -> McpReply,
    ] {
        let (url, _log) = spawn_mcp(rejects_initialized(reply)).await;
        let tolerant = tolerant(url.clone(), true);
        tolerant.connect().await.unwrap();
        assert_eq!(tolerant.list_tools().await.unwrap().len(), 1);
        tolerant.cleanup().await.unwrap();

        let strict = self::tolerant(url, false);
        assert!(strict.connect().await.is_err());
    }
}

#[tokio::test]
async fn the_tolerant_transport_preserves_other_failures() {
    // test_transport_preserves_non_initialized_failures
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" => McpReply::Drop,
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = tolerant(url, true).with_max_retry_attempts(0);
    server.connect().await.unwrap();
    assert!(server.list_tools().await.is_err());
}

#[tokio::test]
async fn the_ignored_initialized_failure_log_hides_url_credentials() {
    // test_initialized_notification_transport_exception_hides_url_credentials_in_log
    use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
    let _guard = LOG_LOCK.lock().await;
    install_capture();
    for redacted in [true, false] {
        set_dont_log_tool_data(redacted);
        RECORDS.lock().unwrap().clear();
        let (url, _log) = spawn_mcp(rejects_initialized(|| McpReply::Drop)).await;
        let server = tolerant(credentialed(&url), true);
        server.connect().await.unwrap();
        server.cleanup().await.unwrap();
        let records = RECORDS.lock().unwrap().clone();
        assert!(
            records
                .iter()
                .any(|(_, text)| text.contains("initialized notification")),
            "{redacted}: {records:?}"
        );
        for (_, text) in &records {
            for secret in SECRETS {
                assert!(
                    !text.contains(secret),
                    "{redacted}: {secret} leaked: {text}"
                );
            }
        }
    }
    reset_dont_log_tool_data();
}

// ---------------------------------------------------------------------------
// tests/mcp/test_client_session_retries.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_server_reported_408_is_retried_on_an_isolated_session() {
    // test_streamable_http_retries_mcp_408_on_isolated_session, and the contrast: any other
    // JSON-RPC error is retried on the shared session.
    for (code, retried) in [(408, true), (-32602, false)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let initializes = Arc::new(AtomicUsize::new(0));
        let (seen_calls, seen_init) = (calls.clone(), initializes.clone());
        let (url, _log) = spawn_mcp(Arc::new(move |m: &str, p: &Value| match m {
            "initialize" => {
                seen_init.fetch_add(1, Ordering::SeqCst);
                McpReply::Result(initialize_result(p))
            }
            "tools/call" => {
                if seen_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    McpReply::Error(code, "Timed out while waiting for response".into())
                } else {
                    McpReply::Result(json!({"content": [{"type": "text", "text": "ok"}]}))
                }
            }
            _ => McpReply::Result(json!({})),
        }))
        .await;
        let server = plain_server(&url)
            .with_max_retry_attempts(1)
            .with_retry_backoff_seconds_max(Some(0.0));
        server.connect().await.unwrap();
        // Either way the retry budget makes a second attempt; only a 408 earns a new session.
        let outcome = server.call_tool("tool", None).await.unwrap();
        assert_eq!(outcome["content"][0]["text"], "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "code {code}");
        let expected_sessions = if retried { 2 } else { 1 };
        assert_eq!(
            initializes.load(Ordering::SeqCst),
            expected_sessions,
            "code {code}"
        );
        server.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn the_backoff_cap_is_forwarded_and_validated() {
    // test_retry_backoff_seconds_max_is_forwarded_by_public_servers and
    // test_retry_backoff_seconds_max_rejects_negative_and_non_finite_values (a non-numeric
    // value cannot be spelled with `Option<f64>`).
    use openai_agents_sdk::mcp::{
        MCPServerSse, MCPServerSseParams, MCPServerStdio, MCPServerStdioParams,
    };
    let sse_params = || MCPServerSseParams {
        url: "https://example.test/sse".into(),
        headers: Default::default(),
        timeout: std::time::Duration::from_secs(1),
        sse_read_timeout: std::time::Duration::from_secs(1),
        auth: None,
        http_client: None,
    };
    let stdio = MCPServerStdio::new("stdio", MCPServerStdioParams::default())
        .with_retry_backoff_seconds_max(Some(0.0));
    let sse = MCPServerSse::new("sse", sse_params()).with_retry_backoff_seconds_max(Some(0.0));
    let http = plain_server("https://example.test/mcp").with_retry_backoff_seconds_max(Some(0.0));
    for server in [
        &stdio as &dyn MCPServer,
        &sse as &dyn MCPServer,
        &http as &dyn MCPServer,
    ] {
        assert_eq!(server.config().retry_backoff_seconds_max, Some(0.0));
    }

    for value in [-1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        let server =
            plain_server("https://example.test/mcp").with_retry_backoff_seconds_max(Some(value));
        let error = server.connect().await.unwrap_err();
        assert!(
            error.to_string().contains("non-negative finite"),
            "{value}: {error}"
        );
    }
}

// ---------------------------------------------------------------------------
// tests/mcp/test_message_handler.py::test_message_handler_propagates_to_server_base
// ---------------------------------------------------------------------------

#[test]
fn a_message_handler_reaches_every_public_server_type() {
    use openai_agents_sdk::mcp::{
        MCPServerSse, MCPServerSseParams, MCPServerStdio, MCPServerStdioParams,
    };
    let stdio = MCPServerStdio::new(
        "stdio",
        MCPServerStdioParams {
            command: "python".into(),
            ..Default::default()
        },
    )
    .with_message_handler(|_: &Value| {});
    let sse = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: "https://example.com".into(),
            headers: Default::default(),
            timeout: std::time::Duration::from_secs(1),
            sse_read_timeout: std::time::Duration::from_secs(1),
            auth: None,
            http_client: None,
        },
    )
    .with_message_handler(|_: &Value| {});
    let http = plain_server("https://example.com").with_message_handler(|_: &Value| {});
    for server in [
        &stdio as &dyn MCPServer,
        &sse as &dyn MCPServer,
        &http as &dyn MCPServer,
    ] {
        assert!(server.config().message_handler.is_some());
    }
}
