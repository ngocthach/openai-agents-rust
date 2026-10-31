//! Parity port of the behavioural core of `tests/mcp/test_server_errors.py` and
//! `tests/mcp/test_connect_disconnect.py`: connection errors, credential-free error text and
//! the connect/cleanup lifecycle.
//!
//! The Python file mostly pins how exception graphs, traceback locals and `ExceptionGroup`s
//! are scrubbed; Rust errors are plain strings built from the sanitized server name, so the
//! equivalent contract is that no secret from the URL or from the server's reply reaches the
//! error text. Those are asserted here against a local mock server whose URL carries
//! credentials.
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

#[tokio::test]
async fn request_errors_hide_url_credentials() {
    // test_*_hide_url_credentials_from_exception_graph for list_tools / call_tool / prompts /
    // resources over a credentialed URL.
    let (url, _log) = spawn_mcp(erroring(503)).await;
    let server = server_for(&url);
    server.connect().await.unwrap();

    let errors = vec![
        server.list_tools().await.unwrap_err(),
        server.call_tool("tool", None).await.unwrap_err(),
        server.list_prompts().await.unwrap_err(),
        server.get_prompt("p", None).await.unwrap_err(),
        server.list_resources(None).await.unwrap_err(),
        server.list_resource_templates(None).await.unwrap_err(),
        server.read_resource("file:///safe.txt").await.unwrap_err(),
    ];
    for error in &errors {
        assert_no_secrets(error);
    }
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn connect_http_error_hides_url_credentials() {
    // The initialize request itself is answered with a 5xx.
    let (url, _log) = spawn_mcp(Arc::new(|_m: &str, _p: &Value| McpReply::Status(503))).await;
    let server = server_for(&url);
    let error = server.connect().await.unwrap_err();
    assert_no_secrets(&error);
    // A failed connect leaves the server disconnected.
    assert!(server.list_tools().await.is_err());
}

#[tokio::test]
async fn connect_to_a_closed_port_hides_url_credentials() {
    // `test_call_tool_connect_error_hides_url_credentials_from_exception_graph`.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let server = server_for(&url);
    let error = server.connect().await.unwrap_err();
    assert_no_secrets(&error);
}

#[tokio::test]
async fn failed_connection_cleans_up_and_allows_a_retry() {
    // `test_server_errors_cause_error_and_cleanup_called`: after a failed connect the server
    // holds no session, and a later connect can succeed.
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let (url, _log) = spawn_mcp(Arc::new(move |m: &str, p: &Value| {
        if m == "initialize" && seen.fetch_add(1, Ordering::SeqCst) == 0 {
            return McpReply::Status(503);
        }
        match m {
            "initialize" => McpReply::Result(initialize_result(p)),
            _ => McpReply::Result(json!({"tools": []})),
        }
    }))
    .await;
    let server = MCPServerStreamableHttp::new(
        "retry-connect",
        MCPServerStreamableHttpParams {
            url,
            ..Default::default()
        },
    );
    assert!(server.connect().await.is_err());
    assert!(server.list_tools().await.is_err());
    server.connect().await.unwrap();
    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn timeout_error_names_the_sanitized_server() {
    // A request that outlives `client_session_timeout` reports the server by its sanitized name.
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => {
            std::thread::sleep(std::time::Duration::from_millis(400));
            McpReply::Result(json!({"tools": []}))
        }
    }))
    .await;
    let server =
        server_for(&url).with_client_session_timeout(Some(std::time::Duration::from_millis(50)));
    server.connect().await.unwrap();
    let error = server.list_tools().await.unwrap_err();
    assert_no_secrets(&error);
    assert!(error.to_string().contains("timed out"), "{error}");
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn zero_client_session_timeout_disables_the_timeout() {
    // `test_client_session_read_timeout_treats_zero_as_disabled`.
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => {
            std::thread::sleep(std::time::Duration::from_millis(150));
            McpReply::Result(json!({"tools": []}))
        }
    }))
    .await;
    for timeout in [None, Some(std::time::Duration::ZERO)] {
        let server = MCPServerStreamableHttp::new(
            "zero-timeout",
            MCPServerStreamableHttpParams {
                url: url.clone(),
                ..Default::default()
            },
        )
        .with_client_session_timeout(timeout);
        server.connect().await.unwrap();
        assert!(server.list_tools().await.unwrap().is_empty());
        server.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn connect_disconnect_lifecycle_works() {
    // test_connect_disconnect.py: connect, use, cleanup, and a second connect after cleanup.
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => McpReply::Result(json!({"tools": [
            {"name": "tool1", "inputSchema": {}}, {"name": "tool2", "inputSchema": {}}]})),
    }))
    .await;
    let server = MCPServerStreamableHttp::new(
        "lifecycle",
        MCPServerStreamableHttpParams {
            url,
            ..Default::default()
        },
    )
    .with_cache_tools_list(true);

    assert!(server.list_tools().await.is_err(), "not connected yet");
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap().len(), 2);
    server.cleanup().await.unwrap();
    assert!(server.list_tools().await.is_err(), "disconnected again");

    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap().len(), 2);
    server.cleanup().await.unwrap();
}
