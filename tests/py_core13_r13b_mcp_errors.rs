//! Python `tests/mcp/test_server_errors.py` cases whose observable half is a message or a
//! lifecycle effect rather than an exception-graph shape: how each class of transport failure
//! is worded, and that cleanup swallows a failing session teardown and still resets the server.
#![cfg(feature = "mcp")]
#![allow(dead_code)]

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
    pub listener: Mutex<Option<tokio::task::JoinHandle<()>>>,
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
    let accept_loop = tokio::spawn(async move {
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
    *log.listener.lock().unwrap() = Some(accept_loop);
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

fn server_with(url: &str, timeout: std::time::Duration) -> MCPServerStreamableHttp {
    let full = credentialed(url);
    MCPServerStreamableHttp::new(
        format!("streamable_http: {full}"),
        MCPServerStreamableHttpParams {
            url: full,
            sse_read_timeout: timeout,
            ..Default::default()
        },
    )
    .with_max_retry_attempts(0)
}

fn dropping() -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" | "tools/call" | "prompts/list" | "resources/read" => McpReply::Drop,
        _ => McpReply::Result(json!({})),
    })
}

fn hanging() -> Responder {
    Arc::new(move |m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/list" | "tools/call" | "prompts/list" | "resources/read" => {
            std::thread::sleep(std::time::Duration::from_millis(1500));
            McpReply::Result(json!({}))
        }
        _ => McpReply::Result(json!({})),
    })
}

async fn failures(server: &MCPServerStreamableHttp) -> Vec<AgentsError> {
    vec![
        server.list_tools().await.unwrap_err(),
        server.call_tool("safe_tool", None).await.unwrap_err(),
        server.list_prompts().await.unwrap_err(),
        server.read_resource("file:///safe.txt").await.unwrap_err(),
    ]
}

fn expected(name: &str, reason: &str) -> Vec<String> {
    vec![
        format!("Failed to list tools from MCP server '{name}': {reason}"),
        format!("Failed to call tool 'safe_tool' on MCP server '{name}': {reason}"),
        format!("Failed to list prompts on MCP server '{name}': {reason}"),
        format!("Failed to read resource on MCP server '{name}': {reason}"),
    ]
}

#[tokio::test]
async fn a_connection_dropped_mid_request_is_a_bare_request_failure() {
    // `httpx.ReadError` / `RemoteProtocolError` map to "Request failed." for every operation.
    let (url, _log) = spawn_mcp(dropping()).await;
    let server = server_with(&url, std::time::Duration::from_secs(5));
    server.connect().await.unwrap();
    let name = safe_name(&url);
    for (error, expected) in failures(&server)
        .await
        .iter()
        .zip(expected(&name, "Request failed."))
    {
        assert_message(error, &expected);
    }
    server.cleanup().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_response_that_outlives_the_read_timeout_is_a_connection_timeout() {
    // `httpx.ReadTimeout` maps to "Connection timeout." for every operation.
    let (url, _log) = spawn_mcp(hanging()).await;
    let server = server_with(&url, std::time::Duration::from_millis(300));
    server.connect().await.unwrap();
    let name = safe_name(&url);
    for (error, expected) in failures(&server)
        .await
        .iter()
        .zip(expected(&name, "Connection timeout."))
    {
        assert_message(error, &expected);
    }
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_connection_dropped_during_the_handshake_is_a_bare_request_failure() {
    let (url, _log) = spawn_mcp(Arc::new(|_m: &str, _p: &Value| McpReply::Drop)).await;
    let error = server_with(&url, std::time::Duration::from_secs(5))
        .connect()
        .await
        .unwrap_err();
    assert_message(
        &error,
        &format!(
            "Failed to connect to MCP server '{}': Request failed.",
            safe_name(&url)
        ),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handshake_that_outlives_the_read_timeout_is_a_connection_timeout() {
    let (url, _log) = spawn_mcp(Arc::new(|_m: &str, _p: &Value| {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        McpReply::Status(500)
    }))
    .await;
    let error = server_with(&url, std::time::Duration::from_millis(300))
        .connect()
        .await
        .unwrap_err();
    assert_message(
        &error,
        &format!(
            "Failed to connect to MCP server '{}': Connection timeout.",
            safe_name(&url)
        ),
    );
}

#[tokio::test]
async fn a_sub_microsecond_client_session_timeout_still_applies() {
    // Python rejects a timeout below one microsecond because it would round to zero and
    // silently disable the limit; a `Duration` keeps its nanoseconds, so the limit applies.
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => {
            std::thread::sleep(std::time::Duration::from_millis(100));
            McpReply::Result(json!({"tools": []}))
        }
    }))
    .await;
    let server =
        server_for(&url).with_client_session_timeout(Some(std::time::Duration::from_nanos(500)));
    server.connect().await.unwrap();
    let error = server.list_tools().await.unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error}");
    assert_no_secrets(&error);
}

#[tokio::test]
async fn cleanup_ends_the_session_once_and_leaves_the_server_reconnectable() {
    // Cleanup ends the session with one `DELETE`, leaves the server unconnected (a request now
    // reports it, without secrets) and the server can be connected again.
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => McpReply::Result(json!({"tools": []})),
    }))
    .await;
    let server = server_for(&url);
    server.connect().await.unwrap();
    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
    assert_eq!(log.deletes.load(Ordering::SeqCst), 1);

    let error = server.list_tools().await.unwrap_err();
    assert!(error.to_string().contains("not initialized"), "{error}");
    assert_no_secrets(&error);
    server.connect().await.unwrap();
    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_connection_refused_mid_session_is_a_lost_connection() {
    // `httpx.ConnectError` on a request maps to "Connection lost. The server may have
    // disconnected." for every operation.
    let (url, log) = spawn_mcp(erroring(500)).await;
    let full = credentialed(&url);
    let server = MCPServerStreamableHttp::new(
        format!("streamable_http: {full}"),
        MCPServerStreamableHttpParams {
            url: full,
            // A new connection per request, so the stopped listener is noticed.
            http_client: Some(
                reqwest::Client::builder()
                    .pool_max_idle_per_host(0)
                    .build()
                    .unwrap(),
            ),
            ..Default::default()
        },
    )
    .with_max_retry_attempts(0);
    server.connect().await.unwrap();
    log.listener.lock().unwrap().take().unwrap().abort();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let name = safe_name(&url);
    for (error, expected) in failures(&server).await.iter().zip(expected(
        &name,
        "Connection lost. The server may have disconnected.",
    )) {
        assert_message(error, &expected);
    }
}
