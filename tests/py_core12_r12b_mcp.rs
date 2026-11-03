//! Ports of `tests/mcp/test_client_session_retries.py` cases that need a server to fail in a
//! particular way: the retry budget is shared across list pages, a dropped connection earns an
//! isolated-session retry, and the isolated retry spends the same budget the shared attempts do.
//!
//! Python injects fake `ClientSession` objects; these tests drive a local Streamable HTTP MCP
//! server instead.
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::mcp::{
    MCPServer, MCPServerConfig, MCPServerStreamableHttp, MCPServerStreamableHttpParams,
};
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
    /// Close the connection without answering.
    Drop,
}

type Responder = Arc<dyn Fn(&str, &Value) -> McpReply + Send + Sync>;

/// What a mock recorded: every `(method, params)` pair it was asked for.
#[derive(Default, Debug)]
pub struct McpLog {
    pub requests: Mutex<Vec<(String, Value)>>,
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
                        McpReply::Drop => return,
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

fn server(url: &str, retries: i32) -> MCPServerStreamableHttp {
    let config = MCPServerConfig {
        max_retry_attempts: retries,
        retry_backoff_seconds_base: 0.0,
        client_session_timeout: None,
        ..Default::default()
    };
    MCPServerStreamableHttp::new(
        "retries",
        MCPServerStreamableHttpParams {
            url: url.to_string(),
            ..Default::default()
        },
    )
    .with_config(config)
}

fn tool_result(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": false})
}

/// Python: `test_list_tools_shares_retry_budget_across_pages`. The first page fails and spends
/// the only retry; the second page then fails with nothing left, so the call fails with a
/// request error and the third request asks for the second page.
#[tokio::test]
async fn the_retry_budget_is_shared_across_list_pages() {
    openai_agents_sdk::set_tracing_disabled(true);
    let served = Arc::new(AtomicUsize::new(0));
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/list" => match served.fetch_add(1, Ordering::SeqCst) {
            0 => McpReply::Status(503),
            1 => McpReply::Result(json!({
                "tools": [{"name": "first_page_tool", "inputSchema": {}}],
                "nextCursor": "second-page",
            })),
            _ => McpReply::Status(503),
        },
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = server(&url, 1);
    server.connect().await.unwrap();

    let error = server.list_tools().await.unwrap_err().to_string();
    assert!(error.contains("Failed to list tools"), "{error}");
    let cursors: Vec<Value> = log
        .calls_of("tools/list")
        .iter()
        .map(|params| params.get("cursor").cloned().unwrap_or(Value::Null))
        .collect();
    assert_eq!(cursors, [Value::Null, Value::Null, json!("second-page")]);
    server.cleanup().await.unwrap();
}

/// Python: `test_streamable_http_retries_cancelled_request_on_isolated_session` (and the
/// closed-resource twin): a request whose connection dies is replayed once on a fresh session,
/// which opens its own connection and answers.
#[tokio::test]
async fn a_dropped_connection_is_retried_on_an_isolated_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let calls = Arc::new(AtomicUsize::new(0));
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/call" => {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                McpReply::Drop
            } else {
                McpReply::Result(tool_result("ok"))
            }
        }
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = server(&url, 1);
    server.connect().await.unwrap();

    let result = server.call_tool("tool", None).await.unwrap();
    assert_eq!(result["content"][0]["text"], "ok");
    assert_eq!(log.calls_of("tools/call").len(), 2);
    assert_eq!(
        log.calls_of("initialize").len(),
        2,
        "the retry opened a session"
    );
    server.cleanup().await.unwrap();
}

/// Python: `test_streamable_http_backoff_matches_generic_schedule_on_isolated_retry`. With a
/// budget of six and a server that always fails, each shared attempt is followed by an isolated
/// retry that spends budget too: seven calls on four sessions, with three backoffs between the
/// attempts (the delays themselves are pinned in the crate's `mcp::backoff_tests`).
#[tokio::test]
async fn isolated_retries_spend_the_retry_budget_on_the_generic_schedule() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/call" => McpReply::Status(503),
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = server(&url, 6);
    server.connect().await.unwrap();

    let error = server
        .call_tool("tool", None)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("503"), "{error}");
    assert_eq!(log.calls_of("tools/call").len(), 7);
    assert_eq!(log.calls_of("initialize").len(), 4);
    server.cleanup().await.unwrap();
}
