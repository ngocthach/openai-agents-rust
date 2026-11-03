//! Ports of the sibling-request cases of `tests/mcp/test_client_session_retries.py`: a request
//! that fails does not take a slow request or a prompt request on the same server down with it.
//! Python serializes requests to avoid cancelling siblings inside one client task group; Rust
//! requests run concurrently on separate connections, so the observable guarantee is that each
//! caller gets its own outcome.
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
                        McpReply::Status(code) => status_only(&format!("{code} Error")),
                        McpReply::Result(result) => {
                            let payload =
                                json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
                            let session = if method == "initialize" {
                                "mcp-session-id: sess-siblings\r\n"
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

fn server(url: &str) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "siblings",
        MCPServerStreamableHttpParams {
            url: url.to_string(),
            ..Default::default()
        },
    )
    .with_max_retry_attempts(0)
}

fn responder(in_flight: Arc<AtomicUsize>, peak: Arc<AtomicUsize>) -> Responder {
    Arc::new(move |method: &str, params: &Value| match method {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/call" => match params["name"].as_str() {
            Some("slow") => {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(300));
                in_flight.fetch_sub(1, Ordering::SeqCst);
                McpReply::Result(json!({"content": [{"type": "text", "text": "slow done"}]}))
            }
            _ => McpReply::Status(500),
        },
        "prompts/list" => McpReply::Result(json!({"prompts": [{"name": "prompt"}]})),
        "prompts/get" => McpReply::Result(json!({
            "messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        })),
        _ => McpReply::Result(json!({})),
    })
}

/// Python: `test_serialized_session_requests_prevent_sibling_cancellation`: a failing tool call
/// leaves the slow tool call running next to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_tool_call_does_not_cancel_a_slow_sibling() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, _log) = spawn_mcp(responder(Arc::default(), Arc::default())).await;
    let server = server(&url);
    server.connect().await.unwrap();

    let (slow, failing) = tokio::join!(
        server.call_tool("slow", None),
        server.call_tool("fail", None)
    );

    let slow = slow.expect("the slow call completes");
    assert!(slow.to_string().contains("slow done"), "{slow}");
    assert!(failing.is_err());
    server.cleanup().await.unwrap();
}

/// Python: `test_serialized_prompt_requests_prevent_tool_cancellation` and
/// `test_streamable_http_serializes_call_tool_with_prompt_requests` (`list_prompts` and
/// `get_prompt`): a prompt request issued while a tool call is slow returns its own result and
/// leaves the tool call alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_requests_next_to_a_slow_tool_call_return_their_own_results() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, _log) = spawn_mcp(responder(Arc::default(), Arc::default())).await;
    let server = server(&url);
    server.connect().await.unwrap();

    let (slow, listed, fetched) = tokio::join!(
        server.call_tool("slow", None),
        server.list_prompts(),
        server.get_prompt("prompt", None)
    );

    assert!(slow.unwrap().to_string().contains("slow done"));
    assert_eq!(listed.unwrap()[0]["name"], "prompt");
    assert!(fetched.unwrap().to_string().contains("hi"));
    server.cleanup().await.unwrap();
}
