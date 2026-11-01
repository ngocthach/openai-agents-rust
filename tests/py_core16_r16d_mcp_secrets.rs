//! Python `tests/mcp/test_server_errors.py` cases that have a user-visible half beyond the
//! message class: a failing session teardown is swallowed and resets the server, and neither the
//! session id, the request body nor the response headers/body of a failed request reach the
//! error text.
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
    /// Never answer (the connection stays open).
    Hang,
    /// A raw HTTP answer: status, extra header lines, body.
    Http(u16, String, String),
}

type Responder = Arc<dyn Fn(&str, &Value) -> McpReply + Send + Sync>;

/// What a mock recorded: every `(method, params)` pair it was asked for.
#[derive(Default, Debug)]
pub struct McpLog {
    pub requests: Mutex<Vec<(String, Value)>>,
    pub authorizations: Mutex<Vec<String>>,
    pub deletes: AtomicUsize,
    pub delete_status: std::sync::atomic::AtomicU16,
    pub session_headers: Mutex<Vec<String>>,
    pub bodies: Mutex<Vec<String>>,
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
                    if let Some(id) = head.lines().find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("mcp-session-id")
                            .then(|| v.trim().to_string())
                    }) {
                        log.session_headers.lock().unwrap().push(id);
                    }
                    log.bodies
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&body).to_string());
                    if head.starts_with("DELETE ") {
                        log.deletes.fetch_add(1, Ordering::SeqCst);
                        let status = log.delete_status.load(Ordering::SeqCst);
                        let line = if status == 0 { 200 } else { status };
                        if sock
                            .write_all(status_only(&format!("{line} X")).as_bytes())
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
                            McpReply::Hang => {
                                tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                                return;
                            }
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
                        McpReply::Hang => {
                            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                            return;
                        }
                        McpReply::Http(code, headers, body) => format!(
                            "HTTP/1.1 {code} Error\r\n{headers}content-length: {}\r\n\r\n{body}",
                            body.len()
                        ),
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

fn rendered(error: &AgentsError) -> String {
    format!("{error} {error:?}")
}

const OPS: [&str; 7] = [
    "tools/list",
    "tools/call",
    "prompts/list",
    "prompts/get",
    "resources/list",
    "resources/templates/list",
    "resources/read",
];

async fn every_operation_fails(server: &MCPServerStreamableHttp) -> Vec<AgentsError> {
    let args: std::collections::BTreeMap<String, String> =
        [("note".to_string(), "SECRET_REQUEST_BODY".to_string())].into();
    vec![
        server.list_tools().await.unwrap_err(),
        server
            .call_tool(
                "t",
                json!({"note": "SECRET_REQUEST_BODY"}).as_object().cloned(),
            )
            .await
            .unwrap_err(),
        server.list_prompts().await.unwrap_err(),
        server.get_prompt("p", Some(args)).await.unwrap_err(),
        server.list_resources(None).await.unwrap_err(),
        server.list_resource_templates(None).await.unwrap_err(),
        server.read_resource("file:///safe.txt").await.unwrap_err(),
    ]
}

#[tokio::test]
async fn cleanup_swallows_a_failing_session_delete_and_resets_the_server() {
    // A teardown that fails (`DELETE` answered 502) is not an error of `cleanup()`; the server
    // is left unconnected and can connect again.
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        _ => McpReply::Result(json!({"tools": []})),
    }))
    .await;
    log.delete_status.store(502, Ordering::SeqCst);
    let server = server_for(&url);
    server.connect().await.unwrap();
    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
    assert_eq!(log.deletes.load(Ordering::SeqCst), 1);

    let error = server.list_tools().await.unwrap_err();
    assert!(error.to_string().contains("not initialized"), "{error}");
    for secret in SECRETS {
        assert!(!rendered(&error).contains(secret));
    }
    log.delete_status.store(0, Ordering::SeqCst);
    server.connect().await.unwrap();
    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn errors_omit_the_session_id_and_the_request_body() {
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| {
        if m == "initialize" {
            McpReply::Result(initialize_result(p))
        } else if OPS.contains(&m) {
            McpReply::Status(503)
        } else {
            McpReply::Result(json!({}))
        }
    }))
    .await;
    let server = server_for(&url);
    server.connect().await.unwrap();
    let errors = every_operation_fails(&server).await;
    // The mock really did see the session header and the body the error must not repeat.
    assert!(
        log.session_headers
            .lock()
            .unwrap()
            .iter()
            .any(|s| s == "sess-pagination")
    );
    assert!(
        log.bodies
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("SECRET_REQUEST_BODY"))
    );
    for error in &errors {
        let text = rendered(error);
        assert!(text.contains("HTTP error 503"), "{text}");
        for secret in SECRETS
            .iter()
            .chain(&["sess-pagination", "SECRET_REQUEST_BODY"])
        {
            assert!(!text.contains(secret), "{secret} leaked: {text}");
        }
    }
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn http_status_errors_omit_response_headers_and_body() {
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, p: &Value| {
        if m == "initialize" {
            McpReply::Result(initialize_result(p))
        } else if OPS.contains(&m) {
            McpReply::Http(
                503,
                "set-cookie: SECRET_RESPONSE_COOKIE\r\nlocation: https://u:SECRET_LOCATION@example.com/\r\n"
                    .to_string(),
                "SECRET_RESPONSE_BODY".to_string(),
            )
        } else {
            McpReply::Result(json!({}))
        }
    }))
    .await;
    let server = server_for(&url);
    server.connect().await.unwrap();
    for error in every_operation_fails(&server).await {
        let text = rendered(&error);
        assert!(text.contains("HTTP error 503"), "{text}");
        for secret in SECRETS.iter().chain(&[
            "SECRET_RESPONSE_COOKIE",
            "SECRET_RESPONSE_BODY",
            "SECRET_LOCATION",
        ]) {
            assert!(!text.contains(secret), "{secret} leaked: {text}");
        }
    }
    server.cleanup().await.unwrap();
}

/// `test_v2_streamable_http_preserves_outer_cancellation`: dropping a call that hangs is not
/// retried (the retry budget is for failures, not for the caller giving up) and leaves the
/// connected server usable.
#[tokio::test]
async fn abandoning_a_hanging_call_is_not_retried() {
    let (url, log) = spawn_mcp(Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "tools/call" => McpReply::Hang,
        _ => McpReply::Result(json!({"tools": []})),
    }))
    .await;
    let server = server_for(&url).with_max_retry_attempts(1);
    server.connect().await.unwrap();

    let call = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        server.call_tool("test", None),
    )
    .await;
    assert!(call.is_err(), "the call should still be hanging");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(log.calls_of("tools/call").len(), 1);

    assert!(server.list_tools().await.unwrap().is_empty());
    server.cleanup().await.unwrap();
}
