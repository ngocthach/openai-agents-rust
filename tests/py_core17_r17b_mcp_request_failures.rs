//! Port of `test_prompt_and_resource_request_errors_hide_url_credentials` from
//! `tests/mcp/test_server_errors.py`: when the transport fails mid-request, every prompt and
//! resource operation reports `Failed to <operation> on MCP server '<name>': Request failed.`
//! with the credential-free server name, no secret from the URL reaches the error text, and no
//! log record is written for the failure, whether tool data is redacted or not.
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::mcp::{
    MCPServer, MCPServerStreamableHttp, MCPServerStreamableHttpParams, get_mcp_server_log_name,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

fn safe_name(url: &str) -> String {
    get_mcp_server_log_name(&format!("streamable_http: {}", credentialed(url)))
}

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        // Only the SDK's own records (third-party crates log their own diagnostics).
        if record.target() == "openai.agents" {
            RECORDS
                .lock()
                .unwrap()
                .push((record.level(), format!("{}", record.args())));
        }
    }
    fn flush(&self) {}
}

fn dropping_every_request() -> Responder {
    Arc::new(|m: &str, p: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(p)),
        "prompts/list"
        | "prompts/get"
        | "resources/list"
        | "resources/templates/list"
        | "resources/read" => McpReply::Drop,
        _ => McpReply::Result(json!({})),
    })
}

#[tokio::test]
async fn a_dropped_prompt_or_resource_request_is_a_bare_failure_without_log_records() {
    log::set_logger(&Capture).expect("this binary installs the only logger");
    log::set_max_level(log::LevelFilter::Trace);

    for redacted in [true, false] {
        set_dont_log_tool_data(redacted);
        let (url, _log) = spawn_mcp(dropping_every_request()).await;
        let server = server_for(&url);
        server.connect().await.unwrap();
        let name = safe_name(&url);
        assert!(!name.contains("s3cr3t_pw"));
        RECORDS.lock().unwrap().clear();

        let cases: Vec<(Result<(), AgentsError>, &str)> = vec![
            (server.list_prompts().await.map(drop), "list prompts"),
            (
                server.get_prompt("safe_prompt", None).await.map(drop),
                "get prompt",
            ),
            (
                server.list_resources(None).await.map(drop),
                "list resources",
            ),
            (
                server.list_resource_templates(None).await.map(drop),
                "list resource templates",
            ),
            (
                server.read_resource("file:///safe.txt").await.map(drop),
                "read resource",
            ),
        ];
        for (result, operation) in cases {
            let error = result.unwrap_err();
            assert_no_secrets(&error);
            assert_eq!(
                error.to_string(),
                format!("Failed to {operation} on MCP server '{name}': Request failed."),
                "redacted={redacted}"
            );
        }
        let records = RECORDS.lock().unwrap().clone();
        assert!(records.is_empty(), "redacted={redacted}: {records:?}");
        server.cleanup().await.unwrap();
    }
    reset_dont_log_tool_data();
}
