//! Ports of python-only MCP cases re-audited in round 20:
//!
//! - `tests/mcp/test_list_page_limits.py::test_cancellation_with_page_budget_preserves_cancellation`
//!   (`list_tools`, `list_prompts`): a listing dropped mid-pagination caches nothing.
//! - `tests/mcp/test_client_session_retries.py::test_streamable_http_retries_closed_resource_on_isolated_session`:
//!   a connection the server closes during `tools/call` is retried on an isolated session.
//! - `tests/mcp/test_streamable_http_client_factory.py` custom-factory cases (proxy, all
//!   parameters): Python asserts the factory and the other parameters are forwarded; here the
//!   caller's `reqwest::Client` and the parameters are observed on the wire.
//!
//! Rust's transports own their session, so a local Streamable HTTP MCP server stands in for
//! Python's mocked `ClientSession`.
#![cfg(feature = "mcp")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::mcp::{MCPServer, MCPServerStreamableHttp, MCPServerStreamableHttpParams};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How the mock answers one JSON-RPC request.
enum Reply {
    Result(Value),
    /// Never answer (the request stays in flight).
    Hang,
    /// Close the connection without answering.
    Drop,
}

type Responder = Arc<dyn Fn(&str, &Value) -> Reply + Send + Sync>;

/// One recorded HTTP request: request line, lower-cased headers, JSON-RPC method.
#[derive(Debug, Clone)]
struct Seen {
    line: String,
    headers: HashMap<String, String>,
    method: String,
}

#[derive(Default)]
struct Log(Mutex<Vec<Seen>>);

impl Log {
    fn all(&self) -> Vec<Seen> {
        self.0.lock().unwrap().clone()
    }
    fn count(&self, method: &str) -> usize {
        self.all().iter().filter(|s| s.method == method).count()
    }
}

fn initialize_result(params: &Value) -> Value {
    json!({
        "protocolVersion": params["protocolVersion"],
        "capabilities": {"tools": {}, "prompts": {}},
        "serverInfo": {"name": "r20b-mock", "version": "1"},
    })
}

async fn spawn_mcp(responder: Responder) -> (String, Arc<Log>) {
    let log = Arc::new(Log::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_log = log.clone();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
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
                    let mut lines = head.lines();
                    let line = lines.next().unwrap_or_default().to_string();
                    let headers: HashMap<String, String> = lines
                        .filter_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            Some((k.trim().to_ascii_lowercase(), v.trim().to_string()))
                        })
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
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
                    let record = |method: &str| {
                        log.0.lock().unwrap().push(Seen {
                            line: line.clone(),
                            headers: headers.clone(),
                            method: method.to_string(),
                        })
                    };
                    if line.starts_with("GET ") {
                        record("GET");
                        if sock
                            .write_all(status_only("405 Method Not Allowed").as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    if line.starts_with("DELETE ") {
                        record("DELETE");
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
                    record(&method);
                    let Some(id) = req.get("id").cloned() else {
                        if sock
                            .write_all(status_only("202 Accepted").as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    };
                    let result = match responder(&method, &params) {
                        Reply::Result(result) => result,
                        Reply::Drop => return,
                        Reply::Hang => {
                            tokio::time::sleep(Duration::from_secs(3600)).await;
                            return;
                        }
                    };
                    let payload = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
                    let session = if method == "initialize" {
                        "mcp-session-id: sess-r20b\r\n"
                    } else {
                        ""
                    };
                    let raw = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{session}\
                         content-length: {}\r\n\r\n{payload}",
                        payload.len()
                    );
                    if sock.write_all(raw.as_bytes()).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    (format!("http://{addr}/mcp"), log)
}

fn server(url: &str) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "r20b",
        MCPServerStreamableHttpParams {
            url: url.to_string(),
            ..Default::default()
        },
    )
}

fn tool_result(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": false})
}

fn echo_tool() -> Value {
    json!({"tools": [{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}]})
}

// ---------------------------------------------------------------------------
// test_list_page_limits.py
// ---------------------------------------------------------------------------

/// A listing whose second page never arrives is dropped (Python cancels the task): two page
/// requests were made and no partial tool list is cached.
#[tokio::test]
async fn a_listing_dropped_mid_pagination_caches_nothing() {
    openai_agents_sdk::set_tracing_disabled(true);
    for method in ["tools/list", "prompts/list"] {
        let served = Arc::new(AtomicUsize::new(0));
        let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
            "initialize" => Reply::Result(initialize_result(params)),
            _ if m == method => {
                if served.fetch_add(1, Ordering::SeqCst) == 0 {
                    Reply::Result(if method == "tools/list" {
                        json!({"tools": [{"name": "tool_0", "inputSchema": {"type": "object"}}],
                               "nextCursor": "private-cursor-0"})
                    } else {
                        json!({"prompts": [{"name": "prompt_0"}], "nextCursor": "private-cursor-0"})
                    })
                } else {
                    Reply::Hang
                }
            }
            _ => Reply::Result(json!({})),
        }))
        .await;
        let server = server(&url)
            .with_max_list_pages(Some(2))
            .with_cache_tools_list(true);
        server.connect().await.unwrap();

        let dropped = if method == "tools/list" {
            tokio::time::timeout(Duration::from_millis(300), async {
                server.list_tools().await.map(|_| ())
            })
            .await
        } else {
            tokio::time::timeout(Duration::from_millis(300), async {
                server.list_prompts().await.map(|_| ())
            })
            .await
        };
        assert!(
            dropped.is_err(),
            "the listing was still in flight ({method})"
        );
        assert_eq!(log.count(method), 2, "{method}");
        assert!(server.cached_tools().await.is_none(), "{method}");
        let _ = tokio::time::timeout(Duration::from_secs(2), server.cleanup()).await;
    }
}

// ---------------------------------------------------------------------------
// test_client_session_retries.py
// ---------------------------------------------------------------------------

/// The shared connection is closed during `tools/call` (Python's `ClosedResourceError`); with
/// one retry allowed the call is replayed once on a freshly opened session and succeeds.
#[tokio::test]
async fn a_closed_connection_is_retried_on_an_isolated_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => Reply::Result(initialize_result(params)),
        "tools/list" => Reply::Result(echo_tool()),
        "tools/call" => {
            if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                Reply::Drop
            } else {
                Reply::Result(tool_result("isolated"))
            }
        }
        _ => Reply::Result(json!({})),
    }))
    .await;
    let server = server(&url).with_max_retry_attempts(1);
    server.connect().await.unwrap();

    let result = server.call_tool("echo", None).await.unwrap();
    assert_eq!(result["content"][0]["text"], "isolated");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // The retry opened its own session.
    assert_eq!(log.count("initialize"), 2);
    let _ = tokio::time::timeout(Duration::from_secs(2), server.cleanup()).await;
}

// ---------------------------------------------------------------------------
// test_streamable_http_client_factory.py
// ---------------------------------------------------------------------------

fn plain_responder() -> Responder {
    Arc::new(|m: &str, params: &Value| match m {
        "initialize" => Reply::Result(initialize_result(params)),
        "tools/list" => Reply::Result(echo_tool()),
        _ => Reply::Result(json!({})),
    })
}

/// `test_all_parameters_with_custom_factory` / `test_custom_httpx_client_factory`: the
/// caller's client carries every request next to the configured headers, and
/// `terminate_on_close = false` sends no `DELETE` on cleanup.
#[tokio::test]
async fn every_parameter_applies_with_a_caller_supplied_client() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, log) = spawn_mcp(plain_responder()).await;
    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert("x-test", "value".parse().unwrap());
    let client = reqwest::Client::builder()
        .default_headers(default_headers)
        .timeout(Duration::from_secs(90))
        .build()
        .unwrap();
    let server = MCPServerStreamableHttp::new(
        "r20b",
        MCPServerStreamableHttpParams {
            url,
            headers: HashMap::from([("Authorization".to_string(), "Bearer token".to_string())]),
            timeout: Duration::from_secs(45),
            sse_read_timeout: Duration::from_secs(600),
            terminate_on_close: false,
            http_client: Some(client),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap()[0].name, "echo");
    server.cleanup().await.unwrap();

    let posts: Vec<Seen> = log
        .all()
        .into_iter()
        .filter(|s| s.line.starts_with("POST "))
        .collect();
    assert!(!posts.is_empty());
    for post in &posts {
        assert_eq!(
            post.headers.get("x-test").map(String::as_str),
            Some("value")
        );
        assert_eq!(
            post.headers.get("authorization").map(String::as_str),
            Some("Bearer token")
        );
    }
    assert_eq!(log.count("DELETE"), 0, "terminate_on_close=false");
}

/// `test_custom_httpx_client_factory_with_proxy`: a client configured with a proxy sends the
/// MCP traffic through it (absolute-form request lines reach the proxy).
#[tokio::test]
async fn a_caller_supplied_client_keeps_its_proxy() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (proxy_url, log) = spawn_mcp(plain_responder()).await;
    let proxy_base = proxy_url.trim_end_matches("/mcp").to_string();
    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::http(&proxy_base).unwrap())
        .build()
        .unwrap();
    let server = MCPServerStreamableHttp::new(
        "r20b",
        MCPServerStreamableHttpParams {
            url: "http://mcp-upstream.invalid:8000/mcp".to_string(),
            http_client: Some(client),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap()[0].name, "echo");
    let _ = tokio::time::timeout(Duration::from_secs(2), server.cleanup()).await;

    let posts: Vec<Seen> = log
        .all()
        .into_iter()
        .filter(|s| s.line.starts_with("POST "))
        .collect();
    assert!(!posts.is_empty());
    for post in posts {
        assert!(
            post.line
                .starts_with("POST http://mcp-upstream.invalid:8000/mcp"),
            "{}",
            post.line
        );
    }
}

/// `test_default_httpx_client_factory` defaults: 5 s request timeout, 300 s SSE read timeout,
/// the session terminated on close.
#[test]
fn streamable_http_params_default_like_python() {
    let params = MCPServerStreamableHttpParams::default();
    assert_eq!(params.timeout, Duration::from_secs(5));
    assert_eq!(params.sse_read_timeout, Duration::from_secs(300));
    assert!(params.terminate_on_close);
    assert!(params.http_client.is_none());
}
