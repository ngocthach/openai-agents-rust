//! Ports of `tests/mcp/test_mcp_auth_params.py` (caller-supplied HTTP client, with and without
//! `auth`) and `tests/mcp/test_mcp_version_compat.py`.
//!
//! Python asserts the keyword arguments given to a patched `sse_client` and
//! `streamablehttp_client`. The behaviour behind them is that the caller's HTTP client carries
//! every request, so these tests read the headers a local server receives on the SSE stream, on
//! the POSTs that follow, and on Streamable HTTP requests. The legacy stdio server answers
//! `initialize` with a fixed, older protocol version instead of echoing the requested one.
#![cfg(feature = "mcp")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::mcp::{
    MCPServer, MCPServerSse, MCPServerSseParams, MCPServerStdio, MCPServerStdioParams,
    MCPServerStreamableHttp, MCPServerStreamableHttpParams,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

/// The headers of every request the mock received, as `(method, path, lowercased headers)`.
type Seen = Arc<Mutex<Vec<(String, String, HashMap<String, String>)>>>;

/// A local MCP server speaking both the legacy SSE transport (`/sse` and `/messages`) and
/// Streamable HTTP (`/mcp`), recording the headers of each request.
async fn spawn_mock() -> (String, Seen) {
    let seen: Seen = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let recorded = seen.clone();
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let (seen, tx, rx) = (recorded.clone(), tx.clone(), rx.clone());
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
                    let request_line = lines.next().unwrap_or_default().to_string();
                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or_default().to_string();
                    let path = parts.next().unwrap_or_default().to_string();
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
                    seen.lock()
                        .unwrap()
                        .push((method.clone(), path.clone(), headers));
                    while buf.len() < header_end + len {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    let body = buf[header_end..header_end + len].to_vec();
                    buf.drain(..header_end + len);
                    let status =
                        |code: &str| format!("HTTP/1.1 {code}\r\ncontent-length: 0\r\n\r\n");

                    if method == "GET" && path.starts_with("/sse") {
                        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n";
                        if sock.write_all(head.as_bytes()).await.is_err()
                            || sock
                                .write_all(b"event: endpoint\ndata: /messages?session_id=abc\n\n")
                                .await
                                .is_err()
                        {
                            return;
                        }
                        let Some(mut rx) = rx.lock().await.take() else {
                            return;
                        };
                        while let Some(message) = rx.recv().await {
                            let frame = format!("event: message\ndata: {message}\n\n");
                            if sock.write_all(frame.as_bytes()).await.is_err() {
                                break;
                            }
                        }
                        return;
                    }
                    if method == "GET" {
                        let _ = sock
                            .write_all(status("405 Method Not Allowed").as_bytes())
                            .await;
                        continue;
                    }
                    if method == "DELETE" {
                        let _ = sock.write_all(status("200 OK").as_bytes()).await;
                        continue;
                    }
                    let Ok(request) = serde_json::from_slice::<Value>(&body) else {
                        let _ = sock.write_all(status("400 Bad Request").as_bytes()).await;
                        return;
                    };
                    let Some(id) = request.get("id").cloned() else {
                        let _ = sock.write_all(status("202 Accepted").as_bytes()).await;
                        continue;
                    };
                    let result = match request["method"].as_str().unwrap_or_default() {
                        "initialize" => json!({
                            "protocolVersion": request["params"]["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "header-mock", "version": "1"}}),
                        "tools/list" => json!({"tools": [{"name": "echo", "description": "Echo",
                            "inputSchema": {"type": "object", "properties": {}}}]}),
                        _ => json!({}),
                    };
                    let payload = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
                    if path.starts_with("/messages") {
                        // The SSE transport answers on the stream, not in the POST response.
                        let _ = sock.write_all(status("202 Accepted").as_bytes()).await;
                        let _ = tx.send(payload);
                    } else {
                        let raw = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                             content-length: {}\r\n\r\n{payload}",
                            payload.len()
                        );
                        if sock.write_all(raw.as_bytes()).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    (format!("http://{addr}"), seen)
}

/// A client whose default header every request must carry.
fn custom_client() -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-custom", "1".parse().unwrap());
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap()
}

/// The recorded requests whose path starts with `prefix`, as `(method, headers)`.
fn requests_to(seen: &Seen, prefix: &str) -> Vec<(String, HashMap<String, String>)> {
    seen.lock()
        .unwrap()
        .iter()
        .filter(|(_, path, _)| path.starts_with(prefix))
        .map(|(method, _, headers)| (method.clone(), headers.clone()))
        .collect()
}

/// `test_sse_with_httpx_client_factory`: the caller's client carries the SSE stream and the
/// POSTs.
#[tokio::test]
async fn sse_uses_the_caller_supplied_http_client() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, seen) = spawn_mock().await;
    let server = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: format!("{base}/sse"),
            http_client: Some(custom_client()),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap()[0].name, "echo");
    server.cleanup().await.unwrap();

    let stream = requests_to(&seen, "/sse");
    assert_eq!(stream.len(), 1);
    assert_eq!(stream[0].1.get("x-custom").map(String::as_str), Some("1"));
    let posts = requests_to(&seen, "/messages");
    assert!(!posts.is_empty());
    for (method, headers) in posts {
        assert_eq!(method, "POST");
        assert_eq!(headers.get("x-custom").map(String::as_str), Some("1"));
    }
}

/// `test_sse_with_auth_and_factory`: `headers`, `auth` and the caller's client all take part.
#[tokio::test]
async fn sse_combines_headers_auth_and_the_caller_supplied_http_client() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, seen) = spawn_mock().await;
    let server = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: format!("{base}/sse"),
            headers: HashMap::from([("X-Token".to_string(), "abc".to_string())]),
            auth: Some(Arc::new(|| Some("secret".to_string()))),
            http_client: Some(custom_client()),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();
    server.cleanup().await.unwrap();

    let stream = requests_to(&seen, "/sse");
    let headers = &stream[0].1;
    assert_eq!(headers.get("x-custom").map(String::as_str), Some("1"));
    assert_eq!(headers.get("x-token").map(String::as_str), Some("abc"));
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer secret")
    );
}

/// `test_streamable_http_with_auth_and_factory`.
#[tokio::test]
async fn streamable_http_combines_auth_and_the_caller_supplied_http_client() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, seen) = spawn_mock().await;
    let server = MCPServerStreamableHttp::new(
        "http",
        MCPServerStreamableHttpParams {
            url: format!("{base}/mcp"),
            auth: Some(Arc::new(|| Some("secret".to_string()))),
            http_client: Some(custom_client()),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    server.list_tools().await.unwrap();
    server.cleanup().await.unwrap();

    let posts: Vec<_> = requests_to(&seen, "/mcp")
        .into_iter()
        .filter(|(method, _)| method == "POST")
        .collect();
    assert!(!posts.is_empty());
    for (_, headers) in posts {
        assert_eq!(headers.get("x-custom").map(String::as_str), Some("1"));
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );
    }
}

/// `test_stdio_connects_to_legacy_server`: a server that answers `initialize` with an older
/// protocol version than the client asked for still lists and calls tools.
#[tokio::test]
async fn stdio_connects_to_a_server_that_reports_an_older_protocol_version() {
    openai_agents_sdk::set_tracing_disabled(true);
    let Some(python) = ["python3", "python"].into_iter().find(|p| {
        std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok()
    }) else {
        eprintln!("python not available; skipping");
        return;
    };
    let server = MCPServerStdio::new(
        "legacy-test-server",
        MCPServerStdioParams {
            command: python.to_string(),
            args: vec![
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/legacy_mcp_server.py"
                )
                .into(),
            ],
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    let tools = server.list_tools().await.unwrap();
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["legacy_tool"]
    );
    let result = server.call_tool("legacy_tool", None).await.unwrap();
    assert_eq!(result["content"][0]["type"], "text");
    assert_eq!(result["content"][0]["text"], "legacy-result");
    assert_ne!(result["isError"], json!(true));
    server.cleanup().await.unwrap();
}
