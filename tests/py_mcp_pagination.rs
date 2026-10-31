//! Parity port of `tests/mcp/test_list_page_limits.py` and
//! `tests/mcp/test_mcp_pagination_integration.py`.
//!
//! Python injects a mock `ClientSession` into the server; Rust's transports own their session,
//! so these tests drive a local Streamable HTTP MCP server that serves scripted pages.
#![cfg(feature = "mcp")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::mcp::{MCPServer, MCPServerStreamableHttp, MCPServerStreamableHttpParams};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, MCPConfig, Runner};
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

fn server(url: &str) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "pages",
        MCPServerStreamableHttpParams {
            url: url.to_string(),
            ..Default::default()
        },
    )
}

/// Python's `list_page` helper: one entry per page, with a private cursor unless it is last.
fn tools_page(index: usize, last: bool, empty: bool) -> Value {
    let mut page = json!({
        "tools": if empty {
            json!([])
        } else {
            json!([{"name": format!("tool_{index}"), "inputSchema": {"type": "object", "properties": {}}}])
        }
    });
    if !last {
        page["nextCursor"] = json!(format!("private-cursor-{index}"));
    }
    page
}

fn prompts_page(index: usize, last: bool, empty: bool) -> Value {
    let mut page = json!({
        "prompts": if empty { json!([]) } else { json!([{"name": format!("prompt_{index}")}]) }
    });
    if !last {
        page["nextCursor"] = json!(format!("private-cursor-{index}"));
    }
    page
}

/// A responder that serves successive pages of `method`, never ending.
fn endless_pages(method: &'static str, empty: bool) -> Responder {
    let served = Arc::new(AtomicUsize::new(0));
    Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        _ if m == method => {
            let index = served.fetch_add(1, Ordering::SeqCst);
            McpReply::Result(if method == "tools/list" {
                tools_page(index, false, empty)
            } else {
                prompts_page(index, false, empty)
            })
        }
        _ => McpReply::Result(json!({})),
    })
}

// ---------------------------------------------------------------------------
// test_list_page_limits.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fresh_cursors_stop_at_page_limit() {
    openai_agents_sdk::set_tracing_disabled(true);
    for method in ["tools/list", "prompts/list"] {
        for empty in [false, true] {
            let (url, log) = spawn_mcp(endless_pages(
                if method == "tools/list" {
                    "tools/list"
                } else {
                    "prompts/list"
                },
                empty,
            ))
            .await;
            let server = server(&url)
                .with_max_list_pages(Some(2))
                .with_max_retry_attempts(0)
                .with_cache_tools_list(true);
            server.connect().await.unwrap();

            let error = if method == "tools/list" {
                server.list_tools().await.unwrap_err()
            } else {
                server.list_prompts().await.unwrap_err()
            };
            assert!(
                error.to_string().contains("exceeded max_list_pages"),
                "{error}"
            );
            assert_eq!(log.calls_of(method).len(), 2, "method={method}");
            assert!(server.cached_tools().await.is_none());
            server.cleanup().await.unwrap();
        }
    }
}

#[tokio::test]
async fn complete_listing_at_page_boundary() {
    openai_agents_sdk::set_tracing_disabled(true);
    for method in ["tools/list", "prompts/list"] {
        for limit in [Some(1usize), Some(2), None] {
            let page_count = limit.unwrap_or(4);
            let served = Arc::new(AtomicUsize::new(0));
            let wanted = method.to_string();
            let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| {
                if m == "initialize" {
                    return McpReply::Result(initialize_result(params));
                }
                if m != wanted {
                    return McpReply::Result(json!({}));
                }
                let index = served.fetch_add(1, Ordering::SeqCst);
                let last = index == page_count - 1;
                McpReply::Result(if wanted == "tools/list" {
                    tools_page(index, last, false)
                } else {
                    prompts_page(index, last, false)
                })
            }))
            .await;
            let server = server(&url)
                .with_max_list_pages(limit)
                .with_cache_tools_list(true);
            server.connect().await.unwrap();

            if method == "tools/list" {
                let tools = server.list_tools().await.unwrap();
                let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                let expected: Vec<String> = (0..page_count).map(|i| format!("tool_{i}")).collect();
                assert_eq!(names, expected);
                // The second call is served from the cache.
                assert_eq!(server.list_tools().await.unwrap(), tools);
                assert_eq!(log.calls_of(method).len(), page_count);
            } else {
                let prompts = server.list_prompts().await.unwrap();
                let names: Vec<&str> = prompts
                    .iter()
                    .map(|p| p["name"].as_str().unwrap())
                    .collect();
                let expected: Vec<String> =
                    (0..page_count).map(|i| format!("prompt_{i}")).collect();
                assert_eq!(names, expected);
                assert_eq!(log.calls_of(method).len(), page_count);
            }
            server.cleanup().await.unwrap();
        }
    }
}

#[tokio::test]
async fn failed_refresh_does_not_replace_cache_and_next_call_starts_fresh() {
    openai_agents_sdk::set_tracing_disabled(true);
    let served = Arc::new(AtomicUsize::new(0));
    let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/list" => {
            let index = served.fetch_add(1, Ordering::SeqCst);
            // page 0 completes, page 1 asks for a second page (over the budget), page 2
            // completes again.
            McpReply::Result(match index {
                0 => tools_page(0, true, false),
                1 => tools_page(1, false, false),
                _ => tools_page(2, true, false),
            })
        }
        _ => McpReply::Result(json!({})),
    }))
    .await;

    let server = server(&url)
        .with_max_list_pages(Some(1))
        .with_cache_tools_list(true);
    server.connect().await.unwrap();

    let original = server.list_tools().await.unwrap();
    server.invalidate_tools_cache().await;

    let error = server.list_tools().await.unwrap_err();
    assert!(error.to_string().contains("exceeded max_list_pages"));
    // A failed refresh leaves no cache behind.
    assert!(server.cached_tools().await.is_none());

    let refreshed = server.list_tools().await.unwrap();
    assert_eq!(
        refreshed
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        vec!["tool_2"]
    );
    assert_eq!(server.list_tools().await.unwrap(), refreshed);
    assert_eq!(log.calls_of("tools/list").len(), 3);
    assert_ne!(original, refreshed);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn successful_page_budget_survives_failed_page_retry() {
    openai_agents_sdk::set_tracing_disabled(true);
    for complete in [false, true] {
        let served = Arc::new(AtomicUsize::new(0));
        let (url, log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| match m {
            "initialize" => McpReply::Result(initialize_result(params)),
            "tools/list" => {
                let index = served.fetch_add(1, Ordering::SeqCst);
                match index {
                    0 => McpReply::Result(tools_page(0, false, false)),
                    // A transient transport failure on the second page.
                    1 => McpReply::Status(503),
                    2 => McpReply::Result(tools_page(1, complete, false)),
                    _ => McpReply::Result(tools_page(2, true, false)),
                }
            }
            _ => McpReply::Result(json!({})),
        }))
        .await;

        let server = server(&url)
            .with_max_list_pages(Some(2))
            .with_max_retry_attempts(2)
            .with_cache_tools_list(true);
        server.connect().await.unwrap();

        if complete {
            let tools = server.list_tools().await.unwrap();
            assert_eq!(
                tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
                vec!["tool_0", "tool_1"]
            );
        } else {
            let error = server.list_tools().await.unwrap_err();
            assert!(
                error.to_string().contains("exceeded max_list_pages"),
                "{error}"
            );
            assert!(server.cached_tools().await.is_none());
        }
        // The retried page asks for the same cursor it failed on.
        let calls = log.calls_of("tools/list");
        assert!(calls.len() >= 3, "{calls:?}");
        assert_eq!(calls[1]["cursor"], json!("private-cursor-0"));
        assert_eq!(calls[2]["cursor"], json!("private-cursor-0"));
        server.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn runner_stops_during_tool_enumeration_before_model_execution() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, log) = spawn_mcp(endless_pages("tools/list", false)).await;
    let server = Arc::new(server(&url).with_max_list_pages(Some(1))) as Arc<dyn MCPServer>;
    server.connect().await.unwrap();

    let model = Arc::new(FakeModel::new());
    let agent: Agent<()> = Agent::new("test")
        .with_mcp_server(server.clone())
        .with_model_instance(model.clone());

    let error = Runner::run(&agent, "hello").await.unwrap_err();
    assert!(
        error.to_string().contains("exceeded max_list_pages"),
        "{error}"
    );
    assert!(model.calls().is_empty(), "the model must not be called");
    assert_eq!(log.calls_of("tools/list").len(), 1);
    assert!(server.cached_tools().await.is_none());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn repeated_cursor_is_rejected_without_naming_the_cursor() {
    // `test_paginated_list_cycle_does_not_retain_opaque_cursor` from test_server_errors.py.
    openai_agents_sdk::set_tracing_disabled(true);
    for method in ["tools/list", "prompts/list"] {
        let wanted = method.to_string();
        let (url, _log) = spawn_mcp(Arc::new(move |m: &str, params: &Value| {
            if m == "initialize" {
                return McpReply::Result(initialize_result(params));
            }
            if m != wanted {
                return McpReply::Result(json!({}));
            }
            // Always hand back the same cursor.
            McpReply::Result(if wanted == "tools/list" {
                json!({"tools": [], "nextCursor": "SECRET_OPAQUE_CURSOR"})
            } else {
                json!({"prompts": [], "nextCursor": "SECRET_OPAQUE_CURSOR"})
            })
        }))
        .await;
        let server = server(&url)
            .with_max_retry_attempts(0)
            .with_cache_tools_list(true);
        server.connect().await.unwrap();

        let error = if method == "tools/list" {
            server.list_tools().await.unwrap_err()
        } else {
            server.list_prompts().await.unwrap_err()
        };
        let text = error.to_string();
        let subject = if method == "tools/list" {
            "tools"
        } else {
            "prompts"
        };
        assert!(
            text.contains(&format!("repeated cursor while listing {subject}")),
            "{text}"
        );
        assert!(!text.contains("SECRET_OPAQUE_CURSOR"), "{text}");
        assert!(server.cached_tools().await.is_none());
        server.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn page_limit_failure_does_not_name_the_cursor_or_payload() {
    // `test_page_limit_failure_does_not_retain_payload_or_cursor`.
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, _log) = spawn_mcp(Arc::new(|m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/list" => McpReply::Result(json!({
            "tools": [{"name": "SECRET_LIST_ENTRY", "inputSchema": {"type": "object"}}],
            "nextCursor": "SECRET_OPAQUE_CURSOR",
        })),
        _ => McpReply::Result(json!({})),
    }))
    .await;
    let server = MCPServerStreamableHttp::new(
        "sse: https://user:s3cr3t_pw@mcp.example.com/sse?api_key=SECRET_QS_KEY",
        MCPServerStreamableHttpParams {
            url: url.clone(),
            ..Default::default()
        },
    )
    .with_max_list_pages(Some(1))
    .with_cache_tools_list(true);
    server.connect().await.unwrap();

    let text = server.list_tools().await.unwrap_err().to_string();
    assert!(text.contains("exceeded max_list_pages"), "{text}");
    for secret in [
        "SECRET_OPAQUE_CURSOR",
        "SECRET_LIST_ENTRY",
        "s3cr3t_pw",
        "SECRET_QS_KEY",
    ] {
        assert!(!text.contains(secret), "{secret} leaked: {text}");
    }
    assert!(server.cached_tools().await.is_none());
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// test_mcp_pagination_integration.py
// ---------------------------------------------------------------------------

/// Serves two pages of tools and two pages of prompts, as `tests/mcp/servers/paginated.py` does.
fn paginated_server() -> Responder {
    Arc::new(|m: &str, params: &Value| match m {
        "initialize" => McpReply::Result(initialize_result(params)),
        "tools/list" => {
            let cursor = params.get("cursor").and_then(Value::as_str);
            McpReply::Result(match cursor {
                None => json!({
                    "tools": [{"name": "first_page_tool", "inputSchema": {"type": "object", "properties": {}}}],
                    "nextCursor": "page-2",
                }),
                Some(_) => json!({
                    "tools": [{"name": "second_page_tool", "inputSchema": {"type": "object", "properties": {}}}],
                }),
            })
        }
        "prompts/list" => {
            let cursor = params.get("cursor").and_then(Value::as_str);
            McpReply::Result(match cursor {
                None => json!({
                    "prompts": [{"name": "first_page_prompt"}],
                    "nextCursor": "page-2",
                }),
                Some(_) => json!({"prompts": [{"name": "second_page_prompt"}]}),
            })
        }
        "tools/call" => McpReply::Result(json!({
            "content": [{"type": "text", "text": format!("called:{}", params["name"].as_str().unwrap_or(""))}],
        })),
        _ => McpReply::Result(json!({})),
    })
}

#[tokio::test]
async fn server_auto_paginates_tools_and_prompts() {
    openai_agents_sdk::set_tracing_disabled(true);
    for max_list_pages in [None, Some(2usize)] {
        let (url, _log) = spawn_mcp(paginated_server()).await;
        let server = server(&url)
            .with_cache_tools_list(true)
            .with_max_list_pages(max_list_pages);
        server.connect().await.unwrap();

        let tools = server.list_tools().await.unwrap();
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["first_page_tool", "second_page_tool"]
        );
        let prompts = server.list_prompts().await.unwrap();
        assert_eq!(
            prompts
                .iter()
                .map(|p| p["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["first_page_prompt", "second_page_prompt"]
        );
        assert!(server.server_initialize_result().await.is_some());
        server.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn agent_calls_tool_from_second_page() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (url, _log) = spawn_mcp(paginated_server()).await;
    let server = Arc::new(server(&url).with_cache_tools_list(true)) as Arc<dyn MCPServer>;
    server.connect().await.unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![function_call("second_page_tool", "{}")],
        vec![assistant_message("done")],
    ]);
    let agent: Agent<()> = Agent::new("test")
        .with_mcp_server(server.clone())
        .with_mcp_config(MCPConfig::default())
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Call the second-page tool.")
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let outputs: Vec<String> = result
        .new_items
        .iter()
        .filter_map(|item| {
            let value = item.to_input_item().0;
            (value["type"] == "function_call_output").then(|| value["output"].to_string())
        })
        .collect();
    assert!(
        outputs
            .iter()
            .any(|o| o.contains("called:second_page_tool")),
        "{outputs:?}"
    );
    server.cleanup().await.unwrap();
}
