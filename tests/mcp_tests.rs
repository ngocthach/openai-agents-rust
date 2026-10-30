//! MCP stdio integration test against a stdlib-only Python mock server.
#![cfg(feature = "mcp")]

use std::sync::Arc;

use openai_agents_sdk::mcp::{
    MCPServer, MCPServerManager, MCPServerStdio, MCPServerStdioParams, RequireApproval,
    create_static_tool_filter,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, MCPConfig, Runner};

fn python() -> Option<String> {
    for p in ["python3", "python"] {
        if std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(p.to_string());
        }
    }
    None
}

fn server(name: &str) -> Option<MCPServerStdio> {
    let py = python()?;
    Some(MCPServerStdio::new(
        name,
        MCPServerStdioParams {
            command: py,
            args: vec![
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/mock_mcp_server.py"
                )
                .into(),
            ],
            ..Default::default()
        },
    ))
}

#[tokio::test]
async fn agent_calls_mcp_tools() {
    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("math") else {
        eprintln!("python not available; skipping");
        return;
    };
    let server: Arc<dyn MCPServer> = Arc::new(s.with_cache_tools_list(true).with_tool_filter(
        create_static_tool_filter(None, Some(vec!["danger_delete".into()])),
    ));
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap().len(), 6);
    let prompts = server.list_prompts().await.unwrap();
    assert_eq!(prompts[0]["name"], "review");

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("add", r#"{"a":7,"b":22}"#)]);
    model.add_turn(vec![assistant_message("29")]);
    let agent = Agent::<()>::new("Calc")
        .with_mcp_server(server.clone())
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "Add 7 and 22").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("29"));
    assert_eq!(r.new_items[1].tool_output().unwrap(), "29");
    let tools = &model.calls()[0].tools;
    assert_eq!(tools.len(), 5, "blocked tool is filtered out");
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn exposes_resources_metadata_and_request_meta() {
    use openai_agents_sdk::mcp::{MCPToolMetaContext, get_mcp_server_log_name};
    use serde_json::json;

    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("res") else {
        eprintln!("python not available; skipping");
        return;
    };
    let server: Arc<dyn MCPServer> = Arc::new(
        s.with_cache_tools_list(true)
            .with_tool_meta_resolver(|ctx: &MCPToolMetaContext| {
                let mut meta = serde_json::Map::new();
                meta.insert("tool".into(), json!(ctx.tool_name.clone()));
                Some(meta)
            }),
    );
    assert!(server.cached_tools().await.is_none());
    server.connect().await.unwrap();

    let resources = server.list_resources(None).await.unwrap();
    assert_eq!(resources["resources"][0]["name"], "notes");
    let templates = server.list_resource_templates(None).await.unwrap();
    assert_eq!(
        templates["resourceTemplates"][0]["uriTemplate"],
        "file:///{path}"
    );
    let read = server.read_resource("file:///notes.txt").await.unwrap();
    assert_eq!(read["contents"][0]["text"], "hello resource");

    let init = server.server_initialize_result().await.unwrap();
    assert_eq!(init["serverInfo"]["name"], "mock");

    // A title-only tool reaches the model with its title as the description.
    let tools = server.list_tools().await.unwrap();
    assert!(server.cached_tools().await.is_some());
    let titled = tools.iter().find(|t| t.name == "titled_only").unwrap();
    assert_eq!(
        openai_agents_sdk::mcp::resolve_mcp_tool_description_for_model(titled),
        "Titled Only"
    );

    // The resolver's `_meta` reaches the server verbatim, with explicit keys winning.
    let echo = tools.iter().find(|t| t.name == "echo_meta").unwrap();
    let mut explicit = serde_json::Map::new();
    explicit.insert("extra".into(), json!(1));
    let out = openai_agents_sdk::mcp::util::invoke_mcp_tool(
        &server,
        echo,
        "{}",
        Some(explicit),
        Some("res_echo_meta"),
    )
    .await
    .unwrap();
    match out {
        openai_agents_sdk::tool::ToolOutput::Text(t) => {
            // The transport adds its own protocol keys; both of ours arrive alongside them.
            let echoed: serde_json::Value = serde_json::from_str(&t).unwrap();
            assert_eq!(echoed["tool"], "echo_meta");
            assert_eq!(echoed["extra"], 1);
        }
        other => panic!("{other:?}"),
    }

    assert_eq!(get_mcp_server_log_name("res"), "res");
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn mcp_approvals_and_prefixed_names() {
    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("ops") else {
        return;
    };
    let server: Arc<dyn MCPServer> = Arc::new(s.with_require_approval(RequireApproval::Always));
    let mut manager = MCPServerManager::new(vec![server.clone(), server.clone()]);
    manager.connect_all().await.unwrap();
    assert_eq!(manager.active_servers().len(), 1);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("mcp_ops__add", r#"{"a":1,"b":2}"#)]);
    let agent = Agent::<()>::new("Ops")
        .with_mcp_server(server.clone())
        .with_mcp_config(
            MCPConfig::default()
                .with_convert_schemas_to_strict(true)
                .with_include_server_in_tool_names(true),
        )
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "add").await.unwrap();
    assert_eq!(r.interruptions.len(), 1);
    let mut state = r.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    model.add_turn(vec![assistant_message("3")]);
    let r = Runner::run(&agent, state).await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("3"));
    manager.cleanup_all().await;
}

/// Write one raw HTTP response, reporting a closed socket instead of panicking.
async fn write_raw(sock: &mut tokio::net::TcpStream, raw: String) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    sock.write_all(raw.as_bytes()).await
}

/// What a [`spawn_streamable_http_mcp_server`] mock recorded and how it should misbehave.
#[derive(Default)]
struct StreamableHttpMock {
    /// `Authorization` header values seen, in order.
    authorizations: std::sync::Mutex<Vec<String>>,
    /// `DELETE` requests received (the session-ending request).
    deletes: std::sync::atomic::AtomicUsize,
    /// `initialize` requests received; one per connection the client opens.
    initializes: std::sync::atomic::AtomicUsize,
    /// `tools/call` requests received.
    calls: std::sync::atomic::AtomicUsize,
    /// Reject the `notifications/initialized` POST with 400.
    reject_initialized: bool,
    /// Fail this many leading `tools/call` requests with 503, a transport-class failure.
    fail_calls: usize,
}

/// A minimal MCP server speaking the Streamable HTTP transport, with no SSE stream.
async fn spawn_streamable_http_mcp_server(mock: Arc<StreamableHttpMock>) -> String {
    use std::sync::atomic::Ordering;

    use tokio::io::AsyncReadExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mock = mock.clone();
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
                    let header = |name: &str| -> Option<String> {
                        head.lines().find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
                        })
                    };
                    if let Some(auth) = header("authorization") {
                        mock.authorizations.lock().unwrap().push(auth);
                    }
                    let len: usize = header("content-length")
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

                    if head.starts_with("GET ") {
                        // No server-initiated stream: rmcp treats 405 as "no SSE".
                        if write_raw(&mut sock, status_only("405 Method Not Allowed"))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    if head.starts_with("DELETE ") {
                        mock.deletes.fetch_add(1, Ordering::SeqCst);
                        if write_raw(&mut sock, status_only("200 OK")).await.is_err() {
                            return;
                        }
                        continue;
                    }

                    let req: serde_json::Value = match serde_json::from_slice(&body) {
                        Ok(v) => v,
                        Err(_) => {
                            let _ = write_raw(&mut sock, status_only("400 Bad Request")).await;
                            return;
                        }
                    };
                    let method = req["method"].as_str().unwrap_or("").to_string();
                    if method == "notifications/initialized" {
                        let code = if mock.reject_initialized {
                            "400 Bad Request"
                        } else {
                            "202 Accepted"
                        };
                        if write_raw(&mut sock, status_only(code)).await.is_err() {
                            return;
                        }
                        continue;
                    }
                    let Some(id) = req.get("id").cloned() else {
                        if write_raw(&mut sock, status_only("202 Accepted"))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    };
                    if method == "tools/call"
                        && mock.calls.fetch_add(1, Ordering::SeqCst) < mock.fail_calls
                    {
                        if write_raw(&mut sock, status_only("503 Service Unavailable"))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    let result = match method.as_str() {
                        "initialize" => {
                            mock.initializes.fetch_add(1, Ordering::SeqCst);
                            serde_json::json!({
                                "protocolVersion": req["params"]["protocolVersion"],
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "http-mock", "version": "1"}})
                        }
                        "tools/list" => serde_json::json!({"tools": [{"name": "echo",
                            "description": "Echo", "inputSchema": {"type": "object",
                            "properties": {"text": {"type": "string"}}}}]}),
                        "tools/call" => serde_json::json!({"content": [{"type": "text",
                            "text": "echoed"}]}),
                        _ => serde_json::json!({}),
                    };
                    let payload = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
                        .to_string();
                    let session = if method == "initialize" {
                        "mcp-session-id: sess-abc\r\n"
                    } else {
                        ""
                    };
                    let raw = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{session}\
                         content-length: {}\r\n\r\n{payload}",
                        payload.len()
                    );
                    if write_raw(&mut sock, raw).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}/mcp")
}

/// A minimal MCP server speaking the legacy HTTP+SSE transport.
async fn spawn_sse_mcp_server() -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::mpsc;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let tx = tx.clone();
            let rx = rx.clone();
            tokio::spawn(async move {
                // One task per accepted connection, serving HTTP/1.1 keep-alive: the client
                // pools the POST connection, so answering a single request and dropping the
                // socket would make the next pooled POST fail with a transport send error.
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
                    if head.starts_with("GET /sse") {
                        if sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n\r\n")
                            .await
                            .is_err()
                        {
                            return;
                        }
                        if sock
                            .write_all(b"event: endpoint\ndata: /messages?session_id=abc\n\n")
                            .await
                            .is_err()
                        {
                            return;
                        }
                        let Some(mut rx) = rx.lock().await.take() else {
                            return;
                        };
                        while let Some(msg) = rx.recv().await {
                            let frame = format!("event: message\ndata: {msg}\n\n");
                            if sock.write_all(frame.as_bytes()).await.is_err() {
                                break;
                            }
                        }
                        return;
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
                    assert!(head.starts_with("POST /messages?session_id=abc"));
                    let req: serde_json::Value =
                        serde_json::from_slice(&buf[header_end..header_end + len]).unwrap();
                    buf.drain(..header_end + len);
                    if sock
                        .write_all(b"HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n")
                        .await
                        .is_err()
                    {
                        return;
                    }
                    let Some(id) = req.get("id").cloned() else {
                        continue;
                    };
                    let result = match req["method"].as_str().unwrap() {
                        "initialize" => serde_json::json!({
                            "protocolVersion": req["params"]["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "sse-mock", "version": "1"}}),
                        "tools/list" => serde_json::json!({"tools": [{"name": "echo",
                            "description": "Echo", "inputSchema": {"type": "object",
                            "properties": {"text": {"type": "string"}}}}]}),
                        "tools/call" => serde_json::json!({"content": [{"type": "text",
                            "text": format!("echo: {}", req["params"]["arguments"]["text"].as_str().unwrap())}]}),
                        _ => serde_json::json!({}),
                    };
                    if tx
                        .send(
                            serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
                                .to_string(),
                        )
                        .is_err()
                    {
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}/sse")
}

#[tokio::test]
async fn agent_calls_tools_over_legacy_sse() {
    use openai_agents_sdk::mcp::{MCPServerSse, MCPServerSseParams};
    openai_agents_sdk::set_tracing_disabled(true);
    let url = spawn_sse_mcp_server().await;
    let server = Arc::new(MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url,
            ..Default::default()
        },
    ));
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap()[0].name, "echo");
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("echo", r#"{"text":"hi"}"#)]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("A")
        .with_mcp_server(server.clone())
        .with_model_instance(model);
    let r = Runner::run(&agent, "echo hi").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
    assert_eq!(r.new_items[1].tool_output().unwrap(), "echo: hi");
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn streamable_http_records_the_session_id_and_sends_the_auth_header() {
    use openai_agents_sdk::mcp::{MCPServerStreamableHttp, MCPServerStreamableHttpParams};

    openai_agents_sdk::set_tracing_disabled(true);
    let mock = Arc::new(StreamableHttpMock::default());
    let url = spawn_streamable_http_mcp_server(mock.clone()).await;
    let server = MCPServerStreamableHttp::new(
        "streamable_http: http://user:secret@127.0.0.1/mcp",
        MCPServerStreamableHttpParams {
            url,
            auth: Some(Arc::new(|| Some("from-auth".to_string()))),
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap()[0].name, "echo");
    // The transport-level session id the server handed out is readable.
    assert_eq!(server.session_id().await.as_deref(), Some("sess-abc"));
    // `auth` supplies the Authorization header on every request.
    let auths = mock.authorizations.lock().unwrap().clone();
    assert!(
        !auths.is_empty() && auths.iter().all(|a| a == "Bearer from-auth"),
        "{auths:?}"
    );
    server.cleanup().await.unwrap();
    // `terminate_on_close` defaults to true, so the session is deleted on close.
    assert_eq!(
        mock.deletes.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "session should be terminated on close"
    );
    assert!(server.session_id().await.is_none());
}

#[tokio::test]
async fn streamable_http_can_keep_the_session_and_tolerate_a_rejected_initialized() {
    use openai_agents_sdk::mcp::{MCPServerStreamableHttp, MCPServerStreamableHttpParams};

    openai_agents_sdk::set_tracing_disabled(true);
    let mock = Arc::new(StreamableHttpMock {
        reject_initialized: true,
        ..Default::default()
    });
    let url = spawn_streamable_http_mcp_server(mock.clone()).await;
    let server = MCPServerStreamableHttp::new(
        "http",
        MCPServerStreamableHttpParams {
            url,
            terminate_on_close: false,
            ignore_initialized_notification_failure: true,
            ..Default::default()
        },
    );
    server.connect().await.unwrap();
    assert_eq!(server.list_tools().await.unwrap().len(), 1);
    server.cleanup().await.unwrap();
    assert_eq!(
        mock.deletes.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "terminate_on_close = false must not DELETE the session"
    );
}

#[tokio::test]
async fn a_transport_failure_is_retried_on_an_isolated_session() {
    use openai_agents_sdk::mcp::{MCPServerStreamableHttp, MCPServerStreamableHttpParams};

    openai_agents_sdk::set_tracing_disabled(true);
    let mock = Arc::new(StreamableHttpMock {
        fail_calls: 1,
        ..Default::default()
    });
    let url = spawn_streamable_http_mcp_server(mock.clone()).await;
    let server = MCPServerStreamableHttp::new(
        "http",
        MCPServerStreamableHttpParams {
            url,
            ..Default::default()
        },
    )
    .with_max_retry_attempts(1);
    server.connect().await.unwrap();
    let out = server.call_tool("echo", None).await.unwrap();
    assert_eq!(out["content"][0]["text"], "echoed");
    // The retry ran on a second connection, so the server saw a second `initialize`.
    assert_eq!(
        mock.initializes.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the isolated retry should open its own session"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn transport_errors_never_carry_credentials() {
    use openai_agents_sdk::mcp::{MCPServerStreamableHttp, MCPServerStreamableHttpParams};

    openai_agents_sdk::set_tracing_disabled(true);
    // Nothing listens on this port, so connecting fails with a transport error naming the URL.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let url = format!("http://user:hunter2@{addr}/mcp?token=supersecret");
    let server = MCPServerStreamableHttp::new(
        "http",
        MCPServerStreamableHttpParams {
            url,
            ..Default::default()
        },
    );
    let message = server.connect().await.unwrap_err().to_string();
    assert!(!message.contains("hunter2"), "{message}");
    assert!(!message.contains("supersecret"), "{message}");

    // The legacy SSE transport builds its own errors, so it is checked separately.
    use openai_agents_sdk::mcp::{MCPServerSse, MCPServerSseParams};
    let sse = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: format!("http://user:hunter2@{addr}/sse?token=supersecret"),
            ..Default::default()
        },
    );
    let message = sse.connect().await.unwrap_err().to_string();
    assert!(!message.contains("hunter2"), "{message}");
    assert!(!message.contains("supersecret"), "{message}");
}

#[tokio::test]
async fn async_filters_and_approvals_are_awaited() {
    use openai_agents_sdk::mcp::{RequireApproval, ToolFilter};

    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("filtered") else {
        eprintln!("python not available; skipping");
        return;
    };
    let server: Arc<dyn MCPServer> = Arc::new(
        s.with_tool_filter(ToolFilter::dynamic_async(|ctx, tool| {
            // The run context reaches the filter, as Python's `ToolFilterContext` does.
            let has_ctx = ctx.run_context::<()>().is_some();
            let name = tool.name.clone();
            async move {
                tokio::task::yield_now().await;
                has_ctx && name == "add"
            }
        }))
        .with_require_approval(RequireApproval::dynamic_async(|_agent, tool| {
            let name = tool.name.clone();
            async move {
                tokio::task::yield_now().await;
                name == "add"
            }
        })),
    );
    server.connect().await.unwrap();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("add", r#"{"a":1,"b":2}"#)]);
    let agent = Agent::<()>::new("Filtered")
        .with_mcp_server(server.clone())
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "add").await.unwrap();
    assert_eq!(
        model.calls()[0].tools.len(),
        1,
        "the async filter keeps only `add`"
    );
    assert_eq!(r.interruptions.len(), 1, "the async policy asks to approve");
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_server_tool_policy_overrides_the_agent_failure_formatter() {
    use openai_agents_sdk::mcp::MCPToolPolicy;

    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("policy") else {
        eprintln!("python not available; skipping");
        return;
    };
    let policy = MCPToolPolicy::<()>::new().with_failure_error_function(Some(Arc::new(
        |_ctx: &openai_agents_sdk::RunContext<()>, _e: &openai_agents_sdk::AgentsError| {
            "server-level failure message".to_string()
        },
    )));
    let server: Arc<dyn MCPServer> = Arc::new(s.with_tool_policy(policy));
    server.connect().await.unwrap();
    let model = Arc::new(FakeModel::new());
    // `fail` is the mock server's always-failing tool.
    model.add_turn(vec![function_call("fail", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Policy")
        .with_mcp_server(server.clone())
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "fail").await.unwrap();
    assert_eq!(
        r.new_items[1].tool_output().unwrap(),
        "server-level failure message"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn server_notifications_reach_the_message_handler() {
    use std::sync::Mutex;

    openai_agents_sdk::set_tracing_disabled(true);
    let Some(s) = server("notifier") else {
        eprintln!("python not available; skipping");
        return;
    };
    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let server: Arc<dyn MCPServer> =
        Arc::new(s.with_message_handler(move |m: &serde_json::Value| {
            sink.lock().unwrap().push(m.clone());
        }));
    server.connect().await.unwrap();
    server.call_tool("notify", None).await.unwrap();
    // The notification is delivered on the session task, so give it a turn to land.
    for _ in 0..50 {
        if !seen.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let messages = seen.lock().unwrap().clone();
    assert!(
        messages
            .iter()
            .any(|m| m["method"] == "notifications/resources/updated"
                && m["params"]["uri"] == "file:///notes.txt"),
        "{messages:?}"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_stdio_server_rejects_a_non_utf8_encoding() {
    use openai_agents_sdk::mcp::EncodingErrorHandler;

    openai_agents_sdk::set_tracing_disabled(true);
    let Some(py) = python() else {
        eprintln!("python not available; skipping");
        return;
    };
    let script = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mock_mcp_server.py"
    );
    let bad = MCPServerStdio::new(
        "latin",
        MCPServerStdioParams {
            command: py.clone(),
            args: vec![script.into()],
            encoding: "latin-1".into(),
            ..Default::default()
        },
    );
    let message = bad.connect().await.unwrap_err().to_string();
    assert!(message.contains("UTF-8 only"), "{message}");

    // A non-strict decoder still speaks to a well-behaved server.
    let lenient = MCPServerStdio::new(
        "lenient",
        MCPServerStdioParams {
            command: py,
            args: vec![script.into()],
            encoding_error_handler: EncodingErrorHandler::Replace,
            ..Default::default()
        },
    );
    lenient.connect().await.unwrap();
    assert!(
        lenient
            .list_tools()
            .await
            .unwrap()
            .iter()
            .any(|t| t.name == "add")
    );
    lenient.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_negative_retry_budget_retries_forever() {
    use openai_agents_sdk::mcp::{MCPServerStreamableHttp, MCPServerStreamableHttpParams};

    openai_agents_sdk::set_tracing_disabled(true);
    // Three failures in a row is more than any finite default would tolerate.
    let mock = Arc::new(StreamableHttpMock {
        fail_calls: 3,
        ..Default::default()
    });
    let url = spawn_streamable_http_mcp_server(mock.clone()).await;
    let params = MCPServerStreamableHttpParams {
        url: url.clone(),
        ..Default::default()
    };

    let bounded =
        MCPServerStreamableHttp::new("bounded", params.clone()).with_max_retry_attempts(1);
    bounded.connect().await.unwrap();
    assert!(
        bounded.call_tool("echo", None).await.is_err(),
        "one retry cannot survive three failures"
    );
    bounded.cleanup().await.unwrap();

    let unlimited = MCPServerStreamableHttp::new("unlimited", params)
        .with_max_retry_attempts(-1)
        .with_retry_backoff_seconds_max(Some(0.0));
    unlimited.connect().await.unwrap();
    let out = unlimited.call_tool("echo", None).await.unwrap();
    assert_eq!(out["content"][0]["text"], "echoed");
    unlimited.cleanup().await.unwrap();
}
