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
    assert_eq!(server.list_tools().await.unwrap().len(), 2);
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
    assert_eq!(tools.len(), 1, "blocked tool is filtered out");
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
    model.add_turn(vec![function_call("ops_add", r#"{"a":1,"b":2}"#)]);
    let agent = Agent::<()>::new("Ops")
        .with_mcp_server(server.clone())
        .with_mcp_config(MCPConfig {
            convert_schemas_to_strict: true,
            include_server_in_tool_names: true,
        })
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
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let header_end = loop {
                    let n = sock.read(&mut tmp).await.unwrap();
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break p + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                if head.starts_with("GET /sse") {
                    sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n\r\n")
                        .await
                        .unwrap();
                    sock.write_all(b"event: endpoint\ndata: /messages?session_id=abc\n\n")
                        .await
                        .unwrap();
                    let mut rx = rx.lock().await.take().unwrap();
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
                    let n = sock.read(&mut tmp).await.unwrap();
                    buf.extend_from_slice(&tmp[..n]);
                }
                assert!(head.starts_with("POST /messages?session_id=abc"));
                let req: serde_json::Value =
                    serde_json::from_slice(&buf[header_end..header_end + len]).unwrap();
                sock.write_all(b"HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n")
                    .await
                    .unwrap();
                let Some(id) = req.get("id").cloned() else {
                    return;
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
                tx.send(
                    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                )
                .unwrap();
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
