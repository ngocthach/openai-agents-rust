//! Parity port of `tests/mcp/test_caching.py` and `tests/mcp/test_mcp_approval.py`.
//!
//! Python patches `ClientSession.list_tools` on a stdio server. Rust has no such seam, so the
//! caching tests drive a real `MCPServerStreamableHttp` against a local in-process JSON-RPC
//! server whose `tools/list` / `tools/call` answers are scripted per test.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::Result;
use openai_agents_sdk::mcp::{
    MCPServer, MCPServerConfig, MCPServerStreamableHttp, MCPServerStreamableHttpParams, MCPTool,
    RequireApproval,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_full};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type Handler =
    Arc<dyn Fn(String, Value) -> Pin<Box<dyn Future<Output = Value> + Send>> + Send + Sync>;

/// A minimal JSON-RPC MCP server (Streamable HTTP, plain JSON replies). `handler` answers
/// `tools/list` and `tools/call`; its `Value` is the JSON-RPC `result`.
async fn spawn_mcp(handler: Handler) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let end = loop {
                        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break p + 4;
                        }
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..end]).to_string();
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse().ok())?
                        })
                        .unwrap_or(0);
                    while buf.len() < end + len {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    let body = buf[end..end + len].to_vec();
                    buf.drain(..end + len);
                    let status = |c: &str| format!("HTTP/1.1 {c}\r\ncontent-length: 0\r\n\r\n");
                    if head.starts_with("GET ") {
                        let _ = sock
                            .write_all(status("405 Method Not Allowed").as_bytes())
                            .await;
                        continue;
                    }
                    if head.starts_with("DELETE ") {
                        let _ = sock.write_all(status("200 OK").as_bytes()).await;
                        continue;
                    }
                    let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    let method = req["method"].as_str().unwrap_or("").to_string();
                    let Some(id) = req.get("id").cloned() else {
                        let _ = sock.write_all(status("202 Accepted").as_bytes()).await;
                        continue;
                    };
                    let result = match method.as_str() {
                        "initialize" => json!({
                            "protocolVersion": req["params"]["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "mock", "version": "1"}}),
                        "tools/list" | "tools/call" => {
                            handler(method.clone(), req["params"].clone()).await
                        }
                        _ => json!({}),
                    };
                    let payload = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
                    let raw = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{payload}",
                        payload.len()
                    );
                    if sock.write_all(raw.as_bytes()).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}/mcp")
}

fn tools_result(tools: Vec<Value>) -> Value {
    json!({"tools": tools})
}

fn wire_tool(name: &str, description: Option<&str>, schema: Value) -> Value {
    let mut t = json!({"name": name, "inputSchema": schema});
    if let Some(d) = description {
        t["description"] = json!(d);
    }
    t
}

fn http_server(url: String) -> MCPServerStreamableHttp {
    MCPServerStreamableHttp::new(
        "mock",
        MCPServerStreamableHttpParams {
            url,
            ..Default::default()
        },
    )
    .with_cache_tools_list(true)
}

fn names(tools: &[MCPTool]) -> Vec<&str> {
    tools.iter().map(|t| t.name.as_str()).collect()
}

fn ok_call() -> Value {
    json!({"content": [{"type": "text", "text": "ok"}]})
}

// ---------------------------------------------------------------------------
// test_caching.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_caching_works() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let url = spawn_mcp(Arc::new(move |m, _| {
        let c = c.clone();
        Box::pin(async move {
            if m == "tools/list" {
                c.fetch_add(1, Ordering::SeqCst);
                tools_result(vec![
                    wire_tool("tool1", None, json!({})),
                    wire_tool("tool2", None, json!({})),
                ])
            } else {
                ok_call()
            }
        })
    }))
    .await;
    let server = http_server(url);
    server.connect().await.unwrap();

    let first = server.list_tools().await.unwrap();
    assert_eq!(names(&first), ["tool1", "tool2"]);
    assert_eq!(count.load(Ordering::SeqCst), 1, "fetched once");
    assert_eq!(server.list_tools().await.unwrap(), first);
    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "second call is served from the cache"
    );

    server.invalidate_tools_cache().await;
    assert_eq!(server.list_tools().await.unwrap(), first);
    assert_eq!(
        count.load(Ordering::SeqCst),
        2,
        "invalidation forces a refetch"
    );
    assert_eq!(server.list_tools().await.unwrap(), first);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn cached_tools_is_none_before_the_first_list() {
    let url = spawn_mcp(Arc::new(|_, _| Box::pin(async { tools_result(vec![]) }))).await;
    let server = http_server(url);
    server.connect().await.unwrap();
    assert!(server.cached_tools().await.is_none());
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn pagination_caches_every_page_and_the_list_is_unfiltered() {
    // Python filters after caching, so `cached_tools` holds both pages. In Rust the filter is
    // applied by the run pipeline, so `list_tools` itself always returns the unfiltered list.
    let requests: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let r = requests.clone();
    let url = spawn_mcp(Arc::new(move |m, p| {
        let r = r.clone();
        Box::pin(async move {
            if m != "tools/list" {
                return ok_call();
            }
            r.lock().unwrap().push(p.clone());
            if p.get("cursor").and_then(Value::as_str) == Some("page2") {
                tools_result(vec![wire_tool("second_page_tool", None, json!({}))])
            } else {
                let mut res = tools_result(vec![wire_tool("first_page_tool", None, json!({}))]);
                res["nextCursor"] = json!("page2");
                res
            }
        })
    }))
    .await;
    let server = http_server(url);
    server.connect().await.unwrap();
    let listed = server.list_tools().await.unwrap();
    assert_eq!(names(&listed), ["first_page_tool", "second_page_tool"]);
    assert_eq!(
        names(&server.cached_tools().await.unwrap()),
        ["first_page_tool", "second_page_tool"]
    );
    assert_eq!(
        names(&server.list_tools().await.unwrap()),
        ["first_page_tool", "second_page_tool"]
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "two page requests in total, none on the cached call"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn returned_lists_are_owned_snapshots_of_the_cache() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let url = spawn_mcp(Arc::new(move |m, _| {
        let c = c.clone();
        Box::pin(async move {
            if m == "tools/list" {
                c.fetch_add(1, Ordering::SeqCst);
                tools_result(vec![wire_tool(
                    "tool1",
                    None,
                    json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
                ), wire_tool("tool2", None, json!({}))])
            } else {
                ok_call()
            }
        })
    }))
    .await;
    let server = http_server(url);
    server.connect().await.unwrap();

    // Mutating a returned list, a returned tool or a `cached_tools` snapshot cannot reach the
    // cache: Rust hands out owned clones (Python needs explicit copying to get the same).
    let mut returned = server.list_tools().await.unwrap();
    returned.pop();
    returned[0].description = Some("mutated".into());
    returned[0].input_schema["required"] = json!([]);
    let mut snapshot = server.cached_tools().await.unwrap();
    snapshot.push(MCPTool {
        meta: None,
        name: "injected".into(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations: None,
    });
    snapshot[0].input_schema["required"] = json!([]);

    let later = server.list_tools().await.unwrap();
    assert_eq!(names(&later), ["tool1", "tool2"]);
    assert_eq!(later[0].description, None);
    assert_eq!(later[0].input_schema["required"], json!(["q"]));
    assert_eq!(
        server.cached_tools().await.unwrap()[0].input_schema["required"],
        json!(["q"])
    );
    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "the cache kept serving both tools"
    );

    // Call-time validation still sees the real schema.
    let error = server
        .call_tool("tool1", Some(Map::new()))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("missing required parameters: q"),
        "{error}"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn a_dynamic_filter_cannot_corrupt_cached_tool_schemas() {
    // Python's callable filter receives the cached tool object and could mutate it. Rust passes
    // `&MCPTool`, so the mutation cannot be written; what remains observable is that listing
    // through a filter leaves the cache and call-time validation intact.
    use openai_agents_sdk::mcp::ToolFilter;
    let url = spawn_mcp(Arc::new(|m, _| {
        Box::pin(async move {
            if m == "tools/list" {
                tools_result(vec![wire_tool(
                    "tool1",
                    None,
                    json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
                )])
            } else {
                ok_call()
            }
        })
    }))
    .await;
    let server = http_server(url).with_tool_filter(ToolFilter::dynamic(|_c, _t| true));
    server.connect().await.unwrap();
    let first = server.list_tools().await.unwrap();
    let later = server.list_tools().await.unwrap();
    assert_eq!(first[0].input_schema["required"], json!(["q"]));
    assert_eq!(later[0].input_schema["required"], json!(["q"]));
    assert_eq!(server.cached_tools().await.unwrap()[0].description, None);
    let error = server
        .call_tool("tool1", Some(Map::new()))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("missing required parameters: q"),
        "{error}"
    );
    server.cleanup().await.unwrap();
}

/// Gate letting a scripted `tools/list` answer block until released.
struct Gate {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[tokio::test]
async fn cache_invalidation_during_refresh_is_preserved() {
    let gate = Arc::new(Gate {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    let n = Arc::new(AtomicUsize::new(0));
    let (g, c) = (gate.clone(), n.clone());
    let url = spawn_mcp(Arc::new(move |m, _| {
        let (g, c) = (g.clone(), c.clone());
        Box::pin(async move {
            if m != "tools/list" {
                return ok_call();
            }
            let k = c.fetch_add(1, Ordering::SeqCst) + 1;
            if k == 2 {
                g.started.notify_one();
                g.release.notified().await;
            }
            let (d, s) = match k {
                1 => ("initial", json!({"required": ["q"]})),
                2 => ("before-second-invalidation", json!({})),
                _ => ("after-second-invalidation", json!({"required": ["latest"]})),
            };
            tools_result(vec![wire_tool("tool1", Some(d), s)])
        })
    }))
    .await;
    let server = Arc::new(http_server(url));
    server.connect().await.unwrap();
    assert_eq!(
        server.list_tools().await.unwrap()[0].description.as_deref(),
        Some("initial")
    );

    server.invalidate_tools_cache().await;
    let s2 = server.clone();
    let refresh = tokio::spawn(async move { s2.list_tools().await });
    gate.started.notified().await;
    server.invalidate_tools_cache().await;
    gate.release.notify_one();
    let refreshed = refresh.await.unwrap().unwrap();
    assert_eq!(
        refreshed[0].description.as_deref(),
        Some("before-second-invalidation")
    );
    assert_eq!(
        server.cached_tools().await.unwrap()[0]
            .description
            .as_deref(),
        Some("initial")
    );

    let latest = server.list_tools().await.unwrap();
    assert_eq!(
        latest[0].description.as_deref(),
        Some("after-second-invalidation")
    );
    assert_eq!(n.load(Ordering::SeqCst), 3);
    let error = server
        .call_tool("tool1", Some(Map::new()))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing required parameters: latest"),
        "{error}"
    );
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn older_concurrent_refresh_does_not_overwrite_newer_cache() {
    let gate = Arc::new(Gate {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    let n = Arc::new(AtomicUsize::new(0));
    let (g, c) = (gate.clone(), n.clone());
    let url = spawn_mcp(Arc::new(move |m, _| {
        let (g, c) = (g.clone(), c.clone());
        Box::pin(async move {
            if m != "tools/list" {
                return ok_call();
            }
            let k = c.fetch_add(1, Ordering::SeqCst) + 1;
            if k == 1 {
                g.started.notify_one();
                g.release.notified().await;
                return tools_result(vec![wire_tool(
                    "tool1",
                    Some("first-started"),
                    json!({"required": ["old"]}),
                )]);
            }
            tools_result(vec![wire_tool(
                "tool1",
                Some("second-started"),
                json!({"required": ["latest"]}),
            )])
        })
    }))
    .await;
    let server = Arc::new(http_server(url));
    server.connect().await.unwrap();
    let s1 = server.clone();
    let first = tokio::spawn(async move { s1.list_tools().await });
    gate.started.notified().await;
    let second = server.list_tools().await.unwrap();
    assert_eq!(second[0].description.as_deref(), Some("second-started"));
    gate.release.notify_one();
    let first = first.await.unwrap().unwrap();
    assert_eq!(first[0].description.as_deref(), Some("first-started"));
    let cached = server.cached_tools().await.unwrap();
    assert_eq!(cached[0].description.as_deref(), Some("second-started"));
    assert_eq!(cached[0].input_schema, json!({"required": ["latest"]}));
    server.cleanup().await.unwrap();
}

#[tokio::test]
async fn older_refresh_publishes_when_newer_refresh_fails() {
    let gate = Arc::new(Gate {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    let n = Arc::new(AtomicUsize::new(0));
    let (g, c) = (gate.clone(), n.clone());
    let url = spawn_mcp(Arc::new(move |m, _| {
        let (g, c) = (g.clone(), c.clone());
        Box::pin(async move {
            if m != "tools/list" {
                return ok_call();
            }
            let k = c.fetch_add(1, Ordering::SeqCst) + 1;
            if k == 1 {
                g.started.notify_one();
                g.release.notified().await;
                return tools_result(vec![wire_tool(
                    "tool1",
                    Some("first-started"),
                    json!({"required": ["q"]}),
                )]);
            }
            json!({"__bad": true})
        })
    }))
    .await;
    let server = Arc::new(http_server(url));
    server.connect().await.unwrap();
    let s1 = server.clone();
    let first = tokio::spawn(async move { s1.list_tools().await });
    gate.started.notified().await;
    let _ = server.list_tools().await;
    gate.release.notify_one();
    let first = first.await.unwrap().unwrap();
    assert_eq!(first[0].description.as_deref(), Some("first-started"));
    assert_eq!(
        server.cached_tools().await.unwrap()[0]
            .description
            .as_deref(),
        Some("first-started")
    );
    server.cleanup().await.unwrap();
}

// ---------------------------------------------------------------------------
// test_mcp_approval.py
// ---------------------------------------------------------------------------

/// Echo server that records the tool names it was asked to call.
#[derive(Debug)]
struct ApprovalServer {
    config: MCPServerConfig,
    calls: Mutex<Vec<String>>,
    tool_names: Vec<&'static str>,
}

impl ApprovalServer {
    fn new(tool_names: Vec<&'static str>, approval: RequireApproval) -> Arc<Self> {
        Arc::new(ApprovalServer {
            config: MCPServerConfig {
                require_approval: approval,
                ..Default::default()
            },
            calls: Mutex::new(Vec::new()),
            tool_names,
        })
    }
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl MCPServer for ApprovalServer {
    fn name(&self) -> &str {
        "fake_mcp_server"
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Ok(self
            .tool_names
            .iter()
            .map(|n| MCPTool {
                meta: None,
                name: (*n).into(),
                title: None,
                description: None,
                input_schema: json!({"type": "object", "properties": {}}),
                annotations: None,
            })
            .collect())
    }
    async fn call_tool(&self, name: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        self.calls.lock().unwrap().push(name.to_string());
        Ok(json!({"content": [{"type": "text", "text": format!("result_{name}")}]}))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(&self, _n: &str, _a: Option<BTreeMap<String, String>>) -> Result<Value> {
        Ok(Value::Null)
    }
}

/// Run `tool` once, approve the first interruption when `approve`, and return
/// `(interruptions on the first pass, final output text)`.
async fn run_with_approval(
    server: Arc<ApprovalServer>,
    tool: &str,
    context_text: &str,
) -> (usize, Option<String>) {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_full(tool, "{}", "call_1", None, None)]);
    // The follow-up turn is queued up front: it is consumed right away when the tool needs no
    // approval, and after the resume otherwise.
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("TestAgent")
        .with_mcp_server(server)
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, context_text).await.unwrap();
    let pending = first.interruptions.len();
    if pending == 0 {
        return (0, first.final_output_text());
    }
    assert_eq!(first.interruptions[0].tool_name, tool);
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let resumed = Runner::run(&agent, state).await.unwrap();
    assert!(resumed.interruptions.is_empty());
    (pending, resumed.final_output_text())
}

#[tokio::test]
async fn require_approval_always_pauses_and_resumes() {
    let server = ApprovalServer::new(vec!["add"], RequireApproval::Always);
    let (pending, output) = run_with_approval(server.clone(), "add", "call add").await;
    assert_eq!(pending, 1, "MCP tool should request approval");
    assert_eq!(server.calls(), ["add"]);
    assert_eq!(output.as_deref(), Some("done"));
}

#[tokio::test]
async fn require_approval_per_tool_mapping() {
    // Python: {"add": "always", "noop": "never"} and the `{"always": {"tool_names": [..]}}`
    // form both map to a per-tool policy; Rust has the single `PerTool` form for both.
    let per_tool: std::collections::HashMap<String, bool> =
        [("add".to_string(), true), ("noop".to_string(), false)]
            .into_iter()
            .collect();
    let server = ApprovalServer::new(vec!["add"], RequireApproval::PerTool(per_tool));
    let (pending, output) = run_with_approval(server.clone(), "add", "call add").await;
    assert_eq!(pending, 1);
    assert_eq!(server.calls(), ["add"]);
    assert_eq!(output.as_deref(), Some("done"));
}

#[tokio::test]
async fn policy_keyword_tool_names_are_plain_tool_names() {
    let per_tool: std::collections::HashMap<String, bool> =
        [("always".to_string(), true), ("never".to_string(), false)]
            .into_iter()
            .collect();
    let server = ApprovalServer::new(
        vec!["always", "never"],
        RequireApproval::PerTool(per_tool.clone()),
    );
    let (pending, _) = run_with_approval(server, "always", "call always").await;
    assert_eq!(pending, 1, "tool named 'always' should require approval");

    let server = ApprovalServer::new(vec!["always", "never"], RequireApproval::PerTool(per_tool));
    let (pending, _) = run_with_approval(server.clone(), "never", "call never").await;
    assert_eq!(pending, 0, "tool named 'never' should not require approval");
    assert_eq!(server.calls(), ["never"]);
}

#[tokio::test]
async fn callable_policy_allows_and_blocks_by_tool_name() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    let policy = RequireApproval::dynamic(move |_agent, tool| {
        s.lock().unwrap().push(tool.name.clone());
        tool.name == "guarded"
    });
    let server = ApprovalServer::new(vec!["guarded", "safe"], policy);
    let (pending, output) = run_with_approval(server.clone(), "guarded", "call guarded").await;
    assert_eq!(pending, 1);
    assert_eq!(output.as_deref(), Some("done"));
    let (pending, _) = run_with_approval(server, "safe", "call safe").await;
    assert_eq!(
        pending, 0,
        "safe should bypass approval via the callable policy"
    );
    // The policy is consulted once per tool per run: guarded on its run, then both tools again.
    let seen = seen.lock().unwrap().clone();
    assert!(
        seen.contains(&"guarded".to_string()) && seen.contains(&"safe".to_string()),
        "{seen:?}"
    );
}
