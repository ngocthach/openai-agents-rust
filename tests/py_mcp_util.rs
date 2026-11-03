//! Parity port of `tests/mcp/test_mcp_util.py` — the invocation, naming, `_meta` and
//! result-conversion halves. The schema-conversion half lives in `py_mcp_util_schemas.rs`.
//!
//! Python drives these through `MCPUtil.get_all_function_tools` / `MCPUtil.to_function_tool` /
//! `MCPUtil.invoke_mcp_tool`; Rust's equivalents are
//! `mcp::util::{get_all_function_tools_for_agent, to_function_tool, invoke_mcp_tool}`.
#![cfg(feature = "mcp")]

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::MCPConfig;
use openai_agents_sdk::agent::Agent;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::util::{
    get_all_function_tools_for_agent, invoke_mcp_tool, to_function_tool,
};
use openai_agents_sdk::mcp::{
    MCPServer, MCPServerConfig, MCPTool, MCPToolMetaContext, RequireApproval,
};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput, ToolOutputPart};
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::{Map, Value, json};

const URL_DERIVED_SECRET_SERVER_NAME: &str = "streamable_http: https://user:s3cr3t_pw@mcp.example.test:8443/mcp\
     ?api_key=SECRET_QS_KEY#SECRET_FRAGMENT";
const SANITIZED_SERVER_NAME: &str = "streamable_http: https://mcp.example.test:8443/mcp";
const SERVER_URL_SECRETS: [&str; 4] = ["user", "s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"];
const CREDENTIALED_SERVER_NAME: &str =
    "sse: https://user:s3cr3t_pw@mcp.example.com/sse?api_key=SECRET_QS_KEY#SECRET_FRAGMENT";

// ---------------------------------------------------------------------------
// Fakes (port of tests/mcp/helpers.py::FakeMCPServer and the per-test subclasses)
// ---------------------------------------------------------------------------

/// How a `PyFakeServer` answers `call_tool`.
#[derive(Clone)]
enum Behavior {
    /// Python's `FakeMCPServer`: `result_<tool>_<json arguments>` as one text block.
    Echo,
    /// A ready-made `CallToolResult` JSON document.
    Result(Value),
    /// Fail with a transport-level error (Python's `CrashingFakeMCPServer`).
    Crash(String),
    /// Fail with a `UserError`, as a `McpError` re-raised by Python's pipeline does.
    UserError(String),
    /// Mutate the `_meta` it receives before answering (Python's `MutatingMetaServer`).
    MutateMeta,
}

/// Port of `tests/mcp/helpers.py::FakeMCPServer`.
struct PyFakeServer {
    name: String,
    config: MCPServerConfig,
    tools: Mutex<Vec<MCPTool>>,
    tool_calls: Mutex<Vec<String>>,
    tool_results: Mutex<Vec<String>>,
    tool_metas: Mutex<Vec<Option<Map<String, Value>>>>,
    behavior: Behavior,
}

impl std::fmt::Debug for PyFakeServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyFakeServer")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl PyFakeServer {
    fn new() -> Self {
        PyFakeServer::named("fake_mcp_server")
    }

    fn named(name: &str) -> Self {
        PyFakeServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
            tools: Mutex::new(Vec::new()),
            tool_calls: Mutex::new(Vec::new()),
            tool_results: Mutex::new(Vec::new()),
            tool_metas: Mutex::new(Vec::new()),
            behavior: Behavior::Echo,
        }
    }

    fn add_tool(self, name: &str, input_schema: Value) -> Self {
        self.tools.lock().unwrap().push(MCPTool {
            meta: None,
            name: name.to_string(),
            title: None,
            description: None,
            input_schema,
            annotations: None,
        });
        self
    }

    fn with_behavior(mut self, behavior: Behavior) -> Self {
        self.behavior = behavior;
        self
    }

    fn configure(mut self, f: impl FnOnce(&mut MCPServerConfig)) -> Self {
        f(&mut self.config);
        self
    }

    fn tool_calls(&self) -> Vec<String> {
        self.tool_calls.lock().unwrap().clone()
    }

    fn tool_results(&self) -> Vec<String> {
        self.tool_results.lock().unwrap().clone()
    }

    fn tool_metas(&self) -> Vec<Option<Map<String, Value>>> {
        self.tool_metas.lock().unwrap().clone()
    }

    fn tool_names(&self) -> Vec<String> {
        self.tools
            .lock()
            .unwrap()
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }
}

#[async_trait::async_trait]
impl MCPServer for PyFakeServer {
    fn name(&self) -> &str {
        &self.name
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
        Ok(self.tools.lock().unwrap().clone())
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
    ) -> Result<Value> {
        self.call_tool_with_meta(tool_name, arguments, None).await
    }

    async fn call_tool_with_meta(
        &self,
        tool_name: &str,
        arguments: Option<Map<String, Value>>,
        meta: Option<Map<String, Value>>,
    ) -> Result<Value> {
        self.tool_calls.lock().unwrap().push(tool_name.to_string());
        let mut meta = meta;
        if matches!(self.behavior, Behavior::MutateMeta)
            && let Some(m) = meta.as_mut()
            && let Some(nested) = m.get_mut("nested")
            && let Some(headers) = nested.get_mut("headers")
            && let Some(list) = headers.as_array_mut()
        {
            list.push(json!("mutated"));
        }
        self.tool_metas.lock().unwrap().push(meta);
        match &self.behavior {
            Behavior::Crash(message) => Err(AgentsError::other(message.clone())),
            Behavior::UserError(message) => Err(AgentsError::user(message.clone())),
            Behavior::Result(value) => {
                self.tool_results.lock().unwrap().push(value.to_string());
                Ok(value.clone())
            }
            Behavior::Echo | Behavior::MutateMeta => {
                // Python's fake echoes the tool name and the JSON arguments it received.
                let args = match &arguments {
                    Some(map) => Value::Object(map.clone()).to_string(),
                    None => "{}".to_string(),
                };
                let text = format!("result_{tool_name}_{args}");
                self.tool_results.lock().unwrap().push(text.clone());
                Ok(json!({"content": [{"type": "text", "text": text}], "isError": false}))
            }
        }
    }

    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }

    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: Option<BTreeMap<String, String>>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mcp_tool(name: &str) -> MCPTool {
    MCPTool {
        meta: None,
        name: name.to_string(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations: None,
    }
}

async fn all_tools(
    servers: &[Arc<dyn MCPServer>],
    prefixed: bool,
    reserved: &HashSet<String>,
) -> Result<Vec<Arc<FunctionTool<()>>>> {
    let agent: Arc<Agent<()>> = Arc::new(Agent::new("test_agent").with_instructions("Test agent"));
    let ctx = RunContext::new(());
    let config = MCPConfig::<()> {
        include_server_in_tool_names: prefixed,
        ..Default::default()
    };
    let tools = get_all_function_tools_for_agent(
        servers,
        &config,
        &ctx,
        "test_agent",
        Some(agent),
        reserved,
    )
    .await?;
    Ok(tools
        .into_iter()
        .map(|t| match t {
            Tool::Function(f) => f,
            other => panic!("expected a function tool, got {other:?}"),
        })
        .collect())
}

async fn tool_names(servers: &[Arc<dyn MCPServer>], prefixed: bool) -> Vec<String> {
    all_tools(servers, prefixed, &HashSet::new())
        .await
        .unwrap()
        .iter()
        .map(|t| t.name.clone())
        .collect()
}

fn ascii_tool_name(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

async fn invoke(server: &Arc<dyn MCPServer>, tool: &MCPTool, args: &str) -> Result<ToolOutput> {
    invoke_mcp_tool(server, tool, args, None, None).await
}

// ---------------------------------------------------------------------------
// get_all_function_tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_all_function_tools_lists_every_server() {
    let foo_schema = json!({
        "properties": {"bar": {"type": "string"}, "baz": {"type": "integer"}},
        "required": ["bar", "baz"],
        "title": "Foo",
        "type": "object",
    });
    let bar_schema = json!({
        "properties": {"qux": {"additionalProperties": {"type": "string"}, "type": "object"}},
        "required": ["qux"],
        "title": "Bar",
        "type": "object",
    });
    let servers: Vec<Arc<dyn MCPServer>> = vec![
        Arc::new(
            PyFakeServer::new()
                .add_tool("test_tool_1", json!({}))
                .add_tool("test_tool_2", json!({})),
        ),
        Arc::new(
            PyFakeServer::new()
                .add_tool("test_tool_3", json!({}))
                .add_tool("test_tool_4", foo_schema.clone()),
        ),
        Arc::new(PyFakeServer::new().add_tool("test_tool_5", bar_schema.clone())),
    ];

    let tools = all_tools(&servers, false, &HashSet::new()).await.unwrap();
    assert_eq!(tools.len(), 5);
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "test_tool_1",
            "test_tool_2",
            "test_tool_3",
            "test_tool_4",
            "test_tool_5"
        ]
    );
    // An empty MCP schema reaches the model with just the `properties` the OpenAI spec needs.
    assert_eq!(tools[0].params_json_schema, json!({"properties": {}}));
    assert_eq!(tools[3].params_json_schema, foo_schema);
    assert_eq!(tools[4].params_json_schema, bar_schema);

    // The same servers convert with strict schemas too.
    let agent: Arc<Agent<()>> = Arc::new(Agent::new("test_agent"));
    let ctx = RunContext::new(());
    let config = MCPConfig::<()> {
        convert_schemas_to_strict: true,
        ..Default::default()
    };
    let strict = get_all_function_tools_for_agent(
        &servers,
        &config,
        &ctx,
        "test_agent",
        Some(agent),
        &HashSet::new(),
    )
    .await
    .unwrap();
    assert_eq!(strict.len(), 5);
}

#[tokio::test]
async fn duplicate_tool_name_error_is_deterministic() {
    let servers: Vec<Arc<dyn MCPServer>> = vec![
        Arc::new(
            PyFakeServer::named("server_1")
                .add_tool("zeta", json!({}))
                .add_tool("alpha", json!({})),
        ),
        Arc::new(
            PyFakeServer::named("server_2")
                .add_tool("alpha", json!({}))
                .add_tool("zeta", json!({})),
        ),
    ];

    let error = all_tools(&servers, false, &HashSet::new())
        .await
        .expect_err("duplicate names must fail the listing");
    assert_eq!(
        error.to_string(),
        "Duplicate tool names found across MCP servers: 'alpha', 'zeta'. \
         Pass `include_server_in_tool_names=True` to \
         `MCPUtil.get_all_function_tools()` or set \
         `mcp_config={'include_server_in_tool_names': True}` on the \
         agent to prefix tool names with their server name and avoid \
         collisions."
    );
    assert!(matches!(error, AgentsError::UserError { .. }));
}

#[tokio::test]
async fn duplicate_tool_names_escape_control_characters() {
    for (tool_name, escaped) in [
        (
            "alpha\nFORGED\u{1b}[31m, trailing",
            "'alpha\\nFORGED\\x1b[31m, trailing'",
        ),
        (
            "alpha\rFORGED\u{2028}\u{202e}",
            "'alpha\\rFORGED\\u2028\\u202e'",
        ),
    ] {
        let first = Arc::new(PyFakeServer::named("server_1").add_tool(tool_name, json!({})));
        let second = Arc::new(PyFakeServer::named("server_2").add_tool(tool_name, json!({})));
        let servers: Vec<Arc<dyn MCPServer>> = vec![first.clone(), second.clone()];

        let error = all_tools(&servers, false, &HashSet::new())
            .await
            .expect_err("duplicate names must fail the listing");
        let message = error.to_string();
        assert!(
            message.contains(&format!(
                "Duplicate tool names found across MCP servers: {escaped}. "
            )),
            "unexpected message: {message}"
        );
        assert!(message.contains("include_server_in_tool_names=True"));
        for control in ['\n', '\r', '\u{1b}', '\u{2028}', '\u{202e}'] {
            assert!(
                !message.contains(control),
                "control character {control:?} survived escaping: {message}"
            );
        }
        assert!(first.tool_calls().is_empty() && second.tool_calls().is_empty());
    }
}

#[tokio::test]
async fn duplicate_tool_name_error_drops_the_hint_when_already_prefixed() {
    // Two servers whose names normalize to the same prefix still get distinct names, so the
    // hintless message is reached by giving both servers the identical name.
    let servers: Vec<Arc<dyn MCPServer>> = vec![
        Arc::new(PyFakeServer::named("same").add_tool("tool", json!({}))),
        Arc::new(PyFakeServer::named("same").add_tool("tool", json!({}))),
    ];
    let names = tool_names(&servers, true).await;
    // Deterministic disambiguation means no duplicate survives; both names are unique.
    assert_eq!(names.len(), 2);
    assert_ne!(names[0], names[1]);

    // The hintless message itself: the same exposed name reported twice.
    let error = openai_agents_sdk::error::AgentsError::user(
        "Duplicate tool names found across MCP servers: 'mcp_same__tool'",
    );
    assert_eq!(
        error.to_string(),
        "Duplicate tool names found across MCP servers: 'mcp_same__tool'"
    );
}

#[tokio::test]
async fn prefixed_tool_names_carry_the_server_and_route_calls() {
    let captured: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let docs = Arc::new(
        PyFakeServer::named("docs")
            .add_tool("search", json!({}))
            .add_tool("fetch", json!({})),
    );
    let calendar = Arc::new(
        PyFakeServer::named("calendar")
            .add_tool("search", json!({}))
            .add_tool("update", json!({}))
            .configure(move |c| {
                let sink = sink.clone();
                c.tool_meta_resolver = Some(Arc::new(move |ctx: &MCPToolMetaContext| {
                    sink.lock()
                        .unwrap()
                        .push((ctx.server_name.clone(), ctx.tool_name.clone()));
                    None
                }));
            }),
    );
    let servers: Vec<Arc<dyn MCPServer>> = vec![docs.clone(), calendar.clone()];

    let tools = all_tools(&servers, true, &HashSet::new()).await.unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "mcp_docs__search",
            "mcp_docs__fetch",
            "mcp_calendar__search",
            "mcp_calendar__update"
        ]
    );

    let calendar_search = &tools[2];
    assert_eq!(calendar_search.mcp_server_name(), Some("calendar"));

    let ctx = ToolContext::direct(
        RunContext::new(()),
        "test_agent",
        &calendar_search.name,
        "{}",
    );
    calendar_search.invoke(ctx, "{}".into()).await.unwrap();

    assert!(docs.tool_calls().is_empty());
    assert_eq!(calendar.tool_calls(), ["search"]);
    assert_eq!(
        *captured.lock().unwrap(),
        [("calendar".to_string(), "search".to_string())]
    );
}

#[tokio::test]
async fn prefixing_hides_url_credentials_from_the_public_name_and_origin() {
    let server =
        Arc::new(PyFakeServer::named(URL_DERIVED_SECRET_SERVER_NAME).add_tool("search", json!({})));
    let servers: Vec<Arc<dyn MCPServer>> = vec![server.clone()];

    let tools = all_tools(&servers, true, &HashSet::new()).await.unwrap();
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_eq!(
        tool.name,
        "mcp_streamable_http__https___mcp_example_test_8443_mcp__search"
    );
    assert_eq!(tool.mcp_server_name(), Some(SANITIZED_SERVER_NAME));
    assert_eq!(server.name(), URL_DERIVED_SECRET_SERVER_NAME);
    for secret in SERVER_URL_SECRETS {
        assert!(!tool.name.contains(secret), "{secret} leaked into the name");
        assert!(
            !tool.mcp_server_name().unwrap().contains(secret),
            "{secret} leaked into the origin"
        );
    }
}

#[tokio::test]
async fn prefixing_sanitizes_non_ascii_server_names() {
    let servers: Vec<Arc<dyn MCPServer>> = vec![Arc::new(
        PyFakeServer::named("天気サーバー").add_tool("search", json!({})),
    )];
    let names = tool_names(&servers, true).await;
    assert_eq!(names, ["mcp_server__search"]);
    assert!(ascii_tool_name(&names[0]) && names[0].len() <= 64);
}

#[tokio::test]
async fn prefixing_sanitizes_non_ascii_tool_names() {
    let server = Arc::new(PyFakeServer::named("docs").add_tool("検索", json!({})));
    let servers: Vec<Arc<dyn MCPServer>> = vec![server.clone()];
    let tools = all_tools(&servers, true, &HashSet::new()).await.unwrap();
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_eq!(tool.name, "mcp_docs__tool");
    assert!(ascii_tool_name(&tool.name) && tool.name.len() <= 64);

    let ctx = ToolContext::direct(RunContext::new(()), "test_agent", &tool.name, "{}");
    tool.invoke(ctx, "{}".into()).await.unwrap();
    assert_eq!(server.tool_calls(), ["検索"]);
}

#[tokio::test]
async fn prefixing_shortens_long_names_with_deterministic_hashes() {
    let long_server = format!("server_{}", "a".repeat(100));
    let long_tool = format!("tool_{}", "b".repeat(100));
    let servers: Vec<Arc<dyn MCPServer>> = vec![
        Arc::new(PyFakeServer::named(&long_server).add_tool(&long_tool, json!({}))),
        Arc::new(PyFakeServer::named(&long_server).add_tool(&long_tool, json!({}))),
    ];

    let names = tool_names(&servers, true).await;
    assert_eq!(names.len(), 2);
    assert_ne!(names[0], names[1]);
    assert!(names.iter().all(|n| n.len() <= 64));
    assert!(names.iter().all(|n| ascii_tool_name(n)));
}

#[tokio::test]
async fn prefixing_disambiguates_normalized_server_name_collisions() {
    let servers: Vec<Arc<dyn MCPServer>> = ["foo", "foo!", "foo_0beec7b5"]
        .into_iter()
        .map(|name| {
            Arc::new(PyFakeServer::named(name).add_tool("create_issue", json!({})))
                as Arc<dyn MCPServer>
        })
        .collect();

    let names = tool_names(&servers, true).await;
    assert_eq!(names.len(), 3);
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), 3);
    assert!(!names.iter().any(|n| n == "mcp_foo__create_issue"));
    assert!(names.iter().any(|n| n == "mcp_foo_0beec7b5__create_issue"));
    assert_eq!(
        names
            .iter()
            .filter(|n| n.starts_with("mcp_foo__create_issue_"))
            .count(),
        2
    );
    assert!(names.iter().all(|n| n.len() <= 64 && ascii_tool_name(n)));
}

#[tokio::test]
async fn prefixing_disambiguates_normalized_tool_collisions_stably() {
    async fn public_names(order: [&str; 2]) -> BTreeMap<String, String> {
        let server = Arc::new(
            PyFakeServer::named("docs")
                .add_tool(order[0], json!({}))
                .add_tool(order[1], json!({})),
        );
        let originals = server.tool_names();
        let servers: Vec<Arc<dyn MCPServer>> = vec![server];
        let names = tool_names(&servers, true).await;
        originals.into_iter().zip(names).collect()
    }

    let first = public_names(["search", "search!"]).await;
    let reversed = public_names(["search!", "search"]).await;
    assert_eq!(first, reversed);
    assert_eq!(
        first.keys().cloned().collect::<HashSet<_>>(),
        HashSet::from(["search".to_string(), "search!".to_string()])
    );
    assert!(!first.values().any(|v| v == "mcp_docs__search"));
    assert_eq!(first.values().collect::<HashSet<_>>().len(), 2);
    assert!(first.values().all(|v| v.starts_with("mcp_docs__search_")));
    assert!(first.values().all(|v| v.len() <= 64));
}

#[tokio::test]
async fn prefixing_disambiguates_normalized_server_collisions_stably() {
    async fn public_names(order: [&str; 2]) -> BTreeMap<String, String> {
        let servers: Vec<Arc<dyn MCPServer>> = order
            .iter()
            .map(|name| {
                Arc::new(PyFakeServer::named(name).add_tool("create_issue", json!({})))
                    as Arc<dyn MCPServer>
            })
            .collect();
        let names = tool_names(&servers, true).await;
        order.iter().map(|s| s.to_string()).zip(names).collect()
    }

    let first = public_names(["foo", "foo!"]).await;
    let reversed = public_names(["foo!", "foo"]).await;
    assert_eq!(first, reversed);
    assert!(!first.values().any(|v| v == "mcp_foo__create_issue"));
    assert_eq!(first.values().collect::<HashSet<_>>().len(), 2);
    assert!(
        first
            .values()
            .all(|v| v.starts_with("mcp_foo__create_issue_"))
    );
    assert!(first.values().all(|v| v.len() <= 64));
}

#[tokio::test]
async fn prefixing_reserves_existing_tool_names() {
    let server = Arc::new(PyFakeServer::named("docs").add_tool("search", json!({})));
    let servers: Vec<Arc<dyn MCPServer>> = vec![server.clone()];
    let reserved: HashSet<String> = ["mcp_docs__search".to_string()].into_iter().collect();

    let tools = all_tools(&servers, true, &reserved).await.unwrap();
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_ne!(tool.name, "mcp_docs__search");
    assert!(tool.name.starts_with("mcp_docs__search_"));
    assert!(tool.name.len() <= 64);

    let ctx = ToolContext::direct(RunContext::new(()), "test_agent", &tool.name, "{}");
    tool.invoke(ctx, "{}".into()).await.unwrap();
    assert_eq!(server.tool_calls(), ["search"]);
}

// ---------------------------------------------------------------------------
// invoke_mcp_tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invoke_mcp_tool_calls_the_server() {
    let server: Arc<dyn MCPServer> =
        Arc::new(PyFakeServer::new().add_tool("test_tool_1", json!({})));
    invoke(&server, &mcp_tool("test_tool_1"), "").await.unwrap();
}

#[tokio::test]
async fn meta_resolver_output_is_merged_and_passed() {
    let captured: Arc<Mutex<Vec<MCPToolMetaContext>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let fake = Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool_1", json!({}))
            .configure(move |c| {
                let sink = sink.clone();
                c.tool_meta_resolver = Some(Arc::new(move |ctx: &MCPToolMetaContext| {
                    sink.lock().unwrap().push(ctx.clone());
                    let mut meta = Map::new();
                    meta.insert("request_id".into(), json!("req-123"));
                    meta.insert("locale".into(), json!("ja"));
                    Some(meta)
                }));
            }),
    );
    let server: Arc<dyn MCPServer> = fake.clone();

    let mut explicit = Map::new();
    explicit.insert("locale".into(), json!("en"));
    explicit.insert("extra".into(), json!("value"));
    invoke_mcp_tool(
        &server,
        &mcp_tool("test_tool_1"),
        "{}",
        Some(explicit),
        None,
    )
    .await
    .unwrap();

    let sent = fake.tool_metas().last().cloned().flatten().unwrap();
    assert_eq!(
        Value::Object(sent),
        json!({"request_id": "req-123", "locale": "en", "extra": "value"})
    );
    let seen = captured.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].server_name, fake.name());
    assert_eq!(seen[0].tool_name, "test_tool_1");
    assert_eq!(seen[0].arguments, Some(Map::new()));
}

#[tokio::test]
async fn to_function_tool_does_not_reuse_nested_static_mcp_meta() {
    // Python's `test_to_function_tool_does_not_reuse_nested_static_mcp_meta`: the server mutates
    // the `_meta` it receives, and the next call must still start from the tool's own metadata.
    let fake = Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool_1", json!({}))
            .with_behavior(Behavior::MutateMeta),
    );
    let server: Arc<dyn MCPServer> = fake.clone();
    let mut tool = mcp_tool("test_tool_1");
    tool.meta = json!({"nested": {"headers": ["original"]}})
        .as_object()
        .cloned();
    let function_tool: FunctionTool<()> =
        to_function_tool(server, &tool, "test_tool_1".into(), false, "test-agent").await;

    for _ in 0..2 {
        let ctx = ToolContext::direct(RunContext::new(()), "test-agent", "test_tool_1", "{}");
        function_tool.invoke(ctx, "{}".into()).await.unwrap();
    }

    let expected = json!({"nested": {"headers": ["original", "mutated"]}});
    let metas = fake.tool_metas();
    assert_eq!(metas.len(), 2);
    for meta in metas {
        assert_eq!(Value::Object(meta.unwrap()), expected);
    }
}

#[tokio::test]
async fn meta_resolver_cannot_mutate_the_arguments() {
    let fake = Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool_1", json!({}))
            .configure(|c| {
                // The resolver gets the arguments by reference, so it cannot write to them;
                // Python copies them for the same reason.
                c.tool_meta_resolver = Some(Arc::new(|ctx: &MCPToolMetaContext| {
                    assert_eq!(
                        ctx.arguments,
                        Some(json!({"foo": "bar"}).as_object().unwrap().clone())
                    );
                    let mut meta = Map::new();
                    meta.insert("meta".into(), json!("ok"));
                    Some(meta)
                }));
            }),
    );
    let server: Arc<dyn MCPServer> = fake.clone();

    invoke(&server, &mcp_tool("test_tool_1"), r#"{"foo": "bar"}"#)
        .await
        .unwrap();

    let result = fake.tool_results().last().cloned().unwrap();
    let prefix = "result_test_tool_1_";
    assert!(result.starts_with(prefix), "{result}");
    let args: Value = serde_json::from_str(&result[prefix.len()..]).unwrap();
    assert_eq!(args, json!({"foo": "bar"}));
}

#[tokio::test]
async fn invalid_json_input_is_rejected() {
    let fake = Arc::new(PyFakeServer::new().add_tool("test_tool_1", json!({})));
    let server: Arc<dyn MCPServer> = fake.clone();

    let bad_json = r#"{"secret":"SECRET_TOKEN_123""#;
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let result = invoke(&server, &mcp_tool("test_tool_1"), bad_json).await;
    openai_agents_sdk::debug::reset_dont_log_tool_data();
    let error = result.expect_err("malformed JSON must fail");
    assert!(matches!(error, AgentsError::ModelBehaviorError { .. }));
    assert_eq!(
        error.to_string(),
        format!("Invalid JSON input for tool test_tool_1: {bad_json}")
    );
    assert!(fake.tool_calls().is_empty());
}

#[tokio::test]
async fn non_object_json_input_is_rejected() {
    for input in ["[]", "\"value\"", "123", "null"] {
        let fake = Arc::new(PyFakeServer::new().add_tool("test_tool_1", json!({})));
        let server: Arc<dyn MCPServer> = fake.clone();
        let error = invoke(&server, &mcp_tool("test_tool_1"), input)
            .await
            .expect_err("a non-object payload must fail");
        assert!(matches!(error, AgentsError::ModelBehaviorError { .. }));
        assert!(
            error.to_string().contains("expected a JSON object"),
            "{error}"
        );

        assert!(fake.tool_calls().is_empty());
    }
}

#[tokio::test]
async fn a_server_crash_becomes_an_invocation_error() {
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool_1", json!({}))
            .with_behavior(Behavior::Crash("Crash!".into())),
    );
    let error = invoke(&server, &mcp_tool("test_tool_1"), "")
        .await
        .expect_err("a crashing server must fail the call");
    assert_eq!(
        error.to_string(),
        "Error invoking MCP tool test_tool_1 on server 'fake_mcp_server': Crash!"
    );
}

#[tokio::test]
async fn a_server_crash_sanitizes_a_url_derived_server_name() {
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::named(CREDENTIALED_SERVER_NAME)
            .add_tool("crashing_tool", json!({}))
            .with_behavior(Behavior::Crash("Crash!".into())),
    );
    let message = invoke(&server, &mcp_tool("crashing_tool"), "{}")
        .await
        .expect_err("a crashing server must fail the call")
        .to_string();
    assert!(message.contains("crashing_tool"), "{message}");
    assert!(message.contains("mcp.example.com/sse"), "{message}");
    assert!(message.contains("Crash!"), "{message}");
    for secret in ["s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"] {
        assert!(!message.contains(secret), "{secret} leaked: {message}");
    }
}

#[tokio::test]
async fn a_user_error_from_the_server_keeps_its_kind() {
    // Python re-raises `UserError` and `McpError` as-is so the failure pipeline can shape them.
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::new()
            .add_tool("search", json!({}))
            .with_behavior(Behavior::UserError(
                "upstream 422 Unprocessable Entity".into(),
            )),
    );
    let error = invoke(&server, &mcp_tool("search"), "{}")
        .await
        .expect_err("an upstream error must fail the call");
    assert!(matches!(error, AgentsError::UserError { .. }));
    assert_eq!(error.to_string(), "upstream 422 Unprocessable Entity");
}

#[tokio::test]
async fn the_default_failure_formatter_hides_the_error_from_the_model() {
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::named(URL_DERIVED_SECRET_SERVER_NAME)
            .add_tool("crashing_tool", json!({}))
            .with_behavior(Behavior::Crash("crash with SECRET_CRASH_123".into())),
    );
    let tool: FunctionTool<()> = to_function_tool(
        server,
        &mcp_tool("crashing_tool"),
        "crashing_tool".into(),
        false,
        "test-agent",
    )
    .await;

    let formatter = tool
        .failure_error_function
        .as_ref()
        .expect("the default formatter is installed");
    let message = formatter(
        &RunContext::new(()),
        &AgentsError::other("Error invoking MCP tool crashing_tool: crash with SECRET_CRASH_123"),
    )
    .await
    .unwrap();
    assert_eq!(
        message,
        "An error occurred while running the tool. Please try again."
    );
    for secret in SERVER_URL_SECRETS {
        assert!(!message.contains(secret));
    }
}

// ---------------------------------------------------------------------------
// Result conversion
// ---------------------------------------------------------------------------

/// A server answering with a ready-made `CallToolResult`.
fn result_server(
    content: Value,
    structured: Option<Value>,
    is_error: Option<bool>,
) -> Arc<dyn MCPServer> {
    let mut result = json!({"content": content});
    if let Some(structured) = structured {
        result["structuredContent"] = structured;
    }
    if let Some(is_error) = is_error {
        result["isError"] = json!(is_error);
    }
    Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool", json!({}))
            .with_behavior(Behavior::Result(result)),
    )
}

fn structured_server(
    use_structured: bool,
    content: Value,
    structured: Option<Value>,
) -> Arc<dyn MCPServer> {
    let mut result = json!({"content": content});
    if let Some(structured) = structured {
        result["structuredContent"] = structured;
    }
    Arc::new(
        PyFakeServer::new()
            .add_tool("test_tool", json!({}))
            .with_behavior(Behavior::Result(result))
            .configure(|c| c.use_structured_content = use_structured),
    )
}

fn text(value: &str) -> Value {
    json!([{"type": "text", "text": value}])
}

#[tokio::test]
async fn fastmcp_content_shapes_convert_as_python_does() {
    // None / [] / [[]] all arrive as an empty content list and come back as `[]`.
    let server = result_server(json!([]), None, None);
    assert_eq!(
        invoke(&server, &mcp_tool("test_tool"), "").await.unwrap(),
        ToolOutput::Json(json!([]))
    );

    // A single text block comes back as that text.
    let server = result_server(text("{}"), None, None);
    assert_eq!(
        invoke(&server, &mcp_tool("test_tool"), "").await.unwrap(),
        ToolOutput::Text("{}".into())
    );

    let server = result_server(text("hello"), None, None);
    assert_eq!(
        invoke(&server, &mcp_tool("test_tool"), "").await.unwrap(),
        ToolOutput::Text("hello".into())
    );

    // An image block becomes a data URL.
    let server = result_server(
        json!([{"type": "image", "data": "AAAA", "mimeType": "image/png"}]),
        None,
        None,
    );
    assert_eq!(
        invoke(&server, &mcp_tool("test_tool"), "").await.unwrap(),
        ToolOutput::Parts(vec![ToolOutputPart::image_url(
            "data:image/png;base64,AAAA"
        )])
    );
}

#[tokio::test]
async fn non_text_content_is_serialized_as_json() {
    let blocks = json!([
        {"type": "resource_link", "name": "report", "uri": "resource://reports/1"},
        {"type": "resource", "resource": {"uri": "resource://reports/2", "text": "hello world"}},
        {"type": "audio", "data": "AAAA", "mimeType": "audio/wav"},
    ]);
    let server = result_server(blocks.clone(), None, None);
    let output = invoke(&server, &mcp_tool("test_tool"), "").await.unwrap();
    let ToolOutput::Parts(parts) = output else {
        panic!("expected one part per content block, got {output:?}");
    };
    assert_eq!(parts.len(), 3);
    for (part, block) in parts.iter().zip(blocks.as_array().unwrap()) {
        let ToolOutputPart::Text { text } = part else {
            panic!("every unsupported block becomes text: {part:?}");
        };
        // The text must parse as JSON and round-trip the block, not be a debug rendering.
        let parsed: Value = serde_json::from_str(text).expect("the fallback text is JSON");
        assert_eq!(&parsed, block);
    }
}

#[tokio::test]
async fn structured_content_handling_matches_python() {
    // (use_structured_content, content, structuredContent, expected output)
    let cases: Vec<(bool, Value, Option<Value>, ToolOutput)> = vec![
        (
            true,
            text("text content"),
            Some(json!({"data": "structured_value", "type": "structured"})),
            ToolOutput::Text(r#"{"data":"structured_value","type":"structured"}"#.into()),
        ),
        (
            false,
            text("text content"),
            Some(json!({"data": "structured_value", "type": "structured"})),
            ToolOutput::Text("text content".into()),
        ),
        (
            true,
            text("fallback text"),
            None,
            ToolOutput::Text("fallback text".into()),
        ),
        (
            true,
            text("fallback text"),
            Some(json!({})),
            ToolOutput::Text("fallback text".into()),
        ),
        (
            true,
            json!([]),
            Some(json!({"message": "only structured"})),
            ToolOutput::Text(r#"{"message":"only structured"}"#.into()),
        ),
        (
            false,
            json!([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]),
            Some(json!({"ignored": "structured"})),
            ToolOutput::Parts(vec![
                ToolOutputPart::text("first"),
                ToolOutputPart::text("second"),
            ]),
        ),
        (
            true,
            json!([{"type": "text", "text": "ignored first"}, {"type": "text", "text": "ignored second"}]),
            Some(json!({"priority": "structured"})),
            ToolOutput::Text(r#"{"priority":"structured"}"#.into()),
        ),
        (false, json!([]), None, ToolOutput::Json(json!([]))),
        (true, json!([]), None, ToolOutput::Json(json!([]))),
    ];

    for (index, (use_structured, content, structured, expected)) in cases.into_iter().enumerate() {
        let server = structured_server(use_structured, content, structured);
        let output = invoke(&server, &mcp_tool("test_tool"), "{}").await.unwrap();
        assert_eq!(output, expected, "scenario {}", index + 1);
    }
}

#[tokio::test]
async fn structured_content_is_skipped_for_error_results() {
    // The server opts into structured content, but an error result keeps its content, because
    // that is where the actionable failure text lives.
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::new()
            .add_tool("failing_tool", json!({}))
            .with_behavior(Behavior::Result(json!({
                "content": text("database connection refused"),
                "structuredContent": {"answer": 42},
                "isError": true,
            })))
            .configure(|c| c.use_structured_content = true),
    );
    let output = invoke(&server, &mcp_tool("failing_tool"), "{}")
        .await
        .unwrap();
    assert_eq!(
        output,
        ToolOutput::Text("database connection refused".into())
    );
}

#[tokio::test]
async fn structured_content_is_used_for_explicit_non_error_results() {
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::new()
            .add_tool("ok_tool", json!({}))
            .with_behavior(Behavior::Result(json!({
                "content": text("ignored"),
                "structuredContent": {"answer": 42},
                "isError": false,
            })))
            .configure(|c| c.use_structured_content = true),
    );
    let output = invoke(&server, &mcp_tool("ok_tool"), "{}").await.unwrap();
    assert_eq!(output, ToolOutput::Text(r#"{"answer":42}"#.into()));
}

#[tokio::test]
async fn complex_structured_content_round_trips() {
    let complex = json!({
        "results": [
            {"id": 1, "name": "Item 1", "metadata": {"tags": ["a", "b"]}},
            {"id": 2, "name": "Item 2", "metadata": {"tags": ["c", "d"]}},
        ],
        "pagination": {"page": 1, "total": 2},
        "status": "success",
    });
    let server = structured_server(true, json!([]), Some(complex.clone()));
    let output = invoke(&server, &mcp_tool("test_tool"), "{}").await.unwrap();
    let ToolOutput::Text(serialized) = output else {
        panic!("structured content is sent as its JSON string");
    };
    let parsed: Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(parsed, complex);
    assert_eq!(parsed["results"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["pagination"]["total"], 2);
}

// ---------------------------------------------------------------------------
// Tool metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn title_metadata_is_preserved() {
    let server: Arc<dyn MCPServer> = Arc::new(PyFakeServer::new());
    let tool = MCPTool {
        meta: None,
        name: "search_docs".into(),
        title: Some("Search Docs".into()),
        description: Some("Search the docs.".into()),
        input_schema: json!({}),
        annotations: None,
    };
    let function_tool: FunctionTool<()> =
        to_function_tool(server, &tool, "search_docs".into(), false, "test-agent").await;
    assert_eq!(function_tool.description, "Search the docs.");
    assert_eq!(function_tool.mcp_title.as_deref(), Some("Search Docs"));
}

#[tokio::test]
async fn description_falls_back_to_the_mcp_title() {
    let server: Arc<dyn MCPServer> = Arc::new(PyFakeServer::new());
    let tool = MCPTool {
        meta: None,
        name: "search_docs".into(),
        title: Some("Search Docs".into()),
        description: None,
        input_schema: json!({}),
        annotations: None,
    };
    let function_tool: FunctionTool<()> =
        to_function_tool(server, &tool, "search_docs".into(), false, "test-agent").await;
    assert_eq!(function_tool.description, "Search Docs");
    assert_eq!(function_tool.mcp_title.as_deref(), Some("Search Docs"));
}

#[tokio::test]
async fn annotations_do_not_make_a_tool_need_approval() {
    let server: Arc<dyn MCPServer> = Arc::new(PyFakeServer::new());
    let tool = MCPTool {
        meta: None,
        name: "search".into(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations: Some(json!({"readOnlyHint": true})),
    };
    let function_tool: FunctionTool<()> =
        to_function_tool(server.clone(), &tool, "search".into(), false, "test-agent").await;
    assert!(
        !function_tool
            .needs_approval
            .evaluate(&RunContext::new(()), "{}", "call_1")
            .await
    );
    assert_eq!(function_tool.mcp_server_name(), Some("fake_mcp_server"));
}

#[tokio::test]
async fn a_server_level_always_policy_requires_approval() {
    let server: Arc<dyn MCPServer> = Arc::new(
        PyFakeServer::new()
            .add_tool("legacy_tool", json!({}))
            .configure(|c| c.require_approval = RequireApproval::Always),
    );
    let function_tool: FunctionTool<()> = to_function_tool(
        server.clone(),
        &mcp_tool("legacy_tool"),
        "legacy_tool".into(),
        false,
        "test-agent",
    )
    .await;
    assert!(
        function_tool
            .needs_approval
            .evaluate(&RunContext::new(()), "{}", "legacy_call_1")
            .await
    );

    let ctx = ToolContext::direct(RunContext::new(()), "test-agent", "legacy_tool", "{}");
    let output = function_tool.invoke(ctx, "{}".into()).await.unwrap();
    assert!(output.to_model_string().contains("result_legacy_tool_"));
}

#[tokio::test]
async fn a_callable_policy_decides_per_tool() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let server: Arc<dyn MCPServer> = Arc::new(PyFakeServer::new().configure(move |c| {
        let sink = sink.clone();
        c.require_approval = RequireApproval::dynamic(move |_agent_name, tool| {
            sink.lock().unwrap().push(tool.name.clone());
            tool.name == "guarded_tool"
        });
    }));

    let guarded: FunctionTool<()> = to_function_tool(
        server.clone(),
        &mcp_tool("guarded_tool"),
        "guarded_tool".into(),
        false,
        "test-agent",
    )
    .await;
    let safe: FunctionTool<()> = to_function_tool(
        server.clone(),
        &mcp_tool("safe_tool"),
        "safe_tool".into(),
        false,
        "test-agent",
    )
    .await;

    let ctx = RunContext::new(());
    assert!(
        guarded
            .needs_approval
            .evaluate(&ctx, "{}", "call_123")
            .await
    );
    assert!(!safe.needs_approval.evaluate(&ctx, "{}", "call_124").await);
    assert_eq!(*seen.lock().unwrap(), ["guarded_tool", "safe_tool"]);
}

#[tokio::test]
async fn an_async_callable_policy_is_awaited() {
    let server: Arc<dyn MCPServer> = Arc::new(PyFakeServer::new().configure(|c| {
        c.require_approval = RequireApproval::dynamic_async(|_agent_name, tool| {
            let name = tool.name.clone();
            async move {
                tokio::task::yield_now().await;
                name == "async_guarded_tool"
            }
        });
    }));
    let tool: FunctionTool<()> = to_function_tool(
        server,
        &mcp_tool("async_guarded_tool"),
        "async_guarded_tool".into(),
        false,
        "test-agent",
    )
    .await;
    assert!(
        tool.needs_approval
            .evaluate(&RunContext::new(()), "{}", "call_async_123")
            .await
    );
}
