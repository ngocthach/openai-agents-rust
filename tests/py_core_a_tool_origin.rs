//! Port of `test_tool_origin.py`.
//!
//! Python's `test_tool_call_output_item_preserves_positional_type_argument`,
//! `test_local_mcp_tool_origin_does_not_retain_server_object` (weakref / gc) and
//! `test_json_tool_call_does_not_emit_function_tool_origin` (the SDK-internal `json_tool_call`
//! function tool) have no Rust counterpart.
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::{FunctionCall, RunItem, RunItemKind};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{
    Agent, Runner, ToolOrigin, ToolOriginType,
    items::{OutputItem, ToolApprovalItem},
    run_state::RunState,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn first_item(items: &[RunItem], pred: impl Fn(&RunItemKind) -> bool) -> &RunItem {
    items
        .iter()
        .find(|i| pred(&i.kind))
        .expect("expected item of the requested kind")
}

fn is_call(kind: &RunItemKind) -> bool {
    matches!(kind, RunItemKind::ToolCall { .. })
}

fn is_output(kind: &RunItemKind) -> bool {
    matches!(kind, RunItemKind::ToolCallOutput { .. })
}

// ----------------------------------------------------------------- fake MCP server

#[derive(Debug)]
struct FakeServer {
    name: String,
    config: MCPServerConfig,
    tools: Vec<MCPTool>,
    calls: Mutex<Vec<String>>,
}

impl FakeServer {
    fn new(name: &str, tools: Vec<MCPTool>) -> Arc<Self> {
        Arc::new(FakeServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
            tools,
            calls: Mutex::new(Vec::new()),
        })
    }
}

fn mcp_tool(name: &str, title: Option<&str>) -> MCPTool {
    MCPTool {
        meta: None,
        name: name.into(),
        title: title.map(str::to_string),
        description: Some("Search the docs.".into()),
        input_schema: json!({"type": "object", "properties": {}}),
        annotations: None,
    }
}

#[async_trait::async_trait]
impl MCPServer for FakeServer {
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
        Ok(self.tools.clone())
    }
    async fn call_tool(&self, name: &str, _args: Option<Map<String, Value>>) -> Result<Value> {
        self.calls.lock().unwrap().push(name.to_string());
        Ok(json!({"content": [{"type": "text", "text": format!("result_{name}")}]}))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }
    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
}

// ------------------------------------------------------------------ function tools

fn lookup_account() -> FunctionTool<()> {
    FunctionTool::new("lookup_account", "lookup", |_c, _a: NoArgs| async move {
        Ok("account".to_string())
    })
}

#[tokio::test]
async fn runner_attaches_function_tool_origin_to_call_and_output_items() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "lookup_account",
        "{}",
        "call_lookup",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("tool-origin-agent")
        .with_tool(lookup_account())
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "hello").await.unwrap();

    let expected = Some(ToolOrigin::new(ToolOriginType::Function));
    assert_eq!(first_item(&result.new_items, is_call).tool_origin, expected);
    assert_eq!(
        first_item(&result.new_items, is_output).tool_origin,
        expected
    );
}

#[tokio::test]
async fn rejected_function_tool_output_preserves_tool_origin() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = FunctionTool::new(
        "approval_tool",
        "needs approval",
        |_c, _a: NoArgs| async move {
            panic!("The tool should not run when rejected.");
            #[allow(unreachable_code)]
            Ok("unreachable".to_string())
        },
    )
    .with_needs_approval(true);
    let agent = Agent::<()>::new("approval-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "hello").await.unwrap();
    assert!(!first.interruptions.is_empty());
    let interruption = first.interruptions[0].clone();
    assert_eq!(
        interruption.tool_origin,
        Some(ToolOrigin::new(ToolOriginType::Function))
    );

    let mut state = first.to_state();
    state.reject(&interruption, false, None);
    let resumed = Runner::run(&agent, state).await.unwrap();

    assert_eq!(
        first_item(&resumed.new_items, is_output).tool_origin,
        Some(ToolOrigin::new(ToolOriginType::Function))
    );
}

// --------------------------------------------------------------------- local MCP

fn mcp_agent(name: &str, server: Arc<FakeServer>, model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new(name)
        .with_mcp_server(server)
        .with_model_instance(model.clone())
}

#[tokio::test]
async fn runner_attaches_local_mcp_tool_origin_to_call_and_output_items() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "search_docs",
        "{}",
        "call_search_docs",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let server = FakeServer::new(
        "docs_server",
        vec![mcp_tool("search_docs", Some("Search Docs"))],
    );
    let agent = mcp_agent("mcp-agent", server, &model);

    let result = Runner::run(&agent, "hello").await.unwrap();

    let expected = Some(ToolOrigin::mcp("docs_server"));
    let call = first_item(&result.new_items, is_call);
    assert_eq!(call.tool_origin, expected);
    assert_eq!(call.title.as_deref(), Some("Search Docs"));
    assert_eq!(call.description.as_deref(), Some("Search the docs."));
    assert_eq!(
        first_item(&result.new_items, is_output).tool_origin,
        expected
    );
}

#[tokio::test]
async fn local_mcp_tool_origin_hides_url_credentials_in_run_state() {
    let raw_server_name = "streamable_http: https://user:s3cr3t_pw@mcp.example.test:8443/mcp\
                           ?api_key=SECRET_QS_KEY#SECRET_FRAGMENT";
    let safe_server_name = "streamable_http: https://mcp.example.test:8443/mcp";
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "search_docs",
        "{}",
        "call_search_docs",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let server = FakeServer::new(raw_server_name, vec![mcp_tool("search_docs", None)]);
    let agent = mcp_agent("mcp-agent", server.clone(), &model);

    let result = Runner::run(&agent, "hello").await.unwrap();

    let expected = Some(ToolOrigin::mcp(safe_server_name));
    assert_eq!(first_item(&result.new_items, is_call).tool_origin, expected);
    assert_eq!(
        first_item(&result.new_items, is_output).tool_origin,
        expected
    );
    let serialized = result.to_state().to_json().unwrap().to_string();
    assert!(serialized.contains(safe_server_name));
    for secret in ["user:", "s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"] {
        assert!(
            !serialized.contains(secret),
            "leaked {secret}: {serialized}"
        );
    }
    assert_eq!(server.name(), raw_server_name);
}

#[tokio::test]
async fn streamed_tool_call_item_includes_local_mcp_origin() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "search_docs",
        "{}",
        "call_stream_search",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let server = FakeServer::new(
        "docs_server",
        vec![mcp_tool("search_docs", Some("Search Docs"))],
    );
    let agent = mcp_agent("stream-mcp-agent", server, &model);

    let mut result = Runner::run_streamed(&agent, "hello");
    let mut seen: Option<RunItem> = None;
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { item, .. } = event.unwrap()
                && is_call(&item.kind)
                && seen.is_none()
            {
                seen = Some(item);
            }
        }
    }
    let seen = seen.expect("a tool call item was streamed");
    assert_eq!(seen.tool_origin, Some(ToolOrigin::mcp("docs_server")));
    assert_eq!(seen.title.as_deref(), Some("Search Docs"));
}

// -------------------------------------------------------------------- hosted MCP

fn hosted_tool() -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp",
        "server_label": "docs_server",
        "server_url": "https://example.com/mcp",
    }))
}

fn hosted_outputs() -> Vec<OutputItem> {
    vec![
        OutputItem::Other(json!({
            "type": "mcp_list_tools",
            "id": "list_docs_server",
            "server_label": "docs_server",
            "tools": [{
                "name": "search_docs",
                "input_schema": {},
                "description": "Search the docs.",
                "annotations": {"title": "Search Docs"},
            }],
        })),
        OutputItem::Other(json!({
            "type": "mcp_call",
            "id": "mcp_call_1",
            "arguments": "{}",
            "name": "search_docs",
            "server_label": "docs_server",
            "status": "completed",
        })),
    ]
}

#[tokio::test]
async fn hosted_mcp_call_carries_the_mcp_origin() {
    let model = setup();
    model.add_turn(hosted_outputs());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("hosted-mcp")
        .with_tool(hosted_tool())
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "hello").await.unwrap();

    let call = first_item(&result.new_items, |k| match k {
        RunItemKind::ToolCall { raw_item } => {
            raw_item.get("type").and_then(Value::as_str) == Some("mcp_call")
        }
        _ => false,
    });
    assert_eq!(call.tool_origin, Some(ToolOrigin::mcp("docs_server")));
}

#[tokio::test]
async fn streamed_tool_call_item_includes_hosted_mcp_origin() {
    let model = setup();
    model.add_turn(hosted_outputs());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("stream-hosted-mcp")
        .with_tool(hosted_tool())
        .with_model_instance(model.clone());

    let mut result = Runner::run_streamed(&agent, "hello");
    let mut seen: Option<RunItem> = None;
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { item, .. } = event.unwrap()
                && let RunItemKind::ToolCall { raw_item } = &item.kind
                && raw_item.get("type").and_then(Value::as_str) == Some("mcp_call")
                && seen.is_none()
            {
                seen = Some(item.clone());
            }
        }
    }
    let seen = seen.expect("the hosted MCP call was streamed");
    assert_eq!(seen.tool_origin, Some(ToolOrigin::mcp("docs_server")));
}

// -------------------------------------------------------------- run-state round trips

fn function_call_with(name: &str, call_id: &str) -> FunctionCall {
    let mut call = FunctionCall::new(name, "{}");
    call.call_id = call_id.to_string();
    call
}

fn tool_call_item(agent: &str, call_id: &str, name: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::ToolCall {
            raw_item: function_call_with(name, call_id).to_value(),
        },
    )
}

fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

#[test]
fn run_state_roundtrip_preserves_distinct_agent_tool_names() {
    let worker_a = Agent::<()>::new("worker");
    let worker_b = Agent::<()>::new("worker");
    let tool_a = worker_a.as_tool("worker_lookup_a", "Worker A");
    let tool_b = worker_b.as_tool("worker_lookup_b", "Worker B");

    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "outer".into(),
        ..Default::default()
    };
    state.generated_items = vec![
        tool_call_item("outer", "call_worker_a", &tool_a.name)
            .with_tool_details(Some(tool_a.description.clone()), None)
            .with_tool_origin(Some(tool_a.tool_origin())),
        tool_call_item("outer", "call_worker_b", &tool_b.name)
            .with_tool_details(Some(tool_b.description.clone()), None)
            .with_tool_origin(Some(tool_b.tool_origin())),
    ];

    let restored = round_trip(&state);
    let origins: Vec<Option<ToolOrigin>> = restored
        .generated_items
        .iter()
        .map(|i| i.tool_origin.clone())
        .collect();
    assert_eq!(
        origins,
        vec![
            Some(ToolOrigin {
                r#type: ToolOriginType::AgentAsTool,
                mcp_server_name: None,
                agent_name: Some("worker".into()),
                agent_tool_name: Some("worker_lookup_a".into()),
            }),
            Some(ToolOrigin {
                r#type: ToolOriginType::AgentAsTool,
                mcp_server_name: None,
                agent_name: Some("worker".into()),
                agent_tool_name: Some("worker_lookup_b".into()),
            }),
        ]
    );
}

#[test]
fn run_state_from_json_reads_legacy_snapshot_without_tool_origin() {
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "legacy".into(),
        ..Default::default()
    };
    state.generated_items = vec![
        tool_call_item("legacy", "call_legacy", "legacy_tool")
            .with_tool_details(Some("Legacy tool".into()), None)
            .with_tool_origin(Some(ToolOrigin::new(ToolOriginType::Function))),
    ];

    let mut json = state.to_json().unwrap();
    json["$schemaVersion"] = json!("1.5");
    for item in json["generated_items"].as_array_mut().unwrap() {
        item.as_object_mut().unwrap().remove("tool_origin");
    }
    let restored = RunState::from_json(json).unwrap();

    let item = first_item(&restored.generated_items, is_call);
    assert_eq!(item.description.as_deref(), Some("Legacy tool"));
    assert_eq!(item.tool_origin, None);
}

fn approval(origin: Option<ToolOrigin>) -> ToolApprovalItem {
    ToolApprovalItem {
        agent: "approval-origin".into(),
        raw_item: function_call_with("approval_tool", "call_approval"),
        tool_name: "approval_tool".into(),
        tool_origin: origin,
    }
}

#[test]
fn run_state_roundtrip_preserves_tool_origin_on_approval_interruptions() {
    let item = approval(Some(ToolOrigin::new(ToolOriginType::Function)));
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "approval-origin".into(),
        ..Default::default()
    };
    state.interruptions = vec![item.clone()];
    state.pending_tool_processing = true;

    let restored = round_trip(&state);
    assert_eq!(
        restored.get_interruptions()[0].tool_origin,
        Some(ToolOrigin::new(ToolOriginType::Function))
    );
}

#[test]
fn run_state_from_json_reads_legacy_approval_without_tool_origin() {
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "approval-origin-legacy".into(),
        ..Default::default()
    };
    state.interruptions = vec![approval(Some(ToolOrigin::new(ToolOriginType::Function)))];
    state.pending_tool_processing = true;

    let mut json = state.to_json().unwrap();
    json["$schemaVersion"] = json!("1.6");
    fn strip(value: &mut Value) {
        match value {
            Value::Object(m) => {
                m.remove("tool_origin");
                m.values_mut().for_each(strip);
            }
            Value::Array(a) => a.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut json);

    let restored = RunState::from_json(json).unwrap();
    assert_eq!(restored.get_interruptions()[0].tool_origin, None);
}

#[test]
fn malformed_or_unknown_tool_origin_is_dropped_on_read() {
    assert_eq!(ToolOrigin::from_json_dict(&json!("nope")), None);
    assert_eq!(ToolOrigin::from_json_dict(&json!({"type": 3})), None);
    assert_eq!(
        ToolOrigin::from_json_dict(&json!({"type": "future_kind"})),
        None
    );
    assert_eq!(
        ToolOrigin::from_json_dict(&json!({"type": "mcp", "mcp_server_name": "s", "extra": 1})),
        Some(ToolOrigin::mcp("s"))
    );
    assert_eq!(
        ToolOrigin::mcp("s").to_json_dict(),
        json!({"type": "mcp", "mcp_server_name": "s"})
    );
    // A bare function origin carries only its type.
    assert_eq!(
        ToolOrigin::new(ToolOriginType::Function).to_json_dict(),
        json!({"type": "function"})
    );
}
