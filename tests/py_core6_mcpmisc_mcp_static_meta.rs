//! Parity ports of `test_to_function_tool_passes_static_mcp_meta` and
//! `test_to_function_tool_merges_static_mcp_meta_with_resolver` from `tests/mcp/test_mcp_util.py`:
//! the `_meta` a tool carries is sent with every call, and the resolver's keys never override it.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::RunContext;
use openai_agents_sdk::mcp::util::to_function_tool;
use openai_agents_sdk::mcp::{MCPServer, MCPTool, MCPToolMetaContext};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::{Map, Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

fn static_meta_tool() -> MCPTool {
    MCPTool {
        meta: json!({"locale": "en", "extra": "value"})
            .as_object()
            .cloned(),
        name: "test_tool_1".into(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations: None,
    }
}

async fn invoke(function_tool: &FunctionTool<()>) {
    let ctx = ToolContext::direct(RunContext::new(()), "test-agent", "test_tool_1", "{}");
    function_tool.invoke(ctx, "{}".into()).await.unwrap();
}

#[tokio::test]
async fn to_function_tool_passes_static_mcp_meta() {
    let server = Arc::new(FakeMcpServer::new("fake_mcp_server"));
    let function_tool: FunctionTool<()> = to_function_tool(
        server.clone() as Arc<dyn MCPServer>,
        &static_meta_tool(),
        "test_tool_1".into(),
        false,
        "test-agent",
    )
    .await;
    invoke(&function_tool).await;
    let metas = server.tool_metas();
    assert_eq!(
        metas.last().cloned().flatten().map(Value::Object),
        Some(json!({"locale": "en", "extra": "value"}))
    );
}

#[tokio::test]
async fn to_function_tool_merges_static_mcp_meta_with_resolver() {
    let captured = Arc::new(Mutex::new(Vec::<MCPToolMetaContext>::new()));
    let seen = captured.clone();
    let server = Arc::new(
        FakeMcpServer::new("fake_mcp_server").configure(move |config| {
            config.tool_meta_resolver = Some(Arc::new(move |context: &MCPToolMetaContext| {
                seen.lock().unwrap().push(context.clone());
                let mut meta = Map::new();
                meta.insert("request_id".into(), json!("req-123"));
                meta.insert("locale".into(), json!("ja"));
                Some(meta)
            }));
        }),
    );
    let function_tool: FunctionTool<()> = to_function_tool(
        server.clone() as Arc<dyn MCPServer>,
        &static_meta_tool(),
        "test_tool_1".into(),
        false,
        "test-agent",
    )
    .await;
    invoke(&function_tool).await;
    let metas = server.tool_metas();
    assert_eq!(
        metas.last().cloned().flatten().map(Value::Object),
        Some(json!({"request_id": "req-123", "locale": "en", "extra": "value"}))
    );
    let seen = captured.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].server_name, "fake_mcp_server");
    assert_eq!(seen[0].tool_name, "test_tool_1");
    assert_eq!(seen[0].arguments, Some(Map::new()));
}
