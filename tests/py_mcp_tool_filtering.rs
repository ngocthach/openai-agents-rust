//! Parity port of `tests/mcp/test_tool_filtering.py`.
//!
//! Python applies the filter inside `MCPServer.list_tools`; Rust keeps `MCPServer::list_tools`
//! unfiltered and applies the filter in `mcp::util::get_all_function_tools_for_agent`, so the
//! ported tests list through the agent/util entry point.
#![cfg(feature = "mcp")]

use std::sync::Arc;

use openai_agents_sdk::MCPConfig;
use openai_agents_sdk::agent::Agent;
use openai_agents_sdk::mcp::util::get_all_function_tools_for_agent;
use openai_agents_sdk::mcp::{MCPServer, ToolFilter, ToolFilterContext, create_static_tool_filter};
use openai_agents_sdk::run_context::RunContext;

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

fn server_with(name: &str, tools: &[&str], filter: ToolFilter) -> Arc<dyn MCPServer> {
    let mut server = FakeMcpServer::new(name).configure(|c| c.tool_filter = filter);
    for tool in tools {
        server = server.add_tool(tool, serde_json::json!({}));
    }
    Arc::new(server)
}

async fn filtered_names(server: Arc<dyn MCPServer>, agent_name: &str) -> Vec<String> {
    let agent: Arc<Agent<()>> = Arc::new(Agent::new(agent_name).with_instructions("Test agent"));
    let ctx = RunContext::new(());
    let tools = get_all_function_tools_for_agent(
        &[server],
        &MCPConfig::<()>::default(),
        &ctx,
        agent_name,
        Some(agent),
        &Default::default(),
    )
    .await
    .unwrap();
    let mut names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
    names.sort();
    names
}

#[tokio::test]
async fn static_tool_filtering() {
    let tools = ["tool1", "tool2", "tool3", "tool4"];

    // allowed_tool_names only
    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            create_static_tool_filter(Some(vec!["tool1".into(), "tool2".into()]), None),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["tool1", "tool2"]);

    // blocked_tool_names only
    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            create_static_tool_filter(None, Some(vec!["tool3".into(), "tool4".into()])),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["tool1", "tool2"]);

    // both filters together (allowed first, then blocked)
    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            create_static_tool_filter(
                Some(vec!["tool1".into(), "tool2".into(), "tool3".into()]),
                Some(vec!["tool3".into()]),
            ),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["tool1", "tool2"]);

    // no filter
    let names = filtered_names(
        server_with("test_server", &tools, ToolFilter::All),
        "test_agent",
    )
    .await;
    assert_eq!(names.len(), 4);

    // helper function: allow-list first, then block-list
    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            create_static_tool_filter(
                Some(vec!["tool1".into(), "tool2".into()]),
                Some(vec!["tool2".into()]),
            ),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["tool1"]);
}

#[tokio::test]
async fn dynamic_filter_sync_and_async() {
    let tools = ["allowed_tool", "blocked_tool", "restricted_tool"];

    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            ToolFilter::dynamic(|_ctx, tool| tool.name.starts_with("allowed")),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["allowed_tool"]);

    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            ToolFilter::dynamic_async(|_ctx, tool| {
                let name = tool.name.clone();
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    !name.contains("restricted")
                }
            }),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["allowed_tool", "blocked_tool"]);
}

#[tokio::test]
async fn dynamic_filter_context_handling() {
    let tools = ["admin_tool", "user_tool", "guest_tool"];

    // Context-independent filter.
    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            ToolFilter::dynamic(|_ctx, tool| !tool.name.starts_with("admin")),
        ),
        "test_agent",
    )
    .await;
    assert_eq!(names, vec!["guest_tool", "user_tool"]);

    // Context-dependent filter: the context carries the run context, the agent and the server.
    let context_dependent = |ctx: &ToolFilterContext, tool: &openai_agents_sdk::mcp::MCPTool| {
        assert!(ctx.run_context::<()>().is_some());
        assert!(ctx.agent::<()>().is_some());
        assert_eq!(ctx.server_name, "test_server");
        if ctx.agent_name.to_lowercase().contains("admin") {
            true
        } else {
            !tool.name.starts_with("admin")
        }
    };

    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            ToolFilter::dynamic(context_dependent),
        ),
        "regular_user",
    )
    .await;
    assert_eq!(names, vec!["guest_tool", "user_tool"]);

    let names = filtered_names(
        server_with(
            "test_server",
            &tools,
            ToolFilter::dynamic(context_dependent),
        ),
        "admin_user",
    )
    .await;
    assert_eq!(names.len(), 3);
}

#[tokio::test]
async fn agent_dynamic_filtering_integration() {
    let role_based = |ctx: &ToolFilterContext, tool: &openai_agents_sdk::mcp::MCPTool| {
        let agent_name = ctx.agent_name.to_lowercase();
        if agent_name.contains("admin") {
            true
        } else if agent_name.contains("readonly") {
            tool.name.contains("read") || tool.name.contains("query")
        } else {
            tool.name.starts_with("file_")
        }
    };

    let make = || {
        Arc::new(
            FakeMcpServer::new("fake_mcp_server")
                .configure(|c| c.tool_filter = ToolFilter::dynamic(role_based))
                .add_tool(
                    "file_read",
                    serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                )
                .add_tool(
                    "file_write",
                    serde_json::json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    }),
                )
                .add_tool(
                    "database_query",
                    serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
                )
                .add_tool(
                    "network_request",
                    serde_json::json!({"type": "object", "properties": {"url": {"type": "string"}}}),
                ),
        ) as Arc<dyn MCPServer>
    };

    assert_eq!(filtered_names(make(), "admin_user").await.len(), 4);
    assert_eq!(
        filtered_names(make(), "readonly_viewer").await,
        vec!["database_query", "file_read"]
    );
    assert_eq!(
        filtered_names(make(), "regular_user").await,
        vec!["file_read", "file_write"]
    );

    // `Agent::get_all_tools` surfaces the same filtered MCP tools.
    let agent: Arc<Agent<()>> = Arc::new(
        Agent::new("regular_user")
            .with_instructions("Regular")
            .with_mcp_server(make()),
    );
    let ctx = RunContext::new(());
    let all = agent.get_all_tools(&ctx).await.unwrap();
    let mut mcp_names: Vec<String> = all
        .iter()
        .map(|t| t.name().to_string())
        .filter(|n| {
            [
                "file_read",
                "file_write",
                "database_query",
                "network_request",
            ]
            .contains(&n.as_str())
        })
        .collect();
    mcp_names.sort();
    assert_eq!(mcp_names, vec!["file_read", "file_write"]);
}
