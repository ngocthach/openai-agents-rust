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
