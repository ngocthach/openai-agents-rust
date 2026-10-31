//! `test_run_state.py::test_schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval`:
//! a schema 1.13 snapshot carries per-call hosted MCP decisions with no pending item to bind them
//! to, so a request with that id is asked again instead of being answered from the old decision.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::FakeModel;
use openai_agents_sdk::tool::HostedMCPTool;
use openai_agents_sdk::{Agent, InputItem, RunState, Runner};
use serde_json::{Value, json};

fn mcp_request(id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "id": id, "type": "mcp_approval_request", "name": "lookup_account",
        "server_label": "server-a", "arguments": "{}",
    }))
}

fn hosted_tool() -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": "server-a", "server_url": "https://example.com",
        "require_approval": "always",
    }))
}

fn responses(result: &openai_agents_sdk::RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for request_id in ["request-approved", "request-rejected"] {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![mcp_request(request_id)]);
        let agent = Agent::<()>::new("test")
            .with_tool(hosted_tool())
            .with_model_instance(model.clone());

        let mut state = RunState::default();
        state.current_agent_name = "test".into();
        state.max_turns = Some(10);
        state.original_input = vec![InputItem::user("hello")];
        let mut json = state.to_json().unwrap();
        json["context"]["approvals"] = json!({
            "lookup_account": {
                "approved": ["request-approved"],
                "rejected": ["request-rejected"],
                "rejection_messages": {"request-rejected": "legacy exact denial"},
            }
        });
        json["$schemaVersion"] = json!("1.13");
        json["context"]
            .as_object_mut()
            .unwrap()
            .remove("tool_invocations");
        let restored = RunState::from_json(json).unwrap();

        let result = Runner::run(&agent, restored).await.unwrap();
        assert_eq!(result.interruptions.len(), 1, "{request_id}");
        assert_eq!(result.interruptions[0].call_id(), request_id);
        assert!(responses(&result).is_empty(), "{request_id}");
    }
}

#[allow(dead_code)]
fn _unused(_: Mutex<()>) {}
