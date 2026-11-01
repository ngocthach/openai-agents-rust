//! Port of the `arguments` half of `test_current_schema_unbindable_pending_approval_cannot_bind_
//! replacement` and `test_current_schema_unbindable_pending_with_ledger_requires_reapproval`
//! (`tests/test_run_state.py`): a parked hosted MCP request whose serialized `raw_item` lost its
//! `arguments` cannot be bound to one invocation, so a restored decision does not authorize it
//! and the resumed run asks again. In Rust the decision is observable through the resume.

use std::sync::Arc;

use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::HostedMCPTool;
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn hosted_agent(model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new("ApprovalCurrentAgent")
        .with_tool(HostedMCPTool::new(json!({
            "type": "mcp", "server_label": "server-a", "server_url": "https://example.com",
            "require_approval": "always",
        })))
        .with_model_instance(model.clone())
}

async fn parked_state() -> RunState {
    let model = fake();
    model.add_turn(vec![OutputItem::from_value(json!({
        "id": "request-a", "type": "mcp_approval_request", "name": "lookup_account",
        "server_label": "server-a", "arguments": r#"{"value":"safe"}"#,
    }))]);
    let result = Runner::run(&hosted_agent(&model), "hi").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    result.to_state()
}

async fn resume(wire: Value) -> openai_agents_sdk::RunResult<()> {
    let state = RunState::from_json(wire).unwrap();
    let model = fake();
    model.add_turn(vec![assistant_message("done")]);
    Runner::run(&hosted_agent(&model), state).await.unwrap()
}

/// With the arguments intact the stored decision authorizes the request.
#[tokio::test]
async fn an_intact_hosted_request_honours_its_stored_decision() {
    let mut state = parked_state().await;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);

    let result = resume(state.to_json().unwrap()).await;

    assert!(result.interruptions.is_empty());
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

/// Python: `test_current_schema_unbindable_pending_with_ledger_requires_reapproval` and the
/// `arguments` case of `..._cannot_bind_replacement`: the decision and the ledger entry are
/// both present, and still the request goes back to the user.
#[tokio::test]
async fn a_hosted_request_without_arguments_is_asked_again() {
    let mut state = parked_state().await;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let mut wire = state.to_json().unwrap();
    wire["current_step"]["data"]["interruptions"][0]["raw_item"]
        .as_object_mut()
        .unwrap()
        .remove("arguments");

    let result = resume(wire).await;

    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].call_id(), "request-a");
    assert_ne!(result.final_output_text().as_deref(), Some("done"));
}
