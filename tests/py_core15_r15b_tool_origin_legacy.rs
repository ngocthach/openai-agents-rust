//! Ports of the legacy run-state cases of the Python SDK's `tests/test_tool_origin.py`: a state
//! saved before tool origins existed (schema 1.5 for tool calls, 1.6 for approvals) restores with
//! no origin and keeps the rest of the item.

use std::sync::Arc;

use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunState, Runner};
use serde_json::{Value, json};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn strip_tool_origin(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("tool_origin");
            map.values_mut().for_each(strip_tool_origin);
        }
        Value::Array(items) => items.iter_mut().for_each(strip_tool_origin),
        _ => {}
    }
}

fn agent(needs_approval: bool) -> Agent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "legacy_tool",
        "{}",
        "call_legacy",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("legacy")
        .with_tool(
            FunctionTool::new("legacy_tool", "Legacy tool", |_c, _a: NoArgs| async move {
                Ok("ok".to_string())
            })
            .with_needs_approval(needs_approval),
        )
        .with_model_instance(model)
}

/// `test_run_state_from_json_reads_legacy_1_5_without_tool_origin`.
#[tokio::test]
async fn a_schema_1_5_tool_call_item_restores_without_a_tool_origin() {
    let agent = agent(false);
    let result = Runner::run(&agent, "go").await.unwrap();
    let mut json = result.to_state().to_json().unwrap();
    let current = RunState::from_json(json.clone()).unwrap();
    let current_origin = current
        .generated_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCall { .. }))
        .and_then(|i| i.tool_origin.clone());
    assert!(current_origin.is_some(), "the saved item carries an origin");

    json["$schemaVersion"] = json!("1.5");
    strip_tool_origin(&mut json["generated_items"]);
    let restored = RunState::from_json(json).unwrap();

    let item = restored
        .generated_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCall { .. }))
        .expect("the tool call item");
    assert_eq!(item.description.as_deref(), Some("Legacy tool"));
    assert!(item.tool_origin.is_none());
}

/// `test_run_state_from_json_reads_legacy_1_6_approval_without_tool_origin`.
#[tokio::test]
async fn a_schema_1_6_approval_item_restores_without_a_tool_origin() {
    let agent = agent(true);
    let parked = Runner::run(&agent, "go").await.unwrap();
    let mut json = parked.to_state().to_json().unwrap();
    let current = RunState::from_json(json.clone()).unwrap();
    assert!(
        current
            .generated_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolApproval { .. }) && i.tool_origin.is_some()),
        "the saved approval item carries an origin"
    );

    json["$schemaVersion"] = json!("1.6");
    strip_tool_origin(&mut json["generated_items"]);
    let restored = RunState::from_json(json).unwrap();

    let approval = restored
        .generated_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolApproval { .. }))
        .expect("the approval item");
    assert!(approval.tool_origin.is_none());
}
