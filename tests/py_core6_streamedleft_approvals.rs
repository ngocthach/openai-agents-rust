//! `test_tool_approval_call_id_reuse.py::test_legacy_changed_pending_call_fails_before_approval_callback`
//! (schema 1.13 and 1.14). Python drives the private `_collect_runs_by_approval`; here the same
//! parked state resumes through `Runner`: a legacy snapshot whose history already holds a
//! completed `call_0` with other arguments must refuse the changed pending call before the
//! tool's approval rule is consulted.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{FunctionCall, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunState, Runner};
use serde_json::json;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: String,
}

fn pending_item(value: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new("record_value", json!({"value": value}).to_string());
    raw.call_id = "call_0".into();
    ToolApprovalItem {
        agent: "agent".into(),
        raw_item: raw,
        tool_name: "record_value".into(),
        tool_origin: None,
    }
}

#[tokio::test]
async fn legacy_changed_pending_call_fails_before_approval_callback() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for schema in ["1.13", "1.14"] {
        let approval_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let executed = Arc::new(Mutex::new(Vec::<String>::new()));
        let (calls, ran) = (approval_calls.clone(), executed.clone());
        let tool = FunctionTool::new("record_value", "record", move |_c, a: ValueArgs| {
            let ran = ran.clone();
            async move {
                ran.lock().unwrap().push(a.value.clone());
                Ok(a.value)
            }
        })
        .with_needs_approval_fn(move |_c, args, _id| {
            let calls = calls.clone();
            let value = args["value"].as_str().unwrap_or_default().to_string();
            async move {
                calls.lock().unwrap().push(value);
                true
            }
        });
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(tool)
            .with_model_instance(model.clone());

        let pending = pending_item("changed");
        let mut state = RunState::default();
        state.schema_version = openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.into();
        state.current_agent_name = "agent".into();
        state.max_turns = Some(10);
        state.original_input = vec![
            InputItem(json!({
                "type": "function_call", "call_id": "call_0", "name": "record_value",
                "arguments": "{\"value\":\"safe\"}",
            })),
            InputItem(json!({
                "type": "function_call_output", "call_id": "call_0", "output": "safe",
            })),
        ];
        state
            .generated_items
            .push(openai_agents_sdk::items::RunItem::new(
                "agent",
                RunItemKind::ToolApproval {
                    raw_item: pending.raw_item.clone(),
                    tool_name: pending.tool_name.clone(),
                },
            ));
        state.interruptions = vec![pending.clone()];
        state.pending_tool_processing = true;
        let mut json = state.to_json().unwrap();
        json["context"]["approvals"] = json!({"record_value": {"approved": true, "rejected": []}});
        json["$schemaVersion"] = json!(schema);
        json["context"]
            .as_object_mut()
            .unwrap()
            .remove("tool_invocations");
        let restored = RunState::from_json(json).unwrap();

        let err = Runner::run(&agent, restored).await.unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. })
                && err.to_string().contains("unique call ID"),
            "{schema}: {err:?}"
        );
        assert!(approval_calls.lock().unwrap().is_empty(), "{schema}");
        assert!(executed.lock().unwrap().is_empty(), "{schema}");
    }
}
