//! Ports of the last open cases of `tests/test_tool_identity.py` and
//! `tests/test_tool_use_tracker.py`.
//!
//! The lookup-key decoder and `AgentToolUseTracker` are internal in Rust, so both are observed
//! through the public run-state wire format and `RunResult::to_state()`.

use std::sync::Arc;

use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call_full,
    function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, ShellTool};
use openai_agents_sdk::{Agent, Runner, handoff};
use serde_json::{Value, json};

fn make_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
}

fn approval_item_with_key(key: Value) -> Value {
    json!({
        "type": "tool_approval_item",
        "agent": {"name": "TestAgent"},
        "tool_name": "get_weather",
        "raw_item": {"type": "function_call", "name": "get_weather", "call_id": "call123",
                     "status": "completed", "arguments": "{}"},
        "tool_lookup_key": key,
    })
}

fn restored_key(key: Value) -> Option<Value> {
    let mut wire = make_state().to_json().unwrap();
    wire["generated_items"] = json!([approval_item_with_key(key)]);
    RunState::from_json(wire).unwrap().generated_items[0]
        .tool_lookup_key
        .clone()
}

#[test]
fn deserialize_invalid_lookup_key_returns_none() {
    for invalid in [
        Value::Null,
        json!({}),
        json!({"kind": "bare"}),
        json!({"kind": "bare", "name": ""}),
        json!({"kind": "unknown", "name": "x"}),
        // namespaced kind requires a non-empty namespace
        json!({"kind": "namespaced", "name": "x"}),
        json!({"kind": "namespaced", "namespace": "", "name": "x"}),
        json!({"kind": 1, "name": "x"}),
        json!("bare"),
    ] {
        assert_eq!(restored_key(invalid.clone()), None, "{invalid}");
    }
}

#[test]
fn deserialize_valid_lookup_keys_round_trip() {
    for valid in [
        json!({"kind": "bare", "name": "x"}),
        json!({"kind": "deferred_top_level", "name": "x"}),
        json!({"kind": "namespaced", "namespace": "ns", "name": "x"}),
    ] {
        assert_eq!(restored_key(valid.clone()), Some(valid));
    }
}

fn scripted(turns: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(turns.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

#[tokio::test]
async fn tool_use_tracker_handles_literal_suffix_names_without_collision() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    // `second` is the third agent named like `sandbox`; `sandbox#2` is a literal name that the
    // identity scheme must not hand out to a duplicate.
    let literal_suffix = Arc::new(Agent::<()>::new("sandbox#2").with_model_instance(model.clone()));
    let approval_tool =
        FunctionTool::<()>::new_no_args("approval_tool", "needs approval", |_c| async {
            Ok("approved output")
        })
        .with_needs_approval(true);
    let second = Arc::new(
        Agent::<()>::new("sandbox")
            .with_tool(approval_tool)
            .with_model_instance(model.clone()),
    );
    let first = Agent::<()>::new("sandbox")
        .with_handoff(handoff(literal_suffix).with_tool_name("transfer_to_literal"))
        .with_handoff(handoff(second).with_tool_name("transfer_to_second"))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call_with_id(
        "transfer_to_second",
        "{}",
        "call_handoff",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);

    let result = Runner::run(&first, "go").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let snapshot = result.to_state().to_json().unwrap();
    assert_eq!(
        snapshot["tool_use_tracker"],
        json!({"sandbox#3": ["approval_tool"]})
    );

    // Hydrating the snapshot puts the tool use back on the agent it named.
    let mut restored = RunState::from_json(snapshot).unwrap();
    assert_eq!(
        restored.current_agent_identity.as_deref(),
        Some("sandbox#3")
    );
    let interruption = restored.get_interruptions()[0].clone();
    restored.approve(&interruption, false);
    model.add_turn(vec![assistant_message("all done")]);
    let resumed = Runner::run(&first, restored).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("all done"));
    assert_eq!(
        resumed.to_state().tool_use_tracker["sandbox#3"],
        vec!["approval_tool".to_string()]
    );
}

#[tokio::test]
async fn hosted_shell_output_is_counted_without_shifting_names() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let shell_output = OutputItem::from_value(json!({
        "type": "shell_call_output",
        "id": "sco_1",
        "call_id": "shell-1",
        "output": [],
        "status": "completed",
    }));
    let model = scripted(vec![
        vec![
            shell_output,
            function_call_full("lookup_account", "{}", "call-1", None, None),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("tracked-agent")
        .with_tool(FunctionTool::<()>::new_no_args(
            "lookup_account",
            "d",
            |_c| async { Ok("ok") },
        ))
        .with_tool(ShellTool::<()>::hosted(json!({"type": "container_auto"})).unwrap())
        .with_model_instance(model);
    let result = Runner::run(&agent, "go").await.unwrap();

    assert_eq!(
        result.to_state().tool_use_tracker["tracked-agent"],
        vec!["lookup_account".to_string(), "shell".to_string()]
    );
}
