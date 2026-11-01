//! Run-state identity and restoration cases from `test_run_state_agent_identity.py` and
//! `test_run_state_compatibility_corpus.py` that the earlier ports skipped: duplicate agents that
//! differ only by `tool_use_behavior`, a saved identity that left the graph, and payload-derived
//! agent names that must not leak into restoration errors.

use std::sync::Arc;

use serde_json::{Value, json};

use openai_agents_sdk::agent::{Instructions, ToolUseBehavior};
use openai_agents_sdk::run_state::{ContextRestoreOptions, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn corpus(path: &str) -> Value {
    let full = format!(
        "{}/tests/fixtures/py_run_state_corpus/{path}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(full).unwrap()).unwrap()
}

fn approval_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "approval_tool",
        "needs approval",
        |_c, _a: NoArgs| async move { Ok("approved output".to_string()) },
    )
    .with_needs_approval(true)
}

/// Two `sandbox` agents with identical instructions that differ only by `tool_use_behavior`,
/// reachable from `triage` in the given order (`stop_first` swaps the handoff order).
fn tool_use_graph(model: &Arc<FakeModel>, stop_first: bool) -> Agent<()> {
    let default_agent = Arc::new(
        Agent::<()>::new("sandbox")
            .with_instructions("Shared instructions.")
            .with_tool_use_behavior(ToolUseBehavior::RunLlmAgain)
            .with_tool(approval_tool())
            .with_model_instance(model.clone()),
    );
    let stop_agent = Arc::new(
        Agent::<()>::new("sandbox")
            .with_instructions("Shared instructions.")
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_tool(approval_tool())
            .with_model_instance(model.clone()),
    );
    let to_default = handoff(default_agent).with_tool_name("transfer_to_default");
    let to_stop = handoff(stop_agent).with_tool_name("transfer_to_stop");
    let triage = Agent::<()>::new("triage").with_model_instance(model.clone());
    if stop_first {
        triage.with_handoff(to_stop).with_handoff(to_default)
    } else {
        triage.with_handoff(to_default).with_handoff(to_stop)
    }
}

async fn snapshot_on_stop_agent(stop_first: bool) -> (Agent<()>, Value) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let triage = tool_use_graph(&model, stop_first);
    model.add_turn(vec![function_call_with_id(
        "transfer_to_stop",
        "{}",
        "call_handoff",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);
    let result = Runner::run(&triage, "go").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    (triage, result.to_state().to_json().unwrap())
}

/// `test_build_agent_identity_map_uses_tool_use_behavior_for_duplicate_names`: the identity of a
/// duplicate does not depend on the order the graph is reached in when `tool_use_behavior` is
/// the only difference between same-named agents.
#[tokio::test]
async fn duplicate_identity_is_stable_when_only_tool_use_behavior_differs() {
    let (_, first) = snapshot_on_stop_agent(false).await;
    let (_, second) = snapshot_on_stop_agent(true).await;
    assert_eq!(first["current_agent"], second["current_agent"]);
    assert_eq!(first["current_agent"]["name"], "sandbox");
}

/// `test_from_json_restores_duplicate_name_current_agent_when_tool_use_behavior_differs`: a
/// snapshot taken on the stop-on-first-tool duplicate restores onto that duplicate even when the
/// restored graph lists the duplicates in the other order.
#[tokio::test]
async fn duplicate_agent_restores_by_identity_when_tool_use_behavior_differs() {
    let (_, json) = snapshot_on_stop_agent(false).await;

    let model = Arc::new(FakeModel::new());
    let reordered = tool_use_graph(&model, true);
    let mut restored = RunState::from_json(json).unwrap();
    let interruption = restored.get_interruptions()[0].clone();
    restored.approve(&interruption, false);
    // StopOnFirstTool ends the run with the tool output; a default duplicate would call the
    // model again and find no scripted turn.
    let resumed = Runner::run(&reordered, restored).await.unwrap();
    assert_eq!(
        resumed.final_output_text().as_deref(),
        Some("approved output")
    );
    assert!(matches!(
        resumed.last_agent.tool_use_behavior,
        ToolUseBehavior::StopOnFirstTool
    ));
    assert!(matches!(
        &resumed.last_agent.instructions,
        Some(Instructions::Static(s)) if s == "Shared instructions."
    ));
}

/// `test_from_json_rejects_missing_saved_duplicate_identity`: an identity-aware snapshot fails
/// when the saved duplicate is no longer in the restored graph.
#[tokio::test]
async fn a_saved_duplicate_identity_missing_from_the_graph_is_refused() {
    let (_, json) = snapshot_on_stop_agent(false).await;
    let restored_state = RunState::from_json(json).unwrap();

    let model = Arc::new(FakeModel::new());
    let only_root = Agent::<()>::new("triage").with_model_instance(model);
    let err = Runner::run(&only_root, restored_state).await.unwrap_err();
    assert!(err.to_string().contains("agent identity"), "{err}");
}

async fn agent_not_found_message(current_agent: Value) -> String {
    let mut payload = corpus("minimal/v1_15.json");
    payload["current_agent"] = current_agent;
    let state = RunState::from_json(payload).expect("agent resolution happens on resume");
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("unused")]);
    let agent = Agent::<()>::new("compat-agent").with_model_instance(model);
    Runner::run(&agent, state).await.unwrap_err().to_string()
}

/// `test_run_state_payload_derived_restoration_errors_are_value_free[agent_name]`.
#[tokio::test]
async fn an_unknown_agent_name_error_is_value_free() {
    let sentinel = "RUNSTATE_AGENT_NAME_SECRET_42";
    let message = agent_not_found_message(json!({"name": sentinel})).await;
    assert!(
        message.contains("Run state agent not found in agent map"),
        "{message}"
    );
    assert!(!message.contains(sentinel), "{message}");
}

/// `test_run_state_payload_derived_restoration_errors_are_value_free[agent_identity]`.
#[tokio::test]
async fn an_unknown_agent_identity_error_is_value_free() {
    let sentinel = "RUNSTATE_AGENT_IDENTITY_SECRET_42";
    let message =
        agent_not_found_message(json!({"name": "compat-agent", "identity": sentinel})).await;
    assert!(message.contains("agent identity"), "{message}");
    assert!(!message.contains(sentinel), "{message}");
}

/// `test_run_state_from_string_malformed_json_discards_caller_owned_state`: a parse failure
/// reports only the SDK's message, never the caller's context override.
#[test]
fn malformed_json_discards_caller_owned_state() {
    let sentinel = "RUNSTATE_MALFORMED_JSON_CALLER_SENTINEL_42";
    let options = ContextRestoreOptions {
        context_override: Some(json!({"credential": sentinel})),
        strict_context: false,
    };
    let err = RunState::from_json_str_with("{", options).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("Failed to parse run state JSON"),
        "{message}"
    );
    assert!(!message.contains(sentinel), "{message}");
}

/// `test_run_state_restores_an_exact_context_wrapper_without_replacing_it`: a supplied context
/// is kept as is while the snapshot's tool input and invocation ledger still restore.
#[test]
fn a_supplied_context_keeps_the_snapshots_tool_input_and_ledger() {
    let mut payload = corpus("features/v1_15_canonical_invocation_identity.json");
    payload["context"]["tool_input"] = json!({"durable": "expected"});
    let options = ContextRestoreOptions {
        context_override: Some(json!({"custom": true})),
        strict_context: true,
    };
    let restored = RunState::from_json_with(payload, options).unwrap();
    assert_eq!(restored.context, Some(json!({"custom": true})));
    assert_eq!(restored.tool_input, Some(json!({"durable": "expected"})));
    assert!(!restored.tool_invocations.is_empty());
}

fn restored_lookup_key(key: Value) -> (Option<Value>, Value) {
    let mut item = json!({
        "type": "tool_approval_item",
        "agent": {"name": "TestAgent"},
        "tool_name": "search",
        "raw_item": {"type": "function_call", "name": "search", "call_id": "call1",
                     "status": "completed", "arguments": "{}"},
    });
    item["tool_lookup_key"] = key;
    let mut wire = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
    .to_json()
    .unwrap();
    wire["generated_items"] = json!([item]);
    let restored = RunState::from_json(wire).unwrap();
    let key = restored.generated_items[0].tool_lookup_key.clone();
    let written = restored.to_json().unwrap();
    (key, written["generated_items"][0].clone())
}

/// `TestSerializeRoundTrip::test_roundtrip`: every lookup key kind survives a write and read.
#[test]
fn every_lookup_key_kind_round_trips_through_a_snapshot() {
    for key in [
        json!({"kind": "bare", "name": "search"}),
        json!({"kind": "namespaced", "namespace": "tools", "name": "search"}),
        json!({"kind": "deferred_top_level", "name": "search"}),
    ] {
        let (restored, written) = restored_lookup_key(key.clone());
        assert_eq!(restored, Some(key.clone()));
        assert_eq!(written["tool_lookup_key"], key);
    }
}

/// `test_deserialize_invalid_returns_none` and `test_serialize_none_returns_none`: a malformed
/// or absent lookup key restores as no key and is not written back.
#[test]
fn a_malformed_lookup_key_restores_as_none() {
    for key in [
        Value::Null,
        json!({}),
        json!({"kind": "bare"}),
        json!({"kind": "bare", "name": ""}),
        json!({"kind": "unknown", "name": "x"}),
        json!({"kind": "namespaced", "name": "x"}),
    ] {
        let (restored, written) = restored_lookup_key(key.clone());
        assert_eq!(restored, None, "{key}");
        assert!(
            written.get("tool_lookup_key").is_none_or(Value::is_null),
            "{key}"
        );
    }
}
