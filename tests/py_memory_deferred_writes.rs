//! End-state parity ports from `tests/deferred_session_writes/`.
//!
//! Python's suite is mostly white-box: it drives `_PendingSessionWrite`,
//! `_held_items_safe_to_settle` and `defer_interrupted_session_write`, which do not exist in
//! Rust. What is observable through the public API is the session's final content after a run
//! is interrupted for approval and resumed, and that is what these tests pin: a session must
//! never hold a tool call without its output, and a resume must persist the call/output pair
//! exactly once.

use std::sync::Arc;

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

#[derive(Deserialize, JsonSchema)]
struct Query {
    query: String,
}

fn write_thing() -> FunctionTool<()> {
    FunctionTool::new("write_thing", "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn look_up() -> FunctionTool<()> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

fn types(session: &ListSession) -> Vec<String> {
    session
        .items()
        .iter()
        .map(|i| {
            i.item_type()
                .map(str::to_string)
                .or_else(|| i.role().map(|r| format!("role:{r}")))
                .unwrap_or_default()
        })
        .collect()
}

fn call_ids(session: &ListSession, ty: &str) -> Vec<String> {
    session
        .items()
        .iter()
        .filter(|i| i.item_type() == Some(ty))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Python's `always_fine` output guardrail. An interruption withholds its session write only
/// while an output guardrail could still reject the turn.
fn always_fine() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn deferring_agent(model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new("deferred repro")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_output_guardrail(always_fine())
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["finish".into()],
        }))
        .with_model_instance(model.clone())
}

/// Python defers the interruption-time session write of a parked call when an output guardrail
/// and a `StopAtTools` gate are active (`defer_interrupted_session_write`): the session never
/// holds a call without its output.
#[tokio::test]
async fn a_gated_interrupted_run_never_leaves_an_unpaired_call_in_the_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "look_up",
        r#"{"query":"x"}"#,
        "call_LOOKUP",
    )]);
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"x"}"#,
        "call_PARKED",
    )]);
    let agent = deferring_agent(&model);
    let session = Arc::new(ListSession::new("deferred"));

    let paused =
        Runner::run_with_options(&agent, "go", RunOptions::new(()).session(session.clone()))
            .await
            .unwrap();
    assert_eq!(paused.interruptions.len(), 1);

    // Every stored call has its output; the parked call is not persisted on its own.
    let calls = call_ids(&session, "function_call");
    let outputs = call_ids(&session, "function_call_output");
    for call in &calls {
        assert!(
            outputs.contains(call),
            "unpaired call {call}: {:?}",
            types(&session)
        );
    }
    assert!(
        !calls.contains(&"call_PARKED".to_string()),
        "the parked call must wait for its output: {:?}",
        types(&session)
    );
}

#[tokio::test]
async fn resuming_persists_the_parked_call_and_its_output_once() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "look_up",
        r#"{"query":"x"}"#,
        "call_LOOKUP",
    )]);
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"x"}"#,
        "call_PARKED",
    )]);
    let agent = deferring_agent(&model);
    let session = Arc::new(ListSession::new("deferred"));

    let paused =
        Runner::run_with_options(&agent, "go", RunOptions::new(()).session(session.clone()))
            .await
            .unwrap();
    let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);

    model.add_turn(vec![assistant_message("done")]);
    let resumed =
        Runner::run_with_options(&agent, state, RunOptions::new(()).session(session.clone()))
            .await
            .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));

    let calls = call_ids(&session, "function_call");
    let outputs = call_ids(&session, "function_call_output");
    assert_eq!(
        calls.iter().filter(|c| *c == "call_PARKED").count(),
        1,
        "{:?}",
        types(&session)
    );
    assert_eq!(
        outputs.iter().filter(|c| *c == "call_PARKED").count(),
        1,
        "{:?}",
        types(&session)
    );
    assert_eq!(calls.iter().filter(|c| *c == "call_LOOKUP").count(), 1);
    assert_eq!(outputs.iter().filter(|c| *c == "call_LOOKUP").count(), 1);
}

#[tokio::test]
async fn a_rejected_approval_still_stores_a_paired_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"x"}"#,
        "call_PARKED",
    )]);
    let agent = deferring_agent(&model);
    let session = Arc::new(ListSession::new("deferred"));

    let paused =
        Runner::run_with_options(&agent, "go", RunOptions::new(()).session(session.clone()))
            .await
            .unwrap();
    let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, None);

    model.add_turn(vec![assistant_message("done")]);
    Runner::run_with_options(&agent, state, RunOptions::new(()).session(session.clone()))
        .await
        .unwrap();

    let calls = call_ids(&session, "function_call");
    let outputs = call_ids(&session, "function_call_output");
    for call in &calls {
        assert!(
            outputs.contains(call),
            "unpaired call {call}: {:?}",
            types(&session)
        );
    }
}
