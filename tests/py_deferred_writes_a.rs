//! Parity port of the end-state tests in `tests/deferred_session_writes/test_resume.py`.
//!
//! Python drives the held-batch machinery (`pending_session_write`, `_PendingSessionWrite`)
//! directly. Rust has no such run-pipeline object, so each port asserts only what is visible
//! through `Runner`, `RunState` and `Session`: after an interrupt, approve or reject, and
//! resume, the session holds every call with exactly one paired output and no orphans.
//! Streamed and non-streamed runs are both exercised where Python parametrizes them.

use std::sync::Arc;

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::items::ToolApprovalItem;
use openai_agents_sdk::memory::Session;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, RunOptions, RunState,
    Runner,
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

fn write_other() -> FunctionTool<()> {
    FunctionTool::new("write_other", "writes", |_ctx, q: Query| async move {
        Ok(format!("other:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn look_up() -> FunctionTool<()> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

fn stop_at(names: &[&str]) -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: names.iter().map(|n| n.to_string()).collect(),
    })
}

fn ok_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn tripping_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_trips", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    })
}

fn crashing_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_crashes", |_c, _a, _o: Value| async {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail crashed"))
    })
}

/// Python's `_make_deferring_agent`: look_up, then a gated write, then a final message.
fn deferring_agent(model: &Arc<FakeModel>, behavior: ToolUseBehavior<()>) -> Agent<()> {
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
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("deferred repro")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(behavior)
        .with_model_instance(model.clone())
}

fn two_park_agent(model: &Arc<FakeModel>) -> Agent<()> {
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"a"}"#,
        "call_A",
    )]);
    model.add_turn(vec![function_call_with_id(
        "write_other",
        r#"{"query":"b"}"#,
        "call_B",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("deferred repro (two parks)")
        .with_tool(write_thing())
        .with_tool(write_other())
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(stop_at(&["finish"]))
        .with_model_instance(model.clone())
}

fn multi_approval_agent(model: &Arc<FakeModel>, behavior: ToolUseBehavior<()>) -> Agent<()> {
    model.add_turn(vec![
        function_call_with_id("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
        function_call_with_id("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("deferred repro (multi)")
        .with_tool(write_thing())
        .with_tool(write_other())
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(behavior)
        .with_model_instance(model.clone())
}

/// One run's visible outcome, whether streamed or not.
#[derive(Debug)]
struct Outcome {
    final_text: Option<String>,
    interruptions: Vec<ToolApprovalItem>,
    state: RunState,
}

/// Python's `_run`: drive the run to completion (streamed or not) and report its outcome.
async fn drive(
    agent: &Agent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: Option<Arc<dyn Session>>,
    streamed: bool,
) -> Result<Outcome, AgentsError> {
    let mut options = RunOptions::new(());
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            use futures::StreamExt;
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions(),
            state: result
                .to_state()
                .expect("a finished streamed run has state"),
        })
    } else {
        let result = Runner::run_with_options(agent, input, options).await?;
        Ok(Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions.clone(),
            state: result.to_state(),
        })
    }
}

/// Python's `_serialized_round_trip`: the checkpoint must survive JSON, as a server would store it.
fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

fn items_of(session: &ListSession) -> Vec<InputItem> {
    session.items()
}

fn call_ids(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn output_ids(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call_output"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Python's `_orphaned_outputs`: outputs whose call is not stored.
fn orphaned_outputs(items: &[InputItem]) -> Vec<String> {
    let calls = call_ids(items);
    output_ids(items)
        .into_iter()
        .filter(|o| !calls.contains(o))
        .collect()
}

/// Python's `_parked_pair`: the item types stored under `call_PARKED`, in order.
fn parked_pair(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect()
}

const EXPECTED_PAIR: [&str; 2] = ["function_call", "function_call_output"];

/// Python's `_parked_and_approved`: interrupt, checkpoint through JSON, approve the one call.
async fn parked_and_approved(
    agent: &Agent<()>,
    session: Option<Arc<dyn Session>>,
    streamed: bool,
) -> RunState {
    let first = drive(agent, "do the thing", session, streamed)
        .await
        .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = round_trip(&first.state);
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    state
}

#[tokio::test]
async fn parked_call_is_persisted_when_the_resume_runs_again() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let agent = deferring_agent(&model, stop_at(&["finish"]));
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, Some(session.clone()), streamed).await;

        model.add_turn(vec![assistant_message("done")]);
        let resumed = drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(resumed.final_text.as_deref(), Some("done"));

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
    }
}

#[tokio::test]
async fn park_time_deferral_survives_a_tool_use_behavior_change_on_resume() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let first_agent = deferring_agent(&model, stop_at(&["finish"]));
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&first_agent, Some(session.clone()), streamed).await;

        // The caller resumes under a different tool-use behavior; the checkpoint's decision
        // must still win, so the parked call is not lost.
        let resume_agent = deferring_agent(&model, ToolUseBehavior::RunLlmAgain);
        model.add_turn(vec![assistant_message("done")]);
        drive(&resume_agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
    }
}

#[tokio::test]
async fn non_deferred_park_is_not_double_written_on_resume() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let agent = deferring_agent(&model, ToolUseBehavior::RunLlmAgain);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, Some(session.clone()), streamed).await;

        drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();

        let items = items_of(&session);
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
        assert_eq!(
            call_ids(&items)
                .iter()
                .filter(|c| *c == "call_PARKED")
                .count(),
            1,
            "streamed={streamed}"
        );
    }
}

#[tokio::test]
async fn partial_approval_reinterruption_keeps_one_canonical_batch() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let agent = multi_approval_agent(&model, stop_at(&["finish"]));
        let session = Arc::new(ListSession::new("test"));

        let first = drive(&agent, "go", Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(first.interruptions.len(), 2);
        let mut state = round_trip(&first.state);
        let parked = state
            .get_interruptions()
            .into_iter()
            .find(|i| i.raw_item.call_id == "call_PARKED")
            .expect("call_PARKED is interrupted");
        state.approve(&parked, false);

        let second = drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(second.interruptions.len(), 1, "streamed={streamed}");
        let mut state = round_trip(&second.state);
        for interruption in state.get_interruptions() {
            state.approve(&interruption, false);
        }

        let finished = drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(finished.final_text.as_deref(), Some("done"));

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(
            call_ids(&items)
                .iter()
                .filter(|c| *c == "call_PARKED")
                .count(),
            1
        );
        assert_eq!(
            call_ids(&items)
                .iter()
                .filter(|c| *c == "call_PARKED_2")
                .count(),
            1
        );
        let calls = call_ids(&items);
        let outputs = output_ids(&items);
        for call in &calls {
            assert!(outputs.contains(call), "call {call} lost its output");
        }
    }
}

#[tokio::test]
async fn deferred_prefix_reaches_a_resume_that_ends_in_final_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    for with_guardrails in [true, false] {
        for streamed in [false, true] {
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
            let agent = Agent::<()>::new("deferred repro (terminal)")
                .with_tool(look_up())
                .with_tool(write_thing())
                .with_tool_use_behavior(stop_at(&["write_thing"]))
                .with_model_instance(model.clone());
            let agent = if with_guardrails {
                agent.with_output_guardrail(ok_guardrail())
            } else {
                agent
            };
            let session = Arc::new(ListSession::new("test"));
            let state = parked_and_approved(&agent, Some(session.clone()), streamed).await;

            drive(&agent, state, Some(session.clone()), streamed)
                .await
                .unwrap();

            let items = items_of(&session);
            assert!(
                orphaned_outputs(&items).is_empty(),
                "{with_guardrails}/{streamed}"
            );
            assert_eq!(
                parked_pair(&items),
                EXPECTED_PAIR,
                "{with_guardrails}/{streamed}"
            );
        }
    }
}

#[tokio::test]
async fn reject_persists_the_parked_call_with_its_rejection_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let agent = deferring_agent(&model, stop_at(&["finish"]));
        let session = Arc::new(ListSession::new("test"));

        let first = drive(&agent, "do the thing", Some(session.clone()), streamed)
            .await
            .unwrap();
        let mut state = round_trip(&first.state);
        let item = state.get_interruptions()[0].clone();
        state.reject(&item, false, None);

        let resumed = drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(resumed.final_text.as_deref(), Some("done"));

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
    }
}

#[tokio::test]
async fn a_tripwire_after_approval_keeps_the_sanitized_pair() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            "write_thing",
            r#"{"query":"x"}"#,
            "call_PARKED",
        )]);
        let parked_agent = Agent::<()>::new("deferred repro (tripwire)")
            .with_tool(write_thing())
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(stop_at(&["write_thing"]))
            .with_model_instance(model.clone());
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&parked_agent, Some(session.clone()), streamed).await;

        model.add_turn(vec![assistant_message("preamble text is not persisted")]);
        let resume_agent = Agent::<()>::new("deferred repro (tripwire)")
            .with_tool(write_thing())
            .with_tool_use_behavior(stop_at(&["write_thing"]))
            .with_output_guardrail(tripping_guardrail())
            .with_model_instance(model.clone());
        let err = drive(&resume_agent, state, Some(session.clone()), streamed)
            .await
            .expect_err("the output guardrail must trip");
        assert!(
            matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
            "{err:?}"
        );

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
    }
}

/// Python's crash-path save persists the approved call together with its output
/// (`test_resume.py:407`): when an output guardrail raises, the turn that ran is still written, so
/// history never keeps a call without its output.
#[tokio::test]
async fn a_guardrail_crash_still_persists_the_parked_call() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            "write_thing",
            r#"{"query":"x"}"#,
            "call_PARKED",
        )]);
        let parked_agent = Agent::<()>::new("deferred repro (crash)")
            .with_tool(write_thing())
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(stop_at(&["write_thing"]))
            .with_model_instance(model.clone());
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&parked_agent, Some(session.clone()), streamed).await;

        let resume_agent = Agent::<()>::new("deferred repro (crash)")
            .with_tool(write_thing())
            .with_tool_use_behavior(stop_at(&["write_thing"]))
            .with_output_guardrail(crashing_guardrail())
            .with_model_instance(model.clone());
        let err = drive(&resume_agent, state, Some(session.clone()), streamed)
            .await
            .expect_err("the guardrail crashes");
        assert!(err.to_string().contains("guardrail crashed"), "{err}");

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        assert_eq!(parked_pair(&items), EXPECTED_PAIR, "streamed={streamed}");
    }
}

#[tokio::test]
async fn a_new_park_during_a_detached_resume_joins_the_held_batch() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let agent = two_park_agent(&model);
        let session = Arc::new(ListSession::new("test"));
        // Park A on the first run, then approve it.
        let first = drive(&agent, "do the thing", Some(session.clone()), streamed)
            .await
            .unwrap();
        let mut state = round_trip(&first.state);
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);

        // Detached resume (no session): B parks, and nothing is written yet.
        let detached = drive(&agent, state, None, streamed).await.unwrap();
        assert_eq!(detached.interruptions.len(), 1);
        let mut state = round_trip(&detached.state);
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);

        // Reattaching resume settles everything, including the call parked while detached.
        let reattached = drive(&agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(reattached.final_text.as_deref(), Some("done"));

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        let calls: Vec<String> = call_ids(&items);
        let outputs = output_ids(&items);
        assert_eq!(
            calls.iter().collect::<std::collections::BTreeSet<_>>(),
            outputs.iter().collect::<std::collections::BTreeSet<_>>(),
            "streamed={streamed}"
        );
        assert!(calls.contains(&"call_A".to_string()));
        assert!(calls.contains(&"call_B".to_string()));
    }
}

#[tokio::test]
async fn a_gate_off_reinterruption_keeps_the_still_pending_call() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        let session = Arc::new(ListSession::new("test"));
        let agent = multi_approval_agent(&model, stop_at(&["finish"]));
        let first = drive(&agent, "go", Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(first.interruptions.len(), 2);

        // Approve only one of the two held calls, then resume with the gate turned off.
        let mut state = round_trip(&first.state);
        let parked = state
            .get_interruptions()
            .into_iter()
            .find(|i| i.raw_item.call_id == "call_PARKED")
            .unwrap();
        state.approve(&parked, false);
        let resume_agent = multi_approval_agent(&model, ToolUseBehavior::RunLlmAgain);
        let second = drive(&resume_agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();
        assert_eq!(second.interruptions.len(), 1, "streamed={streamed}");

        // The still-pending call must not be orphaned by the mid-run settle; approving it
        // now completes the pair.
        let mut state = round_trip(&second.state);
        for interruption in state.get_interruptions() {
            state.approve(&interruption, false);
        }
        drive(&resume_agent, state, Some(session.clone()), streamed)
            .await
            .unwrap();

        let items = items_of(&session);
        assert!(orphaned_outputs(&items).is_empty(), "streamed={streamed}");
        let outputs = output_ids(&items);
        for call in call_ids(&items) {
            assert!(
                outputs.contains(&call),
                "{call} lost its output (streamed={streamed})"
            );
        }
    }
}

#[tokio::test]
async fn detached_completion_clears_the_held_record_and_leaves_the_session_untouched() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = deferring_agent(&model, stop_at(&["finish"]));
    let session = Arc::new(ListSession::new("test"));
    let state = parked_and_approved(&agent, Some(session.clone()), false).await;

    // A detached resume has no Session to settle against: the run completes, and the
    // session keeps only what the attached first run wrote.
    let before = items_of(&session).len();
    let done = drive(&agent, state, None, false).await.unwrap();
    assert_eq!(done.final_text.as_deref(), Some("done"));
    assert_eq!(items_of(&session).len(), before);
}
