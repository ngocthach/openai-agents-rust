//! Parity port of `tests/deferred_session_writes/test_filters.py`.
//!
//! Python asserts on the held record (`pending_session_write`, `_pending_session_write`); Rust has
//! no such object, so each port asserts the session's final content, which is the user-visible
//! contract: a handoff filter's verdict over session history is honoured, and no call is stored
//! without its output (or the reverse).

use std::sync::Arc;

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::extensions::handoff_filters::remove_all_tools;
use openai_agents_sdk::handoffs::{HandoffInputData, handoff};
use openai_agents_sdk::items::{RunInput, RunItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::result::CancelMode;
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

fn gated(name: &'static str, out: &'static str) -> FunctionTool<()> {
    FunctionTool::new(name, "gated", move |_ctx, q: Query| async move {
        Ok(format!("{out}:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn write_thing() -> FunctionTool<()> {
    gated("write_thing", "wrote")
}

fn write_other() -> FunctionTool<()> {
    gated("write_other", "other")
}

fn read_secret() -> FunctionTool<()> {
    FunctionTool::new("read_secret", "secret", |_ctx, _q: Query| async move {
        Ok("SECRET-VALUE-42".to_string())
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn look_up() -> FunctionTool<()> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

fn deferring() -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: vec!["finish".into()],
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

fn model_with(turns: Vec<Vec<openai_agents_sdk::OutputItem>>) -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    for turn in turns {
        model.add_turn(turn);
    }
    model
}

fn call(name: &str, args: &str, id: &str) -> openai_agents_sdk::OutputItem {
    function_call_with_id(name, args, id)
}

fn done_target() -> Agent<()> {
    Agent::<()>::new("target")
        .with_instructions("x")
        .with_model_instance(model_with(vec![vec![assistant_message("done")]]))
}

#[derive(Debug)]
struct Outcome {
    final_text: Option<String>,
    interruptions: Vec<ToolApprovalItem>,
    state: Option<RunState>,
}

async fn drive(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<ListSession>>,
    streamed: bool,
    cancel_after_turn: bool,
) -> Result<Outcome, AgentsError> {
    let mut options = RunOptions::new(());
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        if cancel_after_turn {
            result.cancel(CancelMode::AfterTurn);
        }
        {
            use futures::StreamExt;
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions(),
            state: result.to_state(),
        })
    } else {
        let result = Runner::run_with_options(agent, input, options).await?;
        Ok(Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions.clone(),
            state: Some(result.to_state()),
        })
    }
}

fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

async fn parked_and_approved(
    agent: &Agent<()>,
    session: &Arc<ListSession>,
    streamed: bool,
) -> RunState {
    let first = drive(
        agent,
        "do the thing",
        Some(session.clone()),
        streamed,
        false,
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = round_trip(first.state.as_ref().unwrap());
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    state
}

fn calls_of(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn outputs_of(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call_output"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn assert_paired(items: &[InputItem], label: &str) {
    let calls = calls_of(items);
    let outputs = outputs_of(items);
    for c in &calls {
        assert!(outputs.contains(c), "{label}: dangling call {c}");
    }
    for o in &outputs {
        assert!(calls.contains(o), "{label}: orphaned output {o}");
    }
}

fn parked_pair(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect()
}

fn mentions(items: &[InputItem], needle: &str) -> usize {
    items
        .iter()
        .filter(|i| i.as_value().to_string().contains(needle))
        .count()
}

const PAIR: [&str; 2] = ["function_call", "function_call_output"];

fn secret_handoff_agent(
    filter: impl Fn(HandoffInputData) -> HandoffInputData + Send + Sync + 'static,
) -> Agent<()> {
    let model = model_with(vec![
        vec![
            call("read_secret", r#"{"query":"x"}"#, "call_SECRET"),
            call("transfer_to_target", "{}", "call_HANDOFF"),
        ],
        vec![assistant_message("done")],
    ]);
    Agent::<()>::new("deferred repro (secret)")
        .with_instructions("x")
        .with_tool(read_secret())
        .with_handoff(handoff(done_target()).with_input_filter(filter))
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(deferring())
        .with_model_instance(model)
}

/// Python's `test_an_emptied_resolved_turn_honors_the_filters_history_authority`.
#[tokio::test]
async fn an_emptied_resolved_turn_honors_the_filters_history_authority() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let target_model = model_with(vec![
            vec![assistant_message("done")],
            vec![assistant_message("done")],
        ]);
        let target = Agent::<()>::new("target")
            .with_instructions("x")
            .with_model_instance(target_model);
        let model = model_with(vec![
            vec![
                call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
                call("transfer_to_target", "{}", "call_HANDOFF"),
            ],
            vec![assistant_message("done")],
        ]);
        let agent = Agent::<()>::new("deferred repro (emptied turn)")
            .with_instructions("x")
            .with_tool(write_thing())
            .with_handoff(handoff(target).with_input_filter(|data: HandoffInputData| {
                HandoffInputData {
                    input_history: data.input_history,
                    pre_handoff_items: Vec::new(),
                    new_items: Vec::new(),
                    input_items: data.input_items,
                }
            }))
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let items = session.items();
        assert_paired(&items, &format!("streamed={streamed}"));
        assert!(
            !calls_of(&items).contains(&"call_PARKED".to_string()),
            "the filter removed the pair from session history (streamed={streamed})"
        );
        assert!(!calls_of(&items).contains(&"call_HANDOFF".to_string()));
    }
}

/// Python's `test_a_filter_that_drops_one_output_takes_its_held_call_with_it`.
///
/// Python removes a call together with the output a filter dropped (`test_filters.py:148`).
/// Duplicate of `py_deferred_writes_b.rs` (kept here so every test_filters case lives in one file).
#[tokio::test]
async fn a_filter_that_drops_one_output_takes_its_held_call_with_it() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![vec![
            call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
            call("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
            call("transfer_to_target", "{}", "call_HANDOFF"),
        ]]);
        let drops = |items: Vec<RunItem>| -> Vec<RunItem> {
            items
                .into_iter()
                .filter(|item| {
                    !matches!(
                        &item.kind,
                        RunItemKind::ToolCallOutput { raw_item, .. }
                            if raw_item.get("call_id").and_then(Value::as_str) == Some("call_PARKED_2")
                    )
                })
                .collect()
        };
        let agent = Agent::<()>::new("deferred repro (partial filter)")
            .with_instructions("x")
            .with_tool(write_thing())
            .with_tool(write_other())
            .with_handoff(
                handoff(done_target()).with_input_filter(move |data| HandoffInputData {
                    input_history: data.input_history,
                    pre_handoff_items: drops(data.pre_handoff_items),
                    new_items: drops(data.new_items),
                    input_items: data.input_items,
                }),
            )
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(ListSession::new("test"));
        let first = drive(&agent, "go", Some(session.clone()), streamed, false)
            .await
            .unwrap();
        let mut state = round_trip(first.state.as_ref().unwrap());
        for i in state.get_interruptions() {
            state.approve(&i, false);
        }
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        let items = session.items();
        assert_paired(&items, &format!("streamed={streamed}"));
        assert!(calls_of(&items).contains(&"call_PARKED".to_string()));
        assert!(!calls_of(&items).contains(&"call_PARKED_2".to_string()));
    }
}

/// Python's `test_the_librarys_own_filter_keeps_the_secret_out_of_the_session`.
#[tokio::test]
async fn the_librarys_own_filter_keeps_the_secret_out_of_the_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let agent = secret_handoff_agent(remove_all_tools);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let items = session.items();
        assert_eq!(
            mentions(&items, "SECRET-VALUE-42"),
            0,
            "streamed={streamed}"
        );
        assert_paired(&items, &format!("streamed={streamed}"));
    }
}

/// Python's `test_an_input_items_only_filter_preserves_the_pair_in_session`: `input_items` is the
/// model-input axis, so filtering only it leaves session history alone.
#[tokio::test]
async fn an_input_items_only_filter_preserves_the_pair_in_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let agent = secret_handoff_agent(|data| HandoffInputData {
            input_items: Some(Vec::new()),
            ..data
        });
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let items = session.items();
        assert!(
            mentions(&items, "SECRET-VALUE-42") > 0,
            "streamed={streamed}"
        );
        assert!(calls_of(&items).contains(&"call_SECRET".to_string()));
        assert!(outputs_of(&items).contains(&"call_SECRET".to_string()));
        assert_paired(&items, &format!("streamed={streamed}"));
    }
}

/// Python's `test_a_detached_filtered_handoff_drops_the_pair_before_the_reattach`, end-state
/// half: after a detached after-turn cancel and a reattaching resume, the library filter's
/// verdict still keeps the secret out of the session. The checkpoint-JSON assertions
/// (`pending_session_write`) are Python-only.
#[tokio::test]
async fn a_detached_filtered_handoff_keeps_the_secret_out_after_the_reattach() {
    openai_agents_sdk::set_tracing_disabled(true);
    let agent = secret_handoff_agent(remove_all_tools);
    let session = Arc::new(ListSession::new("test"));
    let state = parked_and_approved(&agent, &session, true).await;

    let detached = drive(&agent, state, None, true, true).await.unwrap();
    let Some(checkpoint) = detached.state else {
        return;
    };
    let reattach = round_trip(&checkpoint);
    drive(&agent, reattach, Some(session.clone()), false, false)
        .await
        .ok();

    let items = session.items();
    assert_eq!(mentions(&items, "SECRET-VALUE-42"), 0, "{items:?}");
    assert_paired(&items, "reattach");
}

/// Python's `test_a_carried_pair_survives_a_filtered_handoff_on_a_later_run`, end-state half.
#[tokio::test]
async fn a_carried_pair_survives_a_detached_after_turn_cancel() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![
            vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
            vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")],
            vec![assistant_message("done")],
        ]);
        let agent = Agent::<()>::new("deferred repro")
            .with_tool(look_up())
            .with_tool(write_thing())
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, true).await;

        let detached = drive(&agent, state, None, true, true).await.unwrap();
        let checkpoint = round_trip(
            detached
                .state
                .as_ref()
                .expect("a checkpoint after the cancel"),
        );
        let reattached = drive(&agent, checkpoint, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            reattached.final_text.as_deref(),
            Some("done"),
            "streamed={streamed}"
        );

        assert_eq!(parked_pair(&session.items()), PAIR, "streamed={streamed}");
    }
}

/// Python's `test_a_filtered_unkeyed_sibling_stays_out_of_the_session` (approved and rejected).
#[tokio::test]
async fn a_filtered_unkeyed_sibling_stays_out_of_the_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for approved in [false, true] {
            let drops_preamble = |items: Vec<RunItem>| -> Vec<RunItem> {
                items
                    .into_iter()
                    .filter(|i| !matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    .collect()
            };
            let model = model_with(vec![
                vec![
                    assistant_message("PREAMBLE-THE-FILTER-REMOVED"),
                    call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
                    call("transfer_to_target", "{}", "call_HANDOFF"),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("deferred repro (unkeyed sibling)")
                .with_instructions("x")
                .with_tool(write_thing())
                .with_handoff(handoff(done_target()).with_input_filter(move |data| {
                    HandoffInputData {
                        input_history: data.input_history,
                        pre_handoff_items: drops_preamble(data.pre_handoff_items),
                        new_items: drops_preamble(data.new_items),
                        input_items: data.input_items,
                    }
                }))
                .with_output_guardrail(ok_guardrail())
                .with_tool_use_behavior(deferring())
                .with_model_instance(model);
            let session = Arc::new(ListSession::new("test"));
            let parked = drive(&agent, "go", Some(session.clone()), streamed, false)
                .await
                .unwrap();
            let mut state = round_trip(parked.state.as_ref().unwrap());
            let item = state.get_interruptions()[0].clone();
            if approved {
                state.approve(&item, false);
            } else {
                state.reject(&item, false, None);
            }
            drive(&agent, state, Some(session.clone()), streamed, false)
                .await
                .unwrap();

            let items = session.items();
            let label = format!("streamed={streamed} approved={approved}");
            assert_eq!(
                mentions(&items, "PREAMBLE-THE-FILTER-REMOVED"),
                0,
                "{label}"
            );
            assert!(
                calls_of(&items).contains(&"call_PARKED".to_string()),
                "{label}"
            );
            assert!(
                outputs_of(&items).contains(&"call_PARKED".to_string()),
                "{label}"
            );
            assert_paired(&items, &label);
        }
    }
}

/// Python's `test_handoff_filter_applies_to_sibling_completed_before_approval`.
#[tokio::test]
async fn handoff_filter_applies_to_sibling_completed_before_approval() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![vec![
            call("look_up", r#"{"query":"x"}"#, "call_LOOKUP"),
            call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
            call("transfer_to_target", "{}", "call_HANDOFF"),
        ]]);
        let agent = Agent::<()>::new("source")
            .with_tool(look_up())
            .with_tool(write_thing())
            .with_handoff(handoff(done_target()).with_input_filter(remove_all_tools))
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        let result = drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            result.final_text.as_deref(),
            Some("done"),
            "streamed={streamed}"
        );
        let items = session.items();
        assert!(
            calls_of(&items).is_empty() && outputs_of(&items).is_empty(),
            "no tool traffic survives remove_all_tools (streamed={streamed}): {items:?}"
        );
    }
}

fn terminal_agent(guardrail: Option<OutputGuardrail<()>>) -> Agent<()> {
    let model = model_with(vec![
        vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
        vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")],
    ]);
    let agent = Agent::<()>::new("deferred repro (terminal)")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["write_thing".into()],
        }))
        .with_model_instance(model);
    match guardrail {
        Some(g) => agent.with_output_guardrail(g),
        None => agent,
    }
}

/// Python's `test_detached_terminal_guardrail_failure_keeps_unsettled_checkpoint`, end-state half:
/// a detached terminal resume whose output guardrail crashes leaves the session without the
/// parked pair. (The reload-time "pending Session write is invalid" error is Python-only.)
#[tokio::test]
async fn a_detached_terminal_guardrail_failure_leaves_the_session_unsettled() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(ListSession::new("test"));
        let agent = terminal_agent(Some(ok_guardrail()));
        let state = parked_and_approved(&agent, &session, streamed).await;

        let crashing = Agent::<()>::new("deferred repro (terminal)")
            .with_tool(look_up())
            .with_tool(write_thing())
            .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                stop_at_tool_names: vec!["write_thing".into()],
            }))
            .with_output_guardrail(crashing_guardrail())
            .with_model_instance(model_with(vec![]));
        let err = drive(&crashing, state, None, streamed, false)
            .await
            .expect_err("the guardrail crashes");
        assert!(err.to_string().contains("guardrail crashed"), "{err}");
        assert!(
            parked_pair(&session.items()).is_empty(),
            "a detached run must not settle into the session (streamed={streamed})"
        );
    }
}

fn two_park_agent(behavior: ToolUseBehavior<()>, model: Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new("deferred repro (two parks)")
        .with_instructions("x")
        .with_tool(write_thing())
        .with_tool(write_other())
        .with_tool_use_behavior(behavior)
        .with_model_instance(model)
}

fn stop_at_other() -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: vec!["write_other".into()],
    })
}

/// Python's `test_later_tripwire_preserves_accepted_detached_history`.
#[tokio::test]
async fn a_later_tripwire_preserves_accepted_detached_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![
            vec![call("write_thing", r#"{"query":"a"}"#, "call_A")],
            vec![call("write_other", r#"{"query":"b"}"#, "call_B")],
            vec![assistant_message("done")],
        ]);
        let agent =
            two_park_agent(stop_at_other(), model.clone()).with_output_guardrail(ok_guardrail());
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        let detached = drive(&agent, state, None, streamed, false).await.unwrap();
        let mut state = round_trip(detached.state.as_ref().unwrap());
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);

        let tripping = two_park_agent(stop_at_other(), model.clone())
            .with_output_guardrail(tripping_guardrail());
        let err = drive(&tripping, state, Some(session.clone()), streamed, false)
            .await
            .expect_err("the output guardrail trips");
        assert!(
            matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
            "{err:?}"
        );

        let items = session.items();
        let label = format!("streamed={streamed}");
        assert_eq!(calls_of(&items), vec!["call_A", "call_B"], "{label}");
        assert_eq!(outputs_of(&items), vec!["call_A", "call_B"], "{label}");
        let output_a = items
            .iter()
            .find(|i| {
                i.item_type() == Some("function_call_output")
                    && i.as_value().get("call_id").and_then(Value::as_str) == Some("call_A")
            })
            .unwrap();
        assert_eq!(output_a.as_value()["output"], "wrote:a", "{label}");
        assert_eq!(mentions(&items, "other:b"), 0, "{label}");
    }
}

/// Python's `test_detached_repark_terminal_settle_preserves_each_response_once`.
#[tokio::test]
async fn a_detached_repark_terminal_settle_preserves_each_response_once() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![
            vec![
                assistant_message("FIRST-PREAMBLE"),
                call("write_thing", r#"{"query":"a"}"#, "call_A"),
            ],
            vec![
                assistant_message("SECOND-PREAMBLE"),
                call("write_other", r#"{"query":"b"}"#, "call_B"),
            ],
        ]);
        let agent = two_park_agent(stop_at_other(), model).with_output_guardrail(ok_guardrail());
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        let detached = drive(&agent, state, None, streamed, false).await.unwrap();
        let mut state = round_trip(detached.state.as_ref().unwrap());
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let result = drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            result.final_text.as_deref(),
            Some("other:b"),
            "streamed={streamed}"
        );

        let history = session.items();
        let label = format!("streamed={streamed}");
        assert_eq!(mentions(&history, "FIRST-PREAMBLE"), 1, "{label}");
        assert_eq!(mentions(&history, "SECOND-PREAMBLE"), 1, "{label}");
        assert_eq!(calls_of(&history), vec!["call_A", "call_B"], "{label}");
        assert_paired(&history, &label);
    }
}

/// Python's `test_an_additive_filter_keeps_the_batchs_companions`.
#[tokio::test]
async fn an_additive_filter_keeps_the_batchs_companions() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![
            vec![
                assistant_message("COMPANION-KEPT-BY-FILTER"),
                call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
                call("transfer_to_target", "{}", "call_HANDOFF"),
            ],
            vec![assistant_message("done")],
        ]);
        let agent = Agent::<()>::new("deferred repro (additive filter)")
            .with_instructions("x")
            .with_tool(write_thing())
            .with_handoff(handoff(done_target()).with_input_filter(|data| {
                let injected = OutputItemMessage::new("INJECTED-BY-FILTER").0;
                let mut new_items = data.new_items;
                new_items.push(injected);
                HandoffInputData { new_items, ..data }
            }))
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(ListSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let items = session.items();
        let label = format!("streamed={streamed}");
        assert!(mentions(&items, "COMPANION-KEPT-BY-FILTER") > 0, "{label}");
        assert!(
            calls_of(&items).contains(&"call_PARKED".to_string()),
            "{label}"
        );
        assert!(
            outputs_of(&items).contains(&"call_PARKED".to_string()),
            "{label}"
        );
    }
}

struct OutputItemMessage(RunItem);

impl OutputItemMessage {
    fn new(text: &str) -> Self {
        let openai_agents_sdk::OutputItem::Message(raw_item) = assistant_message(text) else {
            unreachable!("assistant_message builds a message")
        };
        OutputItemMessage(RunItem::new(
            "filler",
            RunItemKind::MessageOutput { raw_item },
        ))
    }
}

/// Python's `test_handoff_filter_distinguishes_equal_preambles` (3 cases x streamed).
#[tokio::test]
async fn a_handoff_filter_distinguishes_equal_preambles() {
    openai_agents_sdk::set_tracing_disabled(true);
    let cases: [(&str, usize, i32, usize); 3] = [
        ("SAME PREAMBLE", 1, -1, 1),
        ("SAME PREAMBLE", 1, 0, 2),
        ("EARLIER PREAMBLE", 2, -1, 1),
    ];
    for streamed in [false, true] {
        for (prior_text, current_count, removed_index, expected_count) in cases {
            let mut turn_two = Vec::new();
            for _ in 0..current_count {
                turn_two.push(assistant_message("SAME PREAMBLE"));
            }
            turn_two.push(call("write_thing", r#"{"query":"x"}"#, "write"));
            turn_two.push(call("transfer_to_target", "{}", "handoff"));
            let model = model_with(vec![
                vec![
                    assistant_message(prior_text),
                    call("look_up", r#"{"query":"x"}"#, "lookup"),
                ],
                turn_two,
            ]);
            let filter = move |data: HandoffInputData| {
                let messages: Vec<usize> = data
                    .pre_handoff_items
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    .map(|(idx, _)| idx)
                    .collect();
                let removed = if removed_index < 0 {
                    messages.len().checked_sub(1)
                } else {
                    Some(removed_index as usize)
                }
                .and_then(|i| messages.get(i).copied());
                HandoffInputData {
                    pre_handoff_items: data
                        .pre_handoff_items
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| Some(*i) != removed)
                        .map(|(_, item)| item)
                        .collect(),
                    ..data
                }
            };
            let agent = Agent::<()>::new("source")
                .with_tool(look_up())
                .with_tool(write_thing())
                .with_handoff(handoff(done_target()).with_input_filter(filter))
                .with_tool_use_behavior(deferring())
                .with_output_guardrail(ok_guardrail())
                .with_model_instance(model);
            let session = Arc::new(ListSession::new("test"));
            let parked = drive(&agent, "go", Some(session.clone()), streamed, false)
                .await
                .unwrap();
            assert_eq!(mentions(&session.items(), prior_text), 1);
            let mut state = round_trip(parked.state.as_ref().unwrap());
            let item = state.get_interruptions()[0].clone();
            state.approve(&item, false);
            drive(&agent, state, Some(session.clone()), streamed, false)
                .await
                .unwrap();

            assert_eq!(
                mentions(&session.items(), "SAME PREAMBLE"),
                expected_count,
                "streamed={streamed} prior={prior_text} removed={removed_index}"
            );
        }
    }
}
