//! Parity port of the resume tail of `test_resume.py` and the portable parts of
//! `test_recovery.py` (`tests/deferred_session_writes/`).
//!
//! Python inspects `pending_session_write` / `_pending_session_write` and wrapper-aware
//! sessions; those assertions are Python-only. What is ported is the observable end state: the
//! session's content and the run outcome after interrupt, approve, failure and resume.
//!
//! Python's `custom_data_extractor` raises to simulate a post-output callback failure. Rust's
//! extractor returns `Option<Value>` and cannot raise, so the equivalent failure at the same
//! boundary is an extractor that returns a non-object, which the runner rejects with
//! "custom_data_extractor must return a JSON object or None".

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::{HandoffInputData, handoff};
use openai_agents_sdk::items::{RunInput, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::memory::{Session, SessionSettings};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, OutputItem,
    RunOptions, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

const PREAMBLE: &str = "About to write the thing.";

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

/// Python's failing-extractor tool: the approved call succeeds, its post-output step fails.
fn failing_write_thing() -> FunctionTool<()> {
    write_thing().with_custom_data_extractor(|_ctx| async { Some(json!("extractor boom")) })
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

fn failing_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("fails", |_c, _a, _o: Value| async {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("output guardrail failed"))
    })
}

fn model_with(turns: Vec<Vec<OutputItem>>) -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    for turn in turns {
        model.add_turn(turn);
    }
    model
}

fn call(name: &str, args: &str, id: &str) -> OutputItem {
    function_call_with_id(name, args, id)
}

fn done_target() -> Agent<()> {
    Agent::<()>::new("target")
        .with_instructions("x")
        .with_model_instance(model_with(vec![vec![assistant_message("done")]]))
}

/// A session recording each `add_items` batch (Python's `_AppendRecordingSession`) that can be
/// told to fail the next append (Python's `_FailingResumeSession`).
#[derive(Debug)]
struct RecordingSession {
    inner: ListSession,
    batches: Mutex<Vec<Vec<InputItem>>>,
    failure: Mutex<Option<&'static str>>,
    id: String,
}

impl RecordingSession {
    fn new(id: &str) -> Self {
        RecordingSession {
            inner: ListSession::new(id),
            batches: Mutex::new(Vec::new()),
            failure: Mutex::new(None),
            id: id.to_string(),
        }
    }

    fn fail_next(&self, mode: &'static str) {
        *self.failure.lock().unwrap() = Some(mode);
    }

    fn items(&self) -> Vec<InputItem> {
        self.inner.items()
    }

    fn batch_count(&self) -> usize {
        self.batches.lock().unwrap().len()
    }

    fn batches_since(&self, start: usize) -> Vec<Vec<InputItem>> {
        self.batches.lock().unwrap()[start..].to_vec()
    }
}

#[async_trait]
impl Session for RecordingSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let failure = self.failure.lock().unwrap().take();
        if failure == Some("before") {
            return Err(AgentsError::other("session append failed"));
        }
        self.batches.lock().unwrap().push(items.clone());
        self.inner.add_items(items).await?;
        if failure == Some("after") {
            return Err(AgentsError::other("session append failed"));
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
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
    session: Option<Arc<RecordingSession>>,
    streamed: bool,
    cancel_after_turn: bool,
) -> std::result::Result<Outcome, AgentsError> {
    drive_with(agent, input, session, streamed, cancel_after_turn, None).await
}

/// `drive` for a run that may fail: a failure leaves its checkpoint in `slot`.
async fn drive_slot(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<RecordingSession>>,
    streamed: bool,
    slot: &RunCheckpoint,
) -> std::result::Result<Outcome, AgentsError> {
    drive_with(agent, input, session, streamed, false, Some(slot)).await
}

async fn drive_with(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<RecordingSession>>,
    streamed: bool,
    cancel_after_turn: bool,
    slot: Option<&RunCheckpoint>,
) -> std::result::Result<Outcome, AgentsError> {
    let mut options = RunOptions::new(());
    if let Some(slot) = slot {
        options = options.checkpoint(slot);
    }
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

/// Python mutates the `RunState` it was handed, so its retry serializes that same object after
/// the failure. A Rust run consumes its state; the failed run leaves its checkpoint in the
/// `RunCheckpoint` slot it was given.
fn checkpoint_of(slot: &RunCheckpoint) -> RunState {
    round_trip(&slot.take().expect("a failed run leaves its checkpoint"))
}

async fn parked_and_approved(
    agent: &Agent<()>,
    session: &Arc<RecordingSession>,
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

fn ids(items: &[InputItem], ty: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some(ty))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn calls_of(items: &[InputItem]) -> Vec<String> {
    ids(items, "function_call")
}

fn outputs_of(items: &[InputItem]) -> Vec<String> {
    ids(items, "function_call_output")
}

fn assert_paired(items: &[InputItem], label: &str) {
    let (calls, outputs) = (calls_of(items), outputs_of(items));
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

fn deferring_agent() -> Agent<()> {
    let model = model_with(vec![
        vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
        vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")],
        vec![assistant_message("done")],
    ]);
    Agent::<()>::new("deferred repro")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(deferring())
        .with_model_instance(model)
}

fn multi_agent(behavior: ToolUseBehavior<()>) -> Agent<()> {
    let model = model_with(vec![
        vec![
            call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
            call("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
        ],
        vec![assistant_message("done")],
    ]);
    Agent::<()>::new("deferred repro (multi)")
        .with_tool(write_thing())
        .with_tool(write_other())
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(behavior)
        .with_model_instance(model)
}

fn terminal_agent(with_guardrails: bool, with_preamble: bool) -> Agent<()> {
    let mut parked = vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")];
    if with_preamble {
        parked.insert(0, assistant_message(PREAMBLE));
    }
    let model = model_with(vec![
        vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
        parked,
    ]);
    let agent = Agent::<()>::new("deferred repro (terminal)")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_tool_use_behavior(stop_at(&["write_thing"]))
        .with_model_instance(model);
    if with_guardrails {
        agent.with_output_guardrail(ok_guardrail())
    } else {
        agent
    }
}

/// Python's `test_a_detached_resume_does_not_make_the_next_one_rewrite_the_session`.
#[tokio::test]
async fn a_detached_resume_does_not_make_the_next_one_rewrite_the_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let agent = multi_agent(ToolUseBehavior::RunLlmAgain);
    let session = Arc::new(RecordingSession::new("test"));

    let parked = drive(&agent, "go", Some(session.clone()), true, false)
        .await
        .unwrap();
    assert_eq!(parked.interruptions.len(), 2);
    assert!(calls_of(&session.items()).contains(&"call_PARKED".to_string()));

    let mut state = round_trip(parked.state.as_ref().unwrap());
    let one = state
        .get_interruptions()
        .into_iter()
        .find(|i| i.raw_item.call_id == "call_PARKED")
        .unwrap();
    state.approve(&one, false);
    let detached = drive(&agent, state, None, true, false).await.unwrap();

    let mut state = round_trip(detached.state.as_ref().unwrap());
    for i in state.get_interruptions() {
        state.approve(&i, false);
    }
    drive(&agent, state, Some(session.clone()), true, false)
        .await
        .unwrap();

    let calls = calls_of(&session.items());
    assert_eq!(calls.iter().filter(|c| *c == "call_PARKED").count(), 1);
    assert_eq!(calls.iter().filter(|c| *c == "call_PARKED_2").count(), 1);
}

/// Python's `test_a_held_resume_with_a_different_session_is_refused`.
///
/// Python refuses the resume (`UserError: pending Session write`) because the withheld batch
/// belongs to the first session. Rust writes the parked call eagerly, so it cannot refuse on
/// that basis and writes the approved output into the other session.
#[tokio::test]
async fn a_held_resume_with_a_different_session_is_refused() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(RecordingSession::new("test"));
        let agent = deferring_agent();
        let state = parked_and_approved(&agent, &session, streamed).await;

        let other = Arc::new(RecordingSession::new("other"));
        let err = drive(&agent, state, Some(other.clone()), streamed, false)
            .await
            .expect_err("resuming against another session must be refused");
        assert!(err.to_string().contains("pending Session write"), "{err}");
        assert!(other.items().is_empty(), "streamed={streamed}");
    }
}

/// Python's `test_after_turn_cancel_keeps_the_held_batch_for_the_next_attach`, end-state half.
#[tokio::test]
async fn an_after_turn_cancel_keeps_the_executed_output_for_the_next_attach() {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(RecordingSession::new("test"));
    let agent = deferring_agent();
    let state = parked_and_approved(&agent, &session, true).await;

    let detached = drive(&agent, state, None, true, true).await.unwrap();
    let checkpoint = round_trip(detached.state.as_ref().unwrap());
    drive(&agent, checkpoint, Some(session.clone()), true, false)
        .await
        .unwrap();

    let items = session.items();
    assert_paired(&items, "reattach");
    assert_eq!(parked_pair(&items), PAIR);
}

/// Python's `test_the_settled_batch_and_the_resolved_turn_land_as_one_ordered_write`.
#[tokio::test]
async fn the_settled_batch_and_the_resolved_turn_land_as_one_ordered_write() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(RecordingSession::new("test"));
        let agent = deferring_agent();
        let state = parked_and_approved(&agent, &session, streamed).await;
        let before = session.batch_count();

        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let settling: Vec<Vec<InputItem>> = session
            .batches_since(before)
            .into_iter()
            .filter(|b| {
                calls_of(b).contains(&"call_PARKED".to_string())
                    || outputs_of(b).contains(&"call_PARKED".to_string())
            })
            .collect();
        assert_eq!(settling.len(), 1, "streamed={streamed}");
        assert_eq!(parked_pair(&settling[0]), PAIR, "streamed={streamed}");
    }
}

/// Python's `test_entry_settle_drops_a_held_call_the_filter_unpaired`: the partial-filter
/// handoff, a detached after-turn cancel, and a reattach must not leave a dangling call.
#[tokio::test]
async fn entry_settle_drops_a_held_call_the_filter_unpaired() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = model_with(vec![
        vec![
            call("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
            call("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
            call("transfer_to_target", "{}", "call_HANDOFF"),
        ],
        vec![assistant_message("done")],
    ]);
    let drops = |items: Vec<openai_agents_sdk::items::RunItem>| -> Vec<_> {
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
    let session = Arc::new(RecordingSession::new("test"));
    let first = drive(&agent, "go", Some(session.clone()), true, false)
        .await
        .unwrap();
    let mut state = round_trip(first.state.as_ref().unwrap());
    for i in state.get_interruptions() {
        state.approve(&i, false);
    }
    let detached = drive(&agent, state, None, true, true).await.unwrap();
    drive(
        &agent,
        detached.state.expect("checkpoint after cancel"),
        Some(session.clone()),
        true,
        false,
    )
    .await
    .unwrap();

    let items = session.items();
    assert_paired(&items, "entry settle");
    assert!(calls_of(&items).contains(&"call_PARKED".to_string()));
    assert!(outputs_of(&items).contains(&"call_PARKED".to_string()));
}

/// Python's `test_a_detached_completion_clears_the_held_record`, outcome half (the
/// `pending_session_write` JSON assertions are Python-only).
#[tokio::test]
async fn a_detached_completion_runs_to_the_final_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(RecordingSession::new("test"));
        let agent = deferring_agent();
        let state = parked_and_approved(&agent, &session, streamed).await;

        let detached = drive(&agent, state, None, streamed, false).await.unwrap();
        assert_eq!(
            detached.final_text.as_deref(),
            Some("done"),
            "streamed={streamed}"
        );
        // A completed detached run leaves a checkpoint that reloads cleanly.
        let reloaded = RunState::from_json(detached.state.unwrap().to_json().unwrap());
        assert!(reloaded.is_ok(), "streamed={streamed}");
    }
}

/// Python's `test_a_terminal_resume_with_a_preamble_lands_it_once`.
#[tokio::test]
async fn a_terminal_resume_with_a_preamble_lands_it_once() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(RecordingSession::new("test"));
        let agent = terminal_agent(true, true);
        let state = parked_and_approved(&agent, &session, streamed).await;
        drive(&agent, state, Some(session.clone()), streamed, false)
            .await
            .unwrap();

        let items = session.items();
        assert_paired(&items, &format!("streamed={streamed}"));
        assert_eq!(parked_pair(&items), PAIR, "streamed={streamed}");
        assert_eq!(mentions(&items, PREAMBLE), 1, "streamed={streamed}");
    }
}

/// Python's `test_a_failed_detached_completion_keeps_the_held_record` (final turn on the resumed
/// turn, or a later one): a failing output guardrail on a detached resume surfaces its error and
/// leaves the original checkpoint resumable, so a reattach with healthy guardrails completes and
/// stores the pair.
#[tokio::test]
async fn a_failed_detached_completion_leaves_the_checkpoint_resumable() {
    openai_agents_sdk::set_tracing_disabled(true);
    for later_turn in [false, true] {
        for streamed in [false, true] {
            let mut turns = vec![
                vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
                vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")],
            ];
            if later_turn {
                turns.push(vec![call("look_up", r#"{"query":"y"}"#, "call_AFTER")]);
            }
            turns.push(vec![assistant_message("done")]);
            // One scripted model is shared by the parked, failing and healthy agents, as in
            // Python where the same agent's guardrails are swapped between attempts.
            let model = model_with(turns);
            let make = || {
                Agent::<()>::new("deferred repro")
                    .with_tool(look_up())
                    .with_tool(write_thing())
                    .with_tool_use_behavior(deferring())
                    .with_model_instance(model.clone())
            };
            let label = format!("later_turn={later_turn} streamed={streamed}");
            let session = Arc::new(RecordingSession::new("test"));
            let agent = make().with_output_guardrail(ok_guardrail());
            let state = parked_and_approved(&agent, &session, streamed).await;
            let retry_state = round_trip(&state);

            let failing = make().with_output_guardrail(failing_guardrail());
            let err = drive(&failing, state, None, streamed, false)
                .await
                .expect_err("the output guardrail fails");
            assert!(
                err.to_string().contains("output guardrail failed"),
                "{label}: {err}"
            );

            // The failed attempt consumed the model's remaining turns; queue them again for the retry.
            if later_turn {
                model.add_turn(vec![call("look_up", r#"{"query":"y"}"#, "call_AFTER")]);
            }
            model.add_turn(vec![assistant_message("done")]);
            let healthy = make().with_output_guardrail(ok_guardrail());
            let retried = drive(
                &healthy,
                retry_state,
                Some(session.clone()),
                streamed,
                false,
            )
            .await
            .unwrap();
            assert_eq!(retried.final_text.as_deref(), Some("done"), "{label}");
            assert_eq!(parked_pair(&session.items()), PAIR, "{label}");
        }
    }
}

/// Python's `test_a_failed_settle_of_the_held_batch_is_recovered_on_the_next_resume` is ported in
/// `py_deferred_writes_b.rs`.
///
/// Python's `test_a_failed_final_settle_fails_closed_with_the_batch_recorded`, outcome half: a
/// session append that fails during a terminal resume surfaces its error. (The recorded
/// `_pending_session_write` and the fail-closed reload error are Python-only.)
#[tokio::test]
async fn a_failed_final_settle_surfaces_the_append_error() {
    openai_agents_sdk::set_tracing_disabled(true);
    for with_guardrails_and_preamble in [false, true] {
        for streamed in [false, true] {
            let session = Arc::new(RecordingSession::new("test"));
            let agent = terminal_agent(with_guardrails_and_preamble, with_guardrails_and_preamble);
            let state = parked_and_approved(&agent, &session, streamed).await;

            session.fail_next("before");
            let slot = RunCheckpoint::new();
            let err = drive_slot(&agent, state, Some(session.clone()), streamed, &slot)
                .await
                .expect_err("the append must fail");
            assert!(
                err.to_string().contains("session append failed"),
                "guarded={with_guardrails_and_preamble} streamed={streamed}: {err}"
            );
        }
    }
}

/// Python's `test_a_post_output_callback_failure_keeps_the_executed_output`, via an extractor
/// that returns a non-object (Rust's extractor cannot raise). The retry keeps the failing
/// extractor and only drops the guardrails: the completed tool supplies the final output without
/// producing new items, and would fail again if the retry executed it a second time.
#[tokio::test]
async fn a_post_output_callback_failure_keeps_the_executed_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = model_with(vec![
            vec![call("look_up", r#"{"query":"x"}"#, "call_LOOKUP")],
            vec![call("write_thing", r#"{"query":"x"}"#, "call_PARKED")],
            vec![assistant_message("done")],
        ]);
        let make = || {
            Agent::<()>::new("deferred repro (failing extractor)")
                .with_tool(look_up())
                .with_tool(failing_write_thing())
                .with_tool_use_behavior(stop_at(&["write_thing"]))
                .with_model_instance(model.clone())
        };
        let session = Arc::new(RecordingSession::new("test"));
        let agent = make().with_output_guardrail(ok_guardrail());
        let state = parked_and_approved(&agent, &session, streamed).await;

        let slot = RunCheckpoint::new();
        let err = drive_slot(&agent, state, Some(session.clone()), streamed, &slot)
            .await
            .expect_err("the extractor result is rejected");
        assert!(
            err.to_string()
                .contains("custom_data_extractor must return a JSON object"),
            "streamed={streamed}: {err}"
        );
        let retry_state = checkpoint_of(&slot);

        let result = drive(&make(), retry_state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            result.final_text.as_deref(),
            Some("wrote:x"),
            "streamed={streamed}"
        );
        assert_eq!(parked_pair(&session.items()), PAIR, "streamed={streamed}");
    }
}

/// Python's `test_the_filters_authority_survives_a_json_retry_of_the_crashed_turn`: a secret
/// tool resolved into a handoff with `remove_all_tools`, whose post-output step fails; the JSON
/// retry must still keep the secret out of the session.
#[tokio::test]
async fn the_filters_authority_survives_a_json_retry_of_the_crashed_turn() {
    use openai_agents_sdk::extensions::handoff_filters::remove_all_tools;
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let secret = FunctionTool::new("read_secret", "secret", |_ctx, _q: Query| async move {
            Ok("SECRET-VALUE-42".to_string())
        })
        .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
        .with_custom_data_extractor(|_ctx| async { Some(json!("extractor boom")) });
        let agent = Agent::<()>::new("deferred repro (secret + failing extractor)")
            .with_instructions("x")
            .with_tool(secret)
            .with_handoff(handoff(done_target()).with_input_filter(remove_all_tools))
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model_with(vec![
                vec![
                    call("read_secret", r#"{"query":"x"}"#, "call_SECRET"),
                    call("transfer_to_target", "{}", "call_HANDOFF"),
                ],
                vec![assistant_message("done")],
            ]));
        let session = Arc::new(RecordingSession::new("test"));
        let state = parked_and_approved(&agent, &session, streamed).await;

        let slot = RunCheckpoint::new();
        let err = drive_slot(&agent, state, Some(session.clone()), streamed, &slot)
            .await
            .expect_err("the extractor result is rejected");
        assert!(err.to_string().contains("custom_data_extractor"), "{err}");
        let reloaded = checkpoint_of(&slot);

        drive(&agent, reloaded, Some(session.clone()), streamed, false)
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

/// Python's `test_checkpointed_mcp_approval_survives_callback_failure_and_json_retry`: a hosted
/// MCP approval and a local gated call share a response; the first final-output decision fails,
/// and the JSON retry must execute the local tool once and store the MCP request/response pair.
#[tokio::test]
async fn a_checkpointed_mcp_approval_survives_callback_failure_and_json_retry() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let executions = Arc::new(Mutex::new(0usize));
        let decisions = Arc::new(Mutex::new(0usize));
        let counted = executions.clone();
        let write_once = FunctionTool::new("write_once", "writes", move |_ctx, _q: Query| {
            let counted = counted.clone();
            async move {
                *counted.lock().unwrap() += 1;
                Ok("written".to_string())
            }
        })
        .with_needs_approval_fn(|_ctx, _args, _id| async move { true });
        let seen = decisions.clone();
        let decide = ToolUseBehavior::custom(move |_ctx, _results| {
            let seen = seen.clone();
            async move {
                let mut n = seen.lock().unwrap();
                *n += 1;
                if *n == 1 {
                    Err(AgentsError::other("decision failed"))
                } else {
                    Ok(ToolsToFinalOutputResult {
                        is_final_output: false,
                        final_output: None,
                    })
                }
            }
        });
        let request = OutputItem::Other(json!({
            "type": "mcp_approval_request",
            "id": "mcpr_test",
            "server_label": "srv",
            "name": "remote",
            "arguments": "{}",
        }));
        let agent = Agent::<()>::new("agent")
            .with_tool(write_once)
            .with_tool(HostedMCPTool::new(json!({
                "type": "mcp",
                "server_label": "srv",
                "server_url": "https://example.com",
                "require_approval": "always",
            })))
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(decide)
            .with_model_instance(model_with(vec![
                vec![
                    call("write_once", r#"{"query":"x"}"#, "local_write"),
                    request,
                ],
                vec![assistant_message("done")],
            ]));
        let session = Arc::new(RecordingSession::new("test"));
        let parked = drive(&agent, "go", Some(session.clone()), false, false)
            .await
            .unwrap();
        let mut state = round_trip(parked.state.as_ref().unwrap());
        for approval in state.get_interruptions() {
            state.approve(&approval, false);
        }

        let slot = RunCheckpoint::new();
        let err = drive_slot(&agent, state, Some(session.clone()), streamed, &slot)
            .await
            .expect_err("the first decision fails");
        assert!(
            err.to_string().contains("decision failed"),
            "streamed={streamed}: {err}"
        );
        let retry_state = checkpoint_of(&slot);

        let result = drive(&agent, retry_state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            result.final_text.as_deref(),
            Some("done"),
            "streamed={streamed}"
        );
        assert_eq!(
            *executions.lock().unwrap(),
            1,
            "the local tool ran once (streamed={streamed})"
        );

        let history = session.items();
        let mcp: Vec<&InputItem> = history
            .iter()
            .filter(|i| {
                matches!(
                    i.item_type(),
                    Some("mcp_approval_request") | Some("mcp_approval_response")
                )
            })
            .collect();
        let kinds: Vec<&str> = mcp.iter().filter_map(|i| i.item_type()).collect();
        assert_eq!(
            kinds,
            vec!["mcp_approval_request", "mcp_approval_response"],
            "streamed={streamed}"
        );
        assert_eq!(mcp[0].as_value()["id"], "mcpr_test");
        assert_eq!(mcp[1].as_value()["approval_request_id"], "mcpr_test");
        assert_eq!(mcp[1].as_value()["approve"], true);
    }
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// Python's `test_callback_failure_preserves_model_order_for_completed_outputs`: two approved
/// calls in one response complete out of model order (the second commits first), the first one's
/// post-output step fails, and the JSON retry must store the outputs in the model's call order
/// with each tool executed once.
#[tokio::test]
async fn a_callback_failure_preserves_model_order_for_completed_outputs() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let second_committed = Arc::new(tokio::sync::Notify::new());
        let second_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let executions = Arc::new(Mutex::new(Vec::<String>::new()));

        let (waiting, flag, log) = (
            second_committed.clone(),
            second_done.clone(),
            executions.clone(),
        );
        let first = FunctionTool::new("first", "first", move |_ctx, _a: NoArgs| {
            let (waiting, flag, log) = (waiting.clone(), flag.clone(), log.clone());
            async move {
                while !flag.load(std::sync::atomic::Ordering::SeqCst) {
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_millis(5),
                        waiting.notified(),
                    )
                    .await;
                }
                log.lock().unwrap().push("first".into());
                Ok("first result".to_string())
            }
        })
        .with_needs_approval_fn(|_c, _a, _i| async move { true })
        .with_custom_data_extractor(|_ctx| async { Some(json!("extractor boom")) });

        let (signal, flag, log) = (
            second_committed.clone(),
            second_done.clone(),
            executions.clone(),
        );
        let second = FunctionTool::new("second", "second", move |_ctx, _a: NoArgs| {
            let log = log.clone();
            async move {
                log.lock().unwrap().push("second".into());
                Ok("second result".to_string())
            }
        })
        .with_needs_approval_fn(|_c, _a, _i| async move { true })
        .with_custom_data_extractor(move |_ctx| {
            let (signal, flag) = (signal.clone(), flag.clone());
            async move {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                signal.notify_waiters();
                None
            }
        });

        let agent = Agent::<()>::new("ordered callback recovery")
            .with_tool(first)
            .with_tool(second)
            .with_output_guardrail(ok_guardrail())
            .with_tool_use_behavior(stop_at(&["first"]))
            .with_model_instance(model_with(vec![vec![
                call("first", "{}", "call_FIRST"),
                call("second", "{}", "call_SECOND"),
            ]]));
        let session = Arc::new(RecordingSession::new("test"));
        let parked = drive(&agent, "go", Some(session.clone()), streamed, false)
            .await
            .unwrap();
        let mut state = round_trip(parked.state.as_ref().unwrap());
        for i in state.get_interruptions() {
            state.approve(&i, false);
        }

        let slot = RunCheckpoint::new();
        let err = drive_slot(&agent, state, Some(session.clone()), streamed, &slot)
            .await
            .expect_err("the first tool's extractor result is rejected");
        assert!(err.to_string().contains("custom_data_extractor"), "{err}");
        let retry_state = checkpoint_of(&slot);

        let result = drive(&agent, retry_state, Some(session.clone()), streamed, false)
            .await
            .unwrap();
        assert_eq!(
            result.final_text.as_deref(),
            Some("first result"),
            "streamed={streamed}"
        );
        assert_eq!(
            *executions.lock().unwrap(),
            vec!["second", "first"],
            "each tool runs once and completion order differs from call order (streamed={streamed})"
        );

        let outputs: Vec<(String, String)> = session
            .items()
            .iter()
            .filter(|i| i.item_type() == Some("function_call_output"))
            .map(|i| {
                (
                    i.as_value()["call_id"].as_str().unwrap().to_string(),
                    i.as_value()["output"].as_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        assert_eq!(
            outputs,
            vec![
                ("call_FIRST".to_string(), "first result".to_string()),
                ("call_SECOND".to_string(), "second result".to_string()),
            ],
            "streamed={streamed}"
        );
    }
}
