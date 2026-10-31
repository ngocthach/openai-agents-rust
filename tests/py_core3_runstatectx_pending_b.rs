//! Port of `tests/test_run_state_pending_input.py`, second half: server-managed resumes, input
//! filters, duplicate-call refusals after acceptance, and the terminal-state guard.
//!
//! Python-only (not ported): the hook-failure cases of
//! `test_server_response_acceptance_commits_before_hook_failure`,
//! `test_server_accepted_computer_start_hook_failure_is_not_replayed` and
//! `test_server_accepted_tool_side_effect_failure_is_safe`. Python hooks raise to fail a run;
//! Rust lifecycle hooks return `()`, so no hook can fail a Rust run and the recovery paths those
//! tests describe have no Rust trigger. The `response_accepted` and `llm_end_hooks_started` flags
//! they read are Python-private step fields.
//!
//! Python's `RunState` is mutated by the run it resumes; a Rust run consumes its state. Where a
//! test reads the state after a run, this file reads `result.to_state()`, and a failed run is read
//! through a [`RunCheckpoint`](openai_agents_sdk::run::RunCheckpoint).

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{FunctionCall, InputItem, OutputItem, RunInput};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, ModelInputData, RunItemStreamEventName, RunResult, Runner, StreamEvent,
};
use serde_json::{Value, json};

type Calls = Arc<Mutex<Vec<String>>>;

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn options(session: Option<Arc<dyn Session>>, auto_previous_response_id: bool) -> RunOptions<()> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.session = session;
    options.auto_previous_response_id = auto_previous_response_id;
    options
}

/// Text of a user message, the way Python's `_message_text` reads it.
fn message_text(item: &InputItem) -> Option<String> {
    let obj = item.0.as_object()?;
    if obj.get("role")?.as_str()? != "user" {
        return None;
    }
    match obj.get("content")? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter(|p| {
                    matches!(
                        p.get("type").and_then(Value::as_str),
                        Some("input_text" | "output_text")
                    )
                })
                .map(|p| {
                    p.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                })
                .collect(),
        ),
        _ => None,
    }
}

/// `type` of an item, or its role for a message without one (`_item_type` in Python).
fn count_text(items: &[InputItem], text: &str) -> usize {
    items
        .iter()
        .filter(|i| message_text(i).as_deref() == Some(text))
        .count()
}

fn destination_tool(calls: &Calls) -> FunctionTool<()> {
    let calls = calls.clone();
    FunctionTool::<()>::new(
        "record_destination",
        "Records a destination",
        move |_c, a: Value| {
            let calls = calls.clone();
            async move {
                let destination = a["destination"].as_str().unwrap_or_default().to_string();
                calls.lock().unwrap().push(destination.clone());
                Ok(format!("recorded:{destination}"))
            }
        },
    )
}

fn destination_call(destination: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(
        "record_destination",
        json!({"destination": destination}).to_string(),
        call_id,
    )
}

/// A run stopped after its first tool output, the way Python's `_make_after_turn_state` leaves it.
struct AfterTurn {
    model: Arc<FakeModel>,
    agent: Arc<Agent<()>>,
    state: RunState,
    calls: Calls,
}

async fn after_turn_state(
    session: Option<Arc<dyn Session>>,
    auto_previous_response_id: bool,
) -> AfterTurn {
    let calls: Calls = Arc::default();
    let model = fake();
    model.add_turn(vec![destination_call("Paris", "call-destination")]);
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model.clone())
            .with_tools(vec![destination_tool(&calls)]),
    );
    let mut streamed = Runner::run_streamed_with_options(
        &agent,
        "Initial request",
        options(session, auto_previous_response_id),
    );
    let handle = streamed.cancel_handle();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            } = event.unwrap()
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    let state = streamed
        .to_state()
        .expect("a cancelled run leaves its state");
    if std::env::var("C3_PROBE").is_ok() {
        eprintln!(
            "PROBE3 final={:?} turn={} step_items={} model_calls={}",
            state.final_output,
            state.current_turn,
            state.generated_items.len(),
            model.remaining_steps()
        );
    }
    assert_eq!(*calls.lock().unwrap(), vec!["Paris".to_string()]);
    AfterTurn {
        model,
        agent,
        state,
        calls,
    }
}

/// Python's `state.to_json()` snapshot with the envelope a Rust reader needs; returns the value.
fn snapshot(state: &RunState) -> Value {
    state.to_json().unwrap()
}

// ---------------------------------------------------------------------------

/// Run a streamed resume until its first tool output, then cancel after that turn, and return the
/// state it leaves. Python's `cancel(mode="after_turn")` loop.
/// Run to completion, streamed or not, and return the result.
async fn run_to_end(
    agent: &Arc<Agent<()>>,
    state: RunState,
    streamed: bool,
    session: Option<Arc<dyn Session>>,
) -> Result<RunResult<()>, AgentsError> {
    if streamed {
        let mut result = Runner::run_streamed_with_options(
            agent,
            RunInput::from(state),
            options(session, false),
        );
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        return Ok(result
            .into_result()
            .expect("completed stream holds its result"));
    }
    Runner::run_with_options(agent, RunInput::from(state), options(session, false)).await
}

fn texts(items: &[InputItem]) -> Vec<Option<String>> {
    items.iter().map(message_text).collect()
}

fn admitted_inputs(state: &RunState) -> Vec<Option<String>> {
    state
        .generated_items
        .iter()
        .filter_map(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::Input { raw_item } => {
                Some(message_text(raw_item))
            }
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum StopName {
    Finish,
    WorkflowFinish,
    Unrelated,
}

async fn completed_stop_sibling(
    streamed: bool,
    serialized: bool,
    stop: StopName,
    has_handoff: bool,
) {
    let calls: Calls = Arc::default();
    let finish_log = calls.clone();
    let finish = FunctionTool::<()>::new_no_args("finish", "Finish the workflow", move |_c| {
        let log = finish_log.clone();
        async move {
            log.lock().unwrap().push("finish".into());
            Ok("Finished".to_string())
        }
    });
    let protected_log = calls.clone();
    let protected =
        FunctionTool::<()>::new_no_args("protected_tool", "Needs approval", move |_c| {
            let log = protected_log.clone();
            async move {
                log.lock().unwrap().push("protected".into());
                Ok("Approved".to_string())
            }
        })
        .with_needs_approval(true);
    let stop_name = match stop {
        StopName::Finish => "finish",
        StopName::WorkflowFinish => "workflow.finish",
        StopName::Unrelated => "unrelated",
    };
    let model = fake();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let mut namespaced = FunctionCall::new("finish", "{}");
    namespaced.call_id = "finish-call".into();
    namespaced
        .extra
        .insert("namespace".into(), json!("workflow"));
    let mut turn = vec![
        OutputItem::FunctionCall(namespaced),
        function_call_with_id("protected_tool", "{}", "protected-call"),
    ];
    if has_handoff {
        turn.push(function_call_with_id(
            "transfer_to_delegate",
            "{}",
            "handoff-call",
        ));
    }
    model.add_turn(turn);
    model.add_turn(vec![assistant_message("Continued")]);

    let mut tools =
        openai_agents_sdk::tool::tool_namespace("workflow", "Workflow tools", vec![finish])
            .unwrap();
    tools.push(protected);
    let mut agent = Agent::<()>::new("assistant")
        .with_model_instance(model.clone())
        .with_tools(tools)
        .with_tool_use_behavior(openai_agents_sdk::agent::ToolUseBehavior::StopAtTools(
            openai_agents_sdk::agent::StopAtTools {
                stop_at_tool_names: vec![stop_name.into()],
            },
        ));
    if has_handoff {
        agent = agent.with_handoffs(vec![openai_agents_sdk::handoff(delegate.clone())]);
    }
    let agent = Arc::new(agent);

    let interrupted = if streamed {
        let mut result =
            Runner::run_streamed_with_options(&agent, "Initial request", options(None, false));
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        result.into_result().unwrap()
    } else {
        Runner::run_with_options(&agent, "Initial request", options(None, false))
            .await
            .unwrap()
    };
    let mut state = interrupted.to_state();
    if serialized {
        state = RunState::from_json(state.to_json().unwrap()).unwrap();
    }
    assert_eq!(*calls.lock().unwrap(), vec!["finish".to_string()]);
    let before = snapshot(&state);
    let continues = matches!(stop, StopName::Unrelated) || has_handoff;
    if continues {
        state.add_input("Late input").unwrap();
    } else {
        let err = state.add_input("Late input").unwrap_err();
        assert!(
            err.to_string().contains("tool result may end the run"),
            "{err}"
        );
        assert_eq!(snapshot(&state), before);
    }

    let approval = state.get_interruptions()[0].clone();
    state.approve(&approval, false);
    let resumed = run_to_end(&agent, state, streamed, None).await.unwrap();
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["finish".to_string(), "protected".to_string()]
    );
    assert!(resumed.to_state().pending_input.is_empty());
    if continues {
        assert_eq!(resumed.final_output_text().as_deref(), Some("Continued"));
        let expected_last = if has_handoff { "delegate" } else { "assistant" };
        assert_eq!(resumed.last_agent.name, expected_last);
        let inputs = model.calls();
        assert_eq!(inputs.len(), 2);
        assert_eq!(count_text(&inputs[1].input, "Late input"), 1);
    } else {
        assert_eq!(resumed.final_output_text().as_deref(), Some("Finished"));
        assert_eq!(model.calls().len(), 1);
    }
}

macro_rules! completed_stop_case {
    ($name:ident, $streamed:expr, $serialized:expr, $stop:expr, $handoff:expr) => {
        #[tokio::test]
        async fn $name() {
            completed_stop_sibling($streamed, $serialized, $stop, $handoff).await;
        }
    };
}

completed_stop_case!(
    completed_stop_sibling_finish_blocking_live,
    false,
    false,
    StopName::Finish,
    false
);
completed_stop_case!(
    completed_stop_sibling_finish_blocking_serialized,
    false,
    true,
    StopName::Finish,
    false
);
completed_stop_case!(
    completed_stop_sibling_finish_streamed_live,
    true,
    false,
    StopName::Finish,
    false
);
completed_stop_case!(
    completed_stop_sibling_finish_streamed_serialized,
    true,
    true,
    StopName::Finish,
    false
);
completed_stop_case!(
    completed_stop_sibling_workflow_blocking_live,
    false,
    false,
    StopName::WorkflowFinish,
    false
);
completed_stop_case!(
    completed_stop_sibling_workflow_blocking_serialized,
    false,
    true,
    StopName::WorkflowFinish,
    false
);
completed_stop_case!(
    completed_stop_sibling_workflow_streamed_live,
    true,
    false,
    StopName::WorkflowFinish,
    false
);
completed_stop_case!(
    completed_stop_sibling_workflow_streamed_serialized,
    true,
    true,
    StopName::WorkflowFinish,
    false
);
completed_stop_case!(
    completed_stop_sibling_unrelated_blocking_live,
    false,
    false,
    StopName::Unrelated,
    false
);
completed_stop_case!(
    completed_stop_sibling_unrelated_blocking_serialized,
    false,
    true,
    StopName::Unrelated,
    false
);
completed_stop_case!(
    completed_stop_sibling_unrelated_streamed_live,
    true,
    false,
    StopName::Unrelated,
    false
);
completed_stop_case!(
    completed_stop_sibling_unrelated_streamed_serialized,
    true,
    true,
    StopName::Unrelated,
    false
);
completed_stop_case!(
    completed_stop_sibling_handoff_blocking_live,
    false,
    false,
    StopName::Finish,
    true
);
completed_stop_case!(
    completed_stop_sibling_handoff_blocking_serialized,
    false,
    true,
    StopName::Finish,
    true
);
completed_stop_case!(
    completed_stop_sibling_handoff_streamed_live,
    true,
    false,
    StopName::Finish,
    true
);
completed_stop_case!(
    completed_stop_sibling_handoff_streamed_serialized,
    true,
    true,
    StopName::Finish,
    true
);

#[tokio::test]
async fn failed_model_request_does_not_duplicate_admitted_input_on_resume() {
    // Python `test_failed_model_request_does_not_duplicate_admitted_input_on_resume`.
    let mut at = after_turn_state(None, false).await;
    at.state.add_input("Late input").unwrap();
    at.model.add_error("model failed");
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let mut run_options = options(None, false);
    run_options = run_options.checkpoint(&checkpoint);
    let agent = at.agent.clone();
    let err = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("model failed"), "{err}");

    let failed = checkpoint.take().expect("a failed run leaves a checkpoint");
    assert!(failed.pending_input.is_empty());
    assert_eq!(
        admitted_inputs(&failed),
        vec![Some("Late input".to_string())]
    );
    let state = RunState::from_json(failed.to_json().unwrap()).unwrap();
    at.model.add_turn(vec![assistant_message("Recovered")]);
    let result = Runner::run_with_options(&agent, RunInput::from(state), options(None, false))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Recovered"));
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
}

#[tokio::test]
async fn failed_model_request_with_session_persists_admitted_input_once() {
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("failed-model"));
    let mut at = after_turn_state(Some(session.clone()), false).await;
    at.state.add_input("Late input").unwrap();
    at.model.add_error("model failed");
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let run_options = options(Some(session.clone()), false).checkpoint(&checkpoint);
    let agent = at.agent.clone();
    let err = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("model failed"), "{err}");

    let failed = checkpoint.take().expect("a failed run leaves a checkpoint");
    assert!(failed.pending_input.is_empty());
    let stored = session.get_items(None).await.unwrap();
    assert_eq!(count_text(&stored, "Late input"), 1);

    let state = RunState::from_json(failed.to_json().unwrap()).unwrap();
    at.model.add_turn(vec![assistant_message("Recovered")]);
    let result = Runner::run_with_options(
        &agent,
        RunInput::from(state),
        options(Some(session.clone()), false),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Recovered"));
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
    assert_eq!(
        count_text(&session.get_items(None).await.unwrap(), "Late input"),
        1
    );
}

#[tokio::test]
async fn server_filter_omission_remains_pending_for_later_nonstream_turn() {
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Late input").unwrap();
    at.model
        .add_turn(vec![destination_call("Rome", "call-filtered-destination")]);
    at.model.add_turn(vec![assistant_message("Done")]);
    let filter_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = filter_calls.clone();
    let config = RunConfig::new().with_call_model_input_filter(move |data| {
        let counter = counter.clone();
        async move {
            let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            Ok(ModelInputData {
                input: if n == 1 {
                    Vec::new()
                } else {
                    data.model_data.input.clone()
                },
                instructions: data.model_data.instructions.clone(),
            })
        }
    });
    let mut run_options = options(None, true);
    run_options.run_config = config;
    let agent = at.agent.clone();
    let result = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    assert_eq!(
        *at.calls.lock().unwrap(),
        vec!["Paris".to_string(), "Rome".to_string()]
    );
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
    assert!(result.to_state().pending_input.is_empty());
}

#[tokio::test]
async fn server_filter_in_place_pending_rewrite_preserves_occurrence() {
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Late input").unwrap();
    at.model.add_turn(vec![assistant_message("Done")]);
    let mut run_options = options(None, true);
    run_options.run_config = RunConfig::new().with_call_model_input_filter(|data| async move {
        let mut input = data.model_data.input.clone();
        for item in input.iter_mut() {
            if message_text(item).as_deref() == Some("Late input") {
                item.0["content"] = json!("Filtered late input");
            }
        }
        Ok(ModelInputData {
            input,
            instructions: data.model_data.instructions.clone(),
        })
    });
    let agent = at.agent.clone();
    let result = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Filtered late input"),
        1
    );
    assert!(result.to_state().pending_input.is_empty());
}

#[tokio::test]
async fn terminal_state_rejects_pending_input_without_mutation() {
    let model = fake();
    model.add_turn(vec![assistant_message("Done")]);
    let agent = Arc::new(Agent::<()>::new("assistant").with_model_instance(model));
    let result = Runner::run_with_options(&agent, "Initial request", options(None, false))
        .await
        .unwrap();
    let mut state = result.to_state();
    let before = snapshot(&state);

    let err = state.add_input("Too late").unwrap_err();
    assert!(err.to_string().contains("terminal RunState"), "{err}");
    assert_eq!(snapshot(&state), before);
}

/// Run a streamed resume until its first tool output, then cancel after that turn.
async fn cancel_after_first_tool_output(
    agent: &Arc<Agent<()>>,
    state: RunState,
    auto_previous_response_id: bool,
) -> RunState {
    let mut streamed = Runner::run_streamed_with_options(
        agent,
        RunInput::from(state),
        options(None, auto_previous_response_id),
    );
    let handle = streamed.cancel_handle();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            } = event.unwrap()
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    streamed
        .to_state()
        .expect("a cancelled run leaves its state")
}

#[tokio::test]
async fn server_filter_omission_survives_streamed_cancel_and_round_trip() {
    // Python keeps staged input pending while the filter withholds it from the request.
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Late input").unwrap();
    at.model
        .add_turn(vec![destination_call("Rome", "call-filtered-destination")]);
    let mut run_options = options(None, true);
    run_options.run_config = RunConfig::new().with_call_model_input_filter(|data| async move {
        Ok(ModelInputData {
            input: Vec::new(),
            instructions: data.model_data.instructions.clone(),
        })
    });
    let mut filtered =
        Runner::run_streamed_with_options(&at.agent, RunInput::from(at.state.clone()), run_options);
    let handle = filtered.cancel_handle();
    {
        let mut events = std::pin::pin!(filtered.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            } = event.unwrap()
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    let state = RunState::from_json(filtered.to_state().unwrap().to_json().unwrap()).unwrap();
    assert_eq!(
        texts(&state.pending_input),
        vec![Some("Late input".to_string())]
    );
    assert!(admitted_inputs(&state).is_empty());

    at.model.add_turn(vec![assistant_message("Done")]);
    let agent = at.agent.clone();
    let result = Runner::run_with_options(&agent, RunInput::from(state), options(None, true))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    assert_eq!(
        *at.calls.lock().unwrap(),
        vec!["Paris".to_string(), "Rome".to_string()]
    );
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
}

/// Rust-specific behaviour, documented as a difference: delta selection is by content
/// fingerprint (`ServerConversationTracker::needs_send`), chosen so a retry or resumed state can
/// neither resend nor skip an item. An identical later occurrence of an input already delivered
/// is therefore not sent again. Python sends it, because it selects by object identity.
async fn identical_late_input_is_not_resent_by_fingerprint(streamed_second_resume: bool) {
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Repeat").unwrap();
    at.model
        .add_turn(vec![destination_call("Rome", "call-second-destination")]);
    let parked = cancel_after_first_tool_output(&at.agent, at.state.clone(), true).await;
    let mut state = RunState::from_json(parked.to_json().unwrap()).unwrap();
    state.add_input("Repeat").unwrap();
    at.model.add_turn(vec![assistant_message("Done")]);

    let agent = at.agent.clone();
    let result = run_to_end_server(&agent, state, streamed_second_resume).await;

    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    assert_eq!(
        *at.calls.lock().unwrap(),
        vec!["Paris".to_string(), "Rome".to_string()]
    );
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Repeat"),
        0
    );
    // Both occurrences are still admitted into the run's own history.
    assert_eq!(
        admitted_inputs(&result.to_state())
            .iter()
            .filter(|t| t.as_deref() == Some("Repeat"))
            .count(),
        2
    );
}

/// Rust-specific behaviour, documented as a difference: staged input is admitted into the run
/// history at turn start, before the model call, so a failed request leaves it admitted rather
/// than pending. The retry sends it exactly once. Python keeps it pending until accepted.
#[tokio::test]
async fn server_managed_staged_input_is_admitted_at_turn_start_and_sent_once_on_retry() {
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Late input").unwrap();
    at.model.add_error("model failed");
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let agent = at.agent.clone();
    let run_options = options(None, true).checkpoint(&checkpoint);
    let err = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("model failed"), "{err}");

    let failed = checkpoint.take().expect("a failed run leaves a checkpoint");
    assert!(failed.pending_input.is_empty());
    assert_eq!(
        admitted_inputs(&failed),
        vec![Some("Late input".to_string())]
    );
    let state = RunState::from_json(failed.to_json().unwrap()).unwrap();
    at.model.add_turn(vec![assistant_message("Recovered")]);
    let result = Runner::run_with_options(&agent, RunInput::from(state), options(None, true))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Recovered"));
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
}

/// A server-managed response refused after acceptance (a reused completed call id) is not asked
/// for again: the retry is refused before any model call, so no second response is produced.
async fn accepted_response_refused_on_retry_without_model_call(streamed_failure: bool) {
    let mut at = after_turn_state(None, true).await;
    at.state.add_input("Late input").unwrap();
    at.model
        .add_turn(vec![destination_call("Rome", "call-destination")]);
    let agent = at.agent.clone();
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let run_options = options(None, true).checkpoint(&checkpoint);
    let err = if streamed_failure {
        run_streamed_error(&agent, at.state.clone(), run_options).await
    } else {
        Runner::run_with_options(&agent, at.state.clone(), run_options)
            .await
            .unwrap_err()
    };
    assert!(err.to_string().contains("completed tool call ID"), "{err}");

    let accepted = checkpoint
        .take()
        .expect("the refused response is checkpointed");
    assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);

    let state = RunState::from_json(accepted.to_json().unwrap()).unwrap();
    let queued = at.model.remaining_steps();
    let err = Runner::run_with_options(&agent, RunInput::from(state), options(None, true))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unrecoverable boundary"), "{err}");
    assert_eq!(at.model.remaining_steps(), queued);
    assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);
}

/// Run to the end, streamed or not, and return the result.
async fn run_to_end_server(
    agent: &Arc<Agent<()>>,
    state: RunState,
    streamed: bool,
) -> RunResult<()> {
    if streamed {
        let mut result =
            Runner::run_streamed_with_options(agent, RunInput::from(state), options(None, true));
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        return result
            .into_result()
            .expect("completed stream holds its result");
    }
    Runner::run_with_options(agent, RunInput::from(state), options(None, true))
        .await
        .unwrap()
}

/// The error of a streamed run that is expected to fail.
async fn run_streamed_error(
    agent: &Arc<Agent<()>>,
    state: RunState,
    run_options: RunOptions<()>,
) -> AgentsError {
    let mut failed = Runner::run_streamed_with_options(agent, RunInput::from(state), run_options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(failed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    error.expect("the streamed run must fail")
}

#[tokio::test]
async fn identical_late_input_is_not_resent_by_fingerprint_blocking() {
    identical_late_input_is_not_resent_by_fingerprint(false).await;
}

#[tokio::test]
async fn identical_late_input_is_not_resent_by_fingerprint_streamed() {
    identical_late_input_is_not_resent_by_fingerprint(true).await;
}

#[tokio::test]
async fn accepted_response_refused_on_retry_without_model_call_blocking() {
    accepted_response_refused_on_retry_without_model_call(false).await;
}

#[tokio::test]
async fn accepted_response_refused_on_retry_without_model_call_streamed() {
    accepted_response_refused_on_retry_without_model_call(true).await;
}
