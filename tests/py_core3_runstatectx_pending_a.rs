//! Port of `tests/test_run_state_pending_input.py`, first half: staging, ordering, the after-turn
//! resume, approval interplay, and the guardrail and session cases of a resumed run.
//!
//! Python's `RunState` is mutated by the run it resumes; a Rust run consumes its state. Where a
//! test reads the state after the run, this file reads `result.to_state()`, and where a failed run
//! is retried it uses a [`RunCheckpoint`], which is how Rust keeps the state of a failed run.
//!
//! Python-only (not ported here): the private `OpenAIServerConversationTracker.prepare_input` call
//! in `test_server_tracker_distinguishes_identical_input_occurrences_after_restore`. The same
//! occurrence rule is exercised through the public runner in `py_core3_runstatectx_pending_b.rs`.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents_sdk::items::{InputItem, RunInput};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunItemStreamEventName, Runner, StreamEvent};
use serde_json::{Value, json};

type Calls = Arc<Mutex<Vec<String>>>;

/// Seen `(source, batch)` pairs of the guardrails in a run.
type GuardrailSeen = Arc<Mutex<Vec<(&'static str, Vec<Option<String>>)>>>;

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
fn item_type(item: &InputItem) -> Option<String> {
    let obj = item.0.as_object()?;
    obj.get("type")
        .or_else(|| obj.get("role"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn texts(items: &[InputItem]) -> Vec<Option<String>> {
    items.iter().map(message_text).collect()
}

fn count_text(items: &[InputItem], text: &str) -> usize {
    items
        .iter()
        .filter(|i| message_text(i).as_deref() == Some(text))
        .count()
}

fn types(items: &[InputItem]) -> Vec<Option<String>> {
    items.iter().map(item_type).collect()
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

#[tokio::test]
async fn pending_input_preserves_order_and_serialization_round_trips() {
    let mut state = RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "assistant".into(),
        original_input: vec![InputItem::user("Initial request")],
        ..Default::default()
    };
    let starting_turn = state.current_turn;

    state.add_input("First late message").unwrap();
    state
        .add_input(vec![InputItem(
            json!({"role": "user", "content": "Second late message"}),
        )])
        .unwrap();
    assert_eq!(state.current_turn, starting_turn);
    assert_eq!(
        texts(&state.pending_input),
        vec![
            Some("First late message".into()),
            Some("Second late message".into())
        ]
    );

    // The pending list handed out is a copy: mutating it does not reach the state.
    let mut detached = state.pending_input.clone();
    detached[0].0["content"] = json!("mutated");
    assert_eq!(
        message_text(&state.pending_input[0]).as_deref(),
        Some("First late message")
    );

    let serialized = snapshot(&state);
    assert_eq!(serialized["$schemaVersion"], CURRENT_SCHEMA_VERSION);
    let restored = RunState::from_json(serialized.clone()).unwrap();
    let restored_from_string = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    for candidate in [&restored, &restored_from_string] {
        assert_eq!(
            texts(&candidate.pending_input),
            vec![
                Some("First late message".into()),
                Some("Second late message".into())
            ]
        );
    }

    // A 1.14 snapshot has no pending_input key and predates the current-step record.
    let mut legacy = serialized;
    legacy["$schemaVersion"] = json!("1.14");
    legacy.as_object_mut().unwrap().remove("pending_input");
    legacy["current_step"] = Value::Null;
    let restored_legacy = RunState::from_json(legacy).unwrap();
    assert!(restored_legacy.pending_input.is_empty());
}

#[tokio::test]
async fn after_turn_resume_admits_input_after_tool_output_exactly_once() {
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("after-turn"));
    let mut at = after_turn_state(Some(session.clone()), false).await;
    at.state
        .add_input("Change the destination to Tokyo")
        .unwrap();
    at.model.add_turn(vec![assistant_message("Updated")]);

    let result = Runner::run_with_options(
        &at.agent,
        RunInput::from(at.state.clone()),
        options(Some(session.clone()), false),
    )
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Updated"));
    assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);
    let model_input = at.model.last_call().unwrap().input;
    assert_eq!(
        types(&model_input),
        vec![
            Some("user".into()),
            Some("function_call".into()),
            Some("function_call_output".into()),
            Some("user".into()),
        ]
    );
    assert_eq!(
        count_text(&model_input, "Change the destination to Tokyo"),
        1
    );
    let end = result.to_state();
    assert!(end.pending_input.is_empty());

    let session_items = session.get_items(None).await.unwrap();
    assert_eq!(
        count_text(&session_items, "Change the destination to Tokyo"),
        1
    );
    assert_eq!(
        count_text(&result.to_input_list(), "Change the destination to Tokyo"),
        1
    );
    let err = end.clone().add_input("Too late").unwrap_err();
    assert!(err.to_string().contains("terminal RunState"), "{err}");
}

#[tokio::test]
async fn streamed_resume_matches_pending_input_ordering() {
    let mut at = after_turn_state(None, false).await;
    at.state
        .add_input("Change the destination to Tokyo")
        .unwrap();
    at.model.add_turn(vec![assistant_message("Updated")]);

    let mut streamed = Runner::run_streamed_with_options(
        &at.agent,
        RunInput::from(at.state.clone()),
        options(None, false),
    );
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    let result = streamed.result().unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Updated"));
    assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);
    let model_input = at.model.last_call().unwrap().input;
    assert_eq!(
        types(&model_input),
        vec![
            Some("user".into()),
            Some("function_call".into()),
            Some("function_call_output".into()),
            Some("user".into()),
        ]
    );
    assert_eq!(
        count_text(&model_input, "Change the destination to Tokyo"),
        1
    );
    assert!(result.to_state().pending_input.is_empty());
    let err = result.to_state().add_input("Too late").unwrap_err();
    assert!(err.to_string().contains("terminal RunState"), "{err}");
}

#[tokio::test]
async fn server_managed_resume_sends_pending_input_as_unsent_delta_once() {
    let mut at = after_turn_state(None, true).await;
    at.state
        .add_input("Change the destination to Tokyo")
        .unwrap();
    at.model.add_turn(vec![assistant_message("Updated")]);

    let result = Runner::run_with_options(&at.agent, at.state.clone(), options(None, true))
        .await
        .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Updated"));
    assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);
    let call = at.model.last_call().unwrap();
    assert_eq!(call.previous_response_id.as_deref(), Some("resp_fake_1"));
    assert_eq!(
        types(&call.input),
        vec![Some("function_call_output".into()), Some("user".into())]
    );
    assert_eq!(
        count_text(&call.input, "Change the destination to Tokyo"),
        1
    );
    assert!(result.to_state().pending_input.is_empty());
}

#[tokio::test]
async fn unresolved_approval_keeps_pending_input_until_tool_finishes() {
    let calls: Calls = Arc::default();
    let log = calls.clone();
    let protected =
        FunctionTool::<()>::new("protected_tool", "Needs approval", move |_c, a: Value| {
            let log = log.clone();
            async move {
                let value = a["value"].as_str().unwrap_or_default().to_string();
                log.lock().unwrap().push(value.clone());
                Ok(format!("approved:{value}"))
            }
        })
        .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![function_call_with_id(
        "protected_tool",
        r#"{"value":"one"}"#,
        "call-protected",
    )]);
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model.clone())
            .with_tools(vec![protected]),
    );
    let interrupted = Runner::run_with_options(&agent, "Initial request", options(None, false))
        .await
        .unwrap();
    let mut state = interrupted.to_state();
    state.add_input("Late input").unwrap();

    let still = Runner::run_with_options(&agent, state.clone(), options(None, false))
        .await
        .unwrap();
    assert!(!still.interruptions.is_empty());
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(
        message_text(&still.to_state().pending_input[0]).as_deref(),
        Some("Late input")
    );

    let mut parked = still.to_state();
    let approval = parked.get_interruptions()[0].clone();
    parked.approve(&approval, false);
    model.add_turn(vec![assistant_message("Done")]);
    let resumed = Runner::run_with_options(&agent, parked, options(None, false))
        .await
        .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("Done"));
    assert_eq!(*calls.lock().unwrap(), vec!["one".to_string()]);
    let model_input = model.last_call().unwrap().input;
    let tail = &model_input[model_input.len() - 2..];
    assert_eq!(
        types(tail),
        vec![Some("function_call_output".into()), Some("user".into())]
    );
    assert_eq!(message_text(&tail[1]).as_deref(), Some("Late input"));
}

#[tokio::test]
async fn streamed_after_turn_cancel_keeps_pending_input_for_next_resume() {
    let calls: Calls = Arc::default();
    let log = calls.clone();
    let protected =
        FunctionTool::<()>::new("protected_tool", "Needs approval", move |_c, a: Value| {
            let log = log.clone();
            async move {
                let value = a["value"].as_str().unwrap_or_default().to_string();
                log.lock().unwrap().push(value.clone());
                Ok(format!("approved:{value}"))
            }
        })
        .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![function_call_with_id(
        "protected_tool",
        r#"{"value":"one"}"#,
        "call-protected",
    )]);
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model.clone())
            .with_tools(vec![protected]),
    );
    let interrupted = Runner::run_with_options(&agent, "Initial request", options(None, false))
        .await
        .unwrap();
    let mut state = interrupted.to_state();
    state.add_input("Late input").unwrap();
    let approval = state.get_interruptions()[0].clone();
    state.approve(&approval, false);

    let mut resumed = Runner::run_streamed_with_options(&agent, state, options(None, false));
    let handle = resumed.cancel_handle();
    {
        let mut events = std::pin::pin!(resumed.stream_events());
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
    assert_eq!(*calls.lock().unwrap(), vec!["one".to_string()]);
    let kept = resumed.to_state().expect("cancelled run leaves its state");
    assert_eq!(
        message_text(&kept.pending_input[0]).as_deref(),
        Some("Late input")
    );

    model.add_turn(vec![assistant_message("Done")]);
    let result = Runner::run_with_options(&agent, kept, options(None, false))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    let model_input = model.last_call().unwrap().input;
    assert_eq!(count_text(&model_input, "Late input"), 1);
}

/// Case parameters of `test_interruption_without_guaranteed_next_model_rejects_input`.
#[derive(Clone, Copy, Debug)]
enum TerminalStop {
    StopOnFirstTool,
    StopAtProtectedTool,
    CustomCallable,
}

async fn interruption_without_next_model(streamed: bool, stop: TerminalStop) {
    use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
    let protected = FunctionTool::<()>::new(
        "protected_tool",
        "Needs approval",
        |_c, a: Value| async move { Ok(a["value"].as_str().unwrap_or_default().to_string()) },
    )
    .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![function_call_with_id(
        "protected_tool",
        r#"{"value":"one"}"#,
        "call-protected-terminal",
    )]);
    let behavior = match stop {
        TerminalStop::StopOnFirstTool => ToolUseBehavior::StopOnFirstTool,
        TerminalStop::StopAtProtectedTool => ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["protected_tool".into()],
        }),
        TerminalStop::CustomCallable => ToolUseBehavior::Custom(Arc::new(|_c, _r| {
            Box::pin(async { Ok(ToolsToFinalOutputResult::not_final()) })
        })),
    };
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model)
            .with_tools(vec![protected])
            .with_tool_use_behavior(behavior),
    );
    let mut state = if streamed {
        let mut result =
            Runner::run_streamed_with_options(&agent, "Initial request", options(None, false));
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        result.to_state().unwrap()
    } else {
        Runner::run_with_options(&agent, "Initial request", options(None, false))
            .await
            .unwrap()
            .to_state()
    };

    let before = snapshot(&state);
    let err = state.add_input("Late input").unwrap_err();
    assert!(
        err.to_string().contains("tool result may end the run"),
        "{err}"
    );
    assert_eq!(snapshot(&state), before);
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_stop_on_first_tool_blocking() {
    interruption_without_next_model(false, TerminalStop::StopOnFirstTool).await;
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_stop_on_first_tool_streamed() {
    interruption_without_next_model(true, TerminalStop::StopOnFirstTool).await;
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_stop_at_tool_blocking() {
    interruption_without_next_model(false, TerminalStop::StopAtProtectedTool).await;
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_stop_at_tool_streamed() {
    interruption_without_next_model(true, TerminalStop::StopAtProtectedTool).await;
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_custom_blocking() {
    interruption_without_next_model(false, TerminalStop::CustomCallable).await;
}

#[tokio::test]
async fn interruption_without_next_model_rejects_input_custom_streamed() {
    interruption_without_next_model(true, TerminalStop::CustomCallable).await;
}

#[tokio::test]
async fn pending_input_guardrail_trip_keeps_input_recoverable() {
    let mut at = after_turn_state(None, false).await;
    let guarded: Arc<Mutex<Vec<Vec<Option<String>>>>> = Arc::default();
    let seen = guarded.clone();
    let guardrail = InputGuardrail::<()>::new("trip_pending_input", move |_c, _a, input| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(texts(&input));
            Ok(GuardrailFunctionOutput::new(Value::Null, true))
        }
    });
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(at.model.clone())
            .with_input_guardrail(guardrail),
    );
    at.state.add_input("Unsafe late input").unwrap();
    at.model.add_turn(vec![assistant_message("Must not run")]);
    let queued = at.model.remaining_steps();

    let err = Runner::run_with_options(&agent, at.state.clone(), options(None, false))
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err}"
    );
    assert_eq!(at.model.remaining_steps(), queued);
    assert_eq!(
        *guarded.lock().unwrap(),
        vec![vec![Some("Unsafe late input".to_string())]]
    );
    // The staged input stays on the state the caller still holds.
    assert_eq!(
        message_text(&at.state.pending_input[0]).as_deref(),
        Some("Unsafe late input")
    );
    at.state.clear_pending_input();
    assert!(at.state.pending_input.is_empty());
}

#[tokio::test]
async fn pending_input_runs_agent_and_run_config_guardrails_on_only_pending() {
    let mut at = after_turn_state(None, false).await;
    let guarded: GuardrailSeen = Arc::default();
    let agent_seen = guarded.clone();
    let agent_guardrail = InputGuardrail::<()>::new("inspect_agent_input", move |_c, _a, input| {
        let agent_seen = agent_seen.clone();
        async move {
            agent_seen.lock().unwrap().push(("agent", texts(&input)));
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }
    });
    let config_seen = guarded.clone();
    let config_guardrail =
        InputGuardrail::<()>::new("inspect_config_input", move |_c, _a, input| {
            let config_seen = config_seen.clone();
            async move {
                config_seen.lock().unwrap().push(("config", texts(&input)));
                Ok(GuardrailFunctionOutput::new(Value::Null, false))
            }
        });
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(at.model.clone())
            .with_input_guardrail(agent_guardrail),
    );
    at.state.add_input("Guard only this").unwrap();
    at.model.add_turn(vec![assistant_message("Done")]);
    let mut run_options = options(None, false);
    run_options.run_config = RunConfig::new().with_input_guardrail(config_guardrail);

    let result = Runner::run_with_options(&agent, at.state.clone(), run_options)
        .await
        .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    let seen = guarded.lock().unwrap().clone();
    let mut sources: Vec<&str> = seen.iter().map(|(source, _)| *source).collect();
    sources.sort();
    assert_eq!(sources, vec!["agent", "config"]);
    for (_, batch) in &seen {
        assert_eq!(batch, &vec![Some("Guard only this".to_string())]);
    }
}

#[tokio::test]
async fn guardrail_retry_persists_successful_turn_with_session_blocking() {
    guardrail_retry_persists_successful_turn_with_session(false).await;
}

#[tokio::test]
async fn guardrail_retry_persists_successful_turn_with_session_streamed() {
    guardrail_retry_persists_successful_turn_with_session(true).await;
}

async fn guardrail_retry_persists_successful_turn_with_session(streamed_retry: bool) {
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("guardrail-retry"));
    let mut at = after_turn_state(Some(session.clone()), false).await;
    let should_trip = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let flag = should_trip.clone();
    let guardrail = InputGuardrail::<()>::new("inspect_pending_input", move |_c, _a, _input| {
        let flag = flag.clone();
        async move {
            Ok(GuardrailFunctionOutput::new(
                Value::Null,
                flag.load(std::sync::atomic::Ordering::SeqCst),
            ))
        }
    });
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(at.model.clone())
            .with_input_guardrail(guardrail),
    );
    at.state.add_input("Late input").unwrap();
    let state = at.state.clone();

    if streamed_retry {
        let mut tripped = Runner::run_streamed_with_options(
            &agent,
            state.clone(),
            options(Some(session.clone()), false),
        );
        let mut failed = None;
        {
            let mut events = std::pin::pin!(tripped.stream_events());
            while let Some(event) = events.next().await {
                if let Err(err) = event {
                    failed = Some(err);
                }
            }
        }
        assert!(matches!(
            failed,
            Some(AgentsError::InputGuardrailTripwireTriggered { .. })
        ));
    } else {
        let err =
            Runner::run_with_options(&agent, state.clone(), options(Some(session.clone()), false))
                .await
                .unwrap_err();
        assert!(matches!(
            err,
            AgentsError::InputGuardrailTripwireTriggered { .. }
        ));
    }

    should_trip.store(false, std::sync::atomic::Ordering::SeqCst);
    at.model.add_turn(vec![assistant_message("Recovered")]);
    let final_output = if streamed_retry {
        let mut retried =
            Runner::run_streamed_with_options(&agent, state, options(Some(session.clone()), false));
        {
            let mut events = std::pin::pin!(retried.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        retried.result().unwrap().final_output_text()
    } else {
        Runner::run_with_options(&agent, state, options(Some(session.clone()), false))
            .await
            .unwrap()
            .final_output_text()
    };

    assert_eq!(final_output.as_deref(), Some("Recovered"));
    let session_items = session.get_items(None).await.unwrap();
    assert_eq!(count_text(&session_items, "Late input"), 1);
    let last = session_items.last().unwrap();
    assert_eq!(item_type(last).as_deref(), Some("message"));
    assert_eq!(last.0["role"], "assistant");
}
