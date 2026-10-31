//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: the terminal marker. An
//! accepted final output whose Session append failed ran its guardrails and end hooks already,
//! so no resume may replay it; a terminal turn that persists cleanly leaves an ordinary state.

mod py_core6_resume_support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use py_core6_resume_support::*;
use serde_json::{Value, json};

/// Python's `_TerminalLifecycleHooks`: counts the agent lifecycle hooks an application can attach
/// its own effects to.
#[derive(Default)]
struct LifecycleHooks {
    starts: AtomicUsize,
    ends: Mutex<Vec<String>>,
}

#[async_trait]
impl RunHooks<()> for LifecycleHooks {
    async fn on_agent_start(&self, _context: &RunContext<()>, _agent: &Agent<()>) {
        self.starts.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_agent_end(&self, _context: &RunContext<()>, _agent: &Agent<()>, output: &Value) {
        let text = output
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| output.to_string());
        self.ends.lock().unwrap().push(text);
    }
}

impl LifecycleHooks {
    fn ends(&self) -> Vec<String> {
        self.ends.lock().unwrap().clone()
    }

    fn starts(&self) -> usize {
        self.starts.load(Ordering::SeqCst)
    }
}

/// Python's `_terminal_output_session_state`: an approval whose tool output becomes the
/// terminal agent output.
async fn terminal_output_session_state(
    streamed: bool,
    hooks: Option<Arc<LifecycleHooks>>,
) -> Paused {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let model = scripted(vec![
        vec![call("charge", json!({"amount": 7}), "charge-1")],
        vec![assistant_message("retry-final")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects))
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    let session = FailingSession::new("resume");
    let paused = drive_with(
        &agent,
        "charge 7",
        Some(session.clone()),
        streamed,
        &RunCheckpoint::new(),
        |o| match &hooks {
            Some(h) => o.hooks(h.clone()),
            None => o,
        },
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);
    Paused {
        agent,
        model,
        session,
        state,
        effects,
    }
}

fn hooks_option(hooks: &Arc<LifecycleHooks>) -> impl FnOnce(RunOptions<()>) -> RunOptions<()> {
    let hooks = hooks.clone();
    move |o| o.hooks(hooks)
}

#[tokio::test]
async fn terminal_session_append_failure_rejects_every_later_resume() {
    for (failing_streamed, retry_streamed) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        for json_trip in [false, true] {
            for failure in ["before", "after"] {
                let label =
                    format!("fs={failing_streamed} rs={retry_streamed} json={json_trip} {failure}");
                let hooks = Arc::new(LifecycleHooks::default());
                let p = terminal_output_session_state(failing_streamed, Some(hooks.clone())).await;
                p.session.fail_next(failure);
                let slot = RunCheckpoint::new();
                let err = drive_with(
                    &p.agent,
                    p.state.clone(),
                    Some(p.session.clone()),
                    failing_streamed,
                    &slot,
                    hooks_option(&hooks),
                )
                .await
                .err()
                .unwrap_or_else(|| panic!("{label}: the append must fail"));
                assert!(
                    err.to_string().contains("session append failed"),
                    "{label}: {err}"
                );

                // The output, its guardrails, and its terminal hooks all completed exactly once.
                assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
                assert_eq!(p.model.calls().len(), 1, "{label}");
                assert_eq!(hooks.ends(), vec!["receipt-7"], "{label}");
                let starts_after_failure = hooks.starts();
                let mut state = slot.take().unwrap();
                assert_eq!(
                    state.to_json().unwrap()["terminal_unrecoverable"],
                    json!(true),
                    "{label}"
                );
                if json_trip {
                    state = round_trip(&state);
                }
                // Every later resume fails closed, including a second one.
                for _ in 0..2 {
                    let err = drive_with(
                        &p.agent,
                        state.clone(),
                        Some(p.session.clone()),
                        retry_streamed,
                        &RunCheckpoint::new(),
                        hooks_option(&hooks),
                    )
                    .await
                    .err()
                    .unwrap_or_else(|| panic!("{label}: the resume must be refused"));
                    assert_user_error(err, "cannot be resumed");
                    assert_eq!(p.model.calls().len(), 1, "{label}");
                    assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
                    assert_eq!(hooks.starts(), starts_after_failure, "{label}");
                    assert_eq!(hooks.ends(), vec!["receipt-7"], "{label}");
                }
            }
        }
    }
}

#[tokio::test]
async fn unrecoverable_terminal_state_rejects_before_any_resumed_work() {
    for streamed in [false, true] {
        let p = terminal_output_session_state(streamed, None).await;
        let state = failed_resume(&p, p.state.clone(), streamed, "before").await;
        // The rejection precedes Session reconciliation.
        p.session.forbid_reads();
        let restored = round_trip(&state);
        let err = drive(
            &p.agent,
            restored,
            Some(p.session.clone()),
            !streamed,
            &RunCheckpoint::new(),
        )
        .await
        .err()
        .expect("the resume must be refused");
        assert_user_error(err, "cannot be resumed");
        assert_eq!(p.model.calls().len(), 1);
        assert_eq!(*p.effects.lock().unwrap(), vec![7]);
    }
}

#[tokio::test]
async fn terminal_marker_is_cleared_once_the_turn_is_persisted() {
    for streamed in [false, true] {
        let p = terminal_output_session_state(streamed, None).await;
        let result = drive(
            &p.agent,
            p.state.clone(),
            Some(p.session.clone()),
            streamed,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap();
        assert_eq!(result.final_output, Some(json!("receipt-7")));
        assert_eq!(*p.effects.lock().unwrap(), vec![7]);
        assert!(
            result
                .state
                .unwrap()
                .to_json()
                .unwrap()
                .get("terminal_unrecoverable")
                .is_none()
        );
        assert_eq!(
            charge_pair(&p.session.items()),
            vec!["function_call", "function_call_output"]
        );
    }
}

#[tokio::test]
async fn terminal_marker_rejects_an_older_schema_label() {
    let p = terminal_output_session_state(false, None).await;
    let state = failed_resume(&p, p.state.clone(), false, "before").await;
    let mut payload = state.to_json().unwrap();
    assert_eq!(payload["terminal_unrecoverable"], json!(true));
    if let Some(entries) = payload["context"]
        .as_object_mut()
        .and_then(|c| c.remove("function_tool_approvals"))
        .and_then(|e| e.as_array().cloned())
    {
        for entry in entries {
            payload["context"]["approvals"][entry["tool_key"].as_str().unwrap()] =
                entry["decision"].clone();
        }
    }
    payload["$schemaVersion"] = json!("1.16");
    let err = RunState::from_json(payload).expect_err("the marker is refused");
    assert!(
        err.to_string().contains("terminal marker is invalid"),
        "{err}"
    );
}

#[tokio::test]
async fn failed_stream_result_checkpoint_keeps_the_terminal_marker() {
    let p = terminal_output_session_state(true, None).await;
    p.session.fail_next("before");
    let mut streamed = Runner::run_streamed_with_options(
        &p.agent,
        p.state.clone(),
        RunOptions::new(()).session(p.session.clone()),
    );
    {
        use futures::StreamExt;
        let mut events = std::pin::pin!(streamed.stream_events());
        let mut failed = false;
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                assert!(e.to_string().contains("session append failed"), "{e}");
                failed = true;
            }
        }
        assert!(failed);
    }
    let checkpoint = streamed
        .to_state()
        .expect("a failed streamed run keeps its state");
    assert_eq!(
        checkpoint.to_json().unwrap()["terminal_unrecoverable"],
        json!(true)
    );
    let restored = round_trip(&checkpoint);
    for candidate in [checkpoint, restored] {
        let err = drive(
            &p.agent,
            candidate,
            Some(p.session.clone()),
            false,
            &RunCheckpoint::new(),
        )
        .await
        .err()
        .expect("the resume must be refused");
        assert_user_error(err, "cannot be resumed");
    }
    assert_eq!(p.model.calls().len(), 1);
    assert_eq!(*p.effects.lock().unwrap(), vec![7]);
}

fn max_turns_handler(handler_calls: &Arc<Mutex<Vec<String>>>) -> RunErrorHandlers<()> {
    let handler_calls = handler_calls.clone();
    RunErrorHandlers::new().on_max_turns(move |_input| {
        let handler_calls = handler_calls.clone();
        async move {
            handler_calls.lock().unwrap().push("handled".into());
            Ok(Some(RunErrorHandlerResult::new("max-turns-output")))
        }
    })
}

#[tokio::test]
async fn max_turns_handler_output_is_marked_before_it_is_persisted() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let model = scripted(vec![
        vec![call("charge", json!({"amount": 7}), "charge-1")],
        vec![assistant_message("unused")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects));
    let session = FailingSession::new("resume");
    let hooks = Arc::new(LifecycleHooks::default());
    let paused = drive_with(
        &agent,
        "charge 7",
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
        |o| o.hooks(hooks.clone()).max_turns(Some(1)),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);

    let handler_calls = Arc::new(Mutex::new(Vec::new()));
    session.fail_on_output("max-turns-output");
    let slot = RunCheckpoint::new();
    let err = drive_with(&agent, state, Some(session.clone()), false, &slot, |o| {
        o.hooks(hooks.clone())
            .max_turns(Some(1))
            .error_handlers(max_turns_handler(&handler_calls))
    })
    .await
    .err()
    .expect("the final append must fail");
    assert!(err.to_string().contains("session append failed"), "{err}");

    // The handler and its end hook each ran exactly once before the append failed.
    assert_eq!(*handler_calls.lock().unwrap(), vec!["handled"]);
    assert_eq!(hooks.ends(), vec!["max-turns-output"]);
    let state = slot.take().unwrap();
    assert_eq!(
        state.to_json().unwrap()["terminal_unrecoverable"],
        json!(true)
    );
    let err = drive_with(
        &agent,
        state,
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
        |o| {
            o.hooks(hooks.clone())
                .max_turns(Some(1))
                .error_handlers(max_turns_handler(&handler_calls))
        },
    )
    .await
    .err()
    .expect("the resume must be refused");
    assert_user_error(err, "cannot be resumed");
    assert_eq!(*handler_calls.lock().unwrap(), vec!["handled"]);
    assert_eq!(hooks.ends(), vec!["max-turns-output"]);
}

#[tokio::test]
async fn max_turns_guardrail_failure_leaves_the_state_retryable() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let guardrail_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = guardrail_calls.clone();
    let exploding = OutputGuardrail::<()>::new("exploding", move |_c, _a, output: Value| {
        let seen = seen.clone();
        async move {
            seen.lock()
                .unwrap()
                .push(output.as_str().unwrap_or_default().to_string());
            Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail exploded"))
        }
    });
    let effects = Arc::new(Mutex::new(Vec::new()));
    let model = scripted(vec![vec![call("charge", json!({"amount": 7}), "charge-1")]]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects))
        .with_output_guardrail(exploding);
    let session = FailingSession::new("resume");
    let paused = drive_with(
        &agent,
        "charge 7",
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
        |o| o.max_turns(Some(1)),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);

    let handler_calls = Arc::new(Mutex::new(Vec::new()));
    session.fail_on_output("max-turns-output");
    let slot = RunCheckpoint::new();
    let err = drive_with(&agent, state, Some(session.clone()), false, &slot, |o| {
        o.max_turns(Some(1))
            .error_handlers(max_turns_handler(&handler_calls))
    })
    .await
    .err()
    .expect("the final append must fail");
    assert!(err.to_string().contains("session append failed"), "{err}");
    assert_eq!(*guardrail_calls.lock().unwrap(), vec!["max-turns-output"]);
    let state = slot.take().unwrap();
    assert!(
        state
            .to_json()
            .unwrap()
            .get("terminal_unrecoverable")
            .is_none()
    );

    // The retry reports the real guardrail failure rather than a fail-closed rejection.
    let err = drive_with(
        &agent,
        state,
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
        |o| {
            o.max_turns(Some(1))
                .error_handlers(max_turns_handler(&handler_calls))
        },
    )
    .await
    .err()
    .expect("the guardrail still fails");
    assert!(err.to_string().contains("guardrail exploded"), "{err}");
}
