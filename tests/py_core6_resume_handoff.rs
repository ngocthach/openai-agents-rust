//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: a resumed approval that
//! shares its response with a handoff, and a fresh streamed handoff, must survive a failed
//! Session append: the handoff and its guardrails/hooks run once, the checkpoint already names
//! the delegate, and the retry stores one call/output pair per call.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::Tool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, ToolOutputGuardrail,
    ToolOutputGuardrailData,
};
use openai_agents_sdk::{Agent, AgentsError, RunOptions, RunState, Runner};
use py_core6_resume_support::*;
use serde_json::{Value, json};

#[derive(Default)]
struct Counting(Mutex<Vec<String>>);

#[async_trait]
impl RunHooks<()> for Counting {
    async fn on_tool_start(&self, _c: &ToolContext<()>, _a: &Agent<()>, _t: &Tool<()>) {
        self.0.lock().unwrap().push("tool-start".into());
    }
    async fn on_tool_end(
        &self,
        _c: &ToolContext<()>,
        _a: &Agent<()>,
        _t: &Tool<()>,
        _r: &openai_agents_sdk::tool::ToolOutput,
    ) {
        self.0.lock().unwrap().push("tool-end".into());
    }
}

struct Setup {
    agent: Agent<()>,
    model: Arc<openai_agents_sdk::testing::ScriptedModel>,
    session: Arc<FailingSession>,
    state: RunState,
    effects: Arc<Mutex<Vec<i64>>>,
    guardrails: Arc<Mutex<Vec<String>>>,
    hooks: Arc<Counting>,
    handoffs: Arc<Mutex<Vec<String>>>,
}

fn info(state: &RunState) -> (Vec<Value>, Vec<Value>) {
    (
        state
            .tool_input_guardrail_results
            .iter()
            .map(|r| r.output.output_info.clone())
            .collect(),
        state
            .tool_output_guardrail_results
            .iter()
            .map(|r| r.output.output_info.clone())
            .collect(),
    )
}

/// Python's `_approved_handoff_session_state`.
async fn approved_handoff_state(streamed: bool) -> Setup {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let guardrails = Arc::new(Mutex::new(Vec::<String>::new()));
    let hooks = Arc::new(Counting::default());
    let handoffs = Arc::new(Mutex::new(Vec::<String>::new()));
    let (gi, go) = (guardrails.clone(), guardrails.clone());
    let charge = charge_tool(&effects)
        .with_input_guardrail(ToolInputGuardrail::new(
            "record_input",
            move |_d: ToolInputGuardrailData<()>| {
                let g = gi.clone();
                async move {
                    g.lock().unwrap().push("input".into());
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("input-checked")))
                }
            },
        ))
        .with_output_guardrail(ToolOutputGuardrail::new(
            "record_output",
            move |_d: ToolOutputGuardrailData<()>| {
                let g = go.clone();
                async move {
                    g.lock().unwrap().push("output".into());
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("output-checked")))
                }
            },
        ));
    let model = scripted(vec![
        vec![
            call("charge", json!({"amount": 7}), "charge-1"),
            call("transfer_to_delegate", json!({}), "handoff-1"),
        ],
        vec![assistant_message("done")],
        vec![assistant_message("fresh")],
    ]);
    let delegate = Agent::<()>::new("delegate").with_model_instance(model.clone());
    let seen = handoffs.clone();
    let route = handoff(delegate).with_on_handoff(move |_c| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push("handoff".into());
            Ok(())
        }
    });
    let agent = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_tool(charge)
        .with_handoff(route);
    let session = FailingSession::new("resume");
    let h = hooks.clone();
    let paused = drive_with(
        &agent,
        "charge 7 then hand off",
        Some(session.clone()),
        streamed,
        &RunCheckpoint::new(),
        move |o| o.hooks(h),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let first = state.get_interruptions()[0].clone();
    state.approve(&first, false);
    Setup {
        agent,
        model,
        session,
        state,
        effects,
        guardrails,
        hooks,
        handoffs,
    }
}

async fn handoff_recovery_case(
    failing_streamed: bool,
    retry_streamed: bool,
    json_trip: bool,
    failure: &'static str,
) -> Result<(), String> {
    let label = format!("fs={failing_streamed} rs={retry_streamed} json={json_trip} {failure}");
    macro_rules! ck {
        ($c:expr) => {
            if !$c {
                return Err(format!("{label}: {}", stringify!($c)));
            }
        };
    }
    let s = approved_handoff_state(failing_streamed).await;
    s.session.fail_next(failure);
    let slot = RunCheckpoint::new();
    let h = s.hooks.clone();
    let err = drive_with(
        &s.agent,
        s.state.clone(),
        Some(s.session.clone()),
        failing_streamed,
        &slot,
        move |o| o.hooks(h),
    )
    .await
    .err()
    .ok_or_else(|| format!("{label}: the append must fail"))?;
    ck!(err.to_string().contains("session append failed"));
    ck!(*s.effects.lock().unwrap() == vec![7]);
    ck!(*s.guardrails.lock().unwrap() == vec!["input", "output"]);
    ck!(*s.hooks.0.lock().unwrap() == vec!["tool-start", "tool-end"]);
    ck!(*s.handoffs.lock().unwrap() == vec!["handoff"]);
    ck!(s.model.calls().len() == 1);
    let mut state = slot.take().ok_or("no checkpoint")?;
    ck!(info(&state) == (vec![json!("input-checked")], vec![json!("output-checked")]));
    let payload = state.to_json().unwrap();
    let pending: Vec<Value> = payload["pending_session_write"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let pend_items: Vec<_> = pending
        .iter()
        .cloned()
        .map(openai_agents_sdk::InputItem)
        .collect();
    ck!(call_pair(&pend_items, "charge-1") == vec!["function_call_output"]);
    ck!(call_pair(&pend_items, "handoff-1") == vec!["function_call_output"]);
    if json_trip {
        state = RunState::from_json(payload).map_err(|e| e.to_string())?;
        ck!(info(&state) == (vec![json!("input-checked")], vec![json!("output-checked")]));
    }
    ck!(state.current_agent_name == "delegate");
    let h = s.hooks.clone();
    let result = drive_with(
        &s.agent,
        state,
        Some(s.session.clone()),
        retry_streamed,
        &RunCheckpoint::new(),
        move |o| o.hooks(h),
    )
    .await
    .map_err(|e| format!("{label}: retry failed: {e}"))?;
    ck!(result.final_output == Some(json!("done")));
    let final_state = result.state.ok_or("no state")?;
    ck!(final_state.current_agent_name == "delegate");
    ck!(*s.effects.lock().unwrap() == vec![7]);
    ck!(*s.guardrails.lock().unwrap() == vec!["input", "output"]);
    ck!(*s.hooks.0.lock().unwrap() == vec!["tool-start", "tool-end"]);
    ck!(*s.handoffs.lock().unwrap() == vec!["handoff"]);
    ck!(s.model.calls().len() == 2);
    let stored = s.session.items();
    ck!(call_pair(&stored, "charge-1") == pair());
    ck!(call_pair(&stored, "handoff-1") == pair());
    ck!(call_pair(&result.input_list, "charge-1") == pair());
    ck!(call_pair(&result.input_list, "handoff-1") == pair());
    ck!(final_state
        .to_json()
        .unwrap()
        .get("pending_session_write")
        .is_none());
    Ok(())
}

#[tokio::test]
async fn resumed_handoff_session_append_is_recovered_before_next_model() {
    let mut failures = Vec::new();
    for (fs, rs) in [(false, false), (false, true), (true, false), (true, true)] {
        for json_trip in [false, true] {
            for failure in ["before", "after"] {
                if let Err(e) = handoff_recovery_case(fs, rs, json_trip, failure).await {
                    failures.push(e);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A session whose second append fails (Python's `_FailSecondAddItemsSession`), optionally
/// yielding to the scheduler first.
fn handoff_pair_agents() -> (Agent<()>, Arc<openai_agents_sdk::testing::ScriptedModel>) {
    let model = scripted(vec![
        vec![call("transfer_to_delegate", json!({}), "handoff-1")],
        vec![assistant_message("done")],
    ]);
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(handoff(delegate));
    (triage, model)
}

#[tokio::test]
async fn fresh_streamed_handoff_preserves_agent_after_session_append_failure() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let (triage, model) = handoff_pair_agents();
    let session = FailingSession::new("fresh");
    // Rust writes the input together with the handoff batch, so fail the append that carries it.
    session.fail_on_output("handoff-1");
    let mut run = Runner::run_streamed_with_options(
        &triage,
        "hello",
        RunOptions::new(()).session(session.clone()),
    );
    let mut events_seen: Vec<String> = Vec::new();
    let mut error: Option<AgentsError> = None;
    {
        let mut events = std::pin::pin!(run.stream_events());
        while let Some(ev) = events.next().await {
            match ev {
                Ok(StreamEvent::AgentUpdated { new_agent }) => {
                    events_seen.push(new_agent.name.clone())
                }
                Ok(_) => {}
                Err(e) => error = Some(e),
            }
        }
    }
    assert!(
        error
            .expect("the append fails")
            .to_string()
            .contains("session append failed")
    );
    // The completed handoff's agent update is delivered before the error surfaces.
    assert!(
        events_seen.contains(&"delegate".to_string()),
        "{events_seen:?}"
    );
    let state = run.to_state().expect("checkpoint");
    assert_eq!(state.current_agent_name, "delegate");
    let result = drive(
        &triage,
        state,
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(json!("done")));
    assert_eq!(model.calls().len(), 2);
    assert_eq!(call_pair(&session.items(), "handoff-1"), pair());
    assert!(
        result
            .state
            .unwrap()
            .to_json()
            .unwrap()
            .get("pending_session_write")
            .is_none()
    );
}

#[tokio::test]
async fn fresh_streamed_handoff_agent_update_reaches_slow_consumers() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for delay_ms in [0u64, 1] {
        let (triage, _model) = handoff_pair_agents();
        let session = FailingSession::new("fresh");
        session.fail_on_output("handoff-1");
        let mut run = Runner::run_streamed_with_options(
            &triage,
            "hello",
            RunOptions::new(()).session(session),
        );
        let mut updates = Vec::new();
        let mut error = None;
        {
            let mut events = std::pin::pin!(run.stream_events());
            while let Some(ev) = events.next().await {
                if delay_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                match ev {
                    Ok(StreamEvent::AgentUpdated { new_agent }) => {
                        updates.push(new_agent.name.clone())
                    }
                    Ok(_) => {}
                    Err(e) => error = Some(e),
                }
            }
        }
        assert!(
            error
                .expect("the append fails")
                .to_string()
                .contains("session append failed")
        );
        assert!(
            updates.contains(&"delegate".to_string()),
            "delay={delay_ms} {updates:?}"
        );
    }
}

#[tokio::test]
async fn fresh_streamed_handoff_failed_guardrail_state_cannot_resume() {
    use openai_agents_sdk::GuardrailFunctionOutput;
    use openai_agents_sdk::InputGuardrail;
    use tokio::sync::Notify;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for raises_error in [false, true] {
        for retry_streamed in [false, true] {
            let label = format!("error={raises_error} retry_streamed={retry_streamed}");
            let observed = Arc::new(Notify::new());
            let o = observed.clone();
            let pending = InputGuardrail::<()>::new("pending_guardrail", move |_c, _a, _i| {
                let o = o.clone();
                async move {
                    tokio::time::timeout(std::time::Duration::from_secs(5), o.notified())
                        .await
                        .map_err(|_| AgentsError::other("handoff never observed"))?;
                    if raises_error {
                        return Err(AgentsError::other("guardrail backend failed"));
                    }
                    Ok(GuardrailFunctionOutput::new(Value::Null, true))
                }
            });
            let model = scripted(vec![vec![call(
                "transfer_to_delegate",
                json!({}),
                "handoff-1",
            )]]);
            let delegate =
                Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
            let triage = Agent::<()>::new("triage")
                .with_model_instance(model.clone())
                .with_handoff(handoff(delegate))
                .with_input_guardrail(pending);
            let session = FailingSession::new("guarded");
            let mut run = Runner::run_streamed_with_options(
                &triage,
                "hello",
                RunOptions::new(()).session(session.clone()),
            );
            let mut updated_to_delegate = false;
            let mut error = None;
            {
                let mut events = std::pin::pin!(run.stream_events());
                while let Some(ev) = events.next().await {
                    match ev {
                        Ok(StreamEvent::RunItem {
                            name:
                                openai_agents_sdk::stream_events::RunItemStreamEventName::HandoffOccured,
                            ..
                        }) => {
                            observed.notify_one();
                            tokio::task::yield_now().await;
                        }
                        Ok(StreamEvent::AgentUpdated { new_agent })
                            if new_agent.name == "delegate" =>
                        {
                            updated_to_delegate = true;
                        }
                        Ok(_) => {}
                        Err(e) => error = Some(e),
                    }
                }
            }
            let error = error.unwrap_or_else(|| panic!("{label}: the guardrail must fail the run"));
            if raises_error {
                assert!(
                    error.to_string().contains("guardrail backend failed"),
                    "{label}: {error}"
                );
            } else {
                assert!(
                    matches!(error, AgentsError::InputGuardrailTripwireTriggered { .. }),
                    "{label}: {error:?}"
                );
            }
            assert!(!updated_to_delegate, "{label}");
            assert_eq!(run.current_agent().name, "triage", "{label}");
            let state = run.to_state().expect("checkpoint");
            let payload = state.to_json().unwrap();
            assert_eq!(payload["terminal_unrecoverable"], json!(true), "{label}");
            let restored = RunState::from_json(payload).unwrap();
            assert_eq!(restored.current_agent_name, "triage", "{label}");
            for checkpoint in [state, restored] {
                let err = drive(
                    &triage,
                    checkpoint,
                    Some(session.clone()),
                    retry_streamed,
                    &RunCheckpoint::new(),
                )
                .await
                .err()
                .expect("the resume is refused");
                assert!(
                    err.to_string().contains("cannot be resumed"),
                    "{label}: {err}"
                );
            }
            assert_eq!(model.calls().len(), 1, "{label}");
            assert!(
                call_pair(&session.items(), "handoff-1").is_empty(),
                "{label}"
            );
        }
    }
}
