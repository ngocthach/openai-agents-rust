//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: after a post-tool callback
//! failed on a resumed approval, the retried run must keep the accepted output away from its
//! consumer until the output guardrails have judged it, whether the retry is streamed or not and
//! whether the checkpoint is live or round-tripped through JSON.
//!
//! The callback failure is an extractor returning a non-object (Rust callbacks cannot raise); see
//! `py_core6_resume_callback_retry.rs`. Python also inspects the in-flight public result object
//! (`new_items`, `to_input_list`, `tool_output_guardrail_results`); a Rust stream exposes only its
//! events, so those mid-run reads are checked through the events and the final state.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolOutputGuardrail, ToolOutputGuardrailData,
};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use py_core6_resume_support::*;
use serde_json::{Value, json};
use tokio::sync::Notify;

const EXTRACTOR_FAILURE: &str = "custom_data_extractor must return a JSON object";

fn record_output() -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::new(
        "record_output",
        |d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(d.output.to_json_value()))
        },
    )
}

fn metadata(state: &RunState) -> Vec<Value> {
    state
        .tool_output_guardrail_results
        .iter()
        .map(|r| r.output.output_info.clone())
        .collect()
}

fn output_texts(state: &RunState) -> Vec<String> {
    state
        .generated_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => {
                Some(output.as_str().unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect()
}

struct Gate {
    entered: Notify,
    release: Notify,
    seen: Mutex<Vec<String>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Retry {
    StreamedLive,
    StreamedJson,
    RunLive,
    RunJson,
}

struct Case {
    outcome: &'static str,
    prior_turn: bool,
    with_session: bool,
    retry: Retry,
}

/// A streamed run driven on its own task so the test can look at what it published while the
/// output guardrail is still pending.
struct Consumer {
    published: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<(Result<(), AgentsError>, Option<RunState>)>,
}

fn spawn_streamed(
    agent: Agent<()>,
    state: RunState,
    session: Option<Arc<FailingSession>>,
) -> Consumer {
    let published = Arc::new(Mutex::new(Vec::new()));
    let sink = published.clone();
    let task = tokio::spawn(async move {
        let mut options = RunOptions::new(());
        if let Some(session) = session {
            options = options.session(session);
        }
        let mut run = Runner::run_streamed_with_options(&agent, state, options);
        let mut failure = None;
        {
            let mut events = std::pin::pin!(run.stream_events());
            while let Some(event) = events.next().await {
                match event {
                    Ok(StreamEvent::RunItem {
                        name: RunItemStreamEventName::ToolOutput,
                        item,
                    }) => {
                        if let RunItemKind::ToolCallOutput { output, .. } = &item.kind {
                            sink.lock()
                                .unwrap()
                                .push(output.as_str().unwrap_or_default().to_string());
                        }
                    }
                    Ok(_) => {}
                    Err(e) => failure = Some(e),
                }
            }
        }
        let state = run.to_state();
        (failure.map_or(Ok(()), Err), state)
    });
    Consumer { published, task }
}

async fn guardrail_case(case: Case) -> Result<(), String> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let label = format!(
        "{} prior={} session={} retry={}",
        case.outcome,
        case.prior_turn,
        case.with_session,
        match case.retry {
            Retry::StreamedLive => "streamed-live",
            Retry::StreamedJson => "streamed-json",
            Retry::RunLive => "run-live",
            Retry::RunJson => "run-json",
        }
    );
    macro_rules! ck {
        ($cond:expr) => {
            if !$cond {
                return Err(format!("{label}: {}", stringify!($cond)));
            }
        };
    }
    let effects = Arc::new(Mutex::new(Vec::<String>::new()));
    let callbacks = Arc::new(Mutex::new(Vec::<String>::new()));
    let gate = Arc::new(Gate {
        entered: Notify::new(),
        release: Notify::new(),
        seen: Mutex::new(Vec::new()),
    });
    let session = case
        .with_session
        .then(|| FailingSession::new("approval-retry"));

    let prelude =
        FunctionTool::new_no_args("prelude", "", |_c| async move { Ok("earlier-result") })
            .with_output_guardrail(record_output());
    let (e, c) = (effects.clone(), callbacks.clone());
    let charge = FunctionTool::new_no_args("charge", "", move |_c| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push("charged".into());
            Ok("receipt-7")
        }
    })
    .with_needs_approval(true)
    .with_custom_data_extractor(move |_ctx| {
        let c = c.clone();
        async move {
            c.lock().unwrap().push("extractor".into());
            Some(json!("post-tool callback failed"))
        }
    })
    .with_output_guardrail(record_output());
    let record_candidate = OutputGuardrail::<()>::new(
        "record_terminal_candidate",
        |_c, _a, output: Value| async move { Ok(GuardrailFunctionOutput::new(output, false)) },
    );
    let (g, outcome) = (gate.clone(), case.outcome);
    let terminal_policy =
        OutputGuardrail::<()>::new("terminal_policy", move |_c, _a, output: Value| {
            let g = g.clone();
            async move {
                let first = {
                    let mut seen = g.seen.lock().unwrap();
                    seen.push(output.as_str().unwrap_or_default().to_string());
                    seen.len() == 1
                };
                g.entered.notify_one();
                g.release.notified().await;
                if outcome == "error" && first {
                    return Err(AgentsError::other("policy temporarily unavailable"));
                }
                Ok(GuardrailFunctionOutput::new(
                    Value::Null,
                    outcome == "blocked",
                ))
            }
        });
    let mut turns: Vec<Vec<OutputItem>> = Vec::new();
    if case.prior_turn {
        turns.push(vec![call("prelude", json!({}), "prelude-1")]);
    }
    turns.push(vec![call("charge", json!({}), "charge-1")]);
    let model = scripted(turns);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(prelude)
        .with_tool(charge)
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["charge".into()],
        }))
        .with_output_guardrail(record_candidate)
        .with_output_guardrail(terminal_policy);

    let paused = drive(
        &agent,
        "charge once",
        session.clone(),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .map_err(|e| format!("{label}: pause failed: {e}"))?;
    let mut state = round_trip(paused.state.as_ref().ok_or("no state")?);
    let first = state.get_interruptions()[0].clone();
    state.approve(&first, false);
    let prior_metadata: Vec<Value> = if case.prior_turn {
        vec![json!("earlier-result")]
    } else {
        vec![]
    };
    let mut with_receipt = prior_metadata.clone();
    with_receipt.push(json!("receipt-7"));

    // The callback fails before the guardrails ran: nothing was published.
    let failed = spawn_streamed(agent.clone(), state.clone(), session.clone());
    let (failure, failed_state) = failed.task.await.map_err(|e| e.to_string())?;
    let err = failure
        .err()
        .ok_or_else(|| format!("{label}: the callback must fail"))?;
    ck!(err.to_string().contains(EXTRACTOR_FAILURE));
    ck!(gate.seen.lock().unwrap().is_empty());
    ck!(failed.published.lock().unwrap().is_empty());
    let failed_state = failed_state.ok_or("no failure checkpoint")?;
    ck!(metadata(&failed_state) == with_receipt);

    let mut state = failed_state;
    if case.retry != Retry::StreamedLive && case.retry != Retry::RunLive {
        state = round_trip(&state);
    }
    ck!(metadata(&state) == with_receipt);

    // The retry parks on the terminal guardrail with the accepted output still withheld.
    let streamed_retry = matches!(case.retry, Retry::StreamedLive | Retry::StreamedJson);
    let retry_consumer = if streamed_retry {
        Some(spawn_streamed(
            agent.clone(),
            state.clone(),
            session.clone(),
        ))
    } else {
        None
    };
    let run_task = if streamed_retry {
        None
    } else {
        let (agent, state, session) = (agent.clone(), state.clone(), session.clone());
        Some(tokio::spawn(async move {
            drive(&agent, state, session, false, &RunCheckpoint::new()).await
        }))
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), gate.entered.notified())
        .await
        .map_err(|_| format!("{label}: the guardrail never ran"))?;
    for _ in 0..3 {
        tokio::task::yield_now().await;
    }
    if let Some(consumer) = &retry_consumer {
        ck!(consumer.published.lock().unwrap().is_empty());
    }
    if let Some(session) = &session {
        ck!(!call_pair(&session.items(), "charge-1").contains(&"function_call_output".to_string()));
    }
    gate.release.notify_one();

    let mut retry_outputs: Vec<String> = Vec::new();
    let mut final_output: Option<Value> = None;
    let mut retry_state: Option<RunState> = None;
    let mut retry_error: Option<AgentsError> = None;
    if let Some(consumer) = retry_consumer {
        let (result, st) = consumer.task.await.map_err(|e| e.to_string())?;
        retry_outputs = consumer.published.lock().unwrap().clone();
        retry_error = result.err();
        retry_state = st;
        if retry_error.is_none() {
            final_output = Some(json!("receipt-7"));
        }
    } else if let Some(task) = run_task {
        match task.await.map_err(|e| e.to_string())? {
            Ok(outcome) => {
                final_output = outcome.final_output;
                retry_state = outcome.state;
            }
            Err(e) => retry_error = Some(e),
        }
    }
    match case.outcome {
        "blocked" => {
            ck!(matches!(
                retry_error,
                Some(AgentsError::OutputGuardrailTripwireTriggered { .. })
            ));
            ck!(retry_outputs.is_empty());
        }
        "error" => {
            let e =
                retry_error.ok_or_else(|| format!("{label}: the guardrail error must surface"))?;
            ck!(e.to_string().contains("policy temporarily unavailable"));
            ck!(retry_outputs.is_empty());
            ck!(*effects.lock().unwrap() == vec!["charged"]);
            if let Some(session) = &session {
                ck!(!call_pair(&session.items(), "charge-1")
                    .contains(&"function_call_output".to_string()));
            }
            // The guardrail failed after the accepted output; a second retry judges it again.
            let mut again = if matches!(case.retry, Retry::RunLive) {
                state.clone()
            } else {
                round_trip(&state)
            };
            if let Some(st) = &retry_state {
                again = if matches!(case.retry, Retry::RunLive) {
                    st.clone()
                } else {
                    round_trip(st)
                };
            }
            ck!(metadata(&again) == with_receipt);
            let second = spawn_streamed(agent.clone(), again, session.clone());
            tokio::time::timeout(std::time::Duration::from_secs(5), gate.entered.notified())
                .await
                .map_err(|_| format!("{label}: the second guardrail never ran"))?;
            for _ in 0..3 {
                tokio::task::yield_now().await;
            }
            ck!(second.published.lock().unwrap().is_empty());
            gate.release.notify_one();
            let (result, _) = second.task.await.map_err(|e| e.to_string())?;
            ck!(result.is_ok());
            ck!(*second.published.lock().unwrap() == vec!["receipt-7"]);
            ck!(*gate.seen.lock().unwrap() == vec!["receipt-7", "receipt-7"]);
            ck!(model.calls().len() == if case.prior_turn { 2 } else { 1 });
            if let Some(session) = &session {
                ck!(call_pair(&session.items(), "charge-1") == pair());
            }
        }
        _ => {
            ck!(retry_error.is_none());
            ck!(final_output == Some(json!("receipt-7")));
            if streamed_retry {
                ck!(retry_outputs == vec!["receipt-7"]);
            }
            let st = retry_state.ok_or("no retry state")?;
            ck!(metadata(&st) == with_receipt);
            let mut expected = Vec::new();
            if case.prior_turn {
                expected.push("earlier-result".to_string());
            }
            expected.push("receipt-7".to_string());
            ck!(output_texts(&round_trip(&st)) == expected);
        }
    }
    ck!(*effects.lock().unwrap() == vec!["charged"]);
    ck!(*callbacks.lock().unwrap() == vec!["extractor"]);
    ck!(gate.seen.lock().unwrap()[0] == "receipt-7");
    ck!(model.calls().len() == if case.prior_turn { 2 } else { 1 });
    Ok(())
}

#[tokio::test]
async fn failed_approval_callback_defers_stream_output_until_retry_guardrail() {
    let cases = [
        ("allowed", false, false, Retry::StreamedLive),
        ("blocked", false, false, Retry::StreamedLive),
        ("error", false, false, Retry::StreamedJson),
        ("allowed", true, false, Retry::StreamedJson),
        ("blocked", true, false, Retry::StreamedJson),
        ("error", true, false, Retry::StreamedJson),
        ("error", true, true, Retry::StreamedJson),
        ("error", true, true, Retry::RunLive),
        ("error", true, true, Retry::RunJson),
    ];
    let mut failures = Vec::new();
    for (outcome, prior_turn, with_session, retry) in cases {
        if let Err(e) = guardrail_case(Case {
            outcome,
            prior_turn,
            with_session,
            retry,
        })
        .await
        {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
