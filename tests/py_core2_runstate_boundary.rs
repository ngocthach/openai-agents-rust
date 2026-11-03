//! Resuming a parked approval turn under output guardrails.
//!
//! Ports `test_ambiguous_serialized_approval_state_fails_before_tool_execution` and
//! `test_serialized_mixed_approval_guardrail_preserves_only_accepted_outputs` (from
//! `test_agent_runner_streamed.py`), which exercise the serialized ownership of the current
//! response: a snapshot whose boundary cannot be proven must be refused before any tool runs,
//! and a proven one must keep only the outputs the guardrail accepted. Also checks that the
//! nested-handoff ownership list survives a snapshot, so a resumed run does not repeat items a
//! handoff already carried into its input.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::run_config::OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, Runner,
};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Calls {
    normal: Arc<AtomicUsize>,
    approval: Arc<AtomicUsize>,
}

impl Calls {
    fn counts(&self) -> (usize, usize) {
        (
            self.normal.load(Ordering::SeqCst),
            self.approval.load(Ordering::SeqCst),
        )
    }
}

fn guardrail(tripwire: bool) -> OutputGuardrail<()> {
    OutputGuardrail::new("check", move |_c, _a, _o| async move {
        Ok(GuardrailFunctionOutput {
            output_info: Value::Null,
            tripwire_triggered: tripwire,
        })
    })
}

/// Two turns: `normal_tool` alone, then `normal_tool` again beside the approval-gated
/// `approval_tool`. `accepted_first` makes the first normal result recognisable.
fn build_agent(calls: &Calls, tripwire: bool, stop_at_approval: bool) -> Agent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![function_call_with_id("normal_tool", "{}", "call-normal")],
        vec![
            function_call_with_id("normal_tool", "{}", "call-extra"),
            function_call_with_id("approval_tool", "{}", "call-approved"),
        ],
        vec![assistant_message("unused")],
    ]);
    let normal = {
        let calls = calls.clone();
        FunctionTool::<()>::new("normal_tool", "Normal", move |_c, _a: Value| {
            let calls = calls.clone();
            async move {
                let n = calls.normal.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(if n == 1 {
                    "accepted-result"
                } else {
                    "sibling-secret"
                }
                .to_string())
            }
        })
    };
    let approval = {
        let calls = calls.clone();
        FunctionTool::<()>::new("approval_tool", "Approval", move |_c, _a: Value| {
            let calls = calls.clone();
            async move {
                calls.approval.fetch_add(1, Ordering::SeqCst);
                Ok("approved-secret".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let mut agent = Agent::<()>::new("test")
        .with_tool(normal)
        .with_tool(approval)
        .with_output_guardrail(guardrail(tripwire))
        .with_model_instance(model);
    if stop_at_approval {
        agent = agent.with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["approval_tool".to_string()],
        }));
    }
    agent
}

async fn run_mode(
    agent: &Agent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: Option<Arc<dyn Session>>,
    streamed: bool,
) -> openai_agents_sdk::Result<Vec<openai_agents_sdk::items::ToolApprovalItem>> {
    run_mode_with(agent, input, session, streamed, None).await
}

async fn run_mode_with(
    agent: &Agent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: Option<Arc<dyn Session>>,
    streamed: bool,
    checkpoint: Option<&openai_agents_sdk::run::RunCheckpoint>,
) -> openai_agents_sdk::Result<Vec<openai_agents_sdk::items::ToolApprovalItem>> {
    let mut options = RunOptions::new(());
    if let Some(checkpoint) = checkpoint {
        options = options.checkpoint(checkpoint);
    }
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut stream = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(stream.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(stream.interruptions())
    } else {
        Ok(Runner::run_with_options(agent, input, options)
            .await?
            .interruptions)
    }
}

async fn parked_state_json(agent: &Agent<()>) -> Value {
    let first = Runner::run(agent, "Use normal_tool, then approval_tool")
        .await
        .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    first.to_state().to_json().unwrap()
}

fn user_error(result: openai_agents_sdk::Result<impl Sized>) -> String {
    match result {
        Err(AgentsError::UserError { message, .. }) => message,
        Err(other) => panic!("expected a UserError, got {other:?}"),
        Ok(_) => panic!("expected the resume to be refused"),
    }
}

#[tokio::test]
async fn ambiguous_serialized_approval_state_fails_before_tool_execution() {
    for streamed in [false, true] {
        for corruption in [
            "missing-prefix",
            "nonterminal-indexes",
            "invalid-interruption-index",
            "processed-mismatch",
            "interruption-mismatch",
        ] {
            let calls = Calls::default();
            let agent = build_agent(&calls, false, false);
            let mut serialized = parked_state_json(&agent).await;
            assert_eq!(calls.counts(), (2, 0));
            let ownership = serialized["current_response_generated_item_ownership"].clone();
            let start = ownership["start"].as_u64().unwrap() as usize;
            let interruption = ownership["interruptions"][0].as_u64().unwrap() as usize;
            match corruption {
                "missing-prefix" => {
                    let items = serialized["generated_items"].as_array().unwrap().clone();
                    serialized["generated_items"] = json!(items[2..].to_vec());
                }
                "nonterminal-indexes" => {
                    let mut trailing = serialized["generated_items"][0].clone();
                    trailing["raw_item"]["call_id"] = json!("call-trailing");
                    serialized["generated_items"]
                        .as_array_mut()
                        .unwrap()
                        .push(trailing);
                }
                "invalid-interruption-index" => {
                    serialized["current_response_generated_item_ownership"]["interruptions"] =
                        json!([3]);
                }
                "processed-mismatch" => {
                    serialized["generated_items"][start]["raw_item"]["call_id"] =
                        json!("call-mismatch");
                }
                _ => {
                    serialized["generated_items"][interruption]["raw_item"]["call_id"] =
                        json!("call-mismatch");
                }
            }
            let mut restored = RunState::from_json(serialized).unwrap();
            assert!(restored.current_response_boundary_unproven, "{corruption}");
            let item = restored.get_interruptions()[0].clone();
            restored.approve(&item, false);
            let message = user_error(run_mode(&agent, restored, None, streamed).await);
            assert!(
                message.contains("current response boundary cannot be proven"),
                "{corruption} streamed={streamed}: {message}"
            );
            assert_eq!(calls.counts(), (2, 0), "{corruption} streamed={streamed}");
        }
    }
}

#[tokio::test]
async fn a_proven_boundary_is_not_refused() {
    let calls = Calls::default();
    let agent = build_agent(&calls, false, false);
    let serialized = parked_state_json(&agent).await;
    assert_eq!(
        serialized["current_response_generated_item_ownership"],
        json!({"start": 2, "end": 6, "interruptions": [5]})
    );
    let restored = RunState::from_json(serialized).unwrap();
    assert!(!restored.current_response_boundary_unproven);
}

#[tokio::test]
async fn a_legacy_1_16_snapshot_without_ownership_cannot_be_proven() {
    let calls = Calls::default();
    let agent = build_agent(&calls, false, false);
    let mut serialized = parked_state_json(&agent).await;
    serialized["$schemaVersion"] = json!("1.16");
    serialized["context"]["function_tool_approvals"] = json!([]);
    serialized
        .as_object_mut()
        .unwrap()
        .remove("current_response_generated_item_ownership");
    let mut restored = RunState::from_json(serialized).unwrap();
    assert!(restored.current_response_boundary_unproven);
    let item = restored.get_interruptions()[0].clone();
    restored.approve(&item, false);
    let message = user_error(run_mode(&agent, restored.clone(), None, false).await);
    assert!(message.contains("cannot be proven"), "{message}");
    // Without output guardrails there is nothing to withhold, so the same state resumes.
    let calls = Calls::default();
    let mut plain = build_agent(&calls, false, false);
    plain.output_guardrails.clear();
    assert!(run_mode(&plain, restored, None, false).await.is_ok());
}

#[tokio::test]
async fn serialized_mixed_approval_guardrail_preserves_only_accepted_outputs() {
    for streamed in [false, true] {
        for ownership in [
            "valid",
            "missing-prefix",
            "missing",
            "invalid",
            "missing-current",
            "missing-approval",
            "prefix-anchor",
            "nonterminal-session",
        ] {
            let calls = Calls::default();
            let agent = build_agent(&calls, true, true);
            let session = Arc::new(InMemorySession::new("mixed"));
            let first = {
                let mut stream_or_run = None;
                let result = Runner::run_with_options(
                    &agent,
                    "Use the tools",
                    RunOptions::new(()).session(session.clone()),
                )
                .await
                .unwrap();
                stream_or_run.replace(result);
                stream_or_run.unwrap()
            };
            assert_eq!(first.interruptions.len(), 1);
            let mut payload = first.to_state().to_json().unwrap();
            let current_start = payload["current_response_generated_item_ownership"]["start"]
                .as_u64()
                .unwrap() as usize;
            let approval_index =
                payload["current_response_generated_item_ownership"]["interruptions"][0]
                    .as_u64()
                    .unwrap() as usize;
            match ownership {
                "missing" => {
                    payload
                        .as_object_mut()
                        .unwrap()
                        .remove("generated_session_item_indexes");
                }
                "missing-prefix" => payload["generated_session_item_indexes"][0] = Value::Null,
                "invalid" => payload["generated_session_item_indexes"][0] = json!(true),
                "missing-current" => {
                    payload["generated_session_item_indexes"][current_start] = Value::Null;
                }
                "missing-approval" => {
                    payload["generated_session_item_indexes"][approval_index] = Value::Null;
                }
                "prefix-anchor" => {
                    let copies: Vec<Value> = payload["last_processed_response"]["new_items"]
                        .as_array()
                        .unwrap()
                        .clone();
                    let shift = copies.len();
                    let mut session_items = copies.clone();
                    session_items.extend(payload["session_items"].as_array().unwrap().clone());
                    payload["session_items"] = json!(session_items);
                    let shifted: Vec<Value> = payload["generated_session_item_indexes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .map_or(Value::Null, |i| json!(i as usize + shift))
                        })
                        .collect();
                    payload["generated_session_item_indexes"] = json!(shifted);
                    for offset in 0..shift {
                        payload["generated_session_item_indexes"][current_start + offset] =
                            json!(offset);
                    }
                }
                "nonterminal-session" => {
                    let first_item = payload["session_items"][0].clone();
                    payload["session_items"]
                        .as_array_mut()
                        .unwrap()
                        .push(first_item);
                }
                _ => {}
            }
            let restored = RunState::from_json_str(&payload.to_string()).unwrap();
            let mut restored =
                RunState::from_json_str(&restored.to_json_string().unwrap()).unwrap();
            let item = restored.get_interruptions()[0].clone();
            restored.approve(&item, false);
            let label = format!("{ownership} streamed={streamed}");
            if !matches!(ownership, "valid" | "missing-prefix") {
                let before = session.get_items(None).await.unwrap();
                let state_session_before = restored.to_json().unwrap()["session_items"].clone();
                let outcome =
                    run_mode(&agent, restored.clone(), Some(session.clone()), streamed).await;
                let message = match outcome {
                    Err(AgentsError::UserError { message, .. }) => message,
                    other => panic!("{label}: expected the boundary refusal, got {other:?}"),
                };
                assert!(
                    message.contains("current response boundary cannot be proven"),
                    "{label}: {message}"
                );
                assert_eq!(calls.counts(), (2, 0), "{label}");
                assert_eq!(session.get_items(None).await.unwrap(), before, "{label}");
                assert_eq!(
                    restored.to_json().unwrap()["session_items"],
                    state_session_before
                );
                assert!(state_session_before.to_string().contains("accepted-result"));
                continue;
            }
            let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
            let outcome = run_mode_with(
                &agent,
                restored.clone(),
                Some(session.clone()),
                streamed,
                Some(&checkpoint),
            )
            .await;
            assert!(
                matches!(
                    outcome,
                    Err(AgentsError::OutputGuardrailTripwireTriggered { .. })
                ),
                "{label}: {outcome:?}"
            );
            assert_eq!(calls.counts(), (2, 1), "{label}");
            let outputs: Vec<(String, String)> = session
                .get_items(None)
                .await
                .unwrap()
                .into_iter()
                .filter(|i| i.item_type() == Some("function_call_output"))
                .map(|i| {
                    (
                        i.0["call_id"].as_str().unwrap().to_string(),
                        i.0["output"].as_str().unwrap().to_string(),
                    )
                })
                .collect();
            assert_eq!(
                outputs,
                vec![
                    ("call-normal".to_string(), "accepted-result".to_string()),
                    (
                        "call-extra".to_string(),
                        OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT.to_string()
                    ),
                    (
                        "call-approved".to_string(),
                        OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT.to_string()
                    ),
                ],
                "{label}"
            );
            // Python mutates the state it resumed; Rust consumes it, so the failed run's
            // checkpoint is where the withheld outputs must be gone.
            let failed = checkpoint
                .take()
                .expect("the failed run leaves a checkpoint");
            let serialized = failed.to_json_string().unwrap();
            assert!(serialized.contains("accepted-result"), "{label}");
            assert!(!serialized.contains("sibling-secret"), "{label}");
            assert!(!serialized.contains("approved-secret"), "{label}");
        }
    }
}

// ---------------------------------------------------------------------------
// Nested handoff ownership survives a snapshot
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nested_handoff_ownership_survives_a_snapshot_without_repeating_items() {
    use openai_agents_sdk::testing::function_call_with_id as call;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let tool_calls = Arc::new(Mutex::new(0usize));
    let seen = tool_calls.clone();
    let gated = FunctionTool::<()>::new("approval_tool", "Gated", move |_c, _a: Value| {
        let seen = seen.clone();
        async move {
            *seen.lock().unwrap() += 1;
            Ok("done".to_string())
        }
    })
    .with_needs_approval(true);
    let target_model = Arc::new(FakeModel::new());
    target_model.add_turns(vec![
        vec![call("approval_tool", "{}", "call-gated")],
        vec![assistant_message("finished")],
    ]);
    let target = Agent::<()>::new("target")
        .with_tool(gated)
        .with_model_instance(target_model);
    let source_model = Arc::new(FakeModel::new());
    source_model.add_turn(vec![
        assistant_message("Handing off."),
        call("transfer_to_target", "{}", "call-handoff"),
    ]);
    let source = Agent::<()>::new("source")
        .with_handoff(target)
        .with_model_instance(source_model);

    let first = Runner::run(&source, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let before = first.to_input_list();
    let state = first.to_state();
    let wire = state.to_json().unwrap();
    // Whatever the handoff carried into the input is recorded with its input position.
    let refs = wire["nested_history_owned_session_item_refs"]
        .as_array()
        .unwrap();
    assert_eq!(refs.len(), state.nested_history_owned_items.len());

    let mut restored = RunState::from_json(wire).unwrap();
    assert_eq!(
        restored.nested_history_owned_items,
        state.nested_history_owned_items
    );
    let item = restored.get_interruptions()[0].clone();
    restored.approve(&item, false);
    let resumed = Runner::run(&source, restored).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("finished"));
    // The replayed history holds each carried item once, the same number as before the pause.
    let after = resumed.to_input_list();
    let count_of = |items: &[openai_agents_sdk::items::InputItem], text: &str| {
        items
            .iter()
            .filter(|i| {
                i.text_content()
                    .as_deref()
                    .is_some_and(|t| t.contains(text))
            })
            .count()
    };
    assert_eq!(
        count_of(&after, "Handing off."),
        count_of(&before, "Handing off.")
    );
    assert_eq!(*tool_calls.lock().unwrap(), 1);
}
