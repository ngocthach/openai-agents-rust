//! Remaining end-to-end ports from `tests/test_agent_runner.py`: tool-not-found recovery,
//! retry-policy replay approval, per-turn session writes, and output-guardrail trips on a
//! resumed run (Python lines ~687-880, 6085-6130, 6750-6970).

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::model_settings::{ModelRetrySettings, ModelSettings};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::retry::{ModelRetryAdvice, ReplaySafety, RetryDecision, RetryPolicy};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::{
    OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT, RunConfig, ToolErrorKind, ToolNotFoundBehavior,
};
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, RunState, Runner,
    handoff,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    text: String,
}

type Log = Arc<Mutex<Vec<String>>>;

fn tool(name: &str, output: &str, log: &Log) -> FunctionTool<()> {
    let (output, log, n) = (output.to_string(), log.clone(), name.to_string());
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let (output, log, n) = (output.clone(), log.clone(), n.clone());
        async move {
            log.lock().unwrap().push(n);
            Ok(output)
        }
    })
}

fn outputs_sent(model: &Arc<FakeModel>) -> Vec<(String, Value)> {
    model
        .calls()
        .last()
        .unwrap()
        .input
        .iter()
        .filter(|i| i.item_type() == Some("function_call_output"))
        .map(|i| {
            (
                i.0["call_id"].as_str().unwrap().to_string(),
                i.0["output"].clone(),
            )
        })
        .collect()
}

fn not_found_config() -> RunConfig<()> {
    RunConfig {
        tool_not_found_behavior: ToolNotFoundBehavior::ReturnErrorToModel,
        ..Default::default()
    }
}

/// Python: `test_tool_not_found_behavior_uses_tool_error_formatter`.
#[tokio::test]
async fn a_missing_tool_message_goes_through_the_error_formatter() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "missing_tool",
        "{}",
        "call_missing",
    )]);
    model.add_turn(vec![assistant_message("recovered")]);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let k = kinds.clone();
    let mut config = not_found_config();
    config.tool_error_formatter = Some(openai_agents_sdk::tool_error_formatter_fn(move |args| {
        k.lock().unwrap().push(args.kind);
        (args.kind == ToolErrorKind::ToolNotFound)
            .then(|| format!("{} unavailable for {}", args.tool_name, args.call_id))
    }));
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let r = Runner::run_with_options(&agent, "start", RunOptions::new(()).run_config(config))
        .await
        .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("recovered"));
    assert_eq!(*kinds.lock().unwrap(), vec![ToolErrorKind::ToolNotFound]);
    assert_eq!(
        outputs_sent(&model),
        vec![(
            "call_missing".to_string(),
            json!("missing_tool unavailable for call_missing")
        )]
    );
}

/// Python: `test_tool_not_found_behavior_handles_mixed_function_tool_calls`.
#[tokio::test]
async fn a_missing_tool_does_not_stop_its_known_sibling() {
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("missing_tool", "{}", "call_missing"),
        function_call_with_id("known_tool", "{}", "call_known"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool("known_tool", "known result", &log));
    let r = Runner::run_with_options(
        &agent,
        "start",
        RunOptions::new(()).run_config(not_found_config()),
    )
    .await
    .unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["known_tool"]);
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
    let mut sent = outputs_sent(&model);
    sent.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        sent,
        vec![
            ("call_known".to_string(), json!("known result")),
            (
                "call_missing".to_string(),
                json!("Tool 'missing_tool' not found.")
            ),
        ]
    );
}

/// Python: `test_auto_previous_response_id_retries_when_policy_approves_unsafe_replay`.
#[tokio::test]
async fn a_policy_can_approve_an_unsafe_stateful_replay() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    let policy = RetryPolicy::new(move |ctx| {
        s.lock().unwrap().push((
            ctx.previous_response_id.map(str::to_string),
            ctx.conversation_id.map(str::to_string),
            ctx.stateful_request(),
            ctx.response_started(),
            ctx.replay_safety(),
        ));
        RetryDecision::retry().with_unsafe_replay_approved()
    });
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call_with_id(
                "test_func",
                r#"{"text": "foo"}"#,
                "c1",
            )]),
            ScriptedStep::raise_error(AgentsError::other(
                "connection closed after response processing started",
            ))
            .with_retry_advice(ModelRetryAdvice {
                suggested: Some(false),
                replay_safety: Some(ReplaySafety::Unsafe),
                response_started: true,
                ..Default::default()
            }),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(tool("test_func", "tool_result", &log))
        .with_model_settings(ModelSettings {
            retry: Some(
                ModelRetrySettings {
                    max_retries: Some(1),
                    ..Default::default()
                }
                .with_policy(policy),
            ),
            ..Default::default()
        });
    let r = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).auto_previous_response_id(true),
    )
    .await
    .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let (prev, conv, stateful, started, safety) = &seen[0];
    assert_eq!(prev.as_deref(), Some("resp-789"));
    assert_eq!(*conv, None);
    assert!(*stateful);
    assert!(*started);
    assert_eq!(*safety, ReplaySafety::Unsafe);
}

/// Records every `add_items` batch.
#[derive(Debug)]
struct RecordingSession {
    inner: InMemorySession,
    writes: Mutex<Vec<Vec<Value>>>,
}

#[async_trait]
impl Session for RecordingSession {
    fn session_id(&self) -> &str {
        "recording"
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        self.writes
            .lock()
            .unwrap()
            .push(items.iter().map(|i| i.0.clone()).collect());
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

/// Python: `test_session_add_items_called_multiple_times_for_multi_turn_completion`: the
/// session is written once per turn, in order, after the user input has been written as its own
/// first batch.
#[tokio::test]
async fn the_session_is_written_once_per_turn() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "echo_tool",
        r#"{"text": "foo"}"#,
        "1",
    )]);
    model.add_turn(vec![function_call_with_id(
        "echo_tool",
        r#"{"text": "bar"}"#,
        "2",
    )]);
    model.add_turn(vec![assistant_message("Summary: Echoed foo and bar")]);
    let echo = FunctionTool::<()>::new("echo_tool", "echo", |_c, a: AnyArgs| async move {
        Ok(format!("Echo: {}", a.text))
    });
    let agent = Agent::<()>::new("orchestrator_agent")
        .with_model_instance(model.clone())
        .with_tool(echo);
    let session = Arc::new(RecordingSession {
        inner: InMemorySession::new("recording"),
        writes: Mutex::new(Vec::new()),
    });
    let r = Runner::run_with_options(
        &agent,
        "foo and bar",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        r.final_output_text().as_deref(),
        Some("Summary: Echoed foo and bar")
    );
    let shape: Vec<Vec<String>> = session
        .writes
        .lock()
        .unwrap()
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|i| {
                    i["type"]
                        .as_str()
                        .or_else(|| i["role"].as_str())
                        .unwrap()
                        .to_string()
                })
                .collect()
        })
        .collect();
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(
        shape,
        vec![
            s(&["user"]),
            s(&["function_call", "function_call_output"]),
            s(&["function_call", "function_call_output"]),
            s(&["message"]),
        ]
    );
    let stored: Vec<Value> = session
        .inner
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|i| i.0)
        .collect();
    assert_eq!(stored[2]["output"], "Echo: foo");
    assert_eq!(stored[4]["output"], "Echo: bar");
}

fn trip() -> OutputGuardrail<()> {
    OutputGuardrail::new("trip", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    })
}

fn stop_at_terminal() -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: vec!["terminal_tool".into()],
    })
}

async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> Result<RunResult<()>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(ev) = events.next().await {
            if let Err(e) = ev {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

/// Python: `test_non_streamed_trip_preserves_prior_run_state_side_effect`: a trip on a resumed
/// run keeps the accepted earlier turns and drops the rejected reasoning-bearing response.
#[tokio::test]
async fn a_resumed_trip_keeps_the_prior_state() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "accepted_tool",
        "{}",
        "accepted-call",
    )]);
    model.add_turn(vec![assistant_message("accepted-final")]);
    let log = Log::default();
    let base = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool("accepted_tool", "accepted-output", &log))
        .with_tool(tool("terminal_tool", "rejected-output", &log));
    let first = Runner::run_with_options(
        base.clone(),
        "run accepted tool",
        RunOptions::new(()).max_turns(Some(5)),
    )
    .await
    .unwrap();
    let mut state = first.to_state();
    let prior_items = state.generated_items.len();
    let prior_responses = state.model_responses.len();
    // A finished state is resumed here as Python does: clear the stored final output.
    state.final_output = None;
    model.add_turn(vec![
        openai_agents_sdk::items::OutputItem::from_value(json!({"type": "reasoning",
            "id": "reasoning-current",
            "summary": [{"type": "summary_text", "text": "calling terminal tool"}]})),
        function_call_with_id("terminal_tool", "{}", "current-call"),
    ]);
    let agent = Arc::new(
        base.with_tool_use_behavior(stop_at_terminal())
            .with_output_guardrail(trip()),
    );
    let checkpoint = RunCheckpoint::new();
    let err = Runner::run_with_options(agent, state, RunOptions::new(()).checkpoint(&checkpoint))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        AgentsError::OutputGuardrailTripwireTriggered { .. }
    ));
    assert_eq!(*log.lock().unwrap(), vec!["accepted_tool", "terminal_tool"]);
    let after = checkpoint.take().expect("a checkpoint");
    assert_eq!(after.generated_items.len(), prior_items);
    assert_eq!(after.model_responses.len(), prior_responses);
    let serialized = after.to_json_string().unwrap();
    assert!(serialized.contains("accepted-output"));
    assert!(!serialized.contains("rejected-output"));
    assert!(!serialized.contains("reasoning-current"));
}

/// Python: `test_resumed_trip_preserves_accepted_turns_and_turn_budget`
/// (`streamed` x `handoff_turn`, 4 cases).
#[tokio::test]
async fn a_resumed_trip_keeps_accepted_turns_and_the_turn_budget() {
    for streamed in [false, true] {
        for handoff_turn in [false, true] {
            let label = format!("streamed={streamed} handoff={handoff_turn}");
            let model = setup();
            let log = Log::default();
            let target = Arc::new(
                Agent::<()>::new("target")
                    .with_model_instance(model.clone())
                    .with_tool(tool("terminal_tool", "rejected-secret", &log))
                    .with_tool_use_behavior(stop_at_terminal())
                    .with_output_guardrail(trip()),
            );
            let accepted = tool("accepted_tool", "accepted-output", &log)
                .with_input_guardrail(ToolInputGuardrail::new("in", |_d| async {
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("accepted-input-audit")))
                }))
                .with_output_guardrail(ToolOutputGuardrail::new("out", |_d| async {
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("accepted-output-audit")))
                }));
            let mut source = Agent::<()>::new("source")
                .with_model_instance(model.clone())
                .with_tool(tool("approval_tool", "approved-output", &log).with_needs_approval(true))
                .with_tool(accepted)
                .with_tool(tool("terminal_tool", "rejected-secret", &log))
                .with_tool_use_behavior(stop_at_terminal());
            if handoff_turn {
                source = source.with_handoff(handoff(target));
            } else {
                source = source.with_output_guardrail(trip());
            }
            let source = Arc::new(source);
            model.add_turn(vec![function_call_with_id(
                "approval_tool",
                "{}",
                "approved-call",
            )]);
            let mut accepted_turn = vec![function_call_with_id(
                "accepted_tool",
                "{}",
                "accepted-call",
            )];
            if handoff_turn {
                accepted_turn.push(openai_agents_sdk::testing::function_call(
                    "transfer_to_target",
                    "{}",
                ));
            }
            model.add_turn(accepted_turn);
            model.add_turn(vec![function_call_with_id(
                "terminal_tool",
                "{}",
                "terminal-call",
            )]);

            let interrupted = Runner::run_with_options(
                source.clone(),
                "run approved tools",
                RunOptions::new(()).max_turns(Some(3)),
            )
            .await
            .unwrap();
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let checkpoint = RunCheckpoint::new();
            let err = if streamed {
                let mut stream = Runner::run_streamed_with_options(
                    source.clone(),
                    state,
                    RunOptions::new(()).checkpoint(&checkpoint),
                );
                let mut error = None;
                {
                    let mut events = std::pin::pin!(stream.stream_events());
                    while let Some(ev) = events.next().await {
                        if let Err(e) = ev {
                            error = Some(e);
                        }
                    }
                }
                error.expect("trips")
            } else {
                Runner::run_with_options(
                    source.clone(),
                    state,
                    RunOptions::new(()).checkpoint(&checkpoint),
                )
                .await
                .unwrap_err()
            };
            assert!(
                matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
                "{label}: {err:?}"
            );
            assert_eq!(
                *log.lock().unwrap(),
                vec!["approval_tool", "accepted_tool", "terminal_tool"],
                "{label}"
            );
            let after: RunState = checkpoint.take().expect("snapshot");
            assert_eq!(after.current_turn, 3, "{label}");
            assert_eq!(after.model_responses.len(), 3, "{label}");
            let infos = |r: Vec<Value>| r;
            assert_eq!(
                infos(
                    after
                        .tool_input_guardrail_results
                        .iter()
                        .map(|r| r.output.output_info.clone())
                        .collect()
                ),
                vec![json!("accepted-input-audit")],
                "{label}"
            );
            assert_eq!(
                infos(
                    after
                        .tool_output_guardrail_results
                        .iter()
                        .map(|r| r.output.output_info.clone())
                        .collect()
                ),
                vec![json!("accepted-output-audit")],
                "{label}"
            );
            let outputs: Vec<Value> = after
                .generated_items
                .iter()
                .filter_map(|i| match &i.kind {
                    RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                outputs,
                vec![
                    json!("approved-output"),
                    json!("accepted-output"),
                    json!(OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT)
                ],
                "{label}"
            );
            let serialized = after.to_json_string().unwrap();
            for kept in [
                "approved-output",
                "accepted-output",
                "accepted-input-audit",
                "accepted-output-audit",
            ] {
                assert!(serialized.contains(kept), "{label}: {kept}");
            }
            assert!(!serialized.contains("rejected-secret"), "{label}");

            let again = run_mode(&source, after, RunOptions::new(()), false)
                .await
                .unwrap_err();
            assert!(
                matches!(again, AgentsError::MaxTurnsExceeded { .. }),
                "{label}: {again:?}"
            );
            assert_eq!(log.lock().unwrap().len(), 3, "{label}");
        }
    }
}

// ------------------------------------------------- serialized approval checkpoints

/// Python: `test_serialized_later_turn_approval_with_output_guardrail_resumes`
/// (`mode` x `attach_session` x `mixed_tool_position`, 12 cases): the ownership range written
/// for a later-turn approval, its round trip, and the resumed session order. The
/// `released`/`malformed` boundary-proof halves are run-state wire validation (deferred).
#[tokio::test]
async fn a_serialized_later_turn_approval_resumes_with_an_output_guardrail() {
    for streamed in [false, true] {
        for attach in [false, true] {
            for mixed in [None, Some("before"), Some("after")] {
                let label = format!("streamed={streamed} session={attach} mixed={mixed:?}");
                let model = setup();
                let log = Log::default();
                let mut approval = vec![function_call_with_id(
                    "approval_tool",
                    "{}",
                    "call-approved",
                )];
                if let Some(pos) = mixed {
                    let extra = function_call_with_id("normal_tool", "{}", "call-extra");
                    approval.insert(if pos == "before" { 0 } else { 1 }, extra);
                }
                model.add_turn(vec![function_call_with_id(
                    "normal_tool",
                    "{}",
                    "call-normal",
                )]);
                model.add_turn(approval);
                model.add_turn(vec![assistant_message("done")]);
                let agent = Arc::new(
                    Agent::<()>::new("test")
                        .with_model_instance(model.clone())
                        .with_tool(tool("normal_tool", "normal-result", &log))
                        .with_tool(
                            tool("approval_tool", "approved-result", &log)
                                .with_needs_approval(true),
                        )
                        .with_output_guardrail(OutputGuardrail::new(
                            "pass",
                            |_c, _a, _o: Value| async {
                                Ok(GuardrailFunctionOutput::new(Value::Null, false))
                            },
                        )),
                );
                let session = Arc::new(RecordingSession {
                    inner: InMemorySession::new("recording"),
                    writes: Mutex::new(Vec::new()),
                });
                let options = || {
                    if attach {
                        RunOptions::new(()).session(session.clone() as Arc<dyn Session>)
                    } else {
                        RunOptions::new(())
                    }
                };
                let first = run_mode(
                    &agent,
                    "Use normal_tool, then approval_tool",
                    options(),
                    streamed,
                )
                .await
                .unwrap();
                let state = first.to_state();
                assert_eq!(state.current_turn, 2, "{label}");
                assert_eq!(
                    first
                        .interruptions
                        .iter()
                        .map(|i| i.tool_name.as_str())
                        .collect::<Vec<_>>(),
                    vec!["approval_tool"]
                );
                let normal_calls = if mixed.is_some() { 2 } else { 1 };
                assert_eq!(log.lock().unwrap().len(), normal_calls, "{label}");
                let serialized = state.to_json().unwrap();
                let expected = json!({
                    "start": 2,
                    "end": if mixed.is_none() { 4 } else { 6 },
                    "interruptions": [match mixed { None => 3, Some("before") => 5, _ => 4 }],
                });
                assert_eq!(
                    serialized["current_response_generated_item_ownership"], expected,
                    "{label}"
                );
                let restored = RunState::from_json(serialized.clone()).unwrap();
                assert_eq!(
                    restored.to_json().unwrap()["current_response_generated_item_ownership"],
                    expected
                );
                let mut restored = RunState::from_json(restored.to_json().unwrap()).unwrap();
                let pending = restored.get_interruptions();
                restored.approve(&pending[0], false);
                let resumed = run_mode(&agent, restored, options(), streamed)
                    .await
                    .unwrap();
                assert_eq!(
                    resumed.final_output_text().as_deref(),
                    Some("done"),
                    "{label}"
                );
                assert_eq!(
                    log.lock()
                        .unwrap()
                        .iter()
                        .filter(|n| *n == "approval_tool")
                        .count(),
                    1,
                    "{label}"
                );
                if attach {
                    let stored: Vec<(String, String)> = session
                        .inner
                        .get_items(None)
                        .await
                        .unwrap()
                        .into_iter()
                        .filter(|i| {
                            matches!(
                                i.item_type(),
                                Some("function_call" | "function_call_output")
                            )
                        })
                        .map(|i| {
                            (
                                i.0["type"].as_str().unwrap().to_string(),
                                i.0["call_id"].as_str().unwrap().to_string(),
                            )
                        })
                        .collect();
                    let p = |t: &str, c: &str| (t.to_string(), c.to_string());
                    let mut expected = vec![
                        p("function_call", "call-normal"),
                        p("function_call_output", "call-normal"),
                        p("function_call", "call-approved"),
                        p("function_call_output", "call-approved"),
                    ];
                    if let Some(pos) = mixed {
                        expected.insert(
                            if pos == "before" { 2 } else { 3 },
                            p("function_call", "call-extra"),
                        );
                        expected.insert(4, p("function_call_output", "call-extra"));
                    }
                    assert_eq!(stored, expected, "{label}");
                }
            }
        }
    }
}

/// Python: `test_serialized_filtered_handoff_approval_with_empty_prefix_resumes`
/// (`mode` x `attach_session` x `tripwire`, 8 cases).
#[tokio::test]
async fn a_filtered_handoff_approval_with_an_empty_prefix_resumes() {
    for streamed in [false, true] {
        for attach in [false, true] {
            for tripwire in [false, true] {
                let label = format!("streamed={streamed} session={attach} trip={tripwire}");
                let target_model = setup();
                target_model.add_turn(vec![function_call_with_id(
                    "approval_tool",
                    "{}",
                    "call-approved",
                )]);
                let log = Log::default();
                let target = Arc::new(
                    Agent::<()>::new("target")
                        .with_model_instance(target_model.clone())
                        .with_tool(
                            tool("approval_tool", "approved-secret", &log)
                                .with_needs_approval(true),
                        )
                        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                        .with_output_guardrail(OutputGuardrail::new(
                            "g",
                            move |_c, _a, _o: Value| async move {
                                Ok(GuardrailFunctionOutput::new(Value::Null, tripwire))
                            },
                        )),
                );
                let starting_model = setup();
                starting_model.add_turn(vec![openai_agents_sdk::testing::function_call(
                    "transfer_to_target",
                    "{}",
                )]);
                let starting = Arc::new(
                    Agent::<()>::new("starting")
                        .with_model_instance(starting_model.clone())
                        .with_handoff(handoff(target).with_input_filter(|data| {
                            openai_agents_sdk::handoffs::HandoffInputData {
                                pre_handoff_items: Vec::new(),
                                new_items: Vec::new(),
                                ..data
                            }
                        })),
                );
                let session = Arc::new(RecordingSession {
                    inner: InMemorySession::new("recording"),
                    writes: Mutex::new(Vec::new()),
                });
                let options = || {
                    if attach {
                        RunOptions::new(()).session(session.clone() as Arc<dyn Session>)
                    } else {
                        RunOptions::new(())
                    }
                };
                let first = run_mode(
                    &starting,
                    "Transfer and run approval_tool",
                    options(),
                    streamed,
                )
                .await
                .unwrap();
                let state = first.to_state();
                assert_eq!(state.current_turn, 2, "{label}");
                assert!(log.lock().unwrap().is_empty());
                let ownership = json!({"start": 0, "end": 2, "interruptions": [1]});
                assert_eq!(
                    state.to_json().unwrap()["current_response_generated_item_ownership"],
                    ownership,
                    "{label}"
                );
                let mut restored =
                    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
                assert_eq!(
                    restored.to_json().unwrap()["current_response_generated_item_ownership"],
                    ownership
                );
                let pending = restored.get_interruptions();
                restored.approve(&pending[0], false);
                let checkpoint = RunCheckpoint::new();
                let outcome = run_mode(
                    &starting,
                    restored,
                    options().checkpoint(&checkpoint),
                    streamed,
                )
                .await;
                if tripwire {
                    assert!(
                        matches!(
                            outcome,
                            Err(AgentsError::OutputGuardrailTripwireTriggered { .. })
                        ),
                        "{label}"
                    );
                    let snapshot = checkpoint.take().expect("checkpoint");
                    assert!(
                        !snapshot
                            .to_json_string()
                            .unwrap()
                            .contains("approved-secret"),
                        "{label}"
                    );
                } else {
                    assert_eq!(
                        outcome.unwrap().final_output_text().as_deref(),
                        Some("approved-secret"),
                        "{label}"
                    );
                }
                assert_eq!(log.lock().unwrap().len(), 1, "{label}");
                if attach {
                    let outputs: Vec<(String, Value)> = session
                        .inner
                        .get_items(None)
                        .await
                        .unwrap()
                        .into_iter()
                        .filter(|i| i.item_type() == Some("function_call_output"))
                        .map(|i| {
                            (
                                i.0["call_id"].as_str().unwrap().to_string(),
                                i.0["output"].clone(),
                            )
                        })
                        .collect();
                    let expected = if tripwire {
                        json!(OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT)
                    } else {
                        json!("approved-secret")
                    };
                    assert_eq!(
                        outputs,
                        vec![("call-approved".to_string(), expected)],
                        "{label}"
                    );
                }
            }
        }
    }
}
