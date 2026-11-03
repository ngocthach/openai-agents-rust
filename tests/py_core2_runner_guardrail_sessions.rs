//! Ports of the output-guardrail / session-persistence tests in `tests/test_agent_runner.py`
//! (Python lines ~5320-5800, 687-960) and `tests/test_agent_runner_streamed.py` (lines
//! ~3054-4170): what a tripped, failing or passing output guardrail leaves in the session, the
//! blocked-message placeholder, and server-managed conversation guards.
//!
//! Python tests parametrized on `mode` (or with a streamed twin) run both modes inside one
//! Rust test. Session item kinds are compared the way Python does (`type` or else `role`).

#![allow(clippy::field_reassign_with_default)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::items::{OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::{CancelMode, RunResult};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::{
    OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT, OutputGuardrailBlockedMessage, RunConfig,
};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolOutputGuardrail};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const BLOCKED: &str = OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NoArgs {}

type Log = Arc<Mutex<Vec<String>>>;

fn tool(name: &str, output: &str, log: &Log) -> FunctionTool<()> {
    let (output, log, name_s) = (output.to_string(), log.clone(), name.to_string());
    FunctionTool::new(name, "a tool", move |_ctx, _a: NoArgs| {
        let (output, log, name_s) = (output.clone(), log.clone(), name_s.clone());
        async move {
            log.lock().unwrap().push(name_s);
            Ok(output)
        }
    })
}

fn guardrail(trip: bool) -> OutputGuardrail<()> {
    OutputGuardrail::new("guardrail", move |_c, _a, _o: Value| async move {
        Ok(GuardrailFunctionOutput::new(Value::Null, trip))
    })
}

fn failing_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("failing", |_c, _a, _o: Value| async {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
    })
}

fn reasoning(id: &str, text: &str) -> OutputItem {
    OutputItem::from_value(json!({"type": "reasoning", "id": id,
        "summary": [{"type": "summary_text", "text": text}]}))
}

/// A list session that may report remote history ownership.
#[derive(Debug)]
struct ListSession {
    inner: InMemorySession,
    remote: bool,
}

impl ListSession {
    fn new(remote: bool) -> Arc<Self> {
        Arc::new(ListSession {
            inner: InMemorySession::new("list"),
            remote,
        })
    }
    async fn all(&self) -> Vec<Value> {
        self.inner
            .get_items(None)
            .await
            .unwrap()
            .into_iter()
            .map(|i| i.0)
            .collect()
    }
    async fn kinds(&self) -> Vec<String> {
        kinds(&self.all().await)
    }
}

#[async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        "list"
    }
    fn manages_history_remotely(&self) -> bool {
        self.remote
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

fn kinds(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|i| {
            i["type"]
                .as_str()
                .or_else(|| i["role"].as_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
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

fn with_session(session: &Arc<ListSession>) -> RunOptions<()> {
    RunOptions::new(()).session(session.clone())
}

// ------------------------------------------------------------- tripped final messages

/// Python: `test_output_guardrail_tripwire_triggered_causes_exception`,
/// `test_blocked_message_final_output_is_not_persisted[non_streamed|streamed]`.
#[tokio::test]
async fn a_blocked_message_is_never_persisted() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![assistant_message("should_not_be_saved")]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_output_guardrail(guardrail(true)),
        );
        let session = ListSession::new(false);
        let err = run_mode(&agent, "user_message", with_session(&session), streamed)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            AgentsError::OutputGuardrailTripwireTriggered { .. }
        ));
        assert_eq!(session.kinds().await, s(&["user"]), "streamed={streamed}");
    }
}

/// Python: `test_output_guardrail_tripwire_does_not_save_assistant_message_to_session_sync`.
#[test]
fn a_blocked_message_is_never_persisted_by_run_sync() {
    let model = setup();
    model.add_turn(vec![assistant_message("should_not_be_saved")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_guardrail(guardrail(true));
    let session = ListSession::new(false);
    let err =
        Runner::run_sync_with_options(&agent, "user_message", with_session(&session)).unwrap_err();
    assert!(matches!(
        err,
        AgentsError::OutputGuardrailTripwireTriggered { .. }
    ));
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    assert_eq!(rt.block_on(session.kinds()), s(&["user"]));
}

/// Python: `test_output_guardrail_error_preserves_final_output_in_session[False|True]`.
#[tokio::test]
async fn a_failing_output_guardrail_keeps_the_final_message() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![assistant_message("preserved_on_guardrail_error")]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_output_guardrail(failing_guardrail()),
        );
        let session = ListSession::new(false);
        let err = run_mode(&agent, "user_message", with_session(&session), streamed)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("guardrail failed"), "{err}");
        assert_eq!(
            session.kinds().await,
            s(&["user", "message"]),
            "streamed={streamed}"
        );
    }
}

/// Python: `test_input_guardrail_no_tripwire_continues_execution`,
/// `test_output_guardrail_no_tripwire_continues_execution`.
#[tokio::test]
async fn passing_guardrails_let_the_run_finish() {
    let model = setup();
    model.add_turn(vec![assistant_message("response")]);
    model.add_turn(vec![assistant_message("response")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_input_guardrail(openai_agents_sdk::InputGuardrail::new(
            "pass",
            |_c, _a, _i| async { Ok(GuardrailFunctionOutput::new(Value::Null, false)) },
        ));
    let r = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("response"));
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_guardrail(guardrail(false));
    let r = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("response"));
}

// --------------------------------------------------------- committed terminal tools

/// Python: `test_stop_on_first_tool_final_persists_committed_tool_items_on_tripwire`
/// (`mode` x `session_kind` x `arguments`, 8 cases).
#[tokio::test]
async fn a_tripped_terminal_tool_keeps_its_committed_call_and_placeholder() {
    for streamed in [false, true] {
        for remote in [false, true] {
            for arguments in ["{}", ""] {
                let label = format!("streamed={streamed} remote={remote} args={arguments:?}");
                let model = setup();
                model.add_turn(vec![function_call_with_id(
                    "commit_tool",
                    arguments,
                    "call-committed",
                )]);
                let log = Log::default();
                let agent = Agent::<()>::new("test")
                    .with_model_instance(model.clone())
                    .with_tool(tool("commit_tool", "committed-result", &log))
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
                let tripping = Arc::new(agent.clone().with_output_guardrail(guardrail(true)));
                let session = ListSession::new(remote);
                let config = || {
                    let mut c = RunConfig::default();
                    c.session_input_callback = Some(Arc::new(|history, new| {
                        let mut out: Vec<InputItem> = history.into_iter().rev().collect();
                        out.extend(new);
                        Box::pin(async move { Ok(out) })
                    }));
                    c
                };
                let err = run_mode(
                    &tripping,
                    "Use commit_tool",
                    with_session(&session).run_config(config()),
                    streamed,
                )
                .await
                .unwrap_err();
                assert!(
                    matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
                    "{label}: {err:?}"
                );
                assert_eq!(*log.lock().unwrap(), s(&["commit_tool"]), "{label}");
                let saved = session.all().await;
                let pairs: Vec<(String, Option<String>)> = saved
                    .iter()
                    .zip(kinds(&saved))
                    .map(|(i, k)| (k, i["call_id"].as_str().map(str::to_string)))
                    .collect();
                assert_eq!(
                    pairs,
                    vec![
                        ("user".to_string(), None),
                        ("function_call".to_string(), Some("call-committed".into())),
                        (
                            "function_call_output".to_string(),
                            Some("call-committed".into())
                        ),
                    ],
                    "{label}"
                );
                assert_eq!(saved.last().unwrap()["output"], BLOCKED);
                assert!(!json!(saved).to_string().contains("committed-result"));

                // The next run sees the completed call instead of re-issuing it.
                model.add_turn(vec![assistant_message("done")]);
                let plain = Arc::new(agent);
                let followup = run_mode(
                    &plain,
                    "Continue",
                    with_session(&session).run_config(config()),
                    streamed,
                )
                .await
                .unwrap();
                assert_eq!(followup.final_output_text().as_deref(), Some("done"));
                assert_eq!(log.lock().unwrap().len(), 1, "{label}");
                let input: Vec<Value> = model
                    .calls()
                    .last()
                    .unwrap()
                    .input
                    .iter()
                    .map(|i| i.0.clone())
                    .collect();
                let replayed: Vec<&Value> = input
                    .iter()
                    .filter(|i| i["call_id"] == "call-committed")
                    .collect();
                assert_eq!(replayed.len(), 2, "{label}: {input:?}");
                let output = replayed
                    .iter()
                    .find(|i| i["type"] == "function_call_output")
                    .unwrap();
                assert_eq!(output["output"], BLOCKED);
                assert!(!json!(input).to_string().contains("committed-result"));
            }
        }
    }
}

fn secret_tool_agent(model: &Arc<FakeModel>, guard: OutputGuardrail<()>) -> Agent<()> {
    let log = Log::default();
    Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(
            tool("secret_tool", "blocked-secret", &log).with_output_guardrail(
                ToolOutputGuardrail::new("retain", |data| async move {
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(data.output.to_json_value()))
                }),
            ),
        )
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_output_guardrail(guard)
}

fn rejecting_with_info() -> OutputGuardrail<()> {
    OutputGuardrail::new("localized_guardrail", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(
            json!({"secret": "blocked-output-info"}),
            true,
        ))
    })
}

/// Python: `test_terminal_tool_trip_uses_custom_blocked_message_everywhere`
/// (`mode` x `customizer_kind`).
#[tokio::test]
async fn a_custom_blocked_message_replaces_the_tool_output_everywhere() {
    let blocked = "出力ガードレールにより非表示になりました。";
    for streamed in [false, true] {
        for fixed in [true, false] {
            let label = format!("streamed={streamed} fixed={fixed}");
            let model = setup();
            model.add_turn(vec![function_call_with_id(
                "secret_tool",
                "{}",
                "call-secret",
            )]);
            let agent = Arc::new(secret_tool_agent(&model, rejecting_with_info()));
            let calls = Arc::new(Mutex::new(Vec::<(String, String, String)>::new()));
            let config = || {
                let mut c = RunConfig::default();
                c.output_guardrail_blocked_message = Some(if fixed {
                    OutputGuardrailBlockedMessage::Text(blocked.to_string())
                } else {
                    let calls = calls.clone();
                    OutputGuardrailBlockedMessage::Formatter(Arc::new(move |args| {
                        calls.lock().unwrap().push((
                            args.default_message.clone(),
                            args.guardrail_name.clone(),
                            args.agent_name().to_string(),
                        ));
                        Some(blocked.to_string())
                    }))
                });
                c
            };
            let session = ListSession::new(false);
            let checkpoint = RunCheckpoint::new();
            let mut stream_state = None;
            let err = if streamed {
                let mut stream = Runner::run_streamed_with_options(
                    agent.clone(),
                    "run",
                    with_session(&session).run_config(config()),
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
                stream_state = stream.to_state();
                error.expect("the guardrail trips")
            } else {
                Runner::run_with_options(
                    agent.clone(),
                    "run",
                    with_session(&session)
                        .run_config(config())
                        .checkpoint(&checkpoint),
                )
                .await
                .unwrap_err()
            };
            let AgentsError::OutputGuardrailTripwireTriggered {
                guardrail_result, ..
            } = &err
            else {
                panic!("{label}: {err:?}");
            };
            assert_eq!(guardrail_result.agent_output, json!(blocked), "{label}");
            assert_eq!(guardrail_result.output.output_info, Value::Null);
            let saved = session.all().await;
            assert_eq!(saved.last().unwrap()["output"], blocked, "{label}");
            let text = json!(saved).to_string();
            assert!(!text.contains("blocked-secret") && !text.contains("blocked-output-info"));
            if fixed {
                assert!(calls.lock().unwrap().is_empty());
            } else {
                assert_eq!(
                    *calls.lock().unwrap(),
                    vec![(
                        BLOCKED.to_string(),
                        "localized_guardrail".to_string(),
                        "test".to_string()
                    )]
                );
            }
            if let Some(state) = stream_state.or_else(|| checkpoint.take()) {
                let outputs: Vec<Value> = state
                    .generated_items
                    .iter()
                    .filter_map(|i| match &i.kind {
                        RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(outputs, vec![json!(blocked)], "{label}");
                if streamed {
                    assert_eq!(
                        state.tool_output_guardrail_results[0].output.output_info,
                        json!(blocked)
                    );
                }
                let serialized = state.to_json_string().unwrap();
                assert!(!serialized.contains("blocked-secret"), "{label}");
                assert!(!serialized.contains("blocked-output-info"), "{label}");
            }
            // The placeholder is what the next run replays.
            model.add_turn(vec![assistant_message("done")]);
            let plain = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
            let followup = run_mode(
                &plain,
                "continue",
                with_session(&session).run_config(config()),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(followup.final_output_text().as_deref(), Some("done"));
            let last = json!(model.calls().last().unwrap().input).to_string();
            assert!(last.contains(blocked), "{label}");
        }
    }
}

/// Python: `test_terminal_tool_trip_falls_back_when_blocked_message_formatter_fails` for the
/// `none` and `empty` outcomes (the other outcomes return values Rust's `Option<String>`
/// formatter cannot produce).
#[tokio::test]
async fn a_formatter_without_a_message_falls_back_to_the_default() {
    for streamed in [false, true] {
        for empty in [false, true] {
            let label = format!("streamed={streamed} empty={empty}");
            let model = setup();
            model.add_turn(vec![function_call_with_id(
                "secret_tool",
                "{}",
                "call-secret",
            )]);
            let agent = Arc::new(secret_tool_agent(
                &model,
                OutputGuardrail::new("g", |_c, _a, _o: Value| async {
                    Ok(GuardrailFunctionOutput::new(
                        json!("blocked-output-info"),
                        true,
                    ))
                }),
            ));
            let count = Arc::new(AtomicUsize::new(0));
            let c2 = count.clone();
            let mut config = RunConfig::default();
            config.output_guardrail_blocked_message = Some(
                OutputGuardrailBlockedMessage::Formatter(Arc::new(move |_args| {
                    c2.fetch_add(1, Ordering::SeqCst);
                    empty.then(String::new)
                })),
            );
            let session = ListSession::new(false);
            let err = run_mode(
                &agent,
                "run",
                with_session(&session).run_config(config),
                streamed,
            )
            .await
            .unwrap_err();
            let AgentsError::OutputGuardrailTripwireTriggered {
                guardrail_result, ..
            } = &err
            else {
                panic!("{label}: {err:?}");
            };
            assert_eq!(count.load(Ordering::SeqCst), 1, "{label}");
            assert_eq!(guardrail_result.agent_output, json!(BLOCKED));
            assert_eq!(guardrail_result.output.output_info, Value::Null);
            let saved = session.all().await;
            assert_eq!(saved.last().unwrap()["output"], BLOCKED);
            let text = json!(saved).to_string();
            assert!(!text.contains("blocked-secret") && !text.contains("blocked-output-info"));
        }
    }
}

/// Python: `test_blocked_final_persists_tool_items_but_not_the_message[non_streamed|streamed]`.
#[tokio::test]
async fn a_blocked_final_message_keeps_the_earlier_tool_record() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "commit_tool",
            "{}",
            "call-mixed",
        )]);
        model.add_turn(vec![assistant_message("should_not_be_saved")]);
        let log = Log::default();
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(tool("commit_tool", "committed-result", &log))
                .with_output_guardrail(OutputGuardrail::new("g", |_c, _a, _o: Value| async {
                    Ok(GuardrailFunctionOutput::new(
                        json!({"reason": "message rejected"}),
                        true,
                    ))
                })),
        );
        let session = ListSession::new(false);
        let err = run_mode(&agent, "Use commit_tool", with_session(&session), streamed)
            .await
            .unwrap_err();
        let AgentsError::OutputGuardrailTripwireTriggered {
            guardrail_result, ..
        } = &err
        else {
            panic!("{err:?}");
        };
        assert_eq!(guardrail_result.agent_output, json!("should_not_be_saved"));
        assert_eq!(
            guardrail_result.output.output_info,
            json!({"reason": "message rejected"})
        );
        assert_eq!(
            session.kinds().await,
            s(&["user", "function_call", "function_call_output"]),
            "streamed={streamed}"
        );
    }
}

fn preamble_then_tool(model: &Arc<FakeModel>) {
    model.add_turn(vec![
        assistant_message("assistant-preamble"),
        function_call_with_id("commit_tool", "{}", "call-mixed"),
    ]);
}

/// Python: `test_mixed_final_turn_session_order_and_committed_items` (`mode` x `tripwire`).
#[tokio::test]
async fn a_mixed_final_turn_keeps_model_order_or_drops_only_the_message() {
    for streamed in [false, true] {
        for trip in [false, true] {
            let model = setup();
            preamble_then_tool(&model);
            let log = Log::default();
            let agent = Arc::new(
                Agent::<()>::new("test")
                    .with_model_instance(model.clone())
                    .with_tool(tool("commit_tool", "committed-result", &log))
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                    .with_output_guardrail(guardrail(trip)),
            );
            let session = ListSession::new(false);
            let outcome =
                run_mode(&agent, "Use commit_tool", with_session(&session), streamed).await;
            let label = format!("streamed={streamed} trip={trip}");
            if trip {
                assert!(outcome.is_err(), "{label}");
                assert_eq!(
                    session.kinds().await,
                    s(&["user", "function_call", "function_call_output"]),
                    "{label}"
                );
            } else {
                assert_eq!(
                    outcome.unwrap().final_output_text().as_deref(),
                    Some("committed-result")
                );
                assert_eq!(
                    session.kinds().await,
                    s(&["user", "message", "function_call", "function_call_output"]),
                    "{label}"
                );
            }
        }
    }
}

/// Python: `test_failing_output_guardrail_keeps_the_whole_final_turn[non_streamed|streamed]`.
#[tokio::test]
async fn a_failing_guardrail_keeps_the_whole_mixed_final_turn() {
    for streamed in [false, true] {
        let model = setup();
        preamble_then_tool(&model);
        let log = Log::default();
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(tool("commit_tool", "committed-result", &log))
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                .with_output_guardrail(failing_guardrail()),
        );
        let session = ListSession::new(false);
        let err = run_mode(&agent, "Use commit_tool", with_session(&session), streamed)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("guardrail failed"));
        assert_eq!(
            session.kinds().await,
            s(&["user", "message", "function_call", "function_call_output"]),
            "streamed={streamed}"
        );
    }
}

/// Python: `test_blocked_tool_final_discards_reasoning_response_suffix_on_trip`
/// (`mode` x `tripwire`).
#[tokio::test]
async fn a_reasoning_bearing_terminal_response_is_dropped_whole_on_a_trip() {
    for streamed in [false, true] {
        for trip in [false, true] {
            let label = format!("streamed={streamed} trip={trip}");
            let model = setup();
            model.add_turn(vec![
                reasoning("rs_committed", "deciding to call the tool"),
                function_call_with_id("commit_tool", "{}", "call-reasoned"),
            ]);
            let log = Log::default();
            let base = Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(tool("commit_tool", "committed-result", &log))
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
            let agent = Arc::new(base.clone().with_output_guardrail(guardrail(trip)));
            let session = ListSession::new(false);
            let outcome =
                run_mode(&agent, "Use commit_tool", with_session(&session), streamed).await;
            let expected = if trip {
                assert!(outcome.is_err(), "{label}");
                s(&["user"])
            } else {
                assert_eq!(
                    outcome.unwrap().final_output_text().as_deref(),
                    Some("committed-result")
                );
                s(&["user", "reasoning", "function_call", "function_call_output"])
            };
            assert_eq!(session.kinds().await, expected, "{label}");

            model.add_turn(vec![assistant_message("done")]);
            let followup = run_mode(
                &Arc::new(base),
                "Continue",
                with_session(&session),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(followup.final_output_text().as_deref(), Some("done"));
            let replayed: Vec<String> = model
                .calls()
                .last()
                .unwrap()
                .input
                .iter()
                .filter_map(|i| i.item_type().map(str::to_string))
                .filter(|t| {
                    ["reasoning", "function_call", "function_call_output"].contains(&t.as_str())
                })
                .collect();
            let expected: Vec<String> = if trip {
                vec![]
            } else {
                s(&["reasoning", "function_call", "function_call_output"])
            };
            assert_eq!(replayed, expected, "{label}");
        }
    }
}

/// Python: `test_blocked_tool_final_discards_suffix_with_multiple_reasoning_groups`.
#[tokio::test]
async fn several_reasoning_groups_are_dropped_whole_on_a_trip() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![
            reasoning("rs_rejected", "drafting the message"),
            assistant_message("rejected-preamble"),
            reasoning("rs_committed", "deciding to call the tool"),
            function_call_with_id("commit_tool", "{}", "call-reasoned"),
        ]);
        let log = Log::default();
        let base = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("commit_tool", "committed-result", &log))
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
        let agent = Arc::new(base.clone().with_output_guardrail(guardrail(true)));
        let session = ListSession::new(false);
        assert!(
            run_mode(&agent, "Use commit_tool", with_session(&session), streamed)
                .await
                .is_err()
        );
        let saved = session.all().await;
        assert_eq!(kinds(&saved), s(&["user"]), "streamed={streamed}");
        assert!(saved.iter().all(|i| i.get("id").is_none()));
        model.add_turn(vec![assistant_message("done")]);
        run_mode(
            &Arc::new(base),
            "Continue",
            with_session(&session),
            streamed,
        )
        .await
        .unwrap();
        let replayed = model
            .calls()
            .last()
            .unwrap()
            .input
            .iter()
            .filter(|i| {
                matches!(
                    i.item_type(),
                    Some("reasoning" | "message" | "function_call" | "function_call_output")
                )
            })
            .count();
        assert_eq!(replayed, 0, "streamed={streamed}");
    }
}

/// Python: `test_streamed_trip_preserves_accepted_tool_prefix[canonical|reasoning]` (and the
/// non-streamed `test_non_streamed_trip_preserves_prior_run_state_side_effect` state checks).
#[tokio::test]
async fn a_trip_keeps_the_accepted_tool_prefix() {
    for streamed in [false, true] {
        for reasoning_suffix in [false, true] {
            let label = format!("streamed={streamed} reasoning={reasoning_suffix}");
            let model = setup();
            model.add_turn(vec![function_call_with_id(
                "accepted_tool",
                "{}",
                "accepted-call",
            )]);
            let mut terminal = Vec::new();
            if reasoning_suffix {
                terminal.push(reasoning("reasoning-current", "calling terminal tool"));
            }
            terminal.push(function_call_with_id("terminal_tool", "{}", "current-call"));
            model.add_turn(terminal);
            let log = Log::default();
            let agent = Arc::new(
                Agent::<()>::new("test")
                    .with_model_instance(model.clone())
                    .with_tool(tool("accepted_tool", "accepted-output", &log))
                    .with_tool(tool("terminal_tool", "rejected-output", &log))
                    .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                        stop_at_tool_names: vec!["terminal_tool".into()],
                    }))
                    .with_output_guardrail(guardrail(true)),
            );
            let checkpoint = RunCheckpoint::new();
            let err = if streamed {
                let mut stream = Runner::run_streamed_with_options(
                    agent.clone(),
                    "run both tools",
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
                    agent.clone(),
                    "run both tools",
                    RunOptions::new(()).checkpoint(&checkpoint),
                )
                .await
                .unwrap_err()
            };
            assert!(
                matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
                "{label}"
            );
            let state: RunState = checkpoint.take().expect("a checkpoint");
            assert_eq!(
                *log.lock().unwrap(),
                s(&["accepted_tool", "terminal_tool"]),
                "{label}"
            );
            let ids: Vec<String> = state
                .generated_items
                .iter()
                .filter_map(|i| {
                    let v = serde_json::to_value(i.to_input_item()).ok()?;
                    v["call_id"].as_str().map(str::to_string)
                })
                .collect();
            let mut expected = s(&["accepted-call", "accepted-call"]);
            if !reasoning_suffix {
                expected.extend(s(&["current-call", "current-call"]));
            }
            assert_eq!(ids, expected, "{label}");
            let serialized = state.to_json_string().unwrap();
            assert!(serialized.contains("accepted-output"), "{label}");
            assert!(!serialized.contains("rejected-output"), "{label}");
            if reasoning_suffix {
                assert!(!serialized.contains("reasoning-current"), "{label}");
            } else {
                assert!(serialized.contains(BLOCKED), "{label}");
            }
        }
    }
}

// ------------------------------------------------- server-managed conversation guards

/// Python: `test_output_guardrails_fail_closed_with_server_managed_history[non_streamed|streamed]`.
#[tokio::test]
async fn output_guardrails_refuse_server_managed_history() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![assistant_message("unreachable")]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_output_guardrail(guardrail(false)),
        );
        let err = run_mode(
            &agent,
            "hello",
            RunOptions::new(()).previous_response_id("response-id"),
            streamed,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("server-managed conversation history"),
            "{err}"
        );
        assert!(model.calls().is_empty());
    }
}

/// Python: `test_run_rejects_session_with_server_managed_conversation` (3 cases),
/// `test_run_rejects_session_with_resumed_conversation_state`, and the streamed
/// `test_run_streamed_rejects_session_*` twins.
#[tokio::test]
async fn sessions_cannot_combine_with_server_managed_conversations() {
    for streamed in [false, true] {
        for case in 0..4 {
            let model = setup();
            model.add_turn(vec![assistant_message("done")]);
            let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
            let session = ListSession::new(false);
            let options = with_session(&session);
            let (options, input): (RunOptions<()>, RunInput) = match case {
                0 => (options.conversation_id("conv-test"), "test".into()),
                1 => (options.previous_response_id("resp-test"), "test".into()),
                2 => (options.auto_previous_response_id(true), "test".into()),
                _ => {
                    let state = RunState {
                        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION
                            .to_string(),
                        current_agent_name: "test".into(),
                        original_input: vec![InputItem::user("hello")],
                        conversation_id: Some("conv-test".into()),
                        ..Default::default()
                    };
                    (options, state.into())
                }
            };
            let err = run_mode(&agent, input, options, streamed)
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("Session persistence"),
                "case {case} streamed={streamed}: {err}"
            );
            assert!(model.calls().is_empty());
        }
    }
}

// ------------------------------------------------------------------ tool use behavior

#[derive(Debug, Deserialize, JsonSchema, PartialEq, serde::Serialize)]
struct Foo {
    bar: String,
}

/// Python: `test_tool_use_behavior_first_output`.
#[tokio::test]
async fn stop_on_first_tool_uses_the_first_tool_output() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call_with_id("test_tool_one", "{}", "tool-one"),
        function_call_with_id("test_tool_two", "{}", "tool-two"),
    ]);
    let log = Log::default();
    let one = FunctionTool::<()>::new("test_tool_one", "one", |_c, _a: NoArgs| async {
        Ok(json!({"bar": "tool_one_result"}))
    });
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool("foo", "tool_result", &log))
        .with_tool(one)
        .with_tool(tool("test_tool_two", "tool_two_result", &log))
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_output_type::<Foo>();
    let result = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(
        result.final_output_as::<Foo>().unwrap(),
        Foo {
            bar: "tool_one_result".into()
        }
    );
}

/// Python: `test_tool_use_behavior_custom_function`.
#[tokio::test]
async fn a_custom_tool_use_behavior_decides_the_final_output() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call_with_id("test_tool_two", "{}", "call-tool-two-first"),
    ]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call_with_id("test_tool_one", "{}", "call-tool-one"),
        function_call_with_id("test_tool_two", "{}", "call-tool-two-second"),
    ]);
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool("foo", "tool_result", &log))
        .with_tool(tool("test_tool_one", "one", &log))
        .with_tool(tool("test_tool_two", "two", &log))
        .with_tool_use_behavior(ToolUseBehavior::Custom(Arc::new(|_ctx, results| {
            let hit = results.iter().any(|r| r.tool_name == "test_tool_one");
            Box::pin(async move {
                Ok(if hit {
                    ToolsToFinalOutputResult {
                        is_final_output: true,
                        final_output: Some(json!("the_final_output")),
                    }
                } else {
                    ToolsToFinalOutputResult {
                        is_final_output: false,
                        final_output: None,
                    }
                })
            })
        })));
    let result = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("the_final_output")
    );
}

// --------------------------------------------------------- resumed final-turn writes

/// Run streamed until the first tool output, then stop after that turn.
async fn stop_after_first_tool(agent: &Arc<Agent<()>>, session: &Arc<ListSession>) -> RunState {
    let mut stream =
        Runner::run_streamed_with_options(agent.clone(), "user_message", with_session(session));
    let handle = stream.cancel_handle();
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(ev) = events.next().await {
            if let Ok(StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            }) = ev
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    stream.to_state().expect("a resumable snapshot")
}

#[derive(Deserialize, JsonSchema)]
struct AArg {
    a: String,
}

/// Python: `test_resumed_final_output_persists_once_after_passing_output_guardrail`.
#[tokio::test]
async fn a_resumed_final_output_is_persisted_once() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("foo", r#"{"a":"b"}"#, "c1")]);
    let foo = FunctionTool::<()>::new("foo", "foo", |_c, a: AArg| async move {
        Ok(format!("result:{}", a.a))
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(foo)
            .with_output_guardrail(guardrail(false)),
    );
    let session = ListSession::new(false);
    let state = stop_after_first_tool(&agent, &session).await;
    let before = session.all().await;
    model.add_turn(vec![assistant_message("accepted_final")]);
    let resumed = Runner::run_with_options(agent, state, with_session(&session))
        .await
        .unwrap();
    assert_eq!(
        resumed.final_output_text().as_deref(),
        Some("accepted_final")
    );
    let after = session.all().await;
    assert_eq!(after[..before.len()], before[..]);
    let assistant: Vec<&Value> = after
        .iter()
        .filter(|i| i["role"] == "assistant" && i["type"] == "message")
        .collect();
    assert_eq!(assistant.len(), 1);
    assert!(
        assistant[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["text"] == "accepted_final")
    );
}

/// Python: `test_resumed_final_tool_persists_call_and_output_after_output_guardrail`
/// (`tripwire_triggered`).
#[tokio::test]
async fn a_resumed_terminal_tool_persists_its_call_and_output() {
    for trip in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "commit_tool",
            "{}",
            "call-first",
        )]);
        let log = Log::default();
        let base = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("commit_tool", "committed-result", &log))
            .with_output_guardrail(guardrail(trip));
        let session = ListSession::new(false);
        let state = stop_after_first_tool(&Arc::new(base.clone()), &session).await;
        let agent = Arc::new(base.with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool));
        model.add_turn(vec![function_call_with_id(
            "commit_tool",
            "{}",
            "call-second",
        )]);
        let outcome = Runner::run_with_options(agent, state, with_session(&session)).await;
        if trip {
            assert!(matches!(
                outcome,
                Err(AgentsError::OutputGuardrailTripwireTriggered { .. })
            ));
        } else {
            assert_eq!(
                outcome.unwrap().final_output_text().as_deref(),
                Some("committed-result")
            );
        }
        let items = session.all().await;
        let pairs: Vec<(String, Option<String>)> = items
            .iter()
            .zip(kinds(&items))
            .map(|(i, k)| (k, i["call_id"].as_str().map(str::to_string)))
            .collect();
        let p = |k: &str, c: Option<&str>| (k.to_string(), c.map(str::to_string));
        assert_eq!(
            pairs,
            vec![
                p("user", None),
                p("function_call", Some("call-first")),
                p("function_call_output", Some("call-first")),
                p("function_call", Some("call-second")),
                p("function_call_output", Some("call-second")),
            ],
            "trip={trip}"
        );
        let last = &items.last().unwrap()["output"];
        assert_eq!(
            last,
            &json!(if trip { BLOCKED } else { "committed-result" })
        );
        if trip {
            assert!(!json!(items[3..]).to_string().contains("committed-result"));
        }
    }
}

// ------------------------------------------------------------- blocked-write failures

/// A session that fails to write the blocked terminal tool output.
#[derive(Debug)]
struct FailingBlockedSession {
    inner: InMemorySession,
}

#[async_trait]
impl Session for FailingBlockedSession {
    fn session_id(&self) -> &str {
        "failing"
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        if items
            .iter()
            .any(|i| i.0["type"] == "function_call_output" && i.0["call_id"] == "terminal-call")
        {
            return Err(AgentsError::other("sandbox-memory-session-failure-secret"));
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

/// Python: `test_non_streamed_trip_uses_safe_items_for_sandbox_memory_after_session_failure`
/// (the session-failure half; Rust has no sandbox memory payload hook on the runner).
#[tokio::test]
async fn a_failed_blocked_write_surfaces_without_the_secret() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "accepted_tool",
        "{}",
        "accepted-call",
    )]);
    model.add_turn(vec![function_call_with_id(
        "terminal_tool",
        "{}",
        "terminal-call",
    )]);
    let log = Log::default();
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("accepted_tool", "accepted-tool-output", &log))
            .with_tool(tool(
                "terminal_tool",
                "sandbox-memory-tool-output-secret",
                &log,
            ))
            .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                stop_at_tool_names: vec!["terminal_tool".into()],
            }))
            .with_output_guardrail(guardrail(true)),
    );
    let session = Arc::new(FailingBlockedSession {
        inner: InMemorySession::new("failing"),
    });
    let err = Runner::run_with_options(
        agent,
        "run terminal tool",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap_err();
    let text = format!("{err:?}");
    assert!(
        !text.contains("sandbox-memory-tool-output-secret"),
        "{text}"
    );
}
