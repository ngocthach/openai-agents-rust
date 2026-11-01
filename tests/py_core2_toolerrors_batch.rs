//! Ports of `test_tool_batch_failure_history.py` and `test_tool_failure_boundaries.py`:
//! completed side effects survive a sibling failure within the same model turn.
//!
//! Python's `asyncio.Event` sequencing maps to [`Event`] below; Python's `SQLiteSession` maps
//! to `InMemorySession` (the persistence contract under test is the runner's, not the store's).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::error::RunErrorDetails;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{InputItem, OutputItem, RunInput};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, ToolOutputGuardrail,
    ToolOutputGuardrailData,
};
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, InputGuardrail, OutputGuardrail, Result, RunHooks,
    RunState, Runner, Session,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `asyncio.Event`.
#[derive(Default)]
struct Event {
    flag: AtomicBool,
    notify: Notify,
}

impl Event {
    fn new() -> Arc<Self> {
        Arc::new(Event::default())
    }
    fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            futures::pin_mut!(notified);
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

/// `_shape`: `type:call_id` for call items, otherwise the type (or role).
fn shape(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|item| match item.get("call_id").and_then(Value::as_str) {
            Some(call_id) => format!("{}:{call_id}", item["type"].as_str().unwrap_or("?")),
            None => item
                .get("type")
                .or_else(|| item.get("role"))
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        })
        .collect()
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.as_value().clone()).collect()
}

fn run_data_items(details: &RunErrorDetails) -> Vec<Value> {
    details
        .new_items
        .iter()
        .map(|i| i.to_input_item().0)
        .collect()
}

fn s(items: &[&str]) -> Vec<String> {
    items.iter().map(|i| i.to_string()).collect()
}

fn call(name: &str, id: &str) -> OutputItem {
    function_call_full(name, "{}", id, None, None)
}

fn reasoning(id: &str) -> OutputItem {
    OutputItem::from_value(json!({"type": "reasoning", "id": id, "summary": []}))
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

fn last_input(model: &ScriptedModel) -> Vec<Value> {
    values(&model.last_call().expect("a model call").request.input)
}

async fn session_items(session: &dyn Session) -> Vec<Value> {
    values(&session.get_items(None).await.unwrap())
}

fn ok_tool(
    name: &str,
    output: &'static str,
    effects: Option<Arc<Mutex<Vec<String>>>>,
) -> FunctionTool<()> {
    let effect = name.to_string();
    FunctionTool::new_no_args(name, "tool", move |_ctx| {
        let effects = effects.clone();
        let effect = effect.clone();
        async move {
            if let Some(e) = effects {
                e.lock().unwrap().push(effect);
            }
            Ok(output.to_string())
        }
    })
}

/// A tool that waits for `gate`, then fails with `message` (Python's `failure_error_function=None`).
fn failing_after(name: &str, gate: Arc<Event>, message: &'static str) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "fails", move |_ctx| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err::<String, _>(AgentsError::other(message))
        }
    })
    .without_failure_error_function()
}

/// `RunHooks` whose `on_tool_end` sets `event` when `tool_name` (or any tool) finishes.
struct EndHook {
    tool_name: Option<String>,
    event: Arc<Event>,
    /// Fails the run from `on_agent_end`: always for `Some(None)`, and only once the shared
    /// effects list holds the ticket for `Some(Some(effects))`.
    final_fails: Option<Option<Arc<Mutex<Vec<String>>>>>,
    /// Fails the run from `on_tool_end` for every tool but `tool_name`.
    fail_other_tools: bool,
}

#[async_trait::async_trait]
impl RunHooks<()> for EndHook {
    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) -> Result<()> {
        if self.tool_name.as_deref().is_none_or(|n| n == tool.name()) {
            self.event.set();
        } else if self.fail_other_tools {
            return Err(AgentsError::user("synthetic hook failure"));
        }
        Ok(())
    }

    async fn try_on_agent_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _output: &Value,
    ) -> Result<()> {
        let fails = match &self.final_fails {
            None => false,
            Some(None) => true,
            Some(Some(effects)) => *effects.lock().unwrap() == vec!["create_ticket".to_string()],
        };
        if fails {
            return Err(AgentsError::user("final hook failure"));
        }
        Ok(())
    }
}

fn end_hook(tool_name: Option<&str>, event: Arc<Event>) -> Arc<dyn RunHooks<()>> {
    Arc::new(EndHook {
        tool_name: tool_name.map(str::to_string),
        event,
        final_fails: None,
        fail_other_tools: false,
    })
}

/// A session whose writes fail while `predicate` matches (Python's `FailingSession`s).
struct FailingSession {
    inner: InMemorySession,
    fail_on_call_id: Option<String>,
    fail_on_tool_output: bool,
    fail_once: Mutex<Option<&'static str>>,
}

impl std::fmt::Debug for FailingSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailingSession").finish()
    }
}

impl FailingSession {
    fn new() -> Self {
        FailingSession {
            inner: InMemorySession::new("failing"),
            fail_on_call_id: None,
            fail_on_tool_output: false,
            fail_once: Mutex::new(None),
        }
    }
}

#[async_trait::async_trait]
impl Session for FailingSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if self.fail_on_tool_output
            && items
                .iter()
                .any(|i| i.as_value()["type"] == "function_call_output")
        {
            return Err(AgentsError::other("synthetic session failure"));
        }
        if let Some(call_id) = &self.fail_on_call_id {
            let matches = items
                .iter()
                .any(|i| i.as_value()["call_id"].as_str() == Some(call_id));
            let failure = *self.fail_once.lock().unwrap();
            if matches && let Some(failure) = failure {
                *self.fail_once.lock().unwrap() = None;
                if failure == "after_append" {
                    self.inner.add_items(items).await?;
                }
                return Err(AgentsError::other("synthetic append failure"));
            }
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

/// Run to completion; returns the result (or error) and the tool output stream events.
async fn run_collect(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    streaming: bool,
    options: RunOptions<()>,
) -> (Result<()>, Vec<Value>) {
    if streaming {
        let mut s = Runner::run_streamed_with_options(agent.clone(), input, options);
        let mut outputs = Vec::new();
        let events = s.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            match event {
                Ok(StreamEvent::RunItem {
                    name: RunItemStreamEventName::ToolOutput,
                    item,
                }) => outputs.push(item.to_input_item().0),
                Ok(_) => {}
                Err(e) => return (Err(e), outputs),
            }
        }
        (Ok(()), outputs)
    } else {
        (
            Runner::run_with_options(agent.clone(), input, options)
                .await
                .map(|_| ()),
            Vec::new(),
        )
    }
}

fn output_infos<T>(results: &[T], info: impl Fn(&T) -> &Value) -> Vec<Value> {
    results.iter().map(|r| info(r).clone()).collect()
}

// ===========================================================================
// test_tool_batch_failure_history.py
// ===========================================================================

#[tokio::test]
async fn completed_sibling_survives_tool_batch_failure() {
    for failure in ["input", "output", "handler", "hook"] {
        for streaming in [false, true] {
            let finished = Event::new();
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let create_ticket = ok_tool("create_ticket", "ticket T-1", Some(effects.clone()))
                .with_input_guardrail(ToolInputGuardrail::new(
                    "allow_input",
                    |_d: ToolInputGuardrailData<()>| async {
                        Ok(ToolGuardrailFunctionOutput::allow()
                            .with_output_info(json!("input accepted")))
                    },
                ))
                .with_output_guardrail(ToolOutputGuardrail::new(
                    "allow_output",
                    |_d: ToolOutputGuardrailData<()>| async {
                        Ok(ToolGuardrailFunctionOutput::allow()
                            .with_output_info(json!("output accepted")))
                    },
                ));
            let gate = finished.clone();
            let mut send_email = FunctionTool::<()>::new_no_args("send_email", "send", move |_c| {
                let gate = gate.clone();
                async move {
                    gate.wait().await;
                    if failure == "handler" {
                        return Err(AgentsError::other("synthetic tool failure"));
                    }
                    Ok("email sent".to_string())
                }
            })
            .without_failure_error_function();
            if failure == "input" {
                let gate = finished.clone();
                send_email = send_email.with_input_guardrail(ToolInputGuardrail::new(
                    "reject_input",
                    move |_d: ToolInputGuardrailData<()>| {
                        let gate = gate.clone();
                        async move {
                            gate.wait().await;
                            Ok(ToolGuardrailFunctionOutput::raise_exception()
                                .with_output_info(json!("blocked")))
                        }
                    },
                ));
            }
            if failure == "output" {
                send_email = send_email.with_output_guardrail(ToolOutputGuardrail::new(
                    "reject_output",
                    |_d: ToolOutputGuardrailData<()>| async {
                        Ok(ToolGuardrailFunctionOutput::raise_exception()
                            .with_output_info(json!("blocked")))
                    },
                ));
            }
            let model = scripted(vec![
                vec![
                    reasoning("rs_before"),
                    call("send_email", "email"),
                    reasoning("rs_ticket"),
                    call("create_ticket", "ticket"),
                    reasoning("rs_after"),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("support")
                .with_model_instance(model.clone())
                .with_tool(send_email)
                .with_tool(create_ticket)
                .with_model_settings(ModelSettings {
                    tool_choice: Some(ToolChoice::required()),
                    ..Default::default()
                });
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("test"));
            let checkpoint = RunCheckpoint::new();
            let hooks: Arc<dyn RunHooks<()>> = Arc::new(EndHook {
                tool_name: Some("create_ticket".into()),
                event: finished.clone(),
                final_fails: None,
                fail_other_tools: failure == "hook",
            });
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(hooks)
                .checkpoint(&checkpoint);
            let (outcome, output_events) = run_collect(&agent, "go", streaming, options).await;
            let error = outcome.unwrap_err();
            let label = format!("failure={failure} streaming={streaming}");
            match failure {
                "input" => assert!(
                    matches!(
                        error,
                        AgentsError::ToolInputGuardrailTripwireTriggered { .. }
                    ),
                    "{label}: {error:?}"
                ),
                "output" => assert!(
                    matches!(
                        error,
                        AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
                    ),
                    "{label}: {error:?}"
                ),
                _ => assert!(
                    matches!(error, AgentsError::UserError { .. }),
                    "{label}: {error:?}"
                ),
            }
            assert_eq!(*effects.lock().unwrap(), vec!["create_ticket"], "{label}");
            // An end hook runs after the output passed its tool guardrails, so its failure keeps
            // the failing call's output too.
            let expected = if failure == "hook" {
                s(&[
                    "reasoning",
                    "function_call:email",
                    "reasoning",
                    "function_call:ticket",
                    "function_call_output:email",
                    "function_call_output:ticket",
                ])
            } else {
                s(&[
                    "reasoning",
                    "function_call:ticket",
                    "function_call_output:ticket",
                ])
            };
            let run_data = error.run_data().expect("run data");
            assert!(
                output_infos(&run_data.tool_input_guardrail_results, |r| &r
                    .output
                    .output_info)
                .contains(&json!("input accepted")),
                "{label}"
            );
            assert!(
                output_infos(&run_data.tool_output_guardrail_results, |r| &r
                    .output
                    .output_info)
                .contains(&json!("output accepted")),
                "{label}: {:?} / {:?}",
                shape(&run_data_items(run_data)),
                run_data.tool_output_guardrail_results.len()
            );
            assert_eq!(shape(&run_data_items(run_data)), expected, "{label}");
            let history = session_items(session.as_ref()).await;
            let mut with_user = s(&["user"]);
            with_user.extend(expected.clone());
            assert_eq!(shape(&history), with_user, "{label}");
            let reasoning_ids: Vec<&str> = history
                .iter()
                .filter(|i| i["type"] == "reasoning")
                .map(|i| i["id"].as_str().unwrap())
                .collect();
            let expected_reasoning = if failure == "hook" {
                vec!["rs_before", "rs_ticket"]
            } else {
                vec!["rs_ticket"]
            };
            assert_eq!(reasoning_ids, expected_reasoning, "{label}");
            assert_eq!(history.last().unwrap()["output"], json!("ticket T-1"));
            if streaming {
                let expected_outputs: Vec<String> = expected
                    .iter()
                    .filter(|i| i.starts_with("function_call_output:"))
                    .cloned()
                    .collect();
                assert_eq!(shape(&output_events), expected_outputs, "{label}");
                let state = checkpoint.take().expect("a failure checkpoint");
                let state = RunState::from_json(state.to_json().unwrap()).unwrap();
                assert!(
                    output_infos(&state.tool_output_guardrail_results, |r| &r
                        .output
                        .output_info)
                    .contains(&json!("output accepted")),
                    "{label}"
                );
                let replay_model = scripted(vec![vec![assistant_message("resumed")]]);
                let replay_agent = agent.clone().with_model_instance(replay_model.clone());
                Runner::run_with_options(replay_agent, state, RunOptions::default())
                    .await
                    .unwrap();
                let replay = replay_model.last_call().unwrap();
                assert_eq!(replay.request.model_settings.tool_choice, None, "{label}");
                assert_eq!(shape(&values(&replay.request.input)), with_user, "{label}");
                assert_eq!(*effects.lock().unwrap(), vec!["create_ticket"], "{label}");
            }
            Runner::run_with_options(
                agent.clone(),
                "finish",
                RunOptions::default().session(session.clone()),
            )
            .await
            .unwrap();
            let mut continued = with_user.clone();
            continued.push("user".into());
            assert_eq!(shape(&last_input(&model)), continued, "{label}");
            assert_eq!(*effects.lock().unwrap(), vec!["create_ticket"], "{label}");
        }
    }
}

#[tokio::test]
async fn session_save_failure_preserves_primary_tool_error() {
    for streaming in [false, true] {
        let finished = Event::new();
        let gate = finished.clone();
        let blocked_tool = FunctionTool::<()>::new_no_args("blocked_tool", "blocked", |_c| async {
            panic!("blocked tool must not execute");
            #[allow(unreachable_code)]
            Ok("never".to_string())
        })
        .with_input_guardrail(ToolInputGuardrail::new(
            "reject",
            move |_d: ToolInputGuardrailData<()>| {
                let gate = gate.clone();
                async move {
                    gate.wait().await;
                    Ok(ToolGuardrailFunctionOutput::raise_exception()
                        .with_output_info(json!("blocked")))
                }
            },
        ));
        let agent = Agent::<()>::new("support")
            .with_model_instance(scripted(vec![vec![
                call("completed_tool", "done"),
                call("blocked_tool", "blocked"),
            ]]))
            .with_tool(ok_tool("completed_tool", "completed", None))
            .with_tool(blocked_tool);
        let mut failing = FailingSession::new();
        failing.fail_on_tool_output = true;
        let session: Arc<dyn Session> = Arc::new(failing);
        let options = RunOptions::default()
            .session(session.clone())
            .hooks(end_hook(None, finished.clone()));
        let (outcome, _) = run_collect(&agent, "go", streaming, options).await;
        let error = outcome.unwrap_err();
        assert!(
            matches!(
                error,
                AgentsError::ToolInputGuardrailTripwireTriggered { .. }
            ),
            "streaming={streaming}: {error:?}"
        );
        let run_data = error.run_data().expect("run data");
        assert_eq!(
            shape(&run_data_items(run_data)),
            s(&["function_call:done", "function_call_output:done"])
        );
        assert_eq!(shape(&session_items(session.as_ref()).await), s(&["user"]));
    }
}

#[tokio::test]
async fn resumed_failure_keeps_accepted_history_and_reconciles_session() {
    // Python also asserts on `_pending_session_write`, an internal of its session writer; the
    // port checks the observable outcome of that reconciliation (the resumed session history).
    for streaming in [false, true] {
        for write_failure in [None, Some("before_append"), Some("after_append")] {
            let label = format!("streaming={streaming} write_failure={write_failure:?}");
            let finished = Event::new();
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let admissions = Arc::new(Mutex::new(0usize));
            let admitted = admissions.clone();
            let approved_tool = ok_tool("approved_tool", "approved", Some(effects.clone()))
                .with_needs_approval(true);
            let completed_tool = ok_tool("completed_tool", "completed", Some(effects.clone()))
                .with_output_guardrail(ToolOutputGuardrail::new(
                    "accept_output",
                    |_d: ToolOutputGuardrailData<()>| async {
                        Ok(ToolGuardrailFunctionOutput::allow()
                            .with_output_info(json!("accepted side effect")))
                    },
                ));
            let model = scripted(vec![
                vec![call("approved_tool", "approved")],
                vec![
                    call("completed_tool", "done"),
                    call("failed_tool", "failed"),
                ],
                vec![assistant_message("resumed")],
            ]);
            let agent = Agent::<()>::new("support")
                .with_model_instance(model.clone())
                .with_tool(approved_tool)
                .with_tool(completed_tool)
                .with_tool(failing_after(
                    "failed_tool",
                    finished.clone(),
                    "synthetic sibling failure",
                ))
                .with_input_guardrail(InputGuardrail::new("admit", move |_c, _a, _i| {
                    let admitted = admitted.clone();
                    async move {
                        *admitted.lock().unwrap() += 1;
                        Ok(GuardrailFunctionOutput::new(json!("safe"), false))
                    }
                }))
                .with_model_settings(ModelSettings {
                    tool_choice: Some(ToolChoice::required()),
                    ..Default::default()
                });
            let mut failing = FailingSession::new();
            failing.fail_on_call_id = Some("done".into());
            *failing.fail_once.lock().unwrap() = write_failure;
            let session: Arc<dyn Session> = Arc::new(failing);
            let interrupted = Runner::run_with_options(
                agent.clone(),
                "go",
                RunOptions::default().session(session.clone()),
            )
            .await
            .unwrap();
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let checkpoint = RunCheckpoint::new();
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(end_hook(Some("completed_tool"), finished.clone()))
                .checkpoint(&checkpoint);
            let (outcome, _) = run_collect(&agent, state, streaming, options).await;
            let error = outcome.unwrap_err();
            assert!(
                matches!(error, AgentsError::UserError { .. }),
                "{label}: {error:?}"
            );
            assert!(
                error.to_string().contains("synthetic sibling failure"),
                "{label}"
            );
            let expected = s(&[
                "function_call:approved",
                "function_call_output:approved",
                "function_call:done",
                "function_call_output:done",
            ]);
            assert_eq!(*admissions.lock().unwrap(), 1, "{label}");
            let run_data = error.run_data().expect("run data");
            let public: Vec<Value> = run_data
                .new_items
                .iter()
                .filter(|i| {
                    !matches!(
                        i.kind,
                        openai_agents_sdk::items::RunItemKind::ToolApproval { .. }
                    )
                })
                .map(|i| i.to_input_item().0)
                .collect();
            assert_eq!(shape(&public), expected, "{label}");
            let state = checkpoint.take().expect("a failure checkpoint");
            assert_eq!(state.model_responses.len(), 2, "{label}");
            assert_eq!(state.current_turn, 2, "{label}");
            assert_eq!(
                output_infos(&state.tool_output_guardrail_results, |r| &r
                    .output
                    .output_info),
                vec![json!("accepted side effect")],
                "{label}"
            );
            let state = RunState::from_json(state.to_json().unwrap()).unwrap();
            let resumed = Runner::run_with_options(
                agent.clone(),
                state,
                RunOptions::default().session(session.clone()),
            )
            .await
            .unwrap();
            assert_eq!(
                resumed.final_output_text().as_deref(),
                Some("resumed"),
                "{label}"
            );
            let last = model.last_call().unwrap();
            let mut with_user = s(&["user"]);
            with_user.extend(expected.clone());
            assert_eq!(shape(&values(&last.request.input)), with_user, "{label}");
            assert_eq!(last.request.model_settings.tool_choice, None, "{label}");
            let mut saved = with_user.clone();
            saved.push("message".into());
            assert_eq!(
                shape(&session_items(session.as_ref()).await),
                saved,
                "{label}"
            );
            assert_eq!(
                *effects.lock().unwrap(),
                vec!["approved_tool", "completed_tool"],
                "{label}"
            );
            assert_eq!(*admissions.lock().unwrap(), 1, "{label}");
        }
    }
}

#[tokio::test]
async fn fresh_guarded_terminal_failure_keeps_output_private() {
    // `tool_hook` and `agent_hook` failures are Python-only: Rust hooks cannot fail.
    for streaming in [false, true] {
        let completed = Event::new();
        let guardrail_outputs = Arc::new(Mutex::new(Vec::<Value>::new()));
        let seen = guardrail_outputs.clone();
        let agent = Agent::<()>::new("guarded-terminal")
            .with_model_instance(scripted(vec![vec![
                call("guarded", "guarded"),
                call("sibling", "sibling"),
            ]]))
            .with_tool(ok_tool("guarded", "synthetic guarded output", None))
            .with_tool(failing_after(
                "sibling",
                completed.clone(),
                "synthetic failure",
            ))
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("reject", move |_c, _a, output| {
                seen.lock().unwrap().push(output);
                async { Ok(GuardrailFunctionOutput::new(Value::Null, true)) }
            }));
        let session: Arc<dyn Session> = Arc::new(InMemorySession::new("fresh"));
        let options = RunOptions::default()
            .session(session.clone())
            .hooks(end_hook(Some("guarded"), completed.clone()));
        let (outcome, _outputs) = run_collect(&agent, "go", streaming, options).await;
        let error = outcome.unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        assert!(error.to_string().contains("synthetic failure"));
        assert!(guardrail_outputs.lock().unwrap().is_empty());
        let run_data = error.run_data().expect("run data");
        assert!(
            run_data.new_items.is_empty(),
            "streaming={streaming}: {:?}",
            run_data.new_items
        );
        assert_eq!(shape(&session_items(session.as_ref()).await), s(&["user"]));
        // Python also asserts that no `tool_call_output_item` event was streamed. Rust streams
        // each tool output as it completes, before the turn resolves (deferred, see report).
    }
}

// ===========================================================================
// test_tool_failure_boundaries.py
// ===========================================================================

#[tokio::test]
async fn completed_history_at_failure_boundaries() {
    // `mcp_callback` (the hosted MCP approval callback cannot fail) and `tool_cancel` /
    // `native_cancel` (asyncio cancellation) are Python-only.
    for boundary in [
        "provider",
        "provider_final_hook",
        "provider_shell",
        "handoff",
        "tool_behavior",
        "final_hook",
        "parent_cancel",
    ] {
        for streaming in [false, true] {
            let label = format!("boundary={boundary} streaming={streaming}");
            let completed = Event::new();
            let blocked = Event::new();
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let gate = completed.clone();
            let blocked_set = blocked.clone();
            let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
                let gate = gate.clone();
                let blocked = blocked_set.clone();
                async move {
                    if boundary != "provider" && boundary != "provider_shell" {
                        gate.wait().await;
                    }
                    if boundary == "parent_cancel" {
                        blocked.set();
                        std::future::pending::<()>().await;
                    }
                    Err::<String, _>(AgentsError::other("synthetic failure"))
                }
            })
            .without_failure_error_function();
            let handoff_effects = effects.clone();
            let target = Agent::<()>::new("target")
                .with_model_instance(scripted(vec![vec![assistant_message("done")]]));
            let transfer = Handoff::new(Arc::new(target)).with_on_handoff(move |_ctx| {
                let effects = handoff_effects.clone();
                async move {
                    assert_eq!(*effects.lock().unwrap(), vec!["create_ticket"]);
                    Err(AgentsError::user("handoff failure"))
                }
            });
            let transfer_name = transfer.tool_name.clone();
            let mut calls = match boundary {
                "provider" => vec![
                    OutputItem::from_value(json!({
                        "id": "ws_done", "type": "web_search_call", "status": "completed",
                        "action": {"type": "search", "query": "synthetic query"},
                    })),
                    call("fail", "fail"),
                ],
                "provider_shell" => vec![
                    OutputItem::from_value(json!({
                        "id": "sh_done", "type": "shell_call", "call_id": "shell",
                        "status": "completed", "action": {"commands": ["echo synthetic"]},
                    })),
                    OutputItem::from_value(json!({
                        "id": "sh_output", "type": "shell_call_output", "call_id": "shell",
                        "status": "completed",
                        "output": [{"stdout": "synthetic", "stderr": "",
                                    "outcome": {"type": "exit", "exit_code": 0}}],
                    })),
                    call("fail", "fail"),
                ],
                "provider_final_hook" => vec![
                    OutputItem::from_value(json!({
                        "id": "ws_done", "type": "web_search_call", "status": "completed",
                        "action": {"type": "search", "query": "synthetic query"},
                    })),
                    assistant_message("done"),
                ],
                _ => vec![
                    call("create_ticket", "ticket"),
                    call(
                        if boundary == "handoff" {
                            &transfer_name
                        } else {
                            "fail"
                        },
                        "fail",
                    ),
                ],
            };
            if boundary == "tool_behavior" || boundary == "final_hook" {
                calls.truncate(1);
            }
            let model = scripted(vec![calls, vec![assistant_message("done")]]);
            let mut agent = Agent::<()>::new("support")
                .with_model_instance(model.clone())
                .with_tool(ok_tool(
                    "create_ticket",
                    "ticket T-1",
                    Some(effects.clone()),
                ))
                .with_tool(fail)
                .with_tool(
                    openai_agents_sdk::tool::ShellTool::<()>::hosted(json!({
                        "type": "container_reference", "container_id": "cntr_synthetic",
                    }))
                    .unwrap(),
                )
                .with_handoff(transfer);
            if boundary == "tool_behavior" {
                agent = agent.with_tool_use_behavior(ToolUseBehavior::custom(
                    |_ctx, results| async move {
                        if !results.is_empty() {
                            return Err(AgentsError::user("tool behavior failure"));
                        }
                        Ok(ToolsToFinalOutputResult::not_final())
                    },
                ));
            }
            if boundary == "final_hook" {
                agent = agent.with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
            }
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("failure-boundaries"));
            let checkpoint = RunCheckpoint::new();
            let hooks: Arc<dyn RunHooks<()>> = Arc::new(EndHook {
                tool_name: Some("create_ticket".into()),
                event: completed.clone(),
                final_fails: match boundary {
                    "provider_final_hook" => Some(None),
                    "final_hook" => Some(Some(effects.clone())),
                    _ => None,
                },
                fail_other_tools: false,
            });
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(hooks)
                .checkpoint(&checkpoint);
            let expected = match boundary {
                "parent_cancel" => Vec::new(),
                "provider" | "provider_final_hook" => s(&["web_search_call"]),
                "provider_shell" => s(&["shell_call:shell", "shell_call_output:shell"]),
                _ => s(&["function_call:ticket", "function_call_output:ticket"]),
            };
            if boundary == "parent_cancel" {
                let run_agent = agent.clone();
                let task =
                    tokio::spawn(
                        async move { run_collect(&run_agent, "go", streaming, options).await },
                    );
                tokio::time::timeout(std::time::Duration::from_secs(2), blocked.wait())
                    .await
                    .expect("the failing tool started");
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled(), "{label}");
            } else {
                let (outcome, outputs) = run_collect(&agent, "go", streaming, options).await;
                let error = outcome.unwrap_err();
                assert!(
                    matches!(error, AgentsError::UserError { .. }),
                    "{label}: {error:?}"
                );
                let run_data = error.run_data().expect("run data");
                assert_eq!(shape(&run_data_items(run_data)), expected, "{label}");
                if streaming {
                    let output_shapes: Vec<String> = expected
                        .iter()
                        .filter(|i| i.contains("output:"))
                        .cloned()
                        .collect();
                    assert_eq!(shape(&outputs), output_shapes, "{label}");
                    let state = checkpoint.take().expect("a failure checkpoint");
                    let restored = RunState::from_json(state.to_json().unwrap())
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    let restored_items: Vec<Value> = restored
                        .generated_items
                        .iter()
                        .map(|i| i.to_input_item().0)
                        .collect();
                    assert_eq!(shape(&restored_items), expected, "{label}");
                }
            }
            let history = session_items(session.as_ref()).await;
            if boundary == "parent_cancel" {
                // Python saves the input before the turn starts; Rust writes it with the first
                // settled turn, so an aborted first turn leaves the session empty (see report).
                assert!(
                    history.iter().all(|i| i["role"] == "user"),
                    "{label}: {history:?}"
                );
            } else {
                assert_eq!(shape(&history[1..]), expected, "{label}");
            }
            if boundary != "parent_cancel" {
                Runner::run_with_options(
                    agent.clone(),
                    "continue",
                    RunOptions::default().session(session.clone()),
                )
                .await
                .unwrap();
                let input = last_input(&model);
                assert_eq!(shape(&input[1..input.len() - 1]), expected, "{label}");
            }
            let expected_effects: Vec<&str> = if boundary.starts_with("provider") {
                vec![]
            } else {
                vec!["create_ticket"]
            };
            assert_eq!(*effects.lock().unwrap(), expected_effects, "{label}");
        }
    }
}

#[tokio::test]
async fn completed_provider_tool_search_survives_local_failure() {
    for scenario in ["completed", "anonymous", "incomplete"] {
        for streaming in [false, true] {
            let label = format!("scenario={scenario} streaming={streaming}");
            let call_id = if scenario == "anonymous" {
                Value::Null
            } else {
                json!("search")
            };
            let search_call = json!({
                "id": "search_call", "call_id": call_id, "type": "tool_search_call",
                "execution": "server", "status": "completed",
                "arguments": {"query": "synthetic lookup"},
            });
            let search_output = json!({
                "id": "search_output", "call_id": call_id, "type": "tool_search_output",
                "execution": "server",
                "status": if scenario == "incomplete" { "incomplete" } else { "completed" },
                "tools": [{"type": "function", "name": "lookup",
                           "parameters": {"type": "object", "properties": {}}}],
                "created_by": "synthetic_provider",
            });
            let mut output_without_created_by = search_output.clone();
            output_without_created_by
                .as_object_mut()
                .unwrap()
                .remove("created_by");
            let expected = if scenario == "incomplete" {
                Vec::new()
            } else {
                vec![search_call.clone(), output_without_created_by]
            };
            let fail = FunctionTool::<()>::new_no_args("fail", "fails", |_c| async {
                Err::<String, _>(AgentsError::other("synthetic sibling failure"))
            })
            .without_failure_error_function();
            let model = scripted(vec![
                vec![
                    OutputItem::from_value(search_call.clone()),
                    OutputItem::from_value(search_output.clone()),
                    call("fail", "fail"),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("search")
                .with_model_instance(model.clone())
                .with_tool(fail);
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("search-failure"));
            let checkpoint = RunCheckpoint::new();
            let mut events = Vec::new();
            let options = RunOptions::default()
                .session(session.clone())
                .checkpoint(&checkpoint);
            let error = if streaming {
                let mut stream = Runner::run_streamed_with_options(agent.clone(), "go", options);
                let mut error = None;
                {
                    let stream_events = stream.stream_events();
                    futures::pin_mut!(stream_events);
                    while let Some(event) = stream_events.next().await {
                        match event {
                            Ok(StreamEvent::RunItem { name, .. })
                                if matches!(
                                    name,
                                    RunItemStreamEventName::ToolSearchCalled
                                        | RunItemStreamEventName::ToolSearchOutputCreated
                                ) =>
                            {
                                events.push(name.as_str());
                            }
                            Ok(_) => {}
                            Err(e) => error = Some(e),
                        }
                    }
                }
                error.expect("the run failed")
            } else {
                Runner::run_with_options(agent.clone(), "go", options)
                    .await
                    .unwrap_err()
            };
            assert!(
                error.to_string().contains("synthetic sibling failure"),
                "{label}"
            );
            assert_eq!(
                run_data_items(error.run_data().expect("run data")),
                expected,
                "{label}"
            );
            assert_eq!(
                session_items(session.as_ref()).await[1..].to_vec(),
                expected,
                "{label}"
            );
            if streaming {
                assert_eq!(
                    events,
                    vec!["tool_search_called", "tool_search_output_created"],
                    "{label}"
                );
                let state = checkpoint.take().expect("a failure checkpoint");
                let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
                let restored_items: Vec<Value> = restored
                    .generated_items
                    .iter()
                    .map(|i| i.to_input_item().0)
                    .collect();
                assert_eq!(restored_items, expected, "{label}");
                Runner::run_with_options(agent.clone(), restored, RunOptions::default())
                    .await
                    .unwrap();
                assert_eq!(last_input(&model)[1..].to_vec(), expected, "{label}");
            } else {
                Runner::run_with_options(
                    agent.clone(),
                    "continue",
                    RunOptions::default().session(session.clone()),
                )
                .await
                .unwrap();
                let input = last_input(&model);
                assert_eq!(input[1..input.len() - 1].to_vec(), expected, "{label}");
            }
        }
    }
}

#[tokio::test]
async fn completed_program_child_retains_parent_after_sibling_failure() {
    for completed_on_followup in [false, true] {
        for streaming in [false, true] {
            let label = format!("followup={completed_on_followup} streaming={streaming}");
            let completed = Event::new();
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let lookup = ok_tool("lookup", "found", Some(effects.clone()))
                .with_allowed_callers(vec!["programmatic".into()]);
            let reasoning = json!({"id": "program_reasoning", "type": "reasoning", "summary": []});
            let program = json!({
                "id": "program_item", "call_id": "program", "code": "lookup()",
                "fingerprint": "synthetic", "type": "program",
            });
            let caller = json!({"type": "program", "caller_id": "program"});
            let child = json!({
                "id": "child", "call_id": "lookup", "name": "lookup", "arguments": "{}",
                "caller": caller, "type": "function_call",
            });
            let program_output = json!({
                "id": "program_output", "call_id": "program", "result": "found",
                "status": "completed", "type": "program_output",
            });
            let item = |v: &Value| OutputItem::from_value(v.clone());
            let first = vec![item(&reasoning), item(&program), item(&child)];
            let fail_call = call("fail", "fail");
            let steps = if completed_on_followup {
                vec![first.clone(), vec![item(&program_output), fail_call]]
            } else {
                let mut failure_step = first.clone();
                failure_step.push(fail_call);
                vec![failure_step]
            };
            let continuation = if completed_on_followup {
                vec![assistant_message("done")]
            } else {
                vec![item(&program_output), assistant_message("done")]
            };
            let mut all = steps;
            all.push(continuation);
            let model = scripted(all);
            let agent = Agent::<()>::new("program")
                .with_model_instance(model.clone())
                .with_tool(Tool::ProgrammaticToolCalling(
                    openai_agents_sdk::tool::ProgrammaticToolCallingTool,
                ))
                .with_tool(lookup)
                .with_tool(failing_after(
                    "fail",
                    completed.clone(),
                    "synthetic sibling failure",
                ));
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("program-failure"));
            let checkpoint = RunCheckpoint::new();
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(end_hook(None, completed.clone()))
                .checkpoint(&checkpoint);
            let (outcome, _) = run_collect(&agent, "go", streaming, options).await;
            let error = outcome.unwrap_err();
            assert!(
                error.to_string().contains("synthetic sibling failure"),
                "{label}: {error}"
            );
            let mut expected = vec![
                reasoning.clone(),
                program.clone(),
                child.clone(),
                json!({"type": "function_call_output", "call_id": "lookup", "output": "found",
                       "caller": caller}),
            ];
            if completed_on_followup {
                expected.push(program_output.clone());
            }
            assert_eq!(
                run_data_items(error.run_data().expect("run data")),
                expected,
                "{label}"
            );
            assert_eq!(
                session_items(session.as_ref()).await[1..].to_vec(),
                expected,
                "{label}"
            );
            if streaming {
                let state = checkpoint.take().expect("a failure checkpoint");
                let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
                let continued =
                    Runner::run_with_options(agent.clone(), restored, RunOptions::default())
                        .await
                        .unwrap();
                assert_eq!(last_input(&model)[1..].to_vec(), expected, "{label}");
                assert_eq!(
                    continued.final_output_text().as_deref(),
                    Some("done"),
                    "{label}"
                );
            }
            assert_eq!(*effects.lock().unwrap(), vec!["lookup"], "{label}");
        }
    }
}

// ===========================================================================
// test_native_tool_failure_history.py
// ===========================================================================

#[tokio::test]
async fn approved_native_output_survives_custom_data_failure() {
    // Python's extractor raises; a Rust extractor returns `Option`, so the failure is a
    // non-object custom data value, which the SDK rejects after the output is committed.
    for streaming in [false, true] {
        let effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let recorded = effects.clone();
        let native = openai_agents_sdk::tool::CustomTool::<()>::new(
            "native",
            "Synthetic native tool",
            move |_ctx, _input| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push("custom".into());
                    Ok("completed".to_string())
                }
            },
        )
        .with_custom_data_extractor(|_ctx| async { Some(json!("not an object")) })
        .with_needs_approval(true);
        let model = scripted(vec![
            vec![OutputItem::from_value(json!({
                "type": "custom_tool_call", "name": "native", "call_id": "native",
                "input": "synthetic",
            }))],
            vec![assistant_message("done")],
        ]);
        let agent = Agent::<()>::new("approved-native")
            .with_model_instance(model)
            .with_tool(native);
        let session: Arc<dyn Session> = Arc::new(InMemorySession::new("approved-native"));
        let interrupted = Runner::run_with_options(
            agent.clone(),
            "go",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        let mut state = interrupted.to_state();
        state.approve(&interrupted.interruptions[0], false);
        let checkpoint = RunCheckpoint::new();
        let options = RunOptions::default()
            .session(session.clone())
            .checkpoint(&checkpoint);
        let (outcome, _) = run_collect(&agent, state, streaming, options).await;
        outcome.unwrap_err();
        assert_eq!(
            *effects.lock().unwrap(),
            vec!["custom"],
            "streaming={streaming}"
        );
        let state = checkpoint.take().expect("a failure checkpoint");
        let outputs: Vec<&openai_agents_sdk::items::RunItem> = state
            .generated_items
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    openai_agents_sdk::items::RunItemKind::ToolCallOutput { .. }
                )
            })
            .collect();
        assert_eq!(outputs.len(), 1, "streaming={streaming}");
        let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
        let resumed = Runner::run_with_options(
            agent.clone(),
            restored,
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            *effects.lock().unwrap(),
            vec!["custom"],
            "streaming={streaming}"
        );
        let saved = session_items(session.as_ref()).await;
        let saved_outputs = saved
            .iter()
            .filter(|i| i["type"] == "custom_tool_call_output")
            .count();
        assert_eq!(saved_outputs, 1, "streaming={streaming}");
    }
}
