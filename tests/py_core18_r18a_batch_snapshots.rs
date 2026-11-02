//! Ports of three cases of the Python SDK's `tests/test_tool_batch_failure_history.py` that the
//! ledger had filed as Python-only:
//!
//! - `test_partial_history_freezes_accepted_output_before_metadata`: a sibling fails while the
//!   completed tool's `custom_data_extractor` is still running. The run fails with the sibling's
//!   error, keeps the accepted output without custom data, saves it, serializes it, and a later
//!   run does not repeat the tool.
//! - `test_approved_mutable_snapshot_recovery_uses_publication_value`: approved tools, one end
//!   hook fails while the other tool's extractor is still running; the retry from the failed
//!   run's state uses the published output and runs nothing again.
//! - `test_approved_live_snapshot_preserves_structured_output_type` (`list` shape): the retry
//!   hands a custom tool-use behavior the structured output, not a flattened string.
//!
//! The Python halves that mutate the output after publication (an extractor that swallows its
//! cancellation and writes into `context.output` later) cannot happen here: the output an
//! extractor sees is borrowed immutably. As in Python, a function tool's extractor that is still
//! running when the sibling failure surfaces keeps running detached and finishes once released
//! (its result is discarded); a native tool's extractor is cancelled (dropped) after the wait.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::{ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::items::{
    InputItem, ModelResponse, OutputItem, RunInput, RunItem, RunItemKind,
};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full, scripted_stream_events,
};
use openai_agents_sdk::tool::{CustomTool, FunctionTool, Tool, ToolOutput, ToolOutputPart};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolOutputGuardrail, ToolOutputGuardrailData,
};
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, Result, RunHooks, RunState, Runner, Session,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

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

/// Sets an event when dropped: an abandoned extractor future drops its locals.
struct DropFlag(Arc<Event>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.set();
    }
}

/// Steps answering both the plain and the streamed path (native call items need their stream
/// events scripted exactly).
fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let steps = steps
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            let mut step = ScriptedStep::output(output.clone());
            let response = ModelResponse {
                output,
                usage: Default::default(),
                response_id: Some(format!("resp-{index}")),
                request_id: None,
                raw_usage: None,
            };
            step.stream_events = Some(scripted_stream_events(&step, response));
            step
        })
        .collect();
    Arc::new(ScriptedModel::with_steps(steps).unwrap())
}

fn call(name: &str, id: &str) -> OutputItem {
    function_call_full(name, "{}", id, None, None)
}

async fn session_items(session: &dyn Session) -> Vec<Value> {
    let items: Vec<InputItem> = session.get_items(None).await.unwrap();
    items.iter().map(|i| i.as_value().clone()).collect()
}

fn outputs(items: &[RunItem]) -> Vec<&RunItem> {
    items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .collect()
}

fn output_value(item: &RunItem) -> Value {
    match &item.kind {
        RunItemKind::ToolCallOutput { output, .. } => output.clone(),
        _ => Value::Null,
    }
}

fn wire_output(item: &RunItem) -> Value {
    item.to_input_item().0["output"].clone()
}

struct Outcome {
    error: AgentsError,
    output_events: Vec<RunItem>,
}

async fn run_failing(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    streaming: bool,
    options: RunOptions<()>,
) -> Outcome {
    let mut output_events = Vec::new();
    let result = if streaming {
        let mut streamed = Runner::run_streamed_with_options(agent.clone(), input, options);
        let mut failure = None;
        {
            let events = streamed.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                match event {
                    Ok(StreamEvent::RunItem {
                        name: RunItemStreamEventName::ToolOutput,
                        item,
                    }) => output_events.push(item),
                    Ok(_) => {}
                    Err(error) => failure = Some(error),
                }
            }
        }
        failure.map_or(Ok(()), Err)
    } else {
        Runner::run_with_options(agent.clone(), input, options)
            .await
            .map(|_| ())
    };
    Outcome {
        error: result.expect_err("the run must fail"),
        output_events,
    }
}

fn assert_user_error(error: &AgentsError, message: &str, label: &str) {
    assert!(
        matches!(error, AgentsError::UserError { .. }),
        "{label}: {error:?}"
    );
    assert!(error.to_string().contains(message), "{label}: {error}");
}

/// The extractor of the completed tool: it never finishes on its own.
struct Extractor {
    extracting: Arc<Event>,
    release: Arc<Event>,
    dropped: Arc<Event>,
}

impl Extractor {
    fn new() -> Arc<Self> {
        Arc::new(Extractor {
            extracting: Event::new(),
            release: Event::new(),
            dropped: Event::new(),
        })
    }

    async fn run(this: Arc<Self>) -> Option<Value> {
        this.extracting.set();
        let _flag = DropFlag(this.dropped.clone());
        this.release.wait().await;
        Some(json!({"ticket": "T-1"}))
    }
}

// ===========================================================================
// test_partial_history_freezes_accepted_output_before_metadata
// ===========================================================================

#[tokio::test]
async fn partial_history_freezes_accepted_output_before_metadata() {
    for kind in ["function", "native"] {
        for approved in [false, true] {
            for streaming in [false, true] {
                freezes_case(kind, approved, streaming).await;
            }
        }
    }
}

async fn freezes_case(kind: &'static str, approved: bool, streaming: bool) {
    let label = format!("kind={kind} approved={approved} streaming={streaming}");
    let extractor = Extractor::new();
    let effects = Arc::new(Mutex::new(Vec::<String>::new()));
    let guardrail_outputs = Arc::new(Mutex::new(Vec::<Value>::new()));

    let recorded = effects.clone();
    let (completed_tool, completed_call, expected_output): (Tool<()>, OutputItem, Value) =
        if kind == "function" {
            let ex = extractor.clone();
            let seen = guardrail_outputs.clone();
            let tool = FunctionTool::<()>::new_no_args("create_ticket", "creates", move |_c| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push("ticket".into());
                    Ok(json!({"ticket": {"status": "created"}}))
                }
            })
            .with_custom_data_extractor(move |_ctx| Extractor::run(ex.clone()))
            .with_needs_approval(approved)
            .with_output_guardrail(ToolOutputGuardrail::new(
                "record_output",
                move |data: ToolOutputGuardrailData<()>| {
                    let seen = seen.clone();
                    async move {
                        let info = match &data.output {
                            ToolOutput::Json(v) => v.clone(),
                            other => Value::String(other.to_model_string()),
                        };
                        seen.lock().unwrap().push(info.clone());
                        Ok(ToolGuardrailFunctionOutput::allow().with_output_info(info))
                    }
                },
            ));
            (
                tool.into(),
                call("create_ticket", "ticket"),
                json!(json!({"ticket": {"status": "created"}}).to_string()),
            )
        } else {
            let ex = extractor.clone();
            let tool = CustomTool::<()>::new("custom", "Synthetic native tool", move |_c, _i| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push("custom".into());
                    Ok("completed".to_string())
                }
            })
            .with_custom_data_extractor(move |_ctx| Extractor::run(ex.clone()))
            .with_needs_approval(approved);
            (
                tool.into(),
                OutputItem::from_value(json!({
                    "type": "custom_tool_call", "name": "custom", "call_id": "native",
                    "input": "synthetic",
                })),
                json!("completed"),
            )
        };
    let gate = extractor.extracting.clone();
    let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err::<String, _>(AgentsError::other("synthetic sibling failure"))
        }
    })
    .without_failure_error_function()
    .with_needs_approval(approved);
    let agent = Agent::<()>::new("metadata")
        .with_model_instance(scripted(vec![
            vec![completed_call, call("fail", "fail")],
            vec![assistant_message("done")],
        ]))
        .with_tool(completed_tool)
        .with_tool(fail);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("pending-metadata"));
    let config = RunConfig::default().with_tool_post_invoke_wait(Duration::from_millis(20));

    let input: RunInput = if approved {
        let parked = Runner::run_with_options(
            agent.clone(),
            "go",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(parked.interruptions.len(), 2, "{label}");
        let mut state = parked.to_state();
        for item in &parked.interruptions {
            state.approve(item, false);
        }
        RunInput::State(Box::new(state))
    } else {
        "go".into()
    };
    let checkpoint = RunCheckpoint::new();
    let options = RunOptions::default()
        .session(session.clone())
        .run_config(config.clone())
        .checkpoint(&checkpoint);
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        run_failing(&agent, input, streaming, options),
    )
    .await
    .unwrap_or_else(|_| panic!("{label}: the run must not wait for the blocked extractor"));
    assert_user_error(&outcome.error, "synthetic sibling failure", &label);
    assert_eq!(effects.lock().unwrap().len(), 1, "{label}");
    if kind == "function" {
        // Python: `assert not settled.is_set()`; the detached extractor is still waiting.
        assert!(
            !extractor.dropped.is_set(),
            "{label}: the extractor keeps running detached"
        );
    } else {
        assert!(
            extractor.dropped.is_set(),
            "{label}: the extractor was abandoned"
        );
    }
    assert!(!extractor.release.is_set());

    let state = checkpoint.take().expect("a failure checkpoint");
    let items = match outcome.error.run_data() {
        Some(data) if !streaming && !approved => data.new_items.clone(),
        _ => state.generated_items.clone(),
    };
    let accepted = outputs(&items);
    assert_eq!(accepted.len(), 1, "{label}: {items:?}");
    assert_eq!(accepted[0].custom_data(), None, "{label}");
    assert_eq!(wire_output(accepted[0]), expected_output, "{label}");
    if kind == "function" {
        assert_eq!(
            output_value(accepted[0]),
            json!({"ticket": {"status": "created"}}),
            "{label}"
        );
        assert_eq!(
            *guardrail_outputs.lock().unwrap(),
            vec![json!({"ticket": {"status": "created"}})],
            "{label}"
        );
        if let Some(data) = outcome.error.run_data() {
            let infos: Vec<Value> = data
                .tool_output_guardrail_results
                .iter()
                .map(|r| r.output.output_info.clone())
                .collect();
            assert_eq!(
                infos,
                vec![json!({"ticket": {"status": "created"}})],
                "{label}"
            );
        }
    }
    if streaming && !approved {
        assert_eq!(outcome.output_events.len(), 1, "{label}");
        assert_eq!(outcome.output_events[0].custom_data(), None, "{label}");
    }

    // The failed run's state is stable and restores the output without custom data.
    let serialized = state.to_json().unwrap();
    let restored = RunState::from_json(serialized.clone()).unwrap();
    assert_eq!(restored.to_json().unwrap(), serialized, "{label}");
    let restored_outputs = outputs(&restored.generated_items);
    assert_eq!(restored_outputs.len(), 1, "{label}");
    assert_eq!(restored_outputs[0].custom_data(), None, "{label}");
    if kind == "function" {
        assert_eq!(
            output_value(restored_outputs[0]),
            json!({"ticket": {"status": "created"}}),
            "{label}"
        );
    }

    if !approved {
        let saved = session_items(session.as_ref()).await;
        assert_eq!(saved.last().unwrap()["output"], expected_output, "{label}");
        let continuation = Runner::run_with_options(
            agent.clone(),
            "continue",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(
            continuation.final_output_text().as_deref(),
            Some("done"),
            "{label}"
        );
        assert_eq!(effects.lock().unwrap().len(), 1, "{label}");
    }

    // Released, the detached extractor settles; its custom data never reaches the output.
    extractor.release.set();
    tokio::time::timeout(Duration::from_secs(2), extractor.dropped.wait())
        .await
        .unwrap_or_else(|_| panic!("{label}: the extractor did not settle"));
    assert_eq!(accepted[0].custom_data(), None, "{label}");
}

// ===========================================================================
// test_approved_mutable_snapshot_recovery_uses_publication_value
// ===========================================================================

/// Fails `try_on_tool_end` of `failing` (only while `armed`).
struct FailingEndHook {
    failing: &'static str,
    message: &'static str,
}

#[async_trait]
impl RunHooks<()> for FailingEndHook {
    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) -> Result<()> {
        if tool.name() == self.failing {
            return Err(AgentsError::user(self.message));
        }
        Ok(())
    }
}

#[tokio::test]
async fn approved_mutable_snapshot_recovery_uses_publication_value() {
    for streaming in [false, true] {
        let label = format!("streaming={streaming}");
        let extractor = Extractor::new();
        let effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let recorded = effects.clone();
        let ex = extractor.clone();
        let create_ticket =
            FunctionTool::<()>::new_no_args("create_ticket", "creates", move |_c| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push("ticket".into());
                    Ok(json!({"ticket": {"status": "created"}}))
                }
            })
            .with_custom_data_extractor(move |_ctx| Extractor::run(ex.clone()))
            .with_needs_approval(true);
        let recorded = effects.clone();
        let gate = extractor.extracting.clone();
        let audit = FunctionTool::<()>::new_no_args("audit", "audits", move |_c| {
            let (recorded, gate) = (recorded.clone(), gate.clone());
            async move {
                gate.wait().await;
                recorded.lock().unwrap().push("audit".into());
                Ok("recorded".to_string())
            }
        })
        .with_needs_approval(true);
        let agent = Agent::<()>::new("approval")
            .with_model_instance(scripted(vec![vec![
                call("create_ticket", "ticket"),
                call("audit", "audit"),
            ]]))
            .with_tool(create_ticket)
            .with_tool(audit)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
        let session: Arc<dyn Session> = Arc::new(InMemorySession::new("mutable-recovery"));
        let parked = Runner::run_with_options(
            agent.clone(),
            "go",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        let mut state = parked.to_state();
        for item in &parked.interruptions {
            state.approve(item, false);
        }
        let checkpoint = RunCheckpoint::new();
        let options = RunOptions::default()
            .session(session.clone())
            .hooks(Arc::new(FailingEndHook {
                failing: "audit",
                message: "synthetic post-output failure",
            }))
            .run_config(RunConfig::default().with_tool_post_invoke_wait(Duration::from_millis(20)))
            .checkpoint(&checkpoint);
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            run_failing(&agent, state, streaming, options),
        )
        .await
        .unwrap_or_else(|_| panic!("{label}: the run must not wait for the blocked extractor"));
        assert_user_error(&outcome.error, "synthetic post-output failure", &label);
        // The function tool's extractor keeps running detached until released.
        assert!(!extractor.dropped.is_set(), "{label}");

        let state = checkpoint.take().expect("a failure checkpoint");
        let serialized = state.to_json().unwrap();
        assert_eq!(state.to_json().unwrap(), serialized, "{label}");
        let restored = RunState::from_json(serialized).unwrap();
        let recovered = Runner::run_with_options(
            agent.clone(),
            restored,
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(
            recovered.final_output_text(),
            Some(json!({"ticket": {"status": "created"}}).to_string()),
            "{label}"
        );
        assert_eq!(*effects.lock().unwrap(), vec!["ticket", "audit"], "{label}");
        extractor.release.set();
        tokio::time::timeout(Duration::from_secs(2), extractor.dropped.wait())
            .await
            .unwrap_or_else(|_| panic!("{label}: the extractor did not settle"));
    }
}

// ===========================================================================
// test_approved_live_snapshot_preserves_structured_output_type (`list` shape)
// ===========================================================================

#[tokio::test]
async fn approved_live_snapshot_preserves_structured_output_type() {
    for streaming in [false, true] {
        let label = format!("streaming={streaming}");
        let effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let recorded = effects.clone();
        let create_ticket =
            FunctionTool::<()>::new_no_args("create_ticket", "creates", move |_c| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push("ticket".into());
                    Ok(vec![ToolOutputPart::Text {
                        text: "ticket created".into(),
                    }])
                }
            })
            .with_needs_approval(true);
        let seen = Arc::new(Mutex::new(Vec::<ToolOutput>::new()));
        let observed = seen.clone();
        let finish = ToolUseBehavior::custom(move |_ctx, results| {
            let observed = observed.clone();
            async move {
                let output = results[0].output.clone();
                observed.lock().unwrap().push(output.clone());
                match output {
                    ToolOutput::Parts(parts) => match parts.first() {
                        Some(ToolOutputPart::Text { text }) => {
                            Ok(ToolsToFinalOutputResult::final_output(text.clone()))
                        }
                        other => Err(AgentsError::user(format!("unexpected part {other:?}"))),
                    },
                    other => Err(AgentsError::user(format!(
                        "the structured output was flattened: {other:?}"
                    ))),
                }
            }
        });
        let agent = Agent::<()>::new("typed-recovery")
            .with_model_instance(scripted(vec![vec![call("create_ticket", "ticket")]]))
            .with_tool(create_ticket)
            .with_tool_use_behavior(finish);
        let session: Arc<dyn Session> = Arc::new(InMemorySession::new("typed-recovery"));
        let parked = Runner::run_with_options(
            agent.clone(),
            "go",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        let mut state = parked.to_state();
        state.approve(&parked.interruptions[0], false);
        let checkpoint = RunCheckpoint::new();
        let options = RunOptions::default()
            .session(session.clone())
            .hooks(Arc::new(FailingEndHook {
                failing: "create_ticket",
                message: "synthetic end-hook failure",
            }))
            .checkpoint(&checkpoint);
        let outcome = run_failing(&agent, state, streaming, options).await;
        assert_user_error(&outcome.error, "synthetic end-hook failure", &label);

        // Retry the failed run's live state, without a JSON round trip.
        let state = checkpoint.take().expect("a failure checkpoint");
        let recovered = Runner::run_with_options(
            agent.clone(),
            state,
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(
            recovered.final_output_text().as_deref(),
            Some("ticket created"),
            "{label}"
        );
        assert_eq!(*effects.lock().unwrap(), vec!["ticket"], "{label}");
        let seen = seen.lock().unwrap();
        assert!(
            seen.iter().all(|o| matches!(o, ToolOutput::Parts(_))),
            "{label}: {seen:?}"
        );
    }
}
