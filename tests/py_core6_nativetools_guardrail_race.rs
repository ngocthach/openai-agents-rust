//! Ports of the `*_partial_history_waits_for_input_verdict` family and
//! `nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail` of
//! `test_tool_batch_failure_history.py`: a turn's tools run while a parallel input guardrail is
//! still pending; after a tool failure the partial history is admitted only on a passing
//! verdict, and the selected tool error is raised whatever the verdict was.
//!
//! Python's `asyncio.Event` maps to [`Event`]; asyncio task cancellation (`cancel`,
//! `cancel_at_verdict`, `tool_cancel` verdicts) has no Rust analog and is not ported.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, function_call_full};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, InputGuardrail, Result, RunHooks, Runner, Session,
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

/// Sets an event when dropped (a cancelled future drops its locals).
struct DropFlag(Option<Arc<Event>>);

impl DropFlag {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for DropFlag {
    fn drop(&mut self) {
        if let Some(event) = &self.0 {
            event.set();
        }
    }
}

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

fn s(items: &[&str]) -> Vec<String> {
    items.iter().map(|i| i.to_string()).collect()
}

fn call(name: &str, id: &str) -> OutputItem {
    function_call_full(name, "{}", id, None, None)
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

async fn session_shape(session: &dyn Session) -> Vec<String> {
    let items: Vec<InputItem> = session.get_items(None).await.unwrap();
    shape(
        &items
            .iter()
            .map(|i| i.as_value().clone())
            .collect::<Vec<_>>(),
    )
}

/// `on_tool_end` of any tool sets `event`.
struct EndHook(Arc<Event>);

#[async_trait::async_trait]
impl RunHooks<()> for EndHook {
    async fn on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
        _result: &ToolOutput,
    ) {
        self.0.set();
    }
}

async fn drive(
    agent: Agent<()>,
    streaming: bool,
    options: RunOptions<()>,
) -> (Result<()>, Vec<String>) {
    let checkpoint = RunCheckpoint::new();
    let options = options.checkpoint(&checkpoint);
    let outcome = if streaming {
        let mut result = Runner::run_streamed_with_options(agent, "go", options);
        let mut failure = None;
        {
            let events = result.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                if let Err(error) = event {
                    failure = Some(error);
                }
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    } else {
        Runner::run_with_options(agent, "go", options)
            .await
            .map(|_| ())
    };
    let kept: Vec<Value> = match &outcome {
        Err(error) if !streaming => error
            .run_data()
            .map(|d| d.new_items.iter().map(|i| i.to_input_item().0).collect())
            .unwrap_or_default(),
        _ => checkpoint
            .take()
            .map(|state| {
                state
                    .generated_items
                    .iter()
                    .map(|i| i.to_input_item().0)
                    .collect()
            })
            .unwrap_or_default(),
    };
    (outcome, shape(&kept))
}

async fn partial_history_waits_for_input_verdict(streaming: bool, verdict: &'static str) {
    let label = format!("streaming={streaming} verdict={verdict}");
    let finished = Event::new();
    let release_verdict = Event::new();
    let effects = Arc::new(Mutex::new(Vec::<String>::new()));

    let release = release_verdict.clone();
    let guardrail = InputGuardrail::new("delayed_verdict", move |_c, _a, _i| {
        let release = release.clone();
        async move {
            release.wait().await;
            if verdict == "error" {
                return Err(AgentsError::other("synthetic input verdict failure"));
            }
            Ok(GuardrailFunctionOutput::new(
                json!("checked"),
                verdict == "reject",
            ))
        }
    });
    let recorded = effects.clone();
    let completed_tool = FunctionTool::<()>::new_no_args("completed_tool", "done", move |_c| {
        let recorded = recorded.clone();
        async move {
            recorded.lock().unwrap().push("done".into());
            Ok("completed".to_string())
        }
    });
    let gate = finished.clone();
    let failed_tool = FunctionTool::<()>::new_no_args("failed_tool", "fails", move |_c| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err::<String, _>(AgentsError::other("synthetic sibling failure"))
        }
    })
    .without_failure_error_function();
    let agent = Agent::<()>::new("support")
        .with_model_instance(scripted(vec![vec![
            call("completed_tool", "done"),
            call("failed_tool", "failed"),
        ]]))
        .with_tool(completed_tool)
        .with_tool(failed_tool)
        .with_input_guardrail(guardrail);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("delayed"));
    let options = RunOptions::default()
        .session(session.clone())
        .hooks(Arc::new(EndHook(finished.clone())));
    let task = tokio::spawn(drive(agent, streaming, options));

    // The tools ran and the run now waits for the verdict: nothing is admitted yet.
    tokio::time::timeout(Duration::from_secs(5), async {
        while effects.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the completed tool ran");
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(*effects.lock().unwrap(), vec!["done"], "{label}");
    assert!(
        !task.is_finished(),
        "{label}: the run waits for the verdict"
    );
    assert_eq!(
        session_shape(session.as_ref()).await,
        s(&["user"]),
        "{label}"
    );

    release_verdict.set();
    let (outcome, kept) = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the run ends")
        .unwrap();
    let error = outcome.unwrap_err();
    assert!(
        matches!(error, AgentsError::UserError { .. }),
        "{label}: {error:?}"
    );
    assert!(
        error.to_string().contains("synthetic sibling failure"),
        "{label}: {error}"
    );
    let expected: Vec<&str> = if verdict == "pass" {
        vec!["function_call:done", "function_call_output:done"]
    } else {
        Vec::new()
    };
    assert_eq!(kept, s(&expected), "{label}");
    let mut saved = vec!["user"];
    saved.extend(&expected);
    assert_eq!(session_shape(session.as_ref()).await, s(&saved), "{label}");
}

#[tokio::test]
async fn streamed_partial_history_waits_for_input_verdict() {
    for verdict in ["pass", "reject", "error"] {
        partial_history_waits_for_input_verdict(true, verdict).await;
    }
}

#[tokio::test]
async fn nonstreamed_partial_history_waits_for_input_verdict() {
    for verdict in ["pass", "reject", "error"] {
        partial_history_waits_for_input_verdict(false, verdict).await;
    }
}

#[tokio::test]
async fn nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail() {
    let started = Event::new();
    let cancelled = Event::new();
    let release = Event::new();
    let (s1, c1, r1) = (started.clone(), cancelled.clone(), release.clone());
    let guardrail = InputGuardrail::new("slow_check", move |_c, _a, _i| {
        let (started, cancelled, release) = (s1.clone(), c1.clone(), r1.clone());
        async move {
            started.set();
            let mut flag = DropFlag(Some(cancelled));
            release.wait().await;
            flag.disarm();
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }
    });
    let gate = started.clone();
    let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err::<String, _>(AgentsError::other("primary tool failure without output"))
        }
    })
    .without_failure_error_function();
    let agent = Agent::<()>::new("no-partial-history")
        .with_model_instance(scripted(vec![vec![call("fail", "failed")]]))
        .with_tool(fail)
        .with_input_guardrail(guardrail);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("no-partial-history"));
    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        Runner::run_with_options(agent, "go", RunOptions::default().session(session.clone())),
    )
    .await
    .expect("the run ends without waiting for the guardrail");
    let error = outcome.unwrap_err();
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("primary tool failure without output"),
        "{error}"
    );
    assert!(cancelled.is_set(), "the pending guardrail was cancelled");
    assert!(error.run_data().expect("run data").new_items.is_empty());
    assert_eq!(session_shape(session.as_ref()).await, s(&["user"]));
}
