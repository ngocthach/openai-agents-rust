//! Ports of the hook-failure cases of the Python SDK's `tests/test_tool_batch_failure_history.py`:
//! a lifecycle hook fails a run through its fallible `try_` twin, and the accepted side effects
//! of the turn must survive the failure.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::{InputItem, OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, OutputGuardrail, Result, RunHooks, RunState, Runner,
    Session,
};
use serde_json::Value;
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
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            futures::pin_mut!(notified);
            notified.as_mut().enable();
            if self.flag.load(Ordering::SeqCst) {
                return;
            }
            notified.await;
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

fn call(name: &str, id: &str) -> OutputItem {
    function_call_full(name, "{}", id, None, None)
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

fn plain_tool(
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

async fn session_items(session: &dyn Session) -> Vec<Value> {
    session
        .get_items(None)
        .await
        .unwrap()
        .iter()
        .map(|i: &InputItem| i.as_value().clone())
        .collect()
}

async fn run_collect(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    streaming: bool,
    options: RunOptions<()>,
) -> Result<()> {
    if streaming {
        let mut s = Runner::run_streamed_with_options(agent.clone(), input, options);
        let events = s.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                return Err(e);
            }
            let _ = matches!(
                event,
                Ok(StreamEvent::RunItem {
                    name: RunItemStreamEventName::ToolOutput,
                    ..
                })
            );
        }
        Ok(())
    } else {
        Runner::run_with_options(agent.clone(), input, options)
            .await
            .map(|_| ())
    }
}

/// Fails `try_on_tool_end` for the named tool and `try_on_agent_end` on request.
struct FailingHooks {
    /// Tool whose completion releases the sibling gate.
    gate_tool: &'static str,
    gate: Arc<Event>,
    /// The tool whose end hook fails (and the counter of those failures).
    failing_tool: Option<&'static str>,
    failures: Arc<AtomicUsize>,
    fail_agent_end: bool,
}

#[async_trait::async_trait]
impl RunHooks<()> for FailingHooks {
    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) -> Result<()> {
        if tool.name() == self.gate_tool {
            self.gate.set();
        }
        if self.failing_tool.is_some_and(|n| n == tool.name()) {
            self.failures.fetch_add(1, Ordering::SeqCst);
            return Err(AgentsError::user("synthetic failure"));
        }
        Ok(())
    }

    async fn try_on_agent_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _output: &Value,
    ) -> Result<()> {
        if self.fail_agent_end {
            return Err(AgentsError::user("synthetic failure"));
        }
        Ok(())
    }
}

/// `test_fresh_guarded_terminal_failure_keeps_output_private[tool_hook, agent_hook]`: a failing
/// end hook of a guarded terminal tool must not publish the output the output guardrail never saw.
#[tokio::test]
async fn fresh_guarded_terminal_failure_from_a_hook_keeps_output_private() {
    for failure in ["tool_hook", "agent_hook"] {
        for streaming in [false, true] {
            let label = format!("failure={failure} streaming={streaming}");
            let completed = Event::new();
            let guardrail_outputs = Arc::new(Mutex::new(Vec::<Value>::new()));
            let seen = guardrail_outputs.clone();
            let agent = Agent::<()>::new("guarded-terminal")
                .with_model_instance(scripted(vec![vec![call("guarded", "guarded")]]))
                .with_tool(plain_tool("guarded", "synthetic guarded output", None))
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                .with_output_guardrail(OutputGuardrail::new("reject", move |_c, _a, output| {
                    seen.lock().unwrap().push(output);
                    async { Ok(GuardrailFunctionOutput::new(Value::Null, true)) }
                }));
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("fresh"));
            let hooks: Arc<dyn RunHooks<()>> = Arc::new(FailingHooks {
                gate_tool: "guarded",
                gate: completed.clone(),
                failing_tool: (failure == "tool_hook").then_some("guarded"),
                failures: Arc::new(AtomicUsize::new(0)),
                fail_agent_end: failure == "agent_hook",
            });
            let options = RunOptions::default().session(session.clone()).hooks(hooks);
            let error = run_collect(&agent, "go", streaming, options)
                .await
                .unwrap_err();
            assert!(
                matches!(error, AgentsError::UserError { .. }),
                "{label}: {error:?}"
            );
            assert!(error.to_string().contains("synthetic failure"), "{label}");
            assert!(guardrail_outputs.lock().unwrap().is_empty(), "{label}");
            let run_data = error.run_data().expect("run data");
            assert!(
                run_data.new_items.is_empty(),
                "{label}: {:?}",
                run_data.new_items
            );
            assert_eq!(
                shape(&session_items(session.as_ref()).await),
                vec!["user"],
                "{label}"
            );
        }
    }
}

/// `test_approved_failure_diagnostics_preserve_retry_boundary`: approved tools run on resume, the
/// post-output hook of the slower one fails, and the retry from the failed run's state neither
/// repeats the accepted side effect nor reruns the hook of the already accepted output.
#[tokio::test]
async fn approved_failure_diagnostics_preserve_retry_boundary() {
    for streaming in [false, true] {
        for guarded_terminal in [false, true] {
            let label = format!("streaming={streaming} guarded_terminal={guarded_terminal}");
            let completed = Event::new();
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let failures = Arc::new(AtomicUsize::new(0));
            let gate = completed.clone();
            let flaky = FunctionTool::<()>::new_no_args("flaky", "flaky", move |_ctx| {
                let gate = gate.clone();
                async move {
                    gate.wait().await;
                    Ok("done".to_string())
                }
            })
            .with_needs_approval(true);
            let mut agent = Agent::<()>::new("approved")
                .with_model_instance(scripted(vec![
                    vec![call("create_ticket", "ticket"), call("flaky", "flaky")],
                    vec![assistant_message("finished")],
                ]))
                .with_tool(
                    plain_tool("create_ticket", "ticket T-1", Some(effects.clone()))
                        .with_needs_approval(true),
                )
                .with_tool(flaky);
            if guarded_terminal {
                agent = agent
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                    .with_output_guardrail(OutputGuardrail::new("accept", |_c, _a, _o| async {
                        Ok(GuardrailFunctionOutput::new(Value::Null, false))
                    }));
            }
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("approved-diagnostics"));
            let make_hooks = || -> Arc<dyn RunHooks<()>> {
                Arc::new(FailingHooks {
                    gate_tool: "create_ticket",
                    gate: completed.clone(),
                    failing_tool: Some("flaky"),
                    failures: failures.clone(),
                    fail_agent_end: false,
                })
            };
            let parked = Runner::run_with_options(
                agent.clone(),
                "go",
                RunOptions::default().session(session.clone()),
            )
            .await
            .unwrap();
            let mut state = parked.to_state();
            for interruption in &parked.interruptions {
                state.approve(interruption, false);
            }
            let checkpoint = RunCheckpoint::new();
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(make_hooks())
                .checkpoint(&checkpoint);
            let error = run_collect(&agent, state, streaming, options)
                .await
                .unwrap_err();
            assert!(
                matches!(error, AgentsError::UserError { .. }),
                "{label}: {error:?}"
            );
            let run_data = error.run_data().expect("run data");
            let public_outputs = run_data
                .new_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
                .count();
            assert_eq!(
                public_outputs,
                if guarded_terminal { 0 } else { 2 },
                "{label}"
            );
            let state = checkpoint.take().expect("a failure checkpoint");
            let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
            let retried = Runner::run_with_options(
                agent.clone(),
                restored,
                RunOptions::default()
                    .session(session.clone())
                    .hooks(make_hooks()),
            )
            .await
            .unwrap();
            assert_eq!(
                retried.final_output_text().as_deref(),
                Some(if guarded_terminal {
                    "ticket T-1"
                } else {
                    "finished"
                }),
                "{label}"
            );
            assert_eq!(*effects.lock().unwrap(), vec!["create_ticket"], "{label}");
            assert_eq!(failures.load(Ordering::SeqCst), 1, "{label}");
            let saved: Vec<String> = session_items(session.as_ref())
                .await
                .iter()
                .filter(|i| i["type"] == "function_call_output")
                .map(|i| i["call_id"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(saved, vec!["ticket", "flaky"], "{label}");
        }
    }
}
