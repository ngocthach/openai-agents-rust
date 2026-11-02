//! Ports of the Python cases where an `is_enabled` / `needs_approval` callable or a hosted MCP
//! `on_approval_request` callback raises. Rust expresses those callbacks through the additive
//! fallible builders (`with_is_enabled_fallible`, `with_needs_approval_fallible`,
//! `with_on_approval_request_fallible`, `AgentAsToolOptions::{is_enabled,needs_approval}_fallible`).
//!
//! Python's `asyncio.Event` maps to [`Event`]; a cancelled Python task maps to a dropped Rust
//! future, observed through [`DropFlag`].

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::error::RunErrorDetails;
use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{InputItem, OutputItem, RunInput};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool, MCPToolApprovalResult};
use openai_agents_sdk::{Agent, AgentsError, InMemorySession, Result, RunState, Runner, Session};
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

/// Sets `cancelled` and `finished` when the owning future is dropped before completing:
/// Python's `except CancelledError: cancelled.set()` plus `finally: finished.set()`.
struct DropFlag {
    cancelled: Arc<Event>,
    finished: Arc<Event>,
}

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.cancelled.set();
        self.finished.set();
    }
}

/// A check that signals `started`, then waits forever until it is cancelled.
async fn hang_until_cancelled(started: Arc<Event>, cancelled: Arc<Event>, finished: Arc<Event>) {
    let _guard = DropFlag {
        cancelled,
        finished,
    };
    started.set();
    std::future::pending::<()>().await;
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

fn mcp_request(id: &str, label: &str, name: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "id": id, "type": "mcp_approval_request", "server_label": label,
        "arguments": "{}", "name": name,
    }))
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

fn hosted_mcp(label: &str) -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": label, "server_url": "https://example.com",
        "require_approval": "always",
    }))
}

/// Run to completion; returns the outcome and the tool output stream events.
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

fn assert_other(error: &AgentsError, message: &str) {
    assert!(
        matches!(error, AgentsError::Other(_)) && error.to_string().contains(message),
        "the callback's own error surfaces unchanged: {error:?}"
    );
}

// ===========================================================================
// test_function_tool.py
// ===========================================================================

/// Python: `test_get_all_tools_cancels_sibling_enablement_checks_on_error`.
#[tokio::test]
async fn get_all_tools_cancels_sibling_enablement_checks_on_error() {
    let started = Event::new();
    let cancelled = Event::new();
    let finished = Event::new();
    let (st, ca, fi) = (started.clone(), cancelled.clone(), finished.clone());
    let slow =
        FunctionTool::<()>::new_no_args("slow_tool", "", |_c| async { Ok("slow".to_string()) })
            .with_is_enabled_fallible(move |_ctx, _agent| {
                let (st, ca, fi) = (st.clone(), ca.clone(), fi.clone());
                async move {
                    hang_until_cancelled(st, ca, fi).await;
                    Ok(true)
                }
            });
    let gate = started.clone();
    let failing = FunctionTool::<()>::new_no_args("failing_tool", "", |_c| async {
        Ok("failing".to_string())
    })
    .with_is_enabled_fallible(move |_ctx, _agent| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err(AgentsError::other("enablement failed"))
        }
    });
    let agent = Arc::new(Agent::<()>::new("t").with_tool(slow).with_tool(failing));

    let error = agent.get_all_tools(&RunContext::new(())).await.unwrap_err();
    assert_other(&error, "enablement failed");
    assert!(cancelled.is_set());
    assert!(finished.is_set());
}

// ===========================================================================
// test_handoff_tool.py
// ===========================================================================

/// Python: `test_get_handoffs_cancels_sibling_enablement_checks_on_error`.
#[tokio::test]
async fn get_handoffs_cancels_sibling_enablement_checks_on_error() {
    let started = Event::new();
    let cancelled = Event::new();
    let finished = Event::new();
    let (st, ca, fi) = (started.clone(), cancelled.clone(), finished.clone());
    let gate = started.clone();
    let parent = Agent::<()>::new("parent")
        .with_handoff(
            Handoff::new(Arc::new(Agent::new("slow"))).with_is_enabled_fallible(
                move |_ctx, _agent| {
                    let (st, ca, fi) = (st.clone(), ca.clone(), fi.clone());
                    async move {
                        hang_until_cancelled(st, ca, fi).await;
                        Ok(true)
                    }
                },
            ),
        )
        .with_handoff(
            Handoff::new(Arc::new(Agent::new("failing"))).with_is_enabled_fallible(
                move |_ctx, _agent| {
                    let gate = gate.clone();
                    async move {
                        gate.wait().await;
                        Err(AgentsError::other("enablement failed"))
                    }
                },
            ),
        );

    let error = parent
        .try_get_enabled_handoffs(&RunContext::new(()))
        .await
        .unwrap_err();
    assert_other(&error, "enablement failed");
    assert!(cancelled.is_set());
    assert!(finished.is_set());
}

// ===========================================================================
// test_run_step_execution.py
// ===========================================================================

/// Python: `test_function_tool_enablement_error_cancels_sibling_checks_before_execution`.
/// The model calls only `slow_tool`; the pre-execution re-check covers every configured
/// function tool, so `failing_tool`'s check fails the batch and cancels `slow_tool`'s check.
/// Python drives the private `execute_tools_and_side_effects`; Rust runs the same turn through
/// `Runner`, streamed and not.
#[tokio::test]
async fn function_tool_enablement_error_cancels_sibling_checks_before_execution() {
    for streaming in [false, true] {
        let started = Event::new();
        let cancelled = Event::new();
        let finished = Event::new();
        let slow_checks = Arc::new(AtomicUsize::new(0));
        let failing_checks = Arc::new(AtomicUsize::new(0));
        let invocations = Arc::new(AtomicUsize::new(0));

        let (st, ca, fi, n) = (
            started.clone(),
            cancelled.clone(),
            finished.clone(),
            slow_checks.clone(),
        );
        let inv = invocations.clone();
        let slow = FunctionTool::<()>::new_no_args("slow_tool", "", move |_c| {
            let inv = inv.clone();
            async move {
                inv.fetch_add(1, Ordering::SeqCst);
                Ok("slow".to_string())
            }
        })
        .with_is_enabled_fallible(move |_ctx, _agent| {
            let (st, ca, fi) = (st.clone(), ca.clone(), fi.clone());
            let first = n.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if !first {
                    hang_until_cancelled(st, ca, fi).await;
                }
                Ok(true)
            }
        });
        let (gate, n) = (started.clone(), failing_checks.clone());
        let inv = invocations.clone();
        let failing = FunctionTool::<()>::new_no_args("failing_tool", "", move |_c| {
            let inv = inv.clone();
            async move {
                inv.fetch_add(1, Ordering::SeqCst);
                Ok("failing".to_string())
            }
        })
        .with_is_enabled_fallible(move |_ctx, _agent| {
            let gate = gate.clone();
            let first = n.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    return Ok(true);
                }
                gate.wait().await;
                Err(AgentsError::other("enablement failed"))
            }
        });
        let agent = Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![call("slow_tool", "call-1")]]))
            .with_tool(slow)
            .with_tool(failing);

        let (outcome, outputs) = run_collect(&agent, "go", streaming, RunOptions::default()).await;
        let error = outcome.unwrap_err();
        assert_other(&error, "enablement failed");
        assert!(cancelled.is_set(), "streaming={streaming}");
        assert!(finished.is_set(), "streaming={streaming}");
        assert_eq!(invocations.load(Ordering::SeqCst), 0, "no tool ran");
        assert!(outputs.is_empty(), "streaming={streaming}: {outputs:?}");
    }
}

// ===========================================================================
// test_tool_failure_boundaries.py
// ===========================================================================

/// Python: `test_completed_history_at_failure_boundaries[mcp_callback]` (streamed and not).
/// The hosted MCP approval callback fails after the sibling function tool completed: its
/// output survives in the session, the error's run data, the stream and a restored state.
#[tokio::test]
async fn completed_history_at_mcp_callback_failure_boundary() {
    for streaming in [false, true] {
        let label = format!("boundary=mcp_callback streaming={streaming}");
        let effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let tool_effects = effects.clone();
        let create_ticket = FunctionTool::<()>::new_no_args("create_ticket", "", move |_c| {
            let effects = tool_effects.clone();
            async move {
                effects.lock().unwrap().push("ticket".into());
                Ok("ticket T-1".to_string())
            }
        });
        let callback_effects = effects.clone();
        let mcp = hosted_mcp("synthetic").with_on_approval_request_fallible(move |_req| {
            let effects = callback_effects.clone();
            async move {
                assert_eq!(*effects.lock().unwrap(), vec!["ticket"]);
                Err::<MCPToolApprovalResult, _>(AgentsError::user("approval callback failure"))
            }
        });
        let model = scripted(vec![
            vec![
                call("create_ticket", "ticket"),
                mcp_request("approval", "synthetic", "synthetic"),
            ],
            vec![assistant_message("done")],
        ]);
        let agent = Agent::<()>::new("support")
            .with_model_instance(model.clone())
            .with_tool(create_ticket)
            .with_tool(mcp);
        let session: Arc<dyn Session> = Arc::new(InMemorySession::new("failure-boundaries"));
        let checkpoint = RunCheckpoint::new();
        let options = RunOptions::default()
            .session(session.clone())
            .checkpoint(&checkpoint);
        let expected = s(&["function_call:ticket", "function_call_output:ticket"]);

        let (outcome, outputs) = run_collect(&agent, "go", streaming, options).await;
        let error = outcome.unwrap_err();
        assert!(
            matches!(error, AgentsError::UserError { .. })
                && error.to_string().contains("approval callback failure"),
            "{label}: {error:?}"
        );
        let run_data = error.run_data().expect("run data");
        assert_eq!(shape(&run_data_items(run_data)), expected, "{label}");
        if streaming {
            assert_eq!(
                shape(&outputs),
                s(&["function_call_output:ticket"]),
                "{label}"
            );
            let state = checkpoint.take().expect("a failure checkpoint");
            let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
            let restored_items: Vec<Value> = restored
                .generated_items
                .iter()
                .map(|i| i.to_input_item().0)
                .collect();
            assert_eq!(shape(&restored_items), expected, "{label}");
        }
        let history = values(&session.get_items(None).await.unwrap());
        assert_eq!(shape(&history[1..]), expected, "{label}");

        Runner::run_with_options(
            agent.clone(),
            "continue",
            RunOptions::default().session(session.clone()),
        )
        .await
        .unwrap();
        let input = values(&model.last_call().expect("a model call").request.input);
        assert_eq!(shape(&input[1..input.len() - 1]), expected, "{label}");
        assert_eq!(*effects.lock().unwrap(), vec!["ticket"], "{label}");
    }
}

// ===========================================================================
// test_hitl_error_scenarios.py
// ===========================================================================

/// Python: `test_mcp_callback_failure_is_not_retried_for_same_request`. Python calls the
/// private `execute_mcp_approval_requests` twice with one context; Rust runs the same request
/// twice through `Runner` on one shared run context (streamed and not). The run-level
/// response validation refuses the retried request first, with its own wording ("A tool call
/// already executed, but its output was not committed"); the error type and the single
/// callback call match Python.
#[tokio::test]
async fn mcp_callback_failure_is_not_retried_for_same_request() {
    for streaming in [false, true] {
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let recorded = calls.clone();
        let mcp = hosted_mcp("server").with_on_approval_request_fallible(move |req| {
            let recorded = recorded.clone();
            async move {
                recorded
                    .lock()
                    .unwrap()
                    .push(req.data["id"].as_str().unwrap_or_default().to_string());
                Err::<MCPToolApprovalResult, _>(AgentsError::other("callback failed"))
            }
        });
        let request = || mcp_request("mcp-callback-failure", "server", "hosted_mcp");
        let agent = Agent::<()>::new("agent")
            .with_model_instance(scripted(vec![vec![request()], vec![request()]]))
            .with_tool(mcp);
        let ctx = RunContext::new(());

        let (first, _) = run_collect(
            &agent,
            "handle mcp",
            streaming,
            RunOptions::with_run_context(ctx.clone()),
        )
        .await;
        assert_other(&first.unwrap_err(), "callback failed");

        let (second, _) = run_collect(
            &agent,
            "handle mcp",
            streaming,
            RunOptions::with_run_context(ctx.clone()),
        )
        .await;
        let error = second.unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. })
                && error.to_string().contains("already")
                && error.to_string().contains("not committed"),
            "streaming={streaming}: {error:?}"
        );
        assert_eq!(*calls.lock().unwrap(), ["mcp-callback-failure"]);
    }
}

// ===========================================================================
// test_decisions_examples.py
// ===========================================================================

/// Python: `test_tool_classifier_failure_never_invokes_tool[approval=True]`. The example's
/// approval policy calls a classifier service; the service failure surfaces from the run and
/// the tool body never runs. Rust ports the SDK behaviour (the example module itself is
/// Python-only), streamed and not.
#[tokio::test]
async fn approval_classifier_failure_never_invokes_tool() {
    for streaming in [false, true] {
        let invocations = Arc::new(AtomicUsize::new(0));
        let inv = invocations.clone();
        let publish = FunctionTool::<()>::new_raw(
            "publish_announcement",
            "Simulate publishing an announcement.",
            json!({"type": "object", "properties": {"text": {"type": "string"}},
                   "required": ["text"], "additionalProperties": false}),
            move |_ctx, args: String| {
                let inv = inv.clone();
                async move {
                    inv.fetch_add(1, Ordering::SeqCst);
                    Ok(format!("Simulated publication: {args}"))
                }
            },
        )
        .with_needs_approval_fallible(|_ctx, _args, _call_id| async {
            Err(AgentsError::other("synthetic service failure"))
        });
        let agent = Agent::<()>::new("Announcement assistant")
            .with_model_instance(scripted(vec![vec![function_call_full(
                "publish_announcement",
                r#"{"text":"Release Friday"}"#,
                "call-1",
                None,
                None,
            )]]))
            .with_tool(publish);

        let (outcome, outputs) = run_collect(
            &agent,
            "Publish the announcement",
            streaming,
            RunOptions::default(),
        )
        .await;
        assert_other(&outcome.unwrap_err(), "synthetic service failure");
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            0,
            "streaming={streaming}"
        );
        assert!(outputs.is_empty(), "streaming={streaming}");
    }
}

// ===========================================================================
// Rust-only coverage of the remaining fallible forms
// ===========================================================================

/// A failing handoff `is_enabled` check aborts the run before the model is called.
#[tokio::test]
async fn failing_handoff_enablement_aborts_the_run() {
    for streaming in [false, true] {
        let model = scripted(vec![vec![assistant_message("done")]]);
        let agent = Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(
                Handoff::new(Arc::new(Agent::new("target"))).with_is_enabled_fallible(
                    |_ctx, _agent| async { Err(AgentsError::other("handoff check failed")) },
                ),
            );
        let (outcome, _) = run_collect(&agent, "go", streaming, RunOptions::default()).await;
        assert_other(&outcome.unwrap_err(), "handoff check failed");
        assert!(model.calls().is_empty(), "streaming={streaming}");
    }
}

/// The agent-as-tool fallible forms behave like the function tool ones.
#[tokio::test]
async fn agent_as_tool_fallible_callbacks() {
    let nested = Agent::<()>::new("nested")
        .with_model_instance(scripted(vec![vec![assistant_message("nested done")]]));
    let failing_enabled = nested.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("nested_disabled")
            .is_enabled_fallible(|_ctx, _agent| async {
                Err(AgentsError::other("agent tool check failed"))
            }),
    );
    let agent = Arc::new(Agent::<()>::new("outer").with_tool(failing_enabled));
    let error = agent.get_all_tools(&RunContext::new(())).await.unwrap_err();
    assert_other(&error, "agent tool check failed");

    for streaming in [false, true] {
        let nested_model = scripted(vec![vec![assistant_message("nested done")]]);
        let nested = Agent::<()>::new("nested").with_model_instance(nested_model.clone());
        let gated = nested.as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("nested_tool")
                .needs_approval_fallible(|_ctx, _args, _call_id| async {
                    Err(AgentsError::other("agent tool approval failed"))
                }),
        );
        let agent = Agent::<()>::new("outer")
            .with_model_instance(scripted(vec![vec![function_call_full(
                "nested_tool",
                r#"{"input":"hi"}"#,
                "call-1",
                None,
                None,
            )]]))
            .with_tool(gated);
        let (outcome, _) = run_collect(&agent, "go", streaming, RunOptions::default()).await;
        assert_other(&outcome.unwrap_err(), "agent tool approval failed");
        assert!(
            nested_model.calls().is_empty(),
            "the nested agent never ran"
        );
    }
}
