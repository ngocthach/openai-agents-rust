//! Sibling-failure and run-drop cases of `tests/test_run_step_execution.py` that the first port
//! classed as asyncio-only: what happens to the tools that are still running when another tool
//! of the same turn fails, and when the whole run is dropped.
//!
//! Python cancels the running siblings and checks `asyncio.Event`s set in `except CancelledError`.
//! A Rust future that loses the race is dropped, so each test holds a drop guard inside the
//! sibling and asserts it ran, and asserts that no post-invoke work (failure handler, output
//! guardrail, `on_tool_end`) ran for the dropped tool.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, function_call_full};
use openai_agents_sdk::tool::{ComputerTool, FunctionTool, ShellTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolOutputGuardrail};
use openai_agents_sdk::{Agent, AgentsError, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::sync::Notify;

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

fn call(name: &str, call_id: &str) -> OutputItem {
    function_call_full(name, "{}", call_id, None, None)
}

/// Sets `flag` when dropped.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A tool that signals `started`, then never finishes. Its future owns a drop guard.
fn waiting_tool(name: &str, started: Arc<Notify>, dropped: Arc<AtomicBool>) -> FunctionTool<()> {
    FunctionTool::new(name, "", move |_c, _a: Value| {
        let started = started.clone();
        let guard = DropFlag(dropped.clone());
        async move {
            let _guard = guard;
            started.notify_one();
            std::future::pending::<()>().await;
            Ok("unreachable".to_string())
        }
    })
}

/// A tool that waits for `gate`, then fails.
fn failing_after(name: &str, gate: Arc<Notify>) -> FunctionTool<()> {
    FunctionTool::new(name, "", move |_c, _a: Value| {
        let gate = gate.clone();
        async move {
            gate.notified().await;
            Err::<String, _>(AgentsError::other("boom"))
        }
    })
    .without_failure_error_function()
}

#[derive(Default)]
struct EndHooks {
    ended: Mutex<Vec<String>>,
    /// When set, `on_tool_end` for the named tool signals `started`, sleeps, then records `done`.
    slow_for: Option<(String, Arc<Notify>, Duration, Arc<AtomicBool>)>,
}

#[async_trait]
impl RunHooks<()> for EndHooks {
    async fn on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) {
        if let Some((name, started, delay, done)) = &self.slow_for
            && tool.name() == *name
        {
            started.notify_one();
            tokio::time::sleep(*delay).await;
            done.store(true, Ordering::SeqCst);
        }
        self.ended.lock().unwrap().push(tool.name().to_string());
    }
}

fn assert_user_error(err: AgentsError, needle: &str) {
    assert!(
        matches!(err, AgentsError::UserError { .. }) && err.to_string().contains(needle),
        "{err:?}"
    );
}

/// `test_multiple_tool_calls_skip_post_invoke_work_for_cancelled_sibling_teardown`.
#[tokio::test]
async fn a_dropped_sibling_skips_its_post_invoke_work() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let handler_called = Arc::new(AtomicBool::new(false));
    let guardrail_called = Arc::new(AtomicBool::new(false));
    let handler = handler_called.clone();
    let guardrail_flag = guardrail_called.clone();

    let waiting = waiting_tool("waiting_tool", started.clone(), dropped.clone())
        .with_failure_error_function(move |_ctx, error| {
            handler.store(true, Ordering::SeqCst);
            format!("handled:{error}")
        })
        .with_output_guardrail(ToolOutputGuardrail::new("allow", move |_data| {
            let flag = guardrail_flag.clone();
            async move {
                flag.store(true, Ordering::SeqCst);
                Ok(ToolGuardrailFunctionOutput::allow())
            }
        }));
    let hooks = Arc::new(EndHooks::default());
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            call("waiting_tool", "1"),
            call("error_tool", "2"),
        ]]))
        .with_tool(waiting)
        .with_tool(failing_after("error_tool", started));

    let err = Runner::run_with_options(
        &agent,
        "hello",
        RunOptions::new(()).hooks(hooks.clone() as Arc<dyn RunHooks<()>>),
    )
    .await
    .unwrap_err();
    assert_user_error(err, "Error running tool error_tool: boom");
    tokio::task::yield_now().await;

    assert!(
        dropped.load(Ordering::SeqCst),
        "the sibling was not dropped"
    );
    assert!(!handler_called.load(Ordering::SeqCst));
    assert!(!guardrail_called.load(Ordering::SeqCst));
    assert!(hooks.ended.lock().unwrap().is_empty());
}

struct StubComputer;

#[async_trait]
impl Computer for StubComputer {
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        Ok("aW1n".into())
    }
    async fn click(
        &self,
        _: i64,
        _: i64,
        _: &str,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn double_click(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn wait(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn move_mouse(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

fn computer_call_with_pending_check() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "computer_call", "id": "computer-1", "call_id": "computer-1",
        "status": "completed", "action": {"type": "screenshot"},
        "pending_safety_checks": [{"id": "sc-1", "code": "malicious", "message": "nope"}]
    }))
}

/// A computer tool whose safety check waits for `gate`, then rejects.
fn rejecting_computer(gate: Arc<Notify>) -> ComputerTool<()> {
    ComputerTool::<()>::new(StubComputer).with_on_safety_check(move |_data| {
        let gate = gate.clone();
        async move {
            gate.notified().await;
            false
        }
    })
}

/// `test_execute_tool_plan_cancels_sibling_category_on_failure`: a local shell command that is
/// still running when a computer call of the same turn is rejected is dropped.
#[tokio::test]
async fn a_running_shell_command_is_dropped_when_a_computer_call_fails() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let shell_started = started.clone();
    let shell_dropped = dropped.clone();
    let shell = ShellTool::<()>::new(move |_req| {
        let started = shell_started.clone();
        let guard = DropFlag(shell_dropped.clone());
        async move {
            let _guard = guard;
            started.notify_one();
            std::future::pending::<()>().await;
            Ok("unreachable")
        }
    });
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            OutputItem::from_value(json!({
                "type": "shell_call", "id": "shell-1", "call_id": "shell-1",
                "status": "completed",
                "action": {"commands": ["sleep 1000"], "timeout_ms": 1000}
            })),
            computer_call_with_pending_check(),
        ]]))
        .with_tool(shell)
        .with_tool(rejecting_computer(started));

    let err = Runner::run(&agent, "hello").await.unwrap_err();
    assert_user_error(err, "safety check was not acknowledged");
    assert!(
        dropped.load(Ordering::SeqCst),
        "the shell command kept running"
    );
}

/// `test_execute_tool_plan_drains_function_tools_on_sibling_failure`: the same for a function
/// tool that is still running when a computer call of the turn is rejected.
#[tokio::test]
async fn a_running_function_tool_is_dropped_when_a_computer_call_fails() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            call("slow_tool", "fn-1"),
            computer_call_with_pending_check(),
        ]]))
        .with_tool(waiting_tool("slow_tool", started.clone(), dropped.clone()))
        .with_tool(rejecting_computer(started));

    let err = Runner::run(&agent, "hello").await.unwrap_err();
    assert_user_error(err, "safety check was not acknowledged");
    assert!(
        dropped.load(Ordering::SeqCst),
        "the function tool kept running"
    );
}

/// `test_execute_tool_plan_preserves_function_tool_post_invoke_work` and
/// `test_multiple_tool_calls_allow_successful_sibling_on_tool_end_to_finish`: a sibling that
/// already finished and is still inside its `on_tool_end` hook when another tool fails keeps
/// running that hook to completion.
#[tokio::test]
async fn a_finished_siblings_on_tool_end_runs_to_completion_after_a_failure() {
    let hook_started = Arc::new(Notify::new());
    let hook_done = Arc::new(AtomicBool::new(false));
    let hooks = Arc::new(EndHooks {
        ended: Mutex::default(),
        slow_for: Some((
            "quick_tool".into(),
            hook_started.clone(),
            Duration::from_millis(50),
            hook_done.clone(),
        )),
    });
    let quick = FunctionTool::new("quick_tool", "", |_c, _a: Value| async move {
        Ok("ok".to_string())
    });
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            call("quick_tool", "1"),
            call("error_tool", "2"),
        ]]))
        .with_tool(quick)
        .with_tool(failing_after("error_tool", hook_started));

    let err = Runner::run_with_options(
        &agent,
        "hello",
        RunOptions::new(()).hooks(hooks.clone() as Arc<dyn RunHooks<()>>),
    )
    .await
    .unwrap_err();
    assert_user_error(err, "Error running tool error_tool: boom");
    assert!(
        hook_done.load(Ordering::SeqCst),
        "the failure cut the finished sibling's on_tool_end short"
    );
}

/// `test_execute_function_tool_calls_parent_cancellation_skips_post_invoke_work` and
/// `test_multiple_tool_calls_cancel_pending_tasks_when_parent_cancelled`: dropping the run
/// (a caller timeout) drops the running tool and runs none of its post-invoke work.
#[tokio::test]
async fn dropping_the_run_drops_the_running_tool_and_skips_post_invoke_work() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let handler_called = Arc::new(AtomicBool::new(false));
    let handler = handler_called.clone();
    let waiting = waiting_tool("slow_tool", started.clone(), dropped.clone())
        .with_failure_error_function(move |_ctx, error| {
            handler.store(true, Ordering::SeqCst);
            format!("handled:{error}")
        });
    let hooks = Arc::new(EndHooks::default());
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![call("slow_tool", "1")]]))
        .with_tool(waiting);

    let run = Runner::run_with_options(
        &agent,
        "hello",
        RunOptions::new(()).hooks(hooks.clone() as Arc<dyn RunHooks<()>>),
    );
    let timed_out = tokio::time::timeout(Duration::from_millis(150), run).await;
    assert!(timed_out.is_err(), "the run should still be waiting");
    started.notified().await;

    assert!(dropped.load(Ordering::SeqCst), "the tool outlived the run");
    assert!(!handler_called.load(Ordering::SeqCst));
    assert!(hooks.ended.lock().unwrap().is_empty());
}
