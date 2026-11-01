//! Port of `test_computer_hook_failure_does_not_repeat_side_effect` and its function-tool
//! counterpart in `tests/test_tool_approval_call_id_reuse.py`: a call whose end hook failed ran
//! its side effect, so when the same call id comes back on a fresh run that shares the context
//! the run is refused instead of repeating it. A call whose turn succeeded stays reusable.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{ComputerTool, FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result, RunContext, Runner};
use serde_json::json;

type Log = Arc<Mutex<Vec<String>>>;

#[derive(Clone)]
struct Recording(Log);

#[async_trait]
impl Computer for Recording {
    async fn screenshot(&self) -> Result<String> {
        self.0.lock().unwrap().push("screenshot".into());
        Ok("aW1n".into())
    }
    async fn click(&self, _: i64, _: i64, _: &str, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn double_click(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn scroll(&self, _: i64, _: i64, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> Result<()> {
        Ok(())
    }
    async fn wait(&self) -> Result<()> {
        Ok(())
    }
    async fn move_mouse(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
}

/// Fails the first tool end hook, then lets every later one through.
#[derive(Default)]
struct FailOnce {
    failed: Mutex<bool>,
}

#[async_trait]
impl RunHooks<()> for FailOnce {
    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
        _result: &ToolOutput,
    ) -> Result<()> {
        let mut failed = self.failed.lock().unwrap();
        if *failed {
            return Ok(());
        }
        *failed = true;
        Err(AgentsError::user("end hook failed"))
    }
}

fn already_executed(error: &AgentsError) -> bool {
    matches!(error, AgentsError::ModelBehaviorError { message, .. } if message.contains("already executed"))
}

#[tokio::test]
async fn a_computer_call_whose_end_hook_failed_is_not_repeated_by_a_fresh_run() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let screenshots: Log = Arc::default();
    let call = || {
        OutputItem::from_value(json!({
            "id": "computer-item", "type": "computer_call", "action": {"type": "screenshot"},
            "call_id": "computer-call", "status": "completed", "pending_safety_checks": [],
        }))
    };
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![call()]);
    model.add_turn(vec![call()]);
    let agent = Agent::<()>::new("computer-agent")
        .with_tool(ComputerTool::new(Recording(screenshots.clone())))
        .with_model_instance(model);
    let context = RunContext::new(());
    let hooks = Arc::new(FailOnce::default());
    let options = || RunOptions::with_run_context(context.clone()).hooks(hooks.clone());

    let first = Runner::run_with_options(&agent, "use computer", options()).await;
    assert!(first.unwrap_err().to_string().contains("end hook failed"));
    let second = Runner::run_with_options(&agent, "use computer", options()).await;
    assert!(already_executed(&second.unwrap_err()));

    assert_eq!(*screenshots.lock().unwrap(), ["screenshot"]);
}

fn record_tool(executed: &Log) -> FunctionTool<()> {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct NoArgs {}
    let executed = executed.clone();
    FunctionTool::new("record", "record", move |_c, _a: NoArgs| {
        let executed = executed.clone();
        async move {
            executed.lock().unwrap().push("ran".into());
            Ok("ok".to_string())
        }
    })
}

#[tokio::test]
async fn a_function_call_whose_end_hook_failed_is_not_repeated_by_a_fresh_run() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let executed: Log = Arc::default();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id("record", "{}", "call_0")]);
    model.add_turn(vec![function_call_with_id("record", "{}", "call_0")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(record_tool(&executed))
        .with_model_instance(model);
    let context = RunContext::new(());
    let hooks = Arc::new(FailOnce::default());
    let options = || RunOptions::with_run_context(context.clone()).hooks(hooks.clone());

    let first = Runner::run_with_options(&agent, "go", options()).await;
    assert!(first.unwrap_err().to_string().contains("end hook failed"));
    let second = Runner::run_with_options(&agent, "go", options()).await;
    assert!(already_executed(&second.unwrap_err()));

    assert_eq!(*executed.lock().unwrap(), ["ran"]);
}

/// A call that finished in a successful run may be issued again by a later run: its output was
/// committed to the history that run built its next turn from.
#[tokio::test]
async fn a_call_id_from_a_successful_run_stays_reusable() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let executed: Log = Arc::default();
    let model = Arc::new(FakeModel::new());
    for _ in 0..2 {
        model.add_turn(vec![function_call_with_id("record", "{}", "call_0")]);
        model.add_turn(vec![assistant_message("done")]);
    }
    let agent = Agent::<()>::new("agent")
        .with_tool(record_tool(&executed))
        .with_model_instance(model);
    let context = RunContext::new(());

    for _ in 0..2 {
        let result =
            Runner::run_with_options(&agent, "go", RunOptions::with_run_context(context.clone()))
                .await
                .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
    }

    assert_eq!(*executed.lock().unwrap(), ["ran", "ran"]);
}
