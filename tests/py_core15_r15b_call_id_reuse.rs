//! Ports of the hook-failure and malformed-replacement cases of the Python SDK's
//! `tests/test_tool_approval_call_id_reuse.py`: a lifecycle hook fails a run through its fallible
//! `try_` twin; the failed run leaves a checkpoint, and resuming from it must not run a tool
//! again whose side effect already happened.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::ToolNotFoundBehavior;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result, RunConfig, RunState, Runner};
use serde_json::{Value, json};
use tokio::sync::Notify;

type Log = Arc<Mutex<Vec<String>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn entries(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn assert_model_error(err: &AgentsError, needle: &str) {
    match err {
        AgentsError::ModelBehaviorError { message, .. } => {
            assert!(message.contains(needle), "{message:?} lacks {needle:?}")
        }
        other => panic!("expected ModelBehaviorError, got {other:?}"),
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

/// A hook that fails once, from the named callback.
#[derive(Default)]
struct FailOnce {
    on_tool_end_for: Option<&'static str>,
    wait_before_failing: Option<Arc<Notify>>,
    on_tool_start_for: Option<&'static str>,
    failed: Mutex<bool>,
    started: Log,
}

impl FailOnce {
    fn fail(&self, message: &str) -> Result<()> {
        let mut failed = self.failed.lock().unwrap();
        if *failed {
            return Ok(());
        }
        *failed = true;
        Err(AgentsError::user(message))
    }
}

#[async_trait]
impl RunHooks<()> for FailOnce {
    async fn try_on_tool_start(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
    ) -> Result<()> {
        if self.on_tool_start_for == Some(tool.name().as_str()) {
            self.started.lock().unwrap().push("ran".into());
            return self.fail("failed after side effect");
        }
        Ok(())
    }

    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &openai_agents_sdk::tool::ToolOutput,
    ) -> Result<()> {
        if self
            .on_tool_end_for
            .is_some_and(|n| n == "*" || n == tool.name().as_str())
        {
            if let Some(gate) = &self.wait_before_failing {
                gate.notified().await;
            }
            return self.fail("end hook failed");
        }
        Ok(())
    }
}

fn formatter_config(calls: &Log) -> RunConfig<()> {
    let calls = calls.clone();
    let mut config = RunConfig::default().with_tool_error_formatter(move |args| {
        calls.lock().unwrap().push(args.tool_name.clone());
        Some("error".to_string())
    });
    config.tool_not_found_behavior = ToolNotFoundBehavior::ReturnErrorToModel;
    config
}

fn record_value(executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    FunctionTool::new("record_value", "record", move |_c, a: ValueArgs| {
        let executed = executed.clone();
        async move {
            executed.lock().unwrap().push(a.value.clone());
            Ok(a.value)
        }
    })
    .with_needs_approval(true)
}

/// Run `agent` from the approved `state`, capturing the failed run's checkpoint.
async fn run_failing(
    agent: &Agent<()>,
    state: RunState,
    hooks: Arc<dyn RunHooks<()>>,
) -> (AgentsError, RunState) {
    let checkpoint = RunCheckpoint::new();
    let err = Runner::run_with_options(
        agent,
        state,
        RunOptions::default().hooks(hooks).checkpoint(&checkpoint),
    )
    .await
    .unwrap_err();
    (err, checkpoint.take().expect("a failure checkpoint"))
}

/// `test_bound_call_id_with_missing_arguments_fails_before_error_formatter`: a replacement that
/// reuses a bound call id without the fields that bind it is a reused id, not an unknown tool.
#[tokio::test]
async fn a_bound_call_id_with_missing_arguments_fails_before_the_error_formatter() {
    let executed = log();
    let formatter_calls = log();
    let e = executed.clone();
    let tool = FunctionTool::new("record_value", "record", move |_c, a: ValueArgs| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push(a.value.clone());
            Ok(a.value)
        }
    });
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "record_value",
        json!({"value": "safe"}).to_string(),
        "shared",
    )]);
    model.add_turn(vec![item(json!({
        "type": "function_call", "name": "missing", "call_id": "shared",
    }))]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let err = Runner::run_with_options(
        &agent,
        "run it",
        RunOptions::new(()).run_config(formatter_config(&formatter_calls)),
    )
    .await
    .unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["safe"]);
    assert!(entries(&formatter_calls).is_empty());
}

fn replay_scenario(executed: &Log) -> (Agent<()>, Arc<FakeModel>) {
    let model = model();
    let args = json!({"value": "safe"}).to_string();
    model.add_turn(vec![function_call_with_id("record_value", &args, "call_0")]);
    model.add_turn(vec![function_call_with_id("record_value", &args, "call_0")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("custom-provider-agent")
        .with_tool(record_value(executed))
        .with_model_instance(model.clone());
    (agent, model)
}

/// `test_failed_tool_end_hook_does_not_reexecute_approved_call`.
#[tokio::test]
async fn a_failed_tool_end_hook_does_not_reexecute_an_approved_call() {
    let executed = log();
    let (agent, _model) = replay_scenario(&executed);
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let hooks = Arc::new(FailOnce {
        on_tool_end_for: Some("record_value"),
        ..Default::default()
    });

    let (err, failed) = run_failing(&agent, state, hooks.clone()).await;
    assert!(err.to_string().contains("end hook failed"), "{err:?}");

    let resumed = Runner::run_with_options(&agent, failed, RunOptions::default().hooks(hooks))
        .await
        .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["safe"]);
}

/// `test_failed_approved_agent_tool_start_does_not_reexecute`: the start hook of an approved
/// agent-as-tool fails; resuming does not start (and so does not run) the tool again.
#[tokio::test]
async fn a_failed_approved_agent_tool_start_does_not_reexecute() {
    let inner = Agent::<()>::new("inner").with_model_instance({
        let m = model();
        m.add_turn(vec![assistant_message("inner done")]);
        m
    });
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("delegate")
            .tool_description("Delegate work.")
            .needs_approval(true),
    );
    let outer_model = model();
    outer_model.add_turn(vec![function_call_with_id(
        "delegate",
        r#"{"input":"hi"}"#,
        "call_0",
    )]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    let outer = Agent::<()>::new("outer")
        .with_tool(agent_tool)
        .with_model_instance(outer_model.clone());
    let first = Runner::run(&outer, "delegate").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let hooks = Arc::new(FailOnce {
        on_tool_start_for: Some("delegate"),
        ..Default::default()
    });

    let (err, failed) = run_failing(&outer, state, hooks.clone()).await;
    assert!(
        err.to_string().contains("failed after side effect"),
        "{err:?}"
    );

    // The start hook failed after the tool was marked as executing: the retry is refused rather
    // than starting the delegate a second time.
    let retry =
        Runner::run_with_options(&outer, failed, RunOptions::default().hooks(hooks.clone()))
            .await
            .unwrap_err();
    assert_model_error(&retry, "already executed");
    assert_eq!(entries(&hooks.started), vec!["ran"]);
}

/// `test_failed_parallel_tool_end_hook_checkpoints_outputs_in_model_order`: the second tool's end
/// hook fails after the first tool finished; the retry keeps both outputs in the model's order.
#[tokio::test]
async fn a_failed_parallel_tool_end_hook_checkpoints_outputs_in_model_order() {
    let executed = log();
    let first_finished = Arc::new(Notify::new());
    let (e1, gate) = (executed.clone(), first_finished.clone());
    let first_tool = FunctionTool::new("first_tool", "first", move |_c, _a: NoArgs| {
        let (e, gate) = (e1.clone(), gate.clone());
        async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            e.lock().unwrap().push("first".into());
            gate.notify_one();
            Ok("first".to_string())
        }
    })
    .with_needs_approval(true);
    let e2 = executed.clone();
    let second_tool = FunctionTool::new("second_tool", "second", move |_c, _a: NoArgs| {
        let e = e2.clone();
        async move {
            e.lock().unwrap().push("second".into());
            Ok("second".to_string())
        }
    })
    .with_needs_approval(true);
    let model = model();
    model.add_turn(vec![
        function_call_with_id("first_tool", "{}", "call_first"),
        function_call_with_id("second_tool", "{}", "call_second"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(first_tool)
        .with_tool(second_tool)
        .with_model_instance(model.clone());
    let hooks = Arc::new(FailOnce {
        on_tool_end_for: Some("second_tool"),
        wait_before_failing: Some(first_finished),
        ..Default::default()
    });

    let first = Runner::run(&agent, "run tools").await.unwrap();
    let mut state = first.to_state();
    for interruption in &first.interruptions {
        state.approve(interruption, false);
    }
    let (err, failed) = run_failing(&agent, state, hooks.clone()).await;
    assert!(
        err.to_string().contains("second end hook failed")
            || err.to_string().contains("end hook failed"),
        "{err:?}"
    );

    let resumed = Runner::run_with_options(&agent, failed, RunOptions::default().hooks(hooks))
        .await
        .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let mut ran = entries(&executed);
    ran.sort();
    assert_eq!(ran, vec!["first", "second"]);
    let input = model.last_call().expect("a model call").input.clone();
    let output_call_ids: Vec<String> = input
        .iter()
        .map(|i| i.as_value().clone())
        .filter(|v| v["type"] == "function_call_output")
        .map(|v| v["call_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(output_call_ids, vec!["call_first", "call_second"]);
}
