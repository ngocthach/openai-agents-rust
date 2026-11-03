//! Ports of `tests/test_tool_approval_call_id_reuse.py`, second half: same-response siblings,
//! response processing errors, computer safety checks, failed or cancelled tool bodies,
//! sticky decisions, legacy (pre-ledger) snapshots and hosted MCP approval requests.
//!
//! Python drives the private `collect_manual_mcp_approvals` / `process_hosted_mcp_approvals`
//! and `get_approval_status(current_invocation=...)` helpers; here the same decisions run end
//! to end through `Runner`, where those helpers are called.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{
    ComputerTool, CustomTool, FunctionTool, HostedMCPTool, MCPToolApprovalResult,
};
use openai_agents_sdk::{
    Agent, AgentsError, ModelResponse, RunContext, RunInput, RunResult, RunState, Runner,
};
use serde_json::{Value, json};

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

fn opts() -> RunOptions<()> {
    RunOptions::new(())
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: serde_json::Value,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn value_text(v: &Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_string)
}

fn record_tool(name: &str, executed: &Log, prefix: &str, needs_approval: bool) -> FunctionTool<()> {
    let executed = executed.clone();
    let prefix = prefix.to_string();
    FunctionTool::new(name, "record", move |_c, a: ValueArgs| {
        let executed = executed.clone();
        let prefix = prefix.clone();
        async move {
            let v = value_text(&a.value);
            executed.lock().unwrap().push(format!("{prefix}{v}"));
            Ok(v)
        }
    })
    .with_needs_approval(needs_approval)
}

async fn run_mode(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> (
    openai_agents_sdk::Result<RunResult<()>>,
    Vec<StreamEvent<()>>,
) {
    if !streamed {
        return (
            Runner::run_with_options(agent, input, options).await,
            Vec::new(),
        );
    }
    let mut stream = Runner::run_streamed_with_options(agent, input, options);
    let mut events = Vec::new();
    let mut error = None;
    {
        let mut s = std::pin::pin!(stream.stream_events());
        while let Some(event) = s.next().await {
            match event {
                Ok(e) => events.push(e),
                Err(e) => error = Some(e),
            }
        }
    }
    match error {
        Some(e) => (Err(e), events),
        None => (
            stream
                .into_result()
                .ok_or_else(|| AgentsError::user("stream produced no result")),
            events,
        ),
    }
}

fn assert_model_error(err: &AgentsError, needle: &str) {
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }) && err.to_string().contains(needle),
        "{err:?}"
    );
}

#[derive(Default)]
struct LlmEndCounter(Mutex<usize>);

#[async_trait]
impl RunHooks<()> for LlmEndCounter {
    async fn on_llm_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _r: &ModelResponse) {
        *self.0.lock().unwrap() += 1;
    }
}

fn shell_call(call_id: &str) -> Value {
    json!({
        "type": "shell_call", "id": call_id, "call_id": call_id, "status": "in_progress",
        "action": {"type": "exec", "commands": ["echo safe"], "timeout_ms": 1000},
    })
}

fn input_types(model: &FakeModel) -> Vec<Value> {
    model
        .last_call()
        .unwrap()
        .input
        .iter()
        .map(|i| serde_json::to_value(i).unwrap())
        .collect()
}

// --------------------------------------------------------- same-response siblings

#[tokio::test]
async fn changed_same_id_siblings_fail_before_approval_callback() {
    for streamed in [false, true] {
        let approval_calls = log();
        let calls = approval_calls.clone();
        let tool = FunctionTool::new("record_value", "record", |_c, a: ValueArgs| async move {
            Ok(value_text(&a.value))
        })
        .with_needs_approval_fn(move |_ctx, args: Value, _call_id: String| {
            let calls = calls.clone();
            async move {
                calls
                    .lock()
                    .unwrap()
                    .push(args["value"].as_str().unwrap_or_default().to_string());
                true
            }
        });
        let model = model();
        model.add_turn(vec![
            function_call_with_id("record_value", r#"{"value":"one"}"#, "call_0"),
            function_call_with_id("record_value", r#"{"value":"two"}"#, "call_0"),
        ]);
        let agent = Agent::<()>::new("agent")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "record values", opts(), streamed).await;
        assert_model_error(&result.unwrap_err(), "one response");
        assert!(entries(&approval_calls).is_empty());
    }
}

#[tokio::test]
async fn changed_same_id_siblings_fail_before_non_approval_execution() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        model.add_turn(vec![
            function_call_with_id("record_value", r#"{"value":"one"}"#, "call_0"),
            function_call_with_id("record_value", r#"{"value":"two"}"#, "call_0"),
        ]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_tool("record_value", &executed, "", false))
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "record values", opts(), streamed).await;
        assert_model_error(&result.unwrap_err(), "one response");
        assert!(entries(&executed).is_empty());
    }
}

#[tokio::test]
async fn response_processing_error_still_invokes_llm_end() {
    for streamed in [false, true] {
        let model = model();
        model.add_turn(vec![item(shell_call("call_0"))]);
        let agent = Agent::<()>::new("agent").with_model_instance(model.clone());
        let hooks = Arc::new(LlmEndCounter::default());
        let (result, _) =
            run_mode(&agent, "run command", opts().hooks(hooks.clone()), streamed).await;
        assert_model_error(&result.unwrap_err(), "without a shell tool");
        assert_eq!(*hooks.0.lock().unwrap(), 1);
    }
}

#[tokio::test]
async fn processing_error_with_changed_call_id_still_suppresses_llm_end() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id(
            "record_value",
            r#"{"value":"safe"}"#,
            "shared",
        )]);
        model.add_turn(vec![
            function_call_with_id("record_value", r#"{"value":"changed"}"#, "shared"),
            item(shell_call("shell_0")),
        ]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_tool("record_value", &executed, "", false))
            .with_model_instance(model.clone());
        let hooks = Arc::new(LlmEndCounter::default());
        let (result, _) = run_mode(
            &agent,
            "record value",
            opts().hooks(hooks.clone()),
            streamed,
        )
        .await;
        assert_model_error(&result.unwrap_err(), "unique call ID");
        assert_eq!(*hooks.0.lock().unwrap(), 1, "streamed={streamed}");
    }
}

#[tokio::test]
async fn processing_error_with_repeated_uncanonical_id_suppresses_llm_end() {
    // Two custom calls without `input` share one call id: neither is a complete invocation.
    for streamed in [false, true] {
        let model = model();
        model.add_turn(vec![
            item(json!({"type": "custom_tool_call", "name": "first_missing_tool", "call_id": "shared"})),
            item(json!({"type": "custom_tool_call", "name": "second_missing_tool", "call_id": "shared"})),
        ]);
        let agent = Agent::<()>::new("agent").with_model_instance(model.clone());
        let hooks = Arc::new(LlmEndCounter::default());
        let (result, _) = run_mode(&agent, "run tool", opts().hooks(hooks.clone()), streamed).await;
        assert_model_error(&result.unwrap_err(), "one response");
        assert_eq!(*hooks.0.lock().unwrap(), 0);
    }
}

// --------------------------------------------------------------- computer calls

#[derive(Clone)]
struct RecordingComputer(Log);

#[async_trait]
impl Computer for RecordingComputer {
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        self.0.lock().unwrap().push("screenshot".into());
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

fn computer_call(item_id: &str, check: &str) -> Value {
    json!({
        "id": item_id, "type": "computer_call", "action": {"type": "screenshot"},
        "call_id": "computer-call", "status": "completed",
        "pending_safety_checks": [{"id": check, "code": format!("code-{check}"), "message": check}],
    })
}

fn computer_tool(screens: &Log, checks: &Log) -> ComputerTool<()> {
    let checks = checks.clone();
    ComputerTool::new(RecordingComputer(screens.clone())).with_on_safety_check(move |data| {
        let checks = checks.clone();
        async move {
            checks.lock().unwrap().push(
                data.safety_check["id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            );
            true
        }
    })
}

#[tokio::test]
async fn changed_computer_safety_checks_fail_before_same_response_effects() {
    for streamed in [false, true] {
        let (screens, checks) = (log(), log());
        let model = model();
        model.add_turn(vec![
            item(computer_call("computer-item-1", "safety-1")),
            item(computer_call("computer-item-2", "safety-2")),
        ]);
        let agent = Agent::<()>::new("computer-agent")
            .with_tool(computer_tool(&screens, &checks))
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "use computer", opts(), streamed).await;
        assert_model_error(&result.unwrap_err(), "one response");
        assert!(entries(&checks).is_empty());
        assert!(entries(&screens).is_empty());
    }
}

#[tokio::test]
async fn changed_computer_safety_checks_fail_before_completed_replay_effects() {
    for streamed in [false, true] {
        let (screens, checks) = (log(), log());
        let model = model();
        model.add_turn(vec![item(computer_call("computer-item-1", "safety-1"))]);
        model.add_turn(vec![item(computer_call("computer-item-2", "safety-2"))]);
        let agent = Agent::<()>::new("computer-agent")
            .with_tool(computer_tool(&screens, &checks))
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "use computer", opts(), streamed).await;
        assert_model_error(&result.unwrap_err(), "completed tool call ID");
        assert_eq!(entries(&checks), vec!["safety-1"]);
        assert_eq!(entries(&screens), vec!["screenshot"]);
    }
}

// ------------------------------------------------------------- exact replays

fn reasoning_replay_model(model: &Arc<FakeModel>) {
    model.add_turn(vec![function_call_with_id(
        "record_value",
        r#"{"value":"safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![
        item(json!({"id": "rs_replay", "summary": [], "type": "reasoning"})),
        function_call_with_id("record_value", r#"{ "value" : "safe" }"#, "call_0"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
}

#[tokio::test]
async fn exact_replay_drops_tied_reasoning_item() {
    let executed = log();
    let model = model();
    reasoning_replay_model(&model);
    let agent = Agent::<()>::new("agent")
        .with_tool(record_tool("record_value", &executed, "", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let resumed = Runner::run(&agent, state).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["safe"]);
    let input = input_types(&model);
    assert!(!input.iter().any(|i| i["id"] == "rs_replay"), "{input:?}");
    assert_eq!(
        input
            .iter()
            .filter(|i| i["type"] == "function_call" && i["call_id"] == "call_0")
            .count(),
        1
    );
}

#[tokio::test]
async fn streamed_exact_replay_does_not_emit_tied_reasoning_item() {
    let executed = log();
    let model = model();
    reasoning_replay_model(&model);
    let agent = Agent::<()>::new("agent")
        .with_tool(record_tool("record_value", &executed, "", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let (result, events) = run_mode(&agent, state, opts(), true).await;
    assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["safe"]);
    let names: Vec<RunItemStreamEventName> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { name, .. } => Some(*name),
            _ => None,
        })
        .collect();
    assert!(
        !names.contains(&RunItemStreamEventName::ReasoningItemCreated),
        "{names:?}"
    );
    assert!(
        !names.contains(&RunItemStreamEventName::ToolCalled),
        "{names:?}"
    );
}

#[tokio::test]
async fn identical_approval_bound_siblings_execute_once() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let dup = function_call_with_id(
            "record_value",
            json!({"value": "safe"}).to_string(),
            "call-duplicate",
        );
        model.add_turn(vec![dup.clone(), dup]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("custom-provider-agent")
            .with_tool(record_tool("record_value", &executed, "", true))
            .with_model_instance(model.clone());
        let (first, _) = run_mode(&agent, "record a value", opts(), streamed).await;
        let first = first.unwrap();
        assert_eq!(first.interruptions.len(), 1);
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let (resumed, _) = run_mode(&agent, state, opts(), streamed).await;
        assert_eq!(
            resumed.unwrap().final_output_text().as_deref(),
            Some("done")
        );
        assert_eq!(entries(&executed), vec!["safe"]);
    }
}

#[tokio::test]
async fn serialized_sticky_identical_siblings_execute_once() {
    let executed = log();
    let model = model();
    let dup = function_call_with_id("record_value", r#"{"value":1}"#, "call_0");
    model.add_turn(vec![dup.clone(), dup]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(record_tool("record_value", &executed, "", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let resumed = Runner::run(&agent, restored).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["1"]);
}

// ---------------------------------------------- side effects that never committed

#[tokio::test]
async fn failed_approved_tool_body_does_not_reexecute() {
    let attempts = log();
    let a = attempts.clone();
    let tool = FunctionTool::new(
        "perform_side_effect",
        "side effect",
        move |_c, _a: NoArgs| {
            let a = a.clone();
            async move {
                a.lock().unwrap().push("ran".into());
                Err::<String, _>(AgentsError::user("failed after side effect"))
            }
        },
    )
    .with_needs_approval(true)
    .without_failure_error_function();
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "perform_side_effect",
        "{}",
        "call_0",
    )]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "run it").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);

    // Python retries the same `RunState` object, whose context the failed run mutated. A Rust
    // run consumes its state, so the retry shares the failed run's `RunContext` instead.
    let ctx = RunContext::new(());
    let err = Runner::run_with_options(
        &agent,
        state.clone(),
        RunOptions::with_run_context(ctx.clone()),
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("failed after side effect"),
        "{err:?}"
    );
    let err = Runner::run_with_options(&agent, state, RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_model_error(&err, "already executed");
    assert_eq!(entries(&attempts), vec!["ran"]);
}

#[tokio::test]
async fn cancelled_approved_tool_body_does_not_reexecute() {
    let attempts = log();
    let started = Arc::new(tokio::sync::Notify::new());
    let (a, s) = (attempts.clone(), started.clone());
    let tool = FunctionTool::new(
        "perform_side_effect",
        "side effect",
        move |_c, _a: NoArgs| {
            let (a, s) = (a.clone(), s.clone());
            async move {
                a.lock().unwrap().push("ran".into());
                s.notify_one();
                std::future::pending::<()>().await;
                Ok("done".to_string())
            }
        },
    )
    .with_needs_approval(true)
    .without_failure_error_function();
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "perform_side_effect",
        "{}",
        "call_0",
    )]);
    let agent = Arc::new(
        Agent::<()>::new("agent")
            .with_tool(tool)
            .with_model_instance(model.clone()),
    );
    let first = Runner::run(&agent, "run it").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);

    // The state is shared with the cancelled run through its context, as Python's state is.
    let ctx = RunContext::new(());
    let task = {
        let (agent, state, ctx) = (agent.clone(), state.clone(), ctx.clone());
        tokio::spawn(async move {
            Runner::run_with_options(&agent, state, RunOptions::with_run_context(ctx)).await
        })
    };
    started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    let err = Runner::run_with_options(&agent, state, RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_model_error(&err, "already executed");
    assert_eq!(entries(&attempts), vec!["ran"]);
}

#[tokio::test]
async fn failed_output_guardrail_does_not_reexecute_approved_call() {
    use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolOutputGuardrail};
    let executed = log();
    let e = executed.clone();
    let tool = FunctionTool::new("record_value", "record", move |_c, _a: NoArgs| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push("ran".into());
            Ok("sensitive".to_string())
        }
    })
    .with_needs_approval(true)
    .with_output_guardrail(ToolOutputGuardrail::new(
        "reject_output",
        |_data| async move { Ok(ToolGuardrailFunctionOutput::raise_exception()) },
    ));
    let model = model();
    model.add_turn(vec![function_call_with_id("record_value", "{}", "call_0")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let ctx = RunContext::new(());
    let err = Runner::run_with_options(
        &agent,
        state.clone(),
        RunOptions::with_run_context(ctx.clone()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            err,
            AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
        ),
        "{err:?}"
    );
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let err = Runner::run_with_options(&agent, restored, RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_model_error(&err, "already executed");
    assert_eq!(entries(&executed), vec!["ran"]);
}

// ------------------------------------------------------------- per-call decisions

fn approval_scenario(
    model: &Arc<FakeModel>,
    second_id: &str,
    second_value: &str,
    executed: &Log,
) -> Agent<()> {
    model.add_turn(vec![function_call_with_id(
        "record_value",
        r#"{"value": "safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![function_call_with_id(
        "record_value",
        json!({"value": second_value}).to_string(),
        second_id,
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("custom-provider-agent")
        .with_tool(record_tool("record_value", executed, "", true))
        .with_model_instance(model.clone())
}

#[tokio::test]
async fn new_call_id_requires_a_new_per_call_approval() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let agent = approval_scenario(&model, "call_1", "changed", &executed);
        let (first, _) = run_mode(&agent, "record a value", opts(), streamed).await;
        let first = first.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let (resumed, _) = run_mode(&agent, state, opts(), streamed).await;
        let resumed = resumed.unwrap();
        assert_eq!(resumed.interruptions.len(), 1);
        assert_eq!(
            resumed.interruptions[0].arguments(),
            json!({"value": "changed"}).to_string()
        );
        assert_eq!(entries(&executed), vec!["safe"]);
    }
}

#[tokio::test]
async fn serialized_sticky_decision_rejects_changed_reused_call_id() {
    for approve in [true, false] {
        let executed = log();
        let model = model();
        let agent = approval_scenario(&model, "call_0", "changed", &executed);
        let first = Runner::run(&agent, "record a value").await.unwrap();
        let mut state = first.to_state();
        if approve {
            state.approve(&first.interruptions[0], true);
        } else {
            state.reject(&first.interruptions[0], true, None);
        }
        let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
        let err = Runner::run(&agent, restored).await.unwrap_err();
        assert_model_error(&err, "unique call ID");
        let expected: Vec<String> = if approve { vec!["safe".into()] } else { vec![] };
        assert_eq!(entries(&executed), expected);
    }
}

#[tokio::test]
async fn sticky_upgrade_rejects_changed_reuse_of_prior_call_id() {
    let executed = log();
    let model = model();
    let agent = approval_scenario(&model, "call_0", "changed", &executed);
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    state.approve(&first.interruptions[0], true);
    let err = Runner::run(&agent, state).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["safe"]);
}

#[tokio::test]
async fn sticky_decision_keeps_other_call_bindings() {
    // `test_sticky_decision_preserves_prior_per_call_binding`,
    // `test_sticky_decision_does_not_disable_other_tool_binding` and
    // `test_matching_sticky_decision_does_not_mask_other_tool_binding`: a later sticky decision
    // about one tool never releases the binding of an earlier call id, of that tool or another.
    for (first_tool, sticky_tool, replay_tool) in [
        ("tool_a", "tool_a", "tool_a"),
        ("tool_b", "tool_a", "tool_b"),
        ("tool_b", "tool_a", "tool_a"),
    ] {
        for approve in [true, false] {
            let executed = log();
            let model = model();
            model.add_turn(vec![function_call_with_id(
                first_tool,
                r#"{"value":1}"#,
                "call_0",
            )]);
            model.add_turn(vec![function_call_with_id(
                sticky_tool,
                r#"{"value":2}"#,
                "call_1",
            )]);
            model.add_turn(vec![function_call_with_id(
                replay_tool,
                r#"{"value":3}"#,
                "call_0",
            )]);
            let agent = Agent::<()>::new("agent")
                .with_tool(record_tool("tool_a", &executed, "a:", true))
                .with_tool(record_tool("tool_b", &executed, "b:", true))
                .with_model_instance(model.clone());
            let first = Runner::run(&agent, "go").await.unwrap();
            let mut state = first.to_state();
            state.approve(&first.interruptions[0], false);
            let second = Runner::run(&agent, state).await.unwrap();
            let mut state = second.to_state();
            if approve {
                state.approve(&second.interruptions[0], true);
            } else {
                state.reject(&second.interruptions[0], true, None);
            }
            let err = Runner::run(&agent, state).await.unwrap_err();
            assert_model_error(&err, "unique call ID");
            assert_eq!(entries(&executed).len(), 1 + usize::from(approve));
        }
    }
}

#[tokio::test]
async fn sticky_function_approval_rejects_same_id_shell_call() {
    let executed = log();
    let model = model();
    model.add_turn(vec![function_call_with_id("shell", "{}", "call_0")]);
    model.add_turn(vec![item(json!({
        "type": "shell_call", "call_id": "call_0", "action": {"commands": ["echo safe"]},
    }))]);
    let e = executed.clone();
    let agent = Agent::<()>::new("agent")
        .with_tool(
            FunctionTool::new("shell", "fn", move |_c, _a: NoArgs| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push("function".into());
                    Ok("ok".to_string())
                }
            })
            .with_needs_approval(true),
        )
        .with_tool(
            openai_agents_sdk::tool::ShellTool::new(|_req| async { Ok("ok") })
                .with_name("local_shell_tool"),
        )
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "go").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let err = Runner::run(&agent, state).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["function"]);
}

#[tokio::test]
async fn changed_tool_under_approved_call_id_fails_before_second_tool_starts() {
    let executed = log();
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "first_tool",
        r#"{"value": "same"}"#,
        "call_0",
    )]);
    model.add_turn(vec![function_call_with_id(
        "second_tool",
        r#"{"value": "same"}"#,
        "call_0",
    )]);
    let agent = Agent::<()>::new("custom-provider-agent")
        .with_tool(record_tool("first_tool", &executed, "first:", true))
        .with_tool(record_tool("second_tool", &executed, "second:", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "run tools").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let err = Runner::run(&agent, state).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["first:same"]);
}

#[tokio::test]
async fn approved_call_id_reused_for_another_invocation_type_fails_before_execution() {
    let executed = log();
    let e = executed.clone();
    let custom = CustomTool::new("raw_editor", "Edit raw text.", move |_c, input: String| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push(format!("custom:{input}"));
            Ok(input)
        }
    })
    .with_format(json!({"type": "text"}));
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "approved_tool",
        r#"{"value": "safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![item(json!({
        "type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0", "input": "changed-kind",
    }))]);
    let agent = Agent::<()>::new("custom-provider-agent")
        .with_tool(record_tool("approved_tool", &executed, "function:", true))
        .with_tool(custom)
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "run tools").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let err = Runner::run(&agent, state).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["function:safe"]);
}

#[tokio::test]
async fn changed_missing_tool_under_approved_call_id_fails_before_sibling_tool() {
    let executed = log();
    let checks = log();
    let c = checks.clone();
    let e = executed.clone();
    let sibling = FunctionTool::new("sibling_tool", "sibling", move |_c, _a: NoArgs| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push("sibling".into());
            Ok("sibling".to_string())
        }
    })
    .with_needs_approval_fn(move |_ctx, _args: Value, call_id: String| {
        let c = c.clone();
        async move {
            c.lock().unwrap().push(call_id);
            false
        }
    });
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "approved_tool",
        r#"{"value": "safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![
        function_call_with_id("sibling_tool", "{}", "call_1"),
        function_call_with_id("missing_tool", "{}", "call_0"),
    ]);
    let mut config = openai_agents_sdk::RunConfig::default();
    config.tool_not_found_behavior =
        openai_agents_sdk::run_config::ToolNotFoundBehavior::ReturnErrorToModel;
    let agent = Agent::<()>::new("custom-provider-agent")
        .with_tool(record_tool("approved_tool", &executed, "approved:", true))
        .with_tool(sibling)
        .with_model_instance(model.clone());
    let first = Runner::run_with_options(&agent, "run tools", opts().run_config(config.clone()))
        .await
        .unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let err = Runner::run_with_options(&agent, state, opts().run_config(config))
        .await
        .unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["approved:safe"]);
    assert!(entries(&checks).is_empty());
}

// ------------------------------------------------------- serialized replays

fn replay_scenario(model: &Arc<FakeModel>, replay_value: &str, executed: &Log) -> Agent<()> {
    model.add_turn(vec![function_call_with_id(
        "record_value",
        r#"{"value": "safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_gate",
        r#"{"value": "pause"}"#,
        "call_1",
    )]);
    model.add_turn(vec![function_call_with_id(
        "record_value",
        json!({"value": replay_value}).to_string(),
        "call_0",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("custom-provider-agent")
        .with_tool(record_tool("record_value", executed, "record:", true))
        .with_tool(record_tool("approval_gate", executed, "gate:", true))
        .with_model_instance(model.clone())
}

#[tokio::test]
async fn serialized_pending_approval_keeps_its_invocation_binding() {
    let executed = log();
    let model = model();
    let agent = approval_scenario(&model, "call_0", "changed", &executed);
    let first = Runner::run(&agent, "record a value").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let err = Runner::run(&agent, restored).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["safe"]);
}

/// Run the replay scenario up to the gate's approval, restored through `transform`.
async fn replay_to_gate(
    agent: &Agent<()>,
    streamed: bool,
    transform: impl Fn(Value) -> Value,
) -> RunState {
    let (first, _) = run_mode(agent, "record a value", opts(), streamed).await;
    let first = first.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let (second, _) = run_mode(agent, state, opts(), streamed).await;
    let second = second.unwrap();
    let mut restored =
        RunState::from_json(transform(second.to_state().to_json().unwrap())).unwrap();
    let gate = restored.get_interruptions()[0].clone();
    restored.approve(&gate, false);
    restored
}

#[tokio::test]
async fn serialized_completed_approval_rejects_changed_replay() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let agent = replay_scenario(&model, "changed", &executed);
        let restored = replay_to_gate(&agent, streamed, |v| v).await;
        let (err, _) = run_mode(&agent, restored, opts(), streamed).await;
        assert_model_error(&err.unwrap_err(), "unique call ID");
        assert_eq!(entries(&executed), vec!["record:safe", "gate:pause"]);
    }
}

#[tokio::test]
async fn serialized_completed_approval_skips_exact_replay() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let agent = replay_scenario(&model, "safe", &executed);
        let restored = replay_to_gate(&agent, streamed, |v| v).await;
        let (result, _) = run_mode(&agent, restored, opts(), streamed).await;
        assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
        assert_eq!(entries(&executed), vec!["record:safe", "gate:pause"]);
        let input = input_types(&model);
        for call_id in ["call_0", "call_1"] {
            let count = |ty: &str| {
                input
                    .iter()
                    .filter(|i| i["type"] == ty && i["call_id"] == call_id)
                    .count()
            };
            assert_eq!(count("function_call"), 1, "{call_id}");
            assert_eq!(count("function_call_output"), 1, "{call_id}");
        }
    }
}

/// Rewrite a snapshot the way a schema 1.13 writer stored it: no invocation ledger and
/// function decisions in the agent-agnostic `approvals` map.
fn as_schema_1_13(mut json: Value) -> Value {
    let entries = json["context"]
        .as_object_mut()
        .unwrap()
        .remove("function_tool_approvals")
        .unwrap_or(json!([]));
    for entry in entries.as_array().unwrap() {
        json["context"]["approvals"][entry["tool_key"].as_str().unwrap()] =
            entry["decision"].clone();
    }
    json["$schemaVersion"] = json!("1.13");
    json["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    json
}

#[tokio::test]
async fn schema_1_13_completed_approval_rejects_changed_replay() {
    let executed = log();
    let model = model();
    let agent = replay_scenario(&model, "changed", &executed);
    let restored = replay_to_gate(&agent, false, as_schema_1_13).await;
    let err = Runner::run(&agent, restored).await.unwrap_err();
    assert_model_error(&err, "unique call ID");
    assert_eq!(entries(&executed), vec!["record:safe", "gate:pause"]);
}

#[tokio::test]
async fn schema_1_13_completed_approval_skips_exact_replay() {
    let executed = log();
    let model = model();
    let agent = replay_scenario(&model, "safe", &executed);
    let restored = replay_to_gate(&agent, false, as_schema_1_13).await;
    let result = Runner::run(&agent, restored).await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["record:safe", "gate:pause"]);
}

#[tokio::test]
async fn legacy_schema_historical_sticky_call_id_is_not_reexecuted() {
    for schema in ["1.13", "1.14"] {
        for streamed in [false, true] {
            for replay in ["safe", "changed"] {
                let executed = log();
                let model = model();
                model.add_turn(vec![function_call_with_id(
                    "record_value",
                    json!({"value": replay}).to_string(),
                    "call_0",
                )]);
                model.add_turn(vec![assistant_message("done")]);
                let agent = Agent::<()>::new("agent")
                    .with_tool(record_tool("record_value", &executed, "", true))
                    .with_model_instance(model.clone());
                let mut state = RunState::default();
                state.current_agent_name = "agent".into();
                state.max_turns = Some(10);
                state.original_input = vec![
                    openai_agents_sdk::InputItem(json!({
                        "type": "function_call", "call_id": "call_0", "name": "record_value",
                        "arguments": "{\"value\":\"safe\"}",
                    })),
                    openai_agents_sdk::InputItem(json!({
                        "type": "function_call_output", "call_id": "call_0", "output": "safe",
                    })),
                ];
                let mut json = state.to_json().unwrap();
                json["context"]["approvals"] =
                    json!({"record_value": {"approved": true, "rejected": []}});
                json["$schemaVersion"] = json!(schema);
                json["context"]
                    .as_object_mut()
                    .unwrap()
                    .remove("tool_invocations");
                let restored = RunState::from_json(json).unwrap();
                let (result, _) = run_mode(&agent, restored, opts(), streamed).await;
                if replay == "changed" {
                    assert_model_error(&result.unwrap_err(), "unique call ID");
                } else {
                    assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
                }
                assert!(
                    entries(&executed).is_empty(),
                    "{schema} {streamed} {replay}"
                );
            }
        }
    }
}

// ------------------------------------------------------------ hosted MCP requests

fn mcp_request(id: &str, server: &str, name: &str, arguments: Option<&str>) -> Value {
    let mut v =
        json!({"id": id, "type": "mcp_approval_request", "name": name, "server_label": server});
    if let Some(a) = arguments {
        v["arguments"] = json!(a);
    }
    v
}

fn mcp_tool(server: &str) -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": server, "server_url": "https://example.com",
        "require_approval": "always",
    }))
}

fn mcp_responses(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn sticky_mcp_approval_rejects_same_id_on_another_server() {
    // Also `test_wrapped_mcp_approval_uses_the_same_canonical_request_id`: a changed request
    // under an approved request id is another invocation.
    for (server, args) in [
        ("server_b", r#"{"query": "safe"}"#),
        ("server_a", r#"{"query":"changed"}"#),
    ] {
        let model = model();
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "server_a",
            "lookup",
            Some(r#"{"query": "safe"}"#),
        ))]);
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            server,
            "lookup",
            Some(args),
        ))]);
        let agent = Agent::<()>::new("mcp-approval-agent")
            .with_tool(mcp_tool("server_a"))
            .with_tool(mcp_tool("server_b"))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "lookup").await.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], true);
        let err = Runner::run(&agent, state).await.unwrap_err();
        assert_model_error(&err, "unique call ID");
    }
}

#[tokio::test]
async fn sticky_mcp_approval_reprompts_new_id_on_another_server() {
    for serialized in [false, true] {
        let model = model();
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "server_a",
            "lookup",
            Some(r#"{"query": "safe"}"#),
        ))]);
        model.add_turn(vec![item(mcp_request(
            "mcp_call_1",
            "server_b",
            "lookup",
            Some(r#"{"query": "safe"}"#),
        ))]);
        let agent = Agent::<()>::new("mcp-approval-agent")
            .with_tool(mcp_tool("server_a"))
            .with_tool(mcp_tool("server_b"))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "lookup").await.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], true);
        if serialized {
            state = RunState::from_json(state.to_json().unwrap()).unwrap();
        }
        let second = Runner::run(&agent, state).await.unwrap();
        assert_eq!(second.interruptions.len(), 1, "serialized={serialized}");
        assert_eq!(second.interruptions[0].call_id(), "mcp_call_1");
        assert_eq!(mcp_responses(&second).len(), 1);
    }
}

#[tokio::test]
async fn sticky_mcp_approval_rejects_same_id_for_changed_tool_name() {
    // The manual and hosted halves of the Python test share one runner path here.
    for (name, args) in [
        ("lookup", r#"{"query": "changed"}"#),
        ("delete_all", r#"{"confirm": true}"#),
    ] {
        let model = model();
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "test_server",
            "lookup",
            Some(r#"{"query": "safe"}"#),
        ))]);
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "test_server",
            name,
            Some(args),
        ))]);
        let agent = Agent::<()>::new("mcp-approval-agent")
            .with_tool(mcp_tool("test_server"))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "lookup").await.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], true);
        let err = Runner::run(&agent, state).await.unwrap_err();
        assert_model_error(&err, "unique call ID");
    }
}

#[tokio::test]
async fn unbindable_mcp_callback_request_requires_manual_reapproval() {
    let calls = Arc::new(Mutex::new(0usize));
    let c = calls.clone();
    let tool = mcp_tool("test_server").with_on_approval_request(move |_req| {
        let c = c.clone();
        async move {
            *c.lock().unwrap() += 1;
            MCPToolApprovalResult {
                approve: true,
                reason: None,
            }
        }
    });
    let model = model();
    model.add_turn(vec![item(mcp_request(
        "mcp_call_0",
        "test_server",
        "lookup",
        None,
    ))]);
    let agent = Agent::<()>::new("mcp-approval-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "lookup").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), 0);
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].call_id(), "mcp_call_0");
}

#[tokio::test]
async fn unbindable_current_mcp_request_requires_reapproval() {
    // `test_unbindable_current_manual_mcp_request_requires_reapproval` (per-call, sticky) and
    // `test_unbindable_current_hosted_mcp_request_requires_reapproval`: a decision never
    // answers a request that cannot be bound to one invocation.
    for always in [false, true] {
        let model = model();
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "test_server",
            "lookup",
            Some(r#"{"query": "safe"}"#),
        ))]);
        model.add_turn(vec![item(mcp_request(
            "mcp_call_1",
            "test_server",
            "lookup",
            None,
        ))]);
        let agent = Agent::<()>::new("mcp-approval-agent")
            .with_tool(mcp_tool("test_server"))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "lookup").await.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], always);
        let second = Runner::run(&agent, state).await.unwrap();
        assert_eq!(second.interruptions.len(), 1, "always={always}");
        assert_eq!(second.interruptions[0].call_id(), "mcp_call_1");
    }
}

#[tokio::test]
async fn unbindable_legacy_mcp_approval_requires_current_reapproval() {
    for schema in ["1.13", "1.14"] {
        for missing in ["arguments", "server_label"] {
            let model = model();
            model.add_turn(vec![item(mcp_request(
                "mcp_call_0",
                "test_server",
                "lookup",
                Some(r#"{"query": "safe"}"#),
            ))]);
            let agent = Agent::<()>::new("mcp-approval-agent")
                .with_tool(mcp_tool("test_server"))
                .with_model_instance(model.clone());
            let first = Runner::run(&agent, "lookup").await.unwrap();
            let mut json = first.to_state().to_json().unwrap();
            json["$schemaVersion"] = json!(schema);
            json["context"]
                .as_object_mut()
                .unwrap()
                .remove("tool_invocations");
            json["context"]["approvals"] =
                json!({"lookup": {"approved": ["mcp_call_0"], "rejected": []}});
            json["current_step"]["data"]["interruptions"][0]["raw_item"]
                .as_object_mut()
                .unwrap()
                .remove(missing);
            let restored = RunState::from_json(json).unwrap();
            let resumed = Runner::run(&agent, restored).await.unwrap();
            assert_eq!(resumed.interruptions.len(), 1, "{schema} {missing}");
            assert!(mcp_responses(&resumed).is_empty(), "{schema} {missing}");
        }
    }
}

#[tokio::test]
async fn runner_omits_completed_mcp_approval_request_replay() {
    // Also `test_serialized_completed_manual_mcp_approval_skips_exact_replay` and
    // `test_completed_hosted_mcp_approval_reconciliation_skips_exact_replay`.
    for with_callback in [false, true] {
        let calls = Arc::new(Mutex::new(0usize));
        let c = calls.clone();
        let mut tool = mcp_tool("test_server");
        if with_callback {
            tool = tool.with_on_approval_request(move |_req| {
                let c = c.clone();
                async move {
                    *c.lock().unwrap() += 1;
                    MCPToolApprovalResult {
                        approve: true,
                        reason: None,
                    }
                }
            });
        }
        let model = model();
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "test_server",
            "lookup",
            Some(r#"{"query": "safe", "limit": 1}"#),
        ))]);
        model.add_turn(vec![item(mcp_request(
            "mcp_call_0",
            "test_server",
            "lookup",
            Some(r#"{"limit":1,"query":"safe"}"#),
        ))]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("mcp-approval-agent")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "lookup").await.unwrap();
        let result = if with_callback {
            first
        } else {
            assert_eq!(first.interruptions.len(), 1);
            let mut state = first.to_state();
            state.approve(&first.interruptions[0], false);
            let state = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
            Runner::run(&agent, state).await.unwrap()
        };
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(*calls.lock().unwrap(), usize::from(with_callback));
        let input = input_types(&model);
        assert_eq!(
            input
                .iter()
                .filter(|i| i["type"] == "mcp_approval_request" && i["id"] == "mcp_call_0")
                .count(),
            1
        );
        assert_eq!(
            input
                .iter()
                .filter(|i| i["type"] == "mcp_approval_response"
                    && i["approval_request_id"] == "mcp_call_0")
                .count(),
            1
        );
    }
}

// ------------------------------------------------------------- history checks

#[tokio::test]
async fn history_call_id_reuse_is_checked_without_current_response_call() {
    for (second, should_raise) in [
        (r#"{ "value" : "safe" }"#, false),
        (r#"{"value":"changed"}"#, true),
    ] {
        let executed = log();
        let model = model();
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_tool("record_value", &executed, "", false))
            .with_model_instance(model.clone());
        let call = |args: &str| {
            openai_agents_sdk::items::RunItem::new(
                "agent",
                RunItemKind::ToolCall {
                    raw_item: json!({"type": "function_call", "call_id": "call_0", "name": "record_value", "arguments": args}),
                },
            )
        };
        let output = || {
            openai_agents_sdk::items::RunItem::new(
                "agent",
                RunItemKind::ToolCallOutput {
                    raw_item: json!({"type": "function_call_output", "call_id": "call_0", "output": "ok"}),
                    output: json!("ok"),
                    custom_data: None,
                },
            )
        };
        let mut state = RunState::default();
        state.schema_version = openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.into();
        state.current_agent_name = "agent".into();
        state.max_turns = Some(10);
        state.original_input = vec![openai_agents_sdk::InputItem::user("go")];
        state.generated_items = vec![
            call(r#"{"value":"safe"}"#),
            output(),
            call(second),
            output(),
        ];
        let result = Runner::run(&agent, state).await;
        if should_raise {
            assert_model_error(&result.unwrap_err(), "Run history reused a tool call ID");
        } else {
            assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
        }
    }
}
