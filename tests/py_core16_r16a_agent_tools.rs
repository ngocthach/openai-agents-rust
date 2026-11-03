//! Ports of the non-streaming sub-agent failure cases of `tests/test_example_workflows.py`.
//! Python raises `asyncio.CancelledError` from the nested tool; Rust has no cancellation
//! exception, so the nested tool fails with an error, which exercises the same agent-tool
//! failure policy (default message vs. propagation).

#![allow(clippy::field_reassign_with_default)]

use std::sync::Arc;

use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, InputItem, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const DEFAULT_FAILURE: &str = "An error occurred while running the tool. Please try again.";

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn text(text: &str) -> ScriptedStep {
    ScriptedStep::output(vec![assistant_message(text)])
}

fn calls(calls: Vec<OutputItem>) -> ScriptedStep {
    ScriptedStep::output(calls)
}

fn tool_call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

fn model(steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model.extend(steps).unwrap();
    model
}

fn agent(name: &str, model: &Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new(name).with_model_instance(model.clone())
}

fn required() -> ModelSettings {
    ModelSettings {
        tool_choice: Some(ToolChoice::required()),
        ..Default::default()
    }
}

fn failing_tool() -> FunctionTool<()> {
    FunctionTool::<()>::new("cancel_tool", "fails", |_ctx, _args: NoArgs| async {
        Err::<String, _>(AgentsError::user("tool-cancelled"))
    })
}

fn tool_outputs(input: &[InputItem]) -> Vec<(String, String)> {
    input
        .iter()
        .filter_map(|item| {
            let value = serde_json::to_value(item).ok()?;
            (value["type"] == "function_call_output").then(|| {
                (
                    value["call_id"].as_str().unwrap_or_default().to_string(),
                    value["output"].as_str().unwrap_or_default().to_string(),
                )
            })
        })
        .collect()
}

fn outer_calls() -> ScriptedStep {
    calls(vec![
        tool_call("status_agent", json!({"input": "Hi"}), "outer_status"),
        tool_call(
            "observability_agent",
            json!({"input": "Hi"}),
            "outer_observability",
        ),
    ])
}

/// Python: `test_agents_as_tools_subagent_cancellation_preserves_parent_final_output`. A nested
/// run that cannot finish after its tool failed is reported to the parent model as the default
/// failure message, and the sibling sub-agent's output is kept.
#[tokio::test]
async fn a_failed_nested_run_keeps_the_sibling_output_and_reports_the_default_message() {
    let status_model = model(vec![text("Status: ok")]);
    let status = agent("status", &status_model);
    // The nested model has nothing after the tool failure, so the nested run itself fails.
    let observability_model = model(vec![
        calls(vec![tool_call("cancel_tool", json!({}), "inner_cancel")]),
        ScriptedStep::output(vec![]),
    ]);
    let observability = agent("observability", &observability_model)
        .with_tool(failing_tool())
        .with_model_settings(required());
    let orchestrator_model = model(vec![outer_calls(), text("Summary complete")]);
    let orchestrator = agent("orchestrator", &orchestrator_model)
        .with_tool(status.as_tool("status_agent", "Status"))
        .with_tool(observability.as_tool("observability_agent", "Observability"))
        .with_model_settings(required());

    let result = Runner::run(&orchestrator, "Hi").await.unwrap();

    assert_eq!(result.final_output_text().unwrap(), "Summary complete");
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(
        status_model.last_call().unwrap().request.input,
        vec![InputItem::user("Hi")]
    );
    assert!(!observability_model.calls().is_empty());
    assert_eq!(
        observability_model.calls()[0].request.input,
        vec![InputItem::user("Hi")]
    );
    assert_eq!(
        tool_outputs(&orchestrator_model.last_call().unwrap().request.input),
        vec![
            ("outer_status".to_string(), "Status: ok".to_string()),
            (
                "outer_observability".to_string(),
                DEFAULT_FAILURE.to_string()
            ),
        ]
    );
}

/// Python: `test_agents_as_tools_failure_error_function_none_reraises_cancelled_error`. With no
/// failure policy on the nested tool or on the agent tool, the failure ends the outer run.
#[tokio::test]
async fn without_a_failure_policy_a_nested_tool_failure_ends_the_outer_run() {
    let status_model = model(vec![text("Status: ok")]);
    let status = agent("status", &status_model);
    let observability_model = model(vec![calls(vec![tool_call(
        "cancel_tool",
        json!({}),
        "inner_cancel",
    )])]);
    let observability = agent("observability", &observability_model)
        .with_tool(failing_tool().without_failure_error_function())
        .with_model_settings(required());
    let orchestrator_model = model(vec![outer_calls()]);
    let orchestrator = agent("orchestrator", &orchestrator_model)
        .with_tool(status.as_tool("status_agent", "Status"))
        .with_tool(
            observability
                .as_tool("observability_agent", "Observability")
                .without_failure_error_function(),
        )
        .with_model_settings(required());

    let error = Runner::run(&orchestrator, "Hi").await.unwrap_err();

    assert!(error.to_string().contains("tool-cancelled"), "{error}");
    assert_eq!(orchestrator_model.calls().len(), 1);
}

// ---------------------------------------------------------------- resuming approved calls

use openai_agents_sdk::RunState;
use openai_agents_sdk::items::{FunctionCall, RunItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::testing::FakeModel;

#[derive(Deserialize, JsonSchema)]
struct CustomerArgs {
    customer_id: String,
}

fn lookup_account(label: &'static str, deferred: bool) -> FunctionTool<()> {
    FunctionTool::<()>::new(
        "lookup_account",
        "look up",
        move |_ctx, args: CustomerArgs| async move { Ok(format!("{label}:{}", args.customer_id)) },
    )
    .with_needs_approval(true)
    .with_defer_loading(deferred)
}

fn parked(agent: &str, item: &ToolApprovalItem) -> RunState {
    let mut state = RunState::default();
    state.schema_version = openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.into();
    state.current_agent_name = agent.into();
    state.max_turns = Some(10);
    state.original_input = vec![InputItem::user("resume approvals")];
    state.generated_items.push(RunItem::new(
        agent,
        RunItemKind::ToolApproval {
            raw_item: item.raw_item.clone(),
            tool_name: item.tool_name.clone(),
        },
    ));
    state.interruptions = vec![item.clone()];
    state.pending_tool_processing = true;
    state
}

fn approval(agent: &str, name: &str, call_id: &str, namespace: Option<&str>) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(name, r#"{"customer_id":"customer_1"}"#);
    raw.call_id = call_id.into();
    raw.status = Some("completed".into());
    if let Some(namespace) = namespace {
        raw.extra.insert("namespace".into(), json!(namespace));
    }
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: raw,
        tool_name: name.into(),
        tool_origin: None,
    }
}

fn call_outputs(result: &openai_agents_sdk::RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.to_string()),
            _ => None,
        })
        .collect()
}

/// Python: `test_resume_rebuilds_deferred_function_runs_from_lookup_key_without_raw_namespace`.
/// A visible and a deferred tool share a name; the approved call is rebuilt against the tool its
/// persisted identity names, not the visible namesake.
#[tokio::test]
async fn an_approved_deferred_call_resumes_on_the_deferred_tool_not_its_visible_namesake() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(lookup_account("visible", false))
        .with_tool(lookup_account("deferred", true))
        .with_model_instance(model.clone());
    let item = approval(
        "TestAgent",
        "lookup_account",
        "call-deferred-rebuild",
        Some("lookup_account"),
    );
    let mut state = parked("TestAgent", &item);
    state.approve(&item, false);

    let result = Runner::run(&agent, state).await.unwrap();

    assert!(result.interruptions.is_empty());
    let outputs = call_outputs(&result);
    assert_eq!(outputs.len(), 1, "{outputs:?}");
    assert!(outputs[0].contains("deferred:customer_1"), "{outputs:?}");
}

fn counted(name: &str, approval: bool, counter: &Arc<std::sync::Mutex<usize>>) -> FunctionTool<()> {
    let counter = counter.clone();
    let tool = FunctionTool::<()>::new(name, "tool", move |_ctx, _args: NoArgs| {
        let counter = counter.clone();
        async move {
            *counter.lock().unwrap() += 1;
            Ok("done".to_string())
        }
    });
    tool.with_needs_approval(approval)
}

fn output_call_ids(result: &openai_agents_sdk::RunResult<()>) -> Vec<(String, String)> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some((
                raw_item["call_id"].as_str().unwrap_or_default().to_string(),
                raw_item["output"].as_str().unwrap_or_default().to_string(),
            )),
            _ => None,
        })
        .collect()
}

/// Python: `test_rebuild_function_runs_handles_pending_and_rejections` and
/// `test_resume_skips_non_hitl_function_calls_with_existing_output`. A turn with a plain call, a
/// call that gets rejected and a call that stays pending: the resume emits the rejection output,
/// keeps the pending interruption, and does not run the plain call again.
#[tokio::test]
async fn a_resume_rejects_one_call_keeps_one_pending_and_does_not_rerun_a_finished_sibling() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let (plain_runs, reject_runs, pending_runs) = (Arc::default(), Arc::default(), Arc::default());
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        function_call_full("plain", "{}", "call-plain", None, None),
        function_call_full("reject_me", "{}", "call-reject", None, None),
        function_call_full("pending_me", "{}", "call-pending", None, None),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(counted("plain", false, &plain_runs))
        .with_tool(counted("reject_me", true, &reject_runs))
        .with_tool(counted("pending_me", true, &pending_runs))
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 2);
    assert_eq!(*plain_runs.lock().unwrap(), 1);

    let mut state = first.to_state();
    let rejected = state
        .get_interruptions()
        .into_iter()
        .find(|item| item.call_id() == "call-reject")
        .unwrap();
    state.reject(&rejected, false, None);
    let second = Runner::run(&agent, state).await.unwrap();

    let pending: Vec<String> = second
        .interruptions
        .iter()
        .map(|item| item.call_id().to_string())
        .collect();
    assert_eq!(pending, vec!["call-pending".to_string()]);
    let outputs = output_call_ids(&second);
    assert!(
        outputs.contains(&(
            "call-reject".to_string(),
            "Tool execution was not approved.".to_string()
        )),
        "{outputs:?}"
    );
    // The result carries the run's earlier items: the plain call's single output is one of them.
    assert_eq!(
        outputs.iter().filter(|(id, _)| id == "call-plain").count(),
        1,
        "{outputs:?}"
    );
    assert_eq!(*plain_runs.lock().unwrap(), 1);
    assert_eq!(*reject_runs.lock().unwrap(), 0);
    assert_eq!(*pending_runs.lock().unwrap(), 0);
}

/// Python: `test_resume_executes_non_hitl_function_calls_without_output`. When the snapshot kept
/// no output for a plain call of the interrupted turn, the resume runs it.
#[tokio::test]
async fn a_resume_runs_a_plain_sibling_whose_output_was_never_recorded() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let (plain_runs, gated_runs) = (Arc::default(), Arc::default());
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        function_call_full("plain", "{}", "call-plain", None, None),
        function_call_full("gated", "{}", "call-gated", None, None),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(counted("plain", false, &plain_runs))
        .with_tool(counted("gated", true, &gated_runs))
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(*plain_runs.lock().unwrap(), 1);

    let mut state = first.to_state();
    state.generated_items.retain(|item| {
        !matches!(&item.kind, RunItemKind::ToolCallOutput { raw_item, .. }
            if raw_item["call_id"] == "call-plain")
    });
    let gated = state.get_interruptions()[0].clone();
    state.approve(&gated, false);
    let second = Runner::run(&agent, state).await.unwrap();

    assert!(second.interruptions.is_empty());
    assert_eq!(*gated_runs.lock().unwrap(), 1);
    assert_eq!(*plain_runs.lock().unwrap(), 2);
    let outputs = output_call_ids(&second);
    assert!(
        outputs.contains(&("call-plain".to_string(), "done".to_string())),
        "{outputs:?}"
    );
}
