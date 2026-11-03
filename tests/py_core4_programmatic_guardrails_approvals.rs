//! Guardrail, approval and rebuilt-state cases of Python's `tests/test_programmatic_tool_calling.py`
//! for programmatic function calls: input and output guardrails, the default timeout result,
//! approval rejection (in-memory and serialized, streamed and not), and approvals resumed from a
//! rebuilt state.
//!
//! Python edits private `RunState` attributes (`_last_processed_response`, `_generated_items`,
//! `_model_responses`) to rebuild a paused state. Rust reproduces the same states through the
//! public JSON form: `RunState::to_json` and `from_json_str`, with the parent program items edited
//! inside that JSON.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run_config::{RunConfig, ToolExecutionConfig};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{FunctionTool, ProgrammaticToolCallingTool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::{Agent, AgentsError, RunResult, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const PROGRAM_CALL_ID: &str = "call_program";
const FUNCTION_CALL_ID: &str = "call_lookup";

fn program_caller() -> Value {
    json!({"type": "program", "caller_id": PROGRAM_CALL_ID})
}

fn program_item() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program", "id": "program_item", "call_id": PROGRAM_CALL_ID,
        "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
    }))
}

fn lookup_call() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
        "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
        "caller": program_caller(),
    }))
}

fn program_output(status: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program_output", "id": "program_output_item", "call_id": PROGRAM_CALL_ID,
        "result": "{\"sku\":\"A-1\",\"available_units\":42}", "status": status,
    }))
}

fn text(message: &str) -> OutputItem {
    assistant_message(message)
}

fn model(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let m = Arc::new(ScriptedModel::new());
    m.extend(steps.into_iter().map(ScriptedStep::output).collect())
        .unwrap();
    m
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct InventoryOutput {
    sku: String,
    available_units: i64,
}

#[derive(Deserialize, JsonSchema)]
struct SkuArgs {
    #[allow(dead_code)]
    sku: String,
}

/// The `lookup_inventory` tool. `executed` records whether the body ran.
fn lookup(executed: Arc<AtomicBool>) -> FunctionTool<()> {
    FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let executed = executed.clone();
            async move {
                executed.store(true, Ordering::SeqCst);
                Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()])
    .with_output_type::<InventoryOutput>()
    .unwrap()
}

fn agent(m: Arc<ScriptedModel>, tool: FunctionTool<()>) -> Agent<()> {
    Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool)
}

fn function_outputs(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. }
                if raw_item["type"] == "function_call_output" =>
            {
                Some(raw_item.clone())
            }
            _ => None,
        })
        .collect()
}

fn caller_of(raw: &Value) -> Value {
    raw["caller"].clone()
}

fn reject_input(message: &'static str) -> ToolInputGuardrail<()> {
    ToolInputGuardrail::new("reject_tool_input", move |_data| async move {
        Ok(ToolGuardrailFunctionOutput::reject_content(message))
    })
}

fn reject_output(message: &'static str) -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::new("reject_tool_output", move |_data| async move {
        Ok(ToolGuardrailFunctionOutput::reject_content(message))
    })
}

fn two_step_model() -> Arc<ScriptedModel> {
    model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), text("request rejected")],
    ])
}

// ---------------------------------------------------------------------------------------------
// Guardrails and timeouts on programmatic calls
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn typed_programmatic_tool_preserves_input_guardrail_rejection() {
    let executed = Arc::new(AtomicBool::new(false));
    let tool =
        lookup(executed.clone()).with_input_guardrail(reject_input("inventory lookup blocked"));
    let result = Runner::run(agent(two_step_model(), tool), "Check inventory")
        .await
        .unwrap();

    assert!(!executed.load(Ordering::SeqCst));
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("request rejected")
    );
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
        json!({"error": "inventory lookup blocked"})
    );
    assert_eq!(caller_of(&outputs[0]), program_caller());
}

#[tokio::test]
async fn typed_programmatic_tool_preserves_default_timeout_result() {
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), text("request timed out")],
    ]);
    let executed = Arc::new(AtomicBool::new(false));
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let executed = executed.clone();
            async move {
                executed.store(true, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()])
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_timeout(Duration::from_millis(10));

    let result = Runner::run(agent(m, tool), "Check inventory")
        .await
        .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("request timed out")
    );
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 1);
    let timeout: Value = serde_json::from_str(outputs[0]["output"].as_str().unwrap()).unwrap();
    assert!(timeout.is_object(), "{timeout}");
    assert!(
        timeout["error"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("timed out"),
        "{timeout}"
    );
    assert_eq!(caller_of(&outputs[0]), program_caller());
}

#[tokio::test]
async fn typed_programmatic_tool_preserves_output_guardrail_rejection() {
    let executed = Arc::new(AtomicBool::new(false));
    let tool = lookup(executed).with_output_guardrail(reject_output("inventory result blocked"));
    let result = Runner::run(agent(two_step_model(), tool), "Check inventory")
        .await
        .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("request rejected")
    );
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
        json!({"error": "inventory result blocked"})
    );
    assert_eq!(caller_of(&outputs[0]), program_caller());
}

#[tokio::test]
async fn typed_programmatic_tool_preserves_pre_approval_guardrail_rejection() {
    // With pre-approval guardrails enabled, the input guardrail rejects before the approval
    // interruption, so the program-owned call never pauses.
    let executed = Arc::new(AtomicBool::new(false));
    let tool = lookup(executed.clone())
        .with_needs_approval(true)
        .with_input_guardrail(reject_input("inventory lookup blocked"));
    let config = RunConfig::default().with_tool_execution(ToolExecutionConfig {
        pre_approval_tool_input_guardrails: true,
        ..Default::default()
    });
    let result = Runner::run_with_options(
        agent(two_step_model(), tool),
        "Check inventory",
        openai_agents_sdk::RunOptions::new(()).run_config(config),
    )
    .await
    .unwrap();

    assert!(result.interruptions.is_empty());
    assert!(!executed.load(Ordering::SeqCst));
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("request rejected")
    );
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
        json!({"error": "inventory lookup blocked"})
    );
    assert_eq!(caller_of(&outputs[0]), program_caller());
}

// ---------------------------------------------------------------------------------------------
// Approval rejection (streamed and not, in-memory and serialized)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn typed_programmatic_tool_preserves_approval_rejection() {
    // Python's 8 cases: streaming x serialized state x default or custom rejection message.
    for streaming in [false, true] {
        for serialize_state in [false, true] {
            for rejection_message in [None, Some("Denied: \"東京\"".to_string())] {
                let ctx = format!("streaming={streaming} serialized={serialize_state}");
                let m = model(vec![
                    vec![program_item(), lookup_call()],
                    vec![program_output("completed"), text("request rejected")],
                ]);
                let executed = Arc::new(AtomicBool::new(false));
                let agent = agent(
                    m.clone(),
                    lookup(executed.clone()).with_needs_approval(true),
                );

                let first = if streaming {
                    let mut run = Runner::run_streamed(agent.clone(), "Check inventory");
                    run.finish().await.unwrap();
                    run.to_state().unwrap()
                } else {
                    Runner::run(agent.clone(), "Check inventory")
                        .await
                        .unwrap()
                        .to_state()
                };
                let mut state = if serialize_state {
                    RunState::from_json_str(&first.to_json_string().unwrap()).unwrap()
                } else {
                    first
                };
                let interruptions = state.get_interruptions();
                assert_eq!(interruptions.len(), 1, "{ctx}");
                state.reject(&interruptions[0], false, rejection_message.clone());

                let result = if streaming {
                    let mut run = Runner::run_streamed(agent, state);
                    let events = stream_events(&mut run).await;
                    run.finish().await.unwrap();
                    run.final_output_text()
                        .map(|t| (t, streamed_outputs(&events)))
                } else {
                    let r = Runner::run(agent, state).await.unwrap();
                    Some((r.final_output_text().unwrap(), function_outputs(&r)))
                };
                let (final_text, outputs) = result.expect("final output");

                assert!(!executed.load(Ordering::SeqCst), "{ctx}");
                assert_eq!(final_text, "request rejected", "{ctx}");
                assert_eq!(outputs.len(), 1, "{ctx}");
                let expected = rejection_message
                    .clone()
                    .unwrap_or_else(|| "Tool execution was not approved.".to_string());
                assert_eq!(
                    serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
                    json!({"error": expected}),
                    "{ctx}"
                );
                assert_eq!(caller_of(&outputs[0]), program_caller(), "{ctx}");

                let replayed = m.calls().last().unwrap().request.input.clone();
                let replayed_output = replayed
                    .iter()
                    .map(|item| item.0.clone())
                    .find(|v| v["type"] == "function_call_output")
                    .expect("replayed function output");
                assert_eq!(
                    serde_json::from_str::<Value>(replayed_output["output"].as_str().unwrap())
                        .unwrap(),
                    json!({"error": expected}),
                    "{ctx}"
                );
            }
        }
    }
}

/// Drain a streamed run and return its events.
async fn stream_events(
    run: &mut openai_agents_sdk::RunResultStreaming<()>,
) -> Vec<StreamEvent<()>> {
    let mut events = Vec::new();
    let mut stream = Box::pin(run.stream_events());
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    events
}

/// Function outputs among a streamed run's item events.
fn streamed_outputs(events: &[StreamEvent<()>]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::RunItem { item, .. } => match &item.kind {
                RunItemKind::ToolCallOutput { raw_item, .. }
                    if raw_item["type"] == "function_call_output" =>
                {
                    Some(raw_item.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Approvals resumed from a rebuilt state
// ---------------------------------------------------------------------------------------------

/// Pause the run on the approval and return the serialized state, with the approval's raw
/// item and the model's program items as public JSON.
async fn paused_state_json(agent: &Agent<()>) -> Value {
    let first = Runner::run(agent.clone(), "Check inventory").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    first.to_state().to_json().unwrap()
}

#[tokio::test]
async fn rebuilt_mapping_programmatic_approval_preserves_caller() {
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), text("done")],
    ]);
    let executed = Arc::new(AtomicBool::new(false));
    let agent = agent(m, lookup(executed.clone()).with_needs_approval(true));

    let json_state = paused_state_json(&agent).await;
    let mut state = RunState::from_json(json_state).unwrap();
    let approval = state.get_interruptions().remove(0);
    state.approve(&approval, false);

    let result = Runner::run(agent, state).await.unwrap();
    assert!(executed.load(Ordering::SeqCst));
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(caller_of(&outputs[0]), program_caller());
}

#[tokio::test]
async fn rebuilt_mapping_programmatic_approval_rechecks_caller_permissions() {
    let m = model(vec![vec![program_item(), lookup_call()]]);
    let executed = Arc::new(AtomicBool::new(false));
    let agent = agent(
        m.clone(),
        lookup(executed.clone()).with_needs_approval(true),
    );

    let json_state = paused_state_json(&agent).await;
    let mut state = RunState::from_json(json_state).unwrap();
    let approval = state.get_interruptions().remove(0);
    state.approve(&approval, false);

    // The tool now allows direct callers only; the approved program call must be refused.
    let restricted = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(
            lookup(executed.clone())
                .with_allowed_callers(vec!["direct".into()])
                .with_needs_approval(true),
        );
    let err = Runner::run(restricted, state).await.unwrap_err();
    match err {
        AgentsError::ModelBehaviorError { message, .. } => {
            assert!(message.contains("caller programmatic"), "{message}")
        }
        other => panic!("expected ModelBehaviorError, got {other}"),
    }
    assert!(!executed.load(Ordering::SeqCst));
}

/// Remove every object of `"type": "program"` from the state's arrays, as a rebuilt state that
/// lost its parent item.
fn drop_program_items(v: &mut Value) {
    match v {
        Value::Array(items) => {
            items.retain(|i| i.get("type").and_then(Value::as_str) != Some("program"));
            for item in items.iter_mut() {
                drop_program_items(item);
            }
        }
        Value::Object(map) => {
            for value in map.values_mut() {
                drop_program_items(value);
            }
        }
        _ => {}
    }
}

/// Append a completed program output after each array that holds a program item.
fn complete_programs(v: &mut Value) {
    match v {
        Value::Array(items) => {
            let has_program = items
                .iter()
                .any(|i| i.get("type").and_then(Value::as_str) == Some("program"));
            for item in items.iter_mut() {
                complete_programs(item);
            }
            if has_program {
                items.push(json!({
                    "type": "program_output", "id": "program_output_item",
                    "call_id": PROGRAM_CALL_ID, "result": "{}", "status": "completed",
                }));
            }
        }
        Value::Object(map) => {
            for value in map.values_mut() {
                complete_programs(value);
            }
        }
        _ => {}
    }
}

#[tokio::test]
async fn rebuilt_programmatic_approval_requires_active_parent() {
    // Python's two cases: the parent program item is missing, or its output already completed.
    for parent_state in ["missing", "completed"] {
        let m = model(vec![
            vec![program_item(), lookup_call()],
            vec![program_output("completed"), text("done")],
        ]);
        let executed = Arc::new(AtomicBool::new(false));
        let agent = agent(m, lookup(executed.clone()).with_needs_approval(true));

        let mut json_state = paused_state_json(&agent).await;
        if parent_state == "missing" {
            drop_program_items(&mut json_state);
        } else {
            complete_programs(&mut json_state);
        }
        let mut state = RunState::from_json(json_state).unwrap();
        let approval = state.get_interruptions().remove(0);
        state.approve(&approval, false);

        let err = Runner::run(agent, state).await.unwrap_err();
        let expected = if parent_state == "missing" {
            "parent program item"
        } else {
            "already completed"
        };
        match err {
            AgentsError::ModelBehaviorError { message, .. } => {
                assert!(message.contains(expected), "{parent_state}: {message}")
            }
            other => panic!("{parent_state}: expected ModelBehaviorError, got {other}"),
        }
        assert!(!executed.load(Ordering::SeqCst), "{parent_state}");
    }
}
