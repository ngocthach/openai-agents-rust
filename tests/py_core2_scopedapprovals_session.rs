//! Port of `tests/test_hitl_session_scenario.py`: three approval round trips against a session
//! that is rehydrated between turns. Python's fake `conversations` client becomes a local mock
//! Conversations endpoint, since Rust's client speaks real HTTP.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{OpenAIConversationsSession, Session};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{
    ModelCall, ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, InputItem, OpenAIClient, Runner, ToolApprovalItem, ToolUseBehavior,
};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, Reply, Request, serve};

const TOOL_ECHO: &str = "approved_echo";
const TOOL_NOTE: &str = "approved_note";
const HITL_REJECTION_MSG: &str = "Tool execution was not approved.";
const USER_MESSAGES: [&str; 3] = [
    "Fetch profile for customer 104.",
    "Update note for customer 104.",
    "Delete note for customer 104.",
];

type Counts = Arc<Mutex<HashMap<&'static str, usize>>>;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Query {
    query: String,
}

fn approval_tool(name: &'static str, prefix: &'static str, counts: &Counts) -> FunctionTool<()> {
    let counts = counts.clone();
    FunctionTool::<()>::new(name, "Runs after approval.", move |_ctx, args: Query| {
        let counts = counts.clone();
        async move {
            *counts.lock().unwrap().entry(name).or_default() += 1;
            Ok(format!("{prefix}:{}", args.query))
        }
    })
    .with_needs_approval(true)
}

fn user_text(item: &Value) -> String {
    match item.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p["type"] == "input_text")
            .filter_map(|p| p["text"].as_str())
            .collect(),
        _ => String::new(),
    }
}

fn output_text(item: &Value) -> String {
    match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(entries)) => entries
            .iter()
            .find(|e| e["type"] == "input_text")
            .and_then(|e| e["text"].as_str())
            .unwrap_or_default()
            .to_string(),
        Some(Value::Object(o)) if o.get("type") == Some(&json!("input_text")) => {
            o["text"].as_str().unwrap_or_default().to_string()
        }
        _ => String::new(),
    }
}

fn input_has_rejection(input: &[InputItem]) -> bool {
    input
        .iter()
        .any(|i| i.0["type"] == "function_call_output" && output_text(&i.0) == HITL_REJECTION_MSG)
}

/// Calls the tool named by `tool_choice` with the latest user message, or answers with the
/// rejection text once a rejection is in the input. Mirrors `make_scenario_model`.
fn scenario_model() -> Arc<ScriptedModel> {
    let counter = Arc::new(Mutex::new(0usize));
    let steps = (0..4)
        .map(|_| {
            let counter = counter.clone();
            ScriptedStep::respond(move |call: &ModelCall| {
                let input = &call.request.input;
                if input_has_rejection(input) {
                    return Ok(ScriptedStep::output(vec![assistant_message(
                        HITL_REJECTION_MSG,
                    )]));
                }
                let tool = match &call.request.model_settings.tool_choice {
                    Some(ToolChoice::Named(name)) => name.clone(),
                    _ => TOOL_ECHO.to_string(),
                };
                let message = input
                    .iter()
                    .rev()
                    .find(|i| i.0["role"] == "user")
                    .map(|i| user_text(&i.0))
                    .unwrap_or_default();
                let mut n = counter.lock().unwrap();
                *n += 1;
                Ok(ScriptedStep::output(vec![function_call_with_id(
                    tool,
                    json!({"query": message}).to_string(),
                    format!("call_{n}"),
                )]))
            })
        })
        .collect();
    Arc::new(ScriptedModel::with_steps(steps).unwrap())
}

struct Step {
    label: &'static str,
    message: &'static str,
    tool_name: &'static str,
    approve: bool,
    expected_output: String,
}

fn steps() -> [Step; 3] {
    [
        Step {
            label: "turn 1",
            message: USER_MESSAGES[0],
            tool_name: TOOL_ECHO,
            approve: true,
            expected_output: format!("approved:{}", USER_MESSAGES[0]),
        },
        Step {
            label: "turn 2 (rehydrated)",
            message: USER_MESSAGES[1],
            tool_name: TOOL_NOTE,
            approve: true,
            expected_output: format!("approved_note:{}", USER_MESSAGES[1]),
        },
        Step {
            label: "turn 3 (rejected)",
            message: USER_MESSAGES[2],
            tool_name: TOOL_ECHO,
            approve: false,
            expected_output: HITL_REJECTION_MSG.to_string(),
        },
    ]
}

/// Run one step: interrupt, decide, resume. Returns the approval item and the session items.
async fn run_step(
    session: Arc<dyn Session>,
    model: &Arc<ScriptedModel>,
    counts: &Counts,
    step: &Step,
) -> (ToolApprovalItem, Vec<InputItem>) {
    let agent = Agent::<()>::new(format!("Scenario {}", step.label))
        .with_instructions(format!("Always call {} before responding.", step.tool_name))
        .with_model_instance(model.clone())
        .with_tool(approval_tool(TOOL_ECHO, "approved", counts))
        .with_tool(approval_tool(TOOL_NOTE, "approved_note", counts))
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::Named(step.tool_name.into())),
            ..Default::default()
        })
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);

    let first = Runner::run_with_options(
        &agent,
        step.message,
        RunOptions::default().session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1, "{}", step.label);
    let approval = first.interruptions[0].clone();
    let mut state = first.to_state();
    if step.approve {
        state.approve(&approval, false);
    } else {
        state.reject(&approval, false, None);
    }
    let resumed = Runner::run_with_options(
        &agent,
        state,
        RunOptions::default().session(session.clone()),
    )
    .await
    .unwrap();
    assert!(resumed.interruptions.is_empty(), "{}", step.label);
    assert_eq!(
        resumed.final_output_text().as_deref(),
        Some(step.expected_output.as_str()),
        "{}",
        step.label
    );
    (approval, session.get_items(None).await.unwrap())
}

fn of_type<'a>(items: &'a [Value], ty: &str) -> Vec<&'a Value> {
    items.iter().filter(|i| i["type"] == ty).collect()
}

fn raw(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.0.clone()).collect()
}

fn assert_counts(items: &[Value], turn: usize) {
    assert_eq!(items.iter().filter(|i| i["role"] == "user").count(), turn);
    assert_eq!(of_type(items, "function_call").len(), turn);
    assert_eq!(of_type(items, "function_call_output").len(), turn);
}

fn assert_step_output(items: &[Value], approval: &ToolApprovalItem, step: &Step) {
    let last_user = items.iter().rev().find(|i| i["role"] == "user").unwrap();
    assert_eq!(user_text(last_user), step.message);
    let last_call = *of_type(items, "function_call").last().unwrap();
    let last_result = *of_type(items, "function_call_output").last().unwrap();
    assert_eq!(last_call["name"], step.tool_name);
    assert_eq!(last_call["call_id"], approval.raw_item.call_id.as_str());
    assert_eq!(last_result["call_id"], approval.raw_item.call_id.as_str());
    assert_eq!(output_text(last_result), step.expected_output);
}

/// Python: `test_memory_session_hitl_scenario`.
#[tokio::test]
async fn memory_session_hitl_scenario() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let counts: Counts = Arc::default();
    let model = scenario_model();
    let steps = steps();

    let session: Arc<dyn Session> = Arc::new(ListSession::new("memory"));
    let (approval, items) = run_step(session, &model, &counts, &steps[0]).await;
    let items = raw(&items);
    assert_counts(&items, 1);
    assert_step_output(&items, &approval, &steps[0]);

    let history: Vec<InputItem> = items.iter().cloned().map(InputItem).collect();
    let rehydrated: Arc<dyn Session> = Arc::new(ListSession::with_history("memory", history));
    let (approval, items) = run_step(rehydrated.clone(), &model, &counts, &steps[1]).await;
    let items = raw(&items);
    assert_counts(&items, 2);
    assert_step_output(&items, &approval, &steps[1]);

    let (approval, items) = run_step(rehydrated.clone(), &model, &counts, &steps[2]).await;
    let items = raw(&items);
    assert_counts(&items, 3);
    assert_step_output(&items, &approval, &steps[2]);

    {
        let counts = counts.lock().unwrap();
        assert_eq!(counts.get(TOOL_ECHO), Some(&1));
        assert_eq!(counts.get(TOOL_NOTE), Some(&1));
    }
    rehydrated.clear_session().await.unwrap();
}

fn assert_step_items(items: &[Value], step: &Step, approval: &ToolApprovalItem) {
    let users: Vec<&Value> = items.iter().filter(|i| i["role"] == "user").collect();
    let calls = of_type(items, "function_call");
    let outputs = of_type(items, "function_call_output");
    assert_eq!(users.len(), 1, "{}: {items:?}", step.label);
    assert_eq!(calls.len(), 1, "{}", step.label);
    assert_eq!(outputs.len(), 1, "{}", step.label);
    assert_eq!(user_text(users[0]), step.message);
    assert_eq!(calls[0]["name"], step.tool_name);
    assert_eq!(calls[0]["call_id"], approval.raw_item.call_id.as_str());
    assert_eq!(outputs[0]["call_id"], approval.raw_item.call_id.as_str());
    assert_eq!(output_text(outputs[0]), step.expected_output);
}

/// Python: `test_openai_conversations_session_hitl_scenario`.
#[tokio::test]
async fn openai_conversations_session_hitl_scenario() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let stored: Arc<Mutex<Vec<Value>>> = Arc::default();
    let store = stored.clone();
    let base = serve(Arc::new(move |req: &Request| {
        if req.method == "POST" && req.path.ends_with("/conversations") {
            return Reply::json(json!({"id": "conv_test"}));
        }
        if req.method == "POST" && req.path.contains("/items") {
            if let Some(items) = req.body.get("items").and_then(Value::as_array) {
                store.lock().unwrap().extend(items.iter().cloned());
            }
            return Reply::json(json!({"data": []}));
        }
        if req.method == "DELETE" {
            return Reply::json(json!({"deleted": true}));
        }
        let mut data = store.lock().unwrap().clone();
        if req.path.contains("order=desc") {
            data.reverse();
        }
        Reply::json(json!({"data": data, "has_more": false}))
    }))
    .await;
    let client = || OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session: Arc<dyn Session> = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client(),
    ));
    let rehydrated: Arc<dyn Session> = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client(),
    ));
    let counts: Counts = Arc::default();
    let model = scenario_model();
    let steps = steps();

    let mut offset = 0;
    let (approval, _) = run_step(session, &model, &counts, &steps[0]).await;
    let snapshot = stored.lock().unwrap().clone();
    assert_step_items(&snapshot[offset..], &steps[0], &approval);
    offset = snapshot.len();

    let (approval, _) = run_step(rehydrated.clone(), &model, &counts, &steps[1]).await;
    let snapshot = stored.lock().unwrap().clone();
    assert_step_items(&snapshot[offset..], &steps[1], &approval);
    offset = snapshot.len();

    let (approval, _) = run_step(rehydrated, &model, &counts, &steps[2]).await;
    let snapshot = stored.lock().unwrap().clone();
    assert_step_items(&snapshot[offset..], &steps[2], &approval);

    let counts = counts.lock().unwrap();
    assert_eq!(counts.get(TOOL_ECHO), Some(&1));
    assert_eq!(counts.get(TOOL_NOTE), Some(&1));
}
