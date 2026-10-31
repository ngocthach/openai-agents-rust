//! Ports of `tests/test_streamed_terminal_output_backfill.py` (a stream that reports its items
//! through `response.output_item.done` and leaves the terminal response's output empty) and
//! `tests/test_streaming_tool_call_arguments.py` (a `tool_called` event carries the final
//! arguments even when the item was first announced with empty ones).

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn response_json(output: &[OutputItem]) -> Value {
    json!({
        "id": "resp_stream",
        "status": "completed",
        "output": output.iter().map(OutputItem::to_value).collect::<Vec<_>>(),
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })
}

/// A stream that finishes `streamed_output` item by item and reports `terminal_output` on the
/// terminal response (Python: `_stream_step`).
fn stream_step(streamed_output: &[OutputItem], terminal_output: &[OutputItem]) -> ScriptedStep {
    let terminal = response_json(terminal_output);
    let mut events = vec![
        json!({"type": "response.created", "response": terminal, "sequence_number": 0}),
        json!({"type": "response.in_progress", "response": terminal, "sequence_number": 1}),
    ];
    for (index, item) in streamed_output.iter().enumerate() {
        events.push(json!({
            "type": "response.output_item.done",
            "item": item.to_value(),
            "output_index": index,
            "sequence_number": 2 + index,
        }));
    }
    events.push(json!({
        "type": "response.completed",
        "response": terminal,
        "sequence_number": 2 + streamed_output.len(),
    }));
    ScriptedStep::stream(events)
}

#[derive(Deserialize, JsonSchema, Default)]
struct ArgA {
    a: String,
}

#[derive(Deserialize, JsonSchema, Default)]
struct ArgB {
    b: String,
}

fn recording_tool_a(name: &str, seen: &Arc<Mutex<Vec<(String, String)>>>) -> FunctionTool<()> {
    let seen = seen.clone();
    let label = name.to_string();
    FunctionTool::new(name, "records its input", move |_ctx, args: ArgA| {
        let (seen, label) = (seen.clone(), label.clone());
        async move {
            seen.lock().unwrap().push((label, args.a));
            Ok("tool_result".to_string())
        }
    })
}

fn recording_tool_b(name: &str, seen: &Arc<Mutex<Vec<(String, String)>>>) -> FunctionTool<()> {
    let seen = seen.clone();
    let label = name.to_string();
    FunctionTool::new(name, "records its input", move |_ctx, args: ArgB| {
        let (seen, label) = (seen.clone(), label.clone());
        async move {
            seen.lock().unwrap().push((label, args.b));
            Ok("tool_result".to_string())
        }
    })
}

async fn drain(result: &mut openai_agents_sdk::result::RunResultStreaming<()>) -> Vec<Value> {
    let mut raw = Vec::new();
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        if let StreamEvent::RawResponse(value) = event.unwrap() {
            raw.push(value);
        }
    }
    raw
}

fn output_types(response: &openai_agents_sdk::items::ModelResponse) -> Vec<String> {
    response
        .output
        .iter()
        .map(|item| item.item_type().to_string())
        .collect()
}

/// Python: `test_streamed_runner_backfills_empty_terminal_output_before_step_resolution`.
#[tokio::test]
async fn streamed_runner_backfills_empty_terminal_output_before_step_resolution() {
    openai_agents_sdk::set_tracing_disabled(true);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let call = function_call_with_id("foo", json!({"a": "b"}).to_string(), "call-1");
    let done = assistant_message("done");
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            stream_step(&[call], &[]),
            stream_step(std::slice::from_ref(&done), std::slice::from_ref(&done)),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(recording_tool_a("foo", &seen));

    let mut result = Runner::run_streamed(agent, "test");
    drain(&mut result).await;

    assert_eq!(
        *seen.lock().unwrap(),
        vec![("foo".to_string(), "b".to_string())]
    );
    let result = result.into_result().unwrap();
    assert_eq!(
        output_types(&result.raw_responses[0]),
        vec!["function_call"]
    );
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

/// Python: `test_streamed_runner_preserves_populated_terminal_output`.
#[tokio::test]
async fn streamed_runner_preserves_populated_terminal_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let call = function_call_with_id("foo", json!({"a": "b"}).to_string(), "call-1");
    let done = assistant_message("done");
    let model = Arc::new(
        ScriptedModel::with_steps(vec![stream_step(&[call], std::slice::from_ref(&done))]).unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(recording_tool_a("foo", &seen));

    let mut result = Runner::run_streamed(agent, "test");
    drain(&mut result).await;

    assert!(seen.lock().unwrap().is_empty());
    let result = result.into_result().unwrap();
    assert_eq!(output_types(&result.raw_responses[0]), vec!["message"]);
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

/// Python: `test_streamed_runner_backfills_multiple_tool_calls_in_order`.
#[tokio::test]
async fn streamed_runner_backfills_multiple_tool_calls_in_order() {
    openai_agents_sdk::set_tracing_disabled(true);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let calls = [
        function_call_with_id("foo", json!({"a": "first"}).to_string(), "call-1"),
        function_call_with_id("bar", json!({"b": "second"}).to_string(), "call-2"),
    ];
    let done = assistant_message("done");
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            stream_step(&calls, &[]),
            stream_step(std::slice::from_ref(&done), std::slice::from_ref(&done)),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(recording_tool_a("foo", &seen))
        .with_tool(recording_tool_b("bar", &seen));

    let mut result = Runner::run_streamed(agent, "test");
    drain(&mut result).await;

    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            ("foo".to_string(), "first".to_string()),
            ("bar".to_string(), "second".to_string())
        ]
    );
    let result = result.into_result().unwrap();
    assert_eq!(
        output_types(&result.raw_responses[0]),
        vec!["function_call", "function_call"]
    );
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

// ---------------------------------------------------------------------------------------------
// tool_called arguments
// ---------------------------------------------------------------------------------------------

/// Announce each function call with empty arguments (as the API does before the deltas), then
/// finish it with the final ones (Python: `_split_argument_step`).
fn split_argument_step(output: &[OutputItem]) -> ScriptedStep {
    let mut events = Vec::new();
    let mut sequence = 0;
    for item in output {
        let mut announced = item.to_value();
        announced["arguments"] = json!("");
        events.push(
            json!({"type": "response.output_item.added", "item": announced,
                           "output_index": 0, "sequence_number": sequence}),
        );
        events.push(
            json!({"type": "response.output_item.done", "item": item.to_value(),
                           "output_index": 0, "sequence_number": sequence + 1}),
        );
        sequence += 2;
    }
    events.push(
        json!({"type": "response.completed", "response": response_json(output),
                       "sequence_number": sequence}),
    );
    ScriptedStep::stream(events)
}

#[derive(Deserialize, JsonSchema, Default)]
struct SumArgs {
    a: i64,
    b: i64,
}

#[derive(Deserialize, JsonSchema, Default)]
struct FormatArgs {
    name: String,
    message: String,
    #[serde(default)]
    urgent: bool,
}

#[derive(Deserialize, JsonSchema, Default)]
struct NoArgs {}

fn calculate_sum() -> FunctionTool<()> {
    FunctionTool::new(
        "calculate_sum",
        "Add two numbers together.",
        |_ctx, a: SumArgs| async move { Ok((a.a + a.b).to_string()) },
    )
}

fn format_message() -> FunctionTool<()> {
    FunctionTool::new(
        "format_message",
        "Format a message with name and urgency.",
        |_ctx, a: FormatArgs| async move {
            let prefix = if a.urgent { "URGENT: " } else { "" };
            Ok(format!("{prefix}Hello {}, {}", a.name, a.message))
        },
    )
}

fn get_current_time() -> FunctionTool<()> {
    FunctionTool::new(
        "get_current_time",
        "Get the current time (no arguments needed).",
        |_ctx, _a: NoArgs| async move { Ok("2024-01-15 10:30:00".to_string()) },
    )
}

/// The raw `arguments` of every `tool_called` event of a streamed run over `step`.
async fn tool_called_arguments(agent: Agent<()>, step: ScriptedStep, input: &str) -> Vec<String> {
    openai_agents_sdk::set_tracing_disabled(true);
    let model =
        Arc::new(ScriptedModel::with_steps(vec![step, ScriptedStep::output(vec![])]).unwrap());
    let agent = agent.with_model_instance(model);
    let mut result = Runner::run_streamed(agent, input);
    let mut arguments = Vec::new();
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        if let StreamEvent::RunItem { name, item } = event.unwrap()
            && name == RunItemStreamEventName::ToolCalled
            && let RunItemKind::ToolCall { raw_item } = &item.kind
        {
            arguments.push(raw_item["arguments"].as_str().unwrap().to_string());
        }
    }
    arguments
}

/// Python: `test_streaming_tool_call_arguments_not_empty`.
#[tokio::test]
async fn streaming_tool_call_arguments_not_empty() {
    let expected = r#"{"a": 5, "b": 3}"#;
    let step = split_argument_step(&[function_call_with_id("calculate_sum", expected, "call_123")]);

    let arguments = tool_called_arguments(
        Agent::<()>::new("TestAgent").with_tool(calculate_sum()),
        step,
        "Add 5 and 3",
    )
    .await;

    assert_eq!(arguments, vec![expected.to_string()]);
    assert_eq!(
        serde_json::from_str::<Value>(&arguments[0]).unwrap(),
        json!({"a": 5, "b": 3})
    );
}

/// Python: `test_streaming_tool_call_arguments_complex`.
#[tokio::test]
async fn streaming_tool_call_arguments_complex() {
    let expected =
        r#"{"name": "Alice", "message": "Your meeting is starting soon", "urgent": true}"#;
    let step = split_argument_step(&[function_call_with_id(
        "format_message",
        expected,
        "call_456",
    )]);

    let arguments = tool_called_arguments(
        Agent::<()>::new("TestAgent").with_tool(format_message()),
        step,
        "Format a message for Alice",
    )
    .await;

    assert_eq!(arguments, vec![expected.to_string()]);
    assert_eq!(
        serde_json::from_str::<Value>(&arguments[0]).unwrap(),
        json!({"name": "Alice", "message": "Your meeting is starting soon", "urgent": true})
    );
}

/// Python: `test_streaming_multiple_tool_calls_arguments`.
#[tokio::test]
async fn streaming_multiple_tool_calls_arguments() {
    let step = split_argument_step(&[
        function_call_with_id("calculate_sum", r#"{"a": 10, "b": 20}"#, "call_1"),
        function_call_with_id(
            "format_message",
            r#"{"name": "Bob", "message": "Test"}"#,
            "call_2",
        ),
    ]);

    let arguments = tool_called_arguments(
        Agent::<()>::new("TestAgent")
            .with_tool(calculate_sum())
            .with_tool(format_message()),
        step,
        "Do some calculations",
    )
    .await;

    assert_eq!(
        arguments,
        vec![
            r#"{"a": 10, "b": 20}"#.to_string(),
            r#"{"name": "Bob", "message": "Test"}"#.to_string()
        ]
    );
}

/// Python: `test_streaming_tool_call_with_empty_arguments`.
#[tokio::test]
async fn streaming_tool_call_with_empty_arguments() {
    let step = split_argument_step(&[function_call_with_id("get_current_time", "{}", "call_time")]);

    let arguments = tool_called_arguments(
        Agent::<()>::new("TestAgent").with_tool(get_current_time()),
        step,
        "What time is it?",
    )
    .await;

    assert_eq!(arguments, vec!["{}".to_string()]);
    assert_eq!(
        serde_json::from_str::<Value>(&arguments[0]).unwrap(),
        json!({})
    );
}
