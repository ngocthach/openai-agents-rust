//! Ports of `test_stream_events.py` and `test_streaming_tool_call_arguments.py`.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::handoffs::HandoffInputData;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct FooArgs {
    #[serde(default)]
    args: String,
    #[serde(default)]
    a: String,
}

fn foo_tool() -> FunctionTool<()> {
    FunctionTool::new("foo", "foo", |_ctx, _a: FooArgs| async move {
        Ok("success!".to_string())
    })
}

/// Collect every run-item event (name, item type) plus the agent names that were announced.
async fn collect(
    stream: &mut openai_agents_sdk::result::RunResultStreaming<()>,
) -> (Vec<(RunItemStreamEventName, String)>, Vec<String>) {
    let mut items = Vec::new();
    let mut agents = Vec::new();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        match event.unwrap() {
            StreamEvent::RunItem { name, item } => items.push((name, item.item_type().to_string())),
            StreamEvent::AgentUpdated { new_agent } => agents.push(new_agent.name.clone()),
            StreamEvent::RawResponse(_) => {}
        }
    }
    (items, agents)
}

#[tokio::test]
async fn stream_events_main() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("foo", ""),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Joker")
        .with_tool(foo_tool())
        .with_model_instance(model.clone());

    let mut stream = Runner::run_streamed(&agent, "Hello");
    let (items, _) = collect(&mut stream).await;
    let names: Vec<RunItemStreamEventName> = items.iter().map(|(n, _)| *n).collect();

    let tool_called = names
        .iter()
        .position(|n| *n == RunItemStreamEventName::ToolCalled)
        .expect("tool_called event");
    let tool_output = names
        .iter()
        .position(|n| *n == RunItemStreamEventName::ToolOutput)
        .expect("tool_output event");
    assert!(
        tool_called < tool_output,
        "tool_called must precede tool_output: {names:?}"
    );
    assert_eq!(
        names
            .iter()
            .filter(|n| **n == RunItemStreamEventName::MessageOutputCreated)
            .count(),
        2
    );
}

#[tokio::test]
async fn stream_events_main_with_handoff() {
    let model = setup();
    let english = Arc::new(Agent::<()>::new("EnglishAgent").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(
            handoff(english.clone()).with_input_filter(|data: HandoffInputData| HandoffInputData {
                pre_handoff_items: Vec::new(),
                new_items: Vec::new(),
                ..data
            }),
        )
        .with_tool(foo_tool())
        .with_model_instance(model.clone());

    model.add_turn(vec![
        assistant_message("Hello"),
        function_call("foo", r#"{"args": "arg1"}"#),
        function_call("transfer_to_englishagent", "{}"),
    ]);
    model.add_turn(vec![assistant_message("Done")]);

    let mut stream = Runner::run_streamed(&triage, "Start");
    let (items, agents) = collect(&mut stream).await;

    assert!(
        items
            .iter()
            .any(|(name, _)| *name == RunItemStreamEventName::HandoffRequested),
        "handoff_requested event not observed: {items:?}"
    );
    assert!(
        agents.iter().any(|name| name == "EnglishAgent"),
        "agent never switched to EnglishAgent: {agents:?}"
    );
}

#[tokio::test]
async fn streamed_handoff_call_is_not_emitted_as_tool_called() {
    let model = setup();
    let english = Arc::new(Agent::<()>::new("EnglishAgent").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(handoff(english.clone()))
        .with_model_instance(model.clone());

    model.add_turn(vec![function_call("transfer_to_englishagent", "{}")]);
    model.add_turn(vec![assistant_message("Done")]);

    let mut stream = Runner::run_streamed(&triage, "Start");
    let (items, _) = collect(&mut stream).await;

    let handoffs: Vec<_> = items
        .iter()
        .filter(|(_, ty)| ty == "handoff_call_item")
        .collect();
    assert_eq!(handoffs.len(), 1);
    assert_eq!(handoffs[0].0, RunItemStreamEventName::HandoffRequested);
    assert!(
        !items
            .iter()
            .any(|(name, _)| *name == RunItemStreamEventName::ToolCalled),
        "handoff call must not also emit tool_called: {items:?}"
    );
}

#[tokio::test]
async fn streamed_tool_call_alongside_handoff_still_emits_tool_called() {
    let model = setup();
    let english = Arc::new(Agent::<()>::new("EnglishAgent").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(
            handoff(english.clone()).with_input_filter(|data: HandoffInputData| HandoffInputData {
                pre_handoff_items: Vec::new(),
                new_items: Vec::new(),
                ..data
            }),
        )
        .with_tool(foo_tool())
        .with_model_instance(model.clone());

    model.add_turn(vec![
        function_call_with_id("foo", r#"{"a": "b"}"#, "tool_call"),
        function_call("transfer_to_englishagent", "{}"),
    ]);
    model.add_turn(vec![assistant_message("Done")]);

    let mut stream = Runner::run_streamed(&triage, "Start");
    let mut tool_called_ids = Vec::new();
    let mut handoff_names = Vec::new();
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { name, item } = event.unwrap() {
                if name == RunItemStreamEventName::ToolCalled
                    && let RunItemKind::ToolCall { raw_item } = &item.kind
                {
                    tool_called_ids.push(
                        raw_item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    );
                }
                if item.item_type() == "handoff_call_item" {
                    handoff_names.push(name);
                }
            }
        }
    }
    assert_eq!(tool_called_ids, vec!["tool_call".to_string()]);
    assert_eq!(
        handoff_names,
        vec![RunItemStreamEventName::HandoffRequested]
    );
}

#[tokio::test]
async fn streamed_handoff_item_events_match_new_items() {
    let model = setup();
    let english = Arc::new(Agent::<()>::new("EnglishAgent").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(handoff(english.clone()))
        .with_model_instance(model.clone());

    model.add_turn(vec![
        assistant_message("Transferring"),
        function_call("transfer_to_englishagent", "{}"),
    ]);
    model.add_turn(vec![assistant_message("Done")]);

    let mut stream = Runner::run_streamed(&triage, "Start");
    let (items, _) = collect(&mut stream).await;
    let mut streamed: Vec<String> = items.into_iter().map(|(_, ty)| ty).collect();
    streamed.sort();

    let result = stream.into_result().expect("run completed");
    let mut recorded: Vec<String> = result
        .new_items
        .iter()
        .map(|i| i.item_type().to_string())
        .collect();
    recorded.sort();
    assert_eq!(streamed, recorded);
}

#[tokio::test]
async fn tool_call_arguments_are_streamed_as_raw_events() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    // `ScriptedModel` is the Rust equivalent of Python's `agents.testing.ScriptedModel`: it
    // synthesizes the full Responses event sequence, including the argument deltas.
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call("foo", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("A")
        .with_tool(foo_tool())
        .with_model_instance(model.clone());

    let mut stream = Runner::run_streamed(&agent, "x");
    let mut argument_deltas = String::new();
    let mut saw_done = false;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RawResponse(value) = event.unwrap() {
                match value.get("type").and_then(|v| v.as_str()) {
                    Some("response.function_call_arguments.delta") => {
                        argument_deltas
                            .push_str(value.get("delta").and_then(|v| v.as_str()).unwrap_or(""));
                    }
                    Some("response.function_call_arguments.done") => saw_done = true,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(argument_deltas, r#"{"a":"b"}"#);
    assert!(saw_done, "a function_call_arguments.done event is emitted");
}
