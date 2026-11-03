//! Leftovers of `test_stream_events.py`: the exact event sequence of a two-turn streamed run,
//! ordering of run-item events around reasoning items, handoff input filters that copy their
//! items, and tool-search items.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::handoffs::{HandoffInputData, nest_handoff_history};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, ItemHelpers, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct FooArgs {
    #[serde(default)]
    arg: String,
}

fn foo_tool() -> FunctionTool<()> {
    FunctionTool::new("foo", "foo", |_ctx, _a: FooArgs| async move {
        Ok("success!".to_string())
    })
}

fn reasoning() -> OutputItem {
    OutputItem::from_value(json!({
        "id": "rid", "type": "reasoning",
        "summary": [{"text": "Thinking", "type": "summary_text"}],
    }))
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    for output in steps {
        model.enqueue(ScriptedStep::output(output)).unwrap();
    }
    model
}

async fn all_events(agent: &Agent<()>, input: &str) -> Vec<StreamEvent<()>> {
    let mut stream = Runner::run_streamed(agent, input);
    let mut out = Vec::new();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        out.push(event.unwrap());
    }
    out
}

fn label(event: &StreamEvent<()>) -> String {
    match event {
        StreamEvent::AgentUpdated { new_agent } => format!("agent:{}", new_agent.name),
        StreamEvent::RawResponse(v) => format!("raw:{}", v["type"].as_str().unwrap_or("?")),
        StreamEvent::RunItem { name, item } => {
            format!("item:{}:{}", name.as_str(), item.item_type())
        }
    }
}

/// `test_complete_streaming_events`: 27 events, in order.
#[tokio::test]
async fn complete_streaming_events() {
    let model = scripted(vec![
        vec![reasoning(), function_call("foo", r#"{"arg": "value"}"#)],
        vec![assistant_message("Final response")],
    ]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(foo_tool())
        .with_model_instance(model);
    let events = all_events(&agent, "Hello").await;
    let labels: Vec<String> = events.iter().map(label).collect();
    let expected = [
        "agent:TestAgent",
        "raw:response.created",
        "raw:response.in_progress",
        "raw:response.output_item.added",
        "raw:response.reasoning_summary_part.added",
        "raw:response.reasoning_summary_text.delta",
        "raw:response.reasoning_summary_text.done",
        "raw:response.reasoning_summary_part.done",
        "raw:response.output_item.done",
        "raw:response.output_item.added",
        "raw:response.function_call_arguments.delta",
        "raw:response.function_call_arguments.done",
        "raw:response.output_item.done",
        "raw:response.completed",
        "item:reasoning_item_created:reasoning_item",
        "item:tool_called:tool_call_item",
        "item:tool_output:tool_call_output_item",
        "raw:response.created",
        "raw:response.in_progress",
        "raw:response.output_item.added",
        "raw:response.content_part.added",
        "raw:response.output_text.delta",
        "raw:response.output_text.done",
        "raw:response.content_part.done",
        "raw:response.output_item.done",
        "raw:response.completed",
        "item:message_output_created:message_output_item",
    ];
    assert_eq!(labels, expected, "{labels:#?}");
}

fn item_event_names(events: &[StreamEvent<()>]) -> Vec<RunItemStreamEventName> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { name, .. } => Some(*name),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn tool_call_event_preserves_order_before_later_reasoning_item() {
    let model = scripted(vec![
        vec![function_call("foo", r#"{"arg": "value"}"#), reasoning()],
        vec![assistant_message("Final response")],
    ]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(foo_tool())
        .with_model_instance(model);
    let names = item_event_names(&all_events(&agent, "Hello").await);
    assert_eq!(
        names[..3],
        [
            RunItemStreamEventName::ToolCalled,
            RunItemStreamEventName::ReasoningItemCreated,
            RunItemStreamEventName::ToolOutput,
        ]
    );
}

#[tokio::test]
async fn handoff_event_preserves_order_before_later_reasoning_item() {
    let english = Arc::new(
        Agent::<()>::new("EnglishAgent")
            .with_model_instance(scripted(vec![vec![assistant_message("Done")]])),
    );
    let model = scripted(vec![vec![
        function_call("transfer_to_englishagent", "{}"),
        reasoning(),
    ]]);
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(handoff(english))
        .with_model_instance(model);
    let names = item_event_names(&all_events(&triage, "Start").await);
    assert_eq!(
        names[..2],
        [
            RunItemStreamEventName::HandoffRequested,
            RunItemStreamEventName::ReasoningItemCreated,
        ]
    );
}

#[tokio::test]
async fn handoff_filter_copy_does_not_duplicate_streamed_model_items() {
    let english = Arc::new(
        Agent::<()>::new("EnglishAgent")
            .with_model_instance(scripted(vec![vec![assistant_message("Done")]])),
    );
    let model = scripted(vec![vec![
        assistant_message("Transferring"),
        function_call("transfer_to_englishagent", "{}"),
    ]]);
    let copying_filter = |data: HandoffInputData| {
        let nested = nest_handoff_history(data, None);
        HandoffInputData {
            new_items: nested.new_items.clone(),
            ..nested
        }
    };
    let triage = Agent::<()>::new("TriageAgent")
        .with_handoff(handoff(english).with_input_filter(copying_filter))
        .with_model_instance(model);
    let events = all_events(&triage, "Start").await;

    let texts: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { item, .. } if item.as_message().is_some() => {
                Some(ItemHelpers::text_message_output(item))
            }
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["Transferring", "Done"]);
    let requested = item_event_names(&events)
        .iter()
        .filter(|n| **n == RunItemStreamEventName::HandoffRequested)
        .count();
    assert_eq!(requested, 1);
}

#[tokio::test]
async fn stream_events_emit_tool_search_items() {
    let call = OutputItem::from_value(json!({
        "id": "tsc_stream", "type": "tool_search_call",
        "arguments": {"paths": ["crm"], "query": "orders"},
        "execution": "server", "status": "completed",
    }));
    let output = OutputItem::from_value(json!({
        "id": "tso_stream", "type": "tool_search_output",
        "execution": "server", "status": "completed",
        "tools": [{
            "type": "function", "name": "list_open_orders",
            "description": "List open orders for a customer.",
            "parameters": {"type": "object",
                           "properties": {"customer_id": {"type": "string"}},
                           "required": ["customer_id"]},
            "defer_loading": true,
        }],
    }));
    let model = scripted(vec![vec![call, output, assistant_message("Done")]]);
    let agent = Agent::<()>::new("ToolSearchAgent").with_model_instance(model);
    let events = all_events(&agent, "Search for CRM order tools").await;
    let seen: Vec<(RunItemStreamEventName, Value)> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { name, item } => Some((*name, json!(item.item_type()))),
            _ => None,
        })
        .collect();
    assert!(
        seen.iter()
            .any(|(n, _)| *n == RunItemStreamEventName::ToolSearchCalled),
        "{seen:?}"
    );
    assert!(
        seen.iter()
            .any(|(n, _)| *n == RunItemStreamEventName::ToolSearchOutputCreated),
        "{seen:?}"
    );
}
