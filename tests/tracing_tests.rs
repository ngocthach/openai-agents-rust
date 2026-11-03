//! Tracing hierarchy produced by a run (single test: tracing processors are global).

use std::sync::Arc;

use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use openai_agents_sdk::{Agent, Runner};
use serde_json::Value;

#[tokio::test]
async fn run_produces_trace_and_nested_spans() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("ping", "{}")]);
    model.add_turn(vec![assistant_message("pong")]);
    let tool = FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async {
        let span = openai_agents_sdk::custom_span("inside_tool", serde_json::json!({"k": 1}));
        span.run(async {}).await;
        Ok("pong")
    });
    let agent = Agent::<()>::new("Pinger")
        .with_tool(tool)
        .with_model_instance(model.clone());

    trace("Ping workflow")
        .group_id("thread_1")
        .run(async {
            Runner::run(&agent, "ping").await.unwrap();
            model.add_turn(vec![assistant_message("again")]);
            Runner::run(&agent, "again").await.unwrap();
        })
        .await;

    let items = processor.items();
    let traces: Vec<&Value> = items.iter().filter(|i| i["object"] == "trace").collect();
    assert_eq!(traces.len(), 1, "both runs share the outer trace");
    assert_eq!(traces[0]["workflow_name"], "Ping workflow");
    assert_eq!(traces[0]["group_id"], "thread_1");
    let trace_id = traces[0]["id"].as_str().unwrap();
    assert!(trace_id.starts_with("trace_") && trace_id.len() == 38);

    let spans: Vec<&Value> = items
        .iter()
        .filter(|i| i["object"] == "trace.span")
        .collect();
    assert!(spans.iter().all(|s| s["trace_id"] == trace_id));
    let by_type = |t: &str| -> Vec<&Value> {
        spans
            .iter()
            .copied()
            .filter(|s| s["span_data"]["type"] == t)
            .collect()
    };
    let agent_spans = by_type("agent");
    assert_eq!(agent_spans.len(), 2);
    assert_eq!(agent_spans[0]["span_data"]["tools"][0], "ping");
    let function_spans = by_type("function");
    assert_eq!(function_spans.len(), 1);
    assert_eq!(function_spans[0]["span_data"]["output"], "pong");
    let custom: Vec<&Value> = by_type("custom");
    let inside = custom
        .iter()
        .find(|s| s["span_data"]["name"] == "inside_tool")
        .expect("custom span inside the tool");
    assert_eq!(inside["parent_id"], function_spans[0]["id"]);
    let turn = custom
        .iter()
        .find(|s| s["span_data"]["name"] == "turn")
        .expect("turn span");
    assert_eq!(turn["parent_id"], agent_spans[0]["id"]);
    assert_eq!(function_spans[0]["parent_id"], turn["id"]);
    let task = custom
        .iter()
        .find(|s| s["span_data"]["name"] == "task")
        .expect("task span");
    assert_eq!(agent_spans[0]["parent_id"], task["id"]);
    assert!(task["parent_id"].is_null());

    // The OpenAI agent harness id configured on the provider is recorded in trace metadata
    // and routed onto spans.
    model.add_turn(vec![assistant_message("harnessed")]);
    let config = openai_agents_sdk::RunConfig::<()>::new().with_model_provider(Arc::new(
        openai_agents_sdk::OpenAIProvider::new().with_agent_harness_id("harness_1"),
    ));
    Runner::run_with_options(
        &agent,
        "hi",
        openai_agents_sdk::RunOptions::new(()).run_config(config),
    )
    .await
    .unwrap();
    let items = processor.items();
    let last_trace = items.iter().rev().find(|i| i["object"] == "trace").unwrap();
    assert_eq!(last_trace["metadata"]["agent_harness_id"], "harness_1");
    let last_span = items
        .iter()
        .rev()
        .find(|i| i["object"] == "trace.span")
        .unwrap();
    assert_eq!(last_span["metadata"]["agent_harness_id"], "harness_1");
}
