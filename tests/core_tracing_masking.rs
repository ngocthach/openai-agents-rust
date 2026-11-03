//! A run with tracing turned off must not export spans into an enabled outer trace.
//! Single test: tracing processors are global.

use std::sync::Arc;

use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};
use serde_json::Value;

#[tokio::test]
async fn a_disabled_run_exports_nothing_into_an_enabled_outer_trace() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let model = Arc::new(FakeModel::new());
    // The disabled run: a tool call plus an answer, so it would produce agent, generation,
    // function and tool spans if it were exporting.
    model.add_turn(vec![function_call("ping", "{}")]);
    model.add_turn(vec![assistant_message("pong")]);
    // The enabled run that follows.
    model.add_turn(vec![assistant_message("visible")]);

    let tool = FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async { Ok("pong") });
    let agent = Agent::<()>::new("Masked")
        .with_tool(tool)
        .with_model_instance(model.clone());

    trace("Outer workflow")
        .run(async {
            let options =
                RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true));
            Runner::run_with_options(&agent, "ping", options)
                .await
                .unwrap();
            Runner::run(&agent, "again").await.unwrap();
        })
        .await;

    let items = processor.items();
    let traces: Vec<&Value> = items.iter().filter(|i| i["object"] == "trace").collect();
    assert_eq!(traces.len(), 1, "only the outer trace is exported");
    let outer_trace_id = traces[0]["id"].as_str().unwrap();

    let spans: Vec<&Value> = items
        .iter()
        .filter(|i| i["object"] == "trace.span")
        .collect();
    assert!(
        spans.iter().all(|s| s["trace_id"] == outer_trace_id),
        "a disabled run must not export spans under any other trace"
    );

    // Only the second (enabled) run exported, so exactly one agent span exists and it belongs
    // to the outer trace with no parent inside the masked run.
    let agent_spans: Vec<&Value> = spans
        .iter()
        .copied()
        .filter(|s| s["span_data"]["type"] == "agent")
        .collect();
    assert_eq!(
        agent_spans.len(),
        1,
        "the disabled run contributed no agent span"
    );
    // One task span, from the enabled run only, and the agent span hangs under it rather
    // than under anything the masked run opened.
    let task_spans: Vec<&Value> = spans
        .iter()
        .copied()
        .filter(|s| s["span_data"]["data"]["sdk_span_type"] == "task")
        .collect();
    assert_eq!(
        task_spans.len(),
        1,
        "the disabled run exported no task span"
    );
    assert_eq!(agent_spans[0]["parent_id"], task_spans[0]["id"]);
    assert!(
        !spans
            .iter()
            .any(|s| s["span_data"]["name"] == openai_agents_sdk::tracing::MASKED_RUN_SPAN_NAME),
        "the mask itself is disabled and never exported"
    );

    // The tool the disabled run called left no function span behind either.
    assert!(
        !spans
            .iter()
            .any(|s| s["span_data"]["type"] == "function" && s["span_data"]["name"] == "ping"),
        "the disabled run's tool span must not be exported"
    );
}
