//! A resumed run keeps the trace settings its snapshot was taken under.
//! Single test: tracing processors are global.

use std::sync::Arc;

use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};

#[tokio::test]
async fn a_resumed_run_reuses_the_workflow_group_and_trace_of_its_snapshot() {
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("ping", "{}")]);
    model.add_turn(vec![assistant_message("pong")]);

    let tool = FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async { Ok("pong") })
        .with_needs_approval(true);
    let agent = Agent::<()>::new("Traced")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let mut config = RunConfig::new()
        .with_workflow_name("Snapshot workflow")
        .with_group_id("grp_resume");
    config.trace_metadata = Some(
        json!({"run": "first"})
            .as_object()
            .cloned()
            .expect("object"),
    );
    let first = Runner::run_with_options(&agent, "ping", RunOptions::new(()).run_config(config))
        .await
        .unwrap();
    assert_eq!(first.interruptions.len(), 1);

    let state = first.to_state();
    let trace_state = state.trace_state.clone().expect("trace state");
    assert_eq!(
        trace_state.workflow_name.as_deref(),
        Some("Snapshot workflow")
    );
    assert_eq!(trace_state.group_id.as_deref(), Some("grp_resume"));
    assert_eq!(
        trace_state.metadata.as_ref().and_then(|m| m.get("run")),
        Some(&json!("first"))
    );
    let trace_id = state.trace_id.clone().expect("trace id");

    // The settings survive the envelope, including the keys Python's `TraceState` reads.
    let wire = state.to_json().unwrap();
    assert_eq!(wire["trace"]["workflow_name"], json!("Snapshot workflow"));
    assert_eq!(wire["trace"]["group_id"], json!("grp_resume"));
    assert_eq!(wire["trace"]["metadata"], json!({"run": "first"}));
    let mut state = RunState::from_json(wire).unwrap();

    // Resume with a default config: the snapshot's trace settings are restored, not replaced
    // by the default workflow name.
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let second = Runner::run(&agent, state).await.unwrap();
    assert_eq!(second.final_output_text().as_deref(), Some("pong"));

    let resumed = second.to_state();
    assert_eq!(resumed.trace_id.as_deref(), Some(trace_id.as_str()));
    let resumed_trace = resumed.trace_state.expect("trace state");
    assert_eq!(
        resumed_trace.workflow_name.as_deref(),
        Some("Snapshot workflow")
    );
    assert_eq!(resumed_trace.group_id.as_deref(), Some("grp_resume"));

    let items = processor.items();
    let traces: Vec<&Value> = items.iter().filter(|i| i["object"] == "trace").collect();
    // The resumed run reattaches instead of starting a second trace.
    assert_eq!(traces.len(), 1, "{traces:?}");
    assert_eq!(traces[0]["id"], json!(trace_id));
    assert_eq!(traces[0]["workflow_name"], json!("Snapshot workflow"));
    assert_eq!(traces[0]["group_id"], json!("grp_resume"));

    let spans: Vec<&Value> = items
        .iter()
        .filter(|i| i["object"] == "trace.span")
        .collect();
    assert!(
        spans.iter().all(|s| s["trace_id"] == json!(trace_id)),
        "a resumed run must keep exporting into the snapshot's trace"
    );
    assert!(
        spans.len() > 1,
        "both the parked and the resumed turn exported spans"
    );
}
