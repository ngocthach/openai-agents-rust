//! `test_handoffs_lead_to_correct_agent_spans` of `test_tracing_errors.py` and
//! `test_tracing_errors_streamed.py`: a handoff graph with a cycle (`test_agent_1` hands back to
//! `test_agent_3`). An `Arc<Agent>` is immutable, so the back edge is a custom handoff that
//! resolves its destination agent when it is invoked.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::{Arc, OnceLock};

use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::TracingProcessor;
use openai_agents_sdk::{Agent, AgentsError, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use support::{NoApiKey, Pipeline, SpanRecorder, lock_async};

#[derive(Deserialize, JsonSchema, Default)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn some_function() -> FunctionTool<()> {
    FunctionTool::new("some_function", "a tool", |_ctx, _a: AnyArgs| async move {
        Ok("result".to_string())
    })
}

fn tool_call(call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id("some_function", r#"{"a": "b"}"#, call_id)
}

fn handoff_call(agent_name: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(format!("transfer_to_{agent_name}"), "{}", call_id)
}

fn agent_data(name: &str, handoffs: &[&str]) -> Value {
    json!({"name": name, "handoffs": handoffs, "tools": ["some_function"], "output_type": "str"})
}

/// `agent_3 -> [agent_1, agent_2]`, `agent_2 -> [agent_1]`, `agent_1 -> [agent_3]`.
fn cyclic_graph(model: &Arc<ScriptedModel>) -> Arc<Agent<()>> {
    let agent_3_slot: Arc<OnceLock<Arc<Agent<()>>>> = Arc::new(OnceLock::new());
    let slot = agent_3_slot.clone();
    let back_edge = Handoff::custom(
        "transfer_to_test_agent_3",
        "Handoff to the test_agent_3 agent to handle the request. ",
        "test_agent_3",
        json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false}),
        move |_ctx, _args| {
            let slot = slot.clone();
            async move {
                slot.get()
                    .cloned()
                    .ok_or_else(|| AgentsError::other("test_agent_3 is not built yet"))
            }
        },
    );
    let agent_1 = Arc::new(
        Agent::<()>::new("test_agent_1")
            .with_model_instance(model.clone())
            .with_tool(some_function())
            .with_handoff(back_edge),
    );
    let agent_2 = Arc::new(
        Agent::<()>::new("test_agent_2")
            .with_model_instance(model.clone())
            .with_tool(some_function())
            .with_handoff(handoff(agent_1.clone())),
    );
    let agent_3 = Arc::new(
        Agent::<()>::new("test_agent_3")
            .with_model_instance(model.clone())
            .with_tool(some_function())
            .with_handoff(handoff(agent_1))
            .with_handoff(handoff(agent_2)),
    );
    agent_3_slot.set(agent_3.clone()).ok().unwrap();
    agent_3
}

async fn run_cycle(streamed: bool) {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let recorder = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![recorder.clone() as Arc<dyn TracingProcessor>]);

    let model = Arc::new(ScriptedModel::new().with_emit_traces(true));
    let agent_3 = cyclic_graph(&model);
    model
        .extend(vec![
            ScriptedStep::output(vec![tool_call("tool_1")]),
            ScriptedStep::output(vec![
                assistant_message("a_message"),
                handoff_call("test_agent_1", "handoff_1"),
                handoff_call("test_agent_2", "handoff_2"),
            ]),
            ScriptedStep::output(vec![tool_call("tool_2")]),
            ScriptedStep::output(vec![handoff_call("test_agent_3", "handoff_3")]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap();

    let last_agent = if streamed {
        let mut run = Runner::run_streamed(&agent_3, "user_message");
        run.finish().await.expect("run");
        run.into_result()
            .expect("a finished run has a result")
            .last_agent
    } else {
        Runner::run(&agent_3, "user_message")
            .await
            .expect("run")
            .last_agent
    };
    assert_eq!(last_agent.name, "test_agent_3");

    let function = json!({"type": "function", "data": {"name": "some_function",
        "input": r#"{"a": "b"}"#, "output": "result"}});
    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [
                {
                    "type": "agent",
                    "data": agent_data("test_agent_3", &["test_agent_1", "test_agent_2"]),
                    "children": [
                        {"type": "generation"},
                        function.clone(),
                        {"type": "generation"},
                        {
                            "type": "handoff",
                            "data": {"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
                            "error": {
                                "data": {"requested_agents": ["test_agent_1", "test_agent_2"]},
                                "message": "Multiple handoffs requested",
                            },
                        },
                    ],
                },
                {
                    "type": "agent",
                    "data": agent_data("test_agent_1", &["test_agent_3"]),
                    "children": [
                        {"type": "generation"},
                        function,
                        {"type": "generation"},
                        {
                            "type": "handoff",
                            "data": {"from_agent": "test_agent_1", "to_agent": "test_agent_3"},
                        },
                    ],
                },
                {
                    "type": "agent",
                    "data": agent_data("test_agent_3", &["test_agent_1", "test_agent_2"]),
                    "children": [{"type": "generation"}],
                },
            ],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn handoffs_lead_to_correct_agent_spans() {
    run_cycle(false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn streamed_handoffs_lead_to_correct_agent_spans() {
    run_cycle(true).await;
}
