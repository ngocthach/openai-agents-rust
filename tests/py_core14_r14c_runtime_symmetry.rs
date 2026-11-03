//! Ports of the two runtime-symmetry cases of `tests/test_runtime_symmetry_contract.py` that had
//! no Rust counterpart: the scripted model records exactly the model-visible request fields (in
//! Rust, an exhaustive destructure pins the field set at compile time), and a streamed and a
//! non-streamed run fail with the same error.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::testing::{ModelCall, ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, AgentsError, Runner};

fn ready_model() -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("READY")]))
        .unwrap();
    model
}

async fn drain(
    mut streamed: openai_agents_sdk::result::RunResultStreaming<()>,
) -> Option<AgentsError> {
    let mut events = std::pin::pin!(streamed.stream_events());
    while let Some(event) = events.next().await {
        if let Err(error) = event {
            return Some(error);
        }
    }
    None
}

/// Python: `test_scripted_model_records_every_model_visible_request_field`.
#[tokio::test]
async fn the_scripted_model_records_every_model_visible_request_field() {
    for streamed in [false, true] {
        let model = ready_model();
        let agent = Agent::<()>::new("request-contract-agent").with_model_instance(model.clone());
        if streamed {
            assert!(drain(Runner::run_streamed(&agent, "go")).await.is_none());
        } else {
            Runner::run(&agent, "go").await.unwrap();
        }
        let ModelCall {
            request,
            streamed: recorded_streamed,
        } = model.last_call().unwrap();
        assert_eq!(recorded_streamed, streamed);
        // No `..`: adding or removing a model-visible field breaks this destructure.
        let openai_agents_sdk::models::ModelRequest {
            system_instructions: _,
            input,
            model_settings: _,
            tools: _,
            output_schema: _,
            handoffs: _,
            tracing: _,
            previous_response_id: _,
            conversation_id: _,
            prompt: _,
        } = request;
        assert!(!input.is_empty());
    }
}

/// Python: `test_streamed_and_nonstreamed_runs_raise_the_same_exception_class`.
#[tokio::test]
async fn streamed_and_non_streamed_runs_fail_with_the_same_error() {
    let mut failures = Vec::new();
    for streamed in [false, true] {
        let model = Arc::new(ScriptedModel::new());
        model
            .enqueue(ScriptedStep::raise_error(AgentsError::user(
                "release contract failure",
            )))
            .unwrap();
        let agent = Agent::<()>::new("symmetry-agent").with_model_instance(model);
        let error = if streamed {
            drain(Runner::run_streamed(&agent, "go"))
                .await
                .expect("streamed failure")
        } else {
            Runner::run(&agent, "go")
                .await
                .expect_err("non-streamed failure")
        };
        failures.push((std::mem::discriminant(&error), error.to_string()));
    }
    assert_eq!(failures[0].0, failures[1].0);
    assert!(
        failures[0].1.contains("release contract failure"),
        "{}",
        failures[0].1
    );
    assert_eq!(failures[0].1, failures[1].1);
}
