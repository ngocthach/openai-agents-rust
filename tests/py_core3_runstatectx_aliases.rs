//! Port of `tests/test_run_state_aliases.py`: a function call carrying the provider's `async`
//! field keeps it through an approval resume, in every serialization path.
//!
//! Python's pydantic model stores the field as `async_` and writes that name into snapshots, so
//! the json branch rewrites the Rust `async` key to `async_` before reading the snapshot back. That
//! is the shape a Python-written snapshot has, and the Rust reader must map it to the wire name
//! the model expects.
//!
//! Python-only: the `model_fields_set` and `model_extra` assertions check pydantic bookkeeping.
//! Rust keeps unknown provider fields in `FunctionCall::extra`, which the assertions check instead.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::FunctionCall;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunInput, Runner};
use serde_json::{Value, json};

fn alias_call(async_value: Option<bool>) -> FunctionCall {
    let mut call = FunctionCall::new("needs_ok", "{}");
    call.id = Some("fc_alias".into());
    call.call_id = "call_alias".into();
    call.extra.insert(
        "async".into(),
        match async_value {
            None => Value::Null,
            Some(value) => Value::Bool(value),
        },
    );
    call
}

#[derive(Clone, Copy, Debug)]
enum Serialization {
    Live,
    Json,
    String,
}

async fn alias_case(serialization: Serialization, async_value: Option<bool>, approve: bool) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let executions: Arc<Mutex<Vec<String>>> = Arc::default();
    let log = executions.clone();
    let needs_ok = FunctionTool::<()>::new("needs_ok", "Needs approval", move |_c, _a: Value| {
        let log = log.clone();
        async move {
            log.lock().unwrap().push("called".into());
            Ok("approved output".to_string())
        }
    })
    .with_needs_approval(true);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::FunctionCall(alias_call(async_value))]);
    let agent = Agent::<()>::new("Alias approval")
        .with_model_instance(model.clone())
        .with_tools(vec![needs_ok]);
    let config = RunConfig::new();

    let result = Runner::run_with_options(
        &agent,
        "start",
        RunOptions::new(()).run_config(config.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let mut state = result.to_state();
    let state_for_resume = match serialization {
        Serialization::Live => state,
        Serialization::Json => {
            let mut saved = state.to_json().unwrap();
            // Rust writes the provider name; a Python snapshot writes the pydantic field name.
            let raw = &mut saved["current_step"]["data"]["interruptions"][0]["raw_item"];
            assert_eq!(raw["async"], async_value.map_or(Value::Null, Value::Bool));
            let value = raw.as_object_mut().unwrap().remove("async").unwrap();
            raw["async_"] = value;
            saved["$schemaVersion"] = json!("1.18");
            RunState::from_json(saved).unwrap()
        }
        Serialization::String => RunState::from_json_str(&state.to_json_string().unwrap()).unwrap(),
    };
    state = state_for_resume;

    let interruption = state.get_interruptions()[0].clone();
    let raw = &interruption.raw_item;
    assert_eq!(
        raw.extra.get("async"),
        Some(&async_value.map_or(Value::Null, Value::Bool))
    );
    assert!(!raw.extra.contains_key("async_"));
    if approve {
        state.approve(&interruption, false);
    } else {
        state.reject(&interruption, false, None);
    }
    model.add_turn(vec![assistant_message("done")]);
    let resumed = Runner::run_with_options(
        &agent,
        RunInput::from(state),
        RunOptions::new(()).run_config(config),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(resumed.interruptions.is_empty());
    let expected: Vec<String> = if approve {
        vec!["called".into()]
    } else {
        vec![]
    };
    assert_eq!(*executions.lock().unwrap(), expected);
    let restored_call = match &resumed.raw_responses[0].output[0] {
        OutputItem::FunctionCall(call) => call.clone(),
        other => panic!("expected the restored function call, got {other:?}"),
    };
    assert_eq!(
        restored_call.extra.get("async"),
        Some(&async_value.map_or(Value::Null, Value::Bool))
    );
    assert!(!restored_call.extra.contains_key("async_"));
    assert_eq!(model.remaining_steps(), 0);
}

#[tokio::test]
async fn alias_survives_approval_resume_live_approve() {
    alias_case(Serialization::Live, None, true).await;
}

#[tokio::test]
async fn alias_survives_approval_resume_live_reject() {
    alias_case(Serialization::Live, None, false).await;
}

#[tokio::test]
async fn alias_survives_approval_resume_json_approve() {
    alias_case(Serialization::Json, None, true).await;
}

#[tokio::test]
async fn alias_survives_approval_resume_json_reject() {
    alias_case(Serialization::Json, None, false).await;
}

#[tokio::test]
async fn alias_survives_approval_resume_string_approve() {
    alias_case(Serialization::String, None, true).await;
}

#[tokio::test]
async fn alias_survives_approval_resume_string_reject() {
    alias_case(Serialization::String, None, false).await;
}

#[tokio::test]
async fn alias_false_survives_json_round_trip_approve() {
    alias_case(Serialization::Json, Some(false), true).await;
}

#[tokio::test]
async fn alias_false_survives_json_round_trip_reject() {
    alias_case(Serialization::Json, Some(false), false).await;
}

#[tokio::test]
async fn alias_true_survives_string_round_trip_approve() {
    alias_case(Serialization::String, Some(true), true).await;
}

#[tokio::test]
async fn alias_true_survives_string_round_trip_reject() {
    alias_case(Serialization::String, Some(true), false).await;
}
