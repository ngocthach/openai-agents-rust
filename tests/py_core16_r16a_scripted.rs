//! Ports of the recorded-call snapshot cases of `tests/test_scripted_model.py`: what the scripted
//! model records is what the caller passed at call time, and the history it hands out is detached.

use futures::StreamExt;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use serde_json::{Value, json};

fn request(model_settings: ModelSettings, input: Vec<InputItem>) -> ModelRequest {
    ModelRequest {
        system_instructions: None,
        input,
        model_settings,
        tools: vec![],
        output_schema: None,
        handoffs: vec![],
        tracing: ModelTracing::Disabled,
        previous_response_id: None,
        conversation_id: None,
        prompt: None,
    }
}

fn settings(mode: &str) -> ModelSettings {
    let mut settings = ModelSettings::default();
    let mut extra = serde_json::Map::new();
    extra.insert("provider".into(), json!({ "mode": mode }));
    settings.extra_args = Some(extra);
    settings
}

fn mode_of(settings: &ModelSettings) -> Value {
    settings.extra_args.as_ref().unwrap()["provider"]["mode"].clone()
}

/// Python: `test_scripted_model_snapshots_model_settings_in_recorded_calls` and
/// `test_scripted_model_snapshots_input_and_prompt_in_recorded_calls`. A caller that keeps
/// editing its own copy of the settings or input after a call does not change the recording of
/// either a non-streamed or a streamed call.
#[tokio::test]
async fn recorded_calls_keep_the_settings_and_input_of_each_call() {
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep::output(vec![assistant_message("first")]),
        ScriptedStep::output(vec![assistant_message("second")]),
    ])
    .unwrap();
    let mut caller_settings = settings("first");
    let mut caller_input = vec![InputItem::user("first")];

    let stream = model
        .stream_response(request(caller_settings.clone(), caller_input.clone()))
        .await
        .unwrap();
    let _: Vec<_> = stream.collect().await;

    caller_settings.extra_args.as_mut().unwrap()["provider"] = json!({ "mode": "second" });
    caller_input[0] = InputItem::user("second");
    model
        .get_response(request(caller_settings.clone(), caller_input.clone()))
        .await
        .unwrap();
    caller_settings.extra_args.as_mut().unwrap()["provider"] = json!({ "mode": "after" });
    caller_input[0] = InputItem::user("after");

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].streamed);
    assert!(!calls[1].streamed);
    assert_eq!(mode_of(&calls[0].request.model_settings), json!("first"));
    assert_eq!(mode_of(&calls[1].request.model_settings), json!("second"));
    assert_eq!(
        serde_json::to_value(&calls[0].request.input).unwrap(),
        serde_json::to_value(vec![InputItem::user("first")]).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&calls[1].request.input).unwrap(),
        serde_json::to_value(vec![InputItem::user("second")]).unwrap()
    );
}

/// Python: `test_scripted_model_exposes_detached_read_only_call_history`. Editing the history a
/// caller was handed does not change what the model retains.
#[tokio::test]
async fn the_call_history_handed_out_is_detached() {
    let model =
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .unwrap();
    model
        .get_response(request(settings("first"), vec![InputItem::user("first")]))
        .await
        .unwrap();

    let mut history = model.calls();
    history[0].request.input.clear();
    history[0]
        .request
        .model_settings
        .extra_args
        .as_mut()
        .unwrap()["provider"] = json!({ "mode": "changed" });
    history.clear();

    let retained = model.calls();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].request.input.len(), 1);
    assert_eq!(mode_of(&retained[0].request.model_settings), json!("first"));
    let last = model.last_call().unwrap();
    assert_eq!(mode_of(&last.request.model_settings), json!("first"));
    assert_eq!(model.first_call().unwrap().request.input.len(), 1);
}
