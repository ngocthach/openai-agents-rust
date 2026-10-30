//! Resuming an interrupted turn inside a server-managed conversation must deliver every tool
//! output exactly once: the outputs built before the pause are still unsent, while the calls
//! the provider already holds are not replayed.

use std::sync::Arc;

use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    city: String,
}

fn weather_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "get_weather",
        "Get the weather",
        |_ctx, a: CityArgs| async move { Ok(format!("sunny in {}", a.city)) },
    )
    // Only the gated city pauses, so the other call runs and produces an unsent output.
    .with_needs_approval_fn(|_ctx, args, _id| async move {
        args["city"].as_str() == Some("Oakland")
    })
}

#[tokio::test]
async fn a_resumed_conversation_run_sends_every_tool_output_once() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        function_call_with_id("get_weather", r#"{"city":"Tokyo"}"#, "call_open"),
        function_call_with_id("get_weather", r#"{"city":"Oakland"}"#, "call_gated"),
    ]);
    let agent = Agent::<()>::new("W")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());

    let paused = Runner::run_with_options(
        &agent,
        "weather",
        RunOptions::new(()).conversation_id("conv_test"),
    )
    .await
    .unwrap();
    assert_eq!(paused.interruptions.len(), 1, "the gated call paused");
    assert_eq!(
        paused
            .new_items
            .iter()
            .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
            .count(),
        1,
        "the ungated call produced an output before the pause"
    );

    // Round-trip the state through JSON, the way a server-side checkpoint would.
    let state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let mut state = state;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);

    model.add_turn(vec![assistant_message("done")]);
    let resumed = Runner::run_with_options(
        &agent,
        state,
        RunOptions::new(()).conversation_id("conv_test"),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));

    let last = model.last_call().unwrap();
    let types: Vec<Option<&str>> = last.input.iter().map(|i| i.item_type()).collect();
    assert_eq!(
        types
            .iter()
            .filter(|t| **t == Some("function_call_output"))
            .count(),
        2,
        "both tool outputs reach the server: {types:?}"
    );
    assert!(
        !types.contains(&Some("function_call")),
        "the provider already holds the calls it produced: {types:?}"
    );
    let outputs: Vec<&str> = last
        .input
        .iter()
        .filter(|i| i.item_type() == Some("function_call_output"))
        .filter_map(|i| i.as_value().get("call_id").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(outputs, vec!["call_open", "call_gated"]);
}
