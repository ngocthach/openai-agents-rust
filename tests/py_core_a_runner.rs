//! Ports of the end-to-end runner behaviors in `test_agent_runner.py` (non-streamed half) and
//! `test_call_model_input_filter.py`.
//!
//! Python's agents are often all named `test`; the Rust ports give each agent a distinct name so
//! the generated `transfer_to_<name>` handoff tools stay unambiguous.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::handoffs::HandoffInputData;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolNotFoundBehavior};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
    #[serde(default)]
    bar: String,
}

fn tool_returning<C: Send + Sync + 'static>(name: &str, value: &str) -> FunctionTool<C> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

#[tokio::test]
async fn simple_first_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("first")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    let result = Runner::run(&agent, "test").await.unwrap();
    assert_eq!(result.input.len(), 1);
    assert_eq!(result.input[0].text_content().as_deref(), Some("test"));
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.final_output_text().as_deref(), Some("first"));
    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(result.last_agent.name, "test");
    assert_eq!(result.to_input_list().len(), 2);

    model.add_turn(vec![assistant_message("second")]);
    let result = Runner::run(
        &agent,
        vec![
            InputItem::user("message"),
            InputItem::user("another_message"),
        ],
    )
    .await
    .unwrap();
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.final_output_text().as_deref(), Some("second"));
    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(result.to_input_list().len(), 3);
}

#[tokio::test]
async fn subsequent_runs() {
    let model = setup();
    model.add_turn(vec![assistant_message("third")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    let result = Runner::run(&agent, "test").await.unwrap();
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.to_input_list().len(), 2);

    model.add_turn(vec![assistant_message("fourth")]);
    let result = Runner::run(&agent, result.to_input_list()).await.unwrap();
    assert_eq!(result.input.len(), 2);
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.final_output_text().as_deref(), Some("fourth"));
    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(result.to_input_list().len(), 3);
}

#[tokio::test]
async fn tool_call_runs() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("foo", r#"{"a":"b"}"#),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.raw_responses.len(), 2);
    // original input, message, tool call, tool result, done message
    assert_eq!(result.to_input_list().len(), 5);
}

#[tokio::test]
async fn tool_call_context_includes_current_agent() {
    let model = setup();
    model.add_turn(vec![function_call("foo", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = seen.clone();
    let tool = FunctionTool::new("foo", "f", move |ctx, _a: AnyArgs| {
        let recorder = recorder.clone();
        async move {
            recorder.lock().unwrap().push(ctx.agent_name.clone());
            Ok("tool_result".to_string())
        }
    });
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "user_message").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(seen.lock().unwrap().clone(), vec!["test".to_string()]);
}

#[tokio::test]
async fn handoffs_switch_the_active_agent() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
    let agent_2 = Arc::new(Agent::<()>::new("two").with_model_instance(model.clone()));
    let agent_3 = Agent::<()>::new("three")
        .with_handoff(handoff(agent_1.clone()))
        .with_handoff(handoff(agent_2.clone()))
        .with_tool(tool_returning("some_function", "result"))
        .with_model_instance(model.clone());

    model.add_turn(vec![function_call("some_function", r#"{"a":"b"}"#)]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent_3, "user_message").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.raw_responses.len(), 3);
    // input, tool call, tool result, message, handoff call, handoff output, done message
    assert_eq!(result.to_input_list().len(), 7);
    assert_eq!(result.last_agent.name, "one");
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

#[tokio::test]
async fn structured_output_after_a_handoff() {
    let model = setup();
    let agent_1 = Arc::new(
        Agent::<()>::new("one")
            .with_tool(tool_returning("bar", "bar_result"))
            .with_output_type::<Foo>()
            .with_model_instance(model.clone()),
    );
    let agent_2 = Agent::<()>::new("two")
        .with_tool(tool_returning("foo", "foo_result"))
        .with_handoff(handoff(agent_1.clone()))
        .with_model_instance(model.clone());

    model.add_turn(vec![function_call_with_id(
        "foo",
        r#"{"bar":"baz"}"#,
        "call_foo",
    )]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![
        assistant_message(r#"{"bar":"preamble"}"#),
        function_call_with_id("bar", r#"{"bar":"baz"}"#, "call_bar"),
    ]);
    model.add_turn(vec![assistant_message(r#"{"bar":"baz"}"#)]);

    let result = Runner::run(
        &agent_2,
        vec![
            InputItem::user("user_message"),
            InputItem::user("another_message"),
        ],
    )
    .await
    .unwrap();

    assert_eq!(result.raw_responses.len(), 4);
    assert_eq!(result.last_agent.name, "one");
    assert_eq!(result.final_output, Some(json!({"bar": "baz"})));
    assert_eq!(
        result.final_output_as::<Foo>().unwrap(),
        Foo { bar: "baz".into() }
    );
}

#[tokio::test]
async fn handoff_filters_drop_new_items() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
    let agent_2 = Agent::<()>::new("two")
        .with_handoff(
            handoff(agent_1.clone()).with_input_filter(|data: HandoffInputData| HandoffInputData {
                pre_handoff_items: Vec::new(),
                new_items: Vec::new(),
                ..data
            }),
        )
        .with_model_instance(model.clone());

    model.add_turn(vec![
        assistant_message("1"),
        assistant_message("2"),
        function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![assistant_message("last")]);

    let result = Runner::run(&agent_2, "user_message").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("last"));
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(result.to_input_list().len(), 2);
}

#[tokio::test]
async fn model_settings_override() {
    let model = setup();
    model.add_turn(vec![assistant_message("a_message")]);
    let agent = Agent::<()>::new("test")
        .with_model_settings(ModelSettings {
            temperature: Some(1.0),
            max_tokens: Some(1000),
            ..Default::default()
        })
        .with_model_instance(model.clone());

    let run_config = RunConfig {
        model_settings: Some(ModelSettings {
            temperature: Some(0.5),
            ..Default::default()
        }),
        ..Default::default()
    };
    Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).run_config(run_config),
    )
    .await
    .unwrap();

    let settings = model.last_call().unwrap().model_settings;
    assert_eq!(settings.temperature, Some(0.5));
    assert_eq!(settings.max_tokens, Some(1000));
}

#[tokio::test]
async fn previous_response_id_passed_between_runs() {
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    assert!(model.calls().is_empty());
    Runner::run_with_options(
        &agent,
        "test",
        RunOptions::new(()).previous_response_id("resp-non-streamed-test"),
    )
    .await
    .unwrap();
    assert_eq!(
        model.last_call().unwrap().previous_response_id.as_deref(),
        Some("resp-non-streamed-test")
    );
}

#[tokio::test]
async fn conversation_id_only_sends_new_items_multi_turn() {
    let model = setup();
    model.add_turn(vec![function_call("foo", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());

    Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).conversation_id("conv-test"),
    )
    .await
    .unwrap();

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].conversation_id.as_deref(), Some("conv-test"));
    assert_eq!(calls[0].input.len(), 1);
    // The second turn must only carry the items produced since the first call.
    assert!(
        calls[1].input.len() < 3,
        "second turn resent the whole history: {:?}",
        calls[1].input
    );
}

#[tokio::test]
async fn default_send_all_items_multi_turn() {
    let model = setup();
    model.add_turn(vec![function_call("foo", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());

    Runner::run(&agent, "user_message").await.unwrap();

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].input.len(), 1);
    // input + tool call + tool output
    assert_eq!(calls[1].input.len(), 3);
}

#[tokio::test]
async fn dynamic_tool_addition_run() {
    let model = setup();
    model.add_turn(vec![function_call("add_tool", "{}")]);
    model.add_turn(vec![function_call("added_tool", "{}")]);
    model.add_turn(vec![assistant_message("done")]);

    let agent = Arc::new(Mutex::new(None::<Agent<()>>));
    let base = Agent::<()>::new("test")
        .with_tool(tool_returning("add_tool", "added"))
        .with_tool(tool_returning("added_tool", "result"))
        .with_model_instance(model.clone());
    *agent.lock().unwrap() = Some(base.clone_with(|_| {}));

    let agent = agent.lock().unwrap().clone().unwrap();
    let result = Runner::run(&agent, "run").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn tool_not_found_behavior_returns_error_to_model() {
    let model = setup();
    model.add_turn(vec![function_call("missing", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());

    let run_config = RunConfig {
        tool_not_found_behavior: ToolNotFoundBehavior::ReturnErrorToModel,
        ..Default::default()
    };
    let result = Runner::run_with_options(&agent, "x", RunOptions::new(()).run_config(run_config))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let outputs: Vec<String> = result
        .new_items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .filter_map(|i| i.tool_output().map(|v| v.to_string()))
        .collect();
    assert!(
        outputs.iter().any(|o| o.contains("missing")),
        "tool-not-found message should name the tool: {outputs:?}"
    );
}

#[tokio::test]
async fn tool_not_found_behavior_raises_by_default() {
    let model = setup();
    model.add_turn(vec![function_call("missing", "{}")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "x").await.unwrap_err();
    assert!(err.to_string().contains("missing"), "{err}");
}
