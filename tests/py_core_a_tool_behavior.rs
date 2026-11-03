//! Ports of `test_tool_use_behavior.py`, `test_tool_choice_reset.py` and `test_output_tool.py`.
//!
//! Python reaches into `run_loop.check_for_final_output_from_tools` and
//! `run_loop.maybe_reset_tool_choice`; `run_internal` is private in Rust, so the same behavior is
//! exercised end to end through `Runner`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn echo_tool<C: Send + Sync + 'static>(name: &str, value: &str) -> FunctionTool<C> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: Empty| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

// ------------------------------------------------------------ tool_use_behavior

#[tokio::test]
async fn run_llm_again_behavior() {
    let model = setup();
    model.add_turn(vec![function_call("tool1", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "ignored"))
        .with_tool_use_behavior(ToolUseBehavior::RunLlmAgain)
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn stop_on_first_tool_behavior() {
    let model = setup();
    model.add_turn(vec![
        function_call("tool1", "{}"),
        function_call("tool2", "{}"),
    ]);
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "first_tool_output"))
        .with_tool(echo_tool("tool2", "ignored"))
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("first_tool_output")
    );
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn custom_tool_use_behavior_is_awaited_once() {
    let model = setup();
    model.add_turn(vec![
        function_call("tool1", "{}"),
        function_call("tool2", "{}"),
        function_call("tool3", "{}"),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "ignored1"))
        .with_tool(echo_tool("tool2", "ignored2"))
        .with_tool(echo_tool("tool3", "ignored3"))
        .with_tool_use_behavior(ToolUseBehavior::custom(move |_ctx, results| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                assert_eq!(results.len(), 3);
                Ok(ToolsToFinalOutputResult {
                    is_final_output: true,
                    final_output: Some(json!("custom")),
                })
            }
        }))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("custom"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn custom_tool_use_behavior_can_decline_to_finish() {
    let model = setup();
    model.add_turn(vec![function_call("tool1", "{}")]);
    model.add_turn(vec![assistant_message("llm answer")]);
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "pong"))
        .with_tool_use_behavior(ToolUseBehavior::custom(|_ctx, results| async move {
            assert_eq!(results.len(), 1);
            Ok(ToolsToFinalOutputResult::not_final())
        }))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("llm answer"));
}

#[tokio::test]
async fn custom_tool_use_behavior_errors_propagate() {
    let model = setup();
    model.add_turn(vec![function_call("tool1", "{}")]);
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "pong"))
        .with_tool_use_behavior(ToolUseBehavior::custom(|_ctx, _results| async move {
            Err(openai_agents_sdk::AgentsError::user(
                "tool_use_behavior callable must return ToolsToFinalOutputResult",
            ))
        }))
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "x").await.unwrap_err();
    assert!(
        err.to_string()
            .contains("tool_use_behavior callable must return ToolsToFinalOutputResult"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn tool_names_to_stop_at_behavior() {
    // The named tool is not called: the run continues.
    let model = setup();
    model.add_turn(vec![
        function_call("tool2", "{}"),
        function_call("tool3", "{}"),
    ]);
    model.add_turn(vec![assistant_message("kept going")]);
    let agent = Agent::<()>::new("test")
        .with_tool(echo_tool("tool1", "tool1_output"))
        .with_tool(echo_tool("tool2", "tool2_output"))
        .with_tool(echo_tool("tool3", "tool3_output"))
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["tool1".into()],
        }))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("kept going"));

    // The named tool is called: its output is the final output.
    let model = setup();
    model.add_turn(vec![
        function_call("tool1", "{}"),
        function_call("tool2", "{}"),
        function_call("tool3", "{}"),
    ]);
    let agent = agent.clone_with(|a| a.model = Some(model.clone().into()));
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("tool1_output"));
    assert_eq!(model.calls().len(), 1);
}

// ----------------------------------------------------------- tool choice reset

#[tokio::test]
async fn required_tool_choice_with_multiple_runs() {
    let model = setup();
    model.add_turn(vec![assistant_message("First run response")]);
    model.add_turn(vec![assistant_message("Second run response")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(echo_tool("custom_tool", "ok"))
        .with_model_settings(ModelSettings::default().with_tool_choice(ToolChoice::required()))
        .with_model_instance(model.clone());

    let result1 = Runner::run(&agent, "first run").await.unwrap();
    assert_eq!(
        result1.final_output_text().as_deref(),
        Some("First run response")
    );
    assert_eq!(
        model.last_call().unwrap().model_settings.tool_choice,
        Some(ToolChoice::required())
    );

    let result2 = Runner::run(&agent, "second run").await.unwrap();
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second run response")
    );
    assert_eq!(
        model.last_call().unwrap().model_settings.tool_choice,
        Some(ToolChoice::required())
    );
}

#[tokio::test]
async fn required_with_stop_at_tool_name() {
    let model = setup();
    model.add_turn(vec![function_call("second_tool", "{}")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(echo_tool("first_tool", "first tool result"))
        .with_tool(echo_tool("second_tool", "second tool result"))
        .with_model_settings(ModelSettings::default().with_tool_choice(ToolChoice::required()))
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["second_tool".into()],
        }))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "run test").await.unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("second tool result")
    );
}

#[tokio::test]
async fn specific_tool_choice_does_not_loop() {
    let model = setup();
    model.add_turn(vec![assistant_message("Test message")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(echo_tool("tool1", "a"))
        .with_tool(echo_tool("tool2", "b"))
        .with_tool(echo_tool("tool3", "c"))
        .with_model_settings(
            ModelSettings::default().with_tool_choice(ToolChoice::Named("tool1".into())),
        )
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "first run").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Test message"));
}

#[tokio::test]
async fn required_with_single_tool_resets_after_use() {
    let model = setup();
    model.add_turn(vec![function_call("custom_tool", "{}")]);
    model.add_turn(vec![assistant_message("Final response")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(echo_tool("custom_tool", "tool result"))
        .with_model_settings(ModelSettings::default().with_tool_choice(ToolChoice::required()))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "first run").await.unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("Final response")
    );
    // The second model call must no longer force a tool, or the run would loop forever.
    assert_eq!(model.last_call().unwrap().model_settings.tool_choice, None);
}

#[tokio::test]
async fn dont_reset_tool_choice_if_not_required() {
    let model = setup();
    model.add_turn(vec![function_call("custom_tool", "{}")]);
    model.add_turn(vec![assistant_message("Final response")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(echo_tool("custom_tool", "tool result"))
        .with_model_settings(ModelSettings::default().with_tool_choice(ToolChoice::required()))
        .with_reset_tool_choice(false)
        .with_model_instance(model.clone());
    Runner::run(&agent, "test").await.unwrap();
    assert_eq!(
        model.last_call().unwrap().model_settings.tool_choice,
        Some(ToolChoice::required())
    );
}

#[tokio::test]
async fn tool_use_on_a_different_agent_does_not_reset_the_choice() {
    // Python's case 6: a handoff target keeps its own `tool_choice` even though the first agent
    // already called a tool.
    let model = setup();
    model.add_turn(vec![function_call("custom_tool", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("first")
        .with_tool(echo_tool("custom_tool", "tool result"))
        .with_model_settings(
            ModelSettings::default().with_tool_choice(ToolChoice::Named("foo_bar".into())),
        )
        .with_model_instance(model.clone());
    Runner::run(&agent, "x").await.unwrap();
    // Its own tool use resets the choice.
    assert_eq!(model.last_call().unwrap().model_settings.tool_choice, None);

    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let untouched = Agent::<()>::new("second")
        .with_tool(echo_tool("custom_tool", "tool result"))
        .with_model_settings(
            ModelSettings::default().with_tool_choice(ToolChoice::Named("foo_bar".into())),
        )
        .with_model_instance(model.clone());
    Runner::run(&untouched, "x").await.unwrap();
    assert_eq!(
        model.last_call().unwrap().model_settings.tool_choice,
        Some(ToolChoice::Named("foo_bar".into()))
    );
}

// -------------------------------------------------------------- output schema

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
#[allow(dead_code)]
struct Foo {
    bar: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct StrictOutput {
    name: String,
    age: i64,
}

#[test]
fn plain_text_output_has_no_schema() {
    let agent = Agent::<()>::new("test");
    assert!(agent.output_type.is_none());
}

#[test]
fn structured_output_is_not_wrapped() {
    let schema = AgentOutputSchema::new::<Foo>().unwrap();
    assert!(!schema.is_wrapped());
    assert!(!schema.is_plain_text());
    let validated = schema.validate_json(r#"{"bar":"baz"}"#).unwrap();
    assert_eq!(validated, json!({"bar": "baz"}));
}

#[test]
fn structured_output_list_is_wrapped() {
    let schema = AgentOutputSchema::new::<Vec<String>>().unwrap();
    assert!(schema.is_wrapped());
    let validated = schema
        .validate_json(&json!({"response": ["foo", "bar"]}).to_string())
        .unwrap();
    assert_eq!(validated, json!(["foo", "bar"]));
}

#[test]
fn wrapped_output_rejects_a_missing_wrapper_key() {
    let schema = AgentOutputSchema::new::<Vec<String>>().unwrap();
    let err = schema.validate_json(r#"["foo"]"#).unwrap_err();
    assert!(err.to_string().contains("Expected a dict"), "{err}");
    let err = schema.validate_json(r#"{"value":"foo"}"#).unwrap_err();
    assert!(err.to_string().contains("response"), "{err}");
}

#[test]
fn bad_json_raises_model_behavior_error() {
    let schema = AgentOutputSchema::new::<Foo>().unwrap();
    let err = schema.validate_json("not valid json").unwrap_err();
    assert!(
        matches!(
            err,
            openai_agents_sdk::AgentsError::ModelBehaviorError { .. }
        ),
        "unexpected error kind: {err:?}"
    );
}

#[test]
fn structured_output_is_strict_by_default() {
    let schema = AgentOutputSchema::new::<Foo>().unwrap();
    assert!(schema.is_strict_json_schema());
    let js = schema.json_schema().unwrap();
    assert_eq!(js["additionalProperties"], json!(false));
    assert_eq!(js["required"], json!(["bar"]));
}

#[test]
fn setting_strict_false_keeps_the_raw_schema() {
    let schema = AgentOutputSchema::new_with_strict::<Foo>(false).unwrap();
    assert!(!schema.is_strict_json_schema());
    let js = schema.json_schema().unwrap();
    assert!(js.get("additionalProperties").is_none());
}

#[test]
fn agent_output_schema_rejects_type_coercion() {
    let schema = AgentOutputSchema::new::<StrictOutput>().unwrap();
    assert!(schema.is_strict_json_schema());
    let err = schema
        .validate_json(r#"{"name": "Alice", "age": "25"}"#)
        .unwrap_err();
    assert!(err.to_string().contains("Invalid JSON"), "{err}");

    let ok = schema
        .validate_json(r#"{"name": "Alice", "age": 25}"#)
        .unwrap();
    assert_eq!(ok["name"], json!("Alice"));
    assert_eq!(ok["age"], json!(25));
}

#[derive(Debug)]
struct CustomOutputSchema;

const CUSTOM_OUTPUT_SCHEMA: &str =
    r#"{"type":"object","properties":{"foo":{"type":"string"}},"required":["foo"]}"#;

impl AgentOutputSchemaBase for CustomOutputSchema {
    fn is_plain_text(&self) -> bool {
        false
    }
    fn name(&self) -> String {
        "FooBarBaz".into()
    }
    fn json_schema(&self) -> openai_agents_sdk::Result<Value> {
        Ok(serde_json::from_str(CUSTOM_OUTPUT_SCHEMA).unwrap())
    }
    fn is_strict_json_schema(&self) -> bool {
        false
    }
    fn validate_json(&self, _json_str: &str) -> openai_agents_sdk::Result<Value> {
        Ok(json!(["some", "output"]))
    }
}

#[tokio::test]
async fn custom_output_schema_is_used_by_the_agent() {
    let model = setup();
    model.add_turn(vec![assistant_message(r#"{"foo":"bar"}"#)]);
    let schema: Arc<dyn AgentOutputSchemaBase> = Arc::new(CustomOutputSchema);
    let agent = Agent::<()>::new("test")
        .with_output_schema(schema.clone())
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output, Some(json!(["some", "output"])));

    assert_eq!(schema.name(), "FooBarBaz");
    assert!(!schema.is_strict_json_schema());
    assert!(!schema.is_plain_text());
    assert_eq!(
        schema.json_schema().unwrap(),
        serde_json::from_str::<Value>(CUSTOM_OUTPUT_SCHEMA).unwrap()
    );
}

#[tokio::test]
async fn the_output_schema_reaches_the_model_request() {
    let model = setup();
    model.add_turn(vec![assistant_message(r#"{"bar":"baz"}"#)]);
    let agent = Agent::<()>::new("test")
        .with_output_type::<Foo>()
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(result.final_output, Some(json!({"bar": "baz"})));
    let call = model.last_call().unwrap();
    let schema = call.output_schema.expect("output schema sent");
    assert_eq!(schema.json_schema().unwrap()["required"], json!(["bar"]));
}
