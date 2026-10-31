//! Ports of `test_logprobs.py` (Chat Completions half), `test_tool_converter.py` and
//! `test_disabled_trace_provider_flush.py`. Python drives the Chat converter and model with a
//! dummy client; Rust asserts on the request body the model would send.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::handoffs::{self, handoff};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{FunctionToolSpec, ModelRequest, ToolSpec};
use openai_agents_sdk::models::{OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::tool::{FileSearchTool, FunctionTool, Tool, WebSearchTool, tool_namespace};
use openai_agents_sdk::tracing::{TracingProcessor, flush_traces, set_trace_processors};
use openai_agents_sdk::{Agent, AgentsError};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

fn model() -> OpenAIChatCompletionsModel {
    OpenAIChatCompletionsModel::new("gpt-4", OpenAIClient::new("sk-test"))
}

fn body(settings: ModelSettings) -> Value {
    model()
        .build_request_body(
            &ModelRequest {
                input: vec![openai_agents_sdk::InputItem::user("hi")],
                model_settings: settings,
                ..Default::default()
            },
            false,
        )
        .unwrap()
}

fn extra(key: &str, value: Value) -> Option<Map<String, Value>> {
    Some([(key.to_string(), value)].into_iter().collect())
}

// ------------------------------------------------------------------ test_logprobs.py

/// Python: `test_chat_completions_top_logprobs_sets_logprobs_flag`.
#[test]
fn chat_top_logprobs_sets_the_logprobs_flag() {
    let body = body(ModelSettings {
        top_logprobs: Some(2),
        ..Default::default()
    });
    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["logprobs"], true);
}

/// Python: `test_chat_completions_omits_logprobs_when_top_logprobs_unset`.
#[test]
fn chat_omits_logprobs_when_top_logprobs_is_unset() {
    assert!(body(ModelSettings::default()).get("logprobs").is_none());
}

/// Python: `test_chat_completions_extra_args_logprobs_passthrough`.
#[test]
fn chat_extra_args_logprobs_pass_through() {
    let body = body(ModelSettings {
        extra_args: extra("logprobs", json!(true)),
        ..Default::default()
    });
    assert_eq!(body["logprobs"], true);
}

/// Python: `test_chat_completions_top_logprobs_with_extra_args_logprobs_does_not_collide`.
#[test]
fn chat_top_logprobs_defers_to_extra_args_logprobs() {
    let body = body(ModelSettings {
        top_logprobs: Some(2),
        extra_args: extra("logprobs", json!(true)),
        ..Default::default()
    });
    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["logprobs"], true);
}

// ------------------------------------------------------------- test_tool_converter.py

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct SomeFunctionArgs {
    a: String,
    b: Vec<i64>,
}

fn some_function() -> FunctionTool<()> {
    FunctionTool::new(
        "some_function",
        "a tool",
        |_ctx, _args: SomeFunctionArgs| async { Ok("hello".to_string()) },
    )
}

fn chat_param(tool: &FunctionTool<()>) -> Value {
    tool.to_spec().to_chat_param()
}

fn user_error_for(tool: Tool<()>) -> String {
    let spec = match &tool {
        Tool::Function(f) => ToolSpec::Function(f.to_spec()),
        other => {
            let (param, includes) = other.to_responses_param();
            ToolSpec::Hosted { param, includes }
        }
    };
    let err = model()
        .build_request_body(
            &ModelRequest {
                tools: vec![spec],
                ..Default::default()
            },
            false,
        )
        .unwrap_err();
    assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
    err.to_string()
}

/// Python: `test_to_openai_with_function_tool`.
#[test]
fn a_function_tool_converts_to_a_strict_chat_function() {
    let result = chat_param(&some_function());
    assert_eq!(result["type"], "function");
    let function = &result["function"];
    assert_eq!(function["name"], "some_function");
    assert_eq!(function["strict"], true);
    let properties = function["parameters"]["properties"].as_object().unwrap();
    let mut keys: Vec<_> = properties.keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["a", "b"]);
}

/// Python: `test_to_openai_respects_non_strict_function_tool`.
#[test]
fn a_non_strict_function_tool_converts_with_strict_false() {
    let result = chat_param(&some_function().with_strict_json_schema(false));
    assert_eq!(result["function"]["strict"], false);
}

/// Python: `test_convert_handoff_tool`.
#[test]
fn a_handoff_converts_to_a_strict_chat_function() {
    let agent = Agent::<()>::new("test_1").with_handoff_description("test_2");
    let handoff_obj = handoff(agent);
    let result = handoff_obj.to_spec().to_chat_param();
    assert_eq!(result["type"], "function");
    assert_eq!(
        result["function"]["name"],
        handoffs::default_tool_name("test_1")
    );
    assert_eq!(
        result["function"]["description"],
        handoffs::default_tool_description("test_1", Some("test_2"))
    );
    assert_eq!(result["function"]["strict"], true);
    let params = result["function"]["parameters"].as_object().unwrap();
    for (key, value) in handoff_obj.input_json_schema.as_object().unwrap() {
        assert_eq!(&params[key], value);
    }
}

/// Python: `test_tool_converter_hosted_tools_errors`.
#[test]
fn hosted_tools_are_rejected_by_the_chat_converter() {
    let message = user_error_for(Tool::from(WebSearchTool::default()));
    assert!(
        message.contains("Hosted tools are not supported"),
        "{message}"
    );
    let message = user_error_for(Tool::from(FileSearchTool {
        max_num_results: Some(1),
        ..FileSearchTool::new(vec!["abc".to_string()])
    }));
    assert!(
        message.contains("Hosted tools are not supported"),
        "{message}"
    );
}

/// Python: `test_tool_converter_rejects_namespaced_function_tools_for_chat_backends`.
#[test]
fn namespaced_function_tools_are_rejected_by_the_chat_converter() {
    let tool = tool_namespace("crm", "CRM tools", vec![some_function()])
        .unwrap()
        .remove(0);
    let message = user_error_for(Tool::from(tool));
    assert!(message.contains("tool_namespace()"), "{message}");
    assert!(message.contains("`crm.some_function`"), "{message}");
}

/// Python: `test_tool_converter_rejects_deferred_function_tools_for_chat_backends`.
#[test]
fn deferred_function_tools_are_rejected_by_the_chat_converter() {
    let message = user_error_for(Tool::from(some_function().with_defer_loading(true)));
    assert!(message.contains("defer_loading=True"), "{message}");
}

/// The Chat backend also rejects a tool with a strict output schema, as Python's shared
/// Responses-only feature check does.
#[test]
fn spec_with_an_output_schema_is_rejected_by_the_chat_converter() {
    let spec = FunctionToolSpec {
        name: "typed".into(),
        parameters: json!({"type": "object", "properties": {}}),
        output_json_schema: Some(json!({"type": "object"})),
        ..Default::default()
    };
    let err = model()
        .build_request_body(
            &ModelRequest {
                tools: vec![ToolSpec::Function(spec)],
                ..Default::default()
            },
            false,
        )
        .unwrap_err();
    assert!(err.to_string().contains("output_json_schema"), "{err}");
}

// ------------------------------------------------ test_disabled_trace_provider_flush.py

#[derive(Debug, Default)]
struct FlushCounter {
    flushes: AtomicUsize,
}

impl TracingProcessor for FlushCounter {
    fn on_trace_start(&self, _: &openai_agents_sdk::tracing::Trace) {}
    fn on_trace_end(&self, _: &openai_agents_sdk::tracing::Trace) {}
    fn on_span_start(&self, _: &openai_agents_sdk::tracing::Span) {}
    fn on_span_end(&self, _: &openai_agents_sdk::tracing::Span) {}
    fn force_flush(&self) {
        self.flushes.fetch_add(1, Ordering::SeqCst);
    }
}

/// Python: `test_flush_traces_still_flushes_registered_processors_when_disabled`.
#[test]
fn flush_traces_flushes_registered_processors_when_tracing_is_disabled() {
    let counter = Arc::new(FlushCounter::default());
    set_trace_processors(vec![counter.clone()]);
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    flush_traces();
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    set_trace_processors(Vec::new());
    assert_eq!(counter.flushes.load(Ordering::SeqCst), 1);
}
