//! Port of `tests/test_tracing_errors_streamed.py`: the same span errors through the streamed path.
//!
//! Python's exception class names have no Rust counterpart. The Rust error for the same failure is
//! `AgentsError::UserError`, so its span payloads carry that name where Python carries
//! `ValueError`.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::Arc;

use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::TracingProcessor;
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunOptions, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use support::{NoApiKey, Pipeline, SpanRecorder, lock_async, set_tool_data_logging};

#[derive(Deserialize, JsonSchema, Default)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Foo {
    bar: String,
}

/// A function tool returning `value` for any arguments (Python's `get_function_tool`).
fn function_tool(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn text(message: &str) -> openai_agents_sdk::items::OutputItem {
    assistant_message(message)
}

fn tool_call(name: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(name, "", call_id)
}

fn handoff_call(agent_name: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(format!("transfer_to_{agent_name}"), "{}", call_id)
}

fn model(emit: bool) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new().with_emit_traces(emit))
}

fn user_error_span(message: &str) -> Value {
    json!({"message": "Error", "data": {"name": "UserError", "message": message}})
}

fn bare_agent_data(name: &str, handoffs: &[&str], tools: &[&str], output_type: &str) -> Value {
    json!({"name": name, "handoffs": handoffs, "tools": tools, "output_type": output_type})
}

#[tokio::test(flavor = "current_thread")]
async fn test_single_turn_model_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();

    let model = model(true);
    model
        .enqueue(ScriptedStep::raise_error(AgentsError::user("test error")))
        .unwrap();
    let agent = Agent::<()>::new("test_agent").with_model_instance(model.clone());
    assert!(streamed_result(&agent, "first_test").await.is_err());

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "error": {"message": "Error in agent run", "data": {"error": "test error"}},
                "data": bare_agent_data("test_agent", &[], &[], "str"),
                "children": [{
                    "type": "generation",
                    "error": {"message": "Error", "data": {"name": "UserError", "message": "test error"}},
                }],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_multi_turn_no_handoffs() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();

    let model = model(true);
    model
        .extend(vec![
            ScriptedStep::output(vec![text("a_message"), tool_call("foo", "call_1")]),
            ScriptedStep::raise_error(AgentsError::user("test error")),
            ScriptedStep::output(vec![text("done")]),
        ])
        .unwrap();
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_tool(function_tool("foo", "tool_result"));
    assert!(streamed_result(&agent, "first_test").await.is_err());

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "error": {"message": "Error in agent run", "data": {"error": "test error"}},
                "data": bare_agent_data("test_agent", &[], &["foo"], "str"),
                "children": [
                    {"type": "generation"},
                    {"type": "function", "data": {"name": "foo", "input": "", "output": "tool_result"}},
                    {"type": "generation", "error": user_error_span("test error")},
                ],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_tool_call_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    // Local diagnostics do not opt model feedback or default error spans into exception details.
    set_tool_data_logging(true);

    let model = model(true);
    model
        .extend(vec![
            ScriptedStep::output(vec![
                text("a_message"),
                tool_call_with_args("foo", "bad_json", "call_1"),
            ]),
            ScriptedStep::output(vec![text("done")]),
        ])
        .unwrap();
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_tool(function_tool("foo", "tool_result"));
    let result = streamed_result(&agent, "first_test").await.expect("run");
    assert_eq!(
        tool_output_texts(&result.new_items),
        ["An error occurred while running the tool. Please try again."]
    );

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "data": bare_agent_data("test_agent", &[], &["foo"], "str"),
                "children": [
                    {"type": "generation"},
                    {
                        "type": "function",
                        "error": {
                            "message": "Error running tool (non-fatal)",
                            "data": {
                                "tool_name": "foo",
                                "error": "Tool execution failed. Error details are redacted.",
                            },
                        },
                        "data": {
                            "name": "foo",
                            "input": "bad_json",
                            "output": "An error occurred while running the tool. Please try again.",
                        },
                    },
                    {"type": "generation"},
                ],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_multiple_handoff_doesnt_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();

    let model = model(true);
    let agent_1 = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let agent_2 = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let agent_3 = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_handoffs(vec![handoff(agent_1.clone()), handoff(agent_2.clone())])
        .with_tool(function_tool("some_function", "result"));
    model
        .extend(vec![
            ScriptedStep::output(vec![tool_call_with_args(
                "some_function",
                r#"{"a": "b"}"#,
                "call_1",
            )]),
            ScriptedStep::output(vec![
                text("a_message"),
                handoff_call("test", "handoff_1"),
                handoff_call("test", "handoff_2"),
            ]),
            ScriptedStep::output(vec![text("done")]),
        ])
        .unwrap();

    let result = streamed_result(&agent_3, "user_message")
        .await
        .expect("run");
    assert_eq!(result.last_agent.name, "test");

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [
                {
                    "type": "agent",
                    "data": bare_agent_data("test", &["test"], &["some_function"], "str"),
                    "children": [
                        {"type": "generation"},
                        {"type": "function", "data": {"name": "some_function", "input": r#"{"a": "b"}"#, "output": "result"}},
                        {"type": "generation"},
                        {
                            "type": "handoff",
                            "data": {"from_agent": "test", "to_agent": "test"},
                            "error": {
                                "data": {"requested_agents": ["test", "test"]},
                                "message": "Multiple handoffs requested",
                            },
                        },
                    ],
                },
                {
                    "type": "agent",
                    "data": bare_agent_data("test", &[], &[], "str"),
                    "children": [{"type": "generation"}],
                },
            ],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_max_turns_exceeded() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();

    let model = model(true);
    model
        .extend(
            (1..=5)
                .map(|i| ScriptedStep::output(vec![tool_call("foo", &format!("tool_{i}"))]))
                .collect(),
        )
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_type::<Foo>()
        .with_tool(function_tool("foo", "result"));

    let err = streamed_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(2)),
    )
    .await
    .expect_err("max turns");
    assert!(
        matches!(err, AgentsError::MaxTurnsExceeded { .. }),
        "{err:?}"
    );

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "error": {"message": "Max turns exceeded", "data": {"max_turns": 2}},
                "data": bare_agent_data("test", &[], &["foo"], "Foo"),
                "children": [
                    {"type": "generation"},
                    {"type": "function", "data": {"name": "foo", "input": "", "output": "result"}},
                    {"type": "generation"},
                    {"type": "function", "data": {"name": "foo", "input": "", "output": "result"}},
                ],
            }],
        })]
    );
}

/// Run through the streamed path to completion and return the run result, as Python's
/// `Runner.run_streamed` + `stream_events` does.
async fn streamed_result(agent: &Agent<()>, input: &str) -> Result<RunResult<()>, AgentsError> {
    let mut run = Runner::run_streamed(agent, input);
    run.finish().await?;
    Ok(run.into_result().expect("a finished run has a result"))
}

async fn streamed_with_options(
    agent: &Agent<()>,
    input: &str,
    options: RunOptions<()>,
) -> Result<RunResult<()>, AgentsError> {
    let mut run = Runner::run_streamed_with_options(agent, input, options);
    run.finish().await?;
    Ok(run.into_result().expect("a finished run has a result"))
}

#[tokio::test(flavor = "current_thread")]
async fn test_streamed_agent_error_redacts_sensitive_data_inner_handler() {
    // Python's inner and outer handler variants differ only in which exception type they raise.
    // Rust has one error path for both, so each variant is checked against the same expectation.
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = model(false);
    model
        .enqueue(ScriptedStep::raise_error(AgentsError::user(
            "sensitive-error-detail",
        )))
        .unwrap();
    let agent = Agent::<()>::new("test_agent").with_model_instance(model.clone());
    let options =
        RunOptions::new(()).run_config(RunConfig::new().with_trace_include_sensitive_data(false));
    assert!(
        streamed_with_options(&agent, "first_test", options)
            .await
            .is_err()
    );
    assert_eq!(
        span_errors(&recorder, "agent"),
        vec![
            json!({"message": "Error in agent run", "data": {"error": "Error details are redacted."}})
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_streamed_agent_error_redacts_sensitive_data_outer_handler() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = model(false);
    model
        .enqueue(ScriptedStep::raise_error(AgentsError::user(
            "sensitive-error-detail",
        )))
        .unwrap();
    let agent = Agent::<()>::new("test_agent").with_model_instance(model.clone());
    let options =
        RunOptions::new(()).run_config(RunConfig::new().with_trace_include_sensitive_data(false));
    assert!(
        streamed_with_options(&agent, "first_test", options)
            .await
            .is_err()
    );
    assert_eq!(
        span_errors(&recorder, "agent"),
        vec![
            json!({"message": "Error in agent run", "data": {"error": "Error details are redacted."}})
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_multiple_final_output_no_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = model(true);
    model
        .enqueue(ScriptedStep::output(vec![
            text(&json!({"bar": "baz"}).to_string()),
            text(&json!({"bar": "abc"}).to_string()),
        ]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_type::<Foo>();
    let result = streamed_result(&agent, "user_message").await.expect("run");
    assert_eq!(result.final_output, Some(json!({"bar": "abc"})));
    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "data": bare_agent_data("test", &[], &[], "Foo"),
                "children": [{"type": "generation"}],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_input_guardrail_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = model(false);
    model
        .enqueue(ScriptedStep::output(vec![text("some_message")]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_input_guardrail(tripwire_guardrail());
    let err = streamed_result(&agent, "user_message")
        .await
        .expect_err("tripwire");
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "error": {
                    "message": "Guardrail tripwire triggered",
                    "data": {"guardrail": "guardrail_function", "type": "input_guardrail"},
                },
                "data": bare_agent_data("test", &[], &[], "str"),
                "children": [{
                    "type": "guardrail",
                    "data": {"name": "guardrail_function", "triggered": true},
                }],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_output_guardrail_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = model(false);
    model
        .enqueue(ScriptedStep::output(vec![text("some_message")]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_guardrail(output_tripwire_guardrail());
    let err = streamed_result(&agent, "user_message")
        .await
        .expect_err("tripwire");
    assert!(
        matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [{
                "type": "agent",
                "error": {
                    "message": "Guardrail tripwire triggered",
                    "data": {"guardrail": "output_guardrail_function"},
                },
                "data": bare_agent_data("test", &[], &[], "str"),
                "children": [{
                    "type": "guardrail",
                    "data": {"name": "output_guardrail_function", "triggered": true},
                }],
            }],
        })]
    );
}

fn output_tripwire_guardrail() -> openai_agents_sdk::guardrail::OutputGuardrail<()> {
    openai_agents_sdk::guardrail::OutputGuardrail::new(
        "output_guardrail_function",
        |_ctx, _agent, _output| async move {
            Ok(openai_agents_sdk::guardrail::GuardrailFunctionOutput::tripwire())
        },
    )
}

/// The text of every tool output item in a run, in order.
fn tool_output_texts(items: &[openai_agents_sdk::items::RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::ToolCallOutput { output, .. } => {
                Some(output.as_str().unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect()
}

fn tool_call_with_args(
    name: &str,
    args: &str,
    call_id: &str,
) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(name, args, call_id)
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

fn recorder_pipeline() -> (Arc<SpanRecorder>, Pipeline) {
    let recorder = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
    (recorder, pipeline)
}

/// The `error` payloads of every exported span of `span_type`, as Python's `fetch_span_errors`.
fn span_errors(recorder: &SpanRecorder, span_type: &str) -> Vec<Value> {
    recorder
        .ordered_spans()
        .iter()
        .filter_map(|s| s.export())
        .filter(|e| e["span_data"]["type"] == span_type)
        .filter_map(|e| e.get("error").cloned())
        .filter(|e| !e.is_null())
        .collect()
}

fn tripwire_guardrail() -> openai_agents_sdk::guardrail::InputGuardrail<()> {
    openai_agents_sdk::guardrail::InputGuardrail::new(
        "guardrail_function",
        |_ctx, _agent, _input| async move {
            Ok(openai_agents_sdk::guardrail::GuardrailFunctionOutput::tripwire())
        },
    )
}
