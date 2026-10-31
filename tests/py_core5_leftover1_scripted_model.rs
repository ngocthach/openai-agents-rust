//! Port of `tests/test_scripted_model.py`: the scripted model's recording, usage accounting,
//! retry advice, tracing and automatic stream synthesis.
//!
//! Python-only groups (no Rust counterpart): dict step specs and mutable-aliasing snapshots (the
//! Rust steps are typed and moved into the queue; recorded calls are owned clones), pydantic
//! validation of mistyped step fields (the struct makes them unrepresentable), `aclose()` /
//! `BaseException` / `__str__` failure modes and `contextvars` scope checks.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::items::{MessageContent, MessageOutput, OutputItem};
use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::retry::{
    ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety,
};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    ModelCall, ModelScriptError, ModelStepReason, ScriptedModel, ScriptedStep, assistant_message,
    assistant_message_with_id, function_call_full, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{REDACTED_TRACE_ERROR_MESSAGE, TracingProcessor};
use openai_agents_sdk::usage::{InputTokensDetails, OutputTokensDetails, RequestUsage, Usage};
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunInput, RunOptions, RunResult, Runner};
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

fn function_tool(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn request() -> ModelRequest {
    request_with(ModelSettings::default())
}

fn request_with(model_settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        system_instructions: None,
        input: vec![],
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

fn scripted(steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::with_steps(steps).unwrap())
}

fn out(items: Vec<OutputItem>) -> ScriptedStep {
    ScriptedStep::output(items)
}

/// Serializes with the tests that install a recorder, and keeps the rest from exporting spans.
async fn quiet() -> tokio::sync::MutexGuard<'static, ()> {
    let guard = lock_async().await;
    openai_agents_sdk::set_tracing_disabled(true);
    guard
}

fn raw(step: ScriptedStep, raw_usage: Value) -> ScriptedStep {
    ScriptedStep {
        raw_usage: Some(raw_usage),
        ..step
    }
}

fn agent_for(model: &Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new("test").with_model_instance(model.clone())
}

/// Run `agent`, streamed or not, to its end; a stream error is the run's error.
async fn run_mode(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> Result<RunResult<()>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut result = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut failure = None;
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let Err(error) = event {
                failure = Some(error);
            }
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(result.into_result().expect("a finished streamed run")),
    }
}

async fn stream_events_of(model: &ScriptedModel, request: ModelRequest) -> Vec<Value> {
    model
        .stream_response(request)
        .await
        .unwrap()
        .map(|event| event.unwrap())
        .collect()
        .await
}

fn types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap_or_default())
        .collect()
}

fn of_type<'a>(events: &'a [Value], ty: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == ty).collect()
}

fn usage_entry(
    input: u64,
    output: u64,
    total: u64,
    cached: u64,
    write: u64,
    reasoning: u64,
) -> RequestUsage {
    RequestUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        input_tokens_details: InputTokensDetails {
            cached_tokens: cached,
            cache_write_tokens: write,
        },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: reasoning,
        },
    }
}

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

fn recorder_pipeline() -> (Arc<SpanRecorder>, Pipeline) {
    let recorder = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
    (recorder, pipeline)
}

fn traced_model(steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
    let model = ScriptedModel::new().with_emit_traces(true);
    model.extend(steps).unwrap();
    Arc::new(model)
}

fn generation_error(name: &str, message: &str) -> Value {
    json!({"message": "Error", "data": {"name": name, "message": message}})
}

// ---------------------------------------------------------------------------------------------
// Workflow, recording and usage
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_runs_tool_workflow_and_records_calls`.
#[tokio::test]
async fn runs_tool_workflow_and_records_calls() {
    let _serial = quiet().await;
    let model = scripted(vec![
        out(vec![function_call_with_id(
            "lookup",
            r#"{"city":"Tokyo"}"#,
            "call_1",
        )]),
        out(vec![assistant_message("sunny")]),
    ]);
    let agent = agent_for(&model).with_tool(function_tool("lookup", "tool result"));

    let result = Runner::run(&agent, "weather?").await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("sunny"));
    assert_eq!(model.calls().len(), 2);
    let first = model.first_call().unwrap();
    assert_eq!(first.request.input.len(), 1);
    assert_eq!(
        first.request.input[0].0,
        json!({"content": "weather?", "role": "user"})
    );
    assert!(!model.last_call().unwrap().streamed);
    model.assert_complete().unwrap();
}

/// Python: `test_scripted_model_generates_stream_events`.
#[tokio::test]
async fn generates_stream_events() {
    let _serial = quiet().await;
    let model = scripted(vec![out(vec![assistant_message("hello")])]);
    let agent = agent_for(&model);
    let mut result = Runner::run_streamed(agent, "hi");
    let mut count = 0;
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
            count += 1;
        }
    }
    assert_eq!(result.final_output_text().as_deref(), Some("hello"));
    assert!(count > 0);
    assert!(model.last_call().unwrap().streamed);
    model.assert_complete().unwrap();
}

/// Python: `test_scripted_model_stream_preserves_request_usage_details`.
#[tokio::test]
async fn stream_preserves_request_usage_details() {
    let _serial = quiet().await;
    let entries = vec![
        usage_entry(3, 2, 5, 1, 0, 1),
        usage_entry(7, 4, 11, 2, 3, 2),
    ];
    let usage = Usage {
        requests: 5,
        input_tokens: 10,
        output_tokens: 6,
        total_tokens: 16,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 3,
            cache_write_tokens: 3,
        },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 3,
        },
        request_usage_entries: entries.clone(),
    };
    let make = || {
        scripted(vec![
            out(vec![assistant_message("hello")]).with_usage(usage.clone()),
        ])
    };

    let streamed = run_mode(&agent_for(&make()), "hi", RunOptions::default(), true)
        .await
        .unwrap();
    let non_streamed = run_mode(&agent_for(&make()), "hi", RunOptions::default(), false)
        .await
        .unwrap();

    let propagated = streamed.usage();
    assert_eq!(propagated.requests, 5);
    assert_eq!(propagated.input_tokens, 10);
    assert_eq!(propagated.output_tokens, 6);
    assert_eq!(propagated.total_tokens, 16);
    assert_eq!(propagated.request_usage_entries, entries);
    assert_eq!(propagated, non_streamed.usage());
}

/// Python: `test_scripted_model_stream_counts_default_usage_as_one_request`.
#[tokio::test]
async fn stream_counts_default_usage_as_one_request() {
    let _serial = quiet().await;
    for streamed in [true, false] {
        let model = scripted(vec![out(vec![assistant_message("done")])]);
        let result = run_mode(&agent_for(&model), "hi", RunOptions::default(), streamed)
            .await
            .unwrap();
        assert_eq!(result.usage().requests, 1, "streamed={streamed}");
    }
}

/// Python: `test_scripted_model_preserves_explicit_zero_request_usage_across_run_modes`.
#[tokio::test]
async fn preserves_explicit_zero_request_usage_across_run_modes() {
    let _serial = quiet().await;
    let usage = Usage {
        input_tokens: 2,
        output_tokens: 1,
        total_tokens: 3,
        request_usage_entries: vec![usage_entry(1, 1, 2, 0, 0, 0)],
        ..Default::default()
    };
    for streamed in [true, false] {
        let model = scripted(vec![
            out(vec![assistant_message("done")]).with_usage(usage.clone()),
        ]);
        let result = run_mode(&agent_for(&model), "hi", RunOptions::default(), streamed)
            .await
            .unwrap();
        assert_eq!(result.usage(), usage, "streamed={streamed}");
    }
}

fn retry_settings(policy: openai_agents_sdk::retry::RetryPolicy) -> ModelSettings {
    ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: 1,
                backoff: ModelRetryBackoffSettings {
                    initial_delay: 0.0,
                    max_delay: 0.0,
                    multiplier: 1.0,
                    jitter: false,
                },
                ..Default::default()
            }
            .with_policy(policy),
        ),
        ..Default::default()
    }
}

fn network_error_step() -> ScriptedStep {
    ScriptedStep::raise_error(AgentsError::other("connection error")).with_retry_advice(
        ModelRetryAdvice {
            suggested: Some(true),
            replay_safety: Some(ReplaySafety::Safe),
            normalized: Some(ModelRetryNormalizedError {
                is_network_error: true,
                explicit_fields: vec!["is_network_error"],
                ..Default::default()
            }),
            ..Default::default()
        },
    )
}

/// Python: `test_scripted_model_copies_default_usage_before_retry_accounting`.
#[tokio::test]
async fn copies_default_usage_before_retry_accounting() {
    let _serial = quiet().await;
    let default_usage = Usage {
        requests: 1,
        input_tokens: 2,
        total_tokens: 2,
        ..Default::default()
    };
    let model = scripted(vec![
        network_error_step(),
        out(vec![assistant_message("first")]),
        out(vec![assistant_message("second")]),
    ]);
    model.set_default_usage(Some(default_usage.clone()));
    let agent = agent_for(&model).with_model_settings(retry_settings(
        openai_agents_sdk::retry::retry_policies::network_error(),
    ));

    let first = Runner::run(&agent, "first").await.unwrap();
    let second = Runner::run(&agent, "second").await.unwrap();

    assert_eq!(first.usage().requests, 2);
    assert_eq!(second.usage().requests, 1);
    assert_eq!(
        default_usage,
        Usage {
            requests: 1,
            input_tokens: 2,
            total_tokens: 2,
            ..Default::default()
        }
    );
}

/// Python: `test_scripted_model_stream_preserves_response_id[None, ""]`.
#[tokio::test]
async fn stream_preserves_response_id() {
    let _serial = quiet().await;
    for response_id in [None, Some(String::new())] {
        for streamed in [true, false] {
            let model = scripted(vec![
                out(vec![assistant_message("done")]).with_response_id(response_id.clone()),
            ]);
            let result = run_mode(&agent_for(&model), "hi", RunOptions::default(), streamed)
                .await
                .unwrap();
            assert_eq!(
                result.last_response_id(),
                response_id,
                "streamed={streamed} id={response_id:?}"
            );
        }
    }
}

/// Python: `test_scripted_model_stream_does_not_chain_absent_response_id`.
#[tokio::test]
async fn stream_does_not_chain_absent_response_id() {
    let _serial = quiet().await;
    let model = scripted(vec![
        out(vec![function_call_with_id("lookup", "{}", "call_1")]).with_response_id(None),
        ScriptedStep::respond(|call| {
            assert_eq!(call.request.previous_response_id, None);
            Ok(out(vec![assistant_message("done")]))
        }),
    ]);
    let agent = agent_for(&model).with_tool(function_tool("lookup", "tool result"));

    let result = run_mode(
        &agent,
        "hi",
        RunOptions::default().auto_previous_response_id(true),
        true,
    )
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    model.assert_complete().unwrap();
}

/// Python: `test_scripted_model_supports_dynamic_responder`.
#[tokio::test]
async fn supports_dynamic_responder() {
    let _serial = quiet().await;
    let model = scripted(vec![ScriptedStep::respond(|call| {
        Ok(out(vec![assistant_message(
            call.request.input[0].text_content().unwrap_or_default(),
        )]))
    })]);
    let result = Runner::run(&agent_for(&model), "hello").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("hello"));
}

/// Python: `test_scripted_model_accepts_step_mapping` (a struct literal is the Rust spelling).
#[tokio::test]
async fn accepts_a_fully_specified_step() {
    let usage = Usage {
        requests: 2,
        input_tokens: 3,
        output_tokens: 4,
        total_tokens: 7,
        ..Default::default()
    };
    let model = ScriptedModel::with_steps(vec![ScriptedStep {
        output: vec![assistant_message("mapped")],
        usage: Some(usage.clone()),
        response_id: Some("resp_mapped".into()),
        request_id: Some("req_mapped".into()),
        raw_usage: Some(json!({"source": "mapping"})),
        ..Default::default()
    }])
    .unwrap();

    let response = model.get_response(request()).await.unwrap();

    assert_eq!(
        response.output[0].to_value()["content"][0]["text"],
        "mapped"
    );
    assert_eq!(response.usage, usage);
    assert_eq!(response.response_id.as_deref(), Some("resp_mapped"));
    assert_eq!(response.request_id.as_deref(), Some("req_mapped"));
    assert_eq!(response.raw_usage, None);
    model.assert_complete().unwrap();
}

/// Python: `test_scripted_model_accepts_step_mappings_in_queue_methods`.
#[tokio::test]
async fn accepts_steps_in_queue_methods() {
    let _serial = quiet().await;
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(out(vec![assistant_message("first")]))
        .unwrap();
    model
        .extend(vec![out(vec![assistant_message("second")])])
        .unwrap();
    let agent = agent_for(&model);

    let first = Runner::run(&agent, "first").await.unwrap();
    let second = Runner::run(&agent, "second").await.unwrap();

    assert_eq!(first.final_output_text().as_deref(), Some("first"));
    assert_eq!(second.final_output_text().as_deref(), Some("second"));
    model.assert_complete().unwrap();
}

/// Python: `test_scripted_model_responder_can_return_step_mapping`.
#[tokio::test]
async fn responder_can_return_a_step() {
    let _serial = quiet().await;
    let model = scripted(vec![ScriptedStep::respond(|call| {
        Ok(ScriptedStep {
            output: vec![assistant_message(
                call.request.input[0].text_content().unwrap_or_default(),
            )],
            ..Default::default()
        })
    })]);
    let result = Runner::run(&agent_for(&model), "mapped").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("mapped"));
    model.assert_complete().unwrap();
}

// ---------------------------------------------------------------------------------------------
// Raw usage
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_gates_and_snapshots_raw_usage` (6 parameter combinations).
#[tokio::test]
async fn gates_raw_usage_on_the_model_setting() {
    let _serial = quiet().await;
    for streamed in [false, true] {
        for preserve in [None, Some(false), Some(true)] {
            let raw_usage = json!({"provider": {"input_tokens": 3}});
            let model = scripted(vec![raw(
                out(vec![assistant_message("done")]),
                raw_usage.clone(),
            )]);
            let agent = agent_for(&model).with_model_settings(ModelSettings {
                preserve_raw_usage: preserve,
                ..Default::default()
            });

            let mut terminal_usage = None;
            let result = if streamed {
                let mut result = Runner::run_streamed(agent.clone(), "hi");
                {
                    let mut events = std::pin::pin!(result.stream_events());
                    while let Some(event) = events.next().await {
                        if let StreamEvent::RawResponse(raw) = event.unwrap()
                            && raw["type"] == "response.completed"
                        {
                            terminal_usage = Some(raw["response"]["usage"].clone());
                        }
                    }
                }
                result.into_result().unwrap()
            } else {
                Runner::run(&agent, "hi").await.unwrap()
            };

            let expected = (preserve == Some(true)).then(|| raw_usage.clone());
            assert_eq!(
                result.raw_responses[0]
                    .raw_usage
                    .as_ref()
                    .map(|v| v["provider"].clone()),
                expected.as_ref().map(|v| v["provider"].clone()),
                "streamed={streamed} preserve={preserve:?}"
            );
            if streamed {
                let terminal = terminal_usage.expect("a terminal event");
                // The terminal event only carries the provider payload when it is preserved.
                assert_eq!(
                    terminal.get("provider").cloned(),
                    expected.map(|v| v["provider"].clone()),
                    "streamed={streamed} preserve={preserve:?}"
                );
            }
        }
    }
}

/// Python: `test_scripted_model_withholds_raw_usage_until_stream_completion`.
#[tokio::test]
async fn withholds_raw_usage_until_stream_completion() {
    let raw_usage = json!({"provider": {"input_tokens": 3}});
    let model = ScriptedModel::with_steps(vec![raw(
        out(vec![assistant_message("done")]),
        raw_usage.clone(),
    )])
    .unwrap();

    let events = stream_events_of(
        &model,
        request_with(ModelSettings {
            preserve_raw_usage: Some(true),
            ..Default::default()
        }),
    )
    .await;

    let usage_of = |ty: &str| of_type(&events, ty)[0]["response"].get("usage").cloned();
    assert_eq!(usage_of("response.created"), None);
    assert_eq!(usage_of("response.in_progress"), None);
    assert_eq!(
        usage_of("response.completed").unwrap()["provider"],
        raw_usage["provider"]
    );
}

// ---------------------------------------------------------------------------------------------
// Step validation
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_validates_step_envelopes_before_queuing` (the two reasons a typed
/// step can still hit; the other eight are mistyped fields).
#[tokio::test]
async fn validates_step_envelopes_before_queuing() {
    let model = ScriptedModel::with_steps(vec![out(vec![assistant_message("retained")])]).unwrap();

    let conflicting = ScriptedStep {
        error: Some(AgentsError::other("failed")),
        responder: Some(Arc::new(|_| Ok(ScriptedStep::default()))),
        ..Default::default()
    };
    let orphan_advice = out(vec![]).with_retry_advice(ModelRetryAdvice {
        suggested: Some(true),
        ..Default::default()
    });
    for (step, reason) in [
        (conflicting, ModelStepReason::ConflictingOutcomes),
        (orphan_advice, ModelStepReason::InvalidRetryAdvice),
    ] {
        match model.extend(vec![step]) {
            Err(ModelScriptError::InvalidModelStep {
                reason: actual,
                input_index,
                ..
            }) => {
                assert_eq!(actual, reason);
                assert_eq!(input_index, 0);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(model.remaining_steps(), 1);
    }
}

/// Python: `test_scripted_model_reports_the_zero_origin_index_of_an_invalid_step`.
#[tokio::test]
async fn reports_the_zero_origin_index_of_an_invalid_step() {
    let error = ScriptedModel::with_steps(vec![
        out(vec![assistant_message("valid")]),
        ScriptedStep {
            error: Some(AgentsError::other("failed")),
            stream_events: Some(vec![]),
            ..Default::default()
        },
    ])
    .unwrap_err();
    match &error {
        ModelScriptError::InvalidModelStep {
            reason,
            input_index,
            ..
        } => {
            assert_eq!(*reason, ModelStepReason::ConflictingOutcomes);
            assert_eq!(*input_index, 1);
        }
        other => panic!("{other:?}"),
    }
    assert!(error.to_string().contains("step #2"), "{error}");
}

/// Python: `test_scripted_model_supports_awaitable_recursive_responder`.
#[tokio::test]
async fn supports_awaitable_recursive_responder() {
    let _serial = quiet().await;
    let outer_call_input = Arc::new(Mutex::new(None::<String>));
    let seen = outer_call_input.clone();
    let model = scripted(vec![ScriptedStep::respond_async(move |call| {
        let seen = seen.clone();
        async move {
            *seen.lock().unwrap() = call.request.input[0].text_content();
            Ok(ScriptedStep::respond(|_inner| {
                Ok(out(vec![assistant_message("nested")]))
            }))
        }
    })]);

    let result = Runner::run(&agent_for(&model), "hello").await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("nested"));
    assert_eq!(model.calls().len(), 1);
    assert_eq!(outer_call_input.lock().unwrap().as_deref(), Some("hello"));
    model.assert_complete().unwrap();
}

// ---------------------------------------------------------------------------------------------
// Retry advice
// ---------------------------------------------------------------------------------------------

fn advice_request(error: &AgentsError, stream: bool) -> ModelRetryAdviceRequest<'_> {
    ModelRetryAdviceRequest {
        error,
        attempt: 1,
        stream,
        previous_response_id: None,
        conversation_id: None,
    }
}

/// Python: `test_scripted_model_retry_advice_is_error_scoped_and_detached`.
#[tokio::test]
async fn retry_advice_is_error_scoped_and_detached() {
    let error = AgentsError::other("failed");
    let mut advice = ModelRetryAdvice {
        suggested: Some(true),
        replay_safety: Some(ReplaySafety::Safe),
        ..Default::default()
    };
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep::raise_error(AgentsError::other("failed")).with_retry_advice(advice.clone()),
    ])
    .unwrap();
    // Changing the caller's copy afterwards does not reach the model.
    advice.suggested = Some(false);

    let raised = model.get_response(request()).await.unwrap_err();
    assert!(raised.to_string().contains("failed"));

    let mut first = model
        .get_retry_advice(&advice_request(&error, false))
        .unwrap();
    assert_eq!(first.suggested, Some(true));
    first.suggested = Some(false);
    let second = model
        .get_retry_advice(&advice_request(&error, false))
        .unwrap();
    assert_eq!(second.suggested, Some(true));
    let other = AgentsError::other("other");
    assert!(
        model
            .get_retry_advice(&advice_request(&other, false))
            .is_none()
    );
}

/// Python: `test_scripted_model_clears_stale_retry_advice_after_responder_failure`.
#[tokio::test]
async fn clears_stale_retry_advice_after_responder_failure() {
    for streamed in [false, true] {
        let error = AgentsError::other("failed");
        let model = ScriptedModel::with_steps(vec![
            ScriptedStep::raise_error(AgentsError::other("failed")).with_retry_advice(
                ModelRetryAdvice {
                    suggested: Some(true),
                    ..Default::default()
                },
            ),
            ScriptedStep::respond(|_| Err(AgentsError::other("failed"))),
            out(vec![assistant_message("must remain unconsumed")]),
        ])
        .unwrap();
        let invoke = || async {
            if streamed {
                model.stream_response(request()).await.map(|_| ())
            } else {
                model.get_response(request()).await.map(|_| ())
            }
        };

        assert!(invoke().await.unwrap_err().to_string().contains("failed"));
        assert!(
            model
                .get_retry_advice(&advice_request(&error, streamed))
                .is_some()
        );

        assert!(invoke().await.unwrap_err().to_string().contains("failed"));
        assert!(
            model
                .get_retry_advice(&advice_request(&error, streamed))
                .is_none()
        );
        assert_eq!(model.remaining_steps(), 1, "streamed={streamed}");
    }
}

// ---------------------------------------------------------------------------------------------
// Exact streams
// ---------------------------------------------------------------------------------------------

fn completed_event(output: &[OutputItem]) -> Value {
    json!({
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp_exact",
            "status": "completed",
            "output": output.iter().map(OutputItem::to_value).collect::<Vec<_>>(),
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    })
}

/// Python: `test_scripted_model_supports_exact_normalized_stream`.
#[tokio::test]
async fn supports_exact_normalized_stream() {
    let _serial = quiet().await;
    let output = vec![assistant_message("exact")];
    let event = completed_event(&output);
    // Both spellings: a fixed event list and a stream built lazily from the call.
    let eager = ScriptedStep::stream(vec![event.clone()]);
    let lazy = ScriptedStep::stream_with(move |_call| {
        futures::stream::iter(vec![Ok(event.clone())]).boxed()
    });
    for step in [eager, lazy] {
        let model = scripted(vec![step]);
        let result = run_mode(&agent_for(&model), "hi", RunOptions::default(), true)
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("exact"));
    }
}

/// Python: `test_scripted_model_rejects_unsupported_automatic_tool_streams` (4 item kinds).
#[tokio::test]
async fn rejects_unsupported_automatic_tool_streams() {
    let items = [
        json!({"type": "custom_tool_call", "call_id": "call_1", "id": "item_1",
               "input": "payload", "name": "custom"}),
        json!({"type": "code_interpreter_call", "id": "item_1", "code": "print('hello')",
               "container_id": "container_1", "outputs": null, "status": "completed"}),
        json!({"type": "mcp_call", "id": "item_1", "arguments": "{\"query\":\"hello\"}",
               "name": "search", "server_label": "docs", "status": "completed"}),
        json!({"type": "image_generation_call", "id": "item_1", "result": "base64-image",
               "status": "completed"}),
    ];
    for item in items {
        let kind = item["type"].as_str().unwrap().to_string();
        let model =
            ScriptedModel::with_steps(vec![out(vec![OutputItem::from_value(item)])]).unwrap();
        let error = match model.stream_response(request()).await {
            Err(error) => error,
            Ok(_) => panic!("{kind} must be rejected before any event is produced"),
        };
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        let text = error.to_string();
        assert!(
            text.contains(&format!("Automatic streaming does not support {kind}"))
                && text.contains("ScriptedStep::stream"),
            "{text}"
        );
    }
}

/// Raised by a stream part-way, then recorded on the generation span.
/// Python: `test_scripted_model_records_stream_factory_errors_on_generation_span` and the
/// error-preservation half of `test_scripted_model_preserves_exact_stream_error_when_close_fails`.
#[tokio::test(flavor = "current_thread")]
async fn records_stream_factory_errors_on_generation_span() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (recorder, _pipeline) = recorder_pipeline();
    let model = traced_model(vec![ScriptedStep::stream_with(|_call| {
        futures::stream::iter(vec![
            Ok(json!({"type": "response.created", "sequence_number": 0,
                      "response": {"id": "r", "status": "in_progress", "output": []}})),
            Err(AgentsError::user("stream failed")),
        ])
        .boxed()
    })]);

    let error = run_mode(&agent_for(&model), "hi", RunOptions::default(), true)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("stream failed"), "{error}");
    assert_eq!(
        span_errors(&recorder, "generation"),
        vec![generation_error("UserError", "stream failed")]
    );
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// An exact stream that yields `response.created`, then never produces another event.
fn blocking_stream(
    dropped: Arc<AtomicBool>,
    blocked: Option<Arc<tokio::sync::Notify>>,
) -> futures::stream::BoxStream<'static, Result<Value, AgentsError>> {
    let guard = DropFlag(dropped);
    let created = futures::stream::once(async {
        Ok::<_, AgentsError>(json!({"type": "response.created", "sequence_number": 0,
                                    "response": {"id": "r", "status": "in_progress", "output": []}}))
    });
    let never = futures::stream::once(async move {
        if let Some(blocked) = blocked {
            blocked.notify_one();
        }
        futures::future::pending::<()>().await;
        Err(AgentsError::other("unreachable"))
    });
    created
        .chain(never)
        .map(move |event| {
            let _guard = &guard;
            event
        })
        .boxed()
}

/// Python: `test_scripted_model_closes_exact_stream_when_outer_stream_is_closed` (dropping the
/// outer stream is `aclose()`).
#[tokio::test]
async fn drops_the_exact_stream_when_the_outer_stream_is_dropped() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let model = ScriptedModel::with_steps(vec![ScriptedStep::stream_with(move |_call| {
        blocking_stream(flag.clone(), None)
    })])
    .unwrap();

    let mut outer = model.stream_response(request()).await.unwrap();
    assert_eq!(
        outer.next().await.unwrap().unwrap()["type"],
        "response.created"
    );
    assert!(!dropped.load(Ordering::SeqCst));
    drop(outer);

    assert!(dropped.load(Ordering::SeqCst));
}

/// Python: `test_scripted_model_closes_exact_stream_when_consumer_is_cancelled`.
#[tokio::test]
async fn drops_the_exact_stream_when_the_consumer_is_cancelled() {
    let dropped = Arc::new(AtomicBool::new(false));
    let blocked = Arc::new(tokio::sync::Notify::new());
    let (flag, signal) = (dropped.clone(), blocked.clone());
    let model = ScriptedModel::with_steps(vec![ScriptedStep::stream_with(move |_call| {
        blocking_stream(flag.clone(), Some(signal.clone()))
    })])
    .unwrap();
    let mut outer = model.stream_response(request()).await.unwrap();
    outer.next().await.unwrap().unwrap();

    let pending = tokio::spawn(async move { outer.next().await });
    tokio::time::timeout(Duration::from_secs(5), blocked.notified())
        .await
        .unwrap();
    pending.abort();

    assert!(pending.await.unwrap_err().is_cancelled());
    assert!(dropped.load(Ordering::SeqCst));
}

/// Python: `test_scripted_model_early_stream_exit_has_task_safe_span_cleanup[False, True]`.
#[tokio::test(flavor = "current_thread")]
async fn early_stream_exit_finishes_the_span_cleanly() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    for emit_traces in [false, true] {
        let (recorder, _pipeline) = recorder_pipeline();
        let model = ScriptedModel::new().with_emit_traces(emit_traces);
        model
            .enqueue(out(vec![assistant_message("hello")]))
            .unwrap();

        openai_agents_sdk::tracing::scope::with_trace(support::new_trace("early exit"), async {
            let mut stream = model.stream_response(request()).await.unwrap();
            stream.next().await.unwrap().unwrap();
            drop(stream);
        })
        .await;

        let generations = recorder
            .ordered_spans()
            .iter()
            .filter_map(|s| s.export())
            .filter(|e| e["span_data"]["type"] == "generation")
            .count();
        assert_eq!(
            generations,
            usize::from(emit_traces),
            "emit_traces={emit_traces}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Span errors
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_timeout_records_generation_span_error[False, True]`.
#[tokio::test(flavor = "current_thread")]
async fn timeout_records_generation_span_error() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    for streamed in [false, true] {
        let (recorder, _pipeline) = recorder_pipeline();
        let model = traced_model(vec![ScriptedStep::respond_async(|_call| async {
            futures::future::pending::<()>().await;
            Ok(ScriptedStep::default())
        })]);
        let agent = agent_for(&model).with_model_settings(ModelSettings {
            timeout: Some(0.01),
            ..Default::default()
        });

        let error = run_mode(&agent, "hi", RunOptions::default(), streamed)
            .await
            .unwrap_err();

        assert!(
            matches!(error, AgentsError::ModelTimeoutError { .. }),
            "{error:?}"
        );
        assert_eq!(
            span_errors(&recorder, "generation"),
            vec![json!({"message": "Error",
                        "data": {"error": "Model call timed out after 0.01 seconds."}})],
            "streamed={streamed}"
        );
    }
}

/// Python: `test_scripted_model_records_responder_errors_on_generation_span` (streamed x
/// awaitable) and `..._preserves_unformattable_responder_error`'s span half.
#[tokio::test(flavor = "current_thread")]
async fn records_responder_errors_on_generation_span() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    for streamed in [false, true] {
        for awaitable in [false, true] {
            let (recorder, _pipeline) = recorder_pipeline();
            let step = if awaitable {
                ScriptedStep::respond_async(|_call| async {
                    Err(AgentsError::user("responder failed"))
                })
            } else {
                ScriptedStep::respond(|_call| Err(AgentsError::user("responder failed")))
            };
            let model = traced_model(vec![step]);

            let error = run_mode(&agent_for(&model), "hi", RunOptions::default(), streamed)
                .await
                .unwrap_err();

            assert!(error.to_string().contains("responder failed"), "{error}");
            assert_eq!(
                span_errors(&recorder, "generation"),
                vec![generation_error("UserError", "responder failed")],
                "streamed={streamed} awaitable={awaitable}"
            );
        }
    }
}

/// Python: `test_scripted_model_redacts_generation_span_errors_from_run_config[False, True]`.
#[tokio::test(flavor = "current_thread")]
async fn redacts_generation_span_errors_from_run_config() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    for streamed in [false, true] {
        let (recorder, _pipeline) = recorder_pipeline();
        let model = traced_model(vec![ScriptedStep::raise_error(AgentsError::user(
            "sensitive provider payload",
        ))]);
        let run_config = RunConfig {
            trace_include_sensitive_data: false,
            ..Default::default()
        };

        let error = run_mode(
            &agent_for(&model),
            "hi",
            RunOptions::default().run_config(run_config),
            streamed,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("sensitive provider payload"));
        assert_eq!(
            span_errors(&recorder, "generation"),
            vec![generation_error("UserError", REDACTED_TRACE_ERROR_MESSAGE)],
            "streamed={streamed}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// apply_patch_call normalization
// ---------------------------------------------------------------------------------------------

fn apply_patch(call_id: Value, id: Option<Value>, status: Option<Value>) -> OutputItem {
    let mut item = json!({
        "type": "apply_patch_call",
        "call_id": call_id,
        "operation": {"type": "delete_file", "path": "example.txt"},
    });
    if let Some(id) = id {
        item["id"] = id;
    }
    if let Some(status) = status {
        item["status"] = status;
    }
    OutputItem::from_value(item)
}

/// Python: `test_scripted_model_rejects_invalid_apply_patch_call_id[None, "", 123]`.
#[tokio::test]
async fn rejects_invalid_apply_patch_call_id() {
    let _serial = quiet().await;
    for call_id in [Value::Null, json!(""), json!(123)] {
        let model = scripted(vec![out(vec![apply_patch(
            call_id.clone(),
            Some(json!("patch_item")),
            None,
        )])]);
        let error = Runner::run(&agent_for(&model), "hi").await.unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("Tool invocations require a non-empty string call ID before execution"),
            "{call_id}: {error}"
        );
    }
}

/// Python: `test_scripted_model_rejects_invalid_apply_patch_item_id[None, "", 123]`.
#[tokio::test]
async fn rejects_invalid_apply_patch_item_id() {
    let _serial = quiet().await;
    for item_id in [Value::Null, json!(""), json!(123)] {
        let model = scripted(vec![out(vec![apply_patch(
            json!("call_1"),
            Some(item_id.clone()),
            None,
        )])]);
        let error = Runner::run(&agent_for(&model), "hi").await.unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(
                "Apply-patch tool calls require a non-empty string item ID when provided"
            ),
            "{item_id}: {error}"
        );
    }
}

/// Python: `test_scripted_model_defaults_omitted_apply_patch_item_id_to_call_id`.
#[tokio::test]
async fn defaults_omitted_apply_patch_item_id_to_call_id() {
    let model =
        ScriptedModel::with_steps(vec![out(vec![apply_patch(json!("call_1"), None, None)])])
            .unwrap();

    let response = model.get_response(request()).await.unwrap();

    let item = response.output[0].to_value();
    assert_eq!(item["type"], "apply_patch_call");
    assert_eq!(item["id"], "call_1");
    assert_eq!(item["status"], "completed");
}

/// Python: `test_scripted_model_rejects_invalid_explicit_apply_patch_status[None, "", "invalid"]`.
#[tokio::test]
async fn rejects_invalid_explicit_apply_patch_status() {
    for status in [Value::Null, json!(""), json!("invalid")] {
        let model = ScriptedModel::with_steps(vec![out(vec![apply_patch(
            json!("call_1"),
            Some(json!("patch_item")),
            Some(status.clone()),
        )])])
        .unwrap();
        let error = model.get_response(request()).await.unwrap_err();
        assert!(error.to_string().contains("status"), "{status}: {error}");
    }
}

// ---------------------------------------------------------------------------------------------
// Automatic stream synthesis
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_function_argument_events_use_output_item_id`.
#[tokio::test]
async fn function_argument_events_use_the_output_item_id() {
    let model = ScriptedModel::with_steps(vec![out(vec![function_call_full(
        "lookup",
        "{}",
        "call_1",
        Some("item_1".into()),
        None,
    )])])
    .unwrap();
    let events = stream_events_of(&model, request()).await;
    let ids: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["type"]
                .as_str()
                .is_some_and(|t| t.starts_with("response.function_call_arguments"))
        })
        .map(|e| &e["item_id"])
        .collect();
    assert_eq!(ids, vec!["item_1", "item_1"]);
}

/// Python: `test_scripted_model_function_argument_events_preserve_empty_output_item_id`.
#[tokio::test]
async fn function_argument_events_preserve_an_empty_output_item_id() {
    let model = ScriptedModel::with_steps(vec![out(vec![function_call_full(
        "lookup",
        "{}",
        "call_1",
        Some(String::new()),
        None,
    )])])
    .unwrap();
    let events = stream_events_of(&model, request()).await;
    let item_ids: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["type"] == "response.output_item.added" || e["type"] == "response.output_item.done"
        })
        .map(|e| &e["item"]["id"])
        .collect();
    let argument_ids: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["type"]
                .as_str()
                .is_some_and(|t| t.starts_with("response.function_call_arguments"))
        })
        .map(|e| &e["item_id"])
        .collect();
    assert_eq!(item_ids, vec!["", ""]);
    assert_eq!(argument_ids, vec!["", ""]);
}

/// Python: `test_scripted_model_function_argument_events_fall_back_to_call_id`.
#[tokio::test]
async fn function_argument_events_fall_back_to_the_call_id() {
    let model = ScriptedModel::with_steps(vec![out(vec![OutputItem::from_value(json!({
        "type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "call_1",
    }))])])
    .unwrap();
    let events = stream_events_of(&model, request()).await;
    let ids: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["type"]
                .as_str()
                .is_some_and(|t| t.starts_with("response.function_call_arguments"))
        })
        .map(|e| &e["item_id"])
        .collect();
    assert_eq!(ids, vec!["call_1", "call_1"]);
}

/// Python: `test_scripted_model_automatic_stream_uses_in_progress_added_payloads`.
#[tokio::test]
async fn automatic_stream_uses_in_progress_added_payloads() {
    let reasoning = OutputItem::from_value(json!({
        "type": "reasoning", "id": "reasoning_1", "status": "completed",
        "summary": [{"type": "summary_text", "text": "summary"}],
        "encrypted_content": "encrypted-reasoning",
    }));
    let message = assistant_message_with_id("hello", "message_1");
    let function = function_call_full(
        "lookup",
        r#"{"city":"Tokyo"}"#,
        "call_1",
        Some("function_1".into()),
        None,
    );
    let completed_items = vec![reasoning, message, function];
    let usage = Usage {
        requests: 3,
        input_tokens: 4,
        output_tokens: 5,
        total_tokens: 9,
        ..Default::default()
    };
    let model =
        ScriptedModel::with_steps(vec![out(completed_items.clone()).with_usage(usage)]).unwrap();

    let events = stream_events_of(&model, request()).await;

    for ty in ["response.created", "response.in_progress"] {
        let response = &of_type(&events, ty)[0]["response"];
        assert_eq!(response["output"], json!([]), "{ty}");
        assert_eq!(response["status"], "in_progress", "{ty}");
        assert!(response.get("usage").is_none_or(Value::is_null), "{ty}");
    }

    let added: Vec<&Value> = of_type(&events, "response.output_item.added")
        .into_iter()
        .map(|e| &e["item"])
        .collect();
    assert_eq!(added[0]["type"], "reasoning");
    assert!(added[0].get("encrypted_content").is_none_or(Value::is_null));
    assert_eq!(added[0]["summary"], json!([]));
    assert_eq!(added[0]["status"], "in_progress");
    assert_eq!(added[1]["type"], "message");
    assert_eq!(added[1]["content"], json!([]));
    assert_eq!(added[1]["status"], "in_progress");
    assert_eq!(added[2]["type"], "function_call");
    assert_eq!(added[2]["arguments"], "");
    assert_eq!(added[2]["status"], "in_progress");

    let summary_parts: Vec<&Value> = of_type(&events, "response.reasoning_summary_part.added");
    assert_eq!(
        summary_parts
            .iter()
            .map(|e| e["part"]["text"].clone())
            .collect::<Vec<_>>(),
        vec![json!("")]
    );
    let content_parts: Vec<&Value> = of_type(&events, "response.content_part.added");
    assert_eq!(
        content_parts
            .iter()
            .map(|e| e["part"]["text"].clone())
            .collect::<Vec<_>>(),
        vec![json!("")]
    );

    let done: Vec<Value> = of_type(&events, "response.output_item.done")
        .into_iter()
        .map(|e| e["item"].clone())
        .collect();
    let expected: Vec<Value> = completed_items.iter().map(OutputItem::to_value).collect();
    assert_eq!(done, expected);
    assert_eq!(done[0]["encrypted_content"], "encrypted-reasoning");
    let completed = &of_type(&events, "response.completed")[0]["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"], json!(expected));
    assert_eq!(
        completed["output"][0]["encrypted_content"],
        "encrypted-reasoning"
    );
    assert_eq!(completed["usage"]["input_tokens"], 4);
    assert_eq!(completed["usage"]["output_tokens"], 5);
    assert_eq!(completed["usage"]["total_tokens"], 9);
    assert_eq!(completed["usage"]["requests"], 3);
}

fn message_with_content(content: Vec<MessageContent>) -> OutputItem {
    let mut message = MessageOutput::text("");
    message.id = "message_1".into();
    message.status = Some("completed".into());
    message.content = content;
    OutputItem::Message(message)
}

/// Python: `test_scripted_model_automatic_stream_preserves_text_logprobs`.
#[tokio::test]
async fn automatic_stream_preserves_text_logprobs() {
    let logprobs = json!([
        {"token": "hello", "bytes": [104, 101, 108, 108, 111], "logprob": -0.25,
         "top_logprobs": [{"token": "hi", "bytes": [104, 105], "logprob": -1.5}]},
        {"token": "!", "bytes": [33], "logprob": -0.1, "top_logprobs": []},
    ]);
    let model = ScriptedModel::with_steps(vec![out(vec![message_with_content(vec![
        MessageContent::OutputText {
            text: "hello".into(),
            annotations: vec![],
            logprobs: Some(logprobs.clone()),
        },
    ])])])
    .unwrap();

    let events = stream_events_of(&model, request()).await;

    assert_eq!(
        of_type(&events, "response.content_part.added")[0]["part"]["logprobs"],
        json!([])
    );
    let delta = of_type(&events, "response.output_text.delta")[0];
    let done = of_type(&events, "response.output_text.done")[0];
    assert_eq!(delta["logprobs"], logprobs);
    assert_eq!(done["logprobs"], logprobs);
    assert_eq!(delta["logprobs"][0]["top_logprobs"][0]["token"], "hi");
    assert_eq!(delta["logprobs"][1]["top_logprobs"], json!([]));
    let completed = &of_type(&events, "response.completed")[0]["response"];
    assert_eq!(completed["output"][0]["content"][0]["logprobs"], logprobs);
}

/// Python: `test_scripted_model_automatic_stream_emits_text_annotation_events`.
#[tokio::test]
async fn automatic_stream_emits_text_annotation_events() {
    let annotations = vec![
        json!({"type": "url_citation", "end_index": 5, "start_index": 0, "title": "Example",
               "url": "https://example.test"}),
        json!({"type": "file_path", "file_id": "file_1", "index": 6}),
    ];
    let model = ScriptedModel::with_steps(vec![out(vec![message_with_content(vec![
        MessageContent::OutputText {
            text: "hello file".into(),
            annotations: annotations.clone(),
            logprobs: Some(json!([])),
        },
    ])])])
    .unwrap();

    let events = stream_events_of(&model, request()).await;

    let added = of_type(&events, "response.content_part.added")[0];
    let annotation_events = of_type(&events, "response.output_text.annotation.added");
    let text_done = of_type(&events, "response.output_text.done")[0];
    assert_eq!(added["part"]["annotations"], json!([]));
    assert_eq!(
        annotation_events
            .iter()
            .map(|e| e["annotation_index"].clone())
            .collect::<Vec<_>>(),
        vec![json!(0), json!(1)]
    );
    assert_eq!(
        annotation_events
            .iter()
            .map(|e| e["annotation"].clone())
            .collect::<Vec<_>>(),
        annotations
    );
    for event in &annotation_events {
        assert_eq!(event["item_id"], "message_1");
        assert_eq!(event["output_index"], 0);
        assert_eq!(event["content_index"], 0);
    }
    let seq = |e: &Value| e["sequence_number"].as_u64().unwrap();
    assert_eq!(seq(annotation_events[0]), seq(added) + 2);
    assert_eq!(seq(annotation_events[1]), seq(annotation_events[0]) + 1);
    assert_eq!(seq(text_done), seq(annotation_events[1]) + 1);
}

/// Python: `test_scripted_model_automatic_stream_emits_reasoning_content_events`.
#[tokio::test]
async fn automatic_stream_emits_reasoning_content_events() {
    let reasoning = OutputItem::from_value(json!({
        "type": "reasoning", "id": "reasoning_1", "status": "completed", "summary": [],
        "content": [{"type": "reasoning_text", "text": "think carefully"}],
    }));
    let model = ScriptedModel::with_steps(vec![out(vec![reasoning])]).unwrap();

    let events = stream_events_of(&model, request()).await;

    assert_eq!(
        of_type(&events, "response.output_item.added")[0]["item"]["content"],
        json!([])
    );
    let delta = of_type(&events, "response.reasoning_text.delta")[0];
    let done = of_type(&events, "response.reasoning_text.done")[0];
    assert_eq!(delta["item_id"], "reasoning_1");
    assert_eq!(delta["output_index"], 0);
    assert_eq!(delta["content_index"], 0);
    assert_eq!(delta["delta"], "think carefully");
    assert_eq!(done["item_id"], "reasoning_1");
    assert_eq!(done["output_index"], 0);
    assert_eq!(done["content_index"], 0);
    assert_eq!(done["text"], "think carefully");
    assert_eq!(
        done["sequence_number"].as_u64().unwrap(),
        delta["sequence_number"].as_u64().unwrap() + 1
    );
}

/// Python: `test_scripted_model_automatic_stream_marks_added_apply_patch_call_in_progress`.
#[tokio::test]
async fn automatic_stream_marks_added_apply_patch_call_in_progress() {
    let call = OutputItem::from_value(json!({
        "type": "apply_patch_call", "id": "apply_patch_1", "call_id": "call_1",
        "status": "completed",
        "operation": {"type": "update_file", "path": "test.md", "diff": "-old\n+new\n"},
    }));
    let model = ScriptedModel::with_steps(vec![out(vec![call])]).unwrap();

    let events = stream_events_of(&model, request()).await;

    let added = of_type(&events, "response.output_item.added")[0];
    let done = of_type(&events, "response.output_item.done")[0];
    let completed = of_type(&events, "response.completed")[0];
    assert_eq!(added["item"]["type"], "apply_patch_call");
    assert_eq!(added["item"]["status"], "in_progress");
    assert_eq!(done["item"]["status"], "completed");
    assert_eq!(
        completed["response"]["output"][0]["type"],
        "apply_patch_call"
    );
    assert_eq!(completed["response"]["output"][0]["status"], "completed");
}

/// Python: `test_scripted_model_automatic_stream_emits_refusal_content_events`.
#[tokio::test]
async fn automatic_stream_emits_refusal_content_events() {
    let model = ScriptedModel::with_steps(vec![out(vec![message_with_content(vec![
        MessageContent::Refusal {
            refusal: "I cannot help with that.".into(),
        },
    ])])])
    .unwrap();

    let events = stream_events_of(&model, request()).await;

    assert_eq!(
        types(&events),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.refusal.delta",
            "response.refusal.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let sequence: Vec<u64> = events
        .iter()
        .map(|e| e["sequence_number"].as_u64().unwrap())
        .collect();
    assert_eq!(sequence, (0..events.len() as u64).collect::<Vec<_>>());
    let content: Vec<&Value> = events
        .iter()
        .filter(|e| {
            matches!(
                e["type"].as_str().unwrap(),
                "response.content_part.added"
                    | "response.refusal.delta"
                    | "response.refusal.done"
                    | "response.content_part.done"
            )
        })
        .collect();
    for event in &content {
        assert_eq!(event["item_id"], "message_1");
        assert_eq!(event["output_index"], 0);
        assert_eq!(event["content_index"], 0);
    }
    assert_eq!(
        content[0]["part"],
        json!({"type": "refusal", "refusal": ""})
    );
    assert_eq!(content[1]["delta"], "I cannot help with that.");
    assert_eq!(content[2]["refusal"], "I cannot help with that.");
    assert_eq!(
        content[3]["part"],
        json!({"type": "refusal", "refusal": "I cannot help with that."})
    );
}

// ---------------------------------------------------------------------------------------------
// Unexpected calls and leftovers
// ---------------------------------------------------------------------------------------------

/// Python: `test_scripted_model_rejects_unexpected_call`.
#[tokio::test]
async fn rejects_unexpected_call() {
    let _serial = quiet().await;
    let model = Arc::new(ScriptedModel::new());

    let error = Runner::run(&agent_for(&model), "hi").await.unwrap_err();

    let text = error.to_string();
    assert!(text.contains("no scripted steps remain"), "{text}");
    assert!(text.contains("call #1"), "{text}");
    let recorded = model.calls();
    assert_eq!(recorded.len(), 1);
    assert!(!recorded[0].streamed);
    assert_eq!(
        recorded[0].request.input[0].0,
        json!({"content": "hi", "role": "user"})
    );
}

/// Python: `test_scripted_model_unexpected_streaming_call_records_streamed_attribute`.
#[tokio::test]
async fn unexpected_streaming_call_records_streamed_attribute() {
    let model = ScriptedModel::new();

    let error = match model.stream_response(request()).await {
        Err(error) => error,
        Ok(_) => panic!("an empty script cannot serve a call"),
    };

    assert!(
        error
            .to_string()
            .contains("Unexpected streaming model call #1"),
        "{error}"
    );
    let calls: Vec<ModelCall> = model.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].streamed);
}

/// Python: `test_scripted_model_reports_unconsumed_steps`.
#[tokio::test]
async fn reports_unconsumed_steps() {
    let model = ScriptedModel::with_steps(vec![out(vec![assistant_message("unused")])]).unwrap();
    match model.assert_complete() {
        Err(error @ ModelScriptError::UnconsumedModelSteps { remaining_steps: 1 }) => {
            assert!(
                error.to_string().contains("1 scripted model step"),
                "{error}"
            );
        }
        other => panic!("{other:?}"),
    }
}
