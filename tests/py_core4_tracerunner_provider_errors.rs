//! Port of the Python SDK's `tests/test_provider_span_errors.py`: every model provider records a
//! failed model call on its own span (`generation` for Chat Completions, `response` for the
//! Responses API), and a streamed terminal failure is recorded even when the consumer stops at the
//! terminal event.
//!
//! Python patches `_fetch_response` to raise. Rust talks to a local HTTP server that answers with
//! an error status or a terminal stream event, so the provider's real request and error paths run.

#[path = "py_core4_tracerunner_support/mod.rs"]
mod support;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel, LitellmModel};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::models::{OpenAIChatCompletionsModel, OpenAIClient, OpenAIResponsesModel};
use openai_agents_sdk::tracing::{
    REDACTED_TRACE_ERROR_MESSAGE, TracingProcessor, generation_span, record_model_error_on_span,
    trace,
};
use openai_agents_sdk::{Agent, AgentsError, RunOptions, Runner};
use serde_json::{Value, json};
use support::{
    ModelEndpoint, ModelReply, Pipeline, SpanRecorder, lock_async, responses_object,
    start_model_endpoint,
};

/// The provider's error body, carrying `text` as its message.
fn provider_error(text: &str) -> Value {
    json!({"error": {"message": text, "type": "server_error", "code": null, "param": null}})
}

fn request(tracing: ModelTracing) -> ModelRequest {
    ModelRequest {
        system_instructions: None,
        input: vec![InputItem::user("hi")],
        tracing,
        ..Default::default()
    }
}

/// The error recorded on the first span of `span_type`, if any.
fn span_error(rec: &SpanRecorder, span_type: &str) -> Option<Value> {
    rec.ordered_spans().into_iter().find_map(|span| {
        let export = span.export()?;
        (export["span_data"]["type"] == span_type)
            .then(|| export["error"].clone())
            .filter(|e| !e.is_null())
    })
}

fn chat_model(base_url: &str) -> OpenAIChatCompletionsModel {
    OpenAIChatCompletionsModel::new("gpt-4", OpenAIClient::new("test").with_base_url(base_url))
}

fn responses_model(base_url: &str) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new("gpt-4", OpenAIClient::new("test").with_base_url(base_url))
}

/// Install a recorder for one test. The lock keeps other tests from touching the global pipeline.
async fn recording() -> (
    tokio::sync::MutexGuard<'static, ()>,
    Arc<SpanRecorder>,
    Pipeline,
) {
    let lock = lock_async().await;
    let rec = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    (lock, rec, pipeline)
}

#[tokio::test]
async fn chatcompletions_get_response_records_span_error() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("upstream exploded"),
    )]);
    let model = chat_model(&endpoint.base_url);

    let outcome = trace("test")
        .run(async { model.get_response(request(ModelTracing::Enabled)).await })
        .await;
    assert!(outcome.is_err());

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["message"], "Error getting response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("upstream exploded"),
        "{error}"
    );
}

#[tokio::test]
async fn chatcompletions_stream_response_records_span_error() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("stream exploded"),
    )]);
    let model = chat_model(&endpoint.base_url);

    let outcome: openai_agents_sdk::Result<()> = trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Enabled))
                .await?;
            while let Some(event) = stream.next().await {
                event?;
            }
            Ok(())
        })
        .await;
    assert!(outcome.is_err());

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["message"], "Error streaming response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("stream exploded"),
        "{error}"
    );
}

#[tokio::test]
async fn chatcompletions_span_error_is_redacted_without_sensitive_data() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("secret-connection-string"),
    )]);
    let model = chat_model(&endpoint.base_url);

    let outcome = trace("test")
        .run(async {
            model
                .get_response(request(ModelTracing::EnabledWithoutData))
                .await
        })
        .await;
    assert!(outcome.is_err());

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["data"]["error"], REDACTED_TRACE_ERROR_MESSAGE);
    assert!(!error.to_string().contains("secret-connection-string"));
}

#[tokio::test]
async fn litellm_get_response_records_span_error() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("litellm exploded"),
    )]);
    let model = LitellmModel::with_options("gpt-4", Some(&endpoint.base_url), Some("test"), None)
        .expect("litellm model");

    let outcome = trace("test")
        .run(async { model.get_response(request(ModelTracing::Enabled)).await })
        .await;
    assert!(outcome.is_err());

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["message"], "Error getting response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("litellm exploded"),
        "{error}"
    );
}

#[tokio::test]
async fn litellm_stream_response_records_span_error() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("litellm stream exploded"),
    )]);
    let model = LitellmModel::with_options("gpt-4", Some(&endpoint.base_url), Some("test"), None)
        .expect("litellm model");

    let outcome: openai_agents_sdk::Result<()> = trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Enabled))
                .await?;
            while let Some(event) = stream.next().await {
                event?;
            }
            Ok(())
        })
        .await;
    assert!(outcome.is_err());

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["message"], "Error streaming response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("litellm stream exploded"),
        "{error}"
    );
}

/// Drive one any-llm call, of the four Python cases, against a server that answers with an error.
async fn any_llm_error_case(api: AdapterApi, streaming: bool, span_type: &str, message: &str) {
    let (_lock, rec, _p) = recording().await;
    let endpoint: ModelEndpoint = start_model_endpoint(vec![ModelReply::Status(
        500,
        provider_error("any_llm exploded"),
    )]);
    let model = AnyLLMModel::new(
        "openai/gpt-4",
        Some(&endpoint.base_url),
        Some("test"),
        Some(api),
    )
    .expect("any-llm model");

    let outcome: openai_agents_sdk::Result<()> = trace("test")
        .run(async {
            if streaming {
                let mut stream = model
                    .stream_response(request(ModelTracing::Enabled))
                    .await?;
                while let Some(event) = stream.next().await {
                    event?;
                }
                Ok(())
            } else {
                model
                    .get_response(request(ModelTracing::Enabled))
                    .await
                    .map(|_| ())
            }
        })
        .await;
    assert!(outcome.is_err(), "{span_type} {message}");

    let error =
        span_error(&rec, span_type).unwrap_or_else(|| panic!("{span_type} span carried no error"));
    assert_eq!(error["message"], message);
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("any_llm exploded"),
        "{error}"
    );
}

#[tokio::test]
async fn any_llm_records_span_error_for_each_api_and_mode() {
    any_llm_error_case(
        AdapterApi::Responses,
        false,
        "response",
        "Error getting response",
    )
    .await;
    any_llm_error_case(
        AdapterApi::Responses,
        true,
        "response",
        "Error streaming response",
    )
    .await;
    any_llm_error_case(
        AdapterApi::ChatCompletions,
        false,
        "generation",
        "Error getting response",
    )
    .await;
    any_llm_error_case(
        AdapterApi::ChatCompletions,
        true,
        "generation",
        "Error streaming response",
    )
    .await;
}

/// Counts how often `Display` runs, so a test can tell whether a path stringified the error.
struct SideEffectingError {
    str_calls: AtomicUsize,
}

impl SideEffectingError {
    fn new() -> Self {
        SideEffectingError {
            str_calls: AtomicUsize::new(0),
        }
    }
}

impl fmt::Display for SideEffectingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.str_calls.fetch_add(1, Ordering::SeqCst);
        f.write_str("sensitive detail")
    }
}

impl fmt::Debug for SideEffectingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SideEffectingError")
    }
}

/// A `Display` that always fails, as a provider exception with a broken `__str__` does.
struct BrokenDisplay;

impl fmt::Display for BrokenDisplay {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}

impl fmt::Debug for BrokenDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BrokenDisplay")
    }
}

#[tokio::test]
async fn redacted_tracing_does_not_stringify_the_exception() {
    let (_lock, rec, _p) = recording().await;
    let original = SideEffectingError::new();
    trace("test")
        .run(async {
            let span = generation_span(None, None, None, false);
            span.start();
            record_model_error_on_span(&span, "Error getting response", &original, false);
            span.finish();
        })
        .await;

    assert_eq!(original.str_calls.load(Ordering::SeqCst), 0);
    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["data"]["error"], REDACTED_TRACE_ERROR_MESSAGE);
}

#[tokio::test]
async fn sensitive_tracing_stringifies_once() {
    let (_lock, rec, _p) = recording().await;
    let original = SideEffectingError::new();
    trace("test")
        .run(async {
            let span = generation_span(None, None, None, false);
            span.start();
            record_model_error_on_span(&span, "Error getting response", &original, true);
            span.finish();
        })
        .await;

    assert_eq!(original.str_calls.load(Ordering::SeqCst), 1);
    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["data"]["error"], "sensitive detail");
}

#[tokio::test]
async fn broken_str_preserves_the_provider_exception() {
    for include_sensitive_data in [true, false] {
        let (_lock, rec, _p) = recording().await;
        let original = BrokenDisplay;
        trace("test")
            .run(async {
                let span = generation_span(None, None, None, false);
                span.start();
                record_model_error_on_span(
                    &span,
                    "Error getting response",
                    &original,
                    include_sensitive_data,
                );
                span.finish();
            })
            .await;

        assert!(
            span_error(&rec, "generation").is_some(),
            "include_sensitive_data={include_sensitive_data}"
        );
    }
}

#[tokio::test]
async fn openai_responses_records_terminal_failure_when_consumer_stops() {
    let (_lock, rec, _p) = recording().await;
    let terminal = json!({
        "type": "response.failed",
        "sequence_number": 0,
        "response": responses_object("resp-failed", "failed", None, None),
    });
    let endpoint = start_model_endpoint(vec![ModelReply::Sse(vec![terminal])]);
    let model = responses_model(&endpoint.base_url);

    trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Enabled))
                .await
                .unwrap();
            let first = stream.next().await.expect("terminal event").unwrap();
            assert_eq!(first["type"], "response.failed");
            // The consumer stops at the terminal event and closes the stream.
            drop(stream);
        })
        .await;

    let error = span_error(&rec, "response").expect("response span carried no error");
    assert_eq!(error["message"], "Error streaming response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("response.failed"),
        "{error}"
    );
}

#[tokio::test]
async fn any_llm_responses_records_terminal_failure_when_consumer_stops() {
    let (_lock, rec, _p) = recording().await;
    let terminal = json!({
        "type": "response.failed",
        "sequence_number": 0,
        "response": responses_object("resp-failed", "failed", None, None),
    });
    let endpoint = start_model_endpoint(vec![ModelReply::Sse(vec![terminal])]);
    let model = AnyLLMModel::new(
        "openai/gpt-4",
        Some(&endpoint.base_url),
        Some("test"),
        Some(AdapterApi::Responses),
    )
    .expect("any-llm model");

    trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Enabled))
                .await
                .unwrap();
            let first = stream.next().await.expect("terminal event").unwrap();
            assert_eq!(first["type"], "response.failed");
            drop(stream);
        })
        .await;

    let error = span_error(&rec, "response").expect("response span carried no error");
    assert_eq!(error["message"], "Error streaming response");
    assert!(
        error["data"]["error"]
            .as_str()
            .unwrap()
            .contains("response.failed"),
        "{error}"
    );
}

/// A model call that the runner times out records the timeout on its generation span, and the span
/// is still exported. Python cancels the model task and marks it, so `model_span_errors` records it.
#[tokio::test]
async fn marked_model_timeout_cancellation_records_span_error() {
    let (_lock, rec, _p) = recording().await;
    let endpoint = start_model_endpoint(vec![ModelReply::Stall(std::time::Duration::from_secs(2))]);
    let agent = Agent::<()>::new("timeout")
        .with_model_instance(Arc::new(chat_model(&endpoint.base_url)))
        .with_model_settings(ModelSettings::default().with_timeout(0.01));

    let outcome = Runner::run_with_options(&agent, "hi", RunOptions::new(())).await;
    let err = outcome.expect_err("the call must time out");
    assert!(
        matches!(err, AgentsError::ModelTimeoutError { .. }),
        "{err}"
    );

    let error = span_error(&rec, "generation").expect("generation span carried no error");
    assert_eq!(error["message"], "Error getting response");
    assert_eq!(
        error["data"]["error"],
        "Model call timed out after 0.01 seconds."
    );
}
