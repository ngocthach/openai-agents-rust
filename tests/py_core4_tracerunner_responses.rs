//! Port of the Python SDK's `tests/test_responses_tracing.py`: the `response` span that
//! `OpenAIResponsesModel` records around a non-streamed and a streamed call, with tracing on,
//! with data tracing off, and with tracing disabled.
//!
//! Python replaces `_fetch_response` with a dummy. Rust talks to a local HTTP server that replies
//! with the same response objects, so the model's own request, parse and span code all run.

#[path = "py_core4_tracerunner_support/mod.rs"]
mod support;

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::Result;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::models::{OpenAIClient, OpenAIResponsesModel};
use openai_agents_sdk::tracing::{SpanData, TracingProcessor, trace};
use serde_json::{Value, json};
use support::{
    ModelReply, Pipeline, SpanRecorder, dummy_usage, lock_async, official_client, responses_object,
    start_model_endpoint, start_tls_model_endpoint,
};

fn model_for(base_url: &str) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new(
        "test-model",
        OpenAIClient::new("sk-test").with_base_url(base_url),
    )
}

fn request(tracing: ModelTracing) -> ModelRequest {
    ModelRequest {
        system_instructions: Some("instr".into()),
        input: vec![InputItem::user("input")],
        tracing,
        ..Default::default()
    }
}

/// The `response` span of the expected normalized tree, with the usage of `dummy_usage`.
fn response_child(response_id: &str) -> Value {
    json!({
        "type": "response",
        "data": {
            "response_id": response_id,
            "usage": {
                "requests": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    })
}

/// The `response` span of a stream whose completed response carries no usage: one request, zeros.
fn zero_usage_response_child(response_id: &str) -> Value {
    json!({
        "type": "response",
        "data": {
            "response_id": response_id,
            "usage": {
                "requests": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    })
}

/// A `response.completed` event whose response has no usage, as Python's dummy returns.
fn completed_event(response_id: &str) -> Value {
    json!({
        "type": "response.completed",
        "sequence_number": 0,
        "response": responses_object(response_id, "completed", None, None),
    })
}

#[tokio::test]
async fn get_response_creates_trace() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let endpoint = start_model_endpoint(vec![ModelReply::Json(responses_object(
        "dummy-id",
        "completed",
        None,
        Some(dummy_usage()),
    ))]);
    let model = model_for(&endpoint.base_url);

    trace("test")
        .run(async {
            model
                .get_response(request(ModelTracing::Enabled))
                .await
                .unwrap();
        })
        .await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({"workflow_name": "test", "children": [response_child("dummy-id")]})]
    );
}

#[tokio::test]
async fn disable_tracing_does_not_create_span() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let endpoint = start_model_endpoint(vec![ModelReply::Json(responses_object(
        "dummy-id",
        "completed",
        None,
        Some(dummy_usage()),
    ))]);
    let model = model_for(&endpoint.base_url);

    trace("test")
        .run(async {
            model
                .get_response(request(ModelTracing::Disabled))
                .await
                .unwrap();
        })
        .await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({"workflow_name": "test"})]
    );
    assert!(rec.ordered_spans().is_empty());
}

#[tokio::test]
async fn stream_response_creates_trace() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let endpoint =
        start_model_endpoint(vec![ModelReply::Sse(vec![completed_event("dummy-id-123")])]);
    let model = model_for(&endpoint.base_url);

    trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Enabled))
                .await
                .unwrap();
            while let Some(event) = stream.next().await {
                event.unwrap();
            }
        })
        .await;

    assert_eq!(
        rec.normalized(false, false),
        vec![
            json!({"workflow_name": "test", "children": [zero_usage_response_child("dummy-id-123")]})
        ]
    );
}

#[tokio::test]
async fn stream_response_failed_or_incomplete_terminal_event_creates_trace() {
    for terminal_event_type in ["response.failed", "response.incomplete"] {
        let _lock = lock_async().await;
        let rec = SpanRecorder::new();
        let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
        let status = &terminal_event_type["response.".len()..];
        // Python's dummy response has no status, so the error text carries no `status=` detail.
        let mut terminal_response = responses_object("dummy-id-terminal", status, None, None);
        terminal_response["status"] = Value::Null;
        let endpoint = start_model_endpoint(vec![ModelReply::Sse(vec![json!({
            "type": terminal_event_type,
            "sequence_number": 0,
            "response": terminal_response,
        })])]);
        let model = model_for(&endpoint.base_url);

        let outcome: Result<()> = trace("test")
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
        let err = outcome.expect_err("terminal failure must surface");
        assert!(
            err.to_string().contains(terminal_event_type),
            "{terminal_event_type}: {err}"
        );

        assert_eq!(
            rec.normalized(false, false),
            vec![json!({
                "workflow_name": "test",
                "children": [{
                    "type": "response",
                    "error": {
                        "message": "Error streaming response",
                        "data": {
                            "error": format!(
                                "Responses stream ended with terminal event `{terminal_event_type}`."
                            )
                        },
                    },
                }],
            })],
            "{terminal_event_type}"
        );
    }
}

#[tokio::test]
async fn non_data_tracing_preserves_response_id_without_response() {
    if !support::openssl_available() {
        eprintln!("{}", support::OPENSSL_SKIP_MESSAGE);
        return;
    }
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let endpoint = start_tls_model_endpoint(vec![ModelReply::Json(responses_object(
        "dummy-id",
        "completed",
        None,
        Some(dummy_usage()),
    ))]);
    let model = OpenAIResponsesModel::new("test-model", official_client(&endpoint));

    trace("test")
        .run(async {
            model
                .get_response(request(ModelTracing::EnabledWithoutData))
                .await
                .unwrap();
        })
        .await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({"workflow_name": "test", "children": [response_child("dummy-id")]})]
    );
    let spans = rec.ordered_spans();
    let [span] = spans.as_slice() else {
        panic!("one span, got {}", spans.len())
    };
    let SpanData::Response {
        response, input, ..
    } = span.data()
    else {
        panic!("response span expected")
    };
    assert!(response.is_none());
    assert!(input.is_none());
}

#[tokio::test]
async fn stream_non_data_tracing_preserves_response_id_without_response() {
    if !support::openssl_available() {
        eprintln!("{}", support::OPENSSL_SKIP_MESSAGE);
        return;
    }
    for close_at_completed in [false, true] {
        let _lock = lock_async().await;
        let rec = SpanRecorder::new();
        let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
        let endpoint =
            start_tls_model_endpoint(vec![ModelReply::Sse(vec![completed_event("dummy-id-123")])]);
        let model = OpenAIResponsesModel::new("test-model", official_client(&endpoint));

        trace("test")
            .run(async {
                let mut stream = model
                    .stream_response(request(ModelTracing::EnabledWithoutData))
                    .await
                    .unwrap();
                if close_at_completed {
                    // The consumer stops at the terminal event and closes the stream.
                    let event = stream.next().await.unwrap().unwrap();
                    assert_eq!(event["type"], "response.completed");
                    drop(stream);
                } else {
                    while let Some(event) = stream.next().await {
                        event.unwrap();
                    }
                }
            })
            .await;

        assert_eq!(
            rec.normalized(false, false),
            vec![json!({
                "workflow_name": "test",
                "children": [zero_usage_response_child("dummy-id-123")],
            })],
            "close_at_completed={close_at_completed}"
        );
        let spans = rec.ordered_spans();
        let [span] = spans.as_slice() else {
            panic!("one span, got {}", spans.len())
        };
        let SpanData::Response {
            response, input, ..
        } = span.data()
        else {
            panic!("response span expected")
        };
        assert!(response.is_none());
        assert!(input.is_none());
    }
}

#[tokio::test]
async fn non_data_tracing_omits_custom_endpoint_response_id() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let provider_response_id = "tenant-customer-123";
    let endpoint = start_model_endpoint(vec![ModelReply::Json(responses_object(
        provider_response_id,
        "completed",
        None,
        Some(dummy_usage()),
    ))]);
    let model = model_for(&endpoint.base_url);

    let response = trace("test")
        .run(async {
            model
                .get_response(request(ModelTracing::EnabledWithoutData))
                .await
                .unwrap()
        })
        .await;

    assert_eq!(response.response_id.as_deref(), Some(provider_response_id));
    let spans = rec.ordered_spans();
    let [span] = spans.as_slice() else {
        panic!("one span, got {}", spans.len())
    };
    let SpanData::Response {
        response_id,
        usage,
        response,
        input,
    } = span.data()
    else {
        panic!("response span expected")
    };
    assert_eq!(response_id, None);
    assert!(response.is_none());
    assert!(input.is_none());
    assert!(usage.is_some());
}

#[tokio::test]
async fn stream_non_data_tracing_omits_custom_endpoint_response_id() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let provider_response_id = "tenant-customer-123";
    let endpoint = start_model_endpoint(vec![ModelReply::Sse(vec![json!({
        "type": "response.completed",
        "sequence_number": 0,
        "response": responses_object(provider_response_id, "completed", None, None),
    })])]);
    let model = model_for(&endpoint.base_url);

    let events: Vec<Value> = trace("test")
        .run(async {
            let stream = model
                .stream_response(request(ModelTracing::EnabledWithoutData))
                .await
                .unwrap();
            stream.map(|e| e.unwrap()).collect().await
        })
        .await;

    assert_eq!(events.last().unwrap()["type"], "response.completed");
    assert_eq!(
        events.last().unwrap()["response"]["id"],
        provider_response_id
    );
    let spans = rec.ordered_spans();
    let [span] = spans.as_slice() else {
        panic!("one span, got {}", spans.len())
    };
    let SpanData::Response {
        response_id,
        usage,
        response,
        input,
    } = span.data()
    else {
        panic!("response span expected")
    };
    assert_eq!(response_id, None);
    assert!(response.is_none());
    assert!(input.is_none());
    assert!(usage.is_some());
}

#[tokio::test]
async fn stream_disabled_tracing_doesnt_create_span() {
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let endpoint =
        start_model_endpoint(vec![ModelReply::Sse(vec![completed_event("dummy-id-123")])]);
    let model = model_for(&endpoint.base_url);

    trace("test")
        .run(async {
            let mut stream = model
                .stream_response(request(ModelTracing::Disabled))
                .await
                .unwrap();
            while let Some(event) = stream.next().await {
                event.unwrap();
            }
        })
        .await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({"workflow_name": "test"})]
    );
    assert!(rec.ordered_spans().is_empty());
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn stream_non_data_tracing_preserves_id_for_https_official_websocket() {
    use openai_agents_sdk::models::openai_responses_ws::OpenAIResponsesWSModel;

    if !support::openssl_available() {
        eprintln!("{}", support::OPENSSL_SKIP_MESSAGE);
        return;
    }
    let _lock = lock_async().await;
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let provider_response_id = "resp-official-ws";
    // The model dials a local secure WebSocket server for the official host and trusts its
    // throwaway certificate; the URL and the client's base URL stay the official ones.
    let endpoint = support::start_tls_ws_endpoint(vec![completed_event(provider_response_id)]);
    let model = OpenAIResponsesWSModel::new("test-model", OpenAIClient::new("sk-test"))
        .with_websocket_base_url("https://api.openai.com/v1")
        .with_test_transport(endpoint.addr, support::trusting_client_config(&endpoint));
    assert!(!model.supports_default_prompt_cache_key());

    let events: Vec<Value> = trace("test")
        .run(async {
            let stream = model
                .stream_response(request(ModelTracing::EnabledWithoutData))
                .await
                .unwrap();
            stream.map(|e| e.unwrap()).collect().await
        })
        .await;

    assert_eq!(events.last().unwrap()["type"], "response.completed");
    assert_eq!(
        events.last().unwrap()["response"]["id"],
        provider_response_id
    );
    let spans = rec.ordered_spans();
    let [span] = spans.as_slice() else {
        panic!("one span, got {}", spans.len())
    };
    let SpanData::Response {
        response_id,
        usage,
        response,
        input,
    } = span.data()
    else {
        panic!("response span expected")
    };
    assert_eq!(response_id.as_deref(), Some(provider_response_id));
    assert!(response.is_none());
    assert!(input.is_none());
    assert!(usage.is_some());
}
