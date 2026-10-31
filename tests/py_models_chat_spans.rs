//! Port of the generation-span tests in `tests/models/test_openai_chatcompletions.py` and
//! `tests/models/test_openai_chatcompletions_stream.py`.
//!
//! Trace processors are process-global, so every test serializes on one lock and clears the
//! shared in-memory processor first.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::models::interface::{Model, ModelRequest};
use openai_agents_sdk::models::{ModelTracing, OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---- shared processor ------------------------------------------------------------------

fn processor() -> &'static Arc<InMemoryTraceProcessor> {
    static P: OnceLock<Arc<InMemoryTraceProcessor>> = OnceLock::new();
    P.get_or_init(|| {
        let p = Arc::new(InMemoryTraceProcessor::new());
        set_trace_processors(vec![p.clone() as Arc<dyn TracingProcessor>]);
        p
    })
}

fn lock() -> &'static tokio::sync::Mutex<()> {
    static L: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    L.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// The single generation span exported since the processor was cleared.
fn generation() -> Value {
    let spans: Vec<Value> = processor()
        .items()
        .into_iter()
        .filter(|i| i["span_data"]["type"] == "generation")
        .collect();
    assert_eq!(spans.len(), 1, "{spans:?}");
    spans.into_iter().next().unwrap()
}

// ---- mock server -------------------------------------------------------------------------

/// Serve one reply per request (the last repeats): `(is_sse, body)`.
async fn serve(replies: Vec<(bool, String)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let queue = Arc::new(Mutex::new(VecDeque::from(replies)));
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let queue = queue.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let headers: HashMap<String, String> = text[..idx]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break;
                    }
                }
                let (sse, body) = {
                    let mut q = queue.lock().unwrap();
                    if q.len() > 1 {
                        q.pop_front().unwrap()
                    } else {
                        q.front().unwrap().clone()
                    }
                };
                let content_type = if sse {
                    "text/event-stream"
                } else {
                    "application/json"
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}/v1")
}

fn json_reply(v: Value) -> (bool, String) {
    (false, v.to_string())
}

fn sse_reply(chunks: &[Value]) -> (bool, String) {
    let mut body = String::new();
    for c in chunks {
        body.push_str(&format!("data: {c}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    (true, body)
}

fn completion(message: Value, finish_reason: &str, usage: Option<Value>) -> Value {
    let mut v = json!({"id": "resp-id", "created": 0, "model": "fake", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]});
    if let Some(u) = usage {
        v["usage"] = u;
    }
    v
}

fn chunk(delta: Value, finish_reason: Option<&str>, usage: Option<Value>) -> Value {
    let mut c = json!({"id": "chunk-id", "created": 1, "model": "fake",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

fn model(base: &str) -> OpenAIChatCompletionsModel {
    OpenAIChatCompletionsModel::new("gpt-4", OpenAIClient::new("sk-test").with_base_url(base))
}

fn request(tracing: ModelTracing) -> ModelRequest {
    ModelRequest {
        tracing,
        ..Default::default()
    }
}

/// Run a traced non-streaming call; returns the model result.
async fn traced_get_response(
    reply: (bool, String),
    tracing: ModelTracing,
) -> openai_agents_sdk::Result<openai_agents_sdk::items::ModelResponse> {
    let base = serve(vec![reply]).await;
    let model = model(&base);
    trace("test")
        .run(async move { model.get_response(request(tracing)).await })
        .await
}

// =========================================================================================
// test_openai_chatcompletions.py
// =========================================================================================

#[tokio::test]
async fn get_response_traces_synthesized_content_filter_refusal() {
    let _guard = lock().lock().await;
    processor().clear();
    let reply = json_reply(completion(
        json!({"role": "assistant", "content": null}),
        "content_filter",
        None,
    ));
    traced_get_response(reply, ModelTracing::Enabled)
        .await
        .unwrap();
    let span = generation();
    let output = &span["span_data"]["output"];
    assert_eq!(
        output[0]["content"][0]["refusal"],
        "Response withheld by the provider's content filter."
    );
}

#[tokio::test]
async fn get_response_traces_error_on_truncated_empty_turn() {
    let _guard = lock().lock().await;
    processor().clear();
    let reply = json_reply(completion(
        json!({"role": "assistant", "content": null}),
        "length",
        None,
    ));
    let err = traced_get_response(reply, ModelTracing::Enabled)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("finish_reason='length'"), "{err}");
    let span = generation();
    assert!(!span["error"].is_null(), "{span}");
    // The request is preserved on the span even though the call raised.
    assert_eq!(span["span_data"]["usage"]["requests"], 1);
}

#[tokio::test]
async fn get_response_traces_usage_on_truncated_empty_turn() {
    let _guard = lock().lock().await;
    processor().clear();
    let reply = json_reply(completion(
        json!({"role": "assistant", "content": null}),
        "length",
        Some(
            json!({"completion_tokens": 0, "prompt_tokens": 7, "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2}}),
        ),
    ));
    let err = traced_get_response(reply, ModelTracing::Enabled)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("finish_reason='length'"), "{err}");
    let usage = &generation()["span_data"]["usage"];
    assert_eq!(usage["requests"], 1);
    assert_eq!(usage["input_tokens"], 7);
    assert_eq!(usage["total_tokens"], 7);
}

// =========================================================================================
// test_openai_chatcompletions_stream.py
// =========================================================================================

async fn drain(model: &OpenAIChatCompletionsModel, tracing: ModelTracing) -> Result<(), String> {
    let model = model.clone();
    trace("test")
        .run(async move {
            let mut stream = model
                .stream_response(request(tracing))
                .await
                .map_err(|e| e.to_string())?;
            while let Some(item) = stream.next().await {
                item.map_err(|e| e.to_string())?;
            }
            Ok(())
        })
        .await
}

fn usageless_stream(usage: Option<Value>) -> (bool, String) {
    sse_reply(&[
        chunk(json!({"content": "Hello"}), None, usage),
        chunk(json!({}), Some("stop"), None),
    ])
}

#[tokio::test]
async fn streamed_span_records_the_request_when_provider_omits_usage() {
    let _guard = lock().lock().await;
    processor().clear();
    let base = serve(vec![usageless_stream(None)]).await;
    drain(&model(&base), ModelTracing::Enabled).await.unwrap();
    let usage = &generation()["span_data"]["usage"];
    assert_eq!(usage["requests"], 1);
    // The provider reported no tokens, so every total stays at zero.
    assert_eq!(usage["total_tokens"], 0);
}

#[tokio::test]
async fn stream_span_is_recorded_for_a_consumer_that_stops_at_the_terminal_event() {
    let _guard = lock().lock().await;
    for with_usage in [false, true] {
        for tracing in [ModelTracing::Enabled, ModelTracing::EnabledWithoutData] {
            processor().clear();
            let usage = with_usage.then(|| {
                json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                       "prompt_tokens_details": {"cached_tokens": 2},
                       "completion_tokens_details": {"reasoning_tokens": 3}})
            });
            let base = serve(vec![usageless_stream(usage)]).await;
            let model = model(&base);
            let completed = trace("test")
                .run(async move {
                    let mut stream = model.stream_response(request(tracing)).await.unwrap();
                    while let Some(event) = stream.next().await {
                        let event = event.unwrap();
                        if event["type"] == "response.completed" {
                            // Stopping here drops the stream; the span must already be
                            // recorded.
                            return Some(event);
                        }
                    }
                    None
                })
                .await
                .expect("the stream yielded a completed response");
            let span = generation();
            let u = &span["span_data"]["usage"];
            let n = |v: u64| if with_usage { v } else { 0 };
            assert_eq!(u["requests"], 1);
            assert_eq!(u["input_tokens"], n(7));
            assert_eq!(u["output_tokens"], n(5));
            assert_eq!(u["total_tokens"], n(12));
            assert_eq!(u["input_tokens_details"]["cached_tokens"], n(2));
            assert_eq!(u["output_tokens_details"]["reasoning_tokens"], n(3));
            if tracing == ModelTracing::Enabled {
                assert_eq!(span["span_data"]["output"], completed["response"]["output"]);
            } else {
                assert!(span["span_data"]["output"].is_null());
            }
            assert_eq!(!completed["response"]["usage"].is_null(), with_usage);
        }
    }
}

#[tokio::test]
async fn streamed_span_preserves_usage_on_length_truncation() {
    let _guard = lock().lock().await;
    processor().clear();
    let base = serve(vec![sse_reply(&[chunk(
        json!({}),
        Some("length"),
        Some(
            json!({"completion_tokens": 0, "prompt_tokens": 7, "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2}}),
        ),
    )])])
    .await;
    let err = drain(&model(&base), ModelTracing::Enabled)
        .await
        .unwrap_err();
    assert!(err.contains("finish_reason='length'"), "{err}");
    let span = generation();
    assert!(!span["error"].is_null(), "{span}");
    let usage = &span["span_data"]["usage"];
    assert_eq!(usage["requests"], 1);
    assert_eq!(usage["input_tokens"], 7);
    assert_eq!(usage["total_tokens"], 7);
}

#[tokio::test]
async fn streamed_span_counts_request_on_length_truncation_without_usage() {
    let _guard = lock().lock().await;
    processor().clear();
    let base = serve(vec![sse_reply(&[chunk(json!({}), Some("length"), None)])]).await;
    let err = drain(&model(&base), ModelTracing::Enabled)
        .await
        .unwrap_err();
    assert!(err.contains("finish_reason='length'"), "{err}");
    let usage = &generation()["span_data"]["usage"];
    assert_eq!(usage["requests"], 1);
    assert_eq!(usage["total_tokens"], 0);
}
