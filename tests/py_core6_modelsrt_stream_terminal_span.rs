//! Ports of `test_stream_span_is_populated_before_yielding_completed`
//! (`tests/models/test_litellm_chatcompletions_stream.py`) and
//! `test_any_llm_chat_stream_preserves_trace_data_at_completed`
//! (`tests/models/test_any_llm_model.py`): a consumer that stops reading at the `completed`
//! event still finds the generation span populated with the exact usage and, only when tracing
//! includes data, the output.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel, LitellmModel};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::tracing::{
    Span, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled,
};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve every request with a one-token completion, as JSON or as an SSE stream.
async fn serve(captured: Arc<Mutex<Vec<Value>>>, usage: Option<Value>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let captured = captured.clone();
            let usage = usage.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let body = loop {
                    let Ok(n) = sock.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let len: usize = text[..idx]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                        .and_then(|(_, v)| v.trim().parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break serde_json::from_slice::<Value>(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                };
                let streamed = body["stream"] == json!(true);
                captured.lock().unwrap().push(body);
                let (content_type, payload) = if streamed {
                    let mut chunk = json!({
                        "id": "chatcmpl_test", "created": 0, "model": "test-model",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "delta": {"role": "assistant", "content": "Hello"}}],
                    });
                    if let Some(usage) = &usage {
                        chunk["usage"] = usage.clone();
                    }
                    (
                        "text/event-stream",
                        format!("data: {chunk}\n\ndata: [DONE]\n\n"),
                    )
                } else {
                    let completion = json!({
                        "id": "chatcmpl_test", "created": 0, "model": "test-model",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "ok"}}],
                    });
                    ("application/json", completion.to_string())
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\n\
                     connection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

#[derive(Debug, Default)]
struct Recorder {
    trace_metadata: Mutex<Vec<Option<Map<String, Value>>>>,
    spans: Mutex<Vec<Value>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, trace: &Trace) {
        let metadata = trace
            .export()
            .and_then(|v| v.get("metadata").and_then(Value::as_object).cloned());
        self.trace_metadata.lock().unwrap().push(metadata);
    }
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let Some(exported) = span.export() {
            self.spans.lock().unwrap().push(exported);
        }
    }
}

#[tokio::test]
async fn the_stream_span_is_populated_before_the_completed_event() {
    for adapter in ["litellm", "any-llm"] {
        for with_usage in [false, true] {
            for tracing in [ModelTracing::Enabled, ModelTracing::EnabledWithoutData] {
                let case = format!("{adapter} usage={with_usage} {tracing:?}");
                let usage = with_usage.then(|| {
                    json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                           "prompt_tokens_details": {"cached_tokens": 2},
                           "completion_tokens_details": {"reasoning_tokens": 3}})
                });
                let base = serve(Arc::new(Mutex::new(Vec::new())), usage).await;
                let model: Arc<dyn Model> = if adapter == "litellm" {
                    Arc::new(
                        LitellmModel::with_options(
                            "test-model",
                            Some(&base),
                            Some("sk-test"),
                            None,
                        )
                        .unwrap(),
                    )
                } else {
                    Arc::new(
                        AnyLLMModel::new(
                            "openrouter/openai/gpt-5.4-mini",
                            Some(&base),
                            Some("sk-test"),
                            Some(AdapterApi::ChatCompletions),
                        )
                        .unwrap(),
                    )
                };
                let recorder = Arc::new(Recorder::default());
                set_tracing_disabled(false);
                set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
                let completed = openai_agents_sdk::tracing::trace("terminal-span")
                    .run(async move {
                        let request = ModelRequest {
                            tracing,
                            ..Default::default()
                        };
                        let mut stream = model.stream_response(request).await.unwrap();
                        while let Some(event) = stream.next().await {
                            let event = event.unwrap();
                            if event["type"] == "response.completed" {
                                // Dropping the stream here must not lose the span data.
                                return Some(event);
                            }
                        }
                        None
                    })
                    .await
                    .unwrap_or_else(|| panic!("{case}: the stream yielded no completed response"));
                set_trace_processors(vec![]);
                set_tracing_disabled(true);

                let spans = recorder.spans.lock().unwrap().clone();
                let generation = spans
                    .iter()
                    .find(|s| {
                        s.pointer("/span_data/type").and_then(Value::as_str) == Some("generation")
                    })
                    .unwrap_or_else(|| panic!("{case}: no generation span in {spans:?}"));
                let u = &generation["span_data"]["usage"];
                let n = |v: u64| if with_usage { v } else { 0 };
                assert_eq!(u["requests"], 1, "{case}: {u}");
                assert_eq!(u["input_tokens"], n(7), "{case}");
                assert_eq!(u["output_tokens"], n(5), "{case}");
                assert_eq!(u["total_tokens"], n(12), "{case}");
                assert_eq!(u["input_tokens_details"]["cached_tokens"], n(2), "{case}");
                assert_eq!(
                    u["output_tokens_details"]["reasoning_tokens"],
                    n(3),
                    "{case}"
                );
                if tracing == ModelTracing::Enabled {
                    assert_eq!(
                        generation["span_data"]["output"], completed["response"]["output"],
                        "{case}"
                    );
                } else {
                    assert!(generation["span_data"]["output"].is_null(), "{case}");
                }
                assert_eq!(
                    !completed["response"]["usage"].is_null(),
                    with_usage,
                    "{case}"
                );
            }
        }
    }
}
