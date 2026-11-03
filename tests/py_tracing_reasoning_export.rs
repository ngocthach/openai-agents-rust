//! Port of `tests/tracing/test_reasoning_export.py`: a generation span keeps the model's
//! reasoning for the next turn, but the copy sent to the OpenAI traces endpoint has it removed.

#![allow(clippy::await_holding_lock)] // the lock serializes process-global tracing state for a whole test

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::tracing::processors::sanitize_for_openai;
use openai_agents_sdk::tracing::{
    DefaultTraceProvider, Span, TraceProvider, TracingExporter, TracingProcessor,
};
use openai_agents_sdk::{Agent, OpenAIChatCompletionsModel, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn tracing_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
enum Reply {
    Json(Value),
    Sse(Vec<String>),
}

/// Serve `replies` in order and record every request body.
async fn mock_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
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
                if let Some(idx) = text.find("\r\n\r\n") {
                    let len: usize = text[..idx]
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break serde_json::from_slice::<Value>(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                }
            };
            cap.lock().unwrap().push(body);
            let Some(reply) = replies.next() else { return };
            let resp = match reply {
                Reply::Json(v) => {
                    let b = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
                Reply::Sse(chunks) => {
                    let mut b = String::new();
                    for c in chunks {
                        b.push_str(&format!("data: {c}\n\n"));
                    }
                    b.push_str("data: [DONE]\n\n");
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
            };
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}/v1"), captured)
}

/// Keeps every finished span payload so a test can look up the generation span.
#[derive(Debug, Default)]
struct SpanCollector {
    spans: Mutex<Vec<Value>>,
}

impl SpanCollector {
    fn generations(&self) -> Vec<Value> {
        self.spans
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s["span_data"]["type"] == "generation")
            .cloned()
            .collect()
    }
}

impl TracingProcessor for SpanCollector {
    fn on_trace_start(&self, _t: &openai_agents_sdk::tracing::Trace) {}
    fn on_trace_end(&self, _t: &openai_agents_sdk::tracing::Trace) {}
    fn on_span_start(&self, _s: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let Some(payload) = span.export() {
            self.spans.lock().unwrap().push(payload);
        }
    }
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

fn install_collector() -> Arc<SpanCollector> {
    let collector = Arc::new(SpanCollector::default());
    let provider = Arc::new(DefaultTraceProvider::new());
    provider.set_disabled(false);
    provider.set_processors(vec![collector.clone() as Arc<dyn TracingProcessor>]);
    openai_agents_sdk::tracing::set_trace_provider(provider);
    collector
}

const SECRET: &str = "PRIVATE_REASONING_SENTINEL";

fn reasoning_value(field: &str) -> Value {
    if field == "thinking_blocks" {
        json!([{"type": "thinking", "thinking": SECRET, "signature": "test-signature"}])
    } else {
        json!(SECRET)
    }
}

// ---------------------------------------------------------------------------------------------
// test_runner_reasoning_is_omitted_from_export_but_preserved_for_replay
// ---------------------------------------------------------------------------------------------

async fn run_once(streamed: bool, field: &str) -> (Vec<Value>, String) {
    let value = reasoning_value(field);
    let reply = if streamed {
        let first = json!({
            "id": "test-completion", "created": 0, "model": "test-model",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", field: value}}],
        });
        let second = json!({
            "id": "test-completion", "created": 0, "model": "test-model",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Visible answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        });
        Reply::Sse(vec![first.to_string(), second.to_string()])
    } else {
        Reply::Json(json!({
            "id": "test-completion", "created": 0, "model": "test-model",
            "object": "chat.completion",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Visible answer", field: value},
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }))
    };

    let (base, _captured) = mock_server(vec![reply]).await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = Arc::new(
        OpenAIChatCompletionsModel::new("test-model", client)
            .with_should_replay_reasoning_content(|_| true),
    );
    let agent = Agent::<()>::new("Test").with_model_instance(model);
    let options = RunOptions::<()>::default().run_config({
        openai_agents_sdk::RunConfig::<()> {
            trace_include_sensitive_data: true,
            ..Default::default()
        }
    });

    if streamed {
        let mut result = Runner::run_streamed_with_options(&agent, "Question", options);
        let mut stream = Box::pin(result.stream_events());
        while stream.next().await.is_some() {}
        drop(stream);
        let replay =
            serde_json::to_string(&result.result().expect("completed run").to_input_list())
                .unwrap();
        (Vec::new(), replay)
    } else {
        let result = Runner::run_with_options(&agent, "Question", options)
            .await
            .unwrap();
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("Visible answer")
        );
        let replay = serde_json::to_string(&result.to_input_list()).unwrap();
        (Vec::new(), replay)
    }
}

#[tokio::test]
async fn runner_reasoning_is_omitted_from_export_but_preserved_for_replay() {
    let _lock = tracing_lock();
    for streamed in [false, true] {
        for field in ["reasoning", "reasoning_content", "thinking_blocks"] {
            let collector = install_collector();
            let (_, replay) = run_once(streamed, field).await;
            openai_agents_sdk::tracing::reset_trace_provider();

            assert!(
                replay.contains(SECRET),
                "{field}/streamed={streamed}: reasoning must survive for the next turn"
            );

            let generations = collector.generations();
            assert_eq!(
                generations.len(),
                1,
                "{field}/streamed={streamed}: one generation span"
            );
            let original = generations[0].clone();
            assert!(
                serde_json::to_string(&original).unwrap().contains(SECRET),
                "{field}/streamed={streamed}: the span itself keeps the reasoning"
            );

            let sent = sanitize_for_openai(original.clone());
            let sent_text = serde_json::to_string(&sent["span_data"]).unwrap();
            assert!(
                !sent_text.contains(SECRET),
                "{field}/streamed={streamed}: the exported copy must not carry the reasoning"
            );
            assert!(
                serde_json::to_string(&sent["span_data"]["output"])
                    .unwrap()
                    .contains("Visible answer"),
                "{field}/streamed={streamed}: the answer is still exported"
            );
            assert_eq!(sent["span_data"]["usage"]["output_tokens"], json!(3));
            // Filtering is applied to the exported copy, never to the span.
            assert_eq!(collector.generations()[0], original);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_export_filter_preserves_user_tool_and_custom_export_data
// ---------------------------------------------------------------------------------------------

fn assistant_with_reasoning(user_data: &Value) -> Value {
    json!({
        "role": "assistant",
        "reasoning": "PRIVATE_REASONING_SENTINEL",
        "thinking": "PRIVATE_THINKING_SENTINEL",
        "content": [
            {"type": "thinking", "thinking": "PRIVATE_INLINE_SENTINEL"},
            {"type": "text", "text": "Visible answer"},
        ],
        "tool_calls": [{"id": "call_test", "function": {"arguments": user_data.to_string()}}],
    })
}

fn generation_payload(input: Value, output: Value, usage: Value) -> Value {
    json!({
        "object": "trace.span",
        "id": "span_test",
        "trace_id": "trace_test",
        "parent_id": null,
        "span_data": {"type": "generation", "input": input, "output": output, "usage": usage},
    })
}

#[test]
fn the_export_filter_preserves_user_and_tool_supplied_data() {
    let user_data = json!({
        "type": "thinking", "thinking": "user supplied", "reasoning": "tool supplied"
    });
    let assistant = assistant_with_reasoning(&user_data);
    let payload = generation_payload(
        json!([{"role": "user", "content": [user_data]}, assistant]),
        json!([assistant]),
        json!({"input_tokens": 1, "output_tokens": 2, "reasoning_tokens": 1}),
    );

    let sent = sanitize_for_openai(payload.clone());
    let span_data = &sent["span_data"];
    let text = serde_json::to_string(span_data).unwrap();
    assert!(!text.contains("PRIVATE_REASONING_SENTINEL"));
    assert!(!text.contains("PRIVATE_THINKING_SENTINEL"));
    assert!(!text.contains("PRIVATE_INLINE_SENTINEL"));
    // A user message and a tool call carry the caller's own data, which is never filtered.
    assert_eq!(span_data["input"][0]["content"], json!([user_data]));
    assert_eq!(
        span_data["output"][0]["tool_calls"],
        assistant["tool_calls"]
    );
    assert!(
        serde_json::to_string(&span_data["usage"])
            .unwrap()
            .contains("reasoning_tokens")
    );
    // The filter never mutates the payload it was given.
    assert_eq!(
        payload["span_data"]["output"][0]["reasoning"],
        json!(SECRET)
    );
}

// A plain `#[test]`: the exporter drives its own runtime with `block_on`.
#[test]
fn a_custom_endpoint_receives_the_payload_unfiltered() {
    let _lock = tracing_lock();
    let user_data = json!({
        "type": "thinking", "thinking": "user supplied", "reasoning": "tool supplied"
    });
    let assistant = assistant_with_reasoning(&user_data);
    let payload = generation_payload(
        json!([{"role": "user", "content": [user_data]}, assistant]),
        json!([assistant]),
        json!({"input_tokens": 1, "output_tokens": 2, "reasoning_tokens": 1}),
    );

    let (endpoint, received) = ingest_server();
    openai_agents_sdk::tracing::BackendSpanExporter::new()
        .with_endpoint(endpoint)
        .with_api_key("sk-test")
        .export(vec![openai_agents_sdk::tracing::ExportItem {
            object: None,
            payload,
            api_key: Some("sk-test".into()),
        }]);

    let bodies = received.lock().unwrap().clone();
    let text = serde_json::to_string(&bodies[0]).unwrap();
    assert!(text.contains("PRIVATE_REASONING_SENTINEL"));
    assert!(text.contains("PRIVATE_THINKING_SENTINEL"));
    assert!(text.contains("PRIVATE_INLINE_SENTINEL"));
}

/// A blocking ingest endpoint; the exporter drives its own runtime, so this uses std sockets.
fn ingest_server() -> (String, Arc<Mutex<Vec<Value>>>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = received.clone();
    std::thread::spawn(move || {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        let mut buf = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            let Ok(n) = sock.read(&mut tmp) else { return };
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            let text = String::from_utf8_lossy(&buf).to_string();
            if let Some(idx) = text.find("\r\n\r\n") {
                let len: usize = text[..idx]
                    .lines()
                    .find_map(|l| {
                        l.to_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                if buf.len() >= idx + 4 + len {
                    sink.lock().unwrap().push(
                        serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null),
                    );
                    break;
                }
            }
        }
        let resp = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}";
        let _ = sock.write_all(resp.as_bytes());
        let _ = sock.flush();
    });
    (format!("http://{addr}/traces/ingest"), received)
}

// ---------------------------------------------------------------------------------------------
// test_public_generation_span_filters_declared_message_sequences
// ---------------------------------------------------------------------------------------------

#[test]
fn the_export_filter_drops_declared_reasoning_messages() {
    let secret = "PRIVATE_SEQUENCE_REASONING_SENTINEL";
    let assistant =
        json!({"role": "assistant", "content": "Visible answer", "reasoning_content": secret});
    let reasoning =
        json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": secret}]});
    let inputs = json!([
        {"role": "user", "content": "Question"},
        {"role": "tool", "content": "Tool result", "tool_call_id": "call_test"},
        assistant,
        reasoning,
    ]);
    let outputs = json!([assistant, reasoning]);
    let payload = generation_payload(inputs.clone(), outputs.clone(), Value::Null);

    let sent = sanitize_for_openai(payload.clone());
    let span_data = &sent["span_data"];
    assert!(!serde_json::to_string(span_data).unwrap().contains(secret));
    assert_eq!(
        span_data["input"],
        json!([
            {"role": "user", "content": "Question"},
            {"role": "tool", "content": "Tool result", "tool_call_id": "call_test"},
            {"role": "assistant", "content": "Visible answer"},
        ])
    );
    assert_eq!(
        span_data["output"],
        json!([{"role": "assistant", "content": "Visible answer"}])
    );
    // The span's own data is untouched.
    assert_eq!(payload["span_data"]["input"], inputs);
    assert_eq!(payload["span_data"]["output"], outputs);
}

// ---------------------------------------------------------------------------------------------
// test_export_filter_keeps_absent_generation_data_absent
// ---------------------------------------------------------------------------------------------

#[test]
fn the_export_filter_keeps_absent_generation_data_absent() {
    let payload = json!({
        "object": "trace.span",
        "id": "span_test",
        "trace_id": "trace_test",
        "parent_id": null,
        "span_data": {"type": "generation", "input": null, "output": null},
    });
    let sent = sanitize_for_openai(payload);
    assert_eq!(sent["span_data"]["input"], Value::Null);
    assert_eq!(sent["span_data"]["output"], Value::Null);
}

/// Usage is kept only on a generation span, and only when both token counts are numbers.
#[test]
fn the_export_filter_keeps_usage_only_on_a_valid_generation_span() {
    let function = json!({
        "object": "trace.span", "id": "s", "trace_id": "t", "parent_id": null,
        "span_data": {"type": "function", "name": "test", "input": null, "output": "{}",
                      "usage": {"input_tokens": 1, "output_tokens": 2}},
    });
    assert_eq!(
        sanitize_for_openai(function)["span_data"].get("usage"),
        None
    );

    let incomplete = generation_payload(Value::Null, Value::Null, json!({"input_tokens": 1}));
    assert_eq!(
        sanitize_for_openai(incomplete)["span_data"].get("usage"),
        None
    );

    let valid = generation_payload(
        Value::Null,
        Value::Null,
        json!({"input_tokens": 1, "output_tokens": 2}),
    );
    assert_eq!(
        sanitize_for_openai(valid)["span_data"]["usage"],
        json!({"input_tokens": 1, "output_tokens": 2})
    );
}
