//! Port of `test_generation_metadata_respects_sensitive_capture`
//! (`tests/models/test_generation_metadata.py`) across the Chat Completions model, `LitellmModel`
//! and `AnyLLMModel`, streamed and not, with and without sensitive trace data: the request keeps
//! the caller's metadata, and the exported generation span carries it only when the run opts in
//! to sensitive data.
use std::sync::{Arc, Mutex};

use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel, LitellmModel};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use openai_agents_sdk::models::openai_client::OpenAIClient;
use openai_agents_sdk::tracing::{
    Span, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled,
};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SENTINEL: &str = "synthetic-request-metadata-sentinel";

/// Serve every request with a one-token completion, as JSON or as an SSE stream.
async fn serve(captured: Arc<Mutex<Vec<Value>>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let captured = captured.clone();
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
                    let chunk = json!({
                        "id": "chatcmpl_test", "created": 0, "model": "test-model",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "delta": {"role": "assistant", "content": "ok"}}],
                    });
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
async fn generation_metadata_respects_sensitive_capture() {
    for adapter in ["openai", "litellm", "any-llm"] {
        for stream in [false, true] {
            for include_data in [false, true] {
                let case = format!("{adapter} stream={stream} include_data={include_data}");
                let captured = Arc::new(Mutex::new(Vec::new()));
                let base = serve(captured.clone()).await;
                let model: Arc<dyn openai_agents_sdk::models::interface::Model> = match adapter {
                    "openai" => Arc::new(OpenAIChatCompletionsModel::new(
                        "test-model",
                        OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")),
                    )),
                    "litellm" => Arc::new(
                        LitellmModel::with_options(
                            "test-model",
                            Some(&base),
                            Some("sk-test"),
                            None,
                        )
                        .unwrap(),
                    ),
                    _ => Arc::new(
                        AnyLLMModel::new(
                            "openai/test-model",
                            Some(&base),
                            Some("sk-test"),
                            Some(AdapterApi::ChatCompletions),
                        )
                        .unwrap(),
                    ),
                };

                let metadata: std::collections::HashMap<String, String> =
                    [("request_label".to_string(), SENTINEL.to_string())].into();
                let settings = ModelSettings {
                    temperature: Some(0.25),
                    metadata: Some(metadata.clone()),
                    ..Default::default()
                };
                let trace_metadata: Map<String, Value> =
                    [("metadata".to_string(), json!("intentional-trace-label"))]
                        .into_iter()
                        .collect();
                let recorder = Arc::new(Recorder::default());
                set_tracing_disabled(false);
                set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);

                let agent = Agent::<()>::new("test")
                    .with_model_instance(model)
                    .with_model_settings(settings.clone());
                let config = RunConfig {
                    trace_include_sensitive_data: include_data,
                    trace_metadata: Some(trace_metadata.clone()),
                    ..Default::default()
                };
                let options = RunOptions::default().run_config(config);
                let output = if stream {
                    use futures::StreamExt;
                    let mut result = Runner::run_streamed_with_options(&agent, "hello", options);
                    {
                        let mut events = std::pin::pin!(result.stream_events());
                        while let Some(event) = events.next().await {
                            event.unwrap();
                        }
                    }
                    result.final_output_text()
                } else {
                    Runner::run_with_options(&agent, "hello", options)
                        .await
                        .unwrap()
                        .final_output_text()
                };
                set_trace_processors(vec![]);
                set_tracing_disabled(true);
                assert_eq!(output.as_deref(), Some("ok"), "{case}");

                let requests = captured.lock().unwrap().clone();
                assert_eq!(requests.len(), 1, "{case}");
                assert_eq!(
                    requests[0]["metadata"],
                    json!({"request_label": SENTINEL}),
                    "{case}: {}",
                    requests[0]
                );
                assert_eq!(settings.metadata, Some(metadata), "{case}");
                let traces = recorder.trace_metadata.lock().unwrap().clone();
                assert_eq!(traces, vec![Some(trace_metadata)], "{case}");

                let spans = recorder.spans.lock().unwrap().clone();
                let generations: Vec<&Value> = spans
                    .iter()
                    .filter(|s| {
                        s.pointer("/span_data/type").and_then(Value::as_str) == Some("generation")
                    })
                    .collect();
                assert_eq!(generations.len(), 1, "{case}: {spans:?}");
                let config_data = generations[0]
                    .pointer("/span_data/model_config")
                    .unwrap_or_else(|| panic!("{case}: no model_config in {}", generations[0]));
                assert_eq!(config_data["temperature"], json!(0.25), "{case}");
                if include_data {
                    assert_eq!(
                        config_data["metadata"],
                        json!({"request_label": SENTINEL}),
                        "{case}"
                    );
                } else {
                    assert!(
                        config_data.get("metadata").is_none(),
                        "{case}: {config_data}"
                    );
                    assert!(
                        !serde_json::to_string(&spans).unwrap().contains(SENTINEL),
                        "{case}: {spans:?}"
                    );
                }
            }
        }
    }
}
