//! Parity port of `tests/models/test_any_llm_model.py`.
//!
//! Python drives a `FakeAnyLLMProvider` and reads the parameter objects any-llm's private
//! provider API received. Rust resolves `provider/model` against the endpoint registry and posts
//! OpenAI-shaped requests itself (see the module doc of `src/extensions/models/mod.rs`), so the
//! same expectations are asserted against the captured HTTP traffic of a local mock server.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel, AnyLLMProvider};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelProvider, ModelRequest, ModelTracing};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Local mock server
// ---------------------------------------------------------------------------

/// A canned reply: a JSON body, or a sequence of SSE events.
#[derive(Clone, Debug)]
enum Reply {
    Json(Value),
    Sse(Vec<Value>),
}

/// One captured request.
#[derive(Clone, Debug)]
struct Captured {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Captured {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

async fn mock_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Captured>>>) {
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
            let request = loop {
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
                let head = &text[..idx];
                let headers: Vec<(String, String)> = head
                    .lines()
                    .skip(1)
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                    .collect();
                let len: usize = headers
                    .iter()
                    .find(|(k, _)| k == "content-length")
                    .and_then(|(_, v)| v.parse().ok())
                    .unwrap_or(0);
                if buf.len() < idx + 4 + len {
                    continue;
                }
                break Captured {
                    path: head
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or_default()
                        .to_string(),
                    headers,
                    body: serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                        .unwrap_or(Value::Null),
                };
            };
            cap.lock().unwrap().push(request);
            let resp = match replies.next().expect("unexpected request") {
                Reply::Json(v) => {
                    let b = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
                Reply::Sse(events) => {
                    let mut b = String::new();
                    for e in events {
                        b.push_str(&format!(
                            "event: {}\ndata: {}\n\n",
                            e["type"].as_str().unwrap_or("message"),
                            e
                        ));
                    }
                    b.push_str("data: [DONE]\n\n");
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
            };
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}"), captured)
}

/// Provider resolution reads process-wide environment variables.
async fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn chat_completion(text: &str) -> Value {
    json!({
        "id": "chatcmpl_123",
        "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn responses_payload(text: &str) -> Value {
    json!({
        "id": "resp_123", "object": "response", "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}}
    })
}

fn request_with(settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem(json!({"role": "user", "content": "hi"}))],
        model_settings: settings,
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

/// Add the function tool or handoff a parallel-tool-calls case calls for.
fn add_tools(request: &mut ModelRequest, source: &str) {
    use openai_agents_sdk::handoffs::handoff;
    use openai_agents_sdk::models::interface::ToolSpec;
    use openai_agents_sdk::tool::FunctionTool;

    match source {
        "function" => {
            let tool = FunctionTool::<()>::new("test_tool", "", |_c, _a: NoArgs| async move {
                Ok("ok".to_string())
            });
            request.tools.push(ToolSpec::Function(tool.to_spec()));
        }
        "handoff" => request
            .handoffs
            .push(handoff::<()>(Agent::<()>::new("handoff")).to_spec()),
        _ => {}
    }
}

fn chat_model(name: &str, base_url: &str) -> AnyLLMModel {
    AnyLLMModel::new(
        name,
        Some(base_url),
        Some("sk-test"),
        Some(AdapterApi::ChatCompletions),
    )
    .unwrap()
}

fn responses_model(name: &str, base_url: &str) -> AnyLLMModel {
    AnyLLMModel::new(
        name,
        Some(base_url),
        Some("sk-test"),
        Some(AdapterApi::Responses),
    )
    .unwrap()
}

fn output_values(response: &openai_agents_sdk::items::ModelResponse) -> Vec<Value> {
    response.output.iter().map(|o| o.to_value()).collect()
}

fn first_text(response: &openai_agents_sdk::items::ModelResponse) -> String {
    output_values(response)
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .find(|p| p["type"] == "output_text")
        .and_then(|p| p["text"].as_str().map(str::to_string))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Routing: chat completions vs. responses
// ---------------------------------------------------------------------------

/// A provider without a Responses API is called through Chat Completions.
#[tokio::test]
async fn chat_path_is_used_when_responses_are_unsupported() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;
    let model = AnyLLMModel::new(
        "openrouter/openai/gpt-5.4-mini",
        Some(&url),
        Some("router-key"),
        None,
    )
    .unwrap();

    let mut request = request_with(ModelSettings::default());
    request.system_instructions = Some("You are terse.".into());
    let response = model.get_response(request).await.unwrap();

    let reqs = captured.lock().unwrap().clone();
    assert_eq!(reqs[0].path, "/chat/completions");
    assert_eq!(reqs[0].body["messages"][0]["role"], "system");
    assert_eq!(reqs[0].body["messages"][0]["content"], "You are terse.");
    assert_eq!(reqs[0].header("authorization"), Some("Bearer router-key"));
    assert_eq!(first_text(&response), "Hello");
}

/// A provider that implements the Responses API is called through it, and the response id is
/// carried back.
#[tokio::test]
async fn responses_path_is_used_when_supported() {
    let (url, captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;
    let model =
        AnyLLMModel::new("openai/gpt-5.4-mini", Some(&url), Some("openai-key"), None).unwrap();
    assert_eq!(model.api(), AdapterApi::Responses);

    let mut request = request_with(ModelSettings {
        store: Some(true),
        ..Default::default()
    });
    request.system_instructions = Some("You are terse.".into());
    request.previous_response_id = Some("resp_prev".into());
    request.conversation_id = Some("conv_123".into());
    let response = model.get_response(request).await.unwrap();

    let reqs = captured.lock().unwrap().clone();
    assert_eq!(reqs[0].path, "/responses");
    assert_eq!(reqs[0].body["previous_response_id"], "resp_prev");
    assert_eq!(reqs[0].body["conversation"], "conv_123");
    assert_eq!(
        reqs[0].header("user-agent"),
        Some(format!("Agents/Rust {}", env!("CARGO_PKG_VERSION")).as_str())
    );
    assert_eq!(response.response_id.as_deref(), Some("resp_123"));
    assert_eq!(first_text(&response), "Hello");
}

/// Chat Completions can be forced even where the Responses API is available.
#[tokio::test]
async fn can_force_chat_completions_when_responses_are_supported() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;
    let model = chat_model("openai/gpt-5.4-mini", &url);
    assert_eq!(model.api(), AdapterApi::ChatCompletions);

    let response = model
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert_eq!(captured.lock().unwrap()[0].path, "/chat/completions");
    assert_eq!(first_text(&response), "Hello");
}

/// A terminal `incomplete`/`failed` status is an error rather than an empty turn.
#[tokio::test]
async fn responses_path_rejects_failed_terminal_status() {
    for status in ["incomplete", "failed"] {
        let mut payload = responses_payload("Hello");
        payload["status"] = json!(status);
        payload["output"] = json!([]);
        let (url, _captured) = mock_server(vec![Reply::Json(payload)]).await;

        let err = responses_model("openai/gpt-5.4-mini", &url)
            .get_response(request_with(ModelSettings::default()))
            .await
            .unwrap_err();

        assert!(err.to_string().contains(status), "status {status}: {err}");
    }
}

// ---------------------------------------------------------------------------
// Request shaping
// ---------------------------------------------------------------------------

/// `parallel_tool_calls` only travels with converted tools, on the Chat Completions path.
#[tokio::test]
async fn chat_only_forwards_parallel_tool_calls_with_converted_tools() {
    for parallel in [true, false] {
        for source in ["none", "function", "handoff"] {
            let mut request = request_with(ModelSettings {
                parallel_tool_calls: Some(parallel),
                ..Default::default()
            });
            add_tools(&mut request, source);

            let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;
            chat_model("openrouter/openai/gpt-5.4-mini", &url)
                .get_response(request)
                .await
                .unwrap();
            let body = captured.lock().unwrap()[0].body.clone();

            if source == "none" {
                assert!(body.get("parallel_tool_calls").is_none(), "{source}");
                assert!(body.get("tools").is_none(), "{source}");
            } else {
                assert_eq!(body["parallel_tool_calls"], parallel, "{source}");
                assert!(body["tools"].as_array().is_some_and(|t| !t.is_empty()));
            }
        }
    }
}

/// The same rule on the Responses path, including the `None` case.
#[tokio::test]
async fn responses_parallel_tool_calls_follow_converted_tools() {
    for parallel in [Some(true), Some(false), None] {
        for source in ["none", "function", "handoff"] {
            let mut request = request_with(ModelSettings {
                parallel_tool_calls: parallel,
                ..Default::default()
            });
            add_tools(&mut request, source);

            let (url, captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;
            responses_model("openai/gpt-5.4-mini", &url)
                .get_response(request)
                .await
                .unwrap();
            let body = captured.lock().unwrap()[0].body.clone();

            match (source, parallel) {
                ("none", _) | (_, None) => {
                    assert!(body.get("parallel_tool_calls").is_none(), "{source}")
                }
                (_, Some(p)) => assert_eq!(body["parallel_tool_calls"], p, "{source}"),
            }
            assert_eq!(
                body.get("tools")
                    .is_some_and(|t| !t.as_array().unwrap().is_empty()),
                source != "none",
                "{source}"
            );
        }
    }
}

/// `ModelSettings.reasoning` reaches the Responses request as a mapping.
#[tokio::test]
async fn responses_path_sends_reasoning_as_a_mapping() {
    use openai_agents_sdk::model_settings::Reasoning;

    for stream in [false, true] {
        let reply = if stream {
            Reply::Sse(vec![json!({
                "type": "response.completed", "response": responses_payload("Hello")
            })])
        } else {
            Reply::Json(responses_payload("Hello"))
        };
        let (url, captured) = mock_server(vec![reply]).await;
        let model = responses_model("openai/gpt-5.4-mini", &url);
        let request = request_with(ModelSettings {
            reasoning: Some(Reasoning {
                effort: Some("low".into()),
                summary: Some("concise".into()),
                ..Default::default()
            }),
            ..Default::default()
        });

        if stream {
            use futures::StreamExt;
            let mut events = model.stream_response(request).await.unwrap();
            while events.next().await.is_some() {}
        } else {
            model.get_response(request).await.unwrap();
        }

        let body = captured.lock().unwrap()[0].body.clone();
        assert!(body["reasoning"].is_object(), "stream={stream}");
        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(body["reasoning"]["summary"], "concise");
    }
}

/// An unset `reasoning` stays off the request instead of pinning a default.
#[tokio::test]
async fn responses_path_omits_reasoning_when_unset() {
    let (url, captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;

    responses_model("openrouter/some-model", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert!(captured.lock().unwrap()[0].body.get("reasoning").is_none());
}

#[tokio::test]
async fn responses_path_forwards_prompt_cache_retention() {
    for stream in [false, true] {
        let reply = if stream {
            Reply::Sse(vec![json!({
                "type": "response.completed", "response": responses_payload("Hello")
            })])
        } else {
            Reply::Json(responses_payload("Hello"))
        };
        let (url, captured) = mock_server(vec![reply]).await;
        let model = responses_model("openai/gpt-5.4-mini", &url);
        let request = request_with(ModelSettings {
            prompt_cache_retention: Some("24h".into()),
            ..Default::default()
        });

        if stream {
            use futures::StreamExt;
            let mut events = model.stream_response(request).await.unwrap();
            while events.next().await.is_some() {}
        } else {
            model.get_response(request).await.unwrap();
        }

        assert_eq!(
            captured.lock().unwrap()[0].body["prompt_cache_retention"],
            "24h",
            "stream={stream}"
        );
    }
}

#[tokio::test]
async fn responses_path_omits_prompt_cache_retention_when_unset() {
    let (url, captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;

    responses_model("openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert!(
        captured.lock().unwrap()[0]
            .body
            .get("prompt_cache_retention")
            .is_none()
    );
}

#[tokio::test]
async fn chat_sets_logprobs_when_top_logprobs_set() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;

    chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings {
            top_logprobs: Some(3),
            ..Default::default()
        }))
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert_eq!(body["top_logprobs"], 3);
    assert_eq!(body["logprobs"], true);
}

#[tokio::test]
async fn chat_omits_logprobs_when_top_logprobs_unset() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;

    chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert!(body.get("logprobs").is_none());
    assert!(body.get("top_logprobs").is_none());
}

/// Every request carries the SDK's own user agent on the Chat Completions path too.
#[tokio::test]
async fn user_agent_header_on_the_chat_path() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;

    chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert_eq!(
        captured.lock().unwrap()[0].header("user-agent"),
        Some(format!("Agents/Rust {}", env!("CARGO_PKG_VERSION")).as_str())
    );
}

// ---------------------------------------------------------------------------
// Response handling
// ---------------------------------------------------------------------------

/// Plaintext reasoning on a chat message becomes a reasoning item that can be replayed.
#[tokio::test]
async fn chat_path_preserves_plaintext_reasoning_for_replay() {
    let mut reply = chat_completion("The answer is 42.");
    reply["choices"][0]["message"]["reasoning_content"] = json!("Plaintext reasoning");
    let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;

    let response = chat_model("deepseek/deepseek-reasoner", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let reasoning: Vec<Value> = output_values(&response)
        .into_iter()
        .filter(|i| i["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["summary"][0]["text"], "Plaintext reasoning");
    assert_eq!(first_text(&response), "The answer is 42.");
}

/// A Gemini tool call keeps the model, response id and thought signature that identify it.
#[tokio::test]
async fn chat_path_preserves_gemini_tool_call_metadata() {
    let reply = json!({
        "id": "chatcmpl_tool_123",
        "model": "gemini-2.0-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                "extra_content": {"google": {"thought_signature": "sig_123"}}
            }]
        }}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;

    let response = chat_model("gemini/gemini-2.0-flash", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let calls: Vec<Value> = output_values(&response)
        .into_iter()
        .filter(|i| i["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 1);
    let provider_data = &calls[0]["provider_data"];
    assert_eq!(provider_data["model"], "gemini/gemini-2.0-flash");
    assert_eq!(provider_data["response_id"], "chatcmpl_tool_123");
    assert_eq!(provider_data["thought_signature"], "sig_123");
}

/// A `content_filter` turn that still carries text keeps the text and gains no refusal.
#[tokio::test]
async fn chat_path_content_filter_keeps_real_content() {
    let mut reply = chat_completion("here is the answer");
    reply["choices"][0]["finish_reason"] = json!("content_filter");
    let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;

    let response = chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert_eq!(first_text(&response), "here is the answer");
    let refusals: Vec<Value> = output_values(&response)
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .filter(|p| p["type"] == "refusal")
        .collect();
    assert!(refusals.is_empty());
}

/// A filtered turn with no text must become a refusal instead of an empty output.
#[tokio::test]
async fn chat_path_surfaces_content_filter_refusal() {
    let mut reply = chat_completion("");
    reply["choices"][0]["finish_reason"] = json!("content_filter");
    let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;

    let response = chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let refusals: Vec<Value> = output_values(&response)
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .filter(|p| p["type"] == "refusal")
        .collect();
    assert!(!refusals.is_empty(), "expected a refusal item");
}

/// A Responses payload that omits `cache_write_tokens` defaults it to zero without inventing
/// the key in the raw usage the caller asked to keep.
#[tokio::test]
async fn responses_path_defaults_missing_cache_write_tokens() {
    let (url, _captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;

    let response = responses_model("openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings {
            preserve_raw_usage: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap();

    assert_eq!(first_text(&response), "Hello");
    assert_eq!(response.usage.input_tokens_details.cache_write_tokens, 0);
    let raw = response.raw_usage.expect("raw usage preserved");
    assert!(
        raw["input_tokens_details"]
            .get("cache_write_tokens")
            .is_none()
    );
}

/// Null token counts coming back from a provider are normalized to zero on both wire APIs.
#[tokio::test]
async fn runner_normalizes_nullable_usage() {
    let cases: [([Option<u64>; 3], [u64; 3]); 6] = [
        ([None, Some(5), Some(12)], [0, 5, 12]),
        ([Some(7), None, Some(12)], [7, 0, 12]),
        ([Some(7), Some(5), None], [7, 5, 0]),
        ([None, None, None], [0, 0, 0]),
        ([Some(0), Some(0), Some(0)], [0, 0, 0]),
        ([Some(7), Some(5), Some(12)], [7, 5, 12]),
    ];
    let to_json = |v: Option<u64>| v.map(Value::from).unwrap_or(Value::Null);

    for (counts, expected) in cases {
        // Chat Completions.
        let mut reply = chat_completion("ok");
        reply["usage"] = json!({
            "prompt_tokens": to_json(counts[0]),
            "completion_tokens": to_json(counts[1]),
            "total_tokens": to_json(counts[2]),
        });
        let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;
        let model = chat_model("openrouter/openai/gpt-5.4-mini", &url);
        let agent = Agent::<()>::new("test").with_model_instance(Arc::new(model));
        let usage = Runner::run(&agent, "hi").await.unwrap().usage();
        assert_eq!(usage.requests, 1, "chat {counts:?}");
        assert_eq!(
            [usage.input_tokens, usage.output_tokens, usage.total_tokens],
            expected,
            "chat {counts:?}"
        );

        // Responses.
        let mut reply = responses_payload("ok");
        reply["usage"] = json!({
            "input_tokens": to_json(counts[0]),
            "output_tokens": to_json(counts[1]),
            "total_tokens": to_json(counts[2]),
        });
        let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;
        let model = responses_model("openai/gpt-5.4-mini", &url);
        let agent = Agent::<()>::new("test").with_model_instance(Arc::new(model));
        let usage = Runner::run(&agent, "hi").await.unwrap().usage();
        assert_eq!(usage.requests, 1, "responses {counts:?}");
        assert_eq!(
            [usage.input_tokens, usage.output_tokens, usage.total_tokens],
            expected,
            "responses {counts:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// A provider without a Responses API streams through the Chat Completions handler, which
/// synthesizes the Responses event sequence.
#[tokio::test]
async fn stream_uses_the_chat_handler_when_responses_are_unsupported() {
    use futures::StreamExt;

    let chunk = json!({
        "id": "chunk-1", "object": "chat.completion.chunk", "created": 1, "model": "fake",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"},
                     "finish_reason": null}]
    });
    let done = json!({
        "id": "chunk-1", "object": "chat.completion.chunk", "created": 1, "model": "fake",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    let (url, captured) = mock_server(vec![Reply::Sse(vec![chunk, done])]).await;

    let mut events = chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .stream_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let mut types = Vec::new();
    let mut text = String::new();
    while let Some(event) = events.next().await {
        let event = event.unwrap();
        let kind = event["type"].as_str().unwrap_or_default().to_string();
        if kind == "response.output_text.delta" {
            text.push_str(event["delta"].as_str().unwrap_or_default());
        }
        types.push(kind);
    }

    assert_eq!(captured.lock().unwrap()[0].path, "/chat/completions");
    assert!(captured.lock().unwrap()[0].body["stream"] == json!(true));
    assert_eq!(text, "Hello");
    assert_eq!(types.first().map(String::as_str), Some("response.created"));
    assert_eq!(types.last().map(String::as_str), Some("response.completed"));
}

/// A provider that implements the Responses API has its events passed through untouched.
#[tokio::test]
async fn stream_passthrough_uses_responses_when_supported() {
    use futures::StreamExt;

    let created = json!({
        "type": "response.created",
        "response": {"id": "resp_123", "object": "response", "status": "in_progress",
                     "output": []}
    });
    let completed = json!({"type": "response.completed", "response": responses_payload("Hello")});
    let (url, captured) = mock_server(vec![Reply::Sse(vec![created, completed])]).await;

    let mut events = responses_model("openai/gpt-5.4-mini", &url)
        .stream_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let mut types = Vec::new();
    while let Some(event) = events.next().await {
        types.push(
            event.unwrap()["type"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        );
    }

    assert_eq!(captured.lock().unwrap()[0].path, "/responses");
    assert_eq!(types, ["response.created", "response.completed"]);
}

// ---------------------------------------------------------------------------
// AnyLLMProvider
// ---------------------------------------------------------------------------

#[tokio::test]
async fn provider_passes_api_override() {
    let _guard = env_lock().await;
    let provider = AnyLLMProvider {
        api: Some(AdapterApi::ChatCompletions),
        api_key: Some("sk-test".into()),
        ..Default::default()
    };

    let model = provider.get_model(Some("openai/gpt-4.1-mini")).unwrap();

    assert_eq!(model.model_name().as_deref(), Some("openai/gpt-4.1-mini"));
}

/// The default model is read when `get_model` is called, not when the provider is built.
#[tokio::test]
async fn provider_reads_the_default_model_at_call_time() {
    let _guard = env_lock().await;
    let provider = AnyLLMProvider {
        api_key: Some("sk-test".into()),
        ..Default::default()
    };
    // SAFETY: every test touching the environment holds `env_lock`.
    unsafe { std::env::set_var("OPENAI_DEFAULT_MODEL", "gpt-4.1") };

    let model = provider.get_model(None).unwrap();

    // SAFETY: as above.
    unsafe { std::env::remove_var("OPENAI_DEFAULT_MODEL") };
    assert_eq!(model.model_name().as_deref(), Some("openai/gpt-4.1"));
}

// ---------------------------------------------------------------------------
// Streamed request counting
// ---------------------------------------------------------------------------

/// Run `Runner::run_streamed` against a Responses stream that ends with `completed` and return
/// the usage the run reported.
async fn streamed_usage(completed: Value) -> openai_agents_sdk::usage::Usage {
    use futures::StreamExt;

    let created = json!({
        "type": "response.created",
        "response": {"id": "resp_123", "object": "response", "status": "in_progress",
                     "output": []}
    });
    let (url, _captured) = mock_server(vec![Reply::Sse(vec![
        created,
        json!({"type": "response.completed", "response": completed}),
    ])])
    .await;
    let agent = Agent::<()>::new("test")
        .with_model_instance(Arc::new(responses_model("openai/gpt-5.4-mini", &url)));
    let mut result = Runner::run_streamed(&agent, "hi");
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    result.result().unwrap().usage()
}

/// A streamed call whose provider omitted usage still counts as one request, exactly like the
/// non-streaming path, without inventing token counts.
#[tokio::test]
async fn responses_stream_counts_request_when_usage_is_absent() {
    let mut completed = responses_payload("Hello");
    completed.as_object_mut().unwrap().remove("usage");

    let usage = streamed_usage(completed).await;

    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 0);
}

/// A response that did report usage is counted once, not again by the usage-less path.
#[tokio::test]
async fn responses_stream_with_usage_is_not_double_counted() {
    let usage = streamed_usage(responses_payload("Hello")).await;

    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 15);
}
