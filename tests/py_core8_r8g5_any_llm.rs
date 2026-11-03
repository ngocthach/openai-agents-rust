//! Parity ports of cases in `tests/models/test_any_llm_model.py`, `test_litellm_chatcompletions_stream.py`
//! that the earlier ports left as Python-only: forced-Responses errors, terminal stream
//! failures, transport extras, prompt-managed requests, replay sanitizing, reasoning objects and
//! the connection release when a consumer drops a stream.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

async fn collect_stream(
    model: &dyn Model,
    request: ModelRequest,
) -> (Vec<Value>, Option<openai_agents_sdk::AgentsError>) {
    use futures::StreamExt;
    let mut events = model.stream_response(request).await.unwrap();
    let mut seen = Vec::new();
    while let Some(event) = events.next().await {
        match event {
            Ok(event) => seen.push(event),
            Err(error) => return (seen, Some(error)),
        }
    }
    (seen, None)
}

/// `test_any_llm_forced_responses_errors_when_provider_does_not_support_it`
#[tokio::test]
async fn forced_responses_errors_when_provider_does_not_support_it() {
    let _env = env_lock().await;
    let error = AnyLLMModel::new(
        "openrouter/openai/gpt-4.1-mini",
        None,
        Some("sk-test"),
        Some(AdapterApi::Responses),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not support the Responses API"),
        "{error}"
    );
}

/// `test_any_llm_chat_preserves_falsy_reasoning`: the effort reaches the chat request.
#[tokio::test]
async fn chat_forwards_reasoning_effort() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;
    let settings = ModelSettings {
        reasoning: Some(openai_agents_sdk::model_settings::Reasoning::effort("low")),
        ..Default::default()
    };
    chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(settings))
        .await
        .unwrap();
    assert_eq!(captured.lock().unwrap()[0].body["reasoning_effort"], "low");
}

/// `test_any_llm_responses_stream_rejects_failed_terminal_events` (2 ids) and
/// `test_any_llm_responses_stream_rejects_error_event`.
#[tokio::test]
async fn responses_stream_rejects_failed_terminal_events() {
    for event_type in ["response.incomplete", "response.failed"] {
        let mut response = responses_payload("partial");
        response["id"] = json!("resp-terminal");
        response["status"] = json!(event_type.trim_start_matches("response."));
        let event = json!({"type": event_type, "sequence_number": 1, "response": response});
        let (url, _captured) = mock_server(vec![Reply::Sse(vec![event])]).await;
        let model = responses_model("openai/gpt-5.4-mini", &url);

        let (events, error) = collect_stream(&model, request_with(ModelSettings::default())).await;

        assert_eq!(events.len(), 1, "{event_type}: {events:?}");
        assert_eq!(events[0]["type"], event_type);
        assert_eq!(events[0]["response"]["id"], "resp-terminal");
        let error = error.unwrap_or_else(|| panic!("{event_type}: the stream must fail"));
        assert!(
            matches!(
                error,
                openai_agents_sdk::AgentsError::ModelBehaviorError { .. }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains(event_type), "{error}");
    }
}

#[tokio::test]
async fn responses_stream_rejects_error_event() {
    let event = json!({"type": "error", "sequence_number": 1, "code": "invalid_request_error",
                       "message": "bad request", "param": null});
    let (url, _captured) = mock_server(vec![Reply::Sse(vec![event])]).await;
    let model = responses_model("openai/gpt-5.4-mini", &url);

    let (events, error) = collect_stream(&model, request_with(ModelSettings::default())).await;

    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["type"], "error");
    assert_eq!(events[0]["code"], "invalid_request_error");
    let error = error.expect("the stream must fail");
    assert!(
        matches!(
            error,
            openai_agents_sdk::AgentsError::ModelBehaviorError { .. }
        ),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("invalid_request_error"),
        "{error}"
    );
}

/// `test_any_llm_responses_path_passes_transport_kwargs_via_private_provider_api`: headers,
/// query and body extras reach the request.
#[tokio::test]
async fn responses_path_passes_transport_extras() {
    let (url, captured) = mock_server(vec![Reply::Json(responses_payload("Hello"))]).await;
    let settings = ModelSettings {
        extra_headers: Some([("X-Test-Header".to_string(), "test".to_string())].into()),
        extra_query: Some([("trace".to_string(), "1".to_string())].into()),
        extra_body: Some(serde_json::from_value(json!({"foo": "bar"})).unwrap()),
        ..Default::default()
    };
    responses_model("openai/gpt-5.4-mini", &url)
        .get_response(request_with(settings))
        .await
        .unwrap();
    let request = captured.lock().unwrap()[0].clone();
    assert_eq!(request.header("x-test-header"), Some("test"));
    assert!(request.path.contains("trace=1"), "{}", request.path);
    assert_eq!(request.body["foo"], "bar");
}

/// `test_any_llm_prompt_requests_fail_fast`, on both wire APIs and both call shapes.
#[tokio::test]
async fn prompt_requests_fail_fast() {
    use openai_agents_sdk::prompts::Prompt;

    for api in [AdapterApi::Responses, AdapterApi::ChatCompletions] {
        let (url, captured) = mock_server(vec![]).await;
        let model = AnyLLMModel::new(
            "openai/gpt-5.4-mini",
            Some(&url),
            Some("sk-test"),
            Some(api),
        )
        .unwrap();
        let prompt = Prompt {
            id: "pmpt_123".into(),
            version: None,
            variables: None,
        };
        let mut request = request_with(ModelSettings::default());
        request.prompt = Some(prompt);

        let error = model.get_response(request.clone()).await.unwrap_err();
        assert!(
            error.to_string().contains("prompt-managed requests"),
            "{api:?}: {error}"
        );
        let streamed = model.stream_response(request).await;
        assert!(
            streamed.is_err_and(|e| e.to_string().contains("prompt-managed requests")),
            "{api:?}"
        );
        assert!(
            captured.lock().unwrap().is_empty(),
            "{api:?}: no request is sent"
        );
    }
}

/// `test_any_llm_responses_input_sanitizer_strips_none_fields_from_reasoning_items` and
/// `test_any_llm_responses_path_sanitizes_replayed_items_before_validation`.
#[tokio::test]
async fn responses_replay_input_is_sanitized() {
    let (url, captured) = mock_server(vec![Reply::Json(responses_payload("ok"))]).await;
    let model = responses_model("openai/gpt-5.4-mini", &url);
    let mut request = request_with(ModelSettings::default());
    request.input = [
        json!({"role": "user", "content": "What's the weather in Tokyo?"}),
        json!({"id": "rid1", "type": "reasoning",
               "summary": [{"text": "why", "type": "summary_text"}],
               "content": [{"type": "reasoning_text", "text": "thinking"}],
               "status": null, "encrypted_content": null}),
        json!({"id": openai_agents_sdk::models::FAKE_RESPONSES_ID, "type": "reasoning",
               "summary": [{"text": "I should call the weather tool first.",
                            "type": "summary_text"}],
               "status": null,
               "provider_data": {"model": "anthropic/fake-responses-model"}}),
        json!({"id": openai_agents_sdk::models::FAKE_RESPONSES_ID,
               "arguments": "{\"city\": \"Tokyo\"}", "call_id": "call_weather_123",
               "name": "get_weather", "type": "function_call", "status": null,
               "provider_data": {"model": "anthropic/fake-responses-model"}}),
        json!({"type": "function_call_output", "call_id": "call_weather_123",
               "output": "The weather in Tokyo is sunny and 22C."}),
    ]
    .into_iter()
    .map(InputItem)
    .collect();

    model.get_response(request).await.unwrap();

    assert_eq!(
        captured.lock().unwrap()[0].body["input"],
        json!([
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"id": "rid1", "type": "reasoning",
             "summary": [{"text": "why", "type": "summary_text"}],
             "content": [{"type": "reasoning_text", "text": "thinking"}]},
            {"arguments": "{\"city\": \"Tokyo\"}", "call_id": "call_weather_123",
             "name": "get_weather", "type": "function_call"},
            {"type": "function_call_output", "call_id": "call_weather_123",
             "output": "The weather in Tokyo is sunny and 22C."}
        ])
    );
}

/// `test_any_llm_reasoning_objects_prefer_content_attributes_over_iterable_pairs` and
/// `test_any_llm_stream_flattens_reasoning_object_when_reasoning_content_is_empty`: a
/// `reasoning` object on the wire is flattened to its `content` text.
#[tokio::test]
async fn reasoning_objects_are_flattened_to_their_content() {
    // Non-streamed message.
    let mut completion = chat_completion("Answer");
    completion["choices"][0]["message"]["reasoning"] = json!({"content": "用户"});
    let (url, _captured) = mock_server(vec![Reply::Json(completion)]).await;
    let response = chat_model("openrouter/reasoning-model", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let items = output_values(&response);
    let reasoning = items
        .iter()
        .find(|i| i["type"] == "reasoning")
        .unwrap_or_else(|| panic!("no reasoning item in {items:?}"));
    assert_eq!(reasoning["content"][0]["text"], "用户");

    // Streamed delta whose `reasoning_content` is empty.
    let chunk = json!({
        "id": "chunk-id", "created": 1, "model": "fake", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"reasoning_content": "",
                                           "reasoning": {"content": "Plaintext reasoning"}}}]
    });
    let done = json!({
        "id": "chunk-id", "created": 1, "model": "fake", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    let (url, _captured) = mock_server(vec![Reply::Sse(vec![chunk, done])]).await;
    let model = chat_model("openrouter/reasoning-model", &url);
    let (events, error) = collect_stream(&model, request_with(ModelSettings::default())).await;
    assert!(error.is_none(), "{error:?}");
    let text: String = events
        .iter()
        .filter(|e| e["type"] == "response.reasoning_text.delta")
        .filter_map(|e| e["delta"].as_str())
        .collect();
    assert_eq!(text, "Plaintext reasoning");
}

/// A server that sends the first SSE frame, then keeps the socket open and reports when the
/// client closes it.
async fn hanging_stream_server(
    first_frame: String,
) -> (String, tokio::sync::oneshot::Receiver<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
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
            let len: usize = text[..idx]
                .lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.trim().parse().ok())
                .unwrap_or(0);
            if buf.len() >= idx + 4 + len {
                break;
            }
        }
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                    transfer-encoding: chunked\r\n\r\n";
        let chunk = format!("{:x}\r\n{first_frame}\r\n", first_frame.len());
        sock.write_all(head.as_bytes()).await.unwrap();
        sock.write_all(chunk.as_bytes()).await.unwrap();
        // The client closes the connection once it drops the stream.
        let closed = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => return true,
                    Ok(_) => continue,
                }
            }
        })
        .await
        .unwrap_or(false);
        let _ = tx.send(closed);
    });
    (format!("http://{addr}"), rx)
}

/// `test_*_stream_closes_provider_stream_after_cancellation` (chat completions, Responses and
/// LiteLLM): a consumer that drops the stream mid-flight releases the provider connection.
#[tokio::test]
async fn dropping_a_stream_closes_the_provider_connection() {
    use futures::StreamExt;

    let chat_frame = format!(
        "data: {}\n\n",
        json!({"id": "c", "created": 1, "model": "fake", "object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]})
    );
    let responses_frame = format!(
        "event: response.created\ndata: {}\n\n",
        json!({"type": "response.created", "sequence_number": 0,
               "response": {"id": "resp_1", "object": "response", "status": "in_progress",
                            "output": []}})
    );
    for (label, frame) in [
        ("any-llm chat", chat_frame.clone()),
        ("any-llm responses", responses_frame),
        ("litellm", chat_frame),
    ] {
        let (url, closed) = hanging_stream_server(frame).await;
        let model: Arc<dyn Model> = match label {
            "any-llm chat" => Arc::new(chat_model("openrouter/openai/gpt-5.4-mini", &url)),
            "any-llm responses" => Arc::new(responses_model("openai/gpt-5.4-mini", &url)),
            _ => Arc::new(
                openai_agents_sdk::extensions::models::LitellmModel::with_options(
                    "test-model",
                    Some(&url),
                    Some("sk-test"),
                    None,
                )
                .unwrap(),
            ),
        };
        let mut events = model
            .stream_response(request_with(ModelSettings::default()))
            .await
            .unwrap();
        let first = events.next().await.expect("a first event").unwrap();
        assert!(first["type"].is_string(), "{label}: {first}");

        drop(events);

        assert!(
            closed.await.unwrap(),
            "{label}: the provider connection stayed open after the stream was dropped"
        );
    }
}

/// `test_any_llm_thinking_is_omitted_from_default_trace_export`, replay half: a message that
/// carries only a `thinking` field keeps it as reasoning text, so a replayed turn still has it
/// (the exported span's redaction is covered by `py_tracing_reasoning_export`).
#[tokio::test]
async fn a_thinking_field_is_kept_as_reasoning_for_replay() {
    let mut completion = chat_completion("Visible answer");
    completion["choices"][0]["message"]["thinking"] = json!("PRIVATE_ANY_LLM_THINKING_SENTINEL");
    let (url, _captured) = mock_server(vec![Reply::Json(completion)]).await;
    let response = chat_model("openrouter/test-model", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let items = output_values(&response);
    assert!(
        items.iter().any(|i| i["type"] == "reasoning"
            && i["content"][0]["text"] == "PRIVATE_ANY_LLM_THINKING_SENTINEL"),
        "{items:?}"
    );
    assert_eq!(first_text(&response), "Visible answer");
}

/// `test_litellm_retry_settings_do_not_leak_and_disable_provider_retries_on_runner_retry`: the
/// SDK retry configuration is never sent to the provider.
#[tokio::test]
async fn litellm_does_not_forward_the_sdk_retry_settings() {
    use openai_agents_sdk::model_settings::{ModelRetryBackoffSettings, ModelRetrySettings};

    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("ok"))]).await;
    let settings = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: Some(2),
            backoff: Some(ModelRetryBackoffSettings {
                initial_delay: Some(0.25),
                jitter: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        }),
        extra_args: Some(serde_json::from_value(json!({"custom_param": "custom_value"})).unwrap()),
        ..Default::default()
    };
    openai_agents_sdk::extensions::models::LitellmModel::with_options(
        "test-model",
        Some(&url),
        Some("sk-test"),
        None,
    )
    .unwrap()
    .get_response(request_with(settings))
    .await
    .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert_eq!(body["custom_param"], "custom_value");
    assert!(body.get("retry").is_none(), "{body}");
}

/// A model-call timeout that cannot be a `Duration` (infinite) never fires and never panics.
#[tokio::test]
async fn an_infinite_model_call_timeout_does_not_panic() {
    let (url, _captured) = mock_server(vec![Reply::Json(chat_completion("ok"))]).await;
    let model = chat_model("openrouter/openai/gpt-5.4-mini", &url);
    let agent = Agent::<()>::new("timeout")
        .with_model_instance(Arc::new(model))
        .with_model_settings(ModelSettings {
            timeout: Some(f64::INFINITY),
            ..Default::default()
        });
    let result = Runner::run(&agent, "hi").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("ok"));
}

/// `test_litellm_hook_can_enable_reasoning_content_replay`
#[tokio::test]
async fn litellm_hook_can_enable_reasoning_content_replay() {
    const MODEL: &str = "reasoning-content-model-b";
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("done"))]).await;
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let seen = contexts.clone();
    let hook: openai_agents_sdk::models::openai_chatcompletions::ShouldReplayReasoningContent =
        Arc::new(move |context| {
            seen.lock().unwrap().push(context.clone());
            context.model == MODEL
        });
    let model = openai_agents_sdk::extensions::models::LitellmModel::with_options(
        MODEL,
        Some(&url),
        Some("sk-test"),
        Some(hook),
    )
    .unwrap();
    let mut request = request_with(ModelSettings::default());
    request.input = [
        json!({"role": "user", "content": "What's the weather in Tokyo?"}),
        json!({"id": "__fake_id__", "type": "reasoning",
               "summary": [{"text": "I should call the weather tool first.",
                            "type": "summary_text"}],
               "content": null, "encrypted_content": null, "status": null,
               "provider_data": {"model": MODEL, "response_id": "chatcmpl-test"}}),
        json!({"arguments": "{\"city\": \"Tokyo\"}", "call_id": "call_weather_123",
               "name": "get_weather", "type": "function_call", "id": "__fake_id__",
               "status": null, "provider_data": {"model": MODEL}}),
        json!({"type": "function_call_output", "call_id": "call_weather_123",
               "output": "The weather in Tokyo is sunny and 22C."}),
    ]
    .into_iter()
    .map(InputItem)
    .collect();

    model.get_response(request).await.unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    let assistant = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .unwrap_or_else(|| panic!("no assistant tool-call message in {body}"));
    assert_eq!(
        assistant["reasoning_content"],
        "I should call the weather tool first."
    );
    let contexts = contexts.lock().unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].model, MODEL);
    assert_eq!(contexts[0].origin_model.as_deref(), Some(MODEL));
}
