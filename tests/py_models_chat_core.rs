//! Port of `tests/models/test_openai_chatcompletions.py` and
//! `tests/models/test_gpt6_chat_tool_defaults.py`: non-streaming Chat Completions requests and
//! responses, driven through a local HTTP mock server.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{FunctionToolSpec, Model, ModelRequest, ToolSpec};
use openai_agents_sdk::models::{ModelTracing, OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::retry::{ModelRetryAdviceRequest, ReplaySafety};
use openai_agents_sdk::{Agent, AgentsError, Prompt, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Value,
}

#[derive(Clone)]
struct Resp {
    status: u16,
    content_type: &'static str,
    headers: Vec<(String, String)>,
    body: String,
}

fn json_resp(v: Value) -> Resp {
    Resp {
        status: 200,
        content_type: "application/json",
        headers: vec![("x-request-id".into(), "req_default".into())],
        body: v.to_string(),
    }
}

fn sse_resp(chunks: &[Value]) -> Resp {
    let mut body = String::new();
    for c in chunks {
        body.push_str(&format!("data: {c}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    Resp {
        status: 200,
        content_type: "text/event-stream",
        headers: vec![("x-request-id".into(), "req_default".into())],
        body,
    }
}

type Captured = Arc<Mutex<Vec<Req>>>;

/// Serve `handler` on a local port; returns the `/v1` base URL and every captured request.
async fn serve<F>(handler: F) -> (String, Captured)
where
    F: Fn(&Req) -> Resp + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let cap = cap.clone();
            let handler = handler.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let req = loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let head = &text[..idx];
                    let mut lines = head.lines();
                    let mut first = lines.next().unwrap().split_whitespace();
                    let method = first.next().unwrap().to_string();
                    let path = first.next().unwrap().to_string();
                    let headers: HashMap<String, String> = lines
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                        break Req {
                            method,
                            path,
                            headers,
                            body,
                        };
                    }
                };
                cap.lock().unwrap().push(req.clone());
                let r = handler(&req);
                let mut head = format!(
                    "HTTP/1.1 {} X\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n",
                    r.status,
                    r.content_type,
                    r.body.len()
                );
                for (k, v) in &r.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(r.body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), captured)
}

/// Answer requests in order; the last reply repeats once the queue is exhausted.
async fn serve_queue(replies: Vec<Resp>) -> (String, Captured) {
    let queue = Mutex::new(VecDeque::from(replies));
    serve(move |_| {
        let mut q = queue.lock().unwrap();
        if q.len() > 1 {
            q.pop_front().unwrap()
        } else {
            q.front().unwrap().clone()
        }
    })
    .await
}

fn model_at(name: &str, base: &str) -> OpenAIChatCompletionsModel {
    OpenAIChatCompletionsModel::new(name, OpenAIClient::new("sk-test").with_base_url(base))
}

/// A model on a non-official URL; no request is ever made to it.
fn offline_model(name: &str) -> OpenAIChatCompletionsModel {
    model_at(name, "http://fake.invalid/v1")
}

fn request() -> ModelRequest {
    ModelRequest {
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

fn completion(message: Value, finish_reason: &str) -> Value {
    json!({"id": "resp-id", "created": 0, "model": "fake", "object": "chat.completion",
           "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}]})
}

fn text_completion(text: &str) -> Value {
    completion(json!({"role": "assistant", "content": text}), "stop")
}

async fn respond(choice_message: Value, finish_reason: &str) -> openai_agents_sdk::Result<Value> {
    let (base, _) = serve_queue(vec![json_resp(completion(choice_message, finish_reason))]).await;
    let resp = model_at("gpt-4", &base).get_response(request()).await?;
    Ok(json!(
        resp.output.iter().map(|o| o.to_value()).collect::<Vec<_>>()
    ))
}

fn user_input(text: &str) -> Vec<InputItem> {
    vec![InputItem::user(text)]
}

fn function_tool(name: &str) -> ToolSpec {
    ToolSpec::Function(FunctionToolSpec {
        name: name.into(),
        description: "test".into(),
        parameters: json!({"type": "object", "properties": {}}),
        strict: true,
        ..Default::default()
    })
}

// ---- warning capture -----------------------------------------------------------------

#[derive(Default)]
struct LogSink(Mutex<Vec<String>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let s = SINK.get_or_init(LogSink::default);
    log::set_logger(s).ok();
    log::set_max_level(log::LevelFilter::Trace);
    s
}

/// Serializes tests that assert on captured log lines.
fn log_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn log_text() -> String {
    sink().0.lock().unwrap().join("\n")
}

// --------------------------------------------------------------------------------------
// Runner round trips
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn runner_replays_openai_conversation_assistant_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for stream in [false, true] {
        for include_status in [true, false] {
            let mut assistant = json!({
                "id": "msg_assistant", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello, Ada!", "annotations": []}],
            });
            if !include_status {
                assistant.as_object_mut().unwrap().remove("status");
            }
            let history = json!([
                {"id": "msg_user", "type": "message", "role": "user", "status": "completed",
                 "content": [{"type": "input_text", "text": "My name is Ada."}]},
                assistant,
            ]);
            let (base, captured) = serve(move |req| {
                if req.path.starts_with("/v1/conversations/conv_test/items") {
                    let data = if req.method == "GET" {
                        history.clone()
                    } else {
                        json!([])
                    };
                    return json_resp(json!({"object": "list", "data": data, "has_more": false,
                                            "first_id": null, "last_id": null}));
                }
                if stream {
                    sse_resp(&[
                        json!({"id": "chatcmpl-test", "object": "chat.completion.chunk",
                        "created": 0, "model": "fake", "choices": [{"index": 0,
                        "finish_reason": "stop",
                        "delta": {"role": "assistant", "content": "Your name is Ada."}}]}),
                    ])
                } else {
                    json_resp(text_completion("Your name is Ada."))
                }
            })
            .await;
            let client = OpenAIClient::new("test-placeholder").with_base_url(&base);
            let session = Arc::new(
                openai_agents_sdk::memory::OpenAIConversationsSession::from_conversation_id(
                    "conv_test",
                    client.clone(),
                ),
            );
            let agent = Agent::<()>::new("test")
                .with_model_instance(Arc::new(OpenAIChatCompletionsModel::new("fake", client)));
            let options = RunOptions::new(()).session(session);
            let output = if stream {
                let mut result =
                    Runner::run_streamed_with_options(&agent, "What is my name?", options);
                {
                    use futures::StreamExt;
                    let mut events = std::pin::pin!(result.stream_events());
                    while let Some(ev) = events.next().await {
                        ev.unwrap();
                    }
                }
                result.final_output_text()
            } else {
                Runner::run_with_options(&agent, "What is my name?", options)
                    .await
                    .unwrap()
                    .final_output_text()
            };
            assert_eq!(output.as_deref(), Some("Your name is Ada."));
            let chat: Vec<Req> = captured
                .lock()
                .unwrap()
                .iter()
                .filter(|r| r.path == "/v1/chat/completions")
                .cloned()
                .collect();
            assert_eq!(chat.len(), 1, "stream={stream} status={include_status}");
            assert_eq!(
                chat[0].body["messages"],
                json!([
                    {"role": "user", "content": [{"type": "text", "text": "My name is Ada."}]},
                    {"role": "assistant", "content": "Hello, Ada!"},
                    {"role": "user", "content": "What is my name?"},
                ])
            );
        }
    }
}

#[tokio::test]
async fn runner_normalizes_nullable_chat_usage() {
    openai_agents_sdk::set_tracing_disabled(true);
    type Counts = Option<[Option<u64>; 3]>;
    let cases: Vec<(Counts, [u64; 3])> = vec![
        (Some([None, Some(5), Some(12)]), [0, 5, 12]),
        (Some([Some(7), None, Some(12)]), [7, 0, 12]),
        (Some([Some(7), Some(5), None]), [7, 5, 0]),
        (Some([None, None, None]), [0, 0, 0]),
        (Some([Some(7), Some(5), Some(12)]), [7, 5, 12]),
        (None, [0, 0, 0]),
    ];
    for (counts, expected) in cases {
        let usage = counts.map(|c| {
            json!({"prompt_tokens": c[0], "completion_tokens": c[1], "total_tokens": c[2],
                   "prompt_tokens_details": {"cached_tokens": 3},
                   "completion_tokens_details": {"reasoning_tokens": 2}})
        });
        let mut payload = text_completion("Hello");
        payload["usage"] = usage.clone().unwrap_or(Value::Null);
        let (base, _) = serve_queue(vec![json_resp(payload)]).await;
        let agent = Agent::<()>::new("test")
            .with_model_instance(Arc::new(model_at("fake", &base)))
            .with_model_settings(ModelSettings {
                preserve_raw_usage: Some(true),
                ..Default::default()
            });
        let result = Runner::run_with_options(&agent, "hi", RunOptions::new(()))
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("Hello"));
        let u = result.usage();
        assert_eq!(u.requests, 1, "{counts:?}");
        assert_eq!(
            [u.input_tokens, u.output_tokens, u.total_tokens],
            expected,
            "{counts:?}"
        );
        let has = counts.is_some();
        assert_eq!(
            u.input_tokens_details.cached_tokens,
            if has { 3 } else { 0 }
        );
        assert_eq!(
            u.output_tokens_details.reasoning_tokens,
            if has { 2 } else { 0 }
        );
        assert_eq!(result.raw_responses[0].raw_usage, usage);
    }
}

// --------------------------------------------------------------------------------------
// get_response parsing
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn get_response_with_text_message() {
    let mut payload = text_completion("Hello");
    payload["usage"] = json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 4}});
    let (base, _) = serve_queue(vec![json_resp(payload)]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(ModelRequest {
            input: user_input(""),
            ..request()
        })
        .await
        .unwrap();
    assert_eq!(resp.output.len(), 1);
    let item = resp.output[0].to_value();
    assert_eq!(item["type"], "message");
    assert_eq!(item["content"].as_array().unwrap().len(), 1);
    assert_eq!(item["content"][0]["type"], "output_text");
    assert_eq!(item["content"][0]["text"], "Hello");
    assert_eq!(resp.usage.input_tokens, 7);
    assert_eq!(resp.usage.output_tokens, 5);
    assert_eq!(resp.usage.total_tokens, 12);
    assert_eq!(resp.usage.input_tokens_details.cached_tokens, 3);
    assert_eq!(resp.usage.input_tokens_details.cache_write_tokens, 4);
    assert_eq!(resp.usage.output_tokens_details.reasoning_tokens, 0);
    assert!(resp.response_id.is_none());
    assert!(resp.raw_usage.is_none());
}

#[tokio::test]
async fn get_response_preserves_raw_usage_field_presence() {
    for (details, expected) in [
        (json!({}), json!({})),
        (json!({"cached_tokens": 0}), json!({"cached_tokens": 0})),
    ] {
        let mut payload = text_completion("ok");
        payload["usage"] = json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                                  "prompt_tokens_details": details});
        let (base, _) = serve_queue(vec![json_resp(payload)]).await;
        let resp = model_at("gpt-4", &base)
            .get_response(ModelRequest {
                model_settings: ModelSettings {
                    preserve_raw_usage: Some(true),
                    ..Default::default()
                },
                ..request()
            })
            .await
            .unwrap();
        assert_eq!(
            resp.raw_usage,
            Some(
                json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                        "prompt_tokens_details": expected})
            )
        );
        assert_eq!(resp.usage.input_tokens_details.cached_tokens, 0);
    }
}

#[tokio::test]
async fn get_response_raw_usage_is_none_when_provider_omits_usage() {
    let (base, _) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(ModelRequest {
            model_settings: ModelSettings {
                preserve_raw_usage: Some(true),
                ..Default::default()
            },
            ..request()
        })
        .await
        .unwrap();
    assert!(resp.raw_usage.is_none());
    assert_eq!(resp.usage.total_tokens, 0);
}

#[tokio::test]
async fn get_response_surfaces_empty_content_filter_as_refusal() {
    let out = respond(
        json!({"role": "assistant", "content": null}),
        "content_filter",
    )
    .await
    .unwrap();
    assert_eq!(out.as_array().unwrap().len(), 1);
    assert_eq!(out[0]["type"], "message");
    assert_eq!(out[0]["content"].as_array().unwrap().len(), 1);
    assert_eq!(out[0]["content"][0]["type"], "refusal");
    assert_eq!(
        out[0]["content"][0]["refusal"],
        "Response withheld by the provider's content filter."
    );
}

fn nonempty_messages() -> Vec<(Value, &'static str, Option<&'static str>)> {
    vec![
        (
            json!({"role": "assistant", "content": "partial"}),
            "message",
            Some("output_text"),
        ),
        (
            json!({"role": "assistant", "content": null, "refusal": "provider refusal"}),
            "message",
            Some("refusal"),
        ),
        (
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "call-1", "type": "function",
                 "function": {"name": "do_thing", "arguments": "{}"}}]}),
            "function_call",
            None,
        ),
    ]
}

async fn assert_preserved(finish_reason: &str) {
    for (message, output_type, content_type) in nonempty_messages() {
        let out = respond(message, finish_reason).await.unwrap();
        assert_eq!(out.as_array().unwrap().len(), 1, "{out}");
        assert_eq!(out[0]["type"], output_type);
        if let Some(ct) = content_type {
            assert_eq!(out[0]["content"].as_array().unwrap().len(), 1);
            assert_eq!(out[0]["content"][0]["type"], ct);
            if ct == "refusal" {
                assert_eq!(out[0]["content"][0]["refusal"], "provider refusal");
            }
        }
    }
}

#[tokio::test]
async fn get_response_preserves_nonempty_content_filter_output() {
    assert_preserved("content_filter").await;
}

#[tokio::test]
async fn get_response_preserves_empty_nonfiltered_output() {
    let out = respond(json!({"role": "assistant", "content": null}), "stop")
        .await
        .unwrap();
    assert_eq!(out, json!([]));
}

#[tokio::test]
async fn get_response_raises_on_truncated_empty_turn() {
    let err = respond(json!({"role": "assistant", "content": null}), "length")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("finish_reason='length'"), "{err}");
}

#[tokio::test]
async fn get_response_raises_on_truncated_empty_string_turn() {
    let err = respond(json!({"role": "assistant", "content": ""}), "length")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("finish_reason='length'"), "{err}");
}

#[tokio::test]
async fn get_response_preserves_nonempty_truncated_output() {
    assert_preserved("length").await;
}

#[tokio::test]
async fn get_response_with_no_message() {
    let (base, _) = serve_queue(vec![json_resp(json!({"id": "resp-id", "created": 0,
        "model": "fake", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "content_filter", "message": null}]}))])
    .await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert!(resp.output.is_empty());
}

#[tokio::test]
async fn get_response_attaches_logprobs() {
    let mut payload = completion(json!({"role": "assistant", "content": "Hi!"}), "stop");
    payload["choices"][0]["logprobs"] = json!({"content": [
        {"token": "Hi", "logprob": -0.5, "bytes": [1],
         "top_logprobs": [{"token": "Hi", "logprob": -0.5, "bytes": [1]}]},
        {"token": "!", "logprob": -0.1, "bytes": [2],
         "top_logprobs": [{"token": "!", "logprob": -0.1, "bytes": [2]}]},
    ]});
    payload["usage"] = json!({"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5});
    let (base, _) = serve_queue(vec![json_resp(payload)]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert_eq!(resp.output.len(), 1);
    let part = &resp.output[0].to_value()["content"][0];
    assert_eq!(part["type"], "output_text");
    let tokens: Vec<&str> = part["logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lp| lp["token"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, ["Hi", "!"]);
    assert_eq!(resp.usage.input_tokens, 3);
    assert_eq!(resp.usage.output_tokens, 2);
    assert_eq!(resp.usage.total_tokens, 5);
}

#[tokio::test]
async fn get_response_with_refusal() {
    let (base, _) = serve_queue(vec![json_resp(completion(
        json!({"role": "assistant", "refusal": "No thanks"}),
        "stop",
    ))])
    .await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert_eq!(resp.output.len(), 1);
    let item = resp.output[0].to_value();
    assert_eq!(item["content"][0]["type"], "refusal");
    assert_eq!(item["content"][0]["refusal"], "No thanks");
    // No usage block: the request still happened and counts, tokens stay zero.
    assert_eq!(resp.usage.requests, 1);
    assert_eq!(resp.usage.input_tokens, 0);
    assert_eq!(resp.usage.output_tokens, 0);
    assert_eq!(resp.usage.input_tokens_details.cached_tokens, 0);
    assert_eq!(resp.usage.output_tokens_details.reasoning_tokens, 0);
}

#[tokio::test]
async fn get_response_with_tool_call() {
    let (base, _) = serve_queue(vec![json_resp(completion(
        json!({"role": "assistant", "content": "Hi", "tool_calls": [
            {"id": "call-id", "type": "function",
             "function": {"name": "do_thing", "arguments": "{'x':1}"}}]}),
        "stop",
    ))])
    .await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert_eq!(resp.output.len(), 2);
    assert_eq!(resp.output[0].to_value()["type"], "message");
    let call = resp.output[1].to_value();
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["call_id"], "call-id");
    assert_eq!(call["name"], "do_thing");
    assert_eq!(call["arguments"], "{'x':1}");
}

#[tokio::test]
async fn get_response_rejects_custom_tool_call_in_strict_mode() {
    let (base, _) = serve_queue(vec![json_resp(completion(
        json!({"role": "assistant", "tool_calls": [
            {"id": "tool1", "type": "custom",
             "custom": {"name": "raw_tool", "input": "payload"}}]}),
        "tool_calls",
    ))])
    .await;
    let err = model_at("gpt-4", &base)
        .with_strict_feature_validation(true)
        .get_response(request())
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Custom tool calls are not supported"),
        "{err}"
    );
}

#[tokio::test]
async fn get_response_propagates_request_id() {
    let mut reply = json_resp(text_completion("ok"));
    reply.headers = vec![("x-request-id".into(), "req_nonstreamed_123".into())];
    let (base, _) = serve_queue(vec![reply]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert_eq!(resp.request_id.as_deref(), Some("req_nonstreamed_123"));
}

#[tokio::test]
async fn get_response_request_id_is_none_when_absent() {
    let mut reply = json_resp(text_completion("ok"));
    reply.headers.clear();
    let (base, _) = serve_queue(vec![reply]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert!(resp.request_id.is_none());
}

#[tokio::test]
async fn request_is_counted_when_provider_omits_usage() {
    let (base, _) = serve_queue(vec![json_resp(text_completion("hello"))]).await;
    let resp = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap();
    assert_eq!(resp.usage.requests, 1);
    assert_eq!(resp.usage.input_tokens, 0);
    assert_eq!(resp.usage.output_tokens, 0);
    assert_eq!(resp.usage.total_tokens, 0);
}

#[tokio::test]
async fn get_response_without_choices_reports_provider_error_payload() {
    let (base, _) = serve_queue(vec![json_resp(json!({"error": {"message": "overloaded"}}))]).await;
    let err = model_at("gpt-4", &base)
        .get_response(request())
        .await
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("has no choices"), "{text}");
    assert!(text.contains("overloaded"), "{text}");
}

// --------------------------------------------------------------------------------------
// Responses-only features
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn get_response_warns_and_ignores_server_managed_conversation_state_by_default() {
    let _guard = log_lock().lock().await;
    for (previous, conversation, expected) in [
        (Some("resp_123"), None, "previous_response_id"),
        (None, Some("conv_123"), "conversation_id"),
    ] {
        sink().0.lock().unwrap().clear();
        let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
        model_at("gpt-4", &base)
            .get_response(ModelRequest {
                previous_response_id: previous.map(String::from),
                conversation_id: conversation.map(String::from),
                ..request()
            })
            .await
            .unwrap();
        let text = log_text();
        assert!(text.contains(expected), "{text}");
        assert!(
            text.contains("Ignoring unsupported server-managed conversation state"),
            "{text}"
        );
        let body = &captured.lock().unwrap()[0].body;
        assert!(body.get("previous_response_id").is_none());
        assert!(body.get("conversation").is_none());
    }
}

#[tokio::test]
async fn get_response_warns_and_ignores_prompt_by_default() {
    let _guard = log_lock().lock().await;
    sink().0.lock().unwrap().clear();
    let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    model_at("gpt-4", &base)
        .get_response(ModelRequest {
            prompt: Some(Prompt::new("pmpt_123")),
            ..request()
        })
        .await
        .unwrap();
    let text = log_text();
    assert!(
        text.contains("Reusable prompts are only supported by the Responses API"),
        "{text}"
    );
    assert!(text.contains("Ignoring `prompt`"), "{text}");
    assert!(captured.lock().unwrap()[0].body.get("prompt").is_none());
}

#[tokio::test]
async fn get_response_rejects_server_managed_conversation_state_in_strict_mode() {
    for (previous, conversation, expected) in [
        (Some("resp_123"), None, "previous_response_id"),
        (None, Some("conv_123"), "conversation_id"),
    ] {
        let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
        let err = model_at("gpt-4", &base)
            .with_strict_feature_validation(true)
            .get_response(ModelRequest {
                previous_response_id: previous.map(String::from),
                conversation_id: conversation.map(String::from),
                ..request()
            })
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("server-managed conversation state"), "{err}");
        assert!(err.contains(expected), "{err}");
        assert!(captured.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn get_response_rejects_prompt_in_strict_mode() {
    let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    let err = model_at("gpt-4", &base)
        .with_strict_feature_validation(true)
        .get_response(ModelRequest {
            prompt: Some(Prompt::new("pmpt_123")),
            ..request()
        })
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("Reusable prompts"), "{err}");
    assert!(captured.lock().unwrap().is_empty());
}

#[tokio::test]
async fn unpaired_function_output_rejected_before_chat_request() {
    for omit_call_id in [true, false] {
        for stream in [false, true] {
            for strict in [false, true] {
                let (base, captured) = serve_queue(vec![json_resp(text_completion("x"))]).await;
                let mut item = json!({
                    "type": "function_call_output", "name": "notifications",
                    "namespace": "slack", "output": "Alice mentioned you in #deployments.",
                });
                if !omit_call_id {
                    item["call_id"] = Value::Null;
                }
                let req = ModelRequest {
                    input: vec![InputItem::from_value(item)],
                    ..request()
                };
                let model = model_at("gpt-4", &base).with_strict_feature_validation(strict);
                let err = if stream {
                    model.stream_response(req).await.err().unwrap()
                } else {
                    model.get_response(req).await.unwrap_err()
                }
                .to_string();
                assert!(err.contains("Unpaired function outputs"), "{err}");
                assert!(err.contains("Use a Responses model"), "{err}");
                assert!(captured.lock().unwrap().is_empty());
            }
        }
    }
}

#[tokio::test]
async fn get_response_rejects_non_text_tool_output_in_strict_mode() {
    let model = offline_model("gpt-4").with_strict_feature_validation(true);
    let err = model
        .get_response(ModelRequest {
            input: vec![InputItem::from_value(json!({
                "type": "function_call_output", "call_id": "call_image",
                "output": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
            }))],
            ..request()
        })
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("cannot be empty or contain only non-text content"),
        "{err}"
    );
}

#[tokio::test]
async fn get_response_warns_and_sends_placeholder_for_non_text_tool_output() {
    let _guard = log_lock().lock().await;
    sink().0.lock().unwrap().clear();
    let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    model_at("gpt-4", &base)
        .get_response(ModelRequest {
            input: vec![InputItem::from_value(json!({
                "type": "function_call_output", "call_id": "call_image",
                "output": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
            }))],
            ..request()
        })
        .await
        .unwrap();
    assert_eq!(
        captured.lock().unwrap()[0].body["messages"],
        json!([{"role": "tool", "tool_call_id": "call_image", "content": "[tool output omitted]"}])
    );
    assert!(log_text().contains("Replacing the tool output with a placeholder"));
}

#[test]
fn chat_completions_warns_once_for_responses_only_reasoning_settings() {
    let model = offline_model("gpt-5.6-sol");
    let req = ModelRequest {
        model_settings: ModelSettings {
            reasoning: Some(Reasoning {
                mode: Some("pro".into()),
                effort: Some("max".into()),
                context: Some("all_turns".into()),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..request()
    };
    let _guard = log_lock().blocking_lock();
    sink().0.lock().unwrap().clear();
    model.build_request_body(&req, false).unwrap();
    model.build_request_body(&req, false).unwrap();
    let text = log_text();
    assert_eq!(
        text.matches("Ignoring unsupported reasoning settings")
            .count(),
        1,
        "{text}"
    );
    assert!(text.contains("reasoning.mode"));
    assert!(text.contains("reasoning.context"));
}

#[test]
fn chat_completions_rejects_responses_only_reasoning_settings_in_strict_mode() {
    let model = offline_model("gpt-5.6-sol").with_strict_feature_validation(true);
    let req = ModelRequest {
        model_settings: ModelSettings {
            reasoning: Some(Reasoning {
                mode: Some("pro".into()),
                context: Some("all_turns".into()),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..request()
    };
    let err = model
        .build_request_body(&req, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("reasoning.mode"), "{err}");
}

// --------------------------------------------------------------------------------------
// Request construction
// --------------------------------------------------------------------------------------

#[test]
fn fetch_response_non_stream() {
    let req = ModelRequest {
        system_instructions: Some("sys".into()),
        input: user_input("hi"),
        model_settings: ModelSettings {
            reasoning: Some(Reasoning::effort("xhigh")),
            prompt_cache_retention: Some("24h".into()),
            prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
            ..Default::default()
        },
        ..request()
    };
    let body = offline_model("gpt-4")
        .build_request_body(&req, false)
        .unwrap();
    // Omitted defaults for non-official endpoints.
    for key in [
        "stream",
        "store",
        "tools",
        "tool_choice",
        "response_format",
        "stream_options",
    ] {
        assert!(body.get(key).is_none(), "{key} present in {body}");
    }
    assert_eq!(body["model"], "gpt-4");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "sys");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["reasoning_effort"], "xhigh");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
}

#[tokio::test]
async fn fetch_response_stream() {
    let (base, captured) = serve_queue(vec![sse_resp(&[])]).await;
    let model = model_at("gpt-4", &base);
    let req = ModelRequest {
        input: user_input("hi"),
        ..request()
    };
    let body = model.build_request_body(&req, true).unwrap();
    assert_eq!(body["stream"], true);
    assert!(body.get("store").is_none());
    assert!(body.get("stream_options").is_none());
    use futures::StreamExt;
    let events: Vec<Value> = model
        .stream_response(req)
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await;
    assert_eq!(captured.lock().unwrap()[0].body["stream"], true);
    // A bare in-progress response precedes the terminal event.
    let created = events.iter().find(|e| e["type"] == "response.created");
    if let Some(created) = created {
        assert_eq!(created["response"]["id"], "__fake_id__");
        assert_eq!(created["response"]["model"], "gpt-4");
        assert_eq!(created["response"]["object"], "response");
        assert_eq!(created["response"]["output"], json!([]));
    }
    assert_eq!(events.last().unwrap()["type"], "response.completed");
}

#[tokio::test]
async fn chat_completions_requests_normalize_agent_settings() {
    let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    let settings = ModelSettings {
        reasoning: Some(Reasoning::effort("high")),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        prompt_cache_retention: Some("24h".into()),
        verbosity: Some("low".into()),
        store: Some(false),
        temperature: Some(0.3),
        top_p: Some(1.0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        max_tokens: Some(64),
        parallel_tool_calls: Some(false),
        extra_headers: Some(
            [(
                "x-model-settings-parity".to_string(),
                "preserved".to_string(),
            )]
            .into(),
        ),
        extra_query: Some([("model_settings_parity".to_string(), "verified".to_string())].into()),
        extra_body: Some(
            [(
                "prompt_cache_key".to_string(),
                json!("extra-body-cache-key"),
            )]
            .into_iter()
            .collect(),
        ),
        ..Default::default()
    };
    model_at("gpt-4", &base)
        .get_response(ModelRequest {
            model_settings: settings,
            tools: vec![function_tool("test_tool")],
            ..request()
        })
        .await
        .unwrap();
    let req = captured.lock().unwrap()[0].clone();
    let b = &req.body;
    assert_eq!(b["reasoning_effort"], "high");
    assert_eq!(
        b["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
    assert_eq!(b["prompt_cache_retention"], "24h");
    assert_eq!(b["verbosity"], "low");
    assert_eq!(b["store"], false);
    assert_eq!(b["temperature"], 0.3);
    assert_eq!(b["top_p"], 1.0);
    assert_eq!(b["frequency_penalty"], 0.0);
    assert_eq!(b["presence_penalty"], 0.0);
    assert_eq!(b["max_tokens"], 64);
    assert!(b.get("max_output_tokens").is_none());
    assert_eq!(b["parallel_tool_calls"], false);
    assert_eq!(b["prompt_cache_key"], "extra-body-cache-key");
    assert!(b.get("retry").is_none());
    assert_eq!(req.headers["x-model-settings-parity"], "preserved");
    assert!(
        req.path.contains("model_settings_parity=verified"),
        "{}",
        req.path
    );
}

#[test]
fn custom_base_url_prompt_cache_key_uses_model_settings_only() {
    let model = offline_model("gpt-4");
    let default = model.build_request_body(&request(), false).unwrap();
    let explicit = model
        .build_request_body(
            &ModelRequest {
                model_settings: ModelSettings {
                    extra_args: Some(
                        [("prompt_cache_key".to_string(), json!("cache-key"))]
                            .into_iter()
                            .collect(),
                    ),
                    ..Default::default()
                },
                ..request()
            },
            false,
        )
        .unwrap();
    assert!(default.get("prompt_cache_key").is_none());
    assert_eq!(explicit["prompt_cache_key"], "cache-key");
}

#[test]
fn extra_args_prompt_cache_options_allowed_when_direct_field_is_omitted() {
    let options = json!({"mode": "explicit", "ttl": "30m"});
    let body = offline_model("gpt-4")
        .build_request_body(
            &ModelRequest {
                model_settings: ModelSettings {
                    extra_args: Some(
                        [("prompt_cache_options".to_string(), options.clone())]
                            .into_iter()
                            .collect(),
                    ),
                    ..Default::default()
                },
                ..request()
            },
            false,
        )
        .unwrap();
    assert_eq!(body["prompt_cache_options"], options);
}

#[test]
fn duplicate_prompt_cache_options_rejected() {
    let err = offline_model("gpt-4")
        .build_request_body(
            &ModelRequest {
                model_settings: ModelSettings {
                    prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
                    extra_args: Some(
                        [(
                            "prompt_cache_options".to_string(),
                            json!({"mode": "implicit"}),
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                },
                ..request()
            },
            false,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("multiple values"), "{err}");
    assert!(err.contains("prompt_cache_options"), "{err}");
}

#[tokio::test]
async fn get_response_accepts_raw_chat_completions_image_content() {
    let (base, captured) = serve_queue(vec![json_resp(text_completion("ok"))]).await;
    model_at("gpt-4", &base)
        .get_response(ModelRequest {
            input: vec![InputItem::from_value(json!({"role": "user", "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,AAAA", "detail": "high"}},
            ]}))],
            ..request()
        })
        .await
        .unwrap();
    assert_eq!(
        captured.lock().unwrap()[0].body["messages"],
        json!([{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,AAAA", "detail": "high"}},
        ]}])
    );
}

#[test]
fn store_param() {
    let official = OpenAIChatCompletionsModel::new("gpt-4", OpenAIClient::new("sk-test"));
    let body = |model: &OpenAIChatCompletionsModel, store: Option<bool>| {
        model
            .build_request_body(
                &ModelRequest {
                    model_settings: ModelSettings {
                        store,
                        ..Default::default()
                    },
                    ..request()
                },
                false,
            )
            .unwrap()
    };
    // OpenAI API calls default to store=true; explicit values are respected.
    assert_eq!(body(&official, None)["store"], true);
    assert_eq!(body(&official, Some(false))["store"], false);
    assert_eq!(body(&official, Some(true))["store"], true);
    // Other providers default to omitting it.
    let other = model_at("gpt-4", "http://www.notopenai.com");
    assert!(body(&other, None).get("store").is_none());
    assert_eq!(body(&other, Some(false))["store"], false);
    assert_eq!(body(&other, Some(true))["store"], true);
}

#[tokio::test]
async fn user_agent_header_chat_completions() {
    let (base, captured) = serve_queue(vec![json_resp(text_completion("Hello"))]).await;
    let model = model_at("gpt-4", &base);
    model.get_response(request()).await.unwrap();
    model
        .get_response(ModelRequest {
            model_settings: ModelSettings {
                extra_headers: Some(
                    [("User-Agent".to_string(), "test_user_agent".to_string())].into(),
                ),
                ..Default::default()
            },
            ..request()
        })
        .await
        .unwrap();
    let reqs = captured.lock().unwrap().clone();
    assert!(
        reqs[0].headers["user-agent"].starts_with("Agents/"),
        "{:?}",
        reqs[0].headers
    );
    assert_eq!(reqs[1].headers["user-agent"], "test_user_agent");
}

// --------------------------------------------------------------------------------------
// Retry advice
// --------------------------------------------------------------------------------------

fn api_error(headers: Option<&[(&str, &str)]>, status: u16) -> AgentsError {
    AgentsError::ApiError {
        status,
        message: "rate limited".into(),
        body: Some(Box::new(json!({"error": {"code": "rate_limit"}}))),
        request_id: headers
            .and_then(|h| h.iter().find(|(k, _)| *k == "x-request-id"))
            .map(|(_, v)| v.to_string()),
        headers: headers.map(|h| {
            h.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        }),
    }
}

#[test]
fn get_retry_advice_uses_openai_headers() {
    let error = api_error(
        Some(&[
            ("x-should-retry", "true"),
            ("retry-after-ms", "500"),
            ("x-request-id", "req_123"),
        ]),
        429,
    );
    let advice = offline_model("gpt-4")
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .unwrap();
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.retry_after, Some(0.5));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
    let normalized = advice.normalized.unwrap();
    assert_eq!(normalized.error_code.as_deref(), Some("rate_limit"));
    assert_eq!(normalized.status_code, Some(429));
    assert_eq!(normalized.request_id.as_deref(), Some("req_123"));
}

#[tokio::test]
async fn get_retry_advice_keeps_stateful_transport_failures_ambiguous() {
    // A real connection failure: nothing listens on the port.
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap().port()
    };
    let model = model_at("gpt-4", &format!("http://127.0.0.1:{port}/v1"));
    let error = model.get_response(request()).await.unwrap_err();
    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: Some("resp_prev"),
            conversation_id: None,
        })
        .unwrap();
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.replay_safety, None);
    assert!(advice.normalized.unwrap().is_network_error);
}

#[test]
fn get_retry_advice_marks_stateful_http_failures_replay_safe() {
    let error = api_error(None, 429);
    let advice = offline_model("gpt-4")
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: Some("resp_prev"),
            conversation_id: None,
        })
        .unwrap();
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
    assert_eq!(advice.normalized.unwrap().status_code, Some(429));
}

// --------------------------------------------------------------------------------------
// test_gpt6_chat_tool_defaults.py
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn gpt6_tool_request_defaults() {
    use futures::StreamExt;
    openai_agents_sdk::set_tracing_disabled(true);
    struct Case {
        name: &'static str,
        tools: &'static str,
        settings: ModelSettings,
        expected: Option<&'static str>,
    }
    let extra = |k: &str| -> Option<serde_json::Map<String, Value>> {
        Some([(k.to_string(), json!("low"))].into_iter().collect())
    };
    let cases = vec![
        Case {
            name: "gpt-6-sol",
            tools: "function",
            settings: ModelSettings::default(),
            expected: Some("none"),
        },
        Case {
            name: "gpt-6-luna",
            tools: "function",
            settings: ModelSettings::default(),
            expected: Some("none"),
        },
        Case {
            name: "gpt-6-sol",
            tools: "handoff",
            settings: ModelSettings::default(),
            expected: Some("none"),
        },
        Case {
            name: "gpt-6-sol",
            tools: "function",
            settings: ModelSettings {
                reasoning: Some(Reasoning::effort("high")),
                ..Default::default()
            },
            expected: Some("high"),
        },
        Case {
            name: "gpt-6-sol",
            tools: "function",
            settings: ModelSettings {
                extra_args: extra("reasoning_effort"),
                ..Default::default()
            },
            expected: Some("low"),
        },
        Case {
            name: "gpt-6-sol",
            tools: "function",
            settings: ModelSettings {
                extra_body: extra("reasoning_effort"),
                ..Default::default()
            },
            expected: Some("low"),
        },
        Case {
            name: "gpt-6-sol",
            tools: "none",
            settings: ModelSettings::default(),
            expected: None,
        },
        Case {
            name: "gpt-6.1-sol",
            tools: "function",
            settings: ModelSettings::default(),
            expected: None,
        },
        Case {
            name: "gpt-6-astra",
            tools: "function",
            settings: ModelSettings::default(),
            expected: None,
        },
        Case {
            name: "gpt-6-sol-unknown",
            tools: "function",
            settings: ModelSettings::default(),
            expected: None,
        },
    ];
    for case in cases {
        for stream in [false, true] {
            let reply = if stream {
                sse_resp(&[
                    json!({"id": "c", "object": "chat.completion.chunk", "created": 0,
                           "model": case.name, "choices": [{"index": 0, "finish_reason": null,
                           "delta": {"role": "assistant", "content": "sunny"}}]}),
                    json!({"id": "c", "object": "chat.completion.chunk", "created": 0,
                           "model": case.name, "choices": [{"index": 0, "delta": {},
                           "finish_reason": "stop"}]}),
                ])
            } else {
                json_resp(text_completion("sunny"))
            };
            let (base, captured) = serve_queue(vec![reply]).await;
            let model = model_at(case.name, &base);
            let tools = if case.tools == "function" {
                vec![function_tool("weather")]
            } else {
                vec![]
            };
            let handoffs = if case.tools == "handoff" {
                vec![FunctionToolSpec {
                    name: "transfer_to_other".into(),
                    description: "other".into(),
                    parameters: json!({"type": "object", "properties": {}}),
                    strict: true,
                    ..Default::default()
                }]
            } else {
                vec![]
            };
            let before = serde_json::to_value(&case.settings).unwrap();
            let req = ModelRequest {
                input: user_input("Weather?"),
                model_settings: case.settings.clone(),
                tools,
                handoffs,
                ..request()
            };
            if stream {
                let _: Vec<_> = model.stream_response(req).await.unwrap().collect().await;
            } else {
                model.get_response(req).await.unwrap();
            }
            let reqs = captured.lock().unwrap().clone();
            assert_eq!(reqs.len(), 1);
            let body = &reqs[0].body;
            match case.expected {
                None => assert!(
                    body.get("reasoning_effort").is_none(),
                    "{}: {body}",
                    case.name
                ),
                Some(e) => {
                    // `extra_body` is merged last, so it still wins over the default.
                    assert_eq!(body["reasoning_effort"], e, "{} {body}", case.name)
                }
            }
            assert_eq!(body.get("tools").is_some(), case.tools != "none");
            assert_eq!(serde_json::to_value(&case.settings).unwrap(), before);
        }
    }
}
