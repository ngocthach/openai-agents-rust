//! Port of `tests/models/test_reasoning_content.py`,
//! `tests/models/test_reasoning_content_replay_hook.py` and
//! `tests/models/test_deepseek_reasoning_content.py`: reasoning fields on Chat Completions
//! messages, streamed reasoning events, and replaying reasoning items into later requests.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::models::interface::{Model, ModelRequest};
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatConversionOptions, ReasoningContentReplayContext, items_to_messages_with,
    parse_chat_completion,
};
use openai_agents_sdk::models::{ModelTracing, OpenAIChatCompletionsModel, OpenAIClient};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---- local mock server -----------------------------------------------------------------

#[derive(Clone, Debug)]
struct Req {
    body: Value,
}

#[derive(Clone)]
struct Resp {
    content_type: &'static str,
    body: String,
}

fn json_resp(v: Value) -> Resp {
    Resp {
        content_type: "application/json",
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
        content_type: "text/event-stream",
        body,
    }
}

/// Answer requests in order (the last reply repeats); returns the base URL and request log.
async fn serve_queue(replies: Vec<Resp>) -> (String, Arc<Mutex<Vec<Req>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let queue = Arc::new(Mutex::new(VecDeque::from(replies)));
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let cap = cap.clone();
            let queue = queue.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let body = loop {
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
                        break serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                };
                cap.lock().unwrap().push(Req { body });
                let reply = {
                    let mut q = queue.lock().unwrap();
                    if q.len() > 1 {
                        q.pop_front().unwrap()
                    } else {
                        q.front().unwrap().clone()
                    }
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    reply.content_type,
                    reply.body.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(reply.body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), captured)
}

fn model_at(name: &str, base: &str) -> OpenAIChatCompletionsModel {
    OpenAIChatCompletionsModel::new(name, OpenAIClient::new("sk-test").with_base_url(base))
}

fn request(input: Vec<Value>) -> ModelRequest {
    ModelRequest {
        input: input.into_iter().map(InputItem::from_value).collect(),
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

async fn stream_events(model: &OpenAIChatCompletionsModel, req: ModelRequest) -> Vec<Value> {
    model
        .stream_response(req)
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await
}

fn types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

// ---- conversion helpers ------------------------------------------------------------------

/// Parse a chat completion message, then replay its output items into chat messages, the way
/// Python dumps the items and feeds them back to `Converter.items_to_messages`.
fn round_trip(
    message: Value,
    response_model: &str,
    replay_model: &str,
) -> (Vec<Value>, Vec<Value>) {
    let response = json!({"id": "chatcmpl-test", "model": response_model, "choices": [
        {"index": 0, "finish_reason": "stop", "message": message}]});
    let parsed = parse_chat_completion(&response, None, false).unwrap();
    let items: Vec<Value> = parsed.output.iter().map(|o| o.to_value()).collect();
    let messages = replay(
        items.clone(),
        replay_model,
        ChatConversionOptions::default(),
    );
    (items, messages)
}

fn replay(items: Vec<Value>, model: &str, mut options: ChatConversionOptions) -> Vec<Value> {
    options.model = Some(model.to_string());
    items_to_messages_with(
        &items
            .into_iter()
            .map(InputItem::from_value)
            .collect::<Vec<_>>(),
        None,
        &options,
    )
    .unwrap()
}

fn assistant_message(text: &str) -> Value {
    json!({"id": "__fake_id__", "type": "message", "role": "assistant", "status": "completed",
           "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]})
}

fn plaintext_reasoning(text: &str, model: &str) -> Value {
    json!({"id": "__fake_id__", "summary": [], "type": "reasoning",
           "content": [{"text": text, "type": "reasoning_text"}],
           "provider_data": {"model": model, "_chat_completions_reasoning_field": "reasoning"}})
}

fn summary_reasoning(text: &str, provider_data: Option<Value>) -> Value {
    let mut item = json!({"id": "__fake_id__", "summary": [{"text": text, "type": "summary_text"}],
        "type": "reasoning", "content": null, "encrypted_content": null, "status": null});
    if let Some(pd) = provider_data {
        item["provider_data"] = pd;
    }
    item
}

// =========================================================================================
// test_reasoning_content.py
// =========================================================================================

#[test]
fn plaintext_reasoning_round_trips_on_its_assistant_message() {
    let (items, messages) = round_trip(
        json!({"role": "assistant", "content": "The answer is 42.",
               "reasoning": "I should calculate this carefully."}),
        "openrouter/reasoning-model",
        "openrouter/reasoning-model",
    );
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(
        items[0]["content"][0]["text"],
        "I should calculate this carefully."
    );
    assert_eq!(
        messages,
        vec![json!({"role": "assistant", "content": "The answer is 42.",
                    "reasoning": "I should calculate this carefully."})]
    );
}

#[test]
fn plaintext_reasoning_round_trips_with_a_tool_call() {
    let (_, messages) = round_trip(
        json!({"role": "assistant", "content": null,
               "reasoning": "I should call the weather tool.",
               "tool_calls": [{"id": "call-weather", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}]}),
        "openrouter/reasoning-model",
        "openrouter/reasoning-model",
    );
    assert_eq!(messages.len(), 1);
    let assistant = &messages[0];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant.get("content"), Some(&Value::Null));
    assert_eq!(assistant["reasoning"], "I should call the weather tool.");
    assert_eq!(assistant["tool_calls"].as_array().unwrap().len(), 1);
}

#[test]
fn plaintext_reasoning_is_not_replayed_to_a_different_model() {
    let (_, messages) = round_trip(
        json!({"role": "assistant", "content": "Answer", "reasoning": "Private reasoning"}),
        "openrouter/model-a",
        "openrouter/model-b",
    );
    assert_eq!(messages.len(), 1);
    assert!(messages[0].get("reasoning").is_none());
}

#[test]
fn intervening_reasoning_item_clears_pending_plaintext_reasoning() {
    let intervening = [
        json!({"id": "__fake_id__", "summary": [], "type": "reasoning",
               "content": [{"text": "Foreign reasoning", "type": "reasoning_text"}],
               "provider_data": {"model": "openrouter/model-b",
                                 "_chat_completions_reasoning_field": "reasoning"}}),
        json!({"id": "__fake_id__",
               "summary": [{"text": "Higher-fidelity reasoning", "type": "summary_text"}],
               "type": "reasoning", "content": null,
               "provider_data": {"model": "openrouter/model-a"}}),
    ];
    for item in intervening {
        let messages = replay(
            vec![
                plaintext_reasoning("Stale reasoning", "openrouter/model-a"),
                item,
                assistant_message("Answer"),
            ],
            "openrouter/model-a",
            ChatConversionOptions::default(),
        );
        assert_eq!(messages.len(), 1);
        assert!(messages[0].get("reasoning").is_none());
    }
}

#[test]
fn plaintext_reasoning_item_clears_pending_native_reasoning_content() {
    let messages = replay(
        vec![
            summary_reasoning(
                "Stale native reasoning",
                Some(json!({"model": "deepseek-r1"})),
            ),
            plaintext_reasoning("Fresh plaintext reasoning", "deepseek-r1"),
            assistant_message("Answer"),
        ],
        "deepseek-r1",
        ChatConversionOptions::default(),
    );
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["reasoning"], "Fresh plaintext reasoning");
    assert!(messages[0].get("reasoning_content").is_none());
}

#[test]
fn native_thinking_blocks_take_precedence_over_marked_plaintext_reasoning() {
    let blocks = json!([{"type": "thinking", "thinking": "Native thinking", "signature": "sig"}]);
    let messages = replay(
        vec![
            json!({"id": "__fake_id__", "summary": [], "type": "reasoning",
                   "content": [{"text": "Plaintext reasoning", "type": "reasoning_text"}],
                   "provider_data": {"model": "anthropic/claude-x",
                                     "_chat_completions_reasoning_field": "reasoning",
                                     "thinking_blocks": blocks}}),
            assistant_message("Answer"),
        ],
        "anthropic/claude-x",
        ChatConversionOptions {
            preserve_thinking_blocks: true,
            ..Default::default()
        },
    );
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["thinking_blocks"], blocks);
    assert!(messages[0].get("reasoning").is_none());
}

#[test]
fn plaintext_reasoning_item_clears_pending_thinking_blocks() {
    for native in [false, true] {
        let mut provider_data = json!({"model": "anthropic/claude-x"});
        if native {
            provider_data["thinking_blocks"] =
                json!([{"type": "thinking", "thinking": "Stale thinking", "signature": "sig"}]);
        }
        let messages = replay(
            vec![
                json!({"id": "__fake_id__", "summary": [], "type": "reasoning",
                       "content": [{"text": "Stale thinking", "type": "reasoning_text"}],
                       "provider_data": provider_data}),
                plaintext_reasoning("Fresh plaintext reasoning", "anthropic/claude-x"),
                assistant_message("Answer"),
            ],
            "anthropic/claude-x",
            ChatConversionOptions {
                preserve_thinking_blocks: true,
                ..Default::default()
            },
        );
        assert_eq!(
            messages,
            vec![json!({"role": "assistant", "content": "Answer",
                        "reasoning": "Fresh plaintext reasoning"})],
            "native={native}"
        );
    }
}

#[test]
fn reasoning_content_takes_precedence_over_plaintext_reasoning() {
    let (items, messages) = round_trip(
        json!({"role": "assistant", "content": "Answer", "reasoning": "Plaintext reasoning",
               "reasoning_content": "Existing reasoning content"}),
        "deepseek-reasoner",
        "deepseek-reasoner",
    );
    assert!(items[0].get("content").is_none_or(Value::is_null));
    assert_eq!(items[0]["summary"][0]["text"], "Existing reasoning content");
    assert_eq!(
        messages[0]["reasoning_content"],
        "Existing reasoning content"
    );
    assert!(messages[0].get("reasoning").is_none());
}

fn reasoning_delta(content: &str) -> Value {
    json!({"content": null, "role": null, "function_call": null, "tool_calls": null,
           "reasoning_content": content})
}

fn content_delta(content: &str) -> Value {
    json!({"content": content, "role": null, "function_call": null, "tool_calls": null})
}

fn chunk(delta: Value, usage: bool) -> Value {
    let mut c = json!({"id": "chunk-id", "created": 1, "model": "deepseek is usually expected",
        "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]});
    if usage {
        c["usage"] = json!({"completion_tokens": 4, "prompt_tokens": 2, "total_tokens": 6,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 0}});
    }
    c
}

/// The chunks followed by a terminal `finish_reason: stop` chunk.
fn with_terminal(mut chunks: Vec<Value>) -> Vec<Value> {
    let mut terminal = chunk(json!({}), false);
    terminal["choices"][0]["finish_reason"] = json!("stop");
    chunks.push(terminal);
    chunks
}

#[tokio::test]
async fn stream_response_yields_events_for_reasoning_content() {
    let (base, _) = serve_queue(vec![sse_resp(&with_terminal(vec![
        chunk(reasoning_delta("Let me think"), false),
        chunk(reasoning_delta(" about this"), false),
        chunk(content_delta("The answer"), false),
        chunk(content_delta(" is 42"), true),
    ]))])
    .await;
    let events = stream_events(&model_at("gpt-4", &base), request(vec![])).await;

    let reasoning_deltas: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "response.reasoning_summary_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(reasoning_deltas, ["Let me think", " about this"]);

    let kinds = types(&events);
    let done = kinds
        .iter()
        .position(|t| *t == "response.reasoning_summary_part.done")
        .unwrap();
    let first_text = kinds
        .iter()
        .position(|t| *t == "response.output_text.delta")
        .unwrap();
    assert!(done < first_text);

    let content_deltas: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(content_deltas, ["The answer", " is 42"]);

    // Every assistant-message event sits at output index 1, after the reasoning item.
    let mut message_events = 0;
    for e in &events {
        let t = e["type"].as_str().unwrap();
        let is_message = match t {
            "response.output_item.added" | "response.output_item.done" => {
                e["item"]["type"] == "message"
            }
            "response.content_part.added"
            | "response.output_text.delta"
            | "response.content_part.done" => true,
            _ => false,
        };
        if is_message {
            message_events += 1;
            assert_eq!(e["output_index"], 1, "{e}");
        }
    }
    assert!(message_events > 0);

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["summary"][0]["text"], "Let me think about this");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "The answer is 42");
}

#[tokio::test]
async fn stream_response_keeps_reasoning_item_open_across_interleaved_text() {
    let (base, _) = serve_queue(vec![sse_resp(&with_terminal(vec![
        chunk(reasoning_delta("Let me think"), false),
        chunk(content_delta("The answer"), false),
        chunk(reasoning_delta(" more carefully"), false),
        chunk(content_delta(" is 42"), true),
    ]))])
    .await;
    let events = stream_events(&model_at("gpt-4", &base), request(vec![])).await;

    let of = |t: &str| -> Vec<&Value> { events.iter().filter(|e| e["type"] == t).collect() };
    let added: Vec<Value> = of("response.reasoning_summary_part.added")
        .iter()
        .map(|e| e["summary_index"].clone())
        .collect();
    assert_eq!(added, [json!(0), json!(1)]);
    let done: Vec<Value> = of("response.reasoning_summary_part.done")
        .iter()
        .map(|e| e["summary_index"].clone())
        .collect();
    assert_eq!(done, [json!(0), json!(1)]);

    let pos = |f: &dyn Fn(&Value) -> bool| events.iter().position(f).unwrap();
    let first_done = pos(&|e| e["type"] == "response.reasoning_summary_part.done");
    let first_text = pos(&|e| e["type"] == "response.output_text.delta");
    let second_reasoning =
        pos(&|e| e["type"] == "response.reasoning_summary_text.delta" && e["summary_index"] == 1);
    let item_done =
        pos(&|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "reasoning");
    assert!(first_done < first_text);
    assert!(second_reasoning > first_text);
    assert!(item_done > second_reasoning);

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let summaries: Vec<&str> = completed["response"]["output"][0]["summary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["text"].as_str().unwrap())
        .collect();
    assert_eq!(summaries, ["Let me think", " more carefully"]);
}

#[tokio::test]
async fn get_response_with_reasoning_content() {
    let (base, _) = serve_queue(vec![json_resp(json!({
        "id": "resp-id", "created": 0, "model": "deepseek is expected",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "delta": null,
            "message": {"role": "assistant", "content": "The answer is 42",
                        "reasoning_content": "Let me think about this question carefully"}}],
        "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15,
                  "completion_tokens_details": {"reasoning_tokens": 6},
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }))])
    .await;
    let resp = model_at("gpt-4", &base)
        .get_response(request(vec![]))
        .await
        .unwrap();
    assert_eq!(resp.output.len(), 2);
    let reasoning = resp.output[0].to_value();
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(
        reasoning["summary"][0]["text"],
        "Let me think about this question carefully"
    );
    let message = resp.output[1].to_value();
    assert_eq!(message["type"], "message");
    assert_eq!(message["content"][0]["text"], "The answer is 42");
}

#[tokio::test]
async fn stream_response_preserves_usage_from_earlier_chunk() {
    let (base, _) = serve_queue(vec![sse_resp(&with_terminal(vec![
        chunk(content_delta("Hello"), true),
        chunk(content_delta(""), false),
    ]))])
    .await;
    let events = stream_events(&model_at("gpt-4", &base), request(vec![])).await;
    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let usage = &completed["response"]["usage"];
    assert_eq!(usage["input_tokens"], 2);
    assert_eq!(usage["output_tokens"], 4);
    assert_eq!(usage["total_tokens"], 6);
}

#[tokio::test]
async fn stream_response_with_empty_reasoning_content() {
    let (base, _) = serve_queue(vec![sse_resp(&with_terminal(vec![
        chunk(reasoning_delta(""), false),
        chunk(content_delta("The answer is 42"), true),
    ]))])
    .await;
    let events = stream_events(&model_at("gpt-4", &base), request(vec![])).await;
    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    // Only the message, no empty reasoning item.
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "The answer is 42");
}

// =========================================================================================
// test_reasoning_content_replay_hook.py
// =========================================================================================

const MODEL_A: &str = "reasoning-content-model-a";
const MODEL_B: &str = "reasoning-content-model-b";
// The converter keys Anthropic thinking-block reconstruction off the model name.
const MODEL_C: &str = "reasoning-content-model-c-anthropic";

fn weather_call(model: &str) -> Value {
    json!({"arguments": "{\"city\": \"Tokyo\"}", "call_id": "call_weather_123",
           "name": "get_weather", "type": "function_call", "id": "__fake_id__",
           "status": null, "provider_data": {"model": model}})
}

fn weather_output() -> Value {
    json!({"type": "function_call_output", "call_id": "call_weather_123",
           "output": "The weather in Tokyo is sunny and 22\u{b0}C."})
}

fn first_reasoning(model: &str) -> Value {
    summary_reasoning(
        "I should call the weather tool first.",
        Some(json!({"model": model, "response_id": "chatcmpl-test"})),
    )
}

fn second_turn_input_items(model: &str) -> Vec<Value> {
    vec![
        json!({"role": "user", "content": "What's the weather in Tokyo?"}),
        first_reasoning(model),
        weather_call(model),
        weather_output(),
    ]
}

fn assistant_with_tool_calls(messages: &[Value]) -> Value {
    messages
        .iter()
        .find(|m| {
            m["role"] == "assistant" && m["tool_calls"].as_array().is_some_and(|c| !c.is_empty())
        })
        .cloned()
        .expect("an assistant message with tool_calls")
}

fn always_replay() -> ChatConversionOptions {
    ChatConversionOptions {
        should_replay_reasoning_content: Some(Arc::new(|_ctx: &ReasoningContentReplayContext| {
            true
        })),
        ..Default::default()
    }
}

#[test]
fn converter_keeps_default_reasoning_replay_behavior_for_non_default_model() {
    let messages = replay(
        second_turn_input_items(MODEL_A),
        MODEL_A,
        ChatConversionOptions::default(),
    );
    assert!(
        assistant_with_tool_calls(&messages)
            .get("reasoning_content")
            .is_none()
    );
}

#[test]
fn converter_preserves_reasoning_content_across_output_message_with_hook() {
    let mut message = assistant_message("I'll call the weather tool now.");
    message["provider_data"] = json!({"model": MODEL_A, "response_id": "chatcmpl-test"});
    let messages = replay(
        vec![
            json!({"role": "user", "content": "What's the weather in Tokyo?"}),
            first_reasoning(MODEL_A),
            message,
            weather_call(MODEL_A),
            weather_output(),
        ],
        MODEL_A,
        always_replay(),
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(assistant["content"], "I'll call the weather tool now.");
    assert_eq!(
        assistant["reasoning_content"],
        "I should call the weather tool first."
    );
}

#[test]
fn converter_replays_reasoning_content_when_reasoning_follows_message_with_hook() {
    let mut message = assistant_message("I'll call the weather tool now.");
    message["provider_data"] = json!({"model": MODEL_A, "response_id": "chatcmpl-test"});
    let messages = replay(
        vec![
            json!({"role": "user", "content": "What's the weather in Tokyo?"}),
            message,
            first_reasoning(MODEL_A),
            weather_call(MODEL_A),
            weather_output(),
        ],
        MODEL_A,
        always_replay(),
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(assistant["content"], "I'll call the weather tool now.");
    assert_eq!(
        assistant["reasoning_content"],
        "I should call the weather tool first."
    );
}

#[test]
fn converter_replays_reasoning_content_for_file_search_call_with_hook() {
    let messages = replay(
        vec![
            json!({"role": "user", "content": "Find notes about Tokyo weather."}),
            summary_reasoning(
                "I should search the knowledge base first.",
                Some(json!({"model": MODEL_A, "response_id": "chatcmpl-test"})),
            ),
            json!({"id": "__fake_file_search_id__", "queries": ["Tokyo weather"],
                   "status": "completed", "type": "file_search_call"}),
        ],
        MODEL_A,
        always_replay(),
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "I should search the knowledge base first."
    );
    assert_eq!(
        assistant["tool_calls"][0]["function"]["name"],
        "file_search_call"
    );
}

#[test]
fn converter_replays_reasoning_content_with_thinking_blocks_and_hook() {
    let mut options = always_replay();
    options.preserve_thinking_blocks = true;
    let messages = replay(
        vec![
            json!({"role": "user", "content": "What's the weather in Tokyo?"}),
            json!({"id": "__fake_id__",
                   "summary": [{"text": "I should call the weather tool first.",
                                "type": "summary_text"}],
                   "type": "reasoning",
                   "content": [{"type": "reasoning_text",
                                "text": "First, I need to inspect the request."}],
                   "encrypted_content": "test-signature", "status": null,
                   "provider_data": {"model": MODEL_C, "response_id": "chatcmpl-test"}}),
            weather_call(MODEL_C),
            weather_output(),
        ],
        MODEL_C,
        options,
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "I should call the weather tool first."
    );
    assert_eq!(assistant["content"][0]["type"], "thinking");
    assert_eq!(
        assistant["content"][0]["thinking"],
        "First, I need to inspect the request."
    );
}

#[tokio::test]
async fn openai_chatcompletions_hook_can_enable_reasoning_content_replay() {
    let (base, captured) = serve_queue(vec![json_resp(json!({
        "id": "test-id", "created": 0, "model": MODEL_B, "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "done"}}],
        "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15},
    }))])
    .await;
    let contexts: Arc<Mutex<Vec<ReasoningContentReplayContext>>> = Arc::default();
    let seen = contexts.clone();
    let model = model_at(MODEL_B, &base).with_should_replay_reasoning_content(move |ctx| {
        seen.lock().unwrap().push(ctx.clone());
        ctx.model == MODEL_B
    });
    model
        .get_response(request(second_turn_input_items(MODEL_B)))
        .await
        .unwrap();

    let messages: Vec<Value> = captured.lock().unwrap()[0].body["messages"]
        .as_array()
        .unwrap()
        .clone();
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "I should call the weather tool first."
    );
    let contexts = contexts.lock().unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].model, MODEL_B);
    assert_eq!(contexts[0].base_url.as_deref(), Some(base.as_str()));
    assert_eq!(contexts[0].origin_model.as_deref(), Some(MODEL_B));
}

// =========================================================================================
// test_deepseek_reasoning_content.py
// =========================================================================================

/// Two-turn exchange against a mock provider: the first reply carries `reasoning_content` and a
/// tool call, and the second request must replay that reasoning on the assistant message.
async fn deepseek_two_turns(call_id: &str, output: &str) -> Vec<Value> {
    let (base, captured) = serve_queue(vec![
        json_resp(
            json!({"id": "r1", "created": 0, "model": "deepseek-reasoner",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "reasoning_content": "Let me think about getting the weather for Tokyo...",
                "tool_calls": [{"id": call_id, "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]}}],
            "usage": {"completion_tokens": 50, "prompt_tokens": 100, "total_tokens": 150}}),
        ),
        json_resp(
            json!({"id": "r2", "created": 0, "model": "deepseek-reasoner",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "The weather in Tokyo is sunny."}}],
            "usage": {"completion_tokens": 50, "prompt_tokens": 100, "total_tokens": 150}}),
        ),
    ])
    .await;
    let model = model_at("deepseek-reasoner", &base);
    let first = model
        .get_response(request(vec![json!({"role": "user",
                                           "content": "What's the weather in Tokyo?"})]))
        .await
        .unwrap();
    assert!(!first.output.is_empty());
    let mut input = vec![json!({"role": "user", "content": "What's the weather in Tokyo?"})];
    input.extend(first.output.iter().map(|o| o.to_value()));
    input.push(json!({"type": "function_call_output", "call_id": call_id, "output": output}));
    model.get_response(request(input)).await.unwrap();
    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    requests[1].body["messages"].as_array().unwrap().clone()
}

#[tokio::test]
async fn deepseek_reasoning_content_preserved_in_tool_calls() {
    let messages = deepseek_two_turns("call_123", "The weather in Tokyo is sunny.").await;
    let assistant = assistant_with_tool_calls(&messages);
    assert!(assistant.get("reasoning_content").is_some(), "{assistant}");
}

#[tokio::test]
async fn deepseek_reasoning_content_in_multi_turn_conversation() {
    let messages =
        deepseek_two_turns("call_weather_123", "The weather in Tokyo is sunny and 22C.").await;
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "Let me think about getting the weather for Tokyo..."
    );
}

fn paris_items(
    reasoning_provider_data: Option<Value>,
    call_provider_data: Option<Value>,
    call_id: &str,
) -> Vec<Value> {
    let mut call = json!({"arguments": "{\"city\": \"Paris\"}", "call_id": call_id,
                          "name": "get_weather", "type": "function_call",
                          "id": "__fake_id__", "status": null});
    if let Some(pd) = call_provider_data {
        call["provider_data"] = pd;
    }
    vec![
        json!({"role": "user", "content": "What's the weather in Paris?"}),
        summary_reasoning(
            "I need to check the weather in Paris.",
            reasoning_provider_data,
        ),
        call,
        json!({"type": "function_call_output", "call_id": call_id,
               "output": "The weather in Paris is cloudy and 15\u{b0}C."}),
    ]
}

#[test]
fn deepseek_reasoning_content_with_openai_chatcompletions_path() {
    let messages = replay(
        paris_items(
            Some(json!({"model": "deepseek-reasoner", "response_id": "chatcmpl-test"})),
            Some(json!({"model": "deepseek-reasoner"})),
            "call_weather_456",
        ),
        "deepseek-reasoner",
        ChatConversionOptions::default(),
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "I need to check the weather in Paris."
    );
}

#[test]
fn reasoning_content_from_other_provider_not_attached_to_deepseek() {
    let messages = replay(
        paris_items(
            Some(json!({"model": "claude-sonnet-4-20250514", "response_id": "chatcmpl-test"})),
            Some(json!({"model": "claude-sonnet-4-20250514"})),
            "call_weather_789",
        ),
        "deepseek-reasoner",
        ChatConversionOptions::default(),
    );
    assert!(
        assistant_with_tool_calls(&messages)
            .get("reasoning_content")
            .is_none()
    );
}

#[test]
fn reasoning_content_without_provider_data_attached_for_backward_compat() {
    let messages = replay(
        paris_items(None, None, "call_weather_101"),
        "deepseek-reasoner",
        ChatConversionOptions::default(),
    );
    let assistant = assistant_with_tool_calls(&messages);
    assert_eq!(
        assistant["reasoning_content"],
        "I need to check the weather in Paris."
    );
}
