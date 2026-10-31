//! Parity port of `tests/models/test_anthropic_thinking_blocks.py`,
//! `test_gemini_thought_signatures.py` and `test_gemini_thought_signatures_stream.py`.
//!
//! Python drives `Converter.message_to_output_items` / `Converter.items_to_messages` directly.
//! Rust exposes the replay half as `models::openai_chatcompletions::items_to_messages_with`, and
//! the response half only through a model, so the response-side cases run a chat completion
//! against a local mock server and read the items it produced.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatConversionOptions, OpenAIChatCompletionsModel, items_to_messages_with,
};
use openai_agents_sdk::models::openai_client::OpenAIClient;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Start a server replying with each canned body (JSON, or SSE when `sse` is set) in order.
async fn mock_server(replies: Vec<(bool, Vec<Value>)>) -> (String, Arc<Mutex<Vec<Value>>>) {
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
                let Some(idx) = text.find("\r\n\r\n") else {
                    continue;
                };
                let len: usize = text[..idx]
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse().ok())
                    })
                    .unwrap_or(0);
                if buf.len() < idx + 4 + len {
                    continue;
                }
                break serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null);
            };
            cap.lock().unwrap().push(body);
            let (sse, values) = replies.next().expect("unexpected request");
            let resp = if sse {
                let mut b = String::new();
                for v in values {
                    b.push_str(&format!("data: {v}\n\n"));
                }
                b.push_str("data: [DONE]\n\n");
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{}",
                    b.len(),
                    b
                )
            } else {
                let b = values[0].to_string();
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{}",
                    b.len(),
                    b
                )
            };
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}"), captured)
}

fn chat_reply(model: &str, message: Value) -> Value {
    json!({
        "id": "msg-thinking", "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

fn reasoning_settings() -> ModelSettings {
    ModelSettings {
        reasoning: Some(Reasoning::effort("high")),
        ..Default::default()
    }
}

/// Run one chat completion and return the Responses output items it produced.
async fn output_items(model_name: &str, message: Value) -> Vec<Value> {
    let (base, _captured) = mock_server(vec![(false, vec![chat_reply(model_name, message)])]).await;
    let model = OpenAIChatCompletionsModel::new(
        model_name,
        OpenAIClient::new("sk-test").with_base_url(base),
    )
    .with_preserve_thinking_blocks(true);
    let response = model
        .get_response(ModelRequest {
            input: vec![InputItem(json!({"role": "user", "content": "go"}))],
            model_settings: reasoning_settings(),
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    response.output.iter().map(|o| o.to_value()).collect()
}

fn options(model: &str, preserve_thinking_blocks: bool) -> ChatConversionOptions {
    ChatConversionOptions {
        model: Some(model.to_string()),
        preserve_thinking_blocks,
        ..Default::default()
    }
}

fn to_messages(items: &[Value], model: &str, preserve: bool) -> Vec<Value> {
    let items: Vec<InputItem> = items.iter().cloned().map(InputItem).collect();
    items_to_messages_with(&items, None, &options(model, preserve)).unwrap()
}

fn assistant_thinking_blocks(message: &Value) -> &Vec<Value> {
    let blocks = message
        .get("thinking_blocks")
        .and_then(Value::as_array)
        .expect("thinking_blocks");
    assert!(blocks.iter().all(Value::is_object));
    blocks
}

const CLAUDE: &str = "anthropic/claude-sonnet-4-5";

// ---------------------------------------------------------------------------
// test_anthropic_thinking_blocks.py
// ---------------------------------------------------------------------------

/// A reasoning item in the history contributes no thinking content of its own.
#[test]
fn converter_skips_reasoning_items() {
    let items = [
        json!({"role": "user", "content": "Hello"}),
        json!({"id": "reasoning_123", "type": "reasoning",
               "summary": [{"text": "User said hello", "type": "summary_text"}]}),
        json!({"id": "msg_123", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Hi there!"}]}),
    ];

    let messages = to_messages(&items, "gpt-5", false);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    if let Some(parts) = messages[1]["content"].as_array() {
        assert!(parts.iter().all(|p| p["type"] != "thinking"));
    }
}

/// Anthropic requires the thinking blocks to lead the assistant turn when reasoning is on and
/// tool calls are present; the signatures travel with them.
#[tokio::test]
async fn anthropic_thinking_blocks_with_tool_calls() {
    let message = json!({
        "role": "assistant",
        "content": "I'll check the weather for you.",
        "reasoning_content": "The user wants weather information, I need to call the weather \
                              function",
        "thinking_blocks": [
            {"type": "thinking",
             "thinking": "The user is asking about weather. Let me use the weather tool to get \
                          this information.",
             "signature": "TestSignature123"},
            {"type": "thinking", "thinking": "We should use the city Tokyo as the city.",
             "signature": "TestSignature456"}
        ],
        "tool_calls": [{"id": "call_123", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]
    });

    let items = output_items(CLAUDE, message).await;

    let reasoning: Vec<&Value> = items.iter().filter(|i| i["type"] == "reasoning").collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["content"][0]["type"], "reasoning_text");
    assert_eq!(
        reasoning[0]["encrypted_content"],
        "TestSignature123\nTestSignature456"
    );
    let calls: Vec<&Value> = items
        .iter()
        .filter(|i| i["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 1);

    let messages = to_messages(&items, CLAUDE, true);
    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .collect();
    assert_eq!(assistants.len(), 1);
    let blocks = assistant_thinking_blocks(assistants[0]);
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(
        blocks[0]["thinking"],
        "The user is asking about weather. Let me use the weather tool to get this information."
    );
    assert_eq!(blocks[0]["signature"], "TestSignature123");
    assert_eq!(blocks[1]["type"], "thinking");
    assert_eq!(
        blocks[1]["thinking"],
        "We should use the city Tokyo as the city."
    );
    assert_eq!(blocks[1]["signature"], "TestSignature456");
    assert_eq!(assistants[0]["content"], "I'll check the weather for you.");
    assert_eq!(assistants[0]["tool_calls"].as_array().unwrap().len(), 1);
    assert_eq!(
        assistants[0]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

/// Thinking blocks are attached even when the turn produced no tool calls.
#[tokio::test]
async fn anthropic_thinking_blocks_without_tool_calls() {
    let message = json!({
        "role": "assistant",
        "content": "The weather in Paris is sunny with a temperature of 22°C.",
        "reasoning_content": "The user wants to know about the weather in Paris.",
        "thinking_blocks": [{"type": "thinking",
                             "thinking": "Let me think about the weather in Paris.",
                             "signature": "TestSignatureNoTools123"}]
    });

    let items = output_items(CLAUDE, message).await;

    let reasoning: Vec<&Value> = items.iter().filter(|i| i["type"] == "reasoning").collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["content"][0]["type"], "reasoning_text");
    assert_eq!(
        reasoning[0]["content"][0]["text"],
        "Let me think about the weather in Paris."
    );
    assert_eq!(reasoning[0]["encrypted_content"], "TestSignatureNoTools123");
    assert_eq!(items.iter().filter(|i| i["type"] == "message").count(), 1);

    let messages = to_messages(&items, CLAUDE, true);
    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1);
    let blocks = assistant_thinking_blocks(assistants[0]);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(
        blocks[0]["thinking"],
        "Let me think about the weather in Paris."
    );
    assert_eq!(blocks[0]["signature"], "TestSignatureNoTools123");
    assert_eq!(
        assistants[0]["content"],
        "The weather in Paris is sunny with a temperature of 22°C."
    );
}

/// A block whose visible thinking was omitted still has to survive the round trip verbatim,
/// signature and all — Anthropic validates the sequence it issued.
#[tokio::test]
async fn round_trip_preserves_an_omitted_thinking_block() {
    let blocks = json!([{"type": "thinking", "thinking": "", "signature": "OmittedSignature"}]);
    let message = json!({"role": "assistant", "content": "answer", "thinking_blocks": blocks});

    let items = output_items(CLAUDE, message).await;

    let reasoning: Vec<&Value> = items.iter().filter(|i| i["type"] == "reasoning").collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["summary"], json!([]));
    assert_eq!(reasoning[0]["provider_data"]["thinking_blocks"], blocks);

    let messages = to_messages(&items, CLAUDE, true);
    let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
    assert_eq!(assistant["thinking_blocks"], blocks);
}

/// The complete block sequence — omitted, redacted and visible — is replayed in order.
#[tokio::test]
async fn round_trip_preserves_the_complete_thinking_block_sequence() {
    let blocks = json!([
        {"type": "thinking", "thinking": "", "signature": "OmittedSignature"},
        {"type": "redacted_thinking", "data": "EncryptedRedactedThinking"},
        {"type": "thinking", "thinking": "visible", "signature": "VisibleSignature"}
    ]);
    let message = json!({
        "role": "assistant", "content": null, "thinking_blocks": blocks,
        "tool_calls": [{"id": "toolu-weather", "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": "{\"city\": \"Tokyo\"}"}}]
    });

    let items = output_items(CLAUDE, message).await;
    let messages = to_messages(&items, CLAUDE, true);

    let assistant = messages
        .iter()
        .find(|m| m.get("tool_calls").is_some())
        .unwrap();
    assert_eq!(assistant["thinking_blocks"], blocks);
    assert_eq!(assistant["tool_calls"][0]["id"], "toolu-weather");
    assert_eq!(
        assistant["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

/// Replay is refused when it was turned off, when the stored item came from another model, and
/// when the item carries provider metadata from an unknown origin.
#[tokio::test]
async fn complete_thinking_blocks_respect_replay_guards() {
    let message = json!({
        "role": "assistant", "content": "answer", "reasoning_content": "visible",
        "thinking_blocks": [{"type": "thinking", "thinking": "visible", "signature": "Signature"}]
    });
    let items = output_items(CLAUDE, message.clone()).await;

    let disabled = to_messages(&items, CLAUDE, false);
    let mismatched = to_messages(&items, "anthropic/claude-opus-4-5", true);

    let mut unknown_origin = items.clone();
    for item in &mut unknown_origin {
        if item["type"] == "reasoning" {
            item["provider_data"] = json!({
                "response_id": "response-from-unknown-provider",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "visible", "signature": "Signature"}
                ],
            });
        }
    }
    let unknown_origin = to_messages(&unknown_origin, CLAUDE, true);

    for messages in [disabled, mismatched, unknown_origin] {
        assert!(messages.iter().all(|m| m.get("thinking_blocks").is_none()));
    }
}

/// A reasoning item with no provider origin still replays DeepSeek's plaintext reasoning, and
/// never the thinking blocks.
#[tokio::test]
async fn originless_thinking_blocks_preserve_legacy_deepseek_reasoning_replay() {
    let message = json!({
        "role": "assistant", "content": "answer", "reasoning_content": "legacy reasoning",
        "thinking_blocks": [{"type": "thinking", "thinking": "visible", "signature": "Signature"}]
    });

    let mut items = output_items("deepseek-reasoner", message).await;
    for item in &mut items {
        if item["type"] == "reasoning" {
            // Python builds the item without provider data at all; the Rust model always stamps
            // the originating model, so strip it back to the legacy shape under test.
            item["provider_data"] = json!({
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "visible", "signature": "Signature"}
                ]
            });
        }
    }

    let messages = to_messages(&items, "deepseek/deepseek-reasoner", true);

    let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
    assert_eq!(assistant["reasoning_content"], "legacy reasoning");
    assert!(assistant.get("thinking_blocks").is_none());
}

/// A stored reasoning item with no native block list is rebuilt into inline thinking blocks in
/// front of the assistant content, with the signatures taken from `encrypted_content`.
#[test]
fn legacy_reasoning_item_reconstructs_inline_thinking_blocks() {
    for provider_data in [None, Some(json!({"thinking_blocks": ["invalid-block"]}))] {
        let mut reasoning = json!({
            "id": "reasoning_legacy", "type": "reasoning", "summary": [],
            "content": [{"type": "reasoning_text", "text": "legacy thinking"}],
            "encrypted_content": "LegacySignature"
        });
        if let Some(data) = provider_data {
            reasoning["provider_data"] = data;
        }
        let items = [
            reasoning,
            json!({"id": "message_legacy", "type": "message", "status": "completed",
                   "role": "assistant",
                   "content": [{"type": "output_text", "text": "answer", "annotations": []}]}),
        ];

        let messages = to_messages(&items, CLAUDE, true);

        assert_eq!(
            messages[0]["content"],
            json!([
                {"type": "thinking", "thinking": "legacy thinking",
                 "signature": "LegacySignature"},
                {"type": "text", "text": "answer"}
            ])
        );
        assert!(messages[0].get("thinking_blocks").is_none());
    }
}

/// A reasoning item not followed by its own assistant message belongs to an earlier turn, so
/// its signed blocks must not be prepended to a later assistant message.
#[tokio::test]
async fn thinking_blocks_do_not_leak_across_an_intervening_user_turn() {
    let silent_turn = json!({
        "role": "assistant", "content": "", "reasoning_content": "Nothing to say yet.",
        "thinking_blocks": [{"type": "thinking", "thinking": "Earlier private thinking.",
                             "signature": "EarlierTurnSignature"}]
    });

    let items = output_items(CLAUDE, silent_turn).await;
    assert_eq!(
        items
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["reasoning"]
    );

    let mut history = vec![json!({"role": "user", "content": "first question"})];
    history.extend(items);
    history.push(json!({"role": "user", "content": "second question"}));
    history.push(
        json!({"id": "msg_2", "type": "message", "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "an answer",
                                     "annotations": []}]}),
    );

    let messages = to_messages(&history, CLAUDE, true);

    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1);
    assert_eq!(assistants[0]["content"], "an answer");
}

// ---------------------------------------------------------------------------
// test_gemini_thought_signatures.py
// ---------------------------------------------------------------------------

/// Each tool call keeps its own thought signature.
#[tokio::test]
async fn gemini_multiple_tool_calls_with_thought_signatures() {
    let message = json!({
        "role": "assistant", "content": "Calling two functions.",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "func_a", "arguments": "{\"x\": 1}"},
             "extra_content": {"google": {"thought_signature": "sig_aaa"}}},
            {"id": "call_2", "type": "function",
             "function": {"name": "func_b", "arguments": "{\"y\": 2}"},
             "extra_content": {"google": {"thought_signature": "sig_bbb"}}}
        ]
    });

    let items = output_items("gemini/gemini-3-pro", message).await;

    let calls: Vec<&Value> = items
        .iter()
        .filter(|i| i["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["provider_data"]["thought_signature"], "sig_aaa");
    assert_eq!(calls[1]["provider_data"]["thought_signature"], "sig_bbb");
}

/// `items_to_messages` restores the Google `extra_content` shape from the stored provider data.
#[test]
fn gemini_thought_signature_items_to_messages() {
    let items = [
        json!({"role": "user", "content": "test"}),
        json!({
            "id": "fake-id", "call_id": "call_restore", "name": "restore_func",
            "arguments": "{\"test\": true}", "type": "function_call",
            "provider_data": {"model": "gemini/gemini-3-pro",
                              "response_id": "gemini-response-id-123",
                              "thought_signature": "restored_sig_xyz"}
        }),
    ];

    let messages = to_messages(&items, "gemini/gemini-3-pro", false);

    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1);
    let tool_calls = assistants[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0]["extra_content"]["google"]["thought_signature"],
        "restored_sig_xyz"
    );
}

// ---------------------------------------------------------------------------
// test_gemini_thought_signatures_stream.py
// ---------------------------------------------------------------------------

/// Stream one tool call whose signature arrives in `field`, and return the finished item.
async fn streamed_tool_call_item(first_delta_extra: Value) -> Value {
    use futures::StreamExt;

    let mut tool_call = json!({
        "index": 0, "id": "call_stream", "type": "function",
        "function": {"name": "get_weather", "arguments": ""}
    });
    for (k, v) in first_delta_extra.as_object().unwrap() {
        tool_call[k] = v.clone();
    }
    let chunks = vec![
        json!({"id": "chunk-id-123", "object": "chat.completion.chunk", "created": 1,
               "model": "gemini/gemini-3-pro",
               "choices": [{"index": 0, "finish_reason": null,
                            "delta": {"role": "assistant", "tool_calls": [tool_call]}}]}),
        json!({"id": "chunk-id-123", "object": "chat.completion.chunk", "created": 1,
               "model": "gemini/gemini-3-pro",
               "choices": [{"index": 0, "finish_reason": null, "delta": {"tool_calls": [
                   {"index": 0, "function": {"arguments": "{\"city\": \"Tokyo\"}"}}]}}]}),
        json!({"id": "chunk-id-456", "object": "chat.completion.chunk", "created": 1,
               "model": "gemini/gemini-3-pro",
               "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]}),
    ];
    let (base, _captured) = mock_server(vec![(true, chunks)]).await;
    let model = OpenAIChatCompletionsModel::new(
        "gemini/gemini-3-pro",
        OpenAIClient::new("sk-test").with_base_url(base),
    );

    let mut events = model
        .stream_response(ModelRequest {
            input: vec![InputItem(json!({"role": "user", "content": "go"}))],
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let mut done = Vec::new();
    while let Some(event) = events.next().await {
        let event = event.unwrap();
        if event["type"] == "response.output_item.done" && event["item"]["type"] == "function_call"
        {
            done.push(event["item"].clone());
        }
    }
    assert_eq!(done.len(), 1);
    done.remove(0)
}

/// A signature carried in LiteLLM's `provider_specific_fields` reaches the finished call.
#[tokio::test]
async fn stream_captures_provider_specific_fields_thought_signature() {
    let item = streamed_tool_call_item(
        json!({"provider_specific_fields": {"thought_signature": "litellm_sig_123"}}),
    )
    .await;

    let provider_data = &item["provider_data"];
    assert_eq!(provider_data["thought_signature"], "litellm_sig_123");
    assert_eq!(provider_data["model"], "gemini/gemini-3-pro");
    assert_eq!(provider_data["response_id"], "chunk-id-123");
}

/// A signature carried in Google's `extra_content` shape reaches the finished call.
#[tokio::test]
async fn stream_captures_google_extra_content_thought_signature() {
    let item = streamed_tool_call_item(
        json!({"extra_content": {"google": {"thought_signature": "google_sig_456"}}}),
    )
    .await;

    let provider_data = &item["provider_data"];
    assert_eq!(provider_data["thought_signature"], "google_sig_456");
    assert_eq!(provider_data["model"], "gemini/gemini-3-pro");
    assert_eq!(provider_data["response_id"], "chunk-id-123");
}

// ---------------------------------------------------------------------------
// Response-side conversion
// ---------------------------------------------------------------------------

/// Reasoning content and the signed thinking blocks of a response are stored on a reasoning
/// item, next to its summary.
#[tokio::test]
async fn reasoning_items_preserved_in_message_conversion() {
    let blocks = json!([{"type": "thinking",
                         "thinking": "I need to call the weather function for Paris",
                         "signature": "EqMDCkYIBxgCKkBAFZO8"}]);
    let message = json!({
        "role": "assistant",
        "content": "I'll check the weather in Paris for you.",
        "reasoning_content": "I need to call the weather function for Paris",
        "thinking_blocks": blocks,
    });

    let items = output_items(CLAUDE, message).await;

    let reasoning: Vec<&Value> = items.iter().filter(|i| i["type"] == "reasoning").collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(
        reasoning[0]["summary"][0]["text"],
        "I need to call the weather function for Paris"
    );
    assert_eq!(reasoning[0]["provider_data"]["thinking_blocks"], blocks);
    if let Some(content) = reasoning[0]["content"].as_array().filter(|c| !c.is_empty()) {
        assert_eq!(content[0]["type"], "reasoning_text");
        assert_eq!(
            content[0]["text"],
            "I need to call the weather function for Paris"
        );
    }
}
