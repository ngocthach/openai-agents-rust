//! Anthropic interleaved thinking and Gemini thought signatures survive a Chat Completions
//! round trip: the provider's blocks become a reasoning item, and the next request replays
//! them verbatim. Runs against a local mock server.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::{OpenAIChatCompletionsModel, OpenAIClient};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn mock_server(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<Value>>>) {
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
                break serde_json::from_slice::<Value>(&buf[idx + 4..idx + 4 + len])
                    .unwrap_or(Value::Null);
            };
            cap.lock().unwrap().push(body);
            let reply = replies.next().expect("unexpected request").to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{}",
                reply.len(),
                reply
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

fn reasoning_settings() -> ModelSettings {
    ModelSettings {
        reasoning: Some(Reasoning {
            effort: Some("high".into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn claude_thinking_blocks_round_trip() {
    let (base, captured) = mock_server(vec![
        json!({
            "id": "c1", "model": "claude-sonnet-4-5",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": null,
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "let me check", "signature": "sig-1"},
                    {"type": "redacted_thinking", "data": "opaque"}
                ],
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "look", "arguments": "{}"}}]
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }),
        json!({
            "id": "c2", "model": "claude-sonnet-4-5",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }),
    ])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAIChatCompletionsModel::new("claude-sonnet-4-5", client)
        .with_preserve_thinking_blocks(true);

    let first = model
        .get_response(ModelRequest {
            input: vec![InputItem(json!({"role": "user", "content": "go"}))],
            model_settings: reasoning_settings(),
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();

    // The whole signed sequence is kept on the reasoning item, plus the normalized view.
    let reasoning = first.output[0].to_value();
    assert_eq!(reasoning["type"], "reasoning");
    let blocks = &reasoning["provider_data"]["thinking_blocks"];
    assert_eq!(blocks.as_array().unwrap().len(), 2);
    assert_eq!(blocks[1]["type"], "redacted_thinking");
    assert_eq!(reasoning["encrypted_content"], "sig-1");
    assert_eq!(reasoning["content"][0]["text"], "let me check");

    // Replaying the turn puts them back on the assistant message verbatim.
    let mut input: Vec<InputItem> = vec![InputItem(json!({"role": "user", "content": "go"}))];
    input.extend(first.output.iter().map(|o| InputItem(o.to_value())));
    input.push(InputItem(
        json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
    ));
    model
        .get_response(ModelRequest {
            input,
            model_settings: reasoning_settings(),
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();

    let bodies = captured.lock().unwrap().clone();
    let messages = bodies[1]["messages"].as_array().unwrap();
    let assistant = messages
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("assistant message");
    assert_eq!(assistant["thinking_blocks"], *blocks);
}

#[tokio::test]
async fn thinking_blocks_are_only_replayed_when_reasoning_is_requested() {
    let (base, captured) = mock_server(vec![json!({
        "id": "c1", "model": "claude-sonnet-4-5",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAIChatCompletionsModel::new("claude-sonnet-4-5", client)
        .with_preserve_thinking_blocks(true);
    let reasoning_item = json!({
        "type": "reasoning", "id": "r1",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "thought"}],
        "encrypted_content": "sig-1",
        "provider_data": {"model": "claude-sonnet-4-5", "thinking_blocks": [
            {"type": "thinking", "thinking": "thought", "signature": "sig-1"}]},
    });
    model
        .get_response(ModelRequest {
            input: vec![
                InputItem(json!({"role": "user", "content": "go"})),
                InputItem(reasoning_item),
                InputItem(json!({"role": "assistant", "content": "earlier"})),
            ],
            // No `reasoning.effort`: LiteLLM would not preserve the blocks either.
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let bodies = captured.lock().unwrap().clone();
    let assistant = bodies[0]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap()
        .clone();
    assert!(assistant.get("thinking_blocks").is_none());
}

#[tokio::test]
async fn gemini_thought_signatures_survive_a_round_trip() {
    let (base, captured) = mock_server(vec![
        json!({
            "id": "g1", "model": "gemini-2.5-pro",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "extra_content": {"google": {"thought_signature": "sig-g"}},
                                "function": {"name": "look", "arguments": "{}"}}]
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }),
        json!({
            "id": "g2", "model": "gemini-2.5-pro",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }),
    ])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAIChatCompletionsModel::new("gemini-2.5-pro", client);
    let first = model
        .get_response(ModelRequest {
            input: vec![InputItem(json!({"role": "user", "content": "go"}))],
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let call = first.output[0].to_value();
    assert_eq!(call["provider_data"]["thought_signature"], "sig-g");

    let mut input: Vec<InputItem> = vec![InputItem(json!({"role": "user", "content": "go"}))];
    input.extend(first.output.iter().map(|o| InputItem(o.to_value())));
    input.push(InputItem(
        json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
    ));
    model
        .get_response(ModelRequest {
            input,
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let bodies = captured.lock().unwrap().clone();
    let assistant = bodies[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m.get("tool_calls").is_some())
        .unwrap()
        .clone();
    assert_eq!(
        assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        "sig-g"
    );
}

#[tokio::test]
async fn anthropic_native_messages_round_trip() {
    use openai_agents_sdk::extensions::models::anthropic::AnthropicMessagesModel;

    let (base, captured) = mock_server(vec![
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "checking", "signature": "sig-1"},
                {"type": "tool_use", "id": "toolu_1", "name": "look", "input": {"city": "Hue"}}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 3}
        }),
        json!({
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Sunny"}],
            "usage": {"input_tokens": 12, "output_tokens": 2}
        }),
    ])
    .await;
    let client = OpenAIClient::new(String::new())
        .with_base_url(base)
        .with_bearer_auth(false)
        .with_header("x-api-key", "sk-ant-test")
        .with_header("anthropic-version", "2023-06-01");
    let model = AnthropicMessagesModel::with_client("claude-sonnet-4-5", client);

    let first = model
        .get_response(ModelRequest {
            system_instructions: Some("be brief".into()),
            input: vec![InputItem(json!({"role": "user", "content": "weather?"}))],
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(first.usage.input_tokens, 13);
    assert_eq!(first.usage.input_tokens_details.cached_tokens, 3);
    let reasoning = first.output[0].to_value();
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(reasoning["encrypted_content"], "sig-1");
    let call = first.output[1].to_value();
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["call_id"], "toolu_1");
    assert_eq!(call["arguments"], "{\"city\":\"Hue\"}");

    let mut input: Vec<InputItem> = vec![InputItem(json!({"role": "user", "content": "weather?"}))];
    input.extend(first.output.iter().map(|o| InputItem(o.to_value())));
    input.push(InputItem(
        json!({"type": "function_call_output", "call_id": "toolu_1", "output": "sunny"}),
    ));
    let second = model
        .get_response(ModelRequest {
            input,
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(second.output[0].to_value()["content"][0]["text"], "Sunny");

    let bodies = captured.lock().unwrap().clone();
    assert_eq!(bodies[0]["system"], "be brief");
    assert_eq!(bodies[0]["max_tokens"], 4096);
    let replay = bodies[1]["messages"].as_array().unwrap();
    assert_eq!(replay[0]["role"], "user");
    assert_eq!(replay[1]["role"], "assistant");
    // The signed thinking block is replayed before the tool use, as Anthropic requires.
    assert_eq!(replay[1]["content"][0]["type"], "thinking");
    assert_eq!(replay[1]["content"][0]["signature"], "sig-1");
    assert_eq!(replay[1]["content"][1]["type"], "tool_use");
    assert_eq!(replay[2]["content"][0]["type"], "tool_result");
    assert_eq!(replay[2]["content"][0]["tool_use_id"], "toolu_1");
}
