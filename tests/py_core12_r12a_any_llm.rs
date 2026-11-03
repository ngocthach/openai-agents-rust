//! `tests/models/test_any_llm_model.py` cases that the earlier ports left as Python-only although
//! the Rust any-llm adapter shows the behaviour over HTTP: provider-specific `extra_body` fields
//! next to explicit settings, and a chat message carrying both `reasoning` and
//! `reasoning_content`.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Local mock server
// ---------------------------------------------------------------------------

/// A canned JSON reply.
#[derive(Clone, Debug)]
enum Reply {
    Json(Value),
}

/// One captured request.
#[derive(Clone, Debug)]
struct Captured {
    body: Value,
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
            };
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}"), captured)
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

fn map(entries: &[(&str, Value)]) -> serde_json::Map<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// `test_any_llm_chat_nests_extra_body_instead_of_flattening`: provider-specific fields reach the
/// request exactly once, a key that collides with an explicit setting is resolved the way the
/// OpenAI client resolves `extra_body` (the extra field wins), and the caller's own `extra_body`
/// map is not altered by the call.
#[tokio::test]
async fn chat_extra_body_adds_provider_fields_once() {
    let (url, captured) = mock_server(vec![Reply::Json(chat_completion("Hello"))]).await;
    let extra_body = map(&[
        ("cached_content", json!("some_cache")),
        ("foo", json!(123)),
        ("temperature", json!(0.9)),
    ]);
    let settings = ModelSettings {
        temperature: Some(0.1),
        extra_body: Some(extra_body.clone()),
        ..Default::default()
    };

    chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(settings.clone()))
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert_eq!(body["cached_content"], "some_cache");
    assert_eq!(body["foo"], 123);
    assert_eq!(body["temperature"], 0.9, "{body}");
    assert_eq!(
        body.to_string().matches("\"temperature\"").count(),
        1,
        "{body}"
    );
    assert_eq!(settings.extra_body, Some(extra_body));
}

/// `test_any_llm_nonstream_preserves_native_reasoning_content_field`: when a chat message has
/// both the plaintext `reasoning` and the native `reasoning_content`, the native text is what
/// becomes the reasoning item.
#[tokio::test]
async fn native_reasoning_content_wins_over_plaintext_reasoning() {
    let mut reply = chat_completion("Answer");
    reply["choices"][0]["message"]["reasoning"] = json!("Plaintext reasoning");
    reply["choices"][0]["message"]["reasoning_content"] = json!("Native reasoning content");
    let (url, _captured) = mock_server(vec![Reply::Json(reply)]).await;

    let response = chat_model("deepseek/deepseek-reasoner", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let reasoning: Vec<Value> = output_values(&response)
        .into_iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 1, "{reasoning:?}");
    assert_eq!(
        reasoning[0]["summary"][0]["text"],
        "Native reasoning content"
    );
    assert!(
        !reasoning[0].to_string().contains("Plaintext reasoning"),
        "{reasoning:?}"
    );
}
