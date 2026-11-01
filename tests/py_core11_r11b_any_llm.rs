//! Parity ports of three `tests/models/test_any_llm_model.py` cases the earlier audits filed as
//! Python-only although the Rust any-llm adapter has the behaviour: a Responses payload without
//! `cache_write_tokens`, an unset `prompt_cache_retention` staying off the request, and a chat
//! completion filtered by the provider surfacing as a refusal.

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

fn first_text(response: &openai_agents_sdk::items::ModelResponse) -> String {
    output_values(response)
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .find(|p| p["type"] == "output_text")
        .and_then(|p| p["text"].as_str().map(str::to_string))
        .unwrap_or_default()
}

/// `test_any_llm_responses_path_defaults_missing_cache_write_tokens`
#[tokio::test]
async fn responses_usage_without_cache_write_tokens_defaults_to_zero() {
    let payload = responses_payload("Hello");
    assert!(
        payload["usage"]["input_tokens_details"]
            .get("cache_write_tokens")
            .is_none()
    );
    let (url, _captured) = mock_server(vec![Reply::Json(payload)]).await;
    let settings = ModelSettings {
        preserve_raw_usage: Some(true),
        ..Default::default()
    };

    let response = responses_model("openai/gpt-5.4-mini", &url)
        .get_response(request_with(settings))
        .await
        .unwrap();

    assert_eq!(first_text(&response), "Hello");
    assert_eq!(response.usage.input_tokens_details.cache_write_tokens, 0);
    let raw = response
        .raw_usage
        .expect("raw usage is preserved on request");
    assert!(
        raw["input_tokens_details"]
            .get("cache_write_tokens")
            .is_none()
    );
}

/// `test_any_llm_responses_path_omits_prompt_cache_retention_when_unset`
#[tokio::test]
async fn responses_request_omits_prompt_cache_retention_when_unset() {
    let (url, captured) = mock_server(vec![
        Reply::Json(responses_payload("Hello")),
        Reply::Json(responses_payload("Hello")),
    ])
    .await;
    let model = responses_model("openai/gpt-5.4-mini", &url);

    model
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let pinned = ModelSettings {
        prompt_cache_retention: Some("24h".to_string()),
        ..Default::default()
    };
    model.get_response(request_with(pinned)).await.unwrap();

    let captured = captured.lock().unwrap();
    assert!(captured[0].body.get("prompt_cache_retention").is_none());
    assert_eq!(captured[1].body["prompt_cache_retention"], "24h");
}

/// `test_any_llm_chat_path_surfaces_content_filter_refusal`
#[tokio::test]
async fn chat_content_filter_with_no_text_becomes_a_refusal() {
    let mut completion = chat_completion("");
    completion["choices"][0]["finish_reason"] = json!("content_filter");
    let (url, _captured) = mock_server(vec![Reply::Json(completion)]).await;

    let response = chat_model("openrouter/openai/gpt-5.4-mini", &url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let refusals: Vec<Value> = output_values(&response)
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .filter(|part| part["type"] == "refusal")
        .collect();
    assert_eq!(refusals.len(), 1, "{:?}", output_values(&response));
    assert!(!refusals[0]["refusal"].as_str().unwrap_or("").is_empty());
    assert!(first_text(&response).is_empty());
}
