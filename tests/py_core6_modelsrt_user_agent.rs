//! The override half of `test_user_agent_header_litellm` and `test_user_agent_header_any_llm_chat`:
//! a caller-supplied `User-Agent` header replaces the SDK default. Python sets it through the
//! `HEADERS_OVERRIDE` context variable; Rust carries it in `ModelSettings::extra_headers`, as the
//! Chat Completions port already does.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel, LitellmModel};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve one-token completions and record each request's `user-agent` header.
async fn serve(agents: Arc<Mutex<Vec<String>>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let agents = agents.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                loop {
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
                    let header = |name: &str| {
                        text[..idx]
                            .lines()
                            .filter_map(|l| l.split_once(':'))
                            .find(|(k, _)| k.trim().eq_ignore_ascii_case(name))
                            .map(|(_, v)| v.trim().to_string())
                    };
                    let len: usize = header("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() < idx + 4 + len {
                        continue;
                    }
                    agents
                        .lock()
                        .unwrap()
                        .push(header("user-agent").unwrap_or_default());
                    break;
                }
                let body = json!({
                    "id": "chatcmpl_ua", "created": 0, "model": "test-model",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "Hello"}}],
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                     connection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

fn request(user_agent: Option<&str>) -> ModelRequest {
    ModelRequest {
        tracing: ModelTracing::Disabled,
        model_settings: ModelSettings {
            extra_headers: user_agent.map(|ua| {
                [("User-Agent".to_string(), ua.to_string())]
                    .into_iter()
                    .collect()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn a_caller_user_agent_replaces_the_default_on_litellm_and_any_llm() {
    for adapter in ["litellm", "any-llm"] {
        for override_ua in [None, Some("test_user_agent")] {
            let agents = Arc::new(Mutex::new(Vec::new()));
            let base = serve(agents.clone()).await;
            let model: Arc<dyn Model> = if adapter == "litellm" {
                Arc::new(
                    LitellmModel::with_options("test-model", Some(&base), Some("sk-test"), None)
                        .unwrap(),
                )
            } else {
                Arc::new(
                    AnyLLMModel::new(
                        "openrouter/openai/gpt-5.4-mini",
                        Some(&base),
                        Some("sk-test"),
                        Some(AdapterApi::ChatCompletions),
                    )
                    .unwrap(),
                )
            };
            model.get_response(request(override_ua)).await.unwrap();

            let expected = override_ua
                .map(str::to_string)
                .unwrap_or_else(|| format!("Agents/Rust {}", env!("CARGO_PKG_VERSION")));
            assert_eq!(
                *agents.lock().unwrap(),
                vec![expected],
                "{adapter} {override_ua:?}"
            );
        }
    }
}
