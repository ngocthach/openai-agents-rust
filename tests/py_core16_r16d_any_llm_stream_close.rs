//! `tests/models/test_any_llm_model.py` stream-cleanup cases. Python injects `aclose()` failures
//! and slow closes into a fake provider iterator; the Rust adapter streams over HTTP, so the
//! observable halves are that abandoning a stream releases the connection at once (nothing can
//! make that wait), and that a connection that breaks before the terminal event is an error
//! rather than a silently completed response.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::extensions::models::{AdapterApi, AnyLLMModel};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// How the server ends its stream after the first event.
#[derive(Clone, Copy)]
enum Ending {
    /// Keep the connection open until the client closes it.
    WaitForClient,
    /// Drop the connection without finishing the chunked body.
    Truncate,
}

/// Serve one SSE response that sends `first` and then ends as `ending` says. The flag is set once
/// the client side of the connection is closed.
async fn serve(first: Value, ending: Ending) -> (String, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let closed = Arc::new(AtomicBool::new(false));
    let flag = closed.clone();
    tokio::spawn(async move {
        let Ok((mut sock, _)) = listener.accept().await else {
            return;
        };
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
            let len: usize = text[..idx]
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= idx + 4 + len {
                break;
            }
        }
        let event = format!(
            "event: {}\ndata: {first}\n\n",
            first["type"].as_str().unwrap_or("message")
        );
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                    transfer-encoding: chunked\r\n\r\n";
        let chunk = format!("{:x}\r\n{event}\r\n", event.len());
        if sock
            .write_all(format!("{head}{chunk}").as_bytes())
            .await
            .is_err()
        {
            return;
        }
        match ending {
            Ending::Truncate => {}
            Ending::WaitForClient => {
                // Reading returns once the client has closed its end.
                while matches!(sock.read(&mut tmp).await, Ok(n) if n > 0) {}
                flag.store(true, Ordering::SeqCst);
            }
        }
    });
    (format!("http://{addr}"), closed)
}

fn request() -> ModelRequest {
    ModelRequest {
        input: vec![InputItem(json!({"role": "user", "content": "hi"}))],
        model_settings: ModelSettings::default(),
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

fn model(api: AdapterApi, name: &str, url: &str) -> AnyLLMModel {
    AnyLLMModel::new(name, Some(url), Some("sk-test"), Some(api)).unwrap()
}

fn chat_chunk() -> Value {
    json!({"id": "chunk_123", "object": "chat.completion.chunk", "created": 0, "model": "fake",
           "choices": [{"index": 0, "delta": {"content": "He"}, "finish_reason": null}]})
}

fn responses_created() -> Value {
    json!({"type": "response.created",
           "response": {"id": "resp_1", "object": "response", "status": "in_progress",
                        "output": []}})
}

async fn wait_closed(closed: &AtomicBool) -> bool {
    for _ in 0..200 {
        if closed.load(Ordering::SeqCst) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

/// `test_any_llm_chat_stream_closes_provider_stream_after_cancellation`,
/// `..._does_not_block_cancellation_on_slow_close`, `..._lets_in_flight_close_finish_after_
/// cancellation` (chat) and the Responses counterparts: abandoning the consumer releases the
/// provider connection immediately, however far the stream had got.
#[tokio::test]
async fn abandoning_a_stream_closes_the_provider_connection() {
    for (api, name, first) in [
        (
            AdapterApi::ChatCompletions,
            "openrouter/openai/gpt-5.4-mini",
            chat_chunk(),
        ),
        (
            AdapterApi::Responses,
            "openai/gpt-5.4-mini",
            responses_created(),
        ),
    ] {
        let (url, closed) = serve(first, Ending::WaitForClient).await;
        let mut events = model(api, name, &url)
            .stream_response(request())
            .await
            .unwrap();
        let event = tokio::time::timeout(Duration::from_secs(5), events.next())
            .await
            .expect("the first event arrives")
            .expect("an event")
            .expect("no error");
        assert!(event["type"].is_string());
        assert!(!closed.load(Ordering::SeqCst));

        drop(events);
        assert!(
            wait_closed(&closed).await,
            "connection not released ({name})"
        );
    }
}

/// `test_any_llm_chat_stream_propagates_close_failure_before_terminal_event` and the Responses
/// counterpart: a stream that breaks before its terminal event stays observable to the caller.
#[tokio::test]
async fn a_connection_that_breaks_before_the_terminal_event_is_an_error() {
    for (api, name, first) in [
        (
            AdapterApi::ChatCompletions,
            "openrouter/openai/gpt-5.4-mini",
            chat_chunk(),
        ),
        (
            AdapterApi::Responses,
            "openai/gpt-5.4-mini",
            responses_created(),
        ),
    ] {
        let (url, _closed) = serve(first, Ending::Truncate).await;
        let mut events = model(api, name, &url)
            .stream_response(request())
            .await
            .unwrap();
        let mut failure = None;
        let mut types = Vec::new();
        while let Some(event) = tokio::time::timeout(Duration::from_secs(5), events.next())
            .await
            .expect("the stream ends")
        {
            match event {
                Ok(value) => types.push(value["type"].as_str().unwrap_or_default().to_string()),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        assert!(
            failure.is_some(),
            "{name}: the broken stream ended quietly with {types:?}"
        );
        assert!(
            !types.iter().any(|t| t == "response.completed"),
            "{name}: {types:?}"
        );
    }
}
