//! Ports of two cases of `tests/models/test_openai_chatcompletions_stream.py`:
//! a consumer that stops reading closes the provider stream, and a streamed tool call without
//! provider metadata carries no `provider_data`.

use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::models::interface::{Model, ModelRequest};
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatStreamOptions, handle_chat_stream_chunks,
};
use openai_agents_sdk::models::{ModelTracing, OpenAIChatCompletionsModel, OpenAIClient};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Python: `test_stream_response_close_closes_provider_stream_with_async_close`.
#[tokio::test]
async fn dropping_the_event_stream_closes_the_provider_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (closed_tx, closed_rx) = tokio::sync::oneshot::channel::<()>();
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
            if let Some(idx) = text.find("\r\n\r\n") {
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
        }
        let chunk = json!({"id": "chunk-id", "created": 1, "model": "fake",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}]});
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        let first = format!("data: {chunk}\n\n");
        let _ = sock.write_all(head.as_bytes()).await;
        let _ = sock
            .write_all(format!("{:x}\r\n{first}\r\n", first.len()).as_bytes())
            .await;
        // The provider keeps the stream open until the client goes away.
        loop {
            match sock.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = closed_tx.send(());
    });

    let model = OpenAIChatCompletionsModel::new(
        "gpt-4",
        OpenAIClient::new("sk-test").with_base_url(format!("http://{addr}/v1")),
    );
    let mut stream = model
        .stream_response(ModelRequest {
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .unwrap();
    assert!(first.unwrap().is_ok());
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), closed_rx)
        .await
        .expect("the provider connection should be closed")
        .unwrap();
}

/// Python: `test_function_call_item_omits_provider_data_when_absent`.
#[test]
fn streamed_function_call_without_provider_metadata_has_no_provider_data() {
    let chunks = [
        json!({"created": 1, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call-id", "type": "function",
                "function": {"name": "my_func", "arguments": "{}"}}]}}]}),
        json!({"created": 1, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let events = handle_chat_stream_chunks(&chunks, &ChatStreamOptions::default()).unwrap();
    let completed = events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .expect("completed event");
    let call = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call item");
    assert_eq!(call["arguments"], "{}");
    assert!(call.get("provider_data").is_none(), "{call}");
}
