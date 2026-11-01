//! Cases of `tests/models/test_litellm_extra_body.py` and
//! `tests/models/test_litellm_chatcompletions_stream.py` that the earlier audits carried over
//! as python-only: `extra_body["metadata"]` next to top-level `metadata`, and the provider
//! stream being released when the consumer stops reading (Python's `aclose` / cancellation;
//! Rust's drop closes the HTTP connection).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::extensions::models::LitellmModel;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Read one HTTP request and return its JSON body.
async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Value> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
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
            return serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).ok();
        }
    }
}

fn chunk(text: &str, finish: Option<&str>) -> String {
    let chunk = json!({
        "id": "chatcmpl_test", "created": 0, "model": "test-model",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "finish_reason": finish,
                     "delta": {"role": "assistant", "content": text}}],
    });
    format!("data: {chunk}\n\n")
}

fn request_with(settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem(json!({"role": "user", "content": "test input"}))],
        model_settings: settings,
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

fn model(base_url: &str) -> LitellmModel {
    LitellmModel::with_options("test-model", Some(base_url), Some("sk-test"), None).unwrap()
}

#[tokio::test]
async fn extra_body_metadata_stays_distinct_from_top_level_metadata() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captured: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = captured.clone();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let body = read_request(&mut sock).await.unwrap();
        sink.lock().unwrap().push(body);
        let payload = json!({
            "id": "c", "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
        .to_string();
        let _ = sock
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{payload}",
                    payload.len()
                )
                .as_bytes(),
            )
            .await;
    });

    let mut extra_body = Map::new();
    extra_body.insert(
        "metadata".into(),
        json!({"trace_user_id": "user-123", "generation_id": "gen-456"}),
    );
    extra_body.insert("cached_content".into(), json!("some_cache"));
    let metadata = std::collections::HashMap::from([("sdk".to_string(), "agents".to_string())]);
    let settings = ModelSettings {
        metadata: Some(metadata),
        extra_body: Some(extra_body),
        ..Default::default()
    };
    model(&url)
        .get_response(request_with(settings))
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].clone();
    // Both metadata maps are the caller's: the body keeps the nested one intact and the
    // provider-specific field next to it.
    assert_eq!(body["cached_content"], "some_cache");
    let metadata = body["metadata"].as_object().expect("metadata object");
    let nested_kept = metadata.get("trace_user_id") == Some(&json!("user-123"))
        && metadata.get("generation_id") == Some(&json!("gen-456"));
    assert!(
        nested_kept,
        "extra_body metadata must reach the provider unflattened: {body}"
    );
}

/// A provider endpoint that sends one chunk and then keeps the stream open, reporting when the
/// client closes its side of the connection.
async fn open_stream_server() -> (String, tokio::sync::oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        read_request(&mut sock).await.unwrap();
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                    connection: close\r\n\r\n";
        sock.write_all(head.as_bytes()).await.unwrap();
        sock.write_all(chunk("He", None).as_bytes()).await.unwrap();
        // Never finish the stream: wait for the client to hang up.
        let mut tmp = [0u8; 64];
        loop {
            match sock.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = closed_tx.send(());
    });
    (url, closed_rx)
}

#[tokio::test]
async fn dropping_the_stream_early_releases_the_provider_connection() {
    let (url, closed) = open_stream_server().await;
    let mut stream = model(&url)
        .stream_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let first = stream.next().await.expect("first event").unwrap();
    assert_eq!(first["type"], "response.created");
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), closed)
        .await
        .expect("provider connection was not released")
        .unwrap();
}

#[tokio::test]
async fn cancelling_the_consumer_mid_stream_releases_the_provider_connection() {
    let (url, closed) = open_stream_server().await;
    let model = model(&url);
    let consumer = tokio::spawn(async move {
        let mut stream = model
            .stream_response(request_with(ModelSettings::default()))
            .await
            .unwrap();
        while let Some(event) = stream.next().await {
            event.unwrap();
        }
    });
    // Let the consumer receive the chunk and block on the open stream.
    tokio::time::sleep(Duration::from_millis(300)).await;
    consumer.abort();
    assert!(consumer.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), closed)
        .await
        .expect("provider connection was not released after the consumer was cancelled")
        .unwrap();
}

#[tokio::test]
async fn exhausting_the_stream_completes_and_leaves_no_open_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        read_request(&mut sock).await.unwrap();
        let payload = format!(
            "{}{}data: [DONE]\n\n",
            chunk("He", None),
            chunk("llo", Some("stop"))
        );
        sock.write_all(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{payload}"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        // The client must not keep the socket open after the stream ended.
        let mut tmp = [0u8; 64];
        loop {
            match sock.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = closed_tx.send(());
    });
    let mut stream = model(&url)
        .stream_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    let mut types = Vec::new();
    while let Some(event) = stream.next().await {
        types.push(event.unwrap()["type"].as_str().unwrap().to_string());
    }
    assert_eq!(types.last().map(String::as_str), Some("response.completed"));
    tokio::time::timeout(Duration::from_secs(5), closed_rx)
        .await
        .expect("connection stayed open after the stream was exhausted")
        .unwrap();
}
