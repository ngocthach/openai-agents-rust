//! Parity port of the behavioral cases of `tests/models/test_litellm_chatcompletions_stream.py`.
//!
//! Python patches `LitellmModel._fetch_response` to inject a fake chunk stream. Rust has no such
//! seam, so the same chunks are served as SSE by a local mock server and the Responses events the
//! model yields are inspected. The provider-stream `aclose` / cancellation / span-timing cases of
//! that file have no Rust counterpart (see the parity report).

use futures::StreamExt;
use openai_agents_sdk::extensions::models::LitellmModel;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve `chunks` once as an SSE stream and return the base URL.
async fn sse_server(chunks: Vec<Value>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
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
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse().ok())
                })
                .unwrap_or(0);
            if buf.len() >= idx + 4 + len {
                break;
            }
        }
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\
             connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
        sock.shutdown().await.ok();
    });
    format!("http://{addr}")
}

fn chunk(choice: Value) -> Value {
    json!({"id": "chunk-id", "object": "chat.completion.chunk", "created": 1, "model": "fake",
           "choices": [choice]})
}

fn with_usage(mut chunk: Value, usage: Value) -> Value {
    chunk["usage"] = usage;
    chunk
}

async fn stream_events(chunks: Vec<Value>) -> Vec<Value> {
    let base = sse_server(chunks).await;
    let model = LitellmModel::with_options("gpt-4", Some(&base), Some("sk-test"), None).unwrap();
    let mut events = model
        .stream_response(ModelRequest {
            input: vec![InputItem(json!({"role": "user", "content": ""}))],
            model_settings: ModelSettings::default(),
            tracing: ModelTracing::Disabled,
            ..Default::default()
        })
        .await
        .unwrap();
    let mut out = Vec::new();
    while let Some(event) = events.next().await {
        out.push(event.unwrap());
    }
    out
}

fn types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

fn of_type<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == kind).collect()
}

// ---------------------------------------------------------------------------
// Content, refusal and tool-call streams
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_response_yields_events_for_text_content() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"content": "He"}})),
        with_usage(
            chunk(json!({"index": 0, "delta": {"content": "llo"}})),
            json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                   "completion_tokens_details": {"reasoning_tokens": 2},
                   "prompt_tokens_details": {"cached_tokens": 6}}),
        ),
    ])
    .await;

    assert_eq!(
        types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[3]["delta"], "He");
    assert_eq!(events[4]["delta"], "llo");
    let completed = &events[7]["response"];
    assert_eq!(completed["output"][0]["type"], "message");
    assert_eq!(completed["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(completed["output"][0]["content"][0]["text"], "Hello");
    let usage = &completed["usage"];
    assert_eq!(usage["input_tokens"], 7);
    assert_eq!(usage["output_tokens"], 5);
    assert_eq!(usage["total_tokens"], 12);
    assert_eq!(usage["input_tokens_details"]["cached_tokens"], 6);
    assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 2);
}

#[tokio::test]
async fn stream_response_yields_events_for_refusal_content() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"refusal": "No"}})),
        with_usage(
            chunk(json!({"index": 0, "delta": {"refusal": "Thanks"}})),
            json!({"completion_tokens": 2, "prompt_tokens": 2, "total_tokens": 4}),
        ),
    ])
    .await;

    assert_eq!(
        types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.refusal.delta",
            "response.refusal.delta",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[3]["delta"], "No");
    assert_eq!(events[4]["delta"], "Thanks");
    let part = &events[7]["response"]["output"][0]["content"][0];
    assert_eq!(part["type"], "refusal");
    assert_eq!(part["refusal"], "NoThanks");
}

#[tokio::test]
async fn stream_response_yields_events_for_tool_call() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "tool-id", "type": "function",
            "function": {"name": "my_func", "arguments": "arg1"}}]}})),
        with_usage(
            chunk(json!({"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "tool-id", "type": "function",
                "function": {"arguments": "arg2"}}]}})),
            json!({"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}),
        ),
    ])
    .await;

    assert_eq!(
        types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let added = &events[1]["item"];
    assert_eq!(added["type"], "function_call");
    assert_eq!(added["name"], "my_func");
    assert_eq!(added["arguments"], "");
    assert_eq!(events[2]["delta"], "arg1");
    assert_eq!(events[3]["delta"], "arg2");
    let done = &events[4]["item"];
    assert_eq!(done["name"], "my_func");
    assert_eq!(done["arguments"], "arg1arg2");
}

/// Function-call arguments are forwarded as they arrive, not buffered until the call is whole.
#[tokio::test]
async fn stream_response_yields_real_time_function_call_arguments() {
    let delta = |args: &str| {
        chunk(json!({"index": 0, "delta": {"tool_calls": [{
            "index": 0, "type": "function", "function": {"arguments": args}}]}}))
    };
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "litellm-call-456", "type": "function",
            "function": {"name": "generate_code", "arguments": ""}}]}})),
        delta("{\"language\": \""),
        delta("python\", \"task\": \""),
        with_usage(
            delta("hello world\"}"),
            json!({"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}),
        ),
    ])
    .await;

    let deltas = of_type(&events, "response.function_call_arguments.delta");
    let added = of_type(&events, "response.output_item.added");
    assert_eq!(deltas.len(), 3);
    assert_eq!(added.len(), 1);
    let got: Vec<&str> = deltas
        .iter()
        .map(|d| d["delta"].as_str().unwrap())
        .collect();
    assert_eq!(
        got,
        [
            "{\"language\": \"",
            "python\", \"task\": \"",
            "hello world\"}"
        ]
    );
    assert_eq!(added[0]["item"]["type"], "function_call");
    assert_eq!(added[0]["item"]["name"], "generate_code");
    assert_eq!(added[0]["item"]["call_id"], "litellm-call-456");
}

// ---------------------------------------------------------------------------
// content_filter finish reason
// ---------------------------------------------------------------------------

/// A stream that ends with `finish_reason == "content_filter"` and emitted no content (as
/// Anthropic on Bedrock does) must carry an explicit refusal rather than an empty turn.
#[tokio::test]
async fn stream_response_synthesizes_refusal_on_content_filter() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"role": "assistant", "content": ""}})),
        chunk(json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})),
    ])
    .await;
    let kinds = types(&events);

    assert!(kinds.contains(&"response.output_item.added"));
    assert!(kinds.contains(&"response.content_part.added"));
    assert!(kinds.contains(&"response.refusal.delta"));
    assert_eq!(kinds.last(), Some(&"response.completed"));
    assert!(kinds.contains(&"response.output_item.done"));
    let refusal_deltas = of_type(&events, "response.refusal.delta");
    assert!(
        refusal_deltas[0]["delta"]
            .as_str()
            .is_some_and(|d| !d.is_empty())
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == "response.output_item.added")
            .count(),
        1
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == "response.content_part.added")
            .count(),
        kinds
            .iter()
            .filter(|k| **k == "response.content_part.done")
            .count()
    );
    assert!(!kinds.contains(&"response.output_text.delta"));
    let added_parts = of_type(&events, "response.content_part.added");
    assert_eq!(added_parts.len(), 1);
    assert_eq!(added_parts[0]["part"]["type"], "refusal");
    let completed = &events.last().unwrap()["response"];
    assert_eq!(completed["output"][0]["type"], "message");
    assert_eq!(
        completed["output"][0]["content"].as_array().unwrap().len(),
        1
    );
    assert_eq!(completed["output"][0]["content"][0]["type"], "refusal");
    assert!(
        completed["output"][0]["content"][0]["refusal"]
            .as_str()
            .is_some_and(|r| !r.is_empty())
    );
    assert_eq!(added_parts[0]["content_index"], 0);
    assert_eq!(refusal_deltas[0]["content_index"], 0);
    let done_parts = of_type(&events, "response.content_part.done");
    assert_eq!(done_parts.len(), 1);
    assert_eq!(done_parts[0]["content_index"], 0);
}

/// A `content_filter` finish reason that arrives after real text was streamed is left alone.
#[tokio::test]
async fn stream_response_content_filter_does_not_clobber_text() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {"content": "answer"}})),
        chunk(json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})),
    ])
    .await;

    assert!(!types(&events).contains(&"response.refusal.delta"));
    let completed = &events.last().unwrap()["response"];
    assert_eq!(completed["output"][0]["type"], "message");
    assert_eq!(completed["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(completed["output"][0]["content"][0]["text"], "answer");
}

/// A filtered turn preceded by reasoning still puts the refusal on the assistant message, which
/// comes after the reasoning item in the output.
#[tokio::test]
async fn stream_response_content_filter_refusal_after_reasoning() {
    let events = stream_events(vec![
        chunk(json!({"index": 0, "delta": {
            "role": "assistant", "content": null, "reasoning_content": "thinking..."}})),
        chunk(json!({"index": 0, "delta": {"content": ""}})),
        chunk(json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})),
    ])
    .await;

    let completed = &events.last().unwrap()["response"];
    assert_eq!(completed["output"][0]["type"], "reasoning");
    let message = &completed["output"][1];
    assert_eq!(message["type"], "message");
    assert_eq!(message["content"].as_array().unwrap().len(), 1);
    assert_eq!(message["content"][0]["type"], "refusal");
    let added = of_type(&events, "response.content_part.added");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["output_index"], 1);
    assert_eq!(added[0]["content_index"], 0);
}
