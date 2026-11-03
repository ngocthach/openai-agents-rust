//! Port of the Runner-level tests in `tests/models/test_openai_chatcompletions_stream.py` and
//! of `tests/models/test_chatcompletions_stream_eof.py`: complete runs through the Chat
//! Completions model, with an SSE mock server standing in for the provider.
//!
//! The Python EOF tests build an `AsyncOpenAI` against `https://api.openai.com`; a local mock
//! cannot be an official endpoint, so the official-endpoint finish-reason requirement is
//! exercised on the stream handler itself (`require_finish_reason`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
#[cfg(feature = "sqlite")]
use openai_agents_sdk::memory::SqliteSession;
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatStreamOptions, buffer_tool_call_chunks, handle_chat_stream_chunks,
};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, OpenAIChatCompletionsModel, OpenAIClient, RunOptions, Runner, ToolUseBehavior,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---- mock server -------------------------------------------------------------------------

#[derive(Clone)]
enum Reply {
    Sse(Vec<Value>),
    #[cfg_attr(not(feature = "sqlite"), allow(dead_code))]
    Json(Value),
}

type Captured = Arc<Mutex<Vec<Value>>>;

async fn serve(replies: Vec<Reply>) -> (String, Captured) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
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
                cap.lock().unwrap().push(body);
                let reply = {
                    let mut q = queue.lock().unwrap();
                    if q.len() > 1 {
                        q.pop_front().unwrap()
                    } else {
                        q.front().unwrap().clone()
                    }
                };
                let (content_type, payload) = match reply {
                    Reply::Json(v) => ("application/json", v.to_string()),
                    Reply::Sse(chunks) => {
                        let mut p = String::new();
                        for c in &chunks {
                            p.push_str(&format!("data: {c}\n\n"));
                        }
                        p.push_str("data: [DONE]\n\n");
                        ("text/event-stream", p)
                    }
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(payload.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), captured)
}

fn chunk(delta: Value, finish_reason: Option<&str>, usage: Option<Value>) -> Value {
    let mut c = json!({"id": "chunk-id", "created": 1, "model": "fake",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

fn model(name: &str, base: &str, buffered: bool) -> Arc<OpenAIChatCompletionsModel> {
    Arc::new(
        OpenAIChatCompletionsModel::new(name, OpenAIClient::new("sk-test").with_base_url(base))
            .with_buffer_streamed_tool_calls(buffered),
    )
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn lookup_tool(calls: Arc<Mutex<Vec<&'static str>>>, name: &str) -> FunctionTool<()> {
    FunctionTool::new(name, "Look something up", move |_c, _a: NoArgs| {
        let calls = calls.clone();
        async move {
            calls.lock().unwrap().push("lookup");
            Ok("result".to_string())
        }
    })
}

async fn drain<C: Send + Sync + 'static>(
    result: &mut openai_agents_sdk::RunResultStreaming<C>,
) -> Result<Vec<Value>, String> {
    let mut raw = Vec::new();
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(ev) = events.next().await {
        match ev {
            Ok(StreamEvent::RawResponse(data)) => raw.push(data),
            Ok(_) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(raw)
}

// =========================================================================================
// test_openai_chatcompletions_stream.py
// =========================================================================================

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn buffered_tool_call_before_text_replays_as_single_assistant_session_message() {
    openai_agents_sdk::set_tracing_disabled(true);
    let tool_call = json!({"index": 0, "id": "call_lookup_status", "type": "function",
        "function": {"name": "lookup_status", "arguments": "{}"}});
    let (base, captured) = serve(vec![
        Reply::Sse(vec![
            chunk(json!({"tool_calls": [tool_call]}), None, None),
            chunk(
                json!({"content": "I'll look that up first."}),
                None,
                Some(json!({"completion_tokens": 5, "prompt_tokens": 5, "total_tokens": 10})),
            ),
        ]),
        Reply::Sse(vec![chunk(
            json!({"content": "first run done"}),
            None,
            Some(json!({"completion_tokens": 3, "prompt_tokens": 7, "total_tokens": 10})),
        )]),
        Reply::Json(json!({"id": "resp-id", "created": 0, "model": "fake",
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "second run done"}}]})),
    ])
    .await;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::<()>::new("test")
        .with_model_instance(model("gpt-4", &base, true))
        .with_tool(lookup_tool(calls, "lookup_status"));
    let session = Arc::new(SqliteSession::in_memory("buffered").unwrap());

    let mut first = Runner::run_streamed_with_options(
        &agent,
        "first question",
        RunOptions::new(()).session(session.clone()),
    );
    drain(&mut first).await.unwrap();
    assert_eq!(first.final_output_text().as_deref(), Some("first run done"));
    Runner::run_with_options(
        &agent,
        "second question",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 3);
    let messages = requests[2]["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant", "user"]);
    let assistant = &messages[1];
    assert_eq!(assistant["content"], "I'll look that up first.");
    assert_eq!(assistant["tool_calls"].as_array().unwrap().len(), 1);
    let call = &assistant["tool_calls"][0];
    assert_eq!(call["id"], "call_lookup_status");
    assert_eq!(
        call["function"],
        json!({"name": "lookup_status", "arguments": "{}"})
    );
    assert_eq!(messages[2]["tool_call_id"], "call_lookup_status");
    assert_eq!(messages[2]["content"], "result");
    assert_eq!(messages[3]["content"], "first run done");
    assert_eq!(messages[4]["content"], "second question");
}

#[tokio::test]
async fn streamed_run_counts_request_when_provider_omits_usage() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, _) = serve(vec![Reply::Sse(vec![
        chunk(json!({"content": "Hello"}), None, None),
        chunk(json!({}), Some("stop"), None),
    ])])
    .await;
    let agent = Agent::<()>::new("test").with_model_instance(model("gpt-4", &base, false));
    let mut result = Runner::run_streamed_with_options(&agent, "hi", RunOptions::new(()));
    let raw = drain(&mut result).await.unwrap();
    let usage = result.result().unwrap().usage();
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 0);
    // No usage payload is synthesized, so nothing reports token counts that never arrived.
    let completed = raw
        .iter()
        .find(|e| e["type"] == "response.completed")
        .unwrap();
    assert!(completed["response"]["usage"].is_null());
}

#[tokio::test]
async fn streamed_run_does_not_double_count_when_usage_is_present() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, _) = serve(vec![Reply::Sse(vec![
        chunk(
            json!({"content": "Hello"}),
            None,
            Some(json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12})),
        ),
        chunk(json!({}), Some("stop"), None),
    ])])
    .await;
    let agent = Agent::<()>::new("test").with_model_instance(model("gpt-4", &base, false));
    let mut result = Runner::run_streamed_with_options(&agent, "hi", RunOptions::new(()));
    drain(&mut result).await.unwrap();
    let usage = result.result().unwrap().usage();
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 12);
}

// =========================================================================================
// test_chatcompletions_stream_eof.py
// =========================================================================================

/// The synthetic SSE body of the Python EOF tests, as chunks.
fn eof_chunks(delta: Option<Value>, finish_reason: Option<&str>) -> Vec<Value> {
    let mut chunks = Vec::new();
    if let Some(d) = delta {
        chunks.push(chunk(d, None, None));
    }
    if let Some(reason) = finish_reason {
        chunks.push(chunk(json!({}), Some(reason), None));
        let mut trailer = chunk(json!({}), None, None);
        trailer["choices"] = json!([]);
        trailer["usage"] = json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5});
        chunks.push(trailer);
    }
    chunks
}

fn lookup_delta() -> Value {
    json!({"tool_calls": [{"index": 0, "id": "call-synthetic", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"}}]})
}

#[test]
fn official_stream_rejects_eof_before_terminal_choice() {
    for buffered in [false, true] {
        for delta in [
            Some(json!({"content": "The answer is "})),
            Some(lookup_delta()),
            None,
        ] {
            let mut chunks = eof_chunks(delta, None);
            if buffered {
                chunks = buffer_tool_call_chunks(chunks).unwrap();
            }
            let err = handle_chat_stream_chunks(
                &chunks,
                &ChatStreamOptions {
                    model: Some("gpt-4o-mini".into()),
                    raise_on_length_truncation: true,
                    require_finish_reason: true,
                    ..Default::default()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("before receiving a finish_reason"), "{err}");
        }
    }
}

#[tokio::test]
async fn official_stream_accepts_terminal_choice_with_usage_trailer() {
    openai_agents_sdk::set_tracing_disabled(true);
    for buffered in [false, true] {
        for finish_reason in ["stop", "length"] {
            let chunks = eof_chunks(Some(json!({"content": "Answer"})), Some(finish_reason));
            // The official-endpoint requirement is satisfied by the terminal choice.
            let events = handle_chat_stream_chunks(
                &if buffered {
                    buffer_tool_call_chunks(chunks.clone()).unwrap()
                } else {
                    chunks.clone()
                },
                &ChatStreamOptions {
                    model: Some("gpt-4o-mini".into()),
                    raise_on_length_truncation: true,
                    require_finish_reason: true,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e["type"] == "response.completed")
                    .count(),
                1
            );

            let (base, _) = serve(vec![Reply::Sse(chunks)]).await;
            let agent = Agent::<()>::new("Synthetic")
                .with_model_instance(model("gpt-4o-mini", &base, buffered))
                .with_model_settings(openai_agents_sdk::ModelSettings {
                    preserve_raw_usage: Some(true),
                    ..Default::default()
                });
            let mut result =
                Runner::run_streamed_with_options(&agent, "Answer", RunOptions::new(()));
            let raw = drain(&mut result).await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("Answer"));
            assert_eq!(
                raw.iter()
                    .filter(|e| e["type"] == "response.completed")
                    .count(),
                1
            );
            assert_eq!(result.result().unwrap().usage().total_tokens, 5);
        }
    }
}

#[tokio::test]
async fn third_party_stream_preserves_eof_completion() {
    openai_agents_sdk::set_tracing_disabled(true);
    for buffered in [false, true] {
        let (base, _) = serve(vec![Reply::Sse(eof_chunks(
            Some(json!({"content": "Answer"})),
            None,
        ))])
        .await;
        let agent = Agent::<()>::new("Synthetic").with_model_instance(model(
            "provider-model",
            &base,
            buffered,
        ));
        let mut result = Runner::run_streamed_with_options(&agent, "Answer", RunOptions::new(()));
        drain(&mut result).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("Answer"));
    }
}

#[tokio::test]
async fn official_stream_executes_tool_after_terminal_choice() {
    openai_agents_sdk::set_tracing_disabled(true);
    for buffered in [false, true] {
        let (base, _) = serve(vec![Reply::Sse(eof_chunks(
            Some(lookup_delta()),
            Some("tool_calls"),
        ))])
        .await;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = Agent::<()>::new("Synthetic")
            .with_tool(lookup_tool(calls.clone(), "lookup"))
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_model_instance(model("gpt-4o-mini", &base, buffered));
        let mut result = Runner::run_streamed_with_options(&agent, "Answer", RunOptions::new(()));
        drain(&mut result).await.unwrap();
        assert_eq!(*calls.lock().unwrap(), ["lookup"]);
        assert_eq!(result.final_output_text().as_deref(), Some("result"));
        assert_eq!(result.result().unwrap().usage().total_tokens, 5);
    }
}
