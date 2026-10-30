//! Exercise the OpenAI HTTP models against a local mock server: request shape, response
//! parsing and SSE streaming for both the Responses and Chat Completions APIs.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::models::interface::Model;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, OpenAIChatCompletionsModel, OpenAIClient, OpenAIResponsesModel, RunOptions, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A canned HTTP response.
#[derive(Clone)]
enum Reply {
    Json(Value),
    Sse(Vec<Value>),
}

/// Start a server answering requests in order; returns the base URL and captured bodies.
async fn mock_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<(String, Value)>>>) {
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
            let (path, body) = loop {
                let n = sock.read(&mut tmp).await.unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let headers = &text[..idx];
                    let len: usize = headers
                        .lines()
                        .find_map(|l| {
                            let l = l.to_lowercase();
                            l.strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        let path = headers.split_whitespace().nth(1).unwrap().to_string();
                        let body: Value = serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                        break (path, body);
                    }
                }
            };
            cap.lock().unwrap().push((path, body));
            let reply = replies.next().expect("unexpected request");
            let resp = match reply {
                Reply::Json(v) => {
                    let b = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nx-request-id: req_123\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
                Reply::Sse(events) => {
                    let mut b = String::new();
                    for e in events {
                        b.push_str(&format!(
                            "event: {}\ndata: {}\n\n",
                            e["type"].as_str().unwrap_or("message"),
                            e
                        ));
                    }
                    b.push_str("data: [DONE]\n\n");
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    )
                }
            };
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

#[derive(Deserialize, JsonSchema)]
struct Args {
    city: String,
    days: u32,
}

fn tool() -> FunctionTool<()> {
    FunctionTool::new("forecast", "Forecast", |_c, a: Args| async move {
        Ok(format!("{} days of sun in {}", a.days, a.city))
    })
}

#[tokio::test]
async fn responses_api_round_trip() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, captured) = mock_server(vec![
        Reply::Json(json!({
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "forecast", "arguments": "{\"city\":\"Hue\",\"days\":3}", "status": "completed"}],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
        })),
        Reply::Json(json!({
            "id": "resp_2", "object": "response", "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "Sunny for 3 days", "annotations": []}]}],
            "usage": {"input_tokens": 40, "output_tokens": 6, "total_tokens": 46}
        })),
    ])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = Arc::new(OpenAIResponsesModel::new("gpt-5.6-luna", client));
    let agent = Agent::<()>::new("Weather")
        .with_instructions("Use tools")
        .with_tool(tool())
        .with_model_instance(model);
    let r = Runner::run(&agent, "Weather in Hue?").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("Sunny for 3 days"));
    assert_eq!(r.usage().total_tokens, 71);
    assert_eq!(r.raw_responses[0].request_id.as_deref(), Some("req_123"));
    assert_eq!(r.last_response_id().as_deref(), Some("resp_2"));

    let reqs = captured.lock().unwrap().clone();
    assert_eq!(reqs[0].0, "/v1/responses");
    let body = &reqs[0].1;
    assert_eq!(body["model"], "gpt-5.6-luna");
    assert_eq!(body["instructions"], "Use tools");
    assert_eq!(
        body["reasoning"]["effort"], "none",
        "GPT-5 defaults applied"
    );
    let params = &body["tools"][0]["parameters"];
    assert_eq!(params["additionalProperties"], false);
    assert!(
        params["properties"]["days"].get("format").is_none(),
        "schemars formats stripped"
    );
    let second = &reqs[1].1;
    let last = second["input"].as_array().unwrap().last().unwrap();
    assert_eq!(last["type"], "function_call_output");
    assert_eq!(last["output"], "3 days of sun in Hue");
}

#[tokio::test]
async fn responses_api_streaming() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, captured) = mock_server(vec![Reply::Sse(vec![
        json!({"type": "response.created", "response": {"id": "resp_9", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_text.delta", "item_id": "m", "output_index": 0, "content_index": 0, "delta": "Hel"}),
        json!({"type": "response.output_text.delta", "item_id": "m", "output_index": 0, "content_index": 0, "delta": "lo"}),
        json!({"type": "response.completed", "response": {"id": "resp_9", "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "content": [{"type": "output_text", "text": "Hello", "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}}),
    ])])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let agent = Agent::<()>::new("A")
        .with_model_instance(Arc::new(OpenAIResponsesModel::new("gpt-4.1", client)));
    let mut result = Runner::run_streamed(&agent, "hi");
    let mut text = String::new();
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(ev) = events.next().await {
            if let Some(d) = ev.unwrap().text_delta() {
                text.push_str(d);
            }
        }
    }
    assert_eq!(text, "Hello");
    assert_eq!(result.final_output_text().as_deref(), Some("Hello"));
    assert_eq!(captured.lock().unwrap()[0].1["stream"], true);
}

#[tokio::test]
async fn chat_completions_round_trip_and_streaming() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (base, captured) = mock_server(vec![
        Reply::Sse(vec![
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_7", "type": "function", "function": {"name": "forecast", "arguments": "{\"city\":"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Hue\",\"days\":2}"}}]}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}),
        ]),
        Reply::Sse(vec![
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Two "}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "sunny days"}}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}),
        ]),
    ])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = Arc::new(OpenAIChatCompletionsModel::new("some-model", client));
    let agent = Agent::<()>::new("A")
        .with_instructions("sys")
        .with_tool(tool())
        .with_model_instance(model);
    let mut result = Runner::run_streamed_with_options(&agent, "Hue?", RunOptions::new(()));
    let mut text = String::new();
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(ev) = events.next().await {
            if let Some(d) = ev.unwrap().text_delta() {
                text.push_str(d);
            }
        }
    }
    let _ = text;
    let reqs = captured.lock().unwrap().clone();
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[0].0, "/v1/chat/completions");
    assert_eq!(reqs[0].1["messages"][0]["role"], "system");
    assert_eq!(reqs[0].1["tools"][0]["function"]["name"], "forecast");
    let msgs = reqs[1].1["messages"].as_array().unwrap();
    assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_7");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["content"], "2 days of sun in Hue");
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("Two sunny days")
    );
    assert_eq!(result.result().unwrap().usage().total_tokens, 37);
}

#[tokio::test]
async fn api_errors_are_reported() {
    openai_agents_sdk::set_tracing_disabled(true);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut tmp = [0u8; 8192];
        let _ = sock.read(&mut tmp).await;
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        let resp = format!(
            "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
    });
    let client = OpenAIClient::new("bad").with_base_url(format!("http://{addr}/v1"));
    let agent =
        Agent::<()>::new("A").with_model_instance(Arc::new(OpenAIResponsesModel::new("m", client)));
    let err = Runner::run(&agent, "x").await.unwrap_err();
    match err {
        openai_agents_sdk::AgentsError::ApiError {
            status, message, ..
        } => {
            assert_eq!(status, 401);
            assert_eq!(message, "Invalid API key");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn responses_stream_terminal_events_become_errors() {
    openai_agents_sdk::set_tracing_disabled(true);
    for (events, expected) in [
        (
            vec![json!({"type": "response.error", "code": "server_error", "message": "boom"})],
            "Responses stream ended with terminal event `response.error`. \
             code=server_error; message=boom.",
        ),
        (
            vec![json!({"type": "response.incomplete", "response": {
                "id": "resp_1", "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"}}})],
            "Responses stream ended with terminal event `response.incomplete`. \
             status=incomplete; incomplete_details={\"reason\":\"max_output_tokens\"}.",
        ),
    ] {
        let (base, _) = mock_server(vec![Reply::Sse(events)]).await;
        let client = OpenAIClient::new("sk-test").with_base_url(base);
        let model = OpenAIResponsesModel::new("gpt-4.1", client);
        let mut stream = model
            .stream_response(Default::default())
            .await
            .expect("stream opened");
        let err = loop {
            match stream.next().await {
                Some(Ok(_)) => continue,
                Some(Err(e)) => break e,
                None => panic!("stream ended without an error"),
            }
        };
        assert_eq!(err.to_string(), expected);
    }
}
