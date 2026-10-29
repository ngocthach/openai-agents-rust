//! Responses API over WebSocket against a local mock server.
#![cfg(feature = "websocket")]
#![allow(clippy::result_large_err)]

use std::sync::{Arc, Mutex};

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::{OpenAIClient, OpenAIResponsesWSModel};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

fn completed(id: &str, output: Value) -> Value {
    json!({"type": "response.completed", "response": {
        "id": id, "object": "response", "status": "completed", "output": output,
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}}}})
}

#[tokio::test]
async fn runs_agent_over_one_reused_websocket() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let frames: Arc<Mutex<Vec<Value>>> = Arc::default();
    let handshakes: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let (f2, h2) = (frames.clone(), handshakes.clone());
    tokio::spawn(async move {
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let h3 = h2.clone();
            let f3 = f2.clone();
            let cb = move |req: &Request, resp: Response| {
                let auth = req
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                h3.lock().unwrap().push((req.uri().to_string(), auth));
                Ok(resp)
            };
            let mut ws = tokio_tungstenite::accept_hdr_async(tcp, cb).await.unwrap();
            tokio::spawn(async move {
                let mut turn = 0;
                while let Some(Ok(msg)) = ws.next().await {
                    let Message::Text(t) = msg else { continue };
                    let frame: Value = serde_json::from_str(&t).unwrap();
                    f3.lock().unwrap().push(frame);
                    turn += 1;
                    let events = if turn == 1 {
                        vec![
                            json!({"type": "response.created", "response": {"id": "resp_1"}}),
                            completed(
                                "resp_1",
                                json!([{"type": "function_call", "id": "fc_1",
                                "call_id": "call_1", "name": "ping", "arguments": "{}", "status": "completed"}]),
                            ),
                        ]
                    } else {
                        vec![
                            json!({"type": "response.output_text.delta", "delta": "pong!", "item_id": "m", "output_index": 0, "content_index": 0}),
                            completed(
                                "resp_2",
                                json!([{"type": "message", "id": "msg_1", "role": "assistant",
                                "status": "completed", "content": [{"type": "output_text", "text": "pong!", "annotations": []}]}]),
                            ),
                        ]
                    };
                    for e in events {
                        ws.send(Message::text(e.to_string())).await.unwrap();
                    }
                }
            });
        }
    });

    let client = OpenAIClient::new("sk-test").with_base_url(format!("http://{addr}/v1"));
    let model = Arc::new(OpenAIResponsesWSModel::new("gpt-test", client));
    let agent = Agent::<()>::new("Pinger")
        .with_tool(FunctionTool::<()>::new_no_args(
            "ping",
            "Ping",
            |_c| async { Ok("pong") },
        ))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "ping please").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("pong!"));
    assert_eq!(result.usage().total_tokens, 10);

    let hs = handshakes.lock().unwrap().clone();
    assert_eq!(hs.len(), 1, "the connection is reused across turns");
    assert_eq!(hs[0].0, "/v1/responses");
    assert_eq!(hs[0].1, "Bearer sk-test");
    let frames = frames.lock().unwrap().clone();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["type"], "response.create");
    assert_eq!(frames[0]["stream"], true);
    assert_eq!(frames[0]["model"], "gpt-test");
    assert_eq!(frames[0]["tools"][0]["name"], "ping");
    let second_input = frames[1]["input"].as_array().unwrap();
    assert!(
        second_input
            .iter()
            .any(|i| i["type"] == "function_call_output" && i["output"] == "pong")
    );

    // Streaming goes over the same socket.
    let mut streamed = Runner::run_streamed(&agent, "again");
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(ev) = events.next().await {
            ev.unwrap();
        }
    }
    assert_eq!(handshakes.lock().unwrap().len(), 1);
    use openai_agents_sdk::models::Model;
    model.close().await;
}
