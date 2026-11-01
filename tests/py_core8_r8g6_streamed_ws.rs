//! Ports of the WebSocket-model cases of `test_agent_runner_streamed.py` that earlier rounds filed
//! as "OpenAIResponsesWSModel internals": Python swaps `_open_websocket_connection` for a fake;
//! here a real local WebSocket server plays the Responses endpoint.
//!
//! * `test_streamed_run_rejects_failed_terminal_response_payload_events_from_ws_model`
//! * `test_ws_terminal_failure_frees_the_request_lock_for_a_waiting_run`
#![cfg(feature = "websocket")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::models::openai_responses_ws::OpenAIResponsesWSModel;
use openai_agents_sdk::result::RunResultStreaming;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio_tungstenite::tungstenite::Message;

/// What the server does on its one connection, in order.
#[derive(Clone)]
enum Step {
    /// Wait for one client text frame and record it.
    Recv,
    /// Send one JSON text frame.
    Send(Value),
    /// Wait until the test opens the gate.
    Gate(Arc<Notify>),
}

struct Server {
    base_url: String,
    accepted: Arc<AtomicUsize>,
    frames: Arc<Mutex<Vec<Value>>>,
}

impl Server {
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("sk-test").with_base_url(self.base_url.clone())
    }
    fn frame_count(&self) -> usize {
        self.frames.lock().unwrap().len()
    }
}

async fn spawn_server(steps: Vec<Step>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = Server {
        base_url: format!("http://127.0.0.1:{port}/v1"),
        accepted: Arc::default(),
        frames: Arc::default(),
    };
    let (accepted, frames) = (server.accepted.clone(), server.frames.clone());
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            accepted.fetch_add(1, Ordering::SeqCst);
            let (steps, frames) = (steps.clone(), frames.clone());
            tokio::spawn(async move {
                let Ok(mut ws) = tokio_tungstenite::accept_async(tcp).await else {
                    return;
                };
                for step in steps {
                    match step {
                        Step::Recv => loop {
                            match ws.next().await {
                                Some(Ok(Message::Text(t))) => {
                                    frames
                                        .lock()
                                        .unwrap()
                                        .push(serde_json::from_str(&t).unwrap());
                                    break;
                                }
                                Some(Ok(_)) => continue,
                                _ => return,
                            }
                        },
                        Step::Send(v) => {
                            if ws.send(Message::text(v.to_string())).await.is_err() {
                                return;
                            }
                        }
                        Step::Gate(gate) => gate.notified().await,
                    }
                }
                while let Some(Ok(_)) = ws.next().await {}
            });
        }
    });
    server
}

async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

/// A terminal Responses frame whose output is the text `partial final`.
fn terminal_frame(event_type: &str, response_id: &str) -> Value {
    json!({
        "type": event_type,
        "sequence_number": 1,
        "response": {
            "id": response_id, "object": "response", "created_at": 0, "model": "fake",
            "status": if event_type == "response.completed" { "completed" } else { "incomplete" },
            "tool_choice": "none", "tools": [], "parallel_tool_calls": false,
            "top_p": null, "usage": null,
            "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "partial final", "annotations": []}]}],
        }
    })
}

#[derive(Default)]
struct Consumed {
    events: Vec<StreamEvent<()>>,
    error: Option<AgentsError>,
}

async fn consume(stream: &mut RunResultStreaming<()>) -> Consumed {
    let mut out = Consumed::default();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        match event {
            Ok(e) => out.events.push(e),
            Err(e) => out.error = Some(e),
        }
    }
    out
}

fn ws_agent(server: &Server) -> Arc<Agent<()>> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(OpenAIResponsesWSModel::new("gpt-4", server.client()));
    Arc::new(Agent::<()>::new("test").with_model_instance(model))
}

/// Python: `test_streamed_run_rejects_failed_terminal_response_payload_events_from_ws_model`.
#[tokio::test]
async fn a_failed_terminal_frame_from_the_websocket_model_fails_the_streamed_run() {
    for event_type in ["response.incomplete", "response.failed"] {
        let server = spawn_server(vec![
            Step::Recv,
            Step::Send(terminal_frame(event_type, "resp-ws")),
        ])
        .await;
        let agent = ws_agent(&server);
        let mut stream = Runner::run_streamed_with_options(agent, "test", RunOptions::new(()));
        let out = consume(&mut stream).await;

        let err = out
            .error
            .unwrap_or_else(|| panic!("{event_type} must fail the run"));
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{event_type}: {err:?}"
        );
        assert!(err.to_string().contains(event_type), "{event_type}: {err}");
        assert_eq!(out.events.len(), 2, "{event_type}");
        assert!(matches!(out.events[0], StreamEvent::AgentUpdated { .. }));
        match &out.events[1] {
            StreamEvent::RawResponse(data) => assert_eq!(data["type"], event_type),
            other => panic!("{event_type}: {other:?}"),
        }
        assert!(stream.final_output().is_none(), "{event_type}");
        assert!(
            stream.result().is_none_or(|r| r.raw_responses.is_empty()),
            "{event_type}"
        );
    }
}

/// Python: `test_ws_terminal_failure_frees_the_request_lock_for_a_waiting_run`.
///
/// Run A holds the model's request lock while run B queues behind it. A then fails terminally
/// and B has to complete on the surviving connection, which only happens if A's cleanup
/// released the lock without dropping the shared socket.
#[tokio::test]
async fn a_terminal_websocket_failure_hands_the_connection_to_the_waiting_run() {
    let release_run_a = Arc::new(Notify::new());
    let server = spawn_server(vec![
        Step::Recv,
        Step::Gate(release_run_a.clone()),
        Step::Send(terminal_frame("response.incomplete", "resp-a")),
        Step::Recv,
        Step::Send(terminal_frame("response.completed", "resp-b")),
    ])
    .await;
    let agent = ws_agent(&server);

    let agent_a = agent.clone();
    let run_a = tokio::spawn(async move {
        let mut stream = Runner::run_streamed_with_options(agent_a, "a", RunOptions::new(()));
        consume(&mut stream).await.error
    });
    wait_until("run A's request frame", || server.frame_count() == 1).await;

    let agent_b = agent.clone();
    let run_b = tokio::spawn(async move {
        let mut stream = Runner::run_streamed_with_options(agent_b, "b", RunOptions::new(()));
        let out = consume(&mut stream).await;
        (out.error, stream.into_result())
    });
    // B must be queued behind A's request, not past it.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !run_b.is_finished(),
        "run B was expected to wait for the connection"
    );
    assert_eq!(server.frame_count(), 1, "run B sent its request early");
    release_run_a.notify_one();

    let a_error = run_a.await.unwrap().expect("run A fails terminally");
    assert!(
        a_error.to_string().contains("response.incomplete"),
        "{a_error}"
    );

    let (b_error, b_result) = tokio::time::timeout(Duration::from_secs(5), run_b)
        .await
        .expect("run A's failure left the connection locked, so run B never woke")
        .unwrap();
    assert!(b_error.is_none(), "{b_error:?}");
    let b_result = b_result.expect("run B finished");
    assert_eq!(
        b_result.final_output_text().as_deref(),
        Some("partial final")
    );
    let ids: Vec<Option<String>> = b_result
        .raw_responses
        .iter()
        .map(|r| r.response_id.clone())
        .collect();
    assert_eq!(ids, [Some("resp-b".to_string())]);
    assert_eq!(
        server.accepted.load(Ordering::SeqCst),
        1,
        "B reopened the socket"
    );
    assert_eq!(server.frame_count(), 2);
}
