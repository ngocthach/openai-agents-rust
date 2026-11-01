//! Port of the `explicit-unlimited` rows of `test_incoming_message_size_limit`
//! (`tests/realtime/test_websocket_message_size.py`): a transport configured without an incoming
//! size limit accepts a message above the 8 MiB default, on both the websocket and the SIP
//! models. Python spells it `max_size=None`; the Rust option is `usize::MAX`.
#![cfg(feature = "realtime")]

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::realtime::{
    OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, RealtimeModel, RealtimeModelConfig,
    RealtimeModelEvent, RealtimeTransportConfig,
};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;

const LIMIT: usize = 8 * 1024 * 1024;

fn text_delta_of_size(size: usize) -> (String, String) {
    let mut payload = json!({
        "type": "response.output_text.delta", "event_id": "event_test",
        "response_id": "resp_test", "item_id": "item_test",
        "output_index": 0, "content_index": 0, "delta": "",
    });
    let overhead = payload.to_string().len();
    let delta = "x".repeat(size - overhead);
    payload["delta"] = json!(delta);
    let message = payload.to_string();
    assert_eq!(message.len(), size);
    (message, delta)
}

async fn accepted_without_a_limit(sip: bool) -> bool {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (message, delta) = text_delta_of_size(LIMIT + 1);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        let _ = ws.next().await;
        let _ = ws.send(Message::text(message)).await;
        while ws.next().await.is_some() {}
    });
    let config = RealtimeModelConfig::new()
        .with_api_key("test-key")
        .with_url(format!("ws://127.0.0.1:{port}/v1/realtime"))
        .with_transport(RealtimeTransportConfig::new().with_max_size(usize::MAX));
    let (model, mut events): (Box<dyn RealtimeModel>, _) = if sip {
        let model = OpenAIRealtimeSIPModel::new();
        let events = model
            .connect(config.with_call_id("call_test"))
            .await
            .unwrap();
        (Box::new(model), events)
    } else {
        let model = OpenAIRealtimeWebSocketModel::new();
        let events = model.connect(config).await.unwrap();
        (Box::new(model), events)
    };
    let accepted = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match events.recv().await.expect("the stream stays open") {
                RealtimeModelEvent::OutputTextDelta { delta: got, .. } => return got == delta,
                RealtimeModelEvent::Exception { .. } => return false,
                _ => {}
            }
        }
    })
    .await
    .expect("a verdict arrives");
    model.close().await.unwrap();
    server.abort();
    accepted
}

#[tokio::test]
async fn an_unlimited_websocket_accepts_a_message_above_the_default_limit() {
    assert!(accepted_without_a_limit(false).await);
}

#[tokio::test]
async fn an_unlimited_sip_model_accepts_a_message_above_the_default_limit() {
    assert!(accepted_without_a_limit(true).await);
}
