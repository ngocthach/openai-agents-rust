//! Port of `test_call_id_session_update_omits_null_audio_formats` from
//! `tests/realtime/test_realtime_model_settings.py`. A session attached to an existing call
//! already negotiated its audio formats, so the `session.update` it sends leaves an unset format
//! out instead of defaulting it (or sending null).
#![cfg(feature = "realtime")]

use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::realtime::{OpenAIRealtimeSIPModel, RealtimeModel, RealtimeModelConfig};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test]
async fn a_call_session_update_omits_unset_audio_formats() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let _ = tx.send(value);
            }
        }
        let _ = ws.close(None).await;
    });

    let model = OpenAIRealtimeSIPModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_call_id("call_123")
                .with_url(format!(
                    "ws://127.0.0.1:{port}/v1/realtime?call_id=call_123"
                )),
        )
        .await
        .expect("attaches to the call");

    let update = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("session.update received")
        .unwrap();
    assert_eq!(update["type"], "session.update");
    let audio = &update["session"]["audio"];
    assert!(audio["input"].get("format").is_none(), "{audio}");
    assert!(audio["output"].get("format").is_none(), "{audio}");

    model.close().await.unwrap();
    server.abort();
}
