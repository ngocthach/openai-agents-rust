//! Ports of `tests/realtime/test_openai_realtime.py`:
//! `test_connect_session_config_failure_releases_websocket`,
//! `test_connect_preserves_setup_error_when_websocket_close_fails` and
//! `test_session_config_preserves_sip_audio_formats`.
#![cfg(feature = "realtime")]

use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::realtime::{
    OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, RealtimeFunctionToolSpec, RealtimeModel,
    RealtimeModelConfig, RealtimeSessionModelSettings, RealtimeTurnDetectionConfig,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::protocol::Message;

const TIMEOUT: Duration = Duration::from_secs(5);

/// Accepts one client, forwards its JSON frames and reports when the client goes away.
struct Server {
    url: String,
    frames: UnboundedReceiver<Value>,
    closed: oneshot::Receiver<()>,
}

async fn serve_one() -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "ws://127.0.0.1:{}/v1/realtime",
        listener.local_addr().unwrap().port()
    );
    let (frame_tx, frames) = unbounded_channel();
    let (closed_tx, closed) = oneshot::channel();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message
                && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
            {
                let _ = frame_tx.send(value);
            }
        }
        let _ = closed_tx.send(());
    });
    Server {
        url,
        frames,
        closed,
    }
}

fn config_for(url: &str) -> RealtimeModelConfig {
    RealtimeModelConfig::new()
        .with_api_key("test-key")
        .with_url(url)
}

/// A tool the Realtime API cannot serve, so the initial session update fails after the socket
/// is open.
fn unsupported_tool() -> RealtimeFunctionToolSpec {
    RealtimeFunctionToolSpec::new("lookup", "lookup", json!({"type": "object"}))
        .with_defer_loading(true)
}

#[tokio::test]
async fn a_failed_session_update_releases_the_socket_and_restores_the_model_name() {
    let model = OpenAIRealtimeWebSocketModel::new();
    let default_model = model.model_name();

    let first = serve_one().await;
    let mut settings = RealtimeSessionModelSettings::new();
    settings.model_name = Some("failed-model".to_string());
    settings.tools = Some(vec![unsupported_tool()]);
    let error = model
        .connect(config_for(&first.url).with_initial_model_settings(settings))
        .await
        .expect_err("the session update is rejected");
    assert!(
        error
            .to_string()
            .contains("only supported with OpenAI Responses"),
        "{error}"
    );

    // The connection that was opened for the attempt is closed, and the model name the attempt
    // asked for is not kept.
    tokio::time::timeout(TIMEOUT, first.closed)
        .await
        .expect("the socket is released")
        .unwrap();
    assert_eq!(model.model_name(), default_model);

    // The same model connects again.
    let mut second = serve_one().await;
    let _events = model
        .connect(config_for(&second.url))
        .await
        .expect("a second connect succeeds");
    let update = tokio::time::timeout(TIMEOUT, second.frames.recv())
        .await
        .expect("a session update arrives")
        .unwrap();
    assert_eq!(update["type"], "session.update");
    assert_eq!(update["session"]["model"], default_model);
    assert_eq!(model.model_name(), default_model);
    model.close().await.unwrap();
}

#[tokio::test]
async fn an_attached_call_keeps_the_audio_formats_it_negotiated() {
    let mut server = serve_one().await;
    let model = OpenAIRealtimeSIPModel::new();
    let mut settings = RealtimeSessionModelSettings::new();
    settings.turn_detection = Some(Some(RealtimeTurnDetectionConfig {
        kind: Some("semantic_vad".to_string()),
        interrupt_response: Some(true),
        ..RealtimeTurnDetectionConfig::default()
    }));
    let _events = model
        .connect(
            config_for(&server.url)
                .with_call_id("call-123")
                .with_initial_model_settings(settings),
        )
        .await
        .expect("connect");
    let update = tokio::time::timeout(TIMEOUT, server.frames.recv())
        .await
        .expect("a session update arrives")
        .unwrap();
    let audio = &update["session"]["audio"];
    assert!(audio["input"].is_object());
    assert!(
        audio["input"].get("format").is_none_or(Value::is_null),
        "{audio}"
    );
    assert!(audio["output"].is_object());
    assert!(
        audio["output"].get("format").is_none_or(Value::is_null),
        "{audio}"
    );
    model.close().await.unwrap();
}
