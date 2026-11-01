//! Ports of the listener-failure case of `tests/voice/test_openai_stt.py`, against a local
//! websocket server (no network, dummy key).
#![cfg(feature = "voice")]

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::voice::{OpenAISTTModel, STTModel, STTModelSettings, StreamedAudioInput};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

/// A server that completes the session handshake, sends one transcript, then drops the TCP
/// connection without a closing handshake (the transport failing under the listener).
async fn server_dropping_after_a_transcript(transcript: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        ws.send(Message::text(
            json!({"type": "session.created"}).to_string(),
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if message
                .to_text()
                .is_ok_and(|t| t.contains("session.update"))
            {
                break;
            }
        }
        ws.send(Message::text(
            json!({"type": "session.updated"}).to_string(),
        ))
        .await
        .unwrap();
        ws.send(Message::text(
            json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": transcript,
            })
            .to_string(),
        ))
        .await
        .unwrap();
        // Dropping the stream without `close` ends the connection abruptly.
        drop(ws.into_inner());
    });
    format!("http://{addr}/v1")
}

/// Python: `test_listener_timeout_drains_buffered_transcript_before_setup`. A transcript the
/// listener already received is delivered before the listener's failure ends the session.
#[tokio::test]
async fn a_buffered_transcript_is_delivered_before_a_listener_failure() {
    let base = server_dropping_after_a_transcript("Transcript before listener failure").await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let mut session = OpenAISTTModel::new("whisper-1", client)
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();

    let first = tokio::time::timeout(Duration::from_secs(5), session.next_turn())
        .await
        .expect("the transcript must arrive");
    assert_eq!(
        first.unwrap().as_deref(),
        Some("Transcript before listener failure")
    );

    let second = tokio::time::timeout(Duration::from_secs(5), session.next_turn())
        .await
        .expect("the failure must end the session");
    assert!(second.is_err(), "{second:?}");
    let _ = session.close().await;
}
