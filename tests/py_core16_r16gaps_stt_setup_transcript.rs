//! A transcript the server sends before the session finishes setting up is not lost: it is
//! delivered to the consumer once setup completes, as Python's event queue holds it.
#![cfg(feature = "voice")]

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::voice::{OpenAISTTModel, STTModel, STTModelSettings, StreamedAudioInput};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

fn transcript(text: &str) -> Message {
    Message::text(
        json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text,
        })
        .to_string(),
    )
}

fn event(kind: &str) -> Message {
    Message::text(json!({ "type": kind }).to_string())
}

async fn server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        ws.send(event("session.created")).await.unwrap();
        ws.send(transcript("before the update")).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if message
                .to_text()
                .is_ok_and(|t| t.contains("session.update"))
            {
                break;
            }
        }
        ws.send(transcript("between update and ack")).await.unwrap();
        ws.send(event("session.updated")).await.unwrap();
        // Keep the socket open until the client goes away.
        while let Some(Ok(message)) = ws.next().await {
            let _: Option<Value> = serde_json::from_str(message.to_text().unwrap_or("")).ok();
        }
    });
    format!("http://{addr}/v1")
}

#[tokio::test]
async fn transcripts_received_during_setup_are_delivered() {
    let base = server().await;
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

    for expected in ["before the update", "between update and ack"] {
        let turn = tokio::time::timeout(Duration::from_secs(5), session.next_turn())
            .await
            .expect("the transcript arrives");
        assert_eq!(turn.unwrap().as_deref(), Some(expected));
    }
    let _ = session.close().await;
}
