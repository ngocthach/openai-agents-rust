//! Rust analogue of `test_close_during_setup_finishes_transcription_consumer`
//! (`tests/voice/test_openai_stt.py`).
//!
//! Python's `create_session` returns before the setup handshake ends, so the test closes the
//! session while setup is still pending. Rust's `create_session` awaits the whole handshake, so
//! the way to interrupt setup is to drop that future: the websocket it owns must then be closed
//! and nothing may be left running, with tracing on or off.
#![cfg(feature = "voice")]

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::tracing::trace;
use openai_agents_sdk::voice::{OpenAISTTModel, STTModel, STTModelSettings, StreamedAudioInput};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio_tungstenite::tungstenite::Message;

/// Accept one websocket, send `session.created`, read the client's `session.update`, tell the
/// test setup was reached, and then never answer: only the client closing the socket ends it.
async fn silent_setup_server() -> (String, Arc<Notify>, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let reached = Arc::new(Notify::new());
    let signal = reached.clone();
    let task = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        ws.send(Message::text(
            json!({"type": "session.created"}).to_string(),
        ))
        .await
        .unwrap();
        let mut received = Vec::new();
        let mut announced = false;
        while let Some(Ok(message)) = ws.next().await {
            if let Ok(text) = message.to_text()
                && let Ok(event) = serde_json::from_str::<Value>(text)
            {
                received.push(event);
                if !announced {
                    announced = true;
                    signal.notify_one();
                }
            }
        }
        received
    });
    (format!("http://{addr}/v1"), reached, task)
}

// test_close_during_setup_finishes_transcription_consumer[tracing enabled, disabled]
#[tokio::test]
async fn interrupting_setup_closes_the_websocket() {
    for tracing_disabled in [false, true] {
        let (base, reached, server) = silent_setup_server().await;
        let model = OpenAISTTModel::new(
            "gpt-4o-mini-transcribe",
            OpenAIClient::new("test-key").with_base_url(base),
        );

        let setup = tokio::spawn(async move {
            model
                .create_session(
                    &StreamedAudioInput::new(),
                    &STTModelSettings::default(),
                    false,
                    false,
                )
                .await
                .map(|_| ())
        });
        trace("close during STT setup")
            .disabled(tracing_disabled)
            .build()
            .run(async {
                tokio::time::timeout(Duration::from_secs(5), reached.notified())
                    .await
                    .expect("setup was never reached");
                setup.abort();
                assert!(setup.await.unwrap_err().is_cancelled());
            })
            .await;

        let received = tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("the websocket was left open")
            .unwrap();
        assert_eq!(received[0]["type"], "session.update");
        assert_eq!(received.len(), 1, "tracing_disabled={tracing_disabled}");
    }
}

mod logging {
    use std::sync::{Mutex, OnceLock};

    use base64::Engine as _;
    use futures::{SinkExt, StreamExt};
    use openai_agents_sdk::models::OpenAIClient;
    use openai_agents_sdk::voice::{
        OpenAISTTModel, STTModel, STTModelSettings, StreamedAudioInput,
    };
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;

    struct Sink(Mutex<Vec<String>>);

    impl log::Log for Sink {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            self.0
                .lock()
                .unwrap()
                .push(format!("{} {}", record.target(), record.args()));
        }
        fn flush(&self) {}
    }

    fn sink() -> &'static Sink {
        static SINK: OnceLock<&'static Sink> = OnceLock::new();
        SINK.get_or_init(|| {
            let sink: &'static Sink = Box::leak(Box::new(Sink(Mutex::new(Vec::new()))));
            log::set_logger(sink).expect("no other logger in this test binary");
            // Debug, as in the Python test. `tungstenite` itself logs the full handshake request
            // (query string and headers) at TRACE, which no SDK setting can silence.
            log::set_max_level(log::LevelFilter::Debug);
            sink
        })
    }

    // test_openai_websocket_logger_does_not_emit_debug_connection_data
    #[tokio::test]
    async fn a_streaming_session_logs_no_connection_or_audio_data() {
        let sink = sink();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
            ws.send(Message::text(
                json!({"type": "session.created"}).to_string(),
            ))
            .await
            .unwrap();
            let mut updated = false;
            while let Some(Ok(message)) = ws.next().await {
                let Ok(text) = message.to_text() else {
                    continue;
                };
                let Ok(event) = serde_json::from_str::<Value>(text) else {
                    continue;
                };
                if event["type"] == "session.update" && !updated {
                    updated = true;
                    ws.send(Message::text(
                        json!({"type": "session.updated"}).to_string(),
                    ))
                    .await
                    .unwrap();
                }
            }
        });

        let samples = vec![0x1234i16; 2400];
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let encoded_audio = base64::engine::general_purpose::STANDARD.encode(&pcm);

        let client = OpenAIClient::new("test-key")
            .with_base_url(format!("http://{addr}/v1"))
            .with_default_query("proxy_token", "query-secret")
            .with_header("X-Proxy-Token", "header-secret");
        let input = StreamedAudioInput::new();
        let mut session = OpenAISTTModel::new("gpt-4o-transcribe", client)
            .create_session(&input, &STTModelSettings::default(), false, false)
            .await
            .unwrap();
        input.add_audio(samples).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        session.close().await.unwrap();
        server.abort();

        let logged = sink.0.lock().unwrap().join("\n");
        for secret in ["query-secret", "header-secret", encoded_audio.as_str()] {
            assert!(
                !logged.contains(secret),
                "{secret} reached the logs: {:?}",
                logged
                    .lines()
                    .filter(|l| l.contains(secret))
                    .collect::<Vec<_>>()
            );
        }
    }
}
