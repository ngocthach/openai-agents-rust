//! Shared mock-server harness for the `py_realtime_*` test files.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeModel, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimePlaybackTracker,
    RealtimeSessionModelSettings,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::tungstenite::protocol::Message;

/// Envelope keys for frames that are not plain JSON text; see `push_raw` and `push_close`.
const RAW_TEXT_KEY: &str = "__py_test_raw_text__";
const CLOSE_CODE_KEY: &str = "__py_test_close_code__";

/// How long a test waits for an expected frame or event before failing.
pub const TIMEOUT: Duration = Duration::from_secs(5);

/// A local websocket server standing in for the Realtime API.
///
/// Everything the client sends arrives on `client_rx`; anything pushed into `server_tx` is
/// delivered to the client. That is enough to exercise the transport's whole event mapping
/// without reaching for its private methods, which is how the Python tests drive it.
pub struct MockRealtimeServer {
    pub port: u16,
    pub client_rx: UnboundedReceiver<Value>,
    pub server_tx: UnboundedSender<Value>,
    pub handle: tokio::task::JoinHandle<()>,
}

impl MockRealtimeServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (client_tx, client_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        let (server_tx, mut server_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();

        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let (mut sink, mut source) = ws.split();

            // Pump the test's frames out to the client.
            let writer = tokio::spawn(async move {
                while let Some(value) = server_rx.recv().await {
                    let message = if let Some(text) =
                        value.get(RAW_TEXT_KEY).and_then(Value::as_str)
                    {
                        Message::text(text)
                    } else if let Some(code) = value.get(CLOSE_CODE_KEY).and_then(Value::as_u64) {
                        Message::Close(Some(tokio_tungstenite::tungstenite::protocol::CloseFrame {
                            code: (code as u16).into(),
                            reason: "py-test-parity".into(),
                        }))
                    } else {
                        Message::text(value.to_string())
                    };
                    if sink.send(message).await.is_err() {
                        break;
                    }
                }
            });

            // Forward everything the client sends back to the test.
            while let Some(Ok(message)) = source.next().await {
                if let Message::Text(text) = message
                    && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
                {
                    let _ = client_tx.send(value);
                }
            }
            writer.abort();
        });

        MockRealtimeServer {
            port,
            client_rx,
            server_tx,
            handle,
        }
    }

    pub fn url(&self) -> String {
        format!("ws://127.0.0.1:{}/v1/realtime", self.port)
    }

    /// Push a server event to the client.
    pub fn push(&self, event: Value) {
        self.server_tx.send(event).expect("mock server is running");
    }

    /// Push `text` to the client verbatim, so a test can send something that is not valid JSON.
    pub fn push_raw(&self, text: &str) {
        self.push(json!({ RAW_TEXT_KEY: text }));
    }

    /// Close the connection from the server side with a close frame carrying `code`.
    pub fn push_close(&self, code: u16) {
        self.push(json!({ CLOSE_CODE_KEY: code }));
    }

    /// The next client frame whose `type` matches, skipping the ones before it.
    pub async fn next_frame(&mut self, frame_type: &str) -> Value {
        tokio::time::timeout(TIMEOUT, async {
            while let Some(value) = self.client_rx.recv().await {
                if value["type"] == frame_type {
                    return value;
                }
            }
            panic!("the client closed before sending a {frame_type} frame");
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for a {frame_type} frame"))
    }

    /// Every client frame that arrived before the sentinel `session.update`.
    ///
    /// A negative assertion ("the client never sent X") cannot be made by waiting a fixed
    /// interval: under load the frame may simply be late. Instead the test sends a marker
    /// through the same ordered connection with [`send_marker`] and reads up to it, so
    /// everything the client produced for the preceding action has definitely been seen.
    pub async fn frames_before_marker(&mut self) -> Vec<Value> {
        let mut found = Vec::new();
        tokio::time::timeout(TIMEOUT, async {
            while let Some(value) = self.client_rx.recv().await {
                if value["type"] == "session.update"
                    && value["session"]["instructions"] == MARKER_INSTRUCTIONS
                {
                    return;
                }
                found.push(value);
            }
            panic!("the client closed before the marker frame arrived");
        })
        .await
        .expect("timed out waiting for the marker frame");
        found
    }

    pub fn stop(self) {
        self.handle.abort();
    }
}

/// Instructions identifying the marker `session.update` used to bound negative assertions.
pub const MARKER_INSTRUCTIONS: &str = "py-test-parity-marker";

/// Push a marker `session.update` through the transport.
///
/// The websocket preserves order, so every frame the client produced before this call reaches
/// the mock server ahead of the marker.
pub async fn send_marker(model: &OpenAIRealtimeWebSocketModel) {
    model
        .send_event(RealtimeModelSendEvent::session_update(
            RealtimeSessionModelSettings::new().with_instructions(MARKER_INSTRUCTIONS),
        ))
        .await
        .expect("the marker session.update is sent");
}

/// Connect a transport to `server`, with `settings` as the initial session settings.
pub async fn connect(
    server: &MockRealtimeServer,
    settings: Option<RealtimeSessionModelSettings>,
    tracker: Option<Arc<RealtimePlaybackTracker>>,
) -> (Arc<OpenAIRealtimeWebSocketModel>, RealtimeModelEvents) {
    let model = Arc::new(OpenAIRealtimeWebSocketModel::new());
    let mut config = RealtimeModelConfig::new()
        .with_api_key("sk-test")
        .with_url(server.url());
    if let Some(settings) = settings {
        config = config.with_initial_model_settings(settings);
    }
    if let Some(tracker) = tracker {
        config = config.with_playback_tracker(tracker);
    }
    let events = model
        .connect(config)
        .await
        .expect("connects to mock server");
    (model, events)
}

/// Acknowledge the session, optionally negotiating an output audio format.
pub fn session_created(format: Option<Value>) -> Value {
    let mut session = json!({"id": "session_456", "type": "realtime", "model": "gpt-realtime-2.1"});
    if let Some(format) = format {
        session["audio"] = json!({"output": {"format": format}});
    }
    json!({"type": "session.created", "event_id": "event_123", "session": session})
}

/// Wait for the first event matching `predicate`.
pub async fn wait_for_event<F, T>(events: &mut RealtimeModelEvents, predicate: F) -> T
where
    F: Fn(RealtimeModelEvent) -> Option<T>,
{
    tokio::time::timeout(TIMEOUT, async {
        loop {
            let event = events.recv().await.expect("event stream stays open");
            if let Some(value) = predicate(event) {
                return value;
            }
        }
    })
    .await
    .expect("the expected model event never arrived")
}

/// Minimal base64 encoder, so the mock server does not need the `base64` crate.
pub fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}
