//! Port of the Python SDK's `tests/voice/test_openai_stt*.py`, `test_openai_tts.py`,
//! `test_openai_model_provider.py` and `test_pipeline_tracing.py`.
//!
//! The OpenAI models run against local mock HTTP and websocket servers; no network and no real
//! API keys are used.
#![cfg(feature = "voice")]

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::{OpenAIClient, RequestAuth};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, REDACTED_TRACE_ERROR_MESSAGE, TracingProcessor, set_trace_processors,
    trace,
};
use openai_agents_sdk::voice::testing::{FakeTTSModel, FakeVoiceWorkflow};
use openai_agents_sdk::voice::{
    AudioInput, OpenAISTTModel, OpenAITTSModel, OpenAIVoiceModelProvider, STTModel,
    STTModelSettings, StreamedAudioInput, StreamedAudioResult, TTSModel, TTSModelSettings,
    VoiceModelProvider, VoicePipeline, VoicePipelineConfig, VoiceStreamEvent,
    VoiceWebSocketOptions,
};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

// ===== Shared test plumbing =====

/// Record traces in memory. This replaces the OpenAI exporter, so no test can reach the
/// network, while spans stay observable for the tracing assertions.
fn trace_processor() -> Arc<InMemoryTraceProcessor> {
    static PROCESSOR: OnceLock<Arc<InMemoryTraceProcessor>> = OnceLock::new();
    PROCESSOR
        .get_or_init(|| {
            let processor = Arc::new(InMemoryTraceProcessor::new());
            set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
            processor
        })
        .clone()
}

/// Every exported span of `trace_id`.
fn spans_of(processor: &InMemoryTraceProcessor, trace_id: &str) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| item["object"] == "trace.span" && item["trace_id"] == trace_id)
        .collect()
}

fn spans_of_type<'a>(spans: &'a [Value], span_type: &str) -> Vec<&'a Value> {
    spans
        .iter()
        .filter(|span| span["span_data"]["type"] == span_type)
        .collect()
}

/// Collapse pipeline events to labels plus the audio bytes, as Python's `extract_events` does.
async fn extract_events(result: &mut StreamedAudioResult) -> (Vec<String>, Vec<u8>) {
    let mut labels = Vec::new();
    let mut audio = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        match event {
            VoiceStreamEvent::Audio(data) => {
                audio.extend(data.to_bytes());
                labels.push("audio".to_string());
            }
            VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
            VoiceStreamEvent::Error(error) => labels.push(format!("error: {error}")),
        }
    }
    (labels, audio)
}

fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

// ===== A mock HTTP server =====

/// A canned HTTP response.
enum Reply {
    /// A JSON body served with 200.
    Json(Value),
    /// A JSON body served with this status.
    Status(u16, Value),
    /// Raw bytes with a content type, served with 200.
    Bytes(&'static str, Vec<u8>),
}

/// One captured request: the request line target and the raw body.
#[derive(Clone)]
struct Captured {
    path: String,
    body: Vec<u8>,
}

impl Captured {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }

    /// The text of a multipart form field.
    fn field(&self, name: &str) -> Option<String> {
        let body = String::from_utf8_lossy(&self.body).to_string();
        let marker = format!("name=\"{name}\"\r\n\r\n");
        let start = body.find(&marker)? + marker.len();
        let end = body[start..].find("\r\n")?;
        Some(body[start..start + end].to_string())
    }
}

/// Serve `replies` in order; returns the base URL and the captured requests.
async fn mock_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Captured>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let request = loop {
                let Ok(n) = socket.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let Some(index) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&buf[..index]).to_string();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if buf.len() >= index + 4 + length {
                    break Captured {
                        path: headers.split_whitespace().nth(1).unwrap().to_string(),
                        body: buf[index + 4..index + 4 + length].to_vec(),
                    };
                }
            };
            sink.lock().unwrap().push(request);
            let (status, content_type, body) = match replies.next().expect("unexpected request") {
                Reply::Json(value) => (
                    200,
                    "application/json".to_string(),
                    value.to_string().into_bytes(),
                ),
                Reply::Status(status, value) => (
                    status,
                    "application/json".to_string(),
                    value.to_string().into_bytes(),
                ),
                Reply::Bytes(content_type, bytes) => (200, content_type.to_string(), bytes),
            };
            let head = format!(
                "HTTP/1.1 {status} STATUS\r\ncontent-type: {content_type}\r\nx-request-id: req_voice\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            socket.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

// ===== A scripted realtime-transcription websocket server =====

/// One step of the server's script.
#[derive(Clone)]
enum Step {
    /// Send this event.
    Send(Value),
    /// Send this raw text frame.
    SendText(&'static str),
    /// Wait for the next client event and require this type.
    Recv(&'static str),
    /// Send the `input_audio_buffer_commit_empty` error for the last received commit.
    CommitEmptyError,
    /// Wait until the test releases the server.
    Wait(Arc<Notify>),
    /// Tell the test the script reached this point.
    Signal(Arc<Notify>),
    /// Sleep for this many milliseconds.
    Sleep(u64),
    /// Close the socket from the server side.
    Close,
    /// Read until the client closes the socket.
    WaitClosed,
}

/// What the client's handshake looked like.
#[derive(Clone, Default)]
struct Handshake {
    target: String,
    headers: Vec<(String, String)>,
}

impl Handshake {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

struct WsServer {
    base: String,
    received: Arc<Mutex<Vec<Value>>>,
    handshake: Arc<Mutex<Handshake>>,
    task: JoinHandle<()>,
}

impl WsServer {
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("test-key").with_base_url(self.base.clone())
    }

    fn received(&self) -> Vec<Value> {
        self.received.lock().unwrap().clone()
    }

    fn handshake(&self) -> Handshake {
        self.handshake.lock().unwrap().clone()
    }

    fn of_type(&self, event_type: &str) -> Vec<Value> {
        self.received()
            .into_iter()
            .filter(|event| event["type"] == event_type)
            .collect()
    }

    /// The `transcription` block of the `session.update` the client sent.
    fn transcription_config(&self) -> Value {
        self.of_type("session.update")[0]["session"]["audio"]["input"]["transcription"].clone()
    }

    /// Wait for the script to finish, which needs the client to have closed the socket.
    async fn finished(self) {
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("the websocket was left open")
            .unwrap();
    }
}

/// The handshake events of a session, with the given event names.
fn setup_with(created: &str, updated: &str) -> Vec<Step> {
    vec![
        Step::Send(json!({"type": created})),
        Step::Recv("session.update"),
        Step::Send(json!({"type": updated})),
    ]
}

fn setup() -> Vec<Step> {
    setup_with("session.created", "session.updated")
}

fn completed(item_id: &str, transcript: &str) -> Step {
    Step::Send(json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "transcript": transcript,
    }))
}

async fn ws_server(steps: Vec<Step>) -> WsServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let handshake = Arc::new(Mutex::new(Handshake::default()));
    let (sink, shake) = (received.clone(), handshake.clone());
    let task = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        #[allow(clippy::result_large_err)]
        let mut ws =
            tokio_tungstenite::accept_hdr_async(socket, |req: &Request, resp: Response| {
                let headers = req
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string()))
                    .collect();
                *shake.lock().unwrap() = Handshake {
                    target: req.uri().to_string(),
                    headers,
                };
                Ok(resp)
            })
            .await
            .unwrap();
        let mut last_commit_id = String::new();
        for step in steps {
            match step {
                Step::Send(event) => {
                    if ws.send(Message::text(event.to_string())).await.is_err() {
                        return;
                    }
                }
                Step::SendText(text) => {
                    if ws.send(Message::text(text)).await.is_err() {
                        return;
                    }
                }
                Step::CommitEmptyError => {
                    let event = json!({
                        "type": "error",
                        "error": {
                            "code": "input_audio_buffer_commit_empty",
                            "event_id": last_commit_id,
                            "message": "Input buffer is empty",
                        },
                    });
                    if ws.send(Message::text(event.to_string())).await.is_err() {
                        return;
                    }
                }
                Step::Recv(expected) => loop {
                    let Some(Ok(message)) = ws.next().await else {
                        return;
                    };
                    let Ok(text) = message.to_text() else {
                        continue;
                    };
                    let Ok(event) = serde_json::from_str::<Value>(text) else {
                        continue;
                    };
                    if event["type"] == "input_audio_buffer.commit" {
                        last_commit_id = event["event_id"].as_str().unwrap_or_default().into();
                    }
                    sink.lock().unwrap().push(event.clone());
                    assert_eq!(event["type"], expected, "unexpected client event {event}");
                    break;
                },
                Step::Wait(notify) => notify.notified().await,
                Step::Signal(notify) => notify.notify_one(),
                Step::Sleep(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
                Step::Close => {
                    let _ = ws.close(None).await;
                }
                Step::WaitClosed => {
                    while let Some(Ok(message)) = ws.next().await {
                        if let Ok(text) = message.to_text()
                            && let Ok(event) = serde_json::from_str::<Value>(text)
                        {
                            sink.lock().unwrap().push(event);
                        }
                    }
                }
            }
        }
    });
    WsServer {
        base: format!("http://{addr}/v1"),
        received,
        handshake,
        task,
    }
}

/// Open a session against a script that only sets up, then close it.
async fn open_and_close(
    server: WsServer,
    client: OpenAIClient,
    model: &str,
    settings: &STTModelSettings,
) -> WsServer {
    let stt = OpenAISTTModel::new(model, client);
    let mut session = stt
        .create_session(&StreamedAudioInput::new(), settings, false, false)
        .await
        .unwrap();
    session.close().await.unwrap();
    server
}

async fn config_for(model: &str, settings: STTModelSettings) -> Value {
    let mut steps = setup();
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let client = server.client();
    let server = open_and_close(server, client, model, &settings).await;
    let config = server.transcription_config();
    server.finished().await;
    config
}

// ===== tests/voice/test_openai_tts.py =====

async fn speak(settings: &TTSModelSettings, text: &str) -> (Vec<u8>, Value) {
    trace_processor();
    let (base, captured) = mock_server(vec![Reply::Bytes(
        "application/octet-stream",
        b"abcdef".to_vec(),
    )])
    .await;
    let model = OpenAITTSModel::new(
        "test-model",
        OpenAIClient::new("sk-test").with_base_url(base),
    );
    let mut stream = model.run(text, settings).await.unwrap();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    let request = captured.lock().unwrap()[0].clone();
    assert_eq!(request.path, "/v1/audio/speech");
    (out, request.json())
}

#[tokio::test]
async fn openai_tts_default_voice_and_instructions() {
    let settings = TTSModelSettings::default();
    let (out, body) = speak(&settings, "hello world").await;
    assert_eq!(out, b"abcdef");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["voice"], "ash");
    assert_eq!(body["input"], "hello world");
    assert_eq!(body["response_format"], "pcm");
    assert!(body.get("speed").is_none());
    assert_eq!(body["instructions"], settings.instructions);
}

#[tokio::test]
async fn openai_tts_custom_voice_and_instructions() {
    let settings = TTSModelSettings::default()
        .with_voice(openai_agents_sdk::voice::TTSVoice::Fable)
        .with_instructions("Custom instructions");
    let (out, body) = speak(&settings, "hi").await;
    assert_eq!(out, b"abcdef");
    assert_eq!(body["voice"], "fable");
    assert_eq!(body["instructions"], "Custom instructions");
}

#[tokio::test]
async fn openai_tts_forwards_speed() {
    let (out, body) = speak(&TTSModelSettings::default().with_speed(1.5), "hi").await;
    assert_eq!(out, b"abcdef");
    assert_eq!(body["speed"], 1.5);
}

// ===== tests/voice/test_openai_model_provider.py =====

#[tokio::test]
async fn voice_provider_transcription_model_and_language_on_wire() {
    trace_processor();
    for model_name in [None, Some("gpt-4o-transcribe")] {
        for language in [None, Some("fr")] {
            let (base, captured) = mock_server(vec![Reply::Json(json!({"text": "Bonjour"}))]).await;
            let provider = OpenAIVoiceModelProvider::new()
                .with_client(OpenAIClient::new("test-key").with_base_url(base));
            let model = provider.get_stt_model(model_name);
            let mut settings = STTModelSettings::default()
                .with_prompt("A greeting")
                .with_temperature(0.2);
            if let Some(language) = language {
                settings = settings.with_language(language);
            }
            let transcript = model
                .transcribe(&AudioInput::new(vec![0i16; 240]), &settings, false, false)
                .await
                .unwrap();
            assert_eq!(transcript, "Bonjour");

            let request = captured.lock().unwrap()[0].clone();
            assert_eq!(request.path, "/v1/audio/transcriptions");
            let expected_model = model_name.unwrap_or("gpt-transcribe");
            assert_eq!(request.field("model").as_deref(), Some(expected_model));
            assert_eq!(request.field("prompt").as_deref(), Some("A greeting"));
            assert_eq!(request.field("temperature").as_deref(), Some("0.2"));
            match (language, model_name) {
                (None, _) => {
                    assert!(request.field("language").is_none());
                    assert!(request.field("languages[]").is_none());
                }
                (Some(_), None) => {
                    assert_eq!(request.field("languages[]").as_deref(), Some("fr"));
                    assert!(request.field("language").is_none());
                }
                (Some(_), Some(_)) => {
                    assert_eq!(request.field("language").as_deref(), Some("fr"));
                    assert!(request.field("languages[]").is_none());
                }
            }
        }
    }
}

#[tokio::test]
async fn voice_provider_streamed_default_transcription_config() {
    trace_processor();
    let mut steps = setup();
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let provider = OpenAIVoiceModelProvider::new().with_client(server.client());
    let model = provider.get_stt_model(None);
    let mut session = model
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default().with_language("fr"),
            false,
            false,
        )
        .await
        .unwrap();
    session.close().await.unwrap();
    assert_eq!(
        server.transcription_config(),
        json!({"model": "gpt-transcribe", "languages": ["fr"]})
    );
    server.finished().await;
}

#[test]
fn voice_provider_accepts_the_configured_client() {
    // Python also rejects a client combined with `api_key`/`base_url` kwargs; the Rust builder
    // takes the client as the one source of configuration, so only the accepting half applies.
    let client = OpenAIClient::new("test_key").with_base_url("https://example.test/v1");
    let provider = OpenAIVoiceModelProvider::new().with_client(client);
    assert_eq!(provider.client().base_url(), "https://example.test/v1");
    assert_eq!(provider.client().api_key().as_deref(), Some("test_key"));
}

#[test]
fn voice_provider_explicit_options_override_the_default_client() {
    let default = OpenAIClient::new("sk-default").with_base_url("https://default.example.test/v1");
    for value in ["sk-voice", ""] {
        let provider = OpenAIVoiceModelProvider::new()
            .with_client(default.clone())
            .with_api_key(value);
        assert_eq!(provider.client().api_key().as_deref(), Some(value));
        assert_eq!(
            provider.client().base_url(),
            "https://default.example.test/v1"
        );
    }
    for value in ["https://voice.example.test/v1", ""] {
        let provider = OpenAIVoiceModelProvider::new()
            .with_client(default.clone())
            .with_base_url(value);
        assert_eq!(provider.client().base_url(), value);
        assert_eq!(provider.client().api_key().as_deref(), Some("sk-default"));
    }
}

// ===== tests/voice/test_openai_stt_client_config.py =====

async fn handshake_of(client_for: impl FnOnce(String) -> OpenAIClient) -> Handshake {
    trace_processor();
    let mut steps = setup();
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let base = server.base.clone();
    let server = open_and_close(
        server,
        client_for(base),
        "gpt-4o-transcribe",
        &STTModelSettings::default(),
    )
    .await;
    let handshake = server.handshake();
    server.finished().await;
    handshake
}

#[tokio::test]
async fn streaming_stt_websocket_url_uses_client_base_url() {
    // The wss scheme swap is covered by the unit test of the URL builder.
    let handshake =
        handshake_of(|base| OpenAIClient::new("test-key").with_base_url(format!("{base}/"))).await;
    assert_eq!(handshake.target, "/v1/realtime?intent=transcription");
}

#[tokio::test]
async fn streaming_stt_websocket_url_prefers_websocket_base_url() {
    let handshake = handshake_of(|base| {
        let origin = base.trim_end_matches("/v1");
        OpenAIClient::new("test-key")
            .with_base_url("https://ignored.example.test/v1/")
            .with_websocket_base_url(format!("{origin}/custom/?tenant=one"))
    })
    .await;
    assert_eq!(
        handshake.target,
        "/custom/realtime?tenant=one&intent=transcription"
    );
}

#[tokio::test]
async fn streaming_stt_websocket_url_merges_client_default_query() {
    let handshake = handshake_of(|base| {
        let origin = base.trim_end_matches("/v1");
        OpenAIClient::new("test-key")
            .with_websocket_base_url(format!("{origin}/custom/?tenant=one&remove=base"))
            .with_default_query("api-version", "2026-08-01-preview")
            .with_default_query("remove", "default")
    })
    .await;
    let query = handshake.target.split_once('?').unwrap().1.to_string();
    let params: Vec<&str> = query.split('&').collect();
    assert!(params.contains(&"tenant=one"), "{query}");
    assert!(
        params.contains(&"api-version=2026-08-01-preview"),
        "{query}"
    );
    assert!(params.contains(&"intent=transcription"), "{query}");
    // A client default overrides the same parameter of the base URL.
    assert!(params.contains(&"remove=default"), "{query}");
    assert!(!params.contains(&"remove=base"), "{query}");
}

#[tokio::test]
async fn streaming_stt_websocket_headers_use_client_configuration() {
    let handshake = handshake_of(|base| {
        OpenAIClient::new("sk-client")
            .with_base_url(base)
            .with_organization("org-client")
            .with_project("proj-client")
            .with_header("X-Proxy-Token", "proxy-token")
    })
    .await;
    assert_eq!(handshake.header("authorization"), Some("Bearer sk-client"));
    assert_eq!(handshake.header("openai-organization"), Some("org-client"));
    assert_eq!(handshake.header("openai-project"), Some("proj-client"));
    assert_eq!(handshake.header("x-proxy-token"), Some("proxy-token"));
    assert!(handshake.header("openai-log-session").is_none());
}

#[tokio::test]
async fn streaming_stt_websocket_headers_omit_removes_inherited_header() {
    let handshake = handshake_of(|base| {
        OpenAIClient::new("sk-client")
            .with_base_url(base)
            .without_header("authorization")
    })
    .await;
    assert!(handshake.header("authorization").is_none());
    assert!(handshake.header("openai-log-session").is_none());
}

#[tokio::test]
async fn streaming_stt_websocket_session_header_preserves_client_configuration() {
    let handshake = handshake_of(|base| {
        OpenAIClient::new("sk-client")
            .with_base_url(base)
            .with_header("openai-log-session", "0")
    })
    .await;
    assert_eq!(handshake.header("openai-log-session"), Some("0"));
}

// ===== tests/voice/test_openai_stt_api_key_refresh.py =====

/// Rotates the API key on every request, the Rust form of Python's `_refresh_api_key`.
#[derive(Debug, Default)]
struct RotatingAuth {
    refreshes: Mutex<usize>,
}

#[async_trait]
impl RequestAuth for RotatingAuth {
    async fn authorize(
        &self,
        _method: &str,
        _url: &str,
        _body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> openai_agents_sdk::Result<()> {
        *self.refreshes.lock().unwrap() += 1;
        headers.insert("authorization", "Bearer sk-refreshed".parse().unwrap());
        Ok(())
    }
}

#[tokio::test]
async fn streamed_stt_refreshes_the_api_key_before_the_handshake() {
    let auth = Arc::new(RotatingAuth::default());
    let auth_for_client = auth.clone();
    let handshake = handshake_of(move |base| {
        OpenAIClient::new("stale")
            .with_base_url(base)
            .with_request_auth(auth_for_client)
    })
    .await;
    assert_eq!(*auth.refreshes.lock().unwrap(), 1);
    assert_eq!(
        handshake.header("authorization"),
        Some("Bearer sk-refreshed")
    );
}

// ===== tests/voice/test_openai_stt_session_config.py =====

#[tokio::test]
async fn streaming_stt_sends_language_and_prompt() {
    trace_processor();
    for (model, field, value) in [
        ("gpt-4o-transcribe", "language", json!("fr")),
        ("gpt-transcribe", "languages", json!(["fr"])),
        ("gpt-live-transcribe", "languages", json!(["fr"])),
    ] {
        let config = config_for(
            model,
            STTModelSettings::default()
                .with_language("fr")
                .with_prompt("domain vocabulary"),
        )
        .await;
        assert_eq!(
            config,
            json!({"model": model, field: value, "prompt": "domain vocabulary"})
        );
    }
}

#[tokio::test]
async fn streaming_stt_omits_unset_language_and_prompt() {
    trace_processor();
    let config = config_for("gpt-4o-transcribe", STTModelSettings::default()).await;
    assert_eq!(config, json!({"model": "gpt-4o-transcribe"}));
}

#[tokio::test]
async fn streaming_stt_sends_languages_over_language() {
    trace_processor();
    for model in ["gpt-transcribe", "gpt-live-transcribe"] {
        let config = config_for(
            model,
            STTModelSettings::default()
                .with_language("fr")
                .with_languages(vec!["fr".into(), "eng".into(), "zh-tw".into()]),
        )
        .await;
        assert_eq!(
            config,
            json!({"model": model, "languages": ["fr", "eng", "zh-tw"]})
        );
    }
}

#[tokio::test]
async fn streaming_stt_sends_keywords() {
    trace_processor();
    let config = config_for(
        "gpt-live-transcribe",
        STTModelSettings::default().with_keywords(vec!["agents".into(), "sdk".into()]),
    )
    .await;
    assert_eq!(
        config,
        json!({"model": "gpt-live-transcribe", "keywords": ["agents", "sdk"]})
    );
}

/// The `model_config` of the transcription span a session records for `model`.
async fn session_span_config(
    model: &str,
    settings: STTModelSettings,
    include_sensitive: bool,
) -> Value {
    let processor = trace_processor();
    let mut steps = setup();
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let caller = trace("stt session span config").build();
    let trace_id = caller.trace_id().to_string();
    let client = server.client();
    let stt = OpenAISTTModel::new(model, client);
    caller
        .run(async {
            let mut session = stt
                .create_session(
                    &StreamedAudioInput::new(),
                    &settings,
                    include_sensitive,
                    false,
                )
                .await
                .unwrap();
            session.close().await.unwrap();
        })
        .await;
    server.finished().await;
    let spans = spans_of(&processor, &trace_id);
    let transcription = spans_of_type(&spans, "transcription");
    assert!(!transcription.is_empty(), "{spans:#?}");
    transcription[0]["span_data"]["model_config"].clone()
}

#[tokio::test]
async fn streaming_stt_trace_records_transcription_options_with_sensitive_data() {
    let config = session_span_config(
        "gpt-live-transcribe",
        STTModelSettings::default()
            .with_prompt("customer support")
            .with_language("en")
            .with_temperature(0.2)
            .with_languages(vec!["en".into(), "fr".into()])
            .with_keywords(vec!["Agents SDK".into()]),
        true,
    )
    .await;
    assert_eq!(
        config,
        json!({
            "temperature": 0.2f32,
            "language": null,
            "languages": ["en", "fr"],
            "keywords": ["Agents SDK"],
            "prompt": "customer support",
            "turn_detection": {"type": "semantic_vad"},
        })
    );
}

#[tokio::test]
async fn streaming_stt_trace_redacts_text_options_without_sensitive_data() {
    let config = session_span_config(
        "gpt-live-transcribe",
        STTModelSettings::default()
            .with_prompt("customer account vocabulary")
            .with_keywords(vec!["CUSTOMER_SECRET_NAME".into()]),
        false,
    )
    .await;
    assert!(config["keywords"].is_null(), "{config}");
    assert!(config["prompt"].is_null(), "{config}");
    assert!(!config.to_string().contains("CUSTOMER_SECRET_NAME"));
}

// ===== tests/voice/test_openai_stt_eof.py =====

fn eof_pipeline(client: OpenAIClient, turns: usize) -> (VoicePipeline, Arc<FakeVoiceWorkflow>) {
    let mut workflow = FakeVoiceWorkflow::new();
    let mut tts = FakeTTSModel::new();
    for _ in 0..turns {
        workflow = workflow.with_turn(vec!["Reply.".to_string()]);
        tts = tts.with_result(vec![pcm_bytes(&[0; 10]).into()]);
    }
    let workflow = Arc::new(workflow);
    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(Arc::new(OpenAISTTModel::new("gpt-4o-transcribe", client)))
        .with_tts_model(Arc::new(tts))
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));
    (pipeline, workflow)
}

#[tokio::test]
async fn eof_drains_transcriptions_before_finishing() {
    trace_processor();
    for mode in [
        "empty",
        "final_buffer",
        "vad",
        "vad_race",
        "empty_transcript",
        "legacy",
    ] {
        let cleared = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let expected: Vec<&str> = match mode {
            "empty" | "empty_transcript" => vec![],
            "vad_race" => vec!["Second phrase", "First phrase"],
            _ => vec!["Final phrase"],
        };
        let committed = |id: &str| json!({"type": "input_audio_buffer.committed", "item_id": id});
        let mut steps = setup();
        if mode != "empty" {
            steps.push(Step::Recv("input_audio_buffer.append"));
        }
        if mode == "vad" {
            steps.push(Step::Send(committed("first")));
            steps.push(completed("first", "Final phrase"));
        }
        steps.push(Step::Recv("input_audio_buffer.commit"));
        match mode {
            "empty" | "vad" | "vad_race" => {
                if mode == "vad_race" {
                    // VAD won the commit race, but neither transcript has completed yet.
                    steps.push(Step::Send(committed("first")));
                    steps.push(Step::Send(committed("second")));
                }
                steps.push(Step::CommitEmptyError);
            }
            _ => steps.push(Step::Send(committed("first"))),
        }
        steps.push(Step::Recv("input_audio_buffer.clear"));
        steps.push(Step::Send(json!({"type": "input_audio_buffer.cleared"})));
        steps.push(Step::Signal(cleared.clone()));
        if !matches!(mode, "empty" | "vad") {
            steps.push(Step::Wait(release.clone()));
            if mode == "vad_race" {
                steps.push(completed("second", "Second phrase"));
            }
            if mode == "legacy" {
                steps.push(Step::Send(json!({
                    "type": "input_audio_transcription_completed",
                    "transcript": "Final phrase",
                })));
            } else {
                steps.push(completed(
                    "first",
                    match mode {
                        "empty_transcript" => "",
                        "vad_race" => "First phrase",
                        _ => "Final phrase",
                    },
                ));
            }
        }
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;

        let (pipeline, workflow) = eof_pipeline(server.client(), expected.len());
        let input = StreamedAudioInput::new();
        if mode != "empty" {
            input.add_audio(vec![0i16; 4800]).unwrap();
        }
        input.end().unwrap();
        let mut result = pipeline.run(input).await.unwrap();
        let mut consumer = tokio::spawn(async move { extract_events(&mut result).await.0 });

        tokio::time::timeout(Duration::from_secs(2), cleared.notified())
            .await
            .expect("the server never saw the clear");
        if !matches!(mode, "empty" | "vad") {
            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut consumer)
                    .await
                    .is_err(),
                "{mode}: EOF must wait for the outstanding transcripts"
            );
        }
        release.notify_one();
        let labels = tokio::time::timeout(Duration::from_secs(5), consumer)
            .await
            .unwrap_or_else(|_| panic!("{mode}: the session never finished"))
            .unwrap();

        assert_eq!(workflow.transcriptions(), expected, "{mode}");
        let mut expected_labels: Vec<String> = Vec::new();
        for _ in &expected {
            expected_labels.extend(["turn_started", "audio", "turn_ended"].map(String::from));
        }
        expected_labels.push("session_ended".to_string());
        assert_eq!(labels, expected_labels, "{mode}");
        server.finished().await;
    }
}

#[tokio::test]
async fn eof_drain_failure_closes_the_session() {
    trace_processor();
    // Python also parametrizes consumer cancellation, which is the asyncio half of this test.
    for outcome in [
        "timeout",
        "transcription_error",
        "unrelated_error",
        "disconnect",
    ] {
        let draining = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut steps = setup();
        steps.extend([
            Step::Recv("input_audio_buffer.append"),
            Step::Recv("input_audio_buffer.commit"),
            Step::Send(json!({"type": "input_audio_buffer.committed", "item_id": "first"})),
            Step::Recv("input_audio_buffer.clear"),
            Step::Send(json!({"type": "input_audio_buffer.cleared"})),
            Step::Signal(draining.clone()),
            Step::Wait(release.clone()),
        ]);
        match outcome {
            "transcription_error" => steps.push(Step::Send(json!({
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "first",
                "error": {"message": "Synthetic failure"},
            }))),
            "unrelated_error" => steps.push(Step::Send(json!({
                "type": "error",
                "error": {"code": "input_audio_buffer_commit_empty", "event_id": "unrelated"},
            }))),
            "disconnect" => steps.push(Step::Close),
            _ => {}
        }
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;

        let mut client_options = VoiceWebSocketOptions::default();
        if outcome == "timeout" {
            client_options = client_options.with_event_inactivity_timeout(0.05);
        }
        let stt = OpenAISTTModel::new("gpt-4o-transcribe", server.client())
            .with_websocket_options(client_options);
        let workflow = Arc::new(FakeVoiceWorkflow::new());
        let pipeline = VoicePipeline::new(workflow.clone())
            .with_stt_model(Arc::new(stt))
            .with_tts_model(Arc::new(FakeTTSModel::new()))
            .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));
        let input = StreamedAudioInput::new();
        input.add_audio(vec![0i16; 4800]).unwrap();
        input.end().unwrap();
        let mut result = pipeline.run(input).await.unwrap();
        let consumer = tokio::spawn(async move { extract_events(&mut result).await.0 });

        tokio::time::timeout(Duration::from_secs(2), draining.notified())
            .await
            .expect("the server never reached the drain");
        release.notify_one();
        let labels = tokio::time::timeout(Duration::from_secs(5), consumer)
            .await
            .unwrap_or_else(|_| panic!("{outcome}: the session never finished"))
            .unwrap();

        assert!(
            labels.last().unwrap().starts_with("error: "),
            "{outcome}: {labels:?}"
        );
        assert!(workflow.transcriptions().is_empty(), "{outcome}");
        // The pipeline closed the transcription session, which closed the socket.
        server.finished().await;
    }
}

#[tokio::test]
async fn explicit_close_aborts_pending_eof_without_a_provider_error() {
    trace_processor();
    let draining = Arc::new(Notify::new());
    let mut steps = setup();
    steps.extend([
        Step::Recv("input_audio_buffer.append"),
        Step::Recv("input_audio_buffer.commit"),
        Step::Send(json!({"type": "input_audio_buffer.committed", "item_id": "pending"})),
        Step::Recv("input_audio_buffer.clear"),
        Step::Send(json!({"type": "input_audio_buffer.cleared"})),
        Step::Signal(draining.clone()),
        Step::WaitClosed,
    ]);
    let server = ws_server(steps).await;
    let input = StreamedAudioInput::new();
    input.add_audio(vec![0i16; 4800]).unwrap();
    input.end().unwrap();
    let mut session = OpenAISTTModel::new("gpt-4o-transcribe", server.client())
        .create_session(&input, &STTModelSettings::default(), false, false)
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), draining.notified())
        .await
        .expect("the server never reached the drain");
    tokio::time::timeout(Duration::from_secs(2), session.close())
        .await
        .expect("close must not hang")
        .unwrap();
    // The aborted drain is not a provider error: the session just ends.
    assert_eq!(session.next_turn().await.unwrap(), None);
    server.finished().await;
}

// ===== tests/voice/test_openai_stt.py =====

#[tokio::test]
async fn transcription_terminal_paths_after_setup() {
    trace_processor();
    for outcome in ["close", "server_close", "error"] {
        let finish = Arc::new(Notify::new());
        let mut steps = setup();
        steps.extend([
            Step::Recv("input_audio_buffer.append"),
            Step::Send(json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hello",
            })),
            Step::Wait(finish.clone()),
        ]);
        match outcome {
            "error" => steps.push(Step::Send(
                json!({"type": "error", "error": "test provider error"}),
            )),
            "server_close" => steps.push(Step::Close),
            _ => {}
        }
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;

        let input = StreamedAudioInput::new();
        input.add_audio(vec![1, 2]).unwrap();
        let mut session = OpenAISTTModel::new("gpt-4o-mini-transcribe", server.client())
            .create_session(&input, &STTModelSettings::default(), false, false)
            .await
            .unwrap();
        assert_eq!(session.next_turn().await.unwrap().as_deref(), Some("hello"));
        finish.notify_one();

        match outcome {
            "close" => {
                session.close().await.unwrap();
                assert_eq!(session.next_turn().await.unwrap(), None);
            }
            "server_close" => {
                assert_eq!(session.next_turn().await.unwrap(), None);
                session.close().await.unwrap();
            }
            _ => {
                let error = session.next_turn().await.unwrap_err();
                assert!(error.to_string().contains("test provider error"), "{error}");
                session.close().await.unwrap();
            }
        }
        server.finished().await;
    }
}

#[tokio::test]
async fn wait_for_event_returns_matching_event() {
    trace_processor();
    // Events that are not the expected one are skipped until the expected one arrives.
    let mut steps = vec![
        Step::Send(json!({"type": "noise"})),
        Step::Send(json!({"type": "session.created"})),
        Step::Recv("session.update"),
        Step::Send(json!({"type": "session.updated"})),
    ];
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let client = server.client();
    let server = open_and_close(server, client, "whisper-1", &STTModelSettings::default()).await;
    server.finished().await;
}

#[tokio::test]
async fn wait_for_event_uses_one_deadline_across_unrelated_events() {
    trace_processor();
    // Unrelated events keep arriving, so a per-event timeout would never fire; the deadline is
    // shared across the wait.
    let mut steps: Vec<Step> = Vec::new();
    for _ in 0..8 {
        steps.push(Step::Send(json!({"type": "unrelated"})));
        steps.push(Step::Sleep(40));
    }
    steps.push(Step::Send(json!({"type": "session.created"})));
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let stt = OpenAISTTModel::new("whisper-1", server.client()).with_websocket_options(
        VoiceWebSocketOptions::default().with_session_creation_timeout(0.1),
    );
    let started = std::time::Instant::now();
    let error = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Timeout waiting for transcription_session.created event"),
        "{error}"
    );
    assert!(started.elapsed() < Duration::from_millis(300));
}

#[tokio::test]
async fn transcribe_error_respects_sensitive_data_setting() {
    let processor = trace_processor();
    for (include_sensitive, expected) in [
        (false, REDACTED_TRACE_ERROR_MESSAGE.to_string()),
        (true, "sensitive-stt-error".to_string()),
    ] {
        let (base, _) = mock_server(vec![Reply::Status(
            400,
            json!({"error": {"message": "sensitive-stt-error"}}),
        )])
        .await;
        let model = OpenAISTTModel::new(
            "whisper-1",
            OpenAIClient::new("sk-test").with_base_url(base),
        );
        let caller = trace("stt-error").build();
        let trace_id = caller.trace_id().to_string();
        let error = caller
            .run(async {
                model
                    .transcribe(
                        &AudioInput::new(vec![0i16; 2]),
                        &STTModelSettings::default(),
                        include_sensitive,
                        false,
                    )
                    .await
                    .unwrap_err()
            })
            .await;
        assert!(error.to_string().contains("sensitive-stt-error"), "{error}");

        let spans = spans_of(&processor, &trace_id);
        let transcription = spans_of_type(&spans, "transcription");
        assert_eq!(transcription.len(), 1, "{spans:#?}");
        let recorded = &transcription[0]["error"];
        let message = recorded["message"].as_str().unwrap();
        if include_sensitive {
            assert!(message.contains(&expected), "{message}");
        } else {
            assert_eq!(message, expected);
        }
        assert_eq!(recorded["data"], json!({}));
    }
}

#[tokio::test]
async fn transcribe_redacts_prompt_without_changing_request() {
    let processor = trace_processor();
    let (base, captured) = mock_server(vec![Reply::Json(json!({"text": "transcript"}))]).await;
    let model = OpenAISTTModel::new(
        "whisper-1",
        OpenAIClient::new("sk-test").with_base_url(base),
    );
    let caller = trace("stt-prompt").build();
    let trace_id = caller.trace_id().to_string();
    let result = caller
        .run(async {
            model
                .transcribe(
                    &AudioInput::new(vec![0i16; 2]),
                    &STTModelSettings::default().with_prompt("customer account vocabulary"),
                    false,
                    false,
                )
                .await
                .unwrap()
        })
        .await;
    assert_eq!(result, "transcript");

    let spans = spans_of(&processor, &trace_id);
    let transcription = spans_of_type(&spans, "transcription");
    assert!(transcription[0]["span_data"]["model_config"]["prompt"].is_null());
    // The request itself still carries the prompt.
    let request = captured.lock().unwrap()[0].clone();
    assert_eq!(
        request.field("prompt").as_deref(),
        Some("customer account vocabulary")
    );
}

#[tokio::test]
async fn non_json_messages_should_crash() {
    trace_processor();
    // Before the handshake events.
    let server = ws_server(vec![Step::SendText("not a json message"), Step::WaitClosed]).await;
    let error = OpenAISTTModel::new("whisper-1", server.client())
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .expect_err("a non-JSON frame must fail the session");
    assert!(
        error.to_string().contains("Error parsing events"),
        "{error}"
    );
    drop(server);

    // After the handshake, while streaming.
    let mut steps = setup();
    steps.push(Step::SendText("not a json message"));
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let mut session = OpenAISTTModel::new("whisper-1", server.client())
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();
    let error = session.next_turn().await.unwrap_err();
    assert!(
        error.to_string().contains("Error parsing events"),
        "{error}"
    );
    session.close().await.unwrap();
    server.finished().await;
}

#[tokio::test]
async fn session_connects_and_configures_successfully() {
    trace_processor();
    for session_header in [None, Some("0"), Some("1"), Some("omitted")] {
        let mut steps = setup_with(
            "transcription_session.created",
            "transcription_session.updated",
        );
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;
        let mut client = OpenAIClient::new("FAKE_KEY").with_base_url(server.base.clone());
        match session_header {
            Some("omitted") => client = client.without_header("openai-log-session"),
            Some(value) => client = client.with_header("openai-log-session", value),
            None => {}
        }
        let server =
            open_and_close(server, client, "whisper-1", &STTModelSettings::default()).await;

        let handshake = server.handshake();
        assert_eq!(handshake.target, "/v1/realtime?intent=transcription");
        assert_eq!(handshake.header("authorization"), Some("Bearer FAKE_KEY"));
        assert!(handshake.header("openai-beta").is_none());
        let expected = match session_header {
            Some("omitted") | None => None,
            Some(value) => Some(value),
        };
        assert_eq!(handshake.header("openai-log-session"), expected);
        assert_eq!(server.of_type("session.update").len(), 1);
        server.finished().await;
    }
}

#[tokio::test]
async fn stream_audio_sends_pcm16() {
    trace_processor();
    let expected = pcm_bytes(&[-32767, -32767, -16383, 0, 16383, 32767, 32767]);
    for float in [false, true] {
        let mut steps = setup();
        steps.push(Step::Recv("input_audio_buffer.append"));
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;
        let input = StreamedAudioInput::new();
        let expected_bytes = if float {
            input
                .add_audio_f32(&[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
                .unwrap();
            expected.clone()
        } else {
            input.add_audio(vec![1, 2, 3, 4]).unwrap();
            pcm_bytes(&[1, 2, 3, 4])
        };
        let mut session = OpenAISTTModel::new("whisper-1", server.client())
            .create_session(&input, &STTModelSettings::default(), false, false)
            .await
            .unwrap();
        // The append is delivered once the server has read it.
        tokio::time::timeout(Duration::from_secs(2), async {
            while server.of_type("input_audio_buffer.append").is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("no append reached the server");
        session.close().await.unwrap();

        let appends = server.of_type("input_audio_buffer.append");
        assert_eq!(
            appends.len(),
            1,
            "No 'input_audio_buffer.append' message was sent."
        );
        let audio = BASE64
            .decode(appends[0]["audio"].as_str().unwrap())
            .unwrap();
        assert_eq!(audio, expected_bytes);
        server.finished().await;
    }
}

#[tokio::test]
async fn transcription_event_puts_output_in_queue() {
    trace_processor();
    for (created, updated, completed_event) in [
        (
            "transcription_session.created",
            "transcription_session.updated",
            json!({"type": "input_audio_transcription_completed", "transcript": "Hello world!"}),
        ),
        (
            "session.created",
            "session.updated",
            json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hello world!",
            }),
        ),
    ] {
        let mut steps = setup_with(created, updated);
        steps.push(Step::Send(completed_event));
        steps.push(Step::WaitClosed);
        let server = ws_server(steps).await;
        let mut session = OpenAISTTModel::new("whisper-1", server.client())
            .create_session(
                &StreamedAudioInput::new(),
                &STTModelSettings::default(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            session.next_turn().await.unwrap().as_deref(),
            Some("Hello world!")
        );
        session.close().await.unwrap();
        server.finished().await;
    }
}

#[tokio::test]
async fn timeout_waiting_for_created_event() {
    trace_processor();
    // An unrelated event arrives, but never the created event.
    let server = ws_server(vec![
        Step::Send(json!({"type": "unknown"})),
        Step::WaitClosed,
    ])
    .await;
    let stt = OpenAISTTModel::new("whisper-1", server.client()).with_websocket_options(
        VoiceWebSocketOptions::default().with_session_creation_timeout(0.05),
    );
    let error = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .expect_err("the session creation must time out");
    assert!(
        error
            .to_string()
            .contains("Timeout waiting for transcription_session.created event"),
        "{error}"
    );
}

#[tokio::test]
async fn real_clock_session_creation_timeout_is_wrapped() {
    trace_processor();
    let server = ws_server(vec![Step::WaitClosed]).await;
    let stt = OpenAISTTModel::new("whisper-1", server.client()).with_websocket_options(
        VoiceWebSocketOptions::default().with_session_creation_timeout(0.01),
    );
    let error = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .expect_err("the session creation must time out");
    assert!(
        error
            .to_string()
            .contains("Timeout waiting for transcription_session.created event"),
        "{error}"
    );
}

#[tokio::test]
async fn session_error_event() {
    trace_processor();
    let mut steps = setup_with(
        "transcription_session.created",
        "transcription_session.updated",
    );
    steps.push(Step::Send(json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "Transcript before error",
    })));
    steps.push(Step::Send(
        json!({"type": "error", "error": "Simulated server error!"}),
    ));
    steps.push(Step::WaitClosed);
    let server = ws_server(steps).await;
    let stt = OpenAISTTModel::new("whisper-1", server.client()).with_websocket_options(
        VoiceWebSocketOptions::default().with_event_inactivity_timeout(0.1),
    );
    let mut session = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();

    // The transcript that preceded the error is delivered, then the error surfaces.
    assert_eq!(
        session.next_turn().await.unwrap().as_deref(),
        Some("Transcript before error")
    );
    let error = session.next_turn().await.unwrap_err();
    assert!(
        error.to_string().contains("Simulated server error!"),
        "{error}"
    );
    session.close().await.unwrap();
    server.finished().await;
}

#[tokio::test]
async fn session_error_event_before_session_created() {
    trace_processor();
    let server = ws_server(vec![
        Step::Send(json!({"type": "error", "error": "Simulated setup error!"})),
        Step::WaitClosed,
    ])
    .await;
    let stt = OpenAISTTModel::new("whisper-1", server.client());
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        stt.create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        ),
    )
    .await
    .expect("the setup error must not hang")
    .expect_err("a setup error must fail the session");
    assert!(
        error.to_string().contains("Simulated setup error!"),
        "{error}"
    );
}

#[tokio::test]
async fn inactivity_timeout() {
    trace_processor();
    let server = ws_server(vec![
        Step::Send(json!({"type": "transcription_session.created"})),
        Step::Recv("session.update"),
        Step::Send(json!({"type": "transcription_session.updated"})),
        Step::WaitClosed,
    ])
    .await;
    let stt = OpenAISTTModel::new("whisper-1", server.client()).with_websocket_options(
        VoiceWebSocketOptions::default().with_event_inactivity_timeout(0.01),
    );
    let mut session = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();
    // No events arrive: the session ends cleanly instead of failing.
    let turn = tokio::time::timeout(Duration::from_secs(1), session.next_turn())
        .await
        .expect("the inactivity timeout must end the session");
    assert_eq!(turn.unwrap(), None);
    session.close().await.unwrap();
    server.finished().await;
}

#[tokio::test]
async fn stream_audio_buffers_turn_audio_only_for_audio_tracing() {
    let processor = trace_processor();
    for include_audio in [false, true] {
        let received = Arc::new(Notify::new());
        let mut steps = setup();
        steps.extend([
            Step::Recv("input_audio_buffer.append"),
            Step::Recv("input_audio_buffer.append"),
            Step::Signal(received.clone()),
            Step::WaitClosed,
        ]);
        let server = ws_server(steps).await;
        let input = StreamedAudioInput::new();
        input.add_audio(vec![0, 0]).unwrap();
        input.add_audio(vec![1, 1]).unwrap();
        let stt = OpenAISTTModel::new("whisper-1", server.client());
        let caller = trace("stt audio buffer").build();
        let trace_id = caller.trace_id().to_string();
        caller
            .run(async {
                let mut session = stt
                    .create_session(&input, &STTModelSettings::default(), false, include_audio)
                    .await
                    .unwrap();
                tokio::time::timeout(Duration::from_secs(2), received.notified())
                    .await
                    .expect("the frames never reached the server");
                session.close().await.unwrap();
            })
            .await;

        // Every frame reaches the websocket regardless of the tracing setting.
        assert_eq!(server.of_type("input_audio_buffer.append").len(), 2);
        let spans = spans_of(&processor, &trace_id);
        let transcription = spans_of_type(&spans, "transcription");
        let recorded = transcription[0]["span_data"]["input"]["data"].clone();
        let encoded = BASE64.encode(pcm_bytes(&[0, 0, 1, 1]));
        if include_audio {
            assert_eq!(recorded, json!(encoded));
        } else {
            assert!(
                recorded.is_null() || recorded == json!(""),
                "audio must not be buffered: {recorded}"
            );
        }
        server.finished().await;
    }
}
