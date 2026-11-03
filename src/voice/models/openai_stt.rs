//! The OpenAI speech-to-text model. Mirrors `agents.voice.models.openai_stt`.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::{AgentsError, Result};
use crate::models::OpenAIClient;
use crate::tracing::scope::ScopeSnapshot;
use crate::tracing::span_data::SpanData;
use crate::tracing::spans::Span;
use crate::tracing::transcription_span;
use crate::util::gen_hex32;
use crate::voice::input::{AudioInput, StreamedAudioInput, pcm16_to_bytes};
use crate::voice::model::{STTModel, STTModelSettings, StreamedTranscriptionSession};
use crate::voice::result::get_trace_error;

/// How long to wait for a session event before assuming the session is done, in seconds.
pub const EVENT_INACTIVITY_TIMEOUT: f64 = 1000.0;
/// How long to wait for the `session.created` event, in seconds.
pub const SESSION_CREATION_TIMEOUT: f64 = 10.0;
/// How long to wait for the `session.updated` event, in seconds.
pub const SESSION_UPDATE_TIMEOUT: f64 = 10.0;

/// Keepalive and frame-size settings for the realtime transcription websocket.
///
/// Python's voice STT session calls `websockets.connect` without overriding them, so the
/// defaults here are that library's: a ping every 20 seconds, 20 seconds to answer it, and a
/// 1 MiB limit on an incoming message. `tokio_tungstenite` sends no pings of its own, so the
/// keepalive is driven from the event loop. The same three knobs are
/// `OpenAIResponsesWebSocketOptions` on the Responses transport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceWebSocketOptions {
    /// Seconds between keepalive pings. `None` disables the keepalive.
    pub ping_interval: Option<f64>,
    /// Seconds to wait for the pong answering a keepalive ping.
    pub ping_timeout: Option<f64>,
    /// Maximum size of an incoming message, in bytes. `None` means no limit.
    pub max_size: Option<usize>,
    /// Seconds to wait for the `session.created` event.
    pub session_creation_timeout: f64,
    /// Seconds to wait for the `session.updated` event.
    pub session_update_timeout: f64,
    /// Seconds without any session event after which the session is assumed done.
    pub event_inactivity_timeout: f64,
}

impl Default for VoiceWebSocketOptions {
    fn default() -> Self {
        VoiceWebSocketOptions {
            ping_interval: Some(20.0),
            ping_timeout: Some(20.0),
            max_size: Some(1 << 20),
            session_creation_timeout: SESSION_CREATION_TIMEOUT,
            session_update_timeout: SESSION_UPDATE_TIMEOUT,
            event_inactivity_timeout: EVENT_INACTIVITY_TIMEOUT,
        }
    }
}

impl VoiceWebSocketOptions {
    /// Seconds between keepalive pings (`None` disables them).
    pub fn with_ping_interval(mut self, seconds: Option<f64>) -> Self {
        self.ping_interval = seconds;
        self
    }

    /// Seconds to wait for the pong answering a keepalive ping.
    pub fn with_ping_timeout(mut self, seconds: Option<f64>) -> Self {
        self.ping_timeout = seconds;
        self
    }

    /// Maximum size of an incoming message, in bytes (`None` means no limit).
    pub fn with_max_size(mut self, bytes: Option<usize>) -> Self {
        self.max_size = bytes;
        self
    }

    /// Seconds to wait for the `session.created` event.
    pub fn with_session_creation_timeout(mut self, seconds: f64) -> Self {
        self.session_creation_timeout = seconds;
        self
    }

    /// Seconds to wait for the `session.updated` event.
    pub fn with_session_update_timeout(mut self, seconds: f64) -> Self {
        self.session_update_timeout = seconds;
        self
    }

    /// Seconds without any session event after which the session is assumed done.
    pub fn with_event_inactivity_timeout(mut self, seconds: f64) -> Self {
        self.event_inactivity_timeout = seconds;
        self
    }

    /// The websocket configuration these settings imply.
    fn websocket_config(&self) -> tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
        // Python's `max_size` caps a whole message, and `websockets` applies it per frame too.
        tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(self.max_size)
            .max_frame_size(self.max_size)
    }
}

/// Seconds as a [`Duration`], clamped at zero.
fn seconds(value: f64) -> Duration {
    Duration::try_from_secs_f64(value.max(0.0)).unwrap_or(Duration::ZERO)
}

/// Models that take a list of candidate languages instead of a single one.
const MULTI_LANGUAGE_MODELS: [&str; 2] = ["gpt-transcribe", "gpt-live-transcribe"];

/// The turn detection used when [`STTModelSettings::turn_detection`] is not set.
fn default_turn_detection() -> Value {
    json!({"type": "semantic_vad"})
}

/// A speech-to-text model served by the OpenAI `/audio/transcriptions` endpoint and the
/// realtime transcription websocket.
#[derive(Debug, Clone)]
pub struct OpenAISTTModel {
    model: String,
    client: OpenAIClient,
    http: reqwest::Client,
    websocket: VoiceWebSocketOptions,
    /// A provider configuration error every operation reports instead of running.
    config_error: Option<&'static str>,
}

impl OpenAISTTModel {
    /// A speech-to-text model using `client`.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAISTTModel {
            model: model.into(),
            client,
            http: crate::util::http_client(),
            websocket: VoiceWebSocketOptions::default(),
            config_error: None,
        }
    }

    /// Make every operation fail with `error` (a provider configuration error).
    pub(crate) fn with_config_error(mut self, error: Option<&'static str>) -> Self {
        self.config_error = error;
        self
    }

    fn check_config(&self) -> Result<()> {
        match self.config_error {
            Some(message) => Err(AgentsError::user(message)),
            None => Ok(()),
        }
    }

    /// Use a custom `reqwest::Client` for the transcription request.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Set the keepalive and frame-size settings of the streamed transcription websocket.
    pub fn with_websocket_options(mut self, options: VoiceWebSocketOptions) -> Self {
        self.websocket = options;
        self
    }

    /// The keepalive and frame-size settings of the streamed transcription websocket.
    pub fn websocket_options(&self) -> &VoiceWebSocketOptions {
        &self.websocket
    }
}

#[async_trait]
impl STTModel for OpenAISTTModel {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn transcribe(
        &self,
        input: &AudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<String> {
        self.check_config()?;
        let span = transcription_span(
            Some(self.model.clone()),
            Some(json!({
                "temperature": settings.temperature,
                "language": settings.language,
                "prompt": trace_include_sensitive_data
                    .then(|| settings.prompt.clone())
                    .flatten(),
            })),
            false,
        );
        span.update_data(|data| {
            if let SpanData::Transcription { input: field, .. } = data {
                *field = Some(if trace_include_sensitive_audio_data {
                    input.to_base64()
                } else {
                    String::new()
                });
            }
        });
        span.start();
        let result = self.post_transcription(input, settings).await;
        match &result {
            Ok(text) => {
                if trace_include_sensitive_data {
                    let text = text.clone();
                    span.update_data(|data| {
                        if let SpanData::Transcription { output, .. } = data {
                            *output = Some(text);
                        }
                    });
                }
            }
            Err(error) => {
                span.update_data(|data| {
                    if let SpanData::Transcription { output, .. } = data {
                        *output = Some(String::new());
                    }
                });
                span.set_error(
                    get_trace_error(trace_include_sensitive_data, error),
                    Some(json!({})),
                );
            }
        }
        span.finish();
        result
    }

    async fn create_session(
        &self,
        input: &StreamedAudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<Box<dyn StreamedTranscriptionSession>> {
        self.check_config()?;
        let audio = input.take_receiver().await.ok_or_else(|| {
            AgentsError::user("This StreamedAudioInput is already consumed by another session")
        })?;
        let session = OpenAISTTTranscriptionSession::connect(
            &self.client,
            &self.model,
            settings,
            audio,
            trace_include_sensitive_data,
            trace_include_sensitive_audio_data,
            self.websocket,
        )
        .await?;
        Ok(Box::new(session))
    }
}

impl OpenAISTTModel {
    /// POST the audio as `multipart/form-data` and return the transcribed text.
    async fn post_transcription(
        &self,
        input: &AudioInput,
        settings: &STTModelSettings,
    ) -> Result<String> {
        let (filename, bytes, content_type) = input.to_audio_file()?;
        let file = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename)
            .mime_str(&content_type)
            .map_err(|e| AgentsError::user(format!("Invalid audio content type: {e}")))?;
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", file);
        if let Some(prompt) = &settings.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(language) = &settings.language {
            // `gpt-transcribe` takes a repeated `languages[]` field instead.
            if self.model == MULTI_LANGUAGE_MODELS[0] {
                form = form.text("languages[]", language.clone());
            } else {
                form = form.text("language", language.clone());
            }
        }
        if let Some(temperature) = settings.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        // Goes through the shared client so the `RequestAuth` hook, the default query and
        // the OpenAI error mapping apply here as they do to every other request.
        let (body, _request_id) = self
            .client
            .post_multipart("/audio/transcriptions", form)
            .await?;
        body.get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                AgentsError::model_behavior("Transcription response did not contain `text`")
            })
    }
}

/// The websocket halves used by a transcription session.
type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsSink = Arc<AsyncMutex<SplitSink<WsStream, Message>>>;

/// Tracing state shared by the audio and event tasks of a session.
struct SessionTracing {
    span: Mutex<Option<Span>>,
    turn_audio: Mutex<Vec<i16>>,
    model: String,
    model_config: Value,
    include_sensitive_data: bool,
    include_sensitive_audio_data: bool,
}

impl SessionTracing {
    /// Open the span of a new turn.
    fn start_turn(&self) {
        let span = transcription_span(
            Some(self.model.clone()),
            Some(self.model_config.clone()),
            false,
        );
        span.start();
        *self.span.lock().expect("poisoned") = Some(span);
    }

    /// Close the span of the current turn, recording its audio and transcript.
    fn end_turn(&self, transcript: &str) {
        let Some(span) = self.span.lock().expect("poisoned").take() else {
            return;
        };
        let audio = std::mem::take(&mut *self.turn_audio.lock().expect("poisoned"));
        let input = (self.include_sensitive_audio_data && !audio.is_empty())
            .then(|| BASE64.encode(pcm16_to_bytes(&audio)));
        let output = self.include_sensitive_data.then(|| transcript.to_string());
        span.update_data(|data| {
            if let SpanData::Transcription {
                input: input_field,
                input_format,
                output: output_field,
                ..
            } = data
            {
                if let Some(input) = input {
                    *input_field = Some(input);
                }
                *input_format = Some("pcm".to_string());
                if let Some(output) = output {
                    *output_field = Some(output);
                }
            }
        });
        span.finish();
    }

    /// Keep a turn's audio so it can be attached to the span.
    fn record_audio(&self, buffer: &[i16]) {
        if self.include_sensitive_audio_data {
            self.turn_audio
                .lock()
                .expect("poisoned")
                .extend_from_slice(buffer);
        }
    }
}

/// A transcription session over the OpenAI realtime transcription websocket.
pub struct OpenAISTTTranscriptionSession {
    turns: mpsc::UnboundedReceiver<Result<String>>,
    sink: WsSink,
    tracing: Arc<SessionTracing>,
    closing: Arc<AtomicBool>,
    tasks: Vec<JoinHandle<()>>,
    closed: bool,
}

impl std::fmt::Debug for OpenAISTTTranscriptionSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAISTTTranscriptionSession")
            .field("model", &self.tracing.model)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for SessionTracing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionTracing")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl OpenAISTTTranscriptionSession {
    /// Connect, configure the transcription session and start streaming audio.
    pub(crate) async fn connect(
        client: &OpenAIClient,
        model: &str,
        settings: &STTModelSettings,
        audio: mpsc::UnboundedReceiver<Option<Vec<i16>>>,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
        options: VoiceWebSocketOptions,
    ) -> Result<Self> {
        let turn_detection = settings
            .turn_detection
            .clone()
            .unwrap_or_else(default_turn_detection);
        let transcription_config = transcription_config(model, settings);

        let url = websocket_url(client);
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|e| AgentsError::user(format!("Invalid transcription websocket URL: {e}")))?;
        // The client's `RequestAuth` hook runs here, so a rotating API key is refreshed
        // before every handshake.
        request
            .headers_mut()
            .extend(client.websocket_headers(&url).await?);
        crate::util::ensure_crypto_provider();
        let (ws, _) = tokio_tungstenite::connect_async_with_config(
            request,
            Some(options.websocket_config()),
            false,
        )
        .await
        .map_err(|e| websocket_error(format!("Could not open the STT websocket: {e}")))?;
        let (sink, mut stream) = ws.split();
        let mut backlog: VecDeque<Message> = VecDeque::new();

        wait_for_event(
            &mut stream,
            &["session.created", "transcription_session.created"],
            seconds(options.session_creation_timeout),
            "Timeout waiting for transcription_session.created event",
            &mut backlog,
        )
        .await?;

        let sink = Arc::new(AsyncMutex::new(sink));
        send_message(
            &sink,
            json!({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": transcription_config,
                            "turn_detection": turn_detection,
                        }
                    },
                },
            }),
        )
        .await?;

        wait_for_event(
            &mut stream,
            &["session.updated", "transcription_session.updated"],
            seconds(options.session_update_timeout),
            "Timeout waiting for transcription_session.updated event",
            &mut backlog,
        )
        .await?;

        let tracing = Arc::new(SessionTracing {
            span: Mutex::new(None),
            turn_audio: Mutex::new(Vec::new()),
            model: model.to_string(),
            model_config: json!({
                "temperature": settings.temperature,
                "language": transcription_config.get("language"),
                "languages": transcription_config.get("languages"),
                "keywords": trace_include_sensitive_data
                    .then(|| transcription_config.get("keywords").cloned())
                    .flatten(),
                "prompt": trace_include_sensitive_data
                    .then(|| transcription_config.get("prompt").cloned())
                    .flatten(),
                "turn_detection": turn_detection,
            }),
            include_sensitive_data: trace_include_sensitive_data,
            include_sensitive_audio_data: trace_include_sensitive_audio_data,
        });
        // Open the first turn's span now, so a session closed right after setup still records it.
        tracing.start_turn();
        let (turns_tx, turns) = mpsc::unbounded_channel();
        let closing = Arc::new(AtomicBool::new(false));
        let final_commit_id = Arc::new(Mutex::new(None));
        let scope = ScopeSnapshot::capture();

        let audio_task = {
            let sink = sink.clone();
            let tracing = tracing.clone();
            let final_commit_id = final_commit_id.clone();
            let scope = scope.clone();
            tokio::spawn(async move {
                scope
                    .enter(stream_audio(audio, sink, tracing, final_commit_id))
                    .await;
            })
        };
        let event_task = {
            let tracing = tracing.clone();
            let closing = closing.clone();
            let sink = sink.clone();
            tokio::spawn(async move {
                scope
                    .enter(handle_events(
                        stream,
                        sink,
                        options,
                        tracing,
                        turns_tx,
                        final_commit_id,
                        closing,
                        backlog,
                    ))
                    .await;
            })
        };

        Ok(OpenAISTTTranscriptionSession {
            turns,
            sink,
            tracing,
            closing,
            tasks: vec![audio_task, event_task],
            closed: false,
        })
    }
}

impl Drop for OpenAISTTTranscriptionSession {
    fn drop(&mut self) {
        // Dropping a JoinHandle detaches its task, which would keep the socket open.
        for task in &self.tasks {
            task.abort();
        }
        self.tracing.end_turn("");
    }
}

#[async_trait]
impl StreamedTranscriptionSession for OpenAISTTTranscriptionSession {
    async fn next_turn(&mut self) -> Result<Option<String>> {
        match self.turns.recv().await {
            Some(Ok(turn)) => Ok(Some(turn)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.closing.store(true, Ordering::SeqCst);
        let _ = self.sink.lock().await.close().await;
        for task in std::mem::take(&mut self.tasks) {
            task.abort();
            let _ = task.await;
        }
        self.turns.close();
        self.tracing.end_turn("");
        Ok(())
    }
}

/// The `transcription` block of the session configuration.
fn transcription_config(model: &str, settings: &STTModelSettings) -> Value {
    let mut config = json!({ "model": model });
    if let Some(languages) = &settings.languages {
        config["languages"] = json!(languages);
    } else if let Some(language) = &settings.language {
        if MULTI_LANGUAGE_MODELS.contains(&model) {
            config["languages"] = json!([language]);
        } else {
            config["language"] = json!(language);
        }
    }
    if let Some(prompt) = &settings.prompt {
        config["prompt"] = json!(prompt);
    }
    if let Some(keywords) = &settings.keywords {
        config["keywords"] = json!(keywords);
    }
    config
}

/// The realtime transcription websocket URL for this client.
///
/// Port of Python's `_prepare_websocket_url`: the client's websocket base URL (or its HTTP
/// base URL with the scheme swapped), the query already on that URL, and the client's
/// `default_query`, with `intent=transcription` forced last.
fn websocket_url(client: &OpenAIClient) -> String {
    let configured = client.websocket_base_url();
    let configured = match configured.split_once("://") {
        Some(("https", rest)) => format!("wss://{rest}"),
        Some(("http", rest)) => format!("ws://{rest}"),
        _ => configured,
    };
    let (base, base_query) = configured
        .split_once('?')
        .unwrap_or((configured.as_str(), ""));
    let base = base.trim_end_matches('/');

    // Values already on the URL stay as given; the client's default query overrides them.
    let mut params: Vec<(String, String)> = base_query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            (name.to_string(), value.to_string())
        })
        .filter(|(name, _)| name != "intent")
        .collect();
    let mut defaults: Vec<(&String, &String)> = client
        .default_query()
        .iter()
        .filter(|(name, _)| name.as_str() != "intent")
        .collect();
    defaults.sort_unstable();
    for (name, value) in defaults {
        params.retain(|(existing, _)| existing != name);
        params.push((name.clone(), url_encode(value)));
    }
    params.push(("intent".to_string(), "transcription".to_string()));
    let query = params
        .iter()
        .map(|(name, value)| format!("{}={}", url_encode(name), value))
        .collect::<Vec<_>>()
        .join("&");
    format!("{base}/realtime?{query}")
}

/// Percent-encode one query component.
fn url_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// An [`AgentsError`] for a failed STT websocket, Python's `STTWebsocketConnectionError`.
pub(crate) fn websocket_error(message: impl Into<String>) -> AgentsError {
    AgentsError::other(message.into())
}

async fn send_message(sink: &WsSink, payload: Value) -> Result<()> {
    sink.lock()
        .await
        .send(Message::text(payload.to_string()))
        .await
        .map_err(|e| websocket_error(format!("Could not send to the STT websocket: {e}")))
}

/// Read events until one of `expected` arrives, failing on `error` events and timeouts.
async fn wait_for_event(
    stream: &mut SplitStream<WsStream>,
    expected: &[&str],
    timeout: Duration,
    timeout_message: &str,
    backlog: &mut VecDeque<Message>,
) -> Result<Value> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let message = tokio::time::timeout_at(deadline, stream.next())
            .await
            .map_err(|_| websocket_error(timeout_message.to_string()))?;
        let Some(message) = message else {
            return Err(websocket_error(
                "The STT websocket closed during session setup",
            ));
        };
        let message =
            message.map_err(|e| websocket_error(format!("STT websocket read failed: {e}")))?;
        let Some(event) = parse_event(&message)? else {
            continue;
        };
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if event_type == "error" {
            return Err(websocket_error(format!(
                "Error event: {}",
                event.get("error").unwrap_or(&Value::Null)
            )));
        }
        // Every event reaches the turn handler once setup is done, as Python's event queue
        // holds them: a transcript that arrives during setup is not lost.
        backlog.push_back(message);
        if expected.contains(&event_type) {
            return Ok(event);
        }
    }
}

/// A websocket message decoded as a session event, if it carries one.
///
/// A frame that is not valid JSON is an error, as in Python's `json.loads` in the listener.
fn parse_event(message: &Message) -> Result<Option<Value>> {
    let parsed = match message {
        Message::Text(text) => serde_json::from_str(text.as_str()),
        Message::Binary(bytes) => serde_json::from_slice(bytes),
        _ => return Ok(None),
    };
    parsed
        .map(Some)
        .map_err(|e| websocket_error(format!("Error parsing events: {e}")))
}

/// Forward queued audio to the websocket, committing the buffer at the end of the stream.
async fn stream_audio(
    mut audio: mpsc::UnboundedReceiver<Option<Vec<i16>>>,
    sink: WsSink,
    tracing: Arc<SessionTracing>,
    final_commit_id: Arc<Mutex<Option<String>>>,
) {
    while let Some(item) = audio.recv().await {
        let Some(buffer) = item else {
            let commit_id = gen_hex32();
            *final_commit_id.lock().expect("poisoned") = Some(commit_id.clone());
            let _ = send_message(
                &sink,
                json!({"type": "input_audio_buffer.commit", "event_id": commit_id}),
            )
            .await;
            // The clear acknowledgement follows the final commit, even when turn detection
            // wins the commit race. It ends input without cancelling committed transcripts.
            let _ = send_message(&sink, json!({"type": "input_audio_buffer.clear"})).await;
            return;
        };
        tracing.record_audio(&buffer);
        let payload = json!({
            "type": "input_audio_buffer.append",
            "audio": BASE64.encode(pcm16_to_bytes(&buffer)),
        });
        if send_message(&sink, payload).await.is_err() {
            return;
        }
    }
}

/// What the event loop got while waiting for the next session event.
enum NextEvent {
    /// A frame the event loop should interpret.
    Message(Message),
    /// The peer closed the socket.
    Closed,
    /// Reading failed, or the peer stopped answering the keepalive.
    Failed(String),
    /// No event arrived within [`EVENT_INACTIVITY_TIMEOUT`].
    Inactive,
}

/// Await the next session event, sending keepalive pings while nothing arrives.
///
/// `websockets` runs its keepalive on a background task; tungstenite has none, so the ping is
/// driven from this loop. Pongs do not extend the inactivity deadline, so a silent but
/// answering peer still ends the session on time.
async fn next_event(
    stream: &mut SplitStream<WsStream>,
    sink: &WsSink,
    options: &VoiceWebSocketOptions,
) -> NextEvent {
    let deadline = tokio::time::Instant::now() + seconds(options.event_inactivity_timeout);
    let mut awaiting_pong = false;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return NextEvent::Inactive;
        }
        let keepalive = if awaiting_pong {
            options.ping_timeout
        } else {
            options.ping_interval
        }
        .filter(|_| options.ping_interval.is_some())
        .map(seconds);
        let wait = keepalive.map_or(remaining, |k| remaining.min(k));
        match tokio::time::timeout(wait, stream.next()).await {
            Err(_) => {
                if tokio::time::Instant::now() >= deadline {
                    return NextEvent::Inactive;
                }
                if awaiting_pong {
                    return NextEvent::Failed(
                        "Timeout waiting for the STT websocket keepalive pong".to_string(),
                    );
                }
                if options.ping_interval.is_some() {
                    if sink
                        .lock()
                        .await
                        .send(Message::Ping(bytes::Bytes::new()))
                        .await
                        .is_err()
                    {
                        return NextEvent::Failed(
                            "Could not send the STT websocket keepalive ping".to_string(),
                        );
                    }
                    awaiting_pong = true;
                }
            }
            Ok(None) => return NextEvent::Closed,
            Ok(Some(Err(error))) => {
                return NextEvent::Failed(format!("STT websocket read failed: {error}"));
            }
            Ok(Some(Ok(Message::Close(_)))) => return NextEvent::Closed,
            Ok(Some(Ok(Message::Pong(_)))) => awaiting_pong = false,
            Ok(Some(Ok(message))) => return NextEvent::Message(message),
        }
    }
}

/// Turn websocket events into transcription turns.
#[allow(clippy::too_many_arguments)]
async fn handle_events(
    mut stream: SplitStream<WsStream>,
    sink: WsSink,
    options: VoiceWebSocketOptions,
    tracing: Arc<SessionTracing>,
    turns: mpsc::UnboundedSender<Result<String>>,
    final_commit_id: Arc<Mutex<Option<String>>>,
    closing: Arc<AtomicBool>,
    mut backlog: VecDeque<Message>,
) {
    let mut pending_transcripts: HashSet<String> = HashSet::new();
    let mut input_cleared = false;
    loop {
        let message = if let Some(buffered) = backlog.pop_front() {
            buffered
        } else {
            match next_event(&mut stream, &sink, &options).await {
                NextEvent::Inactive => {
                    if final_commit_id.lock().expect("poisoned").is_some() {
                        let _ = turns.send(Err(websocket_error(
                            "Timeout waiting for final transcription",
                        )));
                    }
                    // No new events for a while: assume the session is done.
                    break;
                }
                NextEvent::Closed => {
                    let committed = final_commit_id.lock().expect("poisoned").is_some();
                    if !closing.load(Ordering::SeqCst)
                        && committed
                        && (!input_cleared || !pending_transcripts.is_empty())
                    {
                        let _ = turns.send(Err(websocket_error(
                            "Websocket closed before final transcription completed",
                        )));
                    }
                    break;
                }
                NextEvent::Failed(message) => {
                    let _ = turns.send(Err(websocket_error(message)));
                    break;
                }
                NextEvent::Message(message) => message,
            }
        };
        if matches!(message, Message::Close(_)) {
            break;
        }
        let event = match parse_event(&message) {
            Ok(Some(event)) => event,
            Ok(None) => continue,
            Err(error) => {
                let _ = turns.send(Err(error));
                break;
            }
        };
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();

        if event_type == "error" {
            let error = event.get("error").cloned().unwrap_or(Value::Null);
            let is_expected_commit_noop = final_commit_id
                .lock()
                .expect("poisoned")
                .as_deref()
                .is_some_and(|id| {
                    error.get("event_id").and_then(Value::as_str) == Some(id)
                        && error.get("code").and_then(Value::as_str)
                            == Some("input_audio_buffer_commit_empty")
                });
            if is_expected_commit_noop {
                // Turn detection may already have committed the final buffer.
                continue;
            }
            let _ = turns.send(Err(websocket_error(format!("Error event: {error}"))));
            break;
        }

        match event_type {
            "input_audio_buffer.committed" => {
                if let Some(item_id) = event.get("item_id").and_then(Value::as_str) {
                    pending_transcripts.insert(item_id.to_string());
                }
            }
            "input_audio_buffer.cleared" => input_cleared = true,
            "conversation.item.input_audio_transcription.failed" => {
                let _ = turns.send(Err(websocket_error("Input audio transcription failed")));
                break;
            }
            _ => {}
        }

        if matches!(
            event_type,
            "input_audio_transcription_completed"
                | "conversation.item.input_audio_transcription.completed"
        ) {
            match event.get("item_id").and_then(Value::as_str) {
                Some(item_id) => {
                    pending_transcripts.remove(item_id);
                }
                None => {
                    // Legacy sessions do not identify completed items; each completion still
                    // settles one outstanding transcript.
                    if let Some(any) = pending_transcripts.iter().next().cloned() {
                        pending_transcripts.remove(&any);
                    }
                }
            }
            let transcript = event
                .get("transcript")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !transcript.is_empty() {
                tracing.end_turn(transcript);
                tracing.start_turn();
                if turns.send(Ok(transcript.to_string())).is_err() {
                    break;
                }
            }
        }

        if final_commit_id.lock().expect("poisoned").is_some()
            && input_cleared
            && pending_transcripts.is_empty()
        {
            break;
        }
    }
    tracing.end_turn("");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_realtime_transcription_url() {
        let client = OpenAIClient::new("sk-test").with_base_url("https://api.openai.com/v1");
        assert_eq!(
            websocket_url(&client),
            "wss://api.openai.com/v1/realtime?intent=transcription"
        );
        let local = OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:1234/v1");
        assert_eq!(
            websocket_url(&local),
            "ws://127.0.0.1:1234/v1/realtime?intent=transcription"
        );
    }

    #[test]
    fn the_handshake_url_carries_the_client_default_query() {
        let client = OpenAIClient::new("sk-test")
            .with_base_url("https://example.test/v1")
            .with_default_query("api-version", "2024 preview")
            .with_default_query("intent", "ignored");
        assert_eq!(
            websocket_url(&client),
            "wss://example.test/v1/realtime?api-version=2024%20preview&intent=transcription"
        );
    }

    #[test]
    fn multi_language_models_take_a_language_list() {
        let settings = STTModelSettings::default().with_language("en");
        assert_eq!(
            transcription_config("gpt-transcribe", &settings)["languages"],
            json!(["en"])
        );
        assert_eq!(
            transcription_config("gpt-4o-mini-transcribe", &settings)["language"],
            json!("en")
        );
    }

    #[test]
    fn explicit_languages_win_over_a_single_language() {
        let settings = STTModelSettings::default()
            .with_language("en")
            .with_languages(vec!["fr".to_string(), "de".to_string()])
            .with_keywords(vec!["Hue".to_string()])
            .with_prompt("be precise");
        let config = transcription_config("gpt-transcribe", &settings);
        assert_eq!(config["languages"], json!(["fr", "de"]));
        assert!(config.get("language").is_none());
        assert_eq!(config["keywords"], json!(["Hue"]));
        assert_eq!(config["prompt"], json!("be precise"));
    }
}
