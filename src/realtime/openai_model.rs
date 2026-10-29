//! The OpenAI realtime WebSocket transport. Mirrors `agents.realtime.openai_realtime`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::SinkExt;
use futures::stream::{SplitSink, StreamExt};
use serde_json::{Map, Value, json};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::{AgentsError, Result};
use crate::realtime::config::{
    DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL, DEFAULT_REALTIME_MODEL, DEFAULT_REALTIME_VOICE,
    RealtimeModelTracing, RealtimeSessionModelSettings, RealtimeTurnDetectionConfig,
};
use crate::realtime::items::{
    RealtimeContent, RealtimeItem, RealtimeMessageItem, RealtimeMessageRole, RealtimeToolCallItem,
};
use crate::realtime::model::{
    RealtimeModel, RealtimeModelConfig, RealtimeModelEvents, calculate_audio_length_ms,
};
use crate::realtime::model_events::{
    RealtimeConnectionStatus, RealtimeModelAudioEvent, RealtimeModelEvent,
    RealtimeModelToolCallEvent,
};
use crate::realtime::model_inputs::RealtimeModelSendEvent;
use crate::usage::Usage;

/// The default realtime WebSocket endpoint.
const DEFAULT_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

/// What the transport knows about the audio it has received.
#[derive(Debug, Default)]
struct AudioTracker {
    format: Option<String>,
    last_item: Option<(String, usize)>,
    length_ms: f64,
    first_received: Option<Instant>,
}

impl AudioTracker {
    fn on_delta(&mut self, item_id: &str, content_index: usize, bytes: &[u8]) {
        let ms = calculate_audio_length_ms(self.format.as_deref(), bytes);
        let key = (item_id.to_string(), content_index);
        if self.last_item.as_ref() != Some(&key) {
            self.last_item = Some(key);
            self.length_ms = ms;
            self.first_received = Some(Instant::now());
        } else {
            self.length_ms += ms;
        }
    }

    fn on_interrupted(&mut self) {
        self.last_item = None;
        self.length_ms = 0.0;
        self.first_received = None;
    }

    /// The item being played and how much of it the client has had time to play.
    fn playback_state(&self) -> Option<PlaybackState> {
        let (item_id, content_index) = self.last_item.clone()?;
        let elapsed_ms = self.first_received?.elapsed().as_secs_f64() * 1000.0;
        Some(PlaybackState {
            item_id,
            content_index,
            elapsed_ms: elapsed_ms.clamp(0.0, self.length_ms),
            length_ms: self.length_ms,
        })
    }
}

/// How far the client has had time to play the audio it received.
#[derive(Debug, Clone, PartialEq)]
struct PlaybackState {
    item_id: String,
    content_index: usize,
    elapsed_ms: f64,
    length_ms: f64,
}

impl PlaybackState {
    /// Where to truncate the item, never past the audio the client actually received.
    fn truncate_ms(&self) -> i64 {
        let max_ms = self.length_ms.ceil();
        self.elapsed_ms.floor().clamp(0.0, max_ms) as i64
    }

    /// Whether the client has played everything the model sent.
    ///
    /// `elapsed_ms` is clamped to `length_ms` when the state is built, so equality means the
    /// client has had time to play every byte received.
    fn is_complete(&self) -> bool {
        self.elapsed_ms >= self.length_ms
    }
}

#[derive(Debug, Default)]
struct WsState {
    model: String,
    tracing: Option<RealtimeModelTracing>,
    call_id: Option<String>,
    audio: AudioTracker,
    ongoing_response: bool,
    /// A `response.create` was sent and neither `response.created` nor a matching error arrived.
    create_in_flight: Option<String>,
    /// A response was requested while one was running or in flight.
    pending_response_create: bool,
    create_counter: u64,
    automatic_response_cancellation: bool,
}

impl WsState {
    /// Request a response. Returns the `event_id` to send with it, or `None` to defer.
    ///
    /// The realtime API rejects a second `response.create` while a response is active, and
    /// parallel tool outputs would otherwise both see an idle connection after `response.done`.
    fn request_response_create(&mut self) -> Option<String> {
        if self.ongoing_response || self.create_in_flight.is_some() {
            self.pending_response_create = true;
            return None;
        }
        Some(self.begin_response_create())
    }

    fn begin_response_create(&mut self) -> String {
        self.create_counter += 1;
        let event_id = format!("agents_rs_response_create_{}", self.create_counter);
        self.create_in_flight = Some(event_id.clone());
        event_id
    }

    fn on_response_created(&mut self) {
        self.create_in_flight = None;
        self.ongoing_response = true;
    }

    /// Clear the active response and return the `event_id` of a deferred create to send now.
    fn on_response_done(&mut self) -> Option<String> {
        self.ongoing_response = false;
        if self.pending_response_create && self.create_in_flight.is_none() {
            self.pending_response_create = false;
            return Some(self.begin_response_create());
        }
        None
    }

    /// Release an in-flight create the server rejected, flushing any deferred one.
    fn on_response_create_failed(&mut self) -> Option<String> {
        self.create_in_flight = None;
        if self.pending_response_create && !self.ongoing_response {
            self.pending_response_create = false;
            return Some(self.begin_response_create());
        }
        None
    }

    /// Whether a server error is the rejection of the create this client has in flight.
    ///
    /// Python treats any error carrying an `event_id` as a match; this port requires the id to be
    /// the one it sent, so an unrelated failure (a barge-in truncate, say) cannot drop a deferred
    /// response.
    fn error_rejects_pending_create(&self, error: &Value) -> bool {
        let Some(in_flight) = self.create_in_flight.as_deref() else {
            return false;
        };
        if self.ongoing_response {
            return false;
        }
        if let Some(event_id) = error.get("event_id").and_then(Value::as_str) {
            return event_id == in_flight;
        }
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        code == "bad_response_create" || message.contains("response.create")
    }
}

struct WsInner {
    sink: tokio::sync::Mutex<Option<WsSink>>,
    events: Mutex<Option<mpsc::UnboundedSender<RealtimeModelEvent>>>,
    state: Mutex<WsState>,
    reader: Mutex<Option<JoinHandle<()>>>,
    close_requested: AtomicBool,
}

/// A realtime transport speaking the GA realtime protocol over a WebSocket.
///
/// By default it connects to `wss://api.openai.com/v1/realtime?model=<model>` with an
/// `Authorization: Bearer <key>` header, resolving the key from
/// [`RealtimeModelConfig::api_key`], then the SDK default key, then `OPENAI_API_KEY`.
pub struct OpenAIRealtimeWebSocketModel {
    inner: Arc<WsInner>,
}

impl Default for OpenAIRealtimeWebSocketModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for OpenAIRealtimeWebSocketModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner.state.lock().expect("realtime model poisoned");
        f.debug_struct("OpenAIRealtimeWebSocketModel")
            .field("model", &state.model)
            .field("ongoing_response", &state.ongoing_response)
            .finish_non_exhaustive()
    }
}

impl OpenAIRealtimeWebSocketModel {
    /// A disconnected transport using the default realtime model.
    pub fn new() -> Self {
        OpenAIRealtimeWebSocketModel {
            inner: Arc::new(WsInner {
                sink: tokio::sync::Mutex::new(None),
                events: Mutex::new(None),
                state: Mutex::new(WsState {
                    model: DEFAULT_REALTIME_MODEL.to_string(),
                    ..Default::default()
                }),
                reader: Mutex::new(None),
                close_requested: AtomicBool::new(false),
            }),
        }
    }

    /// The realtime model name the transport will use, or used, to connect.
    pub fn model_name(&self) -> String {
        self.inner
            .state
            .lock()
            .expect("realtime model poisoned")
            .model
            .clone()
    }
}

#[async_trait]
impl RealtimeModel for OpenAIRealtimeWebSocketModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        if self.inner.sink.lock().await.is_some() {
            return Err(AgentsError::user(
                "The realtime model is already connected.",
            ));
        }

        let settings = config.initial_model_settings.clone().unwrap_or_default();
        if config.call_id.is_some() && settings.model_name.is_some() {
            return Err(AgentsError::user(
                "Cannot specify both `call_id` and a model name when attaching to an existing \
                 realtime call.",
            ));
        }

        {
            let mut state = self.inner.state.lock().expect("realtime model poisoned");
            if let Some(model) = &settings.model_name {
                state.model.clone_from(model);
            }
            state.call_id.clone_from(&config.call_id);
            state.tracing = Some(
                settings
                    .tracing
                    .clone()
                    .unwrap_or(RealtimeModelTracing::Auto),
            );
            state.ongoing_response = false;
            state.pending_response_create = false;
            state.create_in_flight = None;
        }

        let url = match (&config.url, &config.call_id) {
            (Some(url), _) => url.clone(),
            (None, Some(call_id)) => format!("{DEFAULT_REALTIME_URL}?call_id={call_id}"),
            (None, None) => format!("{DEFAULT_REALTIME_URL}?model={}", self.model_name()),
        };

        let mut request = url
            .clone()
            .into_client_request()
            .map_err(|e| AgentsError::other(format!("Invalid realtime URL {url}: {e}")))?;
        match &config.headers {
            Some(headers) => {
                for (name, value) in headers {
                    let name = HeaderName::try_from(name.as_str())
                        .map_err(|e| AgentsError::user(format!("Invalid header name: {e}")))?;
                    let value = HeaderValue::from_str(value)
                        .map_err(|e| AgentsError::user(format!("Invalid header value: {e}")))?;
                    request.headers_mut().insert(name, value);
                }
            }
            None => {
                let api_key = resolve_api_key(config.api_key.as_deref()).ok_or_else(|| {
                    AgentsError::user("API key is required but was not provided.")
                })?;
                let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|e| AgentsError::user(format!("Invalid API key: {e}")))?;
                request
                    .headers_mut()
                    .insert(HeaderName::from_static("authorization"), value);
            }
        }

        let (stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| AgentsError::other(format!("Realtime connection failed: {e}")))?;
        let (sink, mut source) = stream.split();
        *self.inner.sink.lock().await = Some(sink);
        self.inner.close_requested.store(false, Ordering::SeqCst);

        let (tx, rx) = mpsc::unbounded_channel();
        *self.inner.events.lock().expect("realtime model poisoned") = Some(tx);
        self.inner.emit(RealtimeModelEvent::ConnectionStatus {
            status: RealtimeConnectionStatus::Connected,
        });

        let reader_inner = self.inner.clone();
        let reader = tokio::spawn(async move {
            while let Some(message) = source.next().await {
                match message {
                    Ok(Message::Text(text)) => match serde_json::from_str::<Value>(text.as_str()) {
                        Ok(value) => Box::pin(reader_inner.handle_server_event(value)).await,
                        Err(error) => reader_inner.emit(RealtimeModelEvent::Exception {
                            message: error.to_string(),
                            context: Some("Failed to parse WebSocket message as JSON".to_string()),
                        }),
                    },
                    Ok(_) => {}
                    Err(error) => {
                        reader_inner.emit(RealtimeModelEvent::Exception {
                            message: error.to_string(),
                            context: Some("WebSocket connection closed unexpectedly".to_string()),
                        });
                        break;
                    }
                }
            }
            if !reader_inner.close_requested.load(Ordering::SeqCst) {
                reader_inner.emit(RealtimeModelEvent::ConnectionStatus {
                    status: RealtimeConnectionStatus::Disconnected,
                });
                reader_inner.emit(RealtimeModelEvent::EndOfStream);
            }
            *reader_inner.events.lock().expect("realtime model poisoned") = None;
        });
        *self.inner.reader.lock().expect("realtime model poisoned") = Some(reader);

        self.inner.send_session_update(&settings).await?;
        Ok(rx)
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        match event {
            RealtimeModelSendEvent::RawMessage { message } => self.inner.send_raw(message).await,
            RealtimeModelSendEvent::UserInput { user_input } => {
                self.inner
                    .send_raw(json!({
                        "type": "conversation.item.create",
                        "item": user_input.to_conversation_item(),
                    }))
                    .await?;
                self.inner.request_response_create().await
            }
            RealtimeModelSendEvent::Audio { audio, commit } => {
                self.inner
                    .send_raw(json!({
                        "type": "input_audio_buffer.append",
                        "audio": BASE64.encode(&audio),
                    }))
                    .await?;
                if commit {
                    self.inner
                        .send_raw(json!({"type": "input_audio_buffer.commit"}))
                        .await?;
                }
                Ok(())
            }
            RealtimeModelSendEvent::ToolOutput {
                tool_call,
                output,
                start_response,
            } => {
                self.inner
                    .send_raw(json!({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": output,
                        },
                    }))
                    .await?;
                self.inner
                    .emit(RealtimeModelEvent::item_updated(RealtimeToolCallItem {
                        item_id: tool_call.id.clone().unwrap_or_default(),
                        previous_item_id: tool_call.previous_item_id.clone(),
                        call_id: Some(tool_call.call_id.clone()),
                        status: "completed".to_string(),
                        arguments: tool_call.arguments.clone(),
                        name: tool_call.name.clone(),
                        output: Some(output),
                    }));
                if start_response {
                    self.inner.request_response_create().await?;
                }
                Ok(())
            }
            RealtimeModelSendEvent::Interrupt {
                force_response_cancel,
            } => self.inner.send_interrupt(force_response_cancel).await,
            RealtimeModelSendEvent::SessionUpdate { session_settings } => {
                self.inner.send_session_update(&session_settings).await
            }
        }
    }

    async fn close(&self) -> Result<()> {
        self.inner.close_requested.store(true, Ordering::SeqCst);
        if let Some(mut sink) = self.inner.sink.lock().await.take() {
            let _ = sink.close().await;
        }
        if let Some(reader) = self
            .inner
            .reader
            .lock()
            .expect("realtime model poisoned")
            .take()
        {
            reader.abort();
        }
        *self.inner.events.lock().expect("realtime model poisoned") = None;
        let mut state = self.inner.state.lock().expect("realtime model poisoned");
        state.audio = AudioTracker::default();
        state.ongoing_response = false;
        state.pending_response_create = false;
        state.create_in_flight = None;
        Ok(())
    }
}

impl WsInner {
    fn emit(&self, event: RealtimeModelEvent) {
        let sender = { self.events.lock().expect("realtime model poisoned").clone() };
        if let Some(sender) = sender {
            let _ = sender.send(event);
        }
    }

    async fn send_raw(&self, message: Value) -> Result<()> {
        let mut guard = self.sink.lock().await;
        let sink = guard
            .as_mut()
            .ok_or_else(|| AgentsError::user("The realtime model is not connected."))?;
        sink.send(Message::text(message.to_string()))
            .await
            .map_err(|e| AgentsError::other(format!("Realtime send failed: {e}")))
    }

    async fn send_session_update(&self, settings: &RealtimeSessionModelSettings) -> Result<()> {
        let session = self.session_payload(settings);
        self.send_raw(json!({"type": "session.update", "session": session}))
            .await
    }

    /// Build the `session` object of a `session.update`, mirroring `_get_session_config`.
    fn session_payload(&self, settings: &RealtimeSessionModelSettings) -> Value {
        let (model, call_id) = {
            let state = self.state.lock().expect("realtime model poisoned");
            (state.model.clone(), state.call_id.clone())
        };
        let model = settings.model_name.clone().unwrap_or(model);

        let mut input = Map::new();
        let input_format = settings
            .input_audio_format
            .clone()
            .or_else(|| call_id.is_none().then(|| "pcm16".to_string()));
        if let Some(format) = input_format.as_deref()
            && let Some(value) = audio_format_payload(format)
        {
            input.insert("format".to_string(), value);
        }
        match &settings.input_audio_noise_reduction {
            Some(Some(config)) => {
                input.insert(
                    "noise_reduction".to_string(),
                    serde_json::to_value(config).unwrap_or(Value::Null),
                );
            }
            Some(None) => {
                input.insert("noise_reduction".to_string(), Value::Null);
            }
            None => {}
        }
        let transcription = settings
            .input_audio_transcription
            .clone()
            .unwrap_or_else(|| {
                crate::realtime::config::RealtimeInputAudioTranscriptionConfig::new(
                    DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL,
                )
            });
        input.insert(
            "transcription".to_string(),
            serde_json::to_value(&transcription).unwrap_or(Value::Null),
        );
        let turn_detection = settings
            .turn_detection
            .clone()
            .unwrap_or_else(|| Some(RealtimeTurnDetectionConfig::semantic_vad()));
        match &turn_detection {
            Some(config) => {
                input.insert(
                    "turn_detection".to_string(),
                    serde_json::to_value(config).unwrap_or(Value::Null),
                );
            }
            None => {
                input.insert("turn_detection".to_string(), Value::Null);
            }
        }
        {
            let mut state = self.state.lock().expect("realtime model poisoned");
            state.automatic_response_cancellation = turn_detection
                .as_ref()
                .and_then(|config| config.interrupt_response)
                .unwrap_or(false);
        }

        let mut output = Map::new();
        output.insert(
            "voice".to_string(),
            Value::String(
                settings
                    .voice
                    .clone()
                    .unwrap_or_else(|| DEFAULT_REALTIME_VOICE.to_string()),
            ),
        );
        let output_format = settings
            .output_audio_format
            .clone()
            .or_else(|| call_id.is_none().then(|| "pcm16".to_string()));
        if let Some(format) = output_format.as_deref()
            && let Some(value) = audio_format_payload(format)
        {
            output.insert("format".to_string(), value);
        }
        if let Some(speed) = settings.speed {
            output.insert("speed".to_string(), json!(speed));
        }

        let modalities = settings
            .output_modalities
            .clone()
            .unwrap_or_else(|| vec!["audio".to_string()]);

        let tools: Vec<Value> = settings
            .tools
            .clone()
            .unwrap_or_default()
            .iter()
            .map(|spec| spec.to_session_param())
            .collect();

        let mut session = Map::new();
        session.insert("type".to_string(), Value::String("realtime".to_string()));
        session.insert("model".to_string(), Value::String(model));
        session.insert("output_modalities".to_string(), json!(modalities));
        session.insert(
            "audio".to_string(),
            json!({"input": Value::Object(input), "output": Value::Object(output)}),
        );
        session.insert("tools".to_string(), Value::Array(tools));
        if let Some(instructions) = &settings.instructions {
            session.insert(
                "instructions".to_string(),
                Value::String(instructions.clone()),
            );
        }
        if let Some(prompt) = &settings.prompt {
            session.insert(
                "prompt".to_string(),
                serde_json::to_value(prompt).unwrap_or(Value::Null),
            );
        }
        if let Some(max_output_tokens) = &settings.max_output_tokens {
            session.insert("max_output_tokens".to_string(), max_output_tokens.clone());
        }
        if let Some(tool_choice) = &settings.tool_choice {
            session.insert(
                "tool_choice".to_string(),
                Value::String(tool_choice.clone()),
            );
        }
        if let Some(parallel_tool_calls) = settings.parallel_tool_calls {
            session.insert(
                "parallel_tool_calls".to_string(),
                json!(parallel_tool_calls),
            );
        }
        if let Some(reasoning) = &settings.reasoning {
            session.insert("reasoning".to_string(), reasoning.clone());
        }
        Value::Object(session)
    }

    /// Ask the model for a new response, deferring while one is already running.
    async fn request_response_create(&self) -> Result<()> {
        let event_id = {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .request_response_create()
        };
        match event_id {
            Some(event_id) => self.send_response_create(event_id).await,
            None => Ok(()),
        }
    }

    /// Tell the server how much of an item's audio the client actually played.
    async fn send_truncate(&self, playback: &PlaybackState) -> Result<()> {
        self.send_raw(json!({
            "type": "conversation.item.truncate",
            "item_id": playback.item_id,
            "content_index": playback.content_index,
            "audio_end_ms": playback.truncate_ms(),
        }))
        .await
    }

    async fn send_response_create(&self, event_id: String) -> Result<()> {
        let result = self
            .send_raw(json!({"type": "response.create", "event_id": event_id}))
            .await;
        if result.is_err() {
            // The create never reached the server, so it must not block later requests.
            let _ = self
                .state
                .lock()
                .expect("realtime model poisoned")
                .on_response_create_failed();
        }
        result
    }

    async fn send_interrupt(&self, force_response_cancel: bool) -> Result<()> {
        let (playback, ongoing, automatic) = {
            let state = self.state.lock().expect("realtime model poisoned");
            (
                state.audio.playback_state(),
                state.ongoing_response,
                state.automatic_response_cancellation,
            )
        };

        if let Some(playback) = playback {
            self.emit(RealtimeModelEvent::AudioInterrupted {
                item_id: playback.item_id.clone(),
                content_index: playback.content_index,
            });
            self.send_truncate(&playback).await?;
            self.state
                .lock()
                .expect("realtime model poisoned")
                .audio
                .on_interrupted();
        }

        if ongoing && (force_response_cancel || !automatic) {
            self.send_raw(json!({"type": "response.cancel"})).await?;
        }
        Ok(())
    }

    /// Translate one server event into model events, reacting where the protocol requires it.
    async fn handle_server_event(&self, event: Value) {
        self.emit(RealtimeModelEvent::RawServerEvent {
            data: event.clone(),
        });
        let Some(event_type) = event.get("type").and_then(Value::as_str) else {
            self.emit(RealtimeModelEvent::Other { data: event });
            return;
        };

        match event_type {
            "response.output_audio.delta" => {
                let item_id = string_field(&event, "item_id");
                let content_index = usize_field(&event, "content_index");
                let response_id = string_field(&event, "response_id");
                let Ok(data) = BASE64.decode(string_field(&event, "delta")) else {
                    self.emit(RealtimeModelEvent::Exception {
                        message: "Invalid base64 audio delta".to_string(),
                        context: Some("response.output_audio.delta".to_string()),
                    });
                    return;
                };
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .audio
                    .on_delta(&item_id, content_index, &data);
                self.emit(RealtimeModelEvent::audio(RealtimeModelAudioEvent {
                    data,
                    response_id,
                    item_id,
                    content_index,
                }));
            }
            "response.output_audio.done" => {
                self.emit(RealtimeModelEvent::AudioDone {
                    item_id: string_field(&event, "item_id"),
                    content_index: usize_field(&event, "content_index"),
                });
            }
            "response.output_audio_transcript.delta" => {
                self.emit(RealtimeModelEvent::TranscriptDelta {
                    item_id: string_field(&event, "item_id"),
                    delta: string_field(&event, "delta"),
                    response_id: string_field(&event, "response_id"),
                });
            }
            "response.output_text.delta" => {
                self.emit(RealtimeModelEvent::OutputTextDelta {
                    item_id: string_field(&event, "item_id"),
                    delta: string_field(&event, "delta"),
                    response_id: string_field(&event, "response_id"),
                });
            }
            "response.created" => {
                let response_id = event
                    .pointer("/response/id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .on_response_created();
                self.emit(RealtimeModelEvent::TurnStarted { response_id });
            }
            "response.done" => {
                let response_id = event
                    .pointer("/response/id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let deferred = {
                    self.state
                        .lock()
                        .expect("realtime model poisoned")
                        .on_response_done()
                };
                if let Some(usage) = event.pointer("/response/usage")
                    && let Some(usage) = convert_usage(usage)
                {
                    self.emit(RealtimeModelEvent::Usage {
                        usage: Box::new(usage),
                    });
                }
                self.emit(RealtimeModelEvent::TurnEnded { response_id });
                if let Some(event_id) = deferred {
                    let _ = self.send_response_create(event_id).await;
                }
            }
            "session.created" | "session.updated" => {
                if let Some(interrupt) = event
                    .pointer("/session/audio/input/turn_detection/interrupt_response")
                    .and_then(Value::as_bool)
                {
                    self.state
                        .lock()
                        .expect("realtime model poisoned")
                        .automatic_response_cancellation = interrupt;
                }
                if event_type == "session.created" {
                    self.send_tracing_config().await;
                }
            }
            "error" => {
                let error = event.get("error").cloned().unwrap_or(Value::Null);
                let deferred = {
                    let mut state = self.state.lock().expect("realtime model poisoned");
                    state
                        .error_rejects_pending_create(&error)
                        .then(|| state.on_response_create_failed())
                        .flatten()
                };
                self.emit(RealtimeModelEvent::Error { error });
                if let Some(event_id) = deferred {
                    let _ = self.send_response_create(event_id).await;
                }
            }
            "conversation.item.deleted" => {
                self.emit(RealtimeModelEvent::ItemDeleted {
                    item_id: string_field(&event, "item_id"),
                });
            }
            "conversation.item.added"
            | "conversation.item.created"
            | "conversation.item.retrieved" => {
                let previous_item_id = (event_type == "conversation.item.created")
                    .then(|| {
                        event
                            .get("previous_item_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .flatten();
                if let Some(item) = event.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("message")
                    && let Some(message) = parse_message_item(item, previous_item_id, None)
                {
                    self.emit(RealtimeModelEvent::item_updated(message));
                }
            }
            "conversation.item.input_audio_transcription.completed" => {
                self.emit(RealtimeModelEvent::InputAudioTranscriptionCompleted {
                    item_id: string_field(&event, "item_id"),
                    transcript: string_field(&event, "transcript"),
                });
            }
            "response.output_item.added" | "response.output_item.done" => {
                let done = event_type == "response.output_item.done";
                self.handle_output_item(event.get("item"), done);
            }
            "input_audio_buffer.timeout_triggered" => {
                self.emit(RealtimeModelEvent::InputAudioTimeoutTriggered {
                    item_id: string_field(&event, "item_id"),
                    audio_start_ms: u64_field(&event, "audio_start_ms"),
                    audio_end_ms: u64_field(&event, "audio_end_ms"),
                });
            }
            "input_audio_buffer.speech_started" => {
                // Barge-in: stop local playback so the user is not talked over.
                let (playback, ongoing, automatic) = {
                    let state = self.state.lock().expect("realtime model poisoned");
                    (
                        state.audio.playback_state(),
                        state.ongoing_response,
                        state.automatic_response_cancellation,
                    )
                };
                if let Some(playback) = playback {
                    self.emit(RealtimeModelEvent::AudioInterrupted {
                        item_id: playback.item_id.clone(),
                        content_index: playback.content_index,
                    });
                    // Truncating audio the client already finished playing is a no-op the server
                    // can reject, so skip it unless a response is still producing audio.
                    if !playback.is_complete() || ongoing {
                        let _ = self.send_truncate(&playback).await;
                    }
                    self.state
                        .lock()
                        .expect("realtime model poisoned")
                        .audio
                        .on_interrupted();
                    if !automatic {
                        // The server will not stop on its own, so stop it here.
                        let _ = self.send_raw(json!({"type": "response.cancel"})).await;
                    }
                }
            }
            "conversation.item.input_audio_transcription.delta"
            | "response.function_call_arguments.delta"
            | "conversation.item.truncated" => {}
            _ => self.emit(RealtimeModelEvent::Other { data: event }),
        }
    }

    fn handle_output_item(&self, item: Option<&Value>, done: bool) {
        let Some(item) = item else { return };
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                if item.get("status").and_then(Value::as_str) != Some("completed") {
                    return;
                }
                let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.emit(RealtimeModelEvent::item_updated(RealtimeToolCallItem {
                    item_id: id.to_string(),
                    previous_item_id: None,
                    call_id: Some(call_id.to_string()),
                    // The call and its output share one item, completed once the output lands.
                    status: "in_progress".to_string(),
                    arguments: arguments.to_string(),
                    name: name.to_string(),
                    output: None,
                }));
                self.emit(RealtimeModelEvent::FunctionCall(
                    RealtimeModelToolCallEvent {
                        name: name.to_string(),
                        call_id: call_id.to_string(),
                        arguments: arguments.to_string(),
                        id: Some(id.to_string()),
                        previous_item_id: None,
                    },
                ));
            }
            Some("message") => {
                let default_status = if done { "completed" } else { "in_progress" };
                if let Some(message) = parse_message_item(item, None, Some(default_status)) {
                    self.emit(RealtimeModelEvent::item_updated(message));
                }
            }
            _ => {}
        }
    }

    async fn send_tracing_config(&self) {
        let (tracing, model) = {
            let state = self.state.lock().expect("realtime model poisoned");
            (state.tracing.clone(), state.model.clone())
        };
        let Some(tracing) = tracing else { return };
        let Some(value) = tracing.to_session_value() else {
            return;
        };
        let _ = self
            .send_raw(json!({
                "type": "session.update",
                "session": {"type": "realtime", "model": model, "tracing": value},
            }))
            .await;
    }
}

/// The API key for a realtime connection: explicit, then the SDK default, then the environment.
fn resolve_api_key(explicit: Option<&str>) -> Option<String> {
    if let Some(key) = explicit {
        return Some(key.to_string());
    }
    if let Some(key) = crate::models::default_openai_key() {
        return Some(key);
    }
    std::env::var("OPENAI_API_KEY").ok()
}

/// The GA `audio.format` object for a legacy or GA format name.
fn audio_format_payload(format: &str) -> Option<Value> {
    match format {
        "pcm16" | "audio/pcm" | "pcm" => Some(json!({"type": "audio/pcm", "rate": 24000})),
        "g711_ulaw" | "audio/pcmu" | "pcmu" => Some(json!({"type": "audio/pcmu"})),
        "g711_alaw" | "audio/pcma" | "pcma" => Some(json!({"type": "audio/pcma"})),
        other => {
            log::debug!("Unknown realtime audio format: {other}");
            None
        }
    }
}

fn string_field(event: &Value, key: &str) -> String {
    event
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn usize_field(event: &Value, key: &str) -> usize {
    event.get(key).and_then(Value::as_u64).unwrap_or_default() as usize
}

fn u64_field(event: &Value, key: &str) -> u64 {
    event.get(key).and_then(Value::as_u64).unwrap_or_default()
}

/// Parse a wire message item into a [`RealtimeItem`], normalizing GA content types.
fn parse_message_item(
    item: &Value,
    previous_item_id: Option<String>,
    default_status: Option<&str>,
) -> Option<RealtimeItem> {
    let role = match item
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("assistant")
    {
        "user" => RealtimeMessageRole::User,
        "system" => RealtimeMessageRole::System,
        _ => RealtimeMessageRole::Assistant,
    };
    let mut content = Vec::new();
    for part in item
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let audio = part
            .get("audio")
            .and_then(Value::as_str)
            .map(str::to_string);
        let transcript = part
            .get("transcript")
            .and_then(Value::as_str)
            .map(str::to_string);
        let text = part.get("text").and_then(Value::as_str).map(str::to_string);
        match part.get("type").and_then(Value::as_str) {
            Some("audio") | Some("output_audio") => {
                content.push(RealtimeContent::Audio { audio, transcript })
            }
            Some("input_audio") => content.push(RealtimeContent::InputAudio { audio, transcript }),
            Some("text") | Some("output_text") => content.push(RealtimeContent::Text { text }),
            Some("input_text") => content.push(RealtimeContent::InputText { text }),
            Some("input_image") => content.push(RealtimeContent::InputImage {
                image_url: part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                detail: part
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }),
            _ => {}
        }
    }

    let status = item
        .get("status")
        .and_then(Value::as_str)
        .filter(|status| matches!(*status, "in_progress" | "completed" | "incomplete"))
        .map(str::to_string)
        .or_else(|| default_status.map(str::to_string));

    Some(RealtimeItem::Message(RealtimeMessageItem {
        item_id: item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        previous_item_id,
        role,
        status,
        content,
    }))
}

/// Convert a realtime `response.usage` object into SDK usage.
fn convert_usage(usage: &Value) -> Option<Usage> {
    let input_tokens = usage.get("input_tokens").and_then(Value::as_u64)?;
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);
    let mut converted = Usage {
        requests: 1,
        input_tokens,
        output_tokens,
        total_tokens,
        ..Default::default()
    };
    if let Some(cached) = usage
        .pointer("/input_token_details/cached_tokens")
        .and_then(Value::as_u64)
    {
        converted.input_tokens_details.cached_tokens = cached;
    }
    Some(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::config::RealtimeFunctionToolSpec;

    #[test]
    fn default_model_matches_python() {
        assert_eq!(DEFAULT_REALTIME_MODEL, "gpt-realtime-2.1");
        let model = OpenAIRealtimeWebSocketModel::new();
        assert_eq!(model.model_name(), "gpt-realtime-2.1");
        assert_eq!(
            format!("{DEFAULT_REALTIME_URL}?model={}", model.model_name()),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
        );
    }

    #[test]
    fn session_payload_uses_ga_defaults() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let settings = RealtimeSessionModelSettings::new()
            .with_instructions("Be brief")
            .with_output_modalities(["text"]);
        let payload = model.inner.session_payload(&settings);

        assert_eq!(payload["type"], "realtime");
        assert_eq!(payload["model"], "gpt-realtime-2.1");
        assert_eq!(payload["output_modalities"], json!(["text"]));
        assert_eq!(payload["instructions"], "Be brief");
        assert_eq!(payload["audio"]["output"]["voice"], "ash");
        assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcm");
        assert_eq!(payload["audio"]["input"]["format"]["rate"], 24000);
        assert_eq!(payload["audio"]["output"]["format"]["type"], "audio/pcm");
        assert_eq!(
            payload["audio"]["input"]["transcription"]["model"],
            "gpt-4o-mini-transcribe"
        );
        assert_eq!(
            payload["audio"]["input"]["turn_detection"]["type"],
            "semantic_vad"
        );
        assert_eq!(
            payload["audio"]["input"]["turn_detection"]["interrupt_response"],
            true
        );
        assert_eq!(payload["tools"], json!([]));
    }

    #[test]
    fn session_payload_serializes_tools_and_disabled_turn_detection() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let mut settings = RealtimeSessionModelSettings::new().without_turn_detection();
        settings.tools = Some(vec![RealtimeFunctionToolSpec::new(
            "get_weather",
            "Get the weather",
            json!({"type": "object", "properties": {}}),
        )]);
        let payload = model.inner.session_payload(&settings);
        assert_eq!(payload["audio"]["input"]["turn_detection"], Value::Null);
        assert_eq!(payload["tools"][0]["type"], "function");
        assert_eq!(payload["tools"][0]["name"], "get_weather");
        assert_eq!(payload["tools"][0]["description"], "Get the weather");
    }

    #[test]
    fn audio_formats_map_to_ga_objects() {
        assert_eq!(
            audio_format_payload("g711_ulaw"),
            Some(json!({"type": "audio/pcmu"}))
        );
        assert_eq!(
            audio_format_payload("g711_alaw"),
            Some(json!({"type": "audio/pcma"}))
        );
        assert_eq!(audio_format_payload("nonsense"), None);
    }

    #[test]
    fn parses_ga_message_content() {
        let item = json!({
            "id": "item_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_audio", "transcript": "hi"}],
        });
        let parsed = parse_message_item(&item, None, Some("completed")).unwrap();
        let message = parsed.as_message().unwrap();
        assert_eq!(message.item_id, "item_1");
        assert_eq!(message.role, RealtimeMessageRole::Assistant);
        assert_eq!(message.status.as_deref(), Some("completed"));
        assert_eq!(message.text(), "hi");
    }

    #[test]
    fn converts_usage() {
        let usage = convert_usage(&json!({
            "input_tokens": 10,
            "output_tokens": 4,
            "input_token_details": {"cached_tokens": 2},
        }))
        .unwrap();
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.total_tokens, 14);
        assert_eq!(usage.input_tokens_details.cached_tokens, 2);
    }
}

#[cfg(test)]
mod sequencing_tests {
    use super::*;

    fn state() -> WsState {
        WsState {
            model: DEFAULT_REALTIME_MODEL.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn a_second_create_defers_until_the_first_response_starts() {
        let mut state = state();
        let first = state
            .request_response_create()
            .expect("first create is sent");
        // Parallel tool outputs both see an idle connection until `response.created` lands.
        assert_eq!(state.request_response_create(), None);
        assert!(state.pending_response_create);

        state.on_response_created();
        assert_eq!(state.create_in_flight, None);
        assert!(state.ongoing_response);

        let second = state
            .on_response_done()
            .expect("deferred create is flushed");
        assert_ne!(first, second);
        assert!(!state.pending_response_create);
        assert_eq!(state.on_response_done(), None);
    }

    #[test]
    fn an_unrelated_error_keeps_a_deferred_create() {
        let mut state = state();
        let event_id = state.request_response_create().unwrap();
        assert_eq!(state.request_response_create(), None);

        // A barge-in truncate failure must not drop the deferred response.
        let unrelated = json!({"event_id": "agents_rs_truncate", "message": "bad truncate"});
        assert!(!state.error_rejects_pending_create(&unrelated));
        assert!(state.pending_response_create);

        let rejection = json!({"event_id": event_id, "message": "bad response.create"});
        assert!(state.error_rejects_pending_create(&rejection));
        assert!(state.on_response_create_failed().is_some());
        assert!(!state.pending_response_create);
    }

    #[test]
    fn errors_without_an_event_id_match_on_code_or_message() {
        let mut state = state();
        state.request_response_create().unwrap();
        assert!(state.error_rejects_pending_create(&json!({"code": "bad_response_create"})));
        assert!(state.error_rejects_pending_create(
            &json!({"message": "Cannot send response.create right now"})
        ));
        assert!(!state.error_rejects_pending_create(&json!({"message": "input buffer too small"})));

        // An error arriving while a response runs is never a create rejection.
        state.on_response_created();
        assert!(!state.error_rejects_pending_create(&json!({"code": "bad_response_create"})));
    }

    #[test]
    fn truncation_never_exceeds_the_received_audio() {
        let playback = PlaybackState {
            item_id: "item_1".to_string(),
            content_index: 0,
            elapsed_ms: 10.2,
            length_ms: 10.2,
        };
        assert_eq!(playback.truncate_ms(), 10);
        assert!(playback.is_complete());

        let partial = PlaybackState {
            elapsed_ms: 4.9,
            ..playback
        };
        assert_eq!(partial.truncate_ms(), 4);
        assert!(!partial.is_complete());
    }
}
