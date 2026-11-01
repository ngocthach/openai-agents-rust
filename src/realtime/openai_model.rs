//! The OpenAI realtime WebSocket transport. Mirrors `agents.realtime.openai_realtime`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
use tokio_tungstenite::tungstenite::protocol::{Message, WebSocketConfig};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::{AgentsError, Result};
use crate::realtime::agent::RealtimeAgent;
use crate::realtime::config::{
    DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL, DEFAULT_REALTIME_MODEL, DEFAULT_REALTIME_VOICE,
    RealtimeAudioFormat, RealtimeInputAudioTranscriptionConfig, RealtimeModelTracing,
    RealtimeRunConfig, RealtimeSessionModelSettings, RealtimeTurnDetectionConfig, RealtimeVoice,
};
use crate::realtime::debug::log_model_action_error;
use crate::realtime::event_validation::{validate_client_event, validate_server_event};
use crate::realtime::items::{
    RealtimeContent, RealtimeItem, RealtimeMessageItem, RealtimeMessageRole, RealtimeToolCallItem,
};
use crate::realtime::model::{
    RealtimeApiKey, RealtimeModel, RealtimeModelConfig, RealtimeModelEvents, RealtimePlaybackState,
    RealtimePlaybackTracker, RealtimeSendCondition, calculate_audio_length_ms,
};
use crate::realtime::model_events::{
    RealtimeConnectionStatus, RealtimeModelAudioEvent, RealtimeModelCachedTokensDetails,
    RealtimeModelEvent, RealtimeModelInputTokensDetails, RealtimeModelOutputTokensDetails,
    RealtimeModelToolCallEvent,
};
use crate::realtime::model_inputs::{RealtimeInterrupt, RealtimeModelSendEvent};
use crate::realtime::session::build_model_settings_from_agent;
use crate::run_context::RunContext;
use crate::usage::Usage;

/// The default realtime WebSocket endpoint.
const DEFAULT_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";

/// Sent as the `User-Agent` of the WebSocket handshake.
const USER_AGENT: &str = concat!("Agents/Rust ", env!("CARGO_PKG_VERSION"));

/// Largest incoming message accepted by default, matching the Python client.
const DEFAULT_MAX_MESSAGE_SIZE: usize = 8 * 1024 * 1024;

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

/// What the transport received for one `(item_id, content_index)` pair.
#[derive(Debug, Clone)]
struct ModelAudioState {
    initial_received: Instant,
    audio_length_ms: f64,
    response_id: Option<String>,
}

/// The transport's own record of the audio it has sent out, keyed by item and by response.
///
/// Port of `agents.realtime._default_tracker.ModelAudioTracker`. Response-scoped bookkeeping is
/// what lets an interruption target one response without disturbing a newer one.
#[derive(Debug, Default)]
struct ModelAudioTracker {
    format: Option<String>,
    states: HashMap<(String, usize), ModelAudioState>,
    last_audio_item: Option<(String, usize)>,
    items_by_response: HashMap<String, Vec<(String, usize)>>,
}

impl ModelAudioTracker {
    fn set_audio_format(&mut self, format: impl Into<String>) {
        self.format = Some(format.into());
    }

    fn on_audio_delta(
        &mut self,
        item_id: &str,
        content_index: usize,
        bytes: &[u8],
        response_id: Option<&str>,
    ) {
        let ms = calculate_audio_length_ms(self.format.as_deref(), bytes);
        let key = (item_id.to_string(), content_index);
        self.last_audio_item = Some(key.clone());
        if let Some(response_id) = response_id {
            let items = self
                .items_by_response
                .entry(response_id.to_string())
                .or_default();
            if !items.contains(&key) {
                items.push(key.clone());
            }
        }
        match self.states.get_mut(&key) {
            Some(state) => {
                state.audio_length_ms += ms;
                if let Some(response_id) = response_id {
                    state.response_id = Some(response_id.to_string());
                }
            }
            None => {
                self.states.insert(
                    key,
                    ModelAudioState {
                        initial_received: Instant::now(),
                        audio_length_ms: ms,
                        response_id: response_id.map(str::to_string),
                    },
                );
            }
        }
    }

    fn on_interrupted(&mut self) {
        if let Some(last) = self.last_audio_item.take()
            && let Some(state) = self.states.get(&last)
            && let Some(response_id) = state.response_id.clone()
        {
            self.items_by_response.remove(&response_id);
        }
    }

    /// Release the indexes of one response, keeping a newer response's audio intact.
    fn on_response_interrupted(&mut self, response_id: &str) {
        if let Some(items) = self.items_by_response.remove(response_id)
            && let Some(last) = self.last_audio_item.as_ref()
            && items.contains(last)
        {
            self.last_audio_item = None;
        }
    }

    fn on_response_done(&mut self, response_id: &str) {
        self.items_by_response.remove(response_id);
    }

    fn state(&self, item_id: &str, content_index: usize) -> Option<&ModelAudioState> {
        self.states.get(&(item_id.to_string(), content_index))
    }

    fn last_audio_item(&self) -> Option<(String, usize)> {
        self.last_audio_item.clone()
    }

    fn audio_items_for_response(&self, response_id: &str) -> Vec<(String, usize)> {
        self.items_by_response
            .get(response_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Where playback is, from the tracker the application supplied or the transport's own estimate.
#[derive(Debug, Clone, PartialEq, Default)]
struct PlaybackPosition {
    item_id: Option<String>,
    content_index: Option<usize>,
    elapsed_ms: Option<f64>,
}

impl PlaybackPosition {
    /// The item being played, with its content index defaulted as Python does.
    fn audio_item(&self) -> Option<(String, usize)> {
        Some((
            self.item_id.clone()?,
            self.content_index.unwrap_or_default(),
        ))
    }
}

impl From<RealtimePlaybackState> for PlaybackPosition {
    fn from(value: RealtimePlaybackState) -> Self {
        PlaybackPosition {
            item_id: value.current_item_id,
            content_index: value.current_item_content_index,
            elapsed_ms: value.elapsed_ms,
        }
    }
}

#[derive(Debug, Default)]
struct WsState {
    model: String,
    tracing: Option<RealtimeModelTracing>,
    call_id: Option<String>,
    audio: ModelAudioTracker,
    /// Responses whose buffered audio was interrupted; their later deltas are dropped.
    interrupted_audio_response_ids: HashSet<String>,
    ongoing_response: bool,
    ongoing_response_id: Option<String>,
    /// A `response.cancel` was sent for the active response and not yet resolved.
    cancel_requested: bool,
    /// The item the model is currently producing audio for.
    current_item_id: Option<String>,
    /// A `response.create` was sent and neither `response.created` nor a matching error arrived.
    create_in_flight: Option<String>,
    /// A response was requested while one was running or in flight.
    pending_response_create: bool,
    /// The body of a caller-supplied `response.create`, sent in place of a bare one.
    create_payload: Option<Value>,
    create_counter: u64,
    automatic_response_cancellation: bool,
}

impl WsState {
    /// Request a response. Returns the `event_id` to send with it, or `None` to defer.
    ///
    /// The realtime API rejects a second `response.create` while a response is active, and
    /// parallel tool outputs would otherwise both see an idle connection after `response.done`.
    fn request_response_create(&mut self, payload: Option<Value>) -> Option<String> {
        if payload.is_some() {
            self.create_payload = payload;
        }
        if self.ongoing_response || self.create_in_flight.is_some() {
            self.pending_response_create = true;
            return None;
        }
        Some(self.begin_response_create())
    }

    fn begin_response_create(&mut self) -> String {
        self.create_counter += 1;
        // A caller-chosen id wins, so an error that names it still matches this create.
        let event_id = self
            .create_payload
            .as_ref()
            .and_then(|payload| payload.get("event_id"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("agents_rs_response_create_{}", self.create_counter));
        self.create_in_flight = Some(event_id.clone());
        event_id
    }

    fn on_response_created(&mut self, response_id: Option<&str>) {
        self.create_in_flight = None;
        self.ongoing_response = true;
        self.ongoing_response_id = response_id.map(str::to_string);
        self.cancel_requested = false;
    }

    /// Whether a `response.cancel` should go out now, mirroring `begin_cancel_response`.
    ///
    /// The server rejects a cancel when no response is active, so a barge-in that lands after the
    /// response ended, a duplicate cancel, or a cancel aimed at a response that is no longer the
    /// active one must stay local.
    fn begin_cancel(&mut self, response_id: Option<&str>) -> bool {
        if !self.ongoing_response || self.cancel_requested {
            return false;
        }
        if let (Some(requested), Some(ongoing)) = (response_id, self.ongoing_response_id.as_deref())
            && requested != ongoing
        {
            return false;
        }
        self.cancel_requested = true;
        true
    }

    /// Release the cancel slot when the send failed, so a later attempt is not swallowed.
    fn release_cancel(&mut self) {
        self.cancel_requested = false;
    }

    /// Whether the response now generating is the one that produced this item's audio.
    ///
    /// Interruption is preserved when ownership is unknown: with known identities, only the
    /// response that produced an item can extend its received audio.
    fn ongoing_response_owns_audio(&self, item_id: &str, content_index: usize) -> bool {
        if !self.ongoing_response {
            return false;
        }
        let (Some(response_id), Some(state)) = (
            self.ongoing_response_id.as_deref(),
            self.audio.state(item_id, content_index),
        ) else {
            return true;
        };
        match state.response_id.as_deref() {
            None => true,
            Some(owner) => owner == response_id,
        }
    }

    /// The millisecond ceiling of the audio actually received for an item.
    fn max_audio_ms(&self, item_id: &str, content_index: usize) -> Option<i64> {
        self.audio
            .state(item_id, content_index)
            .map(|state| state.audio_length_ms.ceil() as i64)
    }

    /// Where playback is, from the application's tracker or the transport's own estimate.
    fn playback_position(
        &self,
        playback_tracker: Option<&RealtimePlaybackTracker>,
    ) -> PlaybackPosition {
        if let Some(tracker) = playback_tracker {
            return tracker.state().into();
        }
        if let Some((item_id, content_index)) = self.audio.last_audio_item()
            && let Some(state) = self.audio.state(&item_id, content_index)
        {
            return PlaybackPosition {
                item_id: Some(item_id),
                content_index: Some(content_index),
                elapsed_ms: Some(state.initial_received.elapsed().as_secs_f64() * 1000.0),
            };
        }
        PlaybackPosition::default()
    }

    /// Clear the active response and return the `event_id` of a deferred create to send now.
    ///
    /// A `response.done` for an older response says nothing about the one now generating, so it
    /// must not free the slot: a create sent then would collide with the active response.
    fn on_response_done(&mut self, response_id: Option<&str>) -> Option<String> {
        if let (Some(done), Some(ongoing)) = (response_id, self.ongoing_response_id.as_deref())
            && done != ongoing
        {
            return None;
        }
        self.ongoing_response = false;
        self.ongoing_response_id = None;
        self.cancel_requested = false;
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

    /// Drop the in-flight create when its send did not complete, so later requests are not
    /// deferred behind a response that was never requested.
    fn release_create(&mut self) {
        self.create_in_flight = None;
        self.create_payload = None;
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

/// Which pending request a [`SlotGuard`] frees.
#[derive(Clone, Copy)]
enum SlotKind {
    Create,
    Cancel,
}

/// Frees a `response.create` or `response.cancel` slot unless the send is confirmed.
///
/// A send can end three ways: accepted, failed, or abandoned because the caller's future was
/// dropped. The latter two must leave the slot free so a retry is not swallowed.
struct SlotGuard<'a> {
    inner: &'a WsInner,
    kind: SlotKind,
    armed: bool,
}

impl<'a> SlotGuard<'a> {
    fn new(inner: &'a WsInner, kind: SlotKind) -> Self {
        SlotGuard {
            inner,
            kind,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.inner.state.lock().expect("realtime model poisoned");
        match self.kind {
            SlotKind::Create => state.release_create(),
            SlotKind::Cancel => state.release_cancel(),
        }
    }
}

struct WsInner {
    sink: tokio::sync::Mutex<Option<WsSink>>,
    /// When the last pong arrived, for the keepalive deadline.
    last_pong: Mutex<Instant>,
    keepalive: Mutex<Option<JoinHandle<()>>>,
    playback_tracker: Mutex<Option<Arc<RealtimePlaybackTracker>>>,
    events: Mutex<Option<mpsc::UnboundedSender<RealtimeModelEvent>>>,
    state: Mutex<WsState>,
    reader: Mutex<Option<JoinHandle<()>>>,
    close_requested: AtomicBool,
    /// A connection attempt is in progress.
    connecting: AtomicBool,
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
                last_pong: Mutex::new(Instant::now()),
                keepalive: Mutex::new(None),
                playback_tracker: Mutex::new(None),
                events: Mutex::new(None),
                state: Mutex::new(WsState {
                    model: DEFAULT_REALTIME_MODEL.to_string(),
                    ..Default::default()
                }),
                reader: Mutex::new(None),
                close_requested: AtomicBool::new(false),
                connecting: AtomicBool::new(false),
            }),
        }
    }

    /// Render the `session` object this transport would send for `settings`.
    ///
    /// This is what a `session.update` carries, and what the Realtime Calls accept endpoint
    /// expects. It resolves the same precedence the transport uses, so a payload built here
    /// matches what a live session would send.
    pub fn session_payload_for_settings(&self, settings: &RealtimeSessionModelSettings) -> Value {
        self.inner.session_payload(settings)
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

/// Releases the connect slot when a connection attempt ends, however it ends.
struct ConnectSlot<'a>(&'a AtomicBool);

impl Drop for ConnectSlot<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl OpenAIRealtimeWebSocketModel {
    async fn connect_inner(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        if self.inner.sink.lock().await.is_some() {
            return Err(AgentsError::user(
                "Already connected: the realtime model is already connected.",
            ));
        }

        let settings = config.initial_model_settings.clone().unwrap_or_default();
        if config.call_id.is_some() && settings.model_name.is_some() {
            return Err(AgentsError::user(
                "Cannot specify both `call_id` and `model_name` when attaching to an existing \
                 realtime call.",
            ));
        }

        *self
            .inner
            .playback_tracker
            .lock()
            .expect("realtime model poisoned") = config.playback_tracker.clone();
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
            state.ongoing_response_id = None;
            state.cancel_requested = false;
            state.current_item_id = None;
            state.pending_response_create = false;
            state.create_in_flight = None;
            state.create_payload = None;
            state.interrupted_audio_response_ids.clear();
        }

        let url = connect_url(
            config.url.as_deref(),
            config.call_id.as_deref(),
            &self.model_name(),
        );

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
                let api_key = get_api_key(config.api_key.as_ref()).await?;
                let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|e| AgentsError::user(format!("Invalid API key: {e}")))?;
                request
                    .headers_mut()
                    .insert(HeaderName::from_static("authorization"), value);
            }
        }

        crate::util::ensure_crypto_provider();

        request.headers_mut().insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static(USER_AGENT),
        );

        let transport = config.transport.clone().unwrap_or_default();
        let ws_config = WebSocketConfig::default()
            .max_message_size(Some(transport.max_size.unwrap_or(DEFAULT_MAX_MESSAGE_SIZE)));
        let connecting =
            tokio_tungstenite::connect_async_with_config(request, Some(ws_config), false);
        let connected = match transport.handshake_timeout {
            Some(timeout) => tokio::time::timeout(timeout, connecting)
                .await
                .map_err(|_| AgentsError::other("Realtime connection handshake timed out"))?,
            None => connecting.await,
        };
        let (stream, _response) = connected
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
                        Err(error) => {
                            log_model_action_error(
                                "Failed to parse a realtime server message as JSON",
                                &error,
                            );
                            // A bad frame is reported, not fatal: the session carries on, as
                            // Python's `error` event does.
                            reader_inner.emit(RealtimeModelEvent::Error {
                                error: json!({
                                    "message": error.to_string(),
                                    "context": "Failed to parse WebSocket message as JSON",
                                }),
                            });
                        }
                    },
                    Ok(Message::Pong(_)) => {
                        *reader_inner
                            .last_pong
                            .lock()
                            .expect("realtime model poisoned") = Instant::now();
                    }
                    Ok(Message::Close(Some(frame)))
                        if !matches!(u16::from(frame.code), 1000 | 1001) =>
                    {
                        // Only a normal or going-away close is a clean end of the session.
                        reader_inner.emit(RealtimeModelEvent::Exception {
                            message: format!(
                                "WebSocket closed with code {}: {}",
                                u16::from(frame.code),
                                frame.reason
                            ),
                            context: Some("WebSocket connection closed unexpectedly".to_string()),
                        });
                    }
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

        if let Some(interval) = transport.ping_interval {
            let keepalive_inner = self.inner.clone();
            let ping_timeout = transport.ping_timeout;
            let keepalive = tokio::spawn(async move {
                keepalive_inner.keepalive(interval, ping_timeout).await;
            });
            *self
                .inner
                .keepalive
                .lock()
                .expect("realtime model poisoned") = Some(keepalive);
        }

        if let Err(error) = self.inner.send_session_update(&settings).await {
            // A connection that never accepted its session must not keep the model "connected".
            let _ = self.close().await;
            return Err(error);
        }
        Ok(rx)
    }
}

#[async_trait]
impl RealtimeModel for OpenAIRealtimeWebSocketModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        // Claim the model before the first suspension point, so two concurrent attempts cannot
        // both pass the "already connected" check and open a socket each.
        if self.inner.connecting.swap(true, Ordering::SeqCst) {
            return Err(AgentsError::user(
                "Already connected: the realtime model is already connecting.",
            ));
        }
        let _slot = ConnectSlot(&self.inner.connecting);
        self.connect_inner(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        match event {
            RealtimeModelSendEvent::RawMessage { message } => {
                // A caller-supplied `response.create` takes part in the same sequencing as the
                // ones the transport sends itself, or a follow-up input would start a second
                // response while this one is still being created.
                if let Err(problem) = validate_client_event(&message) {
                    // Python drops a raw message that is not a client event and carries on.
                    log_model_action_error("Failed to convert raw message", &message);
                    crate::logger::log_model_action_debug("Raw message rejected", &problem);
                    return Ok(());
                }
                if message.get("type").and_then(Value::as_str) == Some("response.create") {
                    self.inner.request_response_create(Some(message)).await
                } else {
                    self.inner.send_raw(message).await
                }
            }
            RealtimeModelSendEvent::UserInput { user_input } => {
                self.inner
                    .send_raw(json!({
                        "type": "conversation.item.create",
                        "item": user_input.to_conversation_item(),
                    }))
                    .await?;
                self.inner.request_response_create(None).await
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
                        extra: Map::new(),
                    }));
                if start_response {
                    self.inner.request_response_create(None).await?;
                }
                Ok(())
            }
            RealtimeModelSendEvent::Interrupt(interrupt) => {
                self.inner.send_interrupt(&interrupt).await
            }
            RealtimeModelSendEvent::SessionUpdate { session_settings } => {
                self.inner.send_session_update(&session_settings).await
            }
        }
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> Result<bool> {
        // Only user input is safe to recheck here: it is the one event whose payload does not
        // depend on transport state captured before the await.
        let RealtimeModelSendEvent::UserInput { user_input } = &event else {
            return Ok(false);
        };
        // A deferred `response.create` means the model has not answered the previous input yet.
        let pending = {
            let state = self.inner.state.lock().expect("realtime model poisoned");
            state.pending_response_create || state.create_in_flight.is_some()
        };
        if pending || !condition() {
            return Ok(false);
        }
        let message = json!({
            "type": "conversation.item.create",
            "item": user_input.to_conversation_item(),
        });
        if !self.inner.send_raw_if(message, condition.as_ref()).await? {
            return Ok(false);
        }
        self.inner.request_response_create(None).await?;
        Ok(true)
    }

    async fn retire_response_audio(&self, response_id: &str) -> Result<()> {
        self.inner.retire_response_audio_state(response_id);
        Ok(())
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
        if let Some(keepalive) = self
            .inner
            .keepalive
            .lock()
            .expect("realtime model poisoned")
            .take()
        {
            keepalive.abort();
        }
        *self.inner.events.lock().expect("realtime model poisoned") = None;
        if let Some(tracker) = self
            .inner
            .playback_tracker
            .lock()
            .expect("realtime model poisoned")
            .take()
        {
            // A reused model must not report the closed session's item on its next connection.
            tracker.on_interrupted();
        }
        let mut state = self.inner.state.lock().expect("realtime model poisoned");
        state.audio = ModelAudioTracker::default();
        state.interrupted_audio_response_ids.clear();
        state.ongoing_response = false;
        state.ongoing_response_id = None;
        state.cancel_requested = false;
        state.current_item_id = None;
        state.pending_response_create = false;
        state.create_in_flight = None;
        state.create_payload = None;
        Ok(())
    }
}

/// A realtime transport that attaches to an existing SIP-originated call.
///
/// The Realtime Calls API hands your webhook a `call_id`. Accept the call by POSTing a session
/// payload to `/v1/realtime/calls/{call_id}/accept`, then attach this transport with the same
/// `call_id`: it connects to `wss://api.openai.com/v1/realtime?call_id=...` instead of starting a
/// new session, and leaves the audio formats to the ones the call negotiated.
///
/// Build the accept payload with
/// [`OpenAIRealtimeSIPModel::build_initial_session_payload`] so the call starts with the same
/// instructions, tools and handoffs the session would later send.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openai_agents_sdk::realtime::{
/// #     OpenAIRealtimeSIPModel, RealtimeAgent, RealtimeModelConfig, RealtimeRunner,
/// # };
/// # async fn demo(call_id: &str) -> Result<(), Box<dyn std::error::Error>> {
/// let agent = RealtimeAgent::<()>::new("Assistant")
///     .with_instructions("You answer the phone politely")
///     .into_arc();
///
/// // 1. Accept the call with the agent's own session payload.
/// let payload =
///     OpenAIRealtimeSIPModel::build_initial_session_payload(&agent, (), None, None, None).await?;
/// // POST `payload` to /v1/realtime/calls/{call_id}/accept with your HTTP client of choice.
/// assert_eq!(payload["type"], "realtime");
///
/// // 2. Attach to the accepted call.
/// let session = RealtimeRunner::new(agent)
///     .with_model(Arc::new(OpenAIRealtimeSIPModel::new()))
///     .with_model_config(RealtimeModelConfig::new().with_call_id(call_id))
///     .run(())
///     .await?;
/// # let _ = session;
/// # Ok(())
/// # }
/// ```
pub struct OpenAIRealtimeSIPModel {
    inner: OpenAIRealtimeWebSocketModel,
}

impl Default for OpenAIRealtimeSIPModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for OpenAIRealtimeSIPModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIRealtimeSIPModel")
            .field("model", &self.inner.model_name())
            .finish_non_exhaustive()
    }
}

impl OpenAIRealtimeSIPModel {
    /// A disconnected SIP transport.
    pub fn new() -> Self {
        OpenAIRealtimeSIPModel {
            inner: OpenAIRealtimeWebSocketModel::new(),
        }
    }

    /// The realtime model name this transport reports.
    pub fn model_name(&self) -> String {
        self.inner.model_name()
    }

    /// Build the session payload that accepts a SIP call, mirroring what a session would send.
    ///
    /// `overrides` is applied last, after the run config, the connect-time settings and the
    /// agent's own instructions, tools and handoffs. Port of `build_initial_session_payload`.
    pub async fn build_initial_session_payload<C: Send + Sync + 'static>(
        agent: &Arc<RealtimeAgent<C>>,
        context: C,
        model_config: Option<&RealtimeModelConfig>,
        run_config: Option<&RealtimeRunConfig<C>>,
        overrides: Option<&RealtimeSessionModelSettings>,
    ) -> Result<Value> {
        let starting = model_config.and_then(|config| config.initial_model_settings.clone());
        let mut base = run_config
            .and_then(|config| config.model_settings.clone())
            .unwrap_or_default();
        if let Some(starting) = &starting {
            base.merge(starting);
        }

        let run_context = RunContext::new(context);
        let mut resolved = build_model_settings_from_agent(
            agent,
            &run_context,
            &base,
            starting.as_ref(),
            run_config.is_some_and(|config| config.tracing_disabled),
        )
        .await?;
        if let Some(overrides) = overrides {
            resolved.settings.merge(overrides);
        }

        ensure_tools_supported(&resolved.settings)?;
        // A fresh transport, so the payload reflects only the settings, never a live connection.
        Ok(OpenAIRealtimeWebSocketModel::new().session_payload_for_settings(&resolved.settings))
    }
}

#[async_trait]
impl RealtimeModel for OpenAIRealtimeSIPModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        if config.call_id.as_deref().unwrap_or_default().is_empty() {
            return Err(AgentsError::user(
                "OpenAIRealtimeSIPModel requires `call_id` in the model configuration.",
            ));
        }
        self.inner.connect(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        self.inner.send_event(event).await
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> Result<bool> {
        self.inner.send_event_if(event, condition).await
    }

    async fn retire_response_audio(&self, response_id: &str) -> Result<()> {
        self.inner.retire_response_audio(response_id).await
    }

    async fn close(&self) -> Result<()> {
        self.inner.close().await
    }
}

impl WsInner {
    /// Send keepalive pings, and give up when the peer stops answering.
    ///
    /// `tokio-tungstenite` has no built-in keepalive, so this is the equivalent of the Python
    /// client's `ping_interval` / `ping_timeout`.
    async fn keepalive(&self, interval: Duration, timeout: Option<Duration>) {
        *self.last_pong.lock().expect("realtime model poisoned") = Instant::now();
        loop {
            tokio::time::sleep(interval).await;
            if self.close_requested.load(Ordering::SeqCst) {
                return;
            }
            if self
                .send_message(Message::Ping(Vec::new().into()))
                .await
                .is_err()
            {
                return;
            }
            if let Some(timeout) = timeout {
                let since_pong = {
                    self.last_pong
                        .lock()
                        .expect("realtime model poisoned")
                        .elapsed()
                };
                if since_pong > interval + timeout {
                    log::warn!(target: crate::logger::LOG_TARGET, "Realtime keepalive timed out; closing the connection");
                    self.emit(RealtimeModelEvent::ConnectionStatus {
                        status: RealtimeConnectionStatus::Disconnected,
                    });
                    self.emit(RealtimeModelEvent::EndOfStream);
                    if let Some(mut sink) = self.sink.lock().await.take() {
                        let _ = sink.close().await;
                    }
                    return;
                }
            }
        }
    }

    /// Write one frame to the socket.
    async fn send_message(&self, message: Message) -> Result<()> {
        let mut guard = self.sink.lock().await;
        let sink = guard
            .as_mut()
            .ok_or_else(|| AgentsError::user("The realtime model is not connected."))?;
        sink.send(message)
            .await
            .map_err(|e| AgentsError::other(format!("Realtime send failed: {e}")))
    }

    fn emit(&self, event: RealtimeModelEvent) {
        let sender = { self.events.lock().expect("realtime model poisoned").clone() };
        if let Some(sender) = sender {
            let _ = sender.send(event);
        }
    }

    async fn send_raw(&self, message: Value) -> Result<()> {
        self.send_message(Message::text(message.to_string())).await
    }

    /// Recheck `condition` at the write boundary, so the decision and the send are atomic.
    async fn send_raw_if(
        &self,
        message: Value,
        condition: &(dyn Fn() -> bool + Send + Sync),
    ) -> Result<bool> {
        let payload = message.to_string();
        let mut guard = self.sink.lock().await;
        let sink = guard
            .as_mut()
            .ok_or_else(|| AgentsError::user("The realtime model is not connected."))?;
        if !condition() {
            return Ok(false);
        }
        sink.send(Message::text(payload))
            .await
            .map_err(|e| AgentsError::other(format!("Realtime send failed: {e}")))?;
        Ok(true)
    }

    async fn send_session_update(&self, settings: &RealtimeSessionModelSettings) -> Result<()> {
        ensure_tools_supported(settings)?;
        let session = self.session_payload(settings);
        self.send_raw(json!({"type": "session.update", "session": session}))
            .await
    }

    /// Record the audio format the session negotiated on both audio trackers.
    fn set_audio_format(&self, format: &str) {
        self.state
            .lock()
            .expect("realtime model poisoned")
            .audio
            .set_audio_format(format);
        if let Some(tracker) = self
            .playback_tracker
            .lock()
            .expect("realtime model poisoned")
            .as_deref()
        {
            tracker.set_audio_format(format);
        }
    }

    /// Build the `session` object of a `session.update`, mirroring `_get_session_config`.
    ///
    /// The nested `audio.input.*` / `audio.output.*` settings win over the flat legacy keys, which
    /// in turn win over the SDK defaults — except on an attached call, which already negotiated
    /// its audio formats and must not have them overridden.
    fn session_payload(&self, settings: &RealtimeSessionModelSettings) -> Value {
        let (model, call_id) = {
            let state = self.state.lock().expect("realtime model poisoned");
            (state.model.clone(), state.call_id.clone())
        };
        let model = settings.model_name.clone().unwrap_or(model);
        let audio_input = settings
            .audio
            .as_ref()
            .and_then(|audio| audio.input.as_ref());
        let audio_output = settings
            .audio
            .as_ref()
            .and_then(|audio| audio.output.as_ref());

        let mut input = Map::new();
        let input_format = audio_input
            .and_then(|input| input.format.clone())
            .or_else(|| settings.input_audio_format.clone())
            .or_else(|| {
                call_id
                    .is_none()
                    .then(|| RealtimeAudioFormat::Name("pcm16".to_string()))
            });
        if let Some(value) = input_format.as_ref().and_then(|f| f.to_session_value()) {
            input.insert("format".to_string(), value);
        }

        let noise_reduction = audio_input
            .and_then(|input| input.noise_reduction.clone())
            .or_else(|| settings.input_audio_noise_reduction.clone());
        match noise_reduction {
            Some(Some(config)) => {
                input.insert(
                    "noise_reduction".to_string(),
                    serde_json::to_value(&config).unwrap_or(Value::Null),
                );
            }
            Some(None) => {
                input.insert("noise_reduction".to_string(), Value::Null);
            }
            None => {}
        }

        let transcription = audio_input
            .and_then(|input| input.transcription.clone())
            .or_else(|| settings.input_audio_transcription.clone())
            .unwrap_or_else(|| {
                RealtimeInputAudioTranscriptionConfig::new(DEFAULT_INPUT_AUDIO_TRANSCRIPTION_MODEL)
            });
        input.insert(
            "transcription".to_string(),
            serde_json::to_value(&transcription).unwrap_or(Value::Null),
        );

        let turn_detection = audio_input
            .and_then(|input| input.turn_detection.clone())
            .or_else(|| settings.turn_detection.clone())
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
        let voice = audio_output
            .and_then(|output| output.voice.clone())
            .or_else(|| settings.voice.clone())
            .unwrap_or_else(|| RealtimeVoice::Name(DEFAULT_REALTIME_VOICE.to_string()));
        output.insert("voice".to_string(), voice.to_session_value());

        let output_format = audio_output
            .and_then(|output| output.format.clone())
            .or_else(|| settings.output_audio_format.clone())
            .or_else(|| {
                call_id
                    .is_none()
                    .then(|| RealtimeAudioFormat::Name("pcm16".to_string()))
            });
        if let Some(value) = output_format.as_ref().and_then(|f| f.to_session_value()) {
            output.insert("format".to_string(), value);
        }
        if let Some(speed) = audio_output
            .and_then(|output| output.speed)
            .or(settings.speed)
        {
            output.insert("speed".to_string(), json!(speed));
        }

        // `modalities` is the legacy name, used only when `output_modalities` is unset.
        let modalities = settings
            .output_modalities
            .clone()
            .or_else(|| settings.modalities.clone())
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
            session.insert("reasoning".to_string(), reasoning.to_session_value());
        }
        Value::Object(session)
    }

    /// Ask the model for a new response, deferring while one is already running.
    async fn request_response_create(&self, payload: Option<Value>) -> Result<()> {
        let event_id = {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .request_response_create(payload)
        };
        match event_id {
            Some(event_id) => self.send_response_create(event_id).await,
            None => Ok(()),
        }
    }

    /// Tell the server how much of an item's audio the client actually played.
    async fn send_truncate(
        &self,
        item_id: &str,
        content_index: usize,
        audio_end_ms: i64,
    ) -> Result<()> {
        self.send_raw(json!({
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }))
        .await
    }

    async fn send_response_create(&self, event_id: String) -> Result<()> {
        let payload = self
            .state
            .lock()
            .expect("realtime model poisoned")
            .create_payload
            .take();
        let mut message = payload.unwrap_or_else(|| json!({"type": "response.create"}));
        message["event_id"] = Value::String(event_id);
        // Dropping this future mid-send (a cancelled caller) must free the slot as well as an
        // error does; the guard is disarmed only once the send has been accepted.
        let mut slot = SlotGuard::new(self, SlotKind::Create);
        let result = self.send_raw(message).await;
        if result.is_ok() {
            slot.disarm();
        }
        result
    }

    /// Stop audio playback, truncating the item at the point the user actually heard.
    ///
    /// Port of `_interrupt_audio_playback`. A response-scoped interruption only touches audio that
    /// response produced, so a guardrail tripping on an older response cannot cut off a newer one.
    async fn interrupt_audio_playback(&self, interrupt: &RealtimeInterrupt) -> Result<()> {
        let playback_tracker = {
            self.playback_tracker
                .lock()
                .expect("realtime model poisoned")
                .clone()
        };
        let response_scoped = interrupt.response_id.as_deref();

        let mut source_items: Vec<(String, usize)> = Vec::new();
        if let Some(response_id) = response_scoped {
            source_items = {
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .audio
                    .audio_items_for_response(response_id)
            };
            if source_items.is_empty() {
                // This response produced no audio the transport still owns.
                return Ok(());
            }
            for (item_id, content_index) in &source_items {
                self.emit(RealtimeModelEvent::AudioInterrupted {
                    item_id: item_id.clone(),
                    content_index: *content_index,
                });
            }
        }

        let mut position = {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .playback_position(playback_tracker.as_deref())
        };
        if let Some(response_id) = response_scoped
            && !source_items.contains(&position.audio_item().unwrap_or_default())
        {
            // Playback may have moved on between the two reads; look once more before giving up.
            position = {
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .playback_position(playback_tracker.as_deref())
            };
            if !source_items.contains(&position.audio_item().unwrap_or_default()) {
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .audio
                    .on_response_interrupted(response_id);
                return Ok(());
            }
        }

        let interrupted_item = match (position.item_id.clone(), position.elapsed_ms) {
            (Some(item_id), Some(elapsed_ms)) => Some((item_id, elapsed_ms)),
            _ => {
                log::debug!(target: crate::logger::LOG_TARGET,
                    "Skipping realtime interrupt; item id {:?}, elapsed ms {:?}",
                    position.item_id,
                    position.elapsed_ms
                );
                None
            }
        };

        let mut truncate_error = None;
        if let Some((item_id, elapsed_ms)) = interrupted_item.clone() {
            let content_index = position.content_index.unwrap_or_default();
            if elapsed_ms >= 0.0 {
                if response_scoped.is_none() {
                    self.emit(RealtimeModelEvent::AudioInterrupted {
                        item_id: item_id.clone(),
                        content_index,
                    });
                }
                let (max_audio_ms, owns_audio) = {
                    let state = self.state.lock().expect("realtime model poisoned");
                    (
                        state.max_audio_ms(&item_id, content_index),
                        state.ongoing_response_owns_audio(&item_id, content_index),
                    )
                };
                let mut truncated_ms = (elapsed_ms as i64).max(0);
                let should_truncate = (!interrupt.playback_only && owns_audio)
                    || max_audio_ms.is_none_or(|max_ms| truncated_ms < max_ms);
                if should_truncate {
                    if let Some(max_ms) = max_audio_ms {
                        truncated_ms = truncated_ms.min(max_ms);
                    }
                    truncate_error = self
                        .send_truncate(&item_id, content_index, truncated_ms)
                        .await
                        .err();
                } else {
                    log::debug!(target: crate::logger::LOG_TARGET,
                        "Skipping realtime truncate; playback of {item_id} appears complete"
                    );
                }
            } else {
                log::debug!(target: crate::logger::LOG_TARGET, "Skipping realtime truncate; elapsed ms is negative");
            }
        }

        if interrupted_item.is_some() {
            {
                let mut state = self.state.lock().expect("realtime model poisoned");
                match response_scoped {
                    Some(response_id) => state.audio.on_response_interrupted(response_id),
                    None => state.audio.on_interrupted(),
                }
            }
            if let Some(tracker) = playback_tracker.as_deref() {
                let latest = tracker.state().current_item_id.map(|item_id| {
                    (
                        item_id,
                        tracker
                            .state()
                            .current_item_content_index
                            .unwrap_or_default(),
                    )
                });
                let still_ours = response_scoped.is_none()
                    || latest.is_some_and(|item| source_items.contains(&item));
                if still_ours {
                    tracker.on_interrupted();
                }
            }
        }

        match truncate_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn send_interrupt(&self, interrupt: &RealtimeInterrupt) -> Result<()> {
        interrupt.validate()?;

        if interrupt.cancel_response_only {
            return self.cancel_response(interrupt.response_id.as_deref()).await;
        }

        if let Some(response_id) = &interrupt.response_id {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .interrupted_audio_response_ids
                .insert(response_id.clone());
        }

        let playback_error = self.interrupt_audio_playback(interrupt).await.err();

        if !interrupt.playback_only {
            let automatic = {
                self.state
                    .lock()
                    .expect("realtime model poisoned")
                    .automatic_response_cancellation
            };
            // Without server-side cancellation the response keeps producing audio, so stop it
            // here; `cancel_response` decides whether a cancel may actually go out.
            if interrupt.force_response_cancel || !automatic {
                let cancel_error = self
                    .cancel_response(interrupt.response_id.as_deref())
                    .await
                    .err();
                if let Some(error) = playback_error.or(cancel_error) {
                    return Err(error);
                }
                return Ok(());
            }
        }

        match playback_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn cancel_response(&self, response_id: Option<&str>) -> Result<()> {
        let allowed = {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .begin_cancel(response_id)
        };
        if !allowed {
            return Ok(());
        }
        let mut message = json!({"type": "response.cancel"});
        if let Some(response_id) = response_id {
            message["response_id"] = Value::String(response_id.to_string());
        }
        let mut slot = SlotGuard::new(self, SlotKind::Cancel);
        let result = self.send_raw(message).await;
        if result.is_ok() {
            slot.disarm();
        }
        result
    }

    /// Release the audio bookkeeping held for a response once guardrails have settled.
    fn retire_response_audio_state(&self, response_id: &str) {
        let mut state = self.state.lock().expect("realtime model poisoned");
        state.interrupted_audio_response_ids.remove(response_id);
        state.audio.on_response_done(response_id);
    }

    /// Translate one server event into model events, reacting where the protocol requires it.
    async fn handle_server_event(&self, event: Value) {
        self.emit(RealtimeModelEvent::RawServerEvent {
            data: event.clone(),
        });
        // A message item is converted before validation, so a provider that omits the envelope
        // fields around it still updates the history.
        let converted_without_validation = matches!(
            event.get("type").and_then(Value::as_str),
            Some("response.output_item.added" | "response.output_item.done")
        ) && event.pointer("/item/type").and_then(Value::as_str)
            == Some("message");
        if !converted_without_validation && let Err(problem) = validate_server_event(&event) {
            log_model_action_error("Failed to validate server event", &event);
            self.emit(RealtimeModelEvent::Error {
                error: json!({
                    "message": problem,
                    "context": "Failed to validate server event",
                }),
            });
            return;
        }
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .expect("a validated event has a type");

        match event_type {
            "response.output_audio.delta" => {
                let item_id = string_field(&event, "item_id");
                let content_index = usize_field(&event, "content_index");
                let response_id = string_field(&event, "response_id");
                if self
                    .state
                    .lock()
                    .expect("realtime model poisoned")
                    .interrupted_audio_response_ids
                    .contains(&response_id)
                {
                    return;
                }
                let Ok(data) = BASE64.decode(string_field(&event, "delta")) else {
                    log_model_action_error(
                        "Failed to decode a realtime audio delta",
                        &"invalid base64",
                    );
                    self.emit(RealtimeModelEvent::Exception {
                        message: "Invalid base64 audio delta".to_string(),
                        context: Some("response.output_audio.delta".to_string()),
                    });
                    return;
                };
                {
                    let mut state = self.state.lock().expect("realtime model poisoned");
                    state.current_item_id = Some(item_id.clone());
                    state
                        .audio
                        .on_audio_delta(&item_id, content_index, &data, Some(&response_id));
                }
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
                    .on_response_created(response_id.as_deref());
                self.emit(RealtimeModelEvent::TurnStarted { response_id });
            }
            "response.done" => {
                let response_id = event
                    .pointer("/response/id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let deferred = {
                    let mut state = self.state.lock().expect("realtime model poisoned");
                    if let Some(response_id) = &response_id {
                        state.interrupted_audio_response_ids.remove(response_id);
                    }
                    state.on_response_done(response_id.as_deref())
                };
                if let Some(reported) = event.pointer("/response/usage")
                    && let Some(converted) = convert_usage(reported)
                {
                    self.emit(converted);
                }
                self.emit(RealtimeModelEvent::TurnEnded { response_id });
                if let Some(event_id) = deferred {
                    let _ = self.send_response_create(event_id).await;
                }
            }
            "session.created" | "session.updated" => {
                if let Some(format) = event
                    .pointer("/session/audio/output/format")
                    .and_then(read_format_type)
                {
                    self.set_audio_format(&format);
                }
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
                    // An item the server reports without a status is still being built, which is
                    // what Python's `item.status or "in_progress"` encodes. Leaving it unset
                    // would make a fresh item indistinguishable from one whose status the
                    // transport could not read.
                    && let Some(message) =
                        parse_message_item(item, previous_item_id, Some("in_progress"))
                {
                    self.emit(RealtimeModelEvent::item_updated(message));
                }
            }
            "conversation.item.input_audio_transcription.completed" => {
                // Both of these change the item server side, so re-read it rather than let the
                // local history drift.
                self.retrieve_current_item().await;
                self.emit(RealtimeModelEvent::InputAudioTranscriptionCompleted {
                    item_id: string_field(&event, "item_id"),
                    transcript: string_field(&event, "transcript"),
                });
            }
            "conversation.item.truncated" => {
                self.retrieve_current_item().await;
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
                // Barge-in: stop local playback at once so the user is not talked over. The
                // server reports where its own VAD heard speech start, which beats a local
                // real-time estimate when it is available.
                self.handle_speech_started(event.get("audio_end_ms").and_then(Value::as_f64))
                    .await;
            }
            "conversation.item.input_audio_transcription.delta"
            | "response.function_call_arguments.delta" => {}
            _ => self.emit(RealtimeModelEvent::Other { data: event }),
        }
    }

    /// Ask the server to resend the item it is currently producing audio for.
    ///
    /// A transcription or a truncation rewrites that item server side, so without this the local
    /// history keeps the pre-edit copy.
    async fn retrieve_current_item(&self) {
        let item_id = {
            self.state
                .lock()
                .expect("realtime model poisoned")
                .current_item_id
                .clone()
        };
        let Some(item_id) = item_id else {
            return;
        };
        let _ = self
            .send_raw(json!({"type": "conversation.item.retrieve", "item_id": item_id}))
            .await;
    }

    /// Interrupt playback when the server's VAD hears the user start speaking.
    ///
    /// Port of the `input_audio_buffer.speech_started` branch, which differs from a requested
    /// interrupt: it reports the tracker's last audio item, prefers the server's `audio_end_ms`
    /// over a local estimate, and rounds rather than floors.
    async fn handle_speech_started(&self, audio_end_ms: Option<f64>) {
        let playback_tracker = {
            self.playback_tracker
                .lock()
                .expect("realtime model poisoned")
                .clone()
        };
        let (last_audio, position, automatic) = {
            let state = self.state.lock().expect("realtime model poisoned");
            (
                state.audio.last_audio_item(),
                state.playback_position(playback_tracker.as_deref()),
                state.automatic_response_cancellation,
            )
        };
        let Some((item_id, content_index)) = last_audio else {
            return;
        };

        self.emit(RealtimeModelEvent::AudioInterrupted {
            item_id,
            content_index,
        });

        let effective_elapsed_ms = match audio_end_ms {
            Some(server_ms) if server_ms > 0.0 => Some(server_ms),
            _ => position.elapsed_ms,
        };
        if let (Some(playback_item_id), Some(elapsed_ms)) =
            (position.item_id.as_deref(), effective_elapsed_ms)
        {
            let playback_content_index = position.content_index.unwrap_or_default();
            let (max_audio_ms, owns_audio) = {
                let state = self.state.lock().expect("realtime model poisoned");
                (
                    state.max_audio_ms(playback_item_id, playback_content_index),
                    state.ongoing_response_owns_audio(playback_item_id, playback_content_index),
                )
            };
            let mut truncated_ms = (elapsed_ms.round() as i64).max(0);
            let playback_complete =
                max_audio_ms.is_some_and(|max_ms| truncated_ms >= max_ms) && !owns_audio;
            if playback_complete {
                log::debug!(target: crate::logger::LOG_TARGET,
                    "Skipping realtime truncate; playback of {playback_item_id} appears complete"
                );
            } else {
                if let Some(max_ms) = max_audio_ms {
                    truncated_ms = truncated_ms.min(max_ms);
                }
                let _ = self
                    .send_truncate(playback_item_id, playback_content_index, truncated_ms)
                    .await;
            }
        }

        // Reset both trackers so later playback queries do not reference interrupted audio.
        self.state
            .lock()
            .expect("realtime model poisoned")
            .audio
            .on_interrupted();
        if let Some(tracker) = playback_tracker.as_deref() {
            tracker.on_interrupted();
        }

        if !automatic {
            // The server will not stop on its own, so stop it here.
            let _ = self.cancel_response(None).await;
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
                    extra: Map::new(),
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

/// The URL to connect to: an explicit override, then an existing call, then a new session.
/// Reject tools that use a feature only the Responses API supports, as Python's
/// `_tools_to_session_tools` does before it builds the session payload.
fn ensure_tools_supported(settings: &RealtimeSessionModelSettings) -> Result<()> {
    for spec in settings.tools.as_deref().unwrap_or_default() {
        spec.ensure_realtime_supported()?;
    }
    Ok(())
}

fn connect_url(url: Option<&str>, call_id: Option<&str>, model: &str) -> String {
    match (url, call_id) {
        (Some(url), _) => url.to_string(),
        (None, Some(call_id)) => format!("{DEFAULT_REALTIME_URL}?call_id={call_id}"),
        (None, None) => format!("{DEFAULT_REALTIME_URL}?model={model}"),
    }
}

/// The API key for a realtime connection: explicit, then the SDK default, then the environment.
///
/// An explicit key may be a provider resolved at connect time, so a short-lived key can be
/// refreshed without rebuilding the model.
/// Resolve the API key a realtime connection uses: the explicit key or key function, then the
/// SDK default key, then `OPENAI_API_KEY`.
///
/// Port of `agents.realtime.openai_realtime.get_api_key`, which this crate's
/// [`RealtimeApiKey`] generalizes to a literal or a function.
pub async fn get_api_key(explicit: Option<&RealtimeApiKey>) -> Result<String> {
    if let Some(key) = explicit {
        return key.resolve().await;
    }
    crate::models::default_openai_key()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| AgentsError::user("API key is required but was not provided."))
}

/// The format name of a GA `audio.format` value, in either its object or string spelling.
fn read_format_type(format: &Value) -> Option<String> {
    let name = match format {
        Value::String(name) => Some(name.clone()),
        Value::Object(_) => format
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }?;
    // The GA object spellings name the same formats as the legacy strings; the trackers use the
    // legacy names, as Python's `_normalize_audio_format` does for its typed formats.
    Some(match format {
        Value::Object(_) => match name.as_str() {
            "audio/pcm" => "pcm16".to_string(),
            "audio/pcmu" => "g711_ulaw".to_string(),
            "audio/pcma" => "g711_alaw".to_string(),
            _ => name,
        },
        _ => name,
    })
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

/// The fields of `value` that this port does not model, kept so they survive a round trip, as
/// Python's `extra="allow"` does.
fn extra_fields(value: &Value, known: &[&str]) -> Map<String, Value> {
    value
        .as_object()
        .map(|fields| {
            fields
                .iter()
                .filter(|(key, _)| !known.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
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
        // Keep the part's unmodelled fields, as Python's `extra="allow"` does on content parts.
        let part_extra = |known: &[&str]| extra_fields(part, known);
        match part.get("type").and_then(Value::as_str) {
            Some("audio") | Some("output_audio") => content.push(RealtimeContent::Audio {
                audio,
                transcript,
                extra: part_extra(&["type", "audio", "transcript"]),
            }),
            Some("input_audio") => content.push(RealtimeContent::InputAudio {
                audio,
                transcript,
                extra: part_extra(&["type", "audio", "transcript"]),
            }),
            Some("text") | Some("output_text") => content.push(RealtimeContent::Text {
                text,
                extra: part_extra(&["type", "text"]),
            }),
            Some("input_text") => content.push(RealtimeContent::InputText {
                text,
                extra: part_extra(&["type", "text"]),
            }),
            Some("input_image") => content.push(RealtimeContent::InputImage {
                image_url: part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                detail: part
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                extra: part_extra(&["type", "image_url", "detail"]),
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

    // Keep whatever else the server sent, so a field this port does not model survives the round
    // trip, as Python's `extra="allow"` does.
    let extra = extra_fields(item, &["id", "type", "role", "status", "content", "object"]);

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
        extra,
    }))
}

/// Read an optional token count from a usage detail object.
fn detail_tokens(details: Option<&Value>, key: &str) -> Option<u64> {
    details?.get(key).and_then(Value::as_u64)
}

/// Convert a realtime `response.usage` object into a usage event.
///
/// The shared [`Usage`] carries the totals; the modality breakdown the realtime API reports is
/// kept beside it rather than flattened away.
fn convert_usage(usage: &Value) -> Option<RealtimeModelEvent> {
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
    let input_details = usage.get("input_token_details");
    let output_details = usage.get("output_token_details");
    if let Some(cached) = detail_tokens(input_details, "cached_tokens") {
        converted.input_tokens_details.cached_tokens = cached;
    }

    let cached_details = input_details.and_then(|details| details.get("cached_tokens_details"));
    Some(RealtimeModelEvent::Usage {
        usage: Box::new(converted),
        input_tokens_details: input_details.map(|details| RealtimeModelInputTokensDetails {
            text_tokens: detail_tokens(Some(details), "text_tokens"),
            audio_tokens: detail_tokens(Some(details), "audio_tokens"),
            image_tokens: detail_tokens(Some(details), "image_tokens"),
            cached_tokens: detail_tokens(Some(details), "cached_tokens"),
            cached_tokens_details: cached_details.map(|cached| RealtimeModelCachedTokensDetails {
                text_tokens: detail_tokens(Some(cached), "text_tokens"),
                audio_tokens: detail_tokens(Some(cached), "audio_tokens"),
                image_tokens: detail_tokens(Some(cached), "image_tokens"),
            }),
        }),
        output_tokens_details: output_details.map(|details| RealtimeModelOutputTokensDetails {
            text_tokens: detail_tokens(Some(details), "text_tokens"),
            audio_tokens: detail_tokens(Some(details), "audio_tokens"),
        }),
    })
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
            connect_url(None, None, &model.model_name()),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
        );
    }

    #[test]
    fn a_call_id_attaches_to_an_existing_call_instead_of_a_new_session() {
        assert_eq!(
            connect_url(None, Some("call_789"), DEFAULT_REALTIME_MODEL),
            "wss://api.openai.com/v1/realtime?call_id=call_789"
        );
        // An explicit URL always wins, which is what the tests and proxies rely on.
        assert_eq!(
            connect_url(Some("ws://127.0.0.1:9/x"), Some("call_789"), "m"),
            "ws://127.0.0.1:9/x"
        );
    }

    #[test]
    fn a_call_leaves_the_audio_formats_to_the_negotiated_session() {
        let model = OpenAIRealtimeWebSocketModel::new();
        model.inner.state.lock().expect("poisoned").call_id = Some("call_1".to_string());
        let payload = model
            .inner
            .session_payload(&RealtimeSessionModelSettings::new());
        assert!(
            payload["audio"]["input"].get("format").is_none(),
            "a call must not override the format it negotiated"
        );
        assert!(payload["audio"]["output"].get("format").is_none());

        // An explicitly configured format is still sent.
        let payload = model.inner.session_payload(&RealtimeSessionModelSettings {
            input_audio_format: Some(RealtimeAudioFormat::from("g711_ulaw")),
            ..Default::default()
        });
        assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcmu");
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
    fn converts_usage_with_its_modality_breakdown() {
        let event = convert_usage(&json!({
            "input_tokens": 10,
            "output_tokens": 4,
            "input_token_details": {
                "cached_tokens": 2,
                "text_tokens": 6,
                "audio_tokens": 4,
                "cached_tokens_details": {"text_tokens": 1, "audio_tokens": 1},
            },
            "output_token_details": {"text_tokens": 1, "audio_tokens": 3},
        }))
        .unwrap();
        let RealtimeModelEvent::Usage {
            usage,
            input_tokens_details,
            output_tokens_details,
        } = event
        else {
            panic!("expected a usage event");
        };
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.total_tokens, 14);
        assert_eq!(usage.input_tokens_details.cached_tokens, 2);

        let input = input_tokens_details.expect("input details");
        assert_eq!(input.text_tokens, Some(6));
        assert_eq!(input.audio_tokens, Some(4));
        assert_eq!(input.image_tokens, None);
        let cached = input.cached_tokens_details.expect("cached details");
        assert_eq!(cached.audio_tokens, Some(1));

        let output = output_tokens_details.expect("output details");
        assert_eq!(output.audio_tokens, Some(3));

        // A response that reports no breakdown still yields totals.
        let event = convert_usage(&json!({"input_tokens": 1})).unwrap();
        let RealtimeModelEvent::Usage {
            input_tokens_details,
            ..
        } = event
        else {
            panic!("expected a usage event");
        };
        assert!(input_tokens_details.is_none());
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
            .request_response_create(None)
            .expect("first create is sent");
        // Parallel tool outputs both see an idle connection until `response.created` lands.
        assert_eq!(state.request_response_create(None), None);
        assert!(state.pending_response_create);

        state.on_response_created(Some("resp_1"));
        assert_eq!(state.create_in_flight, None);
        assert!(state.ongoing_response);

        let second = state
            .on_response_done(None)
            .expect("deferred create is flushed");
        assert_ne!(first, second);
        assert!(!state.pending_response_create);
        assert_eq!(state.on_response_done(None), None);
    }

    #[test]
    fn an_unrelated_error_keeps_a_deferred_create() {
        let mut state = state();
        let event_id = state.request_response_create(None).unwrap();
        assert_eq!(state.request_response_create(None), None);

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
        state.request_response_create(None).unwrap();
        assert!(state.error_rejects_pending_create(&json!({"code": "bad_response_create"})));
        assert!(state.error_rejects_pending_create(
            &json!({"message": "Cannot send response.create right now"})
        ));
        assert!(!state.error_rejects_pending_create(&json!({"message": "input buffer too small"})));

        // An error arriving while a response runs is never a create rejection.
        state.on_response_created(Some("resp_1"));
        assert!(!state.error_rejects_pending_create(&json!({"code": "bad_response_create"})));
    }

    #[test]
    fn a_cancel_is_only_sent_for_the_active_response() {
        let mut state = state();

        // Nothing to cancel before a response starts.
        assert!(!state.begin_cancel(None));

        state.on_response_created(Some("resp_1"));
        // A cancel aimed at a different response must stay local.
        assert!(!state.begin_cancel(Some("resp_2")));
        assert!(state.begin_cancel(Some("resp_1")));
        // A second cancel for the same response would be rejected by the server.
        assert!(!state.begin_cancel(Some("resp_1")));
        assert!(!state.begin_cancel(None));

        // A failed send releases the slot so a later attempt still goes out.
        state.release_cancel();
        assert!(state.begin_cancel(None));

        // The next response may be cancelled again.
        state.on_response_done(None);
        assert!(!state.begin_cancel(None));
        state.on_response_created(Some("resp_2"));
        assert!(state.begin_cancel(None));
    }

    #[test]
    fn response_scoped_indexes_track_audio_per_response() {
        let mut tracker = ModelAudioTracker::default();
        tracker.set_audio_format("pcm16");
        tracker.on_audio_delta("item_1", 0, &[0u8; 48], Some("resp_1"));
        tracker.on_audio_delta("item_1", 0, &[0u8; 48], Some("resp_1"));
        tracker.on_audio_delta("item_2", 0, &[0u8; 48], Some("resp_2"));

        // Each delta extends the item it belongs to.
        let state = tracker.state("item_1", 0).expect("item_1 state");
        assert!((state.audio_length_ms - 2.0).abs() < 1e-9);
        assert_eq!(state.response_id.as_deref(), Some("resp_1"));
        assert_eq!(
            tracker.audio_items_for_response("resp_1"),
            vec![("item_1".to_string(), 0)]
        );
        assert_eq!(tracker.last_audio_item(), Some(("item_2".to_string(), 0)));

        // Interrupting an older response leaves the newer one playing.
        tracker.on_response_interrupted("resp_1");
        assert!(tracker.audio_items_for_response("resp_1").is_empty());
        assert_eq!(tracker.last_audio_item(), Some(("item_2".to_string(), 0)));

        // Interrupting the response that owns the current item clears it.
        tracker.on_response_interrupted("resp_2");
        assert_eq!(tracker.last_audio_item(), None);
    }

    #[test]
    fn retiring_a_response_releases_only_its_indexes() {
        let mut tracker = ModelAudioTracker::default();
        tracker.on_audio_delta("item_1", 0, &[0u8; 48], Some("resp_1"));
        tracker.on_audio_delta("item_2", 0, &[0u8; 48], Some("resp_2"));
        tracker.on_response_done("resp_1");
        assert!(tracker.audio_items_for_response("resp_1").is_empty());
        assert_eq!(
            tracker.audio_items_for_response("resp_2"),
            vec![("item_2".to_string(), 0)]
        );
    }

    #[test]
    fn audio_ownership_decides_whether_a_finished_item_is_truncated() {
        let mut state = state();
        state.audio.set_audio_format("pcm16");
        state
            .audio
            .on_audio_delta("item_1", 0, &[0u8; 48], Some("resp_1"));
        assert_eq!(state.max_audio_ms("item_1", 0), Some(1));

        // No response running: nothing can still be extending this audio.
        assert!(!state.ongoing_response_owns_audio("item_1", 0));

        state.on_response_created(Some("resp_1"));
        assert!(state.ongoing_response_owns_audio("item_1", 0));

        // A different response is running, so the old item's audio is final.
        state.on_response_created(Some("resp_2"));
        assert!(!state.ongoing_response_owns_audio("item_1", 0));

        // Ownership is unknown for an item the tracker never saw, so interruption is preserved.
        assert!(state.ongoing_response_owns_audio("item_unknown", 0));
    }

    #[test]
    fn playback_position_prefers_the_application_tracker() {
        let mut state = state();
        state.audio.set_audio_format("pcm16");
        state
            .audio
            .on_audio_delta("item_1", 0, &[0u8; 480], Some("resp_1"));

        // Without a tracker the transport assumes real-time playback from first arrival.
        let own = state.playback_position(None);
        assert_eq!(own.item_id.as_deref(), Some("item_1"));
        assert_eq!(own.audio_item(), Some(("item_1".to_string(), 0)));
        assert!(own.elapsed_ms.is_some_and(|ms| ms >= 0.0));

        let tracker = RealtimePlaybackTracker::new();
        tracker.on_play_ms("item_9", 2, 42.0);
        let tracked = state.playback_position(Some(&tracker));
        assert_eq!(tracked.item_id.as_deref(), Some("item_9"));
        assert_eq!(tracked.content_index, Some(2));
        assert_eq!(tracked.elapsed_ms, Some(42.0));
    }

    /// Hold the socket lock so a send cannot complete, run `operation` until it times out (the
    /// caller giving up mid-send), then release the lock.
    async fn abandon_mid_send<F: std::future::Future>(inner: &WsInner, operation: F) {
        let held = inner.sink.lock().await;
        let outcome = tokio::time::timeout(Duration::from_millis(50), operation).await;
        assert!(
            outcome.is_err(),
            "the send must still be waiting for the socket"
        );
        drop(held);
    }

    #[tokio::test]
    async fn an_abandoned_cancel_frees_the_slot_for_a_retry() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let inner = &model.inner;
        inner
            .state
            .lock()
            .unwrap()
            .on_response_created(Some("response_1"));

        abandon_mid_send(inner, inner.cancel_response(Some("response_1"))).await;

        // The cancel never went out, so the next attempt must not be swallowed.
        assert!(inner.state.lock().unwrap().begin_cancel(Some("response_1")));
    }

    #[tokio::test]
    async fn a_failed_cancel_send_can_be_retried() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let inner = &model.inner;
        inner
            .state
            .lock()
            .unwrap()
            .on_response_created(Some("response_1"));

        // Nothing is connected, so the send fails.
        assert!(inner.cancel_response(Some("response_1")).await.is_err());
        assert!(inner.state.lock().unwrap().begin_cancel(Some("response_1")));
    }

    #[tokio::test]
    async fn an_abandoned_response_create_releases_the_in_flight_slot() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let inner = &model.inner;

        abandon_mid_send(inner, inner.request_response_create(None)).await;

        let mut state = inner.state.lock().unwrap();
        assert!(state.create_in_flight.is_none());
        // The next request starts a response instead of deferring behind one that never started.
        assert!(state.request_response_create(None).is_some());
    }

    #[tokio::test]
    async fn a_failed_response_create_send_releases_the_in_flight_slot() {
        let model = OpenAIRealtimeWebSocketModel::new();
        let inner = &model.inner;

        assert!(inner.request_response_create(None).await.is_err());

        let mut state = inner.state.lock().unwrap();
        assert!(state.create_in_flight.is_none());
        assert!(state.request_response_create(None).is_some());
    }

    #[test]
    fn a_response_done_for_another_response_leaves_the_active_one_alone() {
        let mut state = state();
        state.on_response_created(Some("response_2"));
        assert_eq!(state.request_response_create(None), None);

        assert_eq!(state.on_response_done(Some("response_1")), None);
        assert!(state.ongoing_response);
        assert!(state.pending_response_create);

        assert!(state.on_response_done(Some("response_2")).is_some());
        assert!(!state.ongoing_response);
    }
}
