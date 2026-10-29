//! The realtime transport interface. Mirrors `agents.realtime.model`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::realtime::config::RealtimeSessionModelSettings;
use crate::realtime::model_events::RealtimeModelEvent;
use crate::realtime::model_inputs::RealtimeModelSendEvent;

/// The stream of events a connected [`RealtimeModel`] produces.
///
/// The Python SDK registers listener objects; this port hands the single consumer (the
/// [`crate::realtime::RealtimeSession`]) an unbounded channel instead, so a slow consumer can
/// never block the transport's read loop. Applications observe the same events through
/// [`crate::realtime::RealtimeSessionEvent::RawModelEvent`].
pub type RealtimeModelEvents = mpsc::UnboundedReceiver<RealtimeModelEvent>;

/// How much of the model's audio the user has actually heard.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimePlaybackState {
    /// The item being played.
    pub current_item_id: Option<String>,
    /// Index of the audio content inside that item.
    pub current_item_content_index: Option<usize>,
    /// Milliseconds of that item played so far.
    pub elapsed_ms: Option<f64>,
}

/// Tracks audio playback progress on the application's behalf.
///
/// The transport generates audio far faster than it is played, so on an interruption it must know
/// how much the user actually heard in order to truncate the item at the right point. Without a
/// tracker the transport assumes audio plays immediately, at real-time speed, which is right for
/// low-latency local playback. Pass a tracker through
/// [`RealtimeModelConfig::playback_tracker`] when playback is delayed or remote, such as a phone
/// call, and call [`RealtimePlaybackTracker::on_play_bytes`] or
/// [`RealtimePlaybackTracker::on_play_ms`] as audio is played.
#[derive(Default)]
pub struct RealtimePlaybackTracker {
    inner: Mutex<PlaybackInner>,
}

#[derive(Default)]
struct PlaybackInner {
    format: Option<String>,
    current_item: Option<(String, usize)>,
    elapsed_ms: Option<f64>,
}

impl fmt::Debug for RealtimePlaybackTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimePlaybackTracker")
            .field("state", &self.state())
            .finish()
    }
}

impl RealtimePlaybackTracker {
    /// A tracker with no audio played yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// A shared tracker, ready for [`RealtimeModelConfig::with_playback_tracker`].
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PlaybackInner> {
        self.inner.lock().expect("playback tracker poisoned")
    }

    /// Record that `bytes` of `item_id` have been played.
    pub fn on_play_bytes(&self, item_id: &str, content_index: usize, bytes: &[u8]) {
        let ms = {
            let format = self.lock().format.clone();
            calculate_audio_length_ms(format.as_deref(), bytes)
        };
        self.on_play_ms(item_id, content_index, ms);
    }

    /// Record that `ms` milliseconds of `item_id` have been played.
    pub fn on_play_ms(&self, item_id: &str, content_index: usize, ms: f64) {
        let mut inner = self.lock();
        let key = (item_id.to_string(), content_index);
        match inner.current_item.as_ref() {
            Some(current) if *current == key => {
                *inner.elapsed_ms.get_or_insert(0.0) += ms;
            }
            _ => {
                inner.current_item = Some(key);
                inner.elapsed_ms = Some(ms);
            }
        }
    }

    /// Called by the transport when playback was interrupted.
    pub fn on_interrupted(&self) {
        let mut inner = self.lock();
        inner.current_item = None;
        inner.elapsed_ms = None;
    }

    /// Called by the transport once the session negotiates an audio format.
    pub fn set_audio_format(&self, format: impl Into<String>) {
        self.lock().format = Some(format.into());
    }

    /// The audio format the transport reported, if any.
    pub fn audio_format(&self) -> Option<String> {
        self.lock().format.clone()
    }

    /// The current playback position.
    pub fn state(&self) -> RealtimePlaybackState {
        let inner = self.lock();
        match inner.current_item.as_ref() {
            None => RealtimePlaybackState::default(),
            Some((item_id, content_index)) => RealtimePlaybackState {
                current_item_id: Some(item_id.clone()),
                current_item_content_index: Some(*content_index),
                elapsed_ms: inner.elapsed_ms,
            },
        }
    }
}

/// A condition rechecked at the transport's send boundary.
pub type RealtimeSendCondition = Box<dyn Fn() -> bool + Send + Sync>;

/// Resolves the API key for a connection.
pub type RealtimeApiKeyProvider =
    Arc<dyn Fn() -> crate::util::BoxFuture<'static, crate::error::Result<String>> + Send + Sync>;

/// The API key to connect with, either a value or something that produces one.
///
/// A provider is resolved on every connect, so a key with a short lifetime can be refreshed
/// without rebuilding the model. Port of Python's `str | Callable[[], MaybeAwaitable[str]]`.
#[derive(Clone)]
pub enum RealtimeApiKey {
    /// A literal key.
    Value(String),
    /// A function that produces a key when the transport connects.
    Provider(RealtimeApiKeyProvider),
}

impl RealtimeApiKey {
    /// Resolve the key.
    pub async fn resolve(&self) -> crate::error::Result<String> {
        match self {
            RealtimeApiKey::Value(key) => Ok(key.clone()),
            RealtimeApiKey::Provider(provider) => provider().await,
        }
    }
}

impl fmt::Debug for RealtimeApiKey {
    /// Never prints the key itself.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealtimeApiKey::Value(_) => write!(f, "Value(<redacted>)"),
            RealtimeApiKey::Provider(_) => write!(f, "Provider(..)"),
        }
    }
}

impl PartialEq for RealtimeApiKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RealtimeApiKey::Value(a), RealtimeApiKey::Value(b)) => a == b,
            (RealtimeApiKey::Provider(a), RealtimeApiKey::Provider(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl From<&str> for RealtimeApiKey {
    fn from(value: &str) -> Self {
        RealtimeApiKey::Value(value.to_string())
    }
}

impl From<String> for RealtimeApiKey {
    fn from(value: String) -> Self {
        RealtimeApiKey::Value(value)
    }
}

/// Low-level transport settings for the WebSocket connection.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeTransportConfig {
    /// How often to send a keepalive ping. `None` disables pings.
    pub ping_interval: Option<std::time::Duration>,
    /// How long to wait for a pong before treating the connection as dead.
    pub ping_timeout: Option<std::time::Duration>,
    /// How long to wait for the connection handshake.
    pub handshake_timeout: Option<std::time::Duration>,
    /// Maximum size of an incoming message. Defaults to 8 MiB, as in the Python SDK.
    pub max_size: Option<usize>,
}

impl RealtimeTransportConfig {
    /// Default transport settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send a keepalive ping every `interval`.
    pub fn with_ping_interval(mut self, interval: std::time::Duration) -> Self {
        self.ping_interval = Some(interval);
        self
    }

    /// Treat the connection as dead when no pong arrives within `timeout`.
    pub fn with_ping_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.ping_timeout = Some(timeout);
        self
    }

    /// Give up on the handshake after `timeout`.
    pub fn with_handshake_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.handshake_timeout = Some(timeout);
        self
    }

    /// Accept incoming messages up to `max_size` bytes.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }
}

/// Options used when connecting to a realtime model.
#[derive(Debug, Clone, Default)]
pub struct RealtimeModelConfig {
    /// API key to authenticate with. Falls back to the SDK default key, then `OPENAI_API_KEY`.
    pub api_key: Option<RealtimeApiKey>,
    /// Connection URL. Defaults to the OpenAI realtime WebSocket endpoint.
    pub url: Option<String>,
    /// Connection headers. When set, no `Authorization` header is added.
    pub headers: Option<BTreeMap<String, String>>,
    /// Settings applied with the first `session.update`.
    pub initial_model_settings: Option<RealtimeSessionModelSettings>,
    /// Attach to an existing realtime call instead of creating a session.
    pub call_id: Option<String>,
    /// Tracks how much audio the user has heard, for accurate interruption truncation.
    pub playback_tracker: Option<Arc<RealtimePlaybackTracker>>,
    /// Low-level WebSocket settings.
    pub transport: Option<RealtimeTransportConfig>,
}

impl PartialEq for RealtimeModelConfig {
    /// Trackers compare by identity, since they hold mutable playback progress.
    fn eq(&self, other: &Self) -> bool {
        self.api_key == other.api_key
            && self.url == other.url
            && self.headers == other.headers
            && self.initial_model_settings == other.initial_model_settings
            && self.call_id == other.call_id
            && self.transport == other.transport
            && match (&self.playback_tracker, &other.playback_tracker) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl RealtimeModelConfig {
    /// Empty options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<RealtimeApiKey>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Resolve the API key when the transport connects, rather than up front.
    pub fn with_api_key_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = crate::error::Result<String>> + Send + 'static,
    {
        self.api_key = Some(RealtimeApiKey::Provider(Arc::new(move || Box::pin(f()))));
        self
    }

    /// Set the low-level WebSocket settings.
    pub fn with_transport(mut self, transport: RealtimeTransportConfig) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Set the connection URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Add a connection header. Setting any header disables the default `Authorization` header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .get_or_insert_with(BTreeMap::new)
            .insert(name.into(), value.into());
        self
    }

    /// Set the settings applied on connect.
    pub fn with_initial_model_settings(mut self, settings: RealtimeSessionModelSettings) -> Self {
        self.initial_model_settings = Some(settings);
        self
    }

    /// Attach to an existing realtime call.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Track playback progress with `tracker` instead of assuming real-time local playback.
    pub fn with_playback_tracker(mut self, tracker: Arc<RealtimePlaybackTracker>) -> Self {
        self.playback_tracker = Some(tracker);
        self
    }
}

/// A transport that connects to a realtime model and exchanges events with it.
#[async_trait]
pub trait RealtimeModel: Send + Sync + std::fmt::Debug {
    /// Connect to the model and return its event stream.
    ///
    /// The transport keeps the connection alive until [`RealtimeModel::close`] is called or the
    /// peer disconnects, at which point it emits
    /// [`RealtimeModelEvent::EndOfStream`] and
    /// drops the sender.
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents>;

    /// Send an event to the model.
    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()>;

    /// Send an event only if `condition` still holds at the transport's send boundary.
    ///
    /// Returns whether the event was committed. The default implementation sends nothing and
    /// returns `false`, because checking a condition before an `await` is not atomic: a transport
    /// that supports conditional sends must recheck `condition` where it actually writes.
    async fn send_event_if(
        &self,
        _event: RealtimeModelSendEvent,
        _condition: RealtimeSendCondition,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Release transport state held for `response_id` once the session's guardrails have settled.
    async fn retire_response_audio(&self, _response_id: &str) -> Result<()> {
        Ok(())
    }

    /// Close the connection.
    async fn close(&self) -> Result<()>;
}

/// Bytes per PCM16 sample.
const PCM16_SAMPLE_WIDTH_BYTES: usize = 2;
/// PCM16 sample rate used by the realtime API.
const PCM16_SAMPLE_RATE_HZ: f64 = 24_000.0;
/// G.711 sample rate.
const G711_SAMPLE_RATE_HZ: f64 = 8_000.0;

/// Duration in milliseconds of `audio_bytes` in `format`.
///
/// Unknown formats fall back to PCM16 at 24 kHz, as the Python SDK does.
pub fn calculate_audio_length_ms(format: Option<&str>, audio_bytes: &[u8]) -> f64 {
    if audio_bytes.is_empty() {
        return 0.0;
    }
    if is_g711_format(format) {
        return (audio_bytes.len() as f64 / G711_SAMPLE_RATE_HZ) * 1000.0;
    }
    let samples = audio_bytes.len() as f64 / PCM16_SAMPLE_WIDTH_BYTES as f64;
    (samples / PCM16_SAMPLE_RATE_HZ) * 1000.0
}

fn is_g711_format(format: Option<&str>) -> bool {
    let Some(format) = format else {
        return false;
    };
    let format = format.to_ascii_lowercase();
    format.starts_with("g711")
        || matches!(
            format.as_str(),
            "audio/pcmu" | "audio/pcma" | "pcmu" | "pcma"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_length() {
        // 24 kHz, 2 bytes per sample: 48 bytes is one millisecond.
        let ms = calculate_audio_length_ms(Some("pcm16"), &[0u8; 48]);
        assert!((ms - 1.0).abs() < 1e-9);
    }

    #[test]
    fn g711_length() {
        let ms = calculate_audio_length_ms(Some("g711_ulaw"), &[0u8; 8]);
        assert!((ms - 1.0).abs() < 1e-9);
        let ms = calculate_audio_length_ms(Some("audio/pcma"), &[0u8; 8]);
        assert!((ms - 1.0).abs() < 1e-9);
    }

    #[test]
    fn playback_tracker_accumulates_within_one_item() {
        let tracker = RealtimePlaybackTracker::new();
        assert_eq!(tracker.state(), RealtimePlaybackState::default());

        tracker.on_play_ms("item_1", 0, 100.0);
        tracker.on_play_ms("item_1", 0, 50.0);
        let state = tracker.state();
        assert_eq!(state.current_item_id.as_deref(), Some("item_1"));
        assert_eq!(state.current_item_content_index, Some(0));
        assert_eq!(state.elapsed_ms, Some(150.0));

        // A new item restarts the count rather than adding to the old one.
        tracker.on_play_ms("item_2", 1, 20.0);
        let state = tracker.state();
        assert_eq!(state.current_item_id.as_deref(), Some("item_2"));
        assert_eq!(state.current_item_content_index, Some(1));
        assert_eq!(state.elapsed_ms, Some(20.0));

        tracker.on_interrupted();
        assert_eq!(tracker.state(), RealtimePlaybackState::default());
    }

    #[test]
    fn playback_tracker_converts_bytes_with_the_session_format() {
        let tracker = RealtimePlaybackTracker::new();
        tracker.set_audio_format("g711_ulaw");
        assert_eq!(tracker.audio_format().as_deref(), Some("g711_ulaw"));
        tracker.on_play_bytes("item_1", 0, &[0u8; 8]);
        assert_eq!(tracker.state().elapsed_ms, Some(1.0));
    }

    #[test]
    fn unknown_format_falls_back_to_pcm16() {
        let ms = calculate_audio_length_ms(None, &[0u8; 48]);
        assert!((ms - 1.0).abs() < 1e-9);
    }
}
