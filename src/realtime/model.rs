//! The realtime transport interface. Mirrors `agents.realtime.model`.

use std::collections::BTreeMap;

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

/// Options used when connecting to a realtime model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealtimeModelConfig {
    /// API key to authenticate with. Falls back to the SDK default key, then `OPENAI_API_KEY`.
    pub api_key: Option<String>,
    /// Connection URL. Defaults to the OpenAI realtime WebSocket endpoint.
    pub url: Option<String>,
    /// Connection headers. When set, no `Authorization` header is added.
    pub headers: Option<BTreeMap<String, String>>,
    /// Settings applied with the first `session.update`.
    pub initial_model_settings: Option<RealtimeSessionModelSettings>,
    /// Attach to an existing realtime call instead of creating a session.
    pub call_id: Option<String>,
}

impl RealtimeModelConfig {
    /// Empty options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
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
    fn unknown_format_falls_back_to_pcm16() {
        let ms = calculate_audio_length_ms(None, &[0u8; 48]);
        assert!((ms - 1.0).abs() < 1e-9);
    }
}
