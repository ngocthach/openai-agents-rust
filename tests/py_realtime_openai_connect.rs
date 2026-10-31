//! Ports of the connection-lifecycle, session-config and transport tests of
//! `tests/realtime/test_openai_realtime.py` (`TestConnectionLifecycle`, the config half of
//! `TestSendEventAndConfig`, `TestTransportIntegration`), plus
//! `tests/realtime/test_websocket_message_size.py`.
//!
//! Python patches `websockets.connect` and asserts on its keyword arguments. The Rust transport
//! owns its socket, so each test runs against a local server and asserts on what that server
//! observes: the handshake headers, the frames, the pings, and the timing of a slow handshake.
//! No network, no real API key.

#![cfg(feature = "realtime")]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::realtime::{
    OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, RealtimeApiKey, RealtimeAudioConfig,
    RealtimeAudioFormat, RealtimeAudioInputConfig, RealtimeAudioOutputConfig,
    RealtimeConnectionStatus, RealtimeFunctionToolSpec, RealtimeInputAudioTranscriptionConfig,
    RealtimeModel, RealtimeModelConfig, RealtimeModelEvent, RealtimeNoiseReductionConfig,
    RealtimeReasoningConfig, RealtimeSessionModelSettings, RealtimeTransportConfig,
    RealtimeTurnDetectionConfig, RealtimeVoice, calculate_audio_length_ms,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::protocol::Message;

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

// ---------------------------------------------------------------------------------------------
// Local servers
// ---------------------------------------------------------------------------------------------

/// What the server saw in the client's upgrade request.
#[derive(Debug)]
struct Handshake {
    uri: String,
    headers: HashMap<String, String>,
}

/// Accepts one client, records its upgrade request, and forwards every JSON frame it sends.
struct CaptureServer {
    port: u16,
    handshake: Option<tokio::sync::oneshot::Receiver<Handshake>>,
    frames: UnboundedReceiver<Value>,
    handle: JoinHandle<()>,
}

impl CaptureServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (handshake_tx, handshake_rx) = tokio::sync::oneshot::channel();
        let (frame_tx, frames) = unbounded_channel();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut handshake_tx = Some(handshake_tx);
            #[allow(clippy::result_large_err)]
            let callback = |request: &Request, response: Response| {
                let headers = request
                    .headers()
                    .iter()
                    .filter_map(|(name, value)| {
                        Some((name.as_str().to_string(), value.to_str().ok()?.to_string()))
                    })
                    .collect();
                if let Some(tx) = handshake_tx.take() {
                    let _ = tx.send(Handshake {
                        uri: request.uri().to_string(),
                        headers,
                    });
                }
                Ok(response)
            };
            let mut ws = tokio_tungstenite::accept_hdr_async(stream, callback)
                .await
                .unwrap();
            while let Some(Ok(message)) = ws.next().await {
                if let Message::Text(text) = message
                    && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
                {
                    let _ = frame_tx.send(value);
                }
            }
        });
        CaptureServer {
            port,
            handshake: Some(handshake_rx),
            frames,
            handle,
        }
    }

    fn url(&self) -> String {
        format!("ws://127.0.0.1:{}/v1/realtime", self.port)
    }

    async fn handshake(&mut self) -> Handshake {
        let rx = self.handshake.take().expect("handshake already taken");
        tokio::time::timeout(TIMEOUT, rx)
            .await
            .expect("handshake captured")
            .unwrap()
    }

    /// The first `session.update` the client sent.
    async fn session_update(&mut self) -> Value {
        tokio::time::timeout(TIMEOUT, async {
            while let Some(frame) = self.frames.recv().await {
                if frame["type"] == "session.update" {
                    return frame;
                }
            }
            panic!("the client closed before sending session.update");
        })
        .await
        .expect("session.update received")
    }
}

/// A frame a ping-aware server observed, in arrival order.
#[derive(Debug)]
enum Seen {
    Ping,
    Text(Value),
}

/// Accepts one client and reports every ping and text frame, in order. With `respond` false it
/// completes the handshake and then never reads again, so pings go unanswered.
async fn recording_server(respond: bool) -> (u16, UnboundedReceiver<Seen>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = unbounded_channel();
    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        if !respond {
            std::future::pending::<()>().await;
        }
        while let Some(Ok(message)) = ws.next().await {
            match message {
                Message::Ping(_) => {
                    let _ = tx.send(Seen::Ping);
                }
                Message::Text(text) => {
                    if let Ok(value) = serde_json::from_str::<Value>(text.as_str()) {
                        let _ = tx.send(Seen::Text(value));
                    }
                }
                _ => {}
            }
        }
    });
    (port, rx, handle)
}

/// A server whose upgrade completes `delay` after the TCP connection arrives.
async fn slow_handshake_server(delay: Duration) -> (u16, Arc<AtomicUsize>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepted = Arc::new(AtomicUsize::new(0));
    let counter = accepted.clone();
    let handle = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                if let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await {
                    while ws.next().await.is_some() {}
                }
            });
        }
    });
    (port, accepted, handle)
}

fn config_for(url: impl Into<String>) -> RealtimeModelConfig {
    RealtimeModelConfig::new()
        .with_api_key("test-key")
        .with_url(url)
}

fn payload_for(settings: &RealtimeSessionModelSettings) -> Value {
    OpenAIRealtimeWebSocketModel::new().session_payload_for_settings(settings)
}

// ---------------------------------------------------------------------------------------------
// TestConnectionLifecycle
// ---------------------------------------------------------------------------------------------

/// `test_connect_with_call_id_and_model_raises_error`
#[tokio::test]
async fn connect_with_call_id_and_model_raises_error() {
    let model = OpenAIRealtimeWebSocketModel::new();
    let error = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("test-api-key-123")
                .with_call_id("call-123")
                .with_initial_model_settings(
                    RealtimeSessionModelSettings::new().with_model_name("gpt-4o-realtime-preview"),
                ),
        )
        .await
        .expect_err("an attached call has its model already");
    assert!(
        error
            .to_string()
            .contains("Cannot specify both `call_id` and `model_name`"),
        "unexpected error: {error}"
    );
}

/// `test_connect_with_string_api_key`: the bearer header, no beta header, and the model name.
#[tokio::test]
async fn connect_with_string_api_key_sends_the_bearer_header() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("test-api-key-123")
                .with_url(server.url())
                .with_initial_model_settings(
                    RealtimeSessionModelSettings::new().with_model_name("gpt-4o-realtime-preview"),
                ),
        )
        .await
        .unwrap();

    let handshake = server.handshake().await;
    assert_eq!(
        handshake.headers.get("authorization").map(String::as_str),
        Some("Bearer test-api-key-123")
    );
    assert!(!handshake.headers.contains_key("openai-beta"));
    assert!(
        handshake.uri.starts_with("/v1/realtime"),
        "{}",
        handshake.uri
    );
    assert_eq!(model.model_name(), "gpt-4o-realtime-preview");
    assert_eq!(
        server.session_update().await["session"]["model"],
        "gpt-4o-realtime-preview"
    );
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_connect_defaults_to_gpt_realtime_2_1`
#[tokio::test]
async fn connect_defaults_to_gpt_realtime_2_1() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    assert_eq!(model.model_name(), "gpt-realtime-2.1");
    let _events = model.connect(config_for(server.url())).await.unwrap();

    assert_eq!(model.model_name(), "gpt-realtime-2.1");
    assert_eq!(
        server.session_update().await["session"]["model"],
        "gpt-realtime-2.1"
    );
    model.close().await.unwrap();
    server.handle.abort();
}

async fn first_session_update(settings: RealtimeSessionModelSettings) -> Value {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(config_for(server.url()).with_initial_model_settings(settings))
        .await
        .unwrap();
    let update = server.session_update().await;
    model.close().await.unwrap();
    server.handle.abort();
    update
}

/// `test_session_update_includes_noise_reduction`
#[tokio::test]
async fn session_update_includes_noise_reduction() {
    let mut settings =
        RealtimeSessionModelSettings::new().with_model_name("gpt-4o-realtime-preview");
    settings.input_audio_noise_reduction = Some(Some(RealtimeNoiseReductionConfig {
        kind: Some("near_field".to_string()),
    }));
    let update = first_session_update(settings).await;
    assert_eq!(
        update["session"]["audio"]["input"]["noise_reduction"],
        json!({"type": "near_field"})
    );
}

/// `test_session_update_omits_noise_reduction_when_not_provided`
#[tokio::test]
async fn session_update_omits_noise_reduction_when_not_provided() {
    let update = first_session_update(
        RealtimeSessionModelSettings::new().with_model_name("gpt-4o-realtime-preview"),
    )
    .await;
    let input = &update["session"]["audio"]["input"];
    assert!(input.is_object());
    assert!(input.get("noise_reduction").is_none());
}

/// `test_session_update_disable_turn_detection`
#[tokio::test]
async fn session_update_disable_turn_detection() {
    let update = first_session_update(
        RealtimeSessionModelSettings::new()
            .with_model_name("gpt-4o-realtime-preview")
            .without_turn_detection(),
    )
    .await;
    assert!(update["session"]["audio"]["input"]["turn_detection"].is_null());
    assert!(
        update["session"]["audio"]["input"]
            .as_object()
            .unwrap()
            .contains_key("turn_detection")
    );
}

/// `test_connect_with_custom_headers_overrides_defaults`
#[tokio::test]
async fn connect_with_custom_headers_overrides_defaults() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("unused-because-headers-override")
                .with_header("api-key", "azure-key")
                .with_header("x-custom", "1")
                .with_url(server.url()),
        )
        .await
        .unwrap();

    let handshake = server.handshake().await;
    assert_eq!(
        handshake.headers.get("api-key").map(String::as_str),
        Some("azure-key")
    );
    assert_eq!(
        handshake.headers.get("x-custom").map(String::as_str),
        Some("1")
    );
    assert!(!handshake.headers.contains_key("authorization"));
    assert!(!handshake.headers.contains_key("openai-beta"));
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_connect_with_callable_api_key`: a provider that returns the key immediately.
#[tokio::test]
async fn connect_with_callable_api_key() {
    let mut server = CaptureServer::start().await;
    let provider = RealtimeApiKey::Provider(Arc::new(|| {
        Box::pin(std::future::ready(Ok("callable-api-key".to_string())))
    }));
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key(provider)
                .with_url(server.url()),
        )
        .await
        .unwrap();
    assert_eq!(
        server
            .handshake()
            .await
            .headers
            .get("authorization")
            .map(String::as_str),
        Some("Bearer callable-api-key")
    );
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_connect_with_async_callable_api_key`: a provider that awaits before answering.
#[tokio::test]
async fn connect_with_async_callable_api_key() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key_fn(|| async {
                    tokio::task::yield_now().await;
                    Ok("async-api-key".to_string())
                })
                .with_url(server.url()),
        )
        .await
        .unwrap();
    assert_eq!(
        server
            .handshake()
            .await
            .headers
            .get("authorization")
            .map(String::as_str),
        Some("Bearer async-api-key")
    );
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_connect_websocket_failure_propagates`: nothing listens, and the model stays reusable.
#[tokio::test]
async fn connect_websocket_failure_propagates() {
    let dead_port = {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap().port()
    };
    let model = OpenAIRealtimeWebSocketModel::new();
    let error = model
        .connect(config_for(format!(
            "ws://127.0.0.1:{dead_port}/v1/realtime"
        )))
        .await
        .expect_err("nothing is listening");
    assert!(
        error.to_string().contains("Realtime connection failed"),
        "{error}"
    );

    // The failed attempt left no connection behind, so the same model connects normally.
    let mut server = CaptureServer::start().await;
    let _events = model.connect(config_for(server.url())).await.unwrap();
    let _ = server.session_update().await;
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_concurrent_connect_is_rejected_before_acquiring_another_websocket`
#[tokio::test]
async fn concurrent_connect_is_rejected_before_acquiring_another_websocket() {
    let (port, accepted, handle) = slow_handshake_server(Duration::from_millis(300)).await;
    let model = Arc::new(OpenAIRealtimeWebSocketModel::new());
    let url = format!("ws://127.0.0.1:{port}/v1/realtime");

    let first = {
        let model = model.clone();
        let url = url.clone();
        tokio::spawn(async move { model.connect(config_for(url)).await.map(|_| ()) })
    };
    // Wait until the first attempt owns a socket, then try again while it is still handshaking.
    tokio::time::timeout(TIMEOUT, async {
        while accepted.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the first attempt reached the server");

    let error = model
        .connect(config_for(url))
        .await
        .expect_err("a second attempt must not start while the first is connecting");
    assert!(error.to_string().contains("Already connected"), "{error}");
    assert_eq!(
        accepted.load(Ordering::SeqCst),
        1,
        "only one socket is acquired"
    );

    first
        .await
        .unwrap()
        .expect("the first attempt still completes");
    model.close().await.unwrap();
    handle.abort();
}

/// `test_connect_with_empty_transport_config`
#[tokio::test]
async fn connect_with_empty_transport_config() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(config_for(server.url()).with_transport(RealtimeTransportConfig::new()))
        .await
        .expect("an empty transport config keeps every default");
    let _ = server.session_update().await;
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_connect_already_connected_assertion`
#[tokio::test]
async fn connect_already_connected_assertion() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model.connect(config_for(server.url())).await.unwrap();
    let _ = server.session_update().await;

    let error = model
        .connect(config_for(server.url()))
        .await
        .expect_err("a connected model cannot connect again");
    assert!(error.to_string().contains("Already connected"), "{error}");
    model.close().await.unwrap();
    server.handle.abort();
}

// ---------------------------------------------------------------------------------------------
// Session config (`_get_session_config`), through `session_payload_for_settings`
// ---------------------------------------------------------------------------------------------

/// `test_get_and_update_session_config`
#[test]
fn get_and_update_session_config() {
    let mut settings = RealtimeSessionModelSettings::new()
        .with_model_name("gpt-realtime")
        .with_voice("verse")
        .with_turn_detection(RealtimeTurnDetectionConfig::semantic_vad());
    settings.output_audio_format = Some(RealtimeAudioFormat::from("g711_ulaw"));
    settings.modalities = Some(vec!["audio".to_string()]);
    settings.input_audio_format = Some(RealtimeAudioFormat::from("pcm16"));
    settings.input_audio_transcription = Some(RealtimeInputAudioTranscriptionConfig::new(
        "gpt-4o-mini-transcribe",
    ));
    let payload = payload_for(&settings);
    assert_eq!(payload["model"], "gpt-realtime");
    assert_eq!(payload["audio"]["output"]["voice"], "verse");
    assert_eq!(
        payload["audio"]["output"]["format"],
        json!({"type": "audio/pcmu"})
    );
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcm");
    assert_eq!(payload["output_modalities"], json!(["audio"]));
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
}

/// `test_session_config_accepts_custom_voice_object`
#[test]
fn session_config_accepts_custom_voice_object() {
    let mut settings = RealtimeSessionModelSettings::new();
    settings.voice = Some(RealtimeVoice::Custom {
        id: "voice_test".to_string(),
    });
    assert_eq!(
        payload_for(&settings)["audio"]["output"]["voice"],
        json!({"id": "voice_test"})
    );
}

/// `test_session_config_accepts_nested_custom_voice_object`
#[test]
fn session_config_accepts_nested_custom_voice_object() {
    let settings = RealtimeSessionModelSettings::new().with_audio(
        RealtimeAudioConfig::new().with_output(RealtimeAudioOutputConfig {
            voice: Some(RealtimeVoice::Custom {
                id: "voice_test".to_string(),
            }),
            ..Default::default()
        }),
    );
    assert_eq!(
        payload_for(&settings)["audio"]["output"]["voice"],
        json!({"id": "voice_test"})
    );
}

/// `test_session_config_defaults_audio_formats_when_not_call`
#[test]
fn session_config_defaults_audio_formats_when_not_call() {
    let payload = payload_for(&RealtimeSessionModelSettings::new());
    assert_eq!(payload["model"], "gpt-realtime-2.1");
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcm");
    assert_eq!(payload["audio"]["output"]["format"]["type"], "audio/pcm");
}

/// `test_session_config_includes_reasoning_capable_settings`
#[test]
fn session_config_includes_reasoning_capable_settings() {
    let mut settings =
        RealtimeSessionModelSettings::new().with_reasoning(RealtimeReasoningConfig::new("low"));
    settings.parallel_tool_calls = Some(false);
    let payload = payload_for(&settings);
    assert_eq!(payload["model"], "gpt-realtime-2.1");
    assert_eq!(payload["parallel_tool_calls"], false);
    assert_eq!(payload["reasoning"], json!({"effort": "low"}));
}

/// `test_session_config_forwards_ga_input_audio_transcription_options`
#[test]
fn session_config_forwards_ga_input_audio_transcription_options() {
    fn transcription_settings(
        transcription: RealtimeInputAudioTranscriptionConfig,
        turn_detection: Option<Option<RealtimeTurnDetectionConfig>>,
    ) -> Value {
        let settings = RealtimeSessionModelSettings::new().with_audio(
            RealtimeAudioConfig::new().with_input(RealtimeAudioInputConfig {
                transcription: Some(transcription),
                turn_detection,
                ..Default::default()
            }),
        );
        payload_for(&settings)["audio"]["input"].clone()
    }

    let contextual = transcription_settings(
        RealtimeInputAudioTranscriptionConfig {
            model: Some("gpt-transcribe".to_string()),
            keywords: Some(vec!["LegalOn".to_string(), "TomoniAI".to_string()]),
            languages: Some(vec!["ja".to_string(), "en".to_string()]),
            prompt: Some("A Japanese conversation about LegalOn and TomoniAI.".to_string()),
            ..Default::default()
        },
        None,
    );
    assert_eq!(
        contextual["transcription"],
        json!({
            "model": "gpt-transcribe",
            "keywords": ["LegalOn", "TomoniAI"],
            "languages": ["ja", "en"],
            "prompt": "A Japanese conversation about LegalOn and TomoniAI.",
        })
    );

    let low_latency = transcription_settings(
        RealtimeInputAudioTranscriptionConfig::new("gpt-live-transcribe"),
        Some(None),
    );
    assert_eq!(
        low_latency["transcription"],
        json!({"model": "gpt-live-transcribe"})
    );
    assert!(low_latency["turn_detection"].is_null());

    let whisper = transcription_settings(
        RealtimeInputAudioTranscriptionConfig {
            model: Some("gpt-realtime-whisper".to_string()),
            delay: Some("low".to_string()),
            ..Default::default()
        },
        Some(None),
    );
    assert_eq!(
        whisper["transcription"],
        json!({"model": "gpt-realtime-whisper", "delay": "low"})
    );
    assert!(whisper["turn_detection"].is_null());
}

/// `test_session_config_passes_max_output_tokens`
#[test]
fn session_config_passes_max_output_tokens() {
    let mut settings = RealtimeSessionModelSettings::new();
    settings.max_output_tokens = Some(json!(256));
    assert_eq!(payload_for(&settings)["max_output_tokens"], 256);

    settings.max_output_tokens = Some(json!("inf"));
    assert_eq!(payload_for(&settings)["max_output_tokens"], "inf");

    // Omitting it leaves the field out so the server default applies.
    assert!(
        payload_for(&RealtimeSessionModelSettings::new())
            .get("max_output_tokens")
            .is_none()
    );
}

/// `test_session_config_allows_tool_search_as_named_function_tool_choice`
#[test]
fn session_config_allows_tool_search_as_named_function_tool_choice() {
    let mut settings = RealtimeSessionModelSettings::new();
    settings.tool_choice = Some("tool_search".to_string());
    settings.tools = Some(vec![RealtimeFunctionToolSpec::new(
        "tool_search",
        "",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )]);
    let payload = payload_for(&settings);
    assert_eq!(payload["tool_choice"], "tool_search");
    assert_eq!(payload["tools"][0]["name"], "tool_search");
}

/// `test_session_config_treats_none_audio_channels_as_unset`: an audio block with neither half
/// behaves like no audio block.
#[test]
fn session_config_treats_none_audio_channels_as_unset() {
    let payload =
        payload_for(&RealtimeSessionModelSettings::new().with_audio(RealtimeAudioConfig::new()));
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcm");
    assert_eq!(payload["audio"]["output"]["voice"], "ash");
}

/// `test_session_config_respects_audio_block_and_output_modalities`
#[test]
fn session_config_respects_audio_block_and_output_modalities() {
    let mut settings = RealtimeSessionModelSettings::new()
        .with_output_modalities(["text"])
        .with_audio(
            RealtimeAudioConfig::new()
                .with_input(RealtimeAudioInputConfig {
                    format: Some(RealtimeAudioFormat::Object(json!({"type": "audio/pcmu"}))),
                    turn_detection: Some(Some(RealtimeTurnDetectionConfig {
                        kind: Some("server_vad".to_string()),
                        create_response: Some(true),
                        silence_duration_ms: Some(450),
                        model_version: Some("default".to_string()),
                        ..Default::default()
                    })),
                    ..Default::default()
                })
                .with_output(RealtimeAudioOutputConfig {
                    format: Some(RealtimeAudioFormat::Object(json!({"type": "audio/pcma"}))),
                    voice: Some(RealtimeVoice::from("synth-1")),
                    speed: Some(1.5),
                }),
        );
    // The flat keys and the legacy `modalities` lose to the nested block and `output_modalities`.
    settings.input_audio_format = Some(RealtimeAudioFormat::from("pcm16"));
    settings.output_audio_format = Some(RealtimeAudioFormat::from("pcm16"));
    settings.modalities = Some(vec!["audio".to_string()]);

    let payload = payload_for(&settings);
    assert_eq!(payload["output_modalities"], json!(["text"]));
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcmu");
    assert_eq!(payload["audio"]["output"]["format"]["type"], "audio/pcma");
    assert_eq!(payload["audio"]["output"]["voice"], "synth-1");
    assert_eq!(payload["audio"]["output"]["speed"], 1.5);
    assert!(payload["audio"]["input"]["transcription"].is_object());
    let turn_detection = &payload["audio"]["input"]["turn_detection"];
    assert_eq!(turn_detection["create_response"], true);
    assert_eq!(turn_detection["silence_duration_ms"], 450);
    assert_eq!(turn_detection["model_version"], "default");
    assert!(turn_detection.get("silenceDurationMs").is_none());
    assert!(turn_detection.get("modelVersion").is_none());
}

/// `test_calculate_audio_length_ms_pure_function`
#[test]
fn calculate_audio_length_ms_pure_function() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    assert!(close(
        calculate_audio_length_ms(Some("pcm16"), b"test"),
        4.0 / 48_000.0 * 1000.0
    ));
    assert_eq!(calculate_audio_length_ms(Some("pcm16"), b""), 0.0);
    assert!(close(
        calculate_audio_length_ms(Some("pcm16"), &[b'a'; 48]),
        1.0
    ));
    assert_eq!(
        calculate_audio_length_ms(Some("g711_ulaw"), b"test"),
        4.0 / 8000.0 * 1000.0
    );
    assert_eq!(
        calculate_audio_length_ms(Some("g711_alaw"), &[b'a'; 8]),
        1.0
    );
}

// ---------------------------------------------------------------------------------------------
// TestTransportIntegration
// ---------------------------------------------------------------------------------------------

/// Drain every event already queued, without waiting for more.
fn drain(events: &mut openai_agents_sdk::realtime::RealtimeModelEvents) -> Vec<RealtimeModelEvent> {
    let mut drained = Vec::new();
    while let Ok(event) = events.try_recv() {
        drained.push(event);
    }
    drained
}

/// Read events until the stream ends, which the transport signals with `EndOfStream`.
async fn collect_until_end(
    events: &mut openai_agents_sdk::realtime::RealtimeModelEvents,
) -> Vec<RealtimeModelEvent> {
    tokio::time::timeout(TIMEOUT, async {
        let mut collected = Vec::new();
        while let Some(event) = events.recv().await {
            let end = matches!(event, RealtimeModelEvent::EndOfStream);
            collected.push(event);
            if end {
                break;
            }
        }
        collected
    })
    .await
    .expect("the event stream ends")
}

fn disconnects(events: &[RealtimeModelEvent]) -> usize {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                RealtimeModelEvent::ConnectionStatus {
                    status: RealtimeConnectionStatus::Disconnected
                }
            )
        })
        .count()
}

/// `test_connect_to_local_server`
#[tokio::test]
async fn connect_to_local_server() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let transport = RealtimeTransportConfig::new()
        .with_ping_interval(Duration::from_millis(500))
        .with_ping_timeout(Duration::from_millis(500))
        .with_handshake_timeout(Duration::from_secs(1));
    let _events = model
        .connect(
            config_for(server.url())
                .with_transport(transport)
                .with_initial_model_settings(
                    RealtimeSessionModelSettings::new().with_model_name("gpt-4o-realtime-preview"),
                ),
        )
        .await
        .unwrap();
    let update = server.session_update().await;
    assert_eq!(update["session"]["model"], "gpt-4o-realtime-preview");
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_normal_server_close_ends_session_iteration`: a clean close ends the stream without an
/// exception.
#[tokio::test]
async fn normal_server_close_ends_session_iteration() {
    let mut server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;
    server.push_close(1000);

    let collected = collect_until_end(&mut events).await;
    assert_eq!(disconnects(&collected), 1);
    assert!(
        !collected
            .iter()
            .any(|event| matches!(event, RealtimeModelEvent::Exception { .. })),
        "a normal close is not an error: {collected:#?}"
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_client_close_does_not_emit_server_disconnect`
#[tokio::test]
async fn client_close_does_not_emit_server_disconnect() {
    let mut first = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&first, None, None).await;
    let _ = first.next_frame("session.update").await;
    model.close().await.unwrap();
    // After a caller-owned close the stream just ends; it never reports a server disconnect.
    let mut seen = Vec::new();
    while let Some(event) = tokio::time::timeout(TIMEOUT, events.recv()).await.unwrap() {
        seen.push(event);
    }
    assert_eq!(disconnects(&seen), 0, "{seen:#?}");
    first.stop();

    // The same model reconnects, and now the server hangs up: exactly one disconnect.
    let mut second = MockRealtimeServer::start().await;
    let mut config = RealtimeModelConfig::new()
        .with_api_key("sk-test")
        .with_url(second.url());
    config = config.with_initial_model_settings(
        RealtimeSessionModelSettings::new().with_model_name("gpt-realtime"),
    );
    let mut events = model.connect(config).await.unwrap();
    let _ = second.next_frame("session.update").await;
    second.push_close(1000);
    let collected = collect_until_end(&mut events).await;
    assert_eq!(disconnects(&collected), 1);
    model.close().await.unwrap();
    second.stop();
}

/// `test_abnormal_server_close_still_raises`: only a normal or going-away close is clean.
#[tokio::test]
async fn abnormal_server_close_still_raises() {
    let mut server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;
    server.push_close(1011);

    let collected = collect_until_end(&mut events).await;
    let message = collected.iter().find_map(|event| match event {
        RealtimeModelEvent::Exception { message, .. } => Some(message.clone()),
        _ => None,
    });
    let message = message.expect("an abnormal close surfaces as an exception");
    assert!(message.contains("1011"), "{message}");
    model.close().await.unwrap();
    server.stop();
}

/// `test_ping_timeout_success_when_server_responds_quickly`
#[tokio::test]
async fn ping_timeout_success_when_server_responds_quickly() {
    let (port, mut seen, handle) = recording_server(true).await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let transport = RealtimeTransportConfig::new()
        .with_ping_interval(Duration::from_millis(50))
        .with_ping_timeout(Duration::from_secs(1));
    let mut events = model
        .connect(config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(transport))
        .await
        .unwrap();

    // Several ping/pong cycles go by, and the connection is still up afterwards.
    let mut pings = 0;
    tokio::time::timeout(TIMEOUT, async {
        while pings < 3 {
            if matches!(seen.recv().await, Some(Seen::Ping)) {
                pings += 1;
            }
        }
    })
    .await
    .expect("the keepalive keeps pinging");
    send_marker(&model).await;
    assert!(
        drain(&mut events)
            .iter()
            .all(|event| !matches!(event, RealtimeModelEvent::EndOfStream)),
        "a responsive server keeps the connection open"
    );
    model.close().await.unwrap();
    handle.abort();
}

/// `test_ping_timeout_config_is_applied`: a peer that never answers pings is dropped once the
/// timeout passes, which only happens if the configured timeout reached the keepalive.
#[tokio::test]
async fn ping_timeout_config_is_applied() {
    let (port, _seen, handle) = recording_server(false).await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let transport = RealtimeTransportConfig::new()
        .with_ping_interval(Duration::from_millis(50))
        .with_ping_timeout(Duration::from_millis(50));
    let mut events = model
        .connect(config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(transport))
        .await
        .unwrap();

    let collected = collect_until_end(&mut events).await;
    assert_eq!(disconnects(&collected), 1, "{collected:#?}");
    model.close().await.unwrap();
    handle.abort();
}

/// `test_handshake_timeout_config_is_applied`: a server that never completes the upgrade makes
/// the connect fail once the configured timeout passes.
#[tokio::test]
async fn handshake_timeout_config_is_applied() {
    let (port, accepted, handle) = slow_handshake_server(Duration::from_secs(30)).await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let started = std::time::Instant::now();
    let error = model
        .connect(
            config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(
                RealtimeTransportConfig::new().with_handshake_timeout(Duration::from_millis(150)),
            ),
        )
        .await
        .expect_err("the handshake never completes");
    assert!(error.to_string().contains("handshake timed out"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
    handle.abort();
}

/// `test_max_size_config_is_applied`: see [`incoming_message_size_limit`] for the full matrix;
/// this is the explicit-limit case on its own.
#[tokio::test]
async fn max_size_config_is_applied() {
    let outcome = size_limit_outcome(
        Some(RealtimeTransportConfig::new().with_max_size(1024)),
        1025,
    )
    .await;
    assert_eq!(outcome, SizeOutcome::Rejected);
    let outcome = size_limit_outcome(
        Some(RealtimeTransportConfig::new().with_max_size(1024)),
        1024,
    )
    .await;
    assert_eq!(outcome, SizeOutcome::Accepted);
}

/// `test_ping_timeout_disabled_vs_enabled`
#[tokio::test]
async fn ping_timeout_disabled_vs_enabled() {
    // Disabled: nothing but the frames the client itself sends ever reaches the server.
    let (port, mut seen, handle) = recording_server(true).await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(
                // No interval means no keepalive; the timeout alone has nothing to guard.
                RealtimeTransportConfig::new().with_ping_timeout(Duration::from_millis(50)),
            ),
        )
        .await
        .unwrap();
    send_marker(&model).await;
    let mut before_marker = Vec::new();
    tokio::time::timeout(TIMEOUT, async {
        while let Some(item) = seen.recv().await {
            if let Seen::Text(value) = &item
                && value["session"]["instructions"] == MARKER_INSTRUCTIONS
            {
                return;
            }
            before_marker.push(item);
        }
    })
    .await
    .expect("the marker arrives");
    assert!(
        !before_marker.iter().any(|item| matches!(item, Seen::Ping)),
        "pings are off without an interval: {before_marker:#?}"
    );
    model.close().await.unwrap();
    handle.abort();

    // Enabled: the server does see pings.
    let (port, mut seen, handle) = recording_server(true).await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(
            config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(
                RealtimeTransportConfig::new()
                    .with_ping_interval(Duration::from_millis(50))
                    .with_ping_timeout(Duration::from_secs(2)),
            ),
        )
        .await
        .unwrap();
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if matches!(seen.recv().await, Some(Seen::Ping)) {
                return;
            }
        }
    })
    .await
    .expect("an enabled keepalive pings");
    model.close().await.unwrap();
    handle.abort();
}

/// `test_handshake_timeout_success_when_server_responds_quickly`
#[tokio::test]
async fn handshake_timeout_success_when_server_responds_quickly() {
    let mut server = CaptureServer::start().await;
    let model = OpenAIRealtimeWebSocketModel::new();
    let _events = model
        .connect(config_for(server.url()).with_transport(
            RealtimeTransportConfig::new().with_handshake_timeout(Duration::from_secs(5)),
        ))
        .await
        .expect("a quick server fits a generous timeout");
    let _ = server.session_update().await;
    model.close().await.unwrap();
    server.handle.abort();
}

/// `test_handshake_timeout_with_delayed_server`
#[tokio::test]
async fn handshake_timeout_with_delayed_server() {
    let (port, accepted, handle) = slow_handshake_server(Duration::from_millis(500)).await;
    let url = format!("ws://127.0.0.1:{port}/v1/realtime");

    // A timeout shorter than the server's delay gives up.
    let fail = OpenAIRealtimeWebSocketModel::new();
    let error = fail
        .connect(config_for(url.clone()).with_transport(
            RealtimeTransportConfig::new().with_handshake_timeout(Duration::from_millis(200)),
        ))
        .await
        .expect_err("the client gives up before the server answers");
    assert!(error.to_string().contains("handshake timed out"), "{error}");
    assert!(accepted.load(Ordering::SeqCst) >= 1);

    // A longer one waits it out.
    let success = OpenAIRealtimeWebSocketModel::new();
    let _events = success
        .connect(config_for(url).with_transport(
            RealtimeTransportConfig::new().with_handshake_timeout(Duration::from_secs(5)),
        ))
        .await
        .expect("the client waits long enough");
    success.close().await.unwrap();
    handle.abort();
}

/// `test_ping_interval_comparison_fast_vs_slow`: both intervals keep a healthy connection.
#[tokio::test]
async fn ping_interval_comparison_fast_vs_slow() {
    for interval in [Duration::from_millis(50), Duration::from_millis(500)] {
        let (port, mut seen, handle) = recording_server(true).await;
        let model = OpenAIRealtimeWebSocketModel::new();
        let mut events = model
            .connect(
                config_for(format!("ws://127.0.0.1:{port}/v1/realtime")).with_transport(
                    RealtimeTransportConfig::new()
                        .with_ping_interval(interval)
                        .with_ping_timeout(Duration::from_secs(2)),
                ),
            )
            .await
            .unwrap();
        // The session.update the connect sends proves the socket is usable.
        tokio::time::timeout(TIMEOUT, async {
            while let Some(item) = seen.recv().await {
                if matches!(item, Seen::Text(_)) {
                    return;
                }
            }
        })
        .await
        .expect("session.update arrives");
        assert!(
            drain(&mut events)
                .iter()
                .all(|event| !matches!(event, RealtimeModelEvent::EndOfStream))
        );
        model.close().await.unwrap();
        handle.abort();
    }
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_websocket_message_size.py
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum SizeOutcome {
    Accepted,
    Rejected,
}

/// A `response.output_text.delta` event whose encoded size is exactly `size` bytes.
fn text_delta_of_size(size: usize) -> (String, String) {
    let mut payload = json!({
        "type": "response.output_text.delta",
        "event_id": "event_test",
        "response_id": "resp_test",
        "item_id": "item_test",
        "output_index": 0,
        "content_index": 0,
        "delta": "",
    });
    let overhead = payload.to_string().len();
    let delta = "x".repeat(size - overhead);
    payload["delta"] = Value::String(delta.clone());
    let message = payload.to_string();
    assert_eq!(message.len(), size);
    (message, delta)
}

/// Serve one oversized text frame after the client's `session.update`, then report what the
/// client did with it.
async fn size_limit_outcome(
    transport: Option<RealtimeTransportConfig>,
    message_size: usize,
) -> SizeOutcome {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (message, delta) = text_delta_of_size(message_size);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        // Wait for session.update before sending the response.
        let _ = ws.next().await;
        let _ = ws.send(Message::text(message)).await;
        while ws.next().await.is_some() {}
    });

    let model = OpenAIRealtimeWebSocketModel::new();
    let mut config = config_for(format!("ws://127.0.0.1:{port}/v1/realtime"));
    if let Some(transport) = transport {
        config = config.with_transport(transport);
    }
    let mut events = model.connect(config).await.unwrap();

    let mut text_events = 0;
    let mut raw_events = 0;
    let outcome = tokio::time::timeout(TIMEOUT, async {
        loop {
            match events
                .recv()
                .await
                .expect("the stream stays open until a result")
            {
                RealtimeModelEvent::OutputTextDelta {
                    delta: received, ..
                } => {
                    text_events += 1;
                    assert_eq!(received, delta);
                    return SizeOutcome::Accepted;
                }
                RealtimeModelEvent::Exception { .. } => return SizeOutcome::Rejected,
                RealtimeModelEvent::RawServerEvent { .. } => raw_events += 1,
                _ => {}
            }
        }
    })
    .await
    .expect("the client either accepts or rejects the message");
    if outcome == SizeOutcome::Rejected {
        // A rejected message is dropped before anything downstream sees it.
        assert_eq!(text_events, 0);
        assert_eq!(raw_events, 0);
    }
    model.close().await.unwrap();
    server.abort();
    outcome
}

/// `test_incoming_message_size_limit` (parametrized, one case per Python parameter set).
///
/// `{"max_size": None}` (unlimited) has no Rust spelling: `RealtimeTransportConfig::max_size` is
/// an `Option` where `None` means "use the default", so that case is not ported. The close code
/// the peer sees (1009) belongs to the websockets library and is not asserted.
#[tokio::test]
async fn incoming_message_size_limit() {
    const LIMIT: usize = 8 * 1024 * 1024;
    // default-at-limit
    assert_eq!(size_limit_outcome(None, LIMIT).await, SizeOutcome::Accepted);
    // default-over-limit
    assert_eq!(
        size_limit_outcome(None, LIMIT + 1).await,
        SizeOutcome::Rejected
    );
    // other-config-over-limit
    assert_eq!(
        size_limit_outcome(Some(RealtimeTransportConfig::new()), LIMIT + 1).await,
        SizeOutcome::Rejected
    );
    // explicit-smaller-limit
    assert_eq!(
        size_limit_outcome(
            Some(RealtimeTransportConfig::new().with_max_size(1024)),
            1025
        )
        .await,
        SizeOutcome::Rejected
    );
    // explicit-larger-limit
    assert_eq!(
        size_limit_outcome(
            Some(RealtimeTransportConfig::new().with_max_size(9 * 1024 * 1024)),
            LIMIT + 1
        )
        .await,
        SizeOutcome::Accepted
    );
}

/// The same limit applies to the SIP transport (the Python test is parametrized over both).
#[tokio::test]
async fn incoming_message_size_limit_applies_to_the_sip_model() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (message, _) = text_delta_of_size(2048);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        let _ = ws.next().await;
        let _ = ws.send(Message::text(message)).await;
        while ws.next().await.is_some() {}
    });
    let model = OpenAIRealtimeSIPModel::new();
    let mut events = model
        .connect(
            config_for(format!("ws://127.0.0.1:{port}/v1/realtime"))
                .with_call_id("call_test")
                .with_transport(RealtimeTransportConfig::new().with_max_size(1024)),
        )
        .await
        .unwrap();
    let rejected = tokio::time::timeout(TIMEOUT, async {
        loop {
            match events.recv().await.expect("stream open") {
                RealtimeModelEvent::Exception { .. } => return true,
                RealtimeModelEvent::OutputTextDelta { .. } => return false,
                _ => {}
            }
        }
    })
    .await
    .expect("a verdict arrives");
    assert!(rejected);
    model.close().await.unwrap();
    server.abort();
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_ga_session_update_normalization.py
// ---------------------------------------------------------------------------------------------

/// `test_no_auto_interrupt_on_vad_speech_started`: a VAD speech-start with no audio playing must
/// not interrupt anything.
#[tokio::test]
async fn no_auto_interrupt_on_vad_speech_started() {
    let mut server = MockRealtimeServer::start().await;
    let (model, _events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;

    server.push(json!({"type": "input_audio_buffer.speech_started"}));
    send_marker(&model).await;

    let frames = server.frames_before_marker().await;
    assert!(
        !frames.iter().any(|frame| matches!(
            frame["type"].as_str(),
            Some("conversation.item.truncate" | "response.cancel")
        )),
        "speech_started with no audio must not interrupt: {frames:#?}"
    );
    model.close().await.unwrap();
    server.stop();
}
