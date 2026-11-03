//! Ports of `tests/realtime/test_playback_tracker.py` cases that use the transport's own
//! playback estimate (no application tracker): the truncation point never exceeds the audio
//! received, an explicit interrupt and a VAD barge-in truncate at the same point, and audio that
//! arrives before any format was negotiated is measured as 24 kHz PCM16.
//!
//! Python patches `time.monotonic`; here the audio is short (50 ms) and the test waits real time
//! beyond it, so the clamp is what decides the truncation point. No network, no real API key.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::time::Duration;

use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeInterrupt, RealtimeModel, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimePlaybackTracker,
    RealtimeSessionModelSettings, RealtimeTurnDetectionConfig,
};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

const SENTINEL: &str = "input_audio_buffer.cleared";
/// 50 ms of PCM16 at 24 kHz.
const FIFTY_MS: usize = 2400;
/// One second of PCM16 at 24 kHz.
const SECOND: usize = 48_000;

async fn settle(
    server: &MockRealtimeServer,
    events: &mut RealtimeModelEvents,
) -> Vec<RealtimeModelEvent> {
    server.push(json!({"type": SENTINEL, "event_id": "barrier"}));
    let mut collected = Vec::new();
    tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = events.recv().await {
            match &event {
                RealtimeModelEvent::ConnectionStatus { .. } => {}
                RealtimeModelEvent::RawServerEvent { data } if data["type"] == SENTINEL => {}
                RealtimeModelEvent::Other { data } if data["type"] == SENTINEL => return,
                _ => collected.push(event),
            }
        }
        panic!("the event stream ended before the sentinel arrived");
    })
    .await
    .expect("the sentinel event arrived");
    collected
}

fn has_error(events: &[RealtimeModelEvent]) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            RealtimeModelEvent::Error { .. } | RealtimeModelEvent::Exception { .. }
        )
    })
}

fn audio_delta(item_id: &str, bytes: usize) -> Value {
    json!({
        "type": "response.output_audio.delta", "event_id": "event_audio",
        "response_id": "r1", "item_id": item_id, "output_index": 0, "content_index": 0,
        "delta": base64_encode(&vec![0u8; bytes]),
    })
}

fn created() -> Value {
    json!({"type": "response.created", "event_id": "e_created", "response": {
        "id": "r1", "object": "realtime.response", "status": "in_progress", "output": []}})
}

fn settings() -> RealtimeSessionModelSettings {
    RealtimeSessionModelSettings::new().with_turn_detection(RealtimeTurnDetectionConfig {
        kind: Some("semantic_vad".to_string()),
        interrupt_response: Some(false),
        ..Default::default()
    })
}

fn session_created_with_format(format: Option<Value>) -> Value {
    let mut audio = json!({"input": {"turn_detection": {
        "type": "semantic_vad", "interrupt_response": false}}});
    if let Some(format) = format {
        audio["output"] = json!({"format": format});
    }
    json!({"type": "session.created", "event_id": "event_created", "session": {
        "type": "realtime", "model": "gpt-realtime-2.1", "audio": audio}})
}

async fn connected(
    format: Option<Value>,
    tracker: Option<Arc<RealtimePlaybackTracker>>,
) -> (
    MockRealtimeServer,
    Arc<OpenAIRealtimeWebSocketModel>,
    RealtimeModelEvents,
) {
    let mut server = MockRealtimeServer::start().await;
    let (model, events) = connect(&server, Some(settings()), tracker).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created_with_format(format));
    (server, model, events)
}

async fn truncates(
    server: &mut MockRealtimeServer,
    model: &OpenAIRealtimeWebSocketModel,
) -> Vec<(String, i64)> {
    send_marker(model).await;
    server
        .frames_before_marker()
        .await
        .into_iter()
        .filter(|frame| frame["type"] == "conversation.item.truncate")
        .map(|frame| {
            (
                frame["item_id"].as_str().unwrap().to_string(),
                frame["audio_end_ms"].as_i64().unwrap(),
            )
        })
        .collect()
}

fn pcm16() -> Value {
    json!({"type": "audio/pcm", "rate": 24000})
}

/// Python: `test_interrupt_clamps_truncate_to_received_audio_while_response_ongoing`.
#[tokio::test]
async fn the_default_estimate_never_truncates_past_the_received_audio() {
    let (mut server, model, mut events) = connected(Some(pcm16()), None).await;
    server.push(created());
    server.push(audio_delta("item_1", FIFTY_MS));
    let _ = settle(&server, &mut events).await;
    tokio::time::sleep(Duration::from_millis(250)).await;

    model
        .send_event(RealtimeModelSendEvent::Interrupt(
            RealtimeInterrupt::forced(),
        ))
        .await
        .unwrap();
    let emitted = settle(&server, &mut events).await;
    assert!(!has_error(&emitted));
    assert_eq!(
        truncates(&mut server, &model).await,
        [("item_1".to_string(), 50)]
    );
    model.close().await.unwrap();
    server.stop();
}

/// Python: `test_interrupt_matches_speech_started_truncation_point`.
#[tokio::test]
async fn an_interrupt_and_a_speech_start_truncate_at_the_same_point() {
    let mut points = Vec::new();
    for explicit_interrupt in [true, false] {
        let (mut server, model, mut events) = connected(Some(pcm16()), None).await;
        server.push(created());
        server.push(audio_delta("item_1", FIFTY_MS));
        let _ = settle(&server, &mut events).await;
        tokio::time::sleep(Duration::from_millis(250)).await;
        if explicit_interrupt {
            model
                .send_event(RealtimeModelSendEvent::Interrupt(
                    RealtimeInterrupt::forced(),
                ))
                .await
                .unwrap();
        } else {
            server.push(json!({
                "type": "input_audio_buffer.speech_started", "event_id": "e1",
                "item_id": "user_2", "audio_start_ms": 0, "audio_end_ms": 0,
            }));
        }
        let _ = settle(&server, &mut events).await;
        let sent = truncates(&mut server, &model).await;
        assert_eq!(
            sent.len(),
            1,
            "explicit_interrupt={explicit_interrupt}: {sent:?}"
        );
        points.push(sent[0].clone());
        model.close().await.unwrap();
        server.stop();
    }
    assert_eq!(points[0], points[1]);
}

/// Python: `test_audio_delta_before_set_audio_format_does_not_raise`: with no format negotiated
/// the audio is measured as 24 kHz PCM16, so a playback position past the received second is
/// clamped to 1000 ms and nothing errors.
#[tokio::test]
async fn audio_before_a_negotiated_format_is_measured_as_pcm16() {
    let tracker = RealtimePlaybackTracker::shared();
    let (mut server, model, mut events) = connected(None, Some(tracker.clone())).await;
    server.push(created());
    server.push(audio_delta("item_1", SECOND));
    let emitted = settle(&server, &mut events).await;
    assert!(!has_error(&emitted), "{emitted:#?}");

    tracker.on_play_ms("item_1", 0, 5000.0);
    model
        .send_event(RealtimeModelSendEvent::Interrupt(
            RealtimeInterrupt::forced(),
        ))
        .await
        .unwrap();
    let emitted = settle(&server, &mut events).await;
    assert!(!has_error(&emitted), "{emitted:#?}");
    assert_eq!(
        truncates(&mut server, &model).await,
        [("item_1".to_string(), 1000)]
    );
    model.close().await.unwrap();
    server.stop();
}
