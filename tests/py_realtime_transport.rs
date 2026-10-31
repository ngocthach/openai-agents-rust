//! Ports of the Python SDK's transport-level realtime tests.
//!
//! Covers `tests/realtime/test_item_parsing.py`, the transport half of
//! `tests/realtime/test_playback_tracker.py`, and `tests/realtime/test_tracing.py`.
//!
//! Python reaches into `OpenAIRealtimeWebSocketModel` privates (`_handle_ws_event`,
//! `_send_raw_message`, `_audio_state_tracker`). The Rust port keeps those private, so each test
//! drives the same behavior through the public transport against a local mock websocket server
//! and asserts on the JSON frames that actually go over the wire. No network, no real API key.

#![cfg(feature = "realtime")]

use openai_agents_sdk::realtime::{
    RealtimeContent, RealtimeItem, RealtimeMessageRole, RealtimeModel, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimeModelTracing, RealtimeModelTracingConfig,
    RealtimePlaybackTracker, RealtimeSessionModelSettings,
};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_item_parsing.py
// ---------------------------------------------------------------------------------------------

/// Push `item` as a `conversation.item.created` and return the message the transport mapped.
async fn item_round_trip(item: Value) -> openai_agents_sdk::realtime::RealtimeMessageItem {
    let server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    server.push(session_created(None));
    server.push(json!({
        "type": "conversation.item.created",
        "event_id": "event_1",
        "item": item,
    }));

    let message = wait_for_event(&mut events, |event| match event {
        RealtimeModelEvent::ItemUpdated { item } => match *item {
            RealtimeItem::Message(message) => Some(message),
            _ => None,
        },
        _ => None,
    })
    .await;

    model.close().await.unwrap();
    server.stop();
    message
}

#[tokio::test]
async fn user_message_conversion() {
    // A text part with no text at all still produces a user message.
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text"}],
    }))
    .await;
    assert_eq!(message.role, RealtimeMessageRole::User);
    assert_eq!(message.item_id, "123");

    // So does an audio part with no audio.
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_audio"}],
    }))
    .await;
    assert_eq!(message.role, RealtimeMessageRole::User);
    assert!(matches!(
        message.content.first(),
        Some(RealtimeContent::InputAudio { audio: None, .. })
    ));
}

#[tokio::test]
async fn assistant_message_conversion() {
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text"}],
    }))
    .await;
    assert_eq!(message.role, RealtimeMessageRole::Assistant);
    // `output_text` is normalized to `text`, as Python does for backward compatibility.
    assert!(matches!(
        message.content.first(),
        Some(RealtimeContent::Text { text: None, .. })
    ));
}

#[tokio::test]
async fn system_message_conversion() {
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text"}],
    }))
    .await;
    assert_eq!(message.role, RealtimeMessageRole::System);
}

#[tokio::test]
async fn message_status_is_preserved() {
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello"}],
    }))
    .await;
    assert_eq!(message.status.as_deref(), Some("completed"));
}

#[tokio::test]
async fn message_status_defaults_to_in_progress_when_absent() {
    let message = item_round_trip(json!({
        "id": "123",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }))
    .await;
    assert_eq!(message.status.as_deref(), Some("in_progress"));
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_playback_tracker.py (transport half)
// ---------------------------------------------------------------------------------------------

/// PCM16 at 24 kHz: 48_000 bytes is exactly one second.
const ONE_SECOND_PCM16: usize = 48_000;

/// Negotiate pcm16 and deliver `bytes` of audio for `item_1`, then wait for it to arrive.
async fn deliver_audio(
    server: &MockRealtimeServer,
    events: &mut RealtimeModelEvents,
    bytes: usize,
) {
    server.push(session_created(Some(
        json!({"type": "audio/pcm", "rate": 24000}),
    )));
    server.push(json!({
        "type": "response.output_audio.delta",
        "item_id": "item_1",
        "content_index": 0,
        "response_id": "resp_1",
        "delta": base64_encode(&vec![0u8; bytes]),
    }));
    let len = wait_for_event(events, |event| match event {
        RealtimeModelEvent::Audio(audio) => Some(audio.data.len()),
        _ => None,
    })
    .await;
    assert_eq!(len, bytes);
}

#[tokio::test]
async fn interrupt_skipped_when_no_audio_playing() {
    let mut server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    server.push(session_created(None));
    // Let the connection settle without any audio ever arriving.
    let _ = server.next_frame("session.update").await;

    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();
    send_marker(&model).await;

    // With no audio playing there is nothing to truncate.
    let frames = server.frames_before_marker().await;
    assert!(
        !frames
            .iter()
            .any(|frame| frame["type"] == "conversation.item.truncate"),
        "an interrupt with no audio playing must not truncate anything: {frames:#?}"
    );
    let _ = &mut events;
    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn interrupt_skips_when_elapsed_exceeds_audio_length() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    deliver_audio(&server, &mut events, ONE_SECOND_PCM16).await;

    // The application reports more playback than the model ever sent, so the item is finished.
    tracker.on_play_ms("item_1", 0, 2000.0);
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();
    send_marker(&model).await;

    let frames = server.frames_before_marker().await;
    assert!(
        !frames
            .iter()
            .any(|frame| frame["type"] == "conversation.item.truncate"),
        "playback past the end of the audio must not produce a truncate: {frames:#?}"
    );
    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn interrupt_sends_truncate_when_a_response_is_ongoing() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;

    // An ongoing response owns the audio, so the interrupt must still truncate it even though
    // the reported playback has run past the end.
    server.push(session_created(Some(
        json!({"type": "audio/pcm", "rate": 24000}),
    )));
    server.push(json!({"type": "response.created", "response": {"id": "resp_1"}}));
    server.push(json!({
        "type": "response.output_audio.delta",
        "item_id": "item_1",
        "content_index": 0,
        "response_id": "resp_1",
        "delta": base64_encode(&vec![0u8; ONE_SECOND_PCM16]),
    }));
    let len = wait_for_event(&mut events, |event| match event {
        RealtimeModelEvent::Audio(audio) => Some(audio.data.len()),
        _ => None,
    })
    .await;
    assert_eq!(len, ONE_SECOND_PCM16);

    tracker.on_play_ms("item_1", 0, 2000.0);
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();

    let truncate = server.next_frame("conversation.item.truncate").await;
    assert_eq!(truncate["item_id"], "item_1");
    // The truncation point is clamped to the audio the client actually received; the API
    // rejects an `audio_end_ms` past the item's duration.
    assert_eq!(truncate["audio_end_ms"], 1000);

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn interrupt_clamps_truncate_to_received_audio() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    deliver_audio(&server, &mut events, ONE_SECOND_PCM16).await;

    // Report playback just short of the end, so a truncate is still due.
    tracker.on_play_ms("item_1", 0, 999.0);
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();

    let truncate = server.next_frame("conversation.item.truncate").await;
    assert_eq!(truncate["item_id"], "item_1");
    assert_eq!(truncate["content_index"], 0);
    // The Realtime API rejects a truncate past the item's duration, so the value is clamped to
    // the audio the client actually received.
    let end_ms = truncate["audio_end_ms"].as_f64().expect("audio_end_ms");
    assert!(
        (0.0..=1000.0).contains(&end_ms),
        "audio_end_ms {end_ms} must stay within the 1000 ms received"
    );

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn audio_state_accumulates_across_deltas() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;

    server.push(session_created(Some(
        json!({"type": "audio/pcm", "rate": 24000}),
    )));
    // Two half-second deltas for the same item accumulate into one second of audio.
    for _ in 0..2 {
        server.push(json!({
            "type": "response.output_audio.delta",
            "item_id": "item_1",
            "content_index": 0,
            "response_id": "resp_1",
            "delta": base64_encode(&vec![0u8; ONE_SECOND_PCM16 / 2]),
        }));
    }
    let mut seen = 0usize;
    while seen < 2 {
        seen += wait_for_event(&mut events, |event| match event {
            RealtimeModelEvent::Audio(_) => Some(1usize),
            _ => None,
        })
        .await;
    }

    // Playing 750 ms is inside the accumulated second, so a truncate is still due. Had the two
    // deltas not accumulated, the item would look like half a second and the interrupt would be
    // skipped as already finished.
    tracker.on_play_ms("item_1", 0, 750.0);
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();
    let truncate = server.next_frame("conversation.item.truncate").await;
    assert_eq!(truncate["audio_end_ms"], 750);

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn audio_delta_before_a_negotiated_format_does_not_fail() {
    let server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;

    // No audio format in session.created, which is what a transcription-only session looks like.
    server.push(session_created(None));
    server.push(json!({
        "type": "response.output_audio.delta",
        "item_id": "item_1",
        "content_index": 0,
        "response_id": "resp_1",
        "delta": base64_encode(b"test"),
    }));

    // The delta is still mapped; the length math falls back to PCM16.
    let audio = wait_for_event(&mut events, |event| match event {
        RealtimeModelEvent::Audio(audio) => Some(audio),
        _ => None,
    })
    .await;
    assert_eq!(audio.data, b"test");
    assert_eq!(audio.item_id, "item_1");

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn speech_started_truncates_at_the_same_point_as_an_interrupt() {
    // Explicit interrupts and VAD barge-in must truncate at the same place.
    async fn truncate_ms(via_speech_started: bool) -> f64 {
        let mut server = MockRealtimeServer::start().await;
        let tracker = RealtimePlaybackTracker::shared();
        let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
        deliver_audio(&server, &mut events, ONE_SECOND_PCM16).await;
        tracker.on_play_ms("item_1", 0, 400.0);

        if via_speech_started {
            server.push(json!({
                "type": "input_audio_buffer.speech_started",
                "event_id": "e1",
                "item_id": "item_1",
                "audio_start_ms": 0,
                "audio_end_ms": 0,
            }));
        } else {
            model
                .send_event(RealtimeModelSendEvent::interrupt())
                .await
                .unwrap();
        }

        let truncate = server.next_frame("conversation.item.truncate").await;
        let end_ms = truncate["audio_end_ms"].as_f64().expect("audio_end_ms");
        model.close().await.unwrap();
        server.stop();
        end_ms
    }

    assert_eq!(truncate_ms(true).await, truncate_ms(false).await);
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_tracing.py
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn tracing_config_is_sent_on_session_created() {
    let mut server = MockRealtimeServer::start().await;
    let settings = RealtimeSessionModelSettings::new().with_tracing(RealtimeModelTracing::Config(
        RealtimeModelTracingConfig {
            workflow_name: Some("test_workflow".to_string()),
            group_id: Some("group_123".to_string()),
            metadata: None,
        },
    ));
    let (model, _events) = connect(&server, Some(settings), None).await;

    // The initial session.update comes first; the tracing one follows session.created.
    let _initial = server.next_frame("session.update").await;
    server.push(session_created(None));

    let tracing_update = tokio::time::timeout(TIMEOUT, async {
        while let Some(value) = server.client_rx.recv().await {
            if value["type"] == "session.update" && !value["session"]["tracing"].is_null() {
                return value;
            }
        }
        panic!("the tracing session.update never arrived");
    })
    .await
    .expect("tracing session.update received");

    let tracing = &tracing_update["session"]["tracing"];
    assert_eq!(tracing["workflow_name"], "test_workflow");
    assert_eq!(tracing["group_id"], "group_123");

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn tracing_config_with_metadata_is_serialized() {
    let mut server = MockRealtimeServer::start().await;
    let metadata = json!({
        "user_id": "user_123",
        "session_type": "demo",
        "features": ["audio", "tools"],
        "config": {"timeout": 30, "retries": 3},
    });
    let settings = RealtimeSessionModelSettings::new().with_tracing(RealtimeModelTracing::Config(
        RealtimeModelTracingConfig {
            workflow_name: Some("complex_workflow".to_string()),
            group_id: None,
            metadata: Some(metadata.clone()),
        },
    ));
    let (model, _events) = connect(&server, Some(settings), None).await;

    let _initial = server.next_frame("session.update").await;
    server.push(session_created(None));

    let tracing_update = tokio::time::timeout(TIMEOUT, async {
        while let Some(value) = server.client_rx.recv().await {
            if value["type"] == "session.update" && !value["session"]["tracing"].is_null() {
                return value;
            }
        }
        panic!("the tracing session.update never arrived");
    })
    .await
    .expect("tracing session.update received");

    let tracing = &tracing_update["session"]["tracing"];
    assert_eq!(tracing["workflow_name"], "complex_workflow");
    assert_eq!(tracing["metadata"], metadata);

    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn tracing_disabled_skips_the_session_update() {
    let mut server = MockRealtimeServer::start().await;
    let settings = RealtimeSessionModelSettings::new().with_tracing(RealtimeModelTracing::Disabled);
    let (model, _events) = connect(&server, Some(settings), None).await;

    let _initial = server.next_frame("session.update").await;
    server.push(session_created(None));
    send_marker(&model).await;

    // Python's `None` tracing means no frame at all, not a frame carrying a null. Nothing may
    // arrive between the initial session.update and the marker.
    let frames = server.frames_before_marker().await;
    assert!(
        frames.is_empty(),
        "tracing disabled must not push any session.update after session.created: {frames:#?}"
    );

    model.close().await.unwrap();
    server.stop();
}
