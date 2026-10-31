//! Ports of the server-event handling tests of `tests/realtime/test_openai_realtime.py`
//! (`TestEventHandlingRobustness`, the event half of `TestSendEventAndConfig`) and the usage
//! conversion tests of `tests/realtime/test_openai_realtime_conversions.py`.
//!
//! Python calls `_handle_ws_event` and reads what its listener received. Here each event goes
//! over a local websocket and the test reads the transport's public event stream; ordering is
//! made deterministic by pushing a sentinel event last, which the transport forwards as
//! `Other`. No network, no real API key.

#![cfg(feature = "realtime")]

use std::sync::Arc;

use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeAgent, RealtimeContent, RealtimeItem,
    RealtimeMessageRole, RealtimeModel, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimePlaybackTracker, RealtimeRunner,
};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

const SENTINEL: &str = "py.test.sentinel";

/// Everything the transport emitted for the events pushed so far, minus connection status.
///
/// A sentinel server event is pushed last; the transport handles events in order, so when the
/// sentinel's own events arrive everything before it has been reported.
async fn settle(
    server: &MockRealtimeServer,
    events: &mut RealtimeModelEvents,
) -> Vec<RealtimeModelEvent> {
    server.push(json!({"type": SENTINEL}));
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

fn kinds(events: &[RealtimeModelEvent]) -> Vec<&'static str> {
    events.iter().map(RealtimeModelEvent::event_type).collect()
}

async fn ready() -> (
    MockRealtimeServer,
    Arc<OpenAIRealtimeWebSocketModel>,
    RealtimeModelEvents,
) {
    let mut server = MockRealtimeServer::start().await;
    let (model, events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;
    (server, model, events)
}

fn pcm16() -> Value {
    json!({"type": "audio/pcm", "rate": 24000})
}

fn audio_delta(response_id: &str, item_id: &str, content_index: usize, bytes: &[u8]) -> Value {
    json!({
        "type": "response.output_audio.delta",
        "event_id": "event_audio",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": content_index,
        "delta": base64_encode(bytes),
    })
}

fn item_updated(
    events: &[RealtimeModelEvent],
) -> Vec<&openai_agents_sdk::realtime::RealtimeMessageItem> {
    events
        .iter()
        .filter_map(|event| match event {
            RealtimeModelEvent::ItemUpdated { item } => item.as_message(),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// TestEventHandlingRobustness
// ---------------------------------------------------------------------------------------------

/// `test_raw_event_preserves_null_previous_item_id`
#[tokio::test]
async fn raw_event_preserves_null_previous_item_id() {
    let (server, model, mut events) = ready().await;
    let server_event = json!({
        "type": "conversation.item.created",
        "event_id": "event_1",
        "previous_item_id": null,
        "item": {
            "id": "item_1",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    });
    server.push(server_event.clone());
    let emitted = settle(&server, &mut events).await;

    assert_eq!(kinds(&emitted), ["raw_server_event", "item_updated"]);
    match &emitted[0] {
        RealtimeModelEvent::RawServerEvent { data } => {
            assert_eq!(data, &server_event);
            assert!(data["previous_item_id"].is_null());
        }
        other => panic!("expected the raw event first, got {other:?}"),
    }
    // A null previous id is an absent one.
    assert_eq!(item_updated(&emitted)[0].previous_item_id, None);
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_malformed_json_logs_error_continues`: a frame that is not JSON becomes an
/// error event, and the connection keeps working.
#[tokio::test]
async fn handle_malformed_json_continues() {
    let (server, model, mut events) = ready().await;
    server.push_raw("invalid json {");
    server.push(json!({"type": "error", "error": {"message": "still alive"}}));
    let emitted = settle(&server, &mut events).await;

    // The frame is reported as a (non-fatal) error event and the next one is still handled.
    assert!(
        matches!(&emitted[0], RealtimeModelEvent::Error { error }
            if error["context"].as_str().is_some_and(|context| context.contains("JSON"))),
        "{emitted:#?}"
    );
    assert_eq!(kinds(&emitted)[1..], ["raw_server_event", "error"]);
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_invalid_event_schema_logs_error`: an audio delta missing its required fields is
/// reported, not emitted as empty audio.
#[tokio::test]
async fn handle_invalid_event_schema_emits_an_error() {
    let (server, model, mut events) = ready().await;
    server.push(json!({"type": "response.output_audio.delta"}));
    let emitted = settle(&server, &mut events).await;

    assert_eq!(
        kinds(&emitted),
        ["raw_server_event", "error"],
        "{emitted:#?}"
    );
    assert!(matches!(
        &emitted[1],
        RealtimeModelEvent::Error { error } if error["context"] == "Failed to validate server event"
    ));
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_unknown_event_type_ignored`: an unknown type is forwarded, never an error.
#[tokio::test]
async fn handle_unknown_event_type_ignored() {
    let (server, model, mut events) = ready().await;
    server.push(json!({"type": "unknown.event.type", "data": "some data"}));
    let emitted = settle(&server, &mut events).await;

    assert_eq!(kinds(&emitted), ["raw_server_event", "other"]);
    assert!(
        !emitted
            .iter()
            .any(|e| matches!(e, RealtimeModelEvent::Exception { .. }))
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_done_emits_typed_usage_before_turn_ended` (and the conversion half of
/// `test_convert_response_usage_preserves_modality_details`)
#[tokio::test]
async fn response_done_emits_typed_usage_before_turn_ended() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.done",
        "response": {
            "id": "response_1",
            "status": "cancelled",
            "usage": {
                "total_tokens": 20,
                "input_tokens": 12,
                "output_tokens": 8,
                "input_token_details": {"text_tokens": 2, "audio_tokens": 10, "cached_tokens": 4},
                "output_token_details": {"text_tokens": 1, "audio_tokens": 7},
            },
        },
    }));
    let emitted = settle(&server, &mut events).await;

    assert_eq!(kinds(&emitted), ["raw_server_event", "usage", "turn_ended"]);
    match &emitted[1] {
        RealtimeModelEvent::Usage {
            input_tokens_details: Some(input),
            ..
        } => assert_eq!(input.audio_tokens, Some(10)),
        other => panic!("expected usage with input details, got {other:?}"),
    }
    assert!(matches!(
        &emitted[2],
        RealtimeModelEvent::TurnEnded { response_id: Some(id) } if id == "response_1"
    ));
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_done_without_usage_skips_usage_event`
#[tokio::test]
async fn response_done_without_usage_skips_usage_event() {
    let (server, model, mut events) = ready().await;
    server.push(json!({"type": "response.done", "response": {}}));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(kinds(&emitted), ["raw_server_event", "turn_ended"]);
    model.close().await.unwrap();
    server.stop();
}

/// `test_convert_response_usage_preserves_modality_details`
#[tokio::test]
async fn convert_response_usage_preserves_modality_details() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.done",
        "response": {"id": "r", "usage": {
            "total_tokens": 253,
            "input_tokens": 132,
            "output_tokens": 121,
            "input_token_details": {
                "text_tokens": 119,
                "audio_tokens": 13,
                "image_tokens": 0,
                "cached_tokens": 64,
                "cached_tokens_details": {"text_tokens": 60, "audio_tokens": 4, "image_tokens": 0},
            },
            "output_token_details": {"text_tokens": 30, "audio_tokens": 91},
        }},
    }));
    let emitted = settle(&server, &mut events).await;
    let RealtimeModelEvent::Usage {
        usage,
        input_tokens_details,
        output_tokens_details,
    } = &emitted[1]
    else {
        panic!("expected a usage event, got {emitted:#?}");
    };
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.input_tokens, 132);
    assert_eq!(usage.output_tokens, 121);
    assert_eq!(usage.total_tokens, 253);
    assert_eq!(usage.input_tokens_details.cached_tokens, 64);
    let input = input_tokens_details.as_ref().expect("input details");
    assert_eq!(input.text_tokens, Some(119));
    assert_eq!(input.audio_tokens, Some(13));
    assert_eq!(input.image_tokens, Some(0));
    assert_eq!(input.cached_tokens, Some(64));
    let cached = input
        .cached_tokens_details
        .as_ref()
        .expect("cached details");
    assert_eq!(cached.text_tokens, Some(60));
    assert_eq!(cached.audio_tokens, Some(4));
    assert_eq!(cached.image_tokens, Some(0));
    let output = output_tokens_details.as_ref().expect("output details");
    assert_eq!(output.text_tokens, Some(30));
    assert_eq!(output.audio_tokens, Some(91));
    model.close().await.unwrap();
    server.stop();
}

/// `test_convert_response_usage_preserves_missing_details_and_derives_total`
#[tokio::test]
async fn convert_response_usage_preserves_missing_details_and_derives_total() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.done",
        "response": {"id": "r", "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "input_token_details": {"audio_tokens": 0},
        }},
    }));
    let emitted = settle(&server, &mut events).await;
    let RealtimeModelEvent::Usage {
        usage,
        input_tokens_details,
        output_tokens_details,
    } = &emitted[1]
    else {
        panic!("expected a usage event, got {emitted:#?}");
    };
    assert_eq!(usage.total_tokens, 15);
    let input = input_tokens_details.as_ref().expect("input details");
    assert_eq!(input.audio_tokens, Some(0));
    assert_eq!(input.text_tokens, None);
    assert_eq!(input.cached_tokens, None);
    assert!(input.cached_tokens_details.is_none());
    assert!(output_tokens_details.is_none());
    model.close().await.unwrap();
    server.stop();
}

/// `test_retrieved_completed_item_keeps_status`
#[tokio::test]
async fn retrieved_completed_item_keeps_status() {
    let (mut server, model, mut events) = ready().await;
    server.push(session_created(Some(pcm16())));
    server.push(audio_delta("resp_1", "item_1", 0, b"test audio"));
    server.push(json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "event_2",
        "item_id": "item_user",
        "content_index": 0,
        "transcript": "hello",
        "usage": {"type": "tokens", "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }));
    let first = settle(&server, &mut events).await;
    assert!(kinds(&first).contains(&"input_audio_transcription_completed"));
    send_marker(&model).await;

    // The transcription rewrote the user item, so the transport re-reads the audio item it is
    // playing, exactly once.
    let retrieves: Vec<Value> = server
        .frames_before_marker()
        .await
        .into_iter()
        .filter(|frame| frame["type"] == "conversation.item.retrieve")
        .collect();
    assert_eq!(retrieves.len(), 1, "{retrieves:#?}");
    assert_eq!(retrieves[0]["item_id"], "item_1");

    server.push(json!({
        "type": "conversation.item.retrieved",
        "event_id": "event_3",
        "item": {
            "id": "item_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_audio", "transcript": "hi there"}],
        },
    }));
    let emitted = settle(&server, &mut events).await;
    let updated = item_updated(&emitted);
    assert_eq!(
        updated.last().expect("an item update").status.as_deref(),
        Some("completed")
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_retrieved_completed_item_keeps_status_in_session_history`: the session reconciles a
/// retrieved item into history without losing the terminal status or the stored transcript.
#[tokio::test]
async fn retrieved_completed_item_keeps_status_in_session_history() {
    let mut server = MockRealtimeServer::start().await;
    let model = Arc::new(OpenAIRealtimeWebSocketModel::new());
    let session = RealtimeRunner::new(RealtimeAgent::<()>::new("agent").into_arc())
        .with_model(model.clone())
        .with_model_config(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_url(server.url()),
        )
        .run(())
        .await
        .expect("the session connects");
    let _ = server.next_frame("session.update").await;

    server.push(json!({
        "type": "conversation.item.created",
        "event_id": "event_0",
        "item": {
            "id": "item_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_audio", "transcript": "hi there"}],
        },
    }));
    // Wait for the item to land in history, then retrieve it without a transcript.
    wait_for_history(&session, |history| history.len() == 1).await;
    server.push(json!({
        "type": "conversation.item.retrieved",
        "event_id": "event_1",
        "item": {
            "id": "item_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_audio"}],
        },
    }));
    // The retrieve is the second history change for item_1.
    wait_for_history(&session, |history| {
        history
            .first()
            .and_then(RealtimeItem::as_message)
            .is_some_and(|message| message.status.as_deref() == Some("completed"))
    })
    .await;
    // Give the retrieved event time to be processed, ordered behind a sentinel.
    server.push(json!({"type": SENTINEL}));
    wait_for_session_raw_sentinel(&session).await;

    let history = session.history();
    assert_eq!(history.len(), 1);
    let stored = history[0].as_message().expect("a message item");
    assert_eq!(stored.status.as_deref(), Some("completed"));
    assert!(matches!(
        stored.content.first(),
        Some(RealtimeContent::Audio { transcript: Some(transcript), .. }) if transcript == "hi there"
    ));
    session.close().await.unwrap();
    server.stop();
}

async fn wait_for_history<F>(session: &openai_agents_sdk::realtime::RealtimeSession<()>, ready: F)
where
    F: Fn(&[RealtimeItem]) -> bool,
{
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if ready(&session.history()) {
                return;
            }
            session.recv().await.expect("the session stays open");
        }
    })
    .await
    .expect("history reached the expected state");
}

async fn wait_for_session_raw_sentinel(session: &openai_agents_sdk::realtime::RealtimeSession<()>) {
    use openai_agents_sdk::realtime::RealtimeSessionEvent;
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if let RealtimeSessionEvent::RawModelEvent {
                data: RealtimeModelEvent::Other { data },
                ..
            } = session.recv().await.expect("the session stays open")
                && data["type"] == SENTINEL
            {
                return;
            }
        }
    })
    .await
    .expect("the sentinel reached the session");
}

/// `test_handle_audio_delta_event_success`
#[tokio::test]
async fn handle_audio_delta_event_success() {
    let (server, model, mut events) = ready().await;
    server.push(session_created(Some(pcm16())));
    server.push(audio_delta("resp_123", "item_456", 0, b"test audio"));
    let emitted = settle(&server, &mut events).await;

    let audio: Vec<_> = emitted
        .iter()
        .filter_map(|event| match event {
            RealtimeModelEvent::Audio(audio) => Some(audio),
            _ => None,
        })
        .collect();
    assert_eq!(audio.len(), 1);
    assert_eq!(audio[0].response_id, "resp_123");
    assert_eq!(audio[0].item_id, "item_456");
    assert_eq!(audio[0].data, b"test audio");
    // The raw event precedes the audio event.
    let raw = kinds(&emitted)
        .iter()
        .position(|kind| *kind == "raw_server_event");
    let aud = kinds(&emitted).iter().position(|kind| *kind == "audio");
    assert!(raw < aud);
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_audio_delta_state_management`: a delta on content index 5 is tracked under that
/// index, so an interruption truncates exactly that content part.
#[tokio::test]
async fn handle_audio_delta_state_management() {
    let mut server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created(Some(pcm16())));
    server.push(audio_delta("resp_123", "test_item", 5, b"test"));
    let _ = settle(&server, &mut events).await;

    model
        .send_event(RealtimeModelSendEvent::Interrupt(Default::default()))
        .await
        .unwrap();
    let emitted = settle(&server, &mut events).await;
    assert!(emitted.iter().any(|event| matches!(
        event,
        RealtimeModelEvent::AudioInterrupted { item_id, content_index: 5 } if item_id == "test_item"
    )));
    send_marker(&model).await;
    let truncates: Vec<Value> = server
        .frames_before_marker()
        .await
        .into_iter()
        .filter(|frame| frame["type"] == "conversation.item.truncate")
        .collect();
    assert_eq!(truncates.len(), 1);
    assert_eq!(truncates[0]["item_id"], "test_item");
    assert_eq!(truncates[0]["content_index"], 5);
    model.close().await.unwrap();
    server.stop();
}

/// `test_backward_compat_output_item_added_and_done`
#[tokio::test]
async fn backward_compat_output_item_added_and_done() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_item.added",
        "item": {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "audio", "audio": "...", "transcript": "hi"},
            ],
        },
    }));
    server.push(json!({
        "type": "response.output_item.done",
        "item": {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "bye"}],
        },
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(
        kinds(&emitted)
            .iter()
            .filter(|kind| **kind == "item_updated")
            .count(),
        2
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_text_mode_output_item_content`
#[tokio::test]
async fn text_mode_output_item_content() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_item.added",
        "item": {
            "id": "text_item_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "test data"}],
        },
    }));
    let emitted = settle(&server, &mut events).await;
    let updated = item_updated(&emitted);
    assert!(!updated.is_empty());
    assert_eq!(updated[0].role, RealtimeMessageRole::Assistant);
    assert!(matches!(
        updated[0].content.first(),
        Some(RealtimeContent::Text { text: Some(text), .. }) if text == "test data"
    ));
    model.close().await.unwrap();
    server.stop();
}

/// `test_output_text_delta_emits_normalized_event`
#[tokio::test]
async fn output_text_delta_emits_normalized_event() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_text.delta",
        "event_id": "event_1",
        "response_id": "response_1",
        "item_id": "item_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
    }));
    let emitted = settle(&server, &mut events).await;
    let normalized: Vec<_> = emitted
        .iter()
        .filter(|event| matches!(event, RealtimeModelEvent::OutputTextDelta { .. }))
        .collect();
    assert_eq!(
        normalized,
        [&RealtimeModelEvent::OutputTextDelta {
            item_id: "item_1".to_string(),
            delta: "hello".to_string(),
            response_id: "response_1".to_string(),
        }]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_output_audio_content_type_normalized`
#[tokio::test]
async fn output_audio_content_type_normalized() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_item.added",
        "item": {
            "id": "audio_item_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_audio", "audio": "base64data", "transcript": "hi"}],
        },
    }));
    let emitted = settle(&server, &mut events).await;
    let updated = item_updated(&emitted);
    assert_eq!(updated[0].content.len(), 1);
    assert!(matches!(
        &updated[0].content[0],
        RealtimeContent::Audio { transcript: Some(transcript), .. } if transcript == "hi"
    ));
    model.close().await.unwrap();
    server.stop();
}

/// `test_transcription_related_and_timeouts_and_speech_started`
#[tokio::test]
async fn transcription_related_and_timeouts_and_speech_started() {
    let mut server = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created(Some(pcm16())));
    server.push(audio_delta("r1", "i1", 0, &[b'a'; 96]));

    // Speech started interrupts the audio and truncates it where the server heard the user.
    server.push(json!({
        "type": "input_audio_buffer.speech_started",
        "event_id": "es1",
        "item_id": "i1",
        "audio_start_ms": 0,
        "audio_end_ms": 1,
    }));
    let truncate = server.next_frame("conversation.item.truncate").await;
    assert_eq!(truncate["item_id"], "i1");
    assert_eq!(truncate["content_index"], 0);
    assert_eq!(truncate["audio_end_ms"], 1);

    server.push(json!({
        "type": "response.output_audio_transcript.delta",
        "event_id": "e3",
        "item_id": "i3",
        "response_id": "r3",
        "output_index": 0,
        "content_index": 0,
        "delta": "abc",
    }));
    server.push(json!({
        "type": "input_audio_buffer.timeout_triggered",
        "event_id": "e4",
        "item_id": "i4",
        "audio_start_ms": 0,
        "audio_end_ms": 100,
    }));
    let emitted = settle(&server, &mut events).await;
    let seen = kinds(&emitted);
    assert!(seen.contains(&"audio_interrupted"), "{seen:?}");
    assert!(seen.contains(&"transcript_delta"), "{seen:?}");
    assert!(seen.contains(&"input_audio_timeout_triggered"), "{seen:?}");
    model.close().await.unwrap();
    server.stop();
}

/// `test_speech_started_skips_truncate_when_audio_complete`: playback finished (reported by the
/// tracker, where Python rewinds the arrival clock), so there is nothing left to truncate.
#[tokio::test]
async fn speech_started_skips_truncate_when_audio_complete() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created(Some(pcm16())));
    server.push(audio_delta("r1", "i1", 0, &[b'a'; 48_000]));
    let _ = settle(&server, &mut events).await;
    tracker.on_play_ms("i1", 0, 5_000.0);

    server.push(json!({
        "type": "input_audio_buffer.speech_started",
        "event_id": "es2",
        "item_id": "i1",
        "audio_start_ms": 0,
        "audio_end_ms": 0,
    }));
    let _ = settle(&server, &mut events).await;
    send_marker(&model).await;
    let frames = server.frames_before_marker().await;
    assert!(
        !frames
            .iter()
            .any(|frame| frame["type"] == "conversation.item.truncate"),
        "{frames:#?}"
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_speech_started_truncates_when_response_ongoing`
#[tokio::test]
async fn speech_started_truncates_when_response_ongoing() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created(Some(pcm16())));
    server.push(json!({"type": "response.created", "response": {"id": "r1"}}));
    server.push(audio_delta("r1", "i1", 0, &[b'a'; 48_000]));
    let _ = settle(&server, &mut events).await;
    tracker.on_play_ms("i1", 0, 5_000.0);

    server.push(json!({
        "type": "input_audio_buffer.speech_started",
        "event_id": "es3",
        "item_id": "i1",
        "audio_start_ms": 0,
        "audio_end_ms": 0,
    }));
    // The ongoing response owns the audio, so it is truncated at the one second received.
    let truncate = server.next_frame("conversation.item.truncate").await;
    assert_eq!(truncate["audio_end_ms"], 1000);
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_error_event_success`
#[tokio::test]
async fn handle_error_event_success() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "error",
        "event_id": "event_456",
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_api_key",
            "message": "Invalid API key provided",
        },
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(kinds(&emitted), ["raw_server_event", "error"]);
    assert!(matches!(
        &emitted[1],
        RealtimeModelEvent::Error { error } if error["code"] == "invalid_api_key"
    ));
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_tool_call_event_success`
#[tokio::test]
async fn handle_tool_call_event_success() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_item.done",
        "event_id": "event_789",
        "response_id": "resp_789",
        "output_index": 0,
        "item": {
            "id": "call_123",
            "call_id": "call_123",
            "type": "function_call",
            "status": "completed",
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\"}",
        },
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(
        kinds(&emitted),
        ["raw_server_event", "item_updated", "function_call"]
    );
    match &emitted[2] {
        RealtimeModelEvent::FunctionCall(call) => {
            assert_eq!(call.name, "get_weather");
            assert_eq!(call.arguments, "{\"location\": \"San Francisco\"}");
            assert_eq!(call.call_id, "call_123");
        }
        other => panic!("expected a function call, got {other:?}"),
    }
    model.close().await.unwrap();
    server.stop();
}
