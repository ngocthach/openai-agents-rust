//! Wire validation of the OpenAI realtime transport.
//!
//! Ports `test_try_convert_raw_message_invalid_returns_none`
//! (`tests/realtime/test_openai_realtime_conversions.py`), the malformed half of
//! `test_forwarded_provider_errors_have_payload_free_debug_summaries`
//! (`tests/realtime/test_app_server_debug.py`) and the two log-redaction tests for an invalid
//! server event (`tests/realtime/test_openai_realtime.py`). Python validates every server event
//! against the full event union and every raw client message against the client event union, so
//! both are checked here rather than only the audio delta.
#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::debug::{reset_dont_log_model_data, set_dont_log_model_data};
use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeModel, RealtimeModelEvent, RealtimeModelEvents,
    RealtimeModelSendEvent,
};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

/// The redaction flag and the log sink are process-global, so the tests take turns.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct LogSink(Mutex<Vec<(log::Level, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// Restores the redaction flag to the environment default when a test ends.
struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
    }
}

/// A server event the transport recognises but does not act on: the barrier that shows every
/// event pushed before it was handled.
const BARRIER: &str = "input_audio_buffer.cleared";

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

/// Everything the transport emitted for the events pushed so far, minus connection status.
async fn settle(
    server: &MockRealtimeServer,
    events: &mut RealtimeModelEvents,
) -> Vec<RealtimeModelEvent> {
    server.push(json!({"type": BARRIER, "event_id": "barrier"}));
    let mut collected = Vec::new();
    tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = events.recv().await {
            match &event {
                RealtimeModelEvent::ConnectionStatus { .. } => {}
                RealtimeModelEvent::RawServerEvent { data } if data["type"] == BARRIER => {}
                RealtimeModelEvent::Other { data } if data["type"] == BARRIER => return,
                _ => collected.push(event),
            }
        }
        panic!("the event stream ended before the barrier arrived");
    })
    .await
    .expect("the barrier event arrived");
    collected
}

fn kinds(events: &[RealtimeModelEvent]) -> Vec<&'static str> {
    events.iter().map(RealtimeModelEvent::event_type).collect()
}

async fn send_raw(model: &OpenAIRealtimeWebSocketModel, message: Value) {
    model
        .send_event(RealtimeModelSendEvent::RawMessage { message })
        .await
        .expect("a raw message that fails conversion is dropped, not an error");
}

/// `test_try_convert_raw_message_invalid_returns_none`: a raw message that is not a client event
/// never reaches the wire, and sending it is not an error.
#[tokio::test]
async fn raw_message_of_an_unknown_type_is_dropped() {
    let _turn = SERIAL.lock().await;
    let (mut server, model, _events) = ready().await;
    send_raw(&model, json!({"type": "invalid.event", "other_data": {}})).await;
    send_raw(&model, json!({"no_type": true})).await;
    send_marker(&model).await;
    let frames = server.frames_before_marker().await;
    assert!(frames.is_empty(), "nothing is sent: {frames:?}");
    model.close().await.unwrap();
    server.stop();
}

/// A known client event type with a required field missing fails the same validation.
#[tokio::test]
async fn raw_message_missing_a_required_field_is_dropped() {
    let _turn = SERIAL.lock().await;
    let (mut server, model, _events) = ready().await;
    send_raw(
        &model,
        json!({"type": "conversation.item.truncate", "item_id": "item_1", "content_index": 0}),
    )
    .await;
    send_raw(&model, json!({"type": "input_audio_buffer.append"})).await;
    send_raw(&model, json!({"type": "session.update", "session": "text"})).await;
    send_marker(&model).await;
    let frames = server.frames_before_marker().await;
    assert!(frames.is_empty(), "nothing is sent: {frames:?}");
    model.close().await.unwrap();
    server.stop();
}

/// A valid raw message is forwarded as given, extra fields included.
#[tokio::test]
async fn raw_message_of_a_known_type_is_forwarded() {
    let (mut server, model, _events) = ready().await;
    send_raw(
        &model,
        json!({"type": "conversation.item.delete", "item_id": "item_9", "event_id": "mine"}),
    )
    .await;
    let frame = server.next_frame("conversation.item.delete").await;
    assert_eq!(frame["item_id"], "item_9");
    assert_eq!(frame["event_id"], "mine");
    model.close().await.unwrap();
    server.stop();
}

/// Python validates every server event type, not only the audio delta: each event here is
/// complete except for one field, and each is reported as a validation error.
#[tokio::test]
async fn every_known_server_event_type_is_validated() {
    let _turn = SERIAL.lock().await;
    let (server, model, mut events) = ready().await;
    let malformed = [
        json!({"type": "response.output_audio.done", "event_id": "e", "item_id": "i",
               "output_index": 0, "content_index": 0}),
        json!({"type": "response.output_audio_transcript.delta", "event_id": "e", "item_id": "i",
               "response_id": "r", "output_index": 0, "content_index": 0}),
        json!({"type": "response.output_text.delta", "event_id": "e", "item_id": "i",
               "response_id": "r", "output_index": 0, "delta": "x"}),
        json!({"type": "response.created", "event_id": "e"}),
        json!({"type": "response.done", "response": {}}),
        json!({"type": "session.updated", "event_id": "e"}),
        json!({"type": "error", "event_id": "e", "error": {"message": "m"}}),
        json!({"type": "conversation.item.deleted", "event_id": "e"}),
        json!({"type": "conversation.item.created", "event_id": "e"}),
        json!({"type": "conversation.item.truncated", "event_id": "e", "item_id": "i",
               "content_index": 0}),
        json!({"type": "input_audio_buffer.speech_started", "event_id": "e", "item_id": "i"}),
        json!({"type": "input_audio_buffer.timeout_triggered", "event_id": "e", "item_id": "i",
               "audio_start_ms": 0}),
        json!({"type": "response.output_item.done", "event_id": "e", "response_id": "r",
               "output_index": 0}),
        json!({"type": "rate_limits.updated", "event_id": "e", "rate_limits": {}}),
        json!({"type": "response.output_audio_transcript.done", "event_id": "e", "item_id": "i",
               "response_id": "r", "output_index": 0, "content_index": 0, "transcript": 3}),
        json!({"type": "response.output_audio.delta", "event_id": "e", "item_id": "i",
               "response_id": "r", "output_index": 0, "content_index": "zero", "delta": "AA=="}),
    ];
    for event in &malformed {
        server.push(event.clone());
        let emitted = settle(&server, &mut events).await;
        assert_eq!(
            kinds(&emitted),
            ["raw_server_event", "error"],
            "{event}: {emitted:#?}"
        );
        assert!(matches!(
            &emitted[1],
            RealtimeModelEvent::Error { error }
                if error["context"] == "Failed to validate server event"
        ));
    }
    model.close().await.unwrap();
    server.stop();
}

/// A complete event of a known type that the transport does not act on is not an error.
#[tokio::test]
async fn a_complete_known_event_is_not_reported() {
    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_audio_transcript.done", "event_id": "e", "item_id": "i",
        "response_id": "r", "output_index": 0, "content_index": 0, "transcript": "hello",
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(kinds(&emitted), ["raw_server_event", "other"]);
    model.close().await.unwrap();
    server.stop();
}

/// The malformed half of `test_forwarded_provider_errors_have_payload_free_debug_summaries`
/// (both `dont_log_model_data` values): a `transcript` that is a list is a validation error, and
/// no log line carries the payload when model data is redacted.
async fn malformed_transcript_is_reported(redacted: bool) {
    let _turn = SERIAL.lock().await;
    let sink = sink();
    sink.0.lock().unwrap().clear();
    let _policy = PolicyGuard::set(redacted);
    let marker = "synthetic-private-transcript";

    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "response.output_audio_transcript.done",
        "event_id": "event-1",
        "item_id": "item-1",
        "response_id": "response-1",
        "output_index": 0,
        "content_index": 0,
        "transcript": [marker],
    }));
    let emitted = settle(&server, &mut events).await;

    assert_eq!(
        kinds(&emitted),
        ["raw_server_event", "error"],
        "{emitted:#?}"
    );
    let RealtimeModelEvent::Error { error } = &emitted[1] else {
        unreachable!()
    };
    assert_eq!(error["context"], "Failed to validate server event");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("`transcript` must be a string")
    );

    let lines = sink.0.lock().unwrap().clone();
    let failures: Vec<_> = lines
        .iter()
        .filter(|(level, line)| {
            *level == log::Level::Error && line.starts_with("Failed to validate server event")
        })
        .collect();
    assert_eq!(failures.len(), 1, "{lines:?}");
    if redacted {
        assert_eq!(failures[0].1, "Failed to validate server event");
        assert!(
            lines.iter().all(|(_, line)| !line.contains(marker)),
            "{lines:?}"
        );
    } else {
        assert!(failures[0].1.contains(marker), "{lines:?}");
    }
    model.close().await.unwrap();
    server.stop();
}

#[tokio::test]
async fn malformed_transcript_done_is_reported_with_model_data_redacted() {
    malformed_transcript_is_reported(true).await;
}

#[tokio::test]
async fn malformed_transcript_done_is_reported_with_model_data_logged() {
    malformed_transcript_is_reported(false).await;
}

/// `test_handle_invalid_event_schema_redacts_event_from_logs`.
#[tokio::test]
async fn invalid_event_log_omits_all_event_data_when_redacted() {
    let _turn = SERIAL.lock().await;
    let sink = sink();
    sink.0.lock().unwrap().clear();
    let _policy = PolicyGuard::set(true);

    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "SECRET_EVENT_TYPE",
        "event_id": "SECRET_EVENT_ID",
        "delta": "SECRET_EVENT_PAYLOAD",
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(kinds(&emitted), ["raw_server_event", "error"]);

    let lines = sink.0.lock().unwrap().clone();
    let failures: Vec<_> = lines
        .iter()
        .filter(|(_, line)| line.starts_with("Failed to validate server event"))
        .collect();
    assert_eq!(failures.len(), 1, "{lines:?}");
    assert_eq!(failures[0].1, "Failed to validate server event");
    for secret in [
        "SECRET_EVENT_TYPE",
        "SECRET_EVENT_ID",
        "SECRET_EVENT_PAYLOAD",
    ] {
        assert!(
            lines.iter().all(|(_, line)| !line.contains(secret)),
            "{lines:?}"
        );
    }
    model.close().await.unwrap();
    server.stop();
}

/// `test_handle_invalid_event_schema_preserves_diagnostics_when_enabled`.
#[tokio::test]
async fn invalid_event_log_keeps_diagnostics_when_model_data_is_logged() {
    let _turn = SERIAL.lock().await;
    let sink = sink();
    sink.0.lock().unwrap().clear();
    let _policy = PolicyGuard::set(false);

    let (server, model, mut events) = ready().await;
    server.push(json!({
        "type": "diagnostic.event",
        "event_id": "diagnostic_event_id",
        "delta": "diagnostic payload",
    }));
    let emitted = settle(&server, &mut events).await;
    assert_eq!(kinds(&emitted), ["raw_server_event", "error"]);

    let lines = sink.0.lock().unwrap().clone();
    let failures: Vec<_> = lines
        .iter()
        .filter(|(_, line)| line.starts_with("Failed to validate server event"))
        .collect();
    assert_eq!(failures.len(), 1, "{lines:?}");
    for kept in [
        "diagnostic.event",
        "diagnostic_event_id",
        "diagnostic payload",
    ] {
        assert!(failures[0].1.contains(kept), "{lines:?}");
    }
    model.close().await.unwrap();
    server.stop();
}
