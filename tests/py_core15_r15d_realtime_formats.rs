//! Ports of the audio-format cases of `tests/realtime/test_session_payload_and_formats.py`:
//! the format a `session.created` event names reaches the playback tracker under its legacy
//! name (`pcm16`, `g711_ulaw`, `g711_alaw`), string and mapping spellings pass through, and a
//! session without an output format leaves the tracker alone.

#![cfg(feature = "realtime")]

use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

async fn tracker_format_after(created: Value) -> Option<String> {
    let mut server = MockRealtimeServer::start().await;
    let tracker = openai_agents_sdk::realtime::RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    let _ = server.next_frame("session.update").await;
    server.push(created);
    server.push(json!({"type": "input_audio_buffer.cleared", "event_id": "barrier"}));
    wait_for_event(&mut events, |event| match event {
        openai_agents_sdk::realtime::RealtimeModelEvent::Other { data }
            if data["type"] == "input_audio_buffer.cleared" =>
        {
            Some(())
        }
        _ => None,
    })
    .await;
    let format = tracker.audio_format();
    use openai_agents_sdk::realtime::RealtimeModel;
    model.close().await.unwrap();
    server.stop();
    format
}

/// Python: `test_extract_audio_format_from_session_objects`.
#[tokio::test]
async fn known_ga_formats_are_normalized_to_the_legacy_names() {
    for (ga, legacy) in [
        ("audio/pcm", "pcm16"),
        ("audio/pcmu", "g711_ulaw"),
        ("audio/pcma", "g711_alaw"),
    ] {
        let format = tracker_format_after(session_created(Some(json!({"type": ga})))).await;
        assert_eq!(format.as_deref(), Some(legacy), "{ga}");
    }
}

/// Python: `test_extract_audio_format_from_session_objects` (missing format) and
/// `test_normalize_audio_format_fallbacks`.
#[tokio::test]
async fn other_spellings_pass_through_and_a_missing_format_is_ignored() {
    assert_eq!(tracker_format_after(session_created(None)).await, None);
    assert_eq!(
        tracker_format_after(session_created(Some(json!("pcm24"))))
            .await
            .as_deref(),
        Some("pcm24")
    );
    assert_eq!(
        tracker_format_after(session_created(Some(json!({"type": "g711_ulaw"}))))
            .await
            .as_deref(),
        Some("g711_ulaw")
    );
}
