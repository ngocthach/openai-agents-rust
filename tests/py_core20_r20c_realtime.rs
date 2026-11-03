//! Realtime cases the earlier audits left as Python-only:
//!
//! - `tests/realtime/test_openai_realtime_conversions.py::test_tools_to_session_tools_raises_on_non_function_tool`:
//!   a realtime agent that declares a tool other than a function tool is refused with Python's
//!   `Tool {name} is unsupported. Must be a function tool.` user error.
//! - `tests/realtime/test_session_payload_and_formats.py::test_normalize_session_payload_variants`:
//!   only a realtime session payload sets the playback format; a transcription session or one of
//!   an unknown type is ignored, and a realtime one still applies.

#![cfg(feature = "realtime")]

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::RunContext;
use openai_agents_sdk::realtime::RealtimeAgent;
use openai_agents_sdk::tool::{Tool, WebSearchTool};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

/// Python: `test_tools_to_session_tools_raises_on_non_function_tool`.
#[tokio::test]
async fn a_hosted_tool_is_refused_by_a_realtime_agent() {
    let hosted = Tool::<()>::from(WebSearchTool::default());
    let name = hosted.name().to_string();
    let agent = RealtimeAgent::<()>::new("agent").with_tool(hosted);

    let error = agent
        .get_all_tools(&RunContext::new(()))
        .await
        .expect_err("a hosted tool cannot run in a realtime session");

    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(
        error.to_string().contains(&format!(
            "Tool {name} is unsupported. Must be a function tool."
        )),
        "{error}"
    );
}

/// The playback tracker's format after the server sends `event`.
async fn tracker_format_after(event: Value) -> Option<String> {
    let mut server = MockRealtimeServer::start().await;
    let tracker = openai_agents_sdk::realtime::RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    let _ = server.next_frame("session.update").await;
    server.push(event);
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

fn session_updated(session_type: &str) -> Value {
    json!({
        "type": "session.updated",
        "event_id": "event_123",
        "session": {
            "type": session_type,
            "audio": {"output": {"format": {"type": "audio/pcmu"}}},
        },
    })
}

/// Python: `test_normalize_session_payload_variants`.
#[tokio::test]
async fn only_a_realtime_session_payload_sets_the_playback_format() {
    assert_eq!(
        tracker_format_after(session_updated("realtime"))
            .await
            .as_deref(),
        Some("g711_ulaw")
    );
    assert_eq!(
        tracker_format_after(session_updated("transcription")).await,
        None
    );
    assert_eq!(tracker_format_after(session_updated("bogus")).await, None);
}
