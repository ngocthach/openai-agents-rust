//! Port of `test_interrupt_aggregates_failures_in_operation_order` in
//! `tests/realtime/test_openai_realtime.py`: when the truncate and the cancel of one interrupt
//! both fail, both failures are reported, in operation order, instead of the first only.
#![cfg(feature = "realtime")]

use openai_agents_sdk::realtime::{
    RealtimeConnectionStatus, RealtimeInterrupt, RealtimeModel, RealtimeModelEvent,
    RealtimeModelSendEvent, RealtimePlaybackTracker,
};
use serde_json::json;

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

#[tokio::test]
async fn interrupt_reports_every_failed_step_in_operation_order() {
    let mut server = MockRealtimeServer::start().await;
    let tracker = RealtimePlaybackTracker::shared();
    let (model, mut events) = connect(&server, None, Some(tracker.clone())).await;
    let _ = server.next_frame("session.update").await;
    server.push(json!({"type": "response.created", "event_id": "e1",
        "response": {"id": "response_1", "object": "realtime.response",
                     "status": "in_progress", "output": []}}));
    tracker.on_play_ms("audio_item", 0, 50.0);
    // Let the response register, then drop the connection so every send fails.
    wait_for_event(&mut events, |e| {
        matches!(e, RealtimeModelEvent::RawServerEvent { data } if data["type"] == "response.created")
            .then_some(())
    })
    .await;
    server.push_close(1011);
    wait_for_event(&mut events, |e| match e {
        RealtimeModelEvent::ConnectionStatus {
            status: RealtimeConnectionStatus::Disconnected,
        } => Some(()),
        _ => None,
    })
    .await;

    let error = model
        .send_event(RealtimeModelSendEvent::Interrupt(RealtimeInterrupt {
            force_response_cancel: true,
            ..Default::default()
        }))
        .await
        .expect_err("both steps fail on a dropped connection");

    let text = error.to_string();
    assert!(
        text.starts_with("Multiple Realtime interrupt operations failed: truncate_audio="),
        "{text}"
    );
    assert!(text.contains("; cancel_response="), "{text}");
    server.stop();
}
