//! Ports of `test_try_convert_raw_message_invalid_type`, `_malformed_data` and `_missing_type`
//! from `tests/realtime/test_conversion_helpers.py`: a raw message that is not a valid client
//! event is dropped (and logged) instead of reaching the wire, and the session carries on.

#![cfg(feature = "realtime")]

use openai_agents_sdk::realtime::{RealtimeModel, RealtimeModelSendEvent};
use serde_json::json;

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

#[tokio::test]
async fn an_invalid_raw_message_is_dropped_before_the_wire() {
    let mut server = MockRealtimeServer::start().await;
    let (model, _events) = connect(&server, None, None).await;
    let _ = server.next_frame("session.update").await;

    for message in [
        json!({"type": "invalid.message.type", "other_data": {}}),
        json!({"type": "session.update", "session": "invalid_session_data"}),
        json!({"type": "missing.type.test", "some": "data"}),
        json!({"no_type_at_all": true}),
    ] {
        model
            .send_event(RealtimeModelSendEvent::RawMessage { message })
            .await
            .expect("an invalid raw message is dropped without failing the caller");
    }
    send_marker(&model).await;

    assert!(
        server.frames_before_marker().await.is_empty(),
        "no invalid message reached the server"
    );
    model.close().await.unwrap();
    server.stop();
}
