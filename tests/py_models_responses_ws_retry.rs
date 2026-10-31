//! Port of the `get_retry_advice` cluster for `OpenAIResponsesWSModel` in
//! `tests/models/test_openai_responses.py`.
//!
//! Python marks error objects with private attributes (`_openai_agents_ws_replay_safety`,
//! `_openai_agents_ws_response_started`, ...). Rust errors carry no attributes, so the same facts
//! travel as message markers; these tests build errors with those exact markers.
#![cfg(feature = "websocket")]

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::models::openai_responses_ws::{
    OpenAIResponsesWSModel, WS_AMBIGUOUS_REPLAY_MESSAGE, WS_CLOSE_INVALIDATED_MESSAGE,
    WS_PRE_EVENT_DISCONNECT_MESSAGE,
};
use openai_agents_sdk::models::{Model, OpenAIClient};
use openai_agents_sdk::retry::{ModelRetryAdvice, ModelRetryAdviceRequest, ReplaySafety};
use serde_json::{Value, json};

// ------------------------------------------------------------------ retry advice

const REQUEST_SENT_SUFFIX: &str = " (the request frame was already sent)";
const RESPONSE_STARTED_SUFFIX: &str = " (the response had already started)";

fn advice(error: &AgentsError, stream: bool, previous: Option<&str>) -> Option<ModelRetryAdvice> {
    OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test")).get_retry_advice(
        &ModelRetryAdviceRequest {
            error,
            attempt: 1,
            stream,
            previous_response_id: previous,
            conversation_id: None,
        },
    )
}

fn ws_error(payload: Value) -> AgentsError {
    AgentsError::other(format!("Responses websocket error: {payload}"))
}

#[test]
fn ambiguous_replay_is_unsafe_when_stateful_and_retryable_when_stateless() {
    let error = AgentsError::other(WS_AMBIGUOUS_REPLAY_MESSAGE);
    let a = advice(&error, true, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(false));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    let a = advice(&error, true, None).unwrap();
    assert_eq!(a.suggested, Some(true));
    assert_eq!(a.replay_safety, None);
}

#[test]
fn wrapped_pre_send_disconnect_stays_safe() {
    let error = AgentsError::other(WS_PRE_EVENT_DISCONNECT_MESSAGE);
    let a = advice(&error, true, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(true));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Safe));
}

#[test]
fn wrapped_post_send_disconnect_depends_on_statefulness() {
    let error = AgentsError::other(format!(
        "{WS_PRE_EVENT_DISCONNECT_MESSAGE}{REQUEST_SENT_SUFFIX}"
    ));
    for stream in [true, false] {
        let a = advice(&error, stream, None).unwrap();
        assert_eq!(a.suggested, Some(true), "stream={stream}");
        assert_eq!(a.replay_safety, None, "stream={stream}");
    }
    let a = advice(&error, true, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(false));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    assert!(!a.response_started);
}

#[test]
fn partial_failures_report_response_started_and_are_unsafe() {
    let timeout = AgentsError::other(format!(
        "Responses websocket receive timed out after 5.0 seconds.{RESPONSE_STARTED_SUFFIX}"
    ));
    let a = advice(&timeout, false, None).unwrap();
    assert_eq!(a.suggested, Some(false));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    assert!(a.response_started);

    let disconnect = AgentsError::other(format!(
        "Responses websocket transport error: no close frame received or sent{RESPONSE_STARTED_SUFFIX}"
    ));
    let a = advice(&disconnect, false, None).unwrap();
    assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    assert!(a.response_started);
}

#[test]
fn timeout_phases_drive_replay_safety() {
    let connect = AgentsError::other("Responses websocket connect timed out after 5.0 seconds.");
    let a = advice(&connect, true, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(true));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Safe));

    let lock =
        AgentsError::other("Responses websocket request lock wait timed out after 5.0 seconds.");
    let a = advice(&lock, false, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(true));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Safe));

    let receive = AgentsError::other("Responses websocket receive timed out after 5.0 seconds.");
    let a = advice(&receive, true, Some("resp_prev")).unwrap();
    assert_eq!(a.suggested, Some(false));
    assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    let a = advice(&receive, true, None).unwrap();
    assert_eq!(a.suggested, Some(true));
    assert_eq!(a.replay_safety, None);
}

#[test]
fn rejected_handshake_gets_no_advice() {
    let error =
        AgentsError::other("Responses websocket connect failed: malformed HTTP status line");
    assert!(advice(&error, true, Some("resp_prev")).is_none());
}

#[test]
fn close_invalidation_is_an_abort() {
    let error = AgentsError::other(WS_CLOSE_INVALIDATED_MESSAGE);
    let a = advice(&error, false, None).unwrap();
    assert_eq!(a.suggested, Some(false));
    assert!(a.normalized.expect("normalized").is_abort);
}

#[test]
fn pre_response_transient_error_frames_are_retryable() {
    let overloaded = ws_error(json!({"type": "error", "error": {
        "type": "service_unavailable_error", "code": "server_is_overloaded",
        "message": "Our servers are currently overloaded. Please try again later."}}));
    let server_error = ws_error(json!({"type": "error", "error": {
        "type": "server_error", "code": null, "message": "Sorry, something went wrong."}}));
    for (error, stream) in [(&overloaded, false), (&server_error, true)] {
        for previous in [None, Some("resp_prev")] {
            let a = advice(error, stream, previous).unwrap();
            assert_eq!(a.suggested, Some(true), "{previous:?}");
            assert_eq!(a.replay_safety, None, "{previous:?}");
        }
    }
}

#[test]
fn partial_transient_error_frames_stay_unsafe() {
    for (error_type, code) in [
        ("service_unavailable_error", json!("server_is_overloaded")),
        ("server_error", json!(null)),
    ] {
        let error = AgentsError::other(format!(
            "Responses websocket error: {}{RESPONSE_STARTED_SUFFIX}",
            json!({"type": "error", "error": {
                "type": error_type, "code": code, "message": "Transient provider error."}})
        ));
        let a = advice(&error, false, None).unwrap();
        assert_eq!(a.suggested, Some(false));
        assert_eq!(a.replay_safety, Some(ReplaySafety::Unsafe));
    }
}

#[test]
fn non_transient_error_frames_get_no_advice() {
    let invalid_code = ws_error(json!({"type": "error", "error": {
        "type": "server_error", "code": "invalid_request_error", "message": "Invalid request."}}));
    assert!(advice(&invalid_code, true, None).is_none());
    let invalid_request = ws_error(json!({"type": "error", "error": {
        "type": "invalid_request_error", "code": "invalid_request", "message": "Invalid request."}}));
    assert!(advice(&invalid_request, false, None).is_none());
}
