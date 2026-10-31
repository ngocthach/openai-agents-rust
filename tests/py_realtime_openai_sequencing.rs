//! Ports of the response-sequencing tests of `tests/realtime/test_openai_realtime.py`
//! (`test_send_user_input_*` .. `test_raw_response_create_*`), the client-event conversions of
//! `tests/realtime/test_conversion_helpers.py`, and the tool-list conversions of
//! `tests/realtime/test_openai_realtime_conversions.py`.
//!
//! Python replaces `_send_raw_message` and reads the sequencer's private state. Here each test
//! talks to a local websocket and reads the frames the transport really sends, so "the response
//! control is free" becomes "the next input starts a response immediately". No network, no real
//! API key.

#![cfg(feature = "realtime")]

use std::sync::Arc;

use openai_agents_sdk::realtime::{
    OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, RealtimeAgent, RealtimeHandoff,
    RealtimeModel, RealtimeModelEvent, RealtimeModelEvents, RealtimeModelSendEvent,
    RealtimeModelToolCallEvent, RealtimeModelTracing, RealtimeModelTracingConfig,
    RealtimeUserInput, RealtimeUserInputContent,
};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

const SENTINEL: &str = "py.test.sentinel";

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

fn response(response_id: &str) -> Value {
    json!({"id": response_id, "object": "realtime.response", "status": "in_progress", "output": []})
}

fn created(response_id: &str) -> Value {
    json!({"type": "response.created", "event_id": "e_created", "response": response(response_id)})
}

fn done(response_id: &str) -> Value {
    json!({"type": "response.done", "event_id": "e_done", "response": response(response_id)})
}

async fn user_input(model: &OpenAIRealtimeWebSocketModel, text: &str) {
    model
        .send_event(RealtimeModelSendEvent::UserInput {
            user_input: text.into(),
        })
        .await
        .unwrap();
}

async fn tool_output(model: &OpenAIRealtimeWebSocketModel, call_id: &str, start_response: bool) {
    model
        .send_event(RealtimeModelSendEvent::ToolOutput {
            tool_call: RealtimeModelToolCallEvent::new("t", call_id, "{}"),
            output: format!("ok-{call_id}"),
            start_response,
        })
        .await
        .unwrap();
}

/// The types of the item/response frames sent since the last call, in order.
async fn sent_types(
    server: &mut MockRealtimeServer,
    model: &OpenAIRealtimeWebSocketModel,
) -> Vec<String> {
    send_marker(model).await;
    server
        .frames_before_marker()
        .await
        .iter()
        .map(|frame| frame["type"].as_str().unwrap().to_string())
        .collect()
}

fn count(types: &[String], wanted: &str) -> usize {
    types.iter().filter(|kind| kind.as_str() == wanted).count()
}

// ---------------------------------------------------------------------------------------------
// Response sequencing
// ---------------------------------------------------------------------------------------------

/// `test_custom_voice_response_events_update_response_sequencer`: a response carrying a
/// dict-shaped custom voice still starts and ends the sequencer, and the retained raw payload is
/// left untouched.
#[tokio::test]
async fn custom_voice_response_events_update_response_sequencer() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "hi").await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create", "response.create"]
    );

    let with_voice = json!({
        "type": "response.created",
        "response": {"audio": {"output": {"voice": {"id": "voice_test"}}}},
    });
    server.push(with_voice.clone());
    server.push(json!({
        "type": "response.done",
        "response": {"audio": {"output": {"voice": {"id": "voice_test"}}}},
    }));
    let emitted = settle(&server, &mut events).await;
    match &emitted[0] {
        RealtimeModelEvent::RawServerEvent { data } => {
            assert_eq!(data, &with_voice);
            assert_eq!(
                data["response"]["audio"]["output"]["voice"],
                json!({"id": "voice_test"})
            );
        }
        other => panic!("expected the raw event first, got {other:?}"),
    }

    // The response ended, so a tool output starts the next one immediately.
    tool_output(&model, "call_1", true).await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create", "response.create"]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_send_user_input_defers_response_create_without_blocking_caller` (and the
/// `..._from_websocket_listener_...` variant, which Rust has no separate path for: events are
/// consumed off the reader task, so a handler that sends input never blocks the socket loop).
#[tokio::test]
async fn send_user_input_defers_response_create_without_blocking_caller() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;

    // The caller returns immediately, with only the item sent.
    tokio::time::timeout(TIMEOUT, user_input(&model, "hi"))
        .await
        .expect("the caller is not blocked behind the active response");
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );

    server.push(done("r1"));
    let create = server.next_frame("response.create").await;
    assert_eq!(create["type"], "response.create");
    model.close().await.unwrap();
    server.stop();
}

/// `test_conditional_user_input_skips_when_response_create_is_already_pending`
#[tokio::test]
async fn conditional_user_input_skips_when_response_create_is_already_pending() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    user_input(&model, "newer input").await;

    let sent = model
        .send_event_if(
            RealtimeModelSendEvent::UserInput {
                user_input: "guardrail feedback".into(),
            },
            Box::new(|| true),
        )
        .await
        .unwrap();
    assert!(
        !sent,
        "a response is already queued, so the feedback is dropped"
    );
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_conditional_user_input_reserves_response_before_later_normal_input`
#[tokio::test]
async fn conditional_user_input_reserves_response_before_later_normal_input() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;

    let sent = model
        .send_event_if(
            RealtimeModelSendEvent::UserInput {
                user_input: "guardrail feedback".into(),
            },
            Box::new(|| true),
        )
        .await
        .unwrap();
    assert!(sent);
    user_input(&model, "newer input").await;
    send_marker(&model).await;
    let frames = server.frames_before_marker().await;
    let texts: Vec<&str> = frames
        .iter()
        .filter(|frame| frame["type"] == "conversation.item.create")
        .map(|frame| frame["item"]["content"][0]["text"].as_str().unwrap())
        .collect();
    assert_eq!(texts, ["guardrail feedback", "newer input"]);
    assert!(
        !frames
            .iter()
            .any(|frame| frame["type"] == "response.create")
    );

    // One response covers both inputs once the active one ends.
    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    assert_eq!(
        count(&sent_types(&mut server, &model).await, "response.create"),
        0
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_stacked_user_inputs_coalesce_to_one_response_create_per_turn`
#[tokio::test]
async fn stacked_user_inputs_coalesce_to_one_response_create_per_turn() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    user_input(&model, "first").await;
    user_input(&model, "second").await;

    let before = sent_types(&mut server, &model).await;
    assert_eq!(count(&before, "conversation.item.create"), 2);
    assert_eq!(count(&before, "response.create"), 0);

    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    // The one create covers both inputs.
    assert_eq!(
        count(&sent_types(&mut server, &model).await, "response.create"),
        0
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_user_input_after_sent_response_create_starts_follow_up_turn`
#[tokio::test]
async fn user_input_after_sent_response_create_starts_follow_up_turn() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "first").await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create", "response.create"]
    );

    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    user_input(&model, "second").await;
    let types = sent_types(&mut server, &model).await;
    assert_eq!(types, ["conversation.item.create"]);

    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    model.close().await.unwrap();
    server.stop();
}

/// `test_user_inputs_queued_during_response_create_send_start_a_follow_up_turn`: an input that
/// arrives after the create was sent but before the server acknowledged it needs a later turn.
#[tokio::test]
async fn user_inputs_queued_during_response_create_send_start_a_follow_up_turn() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "first").await;
    user_input(&model, "second").await;

    let before = sent_types(&mut server, &model).await;
    assert_eq!(count(&before, "conversation.item.create"), 2);
    assert_eq!(count(&before, "response.create"), 1);

    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    assert_eq!(
        count(&sent_types(&mut server, &model).await, "response.create"),
        0
    );

    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    model.close().await.unwrap();
    server.stop();
}

/// The id the transport put on its in-flight `response.create`.
async fn in_flight_create_id(server: &mut MockRealtimeServer) -> String {
    server.next_frame("response.create").await["event_id"]
        .as_str()
        .expect("the transport tags its creates")
        .to_string()
}

fn error_event(event_id: &str, code: &str, message: &str, nested_event_id: Option<&str>) -> Value {
    let mut error = json!({"type": "invalid_request_error", "code": code, "message": message});
    if let Some(id) = nested_event_id {
        error["event_id"] = Value::String(id.to_string());
    }
    json!({"type": "error", "event_id": event_id, "error": error})
}

/// `test_unrelated_error_does_not_release_in_flight_response_create`
#[tokio::test]
async fn unrelated_error_does_not_release_in_flight_response_create() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "first").await;
    let pending = in_flight_create_id(&mut server).await;

    server.push(error_event(
        "event_err_1",
        "bad_item",
        "bad item",
        Some("other_event_id"),
    ));
    let _ = settle(&server, &mut events).await;
    // Still in flight: a second input only adds its item.
    user_input(&model, "second").await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );

    server.push(error_event(
        "event_err_2",
        "bad_response_create",
        "bad response.create",
        Some(&pending),
    ));
    // The matching error releases the slot, and the deferred input gets its response.
    let create = server.next_frame("response.create").await;
    assert_ne!(create["event_id"], json!(pending));
    model.close().await.unwrap();
    server.stop();
}

/// `test_missing_unrelated_error_event_id_does_not_release_in_flight_response_create`
#[tokio::test]
async fn missing_unrelated_error_event_id_does_not_release_in_flight_response_create() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "first").await;
    let pending = in_flight_create_id(&mut server).await;

    server.push(error_event(
        "event_err_missing_nested",
        "bad_item",
        "bad item",
        None,
    ));
    let _ = settle(&server, &mut events).await;
    user_input(&model, "second").await;
    let types = sent_types(&mut server, &model).await;
    assert_eq!(
        types
            .iter()
            .filter(|kind| *kind == "response.create")
            .count(),
        0,
        "{types:?}"
    );

    // The matching error releases it: the next input starts a response right away.
    server.push(error_event(
        "event_err_matching",
        "bad_response_create",
        "bad response.create",
        Some(&pending),
    ));
    let _ = server.next_frame("response.create").await;
    model.close().await.unwrap();
    server.stop();
}

/// `test_missing_error_event_id_releases_in_flight_response_create`: an error without a nested id
/// that reads as a failed `response.create` releases the slot too.
#[tokio::test]
async fn missing_error_event_id_releases_in_flight_response_create() {
    let (mut server, model, mut events) = ready().await;
    user_input(&model, "first").await;
    let _ = in_flight_create_id(&mut server).await;

    server.push(error_event(
        "event_err_missing_nested",
        "bad_response_create",
        "bad response.create",
        None,
    ));
    let _ = settle(&server, &mut events).await;

    user_input(&model, "second").await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create", "response.create"]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_tool_output_start_response_defers_response_create_without_blocking_caller` (and the
/// `..._from_websocket_listener_...` variant, see the user-input test above)
#[tokio::test]
async fn tool_output_start_response_defers_response_create_without_blocking_caller() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;

    tokio::time::timeout(TIMEOUT, tool_output(&model, "c", true))
        .await
        .expect("the caller is not blocked behind the active response");
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );

    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    model.close().await.unwrap();
    server.stop();
}

/// `test_stacked_tool_outputs_coalesce_to_one_response_create_per_turn`
#[tokio::test]
async fn stacked_tool_outputs_coalesce_to_one_response_create_per_turn() {
    let (mut server, model, mut events) = ready().await;
    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    tool_output(&model, "c1", true).await;
    tool_output(&model, "c2", true).await;

    let before = sent_types(&mut server, &model).await;
    assert_eq!(count(&before, "conversation.item.create"), 2);
    assert_eq!(count(&before, "response.create"), 0);

    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    assert_eq!(
        count(&sent_types(&mut server, &model).await, "response.create"),
        0
    );
    model.close().await.unwrap();
    server.stop();
}

async fn raw_response_create(model: &OpenAIRealtimeWebSocketModel) {
    model
        .send_event(RealtimeModelSendEvent::RawMessage {
            message: json!({"type": "response.create", "response": {"instructions": "Say hello."}}),
        })
        .await
        .unwrap();
}

/// `test_raw_response_create_is_sequenced_with_follow_up_user_input`
#[tokio::test]
async fn raw_response_create_is_sequenced_with_follow_up_user_input() {
    let (mut server, model, mut events) = ready().await;
    raw_response_create(&model).await;
    let create = server.next_frame("response.create").await;
    // The caller's own body is what goes out.
    assert_eq!(create["response"]["instructions"], "Say hello.");

    // The raw create is still in flight, so the input adds its item and waits.
    user_input(&model, "hi").await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );

    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    assert_eq!(
        count(&sent_types(&mut server, &model).await, "response.create"),
        0
    );
    server.push(done("r1"));
    let second = server.next_frame("response.create").await;
    assert!(
        second.get("response").is_none(),
        "the follow-up is a plain create: {second}"
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_raw_response_create_is_sequenced_with_follow_up_tool_output`
#[tokio::test]
async fn raw_response_create_is_sequenced_with_follow_up_tool_output() {
    let (mut server, model, mut events) = ready().await;
    raw_response_create(&model).await;
    let _ = server.next_frame("response.create").await;

    tool_output(&model, "c", true).await;
    assert_eq!(
        sent_types(&mut server, &model).await,
        ["conversation.item.create"]
    );

    server.push(created("r1"));
    let _ = settle(&server, &mut events).await;
    server.push(done("r1"));
    let _ = server.next_frame("response.create").await;
    model.close().await.unwrap();
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_conversion_helpers.py
// ---------------------------------------------------------------------------------------------

/// `test_try_convert_raw_message_valid_session_update`: a raw `session.update` reaches the wire
/// unchanged.
#[tokio::test]
async fn raw_session_update_is_forwarded() {
    let (mut server, model, _events) = ready().await;
    let message = json!({
        "type": "session.update",
        "session": {"model": "gpt-realtime-2.1", "type": "realtime", "voice": "ash"},
    });
    model
        .send_event(RealtimeModelSendEvent::RawMessage {
            message: message.clone(),
        })
        .await
        .unwrap();
    assert_eq!(server.next_frame("session.update").await, message);
    model.close().await.unwrap();
    server.stop();
}

/// `test_try_convert_raw_message_valid_response_create`
#[tokio::test]
async fn raw_response_create_is_forwarded() {
    let (mut server, model, _events) = ready().await;
    model
        .send_event(RealtimeModelSendEvent::RawMessage {
            message: json!({"type": "response.create"}),
        })
        .await
        .unwrap();
    assert_eq!(
        server.next_frame("response.create").await["type"],
        "response.create"
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_convert_tracing_config_none`: no tracing means nothing is sent.
#[test]
fn convert_tracing_config_none() {
    assert_eq!(RealtimeModelTracing::Disabled.to_session_value(), None);
}

/// `test_convert_tracing_config_auto` (and the variants test)
#[test]
fn convert_tracing_config_auto() {
    assert_eq!(
        RealtimeModelTracing::Auto.to_session_value(),
        Some(json!("auto"))
    );
}

/// `test_convert_tracing_config_dict_full`
#[test]
fn convert_tracing_config_dict_full() {
    let config = RealtimeModelTracing::Config(RealtimeModelTracingConfig {
        group_id: Some("test-group".to_string()),
        metadata: Some(json!({"env": "test"})),
        workflow_name: Some("test-workflow".to_string()),
    });
    assert_eq!(
        config.to_session_value(),
        Some(json!({
            "group_id": "test-group",
            "metadata": {"env": "test"},
            "workflow_name": "test-workflow",
        }))
    );
}

/// `test_convert_tracing_config_dict_partial`
#[test]
fn convert_tracing_config_dict_partial() {
    let config = RealtimeModelTracing::Config(RealtimeModelTracingConfig {
        group_id: Some("test-group".to_string()),
        ..Default::default()
    });
    assert_eq!(
        config.to_session_value(),
        Some(json!({"group_id": "test-group"}))
    );
}

/// `test_convert_tracing_config_empty_dict`
#[test]
fn convert_tracing_config_empty_dict() {
    let config = RealtimeModelTracing::Config(RealtimeModelTracingConfig::default());
    assert_eq!(config.to_session_value(), Some(json!({})));
}

/// `test_convert_tracing_config_variants`: all three shapes reach the wire after `session.created`.
#[tokio::test]
async fn convert_tracing_config_variants_reach_the_wire() {
    use openai_agents_sdk::realtime::RealtimeSessionModelSettings;
    let mut server = MockRealtimeServer::start().await;
    let (model, _events) = connect(
        &server,
        Some(
            RealtimeSessionModelSettings::new().with_tracing(RealtimeModelTracing::Config(
                RealtimeModelTracingConfig {
                    group_id: Some("g".to_string()),
                    metadata: Some(json!({"k": "v"})),
                    workflow_name: Some("wf".to_string()),
                },
            )),
        ),
        None,
    )
    .await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created(None));
    let update = server.next_frame("session.update").await;
    assert_eq!(update["session"]["tracing"]["group_id"], "g");
    assert_eq!(update["session"]["tracing"]["workflow_name"], "wf");
    assert_eq!(update["session"]["tracing"]["metadata"], json!({"k": "v"}));
    model.close().await.unwrap();
    server.stop();
}

/// The item a `UserInput` produces on the wire.
async fn user_item_for(input: RealtimeUserInput) -> Value {
    let (mut server, model, _events) = ready().await;
    model
        .send_event(RealtimeModelSendEvent::UserInput { user_input: input })
        .await
        .unwrap();
    let frame = server.next_frame("conversation.item.create").await;
    model.close().await.unwrap();
    server.stop();
    frame
}

/// `test_convert_user_input_to_conversation_item_string` / `..._to_item_create`
#[tokio::test]
async fn convert_user_input_to_conversation_item_string() {
    let frame = user_item_for("Hello, world!".into()).await;
    assert_eq!(frame["type"], "conversation.item.create");
    assert_eq!(frame["item"]["type"], "message");
    assert_eq!(frame["item"]["role"], "user");
    assert_eq!(
        frame["item"]["content"],
        json!([{"type": "input_text", "text": "Hello, world!"}])
    );
}

/// `test_convert_user_input_to_item_create`
#[tokio::test]
async fn convert_user_input_to_item_create() {
    let frame = user_item_for("Test message".into()).await;
    assert_eq!(frame["type"], "conversation.item.create");
    assert_eq!(frame["item"]["type"], "message");
    assert_eq!(frame["item"]["role"], "user");
}

/// `test_convert_user_input_to_conversation_item_dict`
#[tokio::test]
async fn convert_user_input_to_conversation_item_dict() {
    let frame = user_item_for(RealtimeUserInput::Message(vec![
        RealtimeUserInputContent::InputText {
            text: "Hello".to_string(),
        },
        RealtimeUserInputContent::InputText {
            text: "World".to_string(),
        },
    ]))
    .await;
    assert_eq!(frame["item"]["role"], "user");
    assert_eq!(
        frame["item"]["content"],
        json!([{"type": "input_text", "text": "Hello"}, {"type": "input_text", "text": "World"}])
    );
}

/// `test_convert_user_input_to_conversation_item_dict_empty_content`
#[tokio::test]
async fn convert_user_input_to_conversation_item_dict_empty_content() {
    let frame = user_item_for(RealtimeUserInput::Message(vec![])).await;
    assert_eq!(frame["item"]["type"], "message");
    assert_eq!(frame["item"]["role"], "user");
    assert_eq!(frame["item"]["content"], json!([]));
}

/// `test_convert_user_input_to_conversation_item_dict_and_str` (conversions file): text and image
/// parts keep their shape.
#[tokio::test]
async fn convert_user_input_mixed_parts() {
    let frame = user_item_for(RealtimeUserInput::Message(vec![
        RealtimeUserInputContent::InputText {
            text: "hello".to_string(),
        },
        RealtimeUserInputContent::InputImage {
            image_url: "http://x/y.png".to_string(),
            detail: Some("auto".to_string()),
        },
    ]))
    .await;
    assert_eq!(frame["item"]["role"], "user");
    assert_eq!(
        frame["item"]["content"],
        json!([
            {"type": "input_text", "text": "hello"},
            {"type": "input_image", "image_url": "http://x/y.png", "detail": "auto"},
        ])
    );
}

async fn audio_frame(audio: Vec<u8>, commit: bool) -> (Value, bool) {
    let (mut server, model, _events) = ready().await;
    model
        .send_event(RealtimeModelSendEvent::Audio { audio, commit })
        .await
        .unwrap();
    let append = server.next_frame("input_audio_buffer.append").await;
    send_marker(&model).await;
    let committed = server
        .frames_before_marker()
        .await
        .iter()
        .any(|frame| frame["type"] == "input_audio_buffer.commit");
    model.close().await.unwrap();
    server.stop();
    (append, committed)
}

/// `test_convert_audio_to_input_audio_buffer_append`
#[tokio::test]
async fn convert_audio_to_input_audio_buffer_append() {
    let (frame, committed) = audio_frame(b"test audio data".to_vec(), false).await;
    assert_eq!(frame["type"], "input_audio_buffer.append");
    assert_eq!(frame["audio"], base64_encode(b"test audio data"));
    assert!(!committed);
}

/// `test_convert_audio_to_input_audio_buffer_append_empty`
#[tokio::test]
async fn convert_audio_to_input_audio_buffer_append_empty() {
    let (frame, committed) = audio_frame(Vec::new(), true).await;
    assert_eq!(frame["type"], "input_audio_buffer.append");
    assert_eq!(frame["audio"], "");
    assert!(committed, "commit follows the append");
}

/// `test_convert_audio_to_input_audio_buffer_append_large_data`
#[tokio::test]
async fn convert_audio_to_input_audio_buffer_append_large_data() {
    let data = vec![b'x'; 10_000];
    let (frame, _) = audio_frame(data.clone(), false).await;
    assert_eq!(frame["audio"], base64_encode(&data));
}

async fn tool_output_frame(call_id: &str, output: &str, start_response: bool) -> Value {
    let (mut server, model, _events) = ready().await;
    model
        .send_event(RealtimeModelSendEvent::ToolOutput {
            tool_call: RealtimeModelToolCallEvent::new("t", call_id, "{}"),
            output: output.to_string(),
            start_response,
        })
        .await
        .unwrap();
    let frame = server.next_frame("conversation.item.create").await;
    model.close().await.unwrap();
    server.stop();
    frame
}

/// `test_convert_tool_output`
#[tokio::test]
async fn convert_tool_output() {
    let frame = tool_output_frame("call_123", "Function executed successfully", false).await;
    assert_eq!(frame["type"], "conversation.item.create");
    assert_eq!(frame["item"]["type"], "function_call_output");
    assert_eq!(frame["item"]["output"], "Function executed successfully");
    assert_eq!(frame["item"]["call_id"], "call_123");
}

/// `test_convert_tool_output_empty_output`
#[tokio::test]
async fn convert_tool_output_empty_output() {
    let frame = tool_output_frame("call_456", "", true).await;
    assert_eq!(frame["type"], "conversation.item.create");
    assert_eq!(frame["item"]["type"], "function_call_output");
    assert_eq!(frame["item"]["output"], "");
    assert_eq!(frame["item"]["call_id"], "call_456");
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_openai_realtime_conversions.py: `_tools_to_session_tools`
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new(name, "a tool", |_ctx, _args: NoArgs| async move {
        Ok("ok".to_string())
    })
}

fn handoff_to(agent: &str, tool_name: Option<&str>) -> RealtimeHandoff<()> {
    let handoff = RealtimeHandoff::new(RealtimeAgent::<()>::new(agent).into_arc());
    match tool_name {
        Some(name) => handoff.with_tool_name(name),
        None => handoff,
    }
}

/// The tool names the session payload advertises for `agent`, or the build error.
async fn advertised_tools(agent: RealtimeAgent<()>) -> Result<Vec<String>, String> {
    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent.into_arc(),
        (),
        None,
        None,
        None,
    )
    .await
    .map_err(|error| error.to_string())?;
    Ok(payload["tools"]
        .as_array()
        .expect("tools is a list")
        .iter()
        .map(|tool| tool["name"].as_str().unwrap().to_string())
        .collect())
}

/// `test_tools_to_session_tools_includes_handoffs`
#[tokio::test]
async fn tools_to_session_tools_includes_handoffs() {
    let agent = RealtimeAgent::<()>::new("root").with_handoff(handoff_to("a", None));
    let names = advertised_tools(agent).await.unwrap();
    assert!(names[0].starts_with("transfer_to_"), "{names:?}");
}

/// `test_tools_to_session_tools_rejects_duplicate_function_tool_names`
#[tokio::test]
async fn tools_to_session_tools_rejects_duplicate_function_tool_names() {
    let agent = RealtimeAgent::<()>::new("root")
        .with_tool(tool("lookup_account"))
        .with_tool(tool("lookup_account"));
    let error = advertised_tools(agent).await.unwrap_err();
    assert!(
        error.contains("Duplicate realtime tool names") && error.contains("lookup_account"),
        "{error}"
    );
}

/// `test_tools_to_session_tools_rejects_function_handoff_name_conflict`
#[tokio::test]
async fn tools_to_session_tools_rejects_function_handoff_name_conflict() {
    let agent = RealtimeAgent::<()>::new("root")
        .with_tool(tool("transfer_to_billing"))
        .with_handoff(handoff_to("billing", Some("transfer_to_billing")));
    let error = advertised_tools(agent).await.unwrap_err();
    assert!(
        error.contains("Duplicate realtime tool names") && error.contains("transfer_to_billing"),
        "{error}"
    );
}

/// `test_tools_to_session_tools_ignores_disabled_function_tool_name_conflict`
#[tokio::test]
async fn tools_to_session_tools_ignores_disabled_function_tool_name_conflict() {
    let agent = RealtimeAgent::<()>::new("root")
        .with_tool(tool("transfer_to_billing").with_is_enabled(false))
        .with_handoff(handoff_to("billing", Some("transfer_to_billing")));
    assert_eq!(
        advertised_tools(agent).await.unwrap(),
        ["transfer_to_billing"]
    );
}

/// `test_tools_to_session_tools_omits_disabled_function_tool`
#[tokio::test]
async fn tools_to_session_tools_omits_disabled_function_tool() {
    let agent =
        RealtimeAgent::<()>::new("root").with_tool(tool("hidden_tool").with_is_enabled(false));
    assert!(advertised_tools(agent).await.unwrap().is_empty());
}

/// `test_tools_to_session_tools_ignores_disabled_handoff_name_conflict`
#[tokio::test]
async fn tools_to_session_tools_ignores_disabled_handoff_name_conflict() {
    let agent = RealtimeAgent::<()>::new("root")
        .with_tool(tool("transfer_to_billing"))
        .with_handoff(handoff_to("billing", Some("transfer_to_billing")).with_is_enabled(false));
    assert_eq!(
        advertised_tools(agent).await.unwrap(),
        ["transfer_to_billing"]
    );
}

/// `test_tools_to_session_tools_rejects_duplicate_handoff_names`
#[tokio::test]
async fn tools_to_session_tools_rejects_duplicate_handoff_names() {
    let agent = RealtimeAgent::<()>::new("root")
        .with_handoff(handoff_to("billing", Some("transfer_to_support")))
        .with_handoff(handoff_to("technical", Some("transfer_to_support")));
    let error = advertised_tools(agent).await.unwrap_err();
    assert!(
        error.contains("Duplicate realtime tool names") && error.contains("transfer_to_support"),
        "{error}"
    );
}

/// `test_tools_to_session_tools_rejects_namespaced_function_tools`
///
#[tokio::test]
async fn tools_to_session_tools_rejects_namespaced_function_tools() {
    let namespaced = tool_namespace("crm", "CRM tools", vec![tool("lookup_account")])
        .unwrap()
        .remove(0);
    let agent = RealtimeAgent::<()>::new("root").with_tool(namespaced);
    let error = advertised_tools(agent).await.unwrap_err();
    assert!(error.contains("tool_namespace()"), "{error}");
}

/// `test_tools_to_session_tools_rejects_deferred_function_tools`
///
#[tokio::test]
async fn tools_to_session_tools_rejects_deferred_function_tools() {
    let agent =
        RealtimeAgent::<()>::new("root").with_tool(tool("lookup_account").with_defer_loading(true));
    let error = advertised_tools(agent).await.unwrap_err();
    assert!(error.contains("defer_loading=True"), "{error}");
}

/// The same Responses-only rejection holds on a live connect, where the initial settings carry
/// the tool list, and a failed connect leaves the model reusable.
#[tokio::test]
async fn connect_rejects_namespaced_and_deferred_tool_specs() {
    use openai_agents_sdk::realtime::{
        RealtimeFunctionToolSpec, RealtimeModelConfig, RealtimeSessionModelSettings,
    };

    for (spec, feature) in [
        (
            RealtimeFunctionToolSpec::new("lookup_account", "d", json!({})).with_namespace("crm"),
            "tool_namespace()",
        ),
        (
            RealtimeFunctionToolSpec::new("lookup_account", "d", json!({}))
                .with_defer_loading(true),
            "defer_loading=True",
        ),
    ] {
        let server = MockRealtimeServer::start().await;
        let model = OpenAIRealtimeWebSocketModel::new();
        let mut settings = RealtimeSessionModelSettings::new();
        settings.tools = Some(vec![spec]);
        let error = model
            .connect(
                RealtimeModelConfig::new()
                    .with_api_key("sk-test")
                    .with_url(server.url())
                    .with_initial_model_settings(settings),
            )
            .await
            .expect_err("connect rejects the tool")
            .to_string();
        assert!(error.contains(feature), "{error}");
        assert!(
            error.contains("cannot be used with Realtime models"),
            "{error}"
        );
        server.stop();
    }
}
