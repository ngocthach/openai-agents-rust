//! Ports of the interruption, response-ownership and per-connection-state tests of
//! `tests/realtime/test_openai_realtime.py` (`TestAudioResponseOwnership`, the interrupt and close
//! half of `TestSendEventAndConfig`) and `TestConversionHelperInterrupt` of
//! `tests/realtime/test_conversion_helpers.py`.
//!
//! Python mutates the audio tracker and the response sequencer directly and clocks playback with
//! `time.monotonic`. Here the same state is built by pushing server events through a local
//! websocket, playback is reported through a [`RealtimePlaybackTracker`], and every private-state
//! assertion becomes the next observable consequence: a second interrupt sends no second cancel,
//! a later speech-start emits no interruption, a late audio delta produces no audio event. No
//! network, no real API key.

#![cfg(feature = "realtime")]

use std::sync::Arc;

use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeInterrupt, RealtimeModel, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimePlaybackTracker,
    RealtimeSessionModelSettings, RealtimeTurnDetectionConfig,
};
use serde_json::{Value, json};

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

const SENTINEL: &str = "py.test.sentinel";

/// Everything emitted for the events pushed so far, minus connection status; see the events file.
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

fn interrupted_items(events: &[RealtimeModelEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RealtimeModelEvent::AudioInterrupted { item_id, .. } => Some(item_id.clone()),
            _ => None,
        })
        .collect()
}

fn has_error(events: &[RealtimeModelEvent]) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            RealtimeModelEvent::Error { .. } | RealtimeModelEvent::Exception { .. }
        )
    })
}

fn pcm16() -> Value {
    json!({"type": "audio/pcm", "rate": 24000})
}

/// One second of PCM16 at 24 kHz.
const SECOND: usize = 48_000;
/// 50 ms of PCM16 audio.
const FIFTY_MS: usize = 2400;

fn audio_delta(response_id: &str, item_id: &str, bytes: usize) -> Value {
    json!({
        "type": "response.output_audio.delta",
        "event_id": "event_audio",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": base64_encode(&vec![0u8; bytes]),
    })
}

fn response(response_id: &str, status: &str) -> Value {
    json!({"id": response_id, "object": "realtime.response", "status": status, "output": []})
}

fn created(response_id: &str) -> Value {
    json!({"type": "response.created", "event_id": "e_created", "response": response(response_id, "in_progress")})
}

fn done(response_id: &str) -> Value {
    json!({"type": "response.done", "event_id": "e_done", "response": response(response_id, "completed")})
}

fn speech_started() -> Value {
    json!({"type": "input_audio_buffer.speech_started", "event_id": "e_speech", "item_id": "user_2", "audio_start_ms": 0})
}

fn session_created_with(interrupt_response: bool) -> Value {
    json!({
        "type": "session.created",
        "event_id": "event_created",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime-2.1",
            "audio": {
                "input": {"turn_detection": {"type": "semantic_vad", "interrupt_response": interrupt_response}},
                "output": {"format": pcm16()},
            },
        },
    })
}

/// Connect with server-side automatic cancellation off, which is what Python assumes until the
/// server confirms its session (no `session.created` yet).
async fn manual_cancellation(
    tracker: Option<Arc<RealtimePlaybackTracker>>,
) -> (
    MockRealtimeServer,
    Arc<OpenAIRealtimeWebSocketModel>,
    RealtimeModelEvents,
) {
    let mut server = MockRealtimeServer::start().await;
    let settings =
        RealtimeSessionModelSettings::new().with_turn_detection(RealtimeTurnDetectionConfig {
            kind: Some("semantic_vad".to_string()),
            interrupt_response: Some(false),
            ..Default::default()
        });
    let (model, events) = connect(&server, Some(settings), tracker).await;
    let _ = server.next_frame("session.update").await;
    server.push(session_created_with(false));
    (server, model, events)
}

/// A marker that does not touch transport state, unlike a `session.update` the transport renders.
async fn marker(model: &OpenAIRealtimeWebSocketModel) {
    model
        .send_event(RealtimeModelSendEvent::RawMessage {
            message: json!({
                "type": "session.update",
                "session": {"instructions": MARKER_INSTRUCTIONS},
            }),
        })
        .await
        .unwrap();
}

/// The client frames of the given types sent so far, in order.
async fn frames_of(
    server: &mut MockRealtimeServer,
    model: &OpenAIRealtimeWebSocketModel,
    types: &[&str],
) -> Vec<Value> {
    marker(model).await;
    server
        .frames_before_marker()
        .await
        .into_iter()
        .filter(|frame| types.contains(&frame["type"].as_str().unwrap_or_default()))
        .collect()
}

fn truncates(frames: &[Value]) -> Vec<(String, i64)> {
    frames
        .iter()
        .filter(|frame| frame["type"] == "conversation.item.truncate")
        .map(|frame| {
            (
                frame["item_id"].as_str().unwrap().to_string(),
                frame["audio_end_ms"].as_i64().unwrap(),
            )
        })
        .collect()
}

fn cancels(frames: &[Value]) -> Vec<&Value> {
    frames
        .iter()
        .filter(|frame| frame["type"] == "response.cancel")
        .collect()
}

async fn send_interrupt(model: &OpenAIRealtimeWebSocketModel, interrupt: RealtimeInterrupt) {
    model
        .send_event(RealtimeModelSendEvent::Interrupt(interrupt))
        .await
        .unwrap();
}

fn response_only(response_id: &str) -> RealtimeInterrupt {
    RealtimeInterrupt {
        force_response_cancel: true,
        response_id: Some(response_id.to_string()),
        cancel_response_only: true,
        playback_only: false,
    }
}

// ---------------------------------------------------------------------------------------------
// TestAudioResponseOwnership
// ---------------------------------------------------------------------------------------------

/// `test_interrupt_truncates_only_unfinished_audio`: every combination of "how the interruption
/// arrives" (server VAD or the session's `interrupt()`) and "what was heard" (an earlier response,
/// the streaming one, or audio not yet played). Playback comes from a tracker; the Python clock
/// variants need `time.monotonic` patched and have no Rust counterpart.
#[tokio::test]
async fn interrupt_truncates_only_unfinished_audio() {
    // (start_second_response, elapsed_ms, expected_truncates)
    type Scenario = (bool, f64, Vec<(&'static str, i64)>);
    let scenarios: [Scenario; 3] = [
        (true, 5000.0, vec![]),
        (false, 5000.0, vec![("audio_1", 1000)]),
        (true, 500.0, vec![("audio_1", 500)]),
    ];
    for session_interrupt in [false, true] {
        for (start_second_response, elapsed_ms, expected) in &scenarios {
            let tracker = RealtimePlaybackTracker::shared();
            let (mut server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
            server.push(created("r1"));
            server.push(audio_delta("r1", "audio_1", SECOND));
            if *start_second_response {
                server.push(done("r1"));
                server.push(created("r2"));
                // A later response can generate tool arguments without producing audio.
                server.push(json!({
                    "type": "response.function_call_arguments.delta",
                    "event_id": "e5",
                    "response_id": "r2",
                    "item_id": "tool_2",
                    "output_index": 0,
                    "call_id": "call_2",
                    "delta": "{\"query\":",
                }));
            }
            let _ = settle(&server, &mut events).await;

            tracker.on_play_ms("audio_1", 0, elapsed_ms.min(1000.0));
            if session_interrupt {
                send_interrupt(&model, RealtimeInterrupt::forced()).await;
            } else {
                server.push(speech_started());
            }
            let emitted = settle(&server, &mut events).await;
            let frames = frames_of(
                &mut server,
                &model,
                &["conversation.item.truncate", "response.cancel"],
            )
            .await;

            let context = format!(
                "session_interrupt={session_interrupt} second={start_second_response} elapsed={elapsed_ms}"
            );
            let expected: Vec<(String, i64)> = expected
                .iter()
                .map(|(item, ms)| (item.to_string(), *ms))
                .collect();
            assert_eq!(truncates(&frames), expected, "{context}");
            assert_eq!(cancels(&frames).len(), 1, "{context}: {frames:#?}");
            assert!(!has_error(&emitted), "{context}");
            assert_eq!(interrupted_items(&emitted), ["audio_1"], "{context}");
            assert_eq!(tracker.state().current_item_id, None, "{context}");
            model.close().await.unwrap();
            server.stop();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// TestSendEventAndConfig: dispatch and cancellation
// ---------------------------------------------------------------------------------------------

/// `test_send_event_dispatch`: every event kind reaches the wire as the expected frames.
#[tokio::test]
async fn send_event_dispatch() {
    let (mut server, model, mut events) = manual_cancellation(None).await;

    model
        .send_event(RealtimeModelSendEvent::UserInput {
            user_input: "hi".into(),
        })
        .await
        .unwrap();
    // The first response starts and ends, freeing the sequencer for the tool output below.
    server.push(created("r1"));
    server.push(done("r1"));
    let _ = settle(&server, &mut events).await;
    model
        .send_event(RealtimeModelSendEvent::Audio {
            audio: b"a".to_vec(),
            commit: false,
        })
        .await
        .unwrap();
    model
        .send_event(RealtimeModelSendEvent::Audio {
            audio: b"a".to_vec(),
            commit: true,
        })
        .await
        .unwrap();
    model
        .send_event(RealtimeModelSendEvent::ToolOutput {
            tool_call: openai_agents_sdk::realtime::RealtimeModelToolCallEvent::new("t", "c", "{}"),
            output: "ok".to_string(),
            start_response: true,
        })
        .await
        .unwrap();
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();
    model
        .send_event(RealtimeModelSendEvent::session_update(
            RealtimeSessionModelSettings::new().with_voice("nova"),
        ))
        .await
        .unwrap();

    marker(&model).await;
    // The session.update that announces tracing after `session.created` is not part of the list.
    let types: Vec<String> = server
        .frames_before_marker()
        .await
        .iter()
        .filter(|frame| frame["session"].get("tracing").is_none())
        .map(|frame| frame["type"].as_str().unwrap().to_string())
        .collect();
    // user input -> 2, audio append -> 1, append+commit -> 2, tool output -> 2, interrupt -> 0
    // (nothing is playing and no response is active), session update -> 1: eight in all.
    assert_eq!(
        types,
        [
            "conversation.item.create",
            "response.create",
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "conversation.item.create",
            "response.create",
            "session.update",
        ]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_session_interrupt_cancels_active_text_response`: a button interruption cancels an active
/// response exactly once, however the server is configured, and never cancels an idle one.
#[tokio::test]
async fn session_interrupt_cancels_active_text_response() {
    for automatic_interrupt in [true, false] {
        for response_active in [true, false] {
            let mut server = MockRealtimeServer::start().await;
            let (model, mut events) = connect(&server, None, None).await;
            let _ = server.next_frame("session.update").await;
            server.push(json!({
                "type": "session.created",
                "event_id": "event_created",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": {
                        "type": "semantic_vad",
                        "interrupt_response": automatic_interrupt,
                    }}},
                },
            }));
            server.push(created("response_1"));
            if !response_active {
                server.push(done("response_1"));
            }
            let _ = settle(&server, &mut events).await;

            // A text/button interruption supplies no speech event to trigger VAD cancellation.
            model
                .send_event(RealtimeModelSendEvent::interrupt())
                .await
                .unwrap();
            model
                .send_event(RealtimeModelSendEvent::interrupt())
                .await
                .unwrap();

            let frames = frames_of(
                &mut server,
                &model,
                &["response.cancel", "conversation.item.truncate"],
            )
            .await;
            let context = format!("automatic={automatic_interrupt} active={response_active}");
            if response_active {
                assert_eq!(frames.len(), 1, "{context}: {frames:#?}");
                assert_eq!(frames[0], json!({"type": "response.cancel"}), "{context}");
            } else {
                assert!(frames.is_empty(), "{context}: {frames:#?}");
            }
            model.close().await.unwrap();
            server.stop();
        }
    }
}

/// `test_interrupt_force_cancel_overrides_auto_cancellation`
#[tokio::test]
async fn interrupt_force_cancel_overrides_auto_cancellation() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(session_created_with(true));
    server.push(audio_delta("response_1", "item_1", FIFTY_MS));
    server.push(created("response_1"));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, RealtimeInterrupt::forced()).await;
    let frames = frames_of(
        &mut server,
        &model,
        &["conversation.item.truncate", "response.cancel"],
    )
    .await;
    let types: Vec<&str> = frames.iter().map(|f| f["type"].as_str().unwrap()).collect();
    assert_eq!(types.len(), 2, "{frames:#?}");
    assert!(types.contains(&"conversation.item.truncate") && types.contains(&"response.cancel"));
    assert_eq!(cancels(&frames)[0], &json!({"type": "response.cancel"}));

    // The response is still running until the server says otherwise; the cancel is not repeated.
    send_interrupt(&model, RealtimeInterrupt::forced()).await;
    assert!(cancels(&frames_of(&mut server, &model, &["response.cancel"]).await).is_empty());

    // Once the response is done the audio is gone too: nothing is left to interrupt.
    let interrupted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&interrupted), ["item_1"]);
    server.push(done("response_1"));
    server.push(speech_started());
    let emitted = settle(&server, &mut events).await;
    assert!(interrupted_items(&emitted).is_empty(), "{emitted:#?}");
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_only_interrupt_targets_response_without_touching_audio`
#[tokio::test]
async fn response_only_interrupt_targets_response_without_touching_audio() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(audio_delta("response_1", "audio_item", FIFTY_MS));
    server.push(json!({"type": "response.created", "response": {}}));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, response_only("response_1")).await;
    let emitted = settle(&server, &mut events).await;
    assert!(interrupted_items(&emitted).is_empty(), "{emitted:#?}");
    let frames = frames_of(
        &mut server,
        &model,
        &["response.cancel", "conversation.item.truncate"],
    )
    .await;
    assert_eq!(frames.len(), 1, "{frames:#?}");
    assert_eq!(frames[0]["type"], "response.cancel");
    assert_eq!(frames[0]["response_id"], "response_1");

    // The audio is untouched: a speech-start still interrupts it.
    server.push(speech_started());
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["audio_item"]);
    model.close().await.unwrap();
    server.stop();
}

/// `test_concurrent_response_only_interrupts_cancel_response_once`
#[tokio::test]
async fn concurrent_response_only_interrupts_cancel_response_once() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(json!({"type": "response.created", "response": {}}));
    let _ = settle(&server, &mut events).await;

    let (first, second) = tokio::join!(
        model.send_event(RealtimeModelSendEvent::Interrupt(response_only(
            "response_1"
        ))),
        model.send_event(RealtimeModelSendEvent::Interrupt(response_only(
            "response_1"
        ))),
    );
    first.unwrap();
    second.unwrap();

    let frames = frames_of(&mut server, &model, &["response.cancel"]).await;
    assert_eq!(frames.len(), 1, "{frames:#?}");
    assert_eq!(frames[0]["response_id"], "response_1");

    // After the response is done the slot is free again.
    server.push(done("response_1"));
    server.push(created("response_2"));
    let _ = settle(&server, &mut events).await;
    send_interrupt(&model, response_only("response_2")).await;
    assert_eq!(
        cancels(&frames_of(&mut server, &model, &["response.cancel"]).await).len(),
        1
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_only_interrupt_stays_deduplicated_until_response_done`
#[tokio::test]
async fn response_only_interrupt_stays_deduplicated_until_response_done() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(created("response_1"));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, response_only("response_1")).await;
    send_interrupt(&model, response_only("response_1")).await;
    server.push(done("response_1"));
    server.push(created("response_2"));
    let _ = settle(&server, &mut events).await;
    send_interrupt(&model, response_only("response_2")).await;

    // A late `response.done` for the first response says nothing about the second: it is still
    // active and already being cancelled, so neither a second cancel nor a new create goes out.
    server.push(done("response_1"));
    let _ = settle(&server, &mut events).await;
    send_interrupt(&model, response_only("response_2")).await;
    model
        .send_event(RealtimeModelSendEvent::UserInput {
            user_input: "while response_2 runs".into(),
        })
        .await
        .unwrap();
    let frames = frames_of(
        &mut server,
        &model,
        &[
            "response.cancel",
            "response.create",
            "conversation.item.create",
        ],
    )
    .await;
    let cancelled: Vec<&str> = cancels(&frames)
        .iter()
        .map(|frame| frame["response_id"].as_str().unwrap())
        .collect();
    assert_eq!(cancelled, ["response_1", "response_2"]);
    assert!(
        !frames
            .iter()
            .any(|frame| frame["type"] == "response.create"),
        "a response.create must wait for response_2 to finish: {frames:#?}"
    );

    server.push(done("response_2"));
    let _ = settle(&server, &mut events).await;
    let frames = frames_of(&mut server, &model, &["response.create"]).await;
    assert_eq!(
        frames.len(),
        1,
        "the deferred create goes out once response_2 is done"
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_only_interrupt_skips_cancel_after_response_done`
#[tokio::test]
async fn response_only_interrupt_skips_cancel_after_response_done() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(audio_delta("response_1", "audio_item", FIFTY_MS));
    server.push(created("response_1"));
    server.push(done("response_1"));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, response_only("response_1")).await;
    let emitted = settle(&server, &mut events).await;
    assert!(interrupted_items(&emitted).is_empty(), "{emitted:#?}");
    assert!(
        frames_of(
            &mut server,
            &model,
            &["response.cancel", "conversation.item.truncate"]
        )
        .await
        .is_empty()
    );

    // The audio is still the last item, untouched.
    server.push(speech_started());
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["audio_item"]);
    model.close().await.unwrap();
    server.stop();
}

/// `test_normal_interrupt_targets_response_and_interrupts_audio`
#[tokio::test]
async fn normal_interrupt_targets_response_and_interrupts_audio() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(audio_delta("response_1", "audio_item", FIFTY_MS));
    server.push(json!({"type": "response.created", "response": {}}));
    let _ = settle(&server, &mut events).await;

    send_interrupt(
        &model,
        RealtimeInterrupt {
            force_response_cancel: true,
            response_id: Some("response_1".to_string()),
            ..Default::default()
        },
    )
    .await;
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["audio_item"]);

    let frames = frames_of(
        &mut server,
        &model,
        &["conversation.item.truncate", "response.cancel"],
    )
    .await;
    let types: Vec<&str> = frames.iter().map(|f| f["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["conversation.item.truncate", "response.cancel"]);
    assert_eq!(frames[1]["response_id"], "response_1");

    // The audio is released: a speech-start has nothing to interrupt.
    server.push(speech_started());
    assert!(interrupted_items(&settle(&server, &mut events).await).is_empty());
    model.close().await.unwrap();
    server.stop();
}

/// Build the "old response, new response" audio state used by the playback-only tests.
async fn old_and_new_audio(
    tracker: &Arc<RealtimePlaybackTracker>,
) -> (
    MockRealtimeServer,
    Arc<OpenAIRealtimeWebSocketModel>,
    RealtimeModelEvents,
) {
    let (server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(audio_delta("old_response", "old_audio_item", FIFTY_MS));
    server.push(audio_delta("new_response", "new_audio_item", FIFTY_MS));
    let _ = settle(&server, &mut events).await;
    (server, model, events)
}

/// `test_playback_only_interrupt_does_not_stop_newer_response_audio`
#[tokio::test]
async fn playback_only_interrupt_does_not_stop_newer_response_audio() {
    let tracker = RealtimePlaybackTracker::shared();
    let (mut server, model, mut events) = old_and_new_audio(&tracker).await;
    tracker.on_play_ms("new_audio_item", 0, 50.0);
    server.push(json!({"type": "response.created", "response": {}}));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, RealtimeInterrupt::playback_only("old_response")).await;
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["old_audio_item"]);
    assert!(
        frames_of(
            &mut server,
            &model,
            &["conversation.item.truncate", "response.cancel"]
        )
        .await
        .is_empty()
    );
    // The newer response keeps playing, and the old one's index is gone.
    assert_eq!(
        tracker.state().current_item_id.as_deref(),
        Some("new_audio_item")
    );
    send_interrupt(&model, RealtimeInterrupt::playback_only("old_response")).await;
    assert!(interrupted_items(&settle(&server, &mut events).await).is_empty());
    model.close().await.unwrap();
    server.stop();
}

/// `test_release_response_waiters_preserves_audio_for_delayed_guardrail`: a response that has
/// finished keeps its audio index until the session retires it, so a guardrail that settles late
/// can still interrupt playback.
#[tokio::test]
async fn finished_response_keeps_audio_for_a_delayed_guardrail() {
    let tracker = RealtimePlaybackTracker::shared();
    let (server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(created("source_response"));
    server.push(audio_delta("source_response", "source_item", FIFTY_MS));
    server.push(done("source_response"));
    let _ = settle(&server, &mut events).await;
    tracker.on_play_ms("source_item", 0, 50.0);

    send_interrupt(&model, RealtimeInterrupt::playback_only("source_response")).await;
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["source_item"]);
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_scoped_interrupt_suppresses_late_source_audio_until_done`
#[tokio::test]
async fn response_scoped_interrupt_suppresses_late_source_audio_until_done() {
    let (server, model, mut events) = manual_cancellation(None).await;
    send_interrupt(
        &model,
        RealtimeInterrupt {
            force_response_cancel: true,
            response_id: Some("source_response".to_string()),
            ..Default::default()
        },
    )
    .await;

    server.push(audio_delta("source_response", "source_item", 4));
    server.push(audio_delta("newer_response", "newer_item", 4));
    let emitted = settle(&server, &mut events).await;
    let audio: Vec<&str> = emitted
        .iter()
        .filter_map(|event| match event {
            RealtimeModelEvent::Audio(audio) => Some(audio.response_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(audio, ["newer_response"]);

    // The suppression ends with the response: its done event releases it.
    server.push(done("source_response"));
    server.push(audio_delta("source_response", "source_item_again", 4));
    let emitted = settle(&server, &mut events).await;
    assert!(emitted.iter().any(|event| matches!(
        event,
        RealtimeModelEvent::Audio(audio) if audio.item_id == "source_item_again"
    )));
    model.close().await.unwrap();
    server.stop();
}

/// `test_response_audio_indexes_are_bounded_after_retirement`
#[tokio::test]
async fn response_audio_indexes_are_bounded_after_retirement() {
    let (server, model, mut events) = manual_cancellation(None).await;
    for number in 0..20 {
        server.push(audio_delta(
            &format!("response_{number}"),
            &format!("item_{number}"),
            FIFTY_MS,
        ));
    }
    let _ = settle(&server, &mut events).await;
    for number in 0..20 {
        model
            .retire_response_audio(&format!("response_{number}"))
            .await
            .unwrap();
    }

    // Every retired response has lost its index...
    for number in [0, 7, 19] {
        send_interrupt(
            &model,
            RealtimeInterrupt::playback_only(format!("response_{number}")),
        )
        .await;
    }
    assert!(interrupted_items(&settle(&server, &mut events).await).is_empty());
    // ...while the latest item's state survives retirement.
    server.push(speech_started());
    assert_eq!(
        interrupted_items(&settle(&server, &mut events).await),
        ["item_19"]
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_custom_playback_that_never_starts_releases_response_index`
#[tokio::test]
async fn custom_playback_that_never_starts_releases_response_index() {
    let tracker = RealtimePlaybackTracker::shared();
    let (server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(audio_delta("response_1", "first_item", FIFTY_MS));
    server.push(audio_delta("response_1", "second_item", FIFTY_MS));
    let _ = settle(&server, &mut events).await;

    model.retire_response_audio("response_1").await.unwrap();
    send_interrupt(&model, RealtimeInterrupt::playback_only("response_1")).await;
    assert!(interrupted_items(&settle(&server, &mut events).await).is_empty());
    model.close().await.unwrap();
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Close and reconnect
// ---------------------------------------------------------------------------------------------

/// `test_close_releases_pending_response_audio_indexes` and the tracker reset of
/// `test_close_failure_releases_pending_response_audio_indexes`: closing forgets which responses
/// were interrupted, and a caller's tracker stops pointing at the closed session's item.
#[tokio::test]
async fn close_releases_pending_response_audio_indexes() {
    let tracker = RealtimePlaybackTracker::shared();
    let (server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(audio_delta("response_1", "audio_item", FIFTY_MS));
    let _ = settle(&server, &mut events).await;
    tracker.on_play_ms("audio_item", 0, 100.0);
    // Marks response_1 as interrupted, so its later audio would be dropped.
    send_interrupt(
        &model,
        RealtimeInterrupt {
            force_response_cancel: true,
            response_id: Some("response_1".to_string()),
            ..Default::default()
        },
    )
    .await;
    tracker.on_play_ms("audio_item", 0, 100.0);

    model.close().await.unwrap();
    assert_eq!(tracker.state().current_item_id, None);
    server.stop();

    // A new connection starts clean: response_1's audio is delivered again.
    let mut second = MockRealtimeServer::start().await;
    let (model, mut events) = connect(&second, None, Some(tracker.clone())).await;
    let _ = second.next_frame("session.update").await;
    second.push(audio_delta("response_1", "audio_item", FIFTY_MS));
    let emitted = settle(&second, &mut events).await;
    assert!(
        emitted
            .iter()
            .any(|event| matches!(event, RealtimeModelEvent::Audio(_))),
        "{emitted:#?}"
    );
    model.close().await.unwrap();
    second.stop();
}

/// Drive one connection to a state with live audio and a finished response.
async fn play_one_item(
    server: &MockRealtimeServer,
    events: &mut RealtimeModelEvents,
    tracker: Option<&Arc<RealtimePlaybackTracker>>,
) {
    server.push(session_created_with(true));
    server.push(json!({
        "type": "response.output_audio.delta",
        "event_id": "event_1",
        "response_id": "response_old",
        "item_id": "item_old",
        "output_index": 0,
        "content_index": 0,
        "delta": base64_encode(&[0u8, 1u8].repeat(2400)),
    }));
    server.push(
        json!({"type": "response.done", "event_id": "event_2", "response": {"id": "response_old"}}),
    );
    let _ = settle(server, events).await;
    if let Some(tracker) = tracker {
        // Fully played audio still leaves the caller's tracker pointing at the item.
        tracker.on_play_ms("item_old", 0, 100.0);
    }
}

/// `test_close_resets_per_connection_item_and_session_state` and
/// `test_reconnect_after_close_does_not_act_on_previous_session_items` (both tracker variants):
/// the runner reuses one model across runs, so a second connection must start with no memory of
/// the first one's items.
#[tokio::test]
async fn reconnect_after_close_does_not_act_on_previous_session_items() {
    for use_tracker in [false, true] {
        let tracker = use_tracker.then(RealtimePlaybackTracker::shared);
        let model = Arc::new(OpenAIRealtimeWebSocketModel::new());
        let config_for = |server: &MockRealtimeServer| {
            let mut config = openai_agents_sdk::realtime::RealtimeModelConfig::new()
                .with_api_key("test-key")
                .with_url(server.url());
            if let Some(tracker) = &tracker {
                config = config.with_playback_tracker(tracker.clone());
            }
            config
        };

        let mut first = MockRealtimeServer::start().await;
        let mut events = model.connect(config_for(&first)).await.unwrap();
        let _ = first.next_frame("session.update").await;
        play_one_item(&first, &mut events, tracker.as_ref()).await;
        model.close().await.unwrap();
        first.stop();

        let mut second = MockRealtimeServer::start().await;
        let mut events = model.connect(config_for(&second)).await.unwrap();
        let _ = second.next_frame("session.update").await;
        second.push(session_created_with(true));
        second.push(json!({
            "type": "input_audio_buffer.speech_started",
            "event_id": "event_3",
            "audio_start_ms": 0,
            "item_id": "item_user_new",
        }));
        second.push(json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "event_4",
            "item_id": "item_user_new",
            "content_index": 0,
            "transcript": "hello",
            "usage": {"type": "tokens", "total_tokens": 1, "input_tokens": 1, "output_tokens": 0},
        }));
        let _ = settle(&second, &mut events).await;
        model
            .send_event(RealtimeModelSendEvent::interrupt())
            .await
            .unwrap();
        let emitted = settle(&second, &mut events).await;

        assert!(
            interrupted_items(&emitted).is_empty(),
            "tracker={use_tracker}: {emitted:#?}"
        );
        let frames = frames_of(
            &mut second,
            &model,
            &["conversation.item.truncate", "conversation.item.retrieve"],
        )
        .await;
        assert!(frames.is_empty(), "tracker={use_tracker}: {frames:#?}");
        model.close().await.unwrap();
        second.stop();
    }
}

/// `test_close_cancels_waiting_response_create_after_active_response`: a response.create deferred
/// behind an active response dies with the connection and is not replayed on the next one.
#[tokio::test]
async fn close_cancels_waiting_response_create_after_active_response() {
    let model = Arc::new(OpenAIRealtimeWebSocketModel::new());
    let mut first = MockRealtimeServer::start().await;
    let mut events = model
        .connect(
            openai_agents_sdk::realtime::RealtimeModelConfig::new()
                .with_api_key("k")
                .with_url(first.url()),
        )
        .await
        .unwrap();
    let _ = first.next_frame("session.update").await;
    first.push(created("r1"));
    let _ = settle(&first, &mut events).await;
    model
        .send_event(RealtimeModelSendEvent::UserInput {
            user_input: "hi".into(),
        })
        .await
        .unwrap();
    let before = frames_of(
        &mut first,
        &model,
        &["conversation.item.create", "response.create"],
    )
    .await;
    assert_eq!(before.len(), 1);
    assert_eq!(before[0]["type"], "conversation.item.create");
    model.close().await.unwrap();
    first.stop();

    // On the next connection the old deferral must not fire when a response ends.
    let mut second = MockRealtimeServer::start().await;
    let mut events = model
        .connect(
            openai_agents_sdk::realtime::RealtimeModelConfig::new()
                .with_api_key("k")
                .with_url(second.url()),
        )
        .await
        .unwrap();
    let _ = second.next_frame("session.update").await;
    second.push(created("r2"));
    second.push(done("r2"));
    let _ = settle(&second, &mut events).await;
    let frames = frames_of(&mut second, &model, &["response.create"]).await;
    assert!(frames.is_empty(), "{frames:#?}");
    model.close().await.unwrap();
    second.stop();
}

// ---------------------------------------------------------------------------------------------
// Interrupt validation and clock edge cases
// ---------------------------------------------------------------------------------------------

/// `test_response_only_interrupt_requires_response_id`
#[tokio::test]
async fn response_only_interrupt_requires_response_id() {
    let (_server, model, _events) = manual_cancellation(None).await;
    let error = model
        .send_event(RealtimeModelSendEvent::Interrupt(RealtimeInterrupt {
            cancel_response_only: true,
            ..Default::default()
        }))
        .await
        .expect_err("a response-only cancel needs a response");
    assert!(
        error
            .to_string()
            .contains("cancel_response_only requires response_id"),
        "{error}"
    );
}

/// `test_interrupt_rejects_contradictory_modes_before_side_effects`
#[tokio::test]
async fn interrupt_rejects_contradictory_modes_before_side_effects() {
    for mode in [
        RealtimeInterrupt {
            cancel_response_only: true,
            ..Default::default()
        },
        RealtimeInterrupt {
            force_response_cancel: true,
            ..Default::default()
        },
    ] {
        let (mut server, model, mut events) = manual_cancellation(None).await;
        server.push(created("response_1"));
        server.push(audio_delta("response_1", "audio_item", FIFTY_MS));
        let _ = settle(&server, &mut events).await;

        let error = model
            .send_event(RealtimeModelSendEvent::Interrupt(RealtimeInterrupt {
                response_id: Some("response_1".to_string()),
                playback_only: true,
                ..mode
            }))
            .await
            .expect_err("playback_only contradicts an explicit cancellation mode");
        assert!(
            error
                .to_string()
                .contains("playback_only cannot be combined with explicit cancellation modes"),
            "{error}"
        );
        // No frame went out and nothing was reported.
        assert!(
            frames_of(
                &mut server,
                &model,
                &["conversation.item.truncate", "response.cancel"]
            )
            .await
            .is_empty()
        );
        assert!(settle(&server, &mut events).await.is_empty());
        model.close().await.unwrap();
        server.stop();
    }
}

/// `test_interrupt_truncates_at_zero_when_no_time_has_elapsed`
#[tokio::test]
async fn interrupt_truncates_at_zero_when_no_time_has_elapsed() {
    let tracker = RealtimePlaybackTracker::shared();
    let (mut server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(audio_delta("response_1", "item_1", FIFTY_MS));
    let _ = settle(&server, &mut events).await;
    // Playback began but no time has elapsed: still a real position, not "nothing to truncate".
    tracker.on_play_ms("item_1", 0, 0.0);

    send_interrupt(&model, RealtimeInterrupt::default()).await;
    let emitted = settle(&server, &mut events).await;
    assert_eq!(interrupted_items(&emitted), ["item_1"]);
    let frames = frames_of(&mut server, &model, &["conversation.item.truncate"]).await;
    assert_eq!(truncates(&frames), [("item_1".to_string(), 0)]);
    assert_eq!(frames[0]["content_index"], 0);
    model.close().await.unwrap();
    server.stop();
}

/// `test_interrupt_skips_truncate_when_elapsed_is_negative`
#[tokio::test]
async fn interrupt_skips_truncate_when_elapsed_is_negative() {
    let tracker = RealtimePlaybackTracker::shared();
    let (mut server, model, mut events) = manual_cancellation(Some(tracker.clone())).await;
    server.push(audio_delta("response_1", "item_1", FIFTY_MS));
    let _ = settle(&server, &mut events).await;
    // A clock that went backwards is still not a truncation position.
    tracker.on_play_ms("item_1", 0, -1.0);

    send_interrupt(&model, RealtimeInterrupt::default()).await;
    let emitted = settle(&server, &mut events).await;
    assert!(interrupted_items(&emitted).is_empty(), "{emitted:#?}");
    assert!(
        frames_of(&mut server, &model, &["conversation.item.truncate"])
            .await
            .is_empty()
    );
    model.close().await.unwrap();
    server.stop();
}

/// `test_interrupt_respects_auto_cancellation_when_not_forced`
#[tokio::test]
async fn interrupt_respects_auto_cancellation_when_not_forced() {
    let (mut server, model, mut events) = manual_cancellation(None).await;
    server.push(session_created_with(true));
    server.push(audio_delta("response_1", "item_1", FIFTY_MS));
    server.push(created("response_1"));
    let _ = settle(&server, &mut events).await;

    send_interrupt(&model, RealtimeInterrupt::default()).await;
    let frames = frames_of(
        &mut server,
        &model,
        &["conversation.item.truncate", "response.cancel"],
    )
    .await;
    assert_eq!(frames.len(), 1, "{frames:#?}");
    assert_eq!(frames[0]["type"], "conversation.item.truncate");
    model.close().await.unwrap();
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_conversion_helpers.py: TestConversionHelperInterrupt
// ---------------------------------------------------------------------------------------------

/// Interrupt `item_id`/`content_index` at `elapsed_ms` and return the truncate frame it sends.
async fn truncate_for(item_id: &str, content_index: usize, elapsed_ms: f64) -> Value {
    let tracker = RealtimePlaybackTracker::shared();
    let (mut server, model, _events) = manual_cancellation(Some(tracker.clone())).await;
    // The tracker alone places playback, so the truncate carries its values unclamped.
    tracker.on_play_ms(item_id, content_index, elapsed_ms);
    send_interrupt(&model, RealtimeInterrupt::default()).await;
    let frame = server.next_frame("conversation.item.truncate").await;
    model.close().await.unwrap();
    server.stop();
    frame
}

/// `test_convert_interrupt`
#[tokio::test]
async fn convert_interrupt() {
    let frame = truncate_for("item_789", 2, 1500.0).await;
    assert_eq!(frame["type"], "conversation.item.truncate");
    assert_eq!(frame["item_id"], "item_789");
    assert_eq!(frame["content_index"], 2);
    assert_eq!(frame["audio_end_ms"], 1500);
}

/// `test_convert_interrupt_zero_time`
#[tokio::test]
async fn convert_interrupt_zero_time() {
    let frame = truncate_for("item_1", 0, 0.0).await;
    assert_eq!(frame["item_id"], "item_1");
    assert_eq!(frame["content_index"], 0);
    assert_eq!(frame["audio_end_ms"], 0);
}

/// `test_convert_interrupt_large_values`
#[tokio::test]
async fn convert_interrupt_large_values() {
    let frame = truncate_for("item_xyz", 99, 999_999.0).await;
    assert_eq!(frame["item_id"], "item_xyz");
    assert_eq!(frame["content_index"], 99);
    assert_eq!(frame["audio_end_ms"], 999_999);
}

/// `test_convert_interrupt_empty_item_id`
#[tokio::test]
async fn convert_interrupt_empty_item_id() {
    let frame = truncate_for("", 1, 100.0).await;
    assert_eq!(frame["item_id"], "");
    assert_eq!(frame["content_index"], 1);
    assert_eq!(frame["audio_end_ms"], 100);
}
