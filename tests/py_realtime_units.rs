//! Ports of the Python SDK's unit-level realtime tests.
//!
//! Covers `tests/realtime/test_agent.py`, `test_audio_formats_unit.py`, `test_model_events.py`,
//! `test_playback_tracker.py` (the pure-unit half) and `test_playback_tracker_manual_unit.py`.
//! The transport-driven halves live in `py_realtime_transport.rs`.

#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::realtime::{
    RealtimeAgent, RealtimeAudioFormat, RealtimeInstructions, RealtimeInterrupt,
    RealtimeModelCachedTokensDetails, RealtimeModelEvent, RealtimeModelInputTokensDetails,
    RealtimeModelOutputTokensDetails, RealtimePlaybackTracker, calculate_audio_length_ms,
    reset_dont_log_model_data, set_dont_log_model_data,
};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{RunContext, tool::FunctionTool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// Floating point comparison with the tolerance Python's `pytest.approx(..., abs=1e-6)` uses.
fn approx(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_agent.py
// ---------------------------------------------------------------------------------------------

#[test]
fn can_initialize_realtime_agent() {
    let agent = RealtimeAgent::<()>::new("test").with_instructions("Hello");
    assert_eq!(agent.name, "test");
    assert!(matches!(
        agent.instructions.as_ref(),
        Some(RealtimeInstructions::Static(s)) if s == "Hello"
    ));
}

#[tokio::test]
async fn dynamic_instructions() {
    let agent = RealtimeAgent::<()>::new("test").into_arc();
    // An agent with no instructions resolves to nothing, as in Python.
    assert_eq!(
        agent.get_system_prompt(&RunContext::new(())).await.unwrap(),
        None
    );

    let agent = Arc::new(RealtimeAgent::<()>::new("test").with_instructions_fn(
        |_ctx, agt: Arc<RealtimeAgent<()>>| async move {
            assert_eq!(agt.name, "test");
            Ok("Dynamic".to_string())
        },
    ));
    let instructions = agent.get_system_prompt(&RunContext::new(())).await.unwrap();
    assert_eq!(instructions.as_deref(), Some("Dynamic"));
}

#[tokio::test]
async fn async_callable_object_instructions_are_awaited() {
    // Python has to special-case callables whose `__call__` is async; in Rust an instructions
    // function always returns a future, so the awaiting is structural. The observable
    // behavior the Python test pins down is that the body runs exactly once per resolution.
    let calls = Arc::new(Mutex::new(0usize));
    let counter = calls.clone();
    let agent = Arc::new(RealtimeAgent::<()>::new("test").with_instructions_fn(
        move |_ctx, _agt| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok("Dynamic async callable".to_string())
            }
        },
    ));

    let prompt = agent.get_system_prompt(&RunContext::new(())).await.unwrap();
    assert_eq!(prompt.as_deref(), Some("Dynamic async callable"));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

#[test]
fn clone_does_not_mutate_original_lists() {
    let original = RealtimeAgent::<()>::new("orig");
    assert!(original.tools.is_empty());

    let cloned = original.clone().with_tool(FunctionTool::new(
        "t1",
        "t1",
        |_ctx, _a: NoArgs| async move { Ok("ok".to_string()) },
    ));

    // The original keeps its empty list; Rust's `Clone` copies the vectors by value, so there is
    // no shared-reference caveat the Python test has to document.
    assert!(original.tools.is_empty());
    assert_eq!(cloned.tools.len(), 1);
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_audio_formats_unit.py
// ---------------------------------------------------------------------------------------------

/// The normalized `type` of a format, or `None` when the format is unknown.
fn format_type(format: &RealtimeAudioFormat) -> Option<String> {
    format
        .to_session_value()
        .and_then(|value| value["type"].as_str().map(str::to_string))
}

#[test]
fn to_realtime_audio_format_from_strings() {
    for (name, expected) in [
        ("pcm", "audio/pcm"),
        ("pcm16", "audio/pcm"),
        ("audio/pcm", "audio/pcm"),
        ("pcmu", "audio/pcmu"),
        ("audio/pcmu", "audio/pcmu"),
        ("g711_ulaw", "audio/pcmu"),
        ("pcma", "audio/pcma"),
        ("audio/pcma", "audio/pcma"),
        ("g711_alaw", "audio/pcma"),
    ] {
        assert_eq!(
            format_type(&RealtimeAudioFormat::from(name)).as_deref(),
            Some(expected),
            "{name}"
        );
    }
}

#[test]
fn to_realtime_audio_format_passthrough_and_unknown_logs() {
    // A GA format object normalizes to itself.
    let fmt = RealtimeAudioFormat::from(json!({"type": "audio/pcm", "rate": 24000}));
    assert_eq!(
        fmt.to_session_value(),
        Some(json!({"type": "audio/pcm", "rate": 24000}))
    );

    // An unknown name produces nothing (and logs at debug level internally).
    assert_eq!(
        RealtimeAudioFormat::from("something_else").to_session_value(),
        None
    );
}

#[test]
fn to_realtime_audio_format_none() {
    // Python passes `None` through; the Rust API models the absence as `Option`, so the
    // equivalent is that no format value is produced at all.
    let format: Option<RealtimeAudioFormat> = None;
    assert_eq!(format.as_ref().and_then(|f| f.to_session_value()), None);
}

#[test]
fn to_realtime_audio_format_from_mapping() {
    let exact = RealtimeAudioFormat::from(json!({"type": "audio/pcm", "rate": 24000}));
    assert_eq!(
        exact.to_session_value(),
        Some(json!({"type": "audio/pcm", "rate": 24000}))
    );

    // Any other PCM rate is coerced to the one rate the API supports.
    let other_rate = RealtimeAudioFormat::from(json!({"type": "audio/pcm", "rate": 16000}));
    assert_eq!(
        other_rate.to_session_value(),
        Some(json!({"type": "audio/pcm", "rate": 24000}))
    );

    let default_rate = RealtimeAudioFormat::from(json!({"type": "audio/pcm"}));
    assert_eq!(
        default_rate.to_session_value(),
        Some(json!({"type": "audio/pcm", "rate": 24000}))
    );

    assert_eq!(
        RealtimeAudioFormat::from(json!({"type": "audio/pcmu"})).to_session_value(),
        Some(json!({"type": "audio/pcmu"}))
    );
    assert_eq!(
        RealtimeAudioFormat::from(json!({"type": "audio/pcma"})).to_session_value(),
        Some(json!({"type": "audio/pcma"}))
    );
    assert_eq!(
        RealtimeAudioFormat::from(json!({"type": "audio/unknown", "rate": 8000}))
            .to_session_value(),
        None
    );
}

#[test]
fn to_realtime_audio_format_falls_back_for_non_finite_pcm_rate() {
    // JSON cannot carry NaN or an infinity, so the Rust equivalent of Python's non-finite rate
    // is a rate that does not read back as a finite number. Either way the PCM rate is coerced.
    for rate in [json!(null), json!("nan"), json!("inf")] {
        let format = RealtimeAudioFormat::from(json!({"type": "audio/pcm", "rate": rate}));
        assert_eq!(
            format.to_session_value(),
            Some(json!({"type": "audio/pcm", "rate": 24000})),
            "{rate}"
        );
    }
}

/// Captures `log` records so the redaction tests can read what was actually emitted.
#[derive(Default)]
struct CapturingLogger {
    records: Mutex<Vec<String>>,
}

impl log::Log for CapturingLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.records.lock().unwrap().push(record.args().to_string());
    }

    fn flush(&self) {}
}

fn capturing_logger() -> &'static CapturingLogger {
    static LOGGER: OnceLock<&'static CapturingLogger> = OnceLock::new();
    LOGGER.get_or_init(|| {
        let logger: &'static CapturingLogger = Box::leak(Box::new(CapturingLogger::default()));
        // Another integration test binary may already have installed a logger; either way the
        // records this test reads come from the logger it managed to install.
        let _ = log::set_logger(logger);
        log::set_max_level(log::LevelFilter::Trace);
        logger
    })
}

/// Everything logged while `body` ran.
fn captured<F: FnOnce()>(body: F) -> Vec<String> {
    let logger = capturing_logger();
    logger.records.lock().unwrap().clear();
    body();
    logger.records.lock().unwrap().clone()
}

/// The redaction flags are global, so these tests run under one lock and restore them.
fn redaction_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn to_realtime_audio_format_redacts_unknown_values() {
    let _guard = redaction_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_dont_log_model_data(true);

    // An unknown format name, an unknown PCM rate and an unknown mapping all log without the
    // offending value, exactly as Python does when `DONT_LOG_MODEL_DATA` is set.
    let records = captured(|| {
        assert_eq!(
            RealtimeAudioFormat::from("format-secret").to_session_value(),
            None
        );
    });
    assert!(records.iter().any(|r| r == "Unknown realtime audio format"));
    assert!(!records.iter().any(|r| r.contains("format-secret")));

    let records = captured(|| {
        assert_eq!(
            RealtimeAudioFormat::from(json!({"type": "audio/pcm", "rate": 8000}))
                .to_session_value(),
            Some(json!({"type": "audio/pcm", "rate": 24000}))
        );
    });
    assert!(
        records
            .iter()
            .any(|r| r == "Unknown PCM rate in a realtime audio format")
    );
    assert!(!records.iter().any(|r| r.contains("8000")));

    let records = captured(|| {
        assert_eq!(
            RealtimeAudioFormat::from(json!({"type": "format-secret", "nested": "mapping-secret"}))
                .to_session_value(),
            None
        );
    });
    assert!(
        records
            .iter()
            .any(|r| r == "Unknown realtime audio format mapping")
    );
    assert!(!records.iter().any(|r| r.contains("mapping-secret")));

    reset_dont_log_model_data();
}

#[test]
fn to_realtime_audio_format_preserves_diagnostic_mapping() {
    let _guard = redaction_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_dont_log_model_data(false);

    let records = captured(|| {
        assert_eq!(
            RealtimeAudioFormat::from(json!({"type": "format-secret", "nested": "mapping-secret"}))
                .to_session_value(),
            None
        );
    });
    // With redaction off the diagnostic keeps the whole mapping.
    let line = records
        .iter()
        .find(|r| r.starts_with("Unknown realtime audio format mapping"))
        .expect("the mapping diagnostic was logged");
    assert!(line.contains("format-secret"), "{line}");
    assert!(line.contains("mapping-secret"), "{line}");

    reset_dont_log_model_data();
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_model_events.py
// ---------------------------------------------------------------------------------------------

#[test]
fn all_events_have_type() {
    let events = [
        RealtimeModelEvent::Error { error: json!({}) },
        RealtimeModelEvent::AudioInterrupted {
            item_id: "i".into(),
            content_index: 0,
        },
        RealtimeModelEvent::AudioDone {
            item_id: "i".into(),
            content_index: 0,
        },
        RealtimeModelEvent::InputAudioTranscriptionCompleted {
            item_id: "i".into(),
            transcript: "t".into(),
        },
        RealtimeModelEvent::InputAudioTimeoutTriggered {
            item_id: "i".into(),
            audio_start_ms: 0,
            audio_end_ms: 0,
        },
        RealtimeModelEvent::TranscriptDelta {
            item_id: "i".into(),
            delta: "d".into(),
            response_id: "r".into(),
        },
        RealtimeModelEvent::OutputTextDelta {
            item_id: "i".into(),
            delta: "d".into(),
            response_id: "r".into(),
        },
        RealtimeModelEvent::ItemDeleted {
            item_id: "i".into(),
        },
        RealtimeModelEvent::EndOfStream,
        RealtimeModelEvent::TurnStarted { response_id: None },
        RealtimeModelEvent::TurnEnded { response_id: None },
        RealtimeModelEvent::Other { data: json!({}) },
        RealtimeModelEvent::Exception {
            message: "m".into(),
            context: None,
        },
        RealtimeModelEvent::RawServerEvent { data: json!({}) },
        RealtimeModelEvent::usage(Usage::default()),
    ];
    assert!(!events.is_empty());
    for event in &events {
        assert!(!event.event_type().is_empty(), "{event:?}");
    }
}

#[test]
fn usage_event_types_are_publicly_exported() {
    // The three modality-breakdown types and the usage event itself are part of the public
    // surface, so a custom transport can build them without reaching for provider types.
    let _cached = RealtimeModelCachedTokensDetails::default();
    let _input = RealtimeModelInputTokensDetails::default();
    let _output = RealtimeModelOutputTokensDetails::default();
    assert_eq!(
        RealtimeModelEvent::usage(Usage::default()).event_type(),
        "usage"
    );
}

#[test]
fn end_of_stream_event_is_publicly_exported() {
    assert_eq!(
        RealtimeModelEvent::EndOfStream.event_type(),
        "end_of_stream"
    );
}

#[test]
fn custom_model_can_construct_typed_usage_without_provider_types() {
    let event = RealtimeModelEvent::Usage {
        usage: Box::new(Usage {
            requests: 1,
            input_tokens: 8,
            output_tokens: 5,
            total_tokens: 13,
            ..Default::default()
        }),
        input_tokens_details: Some(RealtimeModelInputTokensDetails {
            text_tokens: Some(2),
            audio_tokens: Some(6),
            cached_tokens: Some(3),
            cached_tokens_details: Some(RealtimeModelCachedTokensDetails {
                text_tokens: Some(1),
                audio_tokens: Some(2),
                image_tokens: None,
            }),
            image_tokens: None,
        }),
        output_tokens_details: Some(RealtimeModelOutputTokensDetails {
            text_tokens: Some(1),
            audio_tokens: Some(4),
        }),
    };

    let RealtimeModelEvent::Usage {
        input_tokens_details,
        output_tokens_details,
        ..
    } = &event
    else {
        panic!("expected a usage event");
    };
    assert_eq!(input_tokens_details.as_ref().unwrap().audio_tokens, Some(6));
    assert_eq!(
        output_tokens_details.as_ref().unwrap().audio_tokens,
        Some(4)
    );
}

#[test]
fn turn_ended_response_id_defaults_to_none() {
    let event = RealtimeModelEvent::TurnEnded { response_id: None };
    assert_eq!(event.event_type(), "turn_ended");
    let RealtimeModelEvent::TurnEnded { response_id } = &event else {
        unreachable!();
    };
    assert_eq!(response_id.as_deref(), None);
}

#[test]
fn interrupt_playback_only_defaults_to_false() {
    // Python's positional construction maps to Rust's named fields; `playback_only` stays off
    // unless it is asked for.
    let interrupt = RealtimeInterrupt {
        force_response_cancel: true,
        response_id: Some("response_1".to_string()),
        cancel_response_only: true,
        ..Default::default()
    };
    assert!(interrupt.force_response_cancel);
    assert_eq!(interrupt.response_id.as_deref(), Some("response_1"));
    assert!(interrupt.cancel_response_only);
    assert!(!interrupt.playback_only);
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_playback_tracker.py (unit half) + test_playback_tracker_manual_unit.py
// ---------------------------------------------------------------------------------------------

#[test]
fn state_cleanup_on_interruption() {
    let tracker = RealtimePlaybackTracker::new();
    tracker.on_play_ms("item_1", 0, 100.0);

    let state = tracker.state();
    assert_eq!(state.current_item_id.as_deref(), Some("item_1"));
    assert_eq!(state.elapsed_ms, Some(100.0));

    tracker.on_interrupted();
    let state = tracker.state();
    assert_eq!(state.current_item_id, None);
    assert_eq!(state.elapsed_ms, None);
}

#[test]
fn audio_length_calculation_with_different_formats() {
    // G.711 is one byte per sample at 8 kHz.
    assert_eq!(
        calculate_audio_length_ms(Some("g711_ulaw"), b"12345678"),
        1.0
    );

    // PCM16 at 24 kHz is the default.
    let expected = (4.0 / (24_000.0 * 2.0)) * 1000.0;
    approx(calculate_audio_length_ms(Some("pcm16"), b"test"), expected);
    approx(calculate_audio_length_ms(None, b"test"), expected);
}

#[test]
fn g711_length_is_correct_for_every_format_spelling() {
    // 8000 bytes of G.711 is exactly one second, however the format is spelled.
    for spelling in [
        "audio/pcmu",
        "audio/pcma",
        "pcmu",
        "pcma",
        "g711_ulaw",
        "g711_alaw",
    ] {
        assert_eq!(
            calculate_audio_length_ms(Some(spelling), &[0u8; 8000]),
            1000.0,
            "{spelling}"
        );
    }
}

#[test]
fn pcm_spellings_keep_pcm16_math() {
    // 48000 bytes of PCM16 at 24 kHz is exactly one second.
    for spelling in ["audio/pcm", "pcm", "pcm16"] {
        assert_eq!(
            calculate_audio_length_ms(Some(spelling), &[0u8; 48000]),
            1000.0,
            "{spelling}"
        );
    }
}

#[test]
fn playback_tracker_measures_g711_with_a_typed_format() {
    let tracker = RealtimePlaybackTracker::new();
    tracker.set_audio_format("audio/pcmu");
    tracker.on_play_bytes("item_1", 0, &[0u8; 4000]);
    assert_eq!(tracker.state().elapsed_ms, Some(500.0));
}

#[test]
fn playback_tracker_on_play_bytes_and_state() {
    let tracker = RealtimePlaybackTracker::new();
    tracker.set_audio_format("pcm16");

    // 48k bytes -> (48000 / (24000 * 2)) * 1000 = 1_000 ms
    tracker.on_play_bytes("item1", 0, &[b'x'; 48000]);
    let state = tracker.state();
    assert_eq!(state.current_item_id.as_deref(), Some("item1"));
    approx(state.elapsed_ms.unwrap(), 1000.0);

    // A later report for the same item accumulates.
    tracker.on_play_ms("item1", 0, 500.0);
    approx(tracker.state().elapsed_ms.unwrap(), 1500.0);

    // Interruption clears the state.
    tracker.on_interrupted();
    let state = tracker.state();
    assert_eq!(state.current_item_id, None);
    assert_eq!(state.elapsed_ms, None);
}
