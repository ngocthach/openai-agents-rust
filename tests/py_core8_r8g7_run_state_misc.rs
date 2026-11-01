//! Ports of `tests/test_run_state.py` cases that read a snapshot through the public `RunState`
//! JSON surface: model response messages without a `status` and the warning a snapshot whose
//! context needs explicit restoration logs.

#![allow(clippy::field_reassign_with_default)]

use openai_agents_sdk::InputItem;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use serde_json::json;

fn plain_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".to_string(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
}

/// Python: `test_round_trip_deserializes_statusless_model_response_messages`: a model response
/// message the provider sent without a `status` does not gain one across a round trip.
#[test]
fn a_statusless_model_response_message_round_trips_unchanged() {
    let message = json!({
        "id": "msg_response", "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "world", "annotations": []}],
    });
    let mut wire = plain_state().to_json().unwrap();
    wire["model_responses"] = json!([{
        "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "output": [message.clone()], "response_id": null,
    }]);
    let restored = RunState::from_json(wire).unwrap();
    let again = restored.to_json().unwrap();
    let output = &again["model_responses"][0]["output"][0];
    assert!(output.get("status").is_none(), "{output}");
    assert_eq!(output["id"], "msg_response");
    assert_eq!(output["content"][0]["text"], "world");
}

// ---------------------------------------------------------------------------
// Context restoration warning
// ---------------------------------------------------------------------------

static RECORDS: std::sync::Mutex<Vec<(log::Level, String)>> = std::sync::Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap()
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

/// Python: `test_from_json_warns_without_deserializer`: reading a snapshot whose context was
/// serialized from a custom type warns that it needs explicit restoration; the read still
/// succeeds and reports the need.
#[test]
fn from_json_warns_when_the_context_needs_restoration() {
    let _ = log::set_logger(&Capture);
    log::set_max_level(log::LevelFilter::Trace);
    let mut wire = plain_state().to_json().unwrap();
    wire["context"]["context"] = json!({"value": "hello"});
    wire["context"]["context_meta"] = json!({
        "original_type": "dataclass", "serialized_via": "asdict",
        "requires_deserializer": true, "omitted": false,
    });
    RECORDS.lock().unwrap().clear();
    let state = RunState::from_json(wire).unwrap();
    assert!(state.context_requires_restoration);
    let records = RECORDS.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|(level, text)| *level == log::Level::Warn && text.contains("context override")),
        "{records:?}"
    );
}
