//! Ports of Python cases that read provider or snapshot payloads with absent values:
//! `tests/utils/test_pretty_print_and_items.py` (a message text part whose `text` is null) and
//! `tests/test_run_state.py::test_from_json_preserves_function_call_output_items`.

use openai_agents_sdk::InputItem;
use openai_agents_sdk::items::{ItemHelpers, MessageOutput, OutputItem, RunItem, RunItemKind};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use serde_json::json;

fn null_text_message(text: serde_json::Value) -> MessageOutput {
    serde_json::from_value(json!({
        "id": "msg_null", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }))
    .expect("a message whose text part has a null text still parses")
}

/// Python: `test_text_message_output_returns_empty_string_for_none_text`.
#[test]
fn a_null_text_part_reads_as_an_empty_string() {
    let item = RunItem::new(
        "test",
        RunItemKind::MessageOutput {
            raw_item: null_text_message(json!(null)),
        },
    );
    assert_eq!(ItemHelpers::text_message_output(&item), "");
}

/// Python: `test_text_message_outputs_handles_none_text_across_items`.
#[test]
fn a_null_text_part_does_not_break_the_concatenation_of_several_items() {
    let items = vec![
        RunItem::new(
            "test",
            RunItemKind::MessageOutput {
                raw_item: null_text_message(json!(null)),
            },
        ),
        RunItem::new(
            "test",
            RunItemKind::MessageOutput {
                raw_item: null_text_message(json!("world")),
            },
        ),
    ];
    assert_eq!(ItemHelpers::text_message_outputs(&items), "world");
}

/// Python: `test_extract_last_content_returns_empty_string_for_none_text`.
#[test]
fn extracting_the_last_content_of_a_null_text_part_gives_an_empty_string() {
    let message = OutputItem::Message(null_text_message(json!(null)));
    assert_eq!(ItemHelpers::extract_last_content(&message).unwrap(), "");
}

/// Python: `test_from_json_preserves_function_call_output_items`: a `function_call_output` in the
/// original input is kept as written, `name` and `status` included.
#[test]
fn function_call_output_items_in_the_original_input_are_preserved() {
    let wire = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": 0,
        "current_agent": {"name": "TestAgent"},
        "original_input": [{
            "type": "function_call_output", "call_id": "call123", "name": "test_tool",
            "status": "completed", "output": "result",
        }],
        "max_turns": 5,
        "context": {
            "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "approvals": {}, "context": {},
        },
        "generated_items": [],
        "model_responses": [],
    });
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.original_input.len(), 1);
    let item: &InputItem = &state.original_input[0];
    assert_eq!(item.0["type"], "function_call_output");
    assert_eq!(item.0["name"], "test_tool");
    assert_eq!(item.0["status"], "completed");
    let again = state.to_json().unwrap();
    assert_eq!(again["original_input"][0]["name"], "test_tool");
    assert_eq!(again["original_input"][0]["status"], "completed");
}
