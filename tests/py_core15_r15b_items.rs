//! Ports of item replay cases of the Python SDK's `tests/test_items_helpers.py`.

use openai_agents_sdk::items::{RunItem, RunItemKind};
use serde_json::{Value, json};

fn output_item(raw_item: Value, output: Value) -> RunItem {
    RunItem::new(
        "A",
        RunItemKind::ToolCallOutput {
            raw_item,
            output,
            custom_data: None,
        },
    )
}

/// `test_dict_shell_call_output_item_to_input_item_sanitizes_without_mutation`: replaying a
/// shell output keeps only the input schema's fields, at the item level and in each chunk, and
/// leaves the item's own raw value untouched.
#[test]
fn a_shell_call_output_replays_without_output_only_fields_and_is_not_mutated() {
    let raw_item = json!({
        "type": "shell_call_output",
        "call_id": "call_1",
        "status": "completed",
        "shell_output": "legacy",
        "provider_data": {"provider": "value"},
        "created_by": "server",
        "output": [{
            "stdout": "ok",
            "stderr": "",
            "outcome": {"type": "exit", "exit_code": 0},
            "created_by": "server",
        }],
    });
    let item = output_item(raw_item.clone(), json!("ok"));

    let replayed = item.to_input_item().0;

    assert_eq!(
        replayed,
        json!({
            "type": "shell_call_output",
            "call_id": "call_1",
            "output": [{
                "stdout": "ok",
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": 0},
            }],
        })
    );
    match &item.kind {
        RunItemKind::ToolCallOutput { raw_item: kept, .. } => assert_eq!(kept, &raw_item),
        other => panic!("{other:?}"),
    }
}

/// `test_handoff_output_item_stringifies_object_output`: an object-valued output of a handoff
/// output item is replayed as it is, not turned into text.
#[test]
fn an_object_valued_function_call_output_replays_unchanged() {
    let raw_item = json!({
        "type": "function_call_output",
        "call_id": "call-obj",
        "output": {"assistant": "Weather Assistant"},
    });
    let item = output_item(raw_item, json!({"assistant": "Weather Assistant"}));

    let replayed = item.to_input_item().0;

    assert_eq!(replayed["type"], "function_call_output");
    assert_eq!(replayed["call_id"], "call-obj");
    assert_eq!(
        replayed["output"],
        json!({"assistant": "Weather Assistant"})
    );
}
