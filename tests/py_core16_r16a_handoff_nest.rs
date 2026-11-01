//! Ports of the odd-shape cases of `tests/test_extension_filters.py`: `nest_handoff_history`
//! summarizes items of an unknown type and messages whose content is not a string.

use openai_agents_sdk::handoffs::{HandoffInputData, nest_handoff_history};
use openai_agents_sdk::items::InputItem;
use serde_json::json;

fn data(history: Vec<serde_json::Value>) -> HandoffInputData {
    HandoffInputData {
        input_history: history.into_iter().map(InputItem).collect(),
        pre_handoff_items: vec![],
        new_items: vec![],
        input_items: None,
    }
}

fn summary(nested: &HandoffInputData) -> String {
    assert_eq!(nested.input_history.len(), 1);
    nested.input_history[0].0["content"]
        .as_str()
        .unwrap()
        .to_string()
}

/// Python: `test_nest_handoff_history_handles_unserializable_items` /
/// `test_nest_handoff_history_json_dumps_typeerror`: an item of an unknown type is summarized by
/// its type name rather than failing.
#[test]
fn an_item_of_an_unknown_type_is_summarized() {
    let nested = nest_handoff_history(
        data(vec![json!({"type": "custom_item", "field": "value"})]),
        None,
    );
    assert!(summary(&nested).contains("custom_item"));
}

/// Python: `test_nest_handoff_history_stringify_content_typeerror`: structured message content
/// is summarized under the speaker's role.
#[test]
fn structured_message_content_is_summarized_under_its_role() {
    let nested = nest_handoff_history(
        data(vec![
            json!({"role": "user", "content": {"complex": "object"}}),
        ]),
        None,
    );
    let text = summary(&nested);
    assert!(text.contains("user") || text.contains("object"), "{text}");
}
