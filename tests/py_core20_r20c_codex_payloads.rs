//! Ports of `tests/extensions/experiemental/codex/test_payloads.py`: Python's thread items answer
//! dictionary-style access by field name (with the `type` tag included) and raise `KeyError` for
//! an unknown field. The Rust item's mapping view is its raw JSON form, [`ThreadItem::to_value`],
//! which must expose the same keys and nothing else.

use openai_agents_sdk::extensions::codex::ThreadItem;
use serde_json::Value;

fn agent_message() -> ThreadItem {
    ThreadItem::AgentMessage {
        id: "item-1".to_string(),
        text: "hello".to_string(),
    }
}

/// Python: `test_dict_like_supports_mapping_access_for_dataclass_fields`.
#[test]
fn an_agent_message_is_readable_by_field_name() {
    let value = agent_message().to_value();
    let fields = value.as_object().expect("an item is a JSON object");

    assert_eq!(value["id"], "item-1");
    assert_eq!(value["text"], "hello");
    assert_eq!(value["type"], "agent_message");
    assert_eq!(value.get("text").and_then(Value::as_str), Some("hello"));
    assert!(fields.contains_key("id"));
    assert!(!fields.contains_key("missing"));
    let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["id", "text", "type"]);

    // The mapping view parses back to the same item.
    assert_eq!(
        ThreadItem::from_value(value).expect("parse"),
        agent_message()
    );
}

/// Python: `test_dict_like_raises_key_error_for_unknown_fields`.
#[test]
fn an_unknown_field_is_absent_from_the_mapping_view() {
    let value = agent_message().to_value();

    assert!(value.get("missing").is_none());
    assert!(value["missing"].is_null());
}
