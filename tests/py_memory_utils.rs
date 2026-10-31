//! Parity port of the behavioral tests in `tests/utils/test_pretty_print_and_items.py`.
//!
//! The Python tests build `ResponseOutputText.model_construct(text=None)` to prove the helpers
//! tolerate a null text; Rust's `OutputText.text` is a `String`, so a null text cannot exist.
//! The "normal text" halves are ported; `pretty_print_run_error_details` has no Rust
//! counterpart (reported as a cross-area gap).

use openai_agents_sdk::items::{ItemHelpers, MessageOutput, OutputItem, RunItem, RunItemKind};
use openai_agents_sdk::testing::assistant_message;

fn message_item(text: &str) -> RunItem {
    let OutputItem::Message(raw_item) = assistant_message(text) else {
        unreachable!("assistant_message builds a message")
    };
    RunItem::new("test", RunItemKind::MessageOutput { raw_item })
}

#[test]
fn text_message_output_returns_text_normally() {
    assert_eq!(
        ItemHelpers::text_message_output(&message_item("hello")),
        "hello"
    );
}

#[test]
fn text_message_outputs_concatenates_across_items() {
    let items = vec![message_item("hello "), message_item("world")];
    assert_eq!(ItemHelpers::text_message_outputs(&items), "hello world");
}

#[test]
fn text_message_output_of_an_empty_text_is_the_empty_string() {
    assert_eq!(ItemHelpers::text_message_output(&message_item("")), "");
    let items = vec![message_item(""), message_item("world")];
    assert_eq!(ItemHelpers::text_message_outputs(&items), "world");
}

#[test]
fn extract_last_content_returns_text_normally() {
    assert_eq!(
        ItemHelpers::extract_last_content(&assistant_message("hello")).unwrap(),
        "hello"
    );
}

#[test]
fn extract_last_content_of_an_empty_text_is_the_empty_string() {
    let OutputItem::Message(mut message) = assistant_message("") else {
        unreachable!()
    };
    message.content.clear();
    let empty: OutputItem = OutputItem::Message(MessageOutput { ..message });
    let result = ItemHelpers::extract_last_content(&empty).unwrap();
    assert_eq!(result, "");
}
