//! Port of `tests/test_items_helpers.py::test_extract_text_tolerates_none_text_content`: an
//! `output_text` part whose `text` is null (partial streaming output, provider gateways) must
//! not abort the turn; it contributes nothing to the extracted text.

use openai_agents_sdk::items::{ItemHelpers, MessageContent, MessageOutput, OutputItem};
use serde_json::json;

fn message(parts: Vec<MessageContent>) -> OutputItem {
    let mut message = MessageOutput::text("");
    message.content = parts;
    OutputItem::Message(message)
}

fn none_text() -> MessageContent {
    MessageContent::try_from(json!({
        "type": "output_text", "text": null, "annotations": [], "logprobs": []
    }))
    .expect("a null text part parses")
}

fn real_text() -> MessageContent {
    MessageContent::try_from(json!({
        "type": "output_text", "text": "hello", "annotations": [], "logprobs": []
    }))
    .expect("a text part parses")
}

#[test]
fn extract_text_tolerates_none_text_content() {
    // A single null-text part concatenates to "", which is reported as no text.
    assert_eq!(ItemHelpers::extract_text(&message(vec![none_text()])), None);
    // Mixed content keeps the real text in either order.
    assert_eq!(
        ItemHelpers::extract_text(&message(vec![real_text(), none_text()])).as_deref(),
        Some("hello")
    );
    assert_eq!(
        ItemHelpers::extract_text(&message(vec![none_text(), real_text()])).as_deref(),
        Some("hello")
    );
}
