//! Ports of cases of `tests/test_server_conversation_tracker.py` that carry a reasoning-id
//! policy: a reasoning item delivered without its id must not be sent again when it comes up
//! with the id.

use openai_agents_sdk::ServerConversationTracker;
use openai_agents_sdk::items::{InputItem, RunItem, RunItemKind};
use serde_json::json;

fn reasoning_run_item() -> RunItem {
    RunItem::new(
        "test",
        RunItemKind::Compaction {
            raw_item: json!({
                "type": "reasoning",
                "id": "rs_turn_input",
                "content": [{"type": "input_text", "text": "reasoning trace"}],
            }),
        },
    )
}

fn without_id() -> InputItem {
    InputItem(json!({
        "type": "reasoning",
        "content": [{"type": "input_text", "text": "reasoning trace"}],
    }))
}

/// `test_prepare_input_does_not_resend_reasoning_item_after_marking_omitted_id_as_sent`.
#[test]
fn a_reasoning_item_marked_sent_without_its_id_is_not_resent() {
    let mut tracker = ServerConversationTracker::default();
    tracker.set_omit_reasoning_item_ids(true);
    let generated = reasoning_run_item().to_input_item();
    assert!(tracker.needs_send(&generated));

    tracker.mark_input_as_sent(&[without_id()]);

    assert!(!tracker.needs_send(&generated));
}

/// Without the omit policy the delivered form keeps its id, and an id-less copy that was sent
/// does not stand in for the item with an id.
#[test]
fn the_id_is_part_of_the_item_when_the_policy_preserves_it() {
    let mut tracker = ServerConversationTracker::default();
    let generated = reasoning_run_item().to_input_item();

    tracker.mark_input_as_sent(&[without_id()]);

    assert!(tracker.needs_send(&generated));
}
