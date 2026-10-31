//! Parity port of `tests/models/test_extended_thinking_message_order.py`.
//!
//! Python exercises `LitellmModel._fix_tool_message_ordering`; the Rust counterpart is the
//! free function `extensions::models::fix_tool_message_ordering`, so every case translates
//! one-to-one with the same inputs and the same expected message sequence.

use openai_agents_sdk::extensions::models::fix_tool_message_ordering;
use serde_json::{Value, json};

/// `{"id": …, "type": "function", "function": {"name": …, "arguments": "{}"}}`.
fn call(id: &str, name: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}})
}

fn assistant_calls(calls: Vec<Value>) -> Value {
    json!({"role": "assistant", "tool_calls": calls})
}

fn tool_result(id: &str, content: &str) -> Value {
    json!({"role": "tool", "tool_call_id": id, "content": content})
}

fn role(message: &Value) -> &str {
    message["role"].as_str().unwrap()
}

fn tool_call_ids(message: &Value) -> Vec<&str> {
    message["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect()
}

fn tool_call_id(message: &Value) -> &str {
    message["tool_call_id"].as_str().unwrap()
}

#[test]
fn basic_reordering_tool_result_before_call() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        tool_result("call_123", "Result for call_123"),
        assistant_calls(vec![call("call_123", "test")]),
        json!({"role": "user", "content": "Thanks"}),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 4);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(tool_call_ids(&result[1]), ["call_123"]);
    assert_eq!(role(&result[2]), "tool");
    assert_eq!(tool_call_id(&result[2]), "call_123");
    assert_eq!(role(&result[3]), "user");
}

#[test]
fn consecutive_tool_calls_get_separated() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        assistant_calls(vec![call("call_1", "test1")]),
        assistant_calls(vec![call("call_2", "test2")]),
        tool_result("call_1", "Result 1"),
        tool_result("call_2", "Result 2"),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 5);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(tool_call_ids(&result[1]), ["call_1"]);
    assert_eq!(role(&result[2]), "tool");
    assert_eq!(tool_call_id(&result[2]), "call_1");
    assert_eq!(role(&result[3]), "assistant");
    assert_eq!(tool_call_ids(&result[3]), ["call_2"]);
    assert_eq!(role(&result[4]), "tool");
    assert_eq!(tool_call_id(&result[4]), "call_2");
}

#[test]
fn unmatched_tool_results_preserved() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        assistant_calls(vec![call("call_1", "test")]),
        tool_result("call_1", "Matched result"),
        tool_result("call_orphan", "Orphaned result"),
        json!({"role": "user", "content": "End"}),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 5);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(role(&result[2]), "tool");
    assert_eq!(tool_call_id(&result[2]), "call_1");
    assert_eq!(role(&result[3]), "tool");
    assert_eq!(tool_call_id(&result[3]), "call_orphan");
    assert_eq!(role(&result[4]), "user");
}

#[test]
fn tool_calls_without_results_preserved() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        assistant_calls(vec![call("call_1", "test")]),
        json!({"role": "user", "content": "End"}),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 3);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(tool_call_ids(&result[1]), ["call_1"]);
    assert_eq!(role(&result[2]), "user");
}

#[test]
fn correctly_ordered_messages_unchanged() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        assistant_calls(vec![call("call_1", "test")]),
        tool_result("call_1", "Result"),
        json!({"role": "assistant", "content": "Done"}),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 4);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(tool_call_ids(&result[1]), ["call_1"]);
    assert_eq!(role(&result[2]), "tool");
    assert_eq!(tool_call_id(&result[2]), "call_1");
    assert_eq!(role(&result[3]), "assistant");
}

#[test]
fn multiple_tool_calls_single_message() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        assistant_calls(vec![call("call_1", "test1"), call("call_2", "test2")]),
        tool_result("call_1", "Result 1"),
        tool_result("call_2", "Result 2"),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 5);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(tool_call_ids(&result[1]), ["call_1"]);
    assert_eq!(role(&result[2]), "tool");
    assert_eq!(tool_call_id(&result[2]), "call_1");
    assert_eq!(role(&result[3]), "assistant");
    assert_eq!(tool_call_ids(&result[3]), ["call_2"]);
    assert_eq!(role(&result[4]), "tool");
    assert_eq!(tool_call_id(&result[4]), "call_2");
}

/// Anthropic rejects a request that repeats the same signed thinking block, so only the first
/// split keeps the shared assistant text, thinking blocks and reasoning content.
#[test]
fn split_does_not_duplicate_content_or_thinking() {
    let assistant = json!({
        "role": "assistant",
        "content": "Looking up both queries.",
        "thinking_blocks": [{"type": "thinking", "thinking": "plan", "signature": "sig_abc"}],
        "reasoning_content": "internal plan",
        "tool_calls": [call("call_1", "s"), call("call_2", "s")],
    });
    let messages = vec![
        json!({"role": "user", "content": "Search both"}),
        assistant,
        tool_result("call_1", "ok1"),
        tool_result("call_2", "ok2"),
    ];

    let result = fix_tool_message_ordering(messages);

    let assistants: Vec<&Value> = result.iter().filter(|m| role(m) == "assistant").collect();
    assert_eq!(assistants.len(), 2);
    assert_eq!(assistants[0]["content"], "Looking up both queries.");
    assert!(assistants[0].get("thinking_blocks").is_some());
    assert!(assistants[0].get("reasoning_content").is_some());
    assert!(assistants[1].get("content").is_none());
    assert!(assistants[1].get("thinking_blocks").is_none());
    assert!(assistants[1].get("reasoning_content").is_none());
    assert_eq!(tool_call_ids(assistants[0]), ["call_1"]);
    assert_eq!(tool_call_ids(assistants[1]), ["call_2"]);
}

#[test]
fn empty_messages_list() {
    assert_eq!(fix_tool_message_ordering(Vec::new()), Vec::<Value>::new());
}

#[test]
fn no_tool_messages() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant", "content": "Hi there"}),
        json!({"role": "user", "content": "How are you?"}),
    ];

    assert_eq!(fix_tool_message_ordering(messages.clone()), messages);
}

#[test]
fn complex_mixed_scenario() {
    let messages = vec![
        json!({"role": "user", "content": "Start"}),
        tool_result("call_out_of_order", "Out of order result"),
        json!({"role": "assistant", "content": "Regular response"}),
        assistant_calls(vec![call("call_out_of_order", "test")]),
        assistant_calls(vec![call("call_normal", "test2")]),
        tool_result("call_normal", "Normal result"),
        tool_result("call_orphan", "Orphaned result"),
        json!({"role": "user", "content": "End"}),
    ];

    let result = fix_tool_message_ordering(messages);

    assert_eq!(result.len(), 8);
    assert_eq!(role(&result[0]), "user");
    assert_eq!(role(&result[1]), "assistant");
    assert_eq!(result[1]["content"], "Regular response");
    assert_eq!(role(&result[2]), "assistant");
    assert_eq!(tool_call_ids(&result[2]), ["call_out_of_order"]);
    assert_eq!(role(&result[3]), "tool");
    assert_eq!(tool_call_id(&result[3]), "call_out_of_order");
    assert_eq!(role(&result[4]), "assistant");
    assert_eq!(tool_call_ids(&result[4]), ["call_normal"]);
    assert_eq!(role(&result[5]), "tool");
    assert_eq!(tool_call_id(&result[5]), "call_normal");
    assert_eq!(role(&result[6]), "tool");
    assert_eq!(tool_call_id(&result[6]), "call_orphan");
    assert_eq!(role(&result[7]), "user");
}
