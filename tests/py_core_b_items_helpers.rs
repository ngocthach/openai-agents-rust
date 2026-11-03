//! Ported from the Python SDK's `tests/test_items_helpers.py`.

use openai_agents_sdk::items::{
    FunctionCall, InputItem, ItemHelpers, MessageContent, MessageOutput, ModelResponse, OutputItem,
    ReasoningOutput, RunInput, RunItem, RunItemKind,
};
use openai_agents_sdk::tool::ToolOutput;
use openai_agents_sdk::usage::Usage;
use serde_json::json;

fn make_message(content: Vec<MessageContent>) -> MessageOutput {
    let mut message = MessageOutput::text("");
    message.id = "msg123".to_string();
    message.status = Some("completed".to_string());
    message.content = content;
    message
}

fn text(text: &str) -> MessageContent {
    MessageContent::OutputText {
        text: text.to_string(),
        annotations: Vec::new(),
        logprobs: None,
    }
}

fn refusal(refusal: &str) -> MessageContent {
    MessageContent::Refusal {
        refusal: refusal.to_string(),
    }
}

fn response(output: Vec<OutputItem>) -> ModelResponse {
    ModelResponse {
        output,
        usage: Usage::default(),
        response_id: None,
        request_id: None,
        raw_usage: None,
    }
}

fn run_item(kind: RunItemKind) -> RunItem {
    RunItem::new("test", kind)
}

#[test]
fn extract_last_content_of_text_message() {
    let message = OutputItem::Message(make_message(vec![text("Hello "), text("world!")]));
    assert_eq!(
        ItemHelpers::extract_last_content(&message).unwrap(),
        "world!"
    );
}

#[test]
fn extract_last_content_of_refusal_message() {
    let message = OutputItem::Message(make_message(vec![
        text("Before refusal"),
        refusal("I cannot do that"),
    ]));
    assert_eq!(
        ItemHelpers::extract_last_content(&message).unwrap(),
        "I cannot do that"
    );
}

#[test]
fn extract_last_content_non_message_returns_empty() {
    let tool_call = OutputItem::FunctionCall(FunctionCall::new("func", "{}"));
    assert_eq!(ItemHelpers::extract_last_content(&tool_call).unwrap(), "");
}

#[test]
fn extract_last_content_rejects_unknown_content_type() {
    let message = OutputItem::Message(make_message(vec![MessageContent::Other(
        json!({"type": "mystery"}),
    )]));
    let err = ItemHelpers::extract_last_content(&message).unwrap_err();
    assert!(err.to_string().contains("Unexpected content type"), "{err}");
}

#[test]
fn extract_last_text_returns_text_only() {
    let message = OutputItem::Message(make_message(vec![text("part1"), text("part2")]));
    assert_eq!(
        ItemHelpers::extract_last_text(&message).as_deref(),
        Some("part2")
    );

    let message2 = OutputItem::Message(make_message(vec![text("part1"), refusal("no")]));
    assert_eq!(ItemHelpers::extract_last_text(&message2), None);
}

#[test]
fn extract_text_concatenates_all_text_segments() {
    let message = OutputItem::Message(make_message(vec![
        text("part1"),
        refusal("no"),
        text("part2"),
    ]));
    assert_eq!(
        ItemHelpers::extract_text(&message).as_deref(),
        Some("part1part2")
    );
    assert_eq!(
        ItemHelpers::extract_text(&OutputItem::FunctionCall(FunctionCall::new("func", "{}"))),
        None
    );
}

#[test]
fn extract_text_is_none_for_empty_message() {
    let message = OutputItem::Message(make_message(Vec::new()));
    assert_eq!(ItemHelpers::extract_text(&message), None);
}

#[test]
fn input_to_new_input_list_from_string() {
    let items = ItemHelpers::input_to_new_input_list(&RunInput::from("hi"));
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].0, json!({"content": "hi", "role": "user"}));
}

#[test]
fn input_to_new_input_list_copies_lists() {
    let original = vec![InputItem(json!({"content": "abc", "role": "developer"}))];
    let mut copied = ItemHelpers::input_to_new_input_list(&RunInput::from(original.clone()));
    assert_eq!(copied, original);
    copied.pop();
    assert_eq!(original[0].0["content"], "abc");
}

#[test]
fn text_message_output_concatenates_text_segments() {
    let item = run_item(RunItemKind::MessageOutput {
        raw_item: make_message(vec![text("a"), refusal("denied"), text("b")]),
    });
    assert_eq!(ItemHelpers::text_message_output(&item), "ab");
}

#[test]
fn text_message_outputs_across_list_of_run_items() {
    let item1 = run_item(RunItemKind::MessageOutput {
        raw_item: make_message(vec![text("foo")]),
    });
    let item2 = run_item(RunItemKind::MessageOutput {
        raw_item: make_message(vec![text("bar")]),
    });
    let reasoning = run_item(RunItemKind::Reasoning {
        raw_item: serde_json::from_value::<ReasoningOutput>(json!({
            "id": "rid", "summary": []
        }))
        .unwrap(),
    });
    assert_eq!(
        ItemHelpers::text_message_outputs(&[item1, reasoning, item2]),
        "foobar"
    );
}

#[test]
fn handoff_output_item_converts_protocol_payload() {
    let item = run_item(RunItemKind::HandoffOutput {
        raw_item: InputItem(json!({
            "type": "function_call_output", "call_id": "call-123", "output": "ok"
        })),
        source_agent: "source".into(),
        target_agent: "target".into(),
    });
    let converted = item.to_input_item();
    assert_eq!(converted.0["type"], "function_call_output");
    assert_eq!(converted.0["call_id"], "call-123");
    assert_eq!(converted.0["output"], "ok");
}

#[test]
fn handoff_output_item_keeps_object_output() {
    let item = run_item(RunItemKind::HandoffOutput {
        raw_item: InputItem(json!({
            "type": "function_call_output",
            "call_id": "call-obj",
            "output": {"assistant": "Weather Assistant"},
        })),
        source_agent: "source".into(),
        target_agent: "target".into(),
    });
    let converted = item.to_input_item();
    assert_eq!(converted.0["call_id"], "call-obj");
    assert_eq!(
        converted.0["output"],
        json!({"assistant": "Weather Assistant"})
    );
}

#[test]
fn tool_call_output_item_preserves_function_output_structure() {
    let raw = json!({
        "type": "function_call_output",
        "call_id": "call-keep",
        "output": [{"type": "output_text", "text": "value"}],
    });
    let item = run_item(RunItemKind::ToolCallOutput {
        raw_item: raw.clone(),
        output: json!("value"),
        custom_data: None,
    });
    let payload = item.to_input_item();
    assert_eq!(payload.0["type"], "function_call_output");
    assert_eq!(payload.0["output"], raw["output"]);
}

#[test]
fn tool_call_output_item_constructs_function_call_output_dict() {
    let mut call = FunctionCall::new("do_something", r#"{"x": 1}"#);
    call.call_id = "call-abc".into();
    call.id = Some("call-abc".into());
    let payload =
        ItemHelpers::tool_call_output_item(&call, &ToolOutput::Text("result-string".into()));
    assert_eq!(payload["type"], "function_call_output");
    assert_eq!(payload["call_id"], "call-abc");
    assert_eq!(payload["output"], "result-string");
}

#[test]
fn to_input_items_for_message() {
    let mut message = MessageOutput::text("hello world");
    message.id = "m1".into();
    message.status = Some("completed".into());
    let items = response(vec![OutputItem::Message(message)]).to_input_items();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].0["id"], "m1");
    assert_eq!(items[0].0["role"], "assistant");
    assert_eq!(items[0].0["status"], "completed");
    assert_eq!(items[0].0["type"], "message");
    assert_eq!(items[0].0["content"][0]["type"], "output_text");
    assert_eq!(items[0].0["content"][0]["text"], "hello world");
}

#[test]
fn to_input_items_for_function_call() {
    let mut call = FunctionCall::new("func", "{}");
    call.id = Some("f1".into());
    call.call_id = "c1".into();
    let items = response(vec![OutputItem::FunctionCall(call)]).to_input_items();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].0["id"], "f1");
    assert_eq!(items[0].0["arguments"], "{}");
    assert_eq!(items[0].0["call_id"], "c1");
    assert_eq!(items[0].0["name"], "func");
    assert_eq!(items[0].0["type"], "function_call");
}

#[test]
fn to_input_items_for_hosted_calls_round_trip_unchanged() {
    let cases = vec![
        json!({"id": "fs1", "queries": ["query"], "status": "completed", "type": "file_search_call"}),
        json!({
            "id": "w1", "status": "completed", "type": "web_search_call",
            "action": {"type": "search", "query": "query"}
        }),
        json!({
            "id": "comp1", "type": "computer_call", "action": {"type": "screenshot"},
            "call_id": "comp1", "pending_safety_checks": [], "status": "completed"
        }),
        json!({
            "id": "comp2", "type": "computer_call",
            "actions": [
                {"type": "click", "x": 3, "y": 4, "button": "left"},
                {"type": "type", "text": "hello"},
            ],
            "call_id": "comp2", "pending_safety_checks": [], "status": "completed"
        }),
        json!({"id": "rid1", "summary": [{"text": "why", "type": "summary_text"}], "type": "reasoning"}),
    ];
    for case in cases {
        let items = response(vec![OutputItem::from_value(case.clone())]).to_input_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, case, "round-trip changed {case}");
    }
}

#[test]
fn to_input_items_for_tool_search_strips_created_by() {
    let items = response(vec![
        OutputItem::from_value(json!({
            "id": "tsc_123", "call_id": "call_tsc_123", "arguments": {"query": "profile"},
            "execution": "server", "status": "completed", "type": "tool_search_call",
            "created_by": "server",
        })),
        OutputItem::from_value(json!({
            "id": "tso_123", "call_id": "call_tsc_123", "execution": "server",
            "status": "completed", "tools": [], "type": "tool_search_output",
            "created_by": "server",
        })),
    ])
    .to_input_items();

    assert_eq!(
        items.iter().map(|i| i.0.clone()).collect::<Vec<_>>(),
        vec![
            json!({
                "id": "tsc_123", "call_id": "call_tsc_123", "arguments": {"query": "profile"},
                "execution": "server", "status": "completed", "type": "tool_search_call",
            }),
            json!({
                "id": "tso_123", "call_id": "call_tsc_123", "execution": "server",
                "status": "completed", "tools": [], "type": "tool_search_output",
            }),
        ]
    );
}

#[test]
fn to_input_items_strips_created_by_for_non_tool_search_items() {
    let items = response(vec![
        OutputItem::from_value(json!({
            "id": "apc_1", "call_id": "call_1", "type": "apply_patch_call", "status": "completed",
            "operation": {"type": "delete_file", "path": "foo.py"}, "created_by": "program_1",
        })),
        OutputItem::from_value(json!({
            "id": "fco_1", "call_id": "call_1", "type": "function_call_output",
            "output": "done", "status": "completed", "created_by": "program_1",
        })),
    ])
    .to_input_items();

    assert_eq!(
        items.iter().map(|i| i.0.clone()).collect::<Vec<_>>(),
        vec![
            json!({
                "id": "apc_1", "call_id": "call_1",
                "operation": {"path": "foo.py", "type": "delete_file"},
                "status": "completed", "type": "apply_patch_call",
            }),
            json!({
                "id": "fco_1", "call_id": "call_1", "output": "done",
                "status": "completed", "type": "function_call_output",
            }),
        ]
    );
    assert!(items.iter().all(|i| i.0.get("created_by").is_none()));
}

#[test]
fn tool_call_item_to_input_item_strips_created_by() {
    let raw = json!({
        "id": "fc_1", "arguments": "{}", "call_id": "call_1", "name": "lookup",
        "type": "function_call", "created_by": "server",
    });
    let item = run_item(RunItemKind::ToolCall {
        raw_item: raw.clone(),
    });
    let replayed = item.to_input_item();
    assert_eq!(replayed.0["type"], "function_call");
    assert!(replayed.0.get("created_by").is_none());
    // The original item is untouched.
    assert_eq!(raw["created_by"], "server");
}

#[test]
fn tool_call_output_item_to_input_item_strips_created_by() {
    let raw = json!({
        "type": "function_call_output", "call_id": "call_1", "output": "ok",
        "created_by": "server",
    });
    let item = run_item(RunItemKind::ToolCallOutput {
        raw_item: raw.clone(),
        output: json!("ok"),
        custom_data: None,
    });
    let replayed = item.to_input_item();
    assert_eq!(replayed.0["type"], "function_call_output");
    assert!(replayed.0.get("created_by").is_none());
    assert_eq!(raw["created_by"], "server");
}

#[test]
fn shell_call_output_item_to_input_item_sanitizes_without_mutation() {
    let raw = json!({
        "type": "shell_call_output",
        "call_id": "call_1",
        "status": "completed",
        "shell_output": "legacy",
        "provider_data": {"provider": "value"},
        "created_by": "server",
        "output": [{
            "stdout": "ok", "stderr": "",
            "outcome": {"type": "exit", "exit_code": 0},
            "created_by": "server",
        }],
    });
    let item = run_item(RunItemKind::ToolCallOutput {
        raw_item: raw.clone(),
        output: json!("ok"),
        custom_data: None,
    });

    let replayed = item.to_input_item();
    assert_eq!(
        replayed.0,
        json!({
            "type": "shell_call_output",
            "call_id": "call_1",
            "output": [{
                "stdout": "ok", "stderr": "",
                "outcome": {"type": "exit", "exit_code": 0},
            }],
        })
    );
    // The caller's original mapping (and its nested chunk) is untouched.
    assert_eq!(raw["created_by"], "server");
    assert_eq!(raw["output"][0]["created_by"], "server");
}

#[test]
fn to_input_items_strips_nested_created_by_from_shell_call_output() {
    let raw = json!({
        "id": "sco_1", "call_id": "call_1", "type": "shell_call_output", "status": "completed",
        "created_by": "program_top",
        "output": [{
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "hi", "stderr": "", "created_by": "program_chunk",
        }],
    });
    let items = response(vec![OutputItem::from_value(raw.clone())]).to_input_items();

    assert_eq!(
        items[0].0,
        json!({
            "id": "sco_1", "call_id": "call_1", "type": "shell_call_output",
            "status": "completed",
            "output": [{
                "outcome": {"type": "exit", "exit_code": 0},
                "stdout": "hi", "stderr": "",
            }],
        })
    );
    assert_eq!(raw["created_by"], "program_top");
    assert_eq!(raw["output"][0]["created_by"], "program_chunk");
}
