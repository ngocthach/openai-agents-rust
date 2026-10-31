//! Ports of `TestHandoffHistoryDuplicationFix` and the summary-parsing cases of the Python SDK's
//! `tests/test_handoff_history_duplication.py` (issue 2171: tool items must not be duplicated
//! between the summary and the forwarded input, and earlier summaries must flatten back into
//! the turns they recorded).
//!
//! Python inspects `HandoffInputData` fields that are tuples and dataclasses. Here the same
//! checks read `pre_handoff_items`, `new_items`, `input_items` and `input_history` directly. The
//! Python summary parser (`_extract_nested_history_transcript`) is private, so those cases read
//! the transcript that a custom history mapper receives, which is the same flattened list.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::handoffs::{HandoffHistoryMapper, HandoffInputData, nest_handoff_history};
use openai_agents_sdk::items::{
    FunctionCall, InputItem, MessageOutput, ReasoningOutput, RunItem, RunItemKind,
};
use serde_json::{Value, json};

fn tool_call(agent: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::ToolCall {
            raw_item: json!({
                "type": "function_call",
                "id": "call_tool_123",
                "call_id": "call_tool_123",
                "name": "get_weather",
                "arguments": "{\"city\": \"London\"}",
            }),
        },
    )
}

fn tool_output(agent: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::ToolCallOutput {
            raw_item: json!({
                "type": "function_call_output",
                "call_id": "call_tool_123",
                "output": "Sunny, 22°C",
            }),
            output: json!("Sunny, 22°C"),
            custom_data: None,
        },
    )
}

fn handoff_call(agent: &str) -> RunItem {
    let raw_item: FunctionCall = serde_json::from_value(json!({
        "id": "call_handoff_456",
        "call_id": "call_handoff_456",
        "name": "transfer_to_agent_b",
        "arguments": "{}",
    }))
    .unwrap();
    RunItem::new(agent, RunItemKind::HandoffCall { raw_item })
}

fn handoff_output(agent: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::HandoffOutput {
            raw_item: InputItem(json!({
                "type": "function_call_output",
                "call_id": "call_handoff_456",
                "output": "{\"assistant\": \"agent_b\"}",
            })),
            source_agent: agent.to_string(),
            target_agent: agent.to_string(),
        },
    )
}

fn message(agent: &str, text: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::MessageOutput {
            raw_item: MessageOutput::text(text),
        },
    )
}

fn reasoning(agent: &str) -> RunItem {
    let raw_item: ReasoningOutput = serde_json::from_value(json!({
        "id": "reasoning_123",
        "summary": [{"type": "summary_text", "text": "Thinking about handoff"}],
    }))
    .unwrap();
    RunItem::new(agent, RunItemKind::Reasoning { raw_item })
}

fn approval(agent: &str) -> RunItem {
    let raw_item: FunctionCall = serde_json::from_value(json!({
        "call_id": "call_tool_approve",
        "name": "needs_approval",
        "arguments": "{}",
    }))
    .unwrap();
    RunItem::new(
        agent,
        RunItemKind::ToolApproval {
            raw_item,
            tool_name: "needs_approval".to_string(),
        },
    )
}

/// An assistant summary message whose content is the given lines joined by newlines.
fn summary_item(lines: &[&str]) -> InputItem {
    InputItem(json!({"role": "assistant", "content": lines.join("\n")}))
}

fn user(text: &str) -> InputItem {
    InputItem(json!({"role": "user", "content": text}))
}

fn data(history: Vec<InputItem>, pre: Vec<RunItem>, new: Vec<RunItem>) -> HandoffInputData {
    HandoffInputData {
        input_history: history,
        pre_handoff_items: pre,
        new_items: new,
        input_items: None,
    }
}

fn text(item: &InputItem) -> String {
    item.text_content().unwrap_or_else(|| item.0.to_string())
}

fn types(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|i| i.0["type"].as_str().map(str::to_string))
        .collect()
}

/// A history mapper that records the transcript it is handed and returns it unchanged.
fn capturing_mapper() -> (HandoffHistoryMapper, Arc<Mutex<Vec<InputItem>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let mapper: HandoffHistoryMapper = Arc::new(move |transcript: Vec<InputItem>| {
        sink.lock().unwrap().extend(transcript.iter().cloned());
        transcript
    });
    (mapper, captured)
}

/// The transcript a nested handoff flattens its input back into.
fn flattened_transcript(input: Vec<InputItem>) -> Vec<InputItem> {
    let (mapper, captured) = capturing_mapper();
    nest_handoff_history(data(input, Vec::new(), Vec::new()), Some(&mapper));
    captured.lock().unwrap().clone()
}

// ----------------------------------------------------------------------------- the fix itself

/// Python: `test_pre_handoff_tool_items_are_filtered`.
#[test]
fn pre_handoff_tool_items_are_filtered() {
    let nested = nest_handoff_history(
        data(
            vec![user("Hello")],
            vec![tool_call("A"), tool_output("A")],
            Vec::new(),
        ),
        None,
    );
    assert!(nested.pre_handoff_items.is_empty());
    assert_eq!(nested.input_history.len(), 1);
    assert!(text(&nested.input_history[0]).contains("<CONVERSATION HISTORY>"));
}

/// Python: `test_tool_approval_items_are_skipped`.
#[test]
fn tool_approval_items_are_skipped() {
    let nested = nest_handoff_history(
        data(vec![user("Hello")], vec![approval("A")], Vec::new()),
        None,
    );
    assert!(nested.pre_handoff_items.is_empty());
    assert_eq!(nested.input_items, Some(Vec::new()));
}

/// Python: `test_pre_handoff_reasoning_items_are_filtered`.
#[test]
fn pre_handoff_reasoning_items_are_filtered() {
    let nested = nest_handoff_history(
        data(vec![user("Hello")], vec![reasoning("A")], Vec::new()),
        None,
    );
    assert!(nested.pre_handoff_items.is_empty());
    assert!(text(&nested.input_history[0]).contains("reasoning"));
}

/// Python: `test_new_items_handoff_output_is_filtered_for_input`. `new_items` keeps both items
/// for the session, but the model input carries no `function_call_output`.
#[test]
fn new_items_handoff_output_is_filtered_for_input() {
    let nested = nest_handoff_history(
        data(
            vec![user("Hello")],
            Vec::new(),
            vec![handoff_call("A"), handoff_output("A")],
        ),
        None,
    );
    assert_eq!(nested.new_items.len(), 2);
    assert!(nested.input_items.is_some());
    assert!(!types(&nested.input_history).contains(&"function_call_output".to_string()));
}

/// Python: `test_message_items_are_preserved_in_new_items`.
#[test]
fn message_items_are_preserved_in_new_items() {
    let nested = nest_handoff_history(
        data(
            vec![user("Hello")],
            Vec::new(),
            vec![message("A", "Hello!")],
        ),
        None,
    );
    assert_eq!(nested.new_items.len(), 1);
    assert_eq!(nested.input_items, Some(Vec::new()));
    assert_eq!(nested.input_history.len(), 2);
    assert_eq!(nested.input_history[1].0["role"], "assistant");
    assert!(nested.input_history[1].0.to_string().contains("Hello!"));
}

/// Python: `test_forwarded_items_are_excluded_from_summary_until_the_next_handoff`.
#[test]
fn forwarded_items_are_excluded_from_summary_until_the_next_handoff() {
    let first = nest_handoff_history(
        data(
            vec![user("Hello")],
            Vec::new(),
            vec![message("A", "Hello!")],
        ),
        None,
    );
    assert!(!text(&first.input_history[0]).contains("Hello!"));
    assert_eq!(first.input_items, Some(Vec::new()));
    assert_eq!(first.input_history.len(), 2);
    assert_eq!(first.input_history[1].0["role"], "assistant");

    let second = nest_handoff_history(
        data(first.input_history.clone(), Vec::new(), Vec::new()),
        None,
    );
    let summary = text(&second.input_history[0]);
    assert_eq!(summary.matches("Hello!").count(), 1);
    assert!(second.pre_handoff_items.is_empty());
}

/// Python: `test_custom_mapper_receives_clean_flattened_history`. The mapper of a later handoff
/// sees the earlier turns as the plain items they were, not as a summary wrapped in a summary.
#[test]
fn custom_mapper_receives_clean_flattened_history() {
    let first = nest_handoff_history(
        data(
            vec![user("Hello")],
            Vec::new(),
            vec![message("A", "Hello!")],
        ),
        None,
    );
    let (mapper, captured) = capturing_mapper();
    let second = nest_handoff_history(
        data(first.input_history.clone(), Vec::new(), Vec::new()),
        Some(&mapper),
    );

    let captured = captured.lock().unwrap().clone();
    assert_eq!(captured.len(), 2, "{captured:?}");
    assert_eq!(captured[0].0, json!({"role": "user", "content": "Hello"}));
    assert_eq!(captured[1].0["role"], "assistant");
    assert!(
        captured
            .iter()
            .all(|i| !text(i).contains("<CONVERSATION HISTORY>"))
    );
    assert_eq!(second.input_history, captured);
}

/// Python: `test_summary_and_raw_items_preserve_chronological_order`.
#[test]
fn summary_and_raw_items_preserve_chronological_order() {
    let nested = nest_handoff_history(
        data(
            vec![user("question")],
            Vec::new(),
            vec![
                message("A", "answer"),
                handoff_call("A"),
                handoff_output("A"),
            ],
        ),
        None,
    );
    assert_eq!(nested.input_history.len(), 3);
    let prior_summary = text(&nested.input_history[0]);
    let raw_message = nested.input_history[1].0.to_string();
    let handoff_summary = text(&nested.input_history[2]);
    assert!(prior_summary.contains("question"));
    assert!(raw_message.contains("answer"));
    assert!(handoff_summary.contains("function_call"));
    assert!(!prior_summary.contains("answer"));
    assert!(!handoff_summary.contains("answer"));
}

/// Python: `test_custom_mapper_return_value_is_exact_model_input`. Nothing is appended after a
/// custom mapper's output, and `new_items` is unchanged.
#[test]
fn custom_mapper_return_value_is_exact_model_input() {
    let mapped = user("mapped");
    let mapper: HandoffHistoryMapper = {
        let mapped = mapped.clone();
        Arc::new(move |_| vec![mapped.clone()])
    };
    let input = data(
        vec![user("original")],
        vec![message("A", "before")],
        vec![message("A", "after")],
    );
    let new_items = input.new_items.clone();

    let nested = nest_handoff_history(input, Some(&mapper));

    assert_eq!(nested.input_history, vec![mapped]);
    assert!(nested.pre_handoff_items.is_empty());
    assert_eq!(nested.input_items, Some(Vec::new()));
    assert_eq!(nested.new_items, new_items);
}

/// Python: `test_reasoning_items_are_filtered_from_input_items`.
#[test]
fn reasoning_items_are_filtered_from_input_items() {
    let nested = nest_handoff_history(
        data(
            vec![user("Hello")],
            Vec::new(),
            vec![reasoning("A"), handoff_call("A"), handoff_output("A")],
        ),
        None,
    );
    assert!(nested.input_items.is_some());
    assert!(!types(&nested.input_history).contains(&"reasoning".to_string()));
    assert!(text(&nested.input_history[0]).contains("reasoning"));
}

/// Python: `test_summary_contains_filtered_items_as_text`.
#[test]
fn summary_contains_filtered_items_as_text() {
    let nested = nest_handoff_history(
        data(
            vec![user("Hello")],
            vec![tool_call("A"), tool_output("A")],
            Vec::new(),
        ),
        None,
    );
    let summary = text(&nested.input_history[0]);
    assert!(summary.contains("function_call") || summary.contains("get_weather"));
}

/// Python: `test_input_items_field_exists_after_nesting`.
#[test]
fn input_items_field_exists_after_nesting() {
    let nested = nest_handoff_history(
        data(vec![user("Hello")], Vec::new(), vec![handoff_call("A")]),
        None,
    );
    assert!(nested.input_items.is_some());
}

/// Python: `test_full_handoff_scenario_no_duplication`.
#[test]
fn full_handoff_scenario_has_no_duplicated_tool_items() {
    let nested = nest_handoff_history(
        data(
            vec![user("What's the weather?")],
            vec![tool_call("A"), tool_output("A")],
            vec![
                message("A", "Hello!"),
                handoff_call("A"),
                handoff_output("A"),
            ],
        ),
        None,
    );
    let total_model_items = nested.input_history.len()
        + nested.pre_handoff_items.len()
        + nested.input_items.as_ref().map_or(0, Vec::len);
    assert!(total_model_items <= 3, "{total_model_items}");
    assert!(!types(&nested.input_history).contains(&"function_call_output".to_string()));
    assert!(!types(&nested.input_history).contains(&"function_call".to_string()));
}

// ------------------------------------------------------------------- summary parsing

/// Python: `test_nested_history_keeps_turns_with_no_content` (four cases).
#[test]
fn nested_history_keeps_turns_with_no_content() {
    let empty_items = [
        json!({"role": "user", "content": ""}),
        json!({"role": "assistant", "content": ""}),
        json!({"role": "user", "content": null}),
        json!({"role": "user", "name": "bob", "content": ""}),
    ];
    for empty in empty_items {
        let history = vec![
            user("first question"),
            InputItem(empty.clone()),
            user("second question"),
        ];
        let nested = nest_handoff_history(data(history.clone(), Vec::new(), Vec::new()), None);
        let transcript = flattened_transcript(nested.input_history);
        assert_eq!(transcript.len(), history.len(), "{empty}");
        let roles: Vec<Value> = transcript.iter().map(|i| i.0["role"].clone()).collect();
        assert_eq!(
            roles,
            vec![json!("user"), empty["role"].clone(), json!("user")],
            "{empty}"
        );
    }
}

/// Python: `test_nested_history_survives_repeated_handoffs`.
#[test]
fn nested_history_survives_repeated_handoffs() {
    let history = vec![
        user("first question"),
        InputItem(json!({"role": "assistant", "content": ""})),
        user("second question"),
    ];
    let mut nested = nest_handoff_history(data(history.clone(), Vec::new(), Vec::new()), None);
    for _ in 0..3 {
        nested = nest_handoff_history(
            data(nested.input_history.clone(), Vec::new(), Vec::new()),
            None,
        );
        let transcript = flattened_transcript(nested.input_history.clone());
        assert_eq!(transcript.len(), history.len());
    }
}

/// Python: `test_bare_role_record_from_an_older_summary_is_recovered`.
#[test]
fn bare_role_record_from_an_older_summary_is_recovered() {
    let legacy = summary_item(&[
        "For context, here is the conversation so far between the user and the previous agent:",
        "<CONVERSATION HISTORY>",
        "1. user: first question",
        "2. assistant",
        "3. user: second question",
        "</CONVERSATION HISTORY>",
    ]);
    let transcript = flattened_transcript(vec![legacy]);
    let roles: Vec<Value> = transcript.iter().map(|i| i.0["role"].clone()).collect();
    assert_eq!(
        roles,
        vec![json!("user"), json!("assistant"), json!("user")]
    );
}

/// Python: `test_prose_inside_the_summary_block_is_still_rejected`.
#[test]
fn prose_inside_the_summary_block_is_still_rejected() {
    let with_prose = summary_item(&[
        "For context, here is the conversation so far between the user and the previous agent:",
        "<CONVERSATION HISTORY>",
        "1. user: first question",
        "2. some stray prose with no separator",
        "3. user: second question",
        "</CONVERSATION HISTORY>",
    ]);
    let transcript = flattened_transcript(vec![with_prose]);
    let roles: Vec<Value> = transcript.iter().map(|i| i.0["role"].clone()).collect();
    assert_eq!(roles, vec![json!("user"), json!("user")]);
    let contents: Vec<Value> = transcript.iter().map(|i| i.0["content"].clone()).collect();
    assert_eq!(
        contents,
        vec![json!("first question"), json!("second question")]
    );
}

/// Python: `test_recovered_empty_turn_keeps_explicit_content`. A recovered turn carries a
/// `content` key, so the Chat Completions adapter recognizes it as a message.
#[test]
fn recovered_empty_turn_keeps_explicit_content() {
    let legacy = summary_item(&[
        "For context, here is the conversation so far between the user and the previous agent:",
        "<CONVERSATION HISTORY>",
        "1. user: first question",
        "2. assistant",
        "3. assistant: ",
        "</CONVERSATION HISTORY>",
    ]);
    let transcript = flattened_transcript(vec![legacy]);
    assert_eq!(transcript[1].0, json!({"role": "assistant", "content": ""}));
    assert_eq!(transcript[2].0, json!({"role": "assistant", "content": ""}));
}

/// Python: `test_second_pass_nesting_keeps_empty_turns_provider_valid`.
#[test]
fn second_pass_nesting_keeps_empty_turns() {
    let history = vec![
        user("first question"),
        InputItem(json!({"role": "assistant", "content": ""})),
        user("second question"),
    ];
    let once = nest_handoff_history(data(history, Vec::new(), Vec::new()), None);
    let twice = nest_handoff_history(data(once.input_history, Vec::new(), Vec::new()), None);
    let transcript = flattened_transcript(twice.input_history);
    assert_eq!(transcript[1].0, json!({"role": "assistant", "content": ""}));
    let roles: Vec<Value> = transcript.iter().map(|i| i.0["role"].clone()).collect();
    assert_eq!(
        roles,
        vec![json!("user"), json!("assistant"), json!("user")]
    );
}
