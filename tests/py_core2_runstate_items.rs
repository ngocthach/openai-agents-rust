//! Port of the item and processed-response serialization cases of the Python SDK's
//! `tests/test_run_state.py` (`TestSerializationRoundTrip`, `TestDeserializeHelpers`,
//! `TestRunStateSerializationEdgeCases`, `TestToolApprovalItem`,
//! `TestDeserializeItemsEdgeCases`).
//!
//! Python exercises the private `_deserialize_items` helper with an agent map; Rust resolves
//! agents only at resume, so the same item payloads are fed through `RunState::from_json`
//! inside a minimal envelope and the restored `RunItem`s are checked.

use std::sync::Arc;

use openai_agents_sdk::items::{
    FunctionCall, InputItem, ModelResponse, OutputItem, RunItem, RunItemKind, ToolApprovalItem,
};
use openai_agents_sdk::run_state::{
    CURRENT_SCHEMA_VERSION, ProcessedResponseSnapshot, RunState, ToolRunSnapshot,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::usage::{InputTokensDetails, Usage};
use openai_agents_sdk::{
    Agent, GuardrailFunctionOutput, InputGuardrailResult, OutputGuardrailResult, Runner,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_state(agent: &str) -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: agent.to_string(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
}

fn with_input(agent: &str, original_input: Value) -> RunState {
    let mut state = make_state(agent);
    state.original_input = original_input
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(InputItem)
        .collect();
    state
}

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
}

/// Read `items` through a minimal current-schema envelope, the way `_deserialize_items` sees
/// them inside `RunState.from_json`.
fn deserialize_items(items: Value) -> openai_agents_sdk::Result<Vec<RunItem>> {
    let wire = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": 0,
        "current_agent": {"name": "TestAgent"},
        "original_input": [],
        "max_turns": 5,
        "context": {
            "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "approvals": {},
            "context": {},
        },
        "generated_items": items,
        "model_responses": [],
    });
    RunState::from_json(wire).map(|s| s.generated_items)
}

fn approval(agent: &str, name: &str, call_id: &str, arguments: &str) -> ToolApprovalItem {
    let mut call = FunctionCall::new(name, arguments);
    call.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: agent.to_string(),
        raw_item: call,
        tool_name: name.to_string(),
        tool_origin: None,
    }
}

fn message(text: &str, id: &str) -> RunItem {
    RunItem::new(
        "ItemAgent",
        RunItemKind::MessageOutput {
            raw_item: serde_json::from_value(json!({
                "id": id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }))
            .unwrap(),
        },
    )
}

fn tool_output(agent: &str, raw_item: Value, output: Value) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::ToolCallOutput {
            raw_item,
            output,
            custom_data: None,
        },
    )
}

fn parked(state: &mut RunState, processed: ProcessedResponseSnapshot) {
    state.interruptions = vec![approval(&state.current_agent_name.clone(), "t", "c", "{}")];
    state.pending_tool_processing = true;
    state.last_processed_response = Some(processed);
}

// ---------------------------------------------------------------------------
// TestSerializationRoundTrip
// ---------------------------------------------------------------------------

#[test]
fn preserves_usage_data() {
    let mut state = make_state("UsageAgent");
    state.max_turns = Some(10);
    state.usage = Usage {
        requests: 5,
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 3,
            cache_write_tokens: 7,
        },
        ..Default::default()
    };
    let serialized: Value = serde_json::from_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(serialized["$schemaVersion"], json!(CURRENT_SCHEMA_VERSION));
    assert_eq!(
        serialized["context"]["usage"]["input_tokens_details"],
        json!([{"cached_tokens": 3, "cache_write_tokens": 7}])
    );
    let restored = roundtrip(&state);
    assert_eq!(restored.usage.requests, 5);
    assert_eq!(restored.usage.input_tokens, 100);
    assert_eq!(restored.usage.output_tokens, 50);
    assert_eq!(restored.usage.total_tokens, 150);
    assert_eq!(restored.usage.input_tokens_details.cached_tokens, 3);
    assert_eq!(restored.usage.input_tokens_details.cache_write_tokens, 7);
}

#[test]
fn restores_schema_1_11_usage_without_cache_write_tokens() {
    let mut wire = make_state("UsageAgent").to_json().unwrap();
    wire["$schemaVersion"] = json!("1.11");
    wire["context"]["function_tool_approvals"] = json!([]);
    wire["context"]["usage"]["input_tokens_details"] = json!([{"cached_tokens": 3}]);
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.usage.input_tokens_details.cached_tokens, 3);
    assert_eq!(restored.usage.input_tokens_details.cache_write_tokens, 0);
}

#[test]
fn serializes_generated_items() {
    let mut state = make_state("ItemAgent");
    state.generated_items.push(message("Hello!", "msg_1"));
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["generated_items"].as_array().unwrap().len(), 1);
    assert_eq!(
        json_data["generated_items"][0]["type"],
        json!("message_output_item")
    );
}

#[test]
fn serializes_current_step_interruption() {
    let mut state = make_state("InterruptAgent");
    state.interruptions = vec![approval(
        "InterruptAgent",
        "myTool",
        "cid_int",
        r#"{"arg": "value"}"#,
    )];
    state.pending_tool_processing = true;
    let json_data = state.to_json().unwrap();
    assert_eq!(
        json_data["current_step"]["type"],
        json!("next_step_interruption")
    );
    assert_eq!(
        json_data["current_step"]["data"]["interruptions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let restored = RunState::from_json(json_data).unwrap();
    assert!(restored.pending_tool_processing);
    assert_eq!(restored.interruptions.len(), 1);
    assert_eq!(restored.interruptions[0].name(), "myTool");
}

#[test]
fn deserializes_various_item_types() {
    let mut state = make_state("ItemAgent");
    state.generated_items.push(message("Hello", "msg_1"));
    let mut call = FunctionCall::new("my_tool", r#"{"arg": "val"}"#);
    call.call_id = "call_1".into();
    state.generated_items.push(
        RunItem::new(
            "ItemAgent",
            RunItemKind::ToolCall {
                raw_item: call.to_value(),
            },
        )
        .with_tool_details(
            Some("My tool description".into()),
            Some("My tool title".into()),
        ),
    );
    let mut other = FunctionCall::new("other_tool", "{}");
    other.call_id = "call_2".into();
    state.generated_items.push(RunItem::new(
        "ItemAgent",
        RunItemKind::ToolCall {
            raw_item: other.to_value(),
        },
    ));
    state.generated_items.push(tool_output(
        "ItemAgent",
        json!({"type": "function_call_output", "call_id": "call_1", "output": "result"}),
        json!("result"),
    ));

    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let types: Vec<&str> = restored
        .generated_items
        .iter()
        .map(RunItem::item_type)
        .collect();
    assert_eq!(
        types,
        vec![
            "message_output_item",
            "tool_call_item",
            "tool_call_item",
            "tool_call_output_item"
        ]
    );
    assert_eq!(
        restored.generated_items[1].description.as_deref(),
        Some("My tool description")
    );
    assert_eq!(
        restored.generated_items[1].title.as_deref(),
        Some("My tool title")
    );
    assert_eq!(restored.generated_items[2].description, None);
    assert_eq!(restored.generated_items[2].title, None);
}

#[test]
fn deserializes_custom_tool_call_output_items() {
    let mut state = make_state("ItemAgent");
    let raw = json!({
        "type": "custom_tool_call_output", "call_id": "call_custom_1", "output": "custom result",
    });
    state.generated_items.push(tool_output(
        "ItemAgent",
        raw.clone(),
        json!("custom result"),
    ));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    assert_eq!(restored.generated_items.len(), 1);
    match &restored.generated_items[0].kind {
        RunItemKind::ToolCallOutput {
            raw_item, output, ..
        } => {
            assert_eq!(raw_item, &raw);
            assert_eq!(output, &json!("custom result"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn deserializes_computer_call_output_acknowledged_safety_checks() {
    let mut state = make_state("ItemAgent");
    let checks = json!([{"id": "sc_1", "code": "malicious_instructions", "message": "confirm"}]);
    state.generated_items.push(tool_output(
        "ItemAgent",
        json!({
            "type": "computer_call_output",
            "call_id": "call_computer_1",
            "output": {"type": "computer_screenshot", "image_url": "img"},
            "acknowledged_safety_checks": checks,
        }),
        json!("done"),
    ));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let again = roundtrip(&restored);
    for state in [&restored, &again] {
        match &state.generated_items[0].kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => {
                assert_eq!(raw_item["acknowledged_safety_checks"], checks);
            }
            other => panic!("{other:?}"),
        }
    }
    again.to_json_string().unwrap();
}

#[test]
fn deserializes_tool_call_output_custom_data() {
    let mut state = make_state("ItemAgent");
    let custom = json!({"ui": {"kind": "chart"}, "ids": ["a", "b"]});
    state.generated_items.push(RunItem::new(
        "ItemAgent",
        RunItemKind::ToolCallOutput {
            raw_item: json!({
                "type": "function_call_output", "call_id": "call_custom_data", "output": "result",
            }),
            output: json!("result"),
            custom_data: Some(custom.clone()),
        },
    ));
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["generated_items"][0]["custom_data"], custom);
    assert!(
        json_data["generated_items"][0]["raw_item"]
            .get("custom_data")
            .is_none()
    );
    let restored = RunState::from_json(json_data).unwrap();
    match &restored.generated_items[0].kind {
        RunItemKind::ToolCallOutput { custom_data, .. } => {
            assert_eq!(custom_data.as_ref(), Some(&custom))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn serializes_original_input_with_function_call_output() {
    let state = with_input(
        "TestAgent",
        json!([
            {"type": "function_call", "call_id": "call_123", "name": "test_tool",
             "arguments": r#"{"arg": "value"}"#},
            {"type": "function_call_output", "call_id": "call_123", "output": "result"},
        ]),
    );
    let json_data = state.to_json().unwrap();
    let input = json_data["original_input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["type"], json!("function_call"));
    assert_eq!(input[0]["call_id"], json!("call_123"));
    assert_eq!(input[0]["name"], json!("test_tool"));
    assert_eq!(input[1]["type"], json!("function_call_output"));
    assert_eq!(input[1]["call_id"], json!("call_123"));
    assert!(input[1].get("name").is_none());
    assert!(input[1].get("status").is_none());
    assert_eq!(input[1]["output"], json!("result"));
}

#[test]
fn serializes_assistant_messages() {
    let cases = [
        (
            json!([{"role": "assistant", "content": "This is a summary message"}]),
            "completed",
            "This is a summary message",
        ),
        (
            json!([{"role": "assistant", "status": "in_progress", "content": "In progress message"}]),
            "in_progress",
            "In progress message",
        ),
        (
            json!([{"role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Already array format"}]}]),
            "completed",
            "Already array format",
        ),
    ];
    for (input, status, text) in cases {
        let json_data = with_input("TestAgent", input).to_json().unwrap();
        let input = json_data["original_input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        let msg = &input[0];
        assert_eq!(msg["role"], json!("assistant"));
        assert_eq!(msg["status"], json!(status));
        assert_eq!(msg["content"][0]["type"], json!("output_text"));
        assert_eq!(msg["content"][0]["text"], json!(text));
    }
}

#[test]
fn from_string_normalizes_original_input_dict_items() {
    let state_json = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": 0,
        "current_agent": {"name": "TestAgent"},
        "original_input": [
            {"type": "function_call_output", "call_id": "call123", "name": "test_tool",
             "status": "completed", "output": "result"},
            "simple_string",
        ],
        "model_responses": [],
        "context": {
            "usage": {"requests": 0, "input_tokens": 0, "input_tokens_details": [],
                      "output_tokens": 0, "output_tokens_details": [], "total_tokens": 0,
                      "request_usage_entries": []},
            "approvals": {},
            "context": {},
        },
        "tool_use_tracker": {},
        "max_turns": 10,
        "noActiveAgentRun": true,
        "input_guardrail_results": [],
        "output_guardrail_results": [],
        "generated_items": [],
        "current_step": null,
        "last_model_response": null,
        "last_processed_response": null,
        "current_turn_persisted_item_count": 0,
        "trace": null,
    });
    let state = RunState::from_json(state_json).unwrap();
    assert_eq!(state.original_input.len(), 2);
    assert_eq!(state.original_input[1].0, json!("simple_string"));
    let first = &state.original_input[0].0;
    assert_eq!(first["type"], json!("function_call_output"));
    assert_eq!(first["name"], json!("test_tool"));
    assert_eq!(first["status"], json!("completed"));
    assert_eq!(first["call_id"], json!("call123"));
}

#[test]
fn serializes_original_input_with_non_dict_items() {
    let state = with_input(
        "TestAgent",
        json!([{"role": "user", "content": "Hello"}, "string_item"]),
    );
    let json_data = state.to_json().unwrap();
    let input = json_data["original_input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["role"], json!("user"));
    assert_eq!(input[1], json!("string_item"));
    // And it reads back unchanged.
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(restored.original_input[1].0, json!("string_item"));
}

#[test]
fn from_json_preserves_function_output_original_input() {
    let mut state_json = make_state("TestAgent").to_json().unwrap();
    state_json["original_input"] = json!([
        {"type": "function_call", "call_id": "call_abc", "name": "demo_tool",
         "arguments": r#"{"x":1}"#},
        {"type": "function_call_output", "call_id": "call_abc", "name": "demo_tool",
         "status": "completed", "output": "demo-output"},
    ]);
    let restored = RunState::from_json(state_json).unwrap();
    assert_eq!(restored.original_input.len(), 2);
    let second = &restored.original_input[1].0;
    assert_eq!(restored.original_input[0].0["type"], json!("function_call"));
    assert_eq!(second["type"], json!("function_call_output"));
    assert_eq!(second["call_id"], json!("call_abc"));
    assert_eq!(second["output"], json!("demo-output"));
    assert_eq!(second["name"], json!("demo_tool"));
    assert_eq!(second["status"], json!("completed"));
}

#[test]
fn serialize_tool_call_output_does_not_add_a_name() {
    // Python's `_serialize_item` looks the call's name up but must not write it into the
    // output's raw item.
    let mut state = make_state("TestAgent");
    let mut call = FunctionCall::new("lookup_tool", "{}");
    call.call_id = "call_lookup".into();
    call.id = Some("fc_lookup".into());
    state.generated_items.push(RunItem::new(
        "TestAgent",
        RunItemKind::ToolCall {
            raw_item: call.to_value(),
        },
    ));
    state.generated_items.push(tool_output(
        "TestAgent",
        json!({"type": "function_call_output", "call_id": "call_lookup", "output": "ok"}),
        json!("ok"),
    ));
    let json_data = state.to_json().unwrap();
    let raw = &json_data["generated_items"][1]["raw_item"];
    assert_eq!(raw["type"], json!("function_call_output"));
    assert_eq!(raw["call_id"], json!("call_lookup"));
    assert!(raw.get("name").is_none());
    assert!(raw.get("status").is_none());
}

// ---------------------------------------------------------------------------
// TestDeserializeHelpers
// ---------------------------------------------------------------------------

#[test]
fn serialization_includes_handoff_fields() {
    let mut state = make_state("AgentA");
    state.generated_items.push(RunItem::new(
        "AgentB",
        RunItemKind::HandoffOutput {
            raw_item: InputItem(json!({"type": "handoff_output", "status": "completed"})),
            source_agent: "AgentA".into(),
            target_agent: "AgentB".into(),
        },
    ));
    let json_data = state.to_json().unwrap();
    let item = &json_data["generated_items"][0];
    assert_eq!(item["source_agent"]["name"], json!("AgentA"));
    assert_eq!(item["target_agent"]["name"], json!("AgentB"));
    let restored = roundtrip(&state);
    assert_eq!(restored.generated_items.len(), 1);
    assert_eq!(
        restored.generated_items[0].item_type(),
        "handoff_output_item"
    );
}

#[test]
fn model_response_serialization_roundtrip() {
    let mut state = make_state("TestAgent");
    state.model_responses.push(ModelResponse {
        output: vec![OutputItem::from_value(json!({
            "type": "message", "id": "msg1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
        }))],
        usage: Usage {
            requests: 1,
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
            ..Default::default()
        },
        response_id: Some("resp123".into()),
        request_id: Some("req123".into()),
        raw_usage: Some(json!({"input_tokens": 10, "provider_metric": 0})),
    });
    let serialized = state.to_json().unwrap();
    assert!(serialized["model_responses"][0].get("raw_usage").is_none());
    let restored = roundtrip(&state);
    let response = &restored.model_responses[0];
    assert_eq!(response.response_id.as_deref(), Some("resp123"));
    assert_eq!(response.request_id.as_deref(), Some("req123"));
    assert_eq!(response.raw_usage, None);
    assert_eq!(response.usage.requests, 1);
    assert_eq!(response.usage.input_tokens, 10);
}

#[test]
fn interruptions_serialization_roundtrip() {
    let mut state = make_state("InterruptAgent");
    let mut item = approval(
        "InterruptAgent",
        "sensitive_tool",
        "call789",
        r#"{"data": "value"}"#,
    );
    item.raw_item.id = Some("1".into());
    state.interruptions = vec![item];
    state.pending_tool_processing = true;
    let restored = roundtrip(&state);
    assert_eq!(restored.interruptions.len(), 1);
    assert_eq!(restored.interruptions[0].raw_item.name, "sensitive_tool");
}

#[test]
fn nested_agent_tool_interruptions_roundtrip() {
    let mut state = make_state("OuterAgent");
    state.interruptions = vec![approval("InnerAgent", "sensitive_tool", "inner-1", "{}")];
    state.pending_tool_processing = true;
    let restored = roundtrip(&state);
    let interruptions = restored.get_interruptions();
    assert_eq!(interruptions.len(), 1);
    assert_eq!(interruptions[0].agent, "InnerAgent");
    assert_eq!(interruptions[0].raw_item.name, "sensitive_tool");
}

#[tokio::test]
async fn missing_agent_in_map_error() {
    // Python refuses the snapshot in `from_string(agent_b, ...)`. Rust reads snapshots
    // without an agent graph, so the same check happens when the run resumes.
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let state = roundtrip(&make_state("AgentA"));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("unused")]);
    let agent_b = Agent::<()>::new("AgentB").with_model_instance(model);
    let err = Runner::run(&agent_b, state).await.unwrap_err().to_string();
    assert!(
        err.contains("Run state agent not found in agent map"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// TestRunStateSerializationEdgeCases (wire shape of the processed response)
// ---------------------------------------------------------------------------

fn run(name: &str, call: Value) -> ToolRunSnapshot {
    ToolRunSnapshot {
        tool_name: name.into(),
        tool_call: call,
        ..Default::default()
    }
}

#[test]
fn to_json_keeps_message_raw_item_ids() {
    let mut state = make_state("TestAgent");
    state.generated_items.push(message("Hello", "msg1"));
    let json_data = state.to_json().unwrap();
    assert_eq!(
        json_data["generated_items"][0]["raw_item"]["id"],
        json!("msg1")
    );
}

#[test]
fn from_json_with_last_processed_response() {
    let mut state = make_state("TestAgent");
    let mut call = FunctionCall::new("test_tool", "{}");
    call.call_id = "call123".into();
    let item = RunItem::new(
        "TestAgent",
        RunItemKind::ToolCall {
            raw_item: call.to_value(),
        },
    );
    state.generated_items = vec![item.clone()];
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            new_items: vec![item],
            ..Default::default()
        },
    );
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let processed = restored.last_processed_response.unwrap();
    assert_eq!(processed.new_items.len(), 1);
    assert_eq!(processed.new_items[0].item_type(), "tool_call_item");
}

#[test]
fn last_processed_response_serializes_local_shell_actions() {
    let mut state = make_state("TestAgent");
    let call = json!({
        "type": "local_shell_call", "id": "ls1", "call_id": "call_local", "status": "completed",
        "action": {"commands": ["echo hi"], "timeout_ms": 1000},
    });
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            local_shell_actions: vec![run("local_shell", call)],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let actions = &json_data["last_processed_response"]["local_shell_actions"];
    assert_eq!(actions[0]["local_shell"]["name"], json!("local_shell"));
    let restored = RunState::from_json(json_data).unwrap();
    let processed = restored.last_processed_response.unwrap();
    assert_eq!(processed.local_shell_actions.len(), 1);
    assert_eq!(processed.local_shell_actions[0].tool_name, "local_shell");
    assert_eq!(
        processed.local_shell_actions[0].tool_call["call_id"],
        json!("call_local")
    );
}

#[test]
fn serialize_tool_action_groups() {
    let mut state = make_state("TestAgent");
    let call = |id: &str| json!({"type": "function_call", "call_id": id});
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            functions: vec![run("func_tool", call("func-call"))],
            computer_actions: vec![run("computer_tool", call("comp-call"))],
            local_shell_actions: vec![run("local_shell_tool", call("local-call"))],
            shell_actions: vec![run("shell_tool", call("shell-call"))],
            apply_patch_actions: vec![run("apply_patch_tool", call("patch-call"))],
            handoffs: vec![run("handoff_tool", call("handoff-call"))],
            mcp_approval_requests: vec![json!({
                "request_item": {"raw_item": {"type": "mcp_approval_request"}},
                "mcp_tool": {"name": "mcp_tool"},
            })],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let processed = json_data["last_processed_response"].as_object().unwrap();
    for key in [
        "functions",
        "computer_actions",
        "custom_tool_actions",
        "local_shell_actions",
        "shell_actions",
        "apply_patch_actions",
        "handoffs",
        "mcp_approval_requests",
    ] {
        assert!(processed.contains_key(key), "{key}");
    }
    assert_eq!(
        processed["functions"][0]["tool"]["name"],
        json!("func_tool")
    );
    assert_eq!(
        processed["functions"][0]["tool_call"]["call_id"],
        json!("func-call")
    );
    assert_eq!(
        processed["computer_actions"][0]["computer"]["name"],
        json!("computer_tool")
    );
    assert_eq!(
        processed["shell_actions"][0]["shell"]["name"],
        json!("shell_tool")
    );
    assert_eq!(
        processed["apply_patch_actions"][0]["apply_patch"]["name"],
        json!("apply_patch_tool")
    );
    assert_eq!(
        processed["handoffs"][0]["handoff"]["tool_name"],
        json!("handoff_tool")
    );
    assert_eq!(
        processed["mcp_approval_requests"][0]["mcp_tool"]["name"],
        json!("mcp_tool")
    );
}

#[test]
fn serialize_tool_action_groups_preserves_synthetic_namespace_for_deferred_tools() {
    let mut state = make_state("TestAgent");
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            functions: vec![run(
                "get_weather",
                json!({"type": "function_call", "name": "get_weather", "namespace": "get_weather",
                       "call_id": "weather-call", "arguments": r#"{"city": "Tokyo"}"#}),
            )],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let function = &json_data["last_processed_response"]["functions"][0];
    assert_eq!(function["tool"]["name"], json!("get_weather"));
    assert!(function["tool"].get("namespace").is_none());
    assert!(function["tool"].get("qualifiedName").is_none());
    assert_eq!(
        function["tool"]["lookupKey"],
        json!({"kind": "deferred_top_level", "name": "get_weather"})
    );
    assert_eq!(function["tool_call"]["namespace"], json!("get_weather"));
}

#[test]
fn serialize_guardrail_results() {
    let output = GuardrailFunctionOutput {
        output_info: json!({"info": "details"}),
        tripwire_triggered: false,
    };
    let mut state = make_state("AgentA");
    state.input_guardrail_results = vec![InputGuardrailResult {
        guardrail_name: "input".into(),
        output: output.clone(),
    }];
    state.output_guardrail_results = vec![OutputGuardrailResult {
        guardrail_name: "output".into(),
        agent_name: "AgentA".into(),
        agent_output: json!("some_output"),
        output,
    }];
    let json_data = state.to_json().unwrap();
    assert_eq!(
        json_data["input_guardrail_results"][0]["guardrail"]["type"],
        json!("input")
    );
    let entry = &json_data["output_guardrail_results"][0];
    assert_eq!(entry["guardrail"]["type"], json!("output"));
    assert_eq!(entry["agentOutput"], json!("some_output"));
    assert_eq!(entry["agent"]["name"], json!("AgentA"));
}

#[test]
fn serialize_handoff_with_name_fallback() {
    let mut state = make_state("AgentA");
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            handoffs: vec![run(
                "handoff_tool",
                json!({"type": "function_call", "name": "handoff_tool", "call_id": "call123",
                       "status": "completed", "arguments": "{}"}),
            )],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let handoffs = json_data["last_processed_response"]["handoffs"]
        .as_array()
        .unwrap();
    assert_eq!(handoffs.len(), 1);
    assert_eq!(handoffs[0]["handoff"]["tool_name"], json!("handoff_tool"));
}

#[test]
fn serialize_function_with_description_and_schema() {
    let mut state = make_state("TestAgent");
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            functions: vec![ToolRunSnapshot {
                description: Some("Test tool description".into()),
                params_json_schema: Some(json!({"type": "object", "properties": {}})),
                ..run(
                    "test_tool",
                    json!({"type": "function_call", "name": "test_tool", "call_id": "call123",
                           "arguments": "{}"}),
                )
            }],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let tool = &json_data["last_processed_response"]["functions"][0]["tool"];
    assert_eq!(tool["description"], json!("Test tool description"));
    assert!(tool.get("paramsJsonSchema").is_some());
}

#[test]
fn serialize_local_action_groups_with_description() {
    // Python's computer/shell/apply_patch cases: a described tool writes its description into
    // its group wrapper, and only function tools write a parameter schema.
    let mut state = make_state("TestAgent");
    let described = |name: &str, call: Value| ToolRunSnapshot {
        description: Some(format!("{name} description")),
        params_json_schema: Some(json!({"type": "object"})),
        ..run(name, call)
    };
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            computer_actions: vec![described(
                "computer_use_preview",
                json!({"id": "1", "type": "computer_call", "call_id": "call123",
                       "status": "completed", "action": {"type": "screenshot"},
                       "pending_safety_checks": []}),
            )],
            shell_actions: vec![described(
                "shell",
                json!({"id": "1", "type": "shell_call", "call_id": "call124",
                       "status": "completed", "command": "echo test"}),
            )],
            apply_patch_actions: vec![described(
                "apply_patch",
                json!({"type": "function_call", "name": "apply_patch", "call_id": "call125",
                       "arguments": "{}"}),
            )],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let processed = &json_data["last_processed_response"];
    for (key, wrapper, name) in [
        ("computer_actions", "computer", "computer_use_preview"),
        ("shell_actions", "shell", "shell"),
        ("apply_patch_actions", "apply_patch", "apply_patch"),
    ] {
        let tool = &processed[key][0][wrapper];
        assert_eq!(tool["name"], json!(name));
        assert_eq!(tool["description"], json!(format!("{name} description")));
        assert!(tool.get("paramsJsonSchema").is_none(), "{key}");
    }
}

#[test]
fn serialize_mcp_approval_request() {
    let mut state = make_state("TestAgent");
    parked(
        &mut state,
        ProcessedResponseSnapshot {
            mcp_approval_requests: vec![json!({
                "request_item": {"raw_item": {
                    "id": "req123", "type": "mcp_approval_request", "name": "mcp_tool",
                    "server_label": "test_server", "arguments": "{}",
                }},
                "mcp_tool": {"name": "mcp_tool"},
            })],
            ..Default::default()
        },
    );
    let json_data = state.to_json().unwrap();
    let requests = json_data["last_processed_response"]["mcp_approval_requests"]
        .as_array()
        .unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].get("request_item").is_some());
    assert_eq!(requests[0]["mcp_tool"]["name"], json!("mcp_tool"));
    state.to_json_string().unwrap();
}

#[tokio::test]
async fn custom_tool_snapshot_written_by_a_run_carries_its_description() {
    // The runner's own snapshot: a custom tool call parked on approval records the tool
    // description, as Python's `custom_tool_actions` group does.
    use openai_agents_sdk::tool::CustomTool;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "custom_tool_call", "call_id": "custom-1", "name": "raw_editor",
        "input": "patch",
    }))]);
    let tool = CustomTool::<()>::new(
        "raw_editor",
        "Edits raw text",
        |_ctx, input: String| async move { Ok(input) },
    )
    .with_needs_approval(true);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(tool)
        .with_model_instance(model);
    let result = Runner::run(&agent, "edit").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let wire = result.to_state().to_json().unwrap();
    let group = &wire["last_processed_response"]["custom_tool_actions"][0];
    assert_eq!(group["custom_tool"]["name"], json!("raw_editor"));
    assert_eq!(group["custom_tool"]["description"], json!("Edits raw text"));
}

#[test]
fn deserialize_tool_call_output_item_different_types() {
    let raws = [
        json!({"type": "function_call_output", "call_id": "call123", "output": "result"}),
        json!({"type": "computer_call_output", "call_id": "call123",
               "output": {"type": "computer_screenshot", "screenshot": "screenshot"}}),
        json!({"type": "local_shell_call_output", "id": "shell123", "call_id": "call123",
               "output": "result"}),
    ];
    for raw in raws {
        let items = deserialize_items(json!([{
            "type": "tool_call_output_item", "agent": {"name": "TestAgent"}, "raw_item": raw,
        }]))
        .unwrap();
        assert_eq!(items.len(), 1);
        match &items[0].kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => assert_eq!(raw_item, &raw),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn deserialize_reasoning_compaction_handoff_and_approval_items() {
    let items = deserialize_items(json!([
        {"type": "reasoning_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "reasoning", "id": "reasoning123", "summary": [], "content": []}},
        {"type": "compaction_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "compaction", "summary": "..."}},
        {"type": "handoff_call_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "function_call", "name": "handoff_tool", "call_id": "call123",
                      "status": "completed", "arguments": "{}"}},
        {"type": "tool_approval_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "function_call", "name": "test_tool", "call_id": "call124",
                      "status": "completed", "arguments": "{}"}},
    ]))
    .unwrap();
    let types: Vec<&str> = items.iter().map(RunItem::item_type).collect();
    assert_eq!(
        types,
        vec![
            "reasoning_item",
            "compaction_item",
            "handoff_call_item",
            "tool_approval_item"
        ]
    );
    match &items[1].kind {
        RunItemKind::Compaction { raw_item } => assert_eq!(raw_item["type"], json!("compaction")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn deserialize_handoff_output_item_without_agent() {
    let items = deserialize_items(json!([{
        "type": "handoff_output_item",
        "source_agent": {"name": "SourceAgent"},
        "target_agent": {"name": "TargetAgent"},
        "raw_item": {"type": "function_call_output", "call_id": "call123",
                     "name": "transfer_to_weather", "status": "completed", "output": "payload"},
    }]))
    .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].item_type(), "handoff_output_item");
    assert_eq!(items[0].agent, "SourceAgent");
}

#[test]
fn deserialize_handoff_output_items_with_string_agents() {
    // `TestDeserializeItemsEdgeCases`: string and dict agent references both resolve.
    let items = deserialize_items(json!([
        {"type": "handoff_output_item", "source_agent": "Agent1",
         "target_agent": {"name": "Agent2"},
         "raw_item": {"role": "assistant", "content": "Handoff message"}},
        {"type": "handoff_output_item", "source_agent": {"name": "Agent1"},
         "target_agent": "Agent2",
         "raw_item": {"role": "assistant", "content": "Handoff message"}},
        {"type": "handoff_output_item", "target_agent": "Agent2",
         "raw_item": {"role": "assistant", "content": "Handoff message"}},
    ]))
    .unwrap();
    assert_eq!(items.len(), 3);
    for item in &items {
        match &item.kind {
            RunItemKind::HandoffOutput { target_agent, .. } => assert_eq!(target_agent, "Agent2"),
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(items[0].agent, "Agent1");
    assert_eq!(items[2].agent, "Agent2");
}

#[test]
fn deserialize_mcp_items() {
    let items = deserialize_items(json!([
        {"type": "mcp_list_tools_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "mcp_list_tools", "id": "list123", "server_label": "test_server",
                      "tools": []}},
        {"type": "mcp_approval_request_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "mcp_approval_request", "id": "req123", "name": "mcp_tool",
                      "server_label": "test_server", "arguments": "{}"}},
        {"type": "mcp_approval_response_item", "agent": {"name": "TestAgent"},
         "raw_item": {"type": "mcp_approval_response", "approval_request_id": "req123",
                      "approve": true, "caller": {"type": "program", "caller_id": "program123"}}},
    ]))
    .unwrap();
    let types: Vec<&str> = items.iter().map(RunItem::item_type).collect();
    assert_eq!(
        types,
        vec![
            "mcp_list_tools_item",
            "mcp_approval_request_item",
            "mcp_approval_response_item"
        ]
    );
    match &items[2].kind {
        RunItemKind::MCPApprovalResponse { raw_item } => assert_eq!(
            raw_item["caller"],
            json!({"type": "program", "caller_id": "program123"})
        ),
        other => panic!("{other:?}"),
    }
}

#[test]
fn deserialize_items_restores_tool_search_items() {
    let items = deserialize_items(json!([
        {"type": "tool_search_call_item", "agent": {"name": "TestAgent"},
         "raw_item": {"id": "tsc_state", "type": "tool_search_call",
                      "arguments": {"paths": ["crm"], "query": "profile"},
                      "execution": "server", "status": "completed"}},
        {"type": "tool_search_output_item", "agent": {"name": "TestAgent"},
         "raw_item": {"id": "tso_state", "type": "tool_search_output", "execution": "server",
                      "status": "completed", "tools": [{"type": "function",
                      "name": "get_customer_profile", "defer_loading": true}]}},
    ]))
    .unwrap();
    match (&items[0].kind, &items[1].kind) {
        (
            RunItemKind::ToolCall { raw_item: call },
            RunItemKind::ToolCallOutput { raw_item: out, .. },
        ) => {
            assert_eq!(call["type"], json!("tool_search_call"));
            assert_eq!(out["type"], json!("tool_search_output"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn deserialize_items_handles_string_and_object_agent_fields() {
    let raw = json!({"type": "message", "id": "msg1", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                     "status": "completed"});
    for agent in [json!("TestAgent"), json!({"name": "TestAgent"})] {
        let items = deserialize_items(json!([{
            "type": "message_output_item", "agent": agent, "raw_item": raw,
        }]))
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].item_type(), "message_output_item");
        assert_eq!(items[0].agent, "TestAgent");
    }
}

#[test]
fn deserialize_items_fallback_union_type() {
    let items = deserialize_items(json!([{
        "type": "tool_call_output_item", "agent": {"name": "TestAgent"},
        "raw_item": {"type": "function_call_output", "call_id": "call123", "output": "result"},
    }]))
    .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].item_type(), "tool_call_output_item");
}

#[test]
fn from_json_handles_string_original_input() {
    let wire = json!({
        "$schemaVersion": CURRENT_SCHEMA_VERSION,
        "current_turn": 0,
        "current_agent": {"name": "TestAgent"},
        "original_input": "string_input",
        "max_turns": 5,
        "context": {"usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                              "total_tokens": 0},
                    "approvals": {}, "context": {}},
        "generated_items": [],
        "model_responses": [],
    });
    let state = RunState::from_json(wire).unwrap();
    // Rust keeps the input as items: the bare string is one user message.
    assert_eq!(state.original_input.len(), 1);
    assert_eq!(
        state.original_input[0].text_content().as_deref(),
        Some("string_input")
    );
}

// ---------------------------------------------------------------------------
// TestToolApprovalItem
// ---------------------------------------------------------------------------

#[test]
fn tool_approval_item_names() {
    // Explicit tool name wins over the raw item's name; otherwise the raw name is used.
    let mut explicit = approval("TestAgent", "raw_tool_name", "call123", "{}");
    explicit.tool_name = "explicit_tool_name".into();
    assert_eq!(explicit.name(), "explicit_tool_name");
    let wire = json!({
        "type": "tool_approval_item", "agent": {"name": "TestAgent"},
        "raw_item": {"type": "function_call", "name": "raw_tool_name", "call_id": "call123",
                     "status": "completed", "arguments": "{}"},
    });
    let mut state = make_state("TestAgent").to_json().unwrap();
    state["current_step"] = json!({"type": "next_step_interruption",
                                   "data": {"interruptions": [wire]}});
    let restored = RunState::from_json(state).unwrap();
    assert_eq!(restored.interruptions[0].name(), "raw_tool_name");
}

#[test]
fn approve_and_reject_tool_with_explicit_tool_name() {
    let mut state = make_state("TestAgent");
    let mut approved = approval("TestAgent", "raw_name", "call123", "{}");
    approved.tool_name = "explicit_name".into();
    state.approve(&approved, false);
    assert_eq!(
        state
            .approvals
            .status("TestAgent", "explicit_name", "call123"),
        Some(true)
    );
    let mut rejected = approval("TestAgent", "raw_name", "call789", "{}");
    rejected.tool_name = "explicit_name".into();
    state.reject(&rejected, false, None);
    assert_eq!(
        state
            .approvals
            .status("TestAgent", "explicit_name", "call789"),
        Some(false)
    );
}

#[test]
fn tool_approval_item_with_tool_name_round_trips() {
    let mut state = make_state("TestAgent");
    let mut item = approval("TestAgent", "raw_name", "call123", "{}");
    item.tool_name = "explicit_name".into();
    state.generated_items.push(RunItem::new(
        "TestAgent",
        RunItemKind::ToolApproval {
            raw_item: item.raw_item.clone(),
            tool_name: item.tool_name.clone(),
        },
    ));
    let json_data = state.to_json().unwrap();
    let data = &json_data["generated_items"][0];
    assert_eq!(data["type"], json!("tool_approval_item"));
    assert_eq!(data["tool_name"], json!("explicit_name"));

    let restored = RunState::from_json(json_data).unwrap();
    let restored_item = restored.generated_items[0].as_tool_approval().unwrap();
    assert_eq!(restored_item.tool_name, "explicit_name");
    assert_eq!(restored_item.name(), "explicit_name");

    // `_deserialize_items` of a hand-written payload keeps the explicit name too.
    let items = deserialize_items(json!([{
        "type": "tool_approval_item", "agent": {"name": "TestAgent"},
        "tool_name": "explicit_tool_name",
        "raw_item": {"type": "function_call", "name": "raw_tool_name", "call_id": "call123",
                     "status": "completed", "arguments": "{}"},
    }]))
    .unwrap();
    assert_eq!(
        items[0].as_tool_approval().unwrap().name(),
        "explicit_tool_name"
    );
}

#[test]
fn tool_approval_item_arguments_property() {
    assert_eq!(
        approval("TestAgent", "tool1", "call1", r#"{"city": "Oakland"}"#).arguments(),
        r#"{"city": "Oakland"}"#
    );
}

#[test]
fn tool_approval_item_tracks_namespace() {
    let mut item = approval("TestAgent", "lookup_account", "call-ns-1", "{}");
    item.raw_item.extra.insert("namespace".into(), json!("crm"));
    assert_eq!(item.tool_name, "lookup_account");
    assert_eq!(item.tool_namespace(), Some("crm"));
    assert_eq!(item.qualified_name(), "crm.lookup_account");
}

#[test]
fn tool_approval_item_collapses_synthetic_deferred_namespace_in_qualified_name() {
    let mut item = approval("TestAgent", "get_weather", "call-weather-1", "{}");
    item.raw_item
        .extra
        .insert("namespace".into(), json!("get_weather"));
    assert_eq!(item.tool_namespace(), Some("get_weather"));
    assert_eq!(item.qualified_name(), "get_weather");
}

#[test]
fn round_trip_serialization_with_tool_namespace() {
    let mut state = make_state("TestAgent");
    let mut item = approval("TestAgent", "lookup_account", "call123", "{}");
    item.raw_item
        .extra
        .insert("namespace".into(), json!("billing"));
    state.generated_items.push(RunItem::new(
        "TestAgent",
        RunItemKind::ToolApproval {
            raw_item: item.raw_item.clone(),
            tool_name: item.tool_name.clone(),
        },
    ));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let restored_item = restored.generated_items[0].as_tool_approval().unwrap();
    assert_eq!(restored_item.tool_name, "lookup_account");
    assert_eq!(restored_item.tool_namespace(), Some("billing"));
    assert_eq!(restored_item.qualified_name(), "billing.lookup_account");
}

#[test]
fn round_trip_deserializes_statusless_message_output_items() {
    let raw = json!({
        "id": "msg_constructed", "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
    });
    let mut state = make_state("TestAgent");
    state.generated_items.push(RunItem::new(
        "TestAgent",
        RunItemKind::MessageOutput {
            raw_item: serde_json::from_value(raw.clone()).unwrap(),
        },
    ));
    state.model_responses.push(ModelResponse {
        output: vec![OutputItem::from_value(raw.clone())],
        usage: Usage::default(),
        response_id: None,
        request_id: None,
        raw_usage: None,
    });
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let message = restored.generated_items[0].as_message().unwrap();
    assert_eq!(message.status, None);
    let wire = restored.to_json().unwrap();
    assert!(
        wire["generated_items"][0]["raw_item"]
            .get("status")
            .is_none()
    );
    assert!(
        wire["model_responses"][0]["output"][0]
            .get("status")
            .is_none()
    );
}
