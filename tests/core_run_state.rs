//! `RunState` wire-format interoperability with the Python SDK's `agents.run_state`.

use openai_agents_sdk::items::{FunctionCall, InputItem, RunItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run_context::ApprovalsState;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::usage::Usage;
use serde_json::{Value, json};

fn sample_state() -> RunState {
    let call = FunctionCall::new("get_weather", r#"{"city":"Oslo"}"#);
    let approval = ToolApprovalItem {
        agent: "Weather".into(),
        raw_item: call.clone(),
        tool_name: "get_weather".into(),
        tool_origin: None,
    };
    let mut approvals = ApprovalsState::default();
    approvals.approve(&approval, false);
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_turn: 2,
        max_turns: Some(10),
        current_agent_name: "Weather".into(),
        original_input: vec![InputItem::user("Oslo?")],
        generated_items: vec![
            RunItem::new(
                "Weather",
                RunItemKind::ToolCall {
                    raw_item: call.to_value(),
                },
            ),
            RunItem::new(
                "Weather",
                RunItemKind::ToolCallOutput {
                    raw_item: json!({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": "cold",
                    }),
                    output: json!("cold"),
                    custom_data: Some(json!({"latency_ms": 12})),
                },
            ),
        ],
        usage: Usage::single(7, 3),
        approvals,
        interruptions: vec![approval],
        pending_tool_processing: true,
        conversation_id: Some("conv_1".into()),
        trace_id: Some("trace_abc".into()),
        tool_use_tracker: [("Weather".to_string(), vec!["get_weather".to_string()])]
            .into_iter()
            .collect(),
        ..Default::default()
    }
}

#[test]
fn to_json_emits_the_python_envelope() {
    let wire = sample_state().to_json().unwrap();
    assert_eq!(wire["$schemaVersion"], CURRENT_SCHEMA_VERSION);
    assert_eq!(wire["current_agent"], json!({"name": "Weather"}));
    assert_eq!(wire["no_active_agent_run"], json!(true));
    assert_eq!(wire["max_turns"], json!(10));
    assert_eq!(wire["current_turn"], json!(2));

    // Python keys the per-agent decisions under `context.function_tool_approvals`.
    let approvals = &wire["context"]["function_tool_approvals"];
    assert_eq!(approvals[0]["agent"], json!({"name": "Weather"}));
    assert_eq!(approvals[0]["tool_key"], json!("get_weather"));
    assert_eq!(
        approvals[0]["decision"]["approved"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(approvals[0]["decision"]["rejected"], json!([]));

    // Items carry an agent reference object, not a bare name.
    let item = &wire["generated_items"][0];
    assert_eq!(item["type"], json!("tool_call_item"));
    assert_eq!(item["agent"], json!({"name": "Weather"}));
    assert_eq!(item["raw_item"]["type"], json!("function_call"));
    // SDK-only custom data rides on the item, never inside `raw_item`.
    assert_eq!(
        wire["generated_items"][1]["custom_data"],
        json!({"latency_ms": 12})
    );
    assert!(wire["generated_items"][1]["raw_item"]["custom_data"].is_null());

    // Usage token details are one-element lists in Python.
    assert!(wire["context"]["usage"]["input_tokens_details"].is_array());
    assert_eq!(wire["context"]["usage"]["input_tokens"], json!(7));

    // An interrupted run serializes a `next_step_interruption` step.
    assert_eq!(
        wire["current_step"]["type"],
        json!("next_step_interruption")
    );
    let interruption = &wire["current_step"]["data"]["interruptions"][0];
    assert_eq!(interruption["type"], json!("tool_approval_item"));
    assert_eq!(interruption["tool_name"], json!("get_weather"));

    assert_eq!(wire["trace"], json!({"object": "trace", "id": "trace_abc"}));
    assert_eq!(
        wire["context"]["context_meta"]["requires_deserializer"],
        json!(false)
    );
    assert_eq!(wire["tool_use_tracker"]["Weather"], json!(["get_weather"]));
}

#[test]
fn round_trips_through_the_envelope() {
    let state = sample_state();
    let mut restored = RunState::from_json(state.to_json().unwrap()).unwrap();

    // `sample_state` approves a call that its (empty) tool-invocation ledger does not bind, so
    // reading it back through the 1.21 envelope correctly flags that approval for renewal —
    // the one thing the round trip is meant to add. A snapshot a real run produces always
    // ledgers its parked call, so this does not arise there.
    let call_id = state
        .interruptions
        .first()
        .expect("sample state parks on an approval")
        .call_id()
        .to_string();
    assert_eq!(
        restored.restored_unbound_approval_call_ids,
        std::collections::BTreeSet::from([call_id])
    );

    // Everything else has to survive the envelope untouched.
    restored.restored_unbound_approval_call_ids.clear();
    assert_eq!(restored, state);
}

#[test]
fn reads_a_python_written_snapshot() {
    let wire = json!({
        "$schemaVersion": "1.21",
        "current_turn": 1,
        "current_agent": {"name": "Assistant"},
        "original_input": [{"role": "user", "content": "hi"}],
        "pending_input": [],
        "model_responses": [{
            "usage": {
                "requests": 1,
                "input_tokens": 5,
                "input_tokens_details": [{"cached_tokens": 1, "cache_write_tokens": 0}],
                "output_tokens": 2,
                "output_tokens_details": [{"reasoning_tokens": 0}],
                "total_tokens": 7,
                "request_usage_entries": [],
            },
            "output": [{"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}],
            "response_id": "resp_1",
            "request_id": null,
        }],
        "context": {
            "usage": {
                "requests": 1,
                "input_tokens": 5,
                "input_tokens_details": [{"cached_tokens": 1, "cache_write_tokens": 0}],
                "output_tokens": 2,
                "output_tokens_details": [{"reasoning_tokens": 0}],
                "total_tokens": 7,
                "request_usage_entries": [],
            },
            "approvals": {"legacy_tool": {"approved": true, "rejected": false}},
            "function_tool_approvals": [{
                "agent": {"name": "Assistant"},
                "tool_key": "t",
                "decision": {
                    "approved": ["c1"],
                    "rejected": false,
                    "rejection_messages": {},
                },
            }],
            "tool_invocations": {},
            "context": null,
            "context_meta": {},
        },
        "tool_use_tracker": {"Assistant": ["t"]},
        "max_turns": null,
        "no_active_agent_run": true,
        "input_guardrail_results": [],
        "output_guardrail_results": [{
            "guardrail": {"type": "output", "name": "g"},
            "output": {"tripwireTriggered": false, "outputInfo": {"ok": true}},
            "agentOutput": "hello",
            "agent": {"name": "Assistant"},
        }],
        "tool_input_guardrail_results": [],
        "tool_output_guardrail_results": [],
        "conversation_id": null,
        "previous_response_id": "resp_1",
        "auto_previous_response_id": true,
        "generated_prompt_cache_key": "agents-sdk:conversation:abc",
        "reasoning_item_id_policy": null,
        "nested_history_owned_session_item_refs": [],
        "generated_session_item_indexes": [0],
        "generated_items": [{
            "type": "tool_call_item",
            "raw_item": {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            "agent": {"name": "Assistant"},
        }],
        "session_items": [],
        "current_step": {
            "type": "next_step_interruption",
            "data": {
                "interruptions": [{
                    "type": "tool_approval_item",
                    "raw_item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "t",
                        "arguments": "{}",
                    },
                    "agent": {"name": "Assistant"},
                    "tool_name": "t",
                    "tool_namespace": null,
                }],
            },
        },
        "last_model_response": null,
        "last_processed_response": null,
        "current_turn_persisted_item_count": 1,
        "trace": {"object": "trace", "id": "trace_1"},
    });

    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.current_agent_name, "Assistant");
    assert_eq!(state.current_turn, 1);
    assert_eq!(state.max_turns, None);
    assert_eq!(state.usage.input_tokens, 5);
    assert_eq!(state.usage.input_tokens_details.cached_tokens, 1);
    assert_eq!(state.generated_items.len(), 1);
    assert_eq!(state.generated_items[0].agent, "Assistant");
    assert_eq!(state.model_responses.len(), 1);
    assert_eq!(
        state.model_responses[0].response_id.as_deref(),
        Some("resp_1")
    );
    assert_eq!(state.interruptions.len(), 1);
    assert_eq!(state.interruptions[0].tool_name, "t");
    assert!(state.pending_tool_processing);
    assert_eq!(
        state.approvals.status("Assistant", "t", "c1"),
        Some(true),
        "function-tool approvals keep their owning agent"
    );
    assert_eq!(
        state.approvals.status("", "legacy_tool", "any"),
        Some(true),
        "agent-agnostic decisions are stored under an empty agent"
    );
    assert_eq!(state.output_guardrail_results.len(), 1);
    assert_eq!(
        state.output_guardrail_results[0].agent_output,
        json!("hello")
    );
    assert_eq!(state.output_guardrail_results[0].agent_name, "Assistant");
    assert_eq!(state.trace_id.as_deref(), Some("trace_1"));
    assert_eq!(state.tool_use_tracker["Assistant"], vec!["t".to_string()]);
    assert_eq!(
        state.generated_prompt_cache_key.as_deref(),
        Some("agents-sdk:conversation:abc")
    );
    assert!(state.auto_previous_response_id);
}

#[test]
fn still_reads_the_flat_rust_1_0_snapshot() {
    let legacy = json!({
        "$schemaVersion": "1.0",
        "current_turn": 3,
        "max_turns": 8,
        "current_agent_name": "Legacy",
        "original_input": [{"role": "user", "content": "hi"}],
        "generated_items": [{
            "agent": "Legacy",
            "type": "message_output_item",
            "raw_item": {"type": "message", "role": "assistant", "content": [], "status": "completed"},
        }],
        "model_responses": [],
        "usage": {"requests": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        "approvals": {},
        "server_sent_items": 4,
        "session_saved_items": 2,
        "input_saved_to_session": true,
        "turn_start_index": 1,
    });
    let state = RunState::from_json(legacy).unwrap();
    assert_eq!(state.current_agent_name, "Legacy");
    assert_eq!(state.current_turn, 3);
    assert_eq!(state.max_turns, Some(8));
    assert_eq!(state.generated_items.len(), 1);
    assert_eq!(state.server_sent_items, 4);
    assert_eq!(state.session_saved_items, 2);
    assert!(state.input_saved_to_session);
    assert_eq!(state.turn_start_index, 1);
    assert_eq!(state.schema_version, CURRENT_SCHEMA_VERSION);
}

#[test]
fn rejects_an_unknown_schema_version() {
    let bad = json!({"$schemaVersion": "9.9", "current_agent": {"name": "A"}});
    let err = RunState::from_json(bad).unwrap_err().to_string();
    assert!(err.contains("not supported"), "{err}");
}

#[test]
fn rust_only_state_survives_the_round_trip() {
    let mut state = sample_state();
    state.server_sent_items = 3;
    state.server_input_sent = true;
    state.session_saved_items = 2;
    state.input_saved_to_session = true;
    state.turn_start_index = 1;
    state.nested_agent_tool_states = [("call_1".to_string(), Value::String("nested".into()))]
        .into_iter()
        .collect();
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    assert_eq!(restored.server_sent_items, 3);
    assert!(restored.server_input_sent);
    assert_eq!(restored.session_saved_items, 2);
    assert!(restored.input_saved_to_session);
    assert_eq!(restored.turn_start_index, 1);
    assert_eq!(restored.nested_agent_tool_states.len(), 1);
}

#[test]
fn reads_python_tool_search_items_and_hosted_mcp_interruptions() {
    let wire = json!({
        "$schemaVersion": "1.21",
        "current_turn": 1,
        "current_agent": {"name": "A"},
        "original_input": [],
        "generated_items": [
            {
                "type": "tool_search_call_item",
                "raw_item": {"type": "tool_search_call", "id": "ts_1"},
                "agent": {"name": "A"},
            },
            {
                "type": "tool_search_output_item",
                "raw_item": {"type": "tool_search_output", "id": "ts_1"},
                "agent": {"name": "A"},
            },
        ],
        "context": {"approvals": {}, "function_tool_approvals": []},
        "current_step": {
            "type": "next_step_interruption",
            "data": {
                "interruptions": [{
                    "type": "tool_approval_item",
                    "raw_item": {
                        "type": "mcp_approval_request",
                        "id": "mcpr_1",
                        "name": "search",
                        "server_label": "docs",
                        "arguments": "{}",
                    },
                    "agent": {"name": "A"},
                    "tool_name": "docs:search",
                }],
            },
        },
    });
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.generated_items.len(), 2, "tool search items are kept");
    assert_eq!(state.generated_items[0].item_type(), "tool_call_item");
    assert_eq!(
        state.generated_items[1].item_type(),
        "tool_call_output_item"
    );
    assert_eq!(state.interruptions.len(), 1);
    assert_eq!(state.interruptions[0].raw_item.call_id, "mcpr_1");
    assert_eq!(state.interruptions[0].tool_name, "docs:search");
}

#[test]
fn an_unreadable_item_is_skipped_like_python() {
    let wire = json!({
        "$schemaVersion": "1.21",
        "current_agent": {"name": "A"},
        "generated_items": [{
            "type": "some_future_item",
            "raw_item": {},
            "agent": {"name": "A"},
        }],
    });
    // Python's reader has no branch for an unknown item type and drops it.
    assert!(
        RunState::from_json(wire)
            .unwrap()
            .generated_items
            .is_empty()
    );
}

#[test]
fn an_agent_agnostic_approval_applies_to_every_agent() {
    let wire = json!({
        "$schemaVersion": "1.17",
        "current_agent": {"name": "A"},
        "context": {"approvals": {"shared_tool": {"approved": true, "rejected": false}}},
    });
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.approvals.status("A", "shared_tool", "c1"), Some(true));
    assert_eq!(state.approvals.status("B", "shared_tool", "c1"), Some(true));
    assert_eq!(state.approvals.status("A", "other_tool", "c1"), None);
}
