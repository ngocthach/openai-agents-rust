//! Port of the envelope, schema-version, approval-record and guardrail-result cases of the
//! Python SDK's `tests/test_run_state.py` (`TestRunState`, `TestRunContextApprovals`,
//! `TestRunStateEdgeCases` and the schema-version cases of `TestRunStateSerializationEdgeCases`).
//!
//! Python builds a `RunState` around a live agent and context wrapper; the Rust `RunState` is
//! a plain snapshot, so each case builds the snapshot directly and checks the same wire
//! payload, error message or restored decision.

use std::sync::Arc;

use openai_agents_sdk::items::{FunctionCall, InputItem, RunItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run_config::ReasoningItemIdPolicy;
use openai_agents_sdk::run_state::{
    CURRENT_SCHEMA_VERSION, ProcessedResponseSnapshot, RunState, SUPPORTED_SCHEMA_VERSIONS,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tracing::TraceState;
use openai_agents_sdk::{
    Agent, GuardrailFunctionOutput, InputGuardrailResult, OutputGuardrailResult, Runner,
    ToolGuardrailFunctionOutput, ToolInputGuardrailResult, ToolOutputGuardrailResult,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Helpers (local copies; shared test helpers are owned by other areas).
// ---------------------------------------------------------------------------

/// `make_state(agent, context=..., original_input=..., max_turns=...)`.
fn make_state(agent: &str, original_input: &str, max_turns: Option<usize>) -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: agent.to_string(),
        original_input: if original_input.is_empty() {
            Vec::new()
        } else {
            vec![InputItem::user(original_input)]
        },
        max_turns,
        context: Some(json!({})),
        ..Default::default()
    }
}

/// `make_tool_approval_item(agent, call_id=..., name=..., arguments=...)`.
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

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
}

fn err_text(result: openai_agents_sdk::Result<RunState>) -> String {
    match result {
        Ok(_) => panic!("expected the snapshot to be refused"),
        Err(e) => e.to_string(),
    }
}

/// A minimal Python-shaped envelope, as the schema-version cases build it by hand.
fn bare_envelope(version: Option<&str>) -> Value {
    let mut wire = json!({
        "original_input": "test",
        "current_agent": {"name": "TestAgent"},
        "context": {
            "context": {},
            "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "approvals": {},
        },
        "max_turns": 3,
        "current_turn": 0,
        "model_responses": [],
        "generated_items": [],
    });
    if let Some(version) = version {
        wire["$schemaVersion"] = json!(version);
    }
    wire
}

fn lexicographic_versions() -> String {
    let mut versions = SUPPORTED_SCHEMA_VERSIONS.to_vec();
    versions.sort_unstable();
    versions.join(", ")
}

// ---------------------------------------------------------------------------
// TestRunState: construction and envelope
// ---------------------------------------------------------------------------

#[test]
fn initializes_with_default_values() {
    let state = make_state("TestAgent", "input", Some(3));
    assert_eq!(state.current_turn, 0);
    assert_eq!(state.current_agent_name, "TestAgent");
    assert_eq!(
        state.original_input[0].text_content().as_deref(),
        Some("input")
    );
    assert_eq!(state.max_turns, Some(3));
    assert!(state.model_responses.is_empty());
    assert!(state.generated_items.is_empty());
    assert!(state.interruptions.is_empty());
    assert!(!state.pending_tool_processing);
}

#[test]
fn tool_use_tracker_snapshot_filters_non_strings() {
    let mut wire = make_state("TestAgent", "input", Some(3)).to_json().unwrap();
    wire["tool_use_tracker"] = json!({
        "agent1": ["tool1", "tool2"],
        "agent2": ["tool4", 456, "tool5"],
        "agent3": "not-a-list",
    });
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(
        restored.tool_use_tracker.get("agent1"),
        Some(&vec!["tool1".to_string(), "tool2".to_string()])
    );
    // The non-string tool is dropped, not the whole tracker.
    assert_eq!(
        restored.tool_use_tracker.get("agent2"),
        Some(&vec!["tool4".to_string(), "tool5".to_string()])
    );
    assert!(!restored.tool_use_tracker.contains_key("agent3"));
}

#[test]
fn to_json_and_to_string_produce_valid_json() {
    let state = make_state("Agent1", "input1", Some(2));
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["$schemaVersion"], json!(CURRENT_SCHEMA_VERSION));
    assert_eq!(json_data["current_turn"], json!(0));
    assert_eq!(json_data["current_agent"], json!({"name": "Agent1"}));
    // Rust keeps the input as items; Python reads both a string and a list.
    assert_eq!(
        json_data["original_input"],
        json!([{"role": "user", "content": "input1"}])
    );
    assert_eq!(json_data["max_turns"], json!(2));
    assert_eq!(json_data["generated_items"], json!([]));
    assert_eq!(json_data["model_responses"], json!([]));

    let str_data = state.to_json_string().unwrap();
    let parsed: Value = serde_json::from_str(&str_data).unwrap();
    assert_eq!(parsed, json_data);
}

#[test]
fn max_turns_none_round_trips() {
    let state = make_state("Agent1", "input1", None);
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["max_turns"], Value::Null);
    let restored = RunState::from_json(json_data).unwrap();
    assert_eq!(restored.max_turns, None);
}

#[tokio::test]
async fn reasoning_item_id_policy_survives_serialization() {
    let mut state = make_state("AgentReasoningPolicy", "input1", Some(2));
    state.reasoning_item_id_policy = Some(ReasoningItemIdPolicy::Omit);
    state.generated_items = vec![RunItem::new(
        "AgentReasoningPolicy",
        RunItemKind::Reasoning {
            raw_item: serde_json::from_value(json!({"id": "rs_state", "summary": []})).unwrap(),
        },
    )];

    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["reasoning_item_id_policy"], json!("omit"));

    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(
        restored.reasoning_item_id_policy,
        Some(ReasoningItemIdPolicy::Omit)
    );

    // The restored policy shapes the history a resumed run sends: the reasoning item goes
    // back without its id even though the resuming config keeps the default policy.
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("AgentReasoningPolicy").with_model_instance(model.clone());
    let mut restored = restored;
    restored.add_input("next").unwrap();
    Runner::run(&agent, restored).await.unwrap();
    let call = model.last_call().unwrap();
    let reasoning: Vec<&InputItem> = call
        .input
        .iter()
        .filter(|i| i.item_type() == Some("reasoning"))
        .collect();
    assert_eq!(reasoning.len(), 1);
    assert!(reasoning[0].0.get("id").is_none());
}

#[test]
fn trace_api_key_serialization_is_opt_in() {
    let mut state = make_state("Agent1", "input1", Some(2));
    state.trace_state = Some(TraceState {
        trace_id: Some("trace_0123456789abcdef0123456789abcdef".into()),
        workflow_name: Some("test".into()),
        tracing_api_key: Some("trace-key".into()),
        tracing_api_key_hash: openai_agents_sdk::tracing::hash_tracing_api_key(Some("trace-key")),
        ..Default::default()
    });

    let default_json = state.to_json().unwrap();
    assert!(default_json["trace"].is_object());
    assert!(default_json["trace"].get("tracing_api_key").is_none());
    let hash = default_json["trace"]["tracing_api_key_hash"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(!hash.is_empty());
    assert_ne!(hash, "trace-key");

    let opt_in_json = state.to_json_with_tracing_api_key().unwrap();
    assert_eq!(opt_in_json["trace"]["tracing_api_key"], json!("trace-key"));
    assert_eq!(opt_in_json["trace"]["tracing_api_key_hash"], json!(hash));

    let with_key = RunState::from_json(opt_in_json).unwrap();
    let trace = with_key.trace_state.unwrap();
    assert_eq!(trace.tracing_api_key.as_deref(), Some("trace-key"));
    assert_eq!(trace.tracing_api_key_hash.as_deref(), Some(hash.as_str()));

    let without_key = RunState::from_json(default_json).unwrap();
    let trace = without_key.trace_state.unwrap();
    assert_eq!(trace.tracing_api_key, None);
    assert_eq!(trace.tracing_api_key_hash.as_deref(), Some(hash.as_str()));
}

#[test]
fn throws_error_if_schema_version_is_missing_or_invalid() {
    let mut json_data = make_state("Agent1", "input1", Some(2)).to_json().unwrap();
    json_data.as_object_mut().unwrap().remove("$schemaVersion");
    let message = err_text(RunState::from_json_str(&json_data.to_string()));
    assert!(
        message.contains("Run state is missing schema version"),
        "{message}"
    );

    json_data["$schemaVersion"] = json!("0.1");
    let message = err_text(RunState::from_json_str(&json_data.to_string()));
    let expected = format!(
        "Run state schema version is not supported. Supported versions are: {}. New snapshots \
         are written as version {CURRENT_SCHEMA_VERSION}.",
        lexicographic_versions()
    );
    assert!(message.contains(&expected), "{message}");
}

// ---------------------------------------------------------------------------
// Approval decisions recorded on the state
// ---------------------------------------------------------------------------

#[test]
fn approve_updates_context_approvals_correctly() {
    let mut state = make_state("Agent2", "", Some(1));
    let item = approval("Agent2", "toolX", "cid123", "arguments");
    state.approve(&item, false);
    assert_eq!(
        state.approvals.status("Agent2", "toolX", "cid123"),
        Some(true)
    );
}

#[test]
fn returns_undefined_when_approval_status_is_unknown() {
    let state = make_state("Agent2", "", Some(1));
    assert_eq!(
        state.approvals.status("Agent2", "unknownTool", "cid999"),
        None
    );
}

#[test]
fn reject_updates_context_approvals_correctly() {
    let mut state = make_state("Agent3", "", Some(1));
    let item = approval("Agent3", "toolY", "cid456", "arguments");
    state.reject(&item, false, None);
    assert_eq!(
        state.approvals.status("Agent3", "toolY", "cid456"),
        Some(false)
    );
}

#[test]
fn reject_stores_rejection_message() {
    let mut state = make_state("AgentRejectMessage", "", Some(1));
    let item = approval("AgentRejectMessage", "toolY", "cid456", "arguments");
    state.reject(&item, false, Some("Denied by reviewer".into()));
    assert_eq!(
        state
            .approvals
            .rejection_message("AgentRejectMessage", "toolY", "cid456")
            .as_deref(),
        Some("Denied by reviewer")
    );
}

#[test]
fn reject_permanently_when_always_reject_option_is_passed() {
    let mut state = make_state("Agent4", "", Some(1));
    let item = approval("Agent4", "toolZ", "cid789", "arguments");
    state.reject(&item, true, None);
    assert_eq!(
        state.approvals.status("Agent4", "toolZ", "cid789"),
        Some(false)
    );
    assert_eq!(
        state.approvals.status("Agent4", "toolZ", "future-call"),
        Some(false)
    );
}

#[test]
fn rejection_is_scoped_to_call_ids() {
    let mut state = make_state("AgentRejectReuse", "", Some(1));
    let item = approval("AgentRejectReuse", "toolZ", "cid789", "arguments");
    state.reject(&item, false, None);
    assert_eq!(
        state
            .approvals
            .status("AgentRejectReuse", "toolZ", "cid789"),
        Some(false)
    );
    assert_eq!(
        state
            .approvals
            .status("AgentRejectReuse", "toolZ", "cid999"),
        None
    );
    assert_eq!(
        state
            .approvals
            .rejection_message("AgentRejectReuse", "toolZ", "cid999"),
        None
    );
}

#[test]
fn always_reject_reuses_rejection_message_for_future_calls() {
    let mut state = make_state("AgentStickyReject", "", Some(1));
    let item = approval("AgentStickyReject", "toolZ", "cid789", "arguments");
    state.reject(&item, true, Some(String::new()));
    assert_eq!(
        state
            .approvals
            .rejection_message("AgentStickyReject", "toolZ", "cid789")
            .as_deref(),
        Some("")
    );
    assert_eq!(
        state
            .approvals
            .rejection_message("AgentStickyReject", "toolZ", "cid999")
            .as_deref(),
        Some("")
    );
}

// TestRunContextApprovals

#[test]
fn approval_takes_precedence_over_rejection_when_both_sticky() {
    // Python: `approved=True, rejected=True` resolves to approved. A Rust record can only
    // hold one sticky flag (the last call wins), so read the Python payload directly.
    let mut wire = make_state("Agent", "input", Some(1)).to_json().unwrap();
    wire["$schemaVersion"] = json!("1.14");
    wire["context"]["approvals"] = json!({"tool": {"approved": true, "rejected": true}});
    wire["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(
        restored.approvals.status("Agent", "tool", "cid"),
        Some(true)
    );
}

#[test]
fn individual_approval_takes_precedence_over_individual_rejection() {
    let mut wire = make_state("Agent", "input", Some(1)).to_json().unwrap();
    wire["$schemaVersion"] = json!("1.14");
    wire["context"]["approvals"] = json!({"tool": {"approved": ["cid"], "rejected": ["cid"]}});
    wire["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(
        restored.approvals.status("Agent", "tool", "cid"),
        Some(true)
    );
}

#[test]
fn returns_none_when_no_approval_or_rejection() {
    let mut wire = make_state("Agent", "input", Some(1)).to_json().unwrap();
    wire["$schemaVersion"] = json!("1.14");
    wire["context"]["approvals"] = json!({"tool": {"approved": [], "rejected": []}});
    wire["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.approvals.status("Agent", "tool", "cid"), None);
}

// ---------------------------------------------------------------------------
// Guardrail results
// ---------------------------------------------------------------------------

#[test]
fn guardrail_results_round_trip() {
    let mut state = make_state("GuardrailAgent", "input", Some(1));
    state.input_guardrail_results = vec![InputGuardrailResult {
        guardrail_name: "input_guardrail".into(),
        output: GuardrailFunctionOutput {
            output_info: json!({"input": "info"}),
            tripwire_triggered: false,
        },
    }];
    state.output_guardrail_results = vec![OutputGuardrailResult {
        guardrail_name: "output_guardrail".into(),
        agent_name: "GuardrailAgent".into(),
        agent_output: json!("final"),
        output: GuardrailFunctionOutput {
            output_info: json!({"output": "info"}),
            tripwire_triggered: true,
        },
    }];

    let restored = roundtrip(&state);
    assert_eq!(restored.input_guardrail_results.len(), 1);
    let input = &restored.input_guardrail_results[0];
    assert_eq!(input.guardrail_name, "input_guardrail");
    assert!(!input.output.tripwire_triggered);
    assert_eq!(input.output.output_info, json!({"input": "info"}));

    assert_eq!(restored.output_guardrail_results.len(), 1);
    let output = &restored.output_guardrail_results[0];
    assert_eq!(output.guardrail_name, "output_guardrail");
    assert!(output.output.tripwire_triggered);
    assert_eq!(output.output.output_info, json!({"output": "info"}));
    assert_eq!(output.agent_output, json!("final"));
    assert_eq!(output.agent_name, "GuardrailAgent");
}

fn tool_output_result(name: &str, info: &str) -> ToolOutputGuardrailResult {
    ToolOutputGuardrailResult {
        guardrail_name: name.into(),
        tool_name: "tool".into(),
        output: ToolGuardrailFunctionOutput {
            output_info: json!(info),
            ..ToolGuardrailFunctionOutput::allow()
        },
    }
}

#[test]
fn tool_guardrail_results_round_trip() {
    let mut state = make_state("ToolGuardrailAgent", "input", Some(1));
    state.tool_input_guardrail_results = vec![ToolInputGuardrailResult {
        guardrail_name: "tool_input_guardrail".into(),
        tool_name: "tool".into(),
        output: ToolGuardrailFunctionOutput {
            output_info: json!({"input": "info"}),
            ..ToolGuardrailFunctionOutput::allow()
        },
    }];
    state.tool_output_guardrail_results = vec![ToolOutputGuardrailResult {
        guardrail_name: "tool_output_guardrail".into(),
        tool_name: "tool".into(),
        output: ToolGuardrailFunctionOutput {
            output_info: json!({"output": "info"}),
            ..ToolGuardrailFunctionOutput::allow()
        },
    }];

    let wire = state.to_json().unwrap();
    assert_eq!(
        wire["tool_input_guardrail_results"][0]["output"]["behavior"]["type"],
        json!("allow")
    );
    let restored = roundtrip(&state);
    let input = &restored.tool_input_guardrail_results[0];
    assert_eq!(input.guardrail_name, "tool_input_guardrail");
    assert_eq!(
        input.output.behavior,
        openai_agents_sdk::tool_guardrails::ToolGuardrailBehavior::Allow
    );
    assert_eq!(input.output.output_info, json!({"input": "info"}));
    let output = &restored.tool_output_guardrail_results[0];
    assert_eq!(output.guardrail_name, "tool_output_guardrail");
    assert_eq!(output.output.output_info, json!({"output": "info"}));
}

/// A snapshot parked with a processed response whose pending tool output guardrail results
/// start at `start`.
fn boundary_state(start: Option<usize>, results: Vec<ToolOutputGuardrailResult>) -> RunState {
    let mut state = make_state("ToolGuardrailAgent", "input", Some(3));
    state.interruptions = vec![approval("ToolGuardrailAgent", "tool", "call-1", "{}")];
    state.pending_tool_processing = true;
    state.last_processed_response = Some(ProcessedResponseSnapshot {
        tool_output_guardrail_result_start: start,
        ..Default::default()
    });
    state.tool_output_guardrail_results = results;
    state
}

#[test]
fn tool_guardrail_boundary_requires_its_schema_version() {
    for old_version in ["1.0", "1.18", "1.19"] {
        let state = boundary_state(
            Some(0),
            vec![tool_output_result("test_guardrail", "checkpoint")],
        );
        let mut snapshot: Value = serde_json::from_str(&state.to_json_string().unwrap()).unwrap();
        assert_eq!(snapshot["$schemaVersion"], json!(CURRENT_SCHEMA_VERSION));
        let restored = RunState::from_json(snapshot.clone()).unwrap();
        assert_eq!(
            restored
                .last_processed_response
                .as_ref()
                .unwrap()
                .tool_output_guardrail_result_start,
            Some(0)
        );
        assert_eq!(
            restored.tool_output_guardrail_results[0].output.output_info,
            json!("checkpoint")
        );

        snapshot["$schemaVersion"] = json!(old_version);
        // Legacy versions also predate agent-owned approvals; this case has none.
        snapshot["context"]["function_tool_approvals"] = json!([]);
        let message = err_text(RunState::from_json(snapshot.clone()));
        assert!(
            message.contains("tool output guardrail result boundary requires schema 1.20"),
            "{old_version}: {message}"
        );

        // Older snapshots did not distinguish pending results; they stay readable.
        snapshot["last_processed_response"]
            .as_object_mut()
            .unwrap()
            .remove("tool_output_guardrail_result_start");
        let legacy = RunState::from_json_str(&snapshot.to_string()).unwrap();
        assert_eq!(
            legacy
                .last_processed_response
                .as_ref()
                .unwrap()
                .tool_output_guardrail_result_start,
            None
        );
        assert_eq!(
            legacy.tool_output_guardrail_results[0].output.output_info,
            json!("checkpoint")
        );
    }
}

#[test]
fn tool_guardrail_boundary_rejects_invalid_restored_count() {
    for boundary in [json!(-1), json!(true), json!("0"), json!(3)] {
        let mut snapshot = boundary_state(None, Vec::new()).to_json().unwrap();
        snapshot["last_processed_response"]["tool_output_guardrail_result_start"] =
            boundary.clone();
        snapshot["tool_output_guardrail_results"] = json!([
            {
                "guardrail": {"type": "tool_output", "name": "prior"},
                "output": {"behavior": {"type": "allow"}, "outputInfo": "accepted"},
            },
            {
                "guardrail": {"type": "tool_output", "name": "pending"},
                "output": {"behavior": {"type": "allow"}, "outputInfo": "pending"},
            },
        ]);
        let message = err_text(RunState::from_json(snapshot));
        assert!(
            message.contains("Invalid tool output guardrail result boundary"),
            "{boundary}: {message}"
        );
    }
}

#[test]
fn tool_guardrail_boundary_cannot_shift_when_results_are_dropped() {
    let mut snapshot = boundary_state(None, Vec::new()).to_json().unwrap();
    snapshot["last_processed_response"]["tool_output_guardrail_result_start"] = json!(1);
    snapshot["tool_output_guardrail_results"] = json!([
        {
            "guardrail": {"type": "tool_input", "name": "wrong_kind"},
            "output": {"behavior": {"type": "allow"}, "outputInfo": "accepted"},
        },
        {
            "guardrail": {"type": "tool_output", "name": "same_name"},
            "output": {"behavior": {"type": "allow"}, "outputInfo": "pending-a"},
        },
        {
            "guardrail": {"type": "tool_output", "name": "same_name"},
            "output": {"behavior": {"type": "allow"}, "outputInfo": "pending-b"},
        },
    ]);
    let message = err_text(RunState::from_json_str(&snapshot.to_string()));
    assert!(
        message.contains("Invalid tool output guardrail result boundary"),
        "{message}"
    );

    snapshot["last_processed_response"]
        .as_object_mut()
        .unwrap()
        .remove("tool_output_guardrail_result_start");
    snapshot["$schemaVersion"] = json!("1.19");
    snapshot["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(snapshot).unwrap();
    let infos: Vec<Value> = restored
        .tool_output_guardrail_results
        .iter()
        .map(|r| r.output.output_info.clone())
        .collect();
    assert_eq!(infos, vec![json!("pending-a"), json!("pending-b")]);
}

#[test]
fn tool_guardrail_boundary_accepts_end_of_restored_results() {
    let state = boundary_state(
        Some(1),
        vec![tool_output_result("test_guardrail", "accepted")],
    );
    let restored = roundtrip(&state);
    assert_eq!(
        restored
            .last_processed_response
            .as_ref()
            .unwrap()
            .tool_output_guardrail_result_start,
        Some(1)
    );
    assert_eq!(
        restored.tool_output_guardrail_results[0].output.output_info,
        json!("accepted")
    );
}

// ---------------------------------------------------------------------------
// Generated items and the processed response
// ---------------------------------------------------------------------------

fn tool_call_item(agent: &str, name: &str, call_id: &str) -> RunItem {
    let mut call = FunctionCall::new(name, "{}");
    call.call_id = call_id.into();
    RunItem::new(
        agent,
        RunItemKind::ToolCall {
            raw_item: call.to_value(),
        },
    )
}

#[test]
fn generated_items_not_duplicated_by_last_processed_response() {
    let mut state = make_state("AgentDedup", "input", Some(2));
    let item = tool_call_item("AgentDedup", "get_weather", "call_1");
    state.generated_items = vec![item.clone()];
    state.interruptions = vec![approval("AgentDedup", "get_weather", "call_1", "{}")];
    state.pending_tool_processing = true;
    state.last_processed_response = Some(ProcessedResponseSnapshot {
        new_items: vec![item],
        ..Default::default()
    });

    let json_data = state.to_json().unwrap();
    let items = json_data["generated_items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["raw_item"]["call_id"], json!("call_1"));

    let restored = RunState::from_json(json_data).unwrap();
    assert_eq!(restored.generated_items.len(), 1);
}

#[test]
fn anonymous_tool_search_items_not_duplicated_across_round_trip() {
    let mut state = make_state("AgentToolSearchDedup", "input", Some(2));
    let call = RunItem::new(
        "AgentToolSearchDedup",
        RunItemKind::ToolCall {
            raw_item: json!({
                "type": "tool_search_call",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            }),
        },
    );
    let output = RunItem::new(
        "AgentToolSearchDedup",
        RunItemKind::ToolCallOutput {
            raw_item: json!({
                "type": "tool_search_output",
                "execution": "server",
                "status": "completed",
                "tools": [],
            }),
            output: Value::Null,
            custom_data: None,
        },
    );
    state.generated_items = vec![call.clone(), output.clone(), call.clone(), output.clone()];
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["generated_items"].as_array().unwrap().len(), 4);
    let restored = RunState::from_json(json_data).unwrap();
    let again = restored.to_json().unwrap();
    let raw_types: Vec<Value> = again["generated_items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["raw_item"]["type"].clone())
        .collect();
    assert_eq!(
        raw_types,
        vec![
            json!("tool_search_call"),
            json!("tool_search_output"),
            json!("tool_search_call"),
            json!("tool_search_output"),
        ]
    );
}

#[test]
fn from_string_reconstructs_state_for_simple_agent() {
    let mut state = make_state("Solo", "orig", Some(7)).with_context(json!({"a": 1}));
    state.current_turn = 5;
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(restored.max_turns, Some(7));
    assert_eq!(restored.current_turn, 5);
    assert_eq!(restored.current_agent_name, "Solo");
    assert_eq!(restored.context, Some(json!({"a": 1})));
    assert!(restored.generated_items.is_empty());
    assert!(restored.model_responses.is_empty());
}

#[test]
fn from_json_reconstructs_state() {
    let mut state =
        make_state("JsonAgent", "test input", Some(5)).with_context(json!({"test": "data"}));
    state.current_turn = 2;
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    assert_eq!(restored.max_turns, Some(5));
    assert_eq!(restored.current_turn, 2);
    assert_eq!(restored.current_agent_name, "JsonAgent");
    assert_eq!(restored.context, Some(json!({"test": "data"})));
}

#[test]
fn get_interruptions_returns_empty_when_no_interruptions() {
    assert!(
        make_state("Agent5", "", Some(1))
            .get_interruptions()
            .is_empty()
    );
}

#[test]
fn get_interruptions_returns_interruptions_when_present() {
    let mut state = make_state("Agent6", "", Some(1));
    let item = approval("Agent6", "toolA", "cid111", "args");
    state.interruptions = vec![item.clone()];
    let interruptions = state.get_interruptions();
    assert_eq!(interruptions.len(), 1);
    assert_eq!(interruptions[0].agent, "Agent6");
    assert_eq!(interruptions[0].tool_name, item.tool_name);
    assert_eq!(interruptions[0].raw_item, item.raw_item);
}

#[test]
fn get_interruptions_returns_detached_item_snapshots() {
    // Python's "pydantic" and "mapping" raw items are one typed `FunctionCall` in Rust; the
    // mapping variant's extra metadata rides in `extra`.
    for with_metadata in [false, true] {
        let mut state = make_state("SnapshotAgent", "input", Some(3));
        let mut item = approval(
            "SnapshotAgent",
            "toolA",
            "cid-snapshot",
            r#"{"value": "original"}"#,
        );
        if with_metadata {
            item.raw_item
                .extra
                .insert("metadata".into(), json!({"tags": ["original"]}));
        }
        state.interruptions = vec![item];

        let mut interruption = state.get_interruptions().remove(0);
        interruption.tool_name = "changed".into();
        interruption.raw_item.arguments = r#"{"value": "changed"}"#.into();
        if with_metadata {
            interruption.raw_item.extra["metadata"]["tags"]
                .as_array_mut()
                .unwrap()
                .push(json!("changed"));
        }

        let pending = state.get_interruptions().remove(0);
        assert_eq!(pending.agent, "SnapshotAgent");
        assert_eq!(pending.tool_name, "toolA");
        assert_eq!(pending.raw_item.arguments, r#"{"value": "original"}"#);
        if with_metadata {
            assert_eq!(
                pending.raw_item.extra["metadata"],
                json!({"tags": ["original"]})
            );
        }
    }
}

#[test]
fn get_interruptions_snapshots_can_apply_approval_decisions() {
    for approve in [true, false] {
        let mut state = make_state("DecisionAgent", "input", Some(3));
        state.interruptions = vec![approval("DecisionAgent", "toolA", "cid-decision", "{}")];
        let interruption = state.get_interruptions().remove(0);
        if approve {
            state.approve(&interruption, false);
        } else {
            state.reject(&interruption, false, None);
        }
        assert_eq!(
            state
                .approvals
                .status("DecisionAgent", "toolA", "cid-decision"),
            Some(approve)
        );
    }
}

// ---------------------------------------------------------------------------
// Schema versions (TestRunStateSerializationEdgeCases)
// ---------------------------------------------------------------------------

#[test]
fn from_json_missing_schema_version() {
    let message = err_text(RunState::from_json(bare_envelope(None)));
    assert!(
        message.contains("Run state is missing schema version"),
        "{message}"
    );
}

#[test]
fn from_json_unsupported_schema_version() {
    let next = {
        let (major, minor) = CURRENT_SCHEMA_VERSION.split_once('.').unwrap();
        format!("{major}.{}", minor.parse::<u32>().unwrap() + 1)
    };
    for version in [next.as_str(), "2.0", "9.9"] {
        let message = err_text(RunState::from_json(bare_envelope(Some(version))));
        assert!(
            message.contains("Run state schema version is not supported"),
            "{version}: {message}"
        );
    }
}

#[test]
fn from_json_checks_schema_before_sandbox_envelope() {
    let wire = json!({"$schemaVersion": "9.9", "sandbox": ["future-sandbox-value"]});
    let original = wire.clone();
    let message = err_text(RunState::from_json(wire.clone()));
    assert!(
        message.contains("Run state schema version is not supported"),
        "{message}"
    );
    assert_eq!(wire, original);
}

#[test]
fn malformed_schema_shape_redacts_public_errors() {
    let sentinel = "malformed-schema-secret";
    let cases = [
        (
            json!([{"secret_access_key": sentinel}]),
            "must be an object",
        ),
        (
            json!({"$schemaVersion": {"value": sentinel}}),
            "schema version has an invalid type",
        ),
        (
            json!({"$schemaVersion": sentinel}),
            "schema version is not supported",
        ),
    ];
    for (payload, expected) in cases {
        for from_string in [false, true] {
            let message = if from_string {
                err_text(RunState::from_json_str(&payload.to_string()))
            } else {
                err_text(RunState::from_json(payload.clone()))
            };
            assert!(message.contains(expected), "{message}");
            assert!(!message.contains(sentinel), "{message}");
        }
    }
}

#[test]
fn json_decode_error_handling() {
    let message = err_text(RunState::from_json_str("{ invalid json"));
    assert!(
        message.contains("Failed to parse run state JSON"),
        "{message}"
    );
}

#[test]
fn from_json_accepts_previous_schema_version() {
    let mut wire = bare_envelope(Some("1.0"));
    wire["context"]["context"] = json!({"foo": "bar"});
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.current_agent_name, "TestAgent");
    assert_eq!(restored.context, Some(json!({"foo": "bar"})));
}

fn program_state() -> RunState {
    let program = json!({
        "id": "program_item",
        "call_id": "call_program",
        "code": "lookup()",
        "fingerprint": "fingerprint",
        "type": "program",
    });
    let function_call = json!({
        "id": "function_item",
        "call_id": "call_function",
        "name": "lookup",
        "arguments": "{}",
        "caller": {"type": "program", "caller_id": "call_program"},
        "type": "function_call",
    });
    let program_output = json!({
        "id": "program_output_item",
        "call_id": "call_program",
        "result": "done",
        "status": "completed",
        "type": "program_output",
    });
    let mut state = make_state("TestAgent", "test", Some(3));
    state.model_responses = vec![openai_agents_sdk::items::ModelResponse {
        output: [&program, &function_call, &program_output]
            .into_iter()
            .map(|v| openai_agents_sdk::items::OutputItem::from_value(v.clone()))
            .collect(),
        usage: Default::default(),
        response_id: Some("response_1".into()),
        request_id: None,
        raw_usage: None,
    }];
    state.generated_items = vec![
        RunItem::new("TestAgent", RunItemKind::ToolCall { raw_item: program }),
        RunItem::new(
            "TestAgent",
            RunItemKind::ToolCall {
                raw_item: function_call,
            },
        ),
        RunItem::new(
            "TestAgent",
            RunItemKind::ToolCallOutput {
                raw_item: program_output,
                output: json!("done"),
                custom_data: None,
            },
        ),
    ];
    state
}

#[test]
fn programmatic_tool_calling_round_trip_uses_current_schema() {
    let state = program_state();
    let json_data = state.to_json().unwrap();
    assert_eq!(json_data["$schemaVersion"], json!(CURRENT_SCHEMA_VERSION));
    let restored = RunState::from_json(json_data).unwrap();
    let output: Vec<Value> = restored.model_responses[0]
        .output
        .iter()
        .map(|o| o.to_value())
        .collect();
    assert_eq!(output[0]["type"], json!("program"));
    assert_eq!(
        output[1]["caller"],
        json!({"type": "program", "caller_id": "call_program"})
    );
    assert_eq!(output[2]["type"], json!("program_output"));
    assert_eq!(restored.generated_items, state.generated_items);
}

#[test]
fn programmatic_tool_calling_round_trip_preserves_mapping_items() {
    // Program items without the optional `fingerprint`/`status` fields come back verbatim.
    let program = json!({
        "id": "program_item", "call_id": "call_program", "code": "lookup()", "type": "program",
    });
    let program_output = json!({
        "id": "program_output_item", "call_id": "call_program", "result": "done",
        "type": "program_output",
    });
    let mut state = make_state("TestAgent", "test", Some(3));
    state.model_responses = vec![openai_agents_sdk::items::ModelResponse {
        output: vec![
            openai_agents_sdk::items::OutputItem::from_value(program.clone()),
            openai_agents_sdk::items::OutputItem::from_value(program_output.clone()),
        ],
        usage: Default::default(),
        response_id: Some("response_1".into()),
        request_id: None,
        raw_usage: None,
    }];
    state.generated_items = vec![
        RunItem::new(
            "TestAgent",
            RunItemKind::ToolCall {
                raw_item: program.clone(),
            },
        ),
        RunItem::new(
            "TestAgent",
            RunItemKind::ToolCallOutput {
                raw_item: program_output.clone(),
                output: json!("done"),
                custom_data: None,
            },
        ),
    ];
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let output: Vec<Value> = restored.model_responses[0]
        .output
        .iter()
        .map(|o| o.to_value())
        .collect();
    assert_eq!(output, vec![program.clone(), program_output.clone()]);
    match &restored.generated_items[0].kind {
        RunItemKind::ToolCall { raw_item } => assert_eq!(raw_item, &program),
        other => panic!("{other:?}"),
    }
    match &restored.generated_items[1].kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => assert_eq!(raw_item, &program_output),
        other => panic!("{other:?}"),
    }
}

#[test]
fn previous_schema_rejects_programmatic_tool_calling_items() {
    let mut json_data = program_state().to_json().unwrap();
    json_data["$schemaVersion"] = json!("1.12");
    json_data["context"]["function_tool_approvals"] = json!([]);
    let message = err_text(RunState::from_json(json_data));
    assert!(
        message.contains("Programmatic Tool Calling requires schema version"),
        "{message}"
    );
}

#[test]
fn schema_1_13_accepts_programmatic_tool_calling_items() {
    let mut json_data = program_state().to_json().unwrap();
    json_data["$schemaVersion"] = json!("1.13");
    json_data["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(json_data).unwrap();
    assert_eq!(
        restored.model_responses[0].output[0].to_value()["type"],
        json!("program")
    );
}

#[test]
fn previous_schema_ignores_program_like_arbitrary_context() {
    let context = json!({"payload": {"type": "program", "call_id": "not-a-run-item"}});
    let mut json_data = make_state("TestAgent", "test", Some(3))
        .with_context(context.clone())
        .to_json()
        .unwrap();
    json_data["$schemaVersion"] = json!("1.12");
    json_data["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(json_data).unwrap();
    assert_eq!(restored.context, Some(context));
}

#[test]
fn supported_schema_versions_match_released_boundary() {
    let expected = [
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "1.10",
        "1.11",
        "1.12",
        "1.13",
        "1.14",
        "1.15",
        "1.16",
        "1.17",
        "1.18",
        "1.19",
        "1.20",
        CURRENT_SCHEMA_VERSION,
    ];
    assert_eq!(SUPPORTED_SCHEMA_VERSIONS.to_vec(), expected.to_vec());
    assert!(SUPPORTED_SCHEMA_VERSIONS.contains(&CURRENT_SCHEMA_VERSION));
}

#[test]
fn agent_owned_approvals_require_schema_1_18() {
    // `_build_run_state_from_json`: agent-owned decisions are refused below the schema that
    // introduced them, so a decision cannot silently lose its owner.
    let mut state = make_state("Agent", "input", Some(3));
    state.approve(&approval("Agent", "tool1", "cid1", "{}"), true);
    let mut json_data = state.to_json().unwrap();
    json_data["$schemaVersion"] = json!("1.17");
    let message = err_text(RunState::from_json(json_data));
    assert!(
        message.contains("Agent-owned function approvals require RunState schema 1.18"),
        "{message}"
    );
}

#[test]
fn from_json_accepts_schema_version_1_5_without_sandbox_payload() {
    let mut wire = bare_envelope(Some("1.5"));
    wire["context"]["context"] = json!({"foo": "bar"});
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.current_agent_name, "TestAgent");
    assert_eq!(restored.context, Some(json!({"foo": "bar"})));
}
