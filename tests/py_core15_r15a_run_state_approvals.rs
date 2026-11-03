//! Ports of cases of the Python SDK's `tests/test_run_state.py` that the earlier audits left as
//! Python-only: a decision for an invocation identity that two pending approvals share.

use openai_agents_sdk::RunState;
use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION;
use serde_json::json;

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

/// `test_ambiguous_current_approval_identity_fails_closed`: two pending approvals of one agent
/// that carry the same invocation identity make a decision ambiguous. Approving or rejecting
/// either one fails and records nothing.
#[test]
fn ambiguous_current_approval_identity_fails_closed() {
    for approve in [true, false] {
        let mut state = RunState {
            schema_version: CURRENT_SCHEMA_VERSION.to_string(),
            current_agent_name: "AmbiguousCurrentAgent".into(),
            context: Some(json!({})),
            ..Default::default()
        };
        let first = approval("AmbiguousCurrentAgent", "toolA", "shared-current", "{}");
        let second = first.clone();
        state.interruptions = vec![first, second];

        for selected in [
            state.get_interruptions()[1].clone(),
            state.interruptions[1].clone(),
        ] {
            let error = if approve {
                state.try_approve(&selected, false)
            } else {
                state.try_reject(&selected, false, None)
            }
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("multiple current pending approvals"),
                "{error}"
            );
            assert_eq!(
                state
                    .approvals
                    .status("AmbiguousCurrentAgent", "toolA", "shared-current"),
                None
            );
        }
    }
}

/// The unambiguous case still records the decision.
#[test]
fn a_single_pending_approval_is_decided() {
    let mut state = RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "Agent".into(),
        context: Some(json!({})),
        ..Default::default()
    };
    let item = approval("Agent", "toolA", "only", "{}");
    state.interruptions = vec![item.clone()];
    state.try_approve(&item, false).unwrap();
    assert_eq!(state.approvals.status("Agent", "toolA", "only"), Some(true));
}
