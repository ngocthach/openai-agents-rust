//! Ports of `tests/test_run_context_approvals.py` cases whose request id is not a string.
//! A hosted-call wrapper keeps the provider's payload as JSON, so a malformed id can arrive
//! there; the wrapper's own call id must never stand in for it.

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::{AgentsError, RunContext};
use serde_json::{Value, json};

fn wrapped(provider_id: Value) -> ToolApprovalItem {
    let mut raw = FunctionCall::new("lookup_account", "{}");
    raw.call_id = "outer-id".into();
    raw.extra.insert("type".into(), json!("hosted_tool_call"));
    raw.extra.insert(
        "provider_data".into(),
        json!({
            "type": "mcp_approval_request",
            "id": provider_id,
            "server_label": "server-a",
            "name": "lookup_account",
        }),
    );
    ToolApprovalItem {
        agent: "test-agent".into(),
        raw_item: raw,
        tool_name: "lookup_account".into(),
        tool_origin: None,
    }
}

/// `test_hosted_mcp_provider_invalid_request_id_does_not_fall_back_to_outer_id`.
#[test]
fn a_non_string_provider_request_id_does_not_fall_back_to_the_outer_id() {
    for approve in [true, false] {
        let ctx = RunContext::new(());
        let malformed = wrapped(json!(123));
        let error = if approve {
            ctx.try_approve_tool(&malformed, false)
        } else {
            ctx.try_reject_tool(&malformed, true, Some("must not persist".into()))
        }
        .unwrap_err();
        match error {
            AgentsError::UserError { message, .. } => {
                assert!(message.contains("non-empty request id"), "{message}")
            }
            other => panic!("expected a UserError, got {other:?}"),
        }
        assert!(ctx.export_approvals().0.is_empty());
    }
}
