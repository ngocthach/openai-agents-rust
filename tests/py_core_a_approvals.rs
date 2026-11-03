//! Ports of the portable cases of `test_run_context_approvals.py` and the approval half of
//! `test_run_state.py`.
//!
//! Rust keys a decision by `(agent, tool name)` where the tool name is the qualified one
//! (`namespace.tool`, or `server_label:tool` for hosted MCP). The Python cases that rely on
//! Python's separate hosted-MCP identity records, deferred-top-level lookup keys and legacy
//! bare-name aliasing have no Rust counterpart and are listed in the parity report.

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::{RunContext, RunState};

fn item(agent: &str, tool: &str, call_id: &str) -> ToolApprovalItem {
    let mut call = FunctionCall::new(tool, "{}");
    call.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: call,
        tool_name: tool.into(),
        tool_origin: None,
    }
}

#[test]
fn a_permanent_approval_covers_later_calls_of_the_same_tool_and_agent() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&item("agent", "lookup", "call-1"), true);

    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-1"),
        Some(true)
    );
    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-2"),
        Some(true)
    );
    assert_eq!(ctx.is_tool_approved("other", "lookup", "call-2"), None);
    assert_eq!(ctx.is_tool_approved("agent", "other_tool", "call-2"), None);
}

#[test]
fn a_per_call_approval_covers_only_that_call() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&item("agent", "lookup", "call-1"), false);

    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-1"),
        Some(true)
    );
    assert_eq!(ctx.is_tool_approved("agent", "lookup", "call-2"), None);
}

#[test]
fn the_latest_decision_wins_for_a_call_id() {
    let ctx = RunContext::new(());
    let call = item("agent", "lookup", "call-1");
    ctx.approve_tool(&call, false);
    ctx.reject_tool(&call, false, None);
    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-1"),
        Some(false)
    );

    ctx.approve_tool(&call, false);
    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-1"),
        Some(true)
    );
}

#[test]
fn a_permanent_rejection_message_applies_to_later_calls() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &item("agent", "lookup", "call-1"),
        true,
        Some("server-a denied".into()),
    );

    assert_eq!(
        ctx.rejection_message("agent", "lookup", "call-2")
            .as_deref(),
        Some("server-a denied")
    );
    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "call-2"),
        Some(false)
    );
    assert_eq!(ctx.rejection_message("other", "lookup", "call-2"), None);
}

#[test]
fn an_exact_call_decision_beats_the_sticky_decision_of_the_same_tool() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&item("agent", "lookup", "first"), true);
    ctx.reject_tool(
        &item("agent", "lookup", "second"),
        false,
        Some("hosted denial".into()),
    );

    assert_eq!(
        ctx.is_tool_approved("agent", "lookup", "second"),
        Some(false)
    );
    assert_eq!(ctx.is_tool_approved("agent", "lookup", "later"), Some(true));
    assert_eq!(
        ctx.rejection_message("agent", "lookup", "second")
            .as_deref(),
        Some("hosted denial")
    );
}

#[test]
fn a_namespaced_decision_does_not_apply_to_the_bare_tool_name() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&item("agent", "billing.lookup_account", "call-1"), true);

    assert_eq!(
        ctx.is_tool_approved("agent", "billing.lookup_account", "call-2"),
        Some(true)
    );
    assert_eq!(
        ctx.is_tool_approved("agent", "lookup_account", "call-2"),
        None
    );
}

#[test]
fn a_namespaced_rejection_message_does_not_apply_to_the_bare_tool_name() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &item("agent", "billing.lookup_account", "call-1"),
        true,
        Some("billing denied".into()),
    );

    assert_eq!(
        ctx.rejection_message("agent", "lookup_account", "call-2"),
        None
    );
    assert_eq!(
        ctx.rejection_message("agent", "billing.lookup_account", "call-2")
            .as_deref(),
        Some("billing denied")
    );
}

#[test]
fn get_approval_status_reports_decision_and_message_together() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &item("agent", "lookup", "call-1"),
        false,
        Some("nope".into()),
    );

    let status = ctx.get_approval_status("agent", "lookup", "call-1");
    assert_eq!(status.approved, Some(false));
    assert_eq!(status.rejection_message.as_deref(), Some("nope"));
    let undecided = ctx.get_approval_status("agent", "lookup", "call-2");
    assert_eq!(undecided.approved, None);
    assert_eq!(undecided.rejection_message, None);
}

#[test]
fn run_state_approvals_survive_a_serialized_round_trip() {
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "agent".into(),
        ..Default::default()
    };
    state.approve(&item("agent", "lookup", "call-1"), true);
    state.reject(
        &item("agent", "write", "call-9"),
        false,
        Some("not allowed".into()),
    );

    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let ctx = RunContext::new(());
    ctx.import_approvals(&restored.approvals);

    assert_eq!(ctx.is_tool_approved("agent", "lookup", "later"), Some(true));
    assert_eq!(
        ctx.is_tool_approved("agent", "write", "call-9"),
        Some(false)
    );
    assert_eq!(
        ctx.rejection_message("agent", "write", "call-9").as_deref(),
        Some("not allowed")
    );
}

#[test]
fn approving_a_rejected_call_clears_its_rejection_message_lookup() {
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "agent".into(),
        ..Default::default()
    };
    let call = item("agent", "write", "call-1");
    state.reject(&call, false, Some("no".into()));
    state.approve(&call, false);

    let ctx = RunContext::new(());
    ctx.import_approvals(&state.approvals);
    assert_eq!(ctx.is_tool_approved("agent", "write", "call-1"), Some(true));
}
