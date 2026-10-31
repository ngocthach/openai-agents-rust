//! Ports of `tests/test_run_context_wrapper.py` and the behavioural cases of
//! `tests/test_call_model_input_filter_unit.py`.
//!
//! Python's `RunContextWrapper` keys decisions by `(tool name, call id)` and takes no agent. The
//! Rust `RunContext` also scopes them by agent name, so every query here passes the agent name
//! of the item it was built from.
//!
//! Python-only cases (not ported):
//! - `_to_str_or_none` with an object whose `__str__` raises, and the raw-dict tool-name and
//!   call-id fallbacks: Rust `ToolApprovalItem` is typed, so the name and call id always exist.
//! - a falsy `ToolApprovalItem` (`__bool__` returning False): Rust has no truthiness.
//! - the `"unknown_tool"` fallback for an item without a name: `tool_name` is a required `String`.
//! - the positional `ToolApprovalItem(agent, raw, name, type)` constructor: Rust uses struct
//!   fields.
//! - the call-model filter returning a non-`ModelInputData` value: the Rust closure's return type
//!   is checked at compile time.

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, ModelInputData, RunContext, RunOptions, Runner};
use std::sync::Arc;

const AGENT: &str = "test-agent";

fn approval(call_id: &str) -> ToolApprovalItem {
    approval_for("tool_call", call_id)
}

fn approval_for(name: &str, call_id: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(name, "{}");
    raw.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: AGENT.into(),
        raw_item: raw,
        tool_name: name.to_string(),
        tool_origin: None,
    }
}

fn ctx() -> RunContext<()> {
    RunContext::new(())
}

#[test]
fn scopes_approvals_to_call_ids() {
    let ctx = ctx();
    ctx.approve_tool(&approval("call-1"), false);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "tool_call", "call-1"),
        Some(true)
    );
    // A different call ID should require a fresh approval.
    assert_eq!(ctx.is_tool_approved(AGENT, "tool_call", "call-2"), None);
}

#[test]
fn scopes_rejections_to_call_ids() {
    let ctx = ctx();
    ctx.reject_tool(&approval("call-1"), false, None);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "tool_call", "call-1"),
        Some(false)
    );
    assert_eq!(ctx.is_tool_approved(AGENT, "tool_call", "call-2"), None);
}

#[test]
fn honors_global_approval_and_rejection() {
    let ctx = ctx();
    ctx.approve_tool(&approval("call-1"), true);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "tool_call", "call-2"),
        Some(true)
    );

    ctx.reject_tool(&approval("call-1"), true, None);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "tool_call", "call-3"),
        Some(false)
    );
}

#[test]
fn sticky_decision_from_pending_item_applies_to_later_calls() {
    let item = approval("call-1");

    let approved = ctx();
    approved.approve_tool(&item, true);
    let status = approved.approval_status_for_item(&item, "call-2");
    assert_eq!(status.approved, Some(true));

    let rejected = ctx();
    rejected.reject_tool(&item, true, Some("Denied".into()));
    let status = rejected.approval_status_for_item(&item, "call-2");
    assert_eq!(status.approved, Some(false));
    assert_eq!(status.rejection_message.as_deref(), Some("Denied"));
}

#[test]
fn stores_per_call_rejection_messages() {
    let ctx = ctx();
    ctx.reject_tool(&approval("call-1"), false, Some("Denied by policy".into()));
    assert_eq!(
        ctx.rejection_message(AGENT, "tool_call", "call-1")
            .as_deref(),
        Some("Denied by policy")
    );
    assert_eq!(ctx.rejection_message(AGENT, "tool_call", "call-2"), None);
}

#[test]
fn stores_sticky_rejection_messages_for_always_reject() {
    let ctx = ctx();
    ctx.reject_tool(&approval("call-1"), true, Some(String::new()));
    assert_eq!(
        ctx.rejection_message(AGENT, "tool_call", "call-1")
            .as_deref(),
        Some("")
    );
    assert_eq!(
        ctx.rejection_message(AGENT, "tool_call", "call-2")
            .as_deref(),
        Some("")
    );
}

#[test]
fn approval_clears_rejection_message() {
    let ctx = ctx();
    ctx.reject_tool(&approval("call-1"), false, Some("Denied by policy".into()));
    ctx.approve_tool(&approval("call-1"), false);
    assert_eq!(ctx.rejection_message(AGENT, "tool_call", "call-1"), None);
}

#[test]
fn approval_item_exposes_name_and_namespace() {
    let mut item = approval_for("lookup_account", "call-1");
    item.raw_item
        .extra
        .insert("namespace".into(), serde_json::json!("billing"));
    assert_eq!(item.name(), "lookup_account");
    assert_eq!(item.tool_namespace(), Some("billing"));
}

#[test]
fn exact_call_decisions_override_sticky_defaults() {
    let approved = ctx();
    approved.approve_tool(&approval("approve-sticky"), true);
    approved.reject_tool(
        &approval("approve-exception"),
        false,
        Some("denied by user".into()),
    );

    assert_eq!(
        approved.is_tool_approved(AGENT, "tool_call", "approve-exception"),
        Some(false)
    );
    assert_eq!(
        approved
            .rejection_message(AGENT, "tool_call", "approve-exception")
            .as_deref(),
        Some("denied by user")
    );
    assert_eq!(
        approved.is_tool_approved(AGENT, "tool_call", "approve-other"),
        Some(true)
    );

    let rejected = ctx();
    rejected.reject_tool(
        &approval("reject-sticky"),
        true,
        Some("denied by default".into()),
    );
    rejected.approve_tool(&approval("reject-exception"), false);

    assert_eq!(
        rejected.is_tool_approved(AGENT, "tool_call", "reject-exception"),
        Some(true)
    );
    assert_eq!(
        rejected.rejection_message(AGENT, "tool_call", "reject-exception"),
        None
    );
    assert_eq!(
        rejected.is_tool_approved(AGENT, "tool_call", "reject-other"),
        Some(false)
    );
    assert_eq!(
        rejected
            .rejection_message(AGENT, "tool_call", "reject-other")
            .as_deref(),
        Some("denied by default")
    );
}

#[test]
fn matching_exact_call_decisions_preserve_sticky_defaults() {
    let approved = ctx();
    approved.approve_tool(&approval("approve-sticky"), true);
    approved.approve_tool(&approval("approve-match"), false);
    assert_eq!(
        approved.is_tool_approved(AGENT, "tool_call", "approve-other"),
        Some(true)
    );

    let rejected = ctx();
    rejected.reject_tool(&approval("reject-sticky"), true, None);
    rejected.reject_tool(&approval("reject-match"), false, None);
    assert_eq!(
        rejected.is_tool_approved(AGENT, "tool_call", "reject-other"),
        Some(false)
    );
}

#[test]
fn exact_call_reversals_keep_other_calls_on_sticky_default() {
    let approved = ctx();
    approved.approve_tool(&approval("approve-sticky"), true);
    approved.reject_tool(&approval("approve-exception"), false, Some("denied".into()));
    approved.approve_tool(&approval("approve-exception"), false);

    assert_eq!(
        approved.is_tool_approved(AGENT, "tool_call", "approve-exception"),
        Some(true)
    );
    assert_eq!(
        approved.rejection_message(AGENT, "tool_call", "approve-exception"),
        None
    );
    assert_eq!(
        approved.is_tool_approved(AGENT, "tool_call", "approve-other"),
        Some(true)
    );

    let rejected = ctx();
    rejected.reject_tool(
        &approval("reject-sticky"),
        true,
        Some("denied by default".into()),
    );
    rejected.approve_tool(&approval("reject-exception"), false);
    rejected.reject_tool(
        &approval("reject-exception"),
        false,
        Some("denied exactly".into()),
    );

    assert_eq!(
        rejected.is_tool_approved(AGENT, "tool_call", "reject-exception"),
        Some(false)
    );
    assert_eq!(
        rejected
            .rejection_message(AGENT, "tool_call", "reject-exception")
            .as_deref(),
        Some("denied exactly")
    );
    assert_eq!(
        rejected.is_tool_approved(AGENT, "tool_call", "reject-other"),
        Some(false)
    );
    assert_eq!(
        rejected
            .rejection_message(AGENT, "tool_call", "reject-other")
            .as_deref(),
        Some("denied by default")
    );
}

fn assistant_ok_model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("ok")]);
    model
}

#[tokio::test]
async fn call_model_input_filter_sync_non_streamed_replaces_input_and_instructions() {
    use openai_agents_sdk::run_config::RunConfig;
    let model = assistant_ok_model();
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let cfg = RunConfig::new().with_call_model_input_filter(|data| async move {
        let mut input = data.model_data.input.clone();
        input.push(openai_agents_sdk::InputItem(
            serde_json::json!({"content": "added-sync", "role": "user"}),
        ));
        Ok(ModelInputData {
            input,
            instructions: Some("filtered-sync".into()),
        })
    });
    Runner::run_with_options(&agent, "start", RunOptions::new(()).run_config(cfg))
        .await
        .unwrap();

    let call = model.last_call().unwrap();
    assert_eq!(call.system_instructions.as_deref(), Some("filtered-sync"));
    assert_eq!(call.input.len(), 2);
    assert_eq!(call.input.last().unwrap().0["content"], "added-sync");
}

#[tokio::test]
async fn call_model_input_filter_async_streamed_replaces_input_and_instructions() {
    use futures::StreamExt;
    use openai_agents_sdk::run_config::RunConfig;
    let model = assistant_ok_model();
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let cfg = RunConfig::new().with_call_model_input_filter(|data| async move {
        let mut input = data.model_data.input.clone();
        input.push(openai_agents_sdk::InputItem(
            serde_json::json!({"content": "added-async", "role": "user"}),
        ));
        Ok(ModelInputData {
            input,
            instructions: Some("filtered-async".into()),
        })
    });
    let mut result =
        Runner::run_streamed_with_options(&agent, "start", RunOptions::new(()).run_config(cfg));
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }

    let call = model.last_call().unwrap();
    assert_eq!(call.system_instructions.as_deref(), Some("filtered-async"));
    assert_eq!(call.input.len(), 2);
    assert_eq!(call.input.last().unwrap().0["content"], "added-async");
}
