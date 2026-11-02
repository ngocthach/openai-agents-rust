//! Port of `test_deferred_top_level_permanent_approval_does_not_alias_to_bare_name` of
//! `tests/test_run_context_approvals.py`: a permanent approval of a deferred top-level tool (a
//! call whose namespace equals the tool name) is recorded under the `deferred_top_level:<name>`
//! key. It never authorizes a visible tool with the same bare name, and it does cover later
//! calls of the deferred tool itself.

use openai_agents_sdk::RunContext;
use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use serde_json::json;

const AGENT: &str = "test-agent";

/// A deferred top-level approval item shaped the way the runner builds one.
fn deferred_item(call_id: &str, name: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(name, "{}");
    raw.call_id = call_id.to_string();
    raw.status = Some("completed".into());
    raw.extra.insert("namespace".into(), json!(name));
    ToolApprovalItem {
        agent: AGENT.into(),
        raw_item: raw,
        tool_name: name.to_string(),
        tool_origin: None,
    }
}

/// Python: `test_deferred_top_level_permanent_approval_does_not_alias_to_bare_name`.
#[test]
fn deferred_top_level_permanent_approval_does_not_alias_to_bare_name() {
    let ctx = RunContext::new(());
    let item = deferred_item("call-weather", "get_weather");

    ctx.approve_tool(&item, true);

    assert_eq!(
        ctx.is_tool_approved(AGENT, "get_weather", "call-weather-2"),
        None
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "deferred_top_level:get_weather", "future"),
        Some(true)
    );
    assert_eq!(
        ctx.approval_status_for_item(&item, "call-weather-2")
            .approved,
        Some(true)
    );
}
