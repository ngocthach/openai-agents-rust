//! Ports of `tests/test_run_context_approvals.py` and `tests/test_run_internal_approvals.py`.
//!
//! Python's name-only queries (`is_tool_approved(tool, call_id)`) map to the Rust queries with
//! the agent spelled out; `get_approval_status(..., existing_pending=item)` maps to
//! [`RunContext::approval_status_for_item`]. Decisions restored with `_rebuild_approvals` are
//! restored here through `RunState::from_json`, the public path that calls it.

#![allow(clippy::field_reassign_with_default)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::items::{FunctionCall, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunContext, RunState, Runner};
use serde_json::{Value, json};

const AGENT: &str = "test-agent";

fn fn_item(call_id: &str, name: &str, namespace: Option<&str>) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(name, "{}");
    raw.call_id = call_id.to_string();
    raw.status = Some("completed".into());
    if let Some(ns) = namespace {
        raw.extra.insert("namespace".into(), json!(ns));
    }
    ToolApprovalItem {
        agent: AGENT.into(),
        raw_item: raw,
        tool_name: match namespace {
            Some(ns) => format!("{ns}.{name}"),
            None => name.to_string(),
        },
        tool_origin: None,
    }
}

/// A hosted MCP approval item shaped the way the runner builds one.
fn hosted_raw(request_id: &str, server_label: Option<&str>, tool: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(tool, "{}");
    raw.call_id = request_id.to_string();
    raw.extra
        .insert("type".into(), json!("mcp_approval_request"));
    if let Some(label) = server_label {
        raw.extra.insert("server_label".into(), json!(label));
    }
    ToolApprovalItem {
        agent: AGENT.into(),
        raw_item: raw,
        tool_name: match server_label {
            Some(label) => format!("{label}:{tool}"),
            None => tool.to_string(),
        },
        tool_origin: None,
    }
}

fn hosted(request_id: &str, server_label: &str) -> ToolApprovalItem {
    hosted_raw(request_id, Some(server_label), "lookup_account")
}

/// Restore Python-shaped `context.approvals` through the snapshot reader.
fn restored(approvals: Value, schema: &str) -> RunContext<()> {
    let path = format!(
        "{}/tests/fixtures/py_run_state_corpus/minimal/v1_20.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    json["$schemaVersion"] = json!(schema);
    json["context"]["approvals"] = approvals;
    let state = RunState::from_json(json).expect("snapshot");
    let ctx = RunContext::new(());
    ctx.import_approvals(&state.approvals);
    ctx
}

fn assert_user_error(err: AgentsError, needle: &str) {
    assert!(
        matches!(err, AgentsError::UserError { .. }) && err.to_string().contains(needle),
        "{err:?}"
    );
}

fn assert_model_error(err: AgentsError, needle: &str) {
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }) && err.to_string().contains(needle),
        "{err:?}"
    );
}

// ------------------------------------------------------------- hosted MCP identity scoping

#[test]
fn hosted_mcp_permanent_approval_is_scoped_by_server_label() {
    let ctx = RunContext::new(());
    let server_a = hosted("request-a-1", "server-a");
    ctx.approve_tool(&server_a, true);

    let next = hosted("request-a-2", "server-a");
    let other = hosted("request-b-1", "server-b");
    assert_eq!(
        ctx.approval_status_for_item(&next, "request-a-2").approved,
        Some(true)
    );
    assert_eq!(
        ctx.approval_status_for_item(&other, "request-b-1").approved,
        None
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "lookup_account", "request-a-1"),
        Some(true)
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "lookup_account", "request-a-2"),
        None
    );
    let keys = ctx.export_approvals().0;
    assert!(!keys.contains_key(&format!("{AGENT}\u{1}lookup_account")));
    assert!(!keys.contains_key("\u{1}lookup_account"));
}

#[test]
fn hosted_mcp_permanent_rejection_message_is_scoped_by_server_label() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &hosted("request-a-1", "server-a"),
        true,
        Some("server-a denied".into()),
    );
    let next = ctx.approval_status_for_item(&hosted("request-a-2", "server-a"), "request-a-2");
    assert_eq!(next.rejection_message.as_deref(), Some("server-a denied"));
    let other = ctx.approval_status_for_item(&hosted("request-b-1", "server-b"), "request-b-1");
    assert_eq!(other.approved, None);
    assert_eq!(other.rejection_message, None);
}

#[test]
fn hosted_mcp_name_based_query_preserves_exact_call_decision() {
    for (approved, always) in [(true, false), (true, true), (false, false), (false, true)] {
        let ctx = RunContext::new(());
        let item = hosted("request-a-1", "server-a");
        if approved {
            ctx.approve_tool(&item, always);
        } else {
            ctx.reject_tool(&item, always, Some("server-a denied".into()));
        }
        assert_eq!(
            ctx.is_tool_approved(AGENT, "lookup_account", "request-a-1"),
            Some(approved),
            "{approved} {always}"
        );
        assert_eq!(
            ctx.is_tool_approved(AGENT, "lookup_account", "request-a-2"),
            None
        );
        if !approved {
            assert_eq!(
                ctx.rejection_message(AGENT, "lookup_account", "request-a-1")
                    .as_deref(),
                Some("server-a denied")
            );
        }
    }
}

#[test]
fn hosted_mcp_exact_query_precedes_colliding_function_sticky_decision() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&fn_item("function-call", "lookup_account", None), true);
    ctx.reject_tool(
        &hosted("hosted-call", "server-a"),
        false,
        Some("hosted denial".into()),
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "lookup_account", "hosted-call"),
        Some(false)
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "lookup_account", "function-next"),
        Some(true)
    );
    assert_eq!(
        ctx.rejection_message(AGENT, "lookup_account", "hosted-call")
            .as_deref(),
        Some("hosted denial")
    );
}

#[test]
fn hosted_mcp_exact_query_does_not_inherit_function_rejection_reason() {
    for hosted_approved in [true, false] {
        let ctx = RunContext::new(());
        ctx.reject_tool(
            &fn_item("function-call", "lookup_account", None),
            false,
            Some("function denial".into()),
        );
        let item = hosted("shared-call", "server-a");
        if hosted_approved {
            ctx.approve_tool(&item, false);
        } else {
            ctx.reject_tool(&item, false, None);
        }
        assert_eq!(
            ctx.is_tool_approved(AGENT, "lookup_account", "shared-call"),
            Some(hosted_approved)
        );
        assert_eq!(
            ctx.rejection_message(AGENT, "lookup_account", "shared-call"),
            None
        );
    }
}

#[test]
fn hosted_mcp_exact_query_does_not_authorize_other_server() {
    for approved in [true, false] {
        let ctx = RunContext::new(());
        let server_a = hosted("shared-request", "server-a");
        if approved {
            ctx.approve_tool(&server_a, false);
        } else {
            ctx.reject_tool(&server_a, false, Some("server-a denied".into()));
        }
        assert_eq!(
            ctx.is_tool_approved(AGENT, "lookup_account", "shared-request"),
            Some(approved)
        );
        let server_b =
            ctx.approval_status_for_item(&hosted("shared-request", "server-b"), "shared-request");
        assert_eq!(server_b.approved, None);
        assert_eq!(server_b.rejection_message, None);
    }
}

#[test]
fn hosted_mcp_legacy_bare_name_approval_does_not_grant_access() {
    let ctx = restored(
        json!({"lookup_account": {"approved": true, "rejected": []}}),
        "1.20",
    );
    let pending = hosted("request-a-1", "server-a");
    assert_eq!(
        ctx.approval_status_for_item(&pending, "request-a-1")
            .approved,
        None
    );
}

#[test]
fn hosted_mcp_scoped_identity_cannot_alias_legacy_tool_name() {
    let colliding = r#"["hosted_mcp","server-a","lookup_account"]"#;
    let ctx = restored(
        json!({ colliding: {"approved": true, "rejected": []} }),
        "1.20",
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, colliding, "legacy-call"),
        Some(true)
    );
    let pending = hosted("request-a-1", "server-a");
    assert_eq!(
        ctx.approval_status_for_item(&pending, "request-a-1")
            .approved,
        None
    );
}

#[test]
fn hosted_mcp_legacy_exact_call_decisions_remain_usable() {
    let ctx = restored(
        json!({"lookup_account": {
            "approved": ["request-approved"],
            "rejected": ["request-rejected"],
            "rejection_messages": {"request-rejected": "legacy exact denial"},
        }}),
        "1.14",
    );
    let approved = hosted("request-approved", "server-a");
    let rejected = hosted("request-rejected", "server-a");
    // A persisted legacy item without the raw name or server label: only `tool_name` remains.
    let mut without_raw_name = hosted_raw("request-rejected", None, "");
    without_raw_name.tool_name = "lookup_account".into();

    assert_eq!(
        ctx.approval_status_for_item(&approved, "request-approved")
            .approved,
        Some(true)
    );
    assert_eq!(
        ctx.approval_status_for_item(&rejected, "request-rejected")
            .approved,
        Some(false)
    );
    let legacy = ctx.approval_status_for_item(&without_raw_name, "request-rejected");
    assert_eq!(
        legacy.rejection_message.as_deref(),
        Some("legacy exact denial")
    );
    assert_eq!(legacy.approved, None);
}

#[test]
fn hosted_mcp_persistent_decision_requires_complete_identity() {
    let ctx = RunContext::new(());
    let malformed = hosted_raw("request-a-1", None, "lookup_account");
    assert_user_error(
        ctx.try_approve_tool(&malformed, true).unwrap_err(),
        "non-empty server_label and tool name",
    );
}

#[test]
fn incomplete_hosted_mcp_uses_only_exact_call_decisions() {
    let ctx = restored(
        json!({"lookup_account": {
            "approved": true,
            "rejected": true,
            "sticky_rejection_message": "legacy denial",
        }}),
        "1.20",
    );
    let malformed = hosted_raw("request-a-1", None, "lookup_account");
    let status = ctx.approval_status_for_item(&malformed, "request-a-1");
    assert_eq!(status.approved, None);
    assert_eq!(status.rejection_message, None);
    assert_model_error(
        ctx.try_approve_tool(&malformed, false).unwrap_err(),
        "canonical invocation identity",
    );
    assert_model_error(
        ctx.try_reject_tool(&malformed, false, Some("exact denial".into()))
            .unwrap_err(),
        "canonical invocation identity",
    );
}

#[test]
fn hosted_mcp_decision_requires_request_id() {
    let ctx = RunContext::new(());
    let malformed = hosted_raw("", Some("server-a"), "lookup_account");
    assert_user_error(
        ctx.try_approve_tool(&malformed, false).unwrap_err(),
        "non-empty request id",
    );
    assert!(ctx.export_approvals().0.is_empty());
    // The infallible form records nothing either.
    ctx.approve_tool(&malformed, false);
    assert!(ctx.export_approvals().0.is_empty());
}

#[test]
fn hosted_mcp_invalid_request_id_does_not_mutate_approvals() {
    // Python also parametrizes a non-string id (`123`); Rust types the id as a string.
    let ctx = RunContext::new(());
    let malformed = hosted_raw("", Some("server-a"), "lookup_account");
    assert_user_error(
        ctx.try_reject_tool(&malformed, true, Some("must not persist".into()))
            .unwrap_err(),
        "non-empty request id",
    );
    assert!(ctx.export_approvals().0.is_empty());
}

#[test]
fn hosted_mcp_request_type_is_not_used_as_missing_tool_name() {
    let ctx = RunContext::new(());
    let malformed = hosted_raw("request-a-1", Some("server-a"), "");
    assert_user_error(
        ctx.try_approve_tool(&malformed, true).unwrap_err(),
        "non-empty server_label and tool name",
    );
    assert!(ctx.export_approvals().0.is_empty());
}

#[test]
fn hosted_mcp_decisions_round_trip_with_typed_identities() {
    // Python writes hosted decisions to `context.hosted_mcp_approvals`, not to the function
    // approvals, and reads them back with their server scope.
    let mut state = RunState::default();
    state.approve(&hosted("request-a-1", "server-a"), true);
    let json = state.to_json().unwrap();
    let hosted_entries = json["context"]["hosted_mcp_approvals"].clone();
    assert_eq!(
        hosted_entries,
        json!([
            {"identity": {"type": "server_tool", "server_label": "server-a", "tool_name": "lookup_account"},
             "decision": {"approved": true, "rejected": []}},
            {"identity": {"type": "query", "tool_name": "lookup_account", "request_id": "request-a-1"},
             "decision": {"approved": ["request-a-1"], "rejected": []}},
        ])
    );
    assert_eq!(json["context"]["function_tool_approvals"], json!([]));
    let back = RunState::from_json(json).unwrap();
    let ctx = RunContext::new(());
    ctx.import_approvals(&back.approvals);
    assert_eq!(
        ctx.approval_status_for_item(&hosted("request-a-9", "server-a"), "request-a-9")
            .approved,
        Some(true)
    );
    assert_eq!(
        ctx.approval_status_for_item(&hosted("request-b-1", "server-b"), "request-b-1")
            .approved,
        None
    );
}

// ------------------------------------------------------------ function tool decisions

#[test]
fn latest_approval_decision_wins_for_call_id() {
    let ctx = RunContext::new(());
    let item = fn_item("call-1", "test_tool", None);
    ctx.approve_tool(&item, false);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "test_tool", "call-1"),
        Some(true)
    );
    ctx.reject_tool(&item, false, None);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "test_tool", "call-1"),
        Some(false)
    );
    ctx.approve_tool(&item, false);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "test_tool", "call-1"),
        Some(true)
    );
}

#[test]
fn namespaced_approval_status_does_not_fall_back_to_bare_tool_decisions() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&fn_item("call-bare", "lookup_account", None), true);
    let billing = fn_item("call-billing", "lookup_account", Some("billing"));
    assert_eq!(
        ctx.approval_status_for_item(&billing, "call-billing-2")
            .approved,
        None
    );
    assert_eq!(
        ctx.is_tool_approved(AGENT, "billing.lookup_account", "call-billing-2"),
        None
    );
}

#[test]
fn namespaced_rejection_message_does_not_fall_back_to_bare_tool_decisions() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &fn_item("call-bare", "lookup_account", None),
        true,
        Some("bare denial".into()),
    );
    let billing = fn_item("call-billing", "lookup_account", Some("billing"));
    assert_eq!(
        ctx.approval_status_for_item(&billing, "call-billing-2")
            .rejection_message,
        None
    );
    assert_eq!(
        ctx.rejection_message(AGENT, "lookup_account", "call-bare-2")
            .as_deref(),
        Some("bare denial")
    );
}

#[test]
fn deferred_top_level_per_call_decisions_keep_bare_name_lookup() {
    // `test_deferred_top_level_per_call_approval_keeps_bare_name_lookup` and
    // `test_deferred_top_level_rejection_message_keeps_bare_name_lookup`: a deferred top-level
    // tool is approved under its public name, which a bare-name query finds.
    let ctx = RunContext::new(());
    let item = fn_item("call-weather", "get_weather", None);
    ctx.approve_tool(&item, false);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "get_weather", "call-weather"),
        Some(true)
    );
    let ctx = RunContext::new(());
    ctx.reject_tool(&item, false, Some("weather denied".into()));
    assert_eq!(
        ctx.rejection_message(AGENT, "get_weather", "call-weather")
            .as_deref(),
        Some("weather denied")
    );
}

#[test]
fn deferred_top_level_legacy_permanent_approval_requires_reapproval() {
    // A decision stored under the legacy `namespace.name` key of a deferred tool does not
    // authorize the tool's current identity.
    let ctx = restored(
        json!({"get_weather.get_weather": {"approved": true, "rejected": []}}),
        "1.14",
    );
    let item = fn_item("call-weather", "get_weather", None);
    assert_eq!(
        ctx.approval_status_for_item(&item, "call-weather-2")
            .approved,
        None
    );
}

#[test]
fn rebuild_approvals_ignores_malformed_approval_values() {
    let ctx = restored(json!(["not", "a", "mapping"]), "1.20");
    assert!(ctx.export_approvals().0.is_empty());

    let ctx = restored(
        json!({"get_weather": {
            "approved": {"not": "valid"},
            "rejected": ["call-denied", 123],
            "rejection_messages": {"call-denied": "no"},
        }}),
        "1.20",
    );
    assert_eq!(ctx.is_tool_approved(AGENT, "get_weather", "any-call"), None);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "get_weather", "call-denied"),
        Some(false)
    );
    assert_eq!(
        ctx.rejection_message(AGENT, "get_weather", "call-denied")
            .as_deref(),
        Some("no")
    );
}

#[test]
fn explicit_same_name_namespace_does_not_alias_to_bare_tool() {
    let ctx = RunContext::new(());
    let item = fn_item("call-namespaced", "lookup_account", Some("lookup_account"));
    ctx.approve_tool(&item, true);
    assert_eq!(
        ctx.is_tool_approved(AGENT, "lookup_account", "call-bare-2"),
        None
    );
    assert_eq!(
        ctx.approval_status_for_item(&item, "call-namespaced-2")
            .approved,
        Some(true)
    );
}

#[test]
fn sticky_decisions_replace_the_opposite_decision_wholesale() {
    // `_apply_approval_decision`: a sticky rejection drops earlier per-call approvals (and
    // the reverse), and a sticky approval clears the rejection message.
    let ctx = RunContext::new(());
    ctx.approve_tool(&fn_item("call-a", "t", None), false);
    ctx.reject_tool(&fn_item("call-b", "t", None), true, Some("never".into()));
    assert_eq!(ctx.is_tool_approved(AGENT, "t", "call-a"), Some(false));
    assert_eq!(
        ctx.rejection_message(AGENT, "t", "call-a").as_deref(),
        Some("never")
    );
    ctx.approve_tool(&fn_item("call-c", "t", None), true);
    assert_eq!(ctx.is_tool_approved(AGENT, "t", "call-b"), Some(true));
    assert_eq!(ctx.rejection_message(AGENT, "t", "call-b"), None);
}

// ----------------------------------------------- `run_internal.approvals` through the runner

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn approval_tool(executed: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::new("needs_approval", "needs approval", move |_c, _a: NoArgs| {
        let executed = executed.clone();
        async move {
            executed.fetch_add(1, Ordering::SeqCst);
            Ok("ran".to_string())
        }
    })
    .with_needs_approval(true)
}

#[tokio::test]
async fn interruptions_hold_only_approval_items_and_input_list_skips_them() {
    // `filter_tool_approvals`, `approvals_from_step` and
    // `append_input_items_excluding_approvals`: an interrupted run exposes only approval
    // items, and its input list carries the message and the call but no approval placeholder.
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        assistant_message("done"),
        function_call_with_id("needs_approval", "{}", "call_approval"),
    ]);
    let executed = Arc::new(AtomicUsize::new(0));
    let agent = Agent::<()>::new("test")
        .with_tool(approval_tool(executed.clone()))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "hi").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].call_id(), "call_approval");
    let types: Vec<String> = result
        .to_input_list()
        .iter()
        .map(|i| {
            let v = serde_json::to_value(i).unwrap();
            v.get("type")
                .or_else(|| v.get("role"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert!(!types.iter().any(|t| t.contains("approval")), "{types:?}");
    assert!(
        types.iter().any(|t| t == "message" || t == "assistant"),
        "{types:?}"
    );
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejected_approval_emits_a_function_call_output_with_the_message() {
    // `append_approval_error_output`: a rejection becomes a `function_call_output` carrying the
    // rejection text for the rejected call id.
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "needs_approval",
        "{}",
        "call_rejected",
    )]);
    model.add_turn(vec![assistant_message("ok")]);
    let executed = Arc::new(AtomicUsize::new(0));
    let agent = Agent::<()>::new("test")
        .with_tool(approval_tool(executed.clone()))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "hi").await.unwrap();
    let mut state = first.to_state();
    state.reject(
        &first.interruptions[0],
        false,
        Some("approval denied".into()),
    );
    let resumed = Runner::run(&agent, state).await.unwrap();
    let output = resumed
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("rejection output");
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["call_id"], "call_rejected");
    assert_eq!(output["output"], "approval denied");
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}
