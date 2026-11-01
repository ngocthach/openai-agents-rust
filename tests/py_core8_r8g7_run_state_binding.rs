//! Ports of `tests/test_run_state.py` approval-binding cases that Python probes with the
//! private `RunContextWrapper.get_approval_status(existing_pending=, current_invocation=)`:
//! `legacy_schema_orphaned_per_call_approval_requires_reapproval`,
//! `schema_1_14_sticky_approval_binds_pending_hosted_mcp_invocation`,
//! `serialized_apply_patch_approval_binds_plural_operations`,
//! `schema_1_13_hosted_mcp_orphaned_call_decisions_require_reapproval` (context level) and
//! the unbindable-pending-approval cases. In Rust the decision is observable through a
//! resume, so each case parks a real run, edits the snapshot the way the Python test does and
//! resumes it.

#![allow(clippy::field_reassign_with_default)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::{FunctionCall, OutputItem, ToolApprovalItem};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{ApplyPatchTool, FunctionTool, HostedMCPTool};
use openai_agents_sdk::{Agent, InputItem, RunContext, Runner};
use serde_json::{Value, json};

const AGENT: &str = "ApprovalLegacyAgent";

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Ran(usize),
    Reinterrupted(Vec<String>),
    Error(String),
}

fn tool1(calls: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::<()>::new("tool1", "Sensitive tool", move |_c, a: Value| {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(format!("ok:{}", a["value"].as_str().unwrap_or_default()))
        }
    })
    .with_needs_approval(true)
}

fn approval(name: &str, call_id: &str, arguments: &str) -> ToolApprovalItem {
    let mut call = FunctionCall::new(name, arguments);
    call.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: AGENT.to_string(),
        raw_item: call,
        tool_name: name.to_string(),
        tool_origin: None,
    }
}

fn plain_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: AGENT.to_string(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(3),
        context: Some(json!({})),
        ..Default::default()
    }
}

/// Move the agent-owned decisions into the agent-agnostic bucket, as a pre-1.18 writer did.
fn to_legacy_approvals(wire: &mut Value, version: &str) {
    let entries = wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("function_tool_approvals")
        .unwrap_or(json!([]));
    for entry in entries.as_array().unwrap() {
        let key = entry["tool_key"].as_str().unwrap().to_string();
        wire["context"]["approvals"][key] = entry["decision"].clone();
    }
    wire["$schemaVersion"] = json!(version);
}

async fn resume_function(wire: Value, turns: Vec<Vec<OutputItem>>) -> Outcome {
    let state = match RunState::from_json(wire) {
        Ok(state) => state,
        Err(e) => return Outcome::Error(e.to_string()),
    };
    resume_function_state(state, turns).await
}

async fn resume_function_state(state: RunState, turns: Vec<Vec<OutputItem>>) -> Outcome {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake();
    model.add_turns(turns);
    let agent = Agent::<()>::new(AGENT)
        .with_tool(tool1(calls.clone()))
        .with_model_instance(model);
    match Runner::run(&agent, state).await {
        Err(e) => Outcome::Error(e.to_string()),
        Ok(result) if !result.interruptions.is_empty() => {
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            Outcome::Reinterrupted(
                result
                    .interruptions
                    .iter()
                    .map(|i| i.call_id().to_string())
                    .collect(),
            )
        }
        Ok(_) => Outcome::Ran(calls.load(Ordering::SeqCst)),
    }
}

/// A per-call decision with no reconstructable call is not authority at schema 1.13 / 1.14,
/// whatever arguments the model sends under that call id; a fresh decision restores it.
#[tokio::test]
async fn legacy_schema_orphaned_per_call_approval_requires_reapproval() {
    for version in ["1.13", "1.14"] {
        for arguments in [r#"{"value":"safe"}"#, r#"{"value":"changed"}"#] {
            let mut state = plain_state();
            state.approve(&approval("tool1", "cid1", r#"{"value":"safe"}"#), false);
            let mut wire = state.to_json().unwrap();
            to_legacy_approvals(&mut wire, version);
            wire["context"]
                .as_object_mut()
                .unwrap()
                .remove("tool_invocations");

            let turns = vec![
                vec![function_call_with_id("tool1", arguments, "cid1")],
                vec![assistant_message("done")],
            ];
            assert_eq!(
                resume_function(wire.clone(), turns.clone()).await,
                Outcome::Reinterrupted(vec!["cid1".into()]),
                "{version} {arguments}"
            );

            let mut restored = RunState::from_json(wire).unwrap();
            assert!(!restored.tool_invocations.contains_key("cid1"));
            restored.approve(&approval("tool1", "cid1", arguments), false);
            assert_eq!(
                resume_function_state(restored, turns).await,
                Outcome::Ran(1),
                "{version} {arguments}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Hosted MCP
// ---------------------------------------------------------------------------

fn mcp_request(id: &str, arguments: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "id": id, "type": "mcp_approval_request", "name": "lookup_account",
        "server_label": "server-a", "arguments": arguments,
    }))
}

fn hosted_tool() -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": "server-a", "server_url": "https://example.com",
        "require_approval": "always",
    }))
}

fn hosted_agent(model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new(AGENT)
        .with_tool(hosted_tool())
        .with_model_instance(model.clone())
}

/// Run one turn that raises the hosted MCP request `request-a` (safe arguments).
async fn parked_hosted_state() -> RunState {
    let model = fake();
    model.add_turn(vec![mcp_request("request-a", r#"{"value":"safe"}"#)]);
    let result = Runner::run(&hosted_agent(&model), "hi").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    result.to_state()
}

/// Rewrite the arguments of every stored copy of the request except the pending interruption.
fn change_stored_request(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("mcp_approval_request")
                && map.get("id").and_then(Value::as_str) == Some("request-a")
                && map.contains_key("arguments")
            {
                map.insert("arguments".into(), json!(r#"{"value":"changed"}"#));
            }
            for (key, child) in map.iter_mut() {
                if key != "current_step" {
                    change_stored_request(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(change_stored_request),
        _ => {}
    }
}

async fn resume_hosted(wire: Value) -> Result<openai_agents_sdk::RunResult<()>, String> {
    let state = RunState::from_json(wire).map_err(|e| e.to_string())?;
    let model = fake();
    model.add_turn(vec![assistant_message("done")]);
    Runner::run(&hosted_agent(&model), state)
        .await
        .map_err(|e| e.to_string())
}

/// A restored hosted MCP sticky decision binds the pending request payload: a changed payload
/// under the same request id is refused.
#[tokio::test]
async fn schema_1_14_sticky_approval_binds_pending_hosted_mcp_invocation() {
    let mut state = parked_hosted_state().await;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, true);
    let mut wire = state.to_json().unwrap();
    wire["$schemaVersion"] = json!("1.14");
    wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");

    // The exact pending request is still covered by the sticky decision.
    let result = resume_hosted(wire.clone()).await.expect("exact request");
    assert!(result.interruptions.is_empty());

    change_stored_request(&mut wire);
    let err = resume_hosted(wire).await.expect_err("changed request");
    assert!(err.contains("unique call ID"), "{err}");
}

/// A decision that survives only as a legacy per-call entry (schema 1.13) with no pending item
/// to bind to authorizes nothing until a fresh decision is recorded.
#[tokio::test]
async fn schema_1_13_hosted_mcp_orphaned_decisions_need_a_fresh_decision() {
    let mut state = RunState::default();
    state.current_agent_name = AGENT.into();
    state.max_turns = Some(10);
    state.original_input = vec![InputItem::user("hello")];
    let mut json = state.to_json().unwrap();
    json["context"]["approvals"] = json!({
        "lookup_account": {
            "approved": ["request-approved"],
            "rejected": ["request-rejected"],
            "rejection_messages": {"request-rejected": "legacy exact denial"},
        }
    });
    json["$schemaVersion"] = json!("1.13");
    json["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    let restored = RunState::from_json(json).unwrap();
    let ctx = RunContext::new(());
    ctx.import_approvals(&restored.approvals);

    let hosted = |id: &str| {
        let mut raw = FunctionCall::new("lookup_account", "{}");
        raw.call_id = id.to_string();
        raw.extra
            .insert("type".into(), json!("mcp_approval_request"));
        raw.extra.insert("server_label".into(), json!("server-a"));
        ToolApprovalItem {
            agent: AGENT.into(),
            raw_item: raw,
            tool_name: "server-a:lookup_account".into(),
            tool_origin: None,
        }
    };
    let approved = hosted("request-approved");
    // The persisted item of the rejected call has neither raw name nor server label.
    let mut rejected = hosted("request-rejected");
    rejected.raw_item.name.clear();
    rejected.raw_item.extra.remove("server_label");
    rejected.tool_name = "lookup_account".into();

    // The orphaned exact approval is re-asked once the restored request is the pending item
    // (the runner raises it again), exactly as the run-level resume does.
    assert!(
        restored
            .restored_unbound_approval_call_ids
            .contains("request-approved"),
        "{:?}",
        restored.restored_unbound_approval_call_ids
    );
    ctx.approve_tool(&approved, false);
    assert_eq!(
        ctx.approval_status_for_item(&approved, "request-approved")
            .approved,
        Some(true)
    );
    let status = ctx.approval_status_for_item(&rejected, "request-rejected");
    assert_eq!(status.approved, None);
    assert_eq!(
        status.rejection_message.as_deref(),
        Some("legacy exact denial")
    );
}

/// A pending hosted request that lost its server label cannot lend its approval to anything.
/// (Python also treats a hosted request without `arguments` as unbindable; Rust's hosted
/// request carries no required `arguments`, so only the label case is observable.)
#[tokio::test]
async fn current_schema_unbindable_hosted_pending_requires_reapproval() {
    for missing in ["server_label"] {
        let mut state = parked_hosted_state().await;
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let mut wire = state.to_json().unwrap();
        wire["current_step"]["data"]["interruptions"][0]["raw_item"]
            .as_object_mut()
            .unwrap()
            .remove(missing);
        let restored = RunState::from_json(wire).unwrap();
        let pending = restored.get_interruptions()[0].clone();
        let status = restored.approvals.status_for_item(&pending, "request-a");
        assert_eq!(status.approved, None, "{missing}");
    }
}

// ---------------------------------------------------------------------------
// apply_patch with plural operations
// ---------------------------------------------------------------------------

struct CountingEditor(Arc<AtomicUsize>);

#[async_trait]
impl ApplyPatchEditor for CountingEditor {
    async fn create_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ApplyPatchResult::completed(format!("Created {}", op.path)))
    }
    async fn update_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ApplyPatchResult::completed(format!("Updated {}", op.path)))
    }
    async fn delete_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ApplyPatchResult::completed(format!("Deleted {}", op.path)))
    }
}

fn patch_agent(model: &Arc<FakeModel>, edits: &Arc<AtomicUsize>) -> Agent<()> {
    let tool = ApplyPatchTool::<()>::from_arc(Arc::new(CountingEditor(edits.clone())))
        .with_needs_approval(true);
    Agent::<()>::new(AGENT)
        .with_tool(tool)
        .with_model_instance(model.clone())
}

fn change_stored_operations(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("apply_patch_call")
                && map.get("call_id").and_then(Value::as_str) == Some("patch-call")
                && let Some(ops) = map.get_mut("operations")
            {
                *ops = json!([{"type": "delete_file", "path": "important.txt"}]);
            }
            for (key, child) in map.iter_mut() {
                if key != "current_step" {
                    change_stored_operations(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(change_stored_operations),
        _ => {}
    }
}

/// Changed plural apply-patch operations cannot reuse a restored decision.
#[tokio::test]
async fn serialized_apply_patch_approval_binds_plural_operations() {
    for sticky in [false, true] {
        let model = fake();
        model.add_turn(vec![OutputItem::from_value(json!({
            "type": "apply_patch_call", "call_id": "patch-call",
            "operations": [{"type": "delete_file", "path": "safe.txt"}],
        }))]);
        let edits = Arc::new(AtomicUsize::new(0));
        let result = Runner::run(&patch_agent(&model, &edits), "hi")
            .await
            .unwrap();
        assert_eq!(result.interruptions.len(), 1, "sticky={sticky}");
        let mut state = result.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, sticky);
        let wire = state.to_json().unwrap();

        // The unchanged call runs.
        let model = fake();
        model.add_turn(vec![assistant_message("done")]);
        let edits = Arc::new(AtomicUsize::new(0));
        let restored = RunState::from_json(wire.clone()).unwrap();
        let result = Runner::run(&patch_agent(&model, &edits), restored)
            .await
            .unwrap();
        assert!(result.interruptions.is_empty(), "sticky={sticky}");
        assert_eq!(edits.load(Ordering::SeqCst), 1, "sticky={sticky}");

        // A changed operation list under the same call id is refused before it runs.
        let mut changed = wire;
        change_stored_operations(&mut changed);
        let model = fake();
        model.add_turn(vec![assistant_message("done")]);
        let edits = Arc::new(AtomicUsize::new(0));
        let restored = RunState::from_json(changed).unwrap();
        let err = Runner::run(&patch_agent(&model, &edits), restored)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unique call ID"),
            "sticky={sticky}: {err}"
        );
        assert_eq!(edits.load(Ordering::SeqCst), 0, "sticky={sticky}");
    }
}
