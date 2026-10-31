//! Port of the approval-ledger cases of the Python SDK's `tests/test_run_state.py`
//! (`TestRunState.test_serializes_and_restores_approvals` through
//! `test_context_override_discards_unbound_ids_from_previous_restore`, and the module-level
//! hosted-MCP approval cases).
//!
//! Python probes `RunContextWrapper.get_approval_status(existing_pending=...,
//! current_invocation=...)` directly. That helper is private; in Rust the same decision is
//! only observable through a resume, so each case runs a real interrupted turn, edits the
//! snapshot the way the Python test edits it, resumes, and checks the outcome Python's own
//! runner produces for that snapshot: the tool runs, the approval is asked for again, or the
//! resume fails with "unique call ID". The expected outcomes were confirmed by running the
//! same scenarios through the Python SDK's `Runner`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, ContextRestoreOptions, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunOptions, Runner};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const AGENT: &str = "ApprovalCurrentAgent";

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
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

/// Run one turn that calls `tool1(value=safe)` as `cid1` and stops on its approval.
async fn interrupted_state() -> RunState {
    let model = fake();
    model.add_turn(vec![function_call_with_id(
        "tool1",
        r#"{"value":"safe"}"#,
        "cid1",
    )]);
    let agent = Agent::<()>::new(AGENT)
        .with_tool(tool1(Arc::new(AtomicUsize::new(0))))
        .with_model_instance(model);
    let result = Runner::run(&agent, "hi").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    result.to_state()
}

/// [`interrupted_state`] with the parked call approved (or sticky-approved), as JSON.
async fn approved_wire(sticky: bool) -> Value {
    let mut state = interrupted_state().await;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, sticky);
    state.to_json().unwrap()
}

#[derive(Debug, PartialEq)]
enum Outcome {
    /// The tool ran this many times and the run finished.
    Ran(usize),
    /// The approval was asked for again; the tool did not run.
    Reinterrupted(Vec<String>),
    /// The resume was refused with this message.
    Error(String),
}

/// Resume `wire` with a model whose next turns are `turns` (default: one final message).
async fn resume_with(
    wire: Value,
    turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>,
) -> Outcome {
    let state = match RunState::from_json(wire) {
        Ok(state) => state,
        Err(e) => return Outcome::Error(e.to_string()),
    };
    resume_state(state, turns).await
}

async fn resume_state(
    state: RunState,
    turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>,
) -> Outcome {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake();
    if turns.is_empty() {
        model.add_turn(vec![assistant_message("done")]);
    } else {
        model.add_turns(turns);
    }
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

async fn resume(wire: Value) -> Outcome {
    resume_with(wire, Vec::new()).await
}

fn assert_unique_call_id_error(outcome: Outcome) {
    match outcome {
        Outcome::Error(message) => assert!(message.contains("unique call ID"), "{message}"),
        other => panic!("expected a reused-call-id error, got {other:?}"),
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

/// Rewrite every stored copy of the parked call with changed arguments.
fn change_stored_call(wire: &mut Value) {
    fn change(raw: &mut Value) {
        if raw.get("type").and_then(Value::as_str) == Some("function_call")
            && raw.get("call_id").and_then(Value::as_str) == Some("cid1")
        {
            raw["arguments"] = json!(r#"{"value":"changed"}"#);
        }
    }
    for response in wire["model_responses"].as_array_mut().unwrap() {
        for output in response["output"].as_array_mut().unwrap() {
            change(output);
        }
    }
    if let Some(last) = wire.get_mut("last_model_response")
        && let Some(outputs) = last.get_mut("output").and_then(Value::as_array_mut)
    {
        outputs.iter_mut().for_each(change);
    }
    for key in ["generated_items", "session_items"] {
        if let Some(items) = wire[key].as_array_mut() {
            for item in items {
                if item["type"] == json!("tool_call_item") {
                    change(&mut item["raw_item"]);
                }
            }
        }
    }
    if let Some(processed) = wire
        .get_mut("last_processed_response")
        .filter(|p| p.is_object())
    {
        for f in processed["functions"].as_array_mut().unwrap() {
            change(&mut f["tool_call"]);
        }
        for item in processed["new_items"].as_array_mut().unwrap() {
            if item["type"] == json!("tool_call_item") {
                change(&mut item["raw_item"]);
            }
        }
    }
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

fn plain_state(agent: &str) -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: agent.to_string(),
        original_input: vec![openai_agents_sdk::items::InputItem::user("test")],
        max_turns: Some(3),
        context: Some(json!({})),
        ..Default::default()
    }
}

fn err_text(result: openai_agents_sdk::Result<RunState>) -> String {
    match result {
        Ok(_) => panic!("expected the snapshot to be refused"),
        Err(e) => e.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Recorded decisions survive the envelope
// ---------------------------------------------------------------------------

#[test]
fn serializes_and_restores_approvals() {
    let mut state = plain_state("ApprovalAgent");
    state.approve(&approval("ApprovalAgent", "tool1", "cid1", ""), true);
    state.reject(&approval("ApprovalAgent", "tool2", "cid2", ""), true, None);
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(
        restored.approvals.status("ApprovalAgent", "tool1", "cid1"),
        Some(true)
    );
    assert_eq!(
        restored.approvals.status("ApprovalAgent", "tool2", "cid2"),
        Some(false)
    );
    assert_eq!(
        restored
            .approvals
            .rejection_message("ApprovalAgent", "tool2", "cid2"),
        None
    );
}

/// Record a sticky decision on `sticky` and the opposite exact decision on `exception`.
fn mixed_decisions(agent: &str, sticky_approved: bool, sticky_message: Option<&str>) -> RunState {
    let mut state = plain_state(agent);
    let item = |id: &str| approval(agent, "tool1", id, "{}");
    if sticky_approved {
        state.approve(&item("sticky"), true);
        state.reject(&item("exception"), false, Some("denied exactly".into()));
    } else {
        state.reject(&item("sticky"), true, sticky_message.map(str::to_string));
        state.approve(&item("exception"), false);
    }
    state
}

#[test]
fn exact_call_override_round_trips_with_sticky_default() {
    for sticky_approved in [true, false] {
        let agent = "MixedApprovalAgent";
        let state = mixed_decisions(agent, sticky_approved, Some("denied by default"));
        let serialized = state.to_json().unwrap();
        assert_eq!(serialized["$schemaVersion"], json!(CURRENT_SCHEMA_VERSION));
        let restored = RunState::from_json(serialized).unwrap();
        assert_eq!(
            restored.approvals.status(agent, "tool1", "exception"),
            Some(!sticky_approved)
        );
        assert_eq!(
            restored.approvals.status(agent, "tool1", "other"),
            Some(sticky_approved)
        );
        assert_eq!(
            restored
                .approvals
                .rejection_message(agent, "tool1", "exception")
                .as_deref(),
            sticky_approved.then_some("denied exactly")
        );
    }
}

#[test]
fn schema_1_15_mixed_approval_record_keeps_exact_decision() {
    for sticky_approved in [true, false] {
        let agent = "LegacyMixedApprovalAgent";
        let mut serialized = mixed_decisions(agent, sticky_approved, None)
            .to_json()
            .unwrap();
        to_legacy_approvals(&mut serialized, "1.15");
        let restored = RunState::from_json(serialized).unwrap();
        assert_eq!(
            restored.approvals.status(agent, "tool1", "exception"),
            Some(!sticky_approved)
        );
        assert_eq!(
            restored.approvals.status(agent, "tool1", "other"),
            Some(sticky_approved)
        );
    }
}

#[test]
fn serializes_and_restores_rejection_messages() {
    let agent = "ApprovalMessageAgent";
    let mut state = plain_state(agent);
    state.reject(
        &approval(agent, "tool2", "cid2", ""),
        true,
        Some("Denied by reviewer".into()),
    );
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    for call in ["cid2", "cid3"] {
        assert_eq!(
            restored
                .approvals
                .rejection_message(agent, "tool2", call)
                .as_deref(),
            Some("Denied by reviewer")
        );
    }
}

#[test]
fn from_json_accepts_previous_schema_version_without_rejection_messages() {
    let agent = "ApprovalLegacyAgent";
    let mut state = plain_state(agent);
    state.reject(
        &approval(agent, "tool2", "cid2", ""),
        false,
        Some("Denied by reviewer".into()),
    );
    let mut json_data = state.to_json().unwrap();
    to_legacy_approvals(&mut json_data, "1.5");
    json_data["context"]["approvals"]["tool2"]
        .as_object_mut()
        .unwrap()
        .remove("rejection_messages");
    let restored = RunState::from_json(json_data).unwrap();
    assert_eq!(
        restored.approvals.status(agent, "tool2", "cid2"),
        Some(false)
    );
    assert_eq!(
        restored.approvals.rejection_message(agent, "tool2", "cid2"),
        None
    );
}

#[test]
fn from_json_with_context_override_uses_serialized_rejection_messages() {
    let agent = "ApprovalOverrideAgent";
    let mut state = plain_state(agent).with_context(json!({"source": "saved"}));
    state.reject(
        &approval(agent, "tool2", "cid2", ""),
        true,
        Some("Denied by reviewer".into()),
    );
    let restored = RunState::from_json_with(
        state.to_json().unwrap(),
        ContextRestoreOptions::default().with_override(json!({"source": "override"})),
    )
    .unwrap();
    assert_eq!(restored.context, Some(json!({"source": "override"})));
    for call in ["cid2", "cid3"] {
        assert_eq!(
            restored
                .approvals
                .rejection_message(agent, "tool2", call)
                .as_deref(),
            Some("Denied by reviewer")
        );
    }
}

// ---------------------------------------------------------------------------
// Binding decisions to the tool-invocation ledger
// ---------------------------------------------------------------------------

#[tokio::test]
async fn schema_1_13_restores_pending_approval_binding_from_interruption() {
    let mut wire = approved_wire(false).await;
    to_legacy_approvals(&mut wire, "1.13");
    wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    // The exact pending invocation is still authorized.
    assert_eq!(resume(wire.clone()).await, Outcome::Ran(1));

    // A different payload under the same call id is not.
    change_stored_call(&mut wire);
    assert_unique_call_id_error(resume(wire).await);
}

#[tokio::test]
async fn legacy_schema_sticky_approval_binds_pending_function_invocation() {
    for version in ["1.13", "1.14"] {
        let mut wire = approved_wire(true).await;
        to_legacy_approvals(&mut wire, version);
        wire["context"]
            .as_object_mut()
            .unwrap()
            .remove("tool_invocations");
        change_stored_call(&mut wire);
        assert_unique_call_id_error(resume(wire).await);
    }
}

#[tokio::test]
async fn current_schema_does_not_reconstruct_missing_approval_binding() {
    let mut wire = approved_wire(false).await;
    wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    assert_eq!(
        resume(wire).await,
        Outcome::Reinterrupted(vec!["cid1".into()])
    );
}

#[tokio::test]
async fn current_schema_sticky_approval_requires_restored_pending_binding() {
    let mut wire = approved_wire(true).await;
    wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    let state = RunState::from_json(wire.clone()).unwrap();
    assert!(state.restored_unbound_approval_call_ids.contains("cid1"));
    // The resumed call id is not treated as fresh: the user is asked again.
    assert_eq!(
        resume(wire.clone()).await,
        Outcome::Reinterrupted(vec!["cid1".into()])
    );

    // The sticky decision still covers a genuinely new call. Approve the parked call again,
    // then let the model ask for `cid-fresh`: it runs without another interruption.
    let mut state = RunState::from_json(wire).unwrap();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let outcome = resume_state(
        state,
        vec![
            vec![function_call_with_id(
                "tool1",
                r#"{"value":"fresh"}"#,
                "cid-fresh",
            )],
            vec![assistant_message("done")],
        ],
    )
    .await;
    assert_eq!(outcome, Outcome::Ran(2));
}

#[tokio::test]
async fn current_schema_mismatched_pending_ledger_binding_requires_reapproval() {
    for sticky in [false, true] {
        let mut wire = approved_wire(sticky).await;
        let record = wire["context"]["tool_invocations"]["cid1"].clone();
        // Rewrite the ledger entry to the digests of a call with changed arguments, the way
        // `tool_invocation_identity_and_scope(changed_call)` does in the Python test: same
        // type and scope (arguments are not part of the scope), different fingerprint.
        wire["context"]["tool_invocations"]["cid1"]["fingerprint"] =
            json!(changed_call_fingerprint().await);
        assert_eq!(
            wire["context"]["tool_invocations"]["cid1"]["approval_scope"],
            record["approval_scope"]
        );
        assert_unique_call_id_error(resume(wire).await);
    }
}

/// The ledger fingerprint of `tool1(value=changed)` under `cid1`.
async fn changed_call_fingerprint() -> String {
    let model = fake();
    model.add_turn(vec![function_call_with_id(
        "tool1",
        r#"{"value":"changed"}"#,
        "cid1",
    )]);
    let agent = Agent::<()>::new(AGENT)
        .with_tool(tool1(Arc::new(AtomicUsize::new(0))))
        .with_model_instance(model);
    let result = Runner::run(&agent, "hi").await.unwrap();
    let wire = result.to_state().to_json().unwrap();
    wire["context"]["tool_invocations"]["cid1"]["fingerprint"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn current_schema_rejects_malformed_tool_invocation_ledger() {
    let cases = [
        ("type", json!("unknown_tool_call")),
        ("approval_scope", json!("not-a-digest")),
        ("fingerprint", json!(123)),
        ("fingerprint", json!("A".repeat(64))),
    ];
    for (field, value) in cases {
        let mut wire = approved_wire(false).await;
        wire["context"]["tool_invocations"]["cid1"][field] = value;
        let message = err_text(RunState::from_json(wire));
        assert!(
            message.contains("invalid lifecycle data"),
            "{field}: {message}"
        );
    }
}

#[tokio::test]
async fn current_schema_requires_tool_invocation_lifecycle_fields() {
    for missing in ["executed", "completed"] {
        let mut wire = approved_wire(false).await;
        let invocation = &mut wire["context"]["tool_invocations"]["cid1"];
        invocation["executed"] = json!(true);
        invocation["completed"] = json!(false);
        invocation.as_object_mut().unwrap().remove(missing);
        let message = err_text(RunState::from_json(wire));
        assert!(
            message.contains("invalid lifecycle data"),
            "{missing}: {message}"
        );
    }
}

#[test]
fn current_schema_rejects_null_tool_invocation_ledger() {
    let mut wire = plain_state(AGENT).to_json().unwrap();
    wire["context"]["tool_invocations"] = Value::Null;
    let message = err_text(RunState::from_json(wire));
    assert!(
        message.contains("tool_invocations must be a mapping"),
        "{message}"
    );
}

/// Mark the parked call executed and completed, which only a committed output may claim.
async fn completed_wire() -> Value {
    let mut wire = approved_wire(false).await;
    let invocation = &mut wire["context"]["tool_invocations"]["cid1"];
    invocation["executed"] = json!(true);
    invocation["completed"] = json!(true);
    wire
}

fn assert_completed_mismatch(result: openai_agents_sdk::Result<RunState>) {
    let message = err_text(result);
    assert!(
        message.contains("does not match a restored tool call and output"),
        "{message}"
    );
}

#[tokio::test]
async fn current_schema_rejects_completed_invocation_with_only_output_item_id() {
    let mut wire = completed_wire().await;
    wire["original_input"] = json!([{
        "type": "function_call_output", "call_id": "", "id": "cid1", "output": "forged",
    }]);
    assert_completed_mismatch(RunState::from_json(wire));
}

#[tokio::test]
async fn current_schema_rejects_completed_invocation_without_committed_output() {
    assert_completed_mismatch(RunState::from_json(completed_wire().await));
}

#[tokio::test]
async fn current_schema_rejects_completed_cross_paired_same_id_invocations() {
    // The ledger and pending call say `changed`; history holds an output for `safe`.
    let mut wire = completed_wire().await;
    wire["context"]["tool_invocations"]["cid1"]["fingerprint"] =
        json!(changed_call_fingerprint().await);
    change_stored_call(&mut wire);
    wire["original_input"] = json!([
        {"type": "function_call", "name": "tool1", "call_id": "cid1",
         "arguments": r#"{"value":"safe"}"#},
        {"type": "function_call_output", "call_id": "cid1", "output": "safe"},
    ]);
    assert_completed_mismatch(RunState::from_json(wire));
}

#[tokio::test]
async fn current_schema_rejects_completed_id_with_malformed_call_occurrence() {
    let mut wire = completed_wire().await;
    wire["original_input"] = json!([
        {"type": "function_call", "name": "tool1", "call_id": "cid1",
         "arguments": r#"{"value":"safe"}"#},
        {"type": "function_call", "name": "missing", "call_id": "cid1"},
        {"type": "function_call_output", "call_id": "cid1", "output": "safe"},
    ]);
    assert_completed_mismatch(RunState::from_json(wire));
}

#[tokio::test]
async fn a_completed_invocation_with_its_committed_output_is_accepted() {
    // The positive side of the completed-record check: call and output both restored.
    let mut wire = completed_wire().await;
    wire["original_input"] = json!([
        {"type": "function_call", "name": "tool1", "call_id": "cid1",
         "arguments": r#"{"value":"safe"}"#},
        {"type": "function_call_output", "call_id": "cid1", "output": "safe"},
    ]);
    RunState::from_json(wire).unwrap();
}

#[tokio::test]
async fn current_schema_orphaned_per_call_approval_requires_reapproval() {
    // A per-call decision whose ledger entry is gone authorizes nothing, even for the
    // original payload.
    let mut wire = approved_wire(false).await;
    wire["context"]["tool_invocations"] = json!({});
    let state = RunState::from_json(wire.clone()).unwrap();
    assert!(state.restored_unbound_approval_call_ids.contains("cid1"));
    assert!(!state.tool_invocations.contains_key("cid1"));
    assert_eq!(
        resume(wire).await,
        Outcome::Reinterrupted(vec!["cid1".into()])
    );
}

#[tokio::test]
async fn current_schema_missing_ledger_marks_historical_sticky_call_unbound() {
    let mut wire = approved_wire(true).await;
    wire["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    let state = RunState::from_json(wire).unwrap();
    assert!(state.restored_unbound_approval_call_ids.contains("cid1"));
    assert!(state.tool_invocations.is_empty());
}

#[tokio::test]
async fn context_override_discards_unbound_ids_from_previous_restore() {
    let wire = approved_wire(false).await;
    let mut malformed = wire.clone();
    malformed["context"]["tool_invocations"] = json!({});
    let first = RunState::from_json_with(
        malformed,
        ContextRestoreOptions::default().with_override(json!({})),
    )
    .unwrap();
    assert_eq!(
        first.restored_unbound_approval_call_ids,
        ["cid1".to_string()].into_iter().collect()
    );
    // Each restore derives its unbound set from its own snapshot.
    let restored = RunState::from_json_with(
        wire,
        ContextRestoreOptions::default().with_override(json!({})),
    )
    .unwrap();
    assert!(restored.restored_unbound_approval_call_ids.is_empty());
    assert_eq!(resume_state(restored, Vec::new()).await, Outcome::Ran(1));
}

// ---------------------------------------------------------------------------
// Module-level resume cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resume_pending_function_approval_reinterrupts() {
    let state = interrupted_state().await;
    assert_eq!(
        resume_state(state, Vec::new()).await,
        Outcome::Reinterrupted(vec!["cid1".into()])
    );
}

#[tokio::test]
async fn resume_rejected_function_approval_emits_output() {
    let mut state = interrupted_state().await;
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, None);
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake();
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new(AGENT)
        .with_tool(tool1(calls.clone()))
        .with_model_instance(model);
    let result = Runner::run(&agent, state).await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    // The default rejection message is the model-visible output of the call.
    assert!(
        result
            .new_items
            .iter()
            .any(|i| i.tool_output() == Some(&json!("Tool execution was not approved.")))
    );
}

#[tokio::test]
async fn resume_approved_function_approval_via_json_with_context_override() {
    #[derive(Default, Clone)]
    struct AppContext {
        user: String,
    }
    let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let tool = {
        let seen = seen.clone();
        FunctionTool::<AppContext>::new("tool1", "Sensitive tool", move |ctx, _a: Value| {
            let seen = seen.clone();
            let user = ctx.context().user.clone();
            async move {
                seen.lock().unwrap().push(user);
                Ok("ok".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let model = fake();
    model.add_turns(vec![
        vec![function_call_with_id("tool1", r#"{"value":"one"}"#, "1")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<AppContext>::new("agent")
        .with_tool(tool)
        .with_model_instance(model);
    let first = Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(AppContext {
            user: "original".into(),
        }),
    )
    .await
    .unwrap();
    let mut state = first.to_state();
    let item = first.interruptions[0].clone();
    state.approve(&item, false);
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    assert_eq!(restored.approvals.status("agent", "tool1", "1"), Some(true));

    let resumed = Runner::run_with_options(
        &agent,
        restored,
        RunOptions::new(AppContext {
            user: "reviewer".into(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(resumed.interruptions.is_empty());
    assert_eq!(*seen.lock().unwrap(), vec!["reviewer".to_string()]);
    assert_eq!(resumed.context_wrapper.context().user, "reviewer");
}

#[tokio::test]
async fn resume_approved_function_approval_streamed_with_context_override() {
    #[derive(Default, Clone)]
    struct AppContext {
        user: String,
    }
    let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let tool = {
        let seen = seen.clone();
        FunctionTool::<AppContext>::new("tool1", "Sensitive tool", move |ctx, _a: Value| {
            let seen = seen.clone();
            let user = ctx.context().user.clone();
            async move {
                seen.lock().unwrap().push(user);
                Ok("ok".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let model = fake();
    model.add_turns(vec![
        vec![function_call_with_id("tool1", r#"{"value":"one"}"#, "1")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<AppContext>::new("agent")
        .with_tool(tool)
        .with_model_instance(model);
    let first = Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(AppContext {
            user: "original".into(),
        }),
    )
    .await
    .unwrap();
    let mut state = first.to_state();
    let item = first.interruptions[0].clone();
    state.approve(&item, false);
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();

    let mut resumed = Runner::run_streamed_with_options(
        &agent,
        restored,
        RunOptions::new(AppContext {
            user: "reviewer".into(),
        }),
    );
    {
        use futures::StreamExt;
        let mut events = std::pin::pin!(resumed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(resumed.interruptions().is_empty());
    assert_eq!(*seen.lock().unwrap(), vec!["reviewer".to_string()]);
}
