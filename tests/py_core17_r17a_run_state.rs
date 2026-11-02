//! Ports of cases of the Python SDK's `tests/test_run_state.py` that earlier audits filed as
//! Python-only: decisions on detached pending approvals, ambiguity of a decision between
//! pending approvals that share an invocation identity, a completed invocation next to a
//! pending approval of the same identity, and approvals that carry no invocation identity.
//!
//! Python routes a decision between the current run and nested agent-tool states; a Rust
//! outer state keeps every pending approval, nested ones included, in one list keyed by agent
//! name, so a decision shared by "current and nested" scopes is a decision shared by two
//! entries of that list.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use openai_agents_sdk::items::{
    FunctionCall, InputItem, OutputItem, RunItem, RunItemKind, ToolApprovalItem,
};
use openai_agents_sdk::run_context::{ApprovalRecord, ApprovalsState, RunContext};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};

const AGENT: &str = "Agent";

fn item(agent: &str, tool: &str, kind: &str, call_id: &str, arguments: &str) -> ToolApprovalItem {
    let raw: Value = match kind {
        "custom_tool_call" => json!({
            "type": kind, "name": tool, "call_id": call_id, "input": "original",
        }),
        "shell_call" => json!({
            "type": kind, "name": tool, "id": call_id, "call_id": call_id, "status": "completed",
            "action": {"commands": ["echo", "ok"]},
        }),
        _ => json!({
            "type": "function_call", "name": tool, "call_id": call_id, "arguments": arguments,
        }),
    };
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: serde_json::from_value::<FunctionCall>(raw).unwrap(),
        tool_name: tool.into(),
        tool_origin: None,
    }
}

/// A run state parked on `items`, as a turn that stopped for approvals leaves it.
fn parked(items: Vec<ToolApprovalItem>) -> RunState {
    let mut state = RunState {
        schema_version: CURRENT_SCHEMA_VERSION.into(),
        current_agent_name: AGENT.into(),
        max_turns: Some(10),
        original_input: vec![InputItem::user("go")],
        ..Default::default()
    };
    for pending in &items {
        state.generated_items.push(RunItem::new(
            AGENT,
            RunItemKind::ToolApproval {
                raw_item: pending.raw_item.clone(),
                tool_name: pending.tool_name.clone(),
            },
        ));
    }
    state.interruptions = items;
    state.pending_tool_processing = true;
    state.context = Some(json!({}));
    state
}

fn status(state: &RunState, tool: &str, call_id: &str) -> Option<bool> {
    let context = RunContext::new(());
    context.import_approvals(&state.approvals);
    context.is_tool_approved(AGENT, tool, call_id)
}

fn decide(
    state: &mut RunState,
    pending: &ToolApprovalItem,
    approve: bool,
) -> openai_agents_sdk::Result<()> {
    if approve {
        state.try_approve(pending, false)
    } else {
        state.try_reject(pending, false, None)
    }
}

/// `test_approval_pipeline_models_can_apply_detached_decisions`: a snapshot of a custom-tool or
/// shell approval takes a decision that lands on the call's tool name and call id.
#[test]
fn detached_custom_and_shell_approvals_take_decisions() {
    for (tool, kind, call_id) in [
        ("custom_tool", "custom_tool_call", "cid-custom-decision"),
        ("shell", "shell_call", "cid-shell-decision"),
    ] {
        for approve in [true, false] {
            let label = format!("{kind} approve={approve}");
            let mut state = parked(vec![item(AGENT, tool, kind, call_id, "{}")]);
            let snapshot = state.get_interruptions()[0].clone();
            decide(&mut state, &snapshot, approve).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(status(&state, tool, call_id), Some(approve), "{label}");
        }
    }
}

/// `test_ambiguous_current_and_nested_approval_identity_fails_closed`,
/// `test_native_current_and_nested_approval_identity_fails_closed`,
/// `test_ambiguous_exact_nested_approval_identity_fails_closed` and
/// `test_unsafe_current_sibling_cannot_bypass_approval_ambiguity`: two pending approvals of one
/// agent with the same invocation identity make a decision ambiguous, live or restored from
/// JSON, for function and native calls alike. Nothing is recorded.
#[test]
fn shared_invocation_identity_fails_closed_live_and_restored() {
    for (tool, kind) in [("sensitive", "function_call"), ("shell", "shell_call")] {
        for round_trip in [false, true] {
            for approve in [true, false] {
                let label = format!("{kind} round_trip={round_trip} approve={approve}");
                let first = item(AGENT, tool, kind, "shared", "{}");
                let state = parked(vec![first.clone(), first]);
                let mut target = if round_trip {
                    RunState::from_json(state.to_json().unwrap()).unwrap()
                } else {
                    state
                };
                let selected = target.get_interruptions()[0].clone();
                let error = decide(&mut target, &selected, approve)
                    .expect_err(&label)
                    .to_string();
                assert!(
                    error.contains("multiple current pending approvals")
                        && error.contains("nested agent-tool run"),
                    "{label}: {error}"
                );
                assert_eq!(status(&target, tool, "shared"), None, "{label}");
            }
        }
    }
}

/// `test_completed_current_invocation_does_not_own_nested_approval`: a completed invocation
/// with the identity of a pending approval does not stand in for it; the pending approval
/// still takes its own decision, live or restored.
#[test]
fn a_completed_invocation_does_not_shadow_a_pending_approval() {
    for round_trip in [false, true] {
        for approve in [true, false] {
            let label = format!("round_trip={round_trip} approve={approve}");
            let pending = item(AGENT, "sensitive", "function_call", "shared", "{}");
            let mut state = parked(vec![pending.clone()]);
            let mut completed_call = FunctionCall::new("sensitive", "{}");
            completed_call.call_id = "shared".into();
            state.generated_items.insert(
                0,
                RunItem::new(
                    AGENT,
                    RunItemKind::ToolCall {
                        raw_item: completed_call.to_value(),
                    },
                ),
            );
            state.generated_items.insert(
                1,
                RunItem::new(
                    AGENT,
                    RunItemKind::ToolCallOutput {
                        raw_item: json!({
                            "type": "function_call_output", "call_id": "shared", "output": "done",
                        }),
                        output: json!("done"),
                        custom_data: None,
                    },
                ),
            );
            let mut target = if round_trip {
                RunState::from_json(state.to_json().unwrap()).unwrap()
            } else {
                state
            };
            let selected = target.get_interruptions()[0].clone();
            decide(&mut target, &selected, approve).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                status(&target, "sensitive", "shared"),
                Some(approve),
                "{label}"
            );
        }
    }
}

/// `test_uncopyable_noncanonical_approval_does_not_select_pending_owner`: an item that carries
/// no invocation identity cannot take a decision, and never lands on a pending owner.
#[test]
fn an_item_without_an_identity_never_selects_a_pending_owner() {
    for approve in [true, false] {
        let mut state = parked(vec![item(
            AGENT,
            "toolA",
            "function_call",
            "cid-uncopyable-decision",
            "{}",
        )]);
        let mut raw = FunctionCall::new("", "");
        raw.extra.insert("metadata".into(), json!({"unsafe": true}));
        let supplied = ToolApprovalItem {
            agent: AGENT.into(),
            raw_item: raw,
            tool_name: String::new(),
            tool_origin: None,
        };
        assert!(
            decide(&mut state, &supplied, approve).is_err(),
            "approve={approve}"
        );
        assert_eq!(
            status(&state, "toolA", "cid-uncopyable-decision"),
            None,
            "approve={approve}"
        );
    }
}

/// `test_approval_takes_precedence_over_rejection_when_both_true`: a record that was written
/// with both a permanent approval and a permanent rejection reads as approved.
#[test]
fn approval_takes_precedence_over_rejection_when_both_are_permanent() {
    let mut records = std::collections::BTreeMap::new();
    records.insert(
        format!("{AGENT}\u{1}test_tool"),
        ApprovalRecord {
            approved_always: true,
            rejected_always: true,
            ..Default::default()
        },
    );
    let context = RunContext::new(());
    context.import_approvals(&ApprovalsState(records));
    assert_eq!(
        context.is_tool_approved(AGENT, "test_tool", "call_id"),
        Some(true)
    );
}

// ---------------------------------------------------------------------------
// A decision whose binding is missing from a current-schema snapshot
// ---------------------------------------------------------------------------

fn tool1(calls: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::<()>::new("tool1", "Sensitive tool", move |_c, _a: Value| {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok("ok".to_string())
        }
    })
    .with_needs_approval(true)
}

fn scripted(first: Value) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![OutputItem::from_value(first)]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    )
}

/// `test_current_schema_missing_ledger_rejects_malformed_current_authority`: an approved call
/// whose invocation ledger was lost is not honoured by a replacement call that lacks the
/// canonical payload; the tool does not run.
#[tokio::test]
async fn a_malformed_replacement_call_does_not_consume_a_decision_without_a_binding() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = Agent::<()>::new(AGENT)
        .with_tool(tool1(calls.clone()))
        .with_model_instance(scripted(json!({
            "type": "function_call", "name": "tool1", "call_id": "cid1",
            "arguments": "{\"value\":\"safe\"}",
        })));
    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let mut json = state.to_json().unwrap();
    json["context"]["tool_invocations"] = json!({});
    let restored = RunState::from_json(json).unwrap();

    let malformed = Agent::<()>::new(AGENT)
        .with_tool(tool1(calls.clone()))
        .with_model_instance(scripted(json!({
            "type": "function_call", "name": "tool1", "call_id": "cid1",
        })));
    let outcome = Runner::run(&malformed, restored).await;
    assert_eq!(calls.load(Ordering::SeqCst), 0, "{outcome:?}");
}

// ---------------------------------------------------------------------------
// Nested agent-tool runs paused on an approval survive a JSON round trip
// ---------------------------------------------------------------------------

mod nested {
    use std::sync::Mutex;

    use openai_agents_sdk::agent::AgentAsToolOptions;
    use openai_agents_sdk::testing::{FakeModel, function_call_with_id};

    use super::*;

    pub type Calls = Arc<Mutex<Vec<String>>>;

    pub struct Graph {
        pub outer: Agent<()>,
        pub inner_model: Arc<FakeModel>,
        pub outer_model: Arc<FakeModel>,
    }

    /// `outer` calls `tools` (each an agent tool over its own inner agent whose tool needs
    /// approval) after an optional conditional function tool. The models hold only the steps of
    /// the turn being scripted.
    pub fn graph(
        first_turn: bool,
        nested: &[&str],
        conditional: Option<Arc<AtomicBool>>,
        calls: &Calls,
    ) -> Graph {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let inner_model = Arc::new(FakeModel::new());
        let outer_model = Arc::new(FakeModel::new());
        let mut outer = Agent::<()>::new("OuterAgent").with_model_instance(outer_model.clone());
        if let Some(flag) = conditional {
            outer = outer.with_tool(
                FunctionTool::<()>::new_no_args("conditional_outer_tool", "", |_c| async {
                    Ok("conditional-complete".to_string())
                })
                .with_is_enabled_fn(move |_ctx, _agent| {
                    let flag = flag.clone();
                    async move { flag.load(Ordering::SeqCst) }
                }),
            );
        }
        let mut turn = Vec::new();
        if outer
            .tools
            .iter()
            .any(|t| t.name() == "conditional_outer_tool")
        {
            turn.push(function_call_with_id(
                "conditional_outer_tool",
                "{}",
                "outer-pre",
            ));
        }
        for name in nested {
            let recorded = calls.clone();
            let label = name.to_string();
            let sensitive =
                FunctionTool::<()>::new_no_args(format!("{name}_sensitive"), "", move |_c| {
                    let recorded = recorded.clone();
                    let label = label.clone();
                    async move {
                        recorded.lock().unwrap().push(label.clone());
                        Ok(format!("approved:{label}"))
                    }
                })
                .with_needs_approval(true);
            let inner = Agent::<()>::new(format!("Inner-{name}"))
                .with_model_instance(inner_model.clone())
                .with_tool(sensitive);
            if first_turn {
                inner_model.add_turn(vec![function_call_with_id(
                    format!("{name}_sensitive"),
                    "{}",
                    format!("{name}-inner"),
                )]);
            } else {
                inner_model.add_turn(vec![assistant_message(format!("{name}-complete"))]);
            }
            outer = outer.with_tool(
                inner.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name(format!("{name}_agent_tool"))
                        .tool_description("Inner agent tool"),
                ),
            );
            turn.push(function_call_with_id(
                format!("{name}_agent_tool"),
                r#"{"input":"hello"}"#,
                format!("{name}-outer"),
            ));
        }
        if first_turn {
            outer_model.add_turn(turn);
        } else {
            outer_model.add_turn(vec![assistant_message("outer-complete")]);
        }
        Graph {
            outer,
            inner_model,
            outer_model,
        }
    }
}

/// `test_nested_agent_tool_hitl_resume_survives_json_round_trip_after_gc` and
/// `test_multiple_nested_agent_tool_states_survive_multiple_dropped_functions`: the state of
/// an interrupted turn, with one or two nested agent-tool runs paused on approvals and a
/// conditional function tool that is disabled by the time of the resume, restores from JSON
/// (twice, independently) and completes with the nested decisions.
#[tokio::test]
async fn nested_agent_tool_runs_survive_a_json_round_trip() {
    for nested_tools in [vec!["inner"], vec!["first", "second"]] {
        for approve in [true, false] {
            let label = format!("{nested_tools:?} approve={approve}");
            let calls: nested::Calls = Arc::default();
            let enabled = Arc::new(AtomicBool::new(true));
            let first = nested::graph(true, &nested_tools, Some(enabled.clone()), &calls);
            let paused = Runner::run(&first.outer, "start").await.unwrap();
            assert_eq!(paused.interruptions.len(), nested_tools.len(), "{label}");
            let wire = paused.to_state().to_json().unwrap();
            enabled.store(false, Ordering::SeqCst);
            drop(first);
            drop(paused);

            for attempt in 0..2 {
                let label = format!("{label} attempt={attempt}");
                let mut restored = RunState::from_json(wire.clone()).unwrap();
                let pending = restored.get_interruptions();
                assert_eq!(pending.len(), nested_tools.len(), "{label}");
                for item in &pending {
                    if approve {
                        restored.approve(item, false);
                    } else {
                        restored.reject(item, false, None);
                    }
                }
                let resumed = nested::graph(false, &nested_tools, Some(enabled.clone()), &calls);
                let result = Runner::run(&resumed.outer, restored)
                    .await
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(result.interruptions.is_empty(), "{label}");
                assert_eq!(
                    result.final_output_text().as_deref(),
                    Some("outer-complete"),
                    "{label}"
                );
                let _ = (&resumed.inner_model, &resumed.outer_model);
            }
            let expected = if approve { 2 * nested_tools.len() } else { 0 };
            assert_eq!(calls.lock().unwrap().len(), expected, "{label}");
        }
    }
}
