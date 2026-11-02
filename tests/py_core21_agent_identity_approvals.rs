//! Ports of the owner-identity rows of the Python SDK's `tests/test_agent_scoped_approvals.py`.
//!
//! Python binds a function-tool decision to the agent object that owns it, so an identical clone
//! or copy of that agent is asked again. Rust binds a decision to the tool's capability
//! fingerprint and to the owning agent's identity: `Agent::new` and `Clone::clone` (and
//! `clone_with`, Rust's `agent.clone(...)`) mint a new identity, while the SDK's own copies of an
//! agent (the run's starting agent, handoff targets) keep it. Python's `copy.copy` maps to
//! `Clone::clone`, its `agent.clone()` to `clone_with`.
//!
//! The `json` converter is a JSON string round trip of the exported approvals, tool invocations
//! and capability registry, as in `py_core18_approval_fingerprint.rs`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_context::{ApprovalCapability, ApprovalsState};
use openai_agents_sdk::run_state::ToolInvocationRecord;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Handoff, RunContext, RunState, Runner};
use serde_json::json;

type Log = Arc<Mutex<Vec<String>>>;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn entries(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

/// A tool named `operate` that needs approval and records `label` when it runs.
fn operate(label: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    let label = label.to_string();
    FunctionTool::new("operate", "Operate.", move |_c, _a: NoArgs| {
        let executed = executed.clone();
        let label = label.clone();
        async move {
            executed.lock().unwrap().push(label.clone());
            Ok(label)
        }
    })
    .with_needs_approval(true)
}

fn call_item(agent: &str, call_id: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new("operate", "{}");
    raw.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: raw,
        tool_name: "operate".into(),
        tool_origin: None,
    }
}

/// Python's `_convert_approval_context(context, "json")`.
fn convert_json(ctx: &RunContext<()>) -> RunContext<()> {
    let text = serde_json::to_string(&(
        ctx.export_approvals(),
        ctx.export_tool_invocations(),
        ctx.export_approval_capabilities(),
    ))
    .unwrap();
    let (approvals, invocations, capabilities): (
        ApprovalsState,
        BTreeMap<String, ToolInvocationRecord>,
        BTreeMap<String, ApprovalCapability>,
    ) = serde_json::from_str(&text).unwrap();
    let restored = RunContext::new(());
    restored.import_approvals(&approvals);
    restored.import_tool_invocations(&invocations);
    restored.import_approval_capabilities(&capabilities);
    restored
}

/// `test_runner_requires_reapproval_for_unresolved_context_owners`, rows `source`, `clone`,
/// `copy` and `target` (the `missing` row is `runner_requires_reapproval_for_a_renamed_owner` in
/// `py_core18_approval_fingerprint.rs`): a sticky approval of `source`'s tool, transported
/// through JSON, does not authorize the same tool on another agent with the same name and
/// configuration. The run interrupts on the configured agent, and a fresh decision leaves a
/// serializable state.
#[tokio::test]
async fn runner_requires_reapproval_for_unresolved_context_owners() {
    for ambiguity in ["source", "clone", "copy", "target"] {
        let executed = log();
        let source = Agent::<()>::new("owner").with_tool(operate("unexpected", &executed));
        let ctx = RunContext::new(());
        ctx.approve_tool_for_agent(&source, &call_item("owner", "first"), true)
            .unwrap();
        if ambiguity == "source" {
            let sibling = source.clone_with(|_| {});
            ctx.approve_tool_for_agent(&sibling, &call_item("owner", "other"), true)
                .unwrap();
        }
        let model = model();
        model.add_turn(vec![function_call_with_id("operate", "{}", "first")]);
        let configured = match ambiguity {
            // Python's `copy.copy(source)`.
            "copy" => source.clone(),
            "target" => source.clone_with(|a| {
                a.handoffs = vec![Handoff::from(&source)];
            }),
            _ => source.clone_with(|_| {}),
        }
        .with_model_instance(model);
        let restored = convert_json(&ctx);
        let result =
            Runner::run_with_options(&configured, "start", RunOptions::with_run_context(restored))
                .await
                .unwrap_or_else(|e| panic!("{ambiguity}: {e}"));
        assert_eq!(result.interruptions.len(), 1, "{ambiguity}");
        assert_eq!(result.interruptions[0].agent, "owner", "{ambiguity}");
        assert!(entries(&executed).is_empty(), "{ambiguity}");
        let mut state = result.to_state();
        state.reject(
            &result.interruptions[0],
            false,
            Some("Fresh decision".into()),
        );
        RunState::from_json(state.to_json().unwrap())
            .unwrap_or_else(|e| panic!("{ambiguity}: {e}"));
    }
}

/// The control for the rows above: the very agent the decision was taken for runs the approved
/// call after the same transport, without asking again (Python's `configured = source`).
#[tokio::test]
async fn the_owning_agent_keeps_its_transported_decision() {
    let executed = log();
    let model = model();
    model.add_turn(vec![function_call_with_id("operate", "{}", "first")]);
    model.add_turn(vec![assistant_message("done")]);
    let source = Agent::<()>::new("owner")
        .with_tool(operate("executed", &executed))
        .with_model_instance(model);
    let ctx = RunContext::new(());
    ctx.approve_tool_for_agent(&source, &call_item("owner", "first"), true)
        .unwrap();
    let result = Runner::run_with_options(
        &source,
        "start",
        RunOptions::with_run_context(convert_json(&ctx)),
    )
    .await
    .unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(result.final_output, Some(json!("done")));
    assert_eq!(entries(&executed), vec!["executed"]);
}

/// A decision taken for a handoff target binds to that agent: the target reached through the
/// handoff runs the approved call, and a clone of the target placed in the same position does
/// not.
#[tokio::test]
async fn a_handoff_target_keeps_its_identity_and_its_clone_does_not() {
    for use_clone in [false, true] {
        let executed = log();
        let target_model = model();
        target_model.add_turn(vec![function_call_with_id("operate", "{}", "pending")]);
        target_model.add_turn(vec![assistant_message("done")]);
        let target = Agent::<()>::new("target")
            .with_tool(operate("target", &executed))
            .with_model_instance(target_model);
        let ctx = RunContext::new(());
        ctx.approve_tool_for_agent(&target, &call_item("target", "pending"), false)
            .unwrap();
        let placed = if use_clone { target.clone() } else { target };
        let root_model = model();
        root_model.add_turn(vec![function_call_with_id(
            "transfer_to_target",
            "{}",
            "handoff",
        )]);
        let root = Agent::<()>::new("root")
            .with_handoff(Handoff::from(placed))
            .with_model_instance(root_model);
        let result = Runner::run_with_options(&root, "start", RunOptions::with_run_context(ctx))
            .await
            .unwrap();
        if use_clone {
            assert_eq!(result.interruptions.len(), 1);
            assert!(entries(&executed).is_empty());
        } else {
            assert!(result.interruptions.is_empty());
            assert_eq!(entries(&executed), vec!["target"]);
        }
    }
}

/// `test_tied_duplicate_owner_restore_requires_reapproval` (`tied_at_write` x `always`): an
/// approval taken for one of two same-named handoff targets is not carried to a restored graph
/// whose same-named agents only traversal order tells apart, so the call is asked again and no
/// tool runs.
#[tokio::test]
async fn tied_duplicate_owner_restore_requires_reapproval() {
    for tied_at_write in [false, true] {
        for always in [false, true] {
            let label = format!("tied_at_write={tied_at_write} always={always}");
            let executed = log();
            let make_agent = |name: &str, instructions: &str| {
                let model = model();
                model.add_turn(vec![function_call_with_id("operate", "{}", "pending")]);
                model.add_turn(vec![assistant_message("done")]);
                Arc::new(
                    Agent::<()>::new("same")
                        .with_instructions(instructions)
                        .with_tool(operate(name, &executed))
                        .with_model_instance(model),
                )
            };
            let first = make_agent("first", if tied_at_write { "same" } else { "first" });
            let second = make_agent("second", if tied_at_write { "same" } else { "second" });
            let root_model = model();
            root_model.add_turn(vec![function_call_with_id("to_first", "{}", "handoff")]);
            let root = Agent::<()>::new("root")
                .with_handoff(Handoff::new(first.clone()).with_tool_name("to_first"))
                .with_handoff(Handoff::new(second.clone()).with_tool_name("to_second"))
                .with_model_instance(root_model);
            let result = Runner::run(&root, "start").await.unwrap();
            assert_eq!(result.interruptions.len(), 1, "{label}");
            let mut state = result.to_state();
            state.approve(&result.interruptions[0], always);
            let payload = state.to_json().unwrap();

            // Identical visible signatures cannot tell these different closures apart.
            let restored_first = make_agent("restored-first", "same");
            let restored_second = make_agent("restored-second", "same");
            let configured = root.clone_with(|a| {
                a.handoffs = vec![
                    Handoff::new(restored_second.clone()).with_tool_name("to_second"),
                    Handoff::new(restored_first.clone()).with_tool_name("to_first"),
                ];
            });
            let restored = RunState::from_json(payload).unwrap();
            let resumed = Runner::run(&configured, restored)
                .await
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(resumed.interruptions.len(), 1, "{label}");
            assert!(entries(&executed).is_empty(), "{label}");
        }
    }
}

/// A run state written to JSON binds its decisions to the agents of the graph it is resumed
/// with, as Python's `RunState.from_string` binds an agent-owned decision to the owner it
/// resolves: an agent rebuilt with the same configuration (another process) runs the approved
/// call.
#[tokio::test]
async fn a_json_run_state_binds_its_decisions_to_the_resumed_graph() {
    let executed = log();
    let build = |executed: &Log| {
        let model = model();
        model.add_turn(vec![function_call_with_id("operate", "{}", "pending")]);
        model.add_turn(vec![assistant_message("done")]);
        Agent::<()>::new("owner")
            .with_tool(operate("rebuilt", executed))
            .with_model_instance(model)
    };
    let original = build(&executed);
    let result = Runner::run(&original, "start").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let mut state = result.to_state();
    state.approve(&result.interruptions[0], false);
    let payload = state.to_json().unwrap();
    let rebuilt = build(&executed);
    let resumed = Runner::run(&rebuilt, RunState::from_json(payload).unwrap())
        .await
        .unwrap();
    assert!(resumed.interruptions.is_empty());
    assert_eq!(entries(&executed), vec!["rebuilt"]);
}
