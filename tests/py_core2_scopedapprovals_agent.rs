//! Ports of `tests/test_agent_scoped_approvals.py`: an approval decision belongs to the agent
//! that recorded it, survives transport of the context, and never authorizes another agent's
//! tool of the same name.
//!
//! Rust identifies an agent by name in approval records (`ToolApprovalItem.agent` is the name),
//! so the cases that rely on two *distinct* agents sharing one name (Python object identity) are
//! ported with distinct names where the behaviour is still meaningful, and listed as python-only
//! in the parity report otherwise. Python's three context converters (`TypeAdapter` python mode,
//! JSON mode, Temporal) are covered by the two serde transports of `ApprovalsState`: a
//! `serde_json::Value` round trip and a JSON string round trip.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{FunctionCall, OutputItem, RunInput, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_context::ApprovalsState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, ShellTool};
use openai_agents_sdk::{Agent, RunContext, RunResult, RunState, Runner};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

/// A late-bound agent: filled in after the handoff that targets it is built.
type AgentSlot = Arc<OnceLock<Arc<Agent<()>>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn setup() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn call(name: &str, args: &str, call_id: &str) -> FunctionCall {
    let mut fc = FunctionCall::new(name, args);
    fc.call_id = call_id.into();
    fc
}

fn approval(agent: &str, name: &str, call_id: &str) -> ToolApprovalItem {
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: call(name, "{}", call_id),
        tool_name: name.into(),
        tool_origin: None,
    }
}

fn shell_call(call_id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": call_id, "call_id": call_id, "status": "in_progress",
        "action": {"type": "exec", "commands": ["echo test"], "timeout_ms": 1000}
    }))
}

/// The approval item a native shell call raises (`tool_name` `shell`, the raw call in `extra`).
fn shell_approval(agent: &str, call_id: &str) -> ToolApprovalItem {
    let mut fc = call("shell", "{}", call_id);
    fc.extra.insert("type".into(), json!("shell_call"));
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: fc,
        tool_name: "shell".into(),
        tool_origin: None,
    }
}

fn hosted_approval(agent: &str, request_id: &str, server_label: &str) -> ToolApprovalItem {
    let mut fc = call("lookup_account", "{}", request_id);
    fc.extra
        .insert("type".into(), json!("mcp_approval_request"));
    fc.extra.insert("server_label".into(), json!(server_label));
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: fc,
        tool_name: "lookup_account".into(),
        tool_origin: None,
    }
}

fn model(turns: Vec<Vec<OutputItem>>) -> Arc<FakeModel> {
    let m = Arc::new(FakeModel::new());
    m.add_turns(turns);
    m
}

fn recording_tool(name: &str, output: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    let output = output.to_string();
    FunctionTool::new_no_args(name, "", move |_ctx| {
        let executed = executed.clone();
        let output = output.clone();
        async move {
            executed.lock().unwrap().push(output.clone());
            Ok(output)
        }
    })
}

/// A policy that records each call id it is asked about and always requires approval.
fn recording_policy(tool: FunctionTool<()>, policies: &Log) -> FunctionTool<()> {
    let policies = policies.clone();
    tool.with_needs_approval_fn(move |_ctx, _args, call_id| {
        let policies = policies.clone();
        async move {
            policies.lock().unwrap().push(call_id);
            true
        }
    })
}

/// A handoff back to an agent built later, so two agents can hand off to each other.
fn late_handoff(tool_name: &str, target_name: &str) -> (Handoff<()>, AgentSlot) {
    let slot: AgentSlot = Arc::new(OnceLock::new());
    let target = slot.clone();
    let handoff = Handoff::custom(
        tool_name,
        "Hand back.",
        target_name,
        json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false}),
        move |_ctx, _args| {
            let target = target.clone();
            async move { Ok(target.get().expect("handoff target set").clone()) }
        },
    );
    (handoff, slot)
}

async fn run(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streamed: bool,
    ctx: Option<RunContext<()>>,
) -> openai_agents_sdk::Result<RunResult<()>> {
    let options = match ctx {
        Some(ctx) => RunOptions::with_run_context(ctx),
        None => RunOptions::default(),
    };
    if streamed {
        let mut stream = Runner::run_streamed_with_options(agent, input, options);
        stream.finish().await?;
        Ok(stream.into_result().expect("streamed result"))
    } else {
        Runner::run_with_options(agent, input, options).await
    }
}

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
}

/// `(agent, call_id, output)` of every tool output item.
fn outputs(result: &RunResult<()>) -> Vec<(String, String, String)> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => Some((
                item.agent.clone(),
                raw_item["call_id"].as_str().unwrap_or_default().to_string(),
                output
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| output.to_string()),
            )),
            _ => None,
        })
        .collect()
}

fn output_texts(result: &RunResult<()>) -> Vec<String> {
    outputs(result).into_iter().map(|(_, _, o)| o).collect()
}

/// The two serde transports of a context's decisions. Python's `python` converter is a
/// dict round trip, its `json` converter a string round trip.
#[derive(Clone, Copy, Debug)]
enum Converter {
    Python,
    Json,
}

fn convert(ctx: &RunContext<()>, converter: Converter) -> RunContext<()> {
    let approvals = ctx.export_approvals();
    let restored: ApprovalsState = match converter {
        Converter::Python => {
            serde_json::from_value(serde_json::to_value(&approvals).unwrap()).unwrap()
        }
        Converter::Json => {
            let payload = serde_json::to_string(&approvals).unwrap();
            // Transport carries decisions, never the agent graph.
            assert!(!payload.contains("private-agent-instructions"));
            serde_json::from_str(&payload).unwrap()
        }
    };
    let out = RunContext::new(());
    out.import_approvals(&restored);
    out
}

const CONVERTERS: [Converter; 2] = [Converter::Python, Converter::Json];
const DECISIONS: [&str; 4] = ["approve", "reject", "always_approve", "always_reject"];

/// Python: `test_sticky_approval_stays_with_agent_across_handoffs` (streamed x serialized x
/// approve). Python names both agents `same-name`; Rust keys decisions by agent name, so the
/// agents here carry distinct names (see the module docs).
#[tokio::test]
async fn sticky_approval_stays_with_agent_across_handoffs() {
    setup();
    for streamed in [false, true] {
        for serialized in [false, true] {
            for approve in [false, true] {
                let ctx = format!("streamed={streamed} serialized={serialized} approve={approve}");
                let executed = log();
                let policies = log();
                let first_tool =
                    recording_policy(recording_tool("operate", "first", &executed), &policies);
                let second_tool =
                    recording_policy(recording_tool("operate", "second", &executed), &policies);
                let (to_first, first_slot) = late_handoff("to_first", "first");
                let second = Arc::new(
                    Agent::<()>::new("second")
                        .with_tool(second_tool)
                        .with_handoff(to_first)
                        .with_model_instance(model(vec![
                            vec![function_call_with_id("operate", "{}", "second-1")],
                            vec![function_call_with_id("operate", "{}", "second-2")],
                            vec![function_call_with_id("to_first", "{}", "handoff-2")],
                        ])),
                );
                let first = Arc::new(
                    Agent::<()>::new("first")
                        .with_tool(first_tool)
                        .with_handoff(Handoff::new(second.clone()).with_tool_name("to_second"))
                        .with_model_instance(model(vec![
                            vec![function_call_with_id("operate", "{}", "first-1")],
                            vec![function_call_with_id("to_second", "{}", "handoff-1")],
                            vec![function_call_with_id("operate", "{}", "first-2")],
                            vec![assistant_message("done")],
                        ])),
                );
                first_slot.set(first.clone()).unwrap();

                let result = run(&first, "start", streamed, None).await.unwrap();
                let mut state = result.to_state();
                if approve {
                    state.approve(&result.interruptions[0], true);
                } else {
                    state.reject(&result.interruptions[0], true, Some("first denied".into()));
                }
                if serialized {
                    state = roundtrip(&state);
                }
                let result = run(&first, state, streamed, None).await.unwrap();
                assert_eq!(result.interruptions.len(), 1, "{ctx}");
                assert_eq!(result.interruptions[0].agent, "second", "{ctx}");
                assert_eq!(*policies.lock().unwrap(), ["first-1", "second-1"], "{ctx}");
                let want: &[&str] = if approve { &["first"] } else { &[] };
                assert_eq!(*executed.lock().unwrap(), want, "{ctx}");

                let mut state = result.to_state();
                if approve {
                    state.reject(&result.interruptions[0], true, Some("second denied".into()));
                } else {
                    state.approve(&result.interruptions[0], true);
                }
                if serialized {
                    state = roundtrip(&state);
                }
                let result = run(&first, state, streamed, None).await.unwrap();
                assert_eq!(result.final_output_text().as_deref(), Some("done"), "{ctx}");
                assert!(result.interruptions.is_empty(), "{ctx}");
                assert_eq!(*policies.lock().unwrap(), ["first-1", "second-1"], "{ctx}");
                let want = if approve {
                    ["first", "first"]
                } else {
                    ["second", "second"]
                };
                assert_eq!(*executed.lock().unwrap(), want, "{ctx}");
                let (rejected_agent, reason) = if approve {
                    ("second", "second denied")
                } else {
                    ("first", "first denied")
                };
                let rejections: Vec<String> = outputs(&result)
                    .into_iter()
                    .filter(|(agent, call_id, _)| {
                        agent == rejected_agent
                            && ["first-1", "first-2", "second-1", "second-2"]
                                .contains(&call_id.as_str())
                    })
                    .map(|(_, _, o)| o)
                    .collect();
                assert_eq!(rejections, [reason, reason], "{ctx}");
            }
        }
    }
}

/// Python: `test_legacy_sticky_decisions_require_approval_after_restore` (approve).
#[tokio::test]
async fn legacy_sticky_decisions_require_approval_after_restore() {
    setup();
    for approve in [false, true] {
        let executed = log();
        let policies = log();
        let operate = recording_policy(recording_tool("operate", "result", &executed), &policies);
        let destination = Arc::new(
            Agent::<()>::new("destination")
                .with_tool(operate.clone())
                .with_model_instance(model(vec![
                    vec![function_call_with_id("operate", "{}", "pending")],
                    vec![assistant_message("done")],
                ])),
        );
        let source = Arc::new(
            Agent::<()>::new("source")
                .with_tool(operate)
                .with_handoff(destination.clone())
                .with_model_instance(model(vec![vec![function_call_with_id(
                    "transfer_to_destination",
                    "{}",
                    "transfer",
                )]])),
        );
        let initial = run(&source, "start", false, None).await.unwrap();
        let mut serialized: Value =
            serde_json::from_str(&initial.to_state().to_json_string().unwrap()).unwrap();
        serialized["$schemaVersion"] = json!("1.17");
        // The released format cannot tell which agent granted this permanent decision.
        serialized["context"]["approvals"] =
            json!({"operate": {"approved": approve, "rejected": !approve}});
        if let Some(context) = serialized["context"].as_object_mut() {
            context.remove("function_tool_approvals");
        }
        let restored = RunState::from_json_str(&serialized.to_string()).unwrap();
        let result = run(&source, restored, false, None).await.unwrap();
        assert_eq!(result.interruptions.len(), 1, "approve={approve}");
        assert_eq!(result.interruptions[0].agent, "destination");
        assert!(executed.lock().unwrap().is_empty(), "approve={approve}");
        assert_eq!(*policies.lock().unwrap(), ["pending"]);
        let mut resumed = result.to_state();
        resumed.approve(&result.interruptions[0], false);
        let result = run(&source, resumed, false, None).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(*executed.lock().unwrap(), ["result"]);
    }
}

fn denying_shell() -> ShellTool<()> {
    ShellTool::<()>::new(|_req| async move {
        panic!("rejected shell must not execute");
        #[allow(unreachable_code)]
        Ok::<_, openai_agents_sdk::AgentsError>("never")
    })
    .with_needs_approval(true)
}

/// Python: `test_function_sticky_rejection_does_not_replace_native_rejection_message`.
#[tokio::test]
async fn function_sticky_rejection_does_not_replace_native_rejection_message() {
    setup();
    for native_sticky in [false, true] {
        let mut dest_turns = vec![vec![shell_call("native")]];
        if native_sticky {
            dest_turns.push(vec![shell_call("native-again")]);
        }
        dest_turns.push(vec![assistant_message("done")]);
        let destination = Arc::new(
            Agent::<()>::new("destination")
                .with_tool(denying_shell())
                .with_model_instance(model(dest_turns)),
        );
        let function_shell = FunctionTool::<()>::new_no_args("shell", "", |_ctx| async move {
            panic!("rejected function must not execute");
            #[allow(unreachable_code)]
            Ok(String::new())
        })
        .with_needs_approval(true);
        let source = Arc::new(
            Agent::<()>::new("source")
                .with_tool(function_shell)
                .with_handoff(destination.clone())
                .with_model_instance(model(vec![
                    vec![function_call_with_id("shell", "{}", "function")],
                    vec![function_call_with_id(
                        "transfer_to_destination",
                        "{}",
                        "transfer",
                    )],
                ])),
        );
        let result = run(&source, "start", false, None).await.unwrap();
        let mut state = result.to_state();
        state.reject(
            &result.interruptions[0],
            true,
            Some("function denied".into()),
        );
        let result = run(&source, state, false, None).await.unwrap();
        assert_eq!(result.interruptions.len(), 1);
        assert_eq!(result.interruptions[0].agent, "destination");
        let mut state = result.to_state();
        state.reject(
            &result.interruptions[0],
            native_sticky,
            Some("native denied".into()),
        );
        let result = run(&source, state, false, None).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let mut want = vec!["function denied"];
        want.extend(std::iter::repeat_n(
            "native denied",
            if native_sticky { 2 } else { 1 },
        ));
        assert_eq!(output_texts(&result), want, "native_sticky={native_sticky}");
    }
}

/// Python: `test_function_exact_approval_preserves_native_sticky_rejection`.
#[tokio::test]
async fn function_exact_approval_preserves_native_sticky_rejection() {
    setup();
    let function_shell = FunctionTool::<()>::new_no_args("shell", "", |_ctx| async move {
        Ok("function result".to_string())
    })
    .with_needs_approval(true);
    let (to_native, native_slot) = late_handoff("to_native", "native");
    let function = Arc::new(
        Agent::<()>::new("function")
            .with_tool(function_shell)
            .with_handoff(to_native)
            .with_model_instance(model(vec![
                vec![function_call_with_id("shell", "{}", "function-1")],
                vec![function_call_with_id("to_native", "{}", "handoff-2")],
            ])),
    );
    let native = Arc::new(
        Agent::<()>::new("native")
            .with_tool(denying_shell())
            .with_handoff(Handoff::new(function.clone()).with_tool_name("to_function"))
            .with_model_instance(model(vec![
                vec![shell_call("native-1")],
                vec![function_call_with_id("to_function", "{}", "handoff-1")],
                vec![shell_call("native-2")],
                vec![assistant_message("done")],
            ])),
    );
    native_slot.set(native.clone()).unwrap();

    let result = run(&native, "start", false, None).await.unwrap();
    let mut state = result.to_state();
    state.reject(&result.interruptions[0], true, Some("native denied".into()));
    let result = run(&native, state, false, None).await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].agent, "function");
    let mut state = result.to_state();
    state.approve(&result.interruptions[0], false);
    let result = run(&native, state, false, None).await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        output_texts(&result),
        ["native denied", "function result", "native denied"]
    );
}

/// Python: `test_context_converters_preserve_function_decisions` (python and json converters
/// x decision).
#[test]
fn context_converters_preserve_function_decisions() {
    for converter in CONVERTERS {
        for decision in DECISIONS {
            let ctx_label = format!("{converter:?} {decision}");
            let context = RunContext::new(());
            let item = approval("same-name", "operate", "first");
            let approved = decision.ends_with("approve");
            let always = decision.starts_with("always_");
            if approved {
                context.approve_tool(&item, always);
            } else {
                context.reject_tool(&item, always, Some("Owner rejected".into()));
            }
            let native = shell_approval("same-name", "native");
            context.reject_tool(&native, false, Some("Native rejected".into()));

            // Repeated transport preserves the decisions.
            let restored = convert(&convert(&context, converter), converter);
            assert_eq!(
                restored.is_tool_approved("same-name", "operate", "first"),
                Some(approved),
                "{ctx_label}"
            );
            assert_eq!(
                restored.is_tool_approved("same-name", "operate", "next"),
                always.then_some(approved),
                "{ctx_label}"
            );
            assert_eq!(
                restored.rejection_message("same-name", "operate", "first"),
                (!approved).then(|| "Owner rejected".to_string()),
                "{ctx_label}"
            );
            let status = restored.approval_status_for_item(&native, "native");
            assert_eq!(status.approved, Some(false), "{ctx_label}");
            assert_eq!(status.rejection_message.as_deref(), Some("Native rejected"));
            assert_eq!(
                context.approval_status_for_item(&item, "first").approved,
                Some(approved),
                "{ctx_label}"
            );
        }
    }
}

/// Python: `test_python_context_conversion_preserves_mixed_hosted_decisions` (always x approve).
#[test]
fn python_context_conversion_preserves_mixed_hosted_decisions() {
    for always in [false, true] {
        for approve in [false, true] {
            let label = format!("always={always} approve={approve}");
            let context = RunContext::new(());
            context.approve_tool(&approval("owner", "operate", "function"), true);
            let hosted = hosted_approval("owner", "hosted", "server");
            if approve {
                context.approve_tool(&hosted, always);
            } else {
                context.reject_tool(&hosted, always, Some("Hosted rejected".into()));
            }
            let restored = convert(&convert(&context, Converter::Python), Converter::Python);
            assert_eq!(
                restored.is_tool_approved("owner", "operate", "function"),
                Some(true),
                "{label}"
            );
            let status = restored.approval_status_for_item(&hosted, "hosted");
            assert_eq!(status.approved, Some(approve), "{label}");
            assert_eq!(
                status.rejection_message,
                (!approve).then(|| "Hosted rejected".to_string()),
                "{label}"
            );
            let next_call = hosted_approval("owner", "next", "server");
            let other_server = hosted_approval("owner", "other", "other");
            assert_eq!(
                restored
                    .approval_status_for_item(&next_call, "next")
                    .approved,
                always.then_some(approve),
                "{label}"
            );
            assert_eq!(
                restored
                    .approval_status_for_item(&other_server, "other")
                    .approved,
                None,
                "{label}"
            );
        }
    }
}

fn gated_operate(executed: &Log) -> FunctionTool<()> {
    recording_tool("operate", "result", executed).with_needs_approval(true)
}

/// Python: `test_runner_executes_converted_context_decisions` (python and json converters x
/// decision x streamed).
#[tokio::test]
async fn runner_executes_converted_context_decisions() {
    setup();
    for converter in CONVERTERS {
        for decision in DECISIONS {
            for streamed in [false, true] {
                let label = format!("{converter:?} {decision} streamed={streamed}");
                let executed = log();
                let context = RunContext::new(());
                let approved = decision.ends_with("approve");
                let always = decision.starts_with("always_");
                let item = approval("owner", "operate", "first");
                if approved {
                    context.approve_tool(&item, always);
                } else {
                    context.reject_tool(&item, always, Some("Owner rejected".into()));
                }
                let restored = convert(&convert(&context, converter), converter);
                let mut steps = vec![vec![function_call_with_id("operate", "{}", "first")]];
                if always {
                    steps.push(vec![function_call_with_id("operate", "{}", "next")]);
                }
                steps.push(vec![assistant_message("done")]);
                let agent = Arc::new(
                    Agent::<()>::new("owner")
                        .with_tool(gated_operate(&executed))
                        .with_model_instance(model(steps)),
                );
                let result = run(&agent, "start", streamed, Some(restored))
                    .await
                    .unwrap();
                assert!(result.interruptions.is_empty(), "{label}");
                assert_eq!(
                    result.final_output_text().as_deref(),
                    Some("done"),
                    "{label}"
                );
                let count = if always { 2 } else { 1 };
                let want: Vec<&str> = if approved {
                    vec!["result"; count]
                } else {
                    vec![]
                };
                assert_eq!(*executed.lock().unwrap(), want, "{label}");
                let want = vec![if approved { "result" } else { "Owner rejected" }; count];
                assert_eq!(output_texts(&result), want, "{label}");
                // The checkpoint stays serializable.
                roundtrip(&result.to_state());
            }
        }
    }
}

/// Python: `test_runner_requires_reapproval_for_unresolved_context_owners[missing]`: a
/// decision recorded for another agent does not authorize this one.
#[tokio::test]
async fn runner_requires_reapproval_for_a_missing_context_owner() {
    setup();
    let context = RunContext::new(());
    context.approve_tool(&approval("owner", "operate", "first"), true);
    let restored = convert(&context, Converter::Json);
    let operate = FunctionTool::<()>::new_no_args("operate", "", |_ctx| async move {
        panic!("unresolved decisions must not execute");
        #[allow(unreachable_code)]
        Ok(String::new())
    })
    .with_needs_approval(true);
    let configured = Arc::new(
        Agent::<()>::new("different")
            .with_tool(operate)
            .with_model_instance(model(vec![vec![function_call_with_id(
                "operate", "{}", "first",
            )]])),
    );
    let result = run(&configured, "start", false, Some(restored))
        .await
        .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].agent, "different");
    let mut state = result.to_state();
    state.reject(
        &result.interruptions[0],
        false,
        Some("Fresh decision".into()),
    );
    roundtrip(&state);
}

/// Python: `test_reused_context_snapshots_only_current_graph_approvals`.
#[tokio::test]
async fn reused_context_snapshots_only_current_graph_approvals() {
    setup();
    let executed = log();
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_tool(gated_operate(&executed))
            .with_model_instance(model(vec![
                vec![function_call_with_id("operate", "{}", "first")],
                vec![assistant_message("done")],
            ])),
    );
    let context = RunContext::new(());
    let pending = run(&first, "start", false, Some(context.clone()))
        .await
        .unwrap();
    context.approve_tool(&pending.interruptions[0], true);
    let second = Arc::new(
        Agent::<()>::new("second")
            .with_tool(gated_operate(&executed))
            .with_model_instance(model(vec![vec![function_call_with_id(
                "operate", "{}", "second",
            )]])),
    );
    let result = run(&second, "start", false, Some(context.clone()))
        .await
        .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let mut state = result.to_state();
    state.reject(
        &result.interruptions[0],
        false,
        Some("Second denied".into()),
    );
    let restored = roundtrip(&state);
    let restored_ctx = RunContext::new(());
    restored_ctx.import_approvals(&restored.approvals);
    assert_eq!(
        restored_ctx
            .rejection_message("second", "operate", "second")
            .as_deref(),
        Some("Second denied")
    );
    // Snapshotting does not revoke the reusable context's other owner.
    assert_eq!(
        context.is_tool_approved("first", "operate", "future"),
        Some(true)
    );
}

/// Python: `test_converted_context_retains_other_graph_decisions` (python and json converters
/// x decision).
#[tokio::test]
async fn converted_context_retains_other_graph_decisions() {
    setup();
    for converter in CONVERTERS {
        for decision in DECISIONS {
            let label = format!("{converter:?} {decision}");
            let executed = log();
            let context = RunContext::new(());
            let approved = decision.ends_with("approve");
            let always = decision.starts_with("always_");
            for call_id in ["first", "second"] {
                let item = approval("owner", "operate", call_id);
                if approved {
                    context.approve_tool(&item, always);
                } else {
                    context.reject_tool(&item, always, Some("Denied".into()));
                }
            }
            let restored = convert(&context, converter);
            for call_id in ["first", "second"] {
                let agent = Arc::new(
                    Agent::<()>::new("owner")
                        .with_tool(gated_operate(&executed))
                        .with_model_instance(model(vec![
                            vec![function_call_with_id("operate", "{}", call_id)],
                            vec![assistant_message("done")],
                        ])),
                );
                let result = run(&agent, "start", false, Some(restored.clone()))
                    .await
                    .unwrap();
                assert!(result.interruptions.is_empty(), "{label}");
                assert_eq!(
                    result.final_output_text().as_deref(),
                    Some("done"),
                    "{label}"
                );
                assert_eq!(
                    output_texts(&result),
                    [if approved { "result" } else { "Denied" }],
                    "{label}"
                );
            }
            let want: &[&str] = if approved { &["result", "result"] } else { &[] };
            assert_eq!(*executed.lock().unwrap(), want, "{label}");
        }
    }
}

#[cfg(feature = "sandbox")]
mod py_sandbox_b_support;

/// Python: `test_function_decision_cannot_skip_sandbox_operation_approval`.
#[cfg(feature = "sandbox")]
#[tokio::test]
async fn function_decision_cannot_skip_sandbox_operation_approval() {
    use openai_agents_sdk::editor::ApplyPatchOperationType;
    use openai_agents_sdk::sandbox::capabilities::SandboxApplyPatchTool;
    use openai_agents_sdk::sandbox::editor::SandboxEditor;
    use openai_agents_sdk::tool::Tool;
    use py_sandbox_b_support::{ApplyPatchSession, Store};
    use std::collections::BTreeMap;

    setup();
    let session = Arc::new(ApplyPatchSession::new(Store::Sensitive));
    session.put("/workspace/protected.txt", b"protected");
    let checked = log();
    let seen = checked.clone();
    let patch_tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session.clone()))
        .with_needs_approval_fn(move |_ctx, operation, _call_id| {
            let seen = seen.clone();
            async move {
                let kind = match operation.op_type {
                    ApplyPatchOperationType::CreateFile => "create_file",
                    ApplyPatchOperationType::UpdateFile => "update_file",
                    ApplyPatchOperationType::DeleteFile => "delete_file",
                };
                seen.lock().unwrap().push(kind.to_string());
                operation.op_type == ApplyPatchOperationType::DeleteFile
            }
        });
    let patch = "*** Begin Patch\n*** Add File: harmless.txt\n+hello\n\
                 *** Delete File: protected.txt\n*** End Patch\n";
    let destination = Arc::new(
        Agent::<()>::new("destination")
            .with_tool(Tool::Custom(Arc::new(patch_tool.into_custom_tool())))
            .with_model_instance(model(vec![
                vec![OutputItem::from_value(json!({
                    "type": "custom_tool_call", "name": "apply_patch",
                    "call_id": "native", "input": patch
                }))],
                vec![assistant_message("done")],
            ])),
    );
    let source_tool = FunctionTool::<()>::new_no_args("apply_patch", "", |_ctx| async move {
        panic!("rejected function must not execute");
        #[allow(unreachable_code)]
        Ok(String::new())
    })
    .with_needs_approval(true);
    let source = Arc::new(
        Agent::<()>::new("source")
            .with_tool(source_tool)
            .with_handoff(destination.clone())
            .with_model_instance(model(vec![
                vec![function_call_with_id("apply_patch", "{}", "function")],
                vec![function_call_with_id(
                    "transfer_to_destination",
                    "{}",
                    "handoff",
                )],
            ])),
    );
    let result = run(&source, "start", false, None).await.unwrap();
    let mut state = result.to_state();
    state.reject(&result.interruptions[0], true, None);
    let result = run(&source, state, false, None).await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].agent, "destination");
    assert_eq!(*checked.lock().unwrap(), ["create_file", "delete_file"]);
    assert_eq!(
        session.files(),
        BTreeMap::from([(
            "/workspace/protected.txt".to_string(),
            b"protected".to_vec()
        )])
    );
}
