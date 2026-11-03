//! Approval decisions bind the exact call they were made for. Ports the cases the
//! scoped-approvals round deferred to the call-id-reuse area:
//! `test_agent_scoped_approvals.py::test_converted_context_preserves_approved_invocation`,
//! `::test_converted_context_does_not_repeat_completed_call` (python and json converters; the
//! Temporal converter is Python-only) and from `test_hitl_error_scenarios.py`:
//! `test_resume_rebuilds_function_runs_from_pending_approvals`,
//! `test_resume_rebuilds_function_runs_from_object_approvals`,
//! `test_rebuild_preserves_unmatched_pending_approvals` (shell, apply_patch, missing function),
//! `test_tool_execution_rejects_changed_approval_recorded_while_policy_waits` (shell, custom,
//! apply_patch), `test_resume_rejects_changed_handoff_under_approved_function_call_id`,
//! `test_resume_validates_changed_shell_before_sibling_approval_callback` and the callback half
//! of `test_changed_nested_parent_fails_before_tool_inventory_callbacks`.
//!
//! Python drives `_resolve_interrupted_turn` and the `*Action.execute` helpers directly; here the
//! same turns run through `Runner`. Python's converters carry `_tool_invocations` with the
//! decisions; the Rust converter carries `export_tool_invocations` with `export_approvals`.

#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::{FunctionCall, OutputItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_context::ApprovalsState;
use openai_agents_sdk::run_state::ToolInvocationRecord;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{ApplyPatchTool, CustomTool, FunctionTool, ShellTool};
use openai_agents_sdk::{Agent, handoffs::handoff};
use openai_agents_sdk::{AgentsError, InputItem, RunContext, RunState, Runner};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

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

fn assert_reused(err: &AgentsError) {
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. })
            && err.to_string().contains("unique call ID"),
        "{err:?}"
    );
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    #[serde(default)]
    value: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

fn operate(name: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    FunctionTool::new(name, "operate", move |_c, a: ValueArgs| {
        let executed = executed.clone();
        async move {
            let v = a.value.or(a.reason).unwrap_or_else(|| "executed".into());
            executed.lock().unwrap().push(v.clone());
            Ok(v)
        }
    })
    .with_needs_approval(true)
}

fn fn_item(agent: &str, name: &str, call_id: &str, args: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new(name, args);
    raw.call_id = call_id.into();
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: raw,
        tool_name: name.into(),
        tool_origin: None,
    }
}

#[derive(Clone, Copy, Debug)]
enum Converter {
    Python,
    Json,
}

/// Move a context's decisions, and the invocations they bind, into a fresh context.
fn convert(ctx: &RunContext<()>, converter: Converter) -> RunContext<()> {
    let payload = json!({
        "approvals": ctx.export_approvals(),
        "tool_invocations": ctx.export_tool_invocations(),
    });
    let payload = match converter {
        Converter::Python => payload,
        Converter::Json => serde_json::from_str(&payload.to_string()).unwrap(),
    };
    let approvals: ApprovalsState = serde_json::from_value(payload["approvals"].clone()).unwrap();
    let invocations: BTreeMap<String, ToolInvocationRecord> =
        serde_json::from_value(payload["tool_invocations"].clone()).unwrap();
    let out = RunContext::new(());
    out.import_approvals(&approvals);
    out.import_tool_invocations(&invocations);
    out
}

// ---------------------------------------------------------------- converted contexts

#[tokio::test]
async fn converted_context_preserves_approved_invocation() {
    for converter in [Converter::Python, Converter::Json] {
        let executed = log();
        let ctx = RunContext::new(());
        ctx.approve_tool(
            &fn_item("owner", "operate", "same", r#"{"value":"approved"}"#),
            false,
        );
        let restored = convert(&ctx, converter);
        let model = model();
        model.add_turn(vec![function_call_with_id(
            "operate",
            r#"{"value":"changed"}"#,
            "same",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("owner")
            .with_tool(operate("operate", &executed))
            .with_model_instance(model.clone());
        let err = Runner::run_with_options(&agent, "start", RunOptions::with_run_context(restored))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("reused a tool call ID"),
            "{converter:?} {err:?}"
        );
        assert!(entries(&executed).is_empty());
    }
}

#[tokio::test]
async fn converted_context_does_not_repeat_completed_call() {
    for converter in [Converter::Python, Converter::Json] {
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id("operate", "{}", "same")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("owner")
            .with_tool(operate("operate", &executed))
            .with_model_instance(model.clone());
        let ctx = RunContext::new(());
        ctx.approve_tool(&fn_item("owner", "operate", "same", "{}"), true);
        Runner::run_with_options(&agent, "start", RunOptions::with_run_context(ctx.clone()))
            .await
            .unwrap();
        let restored = convert(&ctx, converter);
        model.add_turn(vec![function_call_with_id("operate", "{}", "same")]);
        model.add_turn(vec![assistant_message("done")]);
        let result =
            Runner::run_with_options(&agent, "start", RunOptions::with_run_context(restored))
                .await
                .unwrap();
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("done"),
            "{converter:?}"
        );
        assert!(result.interruptions.is_empty());
        assert_eq!(entries(&executed), vec!["executed"], "{converter:?}");
    }
}

// ------------------------------------------------- snapshots without a model response

/// A snapshot parked on `items` that kept no model response.
fn parked_state(agent: &str, items: Vec<ToolApprovalItem>) -> RunState {
    let mut state = RunState::default();
    state.schema_version = openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.into();
    state.current_agent_name = agent.into();
    state.max_turns = Some(10);
    state.original_input = vec![InputItem::user("resume approvals")];
    for item in &items {
        state
            .generated_items
            .push(openai_agents_sdk::items::RunItem::new(
                agent,
                RunItemKind::ToolApproval {
                    raw_item: item.raw_item.clone(),
                    tool_name: item.tool_name.clone(),
                },
            ));
    }
    state.interruptions = items;
    state.pending_tool_processing = true;
    state
}

#[tokio::test]
async fn resume_rebuilds_function_runs_from_pending_approvals() {
    // Python's dict and `ResponseFunctionToolCall` approval items are the same typed call here;
    // one case carries a `status`, the other does not.
    for status in [Some("completed"), None] {
        let executed = log();
        let model = model();
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("TestAgent")
            .with_tool(operate("approve_me", &executed))
            .with_model_instance(model.clone());
        let mut item = fn_item(
            "TestAgent",
            "approve_me",
            "call-rebuild-1",
            r#"{"reason": "ok"}"#,
        );
        item.raw_item.status = status.map(str::to_string);
        let mut state = parked_state("TestAgent", vec![item.clone()]);
        state.approve(&item, false);
        let result = Runner::run(&agent, state).await.unwrap();
        assert!(result.interruptions.is_empty(), "{status:?}");
        let outputs: Vec<Value> = result
            .new_items
            .iter()
            .filter_map(|i| match &i.kind {
                RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item["call_id"].clone()),
                _ => None,
            })
            .collect();
        assert!(outputs.contains(&json!("call-rebuild-1")), "{outputs:?}");
        assert_eq!(entries(&executed), vec!["ok"]);
    }
}

#[tokio::test]
async fn rebuild_preserves_unmatched_pending_approvals() {
    let shell = {
        let mut raw = FunctionCall::new("shell", "");
        raw.call_id = "call_shell_pending_rebuild".into();
        raw.extra.insert("type".into(), json!("shell_call"));
        raw.extra.insert(
            "action".into(),
            json!({"type": "exec", "commands": ["echo pending"], "timeout_ms": 1000}),
        );
        (raw, "shell")
    };
    let patch = {
        let mut raw = FunctionCall::new("apply_patch", "");
        raw.call_id = "call_apply_pending_rebuild".into();
        raw.extra.insert("type".into(), json!("apply_patch_call"));
        raw.extra.insert(
            "operation".into(),
            json!({"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"}),
        );
        (raw, "apply_patch")
    };
    let missing = {
        let mut raw = FunctionCall::new("missing_tool", "{}");
        raw.call_id = "call_missing_tool".into();
        (raw, "missing_tool")
    };
    for (raw, tool_name) in [shell, patch, missing] {
        let model = model();
        let agent = Agent::<()>::new("TestAgent").with_model_instance(model.clone());
        let item = ToolApprovalItem {
            agent: "TestAgent".into(),
            raw_item: raw,
            tool_name: tool_name.into(),
            tool_origin: None,
        };
        let state = parked_state("TestAgent", vec![item.clone()]);
        let result = Runner::run(&agent, state).await.unwrap();
        assert!(
            result
                .interruptions
                .iter()
                .any(|i| i.call_id() == item.call_id()),
            "{tool_name}: {:?}",
            result.interruptions
        );
        assert!(model.calls().is_empty(), "{tool_name}");
    }
}

// --------------------------------------- a decision recorded while the policy waits

#[derive(Default, Clone)]
struct RecordingEditor(Log);

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("ok"))
    }
    async fn update_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("ok"))
    }
    async fn delete_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.0.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("ok"))
    }
}

#[tokio::test]
async fn changed_approval_recorded_while_policy_waits_is_refused() {
    for kind in ["shell", "custom", "apply_patch"] {
        let executed = log();
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let gate = {
            let (started, release) = (started.clone(), release.clone());
            move || {
                let (started, release) = (started.clone(), release.clone());
                async move {
                    started.notify_one();
                    release.notified().await;
                    true
                }
            }
        };
        let model = model();
        let (agent, current, changed_item) = match kind {
            "shell" => {
                let e = executed.clone();
                let g = gate.clone();
                let tool = ShellTool::new(move |_req| {
                    let e = e.clone();
                    async move {
                        e.lock().unwrap().push("shell".into());
                        Ok("should-not-run")
                    }
                })
                .with_needs_approval_fn(move |_c, _a, _id| g());
                let current = json!({"type": "shell_call", "call_id": "call-shared", "status": "in_progress",
                    "action": {"type": "exec", "commands": ["safe"], "timeout_ms": 1000}});
                let mut raw = FunctionCall::new("shell", "");
                raw.call_id = "call-shared".into();
                raw.status = Some("in_progress".into());
                raw.extra.insert("type".into(), json!("shell_call"));
                raw.extra.insert(
                    "action".into(),
                    json!({"type": "exec", "commands": ["changed"], "timeout_ms": 1000}),
                );
                (
                    Agent::<()>::new("agent").with_tool(tool),
                    current,
                    (raw, "shell"),
                )
            }
            "custom" => {
                let e = executed.clone();
                let g = gate.clone();
                let tool =
                    CustomTool::new("raw_editor", "Edit raw text.", move |_c, _i: String| {
                        let e = e.clone();
                        async move {
                            e.lock().unwrap().push("custom".into());
                            Ok("should-not-run".to_string())
                        }
                    })
                    .with_format(json!({"type": "text"}))
                    .with_needs_approval_fn(move |_c, _a, _id| g());
                let current = json!({"type": "custom_tool_call", "name": "raw_editor", "call_id": "call-shared", "input": "safe"});
                let mut raw = FunctionCall::new("raw_editor", "");
                raw.call_id = "call-shared".into();
                raw.extra.insert("type".into(), json!("custom_tool_call"));
                raw.extra.insert("input".into(), json!("changed"));
                (
                    Agent::<()>::new("agent").with_tool(tool),
                    current,
                    (raw, "raw_editor"),
                )
            }
            _ => {
                let g = gate.clone();
                let tool = ApplyPatchTool::new(RecordingEditor(executed.clone()))
                    .with_needs_approval_fn(move |_c, _a, _id| g());
                let current = json!({"type": "apply_patch_call", "call_id": "call-shared",
                    "operation": {"type": "delete_file", "path": "safe.txt"}});
                let mut raw = FunctionCall::new("apply_patch", "");
                raw.call_id = "call-shared".into();
                raw.extra.insert("type".into(), json!("apply_patch_call"));
                raw.extra.insert(
                    "operation".into(),
                    json!({"type": "delete_file", "path": "changed.txt"}),
                );
                (
                    Agent::<()>::new("agent").with_tool(tool),
                    current,
                    (raw, "apply_patch"),
                )
            }
        };
        model.add_turn(vec![OutputItem::from_value(current)]);
        let agent = Arc::new(agent.with_model_instance(model.clone()));
        let ctx = RunContext::new(());
        let task = {
            let (agent, ctx) = (agent.clone(), ctx.clone());
            tokio::spawn(async move {
                Runner::run_with_options(&agent, "go", RunOptions::with_run_context(ctx)).await
            })
        };
        started.notified().await;
        ctx.approve_tool(
            &ToolApprovalItem {
                agent: "agent".into(),
                raw_item: changed_item.0,
                tool_name: changed_item.1.into(),
                tool_origin: None,
            },
            false,
        );
        release.notify_one();
        let err = task.await.unwrap().unwrap_err();
        assert_reused(&err);
        assert!(entries(&executed).is_empty(), "{kind}");
    }
}

// ---------------------------------------------------- changed calls under an approved id

#[tokio::test]
async fn changed_handoff_under_approved_function_call_id_is_refused() {
    let calls = log();
    let c = calls.clone();
    let target = Arc::new(Agent::<()>::new("target"));
    let route = handoff(target)
        .with_tool_name("route")
        .with_on_handoff(move |_ctx| {
            let c = c.clone();
            async move {
                c.lock().unwrap().push("route".into());
                Ok(())
            }
        });
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "route",
        r#"{"destination":"safe"}"#,
        "call-shared",
    )]);
    let agent = Agent::<()>::new("agent")
        .with_handoff(route)
        .with_model_instance(model.clone());
    let ctx = RunContext::new(());
    ctx.approve_tool(
        &fn_item("agent", "route", "call-shared", r#"{"destination":"safe"}"#),
        true,
    );
    let err = Runner::run_with_options(&agent, "go", RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_reused(&err);
    assert!(entries(&calls).is_empty());
}

#[tokio::test]
async fn changed_shell_fails_before_sibling_approval_callback() {
    let checks = log();
    let c = checks.clone();
    let sibling = FunctionTool::<()>::new_no_args("sibling_tool", "", |_c| async move {
        Ok("should-not-run".to_string())
    })
    .with_needs_approval_fn(move |_ctx, _args: Value, call_id: String| {
        let c = c.clone();
        async move {
            c.lock().unwrap().push(call_id);
            false
        }
    });
    let shell = ShellTool::new(|_req| async { Ok("should-not-run") }).with_needs_approval(true);
    let model = model();
    model.add_turn(vec![
        function_call_with_id("sibling_tool", "{}", "call-sibling"),
        OutputItem::from_value(
            json!({"type": "shell_call", "call_id": "call-reused", "status": "completed",
            "action": {"type": "exec", "commands": ["echo changed"], "timeout_ms": 1000}}),
        ),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_tool(sibling)
        .with_tool(shell)
        .with_model_instance(model.clone());
    let ctx = RunContext::new(());
    let mut approved = FunctionCall::new("shell", "");
    approved.call_id = "call-reused".into();
    approved.status = Some("completed".into());
    approved.extra.insert("type".into(), json!("shell_call"));
    approved.extra.insert(
        "action".into(),
        json!({"type": "exec", "commands": ["echo safe"], "timeout_ms": 1000}),
    );
    ctx.approve_tool(
        &ToolApprovalItem {
            agent: "agent".into(),
            raw_item: approved,
            tool_name: "shell".into(),
            tool_origin: None,
        },
        false,
    );
    let err = Runner::run_with_options(&agent, "go", RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_reused(&err);
    assert!(entries(&checks).is_empty());
}

#[tokio::test]
async fn changed_nested_parent_fails_before_tool_inventory_callbacks() {
    let executed = log();
    let inner_model = model();
    inner_model.add_turn(vec![function_call_with_id(
        "inner_hitl_tool",
        "{}",
        "inner-1",
    )]);
    let inner = Arc::new(
        Agent::<()>::new("Inner")
            .with_tool(operate("inner_hitl_tool", &executed))
            .with_model_instance(inner_model.clone()),
    );
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Inner agent tool with HITL")
            .needs_approval(true),
    );
    let enabled = log();
    let en = enabled.clone();
    let observer =
        FunctionTool::<()>::new_no_args(
            "observer",
            "",
            |_c| async move { Ok("unused".to_string()) },
        )
        .with_is_enabled_fn(move |_ctx, agent| {
            let en = en.clone();
            async move {
                en.lock().unwrap().push(agent);
                true
            }
        });
    let outer_model = model();
    outer_model.add_turn(vec![function_call_with_id(
        "inner_agent_tool",
        r#"{"input":"safe"}"#,
        "outer-1",
    )]);
    let outer = Agent::<()>::new("Outer")
        .with_tool(agent_tool)
        .with_tool(observer)
        .with_model_instance(outer_model.clone());
    let first = Runner::run(&outer, "start").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let second = Runner::run(&outer, state).await.unwrap();
    assert_eq!(second.interruptions[0].tool_name, "inner_hitl_tool");

    let mut state = second.to_state();
    for response in &mut state.model_responses {
        for out in &mut response.output {
            if let OutputItem::FunctionCall(fc) = out
                && fc.call_id == "outer-1"
            {
                fc.arguments = r#"{"input":"evil"}"#.into();
            }
        }
    }
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    enabled.lock().unwrap().clear();
    let err = Runner::run(&outer, state).await.unwrap_err();
    assert_reused(&err);
    assert!(entries(&enabled).is_empty(), "{:?}", entries(&enabled));
}
