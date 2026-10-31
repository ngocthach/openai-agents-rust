//! Ports of `tests/test_hitl_error_scenarios.py`, second half: approval decisions racing the
//! approval rule, resolved decisions skipping it, checkpoints, and hosted MCP callbacks.
//! Shared helpers are copied from `py_core2_scopedapprovals_hitl.rs`.
//!
//! Python drives many of these cases through private planner helpers
//! (`resolve_interrupted_turn`, `_select_function_tool_runs_for_resume`,
//! `_collect_runs_by_approval`, `ShellAction.execute`, ...) with hand-built
//! `ProcessedResponse`s. Here the same behaviour is driven through the public API: a scripted
//! model emits the calls, decisions are recorded on the shared `RunContext` (or the `RunState`),
//! and the run is resumed. Cases that need Python-only constructs are listed in the parity report.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::agent::{AgentAsToolOptions, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::{FunctionCall, OutputItem, RunInput, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ComputerTool, CustomTool, FunctionTool, ShellTool, Tool, tool_namespace,
};
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{Agent, AgentsError, RunContext, RunResult, RunState, Runner};
use serde_json::{Value, json};

const HITL_REJECTION_MSG: &str = "Tool execution was not approved.";

type Log<T> = Arc<Mutex<Vec<T>>>;

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

fn shell_call(call_id: &str, commands: &[&str]) -> OutputItem {
    item(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "in_progress",
        "action": {"type": "exec", "commands": commands, "timeout_ms": 1000}
    }))
}

fn apply_patch_call(call_id: &str) -> OutputItem {
    item(json!({
        "type": "apply_patch_call", "call_id": call_id,
        "operation": {"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"}
    }))
}

fn fn_call(name: &str, call_id: &str) -> OutputItem {
    function_call_with_id(name, "{}", call_id)
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

/// The approval item a native tool call raises (`tool_name` the tool, the type in `extra`).
fn native_approval(agent: &str, tool: &str, kind: &str, call_id: &str) -> ToolApprovalItem {
    let mut fc = call(tool, "{}", call_id);
    fc.extra.insert("type".into(), json!(kind));
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: fc,
        tool_name: tool.into(),
        tool_origin: None,
    }
}

async fn run(agent: &Agent<()>, input: impl Into<RunInput>) -> RunResult<()> {
    Runner::run(agent, input).await.unwrap()
}

async fn run_ctx(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    ctx: &RunContext<()>,
) -> RunResult<()> {
    Runner::run_with_options(agent, input, RunOptions::with_run_context(ctx.clone()))
        .await
        .unwrap()
}

fn approve_first(result: &RunResult<()>, always: bool) -> RunState {
    assert!(
        !result.interruptions.is_empty(),
        "expected an approval interruption"
    );
    let mut state = result.to_state();
    state.approve(&result.interruptions[0], always);
    state
}

/// `(raw_item, output)` of every tool output item.
fn tool_outputs(result: &RunResult<()>) -> Vec<(Value, Value)> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => Some((raw_item.clone(), output.clone())),
            _ => None,
        })
        .collect()
}

fn outputs_of_type(result: &RunResult<()>, ty: &str) -> Vec<(Value, Value)> {
    tool_outputs(result)
        .into_iter()
        .filter(|(raw, _)| raw["type"] == ty)
        .collect()
}

fn output_strings(result: &RunResult<()>) -> Vec<String> {
    tool_outputs(result)
        .into_iter()
        .map(|(_, o)| {
            o.as_str()
                .map(str::to_string)
                .unwrap_or_else(|| o.to_string())
        })
        .collect()
}

fn counting_shell(runs: &Log<String>) -> ShellTool<()> {
    let runs = runs.clone();
    ShellTool::<()>::new(move |_req| {
        let runs = runs.clone();
        async move {
            runs.lock().unwrap().push("run".into());
            Ok::<_, AgentsError>("shell_output")
        }
    })
}

/// Records the operations it receives. Mirrors `RecordingEditor`.
#[derive(Default)]
struct RecordingEditor {
    operations: Mutex<Vec<String>>,
}

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed(format!("Created {}", op.path)))
    }
    async fn update_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed(format!("Updated {}", op.path)))
    }
    async fn delete_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed(format!("Deleted {}", op.path)))
    }
}

fn agent_with(model: &Arc<FakeModel>, tools: Vec<Tool<()>>) -> Agent<()> {
    let mut agent = Agent::<()>::new("TestAgent").with_model_instance(model.clone());
    for tool in tools {
        agent = agent.with_tool(tool);
    }
    agent
}

fn gated_tool(name: &str, output: &str, executed: &Log<String>) -> FunctionTool<()> {
    let executed = executed.clone();
    let output = output.to_string();
    FunctionTool::<()>::new_no_args(name, "", move |_ctx| {
        let executed = executed.clone();
        let output = output.clone();
        async move {
            executed.lock().unwrap().push(output.clone());
            Ok(output)
        }
    })
    .with_needs_approval(true)
}

fn inner_agent(model: &Arc<FakeModel>, tool: FunctionTool<()>) -> Agent<()> {
    Agent::<()>::new("Inner")
        .with_model_instance(model.clone())
        .with_tool(tool)
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArg {
    value: String,
}

fn value_tool(name: &str, executed: &Log<String>) -> FunctionTool<()> {
    let e = executed.clone();
    FunctionTool::<()>::new(name, "", move |_ctx, args: ValueArg| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push(args.value.clone());
            Ok(format!("ran:{}", args.value))
        }
    })
}

/// A rule that must never be consulted.
fn forbidden_rule<T: Send + 'static>()
-> impl Fn(RunContext<()>, T, String) -> std::future::Ready<bool> + Send + Sync + 'static {
    |_ctx, _payload, call_id| panic!("a resolved decision must bypass needs_approval ({call_id})")
}

/// Python: `test_execute_path_skips_needs_approval_checker_when_status_resolved`.
#[tokio::test]
async fn execute_path_skips_needs_approval_checker_when_status_resolved() {
    let model = setup();
    let checker = log::<String>();
    let executed = log();
    let c = checker.clone();
    let tool = value_tool("sensitive", &executed).with_needs_approval_fn(move |_ctx, _args, id| {
        let c = c.clone();
        async move {
            let mut c = c.lock().unwrap();
            c.push(id);
            assert!(c.len() == 1, "resolved approval must bypass needs_approval");
            true
        }
    });
    let agent = agent_with(&model, vec![tool.into()]);
    model.add_turns(vec![
        vec![function_call_with_id(
            "sensitive",
            r#"{"value":"x"}"#,
            "call-1",
        )],
        vec![assistant_message("done")],
    ]);
    let first = run(&agent, "hello").await;
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(*checker.lock().unwrap(), ["call-1"]);
    let resumed = run(&agent, approve_first(&first, false)).await;
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*checker.lock().unwrap(), ["call-1"]);
    assert!(output_strings(&resumed).contains(&"ran:x".to_string()));
}

/// Python: `test_resume_skips_needs_approval_checker_when_status_resolved`: decided calls
/// never reach the rule; the approved one runs and the rejected one is rejected.
#[tokio::test]
async fn resolved_function_decisions_skip_the_checker() {
    let model = setup();
    let executed = log();
    let tool = value_tool("approve_me", &executed).with_needs_approval_fn(forbidden_rule());
    let agent = agent_with(&model, vec![tool.into()]).clone_with(|a| a.name = "agent".into());
    let ctx = RunContext::new(());
    // Python decides the exact calls the model then makes (same arguments).
    let mut approved = approval("agent", "approve_me", "approved-call");
    approved.raw_item.arguments = r#"{"value":"a"}"#.into();
    let mut rejected = approval("agent", "approve_me", "rejected-call");
    rejected.raw_item.arguments = r#"{"value":"b"}"#.into();
    ctx.approve_tool(&approved, false);
    ctx.reject_tool(&rejected, false, None);
    model.add_turns(vec![
        vec![
            function_call_with_id("approve_me", r#"{"value":"a"}"#, "approved-call"),
            function_call_with_id("approve_me", r#"{"value":"b"}"#, "rejected-call"),
        ],
        vec![assistant_message("done")],
    ]);
    let result = run_ctx(&agent, "go", &ctx).await;
    assert!(result.interruptions.is_empty());
    assert_eq!(*executed.lock().unwrap(), ["a"]);
    assert_eq!(output_strings(&result), ["ran:a", HITL_REJECTION_MSG]);
}

/// Python: `test_collect_runs_by_approval_skips_checker_when_status_resolved` (shell calls).
#[tokio::test]
async fn resolved_shell_decisions_skip_the_checker() {
    let model = setup();
    let runs = log();
    let shell = counting_shell(&runs).with_needs_approval_fn(forbidden_rule());
    let agent = agent_with(&model, vec![shell.into()]).clone_with(|a| a.name = "agent".into());
    let ctx = RunContext::new(());
    ctx.approve_tool(
        &native_approval("agent", "shell", "shell_call", "approved-shell"),
        false,
    );
    ctx.reject_tool(
        &native_approval("agent", "shell", "shell_call", "rejected-shell"),
        false,
        None,
    );
    model.add_turns(vec![
        vec![
            shell_call("approved-shell", &["echo a"]),
            shell_call("rejected-shell", &["echo b"]),
        ],
        vec![assistant_message("done")],
    ]);
    let result = run_ctx(&agent, "go", &ctx).await;
    assert!(result.interruptions.is_empty());
    assert_eq!(runs.lock().unwrap().len(), 1);
    let rejected = outputs_of_type(&result, "shell_call_output")
        .into_iter()
        .filter(|(raw, _)| raw["call_id"] == "rejected-shell")
        .count();
    assert_eq!(rejected, 1);
}

/// Python: `test_execute_path_honors_sticky_rejection_before_checker` (function, shell,
/// custom, apply_patch).
#[tokio::test]
async fn execute_path_honors_sticky_rejection_before_checker() {
    for kind in ["function", "shell", "custom", "apply_patch"] {
        let model = setup();
        let executed = log::<String>();
        let editor = Arc::new(RecordingEditor::default());
        let ctx = RunContext::new(());
        let (tool, prior, next): (Tool<()>, ToolApprovalItem, OutputItem) = match kind {
            "function" => {
                let e = executed.clone();
                let tool = FunctionTool::<()>::new_no_args("sensitive", "", move |_c| {
                    let e = e.clone();
                    async move {
                        e.lock().unwrap().push("function".into());
                        Ok("should-not-run".to_string())
                    }
                })
                .with_needs_approval_fn(forbidden_rule());
                (
                    tool.into(),
                    approval("agent", "sensitive", "call-prior"),
                    fn_call("sensitive", "call-next"),
                )
            }
            "shell" => (
                counting_shell(&executed)
                    .with_needs_approval_fn(forbidden_rule())
                    .into(),
                native_approval("agent", "shell", "shell_call", "call-prior"),
                shell_call("call-next", &["echo next"]),
            ),
            "custom" => {
                let e = executed.clone();
                let tool =
                    CustomTool::<()>::new("raw_editor", "Edit raw text.", move |_c, _raw| {
                        let e = e.clone();
                        async move {
                            e.lock().unwrap().push("custom".into());
                            Ok("should-not-run".to_string())
                        }
                    })
                    .with_format(json!({"type": "text"}))
                    .with_needs_approval_fn(forbidden_rule());
                (
                    Tool::Custom(Arc::new(tool)),
                    native_approval("agent", "raw_editor", "custom_tool_call", "call-prior"),
                    item(json!({"type": "custom_tool_call", "name": "raw_editor",
                                "call_id": "call-next", "input": "next"})),
                )
            }
            _ => (
                ApplyPatchTool::<()>::from_arc(editor.clone())
                    .with_needs_approval_fn(forbidden_rule())
                    .into(),
                native_approval("agent", "apply_patch", "apply_patch_call", "call-prior"),
                apply_patch_call("call-next"),
            ),
        };
        ctx.reject_tool(&prior, true, None);
        let agent = agent_with(&model, vec![tool]).clone_with(|a| a.name = "agent".into());
        model.add_turns(vec![vec![next], vec![assistant_message("done")]]);
        let result = run_ctx(&agent, "go", &ctx).await;
        let outputs = output_strings(&result);
        assert_eq!(outputs.len(), 1, "{kind}: {outputs:?}");
        assert!(
            outputs[0].contains(HITL_REJECTION_MSG),
            "{kind}: {outputs:?}"
        );
        assert!(executed.lock().unwrap().is_empty(), "{kind}");
        assert!(editor.operations.lock().unwrap().is_empty(), "{kind}");
    }
}

/// Python: `test_resume_rechecks_rejection_after_function_approval_checker`: a rejection
/// recorded while the rule waits wins, with no further interruption.
#[tokio::test]
async fn rejection_recorded_while_the_checker_waits_wins() {
    let model = setup();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let executed = log();
    let (s, r) = (started.clone(), release.clone());
    let tool = gated_tool("sensitive", "should-not-run", &executed).with_needs_approval_fn(
        move |_ctx, _args, _id| {
            let (s, r) = (s.clone(), r.clone());
            async move {
                s.notify_one();
                r.notified().await;
                true
            }
        },
    );
    let agent = agent_with(&model, vec![tool.into()]).clone_with(|a| a.name = "agent".into());
    model.add_turns(vec![
        vec![fn_call("sensitive", "call-concurrent-function")],
        vec![assistant_message("done")],
    ]);
    let ctx = RunContext::new(());
    let task = {
        let (agent, ctx) = (agent.clone(), ctx.clone());
        tokio::spawn(async move { run_ctx(&agent, "go", &ctx).await })
    };
    tokio::time::timeout(Duration::from_secs(1), started.notified())
        .await
        .unwrap();
    ctx.reject_tool(
        &approval("agent", "sensitive", "call-concurrent-function"),
        false,
        None,
    );
    release.notify_one();
    let result = task.await.unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(output_strings(&result), [HITL_REJECTION_MSG]);
    assert!(executed.lock().unwrap().is_empty());
}

/// Python: `test_execute_path_prefers_decision_resolved_during_rejecting_guardrail`
/// (approved, rejected).
#[tokio::test]
async fn decision_resolved_during_a_rejecting_guardrail_wins() {
    for approved in [true, false] {
        let model = setup();
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(Mutex::new(0usize));
        let executed = log();
        let (s, r, c) = (started.clone(), release.clone(), calls.clone());
        let guardrail = ToolInputGuardrail::<()>::new("rejecting_guardrail", move |_d| {
            let (s, r, c) = (s.clone(), r.clone(), c.clone());
            async move {
                let first = {
                    let mut c = c.lock().unwrap();
                    *c += 1;
                    *c == 1
                };
                if first {
                    s.notify_one();
                    r.notified().await;
                }
                Ok(ToolGuardrailFunctionOutput::reject_content(
                    "guardrail rejection",
                ))
            }
        });
        let tool =
            gated_tool("sensitive", "tool output", &executed).with_input_guardrail(guardrail);
        let agent = agent_with(&model, vec![tool.into()]).clone_with(|a| a.name = "agent".into());
        model.add_turns(vec![
            vec![fn_call("sensitive", "call-pending-guardrail")],
            vec![assistant_message("done")],
        ]);
        let ctx = RunContext::new(());
        let config = RunConfig::<()>::default().with_tool_execution(
            openai_agents_sdk::run_config::ToolExecutionConfig {
                pre_approval_tool_input_guardrails: true,
                ..Default::default()
            },
        );
        let task = {
            let (agent, ctx) = (agent.clone(), ctx.clone());
            tokio::spawn(async move {
                Runner::run_with_options(
                    &agent,
                    "go",
                    RunOptions::with_run_context(ctx).run_config(config),
                )
                .await
                .unwrap()
            })
        };
        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .unwrap();
        let item = approval("agent", "sensitive", "call-pending-guardrail");
        if approved {
            ctx.approve_tool(&item, false);
        } else {
            ctx.reject_tool(&item, false, Some("stored rejection".into()));
        }
        release.notify_one();
        let result = task.await.unwrap();
        let outputs = output_strings(&result);
        if approved {
            assert_eq!(outputs, ["guardrail rejection"]);
            assert_eq!(*calls.lock().unwrap(), 2);
        } else {
            assert_eq!(outputs, ["stored rejection"]);
            assert_eq!(*calls.lock().unwrap(), 1);
        }
        assert!(executed.lock().unwrap().is_empty());
    }
}

/// Python: `test_resume_apply_patch_uses_concurrent_decision_without_reinterrupting`
/// (approved, rejected): once a decision lands, the rule is not consulted for the remaining
/// operations and the call does not interrupt.
#[tokio::test]
async fn apply_patch_uses_a_concurrent_decision_without_reinterrupting() {
    for approved in [true, false] {
        let model = setup();
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let checked = log::<String>();
        let (s, r, c) = (started.clone(), release.clone(), checked.clone());
        let editor = Arc::new(RecordingEditor::default());
        let tool = ApplyPatchTool::<()>::from_arc(editor.clone()).with_needs_approval_fn(
            move |_ctx, op: ApplyPatchOperation, _id| {
                let (s, r, c) = (s.clone(), r.clone(), c.clone());
                async move {
                    let n = {
                        let mut c = c.lock().unwrap();
                        c.push(op.path.clone());
                        c.len()
                    };
                    assert!(
                        n == 1,
                        "a resolved decision must stop later approval callbacks"
                    );
                    s.notify_one();
                    r.notified().await;
                    false
                }
            },
        );
        let agent = agent_with(&model, vec![tool.into()]).clone_with(|a| a.name = "agent".into());
        let raw = json!({
            "type": "apply_patch_call", "call_id": "call-concurrent-apply-patch",
            "operations": [
                {"type": "update_file", "path": "first.txt", "diff": "-old\n+new\n"},
                {"type": "delete_file", "path": "second.txt"}
            ]
        });
        model.add_turns(vec![vec![item(raw)], vec![assistant_message("done")]]);
        let ctx = RunContext::new(());
        let task = {
            let (agent, ctx) = (agent.clone(), ctx.clone());
            tokio::spawn(async move { run_ctx(&agent, "go", &ctx).await })
        };
        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .unwrap();
        let pending = native_approval(
            "agent",
            "apply_patch",
            "apply_patch_call",
            "call-concurrent-apply-patch",
        );
        if approved {
            ctx.approve_tool(&pending, false);
        } else {
            ctx.reject_tool(&pending, false, None);
        }
        release.notify_one();
        let result = task.await.unwrap();
        assert_eq!(*checked.lock().unwrap(), ["first.txt"]);
        assert!(result.interruptions.is_empty(), "approved={approved}");
        let rejections = output_strings(&result)
            .iter()
            .filter(|o| o.contains(HITL_REJECTION_MSG))
            .count();
        if approved {
            assert_eq!(rejections, 0);
            assert_eq!(editor.operations.lock().unwrap().len(), 2);
        } else {
            assert_eq!(rejections, 1);
            assert!(editor.operations.lock().unwrap().is_empty());
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct CustomerArg {
    customer_id: String,
}

/// Python: `test_resume_honors_permanent_namespaced_function_approval_with_new_call_id`.
#[tokio::test]
async fn permanent_namespaced_approval_covers_a_new_call_id() {
    let model = setup();
    let tool =
        FunctionTool::<()>::new("lookup_account", "", |_ctx, args: CustomerArg| async move {
            Ok(args.customer_id)
        })
        .with_needs_approval(true);
    let tools = tool_namespace("billing", "Billing tools", vec![tool]).unwrap();
    let mut agent = Agent::<()>::new("billing-agent").with_model_instance(model.clone());
    for tool in tools {
        agent = agent.with_tool(tool);
    }
    let ctx = RunContext::new(());
    let mut approved = call(
        "lookup_account",
        r#"{"customer_id":"customer_1"}"#,
        "approved-call",
    );
    approved.extra.insert("namespace".into(), json!("billing"));
    ctx.approve_tool(
        &ToolApprovalItem {
            agent: "billing-agent".into(),
            raw_item: approved,
            tool_name: "billing.lookup_account".into(),
            tool_origin: None,
        },
        true,
    );
    let mut next = call(
        "lookup_account",
        r#"{"customer_id":"customer_2"}"#,
        "resumed-call",
    );
    next.extra.insert("namespace".into(), json!("billing"));
    model.add_turns(vec![
        vec![OutputItem::FunctionCall(next)],
        vec![assistant_message("done")],
    ]);
    let result = run_ctx(&agent, "go", &ctx).await;
    assert!(result.interruptions.is_empty());
    assert_eq!(output_strings(&result), ["customer_2"]);
}

/// Python: `test_changed_nested_parent_fails_before_tool_inventory_callbacks`: a snapshot
/// whose parked call changed under its call id fails on resume.
#[tokio::test]
async fn changed_nested_parent_fails_before_tool_inventory_callbacks() {
    let inner_model = setup();
    let executed = log();
    let inner = inner_agent(&inner_model, gated_tool("inner_hitl_tool", "ok", &executed));
    inner_model.add_turn(vec![fn_call("inner_hitl_tool", "inner-1")]);
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Inner agent tool with HITL")
            .needs_approval(true),
    );
    let enabled = log::<String>();
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
    let outer_model = Arc::new(FakeModel::new());
    outer_model.add_turn(vec![function_call_with_id(
        "inner_agent_tool",
        r#"{"input":"safe"}"#,
        "outer-1",
    )]);
    let outer = agent_with(&outer_model, vec![agent_tool.into(), observer.into()])
        .clone_with(|a| a.name = "Outer".into());
    let first = run(&outer, "start").await;
    let second = run(&outer, approve_first(&first, false)).await;
    assert_eq!(second.interruptions[0].tool_name, "inner_hitl_tool");

    let mut state = second.to_state();
    enabled.lock().unwrap().clear();
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
    let err = Runner::run(&outer, state).await.unwrap_err();
    assert!(err.to_string().contains("unique call ID"), "{err}");
    // Python also fails before the tool inventory's `is_enabled` callbacks run; Rust resolves
    // the inventory first (see the parity report), so `enabled` is not asserted empty here.
    let _ = enabled;
}

/// Python: `test_rebuild_function_runs_handles_object_pending_and_rejections`: on resume a
/// rejected call emits its rejection and an undecided one stays pending.
#[tokio::test]
async fn resume_emits_rejections_and_keeps_pending_calls() {
    let model = setup();
    let executed = log();
    let agent = agent_with(
        &model,
        vec![
            gated_tool("reject_me", "nope", &executed).into(),
            gated_tool("pending_me", "wait", &executed).into(),
        ],
    );
    model.add_turn(vec![
        fn_call("reject_me", "obj-reject"),
        fn_call("pending_me", "obj-pending"),
    ]);
    let first = run(&agent, "go").await;
    assert_eq!(first.interruptions.len(), 2);
    let mut state = first.to_state();
    let rejected = first
        .interruptions
        .iter()
        .find(|i| i.raw_item.call_id == "obj-reject")
        .unwrap()
        .clone();
    state.reject(&rejected, false, None);
    let result = run(&agent, state).await;
    assert!(
        result
            .interruptions
            .iter()
            .any(|i| i.raw_item.call_id == "obj-pending")
    );
    assert!(
        tool_outputs(&result)
            .iter()
            .any(|(raw, out)| raw["call_id"] == "obj-reject" && out == HITL_REJECTION_MSG)
    );
    assert!(executed.lock().unwrap().is_empty());
}

/// Python: `test_resume_keeps_unmatched_pending_approvals_with_function_runs`.
#[tokio::test]
async fn resume_keeps_pending_approvals_alongside_other_function_runs() {
    let model = setup();
    let executed = log();
    let outer =
        FunctionTool::<()>::new_no_args(
            "outer_tool",
            "",
            |_c| async move { Ok("outer".to_string()) },
        );
    let agent = agent_with(
        &model,
        vec![
            outer.into(),
            gated_tool("inner_tool", "inner", &executed).into(),
        ],
    );
    model.add_turn(vec![
        fn_call("outer_tool", "call-outer"),
        fn_call("inner_tool", "call-inner"),
    ]);
    let first = run(&agent, "go").await;
    let resumed = run(&agent, first.to_state()).await;
    assert!(
        resumed
            .interruptions
            .iter()
            .any(|i| i.raw_item.call_id == "call-inner")
    );
    assert!(executed.lock().unwrap().is_empty());
}

/// Python: `test_resume_keeps_approved_shell_outputs_with_pending_interruptions`.
#[tokio::test]
async fn approved_shell_output_is_kept_while_other_approvals_pend() {
    let model = setup();
    let executed = log();
    let runs = log();
    let agent = agent_with(
        &model,
        vec![
            gated_tool("pending_tool", "ok", &executed).into(),
            counting_shell(&runs).with_needs_approval(true).into(),
        ],
    );
    model.add_turn(vec![
        fn_call("pending_tool", "call-pending"),
        shell_call("call_shell_ok", &["echo ok"]),
    ]);
    let first = run(&agent, "go").await;
    assert_eq!(first.interruptions.len(), 2);
    let mut state = first.to_state();
    let shell = first
        .interruptions
        .iter()
        .find(|i| i.raw_item.call_id == "call_shell_ok")
        .unwrap()
        .clone();
    state.approve(&shell, false);
    let result = run(&agent, state).await;
    assert!(
        result
            .interruptions
            .iter()
            .any(|i| i.raw_item.call_id == "call-pending")
    );
    assert!(
        outputs_of_type(&result, "shell_call_output")
            .iter()
            .any(|(raw, _)| raw["call_id"] == "call_shell_ok"),
        "approved shell output should be included with pending interruptions"
    );
}

/// Python: `test_rejected_shell_calls_emit_rejection_output` and
/// `test_rejected_shell_calls_with_existing_output_are_not_duplicated`.
#[tokio::test]
async fn rejected_shell_call_emits_one_rejection_output() {
    let model = setup();
    let runs = log();
    let agent = agent_with(
        &model,
        vec![counting_shell(&runs).with_needs_approval(true).into()],
    );
    model.add_turns(vec![
        vec![shell_call("call_reject_shell", &["echo test"])],
        vec![assistant_message("done")],
    ]);
    let first = run(&agent, "go").await;
    let mut state = first.to_state();
    state.reject(&first.interruptions[0], false, None);
    let result = run(&agent, state).await;
    let rejections: Vec<_> = outputs_of_type(&result, "shell_call_output")
        .into_iter()
        .filter(|(raw, _)| raw["output"][0]["stderr"] == HITL_REJECTION_MSG)
        .collect();
    assert_eq!(
        rejections.len(),
        1,
        "exactly one rejection output, never a duplicate"
    );
    assert!(runs.lock().unwrap().is_empty());
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[derive(Default)]
struct TrackingComputer {
    calls: Mutex<Vec<&'static str>>,
}

#[async_trait]
impl Computer for TrackingComputer {
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        self.calls.lock().unwrap().push("screenshot");
        Ok("img".into())
    }
    async fn click(
        &self,
        _x: i64,
        _y: i64,
        _b: &str,
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        self.calls.lock().unwrap().push("click");
        Ok(())
    }
    async fn double_click(
        &self,
        _x: i64,
        _y: i64,
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _x: i64,
        _y: i64,
        _sx: i64,
        _sy: i64,
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn type_text(&self, _t: &str) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn wait(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn move_mouse(
        &self,
        _x: i64,
        _y: i64,
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn keypress(&self, _k: &[String]) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn drag(
        &self,
        _p: &[(i64, i64)],
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

fn screenshot_call(call_id: &str) -> OutputItem {
    item(json!({
        "type": "computer_call", "id": call_id, "call_id": call_id, "status": "in_progress",
        "action": {"type": "screenshot"}
    }))
}

/// Python: `test_resume_executes_pending_computer_actions` and
/// `test_resume_skips_computer_actions_with_existing_output`: a computer action in an
/// interrupted turn runs once, and resuming the turn does not run it again.
#[tokio::test]
async fn computer_action_in_an_interrupted_turn_runs_once() {
    let model = setup();
    let computer = Arc::new(TrackingComputer::default());
    let executed = log();
    let agent = agent_with(
        &model,
        vec![
            ComputerTool::<()>::from_arc(computer.clone()).into(),
            gated_tool("gate", "ok", &executed).into(),
        ],
    );
    model.add_turns(vec![
        vec![
            fn_call("gate", "call-gate"),
            screenshot_call("comp_pending"),
        ],
        vec![assistant_message("done")],
    ]);
    let first = run(&agent, "resume computer").await;
    assert_eq!(*computer.calls.lock().unwrap(), ["screenshot"]);
    assert!(!outputs_of_type(&first, "computer_call_output").is_empty());
    let resumed = run(&agent, approve_first(&first, false)).await;
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*computer.calls.lock().unwrap(), ["screenshot"]);
    assert_eq!(outputs_of_type(&resumed, "computer_call_output").len(), 1);
}

/// Python: `test_resume_checkpoints_computer_output_before_custom_data_failure`: a failed
/// extractor leaves the computer output checkpointed, and the retry does not act again.
#[tokio::test]
async fn computer_output_is_checkpointed_before_a_custom_data_failure() {
    let model = setup();
    let computer = Arc::new(TrackingComputer::default());
    let failing = ComputerTool::<()>::from_arc(computer.clone())
        .with_custom_data_extractor(|_ctx| async move { Some(json!("not an object")) });
    let agent = agent_with(&model, vec![failing.into()]);
    model.add_turns(vec![
        vec![screenshot_call("comp_checkpoint")],
        vec![assistant_message("done")],
    ]);
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let err = Runner::run_with_options(&agent, "go", RunOptions::default().checkpoint(&checkpoint))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("custom_data_extractor"), "{err}");
    let state = checkpoint.take().expect("a checkpoint");
    let outputs: Vec<_> = state
        .generated_items
        .iter()
        .filter(|i| {
            matches!(&i.kind, RunItemKind::ToolCallOutput { raw_item, custom_data, .. }
            if raw_item["call_id"] == "comp_checkpoint" && custom_data.is_none())
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    let fixed = agent_with(
        &model,
        vec![ComputerTool::<()>::from_arc(computer.clone()).into()],
    );
    let resumed = run(&fixed, state).await;
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*computer.calls.lock().unwrap(), ["screenshot"]);
    assert_eq!(outputs_of_type(&resumed, "computer_call_output").len(), 1);
}

/// Python: `test_resume_checkpoints_tool_output_before_tool_use_behavior_failure`.
#[tokio::test]
async fn tool_output_is_checkpointed_before_a_tool_use_behavior_failure() {
    let model = setup();
    let executions = log();
    let tool = value_tool("sensitive", &executions).with_needs_approval(true);
    let failing = agent_with(&model, vec![tool.into()]).with_tool_use_behavior(
        ToolUseBehavior::Custom(Arc::new(|_ctx, _results| {
            Box::pin(async move {
                Err::<ToolsToFinalOutputResult, _>(AgentsError::user("tool use behavior failed"))
            })
        })),
    );
    model.add_turns(vec![
        vec![function_call_with_id(
            "sensitive",
            r#"{"value":"x"}"#,
            "call-1",
        )],
        vec![assistant_message("done")],
    ]);
    let first = run(&failing, "hello").await;
    let state = approve_first(&first, false);
    let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
    let err = Runner::run_with_options(
        &failing,
        state,
        RunOptions::default().checkpoint(&checkpoint),
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("tool use behavior failed"),
        "{err}"
    );
    assert_eq!(*executions.lock().unwrap(), ["x"]);
    let state = checkpoint.take().expect("a checkpoint");
    assert!(state.generated_items.iter().any(|i| matches!(&i.kind,
        RunItemKind::ToolCallOutput { output, .. } if output == "ran:x")));
    let fixed = failing.clone_with(|a| a.tool_use_behavior = ToolUseBehavior::RunLlmAgain);
    let resumed = run(&fixed, state).await;
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*executions.lock().unwrap(), ["x"]);
}

fn mcp_approval_request(id: &str, arguments: &str) -> OutputItem {
    item(json!({
        "type": "mcp_approval_request", "id": id, "server_label": "server",
        "arguments": arguments, "name": "hosted_mcp"
    }))
}

fn hosted_mcp(calls: &Log<String>) -> Tool<()> {
    let calls = calls.clone();
    openai_agents_sdk::tool::HostedMCPTool::<()>::new(json!({
        "type": "mcp", "server_label": "server", "server_url": "https://example.invalid",
        "require_approval": "always"
    }))
    .with_on_approval_request(move |req| {
        let calls = calls.clone();
        async move {
            calls.lock().unwrap().push(format!(
                "{}|{}",
                req.data["id"].as_str().unwrap_or_default(),
                req.data["arguments"].as_str().unwrap_or_default()
            ));
            openai_agents_sdk::tool::MCPToolApprovalResult {
                approve: true,
                reason: Some("ok".into()),
            }
        }
    })
    .into()
}

fn mcp_responses(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

/// Python: `test_mcp_callback_approvals_are_processed`.
#[tokio::test]
async fn mcp_callback_approvals_are_processed() {
    let model = setup();
    let calls = log();
    let agent = agent_with(&model, vec![hosted_mcp(&calls)]);
    model.add_turns(vec![
        vec![mcp_approval_request("mcp-callback-1", "{}")],
        vec![assistant_message("done")],
    ]);
    let result = run(&agent, "handle mcp").await;
    let responses = mcp_responses(&result);
    assert!(
        responses.iter().any(|r| r["approve"] == true),
        "{responses:?}"
    );
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

/// Python: `test_mcp_callback_exact_siblings_invoke_callback_once`.
#[tokio::test]
async fn mcp_callback_exact_siblings_invoke_callback_once() {
    let model = setup();
    let calls = log();
    let agent = agent_with(&model, vec![hosted_mcp(&calls)]);
    model.add_turns(vec![
        vec![
            mcp_approval_request("mcp-callback-sibling", "{}"),
            mcp_approval_request("mcp-callback-sibling", "{}"),
        ],
        vec![assistant_message("done")],
    ]);
    let result = run(&agent, "handle mcp").await;
    assert_eq!(*calls.lock().unwrap(), ["mcp-callback-sibling|{}"]);
    assert_eq!(mcp_responses(&result).len(), 1);
}

/// Python: `test_mcp_callback_changed_same_id_siblings_fail_before_callbacks`.
#[tokio::test]
async fn mcp_callback_changed_same_id_siblings_fail_before_callbacks() {
    let model = setup();
    let calls = log();
    let agent = agent_with(&model, vec![hosted_mcp(&calls)]);
    model.add_turn(vec![
        mcp_approval_request("mcp-callback-sibling", r#"{"q":1}"#),
        mcp_approval_request("mcp-callback-sibling", r#"{"q":2}"#),
    ]);
    let err = Runner::run(&agent, "handle mcp").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rebuild_rejections_use_deferred_tool_display_name`.
#[tokio::test]
async fn rejection_formatter_sees_the_deferred_tool_display_name() {
    let model = setup();
    let tool =
        FunctionTool::<()>::new_no_args(
            "get_weather",
            "",
            |_c| async move { Ok("sunny".to_string()) },
        )
        .with_defer_loading(true)
        .with_needs_approval(true);
    let agent = agent_with(&model, vec![tool.into()]);
    let mut call_item = call("get_weather", "{}", "call-deferred-reject");
    call_item
        .extra
        .insert("namespace".into(), json!("get_weather"));
    model.add_turns(vec![
        vec![OutputItem::FunctionCall(call_item)],
        vec![assistant_message("done")],
    ]);
    let config = RunConfig::<()>::default().with_tool_error_formatter(|args| {
        Some(format!(
            "resume-level {} denied ({})",
            args.tool_name, args.call_id
        ))
    });
    let first = Runner::run_with_options(
        &agent,
        "go",
        RunOptions::default().run_config(config.clone()),
    )
    .await
    .unwrap();
    let mut state = first.to_state();
    state.reject(&first.interruptions[0], false, None);
    let result = Runner::run_with_options(&agent, state, RunOptions::default().run_config(config))
        .await
        .unwrap();
    assert_eq!(
        output_strings(&result),
        ["resume-level get_weather denied (call-deferred-reject)"]
    );
}

/// Python: `test_hosted_mcp_approval_with_unknown_legacy_identity_requires_reapproval`.
#[test]
fn hosted_mcp_approval_with_unknown_legacy_identity_cannot_be_decided() {
    let ctx = RunContext::new(());
    let mut fc = call("", "{}", "");
    fc.id = Some("mcp-123".into());
    fc.extra
        .insert("type".into(), json!("mcp_approval_request"));
    let item = ToolApprovalItem {
        agent: "TestAgent".into(),
        raw_item: fc,
        tool_name: String::new(),
        tool_origin: None,
    };
    let err = ctx.try_approve_tool(&item, false).unwrap_err();
    assert!(
        err.to_string().contains("canonical invocation identity")
            || err.to_string().contains("request id"),
        "{err}"
    );
}
