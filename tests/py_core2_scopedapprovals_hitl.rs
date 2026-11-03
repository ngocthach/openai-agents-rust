//! Ports of `tests/test_hitl_error_scenarios.py`: human-in-the-loop edge cases across run,
//! interruption and resume.
//!
//! Python drives many of these cases through private planner helpers
//! (`resolve_interrupted_turn`, `_select_function_tool_runs_for_resume`,
//! `_collect_runs_by_approval`, `ShellAction.execute`, ...) with hand-built
//! `ProcessedResponse`s. Here the same behaviour is driven through the public API: a scripted
//! model emits the calls, decisions are recorded on the shared `RunContext` (or the `RunState`),
//! and the run is resumed. Cases that need Python-only constructs are listed in the parity report.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::{FunctionCall, OutputItem, RunInput, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{ApplyPatchTool, FunctionTool, LocalShellTool, ShellTool, Tool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::{Agent, AgentsError, RunResult, RunState, Runner};
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

async fn run_with(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> openai_agents_sdk::Result<RunResult<()>> {
    if streamed {
        let mut stream = Runner::run_streamed_with_options(agent, input, options);
        stream.finish().await?;
        Ok(stream.into_result().expect("streamed result"))
    } else {
        Runner::run_with_options(agent, input, options).await
    }
}

async fn run(agent: &Agent<()>, input: impl Into<RunInput>) -> RunResult<()> {
    Runner::run(agent, input).await.unwrap()
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

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
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

/// Python: `test_resumed_hitl_executes_approved_tools` (shell, apply_patch).
#[tokio::test]
async fn resumed_hitl_executes_approved_tools() {
    // Shell.
    let model = setup();
    let runs = log();
    let shell = counting_shell(&runs).with_needs_approval(true);
    let agent = agent_with(&model, vec![shell.into()]);
    model.add_turn(vec![shell_call("call_shell_1", &["echo test"])]);
    let first = run(&agent, "run shell command").await;
    let state = approve_first(&first, true);
    model.add_turn(vec![assistant_message("done")]);
    let result = run(&agent, state).await;
    let outputs = outputs_of_type(&result, "shell_call_output");
    assert!(!outputs.is_empty(), "shell should run after approval");
    assert!(
        outputs
            .iter()
            .any(|(raw, _)| raw.to_string().contains("shell_output"))
    );

    // Apply patch.
    let model = setup();
    let editor = Arc::new(RecordingEditor::default());
    let tool = ApplyPatchTool::<()>::from_arc(editor.clone()).with_needs_approval(true);
    let agent = agent_with(&model, vec![tool.into()]);
    model.add_turn(vec![apply_patch_call("call_apply_1")]);
    let first = run(&agent, "update file").await;
    let state = approve_first(&first, true);
    model.add_turn(vec![assistant_message("done")]);
    let result = run(&agent, state).await;
    assert!(!outputs_of_type(&result, "apply_patch_call_output").is_empty());
    assert!(!editor.operations.lock().unwrap().is_empty());
}

/// Python: `test_resuming_skips_approvals_for_non_hitl_tools` (shell, apply_patch). Also
/// covers `test_resume_skips_shell_calls_with_existing_output`: the auto-approved call that
/// already ran is not run again on resume.
#[tokio::test]
async fn resuming_skips_approvals_for_non_hitl_tools() {
    for kind in ["shell", "apply_patch"] {
        let model = setup();
        let runs = log();
        let editor = Arc::new(RecordingEditor::default());
        let (auto_tool, raw_call, output_type): (Tool<()>, OutputItem, &str) = if kind == "shell" {
            (
                counting_shell(&runs).into(),
                shell_call("call_shell_auto", &["echo auto"]),
                "shell_call_output",
            )
        } else {
            (
                ApplyPatchTool::<()>::from_arc(editor.clone()).into(),
                apply_patch_call("call_apply_auto"),
                "apply_patch_call_output",
            )
        };
        let executed = log();
        let approval_tool = gated_tool("needs_hitl", "approved", &executed);
        let agent = agent_with(&model, vec![auto_tool, approval_tool.into()]);
        model.add_turns(vec![
            vec![fn_call("needs_hitl", "call-func-auto"), raw_call],
            vec![assistant_message("done")],
        ]);
        let first = run(&agent, "resume approvals").await;
        assert!(
            !first.interruptions.is_empty(),
            "function tool should require approval"
        );
        let resumed = run(&agent, approve_first(&first, true)).await;
        assert!(resumed.interruptions.is_empty(), "{kind}");
        // The resumed result carries the whole turn, including the output of the call that
        // ran before the pause.
        assert_eq!(
            outputs_of_type(&resumed, output_type).len(),
            1,
            "{kind} should run exactly once"
        );
        if kind == "shell" {
            assert_eq!(runs.lock().unwrap().len(), 1);
        } else {
            assert_eq!(editor.operations.lock().unwrap().len(), 1);
        }
    }
}

fn inner_agent(model: &Arc<FakeModel>, tool: FunctionTool<()>) -> Agent<()> {
    Agent::<()>::new("Inner")
        .with_model_instance(model.clone())
        .with_tool(tool)
}

/// Python: `test_nested_agent_tool_resumes_after_rejection`.
#[tokio::test]
async fn nested_agent_tool_resumes_after_rejection() {
    let inner_model = setup();
    let executed = log();
    let inner = inner_agent(&inner_model, gated_tool("inner_hitl_tool", "ok", &executed));
    inner_model.add_turns(vec![
        vec![fn_call("inner_hitl_tool", "inner-1")],
        vec![fn_call("inner_hitl_tool", "inner-2")],
        vec![assistant_message("done")],
    ]);
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Inner agent tool with HITL")
            .needs_approval(true),
    );
    let outer_model = Arc::new(FakeModel::new());
    outer_model.add_turn(vec![function_call_with_id(
        "inner_agent_tool",
        r#"{"input":"hi"}"#,
        "outer-1",
    )]);
    let outer =
        agent_with(&outer_model, vec![agent_tool.into()]).clone_with(|a| a.name = "Outer".into());

    let first = run(&outer, "start").await;
    assert_eq!(first.interruptions[0].tool_name, "inner_agent_tool");
    let second = run(&outer, approve_first(&first, true)).await;
    assert_eq!(second.interruptions[0].tool_name, "inner_hitl_tool");
    let mut state = second.to_state();
    state.reject(&second.interruptions[0], false, None);
    let third = run(&outer, state).await;
    assert!(
        !third.interruptions.is_empty(),
        "nested agent should ask again"
    );
    assert_eq!(third.interruptions[0].tool_name, "inner_hitl_tool");
    assert_eq!(third.interruptions[0].raw_item.call_id, "inner-2");
    let short_circuited = tool_outputs(&third)
        .into_iter()
        .any(|(raw, out)| raw["call_id"] == "outer-1" && out == HITL_REJECTION_MSG);
    assert!(
        !short_circuited,
        "nested rejection must not short-circuit the agent tool"
    );
}

/// Python: `test_nested_agent_tool_interruptions_remain_distinct_across_outer_calls`.
#[tokio::test]
async fn nested_agent_tool_interruptions_remain_distinct_across_outer_calls() {
    let inner_model = setup();
    let executed = log();
    let inner = inner_agent(&inner_model, gated_tool("inner_hitl_tool", "ok", &executed));
    inner_model.add_turns(vec![
        vec![fn_call("inner_hitl_tool", "inner-1")],
        vec![fn_call("inner_hitl_tool", "inner-2")],
    ]);
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Inner agent tool"),
    );
    let outer_model = Arc::new(FakeModel::new());
    outer_model.add_turn(vec![
        function_call_with_id("inner_agent_tool", r#"{"input":"a"}"#, "outer-a"),
        function_call_with_id("inner_agent_tool", r#"{"input":"b"}"#, "outer-b"),
    ]);
    let outer = agent_with(&outer_model, vec![agent_tool.into()]);
    let result = run(&outer, "start").await;
    let nested = result
        .interruptions
        .iter()
        .filter(|i| i.tool_name == "inner_hitl_tool")
        .count();
    assert_eq!(nested, 2);
}

/// Python: `test_nested_agent_tool_does_not_inherit_parent_approvals`.
#[tokio::test]
async fn nested_agent_tool_does_not_inherit_parent_approvals() {
    let inner_model = setup();
    let executed = log();
    let inner = inner_agent(&inner_model, gated_tool("shared_tool", "inner", &executed));
    inner_model.add_turn(vec![fn_call("shared_tool", "dup")]);
    let agent_tool = inner.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Inner agent tool"),
    );
    let outer_model = Arc::new(FakeModel::new());
    outer_model.add_turns(vec![
        vec![fn_call("shared_tool", "dup")],
        vec![function_call_with_id(
            "inner_agent_tool",
            r#"{"input":"hi"}"#,
            "outer-agent",
        )],
    ]);
    let outer = agent_with(
        &outer_model,
        vec![
            gated_tool("shared_tool", "outer", &executed).into(),
            agent_tool.into(),
        ],
    )
    .clone_with(|a| a.name = "Outer".into());
    let first = run(&outer, "start").await;
    assert!(!first.interruptions.is_empty());
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let second = run(&outer, state).await;
    assert!(
        second
            .interruptions
            .iter()
            .any(|i| i.tool_name == "shared_tool" && i.agent == "Inner"),
        "inner approvals must not inherit the parent's: {:?}",
        second.interruptions
    );
}

/// Python: `test_pending_approvals_stay_pending_on_resume` (shell, apply_patch).
#[tokio::test]
async fn pending_approvals_stay_pending_on_resume() {
    for kind in ["shell", "apply_patch"] {
        let model = setup();
        let runs = log();
        let editor = Arc::new(RecordingEditor::default());
        let (tool, raw, output_type, name): (Tool<()>, OutputItem, &str, &str) = if kind == "shell"
        {
            (
                counting_shell(&runs).with_needs_approval(true).into(),
                shell_call("call_shell_pending", &["echo pending"]),
                "shell_call_output",
                "shell",
            )
        } else {
            (
                ApplyPatchTool::<()>::from_arc(editor.clone())
                    .with_needs_approval(true)
                    .into(),
                apply_patch_call("call_apply_pending"),
                "apply_patch_call_output",
                "apply_patch",
            )
        };
        let agent = agent_with(&model, vec![tool]);
        model.add_turn(vec![raw]);
        let first = run(&agent, "resume pending approval").await;
        let resumed = run(&agent, first.to_state()).await;
        assert!(
            resumed.interruptions.iter().any(|i| i.tool_name == name),
            "{kind}: pending approval should remain"
        );
        assert!(outputs_of_type(&resumed, output_type).is_empty(), "{kind}");
        assert!(editor.operations.lock().unwrap().is_empty(), "{kind}");
        assert!(runs.lock().unwrap().is_empty(), "{kind}");
    }
}

/// Python: `test_resume_does_not_duplicate_pending_shell_approvals`.
#[tokio::test]
async fn resume_does_not_duplicate_pending_shell_approvals() {
    let model = setup();
    let runs = log();
    let agent = agent_with(
        &model,
        vec![counting_shell(&runs).with_needs_approval(true).into()],
    );
    model.add_turn(vec![shell_call(
        "call_shell_pending_dup",
        &["echo pending"],
    )]);
    let first = run(&agent, "run shell").await;
    assert!(!first.interruptions.is_empty());
    let resumed = run(&agent, first.to_state()).await;
    let pending = resumed
        .new_items
        .iter()
        .filter(|i| matches!(&i.kind, RunItemKind::ToolApproval { raw_item, .. } if raw_item.call_id == "call_shell_pending_dup"))
        .count();
    assert_eq!(pending, 1);
}

/// Python: `test_route_local_shell_calls_to_remote_shell_tool`: a `local_shell_call` goes to
/// the local shell tool even when a shell tool is also registered.
#[tokio::test]
async fn route_local_shell_calls_to_the_local_shell_tool() {
    let model = setup();
    let remote = log();
    let local = log::<Value>();
    let l = local.clone();
    let local_tool = LocalShellTool::<()>::new(move |req| {
        let l = l.clone();
        async move {
            l.lock().unwrap().push(req.data.clone());
            Ok("local_output".to_string())
        }
    });
    let agent = agent_with(
        &model,
        vec![counting_shell(&remote).into(), local_tool.into()],
    );
    model.add_turns(vec![
        vec![item(json!({
            "id": "local_1", "call_id": "call_local_1", "type": "local_shell_call",
            "action": {"type": "exec", "command": ["echo", "test"], "env": {}},
            "status": "in_progress"
        }))],
        vec![],
    ]);
    run(&agent, "run local shell").await;
    assert!(
        !local.lock().unwrap().is_empty(),
        "LocalShellTool should run"
    );
    assert!(remote.lock().unwrap().is_empty(), "ShellTool must not run");
}

/// Python: `test_preserve_max_turns_when_resuming_from_runresult_state`.
#[tokio::test]
async fn preserve_max_turns_when_resuming_from_runresult_state() {
    let model = setup();
    let executed = log();
    let agent = agent_with(
        &model,
        vec![gated_tool("test_tool", "tool_result", &executed).into()],
    );
    model.add_turn(vec![fn_call("test_tool", "call-1")]);
    let first = Runner::run_with_options(
        &agent,
        "call test_tool",
        RunOptions::default().max_turns(Some(20)),
    )
    .await
    .unwrap();
    let state = approve_first(&first, true);
    // Ten more tool turns: past the default limit of 10, within 20.
    for i in 0..10 {
        model.add_turn(vec![
            assistant_message(format!("turn {}", i + 2)),
            fn_call("test_tool", &format!("call-{}", i + 2)),
        ]);
    }
    model.add_turn(vec![]);
    Runner::run(&agent, state).await.unwrap();
}

/// Python: `test_current_turn_not_preserved_in_to_state` (the turn counter is preserved).
#[tokio::test]
async fn current_turn_is_preserved_in_to_state() {
    let model = setup();
    let executed = log();
    let agent = agent_with(
        &model,
        vec![gated_tool("test_tool", "tool_result", &executed).into()],
    );
    model.add_turn(vec![fn_call("test_tool", "call-1")]);
    let first = run(&agent, "call test_tool").await;
    assert!(!first.interruptions.is_empty());
    assert_eq!(first.to_state().current_turn, 1);
}

/// Python: `test_deserialize_interruptions_preserve_tool_calls` (shell, apply_patch).
#[tokio::test]
async fn deserialize_interruptions_preserve_tool_calls() {
    for kind in ["shell", "apply_patch"] {
        let model = setup();
        let runs = log();
        let (tool, raw): (Tool<()>, OutputItem) = if kind == "shell" {
            (
                counting_shell(&runs).with_needs_approval(true).into(),
                shell_call("call_shell_1", &["echo test"]),
            )
        } else {
            (
                ApplyPatchTool::<()>::from_arc(Arc::new(RecordingEditor::default()))
                    .with_needs_approval(true)
                    .into(),
                apply_patch_call("call_apply_1"),
            )
        };
        let agent = agent_with(&model, vec![tool]);
        model.add_turn(vec![raw]);
        let result = run(&agent, "go").await;
        let restored = roundtrip(&result.to_state());
        let interruptions = restored.get_interruptions();
        assert!(!interruptions.is_empty());
        assert_eq!(interruptions[0].tool_name, kind);
    }
}

/// Python: `test_deserialize_interruptions_preserve_mcp_tools` (include_provider_data).
#[test]
fn deserialize_interruptions_preserve_mcp_tools() {
    for include_provider_data in [true, false] {
        let mut fc = call("test_mcp_tool", "{}", "mcp-approval-1");
        fc.extra.insert("type".into(), json!("hosted_tool_call"));
        if include_provider_data {
            fc.extra.insert(
                "provider_data".into(),
                json!({"type": "mcp_approval_request", "id": "req-1", "server_label": "test_server"}),
            );
        }
        let state = RunState {
            schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
            current_agent_name: "TestAgent".into(),
            max_turns: Some(10),
            original_input: vec![openai_agents_sdk::InputItem(
                json!({"role": "user", "content": "test"}),
            )],
            interruptions: vec![ToolApprovalItem {
                agent: "TestAgent".into(),
                raw_item: fc,
                tool_name: "test_mcp_tool".into(),
                tool_origin: None,
            }],
            ..Default::default()
        };
        let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
        let interruptions = restored.get_interruptions();
        assert!(!interruptions.is_empty(), "{include_provider_data}");
        assert_eq!(interruptions[0].tool_name, "test_mcp_tool");
    }
}

/// Python: `test_shell_call_without_call_id_raises`.
#[tokio::test]
async fn shell_call_without_call_id_raises() {
    let model = setup();
    let runs = log();
    let agent = agent_with(&model, vec![counting_shell(&runs).into()]);
    model.add_turn(vec![item(json!({
        "type": "shell_call", "action": {"commands": ["echo", "hi"]}
    }))]);
    let err = Runner::run(&agent, "test").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
    assert!(runs.lock().unwrap().is_empty());
}

/// Python: `test_preserve_tool_output_types_during_serialization`.
#[test]
fn preserve_tool_output_types_during_serialization() {
    for (raw, expected) in [
        (
            json!({"type": "computer_call_output", "call_id": "call_computer_1",
                   "output": {"type": "computer_screenshot", "image_url": "base64_screenshot_data"}}),
            "computer_call_output",
        ),
        (
            json!({"type": "local_shell_call_output", "id": "shell_1",
                   "call_id": "call_shell_1", "output": "command output"}),
            "local_shell_call_output",
        ),
    ] {
        let state = RunState {
            schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
            current_agent_name: "TestAgent".into(),
            max_turns: Some(3),
            original_input: vec![openai_agents_sdk::InputItem(
                json!({"role": "user", "content": "test"}),
            )],
            generated_items: vec![openai_agents_sdk::items::RunItem::new(
                "TestAgent",
                RunItemKind::ToolCallOutput {
                    raw_item: raw,
                    output: json!("command output"),
                    custom_data: None,
                },
            )],
            ..Default::default()
        };
        let wire = state.to_json().unwrap();
        let items = wire["generated_items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["raw_item"]["type"], expected);
        let restored = RunState::from_json(wire).unwrap();
        assert_eq!(restored.generated_items.len(), 1);
        match &restored.generated_items[0].kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => assert_eq!(raw_item["type"], expected),
            other => panic!("{other:?}"),
        }
    }
}

fn raw_send_email(schema: Value, approvals: &Log<Value>, inputs: &Log<String>) -> FunctionTool<()> {
    let (a, i) = (approvals.clone(), inputs.clone());
    FunctionTool::<()>::new_raw(
        "send_email",
        "Send an email.",
        schema,
        move |_ctx, raw: String| {
            let i = i.clone();
            async move {
                i.lock().unwrap().push(raw);
                Ok("sent".to_string())
            }
        },
    )
    .with_needs_approval_fn(move |_ctx, params, _id| {
        let a = a.clone();
        async move {
            a.lock().unwrap().push(params);
            false
        }
    })
}

/// Python: `test_callable_function_approval_fails_closed_for_invalid_arguments` (7 payloads).
#[tokio::test]
async fn callable_function_approval_fails_closed_for_invalid_arguments() {
    for arguments in [
        "",
        r#"{"subject": "refund""#,
        "null",
        "[]",
        r#"{"amount": NaN}"#,
        r#"{"amount": Infinity}"#,
        r#"{"amount": -Infinity}"#,
    ] {
        let model = setup();
        let approvals = log();
        let inputs = log();
        let tool = raw_send_email(
            json!({"type": "object", "properties": {}}),
            &approvals,
            &inputs,
        );
        let agent = agent_with(&model, vec![tool.into()]);
        model.add_turn(vec![function_call_with_id(
            "send_email",
            arguments,
            "call-invalid",
        )]);
        let result = run(&agent, "send an email").await;
        assert_eq!(result.interruptions.len(), 1, "{arguments:?}");
        assert_eq!(result.interruptions[0].tool_name, "send_email");
        assert!(approvals.lock().unwrap().is_empty(), "{arguments:?}");
        assert!(inputs.lock().unwrap().is_empty(), "{arguments:?}");
    }
}

/// Python: `test_callable_function_approval_receives_valid_object_arguments`.
#[tokio::test]
async fn callable_function_approval_receives_valid_object_arguments() {
    let model = setup();
    let approvals = log();
    let inputs = log();
    let tool = raw_send_email(
        json!({"type": "object", "properties": {"subject": {"type": "string"}}}),
        &approvals,
        &inputs,
    );
    let agent = agent_with(&model, vec![tool.into()]);
    let arguments = r#"{"subject": "status update"}"#;
    model.add_turns(vec![
        vec![function_call_with_id("send_email", arguments, "call-valid")],
        vec![assistant_message("done")],
    ]);
    let result = run(&agent, "send an email").await;
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let seen = approvals.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(
        seen.iter()
            .all(|p| *p == json!({"subject": "status update"}))
    );
    assert_eq!(*inputs.lock().unwrap(), [arguments]);
}

/// Python: `test_agent_as_tool_with_nested_approvals_propagates`.
#[tokio::test]
async fn agent_as_tool_with_nested_approvals_propagates() {
    let nested_model = setup();
    let calls = log();
    let spanish = Agent::<()>::new("spanish_agent")
        .with_model_instance(nested_model.clone())
        .with_tool(gated_tool("get_current_timestamp", "timestamp", &calls));
    nested_model.add_turns(vec![
        vec![fn_call("get_current_timestamp", "call-1")],
        vec![assistant_message("hola")],
    ]);
    let orchestrator_model = Arc::new(FakeModel::new());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_model_instance(orchestrator_model.clone())
        .with_tool(
            spanish.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("respond_spanish")
                    .tool_description("Respond in Spanish")
                    .needs_approval(true),
            ),
        );
    orchestrator_model.add_turns(vec![
        vec![function_call_with_id(
            "respond_spanish",
            r#"{"input": "hola"}"#,
            "spanish-call",
        )],
        vec![assistant_message("done")],
    ]);
    let first = run(&orchestrator, "hola").await;
    assert!(
        !first.interruptions.is_empty(),
        "outer agent tool should ask first"
    );
    let resumed = run(&orchestrator, approve_first(&first, true)).await;
    assert_eq!(resumed.interruptions[0].tool_name, "get_current_timestamp");
    assert!(!resumed.to_input_list().is_empty());
    assert!(
        calls.lock().unwrap().is_empty(),
        "nested tool must not run before approval"
    );
    let done = run(&orchestrator, approve_first(&resumed, true)).await;
    assert_eq!(done.final_output_text().as_deref(), Some("done"));
    assert_eq!(*calls.lock().unwrap(), ["timestamp"]);
}

#[derive(Default)]
struct DelegateCounts {
    input_guardrail: usize,
    start: usize,
    output_guardrail: usize,
    custom_output: usize,
    extractor: usize,
    end: usize,
}

struct CountingHooks(Arc<Mutex<DelegateCounts>>);

#[async_trait]
impl RunHooks<()> for CountingHooks {
    async fn on_tool_start(&self, _ctx: &ToolContext<()>, _agent: &Agent<()>, tool: &Tool<()>) {
        if tool.name() == "delegate" {
            self.0.lock().unwrap().start += 1;
        }
    }
    async fn on_tool_end(
        &self,
        _ctx: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &openai_agents_sdk::ToolOutput,
    ) {
        if tool.name() == "delegate" {
            self.0.lock().unwrap().end += 1;
        }
    }
}

/// Python: `test_nested_agent_tool_continuation_runs_outer_callbacks_once` (streamed).
#[tokio::test]
async fn nested_agent_tool_continuation_runs_outer_callbacks_once() {
    for streamed in [false, true] {
        let nested_model = setup();
        let inner_calls = log();
        let nested = Agent::<()>::new("nested_agent")
            .with_model_instance(nested_model.clone())
            .with_tool(gated_tool("inner_tool", "inner output", &inner_calls));
        nested_model.add_turns(vec![
            vec![fn_call("inner_tool", &format!("inner-call-{streamed}"))],
            vec![assistant_message("nested done")],
        ]);
        let counts = Arc::new(Mutex::new(DelegateCounts::default()));
        let (c1, c2, c3, c4) = (
            counts.clone(),
            counts.clone(),
            counts.clone(),
            counts.clone(),
        );
        let mut outer_tool = nested.as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("delegate")
                .tool_description("Delegate to the nested agent")
                .custom_output_extractor(move |result: RunResult<()>| {
                    let c = c3.clone();
                    async move {
                        c.lock().unwrap().custom_output += 1;
                        Ok(result.final_output_text().unwrap_or_default())
                    }
                }),
        );
        outer_tool = outer_tool
            .with_input_guardrail(ToolInputGuardrail::new("track_input", move |_d| {
                let c = c1.clone();
                async move {
                    c.lock().unwrap().input_guardrail += 1;
                    Ok(ToolGuardrailFunctionOutput::allow())
                }
            }))
            .with_output_guardrail(ToolOutputGuardrail::new("track_output", move |_d| {
                let c = c2.clone();
                async move {
                    c.lock().unwrap().output_guardrail += 1;
                    Ok(ToolGuardrailFunctionOutput::allow())
                }
            }))
            .with_custom_data_extractor(move |_ctx| {
                let c = c4.clone();
                async move {
                    c.lock().unwrap().extractor += 1;
                    Some(json!({"nested": true}))
                }
            });
        let outer_model = Arc::new(FakeModel::new());
        outer_model.add_turns(vec![
            vec![function_call_with_id(
                "delegate",
                r#"{"input":"hello"}"#,
                format!("outer-call-{streamed}"),
            )],
            vec![assistant_message("outer done")],
        ]);
        let outer = Agent::<()>::new("outer_agent")
            .with_model_instance(outer_model.clone())
            .with_tool(outer_tool);
        let hooks: Arc<dyn RunHooks<()>> = Arc::new(CountingHooks(counts.clone()));
        let interrupted = run_with(
            &outer,
            "start",
            RunOptions::default().hooks(hooks.clone()),
            streamed,
        )
        .await
        .unwrap();
        assert!(!interrupted.interruptions.is_empty());
        {
            let c = counts.lock().unwrap();
            assert!(
                c.input_guardrail <= 1 && c.start <= 1,
                "streamed={streamed}"
            );
            assert_eq!(
                (c.output_guardrail, c.custom_output, c.extractor, c.end),
                (0, 0, 0, 0),
                "streamed={streamed}"
            );
        }
        let mut state = interrupted.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        let restored = roundtrip(&state);
        let done = run_with(
            &outer,
            restored,
            RunOptions::default().hooks(hooks),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(done.final_output_text().as_deref(), Some("outer done"));
        assert_eq!(*inner_calls.lock().unwrap(), ["inner output"]);
        let c = counts.lock().unwrap();
        assert_eq!(
            (
                c.input_guardrail,
                c.start,
                c.output_guardrail,
                c.custom_output,
                c.extractor,
                c.end
            ),
            (1, 1, 1, 1, 1, 1),
            "streamed={streamed}"
        );
    }
}
