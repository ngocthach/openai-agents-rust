//! Port of the Python SDK's `tests/test_shell_tool.py` (agent and tool layer only).
//!
//! Each `ShellAction.execute` case runs through `Runner::run`: the scripted model emits the
//! `shell_call`, and the assertions read the `shell_call_output` item the run recorded and the
//! input the model received on its next turn. Span assertions live in
//! `py_core3_shelltools_tool_spans.rs`, which owns the process-wide tracing state.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{
    ApprovalDecision, DEFAULT_APPROVAL_REJECTION_MESSAGE, ShellCallOutcome, ShellCommandOutput,
    ShellResult, ShellTool,
};
use openai_agents_sdk::{Agent, AgentsError, RunResult, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

/// The `shell_call` payload used across these tests (Python's `_shell_call`).
fn shell_call(call_id: &str, commands: &[&str]) -> Value {
    json!({
        "type": "shell_call",
        "id": "shell_call",
        "call_id": call_id,
        "status": "completed",
        "action": {"type": "exec", "commands": commands, "timeout_ms": 1000},
    })
}

fn shell_call_with_action(call_id: &str, action: Value) -> Value {
    json!({
        "type": "shell_call",
        "id": "shell_call",
        "call_id": call_id,
        "status": "completed",
        "action": action,
    })
}

fn outputs(r: &RunResult<()>) -> Vec<Value> {
    r.new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

fn container_reference() -> Value {
    json!({"type": "container_reference", "container_id": "cntr_123"})
}

/// Runs `call` through `shell` once and returns the single shell output and the run result.
async fn run_single_shell_call(shell: ShellTool<()>, call: Value) -> (Value, RunResult<()>) {
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    model.add_turn(vec![assistant_message("done")]);
    let r = Runner::run(&agent, "run").await.unwrap();
    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .expect("shell_call_output recorded");
    (raw, r)
}

#[test]
fn shell_tool_defaults_to_local_environment() {
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("ok") });

    assert_eq!(shell.environment, json!({"type": "local"}));
    assert!(shell.executor.is_some());
}

#[test]
fn shell_tool_supports_hosted_environment_without_executor() {
    let shell = ShellTool::<()>::hosted(container_reference()).unwrap();

    assert_eq!(shell.environment, container_reference());
    assert!(shell.executor.is_none());
}

#[test]
fn shell_tool_normalizes_container_auto_environment() {
    let environment = json!({
        "type": "container_auto",
        "file_ids": ["file_123"],
        "memory_limit": "4g",
        "network_policy": {
            "type": "allowlist",
            "allowed_domains": ["example.com"],
            "domain_secrets": [
                {"domain": "example.com", "name": "API_TOKEN", "value": "secret"}
            ],
        },
        "skills": [
            {"type": "skill_reference", "skill_id": "skill_123", "version": "latest"},
            {
                "type": "inline",
                "name": "csv-workbench",
                "description": "Analyze CSV files.",
                "source": {
                    "type": "base64",
                    "media_type": "application/zip",
                    "data": "ZmFrZS16aXA=",
                },
            },
        ],
    });

    let shell = ShellTool::<()>::hosted(environment.clone()).unwrap();

    assert_eq!(shell.environment, environment);
}

#[test]
fn shell_tool_rejects_local_mode_without_executor() {
    // Python's `ShellTool()` cannot be written in Rust: `ShellTool::new` takes the executor as
    // a required argument. The remaining case, a local environment with no executor, is the
    // `hosted` constructor's local-environment rejection.
    let err = ShellTool::<()>::hosted(json!({"type": "local"})).unwrap_err();
    assert!(err.to_string().contains("requires an executor"), "{err}");

    let mut shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("ok") });
    shell.executor = None;
    let err = shell.validate().unwrap_err();
    assert!(err.to_string().contains("requires an executor"), "{err}");
}

#[test]
fn shell_tool_allows_unvalidated_hosted_environment_shapes() {
    let shell = ShellTool::<()>::hosted(json!({"type": "container_reference"})).unwrap();
    assert_eq!(shell.environment, json!({"type": "container_reference"}));

    let shell = ShellTool::<()>::hosted(json!({
        "type": "container_auto",
        "network_policy": {
            "type": "future_mode",
            "allowed_domains": ["example.com"],
            "some_new_field": true,
        },
        "skills": [{"type": "skill_reference"}],
    }))
    .unwrap();
    assert_eq!(shell.environment["type"], "container_auto");
}

#[tokio::test]
async fn shell_tool_rejects_local_executor_and_approval_for_hosted_environment() {
    let with_executor = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("ok") })
        .with_local_environment(container_reference());
    let err = with_executor.validate().unwrap_err();
    assert!(
        err.to_string().contains("does not accept an executor"),
        "{err}"
    );

    let with_approval = ShellTool::<()>::hosted(container_reference())
        .unwrap()
        .with_needs_approval(true);
    let err = with_approval.validate().unwrap_err();
    assert!(
        err.to_string()
            .contains("does not support needs_approval or on_approval"),
        "{err}"
    );

    let with_callback = ShellTool::<()>::hosted(container_reference())
        .unwrap()
        .with_on_approval(|_ctx, _item| async { Some(ApprovalDecision::approve()) });
    let err = with_callback.validate().unwrap_err();
    assert!(
        err.to_string()
            .contains("does not support needs_approval or on_approval"),
        "{err}"
    );

    // The check also runs when the agent resolves its tools, so the run fails before any model
    // call.
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(
            ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("ok") })
                .with_local_environment(container_reference()),
        )
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "run").await.unwrap_err();
    assert!(
        err.to_string().contains("does not accept an executor"),
        "{err}"
    );
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn hosted_shell_call_is_not_executed_locally() {
    // Python's `execute_shell_calls` reports "Shell tool has no local executor configured." when
    // it is handed a hosted tool directly. The run loop never does that: it skips hosted shell
    // calls (`Skipping local shell execution for hosted shell tool`), and so does Rust. A local
    // tool without an executor cannot exist in Rust, so the message is unreachable.
    let shell = ShellTool::<()>::hosted(container_reference()).unwrap();
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo hi"]))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    assert!(outputs(&r).iter().all(|o| o["type"] != "shell_call_output"));
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn shell_tool_structured_output_is_rendered() {
    let shell = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![
                ShellCommandOutput {
                    command: Some("echo hi".into()),
                    stdout: "hi\n".into(),
                    outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                    ..Default::default()
                },
                ShellCommandOutput {
                    command: Some("ls".into()),
                    stdout: "README.md\nsrc\n".into(),
                    stderr: "warning".into(),
                    outcome: ShellCallOutcome::Exit { exit_code: Some(1) },
                    ..Default::default()
                },
            ],
            provider_data: Some(json!({"runner": "demo"})),
            max_output_length: Some(4096),
        })
    });
    let mut call = shell_call("call_shell", &["echo hi", "ls"]);
    call["action"]["max_output_length"] = json!(4096);

    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    model.add_turn(vec![assistant_message("done")]);
    let r = Runner::run(&agent, "run").await.unwrap();

    let tool_output = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .expect("tool output item");
    let text = tool_output.as_str().expect("text output");
    assert!(text.contains("$ echo hi"), "{text}");
    assert!(text.contains("stderr:\nwarning"), "{text}");

    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    assert_eq!(raw["status"], "completed");
    assert_eq!(raw["provider_data"]["runner"], "demo");
    assert_eq!(raw["max_output_length"], 4096);
    assert_eq!(raw["shell_output"][1]["exit_code"], 1);
    assert!(
        raw["output"][0]["stdout"]
            .as_str()
            .unwrap()
            .starts_with("hi")
    );
    assert_eq!(raw["output"][0]["outcome"]["type"], "exit");
    assert_eq!(raw["output"][0]["outcome"]["exit_code"], 0);
    assert!(raw["output"][0].get("command").is_none());

    // The replayed item carries only the wire fields.
    let second: Vec<Value> = model.calls()[1].input.iter().map(|i| i.0.clone()).collect();
    let replayed = second
        .iter()
        .find(|i| i["type"] == "shell_call_output")
        .expect("replayed shell output");
    assert!(replayed.get("status").is_none());
    assert!(replayed.get("shell_output").is_none());
    assert!(replayed.get("provider_data").is_none());
}

#[tokio::test]
async fn shell_tool_executor_failure_returns_error() {
    let shell = ShellTool::<()>::new(|_req| async {
        Err::<ShellResult, _>(AgentsError::user("boom".repeat(10)))
    });
    let call = shell_call_with_action(
        "call_shell_fail",
        json!({"commands": ["echo boom"], "timeout_ms": 1000, "max_output_length": 6}),
    );

    let (raw, r) = run_single_shell_call(shell, call).await;

    assert_eq!(tool_output_text(&r).as_deref(), Some("boombo"));
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["max_output_length"], 6);
    assert_eq!(raw["output"][0]["stdout"], "boombo");
    assert_eq!(raw["output"][0]["outcome"]["type"], "exit");
    assert_eq!(raw["output"][0]["outcome"]["exit_code"], 1);
    assert!(raw["output"][0].get("command").is_none());
}

/// The text of the first tool output item of a run.
fn tool_output_text(r: &RunResult<()>) -> Option<String> {
    r.new_items.iter().find_map(|i| match &i.kind {
        RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
        _ => None,
    })
}

/// Runs one structured executor result against an action with `max_output_length` and returns
/// the tool text and the shell output item.
async fn run_structured(executor_max: Option<usize>, action_max: Option<i64>) -> (String, Value) {
    let shell = ShellTool::<()>::new(move |_req| async move {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![ShellCommandOutput {
                stdout: "0123456789".into(),
                stderr: "abcdef".into(),
                outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                ..Default::default()
            }],
            max_output_length: executor_max,
            ..Default::default()
        })
    });
    let mut action = json!({"commands": ["echo hi"], "timeout_ms": 1000});
    if let Some(max) = action_max {
        action["max_output_length"] = json!(max);
    }
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call_with_action("call_shell", action))]);
    model.add_turn(vec![assistant_message("done")]);
    let r = Runner::run(&agent, "run").await.unwrap();
    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    (text, raw)
}

#[tokio::test]
async fn shell_tool_output_respects_max_output_length() {
    let (text, raw) = run_structured(None, Some(6)).await;

    assert_eq!(text, "012345");
    assert_eq!(raw["max_output_length"], 6);
    assert_eq!(raw["output"][0]["stdout"], "012345");
    assert_eq!(raw["output"][0]["stderr"], "");
}

#[tokio::test]
async fn shell_tool_uses_smaller_max_output_length() {
    let (text, raw) = run_structured(Some(8), Some(6)).await;

    assert_eq!(text, "012345");
    assert_eq!(raw["max_output_length"], 6);
    assert_eq!(raw["output"][0]["stdout"], "012345");
    assert_eq!(raw["output"][0]["stderr"], "");
}

#[tokio::test]
async fn shell_tool_executor_can_override_max_output_length_to_zero() {
    let (text, raw) = run_structured(Some(0), Some(6)).await;

    assert_eq!(text, "");
    assert_eq!(raw["max_output_length"], 0);
    assert_eq!(raw["output"][0]["stdout"], "");
    assert_eq!(raw["output"][0]["stderr"], "");
}

#[tokio::test]
async fn shell_tool_action_can_request_zero_max_output_length() {
    let (text, raw) = run_structured(None, Some(0)).await;

    assert_eq!(text, "");
    assert_eq!(raw["max_output_length"], 0);
    assert_eq!(raw["output"][0]["stdout"], "");
    assert_eq!(raw["output"][0]["stderr"], "");
}

#[tokio::test]
async fn shell_tool_action_negative_max_output_length_clamps_to_zero() {
    let (text, raw) = run_structured(None, Some(-5)).await;

    assert_eq!(text, "");
    assert_eq!(raw["max_output_length"], 0);
    assert_eq!(raw["output"][0]["stdout"], "");
    assert_eq!(raw["output"][0]["stderr"], "");
}

#[tokio::test]
async fn shell_tool_needs_approval_returns_approval_item() {
    let model = setup();
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("output") })
        .with_needs_approval(true);
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo hi"]))]);

    let r = Runner::run(&agent, "run").await.unwrap();

    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.interruptions[0].name(), "shell");
}

/// Builds an agent whose shell tool requires approval, runs once to an interruption, and returns
/// the paused state (with the pending call rejected or approved by the caller).
async fn paused_shell_run(
    shell: ShellTool<()>,
) -> (Arc<FakeModel>, Agent<()>, openai_agents_sdk::RunState) {
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo test"]))]);
    let first = Runner::run(&agent, "run").await.unwrap();
    let state = first.to_state();
    (model, agent, state)
}

#[tokio::test]
async fn shell_tool_needs_approval_rejected_returns_rejection() {
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("output") })
        .with_needs_approval(true);
    let (model, agent, mut state) = paused_shell_run(shell).await;
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, None);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, state).await.unwrap();

    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    assert_eq!(text, DEFAULT_APPROVAL_REJECTION_MESSAGE);
    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    assert_eq!(raw["type"], "shell_call_output");
    assert_eq!(raw["output"].as_array().unwrap().len(), 1);
    assert_eq!(
        raw["output"][0]["stderr"],
        DEFAULT_APPROVAL_REJECTION_MESSAGE
    );
}

#[tokio::test]
async fn shell_tool_rejection_uses_run_level_formatter() {
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("output") })
        .with_needs_approval(true);
    let (model, agent, mut state) = paused_shell_run(shell).await;
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, None);
    model.add_turn(vec![assistant_message("done")]);

    let config = RunConfig::<()>::default().with_tool_error_formatter(|args| {
        Some(format!("{} denied ({})", args.tool_name, args.call_id))
    });
    let r = Runner::run_with_options(&agent, state, RunOptions::<()>::new(()).run_config(config))
        .await
        .unwrap();

    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    assert_eq!(raw["output"][0]["stderr"], "shell denied (call_shell)");
    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    assert_eq!(text, "shell denied (call_shell)");
}

#[tokio::test]
async fn shell_tool_on_approval_callback_auto_approves() {
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("output") })
        .with_needs_approval(true)
        .with_on_approval(|_ctx, _item| async { Some(ApprovalDecision::approve()) });
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo test"]))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    assert_eq!(text, "output");
    assert!(r.interruptions.is_empty());
}

#[tokio::test]
async fn shell_tool_on_approval_callback_auto_rejects() {
    let calls = Arc::new(Mutex::new(0));
    let ran = calls.clone();
    let shell = ShellTool::<()>::new(move |_req| {
        let ran = ran.clone();
        async move {
            *ran.lock().unwrap() += 1;
            Ok::<_, AgentsError>("output")
        }
    })
    .with_needs_approval(true)
    .with_on_approval(|_ctx, _item| async {
        Some(ApprovalDecision::reject(Some("Not allowed".into())))
    });
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo test"]))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    assert_eq!(text, "Not allowed");
    let raw = outputs(&r)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    assert_eq!(raw["output"][0]["stderr"], "Not allowed");
    assert_eq!(*calls.lock().unwrap(), 0, "rejected call must not execute");
}

#[tokio::test]
async fn shell_tool_on_approval_empty_reason_uses_default_rejection() {
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("output") })
        .with_needs_approval(true)
        .with_on_approval(|_ctx, _item| async {
            Some(ApprovalDecision::reject(Some(String::new())))
        });
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call("call_shell", &["echo test"]))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    let text = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .unwrap();
    assert_eq!(text, DEFAULT_APPROVAL_REJECTION_MESSAGE);
}

/// An empty structured result reaches the next model call unchanged, whether the response was
/// streamed or not.
async fn empty_structured_output_reaches_next_model_unchanged(streamed: bool) {
    let call = json!({
        "type": "shell_call",
        "id": "sc_empty",
        "call_id": "call_empty",
        "status": "completed",
        "action": {"type": "shell_command", "commands": ["true"]},
    });
    let tool = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![],
            provider_data: Some(json!({"runner": "test"})),
            max_output_length: Some(4096),
        })
    });
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![item(call)]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool);

    let (final_text, items, calls) = if streamed {
        let mut result = Runner::run_streamed(&agent, "run");
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        let run = result.result().expect("streamed run completed");
        (
            run.final_output_text(),
            run.new_items.clone(),
            model.calls(),
        )
    } else {
        let r = Runner::run(&agent, "run").await.unwrap();
        (r.final_output_text(), r.new_items.clone(), model.calls())
    };

    let raw = items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => {
                assert_eq!(output.as_str(), Some("(no output)"));
                Some(raw_item.clone())
            }
            _ => None,
        })
        .expect("shell output item");
    assert_eq!(raw["output"], json!([]));
    assert!(raw.get("shell_output").is_none());
    assert_eq!(raw["provider_data"], json!({"runner": "test"}));
    assert_eq!(raw["max_output_length"], 4096);
    assert_eq!(raw["status"], "completed");

    let replayed: Vec<Value> = calls[1]
        .request
        .input
        .iter()
        .map(|i| i.0.clone())
        .filter(|i| i["type"] == "shell_call_output")
        .collect();
    assert_eq!(
        replayed,
        vec![json!({
            "type": "shell_call_output",
            "call_id": "call_empty",
            "output": [],
            "max_output_length": 4096,
        })]
    );
    assert_eq!(final_text.as_deref(), Some("done"));
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn shell_tool_empty_structured_output_reaches_next_model_unchanged() {
    empty_structured_output_reaches_next_model_unchanged(false).await;
}

#[tokio::test]
async fn shell_tool_empty_structured_output_reaches_next_model_unchanged_streamed() {
    empty_structured_output_reaches_next_model_unchanged(true).await;
}
