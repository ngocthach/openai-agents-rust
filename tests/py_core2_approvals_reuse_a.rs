//! Ports of `tests/test_tool_approval_call_id_reuse.py`, first half: canonical invocation
//! identities, completed-replay skipping, same-response siblings, nested agent tools and
//! custom tool resumes.
//!
//! Python reaches the private `_tool_invocation` helpers directly; the pure identity cases use
//! the `#[doc(hidden)]` `tool_invocation_identity_for_tests` seam, the rest run end to end
//! through `Runner`.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{ApplyPatchTool, CustomTool, FunctionTool, ShellTool};
use openai_agents_sdk::{
    Agent, AgentsError, ModelResponse, RunContext, RunInput, RunResult, RunState, Runner,
    tool_invocation_identity_for_tests as identity,
};
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

fn item(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn shell_call(call_id: &str, commands: &[&str]) -> Value {
    json!({
        "type": "shell_call",
        "id": call_id,
        "call_id": call_id,
        "status": "in_progress",
        "action": {"type": "exec", "commands": commands, "timeout_ms": 1000},
    })
}

fn apply_patch_dict(call_id: &str) -> Value {
    json!({
        "type": "apply_patch_call",
        "call_id": call_id,
        "operation": {"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"},
    })
}

fn custom_call(name: &str, call_id: &str, input: &str) -> OutputItem {
    item(json!({"type": "custom_tool_call", "name": name, "call_id": call_id, "input": input}))
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn record_value(name: &str, executed: &Log, needs_approval: bool) -> FunctionTool<()> {
    let executed = executed.clone();
    FunctionTool::new(name, "record", move |_c, a: ValueArgs| {
        let executed = executed.clone();
        async move {
            executed.lock().unwrap().push(a.value.clone());
            Ok(a.value)
        }
    })
    .with_needs_approval(needs_approval)
}

fn custom_tool(name: &str, executed: &Log, prefix: &str, needs_approval: bool) -> CustomTool<()> {
    let executed = executed.clone();
    let prefix = prefix.to_string();
    CustomTool::new(name, "Edit raw text.", move |_c, input: String| {
        let executed = executed.clone();
        let prefix = prefix.clone();
        async move {
            executed.lock().unwrap().push(format!("{prefix}{input}"));
            Ok(input)
        }
    })
    .with_format(json!({"type": "text"}))
    .with_needs_approval(needs_approval)
}

fn recording_shell(name: &str, executed: &Log, needs_approval: bool) -> ShellTool<()> {
    let executed = executed.clone();
    ShellTool::new(move |req| {
        let executed = executed.clone();
        async move {
            executed
                .lock()
                .unwrap()
                .extend(req.data.action.commands.clone());
            Ok("ok")
        }
    })
    .with_name(name)
    .with_needs_approval(needs_approval)
}

#[derive(Default, Clone)]
struct RecordingEditor {
    paths: Log,
}

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.paths.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("created"))
    }
    async fn update_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.paths.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("updated"))
    }
    async fn delete_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.paths.lock().unwrap().push(op.path.clone());
        Ok(ApplyPatchResult::completed("deleted"))
    }
}

/// Run streamed or not, collecting stream events and surfacing a stream failure as `Err`.
async fn run_mode(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> (
    openai_agents_sdk::Result<RunResult<()>>,
    Vec<StreamEvent<()>>,
) {
    if !streamed {
        return (
            Runner::run_with_options(agent, input, options).await,
            Vec::new(),
        );
    }
    let mut stream = Runner::run_streamed_with_options(agent, input, options);
    let mut events = Vec::new();
    let mut error = None;
    {
        let mut s = std::pin::pin!(stream.stream_events());
        while let Some(event) = s.next().await {
            match event {
                Ok(e) => events.push(e),
                Err(e) => error = Some(e),
            }
        }
    }
    match error {
        Some(e) => (Err(e), events),
        None => (
            stream
                .into_result()
                .ok_or_else(|| AgentsError::user("stream produced no result")),
            events,
        ),
    }
}

fn tool_called_count(events: &[StreamEvent<()>]) -> usize {
    events
        .iter()
        .filter(|e| {
            matches!(
                e,
                StreamEvent::RunItem {
                    name: RunItemStreamEventName::ToolCalled,
                    ..
                }
            )
        })
        .count()
}

fn opts() -> RunOptions<()> {
    RunOptions::new(())
}

fn assert_reused(err: &AgentsError) {
    let text = err.to_string();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }) && text.contains("unique call ID"),
        "{err:?}"
    );
}

fn assert_model_error(err: &AgentsError, needle: &str) {
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }) && err.to_string().contains(needle),
        "{err:?}"
    );
}

// ------------------------------------------------------------------ canonical identities

#[tokio::test]
async fn unique_tool_turns_do_not_refingerprint_completed_history() {
    // Python also counts identity computations; that is an internal efficiency check.
    for streamed in [false, true] {
        let turns = 24;
        let executed = log();
        let model = model();
        for i in 0..turns {
            model.add_turn(vec![
                assistant_message("thinking"),
                function_call_with_id(
                    "lookup",
                    json!({"value": i.to_string()}).to_string(),
                    format!("call_{i}"),
                ),
            ]);
        }
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_value("lookup", &executed, false))
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "hi", opts().max_turns(Some(turns + 1)), streamed).await;
        let result = result.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            entries(&executed),
            (0..turns).map(|i| i.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(result.new_items.len(), 3 * turns + 1);
        let outputs: Vec<(String, String)> = result
            .to_input_list()
            .iter()
            .filter_map(|i| {
                let v = serde_json::to_value(i).unwrap();
                (v["type"] == "function_call_output").then(|| {
                    (
                        v["call_id"].as_str().unwrap().to_string(),
                        v["output"].as_str().unwrap_or_default().to_string(),
                    )
                })
            })
            .collect();
        assert_eq!(
            outputs,
            (0..turns)
                .map(|i| (format!("call_{i}"), i.to_string()))
                .collect::<Vec<_>>()
        );
        assert_eq!(model.remaining_steps(), 0);
    }
}

#[test]
fn canonical_shell_identity_ignores_stripped_provider_metadata() {
    let provider = json!({
        "type": "shell_call", "call_id": "shell_0",
        "action": {"commands": ["echo safe"]}, "created_by": "server",
    });
    let mut persisted = provider.clone();
    persisted.as_object_mut().unwrap().remove("created_by");
    assert!(identity(&provider, None, None).is_some());
    assert_eq!(
        identity(&provider, None, None),
        identity(&persisted, None, None)
    );
}

#[test]
fn canonical_shell_identity_treats_optional_nulls_as_omitted() {
    let provider = json!({
        "type": "shell_call", "call_id": "shell_0",
        "action": {"commands": ["echo safe"], "max_output_length": null, "timeout_ms": null},
        "environment": null,
    });
    let normalized = json!({
        "type": "shell_call", "call_id": "shell_0", "action": {"commands": ["echo safe"]},
    });
    assert_eq!(
        identity(&provider, None, None),
        identity(&normalized, None, None)
    );
}

#[test]
fn canonical_function_identity_preserves_null_argument_values() {
    let with_null = json!({
        "type": "function_call", "call_id": "call_0", "name": "lookup",
        "arguments": "{\"value\": null}",
    });
    let without = json!({
        "type": "function_call", "call_id": "call_0", "name": "lookup", "arguments": "{}",
    });
    assert_ne!(
        identity(&with_null, None, None),
        identity(&without, None, None)
    );
}

#[test]
fn canonical_identity_requires_each_tool_payload() {
    for raw in [
        json!({"type": "custom_tool_call", "name": "raw_editor", "call_id": "call_0"}),
        json!({"type": "computer_call", "call_id": "call_0"}),
        json!({"type": "local_shell_call", "call_id": "call_0"}),
        json!({"type": "shell_call", "call_id": "call_0"}),
        json!({"type": "apply_patch_call", "call_id": "call_0"}),
    ] {
        assert_eq!(identity(&raw, None, None), None, "{raw}");
    }
}

#[test]
fn native_tool_scope_includes_the_resolved_tool_name() {
    // The scope half of `test_native_tool_approval_scope_rejects_resolved_tool_replacement`:
    // the same native call resolved to another tool is another invocation.
    for raw in [
        shell_call("call_0", &["echo safe"]),
        apply_patch_dict("call_0"),
    ] {
        let first = identity(&raw, Some("safe_tool"), None).unwrap();
        let other = identity(&raw, Some("other_tool"), None).unwrap();
        assert_eq!(first.1, other.1);
        assert_ne!(first.2, other.2);
    }
}

// --------------------------------------------------------------- apply_patch fallbacks

#[tokio::test]
async fn completed_apply_patch_fallback_run_state_round_trip() {
    for fallback in ["custom", "function"] {
        let editor = RecordingEditor::default();
        let operation =
            json!({"type": "update_file", "path": "safe.txt", "diff": "-old\n+new\n"}).to_string();
        let call = if fallback == "custom" {
            custom_call("apply_patch", "patch_0", &operation)
        } else {
            function_call_with_id("apply_patch", operation, "patch_0")
        };
        let model = model();
        model.add_turn(vec![call]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(ApplyPatchTool::new(editor.clone()))
            .with_model_instance(model.clone());
        let result = Runner::run(&agent, "update the file").await.unwrap();
        let restored = RunState::from_json(result.to_state().to_json().unwrap()).unwrap();
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("done"),
            "{fallback}"
        );
        assert_eq!(entries(&editor.paths), vec!["safe.txt"], "{fallback}");
        assert!(restored.tool_invocations["patch_0"].completed, "{fallback}");
    }
}

#[tokio::test]
async fn streamed_function_apply_patch_replay_emits_one_tool_called_event() {
    let editor = RecordingEditor::default();
    let operation =
        json!({"type": "update_file", "path": "safe.txt", "diff": "-old\n+new\n"}).to_string();
    let model = model();
    model.add_turn(vec![function_call_with_id(
        "apply_patch",
        operation.clone(),
        "patch_0",
    )]);
    model.add_turn(vec![function_call_with_id(
        "apply_patch",
        operation,
        "patch_0",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(ApplyPatchTool::new(editor.clone()))
        .with_model_instance(model.clone());
    let (result, events) = run_mode(&agent, "update the file", opts(), true).await;
    assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&editor.paths), vec!["safe.txt"]);
    assert_eq!(tool_called_count(&events), 1);
}

// ------------------------------------------------------------- custom-named shell tools

#[tokio::test]
async fn custom_named_shell_sticky_approval_applies_to_fresh_call_ids() {
    let executed = log();
    let model = model();
    model.add_turn(vec![item(shell_call("call_0", &["echo first"]))]);
    model.add_turn(vec![item(shell_call("call_1", &["echo second"]))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(recording_shell("safe_shell", &executed, true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "run commands").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let resumed = Runner::run(&agent, state).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["echo first", "echo second"]);
}

#[tokio::test]
async fn native_tool_replacement_rejects_completed_call_id_replay() {
    // `test_custom_named_shell_replacement_rejects_completed_call_id_replay` and the runner
    // half of `test_native_tool_approval_scope_rejects_resolved_tool_replacement` (shell).
    let first_runs = log();
    let replacement_runs = log();
    let call = shell_call("call_0", &["echo safe"]);
    let model = model();
    model.add_turn(vec![item(call.clone())]);
    let agent = Agent::<()>::new("agent")
        .with_tool(recording_shell("safe_shell", &first_runs, true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "run command").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    model.add_turn(vec![assistant_message("done")]);
    let completed = Runner::run(&agent, state).await.unwrap();

    let replaced = Agent::<()>::new("agent")
        .with_tool(recording_shell("other_shell", &replacement_runs, false))
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    let err = Runner::run(&replaced, completed.to_state())
        .await
        .unwrap_err();
    assert_reused(&err);
    assert_eq!(entries(&first_runs), vec!["echo safe"]);
    assert!(entries(&replacement_runs).is_empty());
}

#[tokio::test]
async fn apply_patch_replacement_rejects_completed_call_id_replay() {
    // The apply_patch half of `test_native_tool_approval_scope_rejects_resolved_tool_replacement`.
    let editor = RecordingEditor::default();
    let other = RecordingEditor::default();
    let call = apply_patch_dict("call_0");
    let model = model();
    model.add_turn(vec![item(call.clone())]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(ApplyPatchTool::new(editor.clone()).with_name("safe_patch"))
        .with_model_instance(model.clone());
    let completed = Runner::run(&agent, "patch").await.unwrap();
    assert_eq!(entries(&editor.paths), vec!["test.md"]);

    let replaced = Agent::<()>::new("agent")
        .with_tool(ApplyPatchTool::new(other.clone()).with_name("other_patch"))
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    let err = Runner::run(&replaced, completed.to_state())
        .await
        .unwrap_err();
    assert_reused(&err);
    assert!(entries(&other.paths).is_empty());
}

#[tokio::test]
async fn schema_1_13_custom_named_shell_skips_exact_completed_replay() {
    let executed = log();
    let call = shell_call("legacy-shell", &["echo safe"]);
    let model = model();
    model.add_turn(vec![item(call.clone())]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(recording_shell("safe_shell", &executed, false))
        .with_model_instance(model.clone());
    let completed = Runner::run(&agent, "run").await.unwrap();
    let mut serialized = completed.to_state().to_json().unwrap();
    serialized["$schemaVersion"] = json!("1.13");
    serialized["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    for key in ["generated_items", "session_items"] {
        if let Some(items) = serialized[key].as_array_mut() {
            for it in items {
                if let Some(o) = it.as_object_mut() {
                    o.remove("tool_name");
                }
            }
        }
    }
    let restored = RunState::from_json(serialized).unwrap();
    model.add_turn(vec![item(call)]);
    model.add_turn(vec![assistant_message("done again")]);
    let resumed = Runner::run(&agent, restored).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done again"));
    assert_eq!(entries(&executed), vec!["echo safe"]);
}

// ------------------------------------------------------------------- empty call ids

#[tokio::test]
async fn empty_custom_tool_call_id_fails_before_approval_or_execution() {
    let executed = log();
    let callbacks = log();
    let cb = callbacks.clone();
    let tool = custom_tool("raw_editor", &executed, "", true).with_on_approval(move |_c, _i| {
        let cb = cb.clone();
        async move {
            cb.lock().unwrap().push("approval".into());
            Some(openai_agents_sdk::tool::ApprovalDecision::approve())
        }
    });
    let model = model();
    model.add_turn(vec![custom_call("raw_editor", "", "changed")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "run it").await.unwrap_err();
    assert_model_error(&err, "non-empty string call ID");
    assert!(entries(&callbacks).is_empty());
    assert!(entries(&executed).is_empty());
}

fn formatter_config(calls: &Log) -> openai_agents_sdk::RunConfig<()> {
    let calls = calls.clone();
    let mut config =
        openai_agents_sdk::RunConfig::default().with_tool_error_formatter(move |args| {
            calls.lock().unwrap().push(args.tool_name.clone());
            Some("error".to_string())
        });
    config.tool_not_found_behavior =
        openai_agents_sdk::run_config::ToolNotFoundBehavior::ReturnErrorToModel;
    config
}

#[tokio::test]
async fn empty_unresolved_function_call_id_fails_before_error_formatter() {
    let formatter_calls = log();
    let model = model();
    model.add_turn(vec![item(json!({
        "id": "item_0", "type": "function_call", "name": "missing", "arguments": "{}", "call_id": "",
    }))]);
    let agent = Agent::<()>::new("agent").with_model_instance(model.clone());
    let err = Runner::run_with_options(
        &agent,
        "run it",
        opts().run_config(formatter_config(&formatter_calls)),
    )
    .await
    .unwrap_err();
    assert_model_error(&err, "non-empty string call ID");
    assert!(entries(&formatter_calls).is_empty());
}

#[tokio::test]
async fn empty_handoff_call_id_fails_before_handoff_callback() {
    let calls = log();
    let c = calls.clone();
    let target = Arc::new(Agent::<()>::new("target"));
    let route = openai_agents_sdk::handoffs::handoff(target)
        .with_tool_name("route")
        .with_on_handoff(move |_ctx| {
            let c = c.clone();
            async move {
                c.lock().unwrap().push("route".into());
                Ok(())
            }
        });
    let model = model();
    model.add_turn(vec![item(json!({
        "id": "item_0", "type": "function_call", "name": "route", "arguments": "{}", "call_id": "",
    }))]);
    let agent = Agent::<()>::new("agent")
        .with_handoff(route)
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "route it").await.unwrap_err();
    assert_model_error(&err, "non-empty string call ID");
    assert!(entries(&calls).is_empty());
}

// --------------------------------------------- shared-context reruns ("already executed")

struct FailingHandoffHooks(Log);

#[async_trait]
impl RunHooks<()> for FailingHandoffHooks {
    async fn on_handoff(&self, _ctx: &RunContext<()>, _from: &Agent<()>, to: &Agent<()>) {
        self.0.lock().unwrap().push(to.name.clone());
    }
}

#[tokio::test]
async fn failed_handoff_hook_does_not_commit_output_or_repeat_callback() {
    // Rust hooks cannot fail, so the failure is injected from the handoff callback instead.
    let hook_calls = log();
    let target = Arc::new(Agent::<()>::new("target"));
    let route = openai_agents_sdk::handoffs::handoff(target)
        .with_on_handoff(|_ctx| async move { Err(AgentsError::user("handoff hook failed")) });
    let call = function_call_with_id("transfer_to_target", "{}", "handoff_0");
    let model = model();
    model.add_turn(vec![call.clone()]);
    model.add_turn(vec![call]);
    let agent = Agent::<()>::new("source")
        .with_handoff(route)
        .with_model_instance(model.clone());
    let ctx = RunContext::new(());
    let hooks: Arc<dyn RunHooks<()>> = Arc::new(FailingHandoffHooks(hook_calls.clone()));
    let first = Runner::run_with_options(
        &agent,
        "handoff",
        RunOptions::with_run_context(ctx.clone()).hooks(hooks.clone()),
    )
    .await
    .unwrap_err();
    assert!(
        first.to_string().contains("handoff hook failed"),
        "{first:?}"
    );
    let second = Runner::run_with_options(
        &agent,
        "handoff",
        RunOptions::with_run_context(ctx).hooks(hooks),
    )
    .await
    .unwrap_err();
    assert_model_error(&second, "already executed");
    assert!(entries(&hook_calls).is_empty());
}

#[tokio::test]
async fn non_approval_failed_tool_body_does_not_reexecute() {
    let attempts = log();
    let a = attempts.clone();
    let tool = FunctionTool::new(
        "perform_side_effect",
        "side effect",
        move |_c, _a: NoArgs| {
            let a = a.clone();
            async move {
                a.lock().unwrap().push("ran".into());
                Err::<String, _>(AgentsError::user("failed after side effect"))
            }
        },
    )
    .without_failure_error_function();
    let call = function_call_with_id("perform_side_effect", "{}", "call_0");
    let model = model();
    model.add_turn(vec![call.clone()]);
    model.add_turn(vec![call]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let ctx = RunContext::new(());
    let first =
        Runner::run_with_options(&agent, "run it", RunOptions::with_run_context(ctx.clone()))
            .await
            .unwrap_err();
    assert!(
        first.to_string().contains("failed after side effect"),
        "{first:?}"
    );
    let second = Runner::run_with_options(&agent, "run it", RunOptions::with_run_context(ctx))
        .await
        .unwrap_err();
    assert_model_error(&second, "already executed");
    assert_eq!(entries(&attempts), vec!["ran"]);
}

// ------------------------------------------------------- approval-bound reuse scenarios

fn approval_scenario(
    model: &Arc<FakeModel>,
    second_call_id: &str,
    second_value: &str,
    executed: &Log,
) -> Agent<()> {
    model.add_turn(vec![function_call_with_id(
        "record_value",
        r#"{"value": "safe"}"#,
        "call_0",
    )]);
    model.add_turn(vec![function_call_with_id(
        "record_value",
        json!({"value": second_value}).to_string(),
        second_call_id,
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("custom-provider-agent")
        .with_tool(record_value("record_value", executed, true))
        .with_model_instance(model.clone())
}

#[tokio::test]
async fn changed_arguments_under_reused_call_id_fail_before_second_side_effect() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let agent = approval_scenario(&model, "call_0", "changed", &executed);
        let (first, _) = run_mode(&agent, "record a value", opts(), streamed).await;
        let first = first.unwrap();
        assert_eq!(first.interruptions.len(), 1);
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let (err, _) = run_mode(&agent, state, opts(), streamed).await;
        assert_reused(&err.unwrap_err());
        assert_eq!(entries(&executed), vec!["safe"], "streamed={streamed}");
    }
}

#[tokio::test]
async fn exact_replay_reuses_committed_output_without_executing_again() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let agent = approval_scenario(&model, "call_0", "safe", &executed);
        let (first, _) = run_mode(&agent, "record a value", opts(), streamed).await;
        let first = first.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let (resumed, _) = run_mode(&agent, state, opts(), streamed).await;
        let resumed = resumed.unwrap();
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        assert!(resumed.interruptions.is_empty());
        assert_eq!(entries(&executed), vec!["safe"], "streamed={streamed}");
    }
}

#[tokio::test]
async fn non_approval_identical_siblings_execute_once() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        let dup = function_call_with_id("record_value", r#"{"value":"safe"}"#, "call_0");
        model.add_turn(vec![dup.clone(), dup]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_value("record_value", &executed, false))
            .with_model_instance(model.clone());
        let (result, events) = run_mode(&agent, "record a value", opts(), streamed).await;
        assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
        assert_eq!(entries(&executed), vec!["safe"]);
        if streamed {
            assert_eq!(tool_called_count(&events), 1);
        }
    }
}

#[tokio::test]
async fn non_approval_completed_replay_does_not_execute_again() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id(
            "record_value",
            r#"{"value":"safe"}"#,
            "call_0",
        )]);
        model.add_turn(vec![function_call_with_id(
            "record_value",
            r#"{ "value" : "safe" }"#,
            "call_0",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_value("record_value", &executed, false))
            .with_model_instance(model.clone());
        let (result, _) = run_mode(&agent, "record a value", opts(), streamed).await;
        assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
        assert_eq!(entries(&executed), vec!["safe"]);
    }
}

#[derive(Default)]
struct LlmEndCounter(Mutex<usize>);

#[async_trait]
impl RunHooks<()> for LlmEndCounter {
    async fn on_llm_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _r: &ModelResponse) {
        *self.0.lock().unwrap() += 1;
    }
}

#[tokio::test]
async fn non_approval_changed_completed_call_id_fails_before_execution() {
    for streamed in [false, true] {
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id(
            "record_value",
            r#"{"value":"safe"}"#,
            "call_0",
        )]);
        model.add_turn(vec![function_call_with_id(
            "record_value",
            r#"{"value":"changed"}"#,
            "call_0",
        )]);
        let agent = Agent::<()>::new("agent")
            .with_tool(record_value("record_value", &executed, false))
            .with_model_instance(model.clone());
        let hooks = Arc::new(LlmEndCounter::default());
        let (result, events) = run_mode(
            &agent,
            "record a value",
            opts().hooks(hooks.clone()),
            streamed,
        )
        .await;
        assert_reused(&result.unwrap_err());
        assert_eq!(entries(&executed), vec!["safe"]);
        assert_eq!(*hooks.0.lock().unwrap(), 1, "streamed={streamed}");
        if streamed {
            assert_eq!(tool_called_count(&events), 1);
        }
    }
}

// --------------------------------------------------------------- nested agent tools

#[tokio::test]
async fn nested_agent_tool_approval_allows_outer_call_id_collision() {
    for round_trip in [false, true] {
        let executed = log();
        let inner_model = model();
        inner_model.add_turn(vec![function_call_with_id(
            "inner_sensitive_tool",
            r#"{"value":"safe"}"#,
            "shared",
        )]);
        inner_model.add_turn(vec![assistant_message("inner done")]);
        let inner = Agent::<()>::new("inner")
            .with_tool(record_value("inner_sensitive_tool", &executed, true))
            .with_model_instance(inner_model.clone());
        let outer_model = model();
        outer_model.add_turn(vec![function_call_with_id(
            "nested_agent",
            r#"{"input":"hello"}"#,
            "shared",
        )]);
        outer_model.add_turn(vec![assistant_message("outer done")]);
        let outer = Agent::<()>::new("outer")
            .with_tool(Arc::new(inner).as_tool("nested_agent", "Run the nested agent."))
            .with_model_instance(outer_model.clone());

        let interrupted = Runner::run(&outer, "start").await.unwrap();
        assert_eq!(interrupted.interruptions.len(), 1);
        let mut state = interrupted.to_state();
        if round_trip {
            state = RunState::from_json(state.to_json().unwrap()).unwrap();
        }
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        let resumed = Runner::run(&outer, state).await.unwrap();
        assert_eq!(
            resumed.final_output_text().as_deref(),
            Some("outer done"),
            "round_trip={round_trip}"
        );
        assert!(resumed.interruptions.is_empty());
        assert_eq!(entries(&executed), vec!["safe"]);
    }
}

#[tokio::test]
async fn parent_approval_does_not_authorize_independent_nested_run() {
    for round_trip in [false, true] {
        let executed = log();
        let inner_model = model();
        inner_model.add_turn(vec![function_call_with_id(
            "sensitive",
            r#"{"value":"same"}"#,
            "shared",
        )]);
        inner_model.add_turn(vec![assistant_message("inner done")]);
        let inner = Agent::<()>::new("inner")
            .with_tool(record_value("sensitive", &executed, true))
            .with_model_instance(inner_model.clone());
        let outer_model = model();
        outer_model.add_turn(vec![function_call_with_id(
            "sensitive",
            r#"{"value":"same"}"#,
            "shared",
        )]);
        outer_model.add_turn(vec![function_call_with_id(
            "nested_agent",
            r#"{"input":"hello"}"#,
            "outer-nested",
        )]);
        outer_model.add_turn(vec![assistant_message("outer done")]);
        let outer = Agent::<()>::new("outer")
            .with_tool(record_value("sensitive", &executed, true))
            .with_tool(Arc::new(inner).as_tool("nested_agent", "Run the nested agent."))
            .with_model_instance(outer_model.clone());

        let parent = Runner::run(&outer, "start").await.unwrap();
        let mut state = parent.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        let nested = Runner::run(&outer, state).await.unwrap();
        assert_eq!(nested.interruptions.len(), 1);
        assert_eq!(entries(&executed), vec!["same"]);

        let mut nested_state = nested.to_state();
        if round_trip {
            nested_state = RunState::from_json(nested_state.to_json().unwrap()).unwrap();
        }
        let still_pending = Runner::run(&outer, nested_state).await.unwrap();
        assert_eq!(
            still_pending.interruptions.len(),
            1,
            "round_trip={round_trip}"
        );
        assert_eq!(entries(&executed), vec!["same"]);

        let mut approved = still_pending.to_state();
        let pending = approved.get_interruptions()[0].clone();
        approved.approve(&pending, false);
        let completed = Runner::run(&outer, approved).await.unwrap();
        assert_eq!(completed.final_output_text().as_deref(), Some("outer done"));
        assert!(completed.interruptions.is_empty());
        assert_eq!(entries(&executed), vec!["same", "same"]);
    }
}

// ------------------------------------------------------------------- custom tools

#[tokio::test]
async fn non_approval_custom_tool_identical_siblings_execute_once() {
    let executed = log();
    let model = model();
    model.add_turn(vec![
        custom_call("raw_editor", "call_0", "safe"),
        custom_call("raw_editor", "call_0", "safe"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(custom_tool("raw_editor", &executed, "", false))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "edit text").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["safe"]);
}

#[tokio::test]
async fn approved_custom_tool_resume_rebinds_to_current_tool() {
    let executed = log();
    let model = model();
    model.add_turn(vec![custom_call("raw_editor", "call_0", "safe")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(custom_tool("raw_editor", &executed, "original:", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "edit text").await.unwrap();
    let mut state = first.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    let replaced = Agent::<()>::new("agent")
        .with_tool(custom_tool("raw_editor", &executed, "replacement:", true))
        .with_model_instance(model.clone());
    let resumed = Runner::run(&replaced, state).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&executed), vec!["replacement:safe"]);
}

#[tokio::test]
async fn approved_custom_tool_resume_fails_when_current_tool_is_missing() {
    let executed = log();
    let model = model();
    model.add_turn(vec![custom_call("raw_editor", "call_0", "safe")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(custom_tool("raw_editor", &executed, "", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "edit text").await.unwrap();
    let mut state = first.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    let bare = Agent::<()>::new("agent").with_model_instance(model.clone());
    let err = Runner::run(&bare, state).await.unwrap_err();
    assert_model_error(&err, "Tool raw_editor not found");
    assert!(entries(&executed).is_empty());
}

#[tokio::test]
async fn approved_apply_patch_custom_resume_uses_public_tool() {
    let editor = RecordingEditor::default();
    let model = model();
    model.add_turn(vec![custom_call(
        "apply_patch",
        "patch_0",
        &json!({"type": "update_file", "path": "safe.txt", "diff": "-old\n+new\n"}).to_string(),
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(ApplyPatchTool::new(editor.clone()).with_needs_approval(true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "update the file").await.unwrap();
    let mut state = first.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    let resumed = Runner::run(&agent, state).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(entries(&editor.paths), vec!["safe.txt"]);
}

#[tokio::test]
async fn pending_custom_tool_resume_survives_missing_current_tool() {
    let executed = log();
    let model = model();
    model.add_turn(vec![custom_call("raw_editor", "call_0", "safe")]);
    let agent = Agent::<()>::new("agent")
        .with_tool(custom_tool("raw_editor", &executed, "", true))
        .with_model_instance(model.clone());
    let first = Runner::run(&agent, "edit").await.unwrap();
    let bare = Agent::<()>::new("agent").with_model_instance(model.clone());
    let resumed = Runner::run(&bare, first.to_state()).await.unwrap();
    assert_eq!(resumed.interruptions.len(), 1);
    assert!(entries(&executed).is_empty());
}

#[tokio::test]
async fn rejected_custom_tool_resume_emits_output_when_current_tool_is_missing() {
    for serialize in [false, true] {
        let executed = log();
        let model = model();
        model.add_turn(vec![custom_call("raw_editor", "call_0", "safe")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("agent")
            .with_tool(custom_tool("raw_editor", &executed, "", true))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "edit text").await.unwrap();
        let mut state = first.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.reject(&pending, false, None);
        if serialize {
            state = RunState::from_json(state.to_json().unwrap()).unwrap();
        }
        let bare = Agent::<()>::new("agent").with_model_instance(model.clone());
        let resumed = Runner::run(&bare, state).await.unwrap();
        assert_eq!(
            resumed.final_output_text().as_deref(),
            Some("done"),
            "serialize={serialize}"
        );
        assert!(entries(&executed).is_empty());
        let input = model.last_call().unwrap().input;
        let outputs = input
            .iter()
            .filter(|i| {
                let v = serde_json::to_value(i).unwrap();
                v["type"] == "custom_tool_call_output" && v["call_id"] == "call_0"
            })
            .count();
        assert_eq!(outputs, 1, "serialize={serialize}");
    }
}
