//! Port of `tests/sandbox/test_run_cwd.py`: the run working directory scopes relative paths for
//! the shell, `view_image` and `apply_patch` tools, including when a paused run is resumed.
//! (`test_python_skill_uses_absolute_root_from_nested_workdir` is in `py_sandbox_b_skills.rs`.)
#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use base64::Engine as _;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::sandbox::capabilities::{Capability, Filesystem, Shell};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{RunOptions, RunState};
use serde_json::{Value, json};

const PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a84QAAAAASUVORK5CYII=";
const SVG: &[u8] = br#"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>"#;

fn item(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn function_call(call_id: &str, name: &str, arguments: Value) -> OutputItem {
    item(json!({
        "type": "function_call", "call_id": call_id, "name": name,
        "arguments": arguments.to_string()
    }))
}

fn patch_call(call_id: &str, patch: &str) -> OutputItem {
    item(json!({
        "type": "custom_tool_call", "call_id": call_id, "name": "apply_patch", "input": patch
    }))
}

async fn local_session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(Manifest::new())).await.unwrap();
    session.start().await.unwrap();
    (client, session)
}

fn outputs(result: &openai_agents_sdk::RunResult<()>) -> Vec<(String, Value)> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => Some((
                raw_item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                serde_json::to_value(output).unwrap_or(Value::Null),
            )),
            _ => None,
        })
        .collect()
}

async fn read(session: &Arc<dyn SandboxSession>, path: &str) -> Option<Vec<u8>> {
    session.read_file(path).await.ok()
}

/// Python: `test_concurrent_runs_scope_relative_paths_with_shared_live_session`.
#[tokio::test]
async fn concurrent_runs_scope_relative_paths_with_a_shared_live_session() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let (client, session) = local_session().await;
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_BASE64)
        .unwrap();
    let images: [(&str, &str, Vec<u8>); 2] = [
        ("task-a", "image/png", png),
        ("task-b", "image/svg+xml", SVG.to_vec()),
    ];
    let mut runs = Vec::new();
    for (task, _, bytes) in &images {
        session.mkdir(&format!("tasks/{task}"), true).await.unwrap();
        session
            .write_file(&format!("tasks/{task}/seed.png"), bytes)
            .await
            .unwrap();
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call(
            &format!("{task}_shell"),
            "exec_command",
            json!({"cmd": "cp seed.png plot.png", "login": false}),
        )]);
        model.add_turn(vec![function_call(
            &format!("{task}_image"),
            "view_image",
            json!({"path": "plot.png"}),
        )]);
        model.add_turn(vec![patch_call(
            &format!("{task}_patch"),
            &format!("*** Begin Patch\n*** Add File: notes.md\n+{task}\n*** End Patch\n"),
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let capabilities: Vec<Arc<dyn Capability<()>>> = vec![
            Arc::new(Shell::new()),
            Arc::new(Filesystem::new().with_custom_apply_patch(true)),
        ];
        let agent = SandboxAgent::<()>::new(*task)
            .with_model_instance(model)
            .with_capabilities(capabilities);
        let config = SandboxRunConfig::with_session(session.clone()).cwd(format!("tasks/{task}"));
        runs.push(async move {
            SandboxRunner::run(
                &agent,
                "Create the requested task-local artifacts.",
                &config,
            )
            .await
        });
    }

    let results = futures::future::join_all(runs).await;

    for (result, (task, mime, bytes)) in results.into_iter().zip(&images) {
        let result = result.unwrap_or_else(|e| panic!("{task}: {e}"));
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let by_call: std::collections::HashMap<_, _> = outputs(&result).into_iter().collect();
        let image = by_call[&format!("{task}_image")].to_string();
        assert!(
            image.contains(&format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )),
            "{task}: {image}"
        );
        assert!(
            by_call[&format!("{task}_patch")]
                .to_string()
                .contains("Created notes.md"),
            "{task}"
        );
        assert_eq!(
            read(&session, &format!("tasks/{task}/plot.png"))
                .await
                .as_deref(),
            Some(bytes.as_slice())
        );
        assert_eq!(
            read(&session, &format!("tasks/{task}/notes.md"))
                .await
                .as_deref(),
            Some(task.as_bytes())
        );
    }
    for root_relative in ["plot.png", "notes.md"] {
        assert!(
            read(&session, root_relative).await.is_none(),
            "{root_relative}"
        );
    }
    client.delete(session).await.ok();
}

/// Python: `test_resumed_run_rebinds_cwd_to_pending_sandbox_tool`.
#[tokio::test]
async fn a_resumed_run_rebinds_the_cwd_to_the_pending_tool() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let (client, session) = local_session().await;
    session.mkdir("tasks/resumed-task", true).await.unwrap();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call(
        "resumed_shell",
        "exec_command",
        json!({"cmd": "printf resumed > marker.txt", "login": false}),
    )]);
    let agent = SandboxAgent::<()>::new("resumed-task")
        .with_model_instance(model.clone())
        .with_capabilities(vec![Arc::new(Shell::new().with_needs_approval(true))]);
    let config = SandboxRunConfig::with_session(session.clone()).cwd("tasks/resumed-task");

    let first = SandboxRunner::run_to_outcome(
        &agent,
        "Create the marker.",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .unwrap();
    assert_eq!(first.result.interruptions.len(), 1);
    let mut state = first.result.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    model.add_turn(vec![assistant_message("done")]);

    let resumed = SandboxRunner::run_with_options(
        &agent,
        state,
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        read(&session, "tasks/resumed-task/marker.txt")
            .await
            .as_deref(),
        Some(b"resumed".as_slice())
    );
    assert!(read(&session, "marker.txt").await.is_none());
    client.delete(session).await.ok();
}

/// Python: `test_resumed_apply_patch_uses_current_run_cwd` (in-memory and JSON state).
#[tokio::test]
async fn a_resumed_apply_patch_uses_the_current_run_cwd() {
    for serialize_state in [false, true] {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let (client, session) = local_session().await;
        session.mkdir("tasks/a", true).await.unwrap();
        session.mkdir("tasks/b", true).await.unwrap();
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![patch_call(
            "patch_call",
            "*** Begin Patch\n*** Add File: marker.txt\n+resumed\n*** End Patch\n",
        )]);
        let agent = SandboxAgent::<()>::new("resumed-patch-task")
            .with_model_instance(model.clone())
            .with_capabilities(vec![Arc::new(
                Filesystem::new()
                    .with_custom_apply_patch(true)
                    .with_apply_patch_needs_approval(true),
            )]);
        let in_a = SandboxRunConfig::with_session(session.clone()).cwd("tasks/a");
        let in_b = SandboxRunConfig::with_session(session.clone()).cwd("tasks/b");

        let first = SandboxRunner::run_to_outcome(
            &agent,
            "Create the marker.",
            &in_a,
            RunOptions::default().max_turns(Some(10)),
        )
        .await
        .unwrap();
        assert_eq!(first.result.interruptions.len(), 1);
        let mut state: RunState = first.result.to_state();
        if serialize_state {
            state = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
        }
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        model.add_turn(vec![assistant_message("done")]);

        let resumed = SandboxRunner::run_with_options(
            &agent,
            state,
            &in_b,
            RunOptions::default().max_turns(Some(10)),
        )
        .await
        .unwrap();

        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            read(&session, "tasks/b/marker.txt").await.as_deref(),
            Some(b"resumed".as_slice()),
            "serialize={serialize_state}"
        );
        assert!(read(&session, "tasks/a/marker.txt").await.is_none());
        client.delete(session).await.ok();
    }
}
