//! Port of `tests/sandbox/test_helper_startup.py`: files the model writes into the workspace
//! (`.profile`, `.bashrc`, ...) must never be executed by the SDK's own helper commands.
#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::sandbox::capabilities::{Capability, Filesystem, Shell};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::SandboxClient;
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{RunOptions, RunState};
use serde_json::{Value, json};

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Python: `test_runner_snapshot_does_not_execute_workspace_startup_files` (3 parameter sets).
///
/// The model writes login-shell startup files that would clobber `outside.txt` if anything ran
/// them. A run that then snapshots the workspace must leave the file alone; only the model's own
/// approved command may change it.
#[tokio::test]
async fn runner_snapshot_does_not_execute_workspace_startup_files() {
    for approval in [None, Some(false), Some(true)] {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let temp = tempfile::tempdir().unwrap();
        let snapshots = temp.path().join("snapshots");
        std::fs::create_dir(&snapshots).unwrap();
        let outside = temp.path().join("outside.txt");
        std::fs::write(&outside, "unchanged").unwrap();
        let payload = format!(
            "printf startup > {}",
            shell_quote(outside.to_str().unwrap())
        );
        let mut operations: Vec<Value> = [".profile", ".bash_profile", ".bashrc", ".zshenv"]
            .iter()
            .map(|name| json!({"type": "create_file", "path": name, "diff": format!("+{payload}")}))
            .collect();
        operations
            .push(json!({"type": "create_file", "path": "notes.txt", "diff": "+saved content"}));

        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![OutputItem::from_value(json!({
            "type": "custom_tool_call", "call_id": "edit", "name": "apply_patch",
            "input": json!({"operations": operations}).to_string()
        }))]);
        if approval.is_some() {
            model.add_turn(vec![OutputItem::from_value(json!({
                "type": "function_call", "call_id": "exec", "name": "exec_command",
                "arguments": json!({
                    "cmd": format!("printf approved > {}", shell_quote(outside.to_str().unwrap())),
                    "shell": "/bin/sh", "login": false
                }).to_string()
            }))]);
        }
        model.add_turn(vec![assistant_message("saved")]);

        let capabilities: Vec<Arc<dyn Capability<()>>> = if approval.is_none() {
            vec![Arc::new(Filesystem::new().with_custom_apply_patch(true))]
        } else {
            vec![
                Arc::new(Filesystem::new().with_custom_apply_patch(true)),
                Arc::new(Shell::new().with_needs_approval(true)),
            ]
        };
        let agent = SandboxAgent::<()>::new("File task")
            .with_model_instance(model)
            .with_capabilities(capabilities)
            .with_default_manifest(
                Manifest::new().with_extra_path_grant(
                    // The approved command writes outside the workspace; the macOS confinement
                    // profile only allows that under an explicit grant, as in Python.
                    SandboxPathGrant::new(temp.path().canonicalize().unwrap().to_string_lossy())
                        .unwrap(),
                ),
            );
        let client = UnixLocalSandboxClient::new()
            .with_inherit_host_environment(false)
            .with_snapshot(SnapshotSpec::Local {
                base_path: snapshots.clone(),
            });
        let config = SandboxRunConfig::with_client(Arc::new(client));

        let outcome = SandboxRunner::run_to_outcome(
            &agent,
            "Write the files.",
            &config,
            RunOptions::default().max_turns(Some(10)),
        )
        .await
        .expect("the run completes or pauses");
        assert_eq!(
            std::fs::read_to_string(&outside).unwrap(),
            "unchanged",
            "{approval:?}: nothing executed a startup file"
        );

        let result = match approval {
            None => outcome.result,
            Some(approved) => {
                assert_eq!(outcome.result.interruptions.len(), 1, "{approval:?}");
                let sandbox_state = outcome
                    .sandbox_state
                    .clone()
                    .expect("a paused run returns sandbox state");
                let mut state: RunState = outcome.result.to_state();
                let pending = state.get_interruptions()[0].clone();
                if approved {
                    state.approve(&pending, false);
                } else {
                    state.reject(&pending, false, None);
                }
                SandboxRunner::run_with_options(
                    &agent,
                    state,
                    &config.clone().session_state(sandbox_state),
                    RunOptions::default().max_turns(Some(10)),
                )
                .await
                .expect("the resumed run completes")
            }
        };

        assert!(result.interruptions.is_empty(), "{approval:?}");
        assert_eq!(result.final_output_text().as_deref(), Some("saved"));
        assert_eq!(
            std::fs::read_to_string(&outside).unwrap(),
            if approval == Some(true) {
                "approved"
            } else {
                "unchanged"
            },
            "{approval:?}"
        );
        let outputs: Vec<&RunItemKind> = result.new_items.iter().map(|item| &item.kind).collect();
        assert!(!outputs.is_empty());

        // The snapshot holds the saved file, and the live workspace is gone.
        let archives: Vec<_> = std::fs::read_dir(&snapshots)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert!(!archives.is_empty(), "{approval:?}: a snapshot was stored");
        let bytes = std::fs::read(archives[0].path()).unwrap();
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(bytes.as_slice()));
        let mut notes = None;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            if entry.path().unwrap().ends_with("notes.txt") {
                let mut text = String::new();
                std::io::Read::read_to_string(&mut entry, &mut text).unwrap();
                notes = Some(text);
            }
        }
        assert_eq!(notes.as_deref(), Some("saved content"), "{approval:?}");
    }
}

/// Python: `test_fixed_file_checks_do_not_load_workspace_profile` (5 parameter sets).
///
/// The SDK's own read, write, mkdir, rm and same-file checks run through `sh -c`, never a login
/// shell, so a `.profile` in the workspace is not executed.
#[tokio::test]
async fn fixed_file_checks_do_not_load_the_workspace_profile() {
    for operation in ["read", "write", "mkdir", "rm", "same-file"] {
        let temp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(temp.path()).unwrap();
        let outside = root.join("outside.txt");
        std::fs::write(&outside, "unchanged").unwrap();
        let client = UnixLocalSandboxClient::new().with_inherit_host_environment(false);
        let session = client
            .create(Some(
                Manifest::new().with_root(root.join("workspace").to_string_lossy()),
            ))
            .await
            .unwrap();
        session.start().await.unwrap();
        session
            .write_file(
                ".profile",
                format!(
                    "printf startup > {}\n",
                    shell_quote(outside.to_str().unwrap())
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        session
            .write_file("notes.txt", b"saved content")
            .await
            .unwrap();

        match operation {
            "read" => {
                session.check_read_access("notes.txt", None).await.unwrap();
            }
            "write" => session.check_write_access("notes.txt", None).await.unwrap(),
            "mkdir" => {
                session
                    .check_mkdir_access("nested", true, None)
                    .await
                    .unwrap();
                assert!(!root.join("workspace/nested").exists());
            }
            "rm" => session
                .check_rm_access("notes.txt", false, None)
                .await
                .unwrap(),
            _ => {
                assert!(
                    session
                        .same_file("notes.txt", "notes.txt", true, None)
                        .await
                        .unwrap()
                );
            }
        }

        assert_eq!(
            session.read_file("notes.txt").await.unwrap(),
            b"saved content"
        );
        assert_eq!(
            std::fs::read_to_string(&outside).unwrap(),
            "unchanged",
            "{operation}"
        );
        client.delete(session).await.unwrap();
    }
}
