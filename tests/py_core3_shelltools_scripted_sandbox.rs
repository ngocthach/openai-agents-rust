//! The two agent/tool-layer cases of Python's `tests/test_scripted_sandbox.py`.
//!
//! The rest of that file tests the scripted sandbox session itself (introspection, stream
//! snapshots, step validation), which belongs to the sandbox area. The apply_patch create path
//! is the only part a tool drives, so it is ported here through `SandboxEditor`.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::SandboxEditor;
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};

#[tokio::test]
async fn scripted_sandbox_supports_apply_patch_create() {
    // Creation keeps the existing file steps without requiring a scripted exec call.
    let session = Arc::new(scripted_sandbox_session(vec![
        SandboxStep::new(SandboxMethod::Mkdir),
        SandboxStep::new(SandboxMethod::WriteFile),
    ]));
    let editor = SandboxEditor::new(session.clone());

    let result = editor
        .create_file(&ApplyPatchOperation {
            op_type: ApplyPatchOperationType::CreateFile,
            path: "notes.txt".into(),
            diff: Some("+hello\n".into()),
            move_to: None,
        })
        .await
        .unwrap();

    assert_eq!(result.output.as_deref(), Some("Created notes.txt"));
    assert_eq!(session.remaining_steps(), 0);
    // The recorded paths matter as much as the methods: the create path hands over an
    // unresolved leaf so a symlinked leaf is not followed.
    let calls: Vec<(SandboxMethod, String)> = session
        .calls()
        .into_iter()
        .map(|call| (call.method, call.arguments[0].clone()))
        .collect();
    assert_eq!(
        calls,
        vec![
            (SandboxMethod::Mkdir, "/workspace".to_string()),
            (SandboxMethod::WriteFile, "/workspace/notes.txt".to_string()),
        ]
    );
}

#[tokio::test]
async fn scripted_sandbox_drives_black_box_sandbox_agent_workflow() {
    use openai_agents_sdk::sandbox::capabilities::Shell;
    use openai_agents_sdk::sandbox::pty::PtyExecUpdate;
    use openai_agents_sdk::sandbox::testing::SandboxCallResult;
    use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
    use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let session = Arc::new(scripted_sandbox_session(vec![
        // Rust's `exec_command` runs through the PTY start method, so that is what is scripted.
        SandboxStep::new(SandboxMethod::PtyExecStart)
            .matching(|call| call.has_argument("pwd"))
            .returning(SandboxCallResult::Pty(PtyExecUpdate {
                process_id: None,
                output: b"/workspace\n".to_vec(),
                exit_code: Some(0),
                original_token_count: None,
            })),
    ]));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("exec_command", r#"{"cmd":"pwd"}"#)]);
    model.add_turn(vec![assistant_message("The workspace is /workspace.")]);
    let agent = SandboxAgent::<()>::new("Test agent")
        .with_model_instance(model.clone())
        .with_capability(Arc::new(Shell::new()));

    let result = SandboxRunner::run(
        &agent,
        "Where am I?",
        &SandboxRunConfig::with_session(session.clone()),
    )
    .await
    .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("The workspace is /workspace.")
    );
    assert_eq!(session.calls().len(), 1);
    assert_eq!(model.calls().len(), 2);
    session.assert_complete();
    assert_eq!(model.remaining_steps(), 0);
}
