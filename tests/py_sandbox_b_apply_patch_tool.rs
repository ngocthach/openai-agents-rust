//! Port of `tests/sandbox/capabilities/test_apply_patch_tool.py` and
//! `test_apply_patch_preflight.py`: the freeform `apply_patch` custom tool.
#![cfg(feature = "sandbox")]

mod py_sandbox_b_support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::sandbox::capabilities::SandboxApplyPatchTool;
use openai_agents_sdk::sandbox::capabilities::apply_patch_tool::{
    APPLY_PATCH_CUSTOM_TOOL_DESCRIPTION, APPLY_PATCH_CUSTOM_TOOL_GRAMMAR, parse_custom_input,
};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::types::User;
use openai_agents_sdk::sandbox::workspace_paths::SandboxWorkspaceScope;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::Tool;
use openai_agents_sdk::{Agent, Runner};
use py_sandbox_b_support::{ApplyPatchSession, Store};
use serde_json::{Value, json};

fn session() -> Arc<ApplyPatchSession> {
    Arc::new(ApplyPatchSession::new(Store::Sensitive))
}

fn scoped(session: &Arc<ApplyPatchSession>, cwd: &str) -> SandboxEditor {
    SandboxEditor::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some(cwd)).expect("scope"))
}

fn op(kind: ApplyPatchOperationType, path: &str, diff: Option<&str>) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: kind,
        path: path.to_string(),
        diff: diff.map(str::to_string),
        move_to: None,
    }
}

fn model_item(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

/// What one custom-tool call produced.
struct Outcome {
    output: Option<String>,
    interrupted: bool,
}

/// Drive one `custom_tool_call` through the runner, the way `CustomToolAction.execute` does.
async fn execute(tool: SandboxApplyPatchTool<()>, raw_input: &str) -> Outcome {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![model_item(json!({
        "type": "custom_tool_call", "call_id": "call_apply", "name": "apply_patch",
        "input": raw_input
    }))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("patcher")
        .with_model_instance(model)
        .with_tool(Tool::Custom(Arc::new(tool.into_custom_tool())));
    let result = Runner::run(&agent, "go").await.expect("run");
    let output = result.new_items.iter().find_map(|item| match &item.kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => raw_item
            .get("output")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    });
    Outcome {
        output,
        interrupted: !result.interruptions.is_empty(),
    }
}

// ----------------------------------------------------------------------------------------------
// Tool shape
// ----------------------------------------------------------------------------------------------

/// Python: `test_exposes_custom_apply_patch_tool`.
#[test]
fn exposes_custom_apply_patch_tool() {
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session())).into_custom_tool();

    assert_eq!(tool.name, "apply_patch");
    let param = tool.to_responses_param();
    assert_eq!(param["type"], "custom");
    assert_eq!(param["name"], "apply_patch");
    assert_eq!(param["format"]["type"], "grammar");
    assert_eq!(param["format"]["syntax"], "lark");
}

/// Python: `test_grammar_requires_update_diff_after_optional_move`.
#[test]
fn grammar_requires_update_diff_after_optional_move() {
    let update_rule = APPLY_PATCH_CUSTOM_TOOL_GRAMMAR
        .lines()
        .find(|line| line.starts_with("update_hunk:"))
        .expect("an update_hunk rule");
    assert_eq!(
        update_rule,
        "update_hunk: \"*** Update File: \" filename LF change_move? change"
    );
    assert!(
        APPLY_PATCH_CUSTOM_TOOL_DESCRIPTION
            .contains("UpdateFile := \"*** Update File: \" path NEWLINE [ MoveTo ] Hunk { Hunk }")
    );
}

/// Python: `test_converter_uses_sandbox_custom_apply_patch_tool_config`.
#[test]
fn converter_uses_sandbox_custom_apply_patch_tool_config() {
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session())).into_custom_tool();
    let param = tool.to_responses_param();

    assert_eq!(param["type"], "custom");
    assert_eq!(param["name"], "apply_patch");
    let description = param["description"].as_str().unwrap();
    assert!(description.contains("This is a FREEFORM tool"));
    assert!(description.contains("A full patch can combine several operations"));
    assert_eq!(param["format"]["syntax"], "lark");
    assert_eq!(
        param["format"]["definition"],
        APPLY_PATCH_CUSTOM_TOOL_GRAMMAR
    );
}

/// Python: `test_runtime_rejects_updates_without_diff` (2 parameter sets).
#[tokio::test]
async fn runtime_rejects_updates_without_diff() {
    for update_body in ["", "*** Move to: moved.txt\n"] {
        let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session()));
        let outcome = execute(
            tool,
            &format!("*** Begin Patch\n*** Update File: notes.txt\n{update_body}*** End Patch\n"),
        )
        .await;

        let output = outcome.output.expect("the tool produced an output");
        assert!(
            output.contains("Update File patch for notes.txt must include a hunk"),
            "{output}"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Approvals
// ----------------------------------------------------------------------------------------------

/// Python: `test_needs_approval_exposes_operation_typed_setting` and
/// `test_public_needs_approval_assignment_drives_runtime_approval`, which together say that an
/// operation-typed rule drives the runtime approval of a call.
#[tokio::test]
async fn operation_typed_rule_drives_runtime_approval() {
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session()))
        .with_needs_approval_fn(|_ctx, operation, _call_id| async move {
            operation.op_type == ApplyPatchOperationType::DeleteFile
        });

    let outcome = execute(
        tool,
        "*** Begin Patch\n*** Delete File: notes.txt\n*** End Patch\n",
    )
    .await;

    assert!(outcome.interrupted, "a delete is held for approval");
}

/// Python: `test_needs_approval_receives_canonical_paths` (2 parameter sets).
#[tokio::test]
async fn needs_approval_receives_canonical_paths() {
    let payloads = [
        (
            json!({"type": "create_file", "path": r"sensitive\secret.txt", "diff": "+secret\n"}),
            "sensitive/secret.txt",
            None,
        ),
        (
            json!({"type": "update_file", "path": "notes.txt",
                   "move_to": r"sensitive\secret.txt", "diff": "@@\n-old\n+new\n"}),
            "notes.txt",
            Some("sensitive/secret.txt"),
        ),
    ];
    for (payload, expected_path, expected_move_to) in payloads {
        let checked = Arc::new(Mutex::new(Vec::new()));
        let seen = checked.clone();
        let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session()))
            .with_needs_approval_fn(move |_ctx, operation, _call_id| {
                let seen = seen.clone();
                async move {
                    seen.lock()
                        .unwrap()
                        .push((operation.path.clone(), operation.move_to.clone()));
                    operation.path == "sensitive/secret.txt"
                        || operation.move_to.as_deref() == Some("sensitive/secret.txt")
                }
            });

        let outcome = execute(tool, &payload.to_string()).await;

        assert!(outcome.interrupted);
        assert_eq!(
            *checked.lock().unwrap(),
            vec![(
                expected_path.to_string(),
                expected_move_to.map(str::to_string)
            )]
        );
    }
}

/// Python: `test_invalid_patch_input_surfaces_tool_error_after_approval_precheck`.
#[tokio::test]
async fn invalid_patch_input_surfaces_tool_error_after_approval_precheck() {
    // A malformed patch must not abort the run while the approval rule runs: the rule treats it
    // as "no approval needed" and the invocation reports the parse error to the model.
    let rule_tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session()))
        .with_needs_approval_fn(|_ctx, _operation, _call_id| async { true });

    let outcome = execute(rule_tool, "not a valid patch").await;

    assert!(!outcome.interrupted);
    assert!(
        outcome
            .output
            .expect("the tool produced an output")
            .contains("apply_patch input must start with '*** Begin Patch'")
    );
}

/// Python: `test_scoped_tool_approval_receives_execution_canonical_paths`.
#[tokio::test]
async fn scoped_tool_approval_receives_execution_canonical_paths() {
    let s = session();
    s.put("/workspace/tasks/a/notes.txt", b"old\n");
    let checked = Arc::new(Mutex::new(Vec::new()));
    let seen = checked.clone();
    let tool = SandboxApplyPatchTool::<()>::new(scoped(&s, "tasks/a")).with_needs_approval_fn(
        move |_ctx, operation, _call_id| {
            let seen = seen.clone();
            async move {
                seen.lock()
                    .unwrap()
                    .push((operation.path.clone(), operation.move_to.clone()));
                false
            }
        },
    );

    let outcome = execute(
        tool,
        "*** Begin Patch\n*** Update File: notes.txt\n*** Move to: moved.txt\n@@\n-old\n+new\n*** End Patch\n",
    )
    .await;

    assert_eq!(
        *checked.lock().unwrap(),
        vec![(
            "tasks/a/notes.txt".to_string(),
            Some("tasks/a/moved.txt".to_string())
        )]
    );
    assert_eq!(
        outcome.output.as_deref(),
        Some("Updated notes.txt\nMoved notes.txt to moved.txt")
    );
    assert_eq!(s.files()["/workspace/tasks/a/moved.txt"], b"new\n");
}

// ----------------------------------------------------------------------------------------------
// Preflight
// ----------------------------------------------------------------------------------------------

/// Python: `test_invalid_later_path_does_not_mutate_valid_prefix`.
#[tokio::test]
async fn invalid_later_path_does_not_mutate_valid_prefix() {
    let s = session();
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()));

    let outcome = execute(
        tool,
        "*** Begin Patch\n*** Add File: safe.txt\n+safe\n*** Add File: ../escape.txt\n+escape\n*** End Patch\n",
    )
    .await;

    assert!(outcome.output.is_some());
    assert_eq!(s.files(), BTreeMap::new());
}

// ----------------------------------------------------------------------------------------------
// The editor behind the tool
// ----------------------------------------------------------------------------------------------

/// Python: `test_editor_create_update_delete_round_trip`.
#[tokio::test]
async fn editor_create_update_delete_round_trip() {
    let s = session();
    let editor = SandboxEditor::new(s.clone());

    let created = editor
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "notes.txt",
            Some("+hello\n+world\n"),
        ))
        .await
        .unwrap();
    assert_eq!(created.output.as_deref(), Some("Created notes.txt"));
    assert_eq!(s.files()["/workspace/notes.txt"], b"hello\nworld");

    let updated = editor
        .update_file(&op(
            ApplyPatchOperationType::UpdateFile,
            "notes.txt",
            Some("@@\n-hello\n+hi\n world\n"),
        ))
        .await
        .unwrap();
    assert_eq!(updated.output.as_deref(), Some("Updated notes.txt"));
    assert_eq!(s.files()["/workspace/notes.txt"], b"hi\nworld");

    let deleted = editor
        .delete_file(&op(ApplyPatchOperationType::DeleteFile, "notes.txt", None))
        .await
        .unwrap();
    assert_eq!(deleted.output.as_deref(), Some("Deleted notes.txt"));
    assert!(!s.files().contains_key("/workspace/notes.txt"));
}

/// Python: `test_editor_scopes_paths_and_move_outputs_to_run_cwd`.
#[tokio::test]
async fn editor_scopes_paths_and_move_outputs_to_run_cwd() {
    let s = session();
    let editor = scoped(&s, "tasks/a");

    let created = editor
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "notes.txt",
            Some("+hello\n"),
        ))
        .await
        .unwrap();
    let moved = editor
        .update_file(&ApplyPatchOperation {
            move_to: Some("archive/notes.txt".to_string()),
            ..op(
                ApplyPatchOperationType::UpdateFile,
                "notes.txt",
                Some("@@\n-hello\n+hi\n"),
            )
        })
        .await
        .unwrap();

    assert_eq!(created.output.as_deref(), Some("Created notes.txt"));
    assert_eq!(
        moved.output.as_deref(),
        Some("Updated notes.txt\nMoved notes.txt to archive/notes.txt")
    );
    assert!(!s.files().contains_key("/workspace/tasks/a/notes.txt"));
    assert_eq!(s.files()["/workspace/tasks/a/archive/notes.txt"], b"hi");
}

/// Python: `test_editor_keeps_absolute_path_behavior_with_workspace_scope` and
/// `test_scoped_custom_tool_keeps_absolute_path_at_workspace_root`.
#[tokio::test]
async fn editor_keeps_absolute_paths_at_the_workspace_root_with_a_scope() {
    let s = session();
    let result = scoped(&s, "tasks/a")
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "/workspace/root.txt",
            Some("+root\n"),
        ))
        .await
        .unwrap();
    assert_eq!(result.output.as_deref(), Some("Created root.txt"));
    assert_eq!(s.files()["/workspace/root.txt"], b"root");

    let s = session();
    let tool = SandboxApplyPatchTool::<()>::new(scoped(&s, "tasks/a"));
    let outcome = execute(
        tool,
        "*** Begin Patch\n*** Add File: /workspace/root.txt\n+root\n*** End Patch\n",
    )
    .await;
    assert_eq!(outcome.output.as_deref(), Some("Created root.txt"));
    assert_eq!(s.files()["/workspace/root.txt"], b"root");
    assert!(!s.files().contains_key("/workspace/tasks/a/root.txt"));
}

/// Python: `test_editor_scoped_missing_error_uses_model_relative_path`.
#[tokio::test]
async fn editor_scoped_missing_error_uses_model_relative_path() {
    let s = session();
    s.with(|state| state.provider_private_not_found = true);

    let error = scoped(&s, "tasks/a")
        .update_file(&op(
            ApplyPatchOperationType::UpdateFile,
            "missing.txt",
            Some("@@\n-old\n+new\n"),
        ))
        .await
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "[apply_patch_file_not_found] apply_patch missing file: missing.txt"
    );
    assert!(!error.to_string().contains("/provider/private/root"));
}

/// Python: `test_editor_scoped_decode_error_uses_posix_model_relative_path_on_windows` and
/// `..._keeps_posix_absolute_path_on_windows`. The Windows host monkeypatch has no Rust
/// equivalent (paths here are `&str` on every host); what is checked is the model-relative path
/// in the message, for both a scoped relative path and an absolute one.
#[tokio::test]
async fn editor_scoped_decode_error_names_the_model_path() {
    let s = session();
    s.put("/workspace/tasks/a/nested/binary.txt", b"\xff\xfe\xfd");
    let error = scoped(&s, "tasks/a")
        .update_file(&op(
            ApplyPatchOperationType::UpdateFile,
            "nested/binary.txt",
            Some("@@\n+replacement\n"),
        ))
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "[apply_patch_decode_error] apply_patch could not decode file: nested/binary.txt"
    );

    s.put("/workspace/root/nested/binary.txt", b"\xff\xfe\xfd");
    let error = scoped(&s, "tasks/a")
        .update_file(&op(
            ApplyPatchOperationType::UpdateFile,
            "/workspace/root/nested/binary.txt",
            Some("@@\n+replacement\n"),
        ))
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "[apply_patch_decode_error] apply_patch could not decode file: /workspace/root/nested/binary.txt"
    );
}

/// Python: `test_direct_session_apply_patch_remains_workspace_root_relative`.
#[tokio::test]
async fn unscoped_editor_stays_workspace_root_relative() {
    let s = session();
    SandboxEditor::new(s.clone())
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "direct.txt",
            Some("+root\n"),
        ))
        .await
        .unwrap();
    scoped(&s, "tasks/a")
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "tool.txt",
            Some("+scoped\n"),
        ))
        .await
        .unwrap();

    assert_eq!(s.files()["/workspace/direct.txt"], b"root");
    assert_eq!(s.files()["/workspace/tasks/a/tool.txt"], b"scoped");
}

/// Python: `test_editor_runs_file_operations_as_bound_user`.
#[tokio::test]
async fn editor_runs_file_operations_as_the_bound_user() {
    let s = session();
    s.put("/workspace/existing.txt", b"old\n");
    let editor = SandboxEditor::new(s.clone()).with_run_as(Some(User::new("sandbox-user")));

    editor
        .update_file(&op(
            ApplyPatchOperationType::UpdateFile,
            "existing.txt",
            Some("@@\n-old\n+new\n"),
        ))
        .await
        .unwrap();
    editor
        .create_file(&op(
            ApplyPatchOperationType::CreateFile,
            "created.txt",
            Some("+created\n"),
        ))
        .await
        .unwrap();
    editor
        .delete_file(&op(
            ApplyPatchOperationType::DeleteFile,
            "existing.txt",
            None,
        ))
        .await
        .unwrap();

    let user = || vec![Some("sandbox-user".to_string())];
    let twice = || [user(), user()].concat();
    assert_eq!(s.users_for("read"), twice());
    assert_eq!(s.users_for("mkdir"), twice());
    assert_eq!(s.users_for("write"), twice());
    assert_eq!(s.users_for("rm"), user());
}

/// Python: `test_editor_removes_moved_source_as_bound_user`.
#[tokio::test]
async fn editor_removes_the_moved_source_as_the_bound_user() {
    let s = session();
    s.put("/workspace/existing.txt", b"old\n");
    let editor = SandboxEditor::new(s.clone()).with_run_as(Some(User::new("sandbox-user")));

    let result = editor
        .update_file(&ApplyPatchOperation {
            move_to: Some("moved.txt".to_string()),
            ..op(
                ApplyPatchOperationType::UpdateFile,
                "existing.txt",
                Some("@@\n-old\n+new\n"),
            )
        })
        .await
        .unwrap();

    assert_eq!(
        result.output.as_deref(),
        Some("Updated existing.txt\nMoved existing.txt to moved.txt")
    );
    let user = || vec![Some("sandbox-user".to_string())];
    assert_eq!(s.users_for("read"), user());
    assert_eq!(s.users_for("mkdir"), user());
    assert_eq!(s.users_for("write"), user());
    // Removing the source is part of the move, so it runs as the bound user too.
    assert_eq!(s.users_for("rm"), user());
    assert_eq!(s.files()["/workspace/moved.txt"], b"new\n");
    assert!(!s.files().contains_key("/workspace/existing.txt"));
}

/// Python: `test_editor_move_to_same_path_does_not_remove_the_file`.
#[tokio::test]
async fn editor_move_to_the_same_path_does_not_remove_the_file() {
    let s = session();
    s.put("/workspace/existing.txt", b"old\n");
    let editor = SandboxEditor::new(s.clone()).with_run_as(Some(User::new("sandbox-user")));

    editor
        .update_file(&ApplyPatchOperation {
            move_to: Some("existing.txt".to_string()),
            ..op(
                ApplyPatchOperationType::UpdateFile,
                "existing.txt",
                Some("@@\n-old\n+new\n"),
            )
        })
        .await
        .unwrap();

    assert!(s.users_for("rm").is_empty());
    assert_eq!(s.files()["/workspace/existing.txt"], b"new\n");
}

// ----------------------------------------------------------------------------------------------
// Patch language
// ----------------------------------------------------------------------------------------------

/// Python: `test_custom_tool_create_preserves_embedded_separators` (3 parameter sets).
#[tokio::test]
async fn custom_tool_create_preserves_embedded_separators() {
    for separator in ["\u{c}", "\u{2028}", "\r"] {
        let s = session();
        let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()));
        // A '+' after the embedded character is file content, not another patch line.
        let context = format!("section{separator}+continued");

        let created = tool
            .apply(&format!(
                "*** Begin Patch\n*** Add File: text.txt\n+{context}\n*** End Patch\n"
            ))
            .await
            .unwrap();

        assert_eq!(created, "Created text.txt");
        assert_eq!(s.files()["/workspace/text.txt"], context.as_bytes());
    }
}

/// Python: `test_custom_tool_update_preserves_embedded_separators` (3 parameter sets).
#[tokio::test]
async fn custom_tool_update_preserves_embedded_separators() {
    for separator in ["\u{c}", "\u{2028}", "\r"] {
        let s = session();
        let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()));
        let context = format!("section{separator}+continued");
        let original = format!("old{separator}value");
        let replacement = format!("new{separator}value");
        s.put(
            "/workspace/text.txt",
            format!("{context}\n{original}").as_bytes(),
        );

        let updated = tool
            .apply(&format!(
                "*** Begin Patch\n*** Update File: text.txt\n@@\n {context}\n-{original}\n+{replacement}\n*** End Patch\n"
            ))
            .await
            .unwrap();

        assert_eq!(updated, "Updated text.txt");
        assert_eq!(
            s.files()["/workspace/text.txt"],
            format!("{context}\n{replacement}").as_bytes()
        );
    }
}

/// Python: `test_custom_tool_patch_line_endings` (4 parameter sets).
#[tokio::test]
async fn custom_tool_patch_line_endings() {
    for newline in ["\n", "\r\n"] {
        for final_newline in [false, true] {
            let s = session();
            let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()));
            let mut patch = [
                "*** Begin Patch",
                "*** Add File: text.txt",
                "+first",
                "+second",
                "*** End Patch",
            ]
            .join(newline);
            if final_newline {
                patch.push_str(newline);
            }

            let result = tool.apply(&patch).await.unwrap();

            assert_eq!(result, "Created text.txt");
            assert_eq!(s.files()["/workspace/text.txt"], b"first\nsecond");
        }
    }
}

/// Python: `test_custom_tool_input_create_update_move_delete`.
#[tokio::test]
async fn custom_tool_input_create_update_move_delete() {
    let s = session();
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()));

    tool.apply("*** Begin Patch\n*** Add File: notes.txt\n+hello\n+world\n*** End Patch\n")
        .await
        .unwrap();
    assert_eq!(s.files()["/workspace/notes.txt"], b"hello\nworld");

    let result = tool
        .apply(
            "*** Begin Patch\n*** Update File: notes.txt\n*** Move to: moved.txt\n@@\n-hello\n+hi\n world\n*** End Patch\n",
        )
        .await
        .unwrap();
    assert!(result.contains("Updated notes.txt"), "{result}");
    assert!(result.contains("Moved notes.txt to moved.txt"), "{result}");
    assert!(!s.files().contains_key("/workspace/notes.txt"));
    assert_eq!(s.files()["/workspace/moved.txt"], b"hi\nworld");

    tool.apply("*** Begin Patch\n*** Delete File: moved.txt\n*** End Patch\n")
        .await
        .unwrap();
    assert!(!s.files().contains_key("/workspace/moved.txt"));
}

/// The JSON form models fall back to is accepted in all three shapes.
#[test]
fn json_input_forms_are_accepted() {
    let one = json!({"type": "delete_file", "path": "a.txt"}).to_string();
    let wrapped = json!({"operation": {"type": "delete_file", "path": "a.txt"}}).to_string();
    let many = json!({"operations": [
        {"type": "delete_file", "path": "a.txt"},
        {"type": "create_file", "path": "b.txt", "diff": "+x\n"}
    ]})
    .to_string();
    let array = json!([{"type": "delete_file", "path": "a.txt"}]).to_string();

    assert_eq!(parse_custom_input(&one).unwrap().len(), 1);
    assert_eq!(parse_custom_input(&wrapped).unwrap().len(), 1);
    assert_eq!(parse_custom_input(&many).unwrap().len(), 2);
    assert_eq!(parse_custom_input(&array).unwrap().len(), 1);
    assert!(parse_custom_input("{\"type\": \"create_file\", \"path\": \"a\"}").is_err());
}

// ----------------------------------------------------------------------------------------------
// Multi-operation approval matrix
// ----------------------------------------------------------------------------------------------

/// Python: `test_multi_operation_checker_ignores_native_sticky_approval`, the sandbox half
/// (`sandbox_approved` x `sandbox_always` x `round_trip`, 8 of its 16 parameter sets).
///
/// One patch holds a harmless create and a delete the rule holds for approval. Nothing is applied
/// while the call is pending; after the decision either both operations run or the model gets the
/// rejection message; a sticky decision then decides a follow-up call without asking the rule.
/// The native-sticky half (a native `apply_patch_call` approval sharing the context) is not
/// portable: Rust keys approvals by agent and tool name, and the native and custom tools are both
/// named `apply_patch`, so it is recorded as a cross-area finding in the report.
#[tokio::test]
async fn multi_operation_checker_honours_the_sandbox_decision() {
    for round_trip in [false, true] {
        for sandbox_approved in [true, false] {
            for sandbox_always in [false, true] {
                openai_agents_sdk::tracing::set_tracing_disabled(true);
                let s = session();
                s.put("/workspace/protected.txt", b"protected");
                let checked = Arc::new(Mutex::new(Vec::new()));
                let seen = checked.clone();
                let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(s.clone()))
                    .with_needs_approval_fn(move |_ctx, operation, _call_id| {
                        let seen = seen.clone();
                        async move {
                            seen.lock().unwrap().push(operation.path.clone());
                            operation.op_type == ApplyPatchOperationType::DeleteFile
                        }
                    });
                let model = Arc::new(FakeModel::new());
                let raw = "*** Begin Patch\n*** Add File: harmless.txt\n+harmless\n*** Delete File: protected.txt\n*** End Patch\n";
                model.add_turn(vec![model_item(json!({
                    "type": "custom_tool_call", "call_id": "call_apply",
                    "name": "apply_patch", "input": raw
                }))]);
                let agent = Agent::<()>::new("patcher")
                    .with_model_instance(model.clone())
                    .with_tool(Tool::Custom(Arc::new(tool.into_custom_tool())));

                let first = Runner::run(&agent, "go").await.unwrap();

                assert_eq!(first.interruptions.len(), 1);
                assert_eq!(*checked.lock().unwrap(), ["harmless.txt", "protected.txt"]);
                assert_eq!(
                    s.files(),
                    BTreeMap::from([(
                        "/workspace/protected.txt".to_string(),
                        b"protected".to_vec()
                    )])
                );
                let mut state = first.to_state();
                if round_trip {
                    state = openai_agents_sdk::RunState::from_json_str(
                        &state.to_json_string().unwrap(),
                    )
                    .unwrap();
                }
                let pending = state.get_interruptions()[0].clone();
                if sandbox_approved {
                    state.approve(&pending, sandbox_always);
                } else {
                    state.reject(
                        &pending,
                        sandbox_always,
                        Some("Sandbox patch denied".into()),
                    );
                }
                if round_trip {
                    state = openai_agents_sdk::RunState::from_json_str(
                        &state.to_json_string().unwrap(),
                    )
                    .unwrap();
                }
                model.add_turn(vec![assistant_message("done")]);

                let resumed = Runner::run(&agent, state).await.unwrap();

                let output = resumed.new_items.iter().find_map(|item| match &item.kind {
                    RunItemKind::ToolCallOutput { raw_item, .. } => raw_item
                        .get("output")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    _ => None,
                });
                if sandbox_approved {
                    // The rule ran for both operations before approval, not again on resume.
                    assert_eq!(
                        s.files(),
                        BTreeMap::from([(
                            "/workspace/harmless.txt".to_string(),
                            b"harmless".to_vec()
                        )])
                    );
                } else {
                    assert_eq!(output.as_deref(), Some("Sandbox patch denied"));
                    assert_eq!(
                        s.files(),
                        BTreeMap::from([(
                            "/workspace/protected.txt".to_string(),
                            b"protected".to_vec()
                        )])
                    );
                }
            }
        }
    }
}
