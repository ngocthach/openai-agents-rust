//! Port of `tests/sandbox/test_apply_patch.py` and its fixture
//! `tests/sandbox/_apply_patch_test_session.py`.
//!
//! The fixture is one in-memory session whose path-comparison behaviour is selected with
//! [`Store`], because Python builds a subclass per filesystem model.
#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::apply_diff::ApplyDiffMode;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::error::{AgentsError, Result as AgentsResult};
use openai_agents_sdk::sandbox::editor::{PatchFormat, SandboxEditor};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use serde_json::json;

mod py_sandbox_b_support;
use py_sandbox_b_support::{ApplyPatchSession, Store};

fn op(kind: ApplyPatchOperationType, path: &str, diff: Option<&str>) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: kind,
        path: path.to_string(),
        diff: diff.map(str::to_string),
        move_to: None,
    }
}

fn update(path: &str, diff: &str) -> ApplyPatchOperation {
    op(ApplyPatchOperationType::UpdateFile, path, Some(diff))
}

fn create(path: &str, diff: &str) -> ApplyPatchOperation {
    op(ApplyPatchOperationType::CreateFile, path, Some(diff))
}

fn moving(path: &str, diff: &str, move_to: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        move_to: Some(move_to.to_string()),
        ..update(path, diff)
    }
}

fn editor(session: &Arc<ApplyPatchSession>) -> SandboxEditor {
    SandboxEditor::new(session.clone())
}

/// Apply one operation the way `Session.apply_patch` does.
async fn apply(
    session: &Arc<ApplyPatchSession>,
    operation: ApplyPatchOperation,
) -> AgentsResult<String> {
    let editor = editor(session);
    let result = match operation.op_type {
        ApplyPatchOperationType::CreateFile => editor.create_file(&operation).await?,
        ApplyPatchOperationType::UpdateFile => editor.update_file(&operation).await?,
        ApplyPatchOperationType::DeleteFile => editor.delete_file(&operation).await?,
    };
    Ok(result.output.unwrap_or_default())
}

fn code(error: &AgentsError) -> &str {
    error.sandbox_code().unwrap_or("<not a sandbox error>")
}

fn session(store: Store) -> Arc<ApplyPatchSession> {
    Arc::new(ApplyPatchSession::new(store))
}

const ALPHA_BETA_TO_GAMMA: &str = "@@\n alpha\n-beta\n+gamma\n";

// ----------------------------------------------------------------------------------------------
// Diff application
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn update_invalid_context_raises() {
    let s = session(Store::Sensitive);
    s.put("/workspace/bad.txt", b"alpha\nbeta\n");

    let error = apply(&s, update("bad.txt", "@@\n missing\n-beta\n+gamma\n"))
        .await
        .unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
}

#[tokio::test]
async fn update_uses_anchor_jump() {
    let s = session(Store::Sensitive);
    s.put("/workspace/anchor.txt", b"a\nb\nmarker\nc\nd\n");

    apply(&s, update("anchor.txt", "@@ marker\n c\n-d\n+e\n"))
        .await
        .unwrap();

    assert_eq!(s.files()["/workspace/anchor.txt"], b"a\nb\nmarker\nc\ne\n");
}

#[tokio::test]
async fn update_uses_stacked_anchor_jump() {
    let s = session(Store::Sensitive);
    s.put(
        "/workspace/stacked.py",
        b"class First\n    def target():\n        return 0\n\nclass Second\n    def helper():\n        pass\n\n    def target():\n        pass\n",
    );

    apply(
        &s,
        update(
            "stacked.py",
            "@@ class Second\n@@     def target():\n-        pass\n+        return 1\n",
        ),
    )
    .await
    .unwrap();

    assert_eq!(
        s.files()["/workspace/stacked.py"],
        b"class First\n    def target():\n        return 0\n\nclass Second\n    def helper():\n        pass\n\n    def target():\n        return 1\n"
    );
}

#[tokio::test]
async fn update_rejects_partially_matched_stacked_anchors() {
    let s = session(Store::Sensitive);
    let original =
        b"class Target\n    def helper():\n        pass\n\n    def desired():\n        return 1\n";
    s.put("/workspace/stacked.py", original);

    let error = apply(
        &s,
        update(
            "stacked.py",
            "@@ class Target\n@@     def missing():\n-        pass\n+        return 99\n",
        ),
    )
    .await
    .unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
    assert!(error.to_string().contains("Invalid Anchor"), "{error}");
    assert_eq!(s.files()["/workspace/stacked.py"], original);
}

#[tokio::test]
async fn update_matches_end_of_file_context() {
    let s = session(Store::Sensitive);
    s.put("/workspace/tail.txt", b"one\ntwo\nthree\n");

    apply(
        &s,
        update("tail.txt", "@@\n two\n-three\n+four\n*** End of File\n"),
    )
    .await
    .unwrap();

    assert_eq!(s.files()["/workspace/tail.txt"], b"one\ntwo\nfour\n");
}

#[tokio::test]
async fn update_missing_diff_raises() {
    let s = session(Store::Sensitive);

    let error = apply(
        &s,
        op(ApplyPatchOperationType::UpdateFile, "file.txt", None),
    )
    .await
    .unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
}

#[tokio::test]
async fn update_missing_file_raises() {
    let s = session(Store::Sensitive);

    let error = apply(&s, update("missing.txt", "@@\n-old\n+new\n"))
        .await
        .unwrap_err();

    assert_eq!(code(&error), "apply_patch_file_not_found");
}

#[tokio::test]
async fn delete_missing_file_raises() {
    let s = session(Store::Sensitive);

    let error = apply(
        &s,
        op(ApplyPatchOperationType::DeleteFile, "nope.txt", None),
    )
    .await
    .unwrap_err();

    assert_eq!(code(&error), "apply_patch_file_not_found");
}

#[tokio::test]
async fn missing_file_errors_use_workspace_path() {
    let s = session(Store::Sensitive);
    s.with(|state| state.provider_private_not_found = true);

    let update_error = apply(&s, update("missing.txt", "@@\n-old\n+new\n"))
        .await
        .unwrap_err();
    assert_eq!(code(&update_error), "apply_patch_file_not_found");
    assert_eq!(
        update_error.to_string(),
        "[apply_patch_file_not_found] apply_patch missing file: missing.txt"
    );
    assert!(!update_error.to_string().contains("/provider/private/root"));

    let delete_error = apply(
        &s,
        op(
            ApplyPatchOperationType::DeleteFile,
            "missing-delete.txt",
            None,
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(
        delete_error.to_string(),
        "[apply_patch_file_not_found] apply_patch missing file: missing-delete.txt"
    );
    assert!(!delete_error.to_string().contains("/provider/private/root"));
}

#[tokio::test]
async fn rejects_escape_root_path() {
    let s = session(Store::Sensitive);

    let error = apply(&s, create("../escape.txt", "+nope"))
        .await
        .unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_path");
    assert!(
        error.to_string().contains("must not escape root"),
        "{error}"
    );
}

#[tokio::test]
async fn rejects_empty_path() {
    let s = session(Store::Sensitive);

    let error = apply(&s, create("", "+nope")).await.unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_path");
    assert!(error.to_string().contains("must be non-empty"), "{error}");
}

#[tokio::test]
async fn normalizes_backslashes_in_string_path() {
    let s = session(Store::Sensitive);

    apply(&s, create(r"nested\new.txt", "+hello"))
        .await
        .unwrap();

    assert_eq!(s.files()["/workspace/nested/new.txt"], b"hello");
}

#[tokio::test]
async fn normalizes_backslashes_in_move_to() {
    let s = session(Store::Sensitive);
    s.put("/workspace/source.txt", b"alpha\n");

    apply(
        &s,
        moving("source.txt", "@@\n-alpha\n+beta\n", r"nested\moved.txt"),
    )
    .await
    .unwrap();

    assert_eq!(s.files()["/workspace/nested/moved.txt"], b"beta\n");
    assert!(!s.files().contains_key("/workspace/source.txt"));
}

// ----------------------------------------------------------------------------------------------
// move_to and filesystem aliasing
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn case_only_move_to_keeps_file_on_case_folding_filesystem() {
    let s = session(Store::CaseFolding);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/Notes.txt".to_string(),
            b"alpha\ngamma\n".to_vec()
        )])
    );
    assert_eq!(s.mv_calls().len(), 2);
    assert_eq!(
        s.mv_calls()[1],
        (
            "/workspace/notes.txt".to_string(),
            "/workspace/Notes.txt".to_string()
        )
    );
    assert!(s.rm_calls().is_empty());
}

#[tokio::test]
async fn case_only_move_to_moves_file_on_case_sensitive_filesystem() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/Notes.txt".to_string(),
            b"alpha\ngamma\n".to_vec()
        )])
    );
}

/// Python: `test_apply_patch_case_only_move_to_renames_from_a_case_folding_host`. The premise is
/// a host whose `Path` equality folds case; Rust compares `&str` spellings on every host, so the
/// behaviour it guards (a case-only `move_to` is a real rename) is the case-sensitive test above.
#[tokio::test]
async fn case_only_move_to_is_a_rename_decided_by_the_sandbox() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap();

    assert!(s.files().contains_key("/workspace/Notes.txt"));
    assert!(!s.files().contains_key("/workspace/notes.txt"));
}

#[tokio::test]
async fn same_leaf_through_parent_alias_does_not_move_twice() {
    let s = session(Store::ParentAlias);
    s.put("/workspace/real/notes.txt", b"alpha\nbeta\n");

    apply(
        &s,
        moving("real/notes.txt", ALPHA_BETA_TO_GAMMA, "alias/notes.txt"),
    )
    .await
    .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/real/notes.txt".to_string(),
            b"alpha\ngamma\n".to_vec()
        )])
    );
    assert_eq!(s.mv_calls().len(), 1);
    assert!(s.rm_calls().is_empty());
}

#[tokio::test]
async fn move_to_leaves_the_source_alone_when_the_write_fails() {
    let s = session(Store::CaseFolding);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");
    s.with(|state| state.fail_next_write = true);

    let error = apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("sandbox write failed"),
        "{error}"
    );
    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/notes.txt".to_string(),
            b"alpha\nbeta\n".to_vec()
        )])
    );
    let removed = s.rm_calls();
    assert!(removed.iter().all(|(path, _)| {
        path.rsplit('/')
            .next()
            .unwrap()
            .starts_with(".apply_patch-")
    }));
    assert!(
        !removed
            .iter()
            .any(|(path, _)| path == "/workspace/notes.txt")
    );
}

#[tokio::test]
async fn move_to_does_not_overwrite_a_concurrent_writer_after_a_failure() {
    let s = session(Store::CaseFolding);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");
    s.with(|state| {
        state.concurrent_source = Some("/workspace/notes.txt".to_string());
        state.fail_moves = true;
    });

    let error = apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("sandbox move failed"), "{error}");
    let files = s.files();
    assert_eq!(files["/workspace/notes.txt"], b"written by someone else\n");
    assert!(!files.contains_key("/workspace/Notes.txt"));
    assert!(!files.keys().any(|path| path.ends_with(".tmp")));
}

#[tokio::test]
async fn move_to_keeps_the_file_when_only_unicode_normalization_changes() {
    let s = session(Store::NormalizationFolding);
    let decomposed = "/workspace/cafe\u{301}.txt";
    let composed = "/workspace/caf\u{e9}.txt";
    s.put(decomposed, b"alpha\nbeta\n");

    apply(&s, moving(decomposed, ALPHA_BETA_TO_GAMMA, composed))
        .await
        .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(composed.to_string(), b"alpha\ngamma\n".to_vec())])
    );
}

#[tokio::test]
async fn move_to_an_existing_directory_keeps_the_source() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");
    s.with(|state| {
        state.directories.insert("/workspace/docs".to_string());
    });

    let error = apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "docs"))
        .await
        .unwrap_err();

    // `docs` is a directory in the store, so the write to it is refused before anything moves.
    assert!(error.to_string().contains("IsADirectoryError"), "{error}");
    assert_eq!(s.files()["/workspace/notes.txt"], b"alpha\nbeta\n");
    assert!(!s.files().keys().any(|path| path.ends_with(".tmp")));
}

#[tokio::test]
async fn move_to_commits_the_destination_before_removing_the_source() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap();

    // Ordinary moves write the destination, then remove the source, without staging.
    assert!(s.mv_calls().is_empty());
    assert_eq!(
        s.rm_calls(),
        vec![("/workspace/notes.txt".to_string(), false)]
    );
    assert_eq!(s.files()["/workspace/Notes.txt"], b"alpha\ngamma\n");
}

#[tokio::test]
async fn move_to_removes_a_source_symlink_pointing_at_the_destination() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");
    s.put("/workspace/Notes.txt", b"alpha\nbeta\n");
    s.with(|state| {
        state.symlinks.insert(
            "/workspace/notes.txt".to_string(),
            "/workspace/Notes.txt".to_string(),
        );
    });

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "Notes.txt"))
        .await
        .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/Notes.txt".to_string(),
            b"alpha\ngamma\n".to_vec()
        )])
    );
    assert!(s.mv_calls().is_empty());
    assert_eq!(
        s.rm_calls(),
        vec![("/workspace/notes.txt".to_string(), false)]
    );
}

#[tokio::test]
async fn move_to_the_same_path_writes_in_place() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\nbeta\n");

    apply(&s, moving("notes.txt", ALPHA_BETA_TO_GAMMA, "notes.txt"))
        .await
        .unwrap();

    assert_eq!(
        s.files(),
        BTreeMap::from([(
            "/workspace/notes.txt".to_string(),
            b"alpha\ngamma\n".to_vec()
        )])
    );
    assert!(s.mv_calls().is_empty());
    assert!(s.rm_calls().is_empty());
}

#[tokio::test]
async fn move_to_a_long_name_keeps_the_staging_name_within_the_limit() {
    let s = session(Store::CaseFolding);
    let long_name = format!("{}.txt", "n".repeat(250));
    s.put(&format!("/workspace/{long_name}"), b"alpha\nbeta\n");

    apply(
        &s,
        moving(&long_name, ALPHA_BETA_TO_GAMMA, &long_name.to_uppercase()),
    )
    .await
    .unwrap();

    assert_eq!(s.mv_calls().len(), 2);
    let (staging, _) = &s.mv_calls()[0];
    assert!(
        staging.rsplit('/').next().unwrap().len() <= 255,
        "{staging}"
    );
    assert_eq!(
        s.files(),
        BTreeMap::from([(
            format!("/workspace/{}", long_name.to_uppercase()),
            b"alpha\ngamma\n".to_vec()
        )])
    );
}

// ----------------------------------------------------------------------------------------------
// Paths, diffs and formats
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn allows_absolute_path_within_root() {
    let s = session(Store::Sensitive);

    apply(&s, create("/workspace/abs-ok.txt", "+hello"))
        .await
        .unwrap();

    assert_eq!(s.files()["/workspace/abs-ok.txt"], b"hello");
}

#[tokio::test]
async fn rejects_absolute_path_outside_root() {
    let s = session(Store::Sensitive);

    let error = apply(&s, create("/tmp/outside.txt", "+nope"))
        .await
        .unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_path");
}

#[tokio::test]
async fn create_requires_plus_lines() {
    let s = session(Store::Sensitive);

    let error = apply(&s, create("new.txt", "oops")).await.unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
}

#[tokio::test]
async fn rejects_invalid_diff_line_prefix() {
    let s = session(Store::Sensitive);
    s.put("/workspace/oops.txt", b"alpha\nbeta\n");

    let error = apply(&s, update("oops.txt", "oops")).await.unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
}

#[tokio::test]
async fn update_non_utf8_payload_raises() {
    let s = session(Store::Sensitive);
    s.put("/workspace/binary.txt", b"\xff\xfe\xfd");

    let error = apply(&s, update("binary.txt", "@@\n+\n"))
        .await
        .unwrap_err();

    assert_eq!(code(&error), "apply_patch_decode_error");
}

#[tokio::test]
async fn uses_custom_patch_format() {
    #[derive(Debug)]
    struct StubFormat;
    impl PatchFormat for StubFormat {
        fn apply_diff(
            &self,
            input: &str,
            _diff: &str,
            mode: ApplyDiffMode,
        ) -> AgentsResult<String> {
            let mode = match mode {
                ApplyDiffMode::Default => "default",
                ApplyDiffMode::Create => "create",
            };
            Ok(input.replace("world", mode))
        }
    }
    let s = session(Store::Sensitive);
    s.put("/workspace/custom.txt", b"hello\nworld\n");

    let editor = editor(&s).with_patch_format(Arc::new(StubFormat));
    editor
        .update_file(&update("custom.txt", "@@\n hello\n-world\n+ignored\n"))
        .await
        .unwrap();

    assert_eq!(s.files()["/workspace/custom.txt"], b"hello\ndefault\n");
}

#[tokio::test]
async fn supports_non_default_root() {
    let s = Arc::new(ApplyPatchSession::with_manifest(
        Store::Sensitive,
        Manifest::new().with_root("/custom-workspace"),
    ));

    apply(&s, create("new.txt", "+hello")).await.unwrap();

    assert_eq!(s.files()["/custom-workspace/new.txt"], b"hello");
}

#[tokio::test]
async fn mapping_operation_moves_file() {
    let s = session(Store::Sensitive);
    s.put("/workspace/old.txt", b"alpha\n");
    let operation: ApplyPatchOperation = serde_json::from_value(json!({
        "type": "update_file", "path": "old.txt",
        "diff": "@@\n-alpha\n+beta\n", "move_to": "renamed/new.txt"
    }))
    .unwrap();

    apply(&s, operation).await.unwrap();

    assert_eq!(s.files()["/workspace/renamed/new.txt"], b"beta\n");
    assert!(!s.files().contains_key("/workspace/old.txt"));
}

#[tokio::test]
async fn mapping_operation_without_move_to_updates_in_place() {
    let s = session(Store::Sensitive);
    s.put("/workspace/keep.txt", b"alpha\n");
    let operation: ApplyPatchOperation = serde_json::from_value(json!({
        "type": "update_file", "path": "keep.txt", "diff": "@@\n-alpha\n+beta\n"
    }))
    .unwrap();

    apply(&s, operation).await.unwrap();

    assert_eq!(s.files()["/workspace/keep.txt"], b"beta\n");
    assert!(s.rm_calls().is_empty());
}

/// Python: `test_apply_patch_mapping_operation_rejects_non_string_move_to`. A mapping is a typed
/// struct in Rust, so the malformed value is refused at deserialization, before any editor call.
#[tokio::test]
async fn mapping_operation_rejects_non_string_move_to() {
    let s = session(Store::Sensitive);
    s.put("/workspace/old.txt", b"alpha\n");

    let parsed = serde_json::from_value::<ApplyPatchOperation>(json!({
        "type": "update_file", "path": "old.txt",
        "diff": "@@\n-alpha\n+beta\n", "move_to": 5
    }));

    assert!(parsed.is_err());
    assert_eq!(s.files()["/workspace/old.txt"], b"alpha\n");
}

#[tokio::test]
async fn create_rejects_an_existing_file() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\n");

    let error = apply(&s, create("notes.txt", "+beta\n")).await.unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
    assert!(error.to_string().contains("already exists"), "{error}");
    assert_eq!(s.files()["/workspace/notes.txt"], b"alpha\n");
}

/// A backend with an exclusive create reports the name as taken even when every read says the
/// path is free, so the rejection comes from the primitive and not from a probe.
#[tokio::test]
async fn create_rejects_an_existing_file_without_reading_it() {
    let s = session(Store::Sensitive);
    s.put("/workspace/notes.txt", b"alpha\n");
    s.with(|state| state.reads_always_missing = true);

    let error = apply(&s, create("notes.txt", "+beta\n")).await.unwrap_err();

    assert_eq!(code(&error), "apply_patch_invalid_diff");
    assert_eq!(s.files()["/workspace/notes.txt"], b"alpha\n");
}

// ----------------------------------------------------------------------------------------------
// Default `mv` through a real shell
// ----------------------------------------------------------------------------------------------

/// A session whose `exec` runs the command on the host, so the SDK's default `mv`, which is
/// shell-based, is exercised against a real `mv`.
#[derive(Debug)]
struct ShellSession {
    manifest: Manifest,
    /// Create this directory just before the move, as a concurrent writer would.
    mkdir_before_exec: Option<std::path::PathBuf>,
    /// Replace `mv` in the script with this executable.
    mv_executable: Option<String>,
}

#[async_trait]
impl SandboxSession for ShellSession {
    fn id(&self) -> &str {
        "shell"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "shell"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        if let Some(directory) = &self.mkdir_before_exec {
            std::fs::create_dir(directory).unwrap();
        }
        let mut argv = request.command.clone();
        if let Some(executable) = &self.mv_executable {
            argv[2] = argv[2].replace("exec mv ", &format!("exec '{executable}' "));
        }
        let output = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .unwrap();
        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
    async fn read_file(&self, _path: &str) -> SandboxResult<Vec<u8>> {
        unreachable!()
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        unreachable!()
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

/// `mv -T` is a GNU option. Linux ships it; macOS needs `gmv`. Python skips without one.
fn gnu_mv() -> Option<String> {
    for candidate in ["gmv", "mv"] {
        let probe = std::process::Command::new(candidate)
            .arg("--version")
            .output();
        if let Ok(output) = probe
            && String::from_utf8_lossy(&output.stdout).contains("GNU")
        {
            return Some(candidate.to_string());
        }
    }
    None
}

#[tokio::test]
async fn fallback_move_cannot_put_source_inside_a_new_directory() {
    let Some(executable) = gnu_mv() else {
        eprintln!("skipped: requires a GNU-compatible mv");
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    std::fs::write(&source, b"original").unwrap();
    let destination = temp.path().join("destination");
    let session = ShellSession {
        manifest: Manifest::new().with_root(temp.path().to_string_lossy()),
        // The directory arrives after validation but before the actual move.
        mkdir_before_exec: Some(destination.clone()),
        mv_executable: Some(executable),
    };

    let error = session
        .mv(&source.to_string_lossy(), &destination.to_string_lossy())
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert_eq!(std::fs::read(&source).unwrap(), b"original");
    assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 0);
}

#[tokio::test]
async fn fallback_move_replaces_a_directory_symlink() {
    let Some(executable) = gnu_mv() else {
        eprintln!("skipped: requires a GNU-compatible mv");
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    std::fs::write(&source, b"replacement").unwrap();
    let directory = temp.path().join("directory");
    std::fs::create_dir(&directory).unwrap();
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&directory, &link).unwrap();
    let session = ShellSession {
        manifest: Manifest::new().with_root(temp.path().to_string_lossy()),
        mkdir_before_exec: None,
        mv_executable: Some(executable),
    };

    session
        .mv(&source.to_string_lossy(), &link.to_string_lossy())
        .await
        .unwrap();

    assert!(!link.is_symlink());
    assert_eq!(std::fs::read(&link).unwrap(), b"replacement");
    assert!(directory.is_dir());
    assert!(!source.exists());
}

#[tokio::test]
async fn fallback_move_rejects_directory_source_through_parent_alias() {
    use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let shared = root.join("shared");
    let protected = shared.join("tree").join("private");
    std::fs::create_dir_all(&protected).unwrap();
    let payload = protected.join("file");
    std::fs::write(&payload, b"protected").unwrap();
    std::os::unix::fs::symlink(&shared, workspace.join("alias")).unwrap();
    let grant = |path: &std::path::Path, read_only: bool| {
        SandboxPathGrant::new(path.to_string_lossy())
            .unwrap()
            .read_only(read_only)
    };
    let session = ShellSession {
        manifest: Manifest::new()
            .with_root(workspace.to_string_lossy())
            .with_extra_path_grant(grant(&shared, false))
            .with_extra_path_grant(grant(&protected, true)),
        mkdir_before_exec: None,
        mv_executable: None,
    };

    let error = session
        .mv(
            &workspace.join("alias/tree").to_string_lossy(),
            &workspace.join("moved").to_string_lossy(),
        )
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert!(
        error.message.contains("regular file or symlink"),
        "{}",
        error.message
    );
    assert_eq!(std::fs::read(&payload).unwrap(), b"protected");
    assert!(!workspace.join("moved").exists());
}

/// Python: `test_sandbox_session_forwards_follow_symlinks_to_the_inner_session` tests that the
/// `SandboxSession` wrapper forwards a keyword argument to an inner `MagicMock`. Rust has no
/// wrapper class and the argument is positional on the trait, so there is nothing to forward;
/// what matters is that the flag changes the answer, which this checks directly.
#[tokio::test]
async fn same_file_honors_follow_symlinks() {
    let s = session(Store::Sensitive);
    s.put("/workspace/target.txt", b"x");
    s.put("/workspace/link.txt", b"x");
    s.with(|state| {
        state.symlinks.insert(
            "/workspace/link.txt".to_string(),
            "/workspace/target.txt".to_string(),
        );
    });

    assert!(
        s.same_file("/workspace/link.txt", "/workspace/target.txt", true, None)
            .await
            .unwrap()
    );
    assert!(
        !s.same_file("/workspace/link.txt", "/workspace/target.txt", false, None)
            .await
            .unwrap()
    );
}
