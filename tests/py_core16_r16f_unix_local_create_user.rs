//! The bound-user variants of `tests/sandbox/test_unix_local_create.py`: Add File through an
//! editor that runs as another user keeps the no-clobber guarantee of the owner path.
//!
//! Python swaps the `sudo` boundary for an in-process worker; here a `sudo` stand-in earlier on
//! the manifest's `PATH` drops `-u <user> --` and runs the wrapped command as the current user.
#![cfg(all(unix, feature = "sandbox"))]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxSession;
use openai_agents_sdk::sandbox::types::User;
use openai_agents_sdk::sandbox::{Manifest, SandboxSession};

fn fake_sudo(dir: &Path) {
    let path = dir.join("sudo");
    std::fs::write(&path, "#!/bin/sh\nshift 3\nexec \"$@\"\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn create_op(path: &str, diff: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: None,
    }
}

type AfterValidateHook = Arc<dyn Fn(&Path) + Send + Sync>;

async fn bound_editor(root: &Path, bin: &Path, hook: Option<AfterValidateHook>) -> SandboxEditor {
    let manifest = Manifest::new()
        .with_root(root.to_string_lossy().into_owned())
        .with_env("PATH", format!("{}:/usr/bin:/bin", bin.display()));
    let mut session = UnixLocalSandboxSession::new(manifest, false).expect("session");
    if let Some(hook) = hook {
        session = session.with_after_validate_hook(hook);
    }
    session.start().await.expect("started");
    let session: Arc<dyn SandboxSession> = Arc::new(session);
    SandboxEditor::new(session).with_run_as(Some(User::new("example-user")))
}

fn layout() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap().join("workspace");
    let bin = std::fs::canonicalize(dir.path()).unwrap().join("bin");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&bin).unwrap();
    fake_sudo(&bin);
    (dir, root, bin)
}

/// `test_create_preserves_a_creator_between_validation_and_open[bound-user]`
#[tokio::test]
async fn a_creator_between_validation_and_write_keeps_its_file() {
    let (_dir, root, bin) = layout();
    let target = root.join("notes.txt");
    let intervened = Arc::new(AtomicBool::new(false));
    let (flag, hook_target) = (intervened.clone(), target.clone());
    let editor = bound_editor(
        &root,
        &bin,
        Some(Arc::new(move |host: &Path| {
            if host.ends_with("notes.txt") {
                assert!(!hook_target.exists());
                flag.store(true, Ordering::SeqCst);
                std::fs::write(&hook_target, b"other creator\n").unwrap();
            }
        })),
    )
    .await;

    let error = editor
        .create_file(&create_op("notes.txt", "+replacement\n"))
        .await
        .expect_err("the name was taken in the window");

    assert!(error.to_string().contains("already exists"), "{error}");
    assert!(intervened.load(Ordering::SeqCst));
    assert_eq!(std::fs::read(&target).unwrap(), b"other creator\n");
    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(leftovers.len(), 1, "no staging file is left: {leftovers:?}");
}

/// `test_create_writes_nested_file_and_rejects_dangling_leaf[bound-user]`
#[tokio::test]
async fn nested_create_through_an_alias_and_a_dangling_leaf_is_refused() {
    let (_dir, root, bin) = layout();
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
    let editor = bound_editor(&root, &bin, None).await;

    let result = editor
        .create_file(&create_op("alias/nested/new.txt", "+contents\n"))
        .await
        .expect("created");
    assert_eq!(
        result.output.as_deref(),
        Some("Created alias/nested/new.txt")
    );
    assert_eq!(
        std::fs::read(root.join("real/nested/new.txt")).unwrap(),
        b"contents"
    );

    std::os::unix::fs::symlink(root.join("missing.txt"), root.join("alias/link.txt")).unwrap();
    let error = editor
        .create_file(&create_op("alias/link.txt", "+contents\n"))
        .await
        .expect_err("a dangling leaf is an existing name");
    assert!(error.to_string().contains("already exists"), "{error}");
    assert!(!root.join("missing.txt").exists());
    assert!(
        std::fs::symlink_metadata(root.join("alias/link.txt"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
