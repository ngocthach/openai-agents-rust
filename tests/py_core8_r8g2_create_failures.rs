//! Caller-visible no-clobber and partial-write recovery for Unix-local Add File, from
//! `tests/sandbox/test_unix_local_create.py`.
//!
//! Python injects the faults by monkeypatching `os.open`, `shutil.copyfileobj` and `os.fdopen`.
//! Here a creator that appears between validation and open comes from the session's
//! `with_after_validate_hook`, and a failure in the middle of the write comes from the kernel:
//! a tiny `RLIMIT_FSIZE` makes the second `write(2)` fail with `EFBIG`.
//!
//! The limit is process-wide, so it also applies to the test harness: when its output goes to a
//! large regular file (a CI log), printing another test's result while the limit is lowered
//! fails with `EFBIG` and aborts the binary. The cases therefore run one after another inside a
//! single test, so the harness prints nothing until the limit is restored.
#![cfg(all(unix, feature = "sandbox"))]

use std::path::Path;
use std::sync::Arc;

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxSession;
use openai_agents_sdk::sandbox::{Manifest, SandboxSession};

fn session_at(root: &Path) -> UnixLocalSandboxSession {
    UnixLocalSandboxSession::new(
        Manifest::new().with_root(root.to_string_lossy().into_owned()),
        false,
    )
    .expect("a session over an absolute root")
}

fn create_op(path: &str, diff: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: None,
    }
}

/// The two ways Python's `editor` fixture reaches a session that this port can run: the session
/// itself and the session behind the instrumenting wrapper (`SandboxSession(session)`).
fn editors(session: UnixLocalSandboxSession) -> Vec<(&'static str, SandboxEditor)> {
    let direct: Arc<dyn SandboxSession> = Arc::new(session);
    let wrapped: Arc<dyn SandboxSession> = Arc::new(InstrumentedSession::new(
        direct.clone(),
        Arc::new(CallbackSink::new(|_| {})),
    ));
    vec![
        ("direct", SandboxEditor::new(direct)),
        ("wrapped", SandboxEditor::new(wrapped)),
    ]
}

fn limit_file_size(bytes: libc::rlim_t) -> libc::rlimit {
    let mut previous = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: plain syscalls on stack structs; the caller restores the previous soft limit.
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_FSIZE, &mut previous), 0);
        // Exceeding the limit would otherwise terminate the process with SIGXFSZ.
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        let tightened = libc::rlimit {
            rlim_cur: bytes,
            rlim_max: previous.rlim_max,
        };
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &tightened), 0);
    }
    previous
}

fn restore_file_size(previous: libc::rlimit) {
    // SAFETY: restores the soft limit read by `limit_file_size`.
    unsafe {
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &previous), 0);
    }
}

/// Python: `test_create_preserves_a_creator_between_validation_and_open` (the `direct` and
/// `wrapped` fixtures; `bound-user` runs Python's `sudo` worker).
async fn a_creator_between_validation_and_open_keeps_its_file() {
    for fixture in ["direct", "wrapped"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = std::fs::canonicalize(dir.path()).expect("canonical root");
        let target = root.join("notes.txt");
        let intervened = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = intervened.clone();
        let hook_target = target.clone();
        let session = session_at(&root).with_after_validate_hook(Arc::new(move |host: &Path| {
            if host.ends_with("notes.txt") {
                assert!(!hook_target.exists());
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                std::fs::write(&hook_target, b"other creator\n").expect("another creator");
            }
        }));
        let (_, editor) = editors(session)
            .into_iter()
            .find(|(name, _)| *name == fixture)
            .unwrap();

        let error = editor
            .create_file(&create_op("notes.txt", "+replacement\n"))
            .await
            .expect_err("the name was taken in the window");

        assert!(error.to_string().contains("already exists"), "{error}");
        assert!(intervened.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"other creator\n",
            "{fixture}"
        );
    }
}

/// Python: `test_failed_create_requires_inspection_and_never_deletes_a_replacement`
/// (`replace_target=False`, the `direct` and `wrapped` fixtures).
async fn a_failed_create_requires_inspection_and_keeps_what_was_written() {
    for fixture in ["direct", "wrapped"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = std::fs::canonicalize(dir.path()).expect("canonical root");
        let target = root.join("notes.txt");
        let (_, editor) = editors(session_at(&root))
            .into_iter()
            .find(|(name, _)| *name == fixture)
            .unwrap();

        // The first four bytes fit; the next write hits the limit mid-file.
        let previous = limit_file_size(4);
        let failure = editor
            .create_file(&create_op("notes.txt", "+complete\n"))
            .await;
        restore_file_size(previous);

        let error = failure.expect_err("the write fails after the name is claimed");
        assert_eq!(
            error.sandbox_code(),
            Some("workspace_archive_write_error"),
            "{fixture}: {error}"
        );
        assert!(
            error.to_string().contains("Inspect the destination"),
            "{error}"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"comp", "{fixture}");

        // Retrying without inspecting is refused, and nothing is replaced.
        let again = editor
            .create_file(&create_op("notes.txt", "+complete\n"))
            .await
            .expect_err("the incomplete file still claims the name");
        assert!(again.to_string().contains("already exists"), "{again}");
        assert_eq!(std::fs::read(&target).unwrap(), b"comp", "{fixture}");

        // The caller has inspected and deliberately removed its incomplete result.
        std::fs::remove_file(&target).unwrap();
        let created = editor
            .create_file(&create_op("notes.txt", "+complete\n"))
            .await
            .expect("a retry after inspection");
        assert_eq!(created.output.as_deref(), Some("Created notes.txt"));
        assert_eq!(std::fs::read(&target).unwrap(), b"complete", "{fixture}");
    }
}

/// The error a failed create reports is not retryable, whatever the caller does with it.
async fn an_incomplete_create_is_not_retryable() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = std::fs::canonicalize(dir.path()).expect("canonical root");
    let session = session_at(&root);

    let previous = limit_file_size(4);
    let failure = session.write_new_file("notes.txt", b"complete", None).await;
    restore_file_size(previous);

    let error = failure.expect_err("the write fails after the name is claimed");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(error.retryable, Some(false));
    assert!(error.message.contains("Inspect the destination"));
}

#[tokio::test]
async fn unix_local_create_failures() {
    a_creator_between_validation_and_open_keeps_its_file().await;
    a_failed_create_requires_inspection_and_keeps_what_was_written().await;
    an_incomplete_create_is_not_retryable().await;
}
