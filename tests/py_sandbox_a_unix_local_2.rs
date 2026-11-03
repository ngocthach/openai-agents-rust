//! Parity port of the Unix-local file-operation and PTY suites:
//! `tests/sandbox/test_unix_local_file_io.py`, `test_unix_local_user_file_io.py`,
//! `test_unix_local_special_files.py`, `test_unix_local_create.py`, and the
//! `TestUnixLocalPty` / `TestUnixLocalApplyPatchRename` classes of `test_unix_local.py`.
//!
//! Everything runs against a real workspace directory, as Python's `tmp_path` fixtures do.
#![allow(clippy::await_holding_lock)] // the lock serializes process-wide state for the whole test
#![cfg(all(unix, feature = "sandbox"))]

use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _, PermissionsExt};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::sandboxes::unix_local::{
    UnixLocalSandboxClient, UnixLocalSandboxSession,
};
use openai_agents_sdk::sandbox::{
    ExecShell, Manifest, SandboxClient, SandboxPathGrant, SandboxSession, User,
};

/// Guards the process-wide umask, which one test has to change.
static UMASK_LOCK: Mutex<()> = Mutex::new(());

/// How long a FIFO operation may take before the test calls it a hang. Python bounds the same
/// regressions with a child process and a 15s timeout.
const FIFO_TIMEOUT: Duration = Duration::from_secs(5);

/// Python's `_session(root, grants=...)`: the real backend over `root`, nothing started.
fn session(root: &Path, grants: &[SandboxPathGrant]) -> Arc<dyn SandboxSession> {
    let mut manifest = Manifest::new().with_root(root.to_string_lossy().into_owned());
    for grant in grants {
        manifest = manifest.with_extra_path_grant(grant.clone());
    }
    Arc::new(
        UnixLocalSandboxSession::new(manifest, false).expect("a session over an absolute root"),
    )
}

fn grant(path: impl AsRef<Path>) -> SandboxPathGrant {
    SandboxPathGrant::new(path.as_ref().to_string_lossy()).expect("a grant for an absolute path")
}

fn read_only_grant(path: impl AsRef<Path>) -> SandboxPathGrant {
    grant(path).read_only(true)
}

fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(target.as_ref(), link.as_ref()).expect("create a symlink");
}

fn make_fifo(path: &Path) {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).expect("c string");
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the call.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) }, 0);
}

fn update_op(path: &str, diff: &str, move_to: Option<&str>) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::UpdateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: move_to.map(str::to_string),
    }
}

fn delete_op(path: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::DeleteFile,
        path: path.to_string(),
        diff: None,
        move_to: None,
    }
}

fn create_op(path: &str, diff: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: None,
    }
}

fn is_root() -> bool {
    // SAFETY: `geteuid` reads the calling process's effective uid and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

// ---------------------------------------------------------------------------------------------
// PTY — Python `TestUnixLocalPty`
// ---------------------------------------------------------------------------------------------

/// A started client session over a temporary workspace.
async fn started_session(root: &Path) -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_root(root.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create a session");
    session.start().await.expect("start");
    (client, session)
}

/// Python: `test_pty_exec_write_poll_and_unknown_session_errors`.
#[tokio::test]
async fn pty_exec_write_poll_and_unknown_session_errors() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (_client, session) = started_session(&dir.path().join("workspace")).await;

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("IFS= read -r line; printf '%s\\n' \"$line\"")
                .with_shell(ExecShell::Plain)
                .with_tty(true)
                .with_yield_time_ms(50),
        )
        .await
        .expect("start a pty session");
    let process_id = started.process_id.expect("a live process id");
    assert_eq!(started.exit_code, None);

    let written = session
        .pty_write_stdin(
            PtyWriteRequest::new(process_id, "hello from pty\n").with_yield_time_ms(1_000),
        )
        .await
        .expect("write to the pty");
    assert_eq!(written.process_id, None);
    assert_eq!(written.exit_code, Some(0));
    assert!(
        written.output_text().contains("hello from pty"),
        "{:?}",
        written.output_text()
    );

    for unknown in [process_id, 999_999] {
        let error = session
            .pty_write_stdin(PtyWriteRequest::new(unknown, ""))
            .await
            .expect_err("a finished or unknown session is not writable");
        assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    }
}

/// Python: `test_pty_ctrl_c_interrupts_long_running_process`.
#[tokio::test]
async fn pty_ctrl_c_interrupts_a_long_running_process() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (_client, session) = started_session(&dir.path().join("workspace")).await;

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("sleep 30")
                .with_shell(ExecShell::Plain)
                .with_tty(true)
                .with_yield_time_ms(50),
        )
        .await
        .expect("start a pty session");
    let process_id = started.process_id.expect("a live process id");
    assert_eq!(started.exit_code, None);

    let first = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "\u{3}").with_yield_time_ms(250))
        .await
        .expect("send ^C");
    let interrupted = if first.process_id.is_none() {
        first
    } else {
        session
            .pty_write_stdin(PtyWriteRequest::new(process_id, "").with_yield_time_ms(5_500))
            .await
            .expect("poll after ^C")
    };
    assert_eq!(interrupted.process_id, None);
    assert!(interrupted.exit_code.is_some());

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, ""))
        .await
        .expect_err("the interrupted session is gone");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
}

/// Python: `test_non_tty_pty_session_rejects_stdin_and_can_still_be_polled`.
#[tokio::test]
async fn a_non_tty_pty_session_rejects_stdin_and_can_still_be_polled() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (_client, session) = started_session(&dir.path().join("workspace")).await;

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("printf 'stdout\\n'; printf 'stderr\\n' >&2; sleep 1")
                .with_shell(ExecShell::Plain)
                .with_tty(false)
                .with_yield_time_ms(50),
        )
        .await
        .expect("start a non-tty session");
    let process_id = started.process_id.expect("a live process id");
    assert_eq!(started.exit_code, None);
    let started_text = started.output_text();
    assert!(started_text.contains("stdout"), "{started_text:?}");
    assert!(started_text.contains("stderr"), "{started_text:?}");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "hello"))
        .await
        .expect_err("a non-tty process has no stdin");
    assert_eq!(error.code, ErrorCode::PtyStdinUnavailable);

    let finished = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "").with_yield_time_ms(5_500))
        .await
        .expect("poll to completion");
    assert_eq!(finished.process_id, None);
    assert_eq!(finished.exit_code, Some(0));
    assert_eq!(finished.output_text(), "");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, ""))
        .await
        .expect_err("the finished session is gone");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
}

/// Python: `test_stop_terminates_active_pty_sessions`.
#[tokio::test]
async fn stop_terminates_active_pty_sessions() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (_client, session) = started_session(&dir.path().join("workspace")).await;

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("printf 'ready\\n'; sleep 30")
                .with_shell(ExecShell::Plain)
                .with_tty(true)
                .with_yield_time_ms(250),
        )
        .await
        .expect("start a pty session");
    let process_id = started.process_id.expect("a live process id");
    assert!(started.output_text().contains("ready"));

    session.stop().await.expect("stop");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, ""))
        .await
        .expect_err("stop terminates live pty sessions");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
}

// ---------------------------------------------------------------------------------------------
// apply_patch renames against a real filesystem — Python `TestUnixLocalApplyPatchRename`
// ---------------------------------------------------------------------------------------------

/// Python: `test_move_to_an_existing_directory_keeps_the_source`.
///
/// The destination write must fail before the editor removes the source.
#[tokio::test]
async fn apply_patch_move_to_an_existing_directory_keeps_the_source() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("notes.txt"), b"alpha\nbeta\n").expect("write notes.txt");
    std::fs::create_dir(workspace.join("docs")).expect("create docs");

    let editor = SandboxEditor::new(session(&workspace, &[]));
    editor
        .update_file(&update_op(
            "notes.txt",
            "@@\n alpha\n-beta\n+gamma\n",
            Some("docs"),
        ))
        .await
        .expect_err("a directory destination is refused");

    assert_eq!(
        std::fs::read(workspace.join("notes.txt")).expect("notes.txt survives"),
        b"alpha\nbeta\n"
    );
    assert_eq!(
        std::fs::read_dir(workspace.join("docs"))
            .expect("list docs")
            .count(),
        0
    );
}

/// Python: `test_case_only_move_to_keeps_the_file` and
/// `test_apply_patch_case_alias_on_real_filesystem`.
#[tokio::test]
async fn apply_patch_case_only_move_keeps_the_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("notes.txt"), b"alpha\nbeta\n").expect("write notes.txt");
    if !workspace.join("Notes.txt").exists() {
        // This volume does not fold case, so it cannot exercise the bug.
        return;
    }

    SandboxEditor::new(session(&workspace, &[]))
        .update_file(&update_op(
            "notes.txt",
            "@@\n alpha\n-beta\n+gamma\n",
            Some("Notes.txt"),
        ))
        .await
        .expect("a case-only rename");

    let names: Vec<String> = std::fs::read_dir(&workspace)
        .expect("list workspace")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec!["Notes.txt".to_string()]);
    assert_eq!(
        std::fs::read(workspace.join("Notes.txt")).expect("read Notes.txt"),
        b"alpha\ngamma\n"
    );
}

// ---------------------------------------------------------------------------------------------
// Path authority — Python `test_unix_local_file_io.py`
// ---------------------------------------------------------------------------------------------

/// Python: `test_safe_symlinks_grants_and_listing_paths_remain_supported`.
#[tokio::test]
async fn safe_symlinks_and_grants_remain_supported() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let allowed = dir.path().join("allowed");
    let readonly = dir.path().join("readonly");
    for path in [&workspace, &allowed, &readonly] {
        std::fs::create_dir(path).expect("create directory");
    }
    std::fs::write(readonly.join("file"), b"read only").expect("write readonly file");
    let alias = dir.path().join("alias");
    symlink(&workspace, &alias);

    let session = session(&alias, &[grant(&allowed), read_only_grant(&readonly)]);
    symlink(workspace.join("real"), workspace.join("internal"));
    symlink(&allowed, workspace.join("external"));

    session
        .write_file("internal/nested/file", b"bytes\x00\xff")
        .await
        .expect("write through a safe internal alias");
    assert_eq!(
        std::fs::read(workspace.join("real").join("nested").join("file")).expect("read it back"),
        b"bytes\x00\xff"
    );

    session
        .write_file("external/file", b"allowed")
        .await
        .expect("write into a granted directory");
    assert_eq!(
        session.read_file("external/file").await.expect("read back"),
        b"allowed"
    );
    assert_eq!(
        session
            .read_file(&readonly.join("file").to_string_lossy())
            .await
            .expect("read through a read-only grant"),
        b"read only"
    );
    let error = session
        .write_file(&readonly.join("file").to_string_lossy(), b"denied")
        .await
        .expect_err("a read-only grant refuses writes");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );

    let listed = session.ls("internal/nested").await.expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, "file");
    // The alias is resolved, so the entry reports where the file really is. The session keeps
    // its own root spelling (here the `alias` link) rather than Python's fully resolved path.
    assert_eq!(
        listed[0].path,
        alias
            .join("real")
            .join("nested")
            .join("file")
            .to_string_lossy()
    );

    session
        .rm("internal", true)
        .await
        .expect("remove through the alias");
    assert!(!workspace.join("real").exists());
    assert_eq!(
        std::fs::read(allowed.join("file")).expect("the granted file is untouched"),
        b"allowed"
    );
}

/// Python: `test_root_replacement_cannot_reauthorize_an_outside_directory`.
#[tokio::test]
async fn root_replacement_cannot_reauthorize_an_outside_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().join("workspace");
    std::fs::create_dir(&root).expect("create workspace");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&outside).expect("create outside");
    std::fs::write(outside.join("file"), b"original content").expect("write outside file");

    let session = session(&root, &[]);
    std::fs::rename(&root, dir.path().join("original")).expect("move the root aside");
    symlink(&outside, &root);

    let error = session
        .write_file("file", b"new")
        .await
        .expect_err("the replaced root authorizes nothing");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        std::fs::read(outside.join("file")).expect("the outside file is untouched"),
        b"original content"
    );
}

/// Python: `test_replacing_grant_alias_does_not_expand_authority`.
#[tokio::test]
async fn replacing_a_grant_alias_does_not_expand_authority() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let granted = dir.path().join("granted");
    let outside = dir.path().join("outside");
    for path in [&workspace, &granted, &outside] {
        std::fs::create_dir(path).expect("create directory");
    }
    std::fs::write(outside.join("file"), b"original content").expect("write outside file");
    let alias = dir.path().join("alias");
    symlink(&granted, &alias);

    let session = session(&workspace, &[grant(&alias)]);
    std::fs::remove_file(&alias).expect("drop the alias");
    symlink(&outside, &alias);

    let error = session
        .write_file(&alias.join("file").to_string_lossy(), b"new")
        .await
        .expect_err("a re-pointed alias does not widen the grant");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        std::fs::read(outside.join("file")).expect("the outside file is untouched"),
        b"original content"
    );
}

/// Python: `test_search_only_parent_does_not_require_directory_listing_permission`.
#[tokio::test]
async fn a_search_only_parent_does_not_require_listing_permission() {
    if is_root() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let parent = dir.path().join("parent");
    std::fs::create_dir(&parent).expect("create parent");
    std::fs::write(parent.join("file"), b"inside").expect("write file");
    let session = session(dir.path(), &[]);

    std::fs::set_permissions(&parent, PermissionsExt::from_mode(0o111)).expect("chmod parent");
    let read = session.read_file("parent/file").await;
    let written = session.write_file("parent/file", b"new").await;
    let made = session.mkdir("parent", false).await;
    std::fs::set_permissions(&parent, PermissionsExt::from_mode(0o700)).expect("restore parent");

    assert_eq!(read.expect("read through a search-only parent"), b"inside");
    written.expect("write through a search-only parent");
    made.expect("mkdir on an existing search-only directory");
    assert_eq!(
        std::fs::read(parent.join("file")).expect("read it back"),
        b"new"
    );
}

/// Python: `test_mkdir_existing_directory_does_not_require_search_permission`.
#[tokio::test]
async fn mkdir_on_an_existing_directory_does_not_require_search_permission() {
    if is_root() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let target = dir.path().join("locked");
    std::fs::create_dir(&target).expect("create locked");
    let session = session(dir.path(), &[]);

    std::fs::set_permissions(&target, PermissionsExt::from_mode(0o0)).expect("chmod locked");
    let result = session.mkdir("locked", false).await;
    std::fs::set_permissions(&target, PermissionsExt::from_mode(0o700)).expect("restore locked");

    result.expect("mkdir on an existing, unsearchable directory");
    assert!(target.is_dir());
}

/// Python: `test_rename_replaces_the_destination_entry`.
#[tokio::test]
async fn rename_replaces_the_destination_entry() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("a.txt"), b"from a").expect("write a.txt");
    std::fs::write(workspace.join("b.txt"), b"from b").expect("write b.txt");

    session(&workspace, &[])
        .mv("a.txt", "b.txt")
        .await
        .expect("rename over the destination");

    let mut names: Vec<String> = std::fs::read_dir(&workspace)
        .expect("list workspace")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["b.txt".to_string()]);
    assert_eq!(
        std::fs::read(workspace.join("b.txt")).expect("read b.txt"),
        b"from a"
    );
}

/// Python: `test_rename_onto_a_directory_fails_and_keeps_the_source`.
///
/// `mv` would put the source inside the directory and exit 0; a caller that then removes the
/// source would delete the file. `rename(2)` refuses instead.
#[tokio::test]
async fn rename_onto_a_directory_fails_and_keeps_the_source() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("notes.txt"), b"kept").expect("write notes.txt");
    std::fs::create_dir(workspace.join("docs")).expect("create docs");

    let error = session(&workspace, &[])
        .mv("notes.txt", "docs")
        .await
        .expect_err("a directory destination is refused");
    assert_eq!(error.code, ErrorCode::ExecNonZero);

    assert_eq!(
        std::fs::read(workspace.join("notes.txt")).expect("notes.txt survives"),
        b"kept"
    );
    assert_eq!(
        std::fs::read_dir(workspace.join("docs"))
            .expect("list docs")
            .count(),
        0
    );
}

/// Python: `test_rename_moves_a_symlink_entry_rather_than_its_target` and
/// `test_public_entry_operations_preserve_leaf_symlinks[None]`.
#[tokio::test]
async fn rename_moves_a_symlink_entry_rather_than_its_target() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let target = workspace.join("target");
    std::fs::write(&target, b"untouched").expect("write target");
    let link = workspace.join("link");
    symlink("target", &link);
    symlink(".", workspace.join("alias"));
    let session = session(&workspace, &[]);

    session
        .mv("alias/link", "moved")
        .await
        .expect("move the link through a parent alias");
    assert!(std::fs::symlink_metadata(&link).is_err());
    assert!(
        std::fs::symlink_metadata(workspace.join("moved"))
            .expect("stat moved")
            .file_type()
            .is_symlink()
    );
    assert_eq!(std::fs::read(&target).expect("read target"), b"untouched");

    std::fs::write(workspace.join("replacement"), b"replacement").expect("write replacement");
    session
        .mv("replacement", "moved")
        .await
        .expect("replace the link");
    assert!(
        !std::fs::symlink_metadata(workspace.join("moved"))
            .expect("stat moved")
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::read(workspace.join("moved")).expect("read moved"),
        b"replacement"
    );
    assert_eq!(std::fs::read(&target).expect("read target"), b"untouched");
}

/// Python: `test_entry_operations_honor_exact_file_grants` (minus the `same_file` line, which
/// the Rust session API does not expose).
#[tokio::test]
async fn entry_operations_honor_exact_file_grants() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::create_dir(&outside).expect("create outside");
    let source = outside.join("source");
    let destination = outside.join("destination");
    let readonly = outside.join("readonly");
    std::fs::write(&source, b"granted").expect("write source");
    std::fs::write(&readonly, b"read only").expect("write readonly");

    let session = session(
        &workspace,
        &[
            grant(&source),
            grant(&destination),
            read_only_grant(&readonly),
        ],
    );

    let error = session
        .mv(&source.to_string_lossy(), &readonly.to_string_lossy())
        .await
        .expect_err("a read-only grant is not a move destination");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);

    let error = session
        .mv(
            &source.to_string_lossy(),
            &outside.join("ungranted").to_string_lossy(),
        )
        .await
        .expect_err("an ungranted destination is refused");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);

    session
        .mv(&source.to_string_lossy(), &destination.to_string_lossy())
        .await
        .expect("move between exact file grants");
    assert_eq!(
        std::fs::read(&destination).expect("read destination"),
        b"granted"
    );
    assert_eq!(
        std::fs::read(&readonly).expect("read readonly"),
        b"read only"
    );
    assert!(!source.exists());
}

/// Python: `test_move_rejects_directory_sources_through_parent_aliases[None]`.
#[tokio::test]
async fn move_rejects_directory_sources_through_parent_aliases() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let shared = dir.path().join("shared");
    let protected = shared.join("tree").join("private");
    std::fs::create_dir_all(&protected).expect("create protected");
    let payload = protected.join("file");
    std::fs::write(&payload, b"protected").expect("write payload");
    symlink(&shared, workspace.join("alias"));

    let session = session(&workspace, &[grant(&shared), read_only_grant(&protected)]);
    let error = session
        .mv(
            &workspace.join("alias").join("tree").to_string_lossy(),
            &workspace.join("moved").to_string_lossy(),
        )
        .await
        .expect_err("a directory source is refused");
    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert!(
        error.message.contains("regular file or symlink"),
        "{}",
        error.message
    );

    assert_eq!(
        std::fs::read(&payload).expect("the payload is untouched"),
        b"protected"
    );
    assert!(!workspace.join("moved").exists());
}

// ---------------------------------------------------------------------------------------------
// apply_patch move destinations — Python `test_unix_local_file_io.py`
// ---------------------------------------------------------------------------------------------

/// Python: `test_apply_patch_preserves_existing_move_destination[0o755, 0o600]`.
#[tokio::test]
async fn apply_patch_preserves_an_existing_move_destination() {
    for mode in [0o755u32, 0o600] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        std::fs::create_dir(&workspace).expect("create workspace");
        let source = workspace.join("source");
        std::fs::write(&source, "before\n").expect("write source");
        let destination = workspace.join("destination");
        std::fs::write(&destination, "old destination\n").expect("write destination");
        std::fs::set_permissions(&destination, PermissionsExt::from_mode(mode)).expect("chmod");
        let before = std::fs::metadata(&destination).expect("stat destination");

        SandboxEditor::new(session(&workspace, &[]))
            .update_file(&update_op(
                "source",
                "@@\n-before\n+after\n",
                Some("destination"),
            ))
            .await
            .expect("update and move");

        let after = std::fs::metadata(&destination).expect("stat destination");
        assert_eq!(
            std::fs::read_to_string(&destination).expect("read destination"),
            "after\n"
        );
        assert_eq!(after.ino(), before.ino(), "mode {mode:o}");
        assert_eq!(after.mode(), before.mode(), "mode {mode:o}");
        assert!(!source.exists(), "mode {mode:o}");
    }
}

/// Python: `test_apply_patch_can_write_existing_destination_in_protected_directory`.
#[tokio::test]
async fn apply_patch_can_write_an_existing_destination_in_a_protected_directory() {
    if is_root() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let source = workspace.join("source");
    std::fs::write(&source, "before\n").expect("write source");
    let protected = workspace.join("protected");
    std::fs::create_dir(&protected).expect("create protected");
    let destination = protected.join("destination");
    std::fs::write(&destination, "old\n").expect("write destination");
    std::fs::set_permissions(&protected, PermissionsExt::from_mode(0o555)).expect("chmod");

    let result = SandboxEditor::new(session(&workspace, &[]))
        .update_file(&update_op(
            "source",
            "@@\n-before\n+after\n",
            Some("protected/destination"),
        ))
        .await;
    std::fs::set_permissions(&protected, PermissionsExt::from_mode(0o755)).expect("restore");

    result.expect("an existing destination stays writable in a read-only directory");
    assert_eq!(
        std::fs::read_to_string(&destination).expect("read destination"),
        "after\n"
    );
    assert!(!source.exists());
}

/// Python: `test_apply_patch_move_under_aliased_workspace_root` (minus the `same_file` lines).
#[tokio::test]
async fn apply_patch_moves_under_an_aliased_workspace_root() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let alias = dir.path().join("alias");
    symlink(&workspace, &alias);
    let source = workspace.join("source");
    std::fs::write(&source, "before\n").expect("write source");
    let session = session(&alias, &[]);

    SandboxEditor::new(session.clone())
        .update_file(&update_op(
            "source",
            "@@\n-before\n+after\n",
            Some("destination"),
        ))
        .await
        .expect("update and move under an aliased root");

    let destination = workspace.join("destination");
    assert_eq!(
        std::fs::read_to_string(&destination).expect("read destination"),
        "after\n"
    );
    assert!(!source.exists());

    session
        .mv(
            &destination
                .canonicalize()
                .expect("canonical")
                .to_string_lossy(),
            "renamed",
        )
        .await
        .expect("rename through the canonical spelling");
    assert_eq!(
        std::fs::read_to_string(workspace.join("renamed")).expect("read renamed"),
        "after\n"
    );
}

// ---------------------------------------------------------------------------------------------
// Operations as a user refuse before dispatching — Python `test_unix_local_user_file_io.py`
// ---------------------------------------------------------------------------------------------

/// Python: `test_ungranted_path_never_dispatches` (`ls`, `write`, `mv`; `same_file` has no Rust
/// equivalent).
///
/// `sudo` is never reached: the refusal comes from the path policy, so the test does not need a
/// privileged account.
#[tokio::test]
async fn an_ungranted_path_never_dispatches() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = session(&dir.path().join("workspace"), &[]);
    let user = User::new("example-user");

    let error = session.ls("/ungranted/file").await.expect_err("ls");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);

    let error = session
        .write_file_as("/ungranted/file", b"value", Some(&user))
        .await
        .expect_err("write");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);

    let error = session
        .mv_as("file", "/ungranted/file", Some(&user))
        .await
        .expect_err("mv");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
}

/// Python: `test_read_only_grant_rejects_user_write_before_dispatch`.
#[tokio::test]
async fn a_read_only_grant_rejects_a_user_write_before_dispatch() {
    let dir = tempfile::tempdir().expect("temp dir");
    let shared = dir.path().join("shared");
    std::fs::create_dir(&shared).expect("create shared");
    let session = session(&dir.path().join("workspace"), &[read_only_grant(&shared)]);

    let error = session
        .write_file_as(
            &shared.join("file").to_string_lossy(),
            b"value",
            Some(&User::new("example-user")),
        )
        .await
        .expect_err("a read-only grant refuses a user write");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
}

// ---------------------------------------------------------------------------------------------
// Special files — Python `test_unix_local_special_files.py`
// ---------------------------------------------------------------------------------------------

/// Python: `test_public_file_operations_reject_peerless_fifo` (`read`, `write`, `delete`; the
/// `user-write` case needs a real `sudo`).
///
/// Each operation is bounded by a timeout: a regression opens the FIFO and blocks forever, and
/// the point of the test is that it never gets there.
#[tokio::test]
async fn public_file_operations_reject_a_peerless_fifo() {
    for operation in ["read", "write", "delete"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let target = dir.path().join("pipe");
        make_fifo(&target);
        let session = session(dir.path(), &[]);

        let outcome = tokio::time::timeout(FIFO_TIMEOUT, async {
            match operation {
                "read" => session.read_file("pipe").await.map(|_| ()),
                "write" => session.write_file("pipe", b"payload").await,
                _ => SandboxEditor::new(session.clone())
                    .delete_file(&delete_op("pipe"))
                    .await
                    .map(|_| ())
                    .map_err(|error| {
                        openai_agents_sdk::sandbox::errors::SandboxError::new(
                            ErrorCode::WorkspaceArchiveReadError,
                            error.to_string(),
                        )
                    }),
            }
        })
        .await
        .unwrap_or_else(|_| panic!("`{operation}` blocked on a peerless FIFO"));

        let error = outcome.unwrap_err();
        assert!(
            matches!(
                error.code,
                ErrorCode::WorkspaceArchiveReadError | ErrorCode::WorkspaceArchiveWriteError
            ),
            "{operation}: unexpected code {:?} ({})",
            error.code,
            error.message
        );
        assert!(
            std::fs::symlink_metadata(&target)
                .expect("the FIFO is still there")
                .file_type()
                .is_fifo(),
            "{operation}"
        );
    }
}

/// Python: `test_stable_special_files_are_rejected_without_io_open` (`socket`, `device`).
#[tokio::test]
async fn stable_special_files_are_rejected() {
    // A Unix socket address is short enough only in a short directory on macOS.
    let short = tempfile::tempdir_in("/tmp").expect("short temp dir");
    let socket_path = short.path().join("s");
    let _listener =
        std::os::unix::net::UnixListener::bind(&socket_path).expect("bind a unix socket");

    for target in [socket_path.clone(), std::path::PathBuf::from("/dev/null")] {
        let session = session(short.path(), &[grant("/dev/null")]);
        let spelled = target.to_string_lossy().into_owned();
        let error = session
            .read_file(&spelled)
            .await
            .expect_err("a special file is not readable");
        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveReadError,
            "{spelled}"
        );
        let error = session
            .write_file(&spelled, b"payload")
            .await
            .expect_err("a special file is not writable");
        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{spelled}"
        );
    }
}

/// Python: `test_regular_write_preserves_inode_permissions_and_umask`.
#[tokio::test]
async fn a_regular_write_preserves_the_inode_permissions_and_umask() {
    if is_root() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let session = session(dir.path(), &[]);
    let target = dir.path().join("file");
    std::fs::write(&target, b"long original contents").expect("write target");
    let alias = dir.path().join("hard-link");
    std::fs::hard_link(&target, &alias).expect("hardlink");
    std::fs::set_permissions(&target, PermissionsExt::from_mode(0o200)).expect("chmod target");
    let before = std::fs::metadata(&target).expect("stat target");
    std::fs::set_permissions(dir.path(), PermissionsExt::from_mode(0o500)).expect("chmod dir");

    let written = session.write_file("file", b"new").await;

    std::fs::set_permissions(dir.path(), PermissionsExt::from_mode(0o700)).expect("restore dir");
    std::fs::set_permissions(&target, PermissionsExt::from_mode(0o600)).expect("restore target");
    written.expect("an in-place write into a read-only directory");

    let after = std::fs::metadata(&target).expect("stat target");
    assert_eq!(after.ino(), before.ino());
    assert_eq!(after.uid(), before.uid());
    assert_eq!(after.gid(), before.gid());
    assert_eq!(std::fs::read(&alias).expect("read the hardlink"), b"new");

    let _guard = UMASK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the umask is process-wide; every test in this binary that changes it holds the
    // lock above, and the previous value is restored before the lock is released.
    let previous = unsafe { libc::umask(0o077) };
    let written = session.write_file("new", b"private").await;
    unsafe { libc::umask(previous) };
    written.expect("write a new file");
    assert_eq!(
        std::fs::metadata(dir.path().join("new"))
            .expect("stat new")
            .permissions()
            .mode()
            & 0o777,
        0o600,
        "a new file honours the process umask"
    );
}

// ---------------------------------------------------------------------------------------------
// Add File — Python `test_unix_local_create.py`
// ---------------------------------------------------------------------------------------------

/// Python: `test_create_writes_nested_file_and_rejects_dangling_leaf` (the `direct` fixture
/// parameter; the `bound-user` one executes Python's own worker body).
#[tokio::test]
async fn create_writes_a_nested_file_and_rejects_a_dangling_leaf() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir(dir.path().join("real")).expect("create real");
    symlink(dir.path().join("real"), dir.path().join("alias"));
    let editor = SandboxEditor::new(session(dir.path(), &[]));

    let result = editor
        .create_file(&create_op("alias/nested/new.txt", "+contents\n"))
        .await
        .expect("create a nested file through an alias");
    assert_eq!(
        result.output.as_deref(),
        Some("Created alias/nested/new.txt")
    );
    assert_eq!(
        std::fs::read(dir.path().join("real").join("nested").join("new.txt"))
            .expect("read new.txt"),
        b"contents"
    );

    symlink(
        dir.path().join("missing.txt"),
        dir.path().join("alias").join("link.txt"),
    );
    let error = editor
        .create_file(&create_op("alias/link.txt", "+contents\n"))
        .await
        .expect_err("a dangling leaf is not a free name");
    assert!(format!("{error}").contains("already exists"), "{error}");
    assert!(!dir.path().join("missing.txt").exists());
}

// ---------------------------------------------------------------------------------------------
// `rm` — Python `UnixLocalSandboxSession.rm`
// ---------------------------------------------------------------------------------------------

/// A non-recursive `rm` of a missing path is an error; a recursive one succeeds.
#[tokio::test]
async fn rm_of_a_missing_path_errors_only_when_not_recursive() {
    let dir = tempfile::tempdir().expect("temp dir");
    let session = session(dir.path(), &[]);

    let error = session
        .rm("missing", false)
        .await
        .expect_err("a plain rm of nothing fails");
    assert_eq!(error.code, ErrorCode::ExecNonZero);
    session
        .rm("missing", true)
        .await
        .expect("a recursive rm of nothing succeeds");
}

/// A non-recursive `rm` of a directory is `rmdir`: an empty one goes, a populated one is refused
/// and left alone.
#[tokio::test]
async fn non_recursive_rm_removes_an_empty_directory_and_refuses_a_populated_one() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir(dir.path().join("empty")).expect("create empty");
    std::fs::create_dir(dir.path().join("full")).expect("create full");
    std::fs::write(dir.path().join("full").join("kept"), b"kept").expect("write kept");
    let session = session(dir.path(), &[]);

    session
        .rm("empty", false)
        .await
        .expect("rmdir an empty directory");
    assert!(!dir.path().join("empty").exists());

    let error = session
        .rm("full", false)
        .await
        .expect_err("a populated directory is refused");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        std::fs::read(dir.path().join("full").join("kept")).expect("the file survives"),
        b"kept"
    );
}
