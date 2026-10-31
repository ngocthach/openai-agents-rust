//! Ports of the tar-validation, persist, hydrate and skip-path groups of
//! `tests/extensions/sandbox/test_blaxel.py`.
//!
//! The workspace archive travels through a scratch tar file in the sandbox: `tar` runs there over
//! the process API and the file moves over the filesystem API, so each case is asserted against
//! the shared mock sandbox, which records every request.
#![cfg(feature = "sandbox")]

#[path = "py_core6_blaxel_support/mod.rs"]
mod harness;
#[path = "py_sandbox_support/mod.rs"]
mod support;

use harness::{
    World, context, count_prefixed, multipart_file, options, root_manifest, session, session_with,
    start,
};
use openai_agents_sdk::sandbox::archive::{TarValidationOptions, validate_tar_bytes};
use openai_agents_sdk::sandbox::{ErrorCode, SandboxSession};
use support::{MockServer, Shared};

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::session::with_ephemeral_mounts_removed;

// ---------------------------------------------------------------------------------------------
// Tar validation, persist and hydrate
// ---------------------------------------------------------------------------------------------

const TYPE_FILE: u8 = b'0';
const TYPE_HARDLINK: u8 = b'1';
const TYPE_SYMLINK: u8 = b'2';
const TYPE_CHAR_DEVICE: u8 = b'3';
const TYPE_DIR: u8 = b'5';

fn tar_header(name: &str, size: u64, type_flag: u8, link_name: &str) -> [u8; 512] {
    fn put(header: &mut [u8; 512], offset: usize, bytes: &[u8]) {
        header[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
    let mut header = [0u8; 512];
    put(&mut header, 0, name.as_bytes());
    put(&mut header, 100, b"0000644\0");
    put(&mut header, 108, b"0000000\0");
    put(&mut header, 116, b"0000000\0");
    put(&mut header, 124, format!("{size:011o}\0").as_bytes());
    put(&mut header, 136, b"00000000000\0");
    put(&mut header, 148, b"        ");
    header[156] = type_flag;
    put(&mut header, 157, link_name.as_bytes());
    put(&mut header, 257, b"ustar\0");
    put(&mut header, 263, b"00");
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    put(&mut header, 148, format!("{checksum:06o}\0 ").as_bytes());
    header
}

fn tar_member(name: &str, type_flag: u8, link_name: &str, contents: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tar_header(
        name,
        contents.len() as u64,
        type_flag,
        link_name,
    ));
    out.extend_from_slice(contents);
    out.extend(std::iter::repeat_n(0u8, (512 - contents.len() % 512) % 512));
    out
}

fn tar_archive(members: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out: Vec<u8> = members.into_iter().flatten().collect();
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}

fn file_member(name: &str, contents: &[u8]) -> Vec<u8> {
    tar_member(name, TYPE_FILE, "", contents)
}

fn lenient() -> TarValidationOptions {
    TarValidationOptions {
        allow_symlinks: true,
        allow_external_symlink_targets: true,
        ..TarValidationOptions::default()
    }
}

fn symlink_then_nested_file() -> Vec<u8> {
    tar_archive(vec![
        tar_member("link.txt", TYPE_SYMLINK, "/etc/passwd", b""),
        file_member("link.txt/nested.txt", b"nested"),
    ])
}

/// `TestValidateTarBytes`: `test_valid_tar`, `test_absolute_path_rejected`,
/// `test_parent_traversal_rejected`, `test_tar_member_under_archive_symlink_rejected`,
/// `test_corrupt_tar_rejected` and `test_dot_entries_skipped`.
#[test]
fn tar_validation_accepts_safe_archives_and_names_each_refusal() {
    validate_tar_bytes(
        &tar_archive(vec![
            file_member("hello.txt", b"content"),
            tar_member("subdir/", TYPE_DIR, "", b""),
        ]),
        &lenient(),
    )
    .expect("a valid tar");
    validate_tar_bytes(
        &tar_archive(vec![
            tar_member("./", TYPE_DIR, "", b""),
            file_member("file.txt", b"ok"),
        ]),
        &lenient(),
    )
    .expect("dot entries are skipped");

    let cases: Vec<(Vec<u8>, &str)> = vec![
        (
            tar_archive(vec![file_member("/etc/passwd", b"root")]),
            "absolute path",
        ),
        (
            tar_archive(vec![file_member("../escape.txt", b"data")]),
            "parent traversal",
        ),
        (symlink_then_nested_file(), "descends through symlink"),
        (b"not a tar file at all".to_vec(), "invalid tar"),
        (
            tar_archive(vec![tar_member("device", TYPE_CHAR_DEVICE, "", b"")]),
            "unsupported member type",
        ),
        (
            tar_archive(vec![tar_member("hardlink", TYPE_HARDLINK, "target", b"")]),
            "hardlink",
        ),
    ];
    for (archive, expected) in cases {
        let error = validate_tar_bytes(&archive, &lenient()).expect_err(expected);
        assert!(
            error.message.contains(expected),
            "{expected}: {}",
            error.message
        );
    }
}

fn scratch_tar_path(server: &MockServer, kind: &str) -> String {
    let _ = server;
    format!("/tmp/openai-agents-sandbox-blaxel-{kind}-agents-test.tar")
}

/// `test_persist_workspace` and `test_persist_workspace_read_returns_str`: the archive is made
/// with `tar`, read back, and the scratch file is removed.
#[tokio::test]
async fn persist_archives_the_workspace_and_cleans_up() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    let tar = tar_archive(vec![file_member("file.txt", b"hello")]);
    world.with(|w| {
        w.files
            .insert(scratch_tar_path(&server, "persist"), tar.clone())
    });

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, tar);
    let command = server.find("POST", "/process").json()["command"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(command.starts_with("sh -lc 'tar "), "{command}");
    assert!(command.contains("-C /workspace -cf /tmp/openai-agents-sandbox-blaxel-persist"));
    assert_eq!(
        count_prefixed(&server, "DELETE", "/filesystem/"),
        1,
        "the scratch archive is removed"
    );
    server.stop();
}

/// `test_persist_workspace_tar_fails`
#[tokio::test]
async fn persist_reports_a_failing_tar() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| {
        w.exec_queue
            .push_back((1, String::new(), "tar: error".into()))
    });

    let error = session.persist_workspace().await.expect_err("tar fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(context(&error, "reason").as_deref(), Some("tar_failed"));
    assert_eq!(context(&error, "output").as_deref(), Some("tar: error"));
    assert_eq!(error.retryable, Some(false));
    // The cleanup still ran.
    assert_eq!(count_prefixed(&server, "DELETE", "/filesystem/"), 1);
    server.stop();
}

/// `test_persist_workspace_read_fails`: the tar succeeds but its file cannot be read back.
#[tokio::test]
async fn persist_reports_an_unreadable_archive() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    let error = session.persist_workspace().await.expect_err("no archive");

    assert!(
        matches!(
            error.code,
            ErrorCode::WorkspaceArchiveReadError | ErrorCode::WorkspaceReadNotFound
        ),
        "{error:?}"
    );
    server.stop();
}

/// `test_persist_cleanup_rm_failure_suppressed` and `test_persist_rm_exception_suppressed`
#[tokio::test]
async fn persist_ignores_a_failing_cleanup() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    let tar = tar_archive(vec![file_member("file.txt", b"hello")]);
    world.with(|w| {
        w.files
            .insert(scratch_tar_path(&server, "persist"), tar.clone());
        w.fs_delete_status = Some(500);
    });

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, tar);
    server.stop();
}

/// `test_hydrate_workspace`: the archive is uploaded, extracted into the root, and removed.
#[tokio::test]
async fn hydrate_uploads_extracts_and_cleans_up() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    let tar = tar_archive(vec![file_member("file.txt", b"hello")]);

    session.hydrate_workspace(&tar).await.expect("hydrate");

    let upload_path = scratch_tar_path(&server, "hydrate");
    let uploaded = server
        .requests()
        .into_iter()
        .find(|request| request.method == "PUT" && request.path.contains("blaxel-hydrate"))
        .expect("the archive upload");
    assert_eq!(multipart_file(&uploaded), tar);
    let extract = server
        .requests()
        .into_iter()
        .rfind(|request| request.path == "/process")
        .unwrap()
        .json();
    assert_eq!(
        extract["command"],
        format!("sh -lc 'tar -C /workspace -xf {upload_path}'")
    );
    assert_eq!(count_prefixed(&server, "DELETE", "/filesystem/"), 1);
    server.stop();
}

/// `test_hydrate_invalid_tar` and `test_hydrate_str_payload_encoded`
#[tokio::test]
async fn hydrate_refuses_an_invalid_archive() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    let error = session
        .hydrate_workspace(b"not a tar")
        .await
        .expect_err("invalid tar");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        context(&error, "reason").as_deref(),
        Some("unsafe_or_invalid_tar")
    );
    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 0);
    server.stop();
}

/// `test_hydrate_tar_with_symlink`: a member that descends through an archive symlink is refused
/// before anything is uploaded.
#[tokio::test]
async fn hydrate_refuses_a_member_under_an_archive_symlink() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;
    let raw = tar_archive(vec![
        tar_member("link.txt", TYPE_SYMLINK, "/etc/shadow", b""),
        file_member("link.txt/nested.txt", b"nested"),
    ]);

    let error = session
        .hydrate_workspace(&raw)
        .await
        .expect_err("unsafe tar");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        context(&error, "reason").as_deref(),
        Some("unsafe_or_invalid_tar")
    );
    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 0);
    server.stop();
}

/// `test_hydrate_extract_fails`
#[tokio::test]
async fn hydrate_reports_a_failing_extract() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| {
        w.exec_queue
            .push_back((1, String::new(), "tar: extract error".into()))
    });

    let error = session
        .hydrate_workspace(&tar_archive(vec![file_member("file.txt", b"hello")]))
        .await
        .expect_err("extract fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        context(&error, "reason").as_deref(),
        Some("tar_extract_failed")
    );
    assert_eq!(
        context(&error, "output").as_deref(),
        Some("tar: extract error")
    );
    assert_eq!(count_prefixed(&server, "DELETE", "/filesystem/"), 1);
    server.stop();
}

/// `test_hydrate_write_binary_fails`
#[tokio::test]
async fn hydrate_reports_a_failing_upload() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.fs_write_status = Some(500));

    let error = session
        .hydrate_workspace(&tar_archive(vec![file_member("file.txt", b"hello")]))
        .await
        .expect_err("upload fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    server.stop();
}

/// `test_hydrate_cleanup_rm_failure_suppressed` and `test_hydrate_rm_exception_suppressed`
#[tokio::test]
async fn hydrate_ignores_a_failing_cleanup() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.fs_delete_status = Some(500));

    session
        .hydrate_workspace(&tar_archive(vec![file_member("file.txt", b"hello")]))
        .await
        .expect("hydrate succeeds despite the cleanup failure");
    server.stop();
}

/// `test_exclude_args_with_skip_paths`: an ephemeral entry and a registered skip path are both
/// excluded from the archive; `test_exclude_args_skips_empty_and_dot`: `.` never is.
#[tokio::test]
async fn persist_excludes_skip_paths_but_never_dot() {
    use openai_agents_sdk::sandbox::entries::Entry;

    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let manifest = root_manifest()
        .with_entry("node_modules", Entry::dir().with_ephemeral(true))
        .with_entry(".git", Entry::dir().with_ephemeral(true));
    let session = session_with(&server, options(&server.base_url), manifest).await;
    session
        .register_persist_workspace_skip_path("keep_me")
        .expect("register");
    world.with(|w| {
        w.files.insert(
            scratch_tar_path(&server, "persist"),
            tar_archive(vec![file_member("a", b"a")]),
        )
    });

    session.persist_workspace().await.expect("persist");

    let command = server.find("POST", "/process").json()["command"]
        .as_str()
        .unwrap()
        .to_string();
    for expected in [
        "--exclude ./node_modules",
        "--exclude ./.git",
        "--exclude ./keep_me",
    ] {
        assert!(command.contains(expected), "{expected} in {command}");
    }
    assert!(!command.contains("--exclude ./. "), "{command}");
    assert!(!command.contains("--exclude '.'"), "{command}");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// TestPersistWithMounts
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct RecordingStrategy {
    events: Arc<Mutex<Vec<String>>>,
    fail_teardown: bool,
    fail_restore: bool,
}

#[async_trait]
impl MountStrategy for RecordingStrategy {
    fn strategy_type(&self) -> &str {
        "recording"
    }

    async fn activate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }

    async fn deactivate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }

    async fn teardown_for_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        _: &str,
    ) -> SandboxResult<()> {
        self.events.lock().unwrap().push("teardown".into());
        if self.fail_teardown {
            return Err(SandboxError::new(ErrorCode::MountFailed, "unmount fail"));
        }
        Ok(())
    }

    async fn restore_after_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        _: &str,
    ) -> SandboxResult<()> {
        self.events.lock().unwrap().push("restore".into());
        if self.fail_restore {
            return Err(SandboxError::new(ErrorCode::MountFailed, "remount fail"));
        }
        Ok(())
    }
}

/// Persist a Blaxel workspace with one ephemeral mount around the archive, as the snapshot path
/// does, and return the outcome with the mount's events.
async fn persist_around_a_mount(
    world: Shared<World>,
    fail_teardown: bool,
    fail_restore: bool,
) -> (SandboxResultVec, Vec<String>) {
    let server = start(world).await;
    let session = session(&server).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let mount = Mount::new(Arc::new(RecordingStrategy {
        events: events.clone(),
        fail_teardown,
        fail_restore,
    }));

    let outcome = with_ephemeral_mounts_removed(
        &*session,
        vec![(&mount, "/workspace/mount".to_string())],
        || session.persist_workspace(),
        "/workspace",
        ErrorCode::WorkspaceArchiveReadError,
        None,
        true,
    )
    .await;
    server.stop();
    let events = events.lock().unwrap().clone();
    (outcome, events)
}

type SandboxResultVec = SandboxResult<Vec<u8>>;

/// `test_persist_unmount_error`: a mount that will not detach fails the persist as an archive
/// read error, and the archive is never attempted.
#[tokio::test]
async fn persist_fails_when_a_mount_will_not_detach() {
    let (outcome, events) =
        persist_around_a_mount(Shared::new(World::default()), true, false).await;

    assert_eq!(
        outcome.expect_err("unmount fails").code,
        ErrorCode::WorkspaceArchiveReadError
    );
    assert_eq!(events, vec!["teardown"]);
}

/// `test_persist_remount_error`: the archive is taken, but a mount that will not come back is
/// an archive read error.
#[tokio::test]
async fn persist_fails_when_a_mount_will_not_come_back() {
    let world = Shared::new(World::default());
    world.with(|w| {
        w.files.insert(
            "/tmp/openai-agents-sandbox-blaxel-persist-agents-test.tar".into(),
            tar_archive(vec![file_member("file.txt", b"data")]),
        )
    });

    let (outcome, events) = persist_around_a_mount(world, false, true).await;

    assert_eq!(
        outcome.expect_err("remount fails").code,
        ErrorCode::WorkspaceArchiveReadError
    );
    assert_eq!(events, vec!["teardown", "restore"]);
}

/// `test_persist_snapshot_error_still_remounts`: a failing `tar` still re-attaches the mount,
/// exactly once, before the error comes back.
#[tokio::test]
async fn persist_remounts_after_a_failing_archive() {
    let world = Shared::new(World::default());
    world.with(|w| {
        w.exec_queue
            .push_back((1, String::new(), "tar fail".into()))
    });

    let (outcome, events) = persist_around_a_mount(world, false, false).await;

    assert_eq!(
        outcome.expect_err("tar fails").code,
        ErrorCode::WorkspaceArchiveReadError
    );
    assert_eq!(events, vec!["teardown", "restore"]);
}
