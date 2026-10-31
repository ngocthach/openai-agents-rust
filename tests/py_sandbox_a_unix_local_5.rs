//! Check-then-use path races on the Unix-local file API.
//!
//! Ports `test_parent_swap_after_validation_cannot_access_outside` and
//! `test_leaf_swap_does_not_follow_new_symlink` from `tests/sandbox/test_unix_local_file_io.py`
//! for `read`, `write` and exclusive create. Python injects the swap by monkeypatching
//! `normalize_path`; here the session's `with_after_validate_hook` runs between validation and the
//! open, which is the same window.
//!
//! Not ported: `mkdir`, `ls`, `rm`, `mv` and `same_file` (path-based after validation here, see the
//! report for the residual window) and the FIFO-replacement variants (a path swapped to a FIFO
//! after the entry check blocks the open, which needs `O_NONBLOCK`, unavailable without `libc`).

#![cfg(all(unix, feature = "sandbox"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxSession;
use openai_agents_sdk::sandbox::{Manifest, SandboxSession};

const ORIGINAL: &[u8] = b"original content";

fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(target.as_ref(), link.as_ref()).expect("create a symlink");
}

/// A session over `workspace` whose hook runs `swap` once, the first time `leaf` is about to be
/// opened.
fn session_swapping(
    workspace: &Path,
    leaf: PathBuf,
    swap: impl Fn() + Send + Sync + 'static,
) -> (UnixLocalSandboxSession, Arc<AtomicBool>) {
    let swapped = Arc::new(AtomicBool::new(false));
    let flag = swapped.clone();
    let session = UnixLocalSandboxSession::new(
        Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        false,
    )
    .expect("a session")
    .with_after_validate_hook(Arc::new(move |host: &Path| {
        if host.ends_with(&leaf) && !flag.swap(true, Ordering::SeqCst) {
            swap();
        }
    }));
    (session, swapped)
}

/// Lay out `workspace/parent/target` and `outside/target`, both holding `ORIGINAL`.
fn layout(base: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let workspace = base.join("workspace");
    let parent = workspace.join("parent");
    let outside = base.join("outside");
    std::fs::create_dir_all(&parent).expect("create parent");
    std::fs::create_dir(&outside).expect("create outside");
    std::fs::write(parent.join("target"), ORIGINAL).expect("write inside");
    std::fs::write(outside.join("target"), ORIGINAL).expect("write outside");
    (workspace, parent, outside)
}

/// Replace `parent` with a symlink to `outside`, as a workspace process could.
fn swap_parent(workspace: &Path, outside: &Path) -> impl Fn() + Send + Sync + 'static {
    let parent = workspace.join("parent");
    let moved = workspace.join("original");
    let outside = outside.to_path_buf();
    move || {
        std::fs::rename(&parent, &moved).expect("move the parent aside");
        symlink(&outside, &parent);
    }
}

/// Python: `test_parent_swap_after_validation_cannot_access_outside[read]`.
#[tokio::test]
async fn a_parent_swapped_after_validation_cannot_be_read_through() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, _parent, outside) = layout(dir.path());
    let (session, swapped) = session_swapping(
        &workspace,
        PathBuf::from("parent/target"),
        swap_parent(&workspace, &outside),
    );

    let error = session
        .read_file("parent/target")
        .await
        .expect_err("a read that followed the swap out of the workspace is refused");

    assert!(swapped.load(Ordering::SeqCst), "the swap happened");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        std::fs::read(outside.join("target")).expect("read outside"),
        ORIGINAL
    );
}

/// Python: `test_parent_swap_after_validation_cannot_access_outside[write]`.
///
/// Nothing is truncated or written before the handle is shown to be inside the workspace.
#[tokio::test]
async fn a_parent_swapped_after_validation_cannot_be_written_through() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, _parent, outside) = layout(dir.path());
    let (session, swapped) = session_swapping(
        &workspace,
        PathBuf::from("parent/target"),
        swap_parent(&workspace, &outside),
    );

    let error = session
        .write_file("parent/target", b"new")
        .await
        .expect_err("a write that followed the swap out of the workspace is refused");

    assert!(swapped.load(Ordering::SeqCst));
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        std::fs::read(outside.join("target")).expect("read outside"),
        ORIGINAL,
        "the outside file was not truncated or rewritten"
    );
}

/// A write that would have *created* a file through the swapped parent leaves nothing behind.
#[tokio::test]
async fn a_write_that_created_a_file_outside_removes_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, _parent, outside) = layout(dir.path());
    let (session, _) = session_swapping(
        &workspace,
        PathBuf::from("parent/fresh"),
        swap_parent(&workspace, &outside),
    );

    session
        .write_file("parent/fresh", b"new")
        .await
        .expect_err("refused");

    assert!(!outside.join("fresh").exists(), "no file was left outside");
}

/// Python: `test_parent_swap_after_validation_cannot_access_outside[patch]` (the exclusive
/// create).
#[tokio::test]
async fn a_parent_swapped_after_validation_cannot_be_created_through() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, _parent, outside) = layout(dir.path());
    let (session, swapped) = session_swapping(
        &workspace,
        PathBuf::from("parent/fresh"),
        swap_parent(&workspace, &outside),
    );

    let error = session
        .write_new_file("parent/fresh", b"payload", None)
        .await
        .expect_err("an exclusive create through the swapped parent is refused");

    assert!(swapped.load(Ordering::SeqCst));
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(
        !outside.join("fresh").exists(),
        "the file created outside is removed, and no payload reached it"
    );
}

/// Replace the leaf `workspace/target` with a symlink to `outside/target`.
fn swap_leaf(workspace: &Path, outside: &Path) -> impl Fn() + Send + Sync + 'static {
    let leaf = workspace.join("target");
    let moved = workspace.join("original");
    let outside = outside.join("target");
    move || {
        std::fs::rename(&leaf, &moved).expect("move the leaf aside");
        symlink(&outside, &leaf);
    }
}

fn leaf_layout(base: &Path) -> (PathBuf, PathBuf) {
    let workspace = base.join("workspace");
    let outside = base.join("outside");
    std::fs::create_dir_all(&workspace).expect("create workspace");
    std::fs::create_dir(&outside).expect("create outside");
    std::fs::write(workspace.join("target"), ORIGINAL).expect("write inside");
    std::fs::write(outside.join("target"), ORIGINAL).expect("write outside");
    (workspace, outside)
}

/// Python: `test_leaf_swap_does_not_follow_new_symlink[read]`.
#[tokio::test]
async fn a_leaf_swapped_for_a_symlink_is_not_followed_on_read() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, outside) = leaf_layout(dir.path());
    let (session, swapped) = session_swapping(
        &workspace,
        PathBuf::from("target"),
        swap_leaf(&workspace, &outside),
    );

    session
        .read_file("target")
        .await
        .expect_err("a leaf swapped for an outside link is refused");

    assert!(swapped.load(Ordering::SeqCst));
    assert_eq!(
        std::fs::read(outside.join("target")).expect("read outside"),
        ORIGINAL
    );
}

/// Python: `test_leaf_swap_does_not_follow_new_symlink[write]`.
#[tokio::test]
async fn a_leaf_swapped_for_a_symlink_is_not_followed_on_write() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, outside) = leaf_layout(dir.path());
    let (session, swapped) = session_swapping(
        &workspace,
        PathBuf::from("target"),
        swap_leaf(&workspace, &outside),
    );

    session
        .write_file("target", b"new")
        .await
        .expect_err("a leaf swapped for an outside link is refused");

    assert!(swapped.load(Ordering::SeqCst));
    assert_eq!(
        std::fs::read(outside.join("target")).expect("read outside"),
        ORIGINAL,
        "the outside file was not truncated or rewritten"
    );
}

/// Without any swap the same operations succeed, so the checks are not simply refusing
/// everything.
#[tokio::test]
async fn the_same_operations_succeed_when_nothing_is_swapped() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (workspace, _parent, _outside) = layout(dir.path());
    let (session, swapped) = session_swapping(&workspace, PathBuf::from("parent/target"), || {});

    assert_eq!(
        session.read_file("parent/target").await.expect("read"),
        ORIGINAL
    );
    session
        .write_file("parent/target", b"updated")
        .await
        .expect("write");
    session
        .write_new_file("parent/fresh", b"payload", None)
        .await
        .expect("exclusive create");

    assert!(swapped.load(Ordering::SeqCst), "the hook ran");
    assert_eq!(
        std::fs::read(workspace.join("parent").join("target")).expect("read back"),
        b"updated"
    );
    assert_eq!(
        std::fs::read(workspace.join("parent").join("fresh")).expect("read back"),
        b"payload"
    );
}
