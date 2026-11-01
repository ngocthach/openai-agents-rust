//! Races between pinning an entry's parent and acting on the entry, for `rm`, `rmtree`, `mv` and
//! `same_file` (`tests/sandbox/test_unix_local_file_io.py::test_parent_swap_after_validation_...`
//! and `::test_leaf_swap_does_not_follow_new_symlink`).
//!
//! Python injects the swap by monkeypatching `normalize_path`, which is *before* its descriptor
//! walk. `py_core6_modelsrt_unix_local_swaps.rs` ports that window with the after-validate hook.
//! These tests use the after-pin hook, the later window: the operation already holds the
//! directory that contains its entry, and replacing the parent or the leaf with a link at that
//! moment must not redirect it.
#![cfg(all(unix, feature = "sandbox"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxSession;
use openai_agents_sdk::sandbox::{Manifest, SandboxSession};

const ORIGINAL: &[u8] = b"original content";

fn symlink(to: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(to.as_ref(), link.as_ref()).expect("create a symlink");
}

fn session_swapping_after_pin(
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
    .with_after_pin_hook(Arc::new(move |host: &Path| {
        if host.ends_with(&leaf) && !flag.swap(true, Ordering::SeqCst) {
            swap();
        }
    }));
    (session, swapped)
}

fn make_entry(path: &Path, directory: bool) {
    if directory {
        std::fs::create_dir(path).expect("create a directory");
        std::fs::write(path.join("sentinel"), ORIGINAL).expect("write the sentinel");
    } else {
        std::fs::write(path, ORIGINAL).expect("write a file");
    }
}

fn sentinel(path: &Path, directory: bool) -> PathBuf {
    if directory {
        path.join("sentinel")
    } else {
        path.to_path_buf()
    }
}

async fn operate(session: &UnixLocalSandboxSession, operation: &str, path: &str) -> bool {
    match operation {
        "rm" => session.rm(path, false).await.is_ok(),
        "rmtree" => session.rm(path, true).await.is_ok(),
        "mv" => session.mv(path, "moved").await.is_ok(),
        "same_file" => session.same_file(path, path, false, None).await.is_ok(),
        other => panic!("unknown operation {other}"),
    }
}

/// Python: `test_parent_swap_after_validation_cannot_access_outside` for `rm`, `rmtree`, `mv`
/// and `same_file`, with the swap landing after the parent was pinned. The operation acts on the
/// directory it already holds (now named `original`), and nothing outside is touched.
#[tokio::test]
async fn a_parent_swapped_after_it_was_pinned_cannot_redirect_the_operation() {
    for operation in ["rm", "rmtree", "mv", "same_file"] {
        let directory = operation == "rmtree";
        let dir = tempfile::tempdir().expect("temp dir");
        let base = std::fs::canonicalize(dir.path()).expect("canonical");
        let workspace = base.join("workspace");
        let parent = workspace.join("parent");
        let outside = base.join("outside");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::create_dir(&outside).unwrap();
        make_entry(&parent.join("target"), directory);
        make_entry(&outside.join("target"), directory);
        let (hook_parent, hook_moved, hook_outside) =
            (parent.clone(), workspace.join("original"), outside.clone());
        let (session, swapped) =
            session_swapping_after_pin(&workspace, PathBuf::from("target"), move || {
                std::fs::rename(&hook_parent, &hook_moved).expect("move the parent aside");
                symlink(&hook_outside, &hook_parent);
            });

        operate(&session, operation, "parent/target").await;

        assert!(swapped.load(Ordering::SeqCst), "{operation}: the hook ran");
        assert_eq!(
            std::fs::read(sentinel(&outside.join("target"), directory)).unwrap(),
            ORIGINAL,
            "{operation}: the outside entry is untouched"
        );
        assert!(
            std::fs::symlink_metadata(outside.join("target")).is_ok(),
            "{operation}: the outside entry is still there"
        );
        match operation {
            "rm" | "rmtree" => assert!(
                std::fs::symlink_metadata(workspace.join("original/target")).is_err(),
                "{operation}: the pinned directory's entry was the one removed"
            ),
            "mv" => {
                assert_eq!(
                    std::fs::read(workspace.join("moved")).unwrap(),
                    ORIGINAL,
                    "{operation}: the pinned directory's file was the one moved"
                );
                assert!(
                    std::fs::symlink_metadata(workspace.join("original/target")).is_err(),
                    "{operation}: it left the pinned directory"
                );
            }
            _ => {}
        }
    }
}

/// Python: `test_leaf_swap_does_not_follow_new_symlink` for `rm`, `rmtree`, `mv` and `same_file`,
/// with the leaf replaced by a link to an outside entry after the parent was pinned. The link is
/// removed, moved or compared as a link; the outside entry is never followed.
#[tokio::test]
async fn a_leaf_swapped_for_a_link_after_pinning_is_never_followed() {
    for operation in ["rm", "rmtree", "mv", "same_file"] {
        let directory = operation == "rmtree";
        let dir = tempfile::tempdir().expect("temp dir");
        let base = std::fs::canonicalize(dir.path()).expect("canonical");
        let workspace = base.join("workspace");
        let outside = base.join("outside");
        std::fs::create_dir(&workspace).unwrap();
        make_entry(&workspace.join("target"), directory);
        make_entry(&outside, directory);
        let (hook_target, hook_moved, hook_outside) = (
            workspace.join("target"),
            workspace.join("original"),
            outside.clone(),
        );
        let (session, swapped) =
            session_swapping_after_pin(&workspace, PathBuf::from("target"), move || {
                std::fs::rename(&hook_target, &hook_moved).expect("move the leaf aside");
                symlink(&hook_outside, &hook_target);
            });

        operate(&session, operation, "target").await;

        assert!(swapped.load(Ordering::SeqCst), "{operation}: the hook ran");
        assert_eq!(
            std::fs::read(sentinel(&outside, directory)).unwrap(),
            ORIGINAL,
            "{operation}: the outside entry is untouched"
        );
        match operation {
            "rm" | "rmtree" => assert!(
                std::fs::symlink_metadata(workspace.join("target")).is_err(),
                "{operation}: the swapped-in link itself was removed"
            ),
            "mv" => assert!(
                std::fs::symlink_metadata(workspace.join("moved"))
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false),
                "{operation}: the link itself was moved"
            ),
            _ => {}
        }
    }
}
