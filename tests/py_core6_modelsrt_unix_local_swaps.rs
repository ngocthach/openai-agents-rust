//! The remaining operations of `test_parent_swap_after_validation_cannot_access_outside` and
//! `test_leaf_swap_does_not_follow_new_symlink` (`tests/sandbox/test_unix_local_file_io.py`):
//! `mkdir`, `ls`, `rm`, `rmtree`, `mv` and `same_file` (the read, write and exclusive-create
//! rows are in `py_sandbox_a_unix_local_5.rs`). Python injects the swap by monkeypatching
//! `normalize_path`; `with_after_validate_hook` runs in the same window between validation and
//! use.

#![cfg(all(unix, feature = "sandbox"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxSession;
use openai_agents_sdk::sandbox::{Manifest, SandboxSession};

const ORIGINAL: &[u8] = b"original content";
const OPERATIONS: [&str; 6] = ["mkdir", "ls", "rm", "rmtree", "mv", "same_file"];

fn symlink(to: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(to.as_ref(), link.as_ref()).expect("create a symlink");
}

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

/// A directory holding `sentinel` for the directory operations, a file otherwise.
fn make_entry(path: &Path, directory: bool) {
    if directory {
        std::fs::create_dir_all(path).expect("create entry dir");
        std::fs::write(path.join("sentinel"), ORIGINAL).expect("write sentinel");
    } else {
        std::fs::write(path, ORIGINAL).expect("write entry");
    }
}

fn is_directory(operation: &str) -> bool {
    matches!(operation, "mkdir" | "ls" | "rmtree")
}

async fn operate(
    session: &UnixLocalSandboxSession,
    operation: &str,
    path: &str,
) -> openai_agents_sdk::sandbox::SandboxResult<()> {
    match operation {
        "mkdir" => session.mkdir(path, true).await,
        "ls" => session.ls(path).await.map(|_| ()),
        "rm" => session.rm(path, false).await,
        "rmtree" => session.rm(path, true).await,
        "mv" => session.mv(path, &format!("{path}-moved")).await,
        "same_file" => session.same_file(path, path, false, None).await.map(|_| ()),
        other => panic!("unknown operation {other}"),
    }
}

#[tokio::test]
async fn a_parent_swapped_after_validation_cannot_be_used_by_the_remaining_operations() {
    for operation in OPERATIONS {
        let directory = is_directory(operation);
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        let parent = workspace.join("parent");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::create_dir(&outside).unwrap();
        make_entry(&parent.join("entry"), directory);
        make_entry(&outside.join("entry"), directory);

        let moved = workspace.join("original");
        let (parent_path, outside_path) = (parent.clone(), outside.clone());
        let (session, swapped) =
            session_swapping(&workspace, PathBuf::from("parent/entry"), move || {
                std::fs::rename(&parent_path, &moved).expect("move the parent aside");
                symlink(&outside_path, &parent_path);
            });

        let result = operate(&session, operation, "parent/entry").await;

        assert!(
            swapped.load(Ordering::SeqCst),
            "{operation}: the swap happened"
        );
        assert!(
            result.is_err(),
            "{operation}: the swapped parent must be refused"
        );
        let sentinel = if directory {
            outside.join("entry").join("sentinel")
        } else {
            outside.join("entry")
        };
        assert_eq!(std::fs::read(&sentinel).unwrap(), ORIGINAL, "{operation}");
        assert!(!outside.join("entry-moved").exists(), "{operation}");
    }
}

#[tokio::test]
async fn a_leaf_swapped_for_a_symlink_is_not_followed_by_the_remaining_operations() {
    for operation in OPERATIONS {
        let directory = is_directory(operation);
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        let outside = dir.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        let entry = workspace.join("entry");
        make_entry(&entry, directory);
        make_entry(&outside, directory);

        let moved = workspace.join("original");
        let (leaf, outside_path) = (entry.clone(), outside.clone());
        let (session, swapped) = session_swapping(&workspace, PathBuf::from("entry"), move || {
            std::fs::rename(&leaf, &moved).expect("move the leaf aside");
            symlink(&outside_path, &leaf);
        });

        let result = operate(&session, operation, "entry").await;

        assert!(
            swapped.load(Ordering::SeqCst),
            "{operation}: the swap happened"
        );
        if matches!(operation, "mkdir" | "ls") {
            assert!(
                result.is_err(),
                "{operation}: a swapped-in link must be refused"
            );
        } else {
            // These act on the entry itself, so the link is removed, moved or compared as a
            // link and never followed.
            result.unwrap_or_else(|e| panic!("{operation}: {e}"));
        }
        let sentinel = if directory {
            outside.join("sentinel")
        } else {
            outside.clone()
        };
        assert_eq!(std::fs::read(&sentinel).unwrap(), ORIGINAL, "{operation}");
    }
}

/// `test_create_preserves_a_creator_between_validation_and_open`: a file another writer creates
/// after the path was validated but before the exclusive open is kept, and the create reports
/// that the file already exists instead of replacing it.
#[tokio::test]
async fn an_exclusive_create_keeps_a_file_another_writer_created_in_the_window() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let notes = workspace.join("notes.txt");
    let creator = notes.clone();
    let (session, intervened) =
        session_swapping(&workspace, PathBuf::from("notes.txt"), move || {
            assert!(!creator.exists());
            std::fs::write(&creator, b"other creator\n").expect("the other creator writes");
        });

    let error = session
        .write_new_file("notes.txt", b"replacement\n", None)
        .await
        .expect_err("the exclusive create must lose the race");

    assert!(intervened.load(Ordering::SeqCst), "the other creator ran");
    assert!(error.message.contains("already exists"), "{error}");
    assert_eq!(std::fs::read(&notes).unwrap(), b"other creator\n");
}

/// `test_fifo_replacement_is_rejected_before_io` (read and write): an entry replaced by a FIFO
/// after the entry check is refused instead of blocking on a peer that never comes.
#[tokio::test]
async fn an_entry_replaced_by_a_fifo_is_refused_instead_of_blocking() {
    for operation in ["read", "write"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let entry = workspace.join("entry");
        std::fs::write(&entry, ORIGINAL).unwrap();

        let fifo = entry.clone();
        let (session, swapped) = session_swapping(&workspace, PathBuf::from("entry"), move || {
            std::fs::remove_file(&fifo).expect("remove the regular file");
            let status = std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .expect("run mkfifo");
            assert!(status.success(), "mkfifo failed");
        });

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            if operation == "read" {
                session.read_file("entry").await.map(|_| ())
            } else {
                session.write_file("entry", b"new").await
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{operation}: the operation blocked on the FIFO"));

        assert!(
            swapped.load(Ordering::SeqCst),
            "{operation}: the swap happened"
        );
        assert!(result.is_err(), "{operation}: a FIFO must be refused");
    }
}

/// The `same_file` row of `test_ungranted_path_never_dispatches`: a user-scoped comparison of a
/// path outside every grant is refused by the path policy before anything is dispatched.
#[tokio::test]
async fn an_ungranted_path_never_reaches_the_user_same_file_check() {
    use openai_agents_sdk::sandbox::errors::ErrorCode;
    use openai_agents_sdk::sandbox::types::User;

    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let session = UnixLocalSandboxSession::new(
        Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        false,
    )
    .expect("a session");

    let error = session
        .same_file(
            "/ungranted/file",
            "/ungranted/file",
            false,
            Some(&User::new("example-user")),
        )
        .await
        .expect_err("an ungranted path is refused");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
}
