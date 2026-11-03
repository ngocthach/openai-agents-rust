//! Ports of the Python sandbox suite's descriptor-pinning tests in `tests/sandbox/test_entries.py`:
//! `test_local_dir_copy_pins_parent_directories_during_open`,
//! `test_local_dir_copy_fallback_rejects_swapped_parent_directory`,
//! `test_local_dir_copy_revalidates_swapped_paths_during_open` and the two
//! `test_local_dir_apply_*rejects_source_root_swapped_to_symlink_after_validation` cases.
//!
//! Python patches `os.open` to swap a path between validation and open. Here the swap runs from
//! the recording session's `mkdir`, which the copy calls after listing a directory and before
//! it opens what it listed: that is the window a concurrent writer to the host tree would use.
#![cfg(feature = "sandbox")]
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{Entry, Manifest};

type Hook = Box<dyn FnOnce() + Send>;

/// A session that records writes and runs one hook the first time `mkdir` is called for a path
/// ending in `hook_suffix`.
impl std::fmt::Debug for SwapSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SwapSession")
    }
}

struct SwapSession {
    manifest: Manifest,
    writes: StdMutex<BTreeMap<String, Vec<u8>>>,
    hook: StdMutex<Option<(String, Hook)>>,
}

impl SwapSession {
    fn new(hook_suffix: &str, hook: impl FnOnce() + Send + 'static) -> Self {
        SwapSession {
            manifest: Manifest::new(),
            writes: StdMutex::new(BTreeMap::new()),
            hook: StdMutex::new(Some((hook_suffix.to_string(), Box::new(hook)))),
        }
    }

    fn writes(&self) -> BTreeMap<String, Vec<u8>> {
        self.writes.lock().expect("writes").clone()
    }
}

#[async_trait]
impl SandboxSession for SwapSession {
    fn id(&self) -> &str {
        "swap"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "swap"
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
    async fn exec(&self, _request: ExecRequest) -> SandboxResult<ExecResult> {
        Ok(ExecResult::default())
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.writes
            .lock()
            .expect("writes")
            .get(path)
            .cloned()
            .ok_or_else(|| SandboxError::new(ErrorCode::WorkspaceReadNotFound, path.to_string()))
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.writes
            .lock()
            .expect("writes")
            .insert(path.to_string(), contents.to_vec());
        Ok(())
    }
    async fn mkdir(&self, path: &str, _parents: bool) -> SandboxResult<()> {
        let hook = {
            let mut slot = self.hook.lock().expect("hook");
            match slot.as_ref() {
                Some((suffix, _)) if path.ends_with(suffix.as_str()) => slot.take(),
                _ => None,
            }
        };
        if let Some((_, hook)) = hook {
            hook();
        }
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

/// A temporary directory with its symlinks resolved, so a link the test creates is the only one
/// on any path under it (a macOS temp directory sits behind `/var`, itself a link).
struct Scratch {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = std::fs::canonicalize(dir.path()).expect("canonical tempdir");
        Scratch { _dir: dir, root }
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    fn mkdir(&self, relative: &str) -> PathBuf {
        let path = self.path(relative);
        std::fs::create_dir_all(&path).expect("mkdir");
        path
    }

    fn write(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.path(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir parent");
        }
        std::fs::write(&path, contents).expect("write");
        path
    }
}

fn context<'a>(error: &'a SandboxError, key: &str) -> &'a str {
    error
        .context
        .get(key)
        .unwrap_or_else(|| panic!("no `{key}` in {:?}", error.context))
}

/// `test_local_dir_copy_pins_parent_directories_during_open`: a directory swapped for a link
/// after the walk opened it does not redirect the files already listed in it.
#[tokio::test]
async fn local_dir_copy_pins_parent_directories_during_open() {
    let scratch = Scratch::new();
    scratch.write("src/nested/safe.txt", "safe");
    scratch.mkdir("src/nested/inner");
    let secret_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/safe.txt", "secret");
    let root = scratch.root.clone();
    let session = SwapSession::new("/nested/inner", move || {
        std::fs::rename(root.join("src/nested"), root.join("src/nested-original")).expect("rename");
        std::os::unix::fs::symlink(&secret_dir, root.join("src/nested")).expect("symlink");
    });

    let result = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect("the pinned parent is still the original directory");

    assert_eq!(result.files.len(), 1);
    assert_eq!(
        session.writes().get("/workspace/copied/nested/safe.txt"),
        Some(&b"safe".to_vec())
    );
}

/// `test_local_dir_copy_fallback_rejects_swapped_parent_directory`: a subdirectory swapped for a
/// link between the listing and the open of it is refused, and nothing is written.
#[tokio::test]
async fn local_dir_copy_rejects_swapped_parent_directory() {
    let scratch = Scratch::new();
    scratch.write("src/nested/safe.txt", "safe");
    let secret_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/safe.txt", "secret");
    let root = scratch.root.clone();
    let session = SwapSession::new("/copied/nested", move || {
        std::fs::rename(root.join("src/nested"), root.join("src/nested-original")).expect("rename");
        std::os::unix::fs::symlink(&secret_dir, root.join("src/nested")).expect("symlink");
    });

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a swapped parent directory");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "nested");
    assert!(session.writes().is_empty());
}

/// `test_local_dir_copy_revalidates_swapped_paths_during_open`: a file swapped for a link after
/// the walk listed it is refused when it is opened.
#[tokio::test]
async fn local_dir_copy_revalidates_swapped_file_during_open() {
    let scratch = Scratch::new();
    scratch.write("src/safe.txt", "safe");
    scratch.mkdir("src/sub");
    let secret = scratch.write("secret.txt", "secret");
    let root = scratch.root.clone();
    let session = SwapSession::new("/copied/sub", move || {
        std::fs::remove_file(root.join("src/safe.txt")).expect("unlink");
        std::os::unix::fs::symlink(&secret, root.join("src/safe.txt")).expect("symlink");
    });

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a swapped file");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "safe.txt");
    assert!(session.writes().is_empty());
}

/// A file replaced by something that is not a regular file after the walk listed it is refused
/// as `path_changed_during_copy`, without blocking on it.
#[tokio::test]
async fn local_dir_copy_rejects_file_replaced_by_a_fifo() {
    let scratch = Scratch::new();
    scratch.write("src/safe.txt", "safe");
    scratch.mkdir("src/sub");
    let root = scratch.root.clone();
    let session = SwapSession::new("/copied/sub", move || {
        std::fs::remove_file(root.join("src/safe.txt")).expect("unlink");
        let path = std::ffi::CString::new(root.join("src/safe.txt").to_str().expect("utf8"))
            .expect("cstring");
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    });

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a replaced file");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "path_changed_during_copy");
    assert_eq!(context(&error, "child"), "safe.txt");
    assert!(session.writes().is_empty());
}

/// `test_local_dir_apply_rejects_source_root_swapped_to_symlink_after_validation` (and its
/// fallback twin): the source root swapped for a link is refused, and nothing is written. The
/// swap runs from the destination `mkdir`, which precedes the source validation, so this
/// exercises the validation refusal; the open of the root itself is also `O_NOFOLLOW`.
#[tokio::test]
async fn local_dir_apply_rejects_source_root_swapped_to_symlink() {
    let scratch = Scratch::new();
    scratch.write("src/safe.txt", "safe");
    let secret_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/secret.txt", "secret");
    let root = scratch.root.clone();
    let session = SwapSession::new("/workspace/copied", move || {
        std::fs::rename(root.join("src"), root.join("src-original")).expect("rename");
        std::os::unix::fs::symlink(&secret_dir, root.join("src")).expect("symlink");
    });

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a swapped source root");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "src");
    assert!(session.writes().is_empty());
}
