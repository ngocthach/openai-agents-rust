//! Ports of the Python sandbox suite's `tests/sandbox/test_entries.py`.
//!
//! The Python file drives entries through a `_RecordingSession`; the Rust equivalent below
//! records the same things (the commands, the bytes written, the directories made) behind the
//! `SandboxSession` trait.
//!
//! Host-source confinement is exercised through `Entry::apply_confined` and
//! `Manifest::apply_confined`, which take Python's `base_dir`. The plain `apply` reads host
//! sources wherever they point, because every existing caller relies on that; see the parity
//! report for the default.
//!
//! Python reports a refusal with a context dict (`reason`, `child`, `base_dir`,
//! `extra_path_grants`). Rust's error context holds strings, so the same keys carry the same
//! values, with `extra_path_grants` rendered as a JSON array.
//!
//! Not ported here: `test_local_dir_copy_falls_back_when_safe_dir_fd_open_unavailable` (it
//! switches off Python's `_OPEN_SUPPORTS_DIR_FD` / `_HAS_O_DIRECTORY` flags; Rust always walks
//! with directory descriptors) and `test_git_repo_copies_to_a_posix_destination_from_a_windows_host`
//! (sandbox destinations are `&str` POSIX paths here, so a host-native `PureWindowsPath` cannot
//! reach the copy command). The swap-between-validation-and-open cases are ported in
//! `py_core7_a_entries_swaps.rs`.
#![cfg(feature = "sandbox")]
#![cfg(unix)]

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::materialization::sha256_hex;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, Group};
use openai_agents_sdk::sandbox::workspace_paths::{
    SandboxPathGrant, WorkspacePathPolicy, resolve_workspace_path,
};
use openai_agents_sdk::sandbox::{Entry, Manifest};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// A session that records what entries ask of it. Commands named in `failing` exit non-zero;
/// `git clone` fails when `fail_git_clone` is set.
#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    exec_calls: StdMutex<Vec<Vec<String>>>,
    writes: StdMutex<BTreeMap<String, Vec<u8>>>,
    failing: HashSet<String>,
    fail_git_clone: bool,
    active_writes: AtomicUsize,
    max_active_writes: AtomicUsize,
}

impl RecordingSession {
    fn new() -> Self {
        RecordingSession::with_manifest(Manifest::new())
    }

    fn with_manifest(manifest: Manifest) -> Self {
        RecordingSession {
            manifest,
            exec_calls: StdMutex::new(Vec::new()),
            writes: StdMutex::new(BTreeMap::new()),
            failing: HashSet::new(),
            fail_git_clone: false,
            active_writes: AtomicUsize::new(0),
            max_active_writes: AtomicUsize::new(0),
        }
    }

    fn failing(mut self, commands: &[&str]) -> Self {
        self.failing = commands.iter().map(|command| command.to_string()).collect();
        self
    }

    fn failing_git_clone(mut self) -> Self {
        self.fail_git_clone = true;
        self
    }

    fn exec_calls(&self) -> Vec<Vec<String>> {
        self.exec_calls.lock().expect("exec calls").clone()
    }

    fn writes(&self) -> BTreeMap<String, Vec<u8>> {
        self.writes.lock().expect("writes").clone()
    }

    /// The first recorded command whose first word is `name`.
    fn command(&self, name: &str) -> Vec<String> {
        self.exec_calls()
            .into_iter()
            .find(|call| call.first().map(String::as_str) == Some(name))
            .unwrap_or_else(|| panic!("no `{name}` command in {:?}", self.exec_calls()))
    }
}

#[async_trait]
impl SandboxSession for RecordingSession {
    fn id(&self) -> &str {
        "recording"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "recording"
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
        let argv = request.resolved_argv();
        self.exec_calls
            .lock()
            .expect("exec calls")
            .push(argv.clone());
        let failing_git_clone =
            self.fail_git_clone && argv.len() >= 2 && argv[0] == "git" && argv[1] == "clone";
        let failing_command = argv
            .first()
            .is_some_and(|command| self.failing.contains(command));
        if failing_git_clone || failing_command {
            let stderr: &[u8] = if failing_git_clone {
                b"clone failed"
            } else {
                b"metadata failed"
            };
            return Ok(ExecResult {
                exit_code: 1,
                stdout: Vec::new(),
                stderr: stderr.to_vec(),
            });
        }
        Ok(ExecResult::default())
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.writes
            .lock()
            .expect("writes")
            .get(path)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("no file at `{path}`"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let active = self.active_writes.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active_writes.fetch_max(active, Ordering::SeqCst);
        // Hold the write open until a second one joins it (or a short grace period passes), so
        // the peak concurrency the entry allows is observed rather than raced: each copy first
        // reads its host file on a blocking thread, and those reads finish in any order.
        for _ in 0..50 {
            if self.active_writes.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        self.max_active_writes
            .fetch_max(self.active_writes.load(Ordering::SeqCst), Ordering::SeqCst);
        self.writes
            .lock()
            .expect("writes")
            .insert(path.to_string(), contents.to_vec());
        self.active_writes.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    async fn mkdir(&self, _path: &str, _parents: bool) -> SandboxResult<()> {
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

    fn symlink(&self, link: &str, target: &Path) {
        std::os::unix::fs::symlink(target, self.path(link)).expect("symlink");
    }
}

fn context<'a>(error: &'a SandboxError, key: &str) -> &'a str {
    error
        .context
        .get(key)
        .unwrap_or_else(|| panic!("no `{key}` in {:?}", error.context))
}

fn path_str(path: &Path) -> String {
    path.display().to_string()
}

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| part.to_string()).collect()
}

// ---------------------------------------------------------------------------
// resolve_workspace_path
// ---------------------------------------------------------------------------

/// Python's `test_resolve_workspace_path_rejects_windows_drive_absolute_path`.
#[test]
fn resolve_workspace_path_rejects_windows_drive_absolute_path() {
    let error = resolve_workspace_path("/workspace", "C:/tmp/secret.txt").expect_err("drive path");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must be relative: C:/tmp/secret.txt"
    );
    assert_eq!(context(&error, "rel"), "C:/tmp/secret.txt");
    assert_eq!(context(&error, "reason"), "absolute");
}

/// Python's `test_resolve_workspace_path_rejects_absolute_escape_after_normalization`.
///
/// Python names the path it normalized to (`/etc/passwd`), which is where the escape lands. Rust
/// names the path as the caller wrote it (`/workspace/../etc/passwd`); `workspace_paths.rs`
/// documents that as a deliberate choice.
///
/// DEFERRED: `src/sandbox/workspace_paths.rs::invalid_manifest_path`, which this slice does not
/// own. Python's `rel` for this case is the normalized path; the assertions below are the
/// current behavior. The error code, reason and rejection itself match.
#[test]
fn resolve_workspace_path_rejects_absolute_escape_after_normalization() {
    let error = resolve_workspace_path("/workspace", "/workspace/../etc/passwd")
        .expect_err("an absolute path that escapes");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must be relative: /workspace/../etc/passwd"
    );
    assert_eq!(context(&error, "rel"), "/workspace/../etc/passwd");
    assert_eq!(context(&error, "reason"), "absolute");
}

/// Python's `test_resolve_workspace_path_rejects_absolute_symlink_escape_for_host_root`: for a
/// workspace that is a real host directory, an absolute path that reaches outside through a
/// symlink inside it is refused.
///
/// Rust's `resolve_workspace_path` is purely lexical by design; the symlink-following check is
/// `WorkspacePathPolicy::normalize_host_path`, Python's `resolve_symlinks=True`, which is what
/// is exercised here.
#[test]
fn resolve_workspace_path_rejects_absolute_symlink_escape_for_host_root() {
    let scratch = Scratch::new();
    let root = scratch.mkdir("workspace");
    let outside = scratch.mkdir("outside");
    std::os::unix::fs::symlink(&outside, root.join("link")).expect("symlink");
    let escaped = root.join("link").join("secret.txt");
    let policy = WorkspacePathPolicy::new(path_str(&root)).expect("policy");

    let error = policy
        .normalize_host_path(&path_str(&escaped), false)
        .expect_err("a symlink that leaves the workspace");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        format!("manifest path must be relative: {}", path_str(&escaped))
    );
    assert_eq!(context(&error, "rel"), path_str(&escaped));
    assert_eq!(context(&error, "reason"), "absolute");
}

// ---------------------------------------------------------------------------
// local_file
// ---------------------------------------------------------------------------

/// Python's `test_base_sandbox_session_uses_current_working_directory_for_local_file_sources`.
/// A relative source resolves against the working directory; the source is created under it
/// rather than changing the directory, which would race every other test in the process.
#[tokio::test]
async fn manifest_uses_current_working_directory_for_local_file_sources() {
    let dir = tempfile::tempdir_in(".").expect("tempdir in the working directory");
    std::fs::write(dir.path().join("source.txt"), "hello").expect("write");
    let name = dir
        .path()
        .file_name()
        .expect("name")
        .to_string_lossy()
        .into_owned();
    let session = RecordingSession::new();
    let manifest = Manifest::new().with_entry(
        "copied.txt",
        Entry::local_file(format!("{name}/source.txt")),
    );

    let result = manifest.apply(&session).await.expect("apply");

    assert_eq!(result.files[0].path, "/workspace/copied.txt");
    assert_eq!(result.files[0].sha256, sha256_hex(b"hello"));
    assert_eq!(session.writes()["/workspace/copied.txt"], b"hello".to_vec());
}

/// Python's `test_local_file_rejects_absolute_source_outside_base_dir`.
#[tokio::test]
async fn local_file_rejects_absolute_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    scratch.write("outside/secret.txt", "secret");
    let session = RecordingSession::new();

    let error = Entry::local_file(scratch.path("outside/secret.txt"))
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect_err("a source outside the base directory");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert_eq!(context(&error, "base_dir"), path_str(&base));
    assert!(session.writes().is_empty());
}

/// Python's `test_local_file_rejects_relative_source_outside_base_dir`.
#[tokio::test]
async fn local_file_rejects_relative_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    scratch.write("outside/secret.txt", "secret");
    let session = RecordingSession::new();

    let error = Entry::local_file("../outside/secret.txt")
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect_err("a relative source that climbs out");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert_eq!(context(&error, "base_dir"), path_str(&base));
    assert!(session.writes().is_empty());
}

/// Python's `test_local_file_allows_extra_path_granted_source_outside_base_dir`.
#[tokio::test]
async fn local_file_allows_extra_path_granted_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let outside = scratch.mkdir("outside");
    scratch.write("outside/secret.txt", "secret");
    let grant = SandboxPathGrant::new(path_str(&outside)).expect("grant");
    let session = RecordingSession::with_manifest(Manifest::new().with_extra_path_grant(grant));

    let result = Entry::local_file(scratch.path("outside/secret.txt"))
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect("a granted source");

    assert_eq!(result.files[0].path, "/workspace/copied.txt");
    assert_eq!(
        session.writes()["/workspace/copied.txt"],
        b"secret".to_vec()
    );
}

/// Python's `test_local_file_uses_host_path_for_source_grant`: the grant is judged by the host
/// directory behind it, not by the sandbox path it is visible at.
#[tokio::test]
async fn local_file_uses_host_path_for_source_grant() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let outside = scratch.mkdir("outside");
    scratch.write("outside/secret.txt", "secret");
    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .expect("grant")
        .read_only(true)
        .with_host_path(path_str(&outside))
        .expect("host path");
    let session = RecordingSession::with_manifest(Manifest::new().with_extra_path_grant(grant));

    let result = Entry::local_file(scratch.path("outside/secret.txt"))
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect("a source under the grant's host path");

    assert_eq!(result.files[0].path, "/workspace/copied.txt");
    assert_eq!(
        session.writes()["/workspace/copied.txt"],
        b"secret".to_vec()
    );
}

/// Python's `test_local_file_rejects_source_outside_extra_path_grants`: the refusal lists the
/// grants that were considered.
#[tokio::test]
async fn local_file_rejects_source_outside_extra_path_grants() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    scratch.write("outside/secret.txt", "secret");
    let other = scratch.mkdir("other");
    let grant = SandboxPathGrant::new(path_str(&other)).expect("grant");
    let session = RecordingSession::with_manifest(Manifest::new().with_extra_path_grant(grant));

    let error = Entry::local_file(scratch.path("outside/secret.txt"))
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect_err("a source no grant covers");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert_eq!(
        context(&error, "extra_path_grants"),
        format!("[\"{}\"]", path_str(&other))
    );
    assert!(session.writes().is_empty());
}

/// Python's `test_serialized_manifest_extra_path_grant_allows_local_file_source`.
///
/// Python round-trips the manifest through `model_validate` first; Rust manifests are not
/// deserialized, so this builds the same manifest directly and applies it whole, which is the
/// path that carries the grant from the manifest to the entry.
#[tokio::test]
async fn manifest_extra_path_grant_allows_local_file_source() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let outside = scratch.mkdir("outside");
    scratch.write("outside/secret.txt", "secret");
    let grant = SandboxPathGrant::new(path_str(&outside)).expect("grant");
    let manifest = Manifest::new().with_extra_path_grant(grant).with_entry(
        "copied.txt",
        Entry::local_file(scratch.path("outside/secret.txt")),
    );
    let session = RecordingSession::with_manifest(manifest.clone());

    let result = manifest
        .apply_confined(&session, &base)
        .await
        .expect("apply");

    assert_eq!(result.files[0].path, "/workspace/copied.txt");
    assert_eq!(
        session.writes()["/workspace/copied.txt"],
        b"secret".to_vec()
    );
}

/// Python's `test_local_file_allows_absolute_source_inside_base_dir`.
#[tokio::test]
async fn local_file_allows_absolute_source_inside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    scratch.write("base/source/safe.txt", "safe");
    let session = RecordingSession::new();

    let result = Entry::local_file(scratch.path("base/source/safe.txt"))
        .apply_confined(&session, "/workspace/copied.txt", &base)
        .await
        .expect("a source inside the base directory");

    assert_eq!(result.files[0].path, "/workspace/copied.txt");
    assert_eq!(session.writes()["/workspace/copied.txt"], b"safe".to_vec());
}

/// Python's `test_local_file_rejects_symlinked_source_ancestors`.
#[tokio::test]
async fn local_file_rejects_symlinked_source_ancestors() {
    let scratch = Scratch::new();
    let target_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/sub/secret.txt", "secret");
    scratch.symlink("link", &target_dir);
    let session = RecordingSession::new();

    let error = Entry::local_file("link/sub/secret.txt")
        .apply_confined(&session, "/workspace/copied.txt", &scratch.root)
        .await
        .expect_err("a symlinked ancestor");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "link");
    assert!(session.writes().is_empty());
}

/// Python's `test_local_file_rejects_symlinked_source_leaf`.
#[tokio::test]
async fn local_file_rejects_symlinked_source_leaf() {
    let scratch = Scratch::new();
    let secret = scratch.write("secret.txt", "secret");
    scratch.symlink("link.txt", &secret);
    let session = RecordingSession::new();

    let error = Entry::local_file("link.txt")
        .apply_confined(&session, "/workspace/copied.txt", &scratch.root)
        .await
        .expect_err("a symlinked source");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "link.txt");
    assert!(session.writes().is_empty());
}

/// Python's `test_local_file_rejects_symlinked_source_before_checksum`: a link to a directory
/// is refused as a link, before anything tries to read or hash it.
#[tokio::test]
async fn local_file_rejects_symlinked_source_before_checksum() {
    let scratch = Scratch::new();
    let target_dir = scratch.mkdir("secret-dir");
    scratch.symlink("link.txt", &target_dir);
    let session = RecordingSession::new();

    let error = Entry::local_file("link.txt")
        .apply_confined(&session, "/workspace/copied.txt", &scratch.root)
        .await
        .expect_err("a symlinked source");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "link.txt");
    assert!(session.writes().is_empty());
}

/// Python's default `base_dir` is the working directory (`_manifest_base_dir`): `Entry::apply`
/// with no base directory refuses a host source outside it, so an absolute path to somewhere
/// else on the machine is not readable by merely naming it in a manifest.
#[tokio::test]
async fn apply_defaults_to_confining_host_sources_to_the_working_directory() {
    let scratch = Scratch::new();
    scratch.write("secret.txt", "secret");
    let session = RecordingSession::new();

    let error = Entry::local_file(scratch.path("secret.txt"))
        .apply(&session, "/workspace/copied.txt")
        .await
        .expect_err("a source outside the working directory");

    assert_eq!(error.code, ErrorCode::LocalFileReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert!(session.writes().is_empty());
}

// ---------------------------------------------------------------------------
// local_dir
// ---------------------------------------------------------------------------

/// Python's `test_local_dir_apply_uses_configured_file_copy_fanout`: the entry copies at most
/// the configured number of files at once and reports them in a stable order.
///
/// Python observes the limit by monkeypatching `gather_in_order`; Rust observes the most writes
/// the session ever saw in flight.
#[tokio::test]
async fn local_dir_apply_uses_configured_file_copy_fanout() {
    let scratch = Scratch::new();
    for name in ["a", "b", "c", "d", "e"] {
        scratch.write(&format!("src/{name}.txt"), name);
    }
    let session = RecordingSession::new();

    let result = Entry::local_dir("src")
        .with_copy_concurrency(2)
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect("apply");

    assert_eq!(session.max_active_writes.load(Ordering::SeqCst), 2);
    let paths: Vec<&str> = result.files.iter().map(|file| file.path.as_str()).collect();
    assert_eq!(
        paths,
        vec![
            "/workspace/copied/a.txt",
            "/workspace/copied/b.txt",
            "/workspace/copied/c.txt",
            "/workspace/copied/d.txt",
            "/workspace/copied/e.txt",
        ]
    );
    let writes = session.writes();
    assert_eq!(writes.len(), 5);
    assert_eq!(writes["/workspace/copied/a.txt"], b"a".to_vec());
    assert_eq!(writes["/workspace/copied/e.txt"], b"e".to_vec());
}

/// Python's `test_local_dir_rejects_absolute_source_outside_base_dir`.
#[tokio::test]
async fn local_dir_rejects_absolute_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let outside = scratch.mkdir("outside");
    scratch.write("outside/secret.txt", "secret");
    let session = RecordingSession::new();

    let error = Entry::local_dir(&outside)
        .apply_confined(&session, "/workspace/copied", &base)
        .await
        .expect_err("a source outside the base directory");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert_eq!(context(&error, "base_dir"), path_str(&base));
    assert!(session.writes().is_empty());
}

/// Python's `test_local_dir_rejects_relative_source_outside_base_dir`.
#[tokio::test]
async fn local_dir_rejects_relative_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    scratch.write("outside/secret.txt", "secret");
    let session = RecordingSession::new();

    let error = Entry::local_dir("../outside")
        .apply_confined(&session, "/workspace/copied", &base)
        .await
        .expect_err("a relative source that climbs out");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "outside_base_dir");
    assert_eq!(context(&error, "base_dir"), path_str(&base));
    assert!(session.writes().is_empty());
}

/// Python's `test_local_dir_allows_extra_path_granted_source_outside_base_dir`.
#[tokio::test]
async fn local_dir_allows_extra_path_granted_source_outside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let outside = scratch.mkdir("outside");
    scratch.write("outside/secret.txt", "secret");
    let grant = SandboxPathGrant::new(path_str(&outside))
        .expect("grant")
        .read_only(true);
    let session = RecordingSession::with_manifest(Manifest::new().with_extra_path_grant(grant));

    let result = Entry::local_dir(&outside)
        .apply_confined(&session, "/workspace/copied", &base)
        .await
        .expect("a granted source");

    assert_eq!(result.files[0].path, "/workspace/copied/secret.txt");
    assert_eq!(
        session.writes()["/workspace/copied/secret.txt"],
        b"secret".to_vec()
    );
}

/// Python's `test_local_dir_allows_absolute_source_inside_base_dir`.
#[tokio::test]
async fn local_dir_allows_absolute_source_inside_base_dir() {
    let scratch = Scratch::new();
    let base = scratch.mkdir("base");
    let source = scratch.mkdir("base/source");
    scratch.write("base/source/safe.txt", "safe");
    let session = RecordingSession::new();

    let result = Entry::local_dir(&source)
        .apply_confined(&session, "/workspace/copied", &base)
        .await
        .expect("a source inside the base directory");

    assert_eq!(result.files[0].path, "/workspace/copied/safe.txt");
    assert_eq!(
        session.writes()["/workspace/copied/safe.txt"],
        b"safe".to_vec()
    );
}

/// Python's `test_local_dir_rejects_symlinked_source_ancestors`.
#[tokio::test]
async fn local_dir_rejects_symlinked_source_ancestors() {
    let scratch = Scratch::new();
    let target_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/sub/secret.txt", "secret");
    scratch.symlink("link", &target_dir);
    let session = RecordingSession::new();

    let error = Entry::local_dir("link/sub")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a symlinked ancestor");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "link");
    assert!(session.writes().is_empty());
}

/// Python's `test_local_dir_rejects_symlinked_source_root`.
#[tokio::test]
async fn local_dir_rejects_symlinked_source_root() {
    let scratch = Scratch::new();
    let target_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/secret.txt", "secret");
    scratch.symlink("src", &target_dir);
    let session = RecordingSession::new();

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a symlinked source root");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "src");
    assert!(session.writes().is_empty());
}

/// Python's `test_local_dir_rejects_symlinked_files`: one link anywhere in the tree refuses the
/// whole copy, and the safe file next to it is not written either.
#[tokio::test]
async fn local_dir_rejects_symlinked_files() {
    let scratch = Scratch::new();
    scratch.write("src/safe.txt", "safe");
    let secret = scratch.write("secret.txt", "secret");
    scratch.symlink("src/link.txt", &secret);
    let session = RecordingSession::new();

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a symlinked file");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "link.txt");
    assert!(session.writes().is_empty());
}

/// Python's `test_local_dir_rejects_symlinked_directories`.
#[tokio::test]
async fn local_dir_rejects_symlinked_directories() {
    let scratch = Scratch::new();
    scratch.write("src/safe.txt", "safe");
    let target_dir = scratch.mkdir("secret-dir");
    scratch.write("secret-dir/secret.txt", "secret");
    scratch.symlink("src/linked-dir", &target_dir);
    let session = RecordingSession::new();

    let error = Entry::local_dir("src")
        .apply_confined(&session, "/workspace/copied", &scratch.root)
        .await
        .expect_err("a symlinked directory");

    assert_eq!(error.code, ErrorCode::LocalDirReadError);
    assert_eq!(context(&error, "reason"), "symlink_not_supported");
    assert_eq!(context(&error, "child"), "linked-dir");
    assert!(session.writes().is_empty());
}

// ---------------------------------------------------------------------------
// git_repo
// ---------------------------------------------------------------------------

/// Python's `test_git_repo_uses_fetch_checkout_path_for_commit_refs`: a commit sha is fetched,
/// not cloned.
#[tokio::test]
async fn git_repo_uses_fetch_checkout_path_for_commit_refs() {
    // Mirrors `_GitRefSession`: a `git clone` here would fail, so reaching it fails the apply.
    let session = RecordingSession::new().failing_git_clone();
    let repo = Entry::git_repo("openai/example", "deadbeef");

    repo.apply(&session, "/workspace/repo")
        .await
        .expect("apply");

    let calls = session.exec_calls();
    assert!(
        !calls
            .iter()
            .any(|call| call.len() >= 2 && call[..2] == ["git", "clone"])
    );
    assert!(
        calls
            .iter()
            .any(|call| call.len() >= 2 && call[..2] == ["git", "init"])
    );
    assert!(calls.iter().any(|call| {
        call.len() >= 7
            && call[..2] == ["git", "-C"]
            && call[3..6] == ["remote", "add", "origin"]
            && call[6] == "https://github.com/openai/example.git"
    }));
    assert!(calls.iter().any(|call| {
        call.len() >= 9
            && call[..2] == ["git", "-C"]
            && call[3..7] == ["fetch", "--depth", "1", "--no-tags"]
            && call[call.len() - 2..] == ["origin", "deadbeef"]
    }));
    assert!(calls.iter().any(|call| {
        call.len() >= 6
            && call[..2] == ["git", "-C"]
            && call[3..5] == ["checkout", "--detach"]
            && call.last().map(String::as_str) == Some("FETCH_HEAD")
    }));
}

/// Python's `test_git_repo_rejects_invalid_subpath_before_copy`, all six parametrized cases:
/// each is refused with its reason before a single command runs.
#[tokio::test]
async fn git_repo_rejects_invalid_subpath_before_copy() {
    for (subpath, reason) in [
        ("   ", "empty"),
        ("/docs", "absolute"),
        ("../outside", "parent_traversal"),
        ("docs/../../outside", "parent_traversal"),
        ("C:/repo", "windows_path"),
        ("docs\\outside", "windows_path"),
    ] {
        let session = RecordingSession::new().failing_git_clone();
        let repo = Entry::git_repo("openai/example", "main").with_git_subpath(subpath);

        let error = repo
            .apply(&session, "/workspace/repo")
            .await
            .expect_err("an invalid subpath");

        assert_eq!(error.code, ErrorCode::GitSubpathError, "{subpath:?}");
        assert_eq!(context(&error, "reason"), reason, "{subpath:?}");
        assert_eq!(context(&error, "subpath"), subpath, "{subpath:?}");
        assert!(session.exec_calls().is_empty(), "{subpath:?}");
    }
}

/// Python's `test_git_repo_root_subpath_alias_copies_repo_root`, all five parametrized cases:
/// every spelling of "the repository root" copies the root, with no doubled slash.
#[tokio::test]
async fn git_repo_root_subpath_alias_copies_repo_root() {
    for subpath in ["", ".", "./", "./.", " ./ "] {
        let session = RecordingSession::new();
        let repo = Entry::git_repo("openai/example", "main").with_git_subpath(subpath);

        repo.apply(&session, "/workspace/repo")
            .await
            .expect("apply");

        let copy = session.command("cp");
        assert!(
            copy[3].starts_with("/tmp/sandbox-git-"),
            "{subpath:?}: {copy:?}"
        );
        assert!(copy[3].ends_with("/."), "{subpath:?}: {copy:?}");
        assert!(!copy[3].ends_with("//."), "{subpath:?}: {copy:?}");
        assert_eq!(copy[4], "/workspace/repo/", "{subpath:?}");
    }
}

/// Python's `test_git_repo_allows_relative_subpath_copy`.
#[tokio::test]
async fn git_repo_allows_relative_subpath_copy() {
    let session = RecordingSession::new();
    let repo = Entry::git_repo("openai/example", "main").with_git_subpath("docs/reference");

    repo.apply(&session, "/workspace/repo")
        .await
        .expect("apply");

    let copy = session.command("cp");
    assert!(copy[3].ends_with("/docs/reference/."), "{copy:?}");
    assert_eq!(copy[4], "/workspace/repo/");
}

/// The indices of every `rm -rf -- <dir>` the session ran.
fn scratch_cleanups(session: &RecordingSession) -> Vec<(usize, Vec<String>)> {
    session
        .exec_calls()
        .into_iter()
        .enumerate()
        .filter(|(_, call)| call.len() >= 3 && call[..3] == ["rm", "-rf", "--"])
        .collect()
}

/// Python's `test_git_repo_cleans_temp_clone_after_copy_failure`: the scratch clone is removed
/// before the clone starts and again after the failed copy, and the error says the copy failed.
#[tokio::test]
async fn git_repo_cleans_temp_clone_after_copy_failure() {
    let session = RecordingSession::new().failing(&["cp"]);
    let repo = Entry::git_repo("openai/example", "main");

    let error = repo
        .apply(&session, "/workspace/repo")
        .await
        .expect_err("the copy fails");

    assert_eq!(error.code, ErrorCode::GitCopyError);
    let cleanups = scratch_cleanups(&session);
    assert_eq!(cleanups.len(), 2);
    assert_eq!(cleanups[0].1, cleanups[1].1);
    let copy_index = session
        .exec_calls()
        .iter()
        .position(|call| call.first().map(String::as_str) == Some("cp"))
        .expect("a copy ran");
    assert!(cleanups[1].0 > copy_index);
}

/// Python's `test_git_repo_cleans_temp_clone_after_clone_failure`.
#[tokio::test]
async fn git_repo_cleans_temp_clone_after_clone_failure() {
    let session = RecordingSession::new().failing_git_clone();
    let repo = Entry::git_repo("openai/example", "main");

    let error = repo
        .apply(&session, "/workspace/repo")
        .await
        .expect_err("the clone fails");

    assert_eq!(error.code, ErrorCode::GitCloneError);
    assert!(
        error.message.contains("clone failed"),
        "message: {}",
        error.message
    );
    let cleanups = scratch_cleanups(&session);
    assert_eq!(cleanups.len(), 2);
    assert_eq!(cleanups[0].1, cleanups[1].1);
    let clone_index = session
        .exec_calls()
        .iter()
        .position(|call| call.len() >= 2 && call[..2] == ["git", "clone"])
        .expect("a clone ran");
    assert!(cleanups[1].0 > clone_index);
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------

/// Python's `test_dir_metadata_strips_file_type_bits_before_chmod`: the mode handed to `chmod`
/// is the four-digit permission bits, with no file-type bits in front.
#[tokio::test]
async fn dir_metadata_strips_file_type_bits_before_chmod() {
    let session = RecordingSession::new();

    Entry::dir()
        .apply(&session, "/workspace/dir")
        .await
        .expect("apply");

    assert!(
        session
            .exec_calls()
            .contains(&argv(&["chmod", "0755", "/workspace/dir"])),
        "calls: {:?}",
        session.exec_calls()
    );
}

/// Python's `test_apply_manifest_raises_on_chmod_failure`.
#[tokio::test]
async fn apply_manifest_raises_on_chmod_failure() {
    let session = RecordingSession::new().failing(&["chmod"]);
    let manifest = Manifest::new().with_entry("copied.txt", Entry::text_file("hello"));

    let error = manifest.apply(&session).await.expect_err("chmod fails");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
}

/// Python's `test_apply_manifest_raises_on_chgrp_failure`: the group is set first, and a failure
/// stops the entry before `chmod` runs.
#[tokio::test]
async fn apply_manifest_raises_on_chgrp_failure() {
    let session = RecordingSession::new().failing(&["chgrp"]);
    let manifest = Manifest::new().with_entry(
        "copied.txt",
        Entry::text_file("hello").with_group(Group::new("sandbox-user")),
    );

    let error = manifest.apply(&session).await.expect_err("chgrp fails");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    let calls = session.exec_calls();
    assert!(calls.contains(&argv(&["chgrp", "sandbox-user", "/workspace/copied.txt"])));
    assert!(
        !calls
            .iter()
            .any(|call| call.first().map(String::as_str) == Some("chmod"))
    );
}
