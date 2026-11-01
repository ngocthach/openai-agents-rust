//! Ports of the Python sandbox suite's `tests/sandbox/test_manifest.py`,
//! `tests/sandbox/test_manifest_application.py` and `tests/sandbox/test_manifest_config.py`.
//!
//! Python splits manifest materialization into a `ManifestApplier` fed by callbacks; Rust keeps
//! it on [`Manifest::apply`], which drives a [`SandboxSession`]. The application tests therefore
//! run against a recording session and assert on what the session saw (commands, writes, the
//! order they started and finished in) instead of on the callback arguments.
//!
//! Python's `Manifest` is a pydantic model with a tagged-union registry for env values. Rust
//! reads and writes the payload with `Manifest::from_value` / `to_value`; those serialization
//! and dictionary-configuration tests are ported in `tests/py_core8_r8g1_manifest.rs`.
//!
//! Rust keeps manifest entries in a `BTreeMap`, so entries materialize in key order where
//! Python's dict preserves insertion order. Tests whose Python version depends on insertion
//! order name their entries so the two orders agree.
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::environment::{EnvEntry, EnvValue, Environment};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::manifest::{
    SandboxConcurrencyLimits, truncate_manifest_description,
};
use openai_agents_sdk::sandbox::materialization::{MaterializedFile, sha256_hex};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, Group, User};
use openai_agents_sdk::sandbox::{
    ConfiguredMountStrategy, Entry, EntryKind, GcsMount, Manifest, Mount, SandboxAgent,
    SandboxRunConfig,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// A GCS bucket attached in-container with `mount-s3`'s Google counterpart, the strategy every
/// Python mount test in this file builds with `InContainerMountStrategy(MountpointMountPattern())`.
fn gcs_mount(bucket: &str) -> Mount {
    ConfiguredMountStrategy::mountpoint(GcsMount::new(bucket)).mount()
}

/// A session that records what it is asked to do instead of doing it.
///
/// `write_file` and `mkdir` yield once between logging "start" and "end" so concurrent work
/// interleaves, which is what lets the ordering tests observe whether two operations overlapped.
#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    events: StdMutex<Vec<String>>,
    mkdirs: StdMutex<Vec<String>>,
    execs: StdMutex<Vec<Vec<String>>>,
    writes: StdMutex<Vec<String>>,
    /// A command name whose invocation fails, with the result it fails with.
    failing: Option<(String, ExecResult)>,
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
            events: StdMutex::new(Vec::new()),
            mkdirs: StdMutex::new(Vec::new()),
            execs: StdMutex::new(Vec::new()),
            writes: StdMutex::new(Vec::new()),
            failing: None,
            active_writes: AtomicUsize::new(0),
            max_active_writes: AtomicUsize::new(0),
        }
    }

    fn failing(mut self, command: &str, result: ExecResult) -> Self {
        self.failing = Some((command.to_string(), result));
        self
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().expect("events").clone()
    }

    fn position(&self, event: &str) -> usize {
        self.events()
            .iter()
            .position(|candidate| candidate == event)
            .unwrap_or_else(|| panic!("no event `{event}` in {:?}", self.events()))
    }

    fn exec_commands(&self) -> Vec<Vec<String>> {
        self.execs.lock().expect("execs").clone()
    }

    /// The commands that create or change accounts. Applying an entry also runs `chmod`, so the
    /// provisioning tests look at these alone.
    fn account_commands(&self) -> Vec<Vec<String>> {
        self.exec_commands()
            .into_iter()
            .filter(|command| {
                matches!(
                    command.first().map(String::as_str),
                    Some("groupadd" | "useradd" | "usermod")
                )
            })
            .collect()
    }

    fn written_paths(&self) -> Vec<String> {
        let mut writes = self.writes.lock().expect("writes").clone();
        writes.sort();
        writes
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
        self.events
            .lock()
            .expect("events")
            .push(format!("exec {}", argv.join(" ")));
        self.execs.lock().expect("execs").push(argv.clone());
        if let Some((command, result)) = &self.failing
            && argv.first() == Some(command)
        {
            return Ok(result.clone());
        }
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            format!("no file at `{path}`"),
        ))
    }

    async fn write_file(&self, path: &str, _contents: &[u8]) -> SandboxResult<()> {
        let active = self.active_writes.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active_writes.fetch_max(active, Ordering::SeqCst);
        self.events
            .lock()
            .expect("events")
            .push(format!("start {path}"));
        tokio::task::yield_now().await;
        self.events
            .lock()
            .expect("events")
            .push(format!("end {path}"));
        self.writes.lock().expect("writes").push(path.to_string());
        self.active_writes.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    async fn mkdir(&self, path: &str, _parents: bool) -> SandboxResult<()> {
        self.events
            .lock()
            .expect("events")
            .push(format!("start {path}"));
        self.mkdirs.lock().expect("mkdirs").push(path.to_string());
        tokio::task::yield_now().await;
        self.events
            .lock()
            .expect("events")
            .push(format!("end {path}"));
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn materialized(path: &str, contents: &[u8]) -> MaterializedFile {
    MaterializedFile {
        path: path.to_string(),
        sha256: sha256_hex(contents),
    }
}

// ---------------------------------------------------------------------------
// test_manifest.py: path validation
// ---------------------------------------------------------------------------

/// Python's `test_manifest_rejects_nested_child_paths_that_escape_workspace`: a child named
/// `../outside.txt` inside a directory entry is refused when the manifest is validated, not
/// halfway through materialization.
#[test]
fn manifest_rejects_nested_child_paths_that_escape_workspace() {
    let manifest = Manifest::new().with_entry(
        "safe",
        Entry::dir().with_child("../outside.txt", Entry::text_file("nope")),
    );

    let error = manifest.validated_entries().expect_err("escaping child");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(
        error.message.contains("must not escape root"),
        "message: {}",
        error.message
    );
}

/// Python's `test_manifest_rejects_nested_absolute_child_paths`.
#[test]
fn manifest_rejects_nested_absolute_child_paths() {
    let manifest = Manifest::new().with_entry(
        "safe",
        Entry::dir().with_child("/tmp/outside.txt", Entry::text_file("nope")),
    );

    let error = manifest.validated_entries().expect_err("absolute child");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(
        error.message.contains("must be relative"),
        "message: {}",
        error.message
    );
}

/// Python's `test_manifest_rejects_windows_drive_absolute_entry_paths`: the message and context
/// name the path with POSIX separators.
#[test]
fn manifest_rejects_windows_drive_absolute_entry_paths() {
    let manifest = Manifest::new().with_entry("C:\\tmp\\outside.txt", Entry::text_file("nope"));

    let error = manifest.validated_entries().expect_err("drive path");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must be relative: C:/tmp/outside.txt"
    );
    assert_eq!(
        error.context.get("rel").map(String::as_str),
        Some("C:/tmp/outside.txt")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("absolute")
    );
}

// ---------------------------------------------------------------------------
// test_manifest.py: ephemeral paths and mount targets
// ---------------------------------------------------------------------------

/// Python's `test_manifest_ephemeral_entry_paths_include_nested_children`: an ephemeral file
/// inside a directory puts that one file on the skip list.
#[test]
fn manifest_ephemeral_entry_paths_include_nested_children() {
    let manifest = Manifest::new().with_entry(
        "dir",
        Entry::dir()
            .with_child("keep.txt", Entry::text_file("keep"))
            .with_child("tmp.txt", Entry::text_file("tmp").with_ephemeral(true)),
    );

    assert_eq!(manifest.ephemeral_entry_paths(), vec!["dir/tmp.txt"]);
}

/// Python's `test_manifest_ephemeral_persistence_paths_include_resolved_mount_targets`: the
/// mount's entry path, where it actually landed, and the nested ephemeral file are all skipped.
#[test]
fn manifest_ephemeral_persistence_paths_include_resolved_mount_targets() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(
            "logical",
            Entry::mount(gcs_mount("bucket").with_mount_path("actual")),
        )
        .with_entry(
            "dir",
            Entry::dir().with_child("tmp.txt", Entry::text_file("tmp").with_ephemeral(true)),
        );

    let mut expected = vec!["logical", "actual", "dir/tmp.txt"];
    expected.sort_unstable();
    assert_eq!(
        manifest
            .try_ephemeral_persistence_paths()
            .expect("persistence paths"),
        expected
    );
    assert_eq!(manifest.ephemeral_persistence_paths(), expected);
}

/// Python's `test_manifest_ephemeral_mount_targets_sort_by_resolved_depth`: deeper mounts come
/// first so they detach before the mount that contains them.
#[test]
fn manifest_ephemeral_mount_targets_sort_by_resolved_depth() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(
            "parent",
            Entry::mount(gcs_mount("parent").with_mount_path("repo")),
        )
        .with_entry(
            "nested",
            Entry::dir().with_child(
                "child",
                Entry::mount(gcs_mount("child").with_mount_path("repo/sub")),
            ),
        );

    let targets = manifest
        .try_ephemeral_mount_targets()
        .expect("mount targets");
    let resolved: Vec<(Option<&str>, &str)> = targets
        .iter()
        .map(|(mount, path)| (mount.mount_path.as_deref(), path.as_str()))
        .collect();

    assert_eq!(
        resolved,
        vec![
            (Some("repo/sub"), "/workspace/repo/sub"),
            (Some("repo"), "/workspace/repo"),
        ]
    );
}

/// Python's `test_manifest_ephemeral_mount_targets_normalize_non_escaping_mount_paths`: a `..`
/// that stays inside the workspace is collapsed.
#[test]
fn manifest_ephemeral_mount_targets_normalize_non_escaping_mount_paths() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "logical",
        Entry::mount(gcs_mount("bucket").with_mount_path("/workspace/repo/../actual")),
    );

    let targets = manifest
        .try_ephemeral_mount_targets()
        .expect("mount targets");

    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].1, "/workspace/actual");
    assert_eq!(
        manifest
            .try_ephemeral_persistence_paths()
            .expect("persistence paths"),
        vec!["actual", "logical"]
    );
}

/// Python's `test_manifest_ephemeral_mount_targets_reject_escaping_mount_paths`: both the
/// target listing and the persistence listing refuse a mount path that leaves the workspace.
#[test]
fn manifest_ephemeral_mount_targets_reject_escaping_mount_paths() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "logical",
        Entry::mount(gcs_mount("bucket").with_mount_path("/workspace/../../tmp")),
    );

    let targets_error = manifest
        .try_ephemeral_mount_targets()
        .map(|_| ())
        .expect_err("escaping mount path");
    assert_eq!(targets_error.code, ErrorCode::InvalidManifestPath);
    assert!(
        targets_error.message.contains("must not escape root"),
        "message: {}",
        targets_error.message
    );

    let persistence_error = manifest
        .try_ephemeral_persistence_paths()
        .expect_err("escaping mount path");
    assert!(
        persistence_error.message.contains("must not escape root"),
        "message: {}",
        persistence_error.message
    );
}

/// Python's `test_manifest_ephemeral_mount_targets_reject_windows_drive_mount_path`.
#[test]
fn manifest_ephemeral_mount_targets_reject_windows_drive_mount_path() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "logical",
        Entry::mount(gcs_mount("bucket").with_mount_path("C:\\tmp\\mount")),
    );

    let error = manifest
        .try_ephemeral_mount_targets()
        .map(|_| ())
        .expect_err("drive mount path");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must be relative: C:/tmp/mount"
    );
    assert_eq!(
        error.context.get("rel").map(String::as_str),
        Some("C:/tmp/mount")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("absolute")
    );
}

// ---------------------------------------------------------------------------
// test_manifest.py: describe / truncation
// ---------------------------------------------------------------------------

/// Python's `test_manifest_describe_preserves_tree_rendering_after_renderer_extract`.
///
/// Rust's `Manifest::describe` keeps its indented layout, which `tests/sandbox_tests.rs` and the
/// skills capability pin; the Python tree rendering is `Manifest::render_tree`.
#[test]
fn manifest_describe_preserves_tree_rendering() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(
            "repo",
            Entry::dir().with_description("project root").with_child(
                "README.md",
                Entry::text_file("hi").with_description("overview"),
            ),
        )
        .with_entry(
            "data",
            Entry::mount(gcs_mount("bucket")).with_description("shared data"),
        );

    let description = manifest
        .render_tree(Some(2), Some(5000))
        .expect("render tree");

    assert!(
        description.starts_with("/workspace\n"),
        "description: {description}"
    );
    assert!(description.contains("data/"), "description: {description}");
    assert!(
        description.contains("/workspace/data"),
        "description: {description}"
    );
    assert!(description.contains("repo/"), "description: {description}");
    assert!(
        description.contains("/workspace/repo/README.md"),
        "description: {description}"
    );
    assert!(
        description.contains("shared data"),
        "description: {description}"
    );
}

/// The Python tree shape in full, so a renderer drifting from it is caught by more than
/// substrings: connectors, the aligned comment column, `/`-suffixed directories, descriptions.
#[test]
fn manifest_render_tree_matches_the_python_layout() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(
            "repo",
            Entry::dir()
                .with_description("my repo")
                .with_child("README.md", Entry::text_file("hi")),
        )
        .with_entry("notes.txt", Entry::text_file("n"));

    let rendered = manifest.render_tree(None, None).expect("render tree");

    let expected = [
        "/workspace".to_string(),
        format!(
            "\u{251c}\u{2500}\u{2500} notes.txt{}# /workspace/notes.txt",
            " ".repeat(6)
        ),
        format!(
            "\u{2514}\u{2500}\u{2500} repo/{}# /workspace/repo \u{2014} my repo",
            " ".repeat(10)
        ),
        format!(
            "    \u{2514}\u{2500}\u{2500} README.md{}# /workspace/repo/README.md",
            " ".repeat(2)
        ),
    ]
    .join("\n")
        + "\n";
    assert_eq!(rendered, expected);
}

/// Python rejects a zero depth or budget rather than rendering nothing.
#[test]
fn manifest_render_tree_rejects_zero_limits() {
    let manifest = Manifest::new();

    let depth = manifest.render_tree(Some(0), None).expect_err("zero depth");
    assert!(
        depth
            .message
            .contains("depth must be a non-zero positive integer")
    );
    let budget = manifest
        .render_tree(None, Some(0))
        .expect_err("zero budget");
    assert!(
        budget
            .message
            .contains("max_chars must be a non-zero positive integer")
    );
}

/// Python's `test_manifest_description_truncation_respects_short_limits`: the result never
/// exceeds the budget, including budgets smaller than the truncation marker itself.
#[test]
fn manifest_description_truncation_respects_short_limits() {
    let description = "0123456789".repeat(20);

    for max_chars in 0..40 {
        let truncated = truncate_manifest_description(&description, Some(max_chars));
        assert!(
            truncated.chars().count() <= max_chars,
            "max_chars={max_chars} produced {} chars",
            truncated.chars().count()
        );
    }
}

/// Python's `test_manifest_description_truncation_preserves_unbounded_description`.
#[test]
fn manifest_description_truncation_preserves_unbounded_description() {
    assert_eq!(truncate_manifest_description("short", None), "short");
}

/// The marker a truncated description ends in says how much was dropped and points at `ls`.
#[test]
fn manifest_description_truncation_ends_in_an_accurate_marker() {
    let description = "x".repeat(500);

    let truncated = truncate_manifest_description(&description, Some(300));

    assert_eq!(truncated.chars().count(), 300);
    let omitted: usize = truncated
        .split("(truncated ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|digits| digits.parse().ok())
        .expect("marker states the omitted count");
    let kept = truncated.chars().take_while(|c| *c == 'x').count();
    assert_eq!(kept + omitted, 500);
    assert!(truncated.contains("Use `ls` to explore"));
}

// ---------------------------------------------------------------------------
// test_manifest.py: environment resolution
// ---------------------------------------------------------------------------

/// Python's `_SecretReferenceEnvValue`.
#[derive(Debug)]
struct SecretReferenceEnvValue {
    key: &'static str,
}

#[async_trait]
impl EnvValue for SecretReferenceEnvValue {
    fn env_value_type(&self) -> &str {
        "test.secret_reference"
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(format!("resolved-secret-for-{}", self.key))
    }
}

/// Python's `_BlockingEnvValue`: stands in for a lookup that reaches a secret store. It records
/// that it started, then waits forever; if it is cancelled (dropped) it records that instead.
#[derive(Debug)]
struct BlockingEnvValue {
    started: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

#[async_trait]
impl EnvValue for BlockingEnvValue {
    fn env_value_type(&self) -> &str {
        "test.blocking"
    }

    async fn resolve(&self) -> SandboxResult<String> {
        struct OnDrop(Arc<AtomicBool>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let _cancelled_when_dropped = OnDrop(self.cancelled.clone());
        self.started.store(true, Ordering::SeqCst);
        std::future::pending::<()>().await;
        Ok("unreachable".to_string())
    }
}

/// Python's `_FailingEnvValue`: fails only once the sibling is genuinely in flight, so the test
/// pins the interleaving instead of racing the two resolvers.
#[derive(Debug)]
struct FailingEnvValue {
    sibling_started: Arc<AtomicBool>,
}

#[async_trait]
impl EnvValue for FailingEnvValue {
    fn env_value_type(&self) -> &str {
        "test.failing"
    }

    async fn resolve(&self) -> SandboxResult<String> {
        while !self.sibling_started.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "secret backend rejected the request",
        ))
    }
}

/// Python's `test_environment_resolve_cancels_siblings_when_one_resolver_fails`: a failed lookup
/// must not leave the other resolvers running. `EnvValue` is an extension point, so `resolve`
/// fans out caller-supplied futures that can reach a secret store.
#[tokio::test]
async fn environment_resolve_cancels_siblings_when_one_resolver_fails() {
    let started = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let environment = Environment::new()
        .with_entry(
            "BLOCKING",
            EnvEntry::new(Arc::new(BlockingEnvValue {
                started: started.clone(),
                cancelled: cancelled.clone(),
            })),
        )
        .with_entry(
            "FAILING",
            EnvEntry::new(Arc::new(FailingEnvValue {
                sibling_started: started.clone(),
            })),
        );

    let error = tokio::time::timeout(std::time::Duration::from_secs(5), environment.resolve())
        .await
        .expect("a failed lookup returns at once")
        .expect_err("a lookup fails");

    assert!(
        error
            .message
            .contains("secret backend rejected the request"),
        "message: {}",
        error.message
    );
    assert!(started.load(Ordering::SeqCst), "the sibling was in flight");
    assert!(
        cancelled.load(Ordering::SeqCst),
        "sibling resolver was not cancelled"
    );
}

/// Python's `test_environment_resolve_still_returns_every_value`.
#[tokio::test]
async fn environment_resolve_still_returns_every_value() {
    let environment = Environment::new()
        .with_entry("PLAIN", EnvEntry::literal("literal"))
        .with_entry(
            "REF",
            EnvEntry::new(Arc::new(SecretReferenceEnvValue { key: "alpha" })),
        )
        .with_entry(
            "ENTRY",
            EnvEntry::new(Arc::new(SecretReferenceEnvValue { key: "beta" })),
        );

    let resolved = environment.resolve().await.expect("resolve");

    assert_eq!(resolved.len(), 3);
    assert_eq!(resolved["PLAIN"], "literal");
    assert_eq!(resolved["REF"], "resolved-secret-for-alpha");
    assert_eq!(resolved["ENTRY"], "resolved-secret-for-beta");
}

// ---------------------------------------------------------------------------
// test_manifest_application.py
// ---------------------------------------------------------------------------

/// Python's `test_manifest_applier_only_applies_ephemeral_entries_without_account_provisioning`:
/// a resume re-applies the ephemeral entry alone and never touches the accounts.
#[tokio::test]
async fn manifest_only_applies_ephemeral_entries_without_account_provisioning() {
    let session = RecordingSession::new();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("keep.txt", Entry::text_file("keep"))
        .with_entry("tmp.txt", Entry::text_file("tmp").with_ephemeral(true))
        .with_user(User::new("alice"))
        .with_group(Group::new("dev").with_user(User::new("alice")));

    let result = manifest.apply_ephemeral(&session).await.expect("apply");

    assert_eq!(
        session
            .mkdirs
            .lock()
            .expect("mkdirs")
            .first()
            .map(String::as_str),
        Some("/workspace")
    );
    assert!(session.account_commands().is_empty());
    assert_eq!(session.written_paths(), vec!["/workspace/tmp.txt"]);
    assert_eq!(
        result.files,
        vec![materialized("/workspace/tmp.txt", b"tmp")]
    );
}

/// Python's `test_manifest_applier_only_ephemeral_reapplies_nested_ephemeral_children`.
#[tokio::test]
async fn manifest_only_ephemeral_reapplies_nested_ephemeral_children() {
    let session = RecordingSession::new();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "dir",
        Entry::dir()
            .with_child("keep.txt", Entry::text_file("keep"))
            .with_child("tmp.txt", Entry::text_file("tmp").with_ephemeral(true)),
    );

    let result = manifest.apply_ephemeral(&session).await.expect("apply");

    assert_eq!(session.written_paths(), vec!["/workspace/dir/tmp.txt"]);
    assert_eq!(
        result.files,
        vec![materialized("/workspace/dir/tmp.txt", b"tmp")]
    );
}

/// Python's `test_manifest_applier_only_ephemeral_reapplies_full_ephemeral_directories`: a
/// directory that is itself ephemeral is re-created whole, children and all.
///
/// Python's result lists the one directory it applied; Rust's lists every file written under it.
#[tokio::test]
async fn manifest_only_ephemeral_reapplies_full_ephemeral_directories() {
    let session = RecordingSession::new();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "tmp",
        Entry::dir()
            .with_ephemeral(true)
            .with_child("keep.txt", Entry::text_file("keep"))
            .with_child(
                "nested",
                Entry::dir().with_child("child.txt", Entry::text_file("child")),
            )
            .with_child("tmp.txt", Entry::text_file("tmp").with_ephemeral(true)),
    );

    let result = manifest.apply_ephemeral(&session).await.expect("apply");

    assert_eq!(
        session.written_paths(),
        vec![
            "/workspace/tmp/keep.txt",
            "/workspace/tmp/nested/child.txt",
            "/workspace/tmp/tmp.txt",
        ]
    );
    let mut applied = result.paths();
    applied.sort();
    assert_eq!(applied, session.written_paths());
}

/// Python's `test_manifest_applier_respects_explicit_base_dir`: a relative host source resolves
/// against the base directory the manifest is applied with, not against the working directory.
#[tokio::test]
async fn manifest_respects_explicit_base_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = std::fs::canonicalize(dir.path()).expect("canonical tempdir");
    std::fs::write(base.join("file.txt"), "hello").expect("write");
    let manifest = Manifest::new().with_entry("file.txt", Entry::local_file("file.txt"));
    let session = RecordingSession::new();

    let result = manifest
        .apply_confined(&session, &base)
        .await
        .expect("apply");

    assert_eq!(
        result.files,
        vec![materialized("/workspace/file.txt", b"hello")]
    );
    assert_eq!(session.written_paths(), vec!["/workspace/file.txt"]);
}

/// Python's `test_manifest_applier_caps_parallel_entry_batch`: the manifest-entry concurrency
/// limit bounds how many entries are written at once, and results keep manifest order.
///
/// Python observes the limit by monkeypatching `gather_in_order`; Rust observes the same fact
/// directly, as the most writes the session ever saw in flight.
#[tokio::test]
async fn manifest_caps_parallel_entry_batch() {
    let session = RecordingSession::new();
    let manifest = Manifest::new()
        .with_entry("a.txt", Entry::text_file("a"))
        .with_entry("b.txt", Entry::text_file("b"))
        .with_entry("c.txt", Entry::text_file("c"))
        .with_entry("d.txt", Entry::text_file("d"))
        .with_concurrency_limits(SandboxConcurrencyLimits::new().with_manifest_entries(Some(2)))
        .expect("valid limits");

    let result = manifest.apply(&session).await.expect("apply");

    assert_eq!(session.max_active_writes.load(Ordering::SeqCst), 2);
    assert_eq!(
        result.files,
        vec![
            materialized("/workspace/a.txt", b"a"),
            materialized("/workspace/b.txt", b"b"),
            materialized("/workspace/c.txt", b"c"),
            materialized("/workspace/d.txt", b"d"),
        ]
    );
}

/// Python's `Manifest.entries` is a dict, so entries are materialized, and mounts attached, in the
/// order they were written. No Python test pins this, but `ManifestApplier.apply_manifest`
/// iterates `validated_entries().items()` and reports files in that order, and `mount_targets`
/// breaks depth ties with a stable sort. `Manifest` keeps a key-ordered map for lookup and
/// records insertion order beside it.
#[tokio::test]
async fn manifest_follows_insertion_order_where_python_dicts_do() {
    let manifest = Manifest::new()
        .with_entry("z.txt", Entry::text_file("z"))
        .with_entry("a.txt", Entry::text_file("a"))
        .with_entry("m.txt", Entry::text_file("m"))
        // Replacing a key keeps its original position, as assigning to a dict key does.
        .with_entry("z.txt", Entry::text_file("zz"));

    let order: Vec<&str> = manifest
        .entries_in_order()
        .map(|(path, _)| path.as_str())
        .collect();
    assert_eq!(order, vec!["z.txt", "a.txt", "m.txt"]);
    // Lookup stays key-ordered.
    let keys: Vec<&str> = manifest.entries().keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["a.txt", "m.txt", "z.txt"]);

    let session = RecordingSession::new();
    let result = manifest.apply(&session).await.expect("apply");
    assert_eq!(
        result.files,
        vec![
            materialized("/workspace/z.txt", b"zz"),
            materialized("/workspace/a.txt", b"a"),
            materialized("/workspace/m.txt", b"m"),
        ]
    );

    // Two mounts at the same depth: the one written first comes first.
    let mounts = Manifest::new()
        .with_root("/workspace")
        .with_entry("zeta", Entry::mount(gcs_mount("zeta")))
        .with_entry("alpha", Entry::mount(gcs_mount("alpha")));
    let targets: Vec<String> = mounts
        .try_mount_targets()
        .expect("mount targets")
        .into_iter()
        .map(|(_, path)| path)
        .collect();
    assert_eq!(targets, vec!["/workspace/zeta", "/workspace/alpha"]);

    // How a manifest was built must not change what resume compares.
    let reversed = Manifest::new()
        .with_entry("m.txt", Entry::text_file("m"))
        .with_entry("a.txt", Entry::text_file("a"))
        .with_entry("z.txt", Entry::text_file("zz"));
    assert_eq!(
        openai_agents_sdk::sandbox::manifest_digest(&manifest),
        openai_agents_sdk::sandbox::manifest_digest(&reversed)
    );
}

/// Python's `test_manifest_applier_provisions_groups_and_unique_users_before_entries`.
#[tokio::test]
async fn manifest_provisions_groups_and_unique_users_before_entries() {
    let session = RecordingSession::new();
    let manifest = Manifest::new().with_user(User::new("alice")).with_group(
        Group::new("dev")
            .with_user(User::new("alice"))
            .with_user(User::new("bob")),
    );

    let result = manifest.apply(&session).await.expect("apply");
    let execs = session.exec_commands();
    let command = |parts: &[&str]| -> Vec<String> { parts.iter().map(|p| p.to_string()).collect() };

    assert!(result.is_empty());
    assert_eq!(execs[0], command(&["groupadd", "dev"]));
    assert_eq!(
        execs
            .iter()
            .filter(|c| **c == command(&["groupadd", "alice"]))
            .count(),
        0
    );
    assert_eq!(
        execs
            .iter()
            .filter(|c| **c == command(&["groupadd", "bob"]))
            .count(),
        0
    );
    let useradd = |name: &str| command(&["useradd", "-U", "-M", "-s", "/usr/sbin/nologin", name]);
    // `alice` is both a manifest user and a group member but is created once.
    assert_eq!(execs.iter().filter(|c| **c == useradd("alice")).count(), 1);
    assert!(execs.contains(&useradd("bob")));
    assert!(execs.contains(&command(&["usermod", "-aG", "dev", "alice"])));
    assert!(execs.contains(&command(&["usermod", "-aG", "dev", "bob"])));
}

/// Python's `test_manifest_applier_can_apply_full_manifest_without_account_provisioning`.
#[tokio::test]
async fn manifest_can_apply_full_manifest_without_account_provisioning() {
    let session = RecordingSession::new();
    let manifest = Manifest::new()
        .with_entry("file.txt", Entry::text_file("hello"))
        .with_user(User::new("alice"))
        .with_group(Group::new("dev").with_user(User::new("alice")));

    let result = manifest
        .apply_without_account_provisioning(&session)
        .await
        .expect("apply");

    assert!(session.account_commands().is_empty());
    assert_eq!(session.written_paths(), vec!["/workspace/file.txt"]);
    assert_eq!(
        result.files,
        vec![materialized("/workspace/file.txt", b"hello")]
    );
}

/// Python's `test_manifest_applier_raises_with_command_stdout_and_stderr_on_provision_failure`.
///
/// DEFERRED: Python's `ExecNonZeroError` message is `stdout: ...\nstderr: ...` and its context
/// also carries the `command` tuple. Rust's `SandboxError::exec_nonzero` (src/sandbox/errors.rs,
/// not owned by this slice) words the message `Command `...` exited with code N: <stderr>`; the
/// manifest adds `command_str`, `stdout` and `stderr` to the context.
#[tokio::test]
async fn manifest_raises_with_command_stdout_and_stderr_on_provision_failure() {
    let session = RecordingSession::new().failing(
        "groupadd",
        ExecResult {
            exit_code: 9,
            stdout: b"groupadd output".to_vec(),
            stderr: b"groupadd failed".to_vec(),
        },
    );
    let manifest = Manifest::new().with_group(Group::new("dev"));

    let error = manifest.apply(&session).await.expect_err("provision fails");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert_eq!(error.exit_code, Some(9));
    assert_eq!(
        error.context.get("command_str").map(String::as_str),
        Some("groupadd dev")
    );
    assert_eq!(
        error.context.get("stdout").map(String::as_str),
        Some("groupadd output")
    );
    assert_eq!(
        error.context.get("stderr").map(String::as_str),
        Some("groupadd failed")
    );
    assert!(
        error.message.contains("groupadd failed"),
        "message: {}",
        error.message
    );
}

/// Python's `test_manifest_applier_raises_without_stream_labels_when_only_stdout_is_present`:
/// when stderr is empty the error falls back to what the command printed on stdout.
#[tokio::test]
async fn manifest_raises_without_stream_labels_when_only_stdout_is_present() {
    let session = RecordingSession::new().failing(
        "useradd",
        ExecResult {
            exit_code: 127,
            stdout: b"useradd unavailable".to_vec(),
            stderr: Vec::new(),
        },
    );
    let manifest = Manifest::new().with_user(User::new("sandbox-user"));

    let error = manifest.apply(&session).await.expect_err("provision fails");

    assert_eq!(error.exit_code, Some(127));
    assert_eq!(
        error.context.get("command_str").map(String::as_str),
        Some("useradd -U -M -s /usr/sbin/nologin sandbox-user")
    );
    assert_eq!(
        error.context.get("stdout").map(String::as_str),
        Some("useradd unavailable")
    );
    assert_eq!(error.context.get("stderr").map(String::as_str), Some(""));
    assert!(
        error.message.ends_with("useradd unavailable"),
        "message: {}",
        error.message
    );
}

/// Python's `test_apply_entry_batch_flushes_parallel_work_before_overlapping_paths`: an entry
/// nested under another waits until the batch holding its ancestor has finished.
#[tokio::test]
async fn apply_entry_batch_flushes_parallel_work_before_overlapping_paths() {
    let session = RecordingSession::new();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("a-alpha.txt", Entry::text_file("a"))
        .with_entry("b-beta.txt", Entry::text_file("b"))
        .with_entry("c-nested", Entry::dir())
        .with_entry("c-nested/child.txt", Entry::text_file("c"));

    let result = manifest.apply(&session).await.expect("apply");

    assert_eq!(
        result.paths(),
        vec![
            "/workspace/a-alpha.txt",
            "/workspace/b-beta.txt",
            "/workspace/c-nested/child.txt",
        ]
    );
    let child_start = session.position("start /workspace/c-nested/child.txt");
    assert!(session.position("end /workspace/a-alpha.txt") < child_start);
    assert!(session.position("end /workspace/b-beta.txt") < child_start);
    assert!(session.position("end /workspace/c-nested") < child_start);
}

/// Python's `test_apply_entry_batch_flushes_before_and_after_mount_entries`: a mount settles
/// before anything written next to it starts, and what was queued before it finishes first.
#[tokio::test]
async fn apply_entry_batch_flushes_before_and_after_mount_entries() {
    let session = RecordingSession::new();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("a-alpha.txt", Entry::text_file("a"))
        .with_entry("b-beta.txt", Entry::text_file("b"))
        .with_entry("c-mount", Entry::mount(gcs_mount("sandbox-bucket")))
        .with_entry("d-gamma.txt", Entry::text_file("c"));

    let result = manifest.apply(&session).await.expect("apply");

    assert_eq!(
        result.paths(),
        vec![
            "/workspace/a-alpha.txt",
            "/workspace/b-beta.txt",
            "/workspace/d-gamma.txt",
        ]
    );
    let events = session.events();
    // `chmod` runs after every file is written, so the mount's own commands are the ones that
    // name the mount.
    let is_mount_command = |event: &String| event.starts_with("exec ") && event.contains("mount");
    let first_mount_command = events
        .iter()
        .position(is_mount_command)
        .expect("the mount ran a command");
    let last_mount_command = events
        .iter()
        .rposition(is_mount_command)
        .expect("the mount ran a command");
    assert!(session.position("end /workspace/a-alpha.txt") < first_mount_command);
    assert!(session.position("end /workspace/b-beta.txt") < first_mount_command);
    assert!(session.position("start /workspace/d-gamma.txt") > last_mount_command);
}

// ---------------------------------------------------------------------------
// test_manifest_config.py
// ---------------------------------------------------------------------------

/// Python's `test_configuration_preserves_trusted_manifest_host_sources`, for the two places a
/// Rust caller supplies a manifest: `SandboxRunConfig` and `SandboxAgent`. The third Python
/// fixture (`RunConfig(sandbox={...})`) has no Rust counterpart: `RunConfig` holds no sandbox
/// field here.
///
/// Host sources travel with a typed manifest untouched, so a trusted `LocalFile` keeps its path.
#[test]
fn configuration_preserves_trusted_manifest_host_sources() {
    let manifest = Manifest::new()
        .with_entry("file", Entry::local_file("input.txt"))
        .with_entry(
            "nested",
            Entry::dir().with_child("directory", Entry::local_dir("data")),
        );

    let from_run_config = SandboxRunConfig::default()
        .manifest(manifest.clone())
        .manifest
        .expect("run config manifest");
    let from_agent = SandboxAgent::<()>::new("test")
        .with_default_manifest(manifest.clone())
        .default_manifest
        .expect("agent default manifest");

    for configured in [&manifest, &from_run_config, &from_agent] {
        let file = &configured.entries()["file"];
        let EntryKind::LocalFile { src } = &file.kind else {
            panic!("expected a local file");
        };
        assert_eq!(src, std::path::Path::new("input.txt"));
        let EntryKind::Dir { children } = &configured.entries()["nested"].kind else {
            panic!("expected a dir");
        };
        let EntryKind::LocalDir { src } = &children["directory"].kind else {
            panic!("expected a local dir");
        };
        assert_eq!(src.as_deref(), Some(std::path::Path::new("data")));
    }
}
