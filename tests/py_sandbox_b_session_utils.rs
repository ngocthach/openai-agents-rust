//! Port of `tests/sandbox/test_session_utils.py`.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::events::{EventPhase, SandboxSessionEvent};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, GcsMount};
use openai_agents_sdk::sandbox::mounts::Mount;
use openai_agents_sdk::sandbox::session::{
    ACCESS_PROBE_TIMEOUT, ExecRequest, ExecShell, READ_PATH_PROBE_SCRIPT, SandboxSession,
    safe_decode,
};
use openai_agents_sdk::sandbox::types::{ExecResult, User};

/// Python: `_CaptureExecSession` / `_QueuedExecSession`.
#[derive(Debug)]
struct ExecSession {
    manifest: Manifest,
    results: Mutex<Vec<ExecResult>>,
    requests: Mutex<Vec<ExecRequest>>,
}

impl ExecSession {
    fn new() -> Self {
        Self::queued(Vec::new())
    }

    fn queued(results: Vec<ExecResult>) -> Self {
        ExecSession {
            manifest: Manifest::new().with_root("/workspace"),
            results: Mutex::new(results),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn with_manifest(mut self, manifest: Manifest) -> Self {
        self.manifest = manifest;
        self
    }

    fn argvs(&self) -> Vec<Vec<String>> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(ExecRequest::resolved_argv)
            .collect()
    }

    fn timeouts(&self) -> Vec<Option<Duration>> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.timeout)
            .collect()
    }
}

#[async_trait]
impl SandboxSession for ExecSession {
    fn id(&self) -> &str {
        "exec-session"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "exec-session"
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
        self.requests.lock().unwrap().push(request);
        let mut results = self.results.lock().unwrap();
        if results.is_empty() {
            return Ok(ExecResult::default());
        }
        Ok(results.remove(0))
    }
    async fn read_file(&self, _path: &str) -> SandboxResult<Vec<u8>> {
        panic!("read() should not be called in this test");
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        panic!("write() should not be called in this test");
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn result(stdout: &[u8], stderr: &[u8], exit_code: i32) -> ExecResult {
    ExecResult {
        stdout: stdout.to_vec(),
        stderr: stderr.to_vec(),
        exit_code,
    }
}

/// Python: `test_safe_decode_truncates_and_appends_ellipsis`.
#[test]
fn safe_decode_truncates_and_appends_an_ellipsis() {
    assert_eq!(safe_decode(b"abcdef", 3), "abc\u{2026}");
    assert_eq!(safe_decode(b"abc", 3), "abc");
}

/// Python: `test_event_to_json_line_is_single_line`.
#[test]
fn an_event_serializes_to_a_single_json_line() {
    let event =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_write", SandboxOp::Write)
            .with_data("x", "1");

    let line = event.to_json_line().unwrap();

    assert!(line.ends_with('\n'));
    assert!(!line[..line.len() - 1].contains('\n'));
}

/// Python: `test_validate_sandbox_session_event_uses_phase_discriminator`.
#[test]
fn an_event_round_trips_through_its_phase_discriminator() {
    let event =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_read", SandboxOp::Read);

    let restored: SandboxSessionEvent =
        serde_json::from_str(&event.to_json_line().unwrap()).unwrap();

    assert_eq!(restored.phase, EventPhase::Start);
    assert_eq!(restored.op, "read");
    let finish: SandboxSessionEvent =
        serde_json::from_value(serde_json::to_value(event.finish(true, 0)).unwrap()).unwrap();
    assert_eq!(finish.phase, EventPhase::Finish);
}

/// Python: `test_sandbox_session_finish_event_excludes_raw_bytes_from_json_dump`. Rust events
/// carry decoded, bounded text only when the payload policy allows it, never the raw byte
/// buffers, so the dump of a finish event holds no `*_bytes` field at all.
#[test]
fn a_finish_event_excludes_raw_bytes_from_its_json_dump() {
    let event =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_exec", SandboxOp::Exec)
            .finish(true, 0);

    let dumped = serde_json::to_value(&event).unwrap();

    assert!(dumped.get("stdout_bytes").is_none());
    assert!(dumped.get("stderr_bytes").is_none());
}

/// Python: `test_exec_shell_true_quotes_multi_arg_commands`.
#[test]
fn a_login_shell_command_quotes_multiple_arguments() {
    let request = ExecRequest::argv(["printf", "%s\n", "hello world", "$(whoami)", "semi;colon"])
        .with_shell(ExecShell::Login);

    let argv = request.resolved_argv();

    assert_eq!(&argv[..2], ["sh", "-lc"]);
    assert_eq!(
        argv[2],
        "printf '%s\n' 'hello world' '$(whoami)' 'semi;colon'"
    );
}

/// Python: `test_exec_shell_true_preserves_single_shell_snippet`.
#[test]
fn a_login_shell_command_keeps_a_single_snippet_as_is() {
    let request = ExecRequest::shell("echo hello && echo goodbye");

    assert_eq!(
        request.resolved_argv(),
        ["sh", "-lc", "echo hello && echo goodbye"]
    );
}

/// Python: `test_check_mkdir_with_exec_runs_non_destructive_probe_as_user`. Rust reports success
/// as `Ok(())`; the command the probe ran is what is checked.
#[tokio::test]
async fn the_mkdir_check_runs_a_non_destructive_probe_as_the_user() {
    let session = ExecSession::new();

    session
        .check_mkdir_access("nested/dir", true, Some(&User::new("sandbox-user")))
        .await
        .expect("the probe passes");

    let argv = &session.argvs()[0];
    assert_eq!(&argv[..4], ["sudo", "-u", "sandbox-user", "--"]);
    assert_eq!(&argv[4..6], ["sh", "-c"]);
    assert_eq!(&argv[argv.len() - 2..], ["nested/dir", "1"]);
}

/// Python: `test_check_rm_with_exec_runs_parent_write_probe_as_user`.
#[tokio::test]
async fn the_rm_check_runs_a_parent_write_probe_as_the_user() {
    let session = ExecSession::new();

    session
        .check_rm_access("stale.txt", false, Some(&User::new("sandbox-user")))
        .await
        .expect("the probe passes");

    let argv = &session.argvs()[0];
    assert_eq!(&argv[..4], ["sudo", "-u", "sandbox-user", "--"]);
    assert_eq!(&argv[4..6], ["sh", "-c"]);
    assert_eq!(&argv[argv.len() - 2..], ["stale.txt", "0"]);
}

/// Python: `test_check_read_with_exec_classifies_failure_as_requested_user` (3 parameter sets).
#[tokio::test]
async fn the_read_check_classifies_a_failure_as_the_requested_user() {
    for (probe_exit_code, expected) in [
        (0, ErrorCode::WorkspaceArchiveReadError),
        (1, ErrorCode::WorkspaceReadNotFound),
        (2, ErrorCode::WorkspaceArchiveReadError),
    ] {
        let session = ExecSession::queued(vec![
            result(b"", b"not readable", 1),
            result(b"", b"", probe_exit_code),
        ]);

        let error = session
            .check_read_access("target.txt", Some(&User::new("sandbox-user")))
            .await
            .expect_err("the read fails");

        assert_eq!(error.code, expected, "probe exit {probe_exit_code}");
        let argvs = session.argvs();
        assert_eq!(argvs.len(), 2);
        for argv in &argvs {
            assert_eq!(&argv[..4], ["sudo", "-u", "sandbox-user", "--"]);
        }
        assert!(argvs[1][6].contains("READ_PATH_PROBE_V3"));
        assert_eq!(session.timeouts(), vec![None, Some(ACCESS_PROBE_TIMEOUT)]);
    }
}

/// Python: `test_check_read_with_exec_treats_nonstandard_check_exit_as_archive_error`.
#[tokio::test]
async fn a_nonstandard_read_check_exit_is_an_archive_error() {
    let session = ExecSession::queued(vec![result(b"", b"check failed", 127)]);

    let error = session
        .check_read_access("target.txt", None)
        .await
        .expect_err("the check fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(session.argvs().len(), 1);
}

/// Python: `test_read_error_context_does_not_retain_partial_stdout`.
#[tokio::test]
async fn the_read_error_context_does_not_retain_partial_stdout() {
    let partial_stdout = b"sensitive partial contents".repeat(1024);
    let session = ExecSession::queued(vec![
        result(&partial_stdout, b"read failed", 1),
        result(b"", b"", 2),
    ]);

    let error = session
        .check_read_access("target.txt", None)
        .await
        .expect_err("the read fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert!(!error.context.contains_key("stdout"));
    assert_eq!(
        error.context.get("stdout_bytes").map(String::as_str),
        Some(partial_stdout.len().to_string().as_str())
    );
    assert!(!error.message.contains("sensitive"));
}

/// Python: `test_read_path_probe_resolves_symlinks_before_classifying_missing`. The probe script
/// is shared with Python, so it is run for real against a tree of tricky symlinks.
#[cfg(unix)]
#[test]
fn the_read_path_probe_resolves_symlinks_before_classifying_missing() {
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    symlink("missing", workspace.join("dangling")).unwrap();
    std::fs::write(workspace.join("not-a-directory"), "content").unwrap();
    symlink("not-a-directory/child", workspace.join("invalid-target")).unwrap();
    symlink("missing-directory", workspace.join("dangling-parent")).unwrap();
    symlink("not-a-directory", workspace.join("invalid-parent")).unwrap();
    symlink("loop", workspace.join("loop")).unwrap();
    std::fs::write(workspace.join("newline-target\n"), "content").unwrap();
    symlink("newline-target\n", workspace.join("newline-link")).unwrap();
    std::fs::write(workspace.join("a"), "sibling").unwrap();
    let mut current = workspace.clone();
    let mut symlink_parts = Vec::new();
    for index in 0..41 {
        let real = current.join(format!("real-{index}"));
        std::fs::create_dir(&real).unwrap();
        symlink(
            format!("real-{index}"),
            current.join(format!("link-{index}")),
        )
        .unwrap();
        symlink_parts.push(format!("link-{index}"));
        current = real;
    }

    let probe = |path: &Path, env: Option<(&str, String)>| -> i32 {
        let mut command = Command::new("sh");
        command.args(["-c", READ_PATH_PROBE_SCRIPT, "sh"]).arg(path);
        if let Some((key, value)) = env {
            command.env(key, value);
        }
        command.output().unwrap().status.code().unwrap_or(-1)
    };

    let mut deep = workspace.clone();
    deep.extend(&symlink_parts);
    deep.push("missing");
    let cases: Vec<(PathBuf, i32)> = vec![
        (workspace.join("dangling"), 1),
        (workspace.join("invalid-target"), 2),
        (workspace.join("dangling-parent").join("child"), 1),
        (workspace.join("invalid-parent").join("child"), 2),
        (workspace.join("loop"), 2),
        (workspace.join("newline-link"), 0),
        (workspace.join("[a]"), 1),
        (workspace.join("?"), 1),
        (workspace.join("*"), 1),
        (deep, 2),
        (workspace.join("x".repeat(256)), 2),
    ];
    for (path, expected) in &cases {
        assert_eq!(
            probe(path, None),
            *expected,
            "unexpected probe result for {path:?}"
        );
    }

    // A `find` that fails or prints something must not change the verdict for a missing path.
    let fake_bin = root.join("fake-bin");
    std::fs::create_dir(&fake_bin).unwrap();
    let args_log = root.join("find-args.log");
    let fake_find = fake_bin.join("find");
    std::fs::write(
        &fake_find,
        "#!/bin/sh\nprintf \"%s\\n\" \"$@\" > \"$FIND_ARGS_LOG\"\nprintf \"find: %s: Input/output error\\n\" \"$1\" >&2\nexit 1\n",
    )
    .unwrap();
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&fake_find, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path_env = format!("{}:{}", fake_bin.display(), std::env::var("PATH").unwrap());
    let missing = workspace.join("missing");
    let run = |path_env: String| {
        let mut command = Command::new("sh");
        command
            .args(["-c", READ_PATH_PROBE_SCRIPT, "sh"])
            .arg(&missing)
            .env("PATH", path_env)
            .env("FIND_ARGS_LOG", &args_log);
        command.output().unwrap().status.code().unwrap_or(-1)
    };
    assert_eq!(run(path_env.clone()), 2);
    let logged = std::fs::read_to_string(&args_log).unwrap();
    assert_eq!(
        logged.lines().collect::<Vec<_>>(),
        [missing.to_string_lossy().as_ref(), "-prune", "-print"]
    );
    std::fs::write(&fake_find, "#!/bin/sh\nprintf match\n").unwrap();
    assert_eq!(run(path_env), 2);
}

fn gcs_mount_at(path: &str) -> Entry {
    let strategy = ConfiguredMountStrategy::mountpoint(GcsMount::new("bucket"));
    Entry::mount(Mount::new(Arc::new(strategy)).with_mount_path(path))
}

/// Python: `test_register_persist_workspace_skip_path_rejects_mount_overlaps` (3 sets).
#[test]
fn registering_a_skip_path_rejects_mount_overlaps() {
    for (skip_path, mount_path) in [
        ("data", "data"),
        ("logs", "logs/remote"),
        ("data/tmp", "data"),
    ] {
        let session = ExecSession::new().with_manifest(
            Manifest::new()
                .with_root("/workspace")
                .with_entry("remote", gcs_mount_at(mount_path)),
        );

        let error = session
            .register_persist_workspace_skip_path(skip_path)
            .expect_err("an overlap is refused");

        assert_eq!(
            error.code,
            ErrorCode::MountConfigInvalid,
            "{skip_path} / {mount_path}"
        );
        assert_eq!(
            error.message,
            "persist workspace skip path must not overlap mount path"
        );
    }
}

/// Python: `test_register_persist_workspace_skip_path_allows_non_overlapping_path`.
#[test]
fn registering_a_skip_path_allows_a_non_overlapping_path() {
    let session = ExecSession::new().with_manifest(
        Manifest::new()
            .with_root("/workspace")
            .with_entry("remote", gcs_mount_at("data")),
    );

    let registered = session
        .register_persist_workspace_skip_path("logs/events.jsonl")
        .unwrap();

    assert_eq!(registered, "logs/events.jsonl");
}

/// Python: `test_best_effort_stream_len_*` (2 tests) and `test_file_entry_is_dir_uses_kind`.
/// The stream-length helpers size a Python file-like object, which Rust sessions do not take
/// (writes are `&[u8]`); `FileEntry.is_dir` is a plain field set from the listing, which the
/// listing parser tests in `py_sandbox_b_pty.rs` cover.
#[test]
fn a_listing_marks_directories_by_kind() {
    use openai_agents_sdk::sandbox::types::{ListingKind, try_parse_ls_la};
    let listing = try_parse_ls_la(
        "drwxr-xr-x 2 root root 0 Jan 1 00:00 dir\n-rw-r--r-- 1 root root 3 Jan 1 00:00 file.txt\n",
        "/workspace",
    )
    .unwrap();

    assert_eq!(listing[0].kind, ListingKind::Directory);
    assert!(listing[0].entry.is_dir);
    assert_eq!(listing[1].kind, ListingKind::File);
    assert!(!listing[1].entry.is_dir);
}
