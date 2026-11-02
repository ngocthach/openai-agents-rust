//! Ports of `tests/sandbox/test_unix_local_user_file_io.py`.
//!
//! Python runs a `sudo -u <user> python3 -I -S -c <worker>` body and mocks `os` and
//! `subprocess` around it. Rust runs a user's file operations as commands through `exec`, so the
//! contract is asserted two ways: on the commands a recording session receives (one dispatch,
//! as that user, with both paths resolved, and the command's stderr kept on failure), and on a
//! real Unix-local workspace behind a `sudo` stand-in that records its arguments and runs the
//! wrapped command as the current user.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine as _;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, User};

// ----------------------------------------------------------------------------------------------
// A session that records every command and answers each with a scripted result.
// ----------------------------------------------------------------------------------------------

#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    result: ExecResult,
    requests: Mutex<Vec<ExecRequest>>,
}

impl RecordingSession {
    fn new(exit_code: i32, stderr: &str) -> Arc<Self> {
        Arc::new(RecordingSession {
            manifest: Manifest::new().with_root("/workspace"),
            result: ExecResult {
                exit_code,
                stdout: Vec::new(),
                stderr: stderr.as_bytes().to_vec(),
            },
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<ExecRequest> {
        self.requests.lock().unwrap().clone()
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
        self.requests.lock().unwrap().push(request);
        Ok(self.result.clone())
    }
    async fn read_file(&self, _path: &str) -> SandboxResult<Vec<u8>> {
        unreachable!("a user operation never reads as the session owner")
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        unreachable!("a user operation never writes as the session owner")
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn user() -> User {
    User::new("example-user")
}

// ----------------------------------------------------------------------------------------------
// A `sudo` stand-in in front of a real Unix-local workspace.
// ----------------------------------------------------------------------------------------------

/// Install a `sudo` that logs one record per invocation and runs the command after `-u <user> --`.
fn fake_sudo(dir: &Path, log: &Path) {
    let script = format!(
        "#!/bin/sh\n\
         line=\"\"\n\
         for arg in \"$@\"; do line=\"$line$arg\u{1f}\"; done\n\
         printf '%s\\036' \"$line\" >> '{}'\n\
         shift 3\n\
         exec \"$@\"\n",
        log.display()
    );
    let path = dir.join("sudo");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn invocations(log: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .split('\u{1e}')
        .filter(|record| !record.is_empty())
        .map(|record| {
            record
                .trim_end_matches('\u{1f}')
                .split('\u{1f}')
                .map(str::to_string)
                .collect()
        })
        .collect()
}

/// A started Unix-local session over `<tmp>/workspace` whose `PATH` finds the stand-in first.
/// The log lives inside the workspace (the macOS wrapper confines commands to it) under `logs/`.
struct SudoWorkspace {
    _dir: tempfile::TempDir,
    workspace: std::path::PathBuf,
    log: std::path::PathBuf,
    session: Arc<dyn SandboxSession>,
}

async fn sudo_workspace() -> SudoWorkspace {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    let bin = dir.path().join("bin");
    std::fs::create_dir_all(workspace.join("logs")).unwrap();
    std::fs::create_dir(&bin).unwrap();
    let log = workspace.join("logs/sudo.log");
    fake_sudo(&bin, &log);
    let session = UnixLocalSandboxClient::new()
        .create(Some(
            Manifest::new()
                .with_root(workspace.to_string_lossy().to_string())
                .with_env("PATH", format!("{}:/usr/bin:/bin", bin.display())),
        ))
        .await
        .expect("created");
    session.start().await.expect("started");
    SudoWorkspace {
        _dir: dir,
        workspace,
        log,
        session,
    }
}

// ----------------------------------------------------------------------------------------------
// Writes
// ----------------------------------------------------------------------------------------------

/// Python: `test_user_write_uses_shared_traversal_and_preserves_input`. Binary contents (a NUL
/// and a non-UTF-8 byte) reach the file exactly, every command runs as the user, and the staged
/// file is moved into place last. (The descriptor-level traversal is the Python worker's own.)
#[tokio::test]
async fn a_user_write_carries_binary_contents_exactly() {
    let session = RecordingSession::new(0, "");
    let data: &[u8] = b"a\x00b\xff";

    session
        .write_file_as("parent/file", data, Some(&user()))
        .await
        .expect("the write succeeds");

    assert_eq!(data, b"a\x00b\xff");
    let requests = session.requests();
    assert!(requests.iter().all(|r| r.user.as_ref() == Some(&user())));
    let encoded = base64::engine::general_purpose::STANDARD.encode(data);
    assert!(
        requests
            .iter()
            .any(|r| r.command.iter().any(|part| part.contains(&encoded))),
        "{requests:?}"
    );
    let last = requests.last().expect("a command").command.join(" ");
    assert!(last.contains("mv --"), "{last}");
    assert!(last.contains("/workspace/parent/file"), "{last}");
}

/// Python: `test_user_lookup_failure_stops_before_file_io[write-*]` and
/// `test_user_leaf_permission_failure_does_not_write`: a write the user's command refuses
/// reports a write error and moves nothing into place.
#[tokio::test]
async fn a_refused_user_write_reports_a_write_error_and_moves_nothing() {
    for stderr in [
        "Permission denied",
        "Too many levels of symbolic links",
        "File is not writable",
    ] {
        let session = RecordingSession::new(1, stderr);

        let error = session
            .write_file_as("parent/file", b"value", Some(&user()))
            .await
            .expect_err("the user's command fails");

        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{stderr}"
        );
        let requests = session.requests();
        assert!(requests.iter().all(|r| r.user.as_ref() == Some(&user())));
        assert!(
            !requests
                .iter()
                .any(|r| r.command.iter().any(|part| part.contains("mv --"))),
            "{stderr}: {requests:?}"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Rename
// ----------------------------------------------------------------------------------------------

/// Python: `test_user_rename_runs_in_the_worker_relative_to_both_parents`. One dispatch, as the
/// user, naming both workspace paths in full.
#[tokio::test]
async fn a_user_rename_is_one_dispatch_naming_both_paths() {
    let session = RecordingSession::new(0, "");

    session
        .mv_as("old/file", "new/File", Some(&user()))
        .await
        .expect("the rename succeeds");

    let requests = session.requests();
    assert_eq!(requests.len(), 1, "{requests:?}");
    assert_eq!(requests[0].user.as_ref(), Some(&user()));
    let tail = &requests[0].command[requests[0].command.len() - 2..];
    assert_eq!(tail, ["/workspace/old/file", "/workspace/new/File"]);
}

/// Python: `test_user_rename_failure_keeps_the_error`.
#[tokio::test]
async fn a_failed_user_rename_keeps_the_error() {
    let session = RecordingSession::new(1, "Is a directory");

    let error = session
        .mv_as("file", "docs", Some(&user()))
        .await
        .expect_err("the rename fails");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert!(
        error
            .context
            .get("stderr")
            .is_some_and(|stderr| stderr.contains("Is a directory")),
        "{error:?}"
    );
    assert_eq!(session.requests().len(), 1);
}

// ----------------------------------------------------------------------------------------------
// Same file
// ----------------------------------------------------------------------------------------------

/// Python: `test_user_same_file_asks_the_worker` (`same-entry`, `different-entries`,
/// `link-not-followed`). The answer comes from one command run as the user against the real
/// workspace.
#[tokio::test]
async fn a_user_same_file_check_asks_the_sandbox_as_that_user() {
    for (case, follow_symlinks, expected) in [
        ("same-entry", true, true),
        ("different-entries", true, false),
        ("link-not-followed", false, false),
    ] {
        let ws = sudo_workspace().await;
        std::fs::write(ws.workspace.join("left"), "left").unwrap();
        match case {
            "same-entry" => {
                std::fs::hard_link(ws.workspace.join("left"), ws.workspace.join("right")).unwrap()
            }
            "different-entries" => std::fs::write(ws.workspace.join("right"), "right").unwrap(),
            _ => std::os::unix::fs::symlink("left", ws.workspace.join("right")).unwrap(),
        }

        let answer = ws
            .session
            .same_file("left", "right", follow_symlinks, Some(&user()))
            .await
            .unwrap_or_else(|e| panic!("{case}: {e}"));

        assert_eq!(answer, expected, "{case}");
        let calls = invocations(&ws.log);
        assert_eq!(calls.len(), 1, "{case}: {calls:?}");
        assert_eq!(&calls[0][..3], ["-u", "example-user", "--"], "{case}");
        let names: Vec<&str> = calls[0][calls[0].len() - 2..]
            .iter()
            .map(|p| p.rsplit('/').next().unwrap_or(p))
            .collect();
        assert_eq!(names, ["left", "right"], "{case}");
        assert_eq!(
            calls[0].iter().any(|part| part.contains("-L")),
            !follow_symlinks,
            "{case}: {calls:?}"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Listing
// ----------------------------------------------------------------------------------------------

/// Python: `test_user_lookup_failure_stops_before_file_io[ls-*]` and the `ls` legs of
/// `test_missing_system_python_does_not_retry_with_application_runtime` and
/// `test_user_file_operation_does_not_use_application_runtime`: a listing runs once, as the
/// user, and a refusal is an exec error carrying the command's stderr.
#[tokio::test]
async fn a_refused_user_listing_is_one_dispatch_and_keeps_the_stderr() {
    for stderr in [
        "ls: parent/file: Permission denied",
        "ls: parent/file: Too many levels of symbolic links",
    ] {
        let session = RecordingSession::new(1, stderr);

        let error = session
            .ls_as("parent/file", Some(&user()))
            .await
            .expect_err("the listing fails");

        assert_eq!(error.code, ErrorCode::ExecNonZero, "{stderr}");
        assert!(error.message.contains(stderr), "{}", error.message);
        let requests = session.requests();
        assert_eq!(requests.len(), 1, "{stderr}");
        assert_eq!(requests[0].user.as_ref(), Some(&user()));
        assert_eq!(
            requests[0].command,
            ["ls", "-la", "--", "/workspace/parent/file"]
        );
    }
}

/// Python: `test_user_file_operation_does_not_use_application_runtime[ls]`: an empty listing is
/// one `ls` run as the user in the sandbox, never through the host process.
#[tokio::test]
async fn a_user_listing_runs_only_as_that_user_in_the_sandbox() {
    let session = RecordingSession::new(0, "");

    let listing = session
        .ls_as("directory", Some(&user()))
        .await
        .expect("the listing succeeds");

    assert!(listing.is_empty());
    let requests = session.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].user.as_ref(), Some(&user()));
    assert_eq!(
        requests[0].resolved_argv(),
        [
            "sudo",
            "-u",
            "example-user",
            "--",
            "ls",
            "-la",
            "--",
            "/workspace/directory"
        ]
    );
}

/// Python: `test_user_listing_preserves_metadata_and_names` (`symlink`, `socket`, `directory`).
/// On the Unix-local backend the user's own `ls` gates the listing and the entries are then read
/// natively, so a name holding a newline and spaces is reported exactly, with its kind and size.
/// (Rust's `FileEntry` has no owner or group names.)
#[tokio::test]
async fn a_user_listing_preserves_names_and_kinds() {
    let name = "name\nwith spaces";
    for kind in ["symlink", "socket", "directory"] {
        let ws = sudo_workspace().await;
        let parent = ws.workspace.join("parent");
        std::fs::create_dir(&parent).unwrap();
        let entry = parent.join(name);
        // A socket path must stay short, so that case binds through a relative path.
        let _listener = match kind {
            "symlink" => {
                std::os::unix::fs::symlink("elsewhere", &entry).unwrap();
                None
            }
            "directory" => {
                std::fs::create_dir(&entry).unwrap();
                None
            }
            _ => {
                let short = tempfile::tempdir_in("/tmp").unwrap();
                let socket = short.path().join("s");
                let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
                std::fs::rename(&socket, &entry).unwrap();
                Some((listener, short))
            }
        };

        let listing = ws
            .session
            .ls_as("parent", Some(&user()))
            .await
            .unwrap_or_else(|e| panic!("{kind}: {e}"));

        let found: Vec<_> = listing.iter().filter(|e| e.name == name).collect();
        assert_eq!(found.len(), 1, "{kind}: {listing:?}");
        let found = found[0];
        assert!(
            found.path.ends_with("/parent/name\nwith spaces"),
            "{kind}: {found:?}"
        );
        assert_eq!(found.is_symlink, kind == "symlink", "{kind}");
        assert_eq!(found.is_dir, kind == "directory", "{kind}");
        let calls = invocations(&ws.log);
        assert_eq!(calls.len(), 1, "{kind}: {calls:?}");
        assert_eq!(
            &calls[0][..6],
            ["-u", "example-user", "--", "ls", "-la", "--"]
        );
    }
}
