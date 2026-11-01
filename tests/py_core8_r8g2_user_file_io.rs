//! What a file write on behalf of another user must and must not do, from
//! `tests/sandbox/test_unix_local_user_file_io.py`.
//!
//! Python runs a `sudo -u <user> python3 -I -S -c <worker>` body and mocks `os` and
//! `subprocess`. Rust runs the user's file operations as commands through `exec`, so the
//! observable contract is asserted on the commands a recording session receives: every one is
//! issued as that user, none falls back to the session owner or to the application's own runtime,
//! and a failure reports the command's stderr instead of being retried.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, User};

#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    result: ExecResult,
    requests: Mutex<Vec<ExecRequest>>,
    owner_writes: Mutex<usize>,
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
            owner_writes: Mutex::new(0),
        })
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
        unreachable!("a user write never reads as the session owner")
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        *self.owner_writes.lock().unwrap() += 1;
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn user() -> User {
    User::new("example-user")
}

/// Python: `test_missing_user_runtime_reports_error_without_fallback` and the `write` leg of
/// `test_missing_system_python_does_not_retry_with_application_runtime`: when the user's command
/// fails, the write reports a write error that carries the stderr, nothing is moved into place,
/// and the session owner never writes the file instead.
#[tokio::test]
async fn a_failed_user_write_reports_the_stderr_and_does_not_fall_back() {
    for stderr in [
        "Python: permission denied",
        "sudo: python3: command not found",
    ] {
        let session = RecordingSession::new(1, stderr);

        let error = session
            .write_file_as("file", b"value", Some(&user()))
            .await
            .expect_err("the user's command fails");

        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{stderr}"
        );
        assert!(error.message.contains(stderr), "{}", error.message);
        let requests = session.requests.lock().unwrap();
        assert!(!requests.is_empty());
        for request in requests.iter() {
            assert_eq!(request.user.as_ref(), Some(&user()), "{stderr}");
            assert!(
                !request.command.iter().any(|part| part.contains("mv --")),
                "{stderr}: nothing is moved into place after a failure"
            );
        }
        assert_eq!(*session.owner_writes.lock().unwrap(), 0, "{stderr}");
    }
}

/// Python: `test_user_file_operation_does_not_use_application_runtime[write]`. Every command a
/// user write issues runs as that user through the sandbox's own shell, never the owner's
/// process, and the contents travel as data rather than being written by the host.
#[tokio::test]
async fn a_user_write_runs_only_as_that_user_through_the_sandbox_shell() {
    let session = RecordingSession::new(0, "");

    session
        .write_file_as("file", b"value", Some(&user()))
        .await
        .expect("the write succeeds");

    let requests = session.requests.lock().unwrap();
    assert!(!requests.is_empty());
    for request in requests.iter() {
        assert_eq!(request.user.as_ref(), Some(&user()));
        assert_eq!(request.command.first().map(String::as_str), Some("sh"));
    }
    assert!(
        requests.iter().any(|request| request
            .command
            .iter()
            .any(|part| part.contains("base64 -d"))),
        "the contents are decoded inside the sandbox"
    );
    assert_eq!(*session.owner_writes.lock().unwrap(), 0);
}
