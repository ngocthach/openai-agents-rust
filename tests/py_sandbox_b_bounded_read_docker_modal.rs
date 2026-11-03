//! Ports of the Docker and Modal rows of `tests/sandbox/test_bounded_read.py`.
//!
//! Docker is observed through a recording [`DockerRunner`]; Modal through the bundled helper
//! script run against a stub `modal` package, which is the closest Rust equivalent of Python's
//! mocked `sandbox.open` descriptor. The Modal tests need `python3` and are skipped without it.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::extensions::sandbox::modal::ModalHelper;
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, ExecResult, Manifest, SandboxResult, SandboxSession};
use serde_json::{Value, json};

#[derive(Debug, Default)]
struct Recorder {
    calls: Mutex<Vec<Vec<String>>>,
    stdout: Vec<u8>,
    exit_code: i32,
}

#[async_trait]
impl DockerRunner for Recorder {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        // Path resolution ahead of the read succeeds; only the read itself uses the exit code.
        let is_read = argv.iter().any(|arg| arg.contains("head -c"));
        Ok(ExecResult {
            stdout: if is_read {
                self.stdout.clone()
            } else {
                // The path resolver (run as `<container> <helper> <root> <path> <mode>`) prints
                // the resolved path; every other setup command prints nothing.
                let container = argv.iter().position(|arg| arg == "abc123").unwrap();
                match argv.get(container + 4) {
                    Some(mode) if mode == "0" || mode == "1" => {
                        argv[container + 3].clone().into_bytes()
                    }
                    _ => Vec::new(),
                }
            },
            stderr: Vec::new(),
            exit_code: if is_read { self.exit_code } else { 0 },
        })
    }
}

/// Python: `test_docker_bounded_read_uses_trusted_utilities_without_encoding`.
#[tokio::test]
async fn docker_bounded_read_uses_trusted_utilities_without_encoding() {
    let runner = Arc::new(Recorder {
        stdout: b"\x00\xffabc".to_vec(),
        ..Recorder::default()
    });
    let session = DockerSandboxSession::new("abc123", Manifest::new()).with_runner(runner.clone());

    let bytes = session.read_bounded("out.jsonl", 5).await.unwrap();

    assert_eq!(bytes, b"\x00\xffabc");
    let calls = runner.calls.lock().unwrap();
    let argv = calls.last().expect("an exec");
    let at = argv
        .iter()
        .position(|arg| arg == "/bin/sh")
        .expect("the trusted shell");
    let tail = &argv[at..];
    assert_eq!(tail[1], "-c");
    assert!(tail[2].contains("PATH=/usr/bin:/bin; export PATH"));
    assert!(!tail[2].contains("base64"));
    assert!(tail[tail.len() - 2].ends_with("/out.jsonl"));
    assert_eq!(tail[tail.len() - 1], "5");
}

#[tokio::test]
async fn docker_bounded_read_classifies_missing_and_failing_reads() {
    for (exit_code, code) in [
        (44, ErrorCode::WorkspaceReadNotFound),
        (45, ErrorCode::WorkspaceArchiveReadError),
    ] {
        let runner = Arc::new(Recorder {
            exit_code,
            ..Recorder::default()
        });
        let session = DockerSandboxSession::new("abc123", Manifest::new()).with_runner(runner);
        let error = session.read_bounded("out.jsonl", 5).await.unwrap_err();
        assert_eq!(error.code, code);
    }
    let session = DockerSandboxSession::new("abc123", Manifest::new());
    let error = session.read_bounded("out.jsonl", 0).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}

const STUB: &str = r#"
import asyncio, json, os


class _Stream:
    def __init__(self):
        self.data = b"x" * int(os.environ.get("STUB_SIZE", "3"))
        self.mode = os.environ.get("STUB_MODE", "ok")

    class _Op:
        def __init__(self, fn):
            self.aio = fn

    @property
    def read(self):
        async def read(size):
            with open(os.environ["STUB_LOG"], "a") as log:
                log.write(f"read {size}\n")
            if self.mode.startswith("read_fail"):
                raise _error(self.mode)
            if self.mode.startswith("hang"):
                # A read that never completes: only a cancellation ends it.
                await asyncio.sleep(3600)
            chunk, self.data = self.data[:size], self.data[size:]
            return chunk
        return _Stream._Op(read)

    @property
    def close(self):
        async def close():
            with open(os.environ["STUB_LOG"], "a") as log:
                log.write("close\n")
            if self.mode.endswith("close_fail"):
                raise RuntimeError("synthetic-private-response")
            if self.mode.endswith("close_timeout"):
                raise asyncio.TimeoutError("synthetic-private-response")
        return _Stream._Op(close)


class InternalError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


def _error(mode):
    return InternalError("synthetic-private-response") if mode == "read_fail_retryable" else PermissionDeniedError("synthetic-private-response")


class _Sandbox:
    class _Open:
        async def aio(self, path, mode):
            return _Stream()

    open = _Open()


class Sandbox:
    class from_id:
        @staticmethod
        async def aio(sandbox_id):
            return _Sandbox()
"#;

struct Stub {
    directory: tempfile::TempDir,
}

impl Stub {
    fn new() -> Option<Stub> {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .ok()?;
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("modal")).unwrap();
        std::fs::write(directory.path().join("modal/__init__.py"), STUB).unwrap();
        Some(Stub { directory })
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.directory.path().join("log")).unwrap_or_default()
    }

    fn helper(&self, mode: &str, size: usize) -> ModalHelper {
        ModalHelper::new(vec![
            "/usr/bin/env".to_string(),
            format!("PYTHONPATH={}", self.directory.path().display()),
            format!("STUB_LOG={}", self.directory.path().join("log").display()),
            format!("STUB_MODE={mode}"),
            format!("STUB_SIZE={size}"),
            "python3".to_string(),
            format!(
                "{}/src/extensions/sandbox/modal/helper.py",
                env!("CARGO_MANIFEST_DIR")
            ),
        ])
    }

    async fn read_bounded(
        &self,
        mode: &str,
        size: usize,
        max_bytes: usize,
    ) -> SandboxResult<Vec<u8>> {
        let helper = self.helper(mode, size);
        let response: Value = helper
            .call(json!({
                "op": "read_bounded", "sandbox_id": "sb-1",
                "path": "/workspace/out.jsonl", "max_bytes": max_bytes,
            }))
            .await?;
        use base64::Engine;
        Ok(base64::engine::general_purpose::STANDARD
            .decode(response["data"].as_str().unwrap())
            .unwrap())
    }
}

/// Python: `test_modal_bounded_read_closes_provider_descriptor`.
#[tokio::test]
async fn modal_bounded_read_closes_the_descriptor() {
    let Some(stub) = Stub::new() else { return };
    assert_eq!(stub.read_bounded("ok", 3, 3).await.unwrap(), b"xxx");
    assert_eq!(stub.log(), "read 3\nclose\n");
}

#[tokio::test]
async fn modal_close_failure_fails_a_completed_read_without_leaking() {
    for mode in ["close_fail", "close_timeout"] {
        let Some(stub) = Stub::new() else { return };
        let error = stub.read_bounded(mode, 3, 3).await.unwrap_err();
        assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("bounded_read_failed")
        );
        assert!(!format!("{error:?}").contains("synthetic-private-response"));
        assert!(stub.log().ends_with("close\n"));
    }
}

/// Python: `test_bounded_provider_read_preserves_retry_policy[modal-*]`.
#[tokio::test]
async fn modal_bounded_read_preserves_retry_policy() {
    for (mode, retryable) in [("read_fail_retryable", true), ("read_fail_denied", false)] {
        let Some(stub) = Stub::new() else { return };
        let error = stub.read_bounded(mode, 3, 3).await.unwrap_err();
        assert_eq!(error.retryable, Some(retryable), "{mode}");
        assert!(!format!("{error:?}").contains("synthetic-private-response"));
        assert!(
            stub.log().ends_with("close\n"),
            "the descriptor is closed after a read failure"
        );
    }
}

/// Python: `test_modal_bounded_read_uses_large_bounded_requests`.
#[tokio::test]
async fn modal_bounded_read_uses_large_bounded_requests() {
    const MIB: usize = 1024 * 1024;
    for (limit, expected) in [
        (4 * MIB, vec![4 * MIB]),
        (8 * MIB + 1, vec![8 * MIB + 1, 4 * MIB + 1]),
        (101 * MIB, vec![100 * MIB, 97 * MIB]),
    ] {
        let Some(stub) = Stub::new() else { return };
        let bytes = stub.read_bounded("ok", 4 * MIB, limit).await.unwrap();
        assert_eq!(bytes.len(), 4 * MIB);
        let requests: Vec<usize> = stub
            .log()
            .lines()
            .filter_map(|line| line.strip_prefix("read "))
            .map(|n| n.parse().unwrap())
            .collect();
        assert_eq!(requests, expected, "limit {limit}");
    }
}

/// Python: `test_modal_bounded_read_cancellation_closes_descriptor`. A dropped read is a
/// cancelled operation: the helper gets SIGTERM, its `finally` closes the descriptor, and the
/// close error (none, a failure, or a timeout) does not change that.
#[tokio::test]
async fn modal_cancelled_read_closes_the_descriptor() {
    for mode in ["hang", "hang_close_fail", "hang_close_timeout"] {
        let Some(stub) = Stub::new() else { return };
        let stub = Arc::new(stub);
        let helper = stub.helper(mode, 3);
        let task = tokio::spawn(async move {
            helper
                .call(json!({
                    "op": "read_bounded", "sandbox_id": "sb-1",
                    "path": "/workspace/out.jsonl", "max_bytes": 3,
                }))
                .await
        });
        // The read has started once the helper logs it.
        let started = wait_until(|| stub.log().contains("read 3")).await;
        assert!(started, "{mode}: the helper started the read");

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled(), "{mode}");

        let closed = wait_until(|| stub.log().contains("close")).await;
        assert!(closed, "{mode}: the cancelled helper closed the descriptor");
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..3000 {
        if condition() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    false
}
