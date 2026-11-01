//! Python `tests/sandbox/test_runtime.py::test_remote_realpath_empty_success_output_is_transport_error`.
#![cfg(feature = "sandbox")]

use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError};

/// A session that resolves paths remotely and answers every command with empty success output.
#[derive(Debug)]
struct EmptyRealpathSession {
    manifest: Manifest,
    commands: Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl SandboxSession for EmptyRealpathSession {
    fn id(&self) -> &str {
        "empty-realpath"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "empty-realpath"
    }
    fn resolves_paths_remotely(&self) -> bool {
        true
    }
    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(true)
    }
    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        self.commands.lock().unwrap().push(request.resolved_argv());
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            path.to_string(),
        ))
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn persist_workspace(&self) -> Result<Vec<u8>, SandboxError> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn an_empty_successful_resolver_answer_is_a_transport_error() {
    let session = EmptyRealpathSession {
        manifest: Manifest::new().with_root("/workspace"),
        commands: Mutex::new(Vec::new()),
    };

    let error = session
        .ls("file.txt")
        .await
        .expect_err("resolver answered nothing");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("empty_stdout")
    );
    let commands = session.commands.lock().unwrap().clone();
    let resolver = commands
        .iter()
        .find(|argv| argv.len() == 4 && argv[1] == "/workspace")
        .expect("the resolver ran");
    assert_eq!(&resolver[1..], ["/workspace", "/workspace/file.txt", "0"]);
}
