//! Python `tests/sandbox/test_runtime.py::test_sandbox_agent_as_tool_uses_runner_sandbox_prep`:
//! a sandbox agent exposed as a tool of a plain agent gets the sandbox preparation (capability
//! instructions, manifest entries applied to the session) for its nested run.
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, Runner};

/// A session that is already running and records every way a run could touch it.
#[derive(Debug)]
struct LiveSession {
    manifest: Manifest,
    running: AtomicBool,
    starts: AtomicUsize,
    stops: AtomicUsize,
    shutdowns: AtomicUsize,
    writes: AtomicUsize,
    execs: AtomicUsize,
    validations: AtomicUsize,
}

impl LiveSession {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            manifest: Manifest::new(),
            running: AtomicBool::new(true),
            starts: AtomicUsize::new(0),
            stops: AtomicUsize::new(0),
            shutdowns: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
            execs: AtomicUsize::new(0),
            validations: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl SandboxSession for LiveSession {
    fn id(&self) -> &str {
        "live-session"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "live"
    }
    async fn start(&self) -> Result<(), SandboxError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn shutdown(&self) -> Result<(), SandboxError> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(self.running.load(Ordering::SeqCst))
    }
    async fn validate_manifest_application(
        &self,
        _manifest: &Manifest,
        _session_running: Option<bool>,
    ) -> Result<(), SandboxError> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn exec(&self, _request: ExecRequest) -> Result<ExecResult, SandboxError> {
        self.execs.fetch_add(1, Ordering::SeqCst);
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Contributes one instruction fragment.
#[derive(Debug)]
struct InstructionCapability;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for InstructionCapability {
    fn capability_type(&self) -> &str {
        "recording"
    }

    async fn instructions(
        &self,
        _manifest: &Manifest,
    ) -> openai_agents_sdk::Result<Option<String>> {
        Ok(Some("Use the sandbox carefully.".to_string()))
    }
}

#[tokio::test]
async fn a_sandbox_agent_tool_runs_with_sandbox_preparation() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let child_model = Arc::new(FakeModel::new());
    child_model.add_turn(vec![assistant_message("child done")]);
    let parent_model = Arc::new(FakeModel::new());
    parent_model.add_turn(vec![function_call(
        "delegate_to_child",
        r#"{"input": "check sandbox"}"#,
    )]);
    parent_model.add_turn(vec![assistant_message("parent done")]);

    let session = LiveSession::new();
    let child = SandboxAgent::<()>::new("child")
        .with_model_instance(child_model.clone())
        .with_instructions("Child base instructions.")
        .with_capabilities(vec![Arc::new(InstructionCapability)]);
    let tool = child.as_tool(
        "delegate_to_child",
        "Delegate to the sandbox child.",
        SandboxRunConfig::with_session(session.clone()),
    );
    let parent = Agent::<()>::new("parent")
        .with_model_instance(parent_model.clone())
        .with_instructions("Parent instructions.")
        .with_tool(tool);

    let result = Runner::run(&parent, "hello").await.expect("parent run");

    assert_eq!(result.final_output_text().as_deref(), Some("parent done"));
    let call = child_model.last_call().expect("the child model ran");
    let instructions = call.system_instructions.unwrap_or_default();
    assert!(
        instructions.contains("Child base instructions."),
        "{instructions}"
    );
    assert!(
        instructions.contains("Use the sandbox carefully."),
        "{instructions}"
    );
}
