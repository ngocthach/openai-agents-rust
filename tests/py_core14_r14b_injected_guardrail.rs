//! Python `tests/sandbox/test_runtime.py::test_runner_guardrail_trip_blocks_running_injected_session_mutation`
//! and its streamed twin: a blocking input guardrail that trips leaves a running injected session
//! exactly as the caller handed it over (no capability entries applied, no manifest validation
//! against it, no start, stop or shutdown).
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::AgentsError;
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};

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

    fn untouched(&self) -> bool {
        self.starts.load(Ordering::SeqCst) == 0
            && self.stops.load(Ordering::SeqCst) == 0
            && self.shutdowns.load(Ordering::SeqCst) == 0
            && self.writes.load(Ordering::SeqCst) == 0
            && self.execs.load(Ordering::SeqCst) == 0
            && self.validations.load(Ordering::SeqCst) == 0
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

/// A capability that adds `cap.txt` to the manifest.
#[derive(Debug, Default)]
struct ManifestMutationCapability;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestMutationCapability {
    fn capability_type(&self) -> &str {
        "manifest-mutation"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        manifest.with_entry("cap.txt", Entry::file(b"capability".to_vec()))
    }
}

fn guarded_agent(model: &Arc<FakeModel>) -> SandboxAgent<()> {
    model.add_turn(vec![assistant_message("done")]);
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_capabilities(vec![Arc::new(ManifestMutationCapability)])
        .map_agent(|agent| {
            agent.with_input_guardrail(
                InputGuardrail::new("tripwire", |_ctx, _agent, _input| async move {
                    Ok(GuardrailFunctionOutput::tripwire())
                })
                .blocking(),
            )
        })
}

fn is_input_tripwire(error: &AgentsError) -> bool {
    matches!(error, AgentsError::InputGuardrailTripwireTriggered { .. })
}

#[tokio::test]
async fn a_tripped_guardrail_leaves_a_running_injected_session_untouched() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = guarded_agent(&model);
    let session = LiveSession::new();
    let config = SandboxRunConfig::with_session(session.clone());

    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("the guardrail trips");

    assert!(is_input_tripwire(&error), "{error}");
    assert!(!session.manifest().entries().contains_key("cap.txt"));
    assert!(session.untouched());
}

#[tokio::test]
async fn a_tripped_streamed_guardrail_leaves_a_running_injected_session_untouched() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = guarded_agent(&model);
    let session = LiveSession::new();
    let config = SandboxRunConfig::with_session(session.clone());

    let mut run = SandboxRunner::run_streamed(&agent, "hello", &config)
        .await
        .expect("the stream starts");
    let mut failure = None;
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let Err(error) = event {
                failure = Some(error);
            }
        }
    }
    let failure = match failure {
        Some(error) => error,
        None => run.finish().await.expect_err("the guardrail trips"),
    };

    assert!(is_input_tripwire(&failure), "{failure}");
    assert!(!session.manifest().entries().contains_key("cap.txt"));
    assert!(session.untouched());
}
