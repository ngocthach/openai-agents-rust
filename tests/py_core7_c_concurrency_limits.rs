//! Port of `tests/sandbox/test_runtime.py::test_session_manager_passes_concurrency_limits_from_run_config`
//! for all three sources: a created, a resumed and an injected live session each receive the run
//! config's limits through `SandboxSession::set_concurrency_limits`.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::manifest::{Manifest, SandboxConcurrencyLimits};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};

#[derive(Debug)]
struct Probe {
    manifest: Manifest,
    running: Mutex<bool>,
    limits: Mutex<Vec<SandboxConcurrencyLimits>>,
}

impl Probe {
    fn new() -> Arc<Self> {
        Arc::new(Probe {
            manifest: Manifest::new(),
            running: Mutex::new(false),
            limits: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl SandboxSession for Probe {
    fn id(&self) -> &str {
        "probe-session"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "probe"
    }
    fn set_concurrency_limits(&self, limits: SandboxConcurrencyLimits) {
        self.limits.lock().unwrap().push(limits);
    }
    async fn start(&self) -> Result<(), SandboxError> {
        *self.running.lock().unwrap() = true;
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        *self.running.lock().unwrap() = false;
        Ok(())
    }
    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(*self.running.lock().unwrap())
    }
    async fn exec(&self, _request: ExecRequest) -> Result<ExecResult, SandboxError> {
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            openai_agents_sdk::sandbox::ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

#[derive(Debug)]
struct Client(Arc<Probe>);

#[async_trait]
impl SandboxClient for Client {
    fn backend_id(&self) -> &str {
        "probe"
    }
    async fn create(&self, _m: Option<Manifest>) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.0.clone())
    }
    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn resume(
        &self,
        _state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.0.clone())
    }
}

fn agent() -> SandboxAgent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .expect("scripted model"),
    );
    SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.")
}

#[tokio::test]
async fn the_run_config_limits_reach_every_session_source() {
    let limits = SandboxConcurrencyLimits {
        manifest_entries: Some(2),
        local_dir_files: Some(3),
    };
    for source in ["create", "resume", "live_session"] {
        let probe = Probe::new();
        let config = match source {
            "live_session" => SandboxRunConfig::with_session(probe.clone()),
            "resume" => SandboxRunConfig::with_client(Arc::new(Client(probe.clone())))
                .session_state(SandboxSessionState::new(
                    "probe",
                    "resume",
                    Manifest::new().root(),
                    &Manifest::new(),
                )),
            _ => SandboxRunConfig::with_client(Arc::new(Client(probe.clone()))),
        }
        .concurrency_limits(limits);

        SandboxRunner::run(&agent(), "hello", &config)
            .await
            .expect("the run completes");

        assert_eq!(*probe.limits.lock().unwrap(), vec![limits], "{source}");
    }
}
