//! Ports of `tests/sandbox/test_runtime.py` (part 3): run preparation against a live session —
//! the working-directory probe, the `run_as` and workspace-scope binding handed to capabilities,
//! and per-preparation context processing.
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::{ExecResult, User};
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::Tool;
use serde_json::json;

// ----------------------------------------------------------------------------------------------
// Harness — mirrors `_CwdProbeSession` / `_FakeClient` / `_RecordingCapability`.
// ----------------------------------------------------------------------------------------------

/// Records every command the preparation runs, with the sandbox user it ran as, and can be told
/// to report the working directory as inaccessible.
#[derive(Debug)]
struct CwdProbeSession {
    manifest: Manifest,
    running: std::sync::Mutex<bool>,
    accessible: std::sync::Mutex<bool>,
    probes: std::sync::Mutex<Vec<(Vec<String>, Option<String>)>>,
    start_calls: AtomicUsize,
    stop_calls: AtomicUsize,
}

impl CwdProbeSession {
    fn new(accessible: bool) -> Self {
        CwdProbeSession {
            manifest: Manifest::new(),
            running: std::sync::Mutex::new(false),
            accessible: std::sync::Mutex::new(accessible),
            probes: std::sync::Mutex::new(Vec::new()),
            start_calls: AtomicUsize::new(0),
            stop_calls: AtomicUsize::new(0),
        }
    }

    fn probes(&self) -> Vec<(Vec<String>, Option<String>)> {
        self.probes.lock().unwrap().clone()
    }

    fn starts(&self) -> usize {
        self.start_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SandboxSession for CwdProbeSession {
    fn id(&self) -> &str {
        "cwd-probe"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "probe"
    }

    async fn start(&self) -> Result<(), SandboxError> {
        self.start_calls.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = true;
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = false;
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(*self.running.lock().unwrap())
    }

    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        self.probes.lock().unwrap().push((
            request.command.clone(),
            request.user.as_ref().map(|user| user.name.clone()),
        ));
        let accessible = *self.accessible.lock().unwrap();
        Ok(ExecResult {
            exit_code: if accessible { 0 } else { 1 },
            stdout: Vec::new(),
            stderr: if accessible {
                Vec::new()
            } else {
                b"not accessible".to_vec()
            },
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
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
struct ProbeClient {
    session: Arc<CwdProbeSession>,
    deletes: AtomicUsize,
}

impl ProbeClient {
    fn over(session: Arc<CwdProbeSession>) -> Arc<ProbeClient> {
        Arc::new(ProbeClient {
            session,
            deletes: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl SandboxClient for ProbeClient {
    fn backend_id(&self) -> &str {
        "probe"
    }

    async fn create(
        &self,
        _manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }

    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        self.session.stop().await
    }

    async fn resume(
        &self,
        _state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }
}

/// Records the binding it was handed and counts how often context processing ran.
#[derive(Debug, Default)]
struct BindingRecorder {
    bound_run_as: std::sync::Mutex<Vec<Option<String>>>,
    bound_cwd: std::sync::Mutex<Vec<Option<String>>>,
    process_calls: AtomicUsize,
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for BindingRecorder {
    fn capability_type(&self) -> &str {
        "binding-recorder"
    }

    fn tools(&self, binding: &CapabilityBinding) -> openai_agents_sdk::error::Result<Vec<Tool<C>>> {
        self.bound_run_as
            .lock()
            .unwrap()
            .push(binding.run_as.as_ref().map(|user| user.name.clone()));
        self.bound_cwd
            .lock()
            .unwrap()
            .push(binding.scope.cwd().map(str::to_string));
        Ok(Vec::new())
    }

    fn process_context(&self, mut context: Vec<InputItem>) -> Vec<InputItem> {
        let calls = self.process_calls.fetch_add(1, Ordering::SeqCst) + 1;
        context.push(InputItem::from_value(json!({
            "role": "user",
            "content": format!("process_calls={calls}")
        })));
        context
    }
}

fn scripted(text: &str) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(text)])])
            .expect("scripted model"),
    )
}

// ----------------------------------------------------------------------------------------------
// The working-directory probe
// ----------------------------------------------------------------------------------------------

/// Python: `test_prepare_agent_binds_and_validates_run_workspace_scope`.
///
/// A configured `cwd` is probed with `test -d` and then `test -x`, as the configured sandbox
/// user, and the resolved scope is what the capabilities are bound to.
#[tokio::test]
async fn preparation_probes_the_working_directory_and_binds_the_scope() {
    let session = Arc::new(CwdProbeSession::new(true));
    let client = ProbeClient::over(session.clone());
    let capability = Arc::new(BindingRecorder::default());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(scripted("done"))
        .with_capabilities(vec![capability.clone()])
        .with_run_as(User::new("sandbox-user"));

    let config = SandboxRunConfig::with_client(client).cwd("tasks/a");
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("preparation succeeds");
    prepared.guard.close().await.expect("teardown");

    assert_eq!(
        session.probes(),
        vec![
            (
                vec![
                    "test".to_string(),
                    "-d".to_string(),
                    "/workspace/tasks/a".to_string()
                ],
                Some("sandbox-user".to_string())
            ),
            (
                vec![
                    "test".to_string(),
                    "-x".to_string(),
                    "/workspace/tasks/a".to_string()
                ],
                Some("sandbox-user".to_string())
            ),
        ]
    );
    assert_eq!(
        capability.bound_cwd.lock().unwrap().clone(),
        vec![Some("tasks/a".to_string())]
    );
}

/// Python: `test_prepare_agent_binds_run_as_to_cloned_capabilities`.
///
/// Without a `cwd` there is no probe, and the capability still sees the agent's `run_as`.
#[tokio::test]
async fn preparation_binds_run_as_without_probing_a_working_directory() {
    let session = Arc::new(CwdProbeSession::new(true));
    let client = ProbeClient::over(session.clone());
    let capability = Arc::new(BindingRecorder::default());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(scripted("done"))
        .with_capabilities(vec![capability.clone()])
        .with_run_as(User::new("sandbox-user"));

    let config = SandboxRunConfig::with_client(client);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("preparation succeeds");
    prepared.guard.close().await.expect("teardown");

    assert!(session.probes().is_empty());
    assert_eq!(
        capability.bound_run_as.lock().unwrap().clone(),
        vec![Some("sandbox-user".to_string())]
    );
    assert_eq!(capability.bound_cwd.lock().unwrap().clone(), vec![None]);
}

/// Python: `test_prepare_agent_serializes_cwd_probe_with_posix_sandbox_semantics`.
///
/// The probe path is always POSIX and always absolute under the workspace root, whatever the
/// host platform spells paths like.
#[tokio::test]
async fn the_working_directory_probe_uses_posix_sandbox_paths() {
    let session = Arc::new(CwdProbeSession::new(true));
    let client = ProbeClient::over(session.clone());

    let agent = SandboxAgent::<()>::new("sandbox").with_model_instance(scripted("done"));

    let config = SandboxRunConfig::with_client(client).cwd("tasks/a");
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("preparation succeeds");
    prepared.guard.close().await.expect("teardown");

    let probes = session.probes();
    assert_eq!(probes.len(), 2);
    for (argv, user) in probes {
        assert_eq!(argv[0], "test");
        assert_eq!(argv[2], "/workspace/tasks/a");
        assert_eq!(user, None);
    }
}

/// Python: `test_prepare_agent_rejects_inaccessible_run_workspace_scope`.
///
/// The first failing probe stops preparation; the second probe never runs, and the session the
/// preparation created is released rather than leaked.
#[tokio::test]
async fn an_inaccessible_working_directory_stops_preparation_after_one_probe() {
    let session = Arc::new(CwdProbeSession::new(false));
    let client = ProbeClient::over(session.clone());

    let agent = SandboxAgent::<()>::new("sandbox").with_model_instance(scripted("done"));

    let config = SandboxRunConfig::with_client(client.clone()).cwd("tasks/missing");
    let Err(error) = SandboxRunner::prepare(&agent, &config).await else {
        panic!("the working directory is refused");
    };
    assert!(
        error.to_string().contains(
            "Sandbox working directory `tasks/missing` does not exist or is not accessible"
        ),
        "{error}"
    );

    let probes = session.probes();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].0[1], "-d");
    // The failed preparation released the session it had created.
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
}

/// Python: `test_prepare_agent_starts_new_live_session_even_when_backend_reports_running`.
///
/// A session created by the runner is always started, even when the backend already claims to be
/// running, because "running" is a backend-level fact and the workspace still needs applying.
#[tokio::test]
async fn a_runner_created_session_is_started_even_when_the_backend_reports_running() {
    let session = Arc::new(CwdProbeSession::new(true));
    session.start().await.expect("pre-start");
    session.start_calls.store(0, Ordering::SeqCst);
    let client = ProbeClient::over(session.clone());

    let agent = SandboxAgent::<()>::new("sandbox").with_model_instance(scripted("done"));

    let config = SandboxRunConfig::with_client(client);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("preparation succeeds");
    prepared.guard.close().await.expect("teardown");

    assert_eq!(session.starts(), 1);
}

/// Python: `test_prepare_agent_rechecks_session_liveness_before_reusing_cached_agent`.
///
/// Rust prepares a fresh agent graph per run rather than caching one, so the Python invariant —
/// a second preparation must not hand back an agent bound to a stopped session — holds by
/// construction: the second `prepare` starts the session again.
#[tokio::test]
async fn a_second_preparation_restarts_a_stopped_session() {
    let session = Arc::new(CwdProbeSession::new(true));
    let client = ProbeClient::over(session.clone());
    let agent = SandboxAgent::<()>::new("sandbox").with_model_instance(scripted("done"));

    let config = SandboxRunConfig::with_client(client);
    let first = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("first preparation");
    assert_eq!(session.starts(), 1);
    first.guard.close().await.expect("teardown");
    assert!(!session.running().await.expect("liveness"));

    let second = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("second preparation");
    assert_eq!(session.starts(), 2);
    second.guard.close().await.expect("teardown");
}

/// Python: `test_prepare_agent_processes_context_with_bound_cached_capabilities`.
///
/// Capability context processing runs once per prepared turn, and sees the input of that turn.
#[tokio::test]
async fn capability_context_processing_runs_once_per_turn() {
    let session = Arc::new(CwdProbeSession::new(true));
    let client = ProbeClient::over(session.clone());
    let capability = Arc::new(BindingRecorder::default());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(scripted("done"))
        .with_capabilities(vec![capability.clone()]);

    let config = SandboxRunConfig::with_client(client);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("preparation succeeds");

    let first = prepared.graph.process_context(
        "sandbox",
        vec![InputItem::from_value(
            json!({"role": "user", "content": "hello"}),
        )],
    );
    assert_eq!(
        first,
        vec![
            InputItem::from_value(json!({"role": "user", "content": "hello"})),
            InputItem::from_value(json!({"role": "user", "content": "process_calls=1"})),
        ]
    );

    let second = prepared.graph.process_context(
        "sandbox",
        vec![InputItem::from_value(
            json!({"role": "user", "content": "hello again"}),
        )],
    );
    assert_eq!(
        second,
        vec![
            InputItem::from_value(json!({"role": "user", "content": "hello again"})),
            InputItem::from_value(json!({"role": "user", "content": "process_calls=2"})),
        ]
    );

    prepared.guard.close().await.expect("teardown");
}
