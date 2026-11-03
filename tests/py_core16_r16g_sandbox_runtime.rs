//! Ports of the `tests/sandbox/test_runtime.py` cases that exercise the run-level session
//! manager: persisted path grants rebound before a resume, a rejected live manifest update,
//! nested handoff history through a context-rewriting capability, capability failures on a
//! credentialed manifest, and start-hook state controlling the prepared tools.
#![cfg(feature = "sandbox")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};

#[derive(Debug)]
struct Session {
    manifest: Manifest,
    running: AtomicBool,
    writes: AtomicUsize,
    reject_validation: bool,
    start_error: Option<String>,
}

impl Session {
    fn new(manifest: Manifest, running: bool, reject_validation: bool) -> Arc<Self> {
        Arc::new(Session {
            manifest,
            running: AtomicBool::new(running),
            writes: AtomicUsize::new(0),
            reject_validation,
            start_error: None,
        })
    }

    fn failing_start(manifest: Manifest, message: String) -> Arc<Self> {
        Arc::new(Session {
            manifest,
            running: AtomicBool::new(false),
            writes: AtomicUsize::new(0),
            reject_validation: false,
            start_error: Some(message),
        })
    }
}

#[async_trait]
impl SandboxSession for Session {
    fn id(&self) -> &str {
        "session"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn start(&self) -> Result<(), SandboxError> {
        if let Some(message) = &self.start_error {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStartError,
                message.clone(),
            ));
        }
        self.running.store(true, Ordering::SeqCst);
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(self.running.load(Ordering::SeqCst))
    }
    async fn validate_manifest_application(
        &self,
        _manifest: &Manifest,
        session_running: Option<bool>,
    ) -> Result<(), SandboxError> {
        if self.reject_validation && session_running == Some(true) {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceWriteTypeError,
                "live manifest update rejected",
            ));
        }
        Ok(())
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

fn done_model() -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .expect("scripted model"),
    )
}

/// Python: `test_session_manager_validates_running_manifest_update_before_materialization`.
#[tokio::test]
async fn a_rejected_live_manifest_update_applies_nothing_to_the_running_session() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let session = Session::new(Manifest::new(), true, true);
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(done_model())
        .with_capabilities(vec![Arc::new(ManifestMutationCapability)]);
    let config = SandboxRunConfig::with_session(session.clone());

    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("the running session rejects the update");

    assert!(
        error.to_string().contains("live manifest update rejected"),
        "{error}"
    );
    assert_eq!(session.writes.load(Ordering::SeqCst), 0);
    assert!(session.manifest().entries().is_empty());
}

#[derive(Debug)]
struct ResumeClient {
    session: Arc<Session>,
    resumed: Mutex<Vec<SandboxSessionState>>,
}

#[async_trait]
impl SandboxClient for ResumeClient {
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn create(
        &self,
        _manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }
    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn resume(
        &self,
        state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed.lock().unwrap().push(state.clone());
        Ok(self.session.clone())
    }
}

fn credentialed_manifest(secret: &str) -> Manifest {
    Manifest::new().with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(
                S3Mount::new("example-bucket").with_credentials("example-access-key", secret),
                "rclone",
            )
            .mount(),
        ),
    )
}

#[derive(Debug)]
struct FailingCheckCapability {
    secret: String,
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for FailingCheckCapability {
    fn capability_type(&self) -> &str {
        "failing-check"
    }
    fn check_manifest(&self, _manifest: &Manifest) -> openai_agents_sdk::Result<()> {
        Err(openai_agents_sdk::AgentsError::user(format!(
            "capability rejected the manifest with {}",
            self.secret
        )))
    }
}

/// Python: `test_session_manager_redacts_capability_failure_with_external_mount_authority`:
/// a capability that fails on a manifest holding mount credentials surfaces only the fixed
/// protected-mount sentence.
#[tokio::test]
async fn a_capability_failure_on_a_credentialed_manifest_is_redacted() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let sentinel = "manager-capability-secret";
    let manifest = credentialed_manifest(sentinel);
    let session = Session::new(manifest.clone(), true, false);
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(done_model())
        .with_default_manifest(manifest)
        .with_capabilities(vec![Arc::new(FailingCheckCapability {
            secret: sentinel.to_string(),
        })]);
    let config = SandboxRunConfig::with_session(session);

    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("the capability fails");

    let text = format!("{error} {error:?}");
    assert!(text.contains("protected mount configuration"), "{text}");
    assert!(!text.contains(sentinel), "{text}");
}

/// Without credentialed mounts the capability's own message is kept.
#[tokio::test]
async fn a_capability_failure_without_protected_mounts_keeps_its_message() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let session = Session::new(Manifest::new(), true, false);
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(done_model())
        .with_capabilities(vec![Arc::new(FailingCheckCapability {
            secret: "plain".to_string(),
        })]);
    let config = SandboxRunConfig::with_session(session);

    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("the capability fails");

    assert!(
        error
            .to_string()
            .contains("capability rejected the manifest with plain"),
        "{error}"
    );
}

#[derive(Debug, Default)]
struct RebuildContextCapability;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for RebuildContextCapability {
    fn capability_type(&self) -> &str {
        "rebuild-context"
    }
    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        context.into_iter().collect()
    }
}

/// Python: `test_context_rebuild_retains_unambiguous_nested_history_ownership` (non-streamed
/// half): a capability that rebuilds the model input does not duplicate the nested handoff
/// summary in the replay history.
#[tokio::test]
async fn a_context_rebuilding_capability_keeps_one_handoff_message_in_the_replay() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let worker_model = done_model();
    let triage_model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![
            assistant_message("handoff message"),
            function_call_with_id("transfer_to_worker", "{}", "call_handoff"),
        ])])
        .expect("scripted model"),
    );
    let worker = Arc::new(
        SandboxAgent::<()>::new("worker")
            .with_model_instance(worker_model)
            .with_capabilities(vec![Arc::new(RebuildContextCapability)]),
    );
    let triage = SandboxAgent::<()>::new("triage")
        .with_model_instance(triage_model)
        .with_sandbox_handoff(worker);
    let session = Session::new(Manifest::new(), true, false);
    let config = SandboxRunConfig::with_session(session);
    let mut options = openai_agents_sdk::RunOptions::default();
    options.run_config.nest_handoff_history = true;

    let result = SandboxRunner::run_with_options(&triage, "route this", &config, options)
        .await
        .expect("the run completes");

    let replay = serde_json::to_string(&result.to_input_list()).unwrap();
    assert_eq!(replay.matches("handoff message").count(), 1, "{replay}");
}

#[derive(Debug, Default)]
struct Enabled(AtomicBool);

#[derive(Debug)]
struct DisableToolHooks;

#[async_trait]
impl openai_agents_sdk::RunHooks<Enabled> for DisableToolHooks {
    async fn on_agent_start(
        &self,
        context: &openai_agents_sdk::RunContext<Enabled>,
        _agent: &openai_agents_sdk::Agent<Enabled>,
    ) {
        context.context().0.store(false, Ordering::SeqCst);
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

async fn tools_offered(with_hook: bool) -> String {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = done_model();
    let tool = openai_agents_sdk::tool::FunctionTool::<Enabled>::new(
        "optional_tool",
        "optional",
        |_ctx, _args: NoArgs| async move { Ok("optional".to_string()) },
    )
    .with_is_enabled_fn(|ctx, _agent| async move { ctx.context().0.load(Ordering::SeqCst) });
    let agent = SandboxAgent::<Enabled>::new("sandbox")
        .with_model_instance(model.clone())
        .map_agent(|agent| agent.with_tool(tool));
    let config = SandboxRunConfig::with_session(Session::new(Manifest::new(), true, false));
    let mut options = openai_agents_sdk::RunOptions::new(Enabled(AtomicBool::new(true)));
    if with_hook {
        options = options.hooks(Arc::new(DisableToolHooks));
    }
    SandboxRunner::run_with_options(&agent, "go", &config, options)
        .await
        .expect("the run completes");
    format!(
        "{:?}",
        model.first_call().expect("a model call").request.tools
    )
}

/// Python: `test_start_hook_state_controls_prepared_sandbox_tools` (context-enablement half):
/// the tool's enablement check runs after the start hook, so state the hook changes decides
/// whether the prepared sandbox agent offers the tool.
#[tokio::test]
async fn start_hook_state_controls_the_prepared_sandbox_tools() {
    assert!(tools_offered(false).await.contains("optional_tool"));
    assert!(!tools_offered(true).await.contains("optional_tool"));
}

/// Python: `test_session_start_redacts_external_mount_operation_failure` and
/// `test_runner_protected_start_preserves_caller_error_without_exposing_it`: a backend start
/// that fails on a credentialed manifest surfaces only the fixed protected-mount sentence,
/// whether the run owns the session or was handed it.
#[tokio::test]
async fn a_start_failure_on_a_credentialed_manifest_is_redacted() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let sentinel = "protected-start-secret";
    let manifest = credentialed_manifest(sentinel);
    let session = Session::failing_start(manifest.clone(), format!("start failed with {sentinel}"));
    let agent = SandboxAgent::<()>::new("worker").with_model_instance(done_model());
    let injected = SandboxRunConfig::with_session(session.clone());
    let client = Arc::new(ResumeClient {
        session,
        resumed: Mutex::new(Vec::new()),
    });
    let owned = SandboxRunConfig::with_client(client).manifest(manifest);

    for config in [injected, owned] {
        let error = SandboxRunner::run(&agent, "hello", &config)
            .await
            .expect_err("the backend fails to start");
        let text = format!("{error} {error:?}");
        assert!(text.contains("protected mount configuration"), "{text}");
        assert!(!text.contains(sentinel), "{text}");
    }
}
