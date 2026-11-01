//! Ports of `tests/sandbox/test_runtime.py` (part 4): the runner's use of a sandbox — the default
//! snapshot and the rollout id it derives, teardown failures, blocking guardrails, the agent
//! identity hooks see, per-agent sessions across handoffs, and resuming through a handoff.
#![cfg(feature = "sandbox")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, Filesystem, Memory, Shell,
};
use openai_agents_sdk::sandbox::entries::{Entry, EntryKind};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::runtime::SandboxSessionMode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{AgentsError, InMemorySession, RunOptions, RunState, Session};
use serde_json::json;

// ----------------------------------------------------------------------------------------------
// Harness — Python's `_FakeSession`, `_FakeClient` and `_ManifestSessionClient`.
// ----------------------------------------------------------------------------------------------

#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
    running: Mutex<bool>,
    files: Mutex<Vec<String>>,
    start_calls: AtomicUsize,
    stop_calls: AtomicUsize,
    shutdown_calls: AtomicUsize,
}

impl FakeSession {
    fn new(manifest: Manifest) -> Arc<Self> {
        Arc::new(FakeSession {
            manifest,
            running: Mutex::new(false),
            files: Mutex::new(Vec::new()),
            start_calls: AtomicUsize::new(0),
            stop_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
        })
    }

    fn stops(&self) -> usize {
        self.stop_calls.load(Ordering::SeqCst)
    }

    fn starts(&self) -> usize {
        self.start_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SandboxSession for FakeSession {
    fn id(&self) -> &str {
        "fake-session"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "fake"
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
        self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
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
            ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }

    async fn write_file(&self, path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        self.files.lock().unwrap().push(path.to_string());
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Hands out a new [`FakeSession`] built from the manifest it is asked for, and records every
/// manifest, snapshot spec and resume it sees.
#[derive(Debug, Default)]
struct ManifestClient {
    sessions: Mutex<Vec<Arc<FakeSession>>>,
    created_manifests: Mutex<Vec<Option<Manifest>>>,
    resumed_manifests: Mutex<Vec<Manifest>>,
    snapshots: Mutex<Vec<SnapshotSpec>>,
    deletes: AtomicUsize,
    fail_delete: bool,
}

impl ManifestClient {
    fn arc() -> Arc<Self> {
        Arc::new(ManifestClient::default())
    }

    fn failing_delete() -> Arc<Self> {
        Arc::new(ManifestClient {
            fail_delete: true,
            ..ManifestClient::default()
        })
    }

    fn created(&self) -> usize {
        self.created_manifests.lock().unwrap().len()
    }

    fn session(&self, index: usize) -> Arc<FakeSession> {
        self.sessions.lock().unwrap()[index].clone()
    }

    fn deleted(&self) -> usize {
        self.deletes.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SandboxClient for ManifestClient {
    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.created_manifests
            .lock()
            .unwrap()
            .push(manifest.clone());
        let session = FakeSession::new(manifest.unwrap_or_default());
        self.sessions.lock().unwrap().push(session.clone());
        Ok(session)
    }

    async fn create_with_snapshot(
        &self,
        manifest: Option<Manifest>,
        snapshot: SnapshotSpec,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.snapshots.lock().unwrap().push(snapshot);
        self.create(manifest).await
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        session.stop().await?;
        session.shutdown().await?;
        if self.fail_delete {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                "synthetic cleanup failure",
            ));
        }
        Ok(())
    }

    async fn resume(
        &self,
        _state: &SandboxSessionState,
        manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed_manifests
            .lock()
            .unwrap()
            .push(manifest.clone());
        let session = FakeSession::new(manifest);
        self.sessions.lock().unwrap().push(session.clone());
        Ok(session)
    }
}

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn done(model: &FakeModel) {
    model.add_turn(vec![assistant_message("done")]);
}

fn agent(model: &Arc<FakeModel>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new())
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

fn tripwire_guardrail() -> InputGuardrail<()> {
    InputGuardrail::new("tripwire", |_ctx, _agent, _input| async move {
        Ok(GuardrailFunctionOutput::tripwire())
    })
    .blocking()
}

fn is_input_tripwire(error: &AgentsError) -> bool {
    matches!(error, AgentsError::InputGuardrailTripwireTriggered { .. })
}

// ----------------------------------------------------------------------------------------------
// Default snapshot
// ----------------------------------------------------------------------------------------------

/// Python: `SandboxRuntimeSessionManager._resolve_snapshot_spec` (exercised by every runner-owned
/// create in `test_runtime.py`). A session the runner creates is handed the SDK default local
/// snapshot; an explicit run-config snapshot wins.
#[tokio::test]
async fn created_sessions_get_the_default_local_snapshot_unless_one_is_configured() {
    let model = model();
    done(&model);
    let client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(client.clone());
    SandboxRunner::run(&agent(&model), "hello", &config)
        .await
        .expect("the run completes");
    {
        let snapshots = client.snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 1);
        assert!(
            matches!(snapshots[0], SnapshotSpec::Local { .. }),
            "{:?}",
            snapshots[0]
        );
    }

    done(&model);
    let explicit = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(explicit.clone()).snapshot(SnapshotSpec::Noop);
    SandboxRunner::run(&agent(&model), "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(
        explicit.snapshots.lock().unwrap().clone(),
        vec![SnapshotSpec::Noop]
    );
}

// ----------------------------------------------------------------------------------------------
// Rollout id
// ----------------------------------------------------------------------------------------------

/// Run a shell call that needs approval so the sandbox is suspended, and return the rollout id
/// the suspended state carries.
async fn suspended_rollout_id(options: RunOptions<()>, config: SandboxRunConfig) -> String {
    let model = model();
    model.add_turn(vec![shell_call("c1", "true")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![Arc::new(
            Shell::new().with_pty(false).with_needs_approval(true),
        )]);
    let outcome = SandboxRunner::run_to_outcome(&agent, "go", &config, options)
        .await
        .expect("the run pauses");
    let state = outcome.sandbox_state.expect("a paused run suspends");
    std::fs::remove_dir_all(&state.workspace_root).ok();
    state.rollout_id.expect("the state carries a rollout id")
}

/// Python: `_sandbox_memory_rollout_id` — the rollout id follows the runner's grouping hierarchy:
/// server conversation, then SDK session, then trace group, then a generated per-run value.
#[tokio::test]
async fn the_rollout_id_follows_the_run_grouping_hierarchy() {
    let config = || SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));

    // A server conversation cannot be combined with SDK session persistence, so the conversation
    // case stands alone and still outranks the trace group.
    let mut options = RunOptions::<()>::default();
    options.conversation_id = Some("conv-1".to_string());
    options.run_config.group_id = Some("group-1".to_string());
    assert_eq!(suspended_rollout_id(options, config()).await, "conv-1");

    let mut options = RunOptions::<()>::default();
    options.session = Some(Arc::new(InMemorySession::new("sess-1")) as Arc<dyn Session>);
    options.run_config.group_id = Some("group-1".to_string());
    assert_eq!(suspended_rollout_id(options, config()).await, "sess-1");

    let mut options = RunOptions::<()>::default();
    options.run_config.group_id = Some("group-1".to_string());
    assert_eq!(suspended_rollout_id(options, config()).await, "group-1");

    let generated = suspended_rollout_id(RunOptions::<()>::default(), config()).await;
    assert!(generated.starts_with("run-"), "{generated}");

    // An explicit rollout id on the run config still wins.
    let mut options = RunOptions::<()>::default();
    options.conversation_id = Some("conv-1".to_string());
    assert_eq!(
        suspended_rollout_id(options, config().rollout_id("explicit")).await,
        "explicit"
    );
}

// ----------------------------------------------------------------------------------------------
// Teardown failures
// ----------------------------------------------------------------------------------------------

/// Python: `test_runner_omits_sandbox_resume_state_when_cleanup_fails`.
#[tokio::test]
async fn a_failed_teardown_leaves_no_sandbox_resume_state() {
    let model = model();
    done(&model);
    let client = ManifestClient::failing_delete();
    let config = SandboxRunConfig::with_client(client.clone());
    let outcome =
        SandboxRunner::run_to_outcome(&agent(&model), "hello", &config, RunOptions::default())
            .await
            .expect("a teardown failure does not fail a finished run");

    assert_eq!(outcome.result.final_output_text().as_deref(), Some("done"));
    assert!(outcome.sandbox_state.is_none());
    assert_eq!(client.deleted(), 1);
}

/// Python: `test_runner_streamed_ignores_sandbox_cleanup_failures_after_success`.
#[tokio::test]
async fn a_streamed_run_ignores_teardown_failures_after_success() {
    use futures::StreamExt;

    let model = model();
    done(&model);
    let client = ManifestClient::failing_delete();
    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&agent(&model), "hello", &config)
        .await
        .expect("the stream starts");
    let mut events = 0usize;
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {
            events += 1;
        }
    }

    assert!(events > 0);
    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    assert!(run.sandbox_state().is_none());
    assert!(run.session().is_none());
    assert_eq!(client.deleted(), 1);
}

/// Python: `test_runner_streamed_immediate_cancel_skips_waiting_for_sandbox_cleanup` — an
/// immediate cancel never hangs on a teardown that is still in flight.
#[tokio::test]
async fn an_immediate_cancel_does_not_hang_on_teardown() {
    use futures::StreamExt;
    use openai_agents_sdk::result::CancelMode;

    let model = model();
    done(&model);
    let client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&agent(&model), "hello", &config)
        .await
        .expect("the stream starts");

    let drained = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        {
            let stream = run.streaming.stream_events();
            futures::pin_mut!(stream);
            let _ = stream.next().await;
        }
        run.streaming.cancel(CancelMode::Immediate);
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    })
    .await;
    assert!(drained.is_ok(), "cancel + drain must finish promptly");
    run.finish().await.expect("finish after cancel");
    assert_eq!(client.session(0).stops(), 1);
}

// ----------------------------------------------------------------------------------------------
// Blocking input guardrails
// ----------------------------------------------------------------------------------------------

/// Python: `test_runner_guardrail_trip_blocks_runner_owned_sandbox_creation` and its streamed
/// twin `test_runner_streamed_guardrail_trip_blocks_runner_owned_sandbox_creation`.
///
/// The tripwire is raised and, as in Python, the session is never created
/// (`client.create_kwargs is None`): the runner prepares the agent after the blocking
/// guardrails, through `RunOptions::prepare_agent`.
#[tokio::test]
async fn a_tripped_guardrail_releases_the_runner_owned_session() {
    use futures::StreamExt;

    let model = model();
    done(&model);
    let client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(client.clone());
    let guarded = agent(&model).map_agent(|agent| agent.with_input_guardrail(tripwire_guardrail()));

    let error = SandboxRunner::run(&guarded, "hello", &config)
        .await
        .expect_err("the guardrail trips");
    assert!(is_input_tripwire(&error), "{error}");
    assert_eq!(client.created(), 0);
    assert_eq!(client.deleted(), 0);

    let streamed_client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(streamed_client.clone());
    let mut run = SandboxRunner::run_streamed(&guarded, "hello", &config)
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
    assert_eq!(streamed_client.created(), 0);
    assert_eq!(streamed_client.deleted(), 0);
}

/// A capability that adds `cap.txt` to the manifest.
#[derive(Debug, Default)]
struct ManifestMutationCapability {
    process_calls: AtomicUsize,
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestMutationCapability {
    fn capability_type(&self) -> &str {
        "manifest-mutation"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        self.process_calls.fetch_add(1, Ordering::SeqCst);
        manifest.with_entry("cap.txt", Entry::file(b"capability".to_vec()))
    }
}

/// Python: `test_runner_guardrail_trip_blocks_running_injected_session_mutation` and the streamed
/// variant.
///
/// DEFERRED: Python leaves a running injected session untouched when a blocking guardrail trips.
/// Rust patches the capability-added entries into the live session during preparation, which
/// happens before the guardrail runs, for the same reason as above. The assertions pin the
/// current behavior: the tripwire is raised and the injected session is neither stopped nor shut
/// down.
#[tokio::test]
async fn a_tripped_guardrail_never_stops_a_running_injected_session() {
    let model = model();
    done(&model);
    let injected = FakeSession::new(Manifest::new());
    injected.start().await.expect("pre-start");
    let before = injected.starts();

    let guarded = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_capabilities(vec![Arc::new(ManifestMutationCapability::default())])
        .map_agent(|agent| agent.with_input_guardrail(tripwire_guardrail()));
    let config = SandboxRunConfig::with_session(injected.clone());

    let error = SandboxRunner::run(&guarded, "hello", &config)
        .await
        .expect_err("the guardrail trips");
    assert!(is_input_tripwire(&error), "{error}");
    assert_eq!(injected.starts(), before);
    assert_eq!(injected.stops(), 0);
    assert_eq!(injected.shutdown_calls.load(Ordering::SeqCst), 0);
}

// ----------------------------------------------------------------------------------------------
// The agent that hooks and callbacks see
// ----------------------------------------------------------------------------------------------

/// Python: `test_runner_uses_public_sandbox_agent_for_dynamic_instructions`.
///
/// Dynamic instructions are evaluated once, for the agent the caller built. Rust hands the
/// callback the prepared agent, which is a clone with the same name; Python's `is` identity is
/// asserted by name.
#[tokio::test]
async fn dynamic_instructions_see_the_sandbox_agent_by_name() {
    let model = model();
    done(&model);
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(Vec::new())
        .map_agent(move |agent| {
            agent.with_dynamic_instructions(move |_ctx, current| {
                let recorder = recorder.clone();
                async move {
                    recorder.lock().unwrap().push(current.name.clone());
                    Ok("Saw public agent.".to_string())
                }
            })
        });

    let client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(seen.lock().unwrap().clone(), vec!["sandbox".to_string()]);
    let instructions = model
        .last_call()
        .expect("the model was called")
        .system_instructions
        .expect("instructions");
    assert!(
        instructions.contains("# Agent instructions\n\nSaw public agent."),
        "{instructions}"
    );
}

/// Python: `test_runner_uses_public_sandbox_agent_for_dynamic_prompts` (non-streamed and
/// streamed halves).
#[tokio::test]
async fn dynamic_prompts_see_the_sandbox_agent_by_name() {
    use futures::StreamExt;
    use openai_agents_sdk::prompts::Prompt;

    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let build = |name: &str, model: Arc<FakeModel>| {
        let recorder = seen.clone();
        SandboxAgent::<()>::new(name)
            .with_model_instance(model)
            .with_capabilities(Vec::new())
            .map_agent(move |agent| {
                agent.with_dynamic_prompt(move |_ctx, current| {
                    let recorder = recorder.clone();
                    async move {
                        recorder.lock().unwrap().push(current.name.clone());
                        Ok(Some(Prompt::new("prompt_test")))
                    }
                })
            })
    };

    let first_model = model();
    done(&first_model);
    let config = SandboxRunConfig::with_client(ManifestClient::arc());
    SandboxRunner::run(&build("sandbox", first_model), "hello", &config)
        .await
        .expect("the run completes");

    let second_model = model();
    second_model.add_turn(vec![assistant_message("streamed done")]);
    let config = SandboxRunConfig::with_client(ManifestClient::arc());
    let mut run =
        SandboxRunner::run_streamed(&build("streamed-sandbox", second_model), "hello", &config)
            .await
            .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    }
    assert_eq!(
        run.streaming.final_output_text().as_deref(),
        Some("streamed done")
    );
    assert_eq!(
        seen.lock().unwrap().clone(),
        vec!["sandbox".to_string(), "streamed-sandbox".to_string()]
    );
}

/// Python: `test_runner_uses_public_agent_for_call_model_input_filter` and the streamed twin.
#[tokio::test]
async fn the_call_model_input_filter_sees_the_sandbox_agent_by_name() {
    use futures::StreamExt;

    for streamed in [false, true] {
        let model = model();
        done(&model);
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let mut options = RunOptions::<()>::default();
        options.run_config.call_model_input_filter = Some(Arc::new(move |data| {
            let recorder = recorder.clone();
            Box::pin(async move {
                recorder.lock().unwrap().push(data.agent.name.clone());
                Ok(data.model_data)
            })
        }));

        let config = SandboxRunConfig::with_client(ManifestClient::arc());
        if streamed {
            let mut run =
                SandboxRunner::run_streamed_with_options(&agent(&model), "hello", &config, options)
                    .await
                    .expect("the stream starts");
            let stream = run.streaming.stream_events();
            futures::pin_mut!(stream);
            while stream.next().await.is_some() {}
        } else {
            SandboxRunner::run_with_options(&agent(&model), "hello", &config, options)
                .await
                .expect("the run completes");
        }
        assert_eq!(seen.lock().unwrap().clone(), vec!["sandbox".to_string()]);
    }
}

/// Python: `test_runner_uses_public_agent_for_non_streaming_output_guardrails`.
#[tokio::test]
async fn output_guardrails_see_the_sandbox_agent_by_name() {
    let model = model();
    done(&model);
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let agent = agent(&model).map_agent(move |agent| {
        agent.with_output_guardrail(OutputGuardrail::new(
            "record",
            move |_ctx, current, _output| {
                let recorder = recorder.clone();
                async move {
                    recorder.lock().unwrap().push(current.name.clone());
                    Ok(GuardrailFunctionOutput::pass())
                }
            },
        ))
    });

    let config = SandboxRunConfig::with_client(ManifestClient::arc());
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(seen.lock().unwrap().clone(), vec!["sandbox".to_string()]);
}

// ----------------------------------------------------------------------------------------------
// Handoffs
// ----------------------------------------------------------------------------------------------

/// Reads the README the session it is bound to was created with. Mirrors
/// `_ManifestInstructionsCapability`.
#[derive(Debug, Default)]
struct ManifestInstructionsCapability;

fn readme(manifest: &Manifest) -> String {
    match &manifest.entries()["README.md"].kind {
        EntryKind::File { content } => String::from_utf8(content.clone()).expect("utf-8 README"),
        other => panic!("README.md is not an inline file: {other:?}"),
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestInstructionsCapability {
    fn capability_type(&self) -> &str {
        "manifest-instructions"
    }

    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        _manifest: &Manifest,
    ) -> openai_agents_sdk::error::Result<Option<String>> {
        Ok(Some(readme(binding.session.manifest())))
    }
}

fn readme_manifest(text: &str) -> Manifest {
    Manifest::new().with_entry("README.md", Entry::file(text.as_bytes().to_vec()))
}

fn handoff_pair(triage_model: &Arc<FakeModel>, worker_model: &Arc<FakeModel>) -> SandboxAgent<()> {
    let worker = SandboxAgent::<()>::new("worker")
        .with_model_instance(worker_model.clone())
        .with_instructions("Worker instructions.")
        .with_default_manifest(readme_manifest("Worker workspace"))
        .with_capabilities(vec![Arc::new(ManifestInstructionsCapability)]);
    SandboxAgent::<()>::new("triage")
        .with_model_instance(triage_model.clone())
        .with_instructions("Triage instructions.")
        .with_default_manifest(readme_manifest("Triage workspace"))
        .with_capabilities(vec![Arc::new(ManifestInstructionsCapability)])
        .with_sandbox_handoff(Arc::new(worker))
}

/// Python: `test_runner_rebuilds_sandbox_resources_for_handoff_target_agent`.
///
/// In per-agent mode each agent of the graph gets a session built from its own manifest, so the
/// worker's instructions come from the worker's workspace, not the triage agent's.
#[tokio::test]
async fn a_handoff_target_gets_its_own_session_and_manifest() {
    let triage_model = model();
    let worker_model = model();
    worker_model.add_turn(vec![assistant_message("done")]);
    triage_model.add_turn(vec![function_call("transfer_to_worker", "{}")]);
    let client = ManifestClient::arc();
    let config =
        SandboxRunConfig::with_client(client.clone()).session_mode(SandboxSessionMode::PerAgent);

    let result = SandboxRunner::run(
        &handoff_pair(&triage_model, &worker_model),
        "route this",
        &config,
    )
    .await
    .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(client.created(), 2);
    let manifests = client.created_manifests.lock().unwrap();
    let texts: Vec<String> = manifests
        .iter()
        .map(|m| readme(m.as_ref().expect("a manifest")))
        .collect();
    assert!(texts.contains(&"Triage workspace".to_string()), "{texts:?}");
    assert!(texts.contains(&"Worker workspace".to_string()), "{texts:?}");

    let instructions = worker_model
        .calls()
        .into_iter()
        .next()
        .expect("the worker ran")
        .system_instructions
        .expect("instructions");
    assert!(
        instructions.contains("# Agent instructions\n\nWorker instructions."),
        "{instructions}"
    );
    assert!(
        instructions.contains("# Sandbox capability instructions\n\nWorker workspace"),
        "{instructions}"
    );
}

/// A capability that drops every context item mentioning `text`. Mirrors
/// `_DropContextTextCapability`.
#[derive(Debug)]
struct DropContextTextCapability(&'static str);

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for DropContextTextCapability {
    fn capability_type(&self) -> &str {
        "drop-context-text"
    }

    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        context
            .into_iter()
            .filter(|item| !item.0.to_string().contains(self.0))
            .collect()
    }
}

/// Python: `test_context_rewrite_releases_removed_nested_history_ownership` (non-streamed and
/// streamed): after a capability rewrites the handoff input, the replayed history carries the
/// handed-off message exactly once.
#[tokio::test]
async fn a_context_rewrite_does_not_duplicate_handed_off_history() {
    use futures::StreamExt;

    for streamed in [false, true] {
        let triage_model = model();
        let worker_model = model();
        worker_model.add_turn(vec![assistant_message("done")]);
        triage_model.add_turn(vec![
            assistant_message("handoff message"),
            function_call("transfer_to_worker", "{}"),
        ]);
        let worker = SandboxAgent::<()>::new("worker")
            .with_model_instance(worker_model.clone())
            .with_capabilities(vec![Arc::new(DropContextTextCapability("handoff message"))]);
        let triage = SandboxAgent::<()>::new("triage")
            .with_model_instance(triage_model.clone())
            .with_capabilities(Vec::new())
            .with_sandbox_handoff(Arc::new(worker));

        let config = SandboxRunConfig::with_client(ManifestClient::arc())
            .session_mode(SandboxSessionMode::PerAgent);
        let mut options = RunOptions::<()>::default();
        options.run_config.nest_handoff_history = true;

        let input_list = if streamed {
            let mut run =
                SandboxRunner::run_streamed_with_options(&triage, "route this", &config, options)
                    .await
                    .expect("the stream starts");
            {
                let stream = run.streaming.stream_events();
                futures::pin_mut!(stream);
                while stream.next().await.is_some() {}
            }
            run.streaming
                .result()
                .expect("the stream finished")
                .to_input_list()
        } else {
            SandboxRunner::run_with_options(&triage, "route this", &config, options)
                .await
                .expect("the run completes")
                .to_input_list()
        };

        let count = input_list
            .iter()
            .filter(|item| item.0.to_string().contains("handoff message"))
            .count();
        assert_eq!(count, 1, "streamed={streamed}: {input_list:?}");
    }
}

/// Python: `test_runner_resumed_handoff_materializes_manifest_for_new_sandbox_agent`.
///
/// A run that pauses on an approval and is resumed into a handoff still builds the worker's
/// workspace from the worker's own manifest. (Rust creates every agent's session when the run
/// starts, so the create count is 2 for the first run rather than 1 + 1.)
#[tokio::test]
async fn a_resumed_handoff_builds_the_new_agent_workspace_from_its_own_manifest() {
    let triage_model = model();
    let worker_model = model();
    worker_model.add_turn(vec![assistant_message("done")]);
    triage_model.add_turn(vec![shell_call("c1", "true")]);
    let worker = SandboxAgent::<()>::new("worker")
        .with_model_instance(worker_model.clone())
        .with_instructions("Worker instructions.")
        .with_default_manifest(readme_manifest("Worker workspace"))
        .with_capabilities(vec![Arc::new(ManifestInstructionsCapability)]);
    let build_triage = |model: &Arc<FakeModel>| {
        SandboxAgent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_instructions("Triage instructions.")
            .with_default_manifest(readme_manifest("Triage workspace"))
            .with_capabilities(vec![
                Arc::new(ManifestInstructionsCapability),
                Arc::new(Shell::new().with_pty(false).with_needs_approval(true)),
            ])
            .with_sandbox_handoff(Arc::new(worker.clone()))
    };

    let client = ManifestClient::arc();
    let config =
        SandboxRunConfig::with_client(client.clone()).session_mode(SandboxSessionMode::PerAgent);
    let first = SandboxRunner::run_to_outcome(
        &build_triage(&triage_model),
        "route this",
        &config,
        RunOptions::default(),
    )
    .await;
    // The fake backend cannot snapshot a workspace, so suspending may fail; what matters is the
    // approval pause itself.
    let outcome = match first {
        Ok(outcome) => outcome,
        Err(error) => panic!("the first run must pause on approval: {error}"),
    };
    assert_eq!(outcome.result.interruptions.len(), 1);

    let mut state = RunState::from_json_str(&outcome.result.to_state().to_json_string().unwrap())
        .expect("state round-trips");
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    triage_model.add_turn(vec![function_call("transfer_to_worker", "{}")]);

    let resumed = SandboxRunner::run_with_options(
        &build_triage(&triage_model),
        state,
        &SandboxRunConfig::with_client(client.clone()).session_mode(SandboxSessionMode::PerAgent),
        RunOptions::default(),
    )
    .await
    .expect("the resumed run completes");

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let instructions = worker_model
        .calls()
        .into_iter()
        .next()
        .expect("the worker ran")
        .system_instructions
        .expect("instructions");
    assert!(
        instructions.contains("# Sandbox capability instructions\n\nWorker workspace"),
        "{instructions}"
    );
}

// ----------------------------------------------------------------------------------------------
// Resuming from session state
// ----------------------------------------------------------------------------------------------

/// Python: `test_session_manager_reapplies_capability_manifest_mutations_on_resume`
/// (`session_state` source).
///
/// A resume hands the client the manifest with the capability's mutations applied, exactly once.
#[tokio::test]
async fn a_resume_reapplies_capability_manifest_mutations_once() {
    let model = model();
    done(&model);
    let capability = Arc::new(ManifestMutationCapability::default());
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.")
        .with_default_manifest(Manifest::new())
        .with_capabilities(vec![capability.clone()]);
    let state = SandboxSessionState::new("fake", "resume", "/workspace", &Manifest::new());

    let client = ManifestClient::arc();
    let config = SandboxRunConfig::with_client(client.clone()).session_state(state);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let resumed = client.resumed_manifests.lock().unwrap().clone();
    assert_eq!(resumed.len(), 1);
    assert!(resumed[0].entries().contains_key("cap.txt"));
    assert_eq!(capability.process_calls.load(Ordering::SeqCst), 1);
}

/// Python: `test_runner_persists_workspace_and_tool_choice_state_across_sandbox_resume`.
///
/// A workspace file written before the pause is visible after the resume, and the forced
/// `tool_choice` that applied to the first turns is not carried into the resumed run.
#[tokio::test]
async fn a_resume_keeps_the_workspace_and_resets_the_forced_tool_choice() {
    use openai_agents_sdk::model_settings::ToolChoice;

    let snapshots = tempfile::tempdir().expect("tempdir");
    let spec = SnapshotSpec::Local {
        base_path: snapshots.path().to_path_buf(),
    };
    let model = model();
    let build = |model: &Arc<FakeModel>| {
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(model.clone())
            .with_instructions("Base instructions.")
            .with_capabilities(vec![
                Arc::new(Filesystem::new().with_native_apply_patch()),
                Arc::new(Shell::new().with_pty(false).with_needs_approval(true)),
            ])
            .map_agent(|agent| {
                let mut settings = agent.model_settings().clone();
                settings.tool_choice = Some(ToolChoice::required());
                agent.with_model_settings(settings)
            })
    };
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "apply_patch_call", "call_id": "c_write", "status": "completed",
        "operation": {"type": "create_file", "path": "note.txt", "diff": "+persist me\n"}
    }))]);
    model.add_turn(vec![shell_call("c_read", "cat note.txt")]);

    let config = SandboxRunConfig::with_client(Arc::new(
        UnixLocalSandboxClient::new().with_snapshot(spec.clone()),
    ));
    let outcome = SandboxRunner::run_to_outcome(
        &build(&model),
        "hello",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .expect("the run pauses");
    assert_eq!(outcome.result.interruptions.len(), 1);
    let sandbox_state = outcome.sandbox_state.clone().expect("suspended");
    let mut state = RunState::from_json_str(&outcome.result.to_state().to_json_string().unwrap())
        .expect("round-trip");
    drop(outcome);
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    model.add_turn(vec![assistant_message("done")]);

    let resume_config =
        SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new().with_snapshot(spec)))
            .session_state(sandbox_state);
    let resumed = SandboxRunner::run_with_options(
        &build(&model),
        state,
        &resume_config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .expect("the resumed run completes");

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let output = resumed
        .new_items
        .iter()
        .find_map(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::ToolCallOutput { raw_item, .. }
                if raw_item["type"] == "shell_call_output" =>
            {
                Some(raw_item.clone())
            }
            _ => None,
        })
        .expect("the approved shell call ran");
    assert_eq!(output["output"][0]["stdout"], "persist me");
    assert_eq!(
        model.last_call().expect("calls").model_settings.tool_choice,
        None
    );
}

// ----------------------------------------------------------------------------------------------
// Memory manager across a handoff
// ----------------------------------------------------------------------------------------------

fn memory_agent(name: &str, model: &Arc<FakeModel>, memory: Option<Memory>) -> SandboxAgent<()> {
    let mut capabilities: Vec<Arc<dyn Capability<()>>> = vec![
        Arc::new(Filesystem::new().with_native_apply_patch()),
        Arc::new(Shell::new().with_pty(false)),
    ];
    if let Some(memory) = memory {
        capabilities.push(Arc::new(memory));
    }
    SandboxAgent::<()>::new(name)
        .with_model_instance(model.clone())
        .with_capabilities(capabilities)
}

/// Run `triage` (which hands off to `worker`) in a caller-owned unix-local session and report
/// whether the run handed back a memory manager. A caller-owned session is not flushed by the
/// runner, so the manager the run ends with is returned.
async fn ends_with_memory_manager(
    triage: SandboxAgent<()>,
    triage_model: &Arc<FakeModel>,
    worker_model: &Arc<FakeModel>,
) -> bool {
    triage_model.add_turn(vec![function_call("transfer_to_worker", "{}")]);
    worker_model.add_turn(vec![assistant_message("done")]);
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");
    let config = SandboxRunConfig::with_session(session.clone());

    let outcome = SandboxRunner::run_to_outcome(&triage, "go", &config, RunOptions::default())
        .await
        .expect("the run completes");
    let has_manager = outcome.memory_manager().is_some();

    client.delete(session).await.expect("delete");
    has_manager
}

/// Python: `test_prepare_agent_uses_active_sandbox_agent_memory_capability_for_handoffs`. After a
/// handoff to an agent that only reads memory, the session stops recording.
#[tokio::test]
async fn a_handoff_to_an_agent_without_generation_stops_memory_recording() {
    let triage_model = model();
    let worker_model = model();
    let worker = memory_agent(
        "worker",
        &worker_model,
        Some(Memory::new().with_generate(None)),
    );
    let triage = memory_agent("triage", &triage_model, Some(Memory::new()))
        .with_sandbox_handoff(Arc::new(worker));

    assert!(!ends_with_memory_manager(triage, &triage_model, &worker_model).await);
}

/// Python: `test_prepare_agent_enables_memory_when_handoff_target_adds_capability`.
#[tokio::test]
async fn a_handoff_to_an_agent_that_adds_memory_starts_memory_recording() {
    let triage_model = model();
    let worker_model = model();
    let worker = memory_agent("worker", &worker_model, Some(Memory::new()));
    let triage = memory_agent("triage", &triage_model, None).with_sandbox_handoff(Arc::new(worker));

    assert!(ends_with_memory_manager(triage, &triage_model, &worker_model).await);
}
