//! Ports of the `_sandbox` assertions of `tests/sandbox/test_runtime.py`: a run that completes
//! writes the per-agent sandbox section into its run state, as a run that pauses does (Python's
//! cleanup serializes the resume payload whenever it succeeds). A failed cleanup writes none, a
//! caller-owned session writes none, and a streamed run finds the agent it stopped at by
//! identity, so two agents sharing a name keep their own keys.

#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::RunOptions;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Shell};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig};
use openai_agents_sdk::sandbox::{SandboxRunner, SandboxStreamedRun};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::{Value, json};

/// Counts the lifecycle calls made against it; `stop` can be made to fail, as Python's
/// `_FailingStopSession` does.
#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
    fail_stop: bool,
    starts: AtomicUsize,
    stops: AtomicUsize,
    shutdowns: AtomicUsize,
}

impl FakeSession {
    fn new(fail_stop: bool) -> Arc<Self> {
        Arc::new(FakeSession {
            manifest: Manifest::new(),
            fail_stop,
            starts: AtomicUsize::new(0),
            stops: AtomicUsize::new(0),
            shutdowns: AtomicUsize::new(0),
        })
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
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        if self.fail_stop {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                "synthetic stop failure",
            ));
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(true)
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
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Python's `_FakeClient`: one pre-built session, released by stopping and shutting it down.
#[derive(Debug)]
struct FakeClient {
    session: Arc<FakeSession>,
    deletes: AtomicUsize,
}

impl FakeClient {
    fn over(session: &Arc<FakeSession>) -> Arc<Self> {
        Arc::new(FakeClient {
            session: session.clone(),
            deletes: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl SandboxClient for FakeClient {
    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn create(
        &self,
        _manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        session.stop().await?;
        session.shutdown().await?;
        Ok(())
    }

    async fn resume(
        &self,
        _state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }
}

fn scripted_model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    model
}

fn sandbox_agent(model: Arc<FakeModel>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
}

/// The section a run state must hold after a run of the lone agent `sandbox` on the fake client.
fn assert_lone_agent_section(section: &Option<Value>) {
    let section = section
        .as_ref()
        .expect("the run state has a sandbox section");
    assert_eq!(section["backend_id"], "fake");
    assert_eq!(section["current_agent_name"], "sandbox");
    assert_eq!(section["current_agent_key"], "sandbox");
    let sessions = section["sessions_by_agent"]
        .as_object()
        .expect("sessions_by_agent is a map");
    assert_eq!(
        sessions["sandbox"],
        json!({"agent_name": "sandbox", "session_state": section["session_state"]})
    );
}

async fn drain<C: Send + Sync + 'static>(run: &mut SandboxStreamedRun<C>) {
    let stream = run.streaming.stream_events();
    futures::pin_mut!(stream);
    while stream.next().await.is_some() {}
}

/// Python: `test_runner_merges_sandbox_instructions_and_tools` (the `_sandbox` half). The
/// session is released exactly once, and the run state still names the session it ran in.
#[tokio::test]
async fn a_completed_run_writes_the_sandbox_section() {
    let session = FakeSession::new(false);
    let client = FakeClient::over(&session);
    let config = SandboxRunConfig::with_client(client.clone());
    let outcome = SandboxRunner::run_to_outcome(
        &sandbox_agent(scripted_model()),
        "hello",
        &config,
        RunOptions::default(),
    )
    .await
    .expect("the run completes");

    assert_eq!(outcome.result.final_output_text().as_deref(), Some("done"));
    assert_eq!(session.starts.load(Ordering::SeqCst), 1);
    assert_eq!(session.stops.load(Ordering::SeqCst), 1);
    assert_eq!(session.shutdowns.load(Ordering::SeqCst), 1);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
    assert_lone_agent_section(&outcome.result.to_state().sandbox);
    // The outcome still reports state only for a sandbox the run left suspended.
    assert!(outcome.sandbox_state.is_none());
    assert!(outcome.sandbox_states_by_agent.is_empty());
}

/// Python: `test_runner_streamed_cleans_runner_owned_session` (the `_sandbox` half).
#[tokio::test]
async fn a_completed_streamed_run_writes_the_sandbox_section() {
    let session = FakeSession::new(false);
    let client = FakeClient::over(&session);
    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&sandbox_agent(scripted_model()), "hello", &config)
        .await
        .expect("the stream starts");
    drain(&mut run).await;

    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    assert_eq!(session.stops.load(Ordering::SeqCst), 1);
    assert_eq!(session.shutdowns.load(Ordering::SeqCst), 1);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
    let state = run.to_state().expect("the stream finished");
    assert_lone_agent_section(&state.sandbox);
    assert_eq!(run.sandbox_resume_state(), state.sandbox);
    assert!(
        run.sandbox_state().is_none(),
        "a finished run is not suspended"
    );
}

/// Python: `test_runner_omits_sandbox_resume_state_when_cleanup_fails`: the run still succeeds,
/// and no resume state is written because the session was not released cleanly.
#[tokio::test]
async fn a_failed_cleanup_writes_no_sandbox_section() {
    let session = FakeSession::new(true);
    let client = FakeClient::over(&session);
    let config = SandboxRunConfig::with_client(client.clone());
    let result: RunResult<()> =
        SandboxRunner::run(&sandbox_agent(scripted_model()), "hello", &config)
            .await
            .expect("a cleanup failure does not fail a finished run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
    assert_eq!(result.to_state().sandbox, None);
}

/// The streamed twin of the failed cleanup: no section either.
#[tokio::test]
async fn a_failed_cleanup_of_a_streamed_run_writes_no_sandbox_section() {
    let session = FakeSession::new(true);
    let client = FakeClient::over(&session);
    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&sandbox_agent(scripted_model()), "hello", &config)
        .await
        .expect("the stream starts");
    drain(&mut run).await;

    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
    assert_eq!(run.sandbox_resume_state(), None);
    assert_eq!(run.to_state().expect("the stream finished").sandbox, None);
}

/// A session the caller supplies is not released by the run and has no state to resume from.
#[tokio::test]
async fn a_caller_owned_session_writes_no_sandbox_section() {
    let session = FakeSession::new(false);
    let live: Arc<dyn SandboxSession> = session.clone();
    live.start().await.unwrap();
    let config = SandboxRunConfig::with_session(live);
    let result = SandboxRunner::run(&sandbox_agent(scripted_model()), "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(session.stops.load(Ordering::SeqCst), 0);
    assert_eq!(result.to_state().sandbox, None);
}

fn call(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn handoff_call(tool: &str, call_id: &str) -> OutputItem {
    call(json!({"type": "function_call", "call_id": call_id, "name": tool, "arguments": "{}"}))
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    call(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

/// Two agents share the name `sandbox` and the second one stops the run on an approval. The
/// streamed completion names it by identity, so the section's current key is `sandbox#2` (the
/// first agent of that name would be `sandbox`), and a restore from it resumes the second
/// agent's session.
#[cfg(unix)]
#[tokio::test]
async fn a_streamed_run_stopped_at_the_second_of_two_same_named_agents_keys_it_correctly() {
    use openai_agents_sdk::sandbox::runtime::SandboxSessionMode;
    use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let snapshots = tempfile::tempdir().unwrap();
    let config = || {
        let client: Arc<dyn SandboxClient> = Arc::new(UnixLocalSandboxClient::new().with_snapshot(
            SnapshotSpec::Local {
                base_path: snapshots.path().to_path_buf(),
            },
        ));
        SandboxRunConfig::with_client(client).session_mode(SandboxSessionMode::PerAgent)
    };
    let build = |first_model: Arc<FakeModel>, second_model: Arc<FakeModel>| {
        let second = Arc::new(
            SandboxAgent::<()>::new("sandbox")
                .with_model_instance(second_model)
                .with_instructions("Second instructions.")
                .with_capabilities(vec![Arc::new(
                    Shell::new().with_pty(false).with_needs_approval(true),
                )])
                .with_sandbox_handoff_to("sandbox"),
        );
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(first_model)
            .with_instructions("First instructions.")
            .with_capabilities(vec![
                Arc::new(Filesystem::new().with_native_apply_patch()),
                Arc::new(Shell::new().with_pty(false)),
            ])
            .with_sandbox_handoff(second)
    };
    let (first_model, second_model) = (Arc::new(FakeModel::new()), Arc::new(FakeModel::new()));
    first_model.add_turn(vec![handoff_call("transfer_to_sandbox", "to_second")]);
    second_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let mut run = SandboxRunner::run_streamed_with_options(
        &build(first_model, second_model),
        "hello",
        &config(),
        RunOptions::default().max_turns(Some(12)),
    )
    .await
    .unwrap();
    run.finish().await.unwrap();

    let state = run.to_state().expect("the stream finished");
    let section = state.sandbox.clone().expect("the paused stream writes it");
    assert_eq!(section["current_agent_key"], "sandbox#2");
    assert_eq!(section["current_agent_name"], "sandbox");
    let sessions = section["sessions_by_agent"].as_object().unwrap();
    assert_eq!(
        sessions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["sandbox", "sandbox#2"]
    );
    assert_eq!(
        section["session_state"],
        sessions["sandbox#2"]["session_state"]
    );
    assert_ne!(
        sessions["sandbox"]["session_state"]["session_id"],
        sessions["sandbox#2"]["session_state"]["session_id"]
    );

    // The section restores: the second agent's approval resumes in its own session and the run
    // finishes on the first agent after it hands back.
    let mut restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let pending = restored.get_interruptions()[0].clone();
    restored.approve(&pending, false);
    let (first_model, second_model) = (Arc::new(FakeModel::new()), Arc::new(FakeModel::new()));
    second_model.add_turn(vec![handoff_call("transfer_to_sandbox", "back")]);
    first_model.add_turn(vec![assistant_message("done")]);
    let resumed = SandboxRunner::run_with_options(
        &build(first_model, second_model),
        restored,
        &config(),
        RunOptions::default().max_turns(Some(12)),
    )
    .await
    .expect("the restored run completes");
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let finished = resumed
        .to_state()
        .sandbox
        .expect("a finished run writes it");
    assert_eq!(finished["current_agent_key"], "sandbox");
    assert_eq!(
        finished["sessions_by_agent"]["sandbox#2"]["session_state"]["session_id"],
        sessions["sandbox#2"]["session_state"]["session_id"],
        "the second agent resumed its own session"
    );
}
