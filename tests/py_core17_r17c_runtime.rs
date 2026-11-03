//! Ports of two cases of `tests/sandbox/test_runtime.py`
//! (`test_runner_streamed_run_loop_task_waits_for_sandbox_cleanup_and_persisted_state` and
//! `test_session_manager_rebinds_persisted_path_grants_from_current_manifest`) and of
//! `test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary` from
//! `tests/sandbox/test_mount_security.py`, observed through the runner.
#![cfg(feature = "sandbox")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ConfiguredMountStrategy, Entry, ErrorCode, S3Mount, SandboxAgent, SandboxError,
    SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use tokio::sync::Notify;

#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
    running: Mutex<bool>,
    stop_gate: Option<Arc<Notify>>,
    stop_calls: AtomicUsize,
}

impl FakeSession {
    fn new(manifest: Manifest, stop_gate: Option<Arc<Notify>>) -> Arc<Self> {
        Arc::new(FakeSession {
            manifest,
            running: Mutex::new(false),
            stop_gate,
            stop_calls: AtomicUsize::new(0),
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
        *self.running.lock().unwrap() = true;
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = false;
        if let Some(gate) = &self.stop_gate {
            gate.notified().await;
        }
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

/// Hands out sessions built from the manifest it is asked for, recording what it sees.
#[derive(Debug, Default)]
struct FakeClient {
    stop_gate: Option<Arc<Notify>>,
    sessions: Mutex<Vec<Arc<FakeSession>>>,
    creates: AtomicUsize,
    resumed: Mutex<Vec<Manifest>>,
}

#[async_trait]
impl SandboxClient for FakeClient {
    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        let session = FakeSession::new(manifest.unwrap_or_default(), self.stop_gate.clone());
        self.sessions.lock().unwrap().push(session.clone());
        Ok(session)
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        session.stop().await?;
        session.shutdown().await
    }

    async fn resume(
        &self,
        _state: &SandboxSessionState,
        manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed.lock().unwrap().push(manifest.clone());
        let session = FakeSession::new(manifest, None);
        self.sessions.lock().unwrap().push(session.clone());
        Ok(session)
    }
}

fn agent(model: &Arc<FakeModel>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new())
}

fn model(replies: &[&str]) -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    for reply in replies {
        model.add_turn(vec![assistant_message(*reply)]);
    }
    model
}

/// Python: `test_runner_streamed_run_loop_task_waits_for_sandbox_cleanup_and_persisted_state`.
/// A streamed run does not finish while the runner-owned session is still being stopped, the
/// state of the released session is carried by the finished run, and the same agent runs again.
#[tokio::test]
async fn a_streamed_run_waits_for_the_sandbox_stop_and_carries_its_state() {
    let gate = Arc::new(Notify::new());
    let client = Arc::new(FakeClient {
        stop_gate: Some(gate.clone()),
        ..FakeClient::default()
    });
    let model = model(&["done", "again"]);
    let agent = agent(&model);
    let config = SandboxRunConfig::with_client(client.clone());

    let mut run = SandboxRunner::run_streamed(&agent, "hello", &config)
        .await
        .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        let mut drain = Box::pin(async { while stream.next().await.is_some() {} });
        // The stream stays open until the session's stop returns.
        let blocked = tokio::time::timeout(std::time::Duration::from_millis(200), &mut drain).await;
        assert!(blocked.is_err(), "the run finished before the stop did");
        assert!(
            client.sessions.lock().unwrap()[0]
                .stop_calls
                .load(Ordering::SeqCst)
                >= 1
        );

        gate.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(5), drain)
            .await
            .expect("the run finishes once the stop returns");
    }
    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    let section = run
        .to_state()
        .expect("a finished run has a state")
        .sandbox
        .expect("the released session's state is carried by the run");
    assert!(section.get("session_state").is_some(), "{section}");

    // The stop gate stays open for the second run.
    gate.notify_one();
    gate.notify_one();
    let second = SandboxRunner::run(&agent, "again", &config)
        .await
        .expect("the agent runs again");
    assert_eq!(second.final_output_text().as_deref(), Some("again"));
}

/// Python: `test_session_manager_rebinds_persisted_path_grants_from_current_manifest`. A state
/// that recorded a host-backed grant resumes against the trusted manifest, which supplies the
/// grant again: the manifest handed to the client carries exactly that grant.
#[tokio::test]
async fn a_resume_rebinds_persisted_path_grants_from_the_trusted_manifest() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let trusted = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/mnt/shared-data")
            .expect("grant path")
            .with_host_path(host_dir.path().to_string_lossy())
            .expect("host path")
            .read_only(true),
    );
    let state = SandboxSessionState::new("fake", "resume", trusted.root(), &trusted);
    assert!(state.path_grants_require_rebind);

    let client = Arc::new(FakeClient::default());
    let model = model(&["done"]);
    let agent = agent(&model).with_default_manifest(trusted.clone());
    let config = SandboxRunConfig::with_client(client.clone()).session_state(state);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("the trusted manifest rebinds the grant");

    let resumed = client.resumed.lock().unwrap().clone();
    assert_eq!(resumed.len(), 1);
    assert_eq!(resumed[0].extra_path_grants(), trusted.extra_path_grants());
    prepared.guard.close().await.expect("close");
}

/// Python: `test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary`. A
/// credentialed mount under an entry path that climbs out of the root is an invalid manifest
/// path, and the error never carries the credential.
#[tokio::test]
async fn an_invalid_entry_path_over_a_credentialed_mount_fails_without_the_secret() {
    let sentinel = "invalid-path-secret";
    let mount = ConfiguredMountStrategy::docker_volume(
        S3Mount::new("bucket").with_credentials("access-key", sentinel),
        "rclone",
    )
    .mount();
    let manifest = Manifest::new().with_entry("../data", Entry::mount(mount));

    let client = Arc::new(FakeClient::default());
    let model = model(&["done"]);
    let agent = agent(&model).with_default_manifest(manifest);
    let config = SandboxRunConfig::with_client(client.clone());
    let error = match SandboxRunner::prepare(&agent, &config).await {
        Ok(_) => panic!("an invalid manifest path is refused"),
        Err(error) => error,
    };

    let text = format!("{error} {error:?}");
    assert!(!text.contains(sentinel), "{text}");
    assert!(!text.contains("../data"), "{text}");
    assert!(text.contains("invalid_manifest_path"), "{text}");
    assert!(
        text.contains("protected mount configuration"),
        "the refusal is the redacted message: {text}"
    );
    assert_eq!(
        client.creates.load(Ordering::SeqCst),
        0,
        "nothing was created"
    );
}
