//! The resume rows of `test_session_manager_passes_concurrency_limits_from_run_config` and
//! `test_session_manager_passes_archive_limits_from_run_config`
//! (`tests/sandbox/test_runtime.py`): limits from the run config reach a resumed session as well
//! as a created or injected one. Helpers are copied from `py_sandbox_a_runtime_2.rs`.
#![cfg(feature = "sandbox")]
#![allow(dead_code, unused_imports)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::{Manifest, SandboxConcurrencyLimits};
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, clear_workspace_root_on_resume,
    workspace_resume_mount_skip_relpaths,
};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::Permissions;
use openai_agents_sdk::sandbox::types::{ExecResult, FileEntry, SandboxArchiveLimits, file_mode};
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};

// ----------------------------------------------------------------------------------------------
// Harness
// ----------------------------------------------------------------------------------------------

/// A session that records the limits it was handed and the `ls`/`rm` traffic a resume makes.
#[derive(Debug)]
struct ProbeSession {
    manifest: Manifest,
    running: std::sync::Mutex<bool>,
    archive_limits: std::sync::Mutex<Vec<Option<SandboxArchiveLimits>>>,
    ls_calls: std::sync::Mutex<Vec<String>>,
    rm_calls: std::sync::Mutex<Vec<(String, bool)>>,
    listing: std::sync::Mutex<Vec<(String, Vec<FileEntry>)>>,
    start_calls: AtomicUsize,
    stop_calls: AtomicUsize,
    shutdown_calls: AtomicUsize,
}

impl ProbeSession {
    fn new(manifest: Manifest) -> Self {
        ProbeSession {
            manifest,
            running: std::sync::Mutex::new(false),
            archive_limits: std::sync::Mutex::new(Vec::new()),
            ls_calls: std::sync::Mutex::new(Vec::new()),
            rm_calls: std::sync::Mutex::new(Vec::new()),
            listing: std::sync::Mutex::new(Vec::new()),
            start_calls: AtomicUsize::new(0),
            stop_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
        }
    }

    /// Answer `ls(dir)` with `entries`.
    fn listing_for(self, dir: &str, entries: Vec<FileEntry>) -> Self {
        self.listing
            .lock()
            .unwrap()
            .push((dir.to_string(), entries));
        self
    }

    fn ls_calls(&self) -> Vec<String> {
        self.ls_calls.lock().unwrap().clone()
    }

    fn rm_calls(&self) -> Vec<(String, bool)> {
        self.rm_calls.lock().unwrap().clone()
    }
}

/// A directory or file listing entry, as Python's `_ls_entry` builds one.
fn ls_entry(path: &str, is_dir: bool) -> FileEntry {
    FileEntry {
        name: path.rsplit('/').next().unwrap_or(path).to_string(),
        path: path.to_string(),
        size: Some(0),
        permissions: Some(
            Permissions::new(file_mode::ALL, file_mode::READ, file_mode::READ).as_directory(is_dir),
        ),
        is_dir,
        is_symlink: false,
    }
}

#[async_trait]
impl SandboxSession for ProbeSession {
    fn id(&self) -> &str {
        "probe-session"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "probe"
    }

    fn set_archive_limits(&self, limits: Option<SandboxArchiveLimits>) {
        self.archive_limits.lock().unwrap().push(limits);
    }

    fn archive_limits(&self) -> Option<SandboxArchiveLimits> {
        self.archive_limits
            .lock()
            .unwrap()
            .last()
            .copied()
            .flatten()
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

    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn ls(&self, path: &str) -> Result<Vec<FileEntry>, SandboxError> {
        self.ls_calls.lock().unwrap().push(path.to_string());
        let listing = self.listing.lock().unwrap();
        match listing.iter().find(|(dir, _)| dir == path) {
            Some((_, entries)) => Ok(entries.clone()),
            None => panic!("unexpected ls path: {path}"),
        }
    }

    async fn rm(&self, path: &str, recursive: bool) -> Result<(), SandboxError> {
        self.rm_calls
            .lock()
            .unwrap()
            .push((path.to_string(), recursive));
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// A client over one probe session, recording the manifest every create/resume was given.
#[derive(Debug)]
struct ProbeClient {
    session: Arc<ProbeSession>,
    created: std::sync::Mutex<Vec<Option<Manifest>>>,
    resumed: std::sync::Mutex<Vec<SandboxSessionState>>,
    resumed_manifests: std::sync::Mutex<Vec<Manifest>>,
}

impl ProbeClient {
    fn over(manifest: Manifest) -> (Arc<ProbeSession>, Arc<ProbeClient>) {
        let session = Arc::new(ProbeSession::new(manifest));
        (
            session.clone(),
            Arc::new(ProbeClient {
                session,
                created: std::sync::Mutex::new(Vec::new()),
                resumed: std::sync::Mutex::new(Vec::new()),
                resumed_manifests: std::sync::Mutex::new(Vec::new()),
            }),
        )
    }

    fn created_count(&self) -> usize {
        self.created.lock().unwrap().len()
    }
}

#[async_trait]
impl SandboxClient for ProbeClient {
    fn backend_id(&self) -> &str {
        "probe"
    }

    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.created.lock().unwrap().push(manifest);
        Ok(self.session.clone())
    }

    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.session.stop().await?;
        self.session.shutdown().await?;
        Ok(())
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed.lock().unwrap().push(state.clone());
        self.resumed_manifests.lock().unwrap().push(manifest);
        Ok(self.session.clone())
    }
}

fn scripted(text: &str) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(text)])])
            .expect("scripted model"),
    )
}

fn worker_agent(model: Arc<ScriptedModel>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.")
}

#[tokio::test]
async fn concurrency_limits_from_the_run_config_reach_a_resumed_session() {
    let (_session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));
    let limits = SandboxConcurrencyLimits {
        manifest_entries: Some(2),
        local_dir_files: Some(3),
    };
    let state = SandboxSessionState::new("probe", "resume-1", "/workspace", &Manifest::new());

    let config = SandboxRunConfig::with_client(client.clone())
        .session_state(state)
        .concurrency_limits(limits);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(
        client.created_count(),
        0,
        "the run resumed instead of creating"
    );
    let manifests = client.resumed_manifests.lock().unwrap();
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0].concurrency_limits(), limits);
}

#[tokio::test]
async fn archive_limits_from_the_run_config_reach_a_resumed_session() {
    let limits = SandboxArchiveLimits {
        max_input_bytes: Some(10),
        max_extracted_bytes: Some(20),
        max_members: Some(30),
    };
    let (session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));
    let state = SandboxSessionState::new("probe", "resume-1", "/workspace", &Manifest::new());

    let config = SandboxRunConfig::with_client(client.clone())
        .session_state(state)
        .archive_limits(limits);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(client.created_count(), 0);
    assert_eq!(
        session.archive_limits.lock().unwrap().clone(),
        vec![Some(limits)]
    );
}

/// `test_sandbox_session_rejects_unsafe_manifest_before_workspace_persistence`: persisting or
/// hydrating a workspace whose manifest holds an unacknowledged mount-scoped credential is
/// refused before any archive work, and the error never carries the credential.
#[tokio::test]
async fn an_unsafe_manifest_is_refused_before_workspace_persistence() {
    use openai_agents_sdk::sandbox::entries::Entry;
    use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession};
    use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
    use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;

    let sentinel = "workspace-persistence-secret";
    let manifest = Manifest::new().with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket").with_credentials("access-key", sentinel),
            )
            .mount(),
        ),
    );
    for operation in ["persist", "hydrate"] {
        let inner = UnixLocalSandboxClient::new()
            .create(Some(manifest.clone()))
            .await
            .expect("create");
        let session = InstrumentedSession::new(inner, Arc::new(CallbackSink::new(|_| {})));

        let error = if operation == "persist" {
            session.persist_workspace().await.map(|_| ())
        } else {
            session.hydrate_workspace(b"archive").await
        }
        .expect_err("an unsafe manifest must be refused");

        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{operation}");
        assert!(
            !error.message.contains(sentinel),
            "{operation}: {}",
            error.message
        );
    }
}

/// A capability that contributes a mount carrying unacknowledged mount-scoped credentials.
#[derive(Debug)]
struct CredentialedMountCapability;

#[async_trait]
impl<C: Send + Sync + 'static> openai_agents_sdk::sandbox::capabilities::Capability<C>
    for CredentialedMountCapability
{
    fn capability_type(&self) -> &str {
        "credentialed-mount"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        manifest.with_entry(
            "remote",
            Entry::mount(
                ConfiguredMountStrategy::rclone(
                    S3Mount::new("example-bucket")
                        .with_credentials("example-access-key", "example-secret-key"),
                )
                .mount(),
            ),
        )
    }
}

/// `test_session_manager_rejects_unsafe_stopped_injected_session_manifest[capability]`: the
/// unsafe mount comes from a capability rather than from the session's own manifest. The session
/// is refused before it is started and nothing reaches it.
#[tokio::test]
async fn a_capability_that_adds_unsafe_authority_to_a_stopped_injected_session_is_refused() {
    let session = Arc::new(ProbeSession::new(Manifest::new()));
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(scripted("done"))
        .with_instructions("Worker.")
        .with_capabilities(vec![Arc::new(CredentialedMountCapability)]);

    let config = SandboxRunConfig::with_session(session.clone());
    let Err(error) = SandboxRunner::prepare(&agent, &config).await else {
        panic!("an unsafe capability mount must be refused");
    };

    assert!(
        error.to_string().contains("credential authority"),
        "{error}"
    );
    assert_eq!(session.start_calls.load(Ordering::SeqCst), 0);
    assert!(session.manifest.entries().is_empty());
}
