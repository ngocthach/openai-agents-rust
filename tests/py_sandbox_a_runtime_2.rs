//! Ports of `tests/sandbox/test_runtime.py` (part 2): the session-lifecycle rows — concurrency
//! and archive limits taken from the run config, the resume-time workspace clearing that must
//! not delete a mount, and the session teardown ordering.
#![cfg(feature = "sandbox")]

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
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed.lock().unwrap().push(state.clone());
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

// ----------------------------------------------------------------------------------------------
// Concurrency and archive limits from the run config
// ----------------------------------------------------------------------------------------------

/// Python: `test_session_manager_passes_concurrency_limits_from_run_config[create]`.
///
/// The run config's concurrency limits reach the manifest a fresh session is created with.
#[tokio::test]
async fn concurrency_limits_from_the_run_config_reach_a_created_session() {
    let (_session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));
    let limits = SandboxConcurrencyLimits {
        manifest_entries: Some(2),
        local_dir_files: Some(3),
    };

    let config = SandboxRunConfig::with_client(client.clone()).concurrency_limits(limits);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let created = client.created.lock().unwrap();
    let manifest = created[0].clone().expect("a manifest was passed");
    assert_eq!(manifest.concurrency_limits(), limits);
}

/// Python: `test_session_manager_passes_archive_limits_from_run_config[create]` and
/// `[live_session]` — both paths push the limits onto the session itself.
#[tokio::test]
async fn archive_limits_from_the_run_config_reach_the_session() {
    let limits = SandboxArchiveLimits {
        max_input_bytes: Some(10),
        max_extracted_bytes: Some(20),
        max_members: Some(30),
    };

    // Created session.
    let (session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));
    let config = SandboxRunConfig::with_client(client).archive_limits(limits);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(
        session.archive_limits.lock().unwrap().clone(),
        vec![Some(limits)]
    );

    // Injected live session.
    let injected = Arc::new(ProbeSession::new(Manifest::new()));
    let agent = worker_agent(scripted("done"));
    let config = SandboxRunConfig::with_session(injected.clone()).archive_limits(limits);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(
        injected.archive_limits.lock().unwrap().clone(),
        vec![Some(limits)]
    );
}

/// Python: `test_session_manager_default_archive_limits_preserves_no_resource_limits`.
///
/// Without archive limits the session is told `None` rather than a default budget, so a backend
/// that imposes no limit keeps imposing none.
#[tokio::test]
async fn the_default_archive_limits_are_no_limits_at_all() {
    let (session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(session.archive_limits.lock().unwrap().clone(), vec![None]);
}

/// Python: `test_session_manager_rejects_invalid_archive_limits`.
///
/// The limits are validated before anything is created, so a bad budget never reaches a backend.
#[tokio::test]
async fn invalid_archive_limits_are_rejected_before_the_session_is_created() {
    let (_session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));
    let limits = SandboxArchiveLimits {
        max_input_bytes: Some(0),
        ..SandboxArchiveLimits::default()
    };

    let config = SandboxRunConfig::with_client(client.clone()).archive_limits(limits);
    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("a zero input budget is refused");
    assert!(
        error
            .to_string()
            .contains("archive_limits.max_input_bytes must be at least 1"),
        "{error}"
    );
    assert_eq!(client.created_count(), 0);
}

/// Python: `test_session_manager_rejects_invalid_concurrency_limits` (both parametrizations).
#[tokio::test]
async fn invalid_concurrency_limits_are_rejected_before_the_session_is_created() {
    for (limits, message) in [
        (
            SandboxConcurrencyLimits {
                manifest_entries: Some(0),
                local_dir_files: Some(1),
            },
            "concurrency_limits.manifest_entries must be at least 1",
        ),
        (
            SandboxConcurrencyLimits {
                manifest_entries: Some(1),
                local_dir_files: Some(0),
            },
            "concurrency_limits.local_dir_files must be at least 1",
        ),
    ] {
        let (_session, client) = ProbeClient::over(Manifest::new());
        let agent = worker_agent(scripted("done"));
        let config = SandboxRunConfig::with_client(client.clone()).concurrency_limits(limits);
        let error = SandboxRunner::run(&agent, "hello", &config)
            .await
            .expect_err("the limits are refused");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(client.created_count(), 0);
    }
}

// ----------------------------------------------------------------------------------------------
// Clearing the workspace root on resume
// ----------------------------------------------------------------------------------------------

/// An ephemeral in-container S3 mount at `path`.
fn mount_manifest(path: &str) -> Manifest {
    Manifest::new().with_entry(
        path,
        Entry::mount(ConfiguredMountStrategy::mountpoint(S3Mount::new("bucket")).mount()),
    )
}

/// Python: `test_clear_workspace_root_on_resume_preserves_nested_mounts`.
///
/// A resume clears the workspace but descends into the ancestors of a mount instead of deleting
/// them, because deleting a mount point would delete the bucket behind it.
#[tokio::test]
async fn clearing_the_workspace_on_resume_preserves_a_nested_mount() {
    let session = ProbeSession::new(mount_manifest("a/b"))
        .listing_for(
            "/workspace",
            vec![
                ls_entry("/workspace/a", true),
                ls_entry("/workspace/root.txt", false),
            ],
        )
        .listing_for(
            "/workspace/a",
            vec![
                ls_entry("/workspace/a/b", true),
                ls_entry("/workspace/a/local.txt", false),
            ],
        );

    assert_eq!(
        workspace_resume_mount_skip_relpaths(session.manifest(), "/workspace"),
        vec!["a/b".to_string()]
    );

    clear_workspace_root_on_resume(&session)
        .await
        .expect("the workspace is cleared");

    assert_eq!(session.ls_calls(), vec!["/workspace", "/workspace/a"]);
    assert_eq!(
        session.rm_calls(),
        vec![
            ("/workspace/a/local.txt".to_string(), true),
            ("/workspace/root.txt".to_string(), true),
        ]
    );
}

/// Python: `test_clear_workspace_root_on_resume_deletes_file_ancestor_of_skipped_mount`.
///
/// When the path that *should* be the mount's parent directory is a plain file, it is deleted:
/// the mount is remade from the manifest, and a file there would block it.
#[tokio::test]
async fn clearing_the_workspace_deletes_a_file_standing_where_a_mount_parent_belongs() {
    let session = ProbeSession::new(mount_manifest("a/b")).listing_for(
        "/workspace",
        vec![
            ls_entry("/workspace/a", false),
            ls_entry("/workspace/root.txt", false),
        ],
    );

    clear_workspace_root_on_resume(&session)
        .await
        .expect("the workspace is cleared");

    assert_eq!(session.ls_calls(), vec!["/workspace"]);
    assert_eq!(
        session.rm_calls(),
        vec![
            ("/workspace/a".to_string(), true),
            ("/workspace/root.txt".to_string(), true),
        ]
    );
}

/// Python: `test_clear_workspace_root_on_resume_preserves_workspace_root_mount`.
///
/// A mount at the workspace root itself means there is nothing to clear: the whole workspace is
/// the mount.
#[tokio::test]
async fn a_mount_at_the_workspace_root_stops_the_resume_clear_entirely() {
    let session = ProbeSession::new(mount_manifest("."));

    clear_workspace_root_on_resume(&session)
        .await
        .expect("nothing is cleared");

    assert!(session.ls_calls().is_empty());
    assert!(session.rm_calls().is_empty());
}

// ----------------------------------------------------------------------------------------------
// Session teardown
// ----------------------------------------------------------------------------------------------

/// Python: `test_sandbox_session_aclose_runs_public_cleanup_lifecycle` and
/// `test_runner_clears_sandbox_session_from_non_streamed_results_after_cleanup`.
///
/// A runner-owned session is stopped and shut down exactly once, and the run keeps no resume
/// state for a completed run.
#[tokio::test]
async fn a_completed_run_stops_and_shuts_down_the_session_once() {
    let (session, client) = ProbeClient::over(Manifest::new());
    let agent = worker_agent(scripted("done"));

    let config = SandboxRunConfig::with_client(client);
    let outcome = SandboxRunner::run_to_outcome(
        &agent,
        "hello",
        &config,
        openai_agents_sdk::RunOptions::default(),
    )
    .await
    .expect("the run completes");

    assert_eq!(outcome.result.final_output_text().as_deref(), Some("done"));
    assert!(outcome.sandbox_state.is_none());
    assert_eq!(session.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.shutdown_calls.load(Ordering::SeqCst), 1);
}
