//! Ports of `tests/sandbox/test_runtime.py` (lines 3999 to the end), part 5: the session
//! manager half. Python drives `SandboxRuntimeSessionManager.ensure_session` directly; Rust's
//! equivalent entry point is [`SandboxRunner::prepare`], which resumes, creates or reuses a
//! session, processes the capabilities' manifest edits and reconciles an injected session's
//! manifest. Account provisioning and the snapshot lifecycle are in part 6.
#![cfg(feature = "sandbox")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mounts::BindMountStrategy;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::{ExecResult, Group, User};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};

// ----------------------------------------------------------------------------------------------
// Harness: mirrors `_FakeSession`, `_LiveSessionDeltaRecorder` and `_FakeClient`.
// ----------------------------------------------------------------------------------------------

/// A session double that records its lifecycle and the files the runtime writes into it.
#[derive(Debug)]
struct LiveSession {
    manifest: Manifest,
    running: Mutex<bool>,
    start_calls: AtomicUsize,
    stop_calls: AtomicUsize,
    shutdown_calls: AtomicUsize,
    /// Every `(path, contents)` written, in order.
    writes: Mutex<Vec<(String, Vec<u8>)>>,
    /// The next N writes fail, like `_LiveSessionDeltaRecorder(fail_entry_batch_times=N)`.
    failing_writes: AtomicUsize,
}

impl LiveSession {
    fn new(manifest: Manifest, running: bool) -> Arc<Self> {
        Arc::new(LiveSession {
            manifest,
            running: Mutex::new(running),
            start_calls: AtomicUsize::new(0),
            stop_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
            writes: Mutex::new(Vec::new()),
            failing_writes: AtomicUsize::new(0),
        })
    }

    fn starts(&self) -> usize {
        self.start_calls.load(Ordering::SeqCst)
    }

    fn stops(&self) -> usize {
        self.stop_calls.load(Ordering::SeqCst)
    }

    fn shutdowns(&self) -> usize {
        self.shutdown_calls.load(Ordering::SeqCst)
    }

    fn writes(&self) -> Vec<(String, Vec<u8>)> {
        self.writes.lock().unwrap().clone()
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
        Ok(ExecResult::default())
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            openai_agents_sdk::sandbox::ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
        let failing = self.failing_writes.load(Ordering::SeqCst);
        if failing > 0 {
            self.failing_writes.store(failing - 1, Ordering::SeqCst);
            // Recorded before it fails, like Python's recorder, so the retry is observable.
            self.writes
                .lock()
                .unwrap()
                .push((path.to_string(), contents.to_vec()));
            return Err(SandboxError::new(
                openai_agents_sdk::sandbox::ErrorCode::WorkspaceArchiveWriteError,
                "delta apply failed",
            ));
        }
        self.writes
            .lock()
            .unwrap()
            .push((path.to_string(), contents.to_vec()));
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Records the manifest (and state) the runtime hands to `create` and `resume`.
#[derive(Debug)]
struct RecordingClient {
    session: Arc<LiveSession>,
    created: Mutex<Vec<Option<Manifest>>>,
    resumed: Mutex<Vec<(SandboxSessionState, Manifest)>>,
}

impl RecordingClient {
    fn over(session: Arc<LiveSession>) -> Arc<Self> {
        Arc::new(RecordingClient {
            session,
            created: Mutex::new(Vec::new()),
            resumed: Mutex::new(Vec::new()),
        })
    }

    fn created(&self) -> Vec<Option<Manifest>> {
        self.created.lock().unwrap().clone()
    }

    fn resumed(&self) -> Vec<(SandboxSessionState, Manifest)> {
        self.resumed.lock().unwrap().clone()
    }
}

#[async_trait]
impl SandboxClient for RecordingClient {
    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.created.lock().unwrap().push(manifest);
        Ok(self.session.clone())
    }

    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resumed.lock().unwrap().push((state.clone(), manifest));
        Ok(self.session.clone())
    }
}

/// Adds one file to the manifest, like `_ManifestMutationCapability`.
#[derive(Debug, Default)]
struct ManifestMutation {
    process_calls: AtomicUsize,
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestMutation {
    fn capability_type(&self) -> &str {
        "manifest-mutation"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        self.process_calls.fetch_add(1, Ordering::SeqCst);
        manifest.with_entry("cap.txt", Entry::text_file("capability"))
    }
}

/// Adds an account, like `_ManifestUsersCapability`.
#[derive(Debug)]
struct ManifestUsers;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestUsers {
    fn capability_type(&self) -> &str {
        "manifest-users"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        manifest.with_user("sandbox-user")
    }
}

/// Adds path grants, like `_ManifestPathGrantsCapability`.
#[derive(Debug)]
struct ManifestPathGrants {
    grants: Vec<SandboxPathGrant>,
    process_calls: AtomicUsize,
}

impl ManifestPathGrants {
    fn new(grants: Vec<SandboxPathGrant>) -> Self {
        ManifestPathGrants {
            grants,
            process_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ManifestPathGrants {
    fn capability_type(&self) -> &str {
        "manifest-path-grants"
    }

    fn process_manifest(&self, mut manifest: Manifest) -> Manifest {
        self.process_calls.fetch_add(1, Ordering::SeqCst);
        for grant in &self.grants {
            manifest = manifest.with_extra_path_grant(grant.clone());
        }
        manifest
    }
}

/// Does nothing, like `Capability(type="noop")`.
#[derive(Debug)]
struct Noop;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Noop {
    fn capability_type(&self) -> &str {
        "noop"
    }
}

fn scripted() -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .expect("scripted model"),
    )
}

fn agent_with(capabilities: Vec<Arc<dyn Capability<()>>>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("worker")
        .with_model_instance(scripted())
        .with_instructions("Worker.")
        .with_capabilities(capabilities)
}

fn host_grant(path: &str, host_path: &str, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path)
        .expect("grant path")
        .with_host_path(host_path)
        .expect("host path")
        .read_only(read_only)
}

fn cap_write() -> (String, Vec<u8>) {
    ("/workspace/cap.txt".to_string(), b"capability".to_vec())
}

fn test_state(manifest: &Manifest) -> SandboxSessionState {
    SandboxSessionState::new("fake", "resume", manifest.root(), manifest)
}

// ----------------------------------------------------------------------------------------------
// Resume: what the trusted manifest carries
// ----------------------------------------------------------------------------------------------

/// A persisted state records that mount authority was dropped; the run hands the trusted
/// manifest, credentials and all, to the client so it can bind them back. Python:
/// `test_session_manager_rebinds_redacted_external_mount_authority`.
///
/// The state-level facts (redaction flag, rebind flags, no credentials in the persisted JSON) are
/// asserted by `py_sandbox_a_snapshot_1::mount_authority_is_redacted_and_blocks_a_bare_resume` and
/// `state_json_never_contains_credentials_or_host_paths`; this test covers the runtime half.
#[tokio::test]
async fn resume_receives_the_trusted_manifest_with_its_mount_entries() {
    let trusted =
        Manifest::new().with_entry("data", Entry::mount(BindMountStrategy::mount("/host/data")));
    let state = test_state(&trusted);
    let session = LiveSession::new(Manifest::new(), false);
    let client = RecordingClient::over(session);
    let agent = agent_with(Vec::new()).with_default_manifest(trusted.clone());

    let config = SandboxRunConfig::with_client(client.clone()).session_state(state.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    let resumed = client.resumed();
    assert_eq!(resumed.len(), 1);
    assert_eq!(resumed[0].0, state);
    assert!(
        resumed[0].1.entries().contains_key("data"),
        "the trusted manifest, with its mount, is what the client rebinds against"
    );
    assert!(client.created().is_empty(), "a resume never creates");
    prepared.guard.close().await.expect("close");
}

/// A capability's `process_manifest` runs exactly once for a resume, and the grant it adds
/// reaches `resume`. Python: `test_session_manager_rebinds_capability_host_path_grant_once`.
#[tokio::test]
async fn a_capability_host_path_grant_reaches_resume_exactly_once() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let grant = host_grant("/mnt/shared-data", &host_dir.path().to_string_lossy(), true);
    let capability = Arc::new(ManifestPathGrants::new(vec![grant.clone()]));
    let state = test_state(&Manifest::new().with_extra_path_grant(grant.clone()));
    let session = LiveSession::new(Manifest::new(), false);
    let client = RecordingClient::over(session);
    let agent = agent_with(vec![capability.clone()]);

    let config = SandboxRunConfig::with_client(client.clone()).session_state(state);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    assert_eq!(capability.process_calls.load(Ordering::SeqCst), 1);
    let resumed = client.resumed();
    assert_eq!(resumed.len(), 1);
    assert_eq!(resumed[0].1.extra_path_grants(), [grant]);
    prepared.guard.close().await.expect("close");
}

/// A state that requires a host path the trusted manifest no longer supplies is refused by the
/// backend's resume, naming the grant. Python:
/// `test_session_manager_rejects_unmarked_serialized_host_path`.
///
/// Uses the real Unix-local client, whose `resume` does the rebinding.
#[tokio::test]
async fn resume_without_the_trusted_host_path_is_refused_by_name() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let state = SandboxSessionState::new(
        "unix_local",
        "resume",
        "/workspace",
        &Manifest::new().with_extra_path_grant(host_grant(
            "/mnt/shared-data",
            &host_dir.path().to_string_lossy(),
            false,
        )),
    );
    assert!(state.path_grants_require_rebind);

    let client = Arc::new(UnixLocalSandboxClient::new());
    let agent = agent_with(Vec::new());
    let config = SandboxRunConfig::with_client(client).session_state(state);

    let error = SandboxRunner::prepare(&agent, &config)
        .await
        .expect_err("resume must be refused");
    let message = error.to_string();
    assert!(
        message.contains("requires current trusted host_path")
            || message.contains("/mnt/shared-data"),
        "{message}"
    );
}

/// The agent's `run_as` account is added to the manifest handed to `resume`. Python:
/// `test_session_manager_adds_run_as_user_on_resume`.
#[tokio::test]
async fn resume_adds_the_run_as_user_to_the_manifest() {
    let session = LiveSession::new(Manifest::new(), false);
    let client = RecordingClient::over(session);
    let agent = agent_with(Vec::new()).with_run_as(User::new("sandbox-user"));
    let state = test_state(&Manifest::new());

    let config = SandboxRunConfig::with_client(client.clone()).session_state(state);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    let resumed = client.resumed();
    assert_eq!(resumed[0].1.users(), [User::new("sandbox-user")]);
    prepared.guard.close().await.expect("close");
}

/// A `run_as` user that already belongs to a manifest group is not added a second time. Python:
/// `test_session_manager_does_not_duplicate_run_as_user_from_group`.
#[test]
fn a_run_as_user_already_in_a_group_is_not_added_to_users() {
    let run_as = User::new("sandbox-user");
    let manifest =
        Manifest::new().with_group(Group::new("sandbox-group").with_user(run_as.clone()));
    let agent = agent_with(Vec::new())
        .with_run_as(run_as.clone())
        .with_default_manifest(manifest);

    let processed = agent.resolved_manifest(None);

    assert!(processed.users().is_empty(), "{:?}", processed.users());
    assert_eq!(processed.groups().len(), 1);
}

// ----------------------------------------------------------------------------------------------
// Capability manifest edits reach the session, whichever way it was obtained
// ----------------------------------------------------------------------------------------------

/// Python: `test_session_manager_applies_capability_manifest_mutations_with_session_parity`,
/// `source="live_session"`: a running injected session gets the new file written into it.
#[tokio::test]
async fn capability_edits_reach_a_running_injected_session() {
    let session = LiveSession::new(Manifest::new(), true);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);

    let config = SandboxRunConfig::with_session(session.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    assert_eq!(session.writes(), vec![cap_write()]);
    assert_eq!(session.starts(), 0);
    prepared.guard.close().await.expect("close");
}

/// `source="session_state"`: the edited manifest is what `resume` receives.
#[tokio::test]
async fn capability_edits_reach_the_manifest_handed_to_resume() {
    let session = LiveSession::new(Manifest::new(), false);
    let client = RecordingClient::over(session);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);
    let state = test_state(&Manifest::new());

    let config = SandboxRunConfig::with_client(client.clone()).session_state(state);
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    let resumed = client.resumed();
    assert!(resumed[0].1.entries().contains_key("cap.txt"));
    prepared.guard.close().await.expect("close");
}

/// `source="create"`: the edited manifest is what `create` receives.
#[tokio::test]
async fn capability_edits_reach_the_manifest_handed_to_create() {
    let session = LiveSession::new(Manifest::new(), false);
    let client = RecordingClient::over(session);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);

    let config = SandboxRunConfig::with_client(client.clone()).manifest(Manifest::new());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");

    let created = client.created();
    let manifest = created[0].as_ref().expect("a manifest");
    assert!(manifest.entries().contains_key("cap.txt"));
    prepared.guard.close().await.expect("close");
}

// ----------------------------------------------------------------------------------------------
// Injected sessions that are not running yet
// ----------------------------------------------------------------------------------------------

/// A stopped injected session is started, never stopped or shut down by the run, and receives the
/// capability's file. Python:
/// `test_session_manager_starts_stopped_injected_session_with_manifest_mutation`.
///
/// Adapted: Python folds the new entry into the manifest the session starts with; the Rust
/// session owns its manifest, so the runtime starts it and then writes the entry into it.
#[tokio::test]
async fn a_stopped_injected_session_is_started_and_given_the_capability_file() {
    let session = LiveSession::new(Manifest::new(), false);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);

    let config = SandboxRunConfig::with_session(session.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");
    prepared.guard.close().await.expect("close");

    assert_eq!(session.starts(), 1);
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 0);
    assert_eq!(session.writes(), vec![cap_write()]);
}

/// Python: `test_session_manager_rejects_unsafe_stopped_injected_session_manifest` (the
/// `current_manifest` source): a stopped injected session whose manifest holds a mount with
/// unacknowledged mount-scoped credentials is refused before it is probed or started.
#[tokio::test]
async fn an_unsafe_stopped_injected_session_manifest_is_refused_before_start() {
    let mount = openai_agents_sdk::sandbox::mount_entries::ConfiguredMountStrategy::rclone(
        openai_agents_sdk::sandbox::mount_entries::S3Mount::new("example-bucket")
            .with_credentials("example-access-key", "example-secret-key"),
    )
    .mount();
    let unsafe_manifest = Manifest::new().with_entry(
        "remote",
        openai_agents_sdk::sandbox::entries::Entry::mount(mount),
    );
    let session = LiveSession::new(unsafe_manifest, false);
    let agent = agent_with(Vec::new());

    let config = SandboxRunConfig::with_session(session.clone());
    let Err(error) = SandboxRunner::prepare(&agent, &config).await else {
        panic!("an unsafe manifest must be refused");
    };

    assert!(
        error.to_string().contains("credential authority"),
        "{error}"
    );
    assert_eq!(session.starts(), 0);
    assert!(session.writes().is_empty());
}

/// A capability that changes a host-backed grant on a not-yet-running injected session is
/// refused. Python:
/// `test_session_manager_rejects_stopped_injected_session_host_mount_changes[changed-host-source]`.
#[tokio::test]
async fn a_changed_host_source_on_an_injected_session_is_refused() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let current =
        Manifest::new().with_extra_path_grant(host_grant("/mnt/shared-data", "/native/old", true));
    let session = LiveSession::new(current, false);
    let capability = Arc::new(ManifestPathGrants::new(vec![host_grant(
        "/mnt/shared-data",
        &host_dir.path().to_string_lossy(),
        true,
    )]));
    let agent = agent_with(vec![capability]);

    let config = SandboxRunConfig::with_session(session.clone());
    let Err(error) = SandboxRunner::prepare(&agent, &config).await else {
        panic!("a changed host source must be refused");
    };

    assert!(error.to_string().contains("host-backed"), "{error}");
    let grants = session.manifest().extra_path_grants().to_vec();
    assert_eq!(grants[0].host_path.as_deref(), Some("/native/old"));
}

/// `[mixed-duplicate-target]`: a path-only grant added next to an existing host-backed grant for
/// the same target changes what that target means, so it is refused.
#[tokio::test]
async fn a_mixed_duplicate_grant_target_is_refused() {
    let current =
        Manifest::new().with_extra_path_grant(host_grant("/mnt/shared-data", "/native/old", true));
    let session = LiveSession::new(current, false);
    let capability = Arc::new(ManifestPathGrants::new(vec![
        SandboxPathGrant::new("/mnt/shared-data").expect("grant"),
    ]));
    let agent = agent_with(vec![capability]);

    let config = SandboxRunConfig::with_session(session);
    let Err(error) = SandboxRunner::prepare(&agent, &config).await else {
        panic!("a mixed duplicate target must be refused");
    };

    assert!(error.to_string().contains("host-backed"), "{error}");
}

// ----------------------------------------------------------------------------------------------
// Injected sessions that are already running
// ----------------------------------------------------------------------------------------------

/// A running injected session is neither restarted nor stopped; the capability's file is written
/// into it. Python:
/// `test_session_manager_materializes_running_injected_session_manifest_mutation`.
#[tokio::test]
async fn a_running_injected_session_materializes_the_capability_file() {
    let session = LiveSession::new(Manifest::new(), true);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);

    let config = SandboxRunConfig::with_session(session.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");
    prepared.guard.close().await.expect("close");

    assert_eq!(session.starts(), 0);
    assert_eq!(session.writes(), vec![cap_write()]);
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 0);
}

/// A failed delta apply is reported, and the next preparation applies it again. Python:
/// `test_session_manager_retries_running_injected_session_delta_apply_after_failure`.
#[tokio::test]
async fn a_failed_delta_apply_is_retried_on_the_next_preparation() {
    let session = LiveSession::new(Manifest::new(), true);
    session.failing_writes.store(1, Ordering::SeqCst);
    let agent = agent_with(vec![Arc::new(ManifestMutation::default())]);
    let config = SandboxRunConfig::with_session(session.clone());

    let error = SandboxRunner::prepare(&agent, &config)
        .await
        .expect_err("the first apply fails");
    assert!(error.to_string().contains("delta apply failed"), "{error}");
    assert_eq!(session.writes(), vec![cap_write()]);

    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("the retry succeeds");
    prepared.guard.close().await.expect("close");

    assert_eq!(session.writes(), vec![cap_write(), cap_write()]);
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 0);
}

/// A capability that changes nothing causes no writes at all. Python:
/// `test_session_manager_skips_rematerialization_for_unchanged_running_session`.
#[tokio::test]
async fn an_unchanged_running_session_is_not_rematerialized() {
    let session = LiveSession::new(Manifest::new(), true);
    let agent = agent_with(vec![Arc::new(Noop)]);

    let config = SandboxRunConfig::with_session(session.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");
    prepared.guard.close().await.expect("close");

    assert_eq!(session.starts(), 0);
    assert!(session.writes().is_empty());
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 0);
}

/// Accounts cannot be added to a running session. Python:
/// `test_session_manager_rejects_running_injected_session_account_mutation`.
#[tokio::test]
async fn adding_an_account_to_a_running_injected_session_is_refused() {
    let session = LiveSession::new(Manifest::new(), true);
    let agent = agent_with(vec![Arc::new(ManifestUsers)]);

    let config = SandboxRunConfig::with_session(session.clone());
    let error = SandboxRunner::prepare(&agent, &config)
        .await
        .expect_err("must be refused");

    assert!(
        error
            .to_string()
            .contains("manifest.users` or `manifest.groups"),
        "{error}"
    );
    assert!(session.writes().is_empty());
    assert!(session.manifest().users().is_empty());
}

/// A run over an injected session leaves nothing to resume from: no state, and the session is
/// neither stopped nor shut down. Python:
/// `test_session_manager_omits_existing_payload_for_injected_live_session`.
#[tokio::test]
async fn a_run_over_an_injected_session_returns_no_resume_state() {
    let session = LiveSession::new(Manifest::new(), true);
    let agent = agent_with(Vec::new());
    let config = SandboxRunConfig::with_session(session.clone());

    let outcome = SandboxRunner::run_to_outcome(
        &agent,
        "hello",
        &config,
        openai_agents_sdk::RunOptions::default(),
    )
    .await
    .expect("run");

    assert!(outcome.sandbox_state.is_none());
    assert!(outcome.sandbox_states_by_agent.is_empty());
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 0);
}
