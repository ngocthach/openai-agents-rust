//! Ports of the session cleanup cases of `tests/sandbox/test_runtime.py`: `aclose`, pre-stop
//! hooks, dependency teardown and the runner-owned cleanup, including failure redaction.

#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::FutureExt;
use openai_agents_sdk::sandbox::dependencies::{Dependencies, DependencyCloser};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, PreStopHook, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
    SandboxSessionLifecycle,
};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use tokio::sync::Notify;

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static LOGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        LOGS.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}

fn capture_logs() {
    let _ = log::set_logger(&Capture);
    log::set_max_level(log::LevelFilter::Warn);
}

const PROTECTED: &str = "sandbox operation failed while using a protected mount configuration";

/// Python's `_FakeSession`.
#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
    dependencies: Arc<Dependencies>,
    stop_calls: AtomicUsize,
    shutdown_calls: AtomicUsize,
    closed_dependencies: Arc<AtomicUsize>,
    fail_stop: bool,
}

impl FakeSession {
    fn new(manifest: Manifest) -> Arc<Self> {
        Self::build(manifest, false)
    }

    fn build(manifest: Manifest, fail_stop: bool) -> Arc<Self> {
        let closed = Arc::new(AtomicUsize::new(0));
        let dependencies = Dependencies::new();
        let counter = closed.clone();
        let closer: DependencyCloser = Arc::new(move |_dependency| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            .boxed()
        });
        dependencies
            .bind_factory(
                "resource",
                Arc::new(|| async { Ok(Arc::new(()) as _) }.boxed()),
                true,
                Some(closer),
            )
            .unwrap();
        Arc::new(FakeSession {
            manifest,
            dependencies: Arc::new(dependencies),
            stop_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
            closed_dependencies: closed,
            fail_stop,
        })
    }

    /// Build the owned dependency, so closing the container has something to release.
    async fn open_dependency(&self) {
        self.dependencies.get("resource").await.unwrap();
    }

    fn stops(&self) -> usize {
        self.stop_calls.load(Ordering::SeqCst)
    }

    fn shutdowns(&self) -> usize {
        self.shutdown_calls.load(Ordering::SeqCst)
    }

    fn dependency_closes(&self) -> usize {
        self.closed_dependencies.load(Ordering::SeqCst)
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

    fn dependencies(&self) -> Option<&Arc<Dependencies>> {
        Some(&self.dependencies)
    }

    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_stop {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                "stop failed",
            ));
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
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

fn handle(session: &Arc<FakeSession>) -> Arc<dyn SandboxSession> {
    session.clone()
}

/// Python's `_external_mount_manifest`: a mount whose credentials a failure must not echo.
fn external_mount_manifest(secret: &str) -> Manifest {
    Manifest::new().with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket").with_credentials("access-key", secret),
            )
            .mount(),
        ),
    )
}

fn failing_hook(message: String) -> PreStopHook {
    Arc::new(move || {
        let message = message.clone();
        async move { Err(SandboxError::new(ErrorCode::WorkspaceStopError, message)) }.boxed()
    })
}

/// Python: `test_sandbox_session_aclose_runs_public_cleanup_lifecycle`.
#[tokio::test]
async fn aclose_stops_shuts_down_and_closes_dependencies() {
    let session = FakeSession::new(Manifest::new());
    session.open_dependency().await;

    handle(&session).aclose().await.unwrap();

    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(session.dependency_closes(), 1);
}

/// Python: `test_sandbox_session_aclose_closes_dependencies_when_stop_fails`.
#[tokio::test]
async fn aclose_closes_dependencies_when_stop_fails() {
    let session = FakeSession::build(Manifest::new(), true);
    session.open_dependency().await;

    let error = handle(&session).aclose().await.unwrap_err();

    assert!(error.message.contains("stop failed"), "{error}");
    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 0);
    assert_eq!(session.dependency_closes(), 1);
}

/// Python: `test_sandbox_session_aclose_redacts_pre_stop_hook_failure`.
#[tokio::test]
async fn aclose_redacts_a_pre_stop_hook_failure_for_a_protected_mount() {
    let sentinel = "pre-stop-hook-secret";
    let session = FakeSession::new(external_mount_manifest(sentinel));
    session.open_dependency().await;
    let shared = handle(&session);
    shared.register_pre_stop_hook(failing_hook(format!(
        "pre-stop hook failed with {sentinel}"
    )));

    let error = shared.aclose().await.unwrap_err();

    assert_eq!(error.message, PROTECTED);
    assert!(!format!("{error:?}").contains(sentinel));
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(session.dependency_closes(), 1);
}

/// An unprotected manifest keeps the hook's own message.
#[tokio::test]
async fn a_pre_stop_hook_failure_keeps_its_message_without_protected_mounts() {
    let session = FakeSession::new(Manifest::new());
    let shared = handle(&session);
    shared.register_pre_stop_hook(failing_hook("hook broke".into()));

    let error = shared.aclose().await.unwrap_err();

    assert_eq!(error.message, "hook broke");
}

/// Pre-stop hooks run once, in registration order, and every hook runs even after a failure.
#[tokio::test]
async fn pre_stop_hooks_run_once_in_order_and_survive_a_failing_hook() {
    let session = FakeSession::new(Manifest::new());
    let shared = handle(&session);
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    for (name, fail) in [("first", true), ("second", false)] {
        let order = order.clone();
        shared.register_pre_stop_hook(Arc::new(move || {
            let order = order.clone();
            async move {
                order.lock().unwrap().push(name);
                if fail {
                    Err(SandboxError::new(
                        ErrorCode::WorkspaceStopError,
                        "first failed",
                    ))
                } else {
                    Ok(())
                }
            }
            .boxed()
        }));
    }

    let error = shared.run_pre_stop_hooks().await.unwrap_err();
    assert_eq!(error.message, "first failed");
    // Already ran: a second call does nothing.
    shared.run_pre_stop_hooks().await.unwrap();

    assert_eq!(*order.lock().unwrap(), vec!["first", "second"]);
}

/// Python: `test_repeated_direct_aclose_never_persists_after_pre_stop_failure` (the plain
/// failure variant; the cancelled variant is an asyncio concern).
#[tokio::test]
async fn a_repeated_aclose_never_stops_after_a_pre_stop_failure() {
    let session = FakeSession::new(Manifest::new());
    session.open_dependency().await;
    let shared = handle(&session);
    shared.register_pre_stop_hook(failing_hook("pre-stop hook failed".into()));

    assert!(shared.aclose().await.is_err());
    shared.aclose().await.unwrap();

    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 2);
    assert_eq!(session.dependency_closes(), 1);
}

/// Python: `test_concurrent_direct_aclose_waits_for_pre_stop_failure[cancel_waiter=False]`.
#[tokio::test]
async fn a_concurrent_aclose_waits_for_the_pre_stop_failure() {
    let session = FakeSession::new(Manifest::new());
    let shared = handle(&session);
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    {
        let (started, release) = (started.clone(), release.clone());
        shared.register_pre_stop_hook(Arc::new(move || {
            let (started, release) = (started.clone(), release.clone());
            async move {
                started.notify_one();
                release.notified().await;
                Err(SandboxError::new(
                    ErrorCode::WorkspaceStopError,
                    "pre-stop hook failed",
                ))
            }
            .boxed()
        }));
    }
    let first = tokio::spawn({
        let shared = shared.clone();
        async move { shared.aclose().await }
    });
    started.notified().await;
    let second = tokio::spawn({
        let shared = shared.clone();
        async move { shared.aclose().await }
    });
    tokio::task::yield_now().await;
    release.notify_one();

    let first = first.await.unwrap();
    let second = second.await.unwrap();

    assert!(first.is_err());
    assert!(second.is_ok());
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 2);
}

/// Python: `test_concurrent_direct_aclose_waits_for_pre_stop_failure[cancel_waiter=True]`:
/// a waiter that is cancelled (dropped) while queued does not run its own cleanup.
#[tokio::test]
async fn a_cancelled_waiting_aclose_runs_no_cleanup_of_its_own() {
    let session = FakeSession::new(Manifest::new());
    session.open_dependency().await;
    let shared = handle(&session);
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    {
        let (started, release) = (started.clone(), release.clone());
        shared.register_pre_stop_hook(Arc::new(move || {
            let (started, release) = (started.clone(), release.clone());
            async move {
                started.notify_one();
                release.notified().await;
                Err(SandboxError::new(
                    ErrorCode::WorkspaceStopError,
                    "pre-stop hook failed",
                ))
            }
            .boxed()
        }));
    }
    let first = tokio::spawn({
        let shared = shared.clone();
        async move { shared.aclose().await }
    });
    started.notified().await;
    {
        let mut waiter = shared.aclose();
        assert!(futures::poll!(&mut waiter).is_pending());
        // Dropped here, while queued behind the first cleanup.
    }
    release.notify_one();

    assert!(first.await.unwrap().is_err());
    assert_eq!(session.stops(), 0);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(session.dependency_closes(), 1);
}

// ---------------------------------------------------------------------------------------------
// runner-owned cleanup
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct FakeClient {
    session: Arc<FakeSession>,
    delete_calls: AtomicUsize,
    delete_error: Option<String>,
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
        self.session.open_dependency().await;
        Ok(self.session.clone())
    }

    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.delete_calls.fetch_add(1, Ordering::SeqCst);
        self.session.stop().await?;
        self.session.shutdown().await?;
        match &self.delete_error {
            Some(message) => Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                message.clone(),
            )),
            None => Ok(()),
        }
    }

    async fn resume(
        &self,
        _state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(self.session.clone())
    }
}

async fn run_owned(client: Arc<FakeClient>) -> Result<(), openai_agents_sdk::AgentsError> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .expect("scripted model"),
    );
    let agent = SandboxAgent::<()>::new("worker").with_model_instance(model);
    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run_to_outcome(
        &agent,
        "hi",
        &config,
        openai_agents_sdk::RunOptions::default(),
    )
    .await
    .map(|_| ())
}

/// Python: `test_runner_owned_cleanup_redacts_pre_stop_hook_failure`. Rust's client `delete`
/// releases the session (stop and shutdown), so what the failed hook prevents is the
/// workspace capture; the session is still released and its dependencies closed.
#[tokio::test]
async fn runner_owned_cleanup_redacts_a_pre_stop_hook_failure() {
    let sentinel = "runner-pre-stop-hook-secret";
    let session = FakeSession::new(external_mount_manifest(sentinel));
    let shared = handle(&session);
    shared.register_pre_stop_hook(failing_hook(format!(
        "pre-stop hook failed with {sentinel}"
    )));
    let client = Arc::new(FakeClient {
        session: session.clone(),
        delete_calls: AtomicUsize::new(0),
        delete_error: None,
    });

    let _serial = SERIAL.lock().await;
    capture_logs();
    LOGS.lock().unwrap().clear();
    // A failed teardown is reported in the log; the finished run still returns its result.
    run_owned(client.clone()).await.unwrap();

    let text = LOGS.lock().unwrap().join("\n");
    assert!(
        text.contains("Failed to tear down the sandbox session"),
        "{text}"
    );
    assert!(!text.contains(sentinel), "{text}");
    assert_eq!(client.delete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(session.dependency_closes(), 1);
}

/// Python: `test_runner_owned_cleanup_redacts_client_delete_failure`.
#[tokio::test]
async fn runner_owned_cleanup_redacts_a_client_delete_failure() {
    let sentinel = "client-delete-secret";
    let session = FakeSession::new(external_mount_manifest(sentinel));
    let client = Arc::new(FakeClient {
        session: session.clone(),
        delete_calls: AtomicUsize::new(0),
        delete_error: Some(format!("delete failed with {sentinel}")),
    });

    let _serial = SERIAL.lock().await;
    capture_logs();
    LOGS.lock().unwrap().clear();
    // A failed teardown is reported in the log; the finished run still returns its result.
    run_owned(client.clone()).await.unwrap();

    let text = LOGS.lock().unwrap().join("\n");
    assert!(
        text.contains("Failed to tear down the sandbox session"),
        "{text}"
    );
    assert!(!text.contains(sentinel), "{text}");
    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(client.delete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.dependency_closes(), 1);
}

/// A clean owned run closes the session's dependencies exactly once.
#[tokio::test]
async fn a_completed_owned_run_closes_the_session_dependencies_once() {
    let session = FakeSession::new(Manifest::new());
    let client = Arc::new(FakeClient {
        session: session.clone(),
        delete_calls: AtomicUsize::new(0),
        delete_error: None,
    });

    run_owned(client).await.unwrap();

    assert_eq!(session.dependency_closes(), 1);
}
