//! Session-owned cleanup: pre-stop hooks, `aclose`, and dependency teardown. Mirrors
//! `BaseSandboxSession.register_pre_stop_hook`, `run_pre_stop_hooks` and `aclose`.
//!
//! Python keeps the hook list on the session object. A Rust session is any
//! [`SandboxSession`] implementation, so the state lives in a registry keyed by the live
//! `Arc` the session is shared through: register, run and close with the same handle (or a
//! clone of it). An entry disappears once the last handle is dropped.
//!
//! Hooks run once, in registration order, before the workspace is persisted. A failing hook
//! does not stop the others; the first failure is reported afterwards, and persistence is
//! skipped from then on, so a snapshot is never taken of a workspace a hook failed to quiesce.
//! Failures are redacted when the manifest carries credentialed mounts, as Python's
//! `redact_mount_error_data` does.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use futures::future::BoxFuture;

use crate::sandbox::errors::SandboxResult;
use crate::sandbox::sandboxes::docker::redact_for_protected_mounts;
use crate::sandbox::session::{SandboxSession, persist_session_snapshot};

/// An async hook run once before the session workspace is persisted.
pub type PreStopHook = Arc<dyn Fn() -> BoxFuture<'static, SandboxResult<()>> + Send + Sync>;

#[derive(Default)]
struct Lifecycle {
    hooks: Mutex<Vec<PreStopHook>>,
    ran: AtomicBool,
    failed: AtomicBool,
    hook_lock: tokio::sync::Mutex<()>,
    aclose_lock: tokio::sync::Mutex<()>,
}

type Registry = HashMap<usize, (Weak<dyn SandboxSession>, Arc<Lifecycle>)>;

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn lifecycle_of(session: &Arc<dyn SandboxSession>) -> Arc<Lifecycle> {
    let key = Arc::as_ptr(session) as *const () as usize;
    let mut registry = REGISTRY.lock().expect("session lifecycle registry");
    registry.retain(|_, (weak, _)| weak.strong_count() > 0);
    registry
        .entry(key)
        .or_insert_with(|| (Arc::downgrade(session), Arc::new(Lifecycle::default())))
        .1
        .clone()
}

/// Register `hook` to run once before `session`'s workspace is persisted. Registering after
/// the hooks already ran arms them to run again.
pub fn register_pre_stop_hook(session: &Arc<dyn SandboxSession>, hook: PreStopHook) {
    let lifecycle = lifecycle_of(session);
    lifecycle.hooks.lock().expect("pre-stop hooks").push(hook);
    lifecycle.ran.store(false, Ordering::SeqCst);
}

/// Run the registered pre-stop hooks once. A later call is a no-op, and so is a call with no
/// hooks. Every hook runs even when an earlier one failed; the first failure is returned.
pub async fn run_pre_stop_hooks(session: &Arc<dyn SandboxSession>) -> SandboxResult<()> {
    let lifecycle = lifecycle_of(session);
    let _guard = lifecycle.hook_lock.lock().await;
    let hooks: Vec<PreStopHook> = lifecycle.hooks.lock().expect("pre-stop hooks").clone();
    if hooks.is_empty() || lifecycle.ran.load(Ordering::SeqCst) {
        return Ok(());
    }
    lifecycle.ran.store(true, Ordering::SeqCst);
    let mut first = None;
    for hook in hooks {
        if let Err(error) = hook().await
            && first.is_none()
        {
            first = Some(error);
        }
    }
    match first {
        None => Ok(()),
        Some(error) => {
            lifecycle.failed.store(true, Ordering::SeqCst);
            Err(redact_for_protected_mounts(session.manifest(), error))
        }
    }
}

/// Whether a pre-stop hook of `session` has failed, which stops its workspace from being
/// persisted.
pub fn pre_stop_hooks_failed(session: &Arc<dyn SandboxSession>) -> bool {
    lifecycle_of(session).failed.load(Ordering::SeqCst)
}

/// Close the session-scoped dependencies, once. The container itself makes later calls wait
/// for the same close.
pub(crate) async fn close_dependencies(session: &Arc<dyn SandboxSession>) {
    if let Some(dependencies) = session.dependencies() {
        dependencies.close().await;
    }
}

/// Run the session's own cleanup outside a runner: pre-stop hooks, then (when they all
/// succeeded) the snapshot capture and `stop`, then `shutdown`, then the dependencies close.
/// Every later step runs even after an earlier failure, and the first failure is returned.
///
/// If the session came from a [`SandboxClient`](crate::sandbox::SandboxClient), call the
/// client's `delete` separately for backend-specific removal. Concurrent calls queue behind
/// each other. Mirrors `BaseSandboxSession.aclose`.
pub async fn aclose_session(session: &Arc<dyn SandboxSession>) -> SandboxResult<()> {
    let lifecycle = lifecycle_of(session);
    let _guard = lifecycle.aclose_lock.lock().await;
    let mut first = None;
    if let Err(error) = run_pre_stop_hooks(session).await {
        first = Some(error);
    }
    // A failed capture or stop skips the shutdown, as Python's single `try` does; a failed
    // pre-stop hook skips only the capture and the stop.
    let torn_down = async {
        if first.is_none() && !lifecycle.failed.load(Ordering::SeqCst) {
            persist_session_snapshot(session.as_ref()).await?;
            session.stop().await?;
        }
        session.shutdown().await
    }
    .await;
    if let Err(error) = torn_down
        && first.is_none()
    {
        first = Some(redact_for_protected_mounts(session.manifest(), error));
    }
    close_dependencies(session).await;
    first.map_or(Ok(()), Err)
}

/// `register_pre_stop_hook`, `run_pre_stop_hooks` and `aclose` as methods of the shared
/// session handle.
pub trait SandboxSessionLifecycle {
    /// See [`register_pre_stop_hook`].
    fn register_pre_stop_hook(&self, hook: PreStopHook);
    /// See [`run_pre_stop_hooks`].
    fn run_pre_stop_hooks(&self) -> BoxFuture<'_, SandboxResult<()>>;
    /// See [`aclose_session`].
    fn aclose(&self) -> BoxFuture<'_, SandboxResult<()>>;
}

impl SandboxSessionLifecycle for Arc<dyn SandboxSession> {
    fn register_pre_stop_hook(&self, hook: PreStopHook) {
        register_pre_stop_hook(self, hook);
    }

    fn run_pre_stop_hooks(&self) -> BoxFuture<'_, SandboxResult<()>> {
        Box::pin(run_pre_stop_hooks(self))
    }

    fn aclose(&self) -> BoxFuture<'_, SandboxResult<()>> {
        Box::pin(aclose_session(self))
    }
}
