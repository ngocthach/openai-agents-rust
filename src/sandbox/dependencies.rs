//! A session-scoped dependency container. Mirrors `agents.sandbox.session.dependencies`.
//!
//! Most of what a sandbox session needs is passed by type: a client, a snapshot, a manifest. A
//! few things cannot be, because the application supplies them at runtime and the SDK only knows
//! them by name — a storage client a custom [`Snapshot`](crate::sandbox::snapshot::Snapshot)
//! talks to, a credential broker a manifest entry calls. This is the container for those: a
//! string-keyed map of `Arc<dyn Any + Send + Sync>`, with lazy factories, per-session caching and
//! an owned-resource lifecycle.
//!
//! A client holds a configured template and [`clone_template`](Dependencies::clone_template)s it
//! per session, so each session gets its own cache and closes its own owned resources.
//!
//! A cached factory runs as its own task, shared by everyone waiting on the key. A waiter that
//! goes away does not cancel the build for the others, and a build that fails is shared by all of
//! them and then forgotten, so the next request starts over. Rebinding a key or closing the
//! container cancels the build in flight.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::{BoxFuture, FutureExt, Shared};
use tokio::task::AbortHandle;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// A value held in the container.
pub type Dependency = Arc<dyn Any + Send + Sync>;

/// Builds a dependency the first time it is asked for.
pub type DependencyFactory =
    Arc<dyn Fn() -> BoxFuture<'static, SandboxResult<Dependency>> + Send + Sync>;

/// Releases a dependency the container owns.
pub type DependencyCloser = Arc<dyn Fn(Dependency) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone)]
enum Binding {
    Value(Dependency),
    Factory {
        /// Identifies this binding, so a build can tell whether it was replaced under it.
        id: u64,
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    },
}

type SharedBuild = Shared<BoxFuture<'static, SandboxResult<Dependency>>>;

/// A cached build in flight.
struct PendingBuild {
    binding_id: u64,
    build: SharedBuild,
    abort: AbortHandle,
}

#[derive(Default)]
struct State {
    cache: BTreeMap<String, Dependency>,
    pending: BTreeMap<String, PendingBuild>,
    owned: Vec<(Dependency, DependencyCloser)>,
}

#[derive(Default)]
struct Inner {
    bindings: Mutex<BTreeMap<String, Binding>>,
    state: Mutex<State>,
    next_id: AtomicU64,
    closed: AtomicBool,
    /// The one close in progress or finished; later calls wait on it instead of starting another.
    close_task: Mutex<Option<Shared<BoxFuture<'static, ()>>>>,
}

/// A session-scoped container of runtime objects, keyed by name.
#[derive(Default)]
pub struct Dependencies {
    inner: Arc<Inner>,
}

impl fmt::Debug for Dependencies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<String> = self
            .inner
            .bindings
            .lock()
            .map(|bindings| bindings.keys().cloned().collect())
            .unwrap_or_default();
        f.debug_struct("Dependencies").field("keys", &keys).finish()
    }
}

fn config_error(message: String) -> SandboxError {
    SandboxError::new(ErrorCode::SandboxConfigInvalid, message)
}

impl Inner {
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Whether `id` is still the binding under `key`.
    fn is_current(&self, key: &str, id: u64) -> bool {
        matches!(
            self.bindings.lock().expect("dependency bindings lock").get(key),
            Some(Binding::Factory { id: current, .. }) if *current == id
        )
    }

    /// Fail a finished resolution that the container has moved on from.
    fn check_still_valid(&self, key: &str, id: u64) -> SandboxResult<()> {
        if self.is_closed() {
            return Err(config_error(format!(
                "dependencies container closed while resolving `{key}`"
            )));
        }
        if !self.is_current(key, id) {
            return Err(config_error(format!(
                "dependency `{key}` was rebound while its factory was resolving"
            )));
        }
        Ok(())
    }
}

/// Run one factory and record what it produced.
///
/// The result is registered while both locks are held, so it either lands before a close or a
/// rebind looks at the container, or it sees them and is cleaned up here: a value built too late
/// is released rather than handed to a caller nothing will clean up after.
async fn run_factory(
    inner: Arc<Inner>,
    key: String,
    id: u64,
    factory: DependencyFactory,
    cache: bool,
    closer: Option<DependencyCloser>,
) -> SandboxResult<Dependency> {
    /// Forget this build when it ends, however it ends.
    struct Forget {
        inner: Arc<Inner>,
        key: String,
        id: u64,
    }
    impl Drop for Forget {
        fn drop(&mut self) {
            let mut state = self.inner.state.lock().expect("dependency state lock");
            if state
                .pending
                .get(&self.key)
                .is_some_and(|pending| pending.binding_id == self.id)
            {
                state.pending.remove(&self.key);
            }
        }
    }
    let _forget = cache.then(|| Forget {
        inner: inner.clone(),
        key: key.clone(),
        id,
    });

    let value = factory().await?;

    let too_late = {
        let bindings = inner.bindings.lock().expect("dependency bindings lock");
        let mut state = inner.state.lock().expect("dependency state lock");
        if inner.is_closed() {
            true
        } else {
            // An owned result stays owned even when its binding was replaced meanwhile: the
            // replacement may hand back the same object, and close releases it once.
            if let Some(closer) = &closer {
                state.owned.push((value.clone(), closer.clone()));
            }
            let current = matches!(
                bindings.get(&key),
                Some(Binding::Factory { id: current, .. }) if *current == id
            );
            if cache && current {
                state.cache.insert(key.clone(), value.clone());
            }
            false
        }
    };
    if too_late && let Some(closer) = closer {
        closer(value.clone()).await;
    }
    inner.check_still_valid(&key, id)?;
    Ok(value)
}

impl Dependencies {
    /// An empty container.
    pub fn new() -> Self {
        Dependencies::default()
    }

    /// A container holding the given ready values. Mirrors Python's `Dependencies.with_values`.
    ///
    /// Errors the same way [`bind_value`](Self::bind_value) does: an empty key, or the same key
    /// twice.
    pub fn with_values<K: Into<String>>(
        values: impl IntoIterator<Item = (K, Dependency)>,
    ) -> SandboxResult<Self> {
        let dependencies = Dependencies::new();
        for (key, value) in values {
            dependencies.bind_value(key, value)?;
        }
        Ok(dependencies)
    }

    /// Bind a ready value under `key`.
    pub fn bind_value(&self, key: impl Into<String>, value: Dependency) -> SandboxResult<()> {
        self.bind(key.into(), Binding::Value(value), false)
    }

    /// Bind a ready value, replacing any existing binding.
    pub fn rebind_value(&self, key: impl Into<String>, value: Dependency) -> SandboxResult<()> {
        self.bind(key.into(), Binding::Value(value), true)
    }

    /// Bind a factory that builds the value the first time it is required.
    ///
    /// With `cache` the value is built once per container. With `closer` the container owns the
    /// result and releases it in [`close`](Self::close), newest first.
    pub fn bind_factory(
        &self,
        key: impl Into<String>,
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    ) -> SandboxResult<()> {
        self.bind(
            key.into(),
            Self::factory_binding(factory, cache, closer),
            false,
        )
    }

    /// Bind a factory, replacing any existing binding. Mirrors Python's `overwrite=True`.
    ///
    /// A build already in flight for `key` is cancelled and its waiters fail with a "rebound"
    /// error. A value already cached or owned stays owned until [`close`](Self::close), so a
    /// result the container built is still released exactly once.
    pub fn rebind_factory(
        &self,
        key: impl Into<String>,
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    ) -> SandboxResult<()> {
        self.bind(
            key.into(),
            Self::factory_binding(factory, cache, closer),
            true,
        )
    }

    fn factory_binding(
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    ) -> Binding {
        Binding::Factory {
            id: 0,
            factory,
            cache,
            closer,
        }
    }

    fn bind(&self, key: String, mut binding: Binding, overwrite: bool) -> SandboxResult<()> {
        if key.is_empty() {
            return Err(config_error("dependency key must be non-empty".to_string()));
        }
        if let Binding::Factory { id, .. } = &mut binding {
            *id = self.inner.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        }
        let mut bindings = self
            .inner
            .bindings
            .lock()
            .expect("dependency bindings lock");
        if !overwrite && bindings.contains_key(&key) {
            return Err(config_error(format!("dependency `{key}` is already bound")));
        }
        bindings.insert(key.clone(), binding);
        // Dropping the cached value and cancelling the build in flight happens under the
        // bindings lock, so a build that finishes next sees that it was replaced.
        let pending = {
            let mut state = self.inner.state.lock().expect("dependency state lock");
            state.cache.remove(&key);
            state.pending.remove(&key)
        };
        if let Some(pending) = pending {
            // After the state lock is released: dropping the build runs its cleanup, which
            // takes that lock.
            pending.abort.abort();
        }
        Ok(())
    }

    /// Whether anything is bound under `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.inner
            .bindings
            .lock()
            .expect("dependency bindings lock")
            .contains_key(key)
    }

    /// Resolve `key`, building it if it is a factory binding.
    pub async fn get(&self, key: &str) -> SandboxResult<Option<Dependency>> {
        let binding = {
            let bindings = self
                .inner
                .bindings
                .lock()
                .expect("dependency bindings lock");
            bindings.get(key).cloned()
        };
        let Some(binding) = binding else {
            return Ok(None);
        };
        match binding {
            Binding::Value(value) => Ok(Some(value)),
            Binding::Factory {
                id,
                factory,
                cache,
                closer,
            } => self
                .resolve_factory(key, id, factory, cache, closer)
                .await
                .map(Some),
        }
    }

    async fn resolve_factory(
        &self,
        key: &str,
        id: u64,
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    ) -> SandboxResult<Dependency> {
        let inner = self.inner.clone();
        if inner.is_closed() {
            return Err(config_error(format!(
                "dependencies container is closed; cannot resolve `{key}`"
            )));
        }
        if !cache {
            return run_factory(inner, key.to_string(), id, factory, false, closer).await;
        }
        let build = {
            let mut state = inner.state.lock().expect("dependency state lock");
            if let Some(value) = state.cache.get(key) {
                return Ok(value.clone());
            }
            match state.pending.get(key) {
                Some(pending) if pending.binding_id == id => pending.build.clone(),
                _ => {
                    let handle = tokio::spawn(run_factory(
                        inner.clone(),
                        key.to_string(),
                        id,
                        factory,
                        true,
                        closer,
                    ));
                    let abort = handle.abort_handle();
                    let build: SharedBuild = async move {
                        match handle.await {
                            Ok(result) => result,
                            Err(_) => Err(config_error(
                                "the dependency factory was cancelled".to_string(),
                            )),
                        }
                    }
                    .boxed()
                    .shared();
                    state.pending.insert(
                        key.to_string(),
                        PendingBuild {
                            binding_id: id,
                            build: build.clone(),
                            abort,
                        },
                    );
                    build
                }
            }
        };
        // A waiter that is dropped here leaves the build running for the rest.
        match build.await {
            Ok(value) => {
                inner.check_still_valid(key, id)?;
                Ok(value)
            }
            Err(error) => {
                // A cancelled build is reported as what cancelled it.
                inner.check_still_valid(key, id)?;
                Err(error)
            }
        }
    }

    /// Resolve `key`, failing with a message naming the consumer when it is unbound.
    pub async fn require(&self, key: &str, consumer: &str) -> SandboxResult<Dependency> {
        match self.get(key).await? {
            Some(value) => Ok(value),
            None => Err(config_error(format!(
                "missing dependency `{key}` for {consumer}; bind it on a `Dependencies` and \
                 pass it to the sandbox client"
            ))),
        }
    }

    /// Resolve `key` and downcast it to `T`.
    pub async fn get_as<T: Send + Sync + 'static>(
        &self,
        key: &str,
    ) -> SandboxResult<Option<Arc<T>>> {
        match self.get(key).await? {
            None => Ok(None),
            Some(value) => value.downcast::<T>().map(Some).map_err(|_| {
                config_error(format!(
                    "dependency `{key}` is not a `{}`",
                    std::any::type_name::<T>()
                ))
            }),
        }
    }

    /// A fresh container with the same bindings and an empty cache.
    ///
    /// Mirrors Python's `clone`: a client keeps the template, every session gets its own cache
    /// and its own owned-resource lifetime.
    pub fn clone_template(&self) -> Dependencies {
        let cloned = Dependencies::new();
        let bindings = self
            .inner
            .bindings
            .lock()
            .expect("dependency bindings lock")
            .clone();
        let mut fresh = cloned
            .inner
            .bindings
            .lock()
            .expect("dependency bindings lock");
        for (key, mut binding) in bindings {
            if let Binding::Factory { id, .. } = &mut binding {
                *id = cloned.inner.next_id.fetch_add(1, Ordering::SeqCst) + 1;
            }
            fresh.insert(key, binding);
        }
        drop(fresh);
        cloned
    }

    /// Release every owned result, newest first, and clear the cache.
    ///
    /// Builds in flight are cancelled; one that had already produced a value releases it instead
    /// of handing it to its waiters, who fail with a "closed" error. Idempotent: the first call
    /// starts the close and every later call waits for that same one. The close runs as its own
    /// task, so a caller that stops waiting does not stop a release that is already under way.
    /// A value the container owns twice (one object bound under two keys) is released once.
    pub async fn close(&self) {
        let close = {
            let mut slot = self.inner.close_task.lock().expect("dependency close lock");
            match &*slot {
                Some(existing) => existing.clone(),
                None => {
                    self.inner.closed.store(true, Ordering::SeqCst);
                    let handle = tokio::spawn(close_owned(self.inner.clone()));
                    let close: Shared<BoxFuture<'static, ()>> = async move {
                        let _ = handle.await;
                    }
                    .boxed()
                    .shared();
                    *slot = Some(close.clone());
                    close
                }
            }
        };
        close.await;
    }
}

async fn close_owned(inner: Arc<Inner>) {
    let (aborts, owned) = {
        let mut state = inner.state.lock().expect("dependency state lock");
        let aborts: Vec<AbortHandle> = std::mem::take(&mut state.pending)
            .into_values()
            .map(|pending| pending.abort)
            .collect();
        state.cache.clear();
        (aborts, std::mem::take(&mut state.owned))
    };
    for abort in aborts {
        abort.abort();
    }
    let mut released: Vec<Dependency> = Vec::new();
    for (value, closer) in owned.into_iter().rev() {
        if released.iter().any(|seen| Arc::ptr_eq(seen, &value)) {
            continue;
        }
        released.push(value.clone());
        closer(value).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn a_cached_factory_runs_once_per_container() {
        let calls = Arc::new(AtomicUsize::new(0));
        let template = Dependencies::new();
        let counted = calls.clone();
        template
            .bind_factory(
                "client",
                Arc::new(move || {
                    let counted = counted.clone();
                    Box::pin(async move {
                        counted.fetch_add(1, Ordering::SeqCst);
                        Ok(Arc::new(7usize) as Dependency)
                    })
                }),
                true,
                None,
            )
            .unwrap();

        let session = template.clone_template();
        assert_eq!(
            *session.get_as::<usize>("client").await.unwrap().unwrap(),
            7
        );
        assert_eq!(
            *session.get_as::<usize>("client").await.unwrap().unwrap(),
            7
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // A second session has its own cache, so the factory runs again for it.
        let other = template.clone_template();
        assert!(other.get("client").await.unwrap().is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn owned_results_are_closed_newest_first() {
        let order = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let dependencies = Dependencies::new();
        for index in [1usize, 2] {
            let closed = order.clone();
            dependencies
                .bind_factory(
                    format!("resource-{index}"),
                    Arc::new(move || Box::pin(async move { Ok(Arc::new(index) as Dependency) })),
                    true,
                    Some(Arc::new(move |value: Dependency| {
                        let closed = closed.clone();
                        Box::pin(async move {
                            let index = *value.downcast::<usize>().expect("usize");
                            closed.lock().expect("order").push(index);
                        })
                    })),
                )
                .unwrap();
        }
        dependencies.get("resource-1").await.unwrap();
        dependencies.get("resource-2").await.unwrap();
        dependencies.close().await;
        assert_eq!(*order.lock().expect("order"), vec![2, 1]);
        // A closed container refuses to build anything new.
        assert!(dependencies.get("resource-1").await.is_err());
    }

    #[tokio::test]
    async fn rebinding_requires_an_explicit_overwrite() {
        let dependencies = Dependencies::new();
        dependencies
            .bind_value("token", Arc::new("first".to_string()) as Dependency)
            .unwrap();
        assert!(
            dependencies
                .bind_value("token", Arc::new("second".to_string()) as Dependency)
                .is_err()
        );
        dependencies
            .rebind_value("token", Arc::new("second".to_string()) as Dependency)
            .unwrap();
        assert_eq!(
            dependencies
                .get_as::<String>("token")
                .await
                .unwrap()
                .unwrap()
                .as_str(),
            "second"
        );
        assert!(dependencies.get("absent").await.unwrap().is_none());
        assert!(dependencies.require("absent", "a test").await.is_err());
    }
}
