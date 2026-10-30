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

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::sync::Mutex;

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
        factory: DependencyFactory,
        cache: bool,
        closer: Option<DependencyCloser>,
    },
}

/// A session-scoped container of runtime objects, keyed by name.
#[derive(Default)]
pub struct Dependencies {
    bindings: std::sync::Mutex<BTreeMap<String, Binding>>,
    cache: Mutex<BTreeMap<String, Dependency>>,
    owned: Mutex<Vec<(Dependency, DependencyCloser)>>,
    closed: std::sync::atomic::AtomicBool,
}

impl fmt::Debug for Dependencies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<String> = self
            .bindings
            .lock()
            .map(|bindings| bindings.keys().cloned().collect())
            .unwrap_or_default();
        f.debug_struct("Dependencies").field("keys", &keys).finish()
    }
}

impl Dependencies {
    /// An empty container.
    pub fn new() -> Self {
        Dependencies::default()
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
            Binding::Factory {
                factory,
                cache,
                closer,
            },
            false,
        )
    }

    fn bind(&self, key: String, binding: Binding, overwrite: bool) -> SandboxResult<()> {
        if key.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "dependency key must be non-empty",
            ));
        }
        let mut bindings = self.bindings.lock().expect("dependency bindings lock");
        if !overwrite && bindings.contains_key(&key) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("dependency `{key}` is already bound"),
            ));
        }
        bindings.insert(key, binding);
        Ok(())
    }

    /// Whether anything is bound under `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.bindings
            .lock()
            .expect("dependency bindings lock")
            .contains_key(key)
    }

    /// Resolve `key`, building it if it is a factory binding.
    pub async fn get(&self, key: &str) -> SandboxResult<Option<Dependency>> {
        let binding = {
            let bindings = self.bindings.lock().expect("dependency bindings lock");
            bindings.get(key).cloned()
        };
        let Some(binding) = binding else {
            return Ok(None);
        };
        match binding {
            Binding::Value(value) => Ok(Some(value)),
            Binding::Factory {
                factory,
                cache,
                closer,
            } => {
                if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                    return Err(SandboxError::new(
                        ErrorCode::SandboxConfigInvalid,
                        format!("dependencies container is closed; cannot resolve `{key}`"),
                    ));
                }
                // The cache lock is held across the factory call on purpose: two concurrent
                // resolutions of the same key must not both build, because an owned result
                // built twice would leak the copy nothing closes.
                let mut cached = self.cache.lock().await;
                if cache && let Some(value) = cached.get(key) {
                    return Ok(Some(value.clone()));
                }
                let value = factory().await?;
                if let Some(closer) = closer {
                    self.owned.lock().await.push((value.clone(), closer));
                }
                if cache {
                    cached.insert(key.to_string(), value.clone());
                }
                Ok(Some(value))
            }
        }
    }

    /// Resolve `key`, failing with a message naming the consumer when it is unbound.
    pub async fn require(&self, key: &str, consumer: &str) -> SandboxResult<Dependency> {
        match self.get(key).await? {
            Some(value) => Ok(value),
            None => Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "missing dependency `{key}` for {consumer}; bind it on a `Dependencies` and \
                     pass it to the sandbox client"
                ),
            )),
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
                SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!(
                        "dependency `{key}` is not a `{}`",
                        std::any::type_name::<T>()
                    ),
                )
            }),
        }
    }

    /// A fresh container with the same bindings and an empty cache.
    ///
    /// Mirrors Python's `clone`: a client keeps the template, every session gets its own cache
    /// and its own owned-resource lifetime.
    pub fn clone_template(&self) -> Dependencies {
        let bindings = self
            .bindings
            .lock()
            .expect("dependency bindings lock")
            .clone();
        Dependencies {
            bindings: std::sync::Mutex::new(bindings),
            cache: Mutex::new(BTreeMap::new()),
            owned: Mutex::new(Vec::new()),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Release every owned result, newest first, and clear the cache.
    pub async fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        let owned = std::mem::take(&mut *self.owned.lock().await);
        for (value, closer) in owned.into_iter().rev() {
            closer(value).await;
        }
        self.cache.lock().await.clear();
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
