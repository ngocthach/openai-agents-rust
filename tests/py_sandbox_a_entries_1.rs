//! Ports of the Python sandbox suite's `tests/sandbox/test_materialization.py` and
//! `tests/sandbox/test_dependencies.py`.
//!
//! `test_materialization.py` covers `agents.sandbox.materialization.gather_in_order`: the bounded,
//! order-preserving task runner every manifest materialization funnels its round trips through.
//! `test_dependencies.py` covers the session-scoped dependency container — binding, lazy
//! factories, per-container caching and the owned-resource lifecycle.
#![cfg(feature = "sandbox")]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use openai_agents_sdk::sandbox::dependencies::{Dependencies, Dependency, DependencyCloser};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::materialization::gather_in_order;

/// The boxed future both the dependency factories and the closers hand back. Spelled out rather
/// than imported because `futures` is not a dev-dependency of this crate.
type BoxFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Yield until `condition` holds, so a spawned task can make progress, with a bound so a test
/// that will never settle fails instead of hanging the suite.
async fn yield_until(label: &str, mut condition: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if condition() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("timed out waiting for {label}");
}

// ---------------------------------------------------------------------------
// test_materialization.py
// ---------------------------------------------------------------------------

/// Python's `test_gather_in_order_limits_concurrency_and_preserves_order`.
///
/// Five tasks, a limit of two: only the first two may start before anything is released, no more
/// than two may ever run at once, and the results come back in the order the factories were
/// given rather than the order the tasks happened to finish in.
#[tokio::test]
async fn gather_in_order_limits_concurrency_and_preserves_order() {
    let active_tasks = Arc::new(AtomicUsize::new(0));
    let max_active_tasks = Arc::new(AtomicUsize::new(0));
    let started_tasks: Arc<StdMutex<Vec<usize>>> = Arc::new(StdMutex::new(Vec::new()));
    let (release, release_rx) = tokio::sync::watch::channel(false);

    let task_factories: Vec<_> = (0..5usize)
        .map(|index| {
            let active_tasks = active_tasks.clone();
            let max_active_tasks = max_active_tasks.clone();
            let started_tasks = started_tasks.clone();
            let mut released = release_rx.clone();
            move || async move {
                let now = active_tasks.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_tasks.fetch_max(now, Ordering::SeqCst);
                started_tasks.lock().expect("started").push(index);
                while !*released.borrow() {
                    released.changed().await.expect("release channel");
                }
                active_tasks.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, openai_agents_sdk::sandbox::errors::SandboxError>(format!("result-{index}"))
            }
        })
        .collect();

    let gather_task = tokio::spawn(gather_in_order(task_factories, Some(2)));

    let observed = started_tasks.clone();
    yield_until("two tasks to start", || {
        observed.lock().expect("started").len() >= 2
    })
    .await;

    assert_eq!(*started_tasks.lock().expect("started"), vec![0, 1]);
    assert_eq!(max_active_tasks.load(Ordering::SeqCst), 2);

    release.send(true).expect("release");
    let result = gather_task.await.expect("join").expect("gather");

    assert_eq!(
        result,
        vec!["result-0", "result-1", "result-2", "result-3", "result-4",]
    );
    assert_eq!(max_active_tasks.load(Ordering::SeqCst), 2);
}

/// Python's `test_gather_in_order_rejects_invalid_concurrency`: the limit is validated before the
/// empty-input shortcut, so even an empty task list reports the impossible limit.
#[tokio::test]
async fn gather_in_order_rejects_invalid_concurrency() {
    type NoTasks = fn() -> std::future::Ready<SandboxResult<&'static str>>;

    let error = gather_in_order(std::iter::empty::<NoTasks>(), Some(0))
        .await
        .expect_err("a zero limit must be rejected");

    assert_eq!(error.message, "max_concurrency must be at least 1");
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);

    // Without the impossible limit an empty task list is simply an empty result.
    let empty = gather_in_order(std::iter::empty::<NoTasks>(), Some(4))
        .await
        .expect("an empty task list is not an error");
    assert!(empty.is_empty());
}

/// Not in the Python file, but the property the two call sites that were refactored onto
/// `gather_in_order` depend on: the first error short-circuits, so the factories after it are
/// never called at all.
#[tokio::test]
async fn gather_in_order_stops_calling_factories_after_the_first_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let factories: Vec<_> = (0..5usize)
        .map(|index| {
            let calls = calls.clone();
            move || async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if index == 1 {
                    return Err(openai_agents_sdk::sandbox::errors::SandboxError::new(
                        ErrorCode::LocalFileReadError,
                        "boom",
                    ));
                }
                Ok(index)
            }
        })
        .collect();

    let error = gather_in_order(factories, Some(1))
        .await
        .expect_err("the failing factory must surface");
    assert_eq!(error.message, "boom");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// test_dependencies.py — helpers mirroring the Python test module's closables
// ---------------------------------------------------------------------------

/// Python's `_AsyncClosable`: counts how many times the container released it.
#[derive(Default)]
struct AsyncClosable {
    calls: AtomicUsize,
}

impl AsyncClosable {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// Python's `_BlockingAsyncClosable`: records that the release started, blocks until released,
/// and records that it ran to completion.
#[derive(Default)]
struct BlockingAsyncClosable {
    calls: AtomicUsize,
    completed: AtomicBool,
    started: AtomicBool,
}

/// The closer for an [`AsyncClosable`], as Python's `owns_result=True` installs.
fn async_closer() -> DependencyCloser {
    Arc::new(|value: Dependency| {
        Box::pin(async move {
            let closable = value.downcast::<AsyncClosable>().expect("AsyncClosable");
            closable.calls.fetch_add(1, Ordering::SeqCst);
        }) as BoxFut<()>
    })
}

/// A factory that hands back `value` without doing any work, like Python's
/// `lambda _dependencies: value`.
fn value_factory<T: Send + Sync + 'static>(
    value: Arc<T>,
) -> openai_agents_sdk::sandbox::dependencies::DependencyFactory {
    Arc::new(move || {
        let value = value.clone();
        Box::pin(async move { Ok(value as Dependency) }) as BoxFut<SandboxResult<Dependency>>
    })
}

// ---------------------------------------------------------------------------
// test_dependencies.py
// ---------------------------------------------------------------------------

/// Python's `test_dependencies_with_values_binds_multiple_values`.
#[tokio::test]
async fn dependencies_with_values_binds_multiple_values() {
    let key1 = "tests.with_values.str";
    let key2 = "tests.with_values.int";
    let dependencies = Dependencies::with_values([
        (key1, Arc::new("hello".to_string()) as Dependency),
        (key2, Arc::new(123usize) as Dependency),
    ])
    .expect("bind");

    let first = dependencies.require(key1, "test").await.expect("require");
    assert_eq!(
        first.downcast::<String>().expect("String").as_str(),
        "hello"
    );
    let second = dependencies.require(key2, "test").await.expect("require");
    assert_eq!(*second.downcast::<usize>().expect("usize"), 123);
}

/// Python's `test_dependencies_bind_value_and_require`: both `get` and `require` see a bound
/// value, and neither consults a factory.
#[tokio::test]
async fn dependencies_bind_value_and_require() {
    let dependencies = Dependencies::new();
    let key = "tests.value";
    dependencies
        .bind_value(key, Arc::new("hello".to_string()) as Dependency)
        .expect("bind");

    let via_get = dependencies.get(key).await.expect("get").expect("bound");
    assert_eq!(
        via_get.downcast::<String>().expect("String").as_str(),
        "hello"
    );
    let via_require = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        via_require.downcast::<String>().expect("String").as_str(),
        "hello"
    );
}

/// Python's `test_dependencies_missing_dependency_includes_key_and_consumer`.
#[tokio::test]
async fn dependencies_missing_dependency_includes_key_and_consumer() {
    let dependencies = Dependencies::new();

    let error = dependencies
        .require("tests.missing", "SedimentFile")
        .await
        .expect_err("an unbound key must not resolve");

    assert!(
        error.message.contains("tests.missing"),
        "message names the key: {}",
        error.message
    );
    assert!(
        error.message.contains("SedimentFile"),
        "message names the consumer: {}",
        error.message
    );
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    // `get` on the same key is simply absent rather than an error.
    assert!(
        dependencies
            .get("tests.missing")
            .await
            .expect("get")
            .is_none()
    );
}

/// Python's `test_dependencies_duplicate_binding_raises`.
#[tokio::test]
async fn dependencies_duplicate_binding_raises() {
    let dependencies = Dependencies::new();
    let key = "tests.dup";
    dependencies
        .bind_value(key, Arc::new("a".to_string()) as Dependency)
        .expect("bind");

    let error = dependencies
        .bind_value(key, Arc::new("b".to_string()) as Dependency)
        .expect_err("a second binding must be refused");
    assert!(
        error.message.contains("already bound"),
        "message: {}",
        error.message
    );
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);

    // The explicit overwrite is what Python spells `overwrite=True`.
    dependencies
        .rebind_value(key, Arc::new("b".to_string()) as Dependency)
        .expect("rebind");
    let value = dependencies.require(key, "test").await.expect("require");
    assert_eq!(value.downcast::<String>().expect("String").as_str(), "b");
}

/// Python's `test_dependencies_empty_key_raises`, for both binding kinds.
#[tokio::test]
async fn dependencies_empty_key_raises() {
    let dependencies = Dependencies::new();

    let value_error = dependencies
        .bind_value("", Arc::new("x".to_string()) as Dependency)
        .expect_err("an empty key must be refused");
    assert!(
        value_error.message.contains("non-empty"),
        "message: {}",
        value_error.message
    );

    let factory_error = dependencies
        .bind_factory("", value_factory(Arc::new("x".to_string())), true, None)
        .expect_err("an empty key must be refused");
    assert!(
        factory_error.message.contains("non-empty"),
        "message: {}",
        factory_error.message
    );
}

/// Python's `test_dependencies_cached_factory_resolves_once`.
#[tokio::test]
async fn dependencies_cached_factory_resolves_once() {
    let dependencies = Dependencies::new();
    let key = "tests.cached_factory";
    let calls = Arc::new(AtomicUsize::new(0));

    let counted = calls.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let counted = counted.clone();
                Box::pin(async move {
                    let call = counted.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok(Arc::new(format!("value-{call}")) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            None,
        )
        .expect("bind");

    for _ in 0..2 {
        let value = dependencies.require(key, "test").await.expect("require");
        assert_eq!(
            value.downcast::<String>().expect("String").as_str(),
            "value-1"
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Python's `test_dependencies_uncached_factory_resolves_every_time`.
#[tokio::test]
async fn dependencies_uncached_factory_resolves_every_time() {
    let dependencies = Dependencies::new();
    let key = "tests.uncached_factory";
    let calls = Arc::new(AtomicUsize::new(0));

    let counted = calls.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let counted = counted.clone();
                Box::pin(async move {
                    let call = counted.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok(Arc::new(format!("value-{call}")) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            false,
            None,
        )
        .expect("bind");

    let first = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        first.downcast::<String>().expect("String").as_str(),
        "value-1"
    );
    let second = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        second.downcast::<String>().expect("String").as_str(),
        "value-2"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// Python's `test_dependencies_async_factory_supported`. Every Rust factory is async, so this is
/// the degenerate case: a factory that awaits nothing still resolves.
#[tokio::test]
async fn dependencies_async_factory_supported() {
    let dependencies = Dependencies::new();
    let key = "tests.async_factory";
    dependencies
        .bind_factory(
            key,
            Arc::new(|| {
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    Ok(Arc::new("async-value".to_string()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            None,
        )
        .expect("bind");

    let value = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        value.downcast::<String>().expect("String").as_str(),
        "async-value"
    );
}

/// Python's `test_dependencies_cached_factory_resolves_once_concurrently`: three concurrent
/// requirers of one cached, owned factory see one factory call, one shared value, and one
/// release at close.
#[tokio::test]
async fn dependencies_cached_factory_resolves_once_concurrently() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.concurrent_cached_factory";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(AtomicBool::new(false));
    let (release, release_rx) = tokio::sync::watch::channel(false);

    let counted = calls.clone();
    let signal = started.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let counted = counted.clone();
                let signal = signal.clone();
                let mut released = release_rx.clone();
                Box::pin(async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    signal.store(true, Ordering::SeqCst);
                    while !*released.borrow() {
                        released.changed().await.expect("release channel");
                    }
                    Ok(Arc::new(AsyncClosable::default()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");

    let tasks: Vec<_> = (0..3)
        .map(|_| {
            let dependencies = dependencies.clone();
            tokio::spawn(async move { dependencies.require(key, "test").await })
        })
        .collect();

    let observed = started.clone();
    yield_until("the factory to start", || observed.load(Ordering::SeqCst)).await;
    release.send(true).expect("release");

    let mut values = Vec::new();
    for task in tasks {
        values.push(task.await.expect("join").expect("require"));
    }

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(Arc::ptr_eq(&values[0], &values[1]));
    assert!(Arc::ptr_eq(&values[1], &values[2]));

    dependencies.close().await;
    let closable = values
        .remove(0)
        .downcast::<AsyncClosable>()
        .expect("AsyncClosable");
    assert_eq!(closable.calls(), 1);
}

/// Python's `test_dependencies_cached_factory_failure_allows_retry`: two concurrent requirers
/// share one failing build and both see its error; the failure is not cached, so a later request
/// builds again and succeeds.
#[tokio::test]
async fn dependencies_cached_factory_failure_allows_retry() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.failed_factory_retry";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(AtomicBool::new(false));
    let (release, release_rx) = tokio::sync::watch::channel(false);

    let counted = calls.clone();
    let signal = started.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let counted = counted.clone();
                let signal = signal.clone();
                let mut released = release_rx.clone();
                Box::pin(async move {
                    let call = counted.fetch_add(1, Ordering::SeqCst) + 1;
                    if call == 1 {
                        signal.store(true, Ordering::SeqCst);
                        while !*released.borrow() {
                            released.changed().await.expect("release channel");
                        }
                        return Err(openai_agents_sdk::sandbox::errors::SandboxError::new(
                            ErrorCode::SandboxConfigInvalid,
                            "factory failed",
                        ));
                    }
                    Ok(Arc::new("recovered".to_string()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            None,
        )
        .expect("bind");

    let waiters: Vec<_> = (0..2)
        .map(|_| {
            let dependencies = dependencies.clone();
            tokio::spawn(async move { dependencies.require(key, "test").await.map(|_| ()) })
        })
        .collect();
    let observed = started.clone();
    yield_until("the factory to start", || observed.load(Ordering::SeqCst)).await;
    release.send(true).expect("release");

    for waiter in waiters {
        let error = waiter
            .await
            .expect("join")
            .expect_err("both waiters see the shared failure");
        assert!(
            error.message.contains("factory failed"),
            "message: {}",
            error.message
        );
    }

    let value = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        value.downcast::<String>().expect("String").as_str(),
        "recovered"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// Python's `test_dependencies_cached_factory_survives_waiter_cancellation`: one of two
/// requirers goes away mid-flight and the surviving one still gets the value, built once.
#[tokio::test]
async fn dependencies_cached_factory_survives_waiter_cancellation() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.cancelled_waiter";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(AtomicBool::new(false));
    let (release, release_rx) = tokio::sync::watch::channel(false);

    let counted = calls.clone();
    let signal = started.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let counted = counted.clone();
                let signal = signal.clone();
                let mut released = release_rx.clone();
                Box::pin(async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    signal.store(true, Ordering::SeqCst);
                    while !*released.borrow() {
                        released.changed().await.expect("release channel");
                    }
                    Ok(Arc::new("built".to_string()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            None,
        )
        .expect("bind");

    let cancelled_waiter = {
        let dependencies = dependencies.clone();
        tokio::spawn(async move { dependencies.require(key, "test").await.map(|_| ()) })
    };
    let surviving_waiter = {
        let dependencies = dependencies.clone();
        tokio::spawn(async move { dependencies.require(key, "test").await })
    };

    let observed = started.clone();
    yield_until("the factory to start", || observed.load(Ordering::SeqCst)).await;
    cancelled_waiter.abort();
    assert!(cancelled_waiter.await.expect_err("aborted").is_cancelled());

    release.send(true).expect("release");
    let value = surviving_waiter.await.expect("join").expect("require");
    assert_eq!(
        value.downcast::<String>().expect("String").as_str(),
        "built"
    );

    // The value stayed cached, so a later requirer sees the same one without a second build.
    let again = dependencies.require(key, "test").await.expect("require");
    assert_eq!(
        again.downcast::<String>().expect("String").as_str(),
        "built"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Python's `test_dependencies_close_before_waiter_resumes_rejects_closed_result`: a build that
/// has finished, but whose requirer has not yet been handed the value, is closed under it. The
/// requirer is told the container closed and the value it would have received was released.
///
/// Python orders this with the event loop; here the requirer's future is polled by hand, so the
/// build runs to completion while nobody is waiting on it and the close lands before the
/// requirer looks again.
#[tokio::test]
async fn dependencies_close_before_waiter_resumes_rejects_closed_result() {
    let dependencies = Dependencies::new();
    let key = "tests.close_before_waiter_resumes";
    let value = Arc::new(AsyncClosable::default());
    let produced = Arc::new(AtomicBool::new(false));

    let built = value.clone();
    let signal = produced.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let built = built.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    signal.store(true, Ordering::SeqCst);
                    Ok(built as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");

    let mut resolve = Box::pin(dependencies.require(key, "test"));
    // The first poll starts the build; the requirer is then left unpolled.
    assert!(futures::poll!(resolve.as_mut()).is_pending());
    let observed = produced.clone();
    yield_until("the factory to produce", || observed.load(Ordering::SeqCst)).await;

    dependencies.close().await;

    let error = resolve
        .await
        .expect_err("a result delivered after close must be rejected");
    assert!(
        error.message.contains("closed"),
        "message: {}",
        error.message
    );
    assert_eq!(value.calls(), 1, "the result is released by close");
}

/// Python's `test_dependencies_aclose_closes_owned_results_and_is_idempotent`.
///
/// Rust has one closer shape rather than Python's `aclose`/`close`/sync-`close` probing, so the
/// three Python bindings collapse to a cached owned factory and an uncached owned factory. The
/// behaviour asserted is the same: every produced value is released exactly once, uncached
/// factories produce distinct values and both are released, and a second close is a no-op.
#[tokio::test]
async fn dependencies_close_releases_owned_results_and_is_idempotent() {
    let dependencies = Dependencies::new();
    let cached_key = "tests.cached_owned";
    let uncached_key = "tests.uncached_owned";

    dependencies
        .bind_factory(
            cached_key,
            Arc::new(|| {
                Box::pin(async { Ok(Arc::new(AsyncClosable::default()) as Dependency) })
                    as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");
    dependencies
        .bind_factory(
            uncached_key,
            Arc::new(|| {
                Box::pin(async { Ok(Arc::new(AsyncClosable::default()) as Dependency) })
                    as BoxFut<SandboxResult<Dependency>>
            }),
            false,
            Some(async_closer()),
        )
        .expect("bind");

    let cached = dependencies
        .require(cached_key, "test")
        .await
        .expect("require");
    let first = dependencies
        .require(uncached_key, "test")
        .await
        .expect("require");
    let second = dependencies
        .require(uncached_key, "test")
        .await
        .expect("require");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "an uncached factory builds a fresh value each time"
    );

    dependencies.close().await;
    dependencies.close().await;

    for value in [cached, first, second] {
        let closable = value.downcast::<AsyncClosable>().expect("AsyncClosable");
        assert_eq!(closable.calls(), 1, "released exactly once");
    }
}

/// Python's `test_dependencies_bound_values_are_not_closed`: the container only releases what it
/// built, never what the caller handed it.
#[tokio::test]
async fn dependencies_bound_values_are_not_closed() {
    let dependencies = Dependencies::new();
    let key = "tests.bound_value";
    let value = Arc::new(AsyncClosable::default());
    dependencies
        .bind_value(key, value.clone() as Dependency)
        .expect("bind");

    let _ = dependencies.require(key, "test").await.expect("require");
    dependencies.close().await;

    assert_eq!(value.calls(), 0);
}

/// Python's `test_dependencies_aclose_continues_after_waiter_cancellation`: the caller that
/// started the close can go away, but a release that is already under way still finishes.
#[tokio::test]
async fn dependencies_close_continues_after_waiter_cancellation() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.cancelled_close";
    let value = Arc::new(BlockingAsyncClosable::default());
    let (release, release_rx) = tokio::sync::watch::channel(false);

    let closing = release_rx.clone();
    dependencies
        .bind_factory(
            key,
            value_factory(value.clone()),
            true,
            Some(Arc::new(move |held: Dependency| {
                let mut released = closing.clone();
                Box::pin(async move {
                    let closable = held
                        .downcast::<BlockingAsyncClosable>()
                        .expect("BlockingAsyncClosable");
                    closable.calls.fetch_add(1, Ordering::SeqCst);
                    closable.started.store(true, Ordering::SeqCst);
                    while !*released.borrow() {
                        released.changed().await.expect("release channel");
                    }
                    closable.completed.store(true, Ordering::SeqCst);
                }) as BoxFut<()>
            })),
        )
        .expect("bind");
    let _ = dependencies.require(key, "test").await.expect("require");

    let close_waiter = {
        let dependencies = dependencies.clone();
        tokio::spawn(async move { dependencies.close().await })
    };
    let observed = value.clone();
    yield_until("the closer to start", || {
        observed.started.load(Ordering::SeqCst)
    })
    .await;
    close_waiter.abort();
    assert!(close_waiter.await.expect_err("aborted").is_cancelled());

    release.send(true).expect("release");
    dependencies.close().await;

    assert_eq!(value.calls.load(Ordering::SeqCst), 1);
    assert!(
        value.completed.load(Ordering::SeqCst),
        "an abandoned close still finishes the release it started"
    );
}

/// Python's `test_dependencies_rebind_preserves_aliased_stale_result_until_close`: a build in
/// flight when its key is rebound is cancelled and its waiter fails with a "rebound" error; the
/// replacement resolves to the same object, which is released once, at close, not by the rebind.
#[tokio::test]
async fn dependencies_rebind_preserves_aliased_stale_result_until_close() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.rebind_aliased_result";
    let value = Arc::new(AsyncClosable::default());
    let started = Arc::new(AtomicBool::new(false));

    let signal = started.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let signal = signal.clone();
                Box::pin(async move {
                    signal.store(true, Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    Ok(Arc::new(()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");
    let stale_resolve = {
        let dependencies = dependencies.clone();
        tokio::spawn(async move { dependencies.require(key, "test").await.map(|_| ()) })
    };
    let observed = started.clone();
    yield_until("the factory to start", || observed.load(Ordering::SeqCst)).await;

    dependencies
        .rebind_factory(
            key,
            value_factory(value.clone()),
            true,
            Some(async_closer()),
        )
        .expect("rebind");

    let error = stale_resolve
        .await
        .expect("join")
        .expect_err("the stale waiter is told the key was rebound");
    assert!(
        error.message.contains("rebound"),
        "message: {}",
        error.message
    );
    let replacement = dependencies.require(key, "test").await.expect("require");
    assert!(Arc::ptr_eq(&replacement, &(value.clone() as Dependency)));
    assert_eq!(value.calls(), 0, "a rebind does not release anything");

    // Binding the same object under a second key makes the container own it twice.
    dependencies
        .bind_factory(
            "tests.rebind_alias",
            value_factory(value.clone()),
            true,
            Some(async_closer()),
        )
        .expect("bind alias");
    let _ = dependencies
        .require("tests.rebind_alias", "test")
        .await
        .expect("require");

    dependencies.close().await;
    assert_eq!(value.calls(), 1, "an aliased owned result is released once");
}

/// Python's `test_dependencies_close_cleans_up_owned_in_flight_result`, in Rust's terms.
///
/// Python cancels the in-flight factory, which catches the cancellation and returns a value that
/// the container then releases. A Rust factory cannot return from cancellation: dropping it ends
/// it. What carries over is the contract around it: closing cancels the build, the waiter fails
/// with a "closed" error, and nothing is left running.
#[tokio::test]
async fn dependencies_close_cancels_in_flight_builds() {
    let dependencies = Arc::new(Dependencies::new());
    let key = "tests.close_in_flight";
    let started = Arc::new(AtomicBool::new(false));

    let signal = started.clone();
    dependencies
        .bind_factory(
            key,
            Arc::new(move || {
                let signal = signal.clone();
                Box::pin(async move {
                    signal.store(true, Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    Ok(Arc::new(()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");
    let resolve_task = {
        let dependencies = dependencies.clone();
        tokio::spawn(async move { dependencies.require(key, "test").await.map(|_| ()) })
    };
    let observed = started.clone();
    yield_until("the factory to start", || observed.load(Ordering::SeqCst)).await;

    dependencies.close().await;

    let error = resolve_task
        .await
        .expect("join")
        .expect_err("a build cancelled by close fails its waiter");
    assert!(
        error.message.contains("closed"),
        "message: {}",
        error.message
    );
}

/// A closed container refuses to build anything new, as Python's `_resolve` does. Bound values
/// are still readable, because nothing has to be built to serve them.
#[tokio::test]
async fn dependencies_closed_container_refuses_to_build() {
    let dependencies = Dependencies::new();
    dependencies
        .bind_value("value", Arc::new(1usize) as Dependency)
        .expect("bind");
    dependencies
        .bind_factory(
            "built",
            value_factory(Arc::new(2usize)),
            true,
            Some(async_closer()),
        )
        .expect("bind");

    dependencies.close().await;

    let error = dependencies
        .get("built")
        .await
        .expect_err("a closed container cannot build");
    assert!(
        error.message.contains("closed"),
        "message: {}",
        error.message
    );
    assert!(dependencies.get("value").await.expect("get").is_some());
}

/// Python's `clone`, which Rust spells `clone_template`: a session gets the template's bindings,
/// its own cache and its own owned-resource lifetime.
#[tokio::test]
async fn dependencies_clone_template_gives_each_session_its_own_cache() {
    let template = Dependencies::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    template
        .bind_factory(
            "client",
            Arc::new(move || {
                let counted = counted.clone();
                Box::pin(async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok(Arc::new(AsyncClosable::default()) as Dependency)
                }) as BoxFut<SandboxResult<Dependency>>
            }),
            true,
            Some(async_closer()),
        )
        .expect("bind");

    let first = template.clone_template();
    let second = template.clone_template();
    let from_first = first.require("client", "test").await.expect("require");
    let _ = first.require("client", "test").await.expect("require");
    let from_second = second.require("client", "test").await.expect("require");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(!Arc::ptr_eq(&from_first, &from_second));

    // Closing one session leaves the other's resource alone.
    first.close().await;
    assert_eq!(
        from_first
            .downcast::<AsyncClosable>()
            .expect("AsyncClosable")
            .calls(),
        1
    );
    assert_eq!(
        from_second
            .clone()
            .downcast::<AsyncClosable>()
            .expect("AsyncClosable")
            .calls(),
        0
    );
    second.close().await;
    assert_eq!(
        from_second
            .downcast::<AsyncClosable>()
            .expect("AsyncClosable")
            .calls(),
        1
    );
}
