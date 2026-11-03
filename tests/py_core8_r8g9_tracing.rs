//! Rust analogues of the generator-close cases of `tests/tracing/test_spans_impl.py` and
//! `tests/tracing/test_traces_impl.py`.
//!
//! Python abandons an async generator that is suspended inside `with span:` and closes it from the
//! same task or from another one. Rust scopes the current span and trace to the future that
//! `run` returns: abandoning that future half-way, on any task, must leave the caller's span and
//! trace untouched and must not panic. Both enabled tracing (`SpanImpl` / `TraceImpl`) and
//! disabled tracing (`NoOpSpan` / `NoOpTrace`) are covered, like the Python factories.

#![allow(clippy::await_holding_lock)] // the lock serializes process-global tracing state for a whole test

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use openai_agents_sdk::tracing::{
    DefaultTraceProvider, TraceProvider, agent_span, current_span, current_trace,
    reset_trace_provider, set_trace_provider, trace,
};

fn lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn install_provider(disabled: bool) {
    let provider = Arc::new(DefaultTraceProvider::new());
    provider.set_disabled(disabled);
    provider.set_processors(Vec::new());
    set_trace_provider(provider);
}

type Suspended = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A scoped future that has run until it suspended inside its scope, like a generator that
/// yielded from inside `with span:`.
async fn suspended_inside_span() -> Suspended {
    let span = agent_span("generator-exit", None, None, None);
    let mut scoped: Suspended = Box::pin(span.run(async {
        tokio::task::yield_now().await;
    }));
    assert!(futures::poll!(scoped.as_mut()).is_pending());
    scoped
}

async fn suspended_inside_trace() -> Suspended {
    let scoped_trace = trace("generator-exit")
        .trace_id("trace-generator-exit")
        .build();
    let mut scoped: Suspended = Box::pin(scoped_trace.run(async {
        tokio::task::yield_now().await;
    }));
    assert!(futures::poll!(scoped.as_mut()).is_pending());
    scoped
}

// test_generator_close_in_the_same_task_releases_the_span_scope[NoOpSpan, SpanImpl]
#[tokio::test]
async fn abandoning_a_span_scope_in_the_same_task_restores_the_callers_span() {
    let _guard = lock();
    for disabled in [true, false] {
        install_provider(disabled);
        trace("w")
            .build()
            .run(async move {
                assert!(current_span().is_none());
                let suspended = suspended_inside_span().await;
                drop(suspended);
                assert!(current_span().is_none(), "disabled={disabled}");
            })
            .await;
        reset_trace_provider();
    }
}

// test_generator_close_from_another_task_does_not_raise[NoOpSpan, SpanImpl]
#[tokio::test]
async fn abandoning_a_span_scope_from_another_task_does_not_fail() {
    let _guard = lock();
    for disabled in [true, false] {
        install_provider(disabled);
        trace("w")
            .build()
            .run(async move {
                let suspended = suspended_inside_span().await;
                tokio::spawn(async move { drop(suspended) })
                    .await
                    .expect("dropping from another task must not panic");
                assert!(current_span().is_none(), "disabled={disabled}");
            })
            .await;
        reset_trace_provider();
    }
}

// test_generator_close_in_the_same_task_releases_the_trace_scope[NoOpTrace, TraceImpl]
#[tokio::test]
async fn abandoning_a_trace_scope_in_the_same_task_restores_the_callers_trace() {
    let _guard = lock();
    for disabled in [true, false] {
        install_provider(disabled);
        assert!(current_trace().is_none());
        let suspended = suspended_inside_trace().await;
        drop(suspended);
        assert!(current_trace().is_none(), "disabled={disabled}");
        reset_trace_provider();
    }
}

// test_generator_close_from_another_task_does_not_raise[NoOpTrace, TraceImpl]
#[tokio::test]
async fn abandoning_a_trace_scope_from_another_task_does_not_fail() {
    let _guard = lock();
    for disabled in [true, false] {
        install_provider(disabled);
        let suspended = suspended_inside_trace().await;
        tokio::spawn(async move { drop(suspended) })
            .await
            .expect("dropping from another task must not panic");
        assert!(current_trace().is_none(), "disabled={disabled}");
        reset_trace_provider();
    }
}

mod shutdown {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use openai_agents_sdk::tracing::provider::DEFAULT_SHUTDOWN_TIMEOUT;
    use openai_agents_sdk::tracing::{
        Span, Trace, TraceProvider, TracingProcessor, reset_trace_provider, set_trace_provider,
        shutdown_guard,
    };

    /// A provider that only counts shutdowns, like Python's `_DummyProvider`.
    #[derive(Debug, Default)]
    struct CountingProvider {
        shutdowns: AtomicUsize,
        timeouts: Mutex<Vec<Duration>>,
        records_timeout: bool,
    }

    impl TraceProvider for CountingProvider {
        fn is_disabled(&self) -> bool {
            true
        }
        fn set_disabled(&self, _disabled: bool) {}
        fn set_processors(&self, _processors: Vec<Arc<dyn TracingProcessor>>) {}
        fn add_processor(&self, _processor: Arc<dyn TracingProcessor>) {}
        fn on_trace_start(&self, _trace: &Trace) {}
        fn on_trace_end(&self, _trace: &Trace) {}
        fn on_span_start(&self, _span: &Span) {}
        fn on_span_end(&self, _span: &Span) {}
        fn shutdown(&self) {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
        }
        fn shutdown_with_timeout(&self, timeout: Duration) {
            if self.records_timeout {
                self.timeouts.lock().unwrap().push(timeout);
            } else {
                self.shutdown();
            }
        }
    }

    // test_shutdown_global_trace_provider_calls_shutdown
    #[test]
    fn the_shutdown_guard_shuts_the_installed_provider_down() {
        let _guard = super::lock();
        let provider = Arc::new(CountingProvider::default());
        set_trace_provider(provider.clone());
        drop(shutdown_guard());
        reset_trace_provider();
        assert_eq!(provider.shutdowns.load(Ordering::SeqCst), 1);
    }

    // test_shutdown_global_trace_provider_passes_timeout_to_default_provider
    #[test]
    fn the_shutdown_guard_passes_the_default_timeout() {
        let _guard = super::lock();
        let provider = Arc::new(CountingProvider {
            records_timeout: true,
            ..Default::default()
        });
        set_trace_provider(provider.clone());
        drop(shutdown_guard());
        reset_trace_provider();
        assert_eq!(
            *provider.timeouts.lock().unwrap(),
            vec![DEFAULT_SHUTDOWN_TIMEOUT]
        );
        assert_eq!(DEFAULT_SHUTDOWN_TIMEOUT, Duration::from_secs(5));
    }
}
