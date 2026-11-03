//! Task-local tracking of the current trace and span (Python uses contextvars).

use std::future::Future;

use super::spans::{Span, Trace};

#[derive(Clone, Default)]
struct Scope {
    trace: Option<Trace>,
    span: Option<Span>,
}

tokio::task_local! {
    static CURRENT: Scope;
}

/// The current trace, if any.
pub fn current_trace() -> Option<Trace> {
    CURRENT.try_with(|s| s.trace.clone()).ok().flatten()
}

/// The current span, if any.
pub fn current_span() -> Option<Span> {
    CURRENT.try_with(|s| s.span.clone()).ok().flatten()
}

/// Run `fut` with `trace` as the current trace (and no current span).
pub async fn with_trace<F: Future>(trace: Trace, fut: F) -> F::Output {
    CURRENT
        .scope(
            Scope {
                trace: Some(trace),
                span: None,
            },
            fut,
        )
        .await
}

/// Run `fut` with `span` as the current span.
pub async fn with_span<F: Future>(span: Span, fut: F) -> F::Output {
    let trace = current_trace();
    CURRENT
        .scope(
            Scope {
                trace,
                span: Some(span),
            },
            fut,
        )
        .await
}

/// Capture the current scope so it can be re-entered from a spawned task.
#[derive(Clone, Default, Debug)]
pub struct ScopeSnapshot {
    pub trace: Option<Trace>,
    pub span: Option<Span>,
}

impl ScopeSnapshot {
    /// Snapshot of the current trace and span.
    pub fn capture() -> Self {
        ScopeSnapshot {
            trace: current_trace(),
            span: current_span(),
        }
    }

    /// Run `fut` inside the captured scope.
    pub async fn enter<F: Future>(self, fut: F) -> F::Output {
        CURRENT
            .scope(
                Scope {
                    trace: self.trace,
                    span: self.span,
                },
                fut,
            )
            .await
    }
}

impl std::fmt::Debug for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scope")
            .field("trace", &self.trace)
            .field("span", &self.span)
            .finish()
    }
}
