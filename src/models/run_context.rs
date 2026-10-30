//! The run that currently owns model calls. Mirrors `agents.models._run_context`.
//!
//! The runner wraps every model call (and every model stream) of one run in
//! [`model_run_context`], so a model that keeps per-run state — the hosted multi-agent
//! WebSocket model keeps a paused response — can tell its own run's calls from another run's
//! and release that state in [`Model::cleanup_on_run_end`](crate::models::Model::cleanup_on_run_end).
//!
//! Python stores the owning object itself in a `ContextVar`; Rust stores a process-unique id in
//! a task-local, which is the same identity test without borrowing the runner's state.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::Stream;

/// Identifies the run that owns the model calls made inside a [`model_run_context`].
pub type ModelRunOwner = u64;

tokio::task_local! {
    static MODEL_RUN_OWNER: ModelRunOwner;
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh, process-unique run owner id.
pub fn new_model_run_owner() -> ModelRunOwner {
    NEXT_OWNER.fetch_add(1, Ordering::Relaxed)
}

/// The run owner of the current task, if any.
pub fn get_model_run_owner() -> Option<ModelRunOwner> {
    MODEL_RUN_OWNER.try_with(|o| *o).ok()
}

/// Run `fut` with `owner` as the current model run owner.
pub async fn model_run_context<F: Future>(owner: ModelRunOwner, fut: F) -> F::Output {
    MODEL_RUN_OWNER.scope(owner, fut).await
}

/// Run `f` with `owner` as the current model run owner.
pub fn model_run_context_sync<T>(owner: ModelRunOwner, f: impl FnOnce() -> T) -> T {
    MODEL_RUN_OWNER.sync_scope(owner, f)
}

/// A stream that is polled with `owner` as the current model run owner. Port of
/// `model_run_context_stream`.
#[derive(Debug)]
pub struct ModelRunContextStream<S> {
    owner: ModelRunOwner,
    inner: S,
}

impl<S> ModelRunContextStream<S> {
    /// Wrap `inner` so every poll sees `owner`.
    pub fn new(owner: ModelRunOwner, inner: S) -> Self {
        ModelRunContextStream { owner, inner }
    }
}

impl<S: Stream + Unpin> Stream for ModelRunContextStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let this = self.get_mut();
        let owner = this.owner;
        let inner = &mut this.inner;
        MODEL_RUN_OWNER.sync_scope(owner, || Pin::new(inner).poll_next(cx))
    }
}

/// Wrap a stream so every event is produced with `owner` as the current model run owner.
pub fn model_run_context_stream<S: Stream + Unpin>(
    owner: ModelRunOwner,
    stream: S,
) -> ModelRunContextStream<S> {
    ModelRunContextStream::new(owner, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn scopes_the_owner_to_the_future_and_the_stream() {
        assert_eq!(get_model_run_owner(), None);
        let owner = new_model_run_owner();
        assert_ne!(owner, new_model_run_owner());
        let seen = model_run_context(owner, async { get_model_run_owner() }).await;
        assert_eq!(seen, Some(owner));
        assert_eq!(get_model_run_owner(), None);

        let inner = futures::stream::iter(0..2).map(|_| get_model_run_owner());
        let seen: Vec<_> = model_run_context_stream(owner, Box::pin(inner))
            .collect()
            .await;
        assert_eq!(seen, vec![Some(owner), Some(owner)]);
    }
}
