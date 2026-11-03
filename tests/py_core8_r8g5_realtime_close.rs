//! Ports of `tests/realtime/test_session.py` close-coordination cases the earlier ports left as
//! Python-only: concurrent `close()` callers share one model close and its failure, and
//! dropping one waiting caller does not abandon the cleanup the others wait for.
#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModel, RealtimeModelConfig, RealtimeModelEvents,
    RealtimeModelSendEvent, RealtimeSession,
};
use tokio::sync::Notify;

const WAIT: Duration = Duration::from_secs(5);

/// A transport whose `close` blocks until released and fails on its first call when asked to.
#[derive(Debug)]
struct GatedCloseModel {
    inner: FakeRealtimeModel,
    close_calls: AtomicUsize,
    fail_first: bool,
    started: Notify,
    release: Notify,
}

impl GatedCloseModel {
    fn new(fail_first: bool) -> Arc<Self> {
        Arc::new(GatedCloseModel {
            inner: FakeRealtimeModel::new(),
            close_calls: AtomicUsize::new(0),
            fail_first,
            started: Notify::new(),
            release: Notify::new(),
        })
    }
}

#[async_trait::async_trait]
impl RealtimeModel for GatedCloseModel {
    async fn connect(
        &self,
        config: RealtimeModelConfig,
    ) -> openai_agents_sdk::error::Result<RealtimeModelEvents> {
        self.inner.connect(config).await
    }
    async fn send_event(
        &self,
        event: RealtimeModelSendEvent,
    ) -> openai_agents_sdk::error::Result<()> {
        self.inner.send_event(event).await
    }
    async fn close(&self) -> openai_agents_sdk::error::Result<()> {
        let call = self.close_calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            self.started.notify_one();
            self.release.notified().await;
            if self.fail_first {
                return Err(AgentsError::other("close failed"));
            }
        }
        self.inner.close().await
    }
}

async fn connect(model: &Arc<GatedCloseModel>) -> Arc<RealtimeSession<()>> {
    Arc::new(
        RealtimeSession::connect(
            model.clone(),
            RealtimeAgent::<()>::new("agent").into_arc(),
            (),
            None,
            None,
        )
        .await
        .unwrap(),
    )
}

#[tokio::test]
async fn concurrent_close_callers_share_the_failure_and_the_next_close_retries() {
    let model = GatedCloseModel::new(true);
    let session = connect(&model).await;

    let first = tokio::spawn({
        let session = session.clone();
        async move { session.close().await }
    });
    tokio::time::timeout(WAIT, model.started.notified())
        .await
        .expect("the model close started");
    let second = tokio::spawn({
        let session = session.clone();
        async move { session.close().await }
    });
    tokio::task::yield_now().await;
    assert!(
        !second.is_finished(),
        "the second caller waits for the first"
    );

    model.release.notify_one();
    let first = tokio::time::timeout(WAIT, first).await.unwrap().unwrap();
    let second = tokio::time::timeout(WAIT, second).await.unwrap().unwrap();

    assert!(first.unwrap_err().to_string().contains("close failed"));
    assert!(second.unwrap_err().to_string().contains("close failed"));
    assert_eq!(model.close_calls.load(Ordering::SeqCst), 1);

    session.close().await.expect("the retry succeeds");
    assert_eq!(model.close_calls.load(Ordering::SeqCst), 2);
    assert!(session.is_closed());
}

#[tokio::test]
async fn dropping_one_close_waiter_does_not_cancel_the_cleanup() {
    let model = GatedCloseModel::new(false);
    let session = connect(&model).await;

    let surviving = tokio::spawn({
        let session = session.clone();
        async move { session.close().await }
    });
    tokio::time::timeout(WAIT, model.started.notified())
        .await
        .expect("the model close started");
    let cancelled = tokio::spawn({
        let session = session.clone();
        async move { session.close().await }
    });
    tokio::task::yield_now().await;
    assert!(!cancelled.is_finished());
    cancelled.abort();
    assert!(cancelled.await.unwrap_err().is_cancelled());

    model.release.notify_one();
    tokio::time::timeout(WAIT, surviving)
        .await
        .unwrap()
        .unwrap()
        .expect("the surviving waiter's close succeeds");

    assert_eq!(model.close_calls.load(Ordering::SeqCst), 1);
    assert!(session.is_closed());
}

async fn session_for(model: &Arc<FakeRealtimeModel>) -> RealtimeSession<()> {
    RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("agent").into_arc(),
        (),
        None,
        None,
    )
    .await
    .unwrap()
}

/// `test_exception_context_preservation`: each transport failure keeps its context and message
/// on the error event and in the stored failure, and ends the session.
#[tokio::test]
async fn an_exception_keeps_its_context_and_ends_the_session() {
    use openai_agents_sdk::realtime::{RealtimeModelEvent, RealtimeSessionEvent};

    for (context, message) in [
        ("Failed to send audio", "Audio encoding failed"),
        ("WebSocket error in message listener", "Network error"),
        ("Failed to send event: response.create", "Socket closed"),
    ] {
        let model = Arc::new(FakeRealtimeModel::new());
        let session = session_for(&model).await;
        assert!(model.emit(RealtimeModelEvent::Exception {
            message: message.to_string(),
            context: Some(context.to_string()),
        }));

        let mut error_event = None;
        tokio::time::timeout(WAIT, async {
            while let Some(event) = session.recv().await {
                if let RealtimeSessionEvent::Error { error, .. } = event {
                    error_event = Some(error);
                }
            }
        })
        .await
        .expect("the session stream ends");

        assert_eq!(
            error_event.expect("an error event")["message"],
            format!("{context}: {message}")
        );
        assert!(session.is_closed());
        let stored = session.take_error().expect("the failure is stored");
        assert!(stored.to_string().contains(message), "{stored}");
    }
}

/// `test_multiple_exception_handling_behavior`: several failures queued before the caller reads
/// still end the session with a stored failure.
#[tokio::test]
async fn several_exceptions_before_consumption_still_end_the_session() {
    use openai_agents_sdk::realtime::RealtimeModelEvent;

    let model = Arc::new(FakeRealtimeModel::new());
    let session = session_for(&model).await;
    for (message, context) in [
        ("First error", "First context"),
        ("Second error", "Second context"),
    ] {
        model.emit(RealtimeModelEvent::Exception {
            message: message.to_string(),
            context: Some(context.to_string()),
        });
    }

    tokio::time::timeout(WAIT, async { while session.recv().await.is_some() {} })
        .await
        .expect("the session stream ends");

    assert!(session.is_closed());
    assert!(session.take_error().is_some());
}
