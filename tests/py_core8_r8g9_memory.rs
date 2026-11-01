//! `test_settled_pop_failure_invalidates_response_chain` and
//! `test_settled_clear_failure_invalidates_response_chain` (`outcome="error"`) of
//! `tests/memory/test_openai_responses_compaction_session.py`.
//!
//! The wrapped store commits the deletion and then reports a failure to the caller. The
//! compaction wrapper cannot know whether the deletion landed, so it must drop the stored
//! response chain either way: the next forced compaction needs a fresh response id, and the
//! store stays usable.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "sqlite")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::InputItem;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, Session,
    SessionRunCompletion,
};
use serde_json::json;

mod py_core6_mcpmisc_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::*;
use py_mcpmem_support::ListSession;

/// Commits the mutation, then fails the acknowledgement (the commit reached the store, the
/// answer did not reach the caller).
#[derive(Debug)]
struct LostAckSession {
    inner: ListSession,
    armed: AtomicBool,
}

#[async_trait]
impl Session for LostAckSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let popped = self.inner.pop_item().await?;
        if self.armed.swap(false, Ordering::SeqCst) {
            return Err(AgentsError::other("commit acknowledgement failed"));
        }
        Ok(popped)
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await?;
        if self.armed.swap(false, Ordering::SeqCst) {
            return Err(AgentsError::other("commit acknowledgement failed"));
        }
        Ok(())
    }
}

fn force() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}

fn history() -> Vec<InputItem> {
    vec![
        InputItem::from_value(json!({"role": "user", "content": "keep me"})),
        InputItem::from_value(json!({"role": "assistant", "content": "remove me"})),
    ]
}

async fn seeded(
    endpoint: &Compaction,
    deferring: Arc<AtomicBool>,
) -> (Arc<OpenAIResponsesCompactionSession>, Arc<LostAckSession>) {
    let underlying = Arc::new(LostAckSession {
        inner: ListSession::with_history("settled", history()),
        armed: AtomicBool::new(false),
    });
    let flag = deferring.clone();
    let session = Arc::new(
        compaction_session(
            "settled",
            underlying.clone(),
            endpoint,
            CompactionMode::PreviousResponseId,
        )
        .with_should_trigger_compaction(move |_| flag.load(Ordering::SeqCst)),
    );
    // An unstored response id, then deferred work through the runner's completion hook.
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-old".into()),
            store: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();
    deferring.store(true, Ordering::SeqCst);
    session
        .on_run_completed(&SessionRunCompletion {
            response_id: Some("resp-old".into()),
            has_local_tool_outputs: true,
            ..Default::default()
        })
        .await
        .unwrap();
    (session, underlying)
}

// test_settled_pop_failure_invalidates_response_chain[error]
#[tokio::test]
async fn a_pop_that_committed_but_failed_invalidates_the_response_chain() {
    let endpoint = Compaction::replying(vec![]).await;
    let (session, underlying) = seeded(&endpoint, Arc::new(AtomicBool::new(false))).await;

    underlying.armed.store(true, Ordering::SeqCst);
    let error = session
        .pop_item()
        .await
        .expect_err("the acknowledgement fails");
    assert!(
        error.to_string().contains("commit acknowledgement failed"),
        "{error}"
    );

    // The deletion committed before the failure reached the caller.
    let kept = history()[..1].to_vec();
    assert_eq!(session.get_items(None).await.unwrap(), kept);
    let error = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires a response_id"), "{error}");
    assert!(endpoint.requests().is_empty());
    assert_eq!(underlying.get_items(None).await.unwrap(), kept);
}

// test_settled_clear_failure_invalidates_response_chain[error]
#[tokio::test]
async fn a_clear_that_committed_but_failed_invalidates_the_response_chain() {
    let endpoint = Compaction::replying(vec![]).await;
    let (session, underlying) = seeded(&endpoint, Arc::new(AtomicBool::new(false))).await;

    underlying.armed.store(true, Ordering::SeqCst);
    let error = session
        .clear_session()
        .await
        .expect_err("the acknowledgement fails");
    assert!(
        error.to_string().contains("commit acknowledgement failed"),
        "{error}"
    );

    assert!(session.get_items(None).await.unwrap().is_empty());
    let error = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires a response_id"), "{error}");
    assert!(endpoint.requests().is_empty());

    // The session stays usable: a subsequent write survives and compaction works again.
    let follow_up = InputItem::from_value(json!({"role": "user", "content": "after clear"}));
    session.add_items(vec![follow_up.clone()]).await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap(), vec![follow_up]);
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-new".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(endpoint.requests().len(), 1);
    assert_eq!(endpoint.requests()[0]["previous_response_id"], "resp-new");
}
