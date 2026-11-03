//! Python `tests/test_run_impl_resume_paths.py`:
//! `test_fresh_streamed_handoff_publishes_agent_update_before_session_append_failure` and
//! `test_fresh_streamed_handoff_drains_agent_update_event_for_slow_consumer`.
//! A streamed run that hands off and then fails to save to its session still delivers the
//! agent-updated event before the session error surfaces, even to a slow consumer.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::{Agent, RunOptions, Runner};

/// Accepts the first append and fails the second.
#[derive(Debug)]
struct FailSecondAppend {
    inner: InMemorySession,
    appends: AtomicUsize,
}

#[async_trait]
impl Session for FailSecondAppend {
    fn session_id(&self) -> &str {
        "fail-second-append"
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if self.appends.fetch_add(1, Ordering::SeqCst) == 1 {
            return Err(AgentsError::other("second append failed"));
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

async fn run(consumer_delay: Option<Duration>) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "transfer_to_delegate",
        "{}",
        "handoff-1",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(handoff(delegate));
    let session = Arc::new(FailSecondAppend {
        inner: InMemorySession::new("fail-second-append"),
        appends: AtomicUsize::new(0),
    });

    let mut streamed =
        Runner::run_streamed_with_options(&triage, "hello", RunOptions::new(()).session(session));
    let mut updated = Vec::new();
    let mut failure = None;
    {
        let stream = streamed.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let Some(delay) = consumer_delay {
                tokio::time::sleep(delay).await;
            }
            match event {
                Ok(StreamEvent::AgentUpdated { new_agent }) => updated.push(new_agent.name.clone()),
                Ok(_) => {}
                Err(error) => failure = Some(error),
            }
        }
    }

    let failure = failure.expect("the session failure surfaces");
    assert!(
        failure.to_string().contains("second append failed"),
        "{failure}"
    );
    assert!(updated.contains(&"delegate".to_string()), "{updated:?}");
}

#[tokio::test]
async fn the_agent_update_is_published_before_the_session_failure() {
    run(None).await;
}

#[tokio::test]
async fn a_slow_consumer_still_sees_the_agent_update_before_the_failure() {
    run(Some(Duration::from_millis(1))).await;
}
