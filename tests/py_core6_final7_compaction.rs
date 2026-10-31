//! Ports of the last compaction cases of `tests/memory/test_openai_responses_compaction_session.py`:
//! `test_rollback_does_not_revive_items_expired_during_compaction` (4 parametrized cases),
//! `test_resumed_save_carries_generation_into_compaction`,
//! `test_runner_skips_compaction_after_interleaved_model_wait` and
//! `test_runner_skips_compaction_after_interleaved_persist`.
//!
//! Python tracks a mutation generation across a run's reads and appends and revokes a pending
//! replacement when another writer got in between. The port drives the same interleavings and
//! asserts the same observable outcome: no compaction request, every writer's items retained.
#![allow(clippy::type_complexity)]
#![cfg(feature = "sqlite")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::memory::{
    CompactionMode, CompactionSnapshot, OpenAIResponsesCompactionSession, Session,
    SessionRunCompletion, SessionSettings,
};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, InputItem, RunOptions, RunState, Runner};
use serde_json::json;
use tokio::sync::Notify;

mod py_core6_mcpmisc_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::{Compaction, compaction_session, force, values};
use py_mcpmem_support::ListSession;

fn scripted(text: &'static str, response_id: &'static str) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message(text)])
                .with_response_id(Some(response_id.into())),
        )
        .unwrap();
    model
}

fn stored_text(items: &[InputItem]) -> String {
    py_core6_mcpmisc_support::text(items)
}

/// Forwards every `Session` hook to `inner`, and around the first append whose batch mentions
/// `marker` it can fail the append before it writes (a lost append to resume from) or pause
/// after it returned (Python's `PausingCompactionSession._add_items_with_generation`).
#[derive(Debug)]
struct Gated {
    inner: Arc<OpenAIResponsesCompactionSession>,
    marker: &'static str,
    fail_once: AtomicBool,
    pause_once: AtomicBool,
    persisted: Notify,
    release: Notify,
}

impl Gated {
    fn new(inner: Arc<OpenAIResponsesCompactionSession>, marker: &'static str) -> Arc<Gated> {
        Arc::new(Gated {
            inner,
            marker,
            fail_once: AtomicBool::new(false),
            pause_once: AtomicBool::new(false),
            persisted: Notify::new(),
            release: Notify::new(),
        })
    }
}

#[async_trait::async_trait]
impl Session for Gated {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    fn manages_history_remotely(&self) -> bool {
        self.inner.manages_history_remotely()
    }
    fn ignores_item_ids_for_matching(&self) -> bool {
        self.inner.ignores_item_ids_for_matching()
    }
    fn wants_model_exchange(&self) -> bool {
        self.inner.wants_model_exchange()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let hit = stored_text(&items).contains(self.marker);
        if hit && self.fail_once.swap(false, Ordering::SeqCst) {
            return Err(AgentsError::other("append lost"));
        }
        self.inner.add_items(items).await?;
        if hit && self.pause_once.swap(false, Ordering::SeqCst) {
            self.persisted.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
    async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
        self.inner.on_run_completed(info).await
    }
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        self.inner.compaction_snapshot(limit).await
    }
}

/// A compaction session over a plain list store that only run A's response id may trigger.
fn only_for_run_a(
    id: &str,
    endpoint: &Compaction,
) -> (Arc<ListSession>, Arc<OpenAIResponsesCompactionSession>) {
    let underlying = Arc::new(ListSession::new(id));
    let compaction = Arc::new(
        compaction_session(
            id,
            underlying.clone(),
            endpoint,
            CompactionMode::PreviousResponseId,
        )
        .with_should_trigger_compaction(|ctx| ctx.response_id.as_deref() == Some("resp-a")),
    );
    (underlying, compaction)
}

async fn run_b(session: &Arc<dyn Session>) {
    let agent = Agent::<()>::new("worker-b").with_model_instance(scripted("run-B", "resp-b"));
    Runner::run_with_options(
        &agent,
        "input-B",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
}

fn assert_both_runs_kept(underlying: &ListSession) {
    let stored = stored_text(&underlying.items());
    assert!(stored.contains("run-A"), "{stored}");
    assert!(stored.contains("run-B"), "{stored}");
}

/// Control for the interleaving tests: without another writer the same run does compact.
#[tokio::test]
async fn uninterrupted_run_compacts_so_the_skips_are_meaningful() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let (_, compaction) = only_for_run_a("uninterrupted", &endpoint);
    let session: Arc<dyn Session> = compaction;
    let agent_a = Agent::<()>::new("worker-a").with_model_instance(scripted("run-A", "resp-a"));
    Runner::run_with_options(&agent_a, "input-A", RunOptions::new(()).session(session))
        .await
        .unwrap();
    assert_eq!(endpoint.requests().len(), 1);
}

#[tokio::test]
async fn runner_skips_compaction_after_interleaved_model_wait() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let (underlying, compaction) = only_for_run_a("interleaved-model-wait", &endpoint);
    let session: Arc<dyn Session> = compaction;

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = Arc::new(ScriptedModel::new());
    {
        let (entered, release) = (entered.clone(), release.clone());
        model
            .enqueue(ScriptedStep::respond_async(move |_call| {
                let (entered, release) = (entered.clone(), release.clone());
                async move {
                    entered.notify_one();
                    release.notified().await;
                    Ok(ScriptedStep::output(vec![assistant_message("run-A")])
                        .with_response_id(Some("resp-a".into())))
                }
            }))
            .unwrap();
    }
    let agent_a = Agent::<()>::new("worker-a").with_model_instance(model);
    let run_a = Runner::run_with_options(
        &agent_a,
        "input-A",
        RunOptions::new(()).session(session.clone()),
    );
    // Run B completes (and persists) while A waits for its model.
    let interleave = async {
        entered.notified().await;
        run_b(&session).await;
        release.notify_one();
    };
    let (a, ()) = tokio::join!(run_a, interleave);
    a.unwrap();

    assert!(endpoint.requests().is_empty());
    assert_both_runs_kept(&underlying);
}

#[tokio::test]
async fn runner_skips_compaction_after_interleaved_persist() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let (underlying, compaction) = only_for_run_a("interleaved-persist", &endpoint);
    let gated = Gated::new(compaction, "run-A");
    gated.pause_once.store(true, Ordering::SeqCst);
    let session: Arc<dyn Session> = gated.clone();

    let agent_a = Agent::<()>::new("worker-a").with_model_instance(scripted("run-A", "resp-a"));
    let run_a = Runner::run_with_options(
        &agent_a,
        "input-A",
        RunOptions::new(()).session(session.clone()),
    );
    // Run B appends between A's save and A's compaction.
    let interleave = async {
        gated.persisted.notified().await;
        run_b(&session).await;
        gated.release.notify_one();
    };
    let (a, ()) = tokio::join!(run_a, interleave);
    a.unwrap();

    assert!(endpoint.requests().is_empty());
    assert_both_runs_kept(&underlying);
}

#[tokio::test]
async fn resumed_save_carries_generation_into_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let (underlying, compaction) = only_for_run_a("resumed-interleaved-persist", &endpoint);
    let gated = Gated::new(compaction, "run-A");
    gated.fail_once.store(true, Ordering::SeqCst);
    let session: Arc<dyn Session> = gated.clone();

    // Run A hands off; the append of that turn is lost, so its checkpoint carries the write
    // to resume.
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![
                assistant_message("run-A"),
                function_call_with_id("transfer_to_delegate", "{}", "handoff-1"),
            ])
            .with_response_id(Some("resp-a".into())),
        )
        .unwrap();
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message("done")])
                .with_response_id(Some("resp-delegate".into())),
        )
        .unwrap();
    let delegate = Agent::<()>::new("delegate").with_model_instance(model.clone());
    let agent_a = Agent::<()>::new("worker-a")
        .with_model_instance(model)
        .with_handoff(handoff(delegate));
    let mut failed = Runner::run_streamed_with_options(
        &agent_a,
        "input-A",
        RunOptions::new(()).session(session.clone()),
    );
    {
        let events = failed.stream_events();
        futures::pin_mut!(events);
        let mut error = None;
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
                break;
            }
        }
        assert!(
            error
                .expect("the append fails")
                .to_string()
                .contains("append lost")
        );
    }
    let payload = failed.to_state().expect("checkpoint").to_json().unwrap();
    assert!(payload["pending_session_write"].is_object(), "{payload}");
    assert!(!stored_text(&underlying.items()).contains("run-A"));
    let state = RunState::from_json(payload).unwrap();

    // The resumed append lands, then run B appends before the resumed save is acknowledged.
    gated.pause_once.store(true, Ordering::SeqCst);
    let resumed = Runner::run_with_options(
        &agent_a,
        state,
        RunOptions::new(()).session(session.clone()),
    );
    let interleave = async {
        gated.persisted.notified().await;
        run_b(&session).await;
        gated.release.notify_one();
    };
    let (resumed, ()) = tokio::join!(resumed, interleave);
    resumed.unwrap();

    assert!(endpoint.requests().is_empty());
    assert_both_runs_kept(&underlying);
}

// ---------------------------------------------------------------------------------------------
// test_rollback_does_not_revive_items_expired_during_compaction (budget x failure stage)
// ---------------------------------------------------------------------------------------------

/// Python's `patch.object(encrypted, "clear_session" | "add_items", side_effect=fail_once)`:
/// counts the chosen mutation and would fail its first run after performing it.
#[cfg(feature = "encrypt")]
#[derive(Debug)]
struct FailOnce {
    inner: Arc<dyn Session>,
    stage: &'static str,
    calls: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "encrypt")]
impl FailOnce {
    async fn hit(&self, stage: &str) -> Result<()> {
        if stage == self.stage && self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(AgentsError::other("replacement failed"));
        }
        Ok(())
    }
}

#[cfg(feature = "encrypt")]
#[async_trait::async_trait]
impl Session for FailOnce {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.inner.add_items(items).await?;
        self.hit("add_items").await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await?;
        self.hit("clear_session").await
    }
}

#[cfg(feature = "encrypt")]
async fn rollback_does_not_revive_expired_items(budget: Option<usize>, stage: &'static str) {
    use std::time::Duration;
    const TTL: u64 = 3;
    let live = json!({"role": "user", "content": "still live"});
    // The request outlasts the first item's lifetime but not the second's: the first item
    // (4.05s old at the reply) is expired, the second (about 2.6s old) is live.
    let endpoint = Compaction::with(|_| {
        std::thread::sleep(Duration::from_millis(2550));
        json!({"id": "cmp", "output": [{"role": "assistant", "content": "summary"}],
               "usage": py_core6_mcpmisc_support::usage()})
    })
    .await;
    let encrypted = py_core6_mcpmisc_support::encrypted(
        "ttl-rollback",
        py_core6_mcpmisc_support::sqlite("ttl-rollback", None),
        TTL,
    );
    encrypted
        .add_items(vec![InputItem::from_value(
            json!({"role": "user", "content": "expires during request"}),
        )])
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    encrypted
        .add_items(vec![InputItem::from_value(live.clone())])
        .await
        .unwrap();

    let failing = Arc::new(FailOnce {
        inner: encrypted.clone(),
        stage,
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut session = compaction_session(
        "ttl-rollback",
        failing.clone(),
        &endpoint,
        CompactionMode::Input,
    );
    if let Some(budget) = budget {
        session = session.with_max_rollback_items(budget).unwrap();
    }
    // Both items are live when the request starts.
    assert_eq!(encrypted.get_items(None).await.unwrap().len(), 2);

    // Python replaces its pre-request snapshot and fails that replacement; this port re-reads
    // the store after the request, sees the expiry as a change and never attempts one.
    session.run_compaction(force()).await.unwrap();

    assert_eq!(endpoint.requests().len(), 1, "{budget:?} {stage}");
    assert_eq!(
        failing.calls.load(Ordering::SeqCst),
        0,
        "{budget:?} {stage}"
    );
    assert_eq!(
        values(&encrypted.get_items(None).await.unwrap()),
        vec![live],
        "{budget:?} {stage}"
    );
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn rollback_does_not_revive_expired_items_unbounded_clear() {
    rollback_does_not_revive_expired_items(None, "clear_session").await;
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn rollback_does_not_revive_expired_items_unbounded_add() {
    rollback_does_not_revive_expired_items(None, "add_items").await;
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn rollback_does_not_revive_expired_items_budget_clear() {
    rollback_does_not_revive_expired_items(Some(2), "clear_session").await;
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn rollback_does_not_revive_expired_items_budget_add() {
    rollback_does_not_revive_expired_items(Some(2), "add_items").await;
}
