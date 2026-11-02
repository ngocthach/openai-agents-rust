//! Ports of the cancelled-restore cases of `tests/memory/test_openai_responses_compaction_session.py`.
//!
//! Python cancels the task running `run_compaction` while the restore of a failed replacement
//! is in flight and checks that the restore still finishes and that newer writes through the
//! compaction session wait for it. Rust cancels by dropping (aborting) the caller; the restore
//! runs detached, and must keep the session's mutation lock until it has settled.
//!
//! Python's first replacement `add_items` raises `CancelledError` or `RuntimeError`; here it
//! returns an error, which takes the same restore path.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, Session,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::json;
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::mock_openai;

/// A gate one call waits on: `entered` fires when the call reaches it, `release` lets it go.
#[derive(Debug, Default)]
struct Gate {
    entered: Notify,
    release: Notify,
}

/// An in-memory store whose `fail_add`-th `add_items` fails, and whose `gate_add`-th
/// `add_items` / `gate_clear`-th `clear_session` waits on a gate before mutating.
#[derive(Debug, Default)]
struct GatedSession {
    items: Mutex<Vec<InputItem>>,
    add_calls: AtomicUsize,
    clear_calls: AtomicUsize,
    fail_add: Option<usize>,
    gate_add: Option<usize>,
    gate_clear: Option<usize>,
    gate: Gate,
}

impl GatedSession {
    fn items(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

#[async_trait]
impl Session for GatedSession {
    fn session_id(&self) -> &str {
        "lock-test"
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let items = self.items();
        Ok(match limit {
            None => items,
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let call = self.add_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_add == Some(call) {
            return Err(AgentsError::other("replacement failed"));
        }
        if self.gate_add == Some(call) {
            self.gate.entered.notify_one();
            self.gate.release.notified().await;
        }
        self.items.lock().unwrap().extend(items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> Result<()> {
        let call = self.clear_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.gate_clear == Some(call) {
            self.gate.entered.notify_one();
            self.gate.release.notified().await;
        }
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

fn msg(role: &str, text: &str) -> InputItem {
    InputItem::from_value(json!({"type": "message", "role": role, "content": text}))
}

fn history() -> Vec<InputItem> {
    vec![msg("user", "original"), msg("assistant", "reply")]
}

fn force() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}

async fn compacting(store: Arc<GatedSession>) -> Arc<OpenAIResponsesCompactionSession> {
    let compacted = json!({"type": "message", "role": "assistant", "content": "compacted"});
    let reply = json!({
        "id": "cmp_1",
        "output": [compacted],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    });
    let (base, _) = mock_openai(vec![reply]).await;
    Arc::new(
        OpenAIResponsesCompactionSession::new("lock-test", store)
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_compaction_mode(CompactionMode::Input),
    )
}

/// Poll until `store` holds `expected`, failing after a few seconds.
async fn wait_for_items(store: &GatedSession, expected: &[InputItem]) {
    for _ in 0..400 {
        if store.items() == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(store.items(), expected);
}

/// `test_run_compaction_restores_history_when_cancelled_again_during_restore`: a caller that
/// goes away while the restore rewrite is blocked does not abandon it; the history comes back
/// once the rewrite is released.
#[tokio::test(flavor = "multi_thread")]
async fn a_restore_finishes_after_its_caller_is_dropped() {
    openai_agents_sdk::set_tracing_disabled(true);
    let store = Arc::new(GatedSession {
        items: Mutex::new(history()),
        fail_add: Some(1),
        gate_add: Some(2),
        ..Default::default()
    });
    let session = compacting(store.clone()).await;

    let task = {
        let session = session.clone();
        tokio::spawn(async move { session.run_compaction(force()).await })
    };
    store.gate.entered.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(store.items().is_empty(), "the restore is still blocked");

    store.gate.release.notify_one();
    wait_for_items(&store, &history()).await;
    assert_eq!(store.clear_calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.add_calls.load(Ordering::SeqCst), 2);
}

/// `test_cancel_restore_waits_for_mutation_lock_before_newer_writes`: a write through the
/// compaction session issued after the caller went away waits for the detached restore, so it
/// is neither wiped by the restore's clear nor reordered before the restored history.
#[tokio::test(flavor = "multi_thread")]
async fn a_newer_write_waits_for_the_restore_of_a_dropped_caller() {
    openai_agents_sdk::set_tracing_disabled(true);
    let store = Arc::new(GatedSession {
        items: Mutex::new(history()),
        fail_add: Some(1),
        gate_clear: Some(2),
        ..Default::default()
    });
    let session = compacting(store.clone()).await;

    let task = {
        let session = session.clone();
        tokio::spawn(async move { session.run_compaction(force()).await })
    };
    store.gate.entered.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    let newer = msg("user", "newer-after-cancel");
    let newer_write = {
        let session = session.clone();
        let newer = newer.clone();
        tokio::spawn(async move { session.add_items(vec![newer]).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !newer_write.is_finished(),
        "the newer write must wait for the restore"
    );
    assert_eq!(store.clear_calls.load(Ordering::SeqCst), 2);

    store.gate.release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), newer_write)
        .await
        .expect("the newer write finishes once the restore settles")
        .unwrap()
        .unwrap();

    let mut expected = history();
    expected.push(newer);
    assert_eq!(store.items(), expected);
    assert_eq!(session.get_items(None).await.unwrap(), expected);
    assert_eq!(store.clear_calls.load(Ordering::SeqCst), 2);
    // 1) failed replacement add, 2) restore rewrite, 3) newer write.
    assert_eq!(store.add_calls.load(Ordering::SeqCst), 3);
}

/// `test_exception_restore_drains_when_compaction_is_cancelled`: the restore after a failed
/// replacement drains when the caller goes away, and the session's mutation lock is free
/// afterwards.
#[tokio::test(flavor = "multi_thread")]
async fn an_exception_restore_drains_and_releases_the_lock() {
    openai_agents_sdk::set_tracing_disabled(true);
    let store = Arc::new(GatedSession {
        items: Mutex::new(history()),
        fail_add: Some(1),
        gate_add: Some(2),
        ..Default::default()
    });
    let session = compacting(store.clone()).await;

    let task = {
        let session = session.clone();
        tokio::spawn(async move { session.run_compaction(force()).await })
    };
    store.gate.entered.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(store.items().is_empty());

    store.gate.release.notify_one();
    wait_for_items(&store, &history()).await;
    assert_eq!(session.get_items(None).await.unwrap(), history());
    assert_eq!(store.clear_calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.add_calls.load(Ordering::SeqCst), 2);

    // The lock is free: a mutation through the session completes promptly.
    tokio::time::timeout(Duration::from_secs(5), session.pop_item())
        .await
        .expect("the mutation lock was released")
        .unwrap();
}
