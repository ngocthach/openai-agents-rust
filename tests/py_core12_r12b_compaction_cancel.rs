//! Port of `test_run_compaction_restores_history_when_replacement_add_is_cancelled` from
//! `tests/memory/test_openai_responses_compaction_session.py`.
//!
//! Python cancels the task while the replacement is being added and the session puts the old
//! history back. Dropping a Rust future cannot run that restore, so the replacement runs on its
//! own task: a dropped caller must never leave the store empty or half replaced.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, Session,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::{Request, mock_openai_with};

/// A list store whose replacement add is held until released.
#[derive(Debug)]
struct HeldStore {
    items: Mutex<Vec<InputItem>>,
    hold_adding: Mutex<Option<&'static str>>,
    add_started: Notify,
    release: Notify,
}

#[async_trait]
impl Session for HeldStore {
    fn session_id(&self) -> &str {
        "held"
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let items = self.items.lock().unwrap().clone();
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let hold = {
            let needle = *self.hold_adding.lock().unwrap();
            needle.is_some_and(|n| items.iter().any(|i| i.0.to_string().contains(n)))
        };
        if hold {
            self.add_started.notify_one();
            self.release.notified().await;
        }
        self.items.lock().unwrap().extend(items);
        Ok(())
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }
    async fn clear_session(&self) -> Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

#[tokio::test]
async fn a_dropped_compaction_never_leaves_the_store_empty() {
    openai_agents_sdk::set_tracing_disabled(true);
    let original =
        InputItem::from_value(json!({"type": "message", "role": "user", "content": "original"}));
    let compacted = json!({"type": "message", "role": "assistant", "content": "compacted"});
    let store = Arc::new(HeldStore {
        items: Mutex::new(vec![original]),
        hold_adding: Mutex::new(Some("compacted")),
        add_started: Notify::new(),
        release: Notify::new(),
    });
    let reply = compacted.clone();
    let (base, _requests) = mock_openai_with(move |_req: &Request| {
        json!({"id": "cmp_1", "output": [reply.clone()], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})
    })
    .await;
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("held", store.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_compaction_mode(CompactionMode::Input),
    );

    let caller = {
        let session = session.clone();
        tokio::spawn(async move {
            let _ = session
                .run_compaction(OpenAIResponsesCompactionArgs {
                    force: true,
                    ..Default::default()
                })
                .await;
        })
    };
    tokio::time::timeout(Duration::from_secs(5), store.add_started.notified())
        .await
        .expect("the replacement add never started");
    // The history was cleared and the replacement is in flight: drop the caller.
    assert!(store.items.lock().unwrap().is_empty());
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    store.release.notify_one();

    let mut stored: Vec<Value> = Vec::new();
    for _ in 0..200 {
        stored = store
            .items
            .lock()
            .unwrap()
            .iter()
            .map(|i| i.0.clone())
            .collect();
        if !stored.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        stored,
        vec![compacted],
        "the replacement must still complete"
    );
}
