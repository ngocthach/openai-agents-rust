//! Parity port of the history-restore and rollback tests of
//! `tests/memory/test_openai_responses_compaction_session.py`.
//!
//! The wrapped store is a scripted in-memory session whose `add_items` and `clear_session`
//! can fail after partially mutating, mirroring Python's `PartiallyFailingReplacementSession`
//! family. Tests that depend on asyncio task cancellation or pausing a SQLite commit on a
//! thread are Python-only (see the report).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, Session,
    SessionSettings,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::mock_openai;

#[derive(Debug, Clone, Copy, PartialEq)]
enum AddScript {
    /// Every call succeeds.
    Ok,
    /// The first call stores one item then fails; later calls succeed.
    FailFirstAfterOne,
    /// The first call stores one item then fails; every later call fails outright.
    FailFirstThenAlways,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ClearScript {
    Ok,
    /// Fails before touching the store.
    FailWithoutMutation,
    /// Clears the store, then fails.
    FailAfterMutation,
}

#[derive(Debug)]
struct ScriptedSession {
    items: Mutex<Vec<InputItem>>,
    add_calls: Mutex<usize>,
    clear_calls: Mutex<usize>,
    add: AddScript,
    clear: ClearScript,
    settings: Option<SessionSettings>,
}

impl ScriptedSession {
    fn new(history: Vec<InputItem>, add: AddScript, clear: ClearScript) -> Self {
        ScriptedSession {
            items: Mutex::new(history),
            add_calls: Mutex::new(0),
            clear_calls: Mutex::new(0),
            add,
            clear,
            settings: None,
        }
    }

    fn limited(mut self, limit: usize) -> Self {
        self.settings = Some(SessionSettings::with_limit(limit));
        self
    }

    fn items(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }

    fn add_calls(&self) -> usize {
        *self.add_calls.lock().unwrap()
    }

    fn clear_calls(&self) -> usize {
        *self.clear_calls.lock().unwrap()
    }
}

#[async_trait]
impl Session for ScriptedSession {
    fn session_id(&self) -> &str {
        "scripted"
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = limit.or_else(|| self.settings.as_ref().and_then(|s| s.limit));
        let items = self.items.lock().unwrap();
        Ok(match limit {
            None => items.clone(),
            Some(0) => Vec::new(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let call = {
            let mut calls = self.add_calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        match (self.add, call) {
            (AddScript::Ok, _) | (AddScript::FailFirstAfterOne, 2..) => {
                self.items.lock().unwrap().extend(items);
                Ok(())
            }
            (_, 1) => {
                self.items.lock().unwrap().extend(items.into_iter().take(1));
                Err(AgentsError::other("replacement failed"))
            }
            (AddScript::FailFirstThenAlways, _) => {
                Err(AgentsError::other("SECRET_COMPACTION_RESTORE_FAILURE"))
            }
            _ => unreachable!(),
        }
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> Result<()> {
        *self.clear_calls.lock().unwrap() += 1;
        match self.clear {
            ClearScript::Ok => {
                self.items.lock().unwrap().clear();
                Ok(())
            }
            ClearScript::FailWithoutMutation => Err(AgentsError::other("clear failed")),
            ClearScript::FailAfterMutation => {
                self.items.lock().unwrap().clear();
                Err(AgentsError::other("clear failed"))
            }
        }
    }
}

fn msg(role: &str, text: &str) -> InputItem {
    InputItem::from_value(json!({"type": "message", "role": role, "content": text}))
}

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })
}

async fn compacting(
    store: Arc<ScriptedSession>,
) -> (
    OpenAIResponsesCompactionSession,
    Arc<Mutex<Vec<py_mcpmem_support::Request>>>,
) {
    let compacted = json!({"type": "message", "role": "assistant", "content": "compacted"});
    let (base, captured) = mock_openai(vec![compact_reply(vec![compacted])]).await;
    let session = OpenAIResponsesCompactionSession::new("test", store)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input);
    (session, captured)
}

fn force() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Restoring history after a failed replacement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn history_is_restored_when_the_replacement_add_fails() {
    let history = vec![
        msg("user", "original"),
        InputItem::from_value(json!({
            "type": "function_call", "call_id": "call_123", "name": "lookup", "arguments": "{}",
        })),
    ];
    let store = Arc::new(ScriptedSession::new(
        history.clone(),
        AddScript::FailFirstAfterOne,
        ClearScript::Ok,
    ));
    let (session, _) = compacting(store.clone()).await;

    let err = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("replacement failed"), "{err}");
    assert_eq!(store.items(), history);
    assert_eq!(store.clear_calls(), 2);
    assert_eq!(store.add_calls(), 2);
}

#[tokio::test]
async fn the_full_history_is_restored_when_a_session_limit_applies() {
    for budget in [None, Some(3usize)] {
        let history = vec![
            msg("user", "oldest"),
            msg("assistant", "middle"),
            msg("user", "newest"),
        ];
        let store = Arc::new(
            ScriptedSession::new(
                history.clone(),
                AddScript::FailFirstAfterOne,
                ClearScript::Ok,
            )
            .limited(1),
        );
        let (session, _) = compacting(store.clone()).await;
        let session = match budget {
            Some(b) => session.with_max_rollback_items(b).unwrap(),
            None => session,
        };

        let err = session
            .run_compaction(force())
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("replacement failed"), "{budget:?}: {err}");
        assert_eq!(
            store.get_items(Some(10)).await.unwrap(),
            history,
            "{budget:?}"
        );
        assert_eq!(store.clear_calls(), 2, "{budget:?}");
        assert_eq!(store.add_calls(), 2, "{budget:?}");
    }
}

#[tokio::test]
async fn history_is_not_restored_when_clear_fails_without_mutating() {
    let history = vec![msg("user", "original")];
    let store = Arc::new(ScriptedSession::new(
        history.clone(),
        AddScript::Ok,
        ClearScript::FailWithoutMutation,
    ));
    let (session, _) = compacting(store.clone()).await;

    let err = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("clear failed"), "{err}");
    assert_eq!(store.items(), history);
    assert_eq!(store.clear_calls(), 1);
    assert_eq!(store.add_calls(), 0);
}

#[tokio::test]
async fn history_is_restored_when_clear_fails_after_mutating() {
    let history = vec![msg("user", "original")];
    let store = Arc::new(ScriptedSession::new(
        history.clone(),
        AddScript::Ok,
        ClearScript::FailAfterMutation,
    ));
    let (session, _) = compacting(store.clone()).await;

    let err = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("clear failed"), "{err}");
    assert_eq!(store.items(), history);
    assert_eq!(store.clear_calls(), 1);
    assert_eq!(store.add_calls(), 1);
}

#[tokio::test]
async fn the_replacement_error_wins_when_the_restore_also_fails() {
    let history = vec![msg("user", "original")];
    let store = Arc::new(ScriptedSession::new(
        history,
        AddScript::FailFirstThenAlways,
        ClearScript::Ok,
    ));
    let (session, _) = compacting(store.clone()).await;

    let err = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("replacement failed"), "{err}");
    assert!(!err.contains("SECRET_COMPACTION_RESTORE_FAILURE"), "{err}");
    assert_eq!(store.clear_calls(), 2);
    assert_eq!(store.add_calls(), 2);
}

// ---------------------------------------------------------------------------
// Rollback budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_failed_clear_inspection_is_bounded_by_the_rollback_budget() {
    for clear_mutates in [false, true] {
        let history = vec![msg("user", "old"), msg("assistant", "new")];
        let store = Arc::new(ScriptedSession::new(
            history.clone(),
            AddScript::Ok,
            if clear_mutates {
                ClearScript::FailAfterMutation
            } else {
                ClearScript::FailWithoutMutation
            },
        ));
        let (base, _) = mock_openai(vec![compact_reply(vec![])]).await;
        let session = OpenAIResponsesCompactionSession::new("test", store.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_max_rollback_items(2)
            .unwrap();

        let err = session
            .run_compaction(force())
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("clear failed"), "{clear_mutates}: {err}");
        assert_eq!(store.items(), history, "{clear_mutates}");
    }
}

#[tokio::test]
async fn an_oversized_history_fails_before_the_request_and_any_write() {
    let history: Vec<InputItem> = (0..64).map(|i| msg("user", &format!("{i}"))).collect();
    let store = Arc::new(ScriptedSession::new(
        history.clone(),
        AddScript::Ok,
        ClearScript::Ok,
    ));
    let (session, captured) = compacting(store.clone()).await;
    let session = session.with_max_rollback_items(3).unwrap();

    let err = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("exceeds max_rollback_items"), "{err}");
    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(store.clear_calls(), 0);
    assert_eq!(store.add_calls(), 0);
    assert_eq!(store.items(), history);
}

#[tokio::test]
async fn compaction_with_an_empty_history_still_requests_when_forced() {
    let store = Arc::new(ScriptedSession::new(
        Vec::new(),
        AddScript::Ok,
        ClearScript::Ok,
    ));
    let (session, captured) = compacting(store).await;

    session.run_compaction(force()).await.unwrap();

    assert_eq!(captured.lock().unwrap().len(), 1);
}
