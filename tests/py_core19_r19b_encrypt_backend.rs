//! Ports of the custom-backend cases of `tests/extensions/memory/test_encrypt_session.py`:
//! `test_encrypted_pop_delegates_to_overridden_sqlite_pop` and
//! `test_encrypted_session_preserves_optional_compaction_delegation`.
//!
//! Python subclasses `SQLiteSession` and overrides `pop_item` or adds `run_compaction`; Rust
//! wraps a `SqliteSession` in a `Session` that overrides the same hook (`pop_item`, or
//! `on_run_completed`, which is how the runner reaches a session's compaction).
//!
//! Python's runner calls `run_compaction` only for a turn without local tool outputs (a tool
//! turn goes to the optional `_defer_compaction` instead). Rust has one hook: the runner calls
//! `on_run_completed` for every persisted turn with `has_local_tool_outputs` set, and the
//! session defers. The port asserts the same split through that flag.
#![cfg(all(feature = "sqlite", feature = "encrypt"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{
    EncryptedSession, Session, SessionRunCompletion, SessionSettings, SqliteSession,
};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, InputItem, RunOptions, Runner};
use serde_json::json;

const KEY: &str = "0123456789abcdef0123456789abcdef";

/// A SQLite store whose `pop_item` is overridden to count its calls; every other hook keeps
/// the trait's default or forwards to the store.
#[derive(Debug)]
struct CustomPopSession {
    inner: SqliteSession,
    pop_calls: AtomicUsize,
}

#[async_trait::async_trait]
impl Session for CustomPopSession {
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
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.pop_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

#[tokio::test]
async fn an_encrypted_pop_goes_through_the_backends_overridden_pop() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Arc::new(CustomPopSession {
        inner: SqliteSession::open("test_session", dir.path().join("history.db")).unwrap(),
        pop_calls: AtomicUsize::new(0),
    });
    let session =
        EncryptedSession::new("test_session", backend.clone(), "correct-key", 600).unwrap();
    let saved = InputItem::from_value(json!({"role": "user", "content": "saved"}));

    session.add_items(vec![saved.clone()]).await.unwrap();

    assert_eq!(session.pop_item().await.unwrap(), Some(saved));
    assert_eq!(backend.pop_calls.load(Ordering::SeqCst), 1);
    assert!(session.get_items(None).await.unwrap().is_empty());
}

/// A SQLite store with its own compaction hook, recording the response id and the tool-output
/// flag of every completion the runner hands it.
#[derive(Debug)]
struct CustomCompactionSession {
    inner: SqliteSession,
    calls: Mutex<Vec<(Option<String>, bool)>>,
}

#[async_trait::async_trait]
impl Session for CustomCompactionSession {
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
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
    async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
        self.calls
            .lock()
            .unwrap()
            .push((info.response_id.clone(), info.has_local_tool_outputs));
        Ok(Usage::default())
    }
}

#[tokio::test]
async fn an_encrypted_session_forwards_the_backends_own_compaction_hook() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Arc::new(CustomCompactionSession {
        inner: SqliteSession::open("encrypted-custom-compaction", dir.path().join("history.db"))
            .unwrap(),
        calls: Mutex::new(Vec::new()),
    });
    let session: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("encrypted-custom-compaction", backend.clone(), KEY, 600).unwrap(),
    );

    let lookup = FunctionTool::<()>::new_raw(
        "lookup",
        "look something up",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move { Ok("private lookup result".to_string()) },
    );
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![function_call_with_id("lookup", "{}", "lookup-custom")])
                .with_response_id(Some("resp-tool".into())),
        )
        .unwrap();
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message("private answer")])
                .with_response_id(Some("resp-final".into())),
        )
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(lookup);

    let result = Runner::run_with_options(
        &agent,
        "private input",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("private answer")
    );
    let calls = backend.calls.lock().unwrap().clone();
    let compactions: Vec<_> = calls
        .iter()
        .filter(|(_, has_local_tool_outputs)| !has_local_tool_outputs)
        .map(|(response_id, _)| response_id.clone())
        .collect();
    assert_eq!(compactions, [Some("resp-final".to_string())], "{calls:?}");
    assert_eq!(
        calls.first(),
        Some(&(Some("resp-tool".to_string()), true)),
        "the tool turn reaches the backend flagged for deferral"
    );
    // Input, tool call, tool output and answer are stored, and every row is ciphertext.
    let raw = backend.inner.get_items(None).await.unwrap();
    assert_eq!(raw.len(), 4);
    assert!(
        raw.iter()
            .all(|item| !item.as_value().to_string().contains("private")),
        "{raw:?}"
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
}
