//! Session memory: conversation history persisted across runs. Mirrors `agents.memory`.

#[cfg(feature = "sqlite")]
mod advanced_sqlite_session;
mod dapr_session;
#[cfg(feature = "encrypt")]
mod encrypted_session;
#[cfg(feature = "encrypt")]
mod fernet;
#[cfg(feature = "mongodb")]
mod mongodb_session;
mod openai_conversations_session;
mod openai_responses_compaction_session;
#[cfg(feature = "redis")]
mod redis_session;
#[cfg(feature = "sqlite")]
mod sqlite_session;
#[cfg(feature = "sqlx")]
mod sqlx_session;

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::Result;
use crate::items::InputItem;

#[cfg(feature = "sqlite")]
pub use advanced_sqlite_session::{
    AdvancedSqliteSession, BranchInfo, ConversationTurn, MAIN_BRANCH, SessionUsage, ToolUsage,
    TurnMessage, TurnUsage,
};
pub use dapr_session::{
    DEFAULT_DAPR_HTTP_ENDPOINT, DaprConsistency, DaprSession, dapr_session_keys,
};
#[cfg(feature = "encrypt")]
pub use encrypted_session::EncryptedSession;
#[cfg(feature = "mongodb")]
pub use mongodb_session::{
    DEFAULT_MESSAGES_COLLECTION, DEFAULT_SESSIONS_COLLECTION, MongoDBSession, generation_filter,
};
pub use openai_conversations_session::OpenAIConversationsSession;
pub use openai_responses_compaction_session::{
    CompactionContext, CompactionMode, DEFAULT_COMPACTION_THRESHOLD, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, ResolvedCompactionMode, ShouldTriggerCompaction,
    default_should_trigger_compaction, is_openai_model_name, select_compaction_candidate_items,
};
#[cfg(feature = "redis")]
pub use redis_session::{DEFAULT_KEY_PREFIX, RedisSession, redis_session_keys};
#[cfg(feature = "sqlite")]
pub use sqlite_session::SqliteSession;
#[cfg(feature = "sqlx")]
pub use sqlx_session::{DEFAULT_MESSAGES_TABLE, DEFAULT_SESSIONS_TABLE, SqlxDialect, SqlxSession};

/// Return `explicit`, or fall back to the limit in `settings`.
///
/// Mirrors `agents.memory.session_settings.resolve_session_limit`. Backends call it at the top
/// of `get_items`, so a session configured with a limit honors it even when a caller asks for
/// "everything". `Some(0)` is preserved and means no items.
pub fn resolve_session_limit(
    explicit: Option<usize>,
    settings: Option<&SessionSettings>,
) -> Option<usize> {
    explicit.or_else(|| settings.and_then(|s| s.limit))
}

/// Combine history with the new input of a turn.
///
/// Mirrors `agents.memory.util.SessionInputCallback`: given the session's stored items and the
/// turn's new items, return the list the agent actually runs on. This is the nameable form of
/// [`crate::RunConfig::session_input_callback`].
pub type SessionInputCallback = std::sync::Arc<
    dyn Fn(
            Vec<InputItem>,
            Vec<InputItem>,
        ) -> futures::future::BoxFuture<'static, Result<Vec<InputItem>>>
        + Send
        + Sync,
>;

/// Settings controlling how history is retrieved.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSettings {
    /// Maximum number of most recent items to retrieve (`None` = all).
    pub limit: Option<usize>,
}

impl SessionSettings {
    /// Settings retrieving at most `limit` items.
    pub fn with_limit(limit: usize) -> Self {
        SessionSettings { limit: Some(limit) }
    }

    /// Overlay `Some` values from `other`.
    pub fn resolve(&self, other: Option<&SessionSettings>) -> SessionSettings {
        match other {
            Some(o) if o.limit.is_some() => o.clone(),
            _ => self.clone(),
        }
    }
}

/// Canonical form of an item, used to match stored history against the last model exchange.
///
/// Two items with the same digest are the same conversation item for compaction purposes.
/// Cosmetic differences that storage and the Responses API introduce are normalized away:
/// SDK-only metadata is dropped, an assistant message's plain-string content is expanded to
/// the `output_text` content list the API returns, and a `completed` or absent `status` is
/// removed. The digest is the item's canonical JSON rather than a hash, so it never collides.
pub fn compaction_item_digest(item: &InputItem) -> String {
    let mut value = item.as_value().clone();
    if let Value::Object(map) = &mut value {
        // Written by the Conversations API, never part of a replayed item.
        map.remove("created_by");
        if map.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(Value::String(text)) = map.get("content") {
                let text = text.clone();
                map.insert(
                    "content".into(),
                    json!([{"type": "output_text", "text": text}]),
                );
            }
            let drop_status = match map.get("status") {
                None => true,
                Some(Value::String(status)) => status == "completed",
                Some(_) => false,
            };
            if drop_status {
                map.remove("status");
            }
        }
    }
    value.to_string()
}

/// Information passed to [`Session::on_run_completed`] after the runner persisted a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionRunCompletion {
    /// Id of the last model response (Responses API), if any.
    pub response_id: Option<String>,
    /// The `store` model setting used for the last response.
    pub store: Option<bool>,
    /// Whether the last turn produced local tool outputs.
    pub has_local_tool_outputs: bool,
    /// [`compaction_item_digest`] of every item of the last model exchange, in order:
    /// everything the model was sent followed by everything it produced.
    ///
    /// Sessions that compact history match the stored tail against this, so only history the
    /// model actually just saw is eligible for replacement. The runner fills this only for
    /// sessions whose [`Session::wants_model_exchange`] is `true`; left empty, compaction falls
    /// back to replacing the whole stored history, which is what manual `run_compaction` calls
    /// and pre-existing callers already did.
    pub model_exchange: Vec<String>,
}

/// An opaque, store-specific identifier for one item inside a [`CompactionSnapshot`].
///
/// Only the store that produced a handle interprets it. Callers slice and pass handles back
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotHandle(pub String);

/// A bounded window of the end of a session's stored history, with handles that let the store
/// replace a suffix of it atomically.
///
/// Produced by [`Session::compaction_snapshot`] and consumed by
/// [`Session::replace_compaction_suffix`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompactionSnapshot {
    /// The window's items, oldest first.
    pub items: Vec<InputItem>,
    /// One handle per entry of `items`, in the same order.
    pub handles: Vec<SnapshotHandle>,
    /// True when the window covers the whole stored history rather than just its tail.
    pub complete: bool,
}

/// Stores conversation history for a session.
///
/// The runner retrieves history before each run and stores new items after it.
#[async_trait]
pub trait Session: Send + Sync + fmt::Debug {
    /// The session id.
    fn session_id(&self) -> &str;

    /// Default settings for this session.
    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    /// Whether the store keeps the conversation on the model provider's servers rather than in
    /// items this SDK owns.
    ///
    /// Such a session cannot be wrapped by [`OpenAIResponsesCompactionSession`], which rewrites
    /// stored history. `OpenAIConversationsSession` is the one in-tree session that returns
    /// `true`.
    fn manages_history_remotely(&self) -> bool {
        false
    }

    /// Retrieve history, the most recent `limit` items when set (chronological order).
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>>;

    /// Append items.
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()>;

    /// Remove and return the most recent item.
    async fn pop_item(&self) -> Result<Option<InputItem>>;

    /// Remove all items.
    async fn clear_session(&self) -> Result<()>;

    /// Called after a run finished and its items were persisted. Sessions that compact history
    /// (for example `OpenAIResponsesCompactionSession`) override this and return the usage of
    /// any model requests they made, which the runner adds to the run usage.
    async fn on_run_completed(&self, _info: &SessionRunCompletion) -> Result<crate::usage::Usage> {
        Ok(crate::usage::Usage::default())
    }

    /// Whether the runner should record the last model exchange in
    /// [`SessionRunCompletion::model_exchange`].
    ///
    /// Digesting the exchange costs one serialization per item of the run's history, so only
    /// sessions that actually compact ask for it. `OpenAIResponsesCompactionSession` returns
    /// `true`; every other session leaves this at `false` and pays nothing.
    fn wants_model_exchange(&self) -> bool {
        false
    }

    /// Read up to the last `limit` items together with handles identifying them.
    ///
    /// Stores that can replace a bounded suffix of their history atomically override this;
    /// `OpenAIResponsesCompactionSession` then replaces only the items the last model exchange
    /// produced and keeps older history. Returning `None`, the default, makes compaction fall
    /// back to replacing the whole stored history.
    ///
    /// `limit` bounds the window, so `complete` reports whether it reached the start of the
    /// history. Implementations must return exactly one handle per item, in the same order.
    async fn compaction_snapshot(&self, _limit: usize) -> Result<Option<CompactionSnapshot>> {
        Ok(None)
    }

    /// Atomically replace exactly the items named by `handles` with `items`.
    ///
    /// `handles` is a suffix of the handles of a [`Session::compaction_snapshot`], in order.
    /// Returns `false` without changing anything when the stored history no longer ends with
    /// those exact items, which is how a concurrent write is detected. The default
    /// implementation performs no replacement and returns `false`.
    async fn replace_compaction_suffix(
        &self,
        _handles: &[SnapshotHandle],
        _items: Vec<InputItem>,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// A session stored in process memory.
///
/// Items carry a monotonic id so compaction can name and replace a suffix of history
/// atomically; ids are never reused, so a replaced suffix cannot be confused with a new one.
#[derive(Debug, Default)]
pub struct InMemorySession {
    session_id: String,
    settings: Option<SessionSettings>,
    items: Mutex<Vec<(u64, InputItem)>>,
    next_id: AtomicU64,
}

impl InMemorySession {
    /// A new empty in-memory session.
    pub fn new(session_id: impl Into<String>) -> Self {
        InMemorySession {
            session_id: session_id.into(),
            settings: None,
            items: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }
}

#[async_trait]
impl Session for InMemorySession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = resolve_session_limit(limit, self.settings.as_ref());
        let items = self.items.lock().expect("session poisoned");
        let window = match limit {
            None => &items[..],
            Some(0) => &[][..],
            Some(n) => &items[items.len().saturating_sub(n)..],
        };
        Ok(window.iter().map(|(_, item)| item.clone()).collect())
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let mut stored = self.items.lock().expect("session poisoned");
        for item in items {
            stored.push((self.next_id.fetch_add(1, Ordering::Relaxed), item));
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self
            .items
            .lock()
            .expect("session poisoned")
            .pop()
            .map(|(_, item)| item))
    }

    async fn clear_session(&self) -> Result<()> {
        self.items.lock().expect("session poisoned").clear();
        Ok(())
    }

    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        let stored = self.items.lock().expect("session poisoned");
        let start = stored.len().saturating_sub(limit);
        Ok(Some(CompactionSnapshot {
            items: stored[start..].iter().map(|(_, i)| i.clone()).collect(),
            handles: stored[start..]
                .iter()
                .map(|(id, _)| SnapshotHandle(id.to_string()))
                .collect(),
            complete: start == 0,
        }))
    }

    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        let mut stored = self.items.lock().expect("session poisoned");
        if stored.len() < handles.len() {
            return Ok(false);
        }
        let start = stored.len() - handles.len();
        if stored[start..]
            .iter()
            .zip(handles)
            .any(|((id, _), handle)| id.to_string() != handle.0)
        {
            return Ok(false);
        }
        stored.truncate(start);
        for item in items {
            stored.push((self.next_id.fetch_add(1, Ordering::Relaxed), item));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_session_operations() {
        let s = InMemorySession::new("s1");
        s.add_items(vec![
            InputItem::user("a"),
            InputItem::assistant("b"),
            InputItem::user("c"),
        ])
        .await
        .unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 3);
        let last2 = s.get_items(Some(2)).await.unwrap();
        assert_eq!(last2[0].text_content().as_deref(), Some("b"));
        assert_eq!(
            s.pop_item()
                .await
                .unwrap()
                .unwrap()
                .text_content()
                .as_deref(),
            Some("c")
        );
        s.clear_session().await.unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
    }
}
