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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::items::InputItem;

#[cfg(feature = "sqlite")]
pub use advanced_sqlite_session::{
    AdvancedSqliteSession, BranchInfo, ConversationTurn, MAIN_BRANCH, SessionUsage, ToolUsage,
    TurnUsage,
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

/// Information passed to [`Session::on_run_completed`] after the runner persisted a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionRunCompletion {
    /// Id of the last model response (Responses API), if any.
    pub response_id: Option<String>,
    /// The `store` model setting used for the last response.
    pub store: Option<bool>,
    /// Whether the last turn produced local tool outputs.
    pub has_local_tool_outputs: bool,
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
}

/// A session stored in process memory.
#[derive(Debug, Default)]
pub struct InMemorySession {
    session_id: String,
    settings: Option<SessionSettings>,
    items: Mutex<Vec<InputItem>>,
}

impl InMemorySession {
    /// A new empty in-memory session.
    pub fn new(session_id: impl Into<String>) -> Self {
        InMemorySession {
            session_id: session_id.into(),
            settings: None,
            items: Mutex::new(Vec::new()),
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
        let items = self.items.lock().expect("session poisoned");
        Ok(match limit {
            None => items.clone(),
            Some(0) => Vec::new(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.items.lock().expect("session poisoned").extend(items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().expect("session poisoned").pop())
    }

    async fn clear_session(&self) -> Result<()> {
        self.items.lock().expect("session poisoned").clear();
        Ok(())
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
