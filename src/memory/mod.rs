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
mod session_rewind;
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

/// How many items one Conversations items-create request carries; a longer append is split.
pub(crate) fn openai_conversations_chunk_size() -> usize {
    openai_conversations_session::MAX_ITEMS_PER_REQUEST
}
#[cfg(feature = "mongodb")]
pub use mongodb_session::{
    DEFAULT_MESSAGES_COLLECTION, DEFAULT_SESSIONS_COLLECTION, MongoDBSession, generation_filter,
};
pub use openai_conversations_session::{
    OpenAIConversationsSession, start_openai_conversations_session,
};
pub use openai_responses_compaction_session::{
    CompactionContext, CompactionMode, DEFAULT_COMPACTION_THRESHOLD, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, ResolvedCompactionMode, ShouldTriggerCompaction,
    default_should_trigger_compaction, is_openai_model_name, select_compaction_candidate_items,
};
#[cfg(feature = "redis")]
pub use redis_session::{DEFAULT_KEY_PREFIX, RedisSession, redis_session_keys};
pub use session_rewind::{fingerprint_input_item, rewind_session_items, wait_for_session_cleanup};
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

/// Run a session mutation to completion even if the caller is dropped.
///
/// Mirrors `agents.memory.session.await_session_mutation`. Once this future has been polled
/// once the mutation runs on its own task, so dropping the caller — a cancelled `select!`
/// branch, an elapsed `timeout` — cannot abort a half-written mutation. Wrappers that must not
/// leave a store half-updated (compaction's replacement, the encrypted wrapper's writes) route
/// their mutations through it.
///
/// Difference from Python: `asyncio` can defer a cancellation request until the operation
/// settles and then raise it, while dropping a Rust future is not awaitable. The mutation still
/// completes, but a dropped caller never observes its outcome; as in Python, a caller that goes
/// away must not assume the write was rolled back.
///
/// This provides no timeout, transaction, retry or mutual exclusion.
pub async fn await_session_mutation<T>(
    mutation: impl Future<Output = Result<T>> + Send + 'static,
) -> Result<T>
where
    T: Send + 'static,
{
    tokio::spawn(mutation)
        .await
        .map_err(|e| crate::error::AgentsError::other(format!("session mutation failed: {e}")))?
}

/// Decides whether a stored item at the head of the history may be physically reclaimed during
/// a successful [`Session::replace_compaction_suffix_pruning`].
///
/// Mirrors the `prune_prefix` callable Python's encrypted snapshot passes down. The predicate
/// sees the item exactly as the store holds it — for an encrypted store, the envelope rather
/// than its plaintext.
pub type PrunePrefix = std::sync::Arc<dyn Fn(&InputItem) -> bool + Send + Sync>;

/// Rejects a claimed item before its removal is committed, mirroring the `validate` callback of
/// Python's `SQLiteSession._pop_item_with_validation`.
pub type PopValidator = Box<dyn Fn(&InputItem) -> Result<()> + Send + Sync>;

/// How many of the oldest rows one prefix-reclamation round reads at a time. Bounds the memory
/// a single round uses; the loop keeps going while whole batches are reclaimable.
#[allow(dead_code)]
pub(crate) const PRUNE_PREFIX_BATCH: i64 = 256;

/// Settings controlling how history is retrieved.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    canonical_json(&value).to_string()
}

/// `value` with every object's keys in sorted order, so two spellings of one item that differ
/// only in key order digest the same (Python fingerprints with `sort_keys=True`).
fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonical_json(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_json).collect()),
        other => other.clone(),
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
    /// [`compaction_item_digest`] of every item of the last model exchange, in order:
    /// everything the model was sent followed by everything it produced.
    ///
    /// Sessions that compact history match the stored tail against this, so only history the
    /// model actually just saw is eligible for replacement. The runner fills this only for
    /// sessions whose [`Session::wants_model_exchange`] is `true`; left empty, compaction falls
    /// back to replacing the whole stored history, which is what manual `run_compaction` calls
    /// and pre-existing callers already did.
    pub model_exchange: Vec<String>,
    /// Whether the run omitted reasoning item ids from what it sent the model. A stored
    /// reasoning item that still carries its id then matches the exchange without it, and is
    /// compacted in the form the model saw.
    pub omit_reasoning_item_ids: bool,
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

/// A type-erased view of the run's [`RunContext`](crate::run_context::RunContext), handed to the `*_with_context` methods of
/// [`Session`]. Downcast it with [`SessionRunContext::run_context`] to the context type the run
/// was started with.
#[derive(Clone, Copy)]
pub struct SessionRunContext<'a> {
    inner: &'a dyn ErasedRunContext,
}

/// The run context behind a [`SessionRunContext`], with the one capability the wrapper needs
/// beyond a downcast: making an owned copy that can outlive the borrow.
trait ErasedRunContext: Send + Sync {
    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync);
    fn detach(&self) -> std::sync::Arc<dyn ErasedRunContext>;
}

impl<C: Send + Sync + 'static> ErasedRunContext for crate::run_context::RunContext<C> {
    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync) {
        self
    }

    fn detach(&self) -> std::sync::Arc<dyn ErasedRunContext> {
        // A `RunContext` clone shares the context, usage and approvals of the original.
        std::sync::Arc::new(self.clone())
    }
}

impl<'a> SessionRunContext<'a> {
    /// Wrap the run's context.
    pub fn new<C: Send + Sync + 'static>(ctx: &'a crate::run_context::RunContext<C>) -> Self {
        SessionRunContext { inner: ctx }
    }

    /// The run's context wrapper, when the run was started with a context of type `C`.
    pub fn run_context<C: Send + Sync + 'static>(
        &self,
    ) -> Option<&'a crate::run_context::RunContext<C>> {
        self.inner.as_any().downcast_ref()
    }

    /// The app-defined context object, when the run was started with a context of type `C`.
    pub fn context<C: Send + Sync + 'static>(&self) -> Option<&'a C> {
        self.run_context::<C>().map(|c| c.context())
    }

    /// An owned copy of this context that does not borrow from the caller.
    ///
    /// It shares the run's context, usage and approvals, so a store that must finish a write
    /// after its caller was dropped (see [`await_session_mutation`]) can still hand the store
    /// the context the write belongs to.
    pub fn detach(&self) -> OwnedSessionRunContext {
        OwnedSessionRunContext {
            inner: self.inner.detach(),
        }
    }
}

/// A [`SessionRunContext`] that owns its context, from [`SessionRunContext::detach`].
#[derive(Clone)]
pub struct OwnedSessionRunContext {
    inner: std::sync::Arc<dyn ErasedRunContext>,
}

impl OwnedSessionRunContext {
    /// Borrow it as the context the `*_with_context` methods take.
    pub fn as_context(&self) -> SessionRunContext<'_> {
        SessionRunContext {
            inner: self.inner.as_ref(),
        }
    }
}

impl fmt::Debug for OwnedSessionRunContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSessionRunContext")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for SessionRunContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionRunContext").finish_non_exhaustive()
    }
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

    /// Whether stored items come back without the ids they were written with, so replayed
    /// history must be matched without them. Defaults to [`Session::manages_history_remotely`];
    /// a local store that reassigns ids on write overrides it. Mirrors the
    /// `_ignore_ids_for_matching` attribute of Python sessions.
    fn ignores_item_ids_for_matching(&self) -> bool {
        self.manages_history_remotely()
    }

    /// Retrieve history, the most recent `limit` items when set (chronological order).
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>>;

    /// Append items.
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()>;

    /// Remove and return the most recent item.
    async fn pop_item(&self) -> Result<Option<InputItem>>;

    /// Remove and return the most recent item, rejecting it before the removal is committed.
    ///
    /// `validate` runs on the claimed item inside whatever transaction owns the removal. An
    /// `Err` rolls that removal back, leaving the item and its position untouched, and is
    /// returned to the caller. Stores that cannot claim and validate atomically keep the
    /// default, which pops first and so *does* remove an item `validate` rejects — exactly the
    /// split Python draws between its native `SQLiteSession` and every other backend.
    ///
    /// Port of `SQLiteSession._pop_item_with_validation`.
    async fn pop_item_validated(&self, validate: PopValidator) -> Result<Option<InputItem>> {
        let item = self.pop_item().await?;
        if let Some(item) = &item {
            validate(item)?;
        }
        Ok(item)
    }

    /// Remove all items.
    async fn clear_session(&self) -> Result<()>;

    /// [`Session::pop_item_validated`] for a run: the removal is scoped by the run's context
    /// (see [`Session::get_items_with_context`]). The default pops through
    /// [`Session::pop_item_with_context`] and validates afterwards, so a store that claims and
    /// validates atomically but ignores the context overrides it with its own validated pop.
    async fn pop_item_validated_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        validate: PopValidator,
    ) -> Result<Option<InputItem>> {
        let item = self.pop_item_with_context(ctx).await?;
        if let Some(item) = &item {
            validate(item)?;
        }
        Ok(item)
    }

    /// [`Session::get_items`] for a run: the runner calls this with the run's context, so a
    /// session that scopes its history by the context (a tenant, a user) can override it.
    /// The default ignores the context and reads the unscoped history.
    ///
    /// Port of the opt-in `wrapper` argument of Python's `Session.get_items`.
    async fn get_items_with_context(
        &self,
        _ctx: SessionRunContext<'_>,
        limit: Option<usize>,
    ) -> Result<Vec<InputItem>> {
        self.get_items(limit).await
    }

    /// [`Session::add_items`] for a run; see [`Session::get_items_with_context`].
    async fn add_items_with_context(
        &self,
        _ctx: SessionRunContext<'_>,
        items: Vec<InputItem>,
    ) -> Result<()> {
        self.add_items(items).await
    }

    /// [`Session::pop_item`] for a run; see [`Session::get_items_with_context`].
    async fn pop_item_with_context(
        &self,
        _ctx: SessionRunContext<'_>,
    ) -> Result<Option<InputItem>> {
        self.pop_item().await
    }

    /// [`Session::clear_session`] for a run; see [`Session::get_items_with_context`].
    async fn clear_session_with_context(&self, _ctx: SessionRunContext<'_>) -> Result<()> {
        self.clear_session().await
    }

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

    /// Replace the whole stored history with `items` in one atomic step.
    ///
    /// Stores that can do this without a window where the history is empty override it and
    /// return `true`; the compaction session then uses it instead of `clear_session` followed
    /// by `add_items`. The default returns `false` without changing anything, and wrappers that
    /// transform stored items (such as encryption) keep it, so they are never bypassed.
    /// Port of the `_atomic_replace_items` hook of Python's `RedisSession`.
    async fn replace_history_atomically(&self, _items: Vec<InputItem>) -> Result<bool> {
        Ok(false)
    }

    /// [`Session::replace_compaction_suffix`], additionally reclaiming a contiguous prefix of
    /// items `prune_prefix` accepts.
    ///
    /// The scan starts at the oldest stored item and stops at the first item the predicate
    /// rejects or that the store cannot read, so a rejected item always keeps everything after
    /// it. Reclamation happens in the same transaction as the replacement and only when the
    /// replacement succeeds, so a store that returns `false` deletes nothing. Stores that
    /// cannot do this keep the default, which ignores the predicate.
    ///
    /// Port of the `prune_prefix` argument of `SQLiteSession._get_compaction_snapshot`.
    async fn replace_compaction_suffix_pruning(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
        _prune_prefix: PrunePrefix,
    ) -> Result<bool> {
        self.replace_compaction_suffix(handles, items).await
    }

    /// Let an outer session that stores items in another form (for example encrypted) decide how
    /// a compaction session reads and rewrites the store beneath it.
    ///
    /// A session that compacts history reads and replaces the *stored* items itself. When
    /// another session wraps it and stores those items transformed, the compaction session
    /// would otherwise read the wrapper's stored form: it would send ciphertext to the
    /// compaction API and write its plaintext summary into an encrypted store. `wrap` receives
    /// the store the compaction session works on and returns the view it must use for every
    /// internal read and write instead; the session's own [`Session`] methods keep addressing
    /// the raw store, because the outer wrapper applies its transformation to those.
    ///
    /// Returns `Ok(true)` when the view was installed. Every session that does not compact
    /// ignores the call and returns `Ok(false)`. Port of the `read_items` / `prepare_items` /
    /// `read_snapshot` arguments Python's `EncryptedSession` passes to
    /// `OpenAIResponsesCompactionSession._run_compaction`.
    fn wrap_compaction_store(&self, _wrap: &CompactionStoreWrapper<'_>) -> Result<bool> {
        Ok(false)
    }
}

/// Builds the view of a store a compaction session should use; see
/// [`Session::wrap_compaction_store`].
pub type CompactionStoreWrapper<'a> =
    dyn Fn(std::sync::Arc<dyn Session>) -> Result<std::sync::Arc<dyn Session>> + Send + Sync + 'a;

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

    #[tokio::test]
    async fn a_shielded_mutation_outlives_its_caller() {
        use std::sync::Arc;
        use std::time::Duration;

        let session = Arc::new(InMemorySession::new("s1"));
        let gate = Arc::new(tokio::sync::Notify::new());
        let (store, release) = (session.clone(), gate.clone());
        let mutation = await_session_mutation(async move {
            release.notified().await;
            store.add_items(vec![InputItem::user("a")]).await
        });

        // The caller goes away before the write can land.
        assert!(
            tokio::time::timeout(Duration::from_millis(50), mutation)
                .await
                .is_err()
        );
        gate.notify_one();

        for _ in 0..200 {
            if session.get_items(None).await.unwrap().len() == 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("the mutation was aborted together with its caller");
    }
}
