//! MongoDB-backed session (feature `mongodb`).
//! Mirrors `agents.extensions.memory.mongodb_session`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::bson::{Bson, DateTime, Document, doc};
use mongodb::options::{
    ClientOptions, DriverInfo, IndexOptions, ReadPreference, ReturnDocument, SelectionCriteria,
};
use mongodb::{Client, Collection, IndexModel};
use tokio::sync::OnceCell;

use super::{Session, SessionSettings};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Default name of the collection holding session metadata.
pub const DEFAULT_SESSIONS_COLLECTION: &str = "agent_sessions";
/// Default name of the collection holding conversation history.
pub const DEFAULT_MESSAGES_COLLECTION: &str = "agent_messages";

fn mongo_err(e: mongodb::error::Error) -> AgentsError {
    AgentsError::other(format!("MongoDB error: {e}"))
}

/// The wrapping-library identity sent in the connection handshake, as the Python session does
/// (`DriverInfo(name="openai-agents", ...)`).
fn driver_info() -> DriverInfo {
    DriverInfo::builder()
        .name("openai-agents")
        .version(env!("CARGO_PKG_VERSION").to_string())
        .build()
}

/// The filter matching the active history generation.
///
/// Generation zero also matches documents written before generations existed, which carry no
/// `generation` field.
pub fn generation_filter(generation: i64) -> Document {
    if generation == 0 {
        doc! {"generation": {"$in": [0, Bson::Null]}}
    } else {
        doc! {"generation": generation}
    }
}

/// Conversation history stored in MongoDB.
///
/// Each `add_items` call writes one document holding the whole batch in a `message_data` array,
/// which makes the append failure-atomic without transactions. Documents written by older
/// versions, where `message_data` is a single serialized item, stay readable. Ordering uses an
/// explicit `seq` counter reserved from the session document, not `_id`.
///
/// `clear_session` advances a generation counter on the session document rather than relying on
/// the delete of every message document: the counter bump is the clear boundary, and removing
/// the obsolete documents afterwards is best effort.
///
/// Indexes (`session_id` unique on the sessions collection, `session_id`/`generation`/`seq` on
/// the messages collection) are created lazily on first use.
///
/// # Compaction
///
/// [`Session::compaction_snapshot`] is deliberately left at its default, so compaction replaces
/// the whole stored history rather than a suffix. A suffix here would span several batch
/// documents, and MongoDB offers no multi-document atomicity outside a transaction — which
/// standalone deployments do not support, and which this backend avoids everywhere else for
/// exactly that reason. The Python SDK makes the same choice: only its SQLite and SQLAlchemy
/// backends implement the snapshot hook.
pub struct MongoDBSession {
    session_id: String,
    sessions: Collection<Document>,
    messages: Collection<Document>,
    settings: Option<SessionSettings>,
    indexes: OnceCell<()>,
    /// The client was created by [`MongoDBSession::from_uri`], so [`MongoDBSession::close`]
    /// releases it.
    owns_client: bool,
    /// Set by the first [`MongoDBSession::close`] of a session that owns its client.
    closed: AtomicBool,
}

impl fmt::Debug for MongoDBSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MongoDBSession")
            .field("session_id", &self.session_id)
            .field("sessions", &self.sessions.name())
            .field("messages", &self.messages.name())
            .finish_non_exhaustive()
    }
}

impl MongoDBSession {
    /// Connect to `uri` and use `database` with the default collection names.
    pub async fn from_uri(
        session_id: impl Into<String>,
        uri: &str,
        database: &str,
    ) -> Result<Self> {
        let mut options = ClientOptions::parse(uri).await.map_err(mongo_err)?;
        // A caller-supplied driver identity in the URI options is kept, not replaced.
        if options.driver_info.is_none() {
            options.driver_info = Some(driver_info());
        }
        let client = Client::with_options(options).map_err(mongo_err)?;
        let mut session = Self::with_client(session_id, &client, database);
        session.owns_client = true;
        Ok(session)
    }

    /// Use a client the application already manages.
    pub fn with_client(session_id: impl Into<String>, client: &Client, database: &str) -> Self {
        // Identify this library in the handshake of a caller-supplied client. Appending is
        // idempotent, and a client that already handshook keeps its metadata, so a failure here
        // is not an error.
        let _ = client.append_metadata(driver_info());
        let db = client.database(database);
        MongoDBSession {
            session_id: session_id.into(),
            sessions: db.collection(DEFAULT_SESSIONS_COLLECTION),
            messages: db.collection(DEFAULT_MESSAGES_COLLECTION),
            settings: None,
            indexes: OnceCell::new(),
            owns_client: false,
            closed: AtomicBool::new(false),
        }
    }

    /// Use custom collection names.
    pub fn with_collections(mut self, sessions: &str, messages: &str) -> Self {
        let db = self.sessions.namespace().db;
        let database = self.sessions.client().database(&db);
        self.sessions = database.collection(sessions);
        self.messages = database.collection(messages);
        self.indexes = OnceCell::new();
        self
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Close the client this session created with [`MongoDBSession::from_uri`].
    ///
    /// The session is terminal from the first call: every later operation fails with
    /// `MongoDBSession is closed`, and repeated calls are safe. A session built with
    /// [`MongoDBSession::with_client`] does not own its client, so this does nothing and the
    /// caller keeps managing the client. Port of `MongoDBSession.close`.
    pub async fn close(&self) {
        if !self.owns_client {
            return;
        }
        self.closed.store(true, Ordering::SeqCst);
        self.sessions
            .client()
            .clone()
            .shutdown()
            .immediate(true)
            .await;
    }

    /// True once [`MongoDBSession::close`] closed a session that owns its client.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn check_not_closed(&self) -> Result<()> {
        if self.is_closed() {
            return Err(AgentsError::other("MongoDBSession is closed"));
        }
        Ok(())
    }

    /// True when the server answers a `ping` command.
    ///
    /// # Errors
    ///
    /// Fails with `MongoDBSession is closed` after [`MongoDBSession::close`] closed a session
    /// that owns its client; an unreachable server is `Ok(false)`.
    pub async fn ping(&self) -> Result<bool> {
        self.check_not_closed()?;
        let db = self.sessions.namespace().db;
        Ok(self
            .sessions
            .client()
            .database(&db)
            .run_command(doc! {"ping": 1})
            .await
            .is_ok())
    }

    async fn ensure_indexes(&self) -> Result<()> {
        self.check_not_closed()?;
        self.indexes
            .get_or_try_init(|| async {
                self.sessions
                    .create_index(
                        IndexModel::builder()
                            .keys(doc! {"session_id": 1})
                            .options(IndexOptions::builder().unique(true).build())
                            .build(),
                    )
                    .await
                    .map_err(mongo_err)?;
                self.messages
                    .create_index(
                        IndexModel::builder()
                            .keys(doc! {"session_id": 1, "generation": 1, "seq": 1})
                            .build(),
                    )
                    .await
                    .map_err(mongo_err)?;
                Ok(())
            })
            .await
            .map(|_| ())
    }

    /// The authoritative history generation for this session.
    ///
    /// Read from the primary even when the client prefers secondaries: a lagging secondary could
    /// still report the generation from before a `clear_session`, which would expose cleared
    /// history to reads and pops. Port of `_get_generation`'s `ReadPreference.PRIMARY`.
    async fn generation(&self) -> Result<i64> {
        let doc = self
            .sessions
            .find_one(doc! {"session_id": &self.session_id})
            .selection_criteria(SelectionCriteria::ReadPreference(ReadPreference::Primary))
            .await
            .map_err(mongo_err)?;
        Ok(bson_int(doc.as_ref().and_then(|d| d.get("_generation"))).unwrap_or(0))
    }

    fn message_filter(&self, generation: i64) -> Document {
        let mut filter = doc! {"session_id": &self.session_id};
        filter.extend(generation_filter(generation));
        filter
    }
}

/// Read an integer field, accepting the `Int32` values other drivers (pymongo) write for
/// small integers as well as `Int64`.
fn bson_int(value: Option<&Bson>) -> Option<i64> {
    let value = value?;
    value.as_i64().or_else(|| value.as_i32().map(i64::from))
}

/// Read the serialized items of one message document, newest last.
fn document_items(doc: &Document) -> Vec<String> {
    match doc.get("message_data") {
        Some(Bson::Array(entries)) => entries
            .iter()
            .filter_map(|e| e.as_str().map(str::to_string))
            .collect(),
        Some(Bson::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Decode serialized items, skipping corrupted entries.
fn decode_items(raw: &[String]) -> Vec<InputItem> {
    raw.iter()
        .filter_map(|r| serde_json::from_str(r).ok().map(InputItem))
        .collect()
}

#[async_trait]
impl Session for MongoDBSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.check_not_closed()?;
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        self.ensure_indexes().await?;
        let filter = self.message_filter(self.generation().await?);

        let Some(limit) = limit else {
            let docs: Vec<Document> = self
                .messages
                .find(filter)
                .sort(doc! {"seq": 1})
                .await
                .map_err(mongo_err)?
                .try_collect()
                .await
                .map_err(mongo_err)?;
            let raw: Vec<String> = docs.iter().flat_map(document_items).collect();
            return Ok(decode_items(&raw));
        };

        // One document can hold a whole batch and corrupted entries must not consume the
        // caller's budget, so widen the window until enough valid items are found.
        let mut window = limit;
        loop {
            let mut docs: Vec<Document> = self
                .messages
                .find(filter.clone())
                .sort(doc! {"seq": -1})
                .limit(window as i64)
                .await
                .map_err(mongo_err)?
                .try_collect()
                .await
                .map_err(mongo_err)?;
            docs.reverse();
            let raw: Vec<String> = docs.iter().flat_map(document_items).collect();
            let items = decode_items(&raw);
            if items.len() >= limit {
                return Ok(items[items.len() - limit..].to_vec());
            }
            if docs.len() < window {
                return Ok(items);
            }
            window = window.saturating_mul(2);
        }
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        // The empty fast path still refuses a closed session.
        self.check_not_closed()?;
        if items.is_empty() {
            return Ok(());
        }
        self.ensure_indexes().await?;
        let serialized: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i.as_value()))
            .collect::<std::result::Result<_, _>>()?;
        let now = Bson::DateTime(DateTime::now());
        let count = serialized.len() as i64;

        // Reserve a contiguous block of sequence numbers for this batch.
        let session = self
            .sessions
            .find_one_and_update(
                doc! {"session_id": &self.session_id},
                doc! {
                    "$setOnInsert": {
                        "session_id": &self.session_id,
                        "created_at": now.clone(),
                        "_generation": 0i64,
                    },
                    "$set": {"updated_at": now},
                    "$inc": {"_seq": count},
                },
            )
            .upsert(true)
            .return_document(ReturnDocument::After)
            .await
            .map_err(mongo_err)?;
        let next_seq = bson_int(session.as_ref().and_then(|d| d.get("_seq"))).unwrap_or(count);
        let generation = bson_int(session.as_ref().and_then(|d| d.get("_generation"))).unwrap_or(0);

        self.messages
            .insert_one(doc! {
                "session_id": &self.session_id,
                "seq": next_seq - 1,
                "generation": generation,
                "message_data": serialized,
            })
            .await
            .map_err(mongo_err)?;
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.ensure_indexes().await?;
        let mut generation = self.generation().await?;

        // Sweep exhausted markers left behind by an earlier pop whose cleanup failed. Markers
        // are never claimable or model-visible, so a failed sweep must not block a valid claim.
        let mut sweep = self.message_filter(generation);
        sweep.insert("message_data", Bson::Array(Vec::new()));
        let _ = self.messages.delete_many(sweep).await;

        loop {
            let mut filter = self.message_filter(generation);
            filter.insert("message_data", doc! {"$ne": []});
            // Claim the newest item atomically by trimming it off the batch array. The claim
            // leaves an empty marker so a concurrent pop cannot claim the same item.
            let claimed = self
                .messages
                .find_one_and_update(
                    filter,
                    vec![doc! {"$set": {"message_data": {"$cond": [
                        {"$isArray": "$message_data"},
                        {"$slice": ["$message_data", {"$subtract": [{"$size": "$message_data"}, 1]}]},
                        [],
                    ]}}}],
                )
                .sort(doc! {"seq": -1})
                .return_document(ReturnDocument::Before)
                .await
                .map_err(mongo_err)?;
            let Some(doc) = claimed else {
                // Nothing left in this generation. A `clear_session` that committed while the
                // claim was in flight moved the history to a new one, so look there instead of
                // reporting an empty session.
                let current = self.generation().await?;
                if current != generation {
                    generation = current;
                    continue;
                }
                return Ok(None);
            };
            // The claim succeeded, but a `clear_session` may have committed between reading the
            // generation and the claim. The claimed document then belongs to history that is no
            // longer visible, so it is discarded rather than returned.
            let current = self.generation().await?;
            if current != generation {
                if let Some(id) = doc.get("_id") {
                    let _ = self.messages.delete_one(doc! {"_id": id.clone()}).await;
                }
                generation = current;
                continue;
            }
            let raw = document_items(&doc);
            let Some(last) = raw.last() else {
                continue;
            };
            if raw.len() == 1
                && let Some(id) = doc.get("_id")
            {
                let _ = self
                    .messages
                    .delete_one(doc! {"_id": id.clone(), "message_data": []})
                    .await;
            }
            if let Ok(value) = serde_json::from_str(last) {
                return Ok(Some(InputItem(value)));
            }
        }
    }

    async fn clear_session(&self) -> Result<()> {
        self.ensure_indexes().await?;
        let now = Bson::DateTime(DateTime::now());
        let session = self
            .sessions
            .find_one_and_update(
                doc! {"session_id": &self.session_id},
                doc! {
                    "$setOnInsert": {
                        "session_id": &self.session_id,
                        "created_at": now.clone(),
                        "_seq": 0i64,
                    },
                    "$set": {"updated_at": now},
                    "$inc": {"_generation": 1i64},
                },
            )
            .upsert(true)
            .return_document(ReturnDocument::After)
            .await
            .map_err(mongo_err)?;
        let generation = bson_int(session.as_ref().and_then(|d| d.get("_generation"))).unwrap_or(1);

        // The generation bump above is the clear boundary; obsolete documents are no longer
        // visible, so removing them is best effort.
        let _ = self
            .messages
            .delete_many(doc! {
                "session_id": &self.session_id,
                "$or": [
                    {"generation": {"$lt": generation}},
                    {"generation": {"$exists": false}},
                ],
            })
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_zero_also_matches_legacy_documents() {
        assert_eq!(
            generation_filter(0),
            doc! {"generation": {"$in": [0, Bson::Null]}}
        );
        assert_eq!(generation_filter(3), doc! {"generation": 3i64});
    }

    #[test]
    fn int32_counters_written_by_other_drivers_are_read() {
        // pymongo stores small Python ints as Int32; the counters must still be readable.
        let doc = doc! {"_generation": 1i32, "_seq": 7i64};
        assert_eq!(bson_int(doc.get("_generation")), Some(1));
        assert_eq!(bson_int(doc.get("_seq")), Some(7));
        assert_eq!(bson_int(doc.get("missing")), None);
        assert_eq!(bson_int(doc! {"x": "no"}.get("x")), None);
    }

    #[test]
    fn batch_and_legacy_documents_both_decode() {
        let batch = doc! {"message_data": ["{\"role\":\"user\"}", "{\"role\":\"assistant\"}"]};
        assert_eq!(document_items(&batch).len(), 2);
        let legacy = doc! {"message_data": "{\"role\":\"user\"}"};
        assert_eq!(document_items(&legacy).len(), 1);
        assert!(document_items(&doc! {"message_data": 7}).is_empty());
        assert!(document_items(&doc! {}).is_empty());
    }

    #[test]
    fn corrupted_entries_are_skipped() {
        let raw = vec![
            r#"{"role":"user","content":"a"}"#.to_string(),
            "not json".to_string(),
        ];
        assert_eq!(decode_items(&raw).len(), 1);
    }

    /// Integration coverage against a real server; skipped unless `MONGODB_URI` is set.
    #[tokio::test]
    async fn mongodb_round_trip() {
        let Ok(uri) = std::env::var("MONGODB_URI") else {
            return;
        };
        let s = MongoDBSession::from_uri("rust-test-session", &uri, "agents_rust_test")
            .await
            .unwrap();
        s.clear_session().await.unwrap();
        s.add_items(vec![InputItem::user("a"), InputItem::assistant("b")])
            .await
            .unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 2);
        assert_eq!(
            s.get_items(Some(1)).await.unwrap()[0].role(),
            Some("assistant")
        );
        assert_eq!(
            s.pop_item().await.unwrap().unwrap().role(),
            Some("assistant")
        );
        assert_eq!(s.get_items(None).await.unwrap().len(), 1);
        s.clear_session().await.unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
    }
}
