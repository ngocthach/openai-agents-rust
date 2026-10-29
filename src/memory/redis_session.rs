//! Redis-backed session (feature `redis`). Mirrors `agents.extensions.memory.redis_session`.

use std::fmt;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use tokio::sync::Mutex;

use super::{Session, SessionSettings};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Default prefix for the Redis keys of a session.
pub const DEFAULT_KEY_PREFIX: &str = "agents:session";

/// The three Redis keys a session occupies: metadata hash, message list and id counter.
///
/// The layout matches the Python SDK: `{prefix}:{session_id}`, `…:messages` and `…:counter`.
pub fn redis_session_keys(key_prefix: &str, session_id: &str) -> (String, String, String) {
    let session_key = format!("{key_prefix}:{session_id}");
    let messages_key = format!("{session_key}:messages");
    let counter_key = format!("{session_key}:counter");
    (session_key, messages_key, counter_key)
}

/// Serialize an item the way the Python backend does (compact JSON).
fn serialize_item(item: &InputItem) -> Result<String> {
    Ok(serde_json::to_string(item.as_value())?)
}

fn redis_err(e: redis::RedisError) -> AgentsError {
    AgentsError::other(format!("Redis error: {e}"))
}

/// Reject a session key occupied by a different Redis data type.
///
/// Redis `TYPE` answers `none` for a key that does not exist, which is always acceptable. A
/// colliding session id can otherwise share a legacy key with something else; Python detects
/// that under `WATCH` and refuses to touch it.
fn validate_key_types(keys: &[(&str, &str, &str)]) -> Result<()> {
    for (key, actual, expected) in keys {
        if *actual != "none" && actual != expected {
            return Err(AgentsError::other(format!(
                "WRONGTYPE Redis key {key} holds a {actual}, expected a {expected}"
            )));
        }
    }
    Ok(())
}

/// Conversation history stored in Redis.
///
/// History lives in a list at `{prefix}:{session_id}:messages`, with a metadata hash at
/// `{prefix}:{session_id}` carrying `session_id`, `created_at` and `updated_at`. An optional
/// TTL is applied to all three keys on every append.
///
/// # Differences from the Python SDK
///
/// Appends and clears run as a single `MULTI`/`EXEC` pipeline instead of a `WATCH`-guarded
/// transaction, because `WATCH` is per-connection state and is not safe on the multiplexed
/// connection this session shares. Key-type validation and the server clock are therefore read
/// in the round trip before the write rather than inside the transaction: a key whose type is
/// changed in the instant between the check and the write is not caught, whereas Python's
/// `WATCH` would abort. Expiry uses a relative `PEXPIRE` instead of Python's `PEXPIREAT`
/// computed from server `TIME`, which makes it independent of clock skew. Redis Cluster is not
/// supported.
pub struct RedisSession {
    session_id: String,
    connection: Mutex<MultiplexedConnection>,
    session_key: String,
    messages_key: String,
    counter_key: String,
    ttl: Option<u64>,
    settings: Option<SessionSettings>,
}

impl fmt::Debug for RedisSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedisSession")
            .field("session_id", &self.session_id)
            .field("messages_key", &self.messages_key)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl RedisSession {
    /// Connect to `url` (for example `redis://localhost:6379/0`) and use the default prefix.
    pub async fn from_url(session_id: impl Into<String>, url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(redis_err)?;
        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(redis_err)?;
        Ok(Self::with_connection(session_id, connection))
    }

    /// Use a connection the application already manages.
    pub fn with_connection(
        session_id: impl Into<String>,
        connection: MultiplexedConnection,
    ) -> Self {
        let session_id = session_id.into();
        let (session_key, messages_key, counter_key) =
            redis_session_keys(DEFAULT_KEY_PREFIX, &session_id);
        RedisSession {
            session_id,
            connection: Mutex::new(connection),
            session_key,
            messages_key,
            counter_key,
            ttl: None,
            settings: None,
        }
    }

    /// Use another key prefix (default: [`DEFAULT_KEY_PREFIX`]).
    pub fn with_key_prefix(mut self, key_prefix: &str) -> Self {
        let (session_key, messages_key, counter_key) =
            redis_session_keys(key_prefix, &self.session_id);
        self.session_key = session_key;
        self.messages_key = messages_key;
        self.counter_key = counter_key;
        self
    }

    /// Expire the session's keys `ttl` seconds after the last append.
    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// The key of the metadata hash.
    pub fn session_key(&self) -> &str {
        &self.session_key
    }

    /// The key of the message list.
    pub fn messages_key(&self) -> &str {
        &self.messages_key
    }

    /// True when Redis answers `PING`.
    pub async fn ping(&self) -> bool {
        let mut conn = self.connection.lock().await;
        redis::cmd("PING")
            .query_async::<String>(&mut *conn)
            .await
            .is_ok()
    }

    /// Check that this session's keys hold the expected Redis types and read the server clock.
    ///
    /// A session id can collide with a legacy key of another type; Python rejects that under
    /// `WATCH`, and this performs the same check in the round trip before the write. Returns
    /// the server's current time in seconds, which the metadata hash records so timestamps do
    /// not depend on the client clock.
    async fn check_keys_and_server_time(&self, conn: &mut MultiplexedConnection) -> Result<u64> {
        let (session_type, messages_type, counter_type, server_time): (
            String,
            String,
            String,
            Vec<String>,
        ) = redis::pipe()
            .cmd("TYPE")
            .arg(&self.session_key)
            .cmd("TYPE")
            .arg(&self.messages_key)
            .cmd("TYPE")
            .arg(&self.counter_key)
            .cmd("TIME")
            .query_async(conn)
            .await
            .map_err(redis_err)?;

        validate_key_types(&[
            (&self.session_key, &session_type, "hash"),
            (&self.messages_key, &messages_type, "list"),
            (&self.counter_key, &counter_type, "string"),
        ])?;

        Ok(server_time
            .first()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0))
    }

    /// Decode a window of raw list entries, skipping corrupted ones.
    fn decode(raw: &[String]) -> Vec<InputItem> {
        raw.iter()
            .filter_map(|r| serde_json::from_str(r).ok().map(InputItem))
            .collect()
    }
}

#[async_trait]
impl Session for RedisSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let mut conn = self.connection.lock().await;
        let Some(limit) = limit else {
            let raw: Vec<String> = redis::cmd("LRANGE")
                .arg(&self.messages_key)
                .arg(0)
                .arg(-1)
                .query_async(&mut *conn)
                .await
                .map_err(redis_err)?;
            return Ok(Self::decode(&raw));
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Corrupted entries must not consume the caller's budget, so widen the window until
        // `limit` valid items are found or the list is exhausted.
        let mut window = limit;
        loop {
            let raw: Vec<String> = redis::cmd("LRANGE")
                .arg(&self.messages_key)
                .arg(-(window as i64))
                .arg(-1)
                .query_async(&mut *conn)
                .await
                .map_err(redis_err)?;
            let items = Self::decode(&raw);
            if items.len() >= limit {
                return Ok(items[items.len() - limit..].to_vec());
            }
            if raw.len() < window {
                return Ok(items);
            }
            window = window.saturating_mul(2);
        }
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let serialized = items
            .iter()
            .map(serialize_item)
            .collect::<Result<Vec<_>>>()?;
        let mut conn = self.connection.lock().await;
        let now = self
            .check_keys_and_server_time(&mut conn)
            .await?
            .to_string();

        let mut pipe = redis::pipe();
        pipe.atomic()
            .cmd("HSET")
            .arg(&self.session_key)
            .arg("session_id")
            .arg(&self.session_id)
            .ignore()
            .cmd("HSETNX")
            .arg(&self.session_key)
            .arg("created_at")
            .arg(&now)
            .ignore()
            .cmd("RPUSH")
            .arg(&self.messages_key)
            .arg(&serialized)
            .ignore()
            .cmd("HSET")
            .arg(&self.session_key)
            .arg("updated_at")
            .arg(&now)
            .ignore();
        if let Some(ttl) = self.ttl {
            let ttl_ms = i64::try_from(ttl.saturating_mul(1000))
                .map_err(|_| AgentsError::user("ttl is outside Redis's supported range"))?;
            for key in [&self.session_key, &self.messages_key, &self.counter_key] {
                pipe.cmd("PEXPIRE").arg(key).arg(ttl_ms).ignore();
            }
        }
        pipe.query_async::<()>(&mut *conn).await.map_err(redis_err)
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let mut conn = self.connection.lock().await;
        loop {
            let raw: Option<String> = redis::cmd("RPOP")
                .arg(&self.messages_key)
                .query_async(&mut *conn)
                .await
                .map_err(redis_err)?;
            let Some(raw) = raw else { return Ok(None) };
            if let Ok(v) = serde_json::from_str(&raw) {
                return Ok(Some(InputItem(v)));
            }
        }
    }

    /// Delete this session's keys, refusing when one is occupied by another Redis type.
    ///
    /// A colliding session id can share a legacy key with something else; in that case nothing
    /// is deleted, matching Python.
    async fn clear_session(&self) -> Result<()> {
        let mut conn = self.connection.lock().await;
        self.check_keys_and_server_time(&mut conn).await?;
        redis::cmd("DEL")
            .arg(&self.session_key)
            .arg(&self.messages_key)
            .arg(&self.counter_key)
            .query_async::<()>(&mut *conn)
            .await
            .map_err(redis_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_layout_matches_the_python_backend() {
        let (session, messages, counter) = redis_session_keys(DEFAULT_KEY_PREFIX, "user-123");
        assert_eq!(session, "agents:session:user-123");
        assert_eq!(messages, "agents:session:user-123:messages");
        assert_eq!(counter, "agents:session:user-123:counter");
        let (session, messages, _) = redis_session_keys("app", "u");
        assert_eq!(session, "app:u");
        assert_eq!(messages, "app:u:messages");
    }

    #[test]
    fn items_serialize_as_compact_json() {
        // Compact separators, and a round trip that preserves the item. Key order depends on
        // whether a dependency enables serde_json's `preserve_order`, so it is not asserted.
        let json = serialize_item(&InputItem::user("hi")).unwrap();
        assert!(!json.contains(' '));
        assert_eq!(json.len(), r#"{"role":"user","content":"hi"}"#.len());
        let back: InputItem = serde_json::from_str(&json).unwrap();
        assert_eq!(back, InputItem::user("hi"));
    }

    #[test]
    fn corrupted_entries_are_skipped() {
        let raw = vec![
            r#"{"role":"user","content":"a"}"#.to_string(),
            "not json".to_string(),
            r#"{"role":"assistant","content":"b"}"#.to_string(),
        ];
        let items = RedisSession::decode(&raw);
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].text_content().as_deref(), Some("b"));
    }

    #[test]
    fn key_type_validation_accepts_missing_keys_and_rejects_collisions() {
        // `none` means the key does not exist yet, which every operation tolerates.
        assert!(
            validate_key_types(&[
                ("agents:session:u", "none", "hash"),
                ("agents:session:u:messages", "none", "list"),
                ("agents:session:u:counter", "none", "string"),
            ])
            .is_ok()
        );
        assert!(
            validate_key_types(&[
                ("agents:session:u", "hash", "hash"),
                ("agents:session:u:messages", "list", "list"),
            ])
            .is_ok()
        );

        let err = validate_key_types(&[
            ("agents:session:u", "hash", "hash"),
            ("agents:session:u:messages", "set", "list"),
        ])
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("WRONGTYPE"), "{message}");
        assert!(message.contains("agents:session:u:messages"), "{message}");
        assert!(
            message.contains("holds a set, expected a list"),
            "{message}"
        );

        // The first mismatch is the one reported.
        let err = validate_key_types(&[
            ("agents:session:u", "stream", "hash"),
            ("agents:session:u:messages", "set", "list"),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("holds a stream"), "{err}");
    }

    /// Integration coverage against a real server; skipped unless `REDIS_URL` is set.
    #[tokio::test]
    async fn redis_round_trip() {
        let Ok(url) = std::env::var("REDIS_URL") else {
            return;
        };
        let s = RedisSession::from_url("rust-test-session", &url)
            .await
            .unwrap()
            .with_ttl(60);
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

        // A key occupied by another type is reported instead of being overwritten.
        {
            let mut conn = s.connection.lock().await;
            let _: () = redis::cmd("DEL")
                .arg(s.messages_key())
                .query_async(&mut *conn)
                .await
                .unwrap();
            let _: () = redis::cmd("SADD")
                .arg(s.messages_key())
                .arg("x")
                .query_async(&mut *conn)
                .await
                .unwrap();
        }
        let err = s.add_items(vec![InputItem::user("c")]).await.unwrap_err();
        assert!(err.to_string().contains("WRONGTYPE"), "{err}");
        assert!(s.clear_session().await.is_err());
        {
            let mut conn = s.connection.lock().await;
            let _: () = redis::cmd("DEL")
                .arg(s.messages_key())
                .query_async(&mut *conn)
                .await
                .unwrap();
        }

        s.clear_session().await.unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
    }
}
