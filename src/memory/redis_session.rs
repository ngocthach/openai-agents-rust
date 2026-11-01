//! Redis-backed session (feature `redis`). Mirrors `agents.extensions.memory.redis_session`.

use std::fmt;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use tokio::sync::Mutex;

use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
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

/// Replace a suffix of the message list atomically, refreshing the session's TTL.
///
/// `KEYS` are the messages, metadata and counter keys. `ARGV[1]` is the TTL in milliseconds (0
/// for none), `ARGV[2]` the `updated_at` stamp, `ARGV[3]` the number of handles, then the
/// handles in order, the number of replacement entries, and the entries themselves.
///
/// The script verifies that the list still ends with exactly those handles before trimming them
/// off and appending the replacement, so a concurrent append makes it return 0 and change
/// nothing. Running the whole check-and-replace inside Redis is what makes it atomic without
/// `WATCH`. The expiry is re-applied afterwards because trimming the list to empty deletes the
/// key, which would otherwise resurrect it without a TTL.
/// Lua prelude shared by every writing script.
///
/// `guard()` reproduces the key-type validation Python performs under `WATCH`, and `now()`
/// reads the server clock. Both run inside the script, so the whole check-then-write sequence
/// is one atomic Redis operation and a type that changes between them cannot be missed — the
/// idiomatic Redis equivalent of Python's `WATCH`/`MULTI`.
///
/// `TYPE` answers `none` for a key that does not exist, which is always acceptable.
/// Scripts are replicated by their effects from Redis 5 on, so reading `TIME` is allowed.
const KEY_GUARD_PRELUDE: &str = r#"
local function check(key, expected)
  local actual = redis.call('TYPE', key)['ok']
  if actual ~= 'none' and actual ~= expected then
    return 'WRONGTYPE Redis key ' .. key .. ' holds a ' .. actual .. ', expected a ' .. expected
  end
  return nil
end
local function guard()
  return check(KEYS[1], 'list') or check(KEYS[2], 'hash') or check(KEYS[3], 'string')
end
local function now()
  return redis.call('TIME')[1]
end
"#;

/// Append items, stamping the metadata hash with the server clock.
///
/// `KEYS` are the message list, the metadata hash and the counter; `ARGV` is the TTL in
/// milliseconds (0 for none), the session id, then one serialized item per remaining argument.
const ADD_ITEMS_SCRIPT: &str = r#"
local err = guard()
if err then return redis.error_reply(err) end
local ttl_ms = tonumber(ARGV[1])
local stamp = now()
redis.call('HSET', KEYS[2], 'session_id', ARGV[2])
redis.call('HSETNX', KEYS[2], 'created_at', stamp)
for i = 3, #ARGV do
  redis.call('RPUSH', KEYS[1], ARGV[i])
end
redis.call('HSET', KEYS[2], 'updated_at', stamp)
if ttl_ms > 0 then
  for i = 1, 3 do
    redis.call('PEXPIRE', KEYS[i], ttl_ms)
  end
end
return 1
"#;

/// Delete every key of one session, refusing keys another type occupies.
const CLEAR_SESSION_SCRIPT: &str = r#"
local err = guard()
if err then return redis.error_reply(err) end
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3])
return 1
"#;

/// Replace a suffix of the message list.
///
/// `ARGV` is the TTL in milliseconds, the number of handles, the handles, the number of
/// replacement items, then the items.
const REPLACE_SUFFIX_SCRIPT: &str = r#"
local err = guard()
if err then return redis.error_reply(err) end
local ttl_ms = tonumber(ARGV[1])
local handle_count = tonumber(ARGV[2])
local stamp = now()
local total = redis.call('LLEN', KEYS[1])
if total < handle_count then return 0 end
local start = total - handle_count
if handle_count > 0 then
  local tail = redis.call('LRANGE', KEYS[1], start, -1)
  for i = 1, handle_count do
    if tail[i] ~= ARGV[2 + i] then return 0 end
  end
end
if start == 0 then
  redis.call('DEL', KEYS[1])
else
  redis.call('LTRIM', KEYS[1], 0, start - 1)
end
local item_count = tonumber(ARGV[3 + handle_count])
for i = 1, item_count do
  redis.call('RPUSH', KEYS[1], ARGV[3 + handle_count + i])
end
redis.call('HSET', KEYS[2], 'updated_at', stamp)
if ttl_ms > 0 then
  for i = 1, 3 do
    if redis.call('EXISTS', KEYS[i]) == 1 then
      redis.call('PEXPIRE', KEYS[i], ttl_ms)
    end
  end
end
return 1
"#;

/// Replace the whole history in one step, like `clear_session` followed by an append.
///
/// `ARGV` is the TTL in milliseconds, the session id, then the replacement items. The metadata
/// and counter keys are deleted too, so an empty replacement leaves no key behind and a
/// non-empty one starts a fresh metadata hash.
const REPLACE_ALL_SCRIPT: &str = r#"
local err = guard()
if err then return redis.error_reply(err) end
local ttl_ms = tonumber(ARGV[1])
local stamp = now()
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3])
if #ARGV > 2 then
  redis.call('HSET', KEYS[2], 'session_id', ARGV[2])
  redis.call('HSET', KEYS[2], 'created_at', stamp)
  for i = 3, #ARGV do
    redis.call('RPUSH', KEYS[1], ARGV[i])
  end
  redis.call('HSET', KEYS[2], 'updated_at', stamp)
  if ttl_ms > 0 then
    for i = 1, 3 do
      if redis.call('EXISTS', KEYS[i]) == 1 then
        redis.call('PEXPIRE', KEYS[i], ttl_ms)
      end
    end
  end
end
return 1
"#;

/// Serialize an item the way the Python backend does (compact JSON).
fn serialize_item(item: &InputItem) -> Result<String> {
    Ok(serde_json::to_string(item.as_value())?)
}

fn redis_err(e: redis::RedisError) -> AgentsError {
    AgentsError::other(format!("Redis error: {e}"))
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

    /// An `EVAL` of `body` with this session's three keys and the shared guard prelude.
    ///
    /// Folding the key-type validation and the server clock into the script makes every write
    /// one atomic Redis operation, so a key whose type changes between the check and the write
    /// cannot slip through — what Python's `WATCH`/`MULTI` guarantees. Caller arguments start
    /// at `ARGV[1]`.
    fn script(&self, body: &str) -> redis::Cmd {
        let mut cmd = redis::cmd("EVAL");
        cmd.arg(format!("{KEY_GUARD_PRELUDE}{body}"))
            .arg(3)
            .arg(&self.messages_key)
            .arg(&self.session_key)
            .arg(&self.counter_key);
        cmd
    }

    /// The TTL in milliseconds, or 0 when the session does not expire.
    fn ttl_millis(&self) -> Result<i64> {
        match self.ttl {
            Some(ttl) => i64::try_from(ttl.saturating_mul(1000)).map_err(|_| {
                AgentsError::user("ttl is outside Redis's supported expiration range")
            }),
            None => Ok(0),
        }
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
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
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
        let ttl_ms = self.ttl_millis()?;
        let mut conn = self.connection.lock().await;
        let mut cmd = self.script(ADD_ITEMS_SCRIPT);
        cmd.arg(ttl_ms).arg(&self.session_id);
        for entry in &serialized {
            cmd.arg(entry);
        }
        cmd.query_async::<i64>(&mut *conn)
            .await
            .map(|_| ())
            .map_err(redis_err)
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
    /// Snapshot the tail of the list; handles are the serialized entries themselves.
    ///
    /// A Redis list has no per-item ids, so each entry's exact stored form identifies it. The
    /// replacement re-checks those entries inside Redis, which detects any concurrent write.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        let mut conn = self.connection.lock().await;
        // One entry beyond the window tells us whether it reached the start of the history.
        let window = i64::try_from(limit)
            .unwrap_or(i64::MAX)
            .saturating_add(1)
            .saturating_neg();
        let raw: Vec<String> = redis::cmd("LRANGE")
            .arg(&self.messages_key)
            .arg(window)
            .arg(-1)
            .query_async(&mut *conn)
            .await
            .map_err(redis_err)?;
        let complete = raw.len() <= limit;
        let window = &raw[raw.len() - raw.len().min(limit)..];

        let mut items = Vec::new();
        let mut handles = Vec::new();
        let mut intact = complete;
        for entry in window {
            match serde_json::from_str(entry) {
                Ok(value) => {
                    items.push(InputItem(value));
                    handles.push(SnapshotHandle(entry.clone()));
                }
                Err(_) => {
                    // A corrupted entry cannot be sent to the compaction API, so it and
                    // everything before it stay outside the replaceable window.
                    items.clear();
                    handles.clear();
                    intact = false;
                }
            }
        }
        Ok(Some(CompactionSnapshot {
            items,
            handles,
            complete: intact,
        }))
    }

    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        let replacement = items
            .iter()
            .map(serialize_item)
            .collect::<Result<Vec<_>>>()?;
        let ttl_ms = self.ttl_millis()?;
        let mut conn = self.connection.lock().await;
        let mut cmd = self.script(REPLACE_SUFFIX_SCRIPT);
        cmd.arg(ttl_ms).arg(handles.len());
        for handle in handles {
            cmd.arg(&handle.0);
        }
        cmd.arg(replacement.len());
        for entry in &replacement {
            cmd.arg(entry);
        }
        let replaced: i64 = cmd.query_async(&mut *conn).await.map_err(redis_err)?;
        Ok(replaced == 1)
    }

    async fn replace_history_atomically(&self, items: Vec<InputItem>) -> Result<bool> {
        let replacement = items
            .iter()
            .map(serialize_item)
            .collect::<Result<Vec<_>>>()?;
        let ttl_ms = self.ttl_millis()?;
        let mut conn = self.connection.lock().await;
        let mut cmd = self.script(REPLACE_ALL_SCRIPT);
        cmd.arg(ttl_ms).arg(&self.session_id);
        for entry in &replacement {
            cmd.arg(entry);
        }
        cmd.query_async::<i64>(&mut *conn)
            .await
            .map(|_| true)
            .map_err(redis_err)
    }

    async fn clear_session(&self) -> Result<()> {
        let mut conn = self.connection.lock().await;
        self.script(CLEAR_SESSION_SCRIPT)
            .query_async::<i64>(&mut *conn)
            .await
            .map(|_| ())
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
    fn every_write_runs_the_key_type_guard_inside_one_script() {
        // The check and the write are one `EVAL`, so a type that changes between them cannot
        // slip through — what Python's `WATCH`/`MULTI` guarantees.
        for body in [
            ADD_ITEMS_SCRIPT,
            CLEAR_SESSION_SCRIPT,
            REPLACE_SUFFIX_SCRIPT,
        ] {
            assert!(body.contains("local err = guard()"), "{body}");
            assert!(body.contains("return redis.error_reply(err)"), "{body}");
        }
        assert!(KEY_GUARD_PRELUDE.contains("redis.call('TYPE', key)"));
        assert!(KEY_GUARD_PRELUDE.contains("WRONGTYPE Redis key "));
        // `none` means the key does not exist yet, which every operation tolerates.
        assert!(KEY_GUARD_PRELUDE.contains("actual ~= 'none'"));
        // Timestamps come from the server clock, read inside the same script.
        assert!(KEY_GUARD_PRELUDE.contains("redis.call('TIME')"));
        assert!(ADD_ITEMS_SCRIPT.contains("local stamp = now()"));
        assert!(REPLACE_SUFFIX_SCRIPT.contains("local stamp = now()"));
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

        // Atomic suffix replacement: the Lua script keeps the retained prefix and refuses a
        // stale tail.
        s.add_items(vec![
            InputItem::user("a"),
            InputItem::assistant("b"),
            InputItem::user("c"),
        ])
        .await
        .unwrap();
        let snapshot = s.compaction_snapshot(10).await.unwrap().unwrap();
        assert!(snapshot.complete);
        assert_eq!(snapshot.items.len(), 3);
        let bounded = s.compaction_snapshot(2).await.unwrap().unwrap();
        assert!(!bounded.complete);
        assert_eq!(bounded.items.len(), 2);

        assert!(
            s.replace_compaction_suffix(
                &snapshot.handles[1..],
                vec![InputItem::assistant("recap")]
            )
            .await
            .unwrap()
        );
        let items = s.get_items(None).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].text_content().as_deref(), Some("recap"));
        assert!(
            !s.replace_compaction_suffix(&snapshot.handles[1..], vec![InputItem::user("x")])
                .await
                .unwrap()
        );
        assert_eq!(s.get_items(None).await.unwrap().len(), 2);

        // Replacing the whole list recreates the key, so the TTL has to be re-applied: the
        // messages key must not outlive the metadata and counter keys.
        let snapshot = s.compaction_snapshot(10).await.unwrap().unwrap();
        assert!(
            s.replace_compaction_suffix(&snapshot.handles, vec![InputItem::assistant("all new")])
                .await
                .unwrap()
        );
        {
            let mut conn = s.connection.lock().await;
            for key in [s.messages_key(), s.session_key()] {
                let ttl: i64 = redis::cmd("PTTL")
                    .arg(key)
                    .query_async(&mut *conn)
                    .await
                    .unwrap();
                assert!(ttl > 0, "{key} lost its expiry: PTTL = {ttl}");
            }
        }
        assert_eq!(s.get_items(None).await.unwrap().len(), 1);

        s.clear_session().await.unwrap();
    }
}
