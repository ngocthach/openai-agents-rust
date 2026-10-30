//! Session backed by a Dapr state store, through the sidecar's HTTP state API.
//! Mirrors `agents.extensions.memory.dapr_session`.

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Default HTTP endpoint of the Dapr sidecar.
pub const DEFAULT_DAPR_HTTP_ENDPOINT: &str = "http://localhost:3500";

/// Maximum number of attempts for a conditional write.
const MAX_WRITE_ATTEMPTS: u32 = 5;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(50);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(1);

/// Consistency level requested for Dapr state operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DaprConsistency {
    /// Eventual consistency (the Dapr default).
    #[default]
    Eventual,
    /// Strong consistency, where the state store supports it.
    Strong,
}

impl DaprConsistency {
    /// The wire value Dapr expects.
    pub fn as_str(self) -> &'static str {
        match self {
            DaprConsistency::Eventual => "eventual",
            DaprConsistency::Strong => "strong",
        }
    }
}

/// The two state keys a session occupies, matching the Python backend.
pub fn dapr_session_keys(session_id: &str) -> (String, String) {
    (
        format!("{session_id}:messages"),
        format!("{session_id}:metadata"),
    )
}

/// Conversation history stored in a Dapr state store.
///
/// The history lives under `{session_id}:messages` as a JSON array of serialized items, with
/// bookkeeping (`session_id`, `created_at`, `updated_at`) under `{session_id}:metadata`. Both
/// keys use the same names as the Python SDK, and the stored bytes are the same JSON array, so
/// histories are interchangeable.
///
/// Appends and pops are conditional writes guarded by the ETag read with the value, requested
/// with `first-write` concurrency and retried with exponential backoff on conflict. The
/// metadata write is derived bookkeeping: when it fails after the items were committed the
/// failure is logged instead of raised, so a caller cannot retry the append and store the
/// batch twice.
///
/// # Differences from the Python SDK
///
/// Python talks to the sidecar over gRPC through the Dapr SDK; this port issues the equivalent
/// HTTP state API requests with `reqwest`, sending the TTL as per-item `metadata.ttlInSeconds`
/// in the request body as the HTTP API documents.
pub struct DaprSession {
    session_id: String,
    http: reqwest::Client,
    endpoint: String,
    state_store_name: String,
    api_token: Option<String>,
    ttl: Option<u64>,
    consistency: DaprConsistency,
    messages_key: String,
    metadata_key: String,
    settings: Option<SessionSettings>,
    lock: Mutex<()>,
}

impl fmt::Debug for DaprSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaprSession")
            .field("session_id", &self.session_id)
            .field("endpoint", &self.endpoint)
            .field("state_store_name", &self.state_store_name)
            .field("ttl", &self.ttl)
            .field("consistency", &self.consistency)
            .finish_non_exhaustive()
    }
}

/// A state value read from the sidecar, with the ETag that backs it.
#[derive(Debug, Default)]
struct StateRead {
    body: Option<String>,
    etag: Option<String>,
}

impl DaprSession {
    /// A session talking to the sidecar at `endpoint` (for example `http://localhost:3500`).
    pub fn new(
        session_id: impl Into<String>,
        state_store_name: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        let session_id = session_id.into();
        let (messages_key, metadata_key) = dapr_session_keys(&session_id);
        DaprSession {
            session_id,
            http: crate::util::http_client(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            state_store_name: state_store_name.into(),
            api_token: None,
            ttl: None,
            consistency: DaprConsistency::Eventual,
            messages_key,
            metadata_key,
            settings: None,
            lock: Mutex::new(()),
        }
    }

    /// A session configured from `DAPR_HTTP_ENDPOINT`, `DAPR_HTTP_PORT` and `DAPR_API_TOKEN`.
    pub fn from_env(session_id: impl Into<String>, state_store_name: impl Into<String>) -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let endpoint = env("DAPR_HTTP_ENDPOINT").unwrap_or_else(|| match env("DAPR_HTTP_PORT") {
            Some(port) => format!("http://localhost:{port}"),
            None => DEFAULT_DAPR_HTTP_ENDPOINT.to_string(),
        });
        let mut s = Self::new(session_id, state_store_name, endpoint);
        s.api_token = env("DAPR_API_TOKEN");
        s
    }

    /// Send `dapr-api-token` on every request.
    pub fn with_api_token(mut self, token: impl Into<String>) -> Self {
        self.api_token = Some(token.into());
        self
    }

    /// Request `ttlInSeconds` on every write (support depends on the state store).
    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set the consistency level requested for state operations.
    pub fn with_consistency(mut self, consistency: DaprConsistency) -> Self {
        self.consistency = consistency;
        self
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Use a custom `reqwest::Client` (proxies, timeouts, TLS settings).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The state key holding the conversation history.
    pub fn messages_key(&self) -> &str {
        &self.messages_key
    }

    /// The state key holding the session bookkeeping.
    pub fn metadata_key(&self) -> &str {
        &self.metadata_key
    }

    fn state_url(&self) -> String {
        format!("{}/v1.0/state/{}", self.endpoint, self.state_store_name)
    }

    fn with_token(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_token {
            Some(token) => req.header("dapr-api-token", token),
            None => req,
        }
    }

    /// True when the sidecar answers the metadata endpoint.
    pub async fn ping(&self) -> bool {
        let url = format!("{}/v1.0/metadata", self.endpoint);
        self.with_token(self.http.get(url))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }

    async fn get_state(&self, key: &str) -> Result<StateRead> {
        let url = format!("{}/{key}", self.state_url());
        let resp = self
            .with_token(self.http.get(url))
            .query(&[("consistency", self.consistency.as_str())])
            .send()
            .await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NO_CONTENT || status == reqwest::StatusCode::NOT_FOUND {
            return Ok(StateRead::default());
        }
        if !status.is_success() {
            return Err(dapr_error(status, resp.text().await.unwrap_or_default()));
        }
        let etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .filter(|e| !e.is_empty());
        let body = resp.text().await?;
        Ok(StateRead {
            body: (!body.is_empty()).then_some(body),
            etag,
        })
    }

    async fn save_state(&self, key: &str, value: Value, etag: Option<&str>) -> Result<()> {
        let mut entry = json!({
            "key": key,
            "value": value,
            "options": {"consistency": self.consistency.as_str()},
        });
        if let Some(etag) = etag {
            entry["etag"] = json!(etag);
            entry["options"]["concurrency"] = json!("first-write");
        }
        if let Some(ttl) = self.ttl {
            entry["metadata"] = json!({"ttlInSeconds": ttl.to_string()});
        }
        let req = self.with_token(self.http.post(self.state_url()));
        let resp = req.json(&json!([entry])).send().await?;
        let status = resp.status();
        if status.is_success() {
            return Ok(());
        }
        Err(dapr_error(status, resp.text().await.unwrap_or_default()))
    }

    async fn delete_state(&self, key: &str) -> Result<()> {
        let url = format!("{}/{key}", self.state_url());
        let resp = self
            .with_token(self.http.delete(url))
            .query(&[("consistency", self.consistency.as_str())])
            .send()
            .await?;
        let status = resp.status();
        if status.is_success() || status == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        Err(dapr_error(status, resp.text().await.unwrap_or_default()))
    }

    /// The raw serialized items stored under the messages key.
    async fn read_messages(&self) -> Result<(Vec<String>, Option<String>)> {
        let state = self.get_state(&self.messages_key).await?;
        let Some(body) = state.body else {
            return Ok((Vec::new(), state.etag));
        };
        Ok((decode_message_array(&body), state.etag))
    }

    /// Run a conditional write, retrying while the sidecar reports an ETag conflict.
    async fn with_retries<T, F, Fut>(&self, mut attempt: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        for n in 0..MAX_WRITE_ATTEMPTS {
            match attempt().await {
                Ok(value) => return Ok(value),
                Err(e) if is_concurrency_conflict(&e) && n + 1 < MAX_WRITE_ATTEMPTS => {
                    tokio::time::sleep(retry_delay(n)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(AgentsError::other(
            "DaprSession exhausted its conditional-write attempts",
        ))
    }

    /// Refresh `created_at`/`updated_at`; failures are logged, never raised.
    async fn update_metadata(&self) {
        let now = now_seconds().to_string();
        let result = self
            .with_retries(|| async {
                let state = self.get_state(&self.metadata_key).await?;
                let created_at = state
                    .body
                    .as_deref()
                    .and_then(|b| serde_json::from_str::<Value>(b).ok())
                    .and_then(|v| {
                        v.get("created_at")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| now.clone());
                let value = json!({
                    "session_id": self.session_id,
                    "created_at": created_at,
                    "updated_at": now,
                });
                self.save_state(&self.metadata_key, value, state.etag.as_deref())
                    .await
            })
            .await;
        if let Err(e) = result {
            log::warn!(target: crate::logger::LOG_TARGET,
                "DaprSession stored the new items but could not update the session metadata for \
                 {}: {e}",
                self.session_id
            );
        }
    }
}

/// Seconds since the Unix epoch.
fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Backoff before retry number `attempt` (zero-based), capped at one second.
fn retry_delay(attempt: u32) -> Duration {
    RETRY_BASE_DELAY
        .saturating_mul(1u32 << attempt.min(8))
        .min(RETRY_MAX_DELAY)
}

fn dapr_error(status: reqwest::StatusCode, body: String) -> AgentsError {
    AgentsError::ApiError {
        status: status.as_u16(),
        message: if body.is_empty() {
            status.to_string()
        } else {
            body.clone()
        },
        body: serde_json::from_str(&body).ok().map(Box::new),
        request_id: None,
        headers: None,
    }
}

/// True when the sidecar reported a lost conditional write.
///
/// Dapr answers an ETag mismatch with 409, or with 500 and an `ERR_STATE_SAVE`/`ERR_STATE_DELETE`
/// message naming the mismatch, so both shapes are recognized.
fn is_concurrency_conflict(error: &AgentsError) -> bool {
    let AgentsError::ApiError {
        status, message, ..
    } = error
    else {
        return false;
    };
    if *status == 409 {
        return true;
    }
    let lowered = message.to_lowercase();
    [
        "etag mismatch",
        "etag does not match",
        "invalid etag",
        "possible etag mismatch",
        "failed to set key",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

/// Parse the stored messages value: a JSON array of serialized items.
fn decode_message_array(body: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return Vec::new();
    };
    // Some state stores hand the array back wrapped in a JSON string.
    let value = match value {
        Value::String(s) => serde_json::from_str::<Value>(&s).unwrap_or(Value::Null),
        other => other,
    };
    let Value::Array(entries) = value else {
        return Vec::new();
    };
    entries
        .into_iter()
        .map(|e| match e {
            Value::String(s) => s,
            other => other.to_string(),
        })
        .collect()
}

/// Decode serialized items, skipping corrupted entries.
fn decode_items(raw: &[String]) -> Vec<InputItem> {
    raw.iter()
        .filter_map(|r| serde_json::from_str(r).ok().map(InputItem))
        .collect()
}

#[async_trait]
impl Session for DaprSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        let _guard = self.lock.lock().await;
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let (raw, _) = self.read_messages().await?;
        let items = decode_items(&raw);
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let serialized: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i.as_value()))
            .collect::<std::result::Result<_, _>>()?;
        let _guard = self.lock.lock().await;
        self.with_retries(|| async {
            let (mut raw, etag) = self.read_messages().await?;
            raw.extend(serialized.iter().cloned());
            self.save_state(&self.messages_key, json!(raw), etag.as_deref())
                .await
        })
        .await?;
        self.update_metadata().await;
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let _guard = self.lock.lock().await;
        loop {
            let claimed = self
                .with_retries(|| async {
                    let (mut raw, etag) = self.read_messages().await?;
                    let Some(last) = raw.pop() else {
                        return Ok(None);
                    };
                    self.save_state(&self.messages_key, json!(raw), etag.as_deref())
                        .await?;
                    Ok(Some(last))
                })
                .await?;
            // A corrupted entry is dropped and the next-most-recent one is claimed, matching
            // the other backends.
            let Some(claimed) = claimed else {
                return Ok(None);
            };
            if let Ok(value) = serde_json::from_str(&claimed) {
                return Ok(Some(InputItem(value)));
            }
        }
    }

    async fn clear_session(&self) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.delete_state(&self.messages_key).await?;
        self.delete_state(&self.metadata_key).await
    }

    /// Snapshot the tail of the stored array; handles are the serialized items themselves.
    ///
    /// Dapr stores the whole history under one key, so there are no per-item ids. Using each
    /// item's exact serialized form as its handle lets the replacement verify that the tail is
    /// unchanged, and the ETag makes the read-modify-write itself atomic.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        let _guard = self.lock.lock().await;
        let (raw, _) = self.read_messages().await?;
        let start = raw.len().saturating_sub(limit);
        let window = &raw[start..];
        let mut items = Vec::new();
        let mut handles = Vec::new();
        let mut complete = start == 0;
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
                    complete = false;
                }
            }
        }
        Ok(Some(CompactionSnapshot {
            items,
            handles,
            complete,
        }))
    }

    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        let replacement: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i.as_value()))
            .collect::<std::result::Result<_, _>>()?;
        let _guard = self.lock.lock().await;
        self.with_retries(|| async {
            let (raw, etag) = self.read_messages().await?;
            if raw.len() < handles.len() {
                return Ok(false);
            }
            let start = raw.len() - handles.len();
            if raw[start..]
                .iter()
                .zip(handles)
                .any(|(entry, handle)| *entry != handle.0)
            {
                return Ok(false);
            }
            let mut next: Vec<String> = raw[..start].to_vec();
            next.extend(replacement.iter().cloned());
            self.save_state(&self.messages_key, json!(next), etag.as_deref())
                .await?;
            Ok(true)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_layout_matches_the_python_backend() {
        let (messages, metadata) = dapr_session_keys("user-123");
        assert_eq!(messages, "user-123:messages");
        assert_eq!(metadata, "user-123:metadata");
    }

    #[test]
    fn stored_arrays_decode_from_both_shapes() {
        let direct = r#"["{\"role\":\"user\"}","{\"role\":\"assistant\"}"]"#;
        assert_eq!(decode_message_array(direct).len(), 2);
        let wrapped = serde_json::to_string(direct).unwrap();
        assert_eq!(decode_message_array(&wrapped).len(), 2);
        assert!(decode_message_array("not json").is_empty());
    }

    #[test]
    fn etag_conflicts_are_recognized() {
        let conflict = AgentsError::ApiError {
            status: 500,
            message: "ERR_STATE_SAVE: possible etag mismatch".into(),
            body: None,
            request_id: None,
            headers: None,
        };
        assert!(is_concurrency_conflict(&conflict));
        let http_conflict = AgentsError::ApiError {
            status: 409,
            message: "conflict".into(),
            body: None,
            request_id: None,
            headers: None,
        };
        assert!(is_concurrency_conflict(&http_conflict));
        let other = AgentsError::ApiError {
            status: 500,
            message: "boom".into(),
            body: None,
            request_id: None,
            headers: None,
        };
        assert!(!is_concurrency_conflict(&other));
        assert!(!is_concurrency_conflict(&AgentsError::other("boom")));
    }

    #[test]
    fn retry_delay_grows_and_is_capped() {
        assert_eq!(retry_delay(0), Duration::from_millis(50));
        assert_eq!(retry_delay(1), Duration::from_millis(100));
        assert_eq!(retry_delay(10), RETRY_MAX_DELAY);
    }

    #[test]
    fn endpoint_comes_from_the_environment_or_the_default() {
        let s = DaprSession::new("s", "store", "http://host:3500/");
        assert_eq!(s.state_url(), "http://host:3500/v1.0/state/store");
    }
}
