//! Transparent encryption with TTL-based expiry for any [`Session`] (feature `encrypt`).
//! Mirrors `agents.extensions.memory.encrypt_session`.

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::{STANDARD as B64, URL_SAFE as B64_URL};
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{Key, KeyInit, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use rand::RngCore;
use serde_json::{Value, json};
use sha2::Sha256;

use super::{Session, SessionSettings};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Envelope version written by this implementation.
const ENVELOPE_VERSION: u64 = 1;
/// Key-derivation identifier written into every envelope.
const KEY_ID: &str = "hkdf-v1";
/// HKDF `info` string, identical to the Python SDK's.
const HKDF_INFO: &[u8] = b"agents.session-store.hkdf.v1";
/// Length of an XChaCha20-Poly1305 nonce.
const NONCE_LEN: usize = 24;
/// Length of the authenticated timestamp prefix inside the plaintext.
const TIMESTAMP_LEN: usize = 8;

/// Seconds since the Unix epoch.
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Raw key material for HKDF: a url-safe base64 32-byte key, or the raw string bytes.
fn master_key_bytes(encryption_key: &str) -> Result<Vec<u8>> {
    if encryption_key.is_empty() {
        return Err(AgentsError::user(
            "encryption_key not set; required for EncryptedSession",
        ));
    }
    if let Ok(decoded) = B64_URL.decode(encryption_key)
        && decoded.len() == 32
    {
        return Ok(decoded);
    }
    Ok(encryption_key.as_bytes().to_vec())
}

/// True when `item` looks like an envelope written by this session.
fn is_encrypted_envelope(item: &InputItem) -> bool {
    let v = item.as_value();
    v.get("__enc__").and_then(Value::as_u64) == Some(1)
        && v.get("payload").and_then(Value::as_str).is_some()
        && v.get("kid").is_some()
        && v.get("v").is_some()
}

/// Encrypts the items of a wrapped session and skips expired or unreadable ones.
///
/// Items are stored as `{"__enc__": 1, "v": 1, "kid": "hkdf-v1", "payload": "<base64>"}`, where
/// the payload is `base64(nonce || XChaCha20-Poly1305(nonce, timestamp || json))`. The 8-byte
/// big-endian creation timestamp is part of the authenticated plaintext, so an item's age
/// cannot be altered without breaking authentication. Only authenticated, unexpired envelopes
/// are returned; plaintext records and envelopes that fail to decrypt are skipped, never
/// migrated.
///
/// Expired records stay in the underlying store. Reading valid items may scan the whole
/// retained history unless [`EncryptedSession::with_max_scan_items`] bounds it.
///
/// # Wire format
///
/// The envelope shape matches the Python SDK, but the payload does **not**: Python uses
/// Fernet (AES-128-CBC + HMAC-SHA256), which has no Rust equivalent in this crate's
/// dependencies. Histories written by the Python `EncryptedSession` cannot be read here and
/// vice versa.
///
/// # Composition with compaction
///
/// Wrap this session *inside* [`OpenAIResponsesCompactionSession`]
/// (`Compaction(Encrypted(store))`), never the other way round: a compaction session below
/// the encryption layer would send ciphertext envelopes to the compaction API. This session
/// does not forward [`Session::on_run_completed`] to the wrapped session.
///
/// [`OpenAIResponsesCompactionSession`]: super::OpenAIResponsesCompactionSession
///
/// # Clock drift
///
/// Expiry is decided from the local system clock. Keep servers synchronized (NTP) so valid
/// items are not rejected.
pub struct EncryptedSession {
    session_id: String,
    underlying: Arc<dyn Session>,
    cipher: XChaCha20Poly1305,
    ttl: u64,
    max_scan_items: Option<usize>,
}

impl fmt::Debug for EncryptedSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptedSession")
            .field("session_id", &self.session_id)
            .field("underlying", &self.underlying)
            .field("ttl", &self.ttl)
            .field("max_scan_items", &self.max_scan_items)
            .finish_non_exhaustive()
    }
}

impl EncryptedSession {
    /// Wrap `underlying`, deriving a per-session key from `encryption_key` and `session_id`.
    ///
    /// `encryption_key` must be a high-entropy secret (for example 32 random bytes encoded as
    /// url-safe base64). HKDF does not harden passwords and the session id salt adds no secret
    /// entropy, so low-entropy secrets are unsuitable. `ttl` is the item lifetime in seconds.
    pub fn new(
        session_id: impl Into<String>,
        underlying: Arc<dyn Session>,
        encryption_key: &str,
        ttl: u64,
    ) -> Result<Self> {
        let session_id = session_id.into();
        let ikm = master_key_bytes(encryption_key)?;
        let hkdf = Hkdf::<Sha256>::new(Some(session_id.as_bytes()), &ikm);
        let mut okm = [0u8; 32];
        hkdf.expand(HKDF_INFO, &mut okm)
            .map_err(|e| AgentsError::other(format!("HKDF expand failed: {e}")))?;
        let cipher = XChaCha20Poly1305::new(&Key::from(okm));
        Ok(EncryptedSession {
            session_id,
            underlying,
            cipher,
            ttl,
            max_scan_items: None,
        })
    }

    /// Bound the number of stored items read and unwrapped per `get_items` call.
    ///
    /// `get_items` fails instead of returning a partial history when the budget is exhausted
    /// before a complete result is established.
    pub fn with_max_scan_items(mut self, max_scan_items: usize) -> Result<Self> {
        if max_scan_items == 0 {
            return Err(AgentsError::user("max_scan_items must be positive"));
        }
        self.max_scan_items = Some(max_scan_items);
        Ok(self)
    }

    /// The wrapped session.
    pub fn underlying_session(&self) -> &Arc<dyn Session> {
        &self.underlying
    }

    /// The item lifetime in seconds.
    pub fn ttl(&self) -> u64 {
        self.ttl
    }

    /// Encrypt one item into an envelope.
    fn wrap(&self, item: &InputItem) -> Result<InputItem> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::rng().fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from(nonce_bytes);
        let mut plaintext = Vec::new();
        plaintext.extend_from_slice(&now_seconds().to_be_bytes());
        plaintext.extend_from_slice(&serde_json::to_vec(item.as_value())?);
        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext.as_slice())
            .map_err(|e| AgentsError::other(format!("Session encryption failed: {e}")))?;
        let mut payload = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        payload.extend_from_slice(&nonce_bytes);
        payload.extend_from_slice(&ciphertext);
        Ok(InputItem(json!({
            "__enc__": 1,
            "v": ENVELOPE_VERSION,
            "kid": KEY_ID,
            "payload": B64.encode(payload),
        })))
    }

    /// Decrypt one envelope, returning `None` when it is not an envelope, cannot be
    /// authenticated, or has expired.
    fn unwrap_item(&self, item: &InputItem) -> Option<InputItem> {
        if !is_encrypted_envelope(item) {
            return None;
        }
        let payload = B64.decode(item.as_value().get("payload")?.as_str()?).ok()?;
        if payload.len() <= NONCE_LEN {
            return None;
        }
        let (nonce_bytes, ciphertext) = payload.split_at(NONCE_LEN);
        let nonce = XNonce::try_from(nonce_bytes).ok()?;
        let plaintext = self.cipher.decrypt(&nonce, ciphertext).ok()?;
        if plaintext.len() < TIMESTAMP_LEN {
            return None;
        }
        let (ts_bytes, body) = plaintext.split_at(TIMESTAMP_LEN);
        let created_at = u64::from_be_bytes(ts_bytes.try_into().ok()?);
        if created_at.saturating_add(self.ttl) < now_seconds() {
            return None;
        }
        serde_json::from_slice(body).ok().map(InputItem)
    }

    /// Decrypt every readable envelope, skipping the rest.
    fn unwrap_valid_items(&self, items: &[InputItem]) -> Vec<InputItem> {
        items.iter().filter_map(|i| self.unwrap_item(i)).collect()
    }

    /// Encrypt a batch of items.
    pub fn encrypt_items(&self, items: &[InputItem]) -> Result<Vec<InputItem>> {
        items.iter().map(|i| self.wrap(i)).collect()
    }
}

#[async_trait]
impl Session for EncryptedSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.underlying.session_settings()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let budget = self.max_scan_items;
        match limit {
            Some(0) => return Ok(Vec::new()),
            None if budget.is_none() => {
                let stored = self.underlying.get_items(None).await?;
                return Ok(self.unwrap_valid_items(&stored));
            }
            _ => {}
        }
        // Expired or unreadable records do not count towards `limit`, so widen the read
        // window until enough valid items are found or the store is exhausted.
        let mut remaining = budget;
        let mut window = match (limit, budget) {
            (Some(n), Some(b)) => n.min(b),
            (Some(n), None) => n,
            (None, Some(b)) => b,
            (None, None) => unreachable!("handled above"),
        };
        loop {
            let stored = self.underlying.get_items(Some(window)).await?;
            let valid = self.unwrap_valid_items(&stored);
            if let Some(n) = limit
                && valid.len() >= n
            {
                return Ok(valid[valid.len() - n..].to_vec());
            }
            if stored.len() < window {
                return Ok(valid);
            }
            let mut next = window.saturating_mul(2);
            if let Some(left) = remaining.as_mut() {
                *left = left.saturating_sub(stored.len());
                next = next.min(*left);
                if next <= window {
                    return Err(AgentsError::other(
                        "EncryptedSession max_scan_items exhausted before a complete history \
                         result could be established; increase max_scan_items or reduce retained \
                         history",
                    ));
                }
            }
            window = next;
        }
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.underlying.add_items(self.encrypt_items(&items)?).await
    }

    /// Remove and return the most recent readable item.
    ///
    /// Unreadable or expired envelopes encountered on the way are removed from the underlying
    /// store, unlike the Python SDK's native SQLite path, which authenticates inside the
    /// transaction. Use the correct key before popping.
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        loop {
            let Some(stored) = self.underlying.pop_item().await? else {
                return Ok(None);
            };
            if let Some(item) = self.unwrap_item(&stored) {
                return Ok(Some(item));
            }
        }
    }

    async fn clear_session(&self) -> Result<()> {
        self.underlying.clear_session().await
    }
}

#[cfg(test)]
mod tests {
    use super::super::InMemorySession;
    use super::*;

    fn session(ttl: u64) -> EncryptedSession {
        EncryptedSession::new(
            "s1",
            Arc::new(InMemorySession::new("s1")),
            "0123456789abcdef0123456789abcdef",
            ttl,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn round_trips_items_and_stores_ciphertext() {
        let s = session(600);
        s.add_items(vec![InputItem::user("secret"), InputItem::assistant("ok")])
            .await
            .unwrap();
        let stored = s.underlying.get_items(None).await.unwrap();
        assert!(is_encrypted_envelope(&stored[0]));
        assert!(!stored[0].as_value().to_string().contains("secret"));
        let items = s.get_items(None).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text_content().as_deref(), Some("secret"));
        assert_eq!(
            s.pop_item()
                .await
                .unwrap()
                .unwrap()
                .text_content()
                .as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn base64_master_keys_are_accepted() {
        let raw = [7u8; 32];
        let key = B64_URL.encode(raw);
        assert_eq!(master_key_bytes(&key).unwrap(), raw.to_vec());
        assert_eq!(master_key_bytes("plain").unwrap(), b"plain".to_vec());
        assert!(master_key_bytes("").is_err());
    }

    #[tokio::test]
    async fn plaintext_records_are_skipped() {
        let s = session(600);
        s.underlying
            .add_items(vec![InputItem::user("legacy")])
            .await
            .unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn max_scan_items_bounds_the_read() {
        let inner = Arc::new(InMemorySession::new("s1"));
        let s = EncryptedSession::new("s1", inner.clone(), "key-material-key-material-32byte", 600)
            .unwrap()
            .with_max_scan_items(2)
            .unwrap();
        s.add_items(vec![
            InputItem::user("a"),
            InputItem::user("b"),
            InputItem::user("c"),
        ])
        .await
        .unwrap();
        assert_eq!(s.get_items(Some(1)).await.unwrap().len(), 1);
        assert!(s.get_items(None).await.is_err());
    }
}
