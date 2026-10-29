//! Transparent encryption with TTL-based expiry for any [`Session`] (feature `encrypt`).
//! Mirrors `agents.extensions.memory.encrypt_session`.

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64_STANDARD;
use hkdf::Hkdf;
use rand::RngCore;
use serde_json::{Value, json};
use sha2::Sha256;

use super::fernet::Fernet;
use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Envelope version written by this implementation.
const ENVELOPE_VERSION: u64 = 1;
/// Key-derivation identifier written into every envelope.
const KEY_ID: &str = "hkdf-v1";
/// HKDF `info` string, identical to the Python SDK's.
const HKDF_INFO: &[u8] = b"agents.session-store.hkdf.v1";
/// Length of the AES-CBC initialization vector in a Fernet token.
const IV_LEN: usize = 16;

/// Seconds since the Unix epoch.
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Decode a base64 key the way Python's `base64.urlsafe_b64decode` does.
///
/// That function only *translates* `-` to `+` and `_` to `/` and then calls `b64decode` with
/// `validate=False`, so it still accepts standard-alphabet `+` and `/` and silently discards
/// bytes outside the alphabet, such as a trailing newline from an environment-mounted secret.
/// Padding is still required. Reproducing exactly this matters: a key from
/// `openssl rand -base64 32` usually contains `+` or `/`, and a stricter decoder would fall
/// back to the raw string bytes and derive a different key than Python.
fn decode_base64_key(encryption_key: &str) -> Option<Vec<u8>> {
    let normalized: String = encryption_key
        .chars()
        .map(|c| match c {
            '-' => '+',
            '_' => '/',
            other => other,
        })
        .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')
        .collect();
    B64_STANDARD.decode(normalized).ok()
}

/// Raw key material for HKDF: a base64 32-byte key, or the raw string bytes.
fn master_key_bytes(encryption_key: &str) -> Result<Vec<u8>> {
    if encryption_key.is_empty() {
        return Err(AgentsError::user(
            "encryption_key not set; required for EncryptedSession",
        ));
    }
    if let Some(decoded) = decode_base64_key(encryption_key)
        && decoded.len() == 32
    {
        return Ok(decoded);
    }
    Ok(encryption_key.as_bytes().to_vec())
}

/// Derive the 32-byte Fernet key for one session, exactly as the Python SDK does:
/// HKDF-SHA256 over the master key bytes with the session id as salt.
fn derive_key(master_key: &str, session_id: &str) -> Result<[u8; 32]> {
    let ikm = master_key_bytes(master_key)?;
    let hkdf = Hkdf::<Sha256>::new(Some(session_id.as_bytes()), &ikm);
    let mut okm = [0u8; 32];
    hkdf.expand(HKDF_INFO, &mut okm)
        .map_err(|e| AgentsError::other(format!("HKDF expand failed: {e}")))?;
    Ok(okm)
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
/// Items are stored as `{"__enc__": 1, "v": 1, "kid": "hkdf-v1", "payload": "<fernet token>"}`.
/// The payload is a [Fernet](https://github.com/fernet/spec) token: AES-128-CBC with PKCS7
/// padding, authenticated by HMAC-SHA256 over a 64-bit creation timestamp, the IV and the
/// ciphertext, so an item's age cannot be altered without breaking authentication. Only
/// authenticated, unexpired envelopes are returned as history; plaintext records and envelopes
/// that fail to decrypt are skipped, never migrated.
///
/// Expired records stay in the underlying store. Reading valid items may scan the whole
/// retained history unless [`EncryptedSession::with_max_scan_items`] bounds it.
///
/// # Wire compatibility with the Python SDK
///
/// The format is byte-compatible with `agents.extensions.memory.EncryptedSession`: the same
/// HKDF-SHA256 derivation (`salt = session_id`, `info = agents.session-store.hkdf.v1`), the
/// same 32-byte Fernet key, the same token layout and the same envelope. A history written by
/// the Python SDK decrypts here, and a history written here decrypts in Python, given the same
/// master key and session id.
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
/// Expiry is decided from the local system clock, and a token dated more than a minute in the
/// future is rejected, as in Python. Keep servers synchronized (NTP) so valid items are not
/// rejected.
pub struct EncryptedSession {
    session_id: String,
    underlying: Arc<dyn Session>,
    cipher: Fernet,
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
        let cipher = Fernet::from_key(&derive_key(encryption_key, &session_id)?);
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
    ///
    /// The plaintext is the item's compact JSON, matching Python's
    /// `json.dumps(..., separators=(",", ":"), ensure_ascii=False)`.
    fn wrap(&self, item: &InputItem) -> Result<InputItem> {
        let mut iv = [0u8; IV_LEN];
        rand::rng().fill_bytes(&mut iv);
        let token =
            self.cipher
                .encrypt_at_time(&serde_json::to_vec(item.as_value())?, now_seconds(), &iv);
        Ok(InputItem(json!({
            "__enc__": 1,
            "v": ENVELOPE_VERSION,
            "kid": KEY_ID,
            "payload": token,
        })))
    }

    /// Decrypt one envelope, returning `None` when it is not an envelope, cannot be
    /// authenticated, or has expired.
    fn unwrap_item(&self, item: &InputItem) -> Option<InputItem> {
        if !is_encrypted_envelope(item) {
            return None;
        }
        let payload = item.as_value().get("payload")?.as_str()?;
        let plaintext = self
            .cipher
            .decrypt_at_time(payload, Some(self.ttl), now_seconds())
            .ok()?;
        serde_json::from_slice(&plaintext).ok().map(InputItem)
    }

    /// The authenticated creation timestamp of an envelope this session's key can verify.
    ///
    /// Unlike history reads this ignores the TTL, so an authenticated but expired item still
    /// reports its age. Returns `None` for anything this key cannot authenticate.
    pub fn envelope_timestamp(&self, item: &InputItem) -> Option<u64> {
        if !is_encrypted_envelope(item) {
            return None;
        }
        self.cipher
            .extract_timestamp(item.as_value().get("payload")?.as_str()?)
            .ok()
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
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
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

    /// Forward the wrapped store's snapshot, decrypting its items.
    ///
    /// An authenticated envelope whose TTL has run out is simply skipped: it is already invisible
    /// as history and its absence says nothing about coverage. An envelope this key cannot
    /// authenticate at all is different — it is not necessarily expired, so it and everything
    /// before it are held outside the replaceable window and `complete` is cleared, because
    /// compaction must never summarize away an item it could not read.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        let Some(raw) = self.underlying.compaction_snapshot(limit).await? else {
            return Ok(None);
        };
        let now = now_seconds();
        let mut items = Vec::new();
        let mut handles = Vec::new();
        let mut complete = raw.complete;
        for (envelope, handle) in raw.items.iter().zip(raw.handles) {
            if let Some(item) = self.unwrap_item(envelope) {
                items.push(item);
                handles.push(handle);
                continue;
            }
            if self
                .envelope_timestamp(envelope)
                .is_some_and(|created_at| created_at.saturating_add(self.ttl) < now)
            {
                // Authenticated and expired: already not part of history.
                continue;
            }
            items.clear();
            handles.clear();
            complete = false;
        }
        Ok(Some(CompactionSnapshot {
            items,
            handles,
            complete,
        }))
    }

    /// Re-encrypt the compacted output before it reaches the wrapped store.
    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        self.underlying
            .replace_compaction_suffix(handles, self.encrypt_items(&items)?)
            .await
    }
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::URL_SAFE as B64_URL_TEST;

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

    /// Vectors produced by the Python SDK's `EncryptedSession` with `cryptography` 48:
    /// a raw-string master key and a url-safe base64 32-byte key.
    mod python_vectors {
        /// `_ensure_fernet_key_bytes` falls through to the raw UTF-8 bytes for this one.
        pub(super) const RAW_MASTER_KEY: &str = "0123456789abcdef0123456789abcdef";
        pub(super) const RAW_SESSION_ID: &str = "user-123";
        pub(super) const RAW_DERIVED_HEX: &str =
            "e5efe4b82120e80eb17b26a7dd555a5e873acd7e3b9daecf8e68363c7a8cf86e";
        pub(super) const RAW_PAYLOAD: &str = "gAAAAABlU_EAyicwKYzApg0fhqXu4Mf3MVoIR0uJ44-qrbeqamzFwpJDScnj0z_en8pDpJuSDaVsUBtisfKKy0CLq497sBTG0Y4xoA_8WDOETvmk20rM7SUC2AHvNPk5Z7cctoi6GNrROOABA0B_fT-L9HlY_1DTjA==";
        pub(super) const RAW_CONTENT: &str = "hello from python (raw)";

        /// A real Fernet key: 32 bytes, url-safe base64.
        pub(super) const B64_MASTER_KEY: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
        pub(super) const B64_SESSION_ID: &str = "session-abc";
        pub(super) const B64_DERIVED_HEX: &str =
            "cce8dccfa205a143c9624c94a0919405c1116b20e86193f7f55986e0884b1fdc";
        pub(super) const B64_PAYLOAD: &str = "gAAAAABlU_EAE0IJ3pmZUJU020fmRWDmhNlRBoBde_u-lfxbU6b9JWYNendpOLA6xkOJPeMUM8dROmqE4ck6fNN1mj1tpTbqTew2HbH0kOpdtnRIM9GmoooYmWKQENSav3LtuJThFae6vhfrl7Q0qUvkKOgI08ewdQ==";
        pub(super) const B64_CONTENT: &str = "hello from python (b64)";

        /// The `current_time` both tokens were created at.
        pub(super) const CREATED_AT: u64 = 1_700_000_000;

        /// The same 32 bytes spelled four ways, with the key Python derives for each under
        /// session id `user-123`. The first three must all decode to the same bytes; the
        /// unpadded spelling is a padding error in Python and takes the raw-bytes fallback.
        pub(super) const STANDARD_ALPHABET_KEY: &str =
            "+//+AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxw=";
        pub(super) const URL_SAFE_KEY: &str = "-__-AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxw=";
        pub(super) const URL_SAFE_KEY_WITH_NEWLINE: &str =
            "-__-AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxw=\n";
        pub(super) const UNPADDED_KEY: &str = "-__-AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxw";
        pub(super) const DECODED_DERIVED_HEX: &str =
            "c00aa4e8e97c4f384d9f450a1a29e32a68c37c75884c6be9bbf186d316498dd2";
        pub(super) const UNPADDED_DERIVED_HEX: &str =
            "6bb8b4f76ab91b5211d2a511e79760bbf348b9ba333d639ac842b9738476d802";
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// An envelope exactly as the Python SDK writes it.
    fn python_envelope(payload: &str) -> InputItem {
        InputItem(json!({"__enc__": 1, "v": 1, "kid": "hkdf-v1", "payload": payload}))
    }

    #[test]
    fn key_derivation_matches_the_python_sdk() {
        use python_vectors::*;
        assert_eq!(
            derive_key(RAW_MASTER_KEY, RAW_SESSION_ID).unwrap().to_vec(),
            hex_bytes(RAW_DERIVED_HEX)
        );
        assert_eq!(
            derive_key(B64_MASTER_KEY, B64_SESSION_ID).unwrap().to_vec(),
            hex_bytes(B64_DERIVED_HEX)
        );
        // The session id is the HKDF salt, so another session derives another key.
        assert_ne!(
            derive_key(RAW_MASTER_KEY, "other").unwrap().to_vec(),
            hex_bytes(RAW_DERIVED_HEX)
        );
    }

    #[test]
    fn key_spellings_derive_the_same_key_as_python() {
        use python_vectors::*;
        // Standard alphabet, url-safe alphabet and a trailing newline are all the same key in
        // Python, so they must all derive identically here.
        for key in [
            STANDARD_ALPHABET_KEY,
            URL_SAFE_KEY,
            URL_SAFE_KEY_WITH_NEWLINE,
        ] {
            assert_eq!(
                derive_key(key, RAW_SESSION_ID).unwrap().to_vec(),
                hex_bytes(DECODED_DERIVED_HEX),
                "{key:?}"
            );
        }
        // Dropping the padding makes Python treat the key as raw bytes, which derives a
        // different key; matching that exactly is what keeps both sides in agreement.
        assert_eq!(
            derive_key(UNPADDED_KEY, RAW_SESSION_ID).unwrap().to_vec(),
            hex_bytes(UNPADDED_DERIVED_HEX)
        );
        assert_ne!(UNPADDED_DERIVED_HEX, DECODED_DERIVED_HEX);
    }

    #[tokio::test]
    async fn reads_history_written_by_the_python_sdk() {
        use python_vectors::*;
        for (master, session_id, payload, content) in [
            (RAW_MASTER_KEY, RAW_SESSION_ID, RAW_PAYLOAD, RAW_CONTENT),
            (B64_MASTER_KEY, B64_SESSION_ID, B64_PAYLOAD, B64_CONTENT),
        ] {
            let inner = Arc::new(InMemorySession::new(session_id));
            inner
                .add_items(vec![python_envelope(payload)])
                .await
                .unwrap();
            // The tokens were minted in 2023, so the TTL must still cover them today.
            let ttl = now_seconds() - CREATED_AT + 600;
            let session = EncryptedSession::new(session_id, inner, master, ttl).unwrap();

            let items = session.get_items(None).await.unwrap();
            assert_eq!(items.len(), 1, "{session_id}");
            assert_eq!(items[0].role(), Some("user"));
            assert_eq!(items[0].text_content().as_deref(), Some(content));
            assert_eq!(
                session.envelope_timestamp(&python_envelope(payload)),
                Some(CREATED_AT)
            );
        }
    }

    #[tokio::test]
    async fn python_history_expires_on_the_same_boundary() {
        use python_vectors::*;
        let inner = Arc::new(InMemorySession::new(RAW_SESSION_ID));
        inner
            .add_items(vec![python_envelope(RAW_PAYLOAD)])
            .await
            .unwrap();
        // A TTL that ran out before now: Python skips these items, and so do we.
        let session = EncryptedSession::new(RAW_SESSION_ID, inner, RAW_MASTER_KEY, 600).unwrap();
        assert!(session.get_items(None).await.unwrap().is_empty());
        // The envelope is still authentic, just too old to be served as history.
        assert_eq!(
            session.envelope_timestamp(&python_envelope(RAW_PAYLOAD)),
            Some(CREATED_AT)
        );
    }

    #[test]
    fn tokens_this_session_writes_have_the_python_envelope_shape() {
        let session = session(600);
        let wrapped = session.wrap(&InputItem::user("round trip")).unwrap();
        let v = wrapped.as_value();
        assert_eq!(v["__enc__"], 1);
        assert_eq!(v["v"], 1);
        assert_eq!(v["kid"], "hkdf-v1");
        let payload = v["payload"].as_str().unwrap();
        // Version byte 0x80 plus a 2023+ timestamp always encodes as this prefix.
        assert!(payload.starts_with("gAAAAA"), "{payload}");
        // Url-safe base64 with padding, as Python's `urlsafe_b64encode` produces.
        assert!(
            payload
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_=".contains(c)),
            "{payload}"
        );
        assert!(payload.len().is_multiple_of(4), "{payload}");
        assert_eq!(
            session
                .unwrap_item(&wrapped)
                .unwrap()
                .text_content()
                .as_deref(),
            Some("round trip")
        );
    }

    #[test]
    fn base64_master_keys_are_accepted_as_leniently_as_python() {
        let raw = [7u8; 32];
        assert_eq!(
            master_key_bytes(&B64_URL_TEST.encode(raw)).unwrap(),
            raw.to_vec()
        );

        // A key from `openssl rand -base64 32` uses the standard alphabet; Python's
        // `urlsafe_b64decode` accepts `+` and `/`, so this must decode, not fall back.
        let mixed = [
            251u8, 255, 254, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28,
        ];
        let standard = B64_STANDARD.encode(mixed);
        assert!(
            standard.contains('+') && standard.contains('/'),
            "{standard}"
        );
        assert_eq!(master_key_bytes(&standard).unwrap(), mixed.to_vec());
        // The url-safe spelling of the same bytes derives the same key.
        assert_eq!(
            master_key_bytes(&B64_URL_TEST.encode(mixed)).unwrap(),
            mixed.to_vec()
        );

        // Non-alphabet bytes are discarded, as Python does for a newline-terminated secret.
        assert_eq!(
            master_key_bytes(&format!("{}\n", B64_URL_TEST.encode(mixed))).unwrap(),
            mixed.to_vec()
        );

        // Unpadded 43-character keys are a padding error in Python, so they fall back to the
        // raw string bytes.
        let unpadded = B64_URL_TEST.encode(mixed).trim_end_matches('=').to_string();
        assert_eq!(unpadded.len(), 43);
        assert_eq!(master_key_bytes(&unpadded).unwrap(), unpadded.into_bytes());

        // Anything that is not 32 bytes of base64 is used as raw key material.
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
