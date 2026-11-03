//! Fernet symmetric tokens, as used by Python's `cryptography` package (feature `encrypt`).
//!
//! A token is `urlsafe_base64(0x80 ‖ timestamp ‖ IV ‖ ciphertext ‖ HMAC)`:
//!
//! | field | size | meaning |
//! | --- | --- | --- |
//! | version | 1 byte | always `0x80` |
//! | timestamp | 8 bytes | seconds since the Unix epoch, big endian |
//! | IV | 16 bytes | AES-CBC initialization vector |
//! | ciphertext | multiple of 16 | AES-128-CBC of the PKCS7-padded plaintext |
//! | HMAC | 32 bytes | HMAC-SHA256 over everything before it |
//!
//! The 32-byte key is split into a 16-byte signing key (HMAC) followed by a 16-byte
//! encryption key (AES-128), matching the [Fernet spec](https://github.com/fernet/spec).

use aes::Aes128;
use aes::cipher::block_padding::Pkcs7;
use aes::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE as B64_URL;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

/// The only Fernet token version.
pub(super) const VERSION: u8 = 0x80;
/// How far a token's timestamp may lie in the future before it is rejected, as in Python.
pub(super) const MAX_CLOCK_SKEW: u64 = 60;

const TIMESTAMP_LEN: usize = 8;
const IV_LEN: usize = 16;
const HMAC_LEN: usize = 32;
const BLOCK_LEN: usize = 16;
/// Version, timestamp, IV and HMAC, i.e. everything but the ciphertext.
const OVERHEAD: usize = 1 + TIMESTAMP_LEN + IV_LEN + HMAC_LEN;

type Aes128CbcEnc = cbc::Encryptor<Aes128>;
type Aes128CbcDec = cbc::Decryptor<Aes128>;

/// Why a token could not be read.
///
/// Python raises a single `InvalidToken` for all of these; they are distinguished here only to
/// make failures debuggable. Callers treat every variant the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FernetError {
    /// The token is not valid url-safe base64, or is too short or misaligned.
    Malformed,
    /// The version byte is not `0x80`.
    UnsupportedVersion,
    /// The HMAC does not match the token body.
    InvalidSignature,
    /// The token is older than the requested TTL.
    Expired,
    /// The token's timestamp is further in the future than the allowed clock skew.
    TooFarInFuture,
    /// The ciphertext did not decrypt to validly padded plaintext.
    InvalidPadding,
}

/// The authenticated parts of a parsed token.
#[derive(Debug)]
struct Parts {
    timestamp: u64,
    iv: [u8; IV_LEN],
    ciphertext_range: std::ops::Range<usize>,
}

/// A Fernet key: a signing key and an encryption key derived from 32 bytes.
#[derive(Clone)]
pub(super) struct Fernet {
    signing_key: [u8; 16],
    encryption_key: [u8; 16],
}

impl std::fmt::Debug for Fernet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fernet").finish_non_exhaustive()
    }
}

impl Fernet {
    /// Split 32 raw key bytes into the signing and encryption halves.
    pub(super) fn from_key(key: &[u8; 32]) -> Self {
        let mut signing_key = [0u8; 16];
        let mut encryption_key = [0u8; 16];
        signing_key.copy_from_slice(&key[..16]);
        encryption_key.copy_from_slice(&key[16..]);
        Fernet {
            signing_key,
            encryption_key,
        }
    }

    /// Build a token for `plaintext` with an explicit timestamp and IV.
    pub(super) fn encrypt_at_time(
        &self,
        plaintext: &[u8],
        timestamp: u64,
        iv: &[u8; IV_LEN],
    ) -> String {
        let ciphertext = Aes128CbcEnc::new(&self.encryption_key.into(), iv.into())
            .encrypt_padded_vec::<Pkcs7>(plaintext);

        let mut token = Vec::with_capacity(OVERHEAD + ciphertext.len());
        token.push(VERSION);
        token.extend_from_slice(&timestamp.to_be_bytes());
        token.extend_from_slice(iv);
        token.extend_from_slice(&ciphertext);

        let mut mac = self.mac();
        mac.update(&token);
        token.extend_from_slice(&mac.finalize().into_bytes());
        B64_URL.encode(token)
    }

    /// Decrypt `token`, rejecting it when it is older than `ttl` seconds at `now`.
    pub(super) fn decrypt_at_time(
        &self,
        token: &str,
        ttl: Option<u64>,
        now: u64,
    ) -> Result<Vec<u8>, FernetError> {
        let data = decode(token)?;
        let parts = parse(&data)?;
        // Python checks the age before the signature, so a stale token reports expiry even
        // when it was written with another key.
        if let Some(ttl) = ttl {
            if parts.timestamp.saturating_add(ttl) < now {
                return Err(FernetError::Expired);
            }
            if now.saturating_add(MAX_CLOCK_SKEW) < parts.timestamp {
                return Err(FernetError::TooFarInFuture);
            }
        }
        self.verify(&data)?;
        Aes128CbcDec::new(&self.encryption_key.into(), (&parts.iv).into())
            .decrypt_padded_vec::<Pkcs7>(&data[parts.ciphertext_range])
            .map_err(|_| FernetError::InvalidPadding)
    }

    /// The token's creation timestamp, after verifying its HMAC.
    pub(super) fn extract_timestamp(&self, token: &str) -> Result<u64, FernetError> {
        let data = decode(token)?;
        let parts = parse(&data)?;
        self.verify(&data)?;
        Ok(parts.timestamp)
    }

    fn mac(&self) -> Hmac<Sha256> {
        Hmac::<Sha256>::new_from_slice(&self.signing_key).expect("HMAC accepts any key length")
    }

    /// Verify the trailing HMAC in constant time.
    fn verify(&self, data: &[u8]) -> Result<(), FernetError> {
        let (body, tag) = data.split_at(data.len() - HMAC_LEN);
        let mut mac = self.mac();
        mac.update(body);
        mac.verify_slice(tag)
            .map_err(|_| FernetError::InvalidSignature)
    }
}

/// Decode the url-safe base64 token body. Padding is required, as in Python.
fn decode(token: &str) -> Result<Vec<u8>, FernetError> {
    B64_URL.decode(token).map_err(|_| FernetError::Malformed)
}

/// Validate the token layout and read its unauthenticated header fields.
fn parse(data: &[u8]) -> Result<Parts, FernetError> {
    if data.len() <= OVERHEAD {
        return Err(FernetError::Malformed);
    }
    let ciphertext_len = data.len() - OVERHEAD;
    if !ciphertext_len.is_multiple_of(BLOCK_LEN) {
        return Err(FernetError::Malformed);
    }
    if data[0] != VERSION {
        return Err(FernetError::UnsupportedVersion);
    }
    let mut timestamp_bytes = [0u8; TIMESTAMP_LEN];
    timestamp_bytes.copy_from_slice(&data[1..1 + TIMESTAMP_LEN]);
    let mut iv = [0u8; IV_LEN];
    iv.copy_from_slice(&data[1 + TIMESTAMP_LEN..1 + TIMESTAMP_LEN + IV_LEN]);
    let start = 1 + TIMESTAMP_LEN + IV_LEN;
    Ok(Parts {
        timestamp: u64::from_be_bytes(timestamp_bytes),
        iv,
        ciphertext_range: start..start + ciphertext_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The official Fernet spec vector from <https://github.com/fernet/spec> (`verify.json`).
    const SPEC_SECRET: &str = "cw_0x689RpI-jtRR7oE8h_eQsKImvJapLeSbXpwF4e4=";
    const SPEC_TOKEN: &str = "gAAAAAAdwJ6wAAECAwQFBgcICQoLDA0ODy021cpGVWKZ_eEwCGM4BLLF_5CV9dOPmrhuVUPgJobwOz7JcbmrR64jVmpU4IwqDA==";
    /// `1985-10-26T01:20:01-07:00`, the `now` of the spec vector.
    const SPEC_NOW: u64 = 499_162_801;
    /// The timestamp inside the spec token, one second before `SPEC_NOW`.
    const SPEC_TIMESTAMP: u64 = 499_162_800;

    fn spec_key() -> Fernet {
        let bytes = B64_URL.decode(SPEC_SECRET).unwrap();
        Fernet::from_key(&bytes.try_into().unwrap())
    }

    #[test]
    fn decrypts_the_official_spec_vector() {
        let fernet = spec_key();
        let plaintext = fernet
            .decrypt_at_time(SPEC_TOKEN, Some(60), SPEC_NOW)
            .unwrap();
        assert_eq!(plaintext, b"hello");
        assert_eq!(
            fernet.extract_timestamp(SPEC_TOKEN).unwrap(),
            SPEC_TIMESTAMP
        );
    }

    #[test]
    fn reproduces_the_official_spec_vector_byte_for_byte() {
        // The spec vector fixes the IV, so the whole token must match.
        let iv: [u8; 16] = std::array::from_fn(|i| i as u8);
        let token = spec_key().encrypt_at_time(b"hello", SPEC_TIMESTAMP, &iv);
        assert_eq!(token, SPEC_TOKEN);
    }

    #[test]
    fn rejects_expired_and_future_tokens() {
        let fernet = spec_key();
        assert_eq!(
            fernet.decrypt_at_time(SPEC_TOKEN, Some(60), SPEC_TIMESTAMP + 61),
            Err(FernetError::Expired)
        );
        // Exactly at the TTL boundary the token is still valid.
        assert!(
            fernet
                .decrypt_at_time(SPEC_TOKEN, Some(60), SPEC_TIMESTAMP + 60)
                .is_ok()
        );
        assert_eq!(
            fernet.decrypt_at_time(SPEC_TOKEN, Some(60), SPEC_TIMESTAMP - MAX_CLOCK_SKEW - 1),
            Err(FernetError::TooFarInFuture)
        );
        // Without a TTL, age is not checked at all.
        assert!(fernet.decrypt_at_time(SPEC_TOKEN, None, 0).is_ok());
    }

    #[test]
    fn rejects_tampered_and_malformed_tokens() {
        let fernet = spec_key();
        let mut data = B64_URL.decode(SPEC_TOKEN).unwrap();
        let last = data.len() - 1;
        data[last] ^= 0x01;
        assert_eq!(
            fernet.decrypt_at_time(&B64_URL.encode(&data), None, SPEC_NOW),
            Err(FernetError::InvalidSignature)
        );

        let mut wrong_version = B64_URL.decode(SPEC_TOKEN).unwrap();
        wrong_version[0] = 0x81;
        assert_eq!(
            fernet.decrypt_at_time(&B64_URL.encode(&wrong_version), None, SPEC_NOW),
            Err(FernetError::UnsupportedVersion)
        );

        assert_eq!(
            fernet.decrypt_at_time("not base64!", None, SPEC_NOW),
            Err(FernetError::Malformed)
        );
        assert_eq!(
            fernet.decrypt_at_time(&B64_URL.encode([VERSION; 8]), None, SPEC_NOW),
            Err(FernetError::Malformed)
        );
        // A correctly framed token whose ciphertext length is not a multiple of the block size.
        assert_eq!(
            fernet.decrypt_at_time(&B64_URL.encode([VERSION; OVERHEAD + 8]), None, SPEC_NOW),
            Err(FernetError::Malformed)
        );
        // Another key cannot read the token.
        let other = Fernet::from_key(&[7u8; 32]);
        assert_eq!(
            other.decrypt_at_time(SPEC_TOKEN, None, SPEC_NOW),
            Err(FernetError::InvalidSignature)
        );
        assert_eq!(
            other.extract_timestamp(SPEC_TOKEN),
            Err(FernetError::InvalidSignature)
        );
    }

    /// Every case from the spec's `invalid.json`, under the same secret and TTL.
    #[test]
    fn rejects_the_official_invalid_vectors() {
        let fernet = spec_key();
        // (description, token, now, expected error)
        let cases: [(&str, &str, u64, FernetError); 8] = [
            (
                "incorrect mac",
                "gAAAAAAdwJ6xAAECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPAl1-szkFVzXTuGb4hR8AKtwcaX1YdykQUFBQUFBQUFBQQ==",
                SPEC_NOW,
                FernetError::InvalidSignature,
            ),
            (
                "too short",
                "gAAAAAAdwJ6xAAECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPA==",
                SPEC_NOW,
                FernetError::Malformed,
            ),
            (
                "invalid base64",
                "%%%%%%%%%%%%%AECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPAl1-szkFVzXTuGb4hR8AKtwcaX1YdykRtfsH-p1YsUD2Q==",
                SPEC_NOW,
                FernetError::Malformed,
            ),
            (
                "payload size not multiple of block size",
                "gAAAAAAdwJ6xAAECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPOm73QeoCk9uGib28Xe5vz6oxq5nmxbx_v7mrfyudzUm",
                SPEC_NOW,
                FernetError::Malformed,
            ),
            (
                "payload padding error",
                "gAAAAAAdwJ6xAAECAwQFBgcICQoLDA0ODz4LEpdELGQAad7aNEHbf-JkLPIpuiYRLQ3RtXatOYREu2FWke6CnJNYIbkuKNqOhw==",
                SPEC_NOW,
                FernetError::InvalidPadding,
            ),
            (
                "far-future TS (unacceptable clock skew)",
                "gAAAAAAdwStRAAECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPAnja1xKYyhd-Y6mSkTOyTGJmw2Xc2a6kBd-iX9b_qXQcw==",
                SPEC_NOW,
                FernetError::TooFarInFuture,
            ),
            (
                // `1985-10-26T01:21:31-07:00`, 90 seconds after the token's timestamp.
                "expired TTL",
                "gAAAAAAdwJ6xAAECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPAl1-szkFVzXTuGb4hR8AKtwcaX1YdykRtfsH-p1YsUD2Q==",
                499_162_891,
                FernetError::Expired,
            ),
            (
                "incorrect IV (causes padding error)",
                "gAAAAAAdwJ6xBQECAwQFBgcICQoLDA0OD3HkMATM5lFqGaerZ-fWPAkLhFLHpGtDBRLRTZeUfWgHSv49TF2AUEZ1TIvcZjK1zQ==",
                SPEC_NOW,
                FernetError::InvalidPadding,
            ),
        ];
        for (desc, token, now, expected) in cases {
            assert_eq!(
                fernet.decrypt_at_time(token, Some(60), now),
                Err(expected),
                "{desc}"
            );
        }
    }

    #[test]
    fn round_trips_payloads_across_block_boundaries() {
        let fernet = Fernet::from_key(&[3u8; 32]);
        for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 100] {
            let plaintext = vec![0xABu8; len];
            let token = fernet.encrypt_at_time(&plaintext, 1_700_000_000, &[9u8; 16]);
            assert_eq!(
                fernet
                    .decrypt_at_time(&token, Some(600), 1_700_000_000)
                    .unwrap(),
                plaintext,
                "length {len}"
            );
        }
    }
}
