//! Per-provider request authentication for endpoints that do not take a plain bearer token.
//!
//! any-llm performs this inside each provider class; here it is a
//! [`RequestAuth`] hook on the shared HTTP client, so the ordinary
//! Chat Completions / Responses models reach AWS Bedrock, SageMaker, Google Vertex AI and
//! IBM watsonx.ai without a second model implementation.
//!
//! Requires the `provider-auth` cargo feature.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL};
use hmac::{Hmac, KeyInit, Mac};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{AgentsError, Result};
use crate::models::RequestAuth;

type HmacSha256 = Hmac<Sha256>;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_first(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| env(k))
}

fn set_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<()> {
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|e| AgentsError::user(format!("Invalid header name {name}: {e}")))?;
    let value = HeaderValue::from_str(value)
        .map_err(|e| AgentsError::user(format!("Invalid header value: {e}")))?;
    headers.insert(name, value);
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

/// POST an `application/x-www-form-urlencoded` body (OAuth token requests).
async fn post_form(
    http: &reqwest::Client,
    url: &str,
    fields: &[(&str, &str)],
) -> Result<reqwest::Response> {
    let body = fields
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");
    Ok(http
        .post(url)
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body)
        .send()
        .await?)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

/// `YYYYMMDDThhmmssZ` and `YYYYMMDD` for a unix timestamp.
fn amz_date(epoch_secs: u64) -> (String, String) {
    let days = epoch_secs / 86_400;
    let secs = epoch_secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm), valid for the whole proleptic calendar.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let date = format!("{y:04}{m:02}{d:02}");
    let stamp = format!(
        "{date}T{:02}{:02}{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    );
    (stamp, date)
}

/// Percent-encode a URI path segment or a query component per SigV4's rules.
fn uri_encode(value: &str, encode_slash: bool) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b'/' if !encode_slash => "/".to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Split an absolute URL into (host, path, raw query).
fn split_url(url: &str) -> Result<(String, String, String)> {
    let rest = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .ok_or_else(|| AgentsError::user(format!("Not an absolute URL: {url}")))?;
    let (authority, after) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let (path, query) = match after.split_once('?') {
        Some((p, q)) => (p, q),
        None => (after, ""),
    };
    let path = if path.is_empty() { "/" } else { path };
    Ok((host.to_string(), path.to_string(), query.to_string()))
}

/// AWS credentials used by [`AwsSigV4Auth`].
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    /// `AWS_ACCESS_KEY_ID`.
    pub access_key_id: String,
    /// `AWS_SECRET_ACCESS_KEY`.
    pub secret_access_key: String,
    /// `AWS_SESSION_TOKEN`, for temporary credentials.
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// Read credentials from the standard environment variables.
    pub fn from_env() -> Option<Self> {
        Some(AwsCredentials {
            access_key_id: env("AWS_ACCESS_KEY_ID")?,
            secret_access_key: env("AWS_SECRET_ACCESS_KEY")?,
            session_token: env("AWS_SESSION_TOKEN"),
        })
    }
}

/// Signs requests with AWS Signature Version 4.
///
/// Hand-written against the published algorithm rather than pulling in the AWS SDK: the SDK's
/// signer would bring a large dependency tree for one header, and the crate already depends on
/// `sha2` and `hmac`.
#[derive(Debug, Clone)]
pub struct AwsSigV4Auth {
    credentials: AwsCredentials,
    region: String,
    service: String,
}

impl AwsSigV4Auth {
    /// Sign for `service` in `region`.
    pub fn new(
        credentials: AwsCredentials,
        region: impl Into<String>,
        service: impl Into<String>,
    ) -> Self {
        AwsSigV4Auth {
            credentials,
            region: region.into(),
            service: service.into(),
        }
    }

    /// Build the `Authorization` header value for one request.
    ///
    /// `headers` must already contain every header to be signed except `host` and
    /// `x-amz-date`, which this adds.
    pub fn authorization(
        &self,
        method: &str,
        url: &str,
        body: &[u8],
        extra_signed: &BTreeMap<String, String>,
        epoch_secs: u64,
    ) -> Result<(String, String, String)> {
        let (host, path, query) = split_url(url)?;
        let (stamp, date) = amz_date(epoch_secs);
        let payload_hash = sha256_hex(body);

        let mut signed: BTreeMap<String, String> = extra_signed
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        signed.insert("host".into(), host.clone());
        signed.insert("x-amz-date".into(), stamp.clone());
        if let Some(token) = &self.credentials.session_token {
            signed.insert("x-amz-security-token".into(), token.clone());
        }

        let canonical_path = uri_encode(&path, false);
        let mut params: Vec<(String, String)> = query
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (k, v) = p.split_once('=').unwrap_or((p, ""));
                (k.to_string(), v.to_string())
            })
            .collect();
        params.sort();
        let canonical_query = params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let canonical_headers = signed
            .iter()
            .map(|(k, v)| format!("{k}:{v}\n"))
            .collect::<String>();
        let signed_headers = signed.keys().cloned().collect::<Vec<_>>().join(";");
        let canonical_request = format!(
            "{}\n{canonical_path}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
            method.to_ascii_uppercase()
        );

        let scope = format!("{date}/{}/{}/aws4_request", self.region, self.service);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{}",
            sha256_hex(canonical_request.as_bytes())
        );

        let mut key = hmac(
            format!("AWS4{}", self.credentials.secret_access_key).as_bytes(),
            &date,
        );
        key = hmac(&key, &self.region);
        key = hmac(&key, &self.service);
        key = hmac(&key, "aws4_request");
        let signature = hex(&hmac(&key, &string_to_sign));

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, \
             Signature={signature}",
            self.credentials.access_key_id
        );
        Ok((authorization, stamp, payload_hash))
    }
}

#[async_trait]
impl RequestAuth for AwsSigV4Auth {
    async fn authorize(
        &self,
        method: &str,
        url: &str,
        body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        let body = body.unwrap_or_default();
        // Content-Type is part of the signature whenever it is sent.
        let mut extra = BTreeMap::new();
        if let Some(ct) = headers.get(reqwest::header::CONTENT_TYPE)
            && let Ok(ct) = ct.to_str()
        {
            extra.insert("content-type".to_string(), ct.to_string());
        }
        let (authorization, stamp, payload_hash) =
            self.authorization(method, url, body, &extra, now_secs())?;
        set_header(headers, "x-amz-date", &stamp)?;
        set_header(headers, "x-amz-content-sha256", &payload_hash)?;
        if let Some(token) = &self.credentials.session_token {
            set_header(headers, "x-amz-security-token", token)?;
        }
        set_header(headers, "authorization", &authorization)?;
        Ok(())
    }
}

/// A bearer token with an expiry, refreshed on demand.
#[derive(Debug, Default)]
struct CachedToken {
    token: Option<String>,
    expires_at: u64,
}

impl CachedToken {
    fn get(&self) -> Option<&str> {
        // Refresh a minute early so a token never expires mid-flight.
        self.token
            .as_deref()
            .filter(|_| self.expires_at > now_secs() + 60)
    }
}

/// How a Google OAuth access token is obtained.
#[derive(Debug, Clone)]
pub enum GoogleCredentials {
    /// Use this access token verbatim.
    AccessToken(String),
    /// A service-account JSON key: sign a JWT assertion and exchange it for an access token.
    ServiceAccount(Box<Value>),
    /// An `authorized_user` JSON (`gcloud auth application-default login`): refresh-token grant.
    AuthorizedUser(Box<Value>),
    /// The GCE/Cloud Run metadata server.
    MetadataServer,
}

impl GoogleCredentials {
    /// Application Default Credentials, in Google's documented order: an explicit access token,
    /// `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud ADC file, then the metadata server.
    pub fn from_env() -> Result<Self> {
        if let Some(token) = env_first(&["GOOGLE_OAUTH_ACCESS_TOKEN", "GOOGLE_ACCESS_TOKEN"]) {
            return Ok(GoogleCredentials::AccessToken(token));
        }
        if let Some(json) = env("GOOGLE_SERVICE_ACCOUNT_JSON") {
            return Self::from_json_str(&json);
        }
        if let Some(path) = env("GOOGLE_APPLICATION_CREDENTIALS") {
            let text = std::fs::read_to_string(&path).map_err(|e| {
                AgentsError::user(format!(
                    "Cannot read GOOGLE_APPLICATION_CREDENTIALS {path}: {e}"
                ))
            })?;
            return Self::from_json_str(&text);
        }
        if let Some(home) = env("HOME") {
            let path = format!("{home}/.config/gcloud/application_default_credentials.json");
            if let Ok(text) = std::fs::read_to_string(&path) {
                return Self::from_json_str(&text);
            }
        }
        Ok(GoogleCredentials::MetadataServer)
    }

    /// Parse a service-account or authorized-user credentials document.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| AgentsError::user(format!("Invalid Google credentials JSON: {e}")))?;
        match value.get("type").and_then(Value::as_str) {
            Some("service_account") => Ok(GoogleCredentials::ServiceAccount(Box::new(value))),
            Some("authorized_user") => Ok(GoogleCredentials::AuthorizedUser(Box::new(value))),
            other => Err(AgentsError::user(format!(
                "Unsupported Google credentials type {:?}; expected service_account or \
                 authorized_user.",
                other.unwrap_or("<missing>")
            ))),
        }
    }
}

/// The default scope requested for Vertex AI.
pub const GOOGLE_CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

const GOOGLE_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

/// Adds a Google OAuth bearer token, refreshing it when it expires.
#[derive(Debug)]
pub struct GoogleOAuthAuth {
    credentials: GoogleCredentials,
    scope: String,
    token_uri: Option<String>,
    http: reqwest::Client,
    cached: Mutex<CachedToken>,
}

impl GoogleOAuthAuth {
    /// Authenticate with `credentials` for `scope`.
    pub fn new(credentials: GoogleCredentials, scope: impl Into<String>) -> Self {
        GoogleOAuthAuth {
            credentials,
            scope: scope.into(),
            token_uri: env("GOOGLE_OAUTH_TOKEN_URI"),
            http: crate::util::http_client(),
            cached: Mutex::new(CachedToken::default()),
        }
    }

    /// Application Default Credentials with the cloud-platform scope.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(
            GoogleCredentials::from_env()?,
            GOOGLE_CLOUD_PLATFORM_SCOPE,
        ))
    }

    /// Override the OAuth token endpoint (used by the tests' mock server).
    pub fn with_token_uri(mut self, uri: impl Into<String>) -> Self {
        self.token_uri = Some(uri.into());
        self
    }

    fn token_uri(&self, credentials: &Value) -> String {
        self.token_uri
            .clone()
            .or_else(|| {
                credentials
                    .get("token_uri")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| GOOGLE_TOKEN_URI.to_string())
    }

    /// A fresh access token, from the cache when it is still valid.
    pub async fn access_token(&self) -> Result<String> {
        if let Some(token) = self.cached.lock().expect("token poisoned").get() {
            return Ok(token.to_string());
        }
        let (token, expires_in) = match &self.credentials {
            GoogleCredentials::AccessToken(t) => (t.clone(), u64::MAX / 2),
            GoogleCredentials::ServiceAccount(json) => self.exchange_jwt(json).await?,
            GoogleCredentials::AuthorizedUser(json) => self.refresh_user(json).await?,
            GoogleCredentials::MetadataServer => self.metadata_token().await?,
        };
        let mut cached = self.cached.lock().expect("token poisoned");
        cached.token = Some(token.clone());
        cached.expires_at = now_secs().saturating_add(expires_in);
        Ok(token)
    }

    async fn parse_token_response(response: reqwest::Response) -> Result<(String, u64)> {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(AgentsError::ApiError {
                status: status.as_u16(),
                message: format!("Google OAuth token request failed: {text}"),
                body: serde_json::from_str(&text).ok().map(Box::new),
                request_id: None,
                headers: None,
            });
        }
        let body: Value = serde_json::from_str(&text)
            .map_err(|e| AgentsError::other(format!("Invalid Google OAuth token response: {e}")))?;
        let token = body
            .get("access_token")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                AgentsError::other("Google OAuth token response has no access_token.")
            })?;
        let expires_in = body
            .get("expires_in")
            .and_then(Value::as_u64)
            .unwrap_or(3600);
        Ok((token.to_string(), expires_in))
    }

    async fn exchange_jwt(&self, credentials: &Value) -> Result<(String, u64)> {
        let token_uri = self.token_uri(credentials);
        let issuer = credentials
            .get("client_email")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentsError::user("Google service-account JSON has no client_email."))?;
        let private_key = credentials
            .get("private_key")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentsError::user("Google service-account JSON has no private_key."))?;
        let now = now_secs();
        let header = BASE64_URL.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
        let claims = serde_json::json!({
            "iss": issuer,
            "scope": self.scope,
            "aud": token_uri,
            "iat": now,
            "exp": now + 3600,
        });
        let payload = BASE64_URL.encode(serde_json::to_vec(&claims)?);
        let signing_input = format!("{header}.{payload}");
        let signature = sign_rs256(private_key, signing_input.as_bytes())?;
        let assertion = format!("{signing_input}.{}", BASE64_URL.encode(signature));
        let response = post_form(
            &self.http,
            &token_uri,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", assertion.as_str()),
            ],
        )
        .await?;
        Self::parse_token_response(response).await
    }

    async fn refresh_user(&self, credentials: &Value) -> Result<(String, u64)> {
        let token_uri = self.token_uri(credentials);
        let field = |name: &str| {
            credentials
                .get(name)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| {
                    AgentsError::user(format!("Google authorized_user JSON has no {name}."))
                })
        };
        let response = post_form(
            &self.http,
            &token_uri,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", &field("client_id")?),
                ("client_secret", &field("client_secret")?),
                ("refresh_token", &field("refresh_token")?),
            ],
        )
        .await?;
        Self::parse_token_response(response).await
    }

    async fn metadata_token(&self) -> Result<(String, u64)> {
        let host = env("GCE_METADATA_HOST").unwrap_or_else(|| "metadata.google.internal".into());
        let url = format!(
            "http://{host}/computeMetadata/v1/instance/service-accounts/default/token?scopes={}",
            uri_encode(&self.scope, true)
        );
        let response = self
            .http
            .get(url)
            .header("Metadata-Flavor", "Google")
            .send()
            .await?;
        Self::parse_token_response(response).await
    }
}

#[async_trait]
impl RequestAuth for GoogleOAuthAuth {
    async fn authorize(
        &self,
        _method: &str,
        _url: &str,
        _body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        let token = self.access_token().await?;
        set_header(headers, "authorization", &format!("Bearer {token}"))
    }
}

/// Sign `message` with an RSASSA-PKCS1-v1_5 SHA-256 signature using a PEM PKCS#8 private key.
fn sign_rs256(pem: &str, message: &[u8]) -> Result<Vec<u8>> {
    let der = pem_body(pem)?;
    let key = aws_lc_rs::signature::RsaKeyPair::from_pkcs8(&der).map_err(|e| {
        AgentsError::user(format!(
            "Google service-account private key is not PKCS#8: {e}"
        ))
    })?;
    let mut signature = vec![0u8; key.public_modulus_len()];
    key.sign(
        &aws_lc_rs::signature::RSA_PKCS1_SHA256,
        &aws_lc_rs::rand::SystemRandom::new(),
        message,
        &mut signature,
    )
    .map_err(|e| AgentsError::other(format!("Failed to sign the Google JWT assertion: {e}")))?;
    Ok(signature)
}

/// Decode the base64 body of a PEM document.
fn pem_body(pem: &str) -> Result<Vec<u8>> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect::<Vec<_>>()
        .join("");
    BASE64
        .decode(body.trim().replace(['\r', ' '], ""))
        .map_err(|e| AgentsError::user(format!("Invalid PEM body: {e}")))
}

const IBM_IAM_TOKEN_URL: &str = "https://iam.cloud.ibm.com/identity/token";

/// Exchanges an IBM Cloud API key for an IAM bearer token (watsonx.ai).
#[derive(Debug)]
pub struct IbmIamAuth {
    api_key: String,
    token_url: String,
    http: reqwest::Client,
    cached: Mutex<CachedToken>,
}

impl IbmIamAuth {
    /// Authenticate with an IBM Cloud API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        IbmIamAuth {
            api_key: api_key.into(),
            token_url: env("IBM_IAM_TOKEN_URL").unwrap_or_else(|| IBM_IAM_TOKEN_URL.to_string()),
            http: crate::util::http_client(),
            cached: Mutex::new(CachedToken::default()),
        }
    }

    /// Override the IAM token endpoint (used by the tests' mock server).
    pub fn with_token_url(mut self, url: impl Into<String>) -> Self {
        self.token_url = url.into();
        self
    }

    /// A fresh IAM access token, from the cache when it is still valid.
    pub async fn access_token(&self) -> Result<String> {
        if let Some(token) = self.cached.lock().expect("token poisoned").get() {
            return Ok(token.to_string());
        }
        let response = post_form(
            &self.http,
            &self.token_url,
            &[
                ("grant_type", "urn:ibm:params:oauth:grant-type:apikey"),
                ("apikey", self.api_key.as_str()),
            ],
        )
        .await?;
        let (token, expires_in) = GoogleOAuthAuth::parse_token_response(response).await?;
        let mut cached = self.cached.lock().expect("token poisoned");
        cached.token = Some(token.clone());
        cached.expires_at = now_secs().saturating_add(expires_in);
        Ok(token)
    }
}

#[async_trait]
impl RequestAuth for IbmIamAuth {
    async fn authorize(
        &self,
        _method: &str,
        _url: &str,
        _body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        let token = self.access_token().await?;
        set_header(headers, "authorization", &format!("Bearer {token}"))
    }
}

/// Sends a static header, for providers that authenticate with something other than a bearer
/// token (Azure OpenAI's `api-key`).
#[derive(Debug, Clone)]
pub struct StaticHeaderAuth {
    name: String,
    value: String,
}

impl StaticHeaderAuth {
    /// Send `name: value` on every request.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        StaticHeaderAuth {
            name: name.into(),
            value: value.into(),
        }
    }
}

#[async_trait]
impl RequestAuth for StaticHeaderAuth {
    async fn authorize(
        &self,
        _method: &str,
        _url: &str,
        _body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        set_header(headers, &self.name, &self.value)
    }
}

/// Convenience alias for a shared [`RequestAuth`].
pub type SharedAuth = Arc<dyn RequestAuth>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_amz_dates() {
        // 2015-08-30T12:36:00Z, the timestamp of the AWS SigV4 test suite.
        assert_eq!(
            amz_date(1_440_938_160),
            ("20150830T123600Z".into(), "20150830".into())
        );
        assert_eq!(amz_date(0), ("19700101T000000Z".into(), "19700101".into()));
    }

    /// The `get-vanilla` case of AWS's published SigV4 test suite.
    #[test]
    fn signs_the_aws_test_vector() {
        let auth = AwsSigV4Auth::new(
            AwsCredentials {
                access_key_id: "AKIDEXAMPLE".into(),
                secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
                session_token: None,
            },
            "us-east-1",
            "service",
        );
        let (authorization, stamp, _) = auth
            .authorization(
                "GET",
                "https://example.amazonaws.com/",
                b"",
                &BTreeMap::new(),
                1_440_938_160,
            )
            .unwrap();
        assert_eq!(stamp, "20150830T123600Z");
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 \
             Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    #[test]
    fn signs_a_body_and_a_session_token() {
        let auth = AwsSigV4Auth::new(
            AwsCredentials {
                access_key_id: "AKIDEXAMPLE".into(),
                secret_access_key: "secret".into(),
                session_token: Some("token".into()),
            },
            "us-west-2",
            "bedrock",
        );
        let (a, _, hash) = auth
            .authorization(
                "POST",
                "https://host/openai/v1/chat/completions?x=1",
                b"{\"model\":\"m\"}",
                &BTreeMap::new(),
                1_440_938_160,
            )
            .unwrap();
        assert!(a.contains("SignedHeaders=host;x-amz-date;x-amz-security-token"));
        assert!(a.contains("/20150830/us-west-2/bedrock/aws4_request"));
        assert_eq!(hash, sha256_hex(b"{\"model\":\"m\"}"));
    }

    #[test]
    fn splits_urls() {
        assert_eq!(
            split_url("https://h.example.com/a/b?c=d").unwrap(),
            ("h.example.com".into(), "/a/b".into(), "c=d".into())
        );
        assert_eq!(
            split_url("https://h.example.com").unwrap(),
            ("h.example.com".into(), "/".into(), String::new())
        );
        assert!(split_url("not a url").is_err());
    }
}
