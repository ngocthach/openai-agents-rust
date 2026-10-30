//! Minimal async HTTP client for the OpenAI API (JSON requests and SSE streams).

use std::collections::HashMap;
use std::fmt;

use bytes::BytesMut;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::error::{AgentsError, Result};

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// The host of the official OpenAI API.
const OPENAI_HOST: &str = "api.openai.com";

/// Whether `base_url` points at the official OpenAI API.
///
/// Only the scheme and the host are compared, so a path or a port-less variant of the official
/// endpoint still counts. Mirrors `is_official_openai_base_url`.
pub fn is_official_openai_base_url(base_url: &str, websocket: bool) -> bool {
    let expected_scheme = if websocket { "wss" } else { "https" };
    let Some((scheme, rest)) = base_url.trim().split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case(expected_scheme) {
        return false;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    // Strip userinfo, then the port.
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = host.rsplit_once(':').map_or(host, |(h, _)| h);
    host.eq_ignore_ascii_case(OPENAI_HOST)
}

/// Signs or otherwise authenticates each outgoing request.
///
/// The Rust stand-in for the per-provider auth any-llm performs inside each provider class:
/// AWS SigV4, an Azure `api-key` header, a refreshed Google OAuth bearer. The hook runs after
/// the client's own headers are built and before the request is sent, and sees the final URL
/// (query string included) and the exact request body, which is what a signature covers.
#[async_trait::async_trait]
pub trait RequestAuth: Send + Sync + fmt::Debug {
    /// Add (or replace) authentication headers for one request.
    async fn authorize(
        &self,
        method: &str,
        url: &str,
        body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()>;
}

/// Async client for the OpenAI REST API (or any compatible endpoint).
#[derive(Clone)]
pub struct OpenAIClient {
    http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    default_headers: HashMap<String, String>,
    /// Header names removed after every other source was applied. Python's `Omit`.
    omitted_headers: Vec<String>,
    /// Query parameters added to every request. Mirrors `AsyncOpenAI.default_query`.
    default_query: HashMap<String, String>,
    auth: Option<std::sync::Arc<dyn RequestAuth>>,
    /// Sends `Authorization: Bearer <key>`, unless a provider wants another header.
    send_bearer_auth: bool,
    /// Base URL for websocket endpoints. Unset means "derive it from `base_url`".
    websocket_base_url: Option<String>,
}

impl fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIClient")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("organization", &self.organization)
            .field("project", &self.project)
            .finish_non_exhaustive()
    }
}

impl Default for OpenAIClient {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Response headers relevant to retry decisions, keyed by lowercase name.
pub(crate) fn retry_headers(
    headers: &reqwest::header::HeaderMap,
) -> Option<std::collections::BTreeMap<String, String>> {
    let map: std::collections::BTreeMap<String, String> =
        ["retry-after", "retry-after-ms", "x-should-retry"]
            .iter()
            .filter_map(|k| {
                headers
                    .get(*k)
                    .and_then(|v| v.to_str().ok())
                    .map(|v| (k.to_string(), v.to_string()))
            })
            .collect();
    (!map.is_empty()).then_some(map)
}

impl OpenAIClient {
    /// A client with an explicit API key and the default base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        let mut c = Self::from_env();
        c.api_key = Some(api_key.into());
        c
    }

    /// A client configured from `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID` and
    /// `OPENAI_PROJECT_ID`. The key is resolved lazily if it is set later.
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        OpenAIClient {
            http: crate::util::http_client(),
            api_key: None,
            base_url: env("OPENAI_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            organization: env("OPENAI_ORG_ID"),
            project: env("OPENAI_PROJECT_ID"),
            default_headers: HashMap::new(),
            omitted_headers: Vec::new(),
            default_query: HashMap::new(),
            auth: None,
            send_bearer_auth: true,
            websocket_base_url: None,
        }
    }

    /// Set the base URL (e.g. an OpenAI-compatible provider).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set the project header.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Add a default header sent on every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(name.into(), value.into());
        self
    }

    /// Remove a header that would otherwise be sent (the client's own, a default header or a
    /// per-request extra header). The equivalent of passing OpenAI's `Omit` for that key.
    pub fn without_header(mut self, name: impl Into<String>) -> Self {
        self.omitted_headers.push(name.into());
        self
    }

    /// Add a query parameter sent on every request. Mirrors `AsyncOpenAI.default_query`.
    pub fn with_default_query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_query.insert(name.into(), value.into());
        self
    }

    /// The query parameters sent on every request.
    pub fn default_query(&self) -> &HashMap<String, String> {
        &self.default_query
    }

    /// Authenticate each request through `auth` (AWS SigV4, a Google OAuth bearer, ...)
    /// instead of, or in addition to, the bearer header.
    pub fn with_request_auth(mut self, auth: std::sync::Arc<dyn RequestAuth>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Stop sending `Authorization: Bearer <key>` (for providers that use another header).
    pub fn with_bearer_auth(mut self, enabled: bool) -> Self {
        self.send_bearer_auth = enabled;
        self
    }

    /// Use a custom `reqwest::Client` (proxies, timeouts, TLS settings).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The explicitly configured API key, if any.
    pub(crate) fn explicit_api_key(&self) -> Option<String> {
        self.api_key.clone()
    }

    /// Route websocket endpoints at `url` instead of the scheme-swapped HTTP base URL.
    /// Mirrors `AsyncOpenAI(websocket_base_url=...)`.
    pub fn with_websocket_base_url(mut self, url: impl Into<String>) -> Self {
        self.websocket_base_url = Some(url.into());
        self
    }

    /// The base URL websocket endpoints are built from.
    ///
    /// Defaults to [`base_url`](Self::base_url) with `http`/`https` swapped for `ws`/`wss`,
    /// which is what the OpenAI clients do when no explicit websocket base URL is set.
    pub fn websocket_base_url(&self) -> String {
        if let Some(url) = &self.websocket_base_url {
            return url.clone();
        }
        let base = self.base_url.trim_end_matches('/');
        match base.split_once("://") {
            Some(("https", rest)) => format!("wss://{rest}"),
            Some(("http", rest)) => format!("ws://{rest}"),
            _ => base.to_string(),
        }
    }

    /// The absolute websocket URL of `path`, with the default query parameters applied.
    pub fn websocket_url(&self, path: &str) -> String {
        let base = self.websocket_base_url();
        let url = format!(
            "{}/{}",
            base.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        if self.default_query.is_empty() {
            return url;
        }
        let mut params: Vec<(&str, &str)> = self
            .default_query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        params.sort_unstable();
        let separator = if url.contains('?') { '&' } else { '?' };
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        format!("{url}{separator}{query}")
    }

    /// The handshake headers for a websocket connection to `url`, with the client's
    /// [`RequestAuth`] hook applied. A websocket handshake is a `GET`, so the hook sees the
    /// same method the server does and no body.
    pub async fn websocket_headers(&self, url: &str) -> Result<HeaderMap> {
        let mut headers = self.headers(None)?;
        self.authorize("GET", url, None, &mut headers).await?;
        Ok(headers)
    }

    /// The base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// True when the client targets the official OpenAI API.
    pub fn is_openai_api(&self) -> bool {
        is_official_openai_base_url(&self.base_url, false)
    }

    /// The API key: explicit, then the global default, then `OPENAI_API_KEY`.
    pub fn api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(crate::models::default_openai_key)
            .or_else(|| {
                std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty())
            })
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    pub(crate) fn headers(&self, extra: Option<&HashMap<String, String>>) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        if self.send_bearer_auth
            && let Some(key) = self.api_key()
        {
            h.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {key}"))
                    .map_err(|e| AgentsError::user(format!("Invalid API key header: {e}")))?,
            );
        }
        if let Some(org) = &self.organization {
            insert_header(&mut h, "OpenAI-Organization", org)?;
        }
        if let Some(p) = &self.project {
            insert_header(&mut h, "OpenAI-Project", p)?;
        }
        insert_header(
            &mut h,
            "User-Agent",
            &format!("Agents/Rust {}", env!("CARGO_PKG_VERSION")),
        )?;
        for (k, v) in &self.default_headers {
            insert_header(&mut h, k, v)?;
        }
        if let Some(extra) = extra {
            for (k, v) in extra {
                insert_header(&mut h, k, v)?;
            }
        }
        for name in &self.omitted_headers {
            if let Ok(name) = HeaderName::from_bytes(name.as_bytes()) {
                h.remove(name);
            }
        }
        Ok(h)
    }

    /// The absolute URL of `path`, with the default and per-request query parameters applied.
    fn url_with_query(&self, path: &str, extra: Option<&HashMap<String, String>>) -> String {
        let mut params: Vec<(&str, &str)> = self
            .default_query
            .iter()
            .filter(|(k, _)| extra.is_none_or(|e| !e.contains_key(*k)))
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        params.extend(
            extra
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.as_str(), v.as_str())),
        );
        let url = self.url(path);
        if params.is_empty() {
            return url;
        }
        params.sort_unstable();
        let separator = if url.contains('?') { '&' } else { '?' };
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        format!("{url}{separator}{query}")
    }

    /// Run the client's [`RequestAuth`] hook over a request that is about to be sent.
    pub(crate) async fn authorize(
        &self,
        method: &str,
        url: &str,
        body: Option<&[u8]>,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        match &self.auth {
            Some(auth) => auth.authorize(method, url, body, headers).await,
            None => Ok(()),
        }
    }

    async fn check(resp: reqwest::Response) -> Result<(reqwest::Response, Option<String>)> {
        let request_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let status = resp.status();
        if status.is_success() {
            return Ok((resp, request_id));
        }
        let headers = retry_headers(resp.headers());
        let text = resp.text().await.unwrap_or_default();
        let body: Option<Value> = serde_json::from_str(&text).ok();
        let message = body
            .as_ref()
            .and_then(|b| b.pointer("/error/message"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                if text.is_empty() {
                    status.to_string()
                } else {
                    text.clone()
                }
            });
        Err(AgentsError::ApiError {
            status: status.as_u16(),
            message,
            body: body.map(Box::new),
            request_id,
            headers,
        })
    }

    /// POST a JSON body and parse the JSON response. Returns the body and `x-request-id`.
    pub async fn post_json(
        &self,
        path: &str,
        body: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        extra_query: Option<&HashMap<String, String>>,
    ) -> Result<(Value, Option<String>)> {
        let url = self.url_with_query(path, extra_query);
        let payload = serde_json::to_vec(body)?;
        let mut headers = self.headers(extra_headers)?;
        insert_header(&mut headers, "Content-Type", "application/json")?;
        self.authorize("POST", &url, Some(&payload), &mut headers)
            .await?;
        let req = self.http.post(url).headers(headers).body(payload);
        let (resp, request_id) = Self::check(req.send().await?).await?;
        let value = resp.json::<Value>().await?;
        Ok((value, request_id))
    }

    /// GET a JSON resource.
    pub async fn get_json(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        let extra: HashMap<String, String> = query
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();
        let url = self.url_with_query(path, Some(&extra));
        let mut headers = self.headers(None)?;
        self.authorize("GET", &url, None, &mut headers).await?;
        let req = self.http.get(url).headers(headers);
        let (resp, _) = Self::check(req.send().await?).await?;
        Ok(resp.json::<Value>().await?)
    }

    /// DELETE a resource.
    pub async fn delete(&self, path: &str) -> Result<Value> {
        let url = self.url_with_query(path, None);
        let mut headers = self.headers(None)?;
        self.authorize("DELETE", &url, None, &mut headers).await?;
        let req = self.http.delete(url).headers(headers);
        let (resp, _) = Self::check(req.send().await?).await?;
        let text = resp.text().await?;
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    /// POST a JSON body and stream server-sent events as JSON values.
    pub async fn post_sse(
        &self,
        path: &str,
        body: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        extra_query: Option<&HashMap<String, String>>,
    ) -> Result<(BoxStream<'static, Result<Value>>, Option<String>)> {
        let url = self.url_with_query(path, extra_query);
        let payload = serde_json::to_vec(body)?;
        let mut headers = self.headers(extra_headers)?;
        insert_header(&mut headers, "Content-Type", "application/json")?;
        insert_header(&mut headers, "Accept", "text/event-stream")?;
        self.authorize("POST", &url, Some(&payload), &mut headers)
            .await?;
        let req = self.http.post(url).headers(headers).body(payload);
        let (resp, request_id) = Self::check(req.send().await?).await?;
        Ok((sse_stream(resp.bytes_stream()), request_id))
    }

    /// POST a `multipart/form-data` body and parse the JSON response.
    ///
    /// The `RequestAuth` hook sees the request without a body: a multipart body is streamed,
    /// so there are no bytes to sign at this point.
    ///
    /// Uses reqwest's multipart support.
    #[cfg(feature = "voice")]
    pub async fn post_multipart(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> Result<(Value, Option<String>)> {
        let url = self.url_with_query(path, None);
        let mut headers = self.headers(None)?;
        // `reqwest` sets `Content-Type` itself, with the generated boundary.
        headers.remove(reqwest::header::CONTENT_TYPE);
        self.authorize("POST", &url, None, &mut headers).await?;
        let req = self.http.post(url).headers(headers).multipart(form);
        let (resp, request_id) = Self::check(req.send().await?).await?;
        let value = resp.json::<Value>().await?;
        Ok((value, request_id))
    }

    /// POST a JSON body and stream the raw response bytes, for endpoints that answer with
    /// binary content such as `/audio/speech`.
    pub async fn post_bytes_stream(
        &self,
        path: &str,
        body: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<(BoxStream<'static, Result<bytes::Bytes>>, Option<String>)> {
        let url = self.url_with_query(path, None);
        let payload = serde_json::to_vec(body)?;
        let mut headers = self.headers(extra_headers)?;
        insert_header(&mut headers, "Content-Type", "application/json")?;
        self.authorize("POST", &url, Some(&payload), &mut headers)
            .await?;
        let req = self.http.post(url).headers(headers).body(payload);
        let (resp, request_id) = Self::check(req.send().await?).await?;
        let stream = resp
            .bytes_stream()
            .map(|chunk| chunk.map_err(AgentsError::from));
        Ok((Box::pin(stream), request_id))
    }

    /// List the models the endpoint exposes (`GET /models`), newest first as returned.
    ///
    /// any-llm exposes a `list_models` per provider; every OpenAI-compatible endpoint answers
    /// the same route, so the registry-based adapters get it for free.
    pub async fn list_models(&self) -> Result<Vec<Value>> {
        let body = self.get_json("models", &[]).await?;
        Ok(match body.get("data") {
            Some(Value::Array(items)) => items.clone(),
            _ => Vec::new(),
        })
    }
}

/// Percent-encode a query component.
fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn insert_header(h: &mut HeaderMap, k: &str, v: &str) -> Result<()> {
    let name = HeaderName::from_bytes(k.as_bytes())
        .map_err(|e| AgentsError::user(format!("Invalid header name {k}: {e}")))?;
    let value = HeaderValue::from_str(v)
        .map_err(|e| AgentsError::user(format!("Invalid header value for {k}: {e}")))?;
    h.insert(name, value);
    Ok(())
}

/// Parse a byte stream of server-sent events into JSON `data` payloads.
pub(crate) fn sse_stream<S, B, E>(bytes: S) -> BoxStream<'static, Result<Value>>
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]>,
    E: Into<AgentsError>,
{
    let state = (Box::pin(bytes), BytesMut::new(), false);
    futures::stream::unfold(state, |(mut bytes, mut buf, mut done)| async move {
        loop {
            if done {
                return None;
            }
            if let Some(event) = take_event(&mut buf) {
                match parse_event(&event) {
                    SseEvent::Data(v) => return Some((Ok(v), (bytes, buf, done))),
                    SseEvent::Done => return None,
                    SseEvent::Skip => continue,
                    SseEvent::Error(e) => {
                        return Some((Err(e), (bytes, buf, true)));
                    }
                }
            }
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend_from_slice(chunk.as_ref()),
                Some(Err(e)) => return Some((Err(e.into()), (bytes, buf, true))),
                None => {
                    done = true;
                    if !buf.is_empty() {
                        let rest = String::from_utf8_lossy(&buf).to_string();
                        buf.clear();
                        if let SseEvent::Data(v) = parse_event(&rest) {
                            return Some((Ok(v), (bytes, buf, done)));
                        }
                    }
                    return None;
                }
            }
        }
    })
    .boxed()
}

enum SseEvent {
    Data(Value),
    Done,
    Skip,
    Error(AgentsError),
}

fn take_event(buf: &mut BytesMut) -> Option<String> {
    let s = &buf[..];
    let mut idx = None;
    for i in 0..s.len() {
        if s[i] == b'\n' && i + 1 < s.len() && s[i + 1] == b'\n' {
            idx = Some((i, 2));
            break;
        }
        if s[i] == b'\r' && i + 3 < s.len() && &s[i..i + 4] == b"\r\n\r\n" {
            idx = Some((i, 4));
            break;
        }
    }
    let (i, sep) = idx?;
    let event = String::from_utf8_lossy(&s[..i]).to_string();
    let _ = buf.split_to(i + sep);
    Some(event)
}

fn parse_event(event: &str) -> SseEvent {
    let mut data = String::new();
    let mut event_name: Option<String> = None;
    for line in event.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        } else if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_string());
        }
    }
    if data.is_empty() {
        return SseEvent::Skip;
    }
    if data.trim() == "[DONE]" {
        return SseEvent::Done;
    }
    match serde_json::from_str::<Value>(&data) {
        Ok(v) => {
            if event_name.as_deref() == Some("error")
                || v.get("type").and_then(Value::as_str) == Some("error")
            {
                let msg = v
                    .pointer("/error/message")
                    .or_else(|| v.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("stream error")
                    .to_string();
                return SseEvent::Error(AgentsError::ApiError {
                    status: 0,
                    message: msg,
                    body: Some(Box::new(v)),
                    request_id: None,
                    headers: None,
                });
            }
            SseEvent::Data(v)
        }
        Err(e) => SseEvent::Error(AgentsError::model_behavior(format!(
            "Invalid JSON in stream event: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parses_sse_chunks_across_boundaries() {
        let chunks: Vec<std::result::Result<Vec<u8>, AgentsError>> = vec![
            Ok(b"event: response.created\ndata: {\"type\":\"a\"".to_vec()),
            Ok(b"}\n\ndata: {\"type\":\"b\"}\r\n\r\n: comment\n\ndata: [DONE]\n\n".to_vec()),
        ];
        let s = sse_stream(futures::stream::iter(chunks));
        let items: Vec<_> = s.collect().await;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap()["type"], "a");
        assert_eq!(items[1].as_ref().unwrap()["type"], "b");
    }

    #[tokio::test]
    async fn error_events_become_errors() {
        let chunks: Vec<std::result::Result<Vec<u8>, AgentsError>> = vec![Ok(
            b"data: {\"type\":\"error\",\"message\":\"boom\"}\n\n".to_vec(),
        )];
        let items: Vec<_> = sse_stream(futures::stream::iter(chunks)).collect().await;
        assert!(items[0].is_err());
    }

    #[test]
    fn recognizes_the_official_endpoint_by_scheme_and_host() {
        assert!(is_official_openai_base_url(DEFAULT_BASE_URL, false));
        assert!(is_official_openai_base_url(
            "https://api.openai.com/v2",
            false
        ));
        assert!(is_official_openai_base_url("https://api.openai.com", false));
        assert!(!is_official_openai_base_url(
            "http://api.openai.com/v1",
            false
        ));
        assert!(!is_official_openai_base_url(
            "https://proxy.local/v1",
            false
        ));
        assert!(is_official_openai_base_url("wss://api.openai.com/v1", true));
        assert!(!is_official_openai_base_url(
            "https://api.openai.com/v1",
            true
        ));
    }
}
