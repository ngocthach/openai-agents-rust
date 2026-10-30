//! The OpenAI Responses API over a persistent WebSocket (`response.create` frames).
//!
//! Port of `OpenAIResponsesWSModel`. One connection is kept open and reused across requests,
//! which lowers per-turn latency for agent loops. Requests on the same model are serialized.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use super::interface::{Model, ModelRequest, ResponseEventStream};
use super::openai_client::OpenAIClient;
use super::openai_responses::{
    OpenAIResponsesModel, SpanGuard, finish_response_span, parse_response,
};
use super::response_terminal::{is_terminal_failure_event, terminal_failure_error};
use super::retry_runtime::should_disable_websocket_pre_event_retries;
use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::retry::{ModelRetryAdvice, ModelRetryNormalizedError, ReplaySafety};
use crate::tracing::{record_model_error_on_span, response_span};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// The connection closed after the request frame was sent but before a terminal event, so the
/// server may or may not have acted on it. Mirrors `_is_ambiguous_websocket_replay_error`.
pub const WS_AMBIGUOUS_REPLAY_MESSAGE: &str =
    "Responses websocket connection closed before a terminal response event.";

/// [`OpenAIResponsesWSModel::close_connection`] ran while this request was connecting, so the
/// failure is an intentional abort rather than a transport error.
/// Mirrors `_websocket_close_invalidation_error`.
pub const WS_CLOSE_INVALIDATED_MESSAGE: &str =
    "Responses websocket connection closed while establishing a connection.";

/// The connection dropped before a single response event arrived.
/// Mirrors the `wrapped_disconnect` message.
pub const WS_PRE_EVENT_DISCONNECT_MESSAGE: &str = "Responses websocket connection closed before any response events were received. The feature \
     may not be enabled for this account/model yet, or the server closed the connection.";

/// Appended when the request frame had already reached the server, which makes a replay unsafe.
const REQUEST_SENT_SUFFIX: &str = " (the request frame was already sent)";

/// Appended when response events had already been emitted, which makes a replay unsafe.
const RESPONSE_STARTED_SUFFIX: &str = " (the response had already started)";

/// Request phases that can time out, in `Responses websocket {phase} timed out` messages.
const TIMEOUT_PHASES: &[&str] = &["request lock wait", "connect", "send", "receive"];

struct Connection {
    identity: String,
    socket: Ws,
}

/// Low-level connection options for the Responses WebSocket transport.
/// Port of `OpenAIResponsesWebSocketOptions`.
///
/// The defaults mirror Python's: keepalive pings every 20 s with a 20 s heartbeat timeout, and
/// no limit on the size of an incoming message.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIResponsesWebSocketOptions {
    /// Seconds between client keepalive pings; `None` disables keepalive pings.
    pub ping_interval: Option<f64>,
    /// Seconds to wait for a pong before failing the request; `None` keeps pings but never
    /// times the heartbeat out.
    pub ping_timeout: Option<f64>,
    /// Maximum size in bytes of an incoming message; `None` (the default) means no limit.
    pub max_size: Option<usize>,
}

impl Default for OpenAIResponsesWebSocketOptions {
    fn default() -> Self {
        OpenAIResponsesWebSocketOptions {
            ping_interval: Some(20.0),
            ping_timeout: Some(20.0),
            max_size: None,
        }
    }
}

impl OpenAIResponsesWebSocketOptions {
    /// Seconds between keepalive pings (`None` disables them).
    pub fn with_ping_interval(mut self, seconds: Option<f64>) -> Self {
        self.ping_interval = seconds;
        self
    }
    /// Seconds to wait for a pong (`None` disables the heartbeat timeout).
    pub fn with_ping_timeout(mut self, seconds: Option<f64>) -> Self {
        self.ping_timeout = seconds;
        self
    }
    /// Maximum size of an incoming message in bytes (`None` means no limit).
    pub fn with_max_size(mut self, bytes: Option<usize>) -> Self {
        self.max_size = bytes;
        self
    }

    fn websocket_config(&self) -> WebSocketConfig {
        // Python passes `max_size` straight to `websockets.connect`, whose `None` means
        // unlimited; tungstenite instead defaults to 64 MiB, so the default is overridden.
        let mut config = WebSocketConfig::default();
        config.max_message_size = self.max_size;
        config.max_frame_size = self.max_size;
        config
    }
}

/// Per-request transport timeouts derived from `ModelSettings::timeout`.
/// Port of `_WebsocketRequestTimeouts`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RequestTimeouts {
    lock: Option<f64>,
    connect: Option<f64>,
    send: Option<f64>,
    recv: Option<f64>,
}

impl RequestTimeouts {
    fn from_seconds(timeout: Option<f64>) -> Self {
        match timeout {
            Some(t) if t.is_finite() && t >= 0.0 => RequestTimeouts {
                lock: Some(t),
                connect: Some(t),
                send: Some(t),
                recv: Some(t),
            },
            _ => RequestTimeouts::default(),
        }
    }
}

fn duration(seconds: f64) -> Duration {
    Duration::try_from_secs_f64(seconds.max(0.0)).unwrap_or(Duration::ZERO)
}

fn timeout_error(phase: &str, seconds: f64) -> AgentsError {
    AgentsError::other(format!(
        "Responses websocket {phase} timed out after {seconds} seconds."
    ))
}

/// The request phase named by a timeout error, if any. Port of `_get_websocket_timeout_phase`.
fn websocket_timeout_phase(message: &str) -> Option<&'static str> {
    TIMEOUT_PHASES
        .iter()
        .copied()
        .find(|phase| message.starts_with(&format!("Responses websocket {phase} timed out")))
}

async fn with_timeout<F: Future>(seconds: Option<f64>, phase: &str, fut: F) -> Result<F::Output> {
    match seconds {
        None => Ok(fut.await),
        Some(s) => tokio::time::timeout(duration(s), fut)
            .await
            .map_err(|_| timeout_error(phase, s)),
    }
}

/// A Responses API model that talks to OpenAI over a WebSocket.
#[derive(Clone)]
pub struct OpenAIResponsesWSModel {
    inner: OpenAIResponsesModel,
    client: OpenAIClient,
    websocket_base_url: Option<String>,
    options: OpenAIResponsesWebSocketOptions,
    connection: Arc<Mutex<Option<Connection>>>,
    /// Bumped by [`OpenAIResponsesWSModel::close_connection`]; a request whose generation
    /// changed mid-flight failed because the caller closed the model, not because of the wire.
    close_generation: Arc<AtomicU64>,
}

impl std::fmt::Debug for OpenAIResponsesWSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesWSModel")
            .field("model", &self.inner.model())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

/// Convert an HTTP(S) base URL to its WebSocket equivalent.
pub(crate) fn websocket_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    }
}

fn is_terminal(ty: &str) -> bool {
    matches!(
        ty,
        "response.completed" | "response.failed" | "response.incomplete" | "response.error"
    )
}

/// The error raised for a transport `error` / `response.error` frame.
/// Port of `ResponsesWebSocketError`, including its `"{prefix}: {payload}"` message.
fn ws_error(payload: &Value) -> AgentsError {
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("error");
    let prefix = if event_type == "error" {
        "Responses websocket error".to_string()
    } else {
        format!("Responses websocket {event_type}")
    };
    AgentsError::other(format!("{prefix}: {payload}"))
}

/// Recover the payload of a [`ws_error`] message, so retry advice can read `error.code`.
fn ws_error_payload(message: &str) -> Option<Value> {
    let start = message.find('{')?;
    serde_json::from_str(&message[start..]).ok()
}

impl OpenAIResponsesWSModel {
    /// Create a WebSocket Responses model.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAIResponsesWSModel {
            inner: OpenAIResponsesModel::new(model, client.clone()),
            client,
            websocket_base_url: None,
            options: OpenAIResponsesWebSocketOptions::default(),
            connection: Arc::new(Mutex::new(None)),
            close_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn with_explicit(mut self, explicit: bool) -> Self {
        self.inner = self.inner.with_explicit(explicit);
        self
    }

    /// Override the WebSocket base URL (defaults to the client base URL with a `ws(s)` scheme).
    pub fn with_websocket_base_url(mut self, url: impl Into<String>) -> Self {
        self.websocket_base_url = Some(url.into());
        self
    }

    /// Set the low-level connection options (keepalive pings, message size limit).
    pub fn with_websocket_options(mut self, options: OpenAIResponsesWebSocketOptions) -> Self {
        self.options = options;
        self
    }

    /// The low-level connection options in use.
    pub fn websocket_options(&self) -> &OpenAIResponsesWebSocketOptions {
        &self.options
    }

    /// The model name.
    pub fn model(&self) -> &str {
        self.inner.model()
    }

    /// Build the `response.create` frame for a request.
    pub fn build_frame(&self, request: &ModelRequest) -> Result<Value> {
        let mut frame = self.inner.build_request_body(request, true)?;
        frame["type"] = json!("response.create");
        frame["stream"] = json!(true);
        Ok(frame)
    }

    fn url(&self, extra_query: Option<&HashMap<String, String>>) -> Result<String> {
        let base = self
            .websocket_base_url
            .clone()
            .unwrap_or_else(|| websocket_url(self.client.base_url()));
        let mut url = format!("{}/responses", base.trim_end_matches('/'));
        let mut pairs: Vec<(String, String)> = self
            .client
            .default_query()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if let Some(q) = extra_query {
            for (k, v) in q {
                pairs.retain(|(existing, _)| existing != k);
                pairs.push((k.clone(), v.clone()));
            }
        }
        if !pairs.is_empty() {
            pairs.sort();
            let qs = pairs
                .iter()
                .map(|(k, v)| format!("{}={}", encode(k), encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            url.push('?');
            url.push_str(&qs);
        }
        Ok(url)
    }

    async fn connect(
        &self,
        url: &str,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Ws> {
        let mut req = url
            .into_client_request()
            .map_err(|e| AgentsError::user(format!("Invalid Responses websocket URL: {e}")))?;
        for (k, v) in self.client.headers(extra_headers)?.iter() {
            req.headers_mut().insert(k.clone(), v.clone());
        }
        crate::util::ensure_crypto_provider();
        let (socket, _) = tokio_tungstenite::connect_async_with_config(
            req,
            Some(self.options.websocket_config()),
            false,
        )
        .await
        .map_err(|e| AgentsError::other(format!("Responses websocket connect failed: {e}")))?;
        Ok(socket)
    }

    /// Close the cached connection.
    pub async fn close_connection(&self) {
        self.close_generation.fetch_add(1, Ordering::SeqCst);
        if let Some(mut c) = self.connection.lock().await.take() {
            let _ = c.socket.close(None).await;
        }
    }

    async fn open_stream(&self, request: &ModelRequest) -> Result<ResponseEventStream> {
        let frame = self.build_frame(request)?;
        let settings = &request.model_settings;
        let timeouts = RequestTimeouts::from_seconds(settings.timeout);
        let url = self.url(settings.extra_query.as_ref())?;
        let mut headers: Vec<(String, String)> = settings
            .extra_headers
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        headers.sort();
        let identity = format!("{url}|{headers:?}");
        let text = serde_json::to_string(&frame)?;
        let generation = self.close_generation.load(Ordering::SeqCst);

        let mut guard = with_timeout(
            timeouts.lock,
            "request lock wait",
            self.connection.clone().lock_owned(),
        )
        .await?;
        // A cached connection may have been closed by the server; reconnect once if the frame
        // could not be sent on it (the request was never transmitted, so replay is safe).
        let mut pre_event_retry = !should_disable_websocket_pre_event_retries();
        loop {
            let reuse = guard.as_ref().is_some_and(|c| c.identity == identity);
            if !reuse {
                if let Some(mut old) = guard.take() {
                    let _ = old.socket.close(None).await;
                }
                let socket = with_timeout(
                    timeouts.connect,
                    "connect",
                    self.connect(&url, settings.extra_headers.as_ref()),
                )
                .await??;
                *guard = Some(Connection {
                    identity: identity.clone(),
                    socket,
                });
            }
            let conn = guard.as_mut().expect("connection");
            let sent = with_timeout(
                timeouts.send,
                "send",
                conn.socket.send(Message::text(text.clone())),
            )
            .await;
            match sent {
                Ok(Ok(())) => break,
                // A send timeout may have left a partial frame on the wire.
                Err(e) => {
                    *guard = None;
                    return Err(e);
                }
                Ok(Err(e)) => {
                    *guard = None;
                    if !reuse || !pre_event_retry {
                        if self.close_generation.load(Ordering::SeqCst) != generation {
                            return Err(AgentsError::other(WS_CLOSE_INVALIDATED_MESSAGE));
                        }
                        // Nothing reached the server, so replaying the request is safe.
                        return Err(AgentsError::other(format!(
                            "Responses websocket send failed: {e}"
                        )));
                    }
                    pre_event_retry = false;
                }
            }
        }

        let options = self.options.clone();
        let close_generation = self.close_generation.clone();
        let state = (Some(guard), false, generation);
        let stream = futures::stream::unfold(state, move |state| {
            let options = options.clone();
            let close_generation = close_generation.clone();
            async move {
                let (guard, mut received_any_event, generation) = state;
                let mut guard = guard?;
                loop {
                    let conn = guard.as_mut()?;
                    let next = recv_next(&mut conn.socket, &options, timeouts.recv).await;
                    let frame = match next {
                        Ok(None) | Ok(Some(Message::Close(_))) => {
                            Err(AgentsError::other(WS_AMBIGUOUS_REPLAY_MESSAGE))
                        }
                        Ok(Some(m)) => Ok(m),
                        Err(e) => Err(e),
                    };
                    let frame = match frame {
                        Ok(f) => f,
                        Err(e) => {
                            *guard = None;
                            let invalidated = close_generation.load(Ordering::SeqCst) != generation;
                            let e = classify_stream_failure(e, received_any_event, invalidated);
                            return Some((Err(e), (None, received_any_event, generation)));
                        }
                    };
                    let text = match frame {
                        Message::Text(t) => t.to_string(),
                        Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                        _ => continue,
                    };
                    let payload: Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            *guard = None;
                            return Some((Err(e.into()), (None, received_any_event, generation)));
                        }
                    };
                    let ty = payload
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if ty == "error" || ty == "response.error" {
                        *guard = None;
                        return Some((
                            Err(ws_error(&payload)),
                            (None, received_any_event, generation),
                        ));
                    }
                    received_any_event = true;
                    // The connection stays cached for the next request after a terminal event.
                    let next_state = if is_terminal(&ty) {
                        (None, received_any_event, generation)
                    } else {
                        (Some(guard), received_any_event, generation)
                    };
                    return Some((Ok(payload), next_state));
                }
            }
        });
        Ok(stream.boxed())
    }
}

/// Label a mid-request transport failure with what the server may already have done with the
/// request. Port of the `_openai_agents_ws_replay_safety` / close-invalidation marking.
fn classify_stream_failure(
    error: AgentsError,
    received_any_event: bool,
    close_invalidated: bool,
) -> AgentsError {
    if close_invalidated {
        return AgentsError::other(WS_CLOSE_INVALIDATED_MESSAGE);
    }
    let message = error.to_string();
    if websocket_timeout_phase(&message).is_some() && !received_any_event {
        return error;
    }
    if received_any_event {
        // Events were emitted but no terminal event arrived: the server acted on the request.
        return AgentsError::other(format!("{message}{RESPONSE_STARTED_SUFFIX}"));
    }
    if message.starts_with(WS_AMBIGUOUS_REPLAY_MESSAGE) {
        // The frame was sent and the connection died before any event: a replay could run the
        // request twice.
        return AgentsError::other(format!(
            "{WS_PRE_EVENT_DISCONNECT_MESSAGE}{REQUEST_SENT_SUFFIX}"
        ));
    }
    error
}

/// How long to wait for a frame before the next keepalive action.
///
/// With keepalive on, that is the heartbeat timeout while a pong is outstanding and the ping
/// interval otherwise. `ping_timeout: None` means "keep pinging but never fail", so it falls
/// back to the interval instead of waiting forever.
fn keepalive_wait(
    options: &OpenAIResponsesWebSocketOptions,
    awaiting_pong: bool,
) -> Option<Duration> {
    let interval = options.ping_interval?;
    Some(duration(if awaiting_pong {
        options.ping_timeout.unwrap_or(interval)
    } else {
        interval
    }))
}

/// Read the next frame, sending keepalive pings and enforcing the receive timeout.
///
/// `websockets` runs its keepalive on a background task; tungstenite has none, so the ping is
/// driven from the receive loop. It therefore only runs while a request is in flight, which is
/// the only window in which a dead connection can stall a run.
async fn recv_next(
    socket: &mut Ws,
    options: &OpenAIResponsesWebSocketOptions,
    recv_timeout: Option<f64>,
) -> Result<Option<Message>> {
    let deadline = recv_timeout.map(|t| Instant::now() + duration(t));
    let mut awaiting_pong = false;
    loop {
        let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if remaining == Some(Duration::ZERO) {
            return Err(timeout_error("receive", recv_timeout.unwrap_or_default()));
        }
        let keepalive = keepalive_wait(options, awaiting_pong);
        let wait = match (remaining, keepalive) {
            (Some(r), Some(k)) => Some(r.min(k)),
            (Some(r), None) => Some(r),
            (None, k) => k,
        };
        let next = match wait {
            Some(w) => match tokio::time::timeout(w, socket.next()).await {
                Ok(n) => n,
                Err(_) => {
                    if let Some(d) = deadline
                        && Instant::now() >= d
                    {
                        return Err(timeout_error("receive", recv_timeout.unwrap_or_default()));
                    }
                    if awaiting_pong && let Some(ping_timeout) = options.ping_timeout {
                        return Err(timeout_error("receive", ping_timeout));
                    }
                    if options.ping_interval.is_some() {
                        socket
                            .send(Message::Ping(bytes::Bytes::new()))
                            .await
                            .map_err(|e| {
                                AgentsError::other(format!(
                                    "Responses websocket keepalive ping failed: {e}"
                                ))
                            })?;
                        awaiting_pong = true;
                    }
                    continue;
                }
            },
            None => socket.next().await,
        };
        match next {
            Some(Ok(Message::Pong(_) | Message::Ping(_))) => {
                awaiting_pong = false;
                continue;
            }
            Some(Ok(m)) => return Ok(Some(m)),
            Some(Err(e)) => {
                return Err(AgentsError::other(format!(
                    "Responses websocket transport error: {e}"
                )));
            }
            None => return Ok(None),
        }
    }
}

/// Retry advice for a failed Responses WebSocket request.
/// Port of `OpenAIResponsesWSModel.get_retry_advice`.
pub fn websocket_retry_advice(
    request: &crate::retry::ModelRetryAdviceRequest<'_>,
) -> Option<ModelRetryAdvice> {
    let message = request.error.to_string();
    let reason = Some(message.clone());
    let stateful = request.previous_response_id.is_some() || request.conversation_id.is_some();
    let retry = |safety: Option<ReplaySafety>| ModelRetryAdvice {
        suggested: Some(true),
        replay_safety: safety,
        reason: reason.clone(),
        ..Default::default()
    };
    let dont_retry = |response_started: bool| ModelRetryAdvice {
        suggested: Some(false),
        replay_safety: Some(ReplaySafety::Unsafe),
        reason: reason.clone(),
        response_started,
        ..Default::default()
    };

    if message.contains(WS_CLOSE_INVALIDATED_MESSAGE) {
        return Some(ModelRetryAdvice {
            suggested: Some(false),
            reason,
            normalized: Some(ModelRetryNormalizedError {
                is_abort: true,
                ..Default::default()
            }),
            ..Default::default()
        });
    }
    if message.ends_with(RESPONSE_STARTED_SUFFIX) {
        return Some(dont_retry(true));
    }
    if message.starts_with(WS_PRE_EVENT_DISCONNECT_MESSAGE) {
        if message.ends_with(REQUEST_SENT_SUFFIX) {
            return Some(if stateful {
                dont_retry(false)
            } else {
                retry(None)
            });
        }
        return Some(retry(Some(ReplaySafety::Safe)));
    }
    if message.starts_with(WS_AMBIGUOUS_REPLAY_MESSAGE) {
        return Some(if stateful {
            dont_retry(false)
        } else {
            retry(None)
        });
    }
    if let Some(phase) = websocket_timeout_phase(&message) {
        if matches!(phase, "request lock wait" | "connect") {
            return Some(retry(Some(ReplaySafety::Safe)));
        }
        return Some(if stateful {
            dont_retry(false)
        } else {
            retry(None)
        });
    }
    if message.starts_with("Responses websocket send failed")
        || message.starts_with("Responses websocket connect failed")
    {
        return Some(retry(Some(ReplaySafety::Safe)));
    }
    if message.starts_with("Responses websocket error:")
        && let Some(payload) = ws_error_payload(&message)
    {
        let code = payload.pointer("/error/code").and_then(Value::as_str);
        let error_type = payload.pointer("/error/type").and_then(Value::as_str);
        if code == Some("server_is_overloaded")
            || (error_type == Some("server_error") && code.is_none())
        {
            return Some(retry(None));
        }
    }
    crate::retry::get_openai_retry_advice(request)
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[async_trait]
impl Model for OpenAIResponsesWSModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let span = response_span(request.tracing.is_disabled());
        span.start();
        let preserve = request.model_settings.preserve_raw_usage.unwrap_or(false);
        let result = async {
            let mut stream = self.open_stream(&request).await?;
            while let Some(ev) = stream.next().await {
                let ev = ev?;
                let ty = ev.get("type").and_then(Value::as_str).unwrap_or_default();
                if is_terminal(ty) {
                    let r = ev.get("response").cloned().unwrap_or(Value::Null);
                    return parse_response(&r, None, preserve);
                }
            }
            Err(AgentsError::other(
                "Responses websocket stream ended without a terminal event.",
            ))
        }
        .await;
        let include_data = request.tracing.include_data();
        match &result {
            Ok(resp) => finish_response_span(&span, resp, true),
            Err(e) => {
                record_model_error_on_span(&span, "Error getting response", e, include_data);
                crate::logger::log_model_action_error("Error getting response", e);
            }
        }
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let span = response_span(request.tracing.is_disabled());
        span.start();
        let include_data = request.tracing.include_data();
        let stream = match self.open_stream(&request).await {
            Ok(s) => s,
            Err(e) => {
                record_model_error_on_span(&span, "Error streaming response", &e, include_data);
                crate::logger::log_model_action_error("Error streaming response", &e);
                span.finish();
                return Err(e);
            }
        };
        let guard = SpanGuard(span);
        Ok(stream
            .map(move |ev| {
                let span = &guard.0;
                match ev {
                    Ok(event) => {
                        let ty = event
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if ty == "response.completed" {
                            if let Some(r) = event.get("response")
                                && let Ok(parsed) = parse_response(r, None, false)
                            {
                                finish_response_span(span, &parsed, true);
                            }
                            return Ok(event);
                        }
                        if is_terminal_failure_event(ty) {
                            let err = terminal_failure_error(ty, &event);
                            record_model_error_on_span(
                                span,
                                "Error streaming response",
                                &err,
                                include_data,
                            );
                            return Err(err);
                        }
                        Ok(event)
                    }
                    Err(e) => {
                        record_model_error_on_span(
                            span,
                            "Error streaming response",
                            &e,
                            include_data,
                        );
                        crate::logger::log_model_action_error("Error streaming response", &e);
                        Err(e)
                    }
                }
            })
            .boxed())
    }

    fn model_name(&self) -> Option<String> {
        Some(self.inner.model().to_string())
    }

    fn supports_default_prompt_cache_key(&self) -> bool {
        match &self.websocket_base_url {
            Some(url) => super::openai_client::is_official_openai_base_url(url, true),
            None => self.client.is_openai_api(),
        }
    }

    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        websocket_retry_advice(request)
    }

    async fn close(&self) {
        self.close_connection().await;
    }
}

/// Pins runs to a shared WebSocket-enabled OpenAI provider so consecutive turns and runs reuse
/// one connection per model. Port of `responses_websocket_session()`.
///
/// ```no_run
/// # async fn demo() -> openai_agents_sdk::Result<()> {
/// use openai_agents_sdk::models::openai_responses_ws::ResponsesWebSocketSession;
/// use openai_agents_sdk::{Agent, RunOptions, Runner};
///
/// let ws = ResponsesWebSocketSession::from_env();
/// let agent = Agent::<()>::new("Assistant");
/// let opts = RunOptions::new(()).run_config(ws.run_config());
/// let result = Runner::run_with_options(&agent, "Hello", opts).await?;
/// ws.close().await;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct ResponsesWebSocketSession {
    provider: super::openai_provider::OpenAIProvider,
}

impl ResponsesWebSocketSession {
    /// A session using `client`.
    pub fn new(client: OpenAIClient) -> Self {
        ResponsesWebSocketSession {
            provider: super::openai_provider::OpenAIProvider::new()
                .with_client(client)
                .with_use_responses(true)
                .with_responses_websocket(true),
        }
    }

    /// A session using the default client configuration (`OPENAI_API_KEY`, ...).
    pub fn from_env() -> Self {
        Self::new(OpenAIClient::from_env())
    }

    /// Set the low-level WebSocket connection options.
    pub fn with_websocket_options(mut self, options: OpenAIResponsesWebSocketOptions) -> Self {
        self.provider = self.provider.with_websocket_options(options);
        self
    }

    /// The WebSocket-enabled provider.
    pub fn provider(&self) -> &super::openai_provider::OpenAIProvider {
        &self.provider
    }

    /// A run config whose model provider is this session's provider.
    pub fn run_config<C: Send + Sync + 'static>(&self) -> crate::run_config::RunConfig<C> {
        crate::run_config::RunConfig::new().with_model_provider(Arc::new(self.provider.clone()))
    }

    /// Close cached models and their WebSocket connections.
    pub async fn close(&self) {
        self.provider.close_websocket_models().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::ModelRetryAdviceRequest;

    fn maybe_advice(error: &AgentsError, stateful: bool) -> Option<ModelRetryAdvice> {
        websocket_retry_advice(&ModelRetryAdviceRequest {
            error,
            attempt: 1,
            stream: true,
            previous_response_id: stateful.then_some("resp_1"),
            conversation_id: None,
        })
    }

    fn advice(error: &AgentsError, stateful: bool) -> ModelRetryAdvice {
        maybe_advice(error, stateful).expect("advice")
    }

    #[test]
    fn converts_urls() {
        assert_eq!(
            websocket_url("https://api.openai.com/v1/"),
            "wss://api.openai.com/v1"
        );
        assert_eq!(
            websocket_url("http://localhost:1/v1"),
            "ws://localhost:1/v1"
        );
        let m = OpenAIResponsesWSModel::new("gpt-x", OpenAIClient::new("sk-test"));
        let mut q = HashMap::new();
        q.insert("a b".to_string(), "c&d".to_string());
        assert_eq!(
            m.url(Some(&q)).unwrap(),
            "wss://api.openai.com/v1/responses?a%20b=c%26d"
        );
    }

    #[test]
    fn websocket_options_default_to_pythons() {
        let o = OpenAIResponsesWebSocketOptions::default();
        assert_eq!(o.ping_interval, Some(20.0));
        assert_eq!(o.ping_timeout, Some(20.0));
        assert_eq!(o.max_size, None);
        // Python's `max_size=None` is unlimited; tungstenite's default is 64 MiB.
        assert_eq!(o.websocket_config().max_message_size, None);
        let bounded = o.clone().with_max_size(Some(4096));
        assert_eq!(bounded.websocket_config().max_message_size, Some(4096));
        assert_eq!(
            o.with_ping_interval(None).with_ping_timeout(None),
            OpenAIResponsesWebSocketOptions {
                ping_interval: None,
                ping_timeout: None,
                max_size: None
            }
        );
    }

    #[test]
    fn derives_the_four_timeout_phases() {
        let t = RequestTimeouts::from_seconds(Some(2.5));
        assert_eq!(t.lock, Some(2.5));
        assert_eq!(t.recv, Some(2.5));
        assert_eq!(
            RequestTimeouts::from_seconds(None),
            RequestTimeouts::default()
        );
        for phase in TIMEOUT_PHASES {
            let err = timeout_error(phase, 1.0);
            assert_eq!(websocket_timeout_phase(&err.to_string()), Some(*phase));
        }
        assert_eq!(websocket_timeout_phase("something else"), None);
    }

    #[test]
    fn classifies_replay_safety() {
        // Closed before any event, frame already sent: stateless may replay, stateful may not.
        let pre_event = classify_stream_failure(
            AgentsError::other(WS_AMBIGUOUS_REPLAY_MESSAGE),
            false,
            false,
        );
        assert!(
            pre_event
                .to_string()
                .starts_with(WS_PRE_EVENT_DISCONNECT_MESSAGE)
        );
        assert_eq!(advice(&pre_event, false).suggested, Some(true));
        let stateful = advice(&pre_event, true);
        assert_eq!(stateful.suggested, Some(false));
        assert_eq!(stateful.replay_safety, Some(ReplaySafety::Unsafe));

        // Events already streamed: never replay.
        let started =
            classify_stream_failure(AgentsError::other(WS_AMBIGUOUS_REPLAY_MESSAGE), true, false);
        let a = advice(&started, false);
        assert_eq!(a.suggested, Some(false));
        assert!(a.response_started);

        // close() during the request aborts instead of retrying.
        let invalidated = classify_stream_failure(AgentsError::other("anything"), true, true);
        let a = advice(&invalidated, false);
        assert_eq!(a.suggested, Some(false));
        assert!(a.normalized.expect("normalized").is_abort);

        // Nothing was transmitted: safe to replay.
        let never_sent = AgentsError::other("Responses websocket send failed: broken pipe");
        let a = advice(&never_sent, true);
        assert_eq!(a.suggested, Some(true));
        assert_eq!(a.replay_safety, Some(ReplaySafety::Safe));

        // Timeouts: the phases before the frame is sent are safe.
        assert_eq!(
            advice(&timeout_error("connect", 1.0), true).replay_safety,
            Some(ReplaySafety::Safe)
        );
        assert_eq!(
            advice(&timeout_error("receive", 1.0), true).suggested,
            Some(false)
        );
        assert_eq!(
            advice(&timeout_error("receive", 1.0), false).suggested,
            Some(true)
        );
    }

    /// Pair two in-process WebSockets and run `body` with the client side.
    async fn paired_sockets() -> (Ws, Ws) {
        crate::util::ensure_crypto_provider();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            tokio_tungstenite::accept_async(tokio_tungstenite::MaybeTlsStream::Plain(stream))
                .await
                .unwrap()
        });
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (client, _) = tokio_tungstenite::client_async(
            format!("ws://{addr}/responses"),
            tokio_tungstenite::MaybeTlsStream::Plain(stream),
        )
        .await
        .unwrap();
        (client, server.await.unwrap())
    }

    #[test]
    fn the_keepalive_deadline_falls_back_to_the_interval() {
        let base = OpenAIResponsesWebSocketOptions::default();
        assert_eq!(
            keepalive_wait(&base, false),
            Some(Duration::from_secs_f64(20.0))
        );
        assert_eq!(
            keepalive_wait(&base, true),
            Some(Duration::from_secs_f64(20.0))
        );
        let quick = base.clone().with_ping_timeout(Some(1.0));
        assert_eq!(keepalive_wait(&quick, true), Some(Duration::from_secs(1)));
        // `ping_timeout: None` keeps pinging instead of waiting forever for a pong.
        let forever = base
            .clone()
            .with_ping_timeout(None)
            .with_ping_interval(Some(5.0));
        assert_eq!(keepalive_wait(&forever, true), Some(Duration::from_secs(5)));
        // Keepalive off: no deadline at all.
        assert_eq!(keepalive_wait(&base.with_ping_interval(None), false), None);
    }

    #[tokio::test]
    async fn pings_a_live_connection_while_the_read_is_pending() {
        let (mut client, mut server) = paired_sockets().await;
        let options = OpenAIResponsesWebSocketOptions::default()
            .with_ping_interval(Some(0.02))
            .with_ping_timeout(None);
        let pings = Arc::new(AtomicU64::new(0));
        let counted = pings.clone();
        // Keep the server reading so the connection stays open for the whole test.
        let server_task = tokio::spawn(async move {
            while let Some(Ok(message)) = server.next().await {
                if matches!(message, Message::Ping(_)) {
                    counted.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        let recv = tokio::time::timeout(
            Duration::from_millis(300),
            recv_next(&mut client, &options, None),
        )
        .await;
        assert!(recv.is_err(), "the receive must stay pending while pinging");
        assert!(
            pings.load(Ordering::SeqCst) >= 2,
            "expected repeated keepalive pings, saw {}",
            pings.load(Ordering::SeqCst)
        );
        drop(client);
        server_task.abort();
    }

    #[tokio::test]
    async fn an_unanswered_ping_fails_the_receive() {
        let (mut client, _server) = paired_sockets().await;
        let options = OpenAIResponsesWebSocketOptions::default()
            .with_ping_interval(Some(0.01))
            .with_ping_timeout(Some(0.01));
        let error = recv_next(&mut client, &options, None).await.unwrap_err();
        assert_eq!(
            websocket_timeout_phase(&error.to_string()),
            Some("receive"),
            "{error}"
        );
    }

    #[test]
    fn retries_overloaded_error_frames() {
        let err = ws_error(&json!({"type": "error", "error": {"code": "server_is_overloaded"}}));
        assert!(err.to_string().starts_with("Responses websocket error: {"));
        assert_eq!(advice(&err, true).suggested, Some(true));
        let other = ws_error(&json!({"type": "error", "error": {"code": "invalid_request"}}));
        assert_ne!(
            maybe_advice(&other, true).and_then(|a| a.suggested),
            Some(true)
        );
        let untyped = ws_error(&json!({"type": "error", "error": {"type": "server_error"}}));
        assert_eq!(advice(&untyped, false).suggested, Some(true));
    }
}

// -------------------------------------------------------------------------------------------
// WebSocket dependency log suppression
// -------------------------------------------------------------------------------------------

/// Log targets of the WebSocket stack this model is built on.
///
/// `log`'s macros default their target to the emitting module path, so every record from
/// `tungstenite` and `tokio-tungstenite` arrives under one of these roots.
pub const WEBSOCKET_LOG_TARGETS: [&str; 2] = ["tungstenite", "tokio_tungstenite"];

/// An `RUST_LOG` fragment that silences the WebSocket stack's verbose levels.
///
/// Useful when the application configures its logger from the environment rather than wrapping
/// it in [`WebSocketDebugFilter`]: `RUST_LOG=info,{WEBSOCKET_LOG_FILTER_DIRECTIVES}`.
pub const WEBSOCKET_LOG_FILTER_DIRECTIVES: &str = "tungstenite=info,tokio_tungstenite=info";

/// Whether a record from the WebSocket stack is one this SDK suppresses.
///
/// Mirrors `_OpenAIWebSocketLoggerAdapter.isEnabledFor`, which drops everything at or below
/// Python's `DEBUG` for the `websockets.client` logger. `tungstenite` splits the same material
/// across two Rust levels — connection and close-frame detail at `DEBUG`, and whole decoded
/// frames at `TRACE` (`tungstenite::protocol` logs `Received message {message}`) — so both are
/// covered. Levels at `INFO` and above pass through: suppressing those would hide real faults.
pub fn is_suppressed_websocket_record(target: &str, level: log::Level) -> bool {
    if level < log::Level::Debug {
        return false;
    }
    WEBSOCKET_LOG_TARGETS.iter().any(|root| {
        target == *root
            || target
                .strip_prefix(root)
                .is_some_and(|rest| rest.starts_with("::"))
    })
}

/// A `log::Log` wrapper that drops the WebSocket stack's `DEBUG` and `TRACE` records.
///
/// Those records carry connection detail and whole decoded frames, which for this model means
/// model input and output. Python suppresses the equivalent by swapping the `websockets.client`
/// logger for an adapter; in Rust only the installed logger can filter another crate's target,
/// and the installed logger belongs to the application — so the SDK ships the filter and the
/// application wraps its own logger in it:
///
/// ```no_run
/// use openai_agents_sdk::models::openai_responses_ws::WebSocketDebugFilter;
///
/// let inner = env_logger::Builder::from_default_env().build();
/// let level = inner.filter();
/// log::set_boxed_logger(Box::new(WebSocketDebugFilter::new(inner))).unwrap();
/// log::set_max_level(level);
/// ```
///
/// [`WEBSOCKET_LOG_FILTER_DIRECTIVES`] is the equivalent for an environment-configured logger.
#[derive(Debug, Clone)]
pub struct WebSocketDebugFilter<L> {
    inner: L,
}

impl<L> WebSocketDebugFilter<L> {
    /// Wrap `inner`, dropping the WebSocket stack's verbose records before it sees them.
    pub fn new(inner: L) -> Self {
        Self { inner }
    }

    /// The wrapped logger.
    pub fn inner(&self) -> &L {
        &self.inner
    }
}

impl<L: log::Log> log::Log for WebSocketDebugFilter<L> {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        if is_suppressed_websocket_record(metadata.target(), metadata.level()) {
            return false;
        }
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record<'_>) {
        if is_suppressed_websocket_record(record.target(), record.level()) {
            return;
        }
        self.inner.log(record);
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

#[cfg(test)]
mod websocket_log_suppression_tests {
    use super::*;
    use log::{Level, Log};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Collector(Arc<Mutex<Vec<String>>>);

    impl Log for Collector {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            self.0
                .lock()
                .unwrap()
                .push(format!("{} {}", record.target(), record.args()));
        }
        fn flush(&self) {}
    }

    fn emit(target: &str, level: Level, message: &str, filter: &dyn Log) {
        filter.log(
            &log::Record::builder()
                .target(target)
                .level(level)
                .args(format_args!("{message}"))
                .build(),
        );
    }

    #[test]
    fn the_websocket_stacks_verbose_records_are_dropped() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let filter = WebSocketDebugFilter::new(Collector(seen.clone()));

        // Frame payloads: `tungstenite::protocol` logs whole decoded messages at TRACE.
        emit(
            "tungstenite::protocol",
            Level::Trace,
            "Received message secret",
            &filter,
        );
        emit(
            "tungstenite::handshake::client",
            Level::Debug,
            "Client handshake done.",
            &filter,
        );
        emit(
            "tokio_tungstenite::compat",
            Level::Trace,
            "Read.read",
            &filter,
        );
        // A real fault from the same stack still gets through.
        emit("tungstenite::client", Level::Warn, "No `Location`", &filter);
        // Another crate that merely starts with the same letters is untouched.
        emit("tungstenite_extra", Level::Debug, "unrelated", &filter);
        // The SDK's own records are never WebSocket-stack records.
        emit("openai.agents", Level::Debug, "sdk detail", &filter);

        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![
                "tungstenite::client No `Location`".to_string(),
                "tungstenite_extra unrelated".to_string(),
                "openai.agents sdk detail".to_string(),
            ]
        );
    }

    #[test]
    fn enabled_reports_the_same_decision_as_log() {
        let filter = WebSocketDebugFilter::new(Collector::default());
        let suppressed = log::Metadata::builder()
            .target("tungstenite::protocol")
            .level(Level::Trace)
            .build();
        let kept = log::Metadata::builder()
            .target("tungstenite::protocol")
            .level(Level::Info)
            .build();
        assert!(!filter.enabled(&suppressed));
        assert!(filter.enabled(&kept));
    }
}
