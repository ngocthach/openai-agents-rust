//! The OpenAI Responses API over a persistent WebSocket (`response.create` frames).
//!
//! Port of `OpenAIResponsesWSModel`. One connection is kept open and reused across requests,
//! which lowers per-turn latency for agent loops. Requests on the same model are serialized.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use super::interface::{Model, ModelRequest, ResponseEventStream};
use super::openai_client::OpenAIClient;
use super::openai_responses::{
    OpenAIResponsesModel, SpanGuard, finish_response_span, parse_response,
};
use super::response_terminal::{is_terminal_failure_event, terminal_failure_error};
use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::tracing::{record_model_error_on_span, response_span};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct Connection {
    identity: String,
    socket: Ws,
}

/// A Responses API model that talks to OpenAI over a WebSocket.
#[derive(Clone)]
pub struct OpenAIResponsesWSModel {
    inner: OpenAIResponsesModel,
    client: OpenAIClient,
    websocket_base_url: Option<String>,
    connection: Arc<Mutex<Option<Connection>>>,
}

impl std::fmt::Debug for OpenAIResponsesWSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesWSModel")
            .field("model", &self.inner.model())
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

fn ws_error(payload: &Value) -> AgentsError {
    let err = payload.get("error").unwrap_or(payload);
    let message = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Responses websocket error");
    let code = err.get("code").and_then(Value::as_str);
    AgentsError::other(match code {
        Some(c) => format!("Responses websocket error ({c}): {message}"),
        None => format!("Responses websocket error: {message}"),
    })
}

impl OpenAIResponsesWSModel {
    /// Create a WebSocket Responses model.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAIResponsesWSModel {
            inner: OpenAIResponsesModel::new(model, client.clone()),
            client,
            websocket_base_url: None,
            connection: Arc::new(Mutex::new(None)),
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
        if let Some(q) = extra_query.filter(|q| !q.is_empty()) {
            let mut pairs: Vec<_> = q.iter().collect();
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
        let (socket, _) = tokio_tungstenite::connect_async(req)
            .await
            .map_err(|e| AgentsError::other(format!("Responses websocket connect failed: {e}")))?;
        Ok(socket)
    }

    /// Close the cached connection.
    pub async fn close_connection(&self) {
        if let Some(mut c) = self.connection.lock().await.take() {
            let _ = c.socket.close(None).await;
        }
    }

    async fn open_stream(&self, request: &ModelRequest) -> Result<ResponseEventStream> {
        let frame = self.build_frame(request)?;
        let settings = &request.model_settings;
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

        let mut guard = self.connection.clone().lock_owned().await;
        // A cached connection may have been closed by the server; reconnect once if the frame
        // could not be sent on it (the request was never transmitted, so replay is safe).
        let mut attempts = 0;
        loop {
            attempts += 1;
            let reuse = guard.as_ref().is_some_and(|c| c.identity == identity);
            if !reuse {
                if let Some(mut old) = guard.take() {
                    let _ = old.socket.close(None).await;
                }
                let socket = self.connect(&url, settings.extra_headers.as_ref()).await?;
                *guard = Some(Connection {
                    identity: identity.clone(),
                    socket,
                });
            }
            let conn = guard.as_mut().expect("connection");
            match conn.socket.send(Message::text(text.clone())).await {
                Ok(()) => break,
                Err(e) => {
                    *guard = None;
                    if !reuse || attempts > 1 {
                        return Err(AgentsError::other(format!(
                            "Responses websocket send failed: {e}"
                        )));
                    }
                }
            }
        }

        let stream = futures::stream::unfold(Some(guard), |state| async move {
            let mut guard = state?;
            loop {
                let conn = guard.as_mut()?;
                let next = conn.socket.next().await;
                let frame = match next {
                    None => {
                        *guard = None;
                        return Some((
                            Err(AgentsError::other(
                                "Responses websocket connection closed before a terminal \
                                 response event.",
                            )),
                            None,
                        ));
                    }
                    Some(Err(e)) => {
                        *guard = None;
                        return Some((
                            Err(AgentsError::other(format!(
                                "Responses websocket error: {e}"
                            ))),
                            None,
                        ));
                    }
                    Some(Ok(m)) => m,
                };
                let text = match frame {
                    Message::Text(t) => t.to_string(),
                    Message::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                    Message::Close(_) => {
                        *guard = None;
                        return Some((
                            Err(AgentsError::other(
                                "Responses websocket connection closed before a terminal \
                                 response event.",
                            )),
                            None,
                        ));
                    }
                    _ => continue,
                };
                let payload: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        *guard = None;
                        return Some((Err(e.into()), None));
                    }
                };
                let ty = payload
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if ty == "error" || ty == "response.error" {
                    *guard = None;
                    return Some((Err(ws_error(&payload)), None));
                }
                // The connection stays cached for the next request after a terminal event.
                let next_state = if is_terminal(&ty) { None } else { Some(guard) };
                return Some((Ok(payload), next_state));
            }
        });
        Ok(stream.boxed())
    }
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

    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        crate::retry::get_openai_retry_advice(request)
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
}
