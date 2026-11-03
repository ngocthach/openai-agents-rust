//! The Streamable HTTP client the `MCPServerStreamableHttp` transport runs on.
//!
//! `rmcp` builds its own `reqwest::Client` and exposes no session id after the transport is
//! handed to the service. [`ObservedHttpClient`] wraps the client Python's
//! `httpx_client_factory` would have produced and adds the three behaviours Python's parameters
//! control: it records the `Mcp-Session-Id` the server hands out
//! ([`crate::mcp::MCPServer::session_id`]), it can skip the session-ending `DELETE`
//! (`terminate_on_close`), and it can treat a rejected `notifications/initialized` as success
//! (`ignore_initialized_notification_failure`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reqwest::header::{HeaderName, HeaderValue};
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::common::client_side_sse::BoxedSseResponse;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};

/// The JSON-RPC method of the notification a strict server may reject.
const INITIALIZED_NOTIFICATION: &str = "notifications/initialized";

#[derive(Clone)]
pub(crate) struct ObservedHttpClient {
    inner: reqwest::Client,
    session_id: Arc<Mutex<Option<String>>>,
    terminate_on_close: bool,
    ignore_initialized_notification_failure: bool,
}

impl ObservedHttpClient {
    pub(crate) fn new(
        inner: reqwest::Client,
        session_id: Arc<Mutex<Option<String>>>,
        terminate_on_close: bool,
        ignore_initialized_notification_failure: bool,
    ) -> Self {
        ObservedHttpClient {
            inner,
            session_id,
            terminate_on_close,
            ignore_initialized_notification_failure,
        }
    }

    fn record(&self, session_id: Option<&String>) {
        if let Some(id) = session_id
            && let Ok(mut slot) = self.session_id.lock()
        {
            *slot = Some(id.clone());
        }
    }

    fn observe(&self, response: StreamableHttpPostResponse) -> StreamableHttpPostResponse {
        match &response {
            StreamableHttpPostResponse::Json(_, id) | StreamableHttpPostResponse::Sse(_, id) => {
                self.record(id.as_ref());
            }
            _ => {}
        }
        response
    }

    /// Whether this message is the `initialized` notification whose failure may be ignored.
    fn is_ignorable_initialized(&self, message: &ClientJsonRpcMessage) -> bool {
        self.ignore_initialized_notification_failure
            && serde_json::to_value(message)
                .ok()
                .and_then(|v| {
                    v.get("method")
                        .and_then(serde_json::Value::as_str)
                        .map(|m| m == INITIALIZED_NOTIFICATION)
                })
                .unwrap_or(false)
    }
}

impl StreamableHttpClient for ObservedHttpClient {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let ignorable = self.is_ignorable_initialized(&message);
        match self
            .inner
            .post_message(uri, message, session_id, auth_header, custom_headers)
            .await
        {
            Ok(response) => Ok(self.observe(response)),
            Err(e) if ignorable => {
                crate::logger::log_tool_action_warning(
                    "Ignoring failed MCP initialized notification",
                    &crate::error::AgentsError::other(super::logging::redact_credentials(
                        &e.to_string(),
                    )),
                );
                Ok(StreamableHttpPostResponse::Accepted)
            }
            Err(e) => Err(e),
        }
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let ignorable = self.is_ignorable_initialized(&message);
        match self
            .inner
            .post_message_with_max_sse_event_size(
                uri,
                message,
                session_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
        {
            Ok(response) => Ok(self.observe(response)),
            Err(e) if ignorable => {
                crate::logger::log_tool_action_warning(
                    "Ignoring failed MCP initialized notification",
                    &crate::error::AgentsError::other(super::logging::redact_credentials(
                        &e.to_string(),
                    )),
                );
                Ok(StreamableHttpPostResponse::Accepted)
            }
            Err(e) => Err(e),
        }
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        if !self.terminate_on_close {
            return Ok(());
        }
        self.inner
            .delete_session(uri, session_id, auth_header, custom_headers)
            .await
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxedSseResponse, StreamableHttpError<Self::Error>> {
        self.inner
            .get_stream(uri, session_id, last_event_id, auth_header, custom_headers)
            .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxedSseResponse, StreamableHttpError<Self::Error>> {
        self.inner
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
    }
}
