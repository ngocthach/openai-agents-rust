//! The legacy MCP HTTP+SSE client transport (deprecated by the MCP project, kept for parity
//! with `MCPServerSse`). The client opens an SSE stream with `GET`, receives an `endpoint`
//! event naming the URL to `POST` JSON-RPC messages to, and reads replies as `message` events.

use std::collections::HashMap;

use futures::{Sink, Stream, StreamExt};
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};

use super::logging::{get_mcp_server_log_name, redact_credentials};
use crate::error::{AgentsError, Result};

/// One parsed SSE event.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub event: String,
    pub data: String,
}

/// Parse a byte stream into SSE events.
pub(crate) fn sse_events<S, B, E>(
    bytes: S,
) -> impl Stream<Item = std::result::Result<SseEvent, String>>
where
    S: Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    struct State<S> {
        inner: std::pin::Pin<Box<S>>,
        buf: String,
        event: SseEvent,
        has_data: bool,
        done: bool,
    }
    futures::stream::unfold(
        State {
            inner: Box::pin(bytes),
            buf: String::new(),
            event: SseEvent::default(),
            has_data: false,
            done: false,
        },
        |mut st| async move {
            loop {
                while let Some(pos) = st.buf.find('\n') {
                    let line: String = st.buf.drain(..=pos).collect();
                    let line = line.trim_end_matches(['\n', '\r']);
                    if line.is_empty() {
                        if st.has_data || !st.event.event.is_empty() {
                            let mut ev = std::mem::take(&mut st.event);
                            st.has_data = false;
                            if ev.event.is_empty() {
                                ev.event = "message".into();
                            }
                            return Some((Ok(ev), st));
                        }
                        continue;
                    }
                    if line.starts_with(':') {
                        continue;
                    }
                    let (field, value) = match line.split_once(':') {
                        Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
                        None => (line, ""),
                    };
                    match field {
                        "event" => st.event.event = value.to_string(),
                        "data" => {
                            if st.has_data {
                                st.event.data.push('\n');
                            }
                            st.event.data.push_str(value);
                            st.has_data = true;
                        }
                        _ => {}
                    }
                }
                if st.done {
                    return None;
                }
                match st.inner.next().await {
                    Some(Ok(chunk)) => st.buf.push_str(&String::from_utf8_lossy(chunk.as_ref())),
                    Some(Err(e)) => {
                        st.done = true;
                        return Some((Err(e.to_string()), st));
                    }
                    None => {
                        st.done = true;
                        st.buf.push('\n');
                    }
                }
            }
        },
    )
}

fn build_headers(headers: &HashMap<String, String>) -> Result<reqwest::header::HeaderMap> {
    let mut map = reqwest::header::HeaderMap::new();
    for (k, v) in headers {
        let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| AgentsError::user(format!("Invalid header {k}: {e}")))?;
        let value = reqwest::header::HeaderValue::from_str(v)
            .map_err(|e| AgentsError::user(format!("Invalid header value for {k}: {e}")))?;
        map.insert(name, value);
    }
    Ok(map)
}

/// Open the SSE stream and return a `(sink, stream)` pair usable as an rmcp transport.
pub(crate) async fn connect(
    params: &super::MCPServerSseParams,
) -> Result<(
    impl Sink<ClientJsonRpcMessage, Error = std::io::Error> + Send + Unpin + 'static,
    impl Stream<Item = ServerJsonRpcMessage> + Send + Unpin + 'static,
)> {
    let url = params.url.as_str();
    let timeout = params.timeout;
    let mut header_map = build_headers(&params.headers)?;
    if let Some(auth) = &params.auth
        && let Some(value) = auth()
    {
        let value = reqwest::header::HeaderValue::from_str(&format!(
            "Bearer {}",
            super::bearer_token(&value)
        ))
        .map_err(|e| AgentsError::user(format!("Invalid MCP auth header value: {e}")))?;
        header_map.insert(reqwest::header::AUTHORIZATION, value);
    }
    // `httpx_client_factory`'s counterpart: a caller-built client replaces the default one.
    let client = match &params.http_client {
        Some(c) => c.clone(),
        None => {
            crate::util::ensure_crypto_provider();
            reqwest::Client::builder()
                .connect_timeout(timeout)
                .read_timeout(params.sse_read_timeout)
                .build()?
        }
    };
    // A reqwest error names the request URL, which is where an MCP credential lives.
    let resp = client
        .get(url)
        .headers(header_map.clone())
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .send()
        .await
        .map_err(|e| AgentsError::other(redact_credentials(&format!("MCP SSE error: {e}"))))?;
    if !resp.status().is_success() {
        return Err(AgentsError::other(format!(
            "MCP SSE connection failed with status {}",
            resp.status()
        )));
    }
    let mut events = Box::pin(sse_events(resp.bytes_stream()));
    let base = reqwest::Url::parse(url).map_err(|e| {
        AgentsError::user(format!(
            "Invalid MCP SSE URL {}: {e}",
            get_mcp_server_log_name(url)
        ))
    })?;
    let endpoint = loop {
        let next = tokio::time::timeout(timeout, events.next())
            .await
            .map_err(|_| AgentsError::other("Timed out waiting for the MCP SSE endpoint event"))?;
        match next {
            Some(Ok(ev)) if ev.event == "endpoint" => {
                let ep = base.join(ev.data.trim()).map_err(|e| {
                    AgentsError::other(redact_credentials(&format!(
                        "Invalid MCP SSE endpoint {}: {e}",
                        ev.data
                    )))
                })?;
                if ep.origin() != base.origin() {
                    return Err(AgentsError::other(format!(
                        "MCP SSE endpoint origin does not match the connection origin: {}",
                        get_mcp_server_log_name(ep.as_str())
                    )));
                }
                break ep;
            }
            Some(Ok(_)) => continue,
            Some(Err(e)) => {
                return Err(AgentsError::other(redact_credentials(&format!(
                    "MCP SSE error: {e}"
                ))));
            }
            None => {
                return Err(AgentsError::other(
                    "MCP SSE stream closed before the endpoint event",
                ));
            }
        }
    };

    let stream = events
        .filter_map(|ev| async move {
            match ev {
                Ok(ev) if ev.event == "message" => match serde_json::from_str(&ev.data) {
                    Ok(msg) => Some(msg),
                    Err(e) => {
                        log::warn!(target: crate::logger::LOG_TARGET, "Ignoring invalid MCP SSE message: {e}");
                        None
                    }
                },
                Ok(_) => None,
                Err(e) => {
                    log::warn!(target: crate::logger::LOG_TARGET, "MCP SSE stream error: {e}");
                    None
                }
            }
        })
        .boxed();

    let sink = futures::sink::unfold(
        (client, endpoint, header_map),
        |(client, endpoint, headers), msg: ClientJsonRpcMessage| async move {
            let resp = client
                .post(endpoint.clone())
                .headers(headers.clone())
                .json(&msg)
                .send()
                .await
                .map_err(|e| {
                    std::io::Error::other(redact_credentials(&format!("MCP SSE POST failed: {e}")))
                })?;
            if !resp.status().is_success() {
                return Err(std::io::Error::other(format!(
                    "MCP SSE POST failed with status {}",
                    resp.status()
                )));
            }
            Ok((client, endpoint, headers))
        },
    );
    Ok((Box::pin(sink), stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parses_sse_events() {
        let chunks: Vec<std::result::Result<&[u8], String>> = vec![
            Ok(b"event: endpoint\ndata: /messages?session=1\n\n: comment\n"),
            Ok(b"data: {\"a\":\n"),
            Ok(b"data: 1}\n\nevent: message\r\ndata: x\r\n\r\n"),
        ];
        let events: Vec<_> = sse_events(futures::stream::iter(chunks))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|e| e.unwrap())
            .collect();
        assert_eq!(events[0].event, "endpoint");
        assert_eq!(events[0].data, "/messages?session=1");
        assert_eq!(events[1].event, "message");
        assert_eq!(events[1].data, "{\"a\":\n1}");
        assert_eq!(events[2].data, "x");
    }
}
