//! PTY sessions over the Blaxel sandbox's terminal WebSocket. Mirrors the `pty_exec_start` /
//! `pty_write_stdin` pair in `agents.extensions.sandbox.blaxel.sandbox`.
//!
//! The sandbox serves a terminal at
//! `wss://{sandbox}/terminal/ws?token=&cols=&rows=&sessionId=&workingDir=`. Both directions are
//! JSON text frames: the SDK sends `{"type":"input","data":…}` and the sandbox answers with
//! `{"type":"output","data":…}`, or `{"type":"error","data":…}` when the terminal fails. The
//! terminal reports no exit code, so a finished session simply ends its stream.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::sync::Mutex as AsyncMutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::pty::{
    PtyExecRequest, PtyExecUpdate, PtyWriteRequest, RemotePtyManager, RemotePtyProcess,
    clamp_pty_yield_time_ms, resolve_pty_write_yield_time_ms,
};
use crate::sandbox::session::shell_quote;

type WebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WebSocketSink = futures::stream::SplitSink<WebSocket, Message>;

/// The writable half of one terminal, plus the task pumping its output.
struct PtyTransport {
    sink: AsyncMutex<WebSocketSink>,
    pump: tokio::task::JoinHandle<()>,
}

impl PtyTransport {
    async fn close(&self) {
        let mut sink = self.sink.lock().await;
        let _ = sink.send(Message::Close(None)).await;
        let _ = sink.close().await;
        self.pump.abort();
    }
}

/// The live Blaxel terminals of one session.
#[derive(Default)]
pub struct BlaxelPtySessions {
    processes: RemotePtyManager,
    transports: Mutex<HashMap<u32, Arc<PtyTransport>>>,
}

impl std::fmt::Debug for BlaxelPtySessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlaxelPtySessions")
            .field("live", &self.processes.len())
            .finish()
    }
}

impl BlaxelPtySessions {
    /// An empty set of terminals.
    pub fn new() -> Self {
        BlaxelPtySessions::default()
    }

    /// Open a terminal, run `request`'s command in it, and collect its first output window.
    pub async fn start(
        &self,
        ws_url: &str,
        request: &PtyExecRequest,
        connect_timeout: Duration,
    ) -> SandboxResult<PtyExecUpdate> {
        let command = request
            .resolved_argv()
            .iter()
            .map(|part| shell_quote(part))
            .collect::<Vec<_>>()
            .join(" ");

        let socket = match tokio::time::timeout(connect_timeout, connect(ws_url)).await {
            Ok(socket) => socket?,
            Err(_) => {
                return Err(SandboxError::new(
                    ErrorCode::ExecTimeout,
                    format!("Command `{command}` timed out opening the terminal"),
                ));
            }
        };
        let (mut sink, stream) = socket.split();
        sink.send(input_frame(&format!("{command}\n")))
            .await
            .map_err(|error| transport_error(format!("failed to send the PTY command: {error}")))?;

        // The terminal always accepts input, so the session is writable regardless of `tty`.
        let (process, pruned) = self.processes.register(true)?;
        let pump = tokio::spawn(pump_output(stream, process.clone()));
        let transport = Arc::new(PtyTransport {
            sink: AsyncMutex::new(sink),
            pump,
        });
        self.transports
            .lock()
            .expect("poisoned")
            .insert(process.id(), transport);
        self.close_pruned(&pruned).await;

        let update = process
            .collect(
                clamp_pty_yield_time_ms(request.yield_time_ms),
                request.max_output_tokens,
            )
            .await;
        self.settle(&process, &update).await;
        Ok(update)
    }

    /// Write to (or poll) a live terminal.
    pub async fn write_stdin(&self, request: &PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let process = self.processes.get(request.session_id)?;
        if !request.chars.is_empty() {
            let transport = self
                .transports
                .lock()
                .expect("poisoned")
                .get(&request.session_id)
                .cloned()
                .ok_or_else(|| {
                    SandboxError::new(
                        ErrorCode::PtySessionNotFound,
                        format!("PTY session `{}` is not running", request.session_id),
                    )
                })?;
            transport
                .sink
                .lock()
                .await
                .send(input_frame(&request.chars))
                .await
                .map_err(|error| transport_error(format!("failed to write to the PTY: {error}")))?;
            // Give the terminal a moment to echo before the window starts.
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let update = process
            .collect(
                resolve_pty_write_yield_time_ms(request.yield_time_ms, request.chars.is_empty()),
                request.max_output_tokens,
            )
            .await;
        self.settle(&process, &update).await;
        Ok(update)
    }

    /// Close every live terminal.
    pub async fn terminate_all(&self) {
        let processes = self.processes.take_all();
        let transports: Vec<Arc<PtyTransport>> = {
            let mut guard = self.transports.lock().expect("poisoned");
            processes
                .iter()
                .filter_map(|process| guard.remove(&process.id()))
                .collect()
        };
        for transport in transports {
            transport.close().await;
        }
    }

    async fn settle(&self, process: &Arc<RemotePtyProcess>, update: &PtyExecUpdate) {
        if update.process_id.is_some() {
            return;
        }
        self.processes.remove(process.id());
        let transport = self
            .transports
            .lock()
            .expect("poisoned")
            .remove(&process.id());
        if let Some(transport) = transport {
            transport.close().await;
        }
    }

    async fn close_pruned(&self, pruned: &[Arc<RemotePtyProcess>]) {
        for process in pruned {
            let transport = self
                .transports
                .lock()
                .expect("poisoned")
                .remove(&process.id());
            if let Some(transport) = transport {
                transport.close().await;
            }
        }
    }
}

/// The terminal URL for a sandbox.
///
/// The query values are percent-encoded: the workspace path and session id are caller-controlled
/// and may hold characters that are structural in a query string, so interpolating them raw would
/// let a path such as `/w/a&rows=1` add or override parameters and let a `#` truncate the rest
/// into a fragment. `/` stays literal because it is legal in a query value and keeps paths
/// readable.
pub fn terminal_url(
    sandbox_url: &str,
    token: &str,
    session_id: &str,
    working_dir: &str,
    cols: u16,
    rows: u16,
) -> String {
    let base = sandbox_url.trim_end_matches('/');
    // Only the scheme is rewritten: replacing every occurrence would also rewrite one inside the
    // path, such as a proxied URL.
    let ws_base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    format!(
        "{ws_base}/terminal/ws?token={}&cols={cols}&rows={rows}&sessionId={}&workingDir={}",
        encode_query_value(token),
        encode_query_value(session_id),
        encode_query_value(working_dir),
    )
}

/// Percent-encode a query value, keeping `/` literal.
fn encode_query_value(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                encoded.push(*byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn input_frame(data: &str) -> Message {
    Message::Text(json!({"type": "input", "data": data}).to_string().into())
}

async fn connect(ws_url: &str) -> SandboxResult<WebSocket> {
    let request = ws_url
        .into_client_request()
        .map_err(|error| transport_error(format!("invalid terminal URL: {error}")))?;
    let (socket, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|error| {
            transport_error(format!("failed to open the terminal WebSocket: {error}"))
        })?;
    Ok(socket)
}

/// Feed the terminal's frames into `process` until the sandbox ends the stream.
async fn pump_output(
    mut stream: futures::stream::SplitStream<WebSocket>,
    process: Arc<RemotePtyProcess>,
) {
    while let Some(message) = stream.next().await {
        let text = match message {
            Ok(Message::Text(text)) => text.to_string(),
            Ok(Message::Binary(bytes)) => String::from_utf8_lossy(&bytes).into_owned(),
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) | Ok(Message::Frame(_)) | Err(_) => break,
        };
        let Ok(payload) = serde_json::from_str::<Value>(&text) else {
            log::debug!("ignoring a malformed Blaxel PTY frame");
            continue;
        };
        // Older sandbox builds capitalize the field names.
        let frame_type = payload
            .get("type")
            .or_else(|| payload.get("Type"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let data = payload
            .get("data")
            .or_else(|| payload.get("Data"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        match frame_type {
            "output" => process.push_output(data.as_bytes()),
            "error" => {
                process.push_output(data.as_bytes());
                process.close(None);
                return;
            }
            _ => continue,
        }
    }
    process.close(None);
}

fn transport_error(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::ExecTransportError, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_terminal_url() {
        assert_eq!(
            terminal_url(
                "https://sbx.example/",
                "tok en",
                "pty-1",
                "/workspace",
                80,
                24
            ),
            "wss://sbx.example/terminal/ws?token=tok%20en&cols=80&rows=24&sessionId=pty-1\
             &workingDir=/workspace"
        );
    }

    #[test]
    fn encodes_query_values_that_would_add_parameters() {
        // A working directory holding `&` or `#` must not be able to add a parameter or truncate
        // the rest of the query.
        let url = terminal_url("http://host", "t+1", "s", "/w/a&rows=1#x", 80, 24);
        assert!(url.starts_with("ws://host/terminal/ws?token=t%2B1"));
        assert!(url.ends_with("&workingDir=/w/a%26rows%3D1%23x"));
    }

    /// A sandbox terminal that answers each input frame with one output frame.
    async fn mock_terminal(close_after: bool) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
            let mut received = Vec::new();
            while let Some(Ok(message)) = ws.next().await {
                match message {
                    Message::Text(text) => {
                        received.push(text.to_string());
                        ws.send(Message::Text(
                            r#"{"type":"output","data":"out: hi\r\n"}"#.into(),
                        ))
                        .await
                        .unwrap();
                        if close_after {
                            ws.close(None).await.ok();
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    _ => continue,
                }
            }
            received
        });
        (
            format!("ws://{address}/terminal/ws?token=t&cols=80&rows=24"),
            handle,
        )
    }

    #[tokio::test]
    async fn runs_a_command_and_finishes_when_the_terminal_closes() {
        let (url, server) = mock_terminal(true).await;
        let sessions = BlaxelPtySessions::new();
        let update = sessions
            .start(
                &url,
                &PtyExecRequest::new("echo hi").with_yield_time_ms(400),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(update.output_text(), "out: hi\r\n");
        // The terminal reports no exit code, so a closed stream only ends the session.
        assert_eq!(update.process_id, None);
        assert_eq!(update.exit_code, None);
        assert!(sessions.processes.is_empty());
        let sent = server.await.unwrap();
        assert_eq!(
            sent,
            vec![r#"{"data":"sh -lc 'echo hi'\n","type":"input"}"#]
        );
    }

    #[tokio::test]
    async fn writes_stdin_to_a_live_terminal() {
        let (url, server) = mock_terminal(false).await;
        let sessions = BlaxelPtySessions::new();
        let update = sessions
            .start(
                &url,
                &PtyExecRequest::new("cat").with_yield_time_ms(400),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        let session_id = update.process_id.expect("the terminal is still running");

        let update = sessions
            .write_stdin(&PtyWriteRequest::new(session_id, "typed\n").with_yield_time_ms(400))
            .await
            .unwrap();
        assert_eq!(update.output_text(), "out: hi\r\n");
        assert_eq!(
            sessions
                .write_stdin(&PtyWriteRequest::new(1, "x"))
                .await
                .unwrap_err()
                .code,
            ErrorCode::PtySessionNotFound
        );

        sessions.terminate_all().await;
        assert!(sessions.processes.is_empty());
        let sent = server.await.unwrap();
        assert!(sent.iter().any(|frame| frame.contains("typed")), "{sent:?}");
    }

    #[tokio::test]
    async fn reports_a_terminal_that_cannot_be_reached() {
        let sessions = BlaxelPtySessions::new();
        // Port 1 on the loopback interface refuses the connection immediately.
        let error = sessions
            .start(
                "ws://127.0.0.1:1/terminal/ws",
                &PtyExecRequest::new("echo hi"),
                Duration::from_secs(5),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::ExecTransportError);
    }
}
