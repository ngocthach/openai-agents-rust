//! PTY sessions over the Cloudflare worker's WebSocket endpoint. Mirrors the
//! `pty_exec_start` / `pty_write_stdin` pair in
//! `agents.extensions.sandbox.cloudflare.sandbox`.
//!
//! The worker serves a terminal at `wss://{worker}/v1/sandbox/{id}/pty?cols=&rows=`. It opens
//! with a `{"type":"ready"}` text frame, carries the terminal's output as binary frames, and
//! ends with `{"type":"exit","code":N}` or `{"type":"error","message":…}`. The command itself,
//! and anything the model later types, is written back as raw bytes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::pty::{
    PtyExecRequest, PtyExecUpdate, PtyWriteRequest, RemotePtyManager, RemotePtyProcess,
    clamp_pty_yield_time_ms, resolve_pty_write_yield_time_ms,
};
use crate::sandbox::session::shell_quote;

/// How long the worker is given to send its `ready` frame.
pub const PTY_READY_TIMEOUT: Duration = Duration::from_secs(30);

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

/// The live Cloudflare terminals of one session.
#[derive(Default)]
pub struct CloudflarePtySessions {
    processes: RemotePtyManager,
    transports: Mutex<HashMap<u32, Arc<PtyTransport>>>,
}

impl std::fmt::Debug for CloudflarePtySessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflarePtySessions")
            .field("live", &self.processes.len())
            .finish()
    }
}

impl CloudflarePtySessions {
    /// An empty set of terminals.
    pub fn new() -> Self {
        CloudflarePtySessions::default()
    }

    /// Open a terminal, run `request`'s command in it, and collect its first output window.
    pub async fn start(
        &self,
        ws_url: &str,
        api_key: Option<&str>,
        request: &PtyExecRequest,
        cwd: Option<&str>,
    ) -> SandboxResult<PtyExecUpdate> {
        let mut command = request
            .resolved_argv()
            .iter()
            .map(|part| shell_quote(part))
            .collect::<Vec<_>>()
            .join(" ");
        if let Some(cwd) = cwd {
            command = format!("cd -- {} && {command}", shell_quote(cwd));
        }

        let mut socket = connect(ws_url, api_key).await?;
        await_ready(&mut socket, &command).await?;
        let (mut sink, stream) = socket.split();
        // The worker runs whatever is written to the terminal, so the command is the first thing
        // it receives.
        sink.send(Message::Binary(format!("{command}\n").into_bytes().into()))
            .await
            .map_err(|error| transport_error(format!("failed to send the PTY command: {error}")))?;

        let (process, pruned) = self.processes.register(request.tty)?;
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
            if !process.tty() {
                return Err(SandboxError::new(
                    ErrorCode::PtyStdinUnavailable,
                    "stdin is not available for this process",
                ));
            }
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
                .send(Message::Binary(request.chars.clone().into_bytes().into()))
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

    /// Release a terminal whose process has finished.
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

/// The worker's terminal URL for a sandbox.
pub fn pty_url(worker_url: &str, sandbox_id: &str, cols: u16, rows: u16) -> String {
    let base = worker_url.trim_end_matches('/');
    // Only the scheme is rewritten: replacing every occurrence would also rewrite one inside the
    // path, such as a proxied URL.
    let ws_base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    format!("{ws_base}/v1/sandbox/{sandbox_id}/pty?cols={cols}&rows={rows}")
}

async fn connect(ws_url: &str, api_key: Option<&str>) -> SandboxResult<WebSocket> {
    let mut request = ws_url
        .into_client_request()
        .map_err(|error| transport_error(format!("invalid PTY URL `{ws_url}`: {error}")))?;
    if let Some(key) = api_key {
        let value = format!("Bearer {key}").parse().map_err(|_| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "the Cloudflare API key cannot be sent as a header",
            )
        })?;
        request.headers_mut().insert("Authorization", value);
    }
    let (socket, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|error| transport_error(format!("failed to open the PTY WebSocket: {error}")))?;
    Ok(socket)
}

/// Wait for the worker's `ready` frame before sending anything.
async fn await_ready(socket: &mut WebSocket, command: &str) -> SandboxResult<()> {
    let ready = tokio::time::timeout(PTY_READY_TIMEOUT, async {
        while let Some(message) = socket.next().await {
            match message {
                Ok(Message::Text(text)) => {
                    if let Ok(payload) = serde_json::from_str::<Value>(&text)
                        && payload.get("type").and_then(Value::as_str) == Some("ready")
                    {
                        return Ok(());
                    }
                }
                Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
                Ok(Message::Close(_)) | Ok(Message::Frame(_)) => break,
                Err(error) => {
                    return Err(transport_error(format!(
                        "the PTY WebSocket failed before it was ready: {error}"
                    )));
                }
            }
        }
        Err(transport_error("WebSocket closed before PTY ready"))
    })
    .await;
    match ready {
        Ok(result) => result,
        Err(_) => Err(SandboxError::new(
            ErrorCode::ExecTimeout,
            format!("Command `{command}` timed out waiting for the PTY to become ready"),
        )),
    }
}

/// Feed the terminal's frames into `process` until the worker ends the stream.
async fn pump_output(
    mut stream: futures::stream::SplitStream<WebSocket>,
    process: Arc<RemotePtyProcess>,
) {
    while let Some(message) = stream.next().await {
        match message {
            Ok(Message::Binary(bytes)) => process.push_output(&bytes),
            Ok(Message::Text(text)) => {
                let Ok(payload) = serde_json::from_str::<Value>(&text) else {
                    log::debug!("ignoring a non-JSON Cloudflare PTY text frame");
                    continue;
                };
                match payload.get("type").and_then(Value::as_str) {
                    Some("ready") => continue,
                    Some("exit") => {
                        process.close(
                            payload
                                .get("code")
                                .and_then(Value::as_i64)
                                .map(|code| code as i32),
                        );
                        return;
                    }
                    Some("error") => {
                        log::warn!("Cloudflare PTY error frame");
                        process.close(None);
                        return;
                    }
                    _ => continue,
                }
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) | Ok(Message::Frame(_)) | Err(_) => break,
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
            pty_url("https://worker.example/", "sbx-1", 80, 24),
            "wss://worker.example/v1/sandbox/sbx-1/pty?cols=80&rows=24"
        );
        assert_eq!(
            pty_url("http://127.0.0.1:8787", "sbx-2", 100, 40),
            "ws://127.0.0.1:8787/v1/sandbox/sbx-2/pty?cols=100&rows=40"
        );
        // Only the scheme is rewritten, never an occurrence inside the path.
        assert_eq!(
            pty_url("https://proxy.example/https://inner", "s", 80, 24),
            "wss://proxy.example/https://inner/v1/sandbox/s/pty?cols=80&rows=24"
        );
    }

    /// A worker terminal that greets, echoes the command and exits.
    async fn mock_terminal(exit_code: Option<i32>) -> (String, tokio::task::JoinHandle<Vec<u8>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
            ws.send(Message::Text(r#"{"type":"ready"}"#.into()))
                .await
                .unwrap();
            let mut received = Vec::new();
            while let Some(Ok(message)) = ws.next().await {
                match message {
                    Message::Binary(bytes) => {
                        received.extend_from_slice(&bytes);
                        ws.send(Message::Binary(b"out: hi\r\n".to_vec().into()))
                            .await
                            .unwrap();
                        if let Some(code) = exit_code {
                            ws.send(Message::Text(json_exit(code).into()))
                                .await
                                .unwrap();
                        }
                        if exit_code.is_some() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    _ => continue,
                }
            }
            received
        });
        (format!("ws://{address}/v1/sandbox/s/pty"), handle)
    }

    fn json_exit(code: i32) -> String {
        format!("{{\"type\":\"exit\",\"code\":{code}}}")
    }

    #[tokio::test]
    async fn runs_a_command_and_reports_the_exit_code() {
        let (url, server) = mock_terminal(Some(3)).await;
        let sessions = CloudflarePtySessions::new();
        let update = sessions
            .start(
                &url,
                Some("cf-token"),
                &PtyExecRequest::new("echo hi")
                    .with_tty(true)
                    .with_yield_time_ms(2000),
                Some("/workspace/repo"),
            )
            .await
            .unwrap();
        assert_eq!(update.output_text(), "out: hi\r\n");
        // The process finished, so the model gets no id to write to and sees the exit code.
        assert_eq!(update.process_id, None);
        assert_eq!(update.exit_code, Some(3));
        assert!(sessions.processes.is_empty());
        let sent = server.await.unwrap();
        assert_eq!(
            String::from_utf8_lossy(&sent),
            "cd -- /workspace/repo && sh -lc 'echo hi'\n"
        );
    }

    #[tokio::test]
    async fn writes_stdin_to_a_live_terminal() {
        let (url, server) = mock_terminal(None).await;
        let sessions = CloudflarePtySessions::new();
        let update = sessions
            .start(
                &url,
                None,
                &PtyExecRequest::new("cat")
                    .with_tty(true)
                    .with_yield_time_ms(2000),
                None,
            )
            .await
            .unwrap();
        let session_id = update.process_id.expect("the terminal is still running");

        let update = sessions
            .write_stdin(&PtyWriteRequest::new(session_id, "typed\n").with_yield_time_ms(2000))
            .await
            .unwrap();
        assert_eq!(update.output_text(), "out: hi\r\n");
        assert_eq!(update.process_id, Some(session_id));

        sessions.terminate_all().await;
        assert!(sessions.processes.is_empty());
        let sent = server.await.unwrap();
        assert!(String::from_utf8_lossy(&sent).contains("typed"));
    }

    #[tokio::test]
    async fn refuses_stdin_without_a_tty_and_an_unknown_session() {
        let (url, _server) = mock_terminal(None).await;
        let sessions = CloudflarePtySessions::new();
        let update = sessions
            .start(
                &url,
                None,
                &PtyExecRequest::new("cat").with_yield_time_ms(2000),
                None,
            )
            .await
            .unwrap();
        let session_id = update.process_id.unwrap();
        assert_eq!(
            sessions
                .write_stdin(&PtyWriteRequest::new(session_id, "x"))
                .await
                .unwrap_err()
                .code,
            ErrorCode::PtyStdinUnavailable
        );
        assert_eq!(
            sessions
                .write_stdin(&PtyWriteRequest::new(1, "x"))
                .await
                .unwrap_err()
                .code,
            ErrorCode::PtySessionNotFound
        );
        sessions.terminate_all().await;
    }

    #[tokio::test]
    async fn reports_a_terminal_that_closes_before_it_is_ready() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
            ws.close(None).await.unwrap();
        });
        let sessions = CloudflarePtySessions::new();
        let error = sessions
            .start(
                &format!("ws://{address}/pty"),
                None,
                &PtyExecRequest::new("echo hi"),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::ExecTransportError);
        assert!(error.message.contains("before PTY ready"), "{error}");
    }
}
