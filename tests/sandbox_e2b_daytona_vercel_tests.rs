//! The E2B, Daytona and Vercel sandbox backends, driven against a local mock HTTP server.
//!
//! Every test points the provider's base-URL override at a `tokio::net::TcpListener` that records
//! each request and replies from a per-test handler, so the assertions cover the exact request
//! paths, authentication headers, bodies and response parsing without a network or a real key.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions, DaytonaSandboxSession,
};
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BCloudBucketMountStrategy, E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxType,
    WorkspacePersistence,
};
use openai_agents_sdk::extensions::sandbox::vercel::{
    NetworkPolicy, VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
    WorkspacePersistence as VercelWorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSessionState,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ---------------------------------------------------------------------------------------------
// A tiny recording HTTP server
// ---------------------------------------------------------------------------------------------

/// One request the mock server received.
#[derive(Debug, Clone)]
struct Recorded {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    fn query_param(&self, name: &str) -> Option<String> {
        self.query.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == name).then(|| percent_decode(value))
        })
    }

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Decode the small subset of percent-encoding the clients emit in query strings.
fn percent_decode(value: &str) -> String {
    let bytes = value.replace('+', " ").into_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The reply the handler produces for one request.
#[derive(Debug, Clone)]
struct MockResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    /// Extra body parts written after a delay, with no `Content-Length`, for streaming replies.
    stream: Vec<(Vec<u8>, u64)>,
}

impl MockResponse {
    fn json(value: serde_json::Value) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/json",
            body: serde_json::to_vec(&value).expect("serializable"),
            stream: Vec::new(),
        }
    }

    fn empty(status: u16) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            body: Vec::new(),
            stream: Vec::new(),
        }
    }

    fn bytes(body: impl Into<Vec<u8>>) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/octet-stream",
            body: body.into(),
            stream: Vec::new(),
        }
    }

    fn ndjson(lines: &[serde_json::Value]) -> Self {
        let mut body = Vec::new();
        for line in lines {
            body.extend_from_slice(
                serde_json::to_string(line)
                    .expect("serializable")
                    .as_bytes(),
            );
            body.push(b'\n');
        }
        MockResponse {
            status: 200,
            content_type: "application/x-ndjson",
            body,
            stream: Vec::new(),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(&serde_json::json!({ "code": status, "message": message }))
                .expect("serializable"),
            stream: Vec::new(),
        }
    }

    /// A reply whose body arrives in parts, each after the given delay in milliseconds.
    fn streamed(content_type: &'static str, parts: Vec<(Vec<u8>, u64)>) -> Self {
        MockResponse {
            status: 200,
            content_type,
            body: Vec::new(),
            stream: parts,
        }
    }
}

type Handler = Arc<dyn Fn(&Recorded) -> MockResponse + Send + Sync>;

/// A local HTTP server that records every request and answers from a closure.
struct MockServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    shutdown: tokio::sync::oneshot::Sender<()>,
}

impl MockServer {
    async fn start<F>(handler: F) -> Self
    where
        F: Fn(&Recorded) -> MockResponse + Send + Sync + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the mock server");
        let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
        let requests: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));
        let handler: Handler = Arc::new(handler);
        let (shutdown, mut stop) = tokio::sync::oneshot::channel();
        let recorder = requests.clone();
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted,
                    _ = &mut stop => break,
                };
                let Ok((stream, _)) = accepted else { break };
                let handler = handler.clone();
                let recorder = recorder.clone();
                tokio::spawn(async move {
                    let _ = serve_one(stream, handler, recorder).await;
                });
            }
        });
        MockServer {
            base_url,
            requests,
            shutdown,
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().expect("poisoned").clone()
    }

    /// The first request matching `method` and `path`.
    fn find(&self, method: &str, path: &str) -> Recorded {
        self.requests()
            .into_iter()
            .find(|request| request.method == method && request.path == path)
            .unwrap_or_else(|| {
                panic!(
                    "no `{method} {path}` request; saw {:?}",
                    self.requests()
                        .iter()
                        .map(|r| format!("{} {}", r.method, r.path))
                        .collect::<Vec<_>>()
                )
            })
    }

    fn stop(self) {
        let _ = self.shutdown.send(());
    }
}

/// Write a JSON response and close the connection.
async fn write_json(
    stream: &mut tokio::net::TcpStream,
    value: &serde_json::Value,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(value).expect("serializable");
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
         close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    stream.shutdown().await
}

/// Write a bodyless response and close the connection.
async fn write_status(stream: &mut tokio::net::TcpStream, status: u16) -> std::io::Result<()> {
    let head = format!("HTTP/1.1 {status} OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;
    stream.shutdown().await
}

/// Read one request from `stream`, stopping at the end of its body.
async fn read_request(stream: &mut tokio::net::TcpStream) -> std::io::Result<Recorded> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    // Read until the end of the headers; the clients never stream a request body.
    let header_end = loop {
        if let Some(index) = find_subslice(&buffer, b"\r\n\r\n") {
            break index;
        }
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            // The peer closed before sending a complete request.
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), query.to_string()),
        None => (target.clone(), String::new()),
    };
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let content_length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < content_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(content_length);

    Ok(Recorded {
        method,
        path,
        query,
        headers,
        body,
    })
}

/// Read one request, record it, and write the handler's reply.
/// The reply the in-sandbox path resolver gets, in the wire format of the provider `path` names.
fn resolver_reply_for(path: &str, body: &[u8]) -> Option<support::MockResponse> {
    if !String::from_utf8_lossy(body).contains("resolve-workspace-path") {
        return None;
    }
    let kind = if path.ends_with("/process.Process/Start") {
        support::Resolver::E2b
    } else if path.starts_with("/v1/sandbox/") && path.ends_with("/exec") {
        support::Resolver::Cloudflare
    } else if path.contains("/process/session/") && path.ends_with("/exec") {
        support::Resolver::Daytona
    } else {
        return None;
    };
    Some(support::resolver_reply(kind))
}

async fn serve_one(
    mut stream: tokio::net::TcpStream,
    handler: Handler,
    recorder: Arc<Mutex<Vec<Recorded>>>,
) -> std::io::Result<()> {
    let recorded = read_request(&mut stream).await?;
    let response = match resolver_reply_for(&recorded.path, &recorded.body) {
        Some(reply) => MockResponse {
            status: reply.status,
            content_type: reply.content_type,
            body: reply.body,
            stream: Vec::new(),
        },
        None => handler(&recorded),
    };
    recorder.lock().expect("poisoned").push(recorded);

    if response.stream.is_empty() {
        let head = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.status,
            response.content_type,
            response.body.len()
        );
        stream.write_all(head.as_bytes()).await?;
        stream.write_all(&response.body).await?;
    } else {
        // Without `Content-Length`, the body ends when the connection closes, which is how the
        // provider streaming endpoints behave.
        let head = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
            response.status, response.content_type
        );
        stream.write_all(head.as_bytes()).await?;
        for (part, delay_ms) in response.stream {
            if delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
            stream.write_all(&part).await?;
            stream.flush().await?;
        }
    }
    stream.flush().await?;
    stream.shutdown().await
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// ---------------------------------------------------------------------------------------------
// E2B
// ---------------------------------------------------------------------------------------------

/// Encode one Connect envelope: a flag byte, a big-endian length, then the JSON payload.
fn connect_frame(flags: u8, value: serde_json::Value) -> Vec<u8> {
    let payload = serde_json::to_vec(&value).expect("serializable");
    let mut frame = vec![flags];
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// A complete `process.Process/Start` response: stdout, then the end event, then the trailer.
fn e2b_process_stream(stdout: &str, exit_code: i64) -> MockResponse {
    use base64::Engine as _;
    let mut body = connect_frame(
        0,
        serde_json::json!({ "event": { "start": { "pid": 4242 } } }),
    );
    body.extend_from_slice(&connect_frame(
        0,
        serde_json::json!({
            "event": {
                "data": {
                    "stdout": base64::engine::general_purpose::STANDARD.encode(stdout),
                },
            },
        }),
    ));
    body.extend_from_slice(&connect_frame(
        0,
        serde_json::json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
    ));
    body.extend_from_slice(&connect_frame(2, serde_json::json!({})));
    MockResponse {
        status: 200,
        content_type: "application/connect+json",
        body,
        stream: Vec::new(),
    }
}

fn e2b_client(base_url: &str) -> E2BSandboxClient {
    E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(base_url)
            .with_sandbox_type(E2BSandboxType::CodeInterpreter)
            .with_timeout_secs(120)
            .with_metadata("owner", "tests"),
    )
}

#[tokio::test]
async fn e2b_runs_a_create_write_exec_read_delete_flow() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse {
                    status: 201,
                    content_type: "application/json",
                    body: serde_json::to_vec(&serde_json::json!({
                        "templateID": "code-interpreter-v1",
                        "sandboxID": "sbx-1",
                        "clientID": "cl-1",
                        "envdVersion": "0.2.0",
                        "envdAccessToken": "envd-token",
                    }))
                    .expect("serializable"),
                    stream: Vec::new(),
                },
                ("POST", "/filesystem.Filesystem/MakeDir") => {
                    MockResponse::json(serde_json::json!({}))
                }
                ("POST", "/filesystem.Filesystem/Remove") => {
                    MockResponse::json(serde_json::json!({}))
                }
                ("POST", "/process.Process/Start") => e2b_process_stream("hello from e2b\n", 0),
                ("POST", "/files") => MockResponse::json(serde_json::json!([
                    { "path": "/workspace/notes.txt", "name": "notes.txt", "type": "file" }
                ])),
                ("GET", "/files") => MockResponse::bytes("stored bytes"),
                ("DELETE", "/sandboxes/sbx-1") => MockResponse::empty(204),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = e2b_client(&server.base_url);
    let manifest = Manifest::new()
        .with_entry("task.md", Entry::text_file("read me"))
        .with_env("TASK", "demo");
    let session = client.create(Some(manifest)).await.expect("create");
    assert_eq!(session.backend_id(), "e2b");
    assert_eq!(session.workspace_root(), "/workspace");

    session.start().await.expect("start");
    session
        .write_file("notes.txt", b"stored bytes")
        .await
        .expect("write");
    let result = session
        .exec(ExecRequest::shell("echo hello").with_timeout(Some(Duration::from_secs(5))))
        .await
        .expect("exec");
    assert_eq!(result.stdout_text(), "hello from e2b\n");
    assert_eq!(result.exit_code, 0);
    assert_eq!(
        session.read_file("notes.txt").await.expect("read"),
        b"stored bytes"
    );
    session.rm("notes.txt", false).await.expect("rm");
    session.shutdown().await.expect("shutdown");

    // The platform call carries the API key and the create body built from the options.
    let create = server.find("POST", "/v2/sandboxes");
    assert_eq!(create.header("x-api-key"), Some("e2b_test"));
    let body = create.json();
    assert_eq!(body["templateID"], "code-interpreter-v1");
    assert_eq!(body["timeout"], 120);
    assert_eq!(body["metadata"]["owner"], "tests");
    // Manifest environment variables reach the sandbox at creation.
    assert_eq!(body["envVars"]["TASK"], "demo");
    assert_eq!(body["autoPause"], true);
    assert_eq!(body["autoResume"]["enabled"], true);

    // envd calls carry the routing headers and the per-sandbox access token, not the API key.
    let upload = server
        .requests()
        .into_iter()
        .find(|r| {
            r.method == "POST"
                && r.path == "/files"
                && r.query_param("path").as_deref() == Some("/workspace/notes.txt")
        })
        .expect("the notes.txt upload");
    assert_eq!(upload.header("e2b-sandbox-id"), Some("sbx-1"));
    assert_eq!(upload.header("e2b-sandbox-port"), Some("49983"));
    assert_eq!(upload.header("x-access-token"), Some("envd-token"));
    assert_eq!(
        upload.header("content-type"),
        Some("application/octet-stream")
    );
    assert_eq!(upload.body, b"stored bytes");

    let download = server.find("GET", "/files");
    assert_eq!(
        download.query_param("path").as_deref(),
        Some("/workspace/notes.txt")
    );

    // `Process/Start` is a Connect server-streaming call, so its body is enveloped. `start`
    // itself runs a command too, so select the one carrying the test's shell line.
    let start = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/process.Process/Start" && r.text().contains("echo hello"))
        .expect("the exec request");
    assert_eq!(
        start.header("content-type"),
        Some("application/connect+json")
    );
    assert_eq!(start.header("connect-protocol-version"), Some("1"));
    assert_eq!(start.body[0], 0, "the request frame must be uncompressed");
    let framed_length = u32::from_be_bytes(start.body[1..5].try_into().unwrap()) as usize;
    assert_eq!(framed_length, start.body.len() - 5);
    let payload: serde_json::Value = serde_json::from_slice(&start.body[5..]).expect("json");
    assert_eq!(payload["process"]["cmd"], "sh");
    assert_eq!(payload["process"]["args"][0], "-lc");
    assert_eq!(payload["process"]["cwd"], "/workspace");
    assert_eq!(payload["process"]["envs"]["TASK"], "demo");
    assert_eq!(payload["stdin"], false);

    // The manifest entry was materialized through the file API during `start`.
    assert!(
        server.requests().iter().any(|r| r.method == "POST"
            && r.path == "/files"
            && r.query_param("path").as_deref() == Some("/workspace/task.md")),
        "the manifest entry should have been uploaded"
    );
    server.stop();
}

#[tokio::test]
async fn e2b_reports_command_failures_and_missing_files() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-2",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/process.Process/Start") => e2b_process_stream("", 7),
                ("GET", "/files") => MockResponse::error(404, "path does not exist"),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url),
    );
    let session = client.create(None).await.expect("create");
    let result = session
        .exec(ExecRequest::shell("false"))
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 7);

    let error = session.read_file("missing.txt").await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

#[tokio::test]
async fn e2b_rejects_a_missing_api_key() {
    // No API key in the options: the backend must fail before any request is made.
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_base_url("http://127.0.0.1:1")
            .with_api_key(""),
    );
    let error = client.create(None).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}

#[tokio::test]
async fn e2b_streams_pty_output_and_writes_stdin() {
    use base64::Engine as _;
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-3",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/process.Process/Start") => {
                    let mut opening = connect_frame(
                        0,
                        serde_json::json!({ "event": { "start": { "pid": 1234 } } }),
                    );
                    opening.extend_from_slice(&connect_frame(
                        0,
                        serde_json::json!({
                            "event": {
                                "data": {
                                    "pty": base64::engine::general_purpose::STANDARD.encode("$ "),
                                },
                            },
                        }),
                    ));
                    // Hold the stream open past the first yield window, then finish it.
                    MockResponse::streamed(
                        "application/connect+json",
                        vec![
                            (opening, 0),
                            (
                                connect_frame(
                                    0,
                                    serde_json::json!({
                                        "event": {
                                            "data": {
                                                "pty": base64::engine::general_purpose::STANDARD
                                                    .encode("done\n"),
                                            },
                                        },
                                    }),
                                ),
                                400,
                            ),
                            (
                                connect_frame(
                                    0,
                                    serde_json::json!({
                                        "event": { "end": { "exited": true, "exitCode": 0 } },
                                    }),
                                ),
                                0,
                            ),
                            (connect_frame(2, serde_json::json!({})), 0),
                        ],
                    )
                }
                ("POST", "/process.Process/SendInput") => MockResponse::json(serde_json::json!({})),
                ("POST", "/process.Process/SendSignal") => {
                    MockResponse::json(serde_json::json!({}))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url),
    );
    let session = client.create(None).await.expect("create");
    assert!(session.supports_pty());

    let first = session
        .pty_exec_start(
            PtyExecRequest::new("bash")
                .with_tty(true)
                .with_yield_time_ms(250),
        )
        .await
        .expect("pty start");
    assert_eq!(first.output_text(), "$ ");
    let process_id = first.process_id.expect("the process is still running");

    let second = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "exit\n").with_yield_time_ms(1_000))
        .await
        .expect("write stdin");
    assert!(second.output_text().contains("done"), "{second:?}");
    assert_eq!(second.exit_code, Some(0));
    assert_eq!(second.process_id, None, "a finished process is forgotten");

    // A PTY start asks envd for a terminal and an attached stdin.
    let start = server.find("POST", "/process.Process/Start");
    let payload: serde_json::Value = serde_json::from_slice(&start.body[5..]).expect("json");
    assert_eq!(payload["stdin"], true);
    assert_eq!(payload["pty"]["size"]["rows"], 24);
    assert_eq!(payload["pty"]["size"]["cols"], 80);

    // Input is addressed by the pid envd reported and base64-encoded.
    let input = server.find("POST", "/process.Process/SendInput");
    let body = input.json();
    assert_eq!(body["process"]["pid"], 1234);
    assert_eq!(
        body["input"]["pty"],
        base64::engine::general_purpose::STANDARD.encode("exit\n")
    );
    server.stop();
}

#[tokio::test]
async fn e2b_resume_reconnects_to_the_recorded_sandbox() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes/sbx-saved/connect") => {
                    MockResponse::json(serde_json::json!({
                        "templateID": "base",
                        "sandboxID": "sbx-saved",
                        "clientID": "cl",
                        "envdVersion": "0.2.0",
                        "envdAccessToken": "fresh-token",
                    }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url),
    );
    let manifest = Manifest::new().with_entry("task.md", Entry::text_file("read me"));
    let state = SandboxSessionState::new("e2b", "session-1", "/workspace", &manifest)
        .with_provider_id(Some("sbx-saved".to_string()));
    let resumed = client
        .resume(&state, manifest.clone())
        .await
        .expect("resume");
    assert_eq!(resumed.id(), "session-1");
    assert_eq!(
        resumed.session_state().provider_id.as_deref(),
        Some("sbx-saved")
    );

    // The persisted state carries no token, so resume has to fetch a fresh one.
    let connect = server.find("POST", "/v2/sandboxes/sbx-saved/connect");
    assert_eq!(connect.header("x-api-key"), Some("e2b_test"));
    assert!(
        serde_json::to_string(&state.clone())
            .unwrap()
            .find("token")
            .is_none(),
        "the persisted state must stay credential-free"
    );

    // A manifest that does not match the recorded topology is rejected.
    let error = client.resume(&state, Manifest::new()).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    server.stop();
}

#[tokio::test]
async fn e2b_persists_a_native_snapshot_reference() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-4",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/sandboxes/sbx-4/snapshots") => MockResponse::json(serde_json::json!({
                    "snapshotID": "team/snap:default",
                    "names": ["team/snap:default"],
                })),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url)
            .with_workspace_persistence(WorkspacePersistence::Snapshot),
    );
    let session = client.create(None).await.expect("create");
    let archive = session.persist_workspace().await.expect("persist");
    let encoded = String::from_utf8_lossy(&archive);
    assert!(
        encoded.starts_with("E2B_SANDBOX_SNAPSHOT_V1\n"),
        "{encoded}"
    );
    assert!(encoded.contains("team/snap:default"));

    let snapshot = server.find("POST", "/sandboxes/sbx-4/snapshots");
    assert_eq!(snapshot.json()["memory"], false);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Daytona
// ---------------------------------------------------------------------------------------------

/// The platform response for a started sandbox whose toolbox points back at the mock server.
fn daytona_sandbox(base_url: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "sbx-daytona",
        "state": "started",
        "toolboxProxyUrl": format!("{base_url}/toolbox"),
    })
}

#[tokio::test]
async fn daytona_runs_a_create_write_exec_read_delete_flow() {
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            ("DELETE", "/sandbox/sbx-daytona") => MockResponse::json(serde_json::json!({})),
            ("POST", "/toolbox/sbx-daytona/process/session") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("POST", path) if path.ends_with("/exec") => MockResponse::json(serde_json::json!({
                "cmdId": "cmd-1",
                "exitCode": 0,
                "stdout": "hello from daytona\n",
                "stderr": "",
            })),
            ("DELETE", path) if path.starts_with("/toolbox/sbx-daytona/process/session/") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("POST", "/toolbox/sbx-daytona/files/upload-v2") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("GET", "/toolbox/sbx-daytona/files/download") => MockResponse::bytes("stored bytes"),
            ("POST", "/toolbox/sbx-daytona/files/folder") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("DELETE", "/toolbox/sbx-daytona/files") => MockResponse::json(serde_json::json!({})),
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_sandbox_snapshot_name("agents-base")
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");
    assert_eq!(session.backend_id(), "daytona");
    // A caller who supplies no manifest gets the Daytona image's workspace root.
    assert_eq!(session.workspace_root(), "/home/daytona/workspace");

    session.start().await.expect("start");
    session
        .write_file("notes.txt", b"stored bytes")
        .await
        .expect("write");
    let result = session
        .exec(ExecRequest::shell("echo hello").with_timeout(Some(Duration::from_secs(5))))
        .await
        .expect("exec");
    assert_eq!(result.stdout_text(), "hello from daytona\n");
    assert_eq!(
        session.read_file("notes.txt").await.expect("read"),
        b"stored bytes"
    );
    session.rm("notes.txt", true).await.expect("rm");
    session.shutdown().await.expect("shutdown");

    // The platform call authenticates with a bearer token and the organization scope.
    let create = server.find("POST", "/sandbox");
    assert_eq!(create.header("authorization"), Some("Bearer dtn_test"));
    assert_eq!(create.header("x-daytona-organization-id"), Some("org-1"));
    assert_eq!(create.json()["snapshot"], "agents-base");
    assert_eq!(create.json()["autoStopInterval"], 0);

    // Toolbox calls go to the proxy URL the platform returned, prefixed with the sandbox id, and
    // carry the same credentials.
    let session_create = server.find("POST", "/toolbox/sbx-daytona/process/session");
    assert_eq!(
        session_create.header("authorization"),
        Some("Bearer dtn_test")
    );
    assert!(
        session_create.json()["sessionId"]
            .as_str()
            .expect("session id")
            .starts_with("sandbox-")
    );

    // Each command is wrapped so the toolbox runs it in the workspace with the manifest env.
    let exec = server
        .requests()
        .into_iter()
        .find(|r| r.path.ends_with("/exec") && r.text().contains("echo hello"))
        .expect("the exec request");
    let command = exec.json()["command"]
        .as_str()
        .expect("command")
        .to_string();
    assert!(
        command.starts_with("cd /home/daytona/workspace && "),
        "{command}"
    );
    assert!(command.ends_with("sh -lc 'echo hello'"), "{command}");
    assert_eq!(exec.json()["runAsync"], false);

    // Uploads use a single-part multipart body addressed by the absolute path.
    let upload = server.find("POST", "/toolbox/sbx-daytona/files/upload-v2");
    assert_eq!(
        upload.query_param("path").as_deref(),
        Some("/home/daytona/workspace/notes.txt")
    );
    let content_type = upload.header("content-type").expect("content type");
    assert!(content_type.starts_with("multipart/form-data; boundary="));
    let text = upload.text();
    assert!(
        text.contains("name=\"file\"; filename=\"notes.txt\""),
        "{text}"
    );
    assert!(text.contains("stored bytes"), "{text}");

    let remove = server.find("DELETE", "/toolbox/sbx-daytona/files");
    assert_eq!(remove.query_param("recursive").as_deref(), Some("true"));
    server.stop();
}

#[tokio::test]
async fn daytona_streams_an_interactive_command() {
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    // The command's second line, and its exit, only appear after the session has polled the log
    // endpoint a few times, so the first yield window cannot drain the whole command.
    let polls = Arc::new(Mutex::new(0usize));
    let handler_polls = polls.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            ("POST", "/toolbox/sbx-daytona/process/session") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("POST", path) if path.ends_with("/exec") => {
                MockResponse::json(serde_json::json!({ "cmdId": "cmd-7" }))
            }
            ("GET", path) if path.ends_with("/logs") => {
                let seen = {
                    let mut polls = handler_polls.lock().expect("poisoned");
                    *polls += 1;
                    *polls
                };
                // The log endpoint always returns the whole log, so the session must only ever
                // hand the model the part it has not seen yet.
                MockResponse::bytes(if seen >= 6 {
                    "first\nsecond\n"
                } else {
                    "first\n"
                })
            }
            ("GET", path)
                if path.starts_with("/toolbox/sbx-daytona/process/session/")
                    && path.contains("/command/") =>
            {
                let finished = *handler_polls.lock().expect("poisoned") >= 6;
                MockResponse::json(if finished {
                    serde_json::json!({ "id": "cmd-7", "command": "bash", "exitCode": 0 })
                } else {
                    serde_json::json!({ "id": "cmd-7", "command": "bash" })
                })
            }
            ("DELETE", path) if path.starts_with("/toolbox/sbx-daytona/process/session/") => {
                MockResponse::json(serde_json::json!({}))
            }
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_state_poll_interval(Duration::from_millis(100)),
    );
    let session = client.create(None).await.expect("create");
    let first = session
        .pty_exec_start(PtyExecRequest::new("bash").with_yield_time_ms(250))
        .await
        .expect("start");
    assert_eq!(first.output_text(), "first\n");
    let process_id = first.process_id.expect("the command is still running");

    // A command started without a terminal has no stdin, exactly as in Python.
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "echo second\n"))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::PtyStdinUnavailable);

    // An empty write is a poll, which is how the model reads the rest of the output.
    let second = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "").with_yield_time_ms(5_000))
        .await
        .expect("poll");
    assert_eq!(second.output_text(), "second\n");
    assert_eq!(second.exit_code, Some(0));
    assert_eq!(second.process_id, None);

    let start = server
        .requests()
        .into_iter()
        .find(|r| r.path.ends_with("/exec"))
        .expect("the exec request");
    assert_eq!(start.json()["runAsync"], true);
    server.stop();
}

#[tokio::test]
async fn daytona_round_trips_a_workspace_archive() {
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            ("POST", "/toolbox/sbx-daytona/process/session") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("POST", path) if path.ends_with("/exec") => MockResponse::json(serde_json::json!({
                "cmdId": "cmd-1",
                "exitCode": 0,
                "stdout": "",
                "stderr": "",
            })),
            ("DELETE", path) if path.starts_with("/toolbox/sbx-daytona/process/session/") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("GET", "/toolbox/sbx-daytona/files/download") => MockResponse::bytes(vec![0x1f, 0x8b]),
            ("POST", "/toolbox/sbx-daytona/files/upload-v2") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("DELETE", "/toolbox/sbx-daytona/files") => MockResponse::json(serde_json::json!({})),
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");
    let archive = session.persist_workspace().await.expect("persist");
    assert_eq!(archive, vec![0x1f, 0x8b], "the gzip bytes must survive");

    // The archive is produced on disk and downloaded, because command output is text.
    let tar = server
        .requests()
        .into_iter()
        .find(|r| r.path.ends_with("/exec") && r.text().contains("tar -czf"))
        .expect("the tar command");
    let command = tar.json()["command"].as_str().expect("command").to_string();
    assert!(command.contains("-C /home/daytona/workspace"), "{command}");
    assert!(command.contains("/tmp/openai-agents-"), "{command}");

    // The SDK validates archive bytes before the provider unpacks them, so the probe archive is
    // an empty but well-formed tar.
    session
        .hydrate_workspace(&[0u8; 1024])
        .await
        .expect("hydrate");
    let upload = server.find("POST", "/toolbox/sbx-daytona/files/upload-v2");
    assert!(
        upload
            .query_param("path")
            .expect("path")
            .starts_with("/tmp/openai-agents-")
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Vercel
// ---------------------------------------------------------------------------------------------

fn vercel_sandbox_and_session(status: &str) -> serde_json::Value {
    serde_json::json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": "sess-1",
        },
        "session": {
            "id": "sess-1",
            "memory": 2048,
            "vcpus": 2,
            "region": "iad1",
            "timeout": 270000,
            "status": status,
            "requestedAt": 1,
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": "/vercel/sandbox",
        },
        "routes": [],
    })
}

#[tokio::test]
async fn vercel_runs_a_create_write_exec_read_stop_flow() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("running"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let mut body = vercel_sandbox_and_session("running");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/cmd") => MockResponse::ndjson(&[
                    serde_json::json!({
                        "command": {
                            "id": "cmd-1",
                            "name": "sh",
                            "args": ["-lc", "echo hello"],
                            "cwd": "/vercel/sandbox",
                            "sessionId": "sess-1",
                            "exitCode": serde_json::Value::Null,
                            "startedAt": 1,
                        },
                    }),
                    serde_json::json!({ "stream": "stdout", "data": "hello from vercel\n" }),
                    serde_json::json!({
                        "command": {
                            "id": "cmd-1",
                            "name": "sh",
                            "args": [],
                            "cwd": "/vercel/sandbox",
                            "sessionId": "sess-1",
                            "exitCode": 0,
                            "startedAt": 1,
                        },
                    }),
                ]),
                ("POST", "/v2/sandboxes/sessions/sess-1/fs/write") => {
                    MockResponse::json(serde_json::json!({}))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/fs/mkdir") => {
                    MockResponse::json(serde_json::json!({}))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/fs/read") => {
                    MockResponse::bytes("stored bytes")
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/stop") => {
                    let mut body = vercel_sandbox_and_session("stopped");
                    MockResponse::json(serde_json::json!({ "session": body["session"].take() }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_project_id("prj_1")
            .with_team_id("team_1")
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");
    assert_eq!(session.backend_id(), "vercel");
    // The Vercel runtime image rewrites the SDK-wide default workspace root.
    assert_eq!(session.workspace_root(), "/vercel/sandbox");
    assert!(!session.supports_pty(), "Vercel sandboxes have no PTY");

    session.start().await.expect("start");
    session
        .write_file("notes.txt", b"stored bytes")
        .await
        .expect("write");
    let result = session
        .exec(ExecRequest::shell("echo hello"))
        .await
        .expect("exec");
    assert_eq!(result.stdout_text(), "hello from vercel\n");
    assert_eq!(result.exit_code, 0);
    assert_eq!(
        session.read_file("notes.txt").await.expect("read"),
        b"stored bytes"
    );
    assert!(session.running().await.expect("running"));
    session.shutdown().await.expect("shutdown");

    // Every call is a bearer-authenticated request scoped to the team.
    let create = server.find("POST", "/v3/sandboxes");
    assert_eq!(create.header("authorization"), Some("Bearer vercel_test"));
    assert_eq!(create.query_param("teamId").as_deref(), Some("team_1"));
    let body = create.json();
    assert_eq!(body["projectId"], "prj_1");
    assert_eq!(body["persistent"], false);
    assert_eq!(body["timeout"], 270_000);
    // The runtime alias is resolved to a concrete Amazon Linux image, which is what ships the
    // S3 mount tools; `/v3/sandboxes` has no `runtime` field to fall back on.
    assert_eq!(body["image"], "vercel/sandbox/node:al-24");
    assert!(body["runtime"].is_null(), "{body}");
    assert!(
        body["name"].as_str().expect("name").starts_with("agents-"),
        "{body}"
    );

    let cmd = server
        .requests()
        .into_iter()
        .find(|r| r.path.ends_with("/cmd") && r.text().contains("echo hello"))
        .expect("the command request");
    let cmd_body = cmd.json();
    assert_eq!(cmd_body["command"], "sh");
    assert_eq!(cmd_body["args"][0], "-lc");
    assert_eq!(cmd_body["cwd"], "/vercel/sandbox");
    assert_eq!(cmd_body["wait"], true);
    assert_eq!(cmd_body["logs"], true);

    // `fs/write` only accepts a gzipped tar extracted relative to `x-cwd`.
    let write = server.find("POST", "/v2/sandboxes/sessions/sess-1/fs/write");
    assert_eq!(write.header("content-type"), Some("application/gzip"));
    assert_eq!(write.header("x-cwd"), Some("/"));
    assert_eq!(&write.body[..3], &[0x1f, 0x8b, 0x08]);

    let read = server.find("POST", "/v2/sandboxes/sessions/sess-1/fs/read");
    assert_eq!(read.json()["path"], "/vercel/sandbox/notes.txt");
    server.stop();
}

#[tokio::test]
async fn vercel_waits_for_a_pending_session_and_fails_on_terminal_states() {
    let polls = Arc::new(Mutex::new(0usize));
    let handler_polls = polls.clone();
    let server =
        MockServer::start(
            move |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("pending"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let seen = {
                        let mut polls = handler_polls.lock().expect("poisoned");
                        *polls += 1;
                        *polls
                    };
                    let status = if seen >= 2 { "running" } else { "pending" };
                    let mut body = vercel_sandbox_and_session(status);
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    client.create(None).await.expect("create");
    assert!(
        *polls.lock().expect("poisoned") >= 2,
        "the client should poll until the session is running"
    );
    server.stop();

    let failing =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("pending"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let mut body = vercel_sandbox_and_session("failed");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;
    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&failing.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let error = client.create(None).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(error.message.contains("failed"), "{error}");
    failing.stop();
}

#[tokio::test]
async fn vercel_persists_and_restores_a_native_snapshot() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    let created = if request.json()["source"].is_null() {
                        vercel_sandbox_and_session("running")
                    } else {
                        // A snapshot restore replaces the sandbox with a new session.
                        serde_json::json!({
                            "sandbox": {
                                "name": "agents-restored",
                                "persistent": false,
                                "createdAt": 2,
                                "updatedAt": 2,
                                "currentSessionId": "sess-2",
                            },
                            "session": {
                                "id": "sess-2",
                                "memory": 2048,
                                "vcpus": 2,
                                "region": "iad1",
                                "timeout": 270000,
                                "status": "running",
                                "requestedAt": 2,
                                "createdAt": 2,
                                "updatedAt": 2,
                                "cwd": "/vercel/sandbox",
                            },
                            "routes": [],
                        })
                    };
                    MockResponse::json(created)
                }
                ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
                    let mut body = vercel_sandbox_and_session("running");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/snapshot") => {
                    MockResponse::json(serde_json::json!({
                        "snapshot": {
                            "id": "snap-1",
                            "sourceSessionId": "sess-1",
                            "region": "iad1",
                            "status": "created",
                            "sizeBytes": 10,
                            "createdAt": 1,
                            "updatedAt": 1,
                        },
                        "session": vercel_sandbox_and_session("running")["session"],
                    }))
                }
                ("POST", path) if path.ends_with("/stop") => {
                    MockResponse::json(serde_json::json!({
                        "session": vercel_sandbox_and_session("stopped")["session"],
                    }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_workspace_persistence(VercelWorkspacePersistence::Snapshot)
            .with_snapshot_expiration_ms(60_000)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");
    let archive = session.persist_workspace().await.expect("persist");
    assert!(
        String::from_utf8_lossy(&archive).contains("snap-1"),
        "{:?}",
        String::from_utf8_lossy(&archive)
    );
    assert_eq!(
        server
            .find("POST", "/v2/sandboxes/sessions/sess-1/snapshot")
            .json()["expiration"],
        60_000
    );

    session.hydrate_workspace(&archive).await.expect("hydrate");
    // The session now points at the replacement sandbox and its new runtime session.
    assert_eq!(
        session.session_state().provider_id.as_deref(),
        Some("agents-restored:sess-2")
    );
    let restore = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/v3/sandboxes" && !r.json()["source"].is_null())
        .expect("the restore request");
    assert_eq!(restore.json()["source"]["type"], "snapshot");
    assert_eq!(restore.json()["source"]["snapshotId"], "snap-1");
    // The session the snapshot replaced is stopped so the project does not leak sandboxes.
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.path == "/v2/sandboxes/sessions/sess-1/stop"),
        "the replaced session should be stopped"
    );
    server.stop();
}

#[tokio::test]
async fn vercel_rejects_an_unsupported_exec_user() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("running"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let mut body = vercel_sandbox_and_session("running");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/cmd") => MockResponse::ndjson(&[
                    serde_json::json!({ "stream": "stdout", "data": "root\n" }),
                    serde_json::json!({
                        "command": {
                            "id": "cmd-1",
                            "name": "id",
                            "args": [],
                            "cwd": "/vercel/sandbox",
                            "sessionId": "sess-1",
                            "exitCode": 0,
                            "startedAt": 1,
                        },
                    }),
                ]),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    // `root` becomes the API's `sudo` flag instead of a `sudo` command wrapper.
    let result = session
        .exec(ExecRequest::shell("id").with_user(Some("root".into())))
        .await
        .expect("exec as root");
    assert_eq!(result.stdout_text(), "root\n");
    let cmd = server.find("POST", "/v2/sandboxes/sessions/sess-1/cmd");
    assert_eq!(cmd.json()["sudo"], true);
    assert_eq!(cmd.json()["command"], "sh");

    let error = session
        .exec(ExecRequest::shell("id").with_user(Some("agent".into())))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Regressions
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn e2b_start_tolerates_an_existing_workspace_root() {
    // The first `mkdir` of the root succeeds; envd answers the second with Connect's
    // `already_exists`, which must not fail the session. (`start` itself only creates a root that
    // is missing, and the manifest applier's root `mkdir` after it makes no call.)
    let seen = Arc::new(Mutex::new(0usize));
    let handler_seen = seen.clone();
    let server =
        MockServer::start(
            move |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-dup",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/filesystem.Filesystem/MakeDir") => {
                    let count = {
                        let mut seen = handler_seen.lock().expect("poisoned");
                        *seen += 1;
                        *seen
                    };
                    if count == 1 {
                        MockResponse::json(serde_json::json!({}))
                    } else {
                        MockResponse {
                            status: 409,
                            content_type: "application/json",
                            body: serde_json::to_vec(&serde_json::json!({
                                "code": "already_exists",
                                "message": "path /workspace already exists",
                            }))
                            .expect("serializable"),
                            stream: Vec::new(),
                        }
                    }
                }
                ("POST", "/process.Process/Start") => e2b_process_stream("", 0),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url),
    );
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");
    session
        .mkdir("/workspace", true)
        .await
        .expect("first mkdir");
    session
        .mkdir("/workspace", true)
        .await
        .expect("an existing root is tolerated");
    assert_eq!(*seen.lock().expect("poisoned"), 2);
    server.stop();
}

#[tokio::test]
async fn e2b_activates_an_rclone_cloud_bucket_mount() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-mount",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/filesystem.Filesystem/MakeDir") => {
                    MockResponse::json(serde_json::json!({}))
                }
                // Every preflight probe and the mount itself succeed. The sandbox reports the
                // user it runs commands as, which the mount has to hand to `rclone`.
                ("POST", "/process.Process/Start") => {
                    if request.text().contains("id -u; id -g") {
                        e2b_process_stream("1000\n1001\n", 0)
                    } else {
                        e2b_process_stream("", 0)
                    }
                }
                ("POST", "/files") => MockResponse::json(serde_json::json!([])),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url),
    );
    let mount = E2BCloudBucketMountStrategy::new("bucket", "my-bucket", "[bucket]\ntype = s3\n")
        .mount()
        .with_read_only(true);
    let manifest = Manifest::new()
        .with_entry("data", Entry::mount(mount))
        // The mount writes a credential into the sandbox, so the manifest has to acknowledge its
        // exact path before it can activate.
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");

    // The rclone configuration is written inside the workspace, where the session's own file API
    // can reach it, and never passed on a command line.
    let config = server
        .requests()
        .into_iter()
        .find(|r| {
            // The file name is derived from the mount path, so two mounts cannot share one.
            r.method == "POST"
                && r.path == "/files"
                && r.query_param("path").is_some_and(|path| {
                    path.starts_with("/workspace/.sandbox-rclone-config/")
                        && path.ends_with(".conf")
                })
        })
        .expect("the rclone configuration upload");
    assert_eq!(config.text(), "[bucket]\ntype = s3\n");

    let mount_command = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/process.Process/Start" && r.text().contains("bucket:my-bucket"))
        .expect("the rclone mount command");
    let payload: serde_json::Value =
        serde_json::from_slice(&mount_command.body[5..]).expect("json");
    // The mount runs as one shell command, so the flags are read out of the command line.
    let command = payload["process"]["args"]
        .as_array()
        .and_then(|args| args.last())
        .and_then(serde_json::Value::as_str)
        .expect("the mount command")
        .to_string();
    assert!(command.contains("bucket:my-bucket"), "{command}");
    assert!(command.contains("/workspace/data"), "{command}");
    assert!(command.contains("--read-only"), "{command}");
    assert!(command.contains("--daemon"), "{command}");
    // The provider adds the flags its image needs; the core pattern does not. `rclone mount`
    // runs as root, so without the command user's ids the mounted tree is not usable by the
    // user the sandbox runs commands as.
    assert!(command.contains("--allow-other"), "{command}");
    assert!(command.contains("--uid 1000"), "{command}");
    assert!(command.contains("--gid 1001"), "{command}");
    assert!(
        command.contains("/workspace/.sandbox-rclone-config/"),
        "{command}"
    );
    assert!(
        !mount_command.text().contains("type = s3"),
        "the configuration must not reach the command line"
    );
    server.stop();
}

#[tokio::test]
async fn vercel_resumes_from_its_own_persisted_state() {
    // `create` rewrites the default workspace root, so a state built from a live session has to
    // round trip through JSON and back into `resume` without tripping the manifest digest check.
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("running"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let mut body = vercel_sandbox_and_session("running");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;

    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let manifest = Manifest::new().with_entry("task.md", Entry::text_file("read me"));
    let session = client.create(Some(manifest.clone())).await.expect("create");
    let state = session.session_state();
    assert_eq!(state.workspace_root, "/vercel/sandbox");

    let payload = state.to_json_string().expect("serialize");
    assert!(
        !payload.contains("vercel_test"),
        "state must stay credential-free"
    );
    let restored = SandboxSessionState::from_json_str(&payload).expect("deserialize");

    let resumed = client.resume(&restored, manifest).await.expect("resume");
    assert_eq!(resumed.workspace_root(), "/vercel/sandbox");
    assert_eq!(
        resumed.session_state().provider_id.as_deref(),
        Some("agents-test:sess-1")
    );
    server.stop();
}

#[tokio::test]
async fn daytona_resumes_from_its_own_persisted_state_and_restarts_the_sandbox() {
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let started = Arc::new(Mutex::new(false));
    let handler_started = started.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") => MockResponse::json(daytona_sandbox(&base_url)),
            ("GET", "/sandbox/sbx-daytona") => {
                // The previous run stopped the sandbox; it only reports `started` again after an
                // explicit start call.
                let state = if *handler_started.lock().expect("poisoned") {
                    "started"
                } else {
                    "stopped"
                };
                let mut body = daytona_sandbox(&base_url);
                body["state"] = serde_json::json!(state);
                MockResponse::json(body)
            }
            ("POST", "/sandbox/sbx-daytona/start") => {
                *handler_started.lock().expect("poisoned") = true;
                MockResponse::json(serde_json::json!({}))
            }
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let manifest = Manifest::new()
        .with_root("/home/daytona/workspace")
        .with_entry("task.md", Entry::text_file("read me"));
    let state =
        SandboxSessionState::new("daytona", "session-1", "/home/daytona/workspace", &manifest)
            .with_provider_id(Some("sbx-daytona".to_string()));
    let payload = state.to_json_string().expect("serialize");
    let restored = SandboxSessionState::from_json_str(&payload).expect("deserialize");

    let resumed = client.resume(&restored, manifest).await.expect("resume");
    assert_eq!(resumed.id(), "session-1");
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.path == "/sandbox/sbx-daytona/start"),
        "a stopped sandbox must be started before the toolbox is used"
    );
    server.stop();
}

#[tokio::test]
async fn daytona_resumes_a_session_created_without_a_manifest() {
    // `create(None)` picks the Daytona workspace root, so the recorded digest covers that root.
    // Resuming with a freshly built manifest must still verify.
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");
    let state = session.session_state();
    assert_eq!(state.workspace_root, "/home/daytona/workspace");

    let payload = state.to_json_string().expect("serialize");
    let restored = SandboxSessionState::from_json_str(&payload).expect("deserialize");
    let resumed = client
        .resume(&restored, Manifest::new())
        .await
        .expect("resume");
    assert_eq!(resumed.workspace_root(), "/home/daytona/workspace");
    server.stop();
}

#[tokio::test]
async fn daytona_drives_a_terminal_over_the_pty_websocket() {
    use futures::{SinkExt as _, StreamExt as _};
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
    use tokio_tungstenite::tungstenite::protocol::{CloseFrame, Role};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the mock server");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let handshake: Arc<Mutex<Option<Recorded>>> = Arc::new(Mutex::new(None));
    let typed: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let resized = Arc::new(Mutex::new(None::<serde_json::Value>));

    let server_base = base_url.clone();
    let server_handshake = handshake.clone();
    let server_typed = typed.clone();
    let server_resized = resized.clone();
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let server_base = server_base.clone();
            let server_handshake = server_handshake.clone();
            let server_typed = server_typed.clone();
            let server_resized = server_resized.clone();
            // Each connection is served on its own task, so a live websocket never blocks the
            // next upgrade.
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                match (request.method.as_str(), request.path.as_str()) {
                    ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                        let _ = write_json(&mut stream, &daytona_sandbox(&server_base)).await;
                    }
                    ("POST", "/toolbox/sbx-daytona/process/pty/pty-session/resize") => {
                        *server_resized.lock().expect("poisoned") = Some(request.json());
                        let _ = write_json(&mut stream, &serde_json::json!({})).await;
                    }
                    ("DELETE", path) if path.contains("/process/pty/") => {
                        let _ = write_json(&mut stream, &serde_json::json!({})).await;
                    }
                    ("GET", "/toolbox/sbx-daytona/process/pty/create-connect") => {
                        *server_handshake.lock().expect("poisoned") = Some(request.clone());
                        // Complete the upgrade by hand, then speak frames over the raw socket.
                        let key = request
                            .header("sec-websocket-key")
                            .expect("the handshake key")
                            .to_string();
                        let response = format!(
                            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: \
                         Upgrade\r\nSec-WebSocket-Accept: {}\r\nSec-WebSocket-Protocol: \
                         X-Daytona-Pty-Exit-Control\r\n\r\n",
                            derive_accept_key(key.as_bytes())
                        );
                        if stream.write_all(response.as_bytes()).await.is_err() {
                            return;
                        }
                        let mut ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
                            stream,
                            Role::Server,
                            None,
                        )
                        .await;
                        let _ = ws
                            .send(Message::text(
                                r#"{"type":"control","status":"connected"}"#.to_string(),
                            ))
                            .await;
                        let _ = ws.send(Message::text("$ ".to_string())).await;
                        // Wait for the command line the session types into the shell.
                        while let Some(Ok(message)) = ws.next().await {
                            let data = match message {
                                Message::Binary(data) => data.to_vec(),
                                Message::Text(text) => text.as_bytes().to_vec(),
                                Message::Ping(_) => continue,
                                Message::Close(_) => break,
                                _ => continue,
                            };
                            let done = {
                                let mut typed = server_typed.lock().expect("poisoned");
                                typed.extend_from_slice(&data);
                                typed.windows(5).any(|w| w == b"exit\n")
                            };
                            let _ = ws.send(Message::binary(b"hello terminal\n".to_vec())).await;
                            if done {
                                let _ = ws
                                    .send(Message::Close(Some(CloseFrame {
                                        code: 1000u16.into(),
                                        // The daemon encodes the exit code in the close reason.
                                        reason: r#"{"exitCode":7}"#.into(),
                                    })))
                                    .await;
                                break;
                            }
                        }
                    }
                    _ => {
                        let _ = write_status(&mut stream, 404).await;
                    }
                }
            });
        }
    });

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&base_url)
            .with_organization_id("org-1")
            .with_env("TASK", "demo")
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    let first = session
        .pty_exec_start(
            PtyExecRequest::new("bash")
                .with_tty(true)
                .with_yield_time_ms(250),
        )
        .await
        .expect("pty start");
    assert!(first.output_text().contains("$ "), "{first:?}");
    let process_id = first.process_id.expect("the terminal is still open");

    let second = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "exit\n").with_yield_time_ms(1_000))
        .await
        .expect("write stdin");
    assert!(
        second.output_text().contains("hello terminal"),
        "{second:?}"
    );
    // The exit code comes from the websocket close frame, not a REST round trip.
    assert_eq!(second.exit_code, Some(7));
    assert_eq!(second.process_id, None);

    // The upgrade carries the REST credentials, the terminal size and the environment, the last
    // as a subprotocol token rather than a query parameter.
    let recorded = handshake
        .lock()
        .expect("poisoned")
        .clone()
        .expect("handshake");
    assert_eq!(recorded.header("authorization"), Some("Bearer dtn_test"));
    assert_eq!(recorded.header("x-daytona-organization-id"), Some("org-1"));
    assert_eq!(recorded.query_param("cols").as_deref(), Some("80"));
    assert_eq!(recorded.query_param("rows").as_deref(), Some("24"));
    assert_eq!(
        recorded.query_param("cwd").as_deref(),
        Some("/home/daytona/workspace")
    );
    assert!(
        recorded
            .query_param("id")
            .is_some_and(|id| id.starts_with("pty-")),
        "{:?}",
        recorded.query_param("id")
    );
    let protocols = recorded
        .header("sec-websocket-protocol")
        .expect("subprotocols")
        .to_string();
    assert!(
        protocols.contains("X-Daytona-Pty-Exit-Control"),
        "{protocols}"
    );
    let token = protocols
        .split(',')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("X-Daytona-Pty-Envs~"))
        .expect("the environment subprotocol");
    let decoded = {
        use base64::Engine as _;
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(token)
            .expect("base64url")
    };
    let envs: serde_json::Value = serde_json::from_slice(&decoded).expect("json");
    assert_eq!(envs["TASK"], "demo");

    // The command reaches the terminal as a typed line, shell-wrapped exactly as `exec` wraps
    // it, and the model's later input is forwarded verbatim.
    assert_eq!(
        String::from_utf8_lossy(&typed.lock().expect("poisoned")),
        "sh -lc bash\nexit\n"
    );
    server.abort();
}

#[tokio::test]
async fn daytona_resizes_a_terminal_and_survives_a_handshake_without_a_subprotocol() {
    use futures::{SinkExt as _, StreamExt as _};
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
    use tokio_tungstenite::tungstenite::protocol::{CloseFrame, Role};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the mock server");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let typed: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let resized: Arc<Mutex<Vec<(String, serde_json::Value)>>> = Arc::new(Mutex::new(Vec::new()));
    let ids: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let server_base = base_url.clone();
    let server_typed = typed.clone();
    let server_resized = resized.clone();
    let server_ids = ids.clone();
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let server_base = server_base.clone();
            let server_typed = server_typed.clone();
            let server_resized = server_resized.clone();
            let server_ids = server_ids.clone();
            // Each connection is served on its own task, so the first websocket never blocks the
            // retry the client opens when the handshake selected no subprotocol.
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                match (request.method.as_str(), request.path.as_str()) {
                    ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                        let _ = write_json(&mut stream, &daytona_sandbox(&server_base)).await;
                    }
                    ("POST", path)
                        if path.starts_with("/toolbox/sbx-daytona/process/pty/")
                            && path.ends_with("/resize") =>
                    {
                        server_resized
                            .lock()
                            .expect("poisoned")
                            .push((path.to_string(), request.json()));
                        let _ = write_json(&mut stream, &serde_json::json!({})).await;
                    }
                    ("DELETE", path) if path.contains("/process/pty/") => {
                        let _ = write_json(&mut stream, &serde_json::json!({})).await;
                    }
                    ("GET", "/toolbox/sbx-daytona/process/pty/create-connect") => {
                        let id = request.query_param("id").expect("the pty session id");
                        // The daemon rejects a session id that is already in use, so a retry
                        // that reused the first id would fail here.
                        let fresh = {
                            let mut seen = server_ids.lock().expect("poisoned");
                            let fresh = !seen.contains(&id);
                            seen.push(id.clone());
                            fresh
                        };
                        if !fresh {
                            let _ = write_status(&mut stream, 409).await;
                            return;
                        }
                        let key = request
                            .header("sec-websocket-key")
                            .expect("the handshake key")
                            .to_string();
                        // RFC 6455 lets a server accept without selecting a subprotocol; this one
                        // does, which the client must survive.
                        let response = format!(
                            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: \
                         Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
                            derive_accept_key(key.as_bytes())
                        );
                        if stream.write_all(response.as_bytes()).await.is_err() {
                            return;
                        }
                        let mut ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
                            stream,
                            Role::Server,
                            None,
                        )
                        .await;
                        let _ = ws
                            .send(Message::text(
                                r#"{"type":"control","status":"connected"}"#.to_string(),
                            ))
                            .await;
                        while let Some(Ok(message)) = ws.next().await {
                            let data = match message {
                                Message::Binary(data) => data.to_vec(),
                                Message::Text(text) => text.as_bytes().to_vec(),
                                Message::Ping(_) => continue,
                                Message::Close(_) => break,
                                _ => continue,
                            };
                            let done = {
                                let mut typed = server_typed.lock().expect("poisoned");
                                typed.extend_from_slice(&data);
                                typed.windows(5).any(|w| w == b"exit\n")
                            };
                            if done {
                                let _ = ws
                                    .send(Message::Close(Some(CloseFrame {
                                        code: 1000u16.into(),
                                        reason: String::new().into(),
                                    })))
                                    .await;
                                break;
                            }
                        }
                    }
                    _ => {
                        let _ = write_status(&mut stream, 404).await;
                    }
                }
            });
        }
    });

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&base_url)
            .with_env("TASK", "demo")
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    let first = session
        .pty_exec_start(
            PtyExecRequest::new("bash")
                .with_tty(true)
                .with_yield_time_ms(250),
        )
        .await
        .expect("pty start");
    let process_id = first.process_id.expect("the terminal is still open");

    // The concrete session exposes resize, which the trait object reaches through `as_any`.
    let daytona = session
        .as_any()
        .and_then(|any| any.downcast_ref::<DaytonaSandboxSession>())
        .expect("a Daytona session");
    daytona
        .resize_terminal(process_id, 120, 30)
        .await
        .expect("resize");
    let recorded = resized.lock().expect("poisoned").clone();
    let (path, body) = recorded.first().expect("a resize request");
    assert!(
        path.starts_with("/toolbox/sbx-daytona/process/pty/pty-"),
        "{path}"
    );
    assert_eq!(body["cols"], 120);
    assert_eq!(body["rows"], 30);

    let second = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "exit\n").with_yield_time_ms(1_000))
        .await
        .expect("write stdin");
    // A clean close with no structured reason means the shell exited successfully.
    assert_eq!(second.exit_code, Some(0));

    // The handshake could not carry the environment, so it was exported on the typed line.
    let typed = String::from_utf8_lossy(&typed.lock().expect("poisoned")).into_owned();
    assert!(
        typed.starts_with("env -- TASK=demo sh -lc bash\n"),
        "{typed}"
    );
    server.abort();
}

#[tokio::test]
async fn providers_honor_per_command_environment_variables() {
    // `ExecRequest::with_env` is part of the core contract; each backend expresses it in its own
    // wire format, and a per-command variable always wins over the manifest value.
    let e2b_server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "templateID": "base",
                    "sandboxID": "sbx-env",
                    "clientID": "cl",
                    "envdVersion": "0.2.0",
                    "envdAccessToken": "envd-token",
                })),
                ("POST", "/process.Process/Start") => e2b_process_stream("", 0),
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;
    let e2b = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&e2b_server.base_url),
    );
    let manifest = Manifest::new()
        .with_env("TASK", "manifest")
        .with_env("KEEP", "yes");
    let session = e2b.create(Some(manifest.clone())).await.expect("create");
    session
        .exec(
            ExecRequest::shell("env")
                .with_env("TASK", "per-command")
                .with_env("EXTRA", "1"),
        )
        .await
        .expect("exec");
    let start = e2b_server.find("POST", "/process.Process/Start");
    let payload: serde_json::Value = serde_json::from_slice(&start.body[5..]).expect("json");
    assert_eq!(payload["process"]["envs"]["TASK"], "per-command");
    assert_eq!(payload["process"]["envs"]["EXTRA"], "1");
    assert_eq!(payload["process"]["envs"]["KEEP"], "yes");
    e2b_server.stop();

    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let daytona_server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            ("POST", "/toolbox/sbx-daytona/process/session") => {
                MockResponse::json(serde_json::json!({}))
            }
            ("POST", path) if path.ends_with("/exec") => MockResponse::json(serde_json::json!({
                "cmdId": "cmd-1", "exitCode": 0, "stdout": "", "stderr": "",
            })),
            ("DELETE", _) => MockResponse::json(serde_json::json!({})),
            _ => MockResponse::error(404, "unexpected request"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = daytona_server.base_url.clone();
    let daytona = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&daytona_server.base_url)
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = daytona
        .create(Some(manifest.clone().with_root("/home/daytona/workspace")))
        .await
        .expect("create");
    session
        .exec(ExecRequest::shell("env").with_env("TASK", "per-command"))
        .await
        .expect("exec");
    let exec = daytona_server
        .requests()
        .into_iter()
        .find(|r| r.path.ends_with("/exec"))
        .expect("the exec request");
    let command = exec.json()["command"]
        .as_str()
        .expect("command")
        .to_string();
    // Daytona has no environment parameter, so it lands in the command string.
    assert!(command.contains("TASK=per-command"), "{command}");
    assert!(!command.contains("TASK=manifest"), "{command}");
    assert!(command.contains("KEEP=yes"), "{command}");
    daytona_server.stop();

    let vercel_server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => {
                    MockResponse::json(vercel_sandbox_and_session("running"))
                }
                ("GET", "/v2/sandboxes/sessions/sess-1") => {
                    let mut body = vercel_sandbox_and_session("running");
                    MockResponse::json(serde_json::json!({
                        "session": body["session"].take(),
                        "routes": [],
                    }))
                }
                ("POST", "/v2/sandboxes/sessions/sess-1/cmd") => {
                    MockResponse::ndjson(&[serde_json::json!({
                        "command": {
                            "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                            "sessionId": "sess-1", "exitCode": 0, "startedAt": 1,
                        },
                    })])
                }
                _ => MockResponse::error(404, "unexpected request"),
            },
        )
        .await;
    let vercel = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&vercel_server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let session = vercel.create(Some(manifest)).await.expect("create");
    session
        .exec(ExecRequest::shell("env").with_env("TASK", "per-command"))
        .await
        .expect("exec");
    let cmd = vercel_server.find("POST", "/v2/sandboxes/sessions/sess-1/cmd");
    assert_eq!(cmd.json()["env"]["TASK"], "per-command");
    assert_eq!(cmd.json()["env"]["KEEP"], "yes");
    vercel_server.stop();
}

// ---------------------------------------------------------------------------------------------
// Vercel cloud bucket mounts
// ---------------------------------------------------------------------------------------------

/// One finished command, with the stdout it produced.
fn vercel_command_result(stdout: &str, exit_code: i64) -> MockResponse {
    MockResponse::ndjson(&[
        serde_json::json!({ "stream": "stdout", "data": stdout }),
        serde_json::json!({
            "command": {
                "id": "cmd-1",
                "name": "sh",
                "args": [],
                "cwd": "/vercel/sandbox",
                "sessionId": "sess-1",
                "exitCode": exit_code,
                "startedAt": 1,
            },
        }),
    ])
}

/// The command and arguments of a recorded `cmd` request.
fn vercel_command_argv(request: &Recorded) -> Vec<String> {
    let body = request.json();
    let mut argv = vec![body["command"].as_str().unwrap_or_default().to_string()];
    argv.extend(
        body["args"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|arg| arg.as_str().map(str::to_string)),
    );
    argv
}

/// Every recorded `cmd` request whose command is `command`.
fn vercel_commands(server: &MockServer, command: &str) -> Vec<Recorded> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd") && request.json()["command"] == command)
        .collect()
}

/// The one recorded `cmd` request whose command is `command`.
fn vercel_command(server: &MockServer, command: &str) -> Recorded {
    vercel_commands(server, command)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("no `{command}` command was run"))
}

/// Answer the fixed part of the mount flow: a healthy Mountpoint and an empty mount directory.
fn vercel_mount_handler(request: &Recorded) -> Option<MockResponse> {
    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => {
            Some(MockResponse::json(vercel_sandbox_and_session("running")))
        }
        ("GET", "/v2/sandboxes/sessions/sess-1") => {
            let mut body = vercel_sandbox_and_session("running");
            Some(MockResponse::json(serde_json::json!({
                "session": body["session"].take(),
                "routes": [],
            })))
        }
        ("POST", "/v2/sandboxes/sessions/sess-1/fs/mkdir") => {
            Some(MockResponse::json(serde_json::json!({})))
        }
        ("POST", "/v2/sandboxes/sessions/sess-1/stop") => {
            let mut body = vercel_sandbox_and_session("stopped");
            Some(MockResponse::json(
                serde_json::json!({ "session": body["session"].take() }),
            ))
        }
        ("POST", "/v2/sandboxes/sessions/sess-1/cmd") => {
            let argv = vercel_command_argv(request);
            let command = argv.first().cloned().unwrap_or_default();
            Some(match command.as_str() {
                "/usr/bin/rpm" => vercel_command_result("1.21.0", 0),
                "/usr/bin/test" => vercel_command_result("", 0),
                // `realpath -m` echoes the path back when nothing resolves through a symlink.
                "/usr/bin/realpath" => {
                    vercel_command_result(argv.last().map(String::as_str).unwrap_or_default(), 0)
                }
                // An empty mount directory prints nothing.
                "/usr/bin/find" => vercel_command_result("", 0),
                "/usr/bin/id" => vercel_command_result(
                    if argv.contains(&"-u".to_string()) {
                        "1000"
                    } else {
                        "1001"
                    },
                    0,
                ),
                _ => vercel_command_result("", 0),
            })
        }
        _ => None,
    }
}

/// A Vercel client pointed at `server`.
fn vercel_mount_client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    )
}

/// A manifest attaching `strategy` at `/vercel/sandbox/data`.
fn vercel_mount_manifest(strategy: VercelCloudBucketMountStrategy, read_only: bool) -> Manifest {
    Manifest::new()
        .with_entry(
            "data",
            Entry::mount(strategy.mount().with_read_only(read_only)),
        )
        // An authenticated mount hands a credential to the sandbox, so the manifest has to
        // acknowledge its exact path before it can activate.
        .with_in_container_mount_credential_exposure_acknowledged(["/vercel/sandbox/data"])
}

#[tokio::test]
async fn vercel_mounts_a_writable_bucket_as_the_session_user() {
    let server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
        .with_region("us-east-1")
        .with_credentials("AKIAEXAMPLE", "s3cr3t");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy, false)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let mount = vercel_command(&server, "/usr/bin/mount-s3");
    let argv = vercel_command_argv(&mount);
    assert_eq!(
        argv,
        vec![
            "/usr/bin/mount-s3",
            "my-bucket",
            "/vercel/sandbox/data",
            "--allow-other",
            "--allow-overwrite",
            "--allow-delete",
            "--uid",
            "1000",
            "--gid",
            "1001",
            "--region",
            "us-east-1",
        ]
    );
    // Mountpoint needs root, and the backend takes the escalation out of band.
    assert_eq!(mount.json()["sudo"], true);
    // The credentials travel as the command's environment, never in its argument vector.
    assert_eq!(mount.json()["env"]["AWS_ACCESS_KEY_ID"], "AKIAEXAMPLE");
    assert_eq!(mount.json()["env"]["AWS_SECRET_ACCESS_KEY"], "s3cr3t");
    assert!(!argv.join(" ").contains("s3cr3t"), "{argv:?}");

    // The ownership probe must not run as root, or the mount would be owned by root.
    let uid = vercel_commands(&server, "/usr/bin/id");
    assert_eq!(uid.len(), 2, "both the user and the group id are probed");
    assert!(uid.iter().all(|request| request.json()["sudo"].is_null()));

    // The mount path is checked before the directory exists and again after `mkdir -p`, because
    // a symlink planted in between would move the mount somewhere else.
    assert_eq!(vercel_commands(&server, "/usr/bin/realpath").len(), 2);
    // An existing file under the mount path would vanish behind the mount.
    let find = vercel_command_argv(&vercel_command(&server, "/usr/bin/find"));
    assert!(find.contains(&"-mindepth".to_string()), "{find:?}");
    // A healthy Mountpoint is left alone.
    assert!(vercel_commands(&server, "/usr/bin/dnf").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_keeps_a_read_only_mount_read_only_although_it_authenticates() {
    let server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
        .with_prefix("exports")
        .with_credentials("AKIAEXAMPLE", "s3cr3t");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy, true)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let argv = vercel_command_argv(&vercel_command(&server, "/usr/bin/mount-s3"));
    assert_eq!(
        argv,
        vec![
            "/usr/bin/mount-s3",
            "my-bucket",
            "/vercel/sandbox/data",
            "--allow-other",
            "--read-only",
            "--prefix",
            "exports/",
        ]
    );
    // Write flags follow writability, not authentication.
    assert!(!argv.contains(&"--allow-delete".to_string()), "{argv:?}");
    assert!(!argv.contains(&"--allow-overwrite".to_string()), "{argv:?}");
    // A signed mount must not ask for anonymous access.
    assert!(!argv.contains(&"--no-sign-request".to_string()), "{argv:?}");
    // A read-only mount exposes no files the session user has to own.
    assert!(vercel_commands(&server, "/usr/bin/id").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_mounts_a_public_bucket_without_credentials() {
    let server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let manifest = Manifest::new().with_entry(
        "data",
        Entry::mount(
            VercelCloudBucketMountStrategy::new("public-bucket")
                .mount()
                .with_read_only(true),
        ),
    );
    let session = vercel_mount_client(&server)
        .create(Some(manifest))
        .await
        .expect("create");
    session.start().await.expect("start");

    let mount = vercel_command(&server, "/usr/bin/mount-s3");
    let argv = vercel_command_argv(&mount);
    assert!(argv.contains(&"--no-sign-request".to_string()), "{argv:?}");
    assert!(mount.json()["env"]["AWS_ACCESS_KEY_ID"].is_null());
    server.stop();
}

#[tokio::test]
async fn vercel_installs_mountpoint_when_the_image_ships_an_old_one() {
    let probes = Arc::new(Mutex::new(0usize));
    let handler_probes = probes.clone();
    let server = MockServer::start(move |request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/rpm") {
            let mut seen = handler_probes.lock().expect("poisoned");
            *seen += 1;
            // The image ships a release this strategy cannot drive; the install fixes that.
            return vercel_command_result(if *seen == 1 { "1.20.9" } else { "1.21.0" }, 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(
            VercelCloudBucketMountStrategy::new("public-bucket"),
            true,
        )))
        .await
        .expect("create");
    session.start().await.expect("start");

    let install = vercel_command(&server, "/usr/bin/dnf");
    assert_eq!(
        vercel_command_argv(&install),
        vec![
            "/usr/bin/dnf",
            "install",
            "-y",
            "--setopt=gpgcheck=1",
            "fuse",
            "mount-s3",
        ]
    );
    // Installing a package needs root, and the package signature check stays on.
    assert_eq!(install.json()["sudo"], true);
    // The version is verified again after the install rather than assumed.
    assert_eq!(*probes.lock().expect("poisoned"), 2);
    assert!(!vercel_commands(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_refuses_an_unsupported_mountpoint_release() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/rpm") {
            // A new major release is not assumed to take the same flags.
            return vercel_command_result("2.0.0", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(
            VercelCloudBucketMountStrategy::new("public-bucket"),
            true,
        )))
        .await
        .expect("create");
    let error = session
        .start()
        .await
        .expect_err("an unsupported Mountpoint");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("2.0.0"), "{}", error.message);
    assert!(vercel_commands(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_refuses_a_mount_path_that_resolves_through_a_symlink() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/realpath")
        {
            return vercel_command_result("/vercel/sandbox/elsewhere", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(
            VercelCloudBucketMountStrategy::new("public-bucket"),
            true,
        )))
        .await
        .expect("create");
    let error = session.start().await.expect_err("a symlinked mount path");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("elsewhere"), "{}", error.message);
    assert!(vercel_commands(&server, "/usr/bin/mount-s3").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_refuses_to_mount_over_a_populated_directory() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/find") {
            return vercel_command_result("/vercel/sandbox/data/existing.txt", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(
            VercelCloudBucketMountStrategy::new("public-bucket"),
            true,
        )))
        .await
        .expect("create");
    let error = session.start().await.expect_err("a populated mount path");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("empty"), "{}", error.message);
    server.stop();
}

#[tokio::test]
async fn vercel_redacts_a_failing_mount_command_and_stops_the_sandbox() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/mount-s3")
        {
            // Mountpoint is the one command that has seen the credentials.
            return MockResponse::ndjson(&[
                serde_json::json!({
                    "stream": "stderr",
                    "data": "Error: failed to sign request with AKIAEXAMPLE/s3cr3t\n",
                }),
                serde_json::json!({
                    "command": {
                        "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                        "sessionId": "sess-1", "exitCode": 1, "startedAt": 1,
                    },
                }),
            ]);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy =
        VercelCloudBucketMountStrategy::new("my-bucket").with_credentials("AKIAEXAMPLE", "s3cr3t");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy, true)))
        .await
        .expect("create");
    let error = session.start().await.expect_err("the mount fails");
    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(!error.message.contains("AKIAEXAMPLE"), "{}", error.message);
    assert!(!error.message.contains("s3cr3t"), "{}", error.message);
    assert!(
        !error.message.contains("failed to sign"),
        "{}",
        error.message
    );
    assert!(error.message.contains("exit code 1"), "{}", error.message);
    // A privileged mount that failed part way leaves a state nothing can reason about.
    assert!(
        server
            .requests()
            .iter()
            .any(|request| request.path.ends_with("/stop")),
        "the sandbox should be stopped"
    );
    server.stop();
}

#[tokio::test]
async fn vercel_refuses_to_attach_a_mount_after_the_sandbox_started() {
    let server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy.clone(), true)))
        .await
        .expect("create");
    session.start().await.expect("start");
    let mounted = vercel_commands(&server, "/usr/bin/mount-s3").len();

    // Vercel fixes its mount topology when the sandbox is created, so a manifest applied to a
    // running session must not be able to hand it a fresh bucket.
    let mount = strategy.clone().mount().with_read_only(true);
    let error = strategy
        .activate(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect_err("a late mount");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("fixed"), "{}", error.message);
    assert_eq!(vercel_commands(&server, "/usr/bin/mount-s3").len(), mounted);
    server.stop();
}

#[tokio::test]
async fn vercel_detaches_only_the_mounts_it_attached() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/findmnt") {
            return vercel_command_result("mountpoint-s3", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy.clone(), true)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let mount = strategy.clone().mount().with_read_only(true);
    // A path this session never mounted is left alone rather than unmounted blindly.
    strategy
        .deactivate(&mount, session.as_ref(), "/vercel/sandbox/other")
        .await
        .expect("an untracked path is a no-op");
    assert!(vercel_commands(&server, "/usr/bin/findmnt").is_empty());

    strategy
        .deactivate(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect("deactivate");
    let unmount = vercel_command(&server, "/usr/bin/umount");
    assert_eq!(
        vercel_command_argv(&unmount),
        vec!["/usr/bin/umount", "/vercel/sandbox/data"]
    );
    assert_eq!(unmount.json()["sudo"], true);

    // The mount is gone, so detaching it again does nothing.
    let unmounts = vercel_commands(&server, "/usr/bin/umount").len();
    strategy
        .deactivate(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect("a detached path is a no-op");
    assert_eq!(vercel_commands(&server, "/usr/bin/umount").len(), unmounts);
    server.stop();
}

#[tokio::test]
async fn vercel_refuses_to_unmount_a_filesystem_it_did_not_create() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/findmnt") {
            // Something else took the mount path while the session was running.
            return vercel_command_result("/dev/nvme0n1p1", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy.clone(), true)))
        .await
        .expect("create");
    session.start().await.expect("start");

    let mount = strategy.clone().mount().with_read_only(true);
    let error = strategy
        .deactivate(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect_err("a foreign filesystem");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("unexpected filesystem"),
        "{}",
        error.message
    );
    assert!(vercel_commands(&server, "/usr/bin/umount").is_empty());
    server.stop();
}

#[tokio::test]
async fn vercel_detaches_and_restores_a_mount_around_a_snapshot() {
    let server = MockServer::start(|request| {
        let argv = vercel_command_argv(request);
        if request.path.ends_with("/cmd") && argv.first().is_some_and(|c| c == "/usr/bin/findmnt") {
            return vercel_command_result("mountpoint-s3", 0);
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy.clone(), true)))
        .await
        .expect("create");
    session.start().await.expect("start");
    assert_eq!(vercel_commands(&server, "/usr/bin/mount-s3").len(), 1);

    // A native snapshot cannot capture a live FUSE mount, so it is detached first.
    let mount = strategy.clone().mount().with_read_only(true);
    strategy
        .teardown_for_snapshot(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect("teardown");
    assert_eq!(
        vercel_command(&server, "/usr/bin/umount").json()["sudo"],
        true
    );

    // Re-attaching happens after the sandbox started, so it is the detached state, not the
    // start window, that authorizes it.
    strategy
        .restore_after_snapshot(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect("restore");
    assert_eq!(vercel_commands(&server, "/usr/bin/mount-s3").len(), 2);

    // Restoring again is a no-op: the mount is attached, not detached.
    strategy
        .restore_after_snapshot(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect("a second restore");
    assert_eq!(vercel_commands(&server, "/usr/bin/mount-s3").len(), 2);

    // The snapshot bypass does not reopen the start window for a new mount.
    let error = strategy
        .activate(&mount, session.as_ref(), "/vercel/sandbox/data")
        .await
        .expect_err("a late mount");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Exposed ports, the E2B MCP option and the Vercel network policy wire shape
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn e2b_sends_the_mcp_configuration_and_resolves_an_exposed_port() {
    let server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "sandboxID": "sbx-e2b",
                    "envdAccessToken": "envd-token",
                })),
                _ => MockResponse::error(404, "unexpected"),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base_url)
            .with_domain("e2b.test")
            .with_exposed_port(3000)
            .with_mcp_server("github", [("url", "https://mcp.example/github")]),
    );
    let session = client.create(None).await.expect("create");

    // The MCP mapping travels verbatim as `NewSandbox.mcp`.
    let create = server.find("POST", "/v2/sandboxes");
    assert_eq!(
        create.json()["mcp"],
        serde_json::json!({ "github": { "url": "https://mcp.example/github" } })
    );

    // `get_host` is client-side string building, so resolving issues no request.
    let endpoint = session.resolve_exposed_port(3000).await.expect("resolve");
    assert_eq!(endpoint.host, "3000-sbx-e2b.e2b.test");
    assert!(endpoint.tls);
    assert_eq!(endpoint.port, 443);
    assert_eq!(
        endpoint.url_for("http").unwrap(),
        "https://3000-sbx-e2b.e2b.test/"
    );

    // A port the client never declared is refused before any provider call.
    let error = session
        .resolve_exposed_port(8080)
        .await
        .expect_err("undeclared");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
    server.stop();
}

#[tokio::test]
async fn daytona_resolves_an_exposed_port_through_a_signed_preview_url() {
    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            ("GET", "/sandbox/sbx-daytona/ports/3000/signed-preview-url") => {
                MockResponse::json(serde_json::json!({
                    "url": "https://3000-sbx-daytona.proxy.daytona.works/?token=abc",
                    "token": "abc",
                }))
            }
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = server.base_url.clone();

    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dt_test")
            .with_api_url(&server.base_url)
            .with_exposed_port(3000)
            .with_exposed_port_url_ttl_secs(120),
    )
    .create(None)
    .await
    .expect("create");
    let endpoint = session.resolve_exposed_port(3000).await.expect("resolve");
    assert_eq!(endpoint.host, "3000-sbx-daytona.proxy.daytona.works");
    assert!(endpoint.tls);
    // The signature is part of the URL, so it has to survive into the endpoint.
    assert_eq!(endpoint.query, "token=abc");
    assert_eq!(
        endpoint.url_for("http").unwrap(),
        "https://3000-sbx-daytona.proxy.daytona.works/?token=abc"
    );

    let preview = server.find("GET", "/sandbox/sbx-daytona/ports/3000/signed-preview-url");
    assert!(
        preview.query.contains("expiresInSeconds=120"),
        "{}",
        preview.query
    );
    server.stop();
}

#[tokio::test]
async fn vercel_resolves_an_exposed_port_from_the_create_routes() {
    let server = MockServer::start(|request| match (request.method.as_str(), request.path.as_str())
    {
        ("POST", "/v3/sandboxes") => {
            let mut body = vercel_sandbox_and_session("running");
            body["routes"] = serde_json::json!([
                { "port": 3000, "subdomain": "agents-test-3000", "url": "https://agents-test-3000.vercel.run" },
            ]);
            MockResponse::json(body)
        }
        ("GET", "/v2/sandboxes/sessions/sess-1") => {
            let mut body = vercel_sandbox_and_session("running");
            MockResponse::json(serde_json::json!({ "session": body["session"].take() }))
        }
        ("POST", "/v2/sandboxes/sessions/sess-1/fs/mkdir") => {
            MockResponse::json(serde_json::json!({}))
        }
        _ => MockResponse::error(404, "unexpected"),
    })
    .await;

    let session = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1))
            .with_exposed_port(3000)
            .with_exposed_port(8080),
    )
    .create(None)
    .await
    .expect("create");
    let endpoint = session.resolve_exposed_port(3000).await.expect("resolve");
    assert_eq!(endpoint.host, "agents-test-3000.vercel.run");
    assert_eq!(endpoint.port, 443);
    assert!(endpoint.tls);

    // A declared port the API published no route for is a backend failure, not a
    // configuration error.
    let error = session
        .resolve_exposed_port(8080)
        .await
        .expect_err("no route");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("backend_unavailable")
    );
    // A port the client never declared is refused before the routes are consulted.
    let undeclared = session
        .resolve_exposed_port(9090)
        .await
        .expect_err("undeclared");
    assert_eq!(
        undeclared.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
    server.stop();
}

#[tokio::test]
async fn vercel_sends_the_mode_form_for_allow_all_and_deny_all() {
    for (policy, expected) in [
        (NetworkPolicy::AllowAll, "allow-all"),
        (NetworkPolicy::DenyAll, "deny-all"),
    ] {
        let server =
            MockServer::start(
                |request| match (request.method.as_str(), request.path.as_str()) {
                    ("POST", "/v3/sandboxes") => {
                        MockResponse::json(vercel_sandbox_and_session("running"))
                    }
                    ("GET", "/v2/sandboxes/sessions/sess-1") => {
                        let mut body = vercel_sandbox_and_session("running");
                        MockResponse::json(serde_json::json!({ "session": body["session"].take() }))
                    }
                    _ => MockResponse::error(404, "unexpected"),
                },
            )
            .await;

        VercelSandboxClient::with_options(
            VercelSandboxClientOptions::new()
                .with_token("vercel_test")
                .with_base_url(&server.base_url)
                .with_status_poll_interval(Duration::from_millis(1))
                .with_network_policy(policy),
        )
        .create(None)
        .await
        .expect("create");

        // `POST /v3/sandboxes` types `networkPolicy` as a `oneOf`; `allow: "all"` matches
        // neither branch, while the mode form is the first one.
        let create = server.find("POST", "/v3/sandboxes");
        assert_eq!(
            create.json()["networkPolicy"],
            serde_json::json!({ "mode": expected })
        );
        server.stop();
    }
}

#[tokio::test]
async fn vercel_refuses_to_start_a_mounted_session_twice() {
    let server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
        .with_region("us-east-1")
        .with_credentials("AKIAEXAMPLE", "s3cr3t");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy, false)))
        .await
        .expect("create");
    session.start().await.expect("start");

    // The mount topology is fixed at creation, so a second start would re-present the
    // credential to a sandbox that has already run model-controlled code.
    let error = session.start().await.expect_err("a second start");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("starting the same mounted session"),
        "{}",
        error.message
    );
    server.stop();
}

#[tokio::test]
async fn vercel_lets_a_mounted_session_start_again_after_an_early_failure() {
    // The first `mkdir` fails, so `start` gives up before any mount attaches and no credential
    // reached the sandbox. That start must be retryable, not leave the session unusable.
    let failed_once = Arc::new(Mutex::new(false));
    let seen = failed_once.clone();
    let server = MockServer::start(move |request| {
        if request.path == "/v2/sandboxes/sessions/sess-1/cmd"
            && vercel_command_argv(request).first().map(String::as_str) == Some("/usr/bin/mkdir")
        {
            let mut seen = seen.lock().expect("poisoned");
            if !*seen {
                *seen = true;
                return MockResponse::error(503, "the runtime is restarting");
            }
        }
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;

    let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
        .with_region("us-east-1")
        .with_credentials("AKIAEXAMPLE", "s3cr3t");
    let session = vercel_mount_client(&server)
        .create(Some(vercel_mount_manifest(strategy, false)))
        .await
        .expect("create");

    session.start().await.expect_err("the first start fails");
    // No mount attached, so the gate is still open.
    session.start().await.expect("the retry starts the session");
    assert_eq!(vercel_commands(&server, "/usr/bin/mount-s3").len(), 1);

    // Once a mount did attach, a further start is refused.
    let error = session.start().await.expect_err("a third start");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// The core hooks the sandbox-core area added: write grants and exposed ports through the trait
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn providers_refuse_a_write_through_a_read_only_path_grant() {
    // `write_file` is implemented per backend, so the read-only grant check has to be requested
    // by each of them. These three share one assertion because the rule is the SDK's, not the
    // provider's: a grant marked read-only must refuse a write the backend would otherwise
    // happily perform.
    let grant = SandboxPathGrant::new("/shared/dataset")
        .expect("grant")
        .read_only(true);
    let manifest = Manifest::new().with_extra_path_grant(grant);

    let e2b_server =
        MockServer::start(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v2/sandboxes") => MockResponse::json(serde_json::json!({
                    "sandboxID": "sbx-e2b",
                    "envdAccessToken": "t",
                })),
                _ => MockResponse::error(404, "unexpected"),
            },
        )
        .await;
    let e2b = e2b_client(&e2b_server.base_url)
        .create(Some(manifest.clone()))
        .await
        .expect("create");
    let error = e2b
        .write_file("/shared/dataset/out.txt", b"x")
        .await
        .expect_err("a write through a read-only grant");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    // A read through the same grant is still allowed.
    assert!(
        e2b.normalize_path("/shared/dataset/out.txt").is_ok(),
        "the grant still permits reads"
    );
    e2b_server.stop();

    let vercel_server = MockServer::start(|request| {
        vercel_mount_handler(request).unwrap_or_else(|| MockResponse::error(404, "unexpected"))
    })
    .await;
    let vercel = vercel_mount_client(&vercel_server)
        .create(Some(manifest.clone()))
        .await
        .expect("create");
    assert_eq!(
        vercel
            .write_file("/shared/dataset/out.txt", b"x")
            .await
            .expect_err("a write through a read-only grant")
            .code,
        ErrorCode::WorkspaceArchiveWriteError
    );
    vercel_server.stop();

    let base = Arc::new(Mutex::new(String::new()));
    let handler_base = base.clone();
    let daytona_server = MockServer::start(move |request| {
        let base_url = handler_base.lock().expect("poisoned").clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => {
                MockResponse::json(daytona_sandbox(&base_url))
            }
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await;
    *base.lock().expect("poisoned") = daytona_server.base_url.clone();
    let daytona = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dt_test")
            .with_api_url(&daytona_server.base_url),
    )
    .create(Some(manifest))
    .await
    .expect("create");
    assert_eq!(
        daytona
            .write_file("/shared/dataset/out.txt", b"x")
            .await
            .expect_err("a write through a read-only grant")
            .code,
        ErrorCode::WorkspaceArchiveWriteError
    );
    daytona_server.stop();
}

#[tokio::test]
async fn vercel_fails_a_create_once_on_a_permanent_status() {
    // `http.rs` now classifies the status, so a 402 is not retried three times while a 503 is.
    for (status, expected_attempts) in [(402u16, 1usize), (503u16, 3usize)] {
        let server = MockServer::start(move |request| {
            match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v3/sandboxes") => MockResponse::error(status, "nope"),
                _ => MockResponse::error(404, "unexpected"),
            }
        })
        .await;
        let error = VercelSandboxClient::with_options(
            VercelSandboxClientOptions::new()
                .with_token("vercel_test")
                .with_base_url(&server.base_url)
                .with_status_poll_interval(Duration::from_millis(1)),
        )
        .create(None)
        .await
        .expect_err("create fails");
        assert_eq!(
            error.context.get("http_status").map(String::as_str),
            Some(status.to_string().as_str())
        );
        assert_eq!(
            server
                .requests()
                .iter()
                .filter(|request| request.path == "/v3/sandboxes")
                .count(),
            expected_attempts,
            "status {status}"
        );
        server.stop();
    }
}
