//! Shared mock-server helpers for the ported Python sandbox-provider tests.
//!
//! Python's `tests/extensions/sandbox/` suite patches the vendor SDK objects; the Rust providers
//! speak REST (or, for Modal, a Python helper over stdio), so the same observable behavior is
//! asserted here against a local `TcpListener` that records every request and answers from a
//! per-test handler. The module is a directory module, so Cargo does not compile it as a test
//! crate of its own; each `tests/py_sandbox_*.rs` file pulls it in with `#[path]`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// One request the mock server received.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn query_param(&self, name: &str) -> Option<String> {
        self.query.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == name).then(|| percent_decode(value))
        })
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// `method path`, for assertion messages.
    pub fn route(&self) -> String {
        format!("{} {}", self.method, self.path)
    }
}

/// Decode the small subset of percent-encoding the clients emit in query strings.
pub fn percent_decode(value: &str) -> String {
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
pub struct MockResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
    /// Extra body parts written after a delay, with no `Content-Length`, for streaming replies.
    pub stream: Vec<(Vec<u8>, u64)>,
    /// Extra response headers.
    pub headers: Vec<(String, String)>,
}

impl MockResponse {
    pub fn json(value: serde_json::Value) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/json",
            body: serde_json::to_vec(&value).expect("serializable"),
            stream: Vec::new(),
            headers: Vec::new(),
        }
    }

    pub fn status(status: u16, value: serde_json::Value) -> Self {
        MockResponse {
            status,
            ..MockResponse::json(value)
        }
    }

    pub fn empty(status: u16) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            body: Vec::new(),
            stream: Vec::new(),
            headers: Vec::new(),
        }
    }

    pub fn text(body: impl Into<String>) -> Self {
        MockResponse {
            status: 200,
            content_type: "text/plain",
            body: body.into().into_bytes(),
            stream: Vec::new(),
            headers: Vec::new(),
        }
    }

    pub fn bytes(body: impl Into<Vec<u8>>) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/octet-stream",
            body: body.into(),
            stream: Vec::new(),
            headers: Vec::new(),
        }
    }

    pub fn ndjson(lines: &[serde_json::Value]) -> Self {
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
            headers: Vec::new(),
        }
    }

    pub fn error(status: u16, message: &str) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(&serde_json::json!({ "code": status, "message": message }))
                .expect("serializable"),
            stream: Vec::new(),
            headers: Vec::new(),
        }
    }

    /// A reply whose body arrives in parts, each after the given delay in milliseconds.
    pub fn streamed(content_type: &'static str, parts: Vec<(Vec<u8>, u64)>) -> Self {
        MockResponse {
            status: 200,
            content_type,
            body: Vec::new(),
            stream: parts,
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_content_type(mut self, content_type: &'static str) -> Self {
        self.content_type = content_type;
        self
    }
}

pub type Handler = Arc<dyn Fn(&Recorded) -> MockResponse + Send + Sync>;

/// A local HTTP server that records every request and answers from a closure.
pub struct MockServer {
    pub base_url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MockServer {
    pub async fn start<F>(handler: F) -> Self
    where
        F: Fn(&Recorded) -> MockResponse + Send + Sync + 'static,
    {
        let handler: Handler = Arc::new(handler);
        MockServer::start_raw(move |recorded, mut stream| {
            let handler = handler.clone();
            async move {
                let response = handler(&recorded);
                let _ = write_response(&mut stream, response).await;
            }
        })
        .await
    }

    /// Start a server whose per-connection task owns the socket after the request is parsed.
    ///
    /// Websocket tests need this: they answer the upgrade by hand and then speak frames.
    pub async fn start_raw<F, Fut>(serve: F) -> Self
    where
        F: Fn(Recorded, TcpStream) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the mock server");
        let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
        let requests: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));
        let (shutdown, mut stop) = tokio::sync::oneshot::channel();
        let recorder = requests.clone();
        let serve = Arc::new(serve);
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted,
                    _ = &mut stop => break,
                };
                let Ok((mut stream, _)) = accepted else { break };
                let serve = serve.clone();
                let recorder = recorder.clone();
                tokio::spawn(async move {
                    let Ok(recorded) = read_request(&mut stream).await else {
                        return;
                    };
                    recorder.lock().expect("poisoned").push(recorded.clone());
                    serve(recorded, stream).await;
                });
            }
        });
        MockServer {
            base_url,
            requests,
            shutdown: Some(shutdown),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().expect("poisoned").clone()
    }

    pub fn routes(&self) -> Vec<String> {
        self.requests().iter().map(Recorded::route).collect()
    }

    /// Every request matching `method` and `path`, in arrival order.
    pub fn all(&self, method: &str, path: &str) -> Vec<Recorded> {
        self.requests()
            .into_iter()
            .filter(|request| request.method == method && request.path == path)
            .collect()
    }

    /// The first request matching `method` and `path`.
    pub fn find(&self, method: &str, path: &str) -> Recorded {
        self.requests()
            .into_iter()
            .find(|request| request.method == method && request.path == path)
            .unwrap_or_else(|| panic!("no `{method} {path}` request; saw {:?}", self.routes()))
    }

    pub fn find_opt(&self, method: &str, path: &str) -> Option<Recorded> {
        self.requests()
            .into_iter()
            .find(|request| request.method == method && request.path == path)
    }

    pub fn count(&self, method: &str, path: &str) -> usize {
        self.all(method, path).len()
    }

    pub fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

/// Write one `MockResponse` and close the connection.
pub async fn write_response(stream: &mut TcpStream, response: MockResponse) -> std::io::Result<()> {
    let mut extra = String::new();
    for (name, value) in &response.headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    if response.stream.is_empty() {
        let head = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n{extra}Connection: \
             close\r\n\r\n",
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
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\n{extra}Connection: close\r\n\r\n",
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

/// Write a JSON response and close the connection.
pub async fn write_json(stream: &mut TcpStream, value: &serde_json::Value) -> std::io::Result<()> {
    write_response(stream, MockResponse::json(value.clone())).await
}

/// Write a bodyless response and close the connection.
pub async fn write_status(stream: &mut TcpStream, status: u16) -> std::io::Result<()> {
    write_response(stream, MockResponse::empty(status)).await
}

/// Read one request from `stream`, stopping at the end of its body.
pub async fn read_request(stream: &mut TcpStream) -> std::io::Result<Recorded> {
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

pub fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Complete a websocket upgrade by hand and return the server-side stream.
pub async fn accept_websocket(
    mut stream: TcpStream,
    request: &Recorded,
    subprotocol: Option<&str>,
) -> tokio_tungstenite::WebSocketStream<TcpStream> {
    use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
    use tokio_tungstenite::tungstenite::protocol::Role;

    let key = request
        .header("sec-websocket-key")
        .expect("the handshake key")
        .to_string();
    let protocol = subprotocol
        .map(|value| format!("Sec-WebSocket-Protocol: {value}\r\n"))
        .unwrap_or_default();
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: \
         Upgrade\r\nSec-WebSocket-Accept: {}\r\n{protocol}\r\n",
        derive_accept_key(key.as_bytes())
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write the upgrade response");
    tokio_tungstenite::WebSocketStream::from_raw_socket(stream, Role::Server, None).await
}

/// Base64-encode with the standard alphabet, as the provider wire formats do.
pub fn b64(bytes: impl AsRef<[u8]>) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes.as_ref())
}

/// Decode a standard-alphabet base64 payload.
pub fn unb64(text: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64")
}

/// A counter the handlers use to vary a reply by attempt.
#[derive(Clone, Default)]
pub struct Counter(Arc<Mutex<HashMap<String, usize>>>);

impl Counter {
    pub fn new() -> Self {
        Counter::default()
    }

    /// Increment `key` and return the value *before* the increment.
    pub fn next(&self, key: &str) -> usize {
        let mut map = self.0.lock().expect("poisoned");
        let slot = map.entry(key.to_string()).or_insert(0);
        let current = *slot;
        *slot += 1;
        current
    }

    pub fn get(&self, key: &str) -> usize {
        *self.0.lock().expect("poisoned").get(key).unwrap_or(&0)
    }
}

/// Shared, mutable state a handler closure can read and write.
#[derive(Clone)]
pub struct Shared<T>(Arc<Mutex<T>>);

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Shared(Arc::new(Mutex::new(value)))
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.0.lock().expect("poisoned"))
    }
}

impl<T: Clone> Shared<T> {
    pub fn get(&self) -> T {
        self.0.lock().expect("poisoned").clone()
    }

    pub fn set(&self, value: T) {
        *self.0.lock().expect("poisoned") = value;
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Shared::new(T::default())
    }
}

// ---------------------------------------------------------------------------------------------
// In-sandbox path resolver
// ---------------------------------------------------------------------------------------------

/// The wire format a provider answers a command with, for [`start_resolving`].
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Resolver {
    /// Cloudflare: a Server-Sent Events stream with `stdout` and `exit` events.
    Cloudflare,
    /// E2B: Connect-framed `process.Process/Start` events.
    E2b,
    /// Daytona: a toolbox session command result.
    Daytona,
}

const RESOLVER_HELPER_PATH: &str = "resolve-workspace-path";

/// The reply the in-sandbox resolver (and its presence probe) gets from a healthy sandbox.
#[allow(dead_code)]
pub fn resolver_reply(kind: Resolver) -> MockResponse {
    resolver_result(kind, 0, "/resolved\n", "")
}

/// A command result with the given exit code and output, in the wire format of `kind`.
pub fn resolver_result(kind: Resolver, exit_code: i64, stdout: &str, stderr: &str) -> MockResponse {
    match kind {
        Resolver::Cloudflare => {
            let mut body = String::new();
            if !stdout.is_empty() {
                body.push_str(&format!("event: stdout\ndata: {}\n\n", b64(stdout)));
            }
            if !stderr.is_empty() {
                body.push_str(&format!("event: stderr\ndata: {}\n\n", b64(stderr)));
            }
            body.push_str(&format!(
                "event: exit\ndata: {{\"exit_code\":{exit_code}}}\n\n"
            ));
            MockResponse::text(body).with_content_type("text/event-stream")
        }
        Resolver::E2b => {
            fn frame(flags: u8, value: serde_json::Value) -> Vec<u8> {
                let payload = serde_json::to_vec(&value).expect("serializable");
                let mut frame = vec![flags];
                frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
                frame.extend_from_slice(&payload);
                frame
            }
            let mut body = frame(0, serde_json::json!({ "event": { "start": { "pid": 1 } } }));
            body.extend(frame(
                0,
                serde_json::json!({ "event": { "data": { "stdout": b64(stdout), "stderr": b64(stderr) } } }),
            ));
            body.extend(frame(
                0,
                serde_json::json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
            ));
            body.extend(frame(2, serde_json::json!({})));
            MockResponse {
                status: 200,
                content_type: "application/connect+json",
                body,
                stream: Vec::new(),
                headers: Vec::new(),
            }
        }
        Resolver::Daytona => MockResponse::json(serde_json::json!({
            "output": stdout, "stdout": stdout, "stderr": stderr, "exitCode": exit_code,
        })),
    }
}

/// Whether `request` runs (or probes for) the in-sandbox path resolver.
#[allow(dead_code)]
pub fn is_resolver_request(request: &Recorded) -> bool {
    request.method == "POST" && request.text().contains(RESOLVER_HELPER_PATH)
}

/// Start a mock server that plays the sandbox's own path resolver before `handler` sees a request.
///
/// Every read, write, mkdir, rm, ls and mv first runs the resolver inside the sandbox, as Python
/// does, so a provider mock has to answer it.
#[allow(dead_code)]
pub async fn start_resolving<F>(kind: Resolver, handler: F) -> MockServer
where
    F: Fn(&Recorded) -> MockResponse + Send + Sync + 'static,
{
    MockServer::start(move |request: &Recorded| {
        if is_resolver_request(request) {
            return resolver_reply(kind);
        }
        handler(request)
    })
    .await
}

/// The argument vector of a command a provider was asked to run, whatever its wire format.
pub fn command_argv(request: &Recorded) -> Vec<String> {
    let body = if request.body.len() > 5 && request.body[0] == 0 {
        // A Connect envelope: a flag byte, a length, then the JSON payload.
        serde_json::from_slice::<serde_json::Value>(&request.body[5..]).unwrap_or_default()
    } else {
        request.json()
    };
    if let Some(argv) = body.get("argv").and_then(|v| v.as_array()) {
        return argv
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
    if let Some(process) = body.get("process") {
        let mut argv = vec![process["cmd"].as_str().unwrap_or_default().to_string()];
        argv.extend(
            process["args"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string)),
        );
        return argv;
    }
    body.get("command")
        .and_then(|v| v.as_str())
        .map(shell_split)
        .unwrap_or_default()
}

/// Split a command line quoted the way the SDK quotes it (single quotes, `'\''` escapes).
pub fn shell_split(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut started = false;
    let mut in_quote = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_quote = !in_quote;
                started = true;
            }
            '\\' if !in_quote => {
                if let Some(next) = chars.next() {
                    current.push(next);
                    started = true;
                }
            }
            c if c.is_whitespace() && !in_quote => {
                if started {
                    words.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            c => {
                current.push(c);
                started = true;
            }
        }
    }
    if started {
        words.push(current);
    }
    words
}

/// The path resolver as a sandbox with the given workspace symlinks (`link -> target`) runs it:
/// the exit code, stdout and stderr of `resolve-workspace-path root path mode [grant ro]...`.
pub fn fake_resolve(
    symlinks: &HashMap<String, String>,
    request: &Recorded,
) -> (i64, String, String) {
    let argv = command_argv(request);
    let Some(at) = argv
        .iter()
        .position(|a| a.contains("resolve-workspace-path"))
    else {
        return (0, "/resolved\n".to_string(), String::new());
    };
    let tokens = &argv[at + 1..];
    if tokens.len() < 3 {
        return (0, "/resolved\n".to_string(), String::new());
    }
    let (root, lexical, mode) = (tokens[0].as_str(), tokens[1].as_str(), tokens[2].as_str());
    let mut resolved = lexical.to_string();
    for (link, target) in symlinks {
        if lexical == link || lexical.starts_with(&format!("{link}/")) {
            resolved = format!("{target}{}", &lexical[link.len()..]);
        }
    }
    let within = |path: &str, base: &str| {
        path == base || path.starts_with(&format!("{}/", base.trim_end_matches('/')))
    };
    let grants: Vec<(&str, bool)> = tokens[3..]
        .chunks(2)
        .filter(|pair| pair.len() == 2)
        .map(|pair| (pair[0].as_str(), pair[1] == "1"))
        .collect();
    if mode != "0"
        && let Some((grant, _)) = grants
            .iter()
            .find(|(grant, read_only)| *read_only && within(&resolved, grant))
    {
        return (
            114,
            String::new(),
            format!("read-only extra path grant: {grant}\nresolved path: {resolved}\n"),
        );
    }
    let allowed =
        within(&resolved, root) || grants.iter().any(|(grant, _)| within(&resolved, grant));
    if !allowed {
        return (111, String::new(), format!("workspace escape: {resolved}"));
    }
    (0, resolved, String::new())
}

/// Like [`start_resolving`], but the resolver follows `symlinks` the way a real sandbox does.
pub async fn start_resolving_links<F>(
    kind: Resolver,
    symlinks: Shared<HashMap<String, String>>,
    handler: F,
) -> MockServer
where
    F: Fn(&Recorded) -> MockResponse + Send + Sync + 'static,
{
    MockServer::start(move |request: &Recorded| {
        if is_resolver_request(request) {
            let links = symlinks.get();
            let (code, out, err) = fake_resolve(&links, request);
            return resolver_result(kind, code, &out, &err);
        }
        handler(request)
    })
    .await
}
