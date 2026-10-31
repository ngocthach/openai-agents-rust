//! Port of `test_bounded_provider_read_closes_at_prefix` (`tests/sandbox/test_bounded_read.py`)
//! across every provider and every limit: the read returns only the requested prefix and the
//! client hangs up instead of draining an endless body. Helpers are copied from
//! `py_sandbox_b_bounded_read_providers.rs`.
#![cfg(feature = "sandbox")]
#![allow(dead_code, unused_imports)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelSandboxClient, BlaxelSandboxClientOptions, BlaxelTimeouts,
};
use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareSandboxClient, CloudflareSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions, DaytonaSandboxTimeouts,
};
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxType,
};
use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession, SandboxSessionEvent};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError, SandboxResult};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// The body a provider answers with. It must never reach an error.
const PRIVATE: &str = "synthetic-private-response";

/// How the file endpoint answers.
#[derive(Clone)]
enum Reply {
    /// Headers and a complete body.
    Full { status: u16, body: Vec<u8> },
    /// Headers, an optional prefix, then blocks until the client hangs up.
    Endless { status: u16, prefix: Vec<u8> },
    /// Headers, an optional prefix, then nothing until the client hangs up.
    Stall { status: u16, prefix: Vec<u8> },
    /// Nothing at all until the client hangs up: the request never gets an answer.
    NoAnswer,
}

/// What the server observed.
#[derive(Default)]
struct Flags {
    /// The response headers were written.
    headers_sent: AtomicBool,
    /// Some body bytes were written.
    body_sent: AtomicBool,
    /// The client closed the connection before the response was complete.
    hung_up: AtomicBool,
}

struct Mock {
    base: String,
    flags: Arc<Flags>,
}

type IsFile = fn(&str, &str) -> bool;
type Control = fn(&str, &str, &str) -> Option<Value>;

async fn serve(reply: Reply, is_file: IsFile, control: Control) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let flags = Arc::new(Flags::default());
    let (accept_base, accept_flags) = (base.clone(), flags.clone());
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let (reply, base, flags) = (reply.clone(), accept_base.clone(), accept_flags.clone());
            tokio::spawn(async move {
                handle(stream, reply, is_file, control, &base, &flags).await;
            });
        }
    });
    Mock { base, flags }
}

async fn read_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") && head.len() < 65536 {
        match stream.read(&mut byte).await {
            Ok(1) => head.push(byte[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&head).into_owned()
}

/// Wait until the client closes its end. A client that has finished reading never does.
async fn until_hung_up(stream: &mut TcpStream) {
    let mut buf = [0u8; 512];
    loop {
        match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => continue,
        }
    }
}

async fn handle(
    mut stream: TcpStream,
    reply: Reply,
    is_file: IsFile,
    control: Control,
    base: &str,
    flags: &Flags,
) {
    let head = read_head(&mut stream).await;
    let mut line = head.lines().next().unwrap_or("").split_whitespace();
    let method = line.next().unwrap_or("").to_string();
    // The route is matched without its query string; the query is the provider's own argument.
    let path = line
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();

    if !is_file(&method, &path) {
        // Every read first runs the path resolver inside the sandbox; answer it in the
        // provider's own wire format.
        let content_length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        let mut body = vec![0u8; content_length];
        if content_length > 0 && stream.read_exact(&mut body).await.is_err() {
            return;
        }
        if String::from_utf8_lossy(&body).contains("resolve-workspace-path") {
            let kind = if path.ends_with("/process.Process/Start") {
                Some(support::Resolver::E2b)
            } else if path.starts_with("/v1/sandbox/") && path.ends_with("/exec") {
                Some(support::Resolver::Cloudflare)
            } else if path.contains("/process/session/") && path.ends_with("/exec") {
                Some(support::Resolver::Daytona)
            } else {
                None
            };
            if let Some(kind) = kind {
                let _ = support::write_response(&mut stream, support::resolver_reply(kind)).await;
                return;
            }
        }
        let (status, body) = match control(&method, &path, base) {
            Some(json) => (200, json.to_string()),
            None => (404, "{}".to_string()),
        };
        let response = format!(
            "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes()).await;
        let _ = stream.shutdown().await;
        return;
    }

    match reply {
        Reply::Full { status, body } => {
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            flags.headers_sent.store(true, Ordering::SeqCst);
            let _ = stream.write_all(head.as_bytes()).await;
            if !body.is_empty() {
                flags.body_sent.store(true, Ordering::SeqCst);
                let _ = stream.write_all(&body).await;
            }
            let _ = stream.shutdown().await;
        }
        Reply::Endless { status, prefix } => {
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: 1099511627776\r\nConnection: close\r\n\r\n"
            );
            flags.headers_sent.store(true, Ordering::SeqCst);
            if stream.write_all(head.as_bytes()).await.is_err() {
                flags.hung_up.store(true, Ordering::SeqCst);
                return;
            }
            if !prefix.is_empty() {
                flags.body_sent.store(true, Ordering::SeqCst);
                let _ = stream.write_all(&prefix).await;
            }
            let block = vec![b'x'; 16 * 1024];
            loop {
                if stream.write_all(&block).await.is_err() {
                    flags.hung_up.store(true, Ordering::SeqCst);
                    break;
                }
                flags.body_sent.store(true, Ordering::SeqCst);
            }
        }
        Reply::Stall { status, prefix } => {
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: 100000\r\nConnection: close\r\n\r\n"
            );
            flags.headers_sent.store(true, Ordering::SeqCst);
            let _ = stream.write_all(head.as_bytes()).await;
            if !prefix.is_empty() {
                flags.body_sent.store(true, Ordering::SeqCst);
                let _ = stream.write_all(&prefix).await;
            }
            until_hung_up(&mut stream).await;
            flags.hung_up.store(true, Ordering::SeqCst);
        }
        Reply::NoAnswer => {
            until_hung_up(&mut stream).await;
            flags.hung_up.store(true, Ordering::SeqCst);
        }
    }
}

async fn wait_for(flag: &AtomicBool) -> bool {
    for _ in 0..500 {
        if flag.load(Ordering::SeqCst) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

fn data() -> Vec<u8> {
    b"\x00\xffbinary\n".to_vec()
}

// ----------------------------------------------------------------------------------------------
// Providers: the REST surface each backend reads files through
// ----------------------------------------------------------------------------------------------

fn e2b_is_file(method: &str, path: &str) -> bool {
    method == "GET" && path == "/files"
}

fn e2b_control(method: &str, path: &str, _base: &str) -> Option<Value> {
    match (method, path) {
        ("POST", "/v2/sandboxes") => Some(json!({
            "templateID": "code-interpreter-v1", "sandboxID": "sbx-1", "clientID": "cl-1",
            "envdVersion": "0.2.0", "envdAccessToken": "envd-token"
        })),
        ("POST", "/filesystem.Filesystem/MakeDir") => Some(json!({})),
        _ => None,
    }
}

fn daytona_is_file(method: &str, path: &str) -> bool {
    method == "GET" && path == "/toolbox/sbx-daytona/files/download"
}

fn daytona_control(method: &str, path: &str, base: &str) -> Option<Value> {
    match (method, path) {
        ("POST", "/sandbox") | ("GET", "/sandbox/sbx-daytona") => Some(json!({
            "id": "sbx-daytona", "state": "started", "toolboxProxyUrl": format!("{base}/toolbox")
        })),
        ("POST", "/toolbox/sbx-daytona/process/session") => Some(json!({})),
        _ => None,
    }
}

fn vercel_is_file(method: &str, path: &str) -> bool {
    method == "POST" && path == "/v2/sandboxes/sessions/sess-1/fs/read"
}

fn vercel_control(method: &str, path: &str, _base: &str) -> Option<Value> {
    let sandbox = json!({
        "sandbox": {"name": "agents-test", "persistent": false, "createdAt": 1, "updatedAt": 1, "currentSessionId": "sess-1"},
        "session": {"id": "sess-1", "memory": 2048, "vcpus": 2, "region": "iad1", "timeout": 270000,
                    "status": "running", "requestedAt": 1, "createdAt": 1, "updatedAt": 1, "cwd": "/vercel/sandbox"},
        "routes": []
    });
    match (method, path) {
        ("POST", "/v3/sandboxes") => Some(sandbox),
        ("GET", "/v2/sandboxes/sessions/sess-1") => {
            Some(json!({"session": sandbox["session"].clone(), "routes": []}))
        }
        _ => None,
    }
}

fn cloudflare_is_file(method: &str, path: &str) -> bool {
    method == "GET" && path.starts_with("/v1/sandbox/sbx-1/file/")
}

fn cloudflare_control(method: &str, path: &str, _base: &str) -> Option<Value> {
    match (method, path) {
        ("POST", "/v1/sandbox") => Some(json!({ "id": "sbx-1" })),
        _ => None,
    }
}

fn blaxel_is_file(method: &str, path: &str) -> bool {
    method == "GET" && path.starts_with("/filesystem/")
}

fn blaxel_control(method: &str, path: &str, base: &str) -> Option<Value> {
    match (method, path) {
        ("POST", "/v0/sandboxes") => Some(json!({
            "metadata": { "name": "agents-test", "url": base }, "status": "DEPLOYED"
        })),
        // The workspace path resolver the session runs before each read.
        ("POST", "/process") => {
            Some(json!({ "exitCode": 0, "stdout": "/workspace/big.log\n", "stderr": "" }))
        }
        _ => None,
    }
}

/// The provider names a test exercises.
#[derive(Clone, Copy, Debug)]
enum Provider {
    E2b,
    Daytona,
    Vercel,
    Cloudflare,
    Blaxel,
}

impl Provider {
    fn is_file(self) -> IsFile {
        match self {
            Provider::E2b => e2b_is_file,
            Provider::Daytona => daytona_is_file,
            Provider::Vercel => vercel_is_file,
            Provider::Cloudflare => cloudflare_is_file,
            Provider::Blaxel => blaxel_is_file,
        }
    }

    fn control(self) -> Control {
        match self {
            Provider::E2b => e2b_control,
            Provider::Daytona => daytona_control,
            Provider::Vercel => vercel_control,
            Provider::Cloudflare => cloudflare_control,
            Provider::Blaxel => blaxel_control,
        }
    }
}

async fn start(provider: Provider, reply: Reply) -> Mock {
    serve(reply, provider.is_file(), provider.control()).await
}

/// A session of `provider` against `base`, with an optional short Daytona file budget.
async fn session_at(
    provider: Provider,
    base: &str,
    daytona_file_download: Option<Duration>,
) -> Arc<dyn SandboxSession> {
    match provider {
        Provider::E2b => E2BSandboxClient::with_options(
            E2BSandboxClientOptions::new()
                .with_api_key("e2b_test")
                .with_base_url(base)
                .with_sandbox_type(E2BSandboxType::CodeInterpreter)
                .with_timeout_secs(120),
        )
        .create(None)
        .await
        .expect("create"),
        Provider::Daytona => {
            let mut options = DaytonaSandboxClientOptions::new()
                .with_api_key("dtn_test")
                .with_api_url(base)
                .with_state_poll_interval(Duration::from_millis(1));
            if let Some(file_download) = daytona_file_download {
                options = options.with_timeouts(DaytonaSandboxTimeouts {
                    file_download,
                    ..DaytonaSandboxTimeouts::default()
                });
            }
            DaytonaSandboxClient::with_options(options)
                .create(None)
                .await
                .expect("create")
        }
        Provider::Vercel => VercelSandboxClient::with_options(
            VercelSandboxClientOptions::new()
                .with_token("vercel_test")
                .with_base_url(base)
                .with_project_id("prj_1")
                .with_team_id("team_1")
                .with_status_poll_interval(Duration::from_millis(1)),
        )
        .create(None)
        .await
        .expect("create"),
        Provider::Cloudflare => {
            CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(base))
                .create(None)
                .await
                .expect("create")
        }
        Provider::Blaxel => BlaxelSandboxClient::new(
            BlaxelSandboxClientOptions::new()
                .with_name("agents-test")
                .with_api_key("bl-token")
                .with_workspace("acme")
                .with_region("us-pdx-1")
                .with_api_base_url(format!("{base}/v0"))
                .with_timeouts(BlaxelTimeouts {
                    exec: Duration::from_secs(5),
                    ..BlaxelTimeouts::default()
                }),
        )
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create"),
    }
}

async fn make_session(provider: Provider, mock: &Mock) -> Arc<dyn SandboxSession> {
    session_at(provider, &mock.base, None).await
}

/// A read with a guard: a read that never returns is a failure, not a hang.
async fn read(session: &dyn SandboxSession, max: usize) -> SandboxResult<Vec<u8>> {
    tokio::time::timeout(
        Duration::from_secs(10),
        session.read_bounded("out.jsonl", max),
    )
    .await
    .expect("the bounded read returned")
}

/// The error must name no provider payload, in its message or its context.
fn assert_no_payload(error: &SandboxError) {
    assert!(!error.message.contains(PRIVATE), "{}", error.message);
    assert!(
        error.context.values().all(|value| !value.contains(PRIVATE)),
        "{:?}",
        error.context
    );
}

/// The SSE body the Cloudflare worker answers a binary file with.
fn cloudflare_sse(payload: &[u8]) -> Vec<u8> {
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(payload);
    format!(
        "data: {{\"type\":\"metadata\",\"isBinary\":true}}\n\n\
         data: {{\"type\":\"chunk\",\"data\":\"{encoded}\"}}\n\n\
         data: {{\"type\":\"complete\"}}\n\n"
    )
    .into_bytes()
}

#[tokio::test]
async fn every_provider_stops_at_the_prefix_and_closes_the_stream() {
    for provider in [
        Provider::E2b,
        Provider::Vercel,
        Provider::Daytona,
        Provider::Blaxel,
        Provider::Cloudflare,
    ] {
        for limit in [1usize, 5, 100] {
            // A raw endless body: the Cloudflare worker answers a file as raw bytes unless it
            // is binary, and a raw body is bounded by the limit itself.
            let reply = Reply::Endless {
                status: 200,
                prefix: Vec::new(),
            };
            let mock = start(provider, reply).await;
            let session = make_session(provider, &mock).await;

            let bytes = read(session.as_ref(), limit)
                .await
                .unwrap_or_else(|e| panic!("{provider:?} limit {limit}: {e:?}"));
            assert_eq!(bytes, vec![b'x'; limit], "{provider:?} limit {limit}");
            assert!(
                wait_for(&mock.flags.hung_up).await,
                "{provider:?} limit {limit}: the stream was closed"
            );
        }
    }
}

/// `test_bounded_http_read_missing_file_closes_without_body` for the providers the existing
/// ports do not exercise (Daytona's and Vercel's rows are in `py_sandbox_b_bounded_read.rs`): a
/// 404 is a missing file, and nothing of the response body reaches the error.
#[tokio::test]
async fn a_missing_file_is_not_found_without_the_response_body() {
    for provider in [Provider::Daytona, Provider::Blaxel, Provider::Cloudflare] {
        let mock = start(
            provider,
            Reply::Full {
                status: 404,
                body: PRIVATE.as_bytes().to_vec(),
            },
        )
        .await;
        let session = make_session(provider, &mock).await;

        let error = read(session.as_ref(), 5).await.unwrap_err();

        assert_eq!(
            error.code,
            ErrorCode::WorkspaceReadNotFound,
            "{provider:?}: {error:?}"
        );
        assert_no_payload(&error);
    }
}
