//! Port of `tests/sandbox/test_bounded_read.py` for the HTTP providers: a bounded read returns
//! the requested prefix and closes the response instead of draining it, and a failed stream
//! reports an error that never carries what was received.
//!
//! The Python tests mock each vendor SDK's streaming object and assert it was closed. The Rust
//! providers speak HTTP directly, so the same guarantees are observed on the wire: a local
//! server streams a file in small chunks and records whether the client hung up before it was
//! done. The vendor-SDK specifics (E2B's `files.read` stream, Modal descriptors, Vercel's mount
//! context around `iter_file`, Daytona's `download_file_without_preload_content`) have no Rust
//! counterpart.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxType,
};
use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::http::read_body_prefix;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxClient, SandboxSession};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How the file endpoint answers.
#[derive(Clone)]
enum File {
    /// A well-behaved stream of 3-byte chunks.
    Stream(Vec<u8>),
    /// Three bytes, then a connection cut short of the promised length.
    Broken,
    /// A status the provider has to translate.
    Status(u16),
}

struct Server {
    base: String,
    completed: Arc<AtomicBool>,
    write_failed: Arc<AtomicBool>,
}

/// A server whose `is_file` requests get `file` and whose other requests get `control(path)`.
async fn serve(
    file: File,
    is_file: fn(&str, &str) -> bool,
    control: fn(&str, &str, &str) -> Option<Value>,
) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let completed = Arc::new(AtomicBool::new(false));
    let write_failed = Arc::new(AtomicBool::new(false));
    let (done, failed) = (completed.clone(), write_failed.clone());
    let base_for_handler = base.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let (file, done, failed, base) = (
                file.clone(),
                done.clone(),
                failed.clone(),
                base_for_handler.clone(),
            );
            tokio::spawn(async move {
                let mut buffer = vec![0u8; 8192];
                let mut read = 0;
                loop {
                    let n = stream.read(&mut buffer[read..]).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    read += n;
                    if buffer[..read].windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, target) = (
                    first.next().unwrap_or("").to_string(),
                    first.next().unwrap_or("").to_string(),
                );
                let path = target.split('?').next().unwrap_or("").to_string();
                if !is_file(&method, &path) {
                    // Every read first runs the path resolver inside the sandbox; answer it in
                    // the provider's own wire format.
                    let head_end = buffer[..read]
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|at| at + 4)
                        .unwrap_or(read);
                    let content_length = head
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    let mut body = buffer[head_end..read].to_vec();
                    while body.len() < content_length {
                        let mut more = vec![0u8; content_length - body.len()];
                        match stream.read(&mut more).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => body.extend_from_slice(&more[..n]),
                        }
                    }
                    if String::from_utf8_lossy(&body).contains("resolve-workspace-path") {
                        let kind = if path.ends_with("/process.Process/Start") {
                            Some(support::Resolver::E2b)
                        } else if path.contains("/process/session/") && path.ends_with("/exec") {
                            Some(support::Resolver::Daytona)
                        } else {
                            None
                        };
                        if let Some(kind) = kind {
                            let _ =
                                support::write_response(&mut stream, support::resolver_reply(kind))
                                    .await;
                            return;
                        }
                    }
                    let body = control(&method, &path, &base)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| json!({"message": "unexpected"}).to_string());
                    let status = if control(&method, &path, &base).is_some() {
                        200
                    } else {
                        404
                    };
                    let response = format!(
                        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    return;
                }
                match file {
                    File::Status(status) => {
                        let _ = stream
                            .write_all(
                                format!("HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                                    .as_bytes(),
                            )
                            .await;
                    }
                    File::Broken => {
                        let _ = stream
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nsyn")
                            .await;
                    }
                    File::Stream(data) => {
                        let _ = stream
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n")
                            .await;
                        for chunk in data.chunks(3) {
                            if stream.write_all(chunk).await.is_err()
                                || stream.flush().await.is_err()
                            {
                                failed.store(true, Ordering::SeqCst);
                                return;
                            }
                            tokio::time::sleep(Duration::from_millis(15)).await;
                        }
                        done.store(true, Ordering::SeqCst);
                    }
                }
            });
        }
    });
    Server {
        base,
        completed,
        write_failed,
    }
}

fn data() -> Vec<u8> {
    (0..90u8).collect()
}

async fn assert_closed_early(server: &Server) {
    // Give the server long enough to have finished if the client had kept reading.
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        !server.completed.load(Ordering::SeqCst),
        "the whole body was drained"
    );
    assert!(
        server.write_failed.load(Ordering::SeqCst),
        "the server saw the hang-up"
    );
}

// ----------------------------------------------------------------------------------------------
// The shared reader
// ----------------------------------------------------------------------------------------------

/// Python: `test_bounded_provider_read_closes_at_prefix` (the limit axis: 1, 5 and 100).
#[tokio::test]
async fn the_reader_stops_at_the_prefix_and_hangs_up() {
    for limit in [1usize, 5, 100] {
        let server = serve(File::Stream(data()), |_, _| true, |_, _, _| None).await;
        let response = reqwest::get(format!("{}/file", server.base)).await.unwrap();

        let prefix = read_body_prefix(response, Some(limit), "read a file")
            .await
            .unwrap();

        assert_eq!(
            prefix,
            data()[..limit.min(data().len())].to_vec(),
            "limit {limit}"
        );
        if limit < data().len() {
            assert_closed_early(&server).await;
        }
    }
}

/// Python: `test_bounded_provider_read_closes_and_discards_failed_response`: a stream that dies
/// reports a transport error that does not contain what arrived before it died.
#[tokio::test]
async fn a_failed_stream_is_an_error_without_the_received_bytes() {
    let server = serve(File::Broken, |_, _| true, |_, _, _| None).await;
    let response = reqwest::get(format!("{}/file", server.base)).await.unwrap();

    let error = read_body_prefix(response, Some(50), "read a file")
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(error.retryable, Some(true));
    assert!(!error.message.contains("syn"), "{}", error.message);
}

// ----------------------------------------------------------------------------------------------
// Providers
// ----------------------------------------------------------------------------------------------

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

async fn e2b_session(server: &Server) -> Arc<dyn SandboxSession> {
    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_base_url(&server.base)
            .with_sandbox_type(E2BSandboxType::CodeInterpreter)
            .with_timeout_secs(120),
    );
    client.create(None).await.expect("create")
}

/// Python: `test_bounded_provider_read_closes_at_prefix[e2b]`.
#[tokio::test]
async fn e2b_reads_a_prefix_and_closes() {
    for limit in [1usize, 5, 100] {
        let server = serve(
            File::Stream(data()),
            |method, path| method == "GET" && path == "/files",
            e2b_control,
        )
        .await;
        let session = e2b_session(&server).await;

        let prefix = session.read_bounded("out.jsonl", limit).await.unwrap();

        assert_eq!(
            prefix,
            data()[..limit.min(data().len())].to_vec(),
            "limit {limit}"
        );
        if limit < data().len() {
            assert_closed_early(&server).await;
        }
    }
}

/// Python: `test_bounded_http_read_missing_file_closes_without_body` and
/// `test_daytona_bounded_read_classifies_http_status`: a missing file is not-found, and an
/// error status is an error rather than data.
#[tokio::test]
async fn e2b_classifies_missing_and_failing_reads() {
    let missing = serve(
        File::Status(404),
        |method, path| method == "GET" && path == "/files",
        e2b_control,
    )
    .await;
    let session = e2b_session(&missing).await;
    let error = session.read_bounded("out.jsonl", 10).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);

    let failing = serve(
        File::Status(500),
        |method, path| method == "GET" && path == "/files",
        e2b_control,
    )
    .await;
    let session = e2b_session(&failing).await;
    let error = session.read_bounded("out.jsonl", 10).await.unwrap_err();
    assert_ne!(error.code, ErrorCode::WorkspaceReadNotFound);

    let error = session.read_bounded("out.jsonl", 0).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
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

/// Python: `test_bounded_provider_read_closes_at_prefix[daytona]`.
#[tokio::test]
async fn daytona_reads_a_prefix_and_closes() {
    let server = serve(
        File::Stream(data()),
        |method, path| method == "GET" && path == "/toolbox/sbx-daytona/files/download",
        daytona_control,
    )
    .await;
    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base)
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    let prefix = session.read_bounded("out.jsonl", 5).await.unwrap();

    assert_eq!(prefix, data()[..5].to_vec());
    assert_closed_early(&server).await;
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

/// Python: `test_bounded_provider_read_closes_at_prefix[vercel]`.
#[tokio::test]
async fn vercel_reads_a_prefix_and_closes() {
    let server = serve(
        File::Stream(data()),
        |method, path| method == "POST" && path == "/v2/sandboxes/sessions/sess-1/fs/read",
        vercel_control,
    )
    .await;
    let client = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base)
            .with_project_id("prj_1")
            .with_team_id("team_1")
            .with_status_poll_interval(Duration::from_millis(1)),
    );
    let session = client.create(None).await.expect("create");

    let prefix = session.read_bounded("out.jsonl", 5).await.unwrap();

    assert_eq!(prefix, data()[..5].to_vec());
    assert_closed_early(&server).await;
}

// ----------------------------------------------------------------------------------------------
// Retry policy
// ----------------------------------------------------------------------------------------------

async fn daytona_session(server: &Server) -> Arc<dyn SandboxSession> {
    DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base)
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create")
}

async fn vercel_session(server: &Server) -> Arc<dyn SandboxSession> {
    VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base)
            .with_project_id("prj_1")
            .with_team_id("team_1")
            .with_status_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create")
}

/// Python: `test_bounded_provider_read_preserves_retry_policy`: a transient status is
/// retryable, an authorization failure is not, and the error never carries a response body.
#[tokio::test]
async fn bounded_read_preserves_http_retry_policy() {
    for (status, retryable) in [(503u16, true), (429, true), (403, false)] {
        let server = serve(
            File::Status(status),
            |method, path| method == "GET" && path == "/files",
            e2b_control,
        )
        .await;
        let error = e2b_session(&server)
            .await
            .read_bounded("out.jsonl", 5)
            .await
            .unwrap_err();
        assert_eq!(error.retryable, Some(retryable), "e2b {status}");

        let server = serve(
            File::Status(status),
            |method, path| method == "GET" && path == "/toolbox/sbx-daytona/files/download",
            daytona_control,
        )
        .await;
        let error = daytona_session(&server)
            .await
            .read_bounded("out.jsonl", 5)
            .await
            .unwrap_err();
        assert_eq!(error.retryable, Some(retryable), "daytona {status}");

        let server = serve(
            File::Status(status),
            |method, path| method == "POST" && path == "/v2/sandboxes/sessions/sess-1/fs/read",
            vercel_control,
        )
        .await;
        let error = vercel_session(&server)
            .await
            .read_bounded("out.jsonl", 5)
            .await
            .unwrap_err();
        assert_eq!(error.retryable, Some(retryable), "vercel {status}");
    }
}

/// Python: `test_bounded_http_read_missing_file_closes_without_body[daytona]` (and vercel).
#[tokio::test]
async fn bounded_read_of_a_missing_file_is_not_found() {
    let server = serve(
        File::Status(404),
        |method, path| method == "GET" && path == "/toolbox/sbx-daytona/files/download",
        daytona_control,
    )
    .await;
    let error = daytona_session(&server)
        .await
        .read_bounded("out.jsonl", 5)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);

    let server = serve(
        File::Status(404),
        |method, path| method == "POST" && path == "/v2/sandboxes/sessions/sess-1/fs/read",
        vercel_control,
    )
    .await;
    let error = vercel_session(&server)
        .await
        .read_bounded("out.jsonl", 5)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
}

/// Python: `test_bounded_provider_read_closes_and_discards_failed_response` for the providers
/// whose failure is a cut stream: the error names no received bytes.
#[tokio::test]
async fn bounded_read_failure_never_carries_the_response() {
    let server = serve(
        File::Broken,
        |method, path| method == "GET" && path == "/toolbox/sbx-daytona/files/download",
        daytona_control,
    )
    .await;
    let error = daytona_session(&server)
        .await
        .read_bounded("out.jsonl", 50)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert!(!format!("{error:?}").contains("syn"));
}
