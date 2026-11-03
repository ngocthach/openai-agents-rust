//! Ports of the HTTP-observable behaviour in `tests/extensions/sandbox/test_cloudflare.py`.
//!
//! Python injects a fake `aiohttp` session and asserts on the calls it recorded; the Rust client
//! talks to a Cloudflare Worker over HTTP, so the same behaviour is asserted against a local mock
//! worker that records every request.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareSandboxClient, CloudflareSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSession};
use serde_json::json;
use std::sync::Arc;
use support::{MockResponse, MockServer, Recorded};

/// An SSE reply with the given raw body parts.
fn sse(body: &str) -> MockResponse {
    MockResponse::text(body.to_string()).with_content_type("text/event-stream")
}

/// A successful exec: stdout, stderr and an exit event.
fn exec_ok(stdout: &str, stderr: &str, exit_code: i64) -> MockResponse {
    let mut body = String::new();
    if !stdout.is_empty() {
        body.push_str(&format!(
            "event: stdout\ndata: {}\n\n",
            support::b64(stdout)
        ));
    }
    if !stderr.is_empty() {
        body.push_str(&format!(
            "event: stderr\ndata: {}\n\n",
            support::b64(stderr)
        ));
    }
    body.push_str(&format!(
        "event: exit\ndata: {}\n\n",
        json!({ "exit_code": exit_code })
    ));
    sse(&body)
}

fn worker(
    handler: impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static,
) -> impl std::future::Future<Output = MockServer> {
    support::start_resolving(support::Resolver::Cloudflare, move |request: &Recorded| {
        if request.method == "POST" && request.path == "/v1/sandbox" {
            return MockResponse::json(json!({ "id": "sbx-1" }));
        }
        handler(request)
    })
}

async fn session(server: &MockServer) -> Arc<dyn SandboxSession> {
    CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
        .create(None)
        .await
        .expect("create")
}

/// `test_cloudflare_create_calls_post_sandbox_for_id`: the id comes from `POST /v1/sandbox`.
#[tokio::test]
async fn cloudflare_create_calls_post_sandbox_for_id() {
    let server = support::start_resolving(support::Resolver::Cloudflare, |_| {
        MockResponse::json(json!({ "id": "server2generated3id" }))
    })
    .await;

    let session = session(&server).await;

    assert_eq!(server.count("POST", "/v1/sandbox"), 1);
    assert_eq!(
        session.session_state().provider_id.as_deref(),
        Some("server2generated3id")
    );
    server.stop();
}

/// `test_cloudflare_create_uses_injected_api_key_for_auth_header`: the configured key wins over
/// the environment and rides every request as a bearer token.
#[tokio::test]
async fn cloudflare_create_uses_the_injected_api_key_for_the_auth_header() {
    let server = worker(|_| exec_ok("", "", 0)).await;
    let client = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url).with_api_key("injected-token"),
    );

    let session = client.create(None).await.expect("create");
    session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("exec");

    for request in server.requests() {
        assert_eq!(
            request.header("authorization"),
            Some("Bearer injected-token"),
            "{}",
            request.route()
        );
    }
    server.stop();
}

/// `test_cloudflare_create_rejects_non_workspace_root`
#[tokio::test]
async fn cloudflare_create_rejects_a_non_workspace_root() {
    let server = worker(|_| MockResponse::empty(204)).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));

    let error = client
        .create(Some(Manifest::new().with_root("/tmp/app")))
        .await
        .expect_err("persistence is fixed to /workspace");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.message.contains("/tmp/app"), "{}", error.message);
    assert!(server.requests().is_empty(), "no request before validation");
    server.stop();
}

/// `test_cloudflare_create_raises_on_post_sandbox_failure`
#[tokio::test]
async fn cloudflare_create_raises_on_post_sandbox_failure() {
    let server = support::start_resolving(support::Resolver::Cloudflare, |_| {
        MockResponse::error(500, "boom")
    })
    .await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));

    let error = client.create(None).await.expect_err("create failed");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.message.contains("POST /sandbox"), "{}", error.message);
    server.stop();
}

/// A reply without a usable id is refused.
#[tokio::test]
async fn cloudflare_create_rejects_a_reply_without_an_id() {
    let server = support::start_resolving(support::Resolver::Cloudflare, |_| {
        MockResponse::json(json!({ "id": "" }))
    })
    .await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));

    let error = client.create(None).await.expect_err("no id");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    server.stop();
}

/// `test_cloudflare_exec_decodes_sse_output`
#[tokio::test]
async fn cloudflare_exec_decodes_sse_output() {
    let server = worker(|_| exec_ok("hello\n", "warn", 0)).await;
    let session = session(&server).await;

    let result = session
        .exec(ExecRequest::argv(["echo", "hello"]).with_timeout(Some(Duration::from_secs(5))))
        .await
        .expect("exec");

    assert_eq!(result.stdout, b"hello\n");
    assert_eq!(result.stderr, b"warn");
    assert_eq!(result.exit_code, 0);
    server.stop();
}

/// `test_cloudflare_exec_handles_utf8_split_across_sse_chunks`: an `error` event whose multibyte
/// payload straddles two writes is decoded whole.
#[tokio::test]
async fn cloudflare_exec_handles_utf8_split_across_sse_chunks() {
    let body = "event: error\ndata: {\"error\":\"失败\"}\n\n"
        .as_bytes()
        .to_vec();
    let at = body
        .windows("失败".len())
        .position(|w| w == "失败".as_bytes())
        .expect("payload")
        + 1;
    let (head, tail) = (body[..at].to_vec(), body[at..].to_vec());
    let server = worker(move |_| {
        MockResponse::streamed(
            "text/event-stream",
            vec![(head.clone(), 0), (tail.clone(), 20)],
        )
    })
    .await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["echo", "hello"]))
        .await
        .expect_err("the worker reported an error");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("失败"), "{}", error.message);
    server.stop();
}

/// `test_cloudflare_exec_applies_manifest_environment`: the worker's exec body is argv only, so
/// the manifest environment becomes an `env` prefix.
#[tokio::test]
async fn cloudflare_exec_applies_manifest_environment() {
    let server = worker(|_| exec_ok("hello", "", 0)).await;
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .create(Some(
                Manifest::new().with_env("A", "1").with_env("B", "two"),
            ))
            .await
            .expect("create");

    let result = session
        .exec(ExecRequest::argv(["printenv", "A"]))
        .await
        .expect("exec");

    assert_eq!(result.exit_code, 0);
    let exec = server.find("POST", "/v1/sandbox/sbx-1/exec");
    assert_eq!(
        exec.json()["argv"],
        json!(["env", "A=1", "B=two", "printenv", "A"])
    );
    server.stop();
}

/// `test_cloudflare_exec_timeout_raises_exec_timeout_error`: the timeout also reaches the worker
/// as `timeout_ms`.
#[tokio::test]
async fn cloudflare_exec_timeout_raises_exec_timeout_error() {
    let server =
        worker(|_| MockResponse::streamed("text/event-stream", vec![(Vec::new(), 30_000)])).await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["sleep", "999"]).with_timeout(Some(Duration::from_millis(50))))
        .await
        .expect_err("timed out");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert_eq!(
        server.find("POST", "/v1/sandbox/sbx-1/exec").json()["timeout_ms"],
        50
    );
    server.stop();
}

/// `test_cloudflare_exec_non_200_includes_provider_error_details`: a worker error keeps its HTTP
/// status and its `code: error` text.
#[tokio::test]
async fn cloudflare_exec_non_200_includes_provider_error_details() {
    let server = worker(|_| {
        MockResponse::status(
            502,
            json!({ "error": "pool error: Failed to start container", "code": "pool_error" }),
        )
    })
    .await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["mkdir", "-p", "--", "/workspace"]))
        .await
        .expect_err("worker error");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("HTTP 502"), "{}", error.message);
    assert!(
        error
            .message
            .contains("pool_error: pool error: Failed to start container"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_cloudflare_exec_retryability_follows_documented_status_semantics` and
/// `test_cloudflare_retryability_status_table`: 400 and 500 are not retryable, 503 is.
#[tokio::test]
async fn cloudflare_exec_retryability_follows_documented_status_semantics() {
    for (status, expected) in [(400u16, Some(false)), (500, Some(false)), (503, Some(true))] {
        let server = worker(move |_| {
            MockResponse::status(
                status,
                json!({ "error": "cloudflare sandbox error", "code": "cloudflare_error" }),
            )
        })
        .await;
        let session = session(&server).await;

        let error = session
            .exec(ExecRequest::argv(["mkdir", "-p", "--", "/workspace"]))
            .await
            .expect_err("worker error");

        assert_eq!(error.retryable, expected, "status {status}: {error:?}");
        server.stop();
    }
}

/// `test_cloudflare_exec_non_sse_json_body_includes_provider_error_details`: a 200 reply whose
/// body is a JSON error rather than an event stream is still reported with the provider's text.
#[tokio::test]
async fn cloudflare_exec_non_sse_json_body_includes_provider_error_details() {
    let server = worker(|_| {
        MockResponse::json(
            json!({ "error": "pool error: Failed to start container", "code": "pool_error" }),
        )
    })
    .await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["mkdir", "-p", "--", "/workspace"]))
        .await
        .expect_err("not an event stream");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(
        error
            .message
            .contains("pool_error: pool error: Failed to start container"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_cloudflare_exec_stream_without_exit_raises_transport_error`
#[tokio::test]
async fn cloudflare_exec_stream_without_exit_raises_transport_error() {
    let server = worker(|_| sse("event: stdout\ndata: aGVsbG8=\n\n")).await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["echo", "hello"]))
        .await
        .expect_err("no exit event");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    server.stop();
}

/// `test_cloudflare_exec_client_error_includes_provider_context`: a connection that is refused
/// is a transport error, not a timeout.
#[tokio::test]
async fn cloudflare_exec_client_error_is_a_transport_error() {
    let server = worker(|_| exec_ok("", "", 0)).await;
    let session = session(&server).await;
    // Drop the worker so the next request cannot connect.
    server.stop();

    let error = session
        .exec(ExecRequest::argv(["echo", "hello"]).with_timeout(Some(Duration::from_secs(1))))
        .await
        .expect_err("worker is gone");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
}

/// `test_cloudflare_prepare_workspace_preserves_exec_error_context`: a failing worker during
/// `start` surfaces as a transport error naming the status.
#[tokio::test]
async fn cloudflare_start_surfaces_a_worker_error() {
    let server = worker(|_| {
        MockResponse::status(
            502,
            json!({ "error": "pool error: Failed to start container", "code": "pool_error" }),
        )
    })
    .await;
    let session = session(&server).await;

    let error = session.start().await.expect_err("the worker refused");

    assert!(error.message.contains("HTTP 502"), "{}", error.message);
    server.stop();
}

/// `test_cloudflare_read_and_write_use_file_endpoints` and
/// `test_cloudflare_read_and_write_normalize_workspace_paths`.
#[tokio::test]
async fn cloudflare_read_and_write_use_file_endpoints() {
    let server = worker(|request| {
        if request.method == "GET" {
            return MockResponse::bytes(b"file data".to_vec());
        }
        MockResponse::empty(204)
    })
    .await;
    let session = session(&server).await;

    session
        .write_file("dir/a b.txt", b"payload")
        .await
        .expect("write");
    let read = session
        .read_file("/workspace/dir/a b.txt")
        .await
        .expect("read");

    assert_eq!(read, b"file data");
    let put = server.find("PUT", "/v1/sandbox/sbx-1/file/workspace/dir/a%20b.txt");
    assert_eq!(put.text(), "payload");
    assert_eq!(put.header("content-type"), Some("application/octet-stream"));
    server.find("GET", "/v1/sandbox/sbx-1/file/workspace/dir/a%20b.txt");
    server.stop();
}

/// `test_cloudflare_read_404_and_write_non_bytes_raise_structured_errors`
#[tokio::test]
async fn cloudflare_read_404_is_a_structured_not_found() {
    let server = worker(|_| MockResponse::error(404, "no such file")).await;
    let session = session(&server).await;

    let error = session.read_file("missing.txt").await.expect_err("missing");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

/// A path outside the workspace root is refused before any request is made.
#[tokio::test]
async fn cloudflare_read_rejects_a_path_outside_the_workspace() {
    let server = worker(|_| MockResponse::bytes(Vec::new())).await;
    let session = session(&server).await;
    let before = server.requests().len();

    let error = session
        .read_file("/etc/passwd")
        .await
        .expect_err("outside the workspace");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(server.requests().len(), before);
    server.stop();
}

/// `test_cloudflare_persist_and_hydrate_use_http_endpoints`: the archive travels as raw bytes,
/// and the manifest's ephemeral paths are excluded from the persist query.
#[tokio::test]
async fn cloudflare_persist_and_hydrate_use_http_endpoints() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::bytes(b"tar-bytes".to_vec());
        }
        MockResponse::empty(204)
    })
    .await;
    let session = session(&server).await;

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, b"tar-bytes");
    let persist = server.find("POST", "/v1/sandbox/sbx-1/persist");
    assert!(persist.query.starts_with("excludes="), "{}", persist.query);
    server.stop();
}

/// `test_cloudflare_persist_retries_only_documented_503_status` and
/// `test_cloudflare_persist_does_not_retry_documented_fail_fast_500`.
#[tokio::test]
async fn cloudflare_persist_retries_a_503_but_not_a_500() {
    for (status, expected_calls) in [(503u16, 2usize), (500, 1)] {
        let calls = support::Counter::new();
        let server = worker(move |request| {
            if request.path.ends_with("/persist") {
                return match calls.next("persist") {
                    0 => MockResponse::error(status, "worker busy"),
                    _ => MockResponse::bytes(b"tar-bytes".to_vec()),
                };
            }
            MockResponse::empty(204)
        })
        .await;
        let session = session(&server).await;

        let outcome = session.persist_workspace().await;

        assert_eq!(
            server.count("POST", "/v1/sandbox/sbx-1/persist"),
            expected_calls,
            "status {status}"
        );
        assert_eq!(
            outcome.is_ok(),
            status == 503,
            "status {status}: {outcome:?}"
        );
        server.stop();
    }
}

/// `test_cloudflare_delete_calls_shutdown` and the shutdown idempotence contract.
#[tokio::test]
async fn cloudflare_delete_calls_shutdown_exactly_once() {
    let server = worker(|_| MockResponse::empty(204)).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let session = client.create(None).await.expect("create");

    client.delete(session.clone()).await.expect("delete");
    session.shutdown().await.expect("shutdown again");

    assert_eq!(server.count("DELETE", "/v1/sandbox/sbx-1"), 1);
    server.stop();
}

/// `test_cloudflare_supports_pty`
#[tokio::test]
async fn cloudflare_supports_pty() {
    let server = worker(|_| MockResponse::empty(204)).await;
    let session = session(&server).await;

    assert!(session.supports_pty());
    server.stop();
}

/// `test_cloudflare_resume_*`: a worker that no longer serves the recorded id gets a fresh
/// sandbox, and one that still does is reused.
#[tokio::test]
async fn cloudflare_resume_reuses_a_running_sandbox_and_replaces_a_missing_one() {
    for (running, expected_creates) in [(true, 0usize), (false, 1)] {
        let server =
            support::start_resolving(support::Resolver::Cloudflare, move |request: &Recorded| {
                if request.method == "POST" && request.path == "/v1/sandbox" {
                    return MockResponse::json(json!({ "id": "sbx-new" }));
                }
                if request.path.ends_with("/running") {
                    return MockResponse::json(json!({ "running": running }));
                }
                MockResponse::empty(204)
            })
            .await;
        let client =
            CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
        let state = openai_agents_sdk::sandbox::SandboxSessionState::new(
            "cloudflare",
            "session-1",
            "/workspace",
            &Manifest::new(),
        )
        .with_provider_id(Some("sbx-old".to_string()));

        let session = client
            .resume(&state, Manifest::new())
            .await
            .expect("resume");

        assert_eq!(server.count("POST", "/v1/sandbox"), expected_creates);
        let id = session.session_state().provider_id.clone();
        assert_eq!(
            id.as_deref(),
            Some(if running { "sbx-old" } else { "sbx-new" })
        );
        server.stop();
    }
}

/// A state from another backend is refused.
#[tokio::test]
async fn cloudflare_resume_rejects_a_foreign_backend_state() {
    let server = worker(|_| MockResponse::empty(204)).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let state = openai_agents_sdk::sandbox::SandboxSessionState::new(
        "e2b",
        "session-1",
        "/workspace",
        &Manifest::new(),
    )
    .with_provider_id(Some("sbx".to_string()));

    let error = client
        .resume(&state, Manifest::new())
        .await
        .expect_err("foreign state");

    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    server.stop();
}
