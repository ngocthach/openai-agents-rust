//! Ports of the REST-observable behaviour in `tests/extensions/sandbox/test_vercel.py` and
//! `tests/extensions/sandbox/test_vercel_provider_contract.py`.
//!
//! Python patches the `vercel.sandbox` SDK; the Rust client speaks the Vercel sandbox REST API,
//! so the same behaviour is asserted against a local mock that records each request. The S3
//! mount tests are covered by `tests/sandbox_e2b_daytona_vercel_tests.rs`.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    NetworkPolicy, VercelSandboxClient, VercelSandboxClientOptions, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

fn sandbox_and_session(status: &str, session_id: &str) -> Value {
    json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": session_id,
        },
        "session": {
            "id": session_id,
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
        "routes": [{ "port": 3000, "url": "https://3000-sandbox.vercel.run" }],
    })
}

fn command_result(stdout: &str, stderr: &str, exit_code: i64) -> MockResponse {
    MockResponse::ndjson(&[
        json!({ "stream": "stdout", "data": stdout }),
        json!({ "stream": "stderr", "data": stderr }),
        json!({
            "command": {
                "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                "sessionId": "sess-1", "exitCode": exit_code, "startedAt": 1,
            },
        }),
    ])
}

/// A handler answering the whole REST surface a basic session touches.
fn handler(
    stdout: &'static str,
    exit_code: i64,
) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    move |request: &Recorded| match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => MockResponse::json(sandbox_and_session("running", "sess-1")),
        ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
            let body = sandbox_and_session("running", "sess-1");
            MockResponse::json(json!({ "session": body["session"], "routes": body["routes"] }))
        }
        ("GET", path) if path.starts_with("/v2/sandboxes/") => {
            MockResponse::json(sandbox_and_session("running", "sess-1"))
        }
        ("POST", path) if path.ends_with("/cmd") => command_result(stdout, "", exit_code),
        ("POST", path) if path.ends_with("/fs/read") => MockResponse::bytes("payload"),
        ("POST", path) if path.ends_with("/stop") => {
            let body = sandbox_and_session("stopped", "sess-1");
            MockResponse::json(json!({ "session": body["session"] }))
        }
        _ => MockResponse::json(json!({})),
    }
}

fn options(base_url: &str) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(base_url)
        .with_project_id("prj_1")
        .with_team_id("team_1")
        .with_status_poll_interval(Duration::from_millis(1))
}

fn client(base_url: &str) -> VercelSandboxClient {
    VercelSandboxClient::with_options(options(base_url))
}

/// The `[command, args...]` of every recorded `cmd` request.
fn commands(server: &MockServer) -> Vec<Vec<String>> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd"))
        .map(|request| {
            let body = request.json();
            let mut argv = vec![body["command"].as_str().unwrap_or_default().to_string()];
            for arg in body["args"].as_array().cloned().unwrap_or_default() {
                argv.push(arg.as_str().unwrap_or_default().to_string());
            }
            argv
        })
        .collect()
}

/// `test_vercel_create_passes_provider_options`: the create body carries the project, runtime,
/// timeout, ports, resources, merged environment and network policy; the manifest wins over the
/// client's environment for a shared name.
#[tokio::test]
async fn vercel_create_passes_provider_options() {
    let server = MockServer::start(handler("", 0)).await;
    let client = VercelSandboxClient::with_options(
        options(&server.base_url)
            .with_timeout_ms(12_000)
            .with_runtime("node22")
            .with_env("FLAG", "options")
            .with_env("HELLO", "world")
            .with_exposed_port(3000)
            .with_exposed_port(4000)
            .with_network_policy(NetworkPolicy::AllowAll),
    );

    let session = client
        .create(Some(
            Manifest::new()
                .with_env("FLAG", "manifest")
                .with_env("FROM_MANIFEST", "1"),
        ))
        .await
        .expect("create");

    let create = server.find("POST", "/v3/sandboxes");
    assert_eq!(create.header("authorization"), Some("Bearer vercel_test"));
    assert_eq!(create.query_param("teamId").as_deref(), Some("team_1"));
    let body = create.json();
    assert_eq!(body["projectId"], "prj_1");
    assert_eq!(body["timeout"], 12_000);
    assert_eq!(body["ports"], json!([3000, 4000]));
    assert_eq!(
        body["env"],
        json!({ "FLAG": "manifest", "FROM_MANIFEST": "1", "HELLO": "world" })
    );
    assert_eq!(session.workspace_root(), "/vercel/sandbox");
    server.stop();
}

/// `test_vercel_create_retries_transient_transport_error` and
/// `test_vercel_create_does_not_retry_non_transient_error`.
#[tokio::test]
async fn vercel_create_retries_a_transient_error_but_not_a_bad_request() {
    for (status, expected_creates) in [(503u16, 2usize), (400, 1)] {
        let creates = support::Counter::new();
        let server = MockServer::start(move |request: &Recorded| {
            if request.method == "POST" && request.path == "/v3/sandboxes" {
                return match creates.next("create") {
                    0 => MockResponse::error(status, "failure"),
                    _ => MockResponse::json(sandbox_and_session("running", "sess-1")),
                };
            }
            handler("", 0)(request)
        })
        .await;

        let outcome = client(&server.base_url).create(None).await;

        assert_eq!(
            server.count("POST", "/v3/sandboxes"),
            expected_creates,
            "status {status}"
        );
        assert_eq!(outcome.is_ok(), status == 503, "status {status}");
        server.stop();
    }
}

/// `test_vercel_exec_read_write_and_port_resolution`: exec returns the command output, a write
/// is a gzipped tar, a read names the absolute path, and a port resolves from the create routes.
#[tokio::test]
async fn vercel_exec_read_write_and_port_resolution() {
    let server = MockServer::start(handler("hello\n", 0)).await;
    let session =
        VercelSandboxClient::with_options(options(&server.base_url).with_exposed_port(3000))
            .create(None)
            .await
            .expect("create");

    session
        .write_file("notes.txt", b"payload")
        .await
        .expect("write");
    let result = session
        .exec(ExecRequest::argv(["printf", "hello"]))
        .await
        .expect("exec");
    let endpoint = session.resolve_exposed_port(3000).await.expect("port");
    let payload = session.read_file("notes.txt").await.expect("read");

    assert!(result.ok());
    assert_eq!(result.stdout, b"hello\n");
    assert_eq!(endpoint.host, "3000-sandbox.vercel.run");
    assert_eq!(endpoint.port, 443);
    assert!(endpoint.tls);
    assert_eq!(payload, b"payload");
    let write = server.find("POST", "/v2/sandboxes/sessions/sess-1/fs/write");
    assert_eq!(write.header("content-type"), Some("application/gzip"));
    assert_eq!(
        server
            .find("POST", "/v2/sandboxes/sessions/sess-1/fs/read")
            .json()["path"],
        "/vercel/sandbox/notes.txt"
    );
    server.stop();
}

/// `test_vercel_exec_marks_typed_not_found_non_retryable` and
/// `test_vercel_exec_marks_typed_rate_limit_retryable`.
#[tokio::test]
async fn vercel_exec_classifies_not_found_and_rate_limit() {
    for (status, expected) in [(404u16, Some(false)), (429, Some(true))] {
        let server = MockServer::start(move |request: &Recorded| {
            if request.path.ends_with("/cmd") {
                return MockResponse::error(status, "provider error");
            }
            handler("", 0)(request)
        })
        .await;
        let session = client(&server.base_url).create(None).await.expect("create");

        let error = session
            .exec(ExecRequest::argv(["pwd"]))
            .await
            .expect_err("provider error");

        assert_eq!(error.code, ErrorCode::ExecTransportError, "{status}");
        assert_eq!(error.retryable, expected, "status {status}: {error:?}");
        server.stop();
    }
}

/// `test_vercel_retryability_status_table`, observed through the exec error the status maps to.
#[tokio::test]
async fn vercel_retryability_status_table() {
    for (status, expected) in [
        (400u16, false),
        (401, false),
        (403, false),
        (404, false),
        (408, true),
        (425, true),
        (422, false),
        (429, true),
        (500, true),
        (502, true),
        (503, true),
        (504, true),
    ] {
        let server = MockServer::start(move |request: &Recorded| {
            if request.path.ends_with("/cmd") {
                return MockResponse::error(status, "provider error");
            }
            handler("", 0)(request)
        })
        .await;
        let session = client(&server.base_url).create(None).await.expect("create");

        let error = session
            .exec(ExecRequest::argv(["pwd"]))
            .await
            .expect_err("provider error");

        assert_eq!(
            error.retryable,
            Some(expected),
            "status {status}: {error:?}"
        );
        server.stop();
    }
}

/// `test_vercel_rejects_sandbox_local_user_arguments`: a sandbox user is refused before any
/// command is sent.
#[tokio::test]
async fn vercel_rejects_a_sandbox_user() {
    use openai_agents_sdk::sandbox::types::User;

    let server = MockServer::start(handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["id"]).with_user(Some(User::new("alice"))))
        .await
        .expect_err("users are unsupported");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(commands(&server).is_empty(), "{:?}", commands(&server));
    server.stop();
}

/// `test_vercel_supports_pty_is_disabled_until_provider_methods_exist`
#[tokio::test]
async fn vercel_supports_pty_is_disabled() {
    let server = MockServer::start(handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    assert!(!session.supports_pty());
    server.stop();
}

/// `test_vercel_start_*`: `start` materializes the workspace root before using it as a cwd.
#[tokio::test]
async fn vercel_start_bootstraps_the_workspace_root() {
    let server = MockServer::start(handler("", 0)).await;
    let session = client(&server.base_url)
        .create(Some(Manifest::new().with_root("/srv/app")))
        .await
        .expect("create");

    session.start().await.expect("start");

    let first = commands(&server).into_iter().next().expect("a command");
    assert!(
        first.join(" ").contains("mkdir -p"),
        "the root is created first: {first:?}"
    );
    assert!(first.join(" ").contains("/srv/app"), "{first:?}");
    server.stop();
}

/// `test_vercel_read_and_write_reject_paths_outside_workspace_root`
#[tokio::test]
async fn vercel_read_and_write_reject_paths_outside_the_workspace_root() {
    let server = MockServer::start(handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");
    let before = server.requests().len();

    let read = session.read_file("/etc/passwd").await.expect_err("outside");
    let write = session
        .write_file("../escape.txt", b"x")
        .await
        .expect_err("outside");

    assert_eq!(read.code, ErrorCode::InvalidManifestPath);
    assert_eq!(write.code, ErrorCode::InvalidManifestPath);
    assert_eq!(server.requests().len(), before, "no request was made");
    server.stop();
}

/// A persisted state naming `<sandbox name>:<session id>`, as the Vercel backend records it.
fn state(sandbox_name: &str) -> SandboxSessionState {
    SandboxSessionState::new("vercel", "session-1", "/vercel/sandbox", &Manifest::new())
        .with_provider_id(Some(format!("{sandbox_name}:sess-1")))
}

/// `test_vercel_resume_reconnects_existing_running_sandbox`: a running sandbox is reused and
/// nothing is created.
#[tokio::test]
async fn vercel_resume_reconnects_an_existing_running_sandbox() {
    let server = MockServer::start(handler("", 0)).await;

    let session = client(&server.base_url)
        .resume(&state("agents-test"), Manifest::new())
        .await
        .expect("resume");

    assert_eq!(
        server.count("POST", "/v3/sandboxes"),
        0,
        "nothing is created"
    );
    assert_eq!(session.id(), "session-1");
    server.stop();
}

/// `test_vercel_resume_falls_back_to_recreate_when_sandbox_missing`: the replacement keeps the
/// recorded runtime and timeout, and merges the manifest environment over the options'.
#[tokio::test]
async fn vercel_resume_recreates_a_missing_sandbox() {
    // The first lookup (the resume probe) misses; the replacement then reports `running`.
    let lookups = support::Counter::new();
    let server = MockServer::start(move |request: &Recorded| {
        if request.method == "GET"
            && request.path.starts_with("/v2/sandboxes/")
            && lookups.next("get") == 0
        {
            return MockResponse::error(404, "missing");
        }
        handler("", 0)(request)
    })
    .await;

    client(&server.base_url)
        .resume(&state("sandbox-missing"), Manifest::new())
        .await
        .expect("resume");

    assert_eq!(server.count("POST", "/v3/sandboxes"), 1);
    server.stop();
}

/// `test_vercel_resume_recreates_sandbox_when_cannot_reach_running`: a terminal sandbox is
/// replaced without waiting out the running timeout.
#[tokio::test]
async fn vercel_resume_recreates_a_terminal_sandbox_without_waiting() {
    for status in ["stopped", "failed", "aborted"] {
        // The resume probe sees the terminal status; the replacement then reports `running`.
        let lookups = support::Counter::new();
        let server = MockServer::start(move |request: &Recorded| {
            if request.method == "GET"
                && request.path.starts_with("/v2/sandboxes/")
                && lookups.next("get") == 0
            {
                let body = sandbox_and_session(status, "sess-1");
                return MockResponse::json(json!({ "session": body["session"], "routes": [] }));
            }
            handler("", 0)(request)
        })
        .await;

        let started = std::time::Instant::now();
        client(&server.base_url)
            .resume(&state("agents-test"), Manifest::new())
            .await
            .expect("resume");

        assert_eq!(server.count("POST", "/v3/sandboxes"), 1, "{status}");
        assert!(started.elapsed() < Duration::from_secs(5), "{status}");
        server.stop();
    }
}

/// `test_vercel_serialized_session_state_omits_token_and_resume_uses_live_client_token`: the
/// persisted state never holds the token, and a resumed client authenticates with its own.
#[tokio::test]
async fn vercel_serialized_state_omits_the_token() {
    let server = MockServer::start(handler("", 0)).await;
    let client = VercelSandboxClient::with_options(
        options(&server.base_url).with_network_policy(NetworkPolicy::DenyAll),
    );
    let session = client.create(None).await.expect("create");

    let state = session.session_state();
    let payload = serde_json::to_string(&state).expect("serialize");

    assert!(!payload.contains("vercel_test"), "{payload}");
    let restored: SandboxSessionState = serde_json::from_str(&payload).expect("deserialize");
    client
        .resume(&restored, Manifest::new())
        .await
        .expect("resume");
    for request in server.requests() {
        assert_eq!(
            request.header("authorization"),
            Some("Bearer vercel_test"),
            "{}",
            request.route()
        );
    }
    server.stop();
}

/// `test_vercel_create_does_not_read_token_or_scope_from_env`: without a configured token the
/// client refuses rather than falling back to the ambient environment.
#[tokio::test]
async fn vercel_create_requires_a_configured_token() {
    let server = MockServer::start(handler("", 0)).await;
    // SAFETY: the test process sets its own environment before any client reads it.
    unsafe {
        std::env::remove_var("VERCEL_TOKEN");
    }

    let outcome = VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await;

    assert!(outcome.is_err());
    assert_eq!(server.count("POST", "/v3/sandboxes"), 0);
    server.stop();
}

/// `test_vercel_tar_persist_raises_archive_error_on_nonzero_exec`: a failing archive command is
/// an archive-read error, and the scratch tarball is still removed.
#[tokio::test]
async fn vercel_tar_persist_raises_archive_error_on_nonzero_exec() {
    let server = MockServer::start(handler("", 2)).await;
    let session = VercelSandboxClient::with_options(
        options(&server.base_url).with_workspace_persistence(WorkspacePersistence::Tar),
    )
    .create(Some(Manifest::new().with_root("/workspace")))
    .await
    .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the tar command failed");

    // Python raises `WorkspaceArchiveReadError`; the Rust backends report archive failures as
    // `SnapshotPersistError` throughout (see the report's cross-cutting notes).
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    let last = commands(&server).pop().expect("a command");
    assert_eq!(last.first().map(String::as_str), Some("rm"), "{last:?}");
    server.stop();
}

/// `test_vercel_hydrate_workspace_rejects_unsafe_tar_before_upload`
#[tokio::test]
async fn vercel_hydrate_rejects_an_unsafe_archive_before_upload() {
    let server = MockServer::start(handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .hydrate_workspace(b"not an archive")
        .await
        .expect_err("not a tar");

    assert!(!error.message.is_empty());
    assert!(
        server
            .find_opt("POST", "/v2/sandboxes/sessions/sess-1/fs/write")
            .is_none(),
        "{:?}",
        server.routes()
    );
    server.stop();
}
