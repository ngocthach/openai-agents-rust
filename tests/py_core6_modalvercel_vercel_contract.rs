//! Ports of `tests/extensions/sandbox/test_vercel_provider_contract.py`.
//!
//! Python runs the installed `vercel.sandbox` SDK against a fake HTTP transport; the Rust client
//! is that transport's other end, so the same wire contract is asserted against a local mock that
//! records each request.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    NetworkPolicy, NetworkPolicyRule, NetworkTransformer, VercelSandboxClient,
    VercelSandboxClientOptions, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{Counter, MockResponse, MockServer, Recorded};

fn body(status: &str, session_id: &str) -> Value {
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
            "timeout": 12500,
            "status": status,
            "requestedAt": 1,
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": "/vercel/sandbox",
        },
        "routes": [{ "port": 3000, "url": "https://app.example.com" }],
    })
}

fn handler() -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    move |request: &Recorded| match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => MockResponse::json(body("running", "session-1")),
        ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
            let loaded = body("running", "session-1");
            MockResponse::json(json!({ "session": loaded["session"], "routes": loaded["routes"] }))
        }
        ("POST", path) if path.ends_with("/cmd") => MockResponse::ndjson(&[
            json!({ "stream": "stdout", "data": "synthetic stdout" }),
            json!({ "stream": "stderr", "data": "synthetic stderr" }),
            json!({
                "command": {
                    "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                    "sessionId": "session-1", "exitCode": 7, "startedAt": 1,
                },
            }),
        ]),
        ("POST", path) if path.ends_with("/snapshot") => {
            MockResponse::json(json!({ "snapshot": { "id": "snapshot-synthetic" } }))
        }
        ("POST", path) if path.ends_with("/stop") => {
            let stopped = body("stopped", "session-1");
            MockResponse::json(json!({ "session": stopped["session"] }))
        }
        _ => MockResponse::json(json!({})),
    }
}

fn options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("synthetic-token")
        .with_project_id("synthetic-project")
        .with_team_id("synthetic-team")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

/// `test_installed_provider_wire_options_output_resume_and_cleanup`: the create request carries the
/// options on the wire, a command's output and exit code come back whole, the exposed port
/// resolves, a resume reuses the running session, and delete stops it with a stop request, never
/// a `DELETE`.
#[tokio::test]
async fn wire_options_output_resume_and_cleanup() {
    let server = MockServer::start(handler()).await;
    let policy = NetworkPolicy::CustomRules {
        allow: std::collections::BTreeMap::from([(
            "example.com".to_string(),
            vec![
                NetworkPolicyRule::new()
                    .with_transform(NetworkTransformer::new().with_header("X-Test", "synthetic")),
            ],
        )]),
        allow_subnets: Vec::new(),
        deny_subnets: Vec::new(),
    };
    let client = VercelSandboxClient::with_options(
        options(&server)
            .with_timeout_ms(12_500)
            .with_runtime("node22")
            .with_exposed_port(3000)
            .with_network_policy(policy),
    );

    let session = client.create(None).await.expect("create");

    let request = server.find("POST", "/v3/sandboxes");
    let created = request.json();
    assert_eq!(created["timeout"], 12_500);
    assert_eq!(created["image"], "vercel/sandbox/node:al-22");
    assert_eq!(created["persistent"], false);
    assert_eq!(
        created["networkPolicy"],
        json!({"allow": {"example.com": [{"transform": [{"headers": {"X-Test": "synthetic"}}]}]}})
    );
    assert_eq!(
        request.header("authorization"),
        Some("Bearer synthetic-token")
    );
    assert_eq!(created["projectId"], "synthetic-project");

    let result = session
        .exec(ExecRequest::shell("exit 7"))
        .await
        .expect("exec");
    assert_eq!(result.stdout, b"synthetic stdout");
    assert_eq!(result.stderr, b"synthetic stderr");
    assert_eq!(result.exit_code, 7);

    let endpoint = session.resolve_exposed_port(3000).await.expect("port");
    assert_eq!(
        (endpoint.host.as_str(), endpoint.port, endpoint.tls),
        ("app.example.com", 443, true)
    );

    let state = session.session_state();
    assert_eq!(state.provider_id.as_deref(), Some("agents-test:session-1"));
    let resumed = client
        .resume(&state, Manifest::new())
        .await
        .expect("resume");
    assert_eq!(resumed.session_state().provider_id, state.provider_id);
    assert_eq!(
        server.count("POST", "/v3/sandboxes"),
        1,
        "nothing was created again"
    );

    client.delete(session).await.expect("delete");
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/session-1/stop"),
        1
    );
    assert!(
        server
            .requests()
            .iter()
            .all(|request| request.method != "DELETE"),
        "cleanup never deletes the sandbox"
    );
    server.stop();
}

/// `test_installed_provider_snapshot_preserves_millisecond_lifetime`: the snapshot request carries
/// the requested lifetime in milliseconds, an explicit zero included, and nothing when none was
/// asked for.
#[tokio::test]
async fn snapshot_preserves_the_millisecond_lifetime() {
    for expiration in [None, Some(0u64), Some(86_400_000)] {
        let server = MockServer::start(handler()).await;
        let mut configured =
            options(&server).with_workspace_persistence(WorkspacePersistence::Snapshot);
        if let Some(milliseconds) = expiration {
            configured = configured.with_snapshot_expiration_ms(milliseconds);
        }
        let session = VercelSandboxClient::with_options(configured)
            .create(None)
            .await
            .expect("create");

        let reference = session.persist_workspace().await.expect("persist");

        assert!(
            String::from_utf8_lossy(&reference).contains("snapshot-synthetic"),
            "{expiration:?}"
        );
        let snapshot = server.find("POST", "/v2/sandboxes/sessions/session-1/snapshot");
        match expiration {
            None => assert_eq!(snapshot.json(), json!({}), "{expiration:?}"),
            Some(milliseconds) => {
                assert_eq!(snapshot.json(), json!({ "expiration": milliseconds }));
            }
        }
        server.stop();
    }
}

/// `test_installed_provider_preserves_runtime_selection`
#[tokio::test]
async fn runtime_selection_picks_the_released_image() {
    for (runtime, image) in [
        (None, "vercel/sandbox/node:al-24"),
        (Some("node22"), "vercel/sandbox/node:al-22"),
        (Some("node24"), "vercel/sandbox/node:al-24"),
        (Some("node26"), "vercel/sandbox/node:al-26"),
        (Some("python3.13"), "vercel/sandbox/python:al-3.13.1"),
    ] {
        let server = MockServer::start(handler()).await;
        let mut configured = options(&server);
        if let Some(runtime) = runtime {
            configured = configured.with_runtime(runtime);
        }

        VercelSandboxClient::with_options(configured)
            .create(None)
            .await
            .expect("create");

        assert_eq!(
            server.find("POST", "/v3/sandboxes").json()["image"],
            image,
            "{runtime:?}"
        );
        server.stop();
    }
}

/// `test_installed_provider_rejects_unknown_runtime_before_allocation`
#[tokio::test]
async fn an_unknown_runtime_is_refused_before_any_request() {
    let server = MockServer::start(handler()).await;

    let error = VercelSandboxClient::with_options(options(&server).with_runtime("unsupported"))
        .create(None)
        .await
        .expect_err("an unsupported runtime");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(
        error.message.contains("unsupported Vercel runtime"),
        "{}",
        error.message
    );
    assert!(server.requests().is_empty(), "no sandbox was allocated");
    server.stop();
}

/// `test_installed_provider_failed_delete_keeps_cleanup_retryable`: a stop that fails is reported,
/// and a second delete tries again instead of treating the session as closed.
#[tokio::test]
async fn a_failed_delete_keeps_cleanup_retryable() {
    let stops = Counter::new();
    let inner = handler();
    let counted = stops.clone();
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/stop") && counted.next("stop") == 0 {
            return MockResponse::error(503, "unavailable");
        }
        inner(request)
    })
    .await;
    let client = VercelSandboxClient::with_options(options(&server));
    let session = client.create(None).await.expect("create");

    let failure = client
        .delete(session.clone())
        .await
        .expect_err("the first stop fails");
    assert_eq!(failure.retryable, Some(true));

    client.delete(session).await.expect("the retry succeeds");
    assert_eq!(stops.get("stop"), 2);
    server.stop();
}

/// `test_installed_provider_pending_reconnect_observes_transition`: while a resume waits on a
/// pending session, a transition to `running` reconnects, and any other transition is replaced
/// promptly rather than waited out.
#[tokio::test]
async fn a_pending_reconnect_observes_the_transition() {
    for next_status in [
        "running",
        "stopping",
        "stopped",
        "failed",
        "aborted",
        "snapshotting",
    ] {
        let lookups = Counter::new();
        let inner = handler();
        let counted = lookups.clone();
        let server = MockServer::start(move |request: &Recorded| {
            if request.method == "GET" && request.path == "/v2/sandboxes/sessions/session-1" {
                let status = match counted.next("get") {
                    0 | 1 => "pending",
                    _ => next_status,
                };
                let loaded = body(status, "session-1");
                return MockResponse::json(json!({ "session": loaded["session"], "routes": [] }));
            }
            if request.method == "POST" && request.path == "/v3/sandboxes" {
                return MockResponse::json(body("running", "session-2"));
            }
            inner(request)
        })
        .await;
        let state =
            SandboxSessionState::new("vercel", "state-1", "/vercel/sandbox", &Manifest::new())
                .with_provider_id(Some("agents-test:session-1".to_string()));
        let started = std::time::Instant::now();

        VercelSandboxClient::with_options(options(&server))
            .resume(&state, Manifest::new())
            .await
            .expect("resume");

        let created = server.count("POST", "/v3/sandboxes");
        if next_status == "running" {
            assert_eq!(created, 0, "{next_status}: the session is kept");
        } else {
            assert_eq!(created, 1, "{next_status}: a replacement is created");
        }
        assert!(started.elapsed() < Duration::from_secs(5), "{next_status}");
        server.stop();
    }
}
