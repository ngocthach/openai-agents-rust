//! Ports of the HTTP-observable behaviour in `tests/extensions/sandbox/test_blaxel.py`.
//!
//! Python patches the `blaxel` SDK objects; the Rust client talks to the Blaxel control plane
//! and the per-sandbox API directly, so the same behaviour is asserted against a local mock that
//! serves both and records each request.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelSandboxClient, BlaxelSandboxClientOptions, BlaxelTimeouts, DEFAULT_BLAXEL_WORKSPACE_ROOT,
    encode_sandbox_path,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSession, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded, Shared};

/// Per-test knobs for the mock sandbox.
#[derive(Clone, Default)]
struct Behavior {
    exec_exit_code: i64,
    exec_stdout: String,
    exec_stderr: String,
    exec_status: Option<u16>,
    read_status: Option<u16>,
    write_status: Option<u16>,
    get_status: Option<u16>,
    delete_status: Option<u16>,
}

fn sandbox_json(url: &str) -> Value {
    json!({ "metadata": { "name": "agents-test", "url": url }, "status": "DEPLOYED" })
}

/// A mock serving the control plane (`/v0/...`) and the sandbox API on one origin.
async fn start(behavior: Shared<Behavior>) -> MockServer {
    let url = Shared::new(String::new());
    let slot = url.clone();
    let server = MockServer::start(move |request: &Recorded| {
        let b = behavior.get();
        let sandbox_url = slot.get();
        let method = request.method.as_str();
        let path = request.path.as_str();
        match (method, path) {
            ("POST", "/v0/sandboxes") => MockResponse::json(sandbox_json(&sandbox_url)),
            ("GET", "/v0/sandboxes/agents-test") => match b.get_status {
                Some(status) => MockResponse::error(status, "gone"),
                None => MockResponse::json(sandbox_json(&sandbox_url)),
            },
            ("DELETE", "/v0/sandboxes/agents-test") => match b.delete_status {
                Some(status) => MockResponse::error(status, "cannot delete"),
                None => MockResponse::empty(204),
            },
            ("POST", "/process") => match b.exec_status {
                Some(status) => MockResponse::error(status, "exec failed"),
                None => MockResponse::json(json!({
                    "exitCode": b.exec_exit_code,
                    "stdout": b.exec_stdout,
                    "stderr": b.exec_stderr,
                    "status": "completed",
                })),
            },
            ("PUT", p) if p.starts_with("/filesystem/") => match b.write_status {
                Some(status) => MockResponse::error(status, "write failed"),
                None => MockResponse::json(json!({ "success": true })),
            },
            ("GET", p) if p.starts_with("/filesystem/") => match b.read_status {
                Some(status) => MockResponse::error(status, "read failed"),
                None => MockResponse::bytes("blob"),
            },
            ("DELETE", p) if p.starts_with("/filesystem/") => {
                MockResponse::json(json!({ "success": true }))
            }
            _ => MockResponse::status(404, json!({ "error": { "code": "ROUTE_NOT_FOUND" } })),
        }
    })
    .await;
    url.set(server.base_url.clone());
    server
}

fn options(base_url: &str) -> BlaxelSandboxClientOptions {
    BlaxelSandboxClientOptions::new()
        .with_name("agents-test")
        .with_api_key("bl-token")
        .with_workspace("acme")
        .with_region("us-pdx-1")
        .with_api_base_url(format!("{base_url}/v0"))
        .with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(5),
            ..BlaxelTimeouts::default()
        })
}

async fn session(server: &MockServer) -> std::sync::Arc<dyn SandboxSession> {
    BlaxelSandboxClient::new(options(&server.base_url))
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .expect("create")
}

/// `test_create` and `test_build_create_config_*`: the create body carries the name, image,
/// memory, region, ports and environment, authenticated with the workspace headers.
#[tokio::test]
async fn blaxel_create_sends_the_full_config() {
    let server = start(Shared::new(Behavior::default())).await;
    let client = BlaxelSandboxClient::new(
        options(&server.base_url)
            .with_image("blaxel/base-image:latest")
            .with_memory(2048)
            .with_env_var("BASE", "1")
            .with_label("owner", "tests")
            .with_ttl("1h"),
    );

    client
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .expect("create");

    let create = server.find("POST", "/v0/sandboxes");
    assert_eq!(create.query, "createIfNotExist=true");
    assert_eq!(
        create.header("x-blaxel-authorization"),
        Some("Bearer bl-token")
    );
    assert_eq!(create.header("x-blaxel-workspace"), Some("acme"));
    let body = create.json();
    assert_eq!(body["metadata"]["name"], "agents-test");
    assert_eq!(body["spec"]["runtime"]["image"], "blaxel/base-image:latest");
    assert_eq!(body["spec"]["runtime"]["memory"], 2048);
    assert_eq!(
        body["spec"]["runtime"]["envs"],
        json!([{ "name": "BASE", "value": "1" }])
    );
    assert_eq!(body["spec"]["region"], "us-pdx-1");
    server.stop();
}

/// `test_exec_success` and `test_exec_success_preserves_split_stderr`
#[tokio::test]
async fn blaxel_exec_preserves_split_stdout_and_stderr() {
    let server = start(Shared::new(Behavior {
        exec_stdout: "out\n".into(),
        exec_stderr: "err\n".into(),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    let result = session
        .exec(ExecRequest::argv(["ls", "-la"]))
        .await
        .expect("exec");

    assert!(result.ok());
    assert_eq!(result.stdout_text(), "out\n");
    assert_eq!(result.stderr_text(), "err\n");
    let process = server.find("POST", "/process").json();
    assert_eq!(process["command"], "ls -la");
    assert_eq!(process["workingDir"], "/workspace");
    assert_eq!(process["waitForCompletion"], true);
    assert_eq!(process["timeout"], 5000);
    server.stop();
}

/// `test_exec_nonzero`
#[tokio::test]
async fn blaxel_exec_reports_a_nonzero_exit_code() {
    let server = start(Shared::new(Behavior {
        exec_exit_code: 2,
        exec_stderr: "bad".into(),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    let result = session
        .exec(ExecRequest::argv(["false"]))
        .await
        .expect("exec");

    assert_eq!(result.exit_code, 2);
    assert_eq!(result.stderr_text(), "bad");
    server.stop();
}

/// `test_exec_transport_error`
#[tokio::test]
async fn blaxel_exec_transport_error() {
    let server = start(Shared::new(Behavior {
        exec_status: Some(502),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    let error = session
        .exec(ExecRequest::argv(["ls"]))
        .await
        .expect_err("the sandbox API failed");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    server.stop();
}

/// `test_mkdir` and `test_mkdir_root_is_noop`
#[tokio::test]
async fn blaxel_mkdir_uses_the_filesystem_api() {
    let server = start(Shared::new(Behavior::default())).await;
    let session = session(&server).await;

    session.mkdir("sub", true).await.expect("mkdir");

    let request = server.find(
        "PUT",
        &format!("/filesystem/{}", encode_sandbox_path("/workspace/sub")),
    );
    assert_eq!(
        request.json(),
        json!({ "isDirectory": true, "permissions": "0755" })
    );
    server.stop();
}

/// `test_mkdir_failure`
#[tokio::test]
async fn blaxel_mkdir_failure_is_reported() {
    let server = start(Shared::new(Behavior {
        write_status: Some(500),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    assert!(session.mkdir("sub", true).await.is_err());
    server.stop();
}

/// `test_read`
#[tokio::test]
async fn blaxel_read_returns_the_file_bytes() {
    let server = start(Shared::new(Behavior::default())).await;
    let session = session(&server).await;

    let bytes = session.read_file("task.md").await.expect("read");

    assert_eq!(bytes, b"blob");
    let request = server
        .requests()
        .into_iter()
        .find(|r| r.method == "GET" && r.path.starts_with("/filesystem/"))
        .expect("a read");
    assert_eq!(request.header("accept"), Some("application/octet-stream"));
    server.stop();
}

/// `test_read_not_found` and `test_read_status_404_via_args_dict`
#[tokio::test]
async fn blaxel_read_404_is_a_not_found() {
    let server = start(Shared::new(Behavior {
        read_status: Some(404),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    let error = session.read_file("missing.txt").await.expect_err("missing");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

/// `test_read_generic_error`
#[tokio::test]
async fn blaxel_read_generic_error_is_not_a_not_found() {
    let server = start(Shared::new(Behavior {
        read_status: Some(500),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    let error = session
        .read_file("task.md")
        .await
        .expect_err("server error");

    assert_ne!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

/// `test_write`, `test_write_str_payload` and `test_write_fs_error`
#[tokio::test]
async fn blaxel_write_uploads_multipart_and_reports_failures() {
    let behavior = Shared::new(Behavior::default());
    let server = start(behavior.clone()).await;
    let session = session(&server).await;

    session
        .write_file("task.md", b"fix it")
        .await
        .expect("write");

    let upload = server
        .requests()
        .into_iter()
        .find(|r| r.method == "PUT" && r.path.starts_with("/filesystem/"))
        .expect("an upload");
    assert!(
        upload
            .header("content-type")
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary=")),
        "{:?}",
        upload.header("content-type")
    );
    assert!(upload.text().contains("fix it"));

    behavior.with(|b| b.write_status = Some(500));
    assert!(session.write_file("task.md", b"x").await.is_err());
    server.stop();
}

/// `test_normalize_path_escape_blocked` and `test_normalize_path_absolute_blocked`: a path
/// outside the workspace root is refused before any request is made.
#[tokio::test]
async fn blaxel_paths_outside_the_workspace_are_refused() {
    let server = start(Shared::new(Behavior::default())).await;
    let session = session(&server).await;
    let before = server.requests().len();

    for path in ["../escape.txt", "/etc/passwd"] {
        let error = session.read_file(path).await.expect_err("outside");
        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "{path}");
    }
    assert_eq!(server.requests().len(), before, "no request was made");
    server.stop();
}

/// `test_running` and `test_running_when_down`: `running` probes the sandbox's filesystem API,
/// and a failing probe means "down".
#[tokio::test]
async fn blaxel_running_probes_the_filesystem_api() {
    let behavior = Shared::new(Behavior::default());
    let server = start(behavior.clone()).await;
    let session = session(&server).await;

    assert!(session.running().await.expect("running"));

    behavior.with(|b| b.read_status = Some(503));
    assert!(!session.running().await.expect("running"));
    server.stop();
}

/// `test_shutdown_deletes`: the sandbox is deleted exactly once.
#[tokio::test]
async fn blaxel_shutdown_deletes_the_sandbox_once() {
    let server = start(Shared::new(Behavior::default())).await;
    let session = session(&server).await;

    session.shutdown().await.expect("shutdown");
    session.shutdown().await.expect("shutdown again");

    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 1);
    server.stop();
}

/// `test_shutdown_pause_on_exit`: with `pause_on_exit` the sandbox is kept.
#[tokio::test]
async fn blaxel_shutdown_pause_on_exit_keeps_the_sandbox() {
    let server = start(Shared::new(Behavior::default())).await;
    let session = BlaxelSandboxClient::new(options(&server.base_url).with_pause_on_exit(true))
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .expect("create");

    session.shutdown().await.expect("shutdown");

    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 0);
    server.stop();
}

/// `test_shutdown_delete_raises`
#[tokio::test]
async fn blaxel_shutdown_surfaces_a_delete_failure() {
    let server = start(Shared::new(Behavior {
        delete_status: Some(500),
        ..Behavior::default()
    }))
    .await;
    let session = session(&server).await;

    assert!(session.shutdown().await.is_err());
    server.stop();
}

/// `test_resume_reconnects`, `test_resume_creates_new` and
/// `test_resume_pause_on_exit_get_fails_falls_back`: only a `pause_on_exit` session reconnects
/// by name; it falls back to creating when the lookup fails, and any other session creates.
#[tokio::test]
async fn blaxel_resume_reconnects_or_creates() {
    for (pause_on_exit, get_status, expected_creates) in [
        (true, None, 0usize),
        (true, Some(404u16), 1),
        (false, None, 1),
    ] {
        let server = start(Shared::new(Behavior {
            get_status,
            ..Behavior::default()
        }))
        .await;
        let client =
            BlaxelSandboxClient::new(options(&server.base_url).with_pause_on_exit(pause_on_exit));
        let state = SandboxSessionState::new(
            "blaxel",
            "session-1",
            DEFAULT_BLAXEL_WORKSPACE_ROOT,
            &Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        )
        .with_provider_id(Some("agents-test".to_string()));

        client
            .resume(
                &state,
                Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
            )
            .await
            .expect("resume");

        assert_eq!(
            server.count("POST", "/v0/sandboxes"),
            expected_creates,
            "{pause_on_exit} {get_status:?}"
        );
        server.stop();
    }
}

/// `test_resume_wrong_state_type`: a state from another backend is refused.
#[tokio::test]
async fn blaxel_resume_rejects_a_foreign_backend_state() {
    let server = start(Shared::new(Behavior::default())).await;
    let state = SandboxSessionState::new("e2b", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("x".to_string()));

    let error = BlaxelSandboxClient::new(options(&server.base_url))
        .resume(&state, Manifest::new())
        .await
        .expect_err("foreign state");

    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    server.stop();
}

/// `test_exposed_port_*`: the endpoint comes from the sandbox's preview.
#[tokio::test]
async fn blaxel_requires_a_workspace() {
    // SAFETY: the test process sets its own environment before any client reads it.
    unsafe {
        std::env::remove_var("BL_WORKSPACE");
    }
    let error = BlaxelSandboxClient::new(BlaxelSandboxClientOptions::new().with_api_key("bl"))
        .create(None)
        .await
        .expect_err("no workspace");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}
