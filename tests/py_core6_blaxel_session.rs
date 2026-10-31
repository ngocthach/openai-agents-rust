//! Ports of the session and client groups of `tests/extensions/sandbox/test_blaxel.py`.
//!
//! Python patches the `blaxel` SDK objects (`SandboxInstance`, `fs`, `process`, `previews`); the
//! Rust client talks to the Blaxel control plane and the per-sandbox API over HTTP, so the same
//! behaviour is asserted against one local mock that serves both and records every request.
//! The mock also plays the in-sandbox path resolver (`resolve-workspace-path`), which is what
//! Python's `_fake_helper_exec_result` does for the same reason.
#![cfg(feature = "sandbox")]

#[path = "py_core6_blaxel_support/mod.rs"]
mod harness;
#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::BTreeMap;
use std::time::Duration;

use harness::{
    World, context, count_prefixed, options, protected_grant, root_manifest, session, session_with,
    start,
};
use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelSandboxClient, BlaxelSandboxClientOptions, BlaxelTimeouts, DEFAULT_API_BASE_URL,
    DEFAULT_BLAXEL_WORKSPACE_ROOT, DEFAULT_REGION, encode_sandbox_path, terminal_url,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSession, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{Shared, percent_decode};

// ---------------------------------------------------------------------------------------------
// TestBlaxelSandboxSession: exec
// ---------------------------------------------------------------------------------------------

/// `test_exec_transport_error`: a failing sandbox API is an exec transport error that names the
/// backend and the provider's own detail.
#[tokio::test]
async fn exec_transport_error_names_the_backend_and_provider_detail() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    // Nothing listens on this port, so the request cannot connect.
    let dead = BlaxelSandboxClient::new(
        options(&server.base_url).with_force_sandbox_url("http://127.0.0.1:1"),
    )
    .create(Some(root_manifest()))
    .await
    .expect("create");

    let error = dead
        .exec(ExecRequest::argv(["echo", "hello"]))
        .await
        .expect_err("transport failure");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(
        error.message.starts_with("Blaxel exec failed: "),
        "{}",
        error.message
    );
    assert_eq!(context(&error, "backend").as_deref(), Some("blaxel"));
    assert!(context(&error, "provider_error").is_some(), "{error:?}");
    drop(session);
    server.stop();
}

/// `test_exec_non_timeout_api_error_becomes_transport`: HTTP 500 keeps its status and message and
/// is retryable.
#[tokio::test]
async fn exec_server_error_becomes_a_retryable_transport_error() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.exec_failure = Some((500, json!({ "message": "internal error" }))));

    let error = session
        .exec(ExecRequest::argv(["echo", "hello"]))
        .await
        .expect_err("500");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(
        error.message,
        "Blaxel exec failed: HTTP 500: internal error"
    );
    assert_eq!(context(&error, "backend").as_deref(), Some("blaxel"));
    assert_eq!(context(&error, "http_status").as_deref(), Some("500"));
    assert_eq!(
        context(&error, "provider_error").as_deref(),
        Some("HTTP 500: internal error")
    );
    assert_eq!(error.retryable, Some(true));
    server.stop();
}

/// `test_exec_maps_sdk_api_error_408_to_timeout` and `..._504_to_timeout`.
#[tokio::test]
async fn exec_maps_408_and_504_to_a_timeout() {
    for status in [408u16, 504] {
        let world = Shared::new(World::default());
        let server = start(world.clone()).await;
        let session = session(&server).await;
        world.with(|w| w.exec_failure = Some((status, json!({ "message": "request timeout" }))));

        let error = session
            .exec(ExecRequest::argv(["sleep", "100"]))
            .await
            .expect_err("timeout status");

        assert_eq!(error.code, ErrorCode::ExecTimeout, "{status}");
        server.stop();
    }
}

/// `test_exec_uses_structured_blaxel_non_retryable_error_code` and
/// `test_exec_uses_structured_blaxel_retryable_error_code`: the structured body decides.
#[tokio::test]
async fn exec_uses_the_structured_blaxel_error_code() {
    for (code, retryable, message) in [
        ("ROUTE_NOT_FOUND", false, "route not found"),
        ("WORKLOAD_UNAVAILABLE", true, "workload unavailable"),
    ] {
        let world = Shared::new(World::default());
        let server = start(world.clone()).await;
        let session = session(&server).await;
        world.with(|w| {
            w.exec_failure = Some((
                404,
                json!({ "error": {
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                    "status": 404,
                }}),
            ))
        });

        let error = session
            .exec(ExecRequest::argv(["echo", "hello"]))
            .await
            .expect_err("404");

        assert_eq!(error.code, ErrorCode::ExecTransportError);
        assert_eq!(
            error.message,
            format!("Blaxel exec failed: HTTP 404: {message}")
        );
        assert_eq!(context(&error, "backend").as_deref(), Some("blaxel"));
        assert_eq!(context(&error, "http_status").as_deref(), Some("404"));
        assert_eq!(
            context(&error, "provider_error").as_deref(),
            Some(format!("HTTP 404: {message}").as_str())
        );
        assert_eq!(
            context(&error, "provider_error_code").as_deref(),
            Some(code)
        );
        assert_eq!(error.retryable, Some(retryable), "{code}");
        server.stop();
    }
}

/// `test_blaxel_retryability_error_code_table`: with no explicit `retryable` flag, the documented
/// code table decides.
#[tokio::test]
async fn exec_error_code_table_decides_retryability() {
    for (code, expected) in [
        ("ROUTE_NOT_FOUND", false),
        ("WORKLOAD_NOT_FOUND", false),
        ("WORKSPACE_NOT_FOUND", false),
        ("WORKLOAD_UNAVAILABLE", true),
        ("AUTHENTICATION_REQUIRED", false),
        ("AUTHENTICATION_FAILED", false),
        ("FORBIDDEN", false),
        ("BAD_REQUEST", false),
        ("USAGE_LIMIT_EXCEEDED", false),
        ("POLICY_VIOLATION", false),
    ] {
        let world = Shared::new(World::default());
        let server = start(world.clone()).await;
        let session = session(&server).await;
        world.with(|w| {
            w.exec_failure = Some((404, json!({ "error": { "code": code, "message": code } })))
        });

        let error = session
            .exec(ExecRequest::argv(["echo", "hello"]))
            .await
            .expect_err("structured error");

        assert_eq!(error.retryable, Some(expected), "{code}");
        assert_eq!(
            context(&error, "provider_error_code").as_deref(),
            Some(code)
        );
        server.stop();
    }
}

/// `test_exec_timeout`: a command that outlives its timeout is an `ExecTimeout`.
#[tokio::test]
async fn exec_timeout_is_an_exec_timeout() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.exec_delay_ms = 1500);

    let error = session
        .exec(ExecRequest::argv(["sleep", "100"]).with_timeout(Some(Duration::from_millis(50))))
        .await
        .expect_err("timeout");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    server.stop();
}

/// `test_exec_timeout_reports_default_timeout`: the timeout the error reports is the session's
/// own exec timeout when the request names none.
#[tokio::test]
async fn exec_timeout_reports_the_default_timeout() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session_with(
        &server,
        options(&server.base_url).with_timeouts(BlaxelTimeouts {
            exec: Duration::from_millis(50),
            ..BlaxelTimeouts::default()
        }),
        root_manifest(),
    )
    .await;
    world.with(|w| w.exec_delay_ms = 1500);

    let error = session
        .exec(ExecRequest::argv(["sleep", "100"]))
        .await
        .expect_err("timeout");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert_eq!(context(&error, "timeout_s").as_deref(), Some("0.05"));
    server.stop();
}

/// `test_coerce_exec_timeout` group: `None` falls back to the configured exec timeout, a positive
/// value passes through, and a zero timeout still sends a strictly positive one.
#[tokio::test]
async fn exec_timeout_is_coerced_into_the_request_body() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session_with(
        &server,
        options(&server.base_url).with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(300),
            ..BlaxelTimeouts::default()
        }),
        root_manifest(),
    )
    .await;

    session.exec(ExecRequest::argv(["a"])).await.expect("exec");
    session
        .exec(ExecRequest::argv(["b"]).with_timeout(Some(Duration::from_millis(42_500))))
        .await
        .expect("exec");
    // A zero timeout is coerced to a millisecond: the request may or may not outrun it, but it
    // is never sent as zero.
    let _ = session
        .exec(ExecRequest::argv(["c"]).with_timeout(Some(Duration::ZERO)))
        .await;

    let timeouts: Vec<(String, u64)> = server
        .requests()
        .iter()
        .filter(|request| request.method == "POST" && request.path == "/process")
        .map(|request| {
            let body = request.json();
            (
                body["command"].as_str().unwrap().to_string(),
                body["timeout"].as_u64().unwrap(),
            )
        })
        .collect();
    // Whole seconds, rounded up, as the Python session sends.
    assert_eq!(timeouts[0], ("a".to_string(), 300_000));
    assert_eq!(timeouts[1], ("b".to_string(), 43_000));
    if let Some(zero) = timeouts.get(2) {
        assert_eq!(zero, &("c".to_string(), 1_000));
    }
    server.stop();
}

/// `test_resolved_envs`: the sandbox's creation environment reaches every command.
#[tokio::test]
async fn exec_carries_the_creation_environment() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session_with(
        &server,
        options(&server.base_url).with_env_var("BASE_KEY", "base_val"),
        root_manifest(),
    )
    .await;

    session
        .exec(ExecRequest::argv(["env"]))
        .await
        .expect("exec");

    let body = server.find("POST", "/process").json();
    assert_eq!(body["env"]["BASE_KEY"], "base_val");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// TestBlaxelSandboxSession: files and paths
// ---------------------------------------------------------------------------------------------

/// `test_read_rejects_workspace_symlink_to_ungranted_path`
#[tokio::test]
async fn read_rejects_a_workspace_symlink_to_an_ungranted_path() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| {
        w.symlinks
            .insert("/workspace/link".into(), "/private".into())
    });

    let error = session
        .read_file("link/secret.txt")
        .await
        .expect_err("the link leaves the workspace");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must not escape root: link/secret.txt"
    );
    assert_eq!(context(&error, "rel").as_deref(), Some("link/secret.txt"));
    assert_eq!(context(&error, "reason").as_deref(), Some("escape_root"));
    assert_eq!(
        context(&error, "resolved_path").as_deref(),
        Some("workspace escape: /private/secret.txt")
    );
    assert_eq!(count_prefixed(&server, "GET", "/filesystem/"), 0);
    server.stop();
}

/// `test_write_rejects_workspace_symlink_to_read_only_extra_path_grant`
#[tokio::test]
async fn write_rejects_a_workspace_symlink_to_a_read_only_grant() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session_with(
        &server,
        options(&server.base_url),
        root_manifest().with_extra_path_grant(protected_grant()),
    )
    .await;
    world.with(|w| {
        w.symlinks
            .insert("/workspace/link".into(), "/tmp/protected".into())
    });

    let error = session
        .write_file("link/out.txt", b"blocked")
        .await
        .expect_err("a write through a read-only grant");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.message,
        "failed to write archive for path: /workspace/link/out.txt"
    );
    assert_eq!(
        context(&error, "path").as_deref(),
        Some("/workspace/link/out.txt")
    );
    assert_eq!(
        context(&error, "reason").as_deref(),
        Some("read_only_extra_path_grant")
    );
    assert_eq!(
        context(&error, "grant_path").as_deref(),
        Some("/tmp/protected")
    );
    assert_eq!(
        context(&error, "resolved_path").as_deref(),
        Some("/tmp/protected/out.txt")
    );
    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 0);
    server.stop();
}

/// `test_mkdir_rejects_workspace_symlink_to_read_only_extra_path_grant`
#[tokio::test]
async fn mkdir_rejects_a_workspace_symlink_to_a_read_only_grant() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session_with(
        &server,
        options(&server.base_url),
        root_manifest().with_extra_path_grant(protected_grant()),
    )
    .await;
    world.with(|w| {
        w.symlinks
            .insert("/workspace/link".into(), "/tmp/protected".into())
    });

    let error = session
        .mkdir("link/newdir", false)
        .await
        .expect_err("a mkdir through a read-only grant");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.message,
        "failed to write archive for path: /workspace/link/newdir"
    );
    assert_eq!(
        context(&error, "path").as_deref(),
        Some("/workspace/link/newdir")
    );
    assert_eq!(
        context(&error, "reason").as_deref(),
        Some("read_only_extra_path_grant")
    );
    assert_eq!(
        context(&error, "grant_path").as_deref(),
        Some("/tmp/protected")
    );
    assert_eq!(
        context(&error, "resolved_path").as_deref(),
        Some("/tmp/protected/newdir")
    );
    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 0);
    server.stop();
}

/// `test_read`, `test_write`, `test_write_str_payload`: a round trip through the filesystem API.
#[tokio::test]
async fn files_round_trip_through_the_filesystem_api() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| {
        w.files
            .insert("/workspace/test.txt".into(), b"file content".to_vec())
    });

    assert_eq!(
        session.read_file("test.txt").await.unwrap(),
        b"file content"
    );

    session
        .write_file("output.txt", b"written data")
        .await
        .unwrap();
    session
        .write_file("text.txt", "hello text".as_bytes())
        .await
        .unwrap();
    let files = world.get().files;
    assert_eq!(files["/workspace/output.txt"], b"written data");
    assert_eq!(files["/workspace/text.txt"], b"hello text");
    server.stop();
}

/// `test_normalize_path_relative`
#[tokio::test]
async fn normalize_path_resolves_relative_to_the_root() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    assert_eq!(
        session.normalize_path("subdir/file.txt").unwrap(),
        "/workspace/subdir/file.txt"
    );
    server.stop();
}

/// `test_normalize_path_escape_blocked` and `test_normalize_path_absolute_blocked`
#[tokio::test]
async fn normalize_path_refuses_escapes_and_foreign_absolute_paths() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    for path in ["../../etc/passwd", "/etc/passwd"] {
        assert_eq!(
            session.normalize_path(path).unwrap_err().code,
            ErrorCode::InvalidManifestPath,
            "{path}"
        );
    }
    server.stop();
}

/// `test_mkdir_root_is_noop`
#[tokio::test]
async fn mkdir_of_the_filesystem_root_is_a_noop() {
    let server = start(Shared::new(World::default())).await;
    let session = session_with(
        &server,
        options(&server.base_url),
        Manifest::new().with_root("/"),
    )
    .await;

    session.mkdir("/", true).await.expect("mkdir /");

    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 0);
    server.stop();
}

/// `test_mkdir_failure`: the filesystem API refusing a directory is an archive write error.
#[tokio::test]
async fn mkdir_failure_is_an_archive_write_error() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.fs_write_status = Some(500));

    let error = session.mkdir("faildir", false).await.expect_err("mkdir");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(context(&error, "reason").as_deref(), Some("mkdir_failed"));
    server.stop();
}

/// `test_read_generic_error`: any non-404 failure is an archive read error.
#[tokio::test]
async fn read_failure_is_an_archive_read_error() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.fs_read_status = Some(500));

    let error = session.read_file("broken.txt").await.expect_err("read");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    server.stop();
}

/// `test_read_not_found`: a file the sandbox does not have is a not-found error.
#[tokio::test]
async fn reading_a_missing_file_is_a_not_found_error() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    let error = session
        .read_file("nonexistent.txt")
        .await
        .expect_err("missing");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

/// `test_write_fs_error`: a failing upload is an archive write error.
#[tokio::test]
async fn write_failure_is_an_archive_write_error() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| w.fs_write_status = Some(500));

    let error = session
        .write_file("fail.txt", b"data")
        .await
        .expect_err("write");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------

/// `test_sandbox_name_property`
#[tokio::test]
async fn the_session_exposes_the_sandbox_name() {
    let server = start(Shared::new(World::default())).await;
    let session = BlaxelSandboxClient::new(options(&server.base_url))
        .create_blaxel_session(Some(root_manifest()))
        .await
        .expect("create");

    assert_eq!(session.sandbox_name(), "agents-test");
    server.stop();
}

/// `test_start_rejects_unsafe_mount_before_provider_work` (both `skip_start` rows): an
/// unacknowledged credential mount is refused before the sandbox is contacted, whether the
/// session materializes the manifest or reconnects to a paused workspace, and the refusal never
/// carries the secret.
#[tokio::test]
async fn start_rejects_an_unsafe_mount_before_any_provider_work() {
    use openai_agents_sdk::extensions::sandbox::blaxel::{
        BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy,
    };
    use openai_agents_sdk::sandbox::entries::Entry;

    let secret = "blaxel-start-secret";
    for reconnect in [false, true] {
        let server = start(Shared::new(World::default())).await;
        let manifest = root_manifest().with_entry(
            "data",
            Entry::mount(BlaxelCloudBucketMountStrategy::mount(
                BlaxelCloudBucketMountConfig::s3("bucket").with_credentials("access-key", secret),
            )),
        );
        let session = if reconnect {
            // A paused sandbox is reconnected by name, and its workspace is not re-materialized.
            let client =
                BlaxelSandboxClient::new(options(&server.base_url).with_pause_on_exit(true));
            let state = SandboxSessionState::new(
                "blaxel",
                "session-1",
                DEFAULT_BLAXEL_WORKSPACE_ROOT,
                &manifest,
            )
            .with_provider_id(Some("agents-test".to_string()));
            client.resume(&state, manifest).await.expect("resume")
        } else {
            session_with(&server, options(&server.base_url), manifest).await
        };

        let error = session
            .start()
            .await
            .expect_err("the mount must be acknowledged");

        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{reconnect}");
        assert!(!format!("{error:?}").contains(secret));
        assert!(!error.message.contains(secret));
        let sandbox_requests = server
            .requests()
            .iter()
            .filter(|request| !request.path.starts_with("/v0/"))
            .count();
        assert_eq!(sandbox_requests, 0, "{reconnect}: {:?}", server.routes());
        server.stop();
    }
}

/// `test_start_mkdir_failure_suppressed`: a workspace root that cannot be created up front is
/// only logged; start carries on and materializes the manifest.
#[tokio::test]
async fn start_survives_a_failing_workspace_root_mkdir() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    // The first write is the up-front root mkdir; materialization retries and succeeds.
    world.with(|w| w.fs_write_failures = 1);

    session.start().await.expect("start");
    assert_eq!(count_prefixed(&server, "PUT", "/filesystem/"), 2);
    server.stop();
}

/// `test_stop_calls_pty_terminate` lives in the PTY file; here `stop` is a plain no-op on a
/// session without terminals.
#[tokio::test]
async fn stop_without_terminals_succeeds() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    session.stop().await.expect("stop");
    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 0);
    server.stop();
}

/// `test_shutdown_delete_raises`, `test_shutdown_delete_logs_warning` and
/// `test_delete_shutdown_raises`: the client's `delete` suppresses a failing sandbox delete too.
#[tokio::test]
async fn client_delete_suppresses_a_failing_sandbox_delete() {
    let world = Shared::new(World {
        delete_status: Some(500),
        ..World::default()
    });
    let server = start(world.clone()).await;
    let client = BlaxelSandboxClient::new(options(&server.base_url));
    let session = client.create(Some(root_manifest())).await.expect("create");

    client.delete(session).await.expect("delete is suppressed");

    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 1);
    server.stop();
}

/// `test_delete`: deleting a healthy session deletes its sandbox.
#[tokio::test]
async fn client_delete_removes_the_sandbox() {
    let server = start(Shared::new(World::default())).await;
    let client = BlaxelSandboxClient::new(options(&server.base_url));
    let session = client.create(Some(root_manifest())).await.expect("create");

    client.delete(session).await.expect("delete");

    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 1);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Exposed ports
// ---------------------------------------------------------------------------------------------

fn preview(url: &str) -> Value {
    json!({ "metadata": { "name": "port" }, "spec": { "url": url } })
}

/// `test_exposed_port` and `test_exposed_port_public_preview`: a public preview resolves to the
/// preview host with TLS, no query string, and is created `public`.
#[tokio::test]
async fn exposed_port_public_preview_has_no_token() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;

    let endpoint = session.resolve_exposed_port(3000).await.expect("endpoint");

    assert_eq!(endpoint.host, "preview.example.com");
    assert!(endpoint.tls);
    assert_eq!(endpoint.query, "");
    let created = server
        .find("POST", "/v0/sandboxes/agents-test/previews")
        .json();
    assert_eq!(created["spec"]["public"], true);
    assert_eq!(created["spec"]["port"], 3000);
    assert_eq!(created["metadata"]["name"], "port-3000");
    assert_eq!(
        count_prefixed(&server, "POST", "/v0/sandboxes/agents-test/previews/"),
        0
    );
    server.stop();
}

/// `test_exposed_port_any_port_without_predeclaration`: no port has to be declared at creation.
#[tokio::test]
async fn exposed_port_needs_no_predeclaration() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    let endpoint = session.resolve_exposed_port(9999).await.expect("endpoint");

    assert_eq!(endpoint.host, "preview.example.com");
    server.stop();
}

/// `test_exposed_port_http_scheme`
#[tokio::test]
async fn exposed_port_http_preview_is_not_tls() {
    let world = Shared::new(World::default());
    world.with(|w| {
        w.previews
            .insert("port-80".into(), preview("http://preview.example.com/"))
    });
    let server = start(world).await;
    let session = session(&server).await;

    let endpoint = session.resolve_exposed_port(80).await.expect("endpoint");

    assert!(!endpoint.tls);
    assert_eq!(endpoint.port, 80);
    server.stop();
}

/// `test_exposed_port_invalid_url` and `test_exposed_port_bad_url_parse`
#[tokio::test]
async fn exposed_port_rejects_unusable_preview_urls() {
    for (url, detail) in [
        ("", "invalid_preview_url"),
        ("https://", "url_parse_failed"),
    ] {
        let world = Shared::new(World::default());
        world.with(|w| w.previews.insert("port-8080".into(), preview(url)));
        let server = start(world).await;
        let session = session(&server).await;

        let error = session
            .resolve_exposed_port(8080)
            .await
            .expect_err("unusable preview");

        assert_eq!(error.code, ErrorCode::ExposedPortUnavailable, "{url:?}");
        assert_eq!(
            context(&error, "detail").as_deref(),
            Some(detail),
            "{url:?}"
        );
        server.stop();
    }
}

/// `test_exposed_port_error`: a control plane failure is an unavailable port.
#[tokio::test]
async fn exposed_port_backend_failure_is_unavailable() {
    let world = Shared::new(World::default());
    world.with(|w| w.preview_create = Some((500, json!({ "message": "backend down" }))));
    let server = start(world).await;
    let session = session(&server).await;

    let error = session
        .resolve_exposed_port(3000)
        .await
        .expect_err("backend down");

    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    assert_eq!(
        context(&error, "detail").as_deref(),
        Some("preview_creation_failed")
    );
    server.stop();
}

/// `test_exposed_port_private_preview`: a private preview gets a token that rides in the query.
#[tokio::test]
async fn exposed_port_private_preview_carries_a_token() {
    let world = Shared::new(World::default());
    world.with(|w| {
        w.preview_create = Some((200, preview("https://preview.example.com:443/")));
        w.token_reply = Some((200, json!({ "spec": { "token": "my-secret-token" } })));
    });
    let server = start(world).await;
    let mut opts = options(&server.base_url);
    opts.exposed_port_public = false;
    let session = session_with(&server, opts, root_manifest()).await;

    let endpoint = session.resolve_exposed_port(8080).await.expect("endpoint");

    let created = server
        .find("POST", "/v0/sandboxes/agents-test/previews")
        .json();
    assert_eq!(created["spec"]["public"], false);
    assert_eq!(
        server.count(
            "POST",
            "/v0/sandboxes/agents-test/previews/port-8080/tokens"
        ),
        1
    );
    assert_eq!(endpoint.query, "bl_preview_token=my-secret-token");
    assert!(
        endpoint
            .url_for("http")
            .unwrap()
            .contains("bl_preview_token=my-secret-token")
    );
    server.stop();
}

/// `test_exposed_port_private_token_error`
#[tokio::test]
async fn exposed_port_private_token_failure_is_unavailable() {
    let world = Shared::new(World::default());
    world.with(|w| {
        w.preview_create = Some((200, preview("https://preview.example.com:443/")));
        w.token_reply = Some((500, json!({ "message": "token service down" })));
    });
    let server = start(world).await;
    let mut opts = options(&server.base_url);
    opts.exposed_port_public = false;
    let session = session_with(&server, opts, root_manifest()).await;

    let error = session
        .resolve_exposed_port(8080)
        .await
        .expect_err("token failure");

    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    assert_eq!(
        context(&error, "detail").as_deref(),
        Some("preview_token_creation_failed")
    );
    server.stop();
}

/// `test_extract_preview_url*`: the URL comes from `spec.url`, `status.url`, a top-level
/// `url`/`endpoint`, or a nested `preview`; none of them is an unavailable port.
#[tokio::test]
async fn preview_url_is_read_from_every_supported_shape() {
    let shapes = [
        json!({ "spec": { "url": "https://p.bl.run" } }),
        json!({ "url": null, "status": { "url": "https://p.bl.run" } }),
        json!({ "url": null, "spec": null, "status": null, "endpoint": "https://p.bl.run" }),
        json!({ "url": "https://p.bl.run" }),
        json!({ "url": null, "endpoint": null, "preview": { "url": "https://p.bl.run" } }),
    ];
    for shape in shapes {
        let world = Shared::new(World::default());
        world.with(|w| w.previews.insert("port-7000".into(), shape.clone()));
        let server = start(world).await;
        let session = session(&server).await;

        let endpoint = session
            .resolve_exposed_port(7000)
            .await
            .unwrap_or_else(|error| panic!("{shape}: {error:?}"));

        assert_eq!(endpoint.host, "p.bl.run", "{shape}");
        server.stop();
    }

    let world = Shared::new(World::default());
    world.with(|w| w.previews.insert("port-7000".into(), json!({})));
    let server = start(world).await;
    let session = session(&server).await;
    let error = session
        .resolve_exposed_port(7000)
        .await
        .expect_err("no url");
    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// supports_pty
// ---------------------------------------------------------------------------------------------

/// `test_supports_pty_with_url_and_token` and `test_supports_pty_without_token`
#[tokio::test]
async fn supports_pty_needs_an_api_key() {
    let server = start(Shared::new(World::default())).await;
    let with_token = session(&server).await;
    assert!(with_token.supports_pty());

    let mut keyless = options(&server.base_url);
    keyless.api_key = None;
    // SAFETY: only this test reads the variable, and it is removed before the client is built.
    unsafe {
        std::env::remove_var("BL_API_KEY");
    }
    let without = session_with(&server, keyless, root_manifest()).await;
    assert!(!without.supports_pty());
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// TestBlaxelSandboxClient and TestBlaxelSandboxClientExtra
// ---------------------------------------------------------------------------------------------

/// `test_create_with_image`, `test_create_with_all_options`, `test_build_create_config_full` and
/// `test_build_create_config_with_ports`: every option reaches the create body.
#[tokio::test]
async fn create_body_carries_every_option() {
    let server = start(Shared::new(World::default())).await;
    BlaxelSandboxClient::new(
        options(&server.base_url)
            .with_image("blaxel/py-app:latest")
            .with_memory(8192)
            .with_region("eu-ams-1")
            .with_port(json!({ "target": 3000, "protocol": "HTTP" }))
            .with_env_var("FOO", "bar")
            .with_label("team", "test")
            .with_ttl("1h")
            .with_pause_on_exit(true),
    )
    .create(Some(root_manifest()))
    .await
    .expect("create");

    let body = server.find("POST", "/v0/sandboxes").json();
    assert_eq!(body["metadata"]["name"], "agents-test");
    assert_eq!(body["metadata"]["labels"], json!({ "team": "test" }));
    assert_eq!(body["spec"]["runtime"]["image"], "blaxel/py-app:latest");
    assert_eq!(body["spec"]["runtime"]["memory"], 8192);
    assert_eq!(body["spec"]["region"], "eu-ams-1");
    assert_eq!(body["spec"]["runtime"]["ttl"], "1h");
    assert_eq!(
        body["spec"]["runtime"]["ports"],
        json!([{ "target": 3000, "protocol": "HTTP" }])
    );
    assert_eq!(
        body["spec"]["runtime"]["envs"],
        json!([{ "name": "FOO", "value": "bar" }])
    );
    server.stop();
}

/// `test_build_create_config_minimal`: only the name, the region and an empty runtime are sent,
/// and no `ports` key appears.
#[tokio::test]
async fn create_body_is_minimal_without_options() {
    let server = start(Shared::new(World::default())).await;
    BlaxelSandboxClient::new(options(&server.base_url))
        .create(None)
        .await
        .expect("create");

    let body = server.find("POST", "/v0/sandboxes").json();
    assert_eq!(body["metadata"]["name"], "agents-test");
    assert!(body["metadata"].get("labels").is_none());
    assert!(body["spec"]["runtime"].get("ports").is_none());
    assert!(body["spec"]["runtime"].get("image").is_none());
    server.stop();
}

/// `test_create_no_name_generates_one`
#[tokio::test]
async fn create_without_a_name_generates_one() {
    let server = start(Shared::new(World::default())).await;
    let mut opts = options(&server.base_url);
    opts.name = None;
    BlaxelSandboxClient::new(opts)
        .create(Some(root_manifest()))
        .await
        .expect("create");

    let body = server.find("POST", "/v0/sandboxes").json();
    assert!(
        body["metadata"]["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("agents-") && name.len() > "agents-".len()),
        "{body}"
    );
    server.stop();
}

/// `test_build_create_config_region_from_env` and `test_build_create_config_default_region`:
/// the region falls back to `BL_REGION`, then to `us-pdx-1`. One test, because it edits the
/// process environment.
#[tokio::test]
async fn create_region_falls_back_to_the_environment_then_the_default() {
    let server = start(Shared::new(World::default())).await;
    let mut opts = options(&server.base_url);
    opts.region = None;

    // SAFETY: no other test in this binary reads `BL_REGION`; each sets its own region.
    unsafe {
        std::env::set_var("BL_REGION", "eu-ams-1");
    }
    BlaxelSandboxClient::new(opts.clone())
        .create(None)
        .await
        .expect("create");
    unsafe {
        std::env::remove_var("BL_REGION");
    }
    BlaxelSandboxClient::new(opts)
        .create(None)
        .await
        .expect("create");

    let regions: Vec<String> = server
        .all("POST", "/v0/sandboxes")
        .iter()
        .map(|request| {
            request.json()["spec"]["region"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        regions,
        vec!["eu-ams-1".to_string(), DEFAULT_REGION.to_string()]
    );
    server.stop();
}

/// `test_client_token_from_env`: the API key falls back to `BL_API_KEY`.
#[tokio::test]
async fn api_key_falls_back_to_the_environment() {
    let mut opts = BlaxelSandboxClientOptions::new();
    // SAFETY: no other test in this binary sets `BL_API_KEY` to a value.
    unsafe {
        std::env::set_var("BL_API_KEY", "env-token");
    }
    let resolved = opts.resolve_api_key();
    opts.api_key = Some("explicit".into());
    let explicit = opts.resolve_api_key();
    unsafe {
        std::env::remove_var("BL_API_KEY");
    }

    assert_eq!(resolved.as_deref(), Some("env-token"));
    assert_eq!(explicit.as_deref(), Some("explicit"));
}

/// `test_create_with_timeouts_dict` and `test_create_with_dictionary_run_config_options`: the
/// configured timeouts are what a command is sent with.
#[tokio::test]
async fn create_applies_the_configured_timeouts() {
    let server = start(Shared::new(World::default())).await;
    let session = session_with(
        &server,
        options(&server.base_url).with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(120),
            cleanup: Duration::from_secs(10),
            ..BlaxelTimeouts::default()
        }),
        root_manifest(),
    )
    .await;

    session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("exec");

    assert_eq!(server.find("POST", "/process").json()["timeout"], 120_000);
    server.stop();
}

/// `test_create_without_manifest`: the default manifest roots the workspace at `/workspace`.
#[tokio::test]
async fn create_without_a_manifest_uses_the_default_root() {
    let server = start(Shared::new(World::default())).await;
    let session = BlaxelSandboxClient::new(options(&server.base_url))
        .create(None)
        .await
        .expect("create");

    assert_eq!(session.manifest().root(), DEFAULT_BLAXEL_WORKSPACE_ROOT);
    server.stop();
}

/// `TestBlaxelTimeouts.test_defaults` and `test_custom_values`
#[test]
fn timeouts_have_the_documented_defaults() {
    let defaults = BlaxelTimeouts::default();
    assert_eq!(defaults.exec, Duration::from_secs(300));
    assert_eq!(defaults.cleanup, Duration::from_secs(30));
    assert_eq!(defaults.file_upload, Duration::from_secs(1800));
    assert_eq!(defaults.file_download, Duration::from_secs(1800));
    assert_eq!(defaults.workspace_tar, Duration::from_secs(300));
    assert_eq!(defaults.fast_op, Duration::from_secs(30));

    let custom = BlaxelTimeouts {
        exec: Duration::from_secs(60),
        cleanup: Duration::from_secs(10),
        fast_op: Duration::from_secs(5),
        ..BlaxelTimeouts::default()
    };
    assert_eq!(custom.exec, Duration::from_secs(60));
    assert_eq!(custom.cleanup, Duration::from_secs(10));
    assert_eq!(custom.fast_op, Duration::from_secs(5));
}

/// `TestBlaxelSandboxClientOptions.test_defaults`
#[test]
fn client_options_have_the_documented_defaults() {
    let opts = BlaxelSandboxClientOptions::new();
    assert!(opts.image.is_none());
    assert!(opts.memory.is_none());
    assert!(opts.region.is_none());
    assert!(opts.ports.is_empty());
    assert!(opts.env_vars.is_empty());
    assert!(opts.labels.is_empty());
    assert!(opts.ttl.is_none());
    assert!(opts.name.is_none());
    assert!(!opts.pause_on_exit);
    assert!(opts.exposed_port_public);
    assert_eq!(opts.timeouts.exec, Duration::from_secs(300));
}

/// `TestBlaxelSandboxSessionState.test_defaults` and `test_serialization_roundtrip`, and
/// `test_deserialize_session_state`: the persisted state names the sandbox and survives a JSON
/// round trip.
#[tokio::test]
async fn session_state_names_the_sandbox_and_round_trips() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    let state = session.session_state();
    assert_eq!(state.backend, "blaxel");
    assert_eq!(state.provider_id.as_deref(), Some("agents-test"));
    let restored: SandboxSessionState =
        serde_json::from_value(serde_json::to_value(&state).expect("serialize")).expect("parse");
    assert_eq!(restored.provider_id.as_deref(), Some("agents-test"));
    assert_eq!(restored.backend, "blaxel");
    assert_eq!(restored.session_id, state.session_id);

    // The restored state resumes the same sandbox by name.
    let resumed = BlaxelSandboxClient::new(options(&server.base_url).with_pause_on_exit(true))
        .resume(&restored, root_manifest())
        .await
        .expect("resume");
    assert_eq!(
        resumed.session_state().provider_id.as_deref(),
        Some("agents-test")
    );
    server.stop();
}

/// `test_resume_reconnects_no_new_url`: a reconnected sandbox that reports no URL falls back to
/// the run endpoint for the workspace.
#[tokio::test]
async fn resume_without_a_reported_url_uses_the_run_endpoint() {
    let world = Shared::new(World {
        sandbox_url: Some(None),
        ..World::default()
    });
    let server = start(world).await;
    let mut opts = options(&server.base_url).with_pause_on_exit(true);
    opts = opts.with_run_base_url(format!("{}/run", server.base_url));
    let client = BlaxelSandboxClient::new(opts);
    let state = SandboxSessionState::new(
        "blaxel",
        "session-1",
        DEFAULT_BLAXEL_WORKSPACE_ROOT,
        &root_manifest(),
    )
    .with_provider_id(Some("agents-test".to_string()));

    let session = client
        .resume(&state, root_manifest())
        .await
        .expect("resume");
    session.exec(ExecRequest::argv(["true"])).await.ok();

    assert_eq!(server.count("GET", "/v0/sandboxes/agents-test"), 1);
    assert_eq!(
        server.count("POST", "/run/acme/sandboxes/agents-test/process"),
        1,
        "{:?}",
        server.routes()
    );
    server.stop();
}

/// `test_get_sandbox_url`, `test_get_sandbox_url_direct_url`, `test_get_sandbox_url_missing` and
/// `test_get_sandbox_url_empty_string`: the reported URL wins; a missing or empty one falls back
/// to the run endpoint.
#[tokio::test]
async fn sandbox_url_is_the_reported_one_or_the_run_endpoint() {
    for (reported, expect_reported) in [(Some("self"), true), (None, false), (Some(""), false)] {
        let world = Shared::new(World::default());
        let server = start(world.clone()).await;
        world.with(|w| {
            w.sandbox_url = Some(match reported {
                Some("self") => Some(server.base_url.clone()),
                Some(other) => Some(other.to_string()),
                None => None,
            })
        });
        let session = BlaxelSandboxClient::new(
            options(&server.base_url).with_run_base_url(format!("{}/run", server.base_url)),
        )
        .create(Some(root_manifest()))
        .await
        .expect("create");

        session.exec(ExecRequest::argv(["true"])).await.ok();

        let reported_hit = server.count("POST", "/process");
        let fallback_hit = server.count("POST", "/run/acme/sandboxes/agents-test/process");
        assert_eq!(
            (reported_hit, fallback_hit),
            if expect_reported { (1, 0) } else { (0, 1) },
            "{reported:?}"
        );
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Terminal URL helper (`_build_ws_url`)
// ---------------------------------------------------------------------------------------------

fn query_pairs(url: &str) -> BTreeMap<String, String> {
    let (_, query) = url.split_once('?').expect("a query string");
    assert!(!query.contains('#'), "{url}");
    query
        .split('&')
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (key.to_string(), percent_decode(&value.replace('+', "%2B")))
        })
        .collect()
}

/// `test_build_ws_url`
#[test]
fn terminal_url_carries_the_session_parameters() {
    let url = terminal_url(
        "https://test.bl.run",
        "tok123",
        "sess-1",
        "/workspace",
        80,
        24,
    );

    assert!(url.starts_with("wss://test.bl.run/terminal/ws?"), "{url}");
    assert!(url.contains("token=tok123"));
    assert!(url.contains("sessionId=sess-1"));
    assert!(url.contains("workingDir=/workspace"));
}

/// `test_build_ws_url_percent_encodes_query_values`: caller-controlled values survive the round
/// trip, and a structural character never adds or overrides a parameter.
#[test]
fn terminal_url_percent_encodes_query_values() {
    let cases = [
        ("cwd", "/workspace/my project"),
        ("cwd", "/workspace/a&rows=9999"),
        ("cwd", "/workspace/a#b"),
        ("cwd", "/workspace/café"),
        ("token", "ab+cd/ef=="),
        ("session_id", "a&b"),
    ];
    for (field, value) in cases {
        let (mut token, mut session_id, mut cwd) = ("tok123", "sess-1", "/workspace");
        match field {
            "cwd" => cwd = value,
            "token" => token = value,
            _ => session_id = value,
        }

        let url = terminal_url("https://test.bl.run", token, session_id, cwd, 80, 24);

        assert!(!url.contains(' '), "{url}");
        let query = query_pairs(&url);
        assert_eq!(query["token"], token, "{field} {value}");
        assert_eq!(query["sessionId"], session_id, "{field} {value}");
        assert_eq!(query["workingDir"], cwd, "{field} {value}");
        assert_eq!(query["rows"], "24");
        assert_eq!(query["cols"], "80");
    }
}

/// `test_build_ws_url_rewrites_only_the_scheme` and `test_build_ws_url_http_scheme`
#[test]
fn terminal_url_rewrites_only_the_scheme() {
    let proxied = terminal_url(
        "https://test.bl.run/proxy/http://inner",
        "t",
        "s",
        "/workspace",
        80,
        24,
    );
    assert!(
        proxied.starts_with("wss://test.bl.run/proxy/http://inner/terminal/ws?"),
        "{proxied}"
    );

    let plain = terminal_url("http://test.bl.run", "tok", "s1", "/w", 80, 24);
    assert!(plain.starts_with("ws://test.bl.run/"), "{plain}");
}

/// The encoder is the one the filesystem paths use, so `/` leads as `%2F`.
#[test]
fn filesystem_paths_encode_the_leading_slash() {
    assert_eq!(encode_sandbox_path("/workspace/a b"), "%2Fworkspace/a%20b");
    assert_eq!(
        DEFAULT_API_BASE_URL.trim_end_matches('/'),
        "https://api.blaxel.ai/v0"
    );
}

// ---------------------------------------------------------------------------------------------
// Logging (`TestShutdownLogging`)
// ---------------------------------------------------------------------------------------------

struct Capture {
    records: std::sync::Mutex<Vec<(log::Level, String)>>,
}

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.records
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture {
    records: std::sync::Mutex::new(Vec::new()),
};

/// `test_shutdown_delete_logs_warning` and `test_running_false_logs_debug`: a failed delete is a
/// warning and a failed health check a debug line, neither silently dropped. One test, because
/// the logger is process-wide.
#[tokio::test]
async fn shutdown_and_health_check_failures_are_logged() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Debug);
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let world = Shared::new(World {
        delete_status: Some(500),
        ..World::default()
    });
    let server = start(world.clone()).await;
    let session = session(&server).await;

    session.shutdown().await.expect("shutdown");
    world.with(|w| w.fs_list_status = Some(503));
    world.with(|w| w.fs_read_status = Some(503));
    assert!(!session.running().await.expect("running"));

    let records = CAPTURE.records.lock().unwrap().clone();
    assert!(
        records
            .iter()
            .any(|(level, message)| *level == log::Level::Warn
                && message.contains("Sandbox delete failed during shutdown")),
        "{records:?}"
    );
    assert!(
        records
            .iter()
            .any(|(level, message)| *level == log::Level::Debug
                && message.contains("Sandbox health check failed")),
        "{records:?}"
    );
    openai_agents_sdk::debug::set_dont_log_tool_data(true);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// rm, ls and mv resolve paths inside the sandbox
// ---------------------------------------------------------------------------------------------

fn non_helper_commands(server: &support::MockServer) -> Vec<String> {
    server
        .requests()
        .iter()
        .filter(|request| request.method == "POST" && request.path == "/process")
        .map(|request| {
            request.json()["command"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .filter(|command| {
            !command.starts_with("test -x") && !command.starts_with(harness::HELPER_PREFIX)
        })
        .collect()
}

/// A workspace symlink that points outside the root cannot be removed, listed or moved through.
#[tokio::test]
async fn rm_ls_and_mv_refuse_a_workspace_symlink_that_leaves_the_root() {
    let world = Shared::new(World::default());
    let server = start(world.clone()).await;
    let session = session(&server).await;
    world.with(|w| {
        w.symlinks
            .insert("/workspace/link".into(), "/private".into())
    });

    for recursive in [false, true] {
        let error = session
            .rm("link/secret.txt", recursive)
            .await
            .expect_err("rm through the link");
        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "rm {recursive}");
        assert_eq!(
            context(&error, "resolved_path").as_deref(),
            Some("workspace escape: /private/secret.txt")
        );
    }
    let error = session.ls("link").await.expect_err("ls through the link");
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    for (source, destination) in [("link/a.txt", "b.txt"), ("a.txt", "link/b.txt")] {
        let error = session
            .mv(source, destination)
            .await
            .expect_err("mv through the link");
        assert_eq!(
            error.code,
            ErrorCode::InvalidManifestPath,
            "{source} {destination}"
        );
    }

    assert_eq!(count_prefixed(&server, "DELETE", "/filesystem/"), 0);
    assert_eq!(count_prefixed(&server, "GET", "/filesystem/"), 0);
    assert!(
        non_helper_commands(&server).is_empty(),
        "{:?}",
        non_helper_commands(&server)
    );
    server.stop();
}

/// Ordinary paths still work, and each operation resolves first.
#[tokio::test]
async fn rm_ls_and_mv_succeed_for_ordinary_paths_after_resolving() {
    let server = start(Shared::new(World::default())).await;
    let session = session(&server).await;

    session.rm("old.txt", false).await.expect("rm");
    session.ls("sub").await.expect("ls");
    session.mv("a.txt", "b.txt").await.expect("mv");

    assert_eq!(count_prefixed(&server, "DELETE", "/filesystem/"), 1);
    assert_eq!(count_prefixed(&server, "GET", "/filesystem/"), 1);
    let commands = non_helper_commands(&server);
    assert_eq!(commands.len(), 1, "{commands:?}");
    assert!(commands[0].contains("mv -fT"), "{commands:?}");
    server.stop();
}

/// A session that relies on every trait default for `rm`, `ls` and `mv`, counting commands.
#[derive(Debug, Default)]
struct DefaultSession {
    manifest: Manifest,
    commands: std::sync::Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl SandboxSession for DefaultSession {
    fn id(&self) -> &str {
        "default"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "default"
    }
    async fn start(&self) -> openai_agents_sdk::sandbox::errors::SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> openai_agents_sdk::sandbox::errors::SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> openai_agents_sdk::sandbox::errors::SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> openai_agents_sdk::sandbox::errors::SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(
        &self,
        request: ExecRequest,
    ) -> openai_agents_sdk::sandbox::errors::SandboxResult<
        openai_agents_sdk::sandbox::types::ExecResult,
    > {
        self.commands.lock().unwrap().push(request.resolved_argv());
        Ok(Default::default())
    }
    async fn read_file(
        &self,
        _: &str,
    ) -> openai_agents_sdk::sandbox::errors::SandboxResult<Vec<u8>> {
        Ok(Vec::new())
    }
    async fn write_file(
        &self,
        _: &str,
        _: &[u8],
    ) -> openai_agents_sdk::sandbox::errors::SandboxResult<()> {
        Ok(())
    }
    async fn hydrate_workspace(
        &self,
        _: &[u8],
    ) -> openai_agents_sdk::sandbox::errors::SandboxResult<()> {
        Ok(())
    }
}

/// Backends that do not opt in keep the default behaviour: `rm`, `ls` and `mv` each make exactly
/// one command, with no resolver round trips.
#[tokio::test]
async fn other_backends_keep_the_unresolved_default() {
    let session = DefaultSession {
        manifest: Manifest::new().with_root("/workspace"),
        ..DefaultSession::default()
    };

    assert!(!session.resolves_paths_remotely());
    session.rm("/workspace/a", false).await.expect("rm");
    session.ls("/workspace").await.expect("ls");
    session
        .mv("/workspace/a", "/workspace/b")
        .await
        .expect("mv");

    let commands = session.commands.lock().unwrap().clone();
    assert_eq!(commands.len(), 3, "{commands:?}");
    assert_eq!(commands[0][0], "rm");
    assert_eq!(commands[1][0], "ls");
    assert!(
        commands[2].iter().any(|part| part.contains("mv -fT")),
        "{commands:?}"
    );
}
