//! Cloudflare and Daytona cases that the earlier ports of `tests/extensions/sandbox/` left as
//! Python-only: the in-sandbox path check on bucket mounts, reads and writes, and the bounded
//! teardown of a PTY whose log stream stalls.
//!
//! Python spies on the private path validator; the observable behaviour is that a path the
//! sandbox's own resolver refuses never reaches the worker's mount or file endpoints.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy, CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions, DaytonaSandboxTimeouts,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecShell, Manifest, PtyExecRequest, S3Mount, SandboxClient, SnapshotSpec,
};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded};

// ---------------------------------------------------------------------------------------------
// Cloudflare
// ---------------------------------------------------------------------------------------------

/// A worker whose in-sandbox resolver refuses every path that mentions `escape-me`.
async fn refusing_worker() -> MockServer {
    MockServer::start(move |request: &Recorded| {
        if support::is_resolver_request(request) {
            if request.text().contains("escape-me") {
                return support::resolver_result(
                    support::Resolver::Cloudflare,
                    111,
                    "",
                    "workspace escape",
                );
            }
            return support::resolver_reply(support::Resolver::Cloudflare);
        }
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/sandbox") => MockResponse::json(json!({ "id": "abc123" })),
            ("POST", path) if path.ends_with("/persist") => MockResponse::bytes(Vec::new()),
            ("POST", path) if path.ends_with("/exec") => {
                support::resolver_result(support::Resolver::Cloudflare, 0, "", "")
            }
            (_, path) if path.ends_with("/file/workspace/escape-me/x") => {
                MockResponse::bytes(b"leaked".to_vec())
            }
            _ => MockResponse::json(json!({ "ok": true })),
        }
    })
    .await
}

fn client(server: &MockServer) -> CloudflareSandboxClient {
    CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
}

fn escaping_mount_manifest() -> Manifest {
    Manifest::new().with_entry(
        "escape-me",
        Entry::mount(CloudflareBucketMountStrategy::mount(
            CloudflareBucketMountConfig::from_configured_mount(&S3Mount::new("bucket"))
                .expect("config"),
        )),
    )
}

fn count_suffix(server: &MockServer, suffix: &str) -> usize {
    server
        .requests()
        .iter()
        .filter(|request| request.path.ends_with(suffix))
        .count()
}

/// `test_cloudflare_mount_and_unmount_validate_path_access_for_write` (mount): a mount path the
/// sandbox resolves outside the workspace is refused before the worker is asked to mount it.
#[tokio::test]
async fn a_mount_path_the_sandbox_resolver_refuses_is_never_mounted() {
    let server = refusing_worker().await;
    let session = client(&server)
        .create(Some(escaping_mount_manifest()))
        .await
        .expect("create");

    let error = session.start().await.expect_err("the mount path escapes");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath, "{error:?}");
    assert_eq!(count_suffix(&server, "/mount"), 0);
    server.stop();
}

/// `test_cloudflare_mount_and_unmount_validate_path_access_for_write` (unmount): the detach that
/// precedes a snapshot checks the path the same way.
#[tokio::test]
async fn an_unmount_path_the_sandbox_resolver_refuses_is_never_unmounted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = refusing_worker().await;
    let session = client(&server)
        .with_snapshot(SnapshotSpec::Local {
            base_path: dir.path().to_path_buf(),
        })
        .create(Some(escaping_mount_manifest()))
        .await
        .expect("create");

    let error = persist_session_snapshot(&*session)
        .await
        .expect_err("the unmount path escapes");

    assert_eq!(
        error.code,
        ErrorCode::WorkspaceArchiveReadError,
        "{error:?}"
    );
    assert_eq!(count_suffix(&server, "/unmount"), 0);
    server.stop();
}

/// `test_cloudflare_read_validates_path_access` and
/// `test_cloudflare_write_validates_path_access_for_write`: neither a read nor a write reaches
/// the file endpoint for a path the resolver refuses.
#[tokio::test]
async fn reads_and_writes_check_the_path_with_the_sandbox_resolver_first() {
    let server = refusing_worker().await;
    let session = client(&server).create(None).await.expect("create");

    let read = session
        .read_file("escape-me/x")
        .await
        .expect_err("read refused");
    let write = session
        .write_file("escape-me/x", b"data")
        .await
        .expect_err("write refused");

    assert_eq!(read.code, ErrorCode::InvalidManifestPath, "{read:?}");
    assert_eq!(write.code, ErrorCode::InvalidManifestPath, "{write:?}");
    assert_eq!(count_suffix(&server, "/file/workspace/escape-me/x"), 0);
    // An allowed path still reaches the endpoint, so the refusals are the resolver's doing.
    session
        .write_file("fine.txt", b"data")
        .await
        .expect("write");
    assert_eq!(count_suffix(&server, "/file/workspace/fine.txt"), 1);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Daytona
// ---------------------------------------------------------------------------------------------

fn daytona_platform(request: &Recorded) -> Option<MockResponse> {
    let path = request.path.as_str();
    if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
        return Some(MockResponse::json(
            json!({ "id": "sbx-daytona", "state": "started" }),
        ));
    }
    None
}

/// A Daytona API whose toolbox answers through `toolbox`, after the delay it names.
async fn delayed_daytona(
    toolbox: impl Fn(&Recorded) -> (u64, MockResponse) + Send + Sync + 'static,
) -> MockServer {
    let toolbox = Arc::new(toolbox);
    MockServer::start_raw(move |request, mut stream| {
        let toolbox = toolbox.clone();
        async move {
            let (delay_ms, response) = match daytona_platform(&request) {
                Some(response) => (0, response),
                None => toolbox(&request),
            };
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let _ = support::write_response(&mut stream, response).await;
        }
    })
    .await
}

/// `test_terminate_pty_entry_awaits_worker_finalizer` and
/// `test_terminate_pty_entry_bounds_worker_finalizer`: tearing down a logged command whose log
/// and status endpoints never answer still returns promptly, and the toolbox session that holds
/// the command is deleted.
#[tokio::test]
async fn terminating_a_logged_command_is_bounded_when_its_log_stream_stalls() {
    let server = delayed_daytona(|request| {
        let path = request.path.as_str();
        if path.ends_with("/exec") {
            return (0, MockResponse::json(json!({ "cmdId": "cmd-1" })));
        }
        if path.ends_with("/logs") || (path.ends_with("/command/cmd-1") && request.method == "GET")
        {
            return (30_000, MockResponse::json(json!({})));
        }
        (0, MockResponse::json(json!({})))
    })
    .await;
    let timeouts = DaytonaSandboxTimeouts {
        cleanup: Duration::from_millis(100),
        ..DaytonaSandboxTimeouts::default()
    };
    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_state_poll_interval(Duration::from_millis(1))
            .with_timeouts(timeouts),
    )
    .create(None)
    .await
    .expect("create");
    let started = session
        .pty_exec_start(
            PtyExecRequest::new("sleep 60")
                .with_shell(ExecShell::None)
                .with_tty(false)
                .with_yield_time_ms(200),
        )
        .await
        .expect("start");
    assert!(started.process_id.is_some(), "{started:?}");

    let began = Instant::now();
    session.pty_terminate_all().await.expect("terminate");

    assert!(
        began.elapsed() < Duration::from_secs(5),
        "{:?}",
        began.elapsed()
    );
    let deleted = server
        .requests()
        .into_iter()
        .filter(|request| request.method == "DELETE" && request.path.contains("/process/session/"))
        .count();
    assert_eq!(deleted, 1, "{:?}", server.routes());
    server.stop();
}
