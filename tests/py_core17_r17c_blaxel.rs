//! Ports of cases of `tests/extensions/sandbox/test_blaxel.py` the earlier rounds filed as
//! Python-only: `test_read_not_found_via_error_string` (a failed read whose error text says the
//! file is missing is a missing file, whatever the status) and
//! `test_drive_strategy_build_docker_volume_returns_none` (a Blaxel mount is never attached by the
//! Docker runtime).
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy, BlaxelDriveMountConfig,
    BlaxelDriveMountStrategy, BlaxelSandboxClient, BlaxelSandboxClientOptions,
    DEFAULT_BLAXEL_WORKSPACE_ROOT,
};
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxClient, SandboxSession};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded, Shared};

/// A mock serving the control plane and a sandbox API whose file reads answer with `read_reply`.
async fn start(read_status: u16, read_message: &'static str) -> MockServer {
    let url = Shared::new(String::new());
    let slot = url.clone();
    let server = MockServer::start(move |request: &Recorded| {
        let sandbox_url = slot.get();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v0/sandboxes") | ("GET", "/v0/sandboxes/agents-test") => {
                MockResponse::json(json!({
                    "metadata": { "name": "agents-test", "url": sandbox_url },
                    "status": "DEPLOYED",
                }))
            }
            ("POST", "/process") => MockResponse::json(json!({
                "exitCode": 0, "stdout": "/resolved\n", "stderr": "", "status": "completed",
            })),
            ("GET", path) if path.starts_with("/filesystem/") => {
                MockResponse::error(read_status, read_message)
            }
            _ => MockResponse::empty(204),
        }
    })
    .await;
    url.set(server.base_url.clone());
    server
}

async fn session(server: &MockServer) -> std::sync::Arc<dyn SandboxSession> {
    let options = BlaxelSandboxClientOptions::new()
        .with_name("agents-test")
        .with_api_key("bl-token")
        .with_workspace("acme")
        .with_region("us-pdx-1")
        .with_api_base_url(format!("{}/v0", server.base_url));
    BlaxelSandboxClient::new(options)
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .expect("create")
}

/// Python: `test_read_not_found_via_error_string`.
#[tokio::test]
async fn a_failed_read_that_says_the_file_is_missing_is_a_missing_file() {
    for message in ["No such file or directory", "File Not Found"] {
        let server = start(500, message).await;
        let session = session(&server).await;

        let error = session
            .read_file("missing.txt")
            .await
            .expect_err("the read fails");

        assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound, "{message}");
        server.stop();
    }

    // An error that says nothing about a missing file stays a read failure.
    let server = start(500, "unexpected").await;
    let session = session(&server).await;
    let error = session
        .read_file("broken.txt")
        .await
        .expect_err("the read fails");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    server.stop();
}

/// Python: `test_drive_strategy_build_docker_volume_returns_none`.
#[test]
fn blaxel_mounts_are_never_attached_by_the_docker_runtime() {
    let drive = BlaxelDriveMountStrategy::mount(BlaxelDriveMountConfig::new("test-drive"));
    assert!(
        drive
            .strategy
            .docker_create_args(&drive, "/mnt/data")
            .is_none()
    );

    let bucket = BlaxelCloudBucketMountStrategy::mount(BlaxelCloudBucketMountConfig::s3("bucket"));
    assert!(
        bucket
            .strategy
            .docker_create_args(&bucket, "/mnt/data")
            .is_none()
    );
}
