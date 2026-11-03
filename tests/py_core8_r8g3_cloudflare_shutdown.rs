//! `test_cloudflare_shutdown_logs_on_failure`,
//! `test_cloudflare_shutdown_logs_respect_tool_data_policy`,
//! `test_cloudflare_shutdown_does_not_log_protected_mount_authority` and
//! `test_cloudflare_shutdown_does_not_log_protected_mount_exception` of
//! `tests/extensions/sandbox/test_cloudflare.py`: a failed delete never fails the shutdown, is
//! logged at debug, and carries provider text only when the tool-data policy and the mount
//! configuration allow it.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy, CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded};

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

fn logged() -> String {
    CAPTURE
        .records
        .lock()
        .unwrap()
        .iter()
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn clear() {
    CAPTURE.records.lock().unwrap().clear();
}

async fn worker(delete_reply: fn() -> MockResponse) -> MockServer {
    support::start_resolving(support::Resolver::Cloudflare, move |request: &Recorded| {
        if request.method == "POST" && request.path == "/v1/sandbox" {
            MockResponse::json(json!({ "id": "abc123" }))
        } else if request.method == "DELETE" {
            delete_reply()
        } else {
            MockResponse::json(json!({ "ok": true }))
        }
    })
    .await
}

fn pool_error() -> MockResponse {
    MockResponse::status(
        502,
        json!({"error": "pool error: Failed to start container", "code": "pool_error"}),
    )
}

fn echoed_secret() -> MockResponse {
    let mut reply = MockResponse::text("provider echoed cloudflare-secret-access-key");
    reply.status = 502;
    reply
}

fn ok() -> MockResponse {
    MockResponse::json(json!({}))
}

fn protected_manifest(secret: &str) -> Manifest {
    Manifest::new().with_entry(
        "remote",
        Entry::mount(CloudflareBucketMountStrategy::mount(
            CloudflareBucketMountConfig::s3("bucket").with_credentials("access-key", secret),
        )),
    )
}

async fn shutdown_against(server: &MockServer, manifest: Option<Manifest>) {
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .create(manifest)
            .await
            .expect("create");
    session
        .shutdown()
        .await
        .expect("a failed delete is not raised");
}

/// One test, because the logger is process-wide.
#[tokio::test]
async fn a_failed_delete_is_logged_by_policy_and_never_raised() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Debug);
    let detail =
        "DELETE /sandbox failed: HTTP 502: pool_error: pool error: Failed to start container";

    // The tool-data policy decides whether the reply body is quoted.
    for redacted in [true, false] {
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        clear();
        let server = worker(pool_error).await;
        shutdown_against(&server, None).await;
        server.stop();
        let text = logged();
        assert!(
            text.contains("Failed to delete Cloudflare sandbox on shutdown"),
            "{text}"
        );
        assert_eq!(
            text.contains(detail),
            !redacted,
            "redacted={redacted}: {text}"
        );
    }

    // Protected mount authority keeps the reply body out of the log even when allowed.
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    clear();
    let sentinel = "cloudflare-secret-access-key";
    let server = worker(echoed_secret).await;
    shutdown_against(&server, Some(protected_manifest(sentinel))).await;
    server.stop();
    let text = logged();
    assert!(
        text.contains("Failed to delete Cloudflare sandbox on shutdown"),
        "{text}"
    );
    assert!(!text.contains(sentinel), "{text}");

    // A transport failure is logged too, without provider text under protected authority.
    clear();
    let server = worker(ok).await;
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .create(Some(protected_manifest(sentinel)))
            .await
            .expect("create");
    server.stop();
    session
        .shutdown()
        .await
        .expect("a failed delete is not raised");
    let text = logged();
    assert!(
        text.contains("Failed to delete Cloudflare sandbox on shutdown"),
        "{text}"
    );
    assert!(!text.contains(sentinel), "{text}");

    openai_agents_sdk::debug::set_dont_log_tool_data(true);
}
