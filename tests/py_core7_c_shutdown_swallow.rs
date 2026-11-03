//! Python's Daytona `_shutdown_backend` swallows every provider error (`except Exception: pass`),
//! so a delete that the provider rejects does not fail the teardown of a finished run.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions,
};
use openai_agents_sdk::sandbox::SandboxClient;
use serde_json::json;
use support::{MockResponse, Recorded};

#[tokio::test]
async fn a_rejected_daytona_delete_does_not_fail_shutdown() {
    let server =
        support::start_resolving(support::Resolver::Daytona, |request: &Recorded| {
            match (request.method.as_str(), request.path.as_str()) {
                ("DELETE", "/sandbox/sbx-daytona") => MockResponse::error(500, "delete refused"),
                (_, "/sandbox" | "/sandbox/sbx-daytona") => {
                    MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
                }
                _ => MockResponse::json(
                    json!({ "output": "", "stdout": "", "stderr": "", "exitCode": 0 }),
                ),
            }
        })
        .await;
    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create");

    session
        .shutdown()
        .await
        .expect("a failed delete is swallowed");

    assert_eq!(server.count("DELETE", "/sandbox/sbx-daytona"), 1);
    server.stop();
}
