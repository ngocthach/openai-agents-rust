//! Ports of the injected-session cases of `tests/extensions/sandbox/test_vercel.py`:
//! `test_vercel_injected_session_accepts_unchanged_s3_manifest`,
//! `test_vercel_injected_session_revalidates_preexisting_s3_topology_mutation` (a processed
//! manifest that changes the mount, since a Rust session cannot be mutated in place),
//! `test_vercel_injected_session_applies_non_mount_delta_with_fixed_s3_topology`,
//! `test_vercel_live_manifest_update_uses_one_running_snapshot` and
//! `test_vercel_stopped_injected_session_rejects_non_mount_delta_before_state_update`.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::runtime::apply_live_session_manifest;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

const ROOT: &str = "/vercel/sandbox";

fn session_body(status: &str) -> Value {
    json!({
        "sandbox": {"name": "agents-test", "persistent": false, "createdAt": 1,
                    "updatedAt": 1, "currentSessionId": "sess-1"},
        "session": {"id": "sess-1", "memory": 2048, "vcpus": 2, "region": "iad1",
                    "timeout": 270000, "status": status, "requestedAt": 1,
                    "createdAt": 1, "updatedAt": 1, "cwd": ROOT},
        "routes": [],
    })
}

async fn server(status: Arc<Mutex<String>>) -> MockServer {
    MockServer::start(move |request: &Recorded| {
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v3/sandboxes") => MockResponse::json(session_body("running")),
            ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
                let body = session_body(&status.lock().unwrap());
                MockResponse::json(json!({"session": body["session"], "routes": []}))
            }
            ("POST", path) if path.ends_with("/fs/mkdir") || path.ends_with("/fs/write") => {
                MockResponse::json(json!({}))
            }
            ("POST", path) if path.ends_with("/cmd") => {
                let body = request.json();
                let last = body["args"]
                    .as_array()
                    .and_then(|args| args.last())
                    .and_then(Value::as_str)
                    .unwrap_or("ok")
                    .to_string();
                let stdout = if body["command"] == "/usr/bin/realpath" {
                    last
                } else {
                    "ok".into()
                };
                MockResponse::ndjson(&[
                    json!({"stream": "stdout", "data": stdout}),
                    json!({"stream": "stderr", "data": ""}),
                    json!({"command": {"id": "cmd-1", "name": "sh", "args": [], "cwd": ROOT,
                       "sessionId": "sess-1", "exitCode": 0, "startedAt": 1}}),
                ])
            }
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await
}

fn s3_manifest(bucket: &str) -> Manifest {
    Manifest::new().with_root(ROOT).with_entry(
        "remote",
        Entry::mount(
            VercelCloudBucketMountStrategy::new(bucket)
                .with_region("us-west-2")
                .mount(),
        ),
    )
}

async fn created(server: &MockServer) -> Arc<dyn openai_agents_sdk::sandbox::SandboxSession> {
    let options = VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1));
    VercelSandboxClient::with_options(options)
        .create(Some(s3_manifest("test-bucket")))
        .await
        .expect("create")
}

fn writes(server: &MockServer) -> usize {
    server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write")
}

fn session_reads(server: &MockServer) -> usize {
    server.count("GET", "/v2/sandboxes/sessions/sess-1")
}

fn with_capability(base: &Manifest) -> Manifest {
    base.clone()
        .with_entry("capability.txt", Entry::file(b"capability".to_vec()))
}

#[tokio::test]
async fn an_unchanged_manifest_is_accepted() {
    let server = server(Arc::new(Mutex::new("running".into()))).await;
    let session = created(&server).await;
    let current = session.manifest().clone();
    apply_live_session_manifest(session.as_ref(), &current)
        .await
        .expect("accepted");
    assert_eq!(writes(&server), 0);
    server.stop();
}

#[tokio::test]
async fn a_changed_mount_topology_is_refused() {
    let server = server(Arc::new(Mutex::new("running".into()))).await;
    let session = created(&server).await;
    let error = apply_live_session_manifest(session.as_ref(), &s3_manifest("tampered-bucket"))
        .await
        .expect_err("topology is fixed");
    assert!(
        error.to_string().contains("dynamic manifest application"),
        "{error}"
    );
    assert_eq!(writes(&server), 0);
    server.stop();
}

#[tokio::test]
async fn a_non_mount_delta_is_applied_with_the_s3_topology_fixed() {
    let server = server(Arc::new(Mutex::new("running".into()))).await;
    let session = created(&server).await;
    let processed = with_capability(session.manifest());
    apply_live_session_manifest(session.as_ref(), &processed)
        .await
        .expect("applied");
    assert_eq!(writes(&server), 1);
    server.stop();
}

#[tokio::test]
async fn a_live_update_asks_for_the_running_state_once() {
    let server = server(Arc::new(Mutex::new("running".into()))).await;
    let session = created(&server).await;
    let before = session_reads(&server);
    let processed = with_capability(session.manifest());
    apply_live_session_manifest(session.as_ref(), &processed)
        .await
        .expect("applied");
    assert_eq!(session_reads(&server) - before, 1);
    server.stop();
}

#[tokio::test]
async fn a_stopped_session_refuses_a_non_mount_delta() {
    let status = Arc::new(Mutex::new("running".to_string()));
    let server = server(status.clone()).await;
    let session = created(&server).await;
    *status.lock().unwrap() = "stopped".into();
    let processed = with_capability(session.manifest());
    let error = apply_live_session_manifest(session.as_ref(), &processed)
        .await
        .expect_err("must be running");
    assert!(error.to_string().contains("must be running"), "{error}");
    assert_eq!(writes(&server), 0);
    server.stop();
}
