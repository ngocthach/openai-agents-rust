//! Vercel provider options in the persisted session state: `tests/extensions/sandbox/test_vercel.py`
//! `test_vercel_create_passes_provider_options`, `..._serialized_session_state_omits_token_and_
//! resume_uses_live_client_token` and `test_vercel_resume_falls_back_to_recreate_when_sandbox_
//! missing`.
//!
//! The runtime, resources, time limits and network policy are written into the state the session
//! hands out, survive a JSON round trip, and are what a replacement sandbox created on resume is
//! built from. A state written before the options existed still reads, and the resuming client's
//! options stand in.
#![cfg(feature = "sandbox")]
#![allow(dead_code, unused_imports)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    NetworkPolicy, VercelPersistedOptions, VercelSandboxClient, VercelSandboxClientOptions,
    VercelSandboxResources,
};
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSessionState};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

const ROOT: &str = "/vercel/sandbox";

fn body(session_id: &str) -> Value {
    json!({
        "sandbox": {
            "name": "agents-test", "persistent": false, "createdAt": 1, "updatedAt": 1,
            "currentSessionId": session_id,
        },
        "session": {
            "id": session_id, "memory": 2048, "vcpus": 2, "region": "iad1", "timeout": 270000,
            "status": "running", "requestedAt": 1, "createdAt": 1, "updatedAt": 1,
            "cwd": ROOT,
        },
        "routes": [],
    })
}

/// A server whose created sandboxes run, and where an existing session lookup answers
/// `existing` (404 when `false`).
async fn server_creating(existing: bool, created: &'static str) -> MockServer {
    MockServer::start(move |request: &Recorded| {
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v3/sandboxes") => MockResponse::json(body(created)),
            ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
                if existing || path.ends_with(created) {
                    let body = body(path.rsplit('/').next().unwrap_or("sess-1"));
                    MockResponse::json(json!({ "session": body["session"], "routes": [] }))
                } else {
                    MockResponse::error(404, "gone")
                }
            }
            ("POST", path) if path.ends_with("/fs/mkdir") || path.ends_with("/fs/write") => {
                MockResponse::json(json!({}))
            }
            ("POST", path) if path.ends_with("/cmd") => MockResponse::ndjson(&[
                json!({ "stream": "stdout", "data": "ok" }),
                json!({
                    "command": {
                        "id": "cmd-1", "name": "sh", "args": [], "cwd": ROOT,
                        "sessionId": created, "exitCode": 0, "startedAt": 1,
                    },
                }),
            ]),
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await
}

async fn server(existing: bool) -> MockServer {
    server_creating(existing, "sess-new").await
}

fn options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

fn policy() -> NetworkPolicy {
    NetworkPolicy::Custom {
        allow: vec!["api.openai.com".to_string()],
        allow_subnets: vec!["10.0.0.0/8".to_string()],
        deny_subnets: vec!["192.168.0.0/16".to_string()],
    }
}

fn manifest() -> Manifest {
    Manifest::new().with_root(ROOT)
}

fn create_bodies(server: &MockServer) -> Vec<Value> {
    server
        .all("POST", "/v3/sandboxes")
        .iter()
        .map(Recorded::json)
        .collect()
}

/// Python's create passes the options on and its state records them: the state a session hands
/// out names the runtime, the resource limits, the time limits and the network policy.
#[tokio::test]
async fn the_session_state_records_the_provider_options() {
    let server = server(true).await;
    let configured = options(&server)
        .with_runtime("node22")
        .with_resources(VercelSandboxResources::new().with_vcpus(4))
        .with_timeout_ms(12_000)
        .with_snapshot_expiration_ms(60_000)
        .with_network_policy(policy());
    let session = VercelSandboxClient::with_options(configured)
        .create(Some(manifest()))
        .await
        .expect("create");

    let state = session.session_state();
    let persisted = VercelPersistedOptions::from_state(&state).expect("options recorded");

    assert_eq!(persisted.runtime.as_deref(), Some("node22"));
    assert_eq!(persisted.resources.vcpus, Some(4));
    assert_eq!(persisted.timeout_ms, Some(12_000));
    assert_eq!(persisted.snapshot_expiration_ms, Some(60_000));
    assert_eq!(persisted.network_policy, Some(policy()));
    let payload = serde_json::to_value(&state).expect("serializes");
    let recorded = &payload["provider_metadata"]["provider_options"];
    assert_eq!(recorded["runtime"], "node22");
    assert_eq!(recorded["resources"]["vcpus"], 4);
    assert_eq!(
        recorded["network_policy"],
        json!({
            "allow": ["api.openai.com"],
            "subnets": { "allow": ["10.0.0.0/8"], "deny": ["192.168.0.0/16"] },
        })
    );
    server.stop();
}

/// `test_vercel_serialized_session_state_omits_token_and_resume_uses_live_client_token`: the
/// state round-trips through JSON with the network policy intact, carries no token, and the
/// resume reattaches with the live client's token without creating a sandbox.
#[tokio::test]
async fn the_options_round_trip_through_json_and_the_token_stays_out() {
    let server = server(true).await;
    let session = VercelSandboxClient::with_options(
        options(&server)
            .with_project_id("project")
            .with_network_policy(policy()),
    )
    .create(Some(manifest()))
    .await
    .expect("create");
    let state = session.session_state();

    let text = serde_json::to_string(&state).expect("serializes");
    let restored: SandboxSessionState = serde_json::from_str(&text).expect("deserializes");

    assert!(!text.contains("vercel_test"), "{text}");
    assert_eq!(restored, state);
    assert_eq!(
        VercelPersistedOptions::from_state(&restored)
            .expect("options")
            .network_policy,
        Some(policy())
    );
    let creates_before = create_bodies(&server).len();
    let resumed = VercelSandboxClient::with_options(options(&server))
        .resume(&restored, manifest())
        .await
        .expect("resume");
    assert_eq!(create_bodies(&server).len(), creates_before, "reattached");
    assert_eq!(
        resumed.session_state().provider_id,
        session.session_state().provider_id
    );
    let lookup = server.find("GET", "/v2/sandboxes/sessions/sess-new");
    assert_eq!(lookup.header("authorization"), Some("Bearer vercel_test"));
    server.stop();
}

/// `test_vercel_resume_falls_back_to_recreate_when_sandbox_missing`: the replacement sandbox is
/// built from the state's runtime, time limit, resources and network policy, not from the
/// options of the client that resumes it.
#[tokio::test]
async fn a_replacement_sandbox_is_built_from_the_recorded_options() {
    let creator = server_creating(true, "sess-old").await;
    let original = VercelSandboxClient::with_options(
        options(&creator)
            .with_runtime("python3.13")
            .with_resources(VercelSandboxResources::new().with_vcpus(4))
            .with_timeout_ms(90_000)
            .with_network_policy(NetworkPolicy::DenyAll),
    )
    .create(Some(manifest()))
    .await
    .expect("create");
    let state = original.session_state();
    let expected_create = create_bodies(&creator).remove(0);
    creator.stop();

    let server = server(false).await;
    let resumed = VercelSandboxClient::with_options(options(&server))
        .resume(&state, manifest())
        .await
        .expect("resume recreates");

    let created = create_bodies(&server);
    assert_eq!(created.len(), 1, "the missing sandbox was replaced");
    for key in ["image", "timeout", "resources", "networkPolicy"] {
        assert_eq!(created[0][key], expected_create[key], "{key}");
    }
    assert_eq!(created[0]["image"], "vercel/sandbox/python:al-3.13.1");
    assert_eq!(created[0]["timeout"], 90_000);
    assert_eq!(created[0]["resources"]["vcpus"], 4);
    assert!(!resumed.system_state_preserved_on_start());
    // The replacement records the same options for the next resume.
    assert_eq!(
        VercelPersistedOptions::from_state(&resumed.session_state()),
        VercelPersistedOptions::from_state(&state)
    );
    server.stop();
}

/// A state written before the options were recorded reads, and the resuming client's options
/// stand in for them.
#[tokio::test]
async fn an_older_payload_without_options_falls_back_to_the_client() {
    let creator = server_creating(true, "sess-old").await;
    let original = VercelSandboxClient::with_options(options(&creator).with_runtime("node22"))
        .create(Some(manifest()))
        .await
        .expect("create");
    let mut payload = serde_json::to_value(original.session_state()).expect("serializes");
    payload
        .as_object_mut()
        .expect("object")
        .remove("provider_metadata");
    creator.stop();

    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");
    assert_eq!(VercelPersistedOptions::from_state(&state), None);
    let server = server(false).await;
    VercelSandboxClient::with_options(options(&server).with_runtime("node26"))
        .resume(&state, manifest())
        .await
        .expect("resume");

    assert_eq!(
        create_bodies(&server)[0]["image"],
        "vercel/sandbox/node:al-26"
    );
    server.stop();
}

/// Every persisted field has a serde default.
#[test]
fn missing_fields_read_as_unset() {
    let restored: VercelPersistedOptions =
        serde_json::from_value(json!({ "runtime": "node22" })).expect("a partial record reads");

    assert_eq!(restored.runtime.as_deref(), Some("node22"));
    assert_eq!(restored.resources, VercelSandboxResources::default());
    assert_eq!(restored.timeout_ms, None);
    assert_eq!(restored.network_policy, None);
}

/// `test_installed_vercel_preserves_released_policy_and_state`: every released policy shape
/// (`allow-all`, `deny-all`, a domain list with subnets, per-domain rules) is written to the
/// state payload as Python writes it and reads back to the same policy.
#[tokio::test]
async fn every_released_policy_shape_round_trips_through_the_state_payload() {
    let cases = [
        json!("allow-all"),
        json!("deny-all"),
        json!({
            "allow": ["example.com"],
            "subnets": { "allow": null, "deny": ["192.168.0.0/16"] },
        }),
        json!({
            "allow": {
                "example.com": [{ "transform": [{ "headers": { "X-Test-Header": "synthetic-value" } }] }]
            },
            "subnets": null,
        }),
    ];
    let server = server(true).await;
    for released in cases {
        let policy = NetworkPolicy::from_state_json(&released).expect("a released shape reads");
        let session = VercelSandboxClient::with_options(
            options(&server)
                .with_runtime("node22")
                .with_network_policy(policy.clone()),
        )
        .create(Some(manifest()))
        .await
        .expect("create");

        let payload = serde_json::to_value(session.session_state()).expect("serializes");
        let recorded = &payload["provider_metadata"]["provider_options"];

        assert_eq!(recorded["network_policy"], released);
        assert_eq!(recorded["runtime"], "node22");
        let restored: SandboxSessionState = serde_json::from_value(payload).expect("reads back");
        assert_eq!(
            VercelPersistedOptions::from_state(&restored)
                .expect("options")
                .network_policy,
            Some(policy)
        );
    }
    server.stop();
}
