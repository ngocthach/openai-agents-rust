//! Vercel create and resume futures that are dropped part way. Ports of:
//!
//! - `tests/extensions/sandbox/test_vercel_provider_contract.py::test_installed_provider_cancelled_creation_cleans_allocation`:
//!   a creation cancelled after the sandbox was allocated stops it (no `DELETE`: the named
//!   resource is kept). Python blocks the create response itself and finds the allocation by
//!   name; the Rust client awaits nothing between the create response and the wait for
//!   `running`, so the cancellation lands in that wait, where the session id is known.
//! - `test_installed_provider_closes_unadopted_reconnect_client[cancelled]`: a resume cancelled
//!   while it waits for the original, still `pending`, session creates nothing and leaves the
//!   original alone.
//!
//! Python's `task.cancel()` is a dropped future in Rust. The cleanup Python shields from the
//! cancellation is spawned on the runtime when the future is dropped, so the test waits for the
//! request to arrive.

#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::future::Future;
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSessionState};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

const ROOT: &str = "/vercel/sandbox";

fn session(id: &str, status: &str) -> Value {
    json!({
        "id": id,
        "memory": 2048,
        "vcpus": 2,
        "region": "iad1",
        "timeout": 270000,
        "status": status,
        "requestedAt": 1,
        "createdAt": 1,
        "updatedAt": 1,
        "cwd": ROOT,
    })
}

fn created(id: &str) -> MockResponse {
    MockResponse::json(json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": id,
        },
        "session": session(id, "pending"),
        "routes": [],
    }))
}

/// A wire where every session stays `pending` and a stop succeeds. `original` answers the
/// session lookup a resume starts with; without it the original session is gone.
async fn pending_wire(original: bool) -> MockServer {
    MockServer::start(move |request: &Recorded| {
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v3/sandboxes") => created("sess-new"),
            ("GET", "/v2/sandboxes/sessions/sess-new") => {
                MockResponse::json(json!({ "session": session("sess-new", "pending") }))
            }
            ("GET", "/v2/sandboxes/sessions/sess-old") if original => MockResponse::json(
                json!({ "session": session("sess-old", "pending"), "routes": [] }),
            ),
            ("POST", path) if path.ends_with("/stop") => {
                MockResponse::json(json!({ "session": session("sess-new", "stopped") }))
            }
            _ => MockResponse::error(404, "unexpected"),
        }
    })
    .await
}

fn client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(
        VercelSandboxClientOptions::new()
            .with_token("vercel_test")
            .with_base_url(&server.base_url)
            .with_status_poll_interval(Duration::from_millis(5))
            .with_wait_for_running(Duration::from_secs(60)),
    )
}

fn old_state() -> SandboxSessionState {
    SandboxSessionState::new("vercel", "state-1", ROOT, &Manifest::new().with_root(ROOT))
        .with_provider_id(Some("agents-test:sess-old".to_string()))
}

/// Poll `future` until `server` has seen `path` polled, then drop it unfinished.
async fn drop_while_polling<F: Future>(future: F, server: &MockServer, path: &str) {
    let mut future = Box::pin(future);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::select! {
                _ = &mut future => panic!("the future finished before it was dropped"),
                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    if server.count("GET", path) > 0 {
                        return;
                    }
                }
            }
        }
    })
    .await
    .expect("the session is polled");
    drop(future);
}

async fn wait_for(server: &MockServer, method: &str, path: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while server.count(method, path) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{method} {path} never arrived: {:?}", server.routes()));
}

fn no_delete(server: &MockServer) -> bool {
    server.requests().iter().all(|r| r.method != "DELETE")
}

#[tokio::test]
async fn a_dropped_create_stops_the_allocated_sandbox() {
    let server = pending_wire(false).await;
    let client = client(&server);

    drop_while_polling(
        client.create(Some(Manifest::new().with_root(ROOT))),
        &server,
        "/v2/sandboxes/sessions/sess-new",
    )
    .await;

    wait_for(&server, "POST", "/v2/sandboxes/sessions/sess-new/stop").await;
    assert_eq!(server.count("POST", "/v3/sandboxes"), 1);
    assert!(no_delete(&server), "{:?}", server.routes());
    server.stop();
}

#[tokio::test]
async fn a_resume_dropped_while_reconnecting_creates_nothing_and_keeps_the_original() {
    let server = pending_wire(true).await;
    let client = client(&server);
    let state = old_state();

    drop_while_polling(
        client.resume(&state, Manifest::new().with_root(ROOT)),
        &server,
        "/v2/sandboxes/sessions/sess-old",
    )
    .await;
    // Give a wrongly spawned cleanup the time to show up.
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(server.count("POST", "/v3/sandboxes"), 0);
    assert!(
        server
            .requests()
            .iter()
            .all(|r| r.method != "DELETE" && !r.path.ends_with("/stop")),
        "{:?}",
        server.routes()
    );
    server.stop();
}

/// The same guarantee on the resume path that replaces a missing session: the replacement the
/// dropped resume created is stopped.
#[tokio::test]
async fn a_resume_dropped_while_its_replacement_starts_stops_the_replacement() {
    let server = pending_wire(false).await;
    let client = client(&server);
    let state = old_state();

    drop_while_polling(
        client.resume(&state, Manifest::new().with_root(ROOT)),
        &server,
        "/v2/sandboxes/sessions/sess-new",
    )
    .await;

    wait_for(&server, "POST", "/v2/sandboxes/sessions/sess-new/stop").await;
    assert_eq!(server.count("POST", "/v3/sandboxes"), 1);
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-old/stop"),
        0
    );
    assert!(no_delete(&server), "{:?}", server.routes());
    server.stop();
}
