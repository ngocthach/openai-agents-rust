//! E2B in the persisted session state and the PTY start timeout: `tests/extensions/sandbox/
//! test_e2b.py` `test_e2b_deserialize_session_state_defaults_missing_mcp`,
//! `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`,
//! `test_e2b_hydrate_workspace_native_snapshot_recreates_from_snapshot_id` and the
//! `test_e2b_pty_start_*` cases that pass `timeout=`.
//!
//! The MCP config is written into the state the session hands out and is what a replacement
//! sandbox created on resume starts. The per-call PTY timeout is the process lifetime envd is
//! told to enforce, and it bounds a start request that stalls.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BPersistedOptions, E2BSandboxClient, E2BSandboxClientOptions, E2BTimeoutAction,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecShell, Manifest, PtyExecRequest, PtyWriteRequest, SandboxClient,
    SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

fn options(server: &MockServer) -> E2BSandboxClientOptions {
    E2BSandboxClientOptions::new()
        .with_api_key("e2b_test")
        .with_api_base_url(&server.base_url)
        .with_sandbox_base_url(&server.base_url)
}

const SECRET_URL: &str = "https://user:pw@mcp.example.com/exa?token=url-token";

/// An `exa` server with a secret option, a URL carrying userinfo and a query, and a plain option.
fn exa() -> E2BSandboxClientOptions {
    E2BSandboxClientOptions::new().with_mcp_server(
        "exa",
        [("apiKey", "exa-key"), ("url", SECRET_URL), ("region", "us")],
    )
}

/// What the state keeps of [`exa`].
fn exa_shape() -> Value {
    json!({ "exa": { "url": "https://mcp.example.com/exa", "region": "us" } })
}

/// What a resuming client that supplies only the secret ends up starting.
fn exa_resumed() -> Value {
    json!({ "exa": {
        "apiKey": "exa-key", "url": "https://mcp.example.com/exa", "region": "us",
    } })
}

fn resuming_client(server: &MockServer) -> E2BSandboxClient {
    E2BSandboxClient::with_options(options(server).with_mcp_server("exa", [("apiKey", "exa-key")]))
}

/// A server that creates `sb-new` and treats an existing sandbox as gone.
async fn creating_server() -> MockServer {
    support::start_resolving(support::Resolver::E2b, |request: &Recorded| {
        match request.path.as_str() {
            "/v2/sandboxes" => {
                MockResponse::json(json!({ "sandboxID": "sb-new", "envdAccessToken": "t" }))
            }
            path if path.ends_with("/connect") => MockResponse::empty(404),
            _ => MockResponse::json(json!({})),
        }
    })
    .await
}

fn create_bodies(server: &MockServer) -> Vec<Value> {
    server
        .all("POST", "/v2/sandboxes")
        .iter()
        .map(Recorded::json)
        .collect()
}

/// A state whose sandbox is gone, written by a session that started `exa`.
async fn state_with_mcp() -> SandboxSessionState {
    let server = creating_server().await;
    let mut creating = options(&server);
    creating.mcp = exa().mcp;
    let session = E2BSandboxClient::with_options(creating)
        .create(None)
        .await
        .expect("create");
    let state = session.session_state();
    server.stop();
    state
}

/// The state records the non-secret shape of the MCP config and round-trips through JSON; no
/// secret value (API key, URL userinfo or query) appears anywhere in the serialized state.
#[tokio::test]
async fn the_session_state_records_the_mcp_config() {
    let state = state_with_mcp().await;

    let restored: SandboxSessionState =
        serde_json::from_str(&serde_json::to_string(&state).expect("serializes"))
            .expect("deserializes");

    assert_eq!(restored, state);
    let persisted = E2BPersistedOptions::from_state(&restored).expect("options recorded");
    assert_eq!(json!(persisted.mcp), exa_shape());
    let text = serde_json::to_string(&state).expect("serializes");
    for secret in ["exa-key", "url-token", "user:pw", "pw@"] {
        assert!(!text.contains(secret), "{secret} leaked into {text}");
    }
    assert_eq!(
        serde_json::to_value(&state).expect("serializes")["provider_metadata"]["provider_options"]
            ["mcp"],
        exa_shape()
    );
}

/// `test_e2b_deserialize_session_state_defaults_missing_mcp`: a record without the field reads
/// as no MCP servers, and a state without the record at all still deserializes.
#[tokio::test]
async fn a_missing_mcp_field_reads_as_none() {
    let restored: E2BPersistedOptions = serde_json::from_value(json!({})).expect("deserializes");
    assert!(restored.mcp.is_empty());

    let mut payload = serde_json::to_value(state_with_mcp().await).expect("serializes");
    payload["provider_metadata"]
        .as_object_mut()
        .expect("object")
        .remove("provider_options");
    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");
    assert_eq!(E2BPersistedOptions::from_state(&state), None);
}

/// The persisted record names its MCP servers in `Debug` without printing what they are given.
#[test]
fn debug_output_keeps_the_mcp_secrets_out() {
    let persisted = E2BPersistedOptions::from_options(&exa());

    let printed = format!("{persisted:?}");

    assert!(printed.contains("exa"), "{printed}");
    assert!(!printed.contains("exa-key"), "{printed}");
    assert!(!printed.contains("mcp.example.com"), "{printed}");
}

/// `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`: the replacement
/// sandbox starts the recorded servers, with the secrets supplied by the resuming client (Python
/// reads them from the state; this state is credential-free by design).
#[tokio::test]
async fn a_replacement_sandbox_starts_the_recorded_mcp_servers() {
    let state = state_with_mcp().await;
    let server = creating_server().await;

    let resumed = resuming_client(&server)
        .resume(&state, Manifest::new())
        .await
        .expect("resume recreates");

    let created = create_bodies(&server);
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["mcp"], exa_resumed());
    let text = serde_json::to_string(&resumed.session_state()).expect("serializes");
    assert!(!text.contains("exa-key"), "{text}");
    assert_eq!(
        E2BPersistedOptions::from_state(&resumed.session_state()),
        E2BPersistedOptions::from_state(&state)
    );
    server.stop();
}

/// A resuming client that supplies no secrets starts the recorded shape without them.
#[tokio::test]
async fn a_client_without_secrets_starts_only_the_recorded_shape() {
    let state = state_with_mcp().await;
    let server = creating_server().await;

    E2BSandboxClient::with_options(options(&server))
        .resume(&state, Manifest::new())
        .await
        .expect("resume recreates");

    assert_eq!(create_bodies(&server)[0]["mcp"], exa_shape());
    server.stop();
}

/// URL userinfo, query and fragment are dropped, and option names that look secret are not kept.
#[test]
fn the_persisted_shape_drops_secrets() {
    let mut options = E2BSandboxClientOptions::new();
    options.mcp = exa().mcp;
    options = options.with_mcp_server(
        "other",
        [
            ("Authorization", "Bearer abc"),
            ("projectId", "project-id"),
            ("endpoint", "https://host:8443/p#frag"),
        ],
    );

    let persisted = E2BPersistedOptions::from_options(&options);

    assert_eq!(
        json!(persisted.mcp),
        json!({
            "exa": { "url": "https://mcp.example.com/exa", "region": "us" },
            "other": { "projectId": "project-id", "endpoint": "https://host:8443/p" },
        })
    );
}

/// A state written before the options were recorded falls back to the resuming client's MCP
/// config, as before.
#[tokio::test]
async fn an_older_payload_falls_back_to_the_client_mcp_config() {
    let mut payload = serde_json::to_value(state_with_mcp().await).expect("serializes");
    payload["provider_metadata"]
        .as_object_mut()
        .expect("object")
        .remove("provider_options");
    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");
    let server = creating_server().await;

    E2BSandboxClient::with_options(
        options(&server)
            .with_mcp_server("browserbase", [("projectId", "project-id")])
            .with_on_timeout(E2BTimeoutAction::Kill),
    )
    .resume(&state, Manifest::new())
    .await
    .expect("resume");

    assert_eq!(
        create_bodies(&server)[0]["mcp"],
        json!({ "browserbase": { "projectId": "project-id" } })
    );
    server.stop();
}

/// `test_e2b_hydrate_workspace_native_snapshot_recreates_from_snapshot_id`: a resumed session
/// that restores a native snapshot starts the MCP servers its state recorded (secrets from the
/// client), with the snapshot id as the template.
#[tokio::test]
async fn a_native_snapshot_restore_starts_the_recorded_mcp_servers() {
    use openai_agents_sdk::extensions::sandbox::e2b::WorkspacePersistence;

    let state = state_with_mcp().await;
    let server =
        support::start_resolving(support::Resolver::E2b, |request: &Recorded| {
            match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(json!({ "sandboxID": "sb-from-snapshot" })),
                path if path.ends_with("/connect") => {
                    MockResponse::json(json!({ "sandboxID": "sb-persisted" }))
                }
                path if path.ends_with("/snapshots") => {
                    MockResponse::json(json!({ "snapshotID": "snap-123" }))
                }
                _ => MockResponse::json(json!({})),
            }
        })
        .await;
    let session = E2BSandboxClient::with_options(
        options(&server)
            .with_mcp_server("exa", [("apiKey", "exa-key")])
            .with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .resume(&state, Manifest::new())
    .await
    .expect("resume reattaches");
    let reference = session.persist_workspace().await.expect("persist");

    session
        .hydrate_workspace(&reference)
        .await
        .expect("hydrate");

    let created = create_bodies(&server);
    assert_eq!(
        created.len(),
        1,
        "only the snapshot replacement was created"
    );
    assert_eq!(created[0]["templateID"], "snap-123");
    assert_eq!(created[0]["mcp"], exa_resumed());
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// PTY start timeout
// ---------------------------------------------------------------------------------------------

fn pty_request(command: &str) -> PtyExecRequest {
    PtyExecRequest::new(command)
        .with_shell(ExecShell::None)
        .with_tty(true)
        .with_yield_time_ms(100)
}

fn connect_frame(flags: u8, value: Value) -> Vec<u8> {
    let payload = serde_json::to_vec(&value).expect("serializable");
    let mut frame = vec![flags];
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn started() -> MockResponse {
    let mut body = connect_frame(0, json!({ "event": { "start": { "pid": 7 } } }));
    body.extend(connect_frame(
        0,
        json!({ "event": { "end": { "exited": true, "exitCode": 0 } } }),
    ));
    body.extend(connect_frame(2, json!({})));
    MockResponse::streamed("application/connect+json", vec![(body, 0)])
}

/// A server that answers `process.Process/Start` after `delay`.
async fn start_after(delay: Duration) -> MockServer {
    MockServer::start_raw(move |request: Recorded, mut stream| async move {
        let response = match request.path.as_str() {
            "/v2/sandboxes" => {
                MockResponse::json(json!({ "sandboxID": "sb-1", "envdAccessToken": "t" }))
            }
            "/process.Process/Start" => {
                tokio::time::sleep(delay).await;
                started()
            }
            _ => MockResponse::json(json!({})),
        };
        let _ = support::write_response(&mut stream, response).await;
    })
    .await
}

fn start_timeout_header(server: &MockServer) -> String {
    server
        .find("POST", "/process.Process/Start")
        .header("connect-timeout-ms")
        .expect("a Connect timeout")
        .to_string()
}

/// The per-call timeout is the lifetime envd is told to enforce for the process; without one the
/// unbounded sentinel (24 hours) goes out, as in Python's `_coerce_exec_timeout`.
#[tokio::test]
async fn the_pty_timeout_is_the_process_lifetime_envd_enforces() {
    for (timeout, expected) in [
        (Some(Duration::from_secs(2)), "2000"),
        (Some(Duration::ZERO), "1000"),
        (None, "86400000"),
    ] {
        let server = start_after(Duration::ZERO).await;
        let session = E2BSandboxClient::with_options(options(&server))
            .create(None)
            .await
            .expect("create");

        session
            .pty_exec_start(pty_request("python3").with_timeout(timeout))
            .await
            .expect("start");

        assert_eq!(start_timeout_header(&server), expected, "{timeout:?}");
        server.stop();
    }
}

/// `test_e2b_pty_start_maps_timeout_failures` and
/// `test_e2b_pty_start_maps_httpcore_read_timeout_to_timeout_error`: a start that stalls past
/// the call's timeout is an `ExecTimeout`, and nothing stays registered.
#[tokio::test]
async fn a_stalled_pty_start_times_out_and_registers_no_session() {
    let server = start_after(Duration::from_secs(5)).await;
    let session = E2BSandboxClient::with_options(options(&server))
        .create(None)
        .await
        .expect("create");
    let began = std::time::Instant::now();

    let error = session
        .pty_exec_start(pty_request("python3").with_timeout(Some(Duration::from_millis(150))))
        .await
        .expect_err("the start stalled");

    assert_eq!(error.code, ErrorCode::ExecTimeout, "{error:?}");
    assert!(began.elapsed() < Duration::from_secs(3));
    let after = session
        .pty_write_stdin(PtyWriteRequest::new(1_000, ""))
        .await
        .expect_err("nothing registered");
    assert_eq!(after.code, ErrorCode::PtySessionNotFound);
    server.stop();
}
