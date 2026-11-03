//! Provider fields of the persisted session state: E2B `template`, `metadata`, `base_envs` and
//! `timeouts`, Vercel `project_id`, `team_id`, `env` and `interactive`
//! (`tests/extensions/sandbox/test_e2b.py` and `test_vercel.py`: the state-field assertions of
//! `test_vercel_create_passes_provider_options`, `test_vercel_resume_uses_client_project_and_team_fallbacks`,
//! `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`).
//!
//! The Rust state stays credential-free: an environment or metadata entry whose name or value
//! looks secret is not recorded, a URL loses userinfo and query, and an id that looks like a
//! credential is dropped. The resuming client supplies those values from its own options. A
//! payload written before the fields existed reads, and the resuming client's options stand in.
#![cfg(feature = "sandbox")]
#![allow(dead_code, unused_imports)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BPersistedOptions, E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxTimeouts,
};
use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelPersistedOptions, VercelSandboxClient, VercelSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSessionState};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

// ---------------------------------------------------------------------------------- E2B

fn e2b_options(server: &MockServer) -> E2BSandboxClientOptions {
    E2BSandboxClientOptions::new()
        .with_api_key("e2b_test")
        .with_api_base_url(&server.base_url)
        .with_sandbox_base_url(&server.base_url)
}

/// A server that creates `sb-new` and treats an existing sandbox as gone.
async fn e2b_server() -> MockServer {
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

fn e2b_creates(server: &MockServer) -> Vec<Value> {
    server
        .all("POST", "/v2/sandboxes")
        .iter()
        .map(Recorded::json)
        .collect()
}

fn custom_timeouts() -> E2BSandboxTimeouts {
    E2BSandboxTimeouts {
        snapshot_tar: Duration::from_secs(7),
        fast_op: Duration::from_millis(250),
        ..E2BSandboxTimeouts::default()
    }
}

/// An options value whose environment and metadata mix plain and credential-shaped entries.
fn configured(server: &MockServer) -> E2BSandboxClientOptions {
    e2b_options(server)
        .with_template("team-template")
        .with_metadata("owner", "alice")
        .with_metadata("api_token", "meta-secret-1")
        .with_env("MODE", "fast")
        .with_env("OPENAI_API_KEY", "sk-test-env")
        .with_env("INNOCENT_NAME", "ghp_leakedtoken")
        .with_env(
            "DATABASE_URL",
            "postgres://user:pw@db.internal/app?sslmode=require",
        )
        .with_timeouts(custom_timeouts())
}

async fn e2b_state() -> SandboxSessionState {
    let server = e2b_server().await;
    let session = E2BSandboxClient::with_options(configured(&server))
        .create(None)
        .await
        .expect("create");
    let state = session.session_state();
    server.stop();
    state
}

/// The state records the template, the timeouts and the redacted metadata and environment, and
/// no secret string appears anywhere in the serialized state.
#[tokio::test]
async fn the_e2b_state_records_template_metadata_envs_and_timeouts() {
    let state = e2b_state().await;

    let restored: SandboxSessionState =
        serde_json::from_str(&serde_json::to_string(&state).expect("serializes")).expect("reads");
    assert_eq!(restored, state);
    let persisted = E2BPersistedOptions::from_state(&restored).expect("options recorded");

    assert_eq!(persisted.template.as_deref(), Some("team-template"));
    assert_eq!(json!(persisted.metadata), json!({ "owner": "alice" }));
    assert_eq!(
        json!(persisted.envs),
        json!({
            "MODE": "fast",
            "DATABASE_URL": "postgres://db.internal/app",
        })
    );
    assert_eq!(persisted.timeouts, Some(custom_timeouts()));
    let text = serde_json::to_string(&state).expect("serializes");
    for secret in [
        "sk-test-env",
        "ghp_leakedtoken",
        "meta-secret-1",
        "user:pw",
        "sslmode",
    ] {
        assert!(!text.contains(secret), "{secret} leaked into {text}");
    }
}

/// The replacement sandbox is created from the recorded template and metadata; the resuming
/// client supplies the values the record left out and its own value wins per key.
#[tokio::test]
async fn an_e2b_replacement_sandbox_is_built_from_the_recorded_options() {
    let state = e2b_state().await;
    let server = e2b_server().await;

    let resumed = E2BSandboxClient::with_options(
        e2b_options(&server)
            .with_template("other-template")
            .with_metadata("api_token", "meta-secret-2")
            .with_env("OPENAI_API_KEY", "sk-test-resume")
            .with_env("MODE", "slow"),
    )
    .resume(&state, Manifest::new())
    .await
    .expect("resume recreates");

    let created = e2b_creates(&server);
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["templateID"], "team-template");
    assert_eq!(
        created[0]["metadata"],
        json!({ "owner": "alice", "api_token": "meta-secret-2" })
    );
    assert_eq!(
        created[0]["envVars"],
        json!({
            "MODE": "slow",
            "OPENAI_API_KEY": "sk-test-resume",
            "DATABASE_URL": "postgres://db.internal/app",
        })
    );
    // The next state is credential-free again and still names the original session's options.
    let next = resumed.session_state();
    let text = serde_json::to_string(&next).expect("serializes");
    for secret in ["sk-test-resume", "meta-secret-2"] {
        assert!(!text.contains(secret), "{secret} leaked into {text}");
    }
    let next = E2BPersistedOptions::from_state(&next).expect("recorded");
    assert_eq!(next.template.as_deref(), Some("team-template"));
    assert_eq!(next.timeouts, Some(custom_timeouts()));
    server.stop();
}

/// A template that looks like a credential is not recorded; the resuming client's own template
/// is used instead.
#[tokio::test]
async fn an_e2b_template_that_looks_secret_is_not_recorded() {
    let server = e2b_server().await;
    let session = E2BSandboxClient::with_options(
        e2b_options(&server).with_template("https://user:pw@templates.example/t"),
    )
    .create(None)
    .await
    .expect("create");
    let state = session.session_state();
    server.stop();

    let persisted = E2BPersistedOptions::from_state(&state).expect("recorded");
    assert_eq!(persisted.template, None);
    let text = serde_json::to_string(&state).expect("serializes");
    assert!(!text.contains("user:pw"), "{text}");

    let server = e2b_server().await;
    E2BSandboxClient::with_options(e2b_options(&server).with_template("client-template"))
        .resume(&state, Manifest::new())
        .await
        .expect("resume");
    assert_eq!(e2b_creates(&server)[0]["templateID"], "client-template");
    server.stop();
}

/// A record that predates the new fields reads with them unset, and a payload without any record
/// falls back to the client's options.
#[tokio::test]
async fn an_older_e2b_record_reads_with_the_new_fields_unset() {
    let restored: E2BPersistedOptions =
        serde_json::from_value(json!({ "mcp": { "exa": { "region": "us" } } })).expect("reads");
    assert_eq!(restored.template, None);
    assert!(restored.metadata.is_empty());
    assert!(restored.envs.is_empty());
    assert_eq!(restored.timeouts, None);

    let mut payload = serde_json::to_value(e2b_state().await).expect("serializes");
    payload["provider_metadata"]["provider_options"] = json!({ "mcp": {} });
    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");
    let server = e2b_server().await;
    E2BSandboxClient::with_options(
        e2b_options(&server)
            .with_template("client-template")
            .with_env("MODE", "client"),
    )
    .resume(&state, Manifest::new())
    .await
    .expect("resume");
    let created = e2b_creates(&server);
    assert_eq!(created[0]["templateID"], "client-template");
    assert_eq!(created[0]["envVars"], json!({ "MODE": "client" }));
    server.stop();
}

// -------------------------------------------------------------------------------- Vercel

const ROOT: &str = "/vercel/sandbox";

fn vercel_body(session_id: &str) -> Value {
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

/// A server whose created sandboxes run, and where an existing session lookup answers `existing`
/// (404 when `false`).
async fn vercel_server(existing: bool) -> MockServer {
    vercel_server_creating(existing, if existing { "sess-old" } else { "sess-new" }).await
}

async fn vercel_server_creating(existing: bool, created: &'static str) -> MockServer {
    MockServer::start(move |request: &Recorded| {
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v3/sandboxes") => MockResponse::json(vercel_body(created)),
            ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
                if existing || path.ends_with(created) {
                    let body = vercel_body(path.rsplit('/').next().unwrap_or("sess-1"));
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

fn vercel_options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

fn manifest() -> Manifest {
    Manifest::new().with_root(ROOT)
}

fn vercel_creates(server: &MockServer) -> Vec<Value> {
    server
        .all("POST", "/v3/sandboxes")
        .iter()
        .map(Recorded::json)
        .collect()
}

async fn vercel_state() -> SandboxSessionState {
    let server = vercel_server(true).await;
    let session = VercelSandboxClient::with_options(
        vercel_options(&server)
            .with_project_id("prj_abc123")
            .with_team_id("team_xyz")
            .with_interactive(true)
            .with_env("FLAG", "options")
            .with_env("HELLO", "world")
            .with_env("VERCEL_API_TOKEN", "vercel-env-secret")
            .with_env("ANOTHER", "xoxb-slack-value"),
    )
    .create(Some(manifest()))
    .await
    .expect("create");
    let state = session.session_state();
    server.stop();
    state
}

/// `test_vercel_create_passes_provider_options`, state half: the state records the project, the
/// team, the interactive flag and the redacted environment, with no secret anywhere.
#[tokio::test]
async fn the_vercel_state_records_scope_env_and_interactive() {
    let state = vercel_state().await;

    let restored: SandboxSessionState =
        serde_json::from_str(&serde_json::to_string(&state).expect("serializes")).expect("reads");
    assert_eq!(restored, state);
    let persisted = VercelPersistedOptions::from_state(&restored).expect("recorded");

    assert_eq!(persisted.project_id.as_deref(), Some("prj_abc123"));
    assert_eq!(persisted.team_id.as_deref(), Some("team_xyz"));
    assert!(persisted.interactive);
    assert_eq!(
        json!(persisted.env),
        json!({ "FLAG": "options", "HELLO": "world" })
    );
    let text = serde_json::to_string(&state).expect("serializes");
    for secret in ["vercel_test", "vercel-env-secret", "xoxb-slack-value"] {
        assert!(!text.contains(secret), "{secret} leaked into {text}");
    }
}

/// A replacement sandbox is created in the recorded project and team with the recorded
/// environment, and the resuming client's values win per key.
#[tokio::test]
async fn a_vercel_replacement_sandbox_is_built_from_the_recorded_scope() {
    let state = vercel_state().await;
    let server = vercel_server(false).await;

    let resumed = VercelSandboxClient::with_options(
        vercel_options(&server)
            .with_project_id("client-project")
            .with_team_id("client-team")
            .with_env("FLAG", "client")
            .with_env("VERCEL_API_TOKEN", "vercel-env-secret-2"),
    )
    .resume(&state, manifest())
    .await
    .expect("resume recreates");

    let created = vercel_creates(&server);
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["projectId"], "prj_abc123");
    assert_eq!(
        created[0]["env"],
        json!({
            "FLAG": "client",
            "HELLO": "world",
            "VERCEL_API_TOKEN": "vercel-env-secret-2",
        })
    );
    assert_eq!(
        server.all("POST", "/v3/sandboxes")[0]
            .query_param("teamId")
            .as_deref(),
        Some("team_xyz"),
        "the recorded team scopes the create request"
    );
    let text = serde_json::to_string(&resumed.session_state()).expect("serializes");
    assert!(!text.contains("vercel-env-secret-2"), "{text}");
    server.stop();
}

/// `test_vercel_resume_uses_client_project_and_team_fallbacks`: a state that names no project or
/// team falls back to the resuming client's, and the next state records them.
#[tokio::test]
async fn the_resuming_client_supplies_a_missing_project_and_team() {
    let creator = vercel_server(true).await;
    let original = VercelSandboxClient::with_options(vercel_options(&creator))
        .create(Some(manifest()))
        .await
        .expect("create");
    let mut payload = serde_json::to_value(original.session_state()).expect("serializes");
    payload["provider_metadata"]
        .as_object_mut()
        .expect("object")
        .remove("provider_options");
    creator.stop();
    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");

    let server = vercel_server(false).await;
    let resumed = VercelSandboxClient::with_options(
        vercel_options(&server)
            .with_project_id("client-project")
            .with_team_id("client-team"),
    )
    .resume(&state, manifest())
    .await
    .expect("resume");

    assert_eq!(vercel_creates(&server)[0]["projectId"], "client-project");
    let persisted = VercelPersistedOptions::from_state(&resumed.session_state()).expect("recorded");
    assert_eq!(persisted.project_id.as_deref(), Some("client-project"));
    assert_eq!(persisted.team_id.as_deref(), Some("client-team"));
    server.stop();
}

/// A project or team id that looks like a credential is kept out of the state.
#[tokio::test]
async fn a_vercel_id_that_looks_secret_is_not_recorded() {
    let server = vercel_server(true).await;
    let session = VercelSandboxClient::with_options(
        vercel_options(&server)
            .with_project_id("sk-live-project-token")
            .with_team_id("https://user:pw@vercel.example/team"),
    )
    .create(Some(manifest()))
    .await
    .expect("create");
    let state = session.session_state();
    server.stop();

    let persisted = VercelPersistedOptions::from_state(&state).expect("recorded");
    assert_eq!(persisted.project_id, None);
    assert_eq!(persisted.team_id, None);
    let text = serde_json::to_string(&state).expect("serializes");
    for secret in ["sk-live-project-token", "user:pw"] {
        assert!(!text.contains(secret), "{secret} leaked into {text}");
    }
}

/// Every new field has a serde default.
#[test]
fn missing_vercel_fields_read_as_unset() {
    let restored: VercelPersistedOptions =
        serde_json::from_value(json!({ "runtime": "node22" })).expect("a partial record reads");

    assert_eq!(restored.project_id, None);
    assert_eq!(restored.team_id, None);
    assert!(restored.env.is_empty());
    assert!(!restored.interactive);
}
