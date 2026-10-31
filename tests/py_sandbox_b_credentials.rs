//! Client-option credentials stay out of every surface except an explicit serialization.
//!
//! Python's `model_dump` includes `api_key`, so a caller who serializes the options on purpose
//! gets the key. Everything else must not: `Debug` output, error text, log records, persisted
//! session state and persisted `RunState` JSON.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::extensions::sandbox::SandboxClientOptionsRecord;
use openai_agents_sdk::extensions::sandbox::blaxel::BlaxelSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareSandboxClient, CloudflareSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::daytona::DaytonaSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::e2b::E2BSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::modal::ModalSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::vercel::VercelSandboxClientOptions;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::sandbox::sandboxes::docker::DockerSandboxClientOptions;
use openai_agents_sdk::sandbox::{ExecRequest, SandboxClient};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

const KEY: &str = "sk-cred-marker-9f3a";

fn cloudflare_options(base: &str) -> CloudflareSandboxClientOptions {
    CloudflareSandboxClientOptions::new(base).with_api_key(KEY)
}

/// Every options type, each holding `KEY` in every field that can carry a credential.
fn credentialed_options() -> Vec<(&'static str, String)> {
    let mut blaxel = BlaxelSandboxClientOptions {
        api_key: Some(KEY.to_string()),
        ..BlaxelSandboxClientOptions::default()
    };
    blaxel.env_vars.insert("TOKEN".into(), KEY.to_string());
    let mut daytona = DaytonaSandboxClientOptions::new();
    daytona.api_key = Some(KEY.to_string());
    daytona.env_vars.insert("TOKEN".into(), KEY.to_string());
    let mut vercel = VercelSandboxClientOptions::new();
    vercel.token = Some(KEY.to_string());
    vercel.env.insert("TOKEN".into(), KEY.to_string());
    let mut e2b = E2BSandboxClientOptions::new();
    e2b.api_key = Some(KEY.to_string());
    e2b.envs.insert("TOKEN".into(), KEY.to_string());
    let mut docker = DockerSandboxClientOptions::new("python:3.12");
    docker.extra_create_args = vec!["-e".into(), format!("TOKEN={KEY}")];
    let mut modal = ModalSandboxClientOptions::new("agents-sdk");
    modal.helper_command = Some(vec![
        "/usr/bin/env".into(),
        format!("MODAL_TOKEN_SECRET={KEY}"),
        "python3".into(),
    ]);
    let cloudflare = cloudflare_options("https://worker.example");
    vec![
        ("cloudflare", format!("{cloudflare:?}")),
        ("cloudflare pretty", format!("{cloudflare:#?}")),
        ("blaxel", format!("{blaxel:?}")),
        ("daytona", format!("{daytona:?}")),
        ("vercel", format!("{vercel:?}")),
        ("e2b", format!("{e2b:?}")),
        ("docker", format!("{docker:?}")),
        ("modal", format!("{modal:?}")),
        (
            "record",
            format!(
                "{:?}",
                SandboxClientOptionsRecord::Cloudflare(cloudflare.clone())
            ),
        ),
        (
            "client",
            format!("{:?}", CloudflareSandboxClient::new(cloudflare)),
        ),
    ]
}

/// Surface: `Debug`.
#[test]
fn debug_output_never_shows_a_credential() {
    for (name, debug) in credentialed_options() {
        assert!(!debug.contains(KEY), "{name}: {debug}");
    }
}

/// A caller who serializes the options on purpose gets Python's payload, key included.
#[test]
fn explicit_serialization_keeps_the_python_compatible_key() {
    let record = SandboxClientOptionsRecord::Cloudflare(cloudflare_options("https://w.example"));
    let payload: Value = serde_json::to_value(&record).unwrap();
    assert_eq!(payload["type"], "cloudflare");
    assert_eq!(payload["api_key"], KEY);
}

fn failing_worker() -> impl std::future::Future<Output = MockServer> {
    MockServer::start(|request: &Recorded| match request.path.as_str() {
        "/v1/sandbox" => MockResponse::json(json!({ "id": "sbx-1" })),
        _ => MockResponse::status(500, json!({ "message": "boom" })),
    })
}

/// Surface: error text.
#[tokio::test]
async fn errors_never_carry_a_credential() {
    // Creation refused by the worker.
    let refusing =
        MockServer::start(|_| MockResponse::status(401, json!({"message": "denied"}))).await;
    let error = CloudflareSandboxClient::new(cloudflare_options(&refusing.base_url))
        .create(None)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(!error.to_string().contains(KEY), "{error}");
    assert!(!format!("{error:?}").contains(KEY), "{error:?}");
    refusing.stop();

    // A command that fails after a session exists.
    let server = failing_worker().await;
    let session = CloudflareSandboxClient::new(cloudflare_options(&server.base_url))
        .create(None)
        .await
        .expect("create");
    let error = session
        .exec(ExecRequest::argv(["echo", "hi"]))
        .await
        .unwrap_err();
    assert!(!error.to_string().contains(KEY), "{error}");
    assert!(!format!("{error:?}").contains(KEY), "{error:?}");
    server.stop();

    // An unreachable worker.
    let error = CloudflareSandboxClient::new(cloudflare_options("http://127.0.0.1:1"))
        .create(None)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(!error.to_string().contains(KEY), "{error}");
    assert!(!format!("{error:?}").contains(KEY), "{error:?}");
}

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

/// Surface: log records, at every level.
#[tokio::test]
async fn log_records_never_carry_a_credential() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });

    let server = failing_worker().await;
    let session = CloudflareSandboxClient::new(cloudflare_options(&server.base_url))
        .create(None)
        .await
        .expect("create");
    let _ = session.exec(ExecRequest::argv(["echo", "hi"])).await;
    session.stop().await.ok();
    server.stop();
    let refusing =
        MockServer::start(|_| MockResponse::status(401, json!({"message": "denied"}))).await;
    let _ = CloudflareSandboxClient::new(cloudflare_options(&refusing.base_url))
        .create(None)
        .await;
    refusing.stop();

    let logged = RECORDS.lock().unwrap().join("\n");
    assert!(!logged.contains(KEY), "{logged}");
}

/// Surface: persisted session state.
#[tokio::test]
async fn persisted_session_state_never_carries_a_credential() {
    let server = failing_worker().await;
    let session: Arc<dyn openai_agents_sdk::sandbox::SandboxSession> =
        CloudflareSandboxClient::new(cloudflare_options(&server.base_url))
            .create(None)
            .await
            .expect("create");

    let state = session.session_state();

    assert!(!state.to_json_string().unwrap().contains(KEY));
    assert!(
        !state
            .sanitized_for_persistence()
            .to_json_string()
            .unwrap()
            .contains(KEY)
    );
    assert!(!format!("{state:?}").contains(KEY));
    server.stop();
}

/// Surface: persisted `RunState` JSON. The state never holds client options, and a sandbox payload
/// that does hold mount credentials is sanitized on the way out.
#[test]
fn persisted_run_state_never_carries_a_credential() {
    let mut state = RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    };
    assert!(!state.to_json().unwrap().to_string().contains(KEY));

    let session_state = json!({
        "type": "cloudflare",
        "worker_url": "https://worker.example",
        "sandbox_id": "sbx-1",
        "manifest": {
            "version": 1, "root": "/workspace",
            "entries": {"data": {
                "type": "s3_mount", "bucket": "bucket",
                "access_key_id": KEY, "secret_access_key": KEY, "session_token": KEY,
                "mount_strategy": {"type": "in_container", "pattern": {"type": "rclone"}},
            }},
            "environment": {"value": {}},
        },
        "snapshot": {"type": "noop", "id": "synthetic"},
    });
    state.sandbox = Some(json!({
        "backend_id": "cloudflare",
        "current_agent_key": "TestAgent",
        "current_agent_name": "TestAgent",
        "session_state": session_state,
        "sessions_by_agent": {"TestAgent": {"agent_name": "TestAgent", "session_state": session_state}},
    }));

    let serialized = state.to_json().unwrap().to_string();

    assert!(!serialized.contains(KEY), "{serialized}");
}
