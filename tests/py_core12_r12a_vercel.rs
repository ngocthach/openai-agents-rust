//! Vercel cases of `tests/extensions/sandbox/test_vercel.py` and
//! `test_vercel_provider_contract.py` that the earlier ports left as Python-only: a mount failure
//! whose output echoes a transformed secret, a hydrate archive with an absolute symlink
//! destination, and a resume whose status polling fails.
//!
//! The Vercel client speaks the REST API, so each case runs against a local mock that records the
//! requests and answers every `cmd` request from a script.
#![cfg(feature = "sandbox")]
#![allow(dead_code, unused_imports)]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelCloudBucketMountStrategy, VercelSandboxClient, VercelSandboxClientOptions,
    WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

/// One scripted answer to a command.
#[derive(Clone)]
struct Answer {
    stdout: String,
    stderr: String,
    exit_code: i64,
}

fn ok(stdout: &str) -> Answer {
    Answer {
        stdout: stdout.to_string(),
        stderr: String::new(),
        exit_code: 0,
    }
}

fn failed(stderr: &str, exit_code: i64) -> Answer {
    Answer {
        stdout: String::new(),
        stderr: stderr.to_string(),
        exit_code,
    }
}

#[derive(Default)]
struct Script {
    /// Queued answers per command, used before the defaults.
    queued: Mutex<HashMap<String, VecDeque<Answer>>>,
    /// How many `fs/write` calls fail with a 500 before one succeeds.
    write_failures: Mutex<usize>,
    /// Whether `fs/read` reports the file missing.
    read_missing: Mutex<bool>,
    /// The archive `fs/read` serves.
    archive: Mutex<Vec<u8>>,
}

impl Script {
    fn queue(&self, command: &str, answers: Vec<Answer>) {
        self.queued
            .lock()
            .expect("poisoned")
            .entry(command.to_string())
            .or_default()
            .extend(answers);
    }

    fn next(&self, command: &str) -> Option<Answer> {
        self.queued
            .lock()
            .expect("poisoned")
            .get_mut(command)
            .and_then(VecDeque::pop_front)
    }
}

fn session_body(status: &str) -> Value {
    json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": "sess-1",
        },
        "session": {
            "id": "sess-1",
            "memory": 2048,
            "vcpus": 2,
            "region": "iad1",
            "timeout": 270000,
            "status": status,
            "requestedAt": 1,
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": "/vercel/sandbox",
        },
        "routes": [],
    })
}

fn command_response(answer: &Answer) -> MockResponse {
    MockResponse::ndjson(&[
        json!({ "stream": "stdout", "data": answer.stdout }),
        json!({ "stream": "stderr", "data": answer.stderr }),
        json!({
            "command": {
                "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                "sessionId": "sess-1", "exitCode": answer.exit_code, "startedAt": 1,
            },
        }),
    ])
}

fn argv_of(request: &Recorded) -> Vec<String> {
    let body = request.json();
    let mut argv = vec![body["command"].as_str().unwrap_or_default().to_string()];
    argv.extend(
        body["args"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|arg| arg.as_str().map(str::to_string)),
    );
    argv
}

/// The default answer for a command the mount flow runs.
fn default_answer(argv: &[String]) -> Answer {
    let fingerprint_helper =
        openai_agents_sdk::sandbox::runtime_helpers::WORKSPACE_FINGERPRINT_HELPER
            .install_path
            .clone();
    match argv.first().map(String::as_str).unwrap_or_default() {
        command if command == fingerprint_helper && argv.len() > 2 => {
            ok(r#"{"fingerprint":"same","version":"workspace_tar_sha256_v1"}"#)
        }
        "/usr/bin/rpm" => ok("1.21.0"),
        "/usr/bin/realpath" => ok(argv.last().map(String::as_str).unwrap_or_default()),
        "/usr/bin/find" => ok(""),
        "/usr/bin/id" => ok(if argv.contains(&"-u".to_string()) {
            "1000"
        } else {
            "1001"
        }),
        "/usr/bin/findmnt" => ok("mountpoint-s3"),
        // Anything else, including the in-sandbox path resolver, answers with something.
        _ => ok("ok"),
    }
}

fn handler(script: Arc<Script>) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    move |request: &Recorded| match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => MockResponse::json(session_body("running")),
        ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
            let body = session_body("running");
            MockResponse::json(json!({ "session": body["session"], "routes": [] }))
        }
        ("POST", path) if path.ends_with("/fs/mkdir") => MockResponse::json(json!({})),
        ("POST", path) if path.ends_with("/fs/write") => {
            let mut failures = script.write_failures.lock().expect("poisoned");
            if *failures > 0 {
                *failures -= 1;
                return MockResponse::error(500, "write failed");
            }
            MockResponse::json(json!({}))
        }
        ("POST", path) if path.ends_with("/fs/read") => {
            if *script.read_missing.lock().expect("poisoned") {
                return MockResponse::error(404, "missing");
            }
            MockResponse::bytes(script.archive.lock().expect("poisoned").clone())
        }
        ("POST", path) if path.ends_with("/snapshot") => {
            MockResponse::json(json!({ "snapshot": { "id": "snap-1" } }))
        }
        ("POST", path) if path.ends_with("/stop") => {
            let body = session_body("stopped");
            MockResponse::json(json!({ "session": body["session"] }))
        }
        ("POST", path) if path.ends_with("/cmd") => {
            let argv = argv_of(request);
            let answer = script
                .next(argv.first().map(String::as_str).unwrap_or_default())
                .unwrap_or_else(|| default_answer(&argv));
            command_response(&answer)
        }
        _ => MockResponse::error(404, "unexpected"),
    }
}

fn s3_strategy() -> VercelCloudBucketMountStrategy {
    VercelCloudBucketMountStrategy::new("test-bucket").with_region("us-west-2")
}

fn credentialed_strategy() -> VercelCloudBucketMountStrategy {
    s3_strategy()
        .with_credentials("test-access-key", "test-secret-key")
        .with_session_token("test-session-token")
}

const ROOT: &str = "/vercel/sandbox";

fn mount_manifest(strategy: VercelCloudBucketMountStrategy) -> Manifest {
    Manifest::new()
        .with_root(ROOT)
        .with_entry("remote", Entry::mount(strategy.mount()))
}

fn acknowledged(strategy: VercelCloudBucketMountStrategy) -> Manifest {
    mount_manifest(strategy)
        .with_in_container_mount_credential_exposure_acknowledged(["/vercel/sandbox/remote"])
}

fn client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(options(server))
}

fn options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

fn commands_named(server: &MockServer, command: &str) -> Vec<Recorded> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.path.ends_with("/cmd") && argv_of(request)[0] == command)
        .collect()
}

/// The command of every `cmd` request, in order, restricted to the mount-lifecycle ones.
fn lifecycle(server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter_map(|request| {
            if request.path.ends_with("/cmd") {
                let command = argv_of(&request)[0].clone();
                if ["/usr/bin/mount-s3", "/usr/bin/findmnt", "/usr/bin/umount"]
                    .contains(&command.as_str())
                    || command == "sh"
                {
                    return Some(command.rsplit('/').next().unwrap_or_default().to_string());
                }
            } else if request.path.ends_with("/snapshot") {
                return Some("snapshot".to_string());
            } else if request.path.ends_with("/stop") {
                return Some("stop".to_string());
            }
            None
        })
        .collect()
}

fn create_count(server: &MockServer) -> usize {
    server.count("POST", "/v3/sandboxes")
}

// ---------------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------------

/// `test_vercel_s3_mount_failure_discards_transformed_stderr`: a failing `mount-s3` whose stderr
/// echoes the URL-encoded secret fails the start with the mount error, nothing of the provider
/// text reaches the error, and the sandbox is stopped.
#[tokio::test]
async fn a_mount_failure_never_repeats_the_transformed_secret() {
    let transformed_secret = "test%2Dsecret%2Dkey";
    let script = Arc::new(Script::default());
    script.queue(
        "/usr/bin/mount-s3",
        vec![failed(&format!("provider echoed {transformed_secret}"), 1)],
    );
    let server = MockServer::start(handler(script)).await;
    let session = client(&server)
        .create(Some(acknowledged(credentialed_strategy())))
        .await
        .expect("create");

    let error = session.start().await.expect_err("the mount failed");

    assert_eq!(error.code, ErrorCode::MountFailed, "{error:?}");
    assert!(!error.message.contains(transformed_secret), "{error:?}");
    assert!(!error.message.contains("test-secret-key"), "{error:?}");
    assert!(error.context.is_empty(), "{:?}", error.context);
    let stops = server
        .requests()
        .iter()
        .filter(|request| request.path.ends_with("/stop"))
        .count();
    assert_eq!(stops, 1);
    server.stop();
}

/// `test_vercel_hydrate_workspace_rejects_external_symlink_target_before_upload`: an archive
/// with an absolute symlink destination is refused before any byte reaches the sandbox.
#[tokio::test]
async fn hydrate_rejects_an_absolute_symlink_destination_before_upload() {
    let server = MockServer::start(handler(Arc::new(Script::default()))).await;
    let session = client(&server).create(None).await.expect("create");
    session.start().await.expect("start");
    let writes = server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write");

    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o777);
    builder
        .append_link(&mut header, "link", "/etc/passwd")
        .expect("append");
    let archive = builder.into_inner().expect("finish");

    let error = session
        .hydrate_workspace(&archive)
        .await
        .expect_err("an absolute symlink destination");

    assert!(error.message.contains("absolute symlink"), "{error:?}");
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write"),
        writes,
        "nothing was uploaded"
    );
    server.stop();
}

fn replacement_body() -> Value {
    let mut body = session_body("running");
    body["sandbox"]["currentSessionId"] = json!("sess-2");
    body["session"]["id"] = json!("sess-2");
    body
}

/// `test_installed_provider_closes_unadopted_reconnect_client` (provider failure): a resume that
/// finds the session still pending and then cannot poll it creates a replacement sandbox.
#[tokio::test]
async fn a_failing_status_poll_replaces_the_pending_session() {
    let polls = support::Counter::new();
    let inner = handler(Arc::new(Script::default()));
    let counted = polls.clone();
    let server = MockServer::start(move |request: &Recorded| {
        if request.method == "GET" && request.path == "/v2/sandboxes/sessions/sess-1" {
            return if counted.next("get") == 0 {
                let body = session_body("pending");
                MockResponse::json(json!({ "session": body["session"], "routes": [] }))
            } else {
                MockResponse::error(500, "synthetic polling failure")
            };
        }
        if request.method == "POST" && request.path == "/v3/sandboxes" {
            return MockResponse::json(replacement_body());
        }
        if request.method == "GET" && request.path == "/v2/sandboxes/sessions/sess-2" {
            let body = replacement_body();
            return MockResponse::json(json!({ "session": body["session"], "routes": [] }));
        }
        inner(request)
    })
    .await;
    let state = openai_agents_sdk::sandbox::SandboxSessionState::new(
        "vercel",
        "state-1",
        ROOT,
        &Manifest::new().with_root(ROOT),
    )
    .with_provider_id(Some("agents-test:sess-1".to_string()));

    let resumed = client(&server)
        .resume(&state, Manifest::new().with_root(ROOT))
        .await
        .expect("a failed poll falls back to a replacement");

    assert_eq!(create_count(&server), 1);
    assert!(!resumed.system_state_preserved_on_start());
    server.stop();
}
