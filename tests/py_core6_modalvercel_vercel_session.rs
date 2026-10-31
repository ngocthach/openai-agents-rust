//! Ports of the session, path, resume, archive and write groups of
//! `tests/extensions/sandbox/test_vercel.py` and `test_vercel_provider_contract.py`.
//!
//! Python patches the `vercel.sandbox` SDK; the Rust client speaks the Vercel sandbox REST API, so
//! the same behaviour is asserted against a local mock that records each request.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::io::Read as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::vercel::{
    VercelSandboxClient, VercelSandboxClientOptions, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::runtime_helpers::RESOLVE_WORKSPACE_PATH_HELPER;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxClient, SandboxSessionState};
use serde_json::{Value, json};
use support::{Counter, MockResponse, MockServer, Recorded};

fn body(status: &str, session_id: &str) -> Value {
    json!({
        "sandbox": {
            "name": "agents-test",
            "persistent": false,
            "createdAt": 1,
            "updatedAt": 1,
            "currentSessionId": session_id,
        },
        "session": {
            "id": session_id,
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

fn command(stdout: &str, stderr: &str, exit_code: i64) -> MockResponse {
    MockResponse::ndjson(&[
        json!({ "stream": "stdout", "data": stdout }),
        json!({ "stream": "stderr", "data": stderr }),
        json!({
            "command": {
                "id": "cmd-1", "name": "sh", "args": [], "cwd": "/vercel/sandbox",
                "sessionId": "sess-1", "exitCode": exit_code, "startedAt": 1,
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

fn argvs(server: &MockServer) -> Vec<Vec<String>> {
    server
        .requests()
        .iter()
        .filter(|request| request.path.ends_with("/cmd"))
        .map(argv_of)
        .collect()
}

/// A handler for the surface a basic session touches; `cmd` is answered by `commands`.
fn handler<F>(commands: F) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static
where
    F: Fn(&[String]) -> MockResponse + Send + Sync + 'static,
{
    move |request: &Recorded| match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v3/sandboxes") => MockResponse::json(body("running", "sess-1")),
        ("GET", path) if path.starts_with("/v2/sandboxes/sessions/") => {
            let loaded = body("running", "sess-1");
            MockResponse::json(json!({ "session": loaded["session"], "routes": [] }))
        }
        ("POST", path) if path.ends_with("/cmd") => commands(&argv_of(request)),
        ("POST", path) if path.ends_with("/fs/read") => MockResponse::bytes("payload"),
        ("POST", path) if path.ends_with("/snapshot") => {
            MockResponse::json(json!({ "snapshot": { "id": "snap-1" } }))
        }
        ("POST", path) if path.ends_with("/stop") => {
            let stopped = body("stopped", "sess-1");
            MockResponse::json(json!({ "session": stopped["session"] }))
        }
        _ => MockResponse::json(json!({})),
    }
}

/// Every command answers with something, so the in-sandbox path resolver succeeds.
fn plain(_argv: &[String]) -> MockResponse {
    command("ok", "", 0)
}

fn options(server: &MockServer) -> VercelSandboxClientOptions {
    VercelSandboxClientOptions::new()
        .with_token("vercel_test")
        .with_base_url(&server.base_url)
        .with_status_poll_interval(Duration::from_millis(1))
}

fn client(server: &MockServer) -> VercelSandboxClient {
    VercelSandboxClient::with_options(options(server))
}

/// The `(x-cwd, member name, contents)` triples of every `fs/write` upload.
fn uploads(server: &MockServer) -> Vec<(String, String, Vec<u8>)> {
    let mut out = Vec::new();
    for request in server.requests() {
        if !request.path.ends_with("/fs/write") {
            continue;
        }
        let cwd = request.header("x-cwd").unwrap_or_default().to_string();
        let decoder = flate2::read::GzDecoder::new(request.body.as_slice());
        let mut archive = tar::Archive::new(decoder);
        for entry in archive.entries().expect("entries") {
            let mut entry = entry.expect("entry");
            let name = entry.path().expect("path").to_string_lossy().into_owned();
            let mut contents = Vec::new();
            entry.read_to_end(&mut contents).expect("read");
            out.push((cwd.clone(), name, contents));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Start and workspace roots
// ---------------------------------------------------------------------------------------------

/// `test_vercel_start_uses_base_session_contract_and_materializes_workspace`
#[tokio::test]
async fn start_materializes_the_workspace_under_the_provider_root() {
    let server = MockServer::start(handler(plain)).await;
    let manifest = Manifest::new().with_entry("notes.txt", Entry::file(b"payload".to_vec()));
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    session.start().await.expect("start");

    assert_eq!(
        argvs(&server)[0],
        vec!["mkdir", "-p", "--", "/vercel/sandbox"],
        "the root is created first"
    );
    let written = uploads(&server);
    assert_eq!(written.len(), 1);
    assert_eq!(written[0].1, "vercel/sandbox/notes.txt");
    assert_eq!(written[0].2, b"payload");
    server.stop();
}

/// `test_vercel_start_materializes_entries_under_literal_manifest_root` and
/// `test_vercel_start_bootstraps_arbitrary_absolute_root_before_using_it_as_cwd`: the root is used
/// literally (a space is not split), and is created before any command runs inside it.
#[tokio::test]
async fn start_uses_the_manifest_root_literally() {
    for root in ["/workspace/my app", "/tmp/outside"] {
        let server = MockServer::start(handler(plain)).await;
        let manifest = Manifest::new()
            .with_root(root)
            .with_entry("notes.txt", Entry::file(b"payload".to_vec()));
        let session = client(&server)
            .create(Some(manifest))
            .await
            .expect("create");

        session.start().await.expect("start");

        assert_eq!(argvs(&server)[0], vec!["mkdir", "-p", "--", root], "{root}");
        let written = uploads(&server);
        assert_eq!(
            written[0].1,
            format!("{}/notes.txt", root.trim_start_matches('/')),
            "{root}"
        );
        server.stop();
    }
}

/// `test_vercel_create_allows_manifest_root_outside_provider_workspace` and
/// `test_vercel_create_allows_manifest_root_within_provider_workspace`
#[tokio::test]
async fn create_keeps_the_manifest_root() {
    for root in ["/tmp/outside", "/vercel/sandbox/my app"] {
        let server = MockServer::start(handler(plain)).await;

        let session = client(&server)
            .create(Some(Manifest::new().with_root(root)))
            .await
            .expect("create");

        assert_eq!(session.session_state().workspace_root, root);
        server.stop();
    }
}

/// `test_vercel_normalize_path_rejects_workspace_escape_and_allows_absolute_in_root`
#[tokio::test]
async fn normalize_path_rejects_an_escape_and_allows_an_absolute_path_in_the_root() {
    let server = MockServer::start(handler(plain)).await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root("/vercel/sandbox/project")))
        .await
        .expect("create");

    assert_eq!(
        session
            .normalize_path("../outside.txt")
            .expect_err("an escape")
            .code,
        ErrorCode::InvalidManifestPath
    );
    assert_eq!(
        session
            .normalize_path("/etc/passwd")
            .expect_err("outside the root")
            .code,
        ErrorCode::InvalidManifestPath
    );
    assert_eq!(
        session
            .normalize_path("/vercel/sandbox/project/nested/file.txt")
            .expect("inside the root"),
        "/vercel/sandbox/project/nested/file.txt"
    );
    server.stop();
}

/// A handler whose in-sandbox path resolver answers with `exit_code` and `stderr`.
fn resolver_failing(
    exit_code: i64,
    stderr: &'static str,
) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    let resolver = RESOLVE_WORKSPACE_PATH_HELPER.install_path.clone();
    handler(move |argv| {
        if argv.first() == Some(&resolver) {
            return command("", stderr, exit_code);
        }
        command("ok", "", 0)
    })
}

/// `test_vercel_read_rejects_workspace_symlink_to_ungranted_path`: a path that resolves through a
/// symlink to somewhere ungranted is refused before the file is read.
#[tokio::test]
async fn read_rejects_a_symlink_to_an_ungranted_path() {
    let server = MockServer::start(resolver_failing(
        111,
        "workspace escape: /private/secret.txt",
    ))
    .await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root("/vercel/sandbox")))
        .await
        .expect("create");

    let error = session
        .read_file("link/secret.txt")
        .await
        .expect_err("the link escapes");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(
        error.message,
        "manifest path must not escape root: link/secret.txt"
    );
    assert_eq!(
        error.context.get("rel").map(String::as_str),
        Some("link/secret.txt")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("escape_root")
    );
    assert_eq!(
        error.context.get("resolved_path").map(String::as_str),
        Some("workspace escape: /private/secret.txt")
    );
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/read"),
        0
    );
    server.stop();
}

/// `test_vercel_write_rejects_workspace_symlink_to_read_only_extra_path_grant`
#[tokio::test]
async fn write_rejects_a_symlink_to_a_read_only_grant() {
    let server = MockServer::start(resolver_failing(
        114,
        "read-only extra path grant: /tmp/protected\nresolved path: /tmp/protected/out.txt",
    ))
    .await;
    let manifest = Manifest::new()
        .with_root("/vercel/sandbox")
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp/protected")
                .expect("grant")
                .read_only(true),
        );
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");

    let error = session
        .write_file("link/out.txt", b"blocked")
        .await
        .expect_err("the grant is read-only");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.message,
        "failed to write archive for path: /vercel/sandbox/link/out.txt"
    );
    for (key, value) in [
        ("path", "/vercel/sandbox/link/out.txt"),
        ("reason", "read_only_extra_path_grant"),
        ("grant_path", "/tmp/protected"),
        ("resolved_path", "/tmp/protected/out.txt"),
    ] {
        assert_eq!(
            error.context.get(key).map(String::as_str),
            Some(value),
            "{key}"
        );
    }
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write"),
        0
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------------------------

/// `test_vercel_write_marks_typed_validation_error_non_retryable`: a 400 from the file API is
/// reported once and flagged as not worth retrying.
#[tokio::test]
async fn write_marks_a_validation_error_non_retryable() {
    let attempts = Counter::new();
    let inner = handler(plain);
    let counted = attempts.clone();
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/fs/write") {
            counted.next("write");
            return MockResponse::error(400, "validation failed");
        }
        inner(request)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .write_file("notes.txt", b"payload")
        .await
        .expect_err("a 400");

    assert_eq!(error.retryable, Some(false));
    assert_eq!(attempts.get("write"), 1);
    server.stop();
}

/// `test_vercel_write_retries_transient_transport_error`: a transient failure is retried and the
/// second attempt succeeds.
#[tokio::test]
async fn write_retries_a_transient_failure() {
    let attempts = Counter::new();
    let inner = handler(plain);
    let counted = attempts.clone();
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/fs/write") && counted.next("write") == 0 {
            return MockResponse::error(503, "try again");
        }
        inner(request)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    session
        .write_file("notes.txt", b"payload")
        .await
        .expect("the retry succeeds");

    assert_eq!(attempts.get("write"), 2);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------------------------

fn state(sandbox_name: &str) -> SandboxSessionState {
    state_for(sandbox_name, &Manifest::new())
}

fn state_for(sandbox_name: &str, manifest: &Manifest) -> SandboxSessionState {
    SandboxSessionState::new("vercel", "session-1", "/vercel/sandbox", manifest)
        .with_provider_id(Some(format!("{sandbox_name}:sess-1")))
}

/// A handler whose first session lookup reports `first`, and every later one `running`.
fn lookup_reporting(
    first: &'static str,
    lookups: Counter,
) -> impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static {
    let inner = handler(plain);
    move |request: &Recorded| {
        if request.method == "GET"
            && request.path == "/v2/sandboxes/sessions/sess-1"
            && lookups.next("get") == 0
        {
            let loaded = body(first, "sess-1");
            return MockResponse::json(json!({ "session": loaded["session"], "routes": [] }));
        }
        inner(request)
    }
}

/// `test_vercel_resume_waits_when_sandbox_pending`: a sandbox that is still starting is waited
/// for, not replaced.
#[tokio::test]
async fn resume_waits_for_a_pending_sandbox() {
    let server = MockServer::start(lookup_reporting("pending", Counter::new())).await;

    client(&server)
        .resume(&state("agents-test"), Manifest::new())
        .await
        .expect("resume");

    assert_eq!(
        server.count("POST", "/v3/sandboxes"),
        0,
        "nothing is created"
    );
    assert!(
        server.count("GET", "/v2/sandboxes/sessions/sess-1") >= 2,
        "the status was polled"
    );
    server.stop();
}

/// `test_vercel_resume_recreates_sandbox_when_cannot_reach_running`: every state that cannot
/// become `running` is replaced at once, without waiting out the running timeout.
#[tokio::test]
async fn resume_recreates_a_sandbox_that_cannot_reach_running() {
    for status in ["stopping", "stopped", "failed", "aborted", "snapshotting"] {
        let server = MockServer::start(lookup_reporting(status, Counter::new())).await;
        let started = std::time::Instant::now();

        client(&server)
            .resume(&state("agents-test"), Manifest::new())
            .await
            .expect("resume");

        assert_eq!(server.count("POST", "/v3/sandboxes"), 1, "{status}");
        assert!(started.elapsed() < Duration::from_secs(5), "{status}");
        server.stop();
    }
}

/// `test_vercel_resume_recreates_sandbox_after_wait_timeout`: a sandbox that stays `pending` past
/// the wait budget is replaced.
#[tokio::test]
async fn resume_recreates_a_sandbox_that_stays_pending() {
    let created = Arc::new(Mutex::new(false));
    let flag = created.clone();
    let inner = handler(plain);
    let server = MockServer::start(move |request: &Recorded| {
        if request.method == "POST" && request.path == "/v3/sandboxes" {
            *flag.lock().expect("poisoned") = true;
        }
        // Until a replacement exists every lookup says `pending`.
        if request.method == "GET"
            && request.path.starts_with("/v2/sandboxes/sessions/")
            && !*flag.lock().expect("poisoned")
        {
            let loaded = body("pending", "sess-1");
            return MockResponse::json(json!({ "session": loaded["session"], "routes": [] }));
        }
        inner(request)
    })
    .await;

    VercelSandboxClient::with_options(
        options(&server).with_wait_for_running(Duration::from_millis(50)),
    )
    .resume(&state("agents-test"), Manifest::new())
    .await
    .expect("resume");

    assert_eq!(server.count("POST", "/v3/sandboxes"), 1);
    server.stop();
}

/// `test_vercel_resume_falls_back_to_recreate_when_sandbox_missing`: the replacement keeps the
/// recorded runtime and timeout, and the manifest environment wins over the options'.
#[tokio::test]
async fn resume_recreates_a_missing_sandbox_with_the_recorded_options() {
    let lookups = Counter::new();
    let inner = handler(plain);
    let server = MockServer::start(move |request: &Recorded| {
        if request.method == "GET"
            && request.path.starts_with("/v2/sandboxes/")
            && lookups.next("get") == 0
        {
            return MockResponse::error(404, "missing");
        }
        inner(request)
    })
    .await;
    let manifest = Manifest::new().with_env("FLAG", "manifest");

    VercelSandboxClient::with_options(
        options(&server)
            .with_runtime("python3.13")
            .with_timeout_ms(90_000)
            .with_env("FLAG", "options")
            .with_env("BASE", "1"),
    )
    .resume(&state_for("sandbox-missing", &manifest), manifest)
    .await
    .expect("resume");

    let create = server.find("POST", "/v3/sandboxes").json();
    assert_eq!(create["timeout"], 90_000);
    assert_eq!(create["env"]["FLAG"], "manifest");
    assert_eq!(create["env"]["BASE"], "1");
    assert!(create.to_string().contains("python:al-3.13"), "{create}");
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Archives
// ---------------------------------------------------------------------------------------------

fn gz_tar(build: impl FnOnce(&mut tar::Builder<Vec<u8>>)) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    build(&mut builder);
    let raw = builder.into_inner().expect("finish");
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &raw).expect("compress");
    encoder.finish().expect("finish")
}

fn file_archive(name: &str, contents: &[u8]) -> Vec<u8> {
    gz_tar(|builder| {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        builder
            .append_data(&mut header, name, contents)
            .expect("append");
    })
}

/// `test_vercel_tar_persistence_round_trip`: what a tar persist downloads can be hydrated into
/// another sandbox, under the same root.
#[tokio::test]
async fn tar_persistence_round_trips() {
    let archive = file_archive("hello.txt", b"world");
    let served = archive.clone();
    let inner = handler(plain);
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/fs/read") {
            return MockResponse::bytes(served.clone());
        }
        inner(request)
    })
    .await;
    let session = VercelSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Tar),
    )
    .create(Some(Manifest::new().with_root("/workspace")))
    .await
    .expect("create");

    let persisted = session.persist_workspace().await.expect("persist");
    assert_eq!(persisted, archive);

    session
        .hydrate_workspace(&persisted)
        .await
        .expect("hydrate");
    let request = server.find("POST", "/v2/sandboxes/sessions/sess-1/fs/write");
    assert_eq!(request.header("x-cwd"), Some("/vercel/sandbox"));
    assert_eq!(request.body, archive);
    server.stop();
}

/// `test_vercel_tar_persist_raises_archive_error_on_nonzero_exec` (the cleanup half): the scratch
/// tarball is removed after a failed archive, and `--no-wildcards` keeps excludes literal.
#[tokio::test]
async fn tar_persist_failure_removes_the_scratch_tarball() {
    let server = MockServer::start(handler(|argv| {
        if argv.iter().any(|part| part.contains("tar -czf")) {
            return command("", "tar failed", 2);
        }
        command("ok", "", 0)
    }))
    .await;
    let session = VercelSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Tar),
    )
    .create(Some(Manifest::new().with_root("/workspace")))
    .await
    .expect("create");

    let error = session.persist_workspace().await.expect_err("tar fails");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(error.message.contains("tar failed"), "{}", error.message);
    let last = argvs(&server).pop().expect("a command");
    assert_eq!(last[0], "rm", "{last:?}");
    assert!(last.last().expect("path").starts_with("/tmp/openai-agents"));
    server.stop();
}

/// `test_vercel_hydrate_workspace_rejects_unsafe_tar_before_upload`,
/// `..._rejects_external_symlink_target_before_upload` and
/// `test_vercel_validate_tar_bytes_rejects_unsafe_members`: nothing unsafe is uploaded.
#[tokio::test]
async fn hydrate_rejects_unsafe_archives_before_upload() {
    let server = MockServer::start(handler(plain)).await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");

    let traversal = {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        let name = b"../escape.txt";
        header.as_old_mut().name[..name.len()].copy_from_slice(name);
        header.set_size(4);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        builder.append(&header, &b"data"[..]).expect("append");
        builder.into_inner().expect("finish")
    };
    let external_link = {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_path("leak").expect("path");
        header.set_link_name("/etc/passwd").expect("link");
        header.set_size(0);
        header.set_mode(0o777);
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_cksum();
        builder.append(&header, std::io::empty()).expect("append");
        builder.into_inner().expect("finish")
    };
    let absolute = {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        let name = b"/etc/passwd";
        header.as_old_mut().name[..name.len()].copy_from_slice(name);
        header.set_size(4);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        builder.append(&header, &b"root"[..]).expect("append");
        builder.into_inner().expect("finish")
    };

    for (label, archive, reason) in [
        ("a traversal member", traversal, "parent traversal"),
        (
            "an external symlink",
            external_link,
            "absolute symlink target not allowed",
        ),
        ("an absolute member", absolute, "absolute path"),
        (
            "a stream that is no tar",
            b"not a tar file".to_vec(),
            "invalid tar stream",
        ),
    ] {
        let error = session.hydrate_workspace(&archive).await.expect_err(label);
        assert!(error.message.contains(reason), "{label}: {}", error.message);
    }
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/fs/write"),
        0
    );
    server.stop();
}

/// `test_vercel_hydrate_workspace_raises_archive_error_on_nonzero_tar_exec`: a failed extraction is
/// a restore error carrying the provider's detail.
#[tokio::test]
async fn hydrate_reports_a_failed_extraction() {
    let inner = handler(plain);
    let server = MockServer::start(move |request: &Recorded| {
        if request.path.ends_with("/fs/write") {
            return MockResponse::error(400, "extract failed");
        }
        inner(request)
    })
    .await;
    let session = client(&server)
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");

    let error = session
        .hydrate_workspace(&file_archive("hello.txt", b"hello"))
        .await
        .expect_err("the extraction fails");

    assert_eq!(error.code, ErrorCode::SnapshotRestoreError);
    assert!(
        error.message.contains("extract failed"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_vercel_snapshot_hydrate_replaces_and_stops_superseded_sandbox`: restoring a native
/// snapshot creates a replacement from the snapshot and stops the sandbox it supersedes.
#[tokio::test]
async fn native_snapshot_hydrate_stops_the_superseded_sandbox() {
    let server = MockServer::start(handler(plain)).await;
    let session = VercelSandboxClient::with_options(
        options(&server).with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .create(Some(Manifest::new().with_root("/workspace")))
    .await
    .expect("create");
    let reference = session.persist_workspace().await;
    let reference = reference.expect("a native snapshot reference");

    session
        .hydrate_workspace(&reference)
        .await
        .expect("restore the snapshot");

    let creates = server.all("POST", "/v3/sandboxes");
    assert_eq!(creates.len(), 2);
    assert_eq!(creates[1].json()["source"]["type"], "snapshot");
    assert_eq!(creates[1].json()["source"]["snapshotId"], "snap-1");
    assert_eq!(
        server.count("POST", "/v2/sandboxes/sessions/sess-1/stop"),
        1,
        "the superseded session is stopped"
    );
    server.stop();
}
