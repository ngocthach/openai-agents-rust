//! Cloudflare cases of `tests/extensions/sandbox/test_cloudflare.py` that
//! `py_sandbox_cloudflare.rs` does not port: bucket-mount configuration and lifecycle, protected
//! state, PTY sessions over the worker's websocket, streamed payloads and the terminal-failure
//! paths. Python injects a fake `aiohttp` session; these run the real client against a local
//! worker that records every request and speaks websocket frames by hand.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use openai_agents_sdk::extensions::sandbox::ProviderMountStrategy;
use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy, CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::session::persist_session_snapshot;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, GcsMount, Manifest, PtyExecRequest, PtyWriteRequest, R2Mount, S3Mount,
    SandboxClient, SandboxSession, SandboxSessionState, SnapshotSpec, resolve_snapshot,
};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded};
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn sse(body: &str) -> MockResponse {
    MockResponse::text(body.to_string()).with_content_type("text/event-stream")
}

fn exec_ok() -> MockResponse {
    sse(&format!(
        "event: exit\ndata: {}\n\n",
        json!({ "exit_code": 0 })
    ))
}

/// A valid empty-member tar holding one file, for hydrate.
fn valid_tar() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(5);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, "hello.txt", &b"hello"[..])
        .expect("append");
    builder.into_inner().expect("tar")
}

/// A tar whose only member escapes the workspace.
fn traversal_tar() -> Vec<u8> {
    let mut header = tar::Header::new_gnu();
    let name = b"../../etc/passwd";
    header.as_old_mut().name[..name.len()].copy_from_slice(name);
    header.set_size(5);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();
    let mut bytes = header.as_bytes().to_vec();
    let mut body = b"evil\n".to_vec();
    body.resize(512, 0);
    bytes.extend(body);
    bytes.extend(vec![0u8; 1024]);
    bytes
}

fn with_status(mut response: MockResponse, status: u16) -> MockResponse {
    response.status = status;
    response
}

fn base_handler(request: &Recorded) -> Option<MockResponse> {
    if request.method == "POST" && request.path == "/v1/sandbox" {
        return Some(MockResponse::json(json!({ "id": "abc123" })));
    }
    None
}

async fn worker(handler: impl Fn(&Recorded) -> MockResponse + Send + Sync + 'static) -> MockServer {
    support::start_resolving(support::Resolver::Cloudflare, move |request: &Recorded| {
        base_handler(request).unwrap_or_else(|| handler(request))
    })
    .await
}

fn client(server: &MockServer) -> CloudflareSandboxClient {
    CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
}

fn s3_config() -> CloudflareBucketMountConfig {
    CloudflareBucketMountConfig::from_configured_mount(
        &S3Mount::new("bucket").with_prefix("nested/prefix/"),
    )
    .expect("config")
}

fn mount_manifest(config: CloudflareBucketMountConfig) -> Manifest {
    Manifest::new().with_entry(
        "data",
        Entry::mount(CloudflareBucketMountStrategy::mount(config)),
    )
}

/// The trailing path segment of every worker request that is a mount or archive operation.
fn lifecycle(server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter(|r| r.method != "DELETE")
        .filter_map(|r| {
            let last = r.path.rsplit('/').next()?.to_string();
            ["running", "mount", "unmount", "persist", "hydrate"]
                .contains(&last.as_str())
                .then_some(last)
        })
        .collect()
}

fn local_snapshot(dir: &tempfile::TempDir) -> SnapshotSpec {
    SnapshotSpec::Local {
        base_path: dir.path().to_path_buf(),
    }
}

/// A persisted state for a worker-side sandbox `abc123` whose snapshot holds `valid_tar()`.
async fn resumable_state(manifest: &Manifest, dir: &tempfile::TempDir) -> SandboxSessionState {
    let spec = local_snapshot(dir);
    let state = SandboxSessionState::new("cloudflare", "session-1", "/workspace", manifest)
        .with_provider_id(Some("abc123".to_string()))
        .with_snapshot(spec.clone());
    resolve_snapshot(Some(&spec), &state.snapshot_id)
        .persist_with(&valid_tar(), None)
        .await
        .expect("store the snapshot");
    state
}

// ---------------------------------------------------------------------------------------------
// Bucket mount configuration
// ---------------------------------------------------------------------------------------------

/// `test_cloudflare_bucket_mount_strategy_round_trips_through_manifest_parse`
#[test]
fn bucket_mount_strategy_round_trips_through_the_provider_registry() {
    let strategy = CloudflareBucketMountStrategy::new(s3_config());
    let payload = serde_json::to_value(ProviderMountStrategy::CloudflareBucketMount(strategy))
        .expect("serialize");
    assert_eq!(payload["type"], "cloudflare_bucket_mount");

    let parsed = ProviderMountStrategy::parse(&payload).expect("parse");

    assert_eq!(parsed.strategy_type(), "cloudflare_bucket_mount");
    assert!(matches!(
        parsed,
        ProviderMountStrategy::CloudflareBucketMount(_)
    ));
}

/// `test_cloudflare_bucket_mount_strategy_builds_s3_config`
#[test]
fn bucket_mount_strategy_builds_s3_config() {
    let config = CloudflareBucketMountConfig::from_configured_mount(
        &S3Mount::new("bucket")
            .with_credentials("access-key", "secret-key")
            .with_prefix("nested/prefix/"),
    )
    .expect("config")
    .with_read_only(false);

    assert_eq!(config.bucket_name, "bucket");
    assert_eq!(config.bucket_endpoint_url, "https://s3.amazonaws.com");
    assert_eq!(config.provider.as_str(), "s3");
    assert_eq!(config.key_prefix.as_deref(), Some("/nested/prefix/"));
    assert_eq!(
        config.credentials,
        Some(("access-key".to_string(), "secret-key".to_string()))
    );
    assert!(!config.read_only);
}

/// `test_cloudflare_bucket_mount_strategy_builds_r2_config`
#[test]
fn bucket_mount_strategy_builds_r2_config() {
    let config = CloudflareBucketMountConfig::from_configured_mount(
        &R2Mount::new("bucket", "abc123accountid").with_credentials("access-key", "secret-key"),
    )
    .expect("config");

    assert_eq!(config.bucket_name, "bucket");
    assert_eq!(
        config.bucket_endpoint_url,
        "https://abc123accountid.r2.cloudflarestorage.com"
    );
    assert_eq!(config.provider.as_str(), "r2");
    assert_eq!(config.key_prefix, None);
    assert_eq!(
        config.credentials,
        Some(("access-key".to_string(), "secret-key".to_string()))
    );
    assert!(config.read_only, "a mount is read-only unless asked");
}

/// `test_cloudflare_bucket_mount_strategy_builds_gcs_hmac_config`
#[test]
fn bucket_mount_strategy_builds_gcs_hmac_config() {
    let config = CloudflareBucketMountConfig::from_configured_mount(
        &GcsMount::new("bucket")
            .with_hmac_credentials("access-id", "secret-key")
            .with_prefix("nested/prefix/"),
    )
    .expect("config")
    .with_read_only(false);

    assert_eq!(config.bucket_name, "bucket");
    assert_eq!(config.bucket_endpoint_url, "https://storage.googleapis.com");
    assert_eq!(config.provider.as_str(), "gcs");
    assert_eq!(config.key_prefix.as_deref(), Some("/nested/prefix/"));
    assert_eq!(
        config.credentials,
        Some(("access-id".to_string(), "secret-key".to_string()))
    );
    assert!(!config.read_only);
}

/// `test_cloudflare_bucket_mount_strategy_rejects_gcs_native_auth`
#[test]
fn bucket_mount_strategy_rejects_gcs_native_auth() {
    let error = CloudflareBucketMountConfig::from_configured_mount(
        &GcsMount::new("bucket").with_service_account_file("/data/config/gcs.json"),
    )
    .expect_err("native auth");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error
            .message
            .contains("gcs cloudflare bucket mounts require"),
        "{}",
        error.message
    );
}

/// `test_cloudflare_bucket_mount_strategy_rejects_s3_session_token`
#[test]
fn bucket_mount_strategy_rejects_s3_session_token() {
    let error = CloudflareBucketMountConfig::from_configured_mount(
        &S3Mount::new("bucket")
            .with_credentials("access-key", "secret-key")
            .with_session_token("session-token"),
    )
    .expect_err("session token");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error
            .message
            .contains("cloudflare bucket mounts do not support s3 session_token credentials"),
        "{}",
        error.message
    );
}

// ---------------------------------------------------------------------------------------------
// Client timeouts and resume
// ---------------------------------------------------------------------------------------------

/// `test_cloudflare_create_uses_client_timeouts`: the exec timeout reaches the worker, and no
/// timeout is persisted in the session state.
#[tokio::test]
async fn create_uses_client_timeouts_without_persisting_them() {
    let server = worker(|_| exec_ok()).await;
    let client = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url)
            .with_exec_timeout(Duration::from_secs(10))
            .with_request_timeout(Duration::from_secs(60)),
    );
    let session = client.create(None).await.expect("create");

    session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("exec");

    assert_eq!(
        server.find("POST", "/v1/sandbox/abc123/exec").json()["timeout_ms"],
        10_000
    );
    let state = serde_json::to_string(&session.session_state()).expect("state");
    assert!(!state.contains("timeout"), "{state}");
    server.stop();
}

/// `test_cloudflare_create_uses_client_timeouts`: the request timeout bounds every non-exec call.
#[tokio::test]
async fn request_timeout_bounds_file_requests() {
    let server = worker(|request| {
        if request.method == "GET" {
            return MockResponse::streamed("application/octet-stream", vec![(Vec::new(), 3_000)]);
        }
        MockResponse::empty(204)
    })
    .await;
    let client = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url)
            .with_request_timeout(Duration::from_millis(150)),
    );
    let session = client.create(None).await.expect("create");

    let started = std::time::Instant::now();
    let error = session
        .read_file("slow.txt")
        .await
        .expect_err("the worker never answers in time");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(started.elapsed() < Duration::from_secs(2));
    server.stop();
}

/// `test_cloudflare_resume_uses_client_timeouts`: a resumed session takes its timeouts from the
/// client, not from the state.
#[tokio::test]
async fn resume_uses_client_timeouts() {
    let server = worker(|request| {
        if request.path.ends_with("/running") {
            return MockResponse::json(json!({ "running": true }));
        }
        exec_ok()
    })
    .await;
    let client = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url)
            .with_exec_timeout(Duration::from_secs(11)),
    );
    let state = SandboxSessionState::new("cloudflare", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("abc123".to_string()));
    let state =
        SandboxSessionState::from_json_str(&state.to_json_string().expect("json")).expect("parse");

    let session = client
        .resume(&state, Manifest::new())
        .await
        .expect("resume");
    session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("exec");

    assert_eq!(
        server.find("POST", "/v1/sandbox/abc123/exec").json()["timeout_ms"],
        11_000
    );
    server.stop();
}

/// `test_cloudflare_protected_mount_state_drops_identity_before_resume`: a state written from a
/// manifest holding bucket credentials carries no sandbox id, and cannot be resumed.
#[tokio::test]
async fn protected_mount_state_drops_identity_before_resume() {
    let server = worker(|_| exec_ok()).await;
    let manifest = mount_manifest(s3_config().with_credentials("access-key", "secret-key"));
    let client = client(&server);
    let session = client.create(Some(manifest.clone())).await.expect("create");

    let payload = session.session_state().to_json_string().expect("persist");
    let restored = SandboxSessionState::from_json_str(&payload).expect("parse");

    assert_eq!(restored.provider_id, None, "the sandbox id is dropped");
    assert!(!restored.workspace_root_ready);
    assert!(!payload.contains("abc123"), "{payload}");
    let rebound = restored.rebind_path_grants(&manifest).expect("rebind");
    let error = client
        .resume(&rebound, manifest)
        .await
        .expect_err("a protected sandbox is not resumed");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("protected bucket configuration"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_cloudflare_resume_rejects_direct_state_with_configured_authority`: the refusal happens
/// before the worker is asked anything.
#[tokio::test]
async fn resume_rejects_direct_state_with_configured_authority() {
    let server = worker(|_| MockResponse::json(json!({ "running": true }))).await;
    let manifest = mount_manifest(s3_config().with_credentials("access-key", "secret-key"));
    let state = SandboxSessionState::new("cloudflare", "session-1", "/workspace", &manifest)
        .with_provider_id(Some("abc123".to_string()));

    let error = client(&server)
        .resume(&state, manifest)
        .await
        .expect_err("protected");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(server.requests().is_empty(), "{:?}", server.routes());
    server.stop();
}

/// `test_cloudflare_resume_sets_preserved_state_from_running`: only a sandbox the worker still
/// serves keeps its operating-system state across the resume.
#[tokio::test]
async fn resume_marks_system_state_preserved_only_for_a_running_sandbox() {
    for running in [false, true] {
        let server = worker(move |request| {
            if request.path.ends_with("/running") {
                return MockResponse::json(json!({ "running": running }));
            }
            MockResponse::empty(204)
        })
        .await;
        let state =
            SandboxSessionState::new("cloudflare", "session-1", "/workspace", &Manifest::new())
                .with_provider_id(Some("abc123".to_string()));

        let session = client(&server)
            .resume(&state, Manifest::new())
            .await
            .expect("resume");

        assert_eq!(
            session.system_state_preserved_on_start(),
            running,
            "running={running}"
        );
        assert_eq!(server.count("GET", "/v1/sandbox/abc123/running"), 1);
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Mount lifecycle around archives
// ---------------------------------------------------------------------------------------------

/// `test_cloudflare_mount_and_unmount_bucket_use_http_endpoints`: a manifest bucket mount is
/// attached through `POST /mount` with the documented body, with credentials in `options`.
#[tokio::test]
async fn start_mounts_the_bucket_through_the_worker() {
    let server = worker(|request| {
        if request.path.ends_with("/mount") {
            return MockResponse::json(json!({ "ok": true }));
        }
        exec_ok()
    })
    .await;
    let session = client(&server)
        .create(Some(mount_manifest(
            CloudflareBucketMountConfig::s3("my-bucket")
                .with_endpoint_url("https://s3.amazonaws.com")
                .with_read_only(true),
        )))
        .await
        .expect("create");

    session.start().await.expect("start");

    let mount = server.find("POST", "/v1/sandbox/abc123/mount");
    assert_eq!(
        mount.json(),
        json!({
            "bucket": "my-bucket",
            "mountPath": "/workspace/data",
            "options": { "endpoint": "https://s3.amazonaws.com", "readOnly": true },
        })
    );
    server.stop();
}

/// `test_cloudflare_mount_and_unmount_bucket_use_http_endpoints` and
/// `test_cloudflare_persist_unmounts_and_remounts_ephemeral_bucket_mounts`: a snapshot detaches
/// the bucket, archives, and re-attaches it, in that order.
#[tokio::test]
async fn persist_unmounts_and_remounts_ephemeral_bucket_mounts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("persist") => MockResponse::bytes(valid_tar()),
        Some("mount") | Some("unmount") => MockResponse::json(json!({ "ok": true })),
        _ => exec_ok(),
    })
    .await;
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .with_snapshot(local_snapshot(&dir))
            .create(Some(mount_manifest(s3_config())))
            .await
            .expect("create");

    persist_session_snapshot(&*session).await.expect("snapshot");

    assert_eq!(lifecycle(&server), ["unmount", "persist", "mount"]);
    let unmount = server.find("POST", "/v1/sandbox/abc123/unmount");
    assert_eq!(unmount.json(), json!({ "mountPath": "/workspace/data" }));
    server.stop();
}

/// `test_cloudflare_resume_start_hydrates_without_preemptive_unmount`: a sandbox the worker no
/// longer serves has no live mount, so the archive is hydrated and the bucket attached.
#[tokio::test]
async fn resume_start_hydrates_without_preemptive_unmount() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("running") => MockResponse::json(json!({ "running": false })),
        Some("mount") | Some("hydrate") => MockResponse::json(json!({ "ok": true })),
        _ => exec_ok(),
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir).await;

    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");
    session.start().await.expect("start");

    assert_eq!(lifecycle(&server), ["running", "hydrate", "mount"]);
    server.stop();
}

/// `test_cloudflare_resume_start_unmounts_before_hydrate_when_sandbox_is_running`: a live bucket
/// is detached first so the archive is never unpacked through it.
#[tokio::test]
async fn resume_start_unmounts_before_hydrate_when_sandbox_is_running() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("running") => MockResponse::json(json!({ "running": true })),
        Some("mount") | Some("unmount") | Some("hydrate") => {
            MockResponse::json(json!({ "ok": true }))
        }
        // Anything run inside the sandbox fails, so the workspace never matches the snapshot.
        _ => exec_ok(),
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir).await;

    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");
    session.start().await.expect("start");

    assert_eq!(
        lifecycle(&server),
        ["running", "unmount", "hydrate", "mount"]
    );
    server.stop();
}

/// `test_cloudflare_resume_start_does_not_restore_after_terminal_reapply_failure`: a bucket that
/// cannot be re-attached on resume deletes the sandbox and leaves the session unusable.
#[tokio::test]
async fn resume_start_deletes_the_sandbox_after_a_terminal_reapply_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| {
        if request.method == "DELETE" {
            return MockResponse::json(json!({ "ok": true }));
        }
        match request.path.rsplit('/').next() {
            Some("running") => MockResponse::json(json!({ "running": true })),
            Some("mount") => MockResponse::status(502, json!({ "error": "mount failed" })),
            Some("unmount") | Some("hydrate") => MockResponse::json(json!({ "ok": true })),
            _ => exec_ok(),
        }
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir).await;
    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");

    let error = session.start().await.expect_err("the mount failed");

    assert_eq!(error.code, ErrorCode::MountFailed, "{error:?}");
    assert!(
        error.message.contains("cloudflare bucket mount"),
        "{}",
        error.message
    );
    assert_eq!(
        lifecycle(&server),
        ["running", "unmount", "hydrate", "mount"]
    );
    assert_eq!(
        server.count("DELETE", "/v1/sandbox/abc123"),
        1,
        "{:?}",
        server.routes()
    );
    let unusable = session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect_err("an ambiguous mount transition ends the session");
    assert!(
        unusable.message.contains("ambiguous mount transition"),
        "{}",
        unusable.message
    );
    server.stop();
}

/// `test_cloudflare_resume_reapply_surfaces_terminal_delete_failure`: when the terminal delete
/// itself fails, that is the error the caller sees, with the status and no provider body.
#[tokio::test]
async fn resume_reapply_surfaces_terminal_delete_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| {
        if request.method == "DELETE" {
            return with_status(
                MockResponse::text("provider response must not be read"),
                502,
            );
        }
        match request.path.rsplit('/').next() {
            Some("running") => MockResponse::json(json!({ "running": true })),
            Some("mount") => MockResponse::status(502, json!({ "error": "mount failed" })),
            Some("unmount") | Some("hydrate") => MockResponse::json(json!({ "ok": true })),
            _ => exec_ok(),
        }
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir).await;
    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");

    let error = session.start().await.expect_err("terminal delete failed");

    assert_eq!(error.code, ErrorCode::WorkspaceStopError, "{error:?}");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("terminal_delete_failed")
    );
    assert_eq!(
        error.context.get("http_status").map(String::as_str),
        Some("502")
    );
    assert_eq!(
        error.context.get("backend").map(String::as_str),
        Some("cloudflare")
    );
    assert!(!error.message.contains("must not be read"), "{error:?}");
    server.stop();
}

/// `test_cloudflare_direct_persistence_redacts_protected_remount_failure`: a remount failure
/// whose provider reply echoes the secret never reaches the error text.
#[tokio::test]
async fn persistence_redacts_a_protected_remount_failure() {
    let sentinel = "cloudflare-direct-persistence-secret";
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(move |request| match request.path.rsplit('/').next() {
        Some("persist") => MockResponse::bytes(valid_tar()),
        Some("unmount") => MockResponse::json(json!({ "ok": true })),
        Some("mount") => MockResponse::status(
            500,
            json!({ "error": format!("provider echoed {sentinel}") }),
        ),
        _ => exec_ok(),
    })
    .await;
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .with_snapshot(local_snapshot(&dir))
            .create(Some(mount_manifest(
                s3_config().with_credentials("access-key", sentinel),
            )))
            .await
            .expect("create");

    let error = persist_session_snapshot(&*session)
        .await
        .expect_err("the bucket could not be re-attached");

    assert!(!format!("{error:?}").contains(sentinel), "{error:?}");
    assert!(!error.to_string().contains(sentinel), "{error}");
    server.stop();
}

/// `test_cloudflare_mount_rejects_read_only_extra_path_grant`: a mount under a read-only grant
/// is refused before any request reaches the worker.
#[tokio::test]
async fn mount_under_a_read_only_extra_path_grant_is_refused() {
    use openai_agents_sdk::sandbox::SandboxPathGrant;
    let server = worker(|_| exec_ok()).await;
    let manifest = Manifest::new()
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp/protected")
                .expect("grant")
                .read_only(true),
        )
        .with_entry(
            "/tmp/protected/data",
            Entry::mount(CloudflareBucketMountStrategy::mount(s3_config())),
        );
    let session = client(&server).create(Some(manifest)).await;

    let outcome = match session {
        Ok(session) => session.start().await,
        Err(error) => Err(error),
    };

    assert!(outcome.is_err(), "the mount must not be attached");
    assert_eq!(
        server.count("POST", "/v1/sandbox/abc123/mount"),
        0,
        "{:?}",
        server.routes()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Streamed payloads, persist excludes and hydrate validation
// ---------------------------------------------------------------------------------------------

fn streamed_payload(payload: &[u8], is_binary: bool, complete: bool) -> Vec<u8> {
    let chunk = if is_binary {
        support::b64(payload)
    } else {
        String::from_utf8(payload.to_vec()).expect("utf8")
    };
    let mut body = format!(
        "data: {{\"type\":\"metadata\",\"isBinary\":{is_binary}}}\n\n\
         data: {{\"type\":\"chunk\",\"data\":\"{chunk}\"}}\n\n"
    );
    if complete {
        body.push_str("data: {\"type\":\"complete\"}\n\n");
    }
    body.into_bytes()
}

async fn file_session(body: Vec<u8>, status: u16) -> (MockServer, Arc<dyn SandboxSession>) {
    let server = worker(move |request| {
        if request.method == "GET" {
            return with_status(MockResponse::bytes(body.clone()), status);
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    (server, session)
}

/// `test_cloudflare_read_decodes_streamed_file_payload`
#[tokio::test]
async fn read_decodes_a_streamed_file_payload() {
    let (_server, session) =
        file_session(streamed_payload(b"file-content", false, true), 200).await;

    assert_eq!(
        session
            .read_file("/workspace/test.txt")
            .await
            .expect("read"),
        b"file-content"
    );
}

/// `test_cloudflare_read_leaves_raw_data_prefix_payload_unchanged`
#[tokio::test]
async fn read_leaves_a_raw_data_prefix_payload_unchanged() {
    let raw = b"data: this is a normal file, not an SSE payload\n{\"ok\": false}\n".to_vec();
    let (_server, session) = file_session(raw.clone(), 200).await;

    assert_eq!(
        session
            .read_file("/workspace/test.txt")
            .await
            .expect("read"),
        raw
    );
}

/// `test_cloudflare_read_rejects_truncated_streamed_file_payload`
#[tokio::test]
async fn read_rejects_a_truncated_streamed_file_payload() {
    let (_server, session) =
        file_session(streamed_payload(b"file-content", false, false), 200).await;

    let error = session
        .read_file("/workspace/test.txt")
        .await
        .expect_err("no complete event");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("complete"), "{}", error.message);
}

/// `test_cloudflare_read_404_and_write_non_bytes_raise_structured_errors` (the 404 half; Rust
/// writes `&[u8]`, so the non-bytes half cannot be expressed).
#[tokio::test]
async fn read_404_names_the_missing_workspace_path() {
    let (_server, session) = file_session(br#"{"error":"not found"}"#.to_vec(), 404).await;

    let error = session
        .read_file("/workspace/missing.txt")
        .await
        .expect_err("missing");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    assert!(
        error.message.contains("/workspace/missing.txt"),
        "{}",
        error.message
    );
}

/// `test_cloudflare_read_and_write_normalize_workspace_paths`: both directions refuse a
/// traversal before any file request is made.
#[tokio::test]
async fn read_and_write_refuse_traversal_before_any_request() {
    let (server, session) = file_session(Vec::new(), 200).await;
    let before = server.requests().len();

    let read = session.read_file("../secret.txt").await.expect_err("read");
    let write = session
        .write_file("/workspace/../secret.txt", b"data")
        .await
        .expect_err("write");

    assert_eq!(read.code, ErrorCode::InvalidManifestPath);
    assert_eq!(write.code, ErrorCode::InvalidManifestPath);
    assert_eq!(server.requests().len(), before);
}

/// `test_cloudflare_persist_and_hydrate_use_http_endpoints` and
/// `test_cloudflare_persist_preserves_hidden_exclude_paths`: the ephemeral path and every
/// registered skip path are excluded, normalized and ordered, and hydrate sends the tar as is.
#[tokio::test]
async fn persist_excludes_ephemeral_and_registered_skip_paths() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::bytes(b"fake-tar".to_vec());
        }
        MockResponse::empty(204)
    })
    .await;
    let manifest = Manifest::new().with_entry("cache", Entry::dir().with_ephemeral(true));
    let session = client(&server)
        .create(Some(manifest))
        .await
        .expect("create");
    session
        .register_persist_workspace_skip_path(".sandbox-blobfuse-config/session")
        .expect("skip path");
    session
        .register_persist_workspace_skip_path("./generated/runtime")
        .expect("skip path");

    let archive = session.persist_workspace().await.expect("persist");
    session
        .hydrate_workspace(&valid_tar())
        .await
        .expect("hydrate");

    assert_eq!(archive, b"fake-tar");
    let excludes = server
        .find("POST", "/v1/sandbox/abc123/persist")
        .query_param("excludes")
        .expect("excludes");
    let excludes: Vec<&str> = excludes.split(',').collect();
    assert!(excludes.contains(&"cache"), "{excludes:?}");
    assert!(
        excludes.contains(&".sandbox-blobfuse-config/session"),
        "{excludes:?}"
    );
    assert!(excludes.contains(&"generated/runtime"), "{excludes:?}");
    assert!(
        excludes.iter().all(|e| !e.starts_with("./")),
        "no ./ prefix survives: {excludes:?}"
    );
    let hydrate = server.find("POST", "/v1/sandbox/abc123/hydrate");
    assert_eq!(hydrate.query, "", "hydrate carries no root parameter");
    assert_eq!(hydrate.body, valid_tar());
    server.stop();
}

/// `test_cloudflare_persist_decodes_streamed_archive_payload`
#[tokio::test]
async fn persist_decodes_a_streamed_archive_payload() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::bytes(streamed_payload(b"fake-tar", true, true));
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    assert_eq!(
        session.persist_workspace().await.expect("persist"),
        b"fake-tar"
    );
    server.stop();
}

/// `test_cloudflare_persist_leaves_raw_data_prefix_archive_unchanged`
#[tokio::test]
async fn persist_leaves_a_raw_data_prefix_archive_unchanged() {
    let raw = b"data: raw tar bytes that happen to share the prefix".to_vec();
    let expected = raw.clone();
    let server = worker(move |request| {
        if request.path.ends_with("/persist") {
            return MockResponse::bytes(raw.clone());
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    assert_eq!(
        session.persist_workspace().await.expect("persist"),
        expected
    );
    server.stop();
}

/// `test_cloudflare_persist_rejects_truncated_streamed_archive_payload`
#[tokio::test]
async fn persist_rejects_a_truncated_streamed_archive_payload() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::bytes(streamed_payload(b"fake-tar", true, false));
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session.persist_workspace().await.expect_err("truncated");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    server.stop();
}

/// `test_cloudflare_persist_retries_only_documented_503_status`: a worker that stays at 503
/// is retried up to three attempts and the final error stays retryable.
#[tokio::test]
async fn persist_gives_up_on_a_persistent_503_after_three_attempts() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::status(503, json!({ "error": "container starting" }));
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session.persist_workspace().await.expect_err("503");

    assert_eq!(server.count("POST", "/v1/sandbox/abc123/persist"), 3);
    assert!(error.message.contains("HTTP 503"), "{}", error.message);
    assert_eq!(error.retryable, Some(true));
    server.stop();
}

/// `test_cloudflare_persist_does_not_retry_documented_fail_fast_500`: the error also names the
/// archive-read class and is marked not retryable.
#[tokio::test]
async fn persist_500_fails_fast_and_is_not_retryable() {
    let server = worker(|request| {
        if request.path.ends_with("/persist") {
            return MockResponse::status(500, json!({ "error": "configuration error" }));
        }
        MockResponse::empty(204)
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session.persist_workspace().await.expect_err("500");

    assert_eq!(server.count("POST", "/v1/sandbox/abc123/persist"), 1);
    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert_eq!(error.retryable, Some(false));
    assert!(error.message.contains("HTTP 500"), "{}", error.message);
    server.stop();
}

/// `test_cloudflare_hydrate_rejects_unsafe_tar`: a traversal member is refused before the
/// worker is contacted.
#[tokio::test]
async fn hydrate_rejects_an_unsafe_tar_without_contacting_the_worker() {
    let server = worker(|_| MockResponse::json(json!({ "ok": true }))).await;
    let session = client(&server).create(None).await.expect("create");
    let before = server.requests().len();

    let error = session
        .hydrate_workspace(&traversal_tar())
        .await
        .expect_err("traversal");

    assert!(
        matches!(
            error.code,
            ErrorCode::SnapshotRestoreError | ErrorCode::WorkspaceArchiveWriteError
        ),
        "{error:?}"
    );
    assert_eq!(server.requests().len(), before);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Shutdown never echoes the provider
// ---------------------------------------------------------------------------------------------

/// `test_cloudflare_shutdown_does_not_log_protected_mount_authority` and
/// `test_cloudflare_shutdown_logs_respect_tool_data_policy` (redacted case): a failed delete
/// reports its status and never the provider body, which may echo a mount secret.
#[tokio::test]
async fn failed_shutdown_never_echoes_the_provider_body() {
    let sentinel = "cloudflare-secret-access-key";
    let server = worker(move |request| {
        if request.method == "DELETE" {
            return with_status(
                MockResponse::text(format!("provider echoed {sentinel}")),
                502,
            );
        }
        exec_ok()
    })
    .await;
    let session = client(&server)
        .create(Some(mount_manifest(
            s3_config().with_credentials("access-key", sentinel),
        )))
        .await
        .expect("create");

    let error = session.shutdown().await.expect_err("delete failed");

    assert_eq!(error.code, ErrorCode::WorkspaceStopError);
    assert!(error.message.contains("HTTP 502"), "{}", error.message);
    assert!(!format!("{error:?}").contains(sentinel), "{error:?}");
    server.stop();
}

/// `test_cloudflare_shutdown_does_not_log_protected_mount_exception`: a transport failure whose
/// text would carry a secret is reported as a stop error. (The connection error text comes from
/// the local socket, so the sentinel cannot appear; this pins the error class.)
#[tokio::test]
async fn shutdown_transport_failure_is_a_stop_error() {
    let server = worker(|_| exec_ok()).await;
    let session = client(&server).create(None).await.expect("create");
    server.stop();

    let error = session.shutdown().await.expect_err("worker is gone");

    assert_eq!(error.code, ErrorCode::WorkspaceStopError);
}

// ---------------------------------------------------------------------------------------------
// PTY sessions over the worker websocket
// ---------------------------------------------------------------------------------------------

/// What the worker's terminal does after the handshake.
#[derive(Clone)]
struct Terminal {
    /// Frames sent right after the connection opens (a `ready` frame is among them).
    opening: Vec<Message>,
    /// Frames sent when binary input arrives; keyed by a substring of that input.
    replies: Vec<(String, Vec<Message>)>,
}

#[derive(Default)]
struct ConnectionLog {
    received: Vec<Vec<u8>>,
    closed: bool,
    query: String,
    path: String,
    authorization: Option<String>,
}

type Logs = Arc<Mutex<Vec<Arc<Mutex<ConnectionLog>>>>>;

fn ready() -> Message {
    Message::Text(r#"{"type":"ready"}"#.into())
}

fn exit(code: i32) -> Message {
    Message::Text(format!(r#"{{"type":"exit","code":{code}}}"#).into())
}

fn bin(bytes: &[u8]) -> Message {
    Message::Binary(bytes.to_vec().into())
}

/// A worker that serves `POST /v1/sandbox` over HTTP and the PTY endpoint over websocket.
async fn terminal_worker(terminal: Terminal) -> (MockServer, Logs) {
    let logs: Logs = Arc::new(Mutex::new(Vec::new()));
    let served = logs.clone();
    let server = MockServer::start_raw(move |request, mut stream| {
        let terminal = terminal.clone();
        let served = served.clone();
        async move {
            let upgrade = request
                .header("upgrade")
                .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
            if !upgrade {
                let response = base_handler(&request)
                    .unwrap_or_else(|| MockResponse::status(500, json!({ "error": "no pty" })));
                let _ = support::write_response(&mut stream, response).await;
                return;
            }
            let log = Arc::new(Mutex::new(ConnectionLog {
                query: request.query.clone(),
                path: request.path.clone(),
                authorization: request.header("authorization").map(str::to_string),
                ..ConnectionLog::default()
            }));
            served.lock().unwrap().push(log.clone());
            let mut socket = support::accept_websocket(stream, &request, None).await;
            for frame in terminal.opening.clone() {
                if socket.send(frame).await.is_err() {
                    return;
                }
            }
            while let Some(message) = socket.next().await {
                match message {
                    Ok(Message::Binary(bytes)) => {
                        let text = String::from_utf8_lossy(&bytes).to_string();
                        log.lock().unwrap().received.push(bytes.to_vec());
                        for (needle, frames) in &terminal.replies {
                            if text.contains(needle.as_str()) {
                                for frame in frames.clone() {
                                    let _ = socket.send(frame).await;
                                }
                            }
                        }
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            log.lock().unwrap().closed = true;
        }
    })
    .await;
    (server, logs)
}

fn pty_request(command: &str) -> PtyExecRequest {
    PtyExecRequest::new(command)
        .with_shell(openai_agents_sdk::sandbox::ExecShell::None)
        .with_tty(true)
        .with_yield_time_ms(300)
}

async fn wait_for_closed(logs: &Logs, index: usize) -> bool {
    for _ in 0..50 {
        if logs
            .lock()
            .unwrap()
            .get(index)
            .is_some_and(|log| log.lock().unwrap().closed)
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    false
}

/// `test_cloudflare_pty_exec_start_opens_websocket_and_sends_command`
#[tokio::test]
async fn pty_exec_start_opens_the_websocket_and_sends_the_command() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready(), bin(b">>> "), exit(0)],
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let started = session
        .pty_exec_start(pty_request("python3"))
        .await
        .expect("pty");

    assert_eq!(started.process_id, None);
    assert_eq!(started.exit_code, Some(0));
    assert_eq!(started.output, b">>> ");
    let log = logs.lock().unwrap()[0].clone();
    assert_eq!(log.lock().unwrap().path, "/v1/sandbox/abc123/pty");
    assert_eq!(log.lock().unwrap().query, "cols=80&rows=24");
    assert_eq!(log.lock().unwrap().received, vec![b"python3\n".to_vec()]);
    assert!(wait_for_closed(&logs, 0).await, "the websocket is closed");
}

/// `test_cloudflare_pty_exec_start_opens_websocket_and_sends_command` (auth half): the bearer
/// token rides the websocket upgrade.
#[tokio::test]
async fn pty_websocket_carries_the_bearer_token() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready(), exit(0)],
        replies: Vec::new(),
    })
    .await;
    let session = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url).with_api_key("cf-token"),
    )
    .create(None)
    .await
    .expect("create");

    session
        .pty_exec_start(pty_request("true"))
        .await
        .expect("pty");

    let auth = logs.lock().unwrap()[0]
        .lock()
        .unwrap()
        .authorization
        .clone();
    assert_eq!(auth.as_deref(), Some("Bearer cf-token"));
}

/// `test_cloudflare_pty_write_stdin_sends_input_and_collects_output`
#[tokio::test]
async fn pty_write_stdin_sends_input_and_collects_output() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready()],
        replies: vec![("5 + 5".to_string(), vec![bin(b"10\n")])],
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let started = session
        .pty_exec_start(pty_request("python3"))
        .await
        .expect("pty");
    let process_id = started.process_id.expect("still running");

    let updated = session
        .pty_write_stdin(PtyWriteRequest::new(process_id, "5 + 5\n").with_yield_time_ms(300))
        .await
        .expect("write");

    assert_eq!(updated.process_id, Some(process_id));
    assert_eq!(updated.exit_code, None);
    assert_eq!(updated.output, b"10\n");
    let received = logs.lock().unwrap()[0].lock().unwrap().received.clone();
    assert_eq!(received, vec![b"python3\n".to_vec(), b"5 + 5\n".to_vec()]);
    session.pty_terminate_all().await.expect("terminate");
}

/// `test_cloudflare_pty_write_stdin_rejects_unknown_session`
#[tokio::test]
async fn pty_write_stdin_rejects_an_unknown_session() {
    let server = worker(|_| exec_ok()).await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(999_999, ""))
        .await
        .expect_err("unknown");

    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    server.stop();
}

/// `test_cloudflare_pty_write_stdin_rejects_non_tty_input`
#[tokio::test]
async fn pty_write_stdin_rejects_input_for_a_non_tty_process() {
    let (server, _logs) = terminal_worker(Terminal {
        opening: vec![ready()],
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let started = session
        .pty_exec_start(pty_request("cat").with_tty(false))
        .await
        .expect("pty");

    let error = session
        .pty_write_stdin(PtyWriteRequest::new(
            started.process_id.expect("running"),
            "hello",
        ))
        .await
        .expect_err("no stdin");

    assert_eq!(error.code, ErrorCode::PtyStdinUnavailable);
    assert!(
        error
            .message
            .contains("stdin is not available for this process"),
        "{}",
        error.message
    );
    session.pty_terminate_all().await.expect("terminate");
}

/// `test_cloudflare_pty_terminate_all_closes_websockets`
#[tokio::test]
async fn pty_terminate_all_closes_every_websocket() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready()],
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let first = session.pty_exec_start(pty_request("a")).await.expect("a");
    let second = session.pty_exec_start(pty_request("b")).await.expect("b");

    session.pty_terminate_all().await.expect("terminate");

    assert!(wait_for_closed(&logs, 0).await);
    assert!(wait_for_closed(&logs, 1).await);
    for id in [first.process_id, second.process_id] {
        let error = session
            .pty_write_stdin(PtyWriteRequest::new(id.expect("id"), ""))
            .await
            .expect_err("terminated");
        assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    }
}

/// `test_cloudflare_pty_exec_start_prunes_oldest_session`: the 65th live terminal closes the
/// oldest one and the table stays at the cap.
#[tokio::test]
async fn pty_exec_start_prunes_the_oldest_session() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready()],
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let cap = openai_agents_sdk::sandbox::pty::PTY_PROCESSES_MAX;

    let oldest = session
        .pty_exec_start(pty_request("oldest"))
        .await
        .expect("oldest")
        .process_id
        .expect("id");
    let mut starts = Vec::new();
    for index in 1..cap {
        let session = session.clone();
        starts.push(async move {
            session
                .pty_exec_start(pty_request(&format!("p{index}")))
                .await
        });
    }
    for result in futures::future::join_all(starts).await {
        result.expect("start");
    }
    assert_eq!(logs.lock().unwrap().len(), cap);
    assert!(!logs.lock().unwrap()[0].lock().unwrap().closed);

    let extra = session
        .pty_exec_start(pty_request("extra"))
        .await
        .expect("extra");

    assert!(extra.process_id.is_some());
    assert!(wait_for_closed(&logs, 0).await, "the oldest was pruned");
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(oldest, ""))
        .await
        .expect_err("pruned");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
    session.pty_terminate_all().await.expect("terminate");
}

/// `test_cloudflare_pty_exec_start_wraps_websocket_connect_failures`
#[tokio::test]
async fn pty_exec_start_wraps_websocket_connect_failures() {
    // The worker answers the upgrade with a plain 500, so the handshake fails.
    let server = worker(|_| MockResponse::status(500, json!({ "error": "no pty" }))).await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .pty_exec_start(pty_request("python3"))
        .await
        .expect_err("connect failed");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("PTY WebSocket"), "{}", error.message);
    server.stop();
}

/// `test_cloudflare_pty_exec_start_wraps_ready_timeout`: a worker that never sends `ready` ends
/// in a timeout error after the 30 second ready window (virtual time).
#[tokio::test]
async fn pty_exec_start_wraps_a_ready_timeout() {
    let (server, logs) = terminal_worker(Terminal {
        opening: Vec::new(),
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    // The handshake is real; only the ready window is waited out in virtual time.
    tokio::time::pause();
    let error = session
        .pty_exec_start(pty_request("python3"))
        .await
        .expect_err("never ready");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert!(wait_for_closed(&logs, 0).await, "the socket is dropped");
}

/// `test_cloudflare_stop_terminates_active_pty_sessions`
#[tokio::test]
async fn stop_terminates_active_pty_sessions() {
    let (server, logs) = terminal_worker(Terminal {
        opening: vec![ready()],
        replies: Vec::new(),
    })
    .await;
    let session = client(&server).create(None).await.expect("create");
    let started = session
        .pty_exec_start(pty_request("cat"))
        .await
        .expect("pty");

    session.stop().await.expect("stop");

    assert!(wait_for_closed(&logs, 0).await);
    let error = session
        .pty_write_stdin(PtyWriteRequest::new(started.process_id.expect("id"), ""))
        .await
        .expect_err("stopped");
    assert_eq!(error.code, ErrorCode::PtySessionNotFound);
}

// ---------------------------------------------------------------------------------------------
// Status table, start error text and the resume fast path
// ---------------------------------------------------------------------------------------------

/// `test_cloudflare_retryability_status_table`: a status outside the documented table is not
/// classified at all (neither retryable nor fail-fast).
#[tokio::test]
async fn exec_retryability_is_unclassified_for_an_undocumented_status() {
    let server = worker(|_| MockResponse::status(418, json!({ "error": "teapot" }))).await;
    let session = client(&server).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect_err("418");

    assert_eq!(error.retryable, None);
    assert!(error.message.contains("HTTP 418"), "{}", error.message);
    server.stop();
}

/// `test_cloudflare_prepare_workspace_preserves_exec_error_context`: a worker failure while the
/// workspace root is prepared surfaces as a start error that carries the provider's text.
#[tokio::test]
async fn start_wraps_the_prepare_workspace_exec_failure() {
    let server = worker(|_| {
        MockResponse::status(
            502,
            json!({ "error": "pool error: Failed to start container", "code": "pool_error" }),
        )
    })
    .await;
    let session = client(&server).create(None).await.expect("create");

    let error = session.start().await.expect_err("worker refused");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.starts_with("failed to start session: "),
        "{}",
        error.message
    );
    assert!(
        error.message.contains(
            "POST /exec failed: HTTP 502: pool_error: pool error: Failed to start container"
        ),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_cloudflare_resume_start_skips_hydrate_when_shared_resume_gate_matches`: a sandbox the
/// worker kept alive whose workspace still hashes to the snapshot's fingerprint is not
/// hydrated, and its buckets are attached without being detached first.
#[tokio::test]
async fn resume_start_skips_hydrate_when_the_workspace_matches_the_snapshot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("running") => MockResponse::json(json!({ "running": true })),
        Some("mount") | Some("unmount") | Some("hydrate") => {
            MockResponse::json(json!({ "ok": true }))
        }
        Some("exec") => {
            let argv = request.json()["argv"].clone();
            // The fingerprint helper is invoked as `helper root version cache manifest-digest`.
            if argv.as_array().is_some_and(|a| {
                a.len() >= 5
                    && a[2]
                        .as_str()
                        .is_some_and(|v| v.starts_with("workspace_tar_sha256"))
            }) {
                let record = json!({
                    "fingerprint": "abc",
                    "version": argv[2],
                })
                .to_string();
                return sse(&format!(
                    "event: stdout\ndata: {}\n\nevent: exit\ndata: {}\n\n",
                    support::b64(record),
                    json!({ "exit_code": 0 })
                ));
            }
            exec_ok()
        }
        _ => exec_ok(),
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir)
        .await
        .with_snapshot_fingerprint(Some(openai_agents_sdk::sandbox::SnapshotFingerprint {
            fingerprint: "abc".to_string(),
            version: "workspace_tar_sha256_v2".to_string(),
        }));

    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");
    session.start().await.expect("start");

    assert_eq!(lifecycle(&server), ["running", "mount"]);
    server.stop();
}

/// `test_cloudflare_resume_start_skips_hydrate_when_shared_resume_gate_matches` (negative half):
/// a fingerprint that no longer matches falls back to the full restore.
#[tokio::test]
async fn resume_start_hydrates_when_the_workspace_fingerprint_differs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("running") => MockResponse::json(json!({ "running": true })),
        Some("mount") | Some("unmount") | Some("hydrate") => {
            MockResponse::json(json!({ "ok": true }))
        }
        Some("exec") => {
            let argv = request.json()["argv"].clone();
            if argv.as_array().is_some_and(|a| {
                a.len() >= 5
                    && a[2]
                        .as_str()
                        .is_some_and(|v| v.starts_with("workspace_tar_sha256"))
            }) {
                let record = json!({ "fingerprint": "changed", "version": argv[2] }).to_string();
                return sse(&format!(
                    "event: stdout\ndata: {}\n\nevent: exit\ndata: {}\n\n",
                    support::b64(record),
                    json!({ "exit_code": 0 })
                ));
            }
            exec_ok()
        }
        _ => exec_ok(),
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir)
        .await
        .with_snapshot_fingerprint(Some(openai_agents_sdk::sandbox::SnapshotFingerprint {
            fingerprint: "abc".to_string(),
            version: "workspace_tar_sha256_v2".to_string(),
        }));

    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");
    session.start().await.expect("start");

    assert_eq!(
        lifecycle(&server),
        ["running", "unmount", "hydrate", "mount"]
    );
    server.stop();
}

/// `test_cloudflare_direct_persistence_redacts_protected_remount_failure` (hydrate variant): a
/// bucket that fails to attach while the workspace is restored never leaks the secret its
/// provider echoed. Credentialed buckets refuse resume, so the attach happens at start.
#[tokio::test]
async fn start_redacts_a_protected_mount_failure() {
    let sentinel = "cloudflare-start-secret";
    let server = worker(move |request| match request.path.rsplit('/').next() {
        Some("mount") => MockResponse::status(
            500,
            json!({ "error": format!("provider echoed {sentinel}") }),
        ),
        _ => exec_ok(),
    })
    .await;
    let session = client(&server)
        .create(Some(mount_manifest(
            s3_config().with_credentials("access-key", sentinel),
        )))
        .await
        .expect("create");

    let error = session.start().await.expect_err("the mount failed");

    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(!format!("{error:?}").contains(sentinel), "{error:?}");
    assert!(!error.to_string().contains(sentinel), "{error}");
    server.stop();
}

/// `test_cloudflare_resume_start_*` (non-mount failure): a hydrate that fails after the buckets
/// were detached puts them back and leaves the session usable.
#[tokio::test]
async fn resume_start_reattaches_buckets_when_the_hydrate_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = worker(|request| match request.path.rsplit('/').next() {
        Some("running") => MockResponse::json(json!({ "running": true })),
        Some("mount") | Some("unmount") => MockResponse::json(json!({ "ok": true })),
        Some("hydrate") => MockResponse::status(500, json!({ "error": "disk full" })),
        _ => exec_ok(),
    })
    .await;
    let manifest = mount_manifest(s3_config());
    let state = resumable_state(&manifest, &dir).await;
    let session = client(&server)
        .resume(&state, manifest)
        .await
        .expect("resume");

    let error = session.start().await.expect_err("the hydrate failed");

    assert!(error.message.contains("HTTP 500"), "{}", error.message);
    assert_eq!(
        lifecycle(&server),
        ["running", "unmount", "hydrate", "mount"],
        "the bucket is put back"
    );
    assert_eq!(server.count("DELETE", "/v1/sandbox/abc123"), 0);
    session
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("the session is still usable");
    server.stop();
}
