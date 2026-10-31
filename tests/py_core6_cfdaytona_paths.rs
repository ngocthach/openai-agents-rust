//! Workspace-escape and resume-fingerprint cases of the Cloudflare, E2B and Daytona Python tests.
//!
//! Every read, write, mkdir, rm, ls and mv first runs the path resolver inside the sandbox, as
//! Python's `_validate_path_access` does, so a workspace symlink that leaves the root (or lands on
//! a read-only grant) is refused before any file request is made. The mock plays the resolver,
//! following the symlinks a test registers, and records every other request.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::HashMap;
use std::sync::Arc;

use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareSandboxClient, CloudflareSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::daytona::{
    DaytonaSandboxClient, DaytonaSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::e2b::{E2BSandboxClient, E2BSandboxClientOptions};
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxPathGrant, SandboxSession, SandboxSessionState,
    SnapshotFingerprint, SnapshotSpec, User, resolve_snapshot,
};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded, Resolver, Shared};

#[derive(Clone, Copy, Debug)]
enum Provider {
    Cloudflare,
    E2b,
    Daytona,
}

const ALL: [Provider; 3] = [Provider::Cloudflare, Provider::E2b, Provider::Daytona];

impl Provider {
    fn root(self) -> &'static str {
        match self {
            Provider::Cloudflare | Provider::E2b => "/workspace",
            Provider::Daytona => "/home/daytona/workspace",
        }
    }

    fn resolver(self) -> Resolver {
        match self {
            Provider::Cloudflare => Resolver::Cloudflare,
            Provider::E2b => Resolver::E2b,
            Provider::Daytona => Resolver::Daytona,
        }
    }

    /// A request that reads, writes or lists files, or runs a command: anything that is not the
    /// sandbox lifecycle.
    fn is_operation(self, request: &Recorded) -> bool {
        let path = request.path.as_str();
        match self {
            Provider::Cloudflare => {
                path.contains("/file/") || path.ends_with("/exec") || path.ends_with("/persist")
            }
            Provider::E2b => {
                path == "/files"
                    || path.starts_with("/filesystem.Filesystem/")
                    || path == "/process.Process/Start"
            }
            Provider::Daytona => {
                path.contains("/files")
                    || (path.contains("/process/session/") && path.ends_with("/exec"))
            }
        }
    }

    /// The reply every non-resolver request gets.
    fn reply(self, request: &Recorded) -> MockResponse {
        let path = request.path.as_str();
        match self {
            Provider::Cloudflare => {
                if request.method == "POST" && path == "/v1/sandbox" {
                    return MockResponse::json(json!({ "id": "abc123" }));
                }
                if path.ends_with("/exec") {
                    return support::resolver_result(Resolver::Cloudflare, 0, "", "");
                }
                if request.method == "GET" {
                    return MockResponse::bytes(b"data".to_vec());
                }
                MockResponse::empty(204)
            }
            Provider::E2b => match path {
                "/v2/sandboxes" => {
                    MockResponse::json(json!({ "sandboxID": "sb-123", "envdAccessToken": "t" }))
                }
                "/process.Process/Start" => support::resolver_result(Resolver::E2b, 0, "", ""),
                "/files" if request.method == "GET" => MockResponse::bytes(b"data".to_vec()),
                _ => MockResponse::json(json!({})),
            },
            Provider::Daytona => {
                if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
                    return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
                }
                if path.ends_with("/exec") {
                    return support::resolver_result(Resolver::Daytona, 0, "", "");
                }
                if path.ends_with("/files/download") {
                    return MockResponse::bytes(b"data".to_vec());
                }
                if request.method == "GET" && path.ends_with("/files") {
                    return MockResponse::json(json!([]));
                }
                MockResponse::json(json!({}))
            }
        }
    }
}

fn manifest(provider: Provider, grant: bool) -> Manifest {
    let manifest = Manifest::new().with_root(provider.root());
    if grant {
        manifest.with_extra_path_grant(
            SandboxPathGrant::new("/tmp/protected")
                .expect("grant")
                .read_only(true),
        )
    } else {
        manifest
    }
}

async fn setup(
    provider: Provider,
    grant: bool,
    links: &[(&str, &str)],
) -> (MockServer, Arc<dyn SandboxSession>) {
    let symlinks: HashMap<String, String> = links
        .iter()
        .map(|(link, target)| (format!("{}/{link}", provider.root()), target.to_string()))
        .collect();
    let server = support::start_resolving_links(
        provider.resolver(),
        Shared::new(symlinks),
        move |request| provider.reply(request),
    )
    .await;
    connect(provider, grant, server).await
}

async fn connect(
    provider: Provider,
    grant: bool,
    server: MockServer,
) -> (MockServer, Arc<dyn SandboxSession>) {
    let manifest = manifest(provider, grant);
    let session = match provider {
        Provider::Cloudflare => {
            CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
                .create(Some(manifest))
                .await
        }
        Provider::E2b => {
            E2BSandboxClient::with_options(
                E2BSandboxClientOptions::new()
                    .with_api_key("e2b_test")
                    .with_api_base_url(&server.base_url)
                    .with_sandbox_base_url(&server.base_url),
            )
            .create(Some(manifest))
            .await
        }
        Provider::Daytona => {
            DaytonaSandboxClient::with_options(
                DaytonaSandboxClientOptions::new()
                    .with_api_key("dtn_test")
                    .with_api_url(&server.base_url)
                    .with_state_poll_interval(std::time::Duration::from_millis(1)),
            )
            .create(Some(manifest))
            .await
        }
    }
    .expect("create");
    (server, session)
}

/// Operations the provider saw that are not the resolver's own commands.
fn operations(provider: Provider, server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter(|r| provider.is_operation(r) && !support::is_resolver_request(r))
        .map(|r| r.route())
        .collect()
}

fn context(error: &openai_agents_sdk::sandbox::SandboxError, key: &str) -> Option<String> {
    error.context.get(key).cloned()
}

// ---------------------------------------------------------------------------------------------
// Reads, writes and mkdir
// ---------------------------------------------------------------------------------------------

/// `test_read_rejects_workspace_symlink_to_ungranted_path` (Daytona), the Cloudflare and E2B
/// `normalize_path_*symlink*` / `read_validates_path_access` cases.
#[tokio::test]
async fn read_rejects_a_workspace_symlink_to_an_ungranted_path() {
    for provider in ALL {
        let (server, session) = setup(provider, false, &[("link", "/private")]).await;

        let error = session
            .read_file("link/secret.txt")
            .await
            .expect_err("the link leaves the workspace");
        let bounded = session
            .read_bounded("link/secret.txt", 16)
            .await
            .expect_err("bounded reads resolve too");

        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "{provider:?}");
        assert_eq!(
            error.message, "manifest path must not escape root: link/secret.txt",
            "{provider:?}"
        );
        assert_eq!(context(&error, "reason").as_deref(), Some("escape_root"));
        assert_eq!(
            context(&error, "resolved_path").as_deref(),
            Some("workspace escape: /private/secret.txt")
        );
        // A bounded read reports every refusal as the same sanitized archive-read error.
        assert_eq!(
            bounded.code,
            ErrorCode::WorkspaceArchiveReadError,
            "{provider:?}"
        );
        assert!(operations(provider, &server).is_empty(), "{provider:?}");
        server.stop();
    }
}

/// An ordinary path resolves first and is then read and written.
#[tokio::test]
async fn read_and_write_resolve_an_ordinary_path_first() {
    for provider in ALL {
        let (server, session) = setup(provider, false, &[]).await;

        assert_eq!(session.read_file("a.txt").await.expect("read"), b"data");
        assert_eq!(
            session.read_bounded("a.txt", 16).await.expect("bounded"),
            b"data"
        );
        session.write_file("b.txt", b"x").await.expect("write");

        let resolvers = server
            .requests()
            .iter()
            .filter(|r| support::is_resolver_request(r))
            .count();
        assert!(resolvers >= 3, "{provider:?}: {resolvers}");
        assert!(!operations(provider, &server).is_empty());
        server.stop();
    }
}

/// `test_write_rejects_workspace_symlink_to_read_only_extra_path_grant`
#[tokio::test]
async fn write_rejects_a_workspace_symlink_to_a_read_only_grant() {
    for provider in ALL {
        let (server, session) = setup(provider, true, &[("link", "/tmp/protected")]).await;

        let error = session
            .write_file("link/out.txt", b"blocked")
            .await
            .expect_err("read-only grant");

        let path = format!("{}/link/out.txt", provider.root());
        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{provider:?}"
        );
        assert_eq!(
            error.message,
            format!("failed to write archive for path: {path}")
        );
        assert_eq!(context(&error, "path").as_deref(), Some(path.as_str()));
        assert_eq!(
            context(&error, "reason").as_deref(),
            Some("read_only_extra_path_grant")
        );
        assert_eq!(
            context(&error, "grant_path").as_deref(),
            Some("/tmp/protected")
        );
        assert_eq!(
            context(&error, "resolved_path").as_deref(),
            Some("/tmp/protected/out.txt")
        );
        assert!(operations(provider, &server).is_empty(), "{provider:?}");
        server.stop();
    }
}

/// `test_mkdir_rejects_workspace_symlink_to_read_only_extra_path_grant`
#[tokio::test]
async fn mkdir_rejects_a_workspace_symlink_to_a_read_only_grant() {
    for provider in ALL {
        let (server, session) = setup(provider, true, &[("link", "/tmp/protected")]).await;

        let error = session
            .mkdir("link/newdir", false)
            .await
            .expect_err("read-only grant");

        let path = format!("{}/link/newdir", provider.root());
        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{provider:?}"
        );
        assert_eq!(
            error.message,
            format!("failed to write archive for path: {path}")
        );
        assert_eq!(
            context(&error, "resolved_path").as_deref(),
            Some("/tmp/protected/newdir")
        );
        assert!(operations(provider, &server).is_empty(), "{provider:?}");
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// rm, ls and mv
// ---------------------------------------------------------------------------------------------

/// A workspace symlink that points outside the root cannot be removed, listed or moved through.
#[tokio::test]
async fn rm_ls_and_mv_refuse_a_workspace_symlink_that_leaves_the_root() {
    for provider in ALL {
        let (server, session) = setup(provider, false, &[("link", "/private")]).await;

        for recursive in [false, true] {
            let error = session
                .rm("link/secret.txt", recursive)
                .await
                .expect_err("rm through the link");
            assert_eq!(
                error.code,
                ErrorCode::InvalidManifestPath,
                "{provider:?} rm"
            );
            assert_eq!(
                context(&error, "resolved_path").as_deref(),
                Some("workspace escape: /private/secret.txt")
            );
        }
        let error = session.ls("link").await.expect_err("ls through the link");
        assert_eq!(
            error.code,
            ErrorCode::InvalidManifestPath,
            "{provider:?} ls"
        );
        for (source, destination) in [("link/a.txt", "b.txt"), ("a.txt", "link/b.txt")] {
            let error = session
                .mv(source, destination)
                .await
                .expect_err("mv through the link");
            assert_eq!(
                error.code,
                ErrorCode::InvalidManifestPath,
                "{provider:?} {source} {destination}"
            );
        }
        assert!(operations(provider, &server).is_empty(), "{provider:?}");
        server.stop();
    }
}

/// Ordinary paths still work, and each operation resolves first.
#[tokio::test]
async fn rm_ls_and_mv_succeed_for_ordinary_paths_after_resolving() {
    for provider in ALL {
        let (server, session) = setup(provider, false, &[]).await;

        session.rm("old.txt", false).await.expect("rm");
        session.ls("sub").await.expect("ls");
        session.mv("a.txt", "b.txt").await.expect("mv");

        assert!(
            server.requests().iter().any(support::is_resolver_request),
            "{provider:?}"
        );
        server.stop();
    }
}

/// `test_mkdir_as_user_checks_permissions_then_uses_files_api`: Daytona checks the requested
/// user's access with the shared access-check script, then creates the folder through the
/// toolbox files API.
#[tokio::test]
async fn daytona_mkdir_as_user_checks_permissions_then_uses_the_files_api() {
    let (server, session) = setup(Provider::Daytona, false, &[]).await;

    session
        .mkdir_as("nested", false, Some(&User::new("sandbox-user")))
        .await
        .expect("mkdir");

    let commands: Vec<String> = server
        .requests()
        .iter()
        .filter(|r| r.path.ends_with("/exec"))
        .map(|r| r.json()["command"].as_str().unwrap_or_default().to_string())
        .collect();
    let check = commands
        .iter()
        .find(|c| c.contains("sudo -u sandbox-user"))
        .unwrap_or_else(|| panic!("no access check in {commands:?}"));
    assert!(check.contains("/home/daytona/workspace/nested"), "{check}");
    assert!(check.contains("target="), "{check}");
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(|r| r.path.ends_with("/files/folder"))
            .count(),
        1
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume fingerprint gate (E2B and Daytona)
// ---------------------------------------------------------------------------------------------

fn tar_with_one_file() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(2);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, "a.txt", &b"hi"[..])
        .expect("append");
    builder.into_inner().expect("tar")
}

/// A resumable state whose snapshot holds an archive and whose recorded fingerprint is `abc`.
async fn fingerprinted_state(
    backend: &str,
    root: &str,
    provider_id: &str,
    dir: &tempfile::TempDir,
) -> SandboxSessionState {
    let spec = SnapshotSpec::Local {
        base_path: dir.path().to_path_buf(),
    };
    let state =
        SandboxSessionState::new(backend, "session-1", root, &Manifest::new().with_root(root))
            .with_provider_id(Some(provider_id.to_string()))
            .with_snapshot(spec.clone())
            .with_snapshot_fingerprint(Some(SnapshotFingerprint {
                fingerprint: "abc".to_string(),
                version: "workspace_tar_sha256_v2".to_string(),
            }));
    resolve_snapshot(Some(&spec), &state.snapshot_id)
        .persist_with(&tar_with_one_file(), None)
        .await
        .expect("store the snapshot");
    state
}

/// The fingerprint helper's reply when `fingerprint` is what the live workspace hashes to.
fn fingerprint_reply(
    kind: Resolver,
    request: &Recorded,
    fingerprint: &str,
) -> Option<MockResponse> {
    let argv = support::command_argv(request);
    // The helper is invoked as `helper root version cache digest`, possibly after a `cd`.
    if let Some(at) = argv
        .iter()
        .position(|a| a.contains("workspace-fingerprint"))
        && argv.len() >= at + 5
    {
        let record = json!({ "fingerprint": fingerprint, "version": argv[at + 2] }).to_string();
        return Some(support::resolver_result(kind, 0, &record, ""));
    }
    None
}

fn e2b_server(fingerprint: &'static str) -> impl std::future::Future<Output = MockServer> {
    MockServer::start(move |request: &Recorded| {
        if support::is_resolver_request(request) {
            return support::resolver_reply(Resolver::E2b);
        }
        if let Some(reply) = fingerprint_reply(Resolver::E2b, request, fingerprint) {
            return reply;
        }
        match request.path.as_str() {
            "/v2/sandboxes/sb-persisted/connect" => {
                MockResponse::json(json!({ "sandboxID": "sb-persisted" }))
            }
            "/process.Process/Start" => support::resolver_result(Resolver::E2b, 0, "", ""),
            _ => MockResponse::json(json!({})),
        }
    })
}

fn daytona_server(fingerprint: &'static str) -> impl std::future::Future<Output = MockServer> {
    MockServer::start(move |request: &Recorded| {
        if support::is_resolver_request(request) {
            return support::resolver_reply(Resolver::Daytona);
        }
        if let Some(reply) = fingerprint_reply(Resolver::Daytona, request, fingerprint) {
            return reply;
        }
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/exec") {
            return support::resolver_result(Resolver::Daytona, 0, "", "");
        }
        MockResponse::json(json!({}))
    })
}

async fn resume_and_start(provider: Provider, server: &MockServer, state: &SandboxSessionState) {
    let root = provider.root();
    let manifest = Manifest::new().with_root(root);
    let session = match provider {
        Provider::E2b => {
            E2BSandboxClient::with_options(
                E2BSandboxClientOptions::new()
                    .with_api_key("e2b_test")
                    .with_api_base_url(&server.base_url)
                    .with_sandbox_base_url(&server.base_url),
            )
            .resume(state, manifest)
            .await
        }
        Provider::Daytona => {
            DaytonaSandboxClient::with_options(
                DaytonaSandboxClientOptions::new()
                    .with_api_key("dtn_test")
                    .with_api_url(&server.base_url)
                    .with_state_poll_interval(std::time::Duration::from_millis(1)),
            )
            .resume(state, manifest)
            .await
        }
        Provider::Cloudflare => unreachable!("covered in the Cloudflare file"),
    }
    .expect("resume");
    session.start().await.expect("start");
}

fn uploaded_archive(provider: Provider, server: &MockServer) -> usize {
    server
        .requests()
        .iter()
        .filter(|r| {
            r.method == "POST"
                && match provider {
                    Provider::E2b => r.path == "/files",
                    _ => r.path.contains("/files/upload"),
                }
        })
        .count()
}

/// `test_e2b_preserved_start_uses_shared_resume_gate_for_restore` and
/// `test_preserved_start_rehydrates_when_snapshot_gate_requests_restore`: a reattached sandbox
/// whose workspace no longer matches the snapshot fingerprint is restored.
#[tokio::test]
async fn resume_restores_when_the_workspace_fingerprint_differs() {
    for provider in [Provider::E2b, Provider::Daytona] {
        let dir = tempfile::tempdir().expect("tempdir");
        let (backend, id) = match provider {
            Provider::E2b => ("e2b", "sb-persisted"),
            _ => ("daytona", "sbx-daytona"),
        };
        let server = match provider {
            Provider::E2b => e2b_server("changed").await,
            _ => daytona_server("changed").await,
        };
        let state = fingerprinted_state(backend, provider.root(), id, &dir).await;

        resume_and_start(provider, &server, &state).await;

        assert_eq!(uploaded_archive(provider, &server), 1, "{provider:?}");
        server.stop();
    }
}

/// A reattached sandbox whose workspace still hashes to the snapshot's fingerprint is not
/// restored.
#[tokio::test]
async fn resume_skips_the_restore_when_the_workspace_fingerprint_matches() {
    for provider in [Provider::E2b, Provider::Daytona] {
        let dir = tempfile::tempdir().expect("tempdir");
        let (backend, id) = match provider {
            Provider::E2b => ("e2b", "sb-persisted"),
            _ => ("daytona", "sbx-daytona"),
        };
        let server = match provider {
            Provider::E2b => e2b_server("abc").await,
            _ => daytona_server("abc").await,
        };
        let state = fingerprinted_state(backend, provider.root(), id, &dir).await;

        resume_and_start(provider, &server, &state).await;

        assert_eq!(uploaded_archive(provider, &server), 0, "{provider:?}");
        server.stop();
    }
}

/// The fingerprint a session persisted travels in its state.
#[tokio::test]
async fn the_snapshot_fingerprint_round_trips_through_the_session_state() {
    for provider in [Provider::E2b, Provider::Daytona] {
        let dir = tempfile::tempdir().expect("tempdir");
        let (backend, id) = match provider {
            Provider::E2b => ("e2b", "sb-persisted"),
            _ => ("daytona", "sbx-daytona"),
        };
        let server = match provider {
            Provider::E2b => e2b_server("abc").await,
            _ => daytona_server("abc").await,
        };
        let state = fingerprinted_state(backend, provider.root(), id, &dir).await;
        let manifest = Manifest::new().with_root(provider.root());
        let session = match provider {
            Provider::E2b => {
                E2BSandboxClient::with_options(
                    E2BSandboxClientOptions::new()
                        .with_api_key("e2b_test")
                        .with_api_base_url(&server.base_url)
                        .with_sandbox_base_url(&server.base_url),
                )
                .resume(&state, manifest)
                .await
            }
            _ => {
                DaytonaSandboxClient::with_options(
                    DaytonaSandboxClientOptions::new()
                        .with_api_key("dtn_test")
                        .with_api_url(&server.base_url)
                        .with_state_poll_interval(std::time::Duration::from_millis(1)),
                )
                .resume(&state, manifest)
                .await
            }
        }
        .expect("resume");

        let persisted = session.session_state();

        assert_eq!(persisted.snapshot_fingerprint.as_deref(), Some("abc"));
        assert_eq!(
            persisted.snapshot_fingerprint_version.as_deref(),
            Some("workspace_tar_sha256_v2")
        );
        server.stop();
    }
}

/// `test_e2b_start_installs_runtime_helpers`: a sandbox without the resolver helper gets it
/// installed (probe, install, then the resolver run) before the first path is resolved. Rust
/// installs it lazily, at the first path check, rather than in `start`.
#[tokio::test]
async fn the_resolver_helper_is_installed_when_the_probe_finds_it_missing() {
    for provider in ALL {
        let missing = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = missing.clone();
        let server = MockServer::start(move |request: &Recorded| {
            if support::is_resolver_request(request) {
                let argv = support::command_argv(request);
                let is_probe = argv.iter().any(|a| a == "-x" || a == "test");
                if is_probe && flag.load(std::sync::atomic::Ordering::SeqCst) {
                    return support::resolver_result(provider.resolver(), 1, "", "");
                }
                if !is_probe {
                    flag.store(false, std::sync::atomic::Ordering::SeqCst);
                }
                return support::resolver_reply(provider.resolver());
            }
            provider.reply(request)
        })
        .await;
        let (server, session) = connect(provider, false, server).await;

        session.read_file("a.txt").await.expect("read");

        let helper_requests = server
            .requests()
            .iter()
            .filter(|r| support::is_resolver_request(r))
            .count();
        assert!(
            helper_requests >= 3,
            "{provider:?}: probe, install, resolve; saw {helper_requests}"
        );
        server.stop();
    }
}
