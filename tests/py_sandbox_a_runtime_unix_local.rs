//! Ports of `tests/sandbox/test_runtime.py` (lines 1-4000), unix-local part: the unix-local
//! client driven through the runner's lifecycle (temp workspace roots, caller-owned roots, mount
//! teardown order, path confinement), the runtime helper scripts a session installs and runs, and
//! mount acknowledgements across a capability's manifest replacement.
#![cfg(feature = "sandbox")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::entries::{Entry, EntryKind};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::{
    InContainerMountStrategy, Mount, MountPattern, MountPatternConfig,
};
use openai_agents_sdk::sandbox::runtime_helpers::{
    RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, User};
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn posix(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// A session over the host shell, for the helper-script tests: `exec` runs the argv as given.
#[derive(Debug)]
struct LocalShellSession {
    manifest: Manifest,
}

#[async_trait]
impl SandboxSession for LocalShellSession {
    fn id(&self) -> &str {
        "local-shell"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "local-shell"
    }

    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        let argv = request.resolved_argv();
        let output = tokio::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .await
            .map_err(|e| SandboxError::new(ErrorCode::ExecTransportError, e.to_string()))?;
        Ok(ExecResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }

    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------------
// Runtime helper scripts
// ----------------------------------------------------------------------------------------------

fn test_helper(dir: &Path) -> RuntimeHelperScript {
    RuntimeHelperScript {
        name: "test-helper",
        content: "#!/bin/sh\nprintf 'expected\\n'",
        install_path: posix(&dir.join("runtime-helpers").join("helper")),
    }
}

fn run_argv(argv: &[String]) -> std::process::Output {
    std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .expect("spawn")
}

fn content_of(path: &str) -> String {
    std::fs::read_to_string(path)
        .expect("read helper")
        .trim_end_matches('\n')
        .to_string()
}

/// Python: `test_runtime_helper_install_replaces_tampered_executable`.
#[test]
fn helper_install_replaces_a_tampered_executable() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let helper = test_helper(dir.path());

    let command = helper.install_command();
    assert_eq!(&command[..2], ["sh", "-c"]);
    assert!(run_argv(&command).status.success());
    assert_eq!(content_of(&helper.install_path), helper.content);

    let path = PathBuf::from(&helper.install_path);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(&path, "#!/bin/sh\nprintf 'tampered\\n'").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert!(run_argv(&helper.install_command()).status.success());
    assert_eq!(content_of(&helper.install_path), helper.content);
}

/// Python: `test_runtime_helper_reinstalls_when_cached_binary_is_missing`.
#[tokio::test]
async fn helper_is_reinstalled_when_the_cached_binary_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let helper = test_helper(dir.path());
    let session = LocalShellSession {
        manifest: Manifest::new().with_root(posix(&dir.path().join("workspace"))),
    };

    let installed = session
        .ensure_runtime_helper_installed(&helper)
        .await
        .expect("installs");
    assert_eq!(installed, helper.install_path);
    assert!(Path::new(&helper.install_path).exists());

    std::fs::remove_file(&helper.install_path).unwrap();
    assert!(!Path::new(&helper.install_path).exists());

    let repaired = session
        .ensure_runtime_helper_installed(&helper)
        .await
        .expect("reinstalls");
    assert_eq!(repaired, helper.install_path);
    assert_eq!(content_of(&helper.install_path), helper.content);
}

/// Python: `test_remote_realpath_guard_fails_closed_on_symlink_cycle`.
///
/// The path-resolution helper refuses a symlink loop rather than spinning on it.
#[tokio::test]
async fn the_path_resolution_helper_fails_closed_on_a_symlink_cycle() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("workspace");
    std::fs::create_dir(&root).unwrap();
    std::os::unix::fs::symlink("loop", root.join("loop")).unwrap();
    let session = LocalShellSession {
        manifest: Manifest::new().with_root(posix(&root)),
    };

    let helper = session
        .ensure_runtime_helper_installed(&RESOLVE_WORKSPACE_PATH_HELPER)
        .await
        .expect("installs the helper");
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        session.exec(ExecRequest::argv([
            helper.as_str(),
            &posix(&root),
            &posix(&root.join("loop")),
            "0",
        ])),
    )
    .await
    .expect("the helper terminates")
    .expect("the helper runs");

    assert_ne!(result.exit_code, 0);
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("symlink resolution depth exceeded"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

// ----------------------------------------------------------------------------------------------
// unix_local client lifecycle
// ----------------------------------------------------------------------------------------------

/// Python: `test_unix_local_client_rewrites_default_manifest_root_to_temp_workspace`.
#[tokio::test]
async fn the_default_manifest_root_is_rewritten_to_a_temp_workspace() {
    let client = UnixLocalSandboxClient::new();
    let manifest = Manifest::new().with_entry("default.txt", Entry::text_file("default"));

    let session = client
        .create(Some(manifest.clone()))
        .await
        .expect("session created");
    let root = PathBuf::from(session.manifest().root());

    assert_ne!(session.manifest().root(), manifest.root());
    assert_eq!(manifest.root(), "/workspace");
    assert!(root.is_absolute());
    assert!(
        root.file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("openai-agents-sandbox-"),
        "{}",
        root.display()
    );
    assert_eq!(
        session.manifest().entries().keys().collect::<Vec<_>>(),
        manifest.entries().keys().collect::<Vec<_>>()
    );

    client.delete(session).await.expect("deleted");
    assert!(!root.exists());
}

/// A mount pattern that records the mount paths it detaches, in order.
#[derive(Debug)]
struct OrderPattern {
    unapplied: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl MountPattern for OrderPattern {
    fn pattern_type(&self) -> &str {
        "order"
    }

    fn exposes_credentials(&self) -> bool {
        false
    }

    async fn apply(
        &self,
        _session: &dyn SandboxSession,
        _mount_path: &str,
        _config: &MountPatternConfig,
    ) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn unapply(
        &self,
        _session: &dyn SandboxSession,
        mount_path: &str,
        _config: &MountPatternConfig,
    ) -> Result<(), SandboxError> {
        self.unapplied.lock().unwrap().push(mount_path.to_string());
        Ok(())
    }
}

fn order_mount(unapplied: &Arc<Mutex<Vec<String>>>) -> Mount {
    Mount::new(Arc::new(InContainerMountStrategy::new(
        Arc::new(OrderPattern {
            unapplied: unapplied.clone(),
        }),
        MountPatternConfig::new(),
    )))
}

/// Python: `test_unix_local_client_delete_unmounts_nested_mounts_deepest_first`.
#[tokio::test]
async fn teardown_detaches_nested_mounts_deepest_first() {
    let unapplied = Arc::new(Mutex::new(Vec::new()));
    let client = UnixLocalSandboxClient::new();
    let probe = client.create(None).await.unwrap();
    let root = probe.workspace_root().to_string();
    client.delete(probe).await.unwrap();

    let manifest = Manifest::new()
        .with_root(root.clone())
        .with_entry("outer", Entry::mount(order_mount(&unapplied)))
        .with_entry("outer/child", Entry::mount(order_mount(&unapplied)));
    let session = client.create(Some(manifest)).await.expect("created");
    session.start().await.expect("started");

    client.delete(session).await.expect("deleted");

    assert_eq!(
        unapplied.lock().unwrap().clone(),
        vec![format!("{root}/outer/child"), format!("{root}/outer")]
    );
}

/// Python: `test_unix_local_persist_workspace_excludes_mounted_directory_contents`.
///
/// A mount that appears at a different `mount_path` than its entry path keeps both locations out
/// of the workspace archive.
#[tokio::test]
async fn persisting_the_workspace_leaves_out_the_contents_of_mounted_directories() {
    let unapplied = Arc::new(Mutex::new(Vec::new()));
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    std::fs::create_dir_all(root.join("logical")).unwrap();
    std::fs::write(root.join("logical/marker.txt"), "logical").unwrap();
    std::fs::create_dir_all(root.join("actual")).unwrap();
    std::fs::write(root.join("actual/mounted.txt"), "mounted").unwrap();

    let manifest = Manifest::new().with_root(posix(&root)).with_entry(
        "logical",
        Entry::mount(order_mount(&unapplied).with_mount_path("actual")),
    );
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("created");

    let archive = session.persist_workspace().await.expect("persists");
    // Python opens the archive with `r:*`, which accepts any compression.
    let raw = if archive.starts_with(&[0x1f, 0x8b]) {
        let mut plain = Vec::new();
        std::io::Read::read_to_end(
            &mut flate2::read::GzDecoder::new(archive.as_slice()),
            &mut plain,
        )
        .expect("gunzip");
        plain
    } else {
        archive
    };
    let mut names = Vec::new();
    let mut read_error = None;
    for entry in tar::Archive::new(raw.as_slice()).entries().expect("tar") {
        match entry {
            Ok(entry) => names.push(entry.path().unwrap().to_string_lossy().to_string()),
            Err(error) => {
                read_error = Some(error.to_string());
                break;
            }
        }
    }

    assert!(
        names
            .iter()
            .all(|name| !name.contains("marker.txt") && !name.contains("mounted.txt")),
        "{names:?}"
    );
    assert!(
        !names.is_empty(),
        "archive of {} bytes had no members (read error: {read_error:?})",
        raw.len()
    );
}

/// Python: `test_unix_local_client_delete_preserves_caller_owned_workspace_root`.
#[tokio::test]
async fn teardown_preserves_a_caller_owned_workspace_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(Manifest::new().with_root(posix(dir.path()))))
        .await
        .expect("created");
    assert!(!session.session_state().workspace_root_owned);

    client.delete(session).await.expect("deleted");
    assert!(dir.path().exists());
}

/// Python: `test_unix_local_runner_cleanup_preserves_resumed_caller_owned_workspace_root`.
#[tokio::test]
async fn a_resumed_run_preserves_a_caller_owned_workspace_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = Manifest::new().with_root(posix(dir.path()));
    let client = Arc::new(UnixLocalSandboxClient::new());
    let created = client
        .create(Some(manifest.clone()))
        .await
        .expect("created");
    let state = created.session_state();

    let model = model();
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new());
    let config = SandboxRunConfig::with_client(client)
        .session_state(state)
        .manifest(manifest);
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(dir.path().exists());
}

async fn started_local_session(root: &Path) -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(Manifest::new().with_root(posix(root))))
        .await
        .expect("created");
    session.start().await.expect("started");
    (client, session)
}

/// Python: `test_unix_local_read_and_write_reject_paths_outside_workspace_root`.
#[tokio::test]
async fn reads_and_writes_outside_the_workspace_root_are_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (client, session) = started_local_session(dir.path()).await;

    let write = session
        .write_file("../secret.txt", b"nope")
        .await
        .expect_err("write escapes the root");
    assert_eq!(write.code, ErrorCode::InvalidManifestPath);
    assert!(
        write.message.contains("must not escape root"),
        "{}",
        write.message
    );

    let read = session
        .read_file("../secret.txt")
        .await
        .expect_err("read escapes the root");
    assert_eq!(read.code, ErrorCode::InvalidManifestPath);
    assert!(
        read.message.contains("must not escape root"),
        "{}",
        read.message
    );

    client.delete(session).await.expect("deleted");
}

/// Python: `test_unix_local_rm_recursive_ignores_missing_paths`.
#[tokio::test]
async fn a_recursive_rm_ignores_a_missing_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (client, session) = started_local_session(dir.path()).await;

    session
        .rm("missing-dir", true)
        .await
        .expect("a recursive rm of nothing succeeds");

    client.delete(session).await.expect("deleted");
}

/// Python: `test_unix_local_rm_non_recursive_still_errors_for_missing_paths`.
///
/// A non-recursive `rm` of a path that is not there is reported as a non-zero `rm`, so a caller
/// that thought it was removing something finds out it was not.
#[tokio::test]
async fn a_non_recursive_rm_of_a_missing_path_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (client, session) = started_local_session(dir.path()).await;

    let error = session
        .rm("missing-dir", false)
        .await
        .expect_err("Python raises for a plain rm of a missing path");
    assert_eq!(error.code, ErrorCode::ExecNonZero);

    client.delete(session).await.expect("deleted");
}

/// Python: `test_wrapped_unix_local_helpers_reject_symlink_escape_paths`.
#[tokio::test]
async fn mkdir_ls_and_rm_reject_a_path_that_escapes_through_a_symlink() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("workspace");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    let (client, session) = started_local_session(&root).await;

    for error in [
        session
            .mkdir("link/nested", true)
            .await
            .expect_err("mkdir escapes"),
        session.ls("link").await.expect_err("ls escapes"),
        session
            .rm("link/file.txt", false)
            .await
            .expect_err("rm escapes"),
    ] {
        assert_eq!(
            error.code,
            ErrorCode::InvalidManifestPath,
            "{}",
            error.message
        );
        // DEFERRED: Python's message is "... must not escape root"; the unix-local symlink-escape
        // check words it "escapes the workspace root" (`src/sandbox/sandboxes/unix_local.rs`).
        assert!(
            error.message.contains("must not escape root")
                || error.message.contains("escapes the workspace root"),
            "{}",
            error.message
        );
    }
    assert!(!outside.join("nested").exists());

    client.delete(session).await.expect("deleted");
}

/// Python: `test_runner_rejects_unix_local_manifest_user_and_group_provisioning`.
///
/// The unix-local backend cannot create OS accounts, so a manifest that declares one is refused
/// when the session starts.
#[tokio::test]
async fn the_unix_local_backend_refuses_manifest_users() {
    let dir = tempfile::tempdir().expect("tempdir");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new()
                .with_root(posix(dir.path()))
                .with_user(User::new("sandbox-user")),
        ))
        .await
        .expect("created");

    let error = session
        .start()
        .await
        .expect_err("manifest users are refused");
    assert!(
        error
            .message
            .contains("does not support manifest users or groups"),
        "{}",
        error.message
    );
}

// ----------------------------------------------------------------------------------------------
// Mount acknowledgements across a capability's manifest replacement
// ----------------------------------------------------------------------------------------------

/// Adds `cap.txt`, and optionally moves the workspace root, the way a capability that returns a
/// replacement manifest does.
#[derive(Debug)]
struct ReplacementCapability {
    new_root: Option<&'static str>,
}

#[async_trait]
impl Capability<()> for ReplacementCapability {
    fn capability_type(&self) -> &str {
        "manifest-replacement"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        let manifest = manifest.with_entry("cap.txt", Entry::file(b"capability".to_vec()));
        match self.new_root {
            Some(root) => manifest.with_root(root),
            None => manifest,
        }
    }
}

fn credentialed_manifest(acknowledged: &str) -> Manifest {
    let mount = ConfiguredMountStrategy::rclone(
        S3Mount::new("example-bucket").with_credentials("example-access-key", "example-secret-key"),
    )
    .mount();
    Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(mount))
        .with_in_container_mount_credential_exposure_acknowledged([acknowledged])
}

fn data_mount(manifest: &Manifest) -> Mount {
    match &manifest.entries()["data"].kind {
        EntryKind::Mount { mount } => mount.clone(),
        other => panic!("data is not a mount: {other:?}"),
    }
}

/// Python: `test_process_manifest_preserves_mount_acknowledgement_across_replacement`.
#[test]
fn a_manifest_replacement_keeps_the_mount_acknowledgement() {
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_capabilities(vec![Arc::new(ReplacementCapability { new_root: None })]);

    let processed = agent.resolved_manifest(Some(credentialed_manifest("/workspace/data")));

    assert!(processed.entries().contains_key("cap.txt"));
    processed
        .authorize_mount_activation(&data_mount(&processed), "/workspace/data")
        .expect("the acknowledged mount path is still acknowledged");
}

/// Python: `test_process_manifest_preserves_absolute_or_relative_acknowledgement_identity`
/// (the absolute-path case).
///
/// An acknowledgement names an absolute mount path, so it stays on that path when a capability
/// moves the workspace root: the mount is no longer acknowledged at its new location.
///
/// Rust acknowledgements are exact absolute paths only; Python's relative-path acknowledgement
/// (`"data"`, re-resolved under the new root) has no Rust counterpart.
#[test]
fn an_absolute_acknowledgement_does_not_follow_a_moved_workspace_root() {
    let agent = SandboxAgent::<()>::new("sandbox").with_capabilities(vec![Arc::new(
        ReplacementCapability {
            new_root: Some("/other"),
        },
    )]);

    let processed = agent.resolved_manifest(Some(credentialed_manifest("/workspace/data")));

    assert_eq!(processed.root(), "/other");
    let mount = data_mount(&processed);
    let moved = processed
        .authorize_mount_activation(&mount, "/other/data")
        .expect_err("the new location was never acknowledged");
    assert_eq!(moved.code, ErrorCode::MountConfigInvalid);
    processed
        .authorize_mount_activation(&mount, "/workspace/data")
        .expect("the original path stays acknowledged");
}
