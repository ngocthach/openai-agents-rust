//! Port of `tests/sandbox/test_recursive_removal_compatibility.py`, against real local
//! workspaces: a recursive removal must never delete a read-only grant nested in the tree, and a
//! writable grant must take precedence over a read-only ancestor.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openai_agents_sdk::sandbox::runtime_helpers::RuntimeHelperScript;
use openai_agents_sdk::sandbox::session::ExecRequest;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError, SandboxResult};

use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;

fn grant(path: &Path, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path.to_string_lossy())
        .unwrap()
        .read_only(read_only)
}

async fn session(manifest: Manifest) -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");
    (client, session)
}

/// A backend that only offers `exec`: every file operation takes the trait's default path, so
/// symlinks are resolved by the installed resolver inside the "sandbox" (here, the host).
#[derive(Debug)]
struct RemoteSession {
    manifest: Manifest,
    helper_dir: tempfile::TempDir,
}

#[async_trait::async_trait]
impl SandboxSession for RemoteSession {
    fn id(&self) -> &str {
        "remote"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "remote"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
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
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        std::fs::read(path)
            .map_err(|e| SandboxError::new(ErrorCode::WorkspaceReadNotFound, e.to_string()))
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
    async fn ensure_runtime_helper_installed(
        &self,
        helper: &RuntimeHelperScript,
    ) -> SandboxResult<String> {
        // Execute the shipped resolver against disposable filesystem data.
        use std::os::unix::fs::PermissionsExt;
        let path = self.helper_dir.path().join("resolve-workspace-path");
        std::fs::write(&path, helper.content).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        Ok(s(&path))
    }
}

#[derive(Clone, Copy, Debug)]
enum Backend {
    Local,
    Remote,
}

const BACKENDS: [Backend; 2] = [Backend::Local, Backend::Remote];

async fn backend_session(backend: Backend, manifest: Manifest) -> Arc<dyn SandboxSession> {
    match backend {
        Backend::Local => session(manifest).await.1,
        Backend::Remote => Arc::new(RemoteSession {
            manifest,
            helper_dir: tempfile::tempdir().unwrap(),
        }),
    }
}

fn root() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    (temp, root)
}

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Python: `test_recursive_remove_preserves_nested_read_only_grant` (local backend, 3 sets).
#[tokio::test]
async fn recursive_remove_preserves_a_nested_read_only_grant() {
    for (backend, alias) in BACKENDS
        .into_iter()
        .flat_map(|b| ["none", "ancestor", "grant"].map(move |a| (b, a)))
    {
        let (_temp, root) = root();
        let workspace = root.join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let external = root.join("external");
        let target = external.join("tree");
        let protected = target.join("protected");
        std::fs::create_dir_all(&protected).unwrap();
        let sentinel = protected.join("config");
        std::fs::write(&sentinel, b"protected").unwrap();
        let writable = target.join("writable");
        std::fs::create_dir(&writable).unwrap();
        let sibling = writable.join("data");
        std::fs::write(&sibling, b"allowed").unwrap();
        let mut grant_path = protected.clone();
        let mut remove_path = target.clone();
        if alias == "ancestor" {
            let link = workspace.join("link");
            symlink(&external, &link).unwrap();
            remove_path = link.join("tree");
        } else if alias == "grant" {
            grant_path = root.join("protected-alias");
            symlink(&protected, &grant_path).unwrap();
        }
        let session = backend_session(
            backend,
            Manifest::new()
                .with_root(s(&workspace))
                .with_extra_path_grant(grant(&external, false))
                .with_extra_path_grant(grant(&grant_path, true)),
        )
        .await;

        let direct = session.rm(&s(&sentinel), false).await.expect_err("direct");
        assert_eq!(
            direct.context.get("reason").map(String::as_str),
            Some("read_only_extra_path_grant"),
            "{alias}"
        );
        session
            .rm(&s(&remove_path), false)
            .await
            .expect_err("a non-recursive remove of a tree fails");
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"protected");
        let recursive = session
            .rm(&s(&remove_path), true)
            .await
            .err()
            .unwrap_or_else(|| panic!("{alias}: the recursive removal succeeded"));
        assert_eq!(
            recursive.context.get("reason").map(String::as_str),
            Some("read_only_extra_path_grant"),
            "{alias}"
        );
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"protected");
        assert_eq!(std::fs::read(&sibling).unwrap(), b"allowed");

        session.rm(&s(&writable), true).await.unwrap();
        assert!(!writable.exists(), "{alias}");
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"protected");
    }
}

/// Python: `test_recursive_remove_preserves_writable_precedence` (local backend, 3 sets).
#[tokio::test]
async fn recursive_remove_preserves_writable_precedence() {
    for (backend, precedence) in BACKENDS
        .into_iter()
        .flat_map(|b| ["workspace", "first-grant", "nested-writable"].map(move |p| (b, p)))
    {
        let (_temp, root) = root();
        let workspace = root.join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let external = root.join("external");
        let target = if precedence == "workspace" {
            workspace.join("tree")
        } else {
            external.join("tree")
        };
        let child = target.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(child.join("data"), b"allowed").unwrap();
        let mut grants = vec![grant(&external, false), grant(&child, true)];
        if precedence == "first-grant" {
            grants.insert(0, grant(&child, false));
        } else if precedence == "nested-writable" {
            grants = vec![grant(&external, true), grant(&target, false)];
        }
        let mut manifest = Manifest::new().with_root(s(&workspace));
        for g in grants {
            manifest = manifest.with_extra_path_grant(g);
        }
        let session = backend_session(backend, manifest).await;

        session
            .rm(&s(&target), true)
            .await
            .unwrap_or_else(|e| panic!("{precedence}: {e:?}"));

        assert!(!target.exists(), "{precedence}");
        assert!(workspace.exists());
    }
}

/// Python: `test_remote_recursive_remove_unlinks_leaf_alias_only`: removing a symlink to a tree
/// with a protected grant unlinks the link and leaves the tree alone.
#[tokio::test]
async fn remote_recursive_remove_unlinks_a_leaf_alias_only() {
    let (_temp, root) = root();
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let external = root.join("external");
    let protected = external.join("protected");
    std::fs::create_dir_all(&protected).unwrap();
    let sentinel = protected.join("data");
    std::fs::write(&sentinel, b"protected").unwrap();
    let link = workspace.join("link");
    symlink(&external, &link).unwrap();
    let session = backend_session(
        Backend::Remote,
        Manifest::new()
            .with_root(s(&workspace))
            .with_extra_path_grant(grant(&external, false))
            .with_extra_path_grant(grant(&protected, true)),
    )
    .await;

    session.rm(&s(&link), true).await.unwrap();

    assert!(link.symlink_metadata().is_err(), "the link is gone");
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"protected");
}

/// Python: `test_unix_local_start_restores_with_unrelated_read_only_grant`: resuming from a
/// snapshot prunes the stale workspace and hydrates the archive without touching a read-only
/// grant that lives outside it.
#[tokio::test]
async fn a_resume_restore_leaves_an_unrelated_read_only_grant_alone() {
    let (_temp, root) = root();
    let workspace = root.join("workspace");
    let toolchain = root.join("toolchain");
    std::fs::create_dir(&toolchain).unwrap();
    let protected = toolchain.join("config");
    std::fs::write(&protected, b"protected").unwrap();
    let snapshots = root.join("snapshots");
    std::fs::create_dir(&snapshots).unwrap();
    let manifest = Manifest::new()
        .with_root(s(&workspace))
        .with_extra_path_grant(grant(&toolchain, true));
    let client = UnixLocalSandboxClient::new().with_snapshot(SnapshotSpec::Local {
        base_path: snapshots,
    });

    // The first session saves a workspace holding `restored.txt`.
    let first = client.create(Some(manifest.clone())).await.unwrap();
    first.start().await.unwrap();
    first
        .write_file("restored.txt", b"saved workspace")
        .await
        .unwrap();
    openai_agents_sdk::sandbox::session::persist_session_snapshot(first.as_ref())
        .await
        .unwrap();
    let state = first.session_state();
    first.stop().await.unwrap();

    // The workspace now holds something the snapshot does not; the resume must remove it.
    let stale = workspace.join("generated/stale.txt");
    std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
    std::fs::write(&stale, b"old workspace").unwrap();
    let resumed = client.resume(&state, manifest).await.unwrap();
    resumed.start().await.unwrap();

    assert!(!stale.parent().unwrap().exists());
    assert_eq!(
        std::fs::read(workspace.join("restored.txt")).unwrap(),
        b"saved workspace"
    );
    assert_eq!(std::fs::read(&protected).unwrap(), b"protected");
    assert!(resumed.running().await.unwrap());
    client.delete(resumed).await.ok();
}
