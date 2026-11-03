//! Python `tests/sandbox/test_runtime.py` unix-local `delete` cases: mounts are detached while
//! the workspace still exists, and a failed detach leaves the workspace on disk.
#![cfg(feature = "sandbox")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mounts::{
    InContainerMountStrategy, Mount, MountPattern, MountPatternConfig,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError};

/// Records, at detach time, whether the mount path still exists on disk; optionally fails.
#[derive(Debug)]
struct ProbePattern {
    existed: Arc<Mutex<Vec<bool>>>,
    fail: bool,
}

#[async_trait]
impl MountPattern for ProbePattern {
    fn pattern_type(&self) -> &str {
        "probe"
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
        self.existed.lock().unwrap().push(
            PathBuf::from(mount_path)
                .parent()
                .is_some_and(|p| p.exists()),
        );
        if self.fail {
            return Err(SandboxError::new(
                ErrorCode::WorkspaceStopError,
                "unmount failed",
            ));
        }
        Ok(())
    }
}

async fn session_with_probe_mount(
    fail: bool,
) -> (
    Arc<dyn SandboxSession>,
    UnixLocalSandboxClient,
    String,
    Arc<Mutex<Vec<bool>>>,
) {
    let existed = Arc::new(Mutex::new(Vec::new()));
    let client = UnixLocalSandboxClient::new();
    let mount = Mount::new(Arc::new(InContainerMountStrategy::new(
        Arc::new(ProbePattern {
            existed: existed.clone(),
            fail,
        }),
        MountPatternConfig::new(),
    )));
    let manifest = Manifest::new().with_entry("remote", Entry::mount(mount));
    let session = client.create(Some(manifest)).await.expect("created");
    let root = session.workspace_root().to_string();
    session.start().await.expect("started");
    (session, client, root, existed)
}

/// Python: `test_unix_local_client_delete_unmounts_workspace_mounts_before_rmtree`.
#[tokio::test]
async fn delete_detaches_mounts_while_the_workspace_still_exists_then_removes_it() {
    let (session, client, root, existed) = session_with_probe_mount(false).await;
    assert!(std::path::Path::new(&root).exists());

    client.delete(session).await.expect("deleted");

    assert_eq!(existed.lock().unwrap().clone(), vec![true]);
    assert!(!std::path::Path::new(&root).exists());
}

/// Python: `test_unix_local_client_delete_skips_rmtree_when_unmount_fails` (the on-disk half).
/// Removing the root over a mount that could not be detached would delete the mounted storage.
#[tokio::test]
async fn delete_keeps_the_workspace_when_a_mount_cannot_be_detached() {
    let (session, client, root, existed) = session_with_probe_mount(true).await;

    let outcome = client.delete(session).await;

    assert!(outcome.is_err(), "the failed detach surfaces");
    assert!(!existed.lock().unwrap().is_empty());
    assert!(std::path::Path::new(&root).exists());
    std::fs::remove_dir_all(&root).ok();
}
