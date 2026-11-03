//! Ports of the behaviour behind two cases of `tests/sandbox/test_docker_removal_worker.py`,
//! against a real local workspace: the Docker worker's descriptor and namespace plumbing has no
//! Rust counterpart, but what a user sees (a recursive removal that must not reach a protected
//! tree, and a read-only grant nested in the workspace) is asserted here through the Rust API.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::Arc;

use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;

fn grant(path: &Path, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path.to_string_lossy())
        .unwrap()
        .read_only(read_only)
}

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

async fn session(manifest: Manifest) -> Arc<dyn SandboxSession> {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");
    session
}

/// Python: `test_removal_preserves_protected_literal_backslash_tree`. A workspace link into a
/// read-only grant whose path holds literal backslashes (`\..\workspace`) is never read as a
/// parent traversal back into the workspace: removing through the link is refused and the
/// protected tree survives.
#[tokio::test]
async fn a_literal_backslash_path_in_a_read_only_grant_is_not_a_traversal() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let protected = root.join("protected");
    let literal = protected.join(["\\", "..", "\\", "workspace"].concat());
    let target = literal.join("data");
    std::fs::create_dir_all(&target).unwrap();
    let sentinel = target.join("sentinel.txt");
    std::fs::write(&sentinel, "protected contents").unwrap();
    symlink(&literal, workspace.join("link")).unwrap();

    let session = session(
        Manifest::new()
            .with_root(s(&workspace))
            .with_extra_path_grant(grant(&protected, true)),
    )
    .await;

    let error = session
        .rm("link/data", true)
        .await
        .expect_err("the protected tree is read-only");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant"),
        "{error:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "protected contents"
    );
    assert!(workspace.join("link").is_symlink());
}

/// Python: `test_workspace_overridden_grant_does_not_invalidate_later_removal` (the non-resume
/// variant). A read-only grant whose path lies inside the workspace is overridden by the
/// workspace, so it can be removed like any other workspace tree, and a removal after it still
/// works. A read-only grant outside the workspace stays protected.
#[tokio::test]
async fn a_read_only_grant_inside_the_workspace_does_not_block_later_removals() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let workspace = root.join("workspace");
    let cache = workspace.join("cache");
    let output = workspace.join("output");
    let protected = root.join("external-protected");
    for dir in [&cache, &output, &protected] {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("file"), "x").unwrap();
    }

    let session = session(
        Manifest::new()
            .with_root(s(&workspace))
            .with_extra_path_grant(grant(&cache, true))
            .with_extra_path_grant(grant(&protected, true)),
    )
    .await;

    session.rm("cache", true).await.unwrap();
    session.rm("output", true).await.unwrap();
    assert!(!cache.exists() && !output.exists());

    session
        .rm(&s(&protected), true)
        .await
        .expect_err("the grant outside the workspace is still protected");
    assert!(protected.join("file").exists());
}
