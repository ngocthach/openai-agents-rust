//! Port of `tests/sandbox/test_unix_local_file_io.py::
//! test_injected_session_uses_current_grants_without_rebinding_aliases`: a capability that
//! changes the extra path grants of an injected session takes effect on it, while an alias a
//! grant was declared through stays pinned to the directory it named first.
#![cfg(all(feature = "sandbox", unix))]

use std::path::Path;
use std::sync::Arc;

use openai_agents_sdk::sandbox::capabilities::Capability;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxAgent, SandboxRunConfig, SandboxRunner};

#[derive(Debug)]
struct ConfigureGrants {
    grants: Vec<SandboxPathGrant>,
}

impl Capability<()> for ConfigureGrants {
    fn capability_type(&self) -> &str {
        "configure_grants"
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        manifest.with_extra_path_grants(self.grants.clone())
    }
}

fn posix(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn grant(path: &Path, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(posix(path))
        .expect("an absolute path")
        .read_only(read_only)
}

async fn configure(session: &Arc<dyn SandboxSession>, grants: Vec<SandboxPathGrant>) {
    let agent = SandboxAgent::<()>::new("File grant lifecycle")
        .with_capabilities(vec![Arc::new(ConfigureGrants { grants })]);
    let config = SandboxRunConfig::with_session(session.clone());
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("the capability's grants are accepted");
    assert!(Arc::ptr_eq(prepared.session(), session));
}

#[tokio::test]
async fn an_injected_session_uses_the_grants_a_capability_changed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (workspace, granted, added, outside) = (
        base.join("workspace"),
        base.join("granted"),
        base.join("added"),
        base.join("outside"),
    );
    for directory in [&workspace, &granted, &added, &outside] {
        std::fs::create_dir(directory).expect("mkdir");
    }
    std::fs::write(outside.join("file"), b"original content").expect("seed");
    let alias = base.join("alias");
    std::os::unix::fs::symlink(&granted, &alias).expect("alias");

    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new()
                .with_root(posix(&workspace))
                .with_extra_path_grant(grant(&alias, true)),
        ))
        .await
        .expect("created");
    session.start().await.expect("started");

    configure(&session, vec![grant(&alias, false), grant(&added, false)]).await;
    session
        .write_file(&posix(&alias.join("file")), b"upgraded")
        .await
        .expect("the alias grant became writable");
    session
        .mkdir(&posix(&added.join("directory")), false)
        .await
        .expect("mkdir under the added grant");
    session
        .write_file(&posix(&added.join("directory/file")), b"added")
        .await
        .expect("write under the added grant");
    assert_eq!(
        session
            .read_file(&posix(&added.join("directory/file")))
            .await
            .expect("read"),
        b"added"
    );
    assert_eq!(
        std::fs::read(granted.join("file")).expect("granted file"),
        b"upgraded"
    );
    session
        .rm(&posix(&added.join("directory")), true)
        .await
        .expect("remove under the added grant");

    std::fs::remove_file(&alias).expect("unlink alias");
    std::os::unix::fs::symlink(&outside, &alias).expect("re-point alias");
    configure(&session, vec![grant(&alias, false), grant(&added, true)]).await;
    let repointed = session
        .write_file(&posix(&alias.join("file")), b"new")
        .await
        .expect_err("a re-pointed alias widens nothing");
    assert_eq!(repointed.code, ErrorCode::InvalidManifestPath);
    let read_only = session
        .write_file(&posix(&added.join("file")), b"new")
        .await
        .expect_err("the grant is read-only now");
    assert_eq!(read_only.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        std::fs::read(outside.join("file")).expect("outside file"),
        b"original content"
    );

    configure(&session, Vec::new()).await;
    let removed = session
        .ls(&posix(&added))
        .await
        .expect_err("the grant is gone");
    assert_eq!(removed.code, ErrorCode::InvalidManifestPath);
}
