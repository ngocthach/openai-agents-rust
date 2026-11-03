//! Ports of the remaining workspace, resume-authority and mount tests in
//! `tests/sandbox/test_docker.py`: root-level mount persistence, the persist skip-path registry,
//! leaf-symlink removal, resume under rebound or tampered mount authority, clearing the workspace
//! root on resume, and the S3 Files volume-driver refusal.
//!
//! Every Docker call goes through a recording `DockerRunner`; no daemon is involved.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3FilesMount, S3Mount};
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy, docker_volume_name};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DOCKER_SESSION_ID_KEY, DockerRunner, DockerSandboxClient, DockerSandboxSession,
};
use openai_agents_sdk::sandbox::session::clear_workspace_root_on_resume;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxResult, SandboxSession,
};

// ----------------------------------------------------------------------------------------------
// Fakes
// ----------------------------------------------------------------------------------------------

struct FakeDocker {
    calls: Mutex<Vec<Vec<String>>>,
    #[allow(clippy::type_complexity)]
    responder: Box<dyn Fn(&[String]) -> Option<ExecResult> + Send + Sync>,
}

impl std::fmt::Debug for FakeDocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeDocker").finish_non_exhaustive()
    }
}

impl FakeDocker {
    /// A runner that answers from `responder`, falling back to [`default_response`].
    fn scripted(
        responder: impl Fn(&[String]) -> Option<ExecResult> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            responder: Box::new(responder),
        })
    }

    fn plain() -> Arc<Self> {
        FakeDocker::scripted(|_| None)
    }

    fn joined(&self) -> Vec<String> {
        self.calls
            .lock()
            .expect("calls")
            .iter()
            .map(|argv| argv.join(" "))
            .collect()
    }
}

#[async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(argv.to_vec());
        Ok((self.responder)(argv).unwrap_or_else(|| default_response(argv)))
    }
}

fn out(stdout: &str) -> ExecResult {
    ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

fn fail(exit_code: i32, stderr: &str) -> ExecResult {
    ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code,
    }
}

fn is_helper(argv: &[String]) -> bool {
    argv.iter()
        .any(|arg| arg.contains("resolve-workspace-path-"))
}

/// `docker create` names a container, an inspect of anything fails, the path helper answers with
/// the candidate it was given, and the rest is a silent success.
fn default_response(argv: &[String]) -> ExecResult {
    if let Some(at) = argv
        .iter()
        .position(|arg| arg.contains("resolve-workspace-path-"))
    {
        return match argv.get(at + 2) {
            Some(candidate) => out(candidate),
            None => out(""),
        };
    }
    match argv.first().map(String::as_str) {
        Some("create") => out("container-1\n"),
        Some("container") | Some("inspect") => fail(1, "Error: No such object"),
        _ => out(""),
    }
}

fn session(manifest: Manifest, runner: Arc<FakeDocker>) -> DockerSandboxSession {
    DockerSandboxSession::new("container-under-test", manifest).with_runner(runner)
}

/// A credentialed S3 mount attached with a Docker volume driver: Python's `S3Mount` with access
/// keys and a `DockerVolumeMountStrategy`. The credential goes to the container runtime on the
/// host, so no acknowledgement is needed, but the mount still carries a secret that persisted
/// state and error text must never reproduce.
fn authority_volume() -> Mount {
    ConfiguredMountStrategy::docker_volume(
        S3Mount::new("example-bucket").with_credentials("example-access-key", "example-secret-key"),
        "rclone",
    )
    .mount()
}

fn authority_manifest() -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(authority_volume()))
}

/// The state a credentialed manifest persists: the provider identity is dropped.
fn persisted_state(manifest: &Manifest) -> SandboxSessionState {
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", manifest)
        .with_provider_id(Some("surviving-container".to_string()))
        .with_workspace_root_ready(true);
    SandboxSessionState::from_json_str(&state.to_json_string().expect("serialize")).expect("parse")
}

fn creates(runner: &FakeDocker) -> Vec<String> {
    runner
        .joined()
        .into_iter()
        .filter(|call| call.starts_with("create "))
        .collect()
}

fn inspected(runner: &FakeDocker) -> bool {
    runner
        .joined()
        .iter()
        .any(|call| call.starts_with("container inspect") || call.starts_with("inspect"))
}

/// A tar of `names` (regular empty files), as the container's `tar -czf -` would write it.
fn gz_tar(names: &[&str]) -> Vec<u8> {
    use std::io::Write as _;
    let mut builder = tar::Builder::new(Vec::new());
    for name in names {
        let mut header = tar::Header::new_gnu();
        header.set_size(0);
        header.set_mode(0o644);
        builder
            .append_data(&mut header, name, std::io::empty())
            .expect("member");
    }
    let raw = builder.into_inner().expect("tar");
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&raw).expect("gzip");
    encoder.finish().expect("gzip")
}

// ----------------------------------------------------------------------------------------------
// Persist
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_persist_workspace_skips_workspace_root_mount_without_traversing_remote_data`.
/// A mount at the workspace root makes the whole workspace remote storage, so the capture is
/// non-recursive and never reads the remote files.
#[tokio::test]
async fn persist_does_not_traverse_a_mount_at_the_workspace_root() {
    let runner = FakeDocker::scripted(|argv| {
        argv.iter()
            .any(|arg| arg.contains("tar -czf -"))
            .then(|| ExecResult {
                stdout: gz_tar(&["."]),
                stderr: Vec::new(),
                exit_code: 0,
            })
    });
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "root-mount",
        Entry::mount(authority_volume().with_mount_path("/workspace")),
    );
    let session = session(manifest, runner.clone());

    session.persist_workspace_once().await.expect("persist");

    let command = runner
        .joined()
        .into_iter()
        .find(|call| call.contains("tar -czf -"))
        .expect("the capture command");
    assert!(command.contains("--no-recursion"), "{command}");
    assert!(command.ends_with(" ."), "{command}");
}

/// Python: `test_docker_persist_workspace_prunes_runtime_only_skip_paths_from_staged_copy`. A path
/// registered at runtime is left out of the archive, beyond the manifest's own ephemeral entries.
#[tokio::test]
async fn persist_excludes_a_runtime_registered_skip_path() {
    let runner = FakeDocker::scripted(|argv| {
        argv.iter()
            .any(|arg| arg.contains("tar -czf -"))
            .then(|| ExecResult {
                stdout: gz_tar(&["logs/keep.txt"]),
                stderr: Vec::new(),
                exit_code: 0,
            })
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let registered = session
        .register_persist_workspace_skip_path("logs/events.jsonl")
        .expect("register");
    assert_eq!(registered, "logs/events.jsonl");
    assert_eq!(
        session.persist_workspace_skip_paths(),
        vec!["logs/events.jsonl".to_string()]
    );

    session.persist_workspace_once().await.expect("persist");

    let command = runner
        .joined()
        .into_iter()
        .find(|call| call.contains("tar -czf -"))
        .expect("the capture command");
    assert!(
        command.contains("--exclude ./logs/events.jsonl"),
        "{command}"
    );
    assert!(!command.contains("--exclude ./logs/keep.txt"), "{command}");
}

// ----------------------------------------------------------------------------------------------
// File operations through the in-container resolver
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_rm_unlinks_safe_internal_leaf_symlink`. Removing a link removes the link,
/// not the file it points at: the command names the *requested* path, not the realpath.
#[tokio::test]
async fn rm_unlinks_a_safe_leaf_symlink_instead_of_its_target() {
    // The helper resolves the link to its target, as the real one does; `rm` must not use it.
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) && argv.iter().any(|a| a == "/workspace/link.txt") {
            return Some(out("/workspace/target.txt\n"));
        }
        None
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    session.rm("link.txt", false).await.expect("rm");

    let removal = runner
        .joined()
        .into_iter()
        .find(|call| call.contains(" rm "))
        .expect("an rm call");
    assert!(
        removal.ends_with("rm -f -- /workspace/link.txt"),
        "{removal}"
    );
    assert!(!removal.contains("target.txt"), "{removal}");
}

/// Python: `test_docker_workspace_file_ops_reject_symlink_escape` — the `ls`, `mkdir` and `rm`
/// halves (read and write are in `py_sandbox_a_docker_2.rs`). Every file operation goes through
/// the resolver, so none of them follows a link out of the workspace.
#[tokio::test]
async fn ls_mkdir_rm_and_mv_refuse_a_symlink_that_escapes_the_workspace() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) && argv.iter().any(|a| a.starts_with("/workspace/")) {
            return Some(fail(111, "workspace escape"));
        }
        None
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let ls = session.ls("link").await.expect_err("ls");
    let mkdir = session.mkdir("link/newdir", true).await.expect_err("mkdir");
    let rm = session.rm("link/secret.txt", false).await.expect_err("rm");
    let recursive = session
        .rm("link/secret.txt", true)
        .await
        .expect_err("rm -rf");
    let mv = session
        .mv("link/secret.txt", "moved.txt")
        .await
        .expect_err("mv");
    for error in [ls, mkdir, rm, recursive, mv] {
        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "{error}");
        assert!(error.message.contains("must not escape root"), "{error}");
    }
    // Nothing reached the container beyond the helper.
    assert!(
        !runner.joined().iter().any(|call| call.contains(" ls ")
            || call.contains(" mkdir ")
            || call.contains(" rm ")
            || call.contains("mv -fT")),
        "{:?}",
        runner.joined()
    );
}

// ----------------------------------------------------------------------------------------------
// Resume under mount authority
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_resume_applies_current_authority_with_fresh_volume_identity`. A state that
/// carried mount credentials is resumed into a *new* container under a *new* volume identity, with
/// the trusted manifest's current authority; the stale volume the old identity named is untouched.
#[tokio::test]
async fn resume_applies_current_authority_under_a_fresh_volume_identity() {
    let runner = FakeDocker::plain();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = authority_manifest();
    let persisted = persisted_state(&manifest);
    assert_eq!(persisted.provider_id, None);

    let resumed = client.resume(&persisted, manifest).await.expect("resume");

    let create = creates(&runner);
    assert_eq!(create.len(), 1, "{:?}", runner.joined());
    let new_session_id = resumed
        .session_state()
        .provider_metadata(DOCKER_SESSION_ID_KEY)
        .and_then(serde_json::Value::as_str)
        .expect("session id")
        .to_string();
    assert_ne!(new_session_id, "sess-1");
    assert_ne!(new_session_id, persisted.session_id);
    assert!(
        create[0].contains(&format!(
            "src={}",
            docker_volume_name(Some(&new_session_id), "/workspace/data")
        )),
        "{}",
        create[0]
    );
    let stale = docker_volume_name(Some(&persisted.session_id), "/workspace/data");
    assert!(
        !runner
            .joined()
            .iter()
            .any(|call| call.starts_with("volume rm") && call.contains(&stale)),
        "{:?}",
        runner.joined()
    );
    assert!(!resumed.session_state().workspace_root_ready);
}

/// Python: `test_docker_resume_recreates_rebound_authority_for_existing_container`. Even when the
/// old container is still there, rebound authority gets a fresh one: the container was attached
/// under credentials this process no longer vouches for.
#[tokio::test]
async fn resume_recreates_the_container_when_authority_was_rebound() {
    let runner = FakeDocker::scripted(|argv| {
        // An old container that would answer, if anyone asked.
        (argv.first().map(String::as_str) == Some("container")).then(|| out("[{}]"))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = authority_manifest();
    let persisted = persisted_state(&manifest);
    let rebound = persisted.rebind_path_grants(&manifest).expect("rebind");
    assert!(rebound.mount_authority_rebound);

    let resumed = client.resume(&rebound, manifest).await.expect("resume");

    assert_eq!(creates(&runner).len(), 1, "{:?}", runner.joined());
    assert!(!inspected(&runner), "{:?}", runner.joined());
    assert_eq!(resumed.id(), "container-1");
}

/// Python: `test_docker_resume_recreates_direct_state_with_configured_authority`. A state built
/// directly from a credentialed manifest (never serialized) is not trusted to reattach either.
#[tokio::test]
async fn resume_recreates_a_directly_built_state_with_configured_authority() {
    let runner = FakeDocker::scripted(|argv| {
        (argv.first().map(String::as_str) == Some("container")).then(|| out("[{}]"))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = authority_manifest();
    let direct = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("existing-container".to_string()))
        .with_workspace_root_ready(true);

    let resumed = client.resume(&direct, manifest).await.expect("resume");

    assert!(!inspected(&runner), "{:?}", runner.joined());
    assert_eq!(creates(&runner).len(), 1, "{:?}", runner.joined());
    assert_eq!(resumed.id(), "container-1");
    assert!(!resumed.session_state().workspace_root_ready);
}

/// Python: `test_docker_resume_does_not_reconnect_identity_from_tampered_state`. A hand-edited
/// state that puts a surviving container's id back must not reattach to it.
#[tokio::test]
async fn resume_does_not_reconnect_an_identity_injected_into_a_persisted_state() {
    let manifest = authority_manifest();
    let persisted = persisted_state(&manifest);
    let mut payload: serde_json::Value =
        serde_json::from_str(&persisted.to_json_string().expect("json")).expect("json");
    payload["provider_id"] = serde_json::json!("surviving-container");
    let tampered = SandboxSessionState::from_json_str(&payload.to_string()).expect("parse");
    assert_eq!(tampered.provider_id.as_deref(), Some("surviving-container"));

    let runner = FakeDocker::scripted(|argv| {
        (argv.first().map(String::as_str) == Some("container")).then(|| out("[{}]"))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());

    let resumed = client.resume(&tampered, manifest).await.expect("resume");

    assert_ne!(resumed.id(), "surviving-container");
    assert!(!inspected(&runner), "{:?}", runner.joined());
    assert_eq!(creates(&runner).len(), 1, "{:?}", runner.joined());
}

// ----------------------------------------------------------------------------------------------
// Clearing the workspace root on resume
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_clear_workspace_root_on_resume_preserves_nested_docker_volume_mounts`.
/// A mount nested in a directory is skipped (its files are remote storage), the directory above it
/// is walked rather than removed, and every stale sibling goes.
#[tokio::test]
async fn clearing_the_workspace_on_resume_preserves_a_nested_volume_mount() {
    let runner = FakeDocker::scripted(|argv| {
        let joined = argv.join(" ");
        if joined.ends_with("ls -la -- /workspace") {
            return Some(out("total 8\n\
                 drwxr-xr-x 3 root root 4096 Jan  1 00:00 a\n\
                 -rw-r--r-- 1 root root    0 Jan  1 00:00 root.txt\n"));
        }
        if joined.ends_with("ls -la -- /workspace/a") {
            return Some(out("total 8\n\
                 drwxr-xr-x 2 root root 4096 Jan  1 00:00 b\n\
                 -rw-r--r-- 1 root root    0 Jan  1 00:00 local.txt\n"));
        }
        None
    });
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "a/b",
        Entry::mount(Mount::new(Arc::new(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone"),
        ))),
    );
    let session = session(manifest, runner.clone());

    clear_workspace_root_on_resume(&session)
        .await
        .expect("clear");

    let joined = runner.joined();
    let listed: Vec<&String> = joined
        .iter()
        .filter(|call| call.contains(" ls -la "))
        .collect();
    assert_eq!(listed.len(), 2, "{joined:?}");
    let removed: Vec<&String> = joined
        .iter()
        .filter(|call| call.contains(" rm -rf "))
        .collect();
    assert_eq!(removed.len(), 2, "{joined:?}");
    assert!(
        removed
            .iter()
            .any(|call| call.ends_with("/workspace/a/local.txt")),
        "{joined:?}"
    );
    assert!(
        removed
            .iter()
            .any(|call| call.ends_with("/workspace/root.txt")),
        "{joined:?}"
    );
    // The mount itself, and the directory that holds it, are left alone.
    assert!(
        !removed
            .iter()
            .any(|call| call.ends_with("/workspace/a/b") || call.ends_with("/workspace/a")),
        "{joined:?}"
    );
}

// ----------------------------------------------------------------------------------------------
// Mount configuration
// ----------------------------------------------------------------------------------------------

/// Python: `test_s3_files_mount_rejects_docker_volume_mount`. Python raises while constructing the
/// mount; here the strategy is refused when the mount is validated, which `create` does before
/// any `docker` call.
#[tokio::test]
async fn an_s3_files_mount_cannot_use_a_docker_volume_driver() {
    let mount = Mount::new(Arc::new(ConfiguredMountStrategy::docker_volume(
        S3FilesMount {
            file_system_id: "fs-1234567890abcdef0".to_string(),
            ..Default::default()
        },
        "rclone",
    )));
    let error = mount.validate().expect_err("the driver is refused");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("Docker volume driver"), "{error}");

    let runner = FakeDocker::plain();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let create_error = match client
        .create(Some(
            Manifest::new().with_entry("data", Entry::mount(mount)),
        ))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("create must refuse the manifest"),
    };
    assert_eq!(create_error.code, ErrorCode::MountConfigInvalid);
    assert!(creates(&runner).is_empty(), "{:?}", runner.joined());
}

/// A strategy the SDK does not ship may hide credentials, so a manifest holding one is protected
/// even though nothing about it reports a credential (Python's `_mount_class_is_trusted`).
#[test]
fn a_mount_strategy_the_sdk_does_not_ship_is_treated_as_protected() {
    use openai_agents_sdk::sandbox::sandboxes::docker::manifest_has_protected_mounts;

    #[derive(Debug)]
    struct Foreign;

    #[async_trait]
    impl MountStrategy for Foreign {
        fn strategy_type(&self) -> &str {
            "third_party_mount"
        }

        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> SandboxResult<()> {
            Ok(())
        }

        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> SandboxResult<()> {
            Ok(())
        }
    }

    let foreign = Manifest::new().with_entry("x", Entry::mount(Mount::new(Arc::new(Foreign))));
    assert!(manifest_has_protected_mounts(&foreign));
    // A credential-free volume of a shipped strategy is not.
    let plain = Manifest::new().with_entry(
        "x",
        Entry::mount(Mount::new(Arc::new(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone"),
        ))),
    );
    assert!(!manifest_has_protected_mounts(&plain));
}

/// Python: `DockerSandboxClient.create(snapshot=...)`. The snapshot the runner passes is the one
/// the session persists to; the default `create` stores nothing.
#[tokio::test]
async fn create_with_snapshot_stores_the_workspace_snapshot_the_runner_passes() {
    use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;

    let runner = FakeDocker::plain();
    let client = DockerSandboxClient::new().with_runner(runner);

    let plain = client.create(None).await.expect("create");
    assert_eq!(plain.snapshot().snapshot_type(), "noop");

    let base = std::env::temp_dir().join("py-sandbox-a-docker-snapshots");
    let stored = client
        .create_with_snapshot(None, SnapshotSpec::Local { base_path: base })
        .await
        .expect("create_with_snapshot");
    assert_eq!(stored.snapshot().snapshot_type(), "local");
    assert_eq!(
        stored.session_state().snapshot,
        SnapshotSpec::Local {
            base_path: std::env::temp_dir().join("py-sandbox-a-docker-snapshots")
        }
    );
}

/// Python: `_set_start_state_preserved(reused_existing_container)`. A reattached container keeps
/// its accounts, so the resume must not provision them again; a replacement must.
#[tokio::test]
async fn system_state_is_preserved_only_when_the_container_was_reused() {
    let live = FakeDocker::scripted(|argv| {
        (argv.first().map(String::as_str) == Some("container")).then(|| {
            out(r#"[{"HostConfig":{"NetworkMode":"bridge"},"NetworkSettings":{"Networks":{"bridge":{}},"Ports":{}},"Mounts":[],"Config":{"Labels":{}}}]"#)
        })
    });
    let manifest = Manifest::new().with_root("/workspace");
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container".to_string()));

    let reused = DockerSandboxClient::new()
        .with_runner(live)
        .resume(&state, manifest.clone())
        .await
        .expect("resume");
    assert_eq!(reused.id(), "container");
    assert!(reused.system_state_preserved_on_start());

    let replaced = DockerSandboxClient::new()
        .with_runner(FakeDocker::plain())
        .resume(&state, manifest)
        .await
        .expect("resume");
    assert_eq!(replaced.id(), "container-1");
    assert!(!replaced.system_state_preserved_on_start());
}

/// Python: `_set_start_state_preserved(reused_existing_container)` as seen from `start`. Starting
/// a resumed session restores the snapshot and then provisions the manifest's accounts only when
/// the container is a replacement: a reattached container still has them, and `useradd` for an
/// existing account fails.
#[tokio::test]
async fn resume_provisions_accounts_only_into_a_replacement_container() {
    use openai_agents_sdk::sandbox::snapshot::{SnapshotSpec, resolve_snapshot};

    let base = tempfile::tempdir().expect("snapshot dir");
    let spec = SnapshotSpec::Local {
        base_path: base.path().to_path_buf(),
    };
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_user("sandbox-user");
    let mut state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container".to_string()))
        .with_snapshot(spec.clone());
    state.snapshot_id = "snap-1".to_string();
    resolve_snapshot(Some(&spec), "snap-1")
        .persist(&gz_tar(&["notes.txt"]))
        .await
        .expect("persist the snapshot");

    let useradds = |runner: &FakeDocker| {
        runner
            .joined()
            .into_iter()
            .filter(|call| call.contains("useradd"))
            .count()
    };

    let live = FakeDocker::scripted(|argv| {
        (argv.first().map(String::as_str) == Some("container")).then(|| {
            out(r#"[{"HostConfig":{"NetworkMode":"bridge"},"NetworkSettings":{"Networks":{"bridge":{}},"Ports":{}},"Mounts":[],"Config":{"Labels":{}}}]"#)
        })
    });
    let reused = DockerSandboxClient::new()
        .with_runner(live.clone())
        .resume(&state, manifest.clone())
        .await
        .expect("resume");
    reused.start().await.expect("start");
    assert_eq!(reused.id(), "container");
    assert_eq!(useradds(&live), 0, "{:?}", live.joined());

    let replacement = FakeDocker::plain();
    let replaced = DockerSandboxClient::new()
        .with_runner(replacement.clone())
        .resume(&state, manifest)
        .await
        .expect("resume");
    replaced.start().await.expect("start");
    assert_eq!(replaced.id(), "container-1");
    assert_eq!(useradds(&replacement), 1, "{:?}", replacement.joined());
}
