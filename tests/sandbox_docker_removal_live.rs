//! The Docker removal service against a real Docker daemon, a real container and the real
//! `docker-removal-worker` executable.
//!
//! Every other removal test drives a recording worker or a recording daemon. These need what the
//! service needs in production — root on the Linux host of the Docker daemon, with that host's
//! process table (`/proc/<pid>` of the sandbox containers) and cgroup paths visible — so they only
//! run when `OPENAI_AGENTS_DOCKER_REMOVAL_LIVE=1` is set and skip (with the reason printed)
//! otherwise. Once that variable is set every requirement is asserted rather than skipped. The
//! image is `OPENAI_AGENTS_DOCKER_LIVE_IMAGE`, defaulting to the SDK's default image; it is never
//! pulled.
//!
//! Each test removes its containers when it ends, including on a failed assertion.
#![cfg(all(feature = "docker-removal", unix))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::errors::SandboxError;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DEFAULT_PYTHON_SANDBOX_IMAGE, DockerSandboxClient, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::sandboxes::docker_removal::{
    DockerRemovalOptions, DockerRemovalService,
};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::User;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;

const LIVE_ENV: &str = "OPENAI_AGENTS_DOCKER_REMOVAL_LIVE";
const IMAGE_ENV: &str = "OPENAI_AGENTS_DOCKER_LIVE_IMAGE";
const TEST_LABEL: &str = "openai-agents-live-test";

/// The image to run, or `None` (with the reason printed) when the live tests must skip.
async fn live_image() -> Option<String> {
    if std::env::var(LIVE_ENV).as_deref() != Ok("1") {
        eprintln!("skipping live Docker removal test: set {LIVE_ENV}=1 to run it");
        return None;
    }
    assert!(
        DockerSandboxClient::is_available().await,
        "{LIVE_ENV}=1 but no Docker daemon answers `docker info`"
    );
    let image = std::env::var(IMAGE_ENV).unwrap_or_else(|_| DEFAULT_PYTHON_SANDBOX_IMAGE.into());
    assert!(
        DockerSandboxClient::image_exists(&image).await,
        "{LIVE_ENV}=1 but image `{image}` is not present locally"
    );
    Some(image)
}

/// A service running the worker this test binary was built with.
fn service(max_entry_visits: u64) -> DockerRemovalService {
    DockerRemovalService::new(
        DockerRemovalOptions::new(4, max_entry_visits, 10)
            .with_worker_program(env!("CARGO_BIN_EXE_docker-removal-worker")),
    )
    .expect("the service starts: this test must run as root on the Docker host")
}

/// Force-removes every container it was told about when the test ends, however it ends.
#[derive(Default)]
struct Containers(Mutex<Vec<String>>);

impl Containers {
    fn track(&self, session: &Arc<dyn SandboxSession>) {
        self.0.lock().unwrap().push(session.id().to_string());
    }
}

impl Drop for Containers {
    fn drop(&mut self) {
        for id in self.0.lock().unwrap().drain(..) {
            // A failed removal can leave the workload paused; `rm -f` removes it either way.
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", "-v", &id])
                .output();
        }
    }
}

/// `docker container inspect` of `id`, or `None` when it does not exist.
fn inspect(id: &str) -> Option<serde_json::Value> {
    let output = std::process::Command::new("docker")
        .args(["container", "inspect", id])
        .output()
        .unwrap();
    if !output.status.success() {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    parsed.as_array()?.first().cloned()
}

/// Every removal, refused or not, leaves the workload running.
fn assert_running_unpaused(session: &Arc<dyn SandboxSession>) {
    let attrs = inspect(session.id()).expect("the container exists");
    assert_eq!(attrs["State"]["Running"], true, "running");
    assert_eq!(attrs["State"]["Paused"], false, "not left paused");
}

/// Run `script` in the sandbox as its default user (root in the default image).
async fn sh(session: &Arc<dyn SandboxSession>, script: &str) -> (i32, String) {
    let result = session
        .exec(ExecRequest::shell(script))
        .await
        .expect("exec");
    (
        result.exit_code,
        format!("{}{}", result.stdout_text(), result.stderr_text()),
    )
}

async fn exists(session: &Arc<dyn SandboxSession>, path: &str) -> bool {
    sh(session, &format!("test -e '{path}' || test -L '{path}'"))
        .await
        .0
        == 0
}

fn context<'a>(error: &'a SandboxError, key: &str) -> Option<&'a str> {
    error.context.get(key).map(String::as_str)
}

async fn create(
    client: &DockerSandboxClient,
    manifest: Manifest,
    containers: &Containers,
) -> Arc<dyn SandboxSession> {
    let session = client
        .create(Some(manifest))
        .await
        .expect("create and bind");
    containers.track(&session);
    session.start().await.expect("start");
    session
}

/// Recursive removal through the service: what it removes, what it refuses, and as whom.
///
/// Python: `test_recursive_removal_preserves_unrelated_writable_trees`,
/// `test_recursive_removal_preserves_live_binding_roots`,
/// `test_missing_target_still_checks_the_requested_user`,
/// `test_removal_preserves_protected_literal_backslash_tree`,
/// `test_worker_removes_deep_tree_beyond_path_max_with_bounded_descriptors` and
/// `test_delete_releases_authority_only_after_confirmed_container_removal`, on a real container.
#[tokio::test]
async fn live_removal_service_removes_refuses_and_releases() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let service = service(100_000);
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(&image).with_label(TEST_LABEL, "1"),
    )
    .with_removal_service(service.clone());
    let manifest = Manifest::new().with_root("/workspace");
    let session = create(&client, manifest.clone(), &containers).await;
    assert!(
        service.is_bound(session.id()),
        "bound before it is returned"
    );
    assert_running_unpaused(&session);

    // An ordinary tree goes; its sibling stays.
    let (code, out) = sh(
        &session,
        "mkdir -p build/obj/deep keep && echo o > build/obj/a.o && echo d > build/obj/deep/d \
         && ln -s /etc build/etc-link && echo k > keep/k",
    )
    .await;
    assert_eq!(code, 0, "{out}");
    session.rm("build", true).await.expect("remove build");
    assert!(!exists(&session, "/workspace/build").await);
    assert!(exists(&session, "/workspace/keep/k").await);
    assert!(
        exists(&session, "/etc/passwd").await,
        "a link is removed, never followed"
    );
    assert_running_unpaused(&session);

    // The workspace root itself is a bound root, however it is spelled.
    for root in ["/workspace", ".", "keep/.."] {
        let error = session.rm(root, true).await.unwrap_err();
        assert_eq!(
            context(&error, "reason"),
            Some("docker_removal_bound_root"),
            "{root}: {error:?}"
        );
    }
    assert!(exists(&session, "/workspace/keep/k").await);
    assert_running_unpaused(&session);

    // A missing target is not an error for root, and still checks an unknown account.
    session
        .rm("never/existed", true)
        .await
        .expect("missing is fine");
    let error = session
        .rm_as("never/existed", true, Some(&User::new("ghost")))
        .await
        .unwrap_err();
    assert_eq!(context(&error, "reason"), Some("docker_removal_failed"));
    assert_eq!(
        context(&error, "worker_reason"),
        Some("ValueError"),
        "{error:?}"
    );
    assert_running_unpaused(&session);

    // The removal runs with the requested user's permissions. `nobody` cannot unlink entries of
    // a root-owned read-only directory (EACCES), nor another user's file in a sticky directory
    // (EPERM); root can do both.
    let (code, out) = sh(
        &session,
        "mkdir -p locked/sub && echo x > locked/sub/file && chmod 0555 locked locked/sub \
         && mkdir sticky && chmod 1777 sticky && echo r > sticky/root-owned",
    )
    .await;
    assert_eq!(code, 0, "{out}");
    let nobody = User::new("nobody");
    let error = session
        .rm_as("locked", true, Some(&nobody))
        .await
        .unwrap_err();
    assert_eq!(
        context(&error, "worker_reason"),
        Some("PermissionError"),
        "{error:?}"
    );
    assert_eq!(context(&error, "errno"), Some("13"), "{error:?}");
    assert!(exists(&session, "/workspace/locked/sub/file").await);
    let error = session
        .rm_as("sticky/root-owned", true, Some(&nobody))
        .await
        .unwrap_err();
    assert_eq!(
        context(&error, "worker_reason"),
        Some("PermissionError"),
        "{error:?}"
    );
    assert_eq!(context(&error, "errno"), Some("1"), "{error:?}");
    assert!(exists(&session, "/workspace/sticky/root-owned").await);
    assert_running_unpaused(&session);
    session.rm("locked", true).await.expect("root removes it");
    session
        .rm("sticky/root-owned", true)
        .await
        .expect("root removes it");
    assert!(!exists(&session, "/workspace/locked").await);
    assert!(!exists(&session, "/workspace/sticky/root-owned").await);

    // A canonical target holding a literal backslash is refused before anything is removed,
    // because the policy would read `\..\` as path syntax.
    let (code, out) = sh(
        &session,
        r"mkdir -p '/workspace/odd/\..\workspace/data' && echo s > '/workspace/odd/\..\workspace/data/sentinel' && ln -s '/workspace/odd/\..\workspace' /workspace/link",
    )
    .await;
    assert_eq!(code, 0, "{out}");
    let error = session.rm("link/data", true).await.unwrap_err();
    assert_eq!(
        context(&error, "reason"),
        Some("docker_removal_canonical_path"),
        "{error:?}"
    );
    assert!(exists(&session, r"/workspace/odd/\..\workspace/data/sentinel").await);
    assert_running_unpaused(&session);

    // A tree deeper than PATH_MAX is removed relative to directory descriptors. A shell `cd`
    // resolves an absolute path and stops at PATH_MAX, so the tree is built with relative
    // `chdir` calls instead; the default image has `python3`.
    let (code, out) = sh(
        &session,
        "python3 -c \"import os\nos.mkdir('deep'); os.chdir('deep')\nfor _ in range(30):\n    \
         os.mkdir('d' * 200); os.chdir('d' * 200)\nopen('leaf', 'w').write('leaf')\n\
         print(len('/workspace/deep') + 30 * 201)\"",
    )
    .await;
    assert_eq!(code, 0, "{out}");
    assert!(
        out.trim().parse::<usize>().unwrap() > 4096,
        "deeper than PATH_MAX: {out}"
    );
    session
        .rm("deep", true)
        .await
        .expect("remove the deep tree");
    assert!(!exists(&session, "/workspace/deep").await);
    assert_running_unpaused(&session);

    // Deleting the session removes the container, then releases its authority.
    let id = session.id().to_string();
    client.delete(session).await.expect("delete");
    assert!(inspect(&id).is_none(), "the container is removed");
    assert!(!service.is_bound(&id), "authority released");
    service.close().await.expect("close");
}

/// The entry-visit budget stops a removal with `E2BIG`, leaving a partial tree.
///
/// Python: `test_worker_stops_entry_visits_before_additional_filesystem_work`, on a real tree.
#[tokio::test]
async fn live_removal_service_enforces_the_entry_visit_budget() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let service = service(3);
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(&image).with_label(TEST_LABEL, "1"),
    )
    .with_removal_service(service.clone());
    let session = create(
        &client,
        Manifest::new().with_root("/workspace"),
        &containers,
    )
    .await;
    let (code, out) = sh(
        &session,
        "mkdir wide && i=0 && while [ $i -lt 10 ]; do echo $i > wide/$i; i=$((i+1)); done",
    )
    .await;
    assert_eq!(code, 0, "{out}");

    let error = session.rm("wide", true).await.unwrap_err();
    assert_eq!(
        context(&error, "worker_reason"),
        Some("OSError"),
        "{error:?}"
    );
    assert_eq!(context(&error, "errno"), Some("7"), "E2BIG: {error:?}");
    let (_, left) = sh(&session, "ls wide | wc -l").await;
    let left: usize = left.trim().parse().unwrap();
    assert!(left > 0 && left < 10, "a partial removal: {left} left");
    assert_running_unpaused(&session);

    client.delete(session).await.expect("delete");
    service.close().await.expect("close");
}

/// A read-only host bind mount outside the workspace keeps its protection, and the binding is
/// the one the manifest was created with.
///
/// Python: `test_read_only_host_mount_preserves_workspace_cleanup`, on a real bind mount.
#[tokio::test]
async fn live_removal_service_preserves_a_read_only_host_grant() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    // The host path must be one the daemon resolves; `TMPDIR` points at such a directory.
    let host = tempfile::tempdir().unwrap();
    let toolchain = host.path().join("toolchain");
    std::fs::create_dir_all(toolchain.join("child")).unwrap();
    std::fs::write(toolchain.join("child/tool"), b"protected").unwrap();
    let grant = |source: &std::path::Path| {
        SandboxPathGrant::new("/toolchain")
            .unwrap()
            .read_only(true)
            .with_host_path(source.to_string_lossy())
            .unwrap()
    };
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant(&toolchain));
    let service = service(100_000);
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(&image).with_label(TEST_LABEL, "1"),
    )
    .with_removal_service(service.clone());
    let session = create(&client, manifest.clone(), &containers).await;
    let mounts = inspect(session.id()).unwrap()["Mounts"].clone();
    assert_eq!(mounts[0]["RW"], false, "{mounts}");
    assert_eq!(mounts[0]["Propagation"], "rprivate", "{mounts}");

    let (code, out) = sh(
        &session,
        "mkdir -p build/x && echo b > build/x/b && ln -s /toolchain /workspace/link",
    )
    .await;
    assert_eq!(code, 0, "{out}");
    session.rm("build", true).await.expect("workspace cleanup");
    assert!(!exists(&session, "/workspace/build").await);

    // Neither through a workspace link nor directly can a removal reach into the grant, nor
    // remove the grant root.
    for path in ["link/child", "/toolchain/child", "/toolchain"] {
        let error = session.rm(path, true).await.unwrap_err();
        assert!(
            matches!(
                context(&error, "reason"),
                Some("read_only_extra_path_grant" | "docker_removal_bound_root")
            ),
            "{path}: {error:?}"
        );
    }
    assert_eq!(
        std::fs::read(toolchain.join("child/tool")).unwrap(),
        b"protected"
    );
    assert_running_unpaused(&session);

    // The live binding belongs to the manifest it was created with.
    let replaced = host.path().join("replaced");
    std::fs::create_dir_all(&replaced).unwrap();
    let changed = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant(&replaced));
    let error = service
        .assert_bound(session.id(), &changed)
        .await
        .unwrap_err();
    assert!(
        error.message.contains("original live authority"),
        "{error:?}"
    );
    service
        .assert_bound(session.id(), &manifest)
        .await
        .expect("the original manifest is still bound");

    client.delete(session).await.expect("delete");
    service.close().await.expect("close");
}

/// The service refuses a container it cannot protect before binding it.
///
/// Python: `test_writable_shared_mounts_cannot_acquire_authority` and
/// `test_host_service_rejects_uncontrolled_execution_modes`, against the real daemon.
#[tokio::test]
async fn live_removal_service_refuses_an_ineligible_container() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let service = service(100_000);

    // A writable shared host path is refused before any container work.
    let host = tempfile::tempdir().unwrap();
    let writable = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            SandboxPathGrant::new("/shared")
                .unwrap()
                .with_host_path(host.path().to_string_lossy())
                .unwrap(),
        );
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(&image).with_label(TEST_LABEL, "1"),
    )
    .with_removal_service(service.clone());
    let Err(error) = client.create(Some(writable)).await else {
        panic!("a writable shared host path must be refused");
    };
    assert!(
        error.message.contains("writable shared host paths"),
        "{error:?}"
    );

    // An added capability makes the container ineligible; the client removes what it created.
    let privileged = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(&image)
            .with_label(TEST_LABEL, "1")
            .with_label("openai-agents-live-ineligible", "1")
            .with_create_arg("--cap-add")
            .with_create_arg("SYS_PTRACE"),
    )
    .with_removal_service(service.clone());
    let Err(error) = privileged
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
    else {
        panic!("a container with an added capability must be refused");
    };
    assert!(
        error.message.contains("running private container"),
        "{error:?}"
    );
    let leftover = std::process::Command::new("docker")
        .args([
            "ps",
            "-aq",
            "--filter",
            "label=openai-agents-live-ineligible=1",
        ])
        .output()
        .unwrap();
    let leftover = String::from_utf8_lossy(&leftover.stdout).trim().to_string();
    for id in leftover.lines() {
        containers.0.lock().unwrap().push(id.to_string());
    }
    assert!(leftover.is_empty(), "the refused container was removed");
    service.close().await.expect("close");
}
