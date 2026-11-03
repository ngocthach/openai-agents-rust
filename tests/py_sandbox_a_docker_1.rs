//! Ports of the container-creation, label, volume and deletion tests in
//! `tests/sandbox/test_docker.py`.
//!
//! Python fakes `docker-py`: it hands `DockerSandboxClient` a fake client object and asserts on
//! the keyword arguments that reach `containers.create(...)`. This port shells out to the
//! `docker` CLI instead, so the equivalent observable is the argument vector that reaches the
//! executable. `DockerRunner` is the seam for that — a fake runner records every `argv` and the
//! bytes piped into it, and no daemon is involved.

#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::{Mount, docker_volume_name};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DEFAULT_PYTHON_SANDBOX_IMAGE, DOCKER_LABELS_KEY, DOCKER_NETWORK_DISABLED_KEY,
    DOCKER_SESSION_ID_KEY, DockerRunner, DockerSandboxClient, DockerSandboxClientOptions,
    DockerSandboxSession, docker_volume_names_for_manifest,
};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxResult, SandboxSession,
};

// ----------------------------------------------------------------------------------------------
// The fake `docker` executable
// ----------------------------------------------------------------------------------------------

/// One recorded `docker` invocation.
#[derive(Debug, Clone)]
struct DockerCall {
    argv: Vec<String>,
}

/// A [`DockerRunner`] that records every invocation and answers from a script.
struct FakeDocker {
    calls: Mutex<Vec<DockerCall>>,
    #[allow(clippy::type_complexity)]
    responder: Box<dyn Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync>,
}

impl std::fmt::Debug for FakeDocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeDocker").finish_non_exhaustive()
    }
}

impl FakeDocker {
    /// A runner that answers every command successfully, naming the created container.
    fn new() -> Arc<Self> {
        FakeDocker::scripted(|argv| Ok(default_response(argv)))
    }

    /// A runner answering from `responder`.
    fn scripted(
        responder: impl Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            responder: Box::new(responder),
        })
    }

    /// Every recorded invocation.
    fn calls(&self) -> Vec<DockerCall> {
        self.calls.lock().expect("calls").clone()
    }

    /// Every recorded invocation as one space-joined string, for substring assertions.
    fn joined(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .map(|call| call.argv.join(" "))
            .collect()
    }

    /// The first invocation whose first argument is `verb`.
    fn first(&self, verb: &str) -> Vec<String> {
        self.calls()
            .into_iter()
            .map(|call| call.argv)
            .find(|argv| argv.first().map(String::as_str) == Some(verb))
            .unwrap_or_else(|| panic!("no `docker {verb}` call in {:?}", self.joined()))
    }
}

#[async_trait::async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(DockerCall {
            argv: argv.to_vec(),
        });
        (self.responder)(argv)
    }
}

/// A successful result carrying `stdout`.
fn out(stdout: &str) -> ExecResult {
    ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

/// A failed result carrying `stderr`.
fn fail(exit_code: i32, stderr: &str) -> ExecResult {
    ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code,
    }
}

/// `docker create` names a container; an inspect of a container that is gone fails; the rest is
/// a silent success, which is what the Python fakes do.
fn default_response(argv: &[String]) -> ExecResult {
    match argv.first().map(String::as_str) {
        Some("create") => out("container-1\n"),
        Some("container") | Some("inspect") => fail(1, "Error: No such object"),
        _ => out(""),
    }
}

/// The value of the argument that follows every occurrence of `flag`.
fn values_for(argv: &[String], flag: &str) -> Vec<String> {
    argv.windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .collect()
}

/// A grant with a host path, as the Python tests build one.
fn host_grant(path: &str, host: &str, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path)
        .expect("grant path")
        .with_host_path(host)
        .expect("host path")
        .read_only(read_only)
}

/// A manifest entry holding one S3 bucket attached as a Docker volume.
fn volume_entry(driver: &str) -> Entry {
    Entry::mount(Mount::new(Arc::new(
        ConfiguredMountStrategy::docker_volume(
            S3Mount::new("bucket").with_credentials("key-id", "secret"),
            driver,
        ),
    )))
}

// ----------------------------------------------------------------------------------------------
// `docker create` arguments
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_create_container_publishes_exposed_ports`. It asserts
/// `ports={"8765/tcp": ("127.0.0.1", None)}`; the CLI spelling of the same request is
/// `-p 127.0.0.1::8765/tcp`, which still lets the daemon pick the host port.
#[tokio::test]
async fn create_publishes_every_exposed_port_on_the_loopback_interface() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE)
            .with_exposed_ports([8765, 9000]),
    )
    .with_runner(runner.clone());

    let session = client.create(None).await.expect("create");

    let argv = runner.first("create");
    assert_eq!(
        values_for(&argv, "-p"),
        vec!["127.0.0.1::8765/tcp", "127.0.0.1::9000/tcp"]
    );
    assert_eq!(session.exposed_ports(), vec![8765, 9000]);
}

/// Python: `test_docker_create_container_applies_labels`.
#[tokio::test]
async fn create_applies_the_configured_labels() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE)
            .with_label("com.example.owner", "worker-123"),
    )
    .with_runner(runner.clone());

    client.create(None).await.expect("create");

    let labels = values_for(&runner.first("create"), "--label");
    assert!(
        labels.contains(&"com.example.owner=worker-123".to_string()),
        "{labels:?}"
    );
    // The SDK's own marker label is always present, so a stray sandbox container is findable.
    assert!(
        labels.contains(&"openai-agents-sandbox=1".to_string()),
        "{labels:?}"
    );
}

/// Python: `test_docker_create_container_omits_empty_labels` — no configured labels means no
/// label arguments beyond the SDK's own marker.
#[tokio::test]
async fn create_omits_labels_when_none_are_configured() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(DockerSandboxClientOptions::new(
        DEFAULT_PYTHON_SANDBOX_IMAGE,
    ))
    .with_runner(runner.clone());

    client.create(None).await.expect("create");

    assert_eq!(
        values_for(&runner.first("create"), "--label"),
        vec!["openai-agents-sandbox=1"]
    );
}

/// The default image is Python's `DEFAULT_PYTHON_SANDBOX_IMAGE`, and PID 1 is kept alive so
/// `docker exec` has something to attach to.
#[tokio::test]
async fn create_uses_the_python_default_image_and_keeps_pid_one_alive() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());

    client.create(None).await.expect("create");

    let argv = runner.first("create");
    assert_eq!(DEFAULT_PYTHON_SANDBOX_IMAGE, "python:3.14-slim");
    let image_at = argv
        .iter()
        .position(|arg| arg == DEFAULT_PYTHON_SANDBOX_IMAGE)
        .expect("the image is in the argument vector");
    assert_eq!(
        &argv[image_at + 1..],
        ["sh", "-c", "while :; do sleep 3600; done"]
    );
    assert_eq!(values_for(&argv, "-w"), vec!["/workspace"]);
}

/// Python spells a network-less container `network_mode="none"`; the CLI spelling is
/// `--network none`. `test_docker_network_mode.py` is another agent's file; this only asserts
/// that the option reaches `docker create` and the session state.
#[tokio::test]
async fn a_network_disabled_container_is_created_without_a_network_stack() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE).with_network_disabled(true),
    )
    .with_runner(runner.clone());

    let session = client.create(None).await.expect("create");

    assert_eq!(
        values_for(&runner.first("create"), "--network"),
        vec!["none"]
    );
    assert_eq!(
        session
            .session_state()
            .provider_metadata(DOCKER_NETWORK_DISABLED_KEY)
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

/// Python: `test_docker_create_container_mounts_explicit_host_path`, both parametrizations
/// (`/workspace` and the un-normalized `/mnt/sibling/../workspace`). Python asserts a
/// `{"Target", "Source", "Type": "bind", "ReadOnly": True}` mount; the CLI spelling is
/// `-v source:target:ro`.
#[tokio::test]
async fn create_binds_an_explicit_host_path_grant_read_only() {
    for root in ["/workspace", "/mnt/sibling/../workspace"] {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::with_options(DockerSandboxClientOptions::new(
            DEFAULT_PYTHON_SANDBOX_IMAGE,
        ))
        .with_runner(runner.clone());
        let manifest = Manifest::new()
            .with_root(root)
            .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", true));

        client.create(Some(manifest)).await.expect("create");

        assert_eq!(
            values_for(&runner.first("create"), "-v"),
            vec!["/host/shared-data:/mnt/shared-data:ro"],
            "root {root}"
        );
    }
}

/// Python: `test_docker_create_container_keeps_path_only_grant_unmounted`. A grant with no host
/// path is a *policy* grant: nothing is attached to the container for it.
#[tokio::test]
async fn create_keeps_a_path_only_grant_unmounted() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/tmp")
            .expect("grant")
            .read_only(true),
    );

    client.create(Some(manifest)).await.expect("create");

    let argv = runner.first("create");
    assert!(values_for(&argv, "-v").is_empty(), "{argv:?}");
    assert!(values_for(&argv, "--mount").is_empty(), "{argv:?}");
}

/// Python: `test_docker_create_container_mounts_s3_with_volume_driver_ignoring_mount_pattern`
/// and the `rclone` sibling. Python asserts the `VolumeOptions.DriverConfig` dictionary; the CLI
/// spelling is one `--mount type=volume,...` specification, and the volume name is derived from
/// the session id and the mount path exactly as Python's `_docker_volume_name` derives it.
#[tokio::test]
async fn create_names_a_volume_after_the_session_and_carries_the_driver_options() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new().with_entry("data", volume_entry("rclone"));

    let session = client.create(Some(manifest.clone())).await.expect("create");
    let session_id = session
        .session_state()
        .provider_metadata(DOCKER_SESSION_ID_KEY)
        .and_then(serde_json::Value::as_str)
        .expect("the session id names the volumes")
        .to_string();

    let mounts = values_for(&runner.first("create"), "--mount");
    assert_eq!(mounts.len(), 1, "{mounts:?}");
    let spec = &mounts[0];
    assert!(spec.starts_with("type=volume,"), "{spec}");
    assert!(
        spec.contains(&format!(
            "src={}",
            docker_volume_name(Some(&session_id), "/workspace/data")
        )),
        "{spec}"
    );
    assert!(spec.contains("dst=/workspace/data"), "{spec}");
    assert!(spec.contains("volume-driver=rclone"), "{spec}");
    // A read-only mount (the default) is spelled `readonly`, as Python's `ReadOnly: True` is.
    assert!(spec.contains("readonly"), "{spec}");
    // The name is a pure function of the session id and the path, so teardown finds it again.
    assert_eq!(
        docker_volume_names_for_manifest(&manifest, Some(&session_id)),
        vec![docker_volume_name(Some(&session_id), "/workspace/data")]
    );
}

/// Python: `test_docker_create_container_grants_sys_admin_for_s3_files_mount` and
/// `..._grants_fuse_for_in_container_rclone_mount`. An in-container mount tool needs the
/// capability at create time, because a capability cannot be added to a running container.
#[tokio::test]
async fn create_grants_only_the_privileges_an_in_container_mount_needs() {
    let fuse = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(fuse.clone());
    let manifest = Manifest::new().with_entry(
        "data",
        Entry::mount(Mount::new(Arc::new(ConfiguredMountStrategy::rclone(
            S3Mount::new("bucket"),
        )))),
    );

    client.create(Some(manifest)).await.expect("create");

    let argv = fuse.first("create");
    assert_eq!(values_for(&argv, "--device"), vec!["/dev/fuse"]);
    assert_eq!(values_for(&argv, "--cap-add"), vec!["SYS_ADMIN"]);
    assert_eq!(
        values_for(&argv, "--security-opt"),
        vec!["apparmor:unconfined"]
    );

    // A Docker volume is attached by the runtime, so the container needs nothing extra.
    let volume = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(volume.clone());
    client
        .create(Some(
            Manifest::new().with_entry("data", volume_entry("rclone")),
        ))
        .await
        .expect("create");
    let argv = volume.first("create");
    assert!(values_for(&argv, "--cap-add").is_empty(), "{argv:?}");
    assert!(values_for(&argv, "--device").is_empty(), "{argv:?}");
}

/// Python: `test_docker_create_preserves_compatible_host_grants_and_storage`, all four
/// parametrizations. A compatible combination is created, not refused.
#[tokio::test]
async fn create_preserves_compatible_host_grants_and_storage() {
    // (storage, explicit host path, read-only, expects SYS_ADMIN, expects a volume)
    let cases: [(&str, bool, bool, bool, bool); 4] = [
        ("none", true, true, false, false),
        ("in-container", true, false, true, false),
        ("in-container", false, true, true, false),
        ("docker-volume", true, true, false, true),
    ];
    for (storage, explicit_host_path, read_only, expects_sys_admin, expects_volume) in cases {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let grant = if explicit_host_path {
            host_grant("/mnt/shared-data", "/host/shared-data", read_only)
        } else {
            SandboxPathGrant::new("/mnt/shared-data")
                .expect("grant")
                .read_only(read_only)
        };
        let mut manifest = Manifest::new().with_extra_path_grant(grant);
        manifest = match storage {
            "in-container" => manifest.with_entry(
                "data",
                Entry::mount(Mount::new(Arc::new(ConfiguredMountStrategy::rclone(
                    S3Mount::new("bucket"),
                )))),
            ),
            "docker-volume" => manifest.with_entry("data", volume_entry("rclone")),
            _ => manifest,
        };

        client.create(Some(manifest)).await.expect("create");

        let argv = runner.first("create");
        let expected_binds: Vec<String> = if explicit_host_path {
            vec![format!(
                "/host/shared-data:/mnt/shared-data{}",
                if read_only { ":ro" } else { "" }
            )]
        } else {
            Vec::new()
        };
        assert_eq!(values_for(&argv, "-v"), expected_binds, "{storage}");
        assert_eq!(
            !values_for(&argv, "--cap-add").is_empty(),
            expects_sys_admin,
            "{storage}"
        );
        assert_eq!(
            values_for(&argv, "--mount")
                .iter()
                .any(|spec| spec.starts_with("type=volume")),
            expects_volume,
            "{storage}"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Configurations refused before the daemon is touched
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_rejects_duplicate_target_shared_by_split_and_path_only_grants`, both
/// orderings. The refusal has to happen before the image lookup, so nothing is created.
#[tokio::test]
async fn a_duplicate_grant_target_is_refused_before_the_daemon_is_touched() {
    for explicit_first in [false, true] {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let path_only = SandboxPathGrant::new("/mnt/shared-data").expect("grant");
        let explicit = host_grant("/mnt/shared-data", "/host/shared-data", false);
        let grants = if explicit_first {
            vec![explicit, path_only]
        } else {
            vec![path_only, explicit]
        };
        let mut manifest = Manifest::new();
        for grant in grants {
            manifest = manifest.with_extra_path_grant(grant);
        }

        let error = client
            .create(Some(manifest))
            .await
            .expect_err("a duplicate grant target is refused");
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
        assert!(
            error
                .message
                .contains("duplicate Docker sandbox path grant target"),
            "{error}"
        );
        assert!(runner.calls().is_empty(), "{:?}", runner.joined());
    }
}

/// Python: `test_docker_rejects_host_path_target_overlapping_workspace_before_image_lookup`,
/// every parametrization that does not depend on Python's `PureWindowsPath`.
#[tokio::test]
async fn a_host_grant_overlapping_the_workspace_root_is_refused_before_the_daemon_is_touched() {
    let cases = [
        ("/workspace", "/workspace/shared-data"),
        ("/workspace/project", "/workspace"),
        ("/mnt/sibling/..", "/mnt/shared-data"),
        ("/mnt/sibling/../shared-data", "/mnt/shared-data"),
        ("/outside/../workspace/project", "/workspace"),
        ("//mnt/sibling/..", "/mnt/shared-data"),
        ("/mnt/sibling/..", "//mnt/shared-data"),
        ("//mnt/sibling/../shared-data", "/mnt/shared-data"),
        ("/outside/../mnt/shared-data/project", "//mnt/shared-data"),
    ];
    for (root, target) in cases {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let manifest = Manifest::new()
            .with_root(root)
            .with_extra_path_grant(host_grant(target, "/host/shared-data", false));

        let error = match client.create(Some(manifest)).await {
            Err(error) => error,
            Ok(_) => panic!("`{root}` / `{target}` must be refused"),
        };
        assert_eq!(
            error.code,
            ErrorCode::SandboxConfigInvalid,
            "{root} / {target}"
        );
        assert!(
            error
                .message
                .contains("host_path target must be outside the workspace root"),
            "{error}"
        );
        assert!(runner.calls().is_empty(), "{:?}", runner.joined());
    }
}

/// Python: `test_docker_rejects_read_only_host_grant_with_privileged_storage_before_side_effects`
/// — the refusal happens before anything reaches the daemon, so no container is created and the
/// persisted state is untouched.
#[tokio::test]
async fn a_read_only_host_grant_with_a_privileged_mount_is_refused_before_side_effects() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new()
        .with_entry(
            "data",
            Entry::mount(Mount::new(Arc::new(ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket"),
            )))),
        )
        .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", true));

    let error = client
        .create(Some(manifest.clone()))
        .await
        .expect_err("a read-only host grant cannot be combined with a privileged mount");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("SYS_ADMIN"), "{error}");
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());

    // The same refusal on the resume path, equally without touching the daemon.
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("existing-container".to_string()))
        .with_workspace_root_ready(true);
    let error = client
        .resume(&state, manifest)
        .await
        .expect_err("resume refuses the same combination");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());
    assert_eq!(state.provider_id.as_deref(), Some("existing-container"));
    assert!(state.workspace_root_ready);
}

/// Python: `test_docker_path_grant_validation_does_not_probe_host_workspace_root`. The workspace
/// root is a container path that need not exist on the host, so the comparison is lexical and a
/// container-only root still reaches the daemon.
#[tokio::test]
async fn path_grant_validation_is_lexical_and_never_probes_the_host() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new()
        .with_root("//container-only/sibling/../workspace")
        .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", false));

    client
        .create(Some(manifest))
        .await
        .expect("a container-only workspace root is created, not probed");

    // The root is a container path: it reaches `docker create` as configured, not probed.
    let argv = runner.first("create");
    assert_eq!(values_for(&argv, "-w").len(), 1);
}

// ----------------------------------------------------------------------------------------------
// Deletion and volume cleanup
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_delete_removes_generated_docker_volumes`. The container is removed, and
/// so is every named volume the manifest generated — but not storage an in-container mount
/// attached, which has no volume of its own.
#[tokio::test]
async fn delete_removes_the_container_and_only_the_volumes_it_generated() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new()
        .with_entry("data", volume_entry("rclone"))
        .with_entry(
            "in-container",
            Entry::mount(Mount::new(Arc::new(ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket"),
            )))),
        );
    let session = client.create(Some(manifest)).await.expect("create");
    let session_id = session
        .session_state()
        .provider_metadata(DOCKER_SESSION_ID_KEY)
        .and_then(serde_json::Value::as_str)
        .expect("session id")
        .to_string();

    client.delete(session).await.expect("delete");

    let joined = runner.joined();
    assert!(
        joined.iter().any(|call| call == "rm -f container-1"),
        "{joined:?}"
    );
    let expected_volume = docker_volume_name(Some(&session_id), "/workspace/data");
    assert!(
        joined
            .iter()
            .any(|call| call == &format!("volume rm {expected_volume}")),
        "{joined:?}"
    );
    assert_eq!(
        joined
            .iter()
            .filter(|call| call.starts_with("volume rm "))
            .count(),
        1,
        "only the generated volume is removed: {joined:?}"
    );
}

/// Shutting a session down twice removes the container once: the second call is a no-op, so a
/// `delete` after an explicit `shutdown` does not fail on a container that is already gone.
#[tokio::test]
async fn a_second_shutdown_does_not_remove_the_container_again() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let session = client.create(None).await.expect("create");

    session.shutdown().await.expect("shutdown");
    session.shutdown().await.expect("second shutdown");

    assert_eq!(
        runner
            .joined()
            .iter()
            .filter(|call| call.starts_with("rm -f "))
            .count(),
        1,
        "{:?}",
        runner.joined()
    );
}

/// A failing `docker rm` is reported rather than swallowed, with Python's `WorkspaceStopError`
/// equivalent error code.
#[tokio::test]
async fn a_failed_container_removal_is_reported() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("rm") {
            return Ok(fail(1, "Error response from daemon: removal in progress"));
        }
        Ok(default_response(argv))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let session = client.create(None).await.expect("create");

    let error = session
        .shutdown()
        .await
        .expect_err("a failed removal is an error");
    assert_eq!(error.code, ErrorCode::WorkspaceStopError);
    assert!(error.message.contains("`docker rm` failed"), "{error}");
}

// ----------------------------------------------------------------------------------------------
// Session state
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_session_state_roundtrip_preserves_labels` and
/// `test_docker_session_state_without_labels_preserves_old_payloads`.
#[test]
fn session_state_round_trips_the_labels_and_tolerates_a_payload_without_them() {
    let manifest = Manifest::new().with_root("/workspace");
    let labels: BTreeMap<String, String> =
        [("com.example.owner".to_string(), "worker-123".to_string())].into();
    let session = DockerSandboxSession::new("container", manifest.clone()).with_resume_identity(
        false,
        labels.clone(),
        Vec::new(),
    );

    let json = session.session_state().to_json_string().expect("serialize");
    let restored = SandboxSessionState::from_json_str(&json).expect("parse");
    assert_eq!(
        restored
            .provider_metadata(DOCKER_LABELS_KEY)
            .and_then(serde_json::Value::as_object)
            .map(|map| map
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                .collect::<BTreeMap<String, String>>()),
        Some(labels)
    );

    // An older payload that never carried labels still parses, and reads back as "no labels".
    let mut payload: serde_json::Value = serde_json::from_str(&json).expect("json");
    payload
        .get_mut("provider_metadata")
        .and_then(serde_json::Value::as_object_mut)
        .expect("provider metadata")
        .remove(DOCKER_LABELS_KEY);
    let older =
        SandboxSessionState::from_json_str(&payload.to_string()).expect("an older payload parses");
    assert!(older.provider_metadata(DOCKER_LABELS_KEY).is_none());
}

/// Python: `test_docker_create_persists_configured_labels` — the labels the client was
/// configured with are the ones the session persists, so a resume can check the container still
/// carries them.
#[tokio::test]
async fn create_persists_the_configured_labels_on_the_session_state() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE)
            .with_label("com.example.owner", "worker-123"),
    )
    .with_runner(runner.clone());

    let session = client.create(None).await.expect("create");

    let state = session.session_state();
    assert_eq!(
        state
            .provider_metadata(DOCKER_LABELS_KEY)
            .and_then(|value| value.get("com.example.owner").cloned())
            .and_then(|value| value.as_str().map(str::to_string)),
        Some("worker-123".to_string())
    );
    assert_eq!(state.provider_id.as_deref(), Some("container-1"));
    assert_eq!(state.backend, "docker");
}
