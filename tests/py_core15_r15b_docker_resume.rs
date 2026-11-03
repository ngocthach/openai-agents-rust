//! Ports of the credentialless external-mount resume cases of the Python SDK's
//! `tests/sandbox/test_docker.py`: a manifest whose Docker-volume mount carries no credentials
//! needs no fresh container, so a live one is reattached (also after a serialized round trip) and
//! a missing one is replaced under the session's own volume identity.
//!
//! Every Docker call goes through a recording `DockerRunner`; no daemon is involved.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::{Mount, docker_volume_name};
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxClient};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxResult};

struct FakeDocker {
    calls: Mutex<Vec<Vec<String>>>,
    live_container: bool,
}

impl std::fmt::Debug for FakeDocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeDocker").finish_non_exhaustive()
    }
}

impl FakeDocker {
    fn new(live_container: bool) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            live_container,
        })
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

fn out(stdout: &str) -> ExecResult {
    ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

#[async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(argv.to_vec());
        Ok(match argv.first().map(String::as_str) {
            Some("create") => out("replacement-container\n"),
            Some("container") | Some("inspect") if self.live_container => out(&inspect_document()),
            Some("container") | Some("inspect") => ExecResult {
                stdout: Vec::new(),
                stderr: b"Error: No such object".to_vec(),
                exit_code: 1,
            },
            _ => out(""),
        })
    }
}

fn inspect_document() -> String {
    serde_json::json!([{
        "HostConfig": {"NetworkMode": "bridge"},
        "NetworkSettings": {"Networks": {"bridge": {}}, "Ports": {}},
        "Mounts": [],
        "Config": {"Labels": {}},
    }])
    .to_string()
}

/// Python's `S3Mount(bucket=...)` with no credentials, attached through a Docker volume driver.
fn credentialless_volume() -> Mount {
    ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone").mount()
}

fn manifest() -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(credentialless_volume()))
}

fn state(manifest: &Manifest, container: &str) -> SandboxSessionState {
    SandboxSessionState::new("docker", "sess-1", "/workspace", manifest)
        .with_provider_id(Some(container.to_string()))
}

fn created(runner: &FakeDocker) -> Vec<String> {
    runner
        .joined()
        .into_iter()
        .filter(|call| call.starts_with("create "))
        .collect()
}

/// `test_docker_resume_reconnects_serialized_credentialless_external_mount`: nothing in the
/// persisted state is withheld, so the container id survives the round trip and is reattached.
#[tokio::test]
async fn a_serialized_credentialless_external_mount_reconnects_to_its_container() {
    let runner = FakeDocker::new(true);
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = manifest();
    let original = state(&manifest, "container");
    let restored =
        SandboxSessionState::from_json_str(&original.to_json_string().expect("serialize"))
            .expect("parse");
    assert_eq!(restored.provider_id.as_deref(), Some("container"));
    assert!(!restored.mount_authority_rebound);

    let resumed = client.resume(&restored, manifest).await.expect("resume");

    assert_eq!(resumed.id(), "container");
    assert_eq!(restored.session_id, original.session_id);
    assert!(created(&runner).is_empty(), "{:?}", runner.joined());
}

/// `test_docker_resume_accepts_live_credentialless_external_mount`: a directly built state with
/// the same manifest reattaches to the live container too, keeping the manifest it was given.
#[tokio::test]
async fn a_live_credentialless_external_mount_is_accepted() {
    let runner = FakeDocker::new(true);
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = manifest();
    let direct = state(&manifest, "container");

    let resumed = client
        .resume(&direct, manifest.clone())
        .await
        .expect("resume");

    assert_eq!(resumed.id(), "container");
    assert_eq!(format!("{:?}", resumed.manifest()), format!("{manifest:?}"));
    assert!(created(&runner).is_empty(), "{:?}", runner.joined());
}

/// `test_docker_resume_preserves_direct_credentialless_volume_identity`: when the container is
/// gone, the replacement keeps the session id, so it mounts the volume the paused container used
/// and nothing removes that volume.
#[tokio::test]
async fn a_missing_container_is_replaced_under_the_same_volume_identity() {
    let runner = FakeDocker::new(false);
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = manifest();
    let direct = state(&manifest, "missing-container");

    let resumed = client.resume(&direct, manifest).await.expect("resume");

    assert_eq!(resumed.id(), "replacement-container");
    let creates = created(&runner);
    assert_eq!(creates.len(), 1, "{:?}", runner.joined());
    let expected = docker_volume_name(Some(&direct.session_id), "/workspace/data");
    assert!(
        creates[0].contains(&format!("src={expected}")),
        "{} lacks {expected}",
        creates[0]
    );
    assert!(
        !runner
            .joined()
            .iter()
            .any(|call| call.starts_with("volume rm")),
        "{:?}",
        runner.joined()
    );
}
