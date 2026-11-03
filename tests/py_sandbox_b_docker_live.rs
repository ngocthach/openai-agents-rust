//! Docker rows of the sandbox compatibility tests.
//!
//! - `test_docker_create_container_omits_network_mode_by_default` and
//!   `test_docker_resume_reapplies_network_mode_to_replacement_container` run against a recording
//!   [`DockerRunner`], as the Python tests run against a mocked Docker client.
//! - `test_live_docker_authority_preserves_resume_and_cleanup` runs against the real daemon. The
//!   Python test name says "live", but its body mocks the Docker client and removal worker; the
//!   Rust port keeps the same assertions (resume, recursive removal of the writable tree, and a
//!   read-only grant nested in it staying intact) on a real container with real bind mounts. It
//!   skips with the reason printed when no daemon answers `docker info`.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::SandboxResult;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DOCKER_NETWORK_DISABLED_KEY, DockerRunner, DockerSandboxClient, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::session::SandboxClient;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;

/// Answers every `docker` call with success, except `inspect`, which reports the configured
/// status. Every argument vector is recorded.
#[derive(Debug)]
struct RecordingDocker {
    calls: Mutex<Vec<Vec<String>>>,
    inspect_exit: i32,
}

impl RecordingDocker {
    fn new(inspect_exit: i32) -> Arc<Self> {
        Arc::new(RecordingDocker {
            calls: Mutex::new(Vec::new()),
            inspect_exit,
        })
    }

    fn creates(&self) -> Vec<Vec<String>> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|argv| argv.first().map(String::as_str) == Some("create"))
            .cloned()
            .collect()
    }
}

#[async_trait]
impl DockerRunner for RecordingDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        let is_inspect = argv.first().map(String::as_str) == Some("inspect");
        Ok(ExecResult {
            stdout: if is_inspect {
                Vec::new()
            } else {
                b"replacement-container\n".to_vec()
            },
            stderr: Vec::new(),
            exit_code: if is_inspect { self.inspect_exit } else { 0 },
        })
    }
}

/// Python: `test_docker_create_container_omits_network_mode_by_default`.
#[tokio::test]
async fn docker_create_omits_the_network_flag_by_default() {
    let docker = RecordingDocker::new(1);
    let client = DockerSandboxClient::with_options(DockerSandboxClientOptions::default())
        .with_runner(docker.clone());

    client.create(None).await.expect("create");

    let creates = docker.creates();
    assert_eq!(creates.len(), 1, "one container is created");
    assert!(
        !creates[0].iter().any(|arg| arg == "--network"),
        "no network flag: {:?}",
        creates[0]
    );
}

/// Python: `test_docker_resume_reapplies_network_mode_to_replacement_container`: the persisted
/// network isolation is applied to the replacement container, whatever the client is set to.
#[tokio::test]
async fn docker_resume_reapplies_network_isolation_to_a_replacement() {
    let docker = RecordingDocker::new(1);
    let client = DockerSandboxClient::with_options(DockerSandboxClientOptions::default())
        .with_runner(docker.clone());
    let manifest = Manifest::new().with_root("/workspace");
    let state = SandboxSessionState::new("docker", "session-123", "/workspace", &manifest)
        .with_provider_id(Some("missing-container".to_string()))
        .with_provider_metadata(DOCKER_NETWORK_DISABLED_KEY, serde_json::Value::Bool(true));

    let resumed = client.resume(&state, manifest).await.expect("resume");

    let creates = docker.creates();
    assert_eq!(creates.len(), 1, "a replacement is created");
    let network = creates[0]
        .windows(2)
        .find(|pair| pair[0] == "--network")
        .map(|pair| pair[1].clone());
    assert_eq!(network.as_deref(), Some("none"), "{:?}", creates[0]);
    assert_eq!(
        resumed.session_state().provider_id.as_deref(),
        Some("replacement-container")
    );
}

fn docker_daemon_available() -> Result<(), String> {
    match std::process::Command::new("docker").arg("info").output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "`docker info` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => Err(format!("`docker` is not runnable: {error}")),
    }
}

/// Python: `test_live_docker_authority_preserves_resume_and_cleanup`, against a real container.
///
/// A writable grant `/external` holds `tree`; a read-only grant `/external/tree/protected` nests
/// inside it. After a resume, removing the writable subtree works, removing `tree` recursively
/// is refused and leaves the protected data intact.
#[tokio::test]
async fn live_docker_authority_preserves_resume_and_cleanup() {
    if let Err(reason) = docker_daemon_available() {
        eprintln!("skipping live Docker removal test: {reason}");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let host_external = temp.path().join("external");
    let tree = host_external.join("tree");
    let protected = tree.join("protected");
    let writable = tree.join("writable");
    std::fs::create_dir_all(&protected).unwrap();
    std::fs::create_dir_all(&writable).unwrap();
    std::fs::write(protected.join("config"), b"protected").unwrap();
    std::fs::write(writable.join("data"), b"allowed").unwrap();

    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            SandboxPathGrant::new("/external")
                .unwrap()
                .with_host_path(host_external.to_string_lossy())
                .unwrap(),
        )
        .with_extra_path_grant(
            SandboxPathGrant::new("/external/tree/protected")
                .unwrap()
                .read_only(true)
                .with_host_path(protected.to_string_lossy())
                .unwrap(),
        );
    let client = DockerSandboxClient::with_options(DockerSandboxClientOptions::new(
        openai_agents_sdk::sandbox::sandboxes::docker::DEFAULT_PYTHON_SANDBOX_IMAGE,
    ));

    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");
    let state = session.session_state();
    session.stop().await.expect("stop");

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("restart");

    resumed
        .rm("/external/tree/writable", true)
        .await
        .expect("the writable subtree is removed");
    assert!(!writable.exists());

    let error = resumed
        .rm("/external/tree", true)
        .await
        .expect_err("removing a tree that holds a read-only grant is refused");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant"),
        "{error:?}"
    );
    assert_eq!(
        std::fs::read(protected.join("config")).unwrap(),
        b"protected"
    );

    client.delete(resumed).await.expect("cleanup");
}
