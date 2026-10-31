//! Docker client refusals and root handling that the existing ports loop only partly:
//! `test_docker_rejects_read_only_host_grant_with_privileged_storage_before_side_effects`
//! (five storage mount kinds, create and resume) and
//! `test_docker_path_grant_validation_does_not_probe_host_workspace_root` (create and resume,
//! with and without a host grant). Python fakes docker-py; this port records the `docker` argv
//! through the `DockerRunner` seam.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{
    AzureBlobMount, BuiltinMountPattern, ConfiguredMountStrategy, S3FilesMount, S3Mount,
};
use openai_agents_sdk::sandbox::mounts::{Mount, RcloneMountPattern};
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxClient};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxClient, SandboxResult};

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

fn storage_mounts() -> Vec<(&'static str, Mount)> {
    let mount = |strategy: ConfiguredMountStrategy| Mount::new(Arc::new(strategy));
    vec![
        (
            "fuse",
            mount(ConfiguredMountStrategy::blobfuse(AzureBlobMount::new(
                "account",
                "container",
            ))),
        ),
        (
            "mountpoint",
            mount(ConfiguredMountStrategy::mountpoint(S3Mount::new("bucket"))),
        ),
        (
            "rclone-fuse",
            mount(ConfiguredMountStrategy::rclone(S3Mount::new("bucket"))),
        ),
        (
            "rclone-nfs",
            mount(ConfiguredMountStrategy::in_container(
                S3Mount::new("bucket"),
                BuiltinMountPattern::Rclone(RcloneMountPattern::nfs()),
            )),
        ),
        (
            "s3-files",
            mount(ConfiguredMountStrategy::s3files(S3FilesMount::new(
                "fs-1234567890abcdef0",
            ))),
        ),
    ]
}

#[tokio::test]
async fn a_read_only_host_grant_with_privileged_storage_is_refused_before_side_effects() {
    for (kind, storage) in storage_mounts() {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let manifest = Manifest::new()
            .with_entry("data", Entry::mount(storage))
            .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", true));

        let error = client
            .create(Some(manifest.clone()))
            .await
            .err()
            .unwrap_or_else(|| panic!("{kind}: create must be refused"));
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{kind}: {error}");
        assert!(error.message.contains("SYS_ADMIN"), "{kind}: {error}");
        assert!(runner.calls().is_empty(), "{kind}: {:?}", runner.joined());

        let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
            .with_provider_id(Some("existing-container".to_string()))
            .with_workspace_root_ready(true);
        let error = client
            .resume(&state, manifest)
            .await
            .err()
            .unwrap_or_else(|| panic!("{kind}: resume must be refused"));
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{kind}: {error}");
        assert!(error.message.contains("SYS_ADMIN"), "{kind}: {error}");
        assert!(runner.calls().is_empty(), "{kind}: {:?}", runner.joined());
        assert_eq!(
            state.provider_id.as_deref(),
            Some("existing-container"),
            "{kind}"
        );
        assert!(state.workspace_root_ready, "{kind}");
    }
}

#[tokio::test]
async fn path_grant_validation_never_probes_the_host_for_create_or_resume() {
    for resume in [false, true] {
        for with_grant in [false, true] {
            let case = format!("resume={resume} host_grant={with_grant}");
            let runner = FakeDocker::new();
            let client = DockerSandboxClient::new().with_runner(runner.clone());
            let mut manifest = Manifest::new().with_root("//container-only/sibling/../workspace");
            if with_grant {
                manifest = manifest.with_extra_path_grant(host_grant(
                    "/mnt/shared-data",
                    "/host/shared-data",
                    false,
                ));
            }
            if resume {
                let state = SandboxSessionState::new(
                    "docker",
                    "sess-1",
                    "//container-only/sibling/../workspace",
                    &manifest,
                )
                .with_provider_id(Some("container".to_string()));
                client
                    .resume(&state, manifest)
                    .await
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
            } else {
                client
                    .create(Some(manifest))
                    .await
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
            }
            // The container-only root reaches docker as configured; nothing on the host is probed.
            let argv = runner.first("create");
            assert_eq!(
                values_for(&argv, "-w").len(),
                1,
                "{case}: {:?}",
                runner.joined()
            );
        }
    }
}
