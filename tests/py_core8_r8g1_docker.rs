//! Ports of the Docker rows of `tests/sandbox/test_docker.py` that had no Rust test: the
//! existence probe behind a failed read (which user it runs as, and what an indeterminate probe
//! reports) and the redaction of a failed container start for a manifest with protected mounts.
//!
//! The fake `DockerRunner` records the argument vector of every `docker` call and answers from a
//! script, so each assertion is on what would reach the daemon.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::Mount;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DOCKER_NETWORK_DISABLED_KEY, DockerRunner, DockerSandboxClient, DockerSandboxSession,
    PROTECTED_MOUNT_ERROR_MESSAGE,
};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxResult, SandboxSession, User,
};

struct FakeDocker {
    calls: Mutex<Vec<Vec<String>>>,
    #[allow(clippy::type_complexity)]
    responder: Box<dyn Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync>,
}

impl std::fmt::Debug for FakeDocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeDocker").finish_non_exhaustive()
    }
}

impl FakeDocker {
    fn scripted(
        responder: impl Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            responder: Box::new(responder),
        })
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().expect("calls").clone()
    }
}

#[async_trait::async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(argv.to_vec());
        (self.responder)(argv)
    }
}

fn result(exit_code: i32, stderr: &str) -> ExecResult {
    ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code,
    }
}

/// Whether `argv` calls the in-container path-resolution helper, which answers with the path.
fn is_helper(argv: &[String]) -> bool {
    argv.iter()
        .any(|arg| arg.contains("resolve-workspace-path-"))
}

fn helper_answer(argv: &[String]) -> ExecResult {
    let at = argv
        .iter()
        .position(|arg| arg.contains("resolve-workspace-path-"))
        .expect("helper path");
    ExecResult {
        stdout: argv.get(at + 2).cloned().unwrap_or_default().into_bytes(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

fn is_cat(argv: &[String]) -> bool {
    argv.iter().any(|arg| arg == "cat")
}

fn session(manifest: Manifest, runner: Arc<FakeDocker>) -> DockerSandboxSession {
    DockerSandboxSession::new("container-under-test", manifest).with_runner(runner)
}

/// The `-u` value of a `docker exec` call, when it has one.
fn exec_user(argv: &[String]) -> Option<&str> {
    let at = argv.iter().position(|arg| arg == "-u")?;
    argv.get(at + 1).map(String::as_str)
}

/// Python: `test_docker_read_probe_uses_requested_user`. A failed read as a named user asks
/// whether the file exists as that user, because the answer depends on who is asking, and the
/// not-found verdict comes from that probe.
#[tokio::test]
async fn the_read_existence_probe_runs_as_the_requested_user() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) {
            return Ok(helper_answer(argv));
        }
        if is_cat(argv) {
            return Ok(result(1, "cat: missing.txt: No such file or directory"));
        }
        // The existence probe: the path is not there.
        Ok(result(1, ""))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());
    let user = User::new("sandbox-user");

    let error = session
        .read_file_as("missing.txt", Some(&user))
        .await
        .expect_err("a missing path is not found");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    let user_calls: Vec<Vec<String>> = runner
        .calls()
        .into_iter()
        .filter(|argv| exec_user(argv) == Some("sandbox-user"))
        .collect();
    assert!(
        user_calls.len() >= 2,
        "both the read and its probe run as the user: {:?}",
        runner.calls()
    );
    assert!(
        user_calls.iter().any(|argv| !is_cat(argv)),
        "the probe itself ran as the user: {user_calls:?}"
    );
}

/// Python: `test_docker_read_indeterminate_probe_raises_archive_error`. When the probe cannot say
/// whether the path exists (it exits with something other than "absent"), the read is an
/// archive-read failure, not a not-found.
#[tokio::test]
async fn an_indeterminate_existence_probe_is_an_archive_read_error() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) {
            return Ok(helper_answer(argv));
        }
        if is_cat(argv) {
            return Ok(result(
                1,
                "cat: inaccessible/missing.txt: Permission denied",
            ));
        }
        Ok(result(2, ""))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);
    let user = User::new("sandbox-user");

    let error = session
        .read_file_as("inaccessible/missing.txt", Some(&user))
        .await
        .expect_err("an indeterminate probe is an error");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        error
            .context
            .get("existence_probe_exit_code")
            .map(String::as_str),
        Some("2")
    );
}

fn credentialed_volume_manifest() -> Manifest {
    Manifest::new().with_entry(
        "data",
        Entry::mount(Mount::new(Arc::new(
            ConfiguredMountStrategy::docker_volume(
                S3Mount::new("bucket").with_credentials("access-key", "secret-key"),
                "rclone",
            ),
        ))),
    )
}

/// Python: `test_docker_create_cleans_generated_volumes_after_start_failure` and
/// `test_docker_create_cleans_resources_after_post_start_failure`, the redaction half. When the
/// daemon refuses to start a container whose manifest carries credentialed mounts, the error
/// carries the code but none of the daemon's text, which may echo the driver options.
#[tokio::test]
async fn a_failed_container_start_redacts_the_daemon_text_for_protected_mounts() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("start") {
            return Ok(result(1, "daemon echoed driver option secret-key"));
        }
        Ok(result(0, ""))
    });
    let session = session(credentialed_volume_manifest(), runner);

    let error = session.start().await.expect_err("the start fails");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert_eq!(error.message, PROTECTED_MOUNT_ERROR_MESSAGE);
    assert!(!error.to_string().contains("secret-key"), "{error}");
    assert!(error.context.is_empty(), "{error:?}");
}

/// The same failure for a manifest without protected mounts keeps the daemon's explanation.
#[tokio::test]
async fn a_failed_container_start_keeps_the_daemon_text_without_protected_mounts() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("start") {
            return Ok(result(1, "port is already allocated"));
        }
        Ok(result(0, ""))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);

    let error = session.start().await.expect_err("the start fails");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.contains("port is already allocated"),
        "{error}"
    );
}

// ----------------------------------------------------------------------------------------------
// Resume of a network-isolated session
// ----------------------------------------------------------------------------------------------

async fn resume_isolated(
    attrs: serde_json::Value,
) -> (SandboxResult<Arc<dyn SandboxSession>>, Arc<FakeDocker>) {
    let runner = FakeDocker::scripted(move |argv| {
        if argv.first().map(String::as_str) == Some("container") {
            return Ok(ExecResult {
                stdout: serde_json::json!([attrs]).to_string().into_bytes(),
                stderr: Vec::new(),
                exit_code: 0,
            });
        }
        Ok(result(0, ""))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new().with_root("/workspace");
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container".to_string()))
        .with_provider_metadata(DOCKER_NETWORK_DISABLED_KEY, serde_json::json!(true));
    (client.resume(&state, manifest).await, runner)
}

/// Python: `test_docker_resume_rejects_reused_container_that_is_not_network_isolated` (the
/// `wrong-host-network-mode`, `attached-after-create` and `missing-network-map` rows). A live
/// container that is not the network-less one the state recorded is refused, and no replacement
/// is created in its place.
#[tokio::test]
async fn resume_rejects_a_reused_container_that_is_not_network_isolated() {
    let cases = [
        (
            "wrong-host-network-mode",
            serde_json::json!({
                "Mounts": [],
                "HostConfig": {"NetworkMode": "bridge"},
                "NetworkSettings": {"Networks": {"bridge": {}}},
            }),
        ),
        (
            "attached-after-create",
            serde_json::json!({
                "Mounts": [],
                "HostConfig": {"NetworkMode": "none"},
                "NetworkSettings": {"Networks": {"bridge": {}}},
            }),
        ),
        (
            "missing-network-map",
            serde_json::json!({
                "Mounts": [],
                "HostConfig": {"NetworkMode": "none"},
                "NetworkSettings": {},
            }),
        ),
    ];
    for (name, attrs) in cases {
        let (resumed, runner) = resume_isolated(attrs).await;

        let error = resumed
            .err()
            .unwrap_or_else(|| panic!("{name} must be refused"));

        assert_eq!(error.code, ErrorCode::SessionStateMismatch, "{name}");
        assert!(error.message.contains("network"), "{name}: {error}");
        assert!(
            !runner
                .calls()
                .iter()
                .any(|argv| argv.first().map(String::as_str) == Some("create")),
            "{name}: no replacement container: {:?}",
            runner.calls()
        );
    }
}

/// Python: `test_docker_resume_reuses_container_that_is_network_isolated` (`empty` and
/// `none-network` rows).
#[tokio::test]
async fn resume_reuses_a_container_that_is_network_isolated() {
    for networks in [serde_json::json!({}), serde_json::json!({"none": {}})] {
        let (resumed, runner) = resume_isolated(serde_json::json!({
            "Mounts": [],
            "HostConfig": {"NetworkMode": "none"},
            "NetworkSettings": {"Networks": networks},
        }))
        .await;

        let session = resumed.expect("an isolated container is reattached");

        assert_eq!(session.id(), "container");
        assert!(
            !runner
                .calls()
                .iter()
                .any(|argv| argv.first().map(String::as_str) == Some("create")),
            "{:?}",
            runner.calls()
        );
    }
}

/// Python: `test_docker_create_cleans_generated_volumes_after_start_failure` and
/// `test_docker_create_cleans_resources_after_post_start_failure`. Python's `create` also starts
/// the container; here the run does, and when the daemon refuses to start it the run releases
/// what it had made: the container is force-removed, the volumes it generated are removed, and
/// the error that reaches the caller carries none of the daemon's text.
#[tokio::test]
async fn a_failed_start_in_a_run_removes_the_container_and_generated_volumes() {
    use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
    use openai_agents_sdk::testing::{FakeModel, assistant_message};

    let runner = FakeDocker::scripted(|argv| match argv.first().map(String::as_str) {
        Some("create") => Ok(ExecResult {
            stdout: b"container-1\n".to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        }),
        Some("start") => Ok(result(1, "daemon echoed driver option secret-key")),
        Some("container") | Some("inspect") => Ok(result(1, "Error: No such object")),
        _ => Ok(result(0, "")),
    });
    let client = Arc::new(DockerSandboxClient::new().with_runner(runner.clone()));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("unreachable")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_default_manifest(credentialed_volume_manifest());
    let config = SandboxRunConfig::with_client(client);

    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("the run fails when the container does not start");

    assert!(
        error.to_string().contains(PROTECTED_MOUNT_ERROR_MESSAGE),
        "{error}"
    );
    assert!(!error.to_string().contains("secret-key"), "{error}");
    let calls: Vec<String> = runner.calls().iter().map(|argv| argv.join(" ")).collect();
    assert!(
        calls
            .iter()
            .any(|call| call.starts_with("rm ") && call.contains("container-1")),
        "the container is removed: {calls:?}"
    );
    assert_eq!(
        calls
            .iter()
            .filter(|call| call.starts_with("volume rm "))
            .count(),
        1,
        "the generated volume is removed: {calls:?}"
    );
}
