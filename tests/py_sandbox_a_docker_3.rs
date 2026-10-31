//! Ports of the PTY, protected-mount redaction and create-failure cleanup tests in
//! `tests/sandbox/test_docker.py`.
//!
//! The PTY tests spawn the program `DockerRunner::program` names. Here that is a small shell
//! script standing in for `docker`: it records the argument vector and runs the in-container
//! command locally, so the PTY manager, pid bookkeeping and teardown run for real with no
//! daemon. The script is handed over by path, never through `PATH`, so tests stay independent.

#![cfg(all(feature = "sandbox", unix))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::Mount;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerRunner, DockerSandboxClient, DockerSandboxSession, PROTECTED_MOUNT_ERROR_MESSAGE,
    manifest_has_protected_mounts,
};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxError, SandboxResult, SandboxSession, User,
};

// ----------------------------------------------------------------------------------------------
// Fakes
// ----------------------------------------------------------------------------------------------

/// Records `docker` calls and answers from a script; its PTY program is a local stand-in.
struct FakeDocker {
    calls: Mutex<Vec<Vec<String>>>,
    program: String,
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
        program: &str,
        responder: impl Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            program: program.to_string(),
            responder: Box::new(responder),
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

#[async_trait::async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(argv.to_vec());
        (self.responder)(argv)
    }

    fn program(&self) -> &str {
        &self.program
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

fn default_response(argv: &[String]) -> ExecResult {
    match argv.first().map(String::as_str) {
        Some("create") => out("container-1\n"),
        Some("container") | Some("inspect") => fail(1, "Error: No such object"),
        _ => out(""),
    }
}

/// A script that stands in for `docker exec`: it logs its arguments to `log` and runs the last
/// argument (the `sh -c` body the session builds) locally.
fn fake_docker_program(dir: &tempfile::TempDir) -> (String, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt as _;
    let log = dir.path().join("argv.log");
    let script = dir.path().join("docker");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{}'\nfor last; do :; done\nexec sh -c \"$last\"\n",
            log.display()
        ),
    )
    .expect("write the stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    (script.to_string_lossy().into_owned(), log)
}

fn pty_session(program: &str, runner_calls: &Arc<FakeDocker>) -> DockerSandboxSession {
    let _ = program;
    DockerSandboxSession::new("container-under-test", Manifest::new().with_root("/tmp"))
        .with_runner(runner_calls.clone())
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

// ----------------------------------------------------------------------------------------------
// PTY
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_pty_exec_write_and_poll`. A running process hands back a session id, a
/// write reaches its stdin, and once it exits the id is gone.
#[tokio::test]
async fn pty_exec_writes_stdin_and_reports_the_exit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (program, _) = fake_docker_program(&dir);
    let runner = FakeDocker::scripted(&program, |argv| Ok(default_response(argv)));
    let session = pty_session(&program, &runner);

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("echo ready; read line; echo got:$line")
                .with_tty(true)
                .with_yield_time_ms(3000),
        )
        .await
        .expect("start");
    assert!(started.process_id.is_some(), "{started:?}");
    assert_eq!(started.exit_code, None);
    assert!(started.output_text().contains("ready"), "{started:?}");

    let mut write = PtyWriteRequest::new(started.process_id.expect("id"), "hello\n");
    write.yield_time_ms = 1500;
    let updated = session.pty_write_stdin(write).await.expect("write");
    assert_eq!(updated.process_id, None);
    assert_eq!(updated.exit_code, Some(0));
    assert!(updated.output_text().contains("got:hello"), "{updated:?}");

    let gone = session
        .pty_write_stdin(PtyWriteRequest::new(started.process_id.expect("id"), ""))
        .await
        .expect_err("a finished session is gone");
    assert_eq!(gone.code, ErrorCode::PtySessionNotFound);
}

/// Python: `test_docker_pty_exec_uses_native_docker_user_without_sudo`. The requested user is
/// `docker exec -u`, and the container-side command is not wrapped in `sudo`.
#[tokio::test]
async fn pty_exec_uses_the_native_docker_user_without_sudo() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (program, log) = fake_docker_program(&dir);
    let runner = FakeDocker::scripted(&program, |argv| Ok(default_response(argv)));
    let session = pty_session(&program, &runner);

    let started = session
        .pty_exec_start(
            PtyExecRequest::new("echo hi")
                .with_tty(true)
                .with_user(Some(User::new("sandbox-user")))
                .with_yield_time_ms(3000),
        )
        .await
        .expect("start");
    assert_eq!(started.exit_code, Some(0), "{started:?}");

    let logged = std::fs::read_to_string(&log).expect("the stand-in logged its arguments");
    let lines: Vec<&str> = logged.lines().collect();
    let user_at = lines.iter().position(|line| *line == "-u").expect("-u");
    assert_eq!(lines[user_at + 1], "sandbox-user");
    assert!(!logged.contains("sudo"), "{logged}");
    assert_eq!(&lines[..3], ["exec", "-i", "-t"]);
}

/// Python: `test_docker_pty_exec_returns_exit_code_for_fast_exit` and
/// `test_docker_pty_exec_waits_for_socket_drain_after_process_exit`. A process that exits inside
/// the yield window returns its exit code and every byte it wrote, with no session left behind.
#[tokio::test]
async fn pty_exec_returns_the_exit_code_and_all_output_for_a_fast_exit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (program, _) = fake_docker_program(&dir);
    let runner = FakeDocker::scripted(&program, |argv| Ok(default_response(argv)));
    let session = pty_session(&program, &runner);

    for command in ["echo done", "sleep 0.3; echo done"] {
        let started = session
            .pty_exec_start(
                PtyExecRequest::new(command)
                    .with_tty(true)
                    .with_yield_time_ms(3000),
            )
            .await
            .expect("start");
        assert_eq!(started.process_id, None, "{command}: {started:?}");
        assert_eq!(started.exit_code, Some(0), "{command}");
        assert!(
            started.output_text().contains("done"),
            "{command}: {started:?}"
        );
    }
}

/// Python: `test_docker_pty_non_tty_rejects_stdin_and_stop_cleans_up`. A non-interactive process
/// refuses stdin, and stopping the session kills the container-side process by its recorded pid.
#[tokio::test]
async fn pty_non_tty_rejects_stdin_and_terminate_cleans_up_in_the_container() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (program, _) = fake_docker_program(&dir);
    let runner = FakeDocker::scripted(&program, |argv| Ok(default_response(argv)));
    let session = pty_session(&program, &runner);

    let started = session
        .pty_exec_start(PtyExecRequest::new("echo out; sleep 30").with_yield_time_ms(3000))
        .await
        .expect("start");
    let id = started.process_id.expect("still running");
    assert!(started.output_text().contains("out"), "{started:?}");

    let refused = session
        .pty_write_stdin(PtyWriteRequest::new(id, "hello\n"))
        .await
        .expect_err("a non-tty process has no stdin");
    assert_eq!(refused.code, ErrorCode::PtyStdinUnavailable);
    assert!(
        refused
            .message
            .contains("stdin is not available for this process"),
        "{refused}"
    );

    session.pty_terminate_all().await.expect("terminate");

    // The host-side `docker exec` is only a proxy, so teardown also kills the process inside
    // the container, by the pid file the command wrote.
    let kills: Vec<String> = runner
        .joined()
        .into_iter()
        .filter(|call| call.contains("kill -TERM") && call.contains("openai-agents-sandbox-pty-"))
        .collect();
    assert_eq!(kills.len(), 1, "{:?}", runner.joined());

    let gone = session
        .pty_write_stdin(PtyWriteRequest::new(id, ""))
        .await
        .expect_err("a terminated session is gone");
    assert_eq!(gone.code, ErrorCode::PtySessionNotFound);
}

// ----------------------------------------------------------------------------------------------
// Protected-mount redaction
// ----------------------------------------------------------------------------------------------

/// A mount whose credentials a provider could echo is recognised; one without is not.
#[test]
fn a_credentialed_mount_is_protected_and_a_plain_one_is_not() {
    assert!(manifest_has_protected_mounts(
        &credentialed_volume_manifest()
    ));
    let plain = Manifest::new().with_entry(
        "data",
        Entry::mount(Mount::new(Arc::new(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone"),
        ))),
    );
    assert!(!manifest_has_protected_mounts(&plain));
    assert!(!manifest_has_protected_mounts(&Manifest::new()));
}

/// Python: `test_docker_create_cleans_generated_volumes_when_container_acquisition_fails` and
/// `..._cleans_generated_volumes_after_start_failure`'s create half. When `docker create` fails
/// the error carries no provider text — the daemon echoed the secret — and the volumes this
/// attempt generated are removed.
#[tokio::test]
async fn a_failed_create_redacts_the_provider_error_and_removes_generated_volumes() {
    let runner = FakeDocker::scripted("docker", |argv| {
        if argv.first().map(String::as_str) == Some("create") {
            return Ok(fail(1, "daemon echoed secret-key while creating"));
        }
        Ok(default_response(argv))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());

    let error = client
        .create(Some(credentialed_volume_manifest()))
        .await
        .expect_err("create fails");

    assert_eq!(error.message, PROTECTED_MOUNT_ERROR_MESSAGE);
    assert!(!error.to_string().contains("secret-key"), "{error}");
    assert!(error.context.is_empty(), "{error:?}");
    let joined = runner.joined();
    assert_eq!(
        joined
            .iter()
            .filter(|call| call.starts_with("volume rm "))
            .count(),
        1,
        "{joined:?}"
    );
}

/// An error on a manifest with no credentialed mount is reported verbatim: the redaction is for
/// protected configurations, not a blanket gag.
#[tokio::test]
async fn a_failed_create_without_credentials_keeps_the_daemon_message() {
    let runner = FakeDocker::scripted("docker", |argv| {
        if argv.first().map(String::as_str) == Some("create") {
            return Ok(fail(1, "no space left on device"));
        }
        Ok(default_response(argv))
    });
    let client = DockerSandboxClient::new().with_runner(runner);

    let error = client.create(None).await.expect_err("create fails");
    assert!(error.message.contains("no space left on device"), "{error}");
}

/// Python: `test_docker_delete_redacts_first_failure_and_settles_all_volumes` and
/// `test_docker_delete_runs_shutdown_and_volume_cleanup_after_lookup_failure`. A failing
/// container removal is redacted, and every volume is still attempted.
#[tokio::test]
async fn delete_redacts_the_first_failure_and_still_settles_every_volume() {
    let runner = FakeDocker::scripted("docker", |argv| match argv.first().map(String::as_str) {
        Some("rm") => Ok(fail(1, "removal echoed secret-key")),
        Some("volume") => Ok(fail(1, "volume busy")),
        _ => Ok(default_response(argv)),
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new()
        .with_entry("left", credentialed_entry())
        .with_entry("right", credentialed_entry());
    let session = client.create(Some(manifest)).await.expect("create");

    let error = client.delete(session).await.expect_err("delete fails");

    assert_eq!(error.message, PROTECTED_MOUNT_ERROR_MESSAGE);
    assert!(!error.to_string().contains("secret-key"), "{error}");
    let joined = runner.joined();
    assert_eq!(
        joined
            .iter()
            .filter(|call| call.starts_with("volume rm "))
            .count(),
        2,
        "every volume is attempted even though the container removal failed: {joined:?}"
    );
}

/// Python: `test_docker_direct_persist_redacts_protected_mount_provider_error`.
#[tokio::test]
async fn persist_redacts_a_provider_error_for_a_protected_mount() {
    let runner = FakeDocker::scripted("docker", |argv| {
        if argv.iter().any(|arg| arg.contains("tar -czf -")) {
            return Ok(fail(2, "provider echoed direct-docker-persist-secret"));
        }
        Ok(default_response(argv))
    });
    let session =
        DockerSandboxSession::new("container", credentialed_volume_manifest()).with_runner(runner);

    let error: SandboxError = session
        .persist_workspace()
        .await
        .expect_err("persist fails");

    assert_eq!(error.message, PROTECTED_MOUNT_ERROR_MESSAGE);
    assert!(
        !error.to_string().contains("direct-docker-persist-secret"),
        "{error}"
    );
    assert!(error.context.is_empty(), "{error:?}");
}

/// Python: `test_docker_resume_uses_fresh_volume_identity_and_cleans_partial_replacement`. A
/// replacement that fails under a *fresh* identity removes only the volumes it generated; the
/// stale volume the persisted state named is left alone.
#[tokio::test]
async fn a_failed_replacement_under_a_fresh_identity_leaves_the_persisted_volume_alone() {
    let runner = FakeDocker::scripted("docker", |argv| {
        if argv.first().map(String::as_str) == Some("create") {
            return Ok(fail(1, "replacement failed with secret-key"));
        }
        Ok(default_response(argv))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = credentialed_volume_manifest().with_root("/workspace");
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("missing".to_string()));

    let persisted =
        SandboxSessionState::from_json_str(&state.to_json_string().expect("json")).expect("parse");
    let error = client
        .resume(&persisted, manifest.clone())
        .await
        .expect_err("the replacement fails");

    assert_eq!(error.message, PROTECTED_MOUNT_ERROR_MESSAGE);
    let stale =
        openai_agents_sdk::sandbox::mounts::docker_volume_name(Some("sess-1"), "/workspace/data");
    assert!(
        !runner
            .joined()
            .iter()
            .any(|call| call == &format!("volume rm {stale}")),
        "{:?}",
        runner.joined()
    );
}

fn credentialed_entry() -> Entry {
    Entry::mount(Mount::new(Arc::new(
        ConfiguredMountStrategy::docker_volume(
            S3Mount::new("bucket").with_credentials("access-key", "secret-key"),
            "rclone",
        ),
    )))
}
