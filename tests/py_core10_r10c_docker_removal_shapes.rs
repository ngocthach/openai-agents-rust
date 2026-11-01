//! Ports of the Docker removal cases of `tests/sandbox/test_docker_removal.py` that assert a
//! request shape the Rust Docker client produces: the path a recursive removal targets, the
//! user it runs as, what it refuses before touching the daemon, and how `delete` treats a
//! container that is already gone.
//!
//! Python routes recursive removal through an optional host-side `DockerRemovalService`; the
//! Rust client always removes inside the container with `docker exec`, so the cases that assert
//! the service's pause, bind and worker protocol have no counterpart. Each test here is named
//! for the Python case whose observable behaviour it checks. A fake `DockerRunner` records the
//! argument vector of every `docker` call.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxError, SandboxResult, SandboxSession, User,
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
        self.calls.lock().unwrap().clone()
    }

    /// The calls that run the `rm` command (as opposed to the path helper and `docker rm`).
    fn rm_calls(&self) -> Vec<Vec<String>> {
        self.calls()
            .into_iter()
            .filter(|argv| exec_command(argv).first().map(String::as_str) == Some("rm"))
            .collect()
    }
}

#[async_trait::async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        (self.responder)(argv)
    }
}

fn ok(stdout: &str) -> ExecResult {
    ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

fn failed(stderr: &str) -> ExecResult {
    ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code: 1,
    }
}

fn is_helper(argv: &[String]) -> bool {
    argv.iter()
        .any(|arg| arg.contains("resolve-workspace-path-"))
}

/// Answers the in-container path helper with success, everything else with `rest`.
fn with_helper(
    rest: impl Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync + 'static,
) -> Arc<FakeDocker> {
    FakeDocker::scripted(move |argv| {
        if is_helper(argv) {
            let at = argv
                .iter()
                .position(|arg| arg.contains("resolve-workspace-path-"))
                .unwrap();
            // A helper call (not the presence or install probe) is `<helper> <root> <path> ...`.
            return Ok(ok(argv.get(at + 2).map_or("", String::as_str)));
        }
        rest(argv)
    })
}

/// The command a `docker exec` call runs: everything after the container id.
fn exec_command(argv: &[String]) -> Vec<String> {
    if argv.first().map(String::as_str) != Some("exec") {
        return Vec::new();
    }
    let at = argv
        .iter()
        .position(|arg| arg == "container-under-test")
        .expect("the container id");
    argv[at + 1..].to_vec()
}

fn exec_user(argv: &[String]) -> Option<&str> {
    let at = argv.iter().position(|arg| arg == "-u")?;
    argv.get(at + 1).map(String::as_str)
}

fn grant(path: &str, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path).unwrap().read_only(read_only)
}

fn manifest() -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/external", false))
        .with_extra_path_grant(grant("/grant-alias", true))
}

fn session(manifest: Manifest, runner: Arc<FakeDocker>) -> DockerSandboxSession {
    DockerSandboxSession::new("container-under-test", manifest).with_runner(runner)
}

fn strings(parts: &[&str]) -> Vec<String> {
    parts.iter().map(ToString::to_string).collect()
}

/// Python: `test_recursive_removal_preserves_unrelated_writable_trees`. A workspace-relative
/// target and a path under a writable grant both reach `rm` as their own absolute path, and a
/// named user is carried to the daemon.
#[tokio::test]
async fn recursive_removal_targets_the_requested_tree_as_the_requested_user() {
    for (target, expected) in [
        ("build", "/workspace/build"),
        ("/external/build", "/external/build"),
    ] {
        let runner = with_helper(|_| Ok(ok("")));
        let session = session(manifest(), runner.clone());
        session
            .rm_as(target, true, Some(&User::new("developer")))
            .await
            .unwrap_or_else(|error| panic!("{target}: {error}"));

        let rm = runner.rm_calls();
        assert_eq!(rm.len(), 1, "{rm:?}");
        assert_eq!(exec_user(&rm[0]), Some("developer"));
        assert_eq!(
            exec_command(&rm[0]),
            strings(&["rm", "-r", "-f", "--", expected])
        );
    }
}

/// Without a user the removal runs as the container's own user.
#[tokio::test]
async fn recursive_removal_without_a_user_runs_as_the_container_user() {
    let runner = with_helper(|_| Ok(ok("")));
    let session = session(manifest(), runner.clone());
    session.rm("build", true).await.unwrap();

    let rm = runner.rm_calls();
    assert_eq!(rm.len(), 1, "{rm:?}");
    assert_eq!(exec_user(&rm[0]), None);
    assert_eq!(
        exec_command(&rm[0]),
        strings(&["rm", "-rf", "--", "/workspace/build"])
    );
}

/// Python: `test_relative_parent_segments_use_the_normalized_request`. A `..` segment is folded
/// before anything is sent: the helper is asked about, and `rm` removes, the normalized path.
#[tokio::test]
async fn relative_parent_segments_use_the_normalized_path() {
    let runner = with_helper(|_| Ok(ok("")));
    let session = session(manifest(), runner.clone());
    session.rm("link/../build", true).await.unwrap();

    let helper: Vec<Vec<String>> = runner
        .calls()
        .into_iter()
        .filter(|argv| is_helper(argv) && exec_command(argv).len() > 3)
        .collect();
    let last = exec_command(helper.last().expect("a helper call"));
    assert_eq!(last[1], "/workspace");
    assert_eq!(last[2], "/workspace/build");
    let rm = runner.rm_calls();
    assert_eq!(
        exec_command(&rm[0]),
        strings(&["rm", "-rf", "--", "/workspace/build"])
    );
}

/// Python: `test_relative_removal_preserves_workspace_alias_traversal` (accessible variant). A
/// workspace root that is itself an alias keeps its spelling: the removal targets
/// `<alias>/build`, not the alias's canonical target.
#[tokio::test]
async fn a_relative_removal_keeps_the_workspace_alias_spelling() {
    let alias = "/private/workspace-alias";
    let runner = with_helper(|_| Ok(ok("")));
    let session = session(Manifest::new().with_root(alias), runner.clone());
    session
        .rm_as("build", true, Some(&User::new("developer")))
        .await
        .unwrap();

    let rm = runner.rm_calls();
    assert_eq!(exec_user(&rm[0]), Some("developer"));
    assert_eq!(
        exec_command(&rm[0]),
        strings(&["rm", "-r", "-f", "--", &format!("{alias}/build")])
    );
}

/// Python: `test_missing_target_still_checks_the_requested_user`. A removal of a path that does
/// not exist still runs as the requested user, so a permission failure for that user is what is
/// reported, and the failure surfaces as an error.
#[tokio::test]
async fn a_failed_removal_was_attempted_as_the_requested_user() {
    let runner = with_helper(|argv| {
        if exec_command(argv).first().map(String::as_str) == Some("rm") {
            return Ok(failed("rm: cannot remove: Permission denied"));
        }
        Ok(ok(""))
    });
    let session = session(manifest(), runner.clone());
    let error = session
        .rm_as("private/missing", true, Some(&User::new("developer")))
        .await
        .expect_err("a denied removal fails");

    assert_eq!(error.code, ErrorCode::ExecNonZero);
    let rm = runner.rm_calls();
    assert_eq!(rm.len(), 1, "{rm:?}");
    assert_eq!(exec_user(&rm[0]), Some("developer"));
    assert_eq!(
        exec_command(&rm[0]),
        strings(&["rm", "-r", "-f", "--", "/workspace/private/missing"])
    );
}

/// Python: `test_recursive_removal_preserves_live_binding_roots`, the part the in-container
/// client shares with the host service. A recursive removal of an ancestor of a read-only grant
/// is refused before any `rm` runs, while a tree below the workspace still goes through. (The
/// service's extra refusal of its own bound roots has no counterpart: without the service,
/// Python removes the workspace root too.)
#[tokio::test]
async fn recursive_removal_of_an_ancestor_of_a_read_only_grant_is_refused() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/external", false))
        .with_extra_path_grant(grant("/external/protected", true));
    let runner = with_helper(|_| Ok(ok("")));
    let session = session(manifest, runner.clone());

    let error = session
        .rm_as("/external", true, Some(&User::new("0")))
        .await
        .expect_err("a nested read-only grant vetoes the removal");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert!(runner.rm_calls().is_empty(), "nothing was removed");

    session.rm("build", true).await.unwrap();
    assert_eq!(runner.rm_calls().len(), 1);
}

/// Python: `test_delete_releases_authority_only_after_confirmed_container_removal`, the
/// container half. A removed container and one that is already gone both release the session;
/// a daemon failure is reported.
#[tokio::test]
async fn delete_treats_a_missing_container_as_removed_and_reports_daemon_failures() {
    for (outcome, stderr, exit_code, fails) in [
        ("removed", "", 0, false),
        (
            "missing",
            "Error response from daemon: No such container: container-under-test",
            1,
            false,
        ),
        (
            "unavailable",
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
            1,
            true,
        ),
    ] {
        let runner = FakeDocker::scripted(move |argv| {
            if argv.first().map(String::as_str) == Some("rm") {
                return Ok(ExecResult {
                    stdout: Vec::new(),
                    stderr: stderr.as_bytes().to_vec(),
                    exit_code,
                });
            }
            Ok(ok(""))
        });
        let session = session(manifest(), runner.clone());

        let result = session.shutdown().await;
        assert_eq!(result.is_err(), fails, "{outcome}: {result:?}");
        if fails {
            let error: SandboxError = result.unwrap_err();
            assert_eq!(error.code, ErrorCode::WorkspaceStopError, "{outcome}");
        }
        let removes: Vec<Vec<String>> = runner
            .calls()
            .into_iter()
            .filter(|argv| argv.first().map(String::as_str) == Some("rm"))
            .collect();
        assert_eq!(
            removes,
            vec![strings(&["rm", "-f", "container-under-test"])],
            "{outcome}"
        );
    }
}

/// A shut-down session does not ask the daemon to remove the container a second time.
#[tokio::test]
async fn a_released_container_is_not_removed_twice() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("rm") {
            return Ok(failed("Error: No such container: container-under-test"));
        }
        Ok(ok(""))
    });
    let session = session(manifest(), runner.clone());
    session.shutdown().await.unwrap();
    session.shutdown().await.unwrap();

    let removes = runner
        .calls()
        .into_iter()
        .filter(|argv| argv.first().map(String::as_str) == Some("rm"))
        .count();
    assert_eq!(removes, 1);
}
