//! Ports of `tests/extensions/sandbox/test_daytona.py`.
//!
//! Python patches the vendor `daytona` SDK and asserts on the Python calls it recorded. The Rust
//! backend speaks the Daytona platform API and the per-sandbox toolbox API directly, so the same
//! behaviour is observed through the requests a local mock server records. The mount preflight
//! helpers take `&dyn SandboxSession`, so they are driven with the scripted session exactly as
//! Python drives them with `_FakePreflightSession`.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::daytona::{
    DEFAULT_DAYTONA_WORKSPACE_ROOT, DaytonaCloudBucketMountStrategy, DaytonaSandboxClient,
    DaytonaSandboxClientOptions, DaytonaSandboxResources,
};
use openai_agents_sdk::extensions::sandbox::e2b::rclone::{
    ensure_fuse_support, ensure_rclone, install_package,
};
use openai_agents_sdk::sandbox::mounts::{MountStrategy, RcloneMountMode};
use openai_agents_sdk::sandbox::testing::{
    SandboxMethod, SandboxStep, ScriptedSandboxSession, scripted_sandbox_session,
};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSessionState,
};
use serde_json::json;
use support::{MockResponse, MockServer, Recorded};

// ---------------------------------------------------------------------------------------------
// Scripted-session helpers, the Rust stand-in for Python's `_FakePreflightSession`
// ---------------------------------------------------------------------------------------------

fn result(exit_code: i32, stdout: &str) -> ExecResult {
    ExecResult {
        exit_code,
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
    }
}

fn ok() -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(result(0, ""))
}

fn fail() -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(result(1, ""))
}

fn out(stdout: &str) -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(result(0, stdout))
}

/// Every command the scripted session ran, argv joined with single spaces, which is how Python's
/// `_FakePreflightSession.exec_calls` renders them.
fn exec_calls(session: &ScriptedSandboxSession) -> Vec<String> {
    session
        .calls()
        .into_iter()
        .filter(|call| call.method == SandboxMethod::Exec)
        .map(|call| call.arguments.join(" "))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Mount strategy shape
// ---------------------------------------------------------------------------------------------

/// `test_daytona_mount_strategy_type_and_default_pattern`
#[test]
fn daytona_mount_strategy_type_and_default_pattern() {
    let strategy = DaytonaCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");

    assert_eq!(strategy.strategy_type(), "daytona_cloud_bucket");
    assert_eq!(strategy.pattern.mode, RcloneMountMode::Fuse);
}

/// `test_build_docker_volume_driver_config_returns_none`: the strategy mounts inside the
/// container, so it offers no docker volume driver configuration.
#[test]
fn daytona_mount_strategy_offers_no_docker_volume_driver_config() {
    let strategy = DaytonaCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");

    assert!(!strategy.supports_native_snapshot_detach());
}

/// `test_daytona_session_guard_rejects_wrong_type`: the preflight assumes this provider's image
/// and privilege model, so another backend's session is refused.
#[tokio::test]
async fn daytona_session_guard_rejects_wrong_type() {
    let strategy = DaytonaCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");
    let mount = strategy.clone().mount();
    let session = scripted_sandbox_session([]);

    let error = strategy
        .activate(&mount, &session, "/workspace/bucket")
        .await
        .expect_err("a scripted session is not a Daytona session");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("Daytona sandbox session"),
        "{}",
        error.message
    );
    assert_eq!(session.calls().len(), 0);
}

// ---------------------------------------------------------------------------------------------
// Preflight helpers
// ---------------------------------------------------------------------------------------------

/// `test_has_command_found` / `test_has_command_not_found`, observed through `ensure_rclone`:
/// the presence probe is one `command -v` shell test and nothing else runs when it succeeds.
#[tokio::test]
async fn daytona_has_command_probe_short_circuits_when_found() {
    let session = scripted_sandbox_session([ok()]);

    ensure_rclone(&session, true).await.expect("already there");

    let calls = exec_calls(&session);
    assert_eq!(calls.len(), 1);
    assert!(calls[0].contains("command -v rclone"), "{}", calls[0]);
}

/// `test_pkg_install_via_apt`: an image with `apt-get` installs with it, as root.
#[tokio::test]
async fn daytona_pkg_install_via_apt() {
    let session = scripted_sandbox_session([ok(), ok()]);

    install_package(&session, "rclone", "rclone")
        .await
        .expect("install");

    let calls = exec_calls(&session);
    assert!(calls[0].contains("command -v apt-get"), "{}", calls[0]);
    assert!(
        calls[1].starts_with("sudo -u root --") && calls[1].contains("apt-get"),
        "{}",
        calls[1]
    );
    assert!(calls[1].contains("rclone"), "{}", calls[1]);
}

/// `test_pkg_install_via_apk`: an Alpine image falls through to `apk add`.
#[tokio::test]
async fn daytona_pkg_install_via_apk() {
    let session = scripted_sandbox_session([fail(), ok(), ok()]);

    install_package(&session, "fuse3", "fusermount")
        .await
        .expect("install");

    let calls = exec_calls(&session);
    assert!(calls[1].contains("command -v apk"), "{}", calls[1]);
    assert!(
        calls[2].starts_with("sudo -u root --") && calls[2].contains("apk add"),
        "{}",
        calls[2]
    );
    assert!(calls[2].contains("fuse3"), "{}", calls[2]);
}

/// `test_pkg_install_no_package_manager`
#[tokio::test]
async fn daytona_pkg_install_without_a_package_manager_is_refused() {
    let session = scripted_sandbox_session([fail(), fail()]);

    let error = install_package(&session, "rclone", "rclone")
        .await
        .expect_err("no package manager");

    assert_eq!(error.code, ErrorCode::MountMissingTool);
    assert!(
        error.message.contains("no supported package manager"),
        "{}",
        error.message
    );
}

/// `test_pkg_install_retries_then_fails`: three install attempts, all as root, then a failure
/// that names the last exit code.
#[tokio::test]
async fn daytona_pkg_install_retries_then_fails() {
    let session = scripted_sandbox_session([ok(), fail(), fail(), fail()]);

    let error = install_package(&session, "rclone", "rclone")
        .await
        .expect_err("the install never succeeded");

    assert_eq!(error.code, ErrorCode::MountMissingTool);
    let calls = exec_calls(&session);
    assert_eq!(calls.len(), 4);
    assert!(
        calls[1..]
            .iter()
            .all(|call| call.starts_with("sudo -u root --")),
        "{calls:?}"
    );
}

/// `test_ensure_fuse_dev_fuse_missing` and `test_ensure_fuse_kernel_module_missing`: a sandbox
/// without `/dev/fuse` or without the kernel module is refused.
///
/// Python probes the two separately; the Rust port runs one combined `test -c /dev/fuse && grep
/// -qw fuse /proc/filesystems`, so both Python cases are the same single failing probe here.
#[tokio::test]
async fn daytona_ensure_fuse_requires_the_device_and_the_kernel_module() {
    let session = scripted_sandbox_session([fail()]);

    let error = ensure_fuse_support(&session, true)
        .await
        .expect_err("no FUSE");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("/dev/fuse"), "{}", error.message);
    let calls = exec_calls(&session);
    assert_eq!(calls.len(), 1);
    assert!(calls[0].contains("/proc/filesystems"), "{}", calls[0]);
}

/// `test_ensure_fuse_fusermount_present`: with `fusermount3` already there nothing is installed;
/// only the `/dev/fuse` permission fix-up follows the two probes.
#[tokio::test]
async fn daytona_ensure_fuse_skips_the_install_when_fusermount_is_present() {
    let session = scripted_sandbox_session([ok(), ok(), ok()]);

    ensure_fuse_support(&session, true).await.expect("fuse");

    let calls = exec_calls(&session);
    assert_eq!(calls.len(), 3);
    assert!(calls[1].contains("command -v fusermount3"), "{}", calls[1]);
    assert!(calls[2].contains("chmod a+rw /dev/fuse"), "{}", calls[2]);
}

/// `test_ensure_fuse_installs_when_missing`: a missing `fusermount` installs `fuse3` and is
/// re-probed before the permission fix-up.
#[tokio::test]
async fn daytona_ensure_fuse_installs_fuse3_when_missing() {
    let session = scripted_sandbox_session([
        ok(),   // the kernel probe
        fail(), // fusermount3 missing
        fail(), // fusermount missing
        ok(),   // apt-get present
        ok(),   // the install succeeds
        ok(),   // fusermount3 now present
        ok(),   // the /dev/fuse permission fix-up
    ]);

    ensure_fuse_support(&session, true).await.expect("fuse");

    let calls = exec_calls(&session);
    assert!(calls.iter().any(|call| call.contains("fuse3")), "{calls:?}");
    assert_eq!(calls.len(), 7);
}

/// A sandbox that may not install anything says so instead of trying.
#[tokio::test]
async fn daytona_ensure_fuse_refuses_to_install_when_tools_are_pinned() {
    let session = scripted_sandbox_session([ok(), fail(), fail()]);

    let error = ensure_fuse_support(&session, false)
        .await
        .expect_err("fusermount is missing and may not be installed");

    assert_eq!(error.code, ErrorCode::MountMissingTool);
    assert!(error.message.contains("fusermount"), "{}", error.message);
}

/// `test_ensure_rclone_present`
#[tokio::test]
async fn daytona_ensure_rclone_keeps_a_preinstalled_binary() {
    let session = scripted_sandbox_session([ok()]);

    ensure_rclone(&session, true).await.expect("already there");

    assert_eq!(session.calls().len(), 1);
}

/// An image that may not install `rclone` is refused before anything else runs.
#[tokio::test]
async fn daytona_ensure_rclone_refuses_to_install_when_tools_are_pinned() {
    let session = scripted_sandbox_session([fail()]);

    let error = ensure_rclone(&session, false)
        .await
        .expect_err("rclone is missing and may not be installed");

    assert_eq!(error.code, ErrorCode::MountMissingTool);
    assert_eq!(session.calls().len(), 1);
}

/// `test_ensure_rclone_installs_when_missing`.
///
/// Python's Daytona helper installs the distribution package (`apt-get install rclone`); the Rust
/// port shares E2B's installer, which downloads the pinned upstream release and verifies its
/// SHA-256 inside the sandbox before moving the binary into place. The observable outcome is the
/// same — `rclone` ends up on `PATH` — but the command sequence is the verified-release one.
#[tokio::test]
async fn daytona_ensure_rclone_installs_the_pinned_release_when_missing() {
    let session = scripted_sandbox_session([
        fail(),        // rclone missing
        ok(),          // apt-get present
        out("x86_64"), // a pinned release exists for this architecture
        ok(),          // apt-get update
        ok(),          // the download prerequisites
        ok(),          // the verified install
        ok(),          // rclone now present
    ]);

    ensure_rclone(&session, true).await.expect("install");

    let calls = exec_calls(&session);
    assert!(
        calls.iter().any(|call| call.contains("rclone")),
        "{calls:?}"
    );
    assert!(
        calls
            .iter()
            .any(|call| call.contains("sha256sum --check --strict -")),
        "{calls:?}"
    );
    assert_eq!(calls.len(), 7);
}

// ---------------------------------------------------------------------------------------------
// Mock platform + toolbox
// ---------------------------------------------------------------------------------------------

/// A toolbox command result.
fn command_result(stdout: &str, exit_code: i64) -> MockResponse {
    MockResponse::json(json!({
        "output": stdout,
        "stdout": stdout,
        "stderr": "",
        "exitCode": exit_code,
    }))
}

/// A handler that answers create, state polls and every toolbox call.
fn handler(
    base_url: String,
    stdout: &'static str,
    exit_code: i64,
) -> impl Fn(&Recorded) -> MockResponse {
    move |request: &Recorded| {
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            let _ = &base_url;
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/exec") {
            return command_result(stdout, exit_code);
        }
        MockResponse::json(json!({}))
    }
}

fn client(base_url: &str) -> DaytonaSandboxClient {
    DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(base_url)
            .with_organization_id("org-1")
            .with_state_poll_interval(Duration::from_millis(1)),
    )
}

/// The command string of every toolbox `exec` the server saw.
fn commands(server: &MockServer) -> Vec<String> {
    server
        .requests()
        .into_iter()
        .filter(|request| request.method == "POST" && request.path.ends_with("/exec"))
        .map(|request| {
            request.json()["command"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Client create
// ---------------------------------------------------------------------------------------------

/// `test_create_uses_daytona_safe_default_workspace_root`: without a manifest the session takes
/// the Daytona image's own home rather than the SDK-wide `/workspace` default.
#[tokio::test]
async fn daytona_create_uses_the_safe_default_workspace_root() {
    let server = support::start_resolving(support::Resolver::Daytona, {
        let base = String::new();
        move |request: &Recorded| {
            let _ = &base;
            if request.path.starts_with("/sandbox") {
                MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
            } else {
                MockResponse::json(json!({}))
            }
        }
    })
    .await;

    let session = client(&server.base_url).create(None).await.expect("create");

    assert_eq!(session.manifest().root(), DEFAULT_DAYTONA_WORKSPACE_ROOT);
    assert_eq!(DEFAULT_DAYTONA_WORKSPACE_ROOT, "/home/daytona/workspace");
    server.stop();
}

/// `test_create_passes_only_option_env_vars_to_daytona`: the create body carries the client's
/// environment merged with the manifest's, plus the labels, target and resource request.
#[tokio::test]
async fn daytona_create_sends_the_environment_labels_and_resources() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        if request.path.starts_with("/sandbox") {
            MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
        } else {
            MockResponse::json(json!({}))
        }
    })
    .await;

    let client = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_organization_id("org-1")
            .with_target("eu")
            .with_name("fixed-name")
            .with_env("FROM_OPTIONS", "1")
            .with_label("owner", "tests")
            .with_auto_stop_interval(30)
            .with_resources(
                DaytonaSandboxResources::new()
                    .with_cpu(2)
                    .with_memory(4)
                    .with_disk(8),
            )
            .with_state_poll_interval(Duration::from_millis(1)),
    );
    client
        .create(Some(Manifest::new().with_env("FROM_MANIFEST", "2")))
        .await
        .expect("create");

    let request = server.find("POST", "/sandbox");
    assert_eq!(request.header("authorization"), Some("Bearer dtn_test"));
    assert_eq!(request.header("x-daytona-organization-id"), Some("org-1"));
    assert_eq!(
        request.header("x-daytona-source"),
        Some("openai-agents-sdk-rust")
    );
    let body = request.json();
    assert_eq!(body["name"], "fixed-name");
    assert_eq!(body["autoStopInterval"], 30);
    assert_eq!(body["target"], "eu");
    assert_eq!(body["labels"], json!({ "owner": "tests" }));
    assert_eq!(
        body["env"],
        json!({ "FROM_MANIFEST": "2", "FROM_OPTIONS": "1" })
    );
    assert_eq!(body["cpu"], 2);
    assert_eq!(body["memory"], 4);
    assert_eq!(body["disk"], 8);
    server.stop();
}

/// An image-based sandbox is built from a one-line Dockerfile, the way the vendor SDK's
/// `CreateSandboxFromImageParams` does; a named snapshot wins over it.
#[tokio::test]
async fn daytona_create_builds_from_an_image_or_a_named_snapshot() {
    for (snapshot, image, expected) in [
        (None, Some("ubuntu:24.04"), None),
        (Some("snap-1"), Some("ubuntu:24.04"), Some("snap-1")),
    ] {
        let server =
            support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
                if request.path.starts_with("/sandbox") {
                    MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
                } else {
                    MockResponse::json(json!({}))
                }
            })
            .await;
        let mut options = DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_state_poll_interval(Duration::from_millis(1));
        if let Some(snapshot) = snapshot {
            options = options.with_sandbox_snapshot_name(snapshot);
        }
        if let Some(image) = image {
            options = options.with_image(image);
        }
        DaytonaSandboxClient::with_options(options)
            .create(None)
            .await
            .expect("create");

        let body = server.find("POST", "/sandbox").json();
        match expected {
            Some(snapshot) => {
                assert_eq!(body["snapshot"], snapshot);
                assert!(body.get("buildInfo").is_none(), "{body}");
            }
            None => {
                assert_eq!(
                    body["buildInfo"],
                    json!({ "dockerfileContent": "FROM ubuntu:24.04" })
                );
            }
        }
        server.stop();
    }
}

/// A sandbox that never becomes ready fails the create with a start error naming the budget.
#[tokio::test]
async fn daytona_create_fails_when_the_sandbox_never_starts() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        if request.path.starts_with("/sandbox") {
            MockResponse::json(json!({ "id": "sbx-daytona", "state": "creating" }))
        } else {
            MockResponse::json(json!({}))
        }
    })
    .await;

    let error = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_create_timeout(Duration::from_millis(20))
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect_err("the sandbox never started");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.contains("did not become ready"),
        "{}",
        error.message
    );
    server.stop();
}

/// A sandbox that reports an error state fails immediately rather than polling to the deadline.
#[tokio::test]
async fn daytona_create_fails_fast_on_an_error_state() {
    for state in ["error", "build_failed"] {
        let server =
            support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
                if request.path.starts_with("/sandbox") {
                    MockResponse::json(json!({ "id": "sbx-daytona", "state": state }))
                } else {
                    MockResponse::json(json!({}))
                }
            })
            .await;

        let error = client(&server.base_url)
            .create(None)
            .await
            .expect_err("the sandbox failed to build");

        assert_eq!(error.code, ErrorCode::WorkspaceStartError);
        assert!(error.message.contains(state), "{}", error.message);
        server.stop();
    }
}

/// A missing API key is refused before any request is made.
#[tokio::test]
async fn daytona_rejects_a_missing_api_key() {
    // SAFETY: the test process sets its own environment before any client reads it.
    unsafe {
        std::env::remove_var("DAYTONA_API_KEY");
    }

    let error = DaytonaSandboxClient::new()
        .create(None)
        .await
        .expect_err("no API key");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}

// ---------------------------------------------------------------------------------------------
// Session start, exec and filesystem
// ---------------------------------------------------------------------------------------------

/// `test_start_prepares_workspace_root_before_runtime_helpers`: `start` creates the workspace
/// root from `/` before anything else runs in it.
#[tokio::test]
async fn daytona_start_prepares_the_workspace_root_first() {
    let server = support::start_resolving(support::Resolver::Daytona, {
        let base = String::new();
        move |request: &Recorded| {
            let _ = &base;
            let path = request.path.as_str();
            if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
                return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
            }
            if path.ends_with("/exec") {
                return command_result("", 0);
            }
            MockResponse::json(json!({}))
        }
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.start().await.expect("start");

    let first = commands(&server).into_iter().next().expect("a command");
    assert_eq!(first, "cd / && mkdir -p -- /home/daytona/workspace");
    // Each command runs in its own throwaway toolbox session, which is deleted afterwards.
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.method == "POST" && r.path.ends_with("/process/session")),
        "{:?}",
        server.routes()
    );
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.method == "DELETE" && r.path.contains("/process/session/")),
        "{:?}",
        server.routes()
    );
    server.stop();
}

/// `test_start_wraps_workspace_root_prepare_failure`
#[tokio::test]
async fn daytona_start_wraps_a_workspace_root_prepare_failure() {
    let server = support::start_resolving(support::Resolver::Daytona, {
        move |request: &Recorded| {
            let path = request.path.as_str();
            if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
                return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
            }
            if path.ends_with("/exec") {
                return MockResponse::json(json!({
                    "stdout": "",
                    "stderr": "mkdir: permission denied",
                    "exitCode": 1,
                }));
            }
            MockResponse::json(json!({}))
        }
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .start()
        .await
        .expect_err("the root could not be made");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.contains("create the Daytona workspace root"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_exec_merges_manifest_env_with_option_precedence`: the toolbox API takes one command
/// string, so the working directory and the environment are expressed in the command itself.
#[tokio::test]
async fn daytona_exec_merges_the_environment_into_the_command() {
    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "out", 0))
            .await;
    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_env("FROM_OPTIONS", "1")
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(Some(
        Manifest::new()
            .with_root("/workspace")
            .with_env("FROM_MANIFEST", "2"),
    ))
    .await
    .expect("create");

    let result = session
        .exec(ExecRequest::argv(["echo", "hi"]).with_env("PER_CALL", "3"))
        .await
        .expect("exec");

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout_text(), "out");
    let command = commands(&server).into_iter().next().expect("a command");
    assert!(command.starts_with("cd /workspace && "), "{command}");
    assert!(command.contains("FROM_MANIFEST=2"), "{command}");
    assert!(command.contains("FROM_OPTIONS=1"), "{command}");
    assert!(command.contains("PER_CALL=3"), "{command}");
    assert!(command.ends_with("echo hi"), "{command}");
    server.stop();
}

/// `test_exec_preserves_session_command_stdout_and_stderr`: separate streams survive the
/// round trip, which is the reason each command gets its own toolbox session.
#[tokio::test]
async fn daytona_exec_preserves_stdout_and_stderr() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/exec") {
            return MockResponse::json(json!({
                "stdout": "to stdout",
                "stderr": "to stderr",
                "exitCode": 7,
            }));
        }
        MockResponse::json(json!({}))
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let result = session
        .exec(ExecRequest::argv(["run"]))
        .await
        .expect("exec");

    assert_eq!(result.stdout_text(), "to stdout");
    assert_eq!(result.stderr_text(), "to stderr");
    assert_eq!(result.exit_code, 7);
    server.stop();
}

/// Daytona has no user parameter, so a `user=` command keeps its `sudo -u` wrapper.
#[tokio::test]
async fn daytona_exec_keeps_the_sudo_wrapper_for_a_user() {
    use openai_agents_sdk::sandbox::types::User;

    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session
        .exec(ExecRequest::argv(["id"]).with_user(Some(User::new("root"))))
        .await
        .expect("exec");

    let command = commands(&server).into_iter().next().expect("a command");
    assert!(command.contains("sudo -u root --"), "{command}");
    server.stop();
}

/// `test_exec_enforces_subsecond_caller_timeout`: a command that outlives its budget is an
/// `ExecTimeout` naming the command, and the throwaway toolbox session is still cleaned up.
#[tokio::test]
async fn daytona_exec_enforces_a_subsecond_caller_timeout() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/exec") {
            // The reply never arrives, so the caller's own budget has to fire.
            return MockResponse::streamed("application/json", vec![(Vec::new(), 5_000)]);
        }
        MockResponse::json(json!({}))
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["sleep", "60"]).with_timeout(Some(Duration::from_millis(50))))
        .await
        .expect_err("the command outlived its budget");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert!(error.message.contains("sleep 60"), "{}", error.message);
    // `test_exec_delete_session_cleanup_is_bounded`: the cleanup still runs.
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.method == "DELETE" && r.path.contains("/process/session/")),
        "{:?}",
        server.routes()
    );
    server.stop();
}

/// Reads and writes go through the toolbox files API with the resolved absolute path.
#[tokio::test]
async fn daytona_reads_and_writes_through_the_toolbox_files_api() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/files/download") {
            return MockResponse::bytes(b"contents".to_vec());
        }
        MockResponse::json(json!({}))
    })
    .await;
    let session = client(&server.base_url)
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");

    session
        .write_file("note.txt", b"hello")
        .await
        .expect("write");
    let read = session.read_file("note.txt").await.expect("read");

    assert_eq!(read, b"contents");
    let upload = server.find("POST", "/toolbox/sbx-daytona/toolbox/files/upload-v2");
    assert_eq!(
        upload.query_param("path").as_deref(),
        Some("/workspace/note.txt")
    );
    assert!(
        upload
            .header("content-type")
            .is_some_and(|value| value.starts_with("multipart/form-data")),
        "{:?}",
        upload.header("content-type")
    );
    let download = server.find("GET", "/toolbox/sbx-daytona/toolbox/files/download");
    assert_eq!(
        download.query_param("path").as_deref(),
        Some("/workspace/note.txt")
    );
    server.stop();
}

/// A missing file is a `WorkspaceReadNotFound`, not a transport error.
#[tokio::test]
async fn daytona_read_reports_a_missing_file() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        let path = request.path.as_str();
        if path == "/sandbox" || path == "/sandbox/sbx-daytona" {
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
        }
        if path.ends_with("/files/download") {
            return MockResponse::empty(404);
        }
        MockResponse::json(json!({}))
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .read_file("gone.txt")
        .await
        .expect_err("the file is gone");

    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    server.stop();
}

/// `test_normalize_path_rejects_workspace_escape_and_allows_absolute_in_root` and
/// `test_read_and_write_reject_paths_outside_workspace_root`: a path that leaves the workspace
/// root is refused before any request is made.
#[tokio::test]
async fn daytona_rejects_paths_outside_the_workspace_root() {
    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "", 0)).await;
    let session = client(&server.base_url)
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");
    let before = server.requests().len();

    for path in ["/etc/passwd", "../escape.txt"] {
        let error = session.read_file(path).await.unwrap_err();
        assert_eq!(
            error.code,
            ErrorCode::InvalidManifestPath,
            "{path}: {error:?}"
        );
    }
    // An absolute path that stays inside the root is accepted and normalized.
    let _ = session.read_file("/workspace/note.txt").await;
    assert!(
        server
            .requests()
            .iter()
            .skip(before)
            .filter(|r| r.path.ends_with("/files/download"))
            .count()
            <= 1,
        "{:?}",
        server.routes()
    );
    server.stop();
}

/// `test_running_*`: `running` reads the platform state and is true only for `started`.
#[tokio::test]
async fn daytona_running_reads_the_platform_state() {
    for (state, expected) in [("started", true), ("stopped", false)] {
        let gets = support::Counter::new();
        let server =
            support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
                if request.path == "/sandbox" {
                    return MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }));
                }
                if request.path == "/sandbox/sbx-daytona" {
                    // The first read is the create poll; later reads are the one under test.
                    let state = if gets.next("get") == 0 {
                        "started"
                    } else {
                        state
                    };
                    return MockResponse::json(json!({ "id": "sbx-daytona", "state": state }));
                }
                MockResponse::json(json!({}))
            })
            .await;
        let session = client(&server.base_url).create(None).await.expect("create");

        assert_eq!(session.running().await.expect("running"), expected);
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Exposed ports
// ---------------------------------------------------------------------------------------------

/// `test_resolve_exposed_port_uses_signed_preview_url`: the endpoint comes from the platform's
/// signed preview URL, whose signature lives in the query.
#[tokio::test]
async fn daytona_resolve_exposed_port_uses_a_signed_preview_url() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        if request.path.contains("/signed-preview-url") {
            return MockResponse::json(json!({
                "url": "https://8080-sbx-daytona.proxy.daytona.test/?token=signed",
            }));
        }
        MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
    })
    .await;
    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_exposed_port(8080)
            .with_exposed_port_url_ttl_secs(600)
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create");

    let endpoint = session.resolve_exposed_port(8080).await.expect("endpoint");

    assert_eq!(endpoint.host, "8080-sbx-daytona.proxy.daytona.test");
    assert_eq!(endpoint.port, 443);
    assert!(endpoint.tls);
    assert_eq!(endpoint.query, "token=signed");
    let request = server.find("GET", "/sandbox/sbx-daytona/ports/8080/signed-preview-url");
    assert_eq!(
        request.query_param("expiresInSeconds").as_deref(),
        Some("600")
    );
    server.stop();
}

/// `test_resolve_exposed_port_rejects_invalid_preview_urls`: an empty or unparsable URL is
/// reported as "the backend cannot route this port", not as a bad endpoint.
#[tokio::test]
async fn daytona_resolve_exposed_port_rejects_an_invalid_preview_url() {
    for url in ["", "not a url"] {
        let server =
            support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
                if request.path.contains("/signed-preview-url") {
                    return MockResponse::json(json!({ "url": url }));
                }
                MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
            })
            .await;
        let session = DaytonaSandboxClient::with_options(
            DaytonaSandboxClientOptions::new()
                .with_api_key("dtn_test")
                .with_api_url(&server.base_url)
                .with_exposed_port(8080)
                .with_state_poll_interval(Duration::from_millis(1)),
        )
        .create(None)
        .await
        .expect("create");

        let error = session
            .resolve_exposed_port(8080)
            .await
            .expect_err("the preview URL is unusable");

        assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Resume, stop and shutdown
// ---------------------------------------------------------------------------------------------

fn resumable_state() -> SandboxSessionState {
    SandboxSessionState::new(
        "daytona",
        "session-1",
        DEFAULT_DAYTONA_WORKSPACE_ROOT,
        &Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
    )
    .with_provider_id(Some("sbx-daytona".to_string()))
}

/// `test_resume_reconnects_paused_sandbox_and_preserves_state`: a stopped sandbox is started
/// again by id rather than replaced, and the session keeps its persisted identity.
#[tokio::test]
async fn daytona_resume_restarts_a_stopped_sandbox() {
    let states = support::Counter::new();
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        if request.path == "/sandbox/sbx-daytona" && request.method == "GET" {
            // The first read reports the stopped sandbox; later reads see it started.
            // Reads 0 and 1 are the resume lookup and the stopped-check.
            let state = if states.next("get") < 2 {
                "stopped"
            } else {
                "started"
            };
            return MockResponse::json(json!({ "id": "sbx-daytona", "state": state }));
        }
        MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
    })
    .await;

    let session = client(&server.base_url)
        .resume(
            &resumable_state(),
            Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
        )
        .await
        .expect("resume");

    assert_eq!(session.id(), "session-1");
    assert_eq!(server.count("POST", "/sandbox/sbx-daytona/start"), 1);
    assert!(
        server.find_opt("POST", "/sandbox").is_none(),
        "no replacement"
    );
    server.stop();
}

/// A sandbox that is already running is not restarted.
#[tokio::test]
async fn daytona_resume_leaves_a_running_sandbox_alone() {
    let server = support::start_resolving(support::Resolver::Daytona, move |_| {
        MockResponse::json(json!({ "id": "sbx-daytona", "state": "started" }))
    })
    .await;

    client(&server.base_url)
        .resume(
            &resumable_state(),
            Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
        )
        .await
        .expect("resume");

    assert_eq!(server.count("POST", "/sandbox/sbx-daytona/start"), 0);
    server.stop();
}

/// `test_resume_recreates_unpaused_sandbox_when_reconnect_fails`: a sandbox that can no longer
/// be read is replaced through the ordinary create path.
#[tokio::test]
async fn daytona_resume_recreates_a_sandbox_that_cannot_be_read() {
    let server = support::start_resolving(support::Resolver::Daytona, move |request: &Recorded| {
        if request.method == "GET" && request.path == "/sandbox/sbx-daytona" {
            return MockResponse::empty(404);
        }
        MockResponse::json(json!({ "id": "sbx-new", "state": "started" }))
    })
    .await;

    client(&server.base_url)
        .resume(
            &resumable_state(),
            Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
        )
        .await
        .expect("resume");

    // The replacement keeps the persisted session id as its sandbox name.
    assert_eq!(server.find("POST", "/sandbox").json()["name"], "session-1");
    server.stop();
}

/// A state another backend wrote is refused before any request is made.
#[tokio::test]
async fn daytona_resume_rejects_a_foreign_backend_state() {
    let server = support::start_resolving(support::Resolver::Daytona, move |_| {
        MockResponse::json(json!({}))
    })
    .await;
    let state = SandboxSessionState::new("e2b", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("sbx-1".to_string()));

    let error = client(&server.base_url)
        .resume(&state, Manifest::new())
        .await
        .expect_err("an e2b state is not a daytona state");

    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    assert!(server.requests().is_empty(), "{:?}", server.routes());
    server.stop();
}

/// A state without a sandbox id cannot be resumed.
#[tokio::test]
async fn daytona_resume_requires_a_recorded_sandbox_id() {
    let server = support::start_resolving(support::Resolver::Daytona, move |_| {
        MockResponse::json(json!({}))
    })
    .await;
    let state = SandboxSessionState::new(
        "daytona",
        "session-1",
        DEFAULT_DAYTONA_WORKSPACE_ROOT,
        &Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
    );

    let error = client(&server.base_url)
        .resume(
            &state,
            Manifest::new().with_root(DEFAULT_DAYTONA_WORKSPACE_ROOT),
        )
        .await
        .expect_err("no sandbox id");

    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    server.stop();
}

/// A `stop` stops the sandbox without deleting it: a stopped Daytona sandbox keeps its
/// filesystem and can be started again by id.
#[tokio::test]
async fn daytona_stop_stops_the_sandbox_without_deleting_it() {
    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.stop().await.expect("stop");

    assert_eq!(server.count("POST", "/sandbox/sbx-daytona/stop"), 1);
    assert_eq!(server.count("DELETE", "/sandbox/sbx-daytona"), 0);
    server.stop();
}

/// A `shutdown` deletes the sandbox, exactly once however often it is called.
#[tokio::test]
async fn daytona_shutdown_deletes_the_sandbox_exactly_once() {
    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.shutdown().await.expect("shutdown");
    session.shutdown().await.expect("shutdown again");

    assert_eq!(server.count("DELETE", "/sandbox/sbx-daytona"), 1);
    server.stop();
}

/// With `pause_on_exit` the sandbox is stopped rather than deleted, so a later run can resume it.
#[tokio::test]
async fn daytona_shutdown_stops_the_sandbox_when_pause_on_exit_is_set() {
    let server =
        support::start_resolving(support::Resolver::Daytona, handler(String::new(), "", 0)).await;
    let session = DaytonaSandboxClient::with_options(
        DaytonaSandboxClientOptions::new()
            .with_api_key("dtn_test")
            .with_api_url(&server.base_url)
            .with_pause_on_exit(true)
            .with_state_poll_interval(Duration::from_millis(1)),
    )
    .create(None)
    .await
    .expect("create");

    session.shutdown().await.expect("shutdown");

    assert_eq!(server.count("POST", "/sandbox/sbx-daytona/stop"), 1);
    assert_eq!(server.count("DELETE", "/sandbox/sbx-daytona"), 0);
    server.stop();
}
