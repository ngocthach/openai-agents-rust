//! Ports of `tests/extensions/sandbox/test_e2b.py`.
//!
//! Python patches the vendor `e2b` SDK and asserts on the Python calls it recorded
//! (`sandbox.commands.run(..., cwd=..., envs=...)`, `sandbox.files.make_dir`, `sandbox.pause`).
//! The Rust backend speaks the envd Connect protocol and the E2B platform REST API directly, so
//! the same behaviour is observed through the requests a local mock server records. Preflight
//! helpers that take `&dyn SandboxSession` are driven with the scripted session, exactly as
//! Python drives them with `_FakeMountSession`.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::e2b::rclone::{
    ensure_fuse_support, rclone_pattern_for_session,
};
use openai_agents_sdk::extensions::sandbox::e2b::{
    E2BCloudBucketMountStrategy, E2BSandboxClient, E2BSandboxClientOptions, E2BSandboxType,
    E2BTimeoutAction, WorkspacePersistence,
};
use openai_agents_sdk::sandbox::mounts::{MountStrategy, RcloneMountMode, RcloneMountPattern};
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSessionState,
};
use serde_json::{Value, json};
use support::{MockResponse, MockServer, Recorded};

// ---------------------------------------------------------------------------------------------
// Scripted-session helpers, the Rust stand-in for Python's `_FakeMountSession`
// ---------------------------------------------------------------------------------------------

fn exec_result(exit_code: i32, stdout: &str) -> ExecResult {
    ExecResult {
        exit_code,
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
    }
}

fn exec_ok() -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_result(0, ""))
}

fn exec_out(stdout: &str) -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_result(0, stdout))
}

fn exec_fail() -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_result(1, ""))
}

/// Every command the scripted session was asked to run, rendered the way Python's
/// `_FakeMountSession.exec_calls` renders them: the argv joined with single spaces.
fn exec_calls(
    session: &openai_agents_sdk::sandbox::testing::ScriptedSandboxSession,
) -> Vec<String> {
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

/// `test_e2b_mount_strategy_type_and_default_pattern`
#[test]
fn e2b_mount_strategy_type_and_default_pattern() {
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "/", "[bucket]\ntype = s3\n");

    assert_eq!(strategy.strategy_type(), "e2b_cloud_bucket");
    assert_eq!(strategy.pattern.mode, RcloneMountMode::Fuse);
}

// ---------------------------------------------------------------------------------------------
// FUSE and rclone preflight
// ---------------------------------------------------------------------------------------------

/// `test_e2b_ensure_fuse_uses_root_chmod`: the kernel probe runs as the command user and the
/// `/dev/fuse` fix-up runs as root.
#[tokio::test]
async fn e2b_ensure_fuse_uses_root_chmod() {
    let session = scripted_sandbox_session([exec_ok(), exec_ok(), exec_ok()]);

    ensure_fuse_support(&session, true)
        .await
        .expect("fuse support");

    let calls = exec_calls(&session);
    assert_eq!(
        calls[0],
        "sh -lc test -c /dev/fuse && grep -qw fuse /proc/filesystems"
    );
    assert_eq!(
        calls[1],
        "sh -lc command -v fusermount3 >/dev/null 2>&1 || test -x /usr/local/bin/fusermount3"
    );
    assert_eq!(
        calls[2],
        "sudo -u root -- sh -lc chmod a+rw /dev/fuse && touch /etc/fuse.conf && (grep -qxF \
         user_allow_other /etc/fuse.conf || printf '\\nuser_allow_other\\n' >> /etc/fuse.conf)"
    );
}

/// A sandbox without `/dev/fuse` is refused before anything is installed.
#[tokio::test]
async fn e2b_ensure_fuse_requires_the_kernel_module() {
    let session = scripted_sandbox_session([exec_fail()]);

    let error = ensure_fuse_support(&session, true)
        .await
        .expect_err("no /dev/fuse");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("/dev/fuse"), "{}", error.message);
    assert_eq!(session.calls().len(), 1);
}

/// `test_e2b_ensure_rclone_installs_verified_release`: the exact command sequence, including the
/// root-wrapped prerequisite installs and the pinned, checksum-verified release installer.
#[tokio::test]
async fn e2b_ensure_rclone_installs_verified_release() {
    use openai_agents_sdk::extensions::sandbox::e2b::rclone::ensure_rclone;

    let session = scripted_sandbox_session([
        exec_fail(),          // rclone missing
        exec_ok(),            // apt-get present
        exec_out("x86_64\n"), // supported release architecture
        exec_ok(),            // apt-get update succeeds
        exec_ok(),            // prerequisite install succeeds
        exec_ok(),            // verified rclone install succeeds
        exec_ok(),            // rclone now present
    ]);

    ensure_rclone(&session, true).await.expect("install rclone");

    let calls = exec_calls(&session);
    assert_eq!(
        calls[0],
        "sh -lc command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone"
    );
    assert_eq!(
        calls[1],
        "sh -lc command -v apt-get >/dev/null 2>&1 || test -x /usr/local/bin/apt-get"
    );
    assert_eq!(calls[2], "uname -m");
    assert_eq!(
        calls[3],
        "sudo -u root -- sh -lc DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes apt-get -o \
         Dpkg::Use-Pty=0 update -qq"
    );
    assert_eq!(
        calls[4],
        "sudo -u root -- sh -lc DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes apt-get -o \
         Dpkg::Use-Pty=0 install -y -qq ca-certificates coreutils curl unzip"
    );
    assert!(
        calls[5].starts_with("sudo -u root -- sh -lc set -eu\n"),
        "{}",
        calls[5]
    );
    assert!(
        calls[5].contains("sha256sum --check --strict -"),
        "{}",
        calls[5]
    );
    assert!(!calls[5].contains("rclone.org/install.sh"), "{}", calls[5]);
    assert_eq!(
        calls[6],
        "sh -lc command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone"
    );
}

/// `test_e2b_rclone_pattern_adds_fuse_access_args`
#[tokio::test]
async fn e2b_rclone_pattern_adds_fuse_access_args() {
    let session = scripted_sandbox_session([exec_out("1000\n1000\n")]);

    let pattern = rclone_pattern_for_session(&session, &RcloneMountPattern::new())
        .await
        .expect("resolve the pattern");

    assert_eq!(
        pattern.extra_args,
        vec!["--allow-other", "--uid", "1000", "--gid", "1000"]
    );
}

/// `test_e2b_rclone_pattern_preserves_explicit_access_args`
#[tokio::test]
async fn e2b_rclone_pattern_preserves_explicit_access_args() {
    let session = scripted_sandbox_session([exec_out("1000\n1000\n")]);
    let source = RcloneMountPattern::new().with_extra_args([
        "--allow-other",
        "--uid",
        "123",
        "--gid",
        "456",
        "--buffer-size",
        "0",
    ]);

    let pattern = rclone_pattern_for_session(&session, &source)
        .await
        .expect("resolve the pattern");

    assert_eq!(
        pattern.extra_args,
        vec![
            "--allow-other",
            "--uid",
            "123",
            "--gid",
            "456",
            "--buffer-size",
            "0"
        ]
    );
}

/// A sandbox that cannot report its ids still mounts, without the ownership flags.
#[tokio::test]
async fn e2b_rclone_pattern_tolerates_an_unreadable_user_id() {
    let session = scripted_sandbox_session([exec_fail()]);

    let pattern = rclone_pattern_for_session(&session, &RcloneMountPattern::new())
        .await
        .expect("resolve the pattern");

    assert_eq!(pattern.extra_args, vec!["--allow-other"]);
}

/// A non-FUSE pattern is left exactly as configured: an NFS mount is served, not owned.
#[tokio::test]
async fn e2b_rclone_pattern_leaves_a_non_fuse_mode_alone() {
    let session = scripted_sandbox_session([]);
    let source = RcloneMountPattern::nfs();

    let pattern = rclone_pattern_for_session(&session, &source)
        .await
        .expect("resolve the pattern");

    assert!(pattern.extra_args.is_empty(), "{:?}", pattern.extra_args);
    assert_eq!(session.calls().len(), 0);
}

// ---------------------------------------------------------------------------------------------
// envd Connect wire helpers
// ---------------------------------------------------------------------------------------------

/// Encode one Connect envelope: a flag byte, a big-endian length, then the JSON payload.
fn connect_frame(flags: u8, value: Value) -> Vec<u8> {
    let payload = serde_json::to_vec(&value).expect("serializable");
    let mut frame = vec![flags];
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// A complete `process.Process/Start` response: stdout, the end event, then the trailer.
fn process_stream(stdout: &str, exit_code: i64) -> MockResponse {
    let mut body = connect_frame(0, json!({ "event": { "start": { "pid": 4242 } } }));
    body.extend_from_slice(&connect_frame(
        0,
        json!({ "event": { "data": { "stdout": support::b64(stdout) } } }),
    ));
    body.extend_from_slice(&connect_frame(
        0,
        json!({ "event": { "end": { "exited": true, "exitCode": exit_code } } }),
    ));
    body.extend_from_slice(&connect_frame(2, json!({})));
    MockResponse {
        status: 200,
        content_type: "application/connect+json",
        body,
        stream: Vec::new(),
        headers: Vec::new(),
    }
}

/// The `process.StartRequest` body of a recorded `process.Process/Start` call.
fn start_request(request: &Recorded) -> Value {
    // The body is one Connect envelope: a flag byte, a big-endian length, then the payload.
    assert!(request.body.len() > 5, "{:?}", request.body);
    serde_json::from_slice(&request.body[5..]).expect("a JSON start request")
}

/// The command line of a recorded `process.Process/Start` call.
fn start_command(request: &Recorded) -> String {
    let value = start_request(request);
    let process = &value["process"];
    let mut argv = vec![process["cmd"].as_str().unwrap_or_default().to_string()];
    for arg in process["args"].as_array().cloned().unwrap_or_default() {
        argv.push(arg.as_str().unwrap_or_default().to_string());
    }
    argv.join(" ")
}

/// Every `process.Process/Start` the mock server saw.
fn start_requests(server: &MockServer) -> Vec<Recorded> {
    server.all("POST", "/process.Process/Start")
}

fn client(base_url: &str) -> E2BSandboxClient {
    E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(base_url)
            .with_sandbox_base_url(base_url),
    )
}

/// The create reply the platform API returns.
fn created_sandbox() -> Value {
    json!({ "sandboxID": "sb-123", "envdAccessToken": "envd-token" })
}

// ---------------------------------------------------------------------------------------------
// Client create
// ---------------------------------------------------------------------------------------------

/// `test_e2b_client_create_enables_public_traffic_for_exposed_ports`: an exposed port turns on
/// public traffic, and a pause-on-timeout sandbox carries the auto-resume lifecycle.
#[tokio::test]
async fn e2b_client_create_enables_public_traffic_for_exposed_ports() {
    let server = support::start_resolving(support::Resolver::E2b, move |request| {
        match request.path.as_str() {
            "/v2/sandboxes" => MockResponse::json(created_sandbox()),
            _ => MockResponse::json(json!({})),
        }
    })
    .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_sandbox_type(E2BSandboxType::E2B)
            .with_exposed_port(8765),
    );
    let session = client.create(None).await.expect("create");

    let body = server.find("POST", "/v2/sandboxes").json();
    assert_eq!(body["network"], json!({ "allowPublicTraffic": true }));
    assert_eq!(body["autoPause"], json!(true));
    assert_eq!(body["autoResume"], json!({ "enabled": true }));
    assert_eq!(session.exposed_ports(), vec![8765]);
    server.stop();
}

/// `test_e2b_client_create_omits_auto_resume_for_kill_timeout`: a kill-on-timeout sandbox is not
/// auto-paused and carries no auto-resume lifecycle.
#[tokio::test]
async fn e2b_client_create_omits_auto_resume_for_kill_timeout() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_on_timeout(E2BTimeoutAction::Kill),
    );
    client.create(None).await.expect("create");

    let body = server.find("POST", "/v2/sandboxes").json();
    assert_eq!(body["autoPause"], json!(false));
    assert!(body.get("autoResume").is_none(), "{body}");
    server.stop();
}

/// `test_e2b_client_create_passes_mcp_config`: the MCP server map reaches the create body
/// unchanged, including the nested per-server options.
#[tokio::test]
async fn e2b_client_create_passes_mcp_config() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_mcp_server("exa", [("apiKey", "exa-key")])
            .with_mcp_server(
                "browserbase",
                [
                    ("apiKey", "browserbase-key"),
                    ("geminiApiKey", "gemini-key"),
                    ("projectId", "project-id"),
                ],
            ),
    );
    client.create(None).await.expect("create");

    let body = server.find("POST", "/v2/sandboxes").json();
    assert_eq!(
        body["mcp"],
        json!({
            "browserbase": {
                "apiKey": "browserbase-key",
                "geminiApiKey": "gemini-key",
                "projectId": "project-id",
            },
            "exa": { "apiKey": "exa-key" },
        })
    );
    server.stop();
}

/// The create body carries the template, timeout, metadata and the merged environment.
#[tokio::test]
async fn e2b_client_create_sends_the_template_metadata_and_environment() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_template("custom-template")
            .with_timeout_secs(120)
            .with_metadata("owner", "tests")
            .with_env("FROM_OPTIONS", "1")
            .with_allow_internet_access(false),
    );
    let manifest = Manifest::new().with_env("FROM_MANIFEST", "2");
    client.create(Some(manifest)).await.expect("create");

    let request = server.find("POST", "/v2/sandboxes");
    assert_eq!(request.header("x-api-key"), Some("e2b_test"));
    let body = request.json();
    assert_eq!(body["templateID"], "custom-template");
    assert_eq!(body["timeout"], 120);
    assert_eq!(body["metadata"], json!({ "owner": "tests" }));
    assert_eq!(body["allow_internet_access"], json!(false));
    assert_eq!(
        body["envVars"],
        json!({ "FROM_MANIFEST": "2", "FROM_OPTIONS": "1" })
    );
    server.stop();
}

/// The default template follows the selected sandbox type, as Python's
/// `_default_template_for_type` does.
#[tokio::test]
async fn e2b_client_create_defaults_the_template_to_the_sandbox_type() {
    for (sandbox_type, template) in [
        (E2BSandboxType::E2B, "base"),
        (E2BSandboxType::CodeInterpreter, "code-interpreter-v1"),
    ] {
        let server = support::start_resolving(support::Resolver::E2b, move |_| {
            MockResponse::json(created_sandbox())
        })
        .await;
        let client = E2BSandboxClient::with_options(
            E2BSandboxClientOptions::new()
                .with_api_key("e2b_test")
                .with_api_base_url(&server.base_url)
                .with_sandbox_base_url(&server.base_url)
                .with_sandbox_type(sandbox_type),
        );
        client.create(None).await.expect("create");
        assert_eq!(
            server.find("POST", "/v2/sandboxes").json()["templateID"],
            template
        );
        server.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Session start, exec and filesystem
// ---------------------------------------------------------------------------------------------

/// A handler that answers the create, the workspace-root preparation and one command. The
/// sandbox starts without a workspace root, so the `test -d` probe `start` runs fails.
fn session_handler(stdout: &'static str, exit_code: i64) -> impl Fn(&Recorded) -> MockResponse {
    move |request: &Recorded| match request.path.as_str() {
        "/v2/sandboxes" => MockResponse::json(created_sandbox()),
        "/filesystem.Filesystem/MakeDir" => MockResponse::json(json!({})),
        "/process.Process/Start" if start_command(request) == "test -d /workspace" => {
            process_stream("", 1)
        }
        "/process.Process/Start" => process_stream(stdout, exit_code),
        _ => MockResponse::json(json!({})),
    }
}

/// `test_e2b_start_prepares_workspace_root_for_command_cwd`: `start` creates the root with the
/// filesystem API first, then proves it is usable as a working directory with `mkdir -p --` run
/// from `/`, and only then do commands run with the root as their `cwd`.
#[tokio::test]
async fn e2b_start_prepares_workspace_root_for_command_cwd() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.start().await.expect("start");
    session
        .exec(ExecRequest::argv(["pwd"]))
        .await
        .expect("exec");

    let make_dir = server.find("POST", "/filesystem.Filesystem/MakeDir");
    assert_eq!(make_dir.json()["path"], "/workspace");
    assert_eq!(make_dir.header("e2b-sandbox-id"), Some("sb-123"));
    assert_eq!(make_dir.header("x-access-token"), Some("envd-token"));

    // The path resolver's own probe is not one of the commands under test.
    let starts: Vec<Recorded> = start_requests(&server)
        .into_iter()
        .filter(|request| !support::is_resolver_request(request))
        .collect();
    assert_eq!(start_command(&starts[0]), "test -d /workspace");
    assert!(start_request(&starts[0])["process"]["cwd"].is_null());
    assert_eq!(start_command(&starts[1]), "mkdir -p -- /workspace");
    assert_eq!(start_request(&starts[1])["process"]["cwd"], "/");
    assert_eq!(start_command(&starts[2]), "pwd");
    assert_eq!(start_request(&starts[2])["process"]["cwd"], "/workspace");
    server.stop();
}

/// `test_e2b_start_raises_on_nonzero_workspace_root_setup_exit`
#[tokio::test]
async fn e2b_start_raises_on_nonzero_workspace_root_setup_exit() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 2)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session.start().await.expect_err("the root setup failed");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.contains("prepare the E2B workspace root"),
        "{}",
        error.message
    );
    server.stop();
}

/// A filesystem API that refuses the root fails `start` rather than running commands blind.
#[tokio::test]
async fn e2b_start_raises_when_the_filesystem_api_refuses_the_root() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/filesystem.Filesystem/MakeDir" => {
                    MockResponse::status(500, json!({ "code": "internal", "message": "nope" }))
                }
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .start()
        .await
        .expect_err("the root could not be made");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.contains("create the E2B workspace root"),
        "{}",
        error.message
    );
    assert!(
        start_requests(&server)
            .iter()
            .all(|request| start_command(request) == "test -d /workspace"),
        "only the root probe ran"
    );
    server.stop();
}

/// `test_e2b_exec_uses_manifest_root_after_workspace_ready`: a command runs in the manifest root
/// and carries the session environment.
#[tokio::test]
async fn e2b_exec_uses_the_manifest_root_and_session_environment() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("out", 0)).await;
    let options = E2BSandboxClientOptions::new()
        .with_api_key("e2b_test")
        .with_api_base_url(&server.base_url)
        .with_sandbox_base_url(&server.base_url)
        .with_env("FROM_OPTIONS", "1");
    let session = E2BSandboxClient::with_options(options)
        .create(Some(Manifest::new().with_env("FROM_MANIFEST", "2")))
        .await
        .expect("create");
    session.start().await.expect("start");

    let result = session
        .exec(ExecRequest::argv(["find", "."]).with_env("PER_CALL", "3"))
        .await
        .expect("exec");

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout_text(), "out");
    let requests = start_requests(&server);
    let find = requests
        .iter()
        .find(|request| start_command(request) == "find .")
        .expect("the find command");
    let value = start_request(find);
    assert_eq!(value["process"]["cmd"], "find");
    assert_eq!(value["process"]["args"], json!(["."]));
    assert_eq!(value["process"]["cwd"], "/workspace");
    assert_eq!(
        value["process"]["envs"],
        json!({ "FROM_MANIFEST": "2", "FROM_OPTIONS": "1", "PER_CALL": "3" })
    );
    server.stop();
}

/// envd takes the system user out of band, so a `user=` command is never `sudo`-wrapped.
#[tokio::test]
async fn e2b_exec_sends_the_user_as_basic_auth_rather_than_sudo() {
    use openai_agents_sdk::sandbox::types::User;

    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session
        .exec(ExecRequest::argv(["id"]).with_user(Some(User::new("root"))))
        .await
        .expect("exec");

    let request = &start_requests(&server)[0];
    assert_eq!(start_command(request), "id");
    // `root:` with an empty password, base64-encoded.
    assert_eq!(
        request.header("authorization"),
        Some(format!("Basic {}", support::b64("root:")).as_str())
    );
    server.stop();
}

/// A command that envd reports as failing keeps its exit code rather than raising.
#[tokio::test]
async fn e2b_exec_reports_a_nonzero_exit_code() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 3)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let result = session
        .exec(ExecRequest::argv(["false"]))
        .await
        .expect("exec");

    assert_eq!(result.exit_code, 3);
    assert!(!result.ok());
    server.stop();
}

/// `test_e2b_exec_transport_preserves_provider_details`: a refused `Start` is a transport error
/// that keeps the provider's own message.
#[tokio::test]
async fn e2b_exec_transport_preserves_provider_details() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/process.Process/Start" => MockResponse::status(
                    502,
                    json!({ "code": "unavailable", "message": "sandbox gone" }),
                ),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["pwd"]))
        .await
        .expect_err("the provider refused the command");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("sandbox gone"), "{}", error.message);
    server.stop();
}

/// A stream cut short before the end event is a transport error, never a silent success: the
/// exit code is simply not known.
#[tokio::test]
async fn e2b_exec_refuses_a_stream_without_an_end_event() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/process.Process/Start" => MockResponse {
                    status: 200,
                    content_type: "application/connect+json",
                    body: connect_frame(0, json!({ "event": { "start": { "pid": 1 } } })),
                    stream: Vec::new(),
                    headers: Vec::new(),
                },
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["pwd"]))
        .await
        .expect_err("the stream never finished");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(
        error.message.contains("ended before the process finished"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_e2b_exec_timeout_preserves_provider_details`: a command that outlives its timeout is an
/// `ExecTimeout`, and the message names the command and the limit.
#[tokio::test]
async fn e2b_exec_timeout_preserves_provider_details() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                // The body never finishes, so the client's own timeout has to fire.
                "/process.Process/Start" => MockResponse::streamed(
                    "application/connect+json",
                    vec![
                        (
                            connect_frame(0, json!({ "event": { "start": { "pid": 1 } } })),
                            0,
                        ),
                        (Vec::new(), 5_000),
                    ],
                ),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .exec(ExecRequest::argv(["sleep", "60"]).with_timeout(Some(Duration::from_millis(50))))
        .await
        .expect_err("the command outlived its timeout");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert!(error.message.contains("sleep 60"), "{}", error.message);
    // envd is told the deadline too, so it can stop the process on its side.
    assert_eq!(
        start_requests(&server)[0].header("connect-timeout-ms"),
        Some("50")
    );
    server.stop();
}

/// `mkdir` on an existing directory and `rm` on a missing path are not failures: the session
/// contract is `mkdir -p` and `rm -f`, which is what the vendor SDK gives Python.
#[tokio::test]
async fn e2b_tolerates_an_existing_directory_and_a_missing_path() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/filesystem.Filesystem/MakeDir" => {
                    MockResponse::status(409, json!({ "code": "already_exists" }))
                }
                "/filesystem.Filesystem/Remove" => {
                    MockResponse::status(404, json!({ "code": "not_found" }))
                }
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session
        .mkdir("sub", true)
        .await
        .expect("mkdir is idempotent");
    // A recursive removal uses envd's `Remove`; a non-recursive one runs `rm -f` instead, so a
    // model removing "a file" cannot wipe a directory tree.
    session.rm("gone.txt", true).await.expect("rm -f semantics");

    assert_eq!(
        server.find("POST", "/filesystem.Filesystem/MakeDir").json()["path"],
        "/workspace/sub"
    );
    assert_eq!(
        server.find("POST", "/filesystem.Filesystem/Remove").json()["path"],
        "/workspace/gone.txt"
    );
    server.stop();
}

/// Any other filesystem error is still an error.
#[tokio::test]
async fn e2b_surfaces_a_filesystem_error_that_is_not_tolerated() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/filesystem.Filesystem/MakeDir" => {
                    MockResponse::status(403, json!({ "code": "permission_denied" }))
                }
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .mkdir("sub", true)
        .await
        .expect_err("permission denied");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    server.stop();
}

/// `test_e2b_running_checks_remote_after_workspace_ready`: once `start` has prepared the
/// workspace root, `running` reads the platform state and is false for anything but `running`.
#[tokio::test]
async fn e2b_running_reads_the_platform_state() {
    for (state, expected) in [("running", true), ("paused", false)] {
        let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
            if request.method == "GET" {
                MockResponse::json(json!({ "state": state }))
            } else {
                session_handler("", 0)(request)
            }
        })
        .await;
        let session = client(&server.base_url).create(None).await.expect("create");
        session.start().await.expect("start");

        assert_eq!(session.running().await.expect("running"), expected);
        assert_eq!(server.find("GET", "/sandboxes/sb-123").method, "GET");
        server.stop();
    }
}

/// `test_e2b_running_requires_workspace_root_ready`: a session whose workspace root has not been
/// prepared is not running, and the platform is not even asked.
#[tokio::test]
async fn e2b_running_requires_the_workspace_root_to_be_ready() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.method == "GET" {
            MockResponse::json(json!({ "state": "running" }))
        } else {
            session_handler("", 0)(request)
        }
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    assert!(!session.running().await.expect("running"));
    assert_eq!(server.count("GET", "/sandboxes/sb-123"), 0);

    session.start().await.expect("start");
    assert!(session.running().await.expect("running"));
    session.stop().await.expect("stop");
    assert!(!session.running().await.expect("running after stop"));
    server.stop();
}

/// `test_e2b_exec_omits_cwd_until_workspace_ready`: before `start` has prepared the workspace
/// root a command carries no working directory.
#[tokio::test]
async fn e2b_exec_omits_the_cwd_until_the_workspace_root_is_ready() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session
        .exec(ExecRequest::argv(["find", "."]))
        .await
        .expect("exec before start");

    let requests = start_requests(&server);
    let find = requests
        .iter()
        .find(|request| start_command(request) == "find .")
        .expect("the find command");
    assert!(start_request(find)["process"].get("cwd").is_none());
    server.stop();
}

/// A platform that cannot be reached reports "not running" rather than failing the caller.
#[tokio::test]
async fn e2b_running_reports_false_when_the_platform_refuses() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.method == "GET" {
            MockResponse::empty(503)
        } else {
            session_handler("", 0)(request)
        }
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");
    session.start().await.expect("start");

    assert!(!session.running().await.expect("running"));
    server.stop();
}

/// `test_e2b_resolve_exposed_port_uses_backend_host`: the endpoint is built from the sandbox id
/// and the configured domain, over TLS on 443, with no round trip.
#[tokio::test]
async fn e2b_resolve_exposed_port_uses_backend_host() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_domain("sandbox.example.test")
            .with_exposed_port(8765),
    )
    .create(None)
    .await
    .expect("create");

    let endpoint = session.resolve_exposed_port(8765).await.expect("endpoint");

    assert_eq!(endpoint.host, "8765-sb-123.sandbox.example.test");
    assert_eq!(endpoint.port, 443);
    assert!(endpoint.tls);
    server.stop();
}

/// A port that was never configured is refused, naming the ports that were.
#[tokio::test]
async fn e2b_resolve_exposed_port_refuses_an_unconfigured_port() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_exposed_port(8765),
    )
    .create(None)
    .await
    .expect("create");

    let error = session
        .resolve_exposed_port(9999)
        .await
        .expect_err("9999 was never exposed");

    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------------------------

fn resumable_state() -> SandboxSessionState {
    SandboxSessionState::new("e2b", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("sb-persisted".to_string()))
}

/// `test_e2b_resume_reuses_paused_timeout_lifecycle_sandbox`: a reachable sandbox is reconnected
/// by id, and no replacement is created.
#[tokio::test]
async fn e2b_resume_reconnects_to_the_recorded_sandbox() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes/sb-persisted/connect" => MockResponse::json(json!({
                    "sandboxID": "sb-persisted",
                    "envdAccessToken": "fresh-token",
                })),
                _ => MockResponse::json(created_sandbox()),
            },
        )
        .await;

    let session = client(&server.base_url)
        .resume(&resumable_state(), Manifest::new())
        .await
        .expect("resume");

    assert_eq!(
        server
            .find("POST", "/v2/sandboxes/sb-persisted/connect")
            .header("x-api-key"),
        Some("e2b_test")
    );
    assert!(
        server.find_opt("POST", "/v2/sandboxes").is_none(),
        "no replacement"
    );
    assert_eq!(session.id(), "session-1");
    server.stop();
}

/// `test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp`: a sandbox that can no
/// longer be reattached is replaced through the ordinary create path, which keeps the configured
/// template, timeout and MCP servers.
#[tokio::test]
async fn e2b_resume_recreates_a_sandbox_that_cannot_be_reattached() {
    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes/sb-persisted/connect" => MockResponse::empty(404),
                "/v2/sandboxes" => MockResponse::json(json!({ "sandboxID": "sb-recreated" })),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;

    let client = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_timeout_secs(15)
            .with_mcp_server("exa", [("apiKey", "exa-key")]),
    );
    client
        .resume(&resumable_state(), Manifest::new())
        .await
        .expect("resume");

    let body = server.find("POST", "/v2/sandboxes").json();
    assert_eq!(body["timeout"], 15);
    assert_eq!(body["mcp"], json!({ "exa": { "apiKey": "exa-key" } }));
    server.stop();
}

/// A state another backend wrote is refused before any request is made.
#[tokio::test]
async fn e2b_resume_rejects_a_foreign_backend_state() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let state = SandboxSessionState::new("daytona", "session-1", "/workspace", &Manifest::new())
        .with_provider_id(Some("sb-1".to_string()));

    let error = client(&server.base_url)
        .resume(&state, Manifest::new())
        .await
        .expect_err("a daytona state is not an e2b state");

    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    assert!(server.requests().is_empty(), "{:?}", server.routes());
    server.stop();
}

/// A state without a sandbox id cannot be resumed.
#[tokio::test]
async fn e2b_resume_requires_a_recorded_sandbox_id() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let state = SandboxSessionState::new("e2b", "session-1", "/workspace", &Manifest::new());

    let error = client(&server.base_url)
        .resume(&state, Manifest::new())
        .await
        .expect_err("no sandbox id");

    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Workspace persistence
// ---------------------------------------------------------------------------------------------

/// `test_e2b_persist_workspace_excludes_runtime_skip_paths`: the archive command runs from the
/// workspace root and excludes the manifest's ephemeral paths.
#[tokio::test]
async fn e2b_persist_workspace_excludes_ephemeral_paths() {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    // A real, empty gzipped tar so the archive validation the session applies is satisfied.
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    {
        let mut builder = tar::Builder::new(&mut encoder);
        builder.finish().expect("finish the tar");
    }
    let archive = encoder.finish().expect("finish the gzip");
    let encoded = support::b64(&archive);

    let server =
        support::start_resolving(
            support::Resolver::E2b,
            move |request: &Recorded| match request.path.as_str() {
                "/v2/sandboxes" => MockResponse::json(created_sandbox()),
                "/process.Process/Start" => process_stream(&encoded, 0),
                _ => MockResponse::json(json!({})),
            },
        )
        .await;
    let manifest = Manifest::new().with_entry(
        "cache",
        openai_agents_sdk::sandbox::entries::Entry::dir().with_ephemeral(true),
    );
    let session = client(&server.base_url)
        .create(Some(manifest))
        .await
        .expect("create");

    let persisted = session.persist_workspace().await.expect("persist");

    assert_eq!(persisted, archive);
    let command = start_command(&start_requests(&server)[0]);
    assert!(
        command.starts_with("sh -lc tar -czf - -C /workspace"),
        "{command}"
    );
    assert!(command.contains("--exclude ./cache"), "{command}");
    assert!(command.ends_with(". | base64 -w0"), "{command}");
    server.stop();
}

/// `test_e2b_persist_workspace_raises_on_nonzero_snapshot_exit`
#[tokio::test]
async fn e2b_persist_workspace_raises_on_nonzero_snapshot_exit() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 1)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the archive command failed");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(
        error.message.contains("Failed to archive workspace"),
        "{}",
        error.message
    );
    server.stop();
}

/// `test_e2b_persist_workspace_native_snapshot_returns_snapshot_ref`: a snapshot-persistence
/// session returns a reference to the platform snapshot instead of a tar.
#[tokio::test]
async fn e2b_persist_workspace_native_snapshot_returns_a_snapshot_reference() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.path.ends_with("/snapshots") {
            MockResponse::json(json!({ "snapshotID": "snap-123" }))
        } else {
            MockResponse::json(created_sandbox())
        }
    })
    .await;
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .create(None)
    .await
    .expect("create");

    let persisted = session.persist_workspace().await.expect("persist");

    assert_eq!(
        server.find("POST", "/sandboxes/sb-123/snapshots").json()["memory"],
        json!(false)
    );
    assert!(
        String::from_utf8_lossy(&persisted).contains("snap-123"),
        "{:?}",
        String::from_utf8_lossy(&persisted)
    );
    server.stop();
}

/// `test_e2b_hydrate_workspace_native_snapshot_recreates_from_snapshot_id`: restoring a native
/// snapshot creates a sandbox from the snapshot template and kills the one it replaces.
#[tokio::test]
async fn e2b_hydrate_workspace_native_snapshot_recreates_from_the_snapshot_id() {
    // The first create backs the session; the second is the replacement built from the snapshot.
    let creates = support::Counter::new();
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.path.ends_with("/snapshots") {
            return MockResponse::json(json!({ "snapshotID": "snap-123" }));
        }
        if request.path == "/v2/sandboxes" {
            return match creates.next("create") {
                0 => MockResponse::json(created_sandbox()),
                _ => MockResponse::json(json!({ "sandboxID": "sb-from-snapshot" })),
            };
        }
        MockResponse::json(json!({}))
    })
    .await;
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_workspace_persistence(WorkspacePersistence::Snapshot),
    )
    .create(None)
    .await
    .expect("create");
    let reference = session.persist_workspace().await.expect("persist");

    session
        .hydrate_workspace(&reference)
        .await
        .expect("hydrate");

    // The replacement is created from the snapshot id, used as the template.
    let creates = server.all("POST", "/v2/sandboxes");
    assert_eq!(
        creates.last().expect("a create").json()["templateID"],
        "snap-123"
    );
    // The sandbox the snapshot replaces is released.
    assert_eq!(server.count("DELETE", "/sandboxes/sb-123"), 1);
    server.stop();
}

/// `test_e2b_hydrate_workspace_raises_on_nonzero_extract_exit`
#[tokio::test]
async fn e2b_hydrate_workspace_raises_on_nonzero_extract_exit() {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    {
        let mut builder = tar::Builder::new(&mut encoder);
        builder.finish().expect("finish the tar");
    }
    let archive = encoder.finish().expect("finish the gzip");

    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 1)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .hydrate_workspace(&archive)
        .await
        .expect_err("the extract command failed");

    assert_eq!(error.code, ErrorCode::SnapshotRestoreError);
    // The archive is uploaded first, then extracted, then removed whatever the outcome.
    assert_eq!(server.count("POST", "/files"), 1);
    let command = start_command(&start_requests(&server)[0]);
    assert!(command.contains("tar -xzf"), "{command}");
    server.stop();
}

/// An archive that is neither a snapshot reference nor a valid tar is refused before anything is
/// uploaded into the sandbox.
#[tokio::test]
async fn e2b_hydrate_workspace_refuses_an_invalid_archive() {
    let server = support::start_resolving(support::Resolver::E2b, session_handler("", 0)).await;
    let session = client(&server.base_url).create(None).await.expect("create");

    let error = session
        .hydrate_workspace(b"not an archive")
        .await
        .expect_err("not an archive");

    assert!(!error.message.is_empty(), "{}", error.message);
    assert!(
        server.find_opt("POST", "/files").is_none(),
        "{:?}",
        server.routes()
    );
    server.stop();
}

// ---------------------------------------------------------------------------------------------
// Stop and shutdown
// ---------------------------------------------------------------------------------------------

/// A `stop` pauses the sandbox: a paused E2B sandbox keeps its filesystem and resumes by id.
#[tokio::test]
async fn e2b_stop_pauses_the_sandbox() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.stop().await.expect("stop");

    assert_eq!(server.count("POST", "/sandboxes/sb-123/pause"), 1);
    server.stop();
}

/// `test_e2b_shutdown_logs_pause_failure_and_falls_back_to_kill`: a pause that cannot be taken
/// must not leak the sandbox, so shutdown kills it instead.
#[tokio::test]
async fn e2b_shutdown_falls_back_to_kill_when_the_pause_fails() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.path.ends_with("/pause") {
            MockResponse::status(500, json!({ "message": "cannot pause" }))
        } else {
            MockResponse::json(created_sandbox())
        }
    })
    .await;
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_pause_on_exit(true),
    )
    .create(None)
    .await
    .expect("create");

    session.shutdown().await.expect("shutdown");

    assert_eq!(server.count("POST", "/sandboxes/sb-123/pause"), 1);
    assert_eq!(server.count("DELETE", "/sandboxes/sb-123"), 1);
    server.stop();
}

/// `test_e2b_shutdown_logs_direct_kill_failure`: without `pause_on_exit` the sandbox is killed
/// directly, and a kill that reports "already gone" is not an error.
#[tokio::test]
async fn e2b_shutdown_kills_directly_and_tolerates_a_missing_sandbox() {
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        if request.method == "DELETE" {
            MockResponse::empty(404)
        } else {
            MockResponse::json(created_sandbox())
        }
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.shutdown().await.expect("shutdown");

    assert_eq!(server.count("POST", "/sandboxes/sb-123/pause"), 0);
    assert_eq!(server.count("DELETE", "/sandboxes/sb-123"), 1);
    server.stop();
}

/// Shutting down twice releases the sandbox exactly once.
#[tokio::test]
async fn e2b_shutdown_releases_the_sandbox_exactly_once() {
    let server = support::start_resolving(support::Resolver::E2b, move |_| {
        MockResponse::json(created_sandbox())
    })
    .await;
    let session = client(&server.base_url).create(None).await.expect("create");

    session.shutdown().await.expect("shutdown");
    session.shutdown().await.expect("shutdown again");

    assert_eq!(server.count("DELETE", "/sandboxes/sb-123"), 1);
    server.stop();
}

/// A missing API key is refused before any request is made.
#[tokio::test]
async fn e2b_rejects_a_missing_api_key() {
    // SAFETY: the test process sets its own environment before any client reads it.
    unsafe {
        std::env::remove_var("E2B_API_KEY");
    }
    let error = E2BSandboxClient::new()
        .create(None)
        .await
        .expect_err("no API key");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}
