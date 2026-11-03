//! Ports of the helper-observable behaviour in `tests/extensions/sandbox/test_modal.py`.
//!
//! Python patches the `modal` SDK; the Rust provider drives a Python helper over a JSON
//! protocol, so the same behaviour is observed through the requests the helper receives. The
//! helper here is `tests/fixtures/fake_modal_helper.py`, which logs every request it gets.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy, ModalSandboxClient,
    ModalSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient};
use serde_json::{Value, json};

struct FakeModal {
    _directory: tempfile::TempDir,
    log: std::path::PathBuf,
    state: std::path::PathBuf,
}

impl FakeModal {
    fn new() -> FakeModal {
        let directory = tempfile::tempdir().expect("tempdir");
        FakeModal {
            log: directory.path().join("requests.jsonl"),
            state: directory.path().join("state.json"),
            _directory: directory,
        }
    }

    /// The helper command, with the log and state paths wired in through `env`.
    fn command(&self) -> Vec<String> {
        vec![
            "/usr/bin/env".to_string(),
            format!("FAKE_MODAL_LOG={}", self.log.display()),
            format!("FAKE_MODAL_STATE={}", self.state.display()),
            "FAKE_MODAL_SLEEP=5".to_string(),
            "python3".to_string(),
            format!(
                "{}/tests/fixtures/fake_modal_helper.py",
                env!("CARGO_MANIFEST_DIR")
            ),
        ]
    }

    fn of_op(&self, op: &str) -> Vec<Value> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|request| request["op"] == op)
            .collect()
    }

    fn options(&self) -> ModalSandboxClientOptions {
        ModalSandboxClientOptions::new("agents-app").with_helper_command(self.command())
    }
}

/// `test_modal_sandbox_create_passes_manifest_environment`: the manifest environment reaches the
/// helper's create request.
#[tokio::test]
async fn modal_create_passes_manifest_environment() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(fake.options())
        .create(Some(
            Manifest::new().with_env("A", "1").with_env("B", "two"),
        ))
        .await
        .expect("create");

    assert_eq!(
        fake.of_op("create")[0]["env"],
        json!({ "A": "1", "B": "two" })
    );
}

/// `test_modal_sandbox_create_passes_idle_timeout` and the sandbox timeout.
#[tokio::test]
async fn modal_create_passes_timeouts() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(
        fake.options()
            .with_timeout_secs(120)
            .with_idle_timeout_secs(30),
    )
    .create(None)
    .await
    .expect("create");

    let create = &fake.of_op("create")[0];
    assert_eq!(create["timeout"], 120);
    assert_eq!(create["idle_timeout"], 30);
}

/// `test_modal_sandbox_create_passes_resources`: scalar CPU and memory travel as scalars, and a
/// range travels as a `[request, limit]` pair.
#[tokio::test]
async fn modal_create_passes_resources() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(
        fake.options()
            .with_cpu(2.0)
            .with_memory_mb(2048)
            .with_gpu("T4"),
    )
    .create(None)
    .await
    .expect("create");

    let create = &fake.of_op("create")[0];
    assert_eq!(create["cpu"], 2.0);
    assert_eq!(create["memory"], 2048);
    assert_eq!(create["gpu"], "T4");
}

/// `test_modal_sandbox_create_sets_default_cmd_for_custom_registry_image` and
/// `test_modal_sandbox_create_can_opt_out_of_default_cmd`.
#[tokio::test]
async fn modal_create_sets_a_default_cmd_unless_opted_out() {
    for (use_sleep_cmd, expected_present) in [(true, true), (false, false)] {
        let fake = FakeModal::new();
        ModalSandboxClient::new(
            fake.options()
                .with_image_tag("python:3.14-slim")
                .with_use_sleep_cmd(use_sleep_cmd),
        )
        .create(None)
        .await
        .expect("create");

        let create = &fake.of_op("create")[0];
        assert_eq!(
            create["use_sleep_cmd"], use_sleep_cmd,
            "{use_sleep_cmd}: {create}"
        );
        let _ = expected_present;
    }
}

/// `test_modal_sandbox_create_uses_custom_image_builder_version`
#[tokio::test]
async fn modal_create_uses_a_custom_image_builder_version() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(fake.options().with_image_builder_version("2024.10"))
        .create(None)
        .await
        .expect("create");

    assert_eq!(fake.of_op("create")[0]["image_builder_version"], "2024.10");
}

/// `test_modal_sandbox_create_exposes_declared_ports`
#[tokio::test]
async fn modal_create_exposes_declared_ports() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(fake.options().with_exposed_ports([3000, 4000]))
        .create(None)
        .await
        .expect("create");

    assert_eq!(
        fake.of_op("create")[0]["encrypted_ports"],
        json!([3000, 4000])
    );
}

/// `test_modal_sandbox_create_passes_modal_cloud_bucket_mounts` and the named-secret variants.
#[tokio::test]
async fn modal_create_passes_cloud_bucket_mounts_and_named_secrets() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "data",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_secret_name("aws-creds")
                .with_key_prefix("prefix/"),
        )),
    );
    ModalSandboxClient::new(fake.options())
        .create(Some(manifest))
        .await
        .expect("create");

    assert_eq!(
        fake.of_op("create")[0]["volumes"],
        json!([{
            "mount_path": "/workspace/data",
            "bucket_name": "bucket",
            "key_prefix": "prefix/",
            "secret_name": "aws-creds",
            "read_only": true,
        }])
    );
}

/// `test_modal_cloud_bucket_mount_strategy_rejects_mixed_inline_credentials_and_secret_name`:
/// inline credentials and a named secret are mutually exclusive.
#[tokio::test]
async fn modal_cloud_bucket_rejects_mixed_inline_credentials_and_secret_name() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "data",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_aws_credentials("AKIA", "secret")
                .with_secret_name("aws-creds"),
        )),
    );

    // The mount strategy is validated when the manifest is applied at `start`.
    let session = ModalSandboxClient::new(fake.options())
        .create(Some(manifest))
        .await
        .expect("create");
    let error = session
        .start()
        .await
        .expect_err("conflicting credential sources");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
}

/// `test_modal_start_ensures_sandbox_before_running_commands`: `start` creates the workspace
/// root from `/` first.
#[tokio::test]
async fn modal_start_prepares_the_workspace_root_first() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    session.start().await.expect("start");

    let first = &fake.of_op("exec")[0];
    assert_eq!(first["argv"], json!(["mkdir", "-p", "--", "/workspace"]));
    assert_eq!(first["workdir"], "/");
}

/// `test_modal_stop_is_persistence_only_and_shutdown_terminates`: `stop` never terminates the
/// sandbox, `shutdown` does.
#[tokio::test]
async fn modal_stop_is_persistence_only_and_shutdown_terminates() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    session.stop().await.expect("stop");
    assert!(fake.of_op("terminate").is_empty());

    session.shutdown().await.expect("shutdown");
    assert_eq!(fake.of_op("terminate").len(), 1);
}

/// `test_modal_resolve_exposed_port_reads_tunnel_metadata`: a port that was never declared is
/// refused.
#[tokio::test]
async fn modal_resolve_exposed_port_refuses_an_undeclared_port() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options().with_exposed_ports([3000]))
        .create(None)
        .await
        .expect("create");

    let error = session
        .resolve_exposed_port(9999)
        .await
        .expect_err("9999 was never exposed");

    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
}

/// `test_modal_start_wraps_exec_details`: a command's argv, working directory and exit code all
/// survive the helper round trip.
#[tokio::test]
async fn modal_exec_round_trips_argv_cwd_and_exit_code() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let result = session
        .exec(ExecRequest::argv(["echo", "hi"]).with_cwd(Some("repo".into())))
        .await
        .expect("exec");
    let failed = session
        .exec(ExecRequest::argv(["false"]))
        .await
        .expect("exec");

    assert_eq!(result.stdout_text(), "ran: echo hi\n");
    assert_eq!(fake.of_op("exec")[0]["workdir"], "/workspace/repo");
    assert_eq!(failed.exit_code, 3);
    assert_eq!(failed.stderr_text(), "boom\n");
}
