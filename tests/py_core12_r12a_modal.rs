//! Modal cases of `tests/extensions/sandbox/test_modal.py` that the earlier ports left as
//! Python-only: the defaults a payload without the newer fields gets, a create request that omits
//! an unset image builder version, and a named secret without an environment.
//!
//! The helper is `tests/fixtures/fake_modal_helper_scripted.py`, which logs every request.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy, ModalSandboxClient,
    ModalSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
use serde_json::{Value, json};

struct FakeModal {
    directory: tempfile::TempDir,
}

impl FakeModal {
    fn new() -> FakeModal {
        FakeModal {
            directory: tempfile::tempdir().expect("tempdir"),
        }
    }

    fn path(&self, name: &str) -> std::path::PathBuf {
        self.directory.path().join(name)
    }

    fn options(&self) -> ModalSandboxClientOptions {
        ModalSandboxClientOptions::new("agents-app").with_helper_command(vec![
            "/usr/bin/env".to_string(),
            format!("FAKE_MODAL_LOG={}", self.path("requests.jsonl").display()),
            format!("FAKE_MODAL_STATE={}", self.path("state.json").display()),
            format!("FAKE_MODAL_RULES={}", self.path("rules.json").display()),
            "python3".to_string(),
            format!(
                "{}/tests/fixtures/fake_modal_helper_scripted.py",
                env!("CARGO_MANIFEST_DIR")
            ),
        ])
    }

    fn of_op(&self, op: &str) -> Vec<Value> {
        std::fs::read_to_string(self.path("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|request| request["op"] == op)
            .collect()
    }
}

/// `test_modal_deserialize_session_state_defaults_missing_image_builder_version`: a payload
/// written before the image builder version existed gets the default version.
#[test]
fn a_payload_without_an_image_builder_version_gets_the_default() {
    let restored: ModalSandboxClientOptions =
        serde_json::from_value(json!({ "app_name": "sandbox-tests" })).expect("deserialize");

    assert_eq!(
        restored.image_builder_version,
        ModalSandboxClientOptions::new("sandbox-tests").image_builder_version
    );
    assert!(restored.image_builder_version.is_some());
}

/// `test_modal_deserialize_session_state_defaults_missing_idle_timeout`
#[test]
fn a_payload_without_an_idle_timeout_has_none() {
    let mut written = serde_json::to_value(
        ModalSandboxClientOptions::new("sandbox-tests").with_idle_timeout_secs(60),
    )
    .expect("serialize");
    assert_eq!(written["idle_timeout_secs"], 60);
    written
        .as_object_mut()
        .expect("object")
        .remove("idle_timeout_secs");

    let restored: ModalSandboxClientOptions = serde_json::from_value(written).expect("deserialize");

    assert_eq!(restored.idle_timeout_secs, None);
}

/// `test_modal_deserialize_session_state_defaults_missing_resources`
#[test]
fn a_payload_without_cpu_and_memory_has_none() {
    let mut written = serde_json::to_value(
        ModalSandboxClientOptions::new("sandbox-tests")
            .with_cpu(1.0)
            .with_memory_mb(2048),
    )
    .expect("serialize");
    assert_eq!(written["memory_mb"], 2048);
    for key in ["cpu", "cpu_limit", "memory_mb", "memory_limit_mb"] {
        written.as_object_mut().expect("object").remove(key);
    }

    let restored: ModalSandboxClientOptions = serde_json::from_value(written).expect("deserialize");

    assert_eq!(restored.cpu, None);
    assert_eq!(restored.memory_mb, None);
}

/// `test_modal_sandbox_create_uses_existing_config_when_image_builder_version_is_none`: with no
/// version configured the create request carries none, leaving the user's own configuration in
/// charge.
#[tokio::test]
async fn an_unset_image_builder_version_stays_off_the_create_request() {
    let fake = FakeModal::new();
    let mut options = fake.options();
    options.image_builder_version = None;

    ModalSandboxClient::new(options)
        .create(None)
        .await
        .expect("create");

    let request = &fake.of_op("create")[0];
    assert!(request.get("image_builder_version").is_none(), "{request}");
}

/// `test_modal_sandbox_create_passes_named_modal_secret_for_cloud_bucket_mount`: a named secret
/// without an environment reaches the helper with no inline credentials and no environment.
#[tokio::test]
async fn create_passes_a_named_secret_without_an_environment() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_secret_name("named-modal-secret")
                .with_key_prefix("nested/prefix/")
                .with_read_only(false),
        )),
    );

    ModalSandboxClient::new(fake.options())
        .create(Some(manifest))
        .await
        .expect("create");

    let volume = &fake.of_op("create")[0]["volumes"][0];
    assert_eq!(volume["mount_path"], "/workspace/remote");
    assert_eq!(volume["bucket_name"], "bucket");
    assert_eq!(volume["key_prefix"], "nested/prefix/");
    assert_eq!(volume["secret_name"], "named-modal-secret");
    assert!(
        volume
            .get("secret_environment_name")
            .is_none_or(Value::is_null)
    );
    assert!(volume.get("credentials").is_none());
    assert_eq!(volume["read_only"], false);
}

// ---------------------------------------------------------------------------------------------
// A failed remount after a directory snapshot
// ---------------------------------------------------------------------------------------------

use openai_agents_sdk::extensions::sandbox::modal::ModalWorkspacePersistence;
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::{ErrorCode, SandboxError, SandboxResult, SandboxSession};

/// A durable mount whose re-attach after a snapshot fails.
#[derive(Debug)]
struct FailingRestore {
    events: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl MountStrategy for FailingRestore {
    fn strategy_type(&self) -> &str {
        "failing_restore"
    }

    async fn activate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }

    async fn deactivate(&self, _: &Mount, _: &dyn SandboxSession, _: &str) -> SandboxResult<()> {
        Ok(())
    }

    async fn teardown_for_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        path: &str,
    ) -> SandboxResult<()> {
        self.events.lock().unwrap().push(format!("unmount:{path}"));
        Ok(())
    }

    async fn restore_after_snapshot(
        &self,
        _: &Mount,
        _: &dyn SandboxSession,
        path: &str,
    ) -> SandboxResult<()> {
        self.events.lock().unwrap().push(format!("mount:{path}"));
        Err(SandboxError::new(ErrorCode::MountFailed, "remount failed"))
    }
}

/// `test_modal_snapshot_directory_failed_remount_does_not_create_replacement_sandbox`: when the
/// durable mount cannot be attached again after the directory image, the persist fails, the sandbox is terminated, and no replacement is created.
#[tokio::test]
async fn a_failed_remount_after_a_directory_snapshot_terminates_without_a_replacement() {
    let fake = FakeModal::new();
    let events: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "remote",
        Entry::mount(
            Mount::new(std::sync::Arc::new(FailingRestore {
                events: events.clone(),
            }))
            .with_mount_path("actual")
            .with_ephemeral(false),
        ),
    );
    let session = ModalSandboxClient::new(
        fake.options()
            .with_workspace_persistence(ModalWorkspacePersistence::SnapshotDirectory),
    )
    .create(Some(manifest))
    .await
    .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the remount fails");

    assert_eq!(
        error.code,
        ErrorCode::WorkspaceArchiveReadError,
        "{error:?}"
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec!["unmount:/workspace/actual", "mount:/workspace/actual"]
    );
    assert_eq!(fake.of_op("create").len(), 1);
    assert_eq!(fake.of_op("terminate").len(), 1);
}
