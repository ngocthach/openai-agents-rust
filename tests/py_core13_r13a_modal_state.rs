//! Modal provider options in the persisted session state: `tests/extensions/sandbox/test_modal.py`
//! `test_modal_sandbox_create_uses_custom_image_builder_version`, `..._passes_resources`, the
//! three `test_modal_deserialize_session_state_defaults_missing_*` cases and the resume that
//! recreates a sandbox.
//!
//! The image builder version, idle timeout and resources are written into the state a session
//! hands out and are what a replacement sandbox created on resume is built from. The helper is
//! `tests/fixtures/fake_modal_helper_scripted.py`, which logs every request.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::modal::{
    DEFAULT_IMAGE_BUILDER_VERSION, ModalPersistedOptions, ModalSandboxClient,
    ModalSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSessionState};
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

fn manifest() -> Manifest {
    Manifest::new().with_root("/workspace")
}

/// `test_modal_sandbox_create_uses_custom_image_builder_version` and
/// `..._passes_resources`: the state records the builder version and the resources the create
/// request carried, a `(request, limit)` pair included.
#[tokio::test]
async fn the_session_state_records_the_builder_version_and_resources() {
    let fake = FakeModal::new();
    let mut options = fake
        .options()
        .with_idle_timeout_secs(60)
        .with_cpu(1.0)
        .with_memory_mb(2048);
    options.cpu_limit = Some(4.0);
    options.memory_limit_mb = Some(8192);
    options.image_builder_version = Some("PREVIEW".to_string());
    let session = ModalSandboxClient::new(options)
        .create(Some(manifest()))
        .await
        .expect("create");

    let state = session.session_state();

    let persisted = ModalPersistedOptions::from_state(&state).expect("options recorded");
    assert_eq!(persisted.image_builder_version.as_deref(), Some("PREVIEW"));
    assert_eq!(persisted.idle_timeout_secs, Some(60));
    assert_eq!(persisted.cpu, Some(1.0));
    assert_eq!(persisted.cpu_limit, Some(4.0));
    assert_eq!(persisted.memory_mb, Some(2048));
    assert_eq!(persisted.memory_limit_mb, Some(8192));
    let request = &fake.of_op("create")[0];
    assert_eq!(request["image_builder_version"], "PREVIEW");
    assert_eq!(request["cpu"], json!([1.0, 4.0]));
    assert_eq!(request["memory"], json!([2048, 8192]));
}

/// `test_modal_sandbox_create_uses_existing_config_when_image_builder_version_is_none`: an
/// unset version is recorded as unset, not replaced by the default.
#[tokio::test]
async fn an_unset_builder_version_is_recorded_as_unset() {
    let fake = FakeModal::new();
    let mut options = fake.options();
    options.image_builder_version = None;
    let session = ModalSandboxClient::new(options)
        .create(Some(manifest()))
        .await
        .expect("create");

    let state = session.session_state();
    let restored: SandboxSessionState =
        serde_json::from_value(serde_json::to_value(&state).expect("serializes"))
            .expect("round trips");

    assert_eq!(
        ModalPersistedOptions::from_state(&restored)
            .expect("options")
            .image_builder_version,
        None
    );
}

/// `test_modal_deserialize_session_state_defaults_missing_image_builder_version`,
/// `..._idle_timeout` and `..._resources`: a record written without those fields reads with the
/// default builder version, no idle timeout and no resources.
#[test]
fn missing_fields_read_as_the_defaults() {
    let written = ModalPersistedOptions {
        image_builder_version: Some("PREVIEW".to_string()),
        idle_timeout_secs: Some(60),
        cpu: Some(1.0),
        cpu_limit: Some(4.0),
        memory_mb: Some(2048),
        memory_limit_mb: Some(8192),
    };
    let mut payload = serde_json::to_value(&written).expect("serializes");
    for key in [
        "image_builder_version",
        "idle_timeout_secs",
        "cpu",
        "cpu_limit",
        "memory_mb",
        "memory_limit_mb",
    ] {
        assert!(!payload[key].is_null(), "{key} was written");
        payload.as_object_mut().expect("object").remove(key);
    }

    let restored: ModalPersistedOptions = serde_json::from_value(payload).expect("deserializes");

    assert_eq!(
        restored.image_builder_version.as_deref(),
        Some(DEFAULT_IMAGE_BUILDER_VERSION)
    );
    assert_eq!(restored.idle_timeout_secs, None);
    assert_eq!(restored.cpu, None);
    assert_eq!(restored.cpu_limit, None);
    assert_eq!(restored.memory_mb, None);
    assert_eq!(restored.memory_limit_mb, None);
}

/// A replacement sandbox created on resume is built from the state's builder version and
/// resources, not from the options of the client that resumes it.
#[tokio::test]
async fn a_replacement_sandbox_is_built_from_the_recorded_options() {
    let creator = FakeModal::new();
    let mut original_options = creator.options().with_idle_timeout_secs(60).with_cpu(2.0);
    original_options.memory_mb = Some(4096);
    original_options.image_builder_version = Some("PREVIEW".to_string());
    let original = ModalSandboxClient::new(original_options)
        .create(Some(manifest()))
        .await
        .expect("create");
    let mut state = original.session_state();
    // No sandbox to reattach to: the resume has to create one.
    state.provider_id = None;
    let text = serde_json::to_string(&state).expect("serializes");
    let state: SandboxSessionState = serde_json::from_str(&text).expect("deserializes");

    let resuming = FakeModal::new();
    let resumed = ModalSandboxClient::new(resuming.options())
        .resume(&state, manifest())
        .await
        .expect("resume");

    let created = &resuming.of_op("create");
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["image_builder_version"], "PREVIEW");
    assert_eq!(created[0]["idle_timeout"], 60);
    assert_eq!(created[0]["cpu"], 2.0);
    assert_eq!(created[0]["memory"], 4096);
    assert_eq!(
        ModalPersistedOptions::from_state(&resumed.session_state()),
        ModalPersistedOptions::from_state(&state)
    );
}

/// A state written before the options were recorded reads, and the resuming client's options
/// stand in for them.
#[tokio::test]
async fn an_older_payload_without_options_falls_back_to_the_client() {
    let creator = FakeModal::new();
    let original = ModalSandboxClient::new(creator.options())
        .create(Some(manifest()))
        .await
        .expect("create");
    let mut payload = serde_json::to_value(original.session_state()).expect("serializes");
    let object = payload.as_object_mut().expect("object");
    object.remove("provider_metadata");
    object.insert("provider_id".to_string(), Value::Null);
    let state: SandboxSessionState = serde_json::from_value(payload).expect("old payload reads");
    assert_eq!(ModalPersistedOptions::from_state(&state), None);

    let resuming = FakeModal::new();
    ModalSandboxClient::new(resuming.options().with_memory_mb(1024))
        .resume(&state, manifest())
        .await
        .expect("resume");

    assert_eq!(resuming.of_op("create")[0]["memory"], 1024);
}
