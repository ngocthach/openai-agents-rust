//! Rust-native ports of the behavioural rows of `tests/sandbox/test_compatibility_guards.py` and
//! the state rows of `test_docker_network_mode.py`.
//!
//! Python pins its pydantic wire shapes and `__all__` export surfaces. Rust pins the equivalents
//! that exist here: the discriminator strings stored in persisted state, the session-state JSON
//! round trip, and the rejection of malformed persisted state before any backend is touched.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::Manifest;
use openai_agents_sdk::sandbox::mounts::{DockerVolumeMountStrategy, MountStrategy};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DOCKER_NETWORK_DISABLED_KEY, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use serde_json::{Value, json};

fn state(backend: &str) -> SandboxSessionState {
    SandboxSessionState::new(backend, "session-1", "/workspace", &Manifest::new())
}

/// Python: `test_core_discriminator_type_strings_are_stable` (snapshots) and
/// `test_optional_sandbox_discriminator_type_strings_are_stable` (backends).
#[test]
fn snapshot_and_backend_discriminators_are_stable() {
    let local = serde_json::to_value(SnapshotSpec::Local {
        base_path: "/tmp/snapshots".into(),
    })
    .unwrap();
    assert_eq!(local["type"], "local");
    assert_eq!(
        serde_json::to_value(SnapshotSpec::Noop).unwrap()["type"],
        "noop"
    );
    assert_eq!(
        serde_json::to_value(SnapshotSpec::Remote {
            client_key: "k".into()
        })
        .unwrap()["type"],
        "remote"
    );
    for backend in ["unix_local", "docker"] {
        let payload: Value =
            serde_json::from_str(&state(backend).to_json_string().unwrap()).unwrap();
        assert_eq!(payload["backend"], backend);
    }
}

/// Python: `test_mount_strategy_type_strings_round_trip_through_registry` and
/// `test_optional_mount_strategy_type_strings_round_trip_through_registry` (the core strategies;
/// the provider strategies are pinned in the extension test below).
#[test]
fn core_mount_strategy_type_strings_are_stable() {
    assert_eq!(
        DockerVolumeMountStrategy::new("rclone").strategy_type(),
        "docker_volume"
    );
    let in_container = openai_agents_sdk::sandbox::mount_entries::ConfiguredMountStrategy::rclone(
        openai_agents_sdk::sandbox::mount_entries::S3Mount::new("bucket"),
    );
    assert_eq!(in_container.strategy_type(), "in_container");
}

/// Python: `test_optional_mount_strategy_type_strings_round_trip_through_registry` (7 sets).
#[test]
fn provider_mount_strategy_type_strings_are_stable() {
    use openai_agents_sdk::extensions::sandbox::e2b::E2BCloudBucketMountStrategy;
    let strategy = E2BCloudBucketMountStrategy::new("bucket", "my-bucket", "[bucket]\ntype = s3\n");
    assert_eq!(strategy.strategy_type(), "e2b_cloud_bucket");
}

/// Python: `test_optional_sandbox_session_state_json_round_trip_preserves_type` (8 sets, one per
/// backend that Rust has) and `test_docker_session_state_network_mode_round_trip`.
#[test]
fn session_state_round_trips_with_its_backend_and_provider_metadata() {
    for backend in ["unix_local", "docker"] {
        let original = state(backend)
            .with_provider_metadata(DOCKER_NETWORK_DISABLED_KEY, json!(true))
            .with_exposed_ports([8000]);

        let json = original.to_json_string().unwrap();
        let restored = SandboxSessionState::from_json_str(&json).unwrap();

        assert_eq!(restored.backend, backend);
        assert_eq!(restored.exposed_ports, vec![8000]);
        assert_eq!(
            restored.provider_metadata(DOCKER_NETWORK_DISABLED_KEY),
            Some(&json!(true))
        );
        assert_eq!(
            restored.to_json_string().unwrap(),
            json,
            "stable re-serialization"
        );
    }
}

/// Python: `test_docker_session_state_without_network_mode_preserves_old_payloads`.
#[test]
fn a_payload_without_the_network_flag_restores_without_one() {
    let json = state("docker").to_json_string().unwrap();
    let mut payload: Value = serde_json::from_str(&json).unwrap();
    if let Some(metadata) = payload
        .get_mut("provider_metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(DOCKER_NETWORK_DISABLED_KEY);
    }

    let restored = SandboxSessionState::from_json_str(&payload.to_string()).unwrap();

    assert_eq!(
        restored.provider_metadata(DOCKER_NETWORK_DISABLED_KEY),
        None
    );
}

/// Python: `test_docker_options_omitted_network_mode_preserves_default_behavior`.
#[test]
fn docker_options_default_to_a_networked_container() {
    let options = DockerSandboxClientOptions::new("python:3.14-slim");
    assert!(!options.network_disabled);
    options.validate().unwrap();
}

/// Python: `test_docker_session_state_rejects_invalid_network_mode_before_provider_access` and
/// `test_docker_state_rejects_no_network_exposed_ports_before_provider_access`, for the part
/// that exists in Rust: malformed persisted state is refused while parsing, with no backend.
#[test]
fn malformed_persisted_state_is_refused_before_any_backend() {
    let json = state("docker").to_json_string().unwrap();
    let mut payload: Value = serde_json::from_str(&json).unwrap();
    payload["exposed_ports"] = json!([0]);
    assert!(SandboxSessionState::from_json_str(&payload.to_string()).is_err());

    payload["exposed_ports"] = json!([]);
    payload["$schemaVersion"] = json!(99);
    assert!(SandboxSessionState::from_json_str(&payload.to_string()).is_err());
    assert!(SandboxSessionState::from_json_str("not json").is_err());
}

/// Python: `test_run_state_round_trip_preserves_docker_network_mode` and
/// `test_run_state_sandbox_payload_json_shape_is_stable`: what a paused run persists is the
/// session state, which keeps the network flag and the exposed ports across serialization.
#[test]
fn a_paused_runs_session_state_keeps_network_flag_and_ports() {
    let original = state("docker")
        .with_provider_metadata(DOCKER_NETWORK_DISABLED_KEY, json!(true))
        .with_exposed_ports([8000]);
    let restored = SandboxSessionState::from_json_str(&original.to_json_string().unwrap()).unwrap();

    assert_eq!(
        restored.provider_metadata(DOCKER_NETWORK_DISABLED_KEY),
        Some(&json!(true))
    );
    assert_eq!(restored.exposed_ports, vec![8000]);
}
