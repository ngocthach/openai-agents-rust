//! Manifest payload cases the earlier audits filed as Python-only because Rust had no
//! `model_dump`/`model_validate`. `Manifest::to_value`, `from_value` and `from_untrusted_value`
//! are that surface:
//!
//! * `tests/sandbox/test_mounts.py`: the mount strategy and pattern subtypes survive a payload
//!   round trip.
//! * `tests/sandbox/test_workspace_paths.py`: extra path grants are written in order.
//! * `tests/sandbox/test_runtime_agent_preparation.py`: a dictionary default manifest cannot name
//!   path grants, and a serialized manifest without grants is accepted.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::Manifest;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3FilesMount, S3Mount};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use serde_json::json;

/// Python: `test_manifest_model_dump_preserves_mount_strategy_subtype_fields`.
#[test]
fn a_payload_keeps_the_mount_strategy_subtype_fields() {
    let manifest = Manifest::new()
        .with_entry(
            "in-container",
            Entry::mount(ConfiguredMountStrategy::mountpoint(S3Mount::new("bucket")).mount()),
        )
        .with_entry(
            "docker-volume",
            Entry::mount(
                ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone")
                    .with_driver_option("vfs-cache-mode", "off")
                    .mount(),
            ),
        );

    let payload = manifest.to_value().unwrap();

    assert_eq!(
        payload["entries"]["in-container"]["mount_strategy"],
        json!({
            "type": "in_container",
            "pattern": {
                "type": "mountpoint",
                "options": {"prefix": null, "region": null, "endpoint_url": null},
            },
        })
    );
    assert_eq!(
        payload["entries"]["docker-volume"]["mount_strategy"],
        json!({
            "type": "docker_volume",
            "driver": "rclone",
            "driver_options": {"vfs-cache-mode": "off"},
        })
    );

    let restored = Manifest::from_value(&payload).unwrap();
    let again = restored.to_value().unwrap();
    assert_eq!(
        again["entries"]["in-container"]["mount_strategy"]["pattern"]["type"],
        "mountpoint"
    );
    assert_eq!(
        again["entries"]["docker-volume"]["mount_strategy"]["driver"],
        "rclone"
    );
    assert_eq!(
        again["entries"]["docker-volume"]["mount_strategy"]["driver_options"],
        json!({"vfs-cache-mode": "off"})
    );
    assert_eq!(again, payload);
}

/// Python: `test_manifest_model_dump_round_trips_s3_files_mount`.
#[test]
fn a_payload_round_trips_an_s3_files_mount() {
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(
            ConfiguredMountStrategy::s3files(
                S3FilesMount::new("fs-1234567890abcdef0")
                    .with_subpath("/datasets")
                    .with_mount_target_ip("10.99.1.209")
                    .with_region("us-east-1"),
            )
            .mount()
            .with_read_only(false),
        ),
    );

    let payload = manifest.to_value().unwrap();

    assert_eq!(payload["entries"]["remote"]["type"], "s3_files_mount");
    assert_eq!(
        payload["entries"]["remote"]["mount_strategy"],
        json!({
            "type": "in_container",
            "pattern": {
                "type": "s3files",
                "options": {
                    "mount_target_ip": null,
                    "access_point": null,
                    "region": null,
                    "extra_options": {},
                },
            },
        })
    );

    let restored = Manifest::from_value(&payload).unwrap().to_value().unwrap();
    let mount = &restored["entries"]["remote"];
    assert_eq!(mount["type"], "s3_files_mount");
    assert_eq!(mount["file_system_id"], "fs-1234567890abcdef0");
    assert_eq!(mount["subpath"], "/datasets");
    assert_eq!(mount["mount_target_ip"], "10.99.1.209");
    assert_eq!(mount["region"], "us-east-1");
    assert_eq!(mount["read_only"], false);
    assert_eq!(mount["mount_strategy"]["pattern"]["type"], "s3files");
    assert_eq!(restored, payload);
}

/// Python: `test_manifest_serializes_extra_path_grants`.
#[test]
fn a_payload_lists_the_extra_path_grants_in_order() {
    let manifest = Manifest::new()
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp")
                .unwrap()
                .with_description("temporary files"),
        )
        .with_extra_path_grant(
            SandboxPathGrant::new("/opt/toolchain")
                .unwrap()
                .read_only(true)
                .with_description("compiler runtime"),
        );

    let payload = manifest.to_value().unwrap();

    assert_eq!(
        payload["extra_path_grants"],
        json!([
            {"path": "/tmp", "read_only": false, "description": "temporary files"},
            {"path": "/opt/toolchain", "read_only": true, "description": "compiler runtime"},
        ])
    );
}

/// Python: `test_sandbox_agent_rejects_untrusted_manifest_path_grants`. A dictionary default
/// manifest is untrusted configuration and may not grant paths outside the workspace.
#[test]
fn a_dictionary_default_manifest_cannot_name_path_grants() {
    let error = Manifest::from_untrusted_value(
        &json!({"extra_path_grants": [{"path": "/tmp"}]}),
        "sandbox.default_manifest",
    )
    .expect_err("path grants need a trusted manifest");

    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(
        error.to_string().contains(
            "sandbox.default_manifest.extra_path_grants must be configured on a trusted Manifest"
        ),
        "{error}"
    );
}

/// Python: `test_sandbox_agent_accepts_serialized_manifest_without_path_grants` (the `dump` and
/// the JSON-mode `dump`): the payload a manifest writes is accepted as agent configuration, and
/// it carries no grants.
#[test]
fn a_serialized_manifest_without_grants_is_accepted_as_configuration() {
    let payload = Manifest::new().with_root("/workspace").to_value().unwrap();
    let through_json: serde_json::Value =
        serde_json::from_str(&payload.to_string()).expect("the payload is JSON");

    for candidate in [payload, through_json] {
        let manifest = Manifest::from_untrusted_value(&candidate, "sandbox.default_manifest")
            .expect("a manifest without grants is accepted");

        assert_eq!(manifest.root(), "/workspace");
        assert!(manifest.extra_path_grants().is_empty());
    }
}
