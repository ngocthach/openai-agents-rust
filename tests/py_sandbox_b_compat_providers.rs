//! Ports of the provider rows of `tests/sandbox/test_compatibility_guards.py`.
//!
//! Python persists every provider's session state and mount strategy through one type-tagged
//! registry, and its client options as well. Rust keeps one `SandboxSessionState` for every
//! backend (discriminated by `backend`, with provider data in `provider_metadata`), and its
//! client options are construction-only values that are never persisted. The state rows are
//! therefore checked per backend against that model, and the options row is out of scope by
//! design (see the test's docs).
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy, BlaxelDriveMountConfig,
    BlaxelDriveMountStrategy,
};
use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy,
};
use openai_agents_sdk::extensions::sandbox::daytona::DaytonaCloudBucketMountStrategy;
use openai_agents_sdk::extensions::sandbox::e2b::E2BCloudBucketMountStrategy;
use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy,
};
use openai_agents_sdk::extensions::sandbox::vercel::VercelCloudBucketMountStrategy;
use openai_agents_sdk::sandbox::SnapshotSpec;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mounts::MountStrategy;
use openai_agents_sdk::sandbox::state::SandboxSessionState;

/// Python: `test_optional_sandbox_session_state_json_round_trip_preserves_type`, one row per
/// backend. The state of each backend is written as JSON, read back, and must keep its backend
/// identity and its provider data exactly.
#[test]
fn every_backend_session_state_round_trips_its_type() {
    for backend in [
        "unix_local",
        "docker",
        "e2b",
        "modal",
        "cloudflare",
        "daytona",
        "blaxel",
        "vercel",
    ] {
        let manifest = Manifest::new().with_root("/workspace");
        let state = SandboxSessionState::new(backend, "session-123", "/workspace", &manifest)
            .with_provider_id(Some("sandbox-123".to_string()))
            .with_provider_metadata("provider_marker", serde_json::json!({"value": backend}))
            .with_snapshot(SnapshotSpec::Noop);
        let payload = state.to_json_string().unwrap();

        let restored = SandboxSessionState::from_json_str(&payload).unwrap();

        assert_eq!(restored.backend, backend);
        assert_eq!(restored, state, "{backend}");
        assert_eq!(restored.to_json_string().unwrap(), payload, "{backend}");
    }
}

/// Python: `test_optional_mount_strategy_type_strings_round_trip_through_registry`. Each
/// provider's bucket or drive strategy reports the type string Python persists.
///
/// Rust has no polymorphic registry to parse the payload back into a strategy: the RunState
/// sanitizer that keeps these fields is private, so the round-trip half is not exercised here.
#[test]
fn provider_mount_strategies_report_their_type_strings() {
    let cases: Vec<(Box<dyn MountStrategy>, &str)> = vec![
        (
            Box::new(E2BCloudBucketMountStrategy::new(
                "bucket",
                "/",
                "[bucket]\ntype = s3\n",
            )),
            "e2b_cloud_bucket",
        ),
        (
            Box::new(ModalCloudBucketMountStrategy::new(
                ModalCloudBucketMountConfig::s3("bucket"),
            )),
            "modal_cloud_bucket",
        ),
        (
            Box::new(DaytonaCloudBucketMountStrategy::new(
                "bucket",
                "/",
                "[bucket]\ntype = s3\n",
            )),
            "daytona_cloud_bucket",
        ),
        (
            Box::new(CloudflareBucketMountStrategy::new(
                CloudflareBucketMountConfig::r2("assets", "acct"),
            )),
            "cloudflare_bucket_mount",
        ),
        (
            Box::new(BlaxelCloudBucketMountStrategy::new(
                BlaxelCloudBucketMountConfig::s3("bucket"),
            )),
            "blaxel_cloud_bucket",
        ),
        (
            Box::new(BlaxelDriveMountStrategy::new(BlaxelDriveMountConfig::new(
                "shared",
            ))),
            "blaxel_drive",
        ),
        (
            Box::new(VercelCloudBucketMountStrategy::new("my-bucket")),
            "vercel_cloud_bucket",
        ),
    ];
    for (strategy, expected) in cases {
        assert_eq!(strategy.strategy_type(), expected);
    }
}

// ----------------------------------------------------------------------------------------------
// Typed registries
// ----------------------------------------------------------------------------------------------

use openai_agents_sdk::extensions::sandbox::cloudflare::CloudflareSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::daytona::DaytonaSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::e2b::E2BSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::modal::ModalSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::vercel::VercelSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::{ProviderMountStrategy, SandboxClientOptionsRecord};
use openai_agents_sdk::sandbox::entries::EntryKind;
use openai_agents_sdk::sandbox::persisted_registry::{
    parse_core_mount_strategy, parse_entry, parse_env_value, parse_mount_pattern,
};
use openai_agents_sdk::sandbox::sandboxes::docker::DockerSandboxClientOptions;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClientOptions;
use serde_json::{Value, json};

/// Python: `test_optional_sandbox_client_options_json_round_trip_preserves_type`, one row per
/// backend: the payload names its backend in `type`, parses back to the same options, and
/// re-serializes to the identical payload.
#[test]
fn every_backend_client_options_round_trip_their_type() {
    let cases: Vec<(SandboxClientOptionsRecord, &str)> = vec![
        (
            SandboxClientOptionsRecord::UnixLocal(UnixLocalSandboxClientOptions::default()),
            "unix_local",
        ),
        (
            SandboxClientOptionsRecord::Docker(DockerSandboxClientOptions::new("python:3.12")),
            "docker",
        ),
        (
            SandboxClientOptionsRecord::E2b(E2BSandboxClientOptions::new().with_template("base")),
            "e2b",
        ),
        (
            SandboxClientOptionsRecord::Modal(ModalSandboxClientOptions::new("agents-sdk")),
            "modal",
        ),
        (
            SandboxClientOptionsRecord::Cloudflare(CloudflareSandboxClientOptions::new(
                "https://worker.example",
            )),
            "cloudflare",
        ),
        (
            SandboxClientOptionsRecord::Daytona(DaytonaSandboxClientOptions::new()),
            "daytona",
        ),
        (
            SandboxClientOptionsRecord::Vercel(VercelSandboxClientOptions::new()),
            "vercel",
        ),
    ];
    for (record, expected) in cases {
        let payload: Value = serde_json::to_value(&record).unwrap();
        assert_eq!(payload["type"], expected);
        assert_eq!(record.type_name(), expected);

        let restored: SandboxClientOptionsRecord = serde_json::from_value(payload.clone()).unwrap();

        assert_eq!(restored, record, "{expected}");
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            payload,
            "{expected}"
        );
    }
}

/// An unknown backend name is refused.
#[test]
fn client_options_reject_an_unknown_backend() {
    let payload = json!({"type": "no_such_backend"});
    assert!(serde_json::from_value::<SandboxClientOptionsRecord>(payload).is_err());
}

/// Python: `test_optional_mount_strategy_type_strings_round_trip_through_registry`: each provider
/// strategy parses through the registry from its own payload, keeps its type string, and
/// re-serializes identically. The rclone configuration and bucket secrets never appear in the
/// payload.
#[test]
fn provider_mount_strategies_round_trip_through_the_registry() {
    let cases: Vec<(ProviderMountStrategy, &str)> = vec![
        (
            ProviderMountStrategy::E2bCloudBucket(E2BCloudBucketMountStrategy::new(
                "bucket",
                "/",
                "[bucket]\ntype = s3\n",
            )),
            "e2b_cloud_bucket",
        ),
        (
            ProviderMountStrategy::ModalCloudBucket(ModalCloudBucketMountStrategy::new(
                ModalCloudBucketMountConfig::s3("bucket"),
            )),
            "modal_cloud_bucket",
        ),
        (
            ProviderMountStrategy::DaytonaCloudBucket(DaytonaCloudBucketMountStrategy::new(
                "bucket",
                "/",
                "[bucket]\ntype = s3\nsecret_access_key = hunter2\n",
            )),
            "daytona_cloud_bucket",
        ),
        (
            ProviderMountStrategy::CloudflareBucketMount(CloudflareBucketMountStrategy::new(
                CloudflareBucketMountConfig::r2("assets", "acct"),
            )),
            "cloudflare_bucket_mount",
        ),
        (
            ProviderMountStrategy::BlaxelCloudBucket(BlaxelCloudBucketMountStrategy::new(
                BlaxelCloudBucketMountConfig::s3("bucket"),
            )),
            "blaxel_cloud_bucket",
        ),
        (
            ProviderMountStrategy::BlaxelDrive(BlaxelDriveMountStrategy::new(
                BlaxelDriveMountConfig::new("shared"),
            )),
            "blaxel_drive",
        ),
        (
            ProviderMountStrategy::VercelCloudBucket(VercelCloudBucketMountStrategy::new(
                "my-bucket",
            )),
            "vercel_cloud_bucket",
        ),
    ];
    for (strategy, expected) in cases {
        let payload = serde_json::to_value(&strategy).unwrap();
        assert_eq!(payload["type"], expected);
        assert_eq!(strategy.strategy_type(), expected);
        let text = payload.to_string();
        assert!(
            !text.contains("hunter2"),
            "{expected}: the secret stays out"
        );
        assert!(
            !text.contains("config_text"),
            "{expected}: the configuration stays out"
        );

        let restored = ProviderMountStrategy::parse(&payload).unwrap();

        assert_eq!(restored.strategy_type(), expected);
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            payload,
            "{expected}"
        );
        assert_eq!(restored.into_mount_strategy().strategy_type(), expected);
    }
    let rejected = ProviderMountStrategy::parse(&json!({"type": "no_such_strategy"}))
        .expect_err("an unknown strategy is refused");
    assert!(!rejected.message.contains("no_such_strategy"), "{rejected}");
}

/// Python: `test_core_discriminator_registries_parse_released_payload_shapes`: each released
/// payload shape the test asserts parses to the matching object.
#[test]
fn core_registries_parse_the_released_payload_shapes() {
    // `SnapshotBase.parse({"type": "noop", "id": ...})` gives the no-op snapshot.
    let snapshot: openai_agents_sdk::sandbox::SnapshotSpec =
        serde_json::from_value(json!({"type": "noop", "id": "snapshot-123"})).unwrap();
    assert_eq!(snapshot, openai_agents_sdk::sandbox::SnapshotSpec::Noop);

    // `BaseEntry.parse({"type": "dir", "permissions": {"directory": True}})` gives a directory.
    let entry = parse_entry(&json!({"type": "dir", "permissions": {"directory": true}})).unwrap();
    assert!(matches!(entry.kind, EntryKind::Dir { .. }));
    assert!(entry.permissions.directory);

    // `TypeAdapter(MountPattern).validate_python({"type": "mountpoint"})`.
    let pattern = parse_mount_pattern(&json!({"type": "mountpoint"})).unwrap();
    assert_eq!(pattern.pattern_type(), "mountpoint");

    // `MountStrategyBase.parse({"type": "docker_volume", "driver": "rclone"})`.
    let strategy =
        parse_core_mount_strategy(&json!({"type": "docker_volume", "driver": "rclone"})).unwrap();
    assert_eq!(strategy.strategy_type(), "docker_volume");

    // `EnvValue.parse({"type": "str", "value": "env-value"})`.
    let value = parse_env_value(&json!({"type": "str", "value": "env-value"})).unwrap();
    assert_eq!(value.env_value_type(), "str");

    assert!(parse_mount_pattern(&json!({"type": "no_such_pattern"})).is_err());
}

// ----------------------------------------------------------------------------------------------
// Every registered variant: write, parse through the registry, write again, compare
// ----------------------------------------------------------------------------------------------

mod every_variant {
    use std::sync::Arc;

    use openai_agents_sdk::sandbox::SnapshotSpec;
    use openai_agents_sdk::sandbox::entries::{Entry, EntryKind};
    use openai_agents_sdk::sandbox::environment::StrEnvValue;
    use openai_agents_sdk::sandbox::mount_entries::{
        AzureBlobMount, BoxMount, ConfiguredMountStrategy, GcsMount, R2Mount, S3FilesMount, S3Mount,
    };
    use openai_agents_sdk::sandbox::mount_patterns::{
        FuseMountPattern, MountpointMountPattern, S3FilesMountPattern,
    };
    use openai_agents_sdk::sandbox::mounts::{
        BindMountStrategy, DockerVolumeMountStrategy, InContainerMountStrategy, MountPatternConfig,
        RcloneMountPattern,
    };
    use openai_agents_sdk::sandbox::persisted_registry::{
        core_mount_strategy_to_value, entry_to_value, env_value_to_value, mount_pattern_to_value,
        parse_core_mount_strategy, parse_entry, parse_env_value, parse_mount_pattern,
        parse_snapshot, snapshot_to_value,
    };
    use openai_agents_sdk::sandbox::types::{Group, Permissions};
    use serde_json::{Value, json};

    /// Write an entry, parse that payload through the registry, and write the result again.
    fn round_trip_entry(entry: &Entry, expected_type: &str) -> Entry {
        let payload = entry_to_value(entry).unwrap();
        assert_eq!(payload["type"], expected_type);
        let parsed = parse_entry(&payload).unwrap();
        assert_eq!(entry_to_value(&parsed).unwrap(), payload, "{expected_type}");
        parsed
    }

    #[test]
    fn file_entry() {
        let entry = Entry::text_file("hello")
            .with_description("a greeting")
            .with_permissions(Permissions::default())
            .with_group(Group::new("staff"));
        let parsed = round_trip_entry(&entry, "file");
        assert!(matches!(&parsed.kind, EntryKind::File { content } if content == b"hello"));
        assert_eq!(parsed.description.as_deref(), Some("a greeting"));
        assert_eq!(
            parsed.group.as_ref().map(|g| g.name.as_str()),
            Some("staff")
        );
    }

    #[test]
    fn dir_entry_with_nested_children() {
        let entry = Entry::dir()
            .with_child("a.txt", Entry::text_file("a"))
            .with_child(
                "sub",
                Entry::dir().with_child("b.txt", Entry::text_file("b")),
            );
        let parsed = round_trip_entry(&entry, "dir");
        let EntryKind::Dir { children } = &parsed.kind else {
            panic!("a directory");
        };
        assert_eq!(children.len(), 2);
        assert!(
            matches!(&children["sub"].kind, EntryKind::Dir { children } if children.len() == 1)
        );
    }

    #[test]
    fn local_file_entry() {
        let parsed = round_trip_entry(&Entry::local_file("/host/data.txt"), "local_file");
        assert!(matches!(&parsed.kind, EntryKind::LocalFile { src } if src.ends_with("data.txt")));
    }

    #[test]
    fn local_dir_entry_with_and_without_a_source() {
        round_trip_entry(&Entry::local_dir("/host/dir"), "local_dir");
        let mut empty = Entry::dir();
        empty.kind = EntryKind::LocalDir { src: None };
        let parsed = round_trip_entry(&empty, "local_dir");
        assert!(matches!(parsed.kind, EntryKind::LocalDir { src: None }));
    }

    #[test]
    fn git_repo_entry() {
        let entry = Entry::git_repo("owner/name", "main")
            .with_git_host("git.example")
            .with_git_subpath("pkg");
        let parsed = round_trip_entry(&entry, "git_repo");
        assert!(matches!(
            &parsed.kind,
            EntryKind::GitRepo { host, subpath, .. }
                if host == "git.example" && subpath.as_deref() == Some("pkg")
        ));
    }

    #[test]
    fn every_mount_entry_with_every_backend() {
        let cases: Vec<(Entry, &str)> = vec![
            (
                Entry::mount(
                    ConfiguredMountStrategy::rclone(
                        S3Mount::new("bucket")
                            .with_credentials("AKIA", "secret")
                            .with_region("us-east-1"),
                    )
                    .mount(),
                ),
                "s3_mount",
            ),
            (
                Entry::mount(ConfiguredMountStrategy::mountpoint(S3Mount::new("bucket")).mount()),
                "s3_mount",
            ),
            (
                Entry::mount(
                    ConfiguredMountStrategy::rclone(R2Mount::new("assets", "acct")).mount(),
                ),
                "r2_mount",
            ),
            (
                Entry::mount(ConfiguredMountStrategy::rclone(GcsMount::new("bucket")).mount()),
                "gcs_mount",
            ),
            (
                Entry::mount(
                    ConfiguredMountStrategy::blobfuse(AzureBlobMount::new("account", "container"))
                        .mount(),
                ),
                "azure_blob_mount",
            ),
            (
                Entry::mount(ConfiguredMountStrategy::rclone(BoxMount::new()).mount()),
                "box_mount",
            ),
            (
                Entry::mount(ConfiguredMountStrategy::s3files(S3FilesMount::new("fs-123")).mount()),
                "s3_files_mount",
            ),
            (
                Entry::mount(
                    ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone")
                        .with_driver_option("path", "/data")
                        .mount(),
                ),
                "s3_mount",
            ),
        ];
        for (entry, expected) in cases {
            let parsed = round_trip_entry(&entry, expected);
            assert!(parsed.as_mount().is_some(), "{expected}");
        }
    }

    #[test]
    fn mount_entry_keeps_its_mount_path_and_read_only_flag() {
        let mut mount = ConfiguredMountStrategy::rclone(S3Mount::new("bucket")).mount();
        mount.mount_path = Some("/data".into());
        mount.read_only = false;
        let parsed = round_trip_entry(&Entry::mount(mount), "s3_mount");
        let mount = parsed.as_mount().unwrap();
        assert_eq!(mount.mount_path.as_deref(), Some("/data"));
        assert!(!mount.read_only);
    }

    #[test]
    fn unknown_entry_kinds_are_refused_without_echo() {
        let error = parse_entry(&json!({"type": "no_such_entry", "secret": "hunter2"}))
            .map(|_| ())
            .unwrap_err();
        assert!(!error.message.contains("hunter2"), "{error}");
        assert!(!error.message.contains("no_such_entry"), "{error}");
    }

    #[test]
    fn every_mount_pattern() {
        let cases: Vec<(
            Arc<dyn openai_agents_sdk::sandbox::mounts::MountPattern>,
            &str,
        )> = vec![
            (Arc::new(MountpointMountPattern::new()), "mountpoint"),
            (Arc::new(RcloneMountPattern::new()), "rclone"),
            (Arc::new(FuseMountPattern::new()), "fuse"),
            (Arc::new(S3FilesMountPattern::new()), "s3files"),
        ];
        for (pattern, expected) in cases {
            let payload = mount_pattern_to_value(pattern.as_ref()).unwrap();
            assert_eq!(payload["type"], expected);
            let parsed = parse_mount_pattern(&payload).unwrap();
            assert_eq!(parsed.pattern_type(), expected);
            assert_eq!(
                mount_pattern_to_value(parsed.as_ref()).unwrap(),
                payload,
                "{expected}"
            );
        }
    }

    #[test]
    fn every_core_mount_strategy() {
        let cases: Vec<(
            Arc<dyn openai_agents_sdk::sandbox::mounts::MountStrategy>,
            &str,
        )> = vec![
            (Arc::new(BindMountStrategy::new("/host/dir")), "bind"),
            (
                Arc::new(DockerVolumeMountStrategy::new("rclone")),
                "docker_volume",
            ),
            (
                Arc::new(InContainerMountStrategy::new(
                    Arc::new(RcloneMountPattern::new()),
                    MountPatternConfig::default(),
                )),
                "in_container",
            ),
            (
                Arc::new(InContainerMountStrategy::new(
                    Arc::new(FuseMountPattern::new()),
                    MountPatternConfig::default(),
                )),
                "in_container",
            ),
        ];
        for (strategy, expected) in cases {
            let payload = core_mount_strategy_to_value(strategy.as_ref()).unwrap();
            assert_eq!(payload["type"], expected);
            let parsed = parse_core_mount_strategy(&payload).unwrap();
            assert_eq!(parsed.strategy_type(), expected);
            assert_eq!(
                core_mount_strategy_to_value(parsed.as_ref()).unwrap(),
                payload,
                "{expected}"
            );
        }
    }

    #[test]
    fn in_container_payload_never_carries_the_runtime_configuration() {
        let mut config = MountPatternConfig::default();
        config.options.insert("remote".into(), "bucket".into());
        let strategy = InContainerMountStrategy::new(Arc::new(RcloneMountPattern::new()), config);
        let payload = core_mount_strategy_to_value(&strategy).unwrap();
        assert!(payload.get("config").is_none());
        assert!(!payload.to_string().contains("bucket"));
    }

    #[test]
    fn env_value_str() {
        let value = StrEnvValue::new("env-value");
        let payload = env_value_to_value(&value).unwrap();
        assert_eq!(payload, json!({"type": "str", "value": "env-value"}));
        let parsed = parse_env_value(&payload).unwrap();
        assert_eq!(parsed.env_value_type(), "str");
        assert_eq!(env_value_to_value(parsed.as_ref()).unwrap(), payload);
        assert!(parse_env_value(&json!({"type": "vault"})).is_err());
    }

    #[test]
    fn every_snapshot_kind() {
        let cases: Vec<(SnapshotSpec, Value)> = vec![
            (SnapshotSpec::Noop, json!({"type": "noop"})),
            (
                SnapshotSpec::Local {
                    base_path: "/tmp/snapshots".into(),
                },
                json!({"type": "local", "base_path": "/tmp/snapshots"}),
            ),
            (
                SnapshotSpec::Remote {
                    client_key: "client".into(),
                },
                json!({"type": "remote", "client_key": "client"}),
            ),
        ];
        for (snapshot, expected) in cases {
            let payload = snapshot_to_value(&snapshot).unwrap();
            assert_eq!(payload, expected);
            let parsed = parse_snapshot(&payload).unwrap();
            assert_eq!(parsed, snapshot);
        }
        assert!(parse_snapshot(&json!({"type": "no_such_snapshot"})).is_err());
    }
}
