//! Ports of cases of `tests/sandbox/test_mount_security.py` the earlier rounds filed as
//! Python-only although the contract is observable through the Rust API:
//! `test_preserves_credentialless_hosted_mount_strategies`,
//! `test_operation_error_with_implicit_broad_authority_is_replaced` and
//! `test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary`.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy,
};
use openai_agents_sdk::extensions::sandbox::daytona::DaytonaCloudBucketMountStrategy;
use openai_agents_sdk::extensions::sandbox::e2b::E2BCloudBucketMountStrategy;
use openai_agents_sdk::sandbox::sandboxes::docker::redact_for_protected_mounts;
use openai_agents_sdk::sandbox::{
    AzureBlobMount, BuiltinMountPattern, ConfiguredMountStrategy, Entry, ErrorCode,
    FuseMountPattern, Manifest, Mount, S3FilesMount, S3Mount, SandboxError,
};

fn manifest_with(mount: &Mount) -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(mount.clone()))
}

/// A hosted cloud-bucket mount with no credential activates without an acknowledgement; once it
/// carries one, the exact mount path has to be acknowledged first.
fn assert_hosted_mount_boundary(public: Mount, credentialed: Mount, label: &str) {
    manifest_with(&public)
        .authorize_mount_activation(&public, "/workspace/data")
        .unwrap_or_else(|error| panic!("{label}: a credentialless mount activates: {error}"));

    let manifest = manifest_with(&credentialed);
    let error = manifest
        .authorize_mount_activation(&credentialed, "/workspace/data")
        .expect_err("a credentialed mount needs an acknowledgement");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{label}");
    assert!(error.message.contains("mount_scoped"), "{label}: {error}");

    manifest
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&credentialed, "/workspace/data")
        .unwrap_or_else(|error| panic!("{label}: the acknowledged mount activates: {error}"));
}

/// Python: `test_preserves_credentialless_hosted_mount_strategies`, one case per hosted backend.
#[test]
fn hosted_bucket_strategies_keep_their_credential_boundary() {
    assert_hosted_mount_boundary(
        DaytonaCloudBucketMountStrategy::for_mount(S3Mount::new("example-bucket")).mount(),
        DaytonaCloudBucketMountStrategy::for_mount(
            S3Mount::new("example-bucket")
                .with_credentials("example-access-key", "example-secret-key"),
        )
        .mount(),
        "daytona",
    );
    assert_hosted_mount_boundary(
        E2BCloudBucketMountStrategy::for_mount(S3Mount::new("example-bucket")).mount(),
        E2BCloudBucketMountStrategy::for_mount(
            S3Mount::new("example-bucket")
                .with_credentials("example-access-key", "example-secret-key"),
        )
        .mount(),
        "e2b",
    );
    assert_hosted_mount_boundary(
        BlaxelCloudBucketMountStrategy::mount(BlaxelCloudBucketMountConfig::s3("example-bucket")),
        BlaxelCloudBucketMountStrategy::mount(
            BlaxelCloudBucketMountConfig::s3("example-bucket")
                .with_credentials("example-access-key", "example-secret-key"),
        ),
        "blaxel",
    );
}

/// Python: `test_operation_error_with_implicit_broad_authority_is_replaced`. A mount that holds
/// no credential but reaches for the container's ambient identity still protects the manifest,
/// so a provider error raised under it loses everything but its code.
#[test]
fn an_error_under_an_implicit_broad_authority_mount_is_replaced() {
    let mounts = [
        ConfiguredMountStrategy::in_container(
            AzureBlobMount::new("example", "private"),
            BuiltinMountPattern::Fuse(FuseMountPattern::new()),
        )
        .mount(),
        ConfiguredMountStrategy::s3files(S3FilesMount::new("fs-123")).mount(),
    ];
    for mount in mounts {
        let sentinel = "implicit-broad-provider-secret";
        let manifest = manifest_with(&mount)
            .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"]);
        let source = SandboxError::new(ErrorCode::ExecTransportError, sentinel)
            .with_context("detail", sentinel);

        let safe = redact_for_protected_mounts(&manifest, source);

        assert_eq!(safe.code, ErrorCode::ExecTransportError);
        assert!(!format!("{safe} {safe:?}").contains(sentinel), "{safe:?}");
        assert!(safe.context.is_empty(), "{:?}", safe.context);
    }
}

/// Python: `test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary`. The
/// manifest check refuses an entry path outside the root; with a credentialed mount in the
/// manifest the refusal keeps its code but carries neither the path nor the credential.
#[test]
fn an_invalid_entry_path_is_refused_without_the_mount_configuration() {
    let sentinel = "invalid-path-secret";
    let mount = ConfiguredMountStrategy::docker_volume(
        S3Mount::new("bucket").with_credentials("access-key", sentinel),
        "rclone",
    )
    .mount();
    let manifest = Manifest::new().with_entry("../data", Entry::mount(mount));

    let error = manifest
        .validate_mount_credential_boundaries()
        .expect_err("an entry path outside the root is refused");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(
        error.message.contains("protected mount configuration"),
        "{error}"
    );
    assert!(error.context.is_empty(), "{:?}", error.context);
    assert!(!format!("{error:?}").contains(sentinel));

    // With no credentialed mount the plain error keeps naming the path.
    let plain = Manifest::new()
        .with_entry("../data", Entry::file(b"x".to_vec()))
        .validate_mount_credential_boundaries()
        .expect_err("an entry path outside the root is refused");
    assert_eq!(plain.code, ErrorCode::InvalidManifestPath);
    assert!(plain.message.contains("../data"), "{plain}");
}
