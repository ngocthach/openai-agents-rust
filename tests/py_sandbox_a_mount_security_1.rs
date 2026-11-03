#![cfg(feature = "sandbox")]
//! Ports the credential-exposure acknowledgement half of Python's
//! `tests/sandbox/test_mount_security.py` (lines 237-660 and 1384-1700): which mounts expose a
//! credential inside the sandbox, what authority that credential carries, and what the
//! application has to acknowledge on the [`Manifest`] before such a mount may activate.
//!
//! Python runs these through `validate_manifest_mount_credential_boundaries`, a separate
//! manifest-wide pass. Rust classifies per mount through
//! `MountStrategy::credential_authority()` and enforces the acknowledgement at activation
//! through `Manifest::authorize_mount_activation`, so each Python assertion is translated into
//! the equivalent pair: the authority the mount reports, and whether activating it at a path is
//! allowed.

use openai_agents_sdk::sandbox::{
    AzureBlobMount, BoxMount, BuiltinMountPattern, ConfiguredMountStrategy, Entry, ErrorCode,
    FuseMountPattern, GcsMount, Manifest, Mount, MountCredentialAuthority, MountpointMountPattern,
    R2Mount, RcloneMountPattern, S3FilesMount, S3Mount,
};

/// An S3 mount attached with rclone inside the container, optionally with inline credentials.
fn s3_rclone_mount(credentialed: bool) -> Mount {
    let spec = if credentialed {
        S3Mount::new("example-bucket").with_credentials("example-access-key", "example-secret-key")
    } else {
        S3Mount::new("example-bucket")
    };
    ConfiguredMountStrategy::rclone(spec).mount()
}

/// A manifest rooted at `/workspace` holding one entry.
fn manifest_with(path: &str, mount: Mount) -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_entry(path, Entry::mount(mount))
}

/// The authority a mount's strategy reports.
fn authority(mount: &Mount) -> MountCredentialAuthority {
    mount.strategy.credential_authority()
}

// ---------------------------------------------------------------------------------------------
// test_rejects_explicit_credentials_for_in_container_mounts
// test_exact_path_acknowledgement_allows_supported_mount_scoped_credentials
// ---------------------------------------------------------------------------------------------

/// An in-container mount carrying inline credentials cannot activate unacknowledged, and the
/// rejection never reproduces the secret. Ports
/// `test_rejects_explicit_credentials_for_in_container_mounts`.
#[test]
fn explicit_credentials_on_an_in_container_mount_require_an_acknowledgement() {
    let mount = s3_rclone_mount(true);
    assert_eq!(authority(&mount), MountCredentialAuthority::MountScoped);

    let manifest = manifest_with("data", mount.clone());
    let error = manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("mount_scoped"), "{}", error.message);
    assert!(!error.message.contains("example-secret-key"));
    assert!(!format!("{error:?}").contains("example-secret-key"));
}

/// The acknowledgement covers one exact path: the same mount at a sibling path is still
/// rejected. Ports `test_exact_path_acknowledgement_allows_supported_mount_scoped_credentials`.
#[test]
fn an_acknowledgement_covers_only_the_exact_mount_path() {
    let mount = s3_rclone_mount(true);
    let manifest = manifest_with("data", mount.clone())
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);

    manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();

    let error = manifest
        .authorize_mount_activation(&mount, "/workspace/other")
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("/workspace/other"), "{error}");
}

/// A broad acknowledgement also covers a mount-scoped credential, but not the other way round.
/// Ports the acknowledgement half of
/// `test_blobfuse_account_key_requires_mount_scoped_and_broad_acknowledgement`.
#[test]
fn a_broad_acknowledgement_subsumes_a_mount_scoped_one() {
    let mount = s3_rclone_mount(true);
    let broad = manifest_with("data", mount.clone())
        .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"]);
    broad
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();

    // A broad-authority mount is not covered by a mount-scoped acknowledgement.
    let blobfuse = ConfiguredMountStrategy::blobfuse(AzureBlobMount::new("example", "public"))
        .mount()
        .with_mount_path("/workspace/blob");
    assert_eq!(authority(&blobfuse), MountCredentialAuthority::Broad);
    let scoped = manifest_with("blob", blobfuse.clone())
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/blob"]);
    let error = scoped
        .authorize_mount_activation(&blobfuse, "/workspace/blob")
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("broad"), "{error}");
}

/// A parent-directory acknowledgement does not reach a mount nested beneath it. Ports
/// `test_mount_credential_acknowledgement_is_not_a_path_prefix`.
#[test]
fn an_acknowledgement_is_not_a_path_prefix() {
    let mount = s3_rclone_mount(true);
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/parent"]);
    assert!(
        manifest
            .authorize_mount_activation(&mount, "/workspace/parent/data")
            .is_err()
    );

    let exact = manifest
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/parent/data"]);
    exact
        .authorize_mount_activation(&mount, "/workspace/parent/data")
        .unwrap();
}

/// Acknowledged paths are compared byte for byte, so trailing whitespace is significant. Ports
/// `test_mount_credential_acknowledgement_preserves_path_whitespace`.
#[test]
fn an_acknowledgement_preserves_path_whitespace() {
    let mount = s3_rclone_mount(true);
    let trimmed = Manifest::new()
        .with_root("/workspace")
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);
    assert!(
        trimmed
            .authorize_mount_activation(&mount, "/workspace/data ")
            .is_err()
    );

    let exact = Manifest::new()
        .with_root("/workspace")
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data "]);
    exact
        .authorize_mount_activation(&mount, "/workspace/data ")
        .unwrap();
}

/// A credential-free in-container mount activates without any acknowledgement, and a
/// docker-volume mount always does: the credential is handed to the container runtime on the
/// host, never to code inside the sandbox. Ports
/// `test_preserves_credentialless_in_container_and_credentialed_docker_mounts`.
#[test]
fn credentialless_in_container_and_credentialed_docker_volume_mounts_need_no_acknowledgement() {
    let manifest = Manifest::new().with_root("/workspace");

    let credentialless = s3_rclone_mount(false);
    assert_eq!(authority(&credentialless), MountCredentialAuthority::None);
    manifest
        .authorize_mount_activation(&credentialless, "/workspace/data")
        .unwrap();

    let docker = ConfiguredMountStrategy::docker_volume(
        S3Mount::new("example-bucket").with_credentials("example-access-key", "example-secret-key"),
        "rclone",
    )
    .mount();
    assert_eq!(authority(&docker), MountCredentialAuthority::None);
    manifest
        .authorize_mount_activation(&docker, "/workspace/data")
        .unwrap();
}

/// Environment values the manifest already exposes to the sandbox are not mount authority, so a
/// credential-free mount stays credential-free beside them. Ports
/// `test_ignores_environment_values_already_exposed_to_the_sandbox`.
#[test]
fn environment_values_already_exposed_are_not_mount_authority() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_env("AWS_SECRET_ACCESS_KEY", "secret")
        .with_env("GITHUB_TOKEN", "unrelated");
    let mount = ConfiguredMountStrategy::mountpoint(S3Mount::new("example-bucket")).mount();

    assert_eq!(authority(&mount), MountCredentialAuthority::None);
    manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Incomplete and blank credential sets
// ---------------------------------------------------------------------------------------------

/// A half-supplied S3 credential set is a misconfiguration, not a credential-free mount. Ports
/// `test_acknowledgement_rejects_incomplete_in_container_s3_credentials`, whose six parametrize
/// cases are looped here.
#[test]
fn an_incomplete_s3_credential_set_is_rejected() {
    let cases: Vec<(S3Mount, &str)> = vec![
        (
            S3Mount {
                access_key_id: Some("access-key".into()),
                ..S3Mount::new("example-bucket")
            },
            "secret_access_key",
        ),
        (
            S3Mount {
                secret_access_key: Some("secret-key".into()),
                ..S3Mount::new("example-bucket")
            },
            "access_key_id",
        ),
        (
            S3Mount {
                access_key_id: Some("access-key".into()),
                secret_access_key: Some(String::new()),
                ..S3Mount::new("example-bucket")
            },
            "secret_access_key",
        ),
        (
            S3Mount {
                access_key_id: Some(" ".into()),
                secret_access_key: Some("secret-key".into()),
                ..S3Mount::new("example-bucket")
            },
            "access_key_id",
        ),
        (
            S3Mount::new("example-bucket")
                .with_credentials("access-key", "secret-key")
                .with_session_token(" "),
            "session_token",
        ),
    ];

    for (spec, field) in cases {
        let mount = ConfiguredMountStrategy::rclone(spec).mount();
        let error = mount.strategy.validate(&mount).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
        assert!(error.message.contains(field), "{field}: {error}");
    }
}

/// A session token alone is a half-supplied credential set. Ports the `{"session_token": ...}`
/// parametrize case of `test_acknowledgement_rejects_incomplete_in_container_s3_credentials`.
#[test]
fn a_session_token_without_a_key_pair_is_rejected() {
    let mount = ConfiguredMountStrategy::rclone(
        S3Mount::new("example-bucket").with_session_token("session-token"),
    )
    .mount();

    let error = mount.strategy.validate(&mount).unwrap_err();

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("access_key_id"), "{error}");
    assert!(error.message.contains("secret_access_key"), "{error}");
    assert!(!error.message.contains("session-token"), "{error}");
}

/// A half-supplied GCS HMAC credential set is rejected. Ports
/// `test_acknowledgement_rejects_incomplete_in_container_gcs_hmac_credentials`.
#[test]
fn an_incomplete_gcs_hmac_credential_set_is_rejected() {
    let cases: Vec<(GcsMount, &str)> = vec![
        (
            GcsMount {
                access_id: Some("access-id".into()),
                ..GcsMount::new("example-bucket")
            },
            "secret_access_key",
        ),
        (
            GcsMount {
                secret_access_key: Some("secret-key".into()),
                ..GcsMount::new("example-bucket")
            },
            "access_id",
        ),
        (
            GcsMount {
                access_id: Some("access-id".into()),
                secret_access_key: Some(String::new()),
                ..GcsMount::new("example-bucket")
            },
            "secret_access_key",
        ),
        (
            GcsMount {
                access_id: Some(" ".into()),
                secret_access_key: Some("secret-key".into()),
                ..GcsMount::new("example-bucket")
            },
            "access_id",
        ),
        (
            GcsMount {
                access_id: Some("access-id".into()),
                service_account_credentials: Some("{\"type\":\"service_account\"}".into()),
                ..GcsMount::new("example-bucket")
            },
            "secret_access_key",
        ),
    ];

    for (spec, field) in cases {
        let mount = ConfiguredMountStrategy::rclone(spec).mount();
        let error = mount.strategy.validate(&mount).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
        assert!(error.message.contains(field), "{field}: {error}");
    }
}

/// A complete HMAC pair is accepted and classified as mount-scoped. Ports
/// `test_acknowledgement_accepts_complete_in_container_gcs_hmac_credentials`.
#[test]
fn a_complete_gcs_hmac_credential_set_is_accepted() {
    let mount = ConfiguredMountStrategy::rclone(
        GcsMount::new("example-bucket").with_hmac_credentials("access-id", "secret-key"),
    )
    .mount();

    mount.strategy.validate(&mount).unwrap();
    assert_eq!(authority(&mount), MountCredentialAuthority::MountScoped);

    let manifest = manifest_with("data", mount.clone())
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);
    manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A half-supplied R2 credential set is rejected. Ports
/// `test_acknowledgement_rejects_incomplete_in_container_r2_credentials`.
#[test]
fn an_incomplete_r2_credential_set_is_rejected() {
    let cases: Vec<(R2Mount, &str)> = vec![
        (
            R2Mount {
                access_key_id: Some("access-key".into()),
                ..R2Mount::new("example-bucket", "example-account")
            },
            "secret_access_key",
        ),
        (
            R2Mount {
                secret_access_key: Some("secret-key".into()),
                ..R2Mount::new("example-bucket", "example-account")
            },
            "access_key_id",
        ),
        (
            R2Mount {
                access_key_id: Some("access-key".into()),
                secret_access_key: Some(String::new()),
                ..R2Mount::new("example-bucket", "example-account")
            },
            "secret_access_key",
        ),
    ];

    for (spec, field) in cases {
        let mount = ConfiguredMountStrategy::rclone(spec).mount();
        let error = mount.strategy.validate(&mount).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
        assert!(error.message.contains(field), "{field}: {error}");
    }
}

/// A scalar authority field that is present but blank is rejected rather than silently written
/// into the generated configuration. Ports
/// `test_acknowledgement_rejects_empty_in_container_scalar_authority` (five mounts x two blank
/// values).
#[test]
fn a_blank_scalar_authority_field_is_rejected() {
    for blank in ["", "   "] {
        let gcs_token = ConfiguredMountStrategy::rclone(GcsMount {
            access_token: Some(blank.into()),
            ..GcsMount::new("example-bucket")
        })
        .mount();
        let gcs_credentials = ConfiguredMountStrategy::rclone(GcsMount {
            service_account_credentials: Some(blank.into()),
            ..GcsMount::new("example-bucket")
        })
        .mount();
        let gcs_file = ConfiguredMountStrategy::rclone(GcsMount {
            service_account_file: Some(blank.into()),
            ..GcsMount::new("example-bucket")
        })
        .mount();
        let azure_key = ConfiguredMountStrategy::rclone(
            AzureBlobMount::new("example-account", "example-container").with_account_key(blank),
        )
        .mount();
        let azure_identity = ConfiguredMountStrategy::rclone(
            AzureBlobMount::new("example-account", "example-container")
                .with_identity_client_id(blank),
        )
        .mount();

        for (mount, field) in [
            (gcs_token, "access_token"),
            (gcs_credentials, "service_account_credentials"),
            (gcs_file, "service_account_file"),
            (azure_key, "account_key"),
            (azure_identity, "identity_client_id"),
        ] {
            let error = mount.strategy.validate(&mount).unwrap_err();
            assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
            assert!(error.message.contains(field), "{field}: {error}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Broad authority: blobfuse, S3 Files, Box
// ---------------------------------------------------------------------------------------------

/// A blobfuse mount reaches for the container's managed identity, which is broad authority even
/// with no credential on the mount. Ports `test_blobfuse_mounts_require_broad_acknowledgement`.
#[test]
fn a_blobfuse_mount_requires_a_broad_acknowledgement() {
    let mount = ConfiguredMountStrategy::blobfuse(AzureBlobMount::new("example", "public")).mount();
    assert_eq!(authority(&mount), MountCredentialAuthority::Broad);

    let manifest = manifest_with("data", mount.clone());
    let error = manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("broad"), "{error}");

    manifest
        .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A blobfuse mount with an account key still reports broad authority, because the mount tool
/// falls back to the ambient identity. Ports
/// `test_blobfuse_account_key_requires_mount_scoped_and_broad_acknowledgement`; Rust collapses
/// Python's two acknowledgement kinds into one broad requirement.
#[test]
fn a_blobfuse_account_key_still_requires_the_broad_acknowledgement() {
    let mount = ConfiguredMountStrategy::blobfuse(
        AzureBlobMount::new("example", "private").with_account_key("account-key"),
    )
    .mount();
    assert_eq!(authority(&mount), MountCredentialAuthority::Broad);

    let manifest = manifest_with("data", mount.clone());
    assert!(
        manifest
            .clone()
            .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"])
            .authorize_mount_activation(&mount, "/workspace/data")
            .is_err()
    );
    manifest
        .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// The S3 Files mount helper authenticates with the instance's ambient IAM role. Ports
/// `test_s3_files_require_broad_acknowledgement_before_ambient_iam_can_be_used`.
#[test]
fn an_s3_files_mount_requires_a_broad_acknowledgement() {
    let mount = ConfiguredMountStrategy::s3files(
        S3FilesMount::new("fs-123").with_extra_option("tlsport", Some("4049")),
    )
    .mount();
    assert_eq!(authority(&mount), MountCredentialAuthority::Broad);

    let manifest = manifest_with("data", mount.clone());
    assert!(
        manifest
            .authorize_mount_activation(&mount, "/workspace/data")
            .is_err()
    );
    manifest
        .clone()
        .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A Box mount holding a direct token exposes a mount-scoped credential. Ports
/// `test_box_mounts_with_direct_credentials_require_exact_acknowledgement`.
#[test]
fn a_box_mount_with_a_direct_token_requires_an_exact_acknowledgement() {
    let mount =
        ConfiguredMountStrategy::rclone(BoxMount::new().with_access_token("box-access-token"))
            .mount();
    assert_eq!(authority(&mount), MountCredentialAuthority::MountScoped);

    let manifest = manifest_with("data", mount.clone());
    assert!(
        manifest
            .authorize_mount_activation(&mount, "/workspace/data")
            .is_err()
    );
    manifest
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A Box configuration file lives outside the manifest, so the SDK cannot see what it grants:
/// that is broad authority. Ports `test_box_config_file_requires_broad_acknowledgement`.
#[test]
fn a_box_configuration_file_requires_a_broad_acknowledgement() {
    let mount = ConfiguredMountStrategy::rclone(BoxMount {
        box_config_file: Some("/run/secrets/box.json".into()),
        ..BoxMount::new()
    })
    .mount();
    assert_eq!(authority(&mount), MountCredentialAuthority::Broad);

    let manifest = manifest_with("data", mount.clone());
    assert!(
        manifest
            .clone()
            .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"])
            .authorize_mount_activation(&mount, "/workspace/data")
            .is_err()
    );
    manifest
        .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"])
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A Box mount with no usable authentication source cannot be attached non-interactively: there
/// is nothing to log in with and rclone would block on a browser prompt. Ports
/// `test_box_in_container_mount_requires_non_interactive_authentication`.
#[test]
fn a_box_in_container_mount_requires_a_non_interactive_authentication_source() {
    let cases = [
        BoxMount::new(),
        BoxMount {
            client_id: Some("client-id".into()),
            ..BoxMount::new()
        },
        BoxMount {
            client_secret: Some("client-secret".into()),
            ..BoxMount::new()
        },
    ];

    for spec in cases {
        let mount = ConfiguredMountStrategy::rclone(spec).mount();
        let error = mount.strategy.validate(&mount).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(
            error.message.contains("non-interactive") || error.message.contains("authentication"),
            "{error}"
        );
    }
}

/// A Box authentication source that is present but blank is rejected. Ports
/// `test_box_in_container_mount_rejects_empty_authentication_sources` (four fields x two blank
/// values).
#[test]
fn a_blank_box_authentication_source_is_rejected() {
    for blank in ["", "   "] {
        let cases: Vec<(BoxMount, &str)> = vec![
            (
                BoxMount {
                    access_token: Some(blank.into()),
                    ..BoxMount::new()
                },
                "access_token",
            ),
            (
                BoxMount {
                    token: Some(blank.into()),
                    ..BoxMount::new()
                },
                "token",
            ),
            (
                BoxMount {
                    config_credentials: Some(blank.into()),
                    ..BoxMount::new()
                },
                "config_credentials",
            ),
            (
                BoxMount {
                    box_config_file: Some(blank.into()),
                    ..BoxMount::new()
                },
                "box_config_file",
            ),
        ];
        for (spec, field) in cases {
            let mount = ConfiguredMountStrategy::rclone(spec).mount();
            let error = mount.strategy.validate(&mount).unwrap_err();
            assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
        }
    }
}

/// A usable Box source beside a blank one is still rejected, naming the blank field. Ports
/// `test_box_in_container_mount_rejects_mixed_usable_and_empty_authentication_sources`.
#[test]
fn a_blank_box_authentication_source_beside_a_usable_one_is_rejected() {
    for blank in ["", "   "] {
        let with_blank_file = ConfiguredMountStrategy::rclone(BoxMount {
            access_token: Some("box-access-token".into()),
            box_config_file: Some(blank.into()),
            ..BoxMount::new()
        })
        .mount();
        let with_blank_token = ConfiguredMountStrategy::rclone(BoxMount {
            access_token: Some(blank.into()),
            box_config_file: Some("/run/secrets/box.json".into()),
            ..BoxMount::new()
        })
        .mount();

        for (mount, field) in [
            (with_blank_file, "box_config_file"),
            (with_blank_token, "access_token"),
        ] {
            let error = mount.strategy.validate(&mount).unwrap_err();
            assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
            assert!(error.message.contains(field), "{field}: {error}");
        }
    }
}

/// A Box mount attached by the container runtime hands its token to the host, so no
/// acknowledgement is required. Ports `test_preserves_box_mounts_with_an_external_strategy`.
#[test]
fn a_box_mount_on_a_docker_volume_strategy_needs_no_acknowledgement() {
    let mount = ConfiguredMountStrategy::docker_volume(
        BoxMount::new().with_access_token("box-access-token"),
        "rclone",
    )
    .mount();

    assert_eq!(authority(&mount), MountCredentialAuthority::None);
    Manifest::new()
        .with_root("/workspace")
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}

/// A multi-line service-account key is ordinary external provider configuration: it is compacted
/// rather than rejected. Ports `test_preserves_multiline_external_mount_credentials`.
#[test]
fn a_multiline_external_credential_is_preserved() {
    let mount = ConfiguredMountStrategy::docker_volume(
        GcsMount::new("bucket")
            .with_service_account_credentials("{\"private_key\":\"line-1\nline-2\"}"),
        "rclone",
    )
    .mount();

    // Rust compacts a JSON credential rather than rejecting the line break, and the docker
    // volume driver configuration is built during validation.
    mount.strategy.validate(&mount).unwrap();
    assert_eq!(authority(&mount), MountCredentialAuthority::None);
}

// ---------------------------------------------------------------------------------------------
// Pattern support
// ---------------------------------------------------------------------------------------------

/// Each mount declares which in-container patterns can attach it; an unsupported pairing is
/// rejected before anything runs. Covers the structural half of
/// `test_rejects_credential_required_patterns_in_inherited_in_container_strategies`, whose
/// subclass-provenance half cannot exist in Rust.
#[test]
fn a_mount_rejects_an_unsupported_in_container_pattern() {
    let azure = ConfiguredMountStrategy::in_container(
        AzureBlobMount::new("example", "public"),
        BuiltinMountPattern::Mountpoint(MountpointMountPattern::new()),
    )
    .mount();
    let error = azure.strategy.validate(&azure).unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("azure_blob_mount"), "{error}");
    assert!(error.message.contains("mountpoint"), "{error}");

    let s3_files = ConfiguredMountStrategy::in_container(
        S3FilesMount::new("fs-123"),
        BuiltinMountPattern::Fuse(FuseMountPattern::new()),
    )
    .mount();
    let error = s3_files.strategy.validate(&s3_files).unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("s3_files_mount"), "{error}");
}

/// A mount attached with rclone reports broad authority as soon as the pattern reads a
/// configuration the caller placed in the workspace, because the SDK cannot see what that file
/// grants. Ports the `has_config_file` half of the Python authority table.
#[test]
fn a_caller_placed_rclone_configuration_is_broad_authority() {
    let mount = ConfiguredMountStrategy::in_container(
        S3Mount::new("bucket"),
        BuiltinMountPattern::Rclone(
            RcloneMountPattern::new().with_config_file_path("credentials.conf"),
        ),
    )
    .mount();

    assert_eq!(authority(&mount), MountCredentialAuthority::Broad);
    let manifest = manifest_with("data", mount.clone())
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);
    assert!(
        manifest
            .authorize_mount_activation(&mount, "/workspace/data")
            .is_err()
    );
}

// ---------------------------------------------------------------------------------------------
// External (docker volume) providers and acknowledgement inputs
// ---------------------------------------------------------------------------------------------

/// A half-supplied or blank credential on an *external* strategy is provider configuration the
/// container runtime sorts out, so it stays valid and reports no sandbox-side authority. Ports
/// `test_incomplete_credentials_remain_external_provider_configuration` (five parametrize cases).
#[test]
fn incomplete_credentials_remain_external_provider_configuration() {
    let mounts: Vec<Mount> = vec![
        ConfiguredMountStrategy::docker_volume(
            S3Mount {
                access_key_id: Some("access-key".into()),
                ..S3Mount::new("example-bucket")
            },
            "rclone",
        )
        .mount(),
        ConfiguredMountStrategy::docker_volume(
            GcsMount {
                access_id: Some("access-id".into()),
                ..GcsMount::new("example-bucket")
            },
            "rclone",
        )
        .mount(),
        ConfiguredMountStrategy::docker_volume(
            R2Mount {
                access_key_id: Some("access-key".into()),
                ..R2Mount::new("example-bucket", "example-account")
            },
            "rclone",
        )
        .mount(),
        ConfiguredMountStrategy::docker_volume(
            GcsMount {
                access_token: Some(String::new()),
                ..GcsMount::new("example-bucket")
            },
            "rclone",
        )
        .mount(),
        ConfiguredMountStrategy::docker_volume(
            AzureBlobMount::new("example-account", "example-container")
                .with_identity_client_id(" "),
            "rclone",
        )
        .mount(),
    ];

    for mount in mounts {
        mount.strategy.validate(&mount).unwrap();
        assert_eq!(authority(&mount), MountCredentialAuthority::None);
    }
}

/// Acknowledgement paths are validated the way Python validates them: no path at all, an empty
/// path, the workspace root, a backslash, wildcard syntax and `..` segments are all rejected by
/// the `try_` builders. Ports `test_manifest_acknowledgement_is_runtime_only_and_rejects_root`
/// and `test_manifest_acknowledgement_rejects_wildcard_and_parent_paths`.
#[test]
fn acknowledgement_paths_are_validated_like_python() {
    let plain = || Manifest::new().with_root("/workspace");

    let no_paths: [&str; 0] = [];
    assert!(
        plain()
            .try_with_in_container_mount_credential_exposure_acknowledged(no_paths)
            .is_err()
    );
    for path in [
        "",
        "/workspace",
        "/",
        "data/*",
        "data?",
        "data[0]",
        "data/../other",
        "/workspace/../outside",
        "data\\child",
    ] {
        let narrow = plain().try_with_in_container_mount_credential_exposure_acknowledged([path]);
        let broad =
            plain().try_with_in_container_mount_broad_credential_exposure_acknowledged([path]);
        for (kind, result) in [("mount-scoped", narrow), ("broad", broad)] {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{kind} accepted {path:?}"));
            assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{path:?}");
        }
    }

    // The old builders stay as fail-closed wrappers: an invalid path acknowledges nothing.
    let mount = s3_rclone_mount(true);
    for path in ["/workspace", "data/*", "data/../other", "data\\child"] {
        let manifest = plain()
            .with_in_container_mount_credential_exposure_acknowledged([path])
            .with_in_container_mount_broad_credential_exposure_acknowledged([path]);
        for target in ["/workspace/data", "/workspace/other", "/outside"] {
            assert!(
                manifest.authorize_mount_activation(&mount, target).is_err(),
                "{path} matched {target}"
            );
        }
    }
}

/// A relative acknowledgement and the absolute path under the workspace root name the same
/// mount, in either direction. Ports the relative/absolute lookup of
/// `test_mount_credential_acknowledgement_accepts_platform_path_objects`.
#[test]
fn a_relative_and_an_absolute_acknowledgement_name_the_same_mount() {
    let mount = s3_rclone_mount(true);

    let relative = Manifest::new()
        .with_root("/workspace")
        .try_with_in_container_mount_credential_exposure_acknowledged(["data"])
        .unwrap();
    relative
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
    assert!(
        relative
            .authorize_mount_activation(&mount, "/workspace/other")
            .is_err()
    );

    let absolute = Manifest::new()
        .with_root("/workspace")
        .try_with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"])
        .unwrap();
    absolute
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
}
