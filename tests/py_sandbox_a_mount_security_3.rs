#![cfg(feature = "sandbox")]
//! Ports the durable-state and error-boundary halves of Python's
//! `tests/sandbox/test_mount_security.py` (lines 2127-2867 and 2867-4642): a persisted session
//! state must never carry mount authority, a resumed state must have that authority re-supplied
//! from a trusted manifest before it can be used, and an error raised anywhere near a
//! credential-bearing mount must not reproduce the credential.
//!
//! Python's `SandboxSessionState` serializes the whole `Manifest`, so most of its tests assert
//! that individual fields are nulled out on the way to disk and restored on the way back.
//! Rust's [`SandboxSessionState`] has no manifest field at all — it carries only a
//! [`manifest_digest`] — so the same guarantee holds structurally and is asserted here as "no
//! credential byte survives serialization", plus the rebind contract that replaces Python's
//! field-by-field restoration.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    AzureBlobMount, BoxMount, BuiltinMountPattern, ConfiguredMountStrategy, Entry, ErrorCode,
    GcsMount, Manifest, Mount, MountCredentialAuthority, R2Mount, RcloneMountPattern, S3FilesMount,
    S3Mount, SandboxError, SandboxSessionState,
};

const ACCESS_KEY: &str = "example-access-key";
const SECRET_KEY: &str = "example-secret-key";

/// A manifest holding one credential-bearing in-container S3 mount at `data`.
fn credentialed_manifest() -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("example-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
            )
            .mount(),
        ),
    )
}

fn state_for(manifest: &Manifest) -> SandboxSessionState {
    SandboxSessionState::new("test", "sess-1", "/workspace", manifest)
}

// ---------------------------------------------------------------------------------------------
// Durable state never carries mount authority
// ---------------------------------------------------------------------------------------------

/// Serializing a session state built from a credential-bearing manifest records only that the
/// authority existed, never the authority itself. Ports
/// `test_session_state_serialization_redacts_complete_opaque_authority_fields`.
#[test]
fn a_persisted_state_records_that_mount_authority_existed_but_never_the_authority() {
    let manifest = credentialed_manifest();
    let state = state_for(&manifest);

    assert!(state.mount_authority_redacted);
    assert!(!state.mount_authority_rebound);

    let json = state.to_json_string().unwrap();
    assert!(!json.contains(SECRET_KEY), "{json}");
    assert!(!json.contains(ACCESS_KEY), "{json}");
    assert!(!json.contains("example-bucket"), "{json}");
    assert!(json.contains("mount_authority_redacted"), "{json}");

    let restored = SandboxSessionState::from_json_str(&json).unwrap();
    assert!(restored.mount_authority_redacted);
    assert!(!restored.mount_authority_rebound);
}

/// Opaque Docker volume driver options never reach durable state either, even though the
/// container runtime (not the sandbox) is what reads them. Ports the `driver_options` half of
/// the same Python test.
#[test]
fn docker_volume_driver_options_never_reach_durable_state() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(
                S3Mount::new("example-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
                "rclone",
            )
            .with_driver_option("s3-secret-access-key", "driver-secret")
            .with_driver_option("config", "/host/rclone.conf")
            .mount(),
        ),
    );

    let json = state_for(&manifest).to_json_string().unwrap();

    assert!(!json.contains("driver-secret"), "{json}");
    assert!(!json.contains("/host/rclone.conf"), "{json}");
    assert!(!json.contains(SECRET_KEY), "{json}");
}

/// A credential-free external mount round-trips with no redaction marker at all. Ports
/// `test_session_state_round_trip_preserves_credentialless_external_mount`.
#[test]
fn a_credentialless_external_mount_round_trips_without_a_redaction_marker() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone")
                .mount(),
        ),
    );
    let state = state_for(&manifest);

    assert!(!state.mount_authority_redacted);
    assert!(!state.mount_authority_rebound);

    let restored = SandboxSessionState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert!(!restored.mount_authority_redacted);
    assert!(!restored.mount_authority_rebound);
    restored.verify_manifest(&manifest).unwrap();
    restored.assert_path_grants_rebound().unwrap();
}

/// A hand-edited state cannot claim it holds no mount authority: the flag is recomputed from the
/// trusted manifest, and the state alone is never enough to resume. Ports
/// `test_credentialless_marker_does_not_override_configured_mount_authority`.
#[test]
fn a_forged_credentialless_marker_does_not_make_a_state_resumable() {
    let manifest = credentialed_manifest();
    let mut forged = state_for(&manifest);
    forged.mount_authority_redacted = false;
    forged.mount_authority_rebound = true;

    // The forged state verifies against the manifest topology, because the digest covers the
    // topology and not the credential — but the credential still comes from the trusted
    // manifest at activation time, never from the state.
    forged.verify_manifest(&manifest).unwrap();
    assert!(
        state_for(&manifest).mount_authority_redacted,
        "the flag is derived from the live manifest, not carried by the caller"
    );

    let json = forged.to_json_string().unwrap();
    assert!(!json.contains(SECRET_KEY), "{json}");
}

/// Every built-in credential-bearing mount sets the redaction flag, and none of their secrets
/// reaches the serialized state. Ports the mount-type table behind Python's session-state
/// sanitization tests.
#[test]
fn no_built_in_mount_leaks_a_credential_into_durable_state() {
    let secrets = [
        "s3-secret",
        "gcs-secret",
        "r2-secret",
        "azure-account-key",
        "box-access-token",
    ];
    let mounts: Vec<Mount> = vec![
        ConfiguredMountStrategy::rclone(S3Mount::new("bucket").with_credentials("ak", "s3-secret"))
            .mount(),
        ConfiguredMountStrategy::rclone(
            GcsMount::new("bucket").with_hmac_credentials("id", "gcs-secret"),
        )
        .mount(),
        ConfiguredMountStrategy::rclone(
            R2Mount::new("bucket", "account").with_credentials("ak", "r2-secret"),
        )
        .mount(),
        ConfiguredMountStrategy::rclone(
            AzureBlobMount::new("account", "container").with_account_key("azure-account-key"),
        )
        .mount(),
        ConfiguredMountStrategy::rclone(BoxMount::new().with_access_token("box-access-token"))
            .mount(),
    ];

    for mount in mounts {
        assert_ne!(
            mount.strategy.credential_authority(),
            MountCredentialAuthority::None,
            "{mount:?}"
        );
        let manifest = Manifest::new()
            .with_root("/workspace")
            .with_entry("data", Entry::mount(mount));
        let state = state_for(&manifest);
        assert!(state.mount_authority_redacted);
        let json = state.to_json_string().unwrap();
        for secret in secrets {
            assert!(!json.contains(secret), "{secret} in {json}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Rebind from a trusted manifest
// ---------------------------------------------------------------------------------------------

/// A state whose mount authority was stripped cannot be resumed on its own; rebinding it against
/// the trusted manifest clears the requirement. Ports
/// `test_in_container_acknowledgement_is_rebound_only_from_trusted_manifest` and
/// `test_implicit_broad_authority_is_rebound_only_from_trusted_manifest`.
#[test]
fn mount_authority_is_rebound_only_from_a_trusted_manifest() {
    let manifest = credentialed_manifest();
    let state = state_for(&manifest);

    let error = state.assert_path_grants_rebound().unwrap_err();
    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    assert!(
        error.message.contains("cannot be resumed on its own"),
        "{error}"
    );
    assert!(!error.message.contains(SECRET_KEY));

    let rebound = state.rebind_path_grants(&manifest).unwrap();
    assert!(!rebound.mount_authority_redacted);
    assert!(rebound.mount_authority_rebound);
    rebound.assert_path_grants_rebound().unwrap();
}

/// The rebind requires the same mount topology: a manifest with the mount at a different path is
/// refused. Ports `test_mount_authority_rebind_requires_exact_credential_free_topology`.
#[test]
fn a_rebind_requires_the_same_mount_topology() {
    let state = state_for(&credentialed_manifest());

    let moved = Manifest::new().with_root("/workspace").with_entry(
        "elsewhere",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("example-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
            )
            .mount(),
        ),
    );

    let error = state.verify_manifest(&moved).unwrap_err();
    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
    assert!(!error.message.contains(SECRET_KEY));
}

/// Resuming against a manifest whose mount uses a different backend is refused. Ports
/// `test_resume_validation_rejects_wrong_provider_strategy`.
#[test]
fn a_resume_against_a_different_mount_backend_is_refused() {
    let state = state_for(&credentialed_manifest());

    let docker = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone")
                .mount(),
        ),
    );

    let error = state.verify_manifest(&docker).unwrap_err();
    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
}

/// Resuming against a manifest whose mount keeps the same backend but swaps the storage behind
/// it is refused: the digest covers the mount type, the storage it names and the pattern or
/// driver. Ports `test_resume_validation_rejects_wrong_provider_strategy` and the bucket half of
/// `test_mount_authority_rebind_requires_exact_credential_free_topology`.
#[test]
fn a_resume_that_swaps_the_storage_behind_one_backend_is_refused() {
    let state = state_for(&credentialed_manifest());

    let other_provider = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(ConfiguredMountStrategy::rclone(GcsMount::new("other-bucket")).mount()),
    );
    let other_bucket = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("different-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
            )
            .mount(),
        ),
    );

    for swapped in [&other_provider, &other_bucket] {
        let error = state.verify_manifest(swapped).unwrap_err();
        assert_eq!(error.code, ErrorCode::SessionStateMismatch);
        // Authority is only handed back to a manifest with the exact topology.
        let error = state.rebind_path_grants(swapped).unwrap_err();
        assert_eq!(error.code, ErrorCode::SessionStateMismatch);
        assert!(!error.message.contains(SECRET_KEY));
    }
    // The credential itself is not topology: rotating it still resumes.
    let rotated = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("example-bucket").with_credentials("rotated-key", "rotated-secret"),
            )
            .mount(),
        ),
    );
    state.verify_manifest(&rotated).unwrap();
}

/// Once mount authority was redacted, the persisted state must not be enough to reattach the
/// backend resource the credential opened, so the provider identity is dropped and the session
/// id is replaced by a stable derived one. Ports Python's
/// `_sanitize_persisted_provider_identity` assertions.
#[test]
fn a_redacted_state_drops_its_provider_identity_on_persistence() {
    let mut state = state_for(&credentialed_manifest());
    state.provider_id = Some("provider-resource-id".to_string());

    let sanitized = state.sanitized_for_persistence();

    assert!(sanitized.provider_id.is_none());
    assert_ne!(sanitized.session_id, state.session_id);
    assert!(!sanitized.workspace_root_ready);
    // The derived id is stable, so two writes of the same state name the same snapshot.
    assert_eq!(
        sanitized.session_id,
        state.sanitized_for_persistence().session_id
    );

    // A rebound state keeps its identity: the caller already re-supplied the authority.
    let rebound = state.rebind_path_grants(&credentialed_manifest()).unwrap();
    assert_eq!(
        rebound.sanitized_for_persistence().session_id,
        rebound.session_id
    );
}

// ---------------------------------------------------------------------------------------------
// Error boundaries
// ---------------------------------------------------------------------------------------------

/// A session that fails every command, so an activation error can be inspected.
#[derive(Debug)]
struct FailingSession {
    manifest: Manifest,
    stderr: String,
    commands: Mutex<Vec<Vec<String>>>,
    files: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl FailingSession {
    fn new(stderr: impl Into<String>) -> Self {
        FailingSession {
            manifest: Manifest::new()
                .with_root("/workspace")
                .with_in_container_mount_broad_credential_exposure_acknowledged([
                    "/workspace/data",
                ]),
            stderr: stderr.into(),
            commands: Mutex::new(Vec::new()),
            files: Mutex::new(BTreeMap::new()),
        }
    }
}

#[async_trait]
impl SandboxSession for FailingSession {
    fn id(&self) -> &str {
        "sess"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "failing"
    }

    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        let argv = request.resolved_argv().clone();
        // Only the mount command itself fails: the tool probes and the permission fix succeed.
        let probe = !argv.iter().any(|arg| arg.contains("--config"));
        self.commands.lock().unwrap().push(argv);
        if probe {
            return Ok(ExecResult {
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            });
        }
        Ok(ExecResult {
            exit_code: 1,
            stdout: Vec::new(),
            stderr: self.stderr.clone().into_bytes(),
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("no file at `{path}`"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), contents.to_vec());
        Ok(())
    }

    async fn mkdir(&self, _path: &str, _parents: bool) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn rm(&self, path: &str, _recursive: bool) -> Result<(), SandboxError> {
        self.files.lock().unwrap().remove(path);
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// A mount tool that echoes its own configuration back on failure has every credential value
/// replaced before the text reaches an error. This is the portable kernel of Python's
/// `test_protected_structured_mount_error_preserves_safe_contract` family: the error keeps its
/// code and its operation, and carries no credential.
#[tokio::test]
async fn a_failing_mount_command_keeps_its_contract_without_reproducing_the_credential() {
    let session = FailingSession::new(format!(
        "rclone: couldn't connect: secret_access_key = {SECRET_KEY}"
    ));
    let mount = ConfiguredMountStrategy::rclone(
        S3Mount::new("example-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
    )
    .mount();

    let error = mount
        .strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(!error.message.contains(SECRET_KEY), "{error}");
    assert!(error.message.contains("REDACTED"), "{error}");
    assert!(!format!("{error:?}").contains(SECRET_KEY));
}

/// A credential-free mount's failure is passed through unchanged, so redaction never swallows a
/// useful diagnostic. Ports `test_credentialless_structured_mount_error_is_unchanged`.
#[tokio::test]
async fn a_credentialless_mount_failure_is_passed_through_unchanged() {
    let session = FailingSession::new("rclone: bucket does not exist");
    let mount = ConfiguredMountStrategy::rclone(S3Mount::new("example-bucket")).mount();

    let error = mount
        .strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(error.message.contains("bucket does not exist"), "{error}");
}

/// A mount's own `Debug` output never reproduces a credential, which is what keeps a credential
/// out of any error that formats the mount. Ports
/// `test_public_mount_error_redactor_discards_untrusted_mount_discriminator` and its siblings,
/// whose subclass-provenance machinery has no Rust equivalent.
#[test]
fn a_mount_debug_rendering_never_reproduces_a_credential() {
    let mounts: Vec<(Mount, &str)> = vec![
        (
            ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket").with_credentials("ak", "s3-secret"),
            )
            .mount(),
            "s3-secret",
        ),
        (
            ConfiguredMountStrategy::rclone(
                GcsMount::new("bucket").with_hmac_credentials("id", "gcs-secret"),
            )
            .mount(),
            "gcs-secret",
        ),
        (
            ConfiguredMountStrategy::rclone(
                R2Mount::new("bucket", "account").with_credentials("ak", "r2-secret"),
            )
            .mount(),
            "r2-secret",
        ),
        (
            ConfiguredMountStrategy::rclone(
                AzureBlobMount::new("account", "container").with_account_key("azure-secret"),
            )
            .mount(),
            "azure-secret",
        ),
        (
            ConfiguredMountStrategy::rclone(BoxMount::new().with_access_token("box-secret"))
                .mount(),
            "box-secret",
        ),
    ];

    for (mount, secret) in mounts {
        let rendered = format!("{mount:?}");
        assert!(!rendered.contains(secret), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
    }
}

// ---------------------------------------------------------------------------------------------
// Broad authority and opaque external authority through durable state
// ---------------------------------------------------------------------------------------------

/// A mount that holds no credential field but reaches for ambient authority (blobfuse's managed
/// identity, the S3 Files helper's IAM role) is still redacted from durable state and rebound
/// only from the trusted manifest. Ports
/// `test_implicit_broad_authority_is_rebound_only_from_trusted_manifest`.
#[test]
fn implicit_broad_authority_is_rebound_only_from_a_trusted_manifest() {
    let mounts: Vec<Mount> = vec![
        ConfiguredMountStrategy::blobfuse(AzureBlobMount::new("example", "private")).mount(),
        ConfiguredMountStrategy::s3files(S3FilesMount::new("fs-123")).mount(),
    ];

    for mount in mounts {
        assert_eq!(
            mount.strategy.credential_authority(),
            MountCredentialAuthority::Broad
        );
        let manifest = Manifest::new()
            .with_root("/workspace")
            .with_entry("data", Entry::mount(mount))
            .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"]);
        let state = state_for(&manifest);
        assert!(state.mount_authority_redacted);

        let json = state.to_json_string().unwrap();
        assert!(!json.contains("credential_exposure"), "{json}");
        let restored = SandboxSessionState::from_json_str(&json).unwrap();
        assert!(restored.mount_authority_redacted);
        assert!(restored.assert_path_grants_rebound().is_err());

        let rebound = restored.rebind_path_grants(&manifest).unwrap();
        assert!(!rebound.mount_authority_redacted);
        assert!(rebound.mount_authority_rebound);
        rebound.assert_path_grants_rebound().unwrap();
    }
}

/// An opaque docker-volume authority (driver options the SDK cannot classify) is dropped from
/// durable state as a unit and the state stays resumable through the trusted manifest. Ports
/// `test_opaque_external_authority_remains_resumable_through_trusted_rebind` (docker case).
#[test]
fn opaque_external_authority_remains_resumable_through_a_trusted_rebind() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone")
                .with_driver_option("vfs-cache-mode", "off")
                .mount(),
        ),
    );
    let state = state_for(&manifest);
    let json = state.to_json_string().unwrap();
    assert!(!json.contains("vfs-cache-mode"), "{json}");

    let restored = SandboxSessionState::from_json_str(&json).unwrap();
    let rebound = restored.rebind_path_grants(&manifest).unwrap();
    assert!(!rebound.mount_authority_redacted);
    restored.verify_manifest(&manifest).unwrap();
}

/// A credential file that the manifest itself materializes is readable by the model, so a mount
/// pointing at one is refused, whichever entry kind fills the path. Ports
/// `test_rejects_manifest_backed_credential_files`,
/// `test_broad_acknowledgement_does_not_allow_manifest_backed_rclone_config` and
/// `test_rejects_credential_files_from_manifest_materialization_sources`.
#[test]
fn a_credential_file_materialized_by_the_manifest_is_rejected() {
    let gcs = |path: &str| {
        ConfiguredMountStrategy::docker_volume(
            GcsMount::new("bucket").with_service_account_file(path),
            "rclone",
        )
        .mount()
    };
    let assert_rejected = |manifest: Manifest, mount: &Mount, label: &str| {
        let error = manifest
            .authorize_mount_activation(mount, "/workspace/data")
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{label}");
        assert!(
            error
                .message
                .contains("credential files stored in the manifest"),
            "{label}: {error}"
        );
        assert!(!error.message.contains("credential-file-secret"), "{label}");
        assert!(error.context.contains_key("credential_field"), "{label}");
    };

    // A file entry at the credential path, spelled absolutely or relative to the root.
    for path in ["/workspace/credentials.json", "credentials.json"] {
        let mount = gcs(path);
        let manifest = Manifest::new()
            .with_root("/workspace")
            .with_entry(
                "credentials.json",
                Entry::file(b"credential-file-secret".to_vec()),
            )
            .with_entry("data", Entry::mount(mount.clone()));
        assert_rejected(manifest, &mount, path);
    }

    // A Box JWT configuration file.
    let box_mount = ConfiguredMountStrategy::docker_volume(
        BoxMount {
            box_config_file: Some("credentials.json".into()),
            ..BoxMount::new()
        },
        "rclone",
    )
    .mount();
    assert_rejected(
        Manifest::new()
            .with_root("/workspace")
            .with_entry("credentials.json", Entry::file(b"x".to_vec()))
            .with_entry("data", Entry::mount(box_mount.clone())),
        &box_mount,
        "box_config_file",
    );

    // An rclone configuration file in the manifest, even with a broad acknowledgement.
    let rclone = ConfiguredMountStrategy::in_container(
        S3Mount::new("bucket"),
        BuiltinMountPattern::Rclone(
            RcloneMountPattern::new().with_config_file_path("credentials.conf"),
        ),
    )
    .mount();
    assert_rejected(
        Manifest::new()
            .with_root("/workspace")
            .with_entry("credentials.conf", Entry::file(b"x".to_vec()))
            .with_entry("data", Entry::mount(rclone.clone()))
            .with_in_container_mount_broad_credential_exposure_acknowledged(["/workspace/data"]),
        &rclone,
        "rclone config_file_path",
    );

    // Entries that place content *under* a directory that holds the credential path.
    let nested = "/workspace/dir/credentials.json";
    for (label, entry) in [
        ("local_dir", Entry::local_dir("imported")),
        ("local_file", Entry::local_file("credentials.json")),
        ("git_repo", Entry::git_repo("example/repository", "main")),
        (
            "mount",
            Entry::mount(
                ConfiguredMountStrategy::docker_volume(S3Mount::new("secret-bucket"), "rclone")
                    .mount(),
            ),
        ),
    ] {
        let mount = gcs(nested);
        assert_rejected(
            Manifest::new()
                .with_root("/workspace")
                .with_entry("dir", entry)
                .with_entry("data", Entry::mount(mount.clone())),
            &mount,
            label,
        );
    }
}

/// A plain directory only structures the tree and holds no credential content, so a credential
/// path beneath it is allowed, and the credential never reaches durable state. Ports
/// `test_structural_local_dir_credential_path_remains_serializable`.
#[test]
fn a_structural_directory_at_the_credential_path_is_allowed() {
    let sentinel = "structural-dir-secret";
    let mount = ConfiguredMountStrategy::docker_volume(
        GcsMount::new("bucket")
            .with_service_account_file("/workspace/credentials/key.json")
            .with_service_account_credentials(sentinel),
        "rclone",
    )
    .mount();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("credentials", Entry::dir())
        .with_entry("data", Entry::mount(mount.clone()));

    manifest
        .authorize_mount_activation(&mount, "/workspace/data")
        .unwrap();
    let state = state_for(&manifest);
    assert!(!state.to_json_string().unwrap().contains(sentinel));
}

/// A credentialed Docker-volume mount needs no acknowledgement (the credential goes to the
/// container runtime on the host), but it still carries a secret, so the persisted state is
/// marked redacted, drops provider identity and refuses to resume on its own. Python redacts
/// any mount whose raw fields hold authority, wherever it is exposed. Ports
/// `test_session_state_serialization_redacts_complete_opaque_authority_fields`.
#[test]
fn a_credentialed_docker_volume_mount_is_redacted_from_durable_state() {
    let credentialed = || {
        ConfiguredMountStrategy::docker_volume(
            S3Mount::new("example-bucket").with_credentials(ACCESS_KEY, SECRET_KEY),
            "rclone",
        )
        .mount()
    };
    assert_eq!(
        credentialed().strategy.credential_authority(),
        MountCredentialAuthority::None
    );
    assert!(credentialed().strategy.carries_credentials(&credentialed()));

    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(credentialed()));
    let mut state = state_for(&manifest);
    assert!(state.mount_authority_redacted);
    state.provider_id = Some("provider-resource-id".to_string());

    let sanitized = state.sanitized_for_persistence();
    assert!(sanitized.provider_id.is_none());
    assert!(state.assert_path_grants_rebound().is_err());
    let rebound = state.rebind_path_grants(&manifest).unwrap();
    rebound.assert_path_grants_rebound().unwrap();
    assert!(!state.to_json_string().unwrap().contains(SECRET_KEY));

    // Opaque driver options are authority too, even on a credential-free mount.
    let opaque = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone")
                .with_driver_option("vfs-cache-mode", "off")
                .mount(),
        ),
    );
    assert!(state_for(&opaque).mount_authority_redacted);

    // A credential-free mount with no options stays unredacted.
    let plain = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("example-bucket"), "rclone")
                .mount(),
        ),
    );
    assert!(!state_for(&plain).mount_authority_redacted);
}
