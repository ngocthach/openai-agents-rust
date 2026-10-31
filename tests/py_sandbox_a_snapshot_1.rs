//! Ports `tests/sandbox/test_snapshot_defaults.py` and
//! `tests/sandbox/test_session_state_roundtrip.py` from the Python SDK.
//!
//! The defaults half pins the platform rules for the SDK-managed snapshot directory; the
//! round-trip half pins what a persisted [`SandboxSessionState`] carries, what it refuses, and
//! what a resume has to be handed again before it is allowed to proceed.
#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mounts::{BindMountStrategy, Mount};
use openai_agents_sdk::sandbox::snapshot::{
    DEFAULT_LOCAL_SNAPSHOT_TTL, SnapshotSpec, cleanup_stale_default_local_snapshots,
    default_local_snapshot_base_dir, default_local_snapshot_base_dir_for,
    resolve_default_local_snapshot_spec_at,
};
use openai_agents_sdk::sandbox::state::{
    SESSION_STATE_SCHEMA_VERSION, SandboxSessionState, SnapshotFingerprint, manifest_digest,
};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{ErrorCode, Manifest};

/// The subdirectory every default snapshot path ends with. Python spells the first segment
/// `openai-agents-python`; the Rust crate names itself instead, which is the only difference.
const SUBDIR: [&str; 3] = ["openai-agents-rust", "sandbox", "snapshots"];

fn suffix(base: &Path) -> PathBuf {
    SUBDIR
        .iter()
        .fold(base.to_path_buf(), |acc, part| acc.join(part))
}

/// An `env` lookup over a fixed map, standing in for Python's `env=` mapping argument.
fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
    let map: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |name: &str| map.get(name).cloned()
}

// ----------------------------------------------------------------------------------------------
// test_snapshot_defaults.py
// ----------------------------------------------------------------------------------------------

/// `XDG_STATE_HOME` wins over the home directory when it is absolute.
#[test]
fn default_local_snapshot_base_dir_uses_xdg_state_home() {
    let home = PathBuf::from("/home/someone");
    let result = default_local_snapshot_base_dir_for(
        &home,
        &env_of(&[("XDG_STATE_HOME", "/state")]),
        "linux",
        "posix",
    );
    assert_eq!(result, suffix(Path::new("/state")));
}

/// An explicitly empty env is honored: the ambient `XDG_STATE_HOME` of the running process must
/// not leak into the resolved path.
#[test]
fn default_local_snapshot_base_dir_honors_explicit_empty_env() {
    let home = PathBuf::from("/home/someone");
    let result = default_local_snapshot_base_dir_for(&home, &env_of(&[]), "linux", "posix");
    assert_eq!(result, suffix(&home.join(".local").join("state")));
}

/// A relative `XDG_STATE_HOME` is not a state directory, so the home fallback is used.
#[test]
fn default_local_snapshot_base_dir_ignores_relative_xdg_state_home() {
    let home = PathBuf::from("/home/someone");
    let result = default_local_snapshot_base_dir_for(
        &home,
        &env_of(&[("XDG_STATE_HOME", "relative-state")]),
        "linux",
        "posix",
    );
    assert_eq!(result, suffix(&home.join(".local").join("state")));
}

/// macOS ignores the environment entirely and uses Application Support.
#[test]
fn default_local_snapshot_base_dir_uses_macos_application_support() {
    let home = PathBuf::from("/Users/someone");
    let result = default_local_snapshot_base_dir_for(
        &home,
        &env_of(&[("XDG_STATE_HOME", "/state")]),
        "darwin",
        "posix",
    );
    assert_eq!(
        result,
        suffix(&home.join("Library").join("Application Support"))
    );
}

/// Windows prefers `LOCALAPPDATA`.
#[test]
fn default_local_snapshot_base_dir_uses_localappdata_on_windows() {
    let local_app_data = r"C:\Users\me\AppData\Local";
    let result = default_local_snapshot_base_dir_for(
        Path::new("/home/someone"),
        &env_of(&[("LOCALAPPDATA", local_app_data)]),
        "win32",
        "nt",
    );
    assert_eq!(result, suffix(Path::new(local_app_data)));
}

/// A relative `LOCALAPPDATA` falls through to an absolute `APPDATA`.
#[test]
fn default_local_snapshot_base_dir_uses_absolute_appdata_when_localappdata_is_relative() {
    let app_data = r"C:\Users\me\AppData\Roaming";
    let result = default_local_snapshot_base_dir_for(
        Path::new("/home/someone"),
        &env_of(&[("LOCALAPPDATA", "relative-local"), ("APPDATA", app_data)]),
        "win32",
        "nt",
    );
    assert_eq!(result, suffix(Path::new(app_data)));
}

/// Two relative Windows env paths leave only the home fallback.
#[test]
fn default_local_snapshot_base_dir_ignores_relative_windows_env_paths() {
    let home = PathBuf::from(r"C:\Users\me");
    let result = default_local_snapshot_base_dir_for(
        &home,
        &env_of(&[
            ("LOCALAPPDATA", "relative-local"),
            ("APPDATA", "relative-roaming"),
        ]),
        "win32",
        "nt",
    );
    assert_eq!(result, suffix(&home.join("AppData").join("Local")));
}

/// A POSIX-absolute value is not absolute on Windows, so it is ignored there. This is the case
/// the host's own `Path::is_absolute` would get wrong when the tests run on Unix.
#[test]
fn default_local_snapshot_base_dir_ignores_posix_absolute_localappdata_on_windows() {
    let home = PathBuf::from(r"C:\Users\me");
    let result = default_local_snapshot_base_dir_for(
        &home,
        &env_of(&[("LOCALAPPDATA", "/tmp/localappdata")]),
        "win32",
        "nt",
    );
    assert_eq!(result, suffix(&home.join("AppData").join("Local")));
}

/// The zero-argument resolver still agrees with the parameterized one on the running host.
#[test]
fn default_local_snapshot_base_dir_is_absolute_and_ends_with_the_managed_subdir() {
    let base = default_local_snapshot_base_dir();
    assert!(base.is_absolute(), "{}", base.display());
    assert!(base.ends_with(suffix(Path::new(""))), "{}", base.display());
}

/// Cleanup removes stale `*.tar` files and leaves fresh ones and non-tar files alone.
#[tokio::test]
async fn cleanup_stale_default_local_snapshots_removes_only_old_tar_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let managed = dir.path().join("snapshots");
    std::fs::create_dir(&managed).expect("mkdir");
    let stale = managed.join("stale.tar");
    let fresh = managed.join("fresh.tar");
    let keep = managed.join("keep.txt");
    std::fs::write(&stale, b"stale").expect("write stale");
    std::fs::write(&fresh, b"fresh").expect("write fresh");
    std::fs::write(&keep, b"keep").expect("write keep");

    set_mtime(&stale, DEFAULT_LOCAL_SNAPSHOT_TTL + Duration::from_secs(60));
    set_mtime(&fresh, Duration::from_secs(60));

    cleanup_stale_default_local_snapshots(&managed, DEFAULT_LOCAL_SNAPSHOT_TTL).await;

    assert!(!stale.exists());
    assert!(fresh.exists());
    assert!(keep.exists());
}

/// Resolving the managed directory creates it without touching the snapshots already in it.
#[tokio::test]
async fn resolve_default_local_snapshot_spec_keeps_existing_stale_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let managed = suffix(&dir.path().join("home").join(".local").join("state"));
    std::fs::create_dir_all(&managed).expect("mkdir");
    let stale = managed.join("stale.tar");
    std::fs::write(&stale, b"stale").expect("write stale");
    set_mtime(&stale, DEFAULT_LOCAL_SNAPSHOT_TTL + Duration::from_secs(60));

    let spec = resolve_default_local_snapshot_spec_at(managed.clone())
        .await
        .expect("resolve spec");

    assert_eq!(
        spec,
        SnapshotSpec::Local {
            base_path: managed.clone()
        }
    );
    assert!(managed.exists());
    assert!(stale.exists());
}

/// The managed directory is created private: it holds whole workspace archives.
#[tokio::test]
async fn resolve_default_local_snapshot_spec_creates_a_private_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let managed = suffix(&dir.path().join("state"));

    resolve_default_local_snapshot_spec_at(managed.clone())
        .await
        .expect("resolve spec");

    assert!(managed.is_dir());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&managed)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "{mode:o}");
    }
}

fn set_mtime(path: &Path, age: Duration) {
    let when = SystemTime::now()
        .checked_sub(age)
        .expect("representable mtime");
    std::fs::File::options()
        .write(true)
        .open(path)
        .expect("open for set_modified")
        .set_modified(when)
        .expect("set_modified");
}

// ----------------------------------------------------------------------------------------------
// test_session_state_roundtrip.py
// ----------------------------------------------------------------------------------------------

fn state_for(manifest: &Manifest) -> SandboxSessionState {
    SandboxSessionState::new("roundtrip", "sid-1", manifest.root(), manifest)
}

/// A state survives JSON and comes back as the same value, with every field it was given.
/// Python's equivalent proves the subclass survives; Rust has one concrete type, so the contract
/// is the field set plus the exact serialized key spelling.
#[test]
fn session_state_round_trips_through_json_with_every_field() {
    let manifest = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));
    let state = state_for(&manifest)
        .with_provider_id(Some("container-1".into()))
        .with_rollout_id(Some("rollout-7".into()))
        .with_workspace_root_owned(true)
        .with_workspace_root_ready(true)
        .with_exposed_ports([8080, 9000])
        .with_provider_metadata("labels", serde_json::json!({"a": "b"}))
        .with_snapshot_fingerprint(Some(SnapshotFingerprint {
            fingerprint: "deadbeef".into(),
            version: "workspace_tar_sha256_v1".into(),
        }))
        .with_snapshot(SnapshotSpec::Local {
            base_path: "/tmp/snapshots".into(),
        });

    let json = state.to_json_string().expect("serialize");
    let restored = SandboxSessionState::from_json_str(&json).expect("parse");

    assert_eq!(restored, state);
    assert_eq!(restored.snapshot_fingerprint.as_deref(), Some("deadbeef"));
    assert_eq!(
        restored.snapshot_fingerprint_version.as_deref(),
        Some("workspace_tar_sha256_v1")
    );
    assert_eq!(restored.provider_id.as_deref(), Some("container-1"));
    assert_eq!(restored.rollout_id.as_deref(), Some("rollout-7"));
}

/// The serialized document uses exactly these keys. A rename here breaks every state written by
/// an older build, so the spelling is the contract.
#[test]
fn session_state_json_uses_the_documented_keys() {
    let manifest = Manifest::new();
    let state = state_for(&manifest)
        .with_provider_id(Some("container-1".into()))
        .with_rollout_id(Some("rollout-7".into()))
        .with_snapshot(SnapshotSpec::Local {
            base_path: "/tmp/snapshots".into(),
        });
    let value: serde_json::Value =
        serde_json::from_str(&state.to_json_string().expect("serialize")).expect("json");
    let object = value.as_object().expect("object");

    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "$schemaVersion",
            "backend",
            "exposed_ports",
            "manifest_digest",
            "mount_authority_rebound",
            "mount_authority_redacted",
            "path_grants_require_rebind",
            "provider_id",
            "rollout_id",
            "session_id",
            "snapshot",
            "snapshot_id",
            "workspace_root",
            "workspace_root_owned",
            "workspace_root_ready",
        ]
    );
    assert_eq!(object["$schemaVersion"], SESSION_STATE_SCHEMA_VERSION);
    assert_eq!(object["backend"], "roundtrip");
    // The snapshot location travels as its own tagged document, the way Python's snapshot
    // subclass keeps its `type` discriminator and its own fields.
    assert_eq!(
        object["snapshot"],
        serde_json::json!({"type": "local", "base_path": "/tmp/snapshots"})
    );
}

/// A `remote` snapshot survives the state round trip, carrying the key its client is registered
/// under and nothing else. Python names the field `client_dependency_key`.
#[test]
fn remote_snapshot_serializes_through_session_state_without_its_client() {
    let manifest = Manifest::new();
    let state = state_for(&manifest).with_snapshot(SnapshotSpec::Remote {
        client_key: "tests.remote_snapshot_client".into(),
    });
    let value: serde_json::Value =
        serde_json::from_str(&state.to_json_string().expect("serialize")).expect("json");

    assert_eq!(
        value["snapshot"],
        serde_json::json!({"type": "remote", "client_key": "tests.remote_snapshot_client"})
    );
    let restored = SandboxSessionState::from_json_str(&value.to_string()).expect("parse");
    assert_eq!(
        restored.snapshot,
        SnapshotSpec::Remote {
            client_key: "tests.remote_snapshot_client".into()
        }
    );
}

/// Payloads that are not a session state are refused, and the error says so rather than
/// producing a half-built state.
#[test]
fn parse_rejects_invalid_payloads() {
    for payload in [
        "{}",
        r#"{"$schemaVersion":"missing"}"#,
        "\"not-a-state\"",
        "not json at all",
    ] {
        let error = SandboxSessionState::from_json_str(payload)
            .expect_err(&format!("expected a rejection for {payload}"));
        assert_eq!(error.code, ErrorCode::SessionStateInvalid, "{payload}");
        assert!(
            error.message.contains("sandbox session state"),
            "{}",
            error.message
        );
    }
}

/// Exposed ports are normalized: duplicates collapse and the first occurrence keeps its place.
#[test]
fn exposed_ports_are_normalized() {
    let manifest = Manifest::new();
    let cases: [(&[u16], &[u16]); 3] = [
        (&[], &[]),
        (&[8080], &[8080]),
        (&[8080, 9000, 8080], &[8080, 9000]),
    ];
    for (raw, expected) in cases {
        let state = state_for(&manifest).with_exposed_ports(raw.iter().copied());
        assert_eq!(state.exposed_ports, expected, "{raw:?}");
    }
}

/// A persisted state claiming port 0 is malformed. Ports above 65535 and non-integer ports
/// cannot be spelled in the Rust type, so they are rejected by the parser as type errors.
#[test]
fn exposed_ports_reject_invalid_values() {
    let manifest = Manifest::new();
    let mut value: serde_json::Value =
        serde_json::from_str(&state_for(&manifest).to_json_string().expect("serialize"))
            .expect("json");

    for (ports, expected_fragment) in [
        (serde_json::json!([0]), "between 1 and 65535"),
        (serde_json::json!([65536]), "sandbox session state"),
        (serde_json::json!([8080, "9000"]), "sandbox session state"),
        (serde_json::json!("8080"), "sandbox session state"),
    ] {
        value["exposed_ports"] = ports.clone();
        let error = SandboxSessionState::from_json_str(&value.to_string())
            .expect_err(&format!("expected a rejection for {ports}"));
        assert_eq!(error.code, ErrorCode::SessionStateInvalid, "{ports}");
        assert!(
            error.message.contains(expected_fragment),
            "{ports}: {}",
            error.message
        );
    }
}

/// A grant backed by a host directory is redacted on the way out, and the state records which
/// sandbox-side paths lost their host side so the caller knows what to supply again.
#[test]
fn serialization_redacts_host_paths_and_rebinds_from_the_trusted_manifest() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let host_path = host_dir.path().to_string_lossy().to_string();
    let trusted = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/mnt/shared-data")
            .expect("grant")
            .with_host_path(&host_path)
            .expect("host path")
            .read_only(true),
    );
    let state = state_for(&trusted);

    let encoded = state.to_json_string().expect("serialize");
    assert!(!encoded.contains(&host_path), "{encoded}");
    assert!(state.path_grants_require_rebind);
    assert_eq!(
        state.redacted_host_path_grant_paths,
        vec!["/mnt/shared-data"]
    );

    // Until the caller supplies the host side again, a resume is refused.
    let error = state
        .assert_path_grants_rebound()
        .expect_err("resume must be refused");
    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    assert!(
        error.message.contains("must be rebound"),
        "{}",
        error.message
    );

    let rebound = state.rebind_path_grants(&trusted).expect("rebind");
    assert!(!rebound.path_grants_require_rebind);
    assert!(rebound.redacted_host_path_grant_paths.is_empty());
    rebound
        .assert_path_grants_rebound()
        .expect("resume allowed");
    // Rebinding does not mutate the state it was called on.
    assert!(state.path_grants_require_rebind);
}

/// A trusted manifest that no longer carries the host path the state recorded cannot rebind it,
/// and the error names the path that is missing.
#[test]
fn rebinding_refuses_a_manifest_missing_the_recorded_grant() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let trusted = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/mnt/shared-data")
            .expect("grant")
            .with_host_path(host_dir.path().to_string_lossy().as_ref())
            .expect("host path"),
    );
    let state = state_for(&trusted);

    let error = state
        .rebind_path_grants(&Manifest::new())
        .expect_err("rebind must be refused");
    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    assert!(
        error.message.contains("/mnt/shared-data"),
        "{}",
        error.message
    );
}

/// A grant that names only a path inside the sandbox carries no host authority, so a state
/// written from it resumes directly without anything being re-supplied.
#[test]
fn path_only_grants_preserve_a_direct_resume_round_trip() {
    let manifest = Manifest::new()
        .with_extra_path_grant(
            SandboxPathGrant::new("/mnt/shared-data")
                .expect("grant")
                .read_only(true),
        )
        .with_extra_path_grant(
            SandboxPathGrant::new("/mnt/shared-data")
                .expect("grant")
                .read_only(false),
        );
    let state = state_for(&manifest);

    assert!(!state.path_grants_require_rebind);
    assert!(state.redacted_host_path_grant_paths.is_empty());
    let restored = SandboxSessionState::from_json_str(&state.to_json_string().expect("serialize"))
        .expect("parse");
    assert!(!restored.path_grants_require_rebind);
    restored
        .assert_path_grants_rebound()
        .expect("direct resume allowed");
}

/// A state whose live manifest held cloud mount credentials cannot be resumed on its own, and
/// the backend resource those credentials attached storage to is not persisted either.
#[test]
fn mount_authority_is_redacted_and_blocks_a_bare_resume() {
    let manifest = Manifest::new().with_entry("m", Entry::mount(credential_bearing_mount()));
    let state = state_for(&manifest).with_provider_id(Some("container-1".into()));
    assert!(
        state.mount_authority_redacted,
        "fixture must bear credentials"
    );

    let error = state
        .assert_path_grants_rebound()
        .expect_err("bare resume must be refused");
    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    assert!(
        error.message.contains("cloud mount credentials"),
        "{}",
        error.message
    );

    let persisted = SandboxSessionState::from_json_str(&state.to_json_string().expect("serialize"))
        .expect("parse");
    assert_eq!(persisted.provider_id, None);
    assert_ne!(persisted.session_id, state.session_id);
    assert!(!persisted.workspace_root_ready);
    // The derived id is stable, so two writes of the same state name the same snapshot.
    let twice = SandboxSessionState::from_json_str(&state.to_json_string().expect("serialize"))
        .expect("parse");
    assert_eq!(persisted.session_id, twice.session_id);

    let rebound = state.rebind_path_grants(&manifest).expect("rebind");
    assert!(!rebound.mount_authority_redacted);
    assert!(rebound.mount_authority_rebound);
    rebound
        .assert_path_grants_rebound()
        .expect("resume allowed");
}

/// Nothing a mount or an environment value holds reaches the persisted document.
#[test]
fn state_json_never_contains_credentials_or_host_paths() {
    let manifest = Manifest::new()
        .with_entry(
            "m",
            Entry::mount(BindMountStrategy::mount("/host/secret-dir")),
        )
        .with_env("AWS_SECRET_ACCESS_KEY", "SUPERSECRET");
    let json = state_for(&manifest).to_json_string().expect("serialize");

    assert!(!json.contains("SUPERSECRET"), "{json}");
    assert!(!json.contains("secret-dir"), "{json}");
}

/// The manifest digest covers the topology a resume must agree on, and nothing else: the digest
/// is written to disk, so hashing contents, host sources or environment values would leak them.
#[test]
fn manifest_digest_covers_topology_but_not_contents() {
    let base = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));

    assert_eq!(
        manifest_digest(&base),
        manifest_digest(&Manifest::new().with_entry("a.txt", Entry::text_file("goodbye"))),
        "file contents must not change the digest"
    );
    assert_eq!(
        manifest_digest(&base),
        manifest_digest(&base.clone().with_root("/other")),
        "the backend-assigned root must not change the digest"
    );
    assert_eq!(
        manifest_digest(&base.clone().with_env("TOKEN", "secret-1")),
        manifest_digest(&base.clone().with_env("TOKEN", "secret-2")),
        "environment values must not change the digest"
    );
    assert_ne!(
        manifest_digest(&base),
        manifest_digest(&base.clone().with_env("TOKEN", "secret-1")),
        "environment names must change the digest"
    );
    assert_ne!(
        manifest_digest(&base),
        manifest_digest(&base.clone().with_entry("b.txt", Entry::text_file("x"))),
        "an added entry must change the digest"
    );
}

/// A resume is refused when the manifest it is handed is not the one the session was created
/// from.
#[test]
fn verify_manifest_refuses_a_different_topology() {
    let manifest = Manifest::new().with_entry("a.txt", Entry::text_file("hello"));
    let state = state_for(&manifest);
    state.verify_manifest(&manifest).expect("same manifest");

    let error = state
        .verify_manifest(
            &manifest
                .clone()
                .with_entry("evil", Entry::local_dir("/etc")),
        )
        .expect_err("a tampered manifest must be refused");
    assert_eq!(error.code, ErrorCode::SessionStateMismatch);
}

/// Teardown may only remove a workspace root that still looks like one the SDK created, whatever
/// the persisted flag claims.
#[test]
fn only_sdk_created_workspace_roots_may_be_removed() {
    let manifest = Manifest::new();
    let owned_root = std::env::temp_dir().join("openai-agents-sandbox-abc123");
    let state = SandboxSessionState::new(
        "roundtrip",
        "sid-1",
        owned_root.to_string_lossy().to_string(),
        &manifest,
    )
    .with_workspace_root_owned(true);
    assert!(state.may_remove_workspace_root());

    let tampered = SandboxSessionState {
        workspace_root: "/Users/someone".to_string(),
        ..state.clone()
    };
    assert!(!tampered.may_remove_workspace_root());

    let unowned = SandboxSessionState {
        workspace_root_owned: false,
        ..state
    };
    assert!(!unowned.may_remove_workspace_root());
}

/// A mount whose strategy carries credentials, used to prove the redaction path.
fn credential_bearing_mount() -> Mount {
    use openai_agents_sdk::sandbox::SandboxResult;
    use openai_agents_sdk::sandbox::mounts::{MountCredentialAuthority, MountStrategy};
    use openai_agents_sdk::sandbox::session::SandboxSession;

    #[derive(Debug)]
    struct CredentialStrategy;

    #[async_trait::async_trait]
    impl MountStrategy for CredentialStrategy {
        fn strategy_type(&self) -> &str {
            "credential-test"
        }
        fn credential_authority(&self) -> MountCredentialAuthority {
            MountCredentialAuthority::MountScoped
        }
        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> SandboxResult<()> {
            Ok(())
        }
        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> SandboxResult<()> {
            Ok(())
        }
    }

    Mount::new(std::sync::Arc::new(CredentialStrategy))
}

/// Removing the redaction marker from a persisted state never brings a host-backed grant back:
/// the state carries no grant to restore, so the resume has nothing host-side to act on.
#[test]
fn removed_redaction_marker_does_not_restore_a_host_backed_grant() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let host_path = host_dir.path().to_string_lossy().to_string();
    let trusted = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/mnt/shared-data")
            .expect("grant")
            .with_host_path(&host_path)
            .expect("host path"),
    );
    let mut value: serde_json::Value =
        serde_json::from_str(&state_for(&trusted).to_json_string().expect("serialize"))
            .expect("json");
    value
        .as_object_mut()
        .expect("object")
        .remove("redacted_host_path_grant_paths");

    let restored = SandboxSessionState::from_json_str(&value.to_string()).expect("parse");

    assert!(restored.redacted_host_path_grant_paths.is_empty());
    assert!(
        !restored
            .to_json_string()
            .expect("serialize")
            .contains(&host_path)
    );
    // Rebinding against an empty manifest has no recorded path to demand, so it adds nothing.
    let rebound = restored
        .rebind_path_grants(&Manifest::new())
        .expect("rebind");
    assert!(rebound.redacted_host_path_grant_paths.is_empty());
}

/// A host path smuggled into the payload under an unrecognized key is discarded on parse and
/// never serialized back out, and the resume still demands the trusted manifest.
#[test]
fn deserialization_discards_an_unmarked_serialized_host_path() {
    let host_dir = tempfile::tempdir().expect("tempdir");
    let host_path = host_dir.path().to_string_lossy().to_string();
    let trusted = Manifest::new().with_extra_path_grant(
        SandboxPathGrant::new("/mnt/shared-data")
            .expect("grant")
            .with_host_path(&host_path)
            .expect("host path"),
    );
    let mut value: serde_json::Value =
        serde_json::from_str(&state_for(&trusted).to_json_string().expect("serialize"))
            .expect("json");
    value["manifest"] = serde_json::json!({
        "extra_path_grants": [{"path": "/mnt/shared-data", "host_path": host_path}]
    });

    let restored = SandboxSessionState::from_json_str(&value.to_string()).expect("parse");

    assert!(
        !restored
            .to_json_string()
            .expect("serialize")
            .contains(&host_path)
    );
    let error = restored
        .assert_path_grants_rebound()
        .expect_err("must be rebound");
    assert!(
        error.message.contains("must be rebound"),
        "{}",
        error.message
    );
    let rebound = restored.rebind_path_grants(&trusted).expect("rebind");
    rebound
        .assert_path_grants_rebound()
        .expect("resume allowed");
}
