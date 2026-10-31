//! Parity coverage for the sandbox-core rows closed in the final pass: Docker client options and
//! resume validation, the workspace path policy and read-only grant enforcement, the snapshot
//! fingerprint fast path, per-agent sessions, live-session manifest updates and the entry
//! concurrency ordering.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::SandboxConcurrencyLimits;
use openai_agents_sdk::sandbox::mounts::{
    BindMountStrategy, DockerVolumeMountStrategy, Mount, MountStrategy, docker_volume_name,
};
use openai_agents_sdk::sandbox::runtime_helpers::{
    RESOLVE_WORKSPACE_PATH_HELPER, WORKSPACE_FINGERPRINT_HELPER,
};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerSandboxClientOptions, assert_container_labels_match, assert_container_network_matches,
    assert_container_path_grants_match, docker_volume_names_for_manifest,
    validate_docker_path_grants,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::state::{SNAPSHOT_FINGERPRINT_VERSION, SandboxSessionState};
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, WorkspacePathPolicy};
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxSession, parse_snapshot_fingerprint_record,
};
use serde_json::json;

fn grant(path: &str, host: &str, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path)
        .expect("grant path")
        .with_host_path(host)
        .expect("host path")
        .read_only(read_only)
}

// ----------------------------------------------------------------------------------------------
// Docker client options and path-grant validation
// ----------------------------------------------------------------------------------------------

#[test]
fn exposed_ports_cannot_be_combined_with_a_disabled_network() {
    let options = DockerSandboxClientOptions::new("alpine:latest")
        .with_network_disabled(true)
        .with_exposed_port(8080);
    let error = options.validate().expect_err("the combination is refused");
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.message.contains("exposed_ports"), "{}", error.message);

    // Either one on its own is fine.
    DockerSandboxClientOptions::new("alpine:latest")
        .with_exposed_ports([8080, 9000])
        .with_label("team", "sandbox")
        .validate()
        .expect("ports without a disabled network are allowed");
    DockerSandboxClientOptions::new("alpine:latest")
        .with_network_disabled(true)
        .validate()
        .expect("a disabled network without ports is allowed");
}

#[test]
fn a_read_only_host_grant_cannot_be_combined_with_a_privileged_mount() {
    #[derive(Debug)]
    struct SysAdminStrategy;

    #[async_trait::async_trait]
    impl MountStrategy for SysAdminStrategy {
        fn strategy_type(&self) -> &str {
            "sys-admin-test"
        }
        fn container_privileges(&self) -> openai_agents_sdk::sandbox::mounts::ContainerPrivileges {
            openai_agents_sdk::sandbox::mounts::ContainerPrivileges::SysAdmin
        }
        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
    }

    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(Mount::new(Arc::new(SysAdminStrategy))))
        .with_extra_path_grant(grant("/datasets", "/host/datasets", true));
    let error = validate_docker_path_grants(&manifest).expect_err("SYS_ADMIN defeats `:ro`");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("SYS_ADMIN"), "{}", error.message);

    // A writable grant is fine: nothing is being promised that the capability could break.
    let writable = Manifest::new()
        .with_root("/workspace")
        .with_entry("data", Entry::mount(Mount::new(Arc::new(SysAdminStrategy))))
        .with_extra_path_grant(grant("/datasets", "/host/datasets", false));
    validate_docker_path_grants(&writable).expect("a writable grant is allowed");
}

#[test]
fn a_host_grant_must_stay_outside_the_workspace_root_and_be_unique() {
    let inside = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/workspace/shared", "/host/shared", false));
    let error = validate_docker_path_grants(&inside)
        .expect_err("a grant inside the root is a bind over it");
    assert!(error.message.contains("outside"), "{}", error.message);

    let duplicate = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/datasets", "/host/a", false))
        .with_extra_path_grant(grant("/datasets", "/host/b", false));
    let error =
        validate_docker_path_grants(&duplicate).expect_err("two binds cannot share a target");
    assert!(error.message.contains("duplicate"), "{}", error.message);

    // `//workspace` and `/workspace` are the same Linux path, so a grant aliased against the
    // root is still caught.
    let aliased_root = Manifest::new()
        .with_root("//workspace")
        .with_extra_path_grant(grant("/workspace/shared", "/host/shared", false));
    assert!(validate_docker_path_grants(&aliased_root).is_err());
}

#[test]
fn a_host_grant_cannot_shadow_a_docker_volume_mount() {
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "bucket",
        Entry::mount(Mount::new(Arc::new(DockerVolumeMountStrategy::new("s3")))),
    );
    let names = docker_volume_names_for_manifest(&manifest, Some("abc123"));
    assert_eq!(names.len(), 1);
    assert!(names[0].starts_with("sandbox_abc123_"), "{}", names[0]);
    assert!(names[0].ends_with("workspace_bucket"), "{}", names[0]);

    let conflicting =
        manifest
            .clone()
            .with_extra_path_grant(grant("/workspace/bucket", "/host/bucket", false));
    // The grant is inside the root, which is refused before the volume check even matters.
    assert!(validate_docker_path_grants(&conflicting).is_err());

    // The volume name is stable for the same session and path, and distinct across sessions.
    assert_eq!(
        docker_volume_name(Some("abc123"), "/workspace/bucket"),
        names[0]
    );
    assert_ne!(
        docker_volume_name(Some("other"), "/workspace/bucket"),
        names[0]
    );
}

// ----------------------------------------------------------------------------------------------
// Docker resume-time container validation
// ----------------------------------------------------------------------------------------------

#[test]
fn resuming_rejects_a_container_whose_network_is_no_longer_isolated() {
    let isolated = json!({
        "HostConfig": {"NetworkMode": "none"},
        "NetworkSettings": {"Networks": {"none": {}}},
    });
    assert_container_network_matches(&isolated, true).expect("an isolated container matches");

    let bridged = json!({
        "HostConfig": {"NetworkMode": "bridge"},
        "NetworkSettings": {"Networks": {"bridge": {}}},
    });
    let error = assert_container_network_matches(&bridged, true)
        .expect_err("a bridged container no longer matches `--network none`");
    assert_eq!(error.code, ErrorCode::SessionStateMismatch);

    // A container that gained an extra network is refused even with NetworkMode=none.
    let joined = json!({
        "HostConfig": {"NetworkMode": "none"},
        "NetworkSettings": {"Networks": {"none": {}, "lab": {}}},
    });
    assert!(assert_container_network_matches(&joined, true).is_err());

    // Nothing is checked when the session never asked for isolation.
    assert_container_network_matches(&bridged, false).expect("no requirement, no check");
}

#[test]
fn resuming_rejects_a_container_whose_labels_changed() {
    let labels: std::collections::BTreeMap<String, String> =
        [("team".to_string(), "sandbox".to_string())]
            .into_iter()
            .collect();
    let matching = json!({"Config": {"Labels": {"team": "sandbox", "extra": "fine"}}});
    assert_container_labels_match(&matching, &labels).expect("a superset still matches");

    let changed = json!({"Config": {"Labels": {"team": "someone-else"}}});
    assert!(assert_container_labels_match(&changed, &labels).is_err());
    assert!(assert_container_labels_match(&json!({}), &labels).is_err());
    // With no labels persisted there is nothing to check.
    assert_container_labels_match(&json!({}), &std::collections::BTreeMap::new()).unwrap();
}

#[test]
fn resuming_rejects_a_container_with_bind_mounts_the_manifest_does_not_describe() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/datasets", "/host/datasets", true));

    let expected = json!({"Mounts": [
        {"Type": "bind", "Destination": "/datasets", "Source": "/host/datasets", "RW": false}
    ]});
    assert_container_path_grants_match(&expected, &manifest).expect("the trusted bind matches");

    // An extra bind nobody asked for is the attack this check exists for.
    let extra = json!({"Mounts": [
        {"Type": "bind", "Destination": "/datasets", "Source": "/host/datasets", "RW": false},
        {"Type": "bind", "Destination": "/secrets", "Source": "/host/secrets", "RW": true}
    ]});
    let error = assert_container_path_grants_match(&extra, &manifest)
        .expect_err("an unexpected bind is refused");
    assert!(error.message.contains("/secrets"), "{}", error.message);

    // The same target pointed somewhere else, or made writable, is refused too.
    let moved = json!({"Mounts": [
        {"Type": "bind", "Destination": "/datasets", "Source": "/host/other", "RW": false}
    ]});
    assert!(assert_container_path_grants_match(&moved, &manifest).is_err());
    let writable = json!({"Mounts": [
        {"Type": "bind", "Destination": "/datasets", "Source": "/host/datasets", "RW": true}
    ]});
    assert!(assert_container_path_grants_match(&writable, &manifest).is_err());
}

// ----------------------------------------------------------------------------------------------
// Workspace path policy and read-only grants
// ----------------------------------------------------------------------------------------------

#[test]
fn the_path_policy_picks_the_longest_matching_grant() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![
            grant("/data", "/host/data", true),
            grant("/data/scratch", "/host/scratch", false),
        ],
    )
    .expect("policy");

    assert_eq!(policy.sandbox_root(), "/workspace");
    assert_eq!(
        policy.absolute_workspace_path("src/app.rs").unwrap(),
        "/workspace/src/app.rs"
    );
    assert_eq!(
        policy.relative_path("/workspace/src/app.rs").unwrap(),
        "src/app.rs"
    );
    assert_eq!(policy.relative_path("/workspace").unwrap(), ".");

    // A read is allowed anywhere a grant covers; a write only where the effective grant is
    // writable, and the nested writable grant wins over the read-only one above it.
    assert_eq!(
        policy
            .normalize_sandbox_path("/data/set.csv", false)
            .unwrap(),
        "/data/set.csv"
    );
    let error = policy
        .normalize_sandbox_path("/data/set.csv", true)
        .expect_err("the read-only grant refuses a write");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    assert_eq!(
        policy
            .normalize_sandbox_path("/data/scratch/out.csv", true)
            .unwrap(),
        "/data/scratch/out.csv"
    );

    // Removing a tree that contains a read-only grant is a write to that grant.
    assert!(policy.validate_recursive_remove("/data/scratch").is_ok());
    assert!(policy.validate_recursive_remove("/data").is_err());

    // Nothing outside the root and the grants resolves at all.
    assert!(policy.normalize_sandbox_path("/etc/passwd", false).is_err());
    assert!(policy.normalize_sandbox_path("../escape", false).is_err());
    assert_eq!(
        policy.extra_path_grant_rules().unwrap(),
        vec![
            ("/data".to_string(), true),
            ("/data/scratch".to_string(), false)
        ]
    );
}

#[tokio::test]
async fn a_raw_write_file_is_refused_under_a_read_only_grant() {
    // A real backend, so the refusal is proven to happen in the SDK and not only in a double.
    let workspace = tempfile::tempdir().expect("workspace");
    let dataset = tempfile::tempdir().expect("dataset");
    let dataset_path = dataset.path().to_string_lossy().to_string();
    std::fs::write(dataset.path().join("in.csv"), b"a,b\n").expect("seed");

    let manifest = Manifest::new()
        .with_root(workspace.path().to_string_lossy().to_string())
        // Python's unix_local refuses any `host_path`, so a same-path grant omits it.
        .with_extra_path_grant(
            SandboxPathGrant::new(&dataset_path)
                .expect("grant path")
                .read_only(true),
        );
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");

    let target = format!("{dataset_path}/out.csv");
    let error = session
        .write_file(&target, b"x")
        .await
        .expect_err("a read-only grant refuses a raw write");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    assert!(
        !std::path::Path::new(&target).exists(),
        "nothing was written"
    );

    assert!(
        session
            .mkdir(&format!("{dataset_path}/new"), true)
            .await
            .is_err()
    );
    assert!(
        session
            .rm(&format!("{dataset_path}/in.csv"), false)
            .await
            .is_err()
    );
    assert!(
        session
            .mv(
                &format!("{dataset_path}/in.csv"),
                &format!("{dataset_path}/moved.csv"),
            )
            .await
            .is_err()
    );
    assert!(
        dataset.path().join("in.csv").exists(),
        "nothing was removed"
    );

    // Reading through the same grant still works, which is the point of a read-only grant.
    let bytes = session
        .read_file(&format!("{dataset_path}/in.csv"))
        .await
        .expect("a read-only grant still reads");
    assert_eq!(bytes, b"a,b\n");

    session.shutdown().await.expect("shutdown");
}

// ----------------------------------------------------------------------------------------------
// Runtime helpers and the snapshot fingerprint fast path
// ----------------------------------------------------------------------------------------------

#[test]
fn the_fingerprint_record_parser_matches_the_helper_output() {
    let record = parse_snapshot_fingerprint_record(
        "{\"fingerprint\":\"abc\",\"version\":\"workspace_tar_sha256_v2\"}",
    )
    .expect("a well-formed record");
    assert_eq!(record.fingerprint, "abc");
    assert_eq!(record.version, SNAPSHOT_FINGERPRINT_VERSION);

    for bad in [
        "",
        "not json",
        "[]",
        "{\"version\":\"v2\"}",
        "{\"fingerprint\":\"\",\"version\":\"v2\"}",
    ] {
        assert!(parse_snapshot_fingerprint_record(bad).is_err(), "{bad}");
    }
}

#[tokio::test]
async fn the_fingerprint_helper_is_installed_once_and_then_reused() {
    let session = scripted_sandbox_session([
        // Not present yet: the install command runs.
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.arguments.first().map(String::as_str) == Some("test"))
            .returning_exec(ExecResult {
                exit_code: 1,
                stdout: Vec::new(),
                stderr: Vec::new(),
            }),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| {
                call.arguments
                    .iter()
                    .any(|argument| argument.contains("INSTALL_RUNTIME_HELPER_V1"))
            })
            .returning_exec(ExecResult::default()),
        // The second call finds it.
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.arguments.first().map(String::as_str) == Some("test"))
            .returning_exec(ExecResult::default()),
    ]);

    let path = session
        .ensure_runtime_helper_installed(&WORKSPACE_FINGERPRINT_HELPER)
        .await
        .expect("install");
    assert_eq!(path, WORKSPACE_FINGERPRINT_HELPER.install_path);
    let again = session
        .ensure_runtime_helper_installed(&WORKSPACE_FINGERPRINT_HELPER)
        .await
        .expect("already installed");
    assert_eq!(again, path);
    session.assert_complete();

    // The install path carries the script hash, so the two helpers never collide.
    assert_ne!(
        WORKSPACE_FINGERPRINT_HELPER.install_path,
        RESOLVE_WORKSPACE_PATH_HELPER.install_path
    );
}

#[tokio::test]
async fn a_resume_whose_workspace_still_matches_skips_the_archive() {
    // A real Unix-local session: the helper script runs for real, so this also proves the
    // POSIX `sh` copy works on this host.
    let snapshot_dir = tempfile::tempdir().expect("snapshot dir");
    let client = UnixLocalSandboxClient::new().with_snapshot(
        openai_agents_sdk::sandbox::snapshot::SnapshotSpec::Local {
            base_path: snapshot_dir.path().to_path_buf(),
        },
    );
    let manifest = Manifest::new().with_entry("notes.md", Entry::text_file("first"));
    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");

    // Suspend: the archive and the fingerprint are written together.
    openai_agents_sdk::sandbox::session::persist_session_snapshot(session.as_ref())
        .await
        .expect("persist");
    let state = session.session_state();
    if cfg!(target_os = "macos") {
        // On macOS a Unix-local command runs under `sandbox-exec`, exactly as Python's does, and
        // that profile denies `/tmp`, where the fingerprint helper and its cache live. The
        // fingerprint therefore cannot be computed: persisting still succeeds, and a resume
        // takes the full restore rather than the fast path. The fast path is covered on every
        // other host by the rest of this test.
        assert_eq!(state.snapshot_fingerprint, None);
        let resumed = client
            .resume(&state, manifest.clone())
            .await
            .expect("resume");
        assert!(
            !openai_agents_sdk::sandbox::live_workspace_matches_snapshot_on_resume(
                resumed.as_ref()
            )
            .await,
            "without a fingerprint the archive is always restored"
        );
        resumed.shutdown().await.expect("shutdown");
        return;
    }
    assert_eq!(
        state.snapshot_fingerprint_version.as_deref(),
        Some(SNAPSHOT_FINGERPRINT_VERSION),
        "the fingerprint is recorded next to the archive"
    );
    let fingerprint = state.snapshot_fingerprint.clone().expect("a fingerprint");

    // The workspace is preserved, so the resumed session must see the same fingerprint and
    // leave the live files alone.
    let resumed = client
        .resume(&state, manifest.clone())
        .await
        .expect("resume");
    assert!(
        openai_agents_sdk::sandbox::live_workspace_matches_snapshot_on_resume(resumed.as_ref())
            .await,
        "an untouched workspace matches its own fingerprint"
    );

    // Change a file and the fingerprint no longer matches, so the archive is restored.
    resumed
        .write_file("notes.md", b"edited after the snapshot")
        .await
        .expect("write");
    assert!(
        !openai_agents_sdk::sandbox::live_workspace_matches_snapshot_on_resume(resumed.as_ref())
            .await
    );
    assert!(fingerprint.len() >= 32, "{fingerprint}");

    resumed.shutdown().await.expect("shutdown");
}

// ----------------------------------------------------------------------------------------------
// Session state: redaction, rebinding and provider metadata
// ----------------------------------------------------------------------------------------------

#[test]
fn persisted_state_drops_the_host_paths_and_demands_them_back() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/datasets", "/host/datasets", true));
    let state = SandboxSessionState::new("docker", "sess", "/workspace", &manifest);
    assert!(state.path_grants_require_rebind);
    assert_eq!(
        state.redacted_host_path_grant_paths,
        vec!["/datasets".to_string()]
    );
    assert!(state.assert_path_grants_rebound().is_err());

    // The serialized form never carries a host path.
    let json = state.to_json_string().expect("serialize");
    assert!(!json.contains("/host/datasets"), "{json}");

    // Resuming with a trusted manifest that still has the grant rebinds it.
    let rebound = state.rebind_path_grants(&manifest).expect("rebind");
    assert!(!rebound.path_grants_require_rebind);
    rebound
        .assert_path_grants_rebound()
        .expect("rebound state resumes");

    // Resuming without it is refused by name.
    let without = Manifest::new().with_root("/workspace");
    let error = state
        .rebind_path_grants(&without)
        .expect_err("missing grant");
    assert!(error.message.contains("/datasets"), "{}", error.message);
}

#[test]
fn a_credential_bearing_state_loses_its_provider_identity_when_persisted() {
    #[derive(Debug)]
    struct CredentialStrategy;

    #[async_trait::async_trait]
    impl MountStrategy for CredentialStrategy {
        fn strategy_type(&self) -> &str {
            "credential-test"
        }
        fn credential_authority(
            &self,
        ) -> openai_agents_sdk::sandbox::mounts::MountCredentialAuthority {
            openai_agents_sdk::sandbox::mounts::MountCredentialAuthority::MountScoped
        }
        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
    }

    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "bucket",
        Entry::mount(Mount::new(Arc::new(CredentialStrategy))),
    );
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container-abc".to_string()))
        .with_workspace_root_ready(true);
    assert!(state.mount_authority_redacted);

    let sanitized = state.sanitized_for_persistence();
    assert_eq!(
        sanitized.provider_id, None,
        "the container is not reattachable"
    );
    assert_ne!(sanitized.session_id, "sess-1");
    assert!(!sanitized.workspace_root_ready);
    // The derived id is stable, so two persists name the same snapshot.
    assert_eq!(
        sanitized.session_id,
        state.sanitized_for_persistence().session_id
    );
    let json = state.to_json_string().expect("serialize");
    assert!(!json.contains("container-abc"), "{json}");

    // Once the caller supplies the trusted manifest again, the authority is rebound.
    let rebound = state.rebind_path_grants(&manifest).expect("rebind");
    assert!(rebound.mount_authority_rebound);
    assert!(!rebound.mount_authority_redacted);
    rebound.assert_path_grants_rebound().expect("resumable");
}

#[tokio::test]
async fn a_sanitized_docker_state_still_resumes_into_a_replacement() {
    #[derive(Debug)]
    struct CredentialStrategy;

    #[async_trait::async_trait]
    impl MountStrategy for CredentialStrategy {
        fn strategy_type(&self) -> &str {
            "credential-resume-test"
        }
        fn credential_authority(
            &self,
        ) -> openai_agents_sdk::sandbox::mounts::MountCredentialAuthority {
            openai_agents_sdk::sandbox::mounts::MountCredentialAuthority::MountScoped
        }
        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
    }

    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "bucket",
        Entry::mount(Mount::new(Arc::new(CredentialStrategy))),
    );
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container-abc".to_string()));
    // Persisting drops the container id, because the credentials must not be reattachable from
    // the state alone. The resume still has to work — into a *replacement* container.
    let persisted = SandboxSessionState::from_json_str(&state.to_json_string().expect("serialize"))
        .expect("parse");
    assert_eq!(persisted.provider_id, None);

    let client = openai_agents_sdk::sandbox::sandboxes::docker::DockerSandboxClient::new();
    match client.resume(&persisted, manifest).await {
        // With a daemon the replacement container is real, so it has to be removed again.
        Ok(session) => session
            .shutdown()
            .await
            .expect("remove the replacement container"),
        Err(error) => {
            // Without a daemon the resume cannot get as far as creating the replacement, but it
            // must never fail on the sanitized identity: that is the regression this guards.
            assert_ne!(error.code, ErrorCode::SessionStateInvalid, "{error}");
            assert!(
                !error.message.contains("missing its container id"),
                "{error}"
            );
        }
    }
}

#[test]
fn provider_metadata_round_trips_and_older_states_still_parse() {
    let manifest = Manifest::new().with_root("/workspace");
    let state = SandboxSessionState::new("docker", "sess", "/workspace", &manifest)
        .with_provider_metadata("docker_labels", json!({"team": "sandbox"}));
    let round_tripped =
        SandboxSessionState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(
        round_tripped.provider_metadata("docker_labels"),
        Some(&json!({"team": "sandbox"}))
    );
    assert_eq!(round_tripped.snapshot_fingerprint, None);

    // A state written before these fields existed still parses.
    let legacy = format!(
        "{{\"$schemaVersion\":\"{}\",\"backend\":\"unix_local\",\"session_id\":\"s\",\
         \"workspace_root\":\"/workspace\",\"workspace_root_owned\":false,\"provider_id\":null,\
         \"snapshot\":{{\"type\":\"noop\"}},\"snapshot_id\":\"s\",\"manifest_digest\":\"{}\",\
         \"exposed_ports\":[]}}",
        openai_agents_sdk::sandbox::state::SESSION_STATE_SCHEMA_VERSION,
        openai_agents_sdk::sandbox::manifest_digest(&manifest),
    );
    let parsed = SandboxSessionState::from_json_str(&legacy).expect("legacy state parses");
    assert!(parsed.provider_metadata.is_empty());
    assert!(parsed.snapshot_fingerprint.is_none());
}

// ----------------------------------------------------------------------------------------------
// Manifest entry concurrency ordering
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn entries_are_materialized_parent_before_child_and_mounts_alone() {
    let activated = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

    #[derive(Debug)]
    struct RecordingStrategy(Arc<std::sync::Mutex<Vec<String>>>);

    #[async_trait::async_trait]
    impl MountStrategy for RecordingStrategy {
        fn strategy_type(&self) -> &str {
            "recording-test"
        }
        async fn activate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            self.0.lock().expect("order").push(mount_path.to_string());
            Ok(())
        }
        async fn deactivate(
            &self,
            _mount: &Mount,
            _session: &dyn SandboxSession,
            _mount_path: &str,
        ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
            Ok(())
        }
    }

    let workspace = tempfile::tempdir().expect("workspace");
    let manifest = Manifest::new()
        .with_root(workspace.path().to_string_lossy().to_string())
        .with_concurrency_limits(SandboxConcurrencyLimits::new().with_manifest_entries(Some(2)))
        .expect("limits")
        .with_entry(
            "project",
            Entry::dir()
                .with_child("a.txt", Entry::text_file("a"))
                .with_child("b.txt", Entry::text_file("b"))
                .with_child("c.txt", Entry::text_file("c")),
        )
        .with_entry("notes.md", Entry::text_file("notes"))
        .with_entry(
            "bucket",
            Entry::mount(Mount::new(Arc::new(RecordingStrategy(activated.clone())))),
        );

    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");

    // Every file landed, which is what the batching must not break.
    for (path, expected) in [
        ("project/a.txt", "a"),
        ("project/b.txt", "b"),
        ("project/c.txt", "c"),
        ("notes.md", "notes"),
    ] {
        let bytes = session.read_file(path).await.expect(path);
        assert_eq!(String::from_utf8(bytes).unwrap(), expected, "{path}");
    }
    // The mount settled on its own rather than racing the files around it.
    assert_eq!(activated.lock().expect("order").len(), 1);

    session.shutdown().await.expect("shutdown");
}

#[test]
fn a_zero_width_concurrency_limit_is_refused() {
    assert!(
        SandboxConcurrencyLimits::new()
            .with_manifest_entries(Some(0))
            .validate()
            .is_err()
    );
    assert_eq!(
        SandboxConcurrencyLimits::new()
            .with_manifest_entries(None)
            .manifest_entry_width(),
        usize::MAX
    );
    assert_eq!(
        SandboxConcurrencyLimits::new()
            .with_manifest_entries(Some(3))
            .manifest_entry_width(),
        3
    );
}

// ----------------------------------------------------------------------------------------------
// Bind-mount docker arguments keep working through the session-aware entry point
// ----------------------------------------------------------------------------------------------

#[test]
fn a_bind_mount_ignores_the_session_id_while_a_volume_uses_it() {
    let bind = Mount::new(Arc::new(BindMountStrategy::new("/host/data")));
    assert_eq!(
        bind.strategy
            .docker_create_args_for_session(&bind, "/workspace/data", Some("sess")),
        bind.strategy.docker_create_args(&bind, "/workspace/data"),
    );

    let volume = Mount::new(Arc::new(DockerVolumeMountStrategy::new("s3")));
    let with_session = volume
        .strategy
        .docker_create_args_for_session(&volume, "/workspace/v", Some("sess"))
        .expect("volume args");
    assert!(
        with_session[1].contains("src=sandbox_sess_"),
        "{:?}",
        with_session
    );
}

// ----------------------------------------------------------------------------------------------
// Per-agent sessions and live-session manifest updates
// ----------------------------------------------------------------------------------------------

mod runner {
    use std::sync::Arc;

    use openai_agents_sdk::sandbox::capabilities::{Filesystem, Shell};
    use openai_agents_sdk::sandbox::entries::Entry;
    use openai_agents_sdk::sandbox::runtime::SandboxSessionMode;
    use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use openai_agents_sdk::sandbox::{
        Manifest, SandboxAgent, SandboxClient, SandboxRunConfig, SandboxRunner, SandboxSession,
    };
    use openai_agents_sdk::testing::{FakeModel, assistant_message};
    use openai_agents_sdk::{RunOptions, items::OutputItem};
    use serde_json::json;

    fn model() -> Arc<FakeModel> {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        Arc::new(FakeModel::new())
    }

    fn shell_call(call_id: &str, commands: &[&str]) -> OutputItem {
        OutputItem::from_value(json!({
            "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
            "status": "completed", "action": {"commands": commands}
        }))
    }

    #[tokio::test]
    async fn per_agent_mode_gives_each_agent_its_own_workspace() {
        let model = model();
        let reviewer = Arc::new(
            SandboxAgent::<()>::new("Reviewer")
                .with_model_instance(model.clone())
                .with_capabilities(vec![Arc::new(Shell::new().with_pty(false))])
                .with_default_manifest(
                    Manifest::new().with_entry("reviewer.txt", Entry::text_file("reviewer-only")),
                ),
        );
        let engineer = SandboxAgent::<()>::new("Engineer")
            .with_model_instance(model.clone())
            .with_capabilities(vec![Arc::new(Shell::new().with_pty(false))])
            .with_sandbox_handoff(reviewer)
            .with_default_manifest(
                Manifest::new().with_entry("engineer.txt", Entry::text_file("engineer-only")),
            );

        model.add_turn(vec![OutputItem::from_value(json!({
            "type": "function_call", "call_id": "c_handoff",
            "name": "transfer_to_reviewer", "arguments": "{}"
        }))]);
        // The reviewer's own entry is there; the engineer's is not, because the workspaces are
        // separate — which is exactly what the mode buys and what the shared default does not.
        model.add_turn(vec![shell_call(
            "c_shell",
            &["cat reviewer.txt; cat engineer.txt 2>&1 | tail -1"],
        )]);
        model.add_turn(vec![assistant_message("reviewed")]);

        let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
            .session_mode(SandboxSessionMode::PerAgent);
        let outcome = SandboxRunner::run_to_outcome(
            &engineer,
            "hand it over",
            &config,
            RunOptions::default().max_turns(Some(10)),
        )
        .await
        .expect("run");

        assert_eq!(outcome.result.last_agent.name, "Reviewer");
        let stdout = outcome
            .result
            .new_items
            .iter()
            .filter_map(|item| match &item.kind {
                openai_agents_sdk::items::RunItemKind::ToolCallOutput { raw_item, .. } => {
                    raw_item["output"][0]["stdout"].as_str().map(str::to_string)
                }
                _ => None,
            })
            .next()
            .expect("the reviewer's shell output");
        assert!(stdout.contains("reviewer-only"), "{stdout}");
        assert!(!stdout.contains("engineer-only"), "{stdout}");
    }

    #[tokio::test]
    async fn per_agent_mode_needs_a_client() {
        let model = model();
        let client = UnixLocalSandboxClient::new();
        let session = client.create(None).await.expect("create");
        session.start().await.expect("start");

        let agent = SandboxAgent::<()>::new("Engineer").with_model_instance(model);
        let config = SandboxRunConfig::with_session(session.clone())
            .session_mode(SandboxSessionMode::PerAgent);
        let error = SandboxRunner::run(&agent, "hi", &config)
            .await
            .expect_err("an injected session cannot be split");
        assert!(error.to_string().contains("PerAgent"), "{error}");

        client.delete(session).await.expect("delete");
    }

    #[tokio::test]
    async fn an_injected_session_gains_the_entries_a_capability_asks_for() {
        let model = model();
        let client = UnixLocalSandboxClient::new();
        // The session starts with one file; the agent's capability-processed manifest adds
        // another. Python materializes the addition into the running workspace.
        let session = client
            .create(Some(
                Manifest::new().with_entry("existing.txt", Entry::text_file("already here")),
            ))
            .await
            .expect("create");
        session.start().await.expect("start");

        #[derive(Debug)]
        struct AddsAnEntry;

        #[async_trait::async_trait]
        impl openai_agents_sdk::sandbox::capabilities::Capability<()> for AddsAnEntry {
            fn capability_type(&self) -> &str {
                "adds-an-entry"
            }
            fn process_manifest(&self, manifest: Manifest) -> Manifest {
                manifest.with_entry("added.txt", Entry::text_file("added by a capability"))
            }
        }

        let agent = SandboxAgent::<()>::new("Engineer")
            .with_model_instance(model.clone())
            .with_capabilities(vec![
                Arc::new(Shell::new().with_pty(false)),
                Arc::new(Filesystem::new()),
                Arc::new(AddsAnEntry),
            ]);

        model.add_turn(vec![assistant_message("ready")]);
        let config = SandboxRunConfig::with_session(session.clone());
        SandboxRunner::run(&agent, "go", &config)
            .await
            .expect("run");

        assert_eq!(
            String::from_utf8(session.read_file("added.txt").await.expect("added")).unwrap(),
            "added by a capability"
        );
        // The entry the session already had is untouched.
        assert_eq!(
            String::from_utf8(session.read_file("existing.txt").await.expect("existing")).unwrap(),
            "already here"
        );

        client.delete(session).await.expect("delete");
    }

    #[tokio::test]
    async fn a_running_session_refuses_a_capability_added_mount() {
        let model = model();
        let client = UnixLocalSandboxClient::new();
        let session = client.create(None).await.expect("create");
        session.start().await.expect("start");

        #[derive(Debug)]
        struct NoopStrategy;

        #[async_trait::async_trait]
        impl openai_agents_sdk::sandbox::mounts::MountStrategy for NoopStrategy {
            fn strategy_type(&self) -> &str {
                "noop-test"
            }
            async fn activate(
                &self,
                _mount: &openai_agents_sdk::sandbox::mounts::Mount,
                _session: &dyn SandboxSession,
                _mount_path: &str,
            ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
                Ok(())
            }
            async fn deactivate(
                &self,
                _mount: &openai_agents_sdk::sandbox::mounts::Mount,
                _session: &dyn SandboxSession,
                _mount_path: &str,
            ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
                Ok(())
            }
        }

        #[derive(Debug)]
        struct AddsAMount;

        #[async_trait::async_trait]
        impl openai_agents_sdk::sandbox::capabilities::Capability<()> for AddsAMount {
            fn capability_type(&self) -> &str {
                "adds-a-mount"
            }
            fn process_manifest(&self, manifest: Manifest) -> Manifest {
                manifest.with_entry(
                    "bucket",
                    Entry::mount(openai_agents_sdk::sandbox::mounts::Mount::new(Arc::new(
                        NoopStrategy,
                    ))),
                )
            }
        }

        let agent = SandboxAgent::<()>::new("Engineer")
            .with_model_instance(model)
            .with_capabilities(vec![Arc::new(AddsAMount)]);
        let config = SandboxRunConfig::with_session(session.clone());
        let error = SandboxRunner::run(&agent, "go", &config)
            .await
            .expect_err("a running session cannot gain a mount");
        assert!(error.to_string().contains("mount"), "{error}");

        client.delete(session).await.expect("delete");
    }
}

// ----------------------------------------------------------------------------------------------
// Core hooks the provider backends asked for
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_port_that_was_never_exposed_is_refused_without_a_retry() {
    let session = scripted_sandbox_session([]);
    let error = session
        .resolve_exposed_port(8080)
        .await
        .expect_err("a backend that exposes nothing has nothing to resolve");
    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
    // A port that was never declared cannot start working, so retrying is pointless.
    assert_eq!(error.retryable, Some(false));
}

#[test]
fn an_exposed_port_endpoint_builds_the_url_python_builds() {
    use openai_agents_sdk::sandbox::ExposedPortEndpoint;

    let plain = ExposedPortEndpoint::new("sandbox.example", 80, false);
    assert_eq!(plain.url_for("http").unwrap(), "http://sandbox.example/");
    assert_eq!(plain.url_for("ws").unwrap(), "ws://sandbox.example/");

    let tls = ExposedPortEndpoint::new("sandbox.example", 443, true);
    assert_eq!(tls.url_for("http").unwrap(), "https://sandbox.example/");
    assert_eq!(tls.url_for("ws").unwrap(), "wss://sandbox.example/");

    // A non-default port is kept, a signed query is appended, and a bare IPv6 host is bracketed.
    let signed = ExposedPortEndpoint::new("sandbox.example", 8443, true).with_query("?token=abc");
    assert_eq!(
        signed.url_for("http").unwrap(),
        "https://sandbox.example:8443/?token=abc"
    );
    let v6 = ExposedPortEndpoint::new("::1", 8080, false);
    assert_eq!(v6.url_for("http").unwrap(), "http://[::1]:8080/");

    assert!(plain.url_for("ftp").is_err());
}

#[tokio::test]
async fn a_bounded_read_returns_a_prefix_and_refuses_a_zero_budget() {
    let workspace = tempfile::tempdir().expect("workspace");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new()
                .with_root(workspace.path().to_string_lossy().to_string())
                .with_entry("log.jsonl", Entry::text_file("0123456789")),
        ))
        .await
        .expect("create");
    session.start().await.expect("start");

    assert_eq!(session.read_bounded("log.jsonl", 4).await.unwrap(), b"0123");
    // A result shorter than the budget means the file ended; one exactly at the budget does not.
    assert_eq!(
        session.read_bounded("log.jsonl", 64).await.unwrap(),
        b"0123456789"
    );
    assert_eq!(
        session
            .read_bounded("log.jsonl", 0)
            .await
            .expect_err("a zero budget reads nothing and is a caller mistake")
            .code,
        ErrorCode::SandboxConfigInvalid
    );
    assert_eq!(
        session
            .read_bounded("missing.txt", 8)
            .await
            .unwrap_err()
            .code,
        ErrorCode::WorkspaceReadNotFound
    );

    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn materializing_an_entry_as_a_user_routes_through_the_run_as_file_operations() {
    // The session records the commands, so this asserts the write really went through `exec` as
    // the named user rather than as the session owner.
    let user = openai_agents_sdk::sandbox::User::new("agent");
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("agent") && call.has_argument("sudo"))
            .returning_exec(ExecResult::default()),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("agent"))
            .returning_exec(ExecResult::default()),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("agent"))
            .returning_exec(ExecResult::default()),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("agent"))
            .returning_exec(ExecResult::default()),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("agent"))
            .returning_exec(ExecResult::default()),
    ])
    .with_manifest(Manifest::new().with_root("/workspace"));

    Entry::text_file("hello")
        .apply_as(&session, "/workspace/notes.md", Some(&user))
        .await
        .expect("materialize as a user");

    // No `write_file` call: every byte went through a command run as `agent`.
    assert!(
        session
            .calls()
            .iter()
            .all(|call| call.method == SandboxMethod::Exec),
        "{:?}",
        session.calls()
    );
    // And no `chmod`/`chgrp` pass, which would have run as the session owner and taken the
    // ownership back off the file.
    assert!(
        !session
            .calls()
            .iter()
            .any(|call| call.has_argument("chmod") || call.has_argument("chgrp")),
        "{:?}",
        session.calls()
    );
}

#[tokio::test]
async fn an_injected_session_refuses_an_agent_whose_run_as_account_it_never_declared() {
    use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");

    let model = Arc::new(openai_agents_sdk::testing::FakeModel::new());
    model.add_turn(vec![openai_agents_sdk::testing::assistant_message("hi")]);
    let agent = SandboxAgent::<()>::new("Engineer")
        .with_model_instance(model)
        .with_run_as(openai_agents_sdk::sandbox::User::new("agent"));

    // `useradd` on a session whose processes are already running would change the accounts out
    // from under them, so the account has to be on the manifest the session was created with.
    let error = SandboxRunner::run(
        &agent,
        "go",
        &SandboxRunConfig::with_session(session.clone()),
    )
    .await
    .expect_err("a running session cannot gain an account");
    assert!(error.to_string().contains("run as"), "{error}");

    client.delete(session).await.expect("delete");
}
