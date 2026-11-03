//! Ports of the path-validation, read, resume-with-mounts and strategy round-trip groups of
//! `tests/extensions/sandbox/test_modal.py`.
//!
//! The helper is `tests/fixtures/fake_modal_helper_scripted.py` (it logs every request, and a rules
//! file overrides one operation) and, for the part of the protocol that lives in the bundled
//! `helper.py`, `tests/fixtures/modal_helper_driver.py`.
#![cfg(feature = "sandbox")]

use base64::Engine as _;
use openai_agents_sdk::extensions::sandbox::ProviderMountStrategy;
use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy, ModalSandboxClient,
    ModalSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::runtime_helpers::RESOLVE_WORKSPACE_PATH_HELPER;
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxClient};
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

    fn rules(&self, rules: Value) {
        std::fs::write(self.path("rules.json"), rules.to_string()).expect("write rules");
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

fn b64(text: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(text)
}

fn resolver_argv(fake: &FakeModal) -> Vec<Vec<String>> {
    let resolver = RESOLVE_WORKSPACE_PATH_HELPER.install_path.clone();
    fake.of_op("exec")
        .into_iter()
        .map(|request| {
            request["argv"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|part| part.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .filter(|argv| argv.first() == Some(&resolver))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Path validation inside the sandbox
// ---------------------------------------------------------------------------------------------

/// `test_modal_normalize_path_preserves_safe_leaf_symlink_path`: a path whose resolution succeeds
/// is read at the requested path, not at where a symlink leaf points, and the resolver is asked
/// with the root, the requested path and the read mode.
#[tokio::test]
async fn read_keeps_the_requested_path_when_the_resolver_accepts_it() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": format!("{} /workspace /workspace/", RESOLVE_WORKSPACE_PATH_HELPER.install_path),
        "response": {"exit_code": 0, "stdout": b64("/workspace/target.txt"), "stderr": ""},
    }]));
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");
    session
        .write_file("link.txt", b"payload")
        .await
        .expect("write");

    let data = session.read_file("link.txt").await;

    // The fake helper has the file under the requested path.
    assert_eq!(data.expect("read"), b"payload");
    assert_eq!(fake.of_op("read_file")[0]["path"], "/workspace/link.txt");
    let resolver = resolver_argv(&fake);
    assert_eq!(
        resolver.last().expect("resolver call")[1..],
        ["/workspace", "/workspace/link.txt", "0"]
    );
}

/// `test_modal_normalize_path_rejects_symlink_escape`: a resolver that reports an escape (exit
/// 111) fails the read before the file is touched.
#[tokio::test]
async fn read_rejects_a_symlink_escape() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": format!("{} /workspace /workspace/", RESOLVE_WORKSPACE_PATH_HELPER.install_path),
        "response": {"exit_code": 111, "stdout": "", "stderr": b64("workspace escape")},
    }]));
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let error = session
        .read_file("link/secret.txt")
        .await
        .expect_err("the link escapes");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(
        error.message.contains("must not escape root"),
        "{}",
        error.message
    );
    assert!(fake.of_op("read_file").is_empty());
}

/// `test_modal_normalize_path_rejects_windows_drive_absolute_paths`: a drive-letter path is not a
/// workspace-relative path, and is refused before any command reaches the sandbox.
#[tokio::test]
async fn a_windows_drive_path_is_refused_before_any_command() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let error = session
        .read_file("C:/tmp/link.txt")
        .await
        .expect_err("a drive path");

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(fake.of_op("exec").is_empty(), "nothing ran in the sandbox");
}

/// `test_modal_normalize_path_reinstalls_helper_after_runtime_replacement`: the resolver is
/// installed again when the sandbox no longer has it, and left alone when it does.
#[tokio::test]
async fn the_resolver_is_reinstalled_when_the_sandbox_lost_it() {
    let fake = FakeModal::new();
    // The first presence check fails (a replaced sandbox), later ones succeed.
    fake.rules(json!([{
        "op": "exec",
        "contains": format!("test -x {}", RESOLVE_WORKSPACE_PATH_HELPER.install_path),
        "once": true,
        "response": {"exit_code": 1, "stdout": "", "stderr": ""},
    }]));
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    session
        .write_file("link.txt", b"x")
        .await
        .expect("first call");
    let after_first = fake.of_op("exec").len();
    session
        .write_file("link.txt", b"x")
        .await
        .expect("second call");

    let install_marker = RESOLVE_WORKSPACE_PATH_HELPER.name;
    let installs = fake
        .of_op("exec")
        .into_iter()
        .filter(|request| {
            request["argv"].to_string().contains(install_marker) && request["argv"][0] == "sh"
        })
        .count();
    assert_eq!(
        installs, 1,
        "installed once, after the failed presence check"
    );
    // The second call: a presence check and the resolver, no install.
    assert_eq!(fake.of_op("exec").len() - after_first, 2);
}

// ---------------------------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------------------------

/// `test_modal_read_classifies_nonzero_cat_with_path_probe`, host half: a missing file and an
/// unreadable one are different errors.
#[tokio::test]
async fn read_distinguishes_a_missing_file_from_an_unreadable_one() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let missing = session.read_file("target.txt").await.expect_err("missing");
    assert_eq!(missing.code, ErrorCode::WorkspaceReadNotFound);

    fake.rules(json!([{
        "op": "read_file",
        "response": {"error": "cat: target.txt: Permission denied", "code": "read_failed"},
    }]));
    let unreadable = session
        .read_file("target.txt")
        .await
        .expect_err("unreadable");
    assert_ne!(unreadable.code, ErrorCode::WorkspaceReadNotFound);
}

fn run_driver(case: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = std::process::Command::new("python3")
        .arg(format!("{manifest}/tests/fixtures/modal_helper_driver.py"))
        .arg(format!("{manifest}/src/extensions/sandbox/modal/helper.py"))
        .arg(case)
        .output()
        .expect("run python3");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("driver output")
}

/// `test_modal_read_classifies_nonzero_cat_with_path_probe`, helper half: the bundled helper maps
/// a failed `cat` to a not-found or a read failure.
#[test]
fn helper_classifies_a_failed_cat() {
    let result = run_driver("read_file");

    assert_eq!(result["missing"]["code"], "not_found");
    assert_eq!(result["denied"]["code"], "read_failed");
    assert!(
        result["denied"]["error"]
            .as_str()
            .is_some_and(|text| text.contains("Permission denied")),
        "{result}"
    );
    assert_eq!(result["silent"]["code"], "read_failed");
    assert_eq!(result["silent"]["error"], "cat exited with 2");
}

// ---------------------------------------------------------------------------------------------
// Start, resume and persisted strategies
// ---------------------------------------------------------------------------------------------

/// `test_modal_start_ensures_sandbox_before_running_commands`: the sandbox exists once the client
/// created the session, and `start` does not create another.
#[tokio::test]
async fn start_does_not_create_a_second_sandbox() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");
    assert_eq!(fake.of_op("create").len(), 1);

    session.start().await.expect("start");

    assert_eq!(fake.of_op("create").len(), 1);
}

fn bucket_manifest() -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket"),
        )),
    )
}

/// `test_modal_resume_eagerly_reconnects_sandbox` and
/// `test_modal_resume_reconnects_deserialized_credentialless_external_mount`: a manifest with a
/// credential-free bucket mount resumes onto the running sandbox from a persisted state.
#[tokio::test]
async fn resume_reconnects_with_a_credentialless_bucket_mount() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(fake.options());
    let session = client
        .create(Some(bucket_manifest()))
        .await
        .expect("create");
    let persisted = serde_json::to_string(&session.session_state()).expect("serialize");
    let restored = serde_json::from_str(&persisted).expect("deserialize");

    let resumed = client
        .resume(&restored, bucket_manifest())
        .await
        .expect("resume");

    assert_eq!(fake.of_op("create").len(), 1, "no second sandbox");
    assert_eq!(fake.of_op("poll")[0]["sandbox_id"], "sb-1");
    assert_eq!(resumed.session_state().provider_id.as_deref(), Some("sb-1"));
}

/// `test_modal_cloud_bucket_mount_strategy_round_trips_through_manifest_parse` and its
/// `secret_name` / `secret_environment_name` variants: a strategy survives the provider registry.
#[test]
fn the_strategy_round_trips_through_the_provider_registry() {
    for config in [
        ModalCloudBucketMountConfig::s3("bucket"),
        ModalCloudBucketMountConfig::s3("bucket").with_secret_name("named-modal-secret"),
        ModalCloudBucketMountConfig::s3("bucket")
            .with_secret_name("named-modal-secret")
            .with_secret_environment_name("staging"),
    ] {
        let payload = serde_json::to_value(ProviderMountStrategy::ModalCloudBucket(
            ModalCloudBucketMountStrategy::new(config.clone()),
        ))
        .expect("serialize");
        assert_eq!(payload["type"], "modal_cloud_bucket");

        let parsed = ProviderMountStrategy::parse(&payload).expect("parse");

        let ProviderMountStrategy::ModalCloudBucket(strategy) = parsed else {
            panic!("a Modal strategy parses back to a Modal strategy");
        };
        assert_eq!(strategy.config().secret_name, config.secret_name);
        assert_eq!(
            strategy.config().secret_environment_name,
            config.secret_environment_name
        );
        assert_eq!(strategy.config().bucket_name, "bucket");
    }
}

// ---------------------------------------------------------------------------------------------
// Protected mount authority
// ---------------------------------------------------------------------------------------------

fn secret_manifest(secret: &str) -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket").with_secret_name(secret),
        )),
    )
}

/// `test_modal_direct_persist_redacts_protected_mount_provider_error`: with a protected mount in
/// play, a failed persist reports the error code and no provider text.
#[tokio::test]
async fn persist_redacts_provider_text_for_a_protected_mount() {
    let sentinel = "direct-modal-persist-secret";
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": "tar -czf -",
        "response": {"error": format!("provider echoed {sentinel}"), "retryable": false},
    }]));
    let session = ModalSandboxClient::new(fake.options())
        .create(Some(secret_manifest(sentinel)))
        .await
        .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the archive fails");

    assert!(
        error.message.contains("protected mount configuration"),
        "{}",
        error.message
    );
    assert!(!error.message.contains(sentinel), "{}", error.message);
    assert!(error.context.is_empty(), "{:?}", error.context);
}

/// `test_modal_deserialize_discards_surviving_resource_identity`: the state written for a
/// protected mount never names the sandbox that holds its authority.
#[tokio::test]
async fn a_persisted_state_drops_the_sandbox_identity_for_a_protected_mount() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(Some(secret_manifest("protected-secret")))
        .await
        .expect("create");
    let state = session.session_state();
    assert!(state.mount_authority_redacted);
    assert_eq!(state.provider_id.as_deref(), Some("sb-1"));

    let persisted = state.sanitized_for_persistence();

    assert_eq!(persisted.provider_id, None);
    assert!(
        !serde_json::to_string(&persisted)
            .expect("serialize")
            .contains("protected-secret")
    );
}

/// `test_modal_resume_creates_fresh_sandbox_for_rebound_mount_authority`: once the caller has
/// re-supplied the mount authority, resume creates a sandbox with the current secret rather than
/// reattaching the one that holds the old authority.
#[tokio::test]
async fn resume_creates_a_fresh_sandbox_for_rebound_authority() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(fake.options());
    let session = client
        .create(Some(secret_manifest("current-secret")))
        .await
        .expect("create");
    let rebound = session.session_state().with_mount_authority_rebound(true);

    let resumed = client
        .resume(&rebound, secret_manifest("current-secret"))
        .await
        .expect("resume");

    let creates = fake.of_op("create");
    assert_eq!(creates.len(), 2, "a fresh sandbox, not a reattached one");
    assert_eq!(creates[1]["volumes"][0]["secret_name"], "current-secret");
    assert_eq!(resumed.session_state().provider_id.as_deref(), Some("sb-2"));
}

/// `test_modal_snapshot_filesystem_rejects_escaping_mount_paths_before_exec`: a mount path that
/// climbs out of the workspace is refused before any command or snapshot reaches the sandbox.
#[tokio::test]
async fn an_escaping_mount_path_is_refused_before_any_command() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "logical",
        Entry::mount(
            ModalCloudBucketMountStrategy::mount(ModalCloudBucketMountConfig::s3("bucket"))
                .with_mount_path("/workspace/../../tmp"),
        ),
    );
    let created = ModalSandboxClient::new(
        fake.options()
            .with_workspace_persistence(
                openai_agents_sdk::extensions::sandbox::modal::ModalWorkspacePersistence::SnapshotFilesystem,
            ),
    )
    .create(Some(manifest))
    .await;

    let error = match created {
        Ok(session) => session
            .persist_workspace()
            .await
            .expect_err("an escaping mount path"),
        Err(error) => error,
    };

    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(fake.of_op("exec").is_empty());
    assert!(fake.of_op("snapshot_filesystem").is_empty());
}
