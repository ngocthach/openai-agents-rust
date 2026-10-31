//! Ports of the persistence, resume and cloud-bucket groups of
//! `tests/extensions/sandbox/test_modal.py`.
//!
//! Python patches the `modal` SDK; the Rust provider drives a Python helper over a JSON protocol,
//! so the same behaviour is observed through the requests the helper receives. The helper is
//! `tests/fixtures/fake_modal_helper_scripted.py`: it logs every request, and a rules file lets a
//! test make one operation fail or answer differently.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy, ModalSandboxClient,
    ModalSandboxClientOptions, ModalWorkspacePersistence, decode_snapshot_filesystem_ref,
    encode_snapshot_directory_ref, encode_snapshot_filesystem_ref,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{GcsMount, R2Mount, S3Mount};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
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

    /// Install the rules the fake helper applies before its default behaviour.
    fn rules(&self, rules: Value) {
        std::fs::write(self.path("rules.json"), rules.to_string()).expect("write rules");
    }

    fn command(&self) -> Vec<String> {
        vec![
            "/usr/bin/env".to_string(),
            format!("FAKE_MODAL_LOG={}", self.path("requests.jsonl").display()),
            format!("FAKE_MODAL_STATE={}", self.path("state.json").display()),
            format!("FAKE_MODAL_RULES={}", self.path("rules.json").display()),
            "python3".to_string(),
            format!(
                "{}/tests/fixtures/fake_modal_helper_scripted.py",
                env!("CARGO_MANIFEST_DIR")
            ),
        ]
    }

    fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(self.path("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect()
    }

    fn of_op(&self, op: &str) -> Vec<Value> {
        self.requests()
            .into_iter()
            .filter(|request| request["op"] == op)
            .collect()
    }

    /// The `op` of every request, with the script of a shell command appended.
    fn trace(&self) -> Vec<String> {
        self.requests()
            .into_iter()
            .map(|request| {
                let op = request["op"].as_str().unwrap_or_default().to_string();
                match request["argv"][2].as_str() {
                    Some(script) if request["op"] == "exec" => format!("exec:{script}"),
                    _ => op,
                }
            })
            .collect()
    }

    fn options(&self, persistence: ModalWorkspacePersistence) -> ModalSandboxClientOptions {
        ModalSandboxClientOptions::new("agents-app")
            .with_workspace_persistence(persistence)
            .with_helper_command(self.command())
    }
}

fn plain_options(fake: &FakeModal) -> ModalSandboxClientOptions {
    ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command())
}

fn exec_scripts(fake: &FakeModal) -> Vec<String> {
    fake.of_op("exec")
        .into_iter()
        .filter_map(|request| request["argv"][2].as_str().map(str::to_string))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Tar persistence
// ---------------------------------------------------------------------------------------------

/// `test_modal_tar_persist_respects_runtime_skip_paths` and
/// `test_modal_tar_persist_preserves_dot_prefixed_skip_paths`: a registered skip path becomes a
/// `./`-prefixed `--exclude`, and a leading dot is kept.
#[tokio::test]
async fn tar_persist_excludes_registered_skip_paths() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(plain_options(&fake))
        .create(None)
        .await
        .expect("create");
    session
        .register_persist_workspace_skip_path("logs/events.jsonl")
        .expect("register");
    session
        .register_persist_workspace_skip_path(".venv")
        .expect("register a dot-prefixed path");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, b"fake-tar-bytes");
    let scripts = exec_scripts(&fake);
    let tar = scripts
        .iter()
        .find(|script| script.contains("tar -czf -"))
        .expect("a tar command");
    assert!(tar.contains("--exclude ./logs/events.jsonl"), "{tar}");
    assert!(tar.contains("--exclude ./.venv"), "{tar}");
    assert!(tar.contains("-C /workspace"), "{tar}");
}

/// `test_modal_tar_persist_uses_resolved_mount_paths_for_excludes`: a mount that is not ephemeral
/// still sits outside the workspace archive, at the path it resolves to rather than the entry
/// name.
#[tokio::test]
async fn tar_persist_excludes_resolved_mount_paths() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "logical",
        Entry::mount(
            ModalCloudBucketMountStrategy::mount(
                ModalCloudBucketMountConfig::s3("bucket").with_secret_name("creds"),
            )
            .with_mount_path("actual")
            .with_ephemeral(false),
        ),
    );
    let session = ModalSandboxClient::new(plain_options(&fake))
        .create(Some(manifest))
        .await
        .expect("create");

    session.persist_workspace().await.expect("persist");

    let scripts = exec_scripts(&fake);
    let tar = scripts
        .iter()
        .find(|script| script.contains("tar -czf -"))
        .expect("a tar command");
    assert!(tar.contains("--exclude ./actual"), "{tar}");
    assert!(!tar.contains("logical"), "{tar}");
}

/// `test_modal_tar_persist_retries_wrapped_exec_transport_error`: a transport failure on the tar
/// command is retried and the second answer is used.
#[tokio::test]
async fn tar_persist_retries_a_transport_error() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": "tar -czf -",
        "once": true,
        "response": {"error": "ConnectionError: modal transport failed"},
    }]));
    let session = ModalSandboxClient::new(plain_options(&fake))
        .create(None)
        .await
        .expect("create");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, b"fake-tar-bytes");
    let tars = exec_scripts(&fake)
        .into_iter()
        .filter(|script| script.contains("tar -czf -"))
        .count();
    assert_eq!(tars, 2);
}

/// `test_modal_tar_persist_does_not_retry_wrapped_non_retryable_exec_transport_error`: an error
/// the helper marks non-retryable is surfaced after one attempt.
#[tokio::test]
async fn tar_persist_does_not_retry_a_non_retryable_error() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": "tar -czf -",
        "response": {"error": "NotFoundError: sandbox is gone", "retryable": false},
    }]));
    let session = ModalSandboxClient::new(plain_options(&fake))
        .create(None)
        .await
        .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("a permanent failure");

    assert_eq!(error.retryable, Some(false));
    let tars = exec_scripts(&fake)
        .into_iter()
        .filter(|script| script.contains("tar -czf -"))
        .count();
    assert_eq!(tars, 1);
}

/// `test_modal_snapshot_filesystem_falls_back_to_tar_for_non_detachable_mounts`: a Modal bucket
/// mount cannot be detached, so a filesystem snapshot is replaced by a tar archive.
#[tokio::test]
async fn snapshot_filesystem_falls_back_to_tar_for_a_bucket_mount() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket").with_secret_name("creds"),
        )),
    );
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
            .create(Some(manifest))
            .await
            .expect("create");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, b"fake-tar-bytes");
    assert!(fake.of_op("snapshot_filesystem").is_empty());
}

/// `test_modal_create_allows_snapshot_filesystem_with_modal_cloud_bucket_mounts`
#[tokio::test]
async fn create_allows_snapshot_filesystem_with_bucket_mounts() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket"),
        )),
    );
    ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
        .create(Some(manifest))
        .await
        .expect("create");

    let volumes = &fake.of_op("create")[0]["volumes"];
    assert_eq!(volumes[0]["mount_path"], "/workspace/remote");
    assert_eq!(volumes.as_array().map(Vec::len), Some(1));
}

/// `test_modal_create_rejects_snapshot_directory_with_cloud_bucket_mount_under_workspace`
#[tokio::test]
async fn create_rejects_snapshot_directory_with_a_bucket_under_the_workspace() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket"),
        )),
    );
    let error = ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
        .create(Some(manifest))
        .await
        .expect_err("a bucket under the snapshot root");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("at or under the workspace root"),
        "{}",
        error.message
    );
    assert!(fake.of_op("create").is_empty());
}

/// `test_modal_create_allows_snapshot_directory_with_cloud_bucket_mount_outside_workspace`
#[tokio::test]
async fn create_allows_snapshot_directory_with_a_bucket_outside_the_workspace() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(
            ModalCloudBucketMountStrategy::mount(ModalCloudBucketMountConfig::s3("bucket"))
                .with_mount_path("/mnt/remote"),
        ),
    );
    ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
        .create(Some(manifest))
        .await
        .expect("create");

    assert_eq!(
        fake.of_op("create")[0]["volumes"][0]["mount_path"],
        "/mnt/remote"
    );
}

// ---------------------------------------------------------------------------------------------
// Filesystem snapshot: the paths the image must not capture are set aside and put back
// ---------------------------------------------------------------------------------------------

fn manifest_with_ephemeral_file() -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "logs/events.jsonl",
        Entry::file(b"skip".to_vec()).with_ephemeral(true),
    )
}

/// A rule that makes the set-aside archive command answer with `ephemeral-backup`.
fn backup_rule() -> Value {
    json!({
        "op": "exec",
        "contains": "tar cf -",
        "response": {"exit_code": 0, "stdout": "ZXBoZW1lcmFsLWJhY2t1cA==", "stderr": ""},
    })
}

fn position(requests: &[Value], matches: impl Fn(&Value) -> bool) -> Option<usize> {
    requests.iter().position(matches)
}

fn is_exec_script(request: &Value, needle: &str) -> bool {
    request["op"] == "exec"
        && request["argv"][2]
            .as_str()
            .is_some_and(|script| script.contains(needle))
}

fn is_removal(request: &Value) -> bool {
    request["op"] == "exec"
        && request["argv"][0] == "rm"
        && request["argv"]
            .to_string()
            .contains("/workspace/logs/events.jsonl")
}

fn is_restore_upload(request: &Value) -> bool {
    request["op"] == "write_file" && request["data"] == "ZXBoZW1lcmFsLWJhY2t1cA=="
}

/// `test_modal_snapshot_filesystem_uses_resolved_mount_paths_for_backup_and_removal`: the
/// ephemeral entry is archived into this process, removed from the sandbox, and only then is the
/// image taken; the archive goes back afterwards. (A manifest with a bucket mount persists as tar
/// instead, so the mount-path half of the Python test has no Rust twin.)
#[tokio::test]
async fn snapshot_filesystem_sets_ephemeral_paths_aside_and_restores_them() {
    let fake = FakeModal::new();
    fake.rules(json!([backup_rule()]));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(
        decode_snapshot_filesystem_ref(&archive).as_deref(),
        Some("im-sb-1")
    );
    let requests = fake.requests();
    let backup = position(&requests, |request| {
        is_exec_script(request, "tar cf -") && is_exec_script(request, "logs/events.jsonl")
    })
    .expect("the ephemeral entry is archived");
    let removal = position(&requests, is_removal).expect("the entry is removed");
    let image = position(&requests, |request| request["op"] == "snapshot_filesystem")
        .expect("the image is taken");
    let upload = position(&requests, is_restore_upload).expect("the archive goes back");
    let extract =
        position(&requests, |request| is_exec_script(request, "tar xf")).expect("and is extracted");
    assert!(backup < removal && removal < image && image < upload && upload < extract);
    // The archive never touched the sandbox's disk before the image: it is not a command's
    // output file, so the image cannot hold it.
    assert!(
        !is_exec_script(&requests[backup], "modal-snapshot"),
        "{}",
        requests[backup]
    );
}

/// `test_modal_snapshot_failure_restores_ephemeral_paths`: a failed image still puts the paths
/// back.
#[tokio::test]
async fn snapshot_filesystem_failure_restores_the_ephemeral_paths() {
    let fake = FakeModal::new();
    fake.rules(json!([
        backup_rule(),
        {
            "op": "snapshot_filesystem",
            "response": {"error": "snapshot failed", "code": "snapshot_failed"},
        },
    ]));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the image fails");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    let requests = fake.requests();
    let image = position(&requests, |request| request["op"] == "snapshot_filesystem")
        .expect("the image was attempted");
    let upload = position(&requests, is_restore_upload).expect("the archive goes back");
    assert!(image < upload);
    assert!(position(&requests, |request| is_exec_script(request, "tar xf")).is_some());
}

/// `test_modal_snapshot_cleanup_failure_raises_before_snapshot`: if the removal fails, no image
/// is taken and the paths are restored.
#[tokio::test]
async fn snapshot_filesystem_cleanup_failure_skips_the_image() {
    let fake = FakeModal::new();
    fake.rules(json!([
        backup_rule(),
        {
            "op": "exec",
            "contains": "logs/events.jsonl",
            "response": {"exit_code": 1, "stdout": "", "stderr": "cm0gZmFpbGVk"},
        },
    ]));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the removal fails");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(error.message.contains("rm failed"), "{}", error.message);
    assert!(fake.of_op("snapshot_filesystem").is_empty());
    let requests = fake.requests();
    assert!(
        position(&requests, is_restore_upload).is_some()
            && position(&requests, |request| is_exec_script(request, "tar xf")).is_some(),
        "the archive is restored"
    );
}

// ---------------------------------------------------------------------------------------------
// Directory snapshot
// ---------------------------------------------------------------------------------------------

/// `test_modal_snapshot_directory_uses_resolved_mount_paths_for_backup_and_removal`
#[tokio::test]
async fn snapshot_directory_backup_names_the_ephemeral_entry_and_a_scratch_archive() {
    let fake = FakeModal::new();
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(
        archive,
        encode_snapshot_directory_ref("dir-sb-1"),
        "the reference envelope"
    );
    assert_eq!(fake.of_op("snapshot_directory")[0]["path"], "/workspace");
    let backup = exec_scripts(&fake)
        .into_iter()
        .find(|script| script.contains("tar cf"))
        .expect("a backup command");
    assert!(backup.contains("logs/events.jsonl"), "{backup}");
    assert!(backup.contains("-T -"), "{backup}");
    assert!(
        backup.contains("modal-snapshot-directory-ephemeral"),
        "{backup}"
    );
    assert!(backup.contains("/tmp/openai-agents"), "{backup}");
    assert!(backup.contains("rm -rf"), "{backup}");
}

/// `test_modal_snapshot_directory_backup_failure_aborts_before_removing_ephemeral_paths`: the
/// removal is chained after the archive, so a failed archive removes nothing, and no image is
/// taken.
#[tokio::test]
async fn snapshot_directory_backup_failure_aborts_before_the_image() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": "tar cf",
        "once": true,
        "response": {"exit_code": 1, "stdout": "", "stderr": "bWtkaXIgZmFpbGVk"},
    }]));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let error = session
        .persist_workspace()
        .await
        .expect_err("the backup fails");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    assert!(error.message.contains("mkdir failed"), "{}", error.message);
    assert!(fake.of_op("snapshot_directory").is_empty());
    // The `rm -rf` lives in the same command as the archive, behind `test -f`.
    let backup = exec_scripts(&fake)
        .into_iter()
        .find(|script| script.contains("tar cf"))
        .expect("a backup command");
    assert!(
        backup.find("test -f").expect("test -f") < backup.find("rm -rf").expect("rm -rf"),
        "{backup}"
    );
}

/// `test_modal_snapshot_directory_tolerates_missing_ephemeral_paths_in_backup_command`: the
/// backup lists only the paths that exist.
#[tokio::test]
async fn snapshot_directory_backup_tolerates_missing_paths() {
    let fake = FakeModal::new();
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    session.persist_workspace().await.expect("persist");

    let backup = exec_scripts(&fake)
        .into_iter()
        .find(|script| script.contains("tar cf"))
        .expect("a backup command");
    assert!(backup.contains("if [ -e \"$rel\" ]"), "{backup}");
    assert!(backup.contains("2>/dev/null"), "{backup}");
}

/// `test_modal_snapshot_unexpected_return_restores_live_session_before_raising`: a helper that
/// answers a directory snapshot without an image id fails the persist, after the paths are put
/// back.
#[tokio::test]
async fn snapshot_directory_unexpected_return_restores_before_raising() {
    let fake = FakeModal::new();
    fake.rules(json!([{"op": "snapshot_directory", "response": {"image_id": ""}}]));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest_with_ephemeral_file()))
            .await
            .expect("create");

    let error = session.persist_workspace().await.expect_err("no image id");

    assert_eq!(error.code, ErrorCode::SnapshotPersistError);
    let trace = fake.trace();
    let snapshot = trace
        .iter()
        .position(|entry| entry == "snapshot_directory")
        .expect("attempted");
    assert!(
        trace[snapshot + 1..]
            .iter()
            .any(|entry| entry.contains("tar xf")),
        "{trace:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Restore
// ---------------------------------------------------------------------------------------------

/// `test_modal_snapshot_filesystem_restore_preserves_exposed_ports`: the replacement sandbox is
/// created with the same ports and resources.
#[tokio::test]
async fn filesystem_restore_preserves_ports_and_resources() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(
        fake.options(ModalWorkspacePersistence::SnapshotFilesystem)
            .with_exposed_ports([8765])
            .with_idle_timeout_secs(60)
            .with_cpu_range(1.0, 4.0)
            .with_memory_mb_range(2048, 8192),
    )
    .create(None)
    .await
    .expect("create");

    session
        .hydrate_workspace(&encode_snapshot_filesystem_ref("snap-123"))
        .await
        .expect("restore");

    let creates = fake.of_op("create");
    assert_eq!(creates.len(), 2);
    let replacement = &creates[1];
    assert_eq!(replacement["image_id"], "snap-123");
    assert_eq!(replacement["encrypted_ports"], json!([8765]));
    assert_eq!(replacement["idle_timeout"], 60);
    assert_eq!(replacement["cpu"], json!([1.0, 4.0]));
    assert_eq!(replacement["memory"], json!([2048, 8192]));
}

/// `test_modal_snapshot_directory_restore_preserves_exposed_ports`: a directory restore keeps the
/// running sandbox and mounts the image over the workspace root.
#[tokio::test]
async fn directory_restore_keeps_the_sandbox_and_mounts_the_image() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(
        fake.options(ModalWorkspacePersistence::SnapshotDirectory)
            .with_exposed_ports([8765]),
    )
    .create(None)
    .await
    .expect("create");

    session
        .hydrate_workspace(&encode_snapshot_directory_ref("snap-dir-123"))
        .await
        .expect("restore");

    assert_eq!(fake.of_op("create").len(), 1);
    assert_eq!(fake.of_op("create")[0]["encrypted_ports"], json!([8765]));
    let mount = &fake.of_op("mount_image")[0];
    assert_eq!(mount["path"], "/workspace");
    assert_eq!(mount["image_id"], "snap-dir-123");
    assert!(fake.of_op("terminate").is_empty());
}

fn local_snapshot_with(directory: &std::path::Path, payload: &[u8]) -> (SnapshotSpec, String) {
    // `LocalSnapshot` stores `<base>/<snapshot id>.tar`; the session id is only known after
    // `create`, so the caller writes the payload through the session's own snapshot instead.
    let _ = payload;
    (
        SnapshotSpec::Local {
            base_path: directory.to_path_buf(),
        },
        String::new(),
    )
}

/// `test_modal_resume_eagerly_reconnects_sandbox` and
/// `test_modal_resume_marks_reconnected_sandbox_preserved_before_snapshot_reuse`: a running
/// sandbox is polled and reused, never recreated.
#[tokio::test]
async fn resume_reconnects_a_running_sandbox_without_creating_one() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(plain_options(&fake));
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");
    let state = session.session_state();
    let manifest = session.manifest().clone();

    let resumed = client.resume(&state, manifest).await.expect("resume");

    assert_eq!(fake.of_op("create").len(), 1, "no second sandbox");
    assert_eq!(fake.of_op("poll")[0]["sandbox_id"], "sb-1");
    assert_eq!(resumed.session_state().provider_id.as_deref(), Some("sb-1"));
}

/// `test_modal_resume_resets_workspace_readiness_when_sandbox_is_recreated`: a stopped sandbox is
/// replaced by a fresh one created from the same options.
#[tokio::test]
async fn resume_recreates_a_stopped_sandbox() {
    let fake = FakeModal::new();
    let client =
        ModalSandboxClient::new(plain_options(&fake).with_image_builder_version("PREVIEW"));
    let session = client.create(None).await.expect("create");
    let state = session.session_state();
    let manifest = session.manifest().clone();
    // The sandbox stopped on its own.
    std::fs::write(
        fake.path("state.json"),
        json!({"sandboxes": 1, "files": {}, "terminated": [], "stopped": ["sb-1"], "rules_used": {}})
            .to_string(),
    )
    .expect("state");

    let resumed = client.resume(&state, manifest).await.expect("resume");

    let creates = fake.of_op("create");
    assert_eq!(creates.len(), 2);
    assert_eq!(creates[1]["image_builder_version"], "PREVIEW");
    assert_eq!(resumed.session_state().provider_id.as_deref(), Some("sb-2"));
}

/// `test_modal_resume_restores_snapshot_when_workspace_readiness_unproven` and its directory
/// variant: a resumed session restores the persisted image reference on `start` rather than
/// trusting the surviving workspace.
#[tokio::test]
async fn resume_restores_the_persisted_snapshot_on_start() {
    for (persistence, archive, expect_create, expect_mount) in [
        (
            ModalWorkspacePersistence::SnapshotFilesystem,
            encode_snapshot_filesystem_ref("snap-123"),
            2usize,
            0usize,
        ),
        (
            ModalWorkspacePersistence::SnapshotDirectory,
            encode_snapshot_directory_ref("snap-dir-123"),
            1,
            1,
        ),
    ] {
        let fake = FakeModal::new();
        let snapshots = tempfile::tempdir().expect("snapshots");
        let (spec, _) = local_snapshot_with(snapshots.path(), &archive);
        let client = ModalSandboxClient::new(fake.options(persistence)).with_snapshot(spec);
        let session = client.create(None).await.expect("create");
        // Store the reference through the session's own snapshot, which is what a prior
        // `stop` does.
        session.snapshot().persist(&archive).await.expect("persist");
        let state = session.session_state();
        let manifest = session.manifest().clone();

        let resumed = client.resume(&state, manifest).await.expect("resume");
        resumed.start().await.expect("start");

        assert_eq!(fake.of_op("create").len(), expect_create, "{persistence:?}");
        assert_eq!(
            fake.of_op("mount_image").len(),
            expect_mount,
            "{persistence:?}"
        );
        if expect_mount == 1 {
            let mount = &fake.of_op("mount_image")[0];
            assert_eq!(mount["path"], "/workspace");
            assert_eq!(mount["image_id"], "snap-dir-123");
        } else {
            assert_eq!(fake.of_op("create")[1]["image_id"], "snap-123");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cloud bucket mount configuration
// ---------------------------------------------------------------------------------------------

/// `test_modal_sandbox_create_passes_modal_cloud_bucket_mounts`: inline credentials, a prefix and
/// a writable mount reach the helper.
#[tokio::test]
async fn create_passes_inline_credentials_to_the_helper() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_aws_credentials("access-key", "secret-key")
                .with_key_prefix("nested/prefix/")
                .with_read_only(false),
        )),
    );
    ModalSandboxClient::new(plain_options(&fake))
        .create(Some(manifest))
        .await
        .expect("create");

    assert_eq!(
        fake.of_op("create")[0]["volumes"],
        json!([{
            "mount_path": "/workspace/remote",
            "bucket_name": "bucket",
            "key_prefix": "nested/prefix/",
            "credentials": {
                "AWS_ACCESS_KEY_ID": "access-key",
                "AWS_SECRET_ACCESS_KEY": "secret-key",
            },
            "read_only": false,
        }])
    );
}

/// `test_modal_sandbox_create_passes_named_modal_secret_environment_for_cloud_bucket_mount`
#[tokio::test]
async fn create_passes_a_named_secret_with_its_environment() {
    let fake = FakeModal::new();
    let manifest = Manifest::new().with_entry(
        "remote",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_secret_name("named-modal-secret")
                .with_secret_environment_name("staging")
                .with_key_prefix("nested/prefix/")
                .with_read_only(false),
        )),
    );
    ModalSandboxClient::new(plain_options(&fake))
        .create(Some(manifest))
        .await
        .expect("create");

    let volume = &fake.of_op("create")[0]["volumes"][0];
    assert_eq!(volume["secret_name"], "named-modal-secret");
    assert_eq!(volume["secret_environment_name"], "staging");
    assert!(volume.get("credentials").is_none());
}

fn config_of(
    mount: &dyn openai_agents_sdk::sandbox::mount_entries::ConfiguredMount,
) -> ModalCloudBucketMountConfig {
    ModalCloudBucketMountConfig::from_configured_mount(mount).expect("a valid mount")
}

/// `test_modal_cloud_bucket_mount_strategy_builds_s3_config`
#[test]
fn bucket_config_for_an_s3_mount() {
    let mount = S3Mount::new("bucket")
        .with_credentials("access-key", "secret-key")
        .with_session_token("session-token")
        .with_prefix("nested/prefix/")
        .with_endpoint_url("https://s3.example.test");

    let config = config_of(&mount).with_read_only(false);

    let spec = config.to_volume_spec("/workspace/remote");
    assert_eq!(spec["bucket_name"], "bucket");
    assert_eq!(spec["bucket_endpoint_url"], "https://s3.example.test");
    assert_eq!(spec["key_prefix"], "nested/prefix/");
    assert_eq!(
        spec["credentials"],
        json!({
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
            "AWS_SESSION_TOKEN": "session-token",
        })
    );
    assert_eq!(spec["read_only"], false);
}

/// `test_modal_cloud_bucket_mount_strategy_builds_s3_config_with_named_secret` and the
/// environment variant: a credential-free mount can name a secret.
#[test]
fn bucket_config_for_an_s3_mount_with_a_named_secret() {
    let mount = S3Mount::new("bucket").with_prefix("nested/prefix/");

    let config = config_of(&mount)
        .with_secret_name("named-modal-secret")
        .with_secret_environment_name("staging")
        .with_read_only(false);

    config.validate().expect("valid");
    let spec = config.to_volume_spec("/workspace/remote");
    assert_eq!(spec["bucket_name"], "bucket");
    assert!(spec.get("bucket_endpoint_url").is_none());
    assert!(spec.get("credentials").is_none());
    assert_eq!(spec["secret_name"], "named-modal-secret");
    assert_eq!(spec["secret_environment_name"], "staging");
    assert_eq!(spec["read_only"], false);
}

/// `test_modal_cloud_bucket_mount_strategy_builds_r2_config`: the R2 endpoint comes from the
/// account id and the mount is read-only unless asked otherwise.
#[test]
fn bucket_config_for_an_r2_mount() {
    let mount =
        R2Mount::new("bucket", "abc123accountid").with_credentials("access-key", "secret-key");

    let config = config_of(&mount);

    let spec = config.to_volume_spec("/workspace/remote");
    assert_eq!(
        spec["bucket_endpoint_url"],
        "https://abc123accountid.r2.cloudflarestorage.com"
    );
    assert!(spec.get("key_prefix").is_none());
    assert_eq!(
        spec["credentials"],
        json!({
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
        })
    );
    assert_eq!(spec["read_only"], true);
}

/// `test_modal_cloud_bucket_mount_strategy_builds_gcs_hmac_config` and the named-secret variant.
#[test]
fn bucket_config_for_a_gcs_hmac_mount() {
    let mount = GcsMount::new("bucket")
        .with_hmac_credentials("access-id", "secret-key")
        .with_prefix("nested/prefix/");

    let spec = config_of(&mount)
        .with_read_only(false)
        .to_volume_spec("/workspace/remote");

    assert_eq!(
        spec["bucket_endpoint_url"],
        "https://storage.googleapis.com"
    );
    assert_eq!(spec["key_prefix"], "nested/prefix/");
    assert_eq!(
        spec["credentials"],
        json!({
            "GOOGLE_ACCESS_KEY_ID": "access-id",
            "GOOGLE_ACCESS_KEY_SECRET": "secret-key",
        })
    );
    assert_eq!(spec["read_only"], false);

    let named = ModalCloudBucketMountConfig::gcs("bucket")
        .with_key_prefix("nested/prefix/")
        .with_secret_name("named-modal-secret")
        .with_read_only(false)
        .to_volume_spec("/workspace/remote");
    assert_eq!(
        named["bucket_endpoint_url"],
        "https://storage.googleapis.com"
    );
    assert!(named.get("credentials").is_none());
    assert_eq!(named["secret_name"], "named-modal-secret");
    assert!(named.get("secret_environment_name").is_none());
}

/// `test_modal_cloud_bucket_mount_strategy_rejects_secret_environment_name_without_secret_name`
#[test]
fn bucket_config_rejects_an_environment_without_a_secret() {
    let error = ModalCloudBucketMountConfig::s3("bucket")
        .with_secret_environment_name("staging")
        .validate()
        .expect_err("an environment needs a secret");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error
            .message
            .contains("secret_environment_name requires secret_name to also be set"),
        "{}",
        error.message
    );
}

/// `test_modal_cloud_bucket_mount_strategy_rejects_mixed_inline_credentials_and_secret_name`
#[test]
fn bucket_config_rejects_inline_credentials_with_a_secret_name() {
    let mount = S3Mount::new("bucket").with_credentials("access-key", "secret-key");

    let error = config_of(&mount)
        .with_secret_name("named-modal-secret")
        .validate()
        .expect_err("mixed credential sources");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error
            .message
            .contains("do not support both inline credentials and secret_name"),
        "{}",
        error.message
    );
}

/// `test_modal_cloud_bucket_mount_strategy_rejects_gcs_native_auth`: Modal speaks only the HMAC
/// interface, so a service-account file is refused.
#[test]
fn bucket_config_rejects_gcs_native_auth() {
    let mount = GcsMount::new("bucket").with_service_account_file("/data/config/gcs.json");

    let error = ModalCloudBucketMountConfig::from_configured_mount(&mount)
        .expect_err("native auth is not supported");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error
            .message
            .contains("require access_id and secret_access_key"),
        "{}",
        error.message
    );
}

// ---------------------------------------------------------------------------------------------
// Durable mounts around a directory snapshot, and the resume fingerprint fast path
// ---------------------------------------------------------------------------------------------

type Events = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// A mount strategy that records every detach and attach with the path it was given.
#[derive(Debug)]
struct RecordingStrategy {
    events: Events,
}

#[async_trait::async_trait]
impl openai_agents_sdk::sandbox::mounts::MountStrategy for RecordingStrategy {
    fn strategy_type(&self) -> &str {
        "recording"
    }
    async fn activate(
        &self,
        _: &openai_agents_sdk::sandbox::mounts::Mount,
        _: &dyn openai_agents_sdk::sandbox::SandboxSession,
        _: &str,
    ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
        Ok(())
    }
    async fn deactivate(
        &self,
        _: &openai_agents_sdk::sandbox::mounts::Mount,
        _: &dyn openai_agents_sdk::sandbox::SandboxSession,
        _: &str,
    ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
        Ok(())
    }
    async fn teardown_for_snapshot(
        &self,
        _: &openai_agents_sdk::sandbox::mounts::Mount,
        _: &dyn openai_agents_sdk::sandbox::SandboxSession,
        path: &str,
    ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
        self.events.lock().unwrap().push(format!("unmount:{path}"));
        Ok(())
    }
    async fn restore_after_snapshot(
        &self,
        _: &openai_agents_sdk::sandbox::mounts::Mount,
        _: &dyn openai_agents_sdk::sandbox::SandboxSession,
        path: &str,
    ) -> openai_agents_sdk::sandbox::SandboxResult<()> {
        self.events.lock().unwrap().push(format!("mount:{path}"));
        Ok(())
    }
}

fn durable_mount(events: &Events, mount_path: &str) -> Entry {
    Entry::mount(
        openai_agents_sdk::sandbox::mounts::Mount::new(std::sync::Arc::new(RecordingStrategy {
            events: events.clone(),
        }))
        .with_mount_path(mount_path)
        .with_ephemeral(false),
    )
}

/// `test_modal_snapshot_directory_restore_reactivates_durable_workspace_mounts`: after the image is
/// mounted over the root, the durable mounts inside the root are attached again.
#[tokio::test]
async fn directory_restore_reactivates_durable_workspace_mounts() {
    let fake = FakeModal::new();
    let events: Events = Events::default();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("remote", durable_mount(&events, "actual"));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest))
            .await
            .expect("create");

    session
        .hydrate_workspace(&encode_snapshot_directory_ref("snap-dir-123"))
        .await
        .expect("restore");

    let mount = &fake.of_op("mount_image")[0];
    assert_eq!(mount["path"], "/workspace");
    assert_eq!(mount["image_id"], "snap-dir-123");
    assert_eq!(*events.lock().unwrap(), vec!["mount:/workspace/actual"]);
}

/// `test_modal_snapshot_directory_persist_only_detaches_durable_workspace_mounts`: only the
/// durable mount inside the root is detached for the image and attached again; one outside the
/// root is left alone.
#[tokio::test]
async fn directory_persist_detaches_only_durable_workspace_mounts() {
    let fake = FakeModal::new();
    let events: Events = Events::default();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("inside", durable_mount(&events, "actual"))
        .with_entry("outside", durable_mount(&events, "/mnt/remote"));
    let session =
        ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotDirectory))
            .create(Some(manifest))
            .await
            .expect("create");

    let archive = session.persist_workspace().await.expect("persist");

    assert_eq!(archive, encode_snapshot_directory_ref("dir-sb-1"));
    assert_eq!(
        *events.lock().unwrap(),
        vec!["unmount:/workspace/actual", "mount:/workspace/actual"]
    );
    // The image was taken between the detach and the attach.
    assert_eq!(fake.of_op("snapshot_directory").len(), 1);
}

fn fingerprint_rule(fingerprint: &str) -> Value {
    use openai_agents_sdk::sandbox::runtime_helpers::WORKSPACE_FINGERPRINT_HELPER;
    json!({
        "op": "exec",
        "contains": format!("{} /workspace ", WORKSPACE_FINGERPRINT_HELPER.install_path),
        "response": {
            "exit_code": 0,
            "stdout": base64_of(&format!(
                "{{\"fingerprint\":\"{fingerprint}\",\"version\":\"workspace_tar_sha256_v1\"}}"
            )),
            "stderr": "",
        },
    })
}

fn base64_of(text: &str) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(text)
}

/// `test_modal_resume_marks_reconnected_sandbox_preserved_before_snapshot_reuse`, and its opposite:
/// a reattached sandbox whose workspace still hashes to the stored fingerprint skips the snapshot
/// restore; one whose workspace changed restores it.
#[tokio::test]
async fn resume_skips_the_restore_when_the_workspace_fingerprint_matches() {
    for (live, restores) in [("same", 0usize), ("changed", 1)] {
        let fake = FakeModal::new();
        fake.rules(json!([fingerprint_rule(live)]));
        let snapshots = tempfile::tempdir().expect("snapshots");
        let client =
            ModalSandboxClient::new(fake.options(ModalWorkspacePersistence::SnapshotFilesystem))
                .with_snapshot(SnapshotSpec::Local {
                    base_path: snapshots.path().to_path_buf(),
                });
        let session = client.create(None).await.expect("create");
        session
            .snapshot()
            .persist(&encode_snapshot_filesystem_ref("snap-123"))
            .await
            .expect("persist");
        let state = session.session_state().with_snapshot_fingerprint(Some(
            openai_agents_sdk::sandbox::SnapshotFingerprint {
                fingerprint: "same".to_string(),
                version: "workspace_tar_sha256_v1".to_string(),
            },
        ));
        let manifest = session.manifest().clone();

        let resumed = client.resume(&state, manifest).await.expect("resume");
        resumed.start().await.expect("start");

        // The reattached sandbox is the first creation; a restore adds a replacement from the image.
        assert_eq!(fake.of_op("create").len(), 1 + restores, "{live}");
    }
}

/// The fingerprint a session records travels in its state, so a later resume can compare it.
#[tokio::test]
async fn the_session_state_carries_the_recorded_fingerprint() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(plain_options(&fake))
        .create(None)
        .await
        .expect("create");

    session.record_snapshot_fingerprint(Some(openai_agents_sdk::sandbox::SnapshotFingerprint {
        fingerprint: "abc".to_string(),
        version: "workspace_tar_sha256_v2".to_string(),
    }));

    let state = session.session_state();
    assert_eq!(state.snapshot_fingerprint.as_deref(), Some("abc"));
    assert_eq!(
        state.snapshot_fingerprint_version.as_deref(),
        Some("workspace_tar_sha256_v2")
    );
}
