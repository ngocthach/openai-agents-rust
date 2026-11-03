//! Parity port of `tests/sandbox/test_unix_local.py` — the snapshot, resume, environment,
//! workspace-archive and `apply_patch` behaviour of the Unix-local backend.
//!
//! Everything here runs against a real workspace directory, exactly as the Python suite does
//! with `tmp_path`. The PTY half of that file lives in `py_sandbox_a_unix_local_2.rs`.
#![allow(clippy::await_holding_lock)] // the lock serializes process-wide state for the whole test
#![cfg(all(unix, feature = "sandbox"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::archive::read_tar_members;
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::{
    UnixLocalSandboxClient, UnixLocalSandboxSession,
};
use openai_agents_sdk::sandbox::{
    ExecRequest, Manifest, SandboxClient, SandboxPathGrant, SandboxSession, SnapshotSpec,
};

/// Serializes every test that reads or writes the process environment: the harness runs test
/// functions on parallel threads, and `std::env::set_var` is process-wide.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A session over `root`, built the way Python's `_RecordingUnixLocalSession` is: the real
/// backend, a manifest that names the directory, and no snapshot.
fn session_at(root: &Path) -> UnixLocalSandboxSession {
    UnixLocalSandboxSession::new(
        Manifest::new().with_root(root.to_string_lossy().into_owned()),
        false,
    )
    .expect("a session over an absolute root")
}

/// A started client session whose workspace is `root` and whose snapshot lives in `snapshots`.
async fn snapshotting_session(
    client: &UnixLocalSandboxClient,
    root: &Path,
) -> Arc<dyn SandboxSession> {
    client
        .create(Some(
            Manifest::new().with_root(root.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create a session")
}

/// The archive member named `name`, with the leading `./` removed.
fn member<'a>(
    members: &'a [openai_agents_sdk::sandbox::archive::TarMember],
    name: &str,
) -> Option<&'a openai_agents_sdk::sandbox::archive::TarMember> {
    members
        .iter()
        .find(|m| m.name.trim_start_matches("./").trim_end_matches('/') == name)
}

/// The recorded link target of `name`.
fn link_target(members: &[openai_agents_sdk::sandbox::archive::TarMember], name: &str) -> String {
    member(members, name)
        .unwrap_or_else(|| panic!("`{name}` is archived"))
        .link_target
        .clone()
        .unwrap_or_default()
}

fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(target.as_ref(), link.as_ref()).expect("create a symlink");
}

// ---------------------------------------------------------------------------------------------
// Snapshot round trips — Python `test_unix_local_snapshot_round_trips_hardlinks`
// ---------------------------------------------------------------------------------------------

/// Python: `test_unix_local_snapshot_round_trips_hardlinks[False, True]`.
///
/// A hardlinked pair must be archived as two independent regular files, a symlink must stay a
/// symlink, and a resume must restore the snapshot rather than keep whatever survived in the
/// workspace. With the first name registered to be skipped, only the second is archived.
#[tokio::test]
async fn snapshot_round_trips_hardlinks() {
    for exclude_first in [false, true] {
        round_trip_hardlinks(exclude_first).await;
    }
}

async fn round_trip_hardlinks(exclude_first: bool) {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let client = UnixLocalSandboxClient::new()
        .with_inherit_host_environment(false)
        .with_snapshot(SnapshotSpec::Local {
            base_path: dir.path().join("snapshots"),
        });
    let session = snapshotting_session(&client, &workspace).await;
    session.start().await.expect("start");

    let first = workspace.join("a.py");
    let second = workspace.join("b.py");
    std::fs::write(&first, b"VALUE = 1\n").expect("write a.py");
    std::fs::set_permissions(&first, std::os::unix::fs::PermissionsExt::from_mode(0o755))
        .expect("chmod a.py");
    std::fs::hard_link(&first, &second).expect("hardlink b.py");
    symlink("b.py", workspace.join("link.py"));
    std::fs::write(workspace.join("copy.py"), b"independent\n").expect("write copy.py");
    if exclude_first {
        session
            .register_persist_workspace_skip_path("a.py")
            .expect("register the skip path");
    }

    session.stop().await.expect("stop");
    openai_agents_sdk::sandbox::session::persist_session_snapshot(session.as_ref())
        .await
        .expect("persist the snapshot");

    let archive = session.snapshot().restore().await.expect("restore archive");
    let members = read_tar_members(&archive).expect("read members");
    assert_eq!(
        member(&members, "b.py").map(|m| m.kind),
        Some(openai_agents_sdk::sandbox::archive::ArchiveMemberKind::Regular),
        "a hardlinked file is archived as a regular file, as Python's `_archive_workspace` does"
    );
    assert_eq!(
        member(&members, "link.py").map(|m| m.kind),
        Some(openai_agents_sdk::sandbox::archive::ArchiveMemberKind::Symlink),
    );
    if exclude_first {
        assert!(
            member(&members, "a.py").is_none(),
            "a registered skip path is left out of the archive"
        );
    } else {
        assert_eq!(
            member(&members, "a.py").map(|m| m.kind),
            Some(openai_agents_sdk::sandbox::archive::ArchiveMemberKind::Regular),
        );
    }

    // Prove that resume restores the snapshot, not the surviving workspace.
    std::fs::write(&second, b"changed after snapshot\n").expect("rewrite b.py");
    std::fs::write(workspace.join("stale.txt"), b"remove on resume").expect("write stale.txt");
    let state = session.session_state();
    let resumed = client
        .resume(
            &state,
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        )
        .await
        .expect("resume");
    resumed.start().await.expect("start the resumed session");

    assert_eq!(std::fs::read(&second).expect("read b.py"), b"VALUE = 1\n");
    assert_eq!(
        std::os::unix::fs::PermissionsExt::mode(
            &std::fs::metadata(&second).expect("stat b.py").permissions()
        ) & 0o777,
        0o755
    );
    assert_eq!(
        std::fs::read(workspace.join("copy.py")).expect("read copy.py"),
        b"independent\n"
    );
    assert!(
        std::fs::symlink_metadata(workspace.join("link.py"))
            .expect("stat link.py")
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::read(workspace.join("link.py")).expect("read through link.py"),
        b"VALUE = 1\n"
    );
    assert!(!workspace.join("stale.txt").exists());
    if exclude_first {
        assert!(!first.exists(), "the skipped file is not restored");
    } else {
        assert_eq!(std::fs::read(&first).expect("read a.py"), b"VALUE = 1\n");
        assert_ne!(
            std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&first).expect("stat a.py")),
            std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&second).expect("stat b.py")),
            "the restored pair are independent copies, not hardlinks"
        );
    }
}

/// A client asked to create with a snapshot spec stores snapshots there, and a snapshot the
/// client was configured with wins over the one passed in. Mirrors Python's
/// `UnixLocalSandboxClient.create(snapshot=...)`, which the runner calls with the SDK default.
#[tokio::test]
async fn create_with_snapshot_stores_snapshots_at_the_given_spec() {
    let dir = tempfile::tempdir().expect("temp dir");
    let passed = dir.path().join("passed");
    let configured = dir.path().join("configured");
    let manifest = |name: &str| {
        Manifest::new().with_root(dir.path().join(name).to_string_lossy().into_owned())
    };

    let client = UnixLocalSandboxClient::new();
    let session = client
        .create_with_snapshot(
            Some(manifest("one")),
            SnapshotSpec::Local {
                base_path: passed.clone(),
            },
        )
        .await
        .expect("create with a snapshot");
    session.start().await.expect("start");
    std::fs::write(dir.path().join("one").join("kept.txt"), b"kept").expect("write");
    openai_agents_sdk::sandbox::session::persist_session_snapshot(session.as_ref())
        .await
        .expect("persist");
    assert!(
        std::fs::read_dir(&passed)
            .expect("the passed spec's directory exists")
            .next()
            .is_some(),
        "the snapshot was stored under the spec the runner passed"
    );

    // The runner resolves client preference before it calls this, so a spec that is passed is used
    // as given, and a client's own snapshot only applies to a plain `create`.
    let client = UnixLocalSandboxClient::new().with_snapshot(SnapshotSpec::Local {
        base_path: configured.clone(),
    });
    let session = client
        .create(Some(manifest("two")))
        .await
        .expect("create with a configured snapshot");
    session.start().await.expect("start");
    openai_agents_sdk::sandbox::session::persist_session_snapshot(session.as_ref())
        .await
        .expect("persist");
    assert!(
        configured.exists(),
        "a plain create uses the client's snapshot"
    );
}

/// Python: `test_unix_local_snapshot_resume_removes_stale_link_without_following_target`
/// (`directory`, `file`, `missing`).
///
/// Clearing the workspace on resume unlinks a stale symlink; it never descends into what the
/// link points at.
#[tokio::test]
async fn snapshot_resume_removes_stale_link_without_following_target() {
    for target_kind in ["directory", "file", "missing"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        let external = dir.path().join("external");
        std::fs::create_dir(&external).expect("create external");
        let sentinel = external.join("keep.txt");
        std::fs::write(&sentinel, b"external data").expect("write sentinel");
        let target = if target_kind == "directory" {
            external.clone()
        } else {
            external.join(target_kind)
        };
        if target_kind == "file" {
            std::fs::write(&target, b"external file").expect("write external file");
        }

        let client = UnixLocalSandboxClient::new()
            .with_inherit_host_environment(false)
            .with_snapshot(SnapshotSpec::Local {
                base_path: dir.path().join("snapshots"),
            });
        let manifest = Manifest::new()
            .with_root(workspace.to_string_lossy().into_owned())
            .with_extra_path_grant(
                SandboxPathGrant::new(external.to_string_lossy()).expect("grant"),
            );
        let session = client
            .create(Some(manifest.clone()))
            .await
            .expect("create a session");
        session.start().await.expect("start");
        std::fs::write(workspace.join("original.txt"), b"snapshot content").expect("write");
        session.stop().await.expect("stop");
        openai_agents_sdk::sandbox::session::persist_session_snapshot(session.as_ref())
            .await
            .expect("persist");

        std::fs::write(workspace.join("original.txt"), b"changed after snapshot").expect("rewrite");
        let stale_link = workspace.join("stale-link");
        symlink(&target, &stale_link);

        let resumed = client
            .resume(&session.session_state(), manifest)
            .await
            .expect("resume");
        resumed.start().await.expect("start the resumed session");

        assert_eq!(
            std::fs::read(workspace.join("original.txt")).expect("read original.txt"),
            b"snapshot content",
            "{target_kind}"
        );
        assert_eq!(
            std::fs::read(&sentinel).expect("read sentinel"),
            b"external data",
            "{target_kind}"
        );
        if target_kind == "file" {
            assert_eq!(
                std::fs::read(&target).expect("read external file"),
                b"external file"
            );
        } else if target_kind == "missing" {
            assert!(!target.exists());
        }
        assert!(
            std::fs::symlink_metadata(&stale_link).is_err(),
            "{target_kind}: the stale link itself is gone"
        );
    }
}

/// Python: `test_unix_local_resume_rejects_invalid_snapshot_before_clearing_workspace`
/// (`hardlink`, `external_symlink`, `invalid_tar`).
///
/// An archive the hydrate policy refuses must be rejected before the live workspace is cleared,
/// and the session must not be left marked as running.
#[tokio::test]
async fn resume_rejects_invalid_snapshot_before_clearing_workspace() {
    for invalid_kind in ["hardlink", "external_symlink", "invalid_tar"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        let client = UnixLocalSandboxClient::new()
            .with_inherit_host_environment(false)
            .with_snapshot(SnapshotSpec::Local {
                base_path: dir.path().join("snapshots"),
            });
        let manifest = Manifest::new().with_root(workspace.to_string_lossy().into_owned());
        let session = client
            .create(Some(manifest.clone()))
            .await
            .expect("create a session");
        session.start().await.expect("start");
        std::fs::write(workspace.join("keep.txt"), b"live workspace").expect("write keep.txt");

        let archive: Vec<u8> = match invalid_kind {
            "invalid_tar" => b"not a tar archive".to_vec(),
            "hardlink" => tar_archive(vec![tar_member("link", TYPE_HARDLINK, "keep.txt")]),
            _ => tar_archive(vec![tar_member("link", TYPE_SYMLINK, "../outside")]),
        };
        session
            .snapshot()
            .persist(&archive)
            .await
            .expect("persist the crafted archive");

        let resumed = client
            .resume(&session.session_state(), manifest)
            .await
            .expect("resume");
        let error = resumed
            .start()
            .await
            .expect_err("an invalid snapshot is refused");
        assert!(
            matches!(
                error.code,
                ErrorCode::SnapshotRestoreError
                    | ErrorCode::WorkspaceArchiveWriteError
                    | ErrorCode::InvalidCompressionScheme
                    | ErrorCode::WorkspaceStartError
            ),
            "{invalid_kind}: unexpected code {:?} ({})",
            error.code,
            error.message
        );
        assert_eq!(
            std::fs::read(workspace.join("keep.txt")).expect("keep.txt survives"),
            b"live workspace",
            "{invalid_kind}"
        );
        let mut names: Vec<String> = std::fs::read_dir(&workspace)
            .expect("list workspace")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["keep.txt".to_string()], "{invalid_kind}");
        assert!(
            !resumed.running().await.expect("running"),
            "{invalid_kind}: a failed start leaves the session stopped"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Environment — Python `test_unix_local_inherits_host_environment_by_default` and friends
// ---------------------------------------------------------------------------------------------

/// Python: `test_unix_local_inherits_host_environment_by_default`.
#[tokio::test]
async fn inherits_the_host_environment_by_default() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    // SAFETY: every test in this binary that touches the environment holds `ENV_LOCK`.
    unsafe {
        std::env::set_var("PY_PARITY_HOST_SECRET", "host-secret");
        std::env::set_var("PY_PARITY_LC_PRIVATE_TOKEN", "locale-secret");
    }
    let manifest = Manifest::new()
        .with_root(workspace.to_string_lossy().into_owned())
        .with_env("HOME", "/manifest-home")
        .with_env("MANIFEST_ONLY", "configured");

    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");
    let result = session
        .exec(ExecRequest::argv([
            "sh",
            "-c",
            "printf '%s|%s|%s|%s|%s' \"${PY_PARITY_HOST_SECRET-unset}\" \"$MANIFEST_ONLY\" \
             \"$HOME\" \"${PATH:+set}\" \"${PY_PARITY_LC_PRIVATE_TOKEN-unset}\"",
        ]))
        .await
        .expect("exec");
    session.shutdown().await.expect("shutdown");
    unsafe {
        std::env::remove_var("PY_PARITY_HOST_SECRET");
        std::env::remove_var("PY_PARITY_LC_PRIVATE_TOKEN");
    }

    assert_eq!(result.exit_code, 0);
    assert_eq!(
        result.stdout_text(),
        format!(
            "host-secret|configured|{}|set|locale-secret",
            workspace.display()
        ),
        "the manifest's HOME is overridden by the workspace root, as Python does"
    );
}

/// Python: `test_unix_local_uses_default_allowlist_when_inheritance_is_disabled`.
#[tokio::test]
async fn uses_the_default_allowlist_when_inheritance_is_disabled() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    // SAFETY: every test in this binary that touches the environment holds `ENV_LOCK`.
    unsafe {
        std::env::set_var("PY_PARITY_HOST_ONLY_VALUE", "host-value");
        std::env::set_var("LC_MESSAGES", "C");
    }
    let manifest = Manifest::new().with_root(workspace.to_string_lossy().into_owned());
    let isolated = UnixLocalSandboxClient::new().with_inherit_host_environment(false);

    let session = isolated
        .create(Some(manifest.clone()))
        .await
        .expect("create");
    session.start().await.expect("start");
    let created = session
        .exec(ExecRequest::argv([
            "sh",
            "-c",
            "printf '%s|%s' \"${PY_PARITY_HOST_ONLY_VALUE-unset}\" \"$LC_MESSAGES\"",
        ]))
        .await
        .expect("exec");
    let state = session.session_state();
    session.shutdown().await.expect("shutdown");
    assert_eq!(
        created.stdout_text(),
        "unset|C",
        "an allowlisted locale variable is passed through; an arbitrary host one is not"
    );

    // The client's inheritance policy is never serialized into the session state, so a resume
    // through a different client gets that client's policy. Python asserts the same.
    let payload = serde_json::to_value(&state).expect("serialize the state");
    assert!(payload.get("inherit_host_environment").is_none());
    assert!(payload.get("host_environment_allowlist").is_none());

    let resumed = isolated
        .resume(&state, manifest.clone())
        .await
        .expect("resume isolated");
    resumed.start().await.expect("start");
    let isolated_after_resume = resumed
        .exec(ExecRequest::argv([
            "sh",
            "-c",
            "printf '%s' \"${PY_PARITY_HOST_ONLY_VALUE-unset}\"",
        ]))
        .await
        .expect("exec");
    resumed.shutdown().await.expect("shutdown");
    assert_eq!(isolated_after_resume.stdout_text(), "unset");

    let inheriting = UnixLocalSandboxClient::new();
    let resumed = inheriting
        .resume(&state, manifest)
        .await
        .expect("resume inheriting");
    resumed.start().await.expect("start");
    let inherited_after_resume = resumed
        .exec(ExecRequest::argv([
            "sh",
            "-c",
            "printf '%s' \"${PY_PARITY_HOST_ONLY_VALUE-unset}\"",
        ]))
        .await
        .expect("exec");
    resumed.shutdown().await.expect("shutdown");
    unsafe {
        std::env::remove_var("PY_PARITY_HOST_ONLY_VALUE");
    }
    assert_eq!(inherited_after_resume.stdout_text(), "host-value");
}

/// Python: `test_unix_local_uses_custom_host_environment_allowlist`.
#[tokio::test]
async fn uses_a_custom_host_environment_allowlist() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    // SAFETY: every test in this binary that touches the environment holds `ENV_LOCK`.
    unsafe {
        std::env::set_var("PY_PARITY_CUSTOM_ALLOWED", "allowed-value");
        std::env::set_var("PY_PARITY_HOST_ONLY_VALUE", "host-value");
    }
    let manifest = Manifest::new().with_root(workspace.to_string_lossy().into_owned());
    let client = UnixLocalSandboxClient::new()
        .with_inherit_host_environment(false)
        .with_host_environment_allowlist(["PATH", "PY_PARITY_CUSTOM_ALLOWED"]);

    let probe = ExecRequest::argv([
        "sh",
        "-c",
        "printf '%s|%s' \"$PY_PARITY_CUSTOM_ALLOWED\" \"${PY_PARITY_HOST_ONLY_VALUE-unset}\"",
    ]);
    let session = client.create(Some(manifest.clone())).await.expect("create");
    session.start().await.expect("start");
    let result = session.exec(probe.clone()).await.expect("exec");
    let state = session.session_state();
    session.shutdown().await.expect("shutdown");
    assert_eq!(result.stdout_text(), "allowed-value|unset");

    let resumed = client.resume(&state, manifest).await.expect("resume");
    resumed.start().await.expect("start");
    let resumed_result = resumed.exec(probe).await.expect("exec");
    resumed.shutdown().await.expect("shutdown");
    unsafe {
        std::env::remove_var("PY_PARITY_CUSTOM_ALLOWED");
        std::env::remove_var("PY_PARITY_HOST_ONLY_VALUE");
    }
    assert_eq!(
        resumed_result.stdout_text(),
        "allowed-value|unset",
        "the allowlist survives a resume through the same client"
    );
}

/// Python: `test_unix_local_rejects_invalid_host_environment_allowlist_configuration`
/// (the `ValueError` half).
///
/// Rust builders cannot fail, so the refusal lands where the session is built. The `TypeError`
/// half is Python-only: the Rust API takes `IntoIterator<Item = Into<String>>`, so a bare string
/// does not compile.
#[tokio::test]
async fn rejects_an_allowlist_without_disabling_inheritance() {
    let dir = tempfile::tempdir().expect("temp dir");
    let client = UnixLocalSandboxClient::new().with_host_environment_allowlist(["PATH"]);
    let error = client
        .create(Some(
            Manifest::new().with_root(dir.path().to_string_lossy().into_owned()),
        ))
        .await
        .expect_err("an allowlist without `inherit_host_environment=false` is refused");
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(
        error
            .message
            .contains("host_environment_allowlist requires inherit_host_environment=False"),
        "{}",
        error.message
    );
}

/// Python: `test_unix_local_rejects_host_path_before_creating_workspace`.
///
/// The refusal has to come before the temporary workspace is created, so the assertion is that
/// no `openai-agents-sandbox-` directory appeared.
#[tokio::test]
async fn rejects_a_sandbox_path_grant_host_path_before_creating_a_workspace() {
    let dir = tempfile::tempdir().expect("temp dir");
    let before = temp_workspace_count();
    let client = UnixLocalSandboxClient::new();
    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .expect("grant")
        .with_host_path(dir.path().to_string_lossy())
        .expect("host path");
    let error = client
        .create(Some(Manifest::new().with_extra_path_grant(grant)))
        .await
        .expect_err("a host_path grant is refused");
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(
        error
            .message
            .contains("UnixLocalSandboxClient does not support sandbox path grant host_path"),
        "{}",
        error.message
    );
    assert_eq!(
        temp_workspace_count(),
        before,
        "no workspace directory was created before the refusal"
    );
}

/// How many SDK-owned temporary workspaces exist right now.
fn temp_workspace_count() -> usize {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("openai-agents-sandbox-")
        })
        .count()
}

// ---------------------------------------------------------------------------------------------
// Restorable archive members — Python `TestUnixLocalPersistWorkspaceRestorable`
// ---------------------------------------------------------------------------------------------

/// Build Python's `TestUnixLocalPersistWorkspaceRestorable._workspace`.
fn restorable_workspace(base: &Path) -> PathBuf {
    let workspace = base.join("workspace");
    std::fs::create_dir_all(workspace.join("sub")).expect("create sub");
    std::fs::write(workspace.join("a.txt"), "shared").expect("write a.txt");
    make_fifo(&workspace.join("dev.fifo"));
    symlink(workspace.join("a.txt"), workspace.join("abs_inside"));
    symlink(
        workspace.join("a.txt"),
        workspace.join("sub").join("abs_up"),
    );
    symlink("a.txt", workspace.join("rel"));
    symlink(
        format!("/{}", workspace.join("a.txt").display()),
        workspace.join("double_slash"),
    );
    symlink(
        format!("{}//a.txt", workspace.display()),
        workspace.join("double_sep"),
    );
    symlink(base.join("elsewhere.txt"), workspace.join("outside"));
    workspace
}

/// Create a FIFO at `path`.
fn make_fifo(path: &Path) {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).expect("c string");
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the call.
    let code = unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) };
    assert_eq!(code, 0, "mkfifo `{}`", path.display());
}

/// Python: `test_persist_emits_restorable_members`.
#[tokio::test]
async fn persist_emits_restorable_members() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = restorable_workspace(dir.path());
    let archive = session_at(&workspace)
        .persist_workspace()
        .await
        .expect("persist");
    let members = read_tar_members(&archive).expect("read members");

    assert!(
        member(&members, "dev.fifo").is_none(),
        "a FIFO is not restorable, so it is left out of the archive"
    );
    assert_eq!(link_target(&members, "abs_inside"), "a.txt");
    assert_eq!(link_target(&members, "sub/abs_up"), "../a.txt");
    assert_eq!(link_target(&members, "rel"), "a.txt");
    assert_eq!(link_target(&members, "double_slash"), "a.txt");
    assert_eq!(link_target(&members, "double_sep"), "a.txt");
    assert_eq!(
        link_target(&members, "outside"),
        dir.path().join("elsewhere.txt").to_string_lossy(),
        "a target outside the workspace keeps its absolute form"
    );
}

/// Python: `test_rebased_symlink_keeps_parent_steps_after_symlink_components`.
///
/// `<root>/current/../config` with `current -> releases/v1` names `releases/config`; collapsing
/// the `..` lexically would silently retarget the restored link.
#[tokio::test]
async fn rebased_symlink_keeps_parent_steps_after_symlink_components() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(workspace.join("releases").join("v1")).expect("create releases/v1");
    std::fs::write(workspace.join("releases").join("config"), "right").expect("write config");
    std::fs::write(workspace.join("config"), "wrong").expect("write wrong config");
    symlink("releases/v1", workspace.join("current"));
    symlink(
        workspace.join("current").join("..").join("config"),
        workspace.join("abs_config"),
    );
    symlink(
        workspace.join("current").join("..").join("config"),
        workspace.join("releases").join("v1").join("abs_up"),
    );
    assert_eq!(
        std::fs::read_to_string(workspace.join("abs_config")).expect("read abs_config"),
        "right"
    );

    let archive = session_at(&workspace)
        .persist_workspace()
        .await
        .expect("persist");
    let restored_root = dir.path().join("restored");
    std::fs::create_dir(&restored_root).expect("create restored root");
    session_at(&restored_root)
        .hydrate_workspace(&archive)
        .await
        .expect("hydrate");

    assert_eq!(
        std::fs::read_link(restored_root.join("abs_config")).expect("read abs_config link"),
        PathBuf::from("current/../config")
    );
    assert_eq!(
        std::fs::read_link(restored_root.join("releases").join("v1").join("abs_up"))
            .expect("read abs_up link"),
        PathBuf::from("../../current/../config")
    );
    assert_eq!(
        std::fs::read_to_string(restored_root.join("abs_config")).expect("read abs_config"),
        "right"
    );
    assert_eq!(
        std::fs::read_to_string(restored_root.join("releases").join("v1").join("abs_up"))
            .expect("read abs_up"),
        "right"
    );
}

/// Python: `test_rebased_symlink_that_escapes_through_a_link_stays_absolute`.
#[tokio::test]
async fn rebased_symlink_that_escapes_through_a_link_stays_absolute() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(workspace.join("a")).expect("create a");
    symlink("..", workspace.join("a").join("link"));
    symlink(
        workspace.join("a").join("link").join("..").join("tmp"),
        workspace.join("victim"),
    );
    symlink(dir.path(), workspace.join("outside"));
    symlink(
        workspace.join("outside").join("workspace").join("a"),
        workspace.join("via_outside"),
    );
    symlink("loop", workspace.join("loop"));
    symlink(
        workspace.join("loop").join("..").join("..").join("etc"),
        workspace.join("via_loop"),
    );
    symlink("a/link", workspace.join("b"));
    symlink(
        workspace.join("b").join("a"),
        workspace.join("a").join("fine"),
    );

    let archive = session_at(&workspace)
        .persist_workspace()
        .await
        .expect("persist");
    let members = read_tar_members(&archive).expect("read members");

    assert_eq!(
        link_target(&members, "victim"),
        workspace
            .join("a")
            .join("link")
            .join("..")
            .join("tmp")
            .to_string_lossy()
    );
    assert_eq!(
        link_target(&members, "via_outside"),
        workspace
            .join("outside")
            .join("workspace")
            .join("a")
            .to_string_lossy()
    );
    assert_eq!(
        link_target(&members, "via_loop"),
        workspace
            .join("loop")
            .join("..")
            .join("..")
            .join("etc")
            .to_string_lossy()
    );
    // `..` after `b -> a/link -> ..` lands on the root, so `b/a` is provably inside.
    assert_eq!(link_target(&members, "a/fine"), "../b/a");
}

/// Python: `test_rebased_symlink_through_components_the_snapshot_does_not_create_stays_absolute`.
///
/// Python injects a runtime skip set; the Rust port derives its archive exclusions from the
/// manifest, so the `via_skipped` leg of that test has no equivalent and is reported as a gap.
/// The other legs — a missing component, a dangling leaf and a file used as a directory — are
/// exactly Python's.
#[tokio::test]
async fn rebased_symlink_through_components_the_archive_does_not_create_stays_absolute() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).expect("create workspace");
    std::fs::write(workspace.join("secret"), "s").expect("write secret");
    std::fs::write(workspace.join("notes.txt"), "n").expect("write notes.txt");
    symlink(
        workspace.join("alias").join("..").join("secret"),
        workspace.join("via_missing"),
    );
    symlink(workspace.join("missing.txt"), workspace.join("dangling"));
    symlink(
        workspace.join("notes.txt").join("..").join("secret"),
        workspace.join("via_file"),
    );
    symlink(workspace.join("secret"), workspace.join("fine"));

    let archive = session_at(&workspace)
        .persist_workspace()
        .await
        .expect("persist");
    let members = read_tar_members(&archive).expect("read members");

    assert_eq!(
        link_target(&members, "via_missing"),
        workspace
            .join("alias")
            .join("..")
            .join("secret")
            .to_string_lossy()
    );
    assert_eq!(
        link_target(&members, "dangling"),
        workspace.join("missing.txt").to_string_lossy()
    );
    assert_eq!(
        link_target(&members, "via_file"),
        workspace
            .join("notes.txt")
            .join("..")
            .join("secret")
            .to_string_lossy()
    );
    assert_eq!(link_target(&members, "fine"), "secret");
}

/// Python: `test_persisted_workspace_hydrates_into_a_new_root`.
#[tokio::test]
async fn a_persisted_workspace_hydrates_into_a_new_root() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = restorable_workspace(dir.path());
    // Hydrate rejects external targets by design.
    std::fs::remove_file(workspace.join("outside")).expect("drop the external link");
    let archive = session_at(&workspace)
        .persist_workspace()
        .await
        .expect("persist");

    let restored_root = dir.path().join("restored");
    std::fs::create_dir(&restored_root).expect("create restored root");
    session_at(&restored_root)
        .hydrate_workspace(&archive)
        .await
        .expect("hydrate");

    assert!(!restored_root.join("dev.fifo").exists());
    assert_eq!(
        std::fs::read_link(restored_root.join("abs_inside")).expect("read link"),
        PathBuf::from("a.txt")
    );
    assert_eq!(
        std::fs::read_to_string(restored_root.join("abs_inside")).expect("read abs_inside"),
        "shared"
    );
    assert_eq!(
        std::fs::read_to_string(restored_root.join("sub").join("abs_up")).expect("read abs_up"),
        "shared"
    );
}

// ---------------------------------------------------------------------------------------------
// apply_patch creates — Python's standalone `test_apply_patch_create_*` functions
// ---------------------------------------------------------------------------------------------

/// The editor Python's `_exclusive_write_session` drives, over `root`.
fn editor_at(root: &Path) -> SandboxEditor {
    SandboxEditor::new(Arc::new(session_at(root)) as Arc<dyn SandboxSession>)
}

fn create_op(path: &str, diff: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: None,
    }
}

/// Python: `test_apply_patch_create_through_the_session_rejects_a_dangling_symlink`.
///
/// A create aimed at a dangling link must not land on the link's absent target.
#[tokio::test]
async fn apply_patch_create_rejects_a_dangling_symlink() {
    let dir = tempfile::tempdir().expect("temp dir");
    symlink(dir.path().join("missing.txt"), dir.path().join("link.txt"));

    let error = editor_at(dir.path())
        .create_file(&create_op("link.txt", "+clobbered\n"))
        .await
        .expect_err("a dangling link is not a free name");
    assert!(format!("{error}").contains("already exists"), "{error}");
    assert!(!dir.path().join("missing.txt").exists());
    assert!(
        std::fs::symlink_metadata(dir.path().join("link.txt"))
            .expect("stat link.txt")
            .file_type()
            .is_symlink()
    );
}

/// Python: `test_apply_patch_create_through_the_session_rejects_a_directory`.
#[tokio::test]
async fn apply_patch_create_rejects_a_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir(dir.path().join("adir")).expect("create adir");

    let error = editor_at(dir.path())
        .create_file(&create_op("adir", "+clobbered\n"))
        .await
        .expect_err("a directory is not a free name");
    assert!(format!("{error}").contains("adir"), "{error}");
    assert_eq!(
        std::fs::read_dir(dir.path().join("adir"))
            .expect("list adir")
            .count(),
        0
    );
}

/// Python: `test_apply_patch_create_through_the_session_keeps_existing_content`.
#[tokio::test]
async fn apply_patch_create_keeps_existing_content() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("notes.txt"), b"important\n").expect("write notes.txt");

    let error = editor_at(dir.path())
        .create_file(&create_op("notes.txt", "+clobbered\n"))
        .await
        .expect_err("an existing file is not overwritten by a create");
    assert!(format!("{error}").contains("already exists"), "{error}");
    assert_eq!(
        std::fs::read(dir.path().join("notes.txt")).expect("read notes.txt"),
        b"important\n"
    );
}

/// Python: `test_apply_patch_create_through_the_session_writes_a_new_nested_file`.
#[tokio::test]
async fn apply_patch_create_writes_a_new_nested_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    editor_at(dir.path())
        .create_file(&create_op("nested/dir/new.txt", "+hello\n"))
        .await
        .expect("create a nested file");

    assert_eq!(
        std::fs::read_to_string(dir.path().join("nested").join("dir").join("new.txt"))
            .expect("read new.txt"),
        "hello"
    );
    let leftovers: Vec<String> = std::fs::read_dir(dir.path().join("nested").join("dir"))
        .expect("list dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "no staging file is left behind");
}

/// Python: `test_apply_patch_create_through_the_session_reports_a_file_parent_as_a_write_error`.
///
/// A parent that is a regular file is a write error, not a collision on the requested name:
/// reporting a collision would tell the model to use `update_file` on a path that cannot exist.
#[tokio::test]
async fn apply_patch_create_reports_a_file_parent_as_a_write_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("parent"), b"i am a file\n").expect("write parent");

    let error = editor_at(dir.path())
        .create_file(&create_op("parent/child.txt", "+hi\n"))
        .await
        .expect_err("a file parent cannot hold a child");
    assert!(
        !format!("{error}").contains("already exists"),
        "a file parent is not a collision on the requested name: {error}"
    );
}

/// Python: `test_apply_patch_create_accepts_a_destination_at_the_component_limit`.
#[tokio::test]
async fn apply_patch_create_accepts_a_destination_at_the_component_limit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let long_name = format!("{}.txt", "a".repeat(250));
    // Confirm the platform really accepts this name, so a failure is about the editor.
    let probe = dir.path().join(&long_name);
    std::fs::write(&probe, "probe").expect("the platform accepts a 254-byte component");
    std::fs::remove_file(&probe).expect("remove the probe");

    editor_at(dir.path())
        .create_file(&create_op(&long_name, "+hello\n"))
        .await
        .expect("create at the component limit");
    assert_eq!(
        std::fs::read_to_string(dir.path().join(&long_name)).expect("read the long name"),
        "hello"
    );
}

/// Python: `test_apply_patch_create_reports_collision_inside_a_read_only_parent`.
#[tokio::test]
async fn apply_patch_create_reports_a_collision_inside_a_read_only_parent() {
    if is_root() {
        // Python skips this too: root bypasses directory write permissions.
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let parent = dir.path().join("locked");
    std::fs::create_dir(&parent).expect("create locked");
    let target = parent.join("notes.txt");
    std::fs::write(&target, b"important\n").expect("write notes.txt");
    std::fs::set_permissions(&parent, std::os::unix::fs::PermissionsExt::from_mode(0o555))
        .expect("chmod locked");

    let error = editor_at(dir.path())
        .create_file(&create_op("locked/notes.txt", "+clobbered\n"))
        .await
        .expect_err("a visible collision is reported as one");
    std::fs::set_permissions(&parent, std::os::unix::fs::PermissionsExt::from_mode(0o755))
        .expect("restore locked");
    assert!(format!("{error}").contains("already exists"), "{error}");
    assert_eq!(
        std::fs::read(&target).expect("read notes.txt"),
        b"important\n"
    );
}

/// Python: `test_apply_patch_create_supports_a_symlinked_parent`.
#[tokio::test]
async fn apply_patch_create_supports_a_symlinked_parent() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir(dir.path().join("real")).expect("create real");
    symlink(dir.path().join("real"), dir.path().join("internal"));

    editor_at(dir.path())
        .create_file(&create_op("internal/new.txt", "+hello\n"))
        .await
        .expect("create through a safe internal alias");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("real").join("new.txt")).expect("read new.txt"),
        "hello"
    );
}

/// Whether the test process is root, which bypasses the permission bits Python relies on.
fn is_root() -> bool {
    // SAFETY: `geteuid` reads the calling process's effective uid and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

// ---------------------------------------------------------------------------------------------
// Raw `ustar` fixtures, for the archives a writer refuses to produce.
// ---------------------------------------------------------------------------------------------

const TYPE_HARDLINK: u8 = b'1';
const TYPE_SYMLINK: u8 = b'2';

fn tar_member(name: &str, type_flag: u8, link_name: &str) -> Vec<u8> {
    fn put(header: &mut [u8; 512], offset: usize, bytes: &[u8]) {
        header[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
    let mut header = [0u8; 512];
    put(&mut header, 0, name.as_bytes());
    put(&mut header, 100, b"0000644\0");
    put(&mut header, 108, b"0000000\0");
    put(&mut header, 116, b"0000000\0");
    put(&mut header, 124, b"00000000000\0");
    put(&mut header, 136, b"00000000000\0");
    put(&mut header, 148, b"        ");
    header[156] = type_flag;
    put(&mut header, 157, link_name.as_bytes());
    put(&mut header, 257, b"ustar\0");
    put(&mut header, 263, b"00");
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    put(&mut header, 148, format!("{checksum:06o}\0 ").as_bytes());
    header.to_vec()
}

fn tar_archive(members: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out: Vec<u8> = members.into_iter().flatten().collect();
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}
