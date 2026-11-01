//! Ports of `TestUnixLocalUserScopedFilesystem` in `tests/sandbox/test_unix_local.py`:
//! `test_mkdir_as_user_checks_permissions_then_uses_local_fs` and
//! `test_rm_as_user_checks_permissions_then_uses_local_fs`.
//!
//! Python replaces the session's `_exec_internal` to record the commands. A Rust session has no
//! such seam, so a `sudo` stand-in earlier on the manifest's `PATH` records its arguments and
//! then runs the wrapped command as the current user.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::User;

/// Install a `sudo` that logs one line per invocation (arguments separated by a unit separator)
/// and runs the command that follows `-u <user> --`.
fn fake_sudo(dir: &Path, log: &Path) {
    let script = format!(
        "#!/bin/sh\n\
         line=\"\"\n\
         for arg in \"$@\"; do line=\"$line$arg\u{1f}\"; done\n\
         printf '%s\\036' \"$line\" >> '{}'\n\
         shift 3\n\
         exec \"$@\"\n",
        log.display()
    );
    let path = dir.join("sudo");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

async fn session_with_fake_sudo(root: &Path, bin: &Path) -> std::sync::Arc<dyn SandboxSession> {
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new()
                .with_root(root.to_string_lossy().to_string())
                .with_env("PATH", path),
        ))
        .await
        .expect("created");
    session.start().await.expect("started");
    session
}

fn invocations(log: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .split('\u{1e}')
        .filter(|record| !record.is_empty())
        .map(|record| {
            record
                .trim_end_matches('\u{1f}')
                .split('\u{1f}')
                .map(str::to_string)
                .collect()
        })
        .collect()
}

#[tokio::test]
async fn mkdir_as_a_user_checks_permissions_then_uses_the_local_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    let bin = dir.path().join("bin");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&bin).unwrap();
    // The macOS wrapper confines commands to the workspace, so the log lives inside it.
    let log = workspace.join("sudo.log");
    fake_sudo(&bin, &log);
    let session = session_with_fake_sudo(&workspace, &bin).await;

    session
        .mkdir_as("nested", false, Some(&User::new("sandbox-user")))
        .await
        .expect("mkdir");

    assert!(workspace.join("nested").is_dir());
    let calls = invocations(&log);
    assert_eq!(calls.len(), 1, "{calls:?}");
    let argv = &calls[0];
    assert_eq!(&argv[..5], ["-u", "sandbox-user", "--", "sh", "-c"]);
    let root = std::fs::canonicalize(&workspace).unwrap();
    let tail = &argv[argv.len() - 2..];
    assert!(
        tail[0] == workspace.join("nested").to_string_lossy()
            || tail[0] == root.join("nested").to_string_lossy(),
        "{argv:?}"
    );
    assert_eq!(tail[1], "0");
    assert!(
        !argv.iter().any(|part| part.starts_with("mkdir ")),
        "the check never creates the directory: {argv:?}"
    );
}

#[tokio::test]
async fn rm_as_a_user_checks_permissions_then_uses_the_local_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    let bin = dir.path().join("bin");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&bin).unwrap();
    let target = workspace.join("stale.txt");
    std::fs::write(&target, "stale").unwrap();
    // The macOS wrapper confines commands to the workspace, so the log lives inside it.
    let log = workspace.join("sudo.log");
    fake_sudo(&bin, &log);
    let session = session_with_fake_sudo(&workspace, &bin).await;

    session
        .rm_as("stale.txt", false, Some(&User::new("sandbox-user")))
        .await
        .expect("rm");

    assert!(!target.exists());
    let calls = invocations(&log);
    assert_eq!(calls.len(), 1, "{calls:?}");
    let argv = &calls[0];
    assert_eq!(&argv[..5], ["-u", "sandbox-user", "--", "sh", "-c"]);
    let root = std::fs::canonicalize(&workspace).unwrap();
    let tail = &argv[argv.len() - 2..];
    assert!(
        tail[0] == target.to_string_lossy() || tail[0] == root.join("stale.txt").to_string_lossy(),
        "{argv:?}"
    );
    assert_eq!(tail[1], "0");
    assert!(
        !argv.iter().any(|part| part.starts_with("rm ")),
        "the check never removes the file: {argv:?}"
    );
}
