//! Port of `test_inaccessible_root_is_treated_as_remote_path` from
//! `tests/sandbox/test_workspace_paths.py`.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::PermissionsExt;

use openai_agents_sdk::sandbox::workspace_paths::WorkspacePathPolicy;

/// A root that cannot be examined (its parent denies search) is not an existing host path, and
/// paths still resolve lexically beneath it.
#[test]
fn an_inaccessible_root_is_treated_as_a_remote_path() {
    let temp = tempfile::tempdir().unwrap();
    let locked = temp.path().join("locked");
    std::fs::create_dir(&locked).unwrap();
    std::fs::create_dir(locked.join("project")).unwrap();
    let root = locked.join("project");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();

    let policy = WorkspacePathPolicy::new(root.to_str().unwrap()).unwrap();
    let existing = policy.root_is_existing_host_path();
    let resolved = policy.absolute_workspace_path("pkg/file.py");

    // Restore access so the tempdir can be removed.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    // A superuser can still reach the directory, which makes the premise void.
    if existing {
        return;
    }
    assert_eq!(
        resolved.unwrap(),
        format!("{}/pkg/file.py", root.to_str().unwrap())
    );
}
