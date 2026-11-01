//! Python `tests/sandbox/test_workspace_paths.py` cases written against Windows path types:
//! the Rust policy takes strings, so a Windows-looking input is the observable equivalent.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::ErrorCode;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, WorkspacePathPolicy};

/// Python: `test_relative_path_rejects_windows_drive_absolute_path_for_host_root`.
#[test]
fn a_windows_drive_path_is_absolute_for_the_policy_whatever_its_separators() {
    let workspace = tempfile::tempdir().expect("workspace");
    let policy = WorkspacePathPolicy::new(workspace.path().to_string_lossy()).unwrap();

    for path in ["C:/tmp/secret.txt", "C:\\tmp\\secret.txt"] {
        let error = policy
            .relative_path(path)
            .expect_err("a drive path is not relative");

        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "{path}");
        assert_eq!(
            error.message, "manifest path must be relative: C:/tmp/secret.txt",
            "{path}"
        );
        assert_eq!(
            error.context.get("rel").map(String::as_str),
            Some("C:/tmp/secret.txt")
        );
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("absolute")
        );
    }
}

/// Python: `test_normalize_sandbox_path_uses_posix_paths_for_windows_inputs` and
/// `test_normalize_path_uses_posix_paths_for_windows_inputs`: separators are read as POSIX.
#[test]
fn windows_style_separators_normalize_to_posix_paths() {
    let policy = WorkspacePathPolicy::new("/workspace").unwrap();

    assert_eq!(policy.sandbox_root(), "/workspace");
    assert_eq!(
        policy
            .normalize_sandbox_path("/workspace/pkg/file.py", false)
            .unwrap(),
        "/workspace/pkg/file.py"
    );
    assert_eq!(
        policy
            .normalize_sandbox_path("pkg\\file.py", false)
            .unwrap(),
        "/workspace/pkg/file.py"
    );
    assert_eq!(
        policy.absolute_workspace_path("pkg\\file.py").unwrap(),
        "/workspace/pkg/file.py"
    );
}

/// Python: `test_recursive_remove_uses_posix_paths_for_windows_inputs`.
#[test]
fn recursive_remove_checks_read_only_grants_for_the_given_path() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![
            SandboxPathGrant::new("/srv/data").unwrap(),
            SandboxPathGrant::new("/srv/data/protected")
                .unwrap()
                .read_only(true),
        ],
    )
    .unwrap();

    assert!(policy.validate_recursive_remove("/srv/data").is_err());
    assert!(
        policy
            .validate_recursive_remove("/srv/data/writable")
            .is_ok()
    );
}
