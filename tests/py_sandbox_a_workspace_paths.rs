//! Parity port of `tests/sandbox/test_workspace_paths.py` from the Python SDK.
//!
//! Every behavioral case from that file is reproduced here with the same inputs and the same
//! expected results, including Python's exact error text and the `rel` / `reason` context it
//! attaches to an invalid manifest path. The Python cases that only exercise `pathlib` typing,
//! `monkeypatch` on `Path.exists`, or a Windows-native host are recorded in the area report
//! instead of being asserted here.

#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;

use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxOp};
use openai_agents_sdk::sandbox::workspace_paths::{
    SandboxPathGrant, SandboxWorkspaceScope, WorkspacePathPolicy, normalize_sandbox_cwd,
    sandbox_path_grant_host_path,
};
use openai_agents_sdk::sandbox::{Manifest, SandboxError};

// ----------------------------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------------------------

fn policy(root: &str) -> WorkspacePathPolicy {
    WorkspacePathPolicy::new(root).expect("policy root")
}

fn scope(cwd: &str) -> SandboxWorkspaceScope {
    SandboxWorkspaceScope::from_cwd(Some(cwd)).expect("scope cwd")
}

/// Python's `exc_info.value.context`, as a comparable map.
fn context(error: &SandboxError) -> BTreeMap<String, String> {
    error.context.clone()
}

fn ctx(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

/// Assert the invalid-manifest-path error Python raises for `path`.
fn assert_invalid_manifest_path(error: &SandboxError, rel: &str, reason: &str) {
    let expected = if reason == "absolute" {
        format!("manifest path must be relative: {rel}")
    } else {
        format!("manifest path must not escape root: {rel}")
    };
    assert_eq!(error.message, expected);
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert_eq!(error.op, Some(SandboxOp::Materialize));
    assert_eq!(context(error), ctx(&[("rel", rel), ("reason", reason)]));
}

// ----------------------------------------------------------------------------------------------
// SandboxWorkspaceScope
// ----------------------------------------------------------------------------------------------

#[test]
fn sandbox_workspace_scope_anchors_relative_paths() {
    let scope = scope("tasks/a");

    assert_eq!(scope.anchor("plot.png"), "tasks/a/plot.png");
    assert_eq!(scope.anchor("reports/plot.png"), "tasks/a/reports/plot.png");
    assert_eq!(scope.anchor("/workspace/plot.png"), "/workspace/plot.png");
    assert_eq!(scope.anchor("C:/plot.png"), "C:/plot.png");
}

#[test]
fn sandbox_workspace_scope_preserves_raw_tool_backslashes() {
    let scope = scope("tasks/a");

    // The model's spelling survives anchoring: a backslash is a literal name character here,
    // not a separator the scope may rewrite.
    assert_eq!(
        scope.anchor(r"reports\plot.png"),
        r"tasks/a/reports\plot.png"
    );
}

#[test]
fn sandbox_workspace_scope_renders_model_paths() {
    let scope = scope("tasks/a");

    assert_eq!(scope.model_path("tasks/a/plot.png").unwrap(), "plot.png");
    assert_eq!(
        scope.model_path("shared/skill/SKILL.md").unwrap(),
        "../../shared/skill/SKILL.md"
    );
    assert_eq!(
        scope
            .display_path("../shared.txt", "tasks/shared.txt")
            .unwrap(),
        "../shared.txt"
    );
    assert_eq!(
        scope
            .display_path("/workspace/shared.txt", "shared.txt")
            .unwrap(),
        "shared.txt"
    );
}

#[test]
fn sandbox_workspace_scope_renders_session_resources_as_absolute_with_cwd() {
    let scope = scope("tasks/a");

    assert_eq!(
        scope
            .model_resource_path("/workspace", ".agents/my-skill")
            .unwrap(),
        "/workspace/.agents/my-skill"
    );
    // A drive-qualified root is accepted and re-spelled in POSIX, as Python does.
    assert_eq!(
        scope
            .model_resource_path(r"C:\workspace", ".agents/my-skill")
            .unwrap(),
        "C:/workspace/.agents/my-skill"
    );
    assert_eq!(
        scope
            .model_resource_path("C:/workspace", ".agents/my-skill")
            .unwrap(),
        "C:/workspace/.agents/my-skill"
    );
}

#[test]
fn sandbox_workspace_scope_preserves_root_relative_resource_paths_without_cwd() {
    let scope = SandboxWorkspaceScope::root();

    assert_eq!(
        scope
            .model_resource_path("/workspace", ".agents/my-skill")
            .unwrap(),
        ".agents/my-skill"
    );
}

#[test]
fn sandbox_workspace_scope_rejects_invalid_session_resource_paths() {
    let scope = scope("tasks/a");

    for (path, message) in [
        ("/workspace/.agents/my-skill", "must be workspace-relative"),
        ("../my-skill", "must be workspace-relative"),
        ("C:/skills/my-skill", "must be workspace-relative"),
        (r"C:\skills\my-skill", "must use POSIX path separators"),
        (r".agents\my-skill", "must use POSIX path separators"),
        ("", "must be non-empty"),
    ] {
        let error = scope
            .model_resource_path("/workspace", path)
            .expect_err(path);
        assert!(
            error.message.contains(message),
            "`{path}` gave `{}`",
            error.message
        );
        assert!(error.message.starts_with("session resource paths "));
    }
}

#[test]
fn sandbox_workspace_scope_rejects_non_posix_absolute_resource_roots() {
    let scope = scope("tasks/a");

    for root in [r"\workspace", "workspace"] {
        let error = scope
            .model_resource_path(root, ".agents/my-skill")
            .expect_err(root);
        assert_eq!(
            error.message,
            "sandbox workspace root must be POSIX absolute"
        );
    }
}

#[test]
fn sandbox_workspace_scope_none_preserves_root_relative_paths() {
    let scope = SandboxWorkspaceScope::root();

    assert_eq!(scope.anchor("plot.png"), "plot.png");
    assert_eq!(
        scope.model_path("reports/plot.png").unwrap(),
        "reports/plot.png"
    );
}

#[test]
fn sandbox_workspace_scope_constructor_validates_cwd() {
    let error = SandboxWorkspaceScope::from_cwd(Some("tasks/../a")).expect_err("parent segment");
    assert_eq!(
        error.message,
        "sandbox.cwd must not contain parent segments"
    );
}

#[test]
fn normalize_sandbox_cwd_rejects_invalid_values() {
    for (cwd, message) in [
        ("", "sandbox.cwd must be non-empty"),
        (
            "/workspace/tasks/a",
            "sandbox.cwd must be workspace-relative",
        ),
        ("tasks/../a", "sandbox.cwd must not contain parent segments"),
        (r"tasks\a", "sandbox.cwd must use POSIX path separators"),
        ("C:/tasks/a", "sandbox.cwd must be workspace-relative"),
    ] {
        let error = normalize_sandbox_cwd(cwd).expect_err(cwd);
        assert_eq!(error.message, message, "for `{cwd}`");
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    }
}

// ----------------------------------------------------------------------------------------------
// WorkspacePathPolicy — absolute_workspace_path / relative_path
// ----------------------------------------------------------------------------------------------

#[test]
fn workspace_path_policy_rejects_relative_root() {
    let error = WorkspacePathPolicy::new("workspace").expect_err("relative root");
    assert_eq!(error.message, "sandbox workspace root must be absolute");
}

#[test]
fn absolute_workspace_path() {
    let policy = policy("/workspace");

    for (path, expected) in [
        ("pkg/file.py", "/workspace/pkg/file.py"),
        ("/workspace/pkg/file.py", "/workspace/pkg/file.py"),
        // A parent segment that stays inside the root resolves rather than being refused.
        ("/workspace/pkg/../file.py", "/workspace/file.py"),
        ("pkg/../secret.txt", "/workspace/secret.txt"),
    ] {
        assert_eq!(policy.absolute_workspace_path(path).unwrap(), expected);
    }

    for (path, rel, reason) in [
        ("/tmp/secret.txt", "/tmp/secret.txt", "absolute"),
        ("../secret.txt", "../secret.txt", "escape_root"),
        (
            "pkg/../../secret.txt",
            "pkg/../../secret.txt",
            "escape_root",
        ),
    ] {
        let error = policy.absolute_workspace_path(path).expect_err(path);
        assert_invalid_manifest_path(&error, rel, reason);
    }
}

#[test]
fn relative_path() {
    let policy = policy("/workspace");

    for (path, expected) in [
        ("pkg/file.py", "pkg/file.py"),
        ("/workspace/pkg/file.py", "pkg/file.py"),
        ("pkg/../secret.txt", "secret.txt"),
        ("/workspace", "."),
    ] {
        assert_eq!(policy.relative_path(path).unwrap(), expected);
    }

    // A provider-private root is never exposed in the relative form.
    let provider = WorkspacePathPolicy::new("/provider/private/root").expect("provider root");
    assert_eq!(
        provider
            .relative_path("/provider/private/root/images/dot.png")
            .unwrap(),
        "images/dot.png"
    );
    assert_eq!(
        provider.relative_path("images/dot.png").unwrap(),
        "images/dot.png"
    );

    for (path, rel, reason) in [
        ("/tmp/secret.txt", "/tmp/secret.txt", "absolute"),
        ("../secret.txt", "../secret.txt", "escape_root"),
    ] {
        let error = policy.relative_path(path).expect_err(path);
        assert_invalid_manifest_path(&error, rel, reason);
    }
}

#[test]
fn normalize_sandbox_path_uses_posix_paths() {
    let policy = policy("/workspace");

    assert_eq!(policy.sandbox_root(), "/workspace");
    assert_eq!(
        policy
            .normalize_sandbox_path("/workspace/pkg/file.py", false)
            .unwrap(),
        "/workspace/pkg/file.py"
    );
    assert_eq!(
        policy.normalize_sandbox_path("pkg/file.py", false).unwrap(),
        "/workspace/pkg/file.py"
    );
    assert_eq!(
        policy.absolute_workspace_path("pkg/file.py").unwrap(),
        "/workspace/pkg/file.py"
    );
}

#[test]
fn windows_drive_absolute_path_is_rejected_before_posix_coercion() {
    let policy = policy("/workspace");

    for path in ["C:/tmp/secret.txt", r"C:\tmp\secret.txt"] {
        for error in [
            policy.absolute_workspace_path(path).expect_err(path),
            policy.normalize_sandbox_path(path, false).expect_err(path),
            policy.relative_path(path).expect_err(path),
        ] {
            assert_invalid_manifest_path(&error, "C:/tmp/secret.txt", "absolute");
        }
    }
}

#[test]
fn absolute_workspace_path_rejects_windows_rooted_escape_as_absolute() {
    let policy = policy("/workspace");

    let error = policy
        .absolute_workspace_path("/tmp/secret.txt")
        .expect_err("outside the root");
    assert_invalid_manifest_path(&error, "/tmp/secret.txt", "absolute");
}

// ----------------------------------------------------------------------------------------------
// Host-filesystem resolution
// ----------------------------------------------------------------------------------------------

#[test]
fn normalize_path_with_symlink_resolution() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    let outside = temp.path().join("outside");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&outside).unwrap();

    let target = workspace.join("target.txt");
    std::fs::write(&target, "hello").unwrap();
    std::os::unix::fs::symlink(&target, workspace.join("link.txt")).unwrap();
    std::os::unix::fs::symlink(&outside, workspace.join("outside-link")).unwrap();

    // The policy root is itself reached through a symlink, as in the Python fixture.
    let alias = temp.path().join("workspace-alias");
    std::os::unix::fs::symlink(&workspace, &alias).unwrap();

    let policy = policy(alias.to_str().unwrap());
    let resolved_target = std::fs::canonicalize(&target).unwrap();
    let resolved_target = resolved_target.to_str().unwrap();

    for path in [
        "target.txt",
        "nested/../target.txt",
        // A safe internal leaf symlink resolves to what it points at.
        "link.txt",
    ] {
        assert_eq!(
            policy.normalize_host_path(path, false).unwrap(),
            resolved_target,
            "for `{path}`"
        );
    }
    // An absolute path through the root alias, and the already-resolved path, both work.
    assert_eq!(
        policy
            .normalize_host_path(alias.join("target.txt").to_str().unwrap(), false)
            .unwrap(),
        resolved_target
    );
    assert_eq!(
        policy
            .normalize_host_path(target.to_str().unwrap(), false)
            .unwrap(),
        resolved_target
    );

    // A symlinked parent that leaves the workspace is refused even though the lexical path
    // stays inside it — this is what the lexical check alone cannot see.
    let error = policy
        .normalize_host_path("outside-link/secret.txt", false)
        .expect_err("symlink parent escape");
    assert_invalid_manifest_path(&error, "outside-link/secret.txt", "escape_root");

    let secret = outside.join("secret.txt");
    let error = policy
        .normalize_host_path(secret.to_str().unwrap(), false)
        .expect_err("absolute path outside the root");
    assert_invalid_manifest_path(&error, secret.to_str().unwrap(), "absolute");
}

#[test]
fn host_io_rejects_write_under_resolved_read_only_extra_path_grant() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    let allowed = temp.path().join("allowed");
    let grant_alias = temp.path().join("allowed-alias");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&allowed).unwrap();
    std::os::unix::fs::symlink(&allowed, &grant_alias).unwrap();
    let target = allowed.join("cache.db");

    let grant = SandboxPathGrant::new(grant_alias.to_str().unwrap())
        .unwrap()
        .read_only(true);
    let grant_path = grant.path.clone();
    let policy =
        WorkspacePathPolicy::with_grants(workspace.to_str().unwrap(), vec![grant]).unwrap();

    let resolved_target = std::fs::canonicalize(&allowed).unwrap().join("cache.db");
    let resolved_target = resolved_target.to_str().unwrap().to_string();

    let error = policy
        .normalize_host_path(target.to_str().unwrap(), true)
        .expect_err("write under a read-only grant");
    assert_eq!(
        error.message,
        format!("failed to write archive for path: {resolved_target}")
    );
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        context(&error),
        ctx(&[
            ("path", resolved_target.as_str()),
            ("reason", "read_only_extra_path_grant"),
            ("grant_path", grant_path.as_str()),
        ])
    );
}

#[test]
fn host_io_rejects_extra_path_grant_symlink_to_root() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    let root_alias = temp.path().join("root-alias");
    std::fs::create_dir(&workspace).unwrap();
    std::os::unix::fs::symlink("/", &root_alias).unwrap();

    let policy = WorkspacePathPolicy::with_grants(
        workspace.to_str().unwrap(),
        vec![SandboxPathGrant::new(root_alias.to_str().unwrap()).unwrap()],
    )
    .unwrap();

    let error = policy
        .normalize_host_path(
            root_alias.join("etc").join("passwd").to_str().unwrap(),
            false,
        )
        .expect_err("a grant aliasing the filesystem root");
    assert_eq!(
        error.message,
        "sandbox path grant path must not resolve to filesystem root"
    );
}

#[test]
fn host_path_grant_rejects_symlink_to_root() {
    let temp = tempfile::tempdir().unwrap();
    let root_alias = temp.path().join("root-alias");
    std::os::unix::fs::symlink("/", &root_alias).unwrap();

    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .unwrap()
        .with_host_path(root_alias.to_str().unwrap())
        .unwrap();

    let error = sandbox_path_grant_host_path(&grant).expect_err("host path aliasing the root");
    assert_eq!(
        error.message,
        "sandbox path grant path must not resolve to filesystem root"
    );
}

#[test]
fn host_path_grant_returns_validated_resolved_source() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    let source_alias = temp.path().join("source-alias");
    std::fs::create_dir(&source).unwrap();
    std::os::unix::fs::symlink(&source, &source_alias).unwrap();

    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .unwrap()
        .with_host_path(source_alias.to_str().unwrap())
        .unwrap();

    let resolved = sandbox_path_grant_host_path(&grant).unwrap();

    // Re-pointing the alias afterwards does not change what was already validated.
    std::fs::remove_file(&source_alias).unwrap();
    std::os::unix::fs::symlink("/", &source_alias).unwrap();

    assert_eq!(
        resolved,
        std::fs::canonicalize(&source).unwrap().to_str().unwrap()
    );
}

// ----------------------------------------------------------------------------------------------
// Extra path grants
// ----------------------------------------------------------------------------------------------

#[test]
fn sandbox_extra_path_grant_rules_use_posix_paths() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![SandboxPathGrant::new("/tmp").unwrap()],
    )
    .unwrap();

    assert_eq!(
        policy.extra_path_grant_rules().unwrap(),
        vec![("/tmp".to_string(), false)]
    );
    assert_eq!(
        policy
            .normalize_sandbox_path("/tmp/result.txt", false)
            .unwrap(),
        "/tmp/result.txt"
    );
}

#[test]
fn extra_path_grant_rules_reject_windows_drive_absolute_path() {
    // Python reaches this through `model_construct`, which skips the field validator; the Rust
    // equivalent is building the struct directly rather than through the constructor.
    let grant = SandboxPathGrant {
        path: "C:/tmp".to_string(),
        read_only: false,
        description: None,
        host_path: None,
    };
    let policy = WorkspacePathPolicy::with_grants("/workspace", vec![grant]).unwrap();

    let error = policy
        .extra_path_grant_rules()
        .expect_err("a drive-qualified grant path");
    assert_eq!(
        error.message,
        "sandbox path grant path must be POSIX absolute"
    );
}

#[test]
fn extra_path_grant_rejects_non_native_windows_drive_absolute_path() {
    for path in ["C:/tmp", r"C:\tmp"] {
        let error = SandboxPathGrant::new(path).expect_err(path);
        assert_eq!(
            error.message,
            "sandbox path grant path must be POSIX absolute"
        );
    }
}

#[test]
fn extra_path_grant_rejects_relative_path() {
    let error = SandboxPathGrant::new("tmp").expect_err("a relative grant path");
    assert_eq!(
        error.message,
        "sandbox path grant path must be POSIX absolute"
    );
}

#[test]
fn extra_path_grant_rejects_root_path() {
    let error = SandboxPathGrant::new("/").expect_err("the filesystem root");
    assert_eq!(
        error.message,
        "sandbox path grant path must not be filesystem root"
    );
}

#[test]
fn extra_path_grant_rejects_root_alias_path() {
    let error = SandboxPathGrant::new("//").expect_err("the filesystem root under an alias");
    assert_eq!(
        error.message,
        "sandbox path grant path must not be filesystem root"
    );
}

#[test]
fn extra_path_grant_rejects_unsupported_host_paths() {
    for (host_path, message) in [
        ("relative/path", "must be an absolute host path"),
        ("/", "must not be filesystem root"),
        ("/srv/../secret", "must not contain parent segments"),
        ("//server/share", "does not support UNC or device paths"),
        (r"\\server\share", "does not support UNC or device paths"),
        (r"C:\", "must not be filesystem root"),
    ] {
        let error = SandboxPathGrant::new("/mnt/shared-data")
            .unwrap()
            .with_host_path(host_path)
            .expect_err(host_path);
        assert!(
            error.message.contains(message),
            "`{host_path}` gave `{}`",
            error.message
        );
        assert!(error.message.starts_with("sandbox path grant host_path "));
    }
}

#[test]
fn extra_path_grant_preserves_host_path_whitespace() {
    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .unwrap()
        .with_host_path("/srv/shared ")
        .unwrap();

    assert_eq!(grant.host_path.as_deref(), Some("/srv/shared "));
}

#[test]
fn extra_path_grant_accepts_absolute_path() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![SandboxPathGrant::new("/tmp").unwrap()],
    )
    .unwrap();

    assert_eq!(
        policy
            .normalize_sandbox_path("/tmp/result.txt", false)
            .unwrap(),
        "/tmp/result.txt"
    );
}

#[test]
fn extra_path_grant_rejects_ungranted_absolute_path() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![SandboxPathGrant::new("/tmp").unwrap()],
    )
    .unwrap();

    let error = policy
        .normalize_sandbox_path("/var/result.txt", false)
        .expect_err("a path under no grant");
    assert_invalid_manifest_path(&error, "/var/result.txt", "absolute");
}

#[test]
fn extra_path_grant_rejects_write_under_read_only_grant() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![
            SandboxPathGrant::new("/opt/toolchain")
                .unwrap()
                .read_only(true),
        ],
    )
    .unwrap();

    let error = policy
        .normalize_sandbox_path("/opt/toolchain/cache.db", true)
        .expect_err("a write under a read-only grant");
    assert_eq!(
        error.message,
        "failed to write archive for path: /opt/toolchain/cache.db"
    );
    assert_eq!(
        context(&error),
        ctx(&[
            ("path", "/opt/toolchain/cache.db"),
            ("reason", "read_only_extra_path_grant"),
            ("grant_path", "/opt/toolchain"),
        ])
    );
}

#[test]
fn extra_path_grant_allows_read_under_read_only_grant() {
    let policy = WorkspacePathPolicy::with_grants(
        "/workspace",
        vec![
            SandboxPathGrant::new("/opt/toolchain")
                .unwrap()
                .read_only(true),
        ],
    )
    .unwrap();

    assert_eq!(
        policy
            .normalize_sandbox_path("/opt/toolchain/cache.db", false)
            .unwrap(),
        "/opt/toolchain/cache.db"
    );
}

#[test]
fn recursive_remove_honours_a_nested_read_only_grant() {
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

    // Removing the tree would delete the read-only grant nested inside it.
    assert!(policy.validate_recursive_remove("/srv/data").is_err());
    // A sibling that holds no read-only grant is removable.
    policy
        .validate_recursive_remove("/srv/data/writable")
        .expect("a subtree with no read-only grant");
}

#[test]
fn manifest_records_extra_path_grants_in_order() {
    // Python asserts the pydantic `model_dump` of the grants. The Rust `Manifest` has no serde
    // representation, so the same contract is asserted on the accessor: the grants keep their
    // order, their normalized path, their access mode and their description, and a grant with
    // no host path does not invent one.
    let manifest = Manifest::new()
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp")
                .unwrap()
                .with_description("temporary files"),
        )
        .with_extra_path_grant(
            SandboxPathGrant::new("/opt/toolchain")
                .unwrap()
                .read_only(true)
                .with_description("compiler runtime"),
        );

    let grants = manifest.extra_path_grants();
    assert_eq!(grants.len(), 2);
    assert_eq!(grants[0].path, "/tmp");
    assert!(!grants[0].read_only);
    assert_eq!(grants[0].description.as_deref(), Some("temporary files"));
    assert_eq!(grants[0].host_path, None);
    assert_eq!(grants[1].path, "/opt/toolchain");
    assert!(grants[1].read_only);
    assert_eq!(grants[1].description.as_deref(), Some("compiler runtime"));
}

#[test]
fn extra_path_grant_normalizes_distinct_host_path() {
    // A drive-qualified host path is kept in Windows spelling, as `PureWindowsPath` renders it.
    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .unwrap()
        .with_host_path("C:/Users/example/shared-data")
        .unwrap()
        .read_only(true);

    assert_eq!(grant.path, "/mnt/shared-data");
    assert_eq!(
        grant.host_path.as_deref(),
        Some(r"C:\Users\example\shared-data")
    );
    assert!(grant.read_only);
}

#[test]
fn existing_host_root_rejects_windows_drive_absolute_paths() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let policy = policy(workspace.to_str().unwrap());
    assert!(policy.root_is_existing_host_path());

    for path in ["C:/tmp/secret.txt", r"C:\tmp\secret.txt"] {
        let errors = [
            policy.absolute_workspace_path(path).expect_err(path),
            policy.normalize_host_path(path, false).expect_err(path),
            policy.relative_path(path).expect_err(path),
        ];
        for error in errors {
            assert_invalid_manifest_path(&error, "C:/tmp/secret.txt", "absolute");
        }
    }
}

#[test]
fn extra_path_grant_keeps_two_leading_slashes_on_the_sandbox_path() {
    // `posixpath.normpath` keeps exactly two leading slashes and collapses three or more, and
    // Python's grant validator applies it to the sandbox-side `path`.
    assert_eq!(
        SandboxPathGrant::new("//server/share").unwrap().path,
        "//server/share"
    );
    assert_eq!(
        SandboxPathGrant::new("///srv/data").unwrap().path,
        "/srv/data"
    );
}
