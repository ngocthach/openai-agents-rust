//! Port of `tests/sandbox/test_runtime_helpers.py`: the helper scripts the SDK installs inside a
//! sandbox are run for real against temporary workspaces.
#![cfg(all(feature = "sandbox", unix))]

use std::os::unix::fs::{PermissionsExt as _, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use openai_agents_sdk::sandbox::runtime_helpers::{
    RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript, WORKSPACE_FINGERPRINT_HELPER,
};

fn install(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn resolve_helper(dir: &Path) -> PathBuf {
    install(
        dir,
        "resolve-workspace-path",
        RESOLVE_WORKSPACE_PATH_HELPER.content,
    )
}

fn fingerprint_helper(dir: &Path) -> PathBuf {
    install(
        dir,
        "workspace-fingerprint",
        WORKSPACE_FINGERPRINT_HELPER.content,
    )
}

fn run(helper: &Path, args: &[&str]) -> Output {
    Command::new(helper).args(args).output().unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn canon(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap()
}

/// Python: `test_runtime_helper_from_content_uses_posix_install_path`.
#[test]
fn runtime_helper_from_content_uses_a_posix_install_path() {
    let helper = RuntimeHelperScript::from_content("test-helper", "#!/bin/sh\nprintf 'ok\\n'");

    assert!(
        helper
            .install_path
            .starts_with("/tmp/openai-agents/bin/test-helper-"),
        "{}",
        helper.install_path
    );
}

/// Python: `test_workspace_fingerprint_helper_treats_exclusions_as_literal`.
#[test]
fn workspace_fingerprint_helper_treats_exclusions_as_literal() {
    let temp = tempfile::tempdir().unwrap();
    let root = canon(temp.path());
    let helper = fingerprint_helper(&root);
    let workspace = root.join("workspace");
    let excluded = workspace.join("cache[1]");
    let durable = workspace.join("cache1");
    std::fs::create_dir_all(&excluded).unwrap();
    std::fs::create_dir(&durable).unwrap();
    let excluded_file = excluded.join("remote.txt");
    let durable_file = durable.join("durable.txt");
    std::fs::write(&excluded_file, "remote-one").unwrap();
    std::fs::write(&durable_file, "durable-one").unwrap();
    let record = root.join("fingerprint.json");

    let fingerprint = || -> String {
        let output = run(
            &helper,
            &[
                workspace.to_str().unwrap(),
                "test-version",
                record.to_str().unwrap(),
                "manifest-digest",
                "cache[1]",
            ],
        );
        assert!(output.status.success(), "{}", stderr(&output));
        let value: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        value["fingerprint"].as_str().unwrap().to_string()
    };

    let first = fingerprint();
    std::fs::write(&excluded_file, "remote-two").unwrap();
    assert_eq!(fingerprint(), first);

    std::fs::write(&durable_file, "durable-two").unwrap();
    assert_ne!(fingerprint(), first);
}

/// Python: `test_workspace_fingerprint_detects_nested_paths_with_excluded_name`. Python runs it
/// on Linux only, because BSD `tar` has different exclusion semantics.
#[cfg(target_os = "linux")]
#[test]
fn workspace_fingerprint_detects_nested_paths_with_an_excluded_name() {
    let temp = tempfile::tempdir().unwrap();
    let root = canon(temp.path());
    let helper = fingerprint_helper(&root);
    let workspace = root.join("workspace");
    let excluded = workspace.join("data/scratch.txt");
    let durable = workspace.join("app/data/users.csv");
    std::fs::create_dir_all(excluded.parent().unwrap()).unwrap();
    std::fs::create_dir_all(durable.parent().unwrap()).unwrap();
    std::fs::write(&excluded, "scratch-one").unwrap();
    std::fs::write(&durable, "id,name").unwrap();
    let ordinary = workspace.join("app/main.py");
    std::fs::write(&ordinary, "print(1)").unwrap();
    let record = root.join("fingerprint.json");

    let fingerprint = || -> String {
        let output = run(
            &helper,
            &[
                workspace.to_str().unwrap(),
                "test-version",
                record.to_str().unwrap(),
                "manifest-digest",
                "data",
            ],
        );
        assert!(output.status.success(), "{}", stderr(&output));
        let value: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        value["fingerprint"].as_str().unwrap().to_string()
    };

    let first = fingerprint();
    assert_eq!(fingerprint(), first);
    std::fs::write(&excluded, "scratch-two").unwrap();
    assert_eq!(fingerprint(), first);

    std::fs::write(&durable, "ID,NAME").unwrap();
    let nested_changed = fingerprint();
    assert_ne!(nested_changed, first);

    std::fs::write(&ordinary, "print(2)").unwrap();
    assert_ne!(fingerprint(), nested_changed);
}

struct Tree {
    _temp: tempfile::TempDir,
    root: PathBuf,
    workspace: PathBuf,
    extra_root: PathBuf,
}

fn tree() -> Tree {
    let temp = tempfile::tempdir().unwrap();
    let root = canon(temp.path());
    let workspace = root.join("workspace");
    let extra_root = root.join("tmp");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::create_dir(&extra_root).unwrap();
    symlink(&extra_root, workspace.join("tmp-link")).unwrap();
    Tree {
        _temp: temp,
        root,
        workspace,
        extra_root,
    }
}

/// Python: `test_resolve_workspace_path_helper_allows_extra_root_symlink_target`.
#[test]
fn resolve_workspace_path_helper_allows_an_extra_root_symlink_target() {
    let tree = tree();
    let helper = resolve_helper(&tree.root);
    let target = tree.extra_root.join("result.txt");
    std::fs::write(&target, "scratch output").unwrap();
    let requested = tree.workspace.join("tmp-link/result.txt");

    let output = run(
        &helper,
        &[
            tree.workspace.to_str().unwrap(),
            requested.to_str().unwrap(),
            "0",
            tree.extra_root.to_str().unwrap(),
            "0",
        ],
    );

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(stdout(&output), format!("{}\n", canon(&target).display()));
    assert_eq!(stderr(&output), "");
}

/// Python: `test_resolve_workspace_path_helper_rejects_extra_root_when_not_allowed`.
#[test]
fn resolve_workspace_path_helper_rejects_an_extra_root_when_not_allowed() {
    let tree = tree();
    let helper = resolve_helper(&tree.root);
    let target = tree.extra_root.join("result.txt");
    std::fs::write(&target, "scratch output").unwrap();
    let requested = tree.workspace.join("tmp-link/result.txt");

    let output = run(
        &helper,
        &[
            tree.workspace.to_str().unwrap(),
            requested.to_str().unwrap(),
            "0",
        ],
    );

    assert_eq!(output.status.code(), Some(111));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        format!("workspace escape: {}\n", canon(&target).display())
    );
}

/// Python: `test_resolve_workspace_path_helper_rejects_extra_root_symlink_to_root`.
#[test]
fn resolve_workspace_path_helper_rejects_an_extra_root_symlink_to_root() {
    let tree = tree();
    let helper = resolve_helper(&tree.root);
    let root_alias = tree.root.join("root-alias");
    symlink("/", &root_alias).unwrap();

    let output = run(
        &helper,
        &[
            tree.workspace.to_str().unwrap(),
            "/etc/passwd",
            "0",
            root_alias.to_str().unwrap(),
            "0",
        ],
    );

    assert_eq!(output.status.code(), Some(113));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        format!(
            "extra path grant must not resolve to filesystem root: {}\n",
            root_alias.display()
        )
    );
}

/// Python: `test_resolve_workspace_path_helper_rejects_nested_read_only_extra_grant_on_write`.
#[test]
fn resolve_workspace_path_helper_rejects_a_nested_read_only_extra_grant_on_write() {
    let tree = tree();
    let helper = resolve_helper(&tree.root);
    let protected_root = tree.extra_root.join("protected");
    std::fs::create_dir(&protected_root).unwrap();
    let target = protected_root.join("result.txt");
    std::fs::write(&target, "scratch output").unwrap();
    let requested = tree.workspace.join("tmp-link/protected/result.txt");

    let output = run(
        &helper,
        &[
            tree.workspace.to_str().unwrap(),
            requested.to_str().unwrap(),
            "1",
            tree.extra_root.to_str().unwrap(),
            "0",
            protected_root.to_str().unwrap(),
            "1",
        ],
    );

    assert_eq!(output.status.code(), Some(114));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        format!(
            "read-only extra path grant: {}\nresolved path: {}\n",
            protected_root.display(),
            canon(&target).display()
        )
    );
}

/// Python: `test_resolve_workspace_path_helper_allows_nested_read_only_extra_grant_on_read`.
#[test]
fn resolve_workspace_path_helper_allows_a_nested_read_only_extra_grant_on_read() {
    let tree = tree();
    let helper = resolve_helper(&tree.root);
    let protected_root = tree.extra_root.join("protected");
    std::fs::create_dir(&protected_root).unwrap();
    let target = protected_root.join("result.txt");
    std::fs::write(&target, "scratch output").unwrap();
    let requested = tree.workspace.join("tmp-link/protected/result.txt");

    let output = run(
        &helper,
        &[
            tree.workspace.to_str().unwrap(),
            requested.to_str().unwrap(),
            "0",
            tree.extra_root.to_str().unwrap(),
            "0",
            protected_root.to_str().unwrap(),
            "1",
        ],
    );

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(stdout(&output), format!("{}\n", canon(&target).display()));
    assert_eq!(stderr(&output), "");
}
