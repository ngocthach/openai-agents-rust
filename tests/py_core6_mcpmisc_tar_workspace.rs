//! Parity port of `tests/sandbox/test_tar_workspace.py`: the `--exclude` arguments a workspace
//! archive uses are rooted at `.`, so excluding `data` never drops `app/data`.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "sandbox")]

use std::io::Read;
use std::path::Path;
use std::process::Command;

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{shell_quote, shell_tar_exclude_args};

#[test]
fn shell_tar_exclude_args_skips_empty_and_dot_paths() {
    let skipped: Vec<&Path> = vec![Path::new(""), Path::new("."), Path::new("/")];
    assert_eq!(shell_tar_exclude_args(skipped), Vec::<String>::new());
}

#[test]
fn shell_tar_exclude_args_sorts_and_roots_patterns() {
    assert_eq!(
        shell_tar_exclude_args([
            Path::new("logs/events.jsonl"),
            Path::new("cache dir/file.txt"),
        ]),
        [
            "--exclude='./cache dir/file.txt'",
            "--exclude=./logs/events.jsonl",
        ]
    );
}

#[test]
fn shell_tar_exclude_args_normalizes_absolute_paths() {
    assert_eq!(
        shell_tar_exclude_args([Path::new("/tmp/workspace/cache")]),
        ["--exclude=./tmp/workspace/cache"]
    );
}

/// The path of a GNU `tar`, the only implementation whose `--exclude` semantics the cloud
/// workspace archives rely on (BSD tar excludes differently).
fn gnu_tar() -> Option<&'static str> {
    ["tar", "gtar"].into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("GNU tar"))
    })
}

#[test]
fn workspace_archive_preserves_nested_paths_with_excluded_name() {
    let Some(tar) = gnu_tar() else {
        eprintln!("skipped: cloud workspace archives use GNU tar; BSD tar excludes differently");
        return;
    };
    for skip_path in ["data", "cache/data", "cache dir"] {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("workspace");
        let excluded = workspace.join(skip_path).join("scratch.txt");
        let durable = workspace.join("app").join(skip_path).join("users.csv");
        std::fs::create_dir_all(excluded.parent().unwrap()).unwrap();
        std::fs::create_dir_all(durable.parent().unwrap()).unwrap();
        std::fs::write(&excluded, "scratch").unwrap();
        std::fs::write(&durable, "id,name").unwrap();
        std::fs::write(workspace.join("app/main.py"), "print(1)").unwrap();

        let manifest = Manifest::new()
            .with_root(workspace.to_str().unwrap())
            .with_entry(skip_path, Entry::dir().with_ephemeral(true));
        let skipped: Vec<std::path::PathBuf> = manifest
            .ephemeral_persistence_paths()
            .into_iter()
            .map(Into::into)
            .collect();
        let excludes = shell_tar_exclude_args(&skipped).join(" ");
        let archive_path = tmp.path().join("workspace.tar");
        let command = format!(
            "{tar} {excludes} -C {} -cf {} .",
            shell_quote(workspace.to_str().unwrap()),
            shell_quote(archive_path.to_str().unwrap()),
        );
        let status = Command::new("sh").arg("-c").arg(&command).status().unwrap();
        assert!(status.success(), "{command}");

        let mut archive = tar::Archive::new(std::fs::File::open(&archive_path).unwrap());
        let mut names = Vec::new();
        let mut contents = std::collections::BTreeMap::new();
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().to_string_lossy().into_owned();
            let mut body = Vec::new();
            entry.read_to_end(&mut body).unwrap();
            contents.insert(name.clone(), body);
            names.push(name);
        }
        assert!(
            !names.contains(&format!("./{skip_path}/scratch.txt")),
            "{skip_path}: {names:?}"
        );
        assert_eq!(
            contents[&format!("./app/{skip_path}/users.csv")],
            b"id,name",
            "{skip_path}"
        );
        assert_eq!(contents["./app/main.py"], b"print(1)", "{skip_path}");
    }
}
