//! Archive member validation, the `extract()` API and the sandbox git helpers.
//!
//! The fixtures are written as raw `ustar` blocks because the point of most of them is a member
//! an archive writer refuses to produce: a `..` path, an absolute path, a hardlink or a device
//! node. Each one is fed to the real `unix_local` backend, and the assertion is that nothing
//! reached the workspace.

#![cfg(feature = "sandbox")]

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use openai_agents_sdk::sandbox::archive::{CompressionScheme, TarValidationOptions};
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::testing::{
    SandboxCallResult, SandboxMethod, SandboxStep, scripted_sandbox_session,
};
use openai_agents_sdk::sandbox::types::{ExecResult, SandboxArchiveLimits};
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSession};

const TYPE_FILE: u8 = b'0';
const TYPE_HARDLINK: u8 = b'1';
const TYPE_SYMLINK: u8 = b'2';
const TYPE_CHAR_DEVICE: u8 = b'3';
const TYPE_DIR: u8 = b'5';

/// One 512-byte `ustar` header.
fn tar_header(name: &str, size: u64, type_flag: u8, link_name: &str, mode: u32) -> [u8; 512] {
    fn put(header: &mut [u8; 512], offset: usize, bytes: &[u8]) {
        header[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    let mut header = [0u8; 512];
    put(&mut header, 0, name.as_bytes());
    put(&mut header, 100, format!("{mode:07o}\0").as_bytes());
    put(&mut header, 108, b"0000000\0");
    put(&mut header, 116, b"0000000\0");
    put(&mut header, 124, format!("{size:011o}\0").as_bytes());
    put(&mut header, 136, b"00000000000\0");
    put(&mut header, 148, b"        ");
    header[156] = type_flag;
    put(&mut header, 157, link_name.as_bytes());
    put(&mut header, 257, b"ustar\0");
    put(&mut header, 263, b"00");
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    put(&mut header, 148, format!("{checksum:06o}\0 ").as_bytes());
    header
}

/// One archive member, header plus padded payload.
fn tar_member(name: &str, type_flag: u8, link_name: &str, contents: &[u8], mode: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tar_header(
        name,
        contents.len() as u64,
        type_flag,
        link_name,
        mode,
    ));
    out.extend_from_slice(contents);
    out.extend(std::iter::repeat_n(0u8, (512 - contents.len() % 512) % 512));
    out
}

/// Close an archive with the two zero blocks that end one.
fn tar_archive(members: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out: Vec<u8> = members.into_iter().flatten().collect();
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}

fn file_member(name: &str, contents: &str) -> Vec<u8> {
    tar_member(name, TYPE_FILE, "", contents.as_bytes(), 0o644)
}

fn zip_archive(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    for (name, contents) in entries {
        writer.start_file(*name, options).expect("zip entry");
        writer.write_all(contents.as_bytes()).expect("zip payload");
    }
    writer.finish().expect("zip archive").into_inner()
}

/// A started session over a fresh temporary workspace.
async fn session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.expect("session");
    session.start().await.expect("start");
    (client, session)
}

fn workspace_entries(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .expect("listing")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn hydrate_refuses_unsafe_members_before_anything_is_written() {
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "traversal",
            tar_archive(vec![file_member("../escape.txt", "x")]),
            "parent traversal",
        ),
        (
            "absolute",
            tar_archive(vec![file_member("/etc/passwd", "x")]),
            "absolute path",
        ),
        (
            "hardlink",
            tar_archive(vec![
                file_member("real.txt", "x"),
                tar_member("hard.txt", TYPE_HARDLINK, "real.txt", b"", 0o644),
            ]),
            "hardlink member not allowed",
        ),
        (
            "device",
            tar_archive(vec![tar_member("null", TYPE_CHAR_DEVICE, "", b"", 0o666)]),
            "unsupported member type",
        ),
        (
            // Python passes `allow_external_symlink_targets=False` at every hydrate site, so a
            // virtualenv-style link to the host is refused rather than restored.
            "absolute symlink target",
            tar_archive(vec![
                tar_member("bin", TYPE_DIR, "", b"", 0o755),
                tar_member("bin/python", TYPE_SYMLINK, "/usr/bin/python3", b"", 0o777),
            ]),
            "absolute symlink target not allowed: /usr/bin/python3",
        ),
        (
            "symlink target leaving the archive",
            tar_archive(vec![tar_member(
                "escape",
                TYPE_SYMLINK,
                "../outside",
                b"",
                0o777,
            )]),
            "symlink target escapes archive root: ../outside",
        ),
        (
            "path through a symlink",
            tar_archive(vec![
                tar_member("link", TYPE_SYMLINK, "sub", b"", 0o777),
                file_member("link/payload.txt", "x"),
            ]),
            "archive path descends through symlink: link",
        ),
        (
            "duplicate",
            tar_archive(vec![
                file_member("a.txt", "one"),
                file_member("a.txt", "two"),
            ]),
            "duplicate archive path: a.txt",
        ),
    ];

    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());
    let before = workspace_entries(&root);
    for (label, archive, reason) in cases {
        let error = session.hydrate_workspace(&archive).await.expect_err(label);
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some(reason),
            "{label}"
        );
    }
    assert_eq!(workspace_entries(&root), before);
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn hydrate_reports_the_member_and_the_reason() {
    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());
    let archive = tar_archive(vec![file_member("../escape.txt", "x")]);
    let error = session.hydrate_workspace(&archive).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("parent traversal")
    );
    assert_eq!(
        error.context.get("member").map(String::as_str),
        Some("../escape.txt")
    );
    // Nothing escaped, and nothing landed in the workspace either.
    assert!(!root.parent().unwrap().join("escape.txt").exists());
    assert_eq!(workspace_entries(&root), Vec::<String>::new());
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn hydrate_restores_files_directories_and_links() {
    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());
    let archive = tar_archive(vec![
        tar_member("sub", TYPE_DIR, "", b"", 0o755),
        tar_member("sub/run.sh", TYPE_FILE, "", b"#!/bin/sh\n", 0o755),
        file_member("sub/notes.txt", "hello"),
        tar_member("alias", TYPE_SYMLINK, "sub/notes.txt", b"", 0o777),
        tar_member("sub/up", TYPE_SYMLINK, "../sub/notes.txt", b"", 0o777),
    ]);
    session.hydrate_workspace(&archive).await.expect("hydrate");

    assert_eq!(
        std::fs::read_to_string(root.join("sub/notes.txt")).unwrap(),
        "hello"
    );
    assert_eq!(
        std::fs::read_link(root.join("alias")).unwrap(),
        std::path::PathBuf::from("sub/notes.txt")
    );
    // A link that climbs out of its own directory but stays inside the archive is restored too.
    assert_eq!(
        std::fs::read_link(root.join("sub/up")).unwrap(),
        std::path::PathBuf::from("../sub/notes.txt")
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(root.join("sub/run.sh"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755, "an executable script stays executable");
        let mode = std::fs::metadata(root.join("sub/notes.txt"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o644);
    }
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn hydrate_refuses_a_member_written_through_an_existing_symlink() {
    let outside = tempfile::tempdir().unwrap();
    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());
    std::os::unix::fs::symlink(outside.path(), root.join("link")).unwrap();

    let archive = tar_archive(vec![file_member("link/payload.txt", "x")]);
    let error = session.hydrate_workspace(&archive).await.unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("path escapes root after resolution")
    );
    assert!(!outside.path().join("payload.txt").exists());

    // A link that stays inside the workspace is refused too: the archive did not create it, so
    // where it points is not something the archive proved.
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink("real", root.join("inner")).unwrap();
    let archive = tar_archive(vec![file_member("inner/payload.txt", "x")]);
    let error = session.hydrate_workspace(&archive).await.unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("symlink in parent path")
    );
    assert!(!root.join("real/payload.txt").exists());
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn extract_writes_a_tar_and_a_zip_into_the_workspace() {
    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());

    let archive = tar_archive(vec![
        tar_member("sub", TYPE_DIR, "", b"", 0o755),
        file_member("sub/a.txt", "one"),
    ]);
    session
        .extract("bundle.tar", &archive, None)
        .await
        .expect("tar extract");
    assert_eq!(
        std::fs::read_to_string(root.join("sub/a.txt")).unwrap(),
        "one"
    );
    // The archive itself is written where the caller asked for it, as Python does.
    assert!(root.join("bundle.tar").exists());

    let zipped = zip_archive(&[("zipped/b.txt", "two")]);
    session
        .extract("bundle.zip", &zipped, None)
        .await
        .expect("zip extract");
    assert_eq!(
        std::fs::read_to_string(root.join("zipped/b.txt")).unwrap(),
        "two"
    );
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn extract_refuses_a_zip_slip_member_and_a_symlink_member() {
    let (client, session) = session().await;
    let root = std::path::PathBuf::from(session.workspace_root());

    let zipped = zip_archive(&[("../escape.txt", "x")]);
    let error = session
        .extract("bundle.zip", &zipped, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("parent traversal")
    );
    assert!(!root.parent().unwrap().join("escape.txt").exists());

    // `extract` refuses symlink members outright, where a workspace restore keeps them.
    let archive = tar_archive(vec![tar_member("alias", TYPE_SYMLINK, "a.txt", b"", 0o777)]);
    let error = session
        .extract("bundle.tar", &archive, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("symlink member not allowed")
    );
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn extract_refuses_an_unknown_compression_scheme() {
    let (client, session) = session().await;
    let error = session
        .extract("bundle.bin", &tar_archive(vec![]), None)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidCompressionScheme);
    // An explicit scheme still wins over the extension.
    session
        .extract(
            "bundle.bin",
            &tar_archive(vec![file_member("a.txt", "one")]),
            Some(CompressionScheme::Tar),
        )
        .await
        .expect("explicit scheme");
    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn extract_enforces_the_archive_limits() {
    let (client, session) = session().await;
    let archive = tar_archive(vec![
        file_member("a.txt", "one"),
        file_member("b.txt", "two"),
        file_member("c.txt", "three"),
    ]);

    session.set_archive_limits(Some(
        SandboxArchiveLimits::new()
            .with_max_input_bytes(None)
            .with_max_extracted_bytes(None)
            .with_max_members(Some(2)),
    ));
    let error = session
        .extract("bundle.tar", &archive, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("archive member count exceeds limit")
    );
    assert_eq!(error.context.get("limit").map(String::as_str), Some("2"));

    session.set_archive_limits(Some(
        SandboxArchiveLimits::new()
            .with_max_input_bytes(None)
            .with_max_extracted_bytes(Some(4))
            .with_max_members(None),
    ));
    let error = session
        .extract("bundle.tar", &archive, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("archive extracted size exceeds limit")
    );

    session.set_archive_limits(Some(
        SandboxArchiveLimits::new().with_max_input_bytes(Some(16)),
    ));
    let error = session
        .extract("bundle.tar", &archive, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("archive input size exceeds limit")
    );

    // Without limits the same archive extracts.
    session.set_archive_limits(None);
    session
        .extract("bundle.tar", &archive, None)
        .await
        .expect("no limits");
    client.delete(session).await.unwrap();
}

#[test]
fn archive_limits_reject_a_zero_ceiling() {
    for limits in [
        SandboxArchiveLimits::new().with_max_input_bytes(Some(0)),
        SandboxArchiveLimits::new().with_max_extracted_bytes(Some(0)),
        SandboxArchiveLimits::new().with_max_members(Some(0)),
    ] {
        let error = limits.validate().unwrap_err();
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
        assert!(error.message.contains("must be at least 1"), "{error}");
    }
    assert!(SandboxArchiveLimits::new().validate().is_ok());
    assert!(
        SandboxArchiveLimits::new()
            .with_max_members(None)
            .validate()
            .is_ok()
    );
}

#[tokio::test]
async fn a_gzipped_workspace_archive_round_trips_through_validation() {
    let source = UnixLocalSandboxClient::new();
    let session = source
        .create(Some(Manifest::new().with_entry(
            "a.txt",
            openai_agents_sdk::sandbox::entries::Entry::text_file("one"),
        )))
        .await
        .unwrap();
    session.start().await.unwrap();
    let archive = session.persist_workspace().await.unwrap();
    // `tar -czf` output passes the in-process validator, so the backend's own archives hydrate.
    openai_agents_sdk::sandbox::archive::validate_tar_bytes(
        &archive,
        &TarValidationOptions::hydrate(),
    )
    .expect("a workspace archive is valid");

    let target = source.create(None).await.unwrap();
    target.start().await.unwrap();
    target.hydrate_workspace(&archive).await.unwrap();
    assert_eq!(
        String::from_utf8(target.read_file("a.txt").await.unwrap()).unwrap(),
        "one"
    );
    source.delete(session).await.unwrap();
    source.delete(target).await.unwrap();
}

#[tokio::test]
async fn clone_repo_falls_back_to_a_pinned_checkout_for_a_sha() {
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.arguments.iter().any(|a| a.contains("command -v git")))
            .returning_exec(ExecResult::default()),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| {
                call.arguments
                    .iter()
                    .any(|a| a.contains("git clone --depth 1 --no-tags --branch"))
            })
            .returning_exec(ExecResult {
                exit_code: 128,
                stdout: Vec::new(),
                stderr: b"Remote branch not found".to_vec(),
            }),
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| {
                call.arguments
                    .iter()
                    .any(|a| a.contains("git clone --no-checkout") && a.contains("checkout"))
            })
            .returning_exec(ExecResult::default()),
    ]);
    openai_agents_sdk::sandbox::util::clone_repo(&session, "openai/codex", "abc1234", "repo")
        .await
        .expect("clone");
    session.assert_complete();
}

#[tokio::test]
async fn clone_repo_reports_a_missing_git() {
    let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec).returning(
        SandboxCallResult::Exec(ExecResult {
            exit_code: 127,
            stdout: Vec::new(),
            stderr: Vec::new(),
        }),
    )]);
    let error = openai_agents_sdk::sandbox::util::ensure_git_available(&session)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::GitMissingInImage);
}
