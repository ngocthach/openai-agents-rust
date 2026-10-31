//! Parity port of `tests/sandbox/test_extract.py` from the Python SDK.
//!
//! Every case runs against the real `unix_local` backend, as Python's do: the point of most of
//! them is that a member which should never reach the workspace does not, so a fake that only
//! records calls would not prove anything. Archives are built as raw `ustar` blocks and real
//! zips, because the interesting members are ones an archive writer refuses to produce.

#![cfg(feature = "sandbox")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openai_agents_sdk::sandbox::archive::CompressionScheme;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::types::{
    DEFAULT_MAX_ARCHIVE_EXTRACTED_BYTES, DEFAULT_MAX_ARCHIVE_INPUT_BYTES,
    DEFAULT_MAX_ARCHIVE_MEMBERS, SandboxArchiveLimits,
};
use openai_agents_sdk::sandbox::{SandboxClient, SandboxError, SandboxSession};

// ----------------------------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------------------------

const TYPE_FILE: u8 = b'0';
const TYPE_SYMLINK: u8 = b'2';
const TYPE_DIR: u8 = b'5';

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

fn tar_archive(members: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out: Vec<u8> = members.into_iter().flatten().collect();
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}

fn tar_file(name: &str, contents: &[u8]) -> Vec<u8> {
    tar_member(name, TYPE_FILE, "", contents, 0o644)
}

fn tar_bytes(members: &[(&str, &[u8])]) -> Vec<u8> {
    tar_archive(
        members
            .iter()
            .map(|(name, payload)| tar_file(name, payload))
            .collect(),
    )
}

fn zip_bytes(members: &[(&str, &[u8])]) -> Vec<u8> {
    use std::io::Write as _;

    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    for (name, payload) in members {
        writer.start_file(*name, options).expect("zip member");
        writer.write_all(payload).expect("zip payload");
    }
    writer.finish().expect("zip finish").into_inner()
}

/// A started session over a fresh temporary workspace.
async fn session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.expect("session");
    session.start().await.expect("start");
    (client, session)
}

fn root_of(session: &Arc<dyn SandboxSession>) -> PathBuf {
    PathBuf::from(session.workspace_root())
}

/// Assert `extract` refused a member, naming the member and the reason Python names.
fn assert_refused_member(error: &SandboxError, member: &str, reason: &str) {
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("member").map(String::as_str),
        Some(member),
        "message was `{}`",
        error.message
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some(reason),
        "message was `{}`",
        error.message
    );
}

// ----------------------------------------------------------------------------------------------
// Happy paths
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn extract_tar_writes_archive_and_unpacks_contents() {
    let (_client, session) = session().await;
    session
        .extract(
            "bundle.tar",
            &tar_bytes(&[("nested/hello.txt", b"hello from tar")]),
            None,
        )
        .await
        .expect("extract");

    let root = root_of(&session);
    // The archive itself stays where the caller asked for it, beside its own members.
    assert!(root.join("bundle.tar").is_file());
    assert_eq!(
        std::fs::read_to_string(root.join("nested/hello.txt")).unwrap(),
        "hello from tar"
    );
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_zip_writes_archive_and_unpacks_contents() {
    let (_client, session) = session().await;
    session
        .extract(
            "bundle.zip",
            &zip_bytes(&[("nested/hello.txt", b"hello from zip")]),
            None,
        )
        .await
        .expect("extract");

    let root = root_of(&session);
    assert!(root.join("bundle.zip").is_file());
    assert_eq!(
        std::fs::read_to_string(root.join("nested/hello.txt")).unwrap(),
        "hello from zip"
    );
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_default_archive_limits_none_preserves_no_resource_limit_behavior() {
    let (_client, session) = session().await;
    // The session default is `None`: no SDK ceiling, so an ordinary archive is unaffected.
    assert!(session.archive_limits().is_none());
    session
        .extract(
            "bundle.tar",
            &tar_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
            None,
        )
        .await
        .expect("extract");

    let root = root_of(&session);
    assert_eq!(std::fs::read_to_string(root.join("one.txt")).unwrap(), "1");
    assert_eq!(std::fs::read_to_string(root.join("two.txt")).unwrap(), "2");
    session.shutdown().await.expect("shutdown");
}

// ----------------------------------------------------------------------------------------------
// Compression scheme
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn extract_archive_rejects_missing_compression_scheme() {
    let (_client, session) = session().await;

    let error = session
        .extract("bundle.unknown", &tar_bytes(&[("a.txt", b"a")]), None)
        .await
        .expect_err("an extension that names no scheme");
    assert_eq!(error.code, ErrorCode::InvalidCompressionScheme);
    assert_eq!(
        error.context.get("path").map(String::as_str),
        Some("bundle.unknown")
    );

    // Naming the scheme explicitly is what makes the same path extractable.
    session
        .extract(
            "bundle.unknown",
            &tar_bytes(&[("a.txt", b"a")]),
            Some(CompressionScheme::Tar),
        )
        .await
        .expect("an explicit scheme");
    session.shutdown().await.expect("shutdown");
}

#[test]
fn compression_scheme_is_inferred_from_the_path() {
    assert_eq!(
        CompressionScheme::infer_from_path("bundle.tar"),
        Some(CompressionScheme::Tar)
    );
    assert_eq!(
        CompressionScheme::infer_from_path("bundle.zip"),
        Some(CompressionScheme::Zip)
    );
    assert_eq!(CompressionScheme::infer_from_path("bundle.unknown"), None);
}

// ----------------------------------------------------------------------------------------------
// Resource limits
// ----------------------------------------------------------------------------------------------

#[test]
fn sandbox_archive_limits_defaults_enable_sdk_thresholds() {
    let limits = SandboxArchiveLimits::default();

    assert_eq!(limits.max_input_bytes, Some(1024 * 1024 * 1024));
    assert_eq!(limits.max_extracted_bytes, Some(4 * 1024 * 1024 * 1024));
    assert_eq!(limits.max_members, Some(100_000));
    // The same numbers, named.
    assert_eq!(DEFAULT_MAX_ARCHIVE_INPUT_BYTES, 1024 * 1024 * 1024);
    assert_eq!(DEFAULT_MAX_ARCHIVE_EXTRACTED_BYTES, 4 * 1024 * 1024 * 1024);
    assert_eq!(DEFAULT_MAX_ARCHIVE_MEMBERS, 100_000);
}

#[test]
fn sandbox_archive_limits_rejects_non_positive_values() {
    for limits in [
        SandboxArchiveLimits {
            max_input_bytes: Some(0),
            ..SandboxArchiveLimits::default()
        },
        SandboxArchiveLimits {
            max_extracted_bytes: Some(0),
            ..SandboxArchiveLimits::default()
        },
        SandboxArchiveLimits {
            max_members: Some(0),
            ..SandboxArchiveLimits::default()
        },
    ] {
        assert!(limits.validate().is_err(), "{limits:?}");
    }
    SandboxArchiveLimits::default()
        .validate()
        .expect("the defaults are valid");
}

#[tokio::test]
async fn extract_rejects_archive_input_over_limit() {
    let (_client, session) = session().await;
    let limits = SandboxArchiveLimits {
        max_input_bytes: Some(4),
        max_extracted_bytes: None,
        max_members: None,
    };

    // Five bytes against a four-byte ceiling: the check is on the input size alone, so the
    // payload does not have to be a valid archive.
    let error = session
        .extract_with_limits("bundle.zip", b"12345", None, Some(limits))
        .await
        .expect_err("an archive over the input ceiling");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("archive input size exceeds limit")
    );
    assert_eq!(error.context.get("limit").map(String::as_str), Some("4"));
    assert_eq!(error.context.get("actual").map(String::as_str), Some("5"));
    // Nothing was written: the ceiling is checked before the archive lands.
    assert!(!root_of(&session).join("bundle.zip").exists());
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_rejects_extracted_bytes_over_limit() {
    for (name, archive) in [
        ("bundle.tar", tar_bytes(&[("large.txt", b"12345")])),
        ("bundle.zip", zip_bytes(&[("large.txt", b"12345")])),
    ] {
        let (_client, session) = session().await;
        let limits = SandboxArchiveLimits {
            max_input_bytes: None,
            max_extracted_bytes: Some(4),
            max_members: None,
        };

        let error = session
            .extract_with_limits(name, &archive, None, Some(limits))
            .await
            .expect_err(name);
        assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError, "{name}");
        assert_eq!(
            error.context.get("member").map(String::as_str),
            Some("large.txt"),
            "{name}"
        );
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("archive extracted size exceeds limit"),
            "{name}"
        );
        assert_eq!(error.context.get("limit").map(String::as_str), Some("4"));
        assert_eq!(error.context.get("actual").map(String::as_str), Some("5"));
        assert!(!root_of(&session).join("large.txt").exists(), "{name}");
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_rejects_member_count_over_limit() {
    for (name, archive) in [
        (
            "bundle.tar",
            tar_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
        ),
        (
            "bundle.zip",
            zip_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
        ),
    ] {
        let (_client, session) = session().await;
        let limits = SandboxArchiveLimits {
            max_input_bytes: None,
            max_extracted_bytes: None,
            max_members: Some(1),
        };

        let error = session
            .extract_with_limits(name, &archive, None, Some(limits))
            .await
            .expect_err(name);
        assert_eq!(
            error.context.get("member").map(String::as_str),
            Some("two.txt"),
            "{name}"
        );
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("archive member count exceeds limit"),
            "{name}"
        );
        assert_eq!(error.context.get("limit").map(String::as_str), Some("1"));
        assert_eq!(error.context.get("actual").map(String::as_str), Some("2"));
        assert!(!root_of(&session).join("two.txt").exists(), "{name}");
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_archive_limits_none_disables_only_selected_limits() {
    let (_client, session) = session().await;
    // Only the member count is capped; the other two are off, so a big declared size passes.
    let limits = SandboxArchiveLimits {
        max_input_bytes: None,
        max_extracted_bytes: None,
        max_members: Some(10),
    };

    session
        .extract_with_limits(
            "bundle.tar",
            &tar_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
            None,
            Some(limits),
        )
        .await
        .expect("limits that are off do not apply");

    let root = root_of(&session);
    assert_eq!(std::fs::read_to_string(root.join("one.txt")).unwrap(), "1");
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_archive_limits_per_call_override_session_default() {
    let (_client, session) = session().await;
    let archive = tar_bytes(&[("a.txt", b"a"), ("b.txt", b"b")]);

    // The session default is `None`, and a per-call ceiling is what refuses the archive.
    session
        .extract("default.tar", &archive, None)
        .await
        .expect("the session default imposes nothing");
    let error = session
        .extract_with_limits(
            "capped.tar",
            &archive,
            None,
            Some(SandboxArchiveLimits {
                max_input_bytes: None,
                max_extracted_bytes: None,
                max_members: Some(1),
            }),
        )
        .await
        .expect_err("a per-call ceiling");
    assert_eq!(error.context.get("limit").map(String::as_str), Some("1"));
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_rejects_invalid_per_call_archive_limits() {
    let (_client, session) = session().await;

    let error = session
        .extract_with_limits(
            "bundle.tar",
            &tar_bytes(&[("a.txt", b"a")]),
            None,
            Some(SandboxArchiveLimits {
                max_input_bytes: Some(0),
                ..SandboxArchiveLimits::default()
            }),
        )
        .await
        .expect_err("a non-positive ceiling");
    assert!(
        error.message.contains("max_input_bytes"),
        "{}",
        error.message
    );
    // The archive never landed, because the limits are validated first.
    assert!(!root_of(&session).join("bundle.tar").exists());
    session.shutdown().await.expect("shutdown");
}

// ----------------------------------------------------------------------------------------------
// Member-path refusals — nothing reaches the workspace
// ----------------------------------------------------------------------------------------------

#[tokio::test]
async fn extract_rejects_windows_drive_member_paths() {
    for (name, archive) in [
        (
            "bundle.tar",
            tar_archive(vec![tar_file("C:/tmp/evil.txt", b"evil")]),
        ),
        ("bundle.zip", zip_bytes(&[("C:/tmp/evil.txt", b"evil")])),
    ] {
        let (_client, session) = session().await;
        let error = session.extract(name, &archive, None).await.expect_err(name);
        assert_refused_member(&error, "C:/tmp/evil.txt", "windows drive path");
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_rejects_windows_separator_member_paths() {
    for (name, archive) in [
        (
            "bundle.tar",
            tar_archive(vec![tar_file(r"nested\evil.txt", b"evil")]),
        ),
        ("bundle.zip", zip_bytes(&[(r"nested\evil.txt", b"evil")])),
    ] {
        let (_client, session) = session().await;
        let error = session.extract(name, &archive, None).await.expect_err(name);
        assert_refused_member(&error, r"nested\evil.txt", "windows path separator");
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_rejects_member_under_non_directory_member() {
    for (name, archive) in [
        (
            "bundle.tar",
            tar_archive(vec![
                tar_file("nested", b"not a directory"),
                tar_file("nested/hello.txt", b"hello"),
            ]),
        ),
        (
            "bundle.zip",
            zip_bytes(&[
                ("nested", b"not a directory".as_slice()),
                ("nested/hello.txt", b"hello".as_slice()),
            ]),
        ),
    ] {
        let (_client, session) = session().await;
        let error = session.extract(name, &archive, None).await.expect_err(name);
        assert!(
            error.message.contains("non-directory") || error.message.contains("descends through"),
            "`{name}` gave `{}`",
            error.message
        );
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_rejects_symlinked_parent_paths() {
    for (name, archive) in [
        (
            "bundle.tar",
            tar_bytes(&[("link/hello.txt", b"hello from tar")]),
        ),
        (
            "bundle.zip",
            zip_bytes(&[("link/hello.txt", b"hello from zip")]),
        ),
    ] {
        let (_client, session) = session().await;
        let root = root_of(&session);
        let outside = root.parent().unwrap().join(format!("outside-{name}"));
        std::fs::create_dir_all(&outside).unwrap();
        // A directory that is already a symlink out of the workspace must not be written through.
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        let error = session.extract(name, &archive, None).await.expect_err(name);
        assert_refused_member(&error, "link/hello.txt", "symlink in parent path: link");
        assert!(
            !outside.join("hello.txt").exists(),
            "`{name}` wrote through the link"
        );
        session.shutdown().await.expect("shutdown");
    }
}

#[tokio::test]
async fn extract_tar_refuses_a_symlink_member() {
    // `extract()` uses the no-symlinks policy on both backends, unlike hydrate.
    let (_client, session) = session().await;
    let archive = tar_archive(vec![
        tar_member("bin", TYPE_DIR, "", b"", 0o755),
        tar_member("bin/python", TYPE_SYMLINK, "/usr/bin/python3", b"", 0o777),
    ]);

    let error = session
        .extract("bundle.tar", &archive, None)
        .await
        .expect_err("a symlink member");
    assert_refused_member(&error, "bin/python", "symlink member not allowed");
    assert!(!root_of(&session).join("bin/python").exists());
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_reuses_directory_listings_during_symlink_checks() {
    // Many members under one directory must not mean one listing per member; the shared listing
    // cache is what keeps a deep archive from becoming quadratic in `ls` calls.
    let (_client, session) = session().await;
    let members: Vec<(String, Vec<u8>)> = (0..25)
        .map(|index| (format!("nested/file-{index}.txt"), b"payload".to_vec()))
        .collect();
    let archive = tar_archive(
        members
            .iter()
            .map(|(name, payload)| tar_file(name, payload))
            .collect(),
    );

    session
        .extract("bundle.tar", &archive, None)
        .await
        .expect("extract");

    let root = root_of(&session);
    for (name, _) in &members {
        assert!(root.join(name).is_file(), "missing `{name}`");
    }
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unix_local_helpers_reject_paths_outside_workspace_root() {
    let (_client, session) = session().await;

    for path in ["/etc/passwd", "../escape.txt"] {
        let error = session
            .extract(
                path,
                &tar_bytes(&[("a.txt", b"a")]),
                Some(CompressionScheme::Tar),
            )
            .await
            .expect_err(path);
        assert_eq!(error.code, ErrorCode::InvalidManifestPath, "for `{path}`");
    }
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unix_local_helpers_reject_symlink_escape_paths() {
    let (_client, session) = session().await;
    let root = root_of(&session);
    let outside = root.parent().unwrap().join("escape-target");
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();

    let error = session
        .extract(
            "escape/bundle.tar",
            &tar_bytes(&[("a.txt", b"a")]),
            Some(CompressionScheme::Tar),
        )
        .await
        .expect_err("a path through a symlink out of the workspace");
    assert!(
        matches!(
            error.code,
            ErrorCode::InvalidManifestPath | ErrorCode::WorkspaceArchiveWriteError
        ),
        "{error:?}"
    );
    assert!(!outside.join("bundle.tar").exists());
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unix_local_write_accepts_a_chunked_payload() {
    // Python's test feeds a non-seekable chunked stream; the Rust API takes the bytes directly,
    // so the equivalent contract is that a payload larger than one buffer round-trips intact.
    let (_client, session) = session().await;
    let payload: Vec<u8> = (0..(256 * 1024)).map(|index| (index % 251) as u8).collect();

    session
        .write_file("chunked.bin", &payload)
        .await
        .expect("write");

    let read_back = session.read_file("chunked.bin").await.expect("read");
    assert_eq!(read_back, payload);
    assert_eq!(
        std::fs::read(Path::new(&root_of(&session)).join("chunked.bin")).unwrap(),
        payload
    );
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_uses_session_default_archive_limits() {
    let (_client, session) = session().await;
    session.set_archive_limits(Some(SandboxArchiveLimits {
        max_input_bytes: None,
        max_extracted_bytes: None,
        max_members: Some(1),
    }));

    let error = session
        .extract(
            "bundle.tar",
            &tar_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
            None,
        )
        .await
        .expect_err("the session default applies when no per-call limits are given");

    assert_eq!(
        error.context.get("member").map(String::as_str),
        Some("two.txt")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("archive member count exceeds limit")
    );
    assert_eq!(error.context.get("limit").map(String::as_str), Some("1"));
    assert_eq!(error.context.get("actual").map(String::as_str), Some("2"));
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn extract_archive_limits_object_with_all_none_overrides_session_default() {
    let (_client, session) = session().await;
    session.set_archive_limits(Some(SandboxArchiveLimits {
        max_input_bytes: None,
        max_extracted_bytes: None,
        max_members: Some(1),
    }));

    // A per-call object whose every field is `None` is still an override: it lifts the default.
    session
        .extract_with_limits(
            "bundle.tar",
            &tar_bytes(&[("one.txt", b"1"), ("two.txt", b"2")]),
            None,
            Some(SandboxArchiveLimits {
                max_input_bytes: None,
                max_extracted_bytes: None,
                max_members: None,
            }),
        )
        .await
        .expect("an all-None override lifts the session default");

    let root = root_of(&session);
    assert_eq!(std::fs::read_to_string(root.join("one.txt")).unwrap(), "1");
    assert_eq!(std::fs::read_to_string(root.join("two.txt")).unwrap(), "2");
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unix_local_hydrate_workspace_rejects_external_symlink_targets() {
    let (_client, session) = session().await;
    let archive = tar_archive(vec![tar_member(
        "leak",
        TYPE_SYMLINK,
        "/etc/passwd",
        b"",
        0o777,
    )]);

    let error = session
        .hydrate_workspace(&archive)
        .await
        .expect_err("a symlink to an absolute path outside the archive");

    assert_refused_member(
        &error,
        "leak",
        "absolute symlink target not allowed: /etc/passwd",
    );
    assert!(std::fs::symlink_metadata(root_of(&session).join("leak")).is_err());
    session.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unix_local_persist_workspace_excludes_resolved_mount_path() {
    use openai_agents_sdk::sandbox::{ConfiguredMountStrategy, Entry, GcsMount, Manifest};

    let temp = tempfile::tempdir().unwrap();
    let workspace_root = temp.path().join("workspace");
    let actual_mount_path = workspace_root.join("actual");
    std::fs::create_dir_all(&actual_mount_path).unwrap();
    std::fs::write(actual_mount_path.join("remote.txt"), "remote").unwrap();
    std::fs::write(workspace_root.join("keep.txt"), "keep").unwrap();

    // The entry is keyed `logical` but mounts at `actual`: the exclusion has to follow the
    // resolved mount path, not the manifest key.
    let manifest = Manifest::new()
        .with_root(workspace_root.to_str().unwrap())
        .with_entry(
            "logical",
            Entry::mount(
                ConfiguredMountStrategy::mountpoint(GcsMount::new("bucket"))
                    .mount()
                    .with_mount_path("actual"),
            ),
        );
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.expect("session");

    let archive = session.persist_workspace().await.expect("persist");

    let names: Vec<String> = openai_agents_sdk::sandbox::archive::read_tar_members(&archive)
        .unwrap()
        .into_iter()
        .map(|member| member.name)
        .collect();
    assert!(
        names
            .iter()
            .any(|name| name.trim_start_matches("./") == "keep.txt"),
        "{names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|name| name.trim_start_matches("./").starts_with("actual")),
        "{names:?}"
    );
}
