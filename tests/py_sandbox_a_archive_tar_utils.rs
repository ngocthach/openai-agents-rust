//! Parity port of `tests/sandbox/test_tar_utils.py` from the Python SDK.
//!
//! Python's `tar_utils` splits into two Rust surfaces: the validator and extractor
//! (`safe_tar_member_rel_path`, `validate_tar_bytes`, `should_skip_tar_member`,
//! `extract_tar_to`), and the capture-time symlink rewrite that Python performs inside
//! `strip_tar_member_prefix` (`rebase_archive_symlink_targets`). The archives are built here as
//! raw `ustar` blocks, because most of the interesting members are ones a tar writer refuses to
//! produce: an absolute path, a `..` path, a hardlink, a FIFO, a root symlink.

#![cfg(feature = "sandbox")]

use std::path::Path;

use openai_agents_sdk::sandbox::archive::{
    ArchiveMemberKind, TarValidationOptions, extract_tar_to, read_tar_members,
    safe_tar_member_rel_path, should_skip_tar_member, strip_tar_member_prefix, validate_tar_bytes,
};
use openai_agents_sdk::sandbox::errors::ErrorCode;

// ----------------------------------------------------------------------------------------------
// Raw ustar fixtures
// ----------------------------------------------------------------------------------------------

const TYPE_FILE: u8 = b'0';
const TYPE_HARDLINK: u8 = b'1';
const TYPE_SYMLINK: u8 = b'2';
const TYPE_FIFO: u8 = b'6';
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

fn dir(name: &str) -> Vec<u8> {
    tar_member(name, TYPE_DIR, "", b"", 0o755)
}

fn file(name: &str, contents: &[u8]) -> Vec<u8> {
    tar_member(name, TYPE_FILE, "", contents, 0o644)
}

fn file_mode(name: &str, contents: &[u8], mode: u32) -> Vec<u8> {
    tar_member(name, TYPE_FILE, "", contents, mode)
}

fn symlink(name: &str, target: &str) -> Vec<u8> {
    tar_member(name, TYPE_SYMLINK, target, b"", 0o777)
}

fn hardlink(name: &str, target: &str) -> Vec<u8> {
    tar_member(name, TYPE_HARDLINK, target, b"", 0o644)
}

fn fifo(name: &str) -> Vec<u8> {
    tar_member(name, TYPE_FIFO, "", b"", 0o644)
}

/// The hydrate policy: symlinks kept, no external targets. Python's
/// `validate_tar_bytes(..., allow_external_symlink_targets=False)`.
fn strict() -> TarValidationOptions {
    TarValidationOptions::hydrate()
}

/// Python's default `validate_tar_bytes`, which tolerates external symlink targets.
fn lenient() -> TarValidationOptions {
    TarValidationOptions {
        allow_symlinks: true,
        allow_external_symlink_targets: true,
        ..TarValidationOptions::default()
    }
}

fn link_target(raw: &[u8], name: &str) -> String {
    read_tar_members(raw)
        .expect("members")
        .into_iter()
        .find(|member| member.name == name)
        .unwrap_or_else(|| panic!("no member named `{name}`"))
        .link_target
        .unwrap_or_default()
}

fn member_names(raw: &[u8]) -> Vec<String> {
    read_tar_members(raw)
        .expect("members")
        .into_iter()
        .map(|member| member.name.trim_end_matches('/').to_string())
        .collect()
}

// ----------------------------------------------------------------------------------------------
// safe_tar_member_rel_path
// ----------------------------------------------------------------------------------------------

#[test]
fn safe_tar_member_rel_path_requires_symlink_opt_in() {
    let error = safe_tar_member_rel_path("link.txt", ArchiveMemberKind::Symlink, false)
        .expect_err("a symlink member without the opt-in");
    assert!(
        error.message.contains("symlink member not allowed"),
        "{}",
        error.message
    );

    assert_eq!(
        safe_tar_member_rel_path("link.txt", ArchiveMemberKind::Symlink, true).unwrap(),
        Some(Path::new("link.txt").to_path_buf())
    );
}

#[test]
fn safe_tar_member_rel_path_rejects_hardlink_members() {
    let error = safe_tar_member_rel_path("b.txt", ArchiveMemberKind::Hardlink, true)
        .expect_err("a hardlink member");
    assert!(
        error.message.contains("hardlink member not allowed"),
        "{}",
        error.message
    );
}

// ----------------------------------------------------------------------------------------------
// validate_tar_bytes
// ----------------------------------------------------------------------------------------------

#[test]
fn validate_tar_bytes_rejects_root_symlink() {
    let raw = tar_archive(vec![symlink(".", "/tmp/outside")]);

    let error = validate_tar_bytes(&raw, &lenient()).expect_err("a symlink as the archive root");
    assert!(
        error.message.contains("archive root symlink"),
        "{}",
        error.message
    );
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
}

#[test]
fn validate_tar_bytes_rejects_windows_drive_member_paths() {
    for name in ["C:/tmp/evil.txt", r"C:\tmp\evil.txt"] {
        let raw = tar_archive(vec![file(name, b"evil")]);
        let error = validate_tar_bytes(&raw, &lenient()).expect_err(name);
        assert!(
            error.message.contains("windows drive path"),
            "`{name}` gave `{}`",
            error.message
        );
    }
}

#[test]
fn validate_tar_bytes_rejects_windows_separator_member_paths() {
    for name in [r"..\evil.txt", r"\evil.txt", r"nested\evil.txt"] {
        let raw = tar_archive(vec![file(name, b"evil")]);
        let error = validate_tar_bytes(&raw, &lenient()).expect_err(name);
        assert!(
            error.message.contains("windows path separator"),
            "`{name}` gave `{}`",
            error.message
        );
    }
}

#[test]
fn validate_tar_bytes_rejects_member_under_non_directory_member() {
    let raw = tar_archive(vec![
        file("nested/hello.txt", b"hello"),
        file("nested", b"not a directory"),
    ]);

    let error = validate_tar_bytes(&raw, &lenient()).expect_err("a file used as a parent");
    assert!(
        error
            .message
            .contains("archive path descends through non-directory: nested"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_rejects_absolute_symlink_target_in_strict_mode() {
    let raw = tar_archive(vec![symlink("leak", "/etc/passwd")]);

    let error = validate_tar_bytes(&raw, &strict()).expect_err("an absolute target");
    assert!(
        error
            .message
            .contains("absolute symlink target not allowed"),
        "{}",
        error.message
    );
    // The lenient policy leaves the decision to the caller, as Python's default does.
    validate_tar_bytes(&raw, &lenient()).expect("external targets are allowed by default");
}

#[test]
fn validate_tar_bytes_rejects_parent_escape_symlink_target_in_strict_mode() {
    let raw = tar_archive(vec![
        dir("nested"),
        symlink("nested/leak", "../../etc/passwd"),
    ]);

    let error = validate_tar_bytes(&raw, &strict()).expect_err("a target climbing out");
    assert!(
        error
            .message
            .contains("symlink target escapes archive root"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_allows_internal_symlink_target_in_strict_mode() {
    let raw = tar_archive(vec![
        dir("nested"),
        symlink("nested/python", "../bin/python3"),
    ]);

    validate_tar_bytes(&raw, &strict()).expect("a target that stays inside the archive");
}

#[test]
fn validate_tar_bytes_rejects_members_under_archive_symlink() {
    let raw = tar_archive(vec![
        symlink("link", "target"),
        file("link/escaped.txt", b"escaped"),
    ]);

    let error = validate_tar_bytes(&raw, &lenient()).expect_err("a member under a symlink");
    assert!(
        error
            .message
            .contains("archive path descends through symlink: link"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_rejects_unsupported_tar_member_types() {
    let raw = tar_archive(vec![fifo("dev.fifo")]);

    let error = validate_tar_bytes(&raw, &lenient()).expect_err("a FIFO member");
    assert!(
        error.message.contains("unsupported member type"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_rejects_hardlink_members() {
    let raw = tar_archive(vec![file("a.txt", b"x"), hardlink("b.txt", "a.txt")]);

    let error = validate_tar_bytes(&raw, &lenient()).expect_err("a hardlink member");
    assert!(
        error.message.contains("hardlink member not allowed"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_rejects_absolute_and_traversing_member_paths() {
    for (name, reason) in [
        ("/etc/passwd", "absolute path"),
        ("../evil.txt", "parent traversal"),
        ("nested/../../evil.txt", "parent traversal"),
    ] {
        let raw = tar_archive(vec![file(name, b"evil")]);
        let error = validate_tar_bytes(&raw, &lenient()).expect_err(name);
        assert!(
            error.message.contains(reason),
            "`{name}` gave `{}`",
            error.message
        );
    }
}

#[test]
fn validate_tar_bytes_ignores_skipped_unsafe_member() {
    // A member excluded by a skip prefix is never validated, so an otherwise fatal member in a
    // skipped subtree does not condemn the whole archive.
    let raw = tar_archive(vec![
        dir("keep"),
        file("keep/ok.txt", b"ok"),
        fifo("skipme/dev.fifo"),
    ]);

    validate_tar_bytes(&raw, &lenient()).expect_err("the FIFO is fatal when it is validated");

    let options = TarValidationOptions {
        skip_rel_paths: vec!["skipme".to_string()],
        ..lenient()
    };
    validate_tar_bytes(&raw, &options).expect("a skipped member is not validated");
}

// ----------------------------------------------------------------------------------------------
// should_skip_tar_member
// ----------------------------------------------------------------------------------------------

#[test]
fn should_skip_tar_member_matches_both_member_spellings() {
    let skip = vec!["cache".to_string()];

    // A bare name, a `./`-prefixed name and a root-prefixed name all name the same member.
    assert!(should_skip_tar_member("cache", &skip, None));
    assert!(should_skip_tar_member("cache/blob", &skip, None));
    assert!(should_skip_tar_member("./cache/blob", &skip, None));
    assert!(should_skip_tar_member(
        "workspace/cache/blob",
        &skip,
        Some("workspace")
    ));

    // A sibling whose name merely starts with the prefix is not skipped.
    assert!(!should_skip_tar_member("cached/blob", &skip, None));
    assert!(!should_skip_tar_member("keep/blob", &skip, None));
}

// ----------------------------------------------------------------------------------------------
// extract_tar_to — Python's safe_extract_tarfile
// ----------------------------------------------------------------------------------------------

#[test]
fn safe_extract_tarfile_preserves_venv_style_symlinks() {
    let raw = tar_archive(vec![
        dir("."),
        dir("./uv-project"),
        dir("./uv-project/.venv"),
        dir("./uv-project/.venv/bin"),
        dir("./uv-project/.venv/lib"),
        file("./uv-project/main.py", b"print(\"snapshot smoke\")\n"),
        symlink("./uv-project/.venv/lib64", "lib"),
        symlink("./uv-project/.venv/bin/python3", "/usr/local/bin/python3"),
        symlink("./uv-project/.venv/bin/python", "python3"),
    ]);

    validate_tar_bytes(&raw, &lenient()).expect("a venv-shaped archive is valid");
    let temp = tempfile::tempdir().unwrap();
    extract_tar_to(temp.path(), &raw, &lenient()).expect("extract");

    let root = temp.path();
    assert_eq!(
        std::fs::read_to_string(root.join("uv-project/main.py")).unwrap(),
        "print(\"snapshot smoke\")\n"
    );
    for (link, target) in [
        ("uv-project/.venv/lib64", "lib"),
        ("uv-project/.venv/bin/python3", "/usr/local/bin/python3"),
        ("uv-project/.venv/bin/python", "python3"),
    ] {
        assert_eq!(
            std::fs::read_link(root.join(link))
                .unwrap()
                .to_str()
                .unwrap(),
            target,
            "for `{link}`"
        );
    }
}

#[test]
fn safe_extract_tarfile_rejects_external_symlink_target_in_strict_mode() {
    let raw = tar_archive(vec![symlink("leak", "/etc/passwd")]);
    let temp = tempfile::tempdir().unwrap();

    let error = extract_tar_to(temp.path(), &raw, &strict()).expect_err("an external target");
    assert!(
        error
            .message
            .contains("absolute symlink target not allowed"),
        "{}",
        error.message
    );
    assert!(!temp.path().join("leak").exists());
}

#[test]
fn safe_extract_tarfile_can_replace_existing_leaf_entries() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // An existing file becomes a symlink.
    std::fs::write(root.join("leaf"), "old").unwrap();
    extract_tar_to(
        root,
        &tar_archive(vec![symlink("leaf", "target")]),
        &lenient(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_link(root.join("leaf"))
            .unwrap()
            .to_str()
            .unwrap(),
        "target"
    );

    // The symlink becomes a regular file again.
    extract_tar_to(root, &tar_archive(vec![file("leaf", b"new")]), &lenient()).unwrap();
    assert_eq!(std::fs::read_to_string(root.join("leaf")).unwrap(), "new");
    assert!(
        !std::fs::symlink_metadata(root.join("leaf"))
            .unwrap()
            .is_symlink()
    );

    // A symlink becomes a directory.
    extract_tar_to(root, &tar_archive(vec![symlink("leaf2", "t")]), &lenient()).unwrap();
    extract_tar_to(root, &tar_archive(vec![dir("leaf2")]), &lenient()).unwrap();
    assert!(root.join("leaf2").is_dir());

    // A regular file becomes a directory.
    std::fs::write(root.join("leaf3"), "old").unwrap();
    extract_tar_to(root, &tar_archive(vec![dir("leaf3")]), &lenient()).unwrap();
    assert!(root.join("leaf3").is_dir());
}

#[test]
fn safe_extract_tarfile_can_rehydrate_existing_leaf_symlink() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::os::unix::fs::symlink("old-target", root.join("link")).unwrap();

    extract_tar_to(
        root,
        &tar_archive(vec![symlink("link", "new-target")]),
        &lenient(),
    )
    .unwrap();

    assert_eq!(
        std::fs::read_link(root.join("link"))
            .unwrap()
            .to_str()
            .unwrap(),
        "new-target"
    );
}

#[test]
fn safe_extract_tarfile_rejects_preexisting_symlink_parent() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    // A directory in the destination is already a link out of the root.
    std::os::unix::fs::symlink(&outside, root.join("nested")).unwrap();

    let raw = tar_archive(vec![file("nested/payload.txt", b"payload")]);
    let error = extract_tar_to(root, &raw, &lenient()).expect_err("a symlinked parent");
    assert!(
        error.message.to_lowercase().contains("symlink"),
        "{}",
        error.message
    );
    assert!(!outside.join("payload.txt").exists());
}

#[test]
fn safe_extract_tarfile_restores_regular_file_modes() {
    use std::os::unix::fs::PermissionsExt as _;

    // The policy of tarfile's `data` filter: setuid, setgid, sticky and group/other write are
    // dropped, execute survives only when the owner had it, and the owner can always read/write.
    for (archived, expected) in [
        (0o755, 0o755),
        (0o644, 0o644),
        (0o600, 0o600),
        (0o700, 0o700),
        (0o444, 0o644),
        (0o000, 0o600),
        (0o777, 0o755),
        (0o655, 0o644),
        (0o4755, 0o755),
        (0o2755, 0o755),
        (0o1755, 0o755),
    ] {
        let temp = tempfile::tempdir().unwrap();
        extract_tar_to(
            temp.path(),
            &tar_archive(vec![file_mode("run.sh", b"#!/bin/sh\n", archived)]),
            &lenient(),
        )
        .unwrap();
        let mode = std::fs::metadata(temp.path().join("run.sh"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode, expected, "archived {archived:o}");
    }
}

#[test]
fn safe_extract_tarfile_keeps_workspace_scripts_executable() {
    use std::os::unix::fs::PermissionsExt as _;

    let temp = tempfile::tempdir().unwrap();
    let raw = tar_archive(vec![
        dir("."),
        dir("./bin"),
        file_mode("./bin/start", b"#!/bin/sh\necho hi\n", 0o755),
        file_mode("./README.md", b"# readme\n", 0o644),
    ]);

    extract_tar_to(temp.path(), &raw, &lenient()).unwrap();

    let executable = |path: &str| {
        std::fs::metadata(temp.path().join(path))
            .unwrap()
            .permissions()
            .mode()
            & 0o111
            != 0
    };
    assert!(executable("bin/start"));
    assert!(!executable("README.md"));
}

#[test]
fn safe_extract_tarfile_restores_mode_when_replacing_an_existing_file() {
    use std::os::unix::fs::PermissionsExt as _;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("script.sh");
    std::fs::write(&path, "old").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

    extract_tar_to(
        temp.path(),
        &tar_archive(vec![file_mode("script.sh", b"#!/bin/sh\n", 0o755)]),
        &lenient(),
    )
    .unwrap();

    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o755
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "#!/bin/sh\n");
}

// ----------------------------------------------------------------------------------------------
// reject_rel_paths / reject_symlink_rel_paths
// ----------------------------------------------------------------------------------------------

#[test]
fn validate_tar_bytes_can_reject_specific_symlink_path() {
    let raw = tar_archive(vec![symlink("workspace", "/tmp/outside")]);
    let options = TarValidationOptions {
        reject_symlink_rel_paths: vec!["workspace".to_string()],
        ..lenient()
    };

    let error = validate_tar_bytes(&raw, &options).expect_err("a rejected symlink path");
    assert!(
        error
            .message
            .contains("symlink member not allowed: workspace"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_specific_symlink_rejection_normalizes_dot_prefix() {
    let raw = tar_archive(vec![symlink("./workspace", "/tmp/outside")]);
    let options = TarValidationOptions {
        reject_symlink_rel_paths: vec!["workspace".to_string()],
        ..lenient()
    };

    let error = validate_tar_bytes(&raw, &options).expect_err("a dot-prefixed symlink");
    assert!(
        error
            .message
            .contains("symlink member not allowed: workspace"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_specific_symlink_rejection_does_not_reject_children() {
    let raw = tar_archive(vec![
        dir("workspace"),
        symlink("workspace/link", "/tmp/outside"),
    ]);
    let options = TarValidationOptions {
        reject_symlink_rel_paths: vec!["workspace".to_string()],
        ..lenient()
    };

    validate_tar_bytes(&raw, &options).expect("only the named path is refused");
}

#[test]
fn validate_tar_bytes_rejects_members_overlapping_protected_path() {
    let options = TarValidationOptions {
        reject_rel_paths: vec!["remote".to_string()],
        ..lenient()
    };
    for (label, member) in [
        ("file under it", file("remote/data.txt", b"payload")),
        ("symlink under it", symlink("remote/link", "../outside")),
        ("file at it", file("remote", b"payload")),
    ] {
        let error = validate_tar_bytes(&tar_archive(vec![member]), &options).expect_err(label);
        assert!(
            error
                .message
                .contains("archive member overlaps protected path: remote"),
            "{label}: {}",
            error.message
        );
    }
}

#[test]
fn validate_tar_bytes_rejects_non_directory_ancestor_of_protected_path() {
    let options = TarValidationOptions {
        reject_rel_paths: vec!["remote/nested".to_string()],
        ..lenient()
    };

    let error = validate_tar_bytes(&tar_archive(vec![file("remote", b"x")]), &options)
        .expect_err("a file where the protected path's parent should be");
    assert!(
        error
            .message
            .contains("archive member overlaps protected path: remote/nested"),
        "{}",
        error.message
    );
}

#[test]
fn validate_tar_bytes_allows_directory_ancestor_of_protected_path() {
    let options = TarValidationOptions {
        reject_rel_paths: vec!["remote/nested".to_string()],
        ..lenient()
    };

    validate_tar_bytes(&tar_archive(vec![dir("remote")]), &options)
        .expect("a directory ancestor is how the protected path is reached");
}

// ----------------------------------------------------------------------------------------------
// extract_tar_to — existing destination state
// ----------------------------------------------------------------------------------------------

#[test]
fn safe_extract_tarfile_rejects_existing_leaf_directory_for_symlink() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("link.txt")).unwrap();

    let error = extract_tar_to(
        temp.path(),
        &tar_archive(vec![symlink("link.txt", "target.txt")]),
        &lenient(),
    )
    .expect_err("a symlink member over an existing directory");
    assert!(
        error
            .message
            .contains("destination directory already exists"),
        "{}",
        error.message
    );
}

#[test]
fn safe_extract_tarfile_rejects_symlink_under_preexisting_symlink_parent() {
    let temp = tempfile::tempdir().unwrap();
    let outside = temp.path().join("outside");
    let root = temp.path().join("root");
    std::fs::create_dir(&outside).unwrap();
    std::fs::create_dir(&root).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();

    let error = extract_tar_to(
        &root,
        &tar_archive(vec![symlink("escape/nested/link.txt", "target.txt")]),
        &lenient(),
    )
    .expect_err("a symlink member beneath a symlinked parent");
    // Python accepts either wording: `path escapes root|symlink in parent path`.
    assert!(
        error.message.contains("path escapes root")
            || error.message.contains("symlink in parent path"),
        "{}",
        error.message
    );
    assert!(!outside.join("nested").exists());
}

// ----------------------------------------------------------------------------------------------
// strip_tar_member_prefix — Python's tar_utils.strip_tar_member_prefix
// ----------------------------------------------------------------------------------------------

fn builder_dir(builder: &mut tar::Builder<Vec<u8>>, name: &str) {
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Directory);
    header.set_mode(0o755);
    header.set_size(0);
    builder.append_data(&mut header, name, &[][..]).unwrap();
}

fn builder_file(builder: &mut tar::Builder<Vec<u8>>, name: &str, payload: &[u8]) {
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Regular);
    header.set_mode(0o644);
    header.set_size(payload.len() as u64);
    builder.append_data(&mut header, name, payload).unwrap();
}

fn builder_symlink(builder: &mut tar::Builder<Vec<u8>>, name: &str, target: &str) {
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_mode(0o777);
    header.set_size(0);
    builder.append_link(&mut header, name, target).unwrap();
}

/// Python's `_prefixed_workspace_archive`: a `workspace/...` archive shaped like Docker's staged
/// copy, carrying members the strict hydrate extractor refuses as they stand. The long names need
/// a GNU extension header, which the `tar` writer emits and a raw ustar block cannot.
fn prefixed_workspace_archive(external_symlink: bool, link_through_alias: bool) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    builder_dir(&mut builder, "workspace");
    builder_dir(&mut builder, "workspace/sub");
    builder_dir(&mut builder, "workspace/sub/deep");
    builder_file(&mut builder, "workspace/a.txt", b"shared");
    builder_file(&mut builder, "workspace/data.txt", b"wrong");
    builder_file(&mut builder, "workspace/sub/data.txt", b"right");
    let mut fifo = tar::Header::new_gnu();
    fifo.set_entry_type(tar::EntryType::Fifo);
    fifo.set_size(0);
    builder
        .append_data(&mut fifo, "workspace/dev.fifo", &[][..])
        .unwrap();
    builder_symlink(&mut builder, "workspace/sub/abs_up", "/workspace/a.txt");
    builder_symlink(&mut builder, "workspace/rel", "a.txt");
    builder_symlink(&mut builder, "workspace/double_slash", "//workspace/a.txt");
    builder_symlink(&mut builder, "workspace/double_sep", "/workspace//a.txt");
    builder_symlink(&mut builder, "workspace/alias", "sub/deep");
    if link_through_alias {
        // `alias/..` resolves against the alias target, so where this lands depends on another
        // symlink and the rewrite must leave it absolute.
        builder_symlink(
            &mut builder,
            "workspace/abs_alias",
            "/workspace/alias/../data.txt",
        );
    }
    let mut nested = "workspace".to_string();
    for _ in 0..5 {
        nested.push_str("/deeply-nested-directory");
        builder_dir(&mut builder, &nested);
    }
    builder_file(&mut builder, &format!("{nested}/target.txt"), b"deep");
    let long_target = format!(
        "/workspace/{}/target.txt",
        ["deeply-nested-directory"; 5].join("/")
    );
    builder_symlink(&mut builder, "workspace/long_link", &long_target);
    if external_symlink {
        builder_symlink(&mut builder, "workspace/outside", "/usr/bin/python3");
    }
    builder.into_inner().unwrap()
}

fn strip(raw: &[u8], root: Option<&str>) -> Vec<u8> {
    strip_tar_member_prefix(raw, "workspace", root).expect("strip")
}

#[test]
fn strip_tar_member_prefix_returns_workspace_relative_archive() {
    let raw = tar_archive(vec![
        dir("workspace"),
        dir("workspace/pkg"),
        file("workspace/pkg/main.py", b"print('hello')\n"),
        symlink("workspace/pkg/python", "python3"),
    ]);

    let normalized = strip_tar_member_prefix(&raw, "workspace", None).unwrap();

    assert_eq!(
        member_names(&normalized),
        [".", "pkg", "pkg/main.py", "pkg/python"]
    );
}

#[test]
fn strip_tar_member_prefix_rejects_an_empty_prefix_and_a_foreign_member() {
    let raw = tar_archive(vec![dir("workspace"), file("elsewhere/a.txt", b"a")]);

    let error = strip_tar_member_prefix(&raw, "", None).expect_err("an empty prefix");
    assert!(
        error
            .message
            .contains("tar member prefix must not be empty"),
        "{}",
        error.message
    );
    let error = strip_tar_member_prefix(&raw, "workspace", None).expect_err("a member off-prefix");
    assert!(
        error
            .message
            .contains("member does not start with prefix: workspace"),
        "{}",
        error.message
    );
}

#[test]
fn strip_tar_member_prefix_fails_on_a_duplicate_member() {
    // Python's "validation" failure mode: the stripped archive is validated before it is returned.
    let raw = tar_archive(vec![
        dir("workspace"),
        file("workspace/data.txt", b""),
        file("workspace/data.txt", b"duplicate"),
    ]);

    let error = strip_tar_member_prefix(&raw, "workspace", None).expect_err("a duplicate path");
    assert!(
        error.message.contains("duplicate archive path: data.txt"),
        "{}",
        error.message
    );
}

#[test]
fn strip_tar_member_prefix_rewrites_members_hydrate_refuses() {
    let stripped = strip(&prefixed_workspace_archive(true, true), Some("/workspace"));

    let names = member_names(&stripped);
    assert!(!names.contains(&"dev.fifo".to_string()), "{names:?}");
    assert_eq!(link_target(&stripped, "sub/abs_up"), "../a.txt");
    assert_eq!(link_target(&stripped, "rel"), "a.txt");
    assert_eq!(link_target(&stripped, "double_slash"), "a.txt");
    assert_eq!(link_target(&stripped, "double_sep"), "a.txt");
    // Components after the root prefix are kept verbatim and `..` is not collapsed; this one
    // depends on how `alias` resolves, so it stays absolute for strict hydration to refuse.
    assert_eq!(
        link_target(&stripped, "abs_alias"),
        "/workspace/alias/../data.txt"
    );
    // A target longer than the 100-byte ustar field lives in an extension header; the rewritten
    // value must be the one a reader sees.
    assert_eq!(
        link_target(&stripped, "long_link"),
        format!("{}/target.txt", ["deeply-nested-directory"; 5].join("/"))
    );
    // External absolute targets are left for hydrate's policy to decide.
    assert_eq!(link_target(&stripped, "outside"), "/usr/bin/python3");
}

#[test]
fn strip_tar_member_prefix_output_passes_strict_hydrate_validation() {
    let stripped = strip(
        &prefixed_workspace_archive(false, false),
        Some("/workspace"),
    );
    let temp = tempfile::tempdir().unwrap();

    validate_tar_bytes(&stripped, &strict()).expect("strict hydrate validation");
    extract_tar_to(temp.path(), &stripped, &strict()).expect("strict extract");

    // Inspect the restored link metadata rather than reading through the links.
    assert_eq!(
        std::fs::read_link(temp.path().join("sub/abs_up"))
            .unwrap()
            .to_str()
            .unwrap(),
        "../a.txt"
    );
    assert_eq!(
        std::fs::read_link(temp.path().join("alias"))
            .unwrap()
            .to_str()
            .unwrap(),
        "sub/deep"
    );
    assert_eq!(
        std::fs::read(temp.path().join("sub/data.txt")).unwrap(),
        b"right"
    );
    assert!(std::fs::symlink_metadata(temp.path().join("dev.fifo")).is_err());
}

#[test]
fn strip_tar_member_prefix_output_with_alias_link_is_refused_by_strict_hydrate() {
    // The archive keeps `/workspace/alias/../data.txt` absolute and the strict validation is
    // what refuses it: the rewrite never guesses through another link.
    let stripped = strip(&prefixed_workspace_archive(false, true), Some("/workspace"));

    assert_eq!(
        link_target(&stripped, "abs_alias"),
        "/workspace/alias/../data.txt"
    );
    let error = validate_tar_bytes(&stripped, &strict()).expect_err("an absolute target");
    assert!(
        error
            .message
            .contains("absolute symlink target not allowed"),
        "{}",
        error.message
    );
}

#[test]
fn strip_tar_member_prefix_leaves_non_simple_targets_absolute() {
    // Each target's destination depends on another symlink, walks through a path the archive does
    // not establish as an ordinary directory, or names a directory. None is rewritten, so strict
    // hydration refuses it instead of a rewrite guessing.
    let cases: Vec<(&str, Vec<Vec<u8>>, &str, &str)> = vec![
        (
            "dot-dot after a link that climbs out",
            vec![dir("workspace/a"), symlink("workspace/a/link", "..")],
            "workspace/victim",
            "/workspace/a/link/../tmp",
        ),
        (
            "dot-dot after an external link",
            vec![symlink("workspace/outside", "/usr")],
            "workspace/victim",
            "/workspace/outside/../x",
        ),
        (
            "dot-dot through a component the archive does not create",
            vec![file("workspace/secret", b"s")],
            "workspace/victim",
            "/workspace/alias/../secret",
        ),
        (
            "leaf the archive does not create",
            vec![],
            "workspace/victim",
            "/workspace/missing.txt",
        ),
        (
            "descends through a regular file",
            vec![file("workspace/notes.txt", b"n")],
            "workspace/victim",
            "/workspace/notes.txt/secret",
        ),
        (
            "descends through a symlink member",
            vec![symlink("workspace/loop", "loop")],
            "workspace/victim",
            "/workspace/loop/x",
        ),
        (
            "leaf is a symlink member",
            vec![symlink("workspace/alias", "sub"), dir("workspace/sub")],
            "workspace/victim",
            "/workspace/alias",
        ),
        (
            "trailing separator after a regular file",
            vec![file("workspace/a.txt", b"a")],
            "workspace/victim",
            "/workspace/a.txt/",
        ),
        (
            "trailing separator after a directory",
            vec![dir("workspace/sub")],
            "workspace/victim",
            "/workspace/sub/",
        ),
        (
            "directory target",
            vec![dir("workspace/sub")],
            "workspace/victim",
            "/workspace/sub",
        ),
        ("workspace root", vec![], "workspace/victim", "/workspace"),
        (
            "workspace root with separator",
            vec![],
            "workspace/victim",
            "/workspace/",
        ),
        (
            "directories only implied by a file path",
            vec![file("workspace/implied/deep/data.txt", b"d")],
            "workspace/victim",
            "/workspace/implied/deep/data.txt",
        ),
        (
            "would resolve inside, but only by interpreting two links",
            vec![
                dir("workspace/a"),
                dir("workspace/sub"),
                file("workspace/data.txt", b"root"),
                symlink("workspace/a/link2", "../sub"),
                symlink("workspace/b", "a/link2"),
            ],
            "workspace/sub/victim",
            "/workspace/b/../data.txt",
        ),
    ];

    for (label, members, victim, target) in cases {
        let mut all = vec![dir("workspace")];
        all.extend(members);
        all.push(symlink(victim, target));
        let stripped = strip(&tar_archive(all), Some("/workspace"));

        let name = victim.strip_prefix("workspace/").unwrap();
        assert_eq!(link_target(&stripped, name), target, "{label}");
        let error = validate_tar_bytes(&stripped, &strict()).expect_err(label);
        assert!(
            error
                .message
                .contains("absolute symlink target not allowed"),
            "{label}: {}",
            error.message
        );
    }
}

#[test]
fn strip_tar_member_prefix_rebases_simple_targets_established_by_the_archive() {
    // Every directory the relative target walks through is an explicit directory member and the
    // leaf is a regular file, so the rewrite is exact.
    let raw = tar_archive(vec![
        dir("workspace"),
        dir("workspace/sub"),
        dir("workspace/sub/deep"),
        file("workspace/sub/deep/data.txt", b"d"),
        dir("workspace/other"),
        symlink("workspace/other/victim", "/workspace/sub/deep/data.txt"),
    ]);

    let stripped = strip(&raw, Some("/workspace"));

    assert_eq!(
        link_target(&stripped, "other/victim"),
        "../sub/deep/data.txt"
    );
    validate_tar_bytes(&stripped, &strict()).expect("an exact rewrite is hydratable");
}

#[test]
fn strip_tar_member_prefix_leaves_link_under_unestablished_parent_absolute() {
    // The link's own parent directory is only implied, so the `..` climb cannot be trusted
    // either: hydration could find a symlink there in the destination.
    let raw = tar_archive(vec![
        dir("workspace"),
        file("workspace/data.txt", b"d"),
        symlink("workspace/implied/victim", "/workspace/data.txt"),
    ]);

    let stripped = strip(&raw, Some("/workspace"));

    assert_eq!(
        link_target(&stripped, "implied/victim"),
        "/workspace/data.txt"
    );
}

#[test]
fn strip_tar_member_prefix_keeps_absolute_symlinks_without_a_root() {
    let stripped = strip(&prefixed_workspace_archive(false, true), None);

    assert_eq!(link_target(&stripped, "sub/abs_up"), "/workspace/a.txt");
}

#[test]
fn strip_tar_member_prefix_still_rejects_hardlink_members() {
    let raw = tar_archive(vec![
        dir("workspace"),
        file("workspace/a.txt", b"x"),
        hardlink("workspace/b.txt", "workspace/a.txt"),
    ]);

    let error = strip_tar_member_prefix(&raw, "workspace", None).expect_err("a hardlink member");
    assert!(
        error.message.contains("hardlink member not allowed"),
        "{}",
        error.message
    );
}

#[test]
fn strip_tar_member_prefix_rewrites_long_path_headers() {
    // Python writes a PAX `path` record; the Rust writer's equivalent for a name past the 100-byte
    // ustar field is a GNU long-name header. Either way the stripped name must be the one a
    // reader sees, with no stale record left to override it.
    let long_name = format!("workspace/{}.txt", "a".repeat(120));
    let mut builder = tar::Builder::new(Vec::new());
    builder_file(&mut builder, &long_name, b"payload");
    let raw = builder.into_inner().unwrap();

    let normalized = strip_tar_member_prefix(&raw, "workspace", None).unwrap();

    assert_eq!(
        member_names(&normalized),
        [format!("{}.txt", "a".repeat(120))]
    );
}
