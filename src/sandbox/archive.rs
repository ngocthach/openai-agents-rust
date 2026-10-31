//! In-process archive validation and extraction. Mirrors `agents.sandbox.util.tar_utils` and
//! `agents.sandbox.session.archive_extraction`.
//!
//! Every archive the SDK unpacks — a snapshot restored into a workspace, a `tar` or `zip` handed
//! to [`SandboxSession::extract`] — is inspected member by member *before* anything is written.
//! Delegating that to the backend's own `tar` makes traversal and symlink safety a property of
//! whichever `tar` the image ships; the checks here belong to the SDK, so they hold on every
//! backend, including the ones that unpack remotely (their bytes are validated on the host
//! first).
//!
//! [`SandboxSession::extract`]: crate::sandbox::SandboxSession::extract

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxOp, SandboxResult};
use crate::sandbox::types::SandboxArchiveLimits;

/// The compression scheme of an archive handed to [`SandboxSession::extract`].
///
/// Mirrors Python's `compression_scheme: Literal["tar", "zip"]`.
///
/// [`SandboxSession::extract`]: crate::sandbox::SandboxSession::extract
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionScheme {
    /// A `tar` archive, optionally gzipped.
    Tar,
    /// A `zip` archive.
    Zip,
}

impl CompressionScheme {
    /// The wire name of the scheme.
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionScheme::Tar => "tar",
            CompressionScheme::Zip => "zip",
        }
    }

    /// The scheme named by `value`, or `None` when it is neither `tar` nor `zip`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tar" => Some(CompressionScheme::Tar),
            "zip" => Some(CompressionScheme::Zip),
            _ => None,
        }
    }

    /// The scheme implied by a path's final extension, as Python infers it from `path.suffix`.
    pub fn infer_from_path(path: &str) -> Option<Self> {
        let name = path.rsplit('/').next().unwrap_or(path);
        let suffix = name.rsplit_once('.').map(|(_, suffix)| suffix)?;
        CompressionScheme::parse(suffix)
    }
}

/// What one archive member is. Mirrors the `tarfile.TarInfo` predicates Python branches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveMemberKind {
    /// A directory.
    Directory,
    /// A regular file.
    Regular,
    /// A symbolic link.
    Symlink,
    /// A hard link.
    Hardlink,
    /// Anything else: a device node, a FIFO, a socket.
    Other,
}

/// The error Python raises as `UnsafeTarMemberError`.
pub fn unsafe_tar_member(member: &str, reason: impl Into<String>) -> SandboxError {
    unsafe_member("tar", member, reason.into())
}

/// The error Python raises as `UnsafeZipMemberError`.
pub fn unsafe_zip_member(member: &str, reason: impl Into<String>) -> SandboxError {
    unsafe_member("zip", member, reason.into())
}

fn unsafe_member(kind: &str, member: &str, reason: String) -> SandboxError {
    SandboxError::new(
        ErrorCode::WorkspaceArchiveWriteError,
        format!("unsafe {kind} member `{member}`: {reason}"),
    )
    .with_op(SandboxOp::Write)
    .with_context("member", member)
    .with_context("reason", reason)
    .with_retryable(false)
}

/// The error Python raises as `ArchiveResourceLimitError`.
pub fn archive_limit_exceeded(
    reason: &str,
    limit: u64,
    actual: u64,
    member: Option<&str>,
) -> SandboxError {
    let mut error = SandboxError::new(ErrorCode::WorkspaceArchiveWriteError, reason)
        .with_op(SandboxOp::Write)
        .with_context("reason", reason)
        .with_context("limit", limit.to_string())
        .with_context("actual", actual.to_string())
        .with_retryable(false);
    if let Some(member) = member {
        error = error.with_context("member", member);
    }
    error
}

/// The error Python raises as `UnsafeTarMemberError(member="<tar>", reason="invalid tar stream")`.
fn invalid_tar_stream(detail: impl std::fmt::Display) -> SandboxError {
    unsafe_tar_member("<tar>", format!("invalid tar stream: {detail}"))
}

/// The error Python raises for a zip that cannot be read at all.
fn invalid_zip_stream(detail: impl std::fmt::Display) -> SandboxError {
    unsafe_zip_member("<zip>", format!("invalid zip stream: {detail}"))
}

/// One member of a tar archive, as the validator sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TarMember {
    /// The raw name recorded in the archive.
    pub name: String,
    /// What the member is.
    pub kind: ArchiveMemberKind,
    /// The declared payload size, for regular files.
    pub size: u64,
    /// The link target, for symlinks and hard links.
    pub link_target: Option<String>,
    /// The permission bits recorded for the member.
    pub mode: u32,
}

/// The policy one archive is validated under. Mirrors the keyword arguments of Python's
/// `validate_tarfile`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TarValidationOptions {
    /// Whether symlink members are accepted at all.
    pub allow_symlinks: bool,
    /// Whether a symlink may point outside the archive root, or at an absolute path.
    pub allow_external_symlink_targets: bool,
    /// Workspace-relative paths no archive member may overlap. A member under one, or a
    /// non-directory member that is an ancestor of one, is refused. Mirrors Python's
    /// `reject_rel_paths`.
    pub reject_rel_paths: Vec<String>,
    /// Workspace-relative paths that may not be symlink members themselves. Their children are
    /// unaffected. Mirrors Python's `reject_symlink_rel_paths`.
    pub reject_symlink_rel_paths: Vec<String>,
    /// Workspace-relative prefixes whose members are ignored instead of validated.
    pub skip_rel_paths: Vec<String>,
    /// The archive's own root directory name, when its members are nested under one.
    pub root_name: Option<String>,
    /// Resource ceilings applied while validating.
    pub limits: Option<SandboxArchiveLimits>,
}

impl TarValidationOptions {
    /// The policy a workspace archive is restored under: symlink members are kept, because real
    /// workspaces contain them, no member may descend through one, and no link may point at an
    /// absolute path or climb out of the archive.
    ///
    /// This is what Python passes at every one of its hydrate call sites
    /// (`allow_external_symlink_targets=False`), so a snapshot that carries a link out of the
    /// workspace — a virtualenv's `bin/python -> /usr/bin/python3` — is refused on both sides.
    /// A link *into* the workspace survives a restore because it is rewritten relative to the
    /// link's own directory when the archive is captured
    /// ([`rebase_archive_symlink_targets`]), which is where Python rewrites it too.
    pub fn hydrate() -> Self {
        TarValidationOptions {
            allow_symlinks: true,
            allow_external_symlink_targets: false,
            ..TarValidationOptions::default()
        }
    }

    /// The policy `extract()` uses: no symlink members at all, plus the configured limits.
    pub fn extract(limits: Option<SandboxArchiveLimits>) -> Self {
        TarValidationOptions {
            allow_symlinks: false,
            allow_external_symlink_targets: false,
            limits,
            ..TarValidationOptions::default()
        }
    }

    /// Apply resource limits to this policy.
    pub fn with_limits(mut self, limits: Option<SandboxArchiveLimits>) -> Self {
        self.limits = limits;
        self
    }
}

/// Validate one tar member's path and return its non-root relative path.
///
/// Mirrors Python's `safe_tar_member_rel_path`. `Ok(None)` is the archive root itself, which has
/// nothing to extract; every other accepted member yields a relative path with no `..`, no root
/// component and no Windows drive.
pub fn safe_tar_member_rel_path(
    name: &str,
    kind: ArchiveMemberKind,
    allow_symlinks: bool,
) -> SandboxResult<Option<PathBuf>> {
    if matches!(name, "" | "." | "./") {
        return validate_archive_root_member(name, kind).map(|()| None);
    }
    raise_if_windows_member_path(name, unsafe_tar_member)?;
    if name.starts_with('/') {
        return Err(unsafe_tar_member(name, "absolute path"));
    }
    if posix_parts(name).iter().any(|part| part == "..") {
        return Err(unsafe_tar_member(name, "parent traversal"));
    }
    if kind == ArchiveMemberKind::Symlink && !allow_symlinks {
        return Err(unsafe_tar_member(name, "symlink member not allowed"));
    }
    if kind == ArchiveMemberKind::Hardlink {
        return Err(unsafe_tar_member(name, "hardlink member not allowed"));
    }
    let supported = matches!(
        kind,
        ArchiveMemberKind::Directory | ArchiveMemberKind::Regular
    ) || (allow_symlinks && kind == ArchiveMemberKind::Symlink);
    if !supported {
        return Err(unsafe_tar_member(name, "unsupported member type"));
    }
    let rel = rel_path_from_parts(&posix_parts(name));
    if rel.as_os_str().is_empty() {
        return validate_archive_root_member(name, kind).map(|()| None);
    }
    Ok(Some(rel))
}

/// Validate one zip member's path. Mirrors Python's `safe_zip_member_rel_path`.
pub fn safe_zip_member_rel_path(
    name: &str,
    unix_mode: Option<u32>,
) -> SandboxResult<Option<PathBuf>> {
    if matches!(name, "" | "." | "./") {
        return Ok(None);
    }
    raise_if_windows_member_path(name, unsafe_zip_member)?;
    if name.starts_with('/') {
        return Err(unsafe_zip_member(name, "absolute path"));
    }
    if posix_parts(name).iter().any(|part| part == "..") {
        return Err(unsafe_zip_member(name, "parent traversal"));
    }
    if let Some(mode) = unix_mode
        && mode & 0o170000 == 0o120000
    {
        return Err(unsafe_zip_member(name, "link member not allowed"));
    }
    let rel = rel_path_from_parts(&posix_parts(name));
    if rel.as_os_str().is_empty() {
        return Ok(None);
    }
    Ok(Some(rel))
}

/// Whether a tar member is excluded by a workspace-relative prefix.
///
/// Mirrors Python's `should_skip_tar_member`. `member_name` is the raw archive name, which may
/// carry a leading `./` or the workspace root directory name depending on how the archive was
/// produced, so both readings are compared against every prefix.
pub fn should_skip_tar_member(
    member_name: &str,
    skip_rel_paths: &[String],
    root_name: Option<&str>,
) -> bool {
    let variants = tar_member_rel_variants(member_name, root_name);
    skip_rel_paths.iter().any(|prefix| {
        let prefix = normalize_rel(prefix);
        variants.iter().any(|rel| is_within(rel, &prefix))
    })
}

/// Read every member header of a tar archive, transparently gunzipping it.
pub fn read_tar_members(raw: &[u8]) -> SandboxResult<Vec<TarMember>> {
    let mut archive = tar::Archive::new(tar_reader(raw)?);
    let mut members = Vec::new();
    for entry in archive.entries().map_err(invalid_tar_stream)? {
        let entry = entry.map_err(invalid_tar_stream)?;
        let header = entry.header();
        let entry_type = header.entry_type();
        if matches!(
            entry_type,
            tar::EntryType::XHeader
                | tar::EntryType::XGlobalHeader
                | tar::EntryType::GNULongName
                | tar::EntryType::GNULongLink
        ) {
            continue;
        }
        members.push(TarMember {
            name: String::from_utf8_lossy(&entry.path_bytes()).into_owned(),
            kind: member_kind(entry_type),
            size: entry.size(),
            link_target: entry
                .link_name_bytes()
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned()),
            mode: header.mode().unwrap_or(0o644),
        });
    }
    Ok(members)
}

/// Validate a member inventory against `options`. Mirrors Python's `validate_tarfile`.
pub fn validate_tar_members(
    members: &[TarMember],
    options: &TarValidationOptions,
) -> SandboxResult<()> {
    let rejected_rel: Vec<PathBuf> = options
        .reject_rel_paths
        .iter()
        .map(|path| normalize_rel(path))
        .collect();
    let rejected_symlink_rel: BTreeSet<PathBuf> = options
        .reject_symlink_rel_paths
        .iter()
        .map(|path| normalize_rel(path))
        .collect();
    let mut members_by_rel: BTreeMap<PathBuf, &TarMember> = BTreeMap::new();
    let mut symlink_rel: BTreeSet<PathBuf> = BTreeSet::new();
    let mut accepted: Vec<(&TarMember, PathBuf)> = Vec::new();
    let mut member_count: u64 = 0;
    let mut extracted_bytes: u64 = 0;

    for member in members {
        if should_skip_tar_member(
            &member.name,
            &options.skip_rel_paths,
            options.root_name.as_deref(),
        ) {
            continue;
        }
        let Some(rel) =
            safe_tar_member_rel_path(&member.name, member.kind, options.allow_symlinks)?
        else {
            continue;
        };

        let variants = tar_member_rel_variants(&member.name, options.root_name.as_deref());
        for rejected in &rejected_rel {
            let overlaps = variants.iter().any(|variant| {
                is_within(variant, rejected)
                    || (member.kind != ArchiveMemberKind::Directory && is_within(rejected, variant))
            });
            if overlaps {
                return Err(unsafe_tar_member(
                    &member.name,
                    format!(
                        "archive member overlaps protected path: {}",
                        posix_display(rejected)
                    ),
                ));
            }
        }

        member_count += 1;
        check_member_count(member_count, &member.name, options.limits.as_ref())?;
        if member.kind == ArchiveMemberKind::Regular {
            extracted_bytes = extracted_bytes.saturating_add(member.size);
            check_extracted_bytes(extracted_bytes, &member.name, options.limits.as_ref())?;
        }

        if let Some(previous) = members_by_rel.get(&rel)
            && !(previous.kind == ArchiveMemberKind::Directory
                && member.kind == ArchiveMemberKind::Directory)
        {
            return Err(unsafe_tar_member(
                &member.name,
                format!("duplicate archive path: {}", posix_display(&rel)),
            ));
        }
        members_by_rel.insert(rel.clone(), member);

        if member.kind == ArchiveMemberKind::Symlink {
            if !options.allow_external_symlink_targets {
                validate_symlink_target(member, &rel)?;
            }
            if rejected_symlink_rel.contains(&rel) {
                return Err(unsafe_tar_member(
                    &member.name,
                    format!("symlink member not allowed: {}", posix_display(&rel)),
                ));
            }
            symlink_rel.insert(rel.clone());
        }
        accepted.push((member, rel));
    }

    for (member, rel) in &accepted {
        for parent in strict_parents(rel) {
            if symlink_rel.contains(&parent) {
                return Err(unsafe_tar_member(
                    &member.name,
                    format!(
                        "archive path descends through symlink: {}",
                        posix_display(&parent)
                    ),
                ));
            }
            if let Some(parent_member) = members_by_rel.get(&parent)
                && parent_member.kind != ArchiveMemberKind::Directory
            {
                return Err(unsafe_tar_member(
                    &member.name,
                    format!(
                        "archive path descends through non-directory: {}",
                        posix_display(&parent)
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validate raw tar bytes with the shared safe-tar policy. Mirrors `validate_tar_bytes`.
pub fn validate_tar_bytes(raw: &[u8], options: &TarValidationOptions) -> SandboxResult<()> {
    validate_tar_members(&read_tar_members(raw)?, options)
}

/// Validate a workspace archive a backend is about to unpack.
///
/// Every `hydrate_workspace` calls this before the bytes leave the host, including the backends
/// that unpack remotely: whether their image's `tar` refuses a traversal or a device member is
/// not something the SDK can see, so the decision is made here instead.
pub fn validate_hydrate_archive(archive: &[u8]) -> SandboxResult<()> {
    validate_tar_bytes(archive, &TarValidationOptions::hydrate())
}

/// Rewrite the absolute symlink targets of a freshly captured workspace archive as targets
/// relative to the link's own directory.
///
/// Mirrors the rebase Python runs while it captures a workspace
/// (`unix_local._archive_workspace` / `tar_utils.strip_tar_member_prefix`): a link recorded as
/// `<root>/lib/thing` names a path that no longer exists once the snapshot is restored under a
/// different root, and [`TarValidationOptions::hydrate`] refuses an absolute target anyway. Only
/// targets under one of `roots` are rewritten; a link out of the workspace keeps its absolute
/// form, which hydrate then refuses, exactly as Python's does.
///
/// A target is rewritten only when the archive itself proves it stays inside the workspace
/// (`archive_target_stays_under`). Returns the archive unchanged when nothing qualifies.
pub fn rebase_archive_symlink_targets(raw: &[u8], roots: &[String]) -> SandboxResult<Vec<u8>> {
    let members = read_tar_members(raw)?;
    let mut inventory: BTreeMap<String, &TarMember> = BTreeMap::new();
    for member in &members {
        inventory.insert(
            posix_display(&rel_path_from_parts(&posix_parts(&member.name))),
            member,
        );
    }

    let mut rebased: BTreeMap<String, String> = BTreeMap::new();
    for member in &members {
        if member.kind != ArchiveMemberKind::Symlink {
            continue;
        }
        let Some(target) = member.link_target.as_deref() else {
            continue;
        };
        if !target.starts_with('/') {
            continue;
        }
        let rel = posix_display(&rel_path_from_parts(&posix_parts(&member.name)));
        let Some(candidate) = rebase_symlink_target(target, &rel, roots) else {
            continue;
        };
        if archive_target_stays_under(&inventory, &rel, &candidate) {
            rebased.insert(member.name.clone(), candidate);
        }
    }
    if rebased.is_empty() {
        return Ok(raw.to_vec());
    }
    rewrite_tar_link_targets(raw, &rebased)
}

/// The link-relative form of an absolute `linkname` that points under one of `roots`.
///
/// Mirrors Python's `unix_local._rebase_symlink_target`. Only the root prefix is replaced; every
/// later component is kept verbatim, because `..` after a symlink component is resolved by the
/// kernel against that link's target, so normalizing here would move the link.
fn rebase_symlink_target(linkname: &str, link_rel: &str, roots: &[String]) -> Option<String> {
    let target = format!("/{}", linkname.trim_start_matches('/'));
    for root in roots {
        let prefix = format!("/{}", root.trim_matches('/'));
        let rest = if target == prefix {
            String::new()
        } else if let Some(rest) = target.strip_prefix(&format!("{prefix}/")) {
            rest.trim_start_matches('/').to_string()
        } else {
            continue;
        };
        // The link's own directory inside the archive holds no symlink components (the validator
        // refuses members beneath a symlink), so climbing it is exact.
        let depth = posix_parts(link_rel).len().saturating_sub(1);
        let climb = vec![".."; depth].join("/");
        return Some(match (climb.is_empty(), rest.is_empty()) {
            (false, false) => format!("{climb}/{rest}"),
            (false, true) => climb,
            (true, false) => rest,
            (true, true) => ".".to_string(),
        });
    }
    None
}

/// Symlink hops followed while proving a rebased target stays under the workspace root. Linux
/// gives up after 40 (`ELOOP`); a workspace that needs more is not worth restoring.
const MAX_SYMLINK_HOPS: usize = 40;

/// Whether a rebased, link-relative `target` provably resolves under the archive root.
///
/// Mirrors Python's `unix_local._symlink_target_stays_under`. The rebase keeps the components
/// after the root verbatim, so `a/link/../tmp` is only inside the workspace when `a/link`
/// resolves inside it. The walk applies `..` to a link's target the way the kernel does and
/// follows only the archive's own relative links, which restore verbatim. Every other component
/// must be established by the archive itself — hydration extracts into an existing root, so a
/// component the archive does not create may already be a symlink in the destination.
fn archive_target_stays_under(
    inventory: &BTreeMap<String, &TarMember>,
    link_rel: &str,
    target: &str,
) -> bool {
    let link_parent = posix_parts(link_rel);
    let link_parent = &link_parent[..link_parent.len().saturating_sub(1)];
    let mut pending: Vec<String> = link_parent
        .iter()
        .cloned()
        .chain(target.split('/').map(str::to_string))
        .rev()
        .collect();
    let mut resolved: Vec<String> = Vec::new();
    let mut hops = 0usize;
    while let Some(part) = pending.pop() {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            if resolved.pop().is_none() {
                return false;
            }
            continue;
        }
        let mut candidate_path = resolved.clone();
        candidate_path.push(part.clone());
        let Some(candidate) = inventory.get(&candidate_path.join("/")) else {
            return false;
        };
        if candidate.kind == ArchiveMemberKind::Symlink {
            hops += 1;
            let link_target = candidate.link_target.clone().unwrap_or_default();
            if hops > MAX_SYMLINK_HOPS || link_target.starts_with('/') {
                return false;
            }
            pending.extend(link_target.split('/').map(str::to_string).rev());
            continue;
        }
        if pending.is_empty() {
            if !matches!(
                candidate.kind,
                ArchiveMemberKind::Directory | ArchiveMemberKind::Regular
            ) {
                return false;
            }
        } else if candidate.kind != ArchiveMemberKind::Directory {
            return false;
        }
        resolved.push(part);
    }
    true
}

/// Copy `raw` member by member, replacing the link target of every member named in `rebased`.
fn rewrite_tar_link_targets(
    raw: &[u8],
    rebased: &BTreeMap<String, String>,
) -> SandboxResult<Vec<u8>> {
    let gzipped = raw.starts_with(&[0x1f, 0x8b]);
    let mut out: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut out);
        let mut archive = tar::Archive::new(tar_reader(raw)?);
        for entry in archive.entries().map_err(invalid_tar_stream)? {
            let mut entry = entry.map_err(invalid_tar_stream)?;
            let mut header = entry.header().clone();
            let path = path_from_bytes(&entry.path_bytes());
            let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
            // `entries()` resolves GNU long names and PAX records before yielding a member, and
            // `append_link`/`append_data` re-emit whatever extension header the name needs, so a
            // rewritten link target cannot be overridden by a stale `linkpath` record.
            let link_target = match rebased.get(&name) {
                Some(rebased) => Some(PathBuf::from(rebased)),
                None => entry.link_name_bytes().map(|bytes| path_from_bytes(&bytes)),
            };
            match link_target {
                Some(target) => builder
                    .append_link(&mut header, &path, &target)
                    .map_err(invalid_tar_stream)?,
                None => {
                    let mut contents = Vec::new();
                    entry
                        .read_to_end(&mut contents)
                        .map_err(invalid_tar_stream)?;
                    builder
                        .append_data(&mut header, &path, &contents[..])
                        .map_err(invalid_tar_stream)?;
                }
            }
        }
        builder.finish().map_err(invalid_tar_stream)?;
    }
    if !gzipped {
        return Ok(out);
    }
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &out).map_err(invalid_tar_stream)?;
    encoder.finish().map_err(invalid_tar_stream)
}

/// The root-relative components of a simple absolute symlink target under `root`.
///
/// Mirrors Python's `tar_utils._simple_rebase_components`. Simple means every component after
/// the root is a plain name: no `.`, `..`, empty segment or trailing separator. The root itself,
/// a target outside it, and any target whose meaning depends on how a symlink component resolves
/// (`alias/../x`) return `None` and stay absolute for the strict hydrate check to refuse.
fn simple_rebase_components(linkname: &str, root: &str) -> Option<Vec<String>> {
    if !linkname.starts_with('/') {
        return None;
    }
    let target = format!("/{}", linkname.trim_start_matches('/'));
    let prefix = format!("/{}", root.trim_matches('/'));
    let rest = if target == prefix {
        String::new()
    } else if let Some(rest) = target.strip_prefix(&format!("{prefix}/")) {
        rest.trim_start_matches('/').to_string()
    } else {
        return None;
    };
    // `/workspace/a.txt/` fails with ENOTDIR when `a.txt` is a file; dropping the separator
    // would turn it into a working link, so it is not a simple target.
    if rest.is_empty() || rest.ends_with('/') {
        return None;
    }
    let parts: Vec<String> = rest
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect();
    if parts.iter().any(|part| part == "." || part == "..") {
        return None;
    }
    Some(parts)
}

/// Whether every component a rebased target walks through is an ordinary archive path.
///
/// Mirrors Python's `_archive_establishes`: hydration extracts into an existing root, so a
/// directory the archive does not create could already be a symlink in the destination and
/// redirect the restored link. Every directory the relative target climbs out of (the link's own
/// parents) and every directory it descends into must be a directory member, and the leaf must be
/// a regular file. A directory target stays absolute.
fn archive_establishes(
    parts: &[String],
    link_name: &str,
    members: &BTreeMap<String, ArchiveMemberKind>,
) -> bool {
    let link_parts = posix_parts(link_name);
    let link_parents = &link_parts[..link_parts.len().saturating_sub(1)];
    for depth in 1..=link_parents.len() {
        if members.get(&link_parents[..depth].join("/")) != Some(&ArchiveMemberKind::Directory) {
            return false;
        }
    }
    for depth in 1..parts.len() {
        if members.get(&parts[..depth].join("/")) != Some(&ArchiveMemberKind::Directory) {
            return false;
        }
    }
    members.get(&parts.join("/")) == Some(&ArchiveMemberKind::Regular)
}

/// Replace a leading member prefix with `.` and make the archive hydratable.
///
/// Mirrors Python's `tar_utils.strip_tar_member_prefix`. Docker archives a workspace copied to
/// `/tmp/stage/workspace` as `workspace/...`; a portable snapshot stores the same files as `.` and
/// `...`, independent of the source backend's root name. FIFOs and device nodes, which a staged
/// `cp -R` legitimately carries and strict hydration refuses, are dropped. When
/// `relativize_symlinks_under` names the workspace root, a *simple* absolute symlink target under it
/// is rebased onto the link's own directory (see `simple_rebase_components` and
/// `archive_establishes`); every other target stays absolute for strict hydration to refuse.
///
/// The result is validated with the default [`TarValidationOptions`] before it is returned, as
/// Python does, so a member that escapes (`..`, an absolute path, a hardlink) fails here.
pub fn strip_tar_member_prefix(
    raw: &[u8],
    prefix: &str,
    relativize_symlinks_under: Option<&str>,
) -> SandboxResult<Vec<u8>> {
    let prefix_rel = normalize_rel(prefix);
    if prefix_rel.as_os_str().is_empty() || prefix_rel == Path::new(".") {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "tar member prefix must not be empty",
        ));
    }
    let prefix_parts = posix_parts(&posix_display(&prefix_rel));

    // First pass: validate each member, compute its stripped name and build the inventory the
    // symlink rewrite needs. Dropped members never reach the inventory.
    struct Planned {
        original: String,
        stripped: String,
    }
    let members = read_tar_members(raw)?;
    let mut planned: Vec<Planned> = Vec::new();
    let mut inventory: BTreeMap<String, ArchiveMemberKind> = BTreeMap::new();
    let mut link_targets: BTreeMap<String, String> = BTreeMap::new();
    for member in &members {
        if member.kind == ArchiveMemberKind::Other {
            continue;
        }
        let rel = safe_tar_member_rel_path(&member.name, member.kind, true)?;
        let stripped = match rel {
            None => ".".to_string(),
            Some(rel) if rel == prefix_rel => ".".to_string(),
            Some(rel) => {
                let parts = posix_parts(&posix_display(&rel));
                if parts.len() >= prefix_parts.len()
                    && parts[..prefix_parts.len()] == prefix_parts[..]
                {
                    parts[prefix_parts.len()..].join("/")
                } else {
                    return Err(unsafe_tar_member(
                        &member.name,
                        format!(
                            "member does not start with prefix: {}",
                            posix_display(&prefix_rel)
                        ),
                    ));
                }
            }
        };
        inventory.insert(stripped.clone(), member.kind);
        if member.kind == ArchiveMemberKind::Symlink
            && let Some(target) = member.link_target.clone()
        {
            link_targets.insert(stripped.clone(), target);
        }
        planned.push(Planned {
            original: member.name.clone(),
            stripped,
        });
    }

    // Second pass: decide which absolute targets can be rebased, now that the inventory is whole.
    let mut rebased: BTreeMap<String, String> = BTreeMap::new();
    if let Some(root) = relativize_symlinks_under {
        for (name, target) in &link_targets {
            let Some(parts) = simple_rebase_components(target, root) else {
                continue;
            };
            if !archive_establishes(&parts, name, &inventory) {
                continue;
            }
            let depth = posix_parts(name).len().saturating_sub(1);
            let climb = vec![".."; depth].join("/");
            let rest = parts.join("/");
            rebased.insert(
                name.clone(),
                if climb.is_empty() {
                    rest
                } else {
                    format!("{climb}/{rest}")
                },
            );
        }
    }

    // Third pass: write the new archive. A rewritten long name or link target is re-emitted with
    // whatever extension header it needs, so a stale PAX `path`/`linkpath` record cannot override it.
    let wanted: BTreeMap<&str, &str> = planned
        .iter()
        .map(|planned| (planned.original.as_str(), planned.stripped.as_str()))
        .collect();
    let mut out: Vec<u8> = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut out);
        let mut archive = tar::Archive::new(tar_reader(raw)?);
        for entry in archive.entries().map_err(invalid_tar_stream)? {
            let mut entry = entry.map_err(invalid_tar_stream)?;
            let original = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
            let Some(stripped) = wanted.get(original.as_str()) else {
                continue;
            };
            let mut header = entry.header().clone();
            let link_target = match rebased.get(*stripped) {
                Some(target) => Some(PathBuf::from(target)),
                None => entry.link_name_bytes().map(|bytes| path_from_bytes(&bytes)),
            };
            let path = Path::new(stripped);
            match link_target {
                Some(target) => builder
                    .append_link(&mut header, path, &target)
                    .map_err(invalid_tar_stream)?,
                None => {
                    let mut contents = Vec::new();
                    entry
                        .read_to_end(&mut contents)
                        .map_err(invalid_tar_stream)?;
                    builder
                        .append_data(&mut header, path, &contents[..])
                        .map_err(invalid_tar_stream)?;
                }
            }
        }
        builder.finish().map_err(invalid_tar_stream)?;
    }
    validate_tar_bytes(
        &out,
        &TarValidationOptions {
            allow_symlinks: true,
            allow_external_symlink_targets: true,
            ..TarValidationOptions::default()
        },
    )?;
    Ok(out)
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;

    PathBuf::from(std::ffi::OsStr::from_bytes(bytes))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

/// One validated member, with its payload, ready to be written through a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    /// The member's path, relative to the extraction root.
    pub rel_path: PathBuf,
    /// Whether the member is a directory.
    pub is_dir: bool,
    /// The file payload, empty for a directory.
    pub contents: Vec<u8>,
}

/// Validate a tar archive and decode every member it extracts.
pub fn validated_tar_entries(
    raw: &[u8],
    options: &TarValidationOptions,
) -> SandboxResult<Vec<ArchiveEntry>> {
    validate_tar_bytes(raw, options)?;

    let mut archive = tar::Archive::new(tar_reader(raw)?);
    let mut entries = Vec::new();
    for entry in archive.entries().map_err(invalid_tar_stream)? {
        let mut entry = entry.map_err(invalid_tar_stream)?;
        let kind = member_kind(entry.header().entry_type());
        let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        if should_skip_tar_member(&name, &options.skip_rel_paths, options.root_name.as_deref()) {
            continue;
        }
        let Some(rel_path) = safe_tar_member_rel_path(&name, kind, options.allow_symlinks)? else {
            continue;
        };
        match kind {
            ArchiveMemberKind::Directory => entries.push(ArchiveEntry {
                rel_path,
                is_dir: true,
                contents: Vec::new(),
            }),
            ArchiveMemberKind::Regular => {
                let mut contents = Vec::new();
                entry
                    .read_to_end(&mut contents)
                    .map_err(invalid_tar_stream)?;
                entries.push(ArchiveEntry {
                    rel_path,
                    is_dir: false,
                    contents,
                });
            }
            // A symlink member reaching here was allowed by the policy; a session's file API
            // cannot create one, so it is left out rather than restored as a regular file.
            _ => continue,
        }
    }
    Ok(entries)
}

/// Validate a zip archive and decode every member it extracts.
///
/// Mirrors Python's `validate_zipfile` plus `WorkspaceArchiveExtractor.extract_zip_archive`.
pub fn validated_zip_entries(
    raw: &[u8],
    limits: Option<&SandboxArchiveLimits>,
) -> SandboxResult<Vec<ArchiveEntry>> {
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(raw)).map_err(invalid_zip_stream)?;
    let mut members_by_rel: BTreeMap<PathBuf, bool> = BTreeMap::new();
    let mut accepted: Vec<(usize, String, PathBuf, bool)> = Vec::new();
    let mut extracted_bytes: u64 = 0;

    for index in 0..archive.len() {
        let member = archive.by_index_raw(index).map_err(invalid_zip_stream)?;
        let name = match member.name() {
            Ok(name) => name.into_owned(),
            // A name that is not UTF-8 still has to be validated, so it is read lossily rather
            // than refused for its encoding alone.
            Err(_) => String::from_utf8_lossy(member.name_raw()).into_owned(),
        };
        let is_dir = member.is_dir();
        let size = member.size();
        let unix_mode = member.unix_mode();
        drop(member);

        let Some(rel_path) = safe_zip_member_rel_path(&name, unix_mode)? else {
            continue;
        };
        check_member_count(accepted.len() as u64 + 1, &name, limits)?;
        extracted_bytes = extracted_bytes.saturating_add(size);
        check_extracted_bytes(extracted_bytes, &name, limits)?;

        if let Some(previous_is_dir) = members_by_rel.get(&rel_path)
            && !(*previous_is_dir && is_dir)
        {
            return Err(unsafe_zip_member(
                &name,
                format!("duplicate archive path: {}", posix_display(&rel_path)),
            ));
        }
        members_by_rel.insert(rel_path.clone(), is_dir);
        accepted.push((index, name, rel_path, is_dir));
    }

    for (_, name, rel_path, _) in &accepted {
        for parent in strict_parents(rel_path) {
            if members_by_rel.get(&parent) == Some(&false) {
                return Err(unsafe_zip_member(
                    name,
                    format!(
                        "archive path descends through non-directory: {}",
                        posix_display(&parent)
                    ),
                ));
            }
        }
    }

    let mut entries = Vec::new();
    for (index, _, rel_path, is_dir) in accepted {
        if is_dir {
            entries.push(ArchiveEntry {
                rel_path,
                is_dir: true,
                contents: Vec::new(),
            });
            continue;
        }
        let mut member = archive.by_index(index).map_err(invalid_zip_stream)?;
        let mut contents = Vec::new();
        member
            .read_to_end(&mut contents)
            .map_err(invalid_zip_stream)?;
        entries.push(ArchiveEntry {
            rel_path,
            is_dir: false,
            contents,
        });
    }
    Ok(entries)
}

/// Validate a zip archive without decoding it. Mirrors Python's `validate_zipfile`.
pub fn validate_zip_bytes(raw: &[u8], limits: Option<&SandboxArchiveLimits>) -> SandboxResult<()> {
    validated_zip_entries(raw, limits).map(|_| ())
}

/// Extract a tar archive into a host directory, validating every member first.
///
/// Mirrors Python's `safe_extract_tarfile`: directories and regular files are restored first and
/// symlinks last, no member is written through an existing symlink, and a regular file keeps the
/// archived permission bits under the same policy as the standard library's `data` filter.
///
/// This is a blocking call; an async caller runs it on a blocking thread.
pub fn extract_tar_to(
    root: &Path,
    raw: &[u8],
    options: &TarValidationOptions,
) -> SandboxResult<()> {
    let members = read_tar_members(raw)?;
    validate_tar_members(&members, options)?;

    std::fs::create_dir_all(root).map_err(|e| {
        SandboxError::new(
            ErrorCode::WorkspaceArchiveWriteError,
            format!("Failed to create `{}`: {e}", root.display()),
        )
    })?;
    let root = std::fs::canonicalize(root).map_err(|e| {
        SandboxError::new(
            ErrorCode::WorkspaceArchiveWriteError,
            format!("Failed to resolve `{}`: {e}", root.display()),
        )
    })?;

    let mut archive = tar::Archive::new(tar_reader(raw)?);
    let mut symlinks: Vec<(PathBuf, String, String)> = Vec::new();
    for entry in archive.entries().map_err(invalid_tar_stream)? {
        let mut entry = entry.map_err(invalid_tar_stream)?;
        let header = entry.header();
        let kind = member_kind(header.entry_type());
        let mode = header.mode().unwrap_or(0o644);
        let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        let link_target = entry
            .link_name_bytes()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        if should_skip_tar_member(&name, &options.skip_rel_paths, options.root_name.as_deref()) {
            continue;
        }
        let Some(rel_path) = safe_tar_member_rel_path(&name, kind, options.allow_symlinks)? else {
            continue;
        };
        let dest = root.join(&rel_path);
        match kind {
            ArchiveMemberKind::Symlink => {
                symlinks.push((rel_path, name, link_target.unwrap_or_default()));
            }
            ArchiveMemberKind::Directory => {
                ensure_no_symlink_parents(&root, &dest, false)?;
                if let Ok(metadata) = std::fs::symlink_metadata(&dest)
                    && !metadata.is_dir()
                {
                    let _ = std::fs::remove_file(&dest);
                }
                std::fs::create_dir_all(&dest).map_err(|e| write_failed(&name, e))?;
            }
            ArchiveMemberKind::Regular => {
                prepare_replaceable_leaf(&root, &dest, &rel_path, &name)?;
                let mut contents = Vec::new();
                entry
                    .read_to_end(&mut contents)
                    .map_err(invalid_tar_stream)?;
                write_regular_file(&dest, &contents, mode).map_err(|e| write_failed(&name, e))?;
            }
            _ => continue,
        }
    }

    for (rel_path, name, target) in symlinks {
        let dest = root.join(&rel_path);
        prepare_replaceable_leaf(&root, &dest, &rel_path, &name)?;
        create_symlink(&target, &dest).map_err(|e| write_failed(&name, e))?;
    }
    Ok(())
}

fn write_failed(member: &str, error: std::io::Error) -> SandboxError {
    SandboxError::new(
        ErrorCode::WorkspaceArchiveWriteError,
        format!("Failed to restore archive member `{member}`: {error}"),
    )
    .with_op(SandboxOp::Write)
    .with_context("member", member)
}

#[cfg(unix)]
fn write_regular_file(dest: &Path, contents: &[u8], mode: u32) -> std::io::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc_o_nofollow())
        .open(dest)?;
    file.write_all(contents)?;
    file.flush()?;
    // The archived bits are restored only once the payload is on disk, so a failed write leaves
    // a private file instead of a readable or executable one.
    file.set_permissions(std::fs::Permissions::from_mode(restored_regular_file_mode(
        mode,
    )))?;
    Ok(())
}

#[cfg(unix)]
fn libc_o_nofollow() -> i32 {
    // `O_NOFOLLOW` is 0o400000 on Linux and 0x0100 on macOS/BSD; both are stable kernel ABI.
    #[cfg(target_os = "linux")]
    {
        0o400000
    }
    #[cfg(not(target_os = "linux"))]
    {
        0x0100
    }
}

#[cfg(not(unix))]
fn write_regular_file(dest: &Path, contents: &[u8], _mode: u32) -> std::io::Result<()> {
    std::fs::write(dest, contents)
}

#[cfg(unix)]
fn create_symlink(target: &str, dest: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, dest)
}

#[cfg(not(unix))]
fn create_symlink(_target: &str, _dest: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "symlink members cannot be restored on this platform",
    ))
}

/// The permission bits restored for an extracted regular file.
///
/// Mirrors Python's `_restored_regular_file_mode`, which in turn mirrors the mode policy of the
/// standard library's `data` filter: setuid, setgid, sticky and group/other write bits are
/// dropped, execute bits survive only when the owner had one, and the owner can always read and
/// write the file.
fn restored_regular_file_mode(mode: u32) -> u32 {
    let mut restored = mode & 0o755;
    if restored & 0o100 == 0 {
        restored &= !0o111;
    }
    restored | 0o600
}

fn prepare_replaceable_leaf(
    root: &Path,
    dest: &Path,
    rel_path: &Path,
    name: &str,
) -> SandboxResult<()> {
    ensure_no_symlink_parents(root, dest, false)?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| write_failed(name, e))?;
    }
    match std::fs::symlink_metadata(dest) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(unsafe_tar_member(
                name,
                format!(
                    "destination directory already exists: {}",
                    posix_display(rel_path)
                ),
            ));
        }
        Ok(_) => {
            std::fs::remove_file(dest).map_err(|e| write_failed(name, e))?;
        }
        Err(_) => {}
    }
    Ok(())
}

/// Refuse a destination whose parents resolve outside `root` or run through an existing symlink.
///
/// Mirrors Python's `_ensure_no_symlink_parents`.
fn ensure_no_symlink_parents(root: &Path, dest: &Path, check_leaf: bool) -> SandboxResult<()> {
    let probe = if check_leaf {
        dest.to_path_buf()
    } else {
        dest.parent().unwrap_or(dest).to_path_buf()
    };
    let resolved = resolve_existing_prefix(&probe);
    if !(resolved == root || resolved.starts_with(root)) {
        return Err(unsafe_tar_member(
            &posix_display(dest),
            "path escapes root after resolution",
        ));
    }
    let Ok(rel) = dest.strip_prefix(root) else {
        return Err(unsafe_tar_member(
            &posix_display(dest),
            "path escapes root after resolution",
        ));
    };
    let parts: Vec<&std::ffi::OsStr> = rel.iter().collect();
    let mut current = root.to_path_buf();
    for part in parts.iter().take(parts.len().saturating_sub(1)) {
        current.push(part);
        if let Ok(metadata) = std::fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            return Err(unsafe_tar_member(
                &posix_display(rel),
                "symlink in parent path",
            ));
        }
    }
    Ok(())
}

/// Canonicalize the longest existing prefix of `path` and re-append the rest.
///
/// `Path::canonicalize` needs the whole path to exist; Python's `Path.resolve()` does not, and
/// the check above runs before the destination is created.
fn resolve_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&existing) {
            let mut resolved = canonical;
            for part in tail.iter().rev() {
                resolved.push(part);
            }
            return resolved;
        }
        let (Some(parent), Some(name)) = (
            existing.parent().map(Path::to_path_buf),
            existing.file_name().map(std::ffi::OsStr::to_os_string),
        ) else {
            return path.to_path_buf();
        };
        tail.push(name);
        if parent.as_os_str().is_empty() {
            return path.to_path_buf();
        }
        existing = parent;
    }
}

fn validate_archive_root_member(name: &str, kind: ArchiveMemberKind) -> SandboxResult<()> {
    match kind {
        ArchiveMemberKind::Directory => Ok(()),
        ArchiveMemberKind::Symlink => Err(unsafe_tar_member(name, "archive root symlink")),
        ArchiveMemberKind::Hardlink => Err(unsafe_tar_member(name, "archive root hardlink")),
        _ => Err(unsafe_tar_member(
            name,
            "archive root member must be directory",
        )),
    }
}

fn raise_if_windows_member_path(
    name: &str,
    error: fn(&str, String) -> SandboxError,
) -> SandboxResult<()> {
    let bytes = name.as_bytes();
    let has_drive = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    if has_drive || name.starts_with("//") || name.starts_with(r"\\") {
        return Err(error(name, "windows drive path".to_string()));
    }
    if name.contains('\\') {
        return Err(error(name, "windows path separator".to_string()));
    }
    Ok(())
}

fn validate_symlink_target(member: &TarMember, rel_path: &Path) -> SandboxResult<()> {
    let target = member.link_target.clone().unwrap_or_default();
    if target.starts_with('/') {
        return Err(unsafe_tar_member(
            &member.name,
            format!("absolute symlink target not allowed: {target}"),
        ));
    }
    let mut parts: Vec<String> = rel_path
        .parent()
        .map(|parent| {
            parent
                .iter()
                .map(|part| part.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    parts.extend(posix_parts(&target));
    if normalize_without_root(&parts).is_none() {
        return Err(unsafe_tar_member(
            &member.name,
            format!("symlink target escapes archive root: {target}"),
        ));
    }
    Ok(())
}

/// Mirrors Python's `_normalize_posix_path_without_root`: `None` when the path climbs out.
fn normalize_without_root(parts: &[String]) -> Option<Vec<String>> {
    let mut normalized: Vec<String> = Vec::new();
    for part in parts {
        if part.is_empty() || part == "." || part == "/" {
            continue;
        }
        if part == ".." {
            normalized.pop()?;
            continue;
        }
        normalized.push(part.clone());
    }
    Some(normalized)
}

fn member_kind(entry_type: tar::EntryType) -> ArchiveMemberKind {
    use tar::EntryType;

    match entry_type {
        EntryType::Directory => ArchiveMemberKind::Directory,
        EntryType::Symlink => ArchiveMemberKind::Symlink,
        EntryType::Link => ArchiveMemberKind::Hardlink,
        EntryType::Regular | EntryType::Continuous | EntryType::GNUSparse => {
            ArchiveMemberKind::Regular
        }
        _ => ArchiveMemberKind::Other,
    }
}

fn tar_reader(raw: &[u8]) -> SandboxResult<Box<dyn Read + '_>> {
    if raw.starts_with(&[0x1f, 0x8b]) {
        return Ok(Box::new(flate2::read::GzDecoder::new(raw)));
    }
    if raw.starts_with(b"BZh") || raw.starts_with(&[0xfd, b'7', b'z', b'X', b'Z', 0x00]) {
        return Err(unsafe_tar_member(
            "<tar>",
            "unsupported archive compression: only tar and tar.gz are read in process",
        ));
    }
    Ok(Box::new(raw))
}

fn check_member_count(
    count: u64,
    member: &str,
    limits: Option<&SandboxArchiveLimits>,
) -> SandboxResult<()> {
    let Some(max) = limits.and_then(|limits| limits.max_members) else {
        return Ok(());
    };
    if count > max {
        return Err(archive_limit_exceeded(
            "archive member count exceeds limit",
            max,
            count,
            Some(member),
        ));
    }
    Ok(())
}

fn check_extracted_bytes(
    total: u64,
    member: &str,
    limits: Option<&SandboxArchiveLimits>,
) -> SandboxResult<()> {
    let Some(max) = limits.and_then(|limits| limits.max_extracted_bytes) else {
        return Ok(());
    };
    if total > max {
        return Err(archive_limit_exceeded(
            "archive extracted size exceeds limit",
            max,
            total,
            Some(member),
        ));
    }
    Ok(())
}

/// Check raw input bytes against [`SandboxArchiveLimits::max_input_bytes`].
pub fn check_input_bytes(total: u64, limits: Option<&SandboxArchiveLimits>) -> SandboxResult<()> {
    let Some(max) = limits.and_then(|limits| limits.max_input_bytes) else {
        return Ok(());
    };
    if total > max {
        return Err(archive_limit_exceeded(
            "archive input size exceeds limit",
            max,
            total,
            None,
        ));
    }
    Ok(())
}

fn posix_parts(path: &str) -> Vec<String> {
    path.split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_string)
        .collect()
}

fn rel_path_from_parts(parts: &[String]) -> PathBuf {
    let mut rel = PathBuf::new();
    for part in parts {
        rel.push(part);
    }
    rel
}

fn posix_display(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            Component::ParentDir => Some("..".to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Every strict ancestor of `rel`, nearest first, excluding the empty root.
fn strict_parents(rel: &Path) -> Vec<PathBuf> {
    rel.ancestors()
        .skip(1)
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .collect()
}

/// Mirrors Python's `_normalize_rel`.
fn normalize_rel(prefix: &str) -> PathBuf {
    rel_path_from_parts(&posix_parts(prefix))
}

/// Mirrors Python's `_is_within`.
fn is_within(path: &Path, prefix: &Path) -> bool {
    prefix.as_os_str().is_empty() || path == prefix || path.starts_with(prefix)
}

/// Mirrors Python's `_tar_member_rel_variants`.
fn tar_member_rel_variants(member_name: &str, root_name: Option<&str>) -> Vec<PathBuf> {
    let parts = posix_parts(member_name);
    if parts.is_empty() {
        return vec![PathBuf::new()];
    }
    let mut variants = vec![rel_path_from_parts(&parts)];
    if let Some(root) = root_name
        && parts[0] == root
    {
        variants.push(rel_path_from_parts(&parts[1..]));
    }
    variants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_traversal_absolute_and_link_members() {
        let cases = [
            (
                "../escape.txt",
                ArchiveMemberKind::Regular,
                "parent traversal",
            ),
            ("/etc/passwd", ArchiveMemberKind::Regular, "absolute path"),
            ("a/../../b", ArchiveMemberKind::Regular, "parent traversal"),
            (
                "link",
                ArchiveMemberKind::Symlink,
                "symlink member not allowed",
            ),
            (
                "hard",
                ArchiveMemberKind::Hardlink,
                "hardlink member not allowed",
            ),
            ("dev", ArchiveMemberKind::Other, "unsupported member type"),
            (
                r"dir\file",
                ArchiveMemberKind::Regular,
                "windows path separator",
            ),
            ("C:/file", ArchiveMemberKind::Regular, "windows drive path"),
        ];
        for (name, kind, reason) in cases {
            let error = safe_tar_member_rel_path(name, kind, false).expect_err(name);
            assert_eq!(
                error.context.get("reason").map(String::as_str),
                Some(reason)
            );
            assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
        }
    }

    #[test]
    fn accepts_ordinary_members_and_the_archive_root() {
        assert_eq!(
            safe_tar_member_rel_path("./a/b.txt", ArchiveMemberKind::Regular, false).unwrap(),
            Some(PathBuf::from("a/b.txt"))
        );
        assert_eq!(
            safe_tar_member_rel_path(".", ArchiveMemberKind::Directory, false).unwrap(),
            None
        );
        assert!(safe_tar_member_rel_path(".", ArchiveMemberKind::Symlink, true).is_err());
        assert_eq!(
            safe_tar_member_rel_path("link", ArchiveMemberKind::Symlink, true).unwrap(),
            Some(PathBuf::from("link"))
        );
    }

    #[test]
    fn skips_members_under_a_workspace_prefix() {
        let skip = vec!["cache".to_string()];
        assert!(should_skip_tar_member("./cache/x", &skip, None));
        assert!(should_skip_tar_member(
            "workspace/cache/x",
            &skip,
            Some("workspace")
        ));
        assert!(!should_skip_tar_member("./src/x", &skip, None));
    }

    #[test]
    fn the_hydrate_policy_refuses_absolute_and_escaping_symlink_targets() {
        for (target, reason) in [
            ("/etc", Some("absolute symlink target not allowed: /etc")),
            (
                "../../outside",
                Some("symlink target escapes archive root: ../../outside"),
            ),
            // A target that stays inside the archive is kept.
            ("../sibling", None),
        ] {
            let members = vec![TarMember {
                name: "sub/link".to_string(),
                kind: ArchiveMemberKind::Symlink,
                size: 0,
                link_target: Some(target.to_string()),
                mode: 0o777,
            }];
            let result = validate_tar_members(&members, &TarValidationOptions::hydrate());
            match reason {
                Some(reason) => assert_eq!(
                    result
                        .expect_err(target)
                        .context
                        .get("reason")
                        .map(String::as_str),
                    Some(reason)
                ),
                None => result.expect(target),
            }
        }
    }

    #[test]
    fn capture_rebases_only_provably_contained_in_root_targets() {
        let roots = vec!["/w".to_string()];
        assert_eq!(
            rebase_symlink_target("/w/lib/y", "bin/x", &roots).as_deref(),
            Some("../lib/y")
        );
        // At the root there is nothing to climb out of.
        assert_eq!(
            rebase_symlink_target("/w/lib/y", "x", &roots).as_deref(),
            Some("lib/y")
        );
        // A link out of the workspace is not this function's to rewrite.
        assert_eq!(rebase_symlink_target("/usr/bin/env", "bin/x", &roots), None);

        let dir = TarMember {
            name: "lib".to_string(),
            kind: ArchiveMemberKind::Directory,
            size: 0,
            link_target: None,
            mode: 0o755,
        };
        let file = TarMember {
            name: "lib/y".to_string(),
            kind: ArchiveMemberKind::Regular,
            size: 2,
            link_target: None,
            mode: 0o644,
        };
        let bin = TarMember {
            name: "bin".to_string(),
            kind: ArchiveMemberKind::Directory,
            size: 0,
            link_target: None,
            mode: 0o755,
        };
        let inventory: BTreeMap<String, &TarMember> = [
            ("lib".to_string(), &dir),
            ("lib/y".to_string(), &file),
            ("bin".to_string(), &bin),
        ]
        .into_iter()
        .collect();
        assert!(archive_target_stays_under(&inventory, "bin/x", "../lib/y"));
        // Every component the target walks through has to be a member of the archive itself:
        // one the archive does not create may already be a symlink in the destination.
        assert!(!archive_target_stays_under(
            &inventory,
            "bin/x",
            "../other/y"
        ));
        // Climbing out of the archive root proves nothing.
        assert!(!archive_target_stays_under(&inventory, "bin/x", "../../y"));
    }

    #[test]
    fn restores_the_python_file_mode_policy() {
        assert_eq!(restored_regular_file_mode(0o777), 0o755);
        assert_eq!(restored_regular_file_mode(0o666), 0o644);
        assert_eq!(restored_regular_file_mode(0o4755), 0o755);
        assert_eq!(restored_regular_file_mode(0o000), 0o600);
    }

    #[test]
    fn infers_the_compression_scheme_from_the_path() {
        assert_eq!(
            CompressionScheme::infer_from_path("/w/a.tar"),
            Some(CompressionScheme::Tar)
        );
        assert_eq!(
            CompressionScheme::infer_from_path("/w/a.zip"),
            Some(CompressionScheme::Zip)
        );
        assert_eq!(CompressionScheme::infer_from_path("/w/archive"), None);
    }
}
