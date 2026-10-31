//! Manifest entries and how they are materialized into a workspace. Mirrors
//! `agents.sandbox.entries`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY;
use crate::sandbox::materialization::{MaterializationResult, MaterializedFile, gather_in_order};
use crate::sandbox::mounts::Mount;
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession};
use crate::sandbox::types::ExecResult;
use crate::sandbox::types::{Group, Permissions, User, file_mode};

/// Default permissions for a directory entry (`rwxr-xr-x`).
fn default_dir_permissions() -> Permissions {
    Permissions {
        owner: file_mode::ALL,
        group: file_mode::READ | file_mode::EXEC,
        other: file_mode::READ | file_mode::EXEC,
        directory: true,
    }
}

/// What a manifest entry materializes into the workspace.
#[derive(Debug, Clone)]
pub enum EntryKind {
    /// A file with inline contents.
    File {
        /// The file contents.
        content: Vec<u8>,
    },
    /// A directory, with optional named children.
    Dir {
        /// Child entries, keyed by their name relative to this directory.
        children: BTreeMap<String, Entry>,
    },
    /// A single file copied from the host filesystem.
    LocalFile {
        /// The host path read by the SDK process.
        src: PathBuf,
    },
    /// A directory tree copied from the host filesystem.
    LocalDir {
        /// The host path read by the SDK process; `None` creates an empty directory.
        src: Option<PathBuf>,
    },
    /// External storage attached inside the workspace.
    Mount {
        /// The mount and its strategy.
        mount: Mount,
    },
    /// A git repository cloned inside the sandbox with the `git` CLI.
    GitRepo {
        /// Host serving the repository (`github.com` by default).
        host: String,
        /// Repository path, for example `owner/name`.
        repo: String,
        /// Tag, branch or commit to check out.
        git_ref: String,
        /// Optional subdirectory of the clone to keep as the entry.
        subpath: Option<String>,
    },
}

/// One manifest entry: what to place in the workspace, and with which metadata.
#[derive(Debug, Clone)]
pub struct Entry {
    /// What the entry materializes.
    pub kind: EntryKind,
    /// Optional description surfaced in the workspace layout instructions.
    pub description: Option<String>,
    /// Permissions applied to the entry after it is written.
    pub permissions: Permissions,
    /// Group or user ownership applied after the entry is written.
    pub group: Option<Group>,
    /// Whether the entry is runtime-attached state that snapshots must exclude and resume must
    /// re-apply. A mount entry takes this from [`Mount::ephemeral`], which defaults to `true`.
    pub ephemeral: bool,
    /// How many files a `local_dir` entry copies at once.
    pub copy_concurrency: usize,
}

impl Entry {
    fn with_kind(kind: EntryKind) -> Self {
        let directory = matches!(
            kind,
            EntryKind::Dir { .. }
                | EntryKind::LocalDir { .. }
                | EntryKind::GitRepo { .. }
                | EntryKind::Mount { .. }
        );
        // A mount carries its own flag, as Python's `Mount.ephemeral` does: it defaults to true,
        // because a mount is runtime-attached storage rather than workspace content, but a mount
        // that declares itself durable must not be overridden into an ephemeral entry — the
        // snapshot and resume paths read the entry's flag, and the two disagreeing is what makes
        // a durable mount get excluded from a snapshot and re-attached on resume anyway.
        let ephemeral = match &kind {
            EntryKind::Mount { mount } => mount.ephemeral,
            _ => false,
        };
        Entry {
            kind,
            description: None,
            permissions: if directory {
                default_dir_permissions()
            } else {
                Permissions::default()
            },
            group: None,
            ephemeral,
            copy_concurrency: DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY,
        }
    }

    /// External storage attached inside the workspace.
    pub fn mount(mount: Mount) -> Self {
        Entry::with_kind(EntryKind::Mount { mount })
    }

    /// Copy at most `concurrency` files at once when materializing a `local_dir` entry.
    ///
    /// Mirrors `SandboxConcurrencyLimits::local_dir_files`.
    pub fn with_copy_concurrency(mut self, concurrency: usize) -> Self {
        self.copy_concurrency = concurrency.max(1);
        self
    }

    /// Mark the entry as runtime-attached state excluded from snapshots.
    pub fn with_ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    /// The mount this entry attaches, when it is a mount entry.
    pub fn as_mount(&self) -> Option<&Mount> {
        match &self.kind {
            EntryKind::Mount { mount } => Some(mount),
            _ => None,
        }
    }

    /// A file with inline contents.
    pub fn file(content: impl Into<Vec<u8>>) -> Self {
        Entry::with_kind(EntryKind::File {
            content: content.into(),
        })
    }

    /// A file with inline UTF-8 text.
    pub fn text_file(content: impl Into<String>) -> Self {
        Entry::file(content.into().into_bytes())
    }

    /// An empty directory.
    pub fn dir() -> Self {
        Entry::with_kind(EntryKind::Dir {
            children: BTreeMap::new(),
        })
    }

    /// A file copied from the host filesystem.
    pub fn local_file(src: impl Into<PathBuf>) -> Self {
        Entry::with_kind(EntryKind::LocalFile { src: src.into() })
    }

    /// A directory tree copied from the host filesystem.
    pub fn local_dir(src: impl Into<PathBuf>) -> Self {
        Entry::with_kind(EntryKind::LocalDir {
            src: Some(src.into()),
        })
    }

    /// A git repository cloned inside the sandbox at `git_ref`.
    pub fn git_repo(repo: impl Into<String>, git_ref: impl Into<String>) -> Self {
        Entry::with_kind(EntryKind::GitRepo {
            host: "github.com".to_string(),
            repo: repo.into(),
            git_ref: git_ref.into(),
            subpath: None,
        })
    }

    /// Set the git host of a [`EntryKind::GitRepo`] entry.
    pub fn with_git_host(mut self, host: impl Into<String>) -> Self {
        if let EntryKind::GitRepo { host: h, .. } = &mut self.kind {
            *h = host.into();
        }
        self
    }

    /// Keep only a subdirectory of a [`EntryKind::GitRepo`] clone.
    pub fn with_git_subpath(mut self, subpath: impl Into<String>) -> Self {
        if let EntryKind::GitRepo { subpath: s, .. } = &mut self.kind {
            *s = Some(subpath.into());
        }
        self
    }

    /// Set the description shown in the workspace layout instructions.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the permissions applied after the entry is written.
    pub fn with_permissions(mut self, permissions: Permissions) -> Self {
        self.permissions = permissions;
        self
    }

    /// Set the owning group applied after the entry is written.
    pub fn with_group(mut self, group: Group) -> Self {
        self.group = Some(group);
        self
    }

    /// Add a child to a [`EntryKind::Dir`] entry.
    pub fn with_child(mut self, name: impl Into<String>, entry: Entry) -> Self {
        if let EntryKind::Dir { children } = &mut self.kind {
            children.insert(name.into(), entry);
        }
        self
    }

    /// Whether the entry becomes a directory in the sandbox.
    pub fn is_dir(&self) -> bool {
        matches!(
            self.kind,
            EntryKind::Dir { .. }
                | EntryKind::LocalDir { .. }
                | EntryKind::GitRepo { .. }
                | EntryKind::Mount { .. }
        )
    }

    /// The entry's Python `type` discriminator, with a mount reported as its own mount type
    /// (`s3_mount`, `gcs_mount`, `r2_mount`, `azure_blob_mount`) rather than the generic
    /// `mount` that [`type_name`](Self::type_name) uses for digests.
    pub fn type_discriminator(&self) -> &'static str {
        if let EntryKind::Mount { mount } = &self.kind
            && let Some(strategy) = mount.strategy.as_any().and_then(|any| {
                any.downcast_ref::<crate::sandbox::mount_entries::ConfiguredMountStrategy>()
            })
        {
            return strategy.mount_type();
        }
        self.type_name()
    }

    /// The entry type name, matching the Python `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self.kind {
            EntryKind::File { .. } => "file",
            EntryKind::Dir { .. } => "dir",
            EntryKind::LocalFile { .. } => "local_file",
            EntryKind::LocalDir { .. } => "local_dir",
            EntryKind::Mount { .. } => "mount",
            EntryKind::GitRepo { .. } => "git_repo",
        }
    }

    /// Materialize this entry at the absolute sandbox path `dest`.
    ///
    /// Returns every file that was written, with the SHA-256 of its contents, so a caller can
    /// tell whether the workspace holds the inputs it expected.
    ///
    /// Host sources are confined to the process working directory, as Python's
    /// `BaseSandboxSession._manifest_base_dir()` does: a `local_file` or `local_dir` outside it
    /// is refused unless the session's manifest grants the path with
    /// [`extra_path_grants`](crate::sandbox::Manifest::extra_path_grants), and a symlink on the
    /// way to the source or inside a `local_dir` tree is refused. Nothing is written to the
    /// sandbox before the check passes. [`apply_confined`](Self::apply_confined) names another
    /// base directory.
    pub async fn apply(
        &self,
        session: &dyn SandboxSession,
        dest: &str,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_confined(session, dest, &default_base_dir())
            .await
    }

    /// Materialize this entry with host sources confined to `base_dir`.
    ///
    /// Mirrors Python's `BaseEntry.apply(session, dest, base_dir)`: a relative source resolves
    /// against `base_dir`, and the refusals are those described on [`apply`](Self::apply).
    pub async fn apply_confined(
        &self,
        session: &dyn SandboxSession,
        dest: &str,
        base_dir: &Path,
    ) -> SandboxResult<MaterializationResult> {
        Box::pin(self.apply_inner(session, dest.to_string(), None, base_dir)).await
    }

    /// Materialize this entry as `user` rather than as the session owner.
    ///
    /// Mirrors the `user=` parameter on Python's `BaseEntry.apply`. A workspace an agent writes
    /// to as a non-root account needs its files owned by that account from the start, otherwise
    /// the first command the agent runs cannot touch them. Permissions and group metadata are
    /// left alone in this mode, exactly as Python does: `chmod`/`chgrp` would run as the session
    /// owner and could undo the ownership the write just established.
    pub async fn apply_as(
        &self,
        session: &dyn SandboxSession,
        dest: &str,
        user: Option<&User>,
    ) -> SandboxResult<MaterializationResult> {
        Box::pin(self.apply_inner(session, dest.to_string(), user, &default_base_dir())).await
    }

    async fn apply_inner(
        &self,
        session: &dyn SandboxSession,
        dest: String,
        user: Option<&User>,
        scope: &Path,
    ) -> SandboxResult<MaterializationResult> {
        let mut written = MaterializationResult::new();
        match &self.kind {
            EntryKind::File { content } => {
                session.write_file_as(&dest, content, user).await?;
                written.push(MaterializedFile::new(dest.clone(), content));
            }
            EntryKind::Dir { children } => {
                session.mkdir_as(&dest, true, user).await?;
                for (name, child) in children {
                    let child_dest = join_sandbox_path(&dest, name)?;
                    written.extend(
                        Box::pin(child.apply_inner(session, child_dest, user, scope)).await?,
                    );
                }
            }
            EntryKind::LocalFile { src } => {
                let src = resolve_host_source(session, src, false, scope)?;
                let content = read_host_file(&src).await?;
                session.write_file_as(&dest, &content, user).await?;
                written.push(MaterializedFile::new(dest.clone(), &content));
            }
            EntryKind::LocalDir { src } => {
                session.mkdir_as(&dest, true, user).await?;
                if let Some(src) = src {
                    let src = resolve_host_source(session, src, true, scope)?;
                    written.extend(
                        copy_host_dir(session, &src, &dest, self.copy_concurrency, user).await?,
                    );
                }
            }
            EntryKind::Mount { mount } => {
                // Mounts attached by the container runtime are already in place; only
                // in-container strategies run a command here.
                if session.attaches_mounts_at_create()
                    && mount.strategy.docker_create_args(mount, &dest).is_some()
                {
                    return Ok(written);
                }
                let mount_path = mount.resolve_mount_path(session.workspace_root(), &dest)?;
                mount.validate()?;
                session.authorize_mount_activation(mount, &mount_path)?;
                mount.strategy.activate(mount, session, &mount_path).await?;
                return Ok(written);
            }
            EntryKind::GitRepo {
                host,
                repo,
                git_ref,
                subpath,
            } => {
                // A clone writes a whole tree, not one file, and the caller already identifies
                // its contents by the ref they asked for, so nothing is recorded here rather
                // than a file record with an empty checksum.
                clone_git_repo(session, host, repo, git_ref, subpath.as_deref(), &dest).await?;
            }
        }
        // Python skips the metadata pass entirely when a user is given: `chmod`/`chgrp` run as
        // the session owner and would fight the ownership the write just set.
        if user.is_none() {
            self.apply_metadata(session, &dest).await?;
        }
        Ok(written)
    }

    async fn apply_metadata(&self, session: &dyn SandboxSession, dest: &str) -> SandboxResult<()> {
        if let Some(group) = &self.group {
            let command = vec!["chgrp".to_string(), group.name.clone(), dest.to_string()];
            let result = session.exec(ExecRequest::argv(command.clone())).await?;
            if !result.ok() {
                return Err(SandboxError::exec_nonzero(&command, &result));
            }
        }
        let command = vec![
            "chmod".to_string(),
            self.permissions.octal_string(),
            dest.to_string(),
        ];
        let result = session.exec(ExecRequest::argv(command.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::exec_nonzero(&command, &result));
        }
        Ok(())
    }
}

/// Join a child name onto an absolute sandbox directory path.
fn join_sandbox_path(dir: &str, name: &str) -> SandboxResult<String> {
    let name = name.trim_matches('/');
    if name.is_empty() || name.split('/').any(|s| s == "..") {
        return Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("manifest entry name `{name}` is not a valid workspace path"),
        ));
    }
    Ok(format!("{}/{}", dir.trim_end_matches('/'), name))
}

async fn read_host_file(src: &Path) -> SandboxResult<Vec<u8>> {
    read_host_file_nofollow(src).await.map_err(|e| {
        SandboxError::new(
            ErrorCode::LocalFileReadError,
            format!("Failed to read host file `{}`: {e}", src.display()),
        )
    })
}

/// Read a host file without ever following a symlink at its final component.
///
/// This is the part of Python's out-of-process `O_DIRECTORY|O_NOFOLLOW` fd-walk worker that
/// actually protects data: the directory scan already classifies entries without following
/// links, but between the scan and the read a concurrent process could replace a regular file
/// with a link to something outside the source tree. Opening with `O_NOFOLLOW` makes that
/// substitution fail with `ELOOP` instead of copying the target into the sandbox.
#[cfg(unix)]
async fn read_host_file_nofollow(src: &Path) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt as _;

    let mut options = tokio::fs::OpenOptions::new();
    options.read(true).custom_flags(libc_o_nofollow());
    let mut file = options.open(src).await?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).await?;
    Ok(contents)
}

#[cfg(not(unix))]
async fn read_host_file_nofollow(src: &Path) -> std::io::Result<Vec<u8>> {
    tokio::fs::read(src).await
}

/// `O_NOFOLLOW`, which is the same value on every platform this crate builds for on unix.
#[cfg(unix)]
const fn libc_o_nofollow() -> i32 {
    #[cfg(target_os = "linux")]
    {
        0o400000
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        0x0100
    }
    #[cfg(any(target_os = "freebsd", target_os = "netbsd", target_os = "openbsd"))]
    {
        0x0100
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd"
    )))]
    {
        0
    }
}

/// The directory host sources are confined to by default: the process working directory.
fn default_base_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
}

/// `os.path.abspath`: absolute, with `.` and `..` collapsed textually and no symlink resolved.
fn lexical_absolute(path: &Path) -> PathBuf {
    use std::path::Component;

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn host_source_error(is_dir: bool, src: &Path, message: String) -> SandboxError {
    let code = if is_dir {
        ErrorCode::LocalDirReadError
    } else {
        ErrorCode::LocalFileReadError
    };
    SandboxError::new(code, message).with_context("src", src.display().to_string())
}

fn symlink_not_supported(src: &Path, is_dir: bool, child: &str) -> SandboxError {
    host_source_error(
        is_dir,
        src,
        format!(
            "Refusing to read `{}`: `{child}` is a symlink, and symlinks in a host source are \
             not supported",
            src.display()
        ),
    )
    .with_context("reason", "symlink_not_supported")
    .with_context("child", child)
}

/// Check a host source against `base_dir` and return the absolute path to read.
///
/// Mirrors Python's `LocalDir._resolve_local_dir_src_root`: containment is judged on the
/// lexically normalized path, so a `..` cannot be used to step out and a symlink cannot be used
/// to step out either, because every component on the way is looked at without being followed.
fn resolve_host_source(
    session: &dyn SandboxSession,
    src: &Path,
    is_dir: bool,
    base_dir: &Path,
) -> SandboxResult<PathBuf> {
    let base = lexical_absolute(base_dir);
    let input = lexical_absolute(&base.join(src));
    let grants = session.manifest().extra_path_grants();
    let granted = grants
        .iter()
        .any(|grant| input.starts_with(lexical_absolute(Path::new(grant.host_source()))));
    if !input.starts_with(&base) && !granted {
        let mut error = host_source_error(
            is_dir,
            &input,
            format!(
                "Refusing to read `{}`: it is outside `{}` and no extra path grant covers it",
                input.display(),
                base.display()
            ),
        )
        .with_context("reason", "outside_base_dir")
        .with_context("base_dir", base.display().to_string());
        if !grants.is_empty() {
            let paths: Vec<&str> = grants.iter().map(|grant| grant.path.as_str()).collect();
            error = error.with_context(
                "extra_path_grants",
                serde_json::to_string(&paths).unwrap_or_default(),
            );
        }
        return Err(error);
    }
    // Walk the path as written, one component at a time, without following anything.
    let (mut current, parts): (PathBuf, Vec<std::ffi::OsString>) = if src.is_absolute() {
        let mut components = src.components();
        let root: PathBuf = components
            .next()
            .map(|component| PathBuf::from(component.as_os_str()))
            .unwrap_or_default();
        (
            root,
            components
                .filter(|component| !matches!(component, std::path::Component::CurDir))
                .map(|component| component.as_os_str().to_owned())
                .collect(),
        )
    } else {
        (
            base.clone(),
            src.components()
                .filter(|component| !matches!(component, std::path::Component::CurDir))
                .map(|component| component.as_os_str().to_owned())
                .collect(),
        )
    };
    for part in parts {
        current.push(part);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let child = current
                    .strip_prefix(&base)
                    .unwrap_or(&current)
                    .to_string_lossy()
                    .replace('\\', "/");
                return Err(symlink_not_supported(&input, is_dir, &child));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(host_source_error(
                    is_dir,
                    &input,
                    format!("Host source `{}` does not exist", input.display()),
                )
                .with_context("reason", "path_not_found"));
            }
            Err(error) => {
                return Err(host_source_error(
                    is_dir,
                    &current,
                    format!("Failed to inspect `{}`: {error}", current.display()),
                ));
            }
        }
    }
    Ok(input)
}

/// Copy a host directory tree into the sandbox, one file at a time.
///
/// The tree is walked through directory file descriptors: the source root is opened once without
/// following a link, every subdirectory is opened relative to its already-open parent, and every
/// file is opened relative to the descriptor of the directory it was listed in. A directory or
/// file swapped for a link after it was validated therefore fails the open instead of being
/// followed, and a directory swapped away after it was opened still yields the files that were
/// listed in it (the same pinning Python's `dir_fd` walk gives).
#[cfg(unix)]
async fn copy_host_dir(
    session: &dyn SandboxSession,
    src: &Path,
    dest: &str,
    concurrency: usize,
    user: Option<&User>,
) -> SandboxResult<MaterializationResult> {
    use std::sync::Arc;

    // Files are copied with a bounded number in flight: one at a time is slow against a hosted
    // backend where each write is a round trip, and all at once exhausts its connections. The
    // whole tree is walked first so the copies keep the order the walk discovered them in,
    // whatever order the backend finishes them in.
    let root_label = src
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let root_fd = pinned::open_root(src).map_err(|e| pinned::open_error(src, &root_label, e))?;
    let mut files: Vec<(Arc<pinned::Fd>, std::ffi::CString, String, String)> = Vec::new();
    let mut pending: Vec<(Arc<pinned::Fd>, String, String)> =
        vec![(Arc::new(root_fd), String::new(), dest.to_string())];
    while let Some((dir_fd, rel_dir, sandbox_dir)) = pending.pop() {
        let mut children = pinned::list_dir(&dir_fd).map_err(|e| {
            SandboxError::new(
                ErrorCode::LocalDirReadError,
                format!(
                    "Failed to read host directory `{}`: {e}",
                    src.join(&rel_dir).display()
                ),
            )
        })?;
        // Directory order is the filesystem's, so sort: the result lists files in the same order
        // on every run.
        children.sort_by(|left, right| left.name.cmp(&right.name));
        for child in children {
            let name = child.name.to_string_lossy().into_owned();
            let rel = if rel_dir.is_empty() {
                name.clone()
            } else {
                format!("{rel_dir}/{name}")
            };
            match child.kind {
                pinned::Kind::Symlink => {
                    // Following a link would copy host state from outside `src`, and skipping
                    // it would hand the sandbox a tree that silently differs from the one named.
                    return Err(symlink_not_supported(src, true, &rel));
                }
                pinned::Kind::Dir => {
                    let target = join_sandbox_path(&sandbox_dir, &name)?;
                    session.mkdir_as(&target, true, user).await?;
                    // Opened only now, relative to the pinned parent, so a swap between the
                    // listing and this open is caught.
                    let fd = pinned::open_dir_at(&dir_fd, &child.name)
                        .map_err(|e| pinned::open_error(src, &rel, e))?;
                    pending.push((Arc::new(fd), rel, target));
                }
                pinned::Kind::File => {
                    let target = join_sandbox_path(&sandbox_dir, &name)?;
                    files.push((Arc::clone(&dir_fd), child.name, rel, target));
                }
                pinned::Kind::Other => {}
            }
        }
    }
    let copies = files.into_iter().map(|(dir_fd, name, rel, target)| {
        move || async move {
            let content = pinned::read_file_at(&dir_fd, &name)
                .await
                .map_err(|e| pinned::open_error(src, &rel, e))?;
            session.write_file_as(&target, &content, user).await?;
            Ok::<_, SandboxError>(MaterializedFile::new(target, &content))
        }
    });
    let written = gather_in_order(copies, Some(concurrency.max(1))).await?;
    Ok(MaterializationResult::from_iter(written))
}

/// Descriptor-relative host reads that never follow a link.
#[cfg(unix)]
mod pinned {
    use std::ffi::{CStr, CString};
    use std::os::fd::OwnedFd;
    use std::path::Path;

    use rustix::fs::{AtFlags, Dir, FileType, Mode, OFlags, openat, statat};
    use rustix::io::Errno;
    use tokio::io::AsyncReadExt as _;

    use super::{SandboxError, host_source_error, symlink_not_supported};

    pub(super) type Fd = OwnedFd;

    pub(super) enum Kind {
        Symlink,
        Dir,
        File,
        Other,
    }

    pub(super) struct Child {
        pub(super) name: CString,
        pub(super) kind: Kind,
    }

    const DIR_FLAGS: OFlags = OFlags::RDONLY
        .union(OFlags::DIRECTORY)
        .union(OFlags::NOFOLLOW)
        .union(OFlags::CLOEXEC);

    pub(super) fn open_root(path: &Path) -> std::io::Result<Fd> {
        Ok(rustix::fs::open(path, DIR_FLAGS, Mode::empty())?)
    }

    pub(super) fn open_dir_at(dir: &Fd, name: &CStr) -> std::io::Result<Fd> {
        Ok(openat(dir, name, DIR_FLAGS, Mode::empty())?)
    }

    pub(super) fn list_dir(dir: &Fd) -> std::io::Result<Vec<Child>> {
        let mut children = Vec::new();
        for entry in Dir::read_from(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_bytes() == b"." || name.to_bytes() == b".." {
                continue;
            }
            let mut file_type = entry.file_type();
            if file_type == FileType::Unknown {
                file_type =
                    FileType::from_raw_mode(statat(dir, name, AtFlags::SYMLINK_NOFOLLOW)?.st_mode);
            }
            let kind = match file_type {
                FileType::Symlink => Kind::Symlink,
                FileType::Directory => Kind::Dir,
                FileType::RegularFile => Kind::File,
                _ => Kind::Other,
            };
            children.push(Child {
                name: name.to_owned(),
                kind,
            });
        }
        Ok(children)
    }

    /// Open `name` under `dir` without following a link, refusing anything but a regular file
    /// (a FIFO swapped in would otherwise block the read), and read it.
    pub(super) async fn read_file_at(dir: &Fd, name: &CStr) -> std::io::Result<Vec<u8>> {
        let fd = openat(
            dir,
            name,
            OFlags::RDONLY
                .union(OFlags::NOFOLLOW)
                .union(OFlags::NONBLOCK)
                .union(OFlags::CLOEXEC),
            Mode::empty(),
        )?;
        let file = std::fs::File::from(fd);
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::other("path_changed_during_copy"));
        }
        let mut file = tokio::fs::File::from_std(file);
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).await?;
        Ok(contents)
    }

    /// Map an open failure onto the refusal Python reports: a link where a directory or file was
    /// expected is `symlink_not_supported`, any other change is `path_changed_during_copy`.
    pub(super) fn open_error(src: &Path, child: &str, error: std::io::Error) -> SandboxError {
        let errno = error.raw_os_error().map(Errno::from_raw_os_error);
        if matches!(errno, Some(Errno::LOOP | Errno::NOTDIR)) {
            return symlink_not_supported(src, true, child);
        }
        host_source_error(
            true,
            src,
            format!(
                "Host source `{}` changed during the copy at `{child}`: {error}",
                src.display()
            ),
        )
        .with_context("reason", "path_changed_during_copy")
        .with_context("child", child)
    }
}

/// Copy a host directory tree into the sandbox, one file at a time.
#[cfg(not(unix))]
async fn copy_host_dir(
    session: &dyn SandboxSession,
    src: &Path,
    dest: &str,
    concurrency: usize,
    user: Option<&User>,
) -> SandboxResult<MaterializationResult> {
    // Files are copied with a bounded number in flight: one at a time is slow against a hosted
    // backend where each write is a round trip, and all at once exhausts its connections. The
    // whole tree is walked first so the copies keep the order the walk discovered them in,
    // whatever order the backend finishes them in.
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    let mut pending = vec![(src.to_path_buf(), dest.to_string())];
    while let Some((host_dir, sandbox_dir)) = pending.pop() {
        let mut children: Vec<(String, PathBuf, std::fs::FileType)> = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&host_dir).await.map_err(|e| {
            SandboxError::new(
                ErrorCode::LocalDirReadError,
                format!(
                    "Failed to read host directory `{}`: {e}",
                    host_dir.display()
                ),
            )
        })?;
        loop {
            let entry = read_dir.next_entry().await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::LocalDirReadError,
                    format!(
                        "Failed to read host directory `{}`: {e}",
                        host_dir.display()
                    ),
                )
            })?;
            let Some(entry) = entry else { break };
            let file_type = entry.file_type().await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::LocalDirReadError,
                    format!("Failed to stat `{}`: {e}", entry.path().display()),
                )
            })?;
            children.push((
                entry.file_name().to_string_lossy().into_owned(),
                entry.path(),
                file_type,
            ));
        }
        // Directory order is the filesystem's, so sort: the result lists files in the same order
        // on every run.
        children.sort_by(|left, right| left.0.cmp(&right.0));
        for (name, path, file_type) in children {
            if file_type.is_symlink() {
                // Following a link would copy host state from outside `src`, and skipping it
                // would hand the sandbox a tree that silently differs from the one named.
                let child = path
                    .strip_prefix(src)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                return Err(symlink_not_supported(src, true, &child));
            }
            let target = join_sandbox_path(&sandbox_dir, &name)?;
            if file_type.is_dir() {
                session.mkdir_as(&target, true, user).await?;
                pending.push((path, target));
            } else if file_type.is_file() {
                files.push((path, target));
            }
        }
    }
    let copies = files.into_iter().map(|(host_path, target)| {
        move || async move {
            let content = read_host_file(&host_path).await?;
            session.write_file_as(&target, &content, user).await?;
            Ok::<_, SandboxError>(MaterializedFile::new(target, &content))
        }
    });
    let written = gather_in_order(copies, Some(concurrency.max(1))).await?;
    Ok(MaterializationResult::from_iter(written))
}

/// Clone a git repository inside the sandbox with the `git` CLI.
///
/// Mirrors Python's `GitRepo.apply`: the repository is acquired into a scratch directory under
/// `/tmp`, the wanted tree (the whole clone, or `subpath` inside it) is copied over `dest`, and
/// the scratch directory is removed however the copy ends. Cloning straight into `dest` would
/// leave a half-populated workspace behind on failure, and would put `.git` next to the files
/// when only a subdirectory was asked for.
async fn clone_git_repo(
    session: &dyn SandboxSession,
    host: &str,
    repo: &str,
    git_ref: &str,
    subpath: Option<&str>,
    dest: &str,
) -> SandboxResult<()> {
    let git_subpath = validate_git_subpath(repo, subpath)?;

    let probe = session
        .exec(ExecRequest::shell("command -v git >/dev/null 2>&1"))
        .await?;
    if !probe.ok() {
        return Err(SandboxError::new(
            ErrorCode::GitMissingInImage,
            "`git` is not available inside the sandbox image",
        )
        .with_context("repo", repo)
        .with_context("ref", git_ref));
    }

    let tmp_dir = format!(
        "/tmp/sandbox-git-{}-{}",
        crate::sandbox::mount_patterns::session_scratch_id(session),
        uuid::Uuid::new_v4().simple()
    );
    let url = format!("https://{host}/{repo}.git");

    remove_scratch_dir(session, &tmp_dir).await?;
    let outcome = acquire_and_copy_git_repo(
        session,
        repo,
        git_ref,
        subpath,
        git_subpath.as_deref(),
        &url,
        &tmp_dir,
        dest,
    )
    .await;
    // The scratch clone goes whether or not the copy worked; a failed cleanup must not hide the
    // error that explains why the copy failed.
    let cleanup = remove_scratch_dir(session, &tmp_dir).await;
    outcome?;
    cleanup
}

async fn remove_scratch_dir(session: &dyn SandboxSession, tmp_dir: &str) -> SandboxResult<()> {
    session
        .exec(ExecRequest::argv(["rm", "-rf", "--", tmp_dir]).with_shell(ExecShell::None))
        .await
        .map(|_| ())
}

async fn git(session: &dyn SandboxSession, args: &[&str]) -> SandboxResult<ExecResult> {
    let mut command = vec!["git"];
    command.extend_from_slice(args);
    session
        .exec(ExecRequest::argv(command).with_shell(ExecShell::None))
        .await
}

#[allow(clippy::too_many_arguments)]
async fn acquire_and_copy_git_repo(
    session: &dyn SandboxSession,
    repo: &str,
    git_ref: &str,
    subpath: Option<&str>,
    git_subpath: Option<&str>,
    url: &str,
    tmp_dir: &str,
    dest: &str,
) -> SandboxResult<()> {
    // A commit sha and a branch name need different commands: `clone --branch` does not accept a
    // sha, and fetching a sha fails on a server that does not allow it. A ref that looks like a
    // sha is fetched first and falls back to a named clone, so a tag that happens to be all hex
    // still resolves, and the first failure is the one reported.
    let mut first_error: Option<ExecResult> = None;
    let mut clone = if looks_like_commit_ref(git_ref) {
        let fetched = fetch_commit_ref(session, url, tmp_dir, git_ref).await?;
        if fetched.ok() {
            fetched
        } else {
            first_error = Some(fetched);
            remove_scratch_dir(session, tmp_dir).await?;
            clone_named_ref(session, url, tmp_dir, git_ref).await?
        }
    } else {
        clone_named_ref(session, url, tmp_dir, git_ref).await?
    };
    if !clone.ok() {
        if let Some(first) = first_error {
            clone = first;
        }
        let mut error = SandboxError::new(
            ErrorCode::GitCloneError,
            format!(
                "Failed to clone `{url}` at `{git_ref}`: {}",
                clone.stderr_text().trim()
            ),
        )
        .with_op(crate::sandbox::errors::SandboxOp::Materialize)
        .with_context("repo", repo);
        if let Some(subpath) = subpath {
            error = error.with_context("subpath", subpath);
        }
        return Err(error);
    }

    let src_root = match git_subpath {
        Some(subpath) => format!("{tmp_dir}/{subpath}"),
        None => tmp_dir.to_string(),
    };
    session.mkdir(dest, true).await?;
    let copy = session
        .exec(
            ExecRequest::argv([
                "cp".to_string(),
                "-R".to_string(),
                "--".to_string(),
                format!("{src_root}/."),
                format!("{}/", dest.trim_end_matches('/')),
            ])
            .with_shell(ExecShell::None),
        )
        .await?;
    if !copy.ok() {
        let mut error = SandboxError::new(
            ErrorCode::GitCopyError,
            format!(
                "Failed to copy `{src_root}` to `{dest}`: {}",
                copy.stderr_text().trim()
            ),
        )
        .with_op(crate::sandbox::errors::SandboxOp::Materialize)
        .with_context("repo", repo)
        .with_context("ref", git_ref);
        if let Some(subpath) = subpath {
            error = error.with_context("subpath", subpath);
        }
        return Err(error);
    }
    Ok(())
}

async fn clone_named_ref(
    session: &dyn SandboxSession,
    url: &str,
    tmp_dir: &str,
    git_ref: &str,
) -> SandboxResult<ExecResult> {
    git(
        session,
        &[
            "clone",
            "--depth",
            "1",
            "--no-tags",
            "--branch",
            git_ref,
            url,
            tmp_dir,
        ],
    )
    .await
}

/// Fetch exactly one commit into a fresh repository, stopping at the first step that fails.
async fn fetch_commit_ref(
    session: &dyn SandboxSession,
    url: &str,
    tmp_dir: &str,
    git_ref: &str,
) -> SandboxResult<ExecResult> {
    let steps: [&[&str]; 4] = [
        &["init", tmp_dir],
        &["-C", tmp_dir, "remote", "add", "origin", url],
        &[
            "-C",
            tmp_dir,
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            git_ref,
        ],
        &["-C", tmp_dir, "checkout", "--detach", "FETCH_HEAD"],
    ];
    let mut last = ExecResult::default();
    for step in steps {
        last = git(session, step).await?;
        if !last.ok() {
            return Ok(last);
        }
    }
    Ok(last)
}

/// Whether `git_ref` looks like a commit sha. Mirrors Python's `_COMMIT_REF_RE`.
fn looks_like_commit_ref(git_ref: &str) -> bool {
    (7..=40).contains(&git_ref.len()) && git_ref.chars().all(|c| c.is_ascii_hexdigit())
}

/// Validate a git `subpath` and reduce it to the POSIX path to copy, or `None` for the repo root.
///
/// Mirrors Python's `GitRepo._validate_subpath`. An empty string and every spelling of "here"
/// (`.`, `./`, `./.`) mean the root; a blank one, an absolute one, one using Windows separators
/// or a drive, and one that climbs out of the clone are refused before anything runs.
fn validate_git_subpath(repo: &str, subpath: Option<&str>) -> SandboxResult<Option<String>> {
    let Some(original) = subpath else {
        return Ok(None);
    };
    if original.is_empty() {
        return Ok(None);
    }
    let refuse = |reason: &str| {
        SandboxError::new(
            ErrorCode::GitSubpathError,
            format!("Invalid git subpath `{original}` for `{repo}`: {reason}"),
        )
        .with_op(crate::sandbox::errors::SandboxOp::Materialize)
        .with_context("repo", repo)
        .with_context("subpath", original)
        .with_context("reason", reason)
    };
    let trimmed = original.trim();
    if trimmed.is_empty() {
        return Err(refuse("empty"));
    }
    let parts: Vec<&str> = trimmed
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if trimmed.starts_with('/') {
        return Err(refuse("absolute"));
    }
    if original.contains('\\') || crate::sandbox::workspace_paths::is_windows_absolute(trimmed) {
        return Err(refuse("windows_path"));
    }
    if parts.contains(&"..") {
        return Err(refuse("parent_traversal"));
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(parts.join("/")))
}

/// Manifest users and groups provisioned before entries are materialized.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Accounts {
    /// Users to create.
    pub users: Vec<User>,
    /// Groups to create.
    pub groups: Vec<Group>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tells_a_commit_sha_from_a_branch_name() {
        assert!(looks_like_commit_ref("0123abc"));
        assert!(looks_like_commit_ref(&"a".repeat(40)));
        // Too short, too long, or not hex: treated as a named ref.
        assert!(!looks_like_commit_ref("abc"));
        assert!(!looks_like_commit_ref(&"a".repeat(41)));
        assert!(!looks_like_commit_ref("main"));
        assert!(!looks_like_commit_ref("release/1.0"));
    }

    #[test]
    fn marks_mounts_ephemeral() {
        use crate::sandbox::mounts::BindMountStrategy;

        let entry = Entry::mount(BindMountStrategy::mount("/host/data"));
        assert!(entry.ephemeral);
        assert!(entry.is_dir());
        assert_eq!(entry.type_name(), "mount");
        assert!(entry.as_mount().is_some());
        assert!(!Entry::text_file("x").ephemeral);
        assert!(Entry::text_file("x").as_mount().is_none());
    }

    #[test]
    fn a_mount_entry_follows_the_mounts_own_ephemeral_flag() {
        use crate::sandbox::mounts::BindMountStrategy;

        let durable = Entry::mount(BindMountStrategy::mount("/host/data").with_ephemeral(false));
        assert!(
            !durable.ephemeral,
            "a durable mount must not become an ephemeral entry"
        );
        // The manifest reads the entry's flag, so a durable mount stays in the snapshot and is
        // not re-applied on resume.
        let manifest = crate::sandbox::manifest::Manifest::new().with_entry("data", durable);
        assert!(manifest.ephemeral_mount_targets().is_empty());
        assert!(manifest.ephemeral_entry_paths().is_empty());

        let manifest = crate::sandbox::manifest::Manifest::new().with_entry(
            "cache",
            Entry::mount(BindMountStrategy::mount("/host/cache")),
        );
        assert_eq!(manifest.ephemeral_mount_targets().len(), 1);
    }

    #[test]
    fn builds_entries() {
        let entry = Entry::dir()
            .with_child("task.md", Entry::text_file("hello"))
            .with_description("the repo");
        assert!(entry.is_dir());
        assert_eq!(entry.type_name(), "dir");
        assert_eq!(entry.description.as_deref(), Some("the repo"));
        let EntryKind::Dir { children } = &entry.kind else {
            panic!("expected a dir");
        };
        assert_eq!(children["task.md"].type_name(), "file");
        assert!(!children["task.md"].is_dir());
    }

    #[test]
    fn rejects_unsafe_child_names() {
        assert!(join_sandbox_path("/workspace", "..").is_err());
        assert!(join_sandbox_path("/workspace", "a/../../b").is_err());
        assert_eq!(
            join_sandbox_path("/workspace/", "a").unwrap(),
            "/workspace/a"
        );
    }
}
