//! Manifest entries and how they are materialized into a workspace. Mirrors
//! `agents.sandbox.entries`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY;
use crate::sandbox::materialization::{MaterializationResult, MaterializedFile};
use crate::sandbox::mounts::Mount;
use crate::sandbox::session::{ExecRequest, SandboxSession, shell_quote};
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
    /// re-apply. Mounts are always ephemeral.
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
        let ephemeral = matches!(kind, EntryKind::Mount { .. });
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

    /// Mark the entry as runtime-attached state excluded from snapshots.
    /// Copy at most `concurrency` files at once when materializing a `local_dir` entry.
    ///
    /// Mirrors `SandboxConcurrencyLimits::local_dir_files`.
    pub fn with_copy_concurrency(mut self, concurrency: usize) -> Self {
        self.copy_concurrency = concurrency.max(1);
        self
    }

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
    pub async fn apply(
        &self,
        session: &dyn SandboxSession,
        dest: &str,
    ) -> SandboxResult<MaterializationResult> {
        Box::pin(self.apply_inner(session, dest.to_string(), None)).await
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
        Box::pin(self.apply_inner(session, dest.to_string(), user)).await
    }

    async fn apply_inner(
        &self,
        session: &dyn SandboxSession,
        dest: String,
        user: Option<&User>,
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
                    written.extend(Box::pin(child.apply_inner(session, child_dest, user)).await?);
                }
            }
            EntryKind::LocalFile { src } => {
                let content = read_host_file(src).await?;
                session.write_file_as(&dest, &content, user).await?;
                written.push(MaterializedFile::new(dest.clone(), &content));
            }
            EntryKind::LocalDir { src } => {
                session.mkdir_as(&dest, true, user).await?;
                if let Some(src) = src {
                    written.extend(
                        copy_host_dir(session, src, &dest, self.copy_concurrency, user).await?,
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

/// Copy a host directory tree into the sandbox, one file at a time.
async fn copy_host_dir(
    session: &dyn SandboxSession,
    src: &Path,
    dest: &str,
    concurrency: usize,
    user: Option<&User>,
) -> SandboxResult<MaterializationResult> {
    let mut written = MaterializationResult::new();
    // Files are copied in bounded batches: one at a time is slow against a hosted backend where
    // each write is a round trip, and all at once exhausts its connections.
    let mut batch: Vec<(PathBuf, String)> = Vec::new();
    let mut pending = vec![(src.to_path_buf(), dest.to_string())];
    while let Some((host_dir, sandbox_dir)) = pending.pop() {
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
            let name = entry.file_name().to_string_lossy().into_owned();
            let target = join_sandbox_path(&sandbox_dir, &name)?;
            let file_type = entry.file_type().await.map_err(|e| {
                SandboxError::new(
                    ErrorCode::LocalDirReadError,
                    format!("Failed to stat `{}`: {e}", entry.path().display()),
                )
            })?;
            if file_type.is_dir() {
                session.mkdir_as(&target, true, user).await?;
                pending.push((entry.path(), target));
            } else if file_type.is_file() {
                batch.push((entry.path(), target));
                if batch.len() >= concurrency.max(1) {
                    written.extend(copy_batch(session, std::mem::take(&mut batch), user).await?);
                }
            }
            // Symlinks are skipped: following them would copy host state from outside `src`.
        }
    }
    written.extend(copy_batch(session, batch, user).await?);
    Ok(written)
}

/// Copy one batch of files concurrently, preserving the order they were discovered in.
async fn copy_batch(
    session: &dyn SandboxSession,
    batch: Vec<(PathBuf, String)>,
    user: Option<&User>,
) -> SandboxResult<MaterializationResult> {
    if batch.is_empty() {
        return Ok(MaterializationResult::new());
    }
    let copies = batch.into_iter().map(|(host_path, target)| async move {
        let content = read_host_file(&host_path).await?;
        session.write_file_as(&target, &content, user).await?;
        Ok::<_, SandboxError>(MaterializedFile::new(target, &content))
    });
    let files = futures::future::try_join_all(copies).await?;
    Ok(MaterializationResult::from_iter(files))
}

/// Clone a git repository inside the sandbox with the `git` CLI.
async fn clone_git_repo(
    session: &dyn SandboxSession,
    host: &str,
    repo: &str,
    git_ref: &str,
    subpath: Option<&str>,
    dest: &str,
) -> SandboxResult<()> {
    let probe = session
        .exec(ExecRequest::shell("command -v git >/dev/null 2>&1"))
        .await?;
    if !probe.ok() {
        return Err(SandboxError::new(
            ErrorCode::GitMissingInImage,
            "`git` is not available inside the sandbox image",
        ));
    }
    let url = format!("https://{host}/{repo}.git");
    let clone_target = match subpath {
        None => dest.to_string(),
        Some(_) => format!("{}.gitclone", dest.trim_end_matches('/')),
    };

    // A commit sha and a branch name need different commands: `clone --branch` does not accept a
    // sha, and fetching a sha fails on a server that does not allow it. Mirrors Python's
    // `_looks_like_commit_ref` fallback, which tries the fetch first for a sha and falls back to
    // a named clone, so a tag that happens to be all hex still resolves.
    let mut result = if looks_like_commit_ref(git_ref) {
        let fetched = session
            .exec(ExecRequest::shell(fetch_commit_script(
                &clone_target,
                &url,
                git_ref,
            )))
            .await?;
        if fetched.ok() {
            fetched
        } else {
            session
                .exec(ExecRequest::shell(clone_named_ref_script(
                    &clone_target,
                    &url,
                    git_ref,
                )))
                .await?
        }
    } else {
        session
            .exec(ExecRequest::shell(clone_named_ref_script(
                &clone_target,
                &url,
                git_ref,
            )))
            .await?
    };
    if !result.ok() && !looks_like_commit_ref(git_ref) {
        // A named ref that is not a branch or tag may still be reachable by sha.
        result = session
            .exec(ExecRequest::shell(fetch_commit_script(
                &clone_target,
                &url,
                git_ref,
            )))
            .await?;
    }
    if !result.ok() {
        return Err(SandboxError::new(
            ErrorCode::GitCloneError,
            format!(
                "Failed to clone `{url}` at `{git_ref}`: {}",
                result.stderr_text().trim()
            ),
        )
        .with_op(crate::sandbox::errors::SandboxOp::Materialize)
        .with_context("repo", repo.to_string()));
    }
    if let Some(subpath) = subpath {
        let source = join_sandbox_path(&clone_target, subpath)?;
        let script = format!(
            "set -e\ntest -d {source}\nrm -rf {dest}\nmv {source} {dest}\nrm -rf {clone}",
            source = shell_quote(&source),
            dest = shell_quote(dest),
            clone = shell_quote(&clone_target),
        );
        let result = session.exec(ExecRequest::shell(script)).await?;
        if !result.ok() {
            // Tell "the subpath is not in this tree" apart from "the copy failed": the first is
            // the caller's manifest to fix, the second is the sandbox's filesystem.
            let present = session
                .exec(ExecRequest::shell(format!(
                    "test -d {}",
                    shell_quote(&source)
                )))
                .await?;
            let code = if present.ok() {
                ErrorCode::GitCopyError
            } else {
                ErrorCode::GitSubpathError
            };
            return Err(SandboxError::new(
                code,
                format!(
                    "Subpath `{subpath}` could not be materialized from `{url}` at `{git_ref}`: {}",
                    result.stderr_text().trim()
                ),
            )
            .with_op(crate::sandbox::errors::SandboxOp::Materialize));
        }
    }
    Ok(())
}

/// Whether `git_ref` looks like a commit sha. Mirrors Python's `_COMMIT_REF_RE`.
fn looks_like_commit_ref(git_ref: &str) -> bool {
    (7..=40).contains(&git_ref.len()) && git_ref.chars().all(|c| c.is_ascii_hexdigit())
}

/// The script that fetches one commit into a fresh repository.
fn fetch_commit_script(target: &str, url: &str, git_ref: &str) -> String {
    format!(
        "set -e\nrm -rf {target}\nmkdir -p {target}\ncd {target}\ngit init -q .\n\
         git remote add origin {url}\ngit fetch --depth 1 --no-tags origin {git_ref}\n\
         git checkout -q --detach FETCH_HEAD",
        target = shell_quote(target),
        url = shell_quote(url),
        git_ref = shell_quote(git_ref),
    )
}

/// The script that clones one branch or tag.
fn clone_named_ref_script(target: &str, url: &str, git_ref: &str) -> String {
    format!(
        "set -e\nrm -rf {target}\ngit clone --depth 1 --no-tags --branch {git_ref} {url} {target}",
        target = shell_quote(target),
        url = shell_quote(url),
        git_ref = shell_quote(git_ref),
    )
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
    fn builds_both_git_acquisition_scripts() {
        let named = clone_named_ref_script("/tmp/t", "https://h/r.git", "main");
        assert!(named.contains("git clone --depth 1 --no-tags --branch main"));
        let fetched = fetch_commit_script("/tmp/t", "https://h/r.git", "deadbeef");
        assert!(fetched.contains("git fetch --depth 1 --no-tags origin deadbeef"));
        assert!(fetched.contains("checkout -q --detach FETCH_HEAD"));
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
