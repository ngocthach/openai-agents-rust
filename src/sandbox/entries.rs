//! Manifest entries and how they are materialized into a workspace. Mirrors
//! `agents.sandbox.entries`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// What the entry materializes.
    pub kind: EntryKind,
    /// Optional description surfaced in the workspace layout instructions.
    pub description: Option<String>,
    /// Permissions applied to the entry after it is written.
    pub permissions: Permissions,
    /// Group or user ownership applied after the entry is written.
    pub group: Option<Group>,
}

impl Entry {
    fn with_kind(kind: EntryKind) -> Self {
        let directory = matches!(
            kind,
            EntryKind::Dir { .. } | EntryKind::LocalDir { .. } | EntryKind::GitRepo { .. }
        );
        Entry {
            kind,
            description: None,
            permissions: if directory {
                default_dir_permissions()
            } else {
                Permissions::default()
            },
            group: None,
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
            EntryKind::Dir { .. } | EntryKind::LocalDir { .. } | EntryKind::GitRepo { .. }
        )
    }

    /// The entry type name, matching the Python `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self.kind {
            EntryKind::File { .. } => "file",
            EntryKind::Dir { .. } => "dir",
            EntryKind::LocalFile { .. } => "local_file",
            EntryKind::LocalDir { .. } => "local_dir",
            EntryKind::GitRepo { .. } => "git_repo",
        }
    }

    /// Materialize this entry at the absolute sandbox path `dest`.
    ///
    /// Returns the absolute sandbox paths of the files that were written.
    pub async fn apply(
        &self,
        session: &dyn SandboxSession,
        dest: &str,
    ) -> SandboxResult<Vec<String>> {
        Box::pin(self.apply_inner(session, dest.to_string())).await
    }

    async fn apply_inner(
        &self,
        session: &dyn SandboxSession,
        dest: String,
    ) -> SandboxResult<Vec<String>> {
        let mut written = Vec::new();
        match &self.kind {
            EntryKind::File { content } => {
                session.write_file(&dest, content).await?;
                written.push(dest.clone());
            }
            EntryKind::Dir { children } => {
                session.mkdir(&dest, true).await?;
                for (name, child) in children {
                    let child_dest = join_sandbox_path(&dest, name)?;
                    written.extend(Box::pin(child.apply_inner(session, child_dest)).await?);
                }
            }
            EntryKind::LocalFile { src } => {
                let content = read_host_file(src).await?;
                session.write_file(&dest, &content).await?;
                written.push(dest.clone());
            }
            EntryKind::LocalDir { src } => {
                session.mkdir(&dest, true).await?;
                if let Some(src) = src {
                    written.extend(copy_host_dir(session, src, &dest).await?);
                }
            }
            EntryKind::GitRepo {
                host,
                repo,
                git_ref,
                subpath,
            } => {
                clone_git_repo(session, host, repo, git_ref, subpath.as_deref(), &dest).await?;
                written.push(dest.clone());
            }
        }
        self.apply_metadata(session, &dest).await?;
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
    tokio::fs::read(src).await.map_err(|e| {
        SandboxError::new(
            ErrorCode::LocalFileReadError,
            format!("Failed to read host file `{}`: {e}", src.display()),
        )
    })
}

/// Copy a host directory tree into the sandbox, one file at a time.
async fn copy_host_dir(
    session: &dyn SandboxSession,
    src: &Path,
    dest: &str,
) -> SandboxResult<Vec<String>> {
    let mut written = Vec::new();
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
                session.mkdir(&target, true).await?;
                pending.push((entry.path(), target));
            } else if file_type.is_file() {
                let content = read_host_file(&entry.path()).await?;
                session.write_file(&target, &content).await?;
                written.push(target);
            }
            // Symlinks are skipped: following them would copy host state from outside `src`.
        }
    }
    Ok(written)
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
    let script = format!(
        "set -e\nrm -rf {target}\nmkdir -p {target}\ncd {target}\ngit init -q .\n\
         git remote add origin {url}\ngit fetch --depth 1 origin {git_ref}\n\
         git checkout -q FETCH_HEAD",
        target = shell_quote(&clone_target),
        url = shell_quote(&url),
        git_ref = shell_quote(git_ref),
    );
    let result = session.exec(ExecRequest::shell(script)).await?;
    if !result.ok() {
        return Err(SandboxError::new(
            ErrorCode::GitCloneError,
            format!(
                "Failed to clone `{url}` at `{git_ref}`: {}",
                result.stderr_text().trim()
            ),
        ));
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
            return Err(SandboxError::new(
                ErrorCode::GitSubpathError,
                format!(
                    "Subpath `{subpath}` is not present in `{url}` at `{git_ref}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
    }
    Ok(())
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
