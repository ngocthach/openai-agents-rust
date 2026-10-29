//! The sandbox manifest: what a fresh workspace contains. Mirrors
//! `agents.sandbox.manifest` and `agents.sandbox.manifest_render`.

use std::collections::{BTreeMap, BTreeSet};

use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mounts::{HostPathGrant, Mount, MountCredentialAuthority};
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{Group, User};
use crate::sandbox::workspace_paths::resolve_workspace_path;

/// Commands the model may run on remote mounts by default. Mirrors
/// `DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST`.
pub const DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST: &[&str] = &[
    "ls", "find", "stat", "cat", "less", "head", "tail", "du", "grep", "rg", "wc", "sort", "cut",
    "cp", "tee", "echo", "mkdir", "rm",
];

/// The default workspace root. Backends that own their workspace (such as Unix-local) replace
/// it with a real directory when they create a session.
pub const DEFAULT_MANIFEST_ROOT: &str = "/workspace";

/// Maximum size of the rendered workspace layout included in instructions.
pub const MAX_MANIFEST_DESCRIPTION_CHARS: usize = 5000;

/// Depth of the rendered workspace layout.
const MANIFEST_DESCRIPTION_DEPTH: usize = 3;

/// The files, directories, mounts and accounts a fresh sandbox session starts from.
#[derive(Debug, Clone)]
pub struct Manifest {
    version: u8,
    root: String,
    entries: BTreeMap<String, Entry>,
    environment: BTreeMap<String, String>,
    users: Vec<User>,
    groups: Vec<Group>,
    remote_mount_command_allowlist: Vec<String>,
    credential_exposure_acknowledged: BTreeSet<String>,
    broad_credential_exposure_acknowledged: BTreeSet<String>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            version: 1,
            root: DEFAULT_MANIFEST_ROOT.to_string(),
            entries: BTreeMap::new(),
            environment: BTreeMap::new(),
            users: Vec::new(),
            groups: Vec::new(),
            remote_mount_command_allowlist: DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST
                .iter()
                .map(|c| (*c).to_string())
                .collect(),
            credential_exposure_acknowledged: BTreeSet::new(),
            broad_credential_exposure_acknowledged: BTreeSet::new(),
        }
    }
}

impl Manifest {
    /// An empty manifest rooted at [`DEFAULT_MANIFEST_ROOT`].
    pub fn new() -> Self {
        Manifest::default()
    }

    /// The manifest schema version (always 1).
    pub fn version(&self) -> u8 {
        self.version
    }

    /// The absolute workspace root inside the sandbox.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// The top-level entries, keyed by their workspace-relative path.
    pub fn entries(&self) -> &BTreeMap<String, Entry> {
        &self.entries
    }

    /// Environment variables exported for every command in the session.
    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.environment
    }

    /// Users provisioned before entries are materialized.
    pub fn users(&self) -> &[User] {
        &self.users
    }

    /// Groups provisioned before entries are materialized.
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// Set the workspace root.
    pub fn with_root(mut self, root: impl Into<String>) -> Self {
        let root = root.into();
        let trimmed = root.trim_end_matches('/');
        self.root = if trimmed.is_empty() {
            "/".to_string()
        } else {
            trimmed.to_string()
        };
        self
    }

    /// Add or replace a top-level entry.
    pub fn with_entry(mut self, path: impl Into<String>, entry: Entry) -> Self {
        self.entries.insert(path.into(), entry);
        self
    }

    /// Export an environment variable to every command in the session.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }

    /// Provision a user before entries are materialized.
    pub fn with_user(mut self, user: impl Into<User>) -> Self {
        self.users.push(user.into());
        self
    }

    /// Provision a group before entries are materialized.
    pub fn with_group(mut self, group: Group) -> Self {
        self.groups.push(group);
        self
    }

    /// Commands the model may run on remote mounts.
    pub fn remote_mount_command_allowlist(&self) -> &[String] {
        &self.remote_mount_command_allowlist
    }

    /// Replace the remote mount command allowlist.
    pub fn with_remote_mount_command_allowlist<I, S>(mut self, commands: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.remote_mount_command_allowlist = commands.into_iter().map(Into::into).collect();
        self
    }

    /// Acknowledge mount-scoped credential exposure for exact in-container mount paths.
    ///
    /// An in-container mount whose pattern writes a credential inside the sandbox cannot be
    /// activated until its exact mount path is acknowledged here. This is trusted
    /// application-side policy: it is runtime-only and never serialized into session state.
    pub fn with_in_container_mount_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.credential_exposure_acknowledged
            .extend(mount_paths.into_iter().map(Into::into));
        self
    }

    /// Acknowledge broad credential exposure (managed or workload identity, external credential
    /// files) for exact in-container mount paths.
    pub fn with_in_container_mount_broad_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.broad_credential_exposure_acknowledged
            .extend(mount_paths.into_iter().map(Into::into));
        self
    }

    /// Visit every entry, including the children of directory entries, with its absolute
    /// workspace path.
    pub fn iter_entries(&self) -> Vec<(String, &Entry)> {
        fn walk<'a>(out: &mut Vec<(String, &'a Entry)>, dest: String, entry: &'a Entry) {
            out.push((dest.clone(), entry));
            if let EntryKind::Dir { children } = &entry.kind {
                for (name, child) in children {
                    walk(
                        out,
                        format!("{}/{}", dest.trim_end_matches('/'), name.trim_matches('/')),
                        child,
                    );
                }
            }
        }
        let mut out = Vec::new();
        if let Ok(validated) = self.validated_entries() {
            for (_, dest, entry) in validated {
                walk(&mut out, dest, entry);
            }
        }
        out
    }

    /// Every mount in the manifest with its resolved mount path, deepest first.
    pub fn mount_targets(&self) -> Vec<(&Mount, String)> {
        let mut mounts: Vec<(&Mount, String)> = Vec::new();
        for (dest, entry) in self.iter_entries() {
            let Some(mount) = entry.as_mount() else {
                continue;
            };
            let Ok(path) = mount.resolve_mount_path(&self.root, &dest) else {
                continue;
            };
            mounts.push((mount, path));
        }
        mounts.sort_by_key(|(_, path)| std::cmp::Reverse(path.matches('/').count()));
        mounts
    }

    /// Host paths the SDK may use directly because a mount exposes them.
    pub fn host_path_grants(&self) -> Vec<HostPathGrant> {
        self.mount_targets()
            .into_iter()
            .filter_map(|(mount, _)| mount.strategy.host_path_grant(mount))
            .collect()
    }

    /// Absolute workspace paths that snapshots must exclude and resume must re-apply.
    pub fn ephemeral_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self
            .iter_entries()
            .into_iter()
            .filter(|(_, entry)| entry.ephemeral)
            .map(|(dest, _)| dest)
            .collect();
        paths.extend(self.mount_targets().into_iter().map(|(_, path)| path));
        paths.sort_unstable();
        paths.dedup();
        paths
    }

    /// Check that activating `mount` at `mount_path` respects the credential-exposure policy.
    ///
    /// Returns [`ErrorCode::MountConfigInvalid`] when the mount would write a credential into
    /// the sandbox at a path the application has not acknowledged. The error text never
    /// includes the mount configuration.
    pub fn authorize_mount_activation(&self, mount: &Mount, mount_path: &str) -> SandboxResult<()> {
        let authority = mount.strategy.credential_authority();
        let acknowledged = match authority {
            MountCredentialAuthority::None => return Ok(()),
            MountCredentialAuthority::MountScoped => {
                self.credential_exposure_acknowledged.contains(mount_path)
                    || self
                        .broad_credential_exposure_acknowledged
                        .contains(mount_path)
            }
            MountCredentialAuthority::Broad => self
                .broad_credential_exposure_acknowledged
                .contains(mount_path),
        };
        if acknowledged {
            return Ok(());
        }
        Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "mount at `{mount_path}` exposes `{}` credential authority inside the sandbox; \
                 acknowledge the exact mount path on the Manifest before activating it",
                authority.as_str()
            ),
        ))
    }

    /// Validate every entry path against the workspace root.
    pub fn validated_entries(&self) -> SandboxResult<Vec<(String, String, &Entry)>> {
        let mut validated = Vec::with_capacity(self.entries.len());
        for (path, entry) in &self.entries {
            let resolved = resolve_workspace_path(&self.root, path)?;
            validated.push((path.clone(), resolved, entry));
        }
        Ok(validated)
    }

    /// Materialize the manifest into a started session.
    ///
    /// Accounts are provisioned first (best effort, as in Python: a sandbox without `useradd`
    /// simply keeps running), then the workspace root, then every entry.
    pub async fn apply(&self, session: &dyn SandboxSession) -> SandboxResult<Vec<String>> {
        self.apply_entries(session, false).await
    }

    /// Re-apply only the ephemeral entries, as a resumed session does after restoring a snapshot.
    pub async fn apply_ephemeral(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<Vec<String>> {
        self.apply_entries(session, true).await
    }

    async fn apply_entries(
        &self,
        session: &dyn SandboxSession,
        only_ephemeral: bool,
    ) -> SandboxResult<Vec<String>> {
        let validated = self.validated_entries()?;
        session.mkdir(&self.root, true).await?;
        if !only_ephemeral {
            self.provision_accounts(session).await;
        }
        let mut written = Vec::new();
        for (_, dest, entry) in validated {
            if only_ephemeral && !entry.ephemeral {
                continue;
            }
            if entry.as_mount().is_none() {
                if entry.is_dir() {
                    session.mkdir(&dest, true).await?;
                } else if let Some(parent) = parent_path(&dest) {
                    session.mkdir(&parent, true).await?;
                }
            }
            written.extend(entry.apply(session, &dest).await?);
        }
        Ok(written)
    }

    /// Detach every mount, deepest first, before the session stops.
    pub async fn unmount_all(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        for (mount, path) in self.mount_targets() {
            if session.attaches_mounts_at_create()
                && mount.strategy.docker_create_args(mount, &path).is_some()
            {
                continue;
            }
            mount.strategy.deactivate(mount, session, &path).await?;
        }
        Ok(())
    }

    async fn provision_accounts(&self, session: &dyn SandboxSession) {
        for group in &self.groups {
            let _ = session
                .exec(ExecRequest::argv(["groupadd", "-f", &group.name]))
                .await;
        }
        let mut user_names: BTreeSet<&str> = self.users.iter().map(|u| u.name.as_str()).collect();
        for group in &self.groups {
            user_names.extend(group.users.iter().map(|u| u.name.as_str()));
        }
        for name in user_names {
            let _ = session
                .exec(ExecRequest::argv(["useradd", "-m", name]))
                .await;
        }
        for group in &self.groups {
            for user in &group.users {
                let _ = session
                    .exec(ExecRequest::argv([
                        "usermod",
                        "-aG",
                        &group.name,
                        &user.name,
                    ]))
                    .await;
            }
        }
    }

    /// Render the workspace layout for the model, truncated to
    /// [`MAX_MANIFEST_DESCRIPTION_CHARS`].
    pub fn describe(&self) -> String {
        self.render(
            MANIFEST_DESCRIPTION_DEPTH,
            Some(MAX_MANIFEST_DESCRIPTION_CHARS),
        )
    }

    /// Render the workspace layout with an explicit depth and character budget.
    pub fn render(&self, depth: usize, max_chars: Option<usize>) -> String {
        let mut out = String::new();
        out.push_str(self.root.trim_end_matches('/'));
        if out.is_empty() {
            out.push('/');
        }
        out.push_str("/\n");
        for (path, entry) in &self.entries {
            render_entry(&mut out, path, entry, 1, depth);
        }
        match max_chars {
            Some(max) if out.chars().count() > max => {
                let marker = "\n... (truncated)\nThe filesystem layout above was truncated. Use \
                              `ls` to explore specific directories before relying on omitted \
                              paths.\n";
                let keep = max.saturating_sub(marker.chars().count());
                let mut truncated: String = out.chars().take(keep).collect();
                truncated.push_str(marker);
                truncated
            }
            _ => out,
        }
    }
}

fn render_entry(out: &mut String, name: &str, entry: &Entry, level: usize, depth: usize) {
    if level > depth {
        return;
    }
    let indent = "  ".repeat(level);
    let suffix = if entry.is_dir() { "/" } else { "" };
    out.push_str(&indent);
    out.push_str(name.trim_matches('/'));
    out.push_str(suffix);
    if entry.as_mount().is_some() {
        out.push_str(" (mount)");
    }
    if let Some(description) = &entry.description {
        out.push_str(" - ");
        out.push_str(description);
    }
    out.push('\n');
    if let EntryKind::Dir { children } = &entry.kind {
        for (child_name, child) in children {
            render_entry(out, child_name, child, level + 1, depth);
        }
    }
}

/// The parent of an absolute sandbox path, when it has one.
fn parent_path(path: &str) -> Option<String> {
    let trimmed = path.trim_end_matches('/');
    let index = trimmed.rfind('/')?;
    if index == 0 {
        return Some("/".to_string());
    }
    Some(trimmed[..index].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_the_workspace_layout() {
        let manifest = Manifest::new()
            .with_entry(
                "repo",
                Entry::dir()
                    .with_description("staged repository")
                    .with_child("task.md", Entry::text_file("fix it")),
            )
            .with_entry("notes.txt", Entry::text_file("hi"));
        let rendered = manifest.describe();
        assert!(rendered.starts_with("/workspace/\n"));
        assert!(rendered.contains("  repo/ - staged repository\n"));
        assert!(rendered.contains("    task.md\n"));
        assert!(rendered.contains("  notes.txt\n"));
    }

    #[test]
    fn validates_entry_paths() {
        let manifest = Manifest::new().with_entry("../escape", Entry::text_file("x"));
        assert!(manifest.validated_entries().is_err());
        let manifest = Manifest::new().with_entry("ok/file.txt", Entry::text_file("x"));
        let validated = manifest.validated_entries().unwrap();
        assert_eq!(validated[0].1, "/workspace/ok/file.txt");
    }

    #[test]
    fn computes_parent_paths() {
        assert_eq!(parent_path("/workspace/a/b"), Some("/workspace/a".into()));
        assert_eq!(parent_path("/a"), Some("/".into()));
        assert_eq!(parent_path("/"), None);
    }

    #[test]
    fn truncates_long_layouts() {
        let mut manifest = Manifest::new();
        for i in 0..400 {
            manifest = manifest.with_entry(
                format!("file-{i:04}.txt"),
                Entry::text_file("x").with_description("a".repeat(40)),
            );
        }
        let rendered = manifest.describe();
        assert!(rendered.chars().count() <= MAX_MANIFEST_DESCRIPTION_CHARS);
        assert!(rendered.contains("(truncated)"));
    }
}
