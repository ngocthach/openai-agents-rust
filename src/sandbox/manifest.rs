//! The sandbox manifest: what a fresh workspace contains. Mirrors
//! `agents.sandbox.manifest` and `agents.sandbox.manifest_render`.

use std::collections::{BTreeMap, BTreeSet};

use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::environment::{EnvEntry, Environment};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::materialization::MaterializationResult;
use crate::sandbox::mounts::{HostPathGrant, Mount, MountCredentialAuthority};
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{Group, User};
use crate::sandbox::workspace_paths::resolve_workspace_path;
use crate::sandbox::workspace_paths::{SandboxPathGrant, to_posix as to_posix_path};

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

/// The default number of manifest entries materialized at once.
pub const DEFAULT_MAX_MANIFEST_ENTRY_CONCURRENCY: usize = 4;

/// The default number of files copied at once for one `local_dir` entry.
pub const DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY: usize = 4;

/// How much materialization work runs at once. Mirrors Python's `SandboxConcurrencyLimits`.
///
/// Copying a large directory one file at a time is slow over a hosted backend, where each file
/// is a round trip; copying all of them at once exhausts file handles and connections. These
/// bound the middle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxConcurrencyLimits {
    /// Manifest entries materialized at once; `None` disables the limit.
    pub manifest_entries: Option<usize>,
    /// Files copied at once for one `local_dir` entry; `None` disables the limit.
    pub local_dir_files: Option<usize>,
}

impl Default for SandboxConcurrencyLimits {
    fn default() -> Self {
        SandboxConcurrencyLimits {
            manifest_entries: Some(DEFAULT_MAX_MANIFEST_ENTRY_CONCURRENCY),
            local_dir_files: Some(DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY),
        }
    }
}

impl SandboxConcurrencyLimits {
    /// The SDK defaults.
    pub fn new() -> Self {
        SandboxConcurrencyLimits::default()
    }

    /// Copy at most `files` files at once for one `local_dir` entry.
    pub fn with_local_dir_files(mut self, files: Option<usize>) -> Self {
        self.local_dir_files = files;
        self
    }

    /// Materialize at most `entries` manifest entries at once.
    pub fn with_manifest_entries(mut self, entries: Option<usize>) -> Self {
        self.manifest_entries = entries;
        self
    }

    /// Reject a limit that would stop work happening at all.
    pub fn validate(&self) -> SandboxResult<()> {
        for (name, value) in [
            ("manifest_entries", self.manifest_entries),
            ("local_dir_files", self.local_dir_files),
        ] {
            if value == Some(0) {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!("concurrency_limits.{name} must be at least 1"),
                ));
            }
        }
        Ok(())
    }

    /// The effective file-copy width, with `None` meaning "no limit".
    pub fn local_dir_file_width(&self) -> usize {
        self.local_dir_files.unwrap_or(usize::MAX).max(1)
    }

    /// The effective manifest-entry width, with `None` meaning "no limit".
    pub fn manifest_entry_width(&self) -> usize {
        self.manifest_entries.unwrap_or(usize::MAX).max(1)
    }
}

/// The files, directories, mounts and accounts a fresh sandbox session starts from.
#[derive(Debug, Clone)]
pub struct Manifest {
    version: u8,
    root: String,
    entries: BTreeMap<String, Entry>,
    environment: BTreeMap<String, String>,
    resolved_environment: Environment,
    concurrency_limits: SandboxConcurrencyLimits,
    extra_path_grants: Vec<SandboxPathGrant>,
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
            resolved_environment: Environment::new(),
            concurrency_limits: SandboxConcurrencyLimits::new(),
            extra_path_grants: Vec::new(),
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

    /// Variables whose values are fetched when the session starts.
    pub fn resolved_environment(&self) -> &Environment {
        &self.resolved_environment
    }

    /// Export a variable whose value is fetched when the session starts.
    ///
    /// Use this for a value that should not be written into the manifest, such as a secret read
    /// from a credential store. See [`EnvValue`](crate::sandbox::EnvValue).
    pub fn with_env_entry(mut self, key: impl Into<String>, entry: EnvEntry) -> Self {
        self.resolved_environment = self.resolved_environment.with_entry(key, entry);
        self
    }

    /// Every environment variable the sandbox exports, with the start-up lookups performed.
    ///
    /// A looked-up value wins over a literal of the same name, because the lookup is the more
    /// specific statement of intent.
    pub async fn resolve_environment(&self) -> SandboxResult<BTreeMap<String, String>> {
        let mut environment = self.environment.clone();
        environment.extend(self.resolved_environment.resolve().await?);
        Ok(environment)
    }

    /// How much materialization work this manifest runs at once.
    pub fn concurrency_limits(&self) -> SandboxConcurrencyLimits {
        self.concurrency_limits
    }

    /// Bound how much materialization work runs at once.
    pub fn with_concurrency_limits(
        mut self,
        limits: SandboxConcurrencyLimits,
    ) -> SandboxResult<Self> {
        limits.validate()?;
        self.concurrency_limits = limits;
        Ok(self)
    }

    /// The extra absolute paths this manifest grants the sandbox outside its workspace.
    pub fn extra_path_grants(&self) -> &[SandboxPathGrant] {
        &self.extra_path_grants
    }

    /// Grant the sandbox access to an absolute path outside its workspace.
    ///
    /// This is trusted application configuration, like the credential-exposure
    /// acknowledgements: it widens what the sandbox can reach, so it is never taken from
    /// untrusted manifest input.
    pub fn with_extra_path_grant(mut self, grant: SandboxPathGrant) -> Self {
        self.extra_path_grants.push(grant);
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

    /// Host paths the SDK may use directly.
    ///
    /// These come from two places: a mount that exposes a host directory, and an
    /// [`extra_path_grant`](Self::with_extra_path_grant) the application declared. Both widen
    /// what the SDK may touch outside the workspace, and both carry their own `read_only` flag,
    /// so the backends enforce one rule rather than two.
    pub fn host_path_grants(&self) -> Vec<HostPathGrant> {
        let mut grants: Vec<HostPathGrant> = self
            .mount_targets()
            .into_iter()
            .filter_map(|(mount, _)| mount.strategy.host_path_grant(mount))
            .collect();
        grants.extend(self.extra_path_grants.iter().map(|grant| HostPathGrant {
            path: std::path::PathBuf::from(grant.host_source()),
            read_only: grant.read_only,
        }));
        grants
    }

    /// Whether `path` falls under a read-only grant, which refuses writes through it.
    pub fn path_grant_is_read_only(&self, path: &str) -> bool {
        let normalized = crate::sandbox::workspace_paths::normalize_segments(&to_posix_path(path));
        self.extra_path_grants.iter().any(|grant| {
            grant.read_only && crate::sandbox::workspace_paths::is_within(&grant.path, &normalized)
        })
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

    /// The workspace-relative paths of the entries marked ephemeral.
    ///
    /// Mirrors Python's `ephemeral_entry_paths`. These are the manifest keys, not resolved
    /// paths, which is the form a caller comparing two manifests works in.
    pub fn ephemeral_entry_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.ephemeral)
            .map(|(path, _)| path.clone())
            .collect();
        paths.sort_unstable();
        paths.dedup();
        paths
    }

    /// The mounts declared by ephemeral entries, with the paths they attach at.
    pub fn ephemeral_mount_targets(&self) -> Vec<(&Mount, String)> {
        let mut mounts: Vec<(&Mount, String)> = Vec::new();
        for (dest, entry) in self.iter_entries() {
            if !entry.ephemeral {
                continue;
            }
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

    /// The workspace-relative paths a persistence pass has to skip.
    ///
    /// Mirrors Python's `ephemeral_persistence_paths`: the ephemeral entries plus wherever their
    /// mounts actually landed, which can differ from the entry's own path when a mount sets an
    /// explicit `mount_path`.
    pub fn ephemeral_persistence_paths(&self) -> Vec<String> {
        let root = self.root.trim_end_matches('/');
        let mut paths = self.ephemeral_entry_paths();
        for (_, path) in self.ephemeral_mount_targets() {
            if let Some(relative) = path.strip_prefix(root) {
                let relative = relative.trim_start_matches('/');
                if !relative.is_empty() {
                    paths.push(relative.to_string());
                }
            }
        }
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
    pub async fn apply(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, false).await
    }

    /// Re-apply only the ephemeral entries, as a resumed session does after restoring a snapshot.
    pub async fn apply_ephemeral(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, true).await
    }

    async fn apply_entries(
        &self,
        session: &dyn SandboxSession,
        only_ephemeral: bool,
    ) -> SandboxResult<MaterializationResult> {
        let validated = self.validated_entries()?;
        if !self.resolved_environment.is_empty() && !session.resolves_environment() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!(
                    "the `{}` sandbox backend does not perform start-up environment lookups, so \
                     the variables declared with `with_env_entry` would be missing inside the \
                     sandbox",
                    session.backend_id()
                ),
            ));
        }
        session.mkdir(&self.root, true).await?;
        if !only_ephemeral {
            self.provision_accounts(session).await;
        }
        let mut written = MaterializationResult::new();
        // Entries materialize concurrently, in batches. Two entries may only share a batch when
        // neither is a mount and neither path contains the other: a mount has to settle before
        // anything can be written next to it, and a parent directory has to exist before its
        // child is created. Everything inside a batch is independent, so a batch is bounded only
        // by `concurrency_limits.manifest_entries`.
        let width = self.concurrency_limits.manifest_entry_width();
        let mut batch: Vec<(String, Entry)> = Vec::new();
        for (_, dest, entry) in validated {
            if only_ephemeral && !entry.ephemeral {
                continue;
            }
            // The manifest owns the limits, so an entry built before they were set still
            // honours them.
            let entry = entry
                .clone()
                .with_copy_concurrency(self.concurrency_limits.local_dir_file_width());
            let is_mount = entry.as_mount().is_some();
            let overlaps = batch.iter().any(|(queued, _)| paths_overlap(queued, &dest));
            if is_mount || overlaps {
                written.extend(apply_entry_batch(session, &mut batch, width).await?);
            }
            if is_mount {
                written.extend(entry.apply(session, &dest).await?);
                continue;
            }
            batch.push((dest, entry));
        }
        written.extend(apply_entry_batch(session, &mut batch, width).await?);
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

/// Whether one path is the other, or contains it.
fn paths_overlap(left: &str, right: &str) -> bool {
    let left = left.trim_end_matches('/');
    let right = right.trim_end_matches('/');
    left == right
        || right.starts_with(&format!("{left}/"))
        || left.starts_with(&format!("{right}/"))
}

/// Materialize a batch of independent entries concurrently and drain it.
///
/// The results are collected in the batch's own order, so a manifest's materialization result is
/// the same whatever order the backend happened to finish the entries in.
async fn apply_entry_batch(
    session: &dyn SandboxSession,
    batch: &mut Vec<(String, Entry)>,
    width: usize,
) -> SandboxResult<MaterializationResult> {
    let mut written = MaterializationResult::new();
    if batch.is_empty() {
        return Ok(written);
    }
    let queued = std::mem::take(batch);
    for chunk in queued.chunks(width.max(1)) {
        let results = futures::future::join_all(chunk.iter().map(|(dest, entry)| async move {
            if entry.is_dir() {
                session.mkdir(dest, true).await?;
            } else if let Some(parent) = parent_path(dest) {
                session.mkdir(&parent, true).await?;
            }
            entry.apply(session, dest).await
        }))
        .await;
        for result in results {
            written.extend(result?);
        }
    }
    Ok(written)
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

    #[tokio::test]
    async fn copies_a_local_directory_under_its_concurrency_limit() {
        use crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
        use crate::sandbox::session::SandboxClient;

        let host = tempfile::tempdir().unwrap();
        for index in 0..9 {
            tokio::fs::write(
                host.path().join(format!("f{index}.txt")),
                format!("{index}"),
            )
            .await
            .unwrap();
        }

        let manifest = Manifest::new()
            .with_entry("data", Entry::local_dir(host.path()))
            .with_concurrency_limits(SandboxConcurrencyLimits::new().with_local_dir_files(Some(2)))
            .unwrap();
        let client = UnixLocalSandboxClient::new();
        let session = client.create(Some(manifest)).await.unwrap();
        session.start().await.unwrap();

        // Every file arrives, with the checksum of what was written.
        let listing = session.ls("data").await.unwrap();
        assert_eq!(listing.len(), 9, "{listing:?}");
        client.delete(session).await.unwrap();
    }

    #[test]
    fn rejects_a_concurrency_limit_that_stops_work() {
        assert!(
            SandboxConcurrencyLimits::new()
                .with_local_dir_files(Some(0))
                .validate()
                .is_err()
        );
        assert!(
            SandboxConcurrencyLimits::new()
                .with_manifest_entries(Some(0))
                .validate()
                .is_err()
        );
        // `None` disables the limit rather than meaning zero.
        assert!(
            SandboxConcurrencyLimits::new()
                .with_local_dir_files(None)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn separates_ephemeral_entries_from_their_mount_targets() {
        use crate::sandbox::mounts::BindMountStrategy;

        let manifest = Manifest::new()
            .with_root("/workspace")
            .with_entry("cache", Entry::dir().with_ephemeral(true))
            .with_entry("keep", Entry::dir())
            .with_entry(
                "bucket",
                Entry::mount(BindMountStrategy::mount("/host/data").with_mount_path("mnt/data")),
            );

        assert_eq!(manifest.ephemeral_entry_paths(), vec!["bucket", "cache"]);
        // The mount landed somewhere other than its entry path, so persistence has to skip both.
        let persistence = manifest.ephemeral_persistence_paths();
        assert!(
            persistence.contains(&"cache".to_string()),
            "{persistence:?}"
        );
        assert!(
            persistence.contains(&"mnt/data".to_string()),
            "{persistence:?}"
        );
        assert!(
            !persistence.contains(&"keep".to_string()),
            "{persistence:?}"
        );
        assert_eq!(manifest.ephemeral_mount_targets().len(), 1);
    }

    #[test]
    fn records_extra_path_grants() {
        let manifest = Manifest::new().with_extra_path_grant(
            SandboxPathGrant::new("/srv/corpus")
                .unwrap()
                .read_only(true),
        );
        assert_eq!(manifest.extra_path_grants().len(), 1);
        assert!(manifest.extra_path_grants()[0].read_only);
    }

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
