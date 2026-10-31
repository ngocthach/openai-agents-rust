//! The sandbox manifest: what a fresh workspace contains. Mirrors
//! `agents.sandbox.manifest` and `agents.sandbox.manifest_render`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::environment::{EnvEntry, Environment};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::materialization::{MaterializationResult, gather_in_order};
use crate::sandbox::mount_entries::ConfiguredMountStrategy;
use crate::sandbox::mounts::{HostPathGrant, Mount, MountCredentialAuthority};
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{Group, User};
use crate::sandbox::workspace_paths::{SandboxPathGrant, to_posix as to_posix_path};
use crate::sandbox::workspace_paths::{is_windows_absolute, normpath, resolve_workspace_path};

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
const MANIFEST_DESCRIPTION_DEPTH: usize = 1;

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
    /// The order top-level entries were added in, which a Python dict keeps and a `BTreeMap`
    /// cannot. A replaced key keeps its original position.
    entry_order: Vec<String>,
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
            entry_order: Vec::new(),
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
        let path = path.into();
        if self.entries.insert(path.clone(), entry).is_none() {
            self.entry_order.push(path);
        }
        self
    }

    /// The top-level entries in the order they were added, as Python's dict iterates them.
    ///
    /// [`entries`](Self::entries) is the key-ordered view for lookup. Materialization, mount
    /// ordering and re-applying ephemeral entries follow this order, so a manifest that lists a
    /// directory before the mount inside it is applied the way it was written. The rendered
    /// layout and [`manifest_digest`](crate::sandbox::manifest_digest) stay key-ordered: Python
    /// sorts names when it renders, and a digest must not depend on how a manifest was built.
    pub fn entries_in_order(&self) -> impl Iterator<Item = (&String, &Entry)> {
        self.entry_order
            .iter()
            .filter_map(|path| self.entries.get_key_value(path))
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
    ///
    /// A path Python would reject (empty, the workspace root, a wildcard, a `..` segment or a
    /// backslash) acknowledges nothing here, so a mistake fails closed. Use
    /// [`try_with_in_container_mount_credential_exposure_acknowledged`](Self::try_with_in_container_mount_credential_exposure_acknowledged)
    /// to be told about it.
    pub fn with_in_container_mount_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for path in mount_paths {
            if let Ok(Some(key)) = self.credential_policy_key(&path.into(), true) {
                self.credential_exposure_acknowledged.insert(key);
            }
        }
        self
    }

    /// Acknowledge broad credential exposure (managed or workload identity, external credential
    /// files) for exact in-container mount paths. Invalid paths acknowledge nothing; see
    /// [`with_in_container_mount_credential_exposure_acknowledged`](Self::with_in_container_mount_credential_exposure_acknowledged).
    pub fn with_in_container_mount_broad_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for path in mount_paths {
            if let Ok(Some(key)) = self.credential_policy_key(&path.into(), true) {
                self.broad_credential_exposure_acknowledged.insert(key);
            }
        }
        self
    }

    /// Like [`with_in_container_mount_credential_exposure_acknowledged`](Self::with_in_container_mount_credential_exposure_acknowledged),
    /// but rejects what Python rejects: no path at all, an empty path, the workspace root, a
    /// backslash, wildcard syntax (`*`, `?`, `[`, `]`) and `..` segments.
    pub fn try_with_in_container_mount_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> SandboxResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let keys = self.credential_policy_keys(mount_paths)?;
        self.credential_exposure_acknowledged.extend(keys);
        Ok(self)
    }

    /// The broad-authority counterpart of
    /// [`try_with_in_container_mount_credential_exposure_acknowledged`](Self::try_with_in_container_mount_credential_exposure_acknowledged).
    pub fn try_with_in_container_mount_broad_credential_exposure_acknowledged<I, S>(
        mut self,
        mount_paths: I,
    ) -> SandboxResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let keys = self.credential_policy_keys(mount_paths)?;
        self.broad_credential_exposure_acknowledged.extend(keys);
        Ok(self)
    }

    fn credential_policy_keys<I, S>(&self, mount_paths: I) -> SandboxResult<Vec<String>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut keys = Vec::new();
        for path in mount_paths {
            if let Some(key) = self.credential_policy_key(&path.into(), true)? {
                keys.push(key);
            }
        }
        if keys.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "at least one in-container mount path is required",
            ));
        }
        Ok(keys)
    }

    /// The policy key for an acknowledged path, mirroring Python's
    /// `_mount_credential_exposure_policy_key`. `Ok(None)` is the root (only reachable when
    /// `reject_root` is false, which is the lookup side).
    fn credential_policy_key(
        &self,
        text: &str,
        reject_root: bool,
    ) -> SandboxResult<Option<String>> {
        let invalid = |message: &str| SandboxError::new(ErrorCode::MountConfigInvalid, message);
        let root_error = "mount credential exposure path must identify a non-root path";
        if text.is_empty() {
            return if reject_root {
                Err(invalid(root_error))
            } else {
                Ok(None)
            };
        }
        if text.contains('\\') {
            return Err(invalid(
                "mount credential exposure paths must use '/' separators",
            ));
        }
        if reject_root && text.contains(['*', '?', '[', ']']) {
            return Err(invalid(
                "mount credential exposure paths must not contain wildcard syntax",
            ));
        }
        if reject_root && text.split('/').any(|part| part == "..") {
            return Err(invalid(
                "mount credential exposure paths must not contain parent segments",
            ));
        }
        let normalized = normpath(text);
        if text.starts_with('/') {
            let root = self.normalized_root();
            if normalized == "/" || normalized == root {
                return if reject_root {
                    Err(invalid(root_error))
                } else {
                    Ok(None)
                };
            }
            return Ok(Some(format!("absolute:{normalized}")));
        }
        if normalized == "." || normalized.is_empty() {
            return if reject_root {
                Err(invalid(root_error))
            } else {
                Ok(None)
            };
        }
        if normalized == ".." || normalized.starts_with("../") {
            return Err(invalid("mount credential exposure path escapes the root"));
        }
        Ok(Some(format!("relative:{normalized}")))
    }

    fn normalized_root(&self) -> String {
        let root = normpath(&self.root);
        if root == "/" {
            root
        } else {
            root.trim_end_matches('/').to_string()
        }
    }

    /// Whether `mount_path` is covered by an acknowledgement in `acknowledged`, in either its
    /// absolute or its root-relative spelling.
    fn acknowledges(&self, acknowledged: &BTreeSet<String>, mount_path: &str) -> bool {
        let Ok(Some(key)) = self.credential_policy_key(mount_path, false) else {
            return false;
        };
        let mut lookup = vec![key.clone()];
        let root = self.normalized_root();
        if let Some(absolute) = key.strip_prefix("absolute:") {
            let prefix = format!("{}/", root.trim_end_matches('/'));
            if let Some(relative) = absolute.strip_prefix(prefix.as_str()) {
                lookup.push(format!("relative:{relative}"));
            }
        } else if let Some(relative) = key.strip_prefix("relative:") {
            lookup.push(format!(
                "absolute:{}/{relative}",
                root.trim_end_matches('/')
            ));
        }
        lookup.iter().any(|key| acknowledged.contains(key))
    }

    /// Visit every entry, including the children of directory entries, with its absolute
    /// workspace path, in key order.
    pub fn iter_entries(&self) -> Vec<(String, &Entry)> {
        self.walk_entries(false)
    }

    /// The same walk in insertion order, for work that has to follow how the manifest was
    /// written.
    fn iter_entries_ordered(&self) -> Vec<(String, &Entry)> {
        self.walk_entries(true)
    }

    fn walk_entries(&self, ordered: bool) -> Vec<(String, &Entry)> {
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
        if let Ok(validated) = self.validate_top_level(ordered) {
            for (_, dest, entry) in validated {
                walk(&mut out, dest, entry);
            }
        }
        out
    }

    /// Every mount in the manifest with its resolved mount path, deepest first.
    ///
    /// A mount whose path cannot be resolved is left out; [`try_mount_targets`](Self::try_mount_targets)
    /// reports it instead.
    pub fn mount_targets(&self) -> Vec<(&Mount, String)> {
        self.collect_mount_targets(false)
            .map(|targets| {
                targets
                    .into_iter()
                    .map(|(_, mount, path)| (mount, path))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Like [`mount_targets`](Self::mount_targets), but a mount path that is absolute in
    /// Windows syntax, or that climbs out of the workspace root, is an error rather than a
    /// silent omission. Mirrors Python's `mount_targets`, which raises
    /// `InvalidManifestPathError`.
    pub fn try_mount_targets(&self) -> SandboxResult<Vec<(&Mount, String)>> {
        Ok(self
            .collect_mount_targets(true)?
            .into_iter()
            .map(|(_, mount, path)| (mount, path))
            .collect())
    }

    fn collect_mount_targets(&self, strict: bool) -> SandboxResult<Vec<(&Entry, &Mount, String)>> {
        let mut mounts: Vec<(&Entry, &Mount, String)> = Vec::new();
        for (dest, entry) in self.iter_entries_ordered() {
            let Some(mount) = entry.as_mount() else {
                continue;
            };
            match self.resolve_mount_target(mount, &dest) {
                Ok(path) => mounts.push((entry, mount, path)),
                Err(error) if strict => return Err(error),
                Err(_) => {}
            }
        }
        mounts.sort_by_key(|(_, _, path)| std::cmp::Reverse(path.matches('/').count()));
        Ok(mounts)
    }

    /// Where `mount` attaches: its explicit `mount_path` (relative ones are taken inside the
    /// workspace root) or the entry's own path.
    ///
    /// A `..` that stays inside the workspace is collapsed (`/workspace/repo/../actual` is
    /// `/workspace/actual`) and one that leaves it is refused, as Python's
    /// `_normalize_in_workspace_path` does. An absolute path outside the root is kept as given.
    fn resolve_mount_target(&self, mount: &Mount, dest: &str) -> SandboxResult<String> {
        let Some(path) = &mount.mount_path else {
            return Ok(dest.to_string());
        };
        if is_windows_absolute(path) {
            return Err(invalid_manifest_path(path));
        }
        let posix = to_posix_path(path);
        let root = self.root.trim_end_matches('/');
        let raw = if posix.starts_with('/') {
            posix
        } else {
            format!("{root}/{posix}")
        };
        let Some(relative) = raw
            .strip_prefix(root)
            .filter(|rest| rest.is_empty() || rest.starts_with('/'))
        else {
            return Ok(raw);
        };
        let mut parts: Vec<&str> = Vec::new();
        for part in relative.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    if parts.pop().is_none() {
                        return Err(manifest_path_error(&raw, "escape_root"));
                    }
                }
                other => parts.push(other),
            }
        }
        if parts.is_empty() {
            Ok(root.to_string())
        } else {
            Ok(format!("{root}/{}", parts.join("/")))
        }
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
        // Nested children count: a `dir` entry holding one ephemeral file has to put that one
        // file on the skip list, not the whole directory and not nothing at all.
        let mut paths: Vec<String> = self
            .iter_relative_entries()
            .into_iter()
            .filter(|(_, entry)| entry.ephemeral)
            .map(|(path, _)| path)
            .collect();
        paths.sort_unstable();
        paths.dedup();
        paths
    }

    /// Visit every entry, children included, with its path relative to the workspace root.
    ///
    /// Mirrors Python's `iter_entries`, which yields manifest-relative paths. The absolute form
    /// is [`iter_entries`](Self::iter_entries).
    pub fn iter_relative_entries(&self) -> Vec<(String, &Entry)> {
        fn walk<'a>(out: &mut Vec<(String, &'a Entry)>, rel: String, entry: &'a Entry) {
            out.push((rel.clone(), entry));
            if let EntryKind::Dir { children } = &entry.kind {
                for (name, child) in children {
                    walk(
                        out,
                        format!("{}/{}", rel.trim_end_matches('/'), name.trim_matches('/')),
                        child,
                    );
                }
            }
        }
        let mut out = Vec::new();
        for (path, entry) in self.entries_in_order() {
            walk(&mut out, path.trim_matches('/').to_string(), entry);
        }
        out
    }

    /// The mounts declared by ephemeral entries, with the paths they attach at.
    pub fn ephemeral_mount_targets(&self) -> Vec<(&Mount, String)> {
        self.collect_mount_targets(false)
            .map(|targets| ephemeral_only(targets))
            .unwrap_or_default()
    }

    /// Like [`ephemeral_mount_targets`](Self::ephemeral_mount_targets), but an unresolvable mount
    /// path is an error. Mirrors Python's `ephemeral_mount_targets`.
    pub fn try_ephemeral_mount_targets(&self) -> SandboxResult<Vec<(&Mount, String)>> {
        Ok(ephemeral_only(self.collect_mount_targets(true)?))
    }

    /// The workspace-relative paths a persistence pass has to skip.
    ///
    /// Mirrors Python's `ephemeral_persistence_paths`: the ephemeral entries plus wherever their
    /// mounts actually landed, which can differ from the entry's own path when a mount sets an
    /// explicit `mount_path`.
    pub fn ephemeral_persistence_paths(&self) -> Vec<String> {
        self.persistence_paths_from(self.ephemeral_mount_targets())
    }

    /// Like [`ephemeral_persistence_paths`](Self::ephemeral_persistence_paths), but an
    /// unresolvable mount path is an error. Mirrors Python's `ephemeral_persistence_paths`.
    pub fn try_ephemeral_persistence_paths(&self) -> SandboxResult<Vec<String>> {
        Ok(self.persistence_paths_from(self.try_ephemeral_mount_targets()?))
    }

    fn persistence_paths_from(&self, targets: Vec<(&Mount, String)>) -> Vec<String> {
        let root = self.root.trim_end_matches('/');
        let mut paths = self.ephemeral_entry_paths();
        for (_, path) in targets {
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

    /// Validate every mount's credential-exposure policy before anything has a side effect.
    ///
    /// Mirrors Python's `validate_manifest_mount_credential_boundaries`, which `stop()` runs
    /// before it touches the snapshot: a manifest that was changed after the session started
    /// could now hold a mount that exposes credentials at a path nobody acknowledged, and finding
    /// that out half-way through detaching mounts would leave the workspace in a state that is
    /// neither the old one nor the new one. The error is
    /// [`ErrorCode::MountConfigInvalid`] and, as with activation, never carries the mount
    /// configuration.
    pub fn validate_mount_credential_boundaries(&self) -> SandboxResult<()> {
        self.validate_mount_credential_files()?;
        for (mount, path) in self.mount_targets() {
            self.authorize_mount_activation(mount, &path)?;
        }
        Ok(())
    }

    /// Refuse a mount whose credential file is one the manifest itself materializes.
    ///
    /// Mirrors Python's `_manifest_materializes_path`: a `service_account_file`, a
    /// `box_config_file` or an rclone `config_file_path` that names a path some entry fills
    /// (a file, a git clone, a host copy, a mount) would be readable by the model, so
    /// credentials have to come from outside the manifest. Plain directories only structure the
    /// tree and are allowed.
    pub fn validate_mount_credential_files(&self) -> SandboxResult<()> {
        let entries = self.iter_entries();
        for (mount, _) in self.mount_targets() {
            let Some(strategy) = mount
                .strategy
                .as_any()
                .and_then(|any| any.downcast_ref::<ConfiguredMountStrategy>())
            else {
                continue;
            };
            for (field, path) in strategy.credential_file_paths() {
                let target = self.absolute_workspace_path(&path);
                let overlaps = entries.iter().any(|(dest, entry)| {
                    let dest = normpath(dest);
                    let covers = target == dest || target.starts_with(&format!("{dest}/"));
                    let structural = match &entry.kind {
                        EntryKind::Dir { .. } => true,
                        EntryKind::LocalDir { src } => src.is_none(),
                        _ => false,
                    };
                    covers && !structural
                });
                if overlaps {
                    return Err(SandboxError::new(
                        ErrorCode::MountConfigInvalid,
                        "credential files stored in the manifest are not supported for cloud \
                         mounts; configure credentials outside the sandbox manifest",
                    )
                    .with_context(
                        "mount_type",
                        strategy
                            .topology_identity()
                            .split('|')
                            .next()
                            .unwrap_or_default()
                            .to_string(),
                    )
                    .with_context("credential_field", field));
                }
            }
        }
        Ok(())
    }

    fn absolute_workspace_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            normpath(path)
        } else {
            normpath(&format!(
                "{}/{path}",
                self.normalized_root().trim_end_matches('/')
            ))
        }
    }

    /// Check that activating `mount` at `mount_path` respects the credential-exposure policy.
    ///
    /// Returns [`ErrorCode::MountConfigInvalid`] when the mount would write a credential into
    /// the sandbox at a path the application has not acknowledged. The error text never
    /// includes the mount configuration.
    pub fn authorize_mount_activation(&self, mount: &Mount, mount_path: &str) -> SandboxResult<()> {
        self.validate_mount_credential_files()?;
        let authority = mount.strategy.credential_authority();
        let acknowledged = match authority {
            MountCredentialAuthority::None => return Ok(()),
            MountCredentialAuthority::MountScoped => {
                self.acknowledges(&self.credential_exposure_acknowledged, mount_path)
                    || self.acknowledges(&self.broad_credential_exposure_acknowledged, mount_path)
            }
            MountCredentialAuthority::Broad => {
                self.acknowledges(&self.broad_credential_exposure_acknowledged, mount_path)
            }
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
        self.validate_top_level(true)
    }

    fn validate_top_level(&self, ordered: bool) -> SandboxResult<Vec<(String, String, &Entry)>> {
        let top_level: Vec<(&String, &Entry)> = if ordered {
            self.entries_in_order().collect()
        } else {
            self.entries.iter().collect()
        };
        // Every entry in the tree is checked, children included: a `dir` whose child is named
        // `../outside.txt` would otherwise pass here and only fail, with a different message,
        // halfway through materialization.
        for (path, entry) in &top_level {
            validate_entry_tree(path, entry)?;
        }
        let mut validated = Vec::with_capacity(top_level.len());
        for (path, entry) in top_level {
            let resolved = resolve_workspace_path(&self.root, path)?;
            validated.push((path.clone(), resolved, entry));
        }
        Ok(validated)
    }

    /// Materialize the manifest into a started session.
    ///
    /// The workspace root is created first, then accounts are provisioned (a failing command
    /// stops the apply), then every entry.
    pub async fn apply(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, false, true, None).await
    }

    /// Materialize the manifest with every `local_file` and `local_dir` source confined to
    /// `base_dir`. See [`Entry::apply_confined`] for what that refuses.
    pub async fn apply_confined(
        &self,
        session: &dyn SandboxSession,
        base_dir: &Path,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, false, true, Some(base_dir))
            .await
    }

    /// Materialize the manifest without creating its users and groups, for a backend that
    /// provisions accounts itself or refuses to. Mirrors Python's `provision_accounts=False`.
    pub async fn apply_without_account_provisioning(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, false, false, None).await
    }

    /// Re-apply only the ephemeral entries, as a resumed session does after restoring a snapshot.
    pub async fn apply_ephemeral(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<MaterializationResult> {
        self.apply_entries(session, true, false, None).await
    }

    async fn apply_entries(
        &self,
        session: &dyn SandboxSession,
        only_ephemeral: bool,
        provision_accounts: bool,
        base_dir: Option<&Path>,
    ) -> SandboxResult<MaterializationResult> {
        let validated = self.validated_entries()?;
        self.validate_mount_credential_files()?;
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
        if provision_accounts && !only_ephemeral {
            self.provision_accounts(session).await?;
        }
        let mut written = MaterializationResult::new();
        // Entries materialize concurrently, in batches. Two entries may only share a batch when
        // neither is a mount and neither path contains the other: a mount has to settle before
        // anything can be written next to it, and a parent directory has to exist before its
        // child is created. Everything inside a batch is independent, so a batch is bounded only
        // by `concurrency_limits.manifest_entries`.
        let width = self.concurrency_limits.manifest_entry_width();
        let queued: Vec<(String, Entry)> = if only_ephemeral {
            // Resume re-applies the ephemeral entries wherever they sit, so the whole tree is
            // walked: a `dir` entry holding one ephemeral file must have that file re-applied,
            // and only that file. An entry that is itself ephemeral is re-applied whole,
            // children and all.
            let mut collected: Vec<(String, &Entry)> = Vec::new();
            for (path, entry) in self.entries_in_order() {
                collect_ephemeral_entries(&mut collected, path.trim_matches('/'), entry);
            }
            let mut resolved = Vec::with_capacity(collected.len());
            for (rel, entry) in collected {
                resolved.push((resolve_workspace_path(&self.root, &rel)?, entry.clone()));
            }
            resolved
        } else {
            validated
                .into_iter()
                .map(|(_, dest, entry)| (dest, entry.clone()))
                .collect()
        };
        let mut batch: Vec<(String, Entry)> = Vec::new();
        for (dest, entry) in queued {
            // The manifest owns the limits, so an entry built before they were set still
            // honours them.
            let entry = entry.with_copy_concurrency(self.concurrency_limits.local_dir_file_width());
            let is_mount = entry.as_mount().is_some();
            let overlaps = batch.iter().any(|(queued, _)| paths_overlap(queued, &dest));
            if is_mount || overlaps {
                written.extend(apply_entry_batch(session, &mut batch, width, base_dir).await?);
            }
            if is_mount {
                written.extend(apply_one(session, &entry, &dest, base_dir).await?);
                continue;
            }
            batch.push((dest, entry));
        }
        written.extend(apply_entry_batch(session, &mut batch, width, base_dir).await?);
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

    /// Create the manifest's groups, users and memberships inside the sandbox.
    ///
    /// A failing command stops provisioning and surfaces: a workspace whose files were meant to
    /// be owned by an account that does not exist would otherwise fail later, far from the cause.
    /// Mirrors Python's `ManifestApplier.provision_accounts`, which runs every command checked.
    ///
    /// [`apply`](Self::apply) already calls this. It is public for the one path that does not go
    /// through `apply`: restoring a snapshot into a fresh sandbox brings the files back but not
    /// the operating-system accounts that own them, so Python's
    /// `_restore_snapshot_and_reapply_ephemeral_on_resume` provisions them again, unless the
    /// backend reports that it kept its system state (a reused Docker container).
    pub async fn provision_accounts(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        let mut commands: Vec<Vec<String>> = Vec::new();
        for group in &self.groups {
            commands.push(vec!["groupadd".to_string(), group.name.clone()]);
        }
        let mut user_names: BTreeSet<&str> = self.users.iter().map(|u| u.name.as_str()).collect();
        for group in &self.groups {
            user_names.extend(group.users.iter().map(|u| u.name.as_str()));
        }
        for name in user_names {
            // Python's exact account shape: a group of the user's own name, no home directory,
            // and no login shell. These accounts exist to own files and run commands, not to be
            // logged into, so giving them a home and a shell widens what a compromised sandbox
            // process can do for no benefit.
            commands.push(
                ["useradd", "-U", "-M", "-s", "/usr/sbin/nologin", name]
                    .map(String::from)
                    .to_vec(),
            );
        }
        for group in &self.groups {
            for user in &group.users {
                commands.push(
                    ["usermod", "-aG", &group.name, &user.name]
                        .map(String::from)
                        .to_vec(),
                );
            }
        }
        for command in commands {
            let result = session.exec(ExecRequest::argv(command.clone())).await?;
            if !result.ok() {
                return Err(SandboxError::exec_nonzero(&command, &result)
                    .with_context("command_str", command.join(" "))
                    .with_context("stdout", result.stdout_text())
                    .with_context("stderr", result.stderr_text()));
            }
        }
        Ok(())
    }

    /// Render the workspace layout for the model. Mirrors Python's `Manifest.describe()`: the
    /// tree of [`render_tree`](Self::render_tree), one level deep, truncated to
    /// [`MAX_MANIFEST_DESCRIPTION_CHARS`].
    pub fn describe(&self) -> String {
        self.render(
            MANIFEST_DESCRIPTION_DEPTH,
            Some(MAX_MANIFEST_DESCRIPTION_CHARS),
        )
    }

    /// Render the workspace layout with an explicit depth and character budget.
    ///
    /// A zero `depth` or `max_chars` renders nothing; [`render_tree`](Self::render_tree) reports
    /// it as an error instead.
    pub fn render(&self, depth: usize, max_chars: Option<usize>) -> String {
        self.render_tree(Some(depth), max_chars).unwrap_or_default()
    }

    /// Render the workspace layout as the tree Python's `describe` produces: box-drawing
    /// connectors, one `# <absolute path>` comment per node (with ` — <description>` when the
    /// entry has one), the columns aligned.
    ///
    /// `depth` limits how many levels are shown (`None` shows all) and `max_chars` truncates the
    /// result with the same marker [`truncate_manifest_description`] uses. A zero `depth` or
    /// `max_chars` is rejected, as Python rejects it. Mirrors `render_manifest_description`.
    pub fn render_tree(
        &self,
        depth: Option<usize>,
        max_chars: Option<usize>,
    ) -> SandboxResult<String> {
        if depth == Some(0) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "depth must be a non-zero positive integer or None",
            ));
        }
        if max_chars == Some(0) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "max_chars must be a non-zero positive integer or None",
            ));
        }
        let root = match self.root.trim_end_matches('/') {
            "" => "/".to_string(),
            trimmed => trimmed.to_string(),
        };
        let root_prefix = root.trim_end_matches('/').to_string();
        let mut tree = TreeNode::default();
        for (path, entry) in &self.entries {
            let parts = path_parts(path);
            let full_path = entry.as_mount().map(|mount| match &mount.mount_path {
                Some(mount_path) if mount_path.starts_with('/') => mount_path.clone(),
                Some(mount_path) => format!("{root_prefix}/{mount_path}"),
                None => format!("{root_prefix}/{}", parts.join("/")),
            });
            insert_entry_tree(&mut tree, &parts, entry, full_path, depth);
        }
        let mut collected: Vec<TreeLine> = Vec::new();
        collect_tree_lines(
            &tree,
            "",
            depth,
            &mut Vec::new(),
            &root_prefix,
            &mut collected,
        );
        let mut lines = vec![root];
        if let Some(max_width) = collected
            .iter()
            .map(|line| line.prefix.chars().count() + line.name.chars().count())
            .max()
        {
            for line in &collected {
                let used = line.prefix.chars().count() + line.name.chars().count();
                let spacer = " ".repeat(max_width - used + 2);
                let comment = match line.description.as_deref() {
                    Some(description) if !description.is_empty() => {
                        format!("# {} \u{2014} {description}", line.full_path)
                    }
                    _ => format!("# {}", line.full_path),
                };
                lines.push(format!("{}{}{spacer}{comment}", line.prefix, line.name));
            }
        }
        let description = format!("{}\n", lines.join("\n"));
        Ok(truncate_manifest_description(&description, max_chars))
    }
}

/// Cut a rendered layout to `max_chars`, ending in a marker that says how much was dropped and
/// tells the model to explore with `ls`. `None` leaves the text alone. The result is never longer
/// than `max_chars`, even when the marker itself does not fit. Mirrors Python's
/// `_truncate_manifest_description`.
pub fn truncate_manifest_description(description: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return description.to_string();
    };
    let total = description.chars().count();
    if total <= max_chars {
        return description.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    // The marker states how many characters were omitted, and how many are omitted depends on
    // how much room the marker leaves, so iterate until the two agree.
    let mut omitted = total - max_chars;
    let mut marker;
    let mut keep;
    let mut settled = 0;
    loop {
        marker = format!(
            "\n... (truncated {omitted} chars)\n\nThe filesystem layout above was truncated. Use \
             `ls` to explore specific directories before relying on omitted paths.\n"
        );
        keep = max_chars.saturating_sub(marker.chars().count());
        let actual = total - keep;
        settled += 1;
        if actual == omitted || settled >= 16 {
            break;
        }
        omitted = actual;
    }
    if marker.chars().count() >= max_chars {
        return marker.chars().take(max_chars).collect();
    }
    let kept: String = description.chars().take(keep).collect();
    let mut truncated = format!("{}{marker}", kept.trim_end());
    if truncated.chars().count() > max_chars {
        truncated = truncated.chars().take(max_chars).collect();
    }
    truncated
}

#[derive(Default)]
struct TreeNode {
    children: BTreeMap<String, TreeNode>,
    description: Option<String>,
    is_dir: bool,
    full_path: Option<String>,
}

struct TreeLine {
    prefix: String,
    name: String,
    full_path: String,
    description: Option<String>,
}

/// The non-empty, non-`.` segments of a manifest path, without any leading `/`.
fn path_parts(path: &str) -> Vec<String> {
    to_posix_path(path)
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_string)
        .collect()
}

fn insert_tree_path(
    root: &mut TreeNode,
    parts: &[String],
    description: Option<&str>,
    is_dir: bool,
    full_path: Option<String>,
    max_depth: Option<usize>,
) {
    if parts.is_empty() {
        return;
    }
    let limit = max_depth.map_or(parts.len(), |depth| depth.min(parts.len()));
    let mut node = root;
    for (index, part) in parts[..limit].iter().enumerate() {
        node = node.children.entry(part.clone()).or_default();
        if index < parts.len() - 1 {
            node.is_dir = true;
        }
    }
    if limit == parts.len() {
        if node.description.is_none() {
            node.description = description.map(str::to_string);
        }
        if full_path.is_some() {
            node.full_path = full_path;
        }
    }
    if is_dir || limit < parts.len() {
        node.is_dir = true;
    }
}

fn insert_entry_tree(
    root: &mut TreeNode,
    parts: &[String],
    entry: &Entry,
    full_path: Option<String>,
    depth: Option<usize>,
) {
    insert_tree_path(
        root,
        parts,
        entry.description.as_deref(),
        entry.permissions.directory,
        full_path.clone(),
        depth,
    );
    let EntryKind::Dir { children } = &entry.kind else {
        return;
    };
    if depth.is_some_and(|depth| parts.len() >= depth) {
        return;
    }
    for (name, child) in children {
        let mut child_parts = parts.to_vec();
        child_parts.extend(path_parts(name));
        let child_full = full_path.as_ref().map(|full| {
            format!(
                "{}/{}",
                full.trim_end_matches('/'),
                path_parts(name).join("/")
            )
        });
        insert_entry_tree(root, &child_parts, child, child_full, depth);
    }
}

fn collect_tree_lines(
    node: &TreeNode,
    prefix: &str,
    remaining: Option<usize>,
    rel_parts: &mut Vec<String>,
    root_prefix: &str,
    out: &mut Vec<TreeLine>,
) {
    if remaining == Some(0) {
        return;
    }
    let next_remaining = remaining.map(|remaining| remaining - 1);
    let count = node.children.len();
    for (index, (name, child)) in node.children.iter().enumerate() {
        let is_last = index + 1 == count;
        let connector = if is_last {
            "\u{2514}\u{2500}\u{2500} "
        } else {
            "\u{251c}\u{2500}\u{2500} "
        };
        rel_parts.push(name.clone());
        let is_dir = child.is_dir || !child.children.is_empty();
        out.push(TreeLine {
            prefix: format!("{prefix}{connector}"),
            name: if is_dir {
                format!("{name}/")
            } else {
                name.clone()
            },
            full_path: child
                .full_path
                .clone()
                .unwrap_or_else(|| format!("{root_prefix}/{}", rel_parts.join("/"))),
            description: child.description.clone(),
        });
        let extension = if is_last { "    " } else { "\u{2502}   " };
        collect_tree_lines(
            child,
            &format!("{prefix}{extension}"),
            next_remaining,
            rel_parts,
            root_prefix,
            out,
        );
        rel_parts.pop();
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

/// Apply one entry, confining its host sources to `base_dir` when there is one.
async fn apply_one(
    session: &dyn SandboxSession,
    entry: &Entry,
    dest: &str,
    base_dir: Option<&Path>,
) -> SandboxResult<MaterializationResult> {
    match base_dir {
        Some(base_dir) => entry.apply_confined(session, dest, base_dir).await,
        None => entry.apply(session, dest).await,
    }
}

/// Materialize a batch of independent entries concurrently and drain it.
///
/// The results are collected in the batch's own order, so a manifest's materialization result is
/// the same whatever order the backend happened to finish the entries in.
async fn apply_entry_batch(
    session: &dyn SandboxSession,
    batch: &mut Vec<(String, Entry)>,
    width: usize,
    base_dir: Option<&Path>,
) -> SandboxResult<MaterializationResult> {
    let mut written = MaterializationResult::new();
    if batch.is_empty() {
        return Ok(written);
    }
    let queued = std::mem::take(batch);
    let applications = queued.iter().map(|(dest, entry)| {
        move || async move {
            if entry.is_dir() {
                session.mkdir(dest, true).await?;
            } else if let Some(parent) = parent_path(dest) {
                session.mkdir(&parent, true).await?;
            }
            apply_one(session, entry, dest, base_dir).await
        }
    });
    for result in gather_in_order(applications, Some(width.max(1))).await? {
        written.extend(result);
    }
    Ok(written)
}

/// The error Python raises for a manifest-relative path that is absolute or climbs out of the
/// workspace. `rel` is the path as the caller wrote it, with POSIX separators.
fn invalid_manifest_path(rel: &str) -> SandboxError {
    let posix = to_posix_path(rel);
    let reason = if posix.starts_with('/') || is_windows_absolute(&posix) {
        "absolute"
    } else {
        "escape_root"
    };
    manifest_path_error(&posix, reason)
}

/// The same error with the reason named by the caller, for a path that is absolute in form but
/// refused because of where it leads (a mount path that climbs out of the workspace root).
fn manifest_path_error(rel: &str, reason: &str) -> SandboxError {
    let message = match reason {
        "absolute" => format!("manifest path must be relative: {rel}"),
        _ => format!("manifest path must not escape root: {rel}"),
    };
    SandboxError::new(ErrorCode::InvalidManifestPath, message)
        .with_op(crate::sandbox::errors::SandboxOp::Materialize)
        .with_context("rel", rel)
        .with_context("reason", reason)
}

/// Check one manifest-relative path: no Windows drive, no leading `/`, no `..` segment.
/// Mirrors Python's `Manifest._validate_rel_path`.
fn validate_rel_path(rel: &str) -> SandboxResult<()> {
    let posix = to_posix_path(rel);
    if is_windows_absolute(rel) || posix.starts_with('/') || posix.split('/').any(|s| s == "..") {
        return Err(invalid_manifest_path(rel));
    }
    Ok(())
}

/// Validate `rel` and, for a directory, every child path beneath it.
///
/// A child name joins the way `PurePath` joins: an absolute child replaces the parent path, so
/// it is reported as absolute rather than being quietly nested.
fn validate_entry_tree(rel: &str, entry: &Entry) -> SandboxResult<()> {
    validate_rel_path(rel)?;
    if let EntryKind::Dir { children } = &entry.kind {
        for (name, child) in children {
            let posix = to_posix_path(name);
            let child_rel = if is_windows_absolute(name) || posix.starts_with('/') {
                name.clone()
            } else {
                format!("{}/{name}", rel.trim_end_matches('/'))
            };
            validate_entry_tree(&child_rel, child)?;
        }
    }
    Ok(())
}

/// Keep the mounts whose entry is ephemeral.
fn ephemeral_only<'a>(targets: Vec<(&'a Entry, &'a Mount, String)>) -> Vec<(&'a Mount, String)> {
    targets
        .into_iter()
        .filter(|(entry, _, _)| entry.ephemeral)
        .map(|(_, mount, path)| (mount, path))
        .collect()
}

/// Collect the entries a resume has to re-apply, with their manifest-relative paths.
///
/// An ephemeral entry is taken whole and its subtree is not descended into, so a directory marked
/// ephemeral is re-created with every child it declares. A non-ephemeral directory is descended
/// into, so only the ephemeral entries inside it are re-applied. Mirrors Python's
/// `ManifestApplier._collect_ephemeral_entries`.
fn collect_ephemeral_entries<'a>(out: &mut Vec<(String, &'a Entry)>, rel: &str, entry: &'a Entry) {
    if entry.ephemeral {
        out.push((rel.to_string(), entry));
        return;
    }
    if let EntryKind::Dir { children } = &entry.kind {
        for (name, child) in children {
            let child_rel = format!("{}/{}", rel.trim_end_matches('/'), name.trim_matches('/'));
            collect_ephemeral_entries(out, &child_rel, child);
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

    #[tokio::test]
    async fn copies_a_local_directory_under_its_concurrency_limit() {
        use crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
        use crate::sandbox::session::SandboxClient;

        // Host sources are confined to the working directory.
        let host = tempfile::tempdir_in(".").unwrap();
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
        assert!(rendered.starts_with("/workspace\n"));
        assert!(rendered.contains("repo/"));
        assert!(rendered.contains("# /workspace/repo \u{2014} staged repository"));
        // The default depth stops at the manifest's own entries.
        assert!(!rendered.contains("task.md"));
        assert!(rendered.contains("notes.txt"));
        assert!(manifest.render(2, None).contains("task.md"));
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
        assert!(rendered.contains("(truncated "));
    }
}
