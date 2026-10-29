//! The sandbox manifest: what a fresh workspace contains. Mirrors
//! `agents.sandbox.manifest` and `agents.sandbox.manifest_render`.

use std::collections::{BTreeMap, BTreeSet};

use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::errors::SandboxResult;
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{Group, User};
use crate::sandbox::workspace_paths::resolve_workspace_path;

/// The default workspace root. Backends that own their workspace (such as Unix-local) replace
/// it with a real directory when they create a session.
pub const DEFAULT_MANIFEST_ROOT: &str = "/workspace";

/// Maximum size of the rendered workspace layout included in instructions.
pub const MAX_MANIFEST_DESCRIPTION_CHARS: usize = 5000;

/// Depth of the rendered workspace layout.
const MANIFEST_DESCRIPTION_DEPTH: usize = 3;

/// The files, directories and accounts a fresh sandbox session starts from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    version: u8,
    root: String,
    entries: BTreeMap<String, Entry>,
    environment: BTreeMap<String, String>,
    users: Vec<User>,
    groups: Vec<Group>,
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
        let validated = self.validated_entries()?;
        session.mkdir(&self.root, true).await?;
        self.provision_accounts(session).await;
        let mut written = Vec::new();
        for (_, dest, entry) in validated {
            if entry.is_dir() {
                session.mkdir(&dest, true).await?;
            } else if let Some(parent) = parent_path(&dest) {
                session.mkdir(&parent, true).await?;
            }
            written.extend(entry.apply(session, &dest).await?);
        }
        Ok(written)
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
