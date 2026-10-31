//! An [`ApplyPatchEditor`] that edits files inside a sandbox session. Mirrors
//! `agents.sandbox.apply_patch`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::apply_diff::{ApplyDiffMode, apply_diff};
use crate::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use crate::error::{AgentsError, Result};
use crate::sandbox::errors::{ErrorCode, SandboxError};
use crate::sandbox::session::SandboxSession;
use crate::sandbox::workspace_paths::SandboxWorkspaceScope;

/// Applies `apply_patch` operations to files inside a [`SandboxSession`].
///
/// Relative paths are anchored at the run's [`SandboxWorkspaceScope`] and then validated against
/// the session workspace root, so `..` segments and absolute paths outside the workspace are
/// rejected before any file is touched.
#[derive(Debug, Clone)]
pub struct SandboxEditor {
    session: Arc<dyn SandboxSession>,
    scope: SandboxWorkspaceScope,
    run_as: Option<crate::sandbox::types::User>,
    patch_format: Arc<dyn PatchFormat>,
}

/// The diff dialect an [`SandboxEditor`] speaks. Mirrors Python's `PatchFormat` protocol.
///
/// `apply_patch` is a wire format, not a filesystem operation: a model trained on a different
/// diff dialect can be served by swapping this out rather than by re-implementing the editor.
pub trait PatchFormat: std::fmt::Debug + Send + Sync {
    /// Apply `diff` to `input` and return the result.
    fn apply_diff(&self, input: &str, diff: &str, mode: ApplyDiffMode) -> Result<String>;
}

/// The default dialect, OpenAI's V4A patch format. Mirrors Python's `V4AFormat`.
#[derive(Debug, Clone, Copy, Default)]
pub struct V4aFormat;

impl PatchFormat for V4aFormat {
    fn apply_diff(&self, input: &str, diff: &str, mode: ApplyDiffMode) -> Result<String> {
        apply_diff(input, diff, mode)
    }
}

impl SandboxEditor {
    /// An editor bound to a session, anchored at the workspace root.
    pub fn new(session: Arc<dyn SandboxSession>) -> Self {
        SandboxEditor {
            session,
            scope: SandboxWorkspaceScope::root(),
            run_as: None,
            patch_format: Arc::new(V4aFormat),
        }
    }

    /// Read diffs in a dialect other than V4A.
    pub fn with_patch_format(mut self, patch_format: Arc<dyn PatchFormat>) -> Self {
        self.patch_format = patch_format;
        self
    }

    /// Rewrite an operation so its paths are in the workspace policy's canonical form.
    ///
    /// Mirrors Python's `normalize_operation`: each path becomes the workspace-relative POSIX
    /// path the operation would touch, after the run's working directory is applied. A caller
    /// that wants to inspect, log, dedupe or approve an operation before applying it needs the
    /// same path the editor will use, not the raw string the model produced, so
    /// `sensitive\secret.txt` and `sensitive/secret.txt` are one path to an approval rule.
    pub fn normalize_operation(
        &self,
        operation: &ApplyPatchOperation,
    ) -> Result<ApplyPatchOperation> {
        let mut normalized = operation.clone();
        normalized.path = self.canonical(&operation.path)?;
        normalized.move_to = match &operation.move_to {
            None => None,
            Some(destination) => Some(self.canonical(destination)?),
        };
        Ok(normalized)
    }

    /// Validate `path` and return it relative to the workspace root.
    fn canonical(&self, path: &str) -> Result<String> {
        let resolved = self.resolve(path)?;
        Ok(self
            .session
            .workspace_path_policy()
            .and_then(|policy| policy.relative_path(&resolved))
            .unwrap_or(resolved))
    }

    /// Edit files as `run_as` rather than as the session owner.
    ///
    /// A patch applied as the session owner can leave a file the agent's own commands cannot
    /// then write, so the editor uses the same user the capability runs commands as.
    pub fn with_run_as(mut self, run_as: Option<crate::sandbox::types::User>) -> Self {
        self.run_as = run_as;
        self
    }

    /// Anchor relative paths at the run's working directory.
    pub fn with_scope(mut self, scope: SandboxWorkspaceScope) -> Self {
        self.scope = scope;
        self
    }

    /// The session the editor writes to.
    pub fn session(&self) -> &Arc<dyn SandboxSession> {
        &self.session
    }

    /// Resolve a model-provided path to an absolute sandbox path.
    pub fn resolve(&self, path: &str) -> Result<String> {
        if path.trim().is_empty() {
            return Err(path_error(
                path,
                "empty",
                "apply_patch path must be non-empty",
            ));
        }
        let anchored = self.scope.anchor(path);
        let resolved = self.session.normalize_path(&anchored).map_err(|_| {
            path_error(
                path,
                "escape_root",
                &format!("apply_patch path must not escape root: {path}"),
            )
        })?;
        // A read-only grant widens what the sandbox can *read*; every `apply_patch` operation is
        // a write, so one aimed into such a grant is refused before anything is touched.
        if self.session.manifest().path_grant_is_read_only(&resolved) {
            return Err(AgentsError::from(
                SandboxError::new(
                    ErrorCode::ApplyPatchInvalidPath,
                    format!("`{path}` is inside a read-only path grant"),
                )
                .with_context("path", path)
                .with_context("reason", "read_only_extra_path_grant"),
            ));
        }
        Ok(resolved)
    }

    /// The path as the model should see it in a result.
    fn display(&self, original: &str, resolved: &str) -> String {
        let relative = self
            .session
            .workspace_path_policy()
            .and_then(|policy| policy.relative_path(resolved))
            .unwrap_or_else(|_| resolved.to_string());
        self.scope
            .display_path(original, &relative)
            .unwrap_or(relative)
    }

    async fn read_text(&self, resolved: &str, path: &str) -> Result<String> {
        let bytes = self
            .session
            .read_file_as(resolved, self.run_as.as_ref())
            .await
            .map_err(|e| match e.code {
                ErrorCode::WorkspaceReadNotFound => missing_file(path),
                _ => AgentsError::from(e),
            })?;
        String::from_utf8(bytes).map_err(|_| {
            AgentsError::from(
                SandboxError::new(
                    ErrorCode::ApplyPatchDecodeError,
                    format!("apply_patch could not decode file: {path}"),
                )
                .with_context("path", path),
            )
        })
    }

    /// Write `text`, creating its parent directories first.
    async fn write_text(&self, destination: &str, text: &str) -> Result<()> {
        if let Some((parent, _)) = destination.rsplit_once('/')
            && !parent.is_empty()
        {
            self.session
                .mkdir_as(parent, true, self.run_as.as_ref())
                .await?;
        }
        self.session
            .write_file_as(destination, text.as_bytes(), self.run_as.as_ref())
            .await?;
        Ok(())
    }

    /// Commit an updated file at its new path.
    ///
    /// Two paths that look different can be one file: a filesystem that folds case stores
    /// `notes.txt` and `Notes.txt` as one entry, and a parent symlink can alias a directory.
    /// Writing the destination and then removing the source would delete the updated file in
    /// that case, so aliases go through a staged replacement. Distinct files keep the ordinary
    /// write-then-remove path, which preserves an existing destination's metadata. Nothing is
    /// removed until the replacement is committed, and a failed move never writes the original
    /// back, because a concurrent writer may have put its own file at the source path since.
    async fn move_updated_text(&self, source: &str, destination: &str, text: &str) -> Result<()> {
        let user = self.run_as.as_ref();
        if source == destination {
            // Not a rename: write in place, which keeps the inode, the mode and the xattrs.
            // The comparison is on the spelling. Whether two sandbox paths are one file is the
            // sandbox's answer, so a case-only difference takes the staging path below.
            return self.write_text(source, text).await;
        }
        if !self
            .session
            .same_file(source, destination, false, user)
            .await?
        {
            self.write_text(destination, text).await?;
            self.session.rm_as(source, false, user).await?;
            return Ok(());
        }

        let staging = staging_path(destination);
        let staged = async {
            self.write_text(&staging, text).await?;
            self.session.mv_as(&staging, destination, user).await?;
            Ok::<_, AgentsError>(())
        }
        .await;
        if let Err(error) = staged {
            let _ = self.session.rm_as(&staging, false, user).await;
            return Err(error);
        }
        let same_entry = self
            .session
            .same_file(source, destination, false, user)
            .await?;
        let leaf = |path: &str| path.rsplit('/').next().unwrap_or(path).to_string();
        if same_entry && leaf(source) != leaf(destination) {
            // On case-folding APFS, replacing an existing entry through a case-variant path
            // keeps its old spelling. Moving that same entry performs the requested case-only
            // rename without touching the committed content.
            self.session.mv_as(source, destination, user).await?;
        } else if !same_entry {
            return Err(invalid_diff_message(
                "Move destination was updated, but source and destination no longer identify \
                 the same file; source was left untouched",
                None,
            ));
        }
        Ok(())
    }
}

/// A sibling name for a staged write that stays inside the 255-byte basename limit, which a name
/// derived from the destination's own name would not.
fn staging_path(destination: &str) -> String {
    let parent = destination
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .unwrap_or("");
    format!(
        "{parent}/.apply_patch-{}.tmp",
        uuid::Uuid::new_v4().simple()
    )
}

fn path_error(path: &str, reason: &str, message: &str) -> AgentsError {
    AgentsError::from(
        SandboxError::new(ErrorCode::ApplyPatchInvalidPath, message)
            .with_context("path", path)
            .with_context("reason", reason),
    )
}

fn missing_file(path: &str) -> AgentsError {
    AgentsError::from(
        SandboxError::new(
            ErrorCode::ApplyPatchFileNotFound,
            format!("apply_patch missing file: {path}"),
        )
        .with_context("path", path),
    )
}

fn invalid_diff_message(message: &str, path: Option<&str>) -> AgentsError {
    let mut error = SandboxError::new(ErrorCode::ApplyPatchInvalidDiff, message);
    if let Some(path) = path {
        error = error.with_context("path", path);
    }
    AgentsError::from(error)
}

#[async_trait]
impl ApplyPatchEditor for SandboxEditor {
    async fn create_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        let display = self.display(&operation.path, &resolved);
        let Some(diff) = operation.diff.as_deref() else {
            return Err(invalid_diff_message(
                &format!(
                    "Missing diff for operation type create_file on path {}",
                    operation.path
                ),
                Some(&operation.path),
            ));
        };
        let content = self
            .patch_format
            .apply_diff("", diff, ApplyDiffMode::Create)
            .map_err(|e| invalid_diff(e, &operation.path))?;
        // The backend's own create primitive decides whether the name is free, so a backend with
        // an exclusive create is not told the path is free by a racy read probe.
        match self
            .session
            .write_new_file(&resolved, content.as_bytes(), self.run_as.as_ref())
            .await
        {
            Ok(()) => {}
            Err(error) if error.message.contains("already exists") => {
                return Err(invalid_diff_message(
                    &format!(
                        "apply_patch cannot create {display} because it already exists. Use an \
                         update_file operation to change an existing file."
                    ),
                    Some(&display),
                ));
            }
            Err(error) => return Err(AgentsError::from(error)),
        }
        Ok(ApplyPatchResult::completed(format!("Created {display}")))
    }

    async fn update_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        let display = self.display(&operation.path, &resolved);
        let Some(diff) = operation.diff.as_deref() else {
            return Err(invalid_diff_message(
                &format!(
                    "Missing diff for operation type update_file on path {}",
                    operation.path
                ),
                Some(&operation.path),
            ));
        };
        let original = self.read_text(&resolved, &operation.path).await?;
        let updated = self
            .patch_format
            .apply_diff(&original, diff, ApplyDiffMode::Default)
            .map_err(|e| invalid_diff(e, &operation.path))?;
        match &operation.move_to {
            None => {
                self.write_text(&resolved, &updated).await?;
                Ok(ApplyPatchResult::completed(format!("Updated {display}")))
            }
            Some(destination) => {
                let move_target = self.resolve(destination)?;
                let moved_display = self.display(destination, &move_target);
                self.move_updated_text(&resolved, &move_target, &updated)
                    .await?;
                Ok(ApplyPatchResult::completed(format!(
                    "Updated {display}\nMoved {display} to {moved_display}"
                )))
            }
        }
    }

    async fn delete_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        let display = self.display(&operation.path, &resolved);
        // `rm -f` succeeds on a missing path, so read the file first: deleting something that is
        // not there is a model error worth reporting. Binary files must stay deletable, so this
        // reads bytes rather than going through `read_text`. The error names the model's path,
        // never a provider-private one.
        self.session
            .read_file_as(&resolved, self.run_as.as_ref())
            .await
            .map_err(|e| match e.code {
                ErrorCode::WorkspaceReadNotFound => missing_file(&display),
                _ => AgentsError::from(e),
            })?;
        self.session
            .rm_as(&resolved, false, self.run_as.as_ref())
            .await?;
        Ok(ApplyPatchResult::completed(format!("Deleted {display}")))
    }
}

fn invalid_diff(error: AgentsError, path: &str) -> AgentsError {
    AgentsError::from(
        SandboxError::new(ErrorCode::ApplyPatchInvalidDiff, error.to_string())
            .with_context("path", path),
    )
}
