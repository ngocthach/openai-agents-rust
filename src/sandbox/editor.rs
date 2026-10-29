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
}

impl SandboxEditor {
    /// An editor bound to a session, anchored at the workspace root.
    pub fn new(session: Arc<dyn SandboxSession>) -> Self {
        SandboxEditor {
            session,
            scope: SandboxWorkspaceScope::root(),
        }
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
        let anchored = self.scope.anchor(path);
        self.session.normalize_path(&anchored).map_err(|e| {
            AgentsError::from(SandboxError::new(
                ErrorCode::ApplyPatchInvalidPath,
                e.message,
            ))
        })
    }

    async fn read_text(&self, resolved: &str, path: &str) -> Result<String> {
        let bytes = self
            .session
            .read_file(resolved)
            .await
            .map_err(|e| match e.code {
                ErrorCode::WorkspaceReadNotFound => AgentsError::from(SandboxError::new(
                    ErrorCode::ApplyPatchFileNotFound,
                    format!("File `{path}` does not exist in the sandbox workspace"),
                )),
                _ => AgentsError::from(e),
            })?;
        String::from_utf8(bytes).map_err(|_| {
            AgentsError::from(SandboxError::new(
                ErrorCode::ApplyPatchDecodeError,
                format!("File `{path}` is not valid UTF-8"),
            ))
        })
    }
}

#[async_trait]
impl ApplyPatchEditor for SandboxEditor {
    async fn create_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        let content = apply_diff(
            "",
            operation.diff.as_deref().unwrap_or(""),
            ApplyDiffMode::Create,
        )
        .map_err(invalid_diff)?;
        self.session
            .write_file(&resolved, content.as_bytes())
            .await?;
        Ok(ApplyPatchResult::completed(format!(
            "Created {}",
            operation.path
        )))
    }

    async fn update_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        let original = self.read_text(&resolved, &operation.path).await?;
        let updated = apply_diff(
            &original,
            operation.diff.as_deref().unwrap_or(""),
            ApplyDiffMode::Default,
        )
        .map_err(invalid_diff)?;
        match &operation.move_to {
            None => {
                self.session
                    .write_file(&resolved, updated.as_bytes())
                    .await?;
                Ok(ApplyPatchResult::completed(format!(
                    "Updated {}",
                    operation.path
                )))
            }
            Some(destination) => {
                let move_target = self.resolve(destination)?;
                self.session
                    .write_file(&move_target, updated.as_bytes())
                    .await?;
                if move_target != resolved {
                    self.session.rm(&resolved, false).await?;
                }
                Ok(ApplyPatchResult::completed(format!(
                    "Updated {} and moved it to {destination}",
                    operation.path
                )))
            }
        }
    }

    async fn delete_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let resolved = self.resolve(&operation.path)?;
        // `rm -f` succeeds on a missing path, so read the file first: deleting something that is
        // not there is a model error worth reporting. Binary files must stay deletable, so this
        // reads bytes rather than going through `read_text`.
        self.session
            .read_file(&resolved)
            .await
            .map_err(|e| match e.code {
                ErrorCode::WorkspaceReadNotFound => AgentsError::from(SandboxError::new(
                    ErrorCode::ApplyPatchFileNotFound,
                    format!(
                        "File `{}` does not exist in the sandbox workspace",
                        operation.path
                    ),
                )),
                _ => AgentsError::from(e),
            })?;
        self.session.rm(&resolved, false).await?;
        Ok(ApplyPatchResult::completed(format!(
            "Deleted {}",
            operation.path
        )))
    }
}

fn invalid_diff(error: AgentsError) -> AgentsError {
    AgentsError::from(SandboxError::new(
        ErrorCode::ApplyPatchInvalidDiff,
        error.to_string(),
    ))
}
