//! The editor interface used by [`ApplyPatchTool`](crate::tool::ApplyPatchTool).
//!
//! Port of `agents.editor`, plus [`WorkspaceEditor`], a ready-to-use editor that applies V4A
//! diffs to files under a root directory.

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::apply_diff::{ApplyDiffMode, apply_diff};
use crate::error::{AgentsError, Result};

/// The kind of file operation requested by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyPatchOperationType {
    /// Create a new file from a create-mode diff.
    CreateFile,
    /// Update an existing file with a V4A diff.
    UpdateFile,
    /// Delete a file.
    DeleteFile,
}

impl ApplyPatchOperationType {
    /// The wire name of the operation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ApplyPatchOperationType::CreateFile => "create_file",
            ApplyPatchOperationType::UpdateFile => "update_file",
            ApplyPatchOperationType::DeleteFile => "delete_file",
        }
    }
}

/// One file operation requested by an `apply_patch_call`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyPatchOperation {
    /// The operation type.
    #[serde(rename = "type")]
    pub op_type: ApplyPatchOperationType,
    /// Target path, relative to the workspace.
    pub path: String,
    /// The V4A diff (create and update operations).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// New path when an update also renames the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_to: Option<String>,
}

/// The status reported by an editor operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyPatchStatus {
    /// The operation succeeded.
    Completed,
    /// The operation failed.
    Failed,
}

/// Optional metadata returned by editor operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApplyPatchResult {
    /// Operation status; `None` keeps the overall status unchanged.
    pub status: Option<ApplyPatchStatus>,
    /// Output text shown to the model.
    pub output: Option<String>,
}

impl ApplyPatchResult {
    /// A completed result with output text.
    pub fn completed(output: impl Into<String>) -> Self {
        ApplyPatchResult {
            status: Some(ApplyPatchStatus::Completed),
            output: Some(output.into()),
        }
    }

    /// A failed result with output text.
    pub fn failed(output: impl Into<String>) -> Self {
        ApplyPatchResult {
            status: Some(ApplyPatchStatus::Failed),
            output: Some(output.into()),
        }
    }
}

impl From<String> for ApplyPatchResult {
    fn from(value: String) -> Self {
        ApplyPatchResult {
            status: None,
            output: Some(value),
        }
    }
}

impl From<&str> for ApplyPatchResult {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

/// Host-defined editor that applies file operations. Errors are reported to the model as a
/// failed `apply_patch_call_output`.
#[async_trait]
pub trait ApplyPatchEditor: Send + Sync {
    /// Create a file.
    async fn create_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;
    /// Update a file.
    async fn update_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;
    /// Delete a file.
    async fn delete_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult>;
}

/// An [`ApplyPatchEditor`] that edits files under a root directory. Paths that escape the root
/// (absolute paths or `..`) are rejected.
#[derive(Debug, Clone)]
pub struct WorkspaceEditor {
    root: PathBuf,
}

impl WorkspaceEditor {
    /// Create an editor rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        WorkspaceEditor { root: root.into() }
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve(&self, relative: &str) -> Result<PathBuf> {
        let path = Path::new(relative);
        let safe = !relative.is_empty()
            && path
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        if !safe {
            return Err(AgentsError::user(format!(
                "Path '{relative}' is outside the workspace"
            )));
        }
        Ok(self.root.join(path))
    }
}

async fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, content).await?;
    Ok(())
}

#[async_trait]
impl ApplyPatchEditor for WorkspaceEditor {
    async fn create_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let target = self.resolve(&op.path)?;
        let content = apply_diff("", op.diff.as_deref().unwrap_or(""), ApplyDiffMode::Create)?;
        write_file(&target, &content).await?;
        Ok(ApplyPatchResult::completed(format!("Created {}", op.path)))
    }

    async fn update_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let target = self.resolve(&op.path)?;
        let original = tokio::fs::read_to_string(&target).await?;
        let updated = apply_diff(
            &original,
            op.diff.as_deref().unwrap_or(""),
            ApplyDiffMode::Default,
        )?;
        match &op.move_to {
            Some(dest) => {
                let dest_path = self.resolve(dest)?;
                write_file(&dest_path, &updated).await?;
                if dest_path != target {
                    tokio::fs::remove_file(&target).await?;
                }
                Ok(ApplyPatchResult::completed(format!(
                    "Updated {} and moved it to {dest}",
                    op.path
                )))
            }
            None => {
                write_file(&target, &updated).await?;
                Ok(ApplyPatchResult::completed(format!("Updated {}", op.path)))
            }
        }
    }

    async fn delete_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let target = self.resolve(&op.path)?;
        tokio::fs::remove_file(&target).await?;
        Ok(ApplyPatchResult::completed(format!("Deleted {}", op.path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn workspace_editor_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let editor = WorkspaceEditor::new(dir.path());
        let create = ApplyPatchOperation {
            op_type: ApplyPatchOperationType::CreateFile,
            path: "src/a.txt".into(),
            diff: Some("+hello\n+world\n".into()),
            move_to: None,
        };
        editor.create_file(&create).await.unwrap();
        let update = ApplyPatchOperation {
            op_type: ApplyPatchOperationType::UpdateFile,
            path: "src/a.txt".into(),
            diff: Some("@@\n hello\n-world\n+rust\n".into()),
            move_to: Some("b.txt".into()),
        };
        editor.update_file(&update).await.unwrap();
        let text = std::fs::read_to_string(dir.path().join("b.txt")).unwrap();
        assert_eq!(text, "hello\nrust");
        assert!(!dir.path().join("src/a.txt").exists());
        let escape = ApplyPatchOperation {
            op_type: ApplyPatchOperationType::DeleteFile,
            path: "../x".into(),
            diff: None,
            move_to: None,
        };
        assert!(editor.delete_file(&escape).await.is_err());
    }
}
