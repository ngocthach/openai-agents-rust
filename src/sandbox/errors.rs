//! Sandbox errors and their stable error codes. Mirrors `agents.sandbox.errors`.

use std::fmt;

use crate::error::AgentsError;
use crate::sandbox::types::ExecResult;

/// Stable, machine-readable error codes attached to every [`SandboxError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// A manifest path was absolute, empty or escaped the workspace root.
    InvalidManifestPath,
    /// A command exited with a non-zero status.
    ExecNonZero,
    /// A command exceeded its timeout.
    ExecTimeout,
    /// The sandbox transport failed before the command produced a result.
    ExecTransportError,
    /// An `apply_patch` operation targeted an invalid path.
    ApplyPatchInvalidPath,
    /// An `apply_patch` diff could not be applied.
    ApplyPatchInvalidDiff,
    /// An `apply_patch` operation targeted a missing file.
    ApplyPatchFileNotFound,
    /// A file read returned bytes that are not valid UTF-8.
    ApplyPatchDecodeError,
    /// A workspace read targeted a missing path.
    WorkspaceReadNotFound,
    /// A workspace write failed because the target has the wrong type.
    WorkspaceWriteTypeError,
    /// Starting the sandbox workspace failed.
    WorkspaceStartError,
    /// Stopping the sandbox workspace failed.
    WorkspaceStopError,
    /// The workspace root does not exist.
    WorkspaceRootNotFound,
    /// A host file referenced by a manifest entry could not be read.
    LocalFileReadError,
    /// A host directory referenced by a manifest entry could not be read.
    LocalDirReadError,
    /// `git` is not available inside the sandbox image.
    GitMissingInImage,
    /// Cloning a git repository failed.
    GitCloneError,
    /// The requested git subpath does not exist in the clone.
    GitSubpathError,
    /// The sandbox configuration is invalid.
    SandboxConfigInvalid,
    /// Persisting a workspace snapshot failed.
    SnapshotPersistError,
    /// Restoring a workspace snapshot failed.
    SnapshotRestoreError,
}

impl ErrorCode {
    /// The wire name of the code.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidManifestPath => "invalid_manifest_path",
            ErrorCode::ExecNonZero => "exec_nonzero",
            ErrorCode::ExecTimeout => "exec_timeout",
            ErrorCode::ExecTransportError => "exec_transport_error",
            ErrorCode::ApplyPatchInvalidPath => "apply_patch_invalid_path",
            ErrorCode::ApplyPatchInvalidDiff => "apply_patch_invalid_diff",
            ErrorCode::ApplyPatchFileNotFound => "apply_patch_file_not_found",
            ErrorCode::ApplyPatchDecodeError => "apply_patch_decode_error",
            ErrorCode::WorkspaceReadNotFound => "workspace_read_not_found",
            ErrorCode::WorkspaceWriteTypeError => "workspace_write_type_error",
            ErrorCode::WorkspaceStartError => "workspace_start_error",
            ErrorCode::WorkspaceStopError => "workspace_stop_error",
            ErrorCode::WorkspaceRootNotFound => "workspace_root_not_found",
            ErrorCode::LocalFileReadError => "local_file_read_error",
            ErrorCode::LocalDirReadError => "local_dir_read_error",
            ErrorCode::GitMissingInImage => "git_missing_in_image",
            ErrorCode::GitCloneError => "git_clone_error",
            ErrorCode::GitSubpathError => "git_subpath_error",
            ErrorCode::SandboxConfigInvalid => "sandbox_config_invalid",
            ErrorCode::SnapshotPersistError => "snapshot_persist_error",
            ErrorCode::SnapshotRestoreError => "snapshot_restore_error",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error raised by a sandbox session, client or manifest entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    /// The stable error code.
    pub code: ErrorCode,
    /// A human-readable message.
    pub message: String,
    /// The exit code of the failed command, when the error came from `exec`.
    pub exit_code: Option<i32>,
}

impl SandboxError {
    /// Build an error with a code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        SandboxError {
            code,
            message: message.into(),
            exit_code: None,
        }
    }

    /// Build an [`ErrorCode::ExecNonZero`] error from a failed command.
    pub fn exec_nonzero(command: &[String], result: &ExecResult) -> Self {
        let stderr = result.stderr_text();
        let detail = if stderr.trim().is_empty() {
            result.stdout_text()
        } else {
            stderr
        };
        SandboxError {
            code: ErrorCode::ExecNonZero,
            message: format!(
                "Command `{}` exited with code {}: {}",
                command.join(" "),
                result.exit_code,
                detail.trim()
            ),
            exit_code: Some(result.exit_code),
        }
    }

    /// True when the error is an exec timeout.
    pub fn is_timeout(&self) -> bool {
        self.code == ErrorCode::ExecTimeout
    }
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for SandboxError {}

impl From<SandboxError> for AgentsError {
    fn from(value: SandboxError) -> Self {
        match value.code {
            ErrorCode::InvalidManifestPath
            | ErrorCode::ApplyPatchInvalidPath
            | ErrorCode::SandboxConfigInvalid => AgentsError::user(value.to_string()),
            _ => AgentsError::other(value.to_string()),
        }
    }
}

/// A `Result` whose error is a [`SandboxError`].
pub type SandboxResult<T> = std::result::Result<T, SandboxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_path_errors_to_user_errors() {
        let err: AgentsError =
            SandboxError::new(ErrorCode::InvalidManifestPath, "absolute path").into();
        assert!(matches!(err, AgentsError::UserError { .. }));
        let err: AgentsError = SandboxError::new(ErrorCode::ExecTimeout, "slow").into();
        assert!(matches!(err, AgentsError::Other(_)));
    }

    #[test]
    fn formats_exec_failures() {
        let err = SandboxError::exec_nonzero(
            &["ls".to_string()],
            &ExecResult {
                stdout: Vec::new(),
                stderr: b"no such file\n".to_vec(),
                exit_code: 2,
            },
        );
        assert_eq!(err.exit_code, Some(2));
        assert!(err.to_string().contains("no such file"));
    }
}
