//! Sandbox errors and their stable error codes. Mirrors `agents.sandbox.errors`.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::AgentsError;
use crate::sandbox::types::ExecResult;

/// Stable, machine-readable error codes attached to every [`SandboxError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// A manifest path was absolute, empty or escaped the workspace root.
    InvalidManifestPath,
    /// A workspace archive used a compression scheme the backend cannot read.
    InvalidCompressionScheme,
    /// A command exited with a non-zero status.
    ExecNonZero,
    /// A command exceeded its timeout.
    ExecTimeout,
    /// The sandbox transport failed before the command produced a result.
    ExecTransportError,
    /// A PTY session id does not refer to a running process.
    PtySessionNotFound,
    /// The PTY process was not started with a TTY, so it has no stdin.
    PtyStdinUnavailable,
    /// A mount is configured in a way the backend cannot satisfy.
    MountConfigInvalid,
    /// Attaching or detaching a mount failed.
    MountFailed,
    /// The tool a mount pattern needs is missing inside the sandbox.
    MountMissingTool,
    /// The persisted sandbox session state does not match the trusted configuration.
    SessionStateMismatch,
    /// The persisted sandbox session state is malformed.
    SessionStateInvalid,
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
    /// Reading a workspace archive failed.
    WorkspaceArchiveReadError,
    /// Writing a workspace archive failed.
    WorkspaceArchiveWriteError,
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
    /// Checksumming a host file while materializing it failed.
    LocalChecksumError,
    /// `git` is not available inside the sandbox image.
    GitMissingInImage,
    /// Cloning a git repository failed.
    GitCloneError,
    /// The requested git subpath does not exist in the clone.
    GitSubpathError,
    /// Copying a cloned git tree into the workspace failed.
    GitCopyError,
    /// The skills capability is misconfigured.
    SkillsConfigInvalid,
    /// The sandbox configuration is invalid.
    SandboxConfigInvalid,
    /// A sandbox event sink that raises failed to record an operation, which fails the operation.
    ///
    /// Mirrors the `RuntimeError("sandbox event sink failed")` Python raises for `on_error="raise"`.
    EventSinkFailed,
    /// A port the caller asked to reach cannot be resolved to a host endpoint.
    ///
    /// Mirrors Python's `ExposedPortUnavailableError`. The `reason` context says which case it
    /// is: `not_configured` when the port was never declared on the client options, and
    /// `backend_unavailable` when the provider cannot hand out an endpoint for it.
    ExposedPortUnavailable,
    /// Persisting a workspace snapshot failed.
    SnapshotPersistError,
    /// Restoring a workspace snapshot failed.
    SnapshotRestoreError,
    /// The snapshot stores no workspace archive.
    SnapshotNotRestorable,
}

impl ErrorCode {
    /// The wire name of the code.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidManifestPath => "invalid_manifest_path",
            ErrorCode::InvalidCompressionScheme => "invalid_compression_scheme",
            ErrorCode::ExecNonZero => "exec_nonzero",
            ErrorCode::ExecTimeout => "exec_timeout",
            ErrorCode::ExecTransportError => "exec_transport_error",
            ErrorCode::PtySessionNotFound => "pty_session_not_found",
            ErrorCode::PtyStdinUnavailable => "pty_stdin_unavailable",
            ErrorCode::MountConfigInvalid => "mount_config_invalid",
            ErrorCode::MountFailed => "mount_failed",
            ErrorCode::MountMissingTool => "mount_missing_tool",
            ErrorCode::SessionStateMismatch => "session_state_mismatch",
            ErrorCode::SessionStateInvalid => "session_state_invalid",
            ErrorCode::ApplyPatchInvalidPath => "apply_patch_invalid_path",
            ErrorCode::ApplyPatchInvalidDiff => "apply_patch_invalid_diff",
            ErrorCode::ApplyPatchFileNotFound => "apply_patch_file_not_found",
            ErrorCode::ApplyPatchDecodeError => "apply_patch_decode_error",
            ErrorCode::WorkspaceReadNotFound => "workspace_read_not_found",
            ErrorCode::WorkspaceArchiveReadError => "workspace_archive_read_error",
            ErrorCode::WorkspaceArchiveWriteError => "workspace_archive_write_error",
            ErrorCode::WorkspaceWriteTypeError => "workspace_write_type_error",
            ErrorCode::WorkspaceStartError => "workspace_start_error",
            ErrorCode::WorkspaceStopError => "workspace_stop_error",
            ErrorCode::WorkspaceRootNotFound => "workspace_root_not_found",
            ErrorCode::LocalFileReadError => "local_file_read_error",
            ErrorCode::LocalDirReadError => "local_dir_read_error",
            ErrorCode::LocalChecksumError => "local_checksum_error",
            ErrorCode::GitMissingInImage => "git_missing_in_image",
            ErrorCode::GitCloneError => "git_clone_error",
            ErrorCode::GitSubpathError => "git_subpath_error",
            ErrorCode::GitCopyError => "git_copy_error",
            ErrorCode::SkillsConfigInvalid => "skills_config_invalid",
            ErrorCode::SandboxConfigInvalid => "sandbox_config_invalid",
            ErrorCode::EventSinkFailed => "event_sink_failed",
            ErrorCode::ExposedPortUnavailable => "exposed_port_unavailable",
            ErrorCode::SnapshotPersistError => "snapshot_persist_error",
            ErrorCode::SnapshotRestoreError => "snapshot_restore_error",
            ErrorCode::SnapshotNotRestorable => "snapshot_not_restorable",
        }
    }

    /// Whether an error with this code is retryable, before a caller says otherwise.
    ///
    /// Mirrors the `retryable` default each Python error class passes to `SandboxError`: a code
    /// that can only mean a deterministic refusal answers `Some(false)`, while a code that covers
    /// both a transport blip and a permanent failure answers `None`, which means "unknown" rather
    /// than "no". [`SandboxError::new`] seeds the field with this, so a caller that knows better
    /// still overrides it with [`with_retryable`](SandboxError::with_retryable).
    pub fn default_retryable(self) -> Option<bool> {
        match self {
            // Deterministic: the same call fails the same way every time.
            ErrorCode::InvalidManifestPath
            | ErrorCode::InvalidCompressionScheme
            | ErrorCode::ExecNonZero
            | ErrorCode::ExecTimeout
            | ErrorCode::PtySessionNotFound
            | ErrorCode::PtyStdinUnavailable
            | ErrorCode::SessionStateMismatch
            | ErrorCode::SessionStateInvalid
            | ErrorCode::ApplyPatchInvalidPath
            | ErrorCode::ApplyPatchInvalidDiff
            | ErrorCode::ApplyPatchFileNotFound
            | ErrorCode::ApplyPatchDecodeError
            | ErrorCode::WorkspaceReadNotFound
            | ErrorCode::WorkspaceWriteTypeError
            | ErrorCode::WorkspaceRootNotFound
            | ErrorCode::LocalFileReadError
            | ErrorCode::LocalDirReadError
            | ErrorCode::LocalChecksumError
            | ErrorCode::GitMissingInImage
            | ErrorCode::GitSubpathError
            | ErrorCode::MountMissingTool
            | ErrorCode::MountConfigInvalid
            | ErrorCode::SkillsConfigInvalid
            | ErrorCode::SandboxConfigInvalid
            | ErrorCode::SnapshotNotRestorable
            | ErrorCode::EventSinkFailed => Some(false),
            // Broad: the same code covers a dropped connection and a permanent refusal.
            ErrorCode::ExecTransportError
            | ErrorCode::ExposedPortUnavailable
            | ErrorCode::MountFailed
            | ErrorCode::WorkspaceArchiveReadError
            | ErrorCode::WorkspaceArchiveWriteError
            | ErrorCode::WorkspaceStartError
            | ErrorCode::WorkspaceStopError
            | ErrorCode::GitCloneError
            | ErrorCode::GitCopyError
            | ErrorCode::SnapshotPersistError
            | ErrorCode::SnapshotRestoreError => None,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The sandbox operation an error came from. Mirrors Python's `OpName`.
///
/// Carrying the operation lets a caller tell a failed `write` from a failed `exec` without
/// parsing the message, which matters when the same error code can come from either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SandboxOp {
    /// Starting the workspace.
    Start,
    /// Stopping the workspace.
    Stop,
    /// Running a command.
    Exec,
    /// Reading a workspace file.
    Read,
    /// Writing a workspace file.
    Write,
    /// Tearing the sandbox down.
    Shutdown,
    /// Checking whether the sandbox is running.
    Running,
    /// Capturing the workspace into an archive.
    PersistWorkspace,
    /// Restoring the workspace from an archive.
    HydrateWorkspace,
    /// Resolving an exposed port to a reachable endpoint.
    ResolveExposedPort,
    /// Materializing a manifest entry.
    Materialize,
    /// Persisting a snapshot.
    SnapshotPersist,
    /// Restoring a snapshot.
    SnapshotRestore,
    /// Applying a patch.
    ApplyPatch,
}

impl SandboxOp {
    /// The wire name of the operation.
    pub fn as_str(self) -> &'static str {
        match self {
            SandboxOp::Start => "start",
            SandboxOp::Stop => "stop",
            SandboxOp::Exec => "exec",
            SandboxOp::Read => "read",
            SandboxOp::Write => "write",
            SandboxOp::Shutdown => "shutdown",
            SandboxOp::Running => "running",
            SandboxOp::PersistWorkspace => "persist_workspace",
            SandboxOp::HydrateWorkspace => "hydrate_workspace",
            SandboxOp::ResolveExposedPort => "resolve_exposed_port",
            SandboxOp::Materialize => "materialize",
            SandboxOp::SnapshotPersist => "snapshot_persist",
            SandboxOp::SnapshotRestore => "snapshot_restore",
            SandboxOp::ApplyPatch => "apply_patch",
        }
    }
}

impl fmt::Display for SandboxOp {
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
    /// The operation that failed, when the raising code names one.
    pub op: Option<SandboxOp>,
    /// Structured metadata for debugging. Never holds a credential.
    pub context: BTreeMap<String, String>,
    /// Whether retrying the same operation is expected to succeed.
    ///
    /// `None` means the SDK cannot safely classify the error, which is not the same as "no":
    /// a caller that retries on `None` is guessing.
    pub retryable: Option<bool>,
}

impl SandboxError {
    /// Build an error with a code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        SandboxError {
            code,
            message: message.into(),
            exit_code: None,
            op: None,
            context: BTreeMap::new(),
            retryable: code.default_retryable(),
        }
    }

    /// Name the operation that failed.
    pub fn with_op(mut self, op: SandboxOp) -> Self {
        self.op = Some(op);
        self
    }

    /// Attach one piece of debugging metadata.
    ///
    /// The value reaches logs and error messages, so it must never be a credential.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Record whether retrying the same operation is expected to succeed.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    /// Inherit the retry classification of the error that caused this one.
    ///
    /// Mirrors Python's `__post_init__`: an unclassified error wrapping a transient one is
    /// itself transient, and wrapping it must not quietly lose that.
    pub fn caused_by(mut self, cause: &SandboxError) -> Self {
        if self.retryable.is_none() {
            self.retryable = cause.retryable;
        }
        self
    }

    /// Build an [`ErrorCode::ExecNonZero`] error from a failed command.
    ///
    /// Mirrors Python's `ExecNonZeroError`: the message is the command's own output, both streams
    /// labelled when both are present, and only falls back to the exit code when the command said
    /// nothing. The command, exit code and decoded streams ride in the context, where a caller
    /// that wants them can read them without parsing the message.
    pub fn exec_nonzero(command: &[String], result: &ExecResult) -> Self {
        let stdout = result.stdout_text();
        let stderr = result.stderr_text();
        let message = match (stdout.is_empty(), stderr.is_empty()) {
            (false, false) => format!("stdout: {stdout}\nstderr: {stderr}"),
            (false, true) => stdout.clone(),
            (true, false) => stderr.clone(),
            (true, true) => format!("command exited with code {}", result.exit_code),
        };
        let command_json = serde_json::to_string(command).unwrap_or_default();
        SandboxError {
            code: ErrorCode::ExecNonZero,
            message,
            exit_code: Some(result.exit_code),
            op: Some(SandboxOp::Exec),
            context: BTreeMap::from([
                ("command".to_string(), command_json),
                ("command_str".to_string(), command.join(" ")),
                ("exit_code".to_string(), result.exit_code.to_string()),
                ("stdout".to_string(), stdout),
                ("stderr".to_string(), stderr),
            ]),
            // A command that ran and failed will fail the same way again.
            retryable: Some(false),
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
    /// Convert into the dedicated [`AgentsError::Sandbox`] variant, keeping the error code
    /// machine-readable so callers can branch on the cause.
    fn from(value: SandboxError) -> Self {
        AgentsError::Sandbox {
            code: value.code.as_str().to_string(),
            message: value.message,
        }
    }
}

/// A `Result` whose error is a [`SandboxError`].
pub type SandboxResult<T> = std::result::Result<T, SandboxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_into_the_sandbox_error_variant() {
        let err: AgentsError =
            SandboxError::new(ErrorCode::InvalidManifestPath, "absolute path").into();
        assert_eq!(err.sandbox_code(), Some("invalid_manifest_path"));
        assert_eq!(err.to_string(), "[invalid_manifest_path] absolute path");

        let err: AgentsError = SandboxError::new(ErrorCode::ExecTimeout, "slow").into();
        assert_eq!(err.sandbox_code(), Some("exec_timeout"));
        assert!(matches!(err, AgentsError::Sandbox { .. }));

        // Non-sandbox errors keep reporting no code.
        assert_eq!(AgentsError::user("plain").sandbox_code(), None);
    }

    #[test]
    fn round_trips_the_code_through_the_agents_error() {
        for code in [
            ErrorCode::ExecNonZero,
            ErrorCode::MountConfigInvalid,
            ErrorCode::SessionStateMismatch,
            ErrorCode::PtySessionNotFound,
        ] {
            let err: AgentsError = SandboxError::new(code, "boom").into();
            assert_eq!(err.sandbox_code(), Some(code.as_str()));
        }
    }

    #[test]
    fn inherits_the_retry_classification_of_a_cause() {
        let transient =
            SandboxError::new(ErrorCode::ExecTransportError, "reset").with_retryable(true);
        let wrapped =
            SandboxError::new(ErrorCode::WorkspaceStartError, "failed").caused_by(&transient);
        assert_eq!(wrapped.retryable, Some(true));

        // An explicit classification wins over the cause's.
        let pinned = SandboxError::new(ErrorCode::WorkspaceStartError, "failed")
            .with_retryable(false)
            .caused_by(&transient);
        assert_eq!(pinned.retryable, Some(false));
    }

    #[test]
    fn carries_the_operation_and_its_context() {
        let error = SandboxError::new(ErrorCode::WorkspaceArchiveWriteError, "no space")
            .with_op(SandboxOp::PersistWorkspace)
            .with_context("root", "/workspace");
        assert_eq!(error.op.map(SandboxOp::as_str), Some("persist_workspace"));
        assert_eq!(
            error.context.get("root").map(String::as_str),
            Some("/workspace")
        );
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
        assert_eq!(err.message, "no such file\n");
        assert!(err.to_string().contains("no such file"));
        assert_eq!(
            err.context.get("command_str").map(String::as_str),
            Some("ls")
        );
        assert_eq!(err.context.get("exit_code").map(String::as_str), Some("2"));
        assert_eq!(err.context.get("stdout").map(String::as_str), Some(""));

        let both = SandboxError::exec_nonzero(
            &["ls".to_string()],
            &ExecResult {
                stdout: b"out".to_vec(),
                stderr: b"err".to_vec(),
                exit_code: 1,
            },
        );
        assert_eq!(both.message, "stdout: out\nstderr: err");

        let silent = SandboxError::exec_nonzero(
            &["true".to_string()],
            &ExecResult {
                stdout: Vec::new(),
                stderr: Vec::new(),
                exit_code: 3,
            },
        );
        assert_eq!(silent.message, "command exited with code 3");
    }
}
