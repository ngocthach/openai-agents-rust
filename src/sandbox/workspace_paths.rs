//! POSIX path handling for sandbox workspaces. Mirrors `agents.sandbox.workspace_paths`.
//!
//! Every path the model or a manifest entry provides is normalized against the workspace root
//! before it reaches a session. Absolute paths outside the root, Windows-style absolute paths
//! and `..` segments are rejected.

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// Normalize a path to POSIX separators without resolving it.
pub fn to_posix(path: &str) -> String {
    path.replace('\\', "/")
}

/// True when the path uses Windows absolute syntax (`C:\x`, `\\server\share`).
pub fn is_windows_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    if path.starts_with("\\\\") {
        return true;
    }
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Collapse `.` segments and redundant separators, keeping any leading `/`.
///
/// `..` segments are preserved so callers can reject them explicitly.
pub fn normalize_segments(path: &str) -> String {
    let posix = to_posix(path);
    let absolute = posix.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for segment in posix.split('/') {
        match segment {
            "" | "." => {}
            ".." => parts.push(".."),
            other => parts.push(other),
        }
    }
    let joined = parts.join("/");
    if absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// True when `path` equals `root` or lies below it.
pub fn is_within(root: &str, path: &str) -> bool {
    let root = root.trim_end_matches('/');
    if root.is_empty() {
        return path.starts_with('/');
    }
    path == root || path.starts_with(&format!("{root}/"))
}

/// Resolve a workspace-relative (or root-anchored absolute) path to an absolute sandbox path.
///
/// Returns [`ErrorCode::InvalidManifestPath`] for Windows absolute paths, `..` segments and
/// absolute paths that fall outside `root`.
/// The directory prefix the SDK uses for files it stages outside the sandbox workspace.
///
/// A backend sometimes has to put a file where the model's workspace cannot reach it: Blaxel
/// stages a snapshot archive before downloading it, and Modal stages one before extracting it.
/// Those paths are SDK-owned, never model-provided, and
/// [`SandboxSession::resolve_internal_path`](crate::sandbox::SandboxSession::resolve_internal_path)
/// is the only place they bypass the workspace guard.
pub const PROVIDER_SCRATCH_PREFIX: &str = "/tmp/openai-agents-sandbox-";

/// Build an SDK-owned scratch path outside the workspace, for `stem`.
///
/// The stem is reduced to `[A-Za-z0-9._-]` so a caller cannot steer the path elsewhere.
pub fn provider_scratch_path(stem: &str) -> String {
    let sanitized: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("{PROVIDER_SCRATCH_PREFIX}{sanitized}")
}

/// Whether `path` is an SDK-owned scratch path built by [`provider_scratch_path`].
///
/// The check is deliberately strict: the path must start with the prefix and hold no `/` or
/// parent segment after it, so nothing under it can point back into the filesystem at large.
pub fn is_provider_scratch_path(path: &str) -> bool {
    let Some(stem) = path.strip_prefix(PROVIDER_SCRATCH_PREFIX) else {
        return false;
    };
    !stem.is_empty()
        && !stem.contains('/')
        && stem
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

pub fn resolve_workspace_path(root: &str, path: &str) -> SandboxResult<String> {
    if is_windows_absolute(path) {
        return Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("sandbox path `{path}` must use POSIX separators and stay workspace-relative"),
        ));
    }
    let root = {
        let trimmed = to_posix(root);
        let trimmed = trimmed.trim_end_matches('/').to_string();
        if trimmed.is_empty() {
            "/".to_string()
        } else {
            trimmed
        }
    };
    let candidate = normalize_segments(path);
    if candidate.split('/').any(|s| s == "..") {
        return Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("sandbox path `{path}` must not contain parent segments"),
        ));
    }
    let resolved = if candidate.starts_with('/') {
        candidate
    } else if candidate == "." {
        root.clone()
    } else {
        format!("{}/{}", root.trim_end_matches('/'), candidate)
    };
    if !is_within(&root, &resolved) {
        return Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("sandbox path `{path}` escapes the workspace root `{root}`"),
        ));
    }
    Ok(resolved)
}

/// Validate and normalize a run working directory relative to the workspace root.
pub fn normalize_sandbox_cwd(cwd: &str) -> SandboxResult<String> {
    if cwd.trim().is_empty() {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "sandbox cwd must be non-empty",
        ));
    }
    if cwd.contains('\\') || is_windows_absolute(cwd) {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "sandbox cwd must use POSIX path separators",
        ));
    }
    let normalized = normalize_segments(cwd);
    if normalized.starts_with('/') {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "sandbox cwd must be workspace-relative",
        ));
    }
    if normalized.split('/').any(|s| s == "..") {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "sandbox cwd must not contain parent segments",
        ));
    }
    Ok(normalized)
}

/// The immutable model-facing relative-path base for one sandbox run.
///
/// The scope only changes how relative paths are anchored; it does not confine the run. The
/// session's workspace policy remains responsible for access validation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SandboxWorkspaceScope {
    cwd: Option<String>,
}

impl SandboxWorkspaceScope {
    /// A scope anchored at the workspace root.
    pub fn root() -> Self {
        SandboxWorkspaceScope::default()
    }

    /// A scope anchored at a workspace-relative directory.
    pub fn from_cwd(cwd: Option<&str>) -> SandboxResult<Self> {
        match cwd {
            None => Ok(SandboxWorkspaceScope::default()),
            Some(cwd) => {
                let normalized = normalize_sandbox_cwd(cwd)?;
                Ok(SandboxWorkspaceScope {
                    cwd: (normalized != ".").then_some(normalized),
                })
            }
        }
    }

    /// The workspace-relative working directory, when one is configured.
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }

    /// Anchor a model-provided path: relative paths resolve from `cwd`, absolute paths pass
    /// through for the session to validate against the workspace root.
    pub fn anchor(&self, path: &str) -> String {
        let posix = to_posix(path);
        if posix.starts_with('/') || is_windows_absolute(&posix) {
            return posix;
        }
        match &self.cwd {
            None => posix,
            Some(cwd) => {
                if posix.is_empty() || posix == "." {
                    cwd.clone()
                } else {
                    format!("{cwd}/{posix}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_paths() {
        assert_eq!(
            resolve_workspace_path("/workspace", "repo/task.md").unwrap(),
            "/workspace/repo/task.md"
        );
        assert_eq!(
            resolve_workspace_path("/workspace/", "./a/./b").unwrap(),
            "/workspace/a/b"
        );
        assert_eq!(
            resolve_workspace_path("/workspace", ".").unwrap(),
            "/workspace"
        );
    }

    #[test]
    fn rejects_traversal_and_outside_paths() {
        for bad in ["../etc/passwd", "repo/../../etc", "/etc/passwd", "C:\\win"] {
            let err = resolve_workspace_path("/workspace", bad).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidManifestPath, "{bad}");
        }
        assert!(resolve_workspace_path("/workspace", "/workspace/ok").is_ok());
    }

    #[test]
    fn builds_and_recognizes_provider_scratch_paths() {
        let path = provider_scratch_path("persist-abc123.tar");
        assert_eq!(path, "/tmp/openai-agents-sandbox-persist-abc123.tar");
        assert!(is_provider_scratch_path(&path));
        // A stem cannot steer the path out of the scratch directory: the separators are
        // replaced, so what is left is one file name inside it.
        assert_eq!(
            provider_scratch_path("../../etc/passwd"),
            "/tmp/openai-agents-sandbox-..-..-etc-passwd"
        );
        for bad in [
            "/tmp/openai-agents-sandbox-",
            "/tmp/openai-agents-sandbox-a/b",
            "/tmp/other",
            "/workspace/file",
            "relative",
        ] {
            assert!(!is_provider_scratch_path(bad), "{bad}");
        }
    }

    #[test]
    fn scope_anchors_relative_paths() {
        let scope = SandboxWorkspaceScope::from_cwd(Some("repo/src")).unwrap();
        assert_eq!(scope.cwd(), Some("repo/src"));
        assert_eq!(scope.anchor("main.rs"), "repo/src/main.rs");
        assert_eq!(scope.anchor("/workspace/x"), "/workspace/x");
        assert_eq!(scope.anchor("."), "repo/src");
        let root = SandboxWorkspaceScope::root();
        assert_eq!(root.anchor("main.rs"), "main.rs");
        assert!(SandboxWorkspaceScope::from_cwd(Some("../x")).is_err());
        assert!(SandboxWorkspaceScope::from_cwd(Some("/abs")).is_err());
    }
}
