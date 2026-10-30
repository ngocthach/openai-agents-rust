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

/// Access to an absolute path outside the sandbox workspace. Mirrors Python's
/// `SandboxPathGrant`.
///
/// The workspace root is the only place a sandbox can reach by default. A grant widens that
/// deliberately and visibly — a shared cache, a read-only dataset — and carries its own
/// `read_only` flag so widening access does not have to mean granting writes. `host_path` names
/// the native directory behind it when the backend has to bind-mount or copy from the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPathGrant {
    /// The POSIX path visible inside the sandbox.
    pub path: String,
    /// Whether the sandbox may only read through the grant.
    pub read_only: bool,
    /// Why the grant exists, for the people reading the manifest.
    pub description: Option<String>,
    /// The native host directory backing the grant, when one is needed.
    pub host_path: Option<String>,
}

impl SandboxPathGrant {
    /// A writable grant for an absolute sandbox path.
    ///
    /// Returns [`ErrorCode::InvalidManifestPath`] for a relative path, a Windows path, or the
    /// filesystem root: granting `/` would hand the sandbox the whole machine, which is never
    /// what a grant is for.
    pub fn new(path: impl AsRef<str>) -> SandboxResult<Self> {
        let path = validate_grant_path("path", path.as_ref())?;
        Ok(SandboxPathGrant {
            path,
            read_only: false,
            description: None,
            host_path: None,
        })
    }

    /// Only allow reads through the grant.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Explain why the grant exists.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Back the grant with a native host directory.
    pub fn with_host_path(mut self, host_path: impl AsRef<str>) -> SandboxResult<Self> {
        self.host_path = Some(validate_grant_path("host_path", host_path.as_ref())?);
        Ok(self)
    }

    /// The host directory this grant reads from, which is `path` when none was set.
    pub fn host_source(&self) -> &str {
        self.host_path.as_deref().unwrap_or(&self.path)
    }
}

/// Validate one absolute path on a [`SandboxPathGrant`].
fn validate_grant_path(field: &str, value: &str) -> SandboxResult<String> {
    let invalid = |message: String| SandboxError::new(ErrorCode::InvalidManifestPath, message);
    if value.starts_with("\\\\") || value.starts_with("//") {
        return Err(invalid(format!(
            "sandbox path grant {field} does not support UNC or device paths"
        )));
    }
    if is_windows_absolute(value) {
        return Err(invalid(format!(
            "sandbox path grant {field} must be POSIX absolute"
        )));
    }
    if value.split(['/', '\\']).any(|part| part == "..") {
        return Err(invalid(format!(
            "sandbox path grant {field} must not contain parent segments"
        )));
    }
    if !value.starts_with('/') {
        return Err(invalid(format!(
            "sandbox path grant {field} must be an absolute path"
        )));
    }
    let normalized = normalize_segments(value);
    if normalized == "/" || normalized.is_empty() {
        return Err(invalid(format!(
            "sandbox path grant {field} must not be the filesystem root"
        )));
    }
    Ok(normalized)
}

/// Express `path` relative to `start`, both workspace-relative POSIX paths.
///
/// Mirrors `posixpath.relpath`: shared leading segments are dropped and each remaining segment
/// of `start` becomes a `..`.
fn relative_to(path: &str, start: &str) -> String {
    let path_parts: Vec<&str> = path
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let start_parts: Vec<&str> = start
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let shared = path_parts
        .iter()
        .zip(start_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut parts: Vec<&str> = vec![".."; start_parts.len() - shared];
    parts.extend(&path_parts[shared..]);
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
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

    /// Render a workspace-root-relative path the way the model should see it.
    ///
    /// Mirrors Python's `model_path`. Without a run `cwd` the path is already in the model's
    /// terms; with one, a tool result that says `src/app.py` while the model is working in
    /// `src/` would send it looking in `src/src/`, so the path is re-expressed relative to the
    /// working directory instead.
    pub fn model_path(&self, workspace_relative_path: &str) -> SandboxResult<String> {
        let posix = to_posix(workspace_relative_path);
        if posix.starts_with('/') || is_windows_absolute(&posix) {
            return Err(SandboxError::new(
                ErrorCode::InvalidManifestPath,
                "a workspace-relative display path must not be absolute",
            ));
        }
        let normalized = normalize_segments(&posix);
        let Some(cwd) = &self.cwd else {
            return Ok(normalized);
        };
        Ok(relative_to(&normalized, cwd))
    }

    /// Render a path to a session-owned workspace resource for the model's instructions.
    ///
    /// Mirrors Python's `model_resource_path`. Without a run `cwd` the workspace-relative form
    /// is kept; with one the path becomes absolute, because a resource named relative to the
    /// workspace root stops resolving as soon as a shell command changes directory.
    pub fn model_resource_path(
        &self,
        workspace_root: &str,
        workspace_relative_path: &str,
    ) -> SandboxResult<String> {
        let invalid =
            |message: &str| SandboxError::new(ErrorCode::InvalidManifestPath, message.to_string());
        if workspace_relative_path.contains('\\') {
            return Err(invalid(
                "session resource paths must use POSIX path separators",
            ));
        }
        if workspace_relative_path.starts_with('/') {
            return Err(invalid("session resource paths must be workspace-relative"));
        }
        let normalized = normalize_segments(workspace_relative_path);
        if normalized.split('/').any(|part| part == "..") {
            return Err(invalid("session resource paths must be workspace-relative"));
        }
        if normalized.is_empty() || normalized == "." {
            return Err(invalid("session resource paths must be non-empty"));
        }
        let Some(_) = &self.cwd else {
            return Ok(normalized);
        };
        let root = to_posix(workspace_root);
        if !root.starts_with('/') {
            return Err(invalid("sandbox workspace root must be POSIX absolute"));
        }
        Ok(format!("{}/{normalized}", root.trim_end_matches('/')))
    }

    /// Render a tool-result path, keeping the shape the model used in its request.
    ///
    /// Mirrors Python's `display_path`: a model that asked with an absolute path gets an
    /// absolute answer, so the result is recognisably about the path it named.
    pub fn display_path(
        &self,
        original_path: &str,
        workspace_relative_path: &str,
    ) -> SandboxResult<String> {
        let original = to_posix(original_path);
        if self.cwd.is_none() || original.starts_with('/') || is_windows_absolute(&original) {
            return Ok(normalize_segments(&to_posix(workspace_relative_path)));
        }
        self.model_path(workspace_relative_path)
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
    fn a_path_grant_refuses_to_widen_access_to_everything() {
        let grant = SandboxPathGrant::new("/srv/data/")
            .unwrap()
            .read_only(true)
            .with_description("shared corpus");
        assert_eq!(grant.path, "/srv/data");
        assert!(grant.read_only);
        assert_eq!(grant.host_source(), "/srv/data");

        for refused in [
            "/",
            "relative/path",
            "/srv/../etc",
            "//server/share",
            "C:\\data",
        ] {
            assert!(
                SandboxPathGrant::new(refused).is_err(),
                "accepted `{refused}`"
            );
        }
    }

    #[test]
    fn a_host_path_backs_the_grant() {
        let grant = SandboxPathGrant::new("/mnt/data")
            .unwrap()
            .with_host_path("/var/lib/corpus")
            .unwrap();
        assert_eq!(grant.host_source(), "/var/lib/corpus");
        assert!(
            SandboxPathGrant::new("/mnt/data")
                .unwrap()
                .with_host_path("relative")
                .is_err()
        );
    }

    #[test]
    fn renders_paths_relative_to_the_run_cwd() {
        let scope = SandboxWorkspaceScope::from_cwd(Some("src")).unwrap();
        assert_eq!(scope.model_path("src/app.py").unwrap(), "app.py");
        assert_eq!(scope.model_path("README.md").unwrap(), "../README.md");
        assert_eq!(scope.model_path("src").unwrap(), ".");

        // Without a cwd the workspace-relative form is already the model's view.
        let root = SandboxWorkspaceScope::root();
        assert_eq!(root.model_path("src/app.py").unwrap(), "src/app.py");
        assert!(root.model_path("/abs").is_err());
    }

    #[test]
    fn keeps_session_resources_addressable_under_a_cwd() {
        let scope = SandboxWorkspaceScope::from_cwd(Some("src")).unwrap();
        assert_eq!(
            scope
                .model_resource_path("/workspace", "skills/list.md")
                .unwrap(),
            "/workspace/skills/list.md"
        );
        let root = SandboxWorkspaceScope::root();
        assert_eq!(
            root.model_resource_path("/workspace", "skills/list.md")
                .unwrap(),
            "skills/list.md"
        );
        assert!(root.model_resource_path("/workspace", "../escape").is_err());
        assert!(root.model_resource_path("/workspace", "").is_err());
        assert!(root.model_resource_path("/workspace", "/abs").is_err());
    }

    #[test]
    fn a_tool_result_keeps_the_shape_the_model_asked_with() {
        let scope = SandboxWorkspaceScope::from_cwd(Some("src")).unwrap();
        // The model asked relatively, so it gets a cwd-relative answer.
        assert_eq!(
            scope.display_path("app.py", "src/app.py").unwrap(),
            "app.py"
        );
        // It asked absolutely, so the answer stays workspace-relative as before.
        assert_eq!(
            scope
                .display_path("/workspace/src/app.py", "src/app.py")
                .unwrap(),
            "src/app.py"
        );
    }

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
