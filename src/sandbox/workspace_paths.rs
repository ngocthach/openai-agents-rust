//! POSIX path handling for sandbox workspaces. Mirrors `agents.sandbox.workspace_paths`.
//!
//! Every path the model or a manifest entry provides is normalized against the workspace root
//! before it reaches a session. Absolute paths outside the root, Windows-style absolute paths
//! and `..` segments are rejected.

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxOp, SandboxResult};

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

/// Resolve `.`, `..` and redundant separators lexically, the way `posixpath.normpath` does.
///
/// A `..` that cannot be popped is kept, so a relative path may still begin with `..` and the
/// caller can recognise an escape. Nothing on the filesystem is consulted: a `..` that crosses a
/// symlink is resolved textually here, exactly as Python does, and symlink containment is
/// checked separately by the backends that have a real host directory to look at.
pub fn normpath(path: &str) -> String {
    let posix = to_posix(path);
    let absolute = posix.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for segment in posix.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                match parts.last() {
                    // An unpoppable `..` is kept so a relative path can still escape.
                    Some(&"..") | None if !absolute => parts.push(".."),
                    // Above the root there is nowhere to go, so `/..` stays `/`.
                    None => {}
                    Some(_) => {
                        parts.pop();
                    }
                }
            }
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

/// The error Python raises for a manifest path that is absolute or escapes the workspace root.
///
/// `rel` is the caller's own path, not the normalized one: the message has to name the path the
/// caller actually wrote, or it reads as being about a path they never mentioned.
fn invalid_manifest_path(rel: &str) -> SandboxError {
    let rel = to_posix(rel);
    let absolute = rel.starts_with('/') || is_windows_absolute(&rel);
    let (reason, message) = if absolute {
        ("absolute", format!("manifest path must be relative: {rel}"))
    } else {
        (
            "escape_root",
            format!("manifest path must not escape root: {rel}"),
        )
    };
    SandboxError::new(ErrorCode::InvalidManifestPath, message)
        .with_op(SandboxOp::Materialize)
        .with_context("rel", rel)
        .with_context("reason", reason)
}

/// Resolve `path` against the real filesystem as far as it exists, keeping the rest lexically.
///
/// Mirrors Python's `Path.resolve(strict=False)`: the longest existing prefix is canonicalized —
/// which is what makes a symlinked parent visible — and the segments that do not exist yet are
/// appended unchanged, so a path can be validated before the file it names is created.
fn resolve_existing_prefix(path: &str) -> String {
    let posix = normpath(path);
    let mut existing = std::path::PathBuf::from(&posix);
    let mut trailing: Vec<String> = Vec::new();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&existing) {
            let mut resolved = canonical.to_string_lossy().into_owned();
            for segment in trailing.iter().rev() {
                resolved = format!("{}/{segment}", resolved.trim_end_matches('/'));
            }
            return normpath(&resolved);
        }
        let Some(name) = existing
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        else {
            return posix;
        };
        trailing.push(name);
        if !existing.pop() {
            return posix;
        }
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
        return Err(invalid_manifest_path(path));
    }
    let root = normpath(root);
    let posix = to_posix(path);
    // Mirrors Python's `_absolute_posix_path`: anchor first, then collapse `..` lexically, so a
    // `..` that stays inside the workspace (`pkg/../secret.txt`) resolves instead of being
    // refused, and only a path that really leaves the root is rejected.
    let resolved = if posix.starts_with('/') {
        normpath(&posix)
    } else {
        normpath(&format!("{}/{posix}", root.trim_end_matches('/')))
    };
    if !is_within(&root, &resolved) {
        return Err(invalid_manifest_path(path));
    }
    Ok(resolved)
}

/// Validate and normalize a run working directory relative to the workspace root.
pub fn normalize_sandbox_cwd(cwd: &str) -> SandboxResult<String> {
    let invalid =
        |message: &str| SandboxError::new(ErrorCode::SandboxConfigInvalid, message.to_string());
    // The separator check comes first because a backslash path is not a cwd that happens to be
    // malformed, it is a cwd written in the wrong path language.
    if cwd.contains('\\') {
        return Err(invalid("sandbox.cwd must use POSIX path separators"));
    }
    if cwd.trim().is_empty() {
        return Err(invalid("sandbox.cwd must be non-empty"));
    }
    if is_windows_absolute(cwd) {
        return Err(invalid("sandbox.cwd must be workspace-relative"));
    }
    if cwd.starts_with('/') {
        return Err(invalid("sandbox.cwd must be workspace-relative"));
    }
    // Checked before normalizing, so `tasks/../a` is refused rather than quietly becoming `a`.
    if cwd.split('/').any(|s| s == "..") {
        return Err(invalid("sandbox.cwd must not contain parent segments"));
    }
    Ok(normpath(cwd))
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

/// The validated native host directory behind a grant. Mirrors Python's
/// `sandbox_path_grant_host_path`.
///
/// Symlinks are followed before the check, because a grant pointed at a link to `/` would
/// otherwise hand the sandbox the whole filesystem under an innocuous-looking path. The resolved
/// path is returned, so a caller that later re-points the link still works from what was checked.
pub fn sandbox_path_grant_host_path(grant: &SandboxPathGrant) -> SandboxResult<String> {
    let raw = grant.host_source();
    if grant.host_path.is_some() && !raw.starts_with('/') {
        return Err(SandboxError::new(
            ErrorCode::InvalidManifestPath,
            format!("sandbox path grant host_path must be absolute on the current host: {raw}"),
        ));
    }
    let resolved = resolve_existing_prefix(raw);
    if resolved == "/" {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "sandbox path grant path must not resolve to filesystem root",
        ));
    }
    Ok(resolved)
}

/// Validate one absolute path on a [`SandboxPathGrant`].
fn validate_grant_path(field: &str, value: &str) -> SandboxResult<String> {
    let invalid = |message: String| SandboxError::new(ErrorCode::InvalidManifestPath, message);
    let is_host = field == "host_path";
    if is_host && (value.starts_with("\\\\") || value.starts_with("//")) {
        // A `host_path` is handed to the platform as written, so anything starting `//` or `\\`
        // is a UNC or device path, which no sandbox grant can stand for.
        return Err(invalid(format!(
            "sandbox path grant {field} does not support UNC or device paths"
        )));
    }
    if !is_host && value.starts_with("\\\\") {
        // A backslash UNC path is a Windows path; the sandbox-side `path` is always POSIX.
        return Err(invalid(format!(
            "sandbox path grant {field} must be POSIX absolute"
        )));
    }
    if is_host && value.split(['/', '\\']).any(|part| part == "..") {
        // A host path is handed to the platform as written, so a parent segment in one is a
        // request the SDK cannot vouch for. The sandbox-side `path` is normalized instead.
        return Err(invalid(format!(
            "sandbox path grant {field} must not contain parent segments"
        )));
    }
    if is_windows_absolute(value) {
        let bytes = value.as_bytes();
        let drive_rooted = bytes.len() >= 3 && matches!(bytes[2], b'/' | b'\\');
        if !is_host || !drive_rooted {
            return Err(invalid(format!(
                "sandbox path grant {field} must be POSIX absolute"
            )));
        }
        // A drive-qualified host path is a legitimate native path on a Windows host. It is kept
        // in Windows spelling, as `PureWindowsPath` would render it, rather than being forced
        // through POSIX normalization.
        let drive = &value[..2];
        let rest: Vec<&str> = value[2..]
            .split(['/', '\\'])
            .filter(|part| !part.is_empty() && *part != ".")
            .collect();
        if rest.is_empty() {
            return Err(invalid(format!(
                "sandbox path grant {field} must not be filesystem root"
            )));
        }
        return Ok(format!("{drive}\\{}", rest.join("\\")));
    }
    if !value.starts_with('/') {
        return Err(invalid(if is_host {
            format!("sandbox path grant {field} must be an absolute host path")
        } else {
            format!("sandbox path grant {field} must be POSIX absolute")
        }));
    }
    let mut normalized = normpath(value);
    if normalized == "/" || normalized.is_empty() {
        return Err(invalid(format!(
            "sandbox path grant {field} must not be filesystem root"
        )));
    }
    // `posixpath.normpath` keeps exactly two leading slashes (POSIX leaves their meaning to the
    // implementation) and collapses three or more, so `//server/share` stays as written.
    if value.starts_with("//") && !value.starts_with("///") {
        normalized.insert(0, '/');
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
                "workspace-relative display paths must not be absolute",
            ));
        }
        let normalized = normpath(&posix);
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
        if is_windows_absolute(workspace_relative_path) {
            return Err(invalid("session resource paths must be workspace-relative"));
        }
        if workspace_relative_path.starts_with('/') {
            return Err(invalid("session resource paths must be workspace-relative"));
        }
        if workspace_relative_path.split('/').any(|part| part == "..") {
            return Err(invalid("session resource paths must be workspace-relative"));
        }
        let normalized = normpath(workspace_relative_path);
        if normalized.is_empty() || normalized == "." {
            return Err(invalid("session resource paths must be non-empty"));
        }
        let Some(_) = &self.cwd else {
            return Ok(normalized);
        };
        // A drive-qualified root is a legitimate workspace root that simply is not written in
        // POSIX; it is accepted and re-spelled, while a bare backslash path is not.
        let root = if is_windows_absolute(workspace_root) {
            to_posix(workspace_root)
        } else {
            if workspace_root.contains('\\') {
                return Err(invalid("sandbox workspace root must be POSIX absolute"));
            }
            let root = workspace_root.to_string();
            if !root.starts_with('/') {
                return Err(invalid("sandbox workspace root must be POSIX absolute"));
            }
            root
        };
        Ok(normpath(&format!(
            "{}/{normalized}",
            root.trim_end_matches('/')
        )))
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
        // The model's own spelling is preserved, backslashes included: anchoring decides *where*
        // a relative path starts, and rewriting separators here would silently change which file
        // a tool was asked about. The session's policy is what normalizes and validates it.
        // A leading backslash is a root once separators are normalized, so `\workspace\x` is as
        // absolute as `/workspace/x`.
        if path.starts_with('/') || path.starts_with('\\') || is_windows_absolute(path) {
            return path.to_string();
        }
        match &self.cwd {
            None => path.to_string(),
            Some(cwd) => {
                if path.is_empty() || path == "." {
                    cwd.clone()
                } else {
                    format!("{cwd}/{path}")
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

        for refused in ["/", "//", "relative/path", "C:\\data", "\\\\server\\share"] {
            assert!(
                SandboxPathGrant::new(refused).is_err(),
                "accepted `{refused}`"
            );
        }
        // `//server/share` is an ordinary absolute POSIX path here, as `posixpath.normpath` keeps it.
        assert_eq!(
            SandboxPathGrant::new("//server/share").unwrap().path,
            "//server/share"
        );
        // A parent segment in the sandbox-side path is collapsed, as Python's validator does.
        assert_eq!(SandboxPathGrant::new("/srv/../etc").unwrap().path, "/etc");
        // A host path is used as written, so a parent segment in one is still refused.
        assert!(
            SandboxPathGrant::new("/mnt/data")
                .unwrap()
                .with_host_path("/srv/../secret")
                .is_err()
        );
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

/// Validate and format paths interpreted relative to a sandbox workspace root. Mirrors Python's
/// `WorkspacePathPolicy`.
///
/// The policy is the single place that decides whether a path is inside the workspace, inside an
/// extra grant, or outside everything — and, for a write, whether the grant it landed in allows
/// one. Keeping that in one object is what lets a backend answer "may this be written?" without
/// re-deriving the rules, and what makes the longest-grant-wins precedence consistent between the
/// file APIs, the editor and the archive paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspacePathPolicy {
    root: String,
    extra_path_grants: Vec<SandboxPathGrant>,
}

impl WorkspacePathPolicy {
    /// A policy for `root` with no extra grants.
    pub fn new(root: impl AsRef<str>) -> SandboxResult<Self> {
        WorkspacePathPolicy::with_grants(root, Vec::new())
    }

    /// A policy for `root` honouring `extra_path_grants`.
    pub fn with_grants(
        root: impl AsRef<str>,
        extra_path_grants: Vec<SandboxPathGrant>,
    ) -> SandboxResult<Self> {
        let root = normalize_segments(&to_posix(root.as_ref()));
        if !root.starts_with('/') {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "sandbox workspace root must be absolute",
            ));
        }
        Ok(WorkspacePathPolicy {
            root,
            extra_path_grants,
        })
    }

    /// The policy for a manifest's root and grants.
    pub fn for_manifest(manifest: &crate::sandbox::manifest::Manifest) -> SandboxResult<Self> {
        WorkspacePathPolicy::with_grants(manifest.root(), manifest.extra_path_grants().to_vec())
    }

    /// The workspace root, normalized.
    pub fn sandbox_root(&self) -> &str {
        &self.root
    }

    /// Whether the root currently exists as a directory on this host.
    ///
    /// Mirrors Python's `root_is_existing_host_path`: only a backend whose workspace really is a
    /// local directory may resolve symlinks through it.
    pub fn root_is_existing_host_path(&self) -> bool {
        std::path::Path::new(&self.root).is_dir()
    }

    /// An absolute workspace path, with no grant allowed.
    ///
    /// Mirrors Python's `absolute_workspace_path`.
    pub fn absolute_workspace_path(&self, path: &str) -> SandboxResult<String> {
        resolve_workspace_path(&self.root, path)
    }

    /// The path relative to the workspace root, or `.` for the root itself.
    pub fn relative_path(&self, path: &str) -> SandboxResult<String> {
        let absolute = self.absolute_workspace_path(path)?;
        let relative = absolute
            .strip_prefix(self.root.trim_end_matches('/'))
            .unwrap_or(&absolute)
            .trim_start_matches('/')
            .to_string();
        Ok(if relative.is_empty() {
            ".".to_string()
        } else {
            relative
        })
    }

    /// A validated absolute sandbox path, allowing the extra grants.
    ///
    /// Mirrors Python's `normalize_sandbox_path`. `for_write` refuses a path whose effective
    /// grant is read-only.
    pub fn normalize_sandbox_path(&self, path: &str, for_write: bool) -> SandboxResult<String> {
        let (resolved, grant) = self.resolve_with_grant(path)?;
        if for_write {
            self.raise_if_read_only(&resolved, grant)?;
        }
        Ok(resolved)
    }

    /// A validated absolute host path, following symlinks on the real filesystem.
    ///
    /// Mirrors Python's `normalize_path(resolve_symlinks=True)`. Only a backend whose workspace
    /// really is a local directory may use this: a lexical check cannot see that `outside-link`
    /// in the workspace points at `/etc`, so a path that reaches outside the root or any grant
    /// through a symlink is refused here rather than at the point it is opened.
    ///
    /// `for_write` additionally refuses a path whose effective grant is read-only, after
    /// resolution — so an alias of a read-only grant is still read-only.
    pub fn normalize_host_path(&self, path: &str, for_write: bool) -> SandboxResult<String> {
        if is_windows_absolute(path) {
            return Err(invalid_manifest_path(path));
        }
        let workspace_root = resolve_existing_prefix(&self.root);
        let posix = to_posix(path);
        let absolute = if posix.starts_with('/') {
            normpath(&posix)
        } else {
            // A relative path is anchored and containment-checked lexically first, exactly as
            // Python does, so `../x` is refused before the filesystem is consulted at all.
            self.absolute_workspace_path(&posix)?
        };
        let resolved = resolve_existing_prefix(&absolute);
        let grant = if is_within(&workspace_root, &resolved) {
            None
        } else {
            match self.matching_resolved_grant(&resolved)? {
                Some(grant) => Some(grant),
                None => return Err(invalid_manifest_path(path)),
            }
        };
        if for_write {
            self.raise_if_read_only(&resolved, grant)?;
        }
        Ok(resolved)
    }

    /// The effective grant for an already-resolved host path, matching on resolved grant roots.
    fn matching_resolved_grant(&self, path: &str) -> SandboxResult<Option<&SandboxPathGrant>> {
        let mut best: Option<(&SandboxPathGrant, usize)> = None;
        for grant in &self.extra_path_grants {
            // A drive-qualified host path names a directory on a Windows host; it cannot be
            // resolved, or match anything, on this one.
            if !grant.host_source().starts_with('/') {
                continue;
            }
            let grant_root = resolve_existing_prefix(grant.host_source());
            if grant_root == "/" {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "sandbox path grant path must not resolve to filesystem root",
                ));
            }
            if is_within(&grant_root, path) {
                let depth = grant_root.len();
                if best.is_none_or(|(_, best_depth)| depth > best_depth) {
                    best = Some((grant, depth));
                }
            }
        }
        Ok(best.map(|(grant, _)| grant))
    }

    /// Refuse a recursive removal that would delete a read-only grant inside the tree.
    ///
    /// Mirrors Python's `validate_recursive_remove`: `rm -rf` on an ancestor is a write to every
    /// grant beneath it, so a read-only grant nested in the tree has to veto the whole removal.
    pub fn validate_recursive_remove(&self, path: &str) -> SandboxResult<()> {
        let root = self.normalize_sandbox_path(path, true)?;
        for grant in &self.extra_path_grants {
            let grant_root = normalize_segments(&to_posix(&grant.path));
            if grant.read_only && is_within(&root, &grant_root) {
                // Go back through the effective-grant matching so workspace precedence and
                // overriding grants are honoured exactly once.
                self.normalize_sandbox_path(&grant_root, true)?;
            }
        }
        Ok(())
    }

    /// The normalized grant roots and their access modes.
    ///
    /// Mirrors Python's `extra_path_grant_rules`, which feeds the in-sandbox realpath helper.
    pub fn extra_path_grant_rules(&self) -> SandboxResult<Vec<(String, bool)>> {
        let mut rules = Vec::new();
        for grant in &self.extra_path_grants {
            if is_windows_absolute(&grant.path) {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "sandbox path grant path must be POSIX absolute",
                ));
            }
            let root = normpath(&grant.path);
            if root == "/" {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "sandbox path grant path must not be filesystem root",
                ));
            }
            rules.push((root, grant.read_only));
        }
        Ok(rules)
    }

    /// The effective grant for an already-normalized path, longest root first.
    pub fn matching_grant(&self, path: &str) -> Option<&SandboxPathGrant> {
        // The most specific grant wins and the first declared wins a tie, as Python's `max`
        // does; `max_by_key` would keep the last of equals.
        let mut best: Option<(&SandboxPathGrant, usize)> = None;
        for grant in &self.extra_path_grants {
            let root = normpath(&grant.path);
            if is_within(&root, path) && best.is_none_or(|(_, depth)| root.len() > depth) {
                best = Some((grant, root.len()));
            }
        }
        best.map(|(grant, _)| grant)
    }

    fn resolve_with_grant(&self, path: &str) -> SandboxResult<(String, Option<&SandboxPathGrant>)> {
        match resolve_workspace_path(&self.root, path) {
            // Inside the workspace: the root always wins over a grant that also covers it.
            Ok(resolved) => Ok((resolved, None)),
            Err(error) => {
                let posix = to_posix(path);
                if !posix.starts_with('/') || is_windows_absolute(path) {
                    return Err(error);
                }
                let normalized = normalize_segments(&posix);
                match self.matching_grant(&normalized) {
                    Some(grant) => Ok((normalized, Some(grant))),
                    None => Err(error),
                }
            }
        }
    }

    fn raise_if_read_only(
        &self,
        path: &str,
        grant: Option<&SandboxPathGrant>,
    ) -> SandboxResult<()> {
        let Some(grant) = grant else {
            return Ok(());
        };
        if !grant.read_only {
            return Ok(());
        }
        Err(SandboxError::new(
            ErrorCode::WorkspaceArchiveWriteError,
            format!("failed to write archive for path: {path}"),
        )
        .with_context("path", path.to_string())
        .with_context("reason", "read_only_extra_path_grant")
        .with_context("grant_path", grant.path.clone()))
    }
}
