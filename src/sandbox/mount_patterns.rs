//! The mount tools the built-in remote storage mounts drive. Mirrors
//! `agents.sandbox.entries.mounts.patterns`.
//!
//! Each pattern turns a [`MountPatternConfig`] into one mount command inside the sandbox:
//!
//! - [`FuseMountPattern`] runs `blobfuse2` for Azure Blob containers.
//! - [`MountpointMountPattern`] runs `mount-s3` for S3 and S3-compatible buckets.
//! - [`S3FilesMountPattern`] runs `mount -t s3files` for Amazon S3 Files file systems.
//!
//! [`RcloneMountPattern`](crate::sandbox::mounts::RcloneMountPattern), the fourth built-in
//! pattern, lives with the mount core because the provider backends drive it directly.
//!
//! Credentials reach a pattern as [`MountPatternConfig`] secrets, never as command arguments:
//! they are written to an owner-only file inside the workspace and sourced by the mount command,
//! so they never appear in the host's process table. Every workspace archive excludes the
//! directories those files live in.

use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mounts::{ContainerPrivileges, MountPattern, MountPatternConfig};
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession, shell_quote};

// ---------------------------------------------------------------------------------------------
// Workspace scratch directories
// ---------------------------------------------------------------------------------------------

/// Directory, relative to the workspace root, holding generated `blobfuse2` configurations.
///
/// Mirrors Python's `.sandbox-blobfuse-config`.
pub const BLOBFUSE_CONFIG_DIR: &str = ".sandbox-blobfuse-config";

/// Directory, relative to the workspace root, holding the `blobfuse2` block or file cache.
///
/// Mirrors Python's `.sandbox-blobfuse-cache`. The cache can grow to tens of gigabytes, so it is
/// excluded from workspace archives along with the configuration.
pub const BLOBFUSE_CACHE_DIR: &str = ".sandbox-blobfuse-cache";

/// Directory, relative to the workspace root, holding generated `mount-s3` environment files.
///
/// Mirrors Python's `.sandbox-mountpoint-env`.
pub const MOUNTPOINT_ENV_DIR: &str = ".sandbox-mountpoint-env";

/// Every workspace directory a built-in mount pattern writes credentials or cache data to.
///
/// [`workspace_archive_excludes`](crate::sandbox::session::workspace_archive_excludes) adds each
/// of these, so a snapshot can never capture a staged credential or a multi-gigabyte blob cache.
pub const MOUNT_SCRATCH_DIRS: &[&str] =
    &[BLOBFUSE_CONFIG_DIR, BLOBFUSE_CACHE_DIR, MOUNTPOINT_ENV_DIR];

// ---------------------------------------------------------------------------------------------
// Shared configuration keys
// ---------------------------------------------------------------------------------------------

/// Option naming the kind of mount a configuration came from, such as `s3_mount`.
///
/// Mirrors the `mount_type` field Python's pattern configs carry: a few mount commands differ by
/// the storage the mount describes rather than by the pattern driving them.
pub const OPTION_MOUNT_TYPE: &str = "mount_type";

/// Option set to `true` when the mount is read-only.
pub const OPTION_READ_ONLY: &str = "read_only";

/// Mount type of an [`S3Mount`](crate::sandbox::mount_entries::S3Mount).
pub const MOUNT_TYPE_S3: &str = "s3_mount";
/// Mount type of a [`GcsMount`](crate::sandbox::mount_entries::GcsMount).
pub const MOUNT_TYPE_GCS: &str = "gcs_mount";
/// Mount type of an [`R2Mount`](crate::sandbox::mount_entries::R2Mount).
pub const MOUNT_TYPE_R2: &str = "r2_mount";
/// Mount type of an [`AzureBlobMount`](crate::sandbox::mount_entries::AzureBlobMount).
pub const MOUNT_TYPE_AZURE_BLOB: &str = "azure_blob_mount";
/// Mount type of a [`BoxMount`](crate::sandbox::mount_entries::BoxMount).
pub const MOUNT_TYPE_BOX: &str = "box_mount";
/// Mount type of an [`S3FilesMount`](crate::sandbox::mount_entries::S3FilesMount).
pub const MOUNT_TYPE_S3_FILES: &str = "s3_files_mount";

// Azure Blob / blobfuse2.

/// Option holding the Azure storage account name.
pub const FUSE_OPTION_ACCOUNT: &str = "account";
/// Option holding the Azure blob container name.
pub const FUSE_OPTION_CONTAINER: &str = "container";
/// Option holding an explicit blob service endpoint.
pub const FUSE_OPTION_ENDPOINT: &str = "endpoint";
/// Option holding the managed-identity client id used when no account key is supplied.
pub const FUSE_OPTION_IDENTITY_CLIENT_ID: &str = "identity_client_id";
/// Secret holding the Azure storage account key.
pub const FUSE_SECRET_ACCOUNT_KEY: &str = "account_key";

// S3 / mount-s3.

/// Option holding the bucket name.
pub const MOUNTPOINT_OPTION_BUCKET: &str = "bucket";
/// Option holding the object prefix exposed at the mount path.
pub const MOUNTPOINT_OPTION_PREFIX: &str = "prefix";
/// Option holding the bucket region.
pub const MOUNTPOINT_OPTION_REGION: &str = "region";
/// Option holding a plain S3-compatible endpoint URL.
pub const MOUNTPOINT_OPTION_ENDPOINT_URL: &str = "endpoint_url";
/// Secret holding an endpoint URL that carries inline authority, such as a signed query string.
pub const MOUNTPOINT_SECRET_ENDPOINT_URL: &str = "endpoint_url";
/// Secret holding the access key id.
pub const MOUNTPOINT_SECRET_ACCESS_KEY_ID: &str = "access_key_id";
/// Secret holding the secret access key.
pub const MOUNTPOINT_SECRET_SECRET_ACCESS_KEY: &str = "secret_access_key";
/// Secret holding the STS session token.
pub const MOUNTPOINT_SECRET_SESSION_TOKEN: &str = "session_token";

// S3 Files.

/// Option holding the S3 Files file system id.
pub const S3FILES_OPTION_FILE_SYSTEM_ID: &str = "file_system_id";
/// Option holding the subdirectory of the file system exposed at the mount path.
pub const S3FILES_OPTION_SUBPATH: &str = "subpath";
/// Option holding the mount target IP address.
pub const S3FILES_OPTION_MOUNT_TARGET_IP: &str = "mount_target_ip";
/// Option holding the access point id.
pub const S3FILES_OPTION_ACCESS_POINT: &str = "access_point";
/// Option holding the file system region.
pub const S3FILES_OPTION_REGION: &str = "region";
/// Option holding extra `mount -o` settings, rendered as `key[=value]` joined with `,`.
pub const S3FILES_OPTION_EXTRA_OPTIONS: &str = "extra_options";

// ---------------------------------------------------------------------------------------------
// Value checks
// ---------------------------------------------------------------------------------------------

/// Reject a value that would break out of the line it is rendered on.
///
/// Every built-in pattern writes its configuration as a line-oriented format — an rclone INI
/// section, a blobfuse YAML document, a shell export file — so a newline inside a bucket name or
/// an endpoint could append a setting the caller never wrote.
pub fn reject_config_line_break(field: &str, value: &str) -> SandboxResult<()> {
    if value.contains(['\n', '\r']) {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("the mount field `{field}` must not contain a line break"),
        ));
    }
    Ok(())
}

/// Reject a value that would split a comma-separated mount option list.
pub fn reject_option_delimiter(field: &str, value: &str) -> SandboxResult<()> {
    reject_config_line_break(field, value)?;
    if value.contains(',') {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("the mount field `{field}` must not contain a comma"),
        ));
    }
    Ok(())
}

/// Whether a URL carries authority of its own, such as a user info part or a query string.
///
/// Mirrors Python's `_url_contains_inline_authority`. Such a URL is a credential: it is passed
/// to the mount tool through the environment rather than on the command line, and is redacted
/// out of any error the mount produces.
pub fn url_contains_inline_authority(value: &str) -> bool {
    if value.contains('@') {
        return true;
    }
    let split = split_url(value);
    split.username.is_some() || split.password.is_some() || !split.query.is_empty()
}

/// The parts of a URL this module needs, following Python's `urlsplit`.
struct UrlParts<'a> {
    username: Option<&'a str>,
    password: Option<&'a str>,
    query: &'a str,
}

/// Split a URL into its user info and query, as Python's `urlsplit` does.
fn split_url(value: &str) -> UrlParts<'_> {
    let without_fragment = value.split('#').next().unwrap_or("");
    let (before_query, query) = match without_fragment.split_once('?') {
        Some((before, query)) => (before, query),
        None => (without_fragment, ""),
    };
    // `scheme://netloc/path`, or `//netloc/path` when the scheme is left out.
    let after_scheme = match before_query.find("//") {
        Some(index)
            // Only a `//` that opens the authority counts; one inside the path does not.
            if before_query[..index]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.' | ':')) =>
        {
            &before_query[index + 2..]
        }
        _ => return UrlParts { username: None, password: None, query },
    };
    let netloc = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let Some((userinfo, _)) = netloc.rsplit_once('@') else {
        return UrlParts {
            username: None,
            password: None,
            query,
        };
    };
    match userinfo.split_once(':') {
        Some((username, password)) => UrlParts {
            username: Some(username),
            password: Some(password),
            query,
        },
        None => UrlParts {
            username: Some(userinfo),
            password: None,
            query,
        },
    }
}

/// Every substring of an authority-carrying URL that must be redacted out of command output.
///
/// Mirrors Python's `_inline_url_authority_values`: the URL itself, its user info and its query,
/// each both percent-encoded and decoded, because a mount tool may echo either form.
pub fn inline_url_authority_values(value: &str) -> Vec<String> {
    if !url_contains_inline_authority(value) {
        return Vec::new();
    }
    let mut values = vec![value.to_string()];
    let split = split_url(value);
    let mut push = |raw: &str| {
        if raw.is_empty() {
            return;
        }
        values.push(raw.to_string());
        let decoded = percent_decode(raw);
        if decoded != raw {
            values.push(decoded);
        }
    };
    if let Some(username) = split.username {
        push(username);
    }
    if let Some(password) = split.password {
        push(password);
    }
    push(split.query);
    for pair in split.query.split('&') {
        let raw = pair.split_once('=').map_or("", |(_, raw)| raw);
        let decoded = percent_decode(&raw.replace('+', " "));
        push(&decoded);
    }
    let mut seen = std::collections::HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
    values
}

/// Decode the `%XX` escapes of a URL component.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
            if let Some(byte) = hex.and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Replace every secret value, and its shell-quoted form, with `REDACTED`.
///
/// Mirrors Python's `_redact_sensitive_values`. A mount tool that fails after reading its
/// environment often prints the credential it tried, so command text and standard error are both
/// put through this before they reach an error message.
pub fn redact_sensitive_values(text: &str, sensitive: &[String]) -> String {
    let mut redacted = text.to_string();
    for value in sensitive {
        if value.is_empty() {
            continue;
        }
        redacted = redacted.replace(value.as_str(), "REDACTED");
        let quoted = shell_quote(value);
        if quoted != *value {
            redacted = redacted.replace(quoted.as_str(), "REDACTED");
        }
    }
    redacted
}

/// A filesystem-safe, stable identifier for the session, for naming scratch directories.
///
/// Python names them after the session's UUID hex. A Rust session id is an opaque string chosen
/// by the backend, so it is hashed instead: the name stays stable for the session and distinct
/// between sessions, and can never contain a path separator.
pub fn session_scratch_id(session: &dyn SandboxSession) -> String {
    let digest = <sha2::Sha256 as sha2::Digest>::digest(session.id().as_bytes());
    digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Read a config option, treating an empty value as absent.
fn option<'a>(config: &'a MountPatternConfig, key: &str) -> Option<&'a str> {
    config
        .options
        .get(key)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

/// Read a config secret, treating an empty value as absent.
fn secret<'a>(config: &'a MountPatternConfig, key: &str) -> Option<&'a str> {
    config.secret(key).filter(|value| !value.is_empty())
}

/// Whether a boolean option is set.
fn flag(config: &MountPatternConfig, key: &str) -> bool {
    option(config, key) == Some("true")
}

/// A required config option.
fn require<'a>(config: &'a MountPatternConfig, key: &str) -> SandboxResult<&'a str> {
    option(config, key).ok_or_else(|| {
        SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("a mount configuration is missing the `{key}` option"),
        )
    })
}

/// Write a credential-bearing file and restrict it to its owner.
///
/// Mirrors Python's `_write_sensitive_config_file`. The file is removed again when `chmod` fails,
/// so a credential is never left world-readable in the workspace.
async fn write_sensitive_file(
    session: &dyn SandboxSession,
    path: &str,
    payload: &[u8],
) -> SandboxResult<()> {
    session.write_file(path, payload).await?;
    let chmod = session
        .exec(ExecRequest::argv(["chmod", "0600", "--", path]))
        .await?;
    if !chmod.ok() {
        let _ = session.rm(path, false).await;
        return Err(SandboxError::new(
            ErrorCode::MountFailed,
            format!("failed to restrict the permissions of the mount configuration at `{path}`"),
        ));
    }
    Ok(())
}

/// Render credential environment variables as a sourceable shell file.
///
/// Mirrors Python's `_render_shell_exports`.
fn render_shell_exports(env: &[(String, String)]) -> String {
    let mut text = String::new();
    for (name, value) in env {
        text.push_str(&format!("export {name}={}\n", shell_quote(value)));
    }
    text
}

/// Read a file the mount command may have written, returning nothing when it did not.
async fn read_text_if_present(session: &dyn SandboxSession, path: &str) -> String {
    match session.read_file(path).await {
        Ok(raw) => String::from_utf8_lossy(&raw).into_owned(),
        Err(_) => String::new(),
    }
}

/// Whether `candidate` is `root` or lives inside it.
fn is_inside(candidate: &str, root: &str) -> bool {
    let root = root.trim_end_matches('/');
    candidate == root || candidate.starts_with(&format!("{root}/"))
}

/// The best-effort unmount every FUSE-backed pattern runs on teardown.
async fn fuse_unmount(session: &dyn SandboxSession, mount_path: &str) -> SandboxResult<()> {
    let quoted = shell_quote(mount_path);
    let command =
        format!("fusermount3 -u -- {quoted} 2>/dev/null || umount -- {quoted} 2>/dev/null || true");
    session
        .exec(
            ExecRequest::argv(["sh", "-lc", command.as_str()])
                .with_shell(ExecShell::None)
                .with_timeout(Some(Duration::from_secs(60))),
        )
        .await?;
    Ok(())
}

/// Fail when the sandbox image does not ship the mount tool.
async fn require_tool(session: &dyn SandboxSession, tool: &str, probe: &str) -> SandboxResult<()> {
    let present = session
        .exec(ExecRequest::argv(["sh", "-lc", probe]).with_shell(ExecShell::None))
        .await?;
    if !present.ok() {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            format!("{tool} is not installed in the sandbox image"),
        ));
    }
    Ok(())
}

/// Join an argument vector into one POSIX shell command.
fn shell_command(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------------------------
// Azure Blob: blobfuse2
// ---------------------------------------------------------------------------------------------

/// Which blobfuse cache component the mount uses.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlobfuseCacheType {
    /// Stream blocks through a bounded memory and disk cache.
    #[default]
    BlockCache,
    /// Cache whole files on disk.
    FileCache,
}

impl BlobfuseCacheType {
    /// The blobfuse component name.
    pub fn as_str(self) -> &'static str {
        match self {
            BlobfuseCacheType::BlockCache => "block_cache",
            BlobfuseCacheType::FileCache => "file_cache",
        }
    }

    /// The cache size blobfuse defaults to for this component, in MB.
    fn default_cache_size_mb(self) -> u64 {
        match self {
            BlobfuseCacheType::BlockCache => 50_000,
            BlobfuseCacheType::FileCache => 4_096,
        }
    }
}

/// Mounts an Azure Blob container with `blobfuse2`. Mirrors
/// `agents.sandbox.entries.mounts.patterns.FuseMountPattern`.
///
/// The generated configuration carries the storage account key, so it is written to a `0600`
/// file under [`BLOBFUSE_CONFIG_DIR`] and the cache lands under [`BLOBFUSE_CACHE_DIR`]; both are
/// excluded from every workspace archive.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FuseMountPattern {
    /// Whether users other than the mounting one may read the mount.
    pub allow_other: bool,
    /// Where blobfuse sends its log.
    pub log_type: String,
    /// How verbose that log is.
    pub log_level: String,
    /// Which cache component to use.
    pub cache_type: BlobfuseCacheType,
    /// Workspace-relative cache directory; `None` derives one per session, account and container.
    pub cache_path: Option<String>,
    /// Cache size in MB; `None` or zero uses the default for the cache type.
    pub cache_size_mb: Option<u64>,
    /// Block size of the block cache, in MB.
    pub block_cache_block_size_mb: u64,
    /// How long the block cache keeps a block on disk, in seconds.
    pub block_cache_disk_timeout_sec: u64,
    /// How long the file cache keeps a file, in seconds.
    pub file_cache_timeout_sec: u64,
    /// Maximum file-cache size in MB; `None` or zero uses `cache_size_mb`.
    pub file_cache_max_size_mb: Option<u64>,
    /// Attribute cache lifetime in seconds; `None` uses blobfuse's 7200.
    pub attr_cache_timeout_sec: Option<u64>,
    /// Directory entry cache lifetime in seconds.
    pub entry_cache_timeout_sec: Option<u64>,
    /// Negative directory entry cache lifetime in seconds.
    pub negative_entry_cache_timeout_sec: Option<u64>,
}

impl Default for FuseMountPattern {
    fn default() -> Self {
        FuseMountPattern {
            allow_other: true,
            log_type: "syslog".to_string(),
            log_level: "log_debug".to_string(),
            cache_type: BlobfuseCacheType::BlockCache,
            cache_path: None,
            cache_size_mb: None,
            block_cache_block_size_mb: 16,
            block_cache_disk_timeout_sec: 3600,
            file_cache_timeout_sec: 120,
            file_cache_max_size_mb: None,
            attr_cache_timeout_sec: None,
            entry_cache_timeout_sec: None,
            negative_entry_cache_timeout_sec: None,
        }
    }
}

impl FuseMountPattern {
    /// A pattern with Python's defaults.
    pub fn new() -> Self {
        FuseMountPattern::default()
    }

    /// Cache whole files on disk instead of streaming blocks.
    pub fn with_cache_type(mut self, cache_type: BlobfuseCacheType) -> Self {
        self.cache_type = cache_type;
        self
    }

    /// Place the cache at an explicit workspace-relative path.
    pub fn with_cache_path(mut self, cache_path: impl Into<String>) -> Self {
        self.cache_path = Some(cache_path.into());
        self
    }

    /// Bound the cache, in MB.
    pub fn with_cache_size_mb(mut self, cache_size_mb: u64) -> Self {
        self.cache_size_mb = Some(cache_size_mb);
        self
    }

    /// Whether users other than the mounting one may read the mount.
    pub fn with_allow_other(mut self, allow_other: bool) -> Self {
        self.allow_other = allow_other;
        self
    }

    /// The workspace-relative cache directory used for one account and container.
    fn cache_dir_for(
        &self,
        scratch_id: &str,
        account: &str,
        container: &str,
    ) -> SandboxResult<String> {
        if let Some(cache_path) = &self.cache_path {
            let windows_absolute = crate::sandbox::workspace_paths::is_windows_absolute(cache_path);
            if windows_absolute
                || cache_path.starts_with('/')
                || cache_path.split('/').any(|part| part == "..")
            {
                let shown = if windows_absolute {
                    cache_path.replace('\\', "/")
                } else {
                    cache_path.clone()
                };
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "blobfuse cache_path must be relative to the workspace root",
                )
                .with_context("cache_path", shown));
            }
            return Ok(cache_path.clone());
        }
        Ok(format!(
            "{BLOBFUSE_CACHE_DIR}/{scratch_id}/{}/{}",
            path_component(account),
            path_component(container),
        ))
    }

    /// The blobfuse YAML configuration for one mount.
    #[allow(clippy::too_many_arguments)]
    fn config_text(
        &self,
        account: &str,
        container: &str,
        endpoint: &str,
        cache_dir: &str,
        identity_client_id: Option<&str>,
        account_key: Option<&str>,
    ) -> String {
        let cache_size_mb = self
            .cache_size_mb
            .filter(|size| *size != 0)
            .unwrap_or_else(|| self.cache_type.default_cache_size_mb());
        let file_cache_max_size_mb = self
            .file_cache_max_size_mb
            .filter(|size| *size != 0)
            .unwrap_or(cache_size_mb);

        let mut lines: Vec<String> = Vec::new();
        if self.allow_other {
            lines.push("allow-other: true".to_string());
            lines.push(String::new());
        }
        lines.extend([
            "logging:".to_string(),
            format!("  type: {}", self.log_type),
            format!("  level: {}", self.log_level),
            String::new(),
            "components:".to_string(),
            "  - libfuse".to_string(),
            format!("  - {}", self.cache_type.as_str()),
            "  - attr_cache".to_string(),
            "  - azstorage".to_string(),
            String::new(),
        ]);

        let mut libfuse: Vec<String> = Vec::new();
        if let Some(seconds) = self.entry_cache_timeout_sec {
            libfuse.push(format!("  entry-expiration-sec: {seconds}"));
        }
        if let Some(seconds) = self.negative_entry_cache_timeout_sec {
            libfuse.push(format!("  negative-entry-expiration-sec: {seconds}"));
        }
        if !libfuse.is_empty() {
            lines.push("libfuse:".to_string());
            lines.extend(libfuse);
            lines.push(String::new());
        }

        match self.cache_type {
            BlobfuseCacheType::BlockCache => lines.extend([
                "block_cache:".to_string(),
                format!("  block-size-mb: {}", self.block_cache_block_size_mb),
                format!("  mem-size-mb: {cache_size_mb}"),
                format!("  path: {cache_dir}"),
                format!("  disk-size-mb: {cache_size_mb}"),
                format!("  disk-timeout-sec: {}", self.block_cache_disk_timeout_sec),
                String::new(),
            ]),
            BlobfuseCacheType::FileCache => lines.extend([
                "file_cache:".to_string(),
                format!("  path: {cache_dir}"),
                format!("  timeout-sec: {}", self.file_cache_timeout_sec),
                format!("  max-size-mb: {file_cache_max_size_mb}"),
                String::new(),
            ]),
        }

        lines.extend([
            "attr_cache:".to_string(),
            format!(
                "  timeout-sec: {}",
                self.attr_cache_timeout_sec.unwrap_or(7200)
            ),
            String::new(),
            "azstorage:".to_string(),
            "  type: block".to_string(),
            format!("  account-name: {account}"),
            format!("  container: {container}"),
            format!("  endpoint: {endpoint}"),
        ]);
        match account_key {
            Some(key) => lines.extend([
                "  auth-type: key".to_string(),
                format!("  account-key: {key}"),
            ]),
            None => lines.push("  mode: msi".to_string()),
        }
        if let Some(client_id) = identity_client_id {
            lines.push(format!("  identity-client-id: {client_id}"));
        }
        lines.push(String::new());
        lines.join("\n")
    }
}

/// Replace path separators in a value used as one directory name.
fn path_component(value: &str) -> String {
    value.replace('/', "_")
}

#[async_trait]
impl MountPattern for FuseMountPattern {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn pattern_type(&self) -> &str {
        "fuse"
    }

    fn container_privileges(&self) -> ContainerPrivileges {
        ContainerPrivileges::FuseDevice
    }

    async fn apply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        let account = require(config, FUSE_OPTION_ACCOUNT)?;
        let container = require(config, FUSE_OPTION_CONTAINER)?;
        let endpoint = option(config, FUSE_OPTION_ENDPOINT)
            .map(str::to_string)
            .unwrap_or_else(|| format!("https://{account}.blob.core.windows.net"));
        let identity_client_id = option(config, FUSE_OPTION_IDENTITY_CLIENT_ID);
        let account_key = secret(config, FUSE_SECRET_ACCOUNT_KEY);
        for (field, value) in [
            (FUSE_OPTION_ACCOUNT, Some(account)),
            (FUSE_OPTION_CONTAINER, Some(container)),
            (FUSE_OPTION_ENDPOINT, Some(endpoint.as_str())),
            (FUSE_OPTION_IDENTITY_CLIENT_ID, identity_client_id),
            (FUSE_SECRET_ACCOUNT_KEY, account_key),
        ] {
            if let Some(value) = value {
                reject_config_line_break(field, value)?;
            }
        }

        require_tool(session, "blobfuse2", "command -v blobfuse2 >/dev/null 2>&1").await?;

        let scratch_id = session_scratch_id(session);
        let cache_dir =
            session.normalize_path(&self.cache_dir_for(&scratch_id, account, container)?)?;
        let normalized_mount = session.normalize_path(mount_path)?;
        if is_inside(&cache_dir, &normalized_mount) {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "blobfuse cache_path must be outside the mount path",
            )
            .with_context("mount_path", normalized_mount.clone())
            .with_context("cache_path", cache_dir.clone()));
        }
        let config_dir = session.normalize_path(&format!("{BLOBFUSE_CONFIG_DIR}/{scratch_id}"))?;
        let config_path = format!(
            "{config_dir}/{}_{}.yaml",
            path_component(account),
            path_component(container),
        );

        session.mkdir(mount_path, true).await?;
        session.mkdir(&cache_dir, true).await?;
        session.mkdir(&config_dir, true).await?;

        let config_text = self.config_text(
            account,
            container,
            &endpoint,
            &cache_dir,
            identity_client_id,
            account_key,
        );
        write_sensitive_file(session, &config_path, config_text.as_bytes()).await?;

        let mut argv = vec!["blobfuse2".to_string(), "mount".to_string()];
        if flag(config, OPTION_READ_ONLY) {
            argv.push("--read-only".to_string());
        }
        argv.extend([
            "--config-file".to_string(),
            config_path.clone(),
            mount_path.to_string(),
        ]);

        let result = session.exec(ExecRequest::argv(argv.clone())).await?;
        if !result.ok() {
            let _ = session.rm(&config_path, false).await;
            let sensitive: Vec<String> = config.secret_values().map(str::to_string).collect();
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "`blobfuse2` failed to mount `{mount_path}` with exit code {}: {}",
                    result.exit_code,
                    redact_sensitive_values(result.stderr_text().trim(), &sensitive),
                ),
            ));
        }
        Ok(())
    }

    async fn unapply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        fuse_unmount(session, mount_path).await?;
        // The credential must not outlive the mount it authenticated.
        let account = option(config, FUSE_OPTION_ACCOUNT);
        let container = option(config, FUSE_OPTION_CONTAINER);
        if let (Some(account), Some(container)) = (account, container) {
            let scratch_id = session_scratch_id(session);
            let path = format!(
                "{BLOBFUSE_CONFIG_DIR}/{scratch_id}/{}_{}.yaml",
                path_component(account),
                path_component(container),
            );
            if let Ok(path) = session.normalize_path(&path) {
                let _ = session.rm(&path, false).await;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// S3 and S3-compatible buckets: mount-s3
// ---------------------------------------------------------------------------------------------

/// Defaults a [`MountpointMountPattern`] applies when the mount itself does not set them.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MountpointOptions {
    /// Object prefix exposed at the mount path.
    pub prefix: Option<String>,
    /// Bucket region.
    pub region: Option<String>,
    /// S3-compatible endpoint URL.
    pub endpoint_url: Option<String>,
}

/// Mounts an S3 or S3-compatible bucket with `mount-s3`. Mirrors
/// `agents.sandbox.entries.mounts.patterns.MountpointMountPattern`.
///
/// Credentials never reach the command line: they are written to a `0600` environment file under
/// [`MOUNTPOINT_ENV_DIR`] and sourced by the mount command, so the host's process table never
/// shows them. An endpoint URL that carries its own authority is passed the same way, through
/// `${OPENAI_AGENTS_MOUNT_ENDPOINT_URL}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MountpointMountPattern {
    /// Defaults applied when the mount does not set them.
    pub options: MountpointOptions,
}

/// The shell reference a protected endpoint URL is passed through.
const ENDPOINT_ENV_REFERENCE: &str = "${OPENAI_AGENTS_MOUNT_ENDPOINT_URL}";

/// The environment variable the reference above expands to.
const ENDPOINT_ENV_NAME: &str = "OPENAI_AGENTS_MOUNT_ENDPOINT_URL";

impl MountpointMountPattern {
    /// A pattern with no defaults of its own.
    pub fn new() -> Self {
        MountpointMountPattern::default()
    }

    /// Default the object prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.options.prefix = Some(prefix.into());
        self
    }

    /// Default the bucket region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.options.region = Some(region.into());
        self
    }

    /// Default the endpoint URL.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.options.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// The `mount-s3` argument vector, and whether the endpoint is passed through the environment.
    fn argv(
        &self,
        mount_path: &str,
        config: &MountPatternConfig,
        signed: bool,
        protected_endpoint: bool,
    ) -> SandboxResult<Vec<String>> {
        let bucket = require(config, MOUNTPOINT_OPTION_BUCKET)?;
        let mount_type = option(config, OPTION_MOUNT_TYPE).unwrap_or_default();
        let read_only = flag(config, OPTION_READ_ONLY);

        let mut argv = vec!["mount-s3".to_string()];
        if !signed {
            argv.push("--no-sign-request".to_string());
        }
        if read_only {
            argv.push("--read-only".to_string());
        } else if matches!(mount_type, MOUNT_TYPE_S3 | MOUNT_TYPE_GCS) {
            argv.extend([
                "--allow-overwrite".to_string(),
                "--allow-delete".to_string(),
            ]);
        }
        if let Some(region) = option(config, MOUNTPOINT_OPTION_REGION) {
            reject_config_line_break(MOUNTPOINT_OPTION_REGION, region)?;
            argv.extend(["--region".to_string(), region.to_string()]);
        }
        if protected_endpoint {
            argv.extend([
                "--endpoint-url".to_string(),
                ENDPOINT_ENV_REFERENCE.to_string(),
            ]);
        } else if let Some(endpoint) = option(config, MOUNTPOINT_OPTION_ENDPOINT_URL) {
            reject_config_line_break(MOUNTPOINT_OPTION_ENDPOINT_URL, endpoint)?;
            argv.extend(["--endpoint-url".to_string(), endpoint.to_string()]);
        }
        if mount_type == MOUNT_TYPE_GCS {
            // The GCS XML API rejects the upload checksums `mount-s3` sends by default.
            argv.extend(["--upload-checksums".to_string(), "off".to_string()]);
        }
        if let Some(prefix) = option(config, MOUNTPOINT_OPTION_PREFIX) {
            reject_config_line_break(MOUNTPOINT_OPTION_PREFIX, prefix)?;
            argv.extend(["--prefix".to_string(), prefix.to_string()]);
        }
        reject_config_line_break(MOUNTPOINT_OPTION_BUCKET, bucket)?;
        argv.extend([bucket.to_string(), mount_path.to_string()]);
        Ok(argv)
    }
}

#[async_trait]
impl MountPattern for MountpointMountPattern {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn pattern_type(&self) -> &str {
        "mountpoint"
    }

    fn container_privileges(&self) -> ContainerPrivileges {
        ContainerPrivileges::FuseDevice
    }

    async fn apply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        // An endpoint pinned on the pattern is not part of the mount's own configuration, so it
        // never reaches the credential-authority classification and no acknowledgement covers
        // it. One carrying its own authority would therefore hand the sandbox a credential the
        // manifest never declared. The value is the secret, so it is not echoed back.
        if self
            .options
            .endpoint_url
            .as_deref()
            .is_some_and(url_contains_inline_authority)
        {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "the `mountpoint` mount pattern does not support exposing credentials to the \
                 sandbox through `mount_strategy.pattern.options.endpoint_url`",
            ));
        }
        let bucket = require(config, MOUNTPOINT_OPTION_BUCKET)?.to_string();
        let access_key_id = secret(config, MOUNTPOINT_SECRET_ACCESS_KEY_ID);
        let secret_access_key = secret(config, MOUNTPOINT_SECRET_SECRET_ACCESS_KEY);
        let session_token = secret(config, MOUNTPOINT_SECRET_SESSION_TOKEN);
        let protected_endpoint = secret(config, MOUNTPOINT_SECRET_ENDPOINT_URL);
        let signed = access_key_id.is_some() && secret_access_key.is_some();

        require_tool(session, "mount-s3", "command -v mount-s3 >/dev/null 2>&1").await?;
        session.mkdir(mount_path, true).await?;

        let argv = self.argv(mount_path, config, signed, protected_endpoint.is_some())?;

        let mut env: Vec<(String, String)> = Vec::new();
        if let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) {
            env.push(("AWS_ACCESS_KEY_ID".to_string(), access_key_id.to_string()));
            env.push((
                "AWS_SECRET_ACCESS_KEY".to_string(),
                secret_access_key.to_string(),
            ));
            if let Some(session_token) = session_token {
                env.push(("AWS_SESSION_TOKEN".to_string(), session_token.to_string()));
            }
        }
        if let Some(endpoint) = protected_endpoint {
            env.push((ENDPOINT_ENV_NAME.to_string(), endpoint.to_string()));
        }
        for (name, value) in &env {
            reject_config_line_break(name, value)?;
        }

        let mut sensitive: Vec<String> = env.iter().map(|(_, value)| value.clone()).collect();
        if let Some(endpoint) = protected_endpoint {
            sensitive.extend(inline_url_authority_values(endpoint));
        }

        // The environment reference has to survive quoting: `sh` must expand it, so it is the one
        // argument that is double-quoted rather than single-quoted.
        let mut command = argv
            .iter()
            .map(|arg| {
                if arg == ENDPOINT_ENV_REFERENCE {
                    format!("\"{arg}\"")
                } else {
                    shell_quote(arg)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        let mut stderr_path: Option<String> = None;
        let mut env_path: Option<String> = None;
        if !env.is_empty() {
            let scratch_id = session_scratch_id(session);
            let digest = <sha2::Sha256 as sha2::Digest>::digest(
                format!("{bucket}\0{mount_path}").as_bytes(),
            );
            let command_hash: String = digest
                .iter()
                .take(8)
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let config_dir =
                session.normalize_path(&format!("{MOUNTPOINT_ENV_DIR}/{scratch_id}"))?;
            session.mkdir(&config_dir, true).await?;
            let staged_env = format!("{config_dir}/{command_hash}.env");
            let stdout = format!("{config_dir}/{command_hash}.stdout");
            let stderr = format!("{config_dir}/{command_hash}.stderr");
            write_sensitive_file(session, &staged_env, render_shell_exports(&env).as_bytes())
                .await?;
            command = format!(
                ". {} && exec {command} >{} 2>{}",
                shell_quote(&staged_env),
                shell_quote(&stdout),
                shell_quote(&stderr),
            );
            stderr_path = Some(stderr);
            env_path = Some(staged_env);
        }

        let result = session
            .exec(
                ExecRequest::argv(["sh", "-lc", command.as_str()])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(Duration::from_secs(120))),
            )
            .await?;
        if !result.ok() {
            if let Some(env_path) = &env_path {
                let _ = session.rm(env_path, false).await;
            }
            let mut stderr = result.stderr_text().trim().to_string();
            if let Some(stderr_path) = &stderr_path {
                stderr.push_str(&read_text_if_present(session, stderr_path).await);
            }
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "`mount-s3` failed to mount `{mount_path}` with exit code {}: {}",
                    result.exit_code,
                    redact_sensitive_values(stderr.trim(), &sensitive),
                ),
            ));
        }
        Ok(())
    }

    async fn unapply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        fuse_unmount(session, mount_path).await?;
        // `mount-s3` keeps the environment file open only while it starts, so the credential can
        // be removed with the mount it authenticated.
        if let Some(bucket) = option(config, MOUNTPOINT_OPTION_BUCKET) {
            let scratch_id = session_scratch_id(session);
            let digest = <sha2::Sha256 as sha2::Digest>::digest(
                format!("{bucket}\0{mount_path}").as_bytes(),
            );
            let command_hash: String = digest
                .iter()
                .take(8)
                .map(|byte| format!("{byte:02x}"))
                .collect();
            if let Ok(path) = session.normalize_path(&format!(
                "{MOUNTPOINT_ENV_DIR}/{scratch_id}/{command_hash}.env"
            )) {
                let _ = session.rm(&path, false).await;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Amazon S3 Files
// ---------------------------------------------------------------------------------------------

/// Defaults an [`S3FilesMountPattern`] applies when the mount itself does not set them.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FilesOptions {
    /// Mount target IP address.
    pub mount_target_ip: Option<String>,
    /// Access point id.
    pub access_point: Option<String>,
    /// File system region.
    pub region: Option<String>,
    /// Extra `mount -o` settings, in the order they are rendered. Written as an ordered map, as
    /// Python's `dict[str, str | None]`; the earlier array of pairs is still read.
    #[serde(with = "ordered_option_map")]
    pub extra_options: Vec<(String, Option<String>)>,
}

/// Serde form of an ordered list of `mount -o` settings: a JSON object whose key order is the
/// rendering order, with `null` for a bare flag.
mod ordered_option_map {
    use serde::de::Error as _;
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        options: &[(String, Option<String>)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(options.len()))?;
        for (key, value) in options {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<(String, Option<String>)>, D::Error> {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(key, value)| match value {
                    serde_json::Value::Null => Ok((key, None)),
                    serde_json::Value::String(text) => Ok((key, Some(text))),
                    _ => Err(D::Error::custom("extra_options values must be strings")),
                })
                .collect(),
            serde_json::Value::Array(pairs) => {
                serde_json::from_value(serde_json::Value::Array(pairs)).map_err(D::Error::custom)
            }
            _ => Err(D::Error::custom("extra_options must be an object")),
        }
    }
}

/// Mounts an existing Amazon S3 Files file system with `mount -t s3files`. Mirrors
/// `agents.sandbox.entries.mounts.patterns.S3FilesMountPattern`.
///
/// The mount helper authenticates from the sandbox's own AWS environment, so this pattern stages
/// no credential of its own. It also creates nothing: the file system, its mount target and the
/// network path to it have to exist already.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct S3FilesMountPattern {
    /// Defaults applied when the mount does not set them.
    pub options: S3FilesOptions,
}

impl S3FilesMountPattern {
    /// A pattern with no defaults of its own.
    pub fn new() -> Self {
        S3FilesMountPattern::default()
    }

    /// Default the mount target IP address.
    pub fn with_mount_target_ip(mut self, mount_target_ip: impl Into<String>) -> Self {
        self.options.mount_target_ip = Some(mount_target_ip.into());
        self
    }

    /// Default the access point id.
    pub fn with_access_point(mut self, access_point: impl Into<String>) -> Self {
        self.options.access_point = Some(access_point.into());
        self
    }

    /// Default the file system region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.options.region = Some(region.into());
        self
    }

    /// Add an extra `mount -o` setting; a `None` value renders the key as a bare flag.
    pub fn with_extra_option(
        mut self,
        key: impl Into<String>,
        value: Option<impl Into<String>>,
    ) -> Self {
        upsert_option(
            &mut self.options.extra_options,
            key.into(),
            value.map(Into::into),
        );
        self
    }

    /// The `mount -t s3files` argument vector.
    fn argv(&self, mount_path: &str, config: &MountPatternConfig) -> SandboxResult<Vec<String>> {
        let file_system_id = require(config, S3FILES_OPTION_FILE_SYSTEM_ID)?;
        reject_option_delimiter(S3FILES_OPTION_FILE_SYSTEM_ID, file_system_id)?;
        let mut device = file_system_id.to_string();
        if let Some(subpath) = option(config, S3FILES_OPTION_SUBPATH) {
            reject_option_delimiter(S3FILES_OPTION_SUBPATH, subpath)?;
            device = format!("{device}:{subpath}");
        }

        let mut options = parse_option_list(option(config, S3FILES_OPTION_EXTRA_OPTIONS))?;
        if flag(config, OPTION_READ_ONLY) {
            upsert_option(&mut options, "ro".to_string(), None);
        }
        for (key, config_key) in [
            ("mounttargetip", S3FILES_OPTION_MOUNT_TARGET_IP),
            ("accesspoint", S3FILES_OPTION_ACCESS_POINT),
            ("region", S3FILES_OPTION_REGION),
        ] {
            if let Some(value) = option(config, config_key) {
                reject_option_delimiter(config_key, value)?;
                upsert_option(&mut options, key.to_string(), Some(value.to_string()));
            }
        }

        let mut argv = vec!["mount".to_string(), "-t".to_string(), "s3files".to_string()];
        if !options.is_empty() {
            argv.extend(["-o".to_string(), render_option_list(&options)]);
        }
        argv.extend([device, mount_path.to_string()]);
        Ok(argv)
    }
}

/// Insert or replace one `mount -o` setting, keeping the position of an existing key.
///
/// Mirrors how Python builds the option dict: an explicit field overrides an extra option of the
/// same name without moving it.
fn upsert_option(options: &mut Vec<(String, Option<String>)>, key: String, value: Option<String>) {
    match options.iter_mut().find(|(existing, _)| *existing == key) {
        Some(entry) => entry.1 = value,
        None => options.push((key, value)),
    }
}

/// Render a `mount -o` list as `key[=value]` joined with `,`.
pub fn render_option_list(options: &[(String, Option<String>)]) -> String {
    options
        .iter()
        .map(|(key, value)| match value {
            Some(value) => format!("{key}={value}"),
            None => key.clone(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse a rendered `mount -o` list back into its entries.
fn parse_option_list(rendered: Option<&str>) -> SandboxResult<Vec<(String, Option<String>)>> {
    let mut options: Vec<(String, Option<String>)> = Vec::new();
    for entry in rendered.unwrap_or_default().split(',') {
        if entry.is_empty() {
            continue;
        }
        let (key, value) = match entry.split_once('=') {
            Some((key, value)) => (key.to_string(), Some(value.to_string())),
            None => (entry.to_string(), None),
        };
        if key.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an s3files mount option has an empty name",
            ));
        }
        upsert_option(&mut options, key, value);
    }
    Ok(options)
}

#[async_trait]
impl MountPattern for S3FilesMountPattern {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn pattern_type(&self) -> &str {
        "s3files"
    }

    fn container_privileges(&self) -> ContainerPrivileges {
        // `mount -t s3files` calls `mount(2)` itself rather than going through FUSE.
        ContainerPrivileges::SysAdmin
    }

    fn exposes_credentials(&self) -> bool {
        false
    }

    async fn apply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        require_tool(
            session,
            "mount.s3files",
            "command -v mount.s3files >/dev/null 2>&1",
        )
        .await?;
        session.mkdir(mount_path, true).await?;

        let argv = self.argv(mount_path, config)?;
        let result = session.exec(ExecRequest::argv(argv.clone())).await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "`{}` failed with exit code {}: {}",
                    shell_command(&argv),
                    result.exit_code,
                    result.stderr_text().trim(),
                ),
            ));
        }
        Ok(())
    }

    async fn unapply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        _config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        let command = format!(
            "umount -- {} >/dev/null 2>&1 || true",
            shell_quote(mount_path)
        );
        session
            .exec(
                ExecRequest::argv(["sh", "-lc", command.as_str()])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(Duration::from_secs(60))),
            )
            .await?;
        Ok(())
    }
}

/// Merge a pattern default with a mount's own value, preferring the mount's.
pub(crate) fn prefer<'a>(
    mount_value: Option<&'a str>,
    pattern_value: Option<&'a str>,
) -> Option<&'a str> {
    mount_value.filter(|v| !v.is_empty()).or(pattern_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(options: &[(&str, &str)]) -> MountPatternConfig {
        let mut config = MountPatternConfig::new();
        for (key, value) in options {
            config = config.with_option(*key, *value);
        }
        config
    }

    #[test]
    fn detects_inline_url_authority() {
        assert!(url_contains_inline_authority("https://user:pw@example.com"));
        assert!(url_contains_inline_authority(
            "https://example.com/?sig=abc"
        ));
        assert!(!url_contains_inline_authority("not a url"));
        assert!(!url_contains_inline_authority("https://s3.amazonaws.com"));
    }

    #[test]
    fn collects_every_form_of_an_inline_authority() {
        let values = inline_url_authority_values("https://example.com/?sig=a%2Bb");
        assert!(values.iter().any(|value| value == "sig=a%2Bb"));
        assert!(values.iter().any(|value| value == "a+b"));
    }

    #[test]
    fn redacts_values_and_their_quoted_forms() {
        let secret = "a b";
        let text = format!("tried {secret} and {}", shell_quote(secret));
        let redacted = redact_sensitive_values(&text, &[secret.to_string()]);
        // The quoted form contains the raw value, so the first pass already covers it; the
        // second only matters when a tool prints the quoted form alone.
        assert!(!redacted.contains(secret), "{redacted}");
    }

    #[test]
    fn renders_a_blobfuse_configuration_with_a_key() {
        let pattern = FuseMountPattern::new();
        let text = pattern.config_text(
            "acct",
            "data",
            "https://acct.blob.core.windows.net",
            "/w/.sandbox-blobfuse-cache/aa/acct/data",
            None,
            Some("KEY"),
        );
        assert!(text.contains("allow-other: true"));
        assert!(text.contains("  - block_cache"));
        assert!(text.contains("  mem-size-mb: 50000"));
        assert!(text.contains("  auth-type: key"));
        assert!(text.contains("  account-key: KEY"));
        assert!(!text.contains("mode: msi"));
    }

    #[test]
    fn renders_a_blobfuse_configuration_with_a_managed_identity() {
        let pattern = FuseMountPattern::new().with_cache_type(BlobfuseCacheType::FileCache);
        let text = pattern.config_text("acct", "data", "https://e", "/cache", Some("CID"), None);
        assert!(text.contains("  - file_cache"));
        assert!(text.contains("  max-size-mb: 4096"));
        assert!(text.contains("  mode: msi"));
        assert!(text.contains("  identity-client-id: CID"));
    }

    #[test]
    fn builds_unsigned_read_only_mountpoint_arguments() {
        let pattern = MountpointMountPattern::new();
        let config = config(&[
            (MOUNTPOINT_OPTION_BUCKET, "my-bucket"),
            (OPTION_MOUNT_TYPE, MOUNT_TYPE_S3),
            (OPTION_READ_ONLY, "true"),
        ]);
        let argv = pattern.argv("/w/bucket", &config, false, false).unwrap();
        assert_eq!(
            argv,
            vec![
                "mount-s3",
                "--no-sign-request",
                "--read-only",
                "my-bucket",
                "/w/bucket"
            ]
        );
    }

    #[test]
    fn builds_writable_gcs_mountpoint_arguments() {
        let pattern = MountpointMountPattern::new();
        let config = config(&[
            (MOUNTPOINT_OPTION_BUCKET, "my-bucket"),
            (OPTION_MOUNT_TYPE, MOUNT_TYPE_GCS),
            (OPTION_READ_ONLY, "false"),
            (MOUNTPOINT_OPTION_REGION, "us-east1"),
            (
                MOUNTPOINT_OPTION_ENDPOINT_URL,
                "https://storage.googleapis.com",
            ),
            (MOUNTPOINT_OPTION_PREFIX, "data/"),
        ]);
        let argv = pattern.argv("/w/bucket", &config, true, false).unwrap();
        assert_eq!(
            argv,
            vec![
                "mount-s3",
                "--allow-overwrite",
                "--allow-delete",
                "--region",
                "us-east1",
                "--endpoint-url",
                "https://storage.googleapis.com",
                "--upload-checksums",
                "off",
                "--prefix",
                "data/",
                "my-bucket",
                "/w/bucket"
            ]
        );
    }

    #[test]
    fn passes_an_authority_bearing_endpoint_through_the_environment() {
        let pattern = MountpointMountPattern::new();
        let config = config(&[
            (MOUNTPOINT_OPTION_BUCKET, "b"),
            (OPTION_MOUNT_TYPE, MOUNT_TYPE_S3),
        ]);
        let argv = pattern.argv("/w/b", &config, true, true).unwrap();
        assert!(argv.contains(&ENDPOINT_ENV_REFERENCE.to_string()));
    }

    #[test]
    fn rejects_a_bucket_name_that_would_add_a_configuration_line() {
        let pattern = MountpointMountPattern::new();
        let config = config(&[(MOUNTPOINT_OPTION_BUCKET, "b\nsecret = 1")]);
        let error = pattern.argv("/w/b", &config, true, false).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }

    #[test]
    fn builds_s3files_arguments_with_merged_options() {
        let pattern = S3FilesMountPattern::new();
        let config = config(&[
            (S3FILES_OPTION_FILE_SYSTEM_ID, "fs-123"),
            (S3FILES_OPTION_SUBPATH, "team"),
            (S3FILES_OPTION_REGION, "us-west-2"),
            (S3FILES_OPTION_EXTRA_OPTIONS, "region=eu-west-1,noatime"),
            (OPTION_READ_ONLY, "true"),
        ]);
        let argv = pattern.argv("/w/files", &config).unwrap();
        assert_eq!(
            argv,
            vec![
                "mount",
                "-t",
                "s3files",
                "-o",
                "region=us-west-2,noatime,ro",
                "fs-123:team",
                "/w/files"
            ]
        );
    }

    #[test]
    fn rejects_an_s3files_value_that_would_add_an_option() {
        let pattern = S3FilesMountPattern::new();
        let config = config(&[
            (S3FILES_OPTION_FILE_SYSTEM_ID, "fs-123"),
            (S3FILES_OPTION_REGION, "us-west-2,rw"),
        ]);
        let error = pattern.argv("/w/files", &config).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }

    #[test]
    fn renders_sourceable_exports() {
        let text = render_shell_exports(&[
            ("AWS_ACCESS_KEY_ID".to_string(), "AKIA".to_string()),
            ("AWS_SECRET_ACCESS_KEY".to_string(), "a b".to_string()),
        ]);
        assert_eq!(
            text,
            "export AWS_ACCESS_KEY_ID=AKIA\nexport AWS_SECRET_ACCESS_KEY='a b'\n"
        );
    }
}
