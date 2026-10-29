//! Mounts: external storage exposed inside the sandbox workspace. Mirrors
//! `agents.sandbox.entries.mounts`.
//!
//! A [`Mount`] holds strategy-independent metadata (where it appears, whether it is read-only)
//! and delegates its lifecycle to a [`MountStrategy`]. The strategy describes *how* the backend
//! makes the storage available:
//!
//! - [`BindMountStrategy`] exposes a host directory (`docker -v` for Docker, a workspace symlink
//!   for Unix-local).
//! - [`DockerVolumeMountStrategy`] asks the container runtime for a named volume.
//! - [`InContainerMountStrategy`] runs a [`MountPattern`] inside the sandbox; cloud providers
//!   implement patterns such as FUSE, `mount-s3` or `rclone` on top of it.
//!
//! Mounts are always ephemeral: they are runtime-attached filesystems, never durable workspace
//! state, so snapshots exclude them.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession, shell_quote};
use crate::sandbox::workspace_paths::{is_windows_absolute, normalize_segments};

/// Whether a mount hands credentials to code running inside the sandbox.
///
/// An in-container mount with any authority other than [`MountCredentialAuthority::None`] writes
/// a credential into the sandbox filesystem or process environment, where the model's commands
/// can read it. Activating such a mount requires the application to acknowledge its exact mount
/// path on the [`Manifest`](crate::sandbox::Manifest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MountCredentialAuthority {
    /// The mount exposes no credential inside the sandbox.
    None,
    /// The credential only grants access to this mount's storage.
    MountScoped,
    /// The credential grants broader authority, such as a managed or workload identity.
    Broad,
}

impl MountCredentialAuthority {
    /// The wire name of the authority level.
    pub fn as_str(self) -> &'static str {
        match self {
            MountCredentialAuthority::None => "none",
            MountCredentialAuthority::MountScoped => "mount_scoped",
            MountCredentialAuthority::Broad => "broad",
        }
    }
}

/// Runtime configuration a [`MountPattern`] needs to attach storage.
///
/// `config_text` holds the secret-bearing configuration written into the sandbox. Its contents
/// never appear in `Debug` output, error messages or serialized session state.
#[derive(Clone, Default)]
pub struct MountPatternConfig {
    config_text: Option<String>,
    /// Non-secret options the pattern passes to its mount command.
    pub options: BTreeMap<String, String>,
}

impl MountPatternConfig {
    /// A configuration with no secret payload.
    pub fn new() -> Self {
        MountPatternConfig::default()
    }

    /// Attach the secret-bearing configuration text.
    pub fn with_config_text(mut self, config_text: impl Into<String>) -> Self {
        self.config_text = Some(config_text.into());
        self
    }

    /// Add a non-secret option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// The secret-bearing configuration text, when the caller needs to write it.
    pub fn config_text(&self) -> Option<&str> {
        self.config_text.as_deref()
    }

    /// Whether this configuration carries a secret.
    pub fn has_config_text(&self) -> bool {
        self.config_text.is_some()
    }

    /// A copy without the secret payload, for teardown paths that must not reproduce it.
    pub fn without_config_text(&self) -> Self {
        MountPatternConfig {
            config_text: None,
            options: self.options.clone(),
        }
    }
}

impl fmt::Debug for MountPatternConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountPatternConfig")
            .field(
                "config_text",
                &self.config_text.as_ref().map(|_| "<redacted>"),
            )
            .field("options", &self.options)
            .finish()
    }
}

/// A command pattern that attaches external storage from inside the sandbox.
///
/// Cloud provider crates implement this for FUSE, `mount-s3`, `rclone` and similar tools.
#[async_trait]
pub trait MountPattern: Send + Sync + fmt::Debug {
    /// The pattern type, matching the Python `type` discriminator.
    fn pattern_type(&self) -> &str;

    /// Whether applying this pattern writes a credential inside the sandbox.
    fn exposes_credentials(&self) -> bool {
        true
    }

    /// Attach the storage at `mount_path`.
    async fn apply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()>;

    /// Detach the storage at `mount_path`.
    async fn unapply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()>;
}

/// A host path the SDK is allowed to read or write through because a mount exposes it.
///
/// Unix-local sessions consult these grants so file operations on a mounted path are not
/// rejected by the workspace symlink guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPathGrant {
    /// The host directory backing the mount.
    pub path: PathBuf,
    /// Whether the SDK may only read through the grant.
    pub read_only: bool,
}

/// How a backend makes a [`Mount`] available.
#[async_trait]
pub trait MountStrategy: Send + Sync + fmt::Debug {
    /// The strategy type, matching the Python `type` discriminator.
    fn strategy_type(&self) -> &str;

    /// This strategy as [`Any`](std::any::Any), for code that needs the concrete strategy type.
    ///
    /// A provider that attaches storage when the sandbox is created has to read a strategy's
    /// configuration back out of a manifest entry, where it is only a `dyn MountStrategy`.
    /// Returning `Some(self)` makes that possible; the default `None` keeps every other strategy
    /// working unchanged.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Validate that this strategy fits `mount`.
    fn validate(&self, mount: &Mount) -> SandboxResult<()> {
        let _ = mount;
        Ok(())
    }

    /// Whether activating this strategy exposes a credential inside the sandbox.
    fn credential_authority(&self) -> MountCredentialAuthority {
        MountCredentialAuthority::None
    }

    /// Whether a native snapshot flow can detach this mount in place.
    fn supports_native_snapshot_detach(&self) -> bool {
        true
    }

    /// Arguments the Docker backend adds to `docker create` for this mount.
    ///
    /// A strategy that returns `Some` is attached by the container runtime before the session
    /// starts, so [`activate`](Self::activate) is skipped on backends that support it.
    fn docker_create_args(&self, mount: &Mount, mount_path: &str) -> Option<Vec<String>> {
        let _ = (mount, mount_path);
        None
    }

    /// The host path this mount exposes, when the SDK may use it directly.
    fn host_path_grant(&self, mount: &Mount) -> Option<HostPathGrant> {
        let _ = mount;
        None
    }

    /// Attach the mount during manifest materialization.
    async fn activate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()>;

    /// Detach the mount during teardown.
    async fn deactivate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()>;

    /// Detach the mount before a snapshot is captured.
    async fn teardown_for_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.deactivate(mount, session, mount_path).await
    }

    /// Re-attach the mount after a snapshot has been captured or restored.
    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.activate(mount, session, mount_path).await
    }
}

/// A manifest entry that exposes external storage inside the sandbox workspace.
#[derive(Debug, Clone)]
pub struct Mount {
    /// Where the mount appears. A relative path resolves under the workspace root, an absolute
    /// path is used as-is; `None` uses the entry's own workspace path.
    pub mount_path: Option<String>,
    /// Whether the SDK treats the mount as read-only.
    pub read_only: bool,
    /// How the backend attaches the storage.
    pub strategy: Arc<dyn MountStrategy>,
}

impl Mount {
    /// A read-only mount using `strategy`.
    pub fn new(strategy: Arc<dyn MountStrategy>) -> Self {
        Mount {
            mount_path: None,
            read_only: true,
            strategy,
        }
    }

    /// Place the mount at an explicit path instead of the entry's workspace path.
    pub fn with_mount_path(mut self, mount_path: impl Into<String>) -> Self {
        self.mount_path = Some(mount_path.into());
        self
    }

    /// Allow the SDK to write through the mount.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Validate the mount against its strategy.
    pub fn validate(&self) -> SandboxResult<()> {
        if let Some(path) = &self.mount_path
            && is_windows_absolute(path)
        {
            return Err(SandboxError::new(
                ErrorCode::InvalidManifestPath,
                format!("mount path `{path}` must use POSIX separators"),
            ));
        }
        self.strategy.clone().validate(self)
    }

    /// Resolve the concrete path where this mount appears, given the workspace root and the
    /// absolute path of the manifest entry that declares it.
    pub fn resolve_mount_path(
        &self,
        workspace_root: &str,
        entry_dest: &str,
    ) -> SandboxResult<String> {
        let root = workspace_root.trim_end_matches('/');
        match &self.mount_path {
            None => Ok(entry_dest.to_string()),
            Some(path) => {
                if is_windows_absolute(path) {
                    return Err(SandboxError::new(
                        ErrorCode::InvalidManifestPath,
                        format!("mount path `{path}` must use POSIX separators"),
                    ));
                }
                let normalized = normalize_segments(path);
                if normalized.split('/').any(|s| s == "..") {
                    return Err(SandboxError::new(
                        ErrorCode::InvalidManifestPath,
                        format!("mount path `{path}` must not contain parent segments"),
                    ));
                }
                if normalized.starts_with('/') {
                    Ok(normalized)
                } else {
                    Ok(format!("{root}/{normalized}"))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bind mounts
// ---------------------------------------------------------------------------------------------

/// Exposes a directory from the machine running the SDK inside the sandbox workspace.
///
/// Docker attaches it with `-v <source>:<mount path>[:ro]` at container creation. Unix-local
/// sessions, which already share the host filesystem, link the mount path to the source.
///
/// <div class="warning">
///
/// `read_only` is enforced by Docker and by the SDK file APIs. On Unix-local sessions a shell
/// command started by the model can still write through the link, because that backend adds no
/// OS-level confinement.
///
/// </div>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindMountStrategy {
    /// The host directory exposed inside the sandbox.
    pub source: PathBuf,
}

impl BindMountStrategy {
    /// Bind `source` into the workspace.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        BindMountStrategy {
            source: source.into(),
        }
    }

    /// A read-only [`Mount`] backed by this strategy.
    pub fn mount(source: impl Into<PathBuf>) -> Mount {
        Mount::new(Arc::new(BindMountStrategy::new(source)))
    }
}

#[async_trait]
impl MountStrategy for BindMountStrategy {
    fn strategy_type(&self) -> &str {
        "bind"
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        if !self.source.is_absolute() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "bind mount source `{}` must be an absolute host path",
                    self.source.display()
                ),
            ));
        }
        Ok(())
    }

    fn docker_create_args(&self, mount: &Mount, mount_path: &str) -> Option<Vec<String>> {
        let suffix = if mount.read_only { ":ro" } else { "" };
        Some(vec![
            "-v".to_string(),
            format!("{}:{mount_path}{suffix}", self.source.display()),
        ])
    }

    fn host_path_grant(&self, mount: &Mount) -> Option<HostPathGrant> {
        Some(HostPathGrant {
            path: self.source.clone(),
            read_only: mount.read_only,
        })
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let script = format!(
            "set -e\nrm -rf {dest}\nmkdir -p \"$(dirname {dest})\"\nln -sfn {src} {dest}",
            dest = shell_quote(mount_path),
            src = shell_quote(&self.source.to_string_lossy()),
        );
        let result = session.exec(ExecRequest::shell(script)).await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "Failed to attach bind mount at `{mount_path}`: {}",
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        // Remove only the link, never the directory it points at.
        let script = format!(
            "if [ -L {dest} ]; then rm -f {dest}; fi",
            dest = shell_quote(mount_path)
        );
        session.exec(ExecRequest::shell(script)).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Docker volume mounts
// ---------------------------------------------------------------------------------------------

/// Asks the container runtime for a named volume, attached before the session starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerVolumeMountStrategy {
    /// The Docker volume driver.
    pub driver: String,
    /// Driver options passed as `volume-opt=<key>=<value>`.
    pub driver_options: BTreeMap<String, String>,
}

impl DockerVolumeMountStrategy {
    /// A volume using `driver`.
    pub fn new(driver: impl Into<String>) -> Self {
        DockerVolumeMountStrategy {
            driver: driver.into(),
            driver_options: BTreeMap::new(),
        }
    }

    /// Add a driver option.
    pub fn with_driver_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.driver_options.insert(key.into(), value.into());
        self
    }
}

#[async_trait]
impl MountStrategy for DockerVolumeMountStrategy {
    fn strategy_type(&self) -> &str {
        "docker_volume"
    }

    fn docker_create_args(&self, mount: &Mount, mount_path: &str) -> Option<Vec<String>> {
        let mut spec = format!("type=volume,dst={mount_path},volume-driver={}", self.driver);
        for (key, value) in &self.driver_options {
            spec.push_str(&format!(",volume-opt={key}={value}"));
        }
        if mount.read_only {
            spec.push_str(",readonly");
        }
        Some(vec!["--mount".to_string(), spec])
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "docker-volume mounts are not supported by the `{}` sandbox backend",
                session.backend_id()
            ),
        ))
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        Ok(())
    }

    async fn teardown_for_snapshot(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        Ok(())
    }

    async fn restore_after_snapshot(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// In-container mounts
// ---------------------------------------------------------------------------------------------

/// Attaches storage by running a [`MountPattern`] inside the sandbox.
///
/// This is the extension point cloud providers build on: they supply a pattern that runs their
/// mount tool and a [`MountPatternConfig`] holding the credentials it needs.
#[derive(Debug, Clone)]
pub struct InContainerMountStrategy {
    /// The command pattern that attaches the storage.
    pub pattern: Arc<dyn MountPattern>,
    /// The runtime configuration handed to the pattern.
    pub config: MountPatternConfig,
}

impl InContainerMountStrategy {
    /// A strategy running `pattern` with `config`.
    pub fn new(pattern: Arc<dyn MountPattern>, config: MountPatternConfig) -> Self {
        InContainerMountStrategy { pattern, config }
    }
}

#[async_trait]
impl MountStrategy for InContainerMountStrategy {
    fn strategy_type(&self) -> &str {
        "in_container"
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        if self.pattern.exposes_credentials() && self.config.has_config_text() {
            MountCredentialAuthority::MountScoped
        } else {
            MountCredentialAuthority::None
        }
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        false
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        session.mkdir(mount_path, true).await?;
        self.pattern.apply(session, mount_path, &self.config).await
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.pattern
            .unapply(session, mount_path, &self.config.without_config_text())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_configuration_text() {
        let config = MountPatternConfig::new()
            .with_config_text("aws_secret_access_key=SUPERSECRET")
            .with_option("region", "us-east-1");
        let debug = format!("{config:?}");
        assert!(!debug.contains("SUPERSECRET"), "{debug}");
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("us-east-1"));
        assert!(!config.without_config_text().has_config_text());
    }

    #[test]
    fn resolves_mount_paths() {
        let mount = BindMountStrategy::mount("/host/data");
        assert_eq!(
            mount
                .resolve_mount_path("/workspace", "/workspace/data")
                .unwrap(),
            "/workspace/data"
        );
        let explicit = BindMountStrategy::mount("/host/data").with_mount_path("mnt/data");
        assert_eq!(
            explicit
                .resolve_mount_path("/workspace/", "/workspace/x")
                .unwrap(),
            "/workspace/mnt/data"
        );
        let absolute = BindMountStrategy::mount("/host/data").with_mount_path("/mnt/data");
        assert_eq!(
            absolute
                .resolve_mount_path("/workspace", "/workspace/x")
                .unwrap(),
            "/mnt/data"
        );
        let escaping = BindMountStrategy::mount("/host/data").with_mount_path("../escape");
        assert!(
            escaping
                .resolve_mount_path("/workspace", "/workspace/x")
                .is_err()
        );
    }

    #[test]
    fn builds_docker_arguments() {
        let mount = BindMountStrategy::mount("/host/data");
        assert_eq!(
            mount.strategy.docker_create_args(&mount, "/workspace/data"),
            Some(vec![
                "-v".to_string(),
                "/host/data:/workspace/data:ro".to_string()
            ])
        );
        let writable = BindMountStrategy::mount("/host/data").with_read_only(false);
        assert_eq!(
            writable
                .strategy
                .docker_create_args(&writable, "/workspace/data"),
            Some(vec![
                "-v".to_string(),
                "/host/data:/workspace/data".to_string()
            ])
        );

        let volume = Mount::new(Arc::new(
            DockerVolumeMountStrategy::new("s3").with_driver_option("bucket", "b"),
        ));
        assert_eq!(
            volume.strategy.docker_create_args(&volume, "/workspace/v"),
            Some(vec![
                "--mount".to_string(),
                "type=volume,dst=/workspace/v,volume-driver=s3,volume-opt=bucket=b,readonly"
                    .to_string()
            ])
        );
    }

    #[test]
    fn rejects_relative_bind_sources() {
        let mount = BindMountStrategy::mount("relative/path");
        let err = mount.validate().unwrap_err();
        assert_eq!(err.code, ErrorCode::MountConfigInvalid);
    }
}

// ---------------------------------------------------------------------------------------------
// The rclone mount pattern
// ---------------------------------------------------------------------------------------------

/// `MountPatternConfig` option holding the rclone remote name (the `[section]` in the config).
pub const RCLONE_OPTION_REMOTE_NAME: &str = "remote_name";
/// `MountPatternConfig` option holding the path inside the remote, for example a bucket name.
pub const RCLONE_OPTION_REMOTE_PATH: &str = "remote_path";
/// `MountPatternConfig` option set to `true` when the mount must be read-only.
pub const RCLONE_OPTION_READ_ONLY: &str = "read_only";

/// Directory, relative to the workspace root, holding generated rclone configurations.
///
/// Mirrors Python's `.sandbox-rclone-config`: the configuration must live inside the workspace so
/// the session's own file API can write it, and every backend's `persist_workspace` excludes this
/// directory so a credential can never be captured in a workspace archive.
pub const RCLONE_CONFIG_DIR: &str = ".sandbox-rclone-config";

/// Shell test that finds `rclone` on `PATH` or where the provider installers put it.
const RCLONE_CHECK: &str = "command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone";

/// How `rclone` exposes the remote inside the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RcloneMountMode {
    /// Mount the remote with `rclone mount`, which needs FUSE.
    #[default]
    Fuse,
    /// Serve the remote over NFS with `rclone serve nfs` and mount that, for images without FUSE.
    Nfs,
}

/// Mounts a cloud bucket inside the sandbox with `rclone`. Mirrors
/// `agents.sandbox.entries.mounts.patterns.RcloneMountPattern`.
///
/// The secret-bearing configuration comes from [`MountPatternConfig::config_text`] and is written
/// to a `600` file under [`RCLONE_CONFIG_DIR`] before the mount runs, then removed when the mount
/// is detached. The non-secret options above select the remote and the mount flags; a provider
/// adds its own flags (`--allow-other`, `--uid`, `--gid`) through [`extra_args`](Self::extra_args).
///
/// Installing `rclone` is the provider's job: this pattern only checks that it is present, so a
/// backend that cannot install it fails with [`ErrorCode::MountMissingTool`] instead of
/// half-attaching a mount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RcloneMountPattern {
    /// How the remote is exposed.
    pub mode: RcloneMountMode,
    /// Extra arguments appended to the `rclone` command.
    pub extra_args: Vec<String>,
    /// Address `rclone serve nfs` listens on, in `nfs` mode.
    pub nfs_addr: Option<String>,
    /// `mount -o` options used in `nfs` mode; `None` uses the defaults Python ships.
    pub nfs_mount_options: Option<Vec<String>>,
    /// Workspace-relative configuration path; `None` derives one per mount path.
    pub config_path: Option<String>,
}

impl Default for RcloneMountPattern {
    fn default() -> Self {
        RcloneMountPattern {
            mode: RcloneMountMode::Fuse,
            extra_args: Vec::new(),
            nfs_addr: None,
            nfs_mount_options: None,
            config_path: None,
        }
    }
}

impl RcloneMountPattern {
    /// A FUSE pattern.
    pub fn new() -> Self {
        RcloneMountPattern::default()
    }

    /// Serve the remote over NFS instead of mounting it with FUSE.
    pub fn nfs() -> Self {
        RcloneMountPattern {
            mode: RcloneMountMode::Nfs,
            ..Default::default()
        }
    }

    /// Append an argument to the `rclone` command.
    pub fn with_extra_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// Append several arguments to the `rclone` command.
    pub fn with_extra_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Set the address `rclone serve nfs` listens on.
    pub fn with_nfs_addr(mut self, addr: impl Into<String>) -> Self {
        self.nfs_addr = Some(addr.into());
        self
    }

    /// Replace the `mount -o` options used in `nfs` mode.
    pub fn with_nfs_mount_options<I, S>(mut self, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.nfs_mount_options = Some(options.into_iter().map(Into::into).collect());
        self
    }

    /// Write the configuration to an explicit workspace-relative path.
    pub fn with_config_path(mut self, config_path: impl Into<String>) -> Self {
        self.config_path = Some(config_path.into());
        self
    }

    /// The workspace-relative configuration path used for `mount_path`.
    ///
    /// The derived name is stable, so teardown finds the file activation wrote, and distinct per
    /// mount, so two mounts in one workspace cannot overwrite each other's credentials.
    pub fn config_path_for(&self, mount_path: &str) -> String {
        self.config_path.clone().unwrap_or_else(|| {
            let digest = <sha2::Sha256 as sha2::Digest>::digest(mount_path.as_bytes());
            let name: String = digest
                .iter()
                .take(8)
                .map(|byte| format!("{byte:02x}"))
                .collect();
            format!("{RCLONE_CONFIG_DIR}/{name}.conf")
        })
    }

    /// The remote this mount attaches, as `name:path`.
    fn remote(&self, config: &MountPatternConfig) -> SandboxResult<(String, String)> {
        let remote_name = config
            .options
            .get(RCLONE_OPTION_REMOTE_NAME)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    format!("an rclone mount requires the `{RCLONE_OPTION_REMOTE_NAME}` option"),
                )
            })?;
        let remote_path = config
            .options
            .get(RCLONE_OPTION_REMOTE_PATH)
            .cloned()
            .unwrap_or_default();
        Ok((remote_name.clone(), remote_path))
    }

    fn read_only(config: &MountPatternConfig) -> bool {
        config
            .options
            .get(RCLONE_OPTION_READ_ONLY)
            .map(String::as_str)
            == Some("true")
    }

    /// The `rclone mount` argument vector.
    fn fuse_argv(
        &self,
        mount_path: &str,
        config_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<Vec<String>> {
        let (remote_name, remote_path) = self.remote(config)?;
        let mut argv = vec![
            "rclone".to_string(),
            "mount".to_string(),
            format!("{remote_name}:{remote_path}"),
            mount_path.to_string(),
        ];
        if Self::read_only(config) {
            argv.push("--read-only".to_string());
        }
        argv.extend([
            "--config".to_string(),
            config_path.to_string(),
            "--daemon".to_string(),
        ]);
        argv.extend(self.extra_args.iter().cloned());
        Ok(argv)
    }

    /// The `rclone serve nfs` argument vector.
    fn nfs_server_argv(
        &self,
        config_path: &str,
        nfs_addr: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<Vec<String>> {
        let (remote_name, remote_path) = self.remote(config)?;
        let mut argv = vec![
            "rclone".to_string(),
            "serve".to_string(),
            "nfs".to_string(),
            format!("{remote_name}:{remote_path}"),
            "--addr".to_string(),
            nfs_addr.to_string(),
            "--config".to_string(),
            config_path.to_string(),
        ];
        if Self::read_only(config) {
            argv.push("--read-only".to_string());
        }
        argv.extend(self.extra_args.iter().cloned());
        Ok(argv)
    }

    /// The `mount -t nfs` command that attaches the served remote.
    ///
    /// `bounded` wraps each attempt in `timeout 10s`, when the image provides it, so one hung
    /// mount cannot stall the retry loop.
    fn nfs_client_command(&self, mount_path: &str, nfs_addr: &str, bounded: bool) -> String {
        let (host, port) = match nfs_addr.rsplit_once(':') {
            Some((host, port)) => (host, port),
            None => (nfs_addr, "2049"),
        };
        // A server bound to every interface is reached over the loopback from inside.
        let host = if matches!(host, "0.0.0.0" | "::" | "") {
            "127.0.0.1"
        } else {
            host
        };
        let options = self.nfs_mount_options.clone().unwrap_or_else(|| {
            vec![
                "vers=4.1".to_string(),
                "tcp".to_string(),
                format!("port={port}"),
                "soft".to_string(),
                "timeo=50".to_string(),
                "retrans=1".to_string(),
            ]
        });
        // The server has just been backgrounded, so the first attempt can race it.
        format!(
            "for i in 1 2 3; do {prefix}mount -v -t nfs -o {options} {source} {target} && \
             exit 0; sleep 1; done; exit 1",
            prefix = if bounded { "timeout 10s " } else { "" },
            options = shell_quote(&options.join(",")),
            source = shell_quote(&format!("{host}:/")),
            target = shell_quote(mount_path),
        )
    }
}

#[async_trait]
impl MountPattern for RcloneMountPattern {
    fn pattern_type(&self) -> &str {
        "rclone"
    }

    fn exposes_credentials(&self) -> bool {
        true
    }

    async fn apply(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        let present = session
            .exec(ExecRequest::argv(["sh", "-lc", RCLONE_CHECK]).with_shell(ExecShell::None))
            .await?;
        if !present.ok() {
            return Err(SandboxError::new(
                ErrorCode::MountMissingTool,
                "rclone is not installed in the sandbox image",
            ));
        }
        let config_text = config.config_text().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an rclone mount requires the remote configuration text",
            )
        })?;
        // A relative configuration path resolves inside the workspace, so the session's own file
        // API can write it; a path outside the workspace root would be rejected.
        let config_path = session.normalize_path(&self.config_path_for(mount_path))?;
        let nfs_addr = self
            .nfs_addr
            .clone()
            .unwrap_or_else(|| "127.0.0.1:2049".to_string());
        let command = match self.mode {
            RcloneMountMode::Fuse => {
                shell_command(&self.fuse_argv(mount_path, &config_path, config)?)
            }
            RcloneMountMode::Nfs => {
                // Not every build of rclone ships the NFS server.
                let serves_nfs = session
                    .exec(
                        ExecRequest::argv([
                            "sh",
                            "-lc",
                            "rclone serve nfs --help >/dev/null 2>&1 || /usr/local/bin/rclone \
                             serve nfs --help >/dev/null 2>&1",
                        ])
                        .with_shell(ExecShell::None),
                    )
                    .await?;
                if !serves_nfs.ok() {
                    return Err(SandboxError::new(
                        ErrorCode::MountMissingTool,
                        "this build of rclone has no `serve nfs` command",
                    ));
                }
                let timeout = session
                    .exec(
                        ExecRequest::argv(["sh", "-lc", "command -v timeout >/dev/null 2>&1"])
                            .with_shell(ExecShell::None),
                    )
                    .await?;
                format!(
                    // The server runs in the background, then the client mounts what it serves.
                    "{server} & {client}",
                    server =
                        shell_command(&self.nfs_server_argv(&config_path, &nfs_addr, config)?),
                    client = self.nfs_client_command(mount_path, &nfs_addr, timeout.ok()),
                )
            }
        };

        // Python creates this directory explicitly; a backend whose upload endpoint does not
        // create parents would otherwise fail on the very first mount.
        if let Some((parent, _)) = config_path.rsplit_once('/')
            && !parent.is_empty()
        {
            session.mkdir(parent, true).await.map_err(|error| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!(
                        "failed to create the rclone configuration directory: {}",
                        error.message
                    ),
                )
            })?;
        }
        session
            .write_file(&config_path, config_text.as_bytes())
            .await
            .map_err(|error| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!(
                        "failed to write the rclone configuration: {}",
                        error.message
                    ),
                )
            })?;
        let chmod = session
            .exec(ExecRequest::argv([
                "chmod",
                "600",
                "--",
                config_path.as_str(),
            ]))
            .await?;
        if !chmod.ok() {
            let _ = session.rm(&config_path, false).await;
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                "failed to restrict the rclone configuration file permissions",
            ));
        }
        let result = session
            .exec(
                ExecRequest::argv(["sh", "-lc", command.as_str()])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(std::time::Duration::from_secs(120))),
            )
            .await?;
        if !result.ok() {
            // The credential must not outlive the attempt that staged it.
            let _ = session.rm(&config_path, false).await;
            // Never echo the command back: it names the remote the credentials belong to.
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "`rclone` failed to mount `{mount_path}` with exit code {}: {}",
                    result.exit_code,
                    result.stderr_text().trim()
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
        let quoted = shell_quote(mount_path);
        let unmount = match self.mode {
            RcloneMountMode::Fuse => format!(
                "fusermount3 -u -- {quoted} 2>/dev/null || fusermount -u -- {quoted} \
                 2>/dev/null || umount -- {quoted} 2>/dev/null || true"
            ),
            RcloneMountMode::Nfs => {
                format!("umount -- {quoted} >/dev/null 2>&1 || true")
            }
        };
        session
            .exec(
                ExecRequest::argv(["sh", "-lc", unmount.as_str()])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(std::time::Duration::from_secs(60))),
            )
            .await?;
        // A daemonized `rclone mount` or a backgrounded `serve nfs` outlives the unmount.
        if let Ok((remote_name, _)) = self.remote(config) {
            let pkill = format!(
                "pkill -f -- {} >/dev/null 2>&1 || true",
                shell_quote(&format!("rclone (mount|serve nfs) {remote_name}:"))
            );
            session
                .exec(ExecRequest::argv(["sh", "-lc", pkill.as_str()]).with_shell(ExecShell::None))
                .await?;
        }
        // The credentials must not outlive the mount.
        if let Ok(config_path) = session.normalize_path(&self.config_path_for(mount_path)) {
            let _ = session.rm(&config_path, false).await;
        }
        Ok(())
    }
}

/// Join an argument vector into a single POSIX shell command.
fn shell_command(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod rclone_tests {
    use super::*;

    fn config() -> MountPatternConfig {
        MountPatternConfig::new()
            .with_config_text("[bucket]\ntype = s3\n")
            .with_option(RCLONE_OPTION_REMOTE_NAME, "bucket")
            .with_option(RCLONE_OPTION_REMOTE_PATH, "my-bucket/prefix")
            .with_option(RCLONE_OPTION_READ_ONLY, "true")
    }

    #[test]
    fn builds_fuse_mount_arguments() {
        let pattern = RcloneMountPattern::new().with_extra_args(["--allow-other", "--uid", "1000"]);
        let argv = pattern
            .fuse_argv("/workspace/data", "/workspace/rclone.conf", &config())
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "rclone",
                "mount",
                "bucket:my-bucket/prefix",
                "/workspace/data",
                "--read-only",
                "--config",
                "/workspace/rclone.conf",
                "--daemon",
                "--allow-other",
                "--uid",
                "1000",
            ]
        );
    }

    #[test]
    fn builds_nfs_server_and_client_commands() {
        let pattern = RcloneMountPattern::nfs();
        let argv = pattern
            .nfs_server_argv("/workspace/rclone.conf", "127.0.0.1:2049", &config())
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "rclone",
                "serve",
                "nfs",
                "bucket:my-bucket/prefix",
                "--addr",
                "127.0.0.1:2049",
                "--config",
                "/workspace/rclone.conf",
                "--read-only",
            ]
        );
        let command = pattern.nfs_client_command("/workspace/data", "0.0.0.0:2049", true);
        // A server bound to every interface is reached over the loopback from inside.
        assert!(command.contains(" 127.0.0.1:/ "), "{command}");
        assert!(command.contains("port=2049"), "{command}");
        assert!(command.contains("for i in 1 2 3"), "{command}");
        // One hung attempt must not stall the retry loop.
        assert!(command.contains("timeout 10s mount"), "{command}");
        assert!(
            !pattern
                .nfs_client_command("/workspace/data", "0.0.0.0:2049", false)
                .contains("timeout 10s"),
        );
    }

    #[test]
    fn requires_a_remote_name() {
        let pattern = RcloneMountPattern::new();
        let bare = MountPatternConfig::new().with_config_text("[bucket]\n");
        let error = pattern
            .fuse_argv("/workspace/data", "/workspace/rclone.conf", &bare)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }

    #[test]
    fn derives_a_distinct_configuration_path_per_mount() {
        let pattern = RcloneMountPattern::new();
        let first = pattern.config_path_for("/workspace/a");
        assert_ne!(first, pattern.config_path_for("/workspace/b"));
        assert!(first.starts_with(RCLONE_CONFIG_DIR), "{first}");
        assert!(first.ends_with(".conf"), "{first}");
        // The derivation is stable, so teardown finds the file activation wrote.
        assert_eq!(first, pattern.config_path_for("/workspace/a"));
        assert_eq!(
            RcloneMountPattern::new()
                .with_config_path("custom.conf")
                .config_path_for("/workspace/a"),
            "custom.conf"
        );
    }
}
