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
use crate::sandbox::session::{ExecRequest, SandboxSession, shell_quote};
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
