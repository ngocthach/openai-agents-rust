//! The remote storage a sandbox can mount. Mirrors
//! `agents.sandbox.entries.mounts.providers`.
//!
//! A [`ConfiguredMount`] describes *what* is mounted — a bucket, a container, a Box folder — in
//! provider terms. [`ConfiguredMountStrategy`] decides *how* it is attached, either by running a
//! [`MountPattern`] inside the sandbox or by asking the container runtime for a volume:
//!
//! ```no_run
//! use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
//! use openai_agents_sdk::sandbox::entries::Entry;
//!
//! let mount = ConfiguredMountStrategy::rclone(S3Mount::new("my-bucket").with_credentials("AKIA", "secret"))
//!     .mount()
//!     .with_read_only(false);
//! let entry = Entry::mount(mount);
//! ```
//!
//! The runtime configuration is built when the mount activates, not when it is declared, so the
//! mount command always agrees with the [`Mount`]'s own `read_only` flag and with the session the
//! mount is attached to.
//!
//! Credentials in these types are never printed: every one of them hand-writes `Debug`.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mount_patterns::{
    FUSE_OPTION_ACCOUNT, FUSE_OPTION_CONTAINER, FUSE_OPTION_ENDPOINT,
    FUSE_OPTION_IDENTITY_CLIENT_ID, FUSE_SECRET_ACCOUNT_KEY, FuseMountPattern,
    MOUNT_TYPE_AZURE_BLOB, MOUNT_TYPE_BOX, MOUNT_TYPE_GCS, MOUNT_TYPE_R2, MOUNT_TYPE_S3,
    MOUNT_TYPE_S3_FILES, MOUNTPOINT_OPTION_BUCKET, MOUNTPOINT_OPTION_ENDPOINT_URL,
    MOUNTPOINT_OPTION_PREFIX, MOUNTPOINT_OPTION_REGION, MOUNTPOINT_SECRET_ACCESS_KEY_ID,
    MOUNTPOINT_SECRET_ENDPOINT_URL, MOUNTPOINT_SECRET_SECRET_ACCESS_KEY,
    MOUNTPOINT_SECRET_SESSION_TOKEN, MountpointMountPattern, OPTION_MOUNT_TYPE, OPTION_READ_ONLY,
    S3FILES_OPTION_ACCESS_POINT, S3FILES_OPTION_EXTRA_OPTIONS, S3FILES_OPTION_FILE_SYSTEM_ID,
    S3FILES_OPTION_MOUNT_TARGET_IP, S3FILES_OPTION_REGION, S3FILES_OPTION_SUBPATH,
    S3FilesMountPattern, prefer, reject_config_line_break, reject_option_delimiter,
    render_option_list, url_contains_inline_authority,
};
use crate::sandbox::mounts::{
    ContainerPrivileges, Mount, MountCredentialAuthority, MountPattern, MountPatternConfig,
    MountStrategy, RCLONE_OPTION_READ_ONLY, RCLONE_OPTION_REMOTE_KIND, RCLONE_OPTION_REMOTE_NAME,
    RCLONE_OPTION_REMOTE_PATH, RcloneMountPattern, docker_volume_mount_args, docker_volume_name,
};
use crate::sandbox::session::SandboxSession;

// ---------------------------------------------------------------------------------------------
// Pattern selection
// ---------------------------------------------------------------------------------------------

/// Pattern type name of [`RcloneMountPattern`].
pub const PATTERN_RCLONE: &str = "rclone";
/// Pattern type name of [`MountpointMountPattern`].
pub const PATTERN_MOUNTPOINT: &str = "mountpoint";
/// Pattern type name of [`FuseMountPattern`].
pub const PATTERN_FUSE: &str = "fuse";
/// Pattern type name of [`S3FilesMountPattern`].
pub const PATTERN_S3FILES: &str = "s3files";

/// Docker volume driver that speaks rclone.
pub const DRIVER_RCLONE: &str = "rclone";
/// Docker volume driver that speaks the Mountpoint for Amazon S3 protocol.
pub const DRIVER_MOUNTPOINT: &str = "mountpoint";

/// One of the mount tools the built-in storage mounts can be attached with.
///
/// The set is closed, as it is in Python: a [`ConfiguredMount`] names the patterns it supports by
/// type, and anything else is rejected before a credential is staged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinMountPattern {
    /// Attach with `rclone`.
    Rclone(RcloneMountPattern),
    /// Attach with `mount-s3`.
    Mountpoint(MountpointMountPattern),
    /// Attach with `blobfuse2`.
    Fuse(FuseMountPattern),
    /// Attach with `mount -t s3files`.
    S3Files(S3FilesMountPattern),
}

impl BuiltinMountPattern {
    /// The pattern's type name.
    pub fn pattern_type(&self) -> &'static str {
        match self {
            BuiltinMountPattern::Rclone(_) => PATTERN_RCLONE,
            BuiltinMountPattern::Mountpoint(_) => PATTERN_MOUNTPOINT,
            BuiltinMountPattern::Fuse(_) => PATTERN_FUSE,
            BuiltinMountPattern::S3Files(_) => PATTERN_S3FILES,
        }
    }

    /// The pattern as the trait the mount lifecycle drives it through.
    pub fn as_pattern(&self) -> &dyn MountPattern {
        match self {
            BuiltinMountPattern::Rclone(pattern) => pattern,
            BuiltinMountPattern::Mountpoint(pattern) => pattern,
            BuiltinMountPattern::Fuse(pattern) => pattern,
            BuiltinMountPattern::S3Files(pattern) => pattern,
        }
    }

    /// The rclone pattern, when this is one.
    pub fn as_rclone(&self) -> Option<&RcloneMountPattern> {
        match self {
            BuiltinMountPattern::Rclone(pattern) => Some(pattern),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The configured mount contract
// ---------------------------------------------------------------------------------------------

/// Storage described in provider terms, which can be attached by more than one mount tool.
///
/// Mirrors Python's `_ConfiguredMount`. Implementors hold only configuration: the session-aware
/// work happens in [`build_in_container_config`](Self::build_in_container_config), which
/// [`ConfiguredMountStrategy`] calls on every activation.
pub trait ConfiguredMount: Send + Sync + fmt::Debug {
    /// The mount type, matching the Python `type` discriminator.
    fn mount_type(&self) -> &'static str;

    /// The in-container patterns this mount can be attached with.
    fn supported_in_container_patterns(&self) -> &'static [&'static str];

    /// The Docker volume drivers this mount can be attached with.
    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Check the mount's own fields, independently of how it is attached.
    fn validate(&self) -> SandboxResult<()> {
        Ok(())
    }

    /// Check the credential fields an in-container mount writes into the sandbox.
    ///
    /// A half-supplied or blank credential is a misconfiguration there, because it would be
    /// written into a configuration file as an empty value. An external provider (a Docker
    /// volume driver) receives the same fields through the container runtime, which sorts out
    /// its own configuration, so Python leaves them alone and so does this check.
    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        Ok(())
    }

    /// The credential fields that name a file inside the workspace, as `(field, path)`.
    ///
    /// Mirrors Python's `_AUTHORITY_FILE_FIELDS_BY_MOUNT_TYPE`. A file the manifest itself
    /// materializes would be readable by the model, so it is not a credential source.
    fn credential_file_fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// Whether any field of this mount holds a credential or an authority identifier.
    ///
    /// Mirrors Python's `_AUTHORITY_FIELDS_BY_MOUNT_TYPE`, independent of how the mount is
    /// attached.
    fn carries_credentials(&self) -> bool {
        false
    }

    /// The non-secret fields that say *which* storage this mount reaches.
    ///
    /// Folded into the manifest digest so a resume cannot swap one bucket, account or file
    /// system for another. It never includes a credential, an endpoint or a file path.
    fn topology_identity(&self) -> String {
        String::new()
    }

    /// What authority the credentials of this mount grant inside the sandbox.
    ///
    /// `pattern_type` is the tool attaching it, because the same storage exposes different
    /// credentials to different tools, and `has_config_file` reports whether the pattern reads a
    /// configuration the caller placed in the workspace, which the SDK cannot inspect.
    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority;

    /// Build the runtime configuration `pattern` needs to attach this storage.
    ///
    /// `include_credentials` is false on teardown paths, which only need the remote's identity.
    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig>;

    /// Build the Docker volume driver options that attach this storage.
    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        Err(invalid_driver(self.mount_type(), driver))
    }

    /// This mount as [`Any`], so a provider strategy can read its concrete fields back.
    fn as_any(&self) -> &dyn Any;
}

fn invalid_pattern(mount_type: &str, pattern_type: &str) -> SandboxError {
    SandboxError::new(
        ErrorCode::MountConfigInvalid,
        format!("`{mount_type}` cannot be attached with the `{pattern_type}` mount pattern"),
    )
}

fn invalid_driver(mount_type: &str, driver: &str) -> SandboxError {
    SandboxError::new(
        ErrorCode::MountConfigInvalid,
        format!("`{mount_type}` cannot be attached with the `{driver}` Docker volume driver"),
    )
}

/// Whether any of `values` is a non-empty string.
fn any_set(values: &[Option<&str>]) -> bool {
    values
        .iter()
        .any(|value| value.is_some_and(|value| !value.is_empty()))
}

/// Whether any of `values` is a URL carrying authority of its own.
fn any_inline_authority(values: &[Option<&str>]) -> bool {
    values
        .iter()
        .any(|value| value.is_some_and(url_contains_inline_authority))
}

/// Combine the authority classes of one mount's fields.
fn authority(broad: bool, mount_scoped: bool) -> MountCredentialAuthority {
    if broad {
        MountCredentialAuthority::Broad
    } else if mount_scoped {
        MountCredentialAuthority::MountScoped
    } else {
        MountCredentialAuthority::None
    }
}

/// Reject a credential field that is present but blank.
///
/// Mirrors Python's `_invalid_in_container_authority_value_fields`: an empty access key is not a
/// credential-free mount, it is a misconfigured one, and letting it through would write
/// `access_key_id = ` into the generated configuration.
fn reject_blank(field: &str, value: Option<&str>) -> SandboxResult<()> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("the mount field `{field}` is set but blank"),
        ));
    }
    Ok(())
}

/// Reject a half-supplied credential pair.
fn require_credential_pair(
    mount_type: &str,
    first: (&str, Option<&str>),
    second: (&str, Option<&str>),
) -> SandboxResult<()> {
    reject_blank(first.0, first.1)?;
    reject_blank(second.0, second.1)?;
    if first.1.is_some() != second.1.is_some() {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "`{mount_type}` credentials must include both `{}` and `{}`",
                first.0, second.0
            ),
        ));
    }
    Ok(())
}

/// Compact a JSON credential so a pretty-printed key file still renders on one line.
///
/// A Google service-account key and a Box JWT configuration are downloaded as indented JSON, so
/// rejecting every value with a line break would reject the normal way of supplying them.
/// Re-serializing compacts the document and still blocks a value that is not JSON at all.
fn compact_json_credential(field: &str, value: &str) -> SandboxResult<String> {
    if !value.contains(['\n', '\r']) {
        return Ok(value.to_string());
    }
    serde_json::from_str::<serde_json::Value>(value)
        .map(|parsed| parsed.to_string())
        .map_err(|_| {
            SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!("the mount field `{field}` must be valid JSON or contain no line break"),
            )
        })
}

/// Join a remote root with an optional object prefix. Mirrors Python's `_join_remote_path`.
fn join_remote_path(root: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(prefix) => format!("{root}/{}", prefix.trim_start_matches('/')),
        None => root.to_string(),
    }
}

/// Build the rclone runtime configuration for one activation.
///
/// The `[remote]` section header is part of the generated text, because
/// [`RcloneMountPattern`] either uses that text as the whole configuration or merges it into the
/// matching section of a configuration the caller placed in the workspace.
fn rclone_config(
    mount_type: &str,
    pattern: &RcloneMountPattern,
    session_id: &str,
    remote_kind: &str,
    remote_path: &str,
    read_only: bool,
) -> SandboxResult<(MountPatternConfig, String)> {
    let mut config = MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, mount_type)
        .with_option(RCLONE_OPTION_REMOTE_KIND, remote_kind)
        .with_option(RCLONE_OPTION_REMOTE_PATH, remote_path)
        .with_option(RCLONE_OPTION_READ_ONLY, read_only.to_string());
    let remote_name = pattern.resolve_remote_name(session_id, &config)?;
    config = config.with_option(RCLONE_OPTION_REMOTE_NAME, remote_name.clone());
    Ok((config, remote_name))
}

/// Append `key = value` to an rclone section, rejecting a value that would add a line of its own.
fn rclone_line(lines: &mut Vec<String>, key: &str, value: &str) -> SandboxResult<()> {
    reject_config_line_break(key, value)?;
    lines.push(format!("{key} = {value}"));
    Ok(())
}

/// Insert a Docker volume driver option, rejecting a value that would add a line of its own.
fn driver_option(
    options: &mut BTreeMap<String, String>,
    key: &str,
    value: &str,
) -> SandboxResult<()> {
    reject_config_line_break(key, value)?;
    options.insert(key.to_string(), value.to_string());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The strategy
// ---------------------------------------------------------------------------------------------

/// How a [`ConfiguredMount`] is attached.
#[derive(Debug, Clone)]
pub enum ConfiguredMountBackend {
    /// Run a mount tool inside the sandbox.
    InContainer(BuiltinMountPattern),
    /// Ask the container runtime for a named volume.
    DockerVolume {
        /// The Docker volume driver.
        driver: String,
        /// Driver options merged over the ones the mount derives.
        driver_options: BTreeMap<String, String>,
    },
}

/// Attaches a [`ConfiguredMount`] with a built-in pattern or a Docker volume driver.
///
/// The runtime configuration is derived on every lifecycle call rather than stored, so changing
/// the [`Mount`]'s `read_only` flag after the strategy was built still reaches the mount command.
#[derive(Debug, Clone)]
pub struct ConfiguredMountStrategy {
    /// The storage being attached.
    pub spec: Arc<dyn ConfiguredMount>,
    /// How it is attached.
    pub backend: ConfiguredMountBackend,
}

impl ConfiguredMountStrategy {
    /// A strategy attaching `spec` with `backend`.
    pub fn new(spec: Arc<dyn ConfiguredMount>, backend: ConfiguredMountBackend) -> Self {
        ConfiguredMountStrategy { spec, backend }
    }

    /// Attach `spec` by running a mount tool inside the sandbox.
    pub fn in_container(
        spec: impl ConfiguredMount + 'static,
        pattern: BuiltinMountPattern,
    ) -> Self {
        ConfiguredMountStrategy::new(Arc::new(spec), ConfiguredMountBackend::InContainer(pattern))
    }

    /// Attach `spec` with `rclone`.
    pub fn rclone(spec: impl ConfiguredMount + 'static) -> Self {
        ConfiguredMountStrategy::in_container(
            spec,
            BuiltinMountPattern::Rclone(RcloneMountPattern::new()),
        )
    }

    /// Attach `spec` with `mount-s3`.
    pub fn mountpoint(spec: impl ConfiguredMount + 'static) -> Self {
        ConfiguredMountStrategy::in_container(
            spec,
            BuiltinMountPattern::Mountpoint(MountpointMountPattern::new()),
        )
    }

    /// Attach `spec` with `blobfuse2`.
    pub fn blobfuse(spec: impl ConfiguredMount + 'static) -> Self {
        ConfiguredMountStrategy::in_container(
            spec,
            BuiltinMountPattern::Fuse(FuseMountPattern::new()),
        )
    }

    /// Attach `spec` with `mount -t s3files`.
    pub fn s3files(spec: impl ConfiguredMount + 'static) -> Self {
        ConfiguredMountStrategy::in_container(
            spec,
            BuiltinMountPattern::S3Files(S3FilesMountPattern::new()),
        )
    }

    /// Attach `spec` with a Docker volume driver.
    pub fn docker_volume(spec: impl ConfiguredMount + 'static, driver: impl Into<String>) -> Self {
        ConfiguredMountStrategy::new(
            Arc::new(spec),
            ConfiguredMountBackend::DockerVolume {
                driver: driver.into(),
                driver_options: BTreeMap::new(),
            },
        )
    }

    /// Override one Docker volume driver option.
    pub fn with_driver_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let ConfiguredMountBackend::DockerVolume { driver_options, .. } = &mut self.backend {
            driver_options.insert(key.into(), value.into());
        }
        self
    }

    /// A read-only [`Mount`] using this strategy.
    pub fn mount(self) -> Mount {
        Mount::new(Arc::new(self))
    }

    /// The credential fields of this mount that name a workspace file, as `(field, path)`,
    /// including an rclone `config_file_path` the pattern reads.
    pub fn credential_file_paths(&self) -> Vec<(String, String)> {
        let mut paths: Vec<(String, String)> = self
            .spec
            .credential_file_fields()
            .into_iter()
            .map(|(field, path)| (field.to_string(), path))
            .collect();
        if let ConfiguredMountBackend::InContainer(pattern) = &self.backend
            && let Some(config_file) = pattern
                .as_rclone()
                .and_then(|rclone| rclone.config_file_path.as_deref())
        {
            paths.push((
                "mount_strategy.pattern.config_file_path".to_string(),
                config_file.to_string(),
            ));
        }
        paths
    }

    /// What this strategy attaches and how, without any credential: the mount type, the
    /// storage it names, and the pattern or driver that attaches it.
    pub fn topology_identity(&self) -> String {
        let how = match &self.backend {
            ConfiguredMountBackend::InContainer(pattern) => pattern.pattern_type().to_string(),
            ConfiguredMountBackend::DockerVolume { driver, .. } => driver.clone(),
        };
        format!(
            "{}|{}|{}",
            self.spec.mount_type(),
            self.spec.topology_identity(),
            how
        )
    }

    /// The in-container pattern, when this strategy uses one.
    pub fn pattern(&self) -> Option<&BuiltinMountPattern> {
        match &self.backend {
            ConfiguredMountBackend::InContainer(pattern) => Some(pattern),
            ConfiguredMountBackend::DockerVolume { .. } => None,
        }
    }

    /// The configuration for one lifecycle call.
    fn config(
        &self,
        session: &dyn SandboxSession,
        mount: &Mount,
        pattern: &BuiltinMountPattern,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        self.spec.build_in_container_config(
            session.id(),
            pattern,
            mount.read_only,
            include_credentials,
        )
    }
}

#[async_trait]
impl MountStrategy for ConfiguredMountStrategy {
    fn strategy_type(&self) -> &str {
        match self.backend {
            ConfiguredMountBackend::InContainer(_) => "in_container",
            ConfiguredMountBackend::DockerVolume { .. } => "docker_volume",
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        self.spec.validate()?;
        match &self.backend {
            ConfiguredMountBackend::InContainer(pattern) => {
                self.spec.validate_in_container_credentials()?;
                let pattern_type = pattern.pattern_type();
                if !self
                    .spec
                    .supported_in_container_patterns()
                    .contains(&pattern_type)
                {
                    return Err(invalid_pattern(self.spec.mount_type(), pattern_type));
                }
            }
            ConfiguredMountBackend::DockerVolume { driver, .. } => {
                if !self
                    .spec
                    .supported_docker_volume_drivers()
                    .contains(&driver.as_str())
                {
                    return Err(invalid_driver(self.spec.mount_type(), driver));
                }
                // `docker_create_args` cannot report an error, so the driver options are built
                // here too: a rejected value surfaces as itself instead of as "this backend does
                // not support docker-volume mounts".
                self.spec.build_docker_volume_driver_config(driver)?;
            }
        }
        Ok(())
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        match &self.backend {
            ConfiguredMountBackend::InContainer(pattern) => {
                let has_config_file = pattern
                    .as_rclone()
                    .is_some_and(|rclone| rclone.config_file_path.is_some());
                self.spec
                    .credential_authority(pattern.pattern_type(), has_config_file)
            }
            // The credential is handed to the container runtime on the host, not to code running
            // inside the sandbox.
            ConfiguredMountBackend::DockerVolume { .. } => MountCredentialAuthority::None,
        }
    }

    fn carries_credentials(&self, _mount: &Mount) -> bool {
        // Driver options are opaque third-party configuration: Python treats the whole field as
        // live authority and removes it from durable state as a unit.
        let opaque_options = matches!(
            &self.backend,
            ConfiguredMountBackend::DockerVolume { driver_options, .. } if !driver_options.is_empty()
        );
        self.credential_authority() != MountCredentialAuthority::None
            || self.spec.carries_credentials()
            || opaque_options
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        matches!(self.backend, ConfiguredMountBackend::DockerVolume { .. })
    }

    fn container_privileges(&self) -> ContainerPrivileges {
        match &self.backend {
            ConfiguredMountBackend::InContainer(pattern) => {
                pattern.as_pattern().container_privileges()
            }
            // The runtime attaches the volume itself; nothing inside the sandbox mounts anything.
            ConfiguredMountBackend::DockerVolume { .. } => ContainerPrivileges::None,
        }
    }

    fn docker_create_args(&self, mount: &Mount, mount_path: &str) -> Option<Vec<String>> {
        self.docker_create_args_for_session(mount, mount_path, None)
    }

    fn docker_create_args_for_session(
        &self,
        mount: &Mount,
        mount_path: &str,
        session_id: Option<&str>,
    ) -> Option<Vec<String>> {
        let ConfiguredMountBackend::DockerVolume {
            driver,
            driver_options,
        } = &self.backend
        else {
            return None;
        };
        let mut options = self.spec.build_docker_volume_driver_config(driver).ok()?;
        options.extend(
            driver_options
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        Some(docker_volume_mount_args(
            mount_path,
            driver,
            &options,
            mount.read_only,
            Some(&docker_volume_name(session_id, mount_path)),
        ))
    }

    async fn activate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let ConfiguredMountBackend::InContainer(pattern) = &self.backend else {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "docker-volume mounts are not supported by this sandbox backend",
            )
            .with_context("backend", session.backend_id()));
        };
        // Every route into a mount runs through here: manifest apply, the ephemeral re-apply a
        // resumed session does, and the re-attach after a snapshot. The manifest has to
        // acknowledge the exact path before a credential is staged at it.
        session.authorize_mount_activation(mount, mount_path)?;
        let config = self.config(session, mount, pattern, true)?;
        session.mkdir(mount_path, true).await?;
        pattern
            .as_pattern()
            .apply(session, mount_path, &config)
            .await
    }

    async fn deactivate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let ConfiguredMountBackend::InContainer(pattern) = &self.backend else {
            return Ok(());
        };
        let config = self.config(session, mount, pattern, false)?;
        pattern
            .as_pattern()
            .unapply(session, mount_path, &config)
            .await
    }

    async fn teardown_for_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        match self.backend {
            ConfiguredMountBackend::InContainer(_) => {
                self.deactivate(mount, session, mount_path).await
            }
            ConfiguredMountBackend::DockerVolume { .. } => Ok(()),
        }
    }

    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        match self.backend {
            ConfiguredMountBackend::InContainer(_) => {
                self.activate(mount, session, mount_path).await
            }
            ConfiguredMountBackend::DockerVolume { .. } => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// S3
// ---------------------------------------------------------------------------------------------

/// An Amazon S3 bucket, or any bucket reachable over the S3 API. Mirrors Python's `S3Mount`.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct S3Mount {
    /// The bucket name.
    pub bucket: String,
    /// The access key id; `None` mounts the bucket unsigned.
    pub access_key_id: Option<String>,
    /// The secret access key.
    pub secret_access_key: Option<String>,
    /// The STS session token, for temporary credentials.
    pub session_token: Option<String>,
    /// The object prefix exposed at the mount path.
    pub prefix: Option<String>,
    /// The bucket region.
    pub region: Option<String>,
    /// An explicit S3-compatible endpoint.
    pub endpoint_url: Option<String>,
    /// The rclone S3 provider name.
    pub s3_provider: String,
}

impl fmt::Debug for S3Mount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Mount")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .field("endpoint_url", &self.endpoint_url)
            .field("s3_provider", &self.s3_provider)
            .field("credentials", &redacted(self.access_key_id.is_some()))
            .finish()
    }
}

/// How credentials are shown in `Debug` output.
fn redacted(present: bool) -> &'static str {
    if present { "<redacted>" } else { "<unset>" }
}

impl S3Mount {
    /// A bucket mounted without credentials.
    pub fn new(bucket: impl Into<String>) -> Self {
        S3Mount {
            bucket: bucket.into(),
            s3_provider: "AWS".to_string(),
            ..Default::default()
        }
    }

    /// Authenticate with an access key pair.
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Add an STS session token.
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }

    /// Expose only one object prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the bucket region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Reach the bucket through an explicit endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Set the rclone S3 provider name, such as `Minio` or `Wasabi`.
    pub fn with_s3_provider(mut self, s3_provider: impl Into<String>) -> Self {
        self.s3_provider = s3_provider.into();
        self
    }

    /// The remote path, bucket plus prefix.
    pub fn remote_path(&self) -> String {
        join_remote_path(&self.bucket, self.prefix.as_deref())
    }

    fn rclone_lines(&self, remote_name: &str) -> SandboxResult<Vec<String>> {
        let mut lines = vec![format!("[{remote_name}]"), "type = s3".to_string()];
        rclone_line(&mut lines, "provider", &self.s3_provider)?;
        if let Some(endpoint_url) = &self.endpoint_url {
            rclone_line(&mut lines, "endpoint", endpoint_url)?;
        }
        if let Some(region) = &self.region {
            rclone_line(&mut lines, "region", region)?;
        }
        lines.push("env_auth = false".to_string());
        if let (Some(access_key_id), Some(secret_access_key)) =
            (&self.access_key_id, &self.secret_access_key)
        {
            rclone_line(&mut lines, "access_key_id", access_key_id)?;
            rclone_line(&mut lines, "secret_access_key", secret_access_key)?;
            if let Some(session_token) = &self.session_token {
                rclone_line(&mut lines, "session_token", session_token)?;
            }
        }
        Ok(lines)
    }
}

impl ConfiguredMount for S3Mount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_S3
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_RCLONE, PATTERN_MOUNTPOINT]
    }

    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[DRIVER_RCLONE, DRIVER_MOUNTPOINT]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.bucket.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an s3 mount requires a bucket name",
            ));
        }
        Ok(())
    }

    fn carries_credentials(&self) -> bool {
        any_set(&[
            self.access_key_id.as_deref(),
            self.secret_access_key.as_deref(),
            self.session_token.as_deref(),
        ]) || any_inline_authority(&[self.endpoint_url.as_deref()])
    }

    fn topology_identity(&self) -> String {
        format!(
            "{}/{}/{}",
            self.bucket,
            self.prefix.as_deref().unwrap_or_default(),
            self.region.as_deref().unwrap_or_default()
        )
    }

    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        require_credential_pair(
            self.mount_type(),
            ("access_key_id", self.access_key_id.as_deref()),
            ("secret_access_key", self.secret_access_key.as_deref()),
        )?;
        reject_blank("session_token", self.session_token.as_deref())?;
        // A session token only scopes an access key pair, so on its own it is a half-supplied
        // credential set rather than a credential-free mount.
        if self.session_token.is_some() && self.access_key_id.is_none() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "`s3_mount` credentials must include `access_key_id` and `secret_access_key` \
                 when a `session_token` is set",
            ));
        }
        Ok(())
    }

    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority {
        let scoped = any_set(&[
            self.access_key_id.as_deref(),
            self.secret_access_key.as_deref(),
            self.session_token.as_deref(),
        ]) || any_inline_authority(&[self.endpoint_url.as_deref()]);
        authority(has_config_file && pattern_type == PATTERN_RCLONE, scoped)
    }

    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        match pattern {
            BuiltinMountPattern::Rclone(rclone) => {
                let (config, remote_name) = rclone_config(
                    self.mount_type(),
                    rclone,
                    session_id,
                    "s3",
                    &self.remote_path(),
                    read_only,
                )?;
                if !include_credentials {
                    return Ok(config);
                }
                Ok(config
                    .with_config_text(format!("{}\n", self.rclone_lines(&remote_name)?.join("\n"))))
            }
            BuiltinMountPattern::Mountpoint(mountpoint) => Ok(mountpoint_config(
                self.mount_type(),
                &self.bucket,
                prefer(self.prefix.as_deref(), mountpoint.options.prefix.as_deref()),
                prefer(self.region.as_deref(), mountpoint.options.region.as_deref()),
                prefer(
                    self.endpoint_url.as_deref(),
                    mountpoint.options.endpoint_url.as_deref(),
                ),
                self.access_key_id.as_deref(),
                self.secret_access_key.as_deref(),
                self.session_token.as_deref(),
                read_only,
                include_credentials,
            )),
            other => Err(invalid_pattern(self.mount_type(), other.pattern_type())),
        }
    }

    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        let mut options = BTreeMap::new();
        if driver == DRIVER_RCLONE {
            driver_option(&mut options, "type", "s3")?;
            driver_option(&mut options, "s3-provider", &self.s3_provider)?;
            driver_option(&mut options, "path", &self.remote_path())?;
            for (key, value) in [
                ("s3-access-key-id", &self.access_key_id),
                ("s3-secret-access-key", &self.secret_access_key),
                ("s3-session-token", &self.session_token),
                ("s3-endpoint", &self.endpoint_url),
                ("s3-region", &self.region),
            ] {
                if let Some(value) = value {
                    driver_option(&mut options, key, value)?;
                }
            }
            return Ok(options);
        }
        if driver == DRIVER_MOUNTPOINT {
            driver_option(&mut options, "bucket", &self.bucket)?;
            for (key, value) in [
                ("access_key_id", &self.access_key_id),
                ("secret_access_key", &self.secret_access_key),
                ("session_token", &self.session_token),
                ("endpoint_url", &self.endpoint_url),
                ("region", &self.region),
                ("prefix", &self.prefix),
            ] {
                if let Some(value) = value {
                    driver_option(&mut options, key, value)?;
                }
            }
            return Ok(options);
        }
        Err(invalid_driver(self.mount_type(), driver))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The `mount-s3` runtime configuration shared by S3 and GCS mounts.
#[allow(clippy::too_many_arguments)]
fn mountpoint_config(
    mount_type: &str,
    bucket: &str,
    prefix: Option<&str>,
    region: Option<&str>,
    endpoint_url: Option<&str>,
    access_key_id: Option<&str>,
    secret_access_key: Option<&str>,
    session_token: Option<&str>,
    read_only: bool,
    include_credentials: bool,
) -> MountPatternConfig {
    let mut config = MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, mount_type)
        .with_option(OPTION_READ_ONLY, read_only.to_string())
        .with_option(MOUNTPOINT_OPTION_BUCKET, bucket);
    if let Some(prefix) = prefix {
        config = config.with_option(MOUNTPOINT_OPTION_PREFIX, prefix);
    }
    if let Some(region) = region {
        config = config.with_option(MOUNTPOINT_OPTION_REGION, region);
    }
    if let Some(endpoint_url) = endpoint_url {
        // An endpoint carrying its own authority is a credential, so it reaches the mount command
        // through the environment instead of the argument vector.
        if url_contains_inline_authority(endpoint_url) {
            if include_credentials {
                config = config.with_secret(MOUNTPOINT_SECRET_ENDPOINT_URL, endpoint_url);
            }
        } else {
            config = config.with_option(MOUNTPOINT_OPTION_ENDPOINT_URL, endpoint_url);
        }
    }
    if include_credentials
        && let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key)
    {
        config = config
            .with_secret(MOUNTPOINT_SECRET_ACCESS_KEY_ID, access_key_id)
            .with_secret(MOUNTPOINT_SECRET_SECRET_ACCESS_KEY, secret_access_key);
        if let Some(session_token) = session_token {
            config = config.with_secret(MOUNTPOINT_SECRET_SESSION_TOKEN, session_token);
        }
    }
    config
}

// ---------------------------------------------------------------------------------------------
// Google Cloud Storage
// ---------------------------------------------------------------------------------------------

/// A Google Cloud Storage bucket. Mirrors Python's `GCSMount`.
///
/// With HMAC credentials (`access_id` and `secret_access_key`) the bucket is reached over the
/// S3-compatible API; otherwise rclone's native Google Cloud Storage backend is used.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct GcsMount {
    /// The bucket name.
    pub bucket: String,
    /// The HMAC access id.
    pub access_id: Option<String>,
    /// The HMAC secret.
    pub secret_access_key: Option<String>,
    /// The object prefix exposed at the mount path.
    pub prefix: Option<String>,
    /// The bucket region.
    pub region: Option<String>,
    /// An explicit S3-compatible endpoint.
    pub endpoint_url: Option<String>,
    /// Path, inside the sandbox, of a service-account key file.
    pub service_account_file: Option<String>,
    /// Service-account key JSON.
    pub service_account_credentials: Option<String>,
    /// A short-lived OAuth access token.
    pub access_token: Option<String>,
}

impl fmt::Debug for GcsMount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcsMount")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .field("endpoint_url", &self.endpoint_url)
            .field("service_account_file", &self.service_account_file)
            .field(
                "credentials",
                &redacted(
                    self.access_id.is_some()
                        || self.service_account_credentials.is_some()
                        || self.access_token.is_some(),
                ),
            )
            .finish()
    }
}

/// The endpoint the S3-compatible Google Cloud Storage API is reached at.
pub const GCS_S3_ENDPOINT: &str = "https://storage.googleapis.com";

impl GcsMount {
    /// A bucket mounted without credentials.
    pub fn new(bucket: impl Into<String>) -> Self {
        GcsMount {
            bucket: bucket.into(),
            ..Default::default()
        }
    }

    /// Authenticate with HMAC credentials, over the S3-compatible API.
    pub fn with_hmac_credentials(
        mut self,
        access_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_id = Some(access_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Authenticate with service-account key JSON.
    pub fn with_service_account_credentials(mut self, credentials: impl Into<String>) -> Self {
        self.service_account_credentials = Some(credentials.into());
        self
    }

    /// Authenticate with a service-account key file already inside the sandbox.
    pub fn with_service_account_file(mut self, path: impl Into<String>) -> Self {
        self.service_account_file = Some(path.into());
        self
    }

    /// Authenticate with a short-lived OAuth access token.
    pub fn with_access_token(mut self, access_token: impl Into<String>) -> Self {
        self.access_token = Some(access_token.into());
        self
    }

    /// Expose only one object prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the bucket region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Reach the bucket through an explicit endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Whether this mount has the HMAC credentials rclone's S3 backend needs.
    pub fn uses_s3_compatible_api(&self) -> bool {
        self.access_id.is_some() && self.secret_access_key.is_some()
    }

    /// The remote path, bucket plus prefix.
    pub fn remote_path(&self) -> String {
        join_remote_path(&self.bucket, self.prefix.as_deref())
    }

    /// The kind of rclone remote this mount generates.
    ///
    /// HMAC mounts keep their own namespace: they drive the same rclone S3 backend as an
    /// [`S3Mount`], so a shared name would let two mounts overwrite each other's section.
    fn remote_kind(&self) -> &'static str {
        if self.uses_s3_compatible_api() {
            "gcs_s3"
        } else {
            "gcs"
        }
    }

    fn rclone_lines(&self, remote_name: &str) -> SandboxResult<Vec<String>> {
        if self.uses_s3_compatible_api() {
            let mut lines = vec![
                format!("[{remote_name}]"),
                "type = s3".to_string(),
                "provider = GCS".to_string(),
                "env_auth = false".to_string(),
            ];
            rclone_line(
                &mut lines,
                "access_key_id",
                self.access_id.as_deref().unwrap_or_default(),
            )?;
            rclone_line(
                &mut lines,
                "secret_access_key",
                self.secret_access_key.as_deref().unwrap_or_default(),
            )?;
            rclone_line(
                &mut lines,
                "endpoint",
                self.endpoint_url.as_deref().unwrap_or(GCS_S3_ENDPOINT),
            )?;
            if let Some(region) = &self.region {
                rclone_line(&mut lines, "region", region)?;
            }
            return Ok(lines);
        }

        let mut lines = vec![
            format!("[{remote_name}]"),
            "type = google cloud storage".to_string(),
        ];
        if let Some(path) = &self.service_account_file {
            rclone_line(&mut lines, "service_account_file", path)?;
        }
        if let Some(credentials) = &self.service_account_credentials {
            let compacted = compact_json_credential("service_account_credentials", credentials)?;
            rclone_line(&mut lines, "service_account_credentials", &compacted)?;
        }
        if let Some(access_token) = &self.access_token {
            rclone_line(&mut lines, "access_token", access_token)?;
        }
        if self.service_account_file.is_none()
            && self.service_account_credentials.is_none()
            && self.access_token.is_none()
        {
            lines.push("anonymous = true".to_string());
        }
        lines.push("env_auth = false".to_string());
        Ok(lines)
    }
}

impl ConfiguredMount for GcsMount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_GCS
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_RCLONE, PATTERN_MOUNTPOINT]
    }

    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[DRIVER_RCLONE, DRIVER_MOUNTPOINT]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.bucket.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a gcs mount requires a bucket name",
            ));
        }
        Ok(())
    }

    fn credential_file_fields(&self) -> Vec<(&'static str, String)> {
        self.service_account_file
            .iter()
            .filter(|path| !path.is_empty())
            .map(|path| ("service_account_file", path.clone()))
            .collect()
    }

    fn carries_credentials(&self) -> bool {
        any_set(&[
            self.access_id.as_deref(),
            self.secret_access_key.as_deref(),
            self.service_account_file.as_deref(),
            self.service_account_credentials.as_deref(),
            self.access_token.as_deref(),
        ]) || any_inline_authority(&[self.endpoint_url.as_deref()])
    }

    fn topology_identity(&self) -> String {
        format!(
            "{}/{}",
            self.bucket,
            self.prefix.as_deref().unwrap_or_default()
        )
    }

    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        require_credential_pair(
            self.mount_type(),
            ("access_id", self.access_id.as_deref()),
            ("secret_access_key", self.secret_access_key.as_deref()),
        )?;
        for (field, value) in [
            ("service_account_file", self.service_account_file.as_deref()),
            (
                "service_account_credentials",
                self.service_account_credentials.as_deref(),
            ),
            ("access_token", self.access_token.as_deref()),
        ] {
            reject_blank(field, value)?;
        }
        Ok(())
    }

    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority {
        let scoped = any_set(&[
            self.access_id.as_deref(),
            self.secret_access_key.as_deref(),
            self.service_account_credentials.as_deref(),
            self.access_token.as_deref(),
        ]) || any_inline_authority(&[self.endpoint_url.as_deref()]);
        // A key file inside the sandbox is read by the mount tool but also by any command the
        // model runs, and it is not scoped to this bucket.
        let broad = any_set(&[self.service_account_file.as_deref()])
            || (has_config_file && pattern_type == PATTERN_RCLONE);
        authority(broad, scoped)
    }

    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        match pattern {
            BuiltinMountPattern::Rclone(rclone) => {
                let (config, remote_name) = rclone_config(
                    self.mount_type(),
                    rclone,
                    session_id,
                    self.remote_kind(),
                    &self.remote_path(),
                    read_only,
                )?;
                if !include_credentials {
                    return Ok(config);
                }
                Ok(config
                    .with_config_text(format!("{}\n", self.rclone_lines(&remote_name)?.join("\n"))))
            }
            BuiltinMountPattern::Mountpoint(mountpoint) => Ok(mountpoint_config(
                self.mount_type(),
                &self.bucket,
                prefer(self.prefix.as_deref(), mountpoint.options.prefix.as_deref()),
                prefer(self.region.as_deref(), mountpoint.options.region.as_deref()),
                Some(
                    prefer(
                        self.endpoint_url.as_deref(),
                        mountpoint.options.endpoint_url.as_deref(),
                    )
                    .unwrap_or(GCS_S3_ENDPOINT),
                ),
                self.access_id.as_deref(),
                self.secret_access_key.as_deref(),
                None,
                read_only,
                include_credentials,
            )),
            other => Err(invalid_pattern(self.mount_type(), other.pattern_type())),
        }
    }

    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        let mut options = BTreeMap::new();
        if driver == DRIVER_RCLONE {
            if self.uses_s3_compatible_api() {
                driver_option(&mut options, "type", "s3")?;
                driver_option(&mut options, "path", &self.remote_path())?;
                driver_option(&mut options, "s3-provider", "GCS")?;
                driver_option(
                    &mut options,
                    "s3-access-key-id",
                    self.access_id.as_deref().unwrap_or_default(),
                )?;
                driver_option(
                    &mut options,
                    "s3-secret-access-key",
                    self.secret_access_key.as_deref().unwrap_or_default(),
                )?;
                driver_option(
                    &mut options,
                    "s3-endpoint",
                    self.endpoint_url.as_deref().unwrap_or(GCS_S3_ENDPOINT),
                )?;
                if let Some(region) = &self.region {
                    driver_option(&mut options, "s3-region", region)?;
                }
                return Ok(options);
            }
            driver_option(&mut options, "type", "google cloud storage")?;
            driver_option(&mut options, "path", &self.remote_path())?;
            for (key, value) in [
                ("gcs-service-account-file", &self.service_account_file),
                ("gcs-access-token", &self.access_token),
            ] {
                if let Some(value) = value {
                    driver_option(&mut options, key, value)?;
                }
            }
            // The container runtime receives driver options over its API rather than through a
            // configuration file, so a multi-line service-account key cannot add a configuration
            // line and is passed through verbatim, as Python does.
            if let Some(credentials) = &self.service_account_credentials {
                options.insert(
                    "gcs-service-account-credentials".to_string(),
                    credentials.clone(),
                );
            }
            return Ok(options);
        }
        if driver == DRIVER_MOUNTPOINT {
            driver_option(&mut options, "bucket", &self.bucket)?;
            driver_option(
                &mut options,
                "endpoint_url",
                self.endpoint_url.as_deref().unwrap_or(GCS_S3_ENDPOINT),
            )?;
            for (key, value) in [
                ("access_key_id", &self.access_id),
                ("secret_access_key", &self.secret_access_key),
                ("region", &self.region),
                ("prefix", &self.prefix),
            ] {
                if let Some(value) = value {
                    driver_option(&mut options, key, value)?;
                }
            }
            return Ok(options);
        }
        Err(invalid_driver(self.mount_type(), driver))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Cloudflare R2
// ---------------------------------------------------------------------------------------------

/// A Cloudflare R2 bucket. Mirrors Python's `R2Mount`.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct R2Mount {
    /// The bucket name.
    pub bucket: String,
    /// The Cloudflare account id, which derives the endpoint.
    pub account_id: String,
    /// The access key id.
    pub access_key_id: Option<String>,
    /// The secret access key.
    pub secret_access_key: Option<String>,
    /// A custom domain used instead of the account's R2 endpoint.
    pub custom_domain: Option<String>,
}

impl fmt::Debug for R2Mount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("R2Mount")
            .field("bucket", &self.bucket)
            .field("account_id", &self.account_id)
            .field("custom_domain", &self.custom_domain)
            .field("credentials", &redacted(self.access_key_id.is_some()))
            .finish()
    }
}

impl R2Mount {
    /// A bucket in `account_id`.
    pub fn new(bucket: impl Into<String>, account_id: impl Into<String>) -> Self {
        R2Mount {
            bucket: bucket.into(),
            account_id: account_id.into(),
            ..Default::default()
        }
    }

    /// Authenticate with an R2 access key pair.
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Reach the bucket through a custom domain.
    pub fn with_custom_domain(mut self, custom_domain: impl Into<String>) -> Self {
        self.custom_domain = Some(custom_domain.into());
        self
    }

    /// The endpoint this bucket is reached at.
    pub fn endpoint(&self) -> String {
        self.custom_domain
            .clone()
            .unwrap_or_else(|| format!("https://{}.r2.cloudflarestorage.com", self.account_id))
    }

    fn rclone_lines(&self, remote_name: &str) -> SandboxResult<Vec<String>> {
        let mut lines = vec![
            format!("[{remote_name}]"),
            "type = s3".to_string(),
            "provider = Cloudflare".to_string(),
        ];
        rclone_line(&mut lines, "endpoint", &self.endpoint())?;
        lines.push("acl = private".to_string());
        lines.push("env_auth = false".to_string());
        if let (Some(access_key_id), Some(secret_access_key)) =
            (&self.access_key_id, &self.secret_access_key)
        {
            rclone_line(&mut lines, "access_key_id", access_key_id)?;
            rclone_line(&mut lines, "secret_access_key", secret_access_key)?;
        }
        Ok(lines)
    }
}

impl ConfiguredMount for R2Mount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_R2
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_RCLONE]
    }

    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[DRIVER_RCLONE]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.bucket.is_empty() || self.account_id.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an r2 mount requires a bucket name and an account id",
            ));
        }
        Ok(())
    }

    fn carries_credentials(&self) -> bool {
        any_set(&[
            self.access_key_id.as_deref(),
            self.secret_access_key.as_deref(),
        ]) || any_inline_authority(&[self.custom_domain.as_deref()])
    }

    fn topology_identity(&self) -> String {
        format!("{}/{}", self.account_id, self.bucket)
    }

    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        require_credential_pair(
            self.mount_type(),
            ("access_key_id", self.access_key_id.as_deref()),
            ("secret_access_key", self.secret_access_key.as_deref()),
        )?;
        Ok(())
    }

    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority {
        let scoped = any_set(&[
            self.access_key_id.as_deref(),
            self.secret_access_key.as_deref(),
        ]) || any_inline_authority(&[self.custom_domain.as_deref()]);
        authority(has_config_file && pattern_type == PATTERN_RCLONE, scoped)
    }

    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        self.validate()?;
        // Python's R2 checks the pair on every build, not only when the strategy validates, so a
        // half-supplied credential is refused here whichever route reached the mount.
        self.validate_in_container_credentials()?;
        let BuiltinMountPattern::Rclone(rclone) = pattern else {
            return Err(invalid_pattern(self.mount_type(), pattern.pattern_type()));
        };
        let (config, remote_name) = rclone_config(
            self.mount_type(),
            rclone,
            session_id,
            "r2",
            &self.bucket,
            read_only,
        )?;
        if !include_credentials {
            return Ok(config);
        }
        Ok(config.with_config_text(format!("{}\n", self.rclone_lines(&remote_name)?.join("\n"))))
    }

    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        if driver != DRIVER_RCLONE {
            return Err(invalid_driver(self.mount_type(), driver));
        }
        self.validate()?;
        let mut options = BTreeMap::new();
        driver_option(&mut options, "type", "s3")?;
        driver_option(&mut options, "path", &self.bucket)?;
        driver_option(&mut options, "s3-provider", "Cloudflare")?;
        driver_option(&mut options, "s3-endpoint", &self.endpoint())?;
        if let Some(access_key_id) = &self.access_key_id {
            driver_option(&mut options, "s3-access-key-id", access_key_id)?;
        }
        if let Some(secret_access_key) = &self.secret_access_key {
            driver_option(&mut options, "s3-secret-access-key", secret_access_key)?;
        }
        Ok(options)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Azure Blob Storage
// ---------------------------------------------------------------------------------------------

/// An Azure Blob Storage container. Mirrors Python's `AzureBlobMount`.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct AzureBlobMount {
    /// The storage account name.
    pub account: String,
    /// The blob container name.
    pub container: String,
    /// An explicit blob service endpoint.
    pub endpoint: Option<String>,
    /// The managed-identity client id used when no account key is supplied.
    pub identity_client_id: Option<String>,
    /// The storage account key.
    pub account_key: Option<String>,
}

impl fmt::Debug for AzureBlobMount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureBlobMount")
            .field("account", &self.account)
            .field("container", &self.container)
            .field("endpoint", &self.endpoint)
            .field("identity_client_id", &self.identity_client_id)
            .field("account_key", &redacted(self.account_key.is_some()))
            .finish()
    }
}

impl AzureBlobMount {
    /// A container in `account`.
    pub fn new(account: impl Into<String>, container: impl Into<String>) -> Self {
        AzureBlobMount {
            account: account.into(),
            container: container.into(),
            ..Default::default()
        }
    }

    /// Authenticate with the storage account key.
    pub fn with_account_key(mut self, account_key: impl Into<String>) -> Self {
        self.account_key = Some(account_key.into());
        self
    }

    /// Authenticate with a managed identity.
    pub fn with_identity_client_id(mut self, identity_client_id: impl Into<String>) -> Self {
        self.identity_client_id = Some(identity_client_id.into());
        self
    }

    /// Reach the container through an explicit endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    fn rclone_lines(&self, remote_name: &str) -> SandboxResult<Vec<String>> {
        let mut lines = vec![format!("[{remote_name}]"), "type = azureblob".to_string()];
        rclone_line(&mut lines, "account", &self.account)?;
        if let Some(endpoint) = &self.endpoint {
            rclone_line(&mut lines, "endpoint", endpoint)?;
        }
        match (&self.account_key, &self.identity_client_id) {
            (Some(account_key), _) => rclone_line(&mut lines, "key", account_key)?,
            (None, Some(client_id)) => {
                lines.push("use_msi = true".to_string());
                rclone_line(&mut lines, "msi_client_id", client_id)?;
            }
            (None, None) => lines.push("use_msi = false".to_string()),
        }
        Ok(lines)
    }
}

impl ConfiguredMount for AzureBlobMount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_AZURE_BLOB
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_RCLONE, PATTERN_FUSE]
    }

    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[DRIVER_RCLONE]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.account.is_empty() || self.container.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an azure blob mount requires an account and a container",
            ));
        }
        Ok(())
    }

    fn carries_credentials(&self) -> bool {
        any_set(&[
            self.identity_client_id.as_deref(),
            self.account_key.as_deref(),
        ]) || any_inline_authority(&[self.endpoint.as_deref()])
    }

    fn topology_identity(&self) -> String {
        format!("{}/{}", self.account, self.container)
    }

    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        reject_blank("account_key", self.account_key.as_deref())?;
        reject_blank("identity_client_id", self.identity_client_id.as_deref())?;
        Ok(())
    }

    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority {
        // `blobfuse2` falls back to the ambient managed identity whenever no account key is
        // configured, so it can discover authority the mount never names. Python marks this
        // capability `enables_broad_credential_discovery`, which requires a broad
        // acknowledgement whatever the fields say.
        if pattern_type == PATTERN_FUSE {
            return MountCredentialAuthority::Broad;
        }
        let scoped = any_set(&[self.account_key.as_deref()])
            || any_inline_authority(&[self.endpoint.as_deref()]);
        // A managed identity is not scoped to this container: whatever the sandbox can reach with
        // it, the model's commands can reach too.
        let broad = any_set(&[self.identity_client_id.as_deref()])
            || (has_config_file && pattern_type == PATTERN_RCLONE);
        authority(broad, scoped)
    }

    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        match pattern {
            BuiltinMountPattern::Rclone(rclone) => {
                let (config, remote_name) = rclone_config(
                    self.mount_type(),
                    rclone,
                    session_id,
                    "azureblob",
                    &self.container,
                    read_only,
                )?;
                if !include_credentials {
                    return Ok(config);
                }
                Ok(config
                    .with_config_text(format!("{}\n", self.rclone_lines(&remote_name)?.join("\n"))))
            }
            BuiltinMountPattern::Fuse(_) => {
                let mut config = MountPatternConfig::new()
                    .with_option(OPTION_MOUNT_TYPE, self.mount_type())
                    .with_option(OPTION_READ_ONLY, read_only.to_string())
                    .with_option(FUSE_OPTION_ACCOUNT, &self.account)
                    .with_option(FUSE_OPTION_CONTAINER, &self.container);
                if let Some(endpoint) = &self.endpoint {
                    config = config.with_option(FUSE_OPTION_ENDPOINT, endpoint);
                }
                if let Some(client_id) = &self.identity_client_id {
                    config = config.with_option(FUSE_OPTION_IDENTITY_CLIENT_ID, client_id);
                }
                if include_credentials && let Some(account_key) = &self.account_key {
                    config = config.with_secret(FUSE_SECRET_ACCOUNT_KEY, account_key);
                }
                Ok(config)
            }
            other => Err(invalid_pattern(self.mount_type(), other.pattern_type())),
        }
    }

    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        if driver != DRIVER_RCLONE {
            return Err(invalid_driver(self.mount_type(), driver));
        }
        let mut options = BTreeMap::new();
        driver_option(&mut options, "type", "azureblob")?;
        driver_option(&mut options, "path", &self.container)?;
        driver_option(&mut options, "azureblob-account", &self.account)?;
        if let Some(endpoint) = &self.endpoint {
            driver_option(&mut options, "azureblob-endpoint", endpoint)?;
        }
        if let Some(client_id) = &self.identity_client_id {
            driver_option(&mut options, "azureblob-msi-client-id", client_id)?;
        }
        if let Some(account_key) = &self.account_key {
            driver_option(&mut options, "azureblob-key", account_key)?;
        }
        Ok(options)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------------------------

/// Which kind of Box account a JWT mount authenticates as.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxSubType {
    /// An individual user.
    #[default]
    User,
    /// The enterprise's service account.
    Enterprise,
}

impl BoxSubType {
    /// The rclone value for this account kind.
    pub fn as_str(self) -> &'static str {
        match self {
            BoxSubType::User => "user",
            BoxSubType::Enterprise => "enterprise",
        }
    }
}

/// A Box folder mounted with rclone. Mirrors Python's `BoxMount`.
///
/// A non-interactive mount needs a minted `token` or `access_token`: see Box's JWT setup guide
/// and rclone's Box documentation.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct BoxMount {
    /// The folder path inside Box; `None` mounts the root.
    pub path: Option<String>,
    /// The Box application client id.
    pub client_id: Option<String>,
    /// The Box application client secret.
    pub client_secret: Option<String>,
    /// A standalone access token.
    pub access_token: Option<String>,
    /// An rclone OAuth token document.
    pub token: Option<String>,
    /// Path, inside the sandbox, of a Box JWT configuration file.
    pub box_config_file: Option<String>,
    /// Inline Box JWT configuration JSON.
    pub config_credentials: Option<String>,
    /// Which kind of account to authenticate as.
    pub box_sub_type: BoxSubType,
    /// The folder id the mount is rooted at.
    pub root_folder_id: Option<String>,
    /// The user to impersonate.
    pub impersonate: Option<String>,
    /// The owner whose items are listed.
    pub owned_by: Option<String>,
}

impl fmt::Debug for BoxMount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxMount")
            .field("path", &self.path)
            .field("client_id", &self.client_id)
            .field("box_config_file", &self.box_config_file)
            .field("box_sub_type", &self.box_sub_type)
            .field("root_folder_id", &self.root_folder_id)
            .field("impersonate", &self.impersonate)
            .field("owned_by", &self.owned_by)
            .field(
                "credentials",
                &redacted(
                    self.client_secret.is_some()
                        || self.access_token.is_some()
                        || self.token.is_some()
                        || self.config_credentials.is_some(),
                ),
            )
            .finish()
    }
}

impl BoxMount {
    /// The root of the Box account.
    pub fn new() -> Self {
        BoxMount::default()
    }

    /// Mount one folder instead of the root.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the Box application credentials.
    pub fn with_client(
        mut self,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
    ) -> Self {
        self.client_id = Some(client_id.into());
        self.client_secret = Some(client_secret.into());
        self
    }

    /// Authenticate with a standalone access token.
    pub fn with_access_token(mut self, access_token: impl Into<String>) -> Self {
        self.access_token = Some(access_token.into());
        self
    }

    /// Authenticate with an rclone OAuth token document.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Authenticate with inline Box JWT configuration JSON.
    pub fn with_config_credentials(mut self, config_credentials: impl Into<String>) -> Self {
        self.config_credentials = Some(config_credentials.into());
        self
    }

    /// Authenticate as the enterprise service account.
    pub fn with_sub_type(mut self, box_sub_type: BoxSubType) -> Self {
        self.box_sub_type = box_sub_type;
        self
    }

    /// The remote path, with any leading separator removed.
    pub fn remote_path(&self) -> String {
        self.path
            .as_deref()
            .map(|path| path.trim_start_matches('/').to_string())
            .unwrap_or_default()
    }

    /// The fields rclone names the same way the SDK does.
    ///
    /// `config_credentials` and `token` are JSON documents, so they are compacted first: Box
    /// hands them over pretty-printed, and an rclone setting has to stay on one line.
    fn rclone_fields(&self) -> SandboxResult<[(&'static str, Option<String>); 9]> {
        let compact = |field: &'static str, value: &Option<String>| {
            value
                .as_deref()
                .map(|value| compact_json_credential(field, value))
                .transpose()
        };
        Ok([
            ("client_id", self.client_id.clone()),
            ("client_secret", self.client_secret.clone()),
            ("access_token", self.access_token.clone()),
            ("token", compact("token", &self.token)?),
            ("box_config_file", self.box_config_file.clone()),
            (
                "config_credentials",
                compact("config_credentials", &self.config_credentials)?,
            ),
            ("root_folder_id", self.root_folder_id.clone()),
            ("impersonate", self.impersonate.clone()),
            ("owned_by", self.owned_by.clone()),
        ])
    }

    fn rclone_lines(&self, remote_name: &str) -> SandboxResult<Vec<String>> {
        let mut lines = vec![format!("[{remote_name}]"), "type = box".to_string()];
        for (key, value) in self.rclone_fields()? {
            if let Some(value) = value {
                // `box_sub_type` is written in Python's declaration order, between
                // `config_credentials` and `root_folder_id`.
                if key == "root_folder_id" && self.box_sub_type != BoxSubType::User {
                    rclone_line(&mut lines, "box_sub_type", self.box_sub_type.as_str())?;
                }
                rclone_line(&mut lines, key, &value)?;
            }
        }
        if self.root_folder_id.is_none() && self.box_sub_type != BoxSubType::User {
            rclone_line(&mut lines, "box_sub_type", self.box_sub_type.as_str())?;
        }
        Ok(lines)
    }
}

impl ConfiguredMount for BoxMount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_BOX
    }

    fn credential_file_fields(&self) -> Vec<(&'static str, String)> {
        self.box_config_file
            .iter()
            .filter(|path| !path.is_empty())
            .map(|path| ("box_config_file", path.clone()))
            .collect()
    }

    fn carries_credentials(&self) -> bool {
        any_set(&[
            self.client_secret.as_deref(),
            self.access_token.as_deref(),
            self.token.as_deref(),
            self.box_config_file.as_deref(),
            self.config_credentials.as_deref(),
        ])
    }

    fn topology_identity(&self) -> String {
        self.path.clone().unwrap_or_default()
    }

    fn validate_in_container_credentials(&self) -> SandboxResult<()> {
        for (field, value) in [
            ("client_id", self.client_id.as_deref()),
            ("client_secret", self.client_secret.as_deref()),
            ("access_token", self.access_token.as_deref()),
            ("token", self.token.as_deref()),
            ("box_config_file", self.box_config_file.as_deref()),
            ("config_credentials", self.config_credentials.as_deref()),
        ] {
            reject_blank(field, value)?;
        }
        // rclone falls back to an interactive OAuth flow with no authentication source, which
        // cannot complete inside a sandbox. Mirrors Python's `required_any_fields`.
        if !any_set(&[
            self.access_token.as_deref(),
            self.token.as_deref(),
            self.config_credentials.as_deref(),
            self.box_config_file.as_deref(),
        ]) {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an in-container box mount requires a non-interactive authentication source: a \
                 token, an access token, or JWT credentials",
            ));
        }
        Ok(())
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_RCLONE]
    }

    fn supported_docker_volume_drivers(&self) -> &'static [&'static str] {
        &[DRIVER_RCLONE]
    }

    fn credential_authority(
        &self,
        pattern_type: &str,
        has_config_file: bool,
    ) -> MountCredentialAuthority {
        let scoped = any_set(&[
            self.client_secret.as_deref(),
            self.access_token.as_deref(),
            self.token.as_deref(),
            self.config_credentials.as_deref(),
        ]);
        // A JWT configuration file in the sandbox authenticates the whole application, not one
        // folder, and the model's commands can read it.
        let broad = any_set(&[self.box_config_file.as_deref()])
            || (has_config_file && pattern_type == PATTERN_RCLONE);
        authority(broad, scoped)
    }

    fn build_in_container_config(
        &self,
        session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        let BuiltinMountPattern::Rclone(rclone) = pattern else {
            return Err(invalid_pattern(self.mount_type(), pattern.pattern_type()));
        };
        self.validate()?;
        self.validate_in_container_credentials()?;
        let (config, remote_name) = rclone_config(
            self.mount_type(),
            rclone,
            session_id,
            "box",
            &self.remote_path(),
            read_only,
        )?;
        if !include_credentials {
            return Ok(config);
        }
        Ok(config.with_config_text(format!("{}\n", self.rclone_lines(&remote_name)?.join("\n"))))
    }

    fn build_docker_volume_driver_config(
        &self,
        driver: &str,
    ) -> SandboxResult<BTreeMap<String, String>> {
        if driver != DRIVER_RCLONE {
            return Err(invalid_driver(self.mount_type(), driver));
        }
        let mut options = BTreeMap::new();
        driver_option(&mut options, "type", "box")?;
        driver_option(&mut options, "path", &self.remote_path())?;
        for (key, value) in self.rclone_fields()? {
            if let Some(value) = value {
                driver_option(
                    &mut options,
                    &format!("box-{}", key.replace('_', "-")),
                    &value,
                )?;
            }
        }
        if self.box_sub_type != BoxSubType::User {
            driver_option(&mut options, "box-box-sub-type", self.box_sub_type.as_str())?;
        }
        Ok(options)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Amazon S3 Files
// ---------------------------------------------------------------------------------------------

/// An existing Amazon S3 Files file system. Mirrors Python's `S3FilesMount`.
///
/// This mount creates nothing: the file system, its mount target, the VPC route to it and the
/// bucket configuration all have to exist, and the sandbox has to run where the mount target is
/// reachable. At mount time `mount -t s3files` runs inside the sandbox with `file_system_id` as
/// the device.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct S3FilesMount {
    /// The file system id, used as the mount device.
    pub file_system_id: String,
    /// A subdirectory of the file system to expose.
    pub subpath: Option<String>,
    /// The mount target IP address.
    pub mount_target_ip: Option<String>,
    /// The access point id.
    pub access_point: Option<String>,
    /// The file system region.
    pub region: Option<String>,
    /// Extra `mount -o` settings, in the order they are rendered.
    pub extra_options: Vec<(String, Option<String>)>,
}

impl S3FilesMount {
    /// A file system mounted with the sandbox's own AWS environment.
    pub fn new(file_system_id: impl Into<String>) -> Self {
        S3FilesMount {
            file_system_id: file_system_id.into(),
            ..Default::default()
        }
    }

    /// Expose only one subdirectory.
    pub fn with_subpath(mut self, subpath: impl Into<String>) -> Self {
        self.subpath = Some(subpath.into());
        self
    }

    /// Reach the file system through an explicit mount target.
    pub fn with_mount_target_ip(mut self, mount_target_ip: impl Into<String>) -> Self {
        self.mount_target_ip = Some(mount_target_ip.into());
        self
    }

    /// Mount through an access point.
    pub fn with_access_point(mut self, access_point: impl Into<String>) -> Self {
        self.access_point = Some(access_point.into());
        self
    }

    /// Set the file system region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Add an extra `mount -o` setting; a `None` value renders the key as a bare flag.
    pub fn with_extra_option(
        mut self,
        key: impl Into<String>,
        value: Option<impl Into<String>>,
    ) -> Self {
        self.extra_options.push((key.into(), value.map(Into::into)));
        self
    }
}

impl ConfiguredMount for S3FilesMount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_S3_FILES
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[PATTERN_S3FILES]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.file_system_id.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an s3 files mount requires a file system id",
            ));
        }
        for (key, value) in &self.extra_options {
            reject_option_delimiter(key, key)?;
            if let Some(value) = value {
                reject_option_delimiter(key, value)?;
            }
        }
        Ok(())
    }

    fn topology_identity(&self) -> String {
        format!(
            "{}/{}",
            self.file_system_id,
            self.subpath.as_deref().unwrap_or_default()
        )
    }

    fn credential_authority(
        &self,
        _pattern_type: &str,
        _has_config_file: bool,
    ) -> MountCredentialAuthority {
        // The mount helper authenticates from the sandbox's own AWS environment, so it reaches
        // whatever identity that environment carries without the mount naming it, and a
        // free-form option list can redirect it further. Python marks this capability
        // `enables_broad_credential_discovery`.
        MountCredentialAuthority::Broad
    }

    fn build_in_container_config(
        &self,
        _session_id: &str,
        pattern: &BuiltinMountPattern,
        read_only: bool,
        _include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        let BuiltinMountPattern::S3Files(s3files) = pattern else {
            return Err(invalid_pattern(self.mount_type(), pattern.pattern_type()));
        };
        self.validate()?;
        self.validate_in_container_credentials()?;
        let mut merged = s3files.options.extra_options.clone();
        for (key, value) in &self.extra_options {
            match merged.iter_mut().find(|(existing, _)| existing == key) {
                Some(entry) => entry.1 = value.clone(),
                None => merged.push((key.clone(), value.clone())),
            }
        }

        let mut config = MountPatternConfig::new()
            .with_option(OPTION_MOUNT_TYPE, self.mount_type())
            .with_option(OPTION_READ_ONLY, read_only.to_string())
            .with_option(S3FILES_OPTION_FILE_SYSTEM_ID, &self.file_system_id);
        if let Some(subpath) = &self.subpath {
            config = config.with_option(S3FILES_OPTION_SUBPATH, subpath);
        }
        for (key, value) in [
            (
                S3FILES_OPTION_MOUNT_TARGET_IP,
                prefer(
                    self.mount_target_ip.as_deref(),
                    s3files.options.mount_target_ip.as_deref(),
                ),
            ),
            (
                S3FILES_OPTION_ACCESS_POINT,
                prefer(
                    self.access_point.as_deref(),
                    s3files.options.access_point.as_deref(),
                ),
            ),
            (
                S3FILES_OPTION_REGION,
                prefer(self.region.as_deref(), s3files.options.region.as_deref()),
            ),
        ] {
            if let Some(value) = value {
                config = config.with_option(key, value);
            }
        }
        if !merged.is_empty() {
            config = config.with_option(S3FILES_OPTION_EXTRA_OPTIONS, render_option_list(&merged));
        }
        Ok(config)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "sess";

    fn rclone() -> BuiltinMountPattern {
        BuiltinMountPattern::Rclone(RcloneMountPattern::new())
    }

    fn config_text(config: &MountPatternConfig) -> String {
        config.config_text().unwrap_or_default().to_string()
    }

    #[test]
    fn builds_an_s3_rclone_section() {
        let mount = S3Mount::new("my-bucket")
            .with_credentials("AKIA", "SECRET")
            .with_session_token("TOKEN")
            .with_prefix("/data")
            .with_region("us-east-1");
        let config = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap();
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_PATH).unwrap(),
            "my-bucket/data"
        );
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_NAME).unwrap(),
            "sandbox_s3_sess"
        );
        let text = config_text(&config);
        assert!(text.starts_with("[sandbox_s3_sess]\ntype = s3\nprovider = AWS\n"));
        assert!(text.contains("region = us-east-1\n"));
        assert!(text.contains("access_key_id = AKIA\n"));
        assert!(text.contains("session_token = TOKEN\n"));
    }

    #[test]
    fn omits_credentials_on_teardown() {
        let mount = S3Mount::new("my-bucket").with_credentials("AKIA", "SECRET");
        let config = mount
            .build_in_container_config(SESSION, &rclone(), true, false)
            .unwrap();
        assert!(!config.has_credentials());
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_NAME).unwrap(),
            "sandbox_s3_sess"
        );
    }

    #[test]
    fn keeps_hmac_gcs_in_its_own_remote_namespace() {
        let mount = GcsMount::new("bucket").with_hmac_credentials("ID", "SECRET");
        let config = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap();
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_NAME).unwrap(),
            "sandbox_gcs_s3_sess"
        );
        let text = config_text(&config);
        assert!(text.contains("provider = GCS\n"));
        assert!(text.contains("endpoint = https://storage.googleapis.com\n"));
    }

    #[test]
    fn falls_back_to_an_anonymous_gcs_remote() {
        let mount = GcsMount::new("bucket");
        let config = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap();
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_NAME).unwrap(),
            "sandbox_gcs_sess"
        );
        assert!(config_text(&config).contains("anonymous = true\n"));
    }

    #[test]
    fn requires_both_halves_of_an_r2_credential() {
        let mount = R2Mount {
            bucket: "b".to_string(),
            account_id: "acct".to_string(),
            access_key_id: Some("ID".to_string()),
            ..Default::default()
        };
        let error = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }

    #[test]
    fn derives_the_r2_endpoint_from_the_account() {
        let mount = R2Mount::new("b", "acct").with_credentials("ID", "SECRET");
        let text = config_text(
            &mount
                .build_in_container_config(SESSION, &rclone(), true, true)
                .unwrap(),
        );
        assert!(text.contains("endpoint = https://acct.r2.cloudflarestorage.com\n"));
        assert!(text.contains("acl = private\n"));
    }

    #[test]
    fn writes_an_azure_managed_identity_section() {
        let mount = AzureBlobMount::new("acct", "data").with_identity_client_id("CID");
        let text = config_text(
            &mount
                .build_in_container_config(SESSION, &rclone(), true, true)
                .unwrap(),
        );
        assert!(text.contains("use_msi = true\n"));
        assert!(text.contains("msi_client_id = CID\n"));
    }

    #[test]
    fn reports_a_managed_identity_as_broad_authority() {
        let mount = AzureBlobMount::new("acct", "data").with_identity_client_id("CID");
        assert_eq!(
            mount.credential_authority(PATTERN_RCLONE, false),
            MountCredentialAuthority::Broad
        );
        let keyed = AzureBlobMount::new("acct", "data").with_account_key("KEY");
        assert_eq!(
            keyed.credential_authority(PATTERN_RCLONE, false),
            MountCredentialAuthority::MountScoped
        );
        assert_eq!(
            AzureBlobMount::new("acct", "data").credential_authority(PATTERN_RCLONE, false),
            MountCredentialAuthority::None
        );
        // `blobfuse2` reaches the ambient managed identity whatever the mount configures.
        for azure in [
            AzureBlobMount::new("acct", "data").with_account_key("KEY"),
            AzureBlobMount::new("acct", "data"),
        ] {
            assert_eq!(
                azure.credential_authority(PATTERN_FUSE, false),
                MountCredentialAuthority::Broad
            );
        }
    }

    #[test]
    fn strips_a_leading_separator_from_a_box_path() {
        let mount = BoxMount::new().with_path("/Team/Docs").with_token("TOKEN");
        let config = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap();
        assert_eq!(
            config.options.get(RCLONE_OPTION_REMOTE_PATH).unwrap(),
            "Team/Docs"
        );
        assert!(config_text(&config).contains("token = TOKEN\n"));
    }

    #[test]
    fn rejects_a_credential_that_would_add_an_rclone_setting() {
        let mount = S3Mount::new("b").with_credentials("AKIA", "SECRET\nendpoint = evil");
        let error = mount
            .build_in_container_config(SESSION, &rclone(), true, true)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }

    #[test]
    fn rejects_an_unsupported_pattern() {
        let mount = ConfiguredMountStrategy::mountpoint(R2Mount::new("b", "acct")).mount();
        let error = mount.validate().unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);

        let supported = ConfiguredMountStrategy::rclone(R2Mount::new("b", "acct")).mount();
        supported.validate().unwrap();
    }

    #[test]
    fn rejects_an_unsupported_docker_volume_driver() {
        let mount =
            ConfiguredMountStrategy::docker_volume(S3FilesMount::new("fs-1"), DRIVER_RCLONE)
                .mount();
        assert_eq!(
            mount.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
    }

    #[test]
    fn honours_a_read_only_flag_set_after_construction() {
        let strategy = ConfiguredMountStrategy::rclone(S3Mount::new("b"));
        for read_only in [true, false] {
            let config = strategy
                .spec
                .build_in_container_config(SESSION, strategy.pattern().unwrap(), read_only, true)
                .unwrap();
            assert_eq!(
                config.options.get(RCLONE_OPTION_READ_ONLY).unwrap(),
                &read_only.to_string()
            );
        }
    }

    #[test]
    fn quotes_a_driver_option_that_contains_a_comma() {
        let mount =
            GcsMount::new("bucket").with_service_account_credentials("{\"a\": 1, \"b\": 2}");
        let strategy = ConfiguredMountStrategy::docker_volume(mount, DRIVER_RCLONE);
        let args = strategy
            .docker_create_args(&Mount::new(Arc::new(strategy.clone())), "/w/bucket")
            .unwrap();
        let spec = &args[1];
        assert!(spec.contains(
            "\"volume-opt=gcs-service-account-credentials={\"\"a\"\": 1, \"\"b\"\": 2}\""
        ));
    }

    #[test]
    fn keeps_credentials_out_of_debug_output() {
        let s3 = S3Mount::new("b").with_credentials("AKIA", "SUPERSECRET");
        assert!(!format!("{s3:?}").contains("SUPERSECRET"));
        let azure = AzureBlobMount::new("a", "c").with_account_key("SUPERSECRET");
        assert!(!format!("{azure:?}").contains("SUPERSECRET"));
        let gcs = GcsMount::new("b").with_hmac_credentials("ID", "SUPERSECRET");
        assert!(!format!("{gcs:?}").contains("SUPERSECRET"));
        let r2 = R2Mount::new("b", "acct").with_credentials("ID", "SUPERSECRET");
        assert!(!format!("{r2:?}").contains("SUPERSECRET"));
        let boxed = BoxMount::new().with_token("SUPERSECRET");
        assert!(!format!("{boxed:?}").contains("SUPERSECRET"));
    }

    #[test]
    fn merges_s3files_options_with_the_pattern_defaults() {
        let pattern = BuiltinMountPattern::S3Files(
            S3FilesMountPattern::new()
                .with_region("eu-west-1")
                .with_extra_option("noatime", None::<String>),
        );
        let mount = S3FilesMount::new("fs-1")
            .with_subpath("team")
            .with_region("us-west-2");
        let config = mount
            .build_in_container_config(SESSION, &pattern, true, true)
            .unwrap();
        assert_eq!(
            config.options.get(S3FILES_OPTION_REGION).unwrap(),
            "us-west-2"
        );
        assert_eq!(
            config.options.get(S3FILES_OPTION_EXTRA_OPTIONS).unwrap(),
            "noatime"
        );
    }
}
