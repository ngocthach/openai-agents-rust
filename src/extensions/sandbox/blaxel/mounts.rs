//! Mounts for Blaxel sandboxes. Mirrors `agents.extensions.sandbox.blaxel.mounts`.
//!
//! Two strategies are ported:
//!
//! - [`BlaxelCloudBucketMountStrategy`] attaches S3, R2 and GCS buckets with the FUSE tools
//!   (`s3fs`, `gcsfuse`) it installs and runs inside the sandbox. A credential-bearing mount
//!   writes a short-lived credential file into the workspace, so it reports
//!   [`MountCredentialAuthority::MountScoped`] and the manifest must acknowledge its exact mount
//!   path.
//! - [`BlaxelDriveMountStrategy`] attaches a Blaxel Drive, a persistent network volume, through
//!   the sandbox's own `POST /drives/mount` endpoint. The drive is described either by a
//!   [`BlaxelDriveMountConfig`] handed straight to the strategy, or by a [`BlaxelDriveMount`]
//!   entry, which is the mount-entry form the other providers' storage types take.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mount_entries::{
    BuiltinMountPattern, ConfiguredMount, GCS_S3_ENDPOINT, GcsMount, R2Mount, S3Mount,
};
use crate::sandbox::mounts::{Mount, MountCredentialAuthority, MountPatternConfig, MountStrategy};
use crate::sandbox::session::{ExecRequest, SandboxSession, shell_quote};

const INSTALL_RETRIES: usize = 3;
const MOUNT_TIMEOUT_SECS: u64 = 60;
const INSTALL_TIMEOUT_SECS: u64 = 180;

/// gcsfuse has no Alpine package, so the static binary is extracted from the official `.deb`.
const GCSFUSE_INSTALL_ALPINE: &str = concat!(
    "apk add --no-cache fuse curl binutils && ",
    "GCSFUSE_VER=$(curl -s https://api.github.com/repos/GoogleCloudPlatform/gcsfuse/releases/latest",
    " | grep -o '\"tag_name\": *\"[^\"]*\"' | head -1 | grep -o 'v[0-9.]*') && ",
    "curl -fsSL https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/",
    "${GCSFUSE_VER}/gcsfuse_${GCSFUSE_VER#v}_amd64.deb -o /tmp/gcsfuse.deb && ",
    "cd /tmp && ar x gcsfuse.deb && ",
    "tar -xf data.tar* -C / && ",
    "rm -f gcsfuse.deb control.tar* data.tar* debian-binary",
);

/// gcsfuse on Debian requires the Google Cloud apt repository first.
const GCSFUSE_INSTALL_DEBIAN: &str = concat!(
    "DEBIAN_FRONTEND=noninteractive apt-get update -qq && ",
    "apt-get install -y -qq curl gpg lsb-release && ",
    "curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg ",
    "| gpg --dearmor -o /etc/apt/keyrings/gcsfuse.gpg && ",
    "CODENAME=$(lsb_release -cs) && ",
    "echo \"deb [signed-by=/etc/apt/keyrings/gcsfuse.gpg] ",
    "https://packages.cloud.google.com/apt gcsfuse-${CODENAME} main\" ",
    "| tee /etc/apt/sources.list.d/gcsfuse.list && ",
    "apt-get update -qq && ",
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcsfuse",
);

/// Which FUSE tool a bucket mount drives.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlaxelBucketProvider {
    /// Amazon S3, mounted with `s3fs`.
    S3,
    /// Cloudflare R2, mounted with `s3fs` in SigV4 mode.
    R2,
    /// Google Cloud Storage, mounted with `gcsfuse`.
    Gcs,
}

impl BlaxelBucketProvider {
    /// The wire name of the provider.
    pub fn as_str(self) -> &'static str {
        match self {
            BlaxelBucketProvider::S3 => "s3",
            BlaxelBucketProvider::R2 => "r2",
            BlaxelBucketProvider::Gcs => "gcs",
        }
    }

    fn tool(self) -> &'static str {
        match self {
            BlaxelBucketProvider::S3 | BlaxelBucketProvider::R2 => "s3fs",
            BlaxelBucketProvider::Gcs => "gcsfuse",
        }
    }
}

/// The bucket a Blaxel sandbox mounts, and the credentials the FUSE tool needs.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq)]
pub struct BlaxelCloudBucketMountConfig {
    /// Which FUSE tool to drive.
    pub provider: BlaxelBucketProvider,
    /// The bucket name.
    pub bucket: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
    /// The S3 access key id.
    pub access_key_id: Option<String>,
    /// The S3 secret access key.
    pub secret_access_key: Option<String>,
    /// An STS session token.
    pub session_token: Option<String>,
    /// The S3 region, used to derive an endpoint when none is set.
    pub region: Option<String>,
    /// An explicit S3-compatible endpoint.
    pub endpoint_url: Option<String>,
    /// Expose only the keys under this prefix.
    pub prefix: Option<String>,
    /// The GCS service-account key JSON.
    pub service_account_key: Option<String>,
}

impl std::fmt::Debug for BlaxelCloudBucketMountConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlaxelCloudBucketMountConfig")
            .field("provider", &self.provider)
            .field("bucket", &self.bucket)
            .field("read_only", &self.read_only)
            .field("region", &self.region)
            .field("endpoint_url", &self.endpoint_url)
            .field("prefix", &self.prefix)
            .field(
                "credentials",
                &if self.access_key_id.is_some() || self.service_account_key.is_some() {
                    "<redacted>"
                } else {
                    "<none>"
                },
            )
            .finish()
    }
}

impl BlaxelCloudBucketMountConfig {
    fn base(provider: BlaxelBucketProvider, bucket: impl Into<String>) -> Self {
        BlaxelCloudBucketMountConfig {
            provider,
            bucket: bucket.into(),
            read_only: true,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            region: None,
            endpoint_url: None,
            prefix: None,
            service_account_key: None,
        }
    }

    /// An S3 bucket mounted with `s3fs`.
    pub fn s3(bucket: impl Into<String>) -> Self {
        BlaxelCloudBucketMountConfig::base(BlaxelBucketProvider::S3, bucket)
    }

    /// An R2 bucket mounted with `s3fs`, reached at the account's R2 endpoint.
    pub fn r2(bucket: impl Into<String>, account_id: impl AsRef<str>) -> Self {
        let mut config = BlaxelCloudBucketMountConfig::base(BlaxelBucketProvider::R2, bucket);
        config.endpoint_url = Some(format!(
            "https://{}.r2.cloudflarestorage.com",
            account_id.as_ref()
        ));
        config
    }

    /// A GCS bucket mounted with `gcsfuse`.
    pub fn gcs(bucket: impl Into<String>) -> Self {
        BlaxelCloudBucketMountConfig::base(BlaxelBucketProvider::Gcs, bucket)
    }

    /// A GCS bucket mounted with `s3fs` over the S3-compatible API.
    pub fn gcs_s3_compatible(
        bucket: impl Into<String>,
        access_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        let mut config = BlaxelCloudBucketMountConfig::base(BlaxelBucketProvider::S3, bucket);
        config.access_key_id = Some(access_id.into());
        config.secret_access_key = Some(secret_access_key.into());
        config.endpoint_url = Some("https://storage.googleapis.com".to_string());
        config
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

    /// Add an STS session token to the credential pair.
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }

    /// Derive the endpoint from an S3 region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Reach the bucket through an explicit endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Expose only the keys under `prefix`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Authenticate `gcsfuse` with a service-account key.
    pub fn with_service_account_key(mut self, key: impl Into<String>) -> Self {
        self.service_account_key = Some(key.into());
        self
    }

    /// Allow writes through the mount.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.bucket.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "blaxel cloud bucket mounts require a bucket name",
            ));
        }
        if self.access_key_id.is_some() != self.secret_access_key.is_some() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "blaxel cloud bucket mounts require both an access key id and a secret access \
                 key when either is provided",
            ));
        }
        if self.provider == BlaxelBucketProvider::Gcs && self.access_key_id.is_some() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "gcsfuse mounts authenticate with a service-account key, not an access key pair; \
                 use `BlaxelCloudBucketMountConfig::gcs_s3_compatible` instead",
            ));
        }
        Ok(())
    }

    fn exposes_credentials(&self) -> bool {
        self.access_key_id.is_some() || self.service_account_key.is_some()
    }

    /// The credential file's contents, when the mount writes one.
    fn credential_file_contents(&self) -> Option<String> {
        match self.provider {
            BlaxelBucketProvider::Gcs => self.service_account_key.clone(),
            _ => match (&self.access_key_id, &self.secret_access_key) {
                (Some(access_key_id), Some(secret_access_key)) => {
                    let mut contents = format!("{access_key_id}:{secret_access_key}");
                    if let Some(session_token) = &self.session_token {
                        contents.push(':');
                        contents.push_str(session_token);
                    }
                    Some(contents)
                }
                _ => None,
            },
        }
    }

    /// The shell command that mounts the bucket at `mount_path`.
    fn mount_command(&self, mount_path: &str, credential_path: Option<&str>) -> String {
        let quoted_mount_path = shell_quote(mount_path);
        match self.provider {
            BlaxelBucketProvider::Gcs => {
                let mut options: Vec<String> = Vec::new();
                match credential_path {
                    Some(path) => options.push(shell_quote(&format!("--key-file={path}"))),
                    None => options.push("--anonymous-access".to_string()),
                }
                if self.read_only {
                    options.push("-o ro".to_string());
                }
                if let Some(prefix) = &self.prefix {
                    options.push(format!(
                        "--only-dir={}",
                        shell_quote(prefix.trim_matches('/'))
                    ));
                }
                format!(
                    "gcsfuse {} {} {quoted_mount_path}",
                    options.join(" "),
                    shell_quote(&self.bucket)
                )
            }
            _ => {
                let bucket = match &self.prefix {
                    Some(prefix) => format!("{}:/{}", self.bucket, prefix.trim_matches('/')),
                    None => self.bucket.clone(),
                };
                let mut options = vec!["allow_other".to_string(), "nonempty".to_string()];
                match credential_path {
                    Some(path) => options.push(format!("passwd_file={path}")),
                    None => options.push("public_bucket=1".to_string()),
                }
                if let Some(endpoint_url) = &self.endpoint_url {
                    options.push(format!("url={endpoint_url}"));
                } else if let Some(region) = &self.region {
                    options.push(format!("url=https://s3.{region}.amazonaws.com"));
                    options.push(format!("endpoint={region}"));
                }
                if self.provider == BlaxelBucketProvider::R2 {
                    options.push("sigv4".to_string());
                }
                if self.read_only {
                    options.push("ro".to_string());
                }
                format!(
                    "s3fs {} {quoted_mount_path} -o {}",
                    shell_quote(&bucket),
                    shell_quote(&options.join(","))
                )
            }
        }
    }
}

/// Mounts an S3, R2 or GCS bucket inside a Blaxel sandbox with a FUSE tool.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct BlaxelCloudBucketMountStrategy {
    config: BlaxelCloudBucketMountConfig,
}

impl BlaxelCloudBucketMountStrategy {
    /// A strategy for one bucket.
    pub fn new(config: BlaxelCloudBucketMountConfig) -> Self {
        BlaxelCloudBucketMountStrategy { config }
    }

    /// A [`Mount`] backed by this strategy.
    pub fn mount(config: BlaxelCloudBucketMountConfig) -> Mount {
        let read_only = config.read_only;
        Mount::new(Arc::new(BlaxelCloudBucketMountStrategy::new(config))).with_read_only(read_only)
    }

    /// The bucket this strategy attaches.
    pub fn config(&self) -> &BlaxelCloudBucketMountConfig {
        &self.config
    }

    /// Write the credential file (when there is one), then create the mount point and mount.
    async fn mount_with_credentials(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
        credential_path: Option<&str>,
    ) -> SandboxResult<()> {
        if let (Some(path), Some(contents)) =
            (credential_path, self.config.credential_file_contents())
        {
            session.write_file(path, contents.as_bytes()).await?;
            let chmod = run_shell(
                session,
                &format!("chmod 600 {}", shell_quote(path)),
                MOUNT_TIMEOUT_SECS,
            )
            .await?;
            if !chmod.ok() {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "failed to restrict mount credential file permissions",
                )
                .with_context("exit_code", chmod.exit_code.to_string()));
            }
        }
        let command = self.config.mount_command(mount_path, credential_path);
        run_shell(
            session,
            &format!("mkdir -p {}", shell_quote(mount_path)),
            MOUNT_TIMEOUT_SECS,
        )
        .await?;
        let result = run_shell(session, &command, MOUNT_TIMEOUT_SECS).await?;
        if !result.ok() {
            // The command text carries the mount options, never the credential itself, which
            // lives only in the file that is about to be removed.
            return Err(SandboxError::new(
                ErrorCode::MountFailed,
                format!(
                    "{} mount failed at `{mount_path}` with exit code {}: {}",
                    self.config.provider.tool(),
                    result.exit_code,
                    result.stderr_text().trim()
                ),
            ));
        }
        Ok(())
    }

    /// Install the FUSE tool when the image does not already ship it.
    async fn ensure_tool(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        let tool = self.config.provider.tool();
        let check = run_shell(
            session,
            &format!("which {} >/dev/null 2>&1", shell_quote(tool)),
            MOUNT_TIMEOUT_SECS,
        )
        .await?;
        if check.ok() {
            return Ok(());
        }
        let detect = run_shell(
            session,
            "which apk >/dev/null 2>&1 && echo apk || echo apt",
            MOUNT_TIMEOUT_SECS,
        )
        .await?;
        let alpine = detect.stdout_text().contains("apk");
        let install = match (alpine, tool) {
            (true, "gcsfuse") => GCSFUSE_INSTALL_ALPINE.to_string(),
            (true, "s3fs") => "apk add --no-cache s3fs-fuse".to_string(),
            (true, other) => format!("apk add --no-cache {}", shell_quote(other)),
            (false, "gcsfuse") => GCSFUSE_INSTALL_DEBIAN.to_string(),
            (false, other) => format!(
                "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {}",
                shell_quote(other)
            ),
        };
        let mut last = None;
        for _ in 0..INSTALL_RETRIES {
            let result = run_shell(session, &install, INSTALL_TIMEOUT_SECS).await?;
            if result.ok() {
                return Ok(());
            }
            last = Some(result.exit_code);
        }
        Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            format!(
                "failed to install `{tool}` after {INSTALL_RETRIES} attempts (last exit code {})",
                last.unwrap_or(-1)
            ),
        ))
    }
}

#[async_trait]
impl MountStrategy for BlaxelCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "blaxel_cloud_bucket"
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        self.config.validate()
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        // A credential-bearing mount writes the FUSE tool's credential file inside the sandbox,
        // where a model-run command could read it before the file is removed.
        if self.config.exposes_credentials() {
            MountCredentialAuthority::MountScoped
        } else {
            MountCredentialAuthority::None
        }
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        // The bucket's contents are not workspace state, so a snapshot unmounts it first.
        false
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.config.validate()?;
        self.ensure_tool(session).await?;

        // The credential file lives in the workspace root, so its path is registered as a
        // persist skip path before anything is written: a snapshot taken while it exists must
        // not archive it, and a registration the session refuses must leave no file behind.
        let credential_path = match self.config.credential_file_contents() {
            None => None,
            Some(_) => {
                let stem = match self.config.provider {
                    BlaxelBucketProvider::Gcs => "gcs-creds",
                    _ => "s3fs-passwd",
                };
                let relative = format!(
                    ".openai-agents-{stem}-{}",
                    &uuid::Uuid::new_v4().simple().to_string()[..8]
                );
                session.register_persist_workspace_skip_path(&relative)?;
                Some(format!(
                    "{}/{relative}",
                    session.workspace_root().trim_end_matches('/')
                ))
            }
        };

        let outcome = self
            .mount_with_credentials(session, mount_path, credential_path.as_deref())
            .await;
        // The credential file is removed whether or not the mount came up, so it never outlives
        // the activation that needed it. A removal that fails outranks the mount's own result:
        // a credential left in the workspace is the more urgent fact.
        if let Some(path) = &credential_path {
            let removal = run_shell(
                session,
                &format!("rm -f {}", shell_quote(path)),
                MOUNT_TIMEOUT_SECS,
            )
            .await;
            match removal {
                Ok(result) if result.ok() => {}
                Ok(result) => {
                    return Err(SandboxError::new(
                        ErrorCode::MountConfigInvalid,
                        "failed to remove mount credential file",
                    )
                    .with_context("exit_code", result.exit_code.to_string()));
                }
                Err(error) => return Err(error),
            }
        }
        outcome
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let quoted = shell_quote(mount_path);
        // `fusermount` first, because it is FUSE-aware; then a plain `umount`; then a lazy one as
        // the last resort. A failed unmount is only logged, as in Python: the mount point may
        // already be gone, and teardown must not fail over it. The mount path is tool data, so
        // the messages carry it only when the tool-data logging policy allows.
        let attempts = [
            (
                format!("fusermount -u {quoted}"),
                "fusermount",
                "trying umount",
            ),
            (format!("umount {quoted}"), "umount", "trying lazy umount"),
            (format!("umount -l {quoted}"), "", ""),
        ];
        let mut last_exit = 0;
        for (index, (command, tool, next)) in attempts.iter().enumerate() {
            let exit_code = match run_shell(session, command, MOUNT_TIMEOUT_SECS).await {
                Ok(result) if result.ok() => return Ok(()),
                Ok(result) => result.exit_code,
                Err(_) => -1,
            };
            last_exit = exit_code;
            if index + 1 == attempts.len() {
                break;
            }
            if crate::debug::dont_log_tool_data() {
                log::debug!(
                    target: crate::logger::LOG_TARGET,
                    "{tool} failed (exit {exit_code}), {next}"
                );
            } else {
                log::debug!(
                    target: crate::logger::LOG_TARGET,
                    "{tool} failed for {mount_path} (exit {exit_code}), {next}"
                );
            }
        }
        if crate::debug::dont_log_tool_data() {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "all unmount attempts failed (last exit {last_exit})"
            );
        } else {
            log::warn!(
                target: crate::logger::LOG_TARGET,
                "all unmount attempts failed for {mount_path} (last exit {last_exit})"
            );
        }
        Ok(())
    }
}

/// The Blaxel Drive a sandbox mounts.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct BlaxelDriveMountConfig {
    /// The drive name.
    pub drive_name: String,
    /// The path inside the drive to expose.
    pub drive_path: String,
    /// Whether the drive is mounted read-only.
    pub read_only: bool,
}

impl BlaxelDriveMountConfig {
    /// Mount the whole of `drive_name`.
    pub fn new(drive_name: impl Into<String>) -> Self {
        BlaxelDriveMountConfig {
            drive_name: drive_name.into(),
            drive_path: "/".to_string(),
            read_only: false,
        }
    }

    /// Expose only `drive_path` from the drive.
    pub fn with_drive_path(mut self, drive_path: impl Into<String>) -> Self {
        self.drive_path = drive_path.into();
        self
    }

    /// Mount the drive read-only.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// The `POST /drives/mount` body for a mount path.
    pub fn to_request_body(&self, mount_path: &str) -> Value {
        let mut body = Map::new();
        body.insert(
            "driveName".to_string(),
            Value::String(self.drive_name.clone()),
        );
        body.insert(
            "mountPath".to_string(),
            Value::String(mount_path.to_string()),
        );
        body.insert(
            "drivePath".to_string(),
            Value::String(self.drive_path.clone()),
        );
        body.insert("readOnly".to_string(), Value::Bool(self.read_only));
        Value::Object(body)
    }

    /// The drive configuration carried by a [`BlaxelDriveMount`] entry.
    ///
    /// Mirrors Python's `BlaxelDriveMountStrategy._resolve_config`, which refuses any other
    /// mount entry.
    pub fn from_configured_mount(mount: &dyn ConfiguredMount) -> SandboxResult<Self> {
        mount.validate()?;
        mount
            .as_any()
            .downcast_ref::<BlaxelDriveMount>()
            .map(BlaxelDriveMount::config)
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    format!(
                        "blaxel drive mounts require a `{MOUNT_TYPE_BLAXEL_DRIVE}` entry, not \
                         `{}`",
                        mount.mount_type()
                    ),
                )
            })
    }
}

/// The Python `type` discriminator of a [`BlaxelDriveMount`].
pub const MOUNT_TYPE_BLAXEL_DRIVE: &str = "blaxel_drive_mount";

/// A Blaxel Drive described as a mount entry. Mirrors Python's `BlaxelDriveMount`.
///
/// A drive is a persistent network volume the Blaxel platform attaches itself, so unlike the
/// bucket mounts it names no in-container pattern and no Docker volume driver: the only way to
/// attach it is [`BlaxelDriveMountStrategy`], which [`mount`](Self::mount) pairs it with.
///
/// ```no_run
/// use openai_agents_sdk::extensions::sandbox::blaxel::BlaxelDriveMount;
/// use openai_agents_sdk::sandbox::entries::Entry;
///
/// let entry = Entry::mount(BlaxelDriveMount::new("my-drive").with_drive_path("/data").mount());
/// ```
///
/// Python's `drive_mount_path` has no field of its own here: the core [`Mount`] already carries
/// the path override, so `BlaxelDriveMount::new("d").mount().with_mount_path("/workspace/d")`
/// does the same job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlaxelDriveMount {
    /// The drive name.
    pub drive_name: String,
    /// The path inside the drive to expose.
    pub drive_path: String,
    /// Whether the drive is mounted read-only.
    pub read_only: bool,
}

impl BlaxelDriveMount {
    /// The whole of `drive_name`, mounted writable.
    pub fn new(drive_name: impl Into<String>) -> Self {
        BlaxelDriveMount {
            drive_name: drive_name.into(),
            drive_path: "/".to_string(),
            read_only: false,
        }
    }

    /// Expose only `drive_path` from the drive.
    pub fn with_drive_path(mut self, drive_path: impl Into<String>) -> Self {
        self.drive_path = drive_path.into();
        self
    }

    /// Mount the drive read-only.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// This entry as the configuration the drives API is called with.
    pub fn config(&self) -> BlaxelDriveMountConfig {
        BlaxelDriveMountConfig {
            drive_name: self.drive_name.clone(),
            drive_path: self.drive_path.clone(),
            read_only: self.read_only,
        }
    }

    /// A [`Mount`] attaching this drive through [`BlaxelDriveMountStrategy`].
    pub fn mount(&self) -> Mount {
        BlaxelDriveMountStrategy::mount(self.config())
    }
}

impl ConfiguredMount for BlaxelDriveMount {
    fn mount_type(&self) -> &'static str {
        MOUNT_TYPE_BLAXEL_DRIVE
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        // The platform attaches the drive over its own API; no mount tool runs in the sandbox.
        &[]
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.drive_name.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "blaxel drive mounts require a drive name",
            ));
        }
        Ok(())
    }

    fn credential_authority(
        &self,
        _pattern_type: &str,
        _has_config_file: bool,
    ) -> MountCredentialAuthority {
        // The sandbox's own control-plane credential attaches the drive; nothing reaches code
        // running inside the sandbox.
        MountCredentialAuthority::None
    }

    fn build_in_container_config(
        &self,
        _session_id: &str,
        pattern: &BuiltinMountPattern,
        _read_only: bool,
        _include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "`{MOUNT_TYPE_BLAXEL_DRIVE}` cannot be attached with the `{}` mount pattern; use \
                 `BlaxelDriveMountStrategy`",
                pattern.pattern_type()
            ),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Attaches a Blaxel Drive through the sandbox's drives API.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct BlaxelDriveMountStrategy {
    config: BlaxelDriveMountConfig,
}

impl BlaxelDriveMountStrategy {
    /// A strategy for one drive.
    pub fn new(config: BlaxelDriveMountConfig) -> Self {
        BlaxelDriveMountStrategy { config }
    }

    /// A [`Mount`] backed by this strategy.
    pub fn mount(config: BlaxelDriveMountConfig) -> Mount {
        let read_only = config.read_only;
        Mount::new(Arc::new(BlaxelDriveMountStrategy::new(config))).with_read_only(read_only)
    }

    /// The drive this strategy attaches.
    pub fn config(&self) -> &BlaxelDriveMountConfig {
        &self.config
    }

    /// A strategy attaching the drive a [`BlaxelDriveMount`] entry describes.
    pub fn for_mount(mount: &dyn ConfiguredMount) -> SandboxResult<Self> {
        Ok(BlaxelDriveMountStrategy::new(
            BlaxelDriveMountConfig::from_configured_mount(mount)?,
        ))
    }

    fn connection(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<Arc<super::sandbox::BlaxelConnection>> {
        if session.backend_id() != "blaxel" {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "blaxel drive mounts are not supported by the `{}` sandbox backend",
                    session.backend_id()
                ),
            ));
        }
        session
            .as_any()
            .and_then(|session| {
                session
                    .downcast_ref::<super::sandbox::BlaxelSandboxSession>()
                    .map(|session| session.connection().clone())
            })
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!(
                        "the Blaxel sandbox behind `{mount_path}` does not expose its sandbox \
                         connection"
                    ),
                )
            })
    }
}

#[async_trait]
impl MountStrategy for BlaxelDriveMountStrategy {
    fn strategy_type(&self) -> &str {
        "blaxel_drive"
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        if self.config.drive_name.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "blaxel drive mounts require a drive name",
            ));
        }
        Ok(())
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        // A drive is shared, durable storage outside the workspace archive.
        false
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let connection = self.connection(session, mount_path)?;
        connection
            .mount_drive(self.config.to_request_body(mount_path), mount_path)
            .await
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let connection = self.connection(session, mount_path)?;
        // As in Python, a drive that will not detach is logged, not raised: the platform
        // reclaims it with the sandbox, and a teardown error would hide the run's own result.
        if let Err(error) = connection.unmount_drive(mount_path).await {
            crate::logger::log_tool_action_warning(
                "Drive detach failed (non-fatal)",
                &error.message,
            );
        }
        Ok(())
    }
}

async fn run_shell(
    session: &dyn SandboxSession,
    command: &str,
    timeout_secs: u64,
) -> SandboxResult<crate::sandbox::types::ExecResult> {
    session
        .exec(
            ExecRequest::argv(["sh", "-c", command])
                .with_timeout(Some(std::time::Duration::from_secs(timeout_secs))),
        )
        .await
}

// -------------------------------------------------------------------------------------------
// Configured mount translation
// -------------------------------------------------------------------------------------------

impl BlaxelCloudBucketMountConfig {
    /// The Blaxel bucket configuration for one of the built-in remote storage mounts.
    ///
    /// Mirrors Python's `_build_mount_config`: `s3fs` covers S3, R2 and HMAC-authenticated GCS,
    /// and only a GCS mount without HMAC credentials falls through to `gcsfuse`.
    pub fn from_configured_mount(mount: &dyn ConfiguredMount) -> SandboxResult<Self> {
        mount.validate()?;
        let any = mount.as_any();

        if let Some(s3) = any.downcast_ref::<S3Mount>() {
            return Ok(BlaxelCloudBucketMountConfig {
                provider: BlaxelBucketProvider::S3,
                bucket: s3.bucket.clone(),
                read_only: true,
                access_key_id: s3.access_key_id.clone(),
                secret_access_key: s3.secret_access_key.clone(),
                session_token: s3.session_token.clone(),
                region: s3.region.clone(),
                endpoint_url: s3.endpoint_url.clone(),
                prefix: s3.prefix.clone(),
                service_account_key: None,
            });
        }

        if let Some(r2) = any.downcast_ref::<R2Mount>() {
            return Ok(BlaxelCloudBucketMountConfig {
                provider: BlaxelBucketProvider::R2,
                bucket: r2.bucket.clone(),
                read_only: true,
                access_key_id: r2.access_key_id.clone(),
                secret_access_key: r2.secret_access_key.clone(),
                session_token: None,
                region: None,
                endpoint_url: Some(r2.endpoint()),
                prefix: None,
                service_account_key: None,
            });
        }

        if let Some(gcs) = any.downcast_ref::<GcsMount>() {
            if gcs.uses_s3_compatible_api() {
                return Ok(BlaxelCloudBucketMountConfig {
                    provider: BlaxelBucketProvider::S3,
                    bucket: gcs.bucket.clone(),
                    read_only: true,
                    access_key_id: gcs.access_id.clone(),
                    secret_access_key: gcs.secret_access_key.clone(),
                    session_token: None,
                    region: gcs.region.clone(),
                    endpoint_url: Some(
                        gcs.endpoint_url
                            .clone()
                            .unwrap_or_else(|| GCS_S3_ENDPOINT.to_string()),
                    ),
                    prefix: gcs.prefix.clone(),
                    service_account_key: None,
                });
            }
            return Ok(BlaxelCloudBucketMountConfig {
                provider: BlaxelBucketProvider::Gcs,
                bucket: gcs.bucket.clone(),
                read_only: true,
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
                region: None,
                endpoint_url: None,
                prefix: gcs.prefix.clone(),
                service_account_key: gcs.service_account_credentials.clone(),
            });
        }

        Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "blaxel cloud bucket mounts do not support `{}`",
                mount.mount_type()
            ),
        ))
    }
}

impl BlaxelCloudBucketMountStrategy {
    /// A strategy attaching one of the built-in remote storage mounts.
    ///
    /// The returned configuration is read-only, because these backends take `read_only` from the
    /// configuration rather than from the [`Mount`]: chain `with_read_only(false)` to allow
    /// writes. Defaulting the other way would silently widen a mount.
    pub fn for_mount(mount: &dyn ConfiguredMount) -> SandboxResult<Self> {
        Ok(BlaxelCloudBucketMountStrategy::new(
            BlaxelCloudBucketMountConfig::from_configured_mount(mount)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_s3fs_mount_command() {
        let config = BlaxelCloudBucketMountConfig::s3("bucket")
            .with_credentials("AKIA", "secret")
            .with_region("eu-west-1")
            .with_prefix("/nested/");
        let command = config.mount_command("/workspace/data", Some("/workspace/.cred"));
        // Every character in the option list is shell-safe, so neither this port's quoting nor
        // Python's `shlex.quote` wraps it.
        assert_eq!(
            command,
            "s3fs bucket:/nested /workspace/data -o \
             allow_other,nonempty,passwd_file=/workspace/.cred,\
url=https://s3.eu-west-1.amazonaws.com,endpoint=eu-west-1,ro"
        );
    }

    #[test]
    fn builds_a_public_bucket_and_r2_command() {
        let public = BlaxelCloudBucketMountConfig::s3("open").mount_command("/mnt", None);
        assert!(public.contains("public_bucket=1"));
        let r2 = BlaxelCloudBucketMountConfig::r2("assets", "acct")
            .with_credentials("id", "secret")
            .mount_command("/mnt", Some("/c"));
        assert!(r2.contains("url=https://acct.r2.cloudflarestorage.com"));
        assert!(r2.contains("sigv4"));
    }

    #[test]
    fn builds_the_gcsfuse_mount_command() {
        let config = BlaxelCloudBucketMountConfig::gcs("bucket")
            .with_service_account_key("{}")
            .with_prefix("sub/dir")
            .with_read_only(true);
        let command = config.mount_command("/workspace/gcs", Some("/workspace/.key"));
        assert_eq!(
            command,
            "gcsfuse --key-file=/workspace/.key -o ro --only-dir=sub/dir bucket /workspace/gcs"
        );
        let anonymous = BlaxelCloudBucketMountConfig::gcs("bucket").mount_command("/mnt", None);
        assert!(anonymous.contains("--anonymous-access"));
    }

    #[test]
    fn builds_credential_file_contents() {
        let config = BlaxelCloudBucketMountConfig::s3("bucket")
            .with_credentials("id", "secret")
            .with_session_token("token");
        assert_eq!(
            config.credential_file_contents().as_deref(),
            Some("id:secret:token")
        );
        assert!(
            BlaxelCloudBucketMountConfig::s3("bucket")
                .credential_file_contents()
                .is_none()
        );
    }

    #[test]
    fn reports_credential_authority_only_when_credentials_are_set() {
        let anonymous =
            BlaxelCloudBucketMountStrategy::new(BlaxelCloudBucketMountConfig::s3("bucket"));
        assert_eq!(
            anonymous.credential_authority(),
            MountCredentialAuthority::None
        );
        let authenticated = BlaxelCloudBucketMountStrategy::new(
            BlaxelCloudBucketMountConfig::s3("bucket").with_credentials("id", "secret"),
        );
        assert_eq!(
            authenticated.credential_authority(),
            MountCredentialAuthority::MountScoped
        );
    }

    #[test]
    fn rejects_half_a_credential_pair() {
        let mut config = BlaxelCloudBucketMountConfig::s3("bucket");
        config.access_key_id = Some("id".to_string());
        assert_eq!(
            config.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
    }

    #[test]
    fn keeps_credentials_out_of_debug_output() {
        let config = BlaxelCloudBucketMountConfig::s3("bucket").with_credentials("id", "topsecret");
        let rendered = format!("{config:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("topsecret"));
    }

    #[test]
    fn builds_the_drive_mount_body() {
        let config = BlaxelDriveMountConfig::new("shared")
            .with_drive_path("/data")
            .with_read_only(true);
        assert_eq!(
            config.to_request_body("/workspace/shared"),
            serde_json::json!({
                "driveName": "shared",
                "mountPath": "/workspace/shared",
                "drivePath": "/data",
                "readOnly": true,
            })
        );
    }

    #[test]
    fn builds_a_drive_strategy_from_a_drive_mount_entry() {
        let entry = BlaxelDriveMount::new("shared")
            .with_drive_path("/data")
            .with_read_only(true);
        assert_eq!(entry.mount_type(), "blaxel_drive_mount");
        let strategy = BlaxelDriveMountStrategy::for_mount(&entry).unwrap();
        assert_eq!(strategy.config(), &entry.config());
        // The entry's own read-only flag reaches the `Mount`, as it does for the bucket mounts.
        assert!(entry.mount().read_only);
        assert!(!BlaxelDriveMount::new("shared").mount().read_only);
    }

    #[test]
    fn rejects_a_drive_mount_without_a_name_and_a_foreign_entry() {
        assert_eq!(
            BlaxelDriveMountStrategy::for_mount(&BlaxelDriveMount::new(""))
                .unwrap_err()
                .code,
            ErrorCode::MountConfigInvalid
        );
        let foreign = BlaxelDriveMountStrategy::for_mount(&S3Mount::new("bucket")).unwrap_err();
        assert_eq!(foreign.code, ErrorCode::MountConfigInvalid);
        assert!(foreign.message.contains("blaxel_drive_mount"), "{foreign}");
    }

    #[test]
    fn refuses_to_attach_a_drive_with_an_in_container_pattern() {
        // A drive names no in-container pattern, so the core strategy rejects it before any
        // mount command is built, the way Python's `validate_mount` does.
        let mount = crate::sandbox::mount_entries::ConfiguredMountStrategy::rclone(
            BlaxelDriveMount::new("shared"),
        )
        .mount();
        assert_eq!(
            mount.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
    }
}
