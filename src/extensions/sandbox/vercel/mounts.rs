//! Create-time-only S3 mounts for Vercel sandboxes. Port of
//! `agents.extensions.sandbox.vercel.mounts`.
//!
//! Vercel's released runtime images are Amazon Linux, so the mount is an in-sandbox command
//! rather than part of the sandbox specification: `mount-s3` (Mountpoint for Amazon S3) is
//! verified, installed when missing, and run under `sudo`. The mount topology is fixed when the
//! sandbox is created, so [`VercelCloudBucketMountStrategy::activate`] only runs while the
//! session is starting, and the session tracks which paths it attached.
//!
//! Credentials reach `mount-s3` as the command's environment, never in its argument vector:
//! the Vercel REST API carries a command's environment out of band, exactly as Python hands
//! `env=` to the provider SDK. A failing credentialed command reports a fixed message instead of
//! the provider's stderr, because that stderr is the one place a credential could surface.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mount_entries::{ConfiguredMount, S3Mount};
use crate::sandbox::mount_patterns::redact_sensitive_values;
use crate::sandbox::mounts::{Mount, MountCredentialAuthority, MountStrategy};
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{ExecResult, User};

use super::VercelSandboxSession;

/// The absolute path of the Mountpoint binary in Vercel's runtime images.
const MOUNTPOINT_BINARY: &str = "/usr/bin/mount-s3";
/// The RPM package providing [`MOUNTPOINT_BINARY`].
const MOUNTPOINT_PACKAGE: &str = "mount-s3";
/// The filesystem source `findmnt` reports for a Mountpoint mount.
const MOUNTPOINT_SOURCE: &str = "mountpoint-s3";
/// The oldest Mountpoint release this strategy drives.
const MOUNTPOINT_MINIMUM_VERSION: (u32, u32, u32) = (1, 21, 0);
/// How long the package manager is given to install Mountpoint.
const MOUNTPOINT_INSTALL_TIMEOUT: Duration = Duration::from_secs(300);
/// How long every other in-sandbox mount command is given.
const MOUNTPOINT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
/// What a failing credentialed command reports instead of its own stderr.
const CREDENTIALED_MOUNT_FAILURE_MESSAGE: &str = "sandbox provider command failed";

/// The error raised when the mount topology is changed after the sandbox was created.
fn fixed_topology() -> SandboxError {
    SandboxError::new(
        ErrorCode::MountConfigInvalid,
        "Vercel S3 mount topology is fixed when the sandbox is created; dynamic manifest \
         application is not supported",
    )
}

/// Recover the Vercel session a mount runs in.
///
/// Mirrors Python's `_require_vercel_session`: the strategy depends on the runtime image shipping
/// `mount-s3`, on this backend's `sudo` handling, and on the session's own mount bookkeeping.
fn vercel_session(session: &dyn SandboxSession) -> SandboxResult<&VercelSandboxSession> {
    session
        .as_any()
        .and_then(|any| any.downcast_ref::<VercelSandboxSession>())
        .ok_or_else(|| {
            SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "a Vercel cloud bucket mount requires a Vercel sandbox session, not the `{}` \
                     backend",
                    session.backend_id()
                ),
            )
        })
}

/// Runs the mount commands of one strategy inside one session.
///
/// Every command's stderr is redacted against the strategy's own credentials before it reaches an
/// error message, so a tool that echoes its environment cannot leak one.
struct MountCommands<'a> {
    session: &'a VercelSandboxSession,
    sensitive: Vec<String>,
}

impl<'a> MountCommands<'a> {
    fn new(session: &'a VercelSandboxSession, sensitive: Vec<String>) -> Self {
        MountCommands { session, sensitive }
    }

    /// Run one command, reporting only a transport failure.
    async fn run(&self, argv: &[&str], sudo: bool, timeout: Duration) -> SandboxResult<ExecResult> {
        let mut request = ExecRequest::argv(argv.iter().copied()).with_timeout(Some(timeout));
        if sudo {
            request = request.with_user(Some(User::new("root")));
        }
        self.session.exec(request).await
    }

    /// Run one command that has to succeed.
    async fn require(
        &self,
        argv: &[&str],
        sudo: bool,
        timeout: Duration,
    ) -> SandboxResult<ExecResult> {
        let result = self.run(argv, sudo, timeout).await?;
        if !result.ok() {
            return Err(self.failure(argv, &result));
        }
        Ok(result)
    }

    /// The error a failing command reports.
    fn failure(&self, argv: &[&str], result: &ExecResult) -> SandboxError {
        SandboxError::new(
            ErrorCode::MountFailed,
            format!(
                "`{}` failed in the Vercel sandbox with exit code {}: {}",
                argv.first().copied().unwrap_or_default(),
                result.exit_code,
                redact_sensitive_values(result.stderr_text().trim(), &self.sensitive),
            ),
        )
    }

    /// The stdout of a command, as trimmed text.
    fn text(result: &ExecResult) -> String {
        result.stdout_text().trim().to_string()
    }
}

/// Parse a `major.minor.patch` version string.
fn parse_mountpoint_version(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.trim().split('.');
    let version = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    parts.next().is_none().then_some(version)
}

/// Whether a reported version is one this strategy drives.
///
/// Mirrors Python: the major version has to match exactly, so a future incompatible release is
/// rejected rather than assumed compatible.
fn is_supported_version(version: Option<(u32, u32, u32)>) -> bool {
    version.is_some_and(|version| {
        version.0 == MOUNTPOINT_MINIMUM_VERSION.0 && version >= MOUNTPOINT_MINIMUM_VERSION
    })
}

/// The arguments that ask `rpm` for the installed Mountpoint version.
const RPM_VERSION_ARGS: [&str; 5] = [
    "/usr/bin/rpm",
    "--query",
    "--queryformat",
    "%{VERSION}",
    MOUNTPOINT_PACKAGE,
];

/// Make sure the sandbox has a usable Mountpoint, installing it when it does not.
///
/// Mirrors Python's `_ensure_mountpoint`.
async fn ensure_mountpoint(commands: &MountCommands<'_>) -> SandboxResult<()> {
    let probe = commands
        .run(&RPM_VERSION_ARGS, false, MOUNTPOINT_COMMAND_TIMEOUT)
        .await?;
    let mut version_text = MountCommands::text(&probe);
    let mut version = probe
        .ok()
        .then(|| parse_mountpoint_version(&version_text))
        .flatten();
    let binary = commands
        .run(
            &["/usr/bin/test", "-x", MOUNTPOINT_BINARY],
            false,
            MOUNTPOINT_COMMAND_TIMEOUT,
        )
        .await?;

    if !binary.ok() || !is_supported_version(version) {
        commands
            .require(
                &[
                    "/usr/bin/dnf",
                    "install",
                    "-y",
                    "--setopt=gpgcheck=1",
                    "fuse",
                    MOUNTPOINT_PACKAGE,
                ],
                true,
                MOUNTPOINT_INSTALL_TIMEOUT,
            )
            .await?;
        commands
            .require(
                &["/usr/bin/test", "-x", MOUNTPOINT_BINARY],
                false,
                MOUNTPOINT_COMMAND_TIMEOUT,
            )
            .await?;
        let probe = commands
            .require(&RPM_VERSION_ARGS, false, MOUNTPOINT_COMMAND_TIMEOUT)
            .await?;
        version_text = MountCommands::text(&probe);
        version = parse_mountpoint_version(&version_text);
    }

    if !is_supported_version(version) {
        let (major, minor, patch) = MOUNTPOINT_MINIMUM_VERSION;
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "unsupported Mountpoint for Amazon S3 version in the Vercel sandbox: `{}` is \
                 older than the required `{major}.{minor}.{patch}`",
                if version_text.is_empty() {
                    "<unknown>"
                } else {
                    version_text.as_str()
                },
            ),
        ));
    }
    Ok(())
}

/// Resolve `$1` the way the sandbox's kernel would, tolerating a leaf that does not exist yet.
///
/// Python installs a shell helper for this. This port has no helper installer, so it runs the
/// equivalent one-liner: `realpath -m` resolves every symlink in the existing prefix and leaves a
/// missing leaf in place, which is what the check needs both before and after `mkdir`.
async fn assert_canonical_mount_path(
    commands: &MountCommands<'_>,
    mount_path: &str,
) -> SandboxResult<()> {
    let resolved = commands
        .require(
            &["/usr/bin/realpath", "-m", "--", mount_path],
            false,
            MOUNTPOINT_COMMAND_TIMEOUT,
        )
        .await?;
    let resolved = MountCommands::text(&resolved);
    if resolved != mount_path {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "Vercel S3 mount paths must not resolve through symlinks: `{mount_path}` \
                 resolves to `{resolved}`"
            ),
        ));
    }
    Ok(())
}

/// Refuse to mount over a directory that already holds something.
///
/// Mirrors Python's `_assert_empty_mount_directory`: Mountpoint hides whatever is underneath, so
/// a non-empty directory would silently disappear for as long as the mount lives.
async fn assert_empty_mount_directory(
    commands: &MountCommands<'_>,
    mount_path: &str,
) -> SandboxResult<()> {
    let listing = commands
        .require(
            &[
                "/usr/bin/find",
                mount_path,
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-print",
                "-quit",
            ],
            false,
            MOUNTPOINT_COMMAND_TIMEOUT,
        )
        .await?;
    if !MountCommands::text(&listing).is_empty() {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("Vercel S3 mounts require an empty mount directory, and `{mount_path}` is not"),
        ));
    }
    Ok(())
}

/// The session user's numeric user and group ids.
///
/// Mirrors Python's `_command_user_ids`. The mount itself runs as root, so a writable mount has
/// to be told which user owns the files it exposes; the probe therefore runs unprivileged.
async fn command_user_ids(commands: &MountCommands<'_>) -> SandboxResult<(String, String)> {
    let uid = commands
        .require(&["/usr/bin/id", "-u"], false, MOUNTPOINT_COMMAND_TIMEOUT)
        .await?;
    let gid = commands
        .require(&["/usr/bin/id", "-g"], false, MOUNTPOINT_COMMAND_TIMEOUT)
        .await?;
    let uid = MountCommands::text(&uid);
    let gid = MountCommands::text(&gid);
    if !is_decimal(&uid) || !is_decimal(&gid) {
        return Err(SandboxError::new(
            ErrorCode::MountFailed,
            "Vercel returned a non-numeric user or group ID",
        ));
    }
    Ok((uid, gid))
}

/// Whether `value` is a non-empty run of decimal digits.
fn is_decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// Whether `mount_path` currently carries a Mountpoint filesystem.
///
/// Mirrors Python's `_is_mounted`: a filesystem that is not Mountpoint's is something this
/// strategy never created, so it refuses to unmount it rather than detaching a stranger's mount.
async fn is_mounted(commands: &MountCommands<'_>, mount_path: &str) -> SandboxResult<bool> {
    let argv = [
        "/usr/bin/findmnt",
        "--noheadings",
        "--output",
        "SOURCE",
        "--mountpoint",
        mount_path,
    ];
    let result = commands
        .run(&argv, false, MOUNTPOINT_COMMAND_TIMEOUT)
        .await?;
    // `findmnt` exits 1 when nothing is mounted there, which is an answer, not a failure.
    if result.exit_code == 1 {
        return Ok(false);
    }
    if !result.ok() {
        return Err(commands.failure(&argv, &result));
    }
    let source = MountCommands::text(&result);
    if source != MOUNTPOINT_SOURCE {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "refusing to manage an unexpected filesystem at the Vercel S3 mount path \
                 `{mount_path}`: expected `{MOUNTPOINT_SOURCE}`, found `{source}`"
            ),
        ));
    }
    Ok(true)
}

/// Detach a Mountpoint mount, failing when the result is ambiguous.
///
/// Mirrors Python's `_unmount_s3`.
async fn unmount_s3(commands: &MountCommands<'_>, mount_path: &str) -> SandboxResult<()> {
    if !is_mounted(commands, mount_path).await? {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("a tracked Vercel S3 mount is missing from its configured path `{mount_path}`"),
        ));
    }
    let argv = ["/usr/bin/umount", mount_path];
    let result = commands
        .run(&argv, true, MOUNTPOINT_COMMAND_TIMEOUT)
        .await?;
    if result.ok() {
        return Ok(());
    }
    if !is_mounted(commands, mount_path).await? {
        // A mount can move with a renamed ancestor, so disappearance after a failed `umount`
        // says nothing about where it went.
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!("the Vercel S3 mount state at `{mount_path}` became ambiguous during unmount"),
        ));
    }
    Err(commands.failure(&argv, &result))
}

/// Mounts an S3 bucket inside a Vercel sandbox with `mount-s3`. Mirrors
/// `agents.extensions.sandbox.vercel.mounts.VercelCloudBucketMountStrategy`.
///
/// The mount topology is fixed when the sandbox is created: the strategy refuses to attach
/// storage outside the session's own start, and the session records which paths it attached, so
/// a teardown can tell a mount it owns from one it never made. When the mount carries credentials
/// they reach the sandbox as the mount command's environment, so the mount reports
/// [`MountCredentialAuthority::MountScoped`] and the manifest has to acknowledge the exact mount
/// path before it activates.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct VercelCloudBucketMountStrategy {
    /// The bucket to mount.
    pub bucket: String,
    /// Optional key prefix inside the bucket.
    pub prefix: Option<String>,
    /// Optional AWS region.
    pub region: Option<String>,
    /// Optional S3-compatible endpoint.
    pub endpoint_url: Option<String>,
    /// Access key id, when the bucket needs credentials.
    pub access_key_id: Option<String>,
    /// Secret access key, when the bucket needs credentials.
    pub secret_access_key: Option<String>,
    /// Session token, for temporary credentials.
    pub session_token: Option<String>,
}

// The secret must never reach a log line through the `MountStrategy: Debug` bound.
impl std::fmt::Debug for VercelCloudBucketMountStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VercelCloudBucketMountStrategy")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .field("endpoint_url", &self.endpoint_url)
            .field(
                "access_key_id",
                &self.access_key_id.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "secret_access_key",
                &self.secret_access_key.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl VercelCloudBucketMountStrategy {
    /// A credential-free (anonymous) mount of `bucket`.
    pub fn new(bucket: impl Into<String>) -> Self {
        VercelCloudBucketMountStrategy {
            bucket: bucket.into(),
            ..Default::default()
        }
    }

    /// Mount only the keys under `prefix`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Pin the bucket's region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Use an S3-compatible endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Authenticate with static credentials.
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Add a session token, for temporary credentials.
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }

    /// A strategy attaching one of the built-in remote storage mounts.
    ///
    /// Mirrors Python's `_validate_s3_mount`: Vercel's runtime images mount buckets with
    /// `mount-s3`, so only an [`S3Mount`] fits. The mount's own `read_only` flag still governs
    /// the mount command, so it is not copied here.
    pub fn for_mount(mount: &dyn ConfiguredMount) -> SandboxResult<Self> {
        mount.validate()?;
        let Some(s3) = mount.as_any().downcast_ref::<S3Mount>() else {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "vercel cloud bucket mounts only support `s3_mount`, not `{}`",
                    mount.mount_type()
                ),
            ));
        };
        Ok(VercelCloudBucketMountStrategy {
            bucket: s3.bucket.clone(),
            prefix: s3.prefix.clone(),
            region: s3.region.clone(),
            endpoint_url: s3.endpoint_url.clone(),
            access_key_id: s3.access_key_id.clone(),
            secret_access_key: s3.secret_access_key.clone(),
            session_token: s3.session_token.clone(),
        })
    }

    /// Whether this mount hands credentials to the sandbox.
    fn authenticated(&self) -> bool {
        self.access_key_id.is_some() && self.secret_access_key.is_some()
    }

    /// The environment `mount-s3` reads its credentials from.
    ///
    /// Mirrors Python's `_runtime_s3_mount_environment`. The values travel as the command's
    /// environment, never in its argument vector, so they stay out of the sandbox's process
    /// table and out of the provider's command log.
    fn credential_env(&self) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        if let (Some(access_key_id), Some(secret_access_key)) =
            (&self.access_key_id, &self.secret_access_key)
        {
            env.insert("AWS_ACCESS_KEY_ID".to_string(), access_key_id.clone());
            env.insert(
                "AWS_SECRET_ACCESS_KEY".to_string(),
                secret_access_key.clone(),
            );
        }
        if let Some(session_token) = &self.session_token {
            env.insert("AWS_SESSION_TOKEN".to_string(), session_token.clone());
        }
        if let Some(region) = &self.region {
            env.insert("AWS_REGION".to_string(), region.clone());
        }
        env
    }

    /// Every credential value this mount carries, for redacting command output.
    fn sensitive_values(&self) -> Vec<String> {
        [
            self.access_key_id.as_ref(),
            self.secret_access_key.as_ref(),
            self.session_token.as_ref(),
        ]
        .into_iter()
        .flatten()
        .filter(|value| !value.is_empty())
        .cloned()
        .collect()
    }

    /// A [`Mount`] using this strategy.
    pub fn mount(self) -> Mount {
        Mount::new(Arc::new(self))
    }

    /// The `mount-s3` argument vector for `mount_path`.
    ///
    /// Mirrors Python's `_mount_args`. Write access is what decides the write flags: an
    /// authenticated mount that the manifest declared read-only must not be able to delete.
    fn mount_argv(
        &self,
        mount_path: &str,
        read_only: bool,
        user_ids: Option<&(String, String)>,
    ) -> Vec<String> {
        let mut argv = vec![
            MOUNTPOINT_BINARY.to_string(),
            self.bucket.clone(),
            mount_path.to_string(),
            "--allow-other".to_string(),
        ];
        if !self.authenticated() {
            argv.push("--no-sign-request".to_string());
        }
        if read_only {
            argv.push("--read-only".to_string());
        } else {
            argv.extend([
                "--allow-overwrite".to_string(),
                "--allow-delete".to_string(),
            ]);
            if let Some((uid, gid)) = user_ids {
                argv.extend([
                    "--uid".to_string(),
                    uid.clone(),
                    "--gid".to_string(),
                    gid.clone(),
                ]);
            }
        }
        if let Some(region) = &self.region {
            argv.extend(["--region".to_string(), region.clone()]);
        }
        if let Some(endpoint_url) = &self.endpoint_url {
            argv.extend(["--endpoint-url".to_string(), endpoint_url.clone()]);
        }
        if let Some(prefix) = &self.prefix {
            let prefix = if prefix.ends_with('/') {
                prefix.clone()
            } else {
                format!("{prefix}/")
            };
            argv.extend(["--prefix".to_string(), prefix]);
        }
        argv
    }

    /// Verify the mount path, install Mountpoint if needed, and run the mount command.
    ///
    /// Mirrors Python's `_mount_s3`, including the second canonical-path check: the first one
    /// runs before the directory exists, and `mkdir -p` can follow a symlink planted in between.
    async fn mount_s3(
        &self,
        commands: &MountCommands<'_>,
        mount_path: &str,
        read_only: bool,
    ) -> SandboxResult<()> {
        assert_canonical_mount_path(commands, mount_path).await?;
        ensure_mountpoint(commands).await?;
        commands
            .require(
                &["/usr/bin/mkdir", "-p", "--", mount_path],
                false,
                MOUNTPOINT_COMMAND_TIMEOUT,
            )
            .await?;
        assert_empty_mount_directory(commands, mount_path).await?;
        let user_ids = match read_only {
            true => None,
            false => Some(command_user_ids(commands).await?),
        };
        assert_canonical_mount_path(commands, mount_path).await?;
        self.run_mount_command(
            commands,
            self.mount_argv(mount_path, read_only, user_ids.as_ref()),
            mount_path,
        )
        .await
    }

    /// Run the credentialed mount command, reporting a fixed message when it fails.
    ///
    /// Mirrors Python's `_run_credentialed_mount_command`: the command's own stderr is the one
    /// output that has seen the credentials, so it never reaches the caller.
    async fn run_mount_command(
        &self,
        commands: &MountCommands<'_>,
        argv: Vec<String>,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let request = ExecRequest::argv(argv)
            .with_envs(self.credential_env())
            .with_user(Some(User::new("root")))
            .with_timeout(Some(MOUNTPOINT_COMMAND_TIMEOUT));
        let outcome = commands.session.exec(request).await;
        let failure = |detail: String| {
            SandboxError::new(
                ErrorCode::MountFailed,
                format!("`mount-s3` failed to mount `{mount_path}`: {detail}"),
            )
        };
        match outcome {
            Err(_) => Err(failure(CREDENTIALED_MOUNT_FAILURE_MESSAGE.to_string())),
            Ok(result) if !result.ok() => Err(failure(format!(
                "exit code {}: {CREDENTIALED_MOUNT_FAILURE_MESSAGE}",
                result.exit_code
            ))),
            Ok(_) => Ok(()),
        }
    }
}

#[async_trait]
impl MountStrategy for VercelCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "vercel_cloud_bucket"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn validate(&self, mount: &Mount) -> SandboxResult<()> {
        // Port of the `if not mount.ephemeral` branch of Python's `_validate_s3_mount`. Vercel
        // attaches the bucket with `mount-s3` at runtime and has nothing that could persist its
        // contents into a snapshot, so a mount asking to be durable is refused rather than
        // silently treated as ephemeral.
        if !mount.ephemeral {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a Vercel cloud bucket mount must be ephemeral",
            ));
        }
        if self.bucket.trim().is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a Vercel cloud bucket mount requires a bucket name",
            ));
        }
        for (name, value) in [
            ("access_key_id", self.access_key_id.as_ref()),
            ("secret_access_key", self.secret_access_key.as_ref()),
            ("session_token", self.session_token.as_ref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    format!("a Vercel cloud bucket mount's `{name}` must not be blank"),
                ));
            }
        }
        // A URL that carries userinfo, a query or a fragment may hold a signature or token, and
        // the endpoint reaches the mount command's argument vector, which is not a secret
        // channel. Refuse it rather than put it there.
        if let Some(endpoint) = &self.endpoint_url {
            let carries_authority = reqwest::Url::parse(endpoint).is_ok_and(|url| {
                url.query().is_some()
                    || url.fragment().is_some()
                    || !url.username().is_empty()
                    || url.password().is_some()
            });
            if carries_authority {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "the Vercel cloud bucket mount does not support exposing the configured \
                     credential fields; use a plain endpoint_url without a signature",
                ));
            }
        }
        if self.access_key_id.is_some() != self.secret_access_key.is_some() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a Vercel cloud bucket mount needs both `access_key_id` and `secret_access_key`",
            ));
        }
        if self.session_token.is_some() && self.access_key_id.is_none() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "a Vercel cloud bucket mount needs `access_key_id` and `secret_access_key` when \
                 a `session_token` is given",
            ));
        }
        Ok(())
    }

    fn credential_authority(&self) -> MountCredentialAuthority {
        if self.authenticated() {
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
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let vercel = vercel_session(session)?;
        // Vercel applies its mount topology while the sandbox starts; a manifest applied later
        // would hand a credential to a sandbox that has already run model-controlled code.
        if !vercel.s3_mount_activation_allowed() {
            return Err(fixed_topology());
        }
        session.authorize_mount_activation(mount, mount_path)?;
        vercel.assert_trusted_s3_mount(mount_path)?;
        let commands = MountCommands::new(vercel, self.sensitive_values());
        match self.mount_s3(&commands, mount_path, mount.read_only).await {
            Ok(()) => {
                vercel.record_s3_mount_active(mount_path);
                Ok(())
            }
            Err(error) => Err(vercel.fail_s3_mount_transition(error).await),
        }
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let vercel = vercel_session(session)?;
        if !vercel.s3_mount_is_active(mount_path) {
            return Ok(());
        }
        let commands = MountCommands::new(vercel, self.sensitive_values());
        match unmount_s3(&commands, mount_path).await {
            Ok(()) => {
                vercel.record_s3_mount_inactive(mount_path);
                Ok(())
            }
            Err(error) => Err(vercel.fail_s3_mount_transition(error).await),
        }
    }

    async fn teardown_for_snapshot(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let vercel = vercel_session(session)?;
        if !vercel.s3_mount_is_active(mount_path) {
            return Ok(());
        }
        let commands = MountCommands::new(vercel, self.sensitive_values());
        match unmount_s3(&commands, mount_path).await {
            Ok(()) => {
                vercel.record_s3_mount_detached(mount_path);
                Ok(())
            }
            Err(error) => Err(vercel.fail_s3_mount_transition(error).await),
        }
    }

    async fn restore_after_snapshot(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let vercel = vercel_session(session)?;
        // Only a mount this session detached for the snapshot may come back, which is what lets
        // the restore run outside the session's start.
        if !vercel.s3_mount_is_detached(mount_path) {
            return Ok(());
        }
        let commands = MountCommands::new(vercel, self.sensitive_values());
        match self.mount_s3(&commands, mount_path, mount.read_only).await {
            Ok(()) => {
                vercel.record_s3_mount_active(mount_path);
                Ok(())
            }
            Err(error) => Err(vercel.fail_s3_mount_transition(error).await),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_bounds_mountpoint_versions() {
        assert_eq!(parse_mountpoint_version(" 1.21.0\n"), Some((1, 21, 0)));
        assert_eq!(parse_mountpoint_version("1.21"), None);
        assert_eq!(parse_mountpoint_version("1.21.0.1"), None);
        assert_eq!(parse_mountpoint_version("package mount-s3 not"), None);

        assert!(is_supported_version(Some((1, 21, 0))));
        assert!(is_supported_version(Some((1, 22, 3))));
        assert!(!is_supported_version(Some((1, 20, 9))));
        // A new major release is not assumed compatible.
        assert!(!is_supported_version(Some((2, 0, 0))));
        assert!(!is_supported_version(None));
    }

    #[test]
    fn ties_the_write_flags_to_writability_not_authentication() {
        let authenticated = VercelCloudBucketMountStrategy::new("my-bucket")
            .with_credentials("AKIAEXAMPLE", "s3cr3t");
        // A read-only mount never gets the write flags, however it authenticates.
        let read_only = authenticated.mount_argv("/vercel/sandbox/data", true, None);
        assert!(!read_only.iter().any(|arg| arg == "--allow-delete"));
        assert!(!read_only.iter().any(|arg| arg == "--allow-overwrite"));
        assert!(read_only.iter().any(|arg| arg == "--read-only"));
        assert!(!read_only.iter().any(|arg| arg == "--no-sign-request"));

        let ids = ("1000".to_string(), "1001".to_string());
        let writable = authenticated.mount_argv("/vercel/sandbox/data", false, Some(&ids));
        assert_eq!(
            writable,
            vec![
                "/usr/bin/mount-s3",
                "my-bucket",
                "/vercel/sandbox/data",
                "--allow-other",
                "--allow-overwrite",
                "--allow-delete",
                "--uid",
                "1000",
                "--gid",
                "1001",
            ]
        );
    }

    #[test]
    fn builds_mount_s3_arguments() {
        let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
            .with_prefix("data")
            .with_region("us-east-1");
        assert_eq!(
            strategy.mount_argv("/vercel/sandbox/data", true, None),
            vec![
                "/usr/bin/mount-s3",
                "my-bucket",
                "/vercel/sandbox/data",
                "--allow-other",
                "--no-sign-request",
                "--read-only",
                "--region",
                "us-east-1",
                "--prefix",
                "data/",
            ]
        );
    }

    #[test]
    fn keeps_bucket_credentials_out_of_the_command_line() {
        let strategy = VercelCloudBucketMountStrategy::new("my-bucket")
            .with_region("us-east-1")
            .with_credentials("AKIAEXAMPLE", "s3cr3t")
            .with_session_token("token");
        let argv = strategy.mount_argv("/vercel/sandbox/data", false, None);
        let command = argv.join(" ");
        assert!(!command.contains("s3cr3t"), "{command}");
        assert!(!command.contains("AKIAEXAMPLE"), "{command}");
        // An authenticated mount signs its requests; an anonymous one must not try.
        assert!(!command.contains("--no-sign-request"), "{command}");

        let env = strategy.credential_env();
        assert_eq!(env["AWS_ACCESS_KEY_ID"], "AKIAEXAMPLE");
        assert_eq!(env["AWS_SECRET_ACCESS_KEY"], "s3cr3t");
        assert_eq!(env["AWS_SESSION_TOKEN"], "token");
        assert_eq!(env["AWS_REGION"], "us-east-1");
        assert_eq!(
            strategy.sensitive_values(),
            vec!["AKIAEXAMPLE".to_string(), "s3cr3t".into(), "token".into()]
        );

        let debug = format!("{strategy:?}");
        assert!(!debug.contains("s3cr3t"), "{debug}");
        assert!(debug.contains("<redacted>"));

        // An anonymous mount carries no credentials at all.
        let anonymous = VercelCloudBucketMountStrategy::new("public-bucket");
        assert!(anonymous.credential_env().is_empty());
        assert_eq!(
            anonymous.credential_authority(),
            MountCredentialAuthority::None
        );
        assert!(
            anonymous
                .mount_argv("/vercel/sandbox/data", true, None)
                .contains(&"--no-sign-request".to_string())
        );
    }

    #[test]
    fn refuses_a_mount_that_asks_not_to_be_ephemeral() {
        let strategy = VercelCloudBucketMountStrategy::new("my-bucket");
        let ephemeral = Mount::new(Arc::new(strategy.clone()));
        // The default stands: a mount is ephemeral unless the caller says otherwise.
        assert!(ephemeral.ephemeral);
        assert!(strategy.validate(&ephemeral).is_ok());

        // Port of Python's `if not mount.ephemeral: raise MountConfigError`.
        let durable = Mount::new(Arc::new(strategy.clone())).with_ephemeral(false);
        let err = strategy.validate(&durable).unwrap_err();
        assert_eq!(err.code, ErrorCode::MountConfigInvalid);
        assert!(err.to_string().contains("ephemeral"), "{err}");

        // The signal reaches the strategy through `Mount::validate` too, which is the path the
        // manifest actually takes.
        assert_eq!(
            durable.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
    }

    #[test]
    fn rejects_half_configured_credentials() {
        let mount = Mount::new(Arc::new(VercelCloudBucketMountStrategy::new("b")));
        let lone_key = VercelCloudBucketMountStrategy {
            access_key_id: Some("AKIAEXAMPLE".into()),
            ..VercelCloudBucketMountStrategy::new("my-bucket")
        };
        assert_eq!(
            lone_key.validate(&mount).unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
        let blank = VercelCloudBucketMountStrategy::new("my-bucket").with_credentials("  ", "s");
        assert_eq!(
            blank.validate(&mount).unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
        let lone_token =
            VercelCloudBucketMountStrategy::new("my-bucket").with_session_token("token");
        assert_eq!(
            lone_token.validate(&mount).unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
        assert!(
            VercelCloudBucketMountStrategy::new("my-bucket")
                .with_credentials("AKIAEXAMPLE", "s3cr3t")
                .with_session_token("token")
                .validate(&mount)
                .is_ok()
        );
    }
}
