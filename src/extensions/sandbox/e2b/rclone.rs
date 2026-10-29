//! The `rclone` FUSE mount pattern shared by the cloud sandbox providers. Mirrors the parts of
//! `agents.sandbox.entries.mounts.patterns.RcloneMountPattern` and
//! `agents.extensions.sandbox._rclone` that a provider needs to attach a cloud bucket.
//!
//! The pattern runs entirely inside the sandbox: it writes the rclone configuration to a file in
//! the workspace, mounts `remote:path` with `rclone mount --daemon`, and unmounts with
//! `fusermount`. Because the configuration carries the bucket credentials, an
//! [`InContainerMountStrategy`] built on it reports
//! [`MountCredentialAuthority::MountScoped`](crate::sandbox::mounts::MountCredentialAuthority),
//! so the application has to acknowledge the exact mount path on the manifest before the mount
//! activates.
//!
//! [`InContainerMountStrategy`]: crate::sandbox::mounts::InContainerMountStrategy

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mounts::{MountPattern, MountPatternConfig};
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession, shell_quote};
use crate::sandbox::types::User;
use sha2::{Digest, Sha256};

/// `MountPatternConfig` option holding the rclone remote name (`[section]` in the config).
pub const OPTION_REMOTE_NAME: &str = "remote_name";
/// `MountPatternConfig` option holding the path inside the remote, for example a bucket name.
pub const OPTION_REMOTE_PATH: &str = "remote_path";
/// `MountPatternConfig` option set to `true` when the mount must be read-only.
pub const OPTION_READ_ONLY: &str = "read_only";
/// `MountPatternConfig` option holding the uid `rclone` presents files as.
pub const OPTION_UID: &str = "uid";
/// `MountPatternConfig` option holding the gid `rclone` presents files as.
pub const OPTION_GID: &str = "gid";

/// Prefix of the generated rclone configuration file names.
///
/// Workspace archives exclude anything matching `<prefix>*`, so a credential staged for a mount
/// can never be captured in a snapshot.
pub const CONFIG_FILE_PREFIX: &str = ".openai-agents-rclone-";

/// Shell test that finds `rclone` on `PATH` or at the location the installers use.
const RCLONE_CHECK: &str = "command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone";

/// How `rclone` exposes the remote inside the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RcloneMountMode {
    /// Mount the remote with `rclone mount`, which needs FUSE.
    #[default]
    Fuse,
}

/// Mounts a cloud bucket inside the sandbox with `rclone mount`.
///
/// The secret-bearing configuration comes from
/// [`MountPatternConfig::config_text`](crate::sandbox::mounts::MountPatternConfig::config_text);
/// it is written under the workspace root at [`config_path`](Self::config_path) with mode `600`
/// before the mount runs, and removed when the mount is detached. The non-secret options listed
/// in this module's constants select the remote and the mount flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RcloneMountPattern {
    /// How the remote is exposed.
    pub mode: RcloneMountMode,
    /// Workspace-relative path of the rclone configuration file inside the sandbox.
    ///
    /// Like Python's `RcloneMountPattern.config_file_path`, a relative path resolves inside the
    /// sandbox workspace rather than on the host running the SDK. `None` derives a distinct name
    /// per mount path, so two mounts cannot overwrite each other's configuration.
    pub config_path: Option<String>,
    /// Extra arguments appended to `rclone mount`.
    pub extra_args: Vec<String>,
    /// Whether a missing `rclone` may be installed with the sandbox's package manager.
    pub install_if_missing: bool,
}

impl Default for RcloneMountPattern {
    fn default() -> Self {
        RcloneMountPattern {
            mode: RcloneMountMode::Fuse,
            config_path: None,
            extra_args: Vec::new(),
            install_if_missing: true,
        }
    }
}

impl RcloneMountPattern {
    /// A FUSE pattern with the default configuration path.
    pub fn new() -> Self {
        RcloneMountPattern::default()
    }

    /// Write the configuration to an explicit workspace-relative path.
    pub fn with_config_path(mut self, config_path: impl Into<String>) -> Self {
        self.config_path = Some(config_path.into());
        self
    }

    /// The workspace-relative configuration path used for `mount_path`.
    fn config_path_for(&self, mount_path: &str) -> String {
        self.config_path.clone().unwrap_or_else(|| {
            // A distinct file per mount path, so concurrent mounts never share one config.
            let digest = Sha256::digest(mount_path.as_bytes());
            format!(
                "{CONFIG_FILE_PREFIX}{}.conf",
                digest
                    .iter()
                    .take(8)
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )
        })
    }

    /// Append an argument to `rclone mount`.
    pub fn with_extra_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// Whether a missing `rclone` may be installed from the sandbox's package manager.
    pub fn with_install_if_missing(mut self, install: bool) -> Self {
        self.install_if_missing = install;
        self
    }

    /// The `rclone mount` argument vector for `mount_path`.
    fn mount_argv(
        &self,
        mount_path: &str,
        config_path: &str,
        config: &MountPatternConfig,
    ) -> SandboxResult<Vec<String>> {
        let remote_name = config.options.get(OPTION_REMOTE_NAME).ok_or_else(|| {
            SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!("an rclone mount requires the `{OPTION_REMOTE_NAME}` option"),
            )
        })?;
        let remote_path = config
            .options
            .get(OPTION_REMOTE_PATH)
            .map(String::as_str)
            .unwrap_or_default();
        let mut argv = vec![
            "rclone".to_string(),
            "mount".to_string(),
            format!("{remote_name}:{remote_path}"),
            mount_path.to_string(),
            "--config".to_string(),
            config_path.to_string(),
            "--daemon".to_string(),
            "--allow-other".to_string(),
        ];
        if config.options.get(OPTION_READ_ONLY).map(String::as_str) == Some("true") {
            argv.push("--read-only".to_string());
        }
        if let Some(uid) = config.options.get(OPTION_UID) {
            argv.extend(["--uid".to_string(), uid.clone()]);
        }
        if let Some(gid) = config.options.get(OPTION_GID) {
            argv.extend(["--gid".to_string(), gid.clone()]);
        }
        argv.extend(self.extra_args.iter().cloned());
        Ok(argv)
    }
}

/// Run a shell snippet as root, as the mount preflight steps need to.
async fn root_shell(
    session: &dyn SandboxSession,
    script: &str,
    timeout_secs: u64,
) -> SandboxResult<crate::sandbox::types::ExecResult> {
    session
        .exec(
            ExecRequest::argv(["sh", "-lc", script])
                .with_shell(ExecShell::None)
                .with_timeout(Some(std::time::Duration::from_secs(timeout_secs)))
                .with_user(Some(User::new("root"))),
        )
        .await
}

/// Whether `command` resolves inside the sandbox.
async fn has_command(session: &dyn SandboxSession, command: &str) -> SandboxResult<bool> {
    let script =
        format!("command -v {command} >/dev/null 2>&1 || test -x /usr/local/bin/{command}");
    let result = session
        .exec(ExecRequest::argv(["sh", "-lc", &script]).with_shell(ExecShell::None))
        .await?;
    Ok(result.ok())
}

/// Install `package` with whichever package manager the sandbox image provides.
///
/// Mirrors `agents.extensions.sandbox.daytona.mounts._pkg_install`. Unlike Python's E2B helper
/// this does not download a pinned `rclone` release: the distribution package is used instead,
/// so the sandbox image's own signed repositories stay the trust root.
pub async fn install_package(
    session: &dyn SandboxSession,
    package: &str,
    what: &str,
) -> SandboxResult<()> {
    let script = if has_command(session, "apt-get").await? {
        format!(
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {package}"
        )
    } else if has_command(session, "apk").await? {
        format!("apk add --no-cache {package}")
    } else {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            format!(
                "{what} is not installed and no supported package manager was found; preinstall \
                 `{package}` in the sandbox image"
            ),
        ));
    };
    let mut last_exit = 0;
    for _ in 0..3 {
        let result = root_shell(session, &script, 300).await?;
        if result.ok() {
            return Ok(());
        }
        last_exit = result.exit_code;
    }
    Err(SandboxError::new(
        ErrorCode::MountMissingTool,
        format!("failed to install `{package}` for {what} (exit code {last_exit})"),
    ))
}

/// Make sure `rclone` is available inside the sandbox.
///
/// Mirrors `agents.extensions.sandbox._rclone.ensure_rclone`.
pub async fn ensure_rclone(
    session: &dyn SandboxSession,
    install_if_missing: bool,
) -> SandboxResult<()> {
    let present = session
        .exec(ExecRequest::argv(["sh", "-lc", RCLONE_CHECK]).with_shell(ExecShell::None))
        .await?;
    if present.ok() {
        return Ok(());
    }
    if !install_if_missing {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            "rclone is not installed in the sandbox image",
        ));
    }
    install_package(session, "rclone", "rclone").await?;
    let present = session
        .exec(ExecRequest::argv(["sh", "-lc", RCLONE_CHECK]).with_shell(ExecShell::None))
        .await?;
    if !present.ok() {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            "rclone was installed but is still not on PATH",
        ));
    }
    Ok(())
}

/// Verify the sandbox can run FUSE mounts, and make `/dev/fuse` usable by non-root users.
///
/// Mirrors `agents.extensions.sandbox.e2b.mounts._ensure_fuse_support` and the Daytona variant
/// that installs `fuse3` when only the userspace tooling is missing.
pub async fn ensure_fuse_support(
    session: &dyn SandboxSession,
    install_if_missing: bool,
) -> SandboxResult<()> {
    let kernel = session
        .exec(
            ExecRequest::argv([
                "sh",
                "-lc",
                "test -c /dev/fuse && grep -qw fuse /proc/filesystems",
            ])
            .with_shell(ExecShell::None),
        )
        .await?;
    if !kernel.ok() {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            "cloud bucket mounts require `/dev/fuse` and the `fuse` filesystem in this sandbox",
        ));
    }
    let fusermount =
        has_command(session, "fusermount3").await? || has_command(session, "fusermount").await?;
    if !fusermount {
        if !install_if_missing {
            return Err(SandboxError::new(
                ErrorCode::MountMissingTool,
                "cloud bucket mounts require `fusermount`, which is missing from the sandbox image",
            ));
        }
        install_package(session, "fuse3", "fusermount").await?;
        if !has_command(session, "fusermount3").await?
            && !has_command(session, "fusermount").await?
        {
            return Err(SandboxError::new(
                ErrorCode::MountMissingTool,
                "`fuse3` was installed but `fusermount` is still missing",
            ));
        }
    }
    // `--allow-other` needs both a writable `/dev/fuse` and `user_allow_other` in `fuse.conf`.
    let allow_other = root_shell(
        session,
        "chmod a+rw /dev/fuse && touch /etc/fuse.conf && (grep -qxF user_allow_other \
         /etc/fuse.conf || printf '\\nuser_allow_other\\n' >> /etc/fuse.conf)",
        30,
    )
    .await?;
    if !allow_other.ok() {
        return Err(SandboxError::new(
            ErrorCode::MountConfigInvalid,
            format!(
                "failed to make `/dev/fuse` accessible (exit code {})",
                allow_other.exit_code
            ),
        ));
    }
    Ok(())
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
        // A relative configuration path resolves inside the workspace, so the session's own file
        // API can write it; a path outside the workspace root would be rejected.
        let config_path = session.normalize_path(&self.config_path_for(mount_path))?;
        let argv = self.mount_argv(mount_path, &config_path, config)?;
        let config_text = config.config_text().ok_or_else(|| {
            SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "an rclone mount requires the remote configuration text",
            )
        })?;
        // The configuration holds the bucket credentials: write it before the mount runs and keep
        // it out of the process table by streaming it through the file API rather than a command.
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
                ExecRequest::argv(argv.clone())
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
                    "`rclone mount` failed for `{mount_path}` with exit code {}: {}",
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
        _config: &MountPatternConfig,
    ) -> SandboxResult<()> {
        let quoted = shell_quote(mount_path);
        let script = format!(
            "fusermount3 -u -- {quoted} 2>/dev/null || fusermount -u -- {quoted} 2>/dev/null || \
             umount -- {quoted} 2>/dev/null || true"
        );
        session
            .exec(
                ExecRequest::argv(["sh", "-lc", script.as_str()])
                    .with_shell(ExecShell::None)
                    .with_timeout(Some(std::time::Duration::from_secs(60))),
            )
            .await?;
        // The credentials must not outlive the mount.
        if let Ok(config_path) = session.normalize_path(&self.config_path_for(mount_path)) {
            let _ = session.rm(&config_path, false).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_mount_arguments() {
        let pattern = RcloneMountPattern::new()
            .with_config_path(".rclone.conf")
            .with_extra_arg("--vfs-cache-mode=off");
        let config = MountPatternConfig::new()
            .with_config_text("[bucket]\ntype = s3\n")
            .with_option(OPTION_REMOTE_NAME, "bucket")
            .with_option(OPTION_REMOTE_PATH, "my-bucket/prefix")
            .with_option(OPTION_READ_ONLY, "true")
            .with_option(OPTION_UID, "1000")
            .with_option(OPTION_GID, "1000");
        let argv = pattern
            .mount_argv("/workspace/data", "/workspace/.rclone.conf", &config)
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "rclone",
                "mount",
                "bucket:my-bucket/prefix",
                "/workspace/data",
                "--config",
                "/workspace/.rclone.conf",
                "--daemon",
                "--allow-other",
                "--read-only",
                "--uid",
                "1000",
                "--gid",
                "1000",
                "--vfs-cache-mode=off",
            ]
        );
    }

    #[test]
    fn derives_a_distinct_configuration_path_per_mount() {
        let pattern = RcloneMountPattern::new();
        let first = pattern.config_path_for("/workspace/a");
        let second = pattern.config_path_for("/workspace/b");
        assert_ne!(first, second);
        assert!(first.starts_with(CONFIG_FILE_PREFIX), "{first}");
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

    #[test]
    fn requires_a_remote_name() {
        let pattern = RcloneMountPattern::new();
        let config = MountPatternConfig::new().with_config_text("[bucket]\n");
        let error = pattern
            .mount_argv("/workspace/data", "/workspace/rclone.conf", &config)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }
}
