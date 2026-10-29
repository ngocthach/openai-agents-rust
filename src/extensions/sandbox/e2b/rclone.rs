//! Provisioning the tools an `rclone` cloud bucket mount needs inside a sandbox. Mirrors
//! `agents.extensions.sandbox._rclone` and the preflight halves of the E2B and Daytona
//! `mounts.py`.
//!
//! The mount itself is [`RcloneMountPattern`](crate::sandbox::mounts::RcloneMountPattern) in the
//! core, which only checks that `rclone` is present. Installing it is provider-specific, so it
//! lives here: E2B downloads the release Python pins and verifies its SHA-256, while Daytona
//! installs the distribution package.

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession};
use crate::sandbox::types::User;

/// Prefix of the generated rclone configuration file names.
///
/// Workspace archives exclude anything matching `<prefix>*`, so a credential staged for a mount
/// can never be captured in a snapshot.
pub const CONFIG_FILE_PREFIX: &str = ".openai-agents-rclone-";

/// Shell test that finds `rclone` on `PATH` or at the location the installers use.
const RCLONE_CHECK: &str = "command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone";

// The pinned release and its published checksums, kept in step with
// `agents.extensions.sandbox._rclone`'s "RCLONE RELEASE PIN" block.
/// The `rclone` release this crate installs when the sandbox image has none.
pub const RCLONE_VERSION: &str = "1.74.4";
/// Published SHA-256 of each pinned `rclone` release archive, by release architecture.
const RCLONE_SHA256_BY_ARCH: &[(&str, &str)] = &[
    (
        "386",
        "7feee086d7ff72652c5a91ef4b4a576941ccd33b2929772a2d70471904e516f0",
    ),
    (
        "amd64",
        "fe435e0c36228e7c2f116a8701f01127bb1f694005fc11d1f27186c8bca4115d",
    ),
    (
        "arm",
        "8135524b9b85111fa512f10a3fa191736a8d4d6ac3b3169af0763503744e95c9",
    ),
    (
        "arm-v6",
        "c9e1048feb597938884c0fff314d5d9a002599933cb94ce17fee19599cbfa3f1",
    ),
    (
        "arm-v7",
        "75844809d25d2534da96220727e7746a300e30ec8c676ca98c47affe5a752e7b",
    ),
    (
        "arm64",
        "97685285c9ad6a0cf17d5844115d2a67245af6444db672187074bd9c358de419",
    ),
];
/// Exit code the install script uses to report a checksum mismatch.
const RCLONE_CHECKSUM_MISMATCH_EXIT: i32 = 86;
/// Packages the pinned-release download needs.
const INSTALL_RCLONE_PREREQUISITES: &[&str] = &[
    "DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes apt-get -o Dpkg::Use-Pty=0 update -qq",
    "DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes apt-get -o Dpkg::Use-Pty=0 install -y \
     -qq ca-certificates coreutils curl unzip",
];

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

/// Map a `uname -m` value onto an `rclone` release architecture.
///
/// Mirrors `agents.extensions.sandbox._rclone._rclone_arch`.
fn rclone_arch(machine: &str) -> Option<&'static str> {
    let normalized = machine.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "x86_64" | "amd64" => Some("amd64"),
        "x86" => Some("386"),
        "aarch64" | "arm64" => Some("arm64"),
        _ => {
            // `i386` through `i686`.
            let bytes = normalized.as_bytes();
            if bytes.len() == 4
                && bytes[0] == b'i'
                && matches!(bytes[1], b'3' | b'4' | b'5' | b'6')
                && &normalized[2..] == "86"
            {
                return Some("386");
            }
            if normalized.starts_with("armv7") {
                return Some("arm-v7");
            }
            if normalized.starts_with("armv6") {
                return Some("arm-v6");
            }
            if normalized.starts_with("arm") {
                return Some("arm");
            }
            None
        }
    }
}

/// The published SHA-256 of the pinned `rclone` release for `arch`.
fn rclone_sha256(arch: &str) -> Option<&'static str> {
    RCLONE_SHA256_BY_ARCH
        .iter()
        .find(|(name, _)| *name == arch)
        .map(|(_, digest)| *digest)
}

/// The shell script that downloads, verifies and installs the pinned `rclone` release.
///
/// Mirrors `agents.extensions.sandbox._rclone._rclone_install_command`: the archive's SHA-256 is
/// checked before anything is installed, and the binary is moved into place only after it
/// reports the expected version.
fn rclone_install_script(arch: &str, sha256: &str) -> String {
    let archive = format!("rclone-v{RCLONE_VERSION}-linux-{arch}.zip");
    let url = format!("https://downloads.rclone.org/v{RCLONE_VERSION}/{archive}");
    [
        "set -eu".to_string(),
        "tmp_dir=\"$(mktemp -d)\"".to_string(),
        "target_tmp=\"\"".to_string(),
        "cleanup() {".to_string(),
        "    rm -rf \"$tmp_dir\"".to_string(),
        "    if [ -n \"$target_tmp\" ]; then rm -f \"$target_tmp\"; fi".to_string(),
        "}".to_string(),
        "trap cleanup EXIT".to_string(),
        "trap 'exit 1' HUP INT TERM".to_string(),
        format!("archive='{archive}'"),
        format!("expected_sha256='{sha256}'"),
        format!("url='{url}'"),
        "curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 --output \
         \"$tmp_dir/$archive\" \"$url\""
            .to_string(),
        "if ! printf '%s  %s\\n' \"$expected_sha256\" \"$tmp_dir/$archive\" | sha256sum \
         --check --strict -; then"
            .to_string(),
        format!("    exit {RCLONE_CHECKSUM_MISMATCH_EXIT}"),
        "fi".to_string(),
        "unzip -q \"$tmp_dir/$archive\" -d \"$tmp_dir/unpacked\"".to_string(),
        "install -d -m 0755 /usr/local/bin".to_string(),
        "target_tmp=\"$(mktemp /usr/local/bin/.rclone.XXXXXX)\"".to_string(),
        "install -m 0755 \"$tmp_dir/unpacked/${archive%.zip}/rclone\" \"$target_tmp\"".to_string(),
        "version_output=\"$(\"$target_tmp\" version)\"".to_string(),
        format!(
            "printf '%s\\n' \"$version_output\" | head -n 1 | grep -Fx 'rclone v{RCLONE_VERSION}'"
        ),
        "mv -f \"$target_tmp\" /usr/local/bin/rclone".to_string(),
        "target_tmp=\"\"".to_string(),
    ]
    .join("\n")
}

/// Make sure `rclone` is available inside the sandbox.
///
/// Mirrors `agents.extensions.sandbox._rclone.ensure_rclone`: a missing `rclone` is installed
/// from the pinned release, whose SHA-256 is verified inside the sandbox before the binary is
/// moved into `/usr/local/bin`. `apt-get` provides the download prerequisites, so an image
/// without it must ship `rclone` already.
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
    if !has_command(session, "apt-get").await? {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            "rclone is not installed and `apt-get` is unavailable; preinstall rclone in the \
             sandbox image",
        ));
    }
    let machine = session
        .exec(
            ExecRequest::argv(["uname", "-m"])
                .with_timeout(Some(std::time::Duration::from_secs(30))),
        )
        .await?;
    let arch = machine
        .ok()
        .then(|| rclone_arch(&machine.stdout_text()))
        .flatten()
        .ok_or_else(|| {
            SandboxError::new(
                ErrorCode::MountMissingTool,
                format!(
                    "rclone is not installed and the sandbox architecture `{}` has no pinned \
                     release",
                    machine.stdout_text().trim()
                ),
            )
        })?;
    let sha256 = rclone_sha256(arch).ok_or_else(|| {
        SandboxError::new(
            ErrorCode::MountMissingTool,
            format!("no pinned rclone release is recorded for `{arch}`"),
        )
    })?;

    for command in INSTALL_RCLONE_PREREQUISITES {
        let result = root_shell(session, command, 300).await?;
        if !result.ok() {
            return Err(SandboxError::new(
                ErrorCode::MountMissingTool,
                format!(
                    "failed to install the rclone download prerequisites (exit code {})",
                    result.exit_code
                ),
            ));
        }
    }
    let install = root_shell(session, &rclone_install_script(arch, sha256), 300).await?;
    if install.exit_code == RCLONE_CHECKSUM_MISMATCH_EXIT {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            format!(
                "the rclone v{RCLONE_VERSION} archive for `{arch}` failed SHA-256 verification"
            ),
        ));
    }
    if !install.ok() {
        return Err(SandboxError::new(
            ErrorCode::MountMissingTool,
            format!(
                "failed to install rclone v{RCLONE_VERSION} for `{arch}` (exit code {})",
                install.exit_code
            ),
        ));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_machine_names_to_release_architectures() {
        assert_eq!(rclone_arch("x86_64"), Some("amd64"));
        assert_eq!(rclone_arch(" AMD64 "), Some("amd64"));
        assert_eq!(rclone_arch("aarch64"), Some("arm64"));
        assert_eq!(rclone_arch("i686"), Some("386"));
        assert_eq!(rclone_arch("armv7l"), Some("arm-v7"));
        assert_eq!(rclone_arch("armv6l"), Some("arm-v6"));
        assert_eq!(rclone_arch("armv5"), Some("arm"));
        assert_eq!(rclone_arch("riscv64"), None);
    }

    #[test]
    fn pins_a_checksum_for_every_supported_architecture() {
        for arch in ["386", "amd64", "arm", "arm-v6", "arm-v7", "arm64"] {
            let digest = rclone_sha256(arch).unwrap_or_else(|| panic!("no digest for {arch}"));
            assert_eq!(digest.len(), 64, "{arch}");
            assert!(digest.chars().all(|c| c.is_ascii_hexdigit()), "{arch}");
        }
        assert_eq!(rclone_sha256("riscv64"), None);
    }

    /// The script is assembled from many pieces, so a quoting slip would only surface inside a
    /// sandbox. `sh -n` parses it here instead, offline.
    #[test]
    fn generates_a_syntactically_valid_install_script() {
        use std::io::Write as _;
        let script = rclone_install_script("amd64", "deadbeef");
        let mut child = std::process::Command::new("sh")
            .arg("-n")
            .stdin(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("run sh -n");
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(script.as_bytes())
            .expect("write the script");
        let output = child.wait_with_output().expect("wait for sh");
        assert!(
            output.status.success(),
            "sh -n rejected the install script: {}\n{script}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn verifies_the_archive_before_installing_it() {
        let script = rclone_install_script("amd64", "deadbeef");
        // The checksum gate must precede unpacking and installing.
        let check = script.find("sha256sum").expect("a checksum check");
        let unpack = script.find("unzip").expect("an unpack step");
        let install = script
            .find("/usr/local/bin/rclone")
            .expect("an install step");
        assert!(check < unpack, "{script}");
        assert!(unpack < install, "{script}");
        assert!(script.contains("expected_sha256='deadbeef'"), "{script}");
        assert!(
            script.contains(&format!("rclone-v{RCLONE_VERSION}-linux-amd64.zip")),
            "{script}"
        );
        assert!(script.contains("--proto '=https'"), "{script}");
        assert!(
            script.contains(&format!("exit {RCLONE_CHECKSUM_MISMATCH_EXIT}")),
            "{script}"
        );
    }
}
