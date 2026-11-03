//! Ports of `tests/extensions/sandbox/test_rclone.py` and
//! `tests/extensions/sandbox/test_blaxel_mounts.py`.
//!
//! Python reaches into the module privates (`_rclone_arch`, `_rclone_install_command`,
//! `_mount_s3`, `_mount_gcs`); the Rust equivalents are private too, so each test drives the
//! public entry point (`ensure_rclone`, `MountStrategy::activate`) against a scripted session and
//! asserts on the exact shell command the sandbox received. That is the same observable the
//! Python assertions look at.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy,
};
use openai_agents_sdk::extensions::sandbox::e2b::rclone::{RCLONE_VERSION, ensure_rclone};
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::testing::{
    SandboxCall, SandboxMethod, SandboxStep, ScriptedSandboxSession, scripted_sandbox_session,
};
use openai_agents_sdk::sandbox::types::ExecResult;

/// Exit code the pinned-release install script uses for a checksum mismatch.
///
/// Mirrors `agents.extensions.sandbox._rclone._RCLONE_CHECKSUM_MISMATCH_EXIT`.
const RCLONE_CHECKSUM_MISMATCH_EXIT: i32 = 86;

fn result(exit_code: i32, stdout: &str) -> ExecResult {
    ExecResult {
        exit_code,
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
    }
}

fn ok() -> ExecResult {
    result(0, "")
}

fn exec_step(result: ExecResult) -> SandboxStep {
    SandboxStep::new(SandboxMethod::Exec).returning_exec(result)
}

/// Every `sh -lc <script>` the session was asked to run.
fn shell_scripts(session: &ScriptedSandboxSession) -> Vec<String> {
    session
        .calls()
        .into_iter()
        .filter(|call| call.method == SandboxMethod::Exec)
        .filter_map(|call: SandboxCall| {
            // A root command is wrapped in `sudo -u root -- sh -lc <script>`, so the script is
            // whatever follows the shell flag rather than a fixed position.
            let argv = call.arguments;
            let at = argv
                .iter()
                .position(|token| token == "-lc" || token == "-c")?;
            argv.get(at + 1).cloned()
        })
        .collect()
}

/// A script that gets as far as running the pinned-release installer.
///
/// The steps are, in order: the `rclone` presence probe, the `apt-get` probe, `uname -m`, the two
/// prerequisite installs, and the installer itself.
fn install_script_steps(machine: &str, install_exit: i32) -> Vec<SandboxStep> {
    vec![
        exec_step(result(1, "")),
        exec_step(ok()),
        exec_step(result(0, machine)),
        exec_step(ok()),
        exec_step(ok()),
        exec_step(result(install_exit, "")),
        // The post-install presence probe only runs when the installer succeeded.
        exec_step(ok()),
    ]
}

// ---------------------------------------------------------------------------------------------
// test_rclone.py
// ---------------------------------------------------------------------------------------------

/// `test_rclone_arch_maps_upstream_linux_archives`: every `uname -m` the upstream archives cover
/// resolves to the release architecture whose archive the installer downloads.
#[tokio::test]
async fn rclone_arch_maps_upstream_linux_archives() {
    for (machine, expected) in [
        ("x86_64", "amd64"),
        ("amd64", "amd64"),
        ("i386", "386"),
        ("i686", "386"),
        ("x86", "386"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("armv7l", "arm-v7"),
        ("armv6l", "arm-v6"),
        ("armv5l", "arm"),
    ] {
        let session = scripted_sandbox_session(install_script_steps(&format!("{machine}\n"), 0));
        ensure_rclone(&session, true)
            .await
            .unwrap_or_else(|error| panic!("{machine}: {}", error.message));
        let install = shell_scripts(&session)
            .into_iter()
            .find(|script| script.contains("downloads.rclone.org"))
            .unwrap_or_else(|| panic!("{machine}: no installer command"));
        assert!(
            install.contains(&format!("rclone-v{RCLONE_VERSION}-linux-{expected}.zip")),
            "{machine} should install the `{expected}` archive: {install}"
        );
    }
}

/// `test_rclone_arch_rejects_unknown_machine`: an unmapped `uname -m` never reaches an install.
#[tokio::test]
async fn rclone_arch_rejects_unknown_machine() {
    let session = scripted_sandbox_session([
        exec_step(result(1, "")),
        exec_step(ok()),
        exec_step(result(0, "mips64\n")),
    ]);

    let error = ensure_rclone(&session, true)
        .await
        .expect_err("an unmapped architecture has no pinned release");

    assert!(error.message.contains("mips64"), "{}", error.message);
    assert_eq!(session.calls().len(), 3);
}

/// `test_rclone_install_command_pins_and_verifies_archive`: the installer downloads the pinned
/// upstream archive, checks its SHA-256 before unpacking, and only moves the binary into place
/// after it reports the pinned version.
#[tokio::test]
async fn rclone_install_command_pins_and_verifies_archive() {
    let session = scripted_sandbox_session(install_script_steps("x86_64\n", 0));
    ensure_rclone(&session, true).await.expect("install rclone");

    let command = shell_scripts(&session)
        .into_iter()
        .find(|script| script.contains("downloads.rclone.org"))
        .expect("the installer command");

    let archive = format!("rclone-v{RCLONE_VERSION}-linux-amd64.zip");
    assert!(
        command.contains(&format!(
            "https://downloads.rclone.org/v{RCLONE_VERSION}/{archive}"
        )),
        "{command}"
    );
    assert!(!command.contains("github.com"), "{command}");
    // The published digest is pinned in the source; the command must carry exactly one.
    assert!(command.contains("expected_sha256='"), "{command}");
    assert!(
        command.contains("sha256sum --check --strict -"),
        "{command}"
    );
    assert!(
        command.contains("mktemp /usr/local/bin/.rclone.XXXXXX"),
        "{command}"
    );
    let verify_execution = "version_output=\"$(\"$target_tmp\" version)\"";
    let verify_version = format!("grep -Fx 'rclone v{RCLONE_VERSION}'");
    let atomic_replace = "mv -f \"$target_tmp\" /usr/local/bin/rclone";
    let execution_at = command.find(verify_execution).expect("runs the binary");
    let version_at = command.find(&verify_version).expect("checks the version");
    let replace_at = command.find(atomic_replace).expect("moves it into place");
    assert!(execution_at < version_at, "{command}");
    assert!(version_at < replace_at, "{command}");
    assert!(
        !command.contains("curl -fsSL https://rclone.org/install.sh | bash"),
        "{command}"
    );
}

/// `test_ensure_rclone_preserves_preinstalled_binary`: a present `rclone` short-circuits.
#[tokio::test]
async fn ensure_rclone_preserves_preinstalled_binary() {
    let session = scripted_sandbox_session([exec_step(ok())]);

    ensure_rclone(&session, true).await.expect("already there");

    assert_eq!(session.calls().len(), 1);
}

/// `test_ensure_rclone_rejects_unsupported_architecture_before_install`: the architecture probe
/// runs before any download, and the failure names the machine it saw.
#[tokio::test]
async fn ensure_rclone_rejects_unsupported_architecture_before_install() {
    let session = scripted_sandbox_session([
        exec_step(result(1, "")),
        exec_step(ok()),
        exec_step(result(0, "mips64\n")),
    ]);

    let error = ensure_rclone(&session, true)
        .await
        .expect_err("an unsupported architecture must not install anything");

    assert!(
        error.message.contains("architecture") && error.message.contains("mips64"),
        "{}",
        error.message
    );
    assert_eq!(session.calls().len(), 3);
}

/// `test_ensure_rclone_reports_checksum_mismatch`: the installer's dedicated exit code is turned
/// into a verification failure that names the version and the architecture.
#[tokio::test]
async fn ensure_rclone_reports_checksum_mismatch() {
    let session = scripted_sandbox_session([
        exec_step(result(1, "")),
        exec_step(ok()),
        exec_step(result(0, "x86_64\n")),
        exec_step(ok()),
        exec_step(ok()),
        exec_step(result(RCLONE_CHECKSUM_MISMATCH_EXIT, "")),
    ]);

    let error = ensure_rclone(&session, true)
        .await
        .expect_err("a checksum mismatch must not install the binary");

    assert!(
        error.message.contains("SHA-256 verification"),
        "{}",
        error.message
    );
    assert!(error.message.contains(RCLONE_VERSION), "{}", error.message);
    assert!(error.message.contains("amd64"), "{}", error.message);
    assert_eq!(session.calls().len(), 6);
}

// ---------------------------------------------------------------------------------------------
// test_blaxel_mounts.py
// ---------------------------------------------------------------------------------------------

/// A command injection attempt, as Python's `_INJECTION` spells it.
const INJECTION: &str = "x; touch /tmp/pwned";

/// Split a shell command the way `shlex.split` does for the simple quoting these commands use.
fn shlex_split(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some('\'') => {
                if c == '\'' {
                    quote = None;
                } else {
                    current.push(c);
                }
            }
            Some('"') => match c {
                '"' => quote = None,
                '\\' => {
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                _ => current.push(c),
            },
            Some(_) => unreachable!(),
            None => match c {
                '\'' | '"' => {
                    started = true;
                    quote = Some(c);
                }
                '\\' => {
                    started = true;
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                c if c.is_whitespace() => {
                    if started {
                        tokens.push(std::mem::take(&mut current));
                        started = false;
                    }
                }
                _ => {
                    started = true;
                    current.push(c);
                }
            },
        }
    }
    if started {
        tokens.push(current);
    }
    tokens
}

/// Activate `config` against a session that answers every command successfully, and return the
/// shell commands it ran.
async fn activated_commands(config: BlaxelCloudBucketMountConfig) -> Vec<String> {
    // `which <tool>` succeeds, so the activation goes straight to `mkdir -p` and the mount.
    let session = scripted_sandbox_session([exec_step(ok()), exec_step(ok()), exec_step(ok())]);
    let strategy = BlaxelCloudBucketMountStrategy::new(config);
    let mount = Mount::new(std::sync::Arc::new(strategy.clone()));
    strategy
        .activate(&mount, &session, "/mnt/data")
        .await
        .expect("activate the mount");
    session.assert_complete();
    shell_scripts(&session)
}

/// `test_s3_mount_options_are_shell_quoted`: an endpoint carrying `; touch` stays inside the `-o`
/// option token instead of becoming its own command.
#[tokio::test]
async fn s3_mount_options_are_shell_quoted() {
    let commands = activated_commands(
        BlaxelCloudBucketMountConfig::s3("bucket").with_endpoint_url(format!("http://{INJECTION}")),
    )
    .await;

    let command = commands
        .into_iter()
        .find(|command| command.starts_with("s3fs"))
        .expect("the s3fs command");
    assert!(
        !shlex_split(&command).contains(&"touch".to_string()),
        "{command}"
    );
}

/// `test_gcs_mount_prefix_is_shell_quoted`: the same for a `gcsfuse` prefix.
#[tokio::test]
async fn gcs_mount_prefix_is_shell_quoted() {
    let commands =
        activated_commands(BlaxelCloudBucketMountConfig::gcs("bucket").with_prefix(INJECTION))
            .await;

    let command = commands
        .into_iter()
        .find(|command| command.starts_with("gcsfuse"))
        .expect("the gcsfuse command");
    assert!(
        !shlex_split(&command).contains(&"touch".to_string()),
        "{command}"
    );
}
