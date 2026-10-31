#![cfg(feature = "sandbox")]
//! Ports the credential-smuggling half of Python's `tests/sandbox/test_mount_security.py`
//! (lines 1459-2130): configuration-line injection, s3fs option injection, rclone remote names
//! and extra arguments that would redirect the mount tool at another credential source, and
//! endpoints that carry their own authority.
//!
//! Python catches these in `validate_manifest_mount_credential_boundaries`, before a session
//! exists. Rust catches them where the value is interpolated — building the rclone
//! configuration, resolving the remote name, or building the mount argument vector — so each
//! Python assertion is translated into the equivalent activation attempt against a recording
//! session that runs no real mount tool.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::mount_patterns::{
    redact_sensitive_values, reject_config_line_break, reject_option_delimiter,
    url_contains_inline_authority,
};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    AzureBlobMount, BuiltinMountPattern, ConfiguredMountStrategy, ErrorCode, FuseMountPattern,
    Manifest, Mount, MountCredentialAuthority, MountPattern, MountPatternConfig,
    MountpointMountPattern, R2Mount, RcloneMountPattern, S3Mount, SandboxError,
};

// ---------------------------------------------------------------------------------------------
// A session that records commands instead of running them
// ---------------------------------------------------------------------------------------------

/// A session that answers every tool probe with success and records what a mount pattern does.
///
/// Mount tools cannot be installed in a test, so this is enough to assert the two things these
/// tests care about: that a rejected configuration never reaches a command, and that an accepted
/// one does.
#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    commands: Mutex<Vec<Vec<String>>>,
    files: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl RecordingSession {
    fn new() -> Self {
        RecordingSession {
            manifest: Manifest::new()
                .with_root("/workspace")
                .with_in_container_mount_broad_credential_exposure_acknowledged([
                    "/workspace/data",
                ]),
            commands: Mutex::new(Vec::new()),
            files: Mutex::new(BTreeMap::new()),
        }
    }

    fn commands(&self) -> Vec<Vec<String>> {
        self.commands.lock().unwrap().clone()
    }

    /// Every recorded argument, flattened, so a test can assert nothing leaked into a command.
    fn flattened(&self) -> String {
        self.commands()
            .into_iter()
            .map(|argv| argv.join(" "))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn written(&self) -> String {
        self.files
            .lock()
            .unwrap()
            .values()
            .map(|raw| String::from_utf8_lossy(raw).into_owned())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl SandboxSession for RecordingSession {
    fn id(&self) -> &str {
        "sess"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "recording"
    }

    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        self.commands
            .lock()
            .unwrap()
            .push(request.resolved_argv().clone());
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("no file at `{path}`"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), contents.to_vec());
        Ok(())
    }

    async fn mkdir(&self, _path: &str, _parents: bool) -> Result<(), SandboxError> {
        Ok(())
    }

    async fn rm(&self, path: &str, _recursive: bool) -> Result<(), SandboxError> {
        self.files.lock().unwrap().remove(path);
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Activate `mount` at `/workspace/data` against a fresh recording session.
async fn activate(mount: &Mount) -> (RecordingSession, Result<(), SandboxError>) {
    let session = RecordingSession::new();
    let result = mount
        .strategy
        .activate(mount, &session, "/workspace/data")
        .await;
    (session, result)
}

// ---------------------------------------------------------------------------------------------
// rclone configuration-line injection
// ---------------------------------------------------------------------------------------------

/// A free-form field interpolated into an rclone configuration line must stay on one line, and
/// the rejection never reproduces the injected value. Ports
/// `test_rejects_and_redacts_rclone_config_line_injection` (three parametrize cases).
#[tokio::test]
async fn an_rclone_configuration_value_that_would_add_a_line_is_rejected_and_redacted() {
    let cases: Vec<(Mount, &str)> = vec![
        (
            ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket").with_s3_provider("AWS\naccess_key_id = injected-value"),
            )
            .mount(),
            // Rust names the rclone configuration key (`provider`), Python the struct field.
            "provider",
        ),
        (
            ConfiguredMountStrategy::rclone(AzureBlobMount::new(
                "account\nkey = injected-value",
                "container",
            ))
            .mount(),
            "account",
        ),
        (
            ConfiguredMountStrategy::rclone(R2Mount::new(
                "bucket",
                "account\nsecret_access_key = injected-value",
            ))
            .mount(),
            // The R2 account id is interpolated into the derived `endpoint` key.
            "endpoint",
        ),
    ];

    for (mount, field) in cases {
        let (session, result) = activate(&mount).await;
        let error = result.unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{field}");
        assert!(error.message.contains("line break"), "{field}: {error}");
        assert!(error.message.contains(field), "{field}: {error}");
        assert!(!error.message.contains("injected-value"), "{error}");
        assert!(!session.written().contains("injected-value"));
        assert!(!session.flattened().contains("injected-value"));
    }
}

/// The line-break guard itself names the field without reproducing the value. Unit-level
/// companion to the Python test above.
#[test]
fn the_configuration_line_guard_names_the_field_only() {
    reject_config_line_break("region", "us-east-1").unwrap();
    for value in ["a\nb", "a\rb", "a\r\nb"] {
        let error = reject_config_line_break("region", value).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(error.message.contains("region"), "{error}");
        assert!(!error.message.contains('\n'), "{error}");
    }
}

/// An s3fs-style option delimiter inside a value would append another `-o` setting, such as a
/// `passwd_file` pointing at a manifest-held credential. Ports
/// `test_rejects_blaxel_s3fs_endpoint_option_injection`, whose Blaxel strategy lives in a
/// provider crate module; the shared guard it calls is asserted directly here.
#[test]
fn an_s3fs_option_delimiter_inside_a_value_is_rejected() {
    reject_option_delimiter("endpoint_url", "https://s3.example").unwrap();
    for value in [
        "https://s3.example,public_bucket=0,passwd_file=/workspace/creds",
        "us-east-1,public_bucket=0,passwd_file=/workspace/creds",
        "account,public_bucket=0,passwd_file=/workspace/creds",
    ] {
        let error = reject_option_delimiter("endpoint_url", value).unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(error.message.contains("endpoint_url"), "{error}");
        assert!(!error.message.contains("passwd_file"), "{error}");
    }
}

// ---------------------------------------------------------------------------------------------
// rclone remote names
// ---------------------------------------------------------------------------------------------

/// A remote name beginning with `:` is a whole on-the-fly rclone backend definition, so it can
/// carry credentials the mount never declared. It is rejected, and the rejection never echoes
/// the name back. Ports `test_rejects_rclone_on_the_fly_remote_name`.
#[test]
fn an_on_the_fly_rclone_remote_name_is_rejected_without_echoing_it() {
    let sentinel = "remote-name-secret";
    let pattern = RcloneMountPattern::new().with_remote_name(format!(
        ":s3,access_key_id=access,secret_access_key={sentinel}"
    ));

    let error = pattern
        .resolve_remote_name("sess", &MountPatternConfig::new())
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(!error.message.contains(sentinel), "{error}");
    assert!(!error.message.contains("access_key_id"), "{error}");
}

/// An activation carrying such a name fails before any command runs and before the
/// configuration is written. Ports the "without mutating the trusted manifest" half of the same
/// Python test.
#[tokio::test]
async fn an_on_the_fly_remote_name_never_reaches_a_command() {
    let sentinel = "activation-remote-name-secret";
    let mount = ConfiguredMountStrategy::in_container(
        S3Mount::new("bucket"),
        BuiltinMountPattern::Rclone(
            RcloneMountPattern::new().with_remote_name(format!(":s3,secret_access_key={sentinel}")),
        ),
    )
    .mount();

    let (session, result) = activate(&mount).await;

    let error = result.unwrap_err();
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(!error.message.contains(sentinel), "{error}");
    assert!(!session.flattened().contains(sentinel));
    assert!(!session.written().contains(sentinel));
}

/// An ordinary remote name is preserved. Ports `test_preserves_ordinary_rclone_remote_name`.
///
/// Python's name regex also admits an inner space (`"public bucket-1"`); Rust keeps rejecting
/// whitespace, which is a deliberate narrowing recorded in the report — a generated name never
/// contains one and an inner space buys nothing but ambiguity in the `[section]` header.
#[test]
fn an_ordinary_rclone_remote_name_is_preserved() {
    for name in ["public-bucket-1", "bucket.one", "_private", "S3_2"] {
        let pattern = RcloneMountPattern::new().with_remote_name(name);
        assert_eq!(
            pattern
                .resolve_remote_name("sess", &MountPatternConfig::new())
                .unwrap(),
            name
        );
    }

    let spaced = RcloneMountPattern::new().with_remote_name("public bucket-1");
    assert!(
        spaced
            .resolve_remote_name("sess", &MountPatternConfig::new())
            .is_err()
    );
}

/// A name that would split the `name:path` argument or open a different configuration section is
/// rejected. Extends the Python regex to the separators Rust's generated name must never carry.
#[test]
fn a_remote_name_with_a_separator_is_rejected() {
    for name in [
        "s3:evil",
        "with/slash",
        "with\\backslash",
        "[section]",
        "a b",
        "name,option=value",
        "-leading-dash",
        "",
    ] {
        let pattern = RcloneMountPattern::new().with_remote_name(name);
        let resolved = pattern.resolve_remote_name("sess", &MountPatternConfig::new());
        if name.is_empty() {
            // An empty pinned name falls through to the derived name, which needs a kind.
            assert!(resolved.is_err(), "{name}");
            continue;
        }
        let error = resolved.unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{name}");
    }
}

// ---------------------------------------------------------------------------------------------
// rclone extra arguments
// ---------------------------------------------------------------------------------------------

/// An extra argument that redirects rclone at another credential source is rejected: it would
/// hand the sandbox an authority the manifest never declared and no acknowledgement covers.
/// Ports `test_rejects_rclone_credential_source_overrides` (five parametrize cases).
#[tokio::test]
async fn rclone_credential_source_overrides_are_rejected() {
    let cases: Vec<Vec<&str>> = vec![
        vec!["--config=/workspace/credentials.conf"],
        vec!["--config", "/workspace/credentials.conf"],
        vec!["--s3-env-auth=true"],
        vec!["--s3-profile=production"],
        vec!["--azureblob-use-msi=true"],
        vec!["--header", "Authorization: Bearer secret"],
        // The option name is normalized the way Python normalizes it, so neither the
        // underscore spelling nor a capitalized one slips past.
        vec!["--s3_env_auth=true"],
        vec!["--S3-Env-Auth=true"],
        // A bare value with no option in front of it is not an option at all.
        vec!["/workspace/credentials.conf"],
        // A safe option name with no value takes the next argument, which must not be another
        // option.
        vec!["--uid", "--config=/workspace/credentials.conf"],
    ];

    for extra_args in cases {
        let mount = ConfiguredMountStrategy::in_container(
            S3Mount::new("bucket"),
            BuiltinMountPattern::Rclone(
                RcloneMountPattern::new().with_extra_args(extra_args.clone()),
            ),
        )
        .mount();

        let (session, result) = activate(&mount).await;
        let error = result.unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{extra_args:?}");
        assert!(
            error.message.contains("does not support exposing"),
            "{extra_args:?}: {error}"
        );
        assert!(!error.message.contains("secret"), "{error}");
        assert!(
            !session.flattened().contains("credentials.conf"),
            "{extra_args:?}"
        );
        assert!(
            !session.flattened().contains("Authorization"),
            "{extra_args:?}"
        );
    }
}

/// The supported, credential-free subset still reaches the mount command, in both the `--opt
/// value` and `--opt=value` spellings. Ports
/// `test_preserves_supported_credentialless_rclone_extra_args`.
#[tokio::test]
async fn supported_credentialless_rclone_extra_args_are_preserved() {
    let mount = ConfiguredMountStrategy::in_container(
        S3Mount::new("bucket"),
        BuiltinMountPattern::Rclone(RcloneMountPattern::new().with_extra_args([
            "--allow-other",
            "--uid",
            "123",
            "--gid=456",
            "--buffer-size",
            "0",
        ])),
    )
    .mount();

    let (session, result) = activate(&mount).await;

    result.unwrap();
    let commands = session.flattened();
    assert!(commands.contains("--allow-other"), "{commands}");
    assert!(commands.contains("--uid 123"), "{commands}");
    assert!(commands.contains("--gid=456"), "{commands}");
    assert!(commands.contains("--buffer-size 0"), "{commands}");
}

// ---------------------------------------------------------------------------------------------
// Endpoints carrying their own authority
// ---------------------------------------------------------------------------------------------

/// A URL that carries credentials of its own is authority, whether it is well formed or not.
/// Ports the classifier behind
/// `test_rejects_malformed_inline_credential_url_without_mutating_trusted_manifest`.
#[test]
fn a_url_carrying_its_own_authority_is_detected_even_when_malformed() {
    for value in [
        "https://user:malformed-secret@[invalid",
        "https:user:malformed-secret@example.test",
        "https://user:pattern-secret@example.test",
        "https://example.test?signature=pattern-secret",
    ] {
        assert!(url_contains_inline_authority(value), "{value}");
    }
    for value in [
        "https://s3.example.test",
        "https://s3.example.test/bucket",
        "",
    ] {
        assert!(!url_contains_inline_authority(value), "{value}");
    }
}

/// An endpoint carrying inline authority makes the mount credential-bearing, so it cannot
/// activate unacknowledged and its value never reaches the argument vector. Ports
/// `test_rejects_malformed_inline_credential_url_without_mutating_trusted_manifest`; Rust
/// classifies such an endpoint as mount-scoped authority instead of rejecting the manifest
/// outright, so the equivalent guarantee is the acknowledgement requirement.
#[test]
fn an_inline_authority_endpoint_makes_an_s3_mount_credential_bearing() {
    for endpoint_url in [
        "https://user:malformed-secret@[invalid",
        "https:user:malformed-secret@example.test",
    ] {
        let mount = ConfiguredMountStrategy::rclone(
            S3Mount::new("example-bucket").with_endpoint_url(endpoint_url),
        )
        .mount();
        assert_eq!(
            mount.strategy.credential_authority(),
            MountCredentialAuthority::MountScoped,
            "{endpoint_url}"
        );
        let error = Manifest::new()
            .with_root("/workspace")
            .authorize_mount_activation(&mount, "/workspace/data")
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(!error.message.contains("malformed-secret"), "{error}");
    }
}

/// An endpoint pinned on the *pattern* is not part of the mount's configuration, so no
/// acknowledgement covers it. One carrying its own authority is rejected outright. Ports
/// `test_rejects_mountpoint_endpoint_authority`.
#[tokio::test]
async fn a_mountpoint_pattern_endpoint_carrying_authority_is_rejected() {
    for endpoint_url in [
        "https://user:pattern-secret@example.test",
        "https://example.test?signature=pattern-secret",
    ] {
        let pattern = MountpointMountPattern::new().with_endpoint_url(endpoint_url);
        let session = RecordingSession::new();
        let config = MountPatternConfig::new().with_option("bucket", "example-bucket");

        let error = pattern
            .apply(&session, "/workspace/data", &config)
            .await
            .unwrap_err();

        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(
            error
                .message
                .contains("mount_strategy.pattern.options.endpoint_url"),
            "{error}"
        );
        assert!(!error.message.contains("pattern-secret"), "{error}");
        assert!(!session.flattened().contains("pattern-secret"));
    }
}

/// An ordinary pattern endpoint still reaches the mount command. Guards the fix above against
/// over-reach.
#[tokio::test]
async fn an_ordinary_mountpoint_pattern_endpoint_is_preserved() {
    let pattern = MountpointMountPattern::new().with_endpoint_url("https://s3.example.test");
    let session = RecordingSession::new();
    let config = MountPatternConfig::new()
        .with_option("bucket", "example-bucket")
        .with_option("endpoint_url", "https://s3.example.test");

    pattern
        .apply(&session, "/workspace/data", &config)
        .await
        .unwrap();

    assert!(
        session.flattened().contains("https://s3.example.test"),
        "{}",
        session.flattened()
    );
}

// ---------------------------------------------------------------------------------------------
// Redaction of command output
// ---------------------------------------------------------------------------------------------

/// A mount tool that echoes its own configuration back has every credential value, and each
/// shell-quoted form of one, replaced before the text reaches an error. Ports the redaction half
/// of `test_rejects_and_redacts_rclone_config_line_injection`.
#[test]
fn command_output_is_redacted_value_by_value() {
    let secret = "SUPERSECRET";
    let text = format!("failed: key={secret} quoted='{secret}' double=\"{secret}\"");

    let redacted = redact_sensitive_values(&text, &[secret.to_string()]);

    assert!(!redacted.contains(secret), "{redacted}");
    assert!(redacted.contains("REDACTED"), "{redacted}");
}

// ---------------------------------------------------------------------------------------------
// blobfuse cache path and rclone configuration merge
// ---------------------------------------------------------------------------------------------

/// A blobfuse cache path must stay inside the workspace, and the error records the offending
/// path in its context the way Python's does. Ports
/// `test_blobfuse_cache_path_must_be_relative_to_workspace` (the mount-security side of the
/// message and context contract).
#[tokio::test]
async fn a_blobfuse_cache_path_outside_the_workspace_is_rejected_with_context() {
    for (cache_path, shown) in [
        ("/tmp/blobfuse-cache", "/tmp/blobfuse-cache"),
        ("../blobfuse-cache", "../blobfuse-cache"),
        ("C:\\cache\\blobfuse", "C:/cache/blobfuse"),
    ] {
        let session = RecordingSession::new();
        let pattern = FuseMountPattern::new().with_cache_path(cache_path);
        let config = MountPatternConfig::new()
            .with_option("mount_type", "azure_blob_mount")
            .with_option("account", "acct")
            .with_option("container", "container");

        let error = pattern
            .apply(&session, "/workspace/data", &config)
            .await
            .unwrap_err();

        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{cache_path}");
        assert_eq!(
            error.message,
            "blobfuse cache_path must be relative to the workspace root"
        );
        assert_eq!(
            error.context.get("cache_path").map(String::as_str),
            Some(shown),
            "{cache_path}"
        );
        assert!(session.written().is_empty(), "{cache_path}");
    }
}

/// A cache inside the mount it caches is rejected before anything is staged, naming both paths.
/// Ports `test_blobfuse_cache_path_must_be_outside_mount_path`.
#[tokio::test]
async fn a_blobfuse_cache_inside_the_mount_is_rejected_with_both_paths() {
    let session = RecordingSession::new();
    let pattern = FuseMountPattern::new().with_cache_path("data/cache");
    let config = MountPatternConfig::new()
        .with_option("mount_type", "azure_blob_mount")
        .with_option("account", "acct")
        .with_option("container", "container");

    let error = pattern
        .apply(&session, "/workspace/data", &config)
        .await
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert_eq!(
        error.message,
        "blobfuse cache_path must be outside the mount path"
    );
    assert_eq!(
        error.context.get("mount_path").map(String::as_str),
        Some("/workspace/data")
    );
    assert_eq!(
        error.context.get("cache_path").map(String::as_str),
        Some("/workspace/data/cache")
    );
    assert!(session.written().is_empty());
}

/// Merging generated settings into a caller-placed rclone configuration keeps the blank line
/// that separated the mounted remote's section from the next one. Ports
/// `test_r2_mount_merges_existing_rclone_config_section`.
#[tokio::test]
async fn an_rclone_merge_keeps_the_blank_line_before_the_next_section() {
    let session = RecordingSession::new();
    session.files.lock().unwrap().insert(
        "/workspace/rclone.conf".to_string(),
        b"[bucket]\ntype = s3\nregion = auto\n\n[other]\ntype = memory\n".to_vec(),
    );
    let pattern = RcloneMountPattern::new().with_config_file_path("rclone.conf");
    let config = MountPatternConfig::new()
        .with_config_text("[bucket]\naccess_key_id = AK\n")
        .with_option("remote_name", "bucket")
        .with_option("remote_path", "bucket");

    pattern
        .apply(&session, "/workspace/data", &config)
        .await
        .unwrap();

    let staged = session
        .files
        .lock()
        .unwrap()
        .iter()
        .find(|(path, _)| path.contains(".sandbox-rclone-config"))
        .map(|(_, raw)| String::from_utf8_lossy(raw).into_owned())
        .expect("a staged configuration");
    assert_eq!(
        staged,
        "[bucket]\ntype = s3\nregion = auto\naccess_key_id = AK\n\n[other]\ntype = memory\n"
    );
}
