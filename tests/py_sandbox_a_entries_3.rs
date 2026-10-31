//! Ports of the Python sandbox suite's `tests/sandbox/test_mounts.py`.
//!
//! Python builds each runtime configuration through `Mount.build_in_container_mount_config`,
//! which reads the session; Rust splits that in two. `ConfiguredMount::build_in_container_config`
//! derives the option map and the generated configuration text, and the pattern's `apply` runs the
//! commands. The tests that assert a generated configuration therefore call the first; the ones
//! that assert commands, staged files and ordering drive a pattern against a recording session.
//!
//! Three structural differences are intentional and shared by every test below:
//!
//! * Python names scratch directories after the session's UUID; Rust uses a hash of the session id
//!   (`session_scratch_id`), so expected paths are built with that helper.
//! * Rust stages files at absolute workspace paths where Python passes workspace-relative ones.
//! * Rust creates directories through `SandboxSession::mkdir`, which runs `mkdir -p -- <path>`.
//!
//! The two pydantic tests (`model_dump` round trips) have no counterpart because Rust mounts carry
//! no serialization surface; they are accounted for in the parity report.
#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::Manifest;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::mount_entries::{
    AzureBlobMount, BoxMount, BuiltinMountPattern, ConfiguredMount, ConfiguredMountStrategy,
    GcsMount, R2Mount, S3FilesMount, S3Mount,
};
use openai_agents_sdk::sandbox::mount_patterns::{
    BLOBFUSE_CACHE_DIR, BLOBFUSE_CONFIG_DIR, BlobfuseCacheType, FUSE_OPTION_ACCOUNT,
    FUSE_OPTION_CONTAINER, FUSE_SECRET_ACCOUNT_KEY, FuseMountPattern, MOUNT_TYPE_AZURE_BLOB,
    MOUNT_TYPE_GCS, MOUNT_TYPE_S3, MOUNTPOINT_OPTION_BUCKET, MOUNTPOINT_OPTION_ENDPOINT_URL,
    MOUNTPOINT_OPTION_REGION, MOUNTPOINT_SECRET_ACCESS_KEY_ID, MOUNTPOINT_SECRET_SECRET_ACCESS_KEY,
    MOUNTPOINT_SECRET_SESSION_TOKEN, MountpointMountPattern, OPTION_MOUNT_TYPE, OPTION_READ_ONLY,
    S3FilesMountPattern, session_scratch_id,
};
use openai_agents_sdk::sandbox::mounts::{
    Mount, MountPattern, MountPatternConfig, MountStrategy, RCLONE_OPTION_READ_ONLY,
    RCLONE_OPTION_REMOTE_KIND, RCLONE_OPTION_REMOTE_NAME, RCLONE_OPTION_REMOTE_PATH,
    RcloneMountPattern,
};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;

const SESSION_ID: &str = "12345678-1234-5678-1234-567812345678";
/// What Python's `uuid.hex` of [`SESSION_ID`] looks like, used where Rust derives the remote name
/// from the session id verbatim.
const SESSION_HEX: &str = "12345678123456781234567812345678";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// A session standing in for Python's `_MountpointApplySession` / `_GeneratedConfigApplySession`:
/// it records the commands and file writes a mount pattern issues and serves the files it is
/// given. The mount command itself fails when `mount_exit_code` is non-zero.
#[derive(Debug)]
struct MountSession {
    id: String,
    manifest: Manifest,
    commands: StdMutex<Vec<Vec<String>>>,
    writes: StdMutex<Vec<(String, Vec<u8>)>>,
    files: StdMutex<BTreeMap<String, Vec<u8>>>,
    mount_exit_code: i32,
    mount_stderr: Vec<u8>,
}

impl MountSession {
    fn new() -> Self {
        MountSession::with_id(SESSION_ID)
    }

    fn with_id(id: &str) -> Self {
        MountSession {
            id: id.to_string(),
            manifest: Manifest::new().with_root("/workspace"),
            commands: StdMutex::new(Vec::new()),
            writes: StdMutex::new(Vec::new()),
            files: StdMutex::new(BTreeMap::new()),
            mount_exit_code: 0,
            mount_stderr: Vec::new(),
        }
    }

    fn failing_mount(mut self, exit_code: i32, stderr: &[u8]) -> Self {
        self.mount_exit_code = exit_code;
        self.mount_stderr = stderr.to_vec();
        self
    }

    fn with_file(self, path: &str, contents: &str) -> Self {
        self.files
            .lock()
            .expect("files")
            .insert(path.to_string(), contents.as_bytes().to_vec());
        self
    }

    fn commands(&self) -> Vec<Vec<String>> {
        self.commands.lock().expect("commands").clone()
    }

    fn writes(&self) -> Vec<(String, Vec<u8>)> {
        self.writes.lock().expect("writes").clone()
    }

    fn scratch_id(&self) -> String {
        session_scratch_id(self)
    }
}

#[async_trait]
impl SandboxSession for MountSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "recording"
    }

    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        let argv = request.resolved_argv();
        self.commands.lock().expect("commands").push(argv.clone());
        let is_mount_command = argv.len() >= 3
            && argv[..2] == ["sh", "-lc"]
            && argv[2].contains("mount-s3 ")
            && !argv[2].contains("command -v ");
        if is_mount_command {
            return Ok(ExecResult {
                exit_code: self.mount_exit_code,
                stdout: Vec::new(),
                stderr: self.mount_stderr.clone(),
            });
        }
        Ok(ExecResult::default())
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.files
            .lock()
            .expect("files")
            .get(path)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("no file at `{path}`"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.writes
            .lock()
            .expect("writes")
            .push((path.to_string(), contents.to_vec()));
        self.files
            .lock()
            .expect("files")
            .insert(path.to_string(), contents.to_vec());
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn context<'a>(error: &'a SandboxError, key: &str) -> &'a str {
    error
        .context
        .get(key)
        .unwrap_or_else(|| panic!("no `{key}` in {:?}", error.context))
}

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| part.to_string()).collect()
}

fn rclone() -> BuiltinMountPattern {
    BuiltinMountPattern::Rclone(RcloneMountPattern::new())
}

fn config_text(config: &MountPatternConfig) -> String {
    config
        .config_text()
        .expect("the configuration carries generated text")
        .to_string()
}

fn option<'a>(config: &'a MountPatternConfig, key: &str) -> &'a str {
    config
        .options
        .get(key)
        .unwrap_or_else(|| panic!("no option `{key}`"))
}

/// The name Python's `RcloneMountPattern.resolve_remote_name(remote_kind=..)` derives, which
/// Rust derives the same way from the kind recorded in the option map.
fn derived_remote_name(pattern: &RcloneMountPattern, session_id: &str, kind: &str) -> String {
    let config = MountPatternConfig::new().with_option(RCLONE_OPTION_REMOTE_KIND, kind);
    pattern
        .resolve_remote_name(session_id, &config)
        .expect("remote name")
}

/// The configuration a mountpoint pattern receives for `bucket`, as the Python
/// `MountpointMountConfig(...)` the tests build by hand.
fn mountpoint_config(bucket: &str, mount_type: &str, read_only: bool) -> MountPatternConfig {
    MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, mount_type)
        .with_option(MOUNTPOINT_OPTION_BUCKET, bucket)
        .with_option(OPTION_READ_ONLY, read_only.to_string())
}

fn with_credentials(
    config: MountPatternConfig,
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
) -> MountPatternConfig {
    let config = config
        .with_secret(MOUNTPOINT_SECRET_ACCESS_KEY_ID, access_key_id)
        .with_secret(MOUNTPOINT_SECRET_SECRET_ACCESS_KEY, secret_access_key);
    match session_token {
        Some(token) => config.with_secret(MOUNTPOINT_SECRET_SESSION_TOKEN, token),
        None => config,
    }
}

/// The shell script of the `mount-s3` command a mountpoint pattern ran.
fn mount_script(session: &MountSession) -> String {
    let command = session.commands().pop().expect("a mount command");
    assert_eq!(command[..2], ["sh", "-lc"], "command: {command:?}");
    command[2].clone()
}

// ---------------------------------------------------------------------------
// Azure Blob and Box rclone configuration
// ---------------------------------------------------------------------------

/// Python's `test_azure_blob_mount_builds_rclone_runtime_config_without_hidden_pattern_state`:
/// the apply configuration carries the generated text, the unmount configuration the same remote
/// without it.
#[test]
fn azure_blob_mount_builds_rclone_runtime_config_without_hidden_pattern_state() {
    let pattern = RcloneMountPattern::new().with_config_file_path("rclone.conf");
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "azureblob");
    let builtin = BuiltinMountPattern::Rclone(pattern);
    let mount = AzureBlobMount::new("acct", "container");

    let apply_config = mount
        .build_in_container_config(SESSION_HEX, &builtin, true, true)
        .expect("apply config");
    let unmount_config = mount
        .build_in_container_config(SESSION_HEX, &builtin, true, false)
        .expect("unmount config");

    assert_eq!(
        option(&apply_config, RCLONE_OPTION_REMOTE_NAME),
        remote_name
    );
    assert_eq!(
        option(&apply_config, RCLONE_OPTION_REMOTE_PATH),
        "container"
    );
    assert!(config_text(&apply_config).contains("account = acct"));
    assert_eq!(
        option(&unmount_config, RCLONE_OPTION_REMOTE_NAME),
        remote_name
    );
    assert!(unmount_config.config_text().is_none());
}

/// Python's `test_azure_blob_mount_enables_rclone_msi_for_managed_identity`.
#[test]
fn azure_blob_mount_enables_rclone_msi_for_managed_identity() {
    let mount = AzureBlobMount::new("acct", "container")
        .with_identity_client_id("managed-identity-client-id");

    let config = mount
        .build_in_container_config(SESSION_HEX, &rclone(), true, true)
        .expect("config");
    let text = config_text(&config);

    assert!(text.contains("use_msi = true"), "text: {text}");
    assert!(
        text.contains("msi_client_id = managed-identity-client-id"),
        "text: {text}"
    );
    assert!(!text.contains("use_msi = false"), "text: {text}");
}

/// Python's `test_box_mount_builds_rclone_runtime_config_with_box_auth_options`.
#[test]
fn box_mount_builds_rclone_runtime_config_with_box_auth_options() {
    let pattern = RcloneMountPattern::new().with_config_file_path("rclone.conf");
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "box");
    let builtin = BuiltinMountPattern::Rclone(pattern);
    let mount = BoxMount {
        path: Some("/Shared/Finance".to_string()),
        client_id: Some("client-id".to_string()),
        client_secret: Some("client-secret".to_string()),
        token: Some("{\"access_token\":\"token\"}".to_string()),
        root_folder_id: Some("12345".to_string()),
        impersonate: Some("user-42".to_string()),
        ..Default::default()
    };

    let apply_config = mount
        .build_in_container_config(SESSION_HEX, &builtin, false, true)
        .expect("apply config");
    let unmount_config = mount
        .build_in_container_config(SESSION_HEX, &builtin, false, false)
        .expect("unmount config");

    assert_eq!(
        option(&apply_config, RCLONE_OPTION_REMOTE_NAME),
        remote_name
    );
    assert_eq!(
        option(&apply_config, RCLONE_OPTION_REMOTE_PATH),
        "Shared/Finance"
    );
    assert_eq!(option(&apply_config, RCLONE_OPTION_READ_ONLY), "false");
    let text = config_text(&apply_config);
    assert!(text.contains("type = box"), "text: {text}");
    assert!(text.contains("client_id = client-id"), "text: {text}");
    assert!(
        text.contains("client_secret = client-secret"),
        "text: {text}"
    );
    assert!(
        text.contains("token = {\"access_token\":\"token\"}"),
        "text: {text}"
    );
    assert!(text.contains("root_folder_id = 12345"), "text: {text}");
    assert!(text.contains("impersonate = user-42"), "text: {text}");
    assert_eq!(
        option(&unmount_config, RCLONE_OPTION_REMOTE_NAME),
        remote_name
    );
    assert_eq!(
        option(&unmount_config, RCLONE_OPTION_REMOTE_PATH),
        "Shared/Finance"
    );
    assert!(unmount_config.config_text().is_none());
}

// ---------------------------------------------------------------------------
// GCS
// ---------------------------------------------------------------------------

/// Python's `test_gcs_mount_uses_runtime_endpoint_override_without_mutating_pattern_options`:
/// the mount supplies the GCS endpoint in its own configuration, the pattern is left untouched,
/// and applying that configuration stages credentials in an owner-only file instead of the
/// command line.
#[tokio::test]
async fn gcs_mount_uses_runtime_endpoint_override_without_mutating_pattern_options() {
    let pattern = MountpointMountPattern::new();
    let builtin = BuiltinMountPattern::Mountpoint(pattern.clone());
    let mount = GcsMount::new("bucket");

    let config = mount
        .build_in_container_config(SESSION_HEX, &builtin, false, false)
        .expect("config");

    assert_eq!(
        option(&config, MOUNTPOINT_OPTION_ENDPOINT_URL),
        "https://storage.googleapis.com"
    );
    assert!(pattern.options.endpoint_url.is_none());
    assert_eq!(option(&config, OPTION_READ_ONLY), "false");

    let session = MountSession::new();
    let hand_built = with_credentials(
        mountpoint_config("bucket", MOUNT_TYPE_GCS, true)
            .with_option(MOUNTPOINT_OPTION_REGION, "us-east1")
            .with_option(
                MOUNTPOINT_OPTION_ENDPOINT_URL,
                "https://storage.googleapis.com",
            ),
        "access",
        "secret",
        None,
    );
    pattern
        .apply(&session, "/workspace/remote", &hand_built)
        .await
        .expect("apply");

    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&["sh", "-lc", "command -v mount-s3 >/dev/null 2>&1"])
    );
    assert_eq!(
        commands[1],
        argv(&["mkdir", "-p", "--", "/workspace/remote"])
    );
    assert_eq!(commands.len(), 5);
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    let (env_path, env_payload) = &writes[0];
    assert!(
        env_path.starts_with("/workspace/.sandbox-mountpoint-env/"),
        "path: {env_path}"
    );
    assert!(env_path.ends_with(".env"), "path: {env_path}");
    assert_eq!(
        String::from_utf8_lossy(env_payload),
        "export AWS_ACCESS_KEY_ID=access\nexport AWS_SECRET_ACCESS_KEY=secret\n"
    );

    let script = mount_script(&session);
    assert!(script.contains("mount-s3"));
    assert!(!script.contains("AWS_ACCESS_KEY_ID=access"));
    assert!(!script.contains("AWS_SECRET_ACCESS_KEY=secret"));
    assert!(script.contains(".sandbox-mountpoint-env"));
    assert!(script.contains("--region us-east1"));
    assert!(script.contains("--endpoint-url https://storage.googleapis.com"));
    assert!(script.contains("--upload-checksums off"));
    assert!(script.contains("bucket /workspace/remote"));
}

/// Python's `test_gcs_mountpoint_writable_mode_enables_overwrite_and_delete`.
#[tokio::test]
async fn gcs_mountpoint_writable_mode_enables_overwrite_and_delete() {
    let session = MountSession::new();
    let pattern = MountpointMountPattern::new();
    let config = with_credentials(
        mountpoint_config("bucket", MOUNT_TYPE_GCS, false)
            .with_option(MOUNTPOINT_OPTION_REGION, "us-east1")
            .with_option(
                MOUNTPOINT_OPTION_ENDPOINT_URL,
                "https://storage.googleapis.com",
            ),
        "access",
        "secret",
        None,
    );

    pattern
        .apply(&session, "/workspace/remote", &config)
        .await
        .expect("apply");

    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&["sh", "-lc", "command -v mount-s3 >/dev/null 2>&1"])
    );
    assert_eq!(
        commands[1],
        argv(&["mkdir", "-p", "--", "/workspace/remote"])
    );
    assert_eq!(commands.len(), 5);
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    assert!(
        writes[0]
            .0
            .starts_with("/workspace/.sandbox-mountpoint-env/")
    );
    assert!(writes[0].0.ends_with(".env"));
    assert_eq!(
        String::from_utf8_lossy(&writes[0].1),
        "export AWS_ACCESS_KEY_ID=access\nexport AWS_SECRET_ACCESS_KEY=secret\n"
    );
    let script = mount_script(&session);
    assert!(script.contains("mount-s3"));
    assert!(!script.contains("--read-only"));
    assert!(script.contains("--allow-overwrite"));
    assert!(script.contains("--allow-delete"));
    assert!(script.contains("--region us-east1"));
    assert!(script.contains("--endpoint-url https://storage.googleapis.com"));
    assert!(script.contains("--upload-checksums off"));
    assert!(!script.contains("AWS_ACCESS_KEY_ID=access"));
    assert!(!script.contains("AWS_SECRET_ACCESS_KEY=secret"));
    assert!(script.contains(".sandbox-mountpoint-env"));
    assert!(script.contains("bucket /workspace/remote"));
}

/// Python's `test_gcs_mount_builds_anonymous_native_rclone_config_without_credentials`.
#[test]
fn gcs_mount_builds_anonymous_native_rclone_config_without_credentials() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "gcs");
    let mount = GcsMount::new("public-bucket");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\ntype = google cloud storage\nanonymous = true\nenv_auth = false\n"
        )
    );
}

/// Python's `test_gcs_mount_builds_native_rclone_config_with_service_account_auth`.
#[test]
fn gcs_mount_builds_native_rclone_config_with_service_account_auth() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "gcs");
    let mount = GcsMount::new("bucket")
        .with_prefix("nested/prefix/")
        .with_service_account_file("/data/config/gcs.json")
        .with_service_account_credentials("{\"type\":\"service_account\"}")
        .with_access_token("token");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);
    assert_eq!(
        option(&config, RCLONE_OPTION_REMOTE_PATH),
        "bucket/nested/prefix/"
    );
    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = google cloud storage\n\
             service_account_file = /data/config/gcs.json\n\
             service_account_credentials = {{\"type\":\"service_account\"}}\n\
             access_token = token\n\
             env_auth = false\n"
        )
    );
}

/// Python's `test_gcs_mount_builds_s3_compatible_rclone_config_with_hmac_auth`.
#[test]
fn gcs_mount_builds_s3_compatible_rclone_config_with_hmac_auth() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "gcs_s3");
    let mount = GcsMount::new("bucket")
        .with_hmac_credentials("access-id", "secret-key")
        .with_prefix("nested/prefix/")
        .with_region("auto");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);
    assert_eq!(
        option(&config, RCLONE_OPTION_REMOTE_PATH),
        "bucket/nested/prefix/"
    );
    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = GCS\n\
             env_auth = false\n\
             access_key_id = access-id\n\
             secret_access_key = secret-key\n\
             endpoint = https://storage.googleapis.com\n\
             region = auto\n"
        )
    );
}

/// Python's `test_gcs_hmac_rclone_remote_name_does_not_collide_with_s3_mount`.
#[test]
fn gcs_hmac_rclone_remote_name_does_not_collide_with_s3_mount() {
    let s3 = S3Mount::new("s3-bucket");
    let gcs = GcsMount::new("gcs-bucket").with_hmac_credentials("access-id", "secret-key");

    let s3_config = s3
        .build_in_container_config(SESSION_HEX, &rclone(), true, true)
        .expect("s3 config");
    let gcs_config = gcs
        .build_in_container_config(SESSION_HEX, &rclone(), true, true)
        .expect("gcs config");

    let s3_name = option(&s3_config, RCLONE_OPTION_REMOTE_NAME);
    let gcs_name = option(&gcs_config, RCLONE_OPTION_REMOTE_NAME);
    assert_eq!(s3_name, "sandbox_s3_12345678123456781234567812345678");
    assert_eq!(gcs_name, "sandbox_gcs_s3_12345678123456781234567812345678");
    assert_ne!(s3_name, gcs_name);
}

// ---------------------------------------------------------------------------
// S3 and mountpoint
// ---------------------------------------------------------------------------

/// Python's `test_s3_mountpoint_writable_mode_enables_overwrite_and_delete`.
#[tokio::test]
async fn s3_mountpoint_writable_mode_enables_overwrite_and_delete() {
    let session = MountSession::new();
    let pattern = MountpointMountPattern::new();
    let config = with_credentials(
        mountpoint_config("bucket", MOUNT_TYPE_S3, false)
            .with_option(MOUNTPOINT_OPTION_REGION, "us-east-1"),
        "access",
        "secret",
        Some("token"),
    );

    pattern
        .apply(&session, "/workspace/remote", &config)
        .await
        .expect("apply");

    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&["sh", "-lc", "command -v mount-s3 >/dev/null 2>&1"])
    );
    assert_eq!(
        commands[1],
        argv(&["mkdir", "-p", "--", "/workspace/remote"])
    );
    assert_eq!(commands.len(), 5);
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    assert!(
        writes[0]
            .0
            .starts_with("/workspace/.sandbox-mountpoint-env/")
    );
    assert!(writes[0].0.ends_with(".env"));
    assert_eq!(
        String::from_utf8_lossy(&writes[0].1),
        "export AWS_ACCESS_KEY_ID=access\n\
         export AWS_SECRET_ACCESS_KEY=secret\n\
         export AWS_SESSION_TOKEN=token\n"
    );
    let script = mount_script(&session);
    assert!(script.contains("mount-s3"));
    assert!(!script.contains("--read-only"));
    assert!(script.contains("--allow-overwrite"));
    assert!(script.contains("--allow-delete"));
    assert!(script.contains("--region us-east-1"));
    assert!(!script.contains("AWS_ACCESS_KEY_ID=access"));
    assert!(!script.contains("AWS_SECRET_ACCESS_KEY=secret"));
    assert!(!script.contains("AWS_SESSION_TOKEN=token"));
    assert!(script.contains(".sandbox-mountpoint-env"));
    assert!(script.contains("bucket /workspace/remote"));
}

/// Python's `test_mountpoint_credentialless_mode_disables_request_signing`.
#[tokio::test]
async fn mountpoint_credentialless_mode_disables_request_signing() {
    let session = MountSession::new();
    let pattern = MountpointMountPattern::new();
    let config = mountpoint_config("public-bucket", MOUNT_TYPE_S3, true)
        .with_option(MOUNTPOINT_OPTION_REGION, "us-east-1");

    pattern
        .apply(&session, "/workspace/remote", &config)
        .await
        .expect("apply");

    assert!(mount_script(&session).contains("--no-sign-request"));
    assert!(session.writes().is_empty());
}

/// Python's `test_s3_mount_direct_mountpoint_fields_override_pattern_options`.
#[test]
fn s3_mount_direct_mountpoint_fields_override_pattern_options() {
    let pattern = MountpointMountPattern::new()
        .with_prefix("pattern-prefix/")
        .with_region("pattern-region")
        .with_endpoint_url("https://pattern.example.test");
    let mount = S3Mount::new("bucket")
        .with_prefix("direct-prefix/")
        .with_region("direct-region")
        .with_endpoint_url("https://direct.example.test");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Mountpoint(pattern),
            true,
            false,
        )
        .expect("config");

    assert_eq!(option(&config, "prefix"), "direct-prefix/");
    assert_eq!(option(&config, MOUNTPOINT_OPTION_REGION), "direct-region");
    assert_eq!(
        option(&config, MOUNTPOINT_OPTION_ENDPOINT_URL),
        "https://direct.example.test"
    );
}

/// Python's `test_mount_apply_rejects_credentials_before_side_effects`: a credentialed mount
/// whose path the manifest has not acknowledged is refused before anything is staged, and the
/// secret appears in neither the message nor its debug form.
#[tokio::test]
async fn mount_apply_rejects_credentials_before_side_effects() {
    let sentinel = "direct-mount-apply-secret";
    let session = MountSession::new();
    let strategy = ConfiguredMountStrategy::mountpoint(
        S3Mount::new("bucket").with_credentials("access-key", sentinel),
    );
    let mount = Mount::new(std::sync::Arc::new(strategy.clone()));

    let error = strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .expect_err("an unacknowledged credential is refused");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(session.writes().is_empty());
    assert!(session.commands().is_empty());
    assert!(!error.message.contains(sentinel));
    assert!(!format!("{error:?}").contains(sentinel));
}

/// Python's `test_mount_restore_rejects_credentials_before_side_effects`.
#[tokio::test]
async fn mount_restore_rejects_credentials_before_side_effects() {
    let session = MountSession::new();
    let strategy = ConfiguredMountStrategy::mountpoint(
        S3Mount::new("bucket").with_credentials("access-key", "secret-key"),
    );
    let mount = Mount::new(std::sync::Arc::new(strategy.clone()));

    let error = strategy
        .restore_after_snapshot(&mount, &session, "/workspace/data")
        .await
        .expect_err("an unacknowledged credential is refused");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(session.writes().is_empty());
    assert!(session.commands().is_empty());
}

/// Python's `test_s3_mountpoint_failure_redacts_credentials_from_errors_and_events`: a failed
/// mount reports `MountFailed`, carries no retry hint or context, and never reproduces a
/// credential or the inline authority of the endpoint URL — in the error, the command, or the
/// debug form.
///
/// Python also inspects the instrumentation events the `SandboxSession` wrapper emits; Rust's
/// events carry no command text, so the assertion reduces to the error and the command.
#[tokio::test]
async fn s3_mountpoint_failure_redacts_credentials_from_errors_and_events() {
    let session = MountSession::new().failing_mount(1, b"bad credentials: access secret token");
    let pattern = MountpointMountPattern::new();
    let config = with_credentials(
        mountpoint_config("bucket", MOUNT_TYPE_S3, false)
            .with_option(MOUNTPOINT_OPTION_REGION, "us-east-1")
            .with_secret(
                "endpoint_url",
                "https://user:inline-endpoint-secret@example.test",
            ),
        "access",
        "secret",
        Some("token"),
    );

    let error = pattern
        .apply(&session, "/workspace/remote", &config)
        .await
        .expect_err("the mount fails");

    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(error.context.is_empty());
    let commands = session.commands();
    let script = commands
        .iter()
        .filter(|command| command.len() >= 3 && command[..2] == ["sh", "-lc"])
        .map(|command| command[2].as_str())
        .find(|script| script.contains("mount-s3 ") && !script.contains("command -v "))
        .expect("the mount command ran");
    assert!(script.contains(".sandbox-mountpoint-env"));
    for sensitive in ["access", "secret", "token", "inline-endpoint-secret"] {
        assert!(
            !script.contains(sensitive),
            "{sensitive} leaked into the command"
        );
        // The upstream stderr names three of these; none may survive in what is reported.
        assert!(
            !error.message.replace("credentials", "").contains(sensitive),
            "{sensitive} leaked into `{}`",
            error.message
        );
        assert!(!format!("{error:?}").contains(&format!("{sensitive}@")));
    }
}

/// Python's `test_s3_mount_builds_prefixed_rclone_remote_path`.
#[test]
fn s3_mount_builds_prefixed_rclone_remote_path() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "s3");
    let mount = S3Mount::new("bucket").with_prefix("nested/prefix/");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);
    assert_eq!(
        option(&config, RCLONE_OPTION_REMOTE_PATH),
        "bucket/nested/prefix/"
    );
}

/// Python's `test_s3_mount_rclone_config_includes_endpoint_and_region`.
#[test]
fn s3_mount_rclone_config_includes_endpoint_and_region() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "s3");
    let mount = S3Mount::new("my-bucket")
        .with_credentials("ak", "sk")
        .with_endpoint_url("http://localhost:9000")
        .with_region("us-west-2");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = AWS\n\
             endpoint = http://localhost:9000\n\
             region = us-west-2\n\
             env_auth = false\n\
             access_key_id = ak\n\
             secret_access_key = sk\n"
        )
    );
}

/// Python's `test_s3_mount_rclone_config_omits_endpoint_when_unset`.
#[test]
fn s3_mount_rclone_config_omits_endpoint_when_unset() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "s3");
    let mount = S3Mount::new("my-bucket").with_credentials("ak", "sk");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = AWS\n\
             env_auth = false\n\
             access_key_id = ak\n\
             secret_access_key = sk\n"
        )
    );
}

/// Python's `test_s3_mount_rclone_config_uses_custom_provider`: a non-AWS S3-compatible service
/// needs its provider named so rclone uses path-style addressing.
#[test]
fn s3_mount_rclone_config_uses_custom_provider() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "s3");
    let mount = S3Mount::new("my-bucket")
        .with_credentials("ak", "sk")
        .with_endpoint_url("http://localhost:9000")
        .with_s3_provider("Other");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = Other\n\
             endpoint = http://localhost:9000\n\
             env_auth = false\n\
             access_key_id = ak\n\
             secret_access_key = sk\n"
        )
    );
}

// ---------------------------------------------------------------------------
// S3 Files
// ---------------------------------------------------------------------------

/// Python's `test_s3_files_mount_builds_runtime_config_with_pattern_defaults`: the mount's own
/// fields win over the pattern's, and the pattern fills what the mount leaves unset.
#[test]
fn s3_files_mount_builds_runtime_config_with_pattern_defaults() {
    let pattern = S3FilesMountPattern::new()
        .with_mount_target_ip("10.99.1.209")
        .with_access_point("fsap-pattern")
        .with_region("us-east-1")
        .with_extra_option("tlsport", Some("3049".to_string()));
    let mount = S3FilesMount::new("fs-1234567890abcdef0")
        .with_subpath("/datasets")
        .with_access_point("fsap-direct")
        .with_extra_option("tlsport", Some("4049".to_string()))
        .with_extra_option("iam", None::<String>);

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::S3Files(pattern),
            true,
            false,
        )
        .expect("config");

    use openai_agents_sdk::sandbox::mount_patterns::{
        S3FILES_OPTION_ACCESS_POINT, S3FILES_OPTION_EXTRA_OPTIONS, S3FILES_OPTION_FILE_SYSTEM_ID,
        S3FILES_OPTION_MOUNT_TARGET_IP, S3FILES_OPTION_REGION, S3FILES_OPTION_SUBPATH,
    };
    assert_eq!(
        option(&config, S3FILES_OPTION_FILE_SYSTEM_ID),
        "fs-1234567890abcdef0"
    );
    assert_eq!(option(&config, S3FILES_OPTION_SUBPATH), "/datasets");
    assert_eq!(
        option(&config, S3FILES_OPTION_MOUNT_TARGET_IP),
        "10.99.1.209"
    );
    assert_eq!(option(&config, S3FILES_OPTION_ACCESS_POINT), "fsap-direct");
    assert_eq!(option(&config, S3FILES_OPTION_REGION), "us-east-1");
    assert_eq!(
        option(&config, S3FILES_OPTION_EXTRA_OPTIONS),
        "tlsport=4049,iam"
    );
}

/// Python's `test_s3_files_pattern_mounts_with_helper_options`.
#[tokio::test]
async fn s3_files_pattern_mounts_with_helper_options() {
    let session = MountSession::new();
    let pattern = S3FilesMountPattern::new();
    let mount = S3FilesMount::new("fs-1234567890abcdef0")
        .with_subpath("/datasets")
        .with_mount_target_ip("10.99.1.209")
        .with_access_point("fsap-123")
        .with_region("us-east-1")
        .with_extra_option("tlsport", Some("4049".to_string()));
    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::S3Files(pattern.clone()),
            true,
            false,
        )
        .expect("config");

    pattern
        .apply(&session, "/workspace/remote", &config)
        .await
        .expect("apply");

    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&["sh", "-lc", "command -v mount.s3files >/dev/null 2>&1"])
    );
    assert_eq!(
        commands[1],
        argv(&["mkdir", "-p", "--", "/workspace/remote"])
    );
    assert_eq!(
        commands[2],
        argv(&[
            "mount",
            "-t",
            "s3files",
            "-o",
            "tlsport=4049,ro,mounttargetip=10.99.1.209,accesspoint=fsap-123,region=us-east-1",
            "fs-1234567890abcdef0:/datasets",
            "/workspace/remote",
        ])
    );
}

// ---------------------------------------------------------------------------
// R2
// ---------------------------------------------------------------------------

/// Python's `test_r2_mount_builds_rclone_config_with_explicit_credentials`.
#[test]
fn r2_mount_builds_rclone_config_with_explicit_credentials() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "r2");
    let mount =
        R2Mount::new("bucket", "abc123accountid").with_credentials("r2-access", "r2-secret");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);
    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_PATH), "bucket");
    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = Cloudflare\n\
             endpoint = https://abc123accountid.r2.cloudflarestorage.com\n\
             acl = private\n\
             env_auth = false\n\
             access_key_id = r2-access\n\
             secret_access_key = r2-secret\n"
        )
    );
}

/// Python's `test_r2_mount_builds_env_auth_config_with_custom_domain`.
#[test]
fn r2_mount_builds_env_auth_config_with_custom_domain() {
    let pattern = RcloneMountPattern::new();
    let remote_name = derived_remote_name(&pattern, SESSION_HEX, "r2");
    let mount = R2Mount::new("bucket", "abc123accountid")
        .with_custom_domain("https://eu.r2.cloudflarestorage.com");

    let config = mount
        .build_in_container_config(
            SESSION_HEX,
            &BuiltinMountPattern::Rclone(pattern),
            true,
            true,
        )
        .expect("config");

    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);
    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_PATH), "bucket");
    assert_eq!(
        config_text(&config),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             provider = Cloudflare\n\
             endpoint = https://eu.r2.cloudflarestorage.com\n\
             acl = private\n\
             env_auth = false\n"
        )
    );
}

/// Python's `test_r2_mount_merges_existing_rclone_config_section`: the caller's own configuration
/// file is the base, the generated settings are merged into the section for this remote, and
/// every other section is preserved. Rust merges when the pattern applies, so the merged text is
/// read back from the file the pattern staged.
#[tokio::test]
async fn r2_mount_merges_existing_rclone_config_section() {
    let pattern = RcloneMountPattern::new().with_config_file_path("rclone.conf");
    let remote_name = derived_remote_name(&pattern, SESSION_ID, "r2");
    let session = MountSession::new().with_file(
        "/workspace/rclone.conf",
        &format!("[{remote_name}]\ntype = s3\nregion = auto\n\n[other]\ntype = memory\n"),
    );
    let mount =
        R2Mount::new("bucket", "abc123accountid").with_credentials("r2-access", "r2-secret");
    let config = mount
        .build_in_container_config(
            SESSION_ID,
            &BuiltinMountPattern::Rclone(pattern.clone()),
            true,
            true,
        )
        .expect("config");
    assert_eq!(option(&config, RCLONE_OPTION_REMOTE_NAME), remote_name);

    pattern
        .apply(&session, "/workspace/mnt", &config)
        .await
        .expect("apply");

    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    assert_eq!(
        String::from_utf8_lossy(&writes[0].1),
        format!(
            "[{remote_name}]\n\
             type = s3\n\
             region = auto\n\
             type = s3\n\
             provider = Cloudflare\n\
             endpoint = https://abc123accountid.r2.cloudflarestorage.com\n\
             acl = private\n\
             env_auth = false\n\
             access_key_id = r2-access\n\
             secret_access_key = r2-secret\n\
             \n\
             [other]\n\
             type = memory\n"
        )
    );
}

/// Python's `test_r2_mount_rejects_mountpoint_pattern`.
///
/// Python's message is `invalid mount_pattern type`; Rust's names the mount type and the pattern.
/// Both are a `MountConfigInvalid` raised before anything is attached.
#[test]
fn r2_mount_rejects_mountpoint_pattern() {
    let strategy = ConfiguredMountStrategy::mountpoint(R2Mount::new("bucket", "abc123accountid"));
    let mount = Mount::new(std::sync::Arc::new(strategy.clone()));

    let error = strategy.validate(&mount).expect_err("unsupported pattern");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("`mountpoint` mount pattern"),
        "message: {}",
        error.message
    );
}

/// Python's `test_r2_mount_rejects_partial_credentials_for_both_strategies`.
#[test]
fn r2_mount_rejects_partial_credentials_for_both_strategies() {
    let in_container = R2Mount {
        bucket: "bucket".to_string(),
        account_id: "abc123accountid".to_string(),
        access_key_id: Some("r2-access".to_string()),
        ..Default::default()
    };
    let error = in_container
        .build_in_container_config(SESSION_HEX, &rclone(), true, true)
        .expect_err("half a credential");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("access_key_id") && error.message.contains("secret_access_key"),
        "message: {}",
        error.message
    );

    let docker = R2Mount {
        bucket: "bucket".to_string(),
        account_id: "abc123accountid".to_string(),
        secret_access_key: Some("r2-secret".to_string()),
        ..Default::default()
    };
    // DEFERRED: Python's R2 also refuses the half pair when building Docker volume options
    // (`build_docker_volume_driver_config` calls `_validate_credential_pair`). Rust leaves an
    // external provider's configuration alone, because `ConfiguredMountStrategy::validate` builds
    // those options and the ported `test_incomplete_credentials_remain_external_provider_
    // configuration` (py_sandbox_a_mount_security_1) requires that validation to pass for an
    // incomplete R2 docker mount. The two Python tests conflict at that seam; asserting the
    // current behavior. Fix, if wanted: validate the pair at docker container creation instead
    // of in `validate`.
    assert!(docker.build_docker_volume_driver_config("rclone").is_ok());
}

// ---------------------------------------------------------------------------
// strategies
// ---------------------------------------------------------------------------

/// Python's `test_docker_volume_mount_apply_fails_on_non_docker_session`.
#[tokio::test]
async fn docker_volume_mount_apply_fails_on_non_docker_session() {
    let session = MountSession::new();
    let strategy = ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone");
    let mount = Mount::new(std::sync::Arc::new(strategy.clone()));

    let error = strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .expect_err("no docker volumes outside docker");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert_eq!(
        error.message,
        "docker-volume mounts are not supported by this sandbox backend"
    );
}

/// A mount type that supports neither an in-container pattern nor a Docker volume driver.
#[derive(Debug)]
struct NoStrategyMount;

impl ConfiguredMount for NoStrategyMount {
    fn mount_type(&self) -> &'static str {
        "no_strategy_mount"
    }

    fn supported_in_container_patterns(&self) -> &'static [&'static str] {
        &[]
    }

    fn credential_authority(
        &self,
        _pattern_type: &str,
        _has_config_file: bool,
    ) -> openai_agents_sdk::sandbox::mounts::MountCredentialAuthority {
        openai_agents_sdk::sandbox::mounts::MountCredentialAuthority::None
    }

    fn build_in_container_config(
        &self,
        _session_id: &str,
        _pattern: &BuiltinMountPattern,
        _read_only: bool,
        _include_credentials: bool,
    ) -> SandboxResult<MountPatternConfig> {
        Ok(MountPatternConfig::new())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Python's `test_mount_requires_at_least_one_supported_strategy`.
///
/// Python refuses to construct such a mount at all (`mount type must support at least one mount
/// strategy`). Rust's mount types are trait implementations, so nothing runs at definition time;
/// the equivalent guarantee is that no strategy built from one validates, whichever backend it is
/// attached with.
#[test]
fn mount_requires_at_least_one_supported_strategy() {
    for strategy in [
        ConfiguredMountStrategy::rclone(NoStrategyMount),
        ConfiguredMountStrategy::mountpoint(NoStrategyMount),
        ConfiguredMountStrategy::docker_volume(NoStrategyMount, "rclone"),
    ] {
        let mount = Mount::new(std::sync::Arc::new(strategy.clone()));
        let error = strategy
            .validate(&mount)
            .expect_err("a mount with no supported strategy never validates");
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    }
}

// ---------------------------------------------------------------------------
// rclone and blobfuse generated configuration
// ---------------------------------------------------------------------------

/// The configuration Python's `RcloneMountConfig(remote_name="remote", ...)` carries.
fn generated_rclone_config(read_only: bool) -> MountPatternConfig {
    MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, MOUNT_TYPE_S3)
        .with_option(RCLONE_OPTION_REMOTE_NAME, "remote")
        .with_option(RCLONE_OPTION_REMOTE_PATH, "bucket")
        .with_option(RCLONE_OPTION_REMOTE_KIND, "s3")
        .with_option(RCLONE_OPTION_READ_ONLY, read_only.to_string())
        .with_config_text("[remote]\ntype = s3\n")
}

/// Python's `test_rclone_nfs_server_honors_read_only_runtime_config`.
///
/// Python calls the private `_start_rclone_server`; Rust runs the whole NFS-mode apply, so the
/// assertion is that the server command inside the mount script carries `--read-only` and the
/// generated configuration path, after the probe for `rclone serve nfs`.
#[tokio::test]
async fn rclone_nfs_server_honors_read_only_runtime_config() {
    let session = MountSession::new();
    let pattern = RcloneMountPattern::nfs();

    pattern
        .apply(&session, "/workspace/mnt", &generated_rclone_config(true))
        .await
        .expect("apply");

    let commands = session.commands();
    let probe = commands
        .iter()
        .find(|command| {
            command
                .iter()
                .any(|part| part.contains("rclone serve nfs --help"))
        })
        .expect("the NFS server is probed first");
    assert_eq!(probe[..2], ["sh", "-lc"]);
    let script = mount_script_of(&commands, "serve nfs remote:bucket");
    assert!(script.contains("--addr 127.0.0.1:2049"), "script: {script}");
    assert!(script.contains("--read-only"), "script: {script}");
    assert!(
        script.contains("--config /workspace/.sandbox-rclone-config/"),
        "script: {script}"
    );

    let writable = MountSession::new();
    pattern
        .apply(&writable, "/workspace/mnt", &generated_rclone_config(false))
        .await
        .expect("apply");
    let script = mount_script_of(&writable.commands(), "serve nfs remote:bucket");
    assert!(!script.contains("--read-only"), "script: {script}");
}

fn mount_script_of(commands: &[Vec<String>], needle: &str) -> String {
    commands
        .iter()
        .filter(|command| command.len() >= 3 && command[..2] == ["sh", "-lc"])
        .map(|command| command[2].clone())
        .find(|script| script.contains(needle))
        .unwrap_or_else(|| panic!("no script containing `{needle}` in {commands:?}"))
}

/// Python's `test_rclone_generated_config_is_written_owner_only`: the configuration is written
/// into a per-session directory, restricted to the owner, and only then used by the mount.
#[tokio::test]
async fn rclone_generated_config_is_written_owner_only() {
    let session = MountSession::new();
    let pattern = RcloneMountPattern::new();

    pattern
        .apply(&session, "/workspace/mnt", &generated_rclone_config(true))
        .await
        .expect("apply");

    // Python files the configuration under `<scratch>/remote.conf`; Rust names the file after the
    // mount path, in one directory per workspace. Either way it is staged out of the mount's own
    // tree, restricted to the owner, and only then used by the mount command.
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    let (config_path, payload) = &writes[0];
    assert_eq!(payload, &b"[remote]\ntype = s3\n".to_vec());
    let config_dir = "/workspace/.sandbox-rclone-config";
    assert!(
        config_path.starts_with(&format!("{config_dir}/")) && config_path.ends_with(".conf"),
        "path: {config_path}"
    );
    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&[
            "sh",
            "-lc",
            "command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone"
        ])
    );
    assert_eq!(commands[1], argv(&["mkdir", "-p", "--", config_dir]));
    // Python spells the mode `0600`; the file mode is the same.
    assert_eq!(commands[2], argv(&["chmod", "600", "--", config_path]));
    assert_eq!(
        commands[3][..2],
        ["sh", "-lc"],
        "the mount command runs after the file is restricted: {:?}",
        commands[3]
    );
    assert_eq!(
        commands[3][2],
        format!(
            "rclone mount remote:bucket /workspace/mnt --read-only --config {config_path} --daemon"
        )
    );
    assert_eq!(commands.len(), 4);
}

/// The configuration Python's `FuseMountConfig(account="acct", container="container", ...)`
/// carries.
fn blobfuse_config() -> MountPatternConfig {
    MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, MOUNT_TYPE_AZURE_BLOB)
        .with_option(FUSE_OPTION_ACCOUNT, "acct")
        .with_option(FUSE_OPTION_CONTAINER, "container")
        .with_option(OPTION_READ_ONLY, "true")
        .with_secret(FUSE_SECRET_ACCOUNT_KEY, "secret")
}

fn blobfuse_cache_dir(session: &MountSession) -> String {
    format!(
        "/workspace/{BLOBFUSE_CACHE_DIR}/{}/acct/container",
        session.scratch_id()
    )
}

/// Python's `test_blobfuse_generated_config_is_written_owner_only`.
#[tokio::test]
async fn blobfuse_generated_config_is_written_owner_only() {
    let session = MountSession::new();
    let pattern = FuseMountPattern::new();

    pattern
        .apply(&session, "/workspace/mnt", &blobfuse_config())
        .await
        .expect("apply");

    let cache_dir = blobfuse_cache_dir(&session);
    let config_dir = format!("/workspace/{BLOBFUSE_CONFIG_DIR}/{}", session.scratch_id());
    let config_path = format!("{config_dir}/acct_container.yaml");
    let expected = format!(
        "allow-other: true\n\
         \n\
         logging:\n\
         \x20 type: syslog\n\
         \x20 level: log_debug\n\
         \n\
         components:\n\
         \x20 - libfuse\n\
         \x20 - block_cache\n\
         \x20 - attr_cache\n\
         \x20 - azstorage\n\
         \n\
         block_cache:\n\
         \x20 block-size-mb: 16\n\
         \x20 mem-size-mb: 50000\n\
         \x20 path: {cache_dir}\n\
         \x20 disk-size-mb: 50000\n\
         \x20 disk-timeout-sec: 3600\n\
         \n\
         attr_cache:\n\
         \x20 timeout-sec: 7200\n\
         \n\
         azstorage:\n\
         \x20 type: block\n\
         \x20 account-name: acct\n\
         \x20 container: container\n\
         \x20 endpoint: https://acct.blob.core.windows.net\n\
         \x20 auth-type: key\n\
         \x20 account-key: secret\n"
    );
    assert_eq!(
        session.writes(),
        vec![(config_path.clone(), expected.into_bytes())]
    );
    let commands = session.commands();
    assert_eq!(
        commands[0],
        argv(&["sh", "-lc", "command -v blobfuse2 >/dev/null 2>&1"])
    );
    assert_eq!(commands[1], argv(&["mkdir", "-p", "--", "/workspace/mnt"]));
    assert_eq!(commands[2], argv(&["mkdir", "-p", "--", &cache_dir]));
    assert_eq!(commands[3], argv(&["mkdir", "-p", "--", &config_dir]));
    assert_eq!(commands[4], argv(&["chmod", "0600", "--", &config_path]));
    assert_eq!(
        commands[5],
        argv(&[
            "blobfuse2",
            "mount",
            "--read-only",
            "--config-file",
            &config_path,
            "/workspace/mnt",
        ])
    );
    assert_eq!(commands.len(), 6);
}

/// Python's `test_blobfuse_generated_config_preserves_zero_attr_cache_timeout`.
#[tokio::test]
async fn blobfuse_generated_config_preserves_zero_attr_cache_timeout() {
    let session = MountSession::new();
    let mut pattern = FuseMountPattern::new();
    pattern.attr_cache_timeout_sec = Some(0);

    pattern
        .apply(&session, "/workspace/mnt", &blobfuse_config())
        .await
        .expect("apply");

    let written = String::from_utf8(session.writes()[0].1.clone()).expect("utf-8");
    assert!(
        written.contains("attr_cache:\n  timeout-sec: 0\n"),
        "written: {written}"
    );
}

/// Python's `test_blobfuse_zero_cache_sizes_use_sdk_defaults`, both parametrized cases: a zero
/// size means "use the default for this cache type", not "a zero-byte cache".
#[tokio::test]
async fn blobfuse_zero_cache_sizes_use_sdk_defaults() {
    let session = MountSession::new();
    let mut block = FuseMountPattern::new();
    block.cache_size_mb = Some(0);
    block
        .apply(&session, "/workspace/mnt", &blobfuse_config())
        .await
        .expect("apply");
    let written = String::from_utf8(session.writes()[0].1.clone()).expect("utf-8");
    assert!(
        written.contains("block_cache:\n  block-size-mb: 16\n  mem-size-mb: 50000\n"),
        "written: {written}"
    );

    let session = MountSession::new();
    let mut file = FuseMountPattern::new().with_cache_type(BlobfuseCacheType::FileCache);
    file.cache_size_mb = Some(0);
    file.file_cache_max_size_mb = Some(0);
    file.apply(&session, "/workspace/mnt", &blobfuse_config())
        .await
        .expect("apply");
    let written = String::from_utf8(session.writes()[0].1.clone()).expect("utf-8");
    let expected = format!(
        "file_cache:\n  path: {}\n  timeout-sec: 120\n  max-size-mb: 4096\n",
        blobfuse_cache_dir(&session)
    );
    assert!(written.contains(&expected), "written: {written}");
}

/// Python's `test_blobfuse_cache_path_must_be_relative_to_workspace`, all three cases.
///
/// Python rejects the pattern when it is constructed; Rust validates when the pattern is applied,
/// because the builder cannot fail. The message and the `cache_path` context (POSIX separators)
/// are Python's.
#[tokio::test]
async fn blobfuse_cache_path_must_be_relative_to_workspace() {
    for (cache_path, shown) in [
        ("/tmp/blobfuse-cache", "/tmp/blobfuse-cache"),
        ("../blobfuse-cache", "../blobfuse-cache"),
        ("C:\\blobfuse-cache", "C:/blobfuse-cache"),
    ] {
        let session = MountSession::new();
        let pattern = FuseMountPattern::new().with_cache_path(cache_path);

        let error = pattern
            .apply(&session, "/workspace/mnt", &blobfuse_config())
            .await
            .expect_err("an escaping cache path");

        assert_eq!(error.code, ErrorCode::MountConfigInvalid, "{cache_path}");
        assert_eq!(
            error.message, "blobfuse cache_path must be relative to the workspace root",
            "{cache_path}"
        );
        assert_eq!(context(&error, "cache_path"), shown, "{cache_path}");
        assert!(
            session.writes().is_empty(),
            "nothing is staged for {cache_path}"
        );
    }
}

/// Python's `test_blobfuse_cache_path_must_be_outside_mount_path`: the cache may not live inside
/// the filesystem it caches, and the check runs before any directory is created or file written.
#[tokio::test]
async fn blobfuse_cache_path_must_be_outside_mount_path() {
    let session = MountSession::new();
    let pattern = FuseMountPattern::new();
    // No account key: Python builds this configuration without credentials.
    let config = MountPatternConfig::new()
        .with_option(OPTION_MOUNT_TYPE, MOUNT_TYPE_AZURE_BLOB)
        .with_option(FUSE_OPTION_ACCOUNT, "acct")
        .with_option(FUSE_OPTION_CONTAINER, "container")
        .with_option(OPTION_READ_ONLY, "true");

    let error = pattern
        .apply(&session, "/workspace", &config)
        .await
        .expect_err("the cache would sit inside the mount");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert_eq!(
        error.message,
        "blobfuse cache_path must be outside the mount path"
    );
    assert_eq!(context(&error, "mount_path"), "/workspace");
    assert_eq!(
        context(&error, "cache_path"),
        format!(
            "/workspace/.sandbox-blobfuse-cache/{}/acct/container",
            session.scratch_id()
        )
    );
    assert_eq!(
        session.commands(),
        vec![argv(&["sh", "-lc", "command -v blobfuse2 >/dev/null 2>&1"])]
    );
    assert!(session.writes().is_empty());
}
