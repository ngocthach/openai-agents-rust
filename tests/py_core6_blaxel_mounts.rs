//! Ports of the `TestMountsModule`, `TestDriveMounts`, `TestUnmountBucketLogging` and
//! `TestPersistWithMounts` groups of `tests/extensions/sandbox/test_blaxel.py`.
//!
//! Python drives the module-private helpers (`_mount_s3`, `_mount_gcs`, `_unmount_bucket`,
//! `_install_tool`, `_attach_drive`) against `_FakeMountSession`. The Rust helpers are private
//! too, so each case drives the public strategy entry points (`activate`, `deactivate`,
//! `teardown_for_snapshot`, `restore_after_snapshot`) against the same kind of fake session: one
//! that records every command and answers from a queue. The drive cases run against the shared
//! mock Blaxel sandbox API instead.
#![cfg(feature = "sandbox")]

#[path = "py_core6_blaxel_support/mod.rs"]
mod harness;
#[path = "py_sandbox_support/mod.rs"]
mod support;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use harness::{World, options, root_manifest, session_with, start};
use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelBucketProvider, BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy,
    BlaxelDriveMount, BlaxelDriveMountConfig, BlaxelDriveMountStrategy,
};
use openai_agents_sdk::sandbox::errors::{SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::mount_entries::{GcsMount, R2Mount, S3Mount};
use openai_agents_sdk::sandbox::mounts::{Mount, MountStrategy};
use openai_agents_sdk::sandbox::types::{ExecResult, FileEntry};
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxSession};
use support::Shared;

// ---------------------------------------------------------------------------------------------
// A fake session that records commands and answers from a queue (`_FakeMountSession`)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Recorder {
    exec_calls: Vec<Vec<String>>,
    writes: Vec<(String, Vec<u8>)>,
    skip_paths: Vec<String>,
    /// `(event, path)` for each credential lifecycle step, in order.
    events: Vec<(String, String)>,
    results: VecDeque<ExecResult>,
}

#[derive(Debug)]
struct FakeMountSession {
    manifest: Manifest,
    state: Mutex<Recorder>,
    skip_error: Mutex<Option<String>>,
}

fn result(exit_code: i32, stdout: &str, stderr: &str) -> ExecResult {
    ExecResult {
        exit_code,
        stdout: stdout.as_bytes().to_vec(),
        stderr: stderr.as_bytes().to_vec(),
    }
}

fn ok() -> ExecResult {
    result(0, "", "")
}

fn failed() -> ExecResult {
    result(1, "", "")
}

impl FakeMountSession {
    fn new(results: Vec<ExecResult>) -> Self {
        FakeMountSession::with_manifest(Manifest::new().with_root("/workspace"), results)
    }

    fn with_manifest(manifest: Manifest, results: Vec<ExecResult>) -> Self {
        FakeMountSession {
            manifest,
            state: Mutex::new(Recorder {
                results: results.into(),
                ..Recorder::default()
            }),
            skip_error: Mutex::new(None),
        }
    }

    /// The `sh -c` script of every command, in order.
    fn commands(&self) -> Vec<String> {
        self.state
            .lock()
            .unwrap()
            .exec_calls
            .iter()
            .map(|argv| argv[2].clone())
            .collect()
    }

    fn exec_repr(&self) -> String {
        format!("{:?}", self.state.lock().unwrap().exec_calls)
    }

    fn writes(&self) -> Vec<(String, Vec<u8>)> {
        self.state.lock().unwrap().writes.clone()
    }

    fn skip_paths(&self) -> Vec<String> {
        self.state.lock().unwrap().skip_paths.clone()
    }

    fn events(&self) -> Vec<String> {
        self.state
            .lock()
            .unwrap()
            .events
            .iter()
            .map(|(event, _)| event.clone())
            .collect()
    }
}

#[async_trait]
impl SandboxSession for FakeMountSession {
    fn id(&self) -> &str {
        "fake"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "blaxel"
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
        let mut state = self.state.lock().unwrap();
        state.exec_calls.push(request.resolved_argv());
        Ok(state.results.pop_front().unwrap_or_else(ok))
    }

    async fn read_file(&self, _path: &str) -> SandboxResult<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let mut state = self.state.lock().unwrap();
        state.events.push(("write".into(), path.to_string()));
        state.writes.push((path.to_string(), contents.to_vec()));
        Ok(())
    }

    async fn mkdir(&self, _path: &str, _parents: bool) -> SandboxResult<()> {
        Ok(())
    }

    async fn rm(&self, _path: &str, _recursive: bool) -> SandboxResult<()> {
        Ok(())
    }

    async fn ls(&self, _path: &str) -> SandboxResult<Vec<FileEntry>> {
        Ok(Vec::new())
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }

    fn register_persist_workspace_skip_path(&self, path: &str) -> SandboxResult<String> {
        let mut state = self.state.lock().unwrap();
        state.events.push(("register".into(), path.to_string()));
        if let Some(message) = self.skip_error.lock().unwrap().clone() {
            return Err(SandboxError::new(ErrorCode::MountConfigInvalid, message));
        }
        state.skip_paths.push(path.to_string());
        Ok(path.to_string())
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn strategy(config: BlaxelCloudBucketMountConfig) -> (BlaxelCloudBucketMountStrategy, Mount) {
    let strategy = BlaxelCloudBucketMountStrategy::new(config);
    let mount = Mount::new(Arc::new(strategy.clone()));
    (strategy, mount)
}

async fn activate(
    config: BlaxelCloudBucketMountConfig,
    session: &FakeMountSession,
    mount_path: &str,
) -> SandboxResult<()> {
    let (strategy, mount) = strategy(config);
    strategy.activate(&mount, session, mount_path).await
}

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
            Some(_) => match c {
                '"' => quote = None,
                '\\' => {
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                _ => current.push(c),
            },
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

fn starts_with_any(commands: &[String], prefix: &str) -> bool {
    commands.iter().any(|command| command.starts_with(prefix))
}

// ---------------------------------------------------------------------------------------------
// _build_mount_config
// ---------------------------------------------------------------------------------------------

/// `test_build_mount_config_s3`
#[test]
fn s3_mount_translates_to_an_s3_bucket_config() {
    let mount = S3Mount::new("my-bucket")
        .with_credentials("AKID", "SECRET")
        .with_region("us-east-1")
        .with_prefix("data/");

    let config = BlaxelCloudBucketMountConfig::from_configured_mount(&mount).expect("config");

    assert_eq!(config.provider, BlaxelBucketProvider::S3);
    assert_eq!(config.bucket, "my-bucket");
    assert_eq!(config.access_key_id.as_deref(), Some("AKID"));
    assert_eq!(config.region.as_deref(), Some("us-east-1"));
    assert_eq!(config.prefix.as_deref(), Some("data/"));
    assert!(config.read_only);
}

/// `test_build_mount_config_r2` and `test_build_mount_config_r2_custom_domain`
#[test]
fn r2_mount_translates_to_an_r2_bucket_config() {
    let mount = R2Mount::new("r2-bucket", "acc123").with_credentials("R2KEY", "R2SECRET");
    let config = BlaxelCloudBucketMountConfig::from_configured_mount(&mount).expect("config");
    assert_eq!(config.provider, BlaxelBucketProvider::R2);
    assert!(
        config
            .endpoint_url
            .as_deref()
            .is_some_and(|url| url.contains("r2.cloudflarestorage.com")),
        "{config:?}"
    );

    let custom = R2Mount::new("r2-bucket", "acc123")
        .with_credentials("R2KEY", "R2SECRET")
        .with_custom_domain("https://custom.example.com");
    let config = BlaxelCloudBucketMountConfig::from_configured_mount(&custom).expect("config");
    assert_eq!(
        config.endpoint_url.as_deref(),
        Some("https://custom.example.com")
    );
}

/// `test_build_mount_config_gcs` and `test_build_mount_config_gcs_hmac`
#[test]
fn gcs_mount_translates_to_gcsfuse_or_s3fs() {
    let mount = GcsMount::new("gcs-bucket")
        .with_service_account_credentials("{\"type\":\"service_account\"}")
        .with_prefix("prefix/");
    let config = BlaxelCloudBucketMountConfig::from_configured_mount(&mount).expect("config");
    assert_eq!(config.provider, BlaxelBucketProvider::Gcs);
    assert!(config.service_account_key.is_some());

    let hmac = GcsMount::new("gcs-bucket")
        .with_hmac_credentials("GOOG1", "SECRET")
        .with_endpoint_url("https://storage.googleapis.com")
        .with_prefix("prefix/");
    let config = BlaxelCloudBucketMountConfig::from_configured_mount(&hmac).expect("config");
    assert_eq!(config.provider, BlaxelBucketProvider::S3);
    assert_eq!(config.access_key_id.as_deref(), Some("GOOG1"));
    assert_eq!(config.secret_access_key.as_deref(), Some("SECRET"));
    assert_eq!(
        config.endpoint_url.as_deref(),
        Some("https://storage.googleapis.com")
    );
    assert_eq!(config.prefix.as_deref(), Some("prefix/"));
}

/// `test_build_mount_config_unsupported`
#[test]
fn an_unsupported_mount_type_is_refused() {
    let error = BlaxelCloudBucketMountConfig::from_configured_mount(&BlaxelDriveMount::new("d"))
        .expect_err("a drive is not a bucket");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(
        error.message.contains("do not support"),
        "{}",
        error.message
    );
}

/// `test_validate_mount`
#[test]
fn a_plain_bucket_mount_validates() {
    let (strategy, mount) = strategy(BlaxelCloudBucketMountConfig::s3("test-bucket"));

    strategy.validate(&mount).expect("a valid mount");
}

// ---------------------------------------------------------------------------------------------
// _mount_s3
// ---------------------------------------------------------------------------------------------

/// `test_mount_s3_with_credentials`: the credential file is written to the workspace, restricted,
/// used by the mount, and removed; the secret never reaches a command.
#[tokio::test]
async fn s3_mount_with_credentials_uses_a_private_credential_file() {
    let secret = "s3-secret-command-sentinel";
    let session =
        FakeMountSession::new(vec![result(0, "/usr/bin/s3fs", ""), ok(), ok(), ok(), ok()]);
    let config = BlaxelCloudBucketMountConfig::s3("my-bucket")
        .with_credentials("AKID", secret)
        .with_region("us-east-1")
        .with_prefix("data/");

    activate(config, &session, "/mnt/s3").await.expect("mount");

    let commands = session.commands();
    assert_eq!(commands.len(), 5, "{commands:?}");
    assert!(commands[0].starts_with("which "));
    assert!(commands[1].starts_with("chmod 600 "));
    assert!(commands[2].starts_with("mkdir -p "));
    assert!(commands[3].starts_with("s3fs "));
    assert!(commands[4].starts_with("rm -f "));
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    let (path, payload) = &writes[0];
    let (parent, name) = path.rsplit_once('/').expect("an absolute path");
    assert_eq!(parent, "/workspace");
    assert!(name.starts_with(".openai-agents-s3fs-passwd-"), "{name}");
    assert_eq!(payload, format!("AKID:{secret}").as_bytes());
    assert!(!session.exec_repr().contains(secret));
    // The mount reads the file at the path it was written to.
    assert!(
        commands[3].contains(&format!("passwd_file={path}")),
        "{}",
        commands[3]
    );
    assert_eq!(session.skip_paths(), vec![name.to_string()]);
}

/// `test_mount_s3_fails_when_credential_cleanup_fails`
#[tokio::test]
async fn s3_mount_fails_when_the_credential_cleanup_fails() {
    let secret = "s3-cleanup-secret";
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok(), failed()]);
    let config = BlaxelCloudBucketMountConfig::s3("my-bucket").with_credentials("AKID", secret);

    let error = activate(config, &session, "/mnt/s3")
        .await
        .expect_err("a credential left behind is an error");

    assert!(
        error
            .message
            .contains("failed to remove mount credential file"),
        "{}",
        error.message
    );
    assert!(session.commands().last().unwrap().starts_with("rm -f "));
    assert!(!session.exec_repr().contains(secret));
    assert!(!format!("{error:?}").contains(secret));
    let writes = session.writes();
    let name = writes[0].0.rsplit('/').next().unwrap().to_string();
    assert_eq!(session.skip_paths(), vec![name]);
    assert_eq!(session.events(), vec!["register", "write"]);
}

/// `test_mount_credentials_reject_registration_before_write`: the skip path is registered before
/// the credential is written, so a registration the session refuses leaves nothing behind.
#[tokio::test]
async fn credential_mounts_register_the_skip_path_before_writing() {
    for provider in ["s3", "gcs"] {
        let session = FakeMountSession::new(vec![ok()]);
        *session.skip_error.lock().unwrap() = Some("registration rejected".into());
        let config = if provider == "s3" {
            BlaxelCloudBucketMountConfig::s3("bucket").with_credentials("AKID", "SECRET")
        } else {
            BlaxelCloudBucketMountConfig::gcs("bucket")
                .with_service_account_key("{\"private_key\":\"SECRET\"}")
        };

        let error = activate(config, &session, "/mnt/data")
            .await
            .expect_err("registration rejected");

        assert!(
            error.message.contains("registration rejected"),
            "{provider}: {}",
            error.message
        );
        assert_eq!(session.events(), vec!["register"], "{provider}");
        assert!(session.skip_paths().is_empty());
        assert!(session.writes().is_empty());
        let commands = session.commands();
        assert_eq!(commands.len(), 1, "{provider}: {commands:?}");
        assert!(commands[0].starts_with("which "));
    }
}

/// `test_mount_s3_public_bucket`
#[tokio::test]
async fn s3_public_bucket_needs_no_credential_file() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok()]);

    activate(
        BlaxelCloudBucketMountConfig::s3("public-bucket"),
        &session,
        "/mnt/pub",
    )
    .await
    .expect("mount");

    let commands = session.commands();
    assert_eq!(commands.len(), 3, "{commands:?}");
    assert!(!starts_with_any(&commands, "rm -f "));
    assert!(session.skip_paths().is_empty());
    assert!(session.writes().is_empty());
}

/// `test_mount_s3_with_endpoint` and `test_mount_bucket_dispatch_s3`
#[tokio::test]
async fn s3_mount_with_an_endpoint_passes_it_to_s3fs() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok()]);

    activate(
        BlaxelCloudBucketMountConfig::s3("endpoint-bucket")
            .with_endpoint_url("https://custom-s3.example.com"),
        &session,
        "/mnt/ep",
    )
    .await
    .expect("mount");

    let commands = session.commands();
    assert!(
        commands[2].contains("url=https://custom-s3.example.com"),
        "{commands:?}"
    );
}

/// `test_mount_s3_r2_sigv4`
#[tokio::test]
async fn r2_mount_uses_sigv4() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok(), ok()]);

    activate(
        BlaxelCloudBucketMountConfig::r2("r2-bucket", "acc")
            .with_credentials("KEY", "SECRET")
            .with_endpoint_url("https://acc.r2.cloudflarestorage.com"),
        &session,
        "/mnt/r2",
    )
    .await
    .expect("mount");

    let commands = session.commands();
    let mount = commands.iter().find(|c| c.starts_with("s3fs ")).unwrap();
    assert!(mount.contains("sigv4"), "{mount}");
    assert!(commands.last().unwrap().starts_with("rm -f "));
}

/// `test_mount_s3_fails`
#[tokio::test]
async fn s3_mount_failure_names_the_tool() {
    let session = FakeMountSession::new(vec![ok(), ok(), result(1, "", "mount error")]);

    let error = activate(
        BlaxelCloudBucketMountConfig::s3("fail-bucket"),
        &session,
        "/mnt/fail",
    )
    .await
    .expect_err("s3fs fails");

    assert!(
        error.message.contains("s3fs mount failed"),
        "{}",
        error.message
    );
    assert!(error.message.contains("mount error"), "{}", error.message);
}

// ---------------------------------------------------------------------------------------------
// _mount_gcs
// ---------------------------------------------------------------------------------------------

/// `test_mount_gcs_with_key`
#[tokio::test]
async fn gcs_mount_with_a_key_uses_a_private_key_file() {
    let key = "{\"private_key\":\"gcs-secret-command-sentinel\"}";
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok(), ok()]);
    let config = BlaxelCloudBucketMountConfig::gcs("gcs-bucket")
        .with_service_account_key(key)
        .with_prefix("data/");

    activate(config, &session, "/mnt/gcs").await.expect("mount");

    assert_eq!(session.commands().len(), 5);
    let writes = session.writes();
    assert_eq!(writes.len(), 1);
    let (path, payload) = &writes[0];
    let (parent, name) = path.rsplit_once('/').unwrap();
    assert_eq!(parent, "/workspace");
    assert!(name.starts_with(".openai-agents-gcs-creds-"), "{name}");
    assert_eq!(payload, key.as_bytes());
    assert!(!session.exec_repr().contains("gcs-secret-command-sentinel"));
}

/// `test_mount_gcs_fails_when_credential_cleanup_fails`
#[tokio::test]
async fn gcs_mount_fails_when_the_key_cleanup_fails() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok(), failed()]);
    let config = BlaxelCloudBucketMountConfig::gcs("gcs-bucket")
        .with_service_account_key("{\"private_key\":\"gcs-cleanup-secret\"}");

    let error = activate(config, &session, "/mnt/gcs")
        .await
        .expect_err("cleanup fails");

    assert!(
        error
            .message
            .contains("failed to remove mount credential file"),
        "{}",
        error.message
    );
    assert!(session.commands().last().unwrap().starts_with("rm -f "));
    assert!(!session.exec_repr().contains("gcs-cleanup-secret"));
    let name = session.writes()[0]
        .0
        .rsplit('/')
        .next()
        .unwrap()
        .to_string();
    assert_eq!(session.skip_paths(), vec![name]);
    assert_eq!(session.events(), vec!["register", "write"]);
}

/// `test_mount_gcs_quotes_generated_key_path`
#[tokio::test]
async fn gcs_key_path_is_shell_quoted() {
    let session = FakeMountSession::with_manifest(
        Manifest::new().with_root("/workspace data;echo not-executed"),
        vec![ok(), ok(), ok(), ok(), ok()],
    );

    activate(
        BlaxelCloudBucketMountConfig::gcs("gcs-bucket")
            .with_service_account_key("{\"private_key\":\"gcs-secret\"}"),
        &session,
        "/mnt/gcs",
    )
    .await
    .expect("mount");

    let key_path = session.writes()[0].0.clone();
    let mount_command = session.commands()[3].clone();
    let tokens = shlex_split(&mount_command);
    assert!(
        tokens.contains(&format!("--key-file={key_path}")),
        "{mount_command}"
    );
    assert!(!tokens.contains(&"echo".to_string()), "{mount_command}");
}

/// `test_mount_gcs_aborts_and_cleans_up_when_credential_chmod_fails`
#[tokio::test]
async fn gcs_mount_aborts_and_cleans_up_when_chmod_fails() {
    let session = FakeMountSession::new(vec![ok(), failed(), ok()]);
    let config = BlaxelCloudBucketMountConfig::gcs("gcs-bucket")
        .with_service_account_key("{\"private_key\":\"gcs-chmod-secret\"}");

    let error = activate(config, &session, "/mnt/gcs")
        .await
        .expect_err("chmod fails");

    assert!(
        error
            .message
            .contains("failed to restrict mount credential file permissions"),
        "{}",
        error.message
    );
    let commands = session.commands();
    assert_eq!(session.writes().len(), 1);
    assert!(starts_with_any(&commands, "chmod 600 "));
    assert!(starts_with_any(&commands, "rm -f "));
    assert!(!starts_with_any(&commands, "mkdir -p "));
    assert!(!starts_with_any(&commands, "gcsfuse "));
    assert!(!session.exec_repr().contains("gcs-chmod-secret"));
}

/// `test_mount_gcs_anonymous` and `test_mount_bucket_dispatch_gcs`
#[tokio::test]
async fn gcs_anonymous_mount_needs_no_key_file() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok()]);

    activate(
        BlaxelCloudBucketMountConfig::gcs("pub-gcs"),
        &session,
        "/mnt/pub-gcs",
    )
    .await
    .expect("mount");

    let commands = session.commands();
    assert_eq!(commands.len(), 3, "{commands:?}");
    assert!(commands[2].contains("--anonymous-access"), "{commands:?}");
    assert!(!starts_with_any(&commands, "rm -f "));
    assert!(session.skip_paths().is_empty());
}

/// `test_mount_gcs_fails`
#[tokio::test]
async fn gcs_mount_failure_names_the_tool() {
    let session = FakeMountSession::new(vec![ok(), ok(), result(1, "", "gcs error")]);

    let error = activate(
        BlaxelCloudBucketMountConfig::gcs("fail-gcs"),
        &session,
        "/mnt/fail-gcs",
    )
    .await
    .expect_err("gcsfuse fails");

    assert!(
        error.message.contains("gcsfuse mount failed"),
        "{}",
        error.message
    );
}

// ---------------------------------------------------------------------------------------------
// _unmount_bucket
// ---------------------------------------------------------------------------------------------

async fn deactivate(session: &FakeMountSession, mount_path: &str) -> SandboxResult<()> {
    let (strategy, mount) = strategy(BlaxelCloudBucketMountConfig::s3("b"));
    strategy.deactivate(&mount, session, mount_path).await
}

/// `test_unmount_bucket_fusermount`, `test_unmount_bucket_umount_fallback` and
/// `test_unmount_bucket_lazy`: each failure falls through to the next, weaker command.
#[tokio::test]
async fn unmount_falls_back_through_umount_to_a_lazy_umount() {
    for (results, expected) in [
        (vec![ok()], vec!["fusermount -u "]),
        (vec![failed(), ok()], vec!["fusermount -u ", "umount "]),
        (
            vec![failed(), failed(), ok()],
            vec!["fusermount -u ", "umount ", "umount -l "],
        ),
    ] {
        let session = FakeMountSession::new(results);

        deactivate(&session, "/mnt/test").await.expect("unmount");

        let commands = session.commands();
        assert_eq!(commands.len(), expected.len(), "{commands:?}");
        for (command, prefix) in commands.iter().zip(&expected) {
            assert!(command.starts_with(prefix), "{command}");
            assert!(command.ends_with("/mnt/test"), "{command}");
        }
    }
}

/// `test_teardown_for_snapshot`
#[tokio::test]
async fn teardown_for_snapshot_unmounts_the_bucket() {
    let session = FakeMountSession::new(vec![ok()]);
    let (strategy, mount) = strategy(BlaxelCloudBucketMountConfig::s3("test"));

    strategy
        .teardown_for_snapshot(&mount, &session, "/workspace/mnt/s3")
        .await
        .expect("teardown");

    assert_eq!(session.commands().len(), 1);
    assert!(session.commands()[0].starts_with("fusermount -u "));
}

/// `test_restore_after_snapshot`
#[tokio::test]
async fn restore_after_snapshot_mounts_the_bucket_again() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok()]);
    let (strategy, mount) = strategy(BlaxelCloudBucketMountConfig::s3("test"));

    strategy
        .restore_after_snapshot(&mount, &session, "/workspace/mnt/s3")
        .await
        .expect("restore");

    let commands = session.commands();
    assert_eq!(commands.len(), 3, "{commands:?}");
    assert!(commands[2].starts_with("s3fs "));
}

/// `test_restore_preserves_cleanup_error_over_mount_error`: when the mount fails and the
/// credential cleanup fails too, the cleanup failure is the one reported.
#[tokio::test]
async fn restore_reports_the_cleanup_error_over_the_mount_error() {
    let session = FakeMountSession::new(vec![
        ok(),
        ok(),
        ok(),
        result(1, "", "mount failed"),
        failed(),
    ]);
    let (strategy, mount) = strategy(
        BlaxelCloudBucketMountConfig::gcs("test")
            .with_service_account_key("{\"private_key\":\"gcs-cleanup-secret\"}"),
    );

    let error = strategy
        .restore_after_snapshot(&mount, &session, "/workspace/data")
        .await
        .expect_err("both steps fail");

    assert!(
        error
            .message
            .contains("failed to remove mount credential file"),
        "{}",
        error.message
    );
    assert!(!session.exec_repr().contains("gcs-cleanup-secret"));
}

// ---------------------------------------------------------------------------------------------
// _install_tool and _ensure_tool
// ---------------------------------------------------------------------------------------------

/// `test_ensure_tool_already_installed` and `test_ensure_tool_needs_install`
#[tokio::test]
async fn the_fuse_tool_is_installed_only_when_missing() {
    let present = FakeMountSession::new(vec![ok(), ok(), ok()]);
    activate(BlaxelCloudBucketMountConfig::s3("b"), &present, "/m")
        .await
        .expect("mount");
    // `which` succeeds, so the next command is the mkdir.
    assert!(present.commands()[1].starts_with("mkdir -p "));

    let missing = FakeMountSession::new(vec![failed(), result(0, "apt", ""), ok(), ok(), ok()]);
    activate(BlaxelCloudBucketMountConfig::s3("b"), &missing, "/m")
        .await
        .expect("mount");
    let commands = missing.commands();
    assert!(commands[0].starts_with("which "), "{commands:?}");
    assert!(commands[1].contains("which apk"), "{commands:?}");
    assert!(commands[2].starts_with("apt-get update"), "{commands:?}");
    assert!(commands[2].contains("install -y -qq s3fs"), "{commands:?}");
}

/// `test_install_tool_with_apk` and `test_install_tool_with_apt`
#[tokio::test]
async fn install_uses_the_images_package_manager() {
    let alpine = FakeMountSession::new(vec![failed(), result(0, "apk", ""), ok(), ok(), ok()]);
    activate(BlaxelCloudBucketMountConfig::s3("b"), &alpine, "/m")
        .await
        .expect("mount");
    assert_eq!(alpine.commands()[2], "apk add --no-cache s3fs-fuse");

    let debian = FakeMountSession::new(vec![failed(), result(0, "apt", ""), ok(), ok(), ok()]);
    activate(BlaxelCloudBucketMountConfig::gcs("b"), &debian, "/m")
        .await
        .expect("mount");
    let install = &debian.commands()[2];
    assert!(install.contains("gcsfuse"), "{install}");
    assert!(
        install.contains("apt-get install -y -qq gcsfuse"),
        "{install}"
    );
}

/// `test_install_tool_fails_after_retries`
#[tokio::test]
async fn install_gives_up_after_three_attempts() {
    let session = FakeMountSession::new(vec![
        failed(),
        result(0, "apt", ""),
        failed(),
        failed(),
        failed(),
    ]);

    let error = activate(BlaxelCloudBucketMountConfig::s3("b"), &session, "/m")
        .await
        .expect_err("install fails");

    assert!(
        error.message.contains("failed to install"),
        "{}",
        error.message
    );
    assert_eq!(error.code, ErrorCode::MountMissingTool);
    // which + detect + three attempts, and nothing after.
    assert_eq!(session.commands().len(), 5);
}

// ---------------------------------------------------------------------------------------------
// _activate
// ---------------------------------------------------------------------------------------------

/// `test_activate` and `test_deactivate`: a strategy mounts at the path it is handed and detaches
/// from the same one.
#[tokio::test]
async fn activate_and_deactivate_use_the_given_mount_path() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok()]);
    let (strategy, mount) = strategy(BlaxelCloudBucketMountConfig::s3("test"));

    strategy
        .activate(&mount, &session, "/workspace/mnt/s3")
        .await
        .expect("activate");
    strategy
        .deactivate(&mount, &session, "/workspace/mnt/s3")
        .await
        .expect("deactivate");

    let commands = session.commands();
    assert!(commands[1].contains("/workspace/mnt/s3"));
    assert!(commands[2].contains("/workspace/mnt/s3"));
    assert!(commands[3].starts_with("fusermount -u "));
}

/// `test_activate_preserves_safe_credential_cleanup_error`: the cleanup failure surfaces from
/// `activate` with no secret in the message.
#[tokio::test]
async fn activate_surfaces_the_credential_cleanup_error_without_the_secret() {
    let session = FakeMountSession::new(vec![ok(), ok(), ok(), ok(), failed()]);
    let (strategy, mount) = strategy(
        BlaxelCloudBucketMountConfig::s3("test").with_credentials("AKID", "s3-cleanup-secret"),
    );

    let error = strategy
        .activate(&mount, &session, "/workspace/data")
        .await
        .expect_err("cleanup fails");

    assert!(
        error
            .message
            .contains("failed to remove mount credential file")
    );
    assert!(!session.exec_repr().contains("s3-cleanup-secret"));
    assert!(!format!("{error:?}").contains("s3-cleanup-secret"));
}

// ---------------------------------------------------------------------------------------------
// Drive mounts (`_attach_drive`, `_detach_drive`)
// ---------------------------------------------------------------------------------------------

fn drive_strategy() -> (BlaxelDriveMountStrategy, Mount) {
    let strategy = BlaxelDriveMountStrategy::new(
        BlaxelDriveMountConfig::new("test-drive").with_drive_path("/"),
    );
    let mount = Mount::new(Arc::new(strategy.clone()));
    (strategy, mount)
}

/// `test_attach_drive_success` and `test_detach_drive_success`
#[tokio::test]
async fn a_drive_is_attached_and_detached_through_the_drives_api() {
    let server = start(Shared::new(World::default())).await;
    let session = session_with(&server, options(&server.base_url), root_manifest()).await;
    let (strategy, mount) = drive_strategy();

    strategy
        .activate(&mount, &*session, "/mnt/data")
        .await
        .expect("attach");
    strategy
        .deactivate(&mount, &*session, "/mnt/data")
        .await
        .expect("detach");

    assert_eq!(
        server.find("POST", "/drives/mount").json(),
        serde_json::json!({
            "driveName": "test-drive",
            "mountPath": "/mnt/data",
            "drivePath": "/",
            "readOnly": false,
        })
    );
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(
                |request| request.method == "DELETE" && request.path.starts_with("/drives/mount/")
            )
            .count(),
        1
    );
    server.stop();
}

/// `test_attach_drive_error`
#[tokio::test]
async fn a_failing_drive_attach_is_a_mount_error() {
    let world = Shared::new(World {
        drive_mount_status: Some(500),
        ..World::default()
    });
    let server = start(world).await;
    let session = session_with(&server, options(&server.base_url), root_manifest()).await;
    let (strategy, mount) = drive_strategy();

    let error = strategy
        .activate(&mount, &*session, "/mnt/data")
        .await
        .expect_err("attach fails");

    assert_eq!(error.code, ErrorCode::MountFailed);
    assert!(error.message.contains("drive mount"), "{}", error.message);
    assert!(error.message.contains("failed"), "{}", error.message);
    server.stop();
}

/// `test_detach_drive_error_logged_not_raised`: a drive that will not detach does not fail
/// teardown.
#[tokio::test]
async fn a_failing_drive_detach_is_not_raised() {
    let world = Shared::new(World {
        drive_unmount_status: Some(500),
        ..World::default()
    });
    let server = start(world).await;
    let session = session_with(&server, options(&server.base_url), root_manifest()).await;
    let (strategy, mount) = drive_strategy();

    strategy
        .deactivate(&mount, &*session, "/mnt/SECRET_DRIVE_PATH")
        .await
        .expect("detach errors are logged");

    assert_eq!(
        server
            .requests()
            .iter()
            .filter(
                |request| request.method == "DELETE" && request.path.starts_with("/drives/mount/")
            )
            .count(),
        1
    );
    server.stop();
}

/// The drive strategy only works against a Blaxel session: another backend is refused with a
/// config error rather than a request to nowhere.
#[tokio::test]
async fn a_drive_needs_a_blaxel_session() {
    let session = FakeMountSession::new(vec![]);
    let (strategy, mount) = drive_strategy();

    // `FakeMountSession` reports the `blaxel` backend but exposes no sandbox connection.
    let error = strategy
        .activate(&mount, &session, "/mnt/data")
        .await
        .expect_err("no connection");

    assert_eq!(error.code, ErrorCode::MountFailed);
}

/// `test_drive_strategy_validate_wrong_mount_type` and `test_drive_strategy_validate_non_drive_mount`:
/// a drive strategy built from a mount entry that is not a drive is refused.
#[test]
fn a_drive_strategy_refuses_a_foreign_mount_entry() {
    let foreign = S3Mount::new("bucket");

    let error = BlaxelDriveMountStrategy::for_mount(&foreign).expect_err("not a drive");

    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert!(error.message.contains("BlaxelDriveMount") || error.message.contains("blaxel_drive"));
}

// ---------------------------------------------------------------------------------------------
// Unmount logging follows the tool-data policy
// ---------------------------------------------------------------------------------------------

struct Capture {
    records: Mutex<Vec<(log::Level, String)>>,
}

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.records
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture {
    records: Mutex::new(Vec::new()),
};

/// `test_unmount_all_attempts_follow_tool_data_policy` (all four rows) and
/// `test_detach_drive_error_logged_not_raised` (both rows): three failing unmount attempts log two
/// debug lines and one warning, and a failing drive detach logs one warning; the mount path
/// appears in them only when tool data may be logged, whatever the model-data policy says.
#[tokio::test]
async fn mount_logs_follow_the_tool_data_policy() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Debug);
    let mount_path = "/mnt/SECRET_BUCKET_PATH";

    for (model_redacted, tool_redacted) in
        [(false, true), (true, true), (false, false), (true, false)]
    {
        openai_agents_sdk::debug::set_dont_log_model_data(model_redacted);
        openai_agents_sdk::debug::set_dont_log_tool_data(tool_redacted);
        CAPTURE.records.lock().unwrap().clear();
        let session = FakeMountSession::new(vec![failed(), failed(), failed()]);

        deactivate(&session, mount_path)
            .await
            .expect("never raises");

        assert_eq!(session.commands().len(), 3);
        let records: Vec<(log::Level, String)> = CAPTURE
            .records
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, message)| {
                message.contains("(exit") || message.contains("unmount attempts failed")
            })
            .cloned()
            .collect();
        assert_eq!(
            records.iter().map(|(level, _)| *level).collect::<Vec<_>>(),
            vec![log::Level::Debug, log::Level::Debug, log::Level::Warn],
            "{model_redacted} {tool_redacted}: {records:?}"
        );
        for (_, message) in &records {
            assert_eq!(
                message.contains(mount_path),
                !tool_redacted,
                "{model_redacted} {tool_redacted}: {message}"
            );
        }
    }

    for redacted in [true, false] {
        openai_agents_sdk::debug::set_dont_log_model_data(false);
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        CAPTURE.records.lock().unwrap().clear();
        let world = Shared::new(World {
            drive_unmount_status: Some(500),
            ..World::default()
        });
        let server = start(world).await;
        let session = session_with(&server, options(&server.base_url), root_manifest()).await;
        let (strategy, mount) = drive_strategy();

        strategy
            .deactivate(&mount, &*session, "/mnt/SECRET_DRIVE_PATH")
            .await
            .expect("detach errors are logged");

        let records = CAPTURE.records.lock().unwrap().clone();
        let warning = records
            .iter()
            .find(|(level, message)| {
                *level == log::Level::Warn && message.contains("Drive detach failed")
            })
            .unwrap_or_else(|| panic!("{redacted}: {records:?}"));
        if redacted {
            assert_eq!(warning.1, "Drive detach failed (non-fatal)");
        } else {
            assert!(warning.1.contains("SECRET_DRIVE_PATH"), "{}", warning.1);
        }
        server.stop();
    }
    openai_agents_sdk::debug::set_dont_log_model_data(true);
    openai_agents_sdk::debug::set_dont_log_tool_data(true);
}
