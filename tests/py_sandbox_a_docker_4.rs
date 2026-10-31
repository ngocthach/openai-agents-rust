//! Ports of the per-provider Docker volume driver tests in `tests/sandbox/test_docker.py`
//! (`test_docker_create_container_mounts_*`).
//!
//! Python asserts the `VolumeOptions.DriverConfig` dictionary handed to `docker-py`; the CLI
//! spelling of the same request is one `--mount type=volume,...,volume-opt=key=value` argument,
//! so each test asserts the driver name, the options and the read-only flag inside it.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{
    AzureBlobMount, BoxMount, ConfiguredMount, ConfiguredMountBackend, ConfiguredMountStrategy,
    GcsMount, S3FilesMount, S3Mount,
};
use openai_agents_sdk::sandbox::mounts::{Mount, docker_volume_name};
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxClient};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxResult};

#[derive(Debug, Default)]
struct Recorder {
    calls: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl DockerRunner for Recorder {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(argv.to_vec());
        let stdout = if argv.first().map(String::as_str) == Some("create") {
            "container-1\n"
        } else {
            ""
        };
        Ok(ExecResult {
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        })
    }
}

/// The `--mount` specification `docker create` received, and the session that was created.
async fn create_with(entry: Entry) -> (String, String) {
    let recorder = Arc::new(Recorder::default());
    let client = DockerSandboxClient::new().with_runner(recorder.clone());
    let session = client
        .create(Some(Manifest::new().with_entry("data", entry)))
        .await
        .expect("create");
    let create = recorder.calls.lock().expect("calls")[0].clone();
    let at = create.iter().position(|a| a == "--mount").expect("--mount");
    let session_id = session
        .session_state()
        .provider_metadata("docker_session_id")
        .and_then(serde_json::Value::as_str)
        .expect("session id")
        .to_string();
    (create[at + 1].clone(), session_id)
}

fn volume(spec: impl ConfiguredMount + 'static, driver: &str, read_only: bool) -> Entry {
    Entry::mount(
        Mount::new(Arc::new(ConfiguredMountStrategy::docker_volume(
            spec, driver,
        )))
        .with_read_only(read_only),
    )
}

fn assert_options(spec: &str, driver: &str, options: &[&str], read_only: bool) {
    assert!(spec.starts_with("type=volume,"), "{spec}");
    assert!(spec.contains("dst=/workspace/data"), "{spec}");
    assert!(spec.contains(&format!("volume-driver={driver}")), "{spec}");
    for option in options {
        assert!(
            spec.contains(&format!("volume-opt={option}")),
            "missing `{option}` in {spec}"
        );
    }
    assert_eq!(spec.contains(",readonly"), read_only, "{spec}");
}

/// Python: `test_docker_create_container_mounts_s3_with_rclone_driver`.
#[tokio::test]
async fn s3_mounts_with_the_rclone_volume_driver() {
    let (spec, session_id) = create_with(volume(
        S3Mount::new("bucket").with_credentials("key-id", "secret"),
        "rclone",
        true,
    ))
    .await;

    assert_options(
        &spec,
        "rclone",
        &[
            "type=s3",
            "s3-provider=AWS",
            "path=bucket",
            "s3-access-key-id=key-id",
            "s3-secret-access-key=secret",
        ],
        true,
    );
    assert!(
        spec.contains(&format!(
            "src={}",
            docker_volume_name(Some(&session_id), "/workspace/data")
        )),
        "{spec}"
    );
}

/// Python: `test_docker_create_container_mounts_s3_with_volume_driver_ignoring_mount_pattern`.
/// The volume driver's own name is used, and the extra driver option is merged over the derived
/// ones; the read-only flag follows the mount.
#[tokio::test]
async fn s3_volume_driver_options_are_merged_and_the_pattern_is_ignored() {
    let mut s3 = S3Mount::new("bucket").with_credentials("key-id", "secret");
    s3 = s3
        .with_prefix("logs/")
        .with_region("us-west-2")
        .with_endpoint_url("https://s3.example.test");
    let mut options = std::collections::BTreeMap::new();
    options.insert("allow_other".to_string(), "true".to_string());
    let entry = Entry::mount(
        Mount::new(Arc::new(ConfiguredMountStrategy::new(
            Arc::new(s3),
            ConfiguredMountBackend::DockerVolume {
                driver: "mountpoint".to_string(),
                driver_options: options,
            },
        )))
        .with_read_only(false),
    );

    let (spec, _) = create_with(entry).await;

    assert!(spec.contains("volume-driver=mountpoint"), "{spec}");
    assert!(spec.contains("volume-opt=allow_other=true"), "{spec}");
    assert!(!spec.contains(",readonly"), "{spec}");
    // The derived options still carry the bucket and its credentials.
    assert!(spec.contains("bucket"), "{spec}");
    assert!(spec.contains("secret"), "{spec}");
}

/// Python: `test_docker_create_container_mounts_gcs_with_rclone_driver`.
#[tokio::test]
async fn gcs_mounts_with_the_rclone_volume_driver() {
    let mut gcs = GcsMount::new("bucket");
    gcs.service_account_file = Some("/data/config/gcs.json".to_string());

    let (spec, _) = create_with(volume(gcs, "rclone", true)).await;

    assert_options(
        &spec,
        "rclone",
        &[
            "type=google cloud storage",
            "path=bucket",
            "gcs-service-account-file=/data/config/gcs.json",
        ],
        true,
    );
}

/// Python: `test_docker_create_container_mounts_gcs_hmac_with_rclone_s3_compat`.
#[tokio::test]
async fn gcs_hmac_mounts_through_rclone_s3_compatibility() {
    let mut gcs = GcsMount::new("bucket");
    gcs.access_id = Some("access-id".to_string());
    gcs.secret_access_key = Some("secret-key".to_string());
    gcs.prefix = Some("prefix/".to_string());
    gcs.region = Some("auto".to_string());

    let (spec, _) = create_with(volume(gcs, "rclone", false)).await;

    assert_options(
        &spec,
        "rclone",
        &[
            "type=s3",
            "path=bucket/prefix/",
            "s3-provider=GCS",
            "s3-access-key-id=access-id",
            "s3-secret-access-key=secret-key",
            "s3-endpoint=https://storage.googleapis.com",
            "s3-region=auto",
        ],
        false,
    );
}

/// Python: `test_docker_create_container_mounts_azure_with_rclone_driver`.
#[tokio::test]
async fn azure_blob_mounts_with_the_rclone_volume_driver() {
    let mut azure = AzureBlobMount::new("acct", "container");
    azure.endpoint = Some("https://blob.example.test".to_string());
    azure.identity_client_id = Some("client-id".to_string());
    azure.account_key = Some("account-key".to_string());

    let (spec, _) = create_with(volume(azure, "rclone", true)).await;

    assert_options(
        &spec,
        "rclone",
        &[
            "type=azureblob",
            "path=container",
            "azureblob-account=acct",
            "azureblob-endpoint=https://blob.example.test",
            "azureblob-msi-client-id=client-id",
            "azureblob-key=account-key",
        ],
        true,
    );
}

/// Python: `test_docker_create_container_mounts_box_with_rclone_driver`.
#[tokio::test]
async fn box_mounts_with_the_rclone_volume_driver() {
    let mut boxed = BoxMount::new();
    boxed.path = Some("/Shared/Finance".to_string());
    boxed.client_id = Some("client-id".to_string());
    boxed.client_secret = Some("client-secret".to_string());
    boxed.access_token = Some("access-token".to_string());
    boxed.root_folder_id = Some("12345".to_string());
    boxed.impersonate = Some("user-42".to_string());

    let (spec, _) = create_with(volume(boxed, "rclone", false)).await;

    assert_options(
        &spec,
        "rclone",
        &[
            "type=box",
            "path=Shared/Finance",
            "box-client-id=client-id",
            "box-client-secret=client-secret",
            "box-access-token=access-token",
            "box-root-folder-id=12345",
            "box-impersonate=user-42",
        ],
        false,
    );
}

/// The `docker create` arguments for a manifest holding `entry` at `data`.
async fn create_args(entry: Entry) -> Vec<String> {
    let recorder = Arc::new(Recorder::default());
    let client = DockerSandboxClient::new().with_runner(recorder.clone());
    client
        .create(Some(
            Manifest::new().with_entry("nested", Entry::dir().with_child("data", entry)),
        ))
        .await
        .expect("create");
    let calls = recorder.calls.lock().expect("calls");
    calls[0].clone()
}

/// Python: `test_manifest_requires_fuse_detects_nested_mounts` and
/// `test_docker_create_container_grants_fuse_for_in_container_rclone_mount`. A FUSE mount nested
/// inside a directory entry still earns the device, the capability and the AppArmor opt-out.
#[tokio::test]
async fn a_nested_fuse_mount_grants_the_fuse_device() {
    let mut azure = AzureBlobMount::new("account", "container");
    azure.account_key = Some("key".to_string());
    let argv = create_args(Entry::mount(Mount::new(Arc::new(
        ConfiguredMountStrategy::blobfuse(azure),
    ))))
    .await;

    assert!(
        argv.windows(2).any(|w| w == ["--device", "/dev/fuse"]),
        "{argv:?}"
    );
    assert!(
        argv.windows(2).any(|w| w == ["--cap-add", "SYS_ADMIN"]),
        "{argv:?}"
    );
    assert!(
        argv.windows(2)
            .any(|w| w == ["--security-opt", "apparmor:unconfined"]),
        "{argv:?}"
    );
}

/// Python: `test_docker_create_container_grants_sys_admin_for_s3_files_mount`. An NFS-style mount
/// needs `CAP_SYS_ADMIN` but not the FUSE device.
#[tokio::test]
async fn an_s3_files_mount_grants_sys_admin_without_the_fuse_device() {
    let argv = create_args(Entry::mount(Mount::new(Arc::new(
        ConfiguredMountStrategy::s3files(S3FilesMount {
            file_system_id: "fs-1234567890abcdef0".to_string(),
            ..Default::default()
        }),
    ))))
    .await;

    assert!(
        argv.windows(2).any(|w| w == ["--cap-add", "SYS_ADMIN"]),
        "{argv:?}"
    );
    assert!(
        argv.windows(2)
            .any(|w| w == ["--security-opt", "apparmor:unconfined"]),
        "{argv:?}"
    );
    assert!(!argv.iter().any(|a| a == "--device"), "{argv:?}");
}
