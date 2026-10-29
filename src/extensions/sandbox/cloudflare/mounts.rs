//! Bucket mounts for Cloudflare sandboxes. Mirrors
//! `agents.extensions.sandbox.cloudflare.mounts`.
//!
//! The worker attaches R2, S3 and GCS buckets itself, through `POST /mount` and `POST /unmount`,
//! so no credential is written inside the sandbox and the mount needs no in-container tool.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mounts::{Mount, MountStrategy};
use crate::sandbox::session::SandboxSession;

/// Which bucket API the worker should speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudflareBucketProvider {
    /// Cloudflare R2.
    R2,
    /// Amazon S3.
    S3,
    /// Google Cloud Storage, over its S3-compatible API.
    Gcs,
}

impl CloudflareBucketProvider {
    /// The wire name of the provider.
    pub fn as_str(self) -> &'static str {
        match self {
            CloudflareBucketProvider::R2 => "r2",
            CloudflareBucketProvider::S3 => "s3",
            CloudflareBucketProvider::Gcs => "gcs",
        }
    }
}

/// The bucket a Cloudflare sandbox mounts, and how the worker reaches it.
#[derive(Clone, PartialEq, Eq)]
pub struct CloudflareBucketMountConfig {
    /// The bucket name.
    pub bucket_name: String,
    /// The S3-compatible endpoint the worker calls.
    pub bucket_endpoint_url: String,
    /// Which bucket API to speak.
    pub provider: CloudflareBucketProvider,
    /// An optional key prefix, normalized to `/prefix/`.
    pub key_prefix: Option<String>,
    /// The access key pair, when the bucket is not public.
    pub credentials: Option<(String, String)>,
    /// Whether the mount is read-only.
    pub read_only: bool,
}

impl std::fmt::Debug for CloudflareBucketMountConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflareBucketMountConfig")
            .field("bucket_name", &self.bucket_name)
            .field("bucket_endpoint_url", &self.bucket_endpoint_url)
            .field("provider", &self.provider)
            .field("key_prefix", &self.key_prefix)
            .field(
                "credentials",
                &self.credentials.as_ref().map(|_| "<redacted>"),
            )
            .field("read_only", &self.read_only)
            .finish()
    }
}

impl CloudflareBucketMountConfig {
    /// An S3 bucket, reached at `https://s3[.<region>].amazonaws.com` unless an endpoint is set.
    pub fn s3(bucket: impl Into<String>) -> Self {
        CloudflareBucketMountConfig {
            bucket_name: bucket.into(),
            bucket_endpoint_url: "https://s3.amazonaws.com".to_string(),
            provider: CloudflareBucketProvider::S3,
            key_prefix: None,
            credentials: None,
            read_only: true,
        }
    }

    /// An R2 bucket, reached at `https://<account_id>.r2.cloudflarestorage.com`.
    pub fn r2(bucket: impl Into<String>, account_id: impl AsRef<str>) -> Self {
        CloudflareBucketMountConfig {
            bucket_name: bucket.into(),
            bucket_endpoint_url: format!(
                "https://{}.r2.cloudflarestorage.com",
                account_id.as_ref()
            ),
            provider: CloudflareBucketProvider::R2,
            key_prefix: None,
            credentials: None,
            read_only: true,
        }
    }

    /// A GCS bucket, reached over the S3-compatible `https://storage.googleapis.com` endpoint.
    pub fn gcs(bucket: impl Into<String>) -> Self {
        CloudflareBucketMountConfig {
            bucket_name: bucket.into(),
            bucket_endpoint_url: "https://storage.googleapis.com".to_string(),
            provider: CloudflareBucketProvider::Gcs,
            key_prefix: None,
            credentials: None,
            read_only: true,
        }
    }

    /// Reach the bucket through an explicit endpoint, such as an S3 region host.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.bucket_endpoint_url = endpoint_url.into();
        self
    }

    /// Reach an S3 bucket through its regional endpoint.
    pub fn with_region(mut self, region: impl AsRef<str>) -> Self {
        self.bucket_endpoint_url = format!("https://s3.{}.amazonaws.com", region.as_ref());
        self
    }

    /// Expose only the keys under `prefix`.
    pub fn with_key_prefix(mut self, prefix: impl AsRef<str>) -> Self {
        self.key_prefix = Some(normalize_prefix(prefix.as_ref()));
        self
    }

    /// Authenticate with an access key pair.
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.credentials = Some((access_key_id.into(), secret_access_key.into()));
        self
    }

    /// Allow writes through the mount.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// The `options` object sent to the worker.
    pub fn to_request_options(&self) -> Value {
        let mut options = Map::new();
        options.insert(
            "endpoint".to_string(),
            Value::String(self.bucket_endpoint_url.clone()),
        );
        options.insert("readOnly".to_string(), Value::Bool(self.read_only));
        if let Some(prefix) = &self.key_prefix {
            options.insert("prefix".to_string(), Value::String(prefix.clone()));
        }
        if let Some((access_key_id, secret_access_key)) = &self.credentials {
            options.insert(
                "credentials".to_string(),
                json!({
                    "accessKeyId": access_key_id,
                    "secretAccessKey": secret_access_key,
                }),
            );
        }
        Value::Object(options)
    }

    fn validate(&self) -> SandboxResult<()> {
        if self.bucket_name.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "cloudflare bucket mounts require a bucket name",
            ));
        }
        if self.provider == CloudflareBucketProvider::Gcs && self.credentials.is_none() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "gcs cloudflare bucket mounts require an access key id and secret access key",
            ));
        }
        Ok(())
    }
}

/// Normalize a key prefix to the `/prefix/` form the worker expects.
fn normalize_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_matches('/');
    if trimmed.is_empty() {
        return "/".to_string();
    }
    format!("/{trimmed}/")
}

/// Attaches a bucket to a Cloudflare sandbox through the worker's mount endpoints.
#[derive(Debug, Clone)]
pub struct CloudflareBucketMountStrategy {
    config: CloudflareBucketMountConfig,
}

impl CloudflareBucketMountStrategy {
    /// A strategy for one bucket.
    pub fn new(config: CloudflareBucketMountConfig) -> Self {
        CloudflareBucketMountStrategy { config }
    }

    /// A read-only [`Mount`] backed by this strategy.
    pub fn mount(config: CloudflareBucketMountConfig) -> Mount {
        let read_only = config.read_only;
        Mount::new(Arc::new(CloudflareBucketMountStrategy::new(config))).with_read_only(read_only)
    }

    /// The bucket this strategy attaches.
    pub fn config(&self) -> &CloudflareBucketMountConfig {
        &self.config
    }

    fn connection(
        &self,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<Arc<super::sandbox::CloudflareConnection>> {
        if session.backend_id() != "cloudflare" {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "cloudflare bucket mounts are not supported by the `{}` sandbox backend",
                    session.backend_id()
                ),
            ));
        }
        session
            .as_any()
            .and_then(|session| {
                session
                    .downcast_ref::<super::sandbox::CloudflareSandboxSession>()
                    .map(|session| session.connection().clone())
            })
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::MountFailed,
                    format!(
                        "the Cloudflare sandbox behind `{mount_path}` does not expose its worker \
                         connection"
                    ),
                )
            })
    }
}

#[async_trait]
impl MountStrategy for CloudflareBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "cloudflare_bucket_mount"
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        self.config.validate()
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        // The worker holds the mount outside the workspace archive, so a snapshot must detach it
        // first and re-attach it afterwards.
        false
    }

    async fn activate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.config.validate()?;
        let connection = self.connection(session, mount_path)?;
        connection
            .mount_bucket(
                &self.config.bucket_name,
                mount_path,
                self.config.to_request_options(),
            )
            .await
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        let connection = self.connection(session, mount_path)?;
        connection.unmount_bucket(mount_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_request_options() {
        let config = CloudflareBucketMountConfig::r2("assets", "acct")
            .with_credentials("AKIA", "secret")
            .with_read_only(false);
        assert_eq!(
            config.to_request_options(),
            json!({
                "endpoint": "https://acct.r2.cloudflarestorage.com",
                "readOnly": false,
                "credentials": {"accessKeyId": "AKIA", "secretAccessKey": "secret"},
            })
        );
    }

    #[test]
    fn normalizes_key_prefixes() {
        assert_eq!(normalize_prefix("/data/"), "/data/");
        assert_eq!(normalize_prefix("data"), "/data/");
        assert_eq!(normalize_prefix("/"), "/");
        let config = CloudflareBucketMountConfig::s3("bucket")
            .with_region("eu-west-1")
            .with_key_prefix("nested/dir");
        assert_eq!(
            config.bucket_endpoint_url,
            "https://s3.eu-west-1.amazonaws.com"
        );
        assert_eq!(config.key_prefix.as_deref(), Some("/nested/dir/"));
    }

    #[test]
    fn requires_credentials_for_gcs() {
        let error = CloudflareBucketMountConfig::gcs("bucket")
            .validate()
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::MountConfigInvalid);
        assert!(
            CloudflareBucketMountConfig::gcs("bucket")
                .with_credentials("id", "secret")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn keeps_credentials_out_of_debug_output() {
        let config = CloudflareBucketMountConfig::s3("bucket").with_credentials("AKIA", "secret");
        let rendered = format!("{config:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("secret"));
    }
}
