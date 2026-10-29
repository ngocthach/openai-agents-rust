//! Cloud bucket mounts for Modal sandboxes. Mirrors
//! `agents.extensions.sandbox.modal.mounts`.
//!
//! Modal attaches buckets when the sandbox is created, as `modal.CloudBucketMount` values in the
//! `volumes` argument, so these strategies carry configuration rather than running commands:
//! [`activate`](MountStrategy::activate) and [`deactivate`](MountStrategy::deactivate) are no-ops
//! and [`ModalSandboxClient`](super::sandbox::ModalSandboxClient) reads the configuration out of
//! the manifest through [`ModalCloudBucketMountStrategy::volume_specs`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::mounts::{Mount, MountStrategy};
use crate::sandbox::session::SandboxSession;

/// The bucket a Modal sandbox mounts, and how Modal authenticates to it.
#[derive(Clone, PartialEq, Eq)]
pub struct ModalCloudBucketMountConfig {
    /// The bucket name.
    pub bucket_name: String,
    /// The endpoint URL, required for R2 and GCS.
    pub bucket_endpoint_url: Option<String>,
    /// A key prefix; Modal requires it to end in `/`.
    pub key_prefix: Option<String>,
    /// Inline credentials, injected as a Modal secret built from this map.
    pub credentials: Vec<(String, String)>,
    /// An existing Modal secret holding the credentials instead.
    pub secret_name: Option<String>,
    /// The Modal environment the secret lives in.
    pub secret_environment_name: Option<String>,
    /// Whether the bucket is mounted read-only.
    pub read_only: bool,
}

impl std::fmt::Debug for ModalCloudBucketMountConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModalCloudBucketMountConfig")
            .field("bucket_name", &self.bucket_name)
            .field("bucket_endpoint_url", &self.bucket_endpoint_url)
            .field("key_prefix", &self.key_prefix)
            .field(
                "credentials",
                &if self.credentials.is_empty() {
                    "<none>"
                } else {
                    "<redacted>"
                },
            )
            .field("secret_name", &self.secret_name)
            .field("secret_environment_name", &self.secret_environment_name)
            .field("read_only", &self.read_only)
            .finish()
    }
}

impl ModalCloudBucketMountConfig {
    fn base(bucket_name: impl Into<String>) -> Self {
        ModalCloudBucketMountConfig {
            bucket_name: bucket_name.into(),
            bucket_endpoint_url: None,
            key_prefix: None,
            credentials: Vec::new(),
            secret_name: None,
            secret_environment_name: None,
            read_only: true,
        }
    }

    /// An S3 bucket, authenticated with `AWS_*` credentials.
    pub fn s3(bucket_name: impl Into<String>) -> Self {
        ModalCloudBucketMountConfig::base(bucket_name)
    }

    /// An R2 bucket, reached at the account's R2 endpoint.
    pub fn r2(bucket_name: impl Into<String>, account_id: impl AsRef<str>) -> Self {
        let mut config = ModalCloudBucketMountConfig::base(bucket_name);
        config.bucket_endpoint_url = Some(format!(
            "https://{}.r2.cloudflarestorage.com",
            account_id.as_ref()
        ));
        config
    }

    /// A GCS bucket, reached over the S3-compatible endpoint.
    pub fn gcs(bucket_name: impl Into<String>) -> Self {
        let mut config = ModalCloudBucketMountConfig::base(bucket_name);
        config.bucket_endpoint_url = Some("https://storage.googleapis.com".to_string());
        config
    }

    /// Reach the bucket through an explicit endpoint.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.bucket_endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Expose only the keys under `prefix`.
    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = Some(prefix.into());
        self
    }

    /// Authenticate with S3 credentials.
    pub fn with_aws_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.credentials
            .push(("AWS_ACCESS_KEY_ID".to_string(), access_key_id.into()));
        self.credentials.push((
            "AWS_SECRET_ACCESS_KEY".to_string(),
            secret_access_key.into(),
        ));
        self
    }

    /// Add an STS session token to the S3 credentials.
    pub fn with_aws_session_token(mut self, session_token: impl Into<String>) -> Self {
        self.credentials
            .push(("AWS_SESSION_TOKEN".to_string(), session_token.into()));
        self
    }

    /// Authenticate with GCS HMAC credentials.
    pub fn with_google_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.credentials
            .push(("GOOGLE_ACCESS_KEY_ID".to_string(), access_key_id.into()));
        self.credentials.push((
            "GOOGLE_ACCESS_KEY_SECRET".to_string(),
            secret_access_key.into(),
        ));
        self
    }

    /// Authenticate with an existing Modal secret instead of inline credentials.
    pub fn with_secret_name(mut self, secret_name: impl Into<String>) -> Self {
        self.secret_name = Some(secret_name.into());
        self
    }

    /// Look the secret up in a specific Modal environment.
    pub fn with_secret_environment_name(mut self, environment_name: impl Into<String>) -> Self {
        self.secret_environment_name = Some(environment_name.into());
        self
    }

    /// Allow writes through the mount.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Validate the combination of inline credentials, secret name and environment.
    pub fn validate(&self) -> SandboxResult<()> {
        if self.bucket_name.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "modal cloud bucket mounts require a bucket name",
            ));
        }
        if let Some(secret_name) = &self.secret_name
            && secret_name.is_empty()
        {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "modal cloud bucket secret_name must be a non-empty string",
            ));
        }
        if let Some(environment_name) = &self.secret_environment_name {
            if environment_name.is_empty() {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "modal cloud bucket secret_environment_name must be a non-empty string",
                ));
            }
            if self.secret_name.is_none() {
                return Err(SandboxError::new(
                    ErrorCode::MountConfigInvalid,
                    "modal cloud bucket secret_environment_name requires secret_name to also be \
                     set",
                ));
            }
        }
        if self.secret_name.is_some() && !self.credentials.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "modal cloud bucket mounts do not support both inline credentials and \
                 secret_name",
            ));
        }
        if let Some(prefix) = &self.key_prefix
            && !prefix.ends_with('/')
        {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                "modal cloud bucket key_prefix must end in `/`",
            ));
        }
        Ok(())
    }

    /// The `volumes` entry the helper turns into a `modal.CloudBucketMount`.
    pub fn to_volume_spec(&self, mount_path: &str) -> Value {
        let mut spec = Map::new();
        spec.insert(
            "mount_path".to_string(),
            Value::String(mount_path.to_string()),
        );
        spec.insert(
            "bucket_name".to_string(),
            Value::String(self.bucket_name.clone()),
        );
        if let Some(endpoint_url) = &self.bucket_endpoint_url {
            spec.insert(
                "bucket_endpoint_url".to_string(),
                Value::String(endpoint_url.clone()),
            );
        }
        if let Some(prefix) = &self.key_prefix {
            spec.insert("key_prefix".to_string(), Value::String(prefix.clone()));
        }
        if !self.credentials.is_empty() {
            spec.insert(
                "credentials".to_string(),
                Value::Object(
                    self.credentials
                        .iter()
                        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                        .collect(),
                ),
            );
        }
        if let Some(secret_name) = &self.secret_name {
            spec.insert(
                "secret_name".to_string(),
                Value::String(secret_name.clone()),
            );
        }
        if let Some(environment_name) = &self.secret_environment_name {
            spec.insert(
                "secret_environment_name".to_string(),
                Value::String(environment_name.clone()),
            );
        }
        spec.insert("read_only".to_string(), Value::Bool(self.read_only));
        Value::Object(spec)
    }
}

/// Declares a bucket Modal attaches when the sandbox is created.
#[derive(Debug, Clone)]
pub struct ModalCloudBucketMountStrategy {
    config: ModalCloudBucketMountConfig,
}

impl ModalCloudBucketMountStrategy {
    /// A strategy for one bucket.
    pub fn new(config: ModalCloudBucketMountConfig) -> Self {
        ModalCloudBucketMountStrategy { config }
    }

    /// A [`Mount`] backed by this strategy.
    pub fn mount(config: ModalCloudBucketMountConfig) -> Mount {
        let read_only = config.read_only;
        Mount::new(Arc::new(ModalCloudBucketMountStrategy::new(config))).with_read_only(read_only)
    }

    /// The bucket this strategy attaches.
    pub fn config(&self) -> &ModalCloudBucketMountConfig {
        &self.config
    }

    /// The `volumes` entries for every Modal bucket mount in `manifest`.
    ///
    /// Modal needs these at sandbox creation, so the client collects them from the manifest
    /// instead of activating each mount after the session starts.
    pub fn volume_specs(manifest: &Manifest) -> Vec<Value> {
        manifest
            .mount_targets()
            .into_iter()
            .filter_map(|(mount, mount_path)| {
                strategy_config(&mount.strategy).map(|config| config.to_volume_spec(&mount_path))
            })
            .collect()
    }
}

/// The bucket configuration `strategy` carries, when it is a Modal bucket mount.
fn strategy_config(strategy: &Arc<dyn MountStrategy>) -> Option<ModalCloudBucketMountConfig> {
    strategy
        .as_any()?
        .downcast_ref::<ModalCloudBucketMountStrategy>()
        .map(|strategy| strategy.config.clone())
}

#[async_trait]
impl MountStrategy for ModalCloudBucketMountStrategy {
    fn strategy_type(&self) -> &str {
        "modal_cloud_bucket"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn validate(&self, _mount: &Mount) -> SandboxResult<()> {
        self.config.validate()
    }

    fn supports_native_snapshot_detach(&self) -> bool {
        // Modal holds the bucket outside the workspace, so a snapshot must not archive it.
        false
    }

    async fn activate(
        &self,
        mount: &Mount,
        session: &dyn SandboxSession,
        mount_path: &str,
    ) -> SandboxResult<()> {
        if session.backend_id() != "modal" {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "modal cloud bucket mounts are not supported by the `{}` sandbox backend",
                    session.backend_id()
                ),
            ));
        }
        self.config.validate()?;
        // Modal attached the bucket when it created the sandbox, so there is nothing to run
        // here. A mount the client could not read back was never attached at all, and failing
        // loudly beats handing the model an empty directory.
        if strategy_config(&mount.strategy).is_none() {
            return Err(SandboxError::new(
                ErrorCode::MountConfigInvalid,
                format!(
                    "the mount at `{mount_path}` does not expose its bucket configuration, so \
                     the client could not pass it to Modal at sandbox creation"
                ),
            ));
        }
        Ok(())
    }

    async fn deactivate(
        &self,
        _mount: &Mount,
        _session: &dyn SandboxSession,
        _mount_path: &str,
    ) -> SandboxResult<()> {
        // The bucket is released with the sandbox itself.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::entries::Entry;
    use serde_json::json;

    #[test]
    fn builds_a_volume_spec() {
        let config = ModalCloudBucketMountConfig::r2("assets", "acct")
            .with_aws_credentials("AKIA", "secret")
            .with_key_prefix("nested/")
            .with_read_only(false);
        assert_eq!(
            config.to_volume_spec("/workspace/assets"),
            json!({
                "mount_path": "/workspace/assets",
                "bucket_name": "assets",
                "bucket_endpoint_url": "https://acct.r2.cloudflarestorage.com",
                "key_prefix": "nested/",
                "credentials": {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "secret"},
                "read_only": false,
            })
        );
    }

    #[test]
    fn rejects_conflicting_credentials() {
        let config = ModalCloudBucketMountConfig::s3("bucket")
            .with_aws_credentials("id", "secret")
            .with_secret_name("aws-creds");
        assert_eq!(
            config.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
        let config = ModalCloudBucketMountConfig::s3("bucket").with_secret_environment_name("prod");
        assert_eq!(
            config.validate().unwrap_err().code,
            ErrorCode::MountConfigInvalid
        );
        assert!(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_secret_name("aws-creds")
                .with_secret_environment_name("prod")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn requires_a_trailing_slash_on_the_prefix() {
        assert_eq!(
            ModalCloudBucketMountConfig::s3("bucket")
                .with_key_prefix("nested")
                .validate()
                .unwrap_err()
                .code,
            ErrorCode::MountConfigInvalid
        );
    }

    #[test]
    fn collects_volume_specs_from_a_manifest() {
        let manifest = Manifest::new()
            .with_entry(
                "assets",
                Entry::mount(ModalCloudBucketMountStrategy::mount(
                    ModalCloudBucketMountConfig::s3("bucket").with_secret_name("aws-creds"),
                )),
            )
            .with_entry("task.md", Entry::text_file("x"));
        let specs = ModalCloudBucketMountStrategy::volume_specs(&manifest);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0]["mount_path"], "/workspace/assets");
        assert_eq!(specs[0]["secret_name"], "aws-creds");
    }

    #[test]
    fn keeps_credentials_out_of_debug_output() {
        let config =
            ModalCloudBucketMountConfig::s3("bucket").with_aws_credentials("id", "topsecret");
        let rendered = format!("{config:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("topsecret"));
    }
}
