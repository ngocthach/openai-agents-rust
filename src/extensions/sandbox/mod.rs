//! Cloud sandbox providers (feature `sandbox`). Mirrors `agents.extensions.sandbox`: each module
//! implements [`crate::sandbox::session::SandboxClient`] for a hosted sandbox service.

pub mod blaxel;
pub mod cloudflare;
pub mod daytona;
pub mod e2b;
pub mod modal;
pub mod ports;
pub mod vercel;

/// A sandbox client's options as Python persists them: the `type` discriminator, then the
/// provider's fields at the same level, so one payload names the backend it configures.
///
/// Mirrors `BaseSandboxClientOptions.parse` for the built-in backends. A payload whose `type`
/// names no backend is rejected rather than guessed at.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SandboxClientOptionsRecord {
    /// The host-shell backend.
    UnixLocal(crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClientOptions),
    /// The Docker backend.
    Docker(crate::sandbox::sandboxes::docker::DockerSandboxClientOptions),
    /// The E2B backend.
    #[serde(rename = "e2b")]
    E2b(e2b::E2BSandboxClientOptions),
    /// The Modal backend.
    Modal(modal::ModalSandboxClientOptions),
    /// The Cloudflare backend.
    Cloudflare(cloudflare::CloudflareSandboxClientOptions),
    /// The Daytona backend.
    Daytona(daytona::DaytonaSandboxClientOptions),
    /// The Vercel backend.
    Vercel(vercel::VercelSandboxClientOptions),
    /// The Blaxel backend.
    Blaxel(blaxel::BlaxelSandboxClientOptions),
}

impl SandboxClientOptionsRecord {
    /// The `type` discriminator this record serializes with.
    pub fn type_name(&self) -> &'static str {
        match self {
            SandboxClientOptionsRecord::UnixLocal(_) => "unix_local",
            SandboxClientOptionsRecord::Docker(_) => "docker",
            SandboxClientOptionsRecord::E2b(_) => "e2b",
            SandboxClientOptionsRecord::Modal(_) => "modal",
            SandboxClientOptionsRecord::Cloudflare(_) => "cloudflare",
            SandboxClientOptionsRecord::Daytona(_) => "daytona",
            SandboxClientOptionsRecord::Vercel(_) => "vercel",
            SandboxClientOptionsRecord::Blaxel(_) => "blaxel",
        }
    }
}

/// A provider mount strategy as Python persists it: the `type` discriminator names the provider
/// strategy and its fields sit beside it.
///
/// Mirrors `MountStrategyBase.parse` for the provider strategies. [`ProviderMountStrategy::parse`]
/// is the registry entry point: a payload whose `type` names no provider strategy is rejected.
/// The error never repeats the payload, because a strategy can carry bucket credentials.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ProviderMountStrategy {
    /// An E2B cloud bucket, mounted through an rclone remote.
    #[serde(rename = "e2b_cloud_bucket")]
    E2bCloudBucket(e2b::E2BCloudBucketMountStrategy),
    /// A Modal cloud bucket.
    #[serde(rename = "modal_cloud_bucket")]
    ModalCloudBucket(modal::ModalCloudBucketMountStrategy),
    /// A Daytona cloud bucket, mounted through an rclone remote.
    #[serde(rename = "daytona_cloud_bucket")]
    DaytonaCloudBucket(daytona::DaytonaCloudBucketMountStrategy),
    /// A Cloudflare bucket, mounted through the worker's mount endpoints.
    #[serde(rename = "cloudflare_bucket_mount")]
    CloudflareBucketMount(cloudflare::CloudflareBucketMountStrategy),
    /// A Blaxel cloud bucket.
    #[serde(rename = "blaxel_cloud_bucket")]
    BlaxelCloudBucket(blaxel::BlaxelCloudBucketMountStrategy),
    /// A Blaxel Drive.
    #[serde(rename = "blaxel_drive")]
    BlaxelDrive(blaxel::BlaxelDriveMountStrategy),
    /// A Vercel cloud bucket.
    #[serde(rename = "vercel_cloud_bucket")]
    VercelCloudBucket(vercel::VercelCloudBucketMountStrategy),
}

// The strategy carries bucket credentials and rclone configuration, so `Debug` names only the
// strategy type.
impl std::fmt::Debug for ProviderMountStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderMountStrategy")
            .field("type", &self.strategy_type())
            .finish()
    }
}

impl ProviderMountStrategy {
    /// Parse a persisted strategy payload through the provider registry.
    pub fn parse(payload: &serde_json::Value) -> crate::sandbox::SandboxResult<Self> {
        serde_json::from_value(payload.clone()).map_err(|_| {
            crate::sandbox::SandboxError::new(
                crate::sandbox::ErrorCode::MountConfigInvalid,
                "mount strategy payload is not a known provider strategy",
            )
        })
    }

    /// The `type` discriminator this strategy serializes with.
    pub fn strategy_type(&self) -> &'static str {
        match self {
            ProviderMountStrategy::E2bCloudBucket(_) => "e2b_cloud_bucket",
            ProviderMountStrategy::ModalCloudBucket(_) => "modal_cloud_bucket",
            ProviderMountStrategy::DaytonaCloudBucket(_) => "daytona_cloud_bucket",
            ProviderMountStrategy::CloudflareBucketMount(_) => "cloudflare_bucket_mount",
            ProviderMountStrategy::BlaxelCloudBucket(_) => "blaxel_cloud_bucket",
            ProviderMountStrategy::BlaxelDrive(_) => "blaxel_drive",
            ProviderMountStrategy::VercelCloudBucket(_) => "vercel_cloud_bucket",
        }
    }

    /// The strategy as the mount machinery consumes it.
    pub fn into_mount_strategy(self) -> std::sync::Arc<dyn crate::sandbox::mounts::MountStrategy> {
        match self {
            ProviderMountStrategy::E2bCloudBucket(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::ModalCloudBucket(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::DaytonaCloudBucket(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::CloudflareBucketMount(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::BlaxelCloudBucket(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::BlaxelDrive(s) => std::sync::Arc::new(s),
            ProviderMountStrategy::VercelCloudBucket(s) => std::sync::Arc::new(s),
        }
    }
}
