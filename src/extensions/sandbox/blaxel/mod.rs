//! Blaxel sandbox provider (port of `agents.extensions.sandbox.blaxel`).
//!
//! ```no_run
//! use openai_agents_sdk::extensions::sandbox::blaxel::{
//!     BlaxelSandboxClient, BlaxelSandboxClientOptions, DEFAULT_BLAXEL_WORKSPACE_ROOT,
//! };
//! use openai_agents_sdk::sandbox::{ExecRequest, Manifest, SandboxClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // `BL_API_KEY` and `BL_WORKSPACE` authenticate the control plane.
//! let client = BlaxelSandboxClient::new(
//!     BlaxelSandboxClientOptions::new()
//!         .with_image("blaxel/base-image:latest")
//!         .with_region("us-pdx-1"),
//! );
//! let manifest = Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT);
//! let session = client.create(Some(manifest)).await?;
//! session.start().await?;
//! println!("{}", session.exec(ExecRequest::shell("ls")).await?.stdout_text());
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod mounts;
pub mod pty;
pub mod sandbox;

pub use mounts::{
    BlaxelBucketProvider, BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy,
    BlaxelDriveMount, BlaxelDriveMountConfig, BlaxelDriveMountStrategy, MOUNT_TYPE_BLAXEL_DRIVE,
};
pub use pty::{BlaxelPtySessions, terminal_url};
pub use sandbox::{
    API_KEY_ENV, BlaxelConnection, BlaxelSandboxClient, BlaxelSandboxClientOptions,
    BlaxelSandboxSession, BlaxelTimeouts, DEFAULT_API_BASE_URL, DEFAULT_BLAXEL_WORKSPACE_ROOT,
    DEFAULT_REGION, DEFAULT_RUN_BASE_URL, REGION_ENV, WORKSPACE_ENV, encode_sandbox_path,
};
