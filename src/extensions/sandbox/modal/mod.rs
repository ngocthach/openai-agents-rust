//! Modal sandbox provider (port of `agents.extensions.sandbox.modal`).
//!
//! Modal publishes no REST API for sandboxes, so this port drives the official `modal` Python SDK
//! through a bundled helper script; see [`sandbox`] for why and for the request protocol. Run
//! `python -m modal setup`, or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, before using it.
//!
//! ```no_run
//! use openai_agents_sdk::extensions::sandbox::modal::{
//!     ModalSandboxClient, ModalSandboxClientOptions,
//! };
//! use openai_agents_sdk::sandbox::{ExecRequest, Manifest, SandboxClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ModalSandboxClient::new(
//!     ModalSandboxClientOptions::new("my-agents-app").with_image_tag("python:3.14-slim"),
//! );
//! let session = client.create(Some(Manifest::new())).await?;
//! session.start().await?;
//! println!("{}", session.exec(ExecRequest::shell("ls")).await?.stdout_text());
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod mounts;
pub mod sandbox;

pub use mounts::{ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy};
pub use sandbox::{
    DEFAULT_IMAGE_BUILDER_VERSION, DEFAULT_SNAPSHOT_FILESYSTEM_TIMEOUT, DEFAULT_TIMEOUT,
    HELPER_COMMAND_ENV, HELPER_PYTHON_ENV, HELPER_SOURCE, ModalHelper, ModalImageSelector,
    ModalSandboxClient, ModalSandboxClientOptions, ModalSandboxSelector, ModalSandboxSession,
    ModalWorkspacePersistence, decode_snapshot_directory_ref, decode_snapshot_filesystem_ref,
    encode_snapshot_directory_ref, encode_snapshot_filesystem_ref,
};
