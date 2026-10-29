//! Cloudflare sandbox provider (port of `agents.extensions.sandbox.cloudflare`).
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openai_agents_sdk::extensions::sandbox::cloudflare::{
//!     CloudflareSandboxClient, CloudflareSandboxClientOptions,
//! };
//! use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
//! use openai_agents_sdk::sandbox::entries::Entry;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = CloudflareSandboxClient::new(
//!     CloudflareSandboxClientOptions::new("https://sandbox.example.workers.dev")
//!         .with_api_key("cf-token"),
//! );
//! let manifest = Manifest::new().with_entry("task.md", Entry::text_file("fix it"));
//! let session = client.create(Some(manifest)).await?;
//! session.start().await?;
//! let result = session.exec(openai_agents_sdk::sandbox::ExecRequest::shell("ls")).await?;
//! println!("{}", result.stdout_text());
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod mounts;
pub mod sandbox;
pub mod sse;

pub use mounts::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy, CloudflareBucketProvider,
};
pub use sandbox::{
    API_KEY_ENV, CloudflareConnection, CloudflareSandboxClient, CloudflareSandboxClientOptions,
    CloudflareSandboxSession, DEFAULT_EXEC_TIMEOUT, DEFAULT_REQUEST_TIMEOUT,
    connection_for_session, quote_path,
};
pub use sse::{ServerSentEvent, SseDecoder, SseLineDecoder};
