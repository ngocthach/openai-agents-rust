//! Sandbox agents (feature `sandbox`). Mirrors `agents.sandbox`.
//!
//! A sandbox agent is an ordinary [`Agent`](crate::Agent) plus a workspace: a [`Manifest`]
//! describing the files it starts from, [`Capability`] values that expose sandbox-native tools,
//! and a [`SandboxClient`] that says where the work runs.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openai_agents_sdk::sandbox::entries::Entry;
//! use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
//! use openai_agents_sdk::sandbox::{Manifest, SandboxAgent, SandboxRunConfig, SandboxRunner};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = SandboxAgent::<()>::new("Sandbox engineer")
//!     .with_model("gpt-5.5")
//!     .with_instructions("Read `repo/task.md` before editing files.")
//!     .with_default_manifest(
//!         Manifest::new().with_entry("repo", Entry::local_dir("./repo")),
//!     );
//!
//! let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
//! let result = SandboxRunner::run(&agent, "Fix the issue in repo/task.md", &config).await?;
//! println!("{}", result.final_output_text().unwrap_or_default());
//! # Ok(())
//! # }
//! ```
//!
//! # What is ported
//!
//! - [`SandboxAgent`], manifest materialization, and the filesystem, shell and compaction
//!   capabilities.
//! - The [`unix_local`](sandboxes::unix_local) and [`docker`](sandboxes::docker) backends.
//! - [`SandboxRunner`], which binds an agent to a session and manages the session lifecycle
//!   around [`Runner`](crate::Runner).
//!
//! # What is not ported
//!
//! - The cloud provider backends in `agents.extensions.sandbox` (Blaxel, Cloudflare, Daytona,
//!   E2B, Modal, Vercel). Implement [`SandboxClient`] and [`SandboxSession`] to add one.
//! - Remote mounts (S3, GCS, Azure Blob, R2, Box) and their credential-exposure policy.
//! - PTY sessions (`write_stdin`), sandbox-backed memory, and the skills capability.
//! - Snapshot specs and session-state serialization: resume by passing a live
//!   [`SandboxRunConfig::session`], or by moving a
//!   [`persist_workspace`](SandboxSession::persist_workspace) archive into a new session.

pub mod agent;
pub mod capabilities;
pub mod editor;
pub mod entries;
pub mod errors;
pub mod manifest;
pub mod runtime;
pub mod sandboxes;
pub mod session;
pub mod types;
pub mod util;
pub mod workspace_paths;

pub use agent::{DEFAULT_SANDBOX_INSTRUCTIONS, SandboxAgent};
pub use capabilities::{
    Capabilities, Capability, CapabilityBinding, Compaction, Filesystem, Shell,
};
pub use editor::SandboxEditor;
pub use entries::{Entry, EntryKind};
pub use errors::{ErrorCode, SandboxError, SandboxResult};
pub use manifest::{DEFAULT_MANIFEST_ROOT, Manifest};
pub use runtime::{PreparedSandboxRun, SandboxRunConfig, SandboxRunner, SandboxSessionGuard};
pub use session::{ExecRequest, ExecShell, SandboxClient, SandboxSession};
pub use types::{ExecResult, FileEntry, Group, Permissions, User, file_mode};
pub use workspace_paths::SandboxWorkspaceScope;
