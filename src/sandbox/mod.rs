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
//! - [`SandboxAgent`], manifest materialization, and the filesystem, shell, compaction, memory
//!   and skills capabilities.
//! - [`Mount`]s with a [`MountStrategy`] trait cloud providers can implement, plus the built-in
//!   [`BindMountStrategy`], [`DockerVolumeMountStrategy`] and [`InContainerMountStrategy`], the
//!   credential-exposure policy, and the remote mount instruction block.
//! - PTY sessions (`exec_command` / `write_stdin`), backed by `portable-pty` on both backends.
//! - Workspace [`Snapshot`]s and [`SandboxSessionState`], so a run that pauses on approvals can
//!   be resumed against the same workspace.
//! - The [`unix_local`](sandboxes::unix_local) and [`docker`](sandboxes::docker) backends.
//! - [`SandboxRunner`], which binds the whole sandbox handoff graph to one session, re-applies
//!   capability context processing on every turn, and manages the session lifecycle around
//!   [`Runner`](crate::Runner).
//!
//! # What is not ported
//!
//! - The cloud provider backends in `agents.extensions.sandbox` (Blaxel, Cloudflare, Daytona,
//!   E2B, Modal, Vercel). Implement [`SandboxClient`] and [`SandboxSession`] to add one, and
//!   [`MountPattern`] for their storage mounts.
//! - The concrete remote storage mounts (S3, S3 Files, GCS, Azure Blob, R2, Box) and their FUSE,
//!   `mount-s3` and `rclone` patterns, which belong with those providers.
//! - Exposed ports, archive extraction limits and the runtime helper scripts.

pub mod agent;
pub mod capabilities;
pub mod editor;
pub mod entries;
pub mod errors;
pub mod manifest;
pub mod memory;
pub mod mounts;
pub mod pty;
pub mod remote_mount_policy;
pub mod runtime;
pub mod sandboxes;
pub mod session;
pub mod snapshot;
pub mod state;
pub mod types;
pub mod util;
pub mod workspace_paths;

pub use agent::{DEFAULT_SANDBOX_INSTRUCTIONS, PreparedSandboxGraph, SandboxAgent};
pub use capabilities::{
    Capabilities, Capability, CapabilityBinding, Compaction, Filesystem, LazySkillSource,
    LocalDirLazySkillSource, Memory, Shell, Skill, SkillMetadata, Skills,
};
pub use editor::SandboxEditor;
pub use entries::{Entry, EntryKind};
pub use errors::{ErrorCode, SandboxError, SandboxResult};
pub use manifest::{DEFAULT_MANIFEST_ROOT, DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST, Manifest};
pub use memory::{MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig};
pub use mounts::{
    BindMountStrategy, DockerVolumeMountStrategy, HostPathGrant, InContainerMountStrategy, Mount,
    MountCredentialAuthority, MountPattern, MountPatternConfig, MountStrategy,
};
pub use pty::{
    PtyExecRequest, PtyExecUpdate, PtyManager, PtyWriteRequest, RemotePtyManager, RemotePtyProcess,
};
pub use remote_mount_policy::build_remote_mount_policy_instructions;
pub use runtime::{
    PreparedSandboxRun, SandboxRunConfig, SandboxRunOutcome, SandboxRunner, SandboxSessionGuard,
    SandboxStreamedRun,
};
pub use session::{ExecRequest, ExecShell, SandboxClient, SandboxSession};
pub use snapshot::{
    LocalSnapshot, NoopSnapshot, Snapshot, SnapshotSpec, cleanup_stale_default_local_snapshots,
    default_local_snapshot_base_dir, resolve_default_local_snapshot_spec, resolve_snapshot,
};
pub use state::{SandboxSessionState, manifest_digest};
pub use types::{ExecResult, FileEntry, Group, Permissions, User, file_mode};
pub use workspace_paths::{
    PROVIDER_SCRATCH_PREFIX, SandboxWorkspaceScope, is_provider_scratch_path, provider_scratch_path,
};
