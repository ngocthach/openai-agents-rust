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
//! - The remote storage mounts [`S3Mount`], [`S3FilesMount`], [`GcsMount`], [`AzureBlobMount`],
//!   [`R2Mount`] and [`BoxMount`], attached by [`ConfiguredMountStrategy`] through the
//!   [`RcloneMountPattern`], [`MountpointMountPattern`], [`FuseMountPattern`] and
//!   [`S3FilesMountPattern`] mount tools.
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
//! - Exposed ports and the runtime helper scripts.

pub mod agent;
pub mod archive;
pub mod capabilities;
pub mod dependencies;
pub mod editor;
pub mod entries;
pub mod environment;
pub mod errors;
pub mod events;
pub mod http;
pub mod manifest;
pub mod materialization;
pub mod memory;
pub mod mount_entries;
pub mod mount_patterns;
pub mod mounts;
pub mod persisted_registry;
pub mod pty;
pub mod remote_mount_policy;
mod resume_payload;
pub mod retry;
pub mod runtime;
pub mod runtime_helpers;
pub mod sandboxes;
pub mod session;
pub mod session_lifecycle;
pub mod snapshot;
pub mod state;
pub mod testing;
pub mod types;
pub mod util;
pub mod workspace_paths;

pub use agent::{DEFAULT_SANDBOX_INSTRUCTIONS, PreparedSandboxGraph, SandboxAgent};
pub use archive::{
    ArchiveMemberKind, CompressionScheme, TarValidationOptions, extract_tar_to,
    rebase_archive_symlink_targets, safe_tar_member_rel_path, safe_zip_member_rel_path,
    should_skip_tar_member, validate_hydrate_archive, validate_tar_bytes, validate_zip_bytes,
};
pub use capabilities::{
    Capabilities, Capability, CapabilityBinding, Compaction, ConfiguredCapability, Filesystem,
    LazySkillSource, LocalDirLazySkillSource, Memory, Shell, Skill, SkillMetadata, Skills,
};
pub use dependencies::{Dependencies, Dependency, DependencyCloser, DependencyFactory};
pub use editor::{PatchFormat, SandboxEditor, V4aFormat};
pub use entries::{Entry, EntryKind};
pub use environment::{EnvEntry, EnvValue, Environment, StrEnvValue};
pub use errors::{ErrorCode, SandboxError, SandboxOp, SandboxResult};
pub use events::{
    CallbackSink, ChainedSink, EventPayloadPolicy, EventPhase, EventSink, HttpProxySink,
    Instrumentation, InstrumentedSession, JsonlFileSink, SandboxSessionEvent, WorkspaceJsonlSink,
    current_trace_id,
};
pub use manifest::{
    DEFAULT_MANIFEST_ROOT, DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY,
    DEFAULT_MAX_MANIFEST_ENTRY_CONCURRENCY, DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST, Manifest,
    SandboxConcurrencyLimits,
};
pub use materialization::{MaterializationResult, MaterializedFile, sha256_file, sha256_hex};
pub use memory::{MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig};
pub use mount_entries::{
    AzureBlobMount, BoxMount, BoxSubType, BuiltinMountPattern, ConfiguredMount,
    ConfiguredMountBackend, ConfiguredMountStrategy, GcsMount, R2Mount, S3FilesMount, S3Mount,
};
pub use mount_patterns::{
    BlobfuseCacheType, FuseMountPattern, MountpointMountPattern, MountpointOptions,
    S3FilesMountPattern, S3FilesOptions,
};
pub use mounts::{
    BindMountStrategy, ContainerPrivileges, DockerVolumeMountStrategy, HostPathGrant,
    InContainerMountStrategy, Mount, MountCredentialAuthority, MountPattern, MountPatternConfig,
    MountStrategy, MountTransition, RCLONE_CONFIG_DIR, RcloneMountMode, RcloneMountPattern,
};
pub use pty::{
    PtyExecRequest, PtyExecUpdate, PtyManager, PtyWriteRequest, RemotePtyManager, RemotePtyProcess,
};
pub use remote_mount_policy::build_remote_mount_policy_instructions;
pub use retry::{BackoffStrategy, RetryPolicy, is_transient_sandbox_error, retry_async};
pub use runtime::{
    PreparedSandboxRun, SandboxRunConfig, SandboxRunOutcome, SandboxRunner, SandboxSessionGuard,
    SandboxStreamedRun,
};
pub use runtime_helpers::{
    RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript, WORKSPACE_FINGERPRINT_HELPER,
};
pub use session::{
    ExecRequest, ExecShell, SandboxClient, SandboxSession, capture_workspace_tar,
    clear_workspace_root_on_resume, compute_and_cache_snapshot_fingerprint,
    live_workspace_matches_snapshot_on_resume, parse_snapshot_fingerprint_record,
    read_cached_snapshot_fingerprint, snapshot_fingerprint_cache_path,
};
pub use session_lifecycle::{
    PreStopHook, SandboxSessionLifecycle, aclose_session, pre_stop_hooks_failed,
    register_pre_stop_hook, run_pre_stop_hooks,
};
pub use snapshot::{
    LocalSnapshot, NoopSnapshot, RemoteSnapshot, RemoteSnapshotClient, Snapshot, SnapshotSpec,
    cleanup_stale_default_local_snapshots, default_local_snapshot_base_dir,
    register_remote_snapshot_client, remote_snapshot_client, resolve_default_local_snapshot_spec,
    resolve_snapshot,
};
pub use state::{SandboxSessionState, SnapshotFingerprint, manifest_digest};
pub use testing::{
    SandboxCall, SandboxCallResult, SandboxMethod, SandboxStep, ScriptedSandboxSession,
    scripted_sandbox_session,
};
pub use types::{
    DEFAULT_MAX_ARCHIVE_EXTRACTED_BYTES, DEFAULT_MAX_ARCHIVE_INPUT_BYTES,
    DEFAULT_MAX_ARCHIVE_MEMBERS, ExecResult, ExposedPortEndpoint, ExposedPortUnavailable,
    FileEntry, Group, Permissions, SandboxArchiveLimits, User, exposed_port_unavailable, file_mode,
};
pub use workspace_paths::{
    PROVIDER_SCRATCH_PREFIX, SandboxPathGrant, SandboxWorkspaceScope, WorkspacePathPolicy,
    is_provider_scratch_path, provider_scratch_path,
};
