//! Sandbox-backed agent memory. Mirrors `agents.sandbox.memory`.
//!
//! Memory is a set of files in the sandbox workspace. During a session each finished run appends
//! a memory-filtered record of itself to a per-rollout JSONL file. When the session closes,
//! phase 1 extracts one raw memory per rollout and phase 2 consolidates them into `MEMORY.md`
//! and `memory_summary.md`, which later sessions read back through the
//! [`Memory`](crate::sandbox::capabilities::Memory) capability.

pub mod config;
pub mod manager;
pub mod prompts;
pub mod rollouts;
pub mod storage;

pub use config::{MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig};
pub use manager::{RolloutExtractionArtifacts, SandboxMemoryGenerationManager};
pub use rollouts::{RolloutTerminalMetadata, TerminalState, build_rollout_payload_from_result};
pub use storage::{PhaseTwoInputSelection, PhaseTwoSelectionItem, SandboxMemoryStorage};
