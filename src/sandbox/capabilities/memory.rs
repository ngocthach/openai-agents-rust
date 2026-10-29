//! The memory capability. Mirrors `agents.sandbox.capabilities.memory`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::sandbox::capabilities::{Capability, CapabilityBinding};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::memory::config::{MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig};
use crate::sandbox::memory::manager::SandboxMemoryGenerationManager;
use crate::sandbox::memory::prompts::render_memory_read_prompt;
use crate::sandbox::memory::storage::SandboxMemoryStorage;
use crate::sandbox::util::truncate_text_by_tokens;

/// Token budget of the memory summary injected into the agent instructions.
pub const MEMORY_SUMMARY_MAX_TOKENS: usize = 15_000;

/// Reads and generates sandbox memory artifacts for an agent.
///
/// [`Shell`](crate::sandbox::Shell) is required for memory reads, and
/// [`Filesystem`](crate::sandbox::Filesystem) as well when live updates are enabled.
///
/// Generation is a two-phase background process that runs when the sandbox session closes:
/// [`SandboxRunner`](crate::sandbox::SandboxRunner) creates one
/// [`SandboxMemoryGenerationManager`] **per prepared session**, records one rollout segment per
/// run, and flushes the manager when it tears the session down. A session the caller owns
/// ([`SandboxRunConfig::session`](crate::sandbox::SandboxRunConfig::session)) is never closed by
/// the runner, so the caller flushes it through
/// [`PreparedSandboxRun::memory_manager`](crate::sandbox::PreparedSandboxRun::memory_manager) or
/// [`SandboxRunOutcome::memory_manager`](crate::sandbox::SandboxRunOutcome::memory_manager).
///
/// This capability holds configuration only, so one `SandboxAgent` can be reused across many
/// sessions and concurrent runs without sharing rollout state.
#[derive(Debug, Clone)]
pub struct Memory {
    /// Filesystem layout used for rollout and memory files.
    pub layout: MemoryLayoutConfig,
    /// Read-side configuration. `None` disables memory reads.
    pub read: Option<MemoryReadConfig>,
    /// Generation configuration. `None` disables background memory generation.
    pub generate: Option<MemoryGenerateConfig>,
}

impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}

impl Memory {
    /// Memory with the default layout, reads and generation enabled.
    pub fn new() -> Self {
        Memory {
            layout: MemoryLayoutConfig::new(),
            read: Some(MemoryReadConfig::new()),
            generate: Some(MemoryGenerateConfig::new()),
        }
    }

    /// Set the filesystem layout.
    pub fn with_layout(mut self, layout: MemoryLayoutConfig) -> Self {
        self.layout = layout;
        self
    }

    /// Set (or disable, with `None`) memory reads.
    pub fn with_read(mut self, read: Option<MemoryReadConfig>) -> Self {
        self.read = read;
        self
    }

    /// Set (or disable, with `None`) background memory generation.
    pub fn with_generate(mut self, generate: Option<MemoryGenerateConfig>) -> Self {
        self.generate = generate;
        self
    }

    /// Validate the configuration.
    ///
    /// Returns an error when neither reads nor generation are enabled, or when the layout is not
    /// a workspace-relative path.
    pub fn validate(&self) -> Result<()> {
        if self.read.is_none() && self.generate.is_none() {
            return Err(crate::error::AgentsError::user(
                "Memory requires at least one of `read` or `generate`",
            ));
        }
        self.layout.validate()?;
        if let Some(generate) = &self.generate {
            generate.validate()?;
        }
        Ok(())
    }

    /// Build a generation manager for one sandbox session.
    ///
    /// Returns `None` when generation is disabled. Each session gets its own manager, so rollout
    /// state is never shared between sessions or concurrent runs.
    pub fn build_manager(&self) -> Result<Option<Arc<SandboxMemoryGenerationManager>>> {
        self.validate()?;
        match &self.generate {
            None => Ok(None),
            Some(generate) => Ok(Some(Arc::new(SandboxMemoryGenerationManager::new(
                self.layout.clone(),
                generate.clone(),
            )?))),
        }
    }

    /// The memory file layout.
    pub fn storage(&self) -> SandboxMemoryStorage {
        SandboxMemoryStorage::new(self.layout.clone())
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Memory {
    fn capability_type(&self) -> &str {
        "memory"
    }

    fn required_capability_types(&self) -> Vec<&'static str> {
        match &self.read {
            None => Vec::new(),
            Some(read) if read.live_update => vec!["filesystem", "shell"],
            Some(_) => vec!["shell"],
        }
    }

    fn as_memory(&self) -> Option<&Memory> {
        Some(self)
    }

    async fn instructions(&self, _manifest: &Manifest) -> Result<Option<String>> {
        Ok(None)
    }

    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Result<Option<String>> {
        let Some(read) = &self.read else {
            return Ok(None);
        };
        let storage = self.storage();
        // The summary only exists once a previous session consolidated memory.
        let Some(summary) = storage
            .read_text_opt(binding.session.as_ref(), &storage.memory_summary_path())
            .await?
        else {
            return Ok(None);
        };
        let (summary, _) = truncate_text_by_tokens(summary.trim(), Some(MEMORY_SUMMARY_MAX_TOKENS));
        if summary.trim().is_empty() {
            return Ok(None);
        }
        let memory_dir = match binding.scope.cwd() {
            None => self.layout.memories_dir.clone(),
            Some(_) => format!(
                "{}/{}",
                manifest.root().trim_end_matches('/'),
                self.layout.memories_dir
            ),
        };
        Ok(Some(render_memory_read_prompt(
            &memory_dir,
            &summary,
            read.live_update,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_read_or_generate() {
        assert!(
            Memory::new()
                .with_read(None)
                .with_generate(None)
                .validate()
                .is_err()
        );
        assert!(Memory::new().with_generate(None).validate().is_ok());
        assert!(Memory::new().with_read(None).validate().is_ok());
    }

    #[test]
    fn declares_required_capabilities() {
        let live = Memory::new();
        assert_eq!(
            Capability::<()>::required_capability_types(&live),
            vec!["filesystem", "shell"]
        );
        let read_only =
            Memory::new().with_read(Some(MemoryReadConfig::new().with_live_update(false)));
        assert_eq!(
            Capability::<()>::required_capability_types(&read_only),
            vec!["shell"]
        );
        let generate_only = Memory::new().with_read(None);
        assert!(Capability::<()>::required_capability_types(&generate_only).is_empty());
    }

    #[test]
    fn builds_one_manager_per_session() {
        let memory = Memory::new();
        let first = memory.build_manager().unwrap().unwrap();
        let second = memory.build_manager().unwrap().unwrap();
        // Separate sessions must not share rollout state.
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(
            Memory::new()
                .with_generate(None)
                .build_manager()
                .unwrap()
                .is_none()
        );
    }
}
