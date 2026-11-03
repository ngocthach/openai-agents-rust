//! Configuration for sandbox-backed memory. Mirrors the memory types in
//! `agents.sandbox.config`.

use crate::agent::AgentModel;
use crate::model_settings::{ModelSettings, Reasoning};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// Filesystem layout for sandbox-backed memory generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLayoutConfig {
    /// Directory used for consolidated memory files.
    pub memories_dir: String,
    /// Directory used for per-rollout JSONL artifacts.
    pub sessions_dir: String,
}

impl Default for MemoryLayoutConfig {
    fn default() -> Self {
        MemoryLayoutConfig {
            memories_dir: "memories".to_string(),
            sessions_dir: "sessions".to_string(),
        }
    }
}

impl MemoryLayoutConfig {
    /// The default layout (`memories/` and `sessions/`).
    pub fn new() -> Self {
        MemoryLayoutConfig::default()
    }

    /// Set the consolidated memory directory.
    pub fn with_memories_dir(mut self, dir: impl Into<String>) -> Self {
        self.memories_dir = dir.into();
        self
    }

    /// Set the per-rollout artifact directory.
    pub fn with_sessions_dir(mut self, dir: impl Into<String>) -> Self {
        self.sessions_dir = dir.into();
        self
    }

    /// Validate that both directories are workspace-relative and non-empty.
    pub fn validate(&self) -> SandboxResult<()> {
        validate_relative_dir("layout.memories_dir", &self.memories_dir)?;
        validate_relative_dir("layout.sessions_dir", &self.sessions_dir)
    }
}

/// Validate one layout directory, reporting the same three reasons Python's
/// `_validate_relative_path` does and in the same order: absolute first, then an escape, then
/// empty. The caller needs to know which of the three it hit, so they are not one message.
fn validate_relative_dir(name: &str, value: &str) -> SandboxResult<()> {
    let invalid = |message: String| SandboxError::new(ErrorCode::SandboxConfigInvalid, message);
    if value.starts_with('/') {
        return Err(invalid(format!(
            "{name} must be relative to the sandbox workspace root, got: {value}"
        )));
    }
    if value.split('/').any(|part| part == "..") {
        return Err(invalid(format!(
            "{name} must not escape root, got: {value}"
        )));
    }
    if value.trim().is_empty() || value == "." {
        return Err(invalid(format!("{name} must be non-empty")));
    }
    Ok(())
}

/// Configuration for sandbox-backed memory reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReadConfig {
    /// Whether the agent may update stale memory files in place during a run.
    pub live_update: bool,
}

impl Default for MemoryReadConfig {
    fn default() -> Self {
        MemoryReadConfig { live_update: true }
    }
}

impl MemoryReadConfig {
    /// The default read configuration (live updates enabled).
    pub fn new() -> Self {
        MemoryReadConfig::default()
    }

    /// Whether the agent may update memory files during the run.
    pub fn with_live_update(mut self, live_update: bool) -> Self {
        self.live_update = live_update;
        self
    }
}

/// Configuration for sandbox-backed memory extraction and consolidation.
///
/// Run segments are appended during the sandbox session. Extraction and consolidation run when
/// the sandbox session closes.
#[derive(Debug, Clone)]
pub struct MemoryGenerateConfig {
    /// Maximum number of recent raw memories considered during consolidation.
    pub max_raw_memories_for_consolidation: usize,
    /// Model used for phase-1 single-rollout extraction.
    pub phase_one_model: AgentModel,
    /// Model settings used for phase-1 extraction.
    pub phase_one_model_settings: Option<ModelSettings>,
    /// Model used for phase-2 memory consolidation.
    pub phase_two_model: AgentModel,
    /// Model settings used for phase-2 consolidation.
    pub phase_two_model_settings: Option<ModelSettings>,
    /// Developer-specific guidance appended to the extraction and consolidation prompts.
    pub extra_prompt: Option<String>,
    /// Maximum agent turns for the separate phase-2 consolidation run.
    pub phase_two_max_turns: usize,
}

fn medium_reasoning() -> ModelSettings {
    ModelSettings {
        reasoning: Some(Reasoning {
            effort: Some("medium".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

impl PartialEq for MemoryGenerateConfig {
    /// Value equality; model instances compare by identity, as Python's `==` on model objects.
    fn eq(&self, other: &Self) -> bool {
        fn same_model(a: &AgentModel, b: &AgentModel) -> bool {
            match (a, b) {
                (AgentModel::Name(a), AgentModel::Name(b)) => a == b,
                (AgentModel::Instance(a), AgentModel::Instance(b)) => std::sync::Arc::ptr_eq(a, b),
                _ => false,
            }
        }
        self.max_raw_memories_for_consolidation == other.max_raw_memories_for_consolidation
            && same_model(&self.phase_one_model, &other.phase_one_model)
            && self.phase_one_model_settings == other.phase_one_model_settings
            && same_model(&self.phase_two_model, &other.phase_two_model)
            && self.phase_two_model_settings == other.phase_two_model_settings
            && self.extra_prompt == other.extra_prompt
            && self.phase_two_max_turns == other.phase_two_max_turns
    }
}

impl Default for MemoryGenerateConfig {
    fn default() -> Self {
        MemoryGenerateConfig {
            max_raw_memories_for_consolidation: 256,
            phase_one_model: AgentModel::Name("gpt-5.4-mini".to_string()),
            phase_one_model_settings: Some(medium_reasoning()),
            phase_two_model: AgentModel::Name("gpt-5.5".to_string()),
            phase_two_model_settings: Some(medium_reasoning()),
            extra_prompt: None,
            phase_two_max_turns: 500,
        }
    }
}

impl MemoryGenerateConfig {
    /// The default generation configuration.
    pub fn new() -> Self {
        MemoryGenerateConfig::default()
    }

    /// Set the phase-1 extraction model. Accepts a model name or a model instance, so tests can
    /// inject a scripted model.
    pub fn with_phase_one_model(mut self, model: impl Into<AgentModel>) -> Self {
        self.phase_one_model = model.into();
        self
    }

    /// Set the phase-2 consolidation model.
    pub fn with_phase_two_model(mut self, model: impl Into<AgentModel>) -> Self {
        self.phase_two_model = model.into();
        self
    }

    /// Set the phase-1 model settings.
    pub fn with_phase_one_model_settings(mut self, settings: Option<ModelSettings>) -> Self {
        self.phase_one_model_settings = settings;
        self
    }

    /// Set the phase-2 model settings.
    pub fn with_phase_two_model_settings(mut self, settings: Option<ModelSettings>) -> Self {
        self.phase_two_model_settings = settings;
        self
    }

    /// Append developer-specific guidance to the memory prompts.
    pub fn with_extra_prompt(mut self, extra_prompt: impl Into<String>) -> Self {
        self.extra_prompt = Some(extra_prompt.into());
        self
    }

    /// Set the turn limit of the phase-2 consolidation run.
    pub fn with_phase_two_max_turns(mut self, max_turns: usize) -> Self {
        self.phase_two_max_turns = max_turns;
        self
    }

    /// Set how many recent raw memories consolidation considers.
    pub fn with_max_raw_memories_for_consolidation(mut self, max: usize) -> Self {
        self.max_raw_memories_for_consolidation = max;
        self
    }

    /// Validate the configured limits.
    pub fn validate(&self) -> SandboxResult<()> {
        if self.max_raw_memories_for_consolidation == 0 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "MemoryGenerateConfig.max_raw_memories_for_consolidation must be greater than 0.",
            ));
        }
        if self.max_raw_memories_for_consolidation > 4096 {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "MemoryGenerateConfig.max_raw_memories_for_consolidation must be less than or \
                 equal to 4096.",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_layouts() {
        MemoryLayoutConfig::new().validate().unwrap();
        for bad in ["", ".", "/abs", "../escape", "a/../../b"] {
            let layout = MemoryLayoutConfig::new().with_memories_dir(bad);
            assert!(layout.validate().is_err(), "{bad}");
        }
    }

    #[test]
    fn validates_generation_limits() {
        MemoryGenerateConfig::new().validate().unwrap();
        assert!(
            MemoryGenerateConfig::new()
                .with_max_raw_memories_for_consolidation(0)
                .validate()
                .is_err()
        );
        assert!(
            MemoryGenerateConfig::new()
                .with_max_raw_memories_for_consolidation(5000)
                .validate()
                .is_err()
        );
    }
}
