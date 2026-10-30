//! Background memory generation for a sandbox session. Mirrors
//! `agents.sandbox.memory.manager`, `phase_one` and `phase_two`.
//!
//! During the session, each finished run appends a segment to a per-rollout JSONL file. When the
//! session closes, [`SandboxMemoryGenerationManager::flush`] runs phase 1 (one extraction run per
//! rollout) and then a single phase-2 consolidation run that edits the memory files in the
//! workspace.

use std::collections::BTreeMap;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::agent::Agent;
use crate::error::Result;
use crate::models::ModelProvider;
use crate::run::{RunOptions, Runner};
use crate::run_config::RunConfig;
use crate::sandbox::capabilities::{Capability, Filesystem, Shell};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::memory::config::{MemoryGenerateConfig, MemoryLayoutConfig};
use crate::sandbox::memory::prompts::{
    render_memory_consolidation_prompt, render_rollout_extraction_prompt,
};
use crate::sandbox::memory::rollouts::render_phase_one_prompt;
use crate::sandbox::memory::storage::{PhaseTwoSelectionItem, SandboxMemoryStorage};
use crate::sandbox::session::SandboxSession;
use crate::sandbox::{CapabilityBinding, SandboxAgent};

/// Token budget of the rollout contents rendered into the phase-1 prompt.
pub const PHASE_ONE_ROLLOUT_TOKEN_LIMIT: usize = 150_000;

/// The structured output phase 1 produces for one rollout.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RolloutExtractionArtifacts {
    /// A short slug naming the rollout.
    pub rollout_slug: String,
    /// A human-readable summary of what happened in the rollout.
    pub rollout_summary: String,
    /// The memory worth keeping from the rollout.
    pub raw_memory: String,
}

impl RolloutExtractionArtifacts {
    /// Whether the artifacts carry anything worth writing.
    ///
    /// Returns an error when phase 1 filled only some of the fields, mirroring Python's
    /// `validate_rollout_artifacts`.
    pub fn validate(&self) -> SandboxResult<bool> {
        let slug = self.rollout_slug.trim();
        let summary = self.rollout_summary.trim();
        let memory = self.raw_memory.trim();
        if slug.is_empty() && summary.is_empty() && memory.is_empty() {
            return Ok(false);
        }
        if slug.is_empty() || summary.is_empty() || memory.is_empty() {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "phase 1 returned partially-empty memory artifacts",
            ));
        }
        Ok(true)
    }
}

/// Normalize a phase-1 rollout slug.
pub fn normalize_rollout_slug(value: &str) -> SandboxResult<String> {
    let slug = value.trim().strip_suffix(".md").unwrap_or(value.trim());
    let valid = (1..=80).contains(&slug.len())
        && slug
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if !valid {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            format!("invalid rollout_slug: `{value}`"),
        ));
    }
    Ok(slug.to_string())
}

/// Validate a rollout id used as a file name.
pub fn validate_rollout_id(value: &str) -> SandboxResult<String> {
    let id = value.trim();
    let valid = (1..=128).contains(&id.len())
        && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-');
    if !valid {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            format!("invalid rollout id for memory: `{value}`"),
        ));
    }
    Ok(id.to_string())
}

/// Manages the rollout files and the two generation phases for one sandbox session.
#[derive(Debug)]
pub struct SandboxMemoryGenerationManager {
    storage: SandboxMemoryStorage,
    config: MemoryGenerateConfig,
    rollout_files: Mutex<BTreeMap<String, String>>,
    stopped: Mutex<bool>,
}

impl SandboxMemoryGenerationManager {
    /// A manager writing into `layout` with the given generation configuration.
    pub fn new(layout: MemoryLayoutConfig, config: MemoryGenerateConfig) -> SandboxResult<Self> {
        layout.validate()?;
        config.validate()?;
        Ok(SandboxMemoryGenerationManager {
            storage: SandboxMemoryStorage::new(layout),
            config,
            rollout_files: Mutex::new(BTreeMap::new()),
            stopped: Mutex::new(false),
        })
    }

    /// The memory file layout.
    pub fn storage(&self) -> &SandboxMemoryStorage {
        &self.storage
    }

    /// Append one run segment to the rollout file of `rollout_id`.
    pub async fn enqueue_rollout_payload(
        &self,
        session: &dyn SandboxSession,
        rollout_id: &str,
        payload: &Value,
    ) -> SandboxResult<()> {
        if *self.stopped.lock().await {
            return Ok(());
        }
        let rollout_id = validate_rollout_id(rollout_id)?;
        self.storage.ensure_layout(session).await?;
        let file_name = format!("{rollout_id}.jsonl");
        let line = crate::sandbox::memory::rollouts::dump_rollout_json(payload);
        let path = self
            .storage
            .append_rollout(session, &file_name, &line)
            .await?;
        self.rollout_files
            .lock()
            .await
            .insert(rollout_id, path.clone());
        Ok(())
    }

    /// Record a finished run as a rollout segment.
    pub async fn enqueue_result<C>(
        &self,
        session: &dyn SandboxSession,
        rollout_id: &str,
        result: &crate::result::RunResult<C>,
    ) -> SandboxResult<()> {
        let payload = crate::sandbox::memory::rollouts::build_rollout_payload_from_result(result);
        self.enqueue_rollout_payload(session, rollout_id, &payload)
            .await
    }

    /// Run phase 1 for every pending rollout, then one phase-2 consolidation.
    ///
    /// This is the session-close path: it is idempotent, and a second call after the manager has
    /// stopped does nothing. Phase agents are built without the memory capability, so a flush can
    /// never trigger another flush.
    ///
    /// The phase runs use their own unit app context: the memory agents only touch the sandbox
    /// workspace, so they never need the caller's context type.
    pub async fn flush(
        &self,
        session: Arc<dyn SandboxSession>,
        model_provider: Arc<dyn ModelProvider>,
    ) -> Result<()> {
        {
            let mut stopped = self.stopped.lock().await;
            if *stopped {
                return Ok(());
            }
            *stopped = true;
        }
        let rollouts: Vec<(String, String)> = self
            .rollout_files
            .lock()
            .await
            .iter()
            .map(|(id, path)| (id.clone(), path.clone()))
            .collect();
        if rollouts.is_empty() {
            return Ok(());
        }

        let mut extracted: Vec<PhaseTwoSelectionItem> = Vec::new();
        for (rollout_id, rollout_path) in rollouts {
            match self
                .run_phase_one(
                    session.as_ref(),
                    &rollout_id,
                    &rollout_path,
                    &model_provider,
                )
                .await
            {
                Ok(Some(item)) => extracted.push(item),
                Ok(None) => {}
                Err(error) => {
                    log::warn!(target: crate::logger::LOG_TARGET, "sandbox memory phase 1 failed for `{rollout_id}`: {error}");
                }
            }
        }
        if extracted.is_empty() {
            return Ok(());
        }

        if let Err(error) = self.run_phase_two(session.clone(), &model_provider).await {
            log::warn!(target: crate::logger::LOG_TARGET, "sandbox memory phase 2 failed: {error}");
        }
        Ok(())
    }

    async fn run_phase_one(
        &self,
        session: &dyn SandboxSession,
        rollout_id: &str,
        rollout_path: &str,
        model_provider: &Arc<dyn ModelProvider>,
    ) -> Result<Option<PhaseTwoSelectionItem>> {
        let Some(contents) = self.storage.read_text_opt(session, rollout_path).await? else {
            return Ok(None);
        };
        let Some(prompt) = render_phase_one_prompt(&contents, PHASE_ONE_ROLLOUT_TOKEN_LIMIT) else {
            return Ok(None);
        };

        let mut agent = Agent::<()>::new("sandbox-memory-phase-one")
            .with_instructions(render_rollout_extraction_prompt(
                self.config.extra_prompt.as_deref(),
            ))
            .with_output_type::<RolloutExtractionArtifacts>()
            .with_model(self.config.phase_one_model.clone());
        if let Some(settings) = &self.config.phase_one_model_settings {
            agent = agent.with_model_settings(settings.clone());
        }

        let run_config = RunConfig::<()> {
            model_provider: model_provider.clone(),
            ..Default::default()
        };
        let result =
            Runner::run_with_options(agent, prompt, RunOptions::default().run_config(run_config))
                .await?;
        let Some(output) = result.final_output.as_ref() else {
            return Ok(None);
        };
        let artifacts: RolloutExtractionArtifacts = serde_json::from_value(output.clone())?;
        if !artifacts.validate()? {
            return Ok(None);
        }

        let slug = normalize_rollout_slug(&artifacts.rollout_slug)?;
        let summary_file = format!("{slug}.md");
        let updated_at = chrono::Utc::now().to_rfc3339();
        let terminal_state = contents
            .lines()
            .rfind(|line| !line.trim().is_empty())
            .and_then(|line| serde_json::from_str::<Value>(line).ok())
            .and_then(|payload| {
                payload
                    .get("terminal_metadata")
                    .and_then(|m| m.get("terminal_state"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string());

        self.storage
            .write_text(
                session,
                &format!("{}/{summary_file}", self.storage.rollout_summaries_dir()),
                artifacts.rollout_summary.trim(),
            )
            .await?;
        let header = format!(
            "rollout_id: {rollout_id}\nupdated_at: {updated_at}\nrollout_path: {rollout_path}\n\
             rollout_summary_file: {summary_file}\nterminal_state: {terminal_state}\n\n{}\n",
            artifacts.raw_memory.trim()
        );
        self.storage
            .write_text(
                session,
                &format!("{}/{rollout_id}.md", self.storage.raw_memories_dir()),
                &header,
            )
            .await?;

        Ok(Some(PhaseTwoSelectionItem {
            rollout_id: rollout_id.to_string(),
            updated_at,
            rollout_path: rollout_path.to_string(),
            rollout_summary_file: summary_file,
            terminal_state,
        }))
    }

    async fn run_phase_two(
        &self,
        session: Arc<dyn SandboxSession>,
        model_provider: &Arc<dyn ModelProvider>,
    ) -> Result<()> {
        let selection = self
            .storage
            .build_phase_two_input_selection(
                session.as_ref(),
                self.config.max_raw_memories_for_consolidation,
            )
            .await?;
        if selection.selected.is_empty() {
            return Ok(());
        }
        self.storage
            .rebuild_raw_memories(session.as_ref(), &selection.selected)
            .await?;

        // The consolidation agent edits memory files with the shell and apply_patch tools, and
        // deliberately carries no memory capability of its own.
        let capabilities: Vec<Arc<dyn Capability<()>>> =
            vec![Arc::new(Filesystem::new()), Arc::new(Shell::new())];
        let mut sandbox_agent = SandboxAgent::<()>::new("sandbox-memory-phase-two")
            .with_capabilities(capabilities)
            .with_model(self.config.phase_two_model.clone());
        if let Some(settings) = &self.config.phase_two_model_settings {
            sandbox_agent = sandbox_agent.map_agent(|a| a.with_model_settings(settings.clone()));
        }
        let prepared = sandbox_agent
            .prepare(&CapabilityBinding::new(session.clone()))
            .await?;

        let prompt = render_memory_consolidation_prompt(
            self.storage.memories_dir(),
            &selection,
            self.config.extra_prompt.as_deref(),
        );
        let run_config = RunConfig::<()> {
            model_provider: model_provider.clone(),
            ..Default::default()
        };
        Runner::run_with_options(
            prepared,
            prompt,
            RunOptions::default()
                .max_turns(Some(self.config.phase_two_max_turns))
                .run_config(run_config),
        )
        .await?;

        self.storage
            .write_phase_two_selection(session.as_ref(), &selection.selected)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_slugs_and_ids() {
        assert_eq!(
            normalize_rollout_slug("fix-the-build.md").unwrap(),
            "fix-the-build"
        );
        assert_eq!(normalize_rollout_slug(" a_b-1 ").unwrap(), "a_b-1");
        for bad in [
            "",
            "-leading",
            "Upper",
            "has space",
            "a".repeat(200).as_str(),
        ] {
            assert!(normalize_rollout_slug(bad).is_err(), "{bad}");
        }
        assert_eq!(validate_rollout_id("run.1-a_b").unwrap(), "run.1-a_b");
        for bad in ["", "-leading", "has space", "a/b"] {
            assert!(validate_rollout_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn validates_extraction_artifacts() {
        let empty = RolloutExtractionArtifacts {
            rollout_slug: " ".into(),
            rollout_summary: String::new(),
            raw_memory: String::new(),
        };
        assert!(!empty.validate().unwrap());

        let partial = RolloutExtractionArtifacts {
            rollout_slug: "slug".into(),
            rollout_summary: String::new(),
            raw_memory: "memory".into(),
        };
        assert!(partial.validate().is_err());

        let full = RolloutExtractionArtifacts {
            rollout_slug: "slug".into(),
            rollout_summary: "summary".into(),
            raw_memory: "memory".into(),
        };
        assert!(full.validate().unwrap());
    }
}
