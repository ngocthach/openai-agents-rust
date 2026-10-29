//! Reading and writing the sandbox memory files. Mirrors
//! `agents.sandbox.memory.storage`.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::sandbox::errors::{ErrorCode, SandboxResult};
use crate::sandbox::memory::config::MemoryLayoutConfig;
use crate::sandbox::session::{ExecRequest, SandboxSession};

/// One phase-1 artifact considered by phase-2 consolidation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhaseTwoSelectionItem {
    /// The rollout this artifact came from.
    pub rollout_id: String,
    /// When the artifact was last written.
    pub updated_at: String,
    /// The rollout JSONL file the artifact was extracted from.
    pub rollout_path: String,
    /// The per-rollout summary file name.
    pub rollout_summary_file: String,
    /// How the rollout ended.
    pub terminal_state: String,
}

impl PhaseTwoSelectionItem {
    fn to_value(&self) -> Value {
        json!({
            "rollout_id": self.rollout_id,
            "updated_at": self.updated_at,
            "rollout_path": self.rollout_path,
            "rollout_summary_file": self.rollout_summary_file,
            "terminal_state": self.terminal_state,
        })
    }

    fn from_value(value: &Value) -> Option<Self> {
        let field = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string()
        };
        let rollout_id = field("rollout_id");
        let rollout_summary_file = field("rollout_summary_file");
        if rollout_id.is_empty() || rollout_summary_file.is_empty() {
            return None;
        }
        Some(PhaseTwoSelectionItem {
            rollout_id,
            updated_at: field("updated_at"),
            rollout_path: field("rollout_path"),
            rollout_summary_file,
            terminal_state: field("terminal_state"),
        })
    }

    /// Parse the front-matter-style header phase 1 writes at the top of a raw memory file.
    pub fn from_raw_memory(raw_memory: &str) -> Option<Self> {
        let mut item = PhaseTwoSelectionItem::default();
        for line in raw_memory.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().to_string();
            match key.trim() {
                "rollout_id" => item.rollout_id = value,
                "updated_at" => item.updated_at = value,
                "rollout_path" => item.rollout_path = value,
                "rollout_summary_file" => item.rollout_summary_file = value,
                "terminal_state" => item.terminal_state = value,
                _ => {}
            }
        }
        (!item.rollout_id.is_empty() && !item.rollout_summary_file.is_empty()).then_some(item)
    }
}

/// Which phase-1 artifacts phase 2 should consolidate.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhaseTwoInputSelection {
    /// The artifacts selected for this consolidation run, newest first.
    pub selected: Vec<PhaseTwoSelectionItem>,
    /// Rollout ids that were also in the previous successful selection.
    pub retained_rollout_ids: BTreeSet<String>,
    /// Artifacts that dropped out of the previous successful selection.
    pub removed: Vec<PhaseTwoSelectionItem>,
}

/// Reads and writes the sandbox memory files for one session and layout.
#[derive(Debug, Clone)]
pub struct SandboxMemoryStorage {
    layout: MemoryLayoutConfig,
}

impl SandboxMemoryStorage {
    /// Storage using `layout`.
    pub fn new(layout: MemoryLayoutConfig) -> Self {
        SandboxMemoryStorage { layout }
    }

    /// The per-rollout artifact directory, relative to the workspace root.
    pub fn sessions_dir(&self) -> &str {
        &self.layout.sessions_dir
    }

    /// The consolidated memory directory, relative to the workspace root.
    pub fn memories_dir(&self) -> &str {
        &self.layout.memories_dir
    }

    /// The directory holding one raw memory file per rollout.
    pub fn raw_memories_dir(&self) -> String {
        format!("{}/raw_memories", self.layout.memories_dir)
    }

    /// The directory holding one summary file per rollout.
    pub fn rollout_summaries_dir(&self) -> String {
        format!("{}/rollout_summaries", self.layout.memories_dir)
    }

    /// The file recording the previous successful phase-2 selection.
    pub fn phase_two_selection_path(&self) -> String {
        format!("{}/phase_two_selection.json", self.layout.memories_dir)
    }

    /// The consolidated memory file the agent reads and may update.
    pub fn memory_file_path(&self) -> String {
        format!("{}/MEMORY.md", self.layout.memories_dir)
    }

    /// The short summary injected into the agent instructions.
    pub fn memory_summary_path(&self) -> String {
        format!("{}/memory_summary.md", self.layout.memories_dir)
    }

    /// Create the directory layout and the files the agent expects to exist.
    pub async fn ensure_layout(&self, session: &dyn SandboxSession) -> SandboxResult<()> {
        for dir in [
            self.layout.sessions_dir.clone(),
            self.layout.memories_dir.clone(),
            self.raw_memories_dir(),
            self.rollout_summaries_dir(),
            format!("{}/skills", self.layout.memories_dir),
        ] {
            session.mkdir(&dir, true).await?;
        }
        self.ensure_text_file(session, &self.memory_file_path())
            .await?;
        self.ensure_text_file(session, &self.memory_summary_path())
            .await
    }

    /// Create an empty file when it does not exist yet.
    pub async fn ensure_text_file(
        &self,
        session: &dyn SandboxSession,
        path: &str,
    ) -> SandboxResult<()> {
        let resolved = session.normalize_path(path)?;
        let exists = session
            .exec(ExecRequest::argv(["test", "-f", &resolved]))
            .await?;
        if exists.ok() {
            return Ok(());
        }
        session.write_file(path, b"").await
    }

    /// Read a memory file as text.
    pub async fn read_text(
        &self,
        session: &dyn SandboxSession,
        path: &str,
    ) -> SandboxResult<String> {
        let bytes = session.read_file(path).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Read a memory file, returning `None` when it does not exist.
    pub async fn read_text_opt(
        &self,
        session: &dyn SandboxSession,
        path: &str,
    ) -> SandboxResult<Option<String>> {
        match session.read_file(path).await {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Write a memory file.
    pub async fn write_text(
        &self,
        session: &dyn SandboxSession,
        path: &str,
        text: &str,
    ) -> SandboxResult<()> {
        session.write_file(path, text.as_bytes()).await
    }

    /// Append one JSONL record to a rollout file.
    pub async fn append_rollout(
        &self,
        session: &dyn SandboxSession,
        file_name: &str,
        line: &str,
    ) -> SandboxResult<String> {
        let path = format!("{}/{file_name}", self.layout.sessions_dir);
        let existing = self
            .read_text_opt(session, &path)
            .await?
            .unwrap_or_default();
        let mut contents = existing;
        if !contents.is_empty() && !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str(line.trim_end_matches('\n'));
        contents.push('\n');
        self.write_text(session, &path, &contents).await?;
        Ok(path)
    }

    /// Choose the phase-1 artifacts phase 2 should consolidate.
    pub async fn build_phase_two_input_selection(
        &self,
        session: &dyn SandboxSession,
        max_raw_memories_for_consolidation: usize,
    ) -> SandboxResult<PhaseTwoInputSelection> {
        let mut current = self.list_current_selection_items(session).await?;
        current.truncate(max_raw_memories_for_consolidation);
        let prior = self.read_phase_two_selection(session).await?;
        let selected_ids: BTreeSet<String> = current.iter().map(|i| i.rollout_id.clone()).collect();
        let prior_ids: BTreeSet<String> = prior.iter().map(|i| i.rollout_id.clone()).collect();
        Ok(PhaseTwoInputSelection {
            retained_rollout_ids: selected_ids.intersection(&prior_ids).cloned().collect(),
            removed: prior
                .into_iter()
                .filter(|i| !selected_ids.contains(&i.rollout_id))
                .collect(),
            selected: current,
        })
    }

    /// Concatenate the selected raw memories into `raw_memories.md`.
    pub async fn rebuild_raw_memories(
        &self,
        session: &dyn SandboxSession,
        selected: &[PhaseTwoSelectionItem],
    ) -> SandboxResult<bool> {
        let mut chunks = Vec::new();
        for item in selected {
            let path = format!("{}/{}.md", self.raw_memories_dir(), item.rollout_id);
            if let Some(text) = self.read_text_opt(session, &path).await? {
                chunks.push(text.trim_end_matches('\n').to_string());
            }
        }
        if chunks.is_empty() {
            return Ok(false);
        }
        let path = format!("{}/raw_memories.md", self.layout.memories_dir);
        self.write_text(session, &path, &chunks.join("\n\n"))
            .await?;
        Ok(true)
    }

    /// Read the previous successful phase-2 selection.
    pub async fn read_phase_two_selection(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<Vec<PhaseTwoSelectionItem>> {
        let Some(raw) = self
            .read_text_opt(session, &self.phase_two_selection_path())
            .await?
        else {
            return Ok(Vec::new());
        };
        let Ok(payload) = serde_json::from_str::<Value>(&raw) else {
            return Ok(Vec::new());
        };
        Ok(payload
            .get("selected")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(PhaseTwoSelectionItem::from_value)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Record the selection phase 2 just consolidated.
    pub async fn write_phase_two_selection(
        &self,
        session: &dyn SandboxSession,
        selected: &[PhaseTwoSelectionItem],
    ) -> SandboxResult<()> {
        let payload = json!({
            "version": 1,
            "updated_at": chrono::Utc::now().to_rfc3339(),
            "selected": selected.iter().map(PhaseTwoSelectionItem::to_value).collect::<Vec<_>>(),
        });
        let text = format!(
            "{}\n",
            serde_json::to_string_pretty(&payload).unwrap_or_default()
        );
        self.write_text(session, &self.phase_two_selection_path(), &text)
            .await
    }

    async fn list_current_selection_items(
        &self,
        session: &dyn SandboxSession,
    ) -> SandboxResult<Vec<PhaseTwoSelectionItem>> {
        let dir = self.raw_memories_dir();
        let Ok(entries) = session.ls(&dir).await else {
            return Ok(Vec::new());
        };
        let mut items: Vec<(String, String, PhaseTwoSelectionItem)> = Vec::new();
        for entry in entries {
            if entry.is_dir || !entry.name.ends_with(".md") {
                continue;
            }
            let path = format!("{dir}/{}", entry.name);
            let Some(raw) = self.read_text_opt(session, &path).await? else {
                continue;
            };
            let Some(item) = PhaseTwoSelectionItem::from_raw_memory(raw.trim_end_matches('\n'))
            else {
                continue;
            };
            items.push((item.updated_at.clone(), item.rollout_id.clone(), item));
        }
        // Newest first, breaking ties on the rollout id so the order is deterministic.
        items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        Ok(items.into_iter().map(|(_, _, item)| item).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::manifest::Manifest;
    use crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use crate::sandbox::session::SandboxClient;

    #[test]
    fn parses_raw_memory_headers() {
        let raw = "rollout_id: r1\nupdated_at: 2026-01-01T00:00:00Z\n\
                   rollout_summary_file: r1.md\nterminal_state: completed\n\n# notes";
        let item = PhaseTwoSelectionItem::from_raw_memory(raw).unwrap();
        assert_eq!(item.rollout_id, "r1");
        assert_eq!(item.terminal_state, "completed");
        assert!(PhaseTwoSelectionItem::from_raw_memory("# nothing here").is_none());
    }

    #[tokio::test]
    async fn creates_the_layout_and_appends_rollouts() {
        let client = UnixLocalSandboxClient::new();
        let session = client.create(Some(Manifest::new())).await.unwrap();
        session.start().await.unwrap();
        let storage = SandboxMemoryStorage::new(MemoryLayoutConfig::new());
        storage.ensure_layout(session.as_ref()).await.unwrap();

        assert_eq!(
            storage
                .read_text(session.as_ref(), &storage.memory_file_path())
                .await
                .unwrap(),
            ""
        );
        storage
            .append_rollout(session.as_ref(), "r1.jsonl", "{\"a\":1}")
            .await
            .unwrap();
        storage
            .append_rollout(session.as_ref(), "r1.jsonl", "{\"a\":2}")
            .await
            .unwrap();
        let contents = storage
            .read_text(session.as_ref(), "sessions/r1.jsonl")
            .await
            .unwrap();
        assert_eq!(contents, "{\"a\":1}\n{\"a\":2}\n");

        client.delete(session).await.unwrap();
    }

    #[tokio::test]
    async fn builds_and_records_the_phase_two_selection() {
        let client = UnixLocalSandboxClient::new();
        let session = client.create(Some(Manifest::new())).await.unwrap();
        session.start().await.unwrap();
        let storage = SandboxMemoryStorage::new(MemoryLayoutConfig::new());
        storage.ensure_layout(session.as_ref()).await.unwrap();

        for (id, updated_at) in [
            ("r1", "2026-01-01T00:00:00Z"),
            ("r2", "2026-02-01T00:00:00Z"),
        ] {
            storage
                .write_text(
                    session.as_ref(),
                    &format!("{}/{id}.md", storage.raw_memories_dir()),
                    &format!(
                        "rollout_id: {id}\nupdated_at: {updated_at}\n\
                         rollout_summary_file: {id}.md\nterminal_state: completed\n\nnotes for {id}"
                    ),
                )
                .await
                .unwrap();
        }

        let selection = storage
            .build_phase_two_input_selection(session.as_ref(), 10)
            .await
            .unwrap();
        // Newest first.
        assert_eq!(
            selection
                .selected
                .iter()
                .map(|i| i.rollout_id.as_str())
                .collect::<Vec<_>>(),
            vec!["r2", "r1"]
        );
        assert!(selection.retained_rollout_ids.is_empty());

        assert!(
            storage
                .rebuild_raw_memories(session.as_ref(), &selection.selected)
                .await
                .unwrap()
        );
        let combined = storage
            .read_text(session.as_ref(), "memories/raw_memories.md")
            .await
            .unwrap();
        assert!(combined.contains("notes for r1"));
        assert!(combined.contains("notes for r2"));

        storage
            .write_phase_two_selection(session.as_ref(), &selection.selected)
            .await
            .unwrap();
        let next = storage
            .build_phase_two_input_selection(session.as_ref(), 1)
            .await
            .unwrap();
        assert_eq!(next.selected.len(), 1);
        assert_eq!(next.retained_rollout_ids.len(), 1);
        assert_eq!(next.removed.len(), 1);
        assert_eq!(next.removed[0].rollout_id, "r1");

        client.delete(session).await.unwrap();
    }
}
