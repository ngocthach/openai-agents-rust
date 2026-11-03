//! Ports the runner-driven half of `tests/sandbox/test_memory.py` from the Python SDK: rollouts
//! written by a sandbox run, and the two generation phases that run when the session closes.
//!
//! Python closes a caller-owned session with `session.aclose()`, which flushes the memory
//! manager. Rust never closes a session the caller owns; the caller flushes the manager the
//! outcome hands back, which is what [`flush_memory`] does. A session the runner owns is flushed
//! by the runner itself.
#![cfg(feature = "sandbox")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::AgentModel;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Memory, Shell};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::memory::{
    MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig, SandboxMemoryGenerationManager,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    Manifest, SandboxAgent, SandboxClient, SandboxResult, SandboxRunConfig, SandboxRunner,
    SandboxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{RunConfig, RunOptions};
use serde_json::{Value, json};

// ----------------------------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------------------------

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

/// A phase-one answer, as the extraction model would return it.
fn phase_one_message(slug: &str, raw_memory: &str) -> OutputItem {
    assistant_message(
        json!({
            "rollout_slug": slug,
            "rollout_summary": "# Task summary\n",
            "raw_memory": raw_memory,
        })
        .to_string(),
    )
}

/// A phase-two model that writes both memory files with the shell, then finishes.
fn consolidating_model(dir: &str, memory: &str, summary: &str) -> Arc<FakeModel> {
    let model = setup();
    model.add_turn(vec![shell_call(
        "consolidate",
        &format!(
            "printf '{memory}\\n' > {dir}/MEMORY.md && printf '{summary}\\n' > {dir}/memory_summary.md"
        ),
    )]);
    model.add_turn(vec![assistant_message("consolidated")]);
    model
}

fn phase_one_model(raw_memory: &str) -> Arc<FakeModel> {
    let model = setup();
    model.add_turn(vec![phase_one_message("task_memory", raw_memory)]);
    model
}

fn generate_config(phase_one: Arc<FakeModel>, phase_two: Arc<FakeModel>) -> MemoryGenerateConfig {
    MemoryGenerateConfig::new()
        .with_phase_one_model(AgentModel::Instance(phase_one))
        .with_phase_two_model(AgentModel::Instance(phase_two))
        .with_phase_one_model_settings(None)
        .with_phase_two_model_settings(None)
}

fn worker(name: &str, model: Arc<FakeModel>, memory: Memory) -> SandboxAgent<()> {
    SandboxAgent::<()>::new(name)
        .with_model_instance(model)
        .with_instructions("Worker.")
        .with_capabilities(vec![
            Arc::new(Filesystem::new()),
            Arc::new(Shell::new().with_pty(false)),
            Arc::new(memory),
        ])
}

async fn local_session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(Manifest::new()))
        .await
        .expect("create session");
    session.start().await.expect("start session");
    (client, session)
}

fn root_of(session: &Arc<dyn SandboxSession>) -> PathBuf {
    PathBuf::from(session.workspace_root())
}

fn read(root: &std::path::Path, relative: &str) -> String {
    std::fs::read_to_string(root.join(relative)).unwrap_or_else(|e| panic!("read {relative}: {e}"))
}

fn files_in(root: &std::path::Path, dir: &str, extension: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.join(dir))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|name| name.ends_with(extension))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// The text of the first input item of the first model call.
fn first_user_text(model: &FakeModel) -> String {
    let calls = model.calls();
    let first = calls.first().expect("the model was called");
    user_text(&first.input[0])
}

fn user_text(item: &InputItem) -> String {
    match &item.0["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts[0]["text"].as_str().expect("text part").to_string(),
        other => panic!("unexpected content {other}"),
    }
}

/// Run a sandbox agent against a caller-owned session and hand back the memory manager.
async fn run_on_session(
    agent: &SandboxAgent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: &Arc<dyn SandboxSession>,
    rollout_id: Option<&str>,
) -> (
    openai_agents_sdk::RunResult<()>,
    Option<Arc<SandboxMemoryGenerationManager>>,
) {
    let mut config = SandboxRunConfig::with_session(session.clone());
    if let Some(rollout_id) = rollout_id {
        config = config.rollout_id(rollout_id);
    }
    let outcome = SandboxRunner::run_to_outcome(agent, input, &config, RunOptions::default())
        .await
        .expect("run");
    let manager = outcome.memory_manager().cloned();
    (outcome.result, manager)
}

/// Close the memory side of a caller-owned session: what Python's `session.aclose()` does.
async fn flush_memory(
    manager: &Arc<SandboxMemoryGenerationManager>,
    session: &Arc<dyn SandboxSession>,
) {
    let provider = RunConfig::<()>::default().model_provider;
    manager
        .flush(session.clone(), provider)
        .await
        .expect("flush memory");
}

/// A client that delegates to the Unix-local backend and records the root of every workspace it
/// deletes, standing in for Python's `_DeleteTrackingFilesystemTestSandboxClient`.
#[derive(Debug, Default)]
struct DeleteTrackingClient {
    inner: UnixLocalSandboxClient,
    deleted_roots: Mutex<Vec<PathBuf>>,
}

#[async_trait]
impl SandboxClient for DeleteTrackingClient {
    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.inner.create(manifest).await
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        self.deleted_roots
            .lock()
            .expect("deleted roots")
            .push(PathBuf::from(session.workspace_root()));
        self.inner.delete(session).await
    }
}

// ----------------------------------------------------------------------------------------------
// Writing rollouts and memory files
// ----------------------------------------------------------------------------------------------

/// A run records one rollout, and nothing is extracted until the session closes. On close phase
/// one writes the raw memory and the rollout summary and phase two consolidates them.
#[tokio::test]
async fn sandbox_memory_writes_rollouts_and_memory_files() {
    let (client, session) = local_session().await;
    let phase_one = phase_one_model("raw memory entry\n");
    let phase_two = consolidating_model("memories", "memory entry", "summary entry");
    let memory = Memory::new().with_generate(Some(
        generate_config(phase_one.clone(), phase_two.clone())
            .with_extra_prompt("Track durable user preferences."),
    ));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory);

    let (result, manager) = run_on_session(&agent, "hello", &session, None).await;
    let manager = manager.expect("memory manager");
    let root = root_of(&session);

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(files_in(&root, "sessions", ".jsonl").len(), 1);
    assert!(
        phase_one.calls().is_empty(),
        "nothing is extracted before close"
    );

    flush_memory(&manager, &session).await;

    assert_eq!(files_in(&root, "memories/raw_memories", ".md").len(), 1);
    let summaries = files_in(&root, "memories/rollout_summaries", ".md");
    assert_eq!(summaries.len(), 1);
    assert_eq!(read(&root, "memories/MEMORY.md"), "memory entry\n");
    assert_eq!(read(&root, "memories/memory_summary.md"), "summary entry\n");
    let raw = read(&root, "memories/raw_memories.md");
    assert!(raw.contains("rollout_id: "), "{raw}");
    assert!(raw.contains("updated_at: "), "{raw}");
    assert!(raw.contains("rollout_path: sessions/"), "{raw}");
    assert!(
        raw.contains("rollout_summary_file: rollout_summaries/"),
        "{raw}"
    );
    assert!(raw.contains("terminal_state: completed"), "{raw}");
    let summary = read(
        &root,
        &format!("memories/rollout_summaries/{}", summaries[0]),
    );
    assert!(summary.contains("session_id: "), "{summary}");
    assert!(summary.contains("updated_at: "), "{summary}");
    assert!(summary.contains("rollout_path: sessions/"), "{summary}");
    assert!(summary.contains("terminal_state: completed"), "{summary}");
    assert!(first_user_text(&phase_one).contains("\"terminal_state\":\"completed\""));
    let system_instructions = phase_one.calls()[0]
        .system_instructions
        .clone()
        .expect("phase one instructions");
    assert!(system_instructions.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
    assert!(system_instructions.contains("Track durable user preferences."));
    let phase_two_prompt = first_user_text(&phase_two);
    assert!(phase_two_prompt.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
    assert!(phase_two_prompt.contains("Track durable user preferences."));
    client.delete(session).await.expect("delete");
}

/// A sandbox agent with no memory capability writes no memory directories at all.
#[tokio::test]
async fn sandbox_agent_without_memory_capability_skips_memory_generation() {
    let (client, session) = local_session().await;
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.");

    let config = SandboxRunConfig::with_session(session.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("run");

    let root = root_of(&session);
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(!root.join("sessions").exists());
    assert!(!root.join("memories").exists());
    client.delete(session).await.expect("delete");
}

#[tokio::test]
async fn sandbox_memory_uses_custom_layout() {
    let (client, session) = local_session().await;
    let phase_two = consolidating_model("agent_memory", "memory entry", "summary entry");
    let memory = Memory::new()
        .with_layout(
            MemoryLayoutConfig::new()
                .with_memories_dir("agent_memory")
                .with_sessions_dir("agent_sessions"),
        )
        .with_read(None)
        .with_generate(Some(generate_config(phase_one_model("raw\n"), phase_two)));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory);

    let (_, manager) = run_on_session(&agent, "hello", &session, None).await;
    let root = root_of(&session);
    assert_eq!(files_in(&root, "agent_sessions", ".jsonl").len(), 1);

    flush_memory(&manager.expect("manager"), &session).await;

    assert_eq!(read(&root, "agent_memory/MEMORY.md"), "memory entry\n");
    assert_eq!(
        read(&root, "agent_memory/memory_summary.md"),
        "summary entry\n"
    );
    client.delete(session).await.expect("delete");
}

/// Two agents with different layouts share one session; each keeps its own memory.
#[tokio::test]
async fn sandbox_memory_supports_multiple_generating_layouts_in_one_session() {
    let (client, session) = local_session().await;
    let mut managers = Vec::new();
    for (label, dir, sessions_dir) in [
        ("a", "agent_a_memory", "agent_a_sessions"),
        ("b", "agent_b_memory", "agent_b_sessions"),
    ] {
        let memory = Memory::new()
            .with_layout(
                MemoryLayoutConfig::new()
                    .with_memories_dir(dir)
                    .with_sessions_dir(sessions_dir),
            )
            .with_generate(Some(generate_config(
                phase_one_model(&format!("agent {label} raw\n")),
                consolidating_model(
                    dir,
                    &format!("agent {label} entry"),
                    &format!("agent {label} summary"),
                ),
            )));
        let model = setup();
        model.add_turn(vec![assistant_message(format!("{label} done"))]);
        let agent = worker(&format!("agent-{label}"), model, memory);
        let (_, manager) = run_on_session(&agent, "first", &session, None).await;
        managers.push(manager.expect("manager"));
    }
    let root = root_of(&session);
    assert_eq!(files_in(&root, "agent_a_sessions", ".jsonl").len(), 1);
    assert_eq!(files_in(&root, "agent_b_sessions", ".jsonl").len(), 1);

    for manager in &managers {
        flush_memory(manager, &session).await;
    }

    assert_eq!(read(&root, "agent_a_memory/MEMORY.md"), "agent a entry\n");
    assert_eq!(read(&root, "agent_b_memory/MEMORY.md"), "agent b entry\n");
    client.delete(session).await.expect("delete");
}

/// The id the file is named after replaces whatever id the payload carried.
#[tokio::test]
async fn sandbox_memory_rollout_payload_uses_validated_rollout_id() {
    let (client, session) = local_session().await;
    let manager = Memory::new()
        .build_manager()
        .expect("manager")
        .expect("generation enabled");

    manager
        .enqueue_rollout_payload(
            session.as_ref(),
            "canonical-id",
            &json!({
                "updated_at": "2026-04-15T00:00:00+00:00",
                "rollout_id": "payload-id",
                "input": [],
                "generated_items": [],
                "terminal_metadata": {"terminal_state": "completed", "has_final_output": false},
            }),
        )
        .await
        .expect("enqueue");

    let root = root_of(&session);
    let payload: Value =
        serde_json::from_str(read(&root, "sessions/canonical-id.jsonl").trim()).expect("json");
    assert_eq!(payload["rollout_id"], "canonical-id");
    let keys: Vec<&str> = payload
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(&keys[..2], &["updated_at", "rollout_id"]);

    let error = manager
        .enqueue_rollout_payload(session.as_ref(), "../escape", &json!({}))
        .await
        .expect_err("path-like id");
    assert!(error.message.contains("file-safe ID"), "{}", error.message);
    client.delete(session).await.expect("delete");
}

// ----------------------------------------------------------------------------------------------
// Grouping segments into rollouts
// ----------------------------------------------------------------------------------------------

/// Segments recorded under one rollout id land in one file, one line each, and phase one sees
/// them as a single multi-segment rollout.
///
/// Adapted: Python derives the id from the SDK session (`chat-session`); Rust takes it from
/// `SandboxRunConfig::rollout_id`.
#[tokio::test]
async fn sandbox_memory_groups_segments_by_rollout_id_until_close() {
    let (client, session) = local_session().await;
    let phase_one = phase_one_model("joined raw\n");
    let phase_two = consolidating_model("memories", "joined entry", "joined summary");
    let memory = Memory::new().with_generate(Some(generate_config(phase_one.clone(), phase_two)));
    let mut last_manager = None;
    for (name, input) in [("first-worker", "first"), ("second-worker", "second")] {
        let model = setup();
        model.add_turn(vec![assistant_message(format!("{input} done"))]);
        let agent = worker(name, model, memory.clone());
        let (result, manager) = run_on_session(&agent, input, &session, Some("chat-session")).await;
        assert_eq!(
            result.final_output_text().as_deref(),
            Some(format!("{input} done").as_str())
        );
        last_manager = manager;
    }
    let root = root_of(&session);

    assert_eq!(
        files_in(&root, "sessions", ".jsonl"),
        vec!["chat-session.jsonl"]
    );
    let contents = read(&root, "sessions/chat-session.jsonl");
    let segments: Vec<Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).expect("segment"))
        .collect();
    assert_eq!(segments.len(), 2);
    let keys: Vec<&str> = segments[0]
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        &keys[..4],
        &["updated_at", "rollout_id", "input", "generated_items"]
    );
    assert_eq!(
        segments[0]["input"],
        json!([{"role": "user", "content": "first"}])
    );
    assert_eq!(
        segments[1]["input"],
        json!([{"role": "user", "content": "second"}])
    );
    assert!(phase_one.calls().is_empty());

    flush_memory(&last_manager.expect("manager"), &session).await;

    let prompt = first_user_text(&phase_one);
    assert!(prompt.contains("first"));
    assert!(prompt.contains("second"));
    assert!(prompt.contains("\"segment_count\":2"), "{prompt}");
    let raw_files = files_in(&root, "memories/raw_memories", ".md");
    assert_eq!(raw_files.len(), 1);
    let raw = read(&root, &format!("memories/raw_memories/{}", raw_files[0]));
    let last_updated_at = segments[1]["updated_at"].as_str().expect("updated_at");
    assert!(
        raw.contains(&format!("updated_at: {last_updated_at}\n")),
        "{raw}"
    );
    assert_eq!(read(&root, "memories/MEMORY.md"), "joined entry\n");
    client.delete(session).await.expect("delete");
}

/// DEFERRED: Python names the rollout after the conversation id when the run has no SDK
/// session (`conversation-123.jsonl`) and after the trace `group_id` when it has no conversation
/// id either. `SandboxRunner::prepare` (`src/sandbox/runtime.rs`, `rollout_id`) only knows
/// `SandboxRunConfig::rollout_id`, the resumed state and a random id, so a run with neither
/// gets a generated id. Asserted as current behavior.
#[tokio::test]
async fn sandbox_memory_without_a_rollout_id_generates_one_per_run() {
    let (client, session) = local_session().await;
    let memory = Memory::new().with_generate(Some(generate_config(
        phase_one_model("raw\n"),
        consolidating_model("memories", "entry", "summary"),
    )));
    let model = setup();
    model.add_turn(vec![assistant_message("first done")]);
    model.add_turn(vec![assistant_message("second done")]);
    let agent = worker("worker", model, memory);

    run_on_session(&agent, "first", &session, None).await;
    run_on_session(&agent, "second", &session, None).await;

    let root = root_of(&session);
    let rollouts = files_in(&root, "sessions", ".jsonl");
    assert_eq!(rollouts.len(), 2, "{rollouts:?}");
    assert!(
        rollouts.iter().all(|name| name != "conversation-123.jsonl"),
        "{rollouts:?}"
    );
    client.delete(session).await.expect("delete");
}

// ----------------------------------------------------------------------------------------------
// Phase one and phase two
// ----------------------------------------------------------------------------------------------

/// The phase-one prompt carries neither developer or system input, nor reasoning or compaction
/// items, and a rollout over the budget is cut in the middle.
#[tokio::test]
async fn runner_memory_generation_sanitizes_and_truncates_phase_one_prompt() {
    let (client, session) = local_session().await;
    let phase_one = phase_one_model("raw memory entry\n");
    let memory = Memory::new().with_generate(Some(generate_config(
        phase_one.clone(),
        consolidating_model("memories", "entry", "summary"),
    )));
    let model = setup();
    model.add_turn(vec![
        OutputItem::from_value(json!({"id": "rs_1", "summary": [], "type": "reasoning"})),
        OutputItem::from_value(json!({
            "id": "compaction_1", "type": "compaction",
            "summary": "compacted-so-far", "encrypted_content": "encrypted",
        })),
        assistant_message("done"),
    ]);
    let agent = worker("worker", model, memory);
    let input = vec![
        InputItem::from_value(json!({"role": "developer", "content": "developer debug"})),
        InputItem::from_value(json!({"role": "system", "content": "system note"})),
        InputItem::from_value(json!({
            "role": "user",
            "content": format!("start{}middle{}end", "a".repeat(400_000), "z".repeat(400_000)),
        })),
        InputItem::from_value(json!({"type": "reasoning", "summary": []})),
        InputItem::from_value(json!({
            "type": "compaction", "summary": "input-compact", "encrypted_content": "encrypted",
        })),
    ];

    let (result, manager) = run_on_session(&agent, input, &session, None).await;
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(phase_one.calls().is_empty());

    flush_memory(&manager.expect("manager"), &session).await;

    let prompt = first_user_text(&phase_one);
    assert!(!prompt.contains("developer debug"));
    assert!(!prompt.contains("system note"));
    assert!(!prompt.contains("reasoning"));
    assert!(!prompt.contains("encrypted_content"));
    assert!(!prompt.contains("input-compact"));
    assert!(!prompt.contains("compacted-so-far"));
    assert!(prompt.contains("start"));
    assert!(!prompt.contains("middle"));
    assert!(prompt.contains("end"));
    assert!(prompt.contains("tokens truncated"));
    assert!(prompt.contains("rollout content omitted"));
    client.delete(session).await.expect("delete");
}

/// With a cap of one raw memory, phase two consolidates only the newest rollout and the prompt
/// says how the selection changed.
#[tokio::test]
async fn sandbox_memory_caps_phase_two_selection_and_surfaces_removed_rollouts() {
    let (client, session) = local_session().await;
    let phase_one = setup();
    phase_one.add_turn(vec![phase_one_message("first", "first raw\n")]);
    phase_one.add_turn(vec![phase_one_message("second", "second raw\n")]);
    let phase_two = consolidating_model("memories", "first entry", "first summary");
    let memory = Memory::new().with_generate(Some(
        generate_config(phase_one, phase_two.clone()).with_max_raw_memories_for_consolidation(1),
    ));
    let root = root_of(&session);
    let mut last_manager = None;
    for (rollout_id, input) in [("first-chat", "first"), ("second-chat", "second")] {
        let model = setup();
        model.add_turn(vec![assistant_message(format!("{input} done"))]);
        let agent = worker("worker", model, memory.clone());
        let (_, manager) = run_on_session(&agent, input, &session, Some(rollout_id)).await;
        // Both managers flush: each owns its own rollout.
        flush_memory(manager.as_ref().expect("manager"), &session).await;
        last_manager = manager;
    }
    drop(last_manager);

    assert_eq!(files_in(&root, "sessions", ".jsonl").len(), 2);
    let selection: Value =
        serde_json::from_str(&read(&root, "memories/phase_two_selection.json")).expect("json");
    let selected: Vec<String> = selection["selected"]
        .as_array()
        .expect("selected")
        .iter()
        .map(|item| item["rollout_id"].as_str().expect("id").to_string())
        .collect();
    assert_eq!(selected.len(), 1);
    let raw = read(&root, "memories/raw_memories.md");
    assert!(
        raw.contains("second raw") || raw.contains("first raw"),
        "{raw}"
    );
    assert!(
        !(raw.contains("second raw") && raw.contains("first raw")),
        "{raw}"
    );
    let calls = phase_two.calls();
    assert!(!calls.is_empty());
    let prompt = user_text(&calls[0].input[0]);
    assert!(
        prompt.contains("newly added since the last successful Phase 2 run: 1"),
        "{prompt}"
    );
    assert!(
        prompt.contains(&format!("rollout_id={}", selected[0])),
        "{prompt}"
    );
    client.delete(session).await.expect("delete");
}

/// The configured turn limit bounds the consolidation run. When it is exhausted the previous
/// selection is left as it was, and the memory file the first turn edited stays edited. A limit
/// that is large enough, and the default, complete both turns and record the selection.
#[tokio::test]
async fn sandbox_memory_consolidation_turn_budget_on_session_close() {
    for (label, turn_limit) in [
        ("exhausted", Some(1usize)),
        ("completed", Some(2)),
        ("default", None),
    ] {
        let (client, session) = local_session().await;
        let phase_two = consolidating_model("memories", "consolidated entry", "summary");
        let mut generate = generate_config(phase_one_model("raw\n"), phase_two.clone());
        if let Some(limit) = turn_limit {
            generate = generate.with_phase_two_max_turns(limit);
        }
        let memory = Memory::new().with_generate(Some(generate));
        let model = setup();
        model.add_turn(vec![assistant_message("done")]);
        let agent = worker("worker", model, memory);

        let (result, manager) = run_on_session(&agent, "hello", &session, None).await;
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("done"),
            "{label}"
        );
        let root = root_of(&session);
        let selection_path = root.join("memories/phase_two_selection.json");
        let previous_selection = "{\"selected\": []}\n";
        std::fs::write(&selection_path, previous_selection).expect("seed selection");

        flush_memory(&manager.expect("manager"), &session).await;

        let expected_calls = if turn_limit == Some(1) { 1 } else { 2 };
        assert_eq!(phase_two.calls().len(), expected_calls, "{label}");
        assert_eq!(
            read(&root, "memories/MEMORY.md"),
            "consolidated entry\n",
            "{label}"
        );
        let selection = read(&root, "memories/phase_two_selection.json");
        if turn_limit == Some(1) {
            assert_eq!(selection, previous_selection, "{label}");
        } else {
            let parsed: Value = serde_json::from_str(&selection).expect("json");
            assert_eq!(
                parsed["selected"].as_array().expect("selected").len(),
                1,
                "{label}"
            );
        }
        client.delete(session).await.expect("delete");
    }
}

/// Exhausting the phase-two turn budget in a runner-owned session still tears the workspace
/// down.
#[tokio::test]
async fn sandbox_memory_turn_exhaustion_releases_runner_owned_session() {
    let client = Arc::new(DeleteTrackingClient::default());
    let phase_two = consolidating_model("memories", "entry", "summary");
    let memory = Memory::new().with_generate(Some(
        generate_config(phase_one_model("raw\n"), phase_two.clone()).with_phase_two_max_turns(1),
    ));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory);

    let config = SandboxRunConfig::with_client(client.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(phase_two.calls().len(), 1);
    let deleted = client.deleted_roots.lock().expect("deleted").clone();
    assert_eq!(deleted.len(), 1);
    assert!(!deleted[0].exists());
}

/// A runner-owned session flushes on its own: the memory files exist before the workspace is
/// deleted, so extraction on close is not lost to the teardown.
#[tokio::test]
async fn sandbox_memory_runs_phase_one_and_phase_two_on_session_close() {
    let (client, session) = local_session().await;
    let phase_two = consolidating_model("memories", "shutdown entry", "shutdown summary");
    let memory =
        Memory::new().with_generate(Some(generate_config(phase_one_model("raw\n"), phase_two)));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory);

    let (_, manager) = run_on_session(&agent, "hello", &session, None).await;
    let root = root_of(&session);
    assert_eq!(
        read(&root, "memories/MEMORY.md"),
        "",
        "nothing is consolidated before close"
    );

    flush_memory(&manager.expect("manager"), &session).await;

    assert_eq!(read(&root, "memories/MEMORY.md"), "shutdown entry\n");
    assert_eq!(
        read(&root, "memories/memory_summary.md"),
        "shutdown summary\n"
    );
    client.delete(session).await.expect("delete");
}

/// Flushing is idempotent: a second flush after the manager stopped does nothing, so a session
/// closed twice does not run phase two twice.
#[tokio::test]
async fn sandbox_memory_flush_runs_once() {
    let (client, session) = local_session().await;
    let phase_one = phase_one_model("raw\n");
    let phase_two = consolidating_model("memories", "entry", "summary");
    let memory =
        Memory::new().with_generate(Some(generate_config(phase_one.clone(), phase_two.clone())));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory);
    let (_, manager) = run_on_session(&agent, "hello", &session, None).await;
    let manager = manager.expect("manager");

    flush_memory(&manager, &session).await;
    flush_memory(&manager, &session).await;

    assert_eq!(phase_one.calls().len(), 1);
    assert_eq!(
        phase_two.calls().len(),
        2,
        "one consolidation run is two model turns"
    );
    client.delete(session).await.expect("delete");
}

/// A rollout that cannot be written is reported and dropped, and the run itself still
/// completes and releases its workspace. Adapted: Python also asserts what the warning log
/// redacts under `DONT_LOG_MODEL_DATA`/`DONT_LOG_TOOL_DATA`, which has no Rust counterpart.
#[tokio::test]
async fn sandbox_memory_enqueue_failure_does_not_fail_the_run() {
    let client = Arc::new(DeleteTrackingClient::default());
    let memory = Memory::new().with_generate(Some(generate_config(
        phase_one_model("raw\n"),
        consolidating_model("memories", "entry", "summary"),
    )));
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = worker("worker", model, memory).with_default_manifest(
        // A file where the sessions directory has to go makes every enqueue fail.
        Manifest::new().with_entry("sessions", Entry::text_file("in the way")),
    );

    let config = SandboxRunConfig::with_client(client.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let deleted = client.deleted_roots.lock().expect("deleted").clone();
    assert_eq!(deleted.len(), 1);
    assert!(!deleted[0].exists());
}

/// A run that paused for approval is recorded as interrupted, and phase one is told so.
#[tokio::test]
async fn sandbox_memory_marks_interrupted_runs_in_phase_one_prompt() {
    let (client, session) = local_session().await;
    let phase_one = phase_one_model("raw\n");
    let memory = Memory::new()
        .with_read(Some(MemoryReadConfig::new()))
        .with_generate(Some(generate_config(
            phase_one.clone(),
            consolidating_model("memories", "interrupted entry", "interrupted summary"),
        )));
    let model = setup();
    model.add_turn(vec![shell_call("approval-call", "echo hi")]);
    let agent = SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.")
        .with_capabilities(vec![
            Arc::new(Filesystem::new()),
            Arc::new(Shell::new().with_pty(false).with_needs_approval(true)),
            Arc::new(memory),
        ]);

    let (result, manager) = run_on_session(&agent, "interrupt me", &session, None).await;
    assert!(!result.interruptions.is_empty());

    flush_memory(&manager.expect("manager"), &session).await;

    assert!(
        first_user_text(&phase_one).contains("\"terminal_state\":\"interrupted\""),
        "{}",
        first_user_text(&phase_one)
    );
    client.delete(session).await.expect("delete");
}
