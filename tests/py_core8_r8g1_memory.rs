//! Ports of the sandbox memory rollout-grouping rows of `tests/sandbox/test_memory.py`: which
//! file a run's rollout segment lands in when the caller names no rollout id. The id comes from
//! the SDK session, else the conversation id, else the trace group id, else one per run.
//!
//! The earlier port of this file's runner tests names the rollout through
//! `SandboxRunConfig::rollout_id`; these go through the run's own session, conversation and
//! group, the way Python's tests do.
#![cfg(all(feature = "sandbox", feature = "sqlite"))]

use std::path::PathBuf;
use std::sync::Arc;

use openai_agents_sdk::memory::{Session, SqliteSession};
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Memory, Shell};
use openai_agents_sdk::sandbox::memory::MemoryGenerateConfig;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    Manifest, SandboxAgent, SandboxClient, SandboxRunConfig, SandboxRunner, SandboxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{AgentModel, RunConfig, RunOptions};
use serde_json::Value;

fn model_with(replies: &[&str]) -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    for reply in replies {
        model.add_turn(vec![assistant_message(*reply)]);
    }
    model
}

fn memory_config() -> Memory {
    // The memory models are never called: nothing here closes the session.
    Memory::new().with_generate(Some(
        MemoryGenerateConfig::new()
            .with_phase_one_model(AgentModel::Instance(model_with(&[])))
            .with_phase_two_model(AgentModel::Instance(model_with(&[]))),
    ))
}

fn worker(model: Arc<FakeModel>) -> SandboxAgent<()> {
    worker_with(model, memory_config())
}

fn worker_with(model: Arc<FakeModel>, memory: Memory) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("worker")
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

fn rollouts(session: &Arc<dyn SandboxSession>) -> Vec<String> {
    let dir = PathBuf::from(session.workspace_root()).join("sessions");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|name| name.ends_with(".jsonl"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn segments(session: &Arc<dyn SandboxSession>, name: &str) -> Vec<Value> {
    let path = PathBuf::from(session.workspace_root())
        .join("sessions")
        .join(name);
    std::fs::read_to_string(path)
        .expect("rollout file")
        .lines()
        .map(|line| serde_json::from_str(line).expect("segment"))
        .collect()
}

async fn run(
    agent: &SandboxAgent<()>,
    input: &str,
    config: &SandboxRunConfig,
    options: RunOptions<()>,
) -> String {
    let outcome = SandboxRunner::run_to_outcome(agent, input, config, options)
        .await
        .expect("run");
    outcome.result.final_output_text().expect("final output")
}

fn chat_session(id: &str) -> Arc<dyn Session> {
    Arc::new(SqliteSession::in_memory(id).expect("in-memory session"))
}

/// Python: `test_sandbox_memory_groups_segments_by_sdk_session_until_close`. Two agents sharing
/// one SDK session write to one rollout file named after it, one segment per run.
#[tokio::test]
async fn segments_are_grouped_by_the_sdk_session() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let chat = chat_session("chat-session");
    // Agents that share a layout share one memory configuration.
    let memory = memory_config();
    let first = worker_with(model_with(&["first done"]), memory.clone());
    let second = worker_with(model_with(&["second done"]), memory);

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat.clone());
    assert_eq!(run(&first, "first", &config, options).await, "first done");
    let mut options = RunOptions::<()>::default();
    options.session = Some(chat);
    assert_eq!(
        run(&second, "second", &config, options).await,
        "second done"
    );

    assert_eq!(rollouts(&session), vec!["chat-session.jsonl"]);
    let written = segments(&session, "chat-session.jsonl");
    assert_eq!(written.len(), 2);
    let keys: Vec<&str> = written[0]
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        &keys[..4],
        &["updated_at", "rollout_id", "input", "generated_items"]
    );
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_fallback_does_not_mutate_run_config`. One run configuration
/// reused with two different SDK sessions yields one rollout per session: the grouping is chosen
/// per run and never written back into the shared configuration.
#[tokio::test]
async fn the_grouping_fallback_does_not_mutate_the_shared_run_config() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let agent = worker(model_with(&["first done", "second done"]));

    for (input, id) in [("first", "first-chat"), ("second", "second-chat")] {
        let mut options = RunOptions::<()>::default();
        options.session = Some(chat_session(id));
        run(&agent, input, &config, options).await;
    }

    assert_eq!(
        rollouts(&session),
        vec!["first-chat.jsonl", "second-chat.jsonl"]
    );
    assert!(config.rollout_id.is_none());
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_uses_conversation_id_when_sdk_session_is_absent`.
#[tokio::test]
async fn the_conversation_id_names_the_rollout_when_there_is_no_session() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let agent = worker(model_with(&["done"]));

    let mut options = RunOptions::<()>::default();
    options.conversation_id = Some("conversation-123".to_string());
    let output = run(&agent, "remember this conversation", &config, options).await;

    assert_eq!(output, "done");
    assert_eq!(rollouts(&session), vec!["conversation-123.jsonl"]);
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_uses_group_id_when_sdk_session_is_absent`. With neither a
/// session nor a conversation id, the trace group id names the rollout and two runs share it.
#[tokio::test]
async fn the_group_id_names_the_rollout_when_there_is_no_session_or_conversation() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let agent = worker(model_with(&["first done", "second done"]));

    for (input, expected) in [("first", "first done"), ("second", "second done")] {
        let mut options = RunOptions::<()>::default();
        options.run_config = RunConfig::default().with_group_id("trace-thread-123");
        assert_eq!(run(&agent, input, &config, options).await, expected);
    }

    assert_eq!(rollouts(&session), vec!["trace-thread-123.jsonl"]);
    assert_eq!(segments(&session, "trace-thread-123.jsonl").len(), 2);
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_uses_per_run_conversation_when_no_conversation_id`. A run with no
/// session, conversation or group gets its own `run-` rollout, so two such runs do not merge.
#[tokio::test]
async fn a_run_without_any_grouping_gets_its_own_rollout() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let agent = worker(model_with(&["first done", "second done"]));

    for (input, expected) in [("first", "first done"), ("second", "second done")] {
        let output = run(&agent, input, &config, RunOptions::<()>::default()).await;
        assert_eq!(output, expected);
    }

    let names = rollouts(&session);
    assert_eq!(names.len(), 2, "{names:?}");
    assert!(
        names
            .iter()
            .all(|name| name.starts_with("run-") && name.ends_with(".jsonl")),
        "{names:?}"
    );
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_input_uses_saved_session_delta_after_persistence`. When the SDK
/// session already holds an earlier turn, the rollout segment records only this run's input, not
/// the replayed history.
#[tokio::test]
async fn the_segment_records_only_the_current_turn_when_the_session_has_history() {
    use openai_agents_sdk::items::InputItem;
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let chat = chat_session("history-chat");
    chat.add_items(vec![InputItem(
        serde_json::json!({"role": "user", "content": "old turn"}),
    )])
    .await
    .expect("seed history");
    let agent = worker(model_with(&["done"]));

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat);
    run(&agent, "current turn", &config, options).await;

    let written = segments(&session, "history-chat.jsonl");
    assert_eq!(written.len(), 1);
    assert_eq!(
        written[0]["input"],
        serde_json::json!([{"role": "user", "content": "current turn"}])
    );
    client.delete(session).await.expect("delete");
}
