//! Ports of the remaining `tests/sandbox/test_memory.py` cases.
//!
//! Python unit-tests two private helpers, `_sandbox_memory_input` and
//! `_stream_memory_input_override`, which decide what `input` a run's memory rollout segment
//! records. Their observable effect is the segment written into `sessions/<rollout>.jsonl`, so
//! these tests drive real sandbox runs and read the segments back:
//!
//! - a run resumed from a `RunState` with an SDK session records an empty `input`: the original
//!   turn was recorded by the run that paused, and the resumed run persisted no new input;
//! - a streamed run whose conversation is managed by the server (`conversation_id`,
//!   `previous_response_id` or `auto_previous_response_id`) records the turn's own input.
//!
//! The model-settings cases go through `ModelSettings::from_value`, the Rust form of Python's
//! dictionary `model_settings`, into `MemoryGenerateConfig`.
#![cfg(all(feature = "sandbox", feature = "sqlite", unix))]

use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::memory::{Session, SqliteSession};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Memory, Shell};
use openai_agents_sdk::sandbox::memory::MemoryGenerateConfig;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    Manifest, SandboxAgent, SandboxClient, SandboxRunConfig, SandboxRunner, SandboxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::{AgentModel, AgentsError, FunctionTool, RunOptions, Tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

fn model_with(turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    for turn in turns {
        model.add_turn(turn);
    }
    model
}

fn memory() -> Memory {
    // Nothing here closes the session, so the phase models are never called.
    Memory::new().with_generate(Some(
        MemoryGenerateConfig::new()
            .with_phase_one_model(AgentModel::Instance(model_with(vec![])))
            .with_phase_two_model(AgentModel::Instance(model_with(vec![]))),
    ))
}

fn worker(model: Arc<FakeModel>) -> SandboxAgent<()> {
    let approval = FunctionTool::<()>::new(
        "approval_tool",
        "needs approval",
        |_ctx, _args: NoArgs| async move { Ok::<_, AgentsError>("approved".to_string()) },
    )
    .with_needs_approval(true);
    SandboxAgent::<()>::new("worker")
        .with_model_instance(model)
        .with_instructions("Worker.")
        .with_tool(Tool::Function(Arc::new(approval)))
        .with_capabilities(vec![
            Arc::new(Filesystem::new()),
            Arc::new(Shell::new().with_pty(false)),
            Arc::new(memory()),
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

fn segments(session: &Arc<dyn SandboxSession>, name: &str) -> Vec<Value> {
    let path = PathBuf::from(session.workspace_root())
        .join("sessions")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("rollout file {}: {e}", path.display()))
        .lines()
        .map(|line| serde_json::from_str(line).expect("segment"))
        .collect()
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

fn chat_session(id: &str) -> Arc<dyn Session> {
    Arc::new(SqliteSession::in_memory(id).expect("in-memory session"))
}

fn paused_then_done() -> Arc<FakeModel> {
    model_with(vec![
        vec![function_call_with_id("approval_tool", "{}", "call-approve")],
        vec![assistant_message("done")],
    ])
}

/// Python: `test_sandbox_memory_input_preserves_empty_session_delta`. A run resumed from its
/// `RunState` with an SDK session persists no new input, so its rollout segment records an empty
/// `input` instead of replaying the turn the paused run already recorded.
#[tokio::test]
async fn a_resumed_run_with_a_session_records_an_empty_input() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let chat = chat_session("resume-chat");
    let agent = worker(paused_then_done());

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat.clone());
    let paused = SandboxRunner::run_to_outcome(&agent, "old turn", &config, options)
        .await
        .expect("the run pauses");
    assert_eq!(paused.result.interruptions.len(), 1);
    let mut state = paused.result.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat);
    let resumed =
        SandboxRunner::run_to_outcome(&agent, RunInput::State(Box::new(state)), &config, options)
            .await
            .expect("the run resumes");
    assert_eq!(resumed.result.final_output_text().as_deref(), Some("done"));

    let written = segments(&session, "resume-chat.jsonl");
    assert_eq!(written.len(), 2, "{written:#?}");
    assert_eq!(
        written[0]["input"],
        json!([{"role": "user", "content": "old turn"}])
    );
    assert_eq!(written[1]["input"], json!([]), "{:#?}", written[1]);
    client.delete(session).await.expect("delete");
}

/// Python: `test_streaming_memory_payload_preserves_empty_input_override`. The streamed form of
/// the resumed run above records an empty `input` too.
#[tokio::test]
async fn a_streamed_resumed_run_with_a_session_records_an_empty_input() {
    let (client, session) = local_session().await;
    let config = SandboxRunConfig::with_session(session.clone());
    let chat = chat_session("stream-resume-chat");
    let agent = worker(paused_then_done());

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat.clone());
    let mut paused = SandboxRunner::run_streamed_with_options(&agent, "old turn", &config, options)
        .await
        .expect("the stream starts");
    {
        let stream = paused.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    }
    paused.finish().await.expect("the paused stream finishes");
    let mut state = paused.to_state().expect("a paused stream has a state");
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);

    let mut options = RunOptions::<()>::default();
    options.session = Some(chat);
    let mut resumed = SandboxRunner::run_streamed_with_options(
        &agent,
        RunInput::State(Box::new(state)),
        &config,
        options,
    )
    .await
    .expect("the stream resumes");
    {
        let stream = resumed.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    }
    resumed.finish().await.expect("the resumed stream finishes");
    assert_eq!(
        resumed.streaming.final_output_text().as_deref(),
        Some("done")
    );

    let written = segments(&session, "stream-resume-chat.jsonl");
    assert_eq!(written.len(), 2, "{written:#?}");
    assert_eq!(
        written[0]["input"],
        json!([{"role": "user", "content": "old turn"}])
    );
    assert_eq!(written[1]["input"], json!([]), "{:#?}", written[1]);
    client.delete(session).await.expect("delete");
}

/// Python: `test_streaming_memory_payload_uses_result_input_for_server_managed_conversation`
/// (`conversation_id`, `previous_response_id`, `auto_previous_response_id`). With the
/// conversation held by the server there is no local session delta, and the streamed segment
/// records the turn's own input rather than an empty list.
#[tokio::test]
async fn a_streamed_server_managed_run_records_the_turn_input() {
    for case in [
        "conversation_id",
        "previous_response_id",
        "auto_previous_response_id",
    ] {
        let (client, session) = local_session().await;
        let config = SandboxRunConfig::with_session(session.clone()).rollout_id("server-managed");
        let agent = worker(model_with(vec![vec![assistant_message("done")]]));

        let mut options = RunOptions::<()>::default();
        match case {
            "conversation_id" => options.conversation_id = Some("conversation-123".into()),
            "previous_response_id" => options.previous_response_id = Some("resp_123".into()),
            _ => options.auto_previous_response_id = true,
        }
        let mut run =
            SandboxRunner::run_streamed_with_options(&agent, "current turn", &config, options)
                .await
                .expect("the stream starts");
        {
            let stream = run.streaming.stream_events();
            futures::pin_mut!(stream);
            while stream.next().await.is_some() {}
        }
        run.finish().await.expect("the stream finishes");
        assert_eq!(
            run.streaming.final_output_text().as_deref(),
            Some("done"),
            "{case}"
        );

        assert_eq!(rollouts(&session), vec!["server-managed.jsonl"], "{case}");
        let written = segments(&session, "server-managed.jsonl");
        assert_eq!(written.len(), 1, "{case}: {written:#?}");
        assert_eq!(
            written[0]["input"],
            json!([{"role": "user", "content": "current turn"}]),
            "{case}"
        );
        client.delete(session).await.expect("delete");
    }
}

/// Python: `test_memory_generate_config_normalizes_dictionary_model_settings`. Settings that
/// arrive as JSON objects normalize into typed settings, nested `reasoning` and `retry` included.
#[test]
fn dictionary_model_settings_normalize_into_typed_settings() {
    let phase_one = ModelSettings::from_value(json!({
        "reasoning": {"effort": "low"},
        "retry": {"max_retries": 0},
    }))
    .expect("phase-one settings");
    let phase_two = ModelSettings::from_value(json!({"temperature": 0.0, "store": false}))
        .expect("phase-two settings");

    let config = MemoryGenerateConfig::new()
        .with_phase_one_model_settings(Some(phase_one))
        .with_phase_two_model_settings(Some(phase_two));

    let phase_one = config.phase_one_model_settings.as_ref().expect("phase one");
    let reasoning = phase_one.reasoning.as_ref().expect("reasoning");
    assert_eq!(reasoning.effort.as_deref(), Some("low"));
    let retry = phase_one.retry.as_ref().expect("retry");
    assert_eq!(retry.max_retries, Some(0));
    let phase_two = config.phase_two_model_settings.as_ref().expect("phase two");
    assert_eq!(phase_two.temperature, Some(0.0));
    assert_eq!(phase_two.store, Some(false));
    config.validate().expect("a valid configuration");
}

/// Python: `test_memory_generate_config_preserves_forward_compatible_reasoning_settings`
/// (`phase_one_model_settings`, `phase_two_model_settings`). A reasoning option this SDK does
/// not model is kept, for either phase.
#[test]
fn forward_compatible_reasoning_options_are_kept_for_either_phase() {
    for phase in ["phase_one", "phase_two"] {
        let settings =
            ModelSettings::from_value(json!({"reasoning": {"future_reasoning_option": "enabled"}}))
                .expect("settings");
        let config = if phase == "phase_one" {
            MemoryGenerateConfig::new().with_phase_one_model_settings(Some(settings))
        } else {
            MemoryGenerateConfig::new().with_phase_two_model_settings(Some(settings))
        };
        let configured = if phase == "phase_one" {
            config.phase_one_model_settings
        } else {
            config.phase_two_model_settings
        }
        .expect("settings are set");
        let reasoning = configured.reasoning.expect("reasoning");
        assert_eq!(
            Value::Object(reasoning.extra),
            json!({"future_reasoning_option": "enabled"}),
            "{phase}"
        );
    }
}

/// Python: `test_memory_generate_config_rejects_invalid_model_settings` (both phases). Settings
/// that are neither typed settings nor an object are refused before they reach the config.
#[test]
fn model_settings_that_are_not_an_object_are_refused() {
    for invalid in [json!("invalid"), json!(3), json!(["reasoning"])] {
        let error = ModelSettings::from_value(invalid.clone()).expect_err("not an object");
        assert!(
            error
                .to_string()
                .contains("must be a ModelSettings instance or a dict"),
            "{invalid}: {error}"
        );
    }
}
