//! Ports of the run-state resume cases of `tests/sandbox/test_runtime.py`: a paused sandbox run
//! writes a per-agent sandbox section into its run state, and a resumed run restores every
//! agent's session from it (not only the starting agent's through
//! `SandboxRunConfig::session_state`).
//!
//! Python's tests run each agent in its own session; the Rust equivalent is
//! `SandboxSessionMode::PerAgent`, so the multi-agent cases use it.

#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use openai_agents_sdk::RunOptions;
use openai_agents_sdk::items::{OutputItem, RunInput};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Shell};
use openai_agents_sdk::sandbox::runtime::SandboxSessionMode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::SandboxClient;
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::{Value, json};

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn call(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn handoff_call(tool: &str, call_id: &str) -> OutputItem {
    call(json!({"type": "function_call", "call_id": call_id, "name": tool, "arguments": "{}"}))
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    call(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

fn write_note(call_id: &str, text: &str) -> OutputItem {
    call(json!({
        "type": "apply_patch_call", "call_id": call_id, "status": "completed",
        "operation": {"type": "create_file", "path": "note.txt", "diff": format!("+{text}\n")}
    }))
}

/// The stdout of every shell call a result holds.
fn shell_stdouts(result: &RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::ToolCallOutput { raw_item, .. } => {
                raw_item["output"][0]["stdout"].as_str().map(str::to_string)
            }
            _ => None,
        })
        .collect()
}

fn config(snapshots: &tempfile::TempDir, mode: SandboxSessionMode) -> SandboxRunConfig {
    let client: Arc<dyn SandboxClient> = Arc::new(UnixLocalSandboxClient::new().with_snapshot(
        SnapshotSpec::Local {
            base_path: snapshots.path().to_path_buf(),
        },
    ));
    SandboxRunConfig::with_client(client).session_mode(mode)
}

fn options() -> RunOptions<()> {
    RunOptions::default().max_turns(Some(12))
}

fn files_and_shell() -> Vec<Arc<dyn openai_agents_sdk::sandbox::capabilities::Capability<()>>> {
    vec![
        Arc::new(Filesystem::new().with_native_apply_patch()),
        Arc::new(Shell::new().with_pty(false)),
    ]
}

/// A worker whose shell needs approval, handing back to `triage` by name.
fn worker(model: Arc<FakeModel>) -> Arc<SandboxAgent<()>> {
    Arc::new(
        SandboxAgent::<()>::new("worker")
            .with_model_instance(model)
            .with_capabilities(vec![Arc::new(
                Shell::new().with_pty(false).with_needs_approval(true),
            )])
            .with_sandbox_handoff_to("triage"),
    )
}

fn triage(model: Arc<FakeModel>, worker: Arc<SandboxAgent<()>>) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("triage")
        .with_model_instance(model)
        .with_capabilities(files_and_shell())
        .with_sandbox_handoff(worker)
}

/// Run triage to the worker's pending approval; returns the run state as JSON text.
async fn paused_triage_run(snapshots: &tempfile::TempDir, mode: SandboxSessionMode) -> String {
    let (triage_model, worker_model) = (model(), model());
    triage_model.add_turn(vec![write_note("c_write", "persist triage")]);
    triage_model.add_turn(vec![handoff_call("transfer_to_worker", "c_handoff")]);
    worker_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let agent = triage(triage_model, worker(worker_model));
    let outcome =
        SandboxRunner::run_to_outcome(&agent, "hello", &config(snapshots, mode), options())
            .await
            .expect("the run pauses");
    assert_eq!(outcome.result.interruptions.len(), 1);
    outcome
        .result
        .to_state()
        .to_json_string()
        .expect("the state serializes")
}

/// Resume `state_json` with the worker approved; the worker hands back to triage, which reads
/// the note in its own session.
async fn resume_and_read(
    snapshots: &tempfile::TempDir,
    mode: SandboxSessionMode,
    mut state: RunState,
) -> RunResult<()> {
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    let (triage_model, worker_model) = (model(), model());
    worker_model.add_turn(vec![handoff_call("transfer_to_triage", "c_back")]);
    triage_model.add_turn(vec![shell_call("c_read", "cat note.txt")]);
    triage_model.add_turn(vec![assistant_message("done")]);
    let agent = triage(triage_model, worker(worker_model));
    SandboxRunner::run_with_options(&agent, state, &config(snapshots, mode), options())
        .await
        .expect("the run resumes")
}

/// Python: `test_runner_restores_all_sandbox_agents_from_run_state_across_handoffs`.
///
/// Both agents' sessions travel in the run state: after a restore from JSON alone, the agent
/// that wrote the note in its own workspace reads it back.
#[tokio::test]
async fn a_resumed_run_restores_every_agents_session_from_the_run_state() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::PerAgent).await;

    let state = RunState::from_json_str(&json).expect("the state restores");
    let section = state
        .sandbox
        .clone()
        .expect("the run state has a sandbox section");
    assert_eq!(section["backend_id"], "unix_local");
    assert_eq!(section["current_agent_name"], "worker");
    assert_eq!(section["current_agent_key"], "worker");
    let sessions = section["sessions_by_agent"].as_object().unwrap();
    assert_eq!(
        sessions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["triage", "worker"]
    );
    assert_eq!(sessions["worker"]["agent_name"], "worker");
    assert_eq!(
        section["session_state"],
        sessions["worker"]["session_state"]
    );

    let resumed = resume_and_read(&snapshots, SandboxSessionMode::PerAgent, state).await;
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(
        shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("persist triage")),
        "{:?}",
        shell_stdouts(&resumed)
    );
}

/// The same run without its sandbox section gets a fresh session for triage, which holds no
/// note: the restore above is what carried it.
#[tokio::test]
async fn without_the_section_the_other_agents_get_fresh_sessions() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::PerAgent).await;

    let mut state = RunState::from_json_str(&json).unwrap();
    state.sandbox = None;
    let resumed = resume_and_read(&snapshots, SandboxSessionMode::PerAgent, state).await;

    assert!(
        !shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("persist triage")),
        "{:?}",
        shell_stdouts(&resumed)
    );
}

/// Python: `test_runner_persists_workspace_and_tool_choice_state_across_sandbox_resume` and
/// `test_runner_restores_sandbox_from_run_state`.
///
/// One agent, resumed from the run state alone (no `SandboxRunConfig::session_state`).
#[tokio::test]
async fn a_single_agent_resumes_from_the_section_alone() {
    let snapshots = tempfile::tempdir().unwrap();
    let agent_model = model();
    // The resumed agent's shell needs no approval, so its later read runs straight through.
    let build = |model: Arc<FakeModel>, approval: bool| {
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(model)
            .with_instructions("Base instructions.")
            .with_capabilities(vec![
                Arc::new(Filesystem::new().with_native_apply_patch()),
                Arc::new(Shell::new().with_pty(false).with_needs_approval(approval)),
            ])
            .map_agent(|agent| {
                agent.with_model_settings(ModelSettings {
                    tool_choice: Some(ToolChoice::required()),
                    ..Default::default()
                })
            })
    };
    agent_model.add_turn(vec![write_note("c_write", "persist me")]);
    agent_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let first = SandboxRunner::run_to_outcome(
        &build(agent_model.clone(), true),
        "hello",
        &config(&snapshots, SandboxSessionMode::SharedGraph),
        options(),
    )
    .await
    .unwrap();
    assert_eq!(first.result.interruptions.len(), 1);

    let state = first.result.to_state();
    let section = state
        .sandbox
        .clone()
        .expect("a paused run writes the section");
    assert_eq!(section["backend_id"], "unix_local");
    assert_eq!(section["current_agent_name"], "sandbox");
    assert_eq!(section["current_agent_key"], "sandbox");
    let session_state = section["session_state"].as_object().unwrap();
    assert_eq!(session_state["snapshot"]["type"], "local");
    assert_eq!(
        section["sessions_by_agent"]["sandbox"],
        json!({"agent_name": "sandbox", "session_state": session_state})
    );

    let mut restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let pending = restored.get_interruptions()[0].clone();
    restored.approve(&pending, false);
    let resumed_model = model();
    resumed_model.add_turn(vec![shell_call("c_read", "cat note.txt")]);
    resumed_model.add_turn(vec![assistant_message("done")]);
    let resumed = SandboxRunner::run_with_options(
        &build(resumed_model.clone(), false),
        restored,
        &config(&snapshots, SandboxSessionMode::SharedGraph),
        options(),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(
        shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("persist me"))
    );
    assert_eq!(
        resumed_model
            .last_call()
            .unwrap()
            .model_settings
            .tool_choice,
        None,
        "a forced tool choice is dropped once a tool ran"
    );
}

/// Python: `test_runner_serializes_unique_sandbox_resume_keys_for_duplicate_agent_names` and
/// `test_runner_restores_duplicate_name_sandbox_sessions_after_json_roundtrip`.
///
/// Two agents share the name `sandbox`: the section files them under `sandbox` and `sandbox#2`,
/// and a restore from JSON gives each its own session back.
#[tokio::test]
async fn same_named_agents_resume_their_own_sessions() {
    let snapshots = tempfile::tempdir().unwrap();
    let build = |first_model: Arc<FakeModel>, second_model: Arc<FakeModel>| {
        let second = Arc::new(
            SandboxAgent::<()>::new("sandbox")
                .with_model_instance(second_model)
                .with_instructions("Second instructions.")
                .with_capabilities(vec![Arc::new(
                    Shell::new().with_pty(false).with_needs_approval(true),
                )])
                .with_sandbox_handoff_to("sandbox"),
        );
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(first_model)
            .with_instructions("First instructions.")
            .with_capabilities(files_and_shell())
            .with_sandbox_handoff(second)
    };
    let (first_model, second_model) = (model(), model());
    first_model.add_turn(vec![write_note("c_write", "first")]);
    first_model.add_turn(vec![handoff_call("transfer_to_sandbox", "to_second")]);
    second_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let outcome = SandboxRunner::run_to_outcome(
        &build(first_model, second_model),
        "hello",
        &config(&snapshots, SandboxSessionMode::PerAgent),
        options(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.result.interruptions.len(), 1);

    let json = outcome.result.to_state().to_json_string().unwrap();
    let mut state = RunState::from_json_str(&json).unwrap();
    let section = state
        .sandbox
        .clone()
        .expect("the section survives the JSON round trip");
    let sessions = section["sessions_by_agent"].as_object().unwrap();
    assert_eq!(
        sessions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["sandbox", "sandbox#2"]
    );
    assert_eq!(section["current_agent_key"], "sandbox#2");
    assert!(sessions.contains_key(section["current_agent_key"].as_str().unwrap()));
    assert_ne!(
        sessions["sandbox"]["session_state"]["session_id"],
        sessions["sandbox#2"]["session_state"]["session_id"]
    );

    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    let (first_model, second_model) = (model(), model());
    second_model.add_turn(vec![handoff_call("transfer_to_sandbox", "to_first")]);
    first_model.add_turn(vec![shell_call("c_read", "cat note.txt")]);
    first_model.add_turn(vec![assistant_message("done")]);
    let resumed = SandboxRunner::run_with_options(
        &build(first_model, second_model),
        state,
        &config(&snapshots, SandboxSessionMode::PerAgent),
        options(),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(
        shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("first"))
    );
}

/// Python: `test_runner_restores_legacy_current_sandbox_payload_after_json_roundtrip`.
///
/// A section with no agent keys (only the current agent's `session_state`, sessions filed under
/// an agent id) still resumes the agent the saved run stopped at.
#[tokio::test]
async fn a_legacy_section_without_agent_keys_resumes_the_current_agent() {
    let snapshots = tempfile::tempdir().unwrap();
    let agent_model = model();
    let build = |model: Arc<FakeModel>, approval: bool| {
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(model)
            .with_capabilities(vec![
                Arc::new(Filesystem::new().with_native_apply_patch()),
                Arc::new(Shell::new().with_pty(false).with_needs_approval(approval)),
            ])
    };
    agent_model.add_turn(vec![write_note("c_write", "legacy")]);
    agent_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let first = SandboxRunner::run_to_outcome(
        &build(agent_model, true),
        "hello",
        &config(&snapshots, SandboxSessionMode::SharedGraph),
        options(),
    )
    .await
    .unwrap();
    let mut state = first.result.to_state();
    let session_state = state.sandbox.as_ref().unwrap()["session_state"].clone();
    state.sandbox = Some(json!({
        "backend_id": "unix_local",
        "current_agent_id": 140_234_001,
        "session_state": session_state,
        "sessions_by_agent": {"140234001": session_state},
    }));

    let mut restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let pending = restored.get_interruptions()[0].clone();
    restored.approve(&pending, false);
    let resumed_model = model();
    resumed_model.add_turn(vec![shell_call("c_read", "cat note.txt")]);
    resumed_model.add_turn(vec![assistant_message("done")]);
    let resumed = SandboxRunner::run_with_options(
        &build(resumed_model, false),
        restored,
        &config(&snapshots, SandboxSessionMode::SharedGraph),
        options(),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(
        shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("legacy"))
    );
}

/// A section written for another backend is refused rather than resumed against the wrong
/// client.
#[tokio::test]
async fn a_section_for_another_backend_is_refused() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::PerAgent).await;
    let mut state = RunState::from_json_str(&json).unwrap();
    state.sandbox.as_mut().unwrap()["backend_id"] = json!("docker");
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);

    let (triage_model, worker_model) = (model(), model());
    let agent = triage(triage_model, worker(worker_model));
    let error = SandboxRunner::run_with_options(
        &agent,
        RunInput::State(Box::new(state)),
        &config(&snapshots, SandboxSessionMode::PerAgent),
        options(),
    )
    .await
    .expect_err("the backend mismatch is refused");
    assert!(
        error
            .to_string()
            .contains("RunState sandbox backend does not match the configured sandbox client"),
        "{error}"
    );
}

/// Python: `test_session_manager_preserves_existing_payload_when_no_sandbox_session_is_used`:
/// a resumed run that finishes leaves the section it was resumed from in its state.
#[tokio::test]
async fn a_finished_resumed_run_keeps_the_section_it_was_resumed_from() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::PerAgent).await;
    let state = RunState::from_json_str(&json).unwrap();
    let before = state.sandbox.clone().unwrap();

    let resumed = resume_and_read(&snapshots, SandboxSessionMode::PerAgent, state).await;

    assert_eq!(resumed.to_state().sandbox, Some(before));
}

/// Python: `test_session_manager_omits_existing_payload_for_injected_live_session`: a session
/// the caller supplies has no state to resume from, so the run state carries no section.
#[tokio::test]
async fn an_injected_live_session_omits_the_section() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::SharedGraph).await;
    let mut state = RunState::from_json_str(&json).unwrap();
    assert!(state.sandbox.is_some());
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);

    let client = UnixLocalSandboxClient::new();
    let live = client.create(None).await.unwrap();
    live.start().await.unwrap();
    let (triage_model, worker_model) = (model(), model());
    worker_model.add_turn(vec![handoff_call("transfer_to_triage", "c_back")]);
    triage_model.add_turn(vec![assistant_message("done")]);
    let agent = triage(triage_model, worker(worker_model));
    let resumed = SandboxRunner::run_with_options(
        &agent,
        state,
        &SandboxRunConfig::with_session(live.clone()),
        options(),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(resumed.to_state().sandbox, None);
    client.delete(live).await.unwrap();
}

/// Python: `test_session_manager_preserves_untouched_run_state_sessions_on_cleanup`.
///
/// Resuming a per-agent section in a shared-graph run restores only the starting agent's session;
/// when the run pauses again the worker's saved session is written back untouched.
#[tokio::test]
async fn sessions_a_run_did_not_touch_are_written_back_unchanged() {
    let snapshots = tempfile::tempdir().unwrap();
    let json = paused_triage_run(&snapshots, SandboxSessionMode::PerAgent).await;
    let mut state = RunState::from_json_str(&json).unwrap();
    let before = state.sandbox.clone().unwrap();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);

    // The approved shell ran; the worker asks for another command that needs approval.
    let (triage_model, worker_model) = (model(), model());
    worker_model.add_turn(vec![shell_call("c_second", "echo again")]);
    let agent = triage(triage_model, worker(worker_model));
    let outcome = SandboxRunner::run_to_outcome(
        &agent,
        state,
        &config(&snapshots, SandboxSessionMode::SharedGraph),
        options(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.result.interruptions.len(), 1);

    let after = outcome
        .result
        .to_state()
        .sandbox
        .expect("the run paused again");
    let sessions = after["sessions_by_agent"].as_object().unwrap();
    assert_eq!(
        sessions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["triage", "worker"]
    );
    assert_eq!(
        sessions["worker"], before["sessions_by_agent"]["worker"],
        "the worker's session was not used, so its entry is carried as it was"
    );
    assert_eq!(sessions["triage"]["agent_name"], "triage");
}

/// A streamed run that pauses writes the same section: `SandboxStreamedRun::to_state` carries it
/// and a run resumed from that state restores both agents' sessions.
#[tokio::test]
async fn a_paused_streamed_run_writes_the_section_too() {
    let snapshots = tempfile::tempdir().unwrap();
    let (triage_model, worker_model) = (model(), model());
    triage_model.add_turn(vec![write_note("c_write", "streamed note")]);
    triage_model.add_turn(vec![handoff_call("transfer_to_worker", "c_handoff")]);
    worker_model.add_turn(vec![shell_call("c_approval", "echo approved")]);
    let agent = triage(triage_model, worker(worker_model));
    let mut run = SandboxRunner::run_streamed_with_options(
        &agent,
        "hello",
        &config(&snapshots, SandboxSessionMode::PerAgent),
        options(),
    )
    .await
    .unwrap();
    run.finish().await.unwrap();

    let state = run.to_state().expect("the stream finished");
    let section = state
        .sandbox
        .clone()
        .expect("a paused stream writes the section");
    assert_eq!(section["current_agent_key"], "worker");
    assert_eq!(
        section["sessions_by_agent"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["triage", "worker"]
    );

    let state = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let resumed = resume_and_read(&snapshots, SandboxSessionMode::PerAgent, state).await;
    assert!(
        shell_stdouts(&resumed)
            .iter()
            .any(|out| out.contains("streamed note"))
    );
}
