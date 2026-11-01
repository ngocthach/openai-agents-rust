//! Rust counterpart of the released-surface cases of `tests/test_released_api_contract.py`.
//!
//! The Python file mostly tests the repository's own contract-validation tooling (`inspect`,
//! `typing` and `pydantic` introspection of the Python package, dependency-installation policy
//! JSON, platform import errors). Those groups have no Rust counterpart. What does carry over is
//! the released surface itself: the members the contract fixture freezes for the runner, the
//! result and state types, the retry context and the testing doubles. Rust pins them at compile
//! time: each coercion below fixes the name and signature, so removing or changing one breaks the
//! build the way the Python validator rejects a changed released callable.

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::result::{RunResult, RunResultStreaming};
use openai_agents_sdk::retry::{ReplaySafety, RetryPolicyContext};
use openai_agents_sdk::testing::{ModelCall, ScriptedModel};
use openai_agents_sdk::{Result, RunState, Runner};
use serde_json::Value;

/// Python: `test_released_api_contract_fixture_matches_installed_version` (the runner members).
#[test]
fn the_runner_keeps_its_released_entry_points() {
    // Compile-only: the probe is never called, so nothing runs.
    fn probe(agent: &openai_agents_sdk::Agent<()>, options: openai_agents_sdk::RunOptions<()>) {
        drop(Runner::run(agent, "input"));
        drop(Runner::run_with_options(agent, "input", options));
        let _: RunResultStreaming<()> = Runner::run_streamed(agent, "input");
        let _: Result<RunResult<()>> = Runner::run_sync(agent, "input");
    }
    let _ = probe;
}

/// Python: the same test, `RunResult` and `RunState` members, and the frozen
/// `ModelSettings.preserve_raw_usage` field.
#[test]
fn results_and_state_keep_their_released_members() {
    let _ = RunResult::<()>::to_input_list as fn(&RunResult<()>) -> Vec<InputItem>;
    let _ = RunResult::<()>::to_state as fn(&RunResult<()>) -> RunState;
    let _ = RunResult::<()>::last_response_id as fn(&RunResult<()>) -> Option<String>;
    let _ = RunResult::<()>::final_output_as::<Value> as fn(&RunResult<()>) -> Result<Value>;
    let _ = RunResultStreaming::<()>::to_state as fn(&RunResultStreaming<()>) -> Option<RunState>;
    let _ = RunState::to_json as fn(&RunState) -> Result<Value>;
    let _ = RunState::from_json as fn(Value) -> Result<RunState>;
    let _ = RunState::from_json_str as fn(&str) -> Result<RunState>;

    let settings = ModelSettings {
        preserve_raw_usage: Some(true),
        ..Default::default()
    };
    assert_eq!(settings.preserve_raw_usage, Some(true));
}

/// Python: `test_repository_release_policy_declares_v020_contract_surfaces`, the curated public
/// properties of `RunState` (`pending_input`) and `RetryPolicyContext`.
#[tokio::test]
async fn the_curated_state_and_retry_properties_exist() {
    let model = std::sync::Arc::new(ScriptedModel::new());
    model
        .enqueue(openai_agents_sdk::testing::ScriptedStep::output(vec![
            openai_agents_sdk::testing::assistant_message("ok"),
        ]))
        .unwrap();
    let agent = openai_agents_sdk::Agent::<()>::new("surface").with_model_instance(model);
    let state = Runner::run(&agent, "go").await.unwrap().to_state();
    let pending: &Vec<InputItem> = &state.pending_input;
    assert!(pending.is_empty());

    fn properties(ctx: &RetryPolicyContext<'_>) -> (bool, ReplaySafety, bool) {
        (
            ctx.response_started(),
            ctx.replay_safety(),
            ctx.stateful_request(),
        )
    }
    let _ = properties;
}

/// Python: `test_repository_release_policy_declares_public_state_surfaces`, `ScriptedModel`.
#[test]
fn the_scripted_model_keeps_its_recording_properties() {
    let model = ScriptedModel::new();
    let _ = ScriptedModel::calls as fn(&ScriptedModel) -> Vec<ModelCall>;
    let _ = ScriptedModel::first_call as fn(&ScriptedModel) -> Option<ModelCall>;
    let _ = ScriptedModel::last_call as fn(&ScriptedModel) -> Option<ModelCall>;
    let _ = ScriptedModel::remaining_steps as fn(&ScriptedModel) -> usize;
    assert!(model.calls().is_empty());
    assert!(model.first_call().is_none());
    assert!(model.last_call().is_none());
    assert_eq!(model.remaining_steps(), 0);
}

#[cfg(feature = "sandbox")]
mod sandbox_surface {
    use openai_agents_sdk::sandbox::SandboxSessionState;
    use openai_agents_sdk::sandbox::testing::{
        SandboxCall, ScriptedSandboxSession, scripted_sandbox_session,
    };

    /// Python: the `SandboxSessionState` properties `mount_authority_redacted` and
    /// `mount_authority_rebound`, and the `scripted_sandbox_session` factory properties.
    #[test]
    fn the_sandbox_state_and_scripted_session_keep_their_properties() {
        fn authority_flags(state: &SandboxSessionState) -> (bool, bool) {
            (
                state.mount_authority_redacted,
                state.mount_authority_rebound,
            )
        }
        let _ = authority_flags;
        let session = scripted_sandbox_session(Vec::new());
        let _ = ScriptedSandboxSession::calls as fn(&ScriptedSandboxSession) -> Vec<SandboxCall>;
        let _ = ScriptedSandboxSession::remaining_steps as fn(&ScriptedSandboxSession) -> usize;
        assert!(session.calls().is_empty());
        assert_eq!(session.remaining_steps(), 0);
    }
}

#[cfg(feature = "realtime")]
mod realtime_surface {
    use openai_agents_sdk::realtime::RealtimeModelSendEvent;
    use openai_agents_sdk::realtime::testing::{RealtimeConnectCall, ScriptedRealtimeModel};

    /// Python: `ScriptedRealtimeModel` properties.
    #[test]
    fn the_scripted_realtime_model_keeps_its_properties() {
        let _ = ScriptedRealtimeModel::remaining_steps as fn(&ScriptedRealtimeModel) -> usize;
        let _ = ScriptedRealtimeModel::sent_events
            as fn(&ScriptedRealtimeModel) -> Vec<RealtimeModelSendEvent>;
        let _ = ScriptedRealtimeModel::connect_calls
            as fn(&ScriptedRealtimeModel) -> Vec<RealtimeConnectCall>;
    }
}

#[cfg(feature = "voice")]
mod voice_surface {
    use openai_agents_sdk::voice::model::{STTModel, TTSModel};
    use openai_agents_sdk::voice::testing::{
        FakeSTTModel, FakeTTSModel, FakeTranscriptionSession, FakeVoiceWorkflow, STTCall,
        STTSessionCall, TTSCall,
    };

    /// Python: the voice testing doubles' properties and `model_name` on the model interfaces.
    #[test]
    fn the_voice_doubles_and_model_interfaces_keep_their_properties() {
        let _ = FakeSTTModel::calls as fn(&FakeSTTModel) -> Vec<STTCall>;
        let _ = FakeSTTModel::session_calls as fn(&FakeSTTModel) -> Vec<STTSessionCall>;
        let _ =
            FakeSTTModel::created_sessions as fn(&FakeSTTModel) -> Vec<FakeTranscriptionSession>;
        let _ = FakeTTSModel::calls as fn(&FakeTTSModel) -> Vec<TTSCall>;
        let _ = FakeVoiceWorkflow::transcriptions as fn(&FakeVoiceWorkflow) -> Vec<String>;
        let _ = <FakeSTTModel as STTModel>::model_name as fn(&FakeSTTModel) -> &str;
        let _ = <FakeTTSModel as TTSModel>::model_name as fn(&FakeTTSModel) -> &str;
    }
}

/// Python: `test_all_public_exception_classes_are_reexported`. Every exception class of
/// `agents.exceptions` is a variant of the one `AgentsError` enum, reachable from the crate root
/// together with `RunErrorDetails`. The match has no wildcard: a new variant must be added here
/// (and so to the released surface) deliberately. `MCPToolCancellationError`, an internal MCP
/// cancellation marker of the Python event loop, has no Rust variant.
#[test]
fn every_exception_class_is_a_variant_reachable_from_the_crate_root() {
    use openai_agents_sdk::{AgentsError, RunErrorDetails};

    fn name(error: &AgentsError) -> &'static str {
        match error {
            AgentsError::MaxTurnsExceeded { .. } => "MaxTurnsExceeded",
            AgentsError::ModelBehaviorError { .. } => "ModelBehaviorError",
            AgentsError::ModelRefusalError { .. } => "ModelRefusalError",
            AgentsError::ModelTimeoutError { .. } => "ModelTimeoutError",
            AgentsError::UserError { .. } => "UserError",
            AgentsError::ToolTimeoutError { .. } => "ToolTimeoutError",
            AgentsError::InputGuardrailTripwireTriggered { .. } => {
                "InputGuardrailTripwireTriggered"
            }
            AgentsError::OutputGuardrailTripwireTriggered { .. } => {
                "OutputGuardrailTripwireTriggered"
            }
            AgentsError::ToolInputGuardrailTripwireTriggered { .. } => {
                "ToolInputGuardrailTripwireTriggered"
            }
            AgentsError::ToolOutputGuardrailTripwireTriggered { .. } => {
                "ToolOutputGuardrailTripwireTriggered"
            }
            AgentsError::ApiError { .. } => "ApiError",
            AgentsError::Http(_) => "Http",
            AgentsError::Json(_) => "Json",
            AgentsError::Io(_) => "Io",
            AgentsError::Sandbox { .. } => "Sandbox",
            AgentsError::Other(_) => "Other",
        }
    }
    assert_eq!(name(&AgentsError::user("boom")), "UserError");
    let _ = RunErrorDetails::pretty_print as fn(&RunErrorDetails) -> String;
}
