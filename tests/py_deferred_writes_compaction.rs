//! Parity port of `tests/deferred_session_writes/test_compaction.py`.
//!
//! Python's `_CompactionRecordingSession` records the `_defer_compaction` / `run_compaction`
//! hooks a compaction-aware backend sees. Rust's equivalent hook is
//! `Session::on_run_completed(&SessionRunCompletion)`: a call carrying
//! `has_local_tool_outputs = true` is Python's `{"deferred": id, "store": s}` entry and one with
//! `false` is `{"response_id": id, "store": s, "force": False}`.
//!
//! Tests that drive Python's private persistence functions directly
//! (`save_result_to_session`, `resume_pending_session_write`,
//! `save_final_turn_items_after_guardrails`, `_pending_session_write`) have no Rust
//! counterpart and are classified Python-only in the report; the observable end states are
//! ported through `Runner`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::error::Result as SdkResult;
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::memory::{
    CompactionContext, OpenAIResponsesCompactionSession, ResolvedCompactionMode, Session,
    SessionRunCompletion,
};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OpenAIClient, OutputGuardrail,
    RunConfig, RunOptions, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

/// Compaction modes the session observed, keyed by response id.
type ObservedModes = Arc<Mutex<Option<Vec<(String, ResolvedCompactionMode)>>>>;

#[derive(Deserialize, JsonSchema)]
struct Query {
    query: String,
}

fn write_thing() -> FunctionTool<()> {
    FunctionTool::new("write_thing", "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn write_other() -> FunctionTool<()> {
    FunctionTool::new("write_other", "writes", |_ctx, q: Query| async move {
        Ok(format!("other:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn stop_at(names: &[&str]) -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: names.iter().map(|n| n.to_string()).collect(),
    })
}

fn always_fine() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn always_trips() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_trips", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    })
}

fn step(output: Vec<openai_agents_sdk::items::OutputItem>, response_id: &str) -> ScriptedStep {
    ScriptedStep::output(output).with_response_id(Some(response_id.to_string()))
}

/// One `on_run_completed` call as Python's recording session sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Completion {
    response_id: Option<String>,
    store: Option<bool>,
    /// `true` is Python's `deferred` entry, `false` its `run_compaction` entry.
    deferred: bool,
}

/// Port of `_CompactionRecordingSession`.
#[derive(Debug)]
struct RecordingSession {
    inner: ListSession,
    completions: Mutex<Vec<Completion>>,
}

impl RecordingSession {
    fn new() -> Self {
        RecordingSession {
            inner: ListSession::new("test"),
            completions: Mutex::new(Vec::new()),
        }
    }

    fn completions(&self) -> Vec<Completion> {
        self.completions.lock().unwrap().clone()
    }

    fn items(&self) -> Vec<InputItem> {
        self.inner.items()
    }
}

#[async_trait]
impl Session for RecordingSession {
    fn session_id(&self) -> &str {
        "test"
    }

    async fn get_items(&self, limit: Option<usize>) -> SdkResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> SdkResult<()> {
        self.inner.add_items(items).await
    }

    async fn pop_item(&self) -> SdkResult<Option<InputItem>> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> SdkResult<()> {
        self.inner.clear_session().await
    }

    async fn on_run_completed(&self, info: &SessionRunCompletion) -> SdkResult<Usage> {
        self.completions.lock().unwrap().push(Completion {
            response_id: info.response_id.clone(),
            store: info.store,
            deferred: info.has_local_tool_outputs,
        });
        Ok(Usage::default())
    }
}

fn call_ids(items: &[InputItem], ty: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some(ty))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn parked_pair(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect()
}

fn expected_pair() -> Vec<String> {
    vec!["function_call".into(), "function_call_output".into()]
}

struct Outcome {
    interruptions: usize,
    state: RunState,
}

fn config_with_store(store: Option<bool>) -> RunConfig<()> {
    let mut settings = ModelSettings::default();
    if let Some(store) = store {
        settings = settings.with_store(store);
    }
    RunConfig::<()>::default().with_model_settings(settings)
}

async fn drive(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<dyn Session>>,
    store: Option<bool>,
    streamed: bool,
) -> Result<Outcome, AgentsError> {
    let mut options = RunOptions::new(()).run_config(config_with_store(store));
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            use futures::StreamExt;
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(Outcome {
            interruptions: result.interruptions().len(),
            state: result
                .to_state()
                .expect("a finished streamed run has state"),
        })
    } else {
        let result = Runner::run_with_options(agent, input, options).await?;
        Ok(Outcome {
            interruptions: result.interruptions.len(),
            state: result.to_state(),
        })
    }
}

fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

fn approve_first(state: &mut RunState) {
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
}

// ---------------------------------------------------------------------------
// test_approval_settlement_preserves_response_storage_mode (3 x 2 x 3 = 18 expanded)
// ---------------------------------------------------------------------------

type Observed = Arc<Mutex<Vec<(String, ResolvedCompactionMode)>>>;

struct StorageModeRun {
    observed: Vec<(String, ResolvedCompactionMode)>,
    /// `observed` as the second model request started, when one started.
    observed_at_second_request: Option<Vec<(String, ResolvedCompactionMode)>>,
    items: Vec<InputItem>,
    resumed_error: Option<String>,
}

async fn storage_mode_scenario(
    streamed: bool,
    completion: &str,
    parked_store: Option<bool>,
    resumed_store: bool,
) -> StorageModeRun {
    openai_agents_sdk::set_tracing_disabled(true);
    let observed: Observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let underlying = Arc::new(ListSession::new("test"));
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("test", underlying.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test"))
            .with_should_trigger_compaction(move |context: &CompactionContext| {
                recorder.lock().unwrap().push((
                    context.response_id.clone().unwrap_or_default(),
                    context.compaction_mode,
                ));
                false
            }),
    );

    let at_second: ObservedModes = Arc::new(Mutex::new(None));
    let probe = observed.clone();
    let slot = at_second.clone();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(step(
            vec![function_call_with_id(
                "write_thing",
                r#"{"query":"x"}"#,
                "call_PARKED",
            )],
            "resp_parked",
        ))
        .unwrap();
    model
        .enqueue(ScriptedStep::respond(move |_call| {
            *slot.lock().unwrap() = Some(probe.lock().unwrap().clone());
            Ok(step(vec![assistant_message("done")], "resp_new"))
        }))
        .unwrap();

    let stop_name = if completion == "continue" {
        "finish"
    } else {
        "write_thing"
    };
    let build = |guardrail: OutputGuardrail<()>| {
        Agent::<()>::new("storage mode")
            .with_tool(write_thing())
            .with_output_guardrail(guardrail)
            .with_tool_use_behavior(stop_at(&[stop_name]))
            .with_model_instance(model.clone())
    };
    let agent = build(always_fine());

    let session_dyn: Arc<dyn Session> = session.clone();
    let first = drive(
        &agent,
        "go",
        Some(session_dyn.clone()),
        parked_store,
        streamed,
    )
    .await
    .unwrap();
    let mut state = round_trip(&first.state);
    approve_first(&mut state);

    let resume_agent = if completion == "blocked" {
        build(always_trips())
    } else {
        agent
    };
    let resumed = drive(
        &resume_agent,
        state,
        Some(session_dyn),
        Some(resumed_store),
        streamed,
    )
    .await;
    let resumed_error = resumed.err().map(|e| e.to_string());

    let observed = observed.lock().unwrap().clone();
    let observed_at_second_request = at_second.lock().unwrap().clone();
    StorageModeRun {
        observed,
        observed_at_second_request,
        items: underlying.items(),
        resumed_error,
    }
}

fn expected_mode(parked_store: Option<bool>) -> ResolvedCompactionMode {
    if parked_store == Some(false) {
        ResolvedCompactionMode::Input
    } else {
        ResolvedCompactionMode::PreviousResponseId
    }
}

async fn check_storage_mode(completion: &str, metadata: bool) {
    for streamed in [false, true] {
        for (parked_store, resumed_store) in
            [(Some(false), true), (Some(true), false), (None, false)]
        {
            let label = format!(
                "streamed={streamed} completion={completion} parked={parked_store:?} resumed={resumed_store}"
            );
            let run =
                storage_mode_scenario(streamed, completion, parked_store, resumed_store).await;
            if completion == "blocked" {
                assert!(
                    run.resumed_error.is_some(),
                    "{label}: the guardrail must trip"
                );
            } else {
                assert!(
                    run.resumed_error.is_none(),
                    "{label}: {:?}",
                    run.resumed_error
                );
            }
            assert_eq!(parked_pair(&run.items), expected_pair(), "{label}");
            if !metadata {
                continue;
            }
            let mode = expected_mode(parked_store);
            assert_eq!(
                run.observed.first(),
                Some(&("resp_parked".to_string(), mode)),
                "{label}: the parked response's compaction decision must use its own storage mode"
            );
            if completion == "continue" {
                assert_eq!(
                    run.observed_at_second_request,
                    Some(vec![("resp_parked".to_string(), mode)]),
                    "{label}: the original response must settle before another model request"
                );
            }
        }
    }
}

#[tokio::test]
async fn approval_settlement_persists_the_pair_when_the_run_continues() {
    check_storage_mode("continue", false).await;
}

#[tokio::test]
async fn approval_settlement_keeps_the_storage_mode_when_the_run_continues() {
    check_storage_mode("continue", true).await;
}

#[tokio::test]
async fn approval_settlement_persists_the_pair_when_the_resume_is_terminal() {
    check_storage_mode("terminal", false).await;
}

#[tokio::test]
async fn approval_settlement_keeps_the_storage_mode_when_the_resume_is_terminal() {
    check_storage_mode("terminal", true).await;
}

#[tokio::test]
async fn approval_settlement_persists_the_pair_when_the_resume_is_blocked() {
    check_storage_mode("blocked", false).await;
}

#[tokio::test]
async fn approval_settlement_keeps_the_storage_mode_when_the_resume_is_blocked() {
    check_storage_mode("blocked", true).await;
}

// ---------------------------------------------------------------------------
// test_detached_repark_advances_response_storage_metadata (2 x 2 expanded)
// ---------------------------------------------------------------------------

async fn detached_repark(streamed: bool, disable_guardrails: bool) {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(RecordingSession::new());
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(vec![
            step(
                vec![function_call_with_id(
                    "write_thing",
                    r#"{"query":"a"}"#,
                    "call_A",
                )],
                "resp_a",
            ),
            step(
                vec![function_call_with_id(
                    "write_other",
                    r#"{"query":"b"}"#,
                    "call_B",
                )],
                "resp_b",
            ),
            step(vec![assistant_message("done")], "resp_c"),
        ])
        .unwrap();
    let build = |guardrails: bool| {
        let mut agent = Agent::<()>::new("new response frontier")
            .with_tool(write_thing())
            .with_tool(write_other())
            .with_tool_use_behavior(stop_at(&["finish"]))
            .with_model_instance(model.clone());
        if guardrails {
            agent = agent.with_output_guardrail(always_fine());
        }
        agent
    };
    let gated = build(true);
    let session_dyn: Arc<dyn Session> = session.clone();

    let first = drive(
        &gated,
        "go",
        Some(session_dyn.clone()),
        Some(true),
        streamed,
    )
    .await
    .unwrap();
    let mut state = round_trip(&first.state);
    approve_first(&mut state);

    let second_agent = if disable_guardrails {
        build(false)
    } else {
        build(true)
    };
    let second = drive(&second_agent, state, None, Some(false), streamed)
        .await
        .unwrap();
    let mut state = round_trip(&second.state);
    approve_first(&mut state);

    drive(
        &second_agent,
        state,
        Some(session_dyn),
        Some(true),
        streamed,
    )
    .await
    .unwrap();

    assert_eq!(
        session.completions(),
        vec![
            Completion {
                response_id: Some("resp_b".into()),
                store: Some(false),
                deferred: true,
            },
            Completion {
                response_id: Some("resp_c".into()),
                store: Some(true),
                deferred: false,
            },
        ],
        "streamed={streamed} disable_guardrails={disable_guardrails}"
    );
    let items = session.items();
    let mut ids = call_ids(&items, "function_call");
    ids.sort();
    assert_eq!(ids, vec!["call_A".to_string(), "call_B".to_string()]);
}

/// Python records `resp_b` as a deferred settle with its own `store=False` when the detached
/// repark's batch lands.
#[tokio::test]
async fn a_detached_repark_advances_the_response_storage_metadata() {
    for streamed in [false, true] {
        for disable_guardrails in [false, true] {
            detached_repark(streamed, disable_guardrails).await;
        }
    }
}

// ---------------------------------------------------------------------------
// test_partial_approval_repark_keeps_response_storage_setting (2 x 3 expanded)
// ---------------------------------------------------------------------------

async fn partial_repark(streamed: bool, parked_store: Option<bool>) {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(RecordingSession::new());
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(vec![
            step(
                vec![
                    function_call_with_id("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
                    function_call_with_id("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
                ],
                "resp_multi",
            ),
            step(vec![assistant_message("done")], "resp_final"),
        ])
        .unwrap();
    let agent = Agent::<()>::new("deferred repro (multi)")
        .with_tool(write_thing())
        .with_tool(write_other())
        .with_output_guardrail(always_fine())
        .with_tool_use_behavior(stop_at(&["finish"]))
        .with_model_instance(model.clone());
    let session_dyn: Arc<dyn Session> = session.clone();
    // Python's `not parked_store`: `None` and `False` both become `True`.
    let resumed_store = Some(parked_store != Some(true));

    let first = drive(
        &agent,
        "go",
        Some(session_dyn.clone()),
        parked_store,
        streamed,
    )
    .await
    .unwrap();
    let mut state = round_trip(&first.state);
    approve_first(&mut state);

    let second = drive(
        &agent,
        state,
        Some(session_dyn.clone()),
        resumed_store,
        streamed,
    )
    .await
    .unwrap();
    assert_eq!(
        second.interruptions, 1,
        "streamed={streamed} parked={parked_store:?}"
    );
    assert!(
        session.completions().is_empty(),
        "streamed={streamed} parked={parked_store:?}: {:?}",
        session.completions()
    );

    let mut state = round_trip(&second.state);
    approve_first(&mut state);
    drive(&agent, state, Some(session_dyn), resumed_store, streamed)
        .await
        .unwrap();

    assert_eq!(
        session.completions().first(),
        Some(&Completion {
            response_id: Some("resp_multi".into()),
            store: parked_store,
            deferred: true,
        }),
        "streamed={streamed} parked={parked_store:?}: {:?}",
        session.completions()
    );
}

/// A partial approval that parks again keeps the response's own storage setting, and the session
/// hears nothing until the batch settles.
#[tokio::test]
async fn a_partial_approval_repark_keeps_the_response_storage_setting() {
    for streamed in [false, true] {
        for parked_store in [None, Some(false), Some(true)] {
            partial_repark(streamed, parked_store).await;
        }
    }
}

// ---------------------------------------------------------------------------
// test_a_partial_settle_on_a_compaction_session_still_fails_the_gated_resume_fast (x2)
// ---------------------------------------------------------------------------

/// A resume that re-enables the output guardrails after a partial settle persisted current-turn
/// items is refused up front.
#[tokio::test]
async fn a_partial_settle_fails_the_regated_resume_fast() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(RecordingSession::new());
        let model = Arc::new(ScriptedModel::new());
        model
            .extend(vec![
                step(
                    vec![
                        function_call_with_id("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
                        function_call_with_id("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
                    ],
                    "resp_multi",
                ),
                step(vec![assistant_message("done")], "resp_final"),
            ])
            .unwrap();
        let build = |guardrails: bool| {
            let mut agent = Agent::<()>::new("deferred repro (multi)")
                .with_tool(write_thing())
                .with_tool(write_other())
                .with_tool_use_behavior(stop_at(&["finish"]))
                .with_model_instance(model.clone());
            if guardrails {
                agent = agent.with_output_guardrail(always_fine());
            }
            agent
        };
        let gated = build(true);
        let ungated = build(false);
        let session_dyn: Arc<dyn Session> = session.clone();

        let first = drive(&gated, "go", Some(session_dyn.clone()), None, streamed)
            .await
            .unwrap();
        let mut state = round_trip(&first.state);
        let parked = state
            .get_interruptions()
            .iter()
            .find(|i| i.raw_item.call_id == "call_PARKED")
            .cloned()
            .expect("call_PARKED is parked");
        state.approve(&parked, false);

        let second = drive(&ungated, state, Some(session_dyn.clone()), None, streamed)
            .await
            .unwrap();
        assert_eq!(second.interruptions, 1);

        let mut state = round_trip(&second.state);
        for item in state.get_interruptions() {
            state.approve(&item, false);
        }
        let error = drive(&gated, state, Some(session_dyn), None, streamed)
            .await
            .err()
            .expect("the re-gated resume must be refused");
        assert!(
            error
                .to_string()
                .contains("output guardrails after current-turn items"),
            "{error}"
        );

        let items = session.items();
        let calls = call_ids(&items, "function_call");
        assert_eq!(calls.iter().filter(|c| *c == "call_PARKED").count(), 1);
        assert_eq!(calls.iter().filter(|c| *c == "call_PARKED_2").count(), 1);
        let outputs = call_ids(&items, "function_call_output");
        assert_eq!(outputs, vec!["call_PARKED".to_string()]);
    }
}
