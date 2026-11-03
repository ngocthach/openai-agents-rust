//! Compaction bookkeeping of settled tool outputs (`tests/deferred_session_writes/
//! test_compaction.py`): which batches make a compaction-aware session defer its compaction.
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::error::Result as SdkResult;
use openai_agents_sdk::memory::{Session, SessionRunCompletion};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{
    Agent, GuardrailFunctionOutput, InputItem, OutputGuardrail, OutputItem, RunOptions, RunState,
    Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

#[derive(Debug)]
struct RecordingSession {
    inner: ListSession,
    completions: Mutex<Vec<(Option<String>, bool)>>,
}

impl RecordingSession {
    fn new() -> Self {
        RecordingSession {
            inner: ListSession::new("test"),
            completions: Mutex::new(Vec::new()),
        }
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
        self.completions
            .lock()
            .unwrap()
            .push((info.response_id.clone(), info.has_local_tool_outputs));
        Ok(Usage::default())
    }
}

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

fn always_fine() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn deferring() -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: vec!["finish".into()],
    })
}

fn step(output: Vec<OutputItem>, response_id: &str) -> ScriptedStep {
    ScriptedStep::output(output).with_response_id(Some(response_id.to_string()))
}

fn mcp_request() -> OutputItem {
    OutputItem::Other(json!({
        "type": "mcp_approval_request", "id": "mcpr_1", "server_label": "srv",
        "name": "do_it", "arguments": "{}",
    }))
}

fn hosted_mcp() -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": "srv", "server_url": "https://example.com",
        "require_approval": "always",
    }))
}

async fn run(
    agent: &Agent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: Arc<RecordingSession>,
    streamed: bool,
) -> RunState {
    use futures::StreamExt;
    let options = RunOptions::new(()).session(session);
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        result
            .to_state()
            .expect("a finished streamed run has state")
    } else {
        Runner::run_with_options(agent, input, options)
            .await
            .unwrap()
            .to_state()
    }
}

// test_the_park_records_the_response_the_batch_belongs_to[streamed]
#[tokio::test]
async fn a_park_records_the_response_its_held_batch_belongs_to() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
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
        let agent = Agent::<()>::new("deferred")
            .with_tool(write_thing())
            .with_output_guardrail(always_fine())
            .with_tool_use_behavior(deferring())
            .with_model_instance(model);
        let session = Arc::new(RecordingSession::new());
        let state = run(&agent, "do the thing", session, streamed).await;
        assert_eq!(state.get_interruptions().len(), 1, "streamed={streamed}");
        let pending = state
            .to_json()
            .unwrap()
            .get("pending_session_write")
            .cloned()
            .expect("the park holds its batch");
        assert_eq!(pending["held"], json!(true), "streamed={streamed}");
        assert_eq!(
            pending["response_id"],
            json!("resp_parked"),
            "streamed={streamed}"
        );
    }
}

// test_a_held_mcp_approval_pair_defers_compaction_when_it_settles and
// test_an_ordinary_mcp_approval_response_defers_compaction_too
#[tokio::test]
async fn a_settled_mcp_approval_pair_defers_compaction_for_its_response() {
    openai_agents_sdk::set_tracing_disabled(true);
    for gated in [false, true] {
        let model = Arc::new(ScriptedModel::new());
        model
            .enqueue(step(vec![mcp_request()], "resp_parked"))
            .unwrap();
        model
            .enqueue(step(vec![assistant_message("done")], "resp_final"))
            .unwrap();
        let mut agent = Agent::<()>::new("agent")
            .with_tool(hosted_mcp())
            .with_model_instance(model);
        if gated {
            agent = agent
                .with_output_guardrail(always_fine())
                .with_tool_use_behavior(deferring());
        }
        let session = Arc::new(RecordingSession::new());
        let first = run(&agent, "go", session.clone(), false).await;
        let mut state = RunState::from_json(first.to_json().unwrap()).unwrap();
        for approval in state.get_interruptions() {
            state.approve(&approval, false);
        }
        session.completions.lock().unwrap().clear();
        run(&agent, state, session.clone(), false).await;

        let completions = session.completions.lock().unwrap().clone();
        assert!(
            completions.contains(&(Some("resp_parked".into()), true)),
            "gated={gated}: {completions:?}"
        );
        assert!(
            !completions.contains(&(Some("resp_parked".into()), false)),
            "gated={gated}: the parked response must not be compacted while its approval \
             response is the only local output: {completions:?}"
        );
        let stored: Vec<String> = session
            .inner
            .items()
            .iter()
            .filter_map(|i| i.item_type().map(str::to_string))
            .collect();
        let request = stored.iter().position(|t| t == "mcp_approval_request");
        let response = stored.iter().position(|t| t == "mcp_approval_response");
        assert!(
            request < response && response.is_some(),
            "gated={gated}: {stored:?}"
        );
    }
}

// test_the_compaction_deferral_reads_the_settling_batch_not_the_callers_input
#[tokio::test]
async fn an_earlier_tool_output_in_the_callers_input_does_not_defer_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(step(vec![assistant_message("fresh")], "resp_fresh"))
        .unwrap();
    let agent = Agent::<()>::new("agent").with_model_instance(model);
    let session = Arc::new(RecordingSession::new());
    let input: Vec<InputItem> = vec![
        InputItem::from_value(json!({
            "type": "function_call", "call_id": "call_EARLIER", "name": "t", "arguments": "{}",
        })),
        InputItem::from_value(json!({
            "type": "function_call_output", "call_id": "call_EARLIER", "output": "old",
        })),
        InputItem::from_value(json!({"role": "user", "content": "go"})),
    ];
    Runner::run_with_options(&agent, input, RunOptions::new(()).session(session.clone()))
        .await
        .unwrap();

    assert_eq!(
        *session.completions.lock().unwrap(),
        vec![(Some("resp_fresh".to_string()), false)]
    );
}

mod max_turns {
    use super::*;
    use futures::StreamExt;
    use openai_agents_sdk::AgentsError;
    use openai_agents_sdk::run::RunCheckpoint;
    use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
    use openai_agents_sdk::testing::FakeModel;

    #[derive(Deserialize, JsonSchema)]
    struct Q {
        query: String,
    }

    fn look_up() -> FunctionTool<()> {
        FunctionTool::new("look_up", "looks up", |_ctx, q: Q| async move {
            Ok(format!("schema for {}", q.query))
        })
    }

    /// Python's `_make_never_finishing_agent`: park on turn two, then keep calling tools.
    fn never_finishing_agent(guardrail: OutputGuardrail<()>) -> Agent<()> {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            "look_up",
            r#"{"query":"a"}"#,
            "call_LOOKUP",
        )]);
        model.add_turn(vec![function_call_with_id(
            "write_thing",
            r#"{"query":"x"}"#,
            "call_PARKED",
        )]);
        for i in 0..8 {
            model.add_turn(vec![function_call_with_id(
                "look_up",
                format!(r#"{{"query":"q{i}"}}"#),
                format!("call_L{i}"),
            )]);
        }
        Agent::<()>::new("deferred repro (never finishing)")
            .with_tool(look_up())
            .with_tool(write_thing())
            .with_output_guardrail(guardrail)
            .with_tool_use_behavior(deferring())
            .with_model_instance(model)
    }

    fn failing_guardrail() -> OutputGuardrail<()> {
        OutputGuardrail::new("fails", |_c, _a, _o: Value| async {
            Err::<GuardrailFunctionOutput, _>(AgentsError::other("output guardrail failed"))
        })
    }

    async fn parked_and_approved(agent: &Agent<()>, session: Arc<RecordingSession>) -> RunState {
        let first = Runner::run_with_options(
            agent,
            "go",
            RunOptions::new(()).session(session).max_turns(Some(3)),
        )
        .await
        .unwrap();
        assert_eq!(first.interruptions.len(), 1);
        let mut state = RunState::from_json(first.to_state().to_json().unwrap()).unwrap();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        assert!(state.pending_session_write.is_some());
        state
    }

    fn handler(output: &'static str) -> RunErrorHandlers<()> {
        RunErrorHandlers::new()
            .on_max_turns(move |_input| async move { Ok(Some(RunErrorHandlerResult::new(output))) })
    }

    // test_a_max_turns_handler_completion_clears_the_held_record[streamed] and
    // test_a_streamed_max_turns_completion_clears_the_held_record
    #[tokio::test]
    async fn a_max_turns_handler_completion_clears_the_held_record() {
        openai_agents_sdk::set_tracing_disabled(true);
        for streamed in [false, true] {
            let agent = never_finishing_agent(always_fine());
            let session = Arc::new(RecordingSession::new());
            let state = parked_and_approved(&agent, session.clone()).await;
            let options = RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3))
                .error_handlers(handler("stopped at max turns"));
            let (output, finished) = if streamed {
                let mut result = Runner::run_streamed_with_options(&agent, state, options);
                {
                    let mut events = std::pin::pin!(result.stream_events());
                    while let Some(event) = events.next().await {
                        event.unwrap();
                    }
                }
                (
                    result.final_output_text(),
                    result.to_state().expect("state").to_json().unwrap(),
                )
            } else {
                let result = Runner::run_with_options(&agent, state, options)
                    .await
                    .unwrap();
                (
                    result.final_output_text(),
                    result.to_state().to_json().unwrap(),
                )
            };
            assert_eq!(
                output.as_deref(),
                Some("stopped at max turns"),
                "streamed={streamed}"
            );
            assert!(
                finished
                    .get("pending_session_write")
                    .is_none_or(Value::is_null),
                "streamed={streamed}: a finished run carries no pending write"
            );
        }
    }

    fn parked_pair(session: &RecordingSession) -> Vec<String> {
        session
            .inner
            .items()
            .iter()
            .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
            .filter_map(|i| i.item_type().map(str::to_string))
            .collect()
    }

    // test_a_failed_max_turns_finalization_keeps_the_held_record: the step that fails after the
    // terminal decision is the output guardrail. Python's batch is still held at that point; the
    // Rust runner settles it when the resume starts, so the executed call and output must already
    // be in the session, once, and the checkpoint of the failed run must not owe a second write.
    #[tokio::test]
    async fn a_failed_max_turns_finalization_keeps_the_executed_output() {
        openai_agents_sdk::set_tracing_disabled(true);
        let agent = never_finishing_agent(failing_guardrail());
        let session = Arc::new(RecordingSession::new());
        let state = parked_and_approved(&agent, session.clone()).await;
        let slot = RunCheckpoint::new();
        let error = Runner::run_with_options(
            &agent,
            state,
            RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3))
                .checkpoint(&slot)
                .error_handlers(handler("stopped at max turns")),
        )
        .await
        .expect_err("the output guardrail fails the run");
        assert!(
            error.to_string().contains("output guardrail failed"),
            "{error}"
        );
        assert_eq!(
            parked_pair(&session),
            ["function_call", "function_call_output"]
        );
        let checkpoint = slot.take().expect("a failed run leaves its checkpoint");
        assert!(checkpoint.pending_session_write.is_none());
    }

    // test_a_rejected_max_turns_handler_output_keeps_the_held_record: a wrongly typed handler
    // output is rejected, the executed output stays stored once, and a retry from the failed
    // run's checkpoint with a valid output completes without writing the pair again.
    #[tokio::test]
    async fn a_rejected_max_turns_handler_output_keeps_the_executed_output() {
        openai_agents_sdk::set_tracing_disabled(true);
        let agent = never_finishing_agent(always_fine()).with_output_type::<i64>();
        let session = Arc::new(RecordingSession::new());
        let state = parked_and_approved(&agent, session.clone()).await;
        let slot = RunCheckpoint::new();
        let error = Runner::run_with_options(
            &agent,
            state,
            RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3))
                .checkpoint(&slot)
                .error_handlers(handler("not an int")),
        )
        .await
        .expect_err("a string is not valid structured output");
        assert!(
            error
                .to_string()
                .contains("Invalid run error handler final_output"),
            "{error}"
        );
        assert_eq!(
            parked_pair(&session),
            ["function_call", "function_call_output"]
        );

        let checkpoint = RunState::from_json(
            slot.take()
                .expect("a failed run leaves its checkpoint")
                .to_json()
                .unwrap(),
        )
        .unwrap();
        let retried = Runner::run_with_options(
            &agent,
            checkpoint,
            RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3))
                .error_handlers(
                    RunErrorHandlers::new()
                        .on_max_turns(|_input| async { Ok(Some(RunErrorHandlerResult::new(42))) }),
                ),
        )
        .await
        .expect("a valid handler output completes the retry");
        assert_eq!(retried.final_output_text().as_deref(), Some("42"));
        assert_eq!(
            parked_pair(&session),
            ["function_call", "function_call_output"]
        );
    }
}

// test_the_held_batch_rides_a_non_streamed_result_into_its_checkpoint
#[tokio::test]
async fn the_held_batch_rides_a_non_streamed_result_into_its_checkpoint() {
    openai_agents_sdk::set_tracing_disabled(true);
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
    let agent = Agent::<()>::new("deferred")
        .with_tool(write_thing())
        .with_output_guardrail(always_fine())
        .with_tool_use_behavior(deferring())
        .with_model_instance(model);
    let session = Arc::new(RecordingSession::new());
    let first = Runner::run_with_options(
        &agent,
        "do the thing",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1);

    let checkpoint = first.to_state().to_json().unwrap();
    let pending = checkpoint
        .get("pending_session_write")
        .filter(|v| !v.is_null())
        .expect("the declaration rides the result into its checkpoint");
    assert_eq!(pending["held"], json!(true));
    let items = pending["items"].as_array().unwrap();
    assert!(items.iter().any(|i| i["call_id"] == "call_PARKED"));
    assert!(pending["before"].is_null());
    // The accepted input persists eagerly even at a deferred park, so a discard of the held
    // batch can never take the session's only copy of the input with it.
    assert!(!items.iter().any(|i| i["role"] == "user"));
    assert!(
        session
            .inner
            .items()
            .iter()
            .any(|i| i.as_value()["role"] == "user")
    );
}

fn held_agent(store: Option<bool>) -> Agent<()> {
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
    let mut settings = openai_agents_sdk::model_settings::ModelSettings::default();
    if let Some(store) = store {
        settings = settings.with_store(store);
    }
    Agent::<()>::new("deferred")
        .with_tool(write_thing())
        .with_output_guardrail(always_fine())
        .with_tool_use_behavior(deferring())
        .with_model_settings(settings)
        .with_model_instance(model)
}

// test_the_park_records_the_store_the_response_was_produced_under[streamed]
#[tokio::test]
async fn a_park_records_the_store_setting_of_its_turn() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let agent = held_agent(Some(true));
        let state = run(
            &agent,
            "do the thing",
            Arc::new(RecordingSession::new()),
            streamed,
        )
        .await;
        assert_eq!(state.get_interruptions().len(), 1, "streamed={streamed}");
        let pending = state.to_json().unwrap()["pending_session_write"].clone();
        assert_eq!(pending["held"], json!(true), "streamed={streamed}");
        assert_eq!(pending["store"], json!(true), "streamed={streamed}");
    }
}

// test_a_held_checkpoint_loads_only_under_the_label_that_introduced_it[streamed]
#[tokio::test]
async fn a_held_checkpoint_loads_only_under_the_label_that_introduced_it() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let agent = held_agent(None);
        let state = run(
            &agent,
            "do the thing",
            Arc::new(RecordingSession::new()),
            streamed,
        )
        .await;
        assert_eq!(state.get_interruptions().len(), 1, "streamed={streamed}");

        let payload = state.to_json().unwrap();
        assert_eq!(payload["pending_session_write"]["held"], json!(true));
        assert_eq!(
            payload["$schemaVersion"],
            json!(openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION)
        );
        RunState::from_json(payload.clone()).expect("the current label loads the held batch");

        // 1.18 and 1.17 are released labels whose readers validate the pending write by exact
        // key set: only the held keys can be the reason they refuse this payload.
        for label in ["1.18", "1.17"] {
            let mut relabeled = payload.clone();
            relabeled["$schemaVersion"] = json!(label);
            let error = RunState::from_json(relabeled)
                .expect_err("a released label must refuse the held keys");
            assert!(
                error
                    .to_string()
                    .contains("pending Session write is invalid"),
                "streamed={streamed} label={label}: {error}"
            );
        }
    }
}
