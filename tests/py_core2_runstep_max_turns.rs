//! Ported from the Python SDK's `tests/test_max_turns.py` (the 43 tests that
//! `tests/py_core_b_max_turns.rs` does not cover yet).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, refusal_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, CancelMode, GuardrailFunctionOutput, InMemorySession, InputItem,
    ItemHelpers, OutputGuardrail, RunItem, RunOptions, RunState, Runner, Session,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// Mirrors `tests/test_responses.py::get_function_tool`.
fn get_function_tool(name: &str, result: &str) -> FunctionTool<()> {
    let result = result.to_string();
    FunctionTool::new(name, "", move |_ctx, _a: Value| {
        let result = result.clone();
        async move { Ok(result) }
    })
}

/// Mirrors `tests/test_responses.py::get_function_tool_call`.
fn get_function_tool_call(name: &str, arguments: &str, call_id: Option<&str>) -> OutputItem {
    openai_agents_sdk::testing::function_call_full(
        name,
        arguments,
        call_id.unwrap_or("2"),
        Some("1".to_string()),
        None,
    )
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

/// `TypedDict Foo` of the Python test module.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
struct Foo {
    a: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct FooModel {
    summary: String,
}

/// `SimpleListSession` whose `add_items` can be made to fail or hang for message batches.
#[derive(Debug)]
struct ListSession {
    inner: InMemorySession,
    fail_message_save: Arc<AtomicBool>,
    hang_message_save: bool,
}

impl ListSession {
    fn new() -> Arc<Self> {
        Arc::new(Self::with(false, false))
    }

    fn with(fail: bool, hang: bool) -> Self {
        ListSession {
            inner: InMemorySession::new("test"),
            fail_message_save: Arc::new(AtomicBool::new(fail)),
            hang_message_save: hang,
        }
    }

    async fn types(&self) -> Vec<String> {
        item_types(&self.inner.get_items(None).await.unwrap())
    }
}

#[async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        "test"
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let has_message = items
            .iter()
            .any(|i| i.0.get("type").and_then(Value::as_str) == Some("message"));
        if has_message && self.hang_message_save {
            futures::future::pending::<()>().await;
        }
        if has_message && self.fail_message_save.load(Ordering::SeqCst) {
            return Err(AgentsError::user("session save failed"));
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

fn item_types(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| {
            item.0
                .get("type")
                .or_else(|| item.0.get("role"))
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

fn max_turns_handler(output: Value) -> RunErrorHandlers<()> {
    RunErrorHandlers::new().on_max_turns(move |_input| {
        let output = output.clone();
        async move { Ok(Some(RunErrorHandlerResult::new(output))) }
    })
}

fn max_turns_handler_result(result: RunErrorHandlerResult) -> RunErrorHandlers<()> {
    RunErrorHandlers::new().on_max_turns(move |_input| {
        let result = result.clone();
        async move { Ok(Some(result)) }
    })
}

/// What a run (streamed or not) left behind: the result fields, or the error plus every event.
struct Outcome {
    final_output: Option<Value>,
    new_items: Vec<RunItem>,
    state: Option<RunState>,
    output_guardrail_infos: Vec<Value>,
    error: Option<AgentsError>,
    events: Vec<StreamEvent<()>>,
}

async fn run_either(
    agent: &Arc<Agent<()>>,
    input: impl Into<openai_agents_sdk::RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> Outcome {
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent.clone(), input, options);
        let mut events = Vec::new();
        let mut error = None;
        {
            let mut stream = std::pin::pin!(result.stream_events());
            while let Some(ev) = stream.next().await {
                match ev {
                    Ok(ev) => events.push(ev),
                    Err(e) => error = Some(e),
                }
            }
        }
        let res = result.result();
        Outcome {
            final_output: result.final_output().cloned(),
            new_items: res.map(|r| r.new_items.clone()).unwrap_or_default(),
            state: result.to_state(),
            output_guardrail_infos: res
                .map(|r| {
                    r.output_guardrail_results
                        .iter()
                        .map(|g| g.output.output_info.clone())
                        .collect()
                })
                .unwrap_or_default(),
            error,
            events,
        }
    } else {
        match Runner::run_with_options(agent.clone(), input, options).await {
            Ok(r) => Outcome {
                final_output: r.final_output.clone(),
                new_items: r.new_items.clone(),
                state: Some(r.to_state()),
                output_guardrail_infos: r
                    .output_guardrail_results
                    .iter()
                    .map(|g| g.output.output_info.clone())
                    .collect(),
                error: None,
                events: Vec::new(),
            },
            Err(e) => Outcome {
                final_output: None,
                new_items: Vec::new(),
                state: None,
                output_guardrail_infos: Vec::new(),
                error: Some(e),
                events: Vec::new(),
            },
        }
    }
}

fn run_item_events(events: &[StreamEvent<()>]) -> Vec<(RunItemStreamEventName, RunItem)> {
    events
        .iter()
        .filter_map(|ev| match ev {
            StreamEvent::RunItem { name, item, .. } => Some((*name, item.clone())),
            _ => None,
        })
        .collect()
}

fn messages(items: &[RunItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
        .count()
}

// ---------------------------------------------------------------------------------------------
// Refusal handlers
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn non_streamed_structured_output_refusal_raises_without_retry() {
    disable_tracing();
    let model = scripted(vec![vec![refusal_message(
        "I cannot help with that request.",
    )]]);
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_output_type::<FooModel>();
    let err = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(3)),
    )
    .await
    .unwrap_err();
    match err {
        AgentsError::ModelRefusalError { refusal, .. } => {
            assert_eq!(refusal, "I cannot help with that request.")
        }
        other => panic!("expected ModelRefusalError, got {other:?}"),
    }
    assert_eq!(model.remaining_steps(), 0);
}

#[tokio::test]
async fn non_streamed_refusal_handler_returns_structured_output() {
    disable_tracing();
    let model = scripted(vec![vec![refusal_message(
        "I cannot help with that request.",
    )]]);
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model)
        .with_output_type::<FooModel>();
    let handlers = RunErrorHandlers::new().on_model_refusal(|input| async move {
        match input.error.as_ref() {
            AgentsError::ModelRefusalError { refusal, .. } => {
                assert_eq!(refusal, "I cannot help with that request.")
            }
            other => panic!("unexpected error {other:?}"),
        }
        assert!(!input.run_data.raw_responses.is_empty());
        Ok(Some(RunErrorHandlerResult::new(
            json!({"summary": "safe fallback"}),
        )))
    });
    let result = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(3))
            .error_handlers(handlers),
    )
    .await
    .unwrap();
    assert_eq!(
        result.final_output_as::<FooModel>().unwrap().summary,
        "safe fallback"
    );
    assert!(
        ItemHelpers::text_message_outputs(&result.new_items)
            .ends_with(r#"{"summary":"safe fallback"}"#)
    );
}

#[tokio::test]
async fn non_streamed_refusal_handler_can_skip_history() {
    disable_tracing();
    let model = scripted(vec![vec![refusal_message(
        "I cannot help with that request.",
    )]]);
    let agent = Agent::<()>::new("test_1").with_model_instance(model);
    let handlers = RunErrorHandlers::new().on_model_refusal(|_input| async move {
        Ok(Some(
            RunErrorHandlerResult::new("safe fallback").without_history(),
        ))
    });
    let result = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).error_handlers(handlers),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("safe fallback"));
    assert_eq!(ItemHelpers::text_message_outputs(&result.new_items), "");
}

#[tokio::test]
async fn streamed_refusal_handler_returns_output() {
    disable_tracing();
    let model = scripted(vec![vec![refusal_message(
        "I cannot help with that request.",
    )]]);
    let agent = Arc::new(Agent::<()>::new("test_1").with_model_instance(model));
    let handlers = RunErrorHandlers::new().on_model_refusal(|_input| async move {
        Ok(Some(RunErrorHandlerResult::new("safe fallback")))
    });
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(()).error_handlers(handlers),
        true,
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(out.final_output, Some(json!("safe fallback")));
    assert!(run_item_events(&out.events).iter().any(|(name, item)| {
        *name == RunItemStreamEventName::MessageOutputCreated
            && matches!(item.kind, RunItemKind::MessageOutput { .. })
            && ItemHelpers::text_message_output(item) == "safe fallback"
    }));
}

// ---------------------------------------------------------------------------------------------
// Structured output with max turns
// ---------------------------------------------------------------------------------------------

fn tool_loop_agent(model: Arc<ScriptedModel>) -> Arc<Agent<()>> {
    Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(model)
            .with_output_type::<Foo>()
            .with_tool(get_function_tool("tool_1", "result")),
    )
}

fn five_tool_calls() -> Arc<ScriptedModel> {
    scripted(
        (0..5)
            .map(|_| vec![get_function_tool_call("tool_1", "", None)])
            .collect(),
    )
}

#[tokio::test]
async fn structured_output_non_streamed_max_turns() {
    disable_tracing();
    let agent = tool_loop_agent(five_tool_calls());
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(3)),
        false,
    )
    .await;
    assert!(matches!(
        out.error,
        Some(AgentsError::MaxTurnsExceeded { .. })
    ));
}

#[tokio::test]
async fn structured_output_streamed_max_turns() {
    disable_tracing();
    let agent = tool_loop_agent(five_tool_calls());
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(3)),
        true,
    )
    .await;
    assert!(matches!(
        out.error,
        Some(AgentsError::MaxTurnsExceeded { .. })
    ));
}

#[tokio::test]
async fn structured_output_max_turns_handler_invalid_output() {
    disable_tracing();
    let agent = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![]))
            .with_output_type::<Foo>(),
    );
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!({"summary": "nope"}))),
        false,
    )
    .await;
    assert!(
        matches!(out.error, Some(AgentsError::UserError { .. })),
        "{:?}",
        out.error
    );
}

#[tokio::test]
async fn structured_output_max_turns_handler_list_output() {
    disable_tracing();
    let agent = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![]))
            .with_output_type::<Vec<String>>(),
    );
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!(["a", "b"]))),
        false,
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(out.final_output, Some(json!(["a", "b"])));
    assert_eq!(
        ItemHelpers::text_message_outputs(&out.new_items),
        r#"{"response":["a","b"]}"#
    );
}

#[tokio::test]
async fn non_streamed_max_turns_handler_raw_output() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test_1").with_model_instance(scripted(vec![])));
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!("summary"))),
        false,
    )
    .await;
    assert_eq!(out.final_output, Some(json!("summary")));
    assert_eq!(ItemHelpers::text_message_outputs(&out.new_items), "summary");
}

#[tokio::test]
async fn non_streamed_max_turns_handler_raw_dict_output() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test_1").with_model_instance(scripted(vec![])));
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!({"summary": "ok"}))),
        false,
    )
    .await;
    assert_eq!(out.final_output, Some(json!({"summary": "ok"})));
}

#[tokio::test]
async fn streamed_max_turns_handler_pydantic_output() {
    disable_tracing();
    let agent = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![]))
            .with_output_type::<FooModel>(),
    );
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!({"summary": "ok"}))),
        true,
    )
    .await;
    assert_eq!(out.final_output, Some(json!({"summary": "ok"})));
    let events = run_item_events(&out.events);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, RunItemStreamEventName::MessageOutputCreated);
    assert_eq!(
        ItemHelpers::text_message_output(&events[0].1),
        r#"{"summary":"ok"}"#
    );
}

#[tokio::test]
async fn streamed_max_turns_handler_list_output() {
    disable_tracing();
    let agent = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![]))
            .with_output_type::<Vec<String>>(),
    );
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .error_handlers(max_turns_handler(json!(["a", "b"]))),
        true,
    )
    .await;
    assert_eq!(out.final_output, Some(json!(["a", "b"])));
    let events = run_item_events(&out.events);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, RunItemStreamEventName::MessageOutputCreated);
    assert_eq!(
        ItemHelpers::text_message_output(&events[0].1),
        r#"{"response":["a","b"]}"#
    );
}

// ---------------------------------------------------------------------------------------------
// Session persistence of the synthesized output
// ---------------------------------------------------------------------------------------------

fn one_tool_turn_agent(text_first: bool) -> Arc<Agent<()>> {
    let mut step = Vec::new();
    if text_first {
        step.push(assistant_message("same answer"));
    }
    step.push(get_function_tool_call(
        "some_function",
        &json!({"a": "b"}).to_string(),
        None,
    ));
    Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![step]))
            .with_tool(get_function_tool("some_function", "result")),
    )
}

#[tokio::test]
async fn streamed_max_turns_handler_persists_output_to_session() {
    disable_tracing();
    let agent = one_tool_turn_agent(false);
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(1))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!("fallback answer"))),
        true,
    )
    .await;
    assert_eq!(out.final_output, Some(json!("fallback answer")));
    assert_eq!(
        session.types().await,
        vec!["user", "function_call", "function_call_output", "message"]
    );
}

async fn persisted_count_case(include_in_history: bool) {
    let mut seen = Vec::new();
    for streamed in [false, true] {
        let agent = one_tool_turn_agent(false);
        let session = ListSession::new();
        let mut result = RunErrorHandlerResult::new("fallback answer");
        if !include_in_history {
            result = result.without_history();
        }
        let out = run_either(
            &agent,
            "user_message",
            RunOptions::new(())
                .max_turns(Some(1))
                .session(session.clone())
                .error_handlers(max_turns_handler_result(result)),
            streamed,
        )
        .await;
        assert!(out.error.is_none(), "{:?}", out.error);
        let state = out.state.unwrap();
        // Rust tracks persistence as "generated items already saved"; every generated item of
        // the run reached the session, the synthesized message included only when kept.
        assert_eq!(state.session_saved_items, state.generated_items.len());
        let synthesized = state
            .generated_items
            .iter()
            .filter(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
            .count();
        seen.push((synthesized, session.types().await));
    }
    let mut expected = vec!["user", "function_call", "function_call_output"];
    let mut count = 0;
    if include_in_history {
        expected.push("message");
        count = 1;
    }
    for (c, types) in seen {
        assert_eq!(c, count);
        assert_eq!(types, expected);
    }
}

#[tokio::test]
async fn max_turns_handler_persisted_count_matches_after_tool_turn_excluded() {
    disable_tracing();
    persisted_count_case(false).await;
}

#[tokio::test]
async fn max_turns_handler_persisted_count_matches_after_tool_turn_included() {
    disable_tracing();
    persisted_count_case(true).await;
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum GuardrailOutcome {
    Pass,
    Error,
    Tripwire,
}

async fn output_guardrail_session_case(streamed: bool, outcome: GuardrailOutcome) {
    let guardrail = OutputGuardrail::new("check", move |_c, _a, _o: Value| async move {
        match outcome {
            GuardrailOutcome::Error => Err(AgentsError::other("guardrail failed")),
            GuardrailOutcome::Pass => Ok(GuardrailFunctionOutput::new(json!("pass"), false)),
            GuardrailOutcome::Tripwire => Ok(GuardrailFunctionOutput::new(json!("tripwire"), true)),
        }
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![]))
            .with_output_guardrail(guardrail),
    );
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!("fallback answer"))),
        streamed,
    )
    .await;

    match outcome {
        GuardrailOutcome::Error => {
            let err = out.error.as_ref().expect("guardrail error surfaces");
            assert!(err.to_string().contains("guardrail failed"), "{err}");
        }
        GuardrailOutcome::Tripwire => match &out.error {
            Some(AgentsError::OutputGuardrailTripwireTriggered {
                guardrail_result, ..
            }) => {
                assert_eq!(guardrail_result.agent_output, json!("fallback answer"));
                assert_eq!(guardrail_result.output.output_info, json!("tripwire"));
            }
            other => panic!("expected tripwire, got {other:?}"),
        },
        GuardrailOutcome::Pass => {
            assert!(out.error.is_none(), "{:?}", out.error);
            assert_eq!(out.final_output, Some(json!("fallback answer")));
            assert_eq!(out.output_guardrail_infos.len(), 1);
            let state = out.state.as_ref().unwrap();
            assert_eq!(state.session_saved_items, 1);
        }
    }

    let types = session.types().await;
    if outcome == GuardrailOutcome::Tripwire {
        assert_eq!(types, vec!["user"]);
    } else {
        assert_eq!(types, vec!["user", "message"]);
    }

    let fallback_events = run_item_events(&out.events)
        .into_iter()
        .filter(|(_, item)| {
            matches!(item.kind, RunItemKind::MessageOutput { .. })
                && ItemHelpers::text_message_output(item) == "fallback answer"
        })
        .count();
    assert_eq!(
        fallback_events,
        usize::from(streamed && outcome == GuardrailOutcome::Pass)
    );

    if streamed {
        // Python reads `new_items` off the failed streamed result; Rust carries the same
        // history on the error's `run_data`.
        // history on the error's `run_data`. A guardrail's own error is `AgentsError::Other`,
        // which carries no run data, so that case is checked through the session above.
        let new_items = match &out.error {
            None => Some(out.new_items.clone()),
            Some(e) => e.run_data().map(|d| d.new_items.clone()),
        };
        if let Some(new_items) = new_items {
            let expected = usize::from(outcome != GuardrailOutcome::Tripwire);
            assert_eq!(messages(&new_items), expected, "{outcome:?}");
        } else {
            assert_eq!(outcome, GuardrailOutcome::Error);
        }
    }
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_pass_run() {
    disable_tracing();
    output_guardrail_session_case(false, GuardrailOutcome::Pass).await;
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_pass_streamed() {
    disable_tracing();
    output_guardrail_session_case(true, GuardrailOutcome::Pass).await;
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_error_run() {
    disable_tracing();
    output_guardrail_session_case(false, GuardrailOutcome::Error).await;
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_error_streamed() {
    disable_tracing();
    output_guardrail_session_case(true, GuardrailOutcome::Error).await;
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_tripwire_run() {
    disable_tracing();
    output_guardrail_session_case(false, GuardrailOutcome::Tripwire).await;
}

#[tokio::test]
async fn max_turns_handler_output_guardrail_session_semantics_tripwire_streamed() {
    disable_tracing();
    output_guardrail_session_case(true, GuardrailOutcome::Tripwire).await;
}

fn call_ids(items: &[RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| {
            item.to_input_item()
                .0
                .get("call_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

#[tokio::test]
async fn streamed_max_turns_trip_preserves_completed_tool_prefix() {
    disable_tracing();
    let reject = OutputGuardrail::new("reject", |_c, _a, _o: Value| async move {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![get_function_tool_call(
                "some_function",
                "{}",
                Some("accepted-call"),
            )]]))
            .with_tool(get_function_tool("some_function", "accepted-output"))
            .with_output_guardrail(reject),
    );
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "run the tool",
        RunOptions::new(())
            .max_turns(Some(1))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!("rejected fallback"))),
        true,
    )
    .await;
    let err = out.error.expect("tripwire");
    assert!(matches!(
        err,
        AgentsError::OutputGuardrailTripwireTriggered { .. }
    ));
    let data = err.run_data().expect("run data");
    assert_eq!(
        call_ids(&data.new_items),
        vec!["accepted-call", "accepted-call"]
    );
    let serialized = serde_json::to_string(
        &data
            .new_items
            .iter()
            .map(|i| i.to_input_item().0)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(serialized.contains("accepted-output"));
    assert_eq!(
        session.types().await,
        vec!["user", "function_call", "function_call_output"]
    );
}

#[tokio::test]
async fn streamed_max_turns_handler_validation_failure_persists_input() {
    disable_tracing();
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![]))
            .with_output_type::<Foo>(),
    );
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!({"summary": "invalid"}))),
        true,
    )
    .await;
    assert!(
        matches!(out.error, Some(AgentsError::UserError { .. })),
        "{:?}",
        out.error
    );
    assert_eq!(session.types().await, vec!["user"]);
}

async fn equal_message_occurrences_case(streamed: bool) {
    let agent = one_tool_turn_agent(true);
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(1))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!("same answer"))),
        streamed,
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    let types = session.types().await;
    assert_eq!(types.iter().filter(|t| *t == "message").count(), 2);
}

#[tokio::test]
async fn max_turns_handler_records_equal_message_occurrences_run() {
    disable_tracing();
    equal_message_occurrences_case(false).await;
}

#[tokio::test]
async fn max_turns_handler_records_equal_message_occurrences_streamed() {
    disable_tracing();
    equal_message_occurrences_case(true).await;
}

#[tokio::test]
async fn streamed_max_turns_handler_can_skip_history_with_session() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(scripted(vec![])));
    let session = ListSession::new();
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .session(session.clone())
            .error_handlers(max_turns_handler_result(
                RunErrorHandlerResult::new("fallback answer").without_history(),
            )),
        true,
    )
    .await;
    assert_eq!(out.final_output, Some(json!("fallback answer")));
    assert!(run_item_events(&out.events).is_empty());
    assert_eq!(session.types().await, vec!["user"]);
}

/// Python raises `asyncio.CancelledError` from the session save. The Rust equivalent of a
/// cancelled await is dropping the future while the save is pending.
async fn session_cancellation_case(streamed: bool) {
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(scripted(vec![])));
    let session = Arc::new(ListSession::with(false, true));
    let options = RunOptions::new(())
        .max_turns(Some(0))
        .session(session.clone())
        .error_handlers(max_turns_handler(json!("fallback answer")));
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent.clone(), "user_message", options);
        let mut published = Vec::new();
        {
            let mut stream = std::pin::pin!(result.stream_events());
            let _ = tokio::time::timeout(Duration::from_millis(300), async {
                while let Some(ev) = stream.next().await {
                    published.push(ev);
                }
            })
            .await;
        }
        result.cancel(CancelMode::Immediate);
        assert!(result.final_output().is_none());
        assert!(result.result().is_none());
        assert!(
            published
                .iter()
                .all(|ev| !matches!(ev, Ok(StreamEvent::RunItem { .. }))),
            "the fallback must not be published before it is persisted"
        );
    } else {
        let run = Runner::run_with_options(agent.clone(), "user_message", options);
        assert!(
            tokio::time::timeout(Duration::from_millis(300), run)
                .await
                .is_err()
        );
    }
    assert_eq!(session.types().await, vec!["user"]);
}

#[tokio::test]
async fn max_turns_handler_session_cancellation_does_not_publish_output_run() {
    disable_tracing();
    session_cancellation_case(false).await;
}

#[tokio::test]
async fn max_turns_handler_session_cancellation_does_not_publish_output_streamed() {
    disable_tracing();
    session_cancellation_case(true).await;
}

#[tokio::test]
async fn non_streamed_max_turns_handler_session_failure_does_not_record_output() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(scripted(vec![])));
    let session = Arc::new(ListSession::with(true, false));
    let out = run_either(
        &agent,
        "user_message",
        RunOptions::new(())
            .max_turns(Some(0))
            .session(session.clone())
            .error_handlers(max_turns_handler(json!("fallback answer"))),
        false,
    )
    .await;
    let err = out.error.expect("session failure surfaces");
    assert!(err.to_string().contains("session save failed"), "{err}");
    assert_eq!(session.types().await, vec!["user"]);
    let data = err.run_data().expect("run data attached");
    assert_eq!(ItemHelpers::text_message_outputs(&data.new_items), "");
}

// ---------------------------------------------------------------------------------------------
// Resumed runs
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum ResumeOutcome {
    Success,
    GuardrailError,
    SessionFailure,
}

async fn resumed_does_not_append_case(
    streamed: bool,
    outcome: ResumeOutcome,
    include_in_history: bool,
) {
    let session = Arc::new(ListSession::with(false, false));
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![assistant_message("first response")]])),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "first input",
        RunOptions::new(())
            .max_turns(Some(1))
            .session(session.clone()),
    )
    .await
    .unwrap();
    let state = first.to_state();
    let before = state.to_json().unwrap();

    let mut resumed_agent = Agent::<()>::new("test").with_model_instance(scripted(vec![]));
    if outcome == ResumeOutcome::GuardrailError {
        resumed_agent = resumed_agent.with_output_guardrail(OutputGuardrail::new(
            "fail",
            |_c, _a, _o: Value| async move {
                Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
            },
        ));
    }
    session
        .fail_message_save
        .store(outcome == ResumeOutcome::SessionFailure, Ordering::SeqCst);
    let mut handler_result = RunErrorHandlerResult::new("fallback answer");
    if !include_in_history {
        handler_result = handler_result.without_history();
    }
    let out = run_either(
        &Arc::new(resumed_agent),
        state.clone(),
        RunOptions::new(())
            .session(session.clone())
            .error_handlers(max_turns_handler_result(handler_result)),
        streamed,
    )
    .await;
    match outcome {
        ResumeOutcome::GuardrailError => {
            let err = out.error.expect("guardrail error");
            assert!(err.to_string().contains("guardrail failed"), "{err}");
        }
        ResumeOutcome::SessionFailure => {
            let err = out.error.expect("session failure");
            assert!(err.to_string().contains("session save failed"), "{err}");
        }
        ResumeOutcome::Success => {
            assert!(out.error.is_none(), "{:?}", out.error);
            assert_eq!(out.final_output, Some(json!("fallback answer")));
        }
    }
    let after = state.to_json().unwrap();
    assert_eq!(after["generated_items"], before["generated_items"]);
    assert_eq!(after["session_items"], before["session_items"]);
}

#[tokio::test]
async fn resumed_max_turns_handler_does_not_append_to_caller_state_items_matrix() {
    disable_tracing();
    for streamed in [false, true] {
        resumed_does_not_append_case(streamed, ResumeOutcome::Success, true).await;
        resumed_does_not_append_case(streamed, ResumeOutcome::Success, false).await;
        resumed_does_not_append_case(streamed, ResumeOutcome::GuardrailError, true).await;
        resumed_does_not_append_case(streamed, ResumeOutcome::SessionFailure, true).await;
    }
}

fn echo_guardrail(trip_on: Option<&'static str>) -> OutputGuardrail<()> {
    OutputGuardrail::new("echo", move |_c, _a, output: Value| async move {
        let trip = trip_on.is_some_and(|t| output == json!(t));
        Ok(GuardrailFunctionOutput::new(output, trip))
    })
}

async fn resumed_preserves_guardrail_results_case(streamed: bool) {
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![assistant_message("first response")]]))
            .with_output_guardrail(echo_guardrail(None)),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "first input",
        RunOptions::new(()).max_turns(Some(1)),
    )
    .await
    .unwrap();
    let out = run_either(
        &agent,
        first.to_state(),
        RunOptions::new(()).error_handlers(max_turns_handler(json!("fallback answer"))),
        streamed,
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(
        out.output_guardrail_infos,
        vec![json!("first response"), json!("fallback answer")]
    );
    let state = out.state.unwrap();
    assert_eq!(
        state
            .output_guardrail_results
            .iter()
            .map(|g| g.output.output_info.clone())
            .collect::<Vec<_>>(),
        vec![json!("first response"), json!("fallback answer")]
    );
}

#[tokio::test]
async fn resumed_max_turns_handler_preserves_output_guardrail_results_run() {
    disable_tracing();
    resumed_preserves_guardrail_results_case(false).await;
}

#[tokio::test]
async fn resumed_max_turns_handler_preserves_output_guardrail_results_streamed() {
    disable_tracing();
    resumed_preserves_guardrail_results_case(true).await;
}

async fn resumed_trip_preserves_current_case(streamed: bool) {
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![assistant_message("first response")]]))
            .with_output_guardrail(echo_guardrail(Some("fallback answer"))),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "first input",
        RunOptions::new(()).max_turns(Some(1)),
    )
    .await
    .unwrap();
    let state = first.to_state();
    let prior = state.output_guardrail_results.clone();
    assert_eq!(prior.len(), 1);
    let out = run_either(
        &agent,
        state.clone(),
        RunOptions::new(()).error_handlers(max_turns_handler(json!("fallback answer"))),
        streamed,
    )
    .await;
    match out.error {
        Some(AgentsError::OutputGuardrailTripwireTriggered {
            guardrail_result, ..
        }) => {
            assert_eq!(guardrail_result.agent_output, json!("fallback answer"));
            assert_eq!(
                guardrail_result.output.output_info,
                json!("fallback answer")
            );
        }
        other => panic!("expected tripwire, got {other:?}"),
    }
    assert_eq!(state.output_guardrail_results, prior);
    assert_eq!(prior[0].output.output_info, json!("first response"));
}

#[tokio::test]
async fn resumed_max_turns_trip_preserves_current_guardrail_result_run() {
    disable_tracing();
    resumed_trip_preserves_current_case(false).await;
}

#[tokio::test]
async fn resumed_max_turns_trip_preserves_current_guardrail_result_streamed() {
    disable_tracing();
    resumed_trip_preserves_current_case(true).await;
}

async fn resumed_preserves_checkpoint_case(streamed: bool) {
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![assistant_message("first response")]])),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "first input",
        RunOptions::new(()).max_turns(Some(2)),
    )
    .await
    .unwrap();
    let state = first.to_state();
    let before = state.to_json().unwrap();
    let continued = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![get_function_tool_call(
                "some_function",
                &json!({"a": "b"}).to_string(),
                None,
            )]]))
            .with_tool(get_function_tool("some_function", "result"))
            .with_output_guardrail(OutputGuardrail::new(
                "fail",
                |_c, _a, _o: Value| async move {
                    Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
                },
            )),
    );
    let out = run_either(
        &continued,
        state.clone(),
        RunOptions::new(()).error_handlers(max_turns_handler(json!("fallback answer"))),
        streamed,
    )
    .await;
    let err = out.error.expect("guardrail failure");
    assert!(err.to_string().contains("guardrail failed"), "{err}");
    let after = state.to_json().unwrap();
    assert_eq!(after["generated_items"], before["generated_items"]);
    assert_eq!(after["session_items"], before["session_items"]);
}

#[tokio::test]
async fn resumed_max_turns_handler_preserves_checkpoint_after_continuation_run() {
    disable_tracing();
    resumed_preserves_checkpoint_case(false).await;
}

#[tokio::test]
async fn resumed_max_turns_handler_preserves_checkpoint_after_continuation_streamed() {
    disable_tracing();
    resumed_preserves_checkpoint_case(true).await;
}
