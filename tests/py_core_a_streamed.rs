//! Ports of the behavioral streamed-runner tests in `test_agent_runner_streamed.py`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::{HandoffInputData, handoff};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::RunResultStreaming;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolNotFoundBehavior};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, OutputGuardrail, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn tool_returning(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

#[derive(Default)]
struct Streamed {
    events: Vec<StreamEvent<()>>,
    error: Option<AgentsError>,
}

impl Streamed {
    fn item_events(&self) -> Vec<(RunItemStreamEventName, &'static str)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::RunItem { name, item } => Some((*name, item.item_type())),
                _ => None,
            })
            .collect()
    }
}

async fn consume(stream: &mut RunResultStreaming<()>) -> Streamed {
    let mut out = Streamed::default();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        match event {
            Ok(e) => out.events.push(e),
            Err(e) => out.error = Some(e),
        }
    }
    out
}

async fn run_streamed(agent: &Agent<()>, input: &str) -> (RunResultStreaming<()>, Streamed) {
    let mut stream = Runner::run_streamed(agent, input);
    let out = consume(&mut stream).await;
    (stream, out)
}

// ------------------------------------------------------------------ basic runs

#[tokio::test]
async fn simple_first_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("first")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    let (stream, out) = run_streamed(&agent, "test").await;
    assert!(out.error.is_none());
    let result = stream.result().expect("complete");
    assert_eq!(result.input[0].text_content().as_deref(), Some("test"));
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.final_output_text().as_deref(), Some("first"));
    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(result.to_input_list().len(), 2);
}

#[tokio::test]
async fn empty_list_input_reaches_the_model() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let mut stream = Runner::run_streamed(&agent, Vec::<InputItem>::new());
    consume(&mut stream).await;
    assert!(model.last_call().unwrap().input.is_empty());
}

#[tokio::test]
async fn tool_call_runs() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("foo", r#"{"a":"b"}"#),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "tool_result"))
        .with_model_instance(model.clone());

    let (stream, _) = run_streamed(&agent, "user_message").await;
    let result = stream.result().unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(result.to_input_list().len(), 5);
}

#[tokio::test]
async fn tool_not_found_behavior_returns_error_to_the_model() {
    let model = setup();
    model.add_turn(vec![function_call("missing", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("known", "ok"))
        .with_model_instance(model.clone());
    let config = RunConfig {
        tool_not_found_behavior: ToolNotFoundBehavior::ReturnErrorToModel,
        ..Default::default()
    };
    let mut stream =
        Runner::run_streamed_with_options(&agent, "go", RunOptions::new(()).run_config(config));
    let out = consume(&mut stream).await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(
        stream.result().unwrap().final_output_text().as_deref(),
        Some("done")
    );
    let kinds = out.item_events();
    assert!(
        kinds
            .iter()
            .any(|(n, _)| *n == RunItemStreamEventName::ToolOutput)
    );
}

// --------------------------------------------------------------------- handoffs

#[tokio::test]
async fn handoffs_end_on_the_final_agent() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
    let agent_2 = Arc::new(Agent::<()>::new("two").with_model_instance(model.clone()));
    let agent_3 = Agent::<()>::new("three")
        .with_handoff(handoff(agent_1.clone()))
        .with_handoff(handoff(agent_2.clone()))
        .with_tool(tool_returning("some_function", "result"))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("some_function", r#"{"a":"b"}"#)]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![assistant_message("done")]);

    let (stream, out) = run_streamed(&agent_3, "user_message").await;

    let result = stream.result().unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.raw_responses.len(), 3);
    assert_eq!(result.to_input_list().len(), 7);
    assert_eq!(result.last_agent.name, "one");
    // The streamed result reports the agent that produced the final output.
    assert_eq!(stream.current_agent().name, "one");
    let updated: Vec<String> = out
        .events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::AgentUpdated { new_agent } => Some(new_agent.name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(updated, vec!["three".to_string(), "one".to_string()]);
}

#[tokio::test]
async fn handoff_filters_drop_new_items() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
    let agent_2 = Agent::<()>::new("two")
        .with_handoff(
            handoff(agent_1).with_input_filter(|data: HandoffInputData| HandoffInputData {
                pre_handoff_items: Vec::new(),
                new_items: Vec::new(),
                ..data
            }),
        )
        .with_model_instance(model.clone());
    model.add_turn(vec![
        assistant_message("1"),
        assistant_message("2"),
        function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![assistant_message("last")]);

    let (stream, _) = run_streamed(&agent_2, "user_message").await;
    let result = stream.result().unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("last"));
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(result.to_input_list().len(), 2);
}

#[tokio::test]
async fn a_handoff_call_streams_only_as_handoff_requested() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
    let agent = Agent::<()>::new("triage")
        .with_handoff(handoff(target))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_target", "{}")]);
    model.add_turn(vec![assistant_message("Done")]);

    let (_, out) = run_streamed(&agent, "Start").await;
    let kinds = out.item_events();
    assert_eq!(
        kinds
            .iter()
            .filter(|(_, ty)| *ty == "handoff_call_item")
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
        vec![RunItemStreamEventName::HandoffRequested]
    );
    assert!(
        !kinds
            .iter()
            .any(|(n, _)| *n == RunItemStreamEventName::ToolCalled)
    );
}

// --------------------------------------------------------------------- guardrails

fn tripping_output_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("trips", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    })
}

#[tokio::test]
async fn input_guardrail_tripwire_raises_from_the_stream() {
    let model = setup();
    model.add_turn(vec![assistant_message("first_test")]);
    let agent = Agent::<()>::new("test")
        .with_input_guardrail(InputGuardrail::new("trips", |_c, _a, _i| async {
            Ok(GuardrailFunctionOutput::new(Value::Null, true))
        }))
        .with_model_instance(model.clone());

    let (stream, out) = run_streamed(&agent, "user_message").await;
    assert!(matches!(
        out.error,
        Some(AgentsError::InputGuardrailTripwireTriggered { .. })
    ));
    assert!(stream.is_complete());
}

#[tokio::test]
async fn output_guardrail_tripwire_raises_from_the_stream() {
    let model = setup();
    model.add_turn(vec![assistant_message("first_test")]);
    let agent = Agent::<()>::new("test")
        .with_output_guardrail(tripping_output_guardrail())
        .with_model_instance(model.clone());

    let (stream, out) = run_streamed(&agent, "user_message").await;
    assert!(matches!(
        out.error,
        Some(AgentsError::OutputGuardrailTripwireTriggered { .. })
    ));
    assert!(stream.final_output().is_none());
    assert!(stream.is_complete());
}

#[tokio::test]
async fn output_guardrail_exception_leaves_no_final_output() {
    let model = setup();
    model.add_turn(vec![assistant_message("first_test")]);
    let agent = Agent::<()>::new("test")
        .with_output_guardrail(OutputGuardrail::new("fails", |_c, _a, _o: Value| async {
            Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
        }))
        .with_model_instance(model.clone());

    let (stream, out) = run_streamed(&agent, "user_message").await;
    let error = out.error.expect("the guardrail failure surfaces");
    assert!(error.to_string().contains("guardrail failed"), "{error}");
    assert!(stream.final_output().is_none());
    assert!(stream.is_complete());
}

/// A session whose appends fail once a flag flips, recording each attempt.
#[derive(Debug)]
struct FailingFinalTurnSession {
    inner: InMemorySession,
    armed: Arc<AtomicBool>,
    attempted: Arc<AtomicBool>,
}

#[async_trait]
impl Session for FailingFinalTurnSession {
    fn session_id(&self) -> &str {
        "failing"
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if self.armed.load(Ordering::SeqCst) {
            self.attempted.store(true, Ordering::SeqCst);
            return Err(AgentsError::other("session save failed"));
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

#[tokio::test]
async fn a_session_save_error_takes_precedence_over_an_output_guardrail_error() {
    let armed = Arc::new(AtomicBool::new(false));
    let attempted = Arc::new(AtomicBool::new(false));
    let session = Arc::new(FailingFinalTurnSession {
        inner: InMemorySession::new("failing"),
        armed: armed.clone(),
        attempted: attempted.clone(),
    });
    let flag = armed.clone();
    let model = setup();
    model.add_turn(vec![assistant_message("assistant-preamble")]);
    let agent = Agent::<()>::new("test")
        .with_output_guardrail(OutputGuardrail::new("fails", move |_c, _a, _o: Value| {
            let flag = flag.clone();
            async move {
                flag.store(true, Ordering::SeqCst);
                Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
            }
        }))
        .with_model_instance(model.clone());

    let mut stream =
        Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()).session(session));
    let out = consume(&mut stream).await;

    let error = out.error.expect("the run fails");
    assert!(error.to_string().contains("session save failed"), "{error}");
    assert!(attempted.load(Ordering::SeqCst), "the final turn is saved");
}

// -------------------------------------------------------------------- max turns

#[tokio::test]
async fn max_turns_still_emits_the_pending_tool_output_events() {
    let model = setup();
    model.add_turn(vec![function_call("test_tool", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("test_tool", "tool_result"))
        .with_model_instance(model.clone());

    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "Use test_tool",
        RunOptions::new(()).max_turns(Some(1)),
    );
    let out = consume(&mut stream).await;

    assert!(matches!(
        out.error,
        Some(AgentsError::MaxTurnsExceeded { .. })
    ));
    let types: Vec<&str> = out.item_events().into_iter().map(|(_, t)| t).collect();
    assert!(types.contains(&"tool_call_item"), "{types:?}");
    assert!(types.contains(&"tool_call_output_item"), "{types:?}");
}

// --------------------------------------------------------------- tool guardrails

#[tokio::test]
async fn tool_guardrail_results_survive_a_handoff() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
    let guarded = tool_returning("guarded", "ok")
        .with_input_guardrail(ToolInputGuardrail::new("allow", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("seen")))
        }));
    let agent = Agent::<()>::new("source")
        .with_tool(guarded)
        .with_handoff(handoff(target))
        .with_model_instance(model.clone());
    model.add_turn(vec![
        function_call_with_id("guarded", "{}", "call_g"),
        function_call("transfer_to_target", "{}"),
    ]);
    model.add_turn(vec![assistant_message("done")]);

    let (stream, out) = run_streamed(&agent, "go").await;
    assert!(out.error.is_none(), "{:?}", out.error);
    let result = stream.result().unwrap();
    assert_eq!(result.tool_input_guardrail_results.len(), 1);
    assert_eq!(
        result.tool_input_guardrail_results[0].output.output_info,
        json!("seen")
    );
}

// ----------------------------------------------------------------- event ordering

#[tokio::test]
async fn streaming_events_arrive_in_order() {
    let model = setup();
    model.add_turn(vec![function_call("foo", r#"{"a":"b"}"#)]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool_returning("foo", "foo_result"))
        .with_model_instance(model.clone());

    let (_, out) = run_streamed(&agent, "go").await;

    let names: Vec<RunItemStreamEventName> =
        out.item_events().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec![
            RunItemStreamEventName::ToolCalled,
            RunItemStreamEventName::ToolOutput,
            RunItemStreamEventName::MessageOutputCreated,
        ]
    );
    // The agent is announced before anything else.
    assert!(matches!(
        out.events.first(),
        Some(StreamEvent::AgentUpdated { .. })
    ));
}

#[tokio::test]
async fn a_streamed_message_is_a_message_output_item() {
    let model = setup();
    model.add_turn(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let (stream, out) = run_streamed(&agent, "go").await;
    let items: Vec<&str> = out.item_events().into_iter().map(|(_, t)| t).collect();
    assert_eq!(items, vec!["message_output_item"]);
    assert!(matches!(
        stream.result().unwrap().new_items[0].kind,
        RunItemKind::MessageOutput { .. }
    ));
}
