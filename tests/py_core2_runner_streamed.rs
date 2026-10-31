//! Ports of the streamed-only tests in `tests/test_agent_runner_streamed.py`: terminal stream
//! events, raw-response metadata, retry usage accounting, session persistence of streamed
//! runs, streamed resumes and tool-guardrail result accounting, plus the streamed
//! previous-response-id / conversation-id tests of `tests/test_agent_runner.py`.
//!
//! Tests that exist in both a non-streamed and a streamed form are ported once (both modes)
//! in the sibling `py_core2_runner_*` files; this file covers what only the streamed runner
//! can show.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::items::{OutputItem, RunInput};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use openai_agents_sdk::result::{RunResult, RunResultStreaming};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT, RunConfig};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    FakeModel, ModelStep, ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, OutputGuardrail,
    Runner, handoff,
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

fn tool(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

#[derive(Default)]
struct Consumed {
    events: Vec<StreamEvent<()>>,
    error: Option<AgentsError>,
}

async fn consume(stream: &mut RunResultStreaming<()>) -> Consumed {
    let mut out = Consumed::default();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        match event {
            Ok(e) => out.events.push(e),
            Err(e) => out.error = Some(e),
        }
    }
    out
}

async fn streamed(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
) -> (RunResultStreaming<()>, Consumed) {
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let out = consume(&mut stream).await;
    (stream, out)
}

async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed_mode: bool,
) -> Result<RunResult<()>, AgentsError> {
    if !streamed_mode {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let (stream, out) = streamed(agent, input, options).await;
    match out.error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

fn retry_settings() -> ModelSettings {
    ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: 1,
                backoff: ModelRetryBackoffSettings {
                    initial_delay: 0.0,
                    max_delay: 0.0,
                    multiplier: 1.0,
                    jitter: false,
                },
                ..Default::default()
            }
            .with_policy(openai_agents_sdk::retry::retry_policies::http_status(vec![
                500,
            ])),
        ),
        ..Default::default()
    }
}

#[derive(Debug)]
struct ListSession {
    inner: InMemorySession,
    remote: bool,
    pops: Mutex<usize>,
    writes: Mutex<Vec<Vec<Value>>>,
}

impl ListSession {
    async fn new(history: Vec<Value>, remote: bool) -> Arc<Self> {
        let inner = InMemorySession::new("list");
        inner
            .add_items(history.into_iter().map(InputItem).collect())
            .await
            .unwrap();
        Arc::new(ListSession {
            inner,
            remote,
            pops: Mutex::new(0),
            writes: Mutex::new(Vec::new()),
        })
    }
    async fn all(&self) -> Vec<Value> {
        self.inner
            .get_items(None)
            .await
            .unwrap()
            .into_iter()
            .map(|i| i.0)
            .collect()
    }
}

#[async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        "list"
    }
    fn manages_history_remotely(&self) -> bool {
        self.remote
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        self.writes
            .lock()
            .unwrap()
            .push(items.iter().map(|i| i.0.clone()).collect());
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        *self.pops.lock().unwrap() += 1;
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

// --------------------------------------------------------------- terminal stream events

fn terminal_payload(event_type: &str) -> Value {
    json!({
        "type": event_type,
        "sequence_number": 0,
        "response": {
            "id": "resp-partial",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "partial final", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        }
    })
}

async fn run_terminal(events: Vec<Value>) -> (RunResultStreaming<()>, Consumed) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::with_steps(vec![ScriptedStep::stream(events)]).unwrap());
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model));
    streamed(&agent, "test", RunOptions::new(())).await
}

/// Python: `test_streamed_run_rejects_failed_terminal_response_payload_events`
/// (`response.incomplete`, `response.failed`), `test_streamed_run_rejects_response_error_terminal_event`
/// and the event half of `test_streamed_terminal_failure_closes_the_model_stream` (4 cases).
#[tokio::test]
async fn a_failed_terminal_event_fails_the_streamed_run() {
    let cases = vec![
        (
            "response.incomplete",
            terminal_payload("response.incomplete"),
        ),
        ("response.failed", terminal_payload("response.failed")),
        (
            "error",
            json!({"type": "error", "code": "invalid_request_error", "message": "bad request",
                   "param": null, "sequence_number": 0}),
        ),
        (
            "response.error",
            json!({"type": "response.error", "code": null, "message": "boom", "param": null,
                   "sequence_number": 0}),
        ),
    ];
    for (event_type, event) in cases {
        let (stream, out) = run_terminal(vec![event]).await;
        let err = out
            .error
            .unwrap_or_else(|| panic!("{event_type} must fail"));
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{event_type}: {err:?}"
        );
        assert!(err.to_string().contains(event_type), "{event_type}: {err}");
        assert_eq!(out.events.len(), 2, "{event_type}");
        assert!(matches!(out.events[0], StreamEvent::AgentUpdated { .. }));
        match &out.events[1] {
            StreamEvent::RawResponse(data) => assert_eq!(data["type"], event_type),
            other => panic!("{event_type}: {other:?}"),
        }
        assert!(stream.final_output().is_none());
    }
}

/// Python: `test_streamed_run_exposes_request_id_on_raw_responses` (`preserve_raw_usage`
/// None / False / True).
#[tokio::test]
async fn raw_responses_carry_the_request_id_and_optional_raw_usage() {
    for preserve in [None, Some(false), Some(true)] {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let raw_usage = json!({"input_tokens": 3, "input_tokens_details": {"cached_tokens": 0}});
        let step = ScriptedStep {
            output: vec![assistant_message("partial final")],
            request_id: Some("req_streamed_result_123".into()),
            raw_usage: Some(raw_usage.clone()),
            response_id: Some("resp-partial".into()),
            usage: Some(Usage {
                requests: 1,
                input_tokens: 3,
                ..Default::default()
            }),
            ..Default::default()
        };
        let model = Arc::new(ScriptedModel::with_steps(vec![step]).unwrap());
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model)
                .with_model_settings(ModelSettings {
                    preserve_raw_usage: preserve,
                    ..Default::default()
                }),
        );
        let (stream, out) = streamed(&agent, "test", RunOptions::new(())).await;
        assert!(out.error.is_none(), "{:?}", out.error);
        let r = stream.result().unwrap();
        assert_eq!(r.raw_responses.len(), 1);
        assert_eq!(
            r.raw_responses[0].request_id.as_deref(),
            Some("req_streamed_result_123")
        );
        // Rust's raw usage is the provider usage object of the terminal event; Python attaches
        // the same snapshot as a separate attribute.
        let raw = &r.raw_responses[0].raw_usage;
        assert_eq!(raw.is_some(), preserve == Some(true), "{preserve:?}");
        if let Some(raw) = raw {
            assert_eq!(raw["input_tokens"], raw_usage["input_tokens"]);
            assert_eq!(
                raw["input_tokens_details"],
                raw_usage["input_tokens_details"]
            );
        }
    }
}

// ------------------------------------------------------------------- retry accounting

/// Python: `test_streamed_run_preserves_request_usage_entries_after_retry` (and the
/// conversation-locked variant, which retries through the same accounting path).
#[tokio::test]
async fn a_streamed_retry_keeps_one_usage_entry_per_request() {
    let model = setup();
    model.set_usage(Usage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        ..Default::default()
    });
    model.enqueue(ModelStep::ApiError(500));
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_model_settings(retry_settings()),
    );
    let (stream, out) = streamed(&agent, "test", RunOptions::new(())).await;
    assert!(out.error.is_none(), "{:?}", out.error);
    let usage = stream.result().unwrap().usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].input_tokens, 10);
    assert_eq!(usage.request_usage_entries[1].output_tokens, 5);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 15);
}

/// Python: `test_streamed_run_counts_retry_attempts_when_terminal_usage_missing`.
#[tokio::test]
async fn a_streamed_retry_counts_attempts_without_terminal_usage() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let completed_without_usage = json!({"type": "response.completed", "sequence_number": 0,
        "response": {"id": "r", "output": [{"type": "message", "id": "m", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]}]}});
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::raise_error(AgentsError::ApiError {
                status: 500,
                message: "connection error".into(),
                body: None,
                request_id: None,
                headers: None,
            }),
            ScriptedStep::stream(vec![completed_without_usage]),
        ])
        .unwrap(),
    );
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_model_settings(retry_settings()),
    );
    let (stream, out) = streamed(&agent, "test", RunOptions::new(())).await;
    assert!(out.error.is_none(), "{:?}", out.error);
    let usage = stream.result().unwrap().usage();
    assert_eq!(model.calls().len(), 2);
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 0);
}

/// Python: `test_streamed_model_retry_does_not_rewind_committed_session_input`.
#[tokio::test]
async fn a_streamed_retry_does_not_rewind_the_session() {
    let model = setup();
    model.enqueue(ModelStep::ApiError(500));
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_model_settings(retry_settings()),
    );
    let session =
        ListSession::new(vec![json!({"role": "user", "content": "previous"})], false).await;
    let (_, out) = streamed(&agent, "test", RunOptions::new(()).session(session.clone())).await;
    assert!(out.error.is_none());
    let roles: Vec<String> = session
        .all()
        .await
        .iter()
        .filter_map(|i| i["role"].as_str().map(str::to_string))
        .collect();
    assert_eq!(roles, vec!["user", "user", "assistant"]);
    assert_eq!(*session.pops.lock().unwrap(), 0);
}

// -------------------------------------------------------------- streamed session writes

/// Python: `test_streamed_run_again_persists_tool_items_to_session`.
#[tokio::test]
async fn a_streamed_run_again_persists_tool_items() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "foo",
        r#"{"a":"b"}"#,
        "call-session-run-again",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("foo", "tool_result")),
    );
    let session = ListSession::new(vec![], false).await;
    streamed(
        &agent,
        "user_message",
        RunOptions::new(()).session(session.clone()),
    )
    .await;
    let saved = session.all().await;
    for ty in ["function_call", "function_call_output"] {
        assert!(
            saved
                .iter()
                .any(|i| i["type"] == ty && i["call_id"] == "call-session-run-again"),
            "{ty}"
        );
    }
}

/// Python: `test_input_guardrail_streamed_does_not_save_assistant_message_to_session`,
/// `test_input_guardrail_streamed_persists_user_input_for_sequential_guardrail`,
/// `test_input_guardrail_streamed_persists_user_input_for_async_sequential_guardrail`.
#[tokio::test]
async fn a_streamed_input_guardrail_trip_saves_only_the_user_input() {
    for (parallel, delay) in [(true, 10u64), (false, 0), (false, 1)] {
        let model = setup();
        model.add_turn(vec![assistant_message("should_not_be_saved")]);
        let mut g = InputGuardrail::new("trip", move |_c, _a, _i| async move {
            if delay > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            }
            Ok(GuardrailFunctionOutput::new(Value::Null, true))
        });
        g.run_in_parallel = parallel;
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_input_guardrail(g),
        );
        let session = ListSession::new(vec![], false).await;
        let (_, out) = streamed(
            &agent,
            "user_message",
            RunOptions::new(()).session(session.clone()),
        )
        .await;
        assert!(matches!(
            out.error,
            Some(AgentsError::InputGuardrailTripwireTriggered { .. })
        ));
        let saved = session.all().await;
        assert_eq!(saved.len(), 1, "parallel={parallel}: {saved:?}");
        assert_eq!(saved[0]["role"], "user");
    }
}

/// Python: `test_stream_input_persistence_saves_only_new_turn_input`.
#[tokio::test]
async fn each_streamed_turn_persists_only_its_new_input() {
    let model = setup();
    model.add_turn(vec![assistant_message("first")]);
    model.add_turn(vec![assistant_message("second")]);
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let session = ListSession::new(vec![], false).await;
    let config = || {
        let mut c = RunConfig::default();
        c.session_input_callback = Some(Arc::new(|existing, new| {
            let mut out = existing;
            out.extend(new);
            Box::pin(async move { Ok(out) })
        }));
        c
    };
    for text in ["hello", "next"] {
        streamed(
            &agent,
            vec![InputItem::user(text)],
            RunOptions::new(())
                .session(session.clone())
                .run_config(config()),
        )
        .await;
    }
    let user_saves: Vec<Value> = session
        .writes
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .filter(|i| i["role"] == "user")
        .cloned()
        .collect();
    let texts: Vec<&str> = user_saves
        .iter()
        .filter_map(|i| i["content"].as_str())
        .collect();
    assert_eq!(texts, vec!["hello", "next"]);
}

/// Python: `test_slow_input_guardrail_still_raises_exception_streamed`,
/// `test_run_input_guardrail_tripwire_triggered_causes_exception_streamed`.
#[tokio::test]
async fn a_slow_or_run_level_input_guardrail_still_trips_the_stream() {
    for run_level in [false, true] {
        let model = setup();
        model.add_turn(vec![assistant_message("ok")]);
        let g = InputGuardrail::new("slow", |_c, _a, _i| async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Ok(GuardrailFunctionOutput::new(Value::Null, true))
        });
        let mut agent = Agent::<()>::new("test").with_model_instance(model.clone());
        let mut config = RunConfig::default();
        if run_level {
            config.input_guardrails = vec![g];
        } else {
            agent = agent.with_input_guardrail(g);
        }
        let (_, out) = streamed(
            &Arc::new(agent),
            "user_message",
            RunOptions::new(()).run_config(config),
        )
        .await;
        assert!(
            matches!(
                out.error,
                Some(AgentsError::InputGuardrailTripwireTriggered { .. })
            ),
            "run_level={run_level}"
        );
    }
}

/// Python: `test_run_output_guardrail_tripwire_triggered_causes_exception_streamed`,
/// `test_output_guardrail_tripwire_raises_from_run_loop_task_before_stream_consumption`
/// (`finish()` drives the run without consuming events).
#[tokio::test]
async fn a_run_level_output_guardrail_trips_the_stream() {
    let model = setup();
    model.add_turn(vec![assistant_message("first_test")]);
    model.add_turn(vec![assistant_message("first_test")]);
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let mut config = RunConfig::default();
    config.output_guardrails = vec![OutputGuardrail::new("trip", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, true))
    })];
    let (_, out) = streamed(
        &agent,
        "user_message",
        RunOptions::new(()).run_config(config),
    )
    .await;
    assert!(matches!(
        out.error,
        Some(AgentsError::OutputGuardrailTripwireTriggered { .. })
    ));

    let tripping = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_output_guardrail(OutputGuardrail::new("trip", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            })),
    );
    let mut stream = Runner::run_streamed(tripping, "user_message");
    let err = stream.finish().await.unwrap_err();
    assert!(matches!(
        err,
        AgentsError::OutputGuardrailTripwireTriggered { .. }
    ));
    assert!(stream.final_output().is_none());
    assert!(stream.is_complete());
}

// ------------------------------------------------------------------ streamed events

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

/// Python: `test_streaming_events`.
#[tokio::test]
async fn streaming_events_report_every_item_and_agent_switch() {
    let model = setup();
    let agent_1 = Arc::new(
        Agent::<()>::new("one")
            .with_tool(tool("bar", "bar_result"))
            .with_output_type::<Foo>()
            .with_model_instance(model.clone()),
    );
    let agent_2 = Arc::new(
        Agent::<()>::new("two")
            .with_tool(tool("foo", "foo_result"))
            .with_handoff(handoff(agent_1))
            .with_model_instance(model.clone()),
    );
    model.add_turn(vec![function_call_with_id(
        "foo",
        r#"{"bar":"baz"}"#,
        "call_foo",
    )]);
    model.add_turn(vec![
        assistant_message("a_message"),
        openai_agents_sdk::testing::function_call("transfer_to_one", "{}"),
    ]);
    model.add_turn(vec![function_call_with_id(
        "bar",
        r#"{"bar":"baz"}"#,
        "call_bar",
    )]);
    model.add_turn(vec![assistant_message(r#"{"bar":"baz"}"#)]);
    let mut config = RunConfig::default();
    config.nest_handoff_history = true;
    let (stream, out) = streamed(
        &agent_2,
        vec![
            InputItem::user("user_message"),
            InputItem::user("another_message"),
        ],
        RunOptions::new(()).run_config(config),
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    let r = stream.result().unwrap();
    assert_eq!(
        r.final_output_as::<Foo>().unwrap(),
        Foo { bar: "baz".into() }
    );
    assert_eq!(r.raw_responses.len(), 4);
    assert_eq!(r.to_input_list().len(), 10);
    assert_eq!(
        r.to_input_list_with(openai_agents_sdk::result::ToInputListMode::Normalized)
            .len(),
        6
    );
    assert_eq!(r.last_agent.name, "one");
    let items: Vec<&'static str> = out
        .events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::RunItem { item, .. } => Some(item.item_type()),
            _ => None,
        })
        .collect();
    assert_eq!(items.len(), 8, "{items:?}");
    let count = |t: &str| items.iter().filter(|x| **x == t).count();
    assert_eq!(count("tool_call_item"), 2);
    assert_eq!(count("tool_call_output_item"), 2);
    assert_eq!(count("message_output_item"), 2);
    assert_eq!(count("handoff_call_item"), 1);
    assert_eq!(count("handoff_output_item"), 1);
    let agents: Vec<String> = out
        .events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::AgentUpdated { new_agent } => Some(new_agent.name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(agents, vec!["two", "one"]);
}

// --------------------------------------------------------------- streamed resumes

fn approval_agent(model: &Arc<FakeModel>) -> Arc<Agent<()>> {
    Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("test_tool", "tool_result").with_needs_approval(true)),
    )
}

/// Python: `test_streaming_hitl_resume_with_approved_tools`,
/// `test_streaming_resume_with_session_does_not_duplicate_items`,
/// `test_streaming_resume_persists_tool_outputs_on_run_again`.
#[tokio::test]
async fn a_streamed_resume_runs_the_approved_tool_once() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "test_tool",
        "{}",
        "call-resume",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = approval_agent(&model);
    let session = ListSession::new(vec![], false).await;
    let (first, _) = streamed(
        &agent,
        "Use test_tool",
        RunOptions::new(()).session(session.clone()),
    )
    .await;
    assert_eq!(first.interruptions().len(), 1);
    let mut state = first.to_state().unwrap();
    state.approve(&first.interruptions()[0], false);
    let (resumed, out) =
        streamed(&agent, state, RunOptions::new(()).session(session.clone())).await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let saved = session.all().await;
    for ty in ["function_call", "function_call_output"] {
        assert_eq!(
            saved
                .iter()
                .filter(|i| i["type"] == ty && i["call_id"] == "call-resume")
                .count(),
            1,
            "{ty}"
        );
    }
}

/// Python: `test_streaming_hitl_resume_enforces_max_turns`.
#[tokio::test]
async fn a_streamed_resume_enforces_max_turns() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("test_tool", "{}", "c1")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = approval_agent(&model);
    let (first, _) = streamed(
        &agent,
        "Use test_tool",
        RunOptions::new(()).max_turns(Some(1)),
    )
    .await;
    let mut state = first.to_state().unwrap();
    state.approve(&first.interruptions()[0], false);
    let (_, out) = streamed(&agent, state, RunOptions::new(())).await;
    assert!(
        matches!(out.error, Some(AgentsError::MaxTurnsExceeded { .. })),
        "{:?}",
        out.error
    );
}

/// Python: `test_streaming_hitl_server_conversation_tracker_priming`.
#[tokio::test]
async fn a_streamed_resume_keeps_the_server_conversation() {
    let model = setup();
    model.add_turn(vec![assistant_message("First response")]);
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let options = || {
        RunOptions::new(())
            .conversation_id("conv123")
            .previous_response_id("resp123")
    };
    let (first, _) = streamed(&agent, "test", options()).await;
    let mut state = first.to_state().unwrap();
    state.final_output = None;
    model.add_turn(vec![assistant_message("Second response")]);
    let (second, out) = streamed(&agent, state, options()).await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(
        second.final_output_text().as_deref(),
        Some("Second response")
    );
    assert!(!second.result().unwrap().new_items.is_empty());
}

// ------------------------------------------------------ approval siblings and sessions

/// Python: `test_run_llm_again_approval_persists_completed_sibling[non_streamed|streamed]`.
#[tokio::test]
async fn a_completed_sibling_of_an_approval_is_persisted() {
    for streamed_mode in [false, true] {
        let model = setup();
        model.add_turn(vec![
            function_call_with_id("approval_tool", "{}", "call-approved"),
            function_call_with_id("sibling_tool", "{}", "call-sibling"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let recorder = |name: &'static str, output: &'static str| {
            let log = log.clone();
            FunctionTool::<()>::new(name, "t", move |_c, _a: AnyArgs| {
                let log = log.clone();
                async move {
                    log.lock().unwrap().push(name);
                    Ok(output.to_string())
                }
            })
        };
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(recorder("approval_tool", "approved-output").with_needs_approval(true))
                .with_tool(recorder("sibling_tool", "sibling-output"))
                .with_output_guardrail(OutputGuardrail::new("pass", |_c, _a, _o: Value| async {
                    Ok(GuardrailFunctionOutput::new(Value::Null, false))
                })),
        );
        let session = ListSession::new(vec![], false).await;
        let first = run_mode(
            &agent,
            "Use both tools",
            RunOptions::new(()).session(session.clone()),
            streamed_mode,
        )
        .await
        .unwrap();
        assert_eq!(first.interruptions.len(), 1);
        assert_eq!(*log.lock().unwrap(), vec!["sibling_tool"]);
        let saved = session.all().await;
        let sibling: Vec<&Value> = saved
            .iter()
            .filter(|i| i["call_id"] == "call-sibling")
            .collect();
        assert_eq!(
            sibling
                .iter()
                .map(|i| i["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["function_call", "function_call_output"],
            "streamed={streamed_mode}"
        );
        assert_eq!(sibling[1]["output"], "sibling-output");
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let resumed = run_mode(
            &agent,
            state,
            RunOptions::new(()).session(session.clone()),
            streamed_mode,
        )
        .await
        .unwrap();
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        assert_eq!(*log.lock().unwrap(), vec!["sibling_tool", "approval_tool"]);
        let saved = session.all().await;
        for call_id in ["call-sibling", "call-approved"] {
            let types: Vec<&str> = saved
                .iter()
                .filter(|i| i["call_id"] == call_id)
                .filter_map(|i| i["type"].as_str())
                .collect();
            assert_eq!(
                types,
                vec!["function_call", "function_call_output"],
                "{call_id}"
            );
        }
    }
}

fn terminal_behaviors() -> Vec<(&'static str, ToolUseBehavior<()>)> {
    use openai_agents_sdk::agent::{StopAtTools, ToolsToFinalOutputResult};
    vec![
        ("first", ToolUseBehavior::StopOnFirstTool),
        (
            "named",
            ToolUseBehavior::StopAtTools(StopAtTools {
                stop_at_tool_names: vec!["approval_tool".into()],
            }),
        ),
        (
            "custom",
            ToolUseBehavior::Custom(Arc::new(|_ctx, results| {
                let output = results.first().map(|r| r.output.to_json_value());
                Box::pin(async move {
                    Ok(ToolsToFinalOutputResult {
                        is_final_output: true,
                        final_output: output,
                    })
                })
            })),
        ),
    ]
}

fn sibling_agent(model: &Arc<FakeModel>, behavior: Option<ToolUseBehavior<()>>) -> Agent<()> {
    let mut agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool("approval_tool", "approved-output").with_needs_approval(true))
        .with_tool(tool("sibling_tool", "sibling-output"))
        .with_output_guardrail(OutputGuardrail::new("pass", |_c, _a, _o: Value| async {
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }));
    if let Some(b) = behavior {
        agent = agent.with_tool_use_behavior(b);
    }
    agent
}

fn sibling_turn(model: &Arc<FakeModel>) {
    model.add_turn(vec![
        function_call_with_id("approval_tool", "{}", "call-approved"),
        function_call_with_id("sibling_tool", "{}", "call-sibling"),
    ]);
}

/// Python: `test_terminal_behaviors_defer_completed_approval_siblings` (`mode` x behavior).
#[tokio::test]
async fn terminal_behaviors_hold_back_completed_approval_siblings() {
    for streamed_mode in [false, true] {
        for (name, behavior) in terminal_behaviors() {
            let model = setup();
            sibling_turn(&model);
            let agent = Arc::new(sibling_agent(&model, Some(behavior)));
            let session = ListSession::new(vec![], false).await;
            let r = run_mode(
                &agent,
                "Use both tools",
                RunOptions::new(()).session(session.clone()),
                streamed_mode,
            )
            .await
            .unwrap();
            assert_eq!(r.interruptions.len(), 1, "{name}");
            assert_eq!(
                r.to_state().session_saved_items,
                0,
                "{name} streamed={streamed_mode}"
            );
            assert!(
                !json!(session.all().await)
                    .to_string()
                    .contains("sibling-output"),
                "{name} streamed={streamed_mode}"
            );
        }
    }
}

/// Python: `test_persisted_run_llm_again_checkpoint_rejects_terminal_behavior_change`.
#[tokio::test]
async fn a_persisted_checkpoint_rejects_a_terminal_behavior_change() {
    for streamed_mode in [false, true] {
        for (name, behavior) in terminal_behaviors() {
            let model = setup();
            sibling_turn(&model);
            let base = sibling_agent(&model, None);
            let session = ListSession::new(vec![], false).await;
            let first = run_mode(
                &Arc::new(base.clone()),
                "Use both tools",
                RunOptions::new(()).session(session.clone()),
                streamed_mode,
            )
            .await
            .unwrap();
            let mut state = first.to_state();
            assert!(state.session_saved_items > 0);
            state.approve(&first.interruptions[0], false);
            let changed = Arc::new(base.with_tool_use_behavior(behavior));
            let err = run_mode(
                &changed,
                state,
                RunOptions::new(()).session(session.clone()),
                streamed_mode,
            )
            .await
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("after current-turn items were persisted"),
                "{name} streamed={streamed_mode}: {err}"
            );
        }
    }
}

/// Python: `test_resumed_approved_tool_final_persists_complete_post_verdict_batch`
/// (`mode` x `outcome`).
#[tokio::test]
async fn a_resumed_approved_terminal_tool_persists_its_verdict_batch() {
    for streamed_mode in [false, true] {
        for outcome in ["passes", "trips", "error"] {
            let label = format!("{outcome} streamed={streamed_mode}");
            let model = setup();
            model.add_turn(vec![function_call_with_id(
                "approval_tool",
                "{}",
                "call-approved",
            )]);
            let state_outcome = Arc::new(Mutex::new(outcome.to_string()));
            let so = state_outcome.clone();
            let agent = Arc::new(
                Agent::<()>::new("test")
                    .with_model_instance(model.clone())
                    .with_tool(
                        tool("approval_tool", "approved-result")
                            .with_needs_approval(true)
                            .with_output_guardrail(ToolOutputGuardrail::new(
                                "record",
                                |d| async move {
                                    Ok(ToolGuardrailFunctionOutput::allow()
                                        .with_output_info(d.output.to_json_value()))
                                },
                            )),
                    )
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                    .with_output_guardrail(OutputGuardrail::new("g", move |_c, _a, _o: Value| {
                        let o = so.lock().unwrap().clone();
                        async move {
                            if o == "error" {
                                return Err(AgentsError::other("guardrail failed"));
                            }
                            Ok(GuardrailFunctionOutput::new(Value::Null, o == "trips"))
                        }
                    })),
            );
            let session = ListSession::new(vec![], false).await;
            let first = run_mode(
                &agent,
                "Use approval_tool",
                RunOptions::new(()).session(session.clone()),
                streamed_mode,
            )
            .await
            .unwrap();
            let mut state = first.to_state();
            state.approve(&first.interruptions[0], false);
            let outcome_result = run_mode(
                &agent,
                state,
                RunOptions::new(()).session(session.clone()),
                streamed_mode,
            )
            .await;
            match outcome {
                "trips" => assert!(
                    matches!(
                        outcome_result,
                        Err(AgentsError::OutputGuardrailTripwireTriggered { .. })
                    ),
                    "{label}"
                ),
                "error" => assert!(
                    outcome_result
                        .unwrap_err()
                        .to_string()
                        .contains("guardrail failed"),
                    "{label}"
                ),
                _ => assert_eq!(
                    outcome_result.unwrap().final_output_text().as_deref(),
                    Some("approved-result")
                ),
            }
            let saved = session.all().await;
            let kinds: Vec<&str> = saved
                .iter()
                .map(|i| {
                    i["type"]
                        .as_str()
                        .or_else(|| i["role"].as_str())
                        .unwrap_or("?")
                })
                .collect();
            assert_eq!(
                kinds,
                vec!["user", "function_call", "function_call_output"],
                "{label}"
            );
            let expected = if outcome == "trips" {
                OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT
            } else {
                "approved-result"
            };
            assert_eq!(saved[2]["output"], expected, "{label}");
            if outcome == "trips" {
                *state_outcome.lock().unwrap() = "passes".into();
                model.add_turn(vec![assistant_message("done")]);
                let next = run_mode(
                    &agent,
                    "Continue",
                    RunOptions::new(()).session(session.clone()),
                    streamed_mode,
                )
                .await
                .unwrap();
                assert_eq!(next.final_output_text().as_deref(), Some("done"));
                let input: Vec<Value> = model
                    .calls()
                    .last()
                    .unwrap()
                    .input
                    .iter()
                    .map(|i| i.0.clone())
                    .collect();
                let replayed: Vec<&Value> = input
                    .iter()
                    .filter(|i| i["call_id"] == "call-approved")
                    .collect();
                assert_eq!(replayed.len(), 2, "{label}");
                assert_eq!(replayed[1]["output"], OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT);
                assert!(!json!(input).to_string().contains("approved-result"));
            }
        }
    }
}

/// Python: `test_legacy_approval_checkpoint_uses_current_session_ownership`
/// (`mode` x `serialized` x `attach_session`).
#[tokio::test]
async fn a_checkpoint_with_persisted_items_follows_the_resume_session() {
    for streamed_mode in [false, true] {
        for serialized in [false, true] {
            for attach in [false, true] {
                let label = format!("streamed={streamed_mode} ser={serialized} attach={attach}");
                let model = setup();
                model.add_turn(vec![function_call_with_id(
                    "approval_tool",
                    "{}",
                    "call-approved",
                )]);
                let log = Arc::new(Mutex::new(0usize));
                let l2 = log.clone();
                let base = Agent::<()>::new("test")
                    .with_model_instance(model.clone())
                    .with_tool(
                        FunctionTool::<()>::new("approval_tool", "t", move |_c, _a: AnyArgs| {
                            let l2 = l2.clone();
                            async move {
                                *l2.lock().unwrap() += 1;
                                Ok("approved-result".to_string())
                            }
                        })
                        .with_needs_approval(true),
                    )
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
                let session = ListSession::new(vec![], false).await;
                let first = Runner::run_with_options(
                    base.clone(),
                    "Use approval_tool",
                    RunOptions::new(()).session(session.clone()),
                )
                .await
                .unwrap();
                let mut state = first.to_state();
                assert!(state.session_saved_items > 0);
                if serialized {
                    state =
                        openai_agents_sdk::RunState::from_json(state.to_json().unwrap()).unwrap();
                }
                state.approve(&first.interruptions[0], false);
                let agent = Arc::new(base.with_output_guardrail(OutputGuardrail::new(
                    "pass",
                    |_c, _a, _o: Value| async {
                        Ok(GuardrailFunctionOutput::new(Value::Null, false))
                    },
                )));
                let options = if attach {
                    RunOptions::new(()).session(session.clone())
                } else {
                    RunOptions::new(())
                };
                let outcome = run_mode(&agent, state, options, streamed_mode).await;
                if attach {
                    let err = outcome.unwrap_err();
                    assert!(
                        err.to_string()
                            .contains("after current-turn items were persisted"),
                        "{label}: {err}"
                    );
                    assert_eq!(*log.lock().unwrap(), 0, "{label}");
                } else {
                    let r = outcome.unwrap();
                    assert_eq!(
                        r.final_output_text().as_deref(),
                        Some("approved-result"),
                        "{label}"
                    );
                    assert_eq!(*log.lock().unwrap(), 1, "{label}");
                }
            }
        }
    }
}

// --------------------------------------------------------- tool guardrail accounting

fn guarded_tool() -> FunctionTool<()> {
    tool("guarded_tool", "tool-result")
        .with_input_guardrail(ToolInputGuardrail::new("in", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("input-checked")))
        }))
        .with_output_guardrail(ToolOutputGuardrail::new("out", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("output-checked")))
        }))
}

/// Python: `test_streamed_run_reports_tool_guardrail_results`,
/// `test_streamed_tool_guardrail_results_persist_into_run_state`.
#[tokio::test]
async fn streamed_results_report_tool_guardrail_results() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("guarded_tool", "{}", "call_1")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(guarded_tool()),
    );
    let (stream, _) = streamed(&agent, "hello", RunOptions::new(())).await;
    let r = stream.result().unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
    assert_eq!(r.tool_input_guardrail_results.len(), 1);
    assert_eq!(
        r.tool_input_guardrail_results[0].output.output_info,
        json!("input-checked")
    );
    assert_eq!(r.tool_output_guardrail_results.len(), 1);
    assert_eq!(
        r.tool_output_guardrail_results[0].output.output_info,
        json!("output-checked")
    );
    let state = r.to_state();
    assert_eq!(state.tool_input_guardrail_results.len(), 1);
    assert_eq!(state.tool_output_guardrail_results.len(), 1);
}

/// Python: `test_streamed_tool_guardrail_results_match_non_streamed`.
#[tokio::test]
async fn streamed_and_non_streamed_tool_guardrail_results_match() {
    let mut counts = Vec::new();
    for streamed_mode in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call_with_id("guarded_tool", "{}", "call_1")]);
        model.add_turn(vec![function_call_with_id("guarded_tool", "{}", "call_2")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(guarded_tool()),
        );
        let r = run_mode(&agent, "hello", RunOptions::new(()), streamed_mode)
            .await
            .unwrap();
        counts.push((
            r.tool_input_guardrail_results.len(),
            r.tool_output_guardrail_results.len(),
        ));
    }
    assert_eq!(counts, vec![(2, 2), (2, 2)]);
}

/// Python: `test_streamed_interruption_reports_tool_guardrail_results`.
#[tokio::test]
async fn a_streamed_interruption_reports_tool_guardrail_results() {
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("plain_tool", "{}", "call_plain"),
        function_call_with_id("approval_tool", "{}", "call_approval"),
    ]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("plain_tool", "plain-result").with_input_guardrail(
                ToolInputGuardrail::new("in", |_d| async {
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("input-checked")))
                }),
            ))
            .with_tool(tool("approval_tool", "approved-result").with_needs_approval(true)),
    );
    let (stream, _) = streamed(&agent, "hello", RunOptions::new(())).await;
    let r = stream.result().unwrap();
    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.tool_input_guardrail_results.len(), 1);
    assert_eq!(
        r.tool_input_guardrail_results[0].output.output_info,
        json!("input-checked")
    );
}

fn approval_guarded() -> FunctionTool<()> {
    tool("approval_tool", "approved-result")
        .with_needs_approval(true)
        .with_input_guardrail(ToolInputGuardrail::new("in", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("input-checked")))
        }))
        .with_output_guardrail(ToolOutputGuardrail::new("out", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("output-checked")))
        }))
}

async fn approve_and_resume(agent: &Arc<Agent<()>>, streamed_mode: bool) -> RunResult<()> {
    let first = run_mode(agent, "hello", RunOptions::new(()), streamed_mode)
        .await
        .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    run_mode(agent, state, RunOptions::new(()), streamed_mode)
        .await
        .unwrap()
}

/// Python: `test_streamed_resume_tool_guardrail_results_match_non_streamed`,
/// `test_streamed_resume_terminal_turn_reports_tool_guardrail_results`,
/// `test_streamed_resume_handoff_turn_reports_tool_guardrail_results`.
#[tokio::test]
async fn resumed_turns_report_tool_guardrail_results_like_python() {
    for streamed_mode in [false, true] {
        // A resumed turn that runs the model again leaves its results out.
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "approval_tool",
            "{}",
            "call_approval",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Arc::new(
            Agent::<()>::new("TestAgent")
                .with_model_instance(model.clone())
                .with_tool(approval_guarded()),
        );
        let r = approve_and_resume(&agent, streamed_mode).await;
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            (
                r.tool_input_guardrail_results.len(),
                r.tool_output_guardrail_results.len()
            ),
            (0, 0),
            "run-again streamed={streamed_mode}"
        );

        // A resumed turn that ends the run reports them.
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "approval_tool",
            "{}",
            "call_approval",
        )]);
        let agent = Arc::new(
            Agent::<()>::new("TestAgent")
                .with_model_instance(model.clone())
                .with_tool(approval_guarded())
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool),
        );
        let r = approve_and_resume(&agent, streamed_mode).await;
        assert_eq!(r.final_output_text().as_deref(), Some("approved-result"));
        assert_eq!(r.tool_input_guardrail_results.len(), 1);
        assert_eq!(
            r.tool_input_guardrail_results[0].output.output_info,
            json!("input-checked")
        );
        assert_eq!(r.tool_output_guardrail_results.len(), 1);
        assert_eq!(
            r.tool_output_guardrail_results[0].output.output_info,
            json!("output-checked")
        );

        // A resumed turn that hands off keeps them.
        let model = setup();
        let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
        model.add_turn(vec![
            function_call_with_id("approval_tool", "{}", "call_approval"),
            openai_agents_sdk::testing::function_call("transfer_to_target", "{}"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Arc::new(
            Agent::<()>::new("TestAgent")
                .with_model_instance(model.clone())
                .with_tool(approval_guarded())
                .with_handoff(handoff(target)),
        );
        let r = approve_and_resume(&agent, streamed_mode).await;
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            r.tool_input_guardrail_results.len(),
            1,
            "handoff streamed={streamed_mode}"
        );
        assert_eq!(r.tool_output_guardrail_results.len(), 1);
    }
}

/// Python: `test_streamed_trip_replaces_current_tool_output_guardrail_results`
/// (`allow`, `reject_content`; the public-list identity checks are Python-only).
#[tokio::test]
async fn a_trip_sanitizes_the_turn_tool_output_guardrail_results() {
    for reject in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "secret_tool",
            "{}",
            "call-secret",
        )]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(tool("secret_tool", "blocked-secret").with_output_guardrail(
                    ToolOutputGuardrail::new("retain", move |d| async move {
                        let info = d.output.to_json_value();
                        Ok(if reject {
                            ToolGuardrailFunctionOutput::reject_content(format!(
                                "Rejected sensitive tool output: {info}"
                            ))
                            .with_output_info(info)
                        } else {
                            ToolGuardrailFunctionOutput::allow().with_output_info(info)
                        })
                    }),
                ))
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                .with_output_guardrail(OutputGuardrail::new("trip", |_c, _a, _o: Value| async {
                    Ok(GuardrailFunctionOutput::new(Value::Null, true))
                })),
        );
        let checkpoint = openai_agents_sdk::run::RunCheckpoint::new();
        let mut stream = Runner::run_streamed_with_options(
            agent,
            "run",
            RunOptions::new(()).checkpoint(&checkpoint),
        );
        let out = consume(&mut stream).await;
        assert!(matches!(
            out.error,
            Some(AgentsError::OutputGuardrailTripwireTriggered { .. })
        ));
        let state = checkpoint.take().expect("a snapshot");
        assert_eq!(state.tool_output_guardrail_results.len(), 1);
        let output = &state.tool_output_guardrail_results[0].output;
        assert_eq!(
            output.output_info,
            json!(OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT)
        );
        let behavior = serde_json::to_value(&output.behavior).unwrap();
        assert_eq!(
            behavior["type"],
            if reject { "reject_content" } else { "allow" }
        );
        if reject {
            assert_eq!(behavior["message"], OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT);
        }
        assert!(!state.to_json_string().unwrap().contains("blocked-secret"));
    }
}

// ------------------------------------------------- previous_response_id / conversation_id

fn chained_agent(model: &Arc<FakeModel>) -> Arc<Agent<()>> {
    Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool("test_func", "tool_result")),
    )
}

fn input_of(model: &Arc<FakeModel>, index: usize) -> Vec<Value> {
    model.calls()[index]
        .input
        .iter()
        .map(|i| i.0.clone())
        .collect()
}

/// Python: `test_previous_response_id_passed_between_runs[_streamed]`,
/// `test_multi_turn_previous_response_id_passed_between_runs`,
/// `test_previous_response_id_passed_between_runs_streamed_multi_turn`,
/// `test_previous_response_id_only_sends_new_items_multi_turn[_streamed]`.
#[tokio::test]
async fn previous_response_id_chains_and_sends_only_new_items() {
    for streamed_mode in [false, true] {
        let model = setup();
        model.add_turn(vec![
            assistant_message("a_message"),
            function_call_with_id("test_func", r#"{"arg": "foo"}"#, "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let r = run_mode(
            &chained_agent(&model),
            "user_message",
            RunOptions::new(()).previous_response_id("initial-response-123"),
            streamed_mode,
        )
        .await
        .unwrap();
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        let calls = model.calls();
        assert_eq!(
            calls[0].previous_response_id.as_deref(),
            Some("initial-response-123")
        );
        let first = input_of(&model, 0);
        assert_eq!(
            first,
            vec![json!({"role": "user", "content": "user_message"})]
        );
        let last = input_of(&model, 1);
        assert_eq!(last.len(), 1, "streamed={streamed_mode}");
        assert_eq!(last[0]["type"], "function_call_output");
        assert_eq!(
            calls[1].previous_response_id.as_deref(),
            Some("resp_fake_1")
        );
    }
}

/// Python: `test_conversation_id_only_sends_new_items_multi_turn[_streamed]`.
#[tokio::test]
async fn conversation_id_sends_only_new_items() {
    for streamed_mode in [false, true] {
        let model = setup();
        model.add_turn(vec![
            assistant_message("a_message"),
            function_call_with_id("test_func", r#"{"arg": "foo"}"#, "call-test-func-first"),
        ]);
        model.add_turn(vec![
            assistant_message("b_message"),
            function_call_with_id("test_func", r#"{"arg": "bar"}"#, "call-test-func-second"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let r = run_mode(
            &chained_agent(&model),
            "user_message",
            RunOptions::new(()).conversation_id("conv-test-123"),
            streamed_mode,
        )
        .await
        .unwrap();
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            input_of(&model, 0),
            vec![json!({"role": "user", "content": "user_message"})]
        );
        let last = input_of(&model, 2);
        assert_eq!(last.len(), 1, "streamed={streamed_mode}: {last:?}");
        assert_eq!(last[0]["type"], "function_call_output");
    }
}

/// Python: `test_auto_previous_response_id_multi_turn[_streamed]`,
/// `test_without_previous_response_id_and_auto_previous_response_id_no_chaining`,
/// `test_default_send_all_items[_streamed]`.
#[tokio::test]
async fn auto_previous_response_id_chains_from_the_first_turn() {
    for streamed_mode in [false, true] {
        for auto in [true, false] {
            let model = setup();
            model.add_turn(vec![
                assistant_message("a_message"),
                function_call_with_id("test_func", r#"{"arg": "foo"}"#, "c1"),
            ]);
            model.add_turn(vec![assistant_message("done")]);
            run_mode(
                &chained_agent(&model),
                "user_message",
                RunOptions::new(()).auto_previous_response_id(auto),
                streamed_mode,
            )
            .await
            .unwrap();
            let calls = model.calls();
            assert_eq!(calls[0].previous_response_id, None);
            assert_eq!(
                input_of(&model, 0),
                vec![json!({"role": "user", "content": "user_message"})]
            );
            let last = input_of(&model, 1);
            if auto {
                assert_eq!(last.len(), 1);
                assert_eq!(last[0]["type"], "function_call_output");
                assert_eq!(
                    calls[1].previous_response_id.as_deref(),
                    Some("resp_fake_1")
                );
            } else {
                assert_eq!(last.len(), 4, "streamed={streamed_mode}");
                assert_eq!(last[0]["content"], "user_message");
                assert_eq!(last[1]["role"], "assistant");
                assert_eq!(last[2]["name"], "test_func");
                assert_eq!(last[2]["arguments"], r#"{"arg": "foo"}"#);
                assert_eq!(last[3]["type"], "function_call_output");
                assert_eq!(calls[1].previous_response_id, None);
            }
        }
    }
}

/// Python: `test_previous_response_id_retry_does_not_resend_initial_input_multi_turn[_streamed]`.
#[tokio::test]
async fn a_retry_does_not_resend_the_initial_input() {
    for streamed_mode in [false, true] {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let model = Arc::new(
            ScriptedModel::with_steps(vec![
                ScriptedStep::raise_error(AgentsError::other("connection error"))
                    .with_retry_advice(openai_agents_sdk::retry::ModelRetryAdvice {
                        suggested: Some(true),
                        replay_safety: Some(openai_agents_sdk::retry::ReplaySafety::Safe),
                        normalized: Some(openai_agents_sdk::retry::ModelRetryNormalizedError {
                            is_network_error: true,
                            explicit_fields: vec!["is_network_error"],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                ScriptedStep::output(vec![
                    assistant_message("a_message"),
                    function_call_with_id("test_func", r#"{"arg": "foo"}"#, "c1"),
                ]),
                ScriptedStep::output(vec![assistant_message("done")]),
            ])
            .unwrap(),
        );
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(tool("test_func", "tool_result"))
                .with_model_settings(ModelSettings {
                    retry: Some(
                        ModelRetrySettings {
                            max_retries: 1,
                            backoff: ModelRetryBackoffSettings {
                                initial_delay: 0.0,
                                max_delay: 0.0,
                                multiplier: 1.0,
                                jitter: false,
                            },
                            ..Default::default()
                        }
                        .with_policy(openai_agents_sdk::retry::retry_policies::network_error()),
                    ),
                    ..Default::default()
                }),
        );
        let r = run_mode(
            &agent,
            "user_message",
            RunOptions::new(()).previous_response_id("initial-response-123"),
            streamed_mode,
        )
        .await
        .unwrap();
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        let last: Vec<Value> = model
            .last_call()
            .unwrap()
            .request
            .input
            .iter()
            .map(|i| i.0.clone())
            .collect();
        assert_eq!(last.len(), 1, "streamed={streamed_mode}");
        assert_eq!(last[0]["type"], "function_call_output");
    }
}

fn shell_call(call_id: &str, status: &str) -> Value {
    json!({"type": "shell_call", "id": "shell_1", "call_id": call_id, "status": status,
           "action": {"commands": ["echo hi"], "timeout_ms": null, "max_output_length": null}})
}

/// Python: `test_manual_pending_shell_call_input_is_preserved_non_streamed[_with_session]`,
/// `test_manual_pending_shell_call_input_is_preserved_streamed[_with_session]`.
#[tokio::test]
async fn a_manual_pending_shell_call_input_is_replayed() {
    for streamed_mode in [false, true] {
        for with_session in [false, true] {
            let model = setup();
            model.add_turn(vec![function_call_with_id(
                "test_func",
                r#"{"arg": "foo"}"#,
                "c1",
            )]);
            model.add_turn(vec![assistant_message("done")]);
            let agent = chained_agent(&model);
            let mut options = RunOptions::new(());
            if with_session {
                options = options.session(ListSession::new(vec![], false).await);
            }
            let r = run_mode(
                &agent,
                vec![InputItem(shell_call("manual_shell", "in_progress"))],
                options,
                streamed_mode,
            )
            .await
            .unwrap();
            assert_eq!(r.final_output_text().as_deref(), Some("done"));
            for index in [0, 1] {
                assert!(
                    input_of(&model, index)
                        .iter()
                        .any(|i| i["type"] == "shell_call" && i["call_id"] == "manual_shell"),
                    "streamed={streamed_mode} session={with_session} call={index}"
                );
            }
        }
    }
}

/// Python: `test_default_multi_turn_drops_orphan_hosted_shell_calls`,
/// `test_default_multi_turn_streamed_drops_orphan_hosted_shell_calls`.
#[tokio::test]
async fn an_orphan_hosted_shell_call_is_not_replayed() {
    use openai_agents_sdk::tool::{ShellTool, Tool};
    for streamed_mode in [false, true] {
        let model = setup();
        model.add_turn(vec![OutputItem::from_value(shell_call(
            "call_shell_1",
            "completed",
        ))]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Arc::new(
            Agent::<()>::new("hosted-shell")
                .with_model_instance(model.clone())
                .with_tool(Tool::Shell(Arc::new(
                    ShellTool::hosted(json!({"type": "container_auto"})).unwrap(),
                ))),
        );
        let r = run_mode(&agent, "user_message", RunOptions::new(()), streamed_mode)
            .await
            .unwrap();
        assert_eq!(r.final_output_text().as_deref(), Some("done"));
        let last: Vec<Value> = model
            .calls()
            .last()
            .unwrap()
            .input
            .iter()
            .map(|i| i.0.clone())
            .collect();
        assert_eq!(
            last,
            vec![json!({"role": "user", "content": "user_message"})],
            "streamed={streamed_mode}"
        );
    }
}
