//! Ports of `tests/test_cancel_streaming.py` and `tests/test_soft_cancel.py`: cancelling a
//! streamed run immediately or after the current turn.
//!
//! Python cancels from inside `async for event in result.stream_events()`. A Rust event stream
//! borrows its `RunResultStreaming`, so the ports cancel through
//! [`RunResultStreaming::cancel_handle`], which does not borrow it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::default_tool_name;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::result::{CancelMode, RunResultStreaming};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{ComputerTool, FunctionTool};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, RunContext, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn setup() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn scripted(turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(turns.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

/// `get_function_tool(name, value)`.
fn tool_returning(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn is_item(event: &StreamEvent<()>, wanted: RunItemStreamEventName) -> bool {
    matches!(event, StreamEvent::RunItem { name, .. } if *name == wanted)
}

/// Drain the stream, calling `on_event` with each event and its position (1-based).
async fn consume_with(
    result: &mut RunResultStreaming<()>,
    mut on_event: impl FnMut(&StreamEvent<()>, usize),
) -> Vec<StreamEvent<()>> {
    let mut events = Vec::new();
    let mut stream = std::pin::pin!(result.stream_events());
    while let Some(event) = stream.next().await {
        let event = event.expect("stream event");
        events.push(event.clone());
        on_event(&event, events.len());
    }
    events
}

/// A streamed run sharing `context`, so its usage stays readable after an immediate cancel.
fn run_streamed(agent: &Agent<()>, context: &RunContext<()>) -> RunResultStreaming<()> {
    Runner::run_streamed_with_options(
        agent,
        "Hello",
        RunOptions::with_run_context(context.clone()),
    )
}

/// A model that delegates to a script but waits before `response.completed`, or blocks a call
/// until released.
#[derive(Debug)]
struct Gated {
    inner: Arc<ScriptedModel>,
    complete_delay: Duration,
    release: Option<Arc<tokio::sync::Notify>>,
    stream_calls: AtomicUsize,
    second_call_started: AtomicBool,
}

impl Gated {
    fn new(inner: Arc<ScriptedModel>) -> Self {
        Gated {
            inner,
            complete_delay: Duration::ZERO,
            release: None,
            stream_calls: AtomicUsize::new(0),
            second_call_started: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Model for Gated {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        if self.stream_calls.fetch_add(1, Ordering::SeqCst) == 1 {
            self.second_call_started.store(true, Ordering::SeqCst);
        }
        if let Some(release) = &self.release {
            release.notified().await;
        }
        let delay = self.complete_delay;
        let events = self.inner.stream_response(request).await?;
        Ok(events
            .then(move |event| async move {
                let completed = event
                    .as_ref()
                    .is_ok_and(|e| e["type"] == "response.completed");
                if completed && !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                event
            })
            .boxed())
    }
}

// ---------------------------------------------------------------------------
// test_cancel_streaming.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn simple_streaming_with_cancel() {
    setup();
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    let cancel = result.cancel_handle();
    let events = consume_with(&mut result, |_, n| {
        if n == 1 {
            cancel.cancel(CancelMode::Immediate);
        }
    })
    .await;
    assert_eq!(events.len(), 1);
}

#[tokio::test]
async fn multiple_events_streaming_with_cancel() {
    setup();
    let model = scripted(vec![
        vec![
            assistant_message("a_message"),
            function_call("foo", r#"{"a": "b"}"#),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("Joker")
        .with_model_instance(model)
        .with_tool(tool_returning("foo", "tool_result"));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    let cancel = result.cancel_handle();
    let events = consume_with(&mut result, |_, n| {
        if n == 2 {
            cancel.cancel(CancelMode::Immediate);
        }
    })
    .await;
    assert_eq!(events.len(), 2);
}

/// Take the first event, cancel, and stop consuming.
async fn first_event_then_cancel(result: &mut RunResultStreaming<()>) -> usize {
    let cancel = result.cancel_handle();
    let mut stream = std::pin::pin!(result.stream_events());
    let mut seen = 0;
    if let Some(event) = stream.next().await {
        event.expect("event");
        seen += 1;
        cancel.cancel(CancelMode::Immediate);
    }
    seen
}

#[tokio::test]
async fn cancel_prevents_further_events() {
    setup();
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    assert_eq!(first_event_then_cancel(&mut result).await, 1);
    let more = consume_with(&mut result, |_, _| {}).await;
    assert!(more.is_empty(), "no events after cancel");
}

#[tokio::test]
async fn cancel_is_idempotent() {
    setup();
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    let cancel = result.cancel_handle();
    let mut seen = 0;
    {
        let mut stream = std::pin::pin!(result.stream_events());
        if let Some(event) = stream.next().await {
            event.unwrap();
            seen += 1;
            cancel.cancel(CancelMode::Immediate);
            cancel.cancel(CancelMode::Immediate);
        }
    }
    result.cancel(CancelMode::Immediate);
    assert_eq!(seen, 1);
    assert!(result.is_complete());
}

/// A provider recording `close`, registered inside a runner-owned `MultiProvider` so the
/// runner's close of the provider it owns is observable.
#[derive(Debug, Default)]
struct ClosingProvider {
    closed: AtomicUsize,
}

#[async_trait]
impl openai_agents_sdk::models::interface::ModelProvider for ClosingProvider {
    fn get_model(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        Err(AgentsError::user("not used"))
    }

    async fn close(&self) {
        self.closed.fetch_add(1, Ordering::SeqCst);
    }
}

fn run_owned_provider(
    recorder: Arc<ClosingProvider>,
) -> Arc<dyn openai_agents_sdk::models::interface::ModelProvider> {
    use openai_agents_sdk::models::{MultiProvider, MultiProviderMap};
    let mut map = MultiProviderMap::new();
    map.add_provider("recorder", recorder);
    Arc::new(
        MultiProvider::new()
            .with_provider_map(&map)
            .into_run_owned(),
    )
}

#[tokio::test]
async fn cancel_before_streaming() {
    setup();
    let recorder = Arc::new(ClosingProvider::default());
    let config = openai_agents_sdk::RunConfig::default()
        .with_model_provider(run_owned_provider(recorder.clone()));
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed_with_options(
        &agent,
        "Please tell me 5 jokes.",
        RunOptions::default().run_config(config),
    );
    result.cancel(CancelMode::Immediate);
    let events = consume_with(&mut result, |_, _| {}).await;
    assert!(
        events.is_empty(),
        "no events when cancelled before streaming"
    );
    assert_eq!(recorder.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_cleans_up_resources() {
    setup();
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    first_event_then_cancel(&mut result).await;
    assert!(result.is_complete(), "complete after cancel");
    result
        .finish()
        .await
        .expect("a cancelled run finishes cleanly");
}

#[tokio::test]
async fn cancel_immediate_mode_explicit() {
    setup();
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    first_event_then_cancel(&mut result).await;
    assert!(result.is_complete());
    assert!(consume_with(&mut result, |_, _| {}).await.is_empty());
}

#[tokio::test]
async fn stream_events_respects_timeout_cancellation() {
    setup();
    let mut model = Gated::new(scripted(vec![vec![assistant_message("Final response")]]));
    model.complete_delay = Duration::from_millis(500);
    let agent = Agent::<()>::new("TimeoutTester").with_model_instance(Arc::new(model));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    {
        let mut stream = std::pin::pin!(result.stream_events());
        loop {
            let event = tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .expect("event in time")
                .expect("event")
                .unwrap();
            if let StreamEvent::RawResponse(raw) = &event
                && raw["type"] == "response.output_item.done"
            {
                break;
            }
        }
        let start = std::time::Instant::now();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), stream.next())
                .await
                .is_err()
        );
        assert!(start.elapsed() < Duration::from_millis(300));
    }
    result.cancel(CancelMode::Immediate);
}

#[tokio::test]
async fn cancel_immediate_unblocks_waiting_stream_consumer() {
    setup();
    let release = Arc::new(tokio::sync::Notify::new());
    let mut model = Gated::new(Arc::new(ScriptedModel::new()));
    model.release = Some(release.clone());
    let agent = Agent::<()>::new("Joker").with_model_instance(Arc::new(model));
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    let cancel = result.cancel_handle();
    let consumer = tokio::spawn(async move {
        let events = consume_with(&mut result, |_, _| {}).await;
        (events.len(), result)
    });
    tokio::task::yield_now().await;
    cancel.cancel(CancelMode::Immediate);
    let (events, result) = tokio::time::timeout(Duration::from_secs(1), consumer)
        .await
        .expect("consumer unblocked")
        .unwrap();
    assert!(events <= 1);
    assert!(result.is_complete());
}

#[tokio::test]
async fn run_loop_exception_is_none_on_success() {
    setup();
    let agent =
        Agent::<()>::new("A").with_model_instance(scripted(vec![vec![assistant_message("hello")]]));
    let mut result = Runner::run_streamed(&agent, "hi");
    consume_with(&mut result, |_, _| {}).await;
    assert!(result.run_loop_error().is_none());
}

#[derive(Debug)]
struct BoomModel;

#[async_trait]
impl Model for BoomModel {
    async fn get_response(&self, _request: ModelRequest) -> Result<ModelResponse> {
        Err(AgentsError::other("run loop boom"))
    }

    async fn stream_response(&self, _request: ModelRequest) -> Result<ResponseEventStream> {
        Err(AgentsError::other("run loop boom"))
    }
}

/// Collect events, returning the error the stream ended with.
async fn consume_until_error(result: &mut RunResultStreaming<()>) -> Option<AgentsError> {
    let mut stream = std::pin::pin!(result.stream_events());
    while let Some(event) = stream.next().await {
        if let Err(error) = event {
            return Some(error);
        }
    }
    None
}

#[tokio::test]
async fn run_loop_exception_surfaced_after_stream() {
    setup();
    let agent = Agent::<()>::new("A").with_model_instance(Arc::new(BoomModel));
    let mut result = Runner::run_streamed(&agent, "hi");
    let error = consume_until_error(&mut result)
        .await
        .expect("error raised");
    assert!(error.to_string().contains("run loop boom"));
    assert!(result.run_loop_error().unwrap().contains("run loop boom"));
}

/// `test_falsy_input_guardrail_exception_is_surfaced_after_stream`: an input guardrail that
/// raises fails the stream (the falsy-exception aspect is Python-only).
#[tokio::test]
async fn input_guardrail_exception_is_surfaced_after_stream() {
    setup();
    let guardrail = InputGuardrail::<()>::new("raising_guardrail", |_ctx, _agent, _input| async {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("falsy guardrail boom"))
    });
    let agent = Agent::<()>::new("A")
        .with_model_instance(scripted(vec![vec![assistant_message("done")]]))
        .with_input_guardrail(guardrail);
    let mut result = Runner::run_streamed(&agent, "hi");
    let error = consume_until_error(&mut result)
        .await
        .expect("error raised");
    assert!(
        error.to_string().contains("falsy guardrail boom"),
        "{error}"
    );
}

#[derive(Debug)]
struct FakeComputer;

#[async_trait]
impl Computer for FakeComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Browser)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((800, 600))
    }
    async fn screenshot(&self) -> Result<String> {
        Ok("aW1n".into())
    }
    async fn wait(&self) -> Result<()> {
        Ok(())
    }
    async fn click(&self, _x: i64, _y: i64, _button: &str, _keys: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn double_click(&self, _x: i64, _y: i64, _keys: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _x: i64,
        _y: i64,
        _scroll_x: i64,
        _scroll_y: i64,
        _keys: Option<&[String]>,
    ) -> Result<()> {
        Ok(())
    }
    async fn type_text(&self, _text: &str) -> Result<()> {
        Ok(())
    }
    async fn move_mouse(&self, _x: i64, _y: i64, _keys: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn keypress(&self, _keys: &[String]) -> Result<()> {
        Ok(())
    }
    async fn drag(&self, _path: &[(i64, i64)], _keys: Option<&[String]>) -> Result<()> {
        Ok(())
    }
}

/// Python runs the tool while the parallel guardrail is still pending and cancels then. Rust
/// waits for parallel input guardrails before it executes tools (see the parity report), so
/// the cancel lands while the guardrail is pending; the cleanup claim is the same.
#[tokio::test]
async fn cancel_with_pending_parallel_input_guardrail_finishes_cleanup() {
    setup();
    let guardrail_started = Arc::new(tokio::sync::Notify::new());
    let disposed = Arc::new(AtomicUsize::new(0));

    let started = guardrail_started.clone();
    let guardrail = InputGuardrail::<()>::new("slow_guardrail", move |_ctx, _agent, _input| {
        let started = started.clone();
        async move {
            started.notify_one();
            std::future::pending::<()>().await;
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }
    });
    let slow_tool =
        FunctionTool::<()>::new("slow_tool", "slow", move |_ctx, _a: AnyArgs| async move {
            std::future::pending::<()>().await;
            Ok("unreachable".to_string())
        });
    let counter = disposed.clone();
    let computer_tool = ComputerTool::<()>::from_provider(
        || async { Ok(Arc::new(FakeComputer) as Arc<dyn Computer>) },
        move |_computer| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        },
    );
    let agent = Agent::<()>::new("A")
        .with_model_instance(scripted(vec![vec![function_call_with_id(
            "slow_tool",
            "{}",
            "call_1",
        )]]))
        .with_tool(slow_tool)
        .with_tool(computer_tool)
        .with_input_guardrail(guardrail);
    let mut result = Runner::run_streamed(&agent, "hi");
    let cancel = result.cancel_handle();
    let consumer = tokio::spawn(async move {
        consume_with(&mut result, |_, _| {}).await;
    });
    tokio::time::timeout(Duration::from_secs(2), guardrail_started.notified())
        .await
        .expect("guardrail started");
    cancel.cancel(CancelMode::Immediate);
    tokio::time::timeout(Duration::from_secs(2), consumer)
        .await
        .expect("consumer finished")
        .unwrap();
    assert_eq!(disposed.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// test_soft_cancel.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn soft_cancel_completes_turn() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(scripted(vec![vec![]]));
    let mut result = Runner::run_streamed(&agent, "Hello");
    let cancel = result.cancel_handle();
    let events = consume_with(&mut result, |_, n| {
        if n == 1 {
            cancel.cancel(CancelMode::AfterTurn);
        }
    })
    .await;
    assert!(events.len() > 1, "soft cancel lets the turn complete");
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_vs_immediate() {
    setup();
    let mut counts = Vec::new();
    for mode in [CancelMode::Immediate, CancelMode::AfterTurn] {
        let agent = Agent::<()>::new("A").with_model_instance(scripted(vec![vec![]]));
        let mut result = Runner::run_streamed(&agent, "Hello");
        let cancel = result.cancel_handle();
        let events = consume_with(&mut result, |_, n| {
            if n == 1 {
                cancel.cancel(mode);
            }
        })
        .await;
        counts.push(events.len());
    }
    assert!(
        counts[1] > counts[0],
        "soft={} immediate={}",
        counts[1],
        counts[0]
    );
}

#[tokio::test]
async fn soft_cancel_with_tool_calls() {
    setup();
    let model = scripted(vec![
        vec![
            assistant_message("Let me calculate"),
            function_call("calc", "{}"),
        ],
        vec![assistant_message("Result is 42")],
    ]);
    let agent = Agent::<()>::new("Assistant")
        .with_model_instance(model)
        .with_tool(tool_returning("calc", "42"));
    let mut result = Runner::run_streamed(&agent, "Calculate");
    let cancel = result.cancel_handle();
    let (mut called, mut output) = (false, false);
    consume_with(&mut result, |event, _| {
        if is_item(event, RunItemStreamEventName::ToolCalled) {
            called = true;
            cancel.cancel(CancelMode::AfterTurn);
        } else if is_item(event, RunItemStreamEventName::ToolOutput) {
            output = true;
        }
    })
    .await;
    assert!(called, "tool call seen");
    assert!(output, "the tool runs before the soft cancel stops the run");
}

#[tokio::test]
async fn soft_cancel_saves_session() {
    setup();
    let model = scripted(vec![vec![], vec![]]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("test_soft_cancel_session"));
    let mut result = Runner::run_streamed_with_options(
        &agent,
        "Hello",
        RunOptions::default().session(session.clone()),
    );
    let cancel = result.cancel_handle();
    consume_with(&mut result, |event, _| {
        if matches!(event, StreamEvent::RunItem { .. }) {
            cancel.cancel(CancelMode::AfterTurn);
        }
    })
    .await;
    assert!(!session.get_items(None).await.unwrap().is_empty());

    let resumed = Runner::run_with_options(
        &agent,
        "Continue",
        RunOptions::default().session(session.clone()),
    )
    .await
    .expect("resume");
    assert!(resumed.final_output.is_some());
}

#[tokio::test]
async fn soft_cancel_tracks_usage() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(scripted(vec![vec![]]));
    let context = RunContext::new(());
    let mut result = run_streamed(&agent, &context);
    let cancel = result.cancel_handle();
    consume_with(&mut result, |event, _| {
        if matches!(event, StreamEvent::RawResponse(_)) {
            cancel.cancel(CancelMode::AfterTurn);
        }
    })
    .await;
    assert!(context.usage().requests > 0);
}

fn tool1_agent(model: Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new("Assistant")
        .with_model_instance(model)
        .with_tool(tool_returning("tool1", "result1"))
}

#[tokio::test]
async fn soft_cancel_stops_next_turn() {
    setup();
    for suspensions in [0, 1, 3] {
        let model = scripted(vec![
            vec![function_call("tool1", "{}")],
            vec![assistant_message("Turn 2")],
            vec![assistant_message("Turn 3")],
        ]);
        let agent = tool1_agent(model);
        let context = RunContext::new(());
        let mut result = run_streamed(&agent, &context);
        let cancel = result.cancel_handle();
        let mut turns = 0;
        {
            let mut stream = std::pin::pin!(result.stream_events());
            while let Some(event) = stream.next().await {
                if is_item(&event.unwrap(), RunItemStreamEventName::ToolOutput) {
                    turns += 1;
                    if turns == 1 {
                        for _ in 0..suspensions {
                            tokio::task::yield_now().await;
                        }
                        cancel.cancel(CancelMode::AfterTurn);
                    }
                }
            }
        }
        assert_eq!(turns, 1, "suspensions={suspensions}");
        assert!(result.final_output().is_none());
        assert_eq!(context.usage().requests, 1, "suspensions={suspensions}");
    }
}

#[tokio::test]
async fn streamed_run_completes_without_an_event_consumer() {
    setup();
    let model = scripted(vec![
        vec![function_call("tool1", "{}")],
        vec![assistant_message("Turn 2")],
    ]);
    let agent = tool1_agent(model.clone());
    let context = RunContext::new(());
    let mut result = run_streamed(&agent, &context);
    // Nobody consumes the events: the run must still reach its second turn on its own.
    tokio::time::timeout(Duration::from_secs(1), async {
        while model.calls().len() < 2 || context.usage().requests < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the run finishes without a consumer");
    result.finish().await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Turn 2"));
    assert_eq!(context.usage().requests, 2);
}

#[tokio::test]
async fn closing_stream_consumer_releases_turn_acknowledgement() {
    setup();
    let model = scripted(vec![
        vec![function_call("tool1", "{}")],
        vec![assistant_message("Turn 2")],
    ]);
    let agent = tool1_agent(model);
    let context = RunContext::new(());
    let mut result = run_streamed(&agent, &context);
    {
        let mut stream = std::pin::pin!(result.stream_events());
        while let Some(event) = stream.next().await {
            if is_item(&event.unwrap(), RunItemStreamEventName::ToolOutput) {
                break;
            }
        }
        // Dropping the stream here is `aclose()`.
    }
    tokio::time::timeout(Duration::from_secs(1), result.finish())
        .await
        .expect("no deadlock")
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Turn 2"));
    assert_eq!(context.usage().requests, 2);
}

#[tokio::test]
async fn cancelled_stream_consumer_releases_turn_acknowledgement() {
    setup();
    let model = scripted(vec![
        vec![function_call("tool1", "{}")],
        vec![assistant_message("Turn 2")],
    ]);
    let agent = tool1_agent(model.clone());
    let context = RunContext::new(());
    let mut result = run_streamed(&agent, &context);
    let suspended = Arc::new(tokio::sync::Notify::new());
    {
        let signal = suspended.clone();
        let consumer = async {
            let mut stream = std::pin::pin!(result.stream_events());
            while let Some(event) = stream.next().await {
                if is_item(&event.unwrap(), RunItemStreamEventName::ToolOutput) {
                    signal.notify_one();
                    std::future::pending::<()>().await;
                }
            }
        };
        // Cancelling the consumer drops it, and its stream with it, while it is suspended
        // after the yield.
        tokio::select! {
            _ = consumer => panic!("the consumer stays suspended"),
            _ = suspended.notified() => {}
        }
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        while context.usage().requests < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the run continues once its consumer is gone");
    result.finish().await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Turn 2"));
    assert_eq!(context.usage().requests, 2);
}

#[tokio::test]
async fn immediate_cancel_releases_turn_acknowledgement() {
    setup();
    let model = scripted(vec![
        vec![function_call("tool1", "{}")],
        vec![assistant_message("Turn 2")],
    ]);
    let agent = tool1_agent(model);
    let context = RunContext::new(());
    let mut result = run_streamed(&agent, &context);
    let cancel = result.cancel_handle();
    {
        let mut stream = std::pin::pin!(result.stream_events());
        while let Some(event) = stream.next().await {
            if is_item(&event.unwrap(), RunItemStreamEventName::ToolOutput) {
                tokio::task::yield_now().await;
                cancel.cancel(CancelMode::Immediate);
            }
        }
    }
    assert!(result.is_complete());
    assert!(result.final_output().is_none());
    assert_eq!(context.usage().requests, 1);
}

#[tokio::test]
async fn cancel_mode_immediate_is_the_plain_cancel() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Hello");
    let cancel = result.cancel_handle();
    let events = consume_with(&mut result, |_, n| {
        if n == 1 {
            cancel.cancel(CancelMode::Immediate);
        }
    })
    .await;
    assert_eq!(events.len(), 1);
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_idempotent() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(scripted(vec![vec![]]));
    let mut result = Runner::run_streamed(&agent, "Hello");
    let cancel = result.cancel_handle();
    let mut called_twice = false;
    consume_with(&mut result, |_, _| {
        if !called_twice {
            cancel.cancel(CancelMode::AfterTurn);
            cancel.cancel(CancelMode::AfterTurn);
            called_twice = true;
        }
    })
    .await;
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_before_streaming() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Hello");
    result.cancel(CancelMode::AfterTurn);
    let events = consume_with(&mut result, |_, _| {}).await;
    assert!(events.len() <= 1, "at most the agent_updated event");
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_mixed_modes() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Hello");
    result.cancel(CancelMode::AfterTurn);
    result.cancel(CancelMode::Immediate);
    let events = consume_with(&mut result, |_, _| {}).await;
    // Immediate takes precedence: nothing is delivered.
    assert!(events.is_empty());
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_explicit_immediate_mode() {
    setup();
    let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "Hello");
    assert_eq!(first_event_then_cancel(&mut result).await, 1);
    assert!(result.is_complete());
}

#[tokio::test]
async fn soft_cancel_with_multiple_tool_calls() {
    setup();
    let model = scripted(vec![
        vec![
            function_call_with_id("tool1", "{}", "tool_1"),
            function_call_with_id("tool2", "{}", "tool_2"),
        ],
        vec![assistant_message("Both tools executed")],
    ]);
    let agent = Agent::<()>::new("Assistant")
        .with_model_instance(model)
        .with_tool(tool_returning("tool1", "result1"))
        .with_tool(tool_returning("tool2", "result2"));
    let mut result = Runner::run_streamed(&agent, "Execute tools");
    let cancel = result.cancel_handle();
    let mut outputs = 0;
    consume_with(&mut result, |event, _| {
        if is_item(event, RunItemStreamEventName::ToolCalled) {
            if outputs == 0 {
                cancel.cancel(CancelMode::AfterTurn);
            }
        } else if is_item(event, RunItemStreamEventName::ToolOutput) {
            outputs += 1;
        }
    })
    .await;
    assert_eq!(outputs, 2, "both tools run before the soft cancel");
}

#[tokio::test]
async fn soft_cancel_preserves_state() {
    setup();
    let model = scripted(vec![
        vec![function_call("tool1", "{}")],
        vec![assistant_message("Done")],
    ]);
    let agent = Agent::<()>::new("Assistant")
        .with_model_instance(model)
        .with_tool(tool_returning("tool1", "result"));
    let mut result = Runner::run_streamed(&agent, "Hello");
    let cancel = result.cancel_handle();
    consume_with(&mut result, |event, _| {
        if is_item(event, RunItemStreamEventName::ToolOutput) {
            cancel.cancel(CancelMode::AfterTurn);
        }
    })
    .await;
    assert!(result.is_complete());
    let run = result
        .result()
        .expect("a soft-cancelled run keeps its result");
    assert!(!run.new_items.is_empty());
    assert!(!run.raw_responses.is_empty());
    assert!(run.context_wrapper.usage().requests > 0);
}

#[tokio::test]
async fn soft_cancel_with_handoff() {
    setup();
    let model = scripted(vec![]);
    let agent2 = Agent::<()>::new("Agent2").with_model_instance(model.clone());
    model
        .extend(vec![
            ScriptedStep::output(vec![function_call(default_tool_name("Agent2"), "{}")]),
            ScriptedStep::output(vec![assistant_message("Agent2 response")]),
        ])
        .unwrap();
    let agent1 = Agent::<()>::new("Agent1")
        .with_model_instance(model)
        .with_handoff(agent2);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("test_soft_cancel_handoff"));
    let mut result = Runner::run_streamed_with_options(
        &agent1,
        "Hello",
        RunOptions::default().session(session.clone()),
    );
    let cancel = result.cancel_handle();
    let mut handoff_seen = false;
    consume_with(&mut result, |event, _| {
        if is_item(event, RunItemStreamEventName::HandoffRequested) {
            handoff_seen = true;
            cancel.cancel(CancelMode::AfterTurn);
        }
    })
    .await;
    assert!(handoff_seen);
    assert!(!session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn soft_cancel_waits_for_handoff_event_consumption_before_next_turn() {
    setup();
    // The run task is spawned before the consumer registers; asyncio runs the consumer's first
    // `__anext__` before the model call, so the model here waits for the consumer to start.
    let start = Arc::new(tokio::sync::Notify::new());
    let mut gated = Gated::new(scripted(vec![
        vec![function_call(default_tool_name("Delegate"), "{}")],
        vec![assistant_message("Delegate response")],
    ]));
    gated.release = Some(start.clone());
    let model = Arc::new(gated);
    let delegate = Agent::<()>::new("Delegate")
        .with_model_instance(model.clone())
        .with_output_type::<i64>();
    let triage = Agent::<()>::new("Triage")
        .with_model_instance(model.clone())
        .with_handoff(delegate);
    let context = RunContext::new(());
    let mut result = Runner::run_streamed_with_options(
        &triage,
        "Route this request",
        RunOptions::with_run_context(context.clone()),
    );
    let cancel = result.cancel_handle();
    let suspended = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (signal, gate) = (suspended.clone(), release.clone());
    let consumer = tokio::spawn(async move {
        {
            let mut stream = std::pin::pin!(result.stream_events());
            start.notify_one();
            while let Some(event) = stream.next().await {
                if is_item(&event.unwrap(), RunItemStreamEventName::HandoffRequested) {
                    signal.notify_one();
                    gate.notified().await;
                    cancel.cancel(CancelMode::AfterTurn);
                }
            }
        }
        result
    });
    tokio::time::timeout(Duration::from_secs(1), suspended.notified())
        .await
        .expect("consumer suspended");
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    assert!(!model.second_call_started.load(Ordering::SeqCst));

    release.notify_one();
    let result = tokio::time::timeout(Duration::from_secs(1), consumer)
        .await
        .expect("consumer done")
        .unwrap();
    assert!(result.final_output().is_none());
    assert_eq!(context.usage().requests, 1);
    assert_eq!(result.current_agent().name, "Delegate");
    let run = result.result().expect("result");
    assert_eq!(run.last_agent.name, "Delegate");
    assert!(run.last_agent.output_type.is_some());
    assert_eq!(result.to_state().unwrap().current_agent_name, "Delegate");
}

#[tokio::test]
async fn soft_cancel_with_session_and_multiple_turns() {
    setup();
    let model = scripted(vec![
        vec![function_call_with_id("tool1", "{}", "tool_1")],
        vec![function_call_with_id("tool1", "{}", "tool_2")],
        vec![assistant_message("Final")],
    ]);
    let agent = tool1_agent(model);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("test_soft_cancel_multi"));
    let mut result = Runner::run_streamed_with_options(
        &agent,
        "Hello",
        RunOptions::default().session(session.clone()),
    );
    let cancel = result.cancel_handle();
    let mut turns = 0;
    consume_with(&mut result, |event, _| {
        if is_item(event, RunItemStreamEventName::ToolOutput) {
            turns += 1;
            if turns == 2 {
                cancel.cancel(CancelMode::AfterTurn);
            }
        }
    })
    .await;
    assert_eq!(turns, 2);
    assert!(!session.get_items(None).await.unwrap().is_empty());
}
