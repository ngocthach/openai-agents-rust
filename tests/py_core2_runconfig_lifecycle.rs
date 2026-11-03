//! Ports of `tests/test_runner_model_provider_lifecycle.py`, `tests/test_agent_runner_sync.py`,
//! `tests/test_run.py` and `tests/test_repl.py`.
//!
//! Python decides that the runner owns its model provider when the caller passed no run config.
//! A Rust `RunConfig` is always present, so ownership travels with the provider instead: the
//! provider `RunConfig::default` creates is marked run-owned (`MultiProvider::into_run_owned`),
//! and these tests register a recording provider inside such a `MultiProvider` to observe the
//! close the runner performs.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::{InputItem, ModelResponse, RunInput};
use openai_agents_sdk::models::interface::{
    Model, ModelProvider, ModelRequest, ResponseEventStream,
};
use openai_agents_sdk::models::{MultiProvider, MultiProviderMap};
use openai_agents_sdk::result::{CancelMode, RunResult, RunResultStreaming};
use openai_agents_sdk::run::{AgentRunner, DefaultAgentRunner, RunOptions};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunConfig, Runner};
use schemars::JsonSchema;

type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'static>>;
use serde::Deserialize;

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

/// Records `close`; optionally holds it until released.
#[derive(Debug, Default)]
struct ClosingProvider {
    closed: AtomicUsize,
    close_started: tokio::sync::Notify,
    release: Option<Arc<tokio::sync::Notify>>,
    completed: AtomicBool,
}

#[async_trait]
impl ModelProvider for ClosingProvider {
    fn get_model(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        Err(AgentsError::user("not used"))
    }

    async fn close(&self) {
        self.closed.fetch_add(1, Ordering::SeqCst);
        self.close_started.notify_one();
        if let Some(release) = &self.release {
            release.notified().await;
        }
        self.completed.store(true, Ordering::SeqCst);
    }
}

/// A runner-owned provider whose close reaches `recorder`.
fn run_owned(recorder: Arc<ClosingProvider>) -> Arc<dyn ModelProvider> {
    let mut map = MultiProviderMap::new();
    map.add_provider("recorder", recorder);
    Arc::new(
        MultiProvider::new()
            .with_provider_map(&map)
            .into_run_owned(),
    )
}

fn owned_options(recorder: &Arc<ClosingProvider>) -> RunOptions<()> {
    RunOptions::default()
        .run_config(RunConfig::default().with_model_provider(run_owned(recorder.clone())))
}

/// Sets a flag when dropped.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A model whose stream waits for `finish` before delivering the scripted events.
#[derive(Debug)]
struct GatedStream {
    inner: Arc<ScriptedModel>,
    started: Arc<tokio::sync::Notify>,
    finish: Arc<tokio::sync::Notify>,
    stream_closed: Arc<AtomicBool>,
}

impl GatedStream {
    fn new(inner: Arc<ScriptedModel>) -> Arc<Self> {
        Arc::new(GatedStream {
            inner,
            started: Arc::default(),
            finish: Arc::default(),
            stream_closed: Arc::default(),
        })
    }
}

#[async_trait]
impl Model for GatedStream {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let events = self.inner.stream_response(request).await?;
        let (started, finish) = (self.started.clone(), self.finish.clone());
        let closed = DropFlag(self.stream_closed.clone());
        Ok(futures::stream::once(async move {
            let _closed = closed;
            started.notify_one();
            finish.notified().await;
            events
        })
        .flatten()
        .boxed())
    }
}

async fn drain(result: &mut RunResultStreaming<()>) {
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event.expect("event");
    }
}

// ---------------------------------------------------------------------------
// test_runner_model_provider_lifecycle.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_closes_implicitly_created_model_provider() {
    setup();
    assert!(RunConfig::<()>::default().model_provider.is_run_owned());
    for tracing_disabled in [false, true] {
        let recorder = Arc::new(ClosingProvider::default());
        let agent = Agent::<()>::new("test")
            .with_model_instance(scripted(vec![vec![assistant_message("done")]]));
        let mut options = owned_options(&recorder);
        options.run_config.tracing_disabled = tracing_disabled;
        let result = Runner::run_with_options(&agent, "hello", options)
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(recorder.closed.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn run_sync_closes_implicitly_created_model_provider() {
    setup();
    let recorder = Arc::new(ClosingProvider::default());
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![assistant_message("done")]]));
    let result = Runner::run_sync_with_options(&agent, "hello", owned_options(&recorder)).unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(recorder.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn run_keeps_explicit_model_provider_open_for_reuse() {
    setup();
    #[derive(Debug)]
    struct ReusableProvider {
        model: Arc<dyn Model>,
        close_calls: AtomicUsize,
    }
    #[async_trait]
    impl ModelProvider for ReusableProvider {
        fn get_model(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>> {
            Ok(self.model.clone())
        }
        async fn close(&self) {
            self.close_calls.fetch_add(1, Ordering::SeqCst);
        }
    }
    let provider = Arc::new(ReusableProvider {
        model: scripted(vec![
            vec![assistant_message("first")],
            vec![assistant_message("second")],
        ]),
        close_calls: AtomicUsize::new(0),
    });
    let config = RunConfig::default().with_model_provider(provider.clone());
    let agent = Agent::<()>::new("test").with_model("test-model");
    let first = Runner::run_with_options(
        &agent,
        "first",
        RunOptions::default().run_config(config.clone()),
    )
    .await
    .unwrap();
    let second =
        Runner::run_with_options(&agent, "second", RunOptions::default().run_config(config))
            .await
            .unwrap();
    assert_eq!(first.final_output_text().as_deref(), Some("first"));
    assert_eq!(second.final_output_text().as_deref(), Some("second"));
    assert_eq!(provider.close_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn run_streamed_closes_provider_only_after_run_settles() {
    setup();
    let recorder = Arc::new(ClosingProvider::default());
    let model = GatedStream::new(scripted(vec![vec![assistant_message("done")]]));
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let mut result = Runner::run_streamed_with_options(&agent, "hello", owned_options(&recorder));
    tokio::time::timeout(Duration::from_secs(1), model.started.notified())
        .await
        .expect("model started");
    assert_eq!(recorder.closed.load(Ordering::SeqCst), 0);

    model.finish.notify_one();
    drain(&mut result).await;
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(recorder.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn run_streamed_closes_provider_after_cancellation() {
    setup();
    let recorder = Arc::new(ClosingProvider::default());
    let model = GatedStream::new(scripted(vec![vec![assistant_message("never")]]));
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let mut result = Runner::run_streamed_with_options(&agent, "hello", owned_options(&recorder));
    tokio::time::timeout(Duration::from_secs(1), model.started.notified())
        .await
        .expect("model started");

    result.cancel(CancelMode::Immediate);
    drain(&mut result).await;
    assert!(model.stream_closed.load(Ordering::SeqCst));
    assert_eq!(recorder.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn run_streamed_repeated_cancellation_waits_for_provider_cleanup() {
    setup();
    let release = Arc::new(tokio::sync::Notify::new());
    let recorder = Arc::new(ClosingProvider {
        release: Some(release.clone()),
        ..Default::default()
    });
    let model = GatedStream::new(scripted(vec![vec![assistant_message("never")]]));
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let mut result = Runner::run_streamed_with_options(&agent, "hello", owned_options(&recorder));
    tokio::time::timeout(Duration::from_secs(1), model.started.notified())
        .await
        .expect("model started");

    result.cancel(CancelMode::Immediate);
    tokio::time::timeout(Duration::from_secs(1), recorder.close_started.notified())
        .await
        .expect("close started");
    result.cancel(CancelMode::Immediate);
    release.notify_one();
    drain(&mut result).await;
    assert!(recorder.completed.load(Ordering::SeqCst));
}

/// `test_run_streamed_cancel_before_start_propagates_through_cleanup_wrapper`: the cleanup a
/// streamed run registered still runs when the run is cancelled before it started. Rust's
/// registered cleanup is the `on_complete` hook a sandbox run installs.
#[tokio::test]
async fn run_streamed_cancel_before_start_runs_registered_cleanup() {
    setup();
    let cleaned = Arc::new(AtomicBool::new(false));
    let flag = cleaned.clone();
    let agent = Agent::<()>::new("test").with_model_instance(Arc::new(ScriptedModel::new()));
    let mut result = Runner::run_streamed(&agent, "hello").with_on_complete(Arc::new(move |_| {
        let flag = flag.clone();
        Box::pin(async move { flag.store(true, Ordering::SeqCst) })
    }));
    result.cancel(CancelMode::Immediate);
    drain(&mut result).await;
    assert!(cleaned.load(Ordering::SeqCst));
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn run_streamed_closes_implicit_responses_websocket_connection() {
    use futures::SinkExt;
    use openai_agents_sdk::models::{
        OpenAIClient, OpenAIResponsesTransport, set_default_openai_client,
        set_default_openai_responses_transport,
    };
    use serde_json::json;
    use tokio_tungstenite::tungstenite::Message;
    setup();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let connection_closed = Arc::new(tokio::sync::Notify::new());
    let closed = connection_closed.clone();
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            let Message::Text(text) = message else {
                if matches!(message, Message::Close(_)) {
                    break;
                }
                continue;
            };
            let request: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(request["type"], "response.create");
            let completed = json!({
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "resp-runner-provider-cleanup", "object": "response",
                    "status": "completed",
                    "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "done",
                                             "annotations": []}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            });
            ws.send(Message::text(completed.to_string())).await.unwrap();
        }
        closed.notify_one();
    });

    let client = OpenAIClient::new("test-key").with_base_url(format!("http://{addr}/v1"));
    set_default_openai_client(client, false);
    set_default_openai_responses_transport(OpenAIResponsesTransport::Websocket);
    let agent = Agent::<()>::new("test").with_model("gpt-4.1-mini");
    let mut result = Runner::run_streamed(&agent, "hello");
    drain(&mut result).await;
    set_default_openai_responses_transport(OpenAIResponsesTransport::Http);
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    tokio::time::timeout(Duration::from_secs(1), connection_closed.notified())
        .await
        .expect("the run closed the websocket it opened");
}

// ---------------------------------------------------------------------------
// test_agent_runner_sync.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_sync_errors_when_loop_already_running() {
    let agent = Agent::<()>::new("test-agent").with_model_instance(Arc::new(ScriptedModel::new()));
    let error = Runner::run_sync(&agent, "input").expect_err("inside a runtime");
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
}

// ---------------------------------------------------------------------------
// test_run.py
// ---------------------------------------------------------------------------

/// A context type only this test uses, so its default runner cannot affect other tests.
#[derive(Debug, Default)]
struct RunnerCtx;

#[derive(Default)]
struct RecordingRunner {
    runs: AtomicUsize,
    streamed: AtomicUsize,
}

impl AgentRunner<RunnerCtx> for RecordingRunner {
    fn run(
        &self,
        agent: Arc<Agent<RunnerCtx>>,
        input: RunInput,
        options: RunOptions<RunnerCtx>,
    ) -> BoxFuture<Result<RunResult<RunnerCtx>>> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        DefaultAgentRunner.run(agent, input, options)
    }

    fn run_streamed(
        &self,
        agent: Arc<Agent<RunnerCtx>>,
        input: RunInput,
        options: RunOptions<RunnerCtx>,
    ) -> RunResultStreaming<RunnerCtx> {
        self.streamed.fetch_add(1, Ordering::SeqCst);
        DefaultAgentRunner.run_streamed(agent, input, options)
    }
}

#[test]
fn static_run_methods_call_into_default_runner() {
    setup();
    let runner = Arc::new(RecordingRunner::default());
    openai_agents_sdk::set_default_agent_runner::<RunnerCtx>(Some(runner.clone()));
    let model = scripted(vec![
        vec![assistant_message("one")],
        vec![assistant_message("two")],
        vec![assistant_message("three")],
    ]);
    let agent = Agent::<RunnerCtx>::new("test").with_model_instance(model);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        Runner::run(&agent, "test").await.unwrap();
        assert_eq!(runner.runs.load(Ordering::SeqCst), 1);

        let mut streamed = Runner::run_streamed(&agent, "test");
        streamed.finish().await.unwrap();
        assert_eq!(runner.streamed.load(Ordering::SeqCst), 1);
    });
    drop(rt);

    // `run_sync` goes through the installed runner's `run`.
    Runner::run_sync(&agent, "test").unwrap();
    assert_eq!(runner.runs.load(Ordering::SeqCst), 2);
    openai_agents_sdk::set_default_agent_runner::<RunnerCtx>(None);
}

#[tokio::test]
async fn run_preserves_duplicate_user_messages() {
    setup();
    let model = scripted(vec![vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let input = vec![InputItem::user("repeat"), InputItem::user("repeat")];
    Runner::run(&agent, input).await.unwrap();
    let sent = model.last_call().unwrap().request.input;
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].text_content().as_deref(), Some("repeat"));
    assert_eq!(sent[1].text_content().as_deref(), Some("repeat"));
}

// ---------------------------------------------------------------------------
// test_repl.py
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

/// A `Write` sink the test can read back.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

async fn demo_loop(agent: Agent<()>, stream: bool, lines: &str) -> String {
    let output = Captured::default();
    openai_agents_sdk::repl::run_demo_loop_io(
        Arc::new(agent),
        stream,
        (),
        Some(openai_agents_sdk::DEFAULT_MAX_TURNS),
        tokio::io::BufReader::new(lines.as_bytes()),
        output.clone(),
    )
    .await
    .unwrap();
    output.text()
}

#[tokio::test]
async fn run_demo_loop_conversation() {
    setup();
    let model = scripted(vec![
        vec![assistant_message("hello")],
        vec![assistant_message("good")],
    ]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let output = demo_loop(agent, false, "Hi\nHow are you?\nquit\n").await;
    assert!(output.contains("hello"));
    assert!(output.contains("good"));
    let input = model.last_call().unwrap().request.input;
    let shape: Vec<(Option<String>, Option<String>)> = input
        .iter()
        .map(|i| (i.role().map(str::to_string), i.text_content()))
        .collect();
    assert_eq!(
        shape,
        vec![
            (Some("user".into()), Some("Hi".into())),
            (Some("assistant".into()), Some("hello".into())),
            (Some("user".into()), Some("How are you?".into())),
        ]
    );
}

#[tokio::test]
async fn run_demo_loop_streaming() {
    setup();
    let model = scripted(vec![]);
    let target = Agent::<()>::new("target").with_model_instance(model.clone());
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(FunctionTool::<()>::new(
            "foo",
            "a tool",
            |_ctx, _a: AnyArgs| async { Ok("tool_result".to_string()) },
        ))
        .with_handoff(target);
    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("foo", "{}")]),
            ScriptedStep::output(vec![function_call(
                openai_agents_sdk::handoffs::default_tool_name("target"),
                "{}",
            )]),
            ScriptedStep::output(vec![assistant_message("all done")]),
        ])
        .unwrap();
    let output = demo_loop(agent, true, "Hello\nexit\n").await;
    assert!(output.contains("all done"), "{output}");
    assert!(output.contains("[tool called]"), "{output}");
    assert!(output.contains("[tool output: tool_result]"), "{output}");
    assert!(output.contains("[Agent updated: target]"), "{output}");
}

#[tokio::test]
async fn run_demo_loop_exits_on_eof() {
    setup();
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    demo_loop(agent, false, "").await;
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn run_demo_loop_skips_empty_input() {
    setup();
    let model = scripted(vec![vec![assistant_message("hello")]]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let output = demo_loop(agent, false, "\nHi\nquit\n").await;
    assert!(output.contains("hello"));
    let input = model.last_call().unwrap().request.input;
    assert_eq!(input.len(), 1);
    assert_eq!(input[0].text_content().as_deref(), Some("Hi"));
}
