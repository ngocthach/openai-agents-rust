//! Ports of the `on_stream` cases of `test_agent_as_tool.py`.
//!
//! Python fakes the nested `RunResultStreaming`; these tests script the nested model's raw event
//! stream instead, so the real streamed runner produces the events. A Rust handler returns a
//! future, so "sync handler" is the same case as an async one. `BaseException`/`CancelledError`
//! raised by a handler, and a failure injected into `RunResultStreaming.stream_events`'s close,
//! have no Rust equivalent: a handler that panics is logged like a Python `Exception`, and
//! cancellation is dropping the tool future.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use futures::stream::BoxStream;
use openai_agents_sdk::agent::{AgentAsToolOptions, AgentToolStreamEvent};
use openai_agents_sdk::items::{FunctionCall, ModelResponse, OutputItem};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
    response_to_stream_events,
};
use openai_agents_sdk::tool::{FunctionTool, ToolOutput};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, AgentsError, Result, RunContext, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

fn setup() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn tool_call(name: &str, call_id: &str, args: &str) -> FunctionCall {
    FunctionCall {
        id: Some(format!("fc_{call_id}")),
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments: args.to_string(),
        status: None,
        extra: Default::default(),
    }
}

fn call_ctx(name: &str, call_id: &str, args: &str) -> ToolContext<()> {
    ToolContext::from_call(
        RunContext::new(()),
        "outer",
        &tool_call(name, call_id, args),
    )
}

async fn invoke_text(tool: &FunctionTool<()>, ctx: ToolContext<()>, args: &str) -> Result<String> {
    match tool.invoke(ctx, args.to_string()).await? {
        ToolOutput::Text(text) => Ok(text),
        other => panic!("expected text output, got {other:?}"),
    }
}

fn delta(index: usize) -> Value {
    json!({
        "type": "response.output_text.delta",
        "delta": "synthetic delta",
        "item_id": "msg_synthetic",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": index,
        "logprobs": [],
    })
}

/// A scripted model whose single streamed turn is `stream`.
fn streaming_agent(
    name: &str,
    stream: impl Fn() -> BoxStream<'static, Result<Value>> + Send + Sync + 'static,
) -> (Agent<()>, Arc<ScriptedModel>) {
    setup();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::stream_with(move |_call| stream()))
        .unwrap();
    (
        Agent::<()>::new(name).with_model_instance(model.clone()),
        model,
    )
}

/// Sets a flag when dropped: a stream or handler future being cancelled.
struct FlagOnDrop(Arc<AtomicBool>);

impl Drop for FlagOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

async fn eventually(flag: &AtomicBool, what: &str) {
    for _ in 0..200 {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

fn ok_events() -> Vec<Value> {
    response_to_stream_events(&ModelResponse {
        output: vec![assistant_message("ok")],
        usage: Usage::default(),
        response_id: Some("resp".to_string()),
        request_id: None,
        raw_usage: None,
    })
}

// ------------------------------------------------------------ events and results

/// Python: `test_agent_as_tool_streams_events_with_on_stream`,
/// `test_agent_as_tool_streaming_accepts_sync_handler` and
/// `test_agent_as_tool_streaming_sets_tool_call_from_context`.
#[tokio::test]
async fn on_stream_receives_every_nested_event_with_the_agent_and_tool_call() {
    setup();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("streamed output")]);
    let agent = Agent::<()>::new("streamer").with_model_instance(model.clone());
    let received = Arc::new(Mutex::new(
        Vec::<(String, String, Option<FunctionCall>)>::new(),
    ));
    let record = received.clone();
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .on_stream(move |payload: AgentToolStreamEvent<()>| {
                let record = record.clone();
                async move {
                    record.lock().unwrap().push((
                        payload.event.event_type().to_string(),
                        payload.agent.name.clone(),
                        payload.tool_call.clone(),
                    ));
                }
            }),
    );
    let args = r#"{"input": "run streaming"}"#;
    let call = tool_call("stream_tool", "call-123", args);
    let ctx = ToolContext::from_call(RunContext::new(()), "outer", &call);
    let output = invoke_text(&tool, ctx, args).await.unwrap();

    assert_eq!(output, "streamed output");
    let received = received.lock().unwrap();
    assert!(received.len() >= 3, "{received:?}");
    assert_eq!(received[0].0, "agent_updated_stream_event");
    assert!(received.iter().any(|e| e.0 == "raw_response_event"));
    assert!(received.iter().all(|e| e.1 == "streamer"));
    assert!(received.iter().all(|e| e.2.as_ref() == Some(&call)));
    assert_eq!(
        model.calls()[0].input[0].text_content().as_deref(),
        Some("run streaming")
    );
}

/// Python: `test_agent_as_tool_streaming_updates_agent_on_handoff`.
#[tokio::test]
async fn the_event_agent_follows_a_nested_handoff() {
    setup();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "transfer_to_delegate",
        "{}",
        "handoff_call",
    )]);
    model.add_turn(vec![assistant_message("delegated output")]);
    let delegate = Agent::<()>::new("delegate").with_model_instance(model.clone());
    let primary = Agent::<()>::new("primary")
        .with_handoff(delegate)
        .with_model_instance(model.clone());
    let seen = Arc::new(Mutex::new(Vec::<(String, Option<String>)>::new()));
    let record = seen.clone();
    let tool = primary.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("delegate_tool")
            .tool_description("Streams handoff events")
            .on_stream(move |payload: AgentToolStreamEvent<()>| {
                let record = record.clone();
                async move {
                    let updated = match &payload.event {
                        StreamEvent::AgentUpdated { new_agent } => Some(new_agent.name.clone()),
                        _ => None,
                    };
                    record
                        .lock()
                        .unwrap()
                        .push((payload.agent.name.clone(), updated));
                }
            }),
    );
    let args = r#"{"input": "handoff"}"#;
    let output = invoke_text(
        &tool,
        call_ctx("delegate_tool", "call-delegate", args),
        args,
    )
    .await
    .unwrap();
    assert_eq!(output, "delegated output");

    let seen = seen.lock().unwrap();
    let first_update = seen
        .iter()
        .position(|(_, updated)| updated.as_deref() == Some("primary"))
        .expect("the run announces its starting agent");
    let handoff_update = seen
        .iter()
        .position(|(_, updated)| updated.as_deref() == Some("delegate"))
        .expect("the handoff announces the new agent");
    assert_eq!(first_update, 0);
    assert!(
        seen[..handoff_update]
            .iter()
            .all(|(agent, _)| agent == "primary")
    );
    assert!(
        seen[handoff_update..]
            .iter()
            .all(|(agent, _)| agent == "delegate")
    );
}

/// Python: `test_agent_as_tool_streaming_works_with_custom_extractor` and
/// `test_agent_as_tool_streaming_extractor_can_access_agent_tool_invocation`.
#[tokio::test]
async fn streaming_runs_feed_the_custom_extractor_and_expose_the_invocation() {
    setup();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("raw output")]);
    let agent = Agent::<()>::new("streamer").with_model_instance(model);
    let callbacks = Arc::new(AtomicUsize::new(0));
    let count = callbacks.clone();
    let invocation = Arc::new(Mutex::new(None));
    let record = invocation.clone();
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .custom_output_extractor(move |result| {
                let record = record.clone();
                async move {
                    *record.lock().unwrap() = result.agent_tool_invocation.clone();
                    Ok("custom value".to_string())
                }
            })
            .on_stream(move |_payload| {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            }),
    );
    let args = r#"{"input": "go"}"#;
    let output = invoke_text(
        &tool,
        call_ctx("stream_tool", "call-stream-123", args),
        args,
    )
    .await
    .unwrap();
    assert_eq!(output, "custom value");
    assert!(callbacks.load(Ordering::SeqCst) > 0);
    let invocation = invocation.lock().unwrap().clone().expect("an invocation");
    assert_eq!(invocation.tool_call_id, "call-stream-123");
    assert_eq!(invocation.tool_name, "stream_tool");
    assert_eq!(invocation.tool_arguments, args);
}

fn message_with_segments(segments: &[&str]) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "message", "id": "msg_multi_segment", "role": "assistant",
        "status": "completed",
        "content": segments.iter().map(|text| json!({
            "type": "output_text", "text": text, "annotations": [], "logprobs": [],
        })).collect::<Vec<_>>(),
    }))
}

fn noop_handler() -> impl Fn(AgentToolStreamEvent<()>) -> std::future::Ready<()> + Send + Sync {
    |_payload| std::future::ready(())
}

/// Python: `test_agent_as_tool_streaming_settles_multi_segment_text_output`.
#[tokio::test]
async fn streaming_joins_a_multi_segment_text_output() {
    setup();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![message_with_segments(&["first ", "second"])]);
    let agent = Agent::<()>::new("streamer").with_model_instance(model);
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .on_stream(noop_handler()),
    );
    let args = r#"{"input": "go"}"#;
    let output = invoke_text(
        &tool,
        call_ctx("stream_tool", "call-settle-text", args),
        args,
    )
    .await
    .unwrap();
    assert_eq!(output, "first second");
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct StructuredOutput {
    answer: String,
}

/// Python: `test_agent_as_tool_streaming_settles_multi_segment_structured_output`.
#[tokio::test]
async fn streaming_joins_a_multi_segment_structured_output() {
    setup();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![message_with_segments(&[
        r#"{"answer":"str"#,
        r#"uctured"}"#,
    ])]);
    let agent = Agent::<()>::new("streamer")
        .with_model_instance(model)
        .with_output_type::<StructuredOutput>();
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .on_stream(noop_handler()),
    );
    let args = r#"{"input": "go"}"#;
    let output = invoke_text(
        &tool,
        call_ctx("stream_tool", "call-settle-structured", args),
        args,
    )
    .await
    .unwrap();
    let parsed: Value = serde_json::from_str(&output).unwrap();
    assert_eq!(parsed, json!({"answer": "structured"}));
}

/// Python: `test_agent_as_tool_streaming_settles_final_text_after_nested_mcp_failure` (the
/// `mcp-error` case; a cancelled MCP call is a Python task-cancellation case).
#[cfg(feature = "mcp")]
#[tokio::test]
async fn streaming_settles_the_final_text_after_a_nested_mcp_failure() {
    use async_trait::async_trait;
    use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
    use serde_json::Map;

    #[derive(Debug)]
    struct ErrorServer {
        config: MCPServerConfig,
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl MCPServer for ErrorServer {
        fn name(&self) -> &str {
            "nested"
        }
        fn config(&self) -> &MCPServerConfig {
            &self.config
        }
        async fn connect(&self) -> Result<()> {
            Ok(())
        }
        async fn cleanup(&self) -> Result<()> {
            Ok(())
        }
        async fn list_tools(&self) -> Result<Vec<MCPTool>> {
            Ok(vec![MCPTool {
                meta: None,
                name: "error_tool".into(),
                title: None,
                description: Some("fails".into()),
                input_schema: json!({"type": "object", "properties": {}}),
                annotations: None,
            }])
        }
        async fn call_tool(
            &self,
            tool_name: &str,
            _arguments: Option<Map<String, Value>>,
        ) -> Result<Value> {
            self.calls.lock().unwrap().push(tool_name.to_string());
            Err(AgentsError::other("synthetic upstream 422"))
        }
        async fn list_prompts(&self) -> Result<Vec<Value>> {
            Ok(Vec::new())
        }
        async fn get_prompt(
            &self,
            _name: &str,
            _arguments: Option<std::collections::BTreeMap<String, String>>,
        ) -> Result<Value> {
            Err(AgentsError::user("no prompts"))
        }
    }

    setup();
    let server = Arc::new(ErrorServer {
        config: MCPServerConfig::default(),
        calls: Mutex::new(Vec::new()),
    });
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id("error_tool", "{}", "mcp_call")]);
    model.add_turn(vec![message_with_segments(&["first ", "second"])]);
    let agent = Agent::<()>::new("streamer")
        .with_mcp_server(server.clone())
        .with_model_instance(model);
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .on_stream(noop_handler()),
    );
    let args = r#"{"input": "go"}"#;
    let output = invoke_text(
        &tool,
        call_ctx("stream_tool", "call-nested-error", args),
        args,
    )
    .await
    .unwrap();
    assert_eq!(*server.calls.lock().unwrap(), ["error_tool"]);
    assert_eq!(output, "first second");
}

/// Python: `test_agent_as_tool_without_stream_uses_run`, and the streamed counterpart: the
/// nested run only streams when `on_stream` is set.
#[tokio::test]
async fn the_nested_run_streams_only_with_an_on_stream_handler() {
    for with_handler in [false, true] {
        setup();
        let model = Arc::new(ScriptedModel::new());
        model
            .enqueue(ScriptedStep::output(vec![assistant_message("plain")]))
            .unwrap();
        let agent = Agent::<()>::new("nostream_agent").with_model_instance(model.clone());
        let mut options = AgentAsToolOptions::new()
            .tool_name("nostream_tool")
            .tool_description("No streaming path");
        if with_handler {
            options = options.on_stream(noop_handler());
        }
        let tool = agent.as_tool_with(options);
        let args = r#"{"input": "plain"}"#;
        let output = invoke_text(&tool, call_ctx("nostream_tool", "call-no", args), args)
            .await
            .unwrap();
        assert_eq!(output, "plain");
        assert_eq!(model.calls().len(), 1);
        assert_eq!(model.calls()[0].streamed, with_handler);
    }
}

// ---------------------------------------------------------- dispatch and handlers

/// Python: `test_agent_as_tool_streaming_dispatches_without_blocking`.
#[tokio::test]
async fn a_slow_handler_does_not_block_the_nested_stream() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let second_yielded = Arc::new(AtomicBool::new(false));
    let yielded = second_yielded.clone();
    let (agent, _model) = streaming_agent("nonblocking_agent", move || {
        let yielded = yielded.clone();
        futures::stream::iter(ok_events().into_iter().enumerate())
            .map(move |(index, event)| {
                if index == 1 {
                    yielded.store(true, Ordering::SeqCst);
                }
                Ok(event)
            })
            .boxed()
    });
    let handled = Arc::new(AtomicUsize::new(0));
    let (handler_started, handler_release, count) =
        (started.clone(), release.clone(), handled.clone());
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("nonblocking_tool")
            .tool_description("Uses non-blocking streaming handler")
            .on_stream(move |_payload| {
                let (started, release, count) = (
                    handler_started.clone(),
                    handler_release.clone(),
                    count.clone(),
                );
                async move {
                    if count.fetch_add(1, Ordering::SeqCst) == 0 {
                        started.notify_one();
                        release.notified().await;
                    }
                }
            }),
    );
    let args = r#"{"input": "go"}"#;
    let invocation = tokio::spawn({
        let ctx = call_ctx("nonblocking_tool", "call-nonblocking", args);
        async move { invoke_text(&tool, ctx, args).await }
    });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("the handler starts");
    eventually(&second_yielded, "the nested stream's second event").await;
    assert!(!invocation.is_finished());

    release.notify_one();
    let output = tokio::time::timeout(Duration::from_secs(2), invocation)
        .await
        .expect("the call finishes")
        .unwrap()
        .unwrap();
    assert_eq!(output, "ok");
    assert!(handled.load(Ordering::SeqCst) > 1);
}

/// Python: `test_agent_as_tool_streaming_reraises_parent_cancellation_without_waiting_for_handler`.
#[tokio::test]
async fn cancelling_the_call_does_not_wait_for_the_handler() {
    let model_dropped = Arc::new(AtomicBool::new(false));
    let handler_dropped = Arc::new(AtomicBool::new(false));
    let handler_started = Arc::new(Notify::new());
    let dropped = model_dropped.clone();
    let (agent, _model) = streaming_agent("streamer", move || {
        let guard = FlagOnDrop(dropped.clone());
        let mut events = ok_events().into_iter();
        let first = events.next().unwrap();
        futures::stream::once(async move { Ok(first) })
            .chain(futures::stream::once(async move {
                let _guard = guard;
                std::future::pending::<Result<Value>>().await
            }))
            .boxed()
    });
    let (started, dropped_flag) = (handler_started.clone(), handler_dropped.clone());
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("stream_tool")
            .tool_description("Streams events")
            .on_stream(move |_payload| {
                let (started, dropped_flag) = (started.clone(), dropped_flag.clone());
                async move {
                    let _guard = FlagOnDrop(dropped_flag);
                    started.notify_one();
                    std::future::pending::<()>().await;
                }
            }),
    );
    let args = r#"{"input": "recover"}"#;
    let ctx = call_ctx("stream_tool", "call-cancelled", args);
    let mut invoke = Box::pin(invoke_text(&tool, ctx, args));
    tokio::select! {
        _ = invoke.as_mut() => panic!("the call cannot finish while the handler is blocked"),
        _ = handler_started.notified() => {}
    }
    // Cancelling the call is dropping it: it returns at once, stopping the handler and the
    // nested run.
    drop(invoke);
    assert!(handler_dropped.load(Ordering::SeqCst));
    eventually(&model_dropped, "the nested model stream to stop").await;
}

#[derive(Default)]
struct Records(Mutex<Vec<(log::Level, String)>>);

static RECORDS: OnceLock<Records> = OnceLock::new();
static LOG_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == "openai.agents" {
            RECORDS
                .get_or_init(Records::default)
                .0
                .lock()
                .unwrap()
                .push((record.level(), format!("{}", record.args())));
        }
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

fn captured() -> Vec<(log::Level, String)> {
    RECORDS
        .get_or_init(Records::default)
        .0
        .lock()
        .unwrap()
        .clone()
}

/// Python: `test_agent_as_tool_streaming_handler_exception_does_not_fail_call` (three
/// model/tool redaction settings).
#[tokio::test]
async fn a_handler_panic_is_logged_and_does_not_fail_the_call() {
    let _lock = LOG_LOCK.lock().await;
    install_logger();
    let secret = "SECRET_AGENT_STREAM_PAYLOAD";
    for (model_redacted, tool_redacted) in [(true, false), (false, true), (false, false)] {
        openai_agents_sdk::debug::set_dont_log_model_data(model_redacted);
        openai_agents_sdk::debug::set_dont_log_tool_data(tool_redacted);
        RECORDS
            .get_or_init(Records::default)
            .0
            .lock()
            .unwrap()
            .clear();

        setup();
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("ok")]);
        let agent = Agent::<()>::new("SECRET_HANDLER_ERROR_AGENT").with_model_instance(model);
        let tool = agent.as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("error_tool")
                .tool_description("Handler throws")
                .on_stream(move |_payload| async move {
                    panic!("{secret}");
                }),
        );
        let args = r#"{"input": "go"}"#;
        let output = invoke_text(&tool, call_ctx("error_tool", "call-bad", args), args).await;
        openai_agents_sdk::debug::reset_dont_log_model_data();
        openai_agents_sdk::debug::reset_dont_log_tool_data();
        assert_eq!(output.unwrap(), "ok");

        let records = captured();
        let record = records
            .iter()
            .find(|(_, message)| {
                message.contains("Error while handling an agent tool on_stream event")
            })
            .unwrap_or_else(|| panic!("no handler error was logged: {records:?}"));
        assert_eq!(record.0, log::Level::Error);
        let text: String = records
            .iter()
            .map(|(_, m)| m.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if model_redacted || tool_redacted {
            assert_eq!(
                record.1,
                "Error while handling an agent tool on_stream event"
            );
            assert!(!text.contains(secret), "{text}");
            assert!(!text.contains("SECRET_HANDLER_ERROR_AGENT"), "{text}");
        } else {
            assert!(record.1.contains(secret), "{}", record.1);
        }
    }
}

/// Python: `test_agent_as_tool_streaming_drains_emitted_events_before_stream_error`.
#[tokio::test]
async fn emitted_events_are_delivered_before_a_stream_error_is_raised() {
    let handler_started = Arc::new(Notify::new());
    let producer_failed = Arc::new(AtomicBool::new(false));
    let (started, failed) = (handler_started.clone(), producer_failed.clone());
    let (agent, _model) = streaming_agent("streamer", move || {
        let (started, failed) = (started.clone(), failed.clone());
        let mut events = ok_events().into_iter();
        let first = events.next().unwrap();
        futures::stream::once(async move { Ok(first) })
            .chain(futures::stream::once(async move {
                started.notified().await;
                failed.store(true, Ordering::SeqCst);
                Err(AgentsError::other("nested stream failed"))
            }))
            .boxed()
    });
    let release = Arc::new(Notify::new());
    let handler_cancelled = Arc::new(AtomicBool::new(false));
    let handled = Arc::new(AtomicUsize::new(0));
    let (handler_started2, release2, cancelled2, handled2) = (
        handler_started.clone(),
        release.clone(),
        handler_cancelled.clone(),
        handled.clone(),
    );
    let tool = agent
        .as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("stream_tool")
                .tool_description("Streams events")
                .on_stream(move |_payload| {
                    let (started, release, cancelled, handled) = (
                        handler_started2.clone(),
                        release2.clone(),
                        cancelled2.clone(),
                        handled2.clone(),
                    );
                    async move {
                        if handled.fetch_add(1, Ordering::SeqCst) == 0 {
                            started.notify_one();
                            let guard = FlagOnDrop(cancelled);
                            release.notified().await;
                            std::mem::forget(guard);
                        }
                    }
                }),
        )
        .without_failure_error_function();
    let args = r#"{"input": "go"}"#;
    let invocation = tokio::spawn({
        let ctx = call_ctx("stream_tool", "call-stream-error", args);
        async move { invoke_text(&tool, ctx, args).await }
    });
    eventually(&producer_failed, "the stream failure").await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!invocation.is_finished());
    assert!(!handler_cancelled.load(Ordering::SeqCst));

    release.notify_one();
    let error = tokio::time::timeout(Duration::from_secs(2), invocation)
        .await
        .expect("the call finishes")
        .unwrap()
        .expect_err("the stream error is raised");
    assert!(
        error.to_string().contains("nested stream failed"),
        "{error}"
    );
    // Every event emitted before the failure reached the handler.
    assert!(handled.load(Ordering::SeqCst) >= 2);
}

/// Python: `test_agent_as_tool_rejects_nonpositive_stream_backlog_limit` (`0`; a negative limit
/// is unrepresentable in a `usize`).
#[test]
fn a_zero_backlog_limit_is_rejected() {
    let error = Agent::<()>::new("streamer")
        .try_as_tool_with(AgentAsToolOptions::new().on_stream_max_pending_events(Some(0)))
        .expect_err("a zero limit is invalid");
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("on_stream_max_pending_events must be a positive integer"),
        "{error}"
    );
}

// ------------------------------------------------------------------------ backlog

fn failure_overload_handler(
    errors: Arc<Mutex<Vec<String>>>,
) -> impl Fn(&RunContext<()>, &AgentsError) -> String + Send + Sync + 'static {
    move |_ctx, error| {
        errors.lock().unwrap().push(format!("{error:?}"));
        "stream callback overloaded".to_string()
    }
}

/// Python: `test_agent_as_tool_stream_backlog_overflow_stops_nested_run` (default limit raising,
/// custom limit with a custom failure message, custom limit with the default failure message).
#[tokio::test]
async fn a_full_backlog_stops_the_nested_run_and_fails_the_call() {
    for (limit, policy) in [
        (None, "raise"),
        (Some(2usize), "custom"),
        (Some(2), "default"),
    ] {
        let expected_limit = limit.unwrap_or(1024);
        let handler_started = Arc::new(Notify::new());
        let model_dropped = Arc::new(AtomicBool::new(false));
        let (started, dropped) = (handler_started.clone(), model_dropped.clone());
        let (agent, _model) = streaming_agent("streamer", move || {
            let (started, guard) = (started.clone(), FlagOnDrop(dropped.clone()));
            futures::stream::once(async move {
                started.notified().await;
            })
            .flat_map(move |_| {
                futures::stream::iter((0..=expected_limit + 1).map(|i| Ok(delta(i))))
            })
            .chain(futures::stream::once(async move {
                let _guard = guard;
                std::future::pending::<Result<Value>>().await
            }))
            .boxed()
        });
        let handler_dropped = Arc::new(AtomicBool::new(false));
        let (started2, dropped2) = (handler_started.clone(), handler_dropped.clone());
        let errors = Arc::new(Mutex::new(Vec::new()));
        let extracted = Arc::new(AtomicUsize::new(0));
        let extracted2 = extracted.clone();
        let mut options = AgentAsToolOptions::new()
            .on_stream(move |_payload| {
                let (started, dropped) = (started2.clone(), dropped2.clone());
                async move {
                    let _guard = FlagOnDrop(dropped);
                    started.notify_one();
                    std::future::pending::<()>().await;
                }
            })
            .custom_output_extractor(move |_result| {
                extracted2.fetch_add(1, Ordering::SeqCst);
                async { Ok("must not succeed".to_string()) }
            });
        if let Some(limit) = limit {
            options = options.on_stream_max_pending_events(Some(limit));
        }
        if policy == "custom" {
            options = options.failure_error_function(failure_overload_handler(errors.clone()));
        }
        let mut tool = agent.as_tool_with(options);
        if policy == "raise" {
            tool = tool.without_failure_error_function();
        }
        let args = r#"{"input":"go"}"#;
        let label = format!("limit={limit:?} policy={policy}");
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            invoke_text(&tool, call_ctx("streamer", "call_overflow", args), args),
        )
        .await
        .unwrap_or_else(|_| panic!("{label}: the call hangs"));
        match policy {
            "raise" => {
                let error = outcome.expect_err(&label);
                assert!(
                    matches!(error, AgentsError::UserError { .. }),
                    "{label}: {error:?}"
                );
                assert!(
                    error
                        .to_string()
                        .contains(&format!("on_stream_max_pending_events={expected_limit}")),
                    "{label}: {error}"
                );
            }
            "custom" => {
                assert_eq!(outcome.unwrap(), "stream callback overloaded", "{label}");
                let errors = errors.lock().unwrap();
                assert_eq!(errors.len(), 1, "{label}");
                assert!(errors[0].contains("UserError"), "{label}: {errors:?}");
            }
            _ => {
                let output = outcome.unwrap();
                assert_eq!(
                    output, "An error occurred while running the tool. Please try again.",
                    "{label}"
                );
                assert!(!output.contains("synthetic delta"));
            }
        }
        assert!(handler_dropped.load(Ordering::SeqCst), "{label}");
        eventually(&model_dropped, "the nested model stream to stop").await;
        assert_eq!(extracted.load(Ordering::SeqCst), 0, "{label}");
    }
}

/// Python: `test_agent_as_tool_cancellation_with_backlog_stops_nested_run` (`parent` case; the
/// `handler` case raises `CancelledError` from the handler, which has no Rust equivalent).
#[tokio::test]
async fn cancelling_the_call_with_a_backlog_stops_the_nested_run() {
    let model_started = Arc::new(AtomicBool::new(false));
    let model_dropped = Arc::new(AtomicBool::new(false));
    let handler_dropped = Arc::new(AtomicBool::new(false));
    let handler_started = Arc::new(Notify::new());
    let (started, dropped) = (model_started.clone(), model_dropped.clone());
    let (agent, _model) = streaming_agent("streamer", move || {
        started.store(true, Ordering::SeqCst);
        let guard = FlagOnDrop(dropped.clone());
        futures::stream::once(async { Ok(delta(0)) })
            .chain(
                futures::stream::once(async {
                    // The burst arrives once the handler had time to stall on the first event.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                })
                .flat_map(|_| futures::stream::iter((1..10).map(|i| Ok(delta(i))))),
            )
            .chain(futures::stream::once(async move {
                let _guard = guard;
                std::future::pending::<Result<Value>>().await
            }))
            .boxed()
    });
    let (notify, flag) = (handler_started.clone(), handler_dropped.clone());
    let tool = agent
        .as_tool_with(
            AgentAsToolOptions::new()
                .on_stream(move |payload| {
                    let (notify, flag) = (notify.clone(), flag.clone());
                    async move {
                        if payload.event.event_type() == "raw_response_event" {
                            let _guard = FlagOnDrop(flag);
                            notify.notify_one();
                            std::future::pending::<()>().await;
                        }
                    }
                })
                .on_stream_max_pending_events(Some(1)),
        )
        .without_failure_error_function();
    let args = r#"{"input":"go"}"#;
    let ctx = call_ctx("streamer", "call_cancel_backlog", args);
    let mut invoke = Box::pin(invoke_text(&tool, ctx, args));
    tokio::select! {
        _ = invoke.as_mut() => panic!("the call cannot finish while the handler is blocked"),
        _ = handler_started.notified() => {}
    }
    drop(invoke);
    assert!(model_started.load(Ordering::SeqCst));
    assert!(handler_dropped.load(Ordering::SeqCst));
    eventually(&model_dropped, "the nested model stream to stop").await;
}

/// Python: `test_agent_as_tool_stream_backlog_delivers_all_pending_events` (`exact-custom-limit`
/// and `unlimited`).
#[tokio::test]
async fn a_backlog_at_the_limit_delivers_every_event_in_order() {
    for custom_limit in [true, false] {
        // Count the events a plain run emits, so the backlog can be sized to the limit exactly.
        let counted = Arc::new(AtomicUsize::new(0));
        let make_agent = |extra: usize| {
            streaming_agent("streamer", move || {
                let mut events = ok_events();
                let completed = events.pop().unwrap();
                let mut all: Vec<Value> = events.into_iter().collect();
                all.extend((0..extra).map(delta));
                all.push(completed);
                futures::stream::iter(all.into_iter().map(Ok)).boxed()
            })
            .0
        };
        let extra = if custom_limit { 0 } else { 1100 };
        {
            let counter = counted.clone();
            let tool = make_agent(extra).as_tool_with(AgentAsToolOptions::new().on_stream(
                move |_payload| {
                    let counter = counter.clone();
                    async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                },
            ));
            let args = r#"{"input":"go"}"#;
            invoke_text(&tool, call_ctx("streamer", "call_count", args), args)
                .await
                .unwrap();
        }
        let total = counted.load(Ordering::SeqCst);
        assert!(total > 2);

        let first_started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let delivered = Arc::new(Mutex::new(Vec::<String>::new()));
        let (started, rel, record) = (first_started.clone(), release.clone(), delivered.clone());
        let limit = custom_limit.then_some(total - 1);
        let tool = make_agent(extra).as_tool_with(
            AgentAsToolOptions::new()
                .on_stream(move |payload| {
                    let (started, release, record) = (started.clone(), rel.clone(), record.clone());
                    async move {
                        if record.lock().unwrap().is_empty() {
                            started.notify_one();
                            release.notified().await;
                        }
                        let tag = match &payload.event {
                            StreamEvent::RawResponse(v) => {
                                v["sequence_number"].to_string() + v["type"].as_str().unwrap_or("")
                            }
                            other => other.event_type().to_string(),
                        };
                        record.lock().unwrap().push(tag);
                    }
                })
                .on_stream_max_pending_events(limit),
        );
        let args = r#"{"input":"go"}"#;
        let ctx = call_ctx("streamer", "call_backlog", args);
        let invocation = tokio::spawn(async move { invoke_text(&tool, ctx, args).await });
        tokio::time::timeout(Duration::from_secs(2), first_started.notified())
            .await
            .expect("the handler starts");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!invocation.is_finished());
        assert!(delivered.lock().unwrap().is_empty());

        release.notify_one();
        let output = tokio::time::timeout(Duration::from_secs(5), invocation)
            .await
            .expect("the call finishes")
            .unwrap()
            .unwrap();
        assert_eq!(output, "ok");
        assert_eq!(delivered.lock().unwrap().len(), total);
    }
}

/// Python: `test_agent_as_tool_stream_backlog_allows_ready_handler_to_drain_burst`.
#[tokio::test]
async fn a_ready_handler_drains_a_burst_with_a_limit_of_one() {
    let (agent, _model) = streaming_agent("streamer", || {
        let mut events = ok_events();
        let completed = events.pop().unwrap();
        let mut all: Vec<Value> = events;
        all.extend((0..10).map(delta));
        all.push(completed);
        futures::stream::iter(all.into_iter().map(Ok)).boxed()
    });
    let delivered = Arc::new(AtomicUsize::new(0));
    let count = delivered.clone();
    let tool = agent
        .as_tool_with(
            AgentAsToolOptions::new()
                .on_stream(move |_payload| {
                    count.fetch_add(1, Ordering::SeqCst);
                    std::future::ready(())
                })
                .on_stream_max_pending_events(Some(1)),
        )
        .without_failure_error_function();
    let args = r#"{"input":"go"}"#;
    let output = invoke_text(&tool, call_ctx("streamer", "call_burst", args), args)
        .await
        .unwrap();
    assert_eq!(output, "ok");
    assert!(delivered.load(Ordering::SeqCst) >= 10);
}

/// Python: `test_agent_as_tool_stream_full_backlog_finishes_without_sentinel_deadlock`
/// (`source-error` and `parent-cancellation`).
#[tokio::test]
async fn a_full_backlog_does_not_deadlock_a_failing_stream() {
    for cancel_parent in [false, true] {
        let started = Arc::new(Notify::new());
        let source_finished = Arc::new(AtomicBool::new(false));
        let stream_started = started.clone();
        let finished = source_finished.clone();
        let (agent, _model) = streaming_agent("streamer", move || {
            let (started, finished) = (stream_started.clone(), finished.clone());
            let mut events = ok_events().into_iter();
            let (e0, e1) = (events.next().unwrap(), events.next().unwrap());
            futures::stream::once(async move { Ok(e0) })
                .chain(futures::stream::once(async move {
                    started.notified().await;
                    Ok(e1)
                }))
                .chain(futures::stream::once(async move {
                    finished.store(true, Ordering::SeqCst);
                    Err(AgentsError::other("source failed"))
                }))
                .boxed()
        });
        let release = Arc::new(Notify::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let handled = Arc::new(AtomicUsize::new(0));
        let (handler_started, rel, flag, count) = (
            started.clone(),
            release.clone(),
            cancelled.clone(),
            handled.clone(),
        );
        let tool = agent
            .as_tool_with(
                AgentAsToolOptions::new()
                    .on_stream(move |_payload| {
                        let (started, release, flag, count) = (
                            handler_started.clone(),
                            rel.clone(),
                            flag.clone(),
                            count.clone(),
                        );
                        async move {
                            // The first event is the run's own agent announcement; the nested
                            // stream's two events then fill a backlog of two.
                            if count.fetch_add(1, Ordering::SeqCst) == 0 {
                                started.notify_one();
                                let guard = FlagOnDrop(flag);
                                release.notified().await;
                                std::mem::forget(guard);
                            }
                        }
                    })
                    .on_stream_max_pending_events(Some(2)),
            )
            .without_failure_error_function();
        let args = r#"{"input":"go"}"#;
        let ctx = call_ctx("streamer", "call_full", args);
        let mut task = tokio::spawn(async move { invoke_text(&tool, ctx, args).await });
        eventually(&source_finished, "the source to finish").await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished());
        assert!(!cancelled.load(Ordering::SeqCst));
        if cancel_parent {
            task.abort();
            let _ = (&mut task).await;
            eventually(&cancelled, "the handler to be cancelled").await;
        } else {
            release.notify_one();
            let error = tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .expect("the call finishes")
                .unwrap()
                .expect_err("the source failure is raised");
            assert!(error.to_string().contains("source failed"), "{error}");
            assert!(handled.load(Ordering::SeqCst) >= 1);
            assert!(!cancelled.load(Ordering::SeqCst));
        }
    }
}
