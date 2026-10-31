//! Port of Python `tests/realtime/test_session_tool_outputs.py`: tool output serialization,
//! delivery and retry ownership.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModel, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeRunConfig,
    RealtimeRunner, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::{FunctionTool, ToolOutput};
use openai_agents_sdk::{AgentsError, Result};
use serde::Serialize;
use serde_json::{Value, json};

const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Records everything on an inner fake, but fails `tool_output` sends while `fail` is set. The
/// failed send is not recorded, like Python's `FailingToolOutputModel`.
#[derive(Debug)]
struct FailingToolOutputModel {
    inner: Arc<FakeRealtimeModel>,
    fail: AtomicBool,
}

impl FailingToolOutputModel {
    fn new(fail: bool) -> Arc<Self> {
        Arc::new(FailingToolOutputModel {
            inner: Arc::new(FakeRealtimeModel::new()),
            fail: AtomicBool::new(fail),
        })
    }
}

#[async_trait]
impl RealtimeModel for FailingToolOutputModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        self.inner.connect(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        if event.as_tool_output().is_some() && self.fail.load(Ordering::SeqCst) {
            return Err(AgentsError::other("send failed"));
        }
        self.inner.send_event(event).await
    }

    async fn close(&self) -> Result<()> {
        self.inner.close().await
    }
}

async fn start(
    model: Arc<dyn RealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    async_tool_calls: bool,
) -> RealtimeSession<()> {
    let config = RealtimeRunConfig {
        async_tool_calls,
        ..Default::default()
    };
    RealtimeRunner::new(agent)
        .with_model(model)
        .with_config(config)
        .run(())
        .await
        .expect("session connects")
}

async fn wait_for<F>(session: &RealtimeSession<()>, predicate: F) -> RealtimeSessionEvent<()>
where
    F: Fn(&RealtimeSessionEvent<()>) -> bool,
{
    tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            let event = session.recv().await.expect("event stream closed early");
            if predicate(&event) {
                return event;
            }
        }
    })
    .await
    .expect("timed out waiting for a session event")
}

async fn error_containing(session: &RealtimeSession<()>, needle: &str) -> String {
    let event = wait_for(session, |event| match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"]
            .as_str()
            .is_some_and(|message| message.contains(needle)),
        _ => false,
    })
    .await;
    match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"].as_str().unwrap().to_string(),
        _ => unreachable!(),
    }
}

async fn eventually(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(EVENT_TIMEOUT, async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("condition never became true");
}

fn call(name: &str, call_id: &str, arguments: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(name, call_id, arguments))
}

fn counted_tool(
    name: &str,
    counter: Arc<AtomicUsize>,
    result: Result<&'static str>,
) -> FunctionTool<()> {
    let result = Arc::new(std::sync::Mutex::new(Some(result)));
    FunctionTool::new_no_args(name, "counted", move |_ctx| {
        counter.fetch_add(1, Ordering::SeqCst);
        let out = match result.lock().unwrap().as_ref().unwrap() {
            Ok(text) => Ok(text.to_string()),
            Err(error) => Err(AgentsError::other(error.to_string())),
        };
        async move { out }
    })
}

#[tokio::test]
async fn approved_function_tool_failure_replay_does_not_rerun() {
    let model = Arc::new(FakeRealtimeModel::new());
    let counter = Arc::new(AtomicUsize::new(0));
    let tool = counted_tool(
        "test_function",
        counter.clone(),
        Err(AgentsError::other("failed after side effect")),
    )
    .with_needs_approval(true)
    .without_failure_error_function();
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(model.clone(), agent, false).await;

    model.emit(call("test_function", "call_failed", "{}"));
    wait_for(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;

    let error = session
        .approve_tool_call("call_failed", false)
        .await
        .expect_err("the failing tool propagates");
    assert!(error.to_string().contains("failed after side effect"));

    model.emit(call("test_function", "call_failed", "{}"));
    error_containing(&session, "already executed").await;

    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(model.tool_outputs().is_empty());
}

async fn send_failure_retries_cached_output(
    async_tool_calls: bool,
    always: bool,
    change_tool_name: bool,
) {
    let model = FailingToolOutputModel::new(true);
    let counter = Arc::new(AtomicUsize::new(0));
    let tool = counted_tool("test_function", counter.clone(), Ok("result"))
        .with_needs_approval(true)
        .without_failure_error_function();
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(model.clone(), agent, async_tool_calls).await;
    let fake = model.inner.clone();

    fake.emit(call("test_function", "call_retry_output", "{}"));
    wait_for(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    // The tool runs, then delivering its output fails; the output is cached.
    let _ = session.approve_tool_call("call_retry_output", always).await;
    error_containing(&session, "cached output will be retried").await;
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(fake.tool_outputs().is_empty());

    // Reusing the id for a different invocation is refused.
    let (name, arguments) = if change_tool_name {
        ("other_function", "{}")
    } else {
        ("test_function", r#"{"changed":true}"#)
    };
    fake.emit(call(name, "call_retry_output", arguments));
    error_containing(&session, "unique call ID").await;

    // The exact same invocation resends the cached output without rerunning the tool.
    model.fail.store(false, Ordering::SeqCst);
    fake.emit(call("test_function", "call_retry_output", "{}"));
    eventually(|| fake.tool_outputs().len() == 1).await;
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert_eq!(
        fake.tool_outputs(),
        vec![("call_retry_output".to_string(), "result".to_string())]
    );
}

#[tokio::test]
async fn function_tool_send_failure_retries_cached_output_without_rerun() {
    for always in [false, true] {
        for change_tool_name in [false, true] {
            send_failure_retries_cached_output(false, always, change_tool_name).await;
        }
    }
}

#[tokio::test]
async fn async_function_tool_send_failure_retries_cached_output_without_rerun() {
    for always in [false, true] {
        for change_tool_name in [false, true] {
            send_failure_retries_cached_output(true, always, change_tool_name).await;
        }
    }
}

#[tokio::test]
async fn pending_function_output_rejects_handoff_role_reuse() {
    let model = FailingToolOutputModel::new(true);
    let counter = Arc::new(AtomicUsize::new(0));
    let target = RealtimeAgent::<()>::new("target").into_arc();
    let route_name = "transfer_to_target";
    let function_agent = RealtimeAgent::<()>::new("function")
        .with_tool(counted_tool(
            route_name,
            counter.clone(),
            Ok("function result"),
        ))
        .into_arc();
    let handoff_agent = RealtimeAgent::<()>::new("handoff")
        .with_handoff(target)
        .into_arc();
    let session = start(model.clone(), function_agent, false).await;
    let fake = model.inner.clone();

    fake.emit(call(route_name, "shared", "{}"));
    error_containing(&session, "cached output will be retried").await;
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    session.update_agent(handoff_agent).await.unwrap();
    fake.emit(call(route_name, "shared", "{}"));
    error_containing(&session, "unique call ID").await;

    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert_eq!(session.current_agent_name(), "handoff");
}

async fn sent_output_for(result: Value) -> String {
    let model = Arc::new(FakeRealtimeModel::new());
    let tool = FunctionTool::<()>::new_no_args("test_function", "returns data", move |_ctx| {
        let result = result.clone();
        async move { Ok(ToolOutput::Json(result)) }
    });
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let _session = start(model.clone(), agent, true).await;
    model.emit(call("test_function", "call_conversion", "{}"));
    eventually(|| !model.tool_outputs().is_empty()).await;
    model.tool_outputs()[0].1.clone()
}

#[tokio::test]
async fn tool_result_conversion_to_string() {
    // Python compares against `json.dumps`; the Rust output is compact JSON, so compare parsed.
    let sent = sent_output_for(json!({"result": "data", "count": 42})).await;
    assert_eq!(
        serde_json::from_str::<Value>(&sent).unwrap(),
        json!({"result": "data", "count": 42})
    );
}

#[derive(Serialize)]
struct ToolResult {
    name: String,
    score: i64,
}

#[tokio::test]
async fn tool_result_conversion_serializes_structured_models() {
    let value = ToolOutput::json(&ToolResult {
        name: "demo".into(),
        score: 7,
    })
    .unwrap();
    let ToolOutput::Json(value) = value else {
        panic!("expected a JSON output");
    };
    let sent = sent_output_for(value).await;
    assert_eq!(
        serde_json::from_str::<Value>(&sent).unwrap(),
        json!({"name": "demo", "score": 7})
    );
}

#[derive(Serialize)]
struct LabelledValues {
    label: String,
    values: Vec<i64>,
}

#[test]
fn serialize_tool_output_edge_cases() {
    assert_eq!(ToolOutput::Json(Value::Null).to_model_string(), "null");
    assert_eq!(
        ToolOutput::Json(json!(["hello", 1, true, null])).to_model_string(),
        r#"["hello",1,true,null]"#
    );
    let dataclass_like = ToolOutput::json(&LabelledValues {
        label: "demo".into(),
        values: vec![1, 2],
    })
    .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&dataclass_like.to_model_string()).unwrap(),
        json!({"label": "demo", "values": [1, 2]})
    );
}
