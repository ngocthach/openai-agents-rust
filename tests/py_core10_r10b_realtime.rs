//! Realtime tool failure formatters that await or fail.
//!
//! Ports `tests/realtime/test_session.py::test_function_tool_timeout_uses_async_error_function_result`
//! and the re-raise contract of `failure_error_function` for realtime function tools: a
//! formatter that fails ends the call with its own error instead of sending an output.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
    RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::{
    FunctionTool, ToolTimeoutBehavior, async_tool_error_fn, try_tool_error_fn,
};
use serde_json::json;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

fn sync_config() -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    }
}

fn call(name: &str, call_id: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(name, call_id, "{}"))
}

async fn start(
    model: &Arc<FakeRealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    run: RealtimeRunConfig<()>,
) -> RealtimeSession<()> {
    RealtimeSession::connect(model.clone(), agent, (), None, Some(run))
        .await
        .expect("session connects")
}

async fn until(session: &RealtimeSession<()>, predicate: impl Fn(&Event) -> bool) -> Vec<Event> {
    let mut seen = Vec::new();
    loop {
        let event = tokio::time::timeout(WAIT, session.recv())
            .await
            .expect("timed out waiting for a session event")
            .expect("session stream ended");
        let done = predicate(&event);
        seen.push(event);
        if done {
            return seen;
        }
    }
}

fn is_tool_end(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::ToolEnd { .. })
}

fn is_error(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::Error { .. })
}

fn outputs(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .sent_events()
        .iter()
        .filter_map(|event| event.as_tool_output().map(|(_, o, _)| o.to_string()))
        .collect()
}

fn failing_tool() -> FunctionTool<()> {
    FunctionTool::new_raw(
        "boom",
        "always fails",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move { Err::<String, _>(AgentsError::user("kapow")) },
    )
}

fn slow_tool() -> FunctionTool<()> {
    FunctionTool::new_raw(
        "slow_tool",
        "slow",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok("done".to_string())
        },
    )
    .with_timeout(Duration::from_millis(10))
    .with_timeout_behavior(ToolTimeoutBehavior::ErrorAsResult)
}

/// Python: `test_function_tool_timeout_uses_async_error_function_result`.
#[tokio::test]
async fn an_async_timeout_formatter_supplies_the_output() {
    let model = Arc::new(FakeRealtimeModel::new());
    let tool = slow_tool().with_timeout_error_handler(async_tool_error_fn(|_ctx, error| {
        let text = match error {
            AgentsError::ToolTimeoutError {
                tool_name,
                timeout_seconds,
                ..
            } => format!("async-timeout:{tool_name}:{timeout_seconds}"),
            other => panic!("unexpected {other:?}"),
        };
        Box::pin(async move {
            tokio::task::yield_now().await;
            Ok(text)
        })
    }));
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, sync_config()).await;
    model.emit(call("slow_tool", "call_timeout_custom"));
    until(&session, is_tool_end).await;
    assert_eq!(outputs(&model), ["async-timeout:slow_tool:0.01"]);
}

#[tokio::test]
async fn an_async_failure_formatter_supplies_the_output() {
    for config in [sync_config(), Default::default()] {
        let model = Arc::new(FakeRealtimeModel::new());
        let tool = failing_tool().with_failure_error_handler(async_tool_error_fn(|_ctx, error| {
            let text = format!("handled:{error}");
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(text)
            })
        }));
        let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
        let session = start(&model, agent, config).await;
        model.emit(call("boom", "call_failure"));
        until(&session, is_tool_end).await;
        assert_eq!(outputs(&model), ["handled:kapow"]);
    }
}

/// A failure formatter that fails sends no output and surfaces its own error.
#[tokio::test]
async fn a_failing_failure_formatter_ends_the_call_with_its_error() {
    for asynchronous in [false, true] {
        for config in [sync_config(), Default::default()] {
            let model = Arc::new(FakeRealtimeModel::new());
            let handler = if asynchronous {
                async_tool_error_fn(|_ctx, _e| {
                    Box::pin(async {
                        tokio::task::yield_now().await;
                        Err(AgentsError::other("formatter failed"))
                    })
                })
            } else {
                try_tool_error_fn(|_ctx, _e| Err(AgentsError::other("formatter failed")))
            };
            let agent = RealtimeAgent::<()>::new("agent")
                .with_tool(failing_tool().with_failure_error_handler(handler))
                .into_arc();
            let session = start(&model, agent, config).await;
            model.emit(call("boom", "call_failure"));
            until(&session, is_error).await;
            assert!(outputs(&model).is_empty());
            while session.recv().await.is_some() {}
            match session.take_error() {
                Some(AgentsError::Other(_)) => {}
                other => panic!("expected the formatter error, got {other:?}"),
            }
        }
    }
}

/// A timeout formatter that fails also ends the call with its own error.
#[tokio::test]
async fn a_failing_timeout_formatter_ends_the_call_with_its_error() {
    let model = Arc::new(FakeRealtimeModel::new());
    let tool = slow_tool().with_timeout_error_handler(try_tool_error_fn(|_ctx, _e| {
        Err(AgentsError::other("timeout formatter failed"))
    }));
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, sync_config()).await;
    model.emit(call("slow_tool", "call_timeout_fail"));
    until(&session, is_error).await;
    assert!(outputs(&model).is_empty());
    while session.recv().await.is_some() {}
    match session.take_error() {
        Some(AgentsError::Other(_)) => {}
        other => panic!("expected the formatter error, got {other:?}"),
    }
}
