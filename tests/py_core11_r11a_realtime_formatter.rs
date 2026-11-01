//! Ports of the realtime formatter cases of `tests/realtime/test_session_approvals.py`
//! (`test_rejection_formatter_error_is_redacted`) and the async-formatter paths of the realtime
//! session: the run-level formatter may await and may fail, and a failure keeps the default
//! message and never ends the session. Python logs "Tool error formatter failed" at error level,
//! with the detail only when tool-data logging is enabled.

#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
    RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{async_tool_error_formatter_fn, try_tool_error_formatter_fn};
use serde_json::{Value, json};

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);
const REJECTED: &str = "Tool execution was not approved.";

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap()
            .push(format!("{} {}", record.level(), record.args()));
    }
    fn flush(&self) {}
}

async fn fixture() -> tokio::sync::MutexGuard<'static, ()> {
    let guard = LOCK.lock().await;
    let _ = log::set_logger(&Capture);
    log::set_max_level(log::LevelFilter::Trace);
    RECORDS.lock().unwrap().clear();
    reset_dont_log_tool_data();
    guard
}

fn logs() -> String {
    RECORDS.lock().unwrap().join("\n")
}

fn guarded_tool() -> FunctionTool<()> {
    FunctionTool::new_raw(
        "test_function",
        "test tool",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async { Ok("ran".to_string()) },
    )
    .with_needs_approval(true)
}

async fn recv(session: &RealtimeSession<()>) -> Event {
    tokio::time::timeout(WAIT, session.recv())
        .await
        .expect("timed out waiting for a session event")
        .expect("session stream ended")
}

/// Open a session whose only tool needs approval, deliver one call and wait for the request.
async fn pending_session(
    run: RealtimeRunConfig<()>,
    call_id: &str,
) -> (Arc<FakeRealtimeModel>, RealtimeSession<()>) {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded_tool())
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(run))
        .await
        .expect("session connects");
    model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("test_function", call_id, "{}"),
    ));
    loop {
        if matches!(
            recv(&session).await,
            RealtimeSessionEvent::ToolApprovalRequired { .. }
        ) {
            return (model, session);
        }
    }
}

fn sent_outputs(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .sent_events()
        .iter()
        .filter_map(|event| {
            event
                .as_tool_output()
                .map(|(_, output, _)| output.to_string())
        })
        .collect()
}

/// Python: `test_rejection_formatter_error_is_redacted`: a rejection whose formatter fails is
/// answered with the default message, and the error log carries the sentence only.
#[tokio::test]
async fn a_failing_formatter_on_a_rejection_is_redacted() {
    let _guard = fixture().await;
    set_dont_log_tool_data(true);
    let run = RealtimeRunConfig {
        tool_error_formatter: Some(try_tool_error_formatter_fn(|_args| {
            Err(AgentsError::other("SECRET_REALTIME_TOOL_FORMATTER"))
        })),
        ..Default::default()
    };
    let (model, session) = pending_session(run, "call_reject_error").await;
    session
        .reject_tool_call("call_reject_error", false, None)
        .await
        .expect("a formatter failure does not fail the rejection");

    assert_eq!(sent_outputs(&model), [REJECTED]);
    let logs = logs();
    assert!(
        logs.lines()
            .any(|line| line.trim_end() == "ERROR Tool error formatter failed"),
        "{logs}"
    );
    assert!(!logs.contains("SECRET_REALTIME_TOOL_FORMATTER"), "{logs}");
    reset_dont_log_tool_data();
}

/// The same failure with tool-data logging on logs the formatter's error.
#[tokio::test]
async fn a_failing_formatter_on_a_rejection_logs_the_error_when_enabled() {
    let _guard = fixture().await;
    set_dont_log_tool_data(false);
    let run = RealtimeRunConfig {
        tool_error_formatter: Some(try_tool_error_formatter_fn(|_args| {
            Err(AgentsError::other("SECRET_REALTIME_TOOL_FORMATTER"))
        })),
        ..Default::default()
    };
    let (model, session) = pending_session(run, "call_reject_error").await;
    session
        .reject_tool_call("call_reject_error", false, None)
        .await
        .unwrap();

    assert_eq!(sent_outputs(&model), [REJECTED]);
    assert!(
        logs().contains("SECRET_REALTIME_TOOL_FORMATTER"),
        "{}",
        logs()
    );
    reset_dont_log_tool_data();
}

/// An async formatter is awaited for a rejection, and sees the call it is formatting.
#[tokio::test]
async fn an_async_formatter_is_awaited_for_a_rejection() {
    let _guard = fixture().await;
    let run = RealtimeRunConfig {
        tool_error_formatter: Some(async_tool_error_formatter_fn(|args| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(Some(format!(
                    "run-level {} denied ({})",
                    args.tool_name, args.call_id
                )))
            })
        })),
        ..Default::default()
    };
    let (model, session) = pending_session(run, "call_reject_async").await;
    session
        .reject_tool_call("call_reject_async", false, None)
        .await
        .unwrap();

    assert_eq!(
        sent_outputs(&model),
        ["run-level test_function denied (call_reject_async)"]
    );
}

/// A call to a tool the agent does not have goes through the formatter too, and a failing one
/// keeps the default "not found" text.
#[tokio::test]
async fn a_failing_formatter_for_a_missing_tool_keeps_the_default_message() {
    let _guard = fixture().await;
    set_dont_log_tool_data(true);
    let run = RealtimeRunConfig {
        async_tool_calls: false,
        tool_error_formatter: Some(async_tool_error_formatter_fn(|_args| {
            Box::pin(async { Err(AgentsError::other("SECRET_REALTIME_TOOL_FORMATTER")) })
        })),
        ..Default::default()
    };
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(run))
        .await
        .expect("session connects");
    model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("missing_tool", "call_missing", "{}"),
    ));
    model.emit(RealtimeModelEvent::Other {
        data: Value::String("barrier".into()),
    });
    loop {
        if matches!(
            recv(&session).await,
            RealtimeSessionEvent::RawModelEvent {
                data: RealtimeModelEvent::Other { .. },
                ..
            }
        ) {
            break;
        }
    }

    assert_eq!(sent_outputs(&model), ["Tool missing_tool not found"]);
    let logs = logs();
    assert!(
        logs.contains("ERROR Tool error formatter failed for missing tool"),
        "{logs}"
    );
    assert!(!logs.contains("SECRET_REALTIME_TOOL_FORMATTER"), "{logs}");
    reset_dont_log_tool_data();
}
