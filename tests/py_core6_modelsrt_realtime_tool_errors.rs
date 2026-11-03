//! Port of `test_async_tool_failure_events_are_safe` (`tests/realtime/test_session_tool_error_events.py`)
//! across the two failure points (tool task, output send) and the four model/tool redaction
//! policy combinations: the error event carries only the fixed message, and the log line keeps
//! the error text only when `dont_log_tool_data` is unset.
//!
//! The ninth Python case replaces the exception with one whose `__str__` raises; Rust errors
//! are values rendered through `Display`, so that case has no counterpart.
#![cfg(feature = "realtime")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::realtime::{
    RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent, RealtimeRunConfig,
    RealtimeRunner, RealtimeSession, RealtimeSessionEvent, RealtimeStep, ScriptedRealtimeModel,
};
use openai_agents_sdk::tool::FunctionTool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Notify;

const TIMEOUT: Duration = Duration::from_secs(5);
const ARGUMENT: &str = "synthetic-argument-secret";
const OUTPUT: &str = "synthetic-output-secret";
const SECRET: &str = "synthetic-exception-secret";
const CALL_ID: &str = "synthetic-call-secret";

#[derive(Deserialize, JsonSchema)]
struct LookupArgs {
    value: String,
}

struct LogSink(Mutex<Vec<(log::Level, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// Restores the redaction flags to the environment defaults when a test ends.
struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool, tool_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        set_dont_log_tool_data(tool_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

fn lookup_call() -> RealtimeModelToolCallEvent {
    RealtimeModelToolCallEvent {
        name: "lookup".to_string(),
        call_id: CALL_ID.to_string(),
        arguments: format!(r#"{{"value":"{ARGUMENT}"}}"#),
        id: None,
        previous_item_id: None,
    }
}

fn lookup_tool(
    started: Arc<Notify>,
    release: Arc<Notify>,
    invocations: Arc<AtomicUsize>,
    fail: bool,
) -> FunctionTool<()> {
    FunctionTool::new(
        "lookup",
        "look something up",
        move |_ctx, args: LookupArgs| {
            let started = started.clone();
            let release = release.clone();
            let invocations = invocations.clone();
            async move {
                invocations.fetch_add(1, Ordering::SeqCst);
                assert_eq!(args.value, ARGUMENT);
                started.notify_one();
                release.notified().await;
                if fail {
                    Err(AgentsError::other(SECRET))
                } else {
                    Ok(OUTPUT.to_string())
                }
            }
        },
    )
    .without_failure_error_function()
}

#[tokio::test]
async fn async_tool_failure_events_are_safe() {
    static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _turn = SERIAL.lock().await;
    let sink = sink();

    for send_failure in [false, true] {
        for (redact_model, redact_tool) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let _policy = PolicyGuard::set(redact_model, redact_tool);
            sink.0.lock().unwrap().clear();
            let case =
                format!("send_failure={send_failure} model={redact_model} tool={redact_tool}");

            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let invocations = Arc::new(AtomicUsize::new(0));
            let steps = if send_failure {
                vec![
                    RealtimeStep::new("tool_output").failing(SECRET),
                    RealtimeStep::new("tool_output"),
                ]
            } else {
                Vec::new()
            };
            let model = Arc::new(ScriptedRealtimeModel::new(steps));
            let agent = RealtimeAgent::<()>::new("test").with_tool(lookup_tool(
                started.clone(),
                release.clone(),
                invocations.clone(),
                !send_failure,
            ));
            let session: RealtimeSession<()> = RealtimeRunner::new(agent.into_arc())
                .with_model(model.clone())
                .with_config(RealtimeRunConfig::default())
                .run(())
                .await
                .expect("session connects");

            assert!(
                model.emit(RealtimeModelEvent::FunctionCall(lookup_call())),
                "{case}"
            );
            tokio::time::timeout(TIMEOUT, started.notified())
                .await
                .expect("the tool starts");
            release.notify_one();

            let mut errors: Vec<Value> = Vec::new();
            let mut events = Vec::new();
            tokio::time::timeout(TIMEOUT, async {
                loop {
                    let Some(event) = session.recv().await else {
                        break;
                    };
                    let is_error = matches!(event, RealtimeSessionEvent::Error { .. });
                    if let RealtimeSessionEvent::Error { error, .. } = &event {
                        errors.push(error.clone());
                    }
                    events.push(event);
                    if is_error && send_failure {
                        break;
                    }
                }
            })
            .await
            .expect("an error event arrives");

            let expected = if send_failure {
                "Tool output send failed; cached output will be retried"
            } else {
                "Tool call task failed"
            };
            assert_eq!(
                errors,
                vec![serde_json::json!({"message": expected})],
                "{case}"
            );
            for value in [SECRET, ARGUMENT, OUTPUT, CALL_ID] {
                assert!(!errors[0].to_string().contains(value), "{case}");
            }

            let prefix = if send_failure {
                "Realtime tool output send failed"
            } else {
                "Realtime tool call task failed"
            };
            let records: Vec<String> = sink
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, message)| message.starts_with(prefix))
                .map(|(_, message)| message.clone())
                .collect();
            assert_eq!(records.len(), 1, "{case}: {records:?}");
            if redact_tool {
                for value in [SECRET, ARGUMENT, OUTPUT, CALL_ID] {
                    assert!(!records[0].contains(value), "{case}: {}", records[0]);
                }
            } else {
                assert!(records[0].contains(SECRET), "{case}: {}", records[0]);
            }

            if send_failure {
                // The retried call is answered from the cached output without running the tool.
                assert!(
                    model.emit(RealtimeModelEvent::FunctionCall(lookup_call())),
                    "{case}"
                );
                let mut tool_end_output: Option<String> = None;
                tokio::time::timeout(TIMEOUT, async {
                    while let Some(event) = session.recv().await {
                        if let RealtimeSessionEvent::ToolEnd { output, .. } = &event {
                            tool_end_output = Some(output.to_model_string());
                            break;
                        }
                    }
                })
                .await
                .expect("the cached output is delivered");
                assert_eq!(tool_end_output.as_deref(), Some(OUTPUT), "{case}");
                assert_eq!(invocations.load(Ordering::SeqCst), 1, "{case}");
                assert!(session.take_error().is_none(), "{case}");
            } else {
                assert!(
                    session.take_error().unwrap().to_string().contains(SECRET),
                    "{case}"
                );
            }
        }
    }
}
