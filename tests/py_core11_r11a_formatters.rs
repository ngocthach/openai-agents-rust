//! The run-level `tool_error_formatter` may await and may fail (Python's `ToolErrorFormatter` is
//! `MaybeAwaitable`, and every call site catches what it raises), and a handoff input filter
//! may fail the run. Ports of:
//!
//! - `test_tool_exception_formatter.py`: the async formatter, `response="raise"` of
//!   `test_formatter_fallback_does_not_expose_exception_or_bad_return`, and
//!   `test_formatter_failure_respects_local_diagnostic_opt_in`;
//! - `test_error_logging_redaction.py`: `test_approval_rejection_formatter_error_*`;
//! - `test_handoff_failure_error_function.py`: `failure="filter"` of
//!   `test_failures_outside_recovery_boundary_propagate`.
//!
//! The logging policy flags, the `log` backend and the trace processors are process-global, so
//! every test holds one lock and reads only the records produced while it holds it.

use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::handoffs::{HandoffInputData, handoff};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolErrorKind};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{
    Agent, AgentsError, Result, RunState, Runner, async_handoff_input_filter,
    async_tool_error_formatter_fn, try_handoff_input_filter, try_tool_error_formatter_fn,
};
use serde_json::{Value, json};

const SECRET: &str = "synthetic-private-exception-detail";
const GENERIC: &str = "An error occurred while running the tool. Please try again.";
const APPROVED: &str = "Order IDs must start with ORD-.";
const NOT_APPROVED: &str = "Tool execution was not approved.";

/// `(kind, tool type, tool name, call id, error text)` for each formatter call.
type Received = Arc<Mutex<Vec<(ToolErrorKind, String, String, String, String)>>>;

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!(
            "{} {}: {}",
            record.level(),
            record.target(),
            record.args()
        ));
    }
    fn flush(&self) {}
}

fn processor() -> &'static Arc<InMemoryTraceProcessor> {
    static PROCESSOR: OnceLock<Arc<InMemoryTraceProcessor>> = OnceLock::new();
    PROCESSOR.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
        let p = Arc::new(InMemoryTraceProcessor::new());
        set_trace_processors(vec![p.clone() as Arc<dyn TracingProcessor>]);
        p
    })
}

/// Holds the global lock; restores the env-default logging policy when dropped.
struct Fixture {
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

async fn fixture() -> Fixture {
    let guard = LOCK.lock().await;
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    processor().clear();
    RECORDS.lock().unwrap().clear();
    // Python's defaults: both data classes are redacted from logs.
    set_dont_log_model_data(true);
    set_dont_log_tool_data(true);
    Fixture { _guard: guard }
}

fn logs() -> String {
    RECORDS.lock().unwrap().join("\n")
}

fn spans() -> String {
    serde_json::to_string(&processor().items()).unwrap()
}

fn call(name: &str, args: &str, id: &str) -> OutputItem {
    function_call_full(name, args, id, None, None)
}

fn model(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

/// One call of `name` with call id `call-5324`, then `done`.
fn make_model(name: &str) -> Arc<ScriptedModel> {
    model(vec![
        vec![call(name, "{}", "call-5324")],
        vec![assistant_message("done")],
    ])
}

fn last_input(model: &ScriptedModel) -> Vec<Value> {
    model
        .last_call()
        .expect("a model call")
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect()
}

fn model_output(model: &ScriptedModel) -> Value {
    last_input(model)
        .into_iter()
        .find(|i| i["type"] == "function_call_output")
        .expect("a function_call_output")["output"]
        .clone()
}

async fn run(
    agent: Agent<()>,
    input: &str,
    streamed: bool,
    options: RunOptions<()>,
) -> Result<RunResult<()>> {
    if streamed {
        let mut s = Runner::run_streamed_with_options(agent, input, options);
        {
            let events = s.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(s.into_result().expect("a finished stream has a result"))
    } else {
        Runner::run_with_options(agent, input, options).await
    }
}

fn with_config(config: RunConfig<()>) -> RunOptions<()> {
    RunOptions::default().run_config(config)
}

/// The secret reaches neither the model, the history, the traces nor the logs.
fn assert_redacted(secret: &str, result: &RunResult<()>, model: &ScriptedModel) {
    assert!(
        !serde_json::to_string(&last_input(model))
            .unwrap()
            .contains(secret)
    );
    let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
    assert!(!serde_json::to_string(&history).unwrap().contains(secret));
    let spans = spans();
    assert_ne!(spans, "[]", "spans were recorded");
    assert!(!spans.contains(secret), "secret leaked to spans: {spans}");
    let logs = logs();
    assert!(!logs.contains(secret), "secret leaked to logs: {logs}");
}

fn failing_tool(name: &str, message: String) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "fails", move |_ctx| {
        let message = message.clone();
        async move { Err::<String, _>(AgentsError::other(message)) }
    })
}

/// Python: `test_run_formatter_returns_only_approved_feedback[async_handler=True]` (streamed x
/// not): the formatter awaits before answering, and still sees the original error.
#[tokio::test]
async fn an_async_run_formatter_returns_only_approved_feedback() {
    for streamed in [false, true] {
        let _f = fixture().await;
        let received: Received = Arc::default();
        let seen = received.clone();
        let formatter = async_tool_error_formatter_fn(move |args| {
            let seen = seen.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                seen.lock().unwrap().push((
                    args.kind,
                    args.tool_type.clone(),
                    args.tool_name.clone(),
                    args.call_id.clone(),
                    args.error.map(|e| e.to_string()).unwrap_or_default(),
                ));
                Ok(Some(APPROVED.to_string()))
            })
        });
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter_handler(formatter);
        let model = make_model("fail");
        let agent = Agent::<()>::new("test")
            .with_tool(failing_tool("fail", SECRET.to_string()))
            .with_model_instance(model.clone());
        let result = run(agent, "start", streamed, with_config(config))
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(model_output(&model), json!(APPROVED));
        let received = received.lock().unwrap().clone();
        assert_eq!(received.len(), 1, "streamed={streamed}");
        let (kind, tool_type, tool_name, call_id, error) = &received[0];
        assert_eq!(*kind, ToolErrorKind::ToolException);
        assert_eq!(
            (tool_type.as_str(), tool_name.as_str()),
            ("function", "fail")
        );
        assert_eq!(call_id, "call-5324");
        assert!(
            error.contains(SECRET),
            "the formatter sees the original error"
        );
        assert_redacted(SECRET, &result, &model);
    }
}

/// Python: `test_formatter_fallback_does_not_expose_exception_or_bad_return[raise]`: a formatter
/// that fails (here after awaiting, as the Python one does) leaves the default message in place
/// and the exception out of the model input, the history, the spans and the logs. Both the sync
/// fallible constructor and the async one.
#[tokio::test]
async fn a_failing_run_formatter_falls_back_to_the_default_message() {
    for streamed in [false, true] {
        for is_async in [false, true] {
            let _f = fixture().await;
            let formatter = if is_async {
                async_tool_error_formatter_fn(|_args| {
                    Box::pin(async {
                        tokio::task::yield_now().await;
                        Err(AgentsError::other(SECRET))
                    })
                })
            } else {
                try_tool_error_formatter_fn(|_args| Err(AgentsError::other(SECRET)))
            };
            let config = RunConfig::<()>::default()
                .with_trace_include_sensitive_data(true)
                .with_tool_error_formatter_handler(formatter);
            let model = make_model("fail");
            let agent = Agent::<()>::new("test")
                .with_tool(failing_tool("fail", SECRET.to_string()))
                .with_model_instance(model.clone());
            let result = run(agent, "start", streamed, with_config(config))
                .await
                .expect("a formatter failure does not fail the run");
            let label = format!("streamed={streamed} async={is_async}");
            assert_eq!(
                result.final_output_text().as_deref(),
                Some("done"),
                "{label}"
            );
            assert_eq!(model_output(&model), json!(GENERIC), "{label}");
            assert_redacted(SECRET, &result, &model);
            assert!(logs().contains("Tool error formatter failed"), "{label}");
        }
    }
}

/// Python: `test_formatter_failure_respects_local_diagnostic_opt_in[tool_redacted]`: the
/// formatter's own error text reaches the log only when tool-data logging is switched on, and
/// never the model, the history or the spans.
#[tokio::test]
async fn a_failing_run_formatter_logs_its_error_only_when_tool_data_logging_is_enabled() {
    for tool_redacted in [false, true] {
        let _f = fixture().await;
        set_dont_log_tool_data(tool_redacted);
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter_handler(try_tool_error_formatter_fn(|args| {
                // Python re-raises with the original error as the cause.
                Err(AgentsError::other(format!(
                    "formatter-{SECRET}: {}",
                    args.error.map(|e| e.to_string()).unwrap_or_default()
                )))
            }));
        let model = make_model("fail");
        let agent = Agent::<()>::new("test")
            .with_tool(failing_tool("fail", SECRET.to_string()))
            .with_model_instance(model.clone());
        let result = run(agent, "start", false, with_config(config))
            .await
            .unwrap();
        assert_eq!(
            model_output(&model),
            json!(GENERIC),
            "redacted={tool_redacted}"
        );
        let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
        assert!(!serde_json::to_string(&history).unwrap().contains(SECRET));
        assert!(!spans().contains(SECRET));
        if tool_redacted {
            assert_redacted(SECRET, &result, &model);
        } else {
            assert!(logs().contains(SECRET), "logs: {}", logs());
        }
    }
}

fn guarded_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "needs approval", |_ctx| async {
        Ok("ran".to_string())
    })
    .with_needs_approval(true)
}

/// Pause on `tool`, reject it, and resume under `config`; returns the model's last input.
async fn reject_and_resume(tool: &str, config: RunConfig<()>) -> (RunResult<()>, Vec<Value>) {
    let model = make_model(tool);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_tool(guarded_tool(tool))
            .with_model_instance(model.clone()),
    );
    let paused = Runner::run_with_options(agent.clone(), "start", with_config(config.clone()))
        .await
        .unwrap();
    assert_eq!(paused.interruptions.len(), 1);
    let mut state: RunState = paused.to_state();
    state.reject(&paused.interruptions[0], false, None);
    let result = Runner::run_with_options(agent, state, with_config(config))
        .await
        .expect("a formatter failure does not fail the run");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    (result, last_input(&model))
}

fn rejection_output(input: &[Value]) -> Value {
    input
        .iter()
        .find(|i| i["type"] == "function_call_output")
        .expect("a function_call_output")["output"]
        .clone()
}

/// Python: `test_approval_rejection_formatter_error_redacts_exception`: with tool-data logging
/// off, a failing formatter keeps the default rejection message, and the log names neither the
/// error nor the tool.
#[tokio::test]
async fn a_failing_formatter_on_a_rejection_logs_only_the_sentence_when_redacted() {
    let _f = fixture().await;
    set_dont_log_tool_data(true);
    let tool_name = "SECRET_FORMATTER_TOOL_NAME";
    let config =
        RunConfig::<()>::default().with_tool_error_formatter_handler(try_tool_error_formatter_fn(
            |_args| Err(AgentsError::other("formatter blew up SECRET_FMT_123")),
        ));
    let (_result, input) = reject_and_resume(tool_name, config).await;

    assert_eq!(rejection_output(&input), json!(NOT_APPROVED));
    let logs = logs();
    assert!(logs.contains("Tool error formatter failed"), "{logs}");
    assert!(!logs.contains("SECRET_FMT_123"), "{logs}");
    assert!(!logs.contains(tool_name), "{logs}");
}

/// Python: `test_approval_rejection_formatter_error_logs_full_when_enabled`.
#[tokio::test]
async fn a_failing_formatter_on_a_rejection_logs_the_error_when_enabled() {
    let _f = fixture().await;
    set_dont_log_tool_data(false);
    let config =
        RunConfig::<()>::default().with_tool_error_formatter_handler(try_tool_error_formatter_fn(
            |_args| Err(AgentsError::other("formatter blew up SECRET_FMT_123")),
        ));
    let (_result, input) = reject_and_resume("diagnostic_tool", config).await;

    assert_eq!(rejection_output(&input), json!(NOT_APPROVED));
    assert!(logs().contains("SECRET_FMT_123"), "{}", logs());
}

/// A formatter that fails for a call to a tool the agent does not have keeps the default
/// message, logged with its own sentence (Python: `_resolve_tool_not_found_message`).
#[tokio::test]
async fn a_failing_formatter_for_a_missing_tool_keeps_the_default_message() {
    let _f = fixture().await;
    set_dont_log_tool_data(false);
    let config = RunConfig::<()>::default()
        .with_tool_not_found_behavior(openai_agents_sdk::ToolNotFoundBehavior::ReturnErrorToModel)
        .with_tool_error_formatter_handler(async_tool_error_formatter_fn(|_args| {
            Box::pin(async { Err(AgentsError::other("formatter blew up SECRET_FMT_123")) })
        }));
    let model = make_model("missing_tool");
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let result = run(agent, "start", false, with_config(config))
        .await
        .expect("a formatter failure does not fail the run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        model_output(&model),
        json!("Tool 'missing_tool' not found.")
    );
    let logs = logs();
    assert!(
        logs.contains("Tool error formatter failed for missing tool"),
        "{logs}"
    );
    assert!(logs.contains("SECRET_FMT_123"), "{logs}");
}

/// A formatter that answers `None` keeps the default; `Some("")` is a message (the empty string
/// is valid feedback), which the async constructor delivers as well.
#[tokio::test]
async fn an_async_formatter_may_decline_or_answer_with_an_empty_message() {
    for (response, expected) in [(None, GENERIC), (Some(""), "")] {
        let _f = fixture().await;
        let config = RunConfig::<()>::default().with_tool_error_formatter_handler(
            async_tool_error_formatter_fn(move |_args| {
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    Ok(response.map(str::to_string))
                })
            }),
        );
        let model = make_model("fail");
        let agent = Agent::<()>::new("test")
            .with_tool(failing_tool("fail", SECRET.to_string()))
            .with_model_instance(model.clone());
        run(agent, "start", false, with_config(config))
            .await
            .unwrap();
        assert_eq!(model_output(&model), json!(expected));
    }
}

fn target_agent() -> Arc<Agent<()>> {
    Arc::new(Agent::<()>::new("target").with_model_instance(model(vec![])))
}

/// Python: `test_failures_outside_recovery_boundary_propagate[filter]`: an input filter that
/// raises ends the run with its own error, after the handoff succeeded, so the handoff's
/// failure formatter is never consulted. Fallible and async filters, direct and streamed.
#[tokio::test]
async fn a_failing_handoff_input_filter_ends_the_run_with_its_error() {
    for streamed in [false, true] {
        for is_async in [false, true] {
            let _f = fixture().await;
            let formatter_calls = Arc::new(Mutex::new(0usize));
            let seen = formatter_calls.clone();
            let filter = if is_async {
                async_handoff_input_filter(|_data: HandoffInputData| async {
                    tokio::task::yield_now().await;
                    Err(AgentsError::other("filter failed"))
                })
            } else {
                try_handoff_input_filter(|_data| Err(AgentsError::other("filter failed")))
            };
            let transfer = handoff(target_agent())
                .with_input_filter_handler(filter)
                .with_failure_error_function(move |_ctx, _error| {
                    *seen.lock().unwrap() += 1;
                    "handled".into()
                });
            let caller_model = model(vec![vec![call(&transfer.tool_name, "{}", "call")]]);
            let caller = Agent::<()>::new("caller")
                .with_model_instance(caller_model)
                .with_handoff(transfer);
            let label = format!("streamed={streamed} async={is_async}");
            let error = run(caller, "go", streamed, RunOptions::default())
                .await
                .expect_err(&label);
            assert!(
                error.to_string().contains("filter failed"),
                "{label}: {error}"
            );
            assert_eq!(*formatter_calls.lock().unwrap(), 0, "{label}");
        }
    }
}
