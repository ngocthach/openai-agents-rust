#![cfg(feature = "realtime")]
//! Port of the behavioral tests of `test_error_logging_redaction.py`: error-path logging and
//! error messages must not leak model/tool payloads when data logging is disabled.
//!
//! Python also locks in traceback frame locals, `__cause__`/`__context__` chains, exception
//! groups and hostile `__str__`/`__bool__` objects; Rust errors carry none of these, so those
//! assertions reduce to the error's message and `run_data` here (see the parity report).
//!
//! The logging policy flags and the `log` backend are process-global, so every test holds one
//! lock and resets the flags it touches.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::logger;
use openai_agents_sdk::realtime::{RealtimeAgent, realtime_handoff};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, InputGuardrail, OutputGuardrail, Result, RunContext,
    Runner, Session, ToolUseBehavior,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const SECRET: &str = "super secret prompt content";
const TOOL_ARGUMENT_SECRET: &str = "SECRET_TOOL_ARGUMENT_123";
const TOOL_OUTPUT_SECRET: &str = "SECRET_TOOL_OUTPUT_123";
const MODEL_OUTPUT_SECRET: &str = "SECRET_MODEL_OUTPUT_123";
const SENSITIVE_SCHEMA_SECRET: &str = "SENSITIVE_HANDOFF_SCHEMA_SECRET_4207";
const SENSITIVE_OUTPUT_SCHEMA_SECRET: &str = "SENSITIVE_OUTPUT_SCHEMA_SECRET_4207";
const REDACTED: &str = "Error details are redacted.";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());
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
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

fn install() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
        openai_agents_sdk::tracing::set_tracing_disabled(true);
    });
}

struct Flags;

impl Drop for Flags {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

/// Take the lock, clear the captured records and set both policies (Python's `monkeypatch`).
fn flags(model: bool, tool: bool) -> Flags {
    install();
    RECORDS.lock().unwrap().clear();
    set_dont_log_model_data(model);
    set_dont_log_tool_data(tool);
    Flags
}

fn records() -> Vec<(log::Level, String)> {
    RECORDS.lock().unwrap().clone()
}

fn logs() -> String {
    records()
        .into_iter()
        .map(|(_, m)| m)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Python's `_HostileException` / `_HostileValue`: formatting it is a test failure.
struct Hostile;

impl fmt::Display for Hostile {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("redacted logging inspected Display")
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RequiredOutput {
    answer: String,
    count: i64,
}

/// Python's `Literal["SENSITIVE_..._SECRET_4207"]`.
#[derive(Debug, Deserialize, JsonSchema)]
enum SensitiveOutput {
    #[serde(rename = "SENSITIVE_OUTPUT_SCHEMA_SECRET_4207")]
    #[allow(dead_code)]
    Only,
}

#[derive(Debug, Deserialize, JsonSchema)]
enum SensitiveHandoffInput {
    #[serde(rename = "SENSITIVE_HANDOFF_SCHEMA_SECRET_4207")]
    #[allow(dead_code)]
    Only,
}

fn secret_payload() -> String {
    format!(r#"{{"answer": "{MODEL_OUTPUT_SECRET}"}}"#)
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

/// A session whose `add_items` fails once `armed` is set (Python's `FailingFinalTurnSession`).
#[derive(Debug)]
struct FailingSession {
    inner: InMemorySession,
    armed: Arc<AtomicBool>,
    message: String,
    only_tool_outputs: bool,
}

impl FailingSession {
    fn new(armed: Arc<AtomicBool>, message: impl Into<String>) -> Self {
        FailingSession {
            inner: InMemorySession::new("failing"),
            armed,
            message: message.into(),
            only_tool_outputs: false,
        }
    }
}

#[async_trait::async_trait]
impl Session for FailingSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let fail = if self.only_tool_outputs {
            items
                .iter()
                .any(|i| i.as_value()["type"] == "function_call_output")
        } else {
            self.armed.load(Ordering::SeqCst)
        };
        if fail {
            return Err(AgentsError::other(self.message.clone()));
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

/// An output guardrail that arms `armed` and then fails validating the secret payload, as
/// Python's `output_guardrail` helpers do.
fn validating_output_guardrail(armed: Arc<AtomicBool>) -> OutputGuardrail<()> {
    OutputGuardrail::new("validate", move |_ctx, _agent, _output| {
        let armed = armed.clone();
        async move {
            armed.store(true, Ordering::SeqCst);
            AgentOutputSchema::new::<RequiredOutput>()?.validate_json(&secret_payload())?;
            unreachable!("validation should fail")
        }
    })
}

async fn run_maybe_streamed(
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

// ===========================================================================
// Shared logging helpers
// ===========================================================================

#[tokio::test]
async fn log_tool_action_error_redacts_by_default() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    logger::log_tool_action_error("Shell executor failed", &"rm -rf /secret/path");
    assert_eq!(
        records(),
        vec![(log::Level::Error, "Shell executor failed".to_string())]
    );
}

#[tokio::test]
async fn shared_error_helpers_do_not_inspect_redacted_exceptions() {
    type Helper = fn(&str, &dyn fmt::Display);
    let cases: [(Helper, bool, bool, log::Level); 10] = [
        (
            logger::log_model_action_error,
            true,
            false,
            log::Level::Error,
        ),
        (
            logger::log_model_action_debug,
            true,
            false,
            log::Level::Debug,
        ),
        (
            logger::log_model_action_warning,
            true,
            false,
            log::Level::Warn,
        ),
        (
            logger::log_tool_action_debug,
            false,
            true,
            log::Level::Debug,
        ),
        (
            logger::log_tool_action_error,
            false,
            true,
            log::Level::Error,
        ),
        (
            logger::log_tool_action_warning,
            false,
            true,
            log::Level::Warn,
        ),
        (
            logger::log_model_and_tool_action_error,
            true,
            false,
            log::Level::Error,
        ),
        (
            logger::log_model_and_tool_action_error,
            false,
            true,
            log::Level::Error,
        ),
        (
            logger::log_model_and_tool_action_debug,
            true,
            false,
            log::Level::Debug,
        ),
        (
            logger::log_model_and_tool_action_warning,
            false,
            true,
            log::Level::Warn,
        ),
    ];
    let _lock = LOCK.lock().await;
    for (helper, model_flag, tool_flag, level) in cases {
        let _f = flags(model_flag, tool_flag);
        helper("Fixed operational message", &Hostile);
        assert_eq!(
            records(),
            vec![(level, "Fixed operational message".to_string())]
        );
    }
}

#[tokio::test]
async fn shared_error_helper_preserves_diagnostics_when_enabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    logger::log_tool_action_error("Tool failed", &SECRET);
    let records = records();
    assert_eq!(records.len(), 1);
    assert!(records[0].1.contains(SECRET), "{records:?}");
}

#[tokio::test]
async fn shared_data_warning_does_not_inspect_redacted_arguments() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    logger::log_model_and_tool_data_warning("Fixed operational warning", || Hostile.to_string());
    assert_eq!(
        records(),
        vec![(log::Level::Warn, "Fixed operational warning".to_string())]
    );
}

#[tokio::test]
async fn shared_data_warning_preserves_diagnostics_when_enabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(false, false);
    logger::log_model_and_tool_data_warning("Fixed operational warning", || {
        format!("Warning for {}", "diagnostic-agent")
    });
    assert_eq!(
        records(),
        vec![(log::Level::Warn, "Warning for diagnostic-agent".to_string())]
    );
}

#[tokio::test]
async fn log_tool_action_error_logs_full_when_tool_data_enabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    logger::log_tool_action_error("Shell executor failed", &"rm -rf /secret/path");
    let records = records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0, log::Level::Error);
    assert!(records[0].1.contains("/secret/path"), "{records:?}");
}

// ===========================================================================
// Tool argument / output validation
// ===========================================================================

#[derive(Deserialize, JsonSchema)]
struct IntegerArgument {
    value: i64,
}

fn integer_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "requires_integer_argument",
        "Requires an integer.",
        |_ctx, args: IntegerArgument| async move { Ok(args.value.to_string()) },
    )
    .without_failure_error_function()
}

async fn invoke(tool: &FunctionTool<()>, payload: &str) -> AgentsError {
    let ctx = ToolContext::direct(RunContext::new(()), "1", &tool.name, payload);
    tool.invoke(ctx, payload.to_string()).await.unwrap_err()
}

#[tokio::test]
async fn function_tool_validation_error_redacts_payload_when_tool_data_disabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let tool = integer_tool();
    let payload = format!(r#"{{"value": "{TOOL_ARGUMENT_SECRET}"}}"#);
    let error = invoke(&tool, &payload).await;
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "Invalid JSON input for tool requires_integer_argument"
    );
}

#[tokio::test]
async fn function_tool_validation_error_preserves_diagnostics_when_tool_data_enabled() {
    let _lock = LOCK.lock().await;
    // Python leaves `DONT_LOG_MODEL_DATA` at its default; argument validation only follows
    // the tool data policy.
    let _f = flags(true, false);
    let tool = integer_tool();
    let payload = format!(r#"{{"value": "{TOOL_ARGUMENT_SECRET}"}}"#);
    let error = invoke(&tool, &payload).await;
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains(TOOL_ARGUMENT_SECRET), "{error}");
}

#[derive(serde::Serialize, Deserialize, JsonSchema)]
struct IntegerOutput {
    value: i64,
}

fn wrong_typed_output_tool() -> FunctionTool<()> {
    FunctionTool::new_no_args("output_tool", "Returns the wrong type.", |_ctx| async {
        Ok(json!({"value": TOOL_OUTPUT_SECRET}))
    })
    .with_output_type::<IntegerOutput>()
    .unwrap()
    .without_failure_error_function()
}

#[tokio::test]
async fn function_tool_output_validation_error_redacts_payload_when_tool_data_disabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let error = invoke(&wrong_typed_output_tool(), "{}").await;
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(!error.to_string().contains(TOOL_OUTPUT_SECRET), "{error}");
}

#[tokio::test]
async fn function_tool_output_validation_error_preserves_diagnostics_when_enabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    let error = invoke(&wrong_typed_output_tool(), "{}").await;
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert!(error.to_string().contains(TOOL_OUTPUT_SECRET), "{error}");
}

// ===========================================================================
// Output schema and handoff input validation
// ===========================================================================

#[tokio::test]
async fn output_schema_validation_error_follows_model_data_policy() {
    let _lock = LOCK.lock().await;
    for (model_redacted, tool_redacted, expected_redacted) in
        [(true, false, true), (false, true, false)]
    {
        let _f = flags(model_redacted, tool_redacted);
        let error = AgentOutputSchema::new::<RequiredOutput>()
            .unwrap()
            .validate_json(&secret_payload())
            .unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert_eq!(
            !error.to_string().contains(MODEL_OUTPUT_SECRET),
            expected_redacted,
            "model={model_redacted} tool={tool_redacted}: {error}"
        );
    }
}

#[tokio::test]
async fn output_schema_redaction_omits_sensitive_schema_metadata() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let error = AgentOutputSchema::new::<SensitiveOutput>()
        .unwrap()
        .validate_json(r#""invalid""#)
        .unwrap_err();
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert!(
        !error.to_string().contains(SENSITIVE_OUTPUT_SCHEMA_SECRET),
        "{error}"
    );
}

const MIXED_POLICY: [(bool, bool, bool); 4] = [
    (true, false, true),
    (false, true, true),
    (true, true, true),
    (false, false, false),
];

#[tokio::test]
async fn handoff_input_validation_error_follows_mixed_data_policy() {
    let _lock = LOCK.lock().await;
    for (model_redacted, tool_redacted, expected_redacted) in MIXED_POLICY {
        let _f = flags(model_redacted, tool_redacted);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let handoff = Handoff::new(Arc::new(Agent::<()>::new("target"))).with_on_handoff_input(
            move |_ctx, _input: RequiredOutput| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            },
        );
        let error = handoff
            .invoke(RunContext::new(()), secret_payload())
            .await
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert_eq!(
            !error.to_string().contains(MODEL_OUTPUT_SECRET),
            expected_redacted,
            "model={model_redacted} tool={tool_redacted}: {error}"
        );
    }
}

#[tokio::test]
async fn handoff_redaction_omits_sensitive_schema_metadata() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let handoff = Handoff::new(Arc::new(Agent::<()>::new("target"))).with_on_handoff_input(
        move |_ctx, _input: SensitiveHandoffInput| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        },
    );
    let error = handoff
        .invoke(RunContext::new(()), r#""invalid""#.into())
        .await
        .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        !error.to_string().contains(SENSITIVE_SCHEMA_SECRET),
        "{error}"
    );
}

#[tokio::test]
async fn realtime_handoff_redaction_omits_sensitive_schema_metadata() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let handoff = realtime_handoff(Arc::new(RealtimeAgent::<()>::new("target")))
        .with_on_handoff_input(move |_ctx, _input: SensitiveHandoffInput| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });
    let error = handoff
        .invoke(RunContext::new(()), r#""invalid""#.into())
        .await
        .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        !error.to_string().contains(SENSITIVE_SCHEMA_SECRET),
        "{error}"
    );
}

#[tokio::test]
async fn realtime_handoff_input_validation_error_follows_mixed_data_policy() {
    let _lock = LOCK.lock().await;
    for (model_redacted, tool_redacted, expected_redacted) in MIXED_POLICY {
        let _f = flags(model_redacted, tool_redacted);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let handoff = realtime_handoff(Arc::new(RealtimeAgent::<()>::new("target")))
            .with_on_handoff_input(move |_ctx, _input: RequiredOutput| {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            });
        let error = handoff
            .invoke(RunContext::new(()), secret_payload())
            .await
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert_eq!(
            !error.to_string().contains(MODEL_OUTPUT_SECRET),
            expected_redacted,
            "model={model_redacted} tool={tool_redacted}: {error}"
        );
    }
}

// ===========================================================================
// Errors surfaced by runs
// ===========================================================================

fn output_type_agent(model: Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new("A")
        .with_model_instance(model)
        .with_output_type::<RequiredOutput>()
}

async fn secret_session(id: &str) -> Arc<dyn Session> {
    let session = InMemorySession::new(id);
    session
        .add_items(vec![InputItem(
            json!({"role": "user", "content": MODEL_OUTPUT_SECRET}),
        )])
        .await
        .unwrap();
    Arc::new(session)
}

fn assert_redacted_model_behavior(error: &AgentsError, secrets: &[&str]) {
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert!(
        error.run_data().is_none(),
        "redacted errors carry no run data"
    );
    for secret in secrets {
        assert!(!error.to_string().contains(secret), "{error}");
        assert!(!format!("{error:?}").contains(secret), "{error:?}");
    }
}

#[tokio::test]
async fn run_surfaces_redacted_output_validation_error() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    for streamed in [false, true] {
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let options = RunOptions::default().session(secret_session("redacted-run").await);
        let error = run_maybe_streamed(
            output_type_agent(model),
            MODEL_OUTPUT_SECRET,
            streamed,
            options,
        )
        .await
        .unwrap_err();
        assert_redacted_model_behavior(&error, &[MODEL_OUTPUT_SECRET]);
    }
}

#[test]
fn run_sync_surfaces_redacted_output_validation_error_without_runner_data() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _lock = rt.block_on(LOCK.lock());
    let _f = flags(true, false);
    let session = rt.block_on(secret_session("redacted-run-sync"));
    let model = scripted(vec![vec![assistant_message(secret_payload())]]);
    drop(rt);
    let error = Runner::run_sync_with_options(
        output_type_agent(model),
        MODEL_OUTPUT_SECRET,
        RunOptions::default().session(session),
    )
    .unwrap_err();
    assert_redacted_model_behavior(&error, &[MODEL_OUTPUT_SECRET]);
}

#[tokio::test]
async fn streamed_task_observers_follow_model_data_policy() {
    let _lock = LOCK.lock().await;
    for redacted in [false, true] {
        let _f = flags(redacted, false);
        let input_secret = "STREAM_TASK_INPUT_CANARY";
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let options = RunOptions::default().session(secret_session("stream-task").await);
        let error = run_maybe_streamed(output_type_agent(model), input_secret, true, options)
            .await
            .unwrap_err();
        if redacted {
            assert_redacted_model_behavior(&error, &[MODEL_OUTPUT_SECRET, input_secret]);
        } else {
            assert!(error.to_string().contains(MODEL_OUTPUT_SECRET), "{error}");
            let run_data = error.run_data().expect("diagnostic run data");
            assert!(format!("{:?}", run_data.input).contains(input_secret));
        }
    }
}

#[tokio::test]
async fn streamed_run_loop_exception_follows_model_data_policy() {
    let _lock = LOCK.lock().await;
    for redacted in [false, true] {
        let _f = flags(redacted, false);
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let mut s = Runner::run_streamed_with_options(
            output_type_agent(model),
            "go",
            RunOptions::default(),
        );
        {
            let events = s.stream_events();
            futures::pin_mut!(events);
            while events.next().await.is_some() {}
        }
        let error = s.run_loop_error().expect("the run loop failed").to_string();
        assert_eq!(!error.contains(MODEL_OUTPUT_SECRET), redacted, "{error}");
    }
}

#[tokio::test]
async fn streamed_output_guardrail_omits_run_data_from_redacted_error() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    let model = scripted(vec![vec![assistant_message(MODEL_OUTPUT_SECRET)]]);
    let agent = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_output_guardrail(validating_output_guardrail(Arc::default()));
    let error = run_maybe_streamed(agent, "go", true, RunOptions::default())
        .await
        .unwrap_err();
    assert_redacted_model_behavior(&error, &[MODEL_OUTPUT_SECRET]);
}

#[tokio::test]
async fn streamed_input_guardrail_omits_run_data_from_redacted_error() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    let model = scripted(vec![vec![assistant_message("unused")]]);
    let agent = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_input_guardrail(InputGuardrail::new("validate", |_c, _a, _i| async {
            AgentOutputSchema::new::<RequiredOutput>()?.validate_json(&secret_payload())?;
            unreachable!("validation should fail")
        }));
    let error = run_maybe_streamed(agent, MODEL_OUTPUT_SECRET, true, RunOptions::default())
        .await
        .unwrap_err();
    assert_redacted_model_behavior(&error, &[MODEL_OUTPUT_SECRET]);
}

#[tokio::test]
async fn streamed_session_error_after_output_guardrail_respects_redaction() {
    // The `cancelled` persistence failure is Python-only (asyncio.CancelledError).
    let _lock = LOCK.lock().await;
    for redacted in [false, true] {
        let _f = flags(redacted, false);
        let armed = Arc::new(AtomicBool::new(false));
        let session = Arc::new(FailingSession::new(
            armed.clone(),
            format!("session save failed: {MODEL_OUTPUT_SECRET}"),
        ));
        let model = scripted(vec![vec![assistant_message(MODEL_OUTPUT_SECRET)]]);
        let agent = Agent::<()>::new("A")
            .with_model_instance(model)
            .with_output_guardrail(validating_output_guardrail(armed));
        let error = run_maybe_streamed(agent, "go", true, RunOptions::default().session(session))
            .await
            .unwrap_err();
        if redacted {
            assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
            assert_eq!(error.to_string(), REDACTED);
            assert!(error.run_data().is_none());
            assert!(!logs().contains(MODEL_OUTPUT_SECRET), "{}", logs());
        } else {
            assert!(error.to_string().contains("session save failed"), "{error}");
            assert!(error.to_string().contains(MODEL_OUTPUT_SECRET), "{error}");
        }
    }
}

#[tokio::test]
async fn blocked_terminal_tool_session_failure_is_data_redacted() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let tool_output_secret = "BLOCKED_TERMINAL_TOOL_OUTPUT_SECRET";
    let persistence_secret = "BLOCKED_TERMINAL_SESSION_FAILURE_SECRET";
    for streamed in [false, true] {
        let tool =
            FunctionTool::<()>::new_no_args("terminal_tool", "terminal", move |_c| async move {
                Ok(tool_output_secret.to_string())
            });
        let mut session = FailingSession::new(
            Arc::default(),
            format!("session save failed: {persistence_secret}"),
        );
        session.only_tool_outputs = true;
        let model = scripted(vec![vec![function_call_full(
            "terminal_tool",
            "{}",
            "terminal-call",
            None,
            None,
        )]]);
        let agent = Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("reject", |_c, _a, _o| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            }));
        let error = run_maybe_streamed(
            agent,
            "run terminal tool",
            streamed,
            RunOptions::default().session(Arc::new(session)),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        assert_eq!(error.to_string(), REDACTED, "streamed={streamed}");
        assert!(error.run_data().is_none());
        for secret in [tool_output_secret, persistence_secret] {
            assert!(!format!("{error:?}").contains(secret));
        }
    }
}

fn max_turns_handlers(fallback: &'static str) -> RunErrorHandlers<()> {
    RunErrorHandlers::new()
        .on_max_turns(move |_data| async move { Ok(Some(RunErrorHandlerResult::new(fallback))) })
}

#[tokio::test]
async fn max_turns_recovery_session_failure_preserves_complete_redaction_boundary() {
    // Only the `exception` failure kind exists in Rust; `cancelled`, `direct_base` and
    // exception groups are Python-only.
    let _lock = LOCK.lock().await;
    let persistence_secret = "MAX_TURNS_SESSION_FAILURE_SECRET";
    let fallback_secret = "MAX_TURNS_HANDLER_OUTPUT_SECRET";
    for streamed in [false, true] {
        let _f = flags(true, false);
        let armed = Arc::new(AtomicBool::new(false));
        let session = Arc::new(FailingSession::new(
            armed.clone(),
            format!("session save failed: {persistence_secret}"),
        ));
        let agent = Agent::<()>::new("A")
            .with_model_instance(Arc::new(ScriptedModel::new()))
            .with_output_guardrail(validating_output_guardrail(armed));
        let options = RunOptions::default()
            .max_turns(Some(0))
            .session(session)
            .error_handlers(max_turns_handlers(fallback_secret));
        let error = run_maybe_streamed(agent, "go", streamed, options)
            .await
            .unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        for secret in [persistence_secret, fallback_secret, MODEL_OUTPUT_SECRET] {
            assert!(!error.to_string().contains(secret), "{error}");
            assert!(!format!("{error:?}").contains(secret), "{error:?}");
            assert!(!logs().contains(secret), "{}", logs());
        }
    }
}

#[tokio::test]
async fn max_turns_recovery_session_failure_preserves_diagnostic_context() {
    let _lock = LOCK.lock().await;
    let persistence_secret = "DIAGNOSTIC_MAX_TURNS_SESSION_SECRET";
    for streamed in [false, true] {
        let _f = flags(false, true);
        let armed = Arc::new(AtomicBool::new(false));
        let session = Arc::new(FailingSession::new(armed.clone(), persistence_secret));
        let guard_armed = armed.clone();
        let agent = Agent::<()>::new("A")
            .with_model_instance(Arc::new(ScriptedModel::new()))
            .with_output_guardrail(OutputGuardrail::new("fails", move |_c, _a, _o| {
                let armed = guard_armed.clone();
                async move {
                    armed.store(true, Ordering::SeqCst);
                    Err(AgentsError::other("DIAGNOSTIC_MAX_TURNS_GUARDRAIL_SECRET"))
                }
            }));
        let options = RunOptions::default()
            .max_turns(Some(0))
            .session(session)
            .error_handlers(max_turns_handlers("fallback"));
        let error = run_maybe_streamed(agent, "go", streamed, options)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(persistence_secret), "{error}");
    }
}

// ----------------------------------------------- invalid_final_output handler

#[tokio::test]
async fn invalid_final_output_handler_receives_detached_redacted_error() {
    let _lock = LOCK.lock().await;
    for streamed in [false, true] {
        let _f = flags(true, true);
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let record = seen.clone();
        let handlers = RunErrorHandlers::<()>::new().on_invalid_final_output(move |data| {
            record.lock().unwrap().push(data.error.to_string());
            async {
                Ok(Some(RunErrorHandlerResult::new(
                    json!({"answer": "safe", "count": 1}),
                )))
            }
        });
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        run_maybe_streamed(
            output_type_agent(model),
            "go",
            streamed,
            RunOptions::default().error_handlers(handlers),
        )
        .await
        .unwrap();
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert!(!seen[0].contains(MODEL_OUTPUT_SECRET), "{seen:?}");
    }
}

#[tokio::test]
async fn invalid_final_output_handler_invalid_fallback_preserves_redaction() {
    let _lock = LOCK.lock().await;
    let fallback_secret = "INVALID_HANDLER_FALLBACK_SECRET";
    for streamed in [false, true] {
        let _f = flags(true, true);
        let handlers =
            RunErrorHandlers::<()>::new().on_invalid_final_output(move |_data| async move {
                Ok(Some(RunErrorHandlerResult::new(
                    json!({"answer": fallback_secret}),
                )))
            });
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let error = run_maybe_streamed(
            output_type_agent(model),
            "go",
            streamed,
            RunOptions::default().error_handlers(handlers),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        assert_eq!(error.to_string(), REDACTED);
        assert!(error.run_data().is_none());
        assert!(!logs().contains(fallback_secret));
        assert!(!logs().contains(MODEL_OUTPUT_SECRET));
    }
}

#[tokio::test]
async fn invalid_final_output_handler_failure_preserves_redaction() {
    let _lock = LOCK.lock().await;
    for streamed in [false, true] {
        let _f = flags(true, true);
        let handlers = RunErrorHandlers::<()>::new().on_invalid_final_output(|data| {
            let raw = format!("{:?}", data.run_data.raw_responses);
            async move { Err(AgentsError::other(raw)) }
        });
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let error = run_maybe_streamed(
            output_type_agent(model),
            "go",
            streamed,
            RunOptions::default().error_handlers(handlers),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        assert_eq!(error.to_string(), REDACTED);
        assert!(error.run_data().is_none());
    }
}

#[tokio::test]
async fn invalid_final_output_handler_failure_preserves_diagnostic_context() {
    let _lock = LOCK.lock().await;
    for streamed in [false, true] {
        let _f = flags(false, true);
        let handlers = RunErrorHandlers::<()>::new()
            .on_invalid_final_output(|_data| async { Err(AgentsError::other("handler failed")) });
        let model = scripted(vec![vec![assistant_message(secret_payload())]]);
        let error = run_maybe_streamed(
            output_type_agent(model),
            "go",
            streamed,
            RunOptions::default().error_handlers(handlers),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("handler failed"), "{error}");
    }
}

#[tokio::test]
async fn multiturn_output_validation_error_run_data_follows_redaction_policy() {
    let _lock = LOCK.lock().await;
    #[derive(Deserialize, JsonSchema)]
    struct RecordArgs {
        #[allow(dead_code)]
        value: String,
    }
    for streamed in [false, true] {
        for redacted in [false, true] {
            let _f = flags(redacted, redacted);
            let tool =
                FunctionTool::<()>::new("record_value", "records", |_c, _a: RecordArgs| async {
                    Ok("recorded".to_string())
                });
            let model = scripted(vec![
                vec![function_call_full(
                    "record_value",
                    json!({"value": MODEL_OUTPUT_SECRET}).to_string(),
                    "call-1",
                    None,
                    None,
                )],
                vec![assistant_message(r#"{"answer": "missing count"}"#)],
            ]);
            let agent = output_type_agent(model).with_tool(tool);
            let error = run_maybe_streamed(agent, "go", streamed, RunOptions::default())
                .await
                .unwrap_err();
            assert!(
                matches!(error, AgentsError::ModelBehaviorError { .. }),
                "{error:?}"
            );
            if redacted {
                assert!(error.run_data().is_none());
                assert!(!error.to_string().contains(MODEL_OUTPUT_SECRET));
            } else {
                let run_data = error.run_data().expect("run data");
                assert!(!run_data.raw_responses.is_empty());
                assert!(!run_data.new_items.is_empty());
            }
        }
    }
}
