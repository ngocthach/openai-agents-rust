//! Ports of the invocation tests in `tests/test_function_tool.py`: failure policies, timeouts,
//! invalid-input messages, tool-data logging and clone semantics.
//!
//! Failure policies and timeouts only apply inside a run, so these go through `Runner` with a
//! scripted model. Raw `invoke` is used where Python calls `on_invoke_tool` with no policy.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, ToolTimeoutBehavior};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

const GENERIC: &str = "An error occurred while running the tool. Please try again.";
const SECRET: &str = "SECRET_TOKEN_123";

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct SimpleArgs {
    a: i64,
    #[serde(default = "five")]
    b: i64,
}

fn five() -> i64 {
    5
}

#[derive(Deserialize, JsonSchema)]
struct EchoArgs {
    value: String,
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

fn install_logger() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// Serializes tests that toggle the process-wide tool-data flag or read the shared log buffer.
/// Restores the default (tool data redacted) on drop.
struct Fixture {
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        reset_dont_log_tool_data();
    }
}

async fn fixture(dont_log_tool_data: bool) -> Fixture {
    let guard = LOCK.lock().await;
    install_logger();
    RECORDS.lock().unwrap().clear();
    set_dont_log_tool_data(dont_log_tool_data);
    Fixture { _guard: guard }
}

fn logs() -> String {
    RECORDS.lock().unwrap().join("\n")
}

/// `model_output`: the `output` of the first `function_call_output` sent back to the model.
fn model_output(model: &ScriptedModel) -> Value {
    model
        .last_call()
        .expect("a model call")
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .find(|i| i["type"] == "function_call_output")
        .expect("a function_call_output")["output"]
        .clone()
}

/// Run one model-driven call of `tool` with `args` through the Runner. Returns the output the
/// model received, or the run error.
async fn call_once(tool: FunctionTool<()>, args: &str) -> Result<String> {
    let name = tool.name.clone();
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call_full(&name, args, "call-1", None, None)]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());
    Runner::run(agent, "start").await?;
    Ok(model_output(&model)
        .as_str()
        .unwrap_or_default()
        .to_string())
}

/// Invoke `tool` directly with raw JSON arguments, bypassing the run's failure policy.
async fn invoke_direct(tool: &FunctionTool<()>, args: &str) -> Result<String> {
    let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, args);
    tool.invoke(ctx, args.to_string())
        .await
        .map(|out| out.to_model_string())
}

/// A tool whose body fails with a plain error, like Python's `raise ValueError("test")`.
fn failing_simple(name: &str) -> FunctionTool<()> {
    FunctionTool::<()>::new(name, "d", |_ctx, _args: SimpleArgs| async {
        Err::<i64, _>(AgentsError::Other("test".to_string()))
    })
}

/// Map an error to the class name the Python tests assert on.
fn error_class(error: &AgentsError) -> &'static str {
    match error {
        AgentsError::ModelBehaviorError { .. } => "ModelBehaviorError",
        _ => "ValueError",
    }
}

// ---------------------------------------------------------------------------
// Failure policies
// ---------------------------------------------------------------------------

#[tokio::test]
async fn function_tool_default_error_works() {
    let _f = fixture(true).await;
    let tool = failing_simple("my_func");
    // A missing argument (`""`, `"{}"`) and a body failure both produce the generic message.
    for args in ["", "{}", r#"{"a": 1}"#, r#"{"a": 1, "b": 2}"#] {
        assert_eq!(
            call_once(tool.clone(), args).await.unwrap(),
            GENERIC,
            "{args}"
        );
    }
}

#[tokio::test]
async fn sync_and_async_custom_error_functions_work() {
    // Rust tool bodies are always async, so the Python sync and async variants are one case.
    let _f = fixture(true).await;
    let tool = failing_simple("my_func")
        .with_failure_error_function(|_ctx, error| format!("error_{}", error_class(error)));
    assert_eq!(
        call_once(tool.clone(), "").await.unwrap(),
        "error_ModelBehaviorError"
    );
    assert_eq!(
        call_once(tool.clone(), "{}").await.unwrap(),
        "error_ModelBehaviorError"
    );
    assert_eq!(
        call_once(tool.clone(), r#"{"a": 1}"#).await.unwrap(),
        "error_ValueError"
    );
    assert_eq!(
        call_once(tool, r#"{"a": 1, "b": 2}"#).await.unwrap(),
        "error_ValueError"
    );
}

#[tokio::test]
async fn failure_error_function_is_applied_to_a_renamed_copy() {
    // Python: `dataclasses.replace` keeps the custom handler. Rust: clone and rename.
    let _f = fixture(true).await;
    let original = FunctionTool::<()>::new_no_args("boom", "d", |_ctx| async {
        Err::<String, _>(AgentsError::Other("boom".to_string()))
    })
    .with_failure_error_function(|_ctx, error| format!("handled:{error}"));
    let mut copied = original.clone();
    copied.name = "copied_tool".to_string();

    assert_eq!(call_once(copied, "{}").await.unwrap(), "handled:boom");
}

#[tokio::test]
async fn replaced_tool_normal_failure_uses_replaced_policy() {
    let _f = fixture(true).await;
    let original = FunctionTool::<()>::new_no_args("boom", "d", |_ctx| async {
        Err::<String, _>(AgentsError::Other("kapow".to_string()))
    });
    let mut replaced = original.clone().without_failure_error_function();
    replaced.name = "replaced_tool".to_string();

    let err = call_once(replaced, "{}").await.unwrap_err();
    assert!(err.to_string().contains("kapow"), "{err}");
}

#[tokio::test]
async fn shallow_copied_tool_normal_failure_uses_copied_policy() {
    let _f = fixture(true).await;
    let original = FunctionTool::<()>::new_no_args("boom", "d", |_ctx| async {
        Err::<String, _>(AgentsError::Other("kapow".to_string()))
    });
    let mut copied = original.clone().without_failure_error_function();
    copied.name = "copied_tool".to_string();

    let err = call_once(copied, "{}").await.unwrap_err();
    assert!(err.to_string().contains("kapow"), "{err}");
    // The original keeps the default policy.
    assert_eq!(call_once(original, "{}").await.unwrap(), GENERIC);
}

#[tokio::test]
async fn default_failure_policy_survives_clone() {
    let _f = fixture(true).await;
    let tool = FunctionTool::<()>::new_no_args("boom", "d", |_ctx| async {
        Err::<String, _>(AgentsError::Other("kapow".to_string()))
    });
    let copied = tool.clone();
    assert_eq!(call_once(copied, "{}").await.unwrap(), GENERIC);
}

#[tokio::test]
async fn manual_function_tool_uses_default_failure_error_function() {
    let _f = fixture(true).await;
    let manual = FunctionTool::<()>::try_new_raw(
        "manual_cancel_tool",
        "manual cancel",
        serde_json::json!({}),
        true,
        |_ctx: ToolContext<()>, _args: String| async {
            Err::<String, _>(AgentsError::Other("manual-tool-failed".to_string()))
        },
    )
    .unwrap();
    assert_eq!(call_once(manual, "{}").await.unwrap(), GENERIC);
}

#[tokio::test]
async fn copied_tool_invalid_input_uses_current_name() {
    // Python covers `dataclasses.replace` and `copy.copy`; both are a clone with a new name here.
    let _f = fixture(true).await;
    let original =
        FunctionTool::<()>::new("original_tool", "d", |_ctx, args: EchoArgs| async move {
            Ok(args.value)
        })
        .without_failure_error_function();
    let mut copied = original.clone();
    copied.name = "copied_tool".to_string();

    let err = call_once(copied, "{}").await.unwrap_err();
    assert!(
        err.to_string()
            .starts_with("Invalid JSON input for tool copied_tool"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Timeouts
// ---------------------------------------------------------------------------

fn slow_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args(name, "d", |_ctx| async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok("slow".to_string())
    })
}

#[tokio::test]
async fn invoke_function_tool_timeout_returns_default_message() {
    let _f = fixture(true).await;
    let tool = slow_tool("slow_tool").with_timeout(Duration::from_millis(10));
    let message = call_once(tool, "{}").await.unwrap();
    assert!(message.to_lowercase().contains("timed out"), "{message}");
    assert!(message.contains("0.01"), "{message}");
    assert_eq!(message, "Tool 'slow_tool' timed out after 0.01 seconds.");
}

#[tokio::test]
async fn invoke_function_tool_timeout_uses_custom_error_function() {
    let _f = fixture(true).await;
    let tool = slow_tool("slow_tool")
        .with_timeout(Duration::from_millis(10))
        .with_timeout_error_function(|_ctx, error| match error {
            AgentsError::ToolTimeoutError {
                tool_name,
                timeout_seconds,
                ..
            } => format!("custom_timeout:{tool_name}:{timeout_seconds}"),
            other => other.to_string(),
        });
    assert_eq!(
        call_once(tool, "{}").await.unwrap(),
        "custom_timeout:slow_tool:0.01"
    );
}

#[tokio::test]
async fn invoke_function_tool_timeout_can_raise_exception() {
    let _f = fixture(true).await;
    let tool = slow_tool("slow_tool")
        .with_timeout(Duration::from_millis(10))
        .with_timeout_behavior(ToolTimeoutBehavior::RaiseException);
    let err = call_once(tool, "{}").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ToolTimeoutError { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("timed out"), "{err}");
}

#[tokio::test]
async fn tool_raised_timeout_error_is_not_rewritten() {
    let _f = fixture(true).await;
    let tool = FunctionTool::<()>::new_no_args("timeout_tool", "d", |_ctx| async {
        Err::<String, _>(AgentsError::Other("tool_internal_timeout".to_string()))
    })
    .with_timeout(Duration::from_secs(1))
    .without_failure_error_function();
    let err = call_once(tool, "{}").await.unwrap_err();
    assert!(
        !matches!(err, AgentsError::ToolTimeoutError { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("tool_internal_timeout"), "{err}");
}

#[tokio::test]
async fn manual_tool_raised_timeout_error_is_not_rewritten() {
    let _f = fixture(true).await;
    let manual = FunctionTool::<()>::try_new_raw(
        "manual_timeout_tool",
        "manual timeout",
        serde_json::json!({}),
        true,
        |_ctx: ToolContext<()>, _args: String| async {
            Err::<String, _>(AgentsError::Other(
                "manual_tool_internal_timeout".to_string(),
            ))
        },
    )
    .unwrap()
    .with_timeout(Duration::from_secs(1))
    .without_failure_error_function();
    let err = call_once(manual, "{}").await.unwrap_err();
    assert!(
        !matches!(err, AgentsError::ToolTimeoutError { .. }),
        "{err:?}"
    );
    assert!(
        err.to_string().contains("manual_tool_internal_timeout"),
        "{err}"
    );
}

#[test]
fn function_tool_timeout_must_be_positive() {
    let tool = FunctionTool::<()>::new_no_args("bad_timeout", "bad", |_ctx| async { Ok("ok") });
    let err = tool.try_with_timeout(Duration::ZERO).unwrap_err();
    assert!(err.to_string().contains("greater than 0"), "{err}");
}

// ---------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn function_tool_rejects_non_object_json_input() {
    let _f = fixture(true).await;
    let tool = FunctionTool::<()>::new("echo_tool", "d", |_ctx, args: EchoArgs| async move {
        Ok(args.value)
    })
    .without_failure_error_function();
    for input in ["[]", r#""value""#, "123", "null", "true"] {
        let err = invoke_direct(&tool, input).await.unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{input}: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "Invalid JSON input for tool echo_tool: expected a JSON object",
            "{input}"
        );
    }
}

#[tokio::test]
async fn function_tool_bad_json_redacts_payload_when_tool_data_is_not_logged() {
    let _f = fixture(true).await;
    let tool = FunctionTool::<()>::new("echo_tool", "d", |_ctx, args: EchoArgs| async move {
        Ok(args.value)
    })
    .without_failure_error_function();
    let bad_json = format!(r#"{{"secret":"{SECRET}""#);

    let err = invoke_direct(&tool, &bad_json).await.unwrap_err();
    assert_eq!(err.to_string(), "Invalid JSON input for tool echo_tool");
    assert!(!err.to_string().contains(SECRET));
    assert!(
        !logs().contains(SECRET),
        "logs leaked the payload: {}",
        logs()
    );
}

#[tokio::test]
async fn function_tool_bad_json_includes_payload_when_tool_logging_enabled() {
    let _f = fixture(false).await;
    let tool = FunctionTool::<()>::new("echo_tool", "d", |_ctx, args: EchoArgs| async move {
        Ok(args.value)
    })
    .without_failure_error_function();
    let bad_json = format!(r#"{{"secret":"{SECRET}""#);

    let err = invoke_direct(&tool, &bad_json).await.unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("Invalid JSON input for tool echo_tool: {bad_json}")
    );
    assert!(err.to_string().contains(SECRET));
    assert!(logs().contains(SECRET), "logs: {}", logs());
}

#[tokio::test]
async fn function_tool_argument_logging_excludes_live_context() {
    struct SensitiveContext;
    impl std::fmt::Debug for SensitiveContext {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CONTEXT_SECRET_SENTINEL")
        }
    }

    let _f = fixture(false).await;
    let model_argument = "MODEL_ARGUMENT_SENTINEL";
    let tool = FunctionTool::<SensitiveContext>::new(
        "echo",
        "d",
        |_ctx: ToolContext<SensitiveContext>, args: EchoArgs| async move { Ok(args.value) },
    );
    let arguments = serde_json::json!({"value": model_argument}).to_string();
    let ctx = ToolContext::direct(
        RunContext::new(SensitiveContext),
        "test",
        &tool.name,
        &arguments,
    );

    assert_eq!(
        tool.invoke(ctx, arguments.clone())
            .await
            .unwrap()
            .to_model_string(),
        model_argument
    );
    let captured = logs();
    assert!(captured.contains(model_argument), "logs: {captured}");
    assert!(
        !captured.contains("CONTEXT_SECRET_SENTINEL"),
        "logs: {captured}"
    );
}
