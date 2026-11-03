//! Function-span assertions for the shell and apply_patch tools (Python's
//! `test_shell_tool.py` and `test_apply_patch_tool.py` span cases).
//!
//! Tracing is process-wide, so these cases live in their own test binary with one recorder.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{ApplyPatchTool, ShellTool};
use openai_agents_sdk::tracing::{Span, Trace, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, AgentsError, Result, Runner};
use serde_json::{Value, json};

/// Records every finished span.
#[derive(Debug, Default)]
struct Recorder {
    ends: Mutex<Vec<Value>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let Some(export) = span.export() {
            self.ends.lock().unwrap().push(export);
        }
    }
}

fn recorder() -> Arc<Recorder> {
    static RECORDER: OnceLock<Arc<Recorder>> = OnceLock::new();
    RECORDER
        .get_or_init(|| {
            openai_agents_sdk::tracing::set_tracing_disabled(false);
            let recorder = Arc::new(Recorder::default());
            set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
            recorder
        })
        .clone()
}

/// Serializes the span tests and clears the recorder, so each test sees only its own spans.
async fn begin() -> tokio::sync::MutexGuard<'static, ()> {
    static SERIAL: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let guard = SERIAL
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    recorder().ends.lock().unwrap().clear();
    guard
}

/// The single function span recorded for `tool_name`.
fn function_span(tool_name: &str) -> Value {
    let spans: Vec<Value> = recorder()
        .ends
        .lock()
        .unwrap()
        .iter()
        .filter(|s| s["span_data"]["type"] == "function" && s["span_data"]["name"] == tool_name)
        .cloned()
        .collect();
    assert_eq!(
        spans.len(),
        1,
        "expected one function span for {tool_name}: {spans:?}"
    );
    spans[0].clone()
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

fn shell_call(call_id: &str) -> Value {
    json!({
        "type": "shell_call",
        "id": "shell_call",
        "call_id": call_id,
        "status": "completed",
        "action": {"type": "exec", "commands": ["echo hi"], "timeout_ms": 1000},
    })
}

fn patch_call(call_id: &str) -> Value {
    json!({
        "type": "apply_patch_call",
        "call_id": call_id,
        "operation": {"type": "update_file", "path": "tasks.md", "diff": "-a\n+b\n"},
    })
}

#[tokio::test]
async fn shell_tool_emits_function_span() {
    let _serial = begin().await;
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![item(shell_call("call_shell_trace"))]);
    model.add_turn(vec![assistant_message("done")]);
    let shell = ShellTool::<()>::new(|_req| async { Ok::<_, AgentsError>("shell span output") });
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());

    Runner::run(&agent, "run").await.unwrap();

    let span = function_span("shell");
    let input = span["span_data"]["input"].as_str().unwrap_or_default();
    assert!(input.contains("echo hi"), "{input}");
    assert_eq!(span["span_data"]["output"], "shell span output");
}

#[tokio::test]
async fn shell_tool_redacts_span_error_when_sensitive_data_disabled() {
    let _serial = begin().await;
    let secret_error = "shell secret output";
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![item(shell_call("call_shell_trace_redacted"))]);
    model.add_turn(vec![assistant_message("done")]);
    let shell =
        ShellTool::<()>::new(
            move |_req| async move { Err::<&str, _>(AgentsError::user(secret_error)) },
        );
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(shell)
        .with_model_instance(model.clone());
    let config = RunConfig::<()>::default().with_trace_include_sensitive_data(false);

    Runner::run_with_options(
        &agent,
        "run",
        openai_agents_sdk::run::RunOptions::<()>::new(()).run_config(config),
    )
    .await
    .unwrap();

    let span = function_span("shell");
    assert_eq!(
        span["error"],
        json!({
            "message": "Error running tool",
            "data": {
                "tool_name": "shell",
                "error": "Tool execution failed. Error details are redacted.",
            },
        })
    );
    assert!(!span.to_string().contains(secret_error));
    assert!(span["span_data"]["input"].is_null());
    assert!(span["span_data"]["output"].is_null());
}

/// An editor whose update always fails with `message`.
struct ExplodingEditor {
    message: String,
}

#[async_trait]
impl ApplyPatchEditor for ExplodingEditor {
    async fn create_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        unreachable!("create is not used by these tests")
    }

    async fn update_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        Err(AgentsError::user(self.message.clone()))
    }

    async fn delete_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        unreachable!("delete is not used by these tests")
    }
}

/// An editor that applies every update with a fixed success message.
struct OkEditor;

#[async_trait]
impl ApplyPatchEditor for OkEditor {
    async fn create_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        unreachable!("create is not used by these tests")
    }

    async fn update_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        Ok(ApplyPatchResult {
            status: None,
            output: Some(format!("Updated {}", op.path)),
        })
    }

    async fn delete_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        unreachable!("delete is not used by these tests")
    }
}

#[tokio::test]
async fn apply_patch_tool_emits_function_span() {
    let _serial = begin().await;
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![item(patch_call("call_apply_trace"))]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = ApplyPatchTool::new(OkEditor);
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());

    Runner::run(&agent, "run").await.unwrap();

    let span = function_span("apply_patch");
    let input = span["span_data"]["input"].as_str().unwrap_or_default();
    assert!(input.contains("tasks.md"), "{input}");
    let output = span["span_data"]["output"].as_str().unwrap_or_default();
    assert!(output.contains("Updated tasks.md"), "{output}");
}

#[tokio::test]
async fn apply_patch_tool_redacts_span_error_when_sensitive_data_disabled() {
    let _serial = begin().await;
    let secret_error = "patch secret output";
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![item(patch_call("call_apply_trace_redacted"))]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = ApplyPatchTool::new(ExplodingEditor {
        message: secret_error.into(),
    });
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let config = RunConfig::<()>::default().with_trace_include_sensitive_data(false);

    Runner::run_with_options(
        &agent,
        "run",
        openai_agents_sdk::run::RunOptions::<()>::new(()).run_config(config),
    )
    .await
    .unwrap();

    let span = function_span("apply_patch");
    assert_eq!(
        span["error"],
        json!({
            "message": "Error running tool",
            "data": {
                "tool_name": "apply_patch",
                "error": "Tool execution failed. Error details are redacted.",
            },
        })
    );
    assert!(!span.to_string().contains(secret_error));
    assert!(span["span_data"]["input"].is_null());
    assert!(span["span_data"]["output"].is_null());
}
