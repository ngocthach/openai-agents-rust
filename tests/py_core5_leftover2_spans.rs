//! Ports of the tracing cases of `test_computer_action.py` (the computer tool's function span,
//! the GA tool name in the agent span, batched-action span input, redacted span errors) and of
//! the span-error case of `test_call_model_input_filter.py`.
//!
//! Tracing is process-wide, so these cases live in their own test binary with one recorder.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::ComputerTool;
use openai_agents_sdk::tracing::{Span, Trace, TracingProcessor, set_trace_processors, trace};
use openai_agents_sdk::{Agent, AgentsError, Result, Runner};
use serde_json::{Value, json};

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

async fn begin() -> tokio::sync::MutexGuard<'static, ()> {
    static SERIAL: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let guard = SERIAL
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    recorder().ends.lock().unwrap().clear();
    guard
}

fn span_of(kind: &str, name: &str) -> Value {
    recorder()
        .ends
        .lock()
        .unwrap()
        .iter()
        .find(|s| s["span_data"]["type"] == kind && s["span_data"]["name"] == name)
        .cloned()
        .unwrap_or_else(|| panic!("no {kind} span named {name}"))
}

struct StubComputer {
    fail_with: Option<String>,
}

#[async_trait]
impl Computer for StubComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Mac)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((800, 600))
    }
    async fn screenshot(&self) -> Result<String> {
        match &self.fail_with {
            Some(message) => Err(AgentsError::other(message.clone())),
            None => Ok("trace_img".to_string()),
        }
    }
    async fn click(&self, _: i64, _: i64, _: &str, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn double_click(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn scroll(&self, _: i64, _: i64, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> Result<()> {
        Ok(())
    }
    async fn wait(&self) -> Result<()> {
        Ok(())
    }
    async fn move_mouse(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
}

fn call(call_id: &str, body: Value) -> OutputItem {
    let mut call = json!({
        "type": "computer_call", "id": call_id, "call_id": call_id,
        "pending_safety_checks": [], "status": "completed",
    });
    for (key, value) in body.as_object().unwrap() {
        call[key] = value.clone();
    }
    OutputItem::from_value(call)
}

fn agent(name: &str, model: &Arc<FakeModel>, fail_with: Option<&str>) -> Agent<()> {
    Agent::<()>::new(name)
        .with_tool(ComputerTool::<()>::new(StubComputer {
            fail_with: fail_with.map(str::to_string),
        }))
        .with_model_instance(model.clone())
}

fn model_with(call: OutputItem) -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![call]);
    model.add_turn(vec![assistant_message("done")]);
    model
}

/// Python: `test_execute_emits_function_span`.
#[tokio::test]
async fn the_computer_tool_emits_a_function_span() {
    let _serial = begin().await;
    let model = model_with(call(
        "tool_trace",
        json!({"action": {"type": "screenshot"}}),
    ));
    let agent = agent("test_agent_trace", &model, None);
    trace("computer-span-test")
        .run(async { Runner::run(&agent, "go").await })
        .await
        .unwrap();

    let span = span_of("function", "computer");
    assert!(!span["span_data"]["input"].is_null());
    assert!(
        span["span_data"]["output"]
            .as_str()
            .unwrap_or_default()
            .starts_with("data:image/png;base64,")
    );
}

/// Python: `test_runner_trace_lists_ga_computer_tool_name`.
#[tokio::test]
async fn the_agent_span_lists_the_ga_computer_tool_name() {
    let _serial = begin().await;
    let model = model_with(call(
        "tool_trace_agent_tools",
        json!({"action": {"type": "screenshot"}}),
    ));
    let agent = agent("test_agent_trace_tools", &model, None);
    let result = trace("computer-agent-span-test")
        .run(async { Runner::run(&agent, "take a screenshot").await })
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let span = span_of("agent", "test_agent_trace_tools");
    assert_eq!(span["span_data"]["tools"], json!(["computer"]));
}

/// Python: `test_execute_emits_batched_actions_in_function_span`.
#[tokio::test]
async fn batched_actions_are_the_function_span_input() {
    let _serial = begin().await;
    let model = model_with(call(
        "tool_trace_batch",
        json!({"actions": [
            {"type": "click", "x": 5, "y": 6, "button": "left"},
            {"type": "type", "text": "batched"},
        ]}),
    ));
    let agent = agent("test_agent_trace_batch", &model, None);
    trace("computer-batch-span-test")
        .run(async { Runner::run(&agent, "go").await })
        .await
        .unwrap();

    let span = span_of("function", "computer");
    let input: Value = serde_json::from_str(span["span_data"]["input"].as_str().unwrap()).unwrap();
    assert_eq!(
        input,
        json!([
            {"type": "click", "x": 5, "y": 6, "button": "left"},
            {"type": "type", "text": "batched"},
        ])
    );
}

/// Python: `test_execute_redacts_span_error_when_sensitive_data_disabled`.
#[tokio::test]
async fn span_errors_are_redacted_when_sensitive_data_is_disabled() {
    let _serial = begin().await;
    let secret_error = "computer secret output";
    let model = model_with(call(
        "tool_trace_error",
        json!({"action": {"type": "screenshot"}}),
    ));
    let agent = agent("test_agent_trace_error", &model, Some(secret_error));
    let config = RunConfig::<()>::default().with_trace_include_sensitive_data(false);
    let outcome = trace("computer-span-redaction-test")
        .run(async {
            Runner::run_with_options(&agent, "go", RunOptions::<()>::new(()).run_config(config))
                .await
        })
        .await;
    // Python raises the driver error; the Rust runner surfaces it as the run error.
    let error = outcome.expect_err("the driver failure fails the run");
    assert!(error.to_string().contains(secret_error), "{error}");

    let span = span_of("function", "computer");
    assert_eq!(
        span["error"],
        json!({
            "message": "Error running tool",
            "data": {
                "tool_name": "computer",
                "error": "Tool execution failed. Error details are redacted.",
            },
        })
    );
    assert!(!span.to_string().contains(secret_error));
    assert!(span["span_data"]["input"].is_null());
    assert!(span["span_data"]["output"].is_null());
}

/// Python: `test_call_model_input_filter_error_respects_sensitive_data_setting` (both settings).
#[tokio::test]
async fn call_model_input_filter_errors_respect_the_sensitive_data_setting() {
    const SENSITIVE: &str = "sensitive-filter-error";
    for (include_sensitive, expected) in [(false, "Error details are redacted."), (true, SENSITIVE)]
    {
        let _serial = begin().await;
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("never reached")]);
        let agent = Agent::<()>::new("test").with_model_instance(model);
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(include_sensitive)
            .with_call_model_input_filter(|_data| async { Err(AgentsError::other(SENSITIVE)) });
        let error = Runner::run_with_options(
            &agent,
            "start",
            RunOptions::<()>::new(()).run_config(config),
        )
        .await
        .expect_err("the filter failure fails the run");
        assert!(error.to_string().contains(SENSITIVE), "{error}");

        let errors: Vec<Value> = recorder()
            .ends
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s["span_data"]["data"]["sdk_span_type"] == "turn")
            .filter_map(|s| s.get("error").filter(|e| !e.is_null()).cloned())
            .collect();
        assert_eq!(
            errors,
            vec![json!({
                "message": "Error in call_model_input_filter",
                "data": {"error": expected},
            })]
        );
    }
}
