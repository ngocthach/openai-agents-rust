//! Ported from the Python SDK's `tests/test_guardrails.py`: guardrails run directly, and the
//! run-in-parallel flag. The Python decorators (`@input_guardrail`, `@output_guardrail`) have no
//! Rust counterpart; the Rust closure constructors take the same name and flag.

use std::sync::Arc;

use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::{Agent, InputGuardrail, OutputGuardrail, RunContext, Runner};
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn agent() -> Arc<Agent<()>> {
    Arc::new(Agent::<()>::new("test"))
}

fn input_guardrail(triggers: bool, output_info: Value) -> InputGuardrail<()> {
    InputGuardrail::new("guardrail", move |_c, _a, _i| {
        let output_info = output_info.clone();
        async move { Ok(GuardrailFunctionOutput::new(output_info, triggers)) }
    })
}

fn output_guardrail(triggers: bool, output_info: Value) -> OutputGuardrail<()> {
    OutputGuardrail::new("guardrail", move |_c, _a, _o| {
        let output_info = output_info.clone();
        async move { Ok(GuardrailFunctionOutput::new(output_info, triggers)) }
    })
}

async fn run_input(
    guardrail: &InputGuardrail<()>,
) -> openai_agents_sdk::guardrail::InputGuardrailResult {
    guardrail
        .run(RunContext::new(()), agent(), vec!["test".into()])
        .await
        .unwrap()
}

async fn run_output(
    guardrail: &OutputGuardrail<()>,
) -> openai_agents_sdk::guardrail::OutputGuardrailResult {
    guardrail
        .run(RunContext::new(()), agent(), json!("test"))
        .await
        .unwrap()
}

#[tokio::test]
async fn no_input_guardrails_leave_the_results_empty() {
    disable_tracing();
    let model = openai_agents_sdk::testing::ScriptedModel::new();
    model
        .enqueue(openai_agents_sdk::testing::ScriptedStep::output(vec![
            openai_agents_sdk::testing::assistant_message("ok"),
        ]))
        .unwrap();
    let agent = Agent::<()>::new("test").with_model_instance(Arc::new(model));

    let result = Runner::run(&agent, "test").await.unwrap();

    assert!(result.input_guardrail_results.is_empty());
}

#[tokio::test]
async fn sync_style_input_guardrail_reports_triggers_and_output_info() {
    disable_tracing();
    let passing = run_input(&input_guardrail(false, Value::Null)).await;
    assert!(!passing.output.tripwire_triggered);
    assert_eq!(passing.output.output_info, Value::Null);

    let tripped = run_input(&input_guardrail(true, Value::Null)).await;
    assert!(tripped.output.tripwire_triggered);
    assert_eq!(tripped.output.output_info, Value::Null);

    let tripped_with_info = run_input(&input_guardrail(true, json!("test"))).await;
    assert!(tripped_with_info.output.tripwire_triggered);
    assert_eq!(tripped_with_info.output.output_info, json!("test"));
}

#[tokio::test]
async fn async_input_guardrail_reports_triggers_and_output_info() {
    disable_tracing();
    let guardrail = InputGuardrail::new("async_guardrail", |_c, _a, _i| async move {
        tokio::task::yield_now().await;
        Ok(GuardrailFunctionOutput::new(json!("test"), true))
    });

    let result = run_input(&guardrail).await;

    assert_eq!(result.guardrail_name, "async_guardrail");
    assert!(result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test"));
}

#[tokio::test]
async fn sync_style_output_guardrail_reports_triggers_and_output_info() {
    disable_tracing();
    let passing = run_output(&output_guardrail(false, Value::Null)).await;
    assert!(!passing.output.tripwire_triggered);
    assert_eq!(passing.output.output_info, Value::Null);

    let tripped = run_output(&output_guardrail(true, Value::Null)).await;
    assert!(tripped.output.tripwire_triggered);
    assert_eq!(tripped.output.output_info, Value::Null);

    let tripped_with_info = run_output(&output_guardrail(true, json!("test"))).await;
    assert!(tripped_with_info.output.tripwire_triggered);
    assert_eq!(tripped_with_info.output.output_info, json!("test"));
}

#[tokio::test]
async fn async_output_guardrail_reports_triggers_and_output_info() {
    disable_tracing();
    let guardrail = OutputGuardrail::new("async_guardrail", |_c, _a, _o| async move {
        tokio::task::yield_now().await;
        Ok(GuardrailFunctionOutput::new(json!("test"), true))
    });

    let result = run_output(&guardrail).await;

    assert_eq!(result.guardrail_name, "async_guardrail");
    assert!(result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test"));
}

#[tokio::test]
async fn input_guardrail_keeps_its_explicit_name() {
    disable_tracing();
    let guardrail = InputGuardrail::new("Custom name", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(json!("test_2"), false))
    });

    let result = run_input(&guardrail).await;

    assert!(!result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test_2"));
    assert_eq!(result.guardrail_name, "Custom name");
    assert_eq!(guardrail.name, "Custom name");
}

#[tokio::test]
async fn output_guardrail_keeps_its_explicit_name() {
    disable_tracing();
    let guardrail = OutputGuardrail::new("Custom name", |_c, _a, _o| async move {
        Ok(GuardrailFunctionOutput::new(json!("test_4"), false))
    });

    let result = run_output(&guardrail).await;

    assert!(!result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test_4"));
    assert_eq!(result.guardrail_name, "Custom name");
    assert_eq!(guardrail.name, "Custom name");
}

#[tokio::test]
async fn input_guardrail_runs_in_parallel_by_default() {
    let guardrail = input_guardrail(false, Value::Null);
    assert!(guardrail.run_in_parallel);
}

#[tokio::test]
async fn input_guardrail_blocking_flag_turns_off_parallel_run() {
    let guardrail = input_guardrail(false, Value::Null).blocking();
    assert!(!guardrail.run_in_parallel);
}

#[tokio::test]
async fn blocking_input_guardrail_runs_and_reports_its_output_info() {
    disable_tracing();
    let guardrail = InputGuardrail::new("blocking_guardrail", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(json!("blocking"), false))
    })
    .blocking();
    assert!(!guardrail.run_in_parallel);

    let result = run_input(&guardrail).await;

    assert!(!result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("blocking"));
}

#[tokio::test]
async fn named_blocking_input_guardrail_keeps_name_and_flag() {
    let guardrail = InputGuardrail::<()>::new("custom_name", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(json!("named_blocking"), false))
    })
    .blocking();

    assert_eq!(guardrail.name, "custom_name");
    assert!(!guardrail.run_in_parallel);
}
