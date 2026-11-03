//! Port of `test_factory_callback_recovery_does_not_expose_error_or_undo_side_effects` from the
//! Python SDK's `tests/test_handoff_failure_error_function.py`.
//!
//! Trace processors and the tracing switch are process-global, so this case lives in its own
//! test binary.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::default_tool_error_function;
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, set_tracing_disabled,
};
use openai_agents_sdk::{Agent, AgentsError, Runner, handoff};
use serde_json::{Value, json};

/// The failing callback's side effects stay, and the caller sees only the formatter's text. The
/// handoff span records the generic failure message without the private detail.
#[tokio::test]
async fn recovery_does_not_expose_the_error_or_undo_side_effects() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    set_tracing_disabled(false);

    let effects = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = effects.clone();
    let transfer = handoff(Agent::<()>::new("target"))
        .with_on_handoff(move |_ctx| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push("callback ran".to_string());
                Err(AgentsError::user("synthetic-private-detail"))
            }
        })
        .with_failure_error_function(default_tool_error_function);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "transfer_to_target",
        "{}",
        "call",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let caller = Arc::new(
        Agent::<()>::new("caller")
            .with_model_instance(model.clone())
            .with_handoff(transfer),
    );

    let result = Runner::run_with_options(caller, "go", RunOptions::new(()))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(*effects.lock().unwrap(), vec!["callback ran".to_string()]);
    assert!(
        !model.calls()[1]
            .input
            .iter()
            .any(|i| i.0.to_string().contains("synthetic-private-detail"))
    );

    let errors: Vec<Value> = processor
        .items()
        .into_iter()
        .filter(|s| s["object"] == "trace.span" && s["span_data"]["type"] == "handoff")
        .filter_map(|s| s.get("error").filter(|e| !e.is_null()).cloned())
        .collect();
    assert_eq!(
        errors,
        vec![json!({"message": "Handoff invocation failed", "data": {}})]
    );
}
