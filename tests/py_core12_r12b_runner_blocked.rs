//! Ports of the blocked-output sanitization cases of `test_agent_runner.py` that were filed as
//! Python-only because Python calls the private `_retained_items_for_blocked_output`; the same
//! behaviour is observable through the session a tripped terminal tool call leaves behind.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::FakeModel;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn tripping_agent(model: &Arc<FakeModel>, runs: &Arc<Mutex<usize>>) -> Arc<Agent<()>> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let runs = runs.clone();
    let tool = FunctionTool::<()>::new("commit_tool", "a tool", move |_ctx, _a: NoArgs| {
        let runs = runs.clone();
        async move {
            *runs.lock().unwrap() += 1;
            Ok("raw-secret".to_string())
        }
    });
    Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("guard", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            })),
    )
}

async fn saved_after_trip(call: Value) -> (Vec<Value>, usize) {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(call)]);
    let runs = Arc::new(Mutex::new(0));
    let agent = tripping_agent(&model, &runs);
    let session = Arc::new(InMemorySession::new("blocked"));
    let err = Runner::run_with_options(
        agent,
        "go",
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    let saved = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|item| item.0)
        .collect();
    let executed = *runs.lock().unwrap();
    (saved, executed)
}

/// Python: `test_blocked_function_batch_is_rebuilt_from_allowlisted_fields` (`json-object` and
/// `empty` arguments). The retained call keeps only its allowlisted fields: a provider's extra
/// payload on the call never reaches the session, and the output is the blocked placeholder.
#[tokio::test]
async fn a_retained_blocked_call_is_rebuilt_from_allowlisted_fields() {
    for arguments in ["{}", ""] {
        let (saved, executed) = saved_after_trip(json!({
            "type": "function_call", "name": "commit_tool", "arguments": arguments,
            "call_id": "call-commit", "provider_data": {"secret": "call-secret"}
        }))
        .await;
        assert_eq!(executed, 1, "{arguments:?}");
        let text = json!(saved).to_string();
        assert!(!text.contains("call-secret"), "{arguments:?}: {text}");
        assert!(!text.contains("raw-secret"), "{arguments:?}: {text}");
        let call = saved
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap_or_else(|| panic!("{arguments:?}: no retained call in {saved:?}"));
        assert_eq!(call["arguments"], arguments);
        assert_eq!(call["name"], "commit_tool");
        assert_eq!(call["call_id"], "call-commit");
        assert!(call.get("provider_data").is_none(), "{call}");
    }
}

/// Python: `test_blocked_function_batch_accepts_exact_typed_direct_caller`: a direct caller
/// survives sanitization. (A program caller needs a programmatic tool to get this far.)
#[tokio::test]
async fn only_a_direct_caller_survives_blocked_output_sanitization() {
    let (saved, _) = saved_after_trip(json!({
        "type": "function_call", "name": "commit_tool", "arguments": "{}",
        "call_id": "call-commit", "caller": {"type": "direct"}
    }))
    .await;
    let call = saved
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap_or_else(|| panic!("no retained call in {saved:?}"));
    assert_eq!(call["caller"], json!({"type": "direct"}));
}

/// Python: `test_blocked_function_batch_rejects_non_direct_typed_callers`: a retained batch whose
/// call was made by a program cannot be replayed partially, so the whole response is dropped.
#[tokio::test]
async fn a_program_caller_discards_the_whole_blocked_response() {
    use openai_agents_sdk::tool::ProgrammaticToolCallingTool;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        OutputItem::from_value(json!({
            "type": "program", "id": "program_item", "call_id": "call_program",
            "code": "commit_tool()", "fingerprint": "fingerprint",
        })),
        OutputItem::from_value(json!({
            "type": "function_call", "id": "function_item", "call_id": "call-commit",
            "name": "commit_tool", "arguments": "{}",
            "caller": {"type": "program", "caller_id": "call_program"},
        })),
    ]);
    let tool = FunctionTool::<()>::new("commit_tool", "a tool", |_ctx, _a: NoArgs| async {
        Ok("raw-secret".to_string())
    })
    .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(tool)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("guard", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            })),
    );
    let session = Arc::new(InMemorySession::new("blocked-program"));
    let err = Runner::run_with_options(
        agent,
        "go",
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    let saved: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|item| item.0)
        .collect();
    let text = json!(saved).to_string();
    assert!(
        !text.contains("raw-secret") && !text.contains("commit_tool"),
        "{text}"
    );
    assert_eq!(saved.len(), 1, "only the user input remains: {saved:?}");
}

/// Python: `test_blocked_unknown_tool_variant_discards_the_complete_response`: a custom tool
/// call next to the terminal function call cannot be retained, so the response is dropped whole.
#[tokio::test]
async fn an_unsupported_tool_variant_discards_the_whole_blocked_response() {
    use openai_agents_sdk::tool::CustomTool;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        OutputItem::from_value(json!({
            "type": "function_call", "call_id": "call-commit", "name": "commit_tool",
            "arguments": "{}",
        })),
        OutputItem::from_value(json!({
            "type": "custom_tool_call", "call_id": "call-custom", "name": "sql",
            "input": "select secret",
        })),
    ]);
    let tool = FunctionTool::<()>::new("commit_tool", "a tool", |_ctx, _a: NoArgs| async {
        Ok("raw-secret".to_string())
    });
    let custom = CustomTool::<()>::new("sql", "Run SQL", |_ctx, _input| async move {
        Ok("custom-secret".to_string())
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool)
            .with_tool(custom)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("guard", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            })),
    );
    let session = Arc::new(InMemorySession::new("blocked-custom"));
    let err = Runner::run_with_options(
        agent,
        "go",
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    let saved: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|item| item.0)
        .collect();
    let text = json!(saved).to_string();
    assert!(
        !text.contains("secret") && !text.contains("commit_tool") && !text.contains("sql"),
        "{text}"
    );
    assert_eq!(saved.len(), 1, "only the user input remains: {saved:?}");
}
