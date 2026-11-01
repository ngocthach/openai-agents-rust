//! Ports of `tests/test_agent_runner.py` cases that Python reaches through private helpers.

use std::sync::Arc;

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

/// Python: `test_blocked_reasoning_item_discards_the_complete_response`. A response carrying a
/// reasoning item beside the terminal function call cannot be replayed partially, so a tripped
/// output guardrail leaves nothing of it in the session.
#[tokio::test]
async fn a_reasoning_item_discards_the_whole_blocked_response() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        OutputItem::from_value(json!({
            "type": "reasoning", "id": "reasoning-id", "summary": [],
            "encrypted_content": "reasoning-secret",
        })),
        OutputItem::from_value(json!({
            "type": "function_call", "call_id": "call-commit", "name": "commit_tool",
            "arguments": "{}",
        })),
    ]);
    let tool = FunctionTool::<()>::new("commit_tool", "a tool", |_ctx, _a: NoArgs| async {
        Ok("raw-secret".to_string())
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(OutputGuardrail::new("guard", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            })),
    );
    let session = Arc::new(InMemorySession::new("blocked-reasoning"));
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
        !text.contains("secret") && !text.contains("commit_tool") && !text.contains("reasoning"),
        "{text}"
    );
    assert_eq!(saved.len(), 1, "only the user input remains: {saved:?}");
}
