//! Ports of cases of `tests/test_result_cast.py`: a final output asked for as the wrong type is
//! an error rather than a silently reinterpreted value, and a run that is not an agent-as-tool
//! invocation carries no agent-tool metadata.

use std::sync::Arc;

use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, Runner};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct Foo {
    bar: i64,
}

async fn text_result(text: &str) -> openai_agents_sdk::result::RunResult<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(text)])])
            .unwrap(),
    );
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model));
    Runner::run(agent, "hi").await.expect("the run completes")
}

/// Python: `test_bad_cast_with_param_raises`. Rust's cast always checks the type (Python's
/// default of returning the value unchecked has no typed equivalent: read `final_output`).
#[tokio::test]
async fn a_final_output_of_the_wrong_type_is_an_error() {
    let result = text_result("test").await;

    assert_eq!(result.final_output_as::<String>().unwrap(), "test");
    assert!(result.final_output_as::<Foo>().is_err());
    assert!(result.final_output_as::<i64>().is_err());
    // The untyped value is still there, so a bad cast loses nothing.
    assert_eq!(result.final_output_text().as_deref(), Some("test"));
}

/// Python: `test_run_result_agent_tool_invocation_returns_none_for_plain_context`.
#[tokio::test]
async fn a_plain_run_has_no_agent_tool_invocation() {
    let result = text_result("ok").await;

    assert!(result.agent_tool_invocation.is_none());
}
