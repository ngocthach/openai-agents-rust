//! Port of the `ordinary` rows of `test_annotated_constraints_reject_arguments_before_tool_execution`
//! in `tests/test_function_schema.py`.
//!
//! An argument that violates its constraint (`PositiveInt`, or `Field(ge=0, le=10)`) is rejected
//! with "Invalid JSON input for tool" before the tool body runs; with `failure_error_function`
//! unset the run fails with a `ModelBehaviorError`, and a valid argument runs the tool once.
//! The variadic rows (`*quantity`, `**quantity`) have no Rust counterpart.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(try_from = "i64")]
#[schemars(with = "i64")]
struct Positive(i64);

impl TryFrom<i64> for Positive {
    type Error = String;
    fn try_from(value: i64) -> Result<Self, String> {
        if value > 0 {
            Ok(Positive(value))
        } else {
            Err(format!("{value} is not positive"))
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(try_from = "i64")]
#[schemars(with = "i64")]
struct Bounded(i64);

impl TryFrom<i64> for Bounded {
    type Error = String;
    fn try_from(value: i64) -> Result<Self, String> {
        if (0..=10).contains(&value) {
            Ok(Bounded(value))
        } else {
            Err(format!("{value} is outside 0..=10"))
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct PositiveArgs {
    quantity: Positive,
}

#[derive(Deserialize, JsonSchema)]
struct BoundedArgs {
    quantity: Bounded,
}

async fn assert_rejected_before_execution(
    agent_for: impl Fn(Arc<FakeModel>, Arc<Mutex<Vec<i64>>>) -> Agent<()>,
    invalid: &str,
    valid: &str,
) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let invocations = Arc::new(Mutex::new(Vec::<i64>::new()));

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "update_inventory",
        invalid,
        "call-1",
    )]);
    let agent = agent_for(model, invocations.clone());
    let error = Runner::run(&agent, "Update inventory").await.unwrap_err();
    assert!(
        matches!(&error, AgentsError::ModelBehaviorError { .. })
            && error.to_string().contains("Invalid JSON input for tool"),
        "{error:?}"
    );
    assert!(invocations.lock().unwrap().is_empty());

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "update_inventory",
        valid,
        "call-1",
    )]);
    let agent = agent_for(model, invocations.clone());
    let result = Runner::run(&agent, "Update inventory").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("5"));
    assert_eq!(*invocations.lock().unwrap(), vec![5]);
}

#[tokio::test]
async fn a_non_positive_argument_is_rejected_before_the_tool_runs() {
    for invalid in [-1, 0] {
        assert_rejected_before_execution(
            |model, seen| {
                let tool = FunctionTool::<()>::new(
                    "update_inventory",
                    "Update inventory.",
                    move |_ctx, args: PositiveArgs| {
                        let seen = seen.clone();
                        async move {
                            seen.lock().unwrap().push(args.quantity.0);
                            Ok(args.quantity.0.to_string())
                        }
                    },
                )
                .without_failure_error_function();
                Agent::<()>::new("inventory")
                    .with_tool(tool)
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                    .with_model_instance(model)
            },
            &format!(r#"{{"quantity": {invalid}}}"#),
            r#"{"quantity": 5}"#,
        )
        .await;
    }
}

#[tokio::test]
async fn an_out_of_range_argument_is_rejected_before_the_tool_runs() {
    for invalid in [-1, 20] {
        assert_rejected_before_execution(
            |model, seen| {
                let tool = FunctionTool::<()>::new(
                    "update_inventory",
                    "Update inventory.",
                    move |_ctx, args: BoundedArgs| {
                        let seen = seen.clone();
                        async move {
                            seen.lock().unwrap().push(args.quantity.0);
                            Ok(args.quantity.0.to_string())
                        }
                    },
                )
                .without_failure_error_function();
                Agent::<()>::new("inventory")
                    .with_tool(tool)
                    .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
                    .with_model_instance(model)
            },
            &format!(r#"{{"quantity": {invalid}}}"#),
            r#"{"quantity": 5}"#,
        )
        .await;
    }
}
