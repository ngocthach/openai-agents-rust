//! Port of the `args` row of `test_variadic_field_constraints_validate_before_invocation` in the
//! Python SDK's `tests/test_function_tool.py`.
//!
//! Python's `*scores: Annotated[int, Field(ge=0, le=10)]` becomes a `scores` list argument with
//! an empty default; in Rust that is a `Vec` of a range-checked integer with `#[serde(default)]`.
//! Every element is validated before the tool body runs, and an absent or empty list is the
//! empty call. The `kwargs` row (`**scores`) has no Rust counterpart: arguments are typed.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{AgentsError, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(try_from = "i64")]
#[schemars(with = "i64")]
struct Score(i64);

impl TryFrom<i64> for Score {
    type Error = String;
    fn try_from(value: i64) -> std::result::Result<Self, String> {
        if (0..=10).contains(&value) {
            Ok(Score(value))
        } else {
            Err(format!("{value} is outside 0..=10"))
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct PositionalArgs {
    #[serde(default)]
    scores: Vec<Score>,
}

async fn invoke(tool: &FunctionTool<()>, payload: serde_json::Value) -> Result<String> {
    let args = payload.to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "1", &tool.name, &args);
    tool.invoke(ctx, args)
        .await
        .map(|out| out.to_model_string())
}

#[tokio::test]
async fn variadic_field_constraints_validate_before_invocation() {
    let calls = Arc::new(Mutex::new(Vec::<i64>::new()));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new(
        "positional",
        "Sum scores.",
        move |_ctx, args: PositionalArgs| {
            let seen = seen.clone();
            async move {
                let total: i64 = args.scores.iter().map(|s| s.0).sum();
                seen.lock().unwrap().push(total);
                Ok(total.to_string())
            }
        },
    )
    .without_failure_error_function();

    for invalid in [-1, 11] {
        let error = invoke(&tool, json!({"scores": [invalid]}))
            .await
            .unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(calls.lock().unwrap().is_empty());
    }

    assert_eq!(
        invoke(&tool, json!({"scores": [0, 10]})).await.unwrap(),
        "10"
    );
    assert_eq!(invoke(&tool, json!({})).await.unwrap(), "0");
    assert_eq!(invoke(&tool, json!({"scores": []})).await.unwrap(), "0");
    assert_eq!(*calls.lock().unwrap(), vec![10, 0, 0]);
}
