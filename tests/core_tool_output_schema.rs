//! `FunctionTool::output_json_schema`: validation and the Responses wire param.

use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

fn tool() -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args("report", "Report", |_c| async { Ok("ok") })
}

#[test]
fn a_declared_output_schema_is_made_strict_and_sent_as_output_schema() {
    let tool = tool()
        .with_output_json_schema(json!({
            "type": "object",
            "properties": {"total": {"type": "number"}},
        }))
        .unwrap();
    let schema = tool.output_json_schema.clone().unwrap();
    assert_eq!(schema["additionalProperties"], json!(false));
    assert_eq!(schema["required"], json!(["total"]));

    let param = tool.to_spec().to_responses_param();
    assert_eq!(param["output_schema"], schema);
    // Chat Completions has no such field.
    assert!(tool.to_spec().to_chat_param()["function"]["output_schema"].is_null());
}

#[test]
fn a_tool_without_a_declared_schema_sends_nothing() {
    assert!(tool().to_spec().to_responses_param()["output_schema"].is_null());
}

#[test]
fn a_non_object_schema_is_rejected() {
    let err = tool()
        .with_output_json_schema(json!({"type": "string"}))
        .unwrap_err()
        .to_string();
    assert!(err.contains("JSON object schema"), "{err}");
}

#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct Report {
    total: f64,
    label: String,
}

#[test]
fn with_output_type_derives_the_strict_schema_from_the_type() {
    let tool = FunctionTool::<()>::new_no_args("report", "Report", |_c| async {
        openai_agents_sdk::tool::ToolOutput::json(&Report {
            total: 3.0,
            label: "q1".into(),
        })
    })
    .with_output_type::<Report>()
    .unwrap();

    let schema = tool.output_json_schema.clone().unwrap();
    assert_eq!(schema["type"], json!("object"));
    assert_eq!(schema["additionalProperties"], json!(false));
    assert_eq!(schema["properties"]["total"]["type"], json!("number"));
    assert_eq!(schema["properties"]["label"]["type"], json!("string"));
    let mut required = schema["required"].as_array().unwrap().clone();
    required.sort_by_key(|v| v.as_str().unwrap_or_default().to_string());
    assert_eq!(required, vec![json!("label"), json!("total")]);
    assert!(schema.get("$schema").is_none());
    assert_eq!(tool.to_spec().to_responses_param()["output_schema"], schema);
}

#[test]
fn with_output_type_rejects_a_non_object_type() {
    let err = FunctionTool::<()>::new_no_args("report", "Report", |_c| async { Ok("ok") })
        .with_output_type::<String>()
        .unwrap_err()
        .to_string();
    assert!(err.contains("JSON object schema"), "{err}");
}

#[tokio::test]
async fn with_output_type_validates_the_tool_output() {
    use openai_agents_sdk::RunContext;
    use openai_agents_sdk::tool_context::ToolContext;

    let ctx = || ToolContext::direct(RunContext::new(()), "A", "report", "{}");

    let good = FunctionTool::<()>::new_no_args("report", "Report", |_c| async {
        openai_agents_sdk::tool::ToolOutput::json(&Report {
            total: 1.5,
            label: "ok".into(),
        })
    })
    .with_output_type::<Report>()
    .unwrap();
    let out = good.invoke(ctx(), String::new()).await.unwrap();
    assert_eq!(out.to_json_value()["label"], json!("ok"));

    let bad = FunctionTool::<()>::new_no_args("report", "Report", |_c| async {
        Ok(json!({"total": "not a number"}))
    })
    .with_output_type::<Report>()
    .unwrap();
    let err = bad
        .invoke(ctx(), String::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match its declared output type"),
        "{err}"
    );
}

#[tokio::test]
async fn a_declared_output_type_failure_fails_the_run() {
    use openai_agents_sdk::testing::{FakeModel, function_call};
    use openai_agents_sdk::{Agent, Runner};
    use std::sync::Arc;

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("report", "{}")]);

    // The tool returns something its declared output type does not accept. Python's
    // `_validate_function_tool_output` raises a `UserError`, and `AgentsException` is
    // re-raised rather than turned into a model-visible message, so the run fails.
    let tool = FunctionTool::<()>::new_no_args("report", "Report", |_c| async {
        Ok(json!({"total": "not a number"}))
    })
    .with_output_type::<Report>()
    .unwrap();
    let agent = Agent::<()>::new("A")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "go").await.unwrap_err().to_string();
    assert!(
        err.contains("does not match its declared output type"),
        "{err}"
    );
}
