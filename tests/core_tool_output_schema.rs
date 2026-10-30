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
